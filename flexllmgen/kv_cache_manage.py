from itertools import count
from typing import Dict, List

import numpy as np
import torch
from flexllmgen.utils import ValueHolder
from flexllmgen.pytorch_backend import DeviceType, TorchTensor,TorchDevice,TorchDisk, general_copy, map_to_torch_tensor, sync_general_copy, timers
from flexllmgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)
from flexllmgen.metadata_manage import CachePointer
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChunkPool:
    chunk_id = count()

    def __init__(self, env, config, batch_size:int = 1, chunk_size:int = 4, gpu_heap_size:int = 0, cpu_heap_size:int = 0, ):
        '''
        目前只用了self.pool:Dict[int, Chunk]={},通过chunk_id得到Chunk. 所有Chunk都在磁盘上
        chunk_size表明存了多少个token的一层的KV缓存
        '''
        self.chunk_size = chunk_size

        self.n_head = config.n_head
        self.head_dim = config.input_dim // self.n_head
        self.n_kv_head = config.num_key_value_heads
        self.chunk_shape = (chunk_size, batch_size * self.n_kv_head, self.head_dim)
        self.env = env
        self.gpu = env.gpu
        self.cpu = env.cpu
        self.disk = env.disk
        # self.gpu_heap_size = gpu_heap_size
        # self.cpu_heap_size = cpu_heap_size

        # self.cpu_chunk = List[Chunk]
        # self.gpu_chunk = List[Chunk]
        self.pool:Dict[int, Chunk]={}

        self.current_id = None
        self.current_offset = None

        self.cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
        self.cpu_buf_k = torch.empty((8192, batch_size * self.n_kv_head, self.head_dim), dtype=torch.float16, pin_memory=True)
        self.cpu_buf_v = torch.empty((8192, batch_size * self.n_kv_head, self.head_dim), dtype=torch.float16, pin_memory=True) 


    @classmethod
    def next_chunk_id(cls):
        return next(cls.chunk_id)

    def init_new_chunk(self, device, name = None):
        chunk_id = ChunkPool.next_chunk_id()
        chunk = Chunk(device=device, cache_shape=self.chunk_shape, chunk_id=chunk_id)
        self.pool[chunk_id] = chunk
        return chunk_id

    def store_prefix_cache(self, k_cache, v_cache, cache_offset):
        '''
        generate完后, 把新计算得到的KV存储到磁盘中
        self.current_id  self.current_offset指向当前可存储的空余位置。
        如果self.current_offset == chunk_size,就会新创建一个Chunk并令current_id=new_chunk.id  current_offset = 0
        '''
        if self.current_id == None:
            self.current_id = self.init_new_chunk(device=self.disk)
            self.current_offset = 0

        tgt_chunk = self.pool[self.current_id]

        full_head_k = tgt_chunk.full_head_k
        full_head_v = tgt_chunk.full_head_v

        full_head_src_indices = (
            slice(cache_offset, cache_offset + 1),
            slice(0, full_head_k.shape[1]),
            slice(0, full_head_k.shape[2])
        )
        full_head_dst_indices = (
            slice(self.current_offset, self.current_offset + 1),
            slice(0, full_head_k.shape[1]),
            slice(0, full_head_k.shape[2])
        )
        sync_general_copy(
            dst=full_head_k,
            dst_indices=full_head_dst_indices,
            src=k_cache,
            src_indices=full_head_src_indices,
            cpu_buf=self.cpu_buf
        )
        sync_general_copy(
            dst=full_head_v,
            dst_indices=full_head_dst_indices,
            src=v_cache,
            src_indices=full_head_src_indices,
            cpu_buf=self.cpu_buf
        )
        ptr = CachePointer(self.current_id, self.current_offset)
        self.current_offset += 1
        if self.current_offset == self.chunk_size:
            self.current_id = self.init_new_chunk(device=self.disk)
            self.current_offset = 0
        return ptr
        
    
    def get_full_head_cache(self, k_cache, v_cache, cache_offset, chunk_id:int, token_offset:int):
        '''
        和get_probe_cache是类似的方法。
        为了同步且不干扰别的流, 新创建了一个CUDA Stream
        '''
        src_chunk = self.pool[chunk_id]

        src_chunk.access_count += 1
        
        full_head_k = src_chunk.full_head_k
        full_head_v = src_chunk.full_head_v

        src_indices = (
            slice(token_offset, token_offset + 1), 
            slice(0, full_head_k.shape[1]),            
            slice(0, full_head_k.shape[2])             
        )

        dst_indices = (
            slice(cache_offset, cache_offset + 1),  
            slice(0, k_cache.shape[1]),                 
            slice(0, k_cache.shape[2])                  
        )
        sync_general_copy(
            dst=k_cache,
            dst_indices=dst_indices,
            src=full_head_k,
            src_indices=src_indices,
            cpu_buf=self.cpu_buf
        )

        sync_general_copy(
            dst=v_cache,
            dst_indices=dst_indices,
            src=full_head_v,
            src_indices=src_indices,
            cpu_buf=self.cpu_buf
        )

    def get_full_head_cache_concurrent_once(self, k_cache, v_cache, req):
        loaded_chunks = {}

        def _load_chunk_data(chunk_id):
            src_chunk = self.pool[chunk_id]
            full_head_k = src_chunk.full_head_k
            full_head_v = src_chunk.full_head_v

            if full_head_k.device.device_type == DeviceType.DISK:
                # k_data = torch.from_numpy(np.load(full_head_k.data)).pin_memory()
                # v_data = torch.from_numpy(np.load(full_head_v.data)).pin_memory()
                k_data = torch.from_numpy(np.load(full_head_k.data))
                v_data = torch.from_numpy(np.load(full_head_v.data))
            else:
                k_data = map_to_torch_tensor(full_head_k, None)
                v_data = map_to_torch_tensor(full_head_v, None)
            
            return chunk_id, k_data, v_data
        
        timers("chunk io").start()
        
        with ThreadPoolExecutor() as executor:
            future_to_chunk_id = {executor.submit(_load_chunk_data, chunk_id): chunk_id for chunk_id in req.keys()}
            
            for future in as_completed(future_to_chunk_id):
                chunk_id, k_data, v_data = future.result()
                loaded_chunks[chunk_id] = (k_data, v_data)

        timers("chunk io").stop()
        last_io_time = timers("chunk io").costs[-1]
        num_chunks = len(req)
        bytes_per_chunk = self.chunk_size * self.n_head * self.head_dim * 2 * 2
        total_bytes_read = num_chunks * bytes_per_chunk

        if last_io_time > 0:
            bandwidth_bps = total_bytes_read / last_io_time
            bandwidth_mbps = bandwidth_bps / (1024 * 1024)
        else:
            bandwidth_mbps = float('inf')

        # print(f"Chunk IO Time: {last_io_time:.4f} s, Chunks Read: {num_chunks}, Calculated Bandwidth: {bandwidth_mbps:.2f} MB/s")


        timers("chunk io").start()

        for chunk_id, store_ptr in req.items():
                k_data, v_data = loaded_chunks[chunk_id]
                if not store_ptr:
                    continue
                src_offsets = [item[0] for item in store_ptr]
                dst_offsets = [item[1] for item in store_ptr]

                self.cpu_buf_k[dst_offsets] = k_data[src_offsets]
                self.cpu_buf_v[dst_offsets] = v_data[src_offsets]
                
        timers("chunk io").stop()
        last_io_time = timers("chunk io").costs[-1]
        # logger.info(f"chunk iteration time:{last_io_time}")

        k_len = k_cache.shape[0]
        k_cache.data.copy_(self.cpu_buf_k[:k_len], non_blocking=True)
        v_cache.data.copy_(self.cpu_buf_v[:k_len], non_blocking=True)

class Chunk:
    chunk_id = count()
    def __init__(self, device, cache_shape, chunk_id=None):
        self.access_count = 0
        self.device = device
        self.chunk_id = chunk_id or Chunk.next_chunk_name()
        # shape = (chunk_size, batch * n_heads, head_dim)
        self.full_head_k = device.allocate(shape=cache_shape, dtype=np.float16, pin_memory=True)
        self.full_head_v = device.allocate(shape=cache_shape, dtype=np.float16, pin_memory=True)
    
    @classmethod
    def next_chunk_name(cls):
        return next(cls.chunk_id)
    
    
class ProbeChunk:
    def __init__(self, device, cache_shape,  chunk_id = None):
        self.device = device
        self.probe_k = device.allocate(shape=cache_shape, dtype=np.float16, pin_memory=True)

    def delete(self):
        self.probe_k.delete()

class ProbeChunkPool:
    probe_chunk_id = count()

    def __init__(self, env, config, batch_size=1, chunk_size=1024):
        self.chunk_size = chunk_size

        n_head = config.n_head
        n_probe_head = 3
        self.head_dim = config.input_dim // config.n_head
        self.chunk_shape = (chunk_size, batch_size * n_probe_head, self.head_dim)

        self.env = env
        self.gpu = env.gpu
        self.cpu = env.cpu
        self.disk = env.disk

        self.chunk_table:Dict[int, ProbeChunk] = {}

        self.current_id = None
        self.current_offset = None

        self.cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)

        self.cpu_buf_k = torch.empty((8192, batch_size * 3 ,self.head_dim), dtype=torch.float16, pin_memory=True)

    @classmethod
    def next_chunk_id(cls):
        return next(cls.probe_chunk_id)
    
    def init_new_chunk(self):
        chunk_id = ProbeChunkPool.next_chunk_id()    
        probe_chunk = ProbeChunk(device=self.disk, cache_shape=self.chunk_shape, chunk_id=chunk_id)
        self.chunk_table[chunk_id] = probe_chunk
        return chunk_id
    
    def switch_to_new_chunk(self):
        new_chunk_id = self.init_new_chunk()
        self.current_id = new_chunk_id
        self.current_offset = 0
        return self.current_id
    
    def store_probe_k(self, k_cache, cache_offset):
        if self.current_id == None:
            self.current_id = self.init_new_chunk()
            self.current_offset = 0

        tgt_chunk = self.chunk_table[self.current_id]

        probe_k = tgt_chunk.probe_k


        src_indices = (
            slice(cache_offset, cache_offset + 1),
            slice(0, probe_k.shape[1]),
            slice(0, probe_k.shape[2])
        )
        
        dst_indices = (
            slice(self.current_offset, self.current_offset + 1),
            slice(0, probe_k.shape[1]),
            slice(0, probe_k.shape[2])
        )

        sync_general_copy(
            dst=probe_k,
            dst_indices=dst_indices,
            src=k_cache,
            src_indices=src_indices,
            cpu_buf=self.cpu_buf
        )

        ptr = CachePointer(self.current_id, self.current_offset)
        self.current_offset += 1

        if self.current_offset == self.chunk_size:
            self.current_id = self.init_new_chunk()
            self.current_offset = 0

        return ptr


    def get_probe_cache_concurrent_merge(self, k_cache, req):
        loaded_chunks = {}
        
        def _load_chunk_data(chunk_id):
            src_chunk = self.chunk_table[chunk_id]
            probe_k = src_chunk.probe_k

            if probe_k.device.device_type == DeviceType.DISK:
                k_data = torch.from_numpy(np.load(probe_k.data))
            else:
                k_data = map_to_torch_tensor(probe_k, None)
            
            return chunk_id, k_data
        
        timers("chunk io").start()
        for chunk_id in req.keys():
            _, loaded_chunks[chunk_id] = _load_chunk_data(chunk_id)
        
        timers("chunk io").stop()
        last_io_time = timers("chunk io").costs[-1]
        num_chunks = len(req)
        bytes_per_chunk = self.chunk_size * 3 * self.head_dim * 2
        total_bytes_read = num_chunks * bytes_per_chunk

        if last_io_time > 0:
            bandwidth_bps = total_bytes_read / last_io_time
            bandwidth_mbps = bandwidth_bps / (1024 * 1024)
        else:
            bandwidth_mbps = float('inf')
        # print(f"Probe Chunk IO Time: {last_io_time:.4f} s, Chunks Read: {num_chunks}, Calculated Bandwidth: {bandwidth_mbps:.2f} MB/s")

        timers("chunk io").reset()
        
        num_merged = 0
        for chunk_id, store_ptr in req.items():
            k_data = loaded_chunks[chunk_id]
            
            if not store_ptr:
                continue
            
            merged_ranges = []
            
            first_src, first_dst = store_ptr[0]
            current_src_start = first_src
            current_dst_start = first_dst
            current_length = 1
            
            for i in range(1, len(store_ptr)):
                src_offset, dst_offset = store_ptr[i]
                prev_src, prev_dst = store_ptr[i - 1]
                
                # 检查是否连续：src 和 dst 都连续递增
                if src_offset == prev_src + 1 and dst_offset == prev_dst + 1:
                    # 连续，扩展当前范围
                    current_length += 1
                else:
                    # 不连续，保存当前范围，开始新范围
                    merged_ranges.append((current_src_start, current_dst_start, current_length))
                    current_src_start = src_offset
                    current_dst_start = dst_offset
                    current_length = 1
            
            # 保存最后一个范围
            merged_ranges.append((current_src_start, current_dst_start, current_length))
            num_merged += len(merged_ranges)
            timers("chunk io").start()
            # 执行合并后的复制
            for src_start, dst_start, length in merged_ranges:
                self.cpu_buf_k[dst_start:dst_start + length, :, :].copy_(
                    k_data[src_start:src_start + length, :, :]
                )
            timers("chunk io").stop()
        # 一次性拷贝到目标缓存
        k_len = k_cache.shape[0]
        k_cache.data.copy_(self.cpu_buf_k[:k_len], non_blocking=True)
        
        last_io_time = timers("chunk io").elapsed("sum")

        logger.info(f"probe chunk transfer time:{last_io_time}, merged copies: {num_merged}")