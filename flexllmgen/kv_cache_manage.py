import dataclasses
import io
from itertools import count
import queue
import threading
from typing import Dict, List, Tuple

import numpy as np
import torch
from flexllmgen.utils import ValueHolder
from flexllmgen.pytorch_backend import TorchTensor,TorchDevice,TorchDisk, general_copy, sync_general_copy, DeviceType, map_to_torch_tensor, sync_general_copy_with_direct_io, sync_general_read 
from flexllmgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)
from flexllmgen.timer import timers
from flexllmgen.metadata_manage import CachePointer
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import asyncio
import aiofiles
from aiofiles import os as aio_os

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Chunk:
    def __init__(self, device, cache_shape, chunk_id=None):
        self.device = device
        self.full_head_k = device.allocate(shape=cache_shape, dtype=np.float16, pin_memory=True)
        self.full_head_v = device.allocate(shape=cache_shape, dtype=np.float16, pin_memory=True)
    
    def delete(self):
        self.full_head_v.delete()
        self.full_head_k.delete()

class ChunkMeta:
    def __init__(self, chunk_size, chunk_id:int, disk_ref:Chunk):
        self.chunk_id = chunk_id
        self.chunk_size = chunk_size

        self.ir_avg = 0
        self.access_count = 0
        self.score = 0

        self.disk_ref:Chunk = disk_ref
        
    def update_on_access(self, hits:int):
        instant_ir = hits / self.chunk_size
        self.ir_avg = (self.ir_avg * self.access_count + instant_ir) / (self.access_count + 1)
        self.access_count += 1
        self.score += instant_ir
    
    def delete(self):
        self.disk_ref.delete()


class ChunkPool:
    chunk_id = count()

    def __init__(self, env, config, batch_size = 1, chunk_size:int = 4, gpu_cache_size:int = 512, cpu_cache_size:int = 512):
        '''
        目前只用了self.pool:Dict[int, Chunk]={},通过chunk_id得到Chunk. 所有Chunk都在磁盘上
        chunk_size表明存了多少个token的一层的KV缓存
        '''
        self.chunk_size = chunk_size

        self.n_head = config.n_head
        n_probe_head = 3
        self.head_dim = config.input_dim // self.n_head

        self.chunk_shape = (chunk_size, batch_size * self.n_head, self.head_dim)

        self.env = env
        self.gpu = env.gpu
        self.cpu = env.cpu
        self.disk = env.disk

        self.gpu_cache_size = gpu_cache_size
        self.cpu_cache_size = cpu_cache_size
        
        self.cpu_cache = ScoredCache(device=self.cpu, cache_shape=self.chunk_shape, capacity=cpu_cache_size)
        self.gpu_cache = ScoredCache(device=self.gpu, cache_shape=self.chunk_shape, capacity=gpu_cache_size)

        self.chunk_table:Dict[int, ChunkMeta] = {}

        self.current_id = None
        self.current_offset = None

        self.cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)

    @classmethod
    def next_chunk_id(cls):
        return next(cls.chunk_id)
    
    def init_new_chunk(self):
        chunk_id = ChunkPool.next_chunk_id()    

        disk_ref = Chunk(device=self.disk, cache_shape=self.chunk_shape, chunk_id=chunk_id)
        chunk_meta = ChunkMeta(chunk_size=self.chunk_size, chunk_id=chunk_id, disk_ref=disk_ref)

        self.chunk_table[chunk_id] = chunk_meta

        return chunk_id
    
    def delete_chunk(self, chunk_id):
        pass
    
    def get_chunk_data(self, chunk_id):
        if chunk_id not in self.chunk_table:
            return None
        
        chunk_data = (self.gpu_cache.get_cache(chunk_id=chunk_id) or
                    self.cpu_cache.get_cache(chunk_id=chunk_id) or
                    self.chunk_table[chunk_id].disk_ref)

        return chunk_data

    def update_score(self, chunk_id:int, hits:int):
        chunk_meta = self.chunk_table[chunk_id]

        chunk_meta.update_on_access(hits)

        score = chunk_meta.score

        chunk_data = self.get_chunk_data(chunk_id)
        
        if self.gpu_cache.insert_cache(chunk_id, score, chunk_data):
            return
        if self.cpu_cache.insert_cache(chunk_id, score, chunk_data):
            return

    def switch_to_new_chunk(self, name=None):
        new_chunk_id = self.init_new_chunk()

        self.current_id = new_chunk_id
        self.current_offset = 0

        return self.current_id
    
    def store_kv_cache(self, k_cache, v_cache, cache_offset):
        '''
        generate完后, 把新计算得到的KV存储到磁盘中
        self.current_id  self.current_offset指向当前可存储的空余位置。
        如果self.current_offset == chunk_size,就会新创建一个Chunk并令current_id=new_chunk.id  current_offset = 0
        '''
        if isinstance(k_cache, int):
            src_chunk = self.get_chunk_data(k_cache)

            k_cache = src_chunk.full_head_k
            v_cache = src_chunk.full_head_v


        if self.current_id == None:
            self.current_id = self.init_new_chunk()
            self.current_offset = 0

        tgt_chunk = self.chunk_table[self.current_id].disk_ref

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
            self.current_id = self.init_new_chunk()
            self.current_offset = 0

        return ptr     

    def get_probe_cache(self, k_cache, cache_offset:int, chunk_id:int, offset:int):
        '''
        把probe_cache复制到k_cache中
        k_cache的shape是(common_prefix_len, b * n_probe_head, head_dim)
        cache_offset的作用是标识现在已经复制了几个token，从而得到要复制到k_cache中的(cache_offset:cache_offset + 1,:,:)
        从chunk内完整的k缓存(chunk_size, b * n_head, head_dim)中切出需要的部分
        '''
        src_chunk = self.get_chunk_data(chunk_id)

        # logger.info(f"ChunkPool_get_probe: use chunk on device={src_chunk.device}")

        src_cache = src_chunk.full_head_k
        
        src_indices = (
            slice(offset, offset + 1), 
            slice(0, k_cache.shape[1]),            
            slice(0, k_cache.shape[2])             
        )

        dst_indices = (
            slice(cache_offset, cache_offset + 1),  
            slice(0, k_cache.shape[1]),                 
            slice(0, k_cache.shape[2])                  
        )
        
        sync_general_read(
            dst=k_cache,
            dst_indices=dst_indices,
            src=src_cache,
            src_indices=src_indices,
            cpu_buf=self.cpu_buf
        )
        # general_copy(
        #     dst=k_cache,
        #     dst_indices=dst_indices,
        #     src=src_cache,
        #     src_indices=src_indices,
        # )
    
    def get_full_head_cache(self, k_cache, v_cache, cache_offset, chunk_id:int, token_offset:int):
        '''
        和get_probe_cache是类似的方法。
        为了同步且不干扰别的流, 新创建了一个CUDA Stream
        '''
        #src_chunk = self.disk_pool[chunk_id]

        #src_chunk.access_count += 1

        src_chunk = self.get_chunk_data(chunk_id)
        
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
        sync_general_read(
            dst=k_cache,
            dst_indices=dst_indices,
            src=full_head_k,
            src_indices=src_indices,
            cpu_buf=self.cpu_buf
        )

        sync_general_read(
            dst=v_cache,
            dst_indices=dst_indices,
            src=full_head_v,
            src_indices=src_indices,
            cpu_buf=self.cpu_buf
        )

    def get_full_head_cache_impress(self, k_cache, v_cache, chunk_id, store_ptr):
        '''
        1. 更新IR access_count score
        2. 获取tgt_chunk
        3. 根据dst src offset 进行复制
        4. 判断是否保存在GPU。。
        '''
        src_chunk = self.get_chunk_data(chunk_id)
        # logger.info(f"ChunkPool_get_full: use chunk on device={src_chunk.device}")

        hits = len(store_ptr)
        full_head_k = src_chunk.full_head_k
        full_head_v = src_chunk.full_head_v
        for src_offset, dst_offset in store_ptr:
            src_indices = (
                slice(src_offset, src_offset + 1), 
                slice(0, full_head_k.shape[1]),            
                slice(0, full_head_k.shape[2])             
            )
            dst_indices = (
                slice(dst_offset, dst_offset + 1),  
                slice(0, k_cache.shape[1]),                 
                slice(0, k_cache.shape[2])                  
            )
            sync_general_read(
                dst=k_cache,
                dst_indices=dst_indices,
                src=full_head_k,
                src_indices=src_indices,
                cpu_buf=self.cpu_buf
            )

            sync_general_read(
                dst=v_cache,
                dst_indices=dst_indices,
                src=full_head_v,
                src_indices=src_indices,
                cpu_buf=self.cpu_buf
            )

        self.update_score(chunk_id, hits)
    
    def get_full_head_cache_load(self, k_cache, v_cache, chunk_id, store_ptr):
        src_chunk = self.get_chunk_data(chunk_id)
        hits = len(store_ptr)
        full_head_k = src_chunk.full_head_k
        full_head_v = src_chunk.full_head_v
        if(full_head_k.device.device_type == DeviceType.DISK):
            k_data = torch.from_numpy(np.load(full_head_k.data)).pin_memory()
            v_data = torch.from_numpy(np.load(full_head_v.data)).pin_memory()
        else:
            k_data = map_to_torch_tensor(full_head_k)
            v_data = map_to_torch_tensor(full_head_v)
        for src_offset, dst_offset in store_ptr:
            k_cache.data[dst_offset:dst_offset + 1, :, :].copy_(k_data[src_offset:src_offset + 1, :, :], non_blocking=True)
            v_cache.data[dst_offset:dst_offset + 1, :, :].copy_(v_data[src_offset:src_offset + 1, :, :], non_blocking=True)

        # 定义单个复制任务
        # def copy_slice(src_offset, dst_offset):
        #     k_cache.data[dst_offset:dst_offset + 1, :, :].copy_(k_data[src_offset:src_offset + 1, :, :], non_blocking=True)
        #     v_cache.data[dst_offset:dst_offset + 1, :, :].copy_(v_data[src_offset:src_offset + 1, :, :], non_blocking=True)

        # # 使用线程池并发执行复制任务
        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     futures = [executor.submit(copy_slice, src_offset, dst_offset) for src_offset, dst_offset in store_ptr]
        #     for future in as_completed(futures):
        #         future.result() # 等待完成并检查异常

        self.update_score(chunk_id, hits)

    def get_full_head_cache_concurrent(self, k_cache, v_cache, req):
        loaded_chunks = {}

        def _load_chunk_data(chunk_id):
            src_chunk = self.get_chunk_data(chunk_id)
            full_head_k = src_chunk.full_head_k
            full_head_v = src_chunk.full_head_v

            if full_head_k.device.device_type == DeviceType.DISK:
                k_data = torch.from_numpy(np.load(full_head_k.data)).pin_memory()
                v_data = torch.from_numpy(np.load(full_head_v.data)).pin_memory()
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
        # for chunk_id in req.keys():
        #     src_chunk = self.get_chunk_data(chunk_id)
        #     full_head_k = src_chunk.full_head_k
        #     full_head_v = src_chunk.full_head_v

        #     if full_head_k.device.device_type == DeviceType.DISK:
        #         k_data = torch.from_numpy(np.load(full_head_k.data))
        #         v_data = torch.from_numpy(np.load(full_head_v.data))
        #     else:
        #         k_data = map_to_torch_tensor(full_head_k, None)
        #         v_data = map_to_torch_tensor(full_head_v, None)
        #     loaded_chunks[chunk_id] = (k_data, v_data)
        timers("chunk io").stop()
        # 获取 timers("chunk io").costs 的最后一个值
        last_io_time = timers("chunk io").costs[-1]
        
        # 获取该层 req 的 chunk_id 个数
        num_chunks = len(req)

        # 根据公式计算带宽
        bytes_per_chunk = self.chunk_size * self.n_head * self.head_dim * 2 * 2
        total_bytes_read = num_chunks * bytes_per_chunk
        
        # 避免除以零错误
        if last_io_time > 0:
            # 带宽 (Bytes/s)
            bandwidth_bps = total_bytes_read / last_io_time
            # 转换为 MB/s
            bandwidth_mbps = bandwidth_bps / (1024 * 1024)
        else:
            bandwidth_mbps = float('inf')

        # 打印信息
        print(f"Chunk IO Time: {last_io_time:.4f} s, Chunks Read: {num_chunks}, Calculated Bandwidth: {bandwidth_mbps:.2f} MB/s")

        # def _copy_slice(k_data, v_data, src_offset, dst_offset):
        #     k_cache.data[dst_offset:dst_offset + 1, :, :].copy_(k_data[src_offset:src_offset + 1, :, :], non_blocking=True)
        #     v_cache.data[dst_offset:dst_offset + 1, :, :].copy_(v_data[src_offset:src_offset + 1, :, :], non_blocking=True)

        # with ThreadPoolExecutor() as executor:
        #     copy_futures = []
        #     for chunk_id, store_ptr in req.items():
        #         k_data, v_data = loaded_chunks[chunk_id]
                
        #         for src_offset, dst_offset in store_ptr:
        #             copy_futures.append(executor.submit(_copy_slice, k_data, v_data, src_offset, dst_offset))
                
        #         self.update_score(chunk_id, len(store_ptr))

        #     for future in as_completed(copy_futures):
        #         future.result() # 检查是否有异常


        for chunk_id, store_ptr in req.items():
            # 从预加载的字典中获取数据
            k_data, v_data = loaded_chunks[chunk_id]
            
            # 串行执行所有复制任务
            for src_offset, dst_offset in store_ptr:
                k_cache.data[dst_offset:dst_offset + 1, :, :].copy_(k_data[src_offset:src_offset + 1, :, :], non_blocking=True)
                v_cache.data[dst_offset:dst_offset + 1, :, :].copy_(v_data[src_offset:src_offset + 1, :, :], non_blocking=True)
            
            # 更新分数
            self.update_score(chunk_id, len(store_ptr))
    
    def get_full_head_cache_async_run(self, k_cache, v_cache, req):
        asyncio.run(self.get_full_head_cache_async(k_cache, v_cache, req))

    async def get_full_head_cache_async(self, k_cache, v_cache, req):
        loaded_chunks = {}
        
        async def _load_chunk_data_async(chunk_id):
            src_chunk = self.get_chunk_data(chunk_id)
            full_head_k = src_chunk.full_head_k
            full_head_v = src_chunk.full_head_v
            
            if full_head_k.device.device_type == DeviceType.DISK:
                # 异步读取文件
                async with aiofiles.open(full_head_k.data, 'rb') as f:
                    k_content = await f.read()
                    k_array = np.load(io.BytesIO(k_content))
                async with aiofiles.open(full_head_v.data, 'rb') as f:
                    v_content = await f.read()
                    v_array = np.load(io.BytesIO(v_content))
                    
                k_data = torch.from_numpy(k_array).pin_memory()
                v_data = torch.from_numpy(v_array).pin_memory()
            else:
                k_data = map_to_torch_tensor(full_head_k, None)
                v_data = map_to_torch_tensor(full_head_v, None)
            
            return chunk_id, k_data, v_data
        
        timers("chunk io").start()
        # 并行执行所有异步任务
        tasks = [_load_chunk_data_async(chunk_id) for chunk_id in req.keys()]
        results = await asyncio.gather(*tasks)
        
        for chunk_id, k_data, v_data in results:
            loaded_chunks[chunk_id] = (k_data, v_data)
        
        timers("chunk io").stop()

        last_io_time = timers("chunk io").costs[-1]

        num_chunks = len(req)
        
        # 根据公式计算带宽
        bytes_per_chunk = self.chunk_size * 7168 * 2 * 2 #chunk_size * hidden_dim * 2(Bytes/fp16) * 2(KV)
        total_bytes_read = num_chunks * bytes_per_chunk
        
        # 避免除以零错误
        if last_io_time > 0:
            # 带宽 (Bytes/s)
            bandwidth_bps = total_bytes_read / last_io_time
            # 转换为 MB/s
            bandwidth_mbps = bandwidth_bps / (1024 * 1024)
        else:
            bandwidth_mbps = float('inf')

        # 打印信息
        print(f"Chunk IO Time: {last_io_time:.4f} s, Chunks Read: {num_chunks}, Calculated Bandwidth: {bandwidth_mbps:.2f} MB/s")


        for chunk_id, store_ptr in req.items():
            # 从预加载的字典中获取数据
            k_data, v_data = loaded_chunks[chunk_id]
            
            # 串行执行所有复制任务
            for src_offset, dst_offset in store_ptr:
                k_cache.data[dst_offset:dst_offset + 1, :, :].copy_(k_data[src_offset:src_offset + 1, :, :], non_blocking=True)
                v_cache.data[dst_offset:dst_offset + 1, :, :].copy_(v_data[src_offset:src_offset + 1, :, :], non_blocking=True)
            
            # 更新分数
            self.update_score(chunk_id, len(store_ptr))

    def close_copy_threads(self):
        self.gpu_cache.close_copy_threads()
        self.cpu_cache.close_copy_threads()
    
    def sync(self):
        self.gpu_cache.synchronize()
        self.cpu_cache.synchronize()

@dataclasses.dataclass
class CacheEntry:
    state:int
    heap_idx:int

class ScoredCache:
    def __init__(self, device:TorchDevice, cache_shape, capacity:int=512):
        self.device = device
        self.capacity = capacity
        self.cache_shape = cache_shape

        self.cache_space = [Chunk(device, cache_shape) for i in range(capacity)]

        self._lock = threading.Lock()

        self.index_heap = IndexMinHeap(capacity) # (score, cache_idx)


        # Copy threads
        cuda_id = 7
        num_copy_threads = 4

        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=self.chunk_copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def chunk_copy_worker_func(self, queue, cuda_id):
        cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
        copy_stream = torch.cuda.Stream()
        with torch.cuda.stream(copy_stream):
            while True:
                item = queue.get()
                if item is None:
                    queue.task_done()
                    return

                src_chunk, src_chunk_id, dst_cache_idx = item
                dst_chunk = self.cache_space[dst_cache_idx]

                #logger.info(f"ChunkCache Copy: src_chunk_id={src_chunk_id}, dst_cache_idx={dst_cache_idx}")

                src = src_chunk.full_head_k
                dst = dst_chunk.full_head_k
                sync_general_copy(dst, None, src, None, cpu_buf)

                src = src_chunk.full_head_v
                dst = dst_chunk.full_head_v
                sync_general_copy(dst, None, src, None, cpu_buf)
                
                #logger.info(f"Chunk_Copy: finish copy src_chunk_id={src_chunk_id}, dst_cache_idx={dst_cache_idx}")
                with self._lock:
                    if src_chunk_id in self.index_heap._pos:
                        self.index_heap.update_cache_state(chunk_id=src_chunk_id, tgt_status=1)
                    
                queue.task_done()

    def has_cache(self, chunk_id):
        with self._lock:
            return self.index_heap.has_cache(chunk_id)

    def get_cache(self, chunk_id):
        with self._lock:
            if self.index_heap.has_cache(chunk_id):
                entry = self.index_heap._pos[chunk_id]
                if entry.state == 1:
                    cache_idx = self.index_heap.get_cache_idx(entry.heap_idx)
                    return self.cache_space[cache_idx]
        return None            

    def insert_cache(self, chunk_id, score, chunk_data):
        if self.capacity == 0:
            return False
        with self._lock:
            if self.index_heap.has_cache(chunk_id):
                self.index_heap.update_score(chunk_id, score)
                return True
            
            min_score, _, _ = self.index_heap.peek_min()
            if len(self.index_heap) >= self.capacity and score <= min_score:
                return False

            cache_idx = self.index_heap.push(score, chunk_id)
            #logger.info(f"ScoredCache submit_copy: chunk_type={self.device} push chunk={chunk_id}, cache_idx={cache_idx}, heap_idx={heap_idx}, pos_state={self._heap._pos[chunk_id]}")
            self.submit_copy(chunk_data, chunk_id, cache_idx)
            return True    
    
    def clear(self):
        self.index_heap.clear()

    def __del__(self):
        for x in self.cache_space:
            x.delete()

        if self.copy_queue:
            self.close_copy_threads()

class IndexMinHeap:
    def __init__(self, capacity:int=512):
        self._heap = []  # list of (score, cache_id, chunk_id)
        self._pos:Dict[int, CacheEntry] = {}  # chunk_id -> state, heap_idx
        self.capacity = capacity
    
    def has_cache(self, chunk_id):
        return chunk_id in self._pos
    
    def get_cache_idx(self, heap_idx):
        return self._heap[heap_idx][1]

    def update_cache_state(self, chunk_id, tgt_status):
        self._pos[chunk_id].status = tgt_status
    
    def _swap(self, i, j):
        ci, cj = self._heap[i][2], self._heap[j][2]

        self._pos[ci].heap_idx = j
        self._pos[cj].heap_idx = i

        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

    def _sift_up(self, idx):
        while idx > 0:
            parent = (idx - 1) // 2
            if self._heap[idx][0] < self._heap[parent][0]:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx):
        n = len(self._heap)
        while True:
            left = idx * 2 + 1
            right = left + 1
            smallest = idx
            if left < n and self._heap[left][0] < self._heap[smallest][0]:
                smallest = left
            if right < n and self._heap[right][0] < self._heap[smallest][0]:
                smallest = right
            if smallest == idx:
                break
            self._swap(idx, smallest)
            idx = smallest

    def push(self, score, chunk_id):
        if len(self._heap) >= self.capacity:
            _, min_cache_idx, min_chunk_id = self.pop_min()
            cache_idx = min_cache_idx
            self._pos.pop(min_chunk_id) 
            #logger.info(f"IndexMinHeap: remove chunk_id={min_chunk_id}")
        else:
            cache_idx = len(self._heap)

        idx = len(self._heap)
        self._pos[chunk_id] = CacheEntry(state=0, heap_idx=idx)
        #logger.info(f"Heap Push: Before heap_push: score={score}, chunk_id={chunk_id},state={self._pos[chunk_id]}, heap={self._heap}")
        self._heap.append((score, cache_idx, chunk_id))
        
        self._sift_up(idx)
        #logger.info(f"Heap Push: After score={score}, chunk_id={chunk_id},state={self._pos[chunk_id]}, heap={self._heap}")
        return cache_idx

    def pop_min(self):
        if not self._heap:
            return None, None, None
        min_score, min_cache_id, min_chunk_id = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._pos[last[2]].heap_idx = 0
            self._sift_down(0)
        return min_score, min_cache_id, min_chunk_id

    def update_score(self, chunk_id, new_score):
        """原地修改 chunk 的 score(自动上浮或下沉)。"""
        idx = self._pos[chunk_id].heap_idx
        if idx is None:
            return
        #logger.info(f"Heap Update Score: Before chunk_id={chunk_id}, new_score={new_score}, state={self._pos[chunk_id]}")
        old_score, cache_idx, chunk_id = self._heap[idx]
        self._heap[idx] = (new_score, cache_idx, chunk_id)
        if new_score < old_score:
            self._sift_up(idx)
        else:
            self._sift_down(idx)
        
        #logger.info(f"Heap Update Score: After chunk_id={chunk_id}, new_score={new_score}, state={self._pos[chunk_id]}")
    
    def peek_min(self):
        if not self._heap:
            return None, None, None
        min_score, min_cache_id, min_chunk_id = self._heap[0]
        return min_score, min_cache_id, min_chunk_id

    def __len__(self):
        return len(self._heap)

    def clear(self):
        self._heap.clear()
        self._pos.clear()
        

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
        head_dim = config.input_dim // config.n_head
        self.chunk_shape = (chunk_size, batch_size * n_probe_head, head_dim)

        self.env = env
        self.gpu = env.gpu
        self.cpu = env.cpu
        self.disk = env.disk

        self.chunk_table:Dict[int, ProbeChunk] = {}

        self.current_id = None
        self.current_offset = None

        self.cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
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
    
    def get_probe_cache(self, k_cache, chunk_id, store_ptr):
        src_chunk = self.chunk_table[chunk_id]

        probe_k = src_chunk.probe_k

        if probe_k.device.device_type == DeviceType.DISK:
            k_data = torch.from_numpy(np.load(probe_k.data)).pin_memory()
        else:
            k_data = map_to_torch_tensor(probe_k)
        
        for src_offset, dst_offset in store_ptr:
            k_cache.data[dst_offset:dst_offset + 1, :, :].copy_(k_data[src_offset:src_offset + 1, :, :], non_blocking=True)
    
