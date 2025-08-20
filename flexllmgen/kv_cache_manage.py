from itertools import count
from typing import Dict, List

import numpy as np
import torch
from flexllmgen.utils import ValueHolder
from flexllmgen.pytorch_backend import TorchTensor,TorchDevice,TorchDisk, general_copy, sync_general_copy
from flexllmgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)
from flexllmgen.metadata_manage import CachePointer

import logging

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChunkPool:
    def __init__(self, env, config, batch_size = 1, chunk_size:int = 4, gpu_heap_size:int = 0, cpu_heap_size:int = 0, ):
        '''
        目前只用了self.pool:Dict[int, Chunk]={},通过chunk_id得到Chunk. 所有Chunk都在磁盘上
        chunk_size表明存了多少个token的一层的KV缓存
        '''
        self.chunk_size = chunk_size

        n_head = config.n_head
        n_probe_head = 3
        head_dim = config.input_dim // n_head
        self.chunk_shape = (chunk_size, batch_size * n_head, head_dim)

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

        self.sync_copy_stream = torch.cuda.Stream()
        self.cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    
    def switch_to_new_chunk(self, name=None):
        new_chunk = Chunk(self.disk, self.chunk_shape, chunk_id=name)
        
        self.pool[new_chunk.chunk_id] = new_chunk

        self.current_id = new_chunk.chunk_id
        self.current_offset = 0

        return self.current_id


    def init_new_chunk(self, device, name = None):
        '''
        创建一个新Chunk, 返回新Chunk的id
        '''
        #chunk_probe_shape = (probe_shape[0] * self.chunk_size, *probe_shape[1:])
        new_chunk = Chunk(device, self.chunk_shape, chunk_id=name)
        #logger.info(f"Init new chunk{new_chunk.chunk_id} with full head shape {full_head_shape} on device {device}")
        self.pool[new_chunk.chunk_id] = new_chunk
        return new_chunk.chunk_id

    def store_kv_cache(self, k_cache, v_cache, cache_offset):
        '''
        generate完后, 把新计算得到的KV存储到磁盘中
        self.current_id  self.current_offset指向当前可存储的空余位置。
        如果self.current_offset == chunk_size,就会新创建一个Chunk并令current_id=new_chunk.id  current_offset = 0
        '''

        if isinstance(k_cache, int):
            src_chunk = self.pool[k_cache]

            k_cache = src_chunk.full_head_k
            v_cache = src_chunk.full_head_v


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
        general_copy(
            dst=full_head_k,
            dst_indices=full_head_dst_indices,
            src=k_cache,
            src_indices=full_head_src_indices
        )
        general_copy(
            dst=full_head_v,
            dst_indices=full_head_dst_indices,
            src=v_cache,
            src_indices=full_head_src_indices
        )
        ptr = CachePointer(self.current_id, self.current_offset)
        self.current_offset += 1
        if self.current_offset == self.chunk_size:
            self.current_id = self.init_new_chunk(device=self.disk)
            self.current_offset = 0
        return ptr
        

    def get_probe_cache(self, k_cache, cache_offset:int, chunk_id:int, offset:int):
        '''
        把probe_cache复制到k_cache中
        k_cache的shape是(common_prefix_len, b * n_probe_head, head_dim)
        cache_offset的作用是标识现在已经复制了几个token，从而得到要复制到k_cache中的(cache_offset:cache_offset + 1,:,:)
        从chunk内完整的k缓存(chunk_size, b * n_head, head_dim)中切出需要的部分
        '''
        src_chunk = self.pool[chunk_id]

        src_chunk.access_count += 1
        
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
        
        general_copy(
            dst=k_cache,
            dst_indices=dst_indices,
            src=src_cache,
            src_indices=src_indices
        )
    
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
        with torch.cuda.stream(self.sync_copy_stream):
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
        self.sync_copy_stream.synchronize()

    def get_full_head_cache_impress(self, k_cache, v_cache, dst_offset, chunk_id, src_offset):
        '''
        1. 更新IR access_count score
        2. 获取tgt_chunk
        3. 根据dst src offset 进行复制
        4. 判断是否保存在GPU。。
        '''
        pass

class Chunk:
    chunk_id = count()
    def __init__(self, device, full_head_shape, chunk_id=None):
        self.access_count = 0
        self.important_ratio = 0
        self.score = 0
        self.device = device
        self.chunk_id = chunk_id or Chunk.next_chunk_name()
        # shape = (1, batch*n_probe_head, head_dim)
        # shape = (chunk_size, batch * n_heads, head_dim)
        self.full_head_k = device.allocate(shape=full_head_shape, dtype=np.float16, pin_memory=True)
        self.full_head_v = device.allocate(shape=full_head_shape, dtype=np.float16, pin_memory=True)
    
    @classmethod
    def next_chunk_name(cls):
        return next(cls.chunk_id)
    
    def delete(self):
        self.full_head_v.delete()
        self.full_head_k.delete()




