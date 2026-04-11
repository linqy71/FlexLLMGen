"""
Usage:
python3 -m flexllmgen.flex_llama --model meta/llama3.1-8b --gpu-batch-size 32 --percent 100 0 100 0 100 0 --path= ~/HDD_POOL/lqy/HF_HOME/hub --overlap=False
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexllmgen.compression import CompressionConfig
from flexllmgen.llama_config import LlamaConfig, get_llama_config_from



from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import, precompute_freqs_cis)
from flexllmgen.timer import timers
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

from flexllmgen.metadata_manage import RadixTree, RadixTreeNode, RadixToken, CachePointer
from flexllmgen.kv_cache_manage import ChunkPool, ProbeChunkPool
fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

from collections import defaultdict
from datasets import load_dataset
import logging

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/datasets/"
DEFAULT_TOKENIZER_PATH = None

LLAMA_MODEL_SPECS = {
    "meta/llama3.1-8b": {
        "config_name": "llama3.1-8b",
        "hf_cache_dir": "models--meta-llama--Meta-Llama-3.1-8B-Instruct",
    },
    "meta/llama3.3-70b": {
        "config_name": "llama3.3-70b",
        "hf_cache_dir": "models--meta-llama--Llama-3.3-70B-Instruct",
    },
}

@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    # the ratio of important tokens in prefix kv cache
    important_ratio: float = 0.25

    # Config of Chunk Pool
    chunk_size: int = 256
    probe_chunk_size: int = 3406
    #gpu_heap_size: int = 0
    #cpu_heap_size: int = 0

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def set_chunk_pool(self, chunk_pool):
        pass

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, j):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j, freqs_cis):
        # Compute input embedding
        donate = [False] * 2
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[1]) = weight_read_buf.pop()
        else:
            (w_token, _) = weight_read_buf.val

        h = self.compute.llama_input_embed(h, w_token, donate)
        hidden.val = h

# norm, lm_head
class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.rms_norm_eps = config.rms_norm_eps

        self.task = None

    def set_task(self, task):
        self.task = task

    def set_chunk_pool(self, chunk_pool):
        pass
    
    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # norm
            ((h,), dtype, path + "decoder.norm.weight"),
            # w_lm
            ((v, h), dtype, path + "lm_head.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        norm, w_lm = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((norm.smart_copy(dst2),
                                   w_lm.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, j):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j, freqs_cis):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (norm, donate[1]), (w_lm, donate[2]) = weight_read_buf.pop()
        else:
            (norm, _), (w_lm, _) = weight_read_buf.val

        h = self.compute.llama_output_embed(h, norm, w_lm, self.rms_norm_eps, donate,
            self.task.temperature)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.rms_norm_eps = config.rms_norm_eps
        
        self.cos_cache = None
        self.sin_cache = None
        
        self.task = None
        self.prefill_cache_shape = 0
        
    def set_task(self, task):
        self.task = task

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()
        
    def set_chunk_pool(self, chunk_pool):
        self.chunk_pool = chunk_pool

    def set_probe_chunk_pool(self, probe_chunk_pool):
        self.probe_chunk_pool = probe_chunk_pool

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        kv_hidden_size = (self.config.hidden_size // self.config.n_head) * self.config.num_key_value_heads
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}"))
        weight_specs = [
            # i_n
            ((h, ), dtype, path + ".input_layernorm.weight"),
            # w_q
            ((h, h), dtype, path + ".self_attn.q_proj.weight"),
            # w_k
            ((kv_hidden_size, h), dtype, path + ".self_attn.k_proj.weight"),
            # w_v
            ((kv_hidden_size, h), dtype, path + ".self_attn.v_proj.weight"),
            # w_out
            ((h, h), dtype, path + ".self_attn.o_proj.weight")
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        i_n, w_q, w_k, w_v, w_out = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                i_n.smart_copy(dst2),
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_out.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
        cache = device.init_cache_one_gpu_batch_llama(self.config, self.task, self.policy, max_prompt_len, max_gen_len)
        cache_home.store(cache)

    def load_probe_cache(self, cache_read_buf, i, j):
        if self.policy.compress_cache:
            dst = self.attention_compute.compressed_device
        else:
            dst = self.attention_compute
        # 首先确定shape,分配空间,然后从probe_chunk_pool中获取数据
        n_head = self.config.n_head
        n_probe_head = 3
        batch_size = self.policy.gpu_batch_size
        head_dim = self.config.input_dim // n_head

        probe_cache_shape = (self.task.common_prefix_len, batch_size * n_probe_head, head_dim)

        pin_memory = True if dst.device_type == DeviceType.CPU else False
        k_cache = dst.allocate(probe_cache_shape, np.float16, pin_memory=pin_memory)

        # 使用ProbeChunkPool并发获取probe cache
        # 构建请求字典: {chunk_id: [(src_offset, dst_offset), ...]}
        req = defaultdict(list)
        for t_idx in range(self.task.common_prefix_len):
            probe_k_ptr = self.task.common_prefix_token[t_idx].probe_ptr
            chunk_id, offset = probe_k_ptr[j].chunk_id, probe_k_ptr[j].offset
            req[chunk_id].append((offset, t_idx))

        timers("probe_cache").start(self.sync)
        self.probe_chunk_pool.get_probe_cache_concurrent_merge(k_cache, req)
        timers("probe_cache").stop(self.sync)
        cache_read_buf.store((k_cache, True))

    def get_prefix_kv(self, imp_token_idx, layer):
        n_head = self.config.n_head
        n_kv_head = self.config.num_key_value_heads
        batch_size = self.policy.gpu_batch_size
        head_dim = self.config.input_dim // n_head

        n_important = len(imp_token_idx) # 如果超过threshold长度就是重要token个数，如果没超过就是common_prefix_len
        dst = self.attention_compute
        shape = (n_important, batch_size * n_kv_head, head_dim)

        pin_memory = True if dst.device_type == DeviceType.CPU else False

        k_cache = dst.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = dst.allocate(shape, np.float16, pin_memory=pin_memory)

        req = defaultdict(list)
        for j in range(n_important):
            kv_ptr = self.task.common_prefix_token[imp_token_idx[j]].kv_ptr[layer]
            chunk_id, offset = kv_ptr.chunk_id, kv_ptr.offset
            req[chunk_id].append((offset, j))

        timers("full_cache").start(self.sync)
        self.chunk_pool.get_full_head_cache_concurrent_once(k_cache, v_cache, req)
        timers("full_cache").stop(self.sync)

        return k_cache, v_cache

    def load_cache(self, cache_home, cache_read_buf, i, j):
        if i == 0:  # prefill, no cache
            if self.task.common_prefix_len > 0:
                self.load_probe_cache(cache_read_buf, i, j)
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            # pos = self.task.prompt_len + i
            pos = self.prefill_cache_shape + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j, freqs_cis):
        n_head = self.config.n_head
        num_key_value_heads = self.config.num_key_value_heads

        donate = [False] * 10
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((i_n, donate[2]), (w_q, donate[3]), (w_k, donate[4]), (w_v, donate[5]), 
                    (w_out, donate[6])) = weight_read_buf.pop()
        else:
            ((i_n, _), (w_q, _), (w_k, _), (w_v, _), (w_out, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            if self.task.common_prefix_len > 0:
                (k_cache, donate[9]) = cache_read_buf.pop()
                timers("imp calc").start(self.sync)
                imp_token_idx = self.compute.get_important_token_idx_llama_modified(h, mask, i_n, w_q, self.rms_norm_eps,
                                freqs_cis, n_head, num_key_value_heads, donate, self.policy.compress_cache, self.policy.comp_cache_config,
                                k_cache, self.policy.important_ratio)
                timers("imp calc").stop(self.sync)

                imp_token_idx = imp_token_idx[0] # 解开batch维度

                # logger.info(f"Get Important token indices in layer_{j}: {imp_token_idx}")

                timers("imp load and compute").start(self.sync)
                k_cache, v_cache = self.get_prefix_kv(imp_token_idx, j)
                timers("imp load and compute").stop(self.sync)
                
                timers("compute").start(self.sync)
                h, new_k_cache, new_v_cache = self.compute.gqa_prefill_modified(h, mask, i_n, w_q, w_k, w_v, w_out, self.rms_norm_eps,
                                freqs_cis, n_head, num_key_value_heads, k_cache, v_cache, donate, self.policy.compress_cache, self.policy.comp_cache_config,
                                self.task.common_prefix_len)
                timers("compute").stop(self.sync)
                
                self.prefill_cache_shape = new_k_cache.shape[0]
            else:
                timers("compute").start(self.sync)
                h, new_k_cache, new_v_cache = self.compute.gqa(h, mask, i_n, w_q,
                    w_k, w_v, w_out, self.rms_norm_eps, freqs_cis,
                    n_head, num_key_value_heads, donate, self.policy.compress_cache, self.policy.comp_cache_config)
                timers("compute").stop(self.sync)
                self.prefill_cache_shape = self.task.prompt_len
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[7]), (v_cache, donate[8]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.gqa_gen(h, mask, i_n, w_q,
                w_k, w_v, w_out, self.rms_norm_eps, freqs_cis,
                n_head, num_key_value_heads, k_cache, v_cache, donate,
                self.policy.compress_cache, self.policy.comp_cache_config, pos=self.prefill_cache_shape + i)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.rms_norm_eps = config.rms_norm_eps

        self.task = None

    def set_task(self, task):
        self.task = task
    
    def set_chunk_pool(self, chunk_pool):
        pass

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        intermediate_size = self.config.intermediate_size
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # pos_n
            ((h, ), dtype, path + "post_attn_layernorm.weight"),
            # gate
            ((intermediate_size, h), dtype, path + "mlp.gate_proj.weight"),
            # up
            ((intermediate_size, h), dtype, path + "mlp.up_proj.weight"),
            # down
            ((h, intermediate_size), dtype, path + "mlp.down_proj.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        pos_n, gate, up, down = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                pos_n.smart_copy(dst2),
                gate.smart_copy(dst1),
                up.smart_copy(dst1),
                down.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, j):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j, freqs_cis):
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((pos_n, donate[1]), (gate, donate[2]), (up, donate[3]), (down, donate[4])) = weight_read_buf.pop()
        else:
            ((pos_n, _), (gate, _), (up, _), (down, _)) = weight_read_buf.val

        h = self.compute.llama_mlp(h, pos_n, gate, up, down, self.rms_norm_eps, donate)
        hidden.val = h


class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)
    
    def set_chunk_pool(self, chunk_pool):
        self.attention.set_chunk_pool(chunk_pool)
        #self.mlp.set_chunk_pool(chunk_pool)

    def set_probe_chunk_pool(self, probe_chunk_pool):
        self.attention.set_probe_chunk_pool(probe_chunk_pool)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        self.attention.init_cache_one_gpu_batch(cache_home, max_prompt_len, max_gen_len)

    def load_cache(self, cache_home, cache_read_buf, i, j):
        self.attention.load_cache(cache_home, cache_read_buf, i, j)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j, freqs_cis):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k, j, freqs_cis)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k, j, freqs_cis)

        self.prefill_cache_shape = self.attention.prefill_cache_shape


class LLAMA:
    def __init__(self,
                 config: LlamaConfig,
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy,
                 max_length: int,
                 max_prompt_len: int,
                 max_gen_len: int):
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        
        self.max_length = max_length
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len

        self.freqs_cis = precompute_freqs_cis(
            self.config.hidden_size // self.config.n_head,
            max_length * 2,
            self.config.rope_theta
        )

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            assert(policy.sep_layer == False)
            layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

        self.radix_tree = RadixTree()
        self.chunk_pool = ChunkPool(self.env, self.config, chunk_size=self.policy.chunk_size) # Initialize chunk pool
        self.probe_chunk_pool = ProbeChunkPool(self.env, self.config, batch_size=self.policy.gpu_batch_size, chunk_size=self.policy.probe_chunk_size) # Initialize probe chunk pool
        self.set_chunk_pool()
        self.set_probe_chunk_pool()
        
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
                self.init_cache(j, k, max_gen_len, max_prompt_len)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy, max_gen_len, max_prompt_len)

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def set_chunk_pool(self):
        for l in self.layers:
            l.set_chunk_pool(self.chunk_pool)

    def set_probe_chunk_pool(self):
        for l in self.layers:
            if hasattr(l, 'set_probe_chunk_pool'):
                l.set_probe_chunk_pool(self.probe_chunk_pool)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        # check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        # if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
        #     download_opt_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)
        

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k, max_gen_len, max_prompt_len):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k], max_gen_len, max_prompt_len)

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        # if i == 0:  # prefill, no cache
        #     return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i, j)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i, j)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return
        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # i: iteration, j: layer, k: batch
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k, j, self.freqs_cis)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def warmup(self):
        typical_shapes = [(1, 1024, 4096, 128)]
        timers('warmup').start()
        self.env.gpu.warmup_gpu(typical_shapes)
        timers('warmup').stop()

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)
    
    def generate_with_prefix(self, inputs: Union[np.array, List[int]]):
        prefix_kv_ptr = self.radix_tree.search(inputs)
        return prefix_kv_ptr, len(prefix_kv_ptr)

    def store_prefix_cache(self):
        common_prefix_len = self.task.common_prefix_len
        inputs = self.task.inputs[0]
        prompt_len = self.task.prompt_len

        kv_ptr = [[] for _ in range(common_prefix_len, prompt_len)]
        probe_ptr = [[] for _ in range(common_prefix_len, prompt_len)]

        for layer in range(self.num_layers):
            for t_idx in range(common_prefix_len, prompt_len):
                if not hasattr(self.layers[layer], 'attention'):
                    kv_ptr[t_idx - common_prefix_len].append(None)
                    probe_ptr[t_idx - common_prefix_len].append(None)
                    continue
                for b in range(self.policy.num_gpu_batches):
                    src = self.cache_home[layer][b]
                    k_cache, v_cache = src.val
                    ptr = self.chunk_pool.store_prefix_cache(k_cache, v_cache, self.layers[layer].prefill_cache_shape + t_idx - prompt_len)
                    kv_ptr[t_idx - common_prefix_len].append(ptr)
                    pr_ptr = self.probe_chunk_pool.store_probe_k(k_cache, self.layers[layer].prefill_cache_shape + t_idx - prompt_len)
                    probe_ptr[t_idx - common_prefix_len].append(pr_ptr)


        logging.info(f"kv_ptr:{len(kv_ptr)}")

        self.radix_tree.insert(inputs, kv_ptr, probe_ptr)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 0.6,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        common_prefix_token, common_prefix_len = self.generate_with_prefix(inputs[0])
        logger.info(f"generate: common_prefix_len={common_prefix_len}")
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            common_prefix_token=common_prefix_token,
            common_prefix_len=common_prefix_len
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        
        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                #self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()

        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        self.set_task(task)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        # for j in range(num_layers):
        #     for k in range(num_gpu_batches):
        #         self.delete_cache(j, k)
        # if self.policy.cpu_cache_compute:
        #     self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def finish_one_query(self, idx=0):
        self.sync()
        if idx == 0:
            self.store_prefix_cache()
        self.sync()
        logger.info("query finished , now sync the model")

    def final_finish(self):
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()


    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            
            self.warmup()
            timers("generate").start()
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            
            self.update_attention_mask(i, 0)
            self.warmup() 
            timers("generate").start()
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts).input_ids
    return (input_ids[0],) * num_prompts

def get_tokenized_inputs(prompt, max_prompt_len, tokenizer):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    inputs_ids = tokenizer(prompt, max_length=max_prompt_len, truncation=True).input_ids
    return inputs_ids


def normalize_llama_model_name(model_name):
    model_name = model_name.lower()
    if model_name in LLAMA_MODEL_SPECS:
        return model_name
    if "70b" in model_name:
        return "meta/llama3.3-70b"
    if "8b" in model_name:
        return "meta/llama3.1-8b"
    raise ValueError(f"Unsupported llama model: {model_name}")


def resolve_snapshot_dir(model_dir):
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_dir

    snapshot_names = sorted(
        name for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    )
    if not snapshot_names:
        raise FileNotFoundError(f"No snapshots found under {snapshots_dir}")
    return os.path.join(snapshots_dir, snapshot_names[-1])


def resolve_llama_artifacts(args):
    model_key = normalize_llama_model_name(args.model)
    spec = LLAMA_MODEL_SPECS[model_key]

    tokenizer_root = args.tokenizer_path
    if not tokenizer_root:
        tokenizer_root = os.path.join(args.path, spec["hf_cache_dir"])
    tokenizer_path = resolve_snapshot_dir(tokenizer_root)

    llama_config = get_llama_config_from(spec["config_name"], tokenizer_path)
    weights_dir = os.path.abspath(os.path.expanduser(
        os.path.join(args.path, f"{llama_config.name}-np")
    ))
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weight directory not found: {weights_dir}")

    return tokenizer_path, llama_config, weights_dir


def get_llama_tokenizer(args):
    tokenizer_path, _, _ = resolve_llama_artifacts(args)
    logger.info("use tokenizer from %s", tokenizer_path)
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        truncation_side="left",
    )


def init_llama_runtime(args):
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    max_prompt_len, gen_len = args.prompt_len, args.gen_len
    _, llama_config, weights_dir = resolve_llama_artifacts(args)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(
        gpu=gpu,
        cpu=cpu,
        disk=disk,
        mixed=TorchMixedDevice([gpu, cpu, disk]),
    )

    policy = Policy(
        args.gpu_batch_size,
        args.num_gpu_batches,
        args.percent[0],
        args.percent[1],
        args.percent[2],
        args.percent[3],
        args.percent[4],
        args.percent[5],
        args.overlap,
        args.sep_layer,
        args.pin_weight,
        args.cpu_cache_compute,
        args.attn_sparsity,
        args.compress_weight,
        CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        args.compress_cache,
        CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
    )
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    cache_size = llama_config.cache_bytes(num_prompts, max_prompt_len + gen_len)
    hidden_size = llama_config.hidden_bytes(num_prompts, max_prompt_len + gen_len)
    print(
        f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
        f"cache size: {cache_size/GB:.3f} GB, "
        f"hidden size (prefill): {hidden_size/GB:.3f} GB"
    )
    logger.info(
        "resolved model=%s config_name=%s num_layers=%d hidden=%d weights=%s",
        args.model,
        llama_config.name,
        llama_config.num_hidden_layers,
        llama_config.hidden_size,
        weights_dir,
    )

    print("init weight...init_cache_home...")
    max_length = max(16384, max_prompt_len + gen_len)
    model = LLAMA(
        llama_config,
        env,
        args.path,
        policy,
        max_length,
        max_prompt_len,
        gen_len,
    )
    return model, env, gpu, cpu


def reset_full_test_timers():
    for timer_name in (
        "generate",
        "warmup",
        "imp io",
        "imp calc",
        "compute",
        "imp load and compute",
        "cache store",
        "probe_cache",
        "full_cache",
        "imp choose1",
        "imp choose2",
        "imp choose3",
        "imp choose4",
        "chunk io",
        "mlp",
        "load cache",
        "load weight",
        "store cache",
        "compute layer",
    ):
        timers(timer_name).reset()


def print_generation_outputs(tokenizer, output_ids, args):
    if DUMMY_WEIGHT in args.path:
        return

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    show_str = "Outputs:\n" + 70 * "-" + "\n"
    for idx, output in enumerate(outputs):
        show_str += f"{idx}: {output}\n"
        show_str += "-" * 70 + "\n"
    if args.verbose >= 2:
        print(show_str)


def log_full_test_metrics():
    print("warmup sum:", timers("warmup").elapsed("sum"))
    print("warmup:", timers("warmup").costs)
    print("imp io average:", timers("imp io").elapsed("average"))
    print("imp io sum:", timers("imp io").elapsed("sum"))
    print("imp calc average:", timers("imp calc").elapsed("average"))
    print("imp calc sum:", timers("imp calc").elapsed("sum"))
    print("imp calc:", timers("imp calc").costs)
    print("compute average:", timers("compute").elapsed("average"))
    print("compute sum:", timers("compute").elapsed("sum"))
    print("prefill:", timers("generate").costs[0])
    print(
        (timers("imp io").elapsed("sum") + timers("imp calc").elapsed("sum"))
        / timers("generate").costs[0]
        * 100
    )
    print("imp load :", timers("imp load and compute").elapsed("average"))
    print("imp sum:", timers("imp load and compute").elapsed("sum"))
    print("cache store average:", timers("cache store").elapsed("average"))
    print("cache store sum:", timers("cache store").elapsed("sum"))
    print("generate:", timers("generate").costs)
    print("generate sum:", timers("generate").elapsed("sum"))
    print(
        "probe_cache: ",
        timers("probe_cache").elapsed("average"),
        "  ",
        timers("probe_cache").elapsed("sum"),
    )
    print(
        "full_cache: ",
        timers("full_cache").elapsed("average"),
        "  ",
        timers("full_cache").elapsed("sum"),
    )


def drop_cache():
    try:
        import subprocess
        subprocess.run(["sudo", "drop_cache"], check=True)
        logger.info("Successfully dropped system caches")
    except subprocess.CalledProcessError as exc:
        logger.warning(f"Failed to drop caches: {exc}")
    except Exception as exc:
        logger.warning(f"Unexpected error when dropping caches: {exc}")


def maybe_drop_cache(enabled: bool):
    if enabled:
        drop_cache()


def run_full_request_set(model, tokenizer, args, context, questions):
    max_prompt_len, cut_gen_len = args.prompt_len, args.cut_gen_len

    prefix_input = get_tokenized_inputs(
        context,
        max_prompt_len=max_prompt_len,
        tokenizer=tokenizer,
    )
    print(len(prefix_input[0]))
    model.generate(
        prefix_input,
        max_new_tokens=1,
        debug_mode=args.debug_mode,
        cut_gen_len=cut_gen_len,
        verbose=args.verbose,
    )
    model.sync()

    inputs = [context + query + "\n" for query in questions]
    inputs_ids = tokenizer(inputs, truncation=True, max_length=max_prompt_len).input_ids
    logger.info("num_questions=%d max_input_tokens=%d", len(inputs_ids), max(len(ids) for ids in inputs_ids))

    prefill_history = []
    for idx, input_ids in enumerate(inputs_ids):
        reset_full_test_timers()
        output_ids = model.generate(
            inputs=[input_ids],
            max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode,
            cut_gen_len=cut_gen_len,
            verbose=args.verbose,
        )
        print_generation_outputs(tokenizer, output_ids, args)
        prefill_history.append(timers("generate").costs[0])
        start_cache_store = torch.cuda.Event(enable_timing=True)
        end_cache_store = torch.cuda.Event(enable_timing=True)
        timers("cache store").start(start_cache_store.record())
        model.finish_one_query(idx)
        maybe_drop_cache(args.drop_cache)
        end_cache_store.record()
        timers("cache store").stop(end_cache_store.synchronize())
        log_full_test_metrics()

    return prefill_history

def run_flexllmgen(args):
    print(f"<run_flexllmgen>: args.model: {args.model}")
    tokenizer = get_llama_tokenizer(args)
    _, llama_config, weights_dir = resolve_llama_artifacts(args)

    # if args.model == "facebook/galactica-30b":
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    cache_size = llama_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = llama_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")
    logger.info(
        "resolved model=%s config_name=%s num_layers=%d hidden=%d weights=%s",
        args.model,
        llama_config.name,
        llama_config.num_hidden_layers,
        llama_config.hidden_size,
        weights_dir,
    )

    print("init weight...")
    max_length = max(4096, prompt_len + gen_len)
    model = LLAMA(llama_config, env, args.path, policy, max_length, prompt_len, gen_len)

    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file


    log_str = write_benchmark_log(filename,
        llama_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)

def process_dapr():
    docs = load_dataset(
        DEFAULT_DATASET_ROOT + "UKPLab___dapr/ConditionalQA-docs/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/",
        split="test",
    )
    qrels = load_dataset(
        DEFAULT_DATASET_ROOT + "UKPLab___dapr/ConditionalQA-qrels/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/",
        split="test",
    )
    queries = load_dataset(
        DEFAULT_DATASET_ROOT + "UKPLab___dapr/ConditionalQA-queries/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/",
        split="test",
    )
    print(queries)
    qrels_dict = defaultdict(set)

    for row in qrels:
        corpus_id = row["corpus_id"]
        doc_id = corpus_id.split('-')[0]
        qrels_dict[doc_id].add(row["query_id"])
    
    doc_id, query_ids = None, None
    for k, v in qrels_dict.items():
        if (len(v) > 10):
            doc_id, query_ids = k, v
            break
    #print(doc_id)
    docs = docs.filter(lambda row: row["doc_id"] == doc_id)

    passages = docs[0]["passages"]
    context = ""
    for psg in passages:
        context += psg + "\n"
    #print(context)
    questions = []
    target_queries = queries.filter(lambda q: q["_id"] in query_ids)
    for q in target_queries:
        questions.append(q["text"])
    #print(questions)
    return context, questions


def process_full_dapr():
    docs = load_dataset(
        DEFAULT_DATASET_ROOT + "UKPLab___dapr/ConditionalQA-docs/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/",
        split="test",
    )
    qrels = load_dataset(
        DEFAULT_DATASET_ROOT + "UKPLab___dapr/ConditionalQA-qrels/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/",
        split="test",
    )
    queries = load_dataset(
        DEFAULT_DATASET_ROOT + "UKPLab___dapr/ConditionalQA-queries/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/",
        split="test",
    )
    print(queries)
    qrels_dict = defaultdict(set)

    for row in qrels:
        corpus_id = row["corpus_id"]
        doc_id = corpus_id.split("-")[0]
        qrels_dict[doc_id].add(row["query_id"])

    def process_doc(target_doc_id):
        target_docs = docs.filter(lambda row: row["doc_id"] == target_doc_id)
        passages = target_docs[0]["passages"]
        return "".join(f"{passage}\n" for passage in passages)

    def process_query(query_ids):
        target_queries = queries.filter(lambda query: query["_id"] in query_ids)
        return [query["text"] for query in target_queries]

    requests = {}
    for doc_id, query_ids in qrels_dict.items():
        if len(query_ids) < 5:
            continue
        requests[doc_id] = (process_doc(doc_id), process_query(query_ids))

    return requests


def process_full_longbench():
    task_name = "narrativeqa"
    file_path = DEFAULT_DATASET_ROOT + f"THUDM___long_bench/data/{task_name}.jsonl"
    dataset = load_dataset("json", data_files=file_path)["train"]

    context_to_questions = defaultdict(list)
    for row in dataset:
        context_to_questions[row["context"]].append(row["input"])

    requests = {}
    request_id = 0
    for context, questions in context_to_questions.items():
        if len(questions) < 10:
            continue
        requests[request_id] = (context[:18000], questions)
        request_id += 1

    return requests


def run_dapr_flexllmgen(args):
    print(f"<run_llama_dapr_flexllmgen>: args.model: {args.model}")
    tokenizer = get_llama_tokenizer(args)
    model, env, gpu, cpu = init_llama_runtime(args)
    max_prompt_len, cut_gen_len = args.prompt_len, args.cut_gen_len

    context, questions = process_dapr()
    inputs = [context +  query + "\n" for query in questions]
    inputs_ids = tokenizer(inputs, truncation=True, max_length=max_prompt_len).input_ids
    logger.info(f"max_input_tokens: {max([len(ids) for ids in inputs_ids])}")
    output_ids = model.generate(
        inputs=[inputs_ids[0]], max_new_tokens = 1, debug_mode=args.debug_mode, 
        cut_gen_len=cut_gen_len, verbose=args.verbose)
    print("="*50,"warmup - generate finished", "=" * 50)
    for i in range(len(inputs)):
        timers("generate").reset()
        output_ids = model.generate(
            inputs=[inputs_ids[i]], max_new_tokens = args.gen_len, debug_mode=args.debug_mode, 
            cut_gen_len=cut_gen_len, verbose=args.verbose)
        if DUMMY_WEIGHT not in args.path:
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # prompt_tail = tokenizer.decode(inputs_ids[i][-100:]) if len(inputs_ids[i]) > 100 else tokenizer.decode(inputs_ids[i])
            # print(f"\n{'='*60}")
            # print(f"Prompt (last 100 chars):\n{prompt_tail}")
            # print(f"{'-'*60}")
            # print(f"Generated:\n{outputs[0][-32:]}")
            # print(f"{'='*60}\n")
            show_str = "Outputs:\n" + 70 * '-' + "\n"
            for i in [0, len(outputs)-1]:
                show_str += f"{i}: {outputs[i]}\n"
                show_str += "-" * 70 + "\n"
            if args.verbose >= 2:
                print(show_str)

        print("prefill:",timers("generate").costs[0])
        model.finish_one_query(i)
        maybe_drop_cache(args.drop_cache)
        print("=" * 50)
    model.final_finish()
    env.close_copy_threads()
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()
    print(f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t" + f"peak cpu mem: {cpu_peak_mem / GB:.3f} GB\t")


def run_full_dapr_flexllmgen(args):
    print(f"<run_full_dapr_flexllmgen>: args.model: {args.model}")
    tokenizer = get_llama_tokenizer(args)
    model, env, gpu, cpu = init_llama_runtime(args)

    requests = process_full_dapr()
    print(len(requests), flush=True)

    for doc_id, (context, questions) in requests.items():
        logger.info("start full_dapr doc_id=%s num_questions=%d", doc_id, len(questions))
        prefill_history = run_full_request_set(model, tokenizer, args, context, questions)

        _, gpu_peak_mem = gpu.mem_stats()
        _, cpu_peak_mem = cpu.mem_stats()
        print(f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t" + f"peak cpu mem: {cpu_peak_mem / GB:.3f} GB\t")
        if prefill_history:
            recent_prefill = prefill_history[1:]
            print(f"Last {len(recent_prefill)} prefill values: {recent_prefill}")
            if recent_prefill:
                print(
                    f"Average of last {len(recent_prefill)} prefill values: "
                    f"{sum(recent_prefill)/len(recent_prefill):.6f}"
                )

        del model.radix_tree
        model.radix_tree = RadixTree()

    model.final_finish()
    env.close_copy_threads()


def run_full_longbench_flexllmgen(args):
    print(f"<run_full_longbench_flexllmgen>: args.model: {args.model}")
    tokenizer = get_llama_tokenizer(args)
    model, env, gpu, cpu = init_llama_runtime(args)

    requests = process_full_longbench()
    print(len(requests), flush=True)

    for request_id, (context, questions) in requests.items():
        logger.info(
            "start full_longbench request_id=%s num_questions=%d",
            request_id,
            len(questions),
        )
        prefill_history = run_full_request_set(model, tokenizer, args, context, questions)

        _, gpu_peak_mem = gpu.mem_stats()
        _, cpu_peak_mem = cpu.mem_stats()
        print(f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t" + f"peak cpu mem: {cpu_peak_mem / GB:.3f} GB\t")
        if prefill_history:
            recent_prefill = prefill_history[1:]
            print(f"Last {len(recent_prefill)} prefill values: {recent_prefill}")
            if recent_prefill:
                print(
                    f"Average of last {len(recent_prefill)} prefill values: "
                    f"{sum(recent_prefill)/len(recent_prefill):.6f}"
                )

        del model.radix_tree
        model.radix_tree = RadixTree()

    model.final_finish()
    env.close_copy_threads()


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH,
        help="The path to the tokenizer.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=1)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=False)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--drop-cache", action="store_true",
        help="Drop system page cache after each query finishes.")
    parser.add_argument("--input", type=str,
        choices=["dapr", "full_dapr", "full_longbench"], default="dapr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    if args.input == "dapr":
        run_dapr_flexllmgen(args)
    elif args.input == "full_dapr":
        run_full_dapr_flexllmgen(args)
    elif args.input == "full_longbench":
        run_full_longbench_flexllmgen(args)
