"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import math
import os
import queue
import shutil
import time
import threading
import math
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from flexllmgen.timer import timers
import logging

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

from flexllmgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexllmgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

### from https://github.com/meta-llama/llama3/blob/main/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_len: int,
    k_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis_q = freqs_cis[:q_len]
    freqs_cis_k = freqs_cis[:k_len]

    xq_ = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], xq.shape[-1] // 2, 2)
    )
    freqs_cis_q = reshape_for_broadcast(freqs_cis_q, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    
    xk_ = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], xk.shape[-1] // 2, 2)
    )
    freqs_cis_k = reshape_for_broadcast(freqs_cis_k, xk_)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_separate(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_q: torch.Tensor,
    freqs_cis_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], xq.shape[-1] // 2, 2)
    )
    freqs_cis_q = reshape_for_broadcast(freqs_cis_q, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)

    xk_ = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], xk.shape[-1] // 2, 2)
    )
    freqs_cis_k = reshape_for_broadcast(freqs_cis_k, xk_)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def select_important_token_idx(topk_idx: torch.Tensor, common_prefix_len: int):
    b, n_probe_head, _ = topk_idx.shape
    threshold = (
        (topk_idx.shape[2] / common_prefix_len) /
        (2 - topk_idx.shape[2] / common_prefix_len)
    ) ** 0.6

    if n_probe_head > 1:
        jaccard_sum = torch.zeros(b, device=topk_idx.device, dtype=torch.float32)
        comb = 0
        for l in range(n_probe_head):
            lhs = topk_idx[:, l, :]
            for r in range(l + 1, n_probe_head):
                rhs = topk_idx[:, r, :]
                inter = (lhs.unsqueeze(2) == rhs.unsqueeze(1)).any(dim=2).sum(dim=1)
                union = lhs.shape[1] + rhs.shape[1] - inter
                jaccard_sum += inter.float() / union.float().clamp_min(1.0)
                comb += 1
        keep_mask = (jaccard_sum / comb) >= threshold
    else:
        keep_mask = torch.ones(b, device=topk_idx.device, dtype=torch.bool)

    sorted_topk_idx = torch.sort(topk_idx[:, 0, :], dim=1).values.cpu().tolist()
    keep_mask = keep_mask.cpu().tolist()

    full_prefix_idx = list(range(common_prefix_len))
    imp_token_idx = []
    for i in range(b):
        if keep_mask[i]:
            imp_token_idx.append(sorted_topk_idx[i])
        else:
            imp_token_idx.append(full_prefix_idx)

    return imp_token_idx

class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy, max_gen_len, max_prompt_len):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = max_gen_len + max_prompt_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy, max_gen_len, max_prompt_len)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_torch_tensor(self, data):
        data = data.to(device=self.dev)
        return TorchTensor.create_from_torch(data, self)

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)
    
    def llama_input_embed(self, inputs, w_token, donate):
        ## inputs: inpuit_ids; w_token: emb weights
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
        
        token_ids = inputs.data
        if donate[0]: inputs.delete()
        
        token_embed = F.embedding(token_ids, w_token.data)
        
        return TorchTensor.create_from_torch(token_embed, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        # logger.info(f">>> opt_input_embed: positions shape:, {positions.shape}")
        # logger.info(f">>> positions min/max:, {positions.min().item()}, {positions.max().item()}")
        # logger.info(f">>> w_pos.weight shape:, {w_pos.data.shape}")

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def topp_temperature_decode(self, logits, temperature=0.6, top_p=0.9):
        """
        Perform Top-p (nucleus) sampling with temperature decoding in PyTorch.
        
        Args:
            logits (torch.Tensor): Input logits of shape [b, 1, vocab_size].
            temperature (float): Temperature for scaling logits.
            top_p (float): Probability threshold for nucleus sampling.
        
        Returns:
            torch.Tensor: Decoded indices of shape [b, 1].
        """
        # Apply temperature scaling
        logits = logits / temperature

        # Compute probabilities using softmax
        probs = torch.softmax(logits, dim=-1)
        
        # Sort probabilities and corresponding indices in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create a mask for tokens that exceed the top-p threshold
        mask = cumulative_probs > top_p
        # Shift the mask to preserve at least one token
        mask[:, :, 1:] = mask[:, :, :-1].clone()
        mask[:, :, 0] = False

        # Mask out probabilities beyond the top-p threshold
        sorted_probs.masked_fill_(mask, 0.0)
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # Re-normalize probabilities

        # Sample from the filtered probability distribution
        sampled_indices = torch.multinomial(sorted_probs.squeeze(1), num_samples=1)

        # Map back to original indices
        final_indices = sorted_indices.gather(dim=-1, index=sampled_indices.unsqueeze(-1))

        return final_indices.squeeze(-1)

    def llama_output_embed(self, inputs, norm, w_lm, eps, donate, temperature=0.6, topp=0.9):
        # decompress weights
        if w_lm.device.device_type == DeviceType.COMPRESSED:
            w_lm = w_lm.device.decompress(w_lm)
            norm = norm.device.decompress(norm)
        
        b, s, h = inputs.shape
        
        hidden = F.rms_norm(inputs.data, (h,), weight=norm.data, eps=eps)
        if donate[0]: inputs.delete()
        
        logits = F.linear(hidden, w_lm.data)
        last_token_logits = logits[:, -1:, :]
        
        if temperature < 0.1:
            ids = last_token_logits.argmax(dim=-1)
        else :
            ids = self.topp_temperature_decode(last_token_logits, temperature, topp)
        
        return TorchTensor.create_from_torch(ids, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch_llama(self, config, task, policy, max_prompt_len, max_gen_len):
        num_head, num_key_value_heads, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            #config.n_head, config.num_key_value_heads, config.input_dim, task.prompt_len, task.gen_len,
            config.n_head, config.num_key_value_heads, config.input_dim, max_prompt_len, max_gen_len,
            policy.gpu_batch_size)
        shape = (max_prompt_len + max_gen_len - 1, gpu_batch_size * num_key_value_heads, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache

    #def init_cache_one_gpu_batch(self, config, task, policy):
    def init_cache_one_gpu_batch(self, config, task, policy, max_prompt_len, max_gen_len):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            #config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            config.n_head, config.input_dim, max_prompt_len, max_gen_len,
            policy.gpu_batch_size)
        shape = (max_prompt_len + max_gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache


    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s) # 下三角矩阵
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)
        
        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        #logger.info(f"mha:k.data = {k.data[:,:3,:10]}")

        return TorchTensor.create_from_torch(value, self), k, v

    def warmup_gpu(self, typical_shapes):
        for shapes in typical_shapes:
            b, s, common_prefix_len, h = shapes
            # 创建假数据
            inputs = torch.randn(b, s, h, device='cuda')
            k_cache = torch.randn(common_prefix_len, b * 3, 128, device='cuda')
            attention_mask = torch.ones(b, common_prefix_len, dtype=torch.bool, device='cuda')
            _ = F.layer_norm(inputs, (h,))
            _ = torch.bmm(inputs, inputs.transpose(1, 2))
            _ = F.softmax(torch.randn(1, s, common_prefix_len, device='cuda'), dim=-1)

    def get_important_token_idx(self, inputs, attention_mask, w_q, b_q,
                w_ln, b_ln, n_head, k_cache, donate,
                compress_cache, comp_config, important_ratio):
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
        '''
            b, s, h = inputs.shape,包含整个inputs的token
            取q的前三个头参数。计算得到q向量shape: (b * n_probe_head, s, head_dim)
            传入的k缓存shape:(common_prefix_len, b * n_probe_head, head_dim)
            调整shape计算attn_weights (b * n_probe_head, s, common_prefix_len)
            mask和softmax后计算在dim=1上的和得到attn_sum(b * n_probe_head, common_prefix_len)
            对attn_sum在dim=1上求topk,得到每个头的重要token下标(b * n_probe_head, n_important)
            计算平均jaccard指数,超过threshold返回第1个头的重要token下标集
            没超过返回range(common_prefix_len)表示后续要使用所有token的KV缓存

            注意力掩码部分没理清,是乱写的0.0
        '''
        if compress_cache:
            # shape: (common_prefix_len, b * n_probe_head, head_dim)
            k = k_cache.device.decompress(k_cache)
        else:
            # shape: (common_prefix_len, b * n_probe_head, head_dim)
            k = k_cache.data

        n_probe_head = 3
        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        
        common_prefix_len = k_cache.shape[0]
        n_important = math.ceil(common_prefix_len * important_ratio)
        
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # w_q_probe = w_q.data.view(h, n_head, head_dim)[:, :n_probe_head, :].reshape(h, n_probe_head * head_dim)
        # b_q_probe = b_q.data.view(n_head, head_dim)[:n_probe_head, :].reshape(n_probe_head * head_dim)

        w_q_probe = w_q.data[:n_probe_head * head_dim,:]
        b_q_probe = b_q.data[:n_probe_head * head_dim]
        
        #logger.info(f"IMP_Token: w_q_probe's shape={w_q_probe.data.shape},b_q_probe's shape={b_q_probe.shape},hidden's shape={b_q.data.shape}")

        # shape: (b * n_probe_head, s, head_dim)
        q = F.linear(hidden, w_q_probe, bias=b_q_probe) * scaling
        # shape: (b, s, n_probe_head, head_dim)
        q = q.view(b, s, n_probe_head, head_dim)
        # shape: (b * n_probe_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_probe_head, s, head_dim)

        # origin k_cache shape: (common_prefix_len, b * n_probe_head, head_dim)
        # shape:(b * n_probe_head, head_dim, common_prefix_len)
        k = k.permute(1, 2, 0).reshape(b * n_probe_head, head_dim, common_prefix_len)

        # shape: (b * n_probe_head, s, common_prefix_len)
        attn_weights = torch.bmm(q, k)
        
        mask = attention_mask.data[:, :common_prefix_len].view(b, 1, 1, common_prefix_len)
        #logger.info(f"IMP_Token mask={mask}")
        # expand to (b, n_probe_head, s, common_prefix_len)
        mask = mask.expand(b, n_probe_head, s, common_prefix_len)
        # reshape to match attn_weights: (b * n_probe_head, s, common_prefix_len)
        mask = mask.reshape(b * n_probe_head, s, common_prefix_len)

        attn_weights = torch.where(mask, attn_weights, -1e4)   
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_sum = attn_weights.sum(dim = 1)
        _, topk_idx = torch.topk(attn_sum, k=n_important, dim=1)
        topk_idx = topk_idx.view(b, n_probe_head, n_important)

        S_imp = [[] for _ in range(b)]
        for i in range(b):
            for j in range(n_probe_head):
                idx_set = { int(x) for x in topk_idx[i,j]}
                S_imp[i].append(idx_set)

        #logger.info(f"IMP_Token Set:{S_imp}")

        thresold = ((n_important/common_prefix_len) / (2 - n_important/common_prefix_len)) ** 0.6
        
        #logger.info(f"IMP_Token: thresold={thresold}")

        imp_token_idx = []
        for i in range(b):
            jaccard = 0
            for l in range(n_probe_head):
                for r in range(l+1, n_probe_head):
                    jaccard += len(S_imp[i][l] & S_imp[i][r])/len(S_imp[i][l] | S_imp[i][r])
            comb = (n_probe_head * (n_probe_head - 1)) / 2
            jaccard /= comb
            if jaccard >= thresold:
                imp_token_idx.append(sorted(S_imp[0][0]))
            else:
                imp_token_idx.append(list(range(common_prefix_len))) #表示加载全部kv

        k_cache.delete()

        return imp_token_idx
        
    def mha_prefill(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                compress_cache, comp_config, common_prefix_len, imp_token_idx):
        """Multi-head attention (prefill with prefix cache)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        if compress_cache:
            # shape: (n_imp, b * n_head, head_dim)
            k = k_cache.device.decompress(k_cache)
            v = v_cache.device.decompress(v_cache)
        else:
            # shape: (n_imp, b * n_head, head_dim)
            k = k_cache.data
            v = v_cache.data
        
        '''
        整个序列计算q向量(b, s, h)
        计算非前缀部分的k_new,v_new向量(b, suffix_len, h)   suffix_len = s - common_prefix_len
        获取前缀部分的kv缓存: (n_imp, b * n_head, head_dim), 把两部分拼接在一起(n_imp + suffix_len, b * n_head, head_dim)
        
        计算attn_weights(b * n_head, s, n_imp + suffix_len)
        mask和softmax后计算attn_values(b * n_head, s, head_dim)

        调整维度后和inputs.data相加返回。同时返回拼接后的kv_cache向量(n_imp + suffix_len, b * n_head, head_dim)
        '''

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        n_imp = k_cache.shape[0] # 重要token的数量
        suffix_len = s - common_prefix_len 

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        suffix_hidden = hidden[:, common_prefix_len:, :]  # hidden for suffix tokens
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        # shape: (b, suffix_len, h)
        k_new = F.linear(suffix_hidden, w_k.data, bias=b_k.data)
        v_new = F.linear(suffix_hidden, w_v.data, bias=b_v.data)

        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        # shape: (b, suffix_len, n_head, head_dim)
        k_new = k_new.view(b, suffix_len, n_head, head_dim)
        v_new = v_new.view(b, suffix_len, n_head, head_dim)
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (suffix_len, b * n_head, head_dim)
        k_new = k_new.permute(1, 0, 2, 3).reshape(suffix_len, b * n_head, head_dim)
        v_new = v_new.permute(1, 0, 2, 3).reshape(suffix_len, b * n_head, head_dim)

        # origin k_cache shape: (n_imp + suffix_len, b * n_head, head_dim)
        k = torch.cat((k, k_new), dim=0)
        v = torch.cat((v, v_new), dim=0)
        # shape: (b * n_head, head_dim, n_imp + suffix_len)
        k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, n_imp + suffix_len)
        # shape: (b * n_head, n_imp + suffix_len, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, n_imp + suffix_len, head_dim)
        
        # shape: (b * n_head, s, n_imp + suffix_len)
        attn_weights = torch.bmm(q, k)

        L = n_imp + suffix_len
        # prefix_mask = torch.ones(b, n_imp, dtype=torch.bool, device=attention_mask.device) 
        # suffix_mask = attention_mask.data[:, common_prefix_len:]
        # pad_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        # pad_mask = pad_mask.view(b, 1, 1, L)   

        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1))
        
        token_idxs = imp_token_idx  +  list(range(common_prefix_len, s))
        causal_mask = causal_mask[:, token_idxs]
        #logger.info(f"mha_prefil:token_idxs = {token_idxs}, causal_mask={causal_mask}")
        causal_mask = causal_mask.view(1, 1, s, L) 
        #logger.info(f"mha_prefill: causal_mask.shape = {causal_mask.shape}")
        #mask = pad_mask & causal_mask
        mask = causal_mask

        attn_weights = attn_weights.view(b, n_head, s, L)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, L)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (n_imp + suffix_len, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        #logger.info(f"mha_prefill:k.data = {k.data[:,:3,:10]}")

        return TorchTensor.create_from_torch(value, self), k, v
        

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config, pos):
        """Multi-head attention (decoding phase).
        新增pos参数表示该层当前KVcache的第一维的长度.
        
        计算部分没有进行mask操作
        
        """
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        # src_s = attention_mask.shape[1]
        src_s = pos
        #logger.info(f"mha_gen: src_s = {src_s}, mask.shape={attention_mask.shape[1]}")
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    #logger.info(f"mha_gen: k.dtype{k.dtype}, v.dtype{v.dtype}, q.dtype{q.dtype}")
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new
     

    def gqa(self, inputs, attention_mask, i_n, w_q, w_k, w_v, w_out, eps,
            freqs_cis, n_head, n_kv_head, donate, compress_cache, comp_config):
        """Group query attention (prefill phase)"""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            i_n = i_n.device.decompress(i_n)
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        
        freqs_cis = freqs_cis[:s].to(self.dev)

        # input_layernorm
        hidden = F.rms_norm(inputs.data, (h,), weight=i_n.data, eps=eps)
        
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_head, head_dim)
        v = v.view(b, s, n_kv_head, head_dim)
        
        # RotaryEmbedding
        q, k = apply_rotary_emb(q, k, freqs_cis, s, s)

        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        k = k.permute(0, 2, 3, 1).reshape(b * n_kv_head, head_dim, s)
        v = v.permute(0, 2, 1, 3).reshape(b * n_kv_head, s, head_dim)

        # store origin kv
        ori_k = k.permute(2, 0, 1)
        ori_v = v.permute(1, 0, 2)

        # Expand kv heads to match attention heads if needed
        repeat_kv = n_head // n_kv_head
        if repeat_kv > 1:
            # shape:(b * n_head, head_dim, s)
            k = k.repeat_interleave(repeat_kv, dim=0) 
            v = v.repeat_interleave(repeat_kv, dim=0)
        
        scores = torch.bmm(q, k) / math.sqrt(head_dim)
        # if attention_mask.data is not None:
        #     scores = scores + attention_mask.data  # (b, n_head, s, s)

        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.bmm(scores, v).view(b, n_head, s, head_dim)
        output = output.transpose(1, 2).contiguous().view(b, s, -1)
        out = F.linear(output, w_out.data)

        out.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            ori_k = self.compressed_device.compress(ori_k, comp_config)
            ori_v = self.compressed_device.compress(ori_v, comp_config)
        else:
            ori_k = TorchTensor.create_from_torch(ori_k, self)
            ori_v = TorchTensor.create_from_torch(ori_v, self)
        return TorchTensor.create_from_torch(out, self), ori_k, ori_v

    def gqa_prefill(self, inputs, attention_mask, i_n, w_q, w_k, w_v, w_out, eps,
            freqs_cis, n_head, n_kv_head, k_cache, v_cache, donate, compress_cache, comp_config,
            common_prefix_len):
        
        if w_q.device.device_type == DeviceType.COMPRESSED:
            i_n = i_n.device.decompress(i_n)
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        if compress_cache:
            # shape: (n_imp, b * n_kv_head, head_dim)
            k = k_cache.device.decompress(k_cache)
            v = v_cache.device.decompress(v_cache)
        else:
            # shape: (n_imp, b * n_kv_head, head_dim)
            k = k_cache.data
            v = v_cache.data

        b, s, h = inputs.shape
        head_dim = h // n_head

        n_imp = k_cache.shape[0] # 重要token的数量
        suffix_len = s - common_prefix_len         

        freqs_cis = freqs_cis[:s].to(self.dev)    

        # input_layernorm
        hidden = F.rms_norm(inputs.data, (h,), weight=i_n.data, eps=eps)
        suffix_hidden = hidden[:, common_prefix_len:, :]  # hidden for suffix tokens

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data)
        # shape: (b, suffix_len, h)
        k_new = F.linear(suffix_hidden, w_k.data)
        v_new = F.linear(suffix_hidden, w_v.data)

        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        # shape: (b, suffix_len, n_kv_head, head_dim)
        k_new = k_new.view(b, suffix_len, n_kv_head, head_dim)
        v_new = v_new.view(b, suffix_len, n_kv_head, head_dim)

        q, k_new = apply_rotary_emb(q, k_new, freqs_cis, q_len=s, k_len=suffix_len)
        
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        k_new = k_new.permute(1, 0, 2, 3).reshape(suffix_len, b * n_kv_head, head_dim)
        v_new = v_new.permute(1, 0, 2, 3).reshape(suffix_len, b * n_kv_head, head_dim)

        # shape: n_imp + suffix_len , b * n_kv_head, head_dim
        ori_k = torch.cat((k, k_new), dim=0)
        ori_v = torch.cat((v, v_new), dim=0)

        L = n_imp + suffix_len

        repeat_kv = n_head // n_kv_head
        # Expand KV for matmul
        if repeat_kv > 1:
            k = ori_k.repeat_interleave(repeat_kv, dim=1) # L, b * n_head, head_dim
            v = ori_v.repeat_interleave(repeat_kv, dim=1)
        
        # shape: (b * n_head, head_dim, L)
        k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, L)
        # shape: (b * n_head, L, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, L, head_dim)

        scores = torch.bmm(q, k) / math.sqrt(head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        output = torch.bmm(scores, v).view(b, n_head, s, head_dim) # (b, n_head, s, head_dim)
        output = output.transpose(1, 2).reshape(b, s, -1)

        out = F.linear(output, w_out.data)

        out.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            ori_k = self.compressed_device.compress(ori_k, comp_config)
            ori_v = self.compressed_device.compress(ori_v, comp_config)
        else:
            ori_k = TorchTensor.create_from_torch(ori_k, self)
            ori_v = TorchTensor.create_from_torch(ori_v, self)
        
        return TorchTensor.create_from_torch(out, self), ori_k, ori_v
   
    
    def get_important_token_idx_llama(self, inputs, attention_mask, i_n, w_q, eps,
            freqs_cis, n_head, n_kv_head, donate, compress_cache, comp_config, k_cache, important_ratio):
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)

        if compress_cache:
            # shape: (common_prefix_len, b * n_probe_head, head_dim)
            k = k_cache.device.decompress(k_cache)
        else:
            # shape: (common_prefix_len, b * n_probe_head, head_dim)
            k = k_cache.data

        b, s, h = inputs.shape
        head_dim = h // n_head
        n_probe_head = 3

        common_prefix_len = k_cache.shape[0]
        n_important = math.ceil(common_prefix_len * important_ratio)
        
        freqs_cis = freqs_cis[:s].to(self.dev)

        # input_layernorm
        hidden = F.rms_norm(inputs.data, (h,), weight=i_n.data, eps=eps)

        # shape: (b * n_probe_head, s, head_dim)
        w_q_probe = w_q.data[:n_probe_head * head_dim,:]
        q = F.linear(hidden, w_q_probe)
        # shape: (b, s, n_probe_head, head_dim)
        q = q.view(b, s, n_probe_head, head_dim)

        # origin k_cache shape: (common_prefix_len, b * n_probe_head, head_dim)
        # shape:(b, common_prefix_len, n_probe_head, head_dim)
        k = k.view(common_prefix_len, b, n_probe_head, head_dim).permute(1, 0, 2, 3)

        q, k = apply_rotary_emb(q, k, freqs_cis, s, common_prefix_len)

        # shape: (b, n_probe_head, s, head_dim)
        q = q.transpose(1, 2)
        # shape: (b, n_probe_head, common_prefix_len, head_dim)
        k = k.transpose(1, 2)  

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        
        scores = scores.sum(dim = 2)

        _, topk_idx = torch.topk(scores, k=n_important, dim=2)
        imp_token_idx = select_important_token_idx(topk_idx, common_prefix_len)

        k_cache.delete()

        return imp_token_idx
    def gqa_prefill_modified(self, inputs, attention_mask, i_n, w_q, w_k, w_v, w_out, eps,
            freqs_cis, n_head, n_kv_head, k_cache, v_cache, donate, compress_cache, comp_config,
            common_prefix_len):
        
        if w_q.device.device_type == DeviceType.COMPRESSED:
            i_n = i_n.device.decompress(i_n)
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        if compress_cache:
            # shape: (n_imp, b * n_kv_head, head_dim)
            k = k_cache.device.decompress(k_cache)
            v = v_cache.device.decompress(v_cache)
        else:
            # shape: (n_imp, b * n_kv_head, head_dim)
            k = k_cache.data
            v = v_cache.data

        b, s, h = inputs.shape
        head_dim = h // n_head

        n_imp = k_cache.shape[0] # 重要token的数量
        suffix_len = s - common_prefix_len         
        L = n_imp + suffix_len

        freqs_cis = freqs_cis[:s].to(self.dev)    

        # input_layernorm
        hidden = F.rms_norm(inputs.data[:], (h,), weight=i_n.data, eps=eps)
        suffix_hidden = hidden[:, common_prefix_len:, :]  # hidden for suffix tokens

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data)
        k_new = F.linear(suffix_hidden, w_k.data)
        v_new = F.linear(suffix_hidden, w_v.data)

        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        # shape: (b, suffix_len, n_kv_head, head_dim)
        k_new = k_new.view(b, suffix_len, n_kv_head, head_dim)
        v_new = v_new.view(b, suffix_len, n_kv_head, head_dim)

        k = k.view(n_imp, b, n_kv_head, head_dim).permute(1, 0, 2, 3)  # (b, n_imp, n_kv_head, head_dim)
        v = v.view(n_imp, b, n_kv_head, head_dim).permute(1, 0, 2, 3)  # (b, n_imp, n_kv_head, head_dim)
        # shape: (b, n_imp + suffix_len, n_kv_head, head_dim)
        k = torch.cat((k, k_new), dim=1)
        v = torch.cat((v, v_new), dim=1)

        q, k = apply_rotary_emb(q, k, freqs_cis, q_len=s, k_len=L)

        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        ori_k = k.permute(0, 2, 3, 1).reshape(b * n_kv_head, head_dim, L)
        ori_v = v.permute(0, 2, 1, 3).reshape(b * n_kv_head, L, head_dim)

        repeat_kv = n_head // n_kv_head
        # Expand KV for matmul
        if repeat_kv > 1:
            k = ori_k.repeat_interleave(repeat_kv, dim=0)  # b * n_head, head_dim, L
            v = ori_v.repeat_interleave(repeat_kv, dim=0)  # b * n_head, L, head_dim
        else:
            k = ori_k
            v = ori_v

        scores = torch.bmm(q, k) / math.sqrt(head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        output = torch.bmm(scores, v).view(b, n_head, s, head_dim) # (b, n_head, s, head_dim)
        output = output.transpose(1, 2).reshape(b, s, -1)

        out = F.linear(output, w_out.data)
        out.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        ori_k = ori_k.permute(2, 0, 1).contiguous() # L, b*n_kv_head, head_dim
        ori_v = ori_v.permute(1, 0, 2).contiguous() # L, b*n_kv_head, head_dim

        if compress_cache:
            ori_k = self.compressed_device.compress(ori_k, comp_config)
            ori_v = self.compressed_device.compress(ori_v, comp_config)
        else:
            ori_k = TorchTensor.create_from_torch(ori_k, self)
            ori_v = TorchTensor.create_from_torch(ori_v, self)
        
        return TorchTensor.create_from_torch(out, self), ori_k, ori_v
    
    def get_important_token_idx_llama_modified(self, inputs, attention_mask, i_n, w_q, eps,
            freqs_cis, n_head, n_kv_head, donate, compress_cache, comp_config, k_cache, important_ratio):
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)

        if compress_cache:
            # shape: (common_prefix_len, b * n_probe_head, head_dim)
            k = k_cache.device.decompress(k_cache)
        else:
            # shape: (common_prefix_len, b * n_probe_head, head_dim)
            k = k_cache.data

        b, s, h = inputs.shape
        head_dim = h // n_head
        n_probe_head = 3
        repeat_kv = n_head // n_kv_head  # GQA: how many query heads share one KV head

        common_prefix_len = k_cache.shape[0]
        suffix_len = s - common_prefix_len
        n_important = math.ceil(common_prefix_len * important_ratio)

        if suffix_len <= 0:
            k_cache.delete()
            return [list(range(common_prefix_len)) for _ in range(b)]

        freqs_cis = freqs_cis[:s].to(self.dev)
        suffix_freqs_cis = freqs_cis[common_prefix_len:s]
        prefix_freqs_cis = freqs_cis[:common_prefix_len]

        # input_layernorm
        hidden = F.rms_norm(inputs.data[:], (h,), weight=i_n.data, eps=eps)
        suffix_hidden = hidden[:, common_prefix_len:, :]

        # Extract n_probe_head * repeat_kv query heads
        # w_q shape: (n_head * head_dim, h) for GQA, take rows instead of columns
        w_q_probe = w_q.data[:n_probe_head * repeat_kv * head_dim, :]
        q = F.linear(suffix_hidden, w_q_probe)
        # shape: (b, suffix_len, n_probe_head * repeat_kv, head_dim)
        q = q.view(b, suffix_len, n_probe_head * repeat_kv, head_dim)

        # Extract n_probe_head key/value heads
        # k_cache shape: (common_prefix_len, b * n_probe_head, head_dim)
        k = k.view(common_prefix_len, b, n_probe_head, head_dim).permute(1, 0, 2, 3) # b, common_prefix_len, n_probe_head, head_dim

        # Apply rotary embedding
        q, k = apply_rotary_emb_separate(q, k, suffix_freqs_cis, prefix_freqs_cis)

        # shape: (b, n_probe_head * repeat_kv, suffix_len, head_dim)
        q = q.transpose(1, 2)
        # shape: (b, n_probe_head, common_prefix_len, head_dim)
        k = k.transpose(1, 2)

        # Expand k from n_probe_head to n_probe_head * repeat_kv to match q
        # shape: (b, n_probe_head * repeat_kv, common_prefix_len, head_dim)
        k = k.repeat_interleave(repeat_kv, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        scores = scores.sum(dim=2)  # (b, n_probe_head * repeat_kv, common_prefix_len)

        group_scores = scores.view(b, n_probe_head, repeat_kv, common_prefix_len)
        group_scores = group_scores.sum(dim=2)

        _, topk_idx = torch.topk(group_scores, k=n_important, dim=2)
        imp_token_idx = select_important_token_idx(topk_idx, common_prefix_len)

        k_cache.delete()

        return imp_token_idx

    def gqa_gen(self, inputs, attention_mask, i_n, w_q, w_k, w_v, w_out, eps,
            freqs_cis, n_head, n_kv_head, k_cache, v_cache, donate,
            compress_cache, comp_config, pos):
        """Grouped-query attention (decoding phase)."""
        
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            i_n = i_n.device.decompress(i_n)
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        # src_s = attention_mask.shape[1]
        src_s = pos
        head_dim = h // n_head
        
        freqs_cis = freqs_cis[src_s - 1 : src_s - 1 + tgt_s].to(self.dev)

        repeat_kv = n_head // n_kv_head

        #input_layernorm
        hidden = F.rms_norm(inputs.data, (h,), weight=i_n.data, eps=eps)

        q = F.linear(hidden, w_q.data)            # (b, 1, h)
        k = F.linear(hidden, w_k.data)            # (b, 1, h)
        v = F.linear(hidden, w_v.data)            # (b, 1, h)

        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_kv_head, head_dim)
        v = v.view(b, tgt_s, n_kv_head, head_dim)

        # Rotary
        q, k = apply_rotary_emb(q, k, freqs_cis, tgt_s, tgt_s)

        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_head, head_dim)
        
        if compress_cache:
            k = k_cache.device.decompress(k_cache)[:src_s]
            v = v_cache.device.decompress(v_cache)[:src_s]
        else:
            k = k_cache.data[:src_s]
            v = v_cache.data[:src_s]


        # concat cur k v to k_cache and v_cache
        # shape: (s, b * n_kv_head, head_dim)
        k[src_s - 1:src_s] = k_new
        v[src_s - 1:src_s] = v_new

        k = k.repeat_interleave(repeat_kv, dim=1)  # (s, b, n_head, head_dim)
        v = v.repeat_interleave(repeat_kv, dim=1)

        k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
        v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

        # q: (b, n_head, tgt_s, head_dim)
        scores = torch.bmm(q, k) / math.sqrt(head_dim)
        # if attention_mask.data is not None:
        #     scores = scores + attention_mask.data  # (b, n_head, tgt_s, src_s)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.bmm(scores, v).view(b, n_head, tgt_s, head_dim) # (b, n_head, tgt_s, head_dim)
        
        # Postprocess
        output = output.transpose(1, 2).contiguous().view(b, tgt_s, -1)
        out = F.linear(output, w_out.data)

        out.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # ---- Prepare new k/v to return ----
        # ori_k : (tgt_s, b, n_kv_head, head_dim)
        assert(not compress_cache)
        
        k_new = TorchTensor.create_from_torch(k_new, self)
        v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(out, self), k_new, v_new
    
    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        #mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        #attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def llama_mlp(self, inputs, pos_n, gate, up, down, eps, donate):
        if gate.device.device_type == DeviceType.COMPRESSED:
            pos_n = pos_n.device.decompress(pos_n)
            gate = gate.device.decompress(gate)
            up = gate.device.decompress(up)
            down = gate.device.decompress(down)
        
        b, s, h = inputs.shape
        out = F.rms_norm(inputs.data, (h,), weight=pos_n.data, eps=eps)
        
        up_out = F.linear(out, up.data) ## up
        gate_out = F.linear(out, gate.data) ## gate
        gate_out = F.silu(gate_out)
        out = gate_out * up_out
        out = F.linear(out, down.data) ## down
        
        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        # self.copy_queue = queue.Queue()
        # self.copy_threads = [
        #     threading.Thread(
        #         target=copy_worker_func, args=(self.copy_queue, cuda_id)
        #     ) for _ in range(num_copy_threads)
        # ]
        # for t in self.copy_threads:
        #     t.start()
        self.copy_queue = None

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch_llama(self, config, task, policy):
        num_head, num_key_value_heads, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.num_key_value_heads, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_key_value_heads, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        if self.copy_queue is not None:
            self.copy_queue.put_nowait(args)

    def synchronize(self):
        if self.copy_queue is not None:
            self.copy_queue.join()

    def close_copy_threads(self):
        if self.copy_queue is not None:
            for _ in range(len(self.copy_threads)):
                self.copy_queue.put_nowait(None)
            for t in self.copy_threads:
                t.join()
            self.copy_queue.join()
            self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch_llama(self, config, task, policy):
        num_head, num_key_value_heads, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.num_key_value_heads, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_key_value_heads, hidden_size // num_head)

        # We have to round to a multiple of `num_key_value_heads`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_key_value_heads * num_key_value_heads
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_key_value_heads * num_key_value_heads
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_key_value_heads * num_key_value_heads
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)
            # logger.info(f"Copying src_indices = {src_indices}, src_data.shape = {src_data.shape}, "
            #             f"dst_indices = {dst_indices}, dst_data.shape = {dst_data.shape}")
            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                # logger.info(f"Copying src_data.shape = {src_data.shape}, "
                #             f"dst_data.shape = {dst_data.shape}, size = {size}")
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf) # dst不是pinned memory,所以不用non_blocking
            else:
                dst_data.copy_(src_data)

            queue.task_done()

def sync_general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice],cpu_buf):
    """synchronous copy between two tensors.
    Only supporting copy among pinned tensors on gpu, cpu, or disk.
    """
    src_dev = src.device.device_type
    dst_dev = dst.device.device_type
    if src_dev == DeviceType.DISK or dst_dev == DeviceType.DISK:    
        src_data = map_to_torch_tensor(src, src_indices)
        dst_data = map_to_torch_tensor(dst, dst_indices)
        if (src_dev == DeviceType.CUDA or
            dst_dev == DeviceType.CUDA):
            # Use a pinned cpu buffer as a relay
            size = np.prod(src_data.shape)
            tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
            tmp_cpu_buf.copy_(src_data)
            dst_data.copy_(tmp_cpu_buf, non_blocking=True)
        else:
            dst_data.copy_(src_data)
    elif src_dev == DeviceType.CPU and dst_dev == DeviceType.CUDA and not src_data.is_pinned():
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)
