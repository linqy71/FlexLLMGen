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

from flexllmgen.opt_config import  get_opt_config
from flexllmgen.pytorch_backend_lsh import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.timer import timers
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

from flexllmgen.lsh_server import LSHServer
fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preview_data(label, data, layer_id, max_items=5):
    if data is None:
        print(f"[Layer {layer_id}] {label}: None")
        return
    if isinstance(data, torch.Tensor):
        flat = data.detach().cpu().reshape(-1)
    else:
        flat = np.asarray(data).reshape(-1)
    sample = flat[:max_items]
    print(f"[Layer {layer_id}] {label} shape: {getattr(data, 'shape', 'N/A')} sample: {sample}")


config = get_opt_config("opt-66b")
num_hidden_layers = config.num_hidden_layers
kv_store_path = "/ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering"

kv_store_path = os.path.join(kv_store_path, "kv_store")
kv_server = LSHServer(config, num_hidden_layers, kv_store_path, K=10, L=100, batch_size=1, max_length=8192, device='cuda:6')


kv_server.recover_kv_store_meta(prefix_id=1)
kv_server.recover_queried_result(prefix_id=1)

# print("\n--- Previewing Recovered Queried Results ---")
# for layer_id in range(num_hidden_layers):
#     # self.query_results is a list of tuples (nnz, results)
#     nnz_data, results_data = kv_server.query_results[layer_id]
#     preview_data(f"Recovered NNZ", nnz_data, layer_id)
#     preview_data(f"Recovered Results", results_data, layer_id)
# print("--- End of Preview ---\n")

timers("io part test").reset()

kv_server.persisted = True
for layer_id in range(num_hidden_layers):
    kv_server.nnz.copy_(kv_server.query_results[layer_id][0])
    kv_server.results_lsh_cpu.copy_(kv_server.query_results[layer_id][1])

    preview_data("NNZ (before load_kv)", kv_server.nnz, layer_id)
    # preview_data("Results LSH CPU (before load_kv)", kv_server.results_lsh_cpu, layer_id)


    k_cache_data, v_cache_data = kv_server.load_kv(0, layer_id, 1)

    preview_data("K cache", k_cache_data, layer_id)
    preview_data("V cache", v_cache_data, layer_id)
    

print("IO part test sum time:", timers("io part test").elapsed("sum"))
print("IO part test avg time:", timers("io part test").elapsed("average"))
print("IO part test time:", timers("io part test").costs)