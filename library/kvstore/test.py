import torch
import os
import numpy as np
from group_kvstore import GroupKVStore
    
def debug():
    store_path = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/tmp_store"
    
    num_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    max_length = 423
    group_size = 8
    
    kvstore = GroupKVStore()
    kvstore.alloc(num_layers, num_attention_heads, num_key_value_heads, head_dim, max_length)
    kvstore.recover_meta(store_path, 0)
    
    nnz = torch.load("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/MagicPIG/examples/snapshots/layer1_nnz_0.pt")
    ind = torch.load("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/MagicPIG/examples/snapshots/layer1_result_0.pt")
    
    kvstore.collect_queried_key_value(0, 1, ind, nnz)
    
    queried_k = kvstore.get_queried_key_cache()
    queried_v = kvstore.get_queried_value_cache()
    print(queried_k.shape)
    


def main():
    # 创建存储目录
    store_path = "kvstore_test_data"
    os.makedirs(store_path, exist_ok=True)

    # 模拟模型参数
    num_layers = 1
    num_attention_heads = 8
    num_key_value_heads = 4
    head_dim = 32
    max_length = 16
    group_size = 8  # 你在C++中默认 group_size 为 8

    # 构造 k, v
    k = torch.randn((num_key_value_heads, max_length, head_dim), dtype=torch.float32)
    v = torch.randn((num_key_value_heads, max_length, head_dim), dtype=torch.float32)

    # 初始化并写入
    kvstore = GroupKVStore()
    kvstore.alloc(num_layers, num_attention_heads, num_key_value_heads, head_dim, 1024)
    kvstore.write_to_storage(store_path, 0, 0, k, v)

    # 构造 ind 和 nnz，模拟访问所有 token（每个 head 全部 group）
    nnz = torch.tensor([max_length for _ in range(num_attention_heads)], dtype=torch.int32)
    ind = torch.tensor([list(range(0,max_length) for _ in range(num_attention_heads))], dtype=torch.int32)
    print(nnz)

    # collect 并获取结果
    kvstore.collect_queried_key_value(0, 0, ind, nnz)
    queried_k = kvstore.get_queried_key_cache().to(float)
    queried_v = kvstore.get_queried_value_cache().to(float)
    
    k = k.to(float)
    v = v.to(float)

    # 用 allclose 做误差判断（浮点数）
    same_k = torch.allclose(k, queried_k[:, :max_length, :], rtol=1e-2, atol=1e-4)
    same_v = torch.allclose(v, queried_v[:, :max_length, :], rtol=1e-2, atol=1e-4)

    print("Key match:", same_k)
    print("Value match:", same_v)

if __name__ == "__main__":
    main()
