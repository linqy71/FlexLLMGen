import torch
import time
import numpy as np
from collections import defaultdict
import random

def generate_test_data():
    """生成测试数据"""
    # 设置随机种子以便复现
    torch.manual_seed(42)
    random.seed(42)
    
    # 生成loaded_chunks: 10个chunk，每个chunk包含k_data和v_data
    loaded_chunks = {}
    for i in range(15):
        k_data = torch.randn(256, 72, 128, dtype=torch.float32)
        v_data = torch.randn(256, 72, 128, dtype=torch.float32)
        loaded_chunks[i] = (k_data, v_data)
    
    # 生成req: 10个items
    # 每个item包含多个(src_offset, dst_offset)对
    req = defaultdict(list)
    
    # 确保不重复的目标偏移
    all_dst_offsets = set()
    
    for chunk_id in range(15):
        # 每个chunk有5-20个偏移对
        num_pairs = random.randint(50, 70)
        for _ in range(num_pairs):
            src_offset = random.randint(0, 255)
            
            # 生成不重复的目标偏移
            while True:
                dst_offset = random.randint(0, 999)
                if dst_offset not in all_dst_offsets:
                    all_dst_offsets.add(dst_offset)
                    break
            
            req[chunk_id].append((src_offset, dst_offset))
    
    # 初始化cpu_buf_k和cpu_buf_v
    cpu_buf_k = torch.zeros(1000, 72, 128, dtype=torch.float32, pin_memory=True)
    cpu_buf_v = torch.zeros(1000, 72, 128, dtype=torch.float32, pin_memory=True)
    
    return loaded_chunks, dict(req), cpu_buf_k, cpu_buf_v

def original_method(loaded_chunks, req, cpu_buf_k, cpu_buf_v):
    """原始方法：逐元素拷贝"""
    for chunk_id, store_ptr in req.items():
        k_data, v_data = loaded_chunks[chunk_id]
        for src_offset, dst_offset in store_ptr:
            cpu_buf_k[dst_offset:dst_offset + 1, :, :].copy_(k_data[src_offset:src_offset + 1, :, :])
            cpu_buf_v[dst_offset:dst_offset + 1, :, :].copy_(v_data[src_offset:src_offset + 1, :, :])
    return cpu_buf_k, cpu_buf_v

def optimized_method(loaded_chunks, req, cpu_buf_k, cpu_buf_v):
    """优化方法：批量索引操作"""
    for chunk_id, store_ptr in req.items():
        k_data, v_data = loaded_chunks[chunk_id]
        if not store_ptr:
            continue
        src_offsets = [item[0] for item in store_ptr]
        dst_offsets = [item[1] for item in store_ptr]

        cpu_buf_k[dst_offsets] = k_data[src_offsets]
        cpu_buf_v[dst_offsets] = v_data[src_offsets]
    
    return cpu_buf_k, cpu_buf_v

def test_performance(num_runs=1000, warmup_runs=10):
    """测试性能"""
    # 生成测试数据
    print("生成测试数据...")
    loaded_chunks, req, cpu_buf_k_orig, cpu_buf_v_orig = generate_test_data()
    
    # 统计req中的总偏移对数量
    total_pairs = sum(len(store_ptr) for store_ptr in req.values())
    print(f"总偏移对数量: {total_pairs}")
    print(f"req中items数量: {len(req)}")
    
    # 预热
    print("\n预热运行...")
    for _ in range(warmup_runs):
        cpu_buf_k_temp = torch.zeros_like(cpu_buf_k_orig)
        cpu_buf_v_temp = torch.zeros_like(cpu_buf_v_orig)
        _ = original_method(loaded_chunks, req, cpu_buf_k_temp, cpu_buf_v_temp)
        
        cpu_buf_k_temp = torch.zeros_like(cpu_buf_k_orig)
        cpu_buf_v_temp = torch.zeros_like(cpu_buf_v_orig)
        _ = optimized_method(loaded_chunks, req, cpu_buf_k_temp, cpu_buf_v_temp)
    
    # 测试原始方法
    print("\n测试原始方法...")
    original_times = []
    for _ in range(num_runs):
        # 每次测试前重置缓冲区
        cpu_buf_k_test = torch.zeros_like(cpu_buf_k_orig)
        cpu_buf_v_test = torch.zeros_like(cpu_buf_v_orig)
        
        start_time = time.perf_counter_ns()
        result_k_orig, result_v_orig = original_method(loaded_chunks, req, cpu_buf_k_orig, cpu_buf_v_orig)
        end_time = time.perf_counter_ns()
        original_times.append((end_time - start_time) / 1e6)  # 转换为毫秒
    
    original_mean = np.mean(original_times)
    original_std = np.std(original_times)
    original_min = np.min(original_times)
    original_max = np.max(original_times)
    
    # 测试优化方法
    print("测试优化方法...")
    optimized_times = []
    for _ in range(num_runs):
        # 每次测试前重置缓冲区
        # cpu_buf_k_test = torch.zeros_like(cpu_buf_k_orig)
        # cpu_buf_v_test = torch.zeros_like(cpu_buf_v_orig)
        
        start_time = time.perf_counter_ns()
        result_k_opt, result_v_opt = optimized_method(loaded_chunks, req, cpu_buf_k_orig, cpu_buf_v_orig)
        end_time = time.perf_counter_ns()
        optimized_times.append((end_time - start_time) / 1e6)  # 转换为毫秒
    
    optimized_mean = np.mean(optimized_times)
    optimized_std = np.std(optimized_times)
    optimized_min = np.min(optimized_times)
    optimized_max = np.max(optimized_times)
    
    # 验证两种方法结果是否一致
    if torch.allclose(result_k_orig, result_k_opt, rtol=1e-6, atol=1e-6) and \
       torch.allclose(result_v_orig, result_v_opt, rtol=1e-6, atol=1e-6):
        print("\n✓ 两种方法结果一致")
    else:
        print("\n✗ 两种方法结果不一致！")
        # 找出差异
        diff_k = torch.sum(torch.abs(result_k_orig - result_k_opt))
        diff_v = torch.sum(torch.abs(result_v_orig - result_v_opt))
        print(f"  k_buf差异总和: {diff_k.item()}")
        print(f"  v_buf差异总和: {diff_v.item()}")
    
    # 输出性能对比
    print("\n" + "="*60)
    print("性能对比 (单位: 毫秒)")
    print("="*60)
    print(f"{'指标':<15} {'原始方法':<15} {'优化方法':<15} {'提升百分比':<15}")
    print("-"*60)
    print(f"{'平均耗时':<15} {original_mean:.6f}ms    {optimized_mean:.6f}ms    {((original_mean - optimized_mean) / original_mean * 100):.2f}%")
    print(f"{'标准差':<15} {original_std:.6f}ms    {optimized_std:.6f}ms    -")
    print(f"{'最小耗时':<15} {original_min:.6f}ms    {optimized_min:.6f}ms    {((original_min - optimized_min) / original_min * 100):.2f}%")
    print(f"{'最大耗时':<15} {original_max:.6f}ms    {optimized_max:.6f}ms    {((original_max - optimized_max) / original_max * 100):.2f}%")
    
    # 额外统计信息
    print("\n" + "="*60)
    print("额外统计信息")
    print("="*60)
    print(f"测试运行次数: {num_runs}")
    print(f"预热运行次数: {warmup_runs}")
    
    # 计算加速比
    speedup = original_mean / optimized_mean
    print(f"平均加速比: {speedup:.2f}x")
    
    return original_times, optimized_times

def test_with_different_sizes():
    """测试不同数据量下的性能"""
    print("测试不同数据量下的性能...")
    print("="*60)
    
    # 测试不同数量的偏移对
    test_cases = [
        (10, 5, 20),   # 少量偏移对
        (20, 10, 30),  # 中等偏移对
        (30, 15, 40),  # 大量偏移对
    ]
    
    for num_items, min_pairs, max_pairs in test_cases:
        print(f"\n测试: {num_items}个items, 每个item {min_pairs}-{max_pairs}个偏移对")
        
        # 生成测试数据
        torch.manual_seed(42)
        loaded_chunks = {}
        for i in range(num_items):
            k_data = torch.randn(256, 72, 128, dtype=torch.float32)
            v_data = torch.randn(256, 72, 128, dtype=torch.float32)
            loaded_chunks[i] = (k_data, v_data)
        
        # 生成req
        req = {}
        all_dst_offsets = set()
        
        for chunk_id in range(num_items):
            req[chunk_id] = []
            num_pairs = random.randint(min_pairs, max_pairs)
            for _ in range(num_pairs):
                src_offset = random.randint(0, 255)
                
                while True:
                    dst_offset = random.randint(0, 999)
                    if dst_offset not in all_dst_offsets:
                        all_dst_offsets.add(dst_offset)
                        break
                
                req[chunk_id].append((src_offset, dst_offset))
        
        total_pairs = sum(len(store_ptr) for store_ptr in req.values())
        print(f"总偏移对数量: {total_pairs}")
        
        # 初始化缓冲区
        cpu_buf_k = torch.zeros(1000, 72, 128, dtype=torch.float32)
        cpu_buf_v = torch.zeros(1000, 72, 128, dtype=torch.float32)
        
        # 预热
        for _ in range(5):
            _ = original_method(loaded_chunks, req, cpu_buf_k.clone(), cpu_buf_v.clone())
            _ = optimized_method(loaded_chunks, req, cpu_buf_k.clone(), cpu_buf_v.clone())
        
        # 测试原始方法
        original_times = []
        for _ in range(100):
            start_time = time.perf_counter_ns()
            _ = original_method(loaded_chunks, req, cpu_buf_k.clone(), cpu_buf_v.clone())
            end_time = time.perf_counter_ns()
            original_times.append((end_time - start_time) / 1e6)
        
        # 测试优化方法
        optimized_times = []
        for _ in range(100):
            start_time = time.perf_counter_ns()
            _ = optimized_method(loaded_chunks, req, cpu_buf_k.clone(), cpu_buf_v.clone())
            end_time = time.perf_counter_ns()
            optimized_times.append((end_time - start_time) / 1e6)
        
        original_mean = np.mean(original_times)
        optimized_mean = np.mean(optimized_times)
        speedup = original_mean / optimized_mean
        
        print(f"原始方法: {original_mean:.4f}ms")
        print(f"优化方法: {optimized_mean:.4f}ms")
        print(f"加速比: {speedup:.2f}x")
        print(f"提升百分比: {((original_mean - optimized_mean) / original_mean * 100):.2f}%")

if __name__ == "__main__":
    print("KV Cache拷贝性能测试")
    print("="*60)
    
    # 运行主测试
    original_times, optimized_times = test_performance(num_runs=1000, warmup_runs=10)
    
    # 测试不同数据量
    # test_with_different_sizes()
    