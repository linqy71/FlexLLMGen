import os
import time
import torch
import numpy as np
import direct_io

def test_file_read(filename):
    """测试直接IO读取与标准读取的对比"""
    if not os.path.exists(filename):
        print(f"错误: 文件 {filename} 不存在!")
        return

    # 使用numpy标准方式读取文件获取信息
    print(f"尝试标准方式读取文件: {filename}")
    start_time = time.time()
    try:
        numpy_array = np.load(filename)
        np_time = time.time() - start_time
        print(f"NumPy读取成功, 耗时: {np_time:.6f} 秒")
        print(f"数组形状: {numpy_array.shape}")
        print(f"数据类型: {numpy_array.dtype}")
        print(f"数据范围: {numpy_array.min()} 到 {numpy_array.max()}")
        print(f"前5个元素: {numpy_array.flatten()[:5]}")
    except Exception as e:
        print(f"NumPy读取失败: {e}")
        return

    # 使用direct_io读取
    print("\n尝试使用direct_io读取同一文件...")
    try:
        # 转换NumPy数据类型到PyTorch数据类型
        dtype_map = {
            np.dtype('float32'): torch.float32,
            np.dtype('float64'): torch.float64,
            np.dtype('float16'): torch.float16,
            np.dtype('int32'): torch.int32,
            np.dtype('int64'): torch.int64,
            np.dtype('uint8'): torch.uint8,
            np.dtype('bool'): torch.bool,
        }
        torch_dtype = dtype_map.get(numpy_array.dtype, torch.float32)
        
        # 使用direct_io读取
        start_time = time.time()
        tensor = direct_io.read_file_direct(filename, list(numpy_array.shape), torch_dtype)
        dio_time = time.time() - start_time
        
        print(f"Direct IO读取成功, 耗时: {dio_time:.6f} 秒")
        print(f"张量形状: {tensor.shape}")
        print(f"张量类型: {tensor.dtype}")
        print(f"数据范围: {tensor.min().item()} 到 {tensor.max().item()}")
        print(f"前5个元素: {tensor.flatten()[:5]}")
        
        # 比较数据是否一致
        numpy_from_tensor = tensor.numpy()
        is_close = np.allclose(numpy_array, numpy_from_tensor, rtol=1e-5, atol=1e-5)
        print(f"\n数据一致性检查: {'通过' if is_close else '失败'}")
        
        # 计算加速比
        speedup = np_time / dio_time
        print(f"Direct IO比标准IO快 {speedup:.2f}x")
        
    except Exception as e:
        print(f"Direct IO读取失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试指定文件
    filename = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/flexllmgen_offload_dir/t_4711"
    test_file_read(filename)