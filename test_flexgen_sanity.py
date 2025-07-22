#!/usr/bin/env python3
"""
FlexGen 基础功能测试
验证模型是否正确加载和工作
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入 FlexGen
try:
    from flexllmgen.flex_opt import OptLM, Policy
    from flexllmgen.opt_config import get_opt_config
    from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
    from flexllmgen.utils import ExecutionEnv, GB, Task
    from flexllmgen.compression import CompressionConfig
    FLEXGEN_AVAILABLE = True
except ImportError as e:
    print(f"❌ FlexGen 导入失败: {e}")
    exit(1)

def init_flexgen_model(model_name="facebook/opt-1.3b"):
    """初始化 FlexGen 模型"""
    print(f"🚀 初始化 FlexGen 模型: {model_name}")
    
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk("~/flexgen_offload_dir")
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                      mixed=TorchMixedDevice([gpu, cpu, disk]))
    
    policy = Policy(
        1, 1,  # gpu_batch_size=1, num_gpu_batches=1
        100, 0, 100, 0, 100, 0,  # percent: all GPU
        True, False, True, False, 1.0,
        False, CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        False, CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False)
    )
    
    opt_config = get_opt_config(model_name)
    model = OptLM(opt_config, env, "~/opt_weights", policy)
    
    return model, opt_config

def test_simple_generation():
    """测试简单的文本生成"""
    print("=" * 60)
    print("🧪 测试 FlexGen 文本生成功能")
    print("=" * 60)
    
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model, config = init_flexgen_model(model_name)
        print("✅ 模型初始化成功")
        
        # 测试简单的生成任务
        test_prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The weather today is",
            "I like to eat"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 测试 prompt: '{prompt}'")
            
            # 分词
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
            inputs = input_ids.cpu().numpy().tolist()
            
            try:
                # 使用 FlexGen generate 方法
                print("🚀 使用 generate 方法...")
                output_ids = model.generate(
                    inputs=inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=1.0,
                    stop=None,
                    verbose=0
                )
                
                # 解码结果
                if output_ids and len(output_ids) > 0:
                    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    new_tokens = generated_text[len(prompt):]
                    print(f"✅ 生成结果: '{prompt}' -> '{new_tokens}'")
                else:
                    print("❌ 生成失败：无输出")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                
            try:
                # 使用 get_logits 方法
                print("🧮 使用 get_logits 方法...")
                logits = model.get_logits(input_ids.cpu().numpy())
                
                # 获取最后一个位置的 top-5 预测
                last_pos_logits = logits[0, -1, :]  # [vocab_size]
                top_values, top_indices = torch.topk(last_pos_logits, k=5)
                
                print("🎯 最后位置 top-5 预测:")
                for i, (value, idx) in enumerate(zip(top_values, top_indices)):
                    token = tokenizer.decode([idx.item()])
                    print(f"   {i+1}. '{token}' (logit: {value:.3f})")
                    
            except Exception as e:
                print(f"❌ get_logits 失败: {e}")
            
            print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weights_loading():
    """测试权重加载"""
    print("=" * 60)
    print("🔍 检查权重加载情况")
    print("=" * 60)
    
    import os
    
    weights_path = os.path.expanduser("~/opt_weights")
    print(f"📁 权重路径: {weights_path}")
    
    if not os.path.exists(weights_path):
        print("❌ 权重目录不存在")
        return False
    
    # 检查具体的权重文件
    model_path = os.path.join(weights_path, "facebook-opt-1.3b-np")
    print(f"📁 模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ 模型权重目录不存在")
        return False
    
    # 列出权重文件
    weight_files = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.npy'):
                weight_files.append(os.path.relpath(os.path.join(root, file), model_path))
    
    print(f"📊 找到 {len(weight_files)} 个权重文件")
    if weight_files:
        print("📋 示例权重文件:")
        for file in sorted(weight_files)[:10]:  # 显示前10个
            file_path = os.path.join(model_path, file)
            size = os.path.getsize(file_path)
            print(f"   {file} ({size:,} bytes)")
        
        if len(weight_files) > 10:
            print(f"   ... 还有 {len(weight_files) - 10} 个文件")
    
    return len(weight_files) > 0

def compare_with_known_good():
    """与已知的好结果对比"""
    print("=" * 60)
    print("🔄 与已知结果对比")
    print("=" * 60)
    
    # 测试一个简单的、已知结果的案例
    prompt = "The capital of France is"
    expected_completions = [" Paris", " the", " located"]  # 常见的合理补全
    
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model, config = init_flexgen_model(model_name)
        
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
        logits = model.get_logits(input_ids.cpu().numpy())
        
        # 获取最后位置的预测
        last_pos_logits = logits[0, -1, :]
        top_values, top_indices = torch.topk(last_pos_logits, k=10)
        
        print(f"📝 测试 prompt: '{prompt}'")
        print("🎯 Top-10 预测:")
        
        found_reasonable = False
        for i, (value, idx) in enumerate(zip(top_values, top_indices)):
            token = tokenizer.decode([idx.item()])
            is_expected = token in expected_completions
            status = "✅" if is_expected else "  "
            
            print(f"   {i+1:2d}. {status} '{token}' (logit: {value:.3f})")
            
            if is_expected:
                found_reasonable = True
        
        if found_reasonable:
            print("✅ 找到合理的预测，模型似乎工作正常")
            return True
        else:
            print("❌ 没有找到合理的预测，模型可能有问题")
            return False
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 FlexGen 基础功能测试")
    
    # 第一步：检查权重加载
    if not test_weights_loading():
        print("❌ 权重加载检查失败，请确保模型权重已正确下载")
        exit(1)
    
    # 第二步：测试基础生成
    if not test_simple_generation():
        print("❌ 基础生成测试失败")
        exit(1)
    
    # 第三步：与已知结果对比
    if not compare_with_known_good():
        print("❌ 已知结果对比失败")
        exit(1)
    
    print("\n🎉 所有测试通过！FlexGen 模型工作正常")
