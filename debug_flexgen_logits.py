#!/usr/bin/env python3
"""
调试 FlexGen logits 输出的脚本
检查 get_logits 方法是否正常工作
"""

import torch
import numpy as np
from transformers import AutoTokenizer

# 尝试导入 FlexGen
try:
    from flexllmgen.flex_opt import OptLM, Policy
    from flexllmgen.opt_config import get_opt_config
    from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
    from flexllmgen.utils import ExecutionEnv, GB, Task
    from flexllmgen.compression import CompressionConfig
    FLEXGEN_AVAILABLE = True
    print("✅ FlexGen 导入成功")
except ImportError as e:
    print(f"❌ FlexGen 导入失败: {e}")
    FLEXGEN_AVAILABLE = False
    exit(1)

def init_flexgen_model(model_name="facebook/opt-350m"):
    """初始化 FlexGen 模型"""
    print(f"🚀 初始化 FlexGen 模型: {model_name}")
    
    # 设置执行环境
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk("~/flexgen_offload_dir")
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                      mixed=TorchMixedDevice([gpu, cpu, disk]))
    
    # 设置策略
    policy = Policy(
        1, 1,  # gpu_batch_size=1, num_gpu_batches=1
        100, 0, 100, 0, 100, 0,  # percent: all GPU
        True, False, True, False, 1.0,
        False, CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        False, CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False)
    )
    
    # 初始化模型
    opt_config = get_opt_config(model_name)
    model = OptLM(opt_config, env, "~/opt_weights", policy)
    
    return model, opt_config

def test_get_logits():
    """测试 get_logits 方法"""
    print("=" * 60)
    print("🧪 测试 FlexGen get_logits 方法")
    print("=" * 60)
    
    # 使用较小的模型进行测试
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model, config = init_flexgen_model(model_name)
        print(f"✅ 模型初始化成功")
        print(f"📋 模型配置: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
        
        # 测试简单输入
        test_text = "The weather is"
        print(f"📝 测试文本: '{test_text}'")
        
        # 分词
        input_ids = tokenizer(test_text, return_tensors="pt").input_ids
        print(f"🔢 Token IDs: {input_ids.tolist()}")
        print(f"📏 序列长度: {input_ids.shape[1]}")
        
        # 转换为 numpy
        input_ids_np = input_ids.cpu().numpy()
        
        # 检查 get_logits 方法是否存在
        if hasattr(model, 'get_logits'):
            print("✅ get_logits 方法存在")
            
            # 调用 get_logits
            print("🔄 调用 get_logits...")
            try:
                logits = model.get_logits(input_ids_np)
                print(f"✅ get_logits 调用成功")
                print(f"📊 Logits 形状: {logits.shape}")
                print(f"📈 Logits 范围: [{logits.min():.3f}, {logits.max():.3f}]")
                print(f"📊 Logits 均值: {logits.mean():.3f}")
                
                # 检查是否为合理的 logits
                if logits.shape[-1] == config.vocab_size:
                    print(f"✅ Vocab size 匹配: {logits.shape[-1]} == {config.vocab_size}")
                else:
                    print(f"❌ Vocab size 不匹配: {logits.shape[-1]} != {config.vocab_size}")
                
                # 计算概率分布
                probs = torch.softmax(logits, dim=-1)
                print(f"📊 概率分布范围: [{probs.min():.6f}, {probs.max():.6f}]")
                
                # 检查最后一个位置的 top-5 预测
                last_pos_logits = logits[0, -1, :]  # [vocab_size]
                top5_values, top5_indices = torch.topk(last_pos_logits, 5)
                
                print(f"🎯 最后位置 top-5 预测:")
                for i, (val, idx) in enumerate(zip(top5_values, top5_indices)):
                    token = tokenizer.decode([idx.item()])
                    print(f"   {i+1}. '{token}' (logit={val:.3f})")
                
                return True
                
            except Exception as e:
                print(f"❌ get_logits 调用失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ get_logits 方法不存在")
            return False
            
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_generate():
    """对比 get_logits 和 generate 方法的结果"""
    print("\n" + "=" * 60)
    print("🔄 对比 get_logits 与 generate 方法")
    print("=" * 60)
    
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model, config = init_flexgen_model(model_name)
        
        test_text = "The weather is"
        input_ids = tokenizer(test_text, return_tensors="pt").input_ids
        input_list = input_ids.tolist()
        
        print(f"📝 测试文本: '{test_text}'")
        print(f"🔢 Input list: {input_list}")
        
        # 使用 generate 方法
        print("\n🎯 使用 generate 方法:")
        try:
            output_ids = model.generate(
                inputs=input_list,
                max_new_tokens=3,
                do_sample=False,
                temperature=1.0,
                verbose=1
            )
            print(f"✅ Generate 输出: {output_ids}")
            
            # 解码生成的文本
            if output_ids and len(output_ids) > 0:
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(f"📝 生成文本: '{generated_text}'")
        except Exception as e:
            print(f"❌ Generate 失败: {e}")
        
        # 使用 get_logits 方法
        print("\n🎯 使用 get_logits 方法:")
        try:
            logits = model.get_logits(input_ids.cpu().numpy())
            
            # 获取下一个 token 的预测
            next_token_logits = logits[0, -1, :]  # 最后一个位置的 logits
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = tokenizer.decode([next_token_id])
            
            print(f"✅ 预测下一个 token: '{next_token}' (ID: {next_token_id})")
            
        except Exception as e:
            print(f"❌ get_logits 失败: {e}")
            
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")

if __name__ == "__main__":
    print("🧪 FlexGen Logits 调试工具")
    
    if test_get_logits():
        print("\n🎉 基础测试通过，继续详细对比...")
        compare_with_generate()
    else:
        print("\n💥 基础测试失败，请检查 get_logits 实现")
    
    print("\n🎯 调试完成!")
