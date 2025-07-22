#!/usr/bin/env python3
"""
调试单个评估样本的处理过程
检查 FlexGen logits 在实际评估中的表现
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer

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

def debug_single_sample():
    """调试单个 HellaSwag 样本"""
    print("=" * 80)
    print("🔍 调试单个 HellaSwag 样本的处理过程")
    print("=" * 80)
    
    # 从 input.jsonl 中读取一个样本
    try:
        with open("input.jsonl", "r") as f:
            first_line = f.readline().strip()
            sample_request = json.loads(first_line)
    except FileNotFoundError:
        print("❌ input.jsonl 文件不存在，请先生成")
        return
    
    print(f"📋 样本请求:")
    print(f"   prompt: '{sample_request['prompt'][:100]}...'")
    
    # 初始化模型和 tokenizer
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model, config = init_flexgen_model(model_name)
        print("✅ 模型初始化成功")
        
        # 处理样本
        prompt = sample_request['prompt']
        print(f"\n📝 完整 prompt: '{prompt}'")
        
        # 分词
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
        print(f"🔢 Token IDs 形状: {input_ids.shape}")
        print(f"🔢 Token IDs: {input_ids.tolist()[0][:20]}...")  # 显示前20个
        
        # 解码回文本验证
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"🔄 解码验证: '{decoded_text[:100]}...'")
        
        # 使用 FlexGen 计算 logits
        print(f"\n🧮 计算 FlexGen logits...")
        input_ids_np = input_ids.cpu().numpy()
        
        try:
            logits = model.get_logits(input_ids_np)
            print(f"✅ Logits 计算成功")
            print(f"📊 Logits 形状: {logits.shape}")
            print(f"📈 Logits 范围: [{logits.min():.3f}, {logits.max():.3f}]")
            
            # 计算 log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            print(f"📊 Log probabilities 形状: {log_probs.shape}")
            print(f"📈 Log probs 范围: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
            
            # 分析每个位置的预测
            seq_len = input_ids.shape[1]
            print(f"\n🎯 分析前5个位置的预测:")
            
            for pos in range(min(5, seq_len-1)):  # 不包括最后一个位置
                current_token_id = input_ids[0, pos].item()
                next_token_id = input_ids[0, pos + 1].item()
                
                current_token = tokenizer.decode([current_token_id])
                next_token = tokenizer.decode([next_token_id])
                
                # 当前位置对下一个 token 的预测 logit
                next_token_logit = logits[0, pos, next_token_id].item()
                next_token_log_prob = log_probs[0, pos, next_token_id].item()
                
                # 当前位置的 top prediction
                top_logit, top_token_id = torch.max(logits[0, pos], dim=0)
                top_token = tokenizer.decode([top_token_id.item()])
                
                print(f"   位置 {pos}: '{current_token}' -> '{next_token}'")
                print(f"      实际 log_prob: {next_token_log_prob:.3f}")
                print(f"      Top 预测: '{top_token}' (logit: {top_logit:.3f})")
                print(f"      匹配: {'✅' if top_token_id.item() == next_token_id else '❌'}")
            
            # 检查最后几个位置（通常是选择题的关键部分）
            print(f"\n🎯 分析最后5个位置的预测:")
            for pos in range(max(0, seq_len-5), seq_len-1):
                current_token_id = input_ids[0, pos].item()
                next_token_id = input_ids[0, pos + 1].item()
                
                current_token = tokenizer.decode([current_token_id])
                next_token = tokenizer.decode([next_token_id])
                
                next_token_logit = logits[0, pos, next_token_id].item()
                next_token_log_prob = log_probs[0, pos, next_token_id].item()
                
                top_logit, top_token_id = torch.max(logits[0, pos], dim=0)
                top_token = tokenizer.decode([top_token_id.item()])
                
                print(f"   位置 {pos}: '{current_token}' -> '{next_token}'")
                print(f"      实际 log_prob: {next_token_log_prob:.3f}")
                print(f"      Top 预测: '{top_token}' (logit: {top_logit:.3f})")
                print(f"      匹配: {'✅' if top_token_id.item() == next_token_id else '❌'}")
            
            # 模拟 run_lm_eval_harness.py 中的处理过程
            print(f"\n🔄 模拟 run_lm_eval_harness.py 处理过程:")
            
            # 计算 token logprobs (跳过第一个)
            gold_indices = input_ids[:, 1:]  # skip first
            logprobs_result = torch.gather(log_probs, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0)
            
            print(f"📊 Gold token logprobs 数量: {len(logprobs_result)}")
            print(f"📊 前10个 logprobs: {logprobs_result[:10].tolist()}")
            print(f"📊 Logprobs 均值: {logprobs_result.mean():.3f}")
            print(f"📊 Logprobs 范围: [{logprobs_result.min():.3f}, {logprobs_result.max():.3f}]")
            
            # 检查是否有异常低的 logprobs
            very_low_logprobs = logprobs_result < -10
            if very_low_logprobs.any():
                print(f"⚠️  发现 {very_low_logprobs.sum()} 个异常低的 logprobs (< -10)")
                
            # 模拟 top logprobs 计算
            values, indices = log_probs.squeeze(0).topk(dim=-1, k=1)
            print(f"📊 Top predictions 形状: {values.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Logits 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_huggingface():
    """与 HuggingFace 模型对比"""
    print("\n" + "=" * 80)
    print("🔄 与 HuggingFace 模型对比")
    print("=" * 80)
    
    # 暂时跳过 HF 对比，专注于 FlexGen 调试
    print("📝 TODO: 实现 HuggingFace 对比（需要下载模型）")

if __name__ == "__main__":
    print("🔍 FlexGen 评估样本调试工具")
    
    if debug_single_sample():
        print("\n✅ 样本调试完成")
        compare_with_huggingface()
    else:
        print("\n❌ 样本调试失败")
    
    print("\n🎯 调试完成!")
