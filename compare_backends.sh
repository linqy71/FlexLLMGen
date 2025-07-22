#!/bin/bash

# 对比测试：HuggingFace vs FlexGen
# 用于诊断 FlexGen 结果异常的问题

set -e

# Clean up previous cache to ensure a fresh run
rm -rf cache_hf

echo "🔍 FlexGen vs HuggingFace 对比测试"
echo "=" * 60

# 配置参数
MODEL_NAME="facebook/opt-1.3b"
# MODEL_NAME="facebook/opt-6.7b"
MODEL_TYPE="opt"
# TASK_NAME="copa"
# TASK_NAME="piqa"
# TASK_NAME="rte"
TASK_NAME="openbookqa"
INPUT_FILE="input.jsonl"

# 限制测试样本数量进行快速对比
LIMIT_SAMPLES=100

echo " 配置信息:"
echo "   模型: $MODEL_NAME"
echo "   任务: $TASK_NAME"
echo "   测试样本: $LIMIT_SAMPLES 条"

# 生成小规模测试数据
echo "生成测试数据..."
python3 generate_task_data.py \
  --output-file test_small.jsonl \
  --task-name $TASK_NAME \
  --num-fewshot 0 \
  --limit $LIMIT_SAMPLES

echo " 测试数据生成完成: $(wc -l < test_small.jsonl) 条"

# # 测试 1：HuggingFace 后端
# echo ""
# echo " 测试 1：HuggingFace 后端"
# echo "命令: python3 -u run_lm_eval_harness.py --input-path test_small.jsonl --output-path hf_results.jsonl --model-name $MODEL_NAME --model-type $MODEL_TYPE"

# python3 -u run_lm_eval_harness.py \
#   --input-path test_small.jsonl \
#   --output-path hf_results.jsonl \
#   --cache-dir "cache_hf" \
#   --model-name "$MODEL_NAME" \
#   --model-type "$MODEL_TYPE"

# if [ -f "hf_results.jsonl" ]; then
#     echo " HuggingFace 推理完成"
#     echo "   结果数: $(wc -l < hf_results.jsonl) 条"
    
#     echo "HuggingFace 评估结果:"
#     python3 evaluate_task_result.py \
#       --result-file hf_results.jsonl \
#       --task-name $TASK_NAME \
#       --model-type $MODEL_TYPE \
#       --limit $LIMIT_SAMPLES
# else
#     echo " HuggingFace 推理失败"
# fi

# 测试 2：FlexGen 后端
echo ""
echo "🔥 测试 2：FlexGen 后端"
echo "命令: python3 -u run_lm_eval_harness.py --input-path test_small.jsonl --output-path flexgen_results.jsonl --model-name $MODEL_NAME --model-type $MODEL_TYPE --use-flexgen"

python3 -u run_lm_eval_harness.py \
  --input-path test_small.jsonl \
  --output-path flexgen_results.jsonl \
  --cache-dir "cache_flex" \
  --model-name "$MODEL_NAME" \
  --model-type "$MODEL_TYPE" \
  --use-flexgen

if [ -f "flexgen_results.jsonl" ]; then
    echo " FlexGen 推理完成"【
    echo "   结果数: $(wc -l < flexgen_results.jsonl) 条"
    
    echo " FlexGen 评估结果:"
    python3 evaluate_task_result.py \
      --result-file flexgen_results.jsonl \
      --task-name $TASK_NAME \
      --model-type $MODEL_TYPE \
      --limit $LIMIT_SAMPLES



    echo "FlexGen 第一行："
    head -1 flexgen_results.jsonl | python3 -c "
import json, sys
data = json.load(sys.stdin)
logprobs = data['result']['choices'][0]['logprobs']['token_logprobs']
print(f'前5个logprobs: {logprobs[1:6]}')
print(f'平均logprob: {sum([x for x in logprobs[1:] if x is not None]) / len([x for x in logprobs[1:] if x is not None]):.4f}')
"

