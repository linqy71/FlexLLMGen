#!/bin/bash

# FlexGen 完整评估流程脚本
# 第一步：生成 logprobs，第二步：计算最终评估结果

set -e  # 遇到错误立即退出

echo "🧪 FlexGen 完整评估流程"
echo "=" * 60

# 配置参数
MODEL_NAME="facebook/opt-1.3b"
MODEL_TYPE="opt"
TASK_NAME="hellaswag"
INPUT_FILE="input.jsonl"
RESULT_FILE="opt-1.3b-flexgen-results.jsonl"

echo "📋 配置信息:"
echo "   模型: $MODEL_NAME"
echo "   类型: $MODEL_TYPE"
echo "   任务: $TASK_NAME"
echo "   输入文件: $INPUT_FILE"
echo "   结果文件: $RESULT_FILE"

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件 $INPUT_FILE 不存在"
    echo "请先运行 generate_task_data.py 生成输入文件"
    exit 1
fi

echo "📁 输入文件检查:"
echo "   大小: $(wc -c < $INPUT_FILE) 字节"
echo "   请求数: $(wc -l < $INPUT_FILE) 条"

# 第一步：运行 FlexGen 推理
echo ""
echo "🚀 第一步：FlexGen 推理生成 logprobs..."
echo "命令: python3 -u run_lm_eval_harness.py --input-path $INPUT_FILE --output-path $RESULT_FILE --model-name $MODEL_NAME --model-type $MODEL_TYPE --use-flexgen"

python3 -u run_lm_eval_harness.py \
  --input-path "$INPUT_FILE" \
  --output-path "$RESULT_FILE" \
  --model-name "$MODEL_NAME" \
  --model-type "$MODEL_TYPE" \
  --use-flexgen

# 检查第一步结果
if [ ! -f "$RESULT_FILE" ]; then
    echo "❌ 第一步失败: 结果文件 $RESULT_FILE 未生成"
    exit 1
fi

echo "✅ 第一步完成!"
echo "📊 结果文件: $RESULT_FILE"
echo "   大小: $(wc -c < $RESULT_FILE) 字节"
echo "   结果数: $(wc -l < $RESULT_FILE) 条"

# 第二步：评估最终结果
echo ""
echo "📈 第二步：计算最终评估指标..."
echo "命令: python3 evaluate_task_result.py --result-file $RESULT_FILE --task-name $TASK_NAME --model-type $MODEL_TYPE"

python3 evaluate_task_result.py \
  --result-file "$RESULT_FILE" \
  --task-name "$TASK_NAME" \
  --model-type "$MODEL_TYPE"

echo ""
echo "🎉 FlexGen 完整评估流程完成!"
echo "📋 生成的文件:"
echo "   输入: $INPUT_FILE"
echo "   logprobs 结果: $RESULT_FILE"
echo "   最终评估指标: 已显示在上方"
