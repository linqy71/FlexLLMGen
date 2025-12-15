#!/bin/bash


# MODEL_NAME="facebook/opt-66b"
MODEL_NAME="facebook/opt-6.7b"
MODEL_TYPE="opt"
# TASK_NAME="copa"
# TASK_NAME="piqa"
# TASK_NAME="rte"
# TASK_NAME="openbookqa"

LIMIT_SAMPLES=50

for TASK_NAME in "rte"
do
INPUT_FILE="${TASK_NAME}.jsonl"

echo " 配置信息:"
echo "   模型: $MODEL_NAME"
echo "   任务: $TASK_NAME"
echo "   测试样本: $LIMIT_SAMPLES 条"

# 生成小规模测试数据
echo "生成测试数据..."
python3 generate_task_data.py \
  --output-file $INPUT_FILE \
  --task-name $TASK_NAME \
  --num-fewshot 10 \
  --limit $LIMIT_SAMPLES

echo " 测试数据生成完成: $(wc -l < $INPUT_FILE) 条"
done