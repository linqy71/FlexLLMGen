#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
###### RTE ########

FEW_SHOT=40
SEQ=4200
LIMIT_SAMPLES=500

##############66b###################
MODEL_NAME="facebook/opt-66b"
MODEL_TYPE="opt"
SIZE="66b"

# for TASK_NAME in "openbookqa" "copa" "rte"
for TASK_NAME in "rte"
do
INPUT_FILE="datasets/${TASK_NAME}_fs${FEW_SHOT}.jsonl"
K=6
for L in 35 32 38 34
do
LOGITS_FILE=logits/${SIZE}/fs${FEW_SHOT}_${TASK_NAME}_K${K}L${L}.jsonl
python3 -u run_lm_eval_harness.py \
  --input-path $INPUT_FILE \
  --output-path ${LOGITS_FILE} \
  --use-flexgen \
  --model $MODEL_NAME \
  --percent 40 60 100 0 100 0 \
  --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
  --prompt-len $SEQ \
  --gen-len 8 \
  --strategy seq \
  --overlap=True \
  --save-res=False \
  --K=$K \
  --L=$L \
  --offload-dir /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/flexllmgen_offload_dir > ../log/${SIZE}/fs${FEW_SHOT}_lm_eval_${TASK_NAME}_K${K}L${L}.log 2>&1

LIMIT_SAMPLES=500
if [ -f "$LOGITS_FILE" ]; then
    echo " FlexGen 推理完成"【
    echo "   结果数: $(wc -l < $LOGITS_FILE) 条"
    
    python3 evaluate_task_result.py \
      --result-file $LOGITS_FILE \
      --task-name $TASK_NAME \
      --model-type $MODEL_TYPE \
      --num-fewshot $FEW_SHOT \
      --limit $LIMIT_SAMPLES \
      --seq $SEQ \
      --acc-file acc/${SIZE}/${TASK_NAME}_K${K}L${L}.json
fi

echo " 66b $TASK_NAME K $K L $L done"

done 

done

