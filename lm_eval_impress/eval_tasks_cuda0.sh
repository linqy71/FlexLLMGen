#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# # #############6.7b###################
# MODEL_NAME="facebook/opt-66b"
MODEL_NAME="facebook/opt-6.7b"
MODEL_TYPE="opt"
SEQ=4600

# for TASK_NAME in "openbookqa" "copa" "rte"
for TASK_NAME in "copa"
do
    FEW_SHOT=200
    INPUT_FILE="datasets/${TASK_NAME}_fs${FEW_SHOT}.jsonl"
    for IMP_R in 0.268
    do
        LIMIT_SAMPLES=500
        LOGITS_FILE=logits/impress/6.7b/fs${FEW_SHOT}_${TASK_NAME}_ret_${IMP_R}.jsonl
        rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress/t*
        python3 -u run_lm_eval_harness.py \
          --input-path $INPUT_FILE \
          --output-path $LOGITS_FILE \
          --use-flexgen \
          --model $MODEL_NAME \
          --percent 100 0 100 0 100 0 \
          --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
          --prompt-len $SEQ \
          --gen-len 24 \
          --important-ratio=${IMP_R} \
          --limited-samples=$LIMIT_SAMPLES \
          --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress > ../log/impress/6.7b/fs${FEW_SHOT}_lm_eval_${TASK_NAME}_ret_${IMP_R}_sq.log 2>&1

        
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
              --acc-file acc/6.7b/impress_${TASK_NAME}_ret_${IMP_R}_sq.json
        fi
    done 

done
