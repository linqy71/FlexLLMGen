#!/bin/bash


# # #############6.7b###################
# MODEL_NAME="facebook/opt-66b"
MODEL_NAME="facebook/opt-30b"
MODEL_TYPE="opt"
SEQ=4600

# for TASK_NAME in "openbookqa" "copa" "rte"
for TASK_NAME in "copa"
do
    FEW_SHOT=200
    INPUT_FILE="datasets/${TASK_NAME}_fs${FEW_SHOT}.jsonl"
    for IMP_R in 0.40 0.30 0.24
    do
        LIMIT_SAMPLES=500
        LOGITS_FILE=logits/impress/30b/fs${FEW_SHOT}_${TASK_NAME}_ret_${IMP_R}.jsonl
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
          --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress > ../log/impress/30b/fs${FEW_SHOT}_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

        
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
              --acc-file acc/30b/impress_${TASK_NAME}_ret_${IMP_R}.json
        fi
    done 

done

for TASK_NAME in "rte"
do
    FEW_SHOT=40
    INPUT_FILE="datasets/${TASK_NAME}_fs40.jsonl"
    for IMP_R in 0.39 0.35 0.27 0.22
    do
        LIMIT_SAMPLES=500
        LOGITS_FILE=logits/impress/30b/fs${FEW_SHOT}_${TASK_NAME}_ret_${IMP_R}.jsonl
        rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress/t*
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
          --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/30b/fs${FEW_SHOT}_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

        
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
              --acc-file acc/30b/impress_${TASK_NAME}_ret_${IMP_R}.json
        fi
    done 

done

############30b###################
# # MODEL_NAME="facebook/opt-66b"
# MODEL_NAME="facebook/opt-30b"
# MODEL_TYPE="opt"

# for TASK_NAME in "rte"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs15.jsonl"
# for IMP_R in 0.34 0.36 0.44 0.46
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/impress/30b/fs15_${TASK_NAME}_ret_${IMP_R}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 100 0 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1700 \
#   --gen-len 24 \
#   --important-ratio=${IMP_R} \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/30b/fs15_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

# done 

# done

# # #############66b###################
# MODEL_NAME="facebook/opt-66b"
# # MODEL_NAME="facebook/opt-6.7b"
# MODEL_TYPE="opt"

# # for TASK_NAME in "openbookqa" "copa" "rte"
# for TASK_NAME in "rte"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs15.jsonl"
# for IMP_R in 0.3 0.35 0.4 0.45 0.5
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/impress/66b/fs15_${TASK_NAME}_ret_${IMP_R}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 40 60 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1700 \
#   --gen-len 24 \
#   --important-ratio=${IMP_R} \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/66b/fs15_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

# done 

# done


##############################
# # #############6.7b###################
# # MODEL_NAME="facebook/opt-66b"
# MODEL_NAME="facebook/opt-6.7b"
# MODEL_TYPE="opt"

# # for TASK_NAME in "openbookqa" "copa" "rte"
# for TASK_NAME in "copa"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs20.jsonl"
# for IMP_R in 0.31 0.32 0.39 0.41 0.48
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/impress/6.7b/fs20_${TASK_NAME}_ret_${IMP_R}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 100 0 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1700 \
#   --gen-len 24 \
#   --important-ratio=${IMP_R} \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/6.7b/fs20_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

# done 

# done

# ############30b###################
# # MODEL_NAME="facebook/opt-66b"
# MODEL_NAME="facebook/opt-30b"
# MODEL_TYPE="opt"

# for TASK_NAME in "copa"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs20.jsonl"
# for IMP_R in 0.3 0.35 0.4 0.45 0.5
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/impress/30b/fs20_${TASK_NAME}_ret_${IMP_R}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 100 0 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1700 \
#   --gen-len 24 \
#   --important-ratio=${IMP_R} \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/30b/fs20_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

# done 

# done

# # #############66b###################
# MODEL_NAME="facebook/opt-66b"
# # MODEL_NAME="facebook/opt-6.7b"
# MODEL_TYPE="opt"

# # for TASK_NAME in "openbookqa" "copa" "rte"
# for TASK_NAME in "copa"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs20.jsonl"
# for IMP_R in 0.3 0.35 0.4 0.45 0.5
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/impress/66b/fs20_${TASK_NAME}_ret_${IMP_R}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 40 60 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1700 \
#   --gen-len 24 \
#   --important-ratio=${IMP_R} \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/66b/fs20_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

# done 

# done
