#!/bin/bash


############30b###################
# MODEL_NAME="facebook/opt-66b"
MODEL_NAME="facebook/opt-30b"
MODEL_TYPE="opt"

for TASK_NAME in "rte"
do

INPUT_FILE="datasets/${TASK_NAME}_fs10.jsonl"
for IMP_R in 1.0
do

python3 -u run_lm_eval_harness.py \
  --input-path $INPUT_FILE \
  --output-path logits/impress/30b/${TASK_NAME}_ret_${IMP_R}.jsonl \
  --use-flexgen \
  --model $MODEL_NAME \
  --percent 100 0 100 0 100 0 \
  --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
  --prompt-len 1300 \
  --gen-len 24 \
  --important-ratio=${IMP_R} \
  --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/30b/lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

done 

done

# #############66b###################
# MODEL_NAME="facebook/opt-66b"
# # MODEL_NAME="facebook/opt-6.7b"
# MODEL_TYPE="opt"

# for TASK_NAME in "openbookqa" "copa" "rte"
# # for TASK_NAME in "copa"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs10.jsonl"
# for IMP_R in 0.25 0.3 0.35 0.4 0.45
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/impress/66b/${TASK_NAME}_ret_${IMP_R}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 40 60 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1300 \
#   --gen-len 24 \
#   --important-ratio=${IMP_R} \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/66b/lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

# done 

# done


# # # #############6.7b###################
# # # MODEL_NAME="facebook/opt-66b"
# MODEL_NAME="facebook/opt-6.7b"
# MODEL_TYPE="opt"

# for TASK_NAME in "openbookqa" "copa" "rte"
# # for TASK_NAME in "copa"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs10.jsonl"
# for IMP_R in 0.25 0.3 0.35 0.4 0.45
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/impress/6.7b/${TASK_NAME}_ret_${IMP_R}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 100 0 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1300 \
#   --gen-len 24 \
#   --important-ratio=${IMP_R} \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/6.7b/lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

# done 

# done