#!/bin/bash


# # #############6.7b###################
# MODEL_NAME="facebook/opt-66b"
MODEL_NAME="facebook/opt-30b"
MODEL_TYPE="opt"

# for TASK_NAME in "openbookqa" "copa" "rte"
for TASK_NAME in "copa"
do

INPUT_FILE="datasets/${TASK_NAME}_fs200.jsonl"
for IMP_R in 0.48 0.40 0.30 0.24
do

python3 -u run_lm_eval_harness.py \
  --input-path $INPUT_FILE \
  --output-path logits/impress/30b/fs200_${TASK_NAME}_ret_${IMP_R}.jsonl \
  --use-flexgen \
  --model $MODEL_NAME \
  --percent 100 0 100 0 100 0 \
  --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
  --prompt-len 4000 \
  --gen-len 24 \
  --important-ratio=${IMP_R} \
  --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/30b/fs200_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

done 

done

for TASK_NAME in "rte"
do

INPUT_FILE="datasets/${TASK_NAME}_fs40.jsonl"
for IMP_R in 0.39 0.35 0.31 0.27 0.22
do

python3 -u run_lm_eval_harness.py \
  --input-path $INPUT_FILE \
  --output-path logits/impress/30b/fs40_${TASK_NAME}_ret_${IMP_R}.jsonl \
  --use-flexgen \
  --model $MODEL_NAME \
  --percent 100 0 100 0 100 0 \
  --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
  --prompt-len 4000 \
  --gen-len 24 \
  --important-ratio=${IMP_R} \
  --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/impress > ../log/impress/30b/fs40_lm_eval_${TASK_NAME}_ret_${IMP_R}.log 2>&1

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
