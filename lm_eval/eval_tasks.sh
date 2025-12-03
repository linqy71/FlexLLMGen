#!/bin/bash


#############30b###################
# MODEL_NAME="facebook/opt-66b"
MODEL_NAME="facebook/opt-30b"
MODEL_TYPE="opt"

# for TASK_NAME in "openbookqa" "copa" "rte"
for TASK_NAME in "rte"
do

INPUT_FILE="datasets/${TASK_NAME}_fs15_6.7b.jsonl"
K=8
for L in 100
do

python3 -u run_lm_eval_harness.py \
  --input-path $INPUT_FILE \
  --output-path logits/30b-3head/fs15_${TASK_NAME}_full.jsonl \
  --use-flexgen \
  --model $MODEL_NAME \
  --percent 100 0 100 0 100 0 \
  --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
  --prompt-len 1700 \
  --gen-len 24 \
  --strategy seq \
  --overlap=True \
  --save-res=False \
  --K=$K \
  --L=$L \
  --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering > ../log/30b-3head/fs15_lm_eval_${TASK_NAME}_full.log 2>&1

echo "K $K L $L done"
done


done

# # # #############66b###################
# MODEL_NAME="facebook/opt-66b"
# # MODEL_NAME="facebook/opt-6.7b"
# MODEL_TYPE="opt"

# # for TASK_NAME in "openbookqa" "copa" "rte"
# for TASK_NAME in "rte"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs15.jsonl"
# K=8
# for L in 8
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/66b-3head/fs15_${TASK_NAME}_K${K}L${L}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 50 50 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1700 \
#   --gen-len 24 \
#   --strategy seq \
#   --overlap=True \
#   --save-res=False \
#   --K=$K \
#   --L=$L \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering > ../log/66b-3head/fs15_lm_eval_${TASK_NAME}_K${K}L${L}.log 2>&1

# echo "K $K L $L done"

# done 

# done


# # # #############6.7b###################
# # MODEL_NAME="facebook/opt-66b"
# MODEL_NAME="facebook/opt-6.7b"
# MODEL_TYPE="opt"

# # for TASK_NAME in "openbookqa" "copa" "rte"
# for TASK_NAME in "rte"
# do

# INPUT_FILE="datasets/${TASK_NAME}_fs15.jsonl"
# K=8
# for L in 8
# do

# python3 -u run_lm_eval_harness.py \
#   --input-path $INPUT_FILE \
#   --output-path logits/6.7b-3head/fs15_${TASK_NAME}_K${K}L${L}.jsonl \
#   --use-flexgen \
#   --model $MODEL_NAME \
#   --percent 100 0 100 0 100 0 \
#   --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#   --prompt-len 1700 \
#   --gen-len 24 \
#   --strategy seq \
#   --overlap=True \
#   --save-res=False \
#   --K=$K \
#   --L=$L \
#   --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering > ../log/6.7b-3head/fs15_lm_eval_${TASK_NAME}_K${K}L${L}.log 2>&1

# done 

# done