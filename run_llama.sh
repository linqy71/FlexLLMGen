#!/bin/bash
LOG_FILE="logs/${1:-llama2}"
python3 -m flexllmgen.dapr_llama_opt --model meta/llama3.1-8b  \
        --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
        --path=/HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/HF_HOME/hub/ \
        --prompt-len 4000 \
        --gen-len 32 \
        --offload-dir ~/HDD_POOL/hyk/my_FlexLLMGen/offload_dir > "$LOG_FILE.log" 2>&1

rm  ~/HDD_POOL/hyk/my_FlexLLMGen/offload_dir/*