#!/bin/bash
python3 -m flexllmgen.dapr_llama --model meta/llama3.1-8b  \
        --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
        --path=/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub \
        --prompt-len 4000 \
        --gen-len 32 \
        --offload-dir /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/flexllmgen_offload_dir > llama.log 2>&1