

python3 -m flexllmgen.LSH_llama_dapr --model meta/llama3.1-8b  \
        --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
        --path=/HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/HF_HOME/hub \
        --prompt-len 4000 \
        --gen-len 32 \
        --collision-threshold 2 \
        --offload-dir /HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/offload_dir/llama > llama.log 2>&1