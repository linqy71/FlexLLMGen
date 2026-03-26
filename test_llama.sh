
# normal infer test
# python3 -m flexllmgen.LSH_llama_dapr --model meta/llama3.1-8b  \
#         --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
#         --path=/HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/HF_HOME/hub \
#         --prompt-len 4000 \
#         --gen-len 32 \
#         --collision-threshold 2 \
#         --offload-dir /HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/offload_dir/llama > llama.log 2>&1

# eval test
python3 -m flexllmgen.LSH_llama_full --model meta/llama3.1-8b \
      --input full_dapr --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
      --path=/HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/HF_HOME/hub \
      --prompt-len 7000 --gen-len 32 \
      --K 8 --L 100 \
      --collision-threshold 2 \
      --offload-dir /HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/offload_dir/llama > llama.log 2>&1