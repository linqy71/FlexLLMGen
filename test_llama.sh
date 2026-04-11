
# normal infer test
# python3 -m flexllmgen.LSH_llama_dapr --model meta/llama3.1-8b  \
#         --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
#         --path=/HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/HF_HOME/hub \
#         --prompt-len 4000 \
#         --gen-len 32 \
#         --collision-threshold 2 \
#         --offload-dir /HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/offload_dir/llama > llama.log 2>&1

# longbench test
rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama/*
python3 -u -m flexllmgen.dapr_llama_full --model meta/llama3.1-8b \
      --input full_longbench --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
      --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub \
      --prompt-len 7000 --gen-len 4 \
      --drop-cache \
      --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama > log/impress_llama/8b_longbench.log 2>&1

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama/*
python3 -u -m flexllmgen.dapr_llama_full --model meta/llama3.3-70b \
      --input full_longbench --gpu-batch-size 1 --percent 30 70 100 0 100 0 \
      --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub \
      --prompt-len 7000 --gen-len 4 \
      --drop-cache \
      --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama > log/impress_llama/70b_longbench.log 2>&1

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama/*
python3 -u -m flexllmgen.dapr_llama_full --model meta/llama3.1-8b \
      --input full_dapr --gpu-batch-size 1 --percent 100 0 100 0 100 0 \
      --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub \
      --prompt-len 5000 --gen-len 4 \
      --drop-cache \
      --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama > log/impress_llama/8b_dapr.log 2>&1

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama/*
python3 -u -m flexllmgen.dapr_llama_full --model meta/llama3.3-70b \
      --input full_dapr --gpu-batch-size 1 --percent 30 70 100 0 100 0 \
      --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub \
      --prompt-len 5000 --gen-len 4 \
      --drop-cache \
      --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/impress_llama > log/impress_llama/70b_dapr.log 2>&1
