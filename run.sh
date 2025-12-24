#!/bin/bash

# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/FlexLLMGen/library/direct_io
# pip install -e .
# cd ../..
export CUDA_VISIBLE_DEVICES=0
rm  /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress/*
python3 -m flexllmgen.dapr_opt --model facebook/opt-30b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
        --prompt-len 4000 \
        --gen-len 24 \
        --request 12 \
        --overlap True \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress  > log/impress_30b_chunk256_imp30_overlap.log 2>&1

# for i in {0..5}; do
#     echo "Running iteration $i"
#     python3 -m flexllmgen.dapr_opt --model facebook/opt-30b \
#             --percent 100 0 100 0 100 0 \
#             --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
#             --prompt-len 2016 \
#             --gen-len 32 \
#             --offload-dir /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/kv_store/impress > "len8192_${i}.log" 2>&1
#     echo "Iteration $i completed. Output saved to len8192_${i}.log"
# done

# echo "All iterations completed."