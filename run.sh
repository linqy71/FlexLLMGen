#!/bin/bash

# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/FlexLLMGen/library/direct_io
# pip install -e .
# cd ../..
export CUDA_VISIBLE_DEVICES=0
# rm  /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress/*
# python3 -m flexllmgen.dapr_opt --model facebook/opt-30b \
#         --percent 100 0 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#         --prompt-len 7000 \
#         --gen-len 24 \
#         --request 12 \
#         --overlap True \
#         --input long \
#         --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress  > log/full_longbench/impress_30b_chunk256_imp25.log 2>&1


# rm  /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress/*
# python3 -m flexllmgen.dapr_opt --model facebook/opt-6.7b \
#         --percent 100 0 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#         --prompt-len 7000 \
#         --gen-len 24 \
#         --request 12 \
#         --overlap True \
#         --input long \
#         --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress  > log/full_longbench/impress_6.7b_chunk256_imp25.log 2>&1

rm  /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress/*
python3 -m flexllmgen.dapr_opt --model facebook/opt-66b \
        --percent 25 75 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
        --prompt-len 7000 \
        --gen-len 24 \
        --request 12 \
        --overlap True \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/kv_store/impress  > log/full_longbench/impress_66b_chunk256_imp25.log 2>&1
