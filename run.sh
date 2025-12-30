#!/bin/bash
# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/lsh

# pip install -e .

# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/kvstore

# python setup.py install

# cd ../../


rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/kv_store/*
K=6
L=8
python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-6.7b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 5000 \
        --gen-len 24 \
        --strategy query \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/  > log/6.7b_test_fulldapr_K${K}L${L}.log 2>&1

# rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/kv_store/1_*
# rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/kv_store/q*

# python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-66b \
#         --percent 30 70 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
#         --prompt-len 4000 \
#         --gen-len 24 \
#         --strategy seq \
#         --overlap=True \
#         --save-res=False \
#         --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/  > log/66b_load_lsh_meta.log 2>&1


# rm /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering/kv_store/1_*
