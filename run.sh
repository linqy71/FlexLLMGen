#!/bin/bash
# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/lsh

# pip install -e .

# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/kvstore

# python setup.py install

# cd ../../


# rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/kv_store/1_*
# rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/kv_store/q*

# python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
#         --percent 100 0 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#         --prompt-len 2024 \
#         --gen-len 24 \
#         --strategy seq \
#         --overlap=True \
#         --save-res=False \
#         --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering  > log/30b_layer_seq_f_8192_dapr_test_3head.log 2>&1


# python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
#         --percent 100 0 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#         --prompt-len 2024 \
#         --gen-len 24 \
#         --strategy seq \
#         --overlap=True \
#         --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering  > log/30b_layer_seq_f_8192_concur_4.log 2>&1

#rm /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering/kv_store/1_*



# python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-66b \
#         --percent 25 75 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#         --prompt-len 2032 \
#         --gen-len 16 \
#         --strategy seq \
#         --offload-dir /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering  > log/66b_layer_seq_d0_8192.log 2>&1



# rm /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering/kv_store/1_*
