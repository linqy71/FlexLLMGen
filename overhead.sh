#!/bin/bash
# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/lsh

# pip install -e .

# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/kvstore

# python setup.py install

# cd ../../


rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*
K=6
L=8
cr=0.8
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 5000 \
        --gen-len 5 \
        --strategy query \
        --merge=True \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --compaction-threshold=$cr \
        --input dapr \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/overhead/30b_lsh_K${K}L${L}_cr${cr}_dapr.log 2>&1
echo "done ${cr}"

# cr=0.5
# python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
#         --percent 100 0 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
#         --prompt-len 5000 \
#         --gen-len 5 \
#         --strategy query \
#         --merge=True \
#         --overlap=True \
#         --save-res=False \
#         --K=$K \
#         --L=$L \
#         --compaction-threshold=$cr \
#         --input dapr \
#         --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/overhead/30b_lsh_K${K}L${L}_cr${cr}_dapr.log 2>&1
# echo "done ${cr}"

################

# rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*
# K=6
# L=20
# python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
#         --percent 60 40 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
#         --prompt-len 8000 \
#         --gen-len 5 \
#         --strategy query \
#         --merge=True \
#         --overlap=True \
#         --save-res=False \
#         --K=$K \
#         --L=$L \
#         --compaction-threshold 1 \
#         --input long \
#         --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/overhead/30b_lsh_K${K}L${L}_cr1_longbench.log 2>&1
# echo "done long"