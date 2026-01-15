#!/bin/bash
# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/lsh

# pip install -e .

# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/kvstore

# python setup.py install

# cd ../../


rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*.bin
K=6
L=15
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-6.7b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 7000 \
        --gen-len 24 \
        --strategy query \
        --merge=True \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/ablation/6.7b_lsh_K${K}L${L}.log 2>&1
echo "done 6.7b"
################

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*
K=6
L=20
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
        --percent 70 30 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 7000 \
        --gen-len 20 \
        --strategy seq \
        --merge=False \
        --overlap=False \
        --save-res=False \
        --K=$K \
        --L=$L \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/ablation/30b_lsh_K${K}L${L}_seq.log 2>&1

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*.bin
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
        --percent 70 30 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 7000 \
        --gen-len 20 \
        --strategy seq \
        --merge=True \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/ablation/30b_lsh_K${K}L${L}_seq_merge.log 2>&1

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*.bin
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
        --percent 70 30 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 7000 \
        --gen-len 20 \
        --strategy query \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/ablation/30b_lsh_K${K}L${L}.log 2>&1
##############


rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*
K=6
L=24
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-66b \
        --percent 20 80 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 7000 \
        --gen-len 20 \
        --strategy seq \
        --merge=False \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/  > log/ablation/66b_lsh_K${K}L${L}_seq.log 2>&1

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*.bin
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-66b \
        --percent 20 80 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 7000 \
        --gen-len 20 \
        --strategy seq \
        --merge=True \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/  > log/ablation/66b_lsh_K${K}L${L}_seq_merge.log 2>&1

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*.bin
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-66b \
        --percent 20 80 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 7000 \
        --gen-len 20 \
        --strategy query \
        --merge=True \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --input long \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering/  > log/ablation/66b_lsh_K${K}L${L}.log 2>&1