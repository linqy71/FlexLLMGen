#!/bin/bash
# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/lsh

# pip install -e .

# cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/FlexLLMGen/library/kvstore

# python setup.py install

# cd ../../


rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*.bin
K=6
L=8
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-6.7b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 5000 \
        --gen-len 24 \
        --strategy query \
        --merge=True \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --compaction-threshold 0.5 \
        --input dapr \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/overhead/6.7b_lsh_K${K}L${L}_cr0.5.log 2>&1
echo "done 6.7b"
################

rm /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/kv_store/*.bin
K=6
L=8
python3 -u -m flexllmgen.LSH_opt_dapr --model facebook/opt-6.7b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights \
        --prompt-len 5000 \
        --gen-len 24 \
        --strategy query \
        --merge=True \
        --overlap=True \
        --save-res=False \
        --K=$K \
        --L=$L \
        --compaction-threshold 1 \
        --input dapr \
        --offload-dir /ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/gathering_1/  > log/overhead/6.7b_lsh_K${K}L${L}_cr1.log 2>&1
echo "done 6.7b"