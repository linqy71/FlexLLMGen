#!/bin/bash
cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/Flex2/library/lsh

pip install -e .

cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/Flex2/library/kvstore

python setup.py install

cd ../../

python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
        --prompt-len 2016 \
        --gen-len 32 \
        --strategy query \
        --offload-dir /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering  > log/layer_gather_qry_f8.log 2>&1

rm /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering/kv_store/1_*



# python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
#         --percent 100 0 100 0 100 0 \
#         --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
#         --prompt-len 2016 \
#         --gen-len 32 \
#         --strategy seq \
#         --offload-dir /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering  > log/layer_gather_seq_f7.log 2>&1



# rm /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/gathering/kv_store/1_*
