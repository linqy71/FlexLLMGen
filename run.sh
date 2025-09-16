#!/bin/bash
cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/FlexLLMGen/library/lsh

pip install -e .

cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/FlexLLMGen/library/kvstore

python setup.py install

cd ../../


python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
        --prompt-len 2016 \
        --gen-len 32 \
        --strategy seq \
        --offload-dir /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir  > log/test_seq_f.log 2>&1

python3 -m flexllmgen.LSH_opt_dapr --model facebook/opt-30b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
        --prompt-len 2016 \
        --gen-len 32 \
        --strategy query \
        --offload-dir /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir  > log/test_query_f.log 2>&1   
