#!/bin/bash
python3 -m flexllmgen.dapr_opt --model facebook/opt-30b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
        --prompt-len 2016 \
        --gen-len 32 \
        --offload-dir /ssd/nsccgz_zgchen_6/flexllmgen_offload_dir/kv_store/impress  > len8192_reorder_cache.log 2>&1
