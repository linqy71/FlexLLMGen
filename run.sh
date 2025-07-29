#!/bin/bash
python3 -m flexllmgen.impress_opt --model facebook/opt-6.7b \
        --percent 100 0 100 0 100 0 \
        --path /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights  \
        --offload-dir /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/flexllmgen_offload_dir  > test.log 2>&1
