#!/bin/bash
cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/Flex2/library/lsh

pip install -e .

cd /HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/my_FlexLLMGen/Flex2/library/kvstore

python setup.py install

cd ../../

sudo drop_cache

python3 -m flexllmgen.test_lsh_io > log/test_lsh_io_0.log 2>&1