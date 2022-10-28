#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/workspace/fairseq"

data_dir=/workspace/fairseq/data-bin
data=wmt14_en_fr
#wmt17_en_de iwslt14.tokenized.de-en wmt14_en_fr
model=transformer

CUDA_VISIBLE_DEVICES=0,1 fairseq-generate $data_dir/$data/ \
    --path $data_dir/$model/$data/checkpoint_last.pt \
    --batch-size 128 --beam 5 --remove-bpe --skip-invalid-size-inputs-valid-test \
    --user-dir ./custom --log-format json 2>&1 | tee $data_dir/$model/$data/test.log