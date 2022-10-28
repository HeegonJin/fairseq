#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/workspace/fairseq"

data_dir=/workspace/fairseq/data
data=iwslt14.tokenized.de-en
#wmt17_en_de iwslt14.tokenized.de-en
strategy=global_level
#generic, global_level, batch_level
model=transformer_tiny
temperature=1.5

CUDA_VISIBLE_DEVICES=3,6 fairseq-generate $data_dir/$data/ \
    --path /$data_dir/$strategy/$model/$data/$temperature/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe --skip-invalid-size-inputs-valid-test \
    --user-dir ./custom --log-format json 2>&1 | tee $data_dir/$strategy/$model/$data/$temperature/test.log