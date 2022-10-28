#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/intern/fairseq"

data_dir=/workspace/fairseq/data-bin
data=wmt14_en_fr
#wmt17_en_de iwslt14.tokenized.de-en wmt14_en_fr
custom_model_dir=/workspace/fairseq/custom
student_model=transformer
touch $data_dir/$student_model/$data/train.log

CUDA_VISIBLE_DEVICES=0 fairseq-train $data_dir/$data \
--amp \
--num-workers 2 \
--log-interval=1000 \
--log-format tqdm \
--max-source-positions 210 --max-target-positions 210 --max-update 1000000 --max-tokens 8192 \
--arch $student_model --activation-fn gelu --dropout 0.2 \
--task translation \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--source-lang en --target-lang fr \
--lr-scheduler inverse_sqrt --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 --warmup-init-lr 1e-07 --lr 0.0003 --warmup-updates 4000 \
--save-dir $data_dir/$student_model/$data --save-interval 1 --keep-last-epochs 5 --patience 10 \
--skip-invalid-size-inputs-valid-test \
--update-freq 8 \
--distributed-world-size 1 \
--user-dir $custom_model_dir \
--eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-args '{"beam": 5, "lenpen": 1}' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 2>&1 | tee $data_dir/$student_model/$data/train.log
# --reset-optimizer \
# --restore-file /workspace/fairseq/data-bin/wmt14.en-fr.fconv-py/model.pt \
#--memory-efficient-fp16 \
