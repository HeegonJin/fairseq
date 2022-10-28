#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/intern/fairseq"

data_dir=/home/intern/fairseq/data-bin
data=wmt14_en_fr
#wmt17_en_de iwslt14.tokenized.de-en wmt14_en_fr
custom_model_dir=/home/intern/fairseq/custom
teacher_model=transformer
student_model=transformer
strategy=global_level
temperature=1.5
#generic, global_level, batch_level
touch $data_dir/$strategy/$student_model/$data/$temperature/train.log

CUDA_VISIBLE_DEVICES=0 fairseq-train $data_dir/$data \
--amp \
--num-workers 2 \
--log-interval=1 \
--log-format tqdm \
--max-source-positions 210 --max-target-positions 210 --max-update 1000000 --max-tokens 8192 \
--arch $student_model --activation-fn gelu --dropout 0.2 \
--task kd_translation --kd-strategy $strategy --alpha 0.5 --kd-rate 0.5 --teacher-temp $temperature --student-temp $temperature \
--teacher-checkpoint-path $data_dir/$teacher_model/$data/checkpoint_best.pt \
--criterion kd_label_smoothed_cross_entropy --label-smoothing 0.1 \
--source-lang en --target-lang fr \
--lr-scheduler inverse_sqrt --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 --warmup-init-lr 1e-07 --lr 0.0003 --warmup-updates 4000 \
--save-dir $data_dir/$strategy/$student_model/$data/$temperature --save-interval 1 --keep-last-epochs 5 --patience 10 \
--skip-invalid-size-inputs-valid-test \
--update-freq 8 \
--distributed-world-size 1 \
--rambda 10 \
--user-dir $custom_model_dir 2>&1 | tee $data_dir/$strategy/$student_model/$data/$temperature/train.log \
# --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-args '{"beam": 5, "lenpen": 1}' --best-checkpoint-metric bleu
# --teacher-checkpoint-path $data_dir/$teacher_model/$data/model.pt \
#--memory-efficient-fp16 \