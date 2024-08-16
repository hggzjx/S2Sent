#!/bin/bash

python train.py \
    --model_name_or_path pretrained_model/roberta-base \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/sup-roberta-s2sent \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.05 \
    --num 3 \
    --dim1 768 \
    --dim2 64 \
    --dct_basis low4 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
