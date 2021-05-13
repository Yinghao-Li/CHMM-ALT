#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=Co03
BERT_EPOCH=15
PHASE2_EPOCH=5
BERT_BATCH_SIZE=16
DENOISING_BATCH_SIZE=32
DENOISING_EPOCH=20
DENOISING_PRETRAIN_EPOCH=5
NN_LR=0.00001
MAX_SEQ_LEN=256
OUTPUT_DIR=./Co03
MODEL=bert-base-uncased
SEED=1

CUDA_VISIBLE_DEVICES=$1 python alt_train.py \
    --data_dir ./data/ \
    --dataset_name $DATASET \
    --denoising_model nhmm \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_SEQ_LEN \
    --num_train_epochs $BERT_EPOCH \
    --phase2_train_epochs $PHASE2_EPOCH \
    --per_device_train_batch_size $BERT_BATCH_SIZE \
    --per_device_eval_batch_size $BERT_BATCH_SIZE \
    --denoising_batch_size $DENOISING_BATCH_SIZE \
    --denoising_epoch $DENOISING_EPOCH \
    --denoising_pretrain_epoch $DENOISING_PRETRAIN_EPOCH \
    --nn_lr $NN_LR \
    --save_steps 999999999 \
    --seed $SEED \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --overwrite_cache \
    --converse_first \
    --disable_tqdm True
