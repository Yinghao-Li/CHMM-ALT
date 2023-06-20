#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run chmm-alt.py
# -------------------------------------

# Quit if there are any errors
set -e

BERT_MODEL="bert-base-uncased"
TRAIN_FILE="./data/LaptopReview/train.json"
VALID_FILE="./data/LaptopReview/valid.json"
TEST_FILE="./data/LaptopReview/test.json"
OUTPUT_DIR="./output"
MAX_SEQ_LENGTH=128

LM_BATCH_SIZE=128
EM_BATCH_SIZE=48

NUM_LM_NN_PRETRAIN_EPOCHS=3
NUM_LM_TRAIN_EPOCHS=20
NUM_EM_TRAIN_EPOCHS=100
NUM_PHASE2_EM_TRAIN_EPOCHS=20
NUM_PHASE2_LOOP=5

NN_LR=0.0001
SEED=0

CUDA_VISIBLE_DEVICES=$1 python chmm-alt.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_file $TRAIN_FILE \
    --valid_file $VALID_FILE \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --max_length $MAX_SEQ_LENGTH \
    --lm_batch_size $LM_BATCH_SIZE \
    --em_batch_size $EM_BATCH_SIZE \
    --num_lm_nn_pretrain_epochs $NUM_LM_NN_PRETRAIN_EPOCHS \
    --num_lm_train_epochs $NUM_LM_TRAIN_EPOCHS \
    --num_em_train_epochs $NUM_EM_TRAIN_EPOCHS \
    --num_phase2_em_train_epochs $NUM_PHASE2_EM_TRAIN_EPOCHS \
    --num_phase2_loop $NUM_PHASE2_LOOP \
    --nn_lr $NN_LR \
    --seed $SEED \
    --obs_normalization \
    --pass_soft_labels
