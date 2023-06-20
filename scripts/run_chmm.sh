#!/bin/bash

# Author: Yinghao Li @ Georgia Tech
# All rights reserved

# -------------------------------------
# This script is for running chmm-train.py
# -------------------------------------

# Quit if there are any errors
set -e

# ne: with neural emission; oe: without neural emission
BERT_MODEL="bert-base-uncased"
TRAIN_FILE="./data/LaptopReview/train.json"
VALID_FILE="./data/LaptopReview/valid.json"
TEST_FILE="./data/LaptopReview/test.json"
OUTPUT_DIR="./output/"

LM_BATCH_SIZE=256

NUM_LM_NN_PRETRAIN_EPOCHS=10
NUM_LM_TRAIN_EPOCHS=30

NN_LR=0.0001
SEED=0

PYTHONPATH="." CUDA_VISIBLE_DEVICES=$1 python ./run/chmm.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_path $TRAIN_FILE \
    --valid_path $VALID_FILE \
    --test_path $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --lm_batch_size $LM_BATCH_SIZE \
    --num_lm_nn_pretrain_epochs $NUM_LM_NN_PRETRAIN_EPOCHS \
    --num_lm_train_epochs $NUM_LM_TRAIN_EPOCHS \
    --nn_lr $NN_LR \
    --seed $SEED \
    --obs_normalization
