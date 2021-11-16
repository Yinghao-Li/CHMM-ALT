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
TRAIN_FILE="../../data/Laptop/Laptop-linked-train.pt"
VALID_FILE="../../data/Laptop/Laptop-linked-dev.pt"
TEST_FILE="../../data/Laptop/Laptop-linked-test.pt"
# OUTPUT_DIR="./output/laptop-dirchmm-de-3-nonorm/"
OUTPUT_DIR="./output/laptop-dirchmm-de-0/"

LM_BATCH_SIZE=128
CONCENTRATION_MAX=100
CONCENTRATION_BASE=10

NUM_LM_NN_PRETRAIN_EPOCHS=3
NUM_LM_TRAIN_EPOCHS=20

NN_LR=0.0001
SEED=0

CUDA_VISIBLE_DEVICES=$1 python dir-chmm-train.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_file $TRAIN_FILE \
    --valid_file $VALID_FILE \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --dirichlet_concentration_base $CONCENTRATION_BASE \
    --dirichlet_concentration_max $CONCENTRATION_MAX \
    --lm_batch_size $LM_BATCH_SIZE \
    --num_lm_nn_pretrain_epochs $NUM_LM_NN_PRETRAIN_EPOCHS \
    --num_lm_train_epochs $NUM_LM_TRAIN_EPOCHS \
    --nn_lr $NN_LR \
    --seed $SEED \
    --use_dirichlet_emiss \
    --obs_normalization
