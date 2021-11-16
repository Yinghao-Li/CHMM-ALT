#!/bin/bash

# Author: Yinghao Li @ Georgia Tech
# All rights reserved

# -------------------------------------
# This script is for running chmm-train.py
# -------------------------------------

# Quit if there are any errors
set -e

# ne: with neural emission; oe: without neural emission
BERT_MODEL="dmis-lab/biobert-v1.1"
TRAIN_FILE="../../data/NCBI/NCBI-linked-train.pt"
VALID_FILE="../../data/NCBI/NCBI-linked-dev.pt"
TEST_FILE="../../data/NCBI/NCBI-linked-test.pt"
OUTPUT_DIR="./output/ncbi-chmm-oe-0/"
# OUTPUT_DIR="./output/ncbi-chmm-ne-0/"  

LM_BATCH_SIZE=64

NUM_LM_NN_PRETRAIN_EPOCHS=5
NUM_LM_TRAIN_EPOCHS=20

NN_LR=0.0005
SEED=0

CUDA_VISIBLE_DEVICES=$1 python chmm-train.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_file $TRAIN_FILE \
    --valid_file $VALID_FILE \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --lm_batch_size $LM_BATCH_SIZE \
    --num_lm_nn_pretrain_epochs $NUM_LM_NN_PRETRAIN_EPOCHS \
    --num_lm_train_epochs $NUM_LM_TRAIN_EPOCHS \
    --nn_lr $NN_LR \
    --seed $SEED \
    --obs_normalization \
    --no_neural_emiss
