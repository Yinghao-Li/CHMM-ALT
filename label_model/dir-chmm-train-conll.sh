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
TRAIN_FILE="../../data/Co03/Co03-linked-train.pt"
VALID_FILE="../../data/Co03/Co03-linked-dev.pt"
TEST_FILE="../../data/Co03/Co03-linked-test.pt"
# OUTPUT_DIR="./output/laptop-dirchmm-de-3-nonorm/"
OUTPUT_DIR="./output/conll-dirchmm-0-v1/"

LM_BATCH_SIZE=256

NUM_LM_NN_PRETRAIN_EPOCHS=5
NUM_LM_TRAIN_EPOCHS=100
NUM_LM_VALID_TOLERANCE=30

CONC_BASE=10
CONC_MAX=500

# The following 3 parameters achieves good performance and should be kept
DIAG_EXP_T1=2
DIAG_EXP_T2=4
NONDIAG_EXP=2

NN_LR=0.0001
SEED=0

for DIAG_EXP_T1 in 1.5 2 3 4
do
for DIAG_EXP_T2 in 3 4 5
do
for NONDIAG_EXP in 1.5 2 3 4
do

OUTPUT_DIR="./output/conll-dirchmm-${SEED}-${DIAG_EXP_T1}-${DIAG_EXP_T2}-${NONDIAG_EXP}/"

CUDA_VISIBLE_DEVICES=$1 python dir-chmm-train.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_file $TRAIN_FILE \
    --valid_file $VALID_FILE \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --dirichlet_concentration_base $CONC_BASE \
    --dirichlet_concentration_max $CONC_MAX \
    --diag_exp_t1 $DIAG_EXP_T1 \
    --diag_exp_t2 $DIAG_EXP_T2 \
    --nondiag_exp $NONDIAG_EXP \
    --lm_batch_size $LM_BATCH_SIZE \
    --num_lm_nn_pretrain_epochs $NUM_LM_NN_PRETRAIN_EPOCHS \
    --num_lm_train_epochs $NUM_LM_TRAIN_EPOCHS \
    --num_lm_valid_tolerance $NUM_LM_VALID_TOLERANCE \
    --nn_lr $NN_LR \
    --seed $SEED \
    --use_dirichlet_emiss

done
done
done
