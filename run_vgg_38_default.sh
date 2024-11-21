#!/bin/bash

# Allowing block type and learning rate as script arguments
BLOCK_TYPE=${1:-'bnrc_block'}
LR=${2:-1e-2}

python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 5 --experiment_name VGG_38_experiment --use_gpu True --num_classes 100 --block_type "$BLOCK_TYPE" --continue_from_epoch -1 --lr "$LR"
