#!/usr/bin/env bash
set -e

# Activate conda environment
# Change this path if your conda installation is located elsewhere
source ~/miniconda3/etc/profile.d/conda.sh
conda activate doomrl

# Go to project root
cd ~/projects/doom-rl  # change to your actual path

# Use only GPU 0 (e.g., RTX A5000)
export CUDA_VISIBLE_DEVICES=0

# Optional: set local wandb directory
export WANDB_DIR="${PWD}/logs/wandb"
mkdir -p "$WANDB_DIR"

mkdir -p logs

# Run training and save console output
python train.py 2>&1 | tee logs/train.log
