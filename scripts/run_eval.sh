#!/usr/bin/env bash
set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

# Go to project root
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# Use only GPU 3
export CUDA_VISIBLE_DEVICES=3

# Create logs directory if it does not exist
mkdir -p logs

# Run evaluation and save console output
python eval.py "$@"
