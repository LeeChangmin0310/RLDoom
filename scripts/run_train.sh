#!/usr/bin/env bash
set -e

# Load env vars from .env (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, WANDB_DIR)
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

# Go to project root
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# Select GPU
export CUDA_VISIBLE_DEVICES=3

# Ensure wandb/logs dirs exist
mkdir -p "${WANDB_DIR:-${PWD}/logs/wandb}"
mkdir -p logs

python train.py "$@"
