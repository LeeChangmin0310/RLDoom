#!/usr/bin/env bash
set -e

# Activate conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

# Go to project root
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# Load .env if exists (for WANDB_API_KEY, WANDB_PROJECT, etc.)
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# Use specific GPU
export CUDA_VISIBLE_DEVICES=3

mkdir -p logs

ALGO=${1:-dqn}
SEED=${2:-0}

python -u train.py --algo "${ALGO}" --seed "${SEED}" \
  2>&1 | tee "logs/train_${ALGO}_seed${SEED}.log"
