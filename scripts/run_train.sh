# scripts/run_train.sh
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

mkdir -p logs checkpoints

# 1st arg: algo name (e.g., dqn, a2c_tuned)
# 2nd arg: seed
ALGO=${1:-${ALGO:-dddqn}}
SEED=${2:-${SEED:-0}}

echo "[INFO] Running algo=${ALGO}, seed=${SEED}"

python train.py \
  --algo "${ALGO}" \
  --seed "${SEED}" \
  2>&1 | tee "logs/train_${ALGO}_seed${SEED}.log"
