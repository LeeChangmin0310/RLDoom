#!/usr/bin/env bash
set -e

# Activate conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

# Go to project root
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# Optionally load .env (for consistency, even though wandb is off in eval)
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# Arguments
ALGO=${1:-dqn}
SEED=${2:-0}
EP_STR=${3:-"003000"}          # checkpoint episode string, e.g., "003000"
EPISODES=${4:-10}              # simple eval episodes
CHUNK_EP=${5:-30}              # episodes per chunk
N_CHUNKS=${6:-10}              # number of chunks

CKPT_PATH="checkpoints/${ALGO}_seed${SEED}_ep${EP_STR}.pth"
LOG_DIR="logs/eval"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/eval_${ALGO}_seed${SEED}_ep${EP_STR}.log"

echo "[RUN_EVAL] algo=${ALGO} seed=${SEED} ckpt=${CKPT_PATH}" | tee "${LOG_FILE}"

python -u eval.py \
  --algo "${ALGO}" \
  --seed "${SEED}" \
  --checkpoint "${CKPT_PATH}" \
  --episodes "${EPISODES}" \
  --chunk_episodes "${CHUNK_EP}" \
  --n_chunks "${N_CHUNKS}" \
  2>&1 | tee -a "${LOG_FILE}"
