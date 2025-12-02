# scripts/launch_batch.sh
#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# 공통 seed (원하면 배열로 바꿔서 여러 seed도 가능)
SEED=0

# Stage 1: 세 개 알고리즘을 GPU 1,2,3에서 동시에 실행
CUDA_VISIBLE_DEVICES=1 bash scripts/run_train.sh dqn    ${SEED} &
PID1=$!

CUDA_VISIBLE_DEVICES=2 bash scripts/run_train.sh dddqn  ${SEED} &
PID2=$!

CUDA_VISIBLE_DEVICES=3 bash scripts/run_train.sh rainbow ${SEED} &
PID3=$!

echo "[Stage 1] PIDs: $PID1 $PID2 $PID3"
wait $PID1 $PID2 $PID3   # 세 개 모두 끝날 때까지 기다림
echo "[Stage 1] done."

# Stage 2: 다음 세 개
CUDA_VISIBLE_DEVICES=1 bash scripts/run_train.sh reinforce ${SEED} &
PID4=$!
CUDA_VISIBLE_DEVICES=2 bash scripts/run_train.sh a2c       ${SEED} &
PID5=$!
CUDA_VISIBLE_DEVICES=3 bash scripts/run_train.sh a3c       ${SEED} &
PID6=$!

echo "[Stage 2] PIDs: $PID4 $PID5 $PID6"
wait $PID4 $PID5 $PID6
echo "[Stage 2] done."

# Stage 3: 마지막 두 개 (원하면 한 개는 비워둬도 됨)
CUDA_VISIBLE_DEVICES=1 bash scripts/run_train.sh ppo  ${SEED} &
PID7=$!
CUDA_VISIBLE_DEVICES=2 bash scripts/run_train.sh trpo ${SEED} &
PID8=$!

echo "[Stage 3] PIDs: $PID7 $PID8"
wait $PID7 $PID8
echo "[Stage 3] done. All experiments finished."
