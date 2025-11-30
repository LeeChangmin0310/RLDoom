# config.py

# 상태(프레임) 크기: (height, width, channels)
STATE_SIZE = (100, 120, 4)

# 학습 관련
TOTAL_EPISODES = 1000      # 학습 에피소드 수
MAX_STEPS = 1000           # 에피소드당 최대 스텝 수
BATCH_SIZE = 16

LEARNING_RATE = 0.00025
GAMMA = 0.95               # 할인율

# Epsilon-greedy
EXPLORE_START = 1.0
EXPLORE_STOP = 0.01
DECAY_RATE = 0.00005

# Fixed Q-target
MAX_TAU = 10000    # C 스텝마다 타깃 네트워크 업데이트

# 메모리 / PER
PRETRAIN_LENGTH = 10000   # 초기 랜덤 경험 쌓는 개수
MEMORY_SIZE = 20000       # 메모리 최대 용량

# 플래그
TRAINING = True            # False면 학습 스킵하고 바로 플레이만
EPISODE_RENDER = False     # Doom 화면 렌더할지 여부

# 저장 경로
MODEL_PATH = "./models/model.ckpt"
LOG_DIR = "./logs"

