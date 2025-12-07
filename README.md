# RLDoom: Comparing Deep RL Algorithms on VizDoom Deadly Corridor

**TL;DR.** This repository implements a **comparative benchmark** of classic and modern Deep RL algorithms on the **VizDoom Deadly Corridor** task, using a **unified environment wrapper, YAML-based configs, and wandb logging**.

All agents share:

- The **same VizDoom environment** (Deadly Corridor)
- The **same preprocessing & frame stacking**
- The **same logging / checkpointing pipeline**

so that we can focus on **algorithmic differences** in sample efficiency and final performance.

<p align="center">
  <!-- Replace this path with your actual GIF file -->
  <img src="images/doom_corridor_demo.gif" alt="RLDoom Deadly Corridor demo" width="480"/>
</p>

---

## Algorithms

All algorithms act on the **same discrete 7-action space** and stacked grayscale frames `(C, H, W)`:

### Off-policy (value-based)

- **DQN**
- **DDQN** (Double DQN)
- **DDDQN** (Dueling Double DQN)
- **Rainbow-style DQN (Rainbow-lite)**
  - Dueling Q-head
  - Double DQN target update
  - Prioritized Experience Replay (PER) via a dedicated buffer  
  - No noisy nets / no distributional head (simplified Rainbow variant)

### On-policy (policy gradient & actor-critic)

- **REINFORCE** (Monte Carlo policy gradient, with optional value baseline)
- **A2C** (Advantage Actor-Critic, synchronous)
- **A3C** (Advantage Actor-Critic, asynchronous-style loop in a single process)
- **PPO** (Proximal Policy Optimization)
- **TRPO** (Trust Region Policy Optimization, simplified with a KL penalty)

The goal is a **clean, reproducible benchmark** that can run on a lab server (SSH + tmux) without touching global system packages.

---

## Environment

- **Scenario:** VizDoom – *Deadly Corridor*
- **Goal:** Reach the vest at the end of the corridor **without dying**.
- **Map:** narrow corridor with monsters on both sides.
- **Reward (shaped):**
  - `+dX` for getting closer to the vest
  - `-dX` for moving away
  - `-100` death penalty

### Actions

We use **7 discrete actions** (one-hot):

- MOVE_LEFT  
- MOVE_RIGHT  
- ATTACK  
- MOVE_FORWARD  
- MOVE_BACKWARD  
- TURN_LEFT  
- TURN_RIGHT  

### Preprocessing & Frame Stacking

Implemented in `rldoom/envs/deadly_corridor.py`:

- Take raw RGB buffer from VizDoom.
- Convert to **grayscale**.
- **Crop** uninformative regions (HUD, floor, etc.).
- **Resize** to `frame_size x frame_size` (default: 84×84).
- Normalize to `[0, 1]`.
- Maintain a **deque of last 4 frames**, stacked to tensor of shape `(4, 84, 84)`.

This ensures all algorithms see **identical observations**.

---

## Repository Structure

```text
RLDoom/
  train.py                 # Main training entrypoint (single algo/seed)
  eval.py                  # Evaluation entrypoint
  requirements.txt         # Python dependencies

  doom_files/
    deadly_corridor.cfg    # VizDoom config
    deadly_corridor.wad    # VizDoom scenario

  rldoom/
    __init__.py

    configs/
      __init__.py
      deadly_corridor.yaml # YAML config (env, train, logging, algo-specific)
      # make_config(algo, seed) → flat cfg object

    envs/
      __init__.py
      deadly_corridor.py   # VizDoom wrapper + preprocessing + frame stacking

    models/
      __init__.py
      cnn_backbone.py      # Shared convolutional encoder for images
      heads.py             # Q-heads, dueling heads, actor/critic heads, etc.

    buffers/
      __init__.py
      replay_buffer.py         # Standard experience replay buffer
      prioritized_replay.py    # Prioritized Experience Replay (Rainbow)
      rollout_buffer.py        # On-policy rollout storage for PPO/TRPO/A2C/A3C

    agents/
      __init__.py
      base.py              # Base Agent class (common interface)
      dqn.py               # Vanilla DQN
      ddqn.py              # Double DQN
      dddqn.py             # Dueling Double DQN
      rainbow.py           # Rainbow-lite DQN with PER
      reinforce.py         # REINFORCE (MC policy gradient, optional baseline)
      a2c.py               # Advantage Actor-Critic
      a3c.py               # A3C-style agent (single-process version)
      ppo.py               # PPO agent
      trpo.py              # TRPO-style agent

    trainers/
      __init__.py
      offpolicy.py         # Training loop for DQN / DDQN / DDDQN / Rainbow
      onpolicy.py          # Training loop for REINFORCE / A2C / A3C / PPO / TRPO

    utils/
      __init__.py
      logger.py            # Wandb/console logger wrapper
      seeding.py           # Seeding helper
      misc.py              # Small utilities (path helpers, etc.)

  scripts/
    run_train.sh           # tmux-friendly single-algorithm launcher
    run_eval.sh            # tmux-friendly evaluation launcher
    launch_queue.py        # Multi-GPU job launcher for training
    launch_eval_queue.py   # Multi-GPU job launcher for evaluation

  checkpoints/             # Saved model checkpoints (created at runtime)
  logs/                    # Text logs + wandb local cache (created at runtime)
  README.md
````

---

## Installation

Create and activate a dedicated conda environment:

```bash
git clone https://github.com/LeeChangmin0310/RLDoom.git
cd RLDoom

conda create -n doomrl python=3.9 -y
conda activate doomrl

pip install -r requirements.txt
```

Check that the environment Python is used:

```bash
which python
# -> .../anaconda3/envs/doomrl/bin/python
```

---

## VizDoom Assets

The YAML config assumes the following paths:

```yaml
env:
  cfg_path: "doom_files/deadly_corridor.cfg"
  wad_path: "doom_files/deadly_corridor.wad"
```

Place these files under `doom_files/`:

```text
RLDoom/
  doom_files/
    deadly_corridor.cfg
    deadly_corridor.wad
```

You can copy them from the official VizDoom examples or tutorial resources.

---

## Configuration (YAML + Python)

All configuration is centralized in:

```text
rldoom/configs/deadly_corridor.yaml
rldoom/configs/__init__.py  # make_config
```

### YAML layout (high-level example)

```yaml
env:
  cfg_path: "doom_files/deadly_corridor.cfg"
  wad_path: "doom_files/deadly_corridor.wad"
  frame_size: 84
  stack_size: 4
  frame_skip: 4

train:
  num_episodes: 3000
  max_steps_per_episode: 3000
  checkpoint_dir: "checkpoints"
  checkpoint_interval: 150
  logs_dir: "logs"

defaults:
  feature_dim: 512
  gamma: 0.99
  grad_clip: 10.0

logging:
  use_wandb: true
  wandb_project: "RLDoom"
  wandb_entity: "lee_changmin-sangmyung-uni"

algos:
  dqn:
    type: "offpolicy"
    # ...
  ddqn:
  dddqn:
  rainbow:
  reinforce:
  a2c:
  a3c:
  ppo:
  trpo:
  # optionally: reinforce_tuned / a2c_tuned / dddqn_tuned / ppo_tuned
```

`make_config(algo, seed)` flattens this into a simple object:

```python
from rldoom.configs import make_config

cfg = make_config("dddqn", seed=0)

cfg.algo            # "dddqn"
cfg.algo_type       # "offpolicy" / "onpolicy"
cfg.cfg_path        # doom_files/deadly_corridor.cfg
cfg.wad_path        # doom_files/deadly_corridor.wad
cfg.frame_size      # 84
cfg.stack_size      # 4
cfg.frame_skip      # 4

cfg.num_episodes
cfg.max_steps_per_episode
cfg.checkpoint_dir
cfg.logs_dir
cfg.feature_dim
cfg.gamma
cfg.grad_clip

# algo-specific examples
cfg.lr
cfg.buffer_size
cfg.batch_size
cfg.learn_start
cfg.eps_start
cfg.eps_end
cfg.eps_decay
cfg.target_update_every
```

All agents and trainers use this same `cfg` object.

---

## Logged Metrics (wandb)

Every algorithm logs per episode:

* `episode`: 1-based episode index (used as wandb step)
* `return`: sum of rewards in the episode
* `length`: number of environment steps in the episode
* `global_step`: total environment steps so far

Algorithm-specific losses:

* **Off-policy (DQN, DDQN, DDDQN, Rainbow)**

  * `loss`: TD loss (total, identical to `value_loss`)
  * `value_loss`: same scalar, logged explicitly

* **REINFORCE**

  * `loss`: total loss (policy + value baseline term)
  * `policy_loss`
  * `value_loss` (baseline fitting; optional)

* **A2C**

  * `loss`: total (policy + vf_coef * value_loss − ent_coef * entropy)
  * `policy_loss`
  * `value_loss`
  * `entropy`

* **PPO**

  * `loss`: total (clipped objective + value term − entropy)
  * `policy_loss`
  * `value_loss`

* **TRPO (simplified)**

  * `loss`: total (policy + KL penalty + value term − entropy)
  * `policy_loss`
  * `value_loss`
  * `kl`: mean KL divergence between old and new policies

This makes it easy to compare **return curves** and **loss dynamics** across algorithms under a unified logging schema.

---

## wandb Setup (.env)

Wandb is configured via **environment variables** and/or the YAML:

Create a `.env` file in the project root (or export in your shell):

```bash
WANDB_API_KEY="YOUR_KEY"
WANDB_ENTITY="lee_changmin-sangmyung-uni"
WANDB_PROJECT="RLDoom"
WANDB_DIR="/home/cia/disk1/bci_intern/AAAI2026/RLDoom/logs/wandb"
```

Then either:

```bash
export $(grep -v '^#' .env | xargs)
```

or your logger can load `.env` internally (e.g., via `python-dotenv`).

`rldoom/utils/logger.py` reads:

* `cfg.use_wandb`
* `cfg.wandb_project`
* `cfg.wandb_entity`
* `WANDB_API_KEY`, `WANDB_DIR` from the environment

and initializes wandb accordingly.

---

## Training

### Single run (command-line)

From the project root:

```bash
conda activate doomrl
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# DDDQN
python train.py --algo dddqn --seed 0

# DQN / DDQN / Rainbow
python train.py --algo dqn --seed 0
python train.py --algo ddqn --seed 0
python train.py --algo rainbow --seed 0

# On-policy algorithms
python train.py --algo reinforce --seed 0
python train.py --algo a2c --seed 0
python train.py --algo a3c --seed 0
python train.py --algo ppo --seed 0
python train.py --algo trpo --seed 0
```

`train.py`:

* Builds `cfg = make_config(algo, seed)`
* Sets seeds (Python / NumPy / Torch)
* Creates `obs_shape = (stack_size, frame_size, frame_size)` and `num_actions = 7`
* Instantiates the corresponding `Agent` subclass
* Wraps logging via `Logger(cfg)`
* Dispatches to:

  * `train_offpolicy(agent, cfg, logger)` for DQN / DDQN / DDDQN / Rainbow
  * `train_onpolicy(agent, cfg, logger)` for REINFORCE / A2C / A3C / PPO / TRPO

Both trainers use **tqdm** to show progress over episodes.

### tmux-friendly script (`scripts/run_train.sh`)

```bash
#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# Optional: load wandb env vars from .env
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_DIR="${PWD}/logs/wandb"

mkdir -p logs checkpoints

ALGO=${1:-dddqn}
SEED=${2:-0}

python train.py --algo "${ALGO}" --seed "${SEED}" \
  2>&1 | tee "logs/train_${ALGO}_seed${SEED}.log"
```

Usage:

```bash
chmod +x scripts/run_train.sh

tmux new -s doomrl
CUDA_VISIBLE_DEVICES=0 ./scripts/run_train.sh dddqn 0
# detach: Ctrl+b, d
# attach: tmux attach -t doomrl
```

### Multi-algorithm / multi-GPU launcher (`scripts/launch_queue.py`)

`launch_queue.py` assigns different algorithms to different GPUs and runs them in parallel:

* Specify a job list: `JOBS = [(algo, seed), ...]`
* Specify available GPUs: `GPUS = ["1", "2", "3"]`
* Each job is launched as: `bash scripts/run_train.sh <algo> <seed>` with `CUDA_VISIBLE_DEVICES` set internally.

Example usage:

```bash
tmux new -s doomrl_queue
python scripts/launch_queue.py
```

---

## Evaluation

### Single-checkpoint evaluation

```bash
conda activate doomrl
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

python eval.py \
  --algo dddqn \
  --checkpoint checkpoints/dddqn_seed0_ep001500.pth \
  --episodes 10
```

`eval.py`:

* Loads `cfg = make_config(algo, seed)` and disables wandb
* Builds the agent and loads weights (`.pth` checkpoint)
* Creates the Deadly Corridor environment
* Runs deterministic evaluation episodes (`deterministic=True` in `agent.act`)
* Prints per-episode return and mean return

The script also supports **repeated evaluation**:

* `--chunk_episodes`: number of episodes per chunk (e.g., 30)
* `--n_chunks`: number of chunks (e.g., 10)

This yields 10 mean returns over 30 episodes each, useful for more stable comparison.

---

## Scope & Extensibility

This repository is intended as a **simple, comparable setup** for:

* Testing classic value-based methods (DQN family, Rainbow-lite)
* Comparing policy-gradient / actor-critic methods (REINFORCE, A2C/A3C, PPO, TRPO)
* Running controlled experiments on the **same Doom task**, with unified logging and preprocessing

Adding a new algorithm typically means:

1. Implementing a new `Agent` class under `rldoom/agents/`.
2. Adding a corresponding entry under `algos:` in `rldoom/configs/deadly_corridor.yaml`.

---

## Reference Hardware (used in our experiments)

Our experiments were mainly run on a **NAS server** with:

* **OS:** Ubuntu 22.04 LTS
* **CPU/RAM:** multi-core CPU with several 512 GB RAM
* **GPUs:**

  * 2 × NVIDIA RTX A5000 (24 GB each)
  * 1 × NVIDIA GeForce RTX 3090 (24 GB)

The code itself does **not** depend on this exact setup: any machine with a recent GPU (or even CPU, at slower speed) and a working VizDoom installation should be able to run the experiments.

---

## Credits & References

This project builds on:

* Thomas Simonini, **“Dueling Double Deep Q-Learning with PER — Doom Deadly Corridor”**
* Mnih et al., **“Human-level control through deep reinforcement learning,”** *Nature*, 2015.
* Van Hasselt et al., **“Deep Reinforcement Learning with Double Q-learning,”** AAAI, 2016.
* Wang et al., **“Dueling Network Architectures for Deep Reinforcement Learning,”** ICML, 2016.
* Schaul et al., **“Prioritized Experience Replay,”** ICLR, 2016.
* Mnih et al., **“Asynchronous Methods for Deep Reinforcement Learning,”** ICML, 2016. (A3C)
* Schulman et al., **“Trust Region Policy Optimization,”** ICML, 2015.
* Schulman et al., **“Proximal Policy Optimization Algorithms,”** arXiv, 2017.
* VizDoom: **“ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning.”**

All original Doom assets belong to their respective copyright holders.

---

## Contributors

<table>
  <tr>
    <td align="center" valign="top" width="160">
      <a href="https://github.com/LeeChangmin0310">
        <img src="https://github.com/LeeChangmin0310.png?size=120" width="96" height="96" alt="LeeChangmin0310 avatar"/><br/>
        <sub><b>Changmin Lee</b></sub><br/>
        <sub>@LeeChangmin0310</sub><br/>
        <sub>Maintainer</sub>
      </a>
    </td>
    <td align="center" valign="top" width="160">
      <a href="https://github.com/suyeonmyeong">
        <img src="https://github.com/suyeonmyeong.png?size=120" width="96" height="96" alt="suyeonmyeong avatar"/><br/>
        <sub><b>Suyeon Myung</b></sub><br/>
        <sub>@suyeonmyeong</sub><br/>
        <sub>Core Contributor</sub>
      </a>
    </td>
    <td align="center" valign="top" width="160">
      <a href="https://github.com/maeng00">
        <img src="https://github.com/maeng00.png?size=120" width="96" height="96" alt="maeng00 avatar"/><br/>
        <sub><b>Ui-Hyun Maeng</b></sub><br/>
        <sub>@maeng00</sub><br/>
        <sub>Core Contributor</sub>
      </a>
    </td>
  </tr>
</table>
