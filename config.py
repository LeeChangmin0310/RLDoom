# config.py

import os


class Config:
    """Configuration for Doom DDDQN training."""

    # Doom config paths (place cfg and wad in project root)
    config_path = "deadly_corridor.cfg"
    scenario_path = "deadly_corridor.wad"

    # Frame & stack settings
    frame_height = 100
    frame_width = 120
    stack_size = 4

    # Training hyperparameters
    learning_rate = 2.5e-4
    gamma = 0.95

    total_episodes = 1000
    max_steps_per_episode = 3000

    # Epsilon-greedy
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 5e-5  # exponential decay rate

    # Replay memory
    memory_size = 100000
    pretrain_length = 50000
    learn_start = 1000

    # Target network
    target_update_freq = 1000  # in steps
    tau = 1.0  # 1.0 => hard update, <1.0 => soft update

    # Optimization
    grad_clip = 10.0
    batch_size = 64

    # Logging & checkpoints
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    logs_dir = os.path.join(base_dir, "logs")
    checkpoint_interval = 50  # in episodes

    # wandb
    wandb_project = "doom-dddqn"
    wandb_run_name = "dddqn_deadly_corridor"
