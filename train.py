# train.py
import argparse
import os

import torch

from rldoom.configs import make_config
from rldoom.utils.logger import Logger
from rldoom.utils.seeding import set_seed

from rldoom.agents.dqn import DQNAgent
from rldoom.agents.ddqn import DDQNAgent
from rldoom.agents.dddqn import DDDQNAgent
from rldoom.agents.rainbow import RainbowAgent
from rldoom.agents.reinforce import ReinforceAgent
from rldoom.agents.a2c import A2CAgent
from rldoom.agents.a3c import A3CAgent
from rldoom.agents.ppo import PPOAgent
from rldoom.agents.trpo import TRPOAgent

from rldoom.trainers.offpolicy import train_offpolicy
from rldoom.trainers.onpolicy import train_onpolicy


def build_agent(algo: str, obs_shape, num_actions: int, cfg, device):
    """Factory method that maps algo string to the corresponding Agent class."""
    # --- Off-policy family ---
    if algo == "dqn":
        return DQNAgent(obs_shape, num_actions, cfg, device)
    if algo == "ddqn":
        return DDQNAgent(obs_shape, num_actions, cfg, device)
    if algo in ("dddqn", "dddqn_tuned"):
        # Tuned variant uses the same agent class but different hyperparameters in cfg.
        return DDDQNAgent(obs_shape, num_actions, cfg, device)
    if algo == "rainbow":
        return RainbowAgent(obs_shape, num_actions, cfg, device)

    # --- On-policy family ---
    if algo in ("reinforce", "reinforce_tuned"):
        return ReinforceAgent(obs_shape, num_actions, cfg, device)
    if algo in ("a2c", "a2c_tuned"):
        return A2CAgent(obs_shape, num_actions, cfg, device)
    if algo == "a3c":
        return A3CAgent(obs_shape, num_actions, cfg, device)
    if algo in ("ppo", "ppo_tuned"):
        return PPOAgent(obs_shape, num_actions, cfg, device)
    if algo == "trpo":
        return TRPOAgent(obs_shape, num_actions, cfg, device)

    raise ValueError(f"Unknown algorithm: {algo}")


def main():
    """Main training entrypoint.

    - Parses CLI arguments.
    - Builds config from YAML.
    - Sets random seeds and selects device.
    - Instantiates the requested Agent.
    - Dispatches to on-policy or off-policy trainer.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=[
            # baseline off-policy
            "dqn",
            "ddqn",
            "dddqn",
            "rainbow",
            # baseline on-policy
            "reinforce",
            "a2c",
            "a3c",
            "ppo",
            "trpo",
            # tuned variants
            "reinforce_tuned",
            "a2c_tuned",
            "dddqn_tuned",
            "ppo_tuned",
        ],
        help="Name of the RL algorithm to train.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Build config object from YAML (env + train + logging + algo hyperparams)
    cfg = make_config(args.algo, args.seed)

    # Set all random seeds (Python, NumPy, Torch, envs if needed)
    set_seed(cfg.seed)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Observation shape (C, H, W)
    obs_shape = (cfg.stack_size, cfg.frame_size, cfg.frame_size)
    # Deadly Corridor has 7 discrete actions
    num_actions = 7

    # Instantiate Agent
    agent = build_agent(cfg.algo, obs_shape, num_actions, cfg, device)

    # Logger wrapper (handles wandb + console)
    logger = Logger(cfg)

    # Dispatch to the appropriate trainer
    if cfg.algo_type == "offpolicy":
        train_offpolicy(agent, cfg, logger)
    elif cfg.algo_type == "onpolicy":
        train_onpolicy(agent, cfg, logger)
    else:
        raise ValueError(f"Unknown algo_type: {cfg.algo_type}")


if __name__ == "__main__":
    main()
