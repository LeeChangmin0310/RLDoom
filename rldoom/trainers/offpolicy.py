# rldoom/trainers/offpolicy.py
from typing import Dict, Any
from tqdm import trange
import os
import shutil

from rldoom.envs import make_env


def train_offpolicy(agent, cfg, logger):
    """
    Generic off-policy training loop (DQN / DDQN / DDDQN / Rainbow).

    - One env instance per process.
    - Per-episode logging via Logger.
    - Periodic checkpoint saving using Agent.save().
    """
    env = make_env(cfg)
    global_step = 0

    # Ensure checkpoint directory exists
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for ep in trange(cfg.train_episodes, desc=f"{cfg.algo} train", dynamic_ncols=True):
        obs = env.reset()
        episode_return = 0.0
        episode_len = 0
        last_metrics: Dict[str, Any] = {}

        for t in range(cfg.max_steps_per_episode):
            # 1) Select action
            action = agent.act(obs, deterministic=False)

            # 2) Step environment
            next_obs, reward, done, info = env.step(action)

            # 3) Give transition to agent
            transition = (obs, action, reward, next_obs, done)
            agent.observe(transition)

            # 4) Try to update agent
            metrics = agent.update()
            if metrics:
                last_metrics = metrics

            # 5) Bookkeeping
            episode_return += reward
            episode_len += 1
            global_step += 1
            obs = next_obs

            if done:
                break

        # -------- logging --------
        log_dict: Dict[str, float] = {
            "return": episode_return,
            "length": episode_len,
            "global_step": float(global_step),
        }
        for k, v in last_metrics.items():
            log_dict[k] = float(v)

        logger.log_metrics(log_dict, step=ep)

        # -------- checkpointing --------
        # Save every cfg.checkpoint_interval episodes, and at the very end.
        ep_idx = ep + 1  # 1-based index for readability
        if (
            ep_idx % cfg.checkpoint_interval == 0
            or ep_idx == cfg.train_episodes
        ):
            ckpt_name = f"{cfg.algo}_seed{cfg.seed}_ep{ep_idx:06d}.pth"
            ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)

            # Agent.save() uses Agent.state_dict() internally
            agent.save(ckpt_path)

            # Also maintain a "latest" checkpoint
            latest_path = os.path.join(
                cfg.checkpoint_dir,
                f"{cfg.algo}_seed{cfg.seed}_latest.pth",
            )
            shutil.copy2(ckpt_path, latest_path)

    env.close()
    logger.close()
