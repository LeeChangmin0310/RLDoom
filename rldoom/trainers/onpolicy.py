# rldoom/trainers/onpolicy.py
from typing import Dict, Any
from tqdm import trange
import os
import shutil

from rldoom.envs import make_env


def train_onpolicy(agent, cfg, logger):
    """
    Generic on-policy training loop
    (REINFORCE / A2C / A3C / PPO / TRPO).

    - Each episode collects transitions into agent's internal buffer.
    - After episode, agent.update() does policy/value updates.
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

        if hasattr(agent, "on_episode_start"):
            agent.on_episode_start()

        while True:
            # 1) Select action
            action = agent.act(obs, deterministic=False)

            # 2) Step environment
            next_obs, reward, done, info = env.step(action)

            # 3) Give transition to agent
            transition = (obs, action, reward, next_obs, done)
            agent.observe(transition)

            # 4) Bookkeeping
            episode_return += reward
            episode_len += 1
            global_step += 1
            obs = next_obs

            if done or episode_len >= cfg.max_steps_per_episode:
                break

        # -------- policy/value update --------
        metrics: Dict[str, Any] = agent.update()

        # -------- logging --------
        log_dict: Dict[str, float] = {
            "return": episode_return,
            "length": episode_len,
            "global_step": float(global_step),
        }
        for k, v in metrics.items():
            log_dict[k] = float(v)

        logger.log_metrics(log_dict, step=ep)

        # -------- checkpointing --------
        ep_idx = ep + 1
        if (
            ep_idx % cfg.checkpoint_interval == 0
            or ep_idx == cfg.train_episodes
        ):
            ckpt_name = f"{cfg.algo}_seed{cfg.seed}_ep{ep_idx:06d}.pth"
            ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
            agent.save(ckpt_path)

            latest_path = os.path.join(
                cfg.checkpoint_dir,
                f"{cfg.algo}_seed{cfg.seed}_latest.pth",
            )
            shutil.copy2(ckpt_path, latest_path)

    env.close()
    logger.close()
