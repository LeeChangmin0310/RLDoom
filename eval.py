# eval.py

import os
import argparse
import numpy as np
import torch
from tqdm import trange

from config import Config
from envs.doom_env import DoomEnv
from models.dddqn import DuelingDQN


def evaluate(ckpt_path, num_episodes=10):
    """Evaluate trained agent in Doom environment."""
    cfg = Config()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    env = DoomEnv(
        config_path=cfg.config_path,
        scenario_path=cfg.scenario_path,
        frame_height=cfg.frame_height,
        frame_width=cfg.frame_width,
        stack_size=cfg.stack_size,
    )
    num_actions = len(env.possible_actions)

    net = DuelingDQN(cfg.stack_size, num_actions).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()

    episode_rewards = []

    for ep in trange(num_episodes, desc="Evaluating"):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = net(state_t)
            action_idx = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done = env.step(action_idx)
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)
        print(
            "[EVAL] Episode {}/{} reward={:.2f}".format(
                ep + 1, num_episodes, total_reward
            )
        )

    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))
    print("[EVAL] mean_reward={:.2f} ± {:.2f}".format(mean_r, std_r))

    env.close()


if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(cfg.checkpoint_dir, "dddqn_last.pt"),
        help="checkpoint path (default: best checkpoint)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="number of evaluation episodes",
    )
    args = parser.parse_args()

    evaluate(ckpt_path=args.ckpt, num_episodes=args.episodes)
