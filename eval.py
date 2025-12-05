import argparse
import os
import torch
from tqdm import trange

from rldoom.configs import make_config
from rldoom.utils.seeding import set_seed
from rldoom.envs.deadly_corridor import DeadlyCorridorEnv

from rldoom.agents.dqn import DQNAgent
from rldoom.agents.ddqn import DDQNAgent
from rldoom.agents.dddqn import DDDQNAgent
from rldoom.agents.rainbow import RainbowAgent
from rldoom.agents.reinforce import ReinforceAgent
from rldoom.agents.a2c import A2CAgent
from rldoom.agents.a3c import A3CAgent
from rldoom.agents.ppo import PPOAgent
from rldoom.agents.trpo import TRPOAgent

ALGOS_OFF = {"dqn", "ddqn", "dddqn", "rainbow"}
ALGOS_ON = {"reinforce", "a2c", "a3c", "ppo", "trpo"}


def build_agent(algo, obs_shape, num_actions, cfg, device):
    if algo == "dqn":
        return DQNAgent(obs_shape, num_actions, cfg, device)
    if algo == "ddqn":
        return DDQNAgent(obs_shape, num_actions, cfg, device)
    if algo == "rainbow":
        return RainbowAgent(obs_shape, num_actions, cfg, device)
    if algo == "reinforce":
        return ReinforceAgent(obs_shape, num_actions, cfg, device)
    if algo == "a2c":
        return A2CAgent(obs_shape, num_actions, cfg, device)
    if algo == "a3c":
        return A3CAgent(obs_shape, num_actions, cfg, device)
    if algo == "ppo":
        return PPOAgent(obs_shape, num_actions, cfg, device)
    if algo == "trpo":
        return TRPOAgent(obs_shape, num_actions, cfg, device)
    raise ValueError(f"Unknown algo: {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = make_config(args.algo, args.seed)
    cfg.use_wandb = False  # not in eval

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_shape = (cfg.stack_size, cfg.frame_size, cfg.frame_size)
    num_actions = 7

    agent = build_agent(args.algo, obs_shape, num_actions, cfg, device)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    if hasattr(agent, "load_state_dict"):
        agent.load_state_dict(state_dict)
    else:
        agent.load(args.checkpoint)

    env = DeadlyCorridorEnv(
        cfg_path=cfg.cfg_path,
        wad_path=cfg.wad_path,
        frame_size=cfg.frame_size,
        stack_size=cfg.stack_size,
        frame_skip=cfg.frame_skip,
    )

    rewards = []

    for ep in trange(1, args.episodes + 1, desc=f"{args.algo} eval", dynamic_ncols=True):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        print(f"[EVAL] algo={cfg.algo} episode={ep} reward={ep_reward:.2f}")

    mean_r = sum(rewards) / len(rewards)
    print(f"[EVAL] mean_reward={mean_r:.2f} over {len(rewards)} episodes")

    env.close()


if __name__ == "__main__":
    main()