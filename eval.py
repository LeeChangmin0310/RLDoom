# eval.py
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


def build_agent(algo, obs_shape, num_actions, cfg, device):
    """Factory function that instantiates the correct agent class.
    Tuned variants (xxx_tuned) share the same architecture as their base versions.
    """
    # Map tuned algo names to their base architectures
    base_algo = algo
    if algo.endswith("_tuned"):
        base_algo = algo.replace("_tuned", "")

    if base_algo == "dqn":
        return DQNAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "ddqn":
        return DDQNAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "dddqn":
        return DDDQNAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "rainbow":
        return RainbowAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "reinforce":
        return ReinforceAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "a2c":
        return A2CAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "a3c":
        return A3CAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "ppo":
        return PPOAgent(obs_shape, num_actions, cfg, device)
    if base_algo == "trpo":
        return TRPOAgent(obs_shape, num_actions, cfg, device)
    raise ValueError(f"Unknown algo (base or tuned): {algo}")


def run_episodes(env, agent, algo_name, num_episodes, use_tqdm=True, desc_suffix=""):
    """Run num_episodes evaluation episodes.

    Returns:
        rewards: list of per-episode total rewards
        mean_r: average reward over all episodes
    """
    rewards = []

    if use_tqdm:
        iterator = trange(
            1,
            num_episodes + 1,
            desc=f"{algo_name} eval{desc_suffix}",
            dynamic_ncols=True,
        )
    else:
        iterator = range(1, num_episodes + 1)

    for ep in iterator:
        obs = env.reset()
        done = False
        ep_reward = 0.0

        # Deterministic evaluation: greedy / no epsilon exploration
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        print(f"[EVAL] algo={algo_name} episode={ep} reward={ep_reward:.2f}")

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"[EVAL] algo={algo_name} mean_reward={mean_r:.2f} "
        f"over {len(rewards)} episodes"
    )
    return rewards, mean_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=[
            # baseline
            "dqn", "ddqn", "dddqn", "rainbow",
            "reinforce", "a2c", "a3c", "ppo", "trpo",
            # tuned
            "reinforce_tuned", "a2c_tuned", "dddqn_tuned", "ppo_tuned",
        ],
        help="Algorithm name (baseline or tuned).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to build config and env.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for the simple evaluation (first pass).",
    )
    parser.add_argument(
        "--chunk_episodes",
        type=int,
        default=30,
        help="Number of episodes in each repeated-eval chunk.",
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=10,
        help="Number of repeated-eval chunks (each averages chunk_episodes episodes).",
    )
    args = parser.parse_args()

    # Build config and disable wandb in eval
    cfg = make_config(args.algo, args.seed)
    cfg.use_wandb = False

    # Seeding and device
    set_seed(cfg.seed)
    # Device is chosen by CUDA_VISIBLE_DEVICES from outside (multi-GPU support)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Observation shape and number of actions
    obs_shape = (cfg.stack_size, cfg.frame_size, cfg.frame_size)
    num_actions = 7  # Deadly Corridor action space size

    # Build agent and load checkpoint
    agent = build_agent(args.algo, obs_shape, num_actions, cfg, device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if hasattr(agent, "load_state_dict"):
        agent.load_state_dict(state_dict)
    else:
        agent.load(args.checkpoint)

    # Create environment
    env = DeadlyCorridorEnv(
        cfg_path=cfg.cfg_path,
        wad_path=cfg.wad_path,
        frame_size=cfg.frame_size,
        stack_size=cfg.stack_size,
        frame_skip=cfg.frame_skip,
    )

    print("========== Simple evaluation ==========")
    _, mean_simple = run_episodes(
        env,
        agent,
        algo_name=args.algo,
        num_episodes=args.episodes,
        use_tqdm=True,
        desc_suffix=f"_{args.episodes}ep",
    )
    print(
        f"[SUMMARY] algo={args.algo} simple_mean_reward={mean_simple:.2f} "
        f"({args.episodes} episodes)"
    )

    print("========== Repeated evaluation: "
          f"{args.chunk_episodes} episodes x {args.n_chunks} chunks ==========")

    chunk_means = []
    for rep in range(1, args.n_chunks + 1):
        print(f"\n[CHUNK] {rep}/{args.n_chunks}: running {args.chunk_episodes} episodes...")
        _, mean_chunk = run_episodes(
            env,
            agent,
            algo_name=args.algo,
            num_episodes=args.chunk_episodes,
            use_tqdm=False,   # avoid nested tqdm bars
            desc_suffix=f"_chunk{rep}",
        )
        chunk_means.append(mean_chunk)
        print(
            f"[CHUNK RESULT] algo={args.algo} chunk={rep} "
            f"mean_reward={mean_chunk:.2f}"
        )

    # Print final summary of chunk means
    if chunk_means:
        overall_mean = sum(chunk_means) / len(chunk_means)
        print("\n========== Repeated-eval summary ==========")
        print(f"[SUMMARY] algo={args.algo} chunk_means (len={len(chunk_means)}):")
        for i, m in enumerate(chunk_means, start=1):
            print(f"  chunk {i:02d}: {m:.2f}")
        print(
            f"[SUMMARY] algo={args.algo} overall_mean_of_chunk_means="
            f"{overall_mean:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
