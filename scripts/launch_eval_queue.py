# scripts/launch_eval_queue.py
import os
import time
import subprocess

# Baseline evaluation jobs: (algo, seed, ep_str)
BASE_EVAL_JOBS = [
    ("dqn", 0, "002000"),
    ("ddqn", 0, "002000"),
    ("dddqn", 0, "002000"),
    ("rainbow", 0, "002000"),
    ("reinforce", 0, "002000"),
    ("a2c", 0, "002000"),
    ("ppo", 0, "002000"),
    ("trpo", 0, "002000"),
    # ("a3c", 0, "002000"),  # if you have A3C checkpoints
]

# Tuned variants evaluation jobs
TUNED_EVAL_JOBS = [
    ("reinforce_tuned", 0, "003000"),
    ("a2c_tuned", 0, "003000"),
    ("dddqn_tuned", 0, "003000"),
    ("ppo_tuned", 0, "003000"),
]

# Merge into a single queue (edit freely)
JOBS = BASE_EVAL_JOBS + TUNED_EVAL_JOBS

# GPUs you want to use
GPUS = ["1", "2", "3"]  # cuda:1 (3090), cuda:2 (A5000), cuda:3 (A5000)

PROJECT_ROOT = "/home/cia/disk1/bci_intern/AAAI2026/RLDoom"

# running: list of (process, gpu, algo, seed, ep_str)
running = []


def launch_job(algo: str, seed: int, ep_str: str, gpu: str):
    """Launch a single evaluation job on a specific GPU."""
    env = os.environ.copy()
    # Pin this process to the chosen GPU
    env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = ["bash", "scripts/run_eval.sh", algo, str(seed), ep_str]
    print(f"[LAUNCH_EVAL] algo={algo} seed={seed} ep={ep_str} on GPU={gpu}")

    p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)
    return p


def main():
    jobs = list(JOBS)

    while jobs or running:
        # Fill free GPUs with new jobs
        while jobs and len(running) < len(GPUS):
            algo, seed, ep_str = jobs.pop(0)

            used_gpus = {gpu for _, gpu, _, _, _ in running}
            free_gpus = [g for g in GPUS if g not in used_gpus]
            if not free_gpus:
                break

            gpu = free_gpus[0]
            p = launch_job(algo, seed, ep_str, gpu)
            running.append((p, gpu, algo, seed, ep_str))

        # Check running processes
        still_running = []
        for p, gpu, algo, seed, ep_str in running:
            ret = p.poll()
            if ret is None:
                still_running.append((p, gpu, algo, seed, ep_str))
            else:
                print(
                    f"[DONE_EVAL] algo={algo} seed={seed} ep={ep_str} "
                    f"on GPU={gpu} exit_code={ret}"
                )
        running[:] = still_running

        if jobs or running:
            time.sleep(10)

    print("All eval jobs finished.")


if __name__ == "__main__":
    main()
