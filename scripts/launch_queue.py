# scripts/launch_queue.py
import os
import time
import subprocess

# Jobs to run: (algo, seed)
JOBS = [
    ("dqn", 0),
    ("dddqn", 0),
    ("rainbow", 0),
    ("reinforce", 0),
    ("a2c", 0),
    ("a3c", 0),
    ("ppo", 0),
    ("trpo", 0),
]

# GPUs you want to use
GPUS = ["1", "2", "3"]  # cuda:1,2,3

PYTHON_BIN = "python"  # or full path
PROJECT_ROOT = "/home/cia/disk1/bci_intern/AAAI2026/RLDoom"

running = []  # list of (process, gpu, algo, seed)

def launch_job(algo, seed, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # load .env inside run_train.sh, so just call that
    cmd = ["bash", "scripts/run_train.sh", algo, str(seed)]
    print(f"[LAUNCH] algo={algo} seed={seed} on GPU={gpu}")
    p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)
    return p

def main():
    jobs = list(JOBS)
    while jobs or running:
        # fill free GPUs
        while jobs and len(running) < len(GPUS):
            algo, seed = jobs.pop(0)
            # pick a free gpu index
            used_gpus = {gpu for _, gpu, _, _ in running}
            free_gpus = [g for g in GPUS if g not in used_gpus]
            if not free_gpus:
                break
            gpu = free_gpus[0]
            p = launch_job(algo, seed, gpu)
            running.append((p, gpu, algo, seed))

        # check running processes
        still_running = []
        for p, gpu, algo, seed in running:
            ret = p.poll()
            if ret is None:
                still_running.append((p, gpu, algo, seed))
            else:
                print(f"[DONE] algo={algo} seed={seed} on GPU={gpu} exit_code={ret}")
        running[:] = still_running

        if jobs or running:
            time.sleep(10)

    print("All jobs finished.")

if __name__ == "__main__":
    main()
