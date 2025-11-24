# RARE-3D: Reinforcement Learning–based Adaptive Path Selection for Efficient Point-Cloud Restoration

**TL;DR.** We modernize **PathNet** (point-wise path selection for denoising) by replacing its **LSTM + REINFORCE** routing with a **Transformer encoder + PPO (with GAE)**. We keep the original 2-stage training protocol and loss design, aiming for better stability, equal-or-better quality, and lower use of complex paths.

---

## What’s New

* **Routing agent:** LSTM → **Transformer Encoder** (2–4 layers, d=256, ≤4 heads)
* **RL algorithm:** REINFORCE → **PPO-Clip + GAE** (alt: **Discrete SAC** for comparison)
* **Training loop:** on-policy **alternating freezing** (update policy while freezing the restorer, then vice versa)

---

## Project Status

This repo is a fork/derivative of **PathNet**. We keep upstream scripts, datasets, and pretrained weights, and add our RL modules under `tools/` and `models/routing/`.

* Upstream PathNet (code/data): [https://github.com/ZeyongWei/PathNet](https://github.com/ZeyongWei/PathNet)
* Paper (TPAMI’24): Path-Selective Point Cloud Denoising

---

## Environment (Ubuntu 22.04 LTS + RTX 5000)

* **OS:** Ubuntu 22.04 LTS
* **GPU:** NVIDIA RTX 5000 (CUDA 12.x supported)
* **Python:** 3.10+
* **Core:** PyTorch 2.x, **either** TorchRL **or** CleanRL (pick one), PyTorch3D, Open3D

> PyTorch3D wheels must match your **PyTorch** and **CUDA** versions. Two safe paths are provided below.

### Option A — Official env (from upstream)

```bash
conda env create -f env.yml
conda activate pathnet
```

### Option B — Modern toolchain (CUDA 12.1 / PyTorch 2.1.2)

```bash
# Create env
conda create -n rare3d python=3.10 -y
conda activate rare3d

# PyTorch 2.1.2 (CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# PyTorch3D wheel matching torch==2.1.2 + cu121
pip install -U fvcore iopath
pip install pytorch3d -f \
  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt212/download.html

# Common libs
pip install open3d tensorboardX tqdm h5py numpy scipy

# RL stack (pick ONE)
pip install cleanrl==1.*          # or: pip install torchrl==0.*
```

**Sanity check**

```bash
python - << 'PY'
import torch, sys
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
try:
    import pytorch3d
    print("pytorch3d OK")
except Exception as e:
    print("pytorch3d import error:", e, file=sys.stderr)
PY
```

---

## Data & Pretrained (from PathNet)

Download **datasets** and **pretrained models** from the PathNet repository (Drive links). Place files as:

```
RARE-3D/
  data/
    train_data.hdf5
    test_data.zip  # unzip if required
```

We follow the original **2-stage** training protocol.

---

## Baselines (Repo-default vs Paper-config)

### A) Repo-default (as in the official README)

```bash
cd PathNet
# Stage 1 (random routing warm-up)
python train.py --epoch 200 --use_random_path 1
# Stage 2 (learn routing with REINFORCE)
python train.py --epoch 300 --use_random_path 0
```

### B) Paper-config (TPAMI’24, for faithful reproduction)

```bash
cd PathNet
# Stage 1
python train.py \
  --use_random_path 1 \
  --epoch 150 \
  --batch_size 64 \
  --lr 1e-6

# Stage 2
python train.py \
  --use_random_path 0 \
  --epoch 100 \
  --batch_size 64 \
  --lr 1e-6
```

> We report which profile (A or B) is used for each experiment.
> Restorer optimizer for paper-config: **Adam, lr=1e-6**, batch **64**.

---

## Our Method (Transformer Routing + PPO)

**Key idea:** Keep the denoiser/backbone and losses; swap the routing agent to a Transformer policy/value and train with PPO using **alternating freezing**.

```bash
# (A) Policy step — freeze restorer, update policy with PPO
python tools/ppo_trainer.py \
  --agent transformer \
  --horizon 6 \
  --clip 0.2 --epochs 6 --mb 16 \
  --gamma 0.99 --gae_lambda 0.95 \
  --entropy_coef 1e-3 --lr 3e-4

# (B) Restorer step — freeze policy, supervised update of the denoiser
python tools/train_restorer.py \
  --epochs 1 \
  --batch_size 64 \
  --lr 1e-6 \
  --loss cd+repulsion

# Repeat (A) and (B)
# IMPORTANT: Do not reuse rollouts across different restorer weights (keep on-policy).
```

Optional comparison (max-entropy exploration):

```bash
# Discrete SAC comparison run (example args)
python tools/sacd_trainer.py \
  --agent transformer --horizon 6 \
  --alpha auto --gamma 0.99 --target_tau 0.005
```

---

## Evaluation

```bash
# Chamfer Distance / F-score@1%
python tools/eval.py --split test --metrics cd fscore

# Report mean ± std over 3 runs (seed=1,2,3)
```

**Primary metrics:** Chamfer Distance (CD), MSE, F-score@1%
**Secondary:** complex-path utilization (%), inference time (ms), qualitative heatmaps

---

## Targets (Course Project)

* **Quality:** CD ↓ **≥3%** vs original PathNet baseline; F-score@1% **+2.0 pt**
* **Efficiency:** complex-path utilization **−20%** at equal quality; inference time **−15%**
* **Generalization:** Synth-A/B → Real-A/B drop ≤ **30%**

---

## Repository Structure (planned)

```
RARE-3D/
  data/                      # datasets (see PathNet)
  models/
    restorer/                # bypass / complex backbones (from PathNet)
    routing/
      transformer.py         # Transformer policy + value heads (ours)
  tools/
    ppo_trainer.py           # PPO loop with GAE & clipping (ours)
    sacd_trainer.py          # Discrete SAC trainer (optional; ours)
    train_restorer.py        # supervised updates for the denoiser (ours)
    eval.py                  # CD / F-score evaluation (ours)
  scripts/
    run_baseline.sh
    run_rare3d.sh
  README.md
  LICENSE
  CITATION.cff
  CONTRIBUTING.md
```

---

## Method Brief

* **State:** local patch features ⊕ previous action one-hot ⊕ block index (positional encoding)
* **Action:** select path (a_t \in {\text{bypass}, \text{complex}}) (extensible to (M>2))
* **Reward:** complexity penalty at intermediate steps; noise/geometry-aware improvement at the final step (kept from PathNet)
* **Losses (restorer):** squared nearest-neighbor distance (L_d) + repulsion (L_r)
* **PPO defaults:** clip 0.1–0.2, epochs 4–8, minibatch 8–16, (\gamma=0.99), (\lambda_{\text{GAE}}=0.95), entropy 1e-3, lr 3e-4, max-grad-norm 0.5

---

## How We Train Stably (Alternating Freezing)

1. **Policy step:** freeze the restorer, collect fresh on-policy rollouts, compute GAE, run PPO updates.
2. **Restorer step:** freeze the policy, use the policy’s chosen paths to supervise the denoiser with (L_d + \lambda_r L_r).
3. **Repeat:** never mix rollouts across different restorer weights.

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


---

## Citing & License

This work builds upon **PathNet** (MIT License). Please cite the original authors.

* PathNet repo: [https://github.com/ZeyongWei/PathNet](https://github.com/ZeyongWei/PathNet)
* PathNet paper: “Path-Selective Point Cloud Denoising,” TPAMI 2024

Our additions are released under **MIT License**. Upstream copyright and notices are preserved.

---

## References

* Schulman et al., “Proximal Policy Optimization Algorithms,” 2017
* Schulman et al., “Generalized Advantage Estimation,” 2016
* TorchRL PPO Tutorial — [https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
* CleanRL — [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
* PyTorch3D Chamfer loss — [https://pytorch3d.readthedocs.io/en/latest/modules/loss.html](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html)

---
