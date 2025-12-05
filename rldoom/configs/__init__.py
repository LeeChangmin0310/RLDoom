# rldoom/configs/__init__.py
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Simple attribute-style config object."""

    def __init__(self, data: Dict[str, Any]):
        # Attach all entries as attributes
        for k, v in data.items():
            setattr(self, k, v)

    def as_dict(self) -> Dict[str, Any]:
        """Return underlying dictionary (useful for logging)."""
        return self.__dict__


def make_config(algo: str, seed: int) -> Config:
    """
    Build a flat config from deadly_corridor.yaml and a given algorithm name.

    - env      -> cfg_path, wad_path, frame_size, stack_size, frame_skip
    - train    -> num_episodes, checkpoint_dir, checkpoint_interval, logs_dir
    - defaults -> feature_dim, gamma, grad_clip
    - logging  -> use_wandb, wandb_project, wandb_entity
    - algos[algo] -> lr, buffer_size, batch_size, ...
    """

    # deadly_corridor.yaml is located in the same directory as this file
    here = Path(__file__).resolve().parent
    yaml_path = here / "deadly_corridor.yaml"

    with open(yaml_path, "r") as f:
        root = yaml.safe_load(f)

    if "algos" not in root or algo not in root["algos"]:
        raise ValueError(f"Algorithm '{algo}' not found in {yaml_path}")

    algo_cfg = root["algos"][algo]

    cfg: Dict[str, Any] = {}

    # Basic meta fields
    cfg["algo"] = algo
    cfg["seed"] = int(seed)
    
    train_cfg = root.get("train", {})  
    
    # Flatten sections
    for section in ("env", "train", "defaults", "logging"):
        if section in root and isinstance(root[section], dict):
            cfg.update(root[section])

    cfg["train_episodes"] = train_cfg.get("num_episodes", 5000)
    cfg["max_steps_per_episode"] = train_cfg.get("max_steps_per_episode", 3000)
    cfg["checkpoint_dir"] = train_cfg.get("checkpoint_dir", "checkpoints")
    cfg["checkpoint_interval"] = train_cfg.get("checkpoint_interval", 200)
    cfg["logs_dir"] = train_cfg.get("logs_dir", "logs")

    # Algorithm type: offpolicy / onpolicy
    cfg["algo_type"] = algo_cfg.get("type", "offpolicy")

    # All other algo-specific hyperparameters
    for k, v in algo_cfg.items():
        if k == "type":
            continue
        cfg[k] = v

    return Config(cfg)
