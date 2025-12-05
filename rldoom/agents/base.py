# rldoom/agents/base.py
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim


class Agent(ABC):
    """Base class for all RL agents."""

    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, cfg: Any, device: torch.device):
        """
        Args:
            obs_shape: Shape of observation (C, H, W).
            num_actions: Number of discrete actions.
            cfg: Config namespace.
            device: torch.device.
        """
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.cfg = cfg
        self.device = device

    # -------- core interface --------
    @abstractmethod
    def act(self, obs, deterministic: bool = False) -> int:
        """Given a single observation (np.ndarray), return an action index."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, transition) -> None:
        """
        Receive one environment transition.
        In this project we use:
        (obs, action, reward, next_obs, done).
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        Perform one learning update if ready.
        Returns a dict of metrics (e.g. loss).
        If no update is performed, can return {}.
        """
        raise NotImplementedError

    # -------- saving / loading --------
    def _modules_and_optimizers(self):
        """Collect all nn.Modules and optim.Optimizers attached to this agent."""
        modules = {}
        optimizers = {}
        for name, value in self.__dict__.items():
            if isinstance(value, nn.Module):
                modules[name] = value
            elif isinstance(value, optim.Optimizer):
                optimizers[name] = value
        return modules, optimizers

    def state_dict(self) -> Dict[str, Any]:
        """Pack all modules and optimizers into a single dict."""
        modules, optimizers = self._modules_and_optimizers()
        return {
            "modules": {k: v.state_dict() for k, v in modules.items()},
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            "cfg": getattr(self.cfg, "__dict__", {}),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state into modules and optimizers (best-effort)."""
        modules, optimizers = self._modules_and_optimizers()

        for k, sd in state.get("modules", {}).items():
            if k in modules:
                modules[k].load_state_dict(sd)

        for k, sd in state.get("optimizers", {}).items():
            if k in optimizers:
                optimizers[k].load_state_dict(sd)

    def save(self, path: str) -> None:
        """Save everything to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load everything from a file."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)
