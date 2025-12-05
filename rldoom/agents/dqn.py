# rldoom/agents/dqn.py
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import Agent
from rldoom.models.cnn_backbone import DoomCNN
from rldoom.models.heads import QHead
from rldoom.buffers.replay_buffer import ReplayBuffer


class DQNAgent(Agent):
    """Vanilla DQN for Doom Deadly Corridor."""

    def __init__(self, obs_shape, num_actions, cfg, device):
        super().__init__(obs_shape, num_actions, cfg, device)

        c, _, _ = obs_shape
        # Online Q-network
        self.online_backbone = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.online_head = QHead(cfg.feature_dim, num_actions).to(device)

        # Target Q-network
        self.target_backbone = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.target_head = QHead(cfg.feature_dim, num_actions).to(device)

        # Initialize target with online params
        self.target_backbone.load_state_dict(self.online_backbone.state_dict())
        self.target_head.load_state_dict(self.online_head.state_dict())

        self.optimizer = optim.Adam(
            list(self.online_backbone.parameters()) + list(self.online_head.parameters()),
            lr=cfg.lr,
        )

        self.gamma = cfg.gamma
        self.eps_start = cfg.eps_start
        self.eps_end = cfg.eps_end
        self.eps_decay = cfg.eps_decay
        self.global_step = 0

        self.update_target_every = cfg.target_update_every
        self.batch_size = cfg.batch_size
        self.learn_start = cfg.learn_start

        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_size,
            obs_shape=obs_shape,
            device=device,
        )

    def _epsilon(self) -> float:
        """Exponential epsilon schedule."""
        return float(
            self.eps_end
            + (self.eps_start - self.eps_end)
            * np.exp(-1.0 * self.global_step / self.eps_decay)
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection using online network."""
        self.global_step += 1
        eps = 0.0 if deterministic else self._epsilon()

        if np.random.rand() < eps:
            return int(np.random.randint(self.num_actions))

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, C, H, W)
        with torch.no_grad():
            feat = self.online_backbone(obs_t)
            q = self.online_head(feat)
        action = int(q.argmax(dim=1).item())
        return action

    def observe(self, transition):
        """Store transition in replay buffer."""
        obs, action, reward, next_obs, done = transition
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> Dict[str, float]:
        """One TD update step for DQN."""
        if self.buffer.size < self.learn_start:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        # Q(s,a) from online network
        q_all = self.online_head(self.online_backbone(obs))  # (B, A)
        q = q_all.gather(1, actions.view(-1, 1)).squeeze(1)  # (B,)

        # Target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_all = self.target_head(self.target_backbone(next_obs))
            next_q_max = next_q_all.max(dim=1)[0]
            target = rewards + self.gamma * (1.0 - dones) * next_q_max

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.online_backbone.parameters()) + list(self.online_head.parameters()),
            self.cfg.grad_clip,
        )
        self.optimizer.step()

        # Periodic target update
        if self.global_step % self.update_target_every == 0:
            self.target_backbone.load_state_dict(self.online_backbone.state_dict())
            self.target_head.load_state_dict(self.online_head.state_dict())

        return {
            "loss": float(loss.item()),          # total loss (same as value loss here)
            "value_loss": float(loss.item()),
        }

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "online_backbone": self.online_backbone.state_dict(),
            "online_head": self.online_head.state_dict(),
            "target_backbone": self.target_backbone.state_dict(),
            "target_head": self.target_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.online_backbone.load_state_dict(state_dict["online_backbone"])
        self.online_head.load_state_dict(state_dict["online_head"])
        self.target_backbone.load_state_dict(state_dict["target_backbone"])
        self.target_head.load_state_dict(state_dict["target_head"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.global_step = state_dict.get("global_step", 0)
