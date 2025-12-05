# rldoom/agents/dqn.py
from typing import Dict, Tuple, Any
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

        c, h, w = obs_shape
        self.net = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.q_head = QHead(cfg.feature_dim, num_actions).to(device)

        self.target_net = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.target_q_head = QHead(cfg.feature_dim, num_actions).to(device)

        self.target_net.load_state_dict(self.net.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())

        self.optimizer = optim.Adam(
            list(self.net.parameters()) + list(self.q_head.parameters()),
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
        self.global_step += 1
        eps = 0.0 if deterministic else self._epsilon()

        if np.random.rand() < eps:
            return int(np.random.randint(self.num_actions))

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1,C,H,W)
        with torch.no_grad():
            feat = self.net(obs_t)
            q = self.q_head(feat)
        action = int(q.argmax(dim=1).item())
        return action

    def observe(self, transition):
        """Store transition in replay buffer."""
        # Before: obs, action, reward, next_obs, done, _ = transition
        obs, action, reward, next_obs, done = transition
        self.buffer.add(obs, action, reward, next_obs, done)


    def update(self) -> Dict[str, float]:
        if self.buffer.size < self.learn_start:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        q = self.q_head(self.net(obs))                # (B,A)
        q = q.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_q_head(self.target_net(next_obs))
            next_q_max = next_q.max(dim=1)[0]
            target = rewards + self.gamma * (1.0 - dones) * next_q_max

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        if self.global_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_q_head.load_state_dict(self.q_head.state_dict())

        return {"loss": float(loss.item())}
