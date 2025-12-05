# rldoom/agents/a3c.py
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rldoom.agents.base import Agent
from rldoom.models.cnn_backbone import DoomCNN
from rldoom.models.heads import PolicyHead, ValueHead
from rldoom.buffers.rollout_buffer import RolloutBuffer


class A3CAgent(Agent):
    """
    Synchronous A3C-like agent:
    - Same architecture as A2C
    - Uses small rollouts and more frequent updates
    """

    def __init__(self, obs_shape, num_actions, cfg, device):
        super().__init__(obs_shape, num_actions, cfg, device)

        c, _, _ = obs_shape
        self.backbone = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.policy_head = PolicyHead(cfg.feature_dim, num_actions).to(device)
        self.value_head = ValueHead(cfg.feature_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.backbone.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters()),
            lr=cfg.lr_actor,
        )

        self.gamma = cfg.gamma
        self.ent_coef = cfg.ent_coef
        self.vf_coef = cfg.vf_coef
        self.rollout_len = cfg.rollout_len

        self.buffer = RolloutBuffer(self.rollout_len, obs_shape, num_actions, device)

    def _forward(self, obs_t: torch.Tensor):
        feat = self.backbone(obs_t)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self._forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = int(dist.probs.argmax(dim=1).item())
            else:
                action = int(dist.sample().item())
        return action

    def observe(self, transition):
        """Store transition in replay buffer."""
        obs, action, reward, next_obs, done = transition
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> Dict[str, float]:
        if not self.buffer.is_ready():
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.get()

        logits, values = self._forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        with torch.no_grad():
            last_obs = next_obs[-1].unsqueeze(0)
            _, last_value = self._forward(last_obs)
            last_value = last_value.squeeze(0)

        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - d)
            returns.append(R)
        returns = torch.stack(list(reversed(returns)))
        advantages = returns - values.detach()

        policy_loss = -(advantages * log_probs).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.backbone.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.buffer.reset()

        return {
            "loss": float(loss.item()),                 # total
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "backbone": self.backbone.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.backbone.load_state_dict(state_dict["backbone"])
        self.policy_head.load_state_dict(state_dict["policy_head"])
        self.value_head.load_state_dict(state_dict["value_head"])
        self.optimizer.load_state_dict(state_dict["optimizer"])