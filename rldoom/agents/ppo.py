# rldoom/agents/ppo.py
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


class PPOAgent(Agent):
    """PPO-Clip with shared CNN backbone and rollout buffer."""

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
        self.lam = cfg.gae_lambda
        self.ent_coef = cfg.ent_coef
        self.vf_coef = cfg.vf_coef
        self.clip_range = cfg.clip_range
        self.rollout_len = cfg.rollout_len
        self.ppo_epochs = cfg.ppo_epochs
        self.batch_size = cfg.batch_size

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
        
    def _compute_gae(self, rewards, values, dones, last_value):
        """Compute GAE-Lambda advantages."""
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * (1.0 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self) -> Dict[str, float]:
        if not self.buffer.is_ready():
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.get()

        with torch.no_grad():
            logits, values = self._forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions)

            last_obs = next_obs[-1].unsqueeze(0)
            _, last_value = self._forward(last_obs)
            last_value = last_value.squeeze(0)

            advantages, returns = self._compute_gae(rewards, values, dones, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_all = obs
        actions_all = actions
        old_log_probs_all = old_log_probs.detach()
        returns_all = returns.detach()
        advantages_all = advantages.detach()

        total_steps = obs_all.shape[0]
        batch_size = self.batch_size

        # If rollout shorter than one batch, skip update
        if total_steps < batch_size:
            self.buffer.reset()
            return {}

        for _ in range(self.ppo_epochs):
            idxs = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, batch_size):
                end = start + batch_size
                if end > total_steps:
                    break
                mb_idx = idxs[start:end]

                mb_obs = obs_all[mb_idx]
                mb_actions = actions_all[mb_idx]
                mb_advantages = advantages_all[mb_idx]
                mb_returns = returns_all[mb_idx]
                mb_old_log_probs = old_log_probs_all[mb_idx]

                logits, values_mb = self._forward(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_mb, mb_returns)

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



