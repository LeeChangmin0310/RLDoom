# Monte Carlo Policy Gradient (REINFORCE) Agent for Doom Environment
# non actor-critic version (pure policy gradient)
# rldoom/agents/reinforce.py

from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rldoom.agents.base import Agent
from rldoom.models.cnn_backbone import DoomCNN
from rldoom.models.heads import PolicyHead


class ReinforceAgent(Agent):
    """Monte Carlo policy gradient (REINFORCE, no value baseline)."""

    def __init__(self, obs_shape, num_actions, cfg, device):
        super().__init__(obs_shape, num_actions, cfg, device)

        c, _, _ = obs_shape
        self.backbone = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.policy_head = PolicyHead(cfg.feature_dim, num_actions).to(device)

        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) + list(self.policy_head.parameters()),
            lr=cfg.lr,
        )

        self.gamma = cfg.gamma
        self.ent_coef = getattr(cfg, "ent_coef", 0.0)

        self.log_probs: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self._done = False

    def _dist(self, obs_t: torch.Tensor):
        """Return categorical policy distribution."""
        feat = self.backbone(obs_t)
        logits = self.policy_head(feat)
        return torch.distributions.Categorical(logits=logits)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Sample action and store log_prob / entropy for MC update."""
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        dist = self._dist(obs_t)

        if deterministic:
            action_t = dist.probs.argmax(dim=1)
        else:
            action_t = dist.sample()

        log_prob = dist.log_prob(action_t)[0]
        entropy = dist.entropy()[0]

        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

        return int(action_t.item())

    def observe(self, transition):
        """
        Receive one transition: (obs, action, reward, next_obs, done).
        Only reward/done are needed since log_probs/entropies are stored in act().
        """
        _, _, reward, _, done = transition
        self.rewards.append(float(reward))
        if done:
            self._done = True

    def update(self) -> Dict[str, float]:
        """Perform one Monte Carlo REINFORCE update at the end of an episode."""
        if not self._done or len(self.rewards) == 0:
            return {}

        # 1) Compute returns G_t
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()

        returns_t = torch.as_tensor(
            returns, device=self.device, dtype=torch.float32
        )  # (T,)
        log_probs_t = torch.stack(self.log_probs)    # (T,)
        entropies_t = torch.stack(self.entropies)    # (T,)

        # 2) Normalize returns (variance reduction)
        returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # 3) Losses
        policy_loss = -(log_probs_t * returns_norm).mean()
        entropy = entropies_t.mean()
        loss = policy_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.backbone.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        # 4) Reset episode buffers
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()
        self._done = False

        return {
            "loss": float(loss.item()),                # total
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
        }
