# Monte Carlo Policy Gradient (REINFORCE) Agent for Doom Environment
# rldoom/agents/reinforce.py

from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rldoom.agents.base import Agent
from rldoom.models.cnn_backbone import DoomCNN
from rldoom.models.heads import PolicyHead


class ReinforceAgent(Agent):
    """Monte Carlo policy gradient (REINFORCE) with CNN policy network."""

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

        # Per-episode storage
        self.log_probs = []
        self.rewards = []
        self._done = False

    def on_episode_start(self):
        """Reset trajectory storage at the beginning of each episode."""
        self.log_probs = []
        self.rewards = []
        self._done = False

    def _distribution(self, obs_t: torch.Tensor):
        """Build categorical distribution over actions given observation."""
        feat = self.backbone(obs_t)
        logits = self.policy_head(feat)
        return torch.distributions.Categorical(logits=logits)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Sample an action (or take argmax if deterministic) and
        store log-prob for REINFORCE updates.
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        dist = self._distribution(obs_t)

        if deterministic:
            # Greedy action for evaluation
            action = int(dist.probs.argmax(dim=1).item())
            return action

        # Stochastic action for training
        action_t = dist.sample()  # shape: (1,)
        log_prob = dist.log_prob(action_t)[0]  # scalar
        self.log_probs.append(log_prob)

        action = int(action_t.item())
        return action

    def observe(self, transition):
        """
        Receive one transition:
        (obs, action, reward, next_obs, done).
        We only need reward and done flag for REINFORCE.
        """
        obs, action, reward, next_obs, done = transition
        self.rewards.append(float(reward))
        if done:
            # Mark the episode as finished so that update() will run once
            self._done = True

    def update(self) -> Dict[str, float]:
        """
        After an episode is finished (self._done=True),
        compute Monte Carlo returns and apply REINFORCE update.
        """
        if not self._done:
            return {}

        if len(self.log_probs) == 0 or len(self.rewards) == 0:
            # Nothing to update (should not normally happen)
            self._done = False
            self.log_probs = []
            self.rewards = []
            return {}

        # 1) Compute discounted returns G_t
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.as_tensor(returns, device=self.device, dtype=torch.float32)

        # 2) Normalize returns for better stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3) Stack log-probs
        log_probs = torch.stack(self.log_probs)  # shape: (T,)

        # 4) Policy gradient loss: -E[ G_t * log pi(a_t|s_t) ]
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.backbone.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        # 5) Clear episode storage
        self.log_probs = []
        self.rewards = []
        self._done = False

        return {"loss": float(loss.item())}
