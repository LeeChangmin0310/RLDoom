# rldoom/agents/rainbow.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rldoom.agents.base import Agent
from rldoom.models.cnn_backbone import DoomCNN
from rldoom.models.heads import DuelingQHead
from rldoom.buffers.prioritized_replay import PrioritizedReplayBuffer


class RainbowAgent(Agent):
    """
    Simplified Rainbow:
    - Double DQN
    - Dueling architecture
    - Prioritized replay
    - n-step return (n_step=1 here for simplicity; can be extended).
    """

    def __init__(self, obs_shape, num_actions, cfg, device):
        super().__init__(obs_shape, num_actions, cfg, device)

        c, _, _ = obs_shape
        self.online_backbone = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.online_head = DuelingQHead(cfg.feature_dim, num_actions).to(device)

        self.target_backbone = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.target_head = DuelingQHead(cfg.feature_dim, num_actions).to(device)

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

        self.buffer = PrioritizedReplayBuffer(
            cfg.buffer_size,
            obs_shape,
            device,
            alpha=cfg.per_alpha,
            beta_start=cfg.per_beta_start,
            beta_frames=cfg.per_beta_frames,
        )

    def _epsilon(self) -> float:
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

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.online_backbone(obs_t)
            q = self.online_head(feat)
        return int(q.argmax(dim=1).item())

    def observe(self, transition):
        # (obs, action, reward, next_obs, done)
        obs, action, reward, next_obs, done = transition
        self.buffer.add(obs, action, reward, next_obs, done)


    def update(self):
        if self.buffer.size < self.learn_start:
            return {}

        obs, actions, rewards, next_obs, dones, indices, weights = self.buffer.sample(
            self.batch_size
        )

        # Online chooses actions
        q_online_next = self.online_head(self.online_backbone(next_obs))
        next_actions = q_online_next.argmax(dim=1, keepdim=True)

        # Target evaluates them
        with torch.no_grad():
            q_target_next = self.target_head(self.target_backbone(next_obs))
            q_next = q_target_next.gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * q_next

        q = self.online_head(self.online_backbone(obs))
        q = q.gather(1, actions.view(-1, 1)).squeeze(1)

        td_error = target - q
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_backbone.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        # Update PER priorities
        new_prios = td_error.detach().cpu().numpy()
        self.buffer.update_priorities(indices, new_prios)

        if self.global_step % self.update_target_every == 0:
            self.target_backbone.load_state_dict(self.online_backbone.state_dict())
            self.target_head.load_state_dict(self.online_head.state_dict())

        return {"loss": float(loss.item())}
