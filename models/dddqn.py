# models/dddqn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """Dueling DQN network for VizDoom frames."""

    def __init__(self, input_channels, num_actions):
        super(DuelingDQN, self).__init__()

        # Convolutional encoder
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)

        # For input (4, 100, 120), conv3 output is (128, 4, 5) => 128*4*5 = 2560
        conv_output_dim = 128 * 4 * 5

        # Value stream
        self.value_fc = nn.Linear(conv_output_dim, 512)
        self.value = nn.Linear(512, 1)

        # Advantage stream
        self.adv_fc = nn.Linear(conv_output_dim, 512)
        self.advantage = nn.Linear(512, num_actions)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass returning Q-values for each action."""
        # x: (B, C, H, W)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        x = x.view(x.size(0), -1)

        value = F.elu(self.value_fc(x))
        value = self.value(value)  # (B, 1)

        adv = F.elu(self.adv_fc(x))
        adv = self.advantage(adv)  # (B, A)

        adv_mean = adv.mean(dim=1, keepdim=True)
        q_values = value + (adv - adv_mean)
        return q_values
