import math

import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.act(out + identity)
        return out


class TinyResNet(nn.Module):
    """
    Small ResNet for CIFAR-10 (32x32).

    base_channels=16, blocks=[2,2,2] is a good fast default.
    """

    def __init__(self, base_channels=16, blocks=(2, 2, 2), num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.in_ch = base_channels
        self.layer1 = self._make_layer(base_channels, blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(self, out_ch, n_blocks, stride):
        layers = [_BasicBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, n_blocks):
            layers.append(_BasicBlock(self.in_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TransformerActor(nn.Module):
    def __init__(self, n_params, max_action, log_std_min, log_std_max,
                 d_model=64, nhead=16, num_layers=3):
        super().__init__()
        self.n_params = n_params
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.embed = nn.Linear(4, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head_mean = nn.Linear(d_model, 1)
        self.head_logstd = nn.Linear(d_model, 1)

    def forward(self, state):
        # state: (B, 4*N) = [w, g, m, v] concatenated
        N = self.n_params
        assert state.size(1) == 4 * N
        w, g, m, v = torch.chunk(state, 4, dim=1)  # each (B, N)
        tokens = torch.stack([w, g, m, v], dim=-1)  # (B, N, 4)

        x = self.embed(tokens)                     # (B, N, d_model)
        y = self.encoder(x)                        # (B, N, d_model)

        mean = self.head_mean(y).squeeze(-1)       # (B, N)
        log_std = self.head_logstd(y).squeeze(-1)  # (B, N)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self(state)
        if deterministic:
            u = mean
        else:
            std = log_std.exp()
            eps = torch.randn_like(std)
            u = mean + std * eps

        a = torch.tanh(u)
        a_scaled = a * self.max_action

        log_prob = None
        if not deterministic:
            # log N(u; mean, std) - log|det d(tanh)/du|
            std = log_std.exp()
            log_prob_gauss = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
            log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
            log_prob_tanh = torch.log(1 - torch.tanh(u).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            log_prob = log_prob_gauss - log_prob_tanh

        return a_scaled, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)
