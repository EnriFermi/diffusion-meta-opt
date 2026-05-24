from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class RealNVPConfig:
    dim: int
    num_layers: int = 8
    hidden_dim: int = 512
    network_depth: int = 2
    log_scale_clamp: float = 2.0
    dropout: float = 0.0


def _make_alternating_mask(dim: int, parity: int, *, device: torch.device | None = None) -> torch.Tensor:
    indices = torch.arange(int(dim), device=device, dtype=torch.float32)
    return ((indices.long() + int(parity)) % 2 == 0).to(dtype=torch.float32)


class AffineCouplingLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        network_depth: int,
        log_scale_clamp: float,
        mask_parity: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.log_scale_clamp = float(log_scale_clamp)
        self.register_buffer("mask", _make_alternating_mask(int(dim), int(mask_parity)), persistent=False)

        layers: list[nn.Module] = []
        in_dim = int(dim)
        depth = max(1, int(network_depth))
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        out = nn.Linear(in_dim, 2 * int(dim))
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def _shift_and_log_scale(self, fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(fixed).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.log_scale_clamp
        inv_mask = 1.0 - self.mask.to(device=fixed.device, dtype=fixed.dtype)
        return shift * inv_mask, log_scale * inv_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or int(x.shape[1]) != self.dim:
            raise ValueError(f"x must be [B,{self.dim}], got {tuple(x.shape)}")
        mask = self.mask.to(device=x.device, dtype=x.dtype)
        fixed = x * mask
        shift, log_scale = self._shift_and_log_scale(fixed)
        inv_mask = 1.0 - mask
        y = fixed + inv_mask * (x * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.ndim != 2 or int(y.shape[1]) != self.dim:
            raise ValueError(f"y must be [B,{self.dim}], got {tuple(y.shape)}")
        mask = self.mask.to(device=y.device, dtype=y.dtype)
        fixed = y * mask
        shift, log_scale = self._shift_and_log_scale(fixed)
        inv_mask = 1.0 - mask
        x = fixed + inv_mask * ((y - shift) * torch.exp(-log_scale))
        log_det = -log_scale.sum(dim=-1)
        return x, log_det


class RealNVPFlow(nn.Module):
    """Identity-initialized RealNVP-style flow for flat latent vectors."""

    def __init__(self, cfg: RealNVPConfig) -> None:
        super().__init__()
        if int(cfg.dim) <= 0:
            raise ValueError(f"flow dim must be positive, got {cfg.dim}")
        if int(cfg.num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {cfg.num_layers}")
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                AffineCouplingLayer(
                    dim=int(cfg.dim),
                    hidden_dim=int(cfg.hidden_dim),
                    network_depth=int(cfg.network_depth),
                    log_scale_clamp=float(cfg.log_scale_clamp),
                    mask_parity=idx % 2,
                    dropout=float(cfg.dropout),
                )
                for idx in range(int(cfg.num_layers))
            ]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z.new_zeros(int(z.shape[0]))
        out = z
        for layer in self.layers:
            out, layer_log_det = layer(out)
            log_det = log_det + layer_log_det
        return out, log_det

    def inverse(self, z_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z_flat.new_zeros(int(z_flat.shape[0]))
        out = z_flat
        for layer in reversed(self.layers):
            out, layer_log_det = layer.inverse(out)
            log_det = log_det + layer_log_det
        return out, log_det

