from __future__ import annotations

import math
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn


@dataclass(slots=True)
class FlowModelConfig:
    latent_dim: int
    dataset_embedding_dim: int
    architecture_feature_dim: int
    d_model: int = 512
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_latent_tokens: int = 64


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / max(half, 1))
    angles = t.float().unsqueeze(-1) * frequency.unsqueeze(0) * max_period
    value = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dim % 2:
        value = torch.cat((value, torch.zeros_like(value[:, :1])), dim=-1)
    return value


def fixed_position_embedding(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32) / max(length - 1, 1)
    return timestep_embedding(positions, dim)


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.modulation(condition).chunk(6, dim=-1)
        hidden = self.norm1(x) * (1.0 + s1.unsqueeze(1)) + b1.unsqueeze(1)
        attended = self.attention(hidden, hidden, hidden, key_padding_mask=None if token_mask is None else ~token_mask, need_weights=False)[0]
        x = x + g1.unsqueeze(1) * attended
        hidden = self.norm2(x) * (1.0 + s2.unsqueeze(1)) + b2.unsqueeze(1)
        return x + g2.unsqueeze(1) * self.mlp(hidden)


class ConditionalVelocityTransformer(nn.Module):
    """One shared flow family for ours and WeightCLIP latent substrates.

    Input/output adapters are substrate-specific and counted by ``parameter_ledger``;
    all core transformer settings can therefore be equality-checked across flows.
    """

    def __init__(self, config: FlowModelConfig) -> None:
        super().__init__()
        if config.depth <= 0 or config.heads <= 0 or config.d_model % config.heads:
            raise ValueError("invalid transformer depth/head dimensions")
        self.config = config
        self.input_adapter = nn.Linear(config.latent_dim, config.d_model)
        self.output_adapter = nn.Linear(config.d_model, config.latent_dim)
        self.architecture_token_adapter = nn.Sequential(
            nn.Linear(config.architecture_feature_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        # Fixed positions keep the core parameter budget identical when codecs expose
        # different latent-token counts. Only latent input/output adapters may differ.
        self.register_buffer("position", fixed_position_embedding(config.max_latent_tokens, config.d_model), persistent=True)
        self.time_mlp = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))
        self.condition_mlp = nn.Sequential(
            nn.Linear(config.dataset_embedding_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.blocks = nn.ModuleList(
            AdaLNBlock(config.d_model, config.heads, config.mlp_ratio, config.dropout) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.d_model, 2 * config.d_model))
        nn.init.zeros_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        *,
        dataset_embedding: torch.Tensor,
        architecture_features: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(1)
            squeeze = True
        elif z_t.ndim == 3:
            squeeze = False
        else:
            raise ValueError(f"z_t must be [B,D] or [B,T,D], got {tuple(z_t.shape)}")
        batch, tokens, latent_dim = z_t.shape
        cfg = self.config
        if latent_dim != cfg.latent_dim or tokens > cfg.max_latent_tokens:
            raise ValueError(f"latent shape {tuple(z_t.shape)} incompatible with {asdict(cfg)}")
        if tuple(dataset_embedding.shape) != (batch, cfg.dataset_embedding_dim):
            raise ValueError("dataset embedding shape mismatch")
        if tuple(architecture_features.shape) != (batch, tokens, cfg.architecture_feature_dim):
            raise ValueError("architecture feature shape mismatch")
        if t.shape != (batch,):
            raise ValueError("t must have shape [batch]")
        if token_mask is not None:
            token_mask = token_mask.to(device=z_t.device, dtype=torch.bool)
            if tuple(token_mask.shape) != (batch, tokens):
                raise ValueError("token_mask shape mismatch")
        x = (
            self.input_adapter(z_t)
            + self.position[:tokens].unsqueeze(0)
            + self.architecture_token_adapter(architecture_features.to(dtype=z_t.dtype))
        )
        condition = self.time_mlp(timestep_embedding(t, cfg.d_model).to(dtype=x.dtype))
        condition = condition + self.condition_mlp(dataset_embedding.to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x, condition, token_mask)
        scale, shift = self.final_modulation(condition).chunk(2, dim=-1)
        x = self.final_norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = self.output_adapter(x)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).to(dtype=output.dtype)
        return output.squeeze(1) if squeeze else output

    def parameter_ledger(self) -> dict[str, int]:
        adapters = sum(p.numel() for module in (self.input_adapter, self.output_adapter) for p in module.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "input_output_adapters": adapters, "core": total - adapters}
