from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistrEncoder(nn.Module):
    """Dual-branch distribution encoder with FiLM fusion.

    Branch A (shape): encodes normalized quantiles -- captures modality,
    tail heaviness, skewness.
    Branch B (location-scale): encodes [log1p(|mean|), sign(mean), log(sigma)]
    -- captures where and how wide the distribution is.

    Fusion: Branch B modulates Branch A via FiLM (gamma * shape + beta).
    """

    def __init__(self, K: int, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.shape_encoder = nn.Sequential(
            nn.Linear(K, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.loc_scale_encoder = nn.Sequential(
            nn.Linear(3, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim // 2),
            nn.GELU(),
        )
        self.film_proj = nn.Linear(dim // 2, 2 * dim)

    def forward(self, quantiles: torch.Tensor, loc_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantiles: [B, K] normalized to [-1, 1].
            loc_scale: [B, 3] = [log1p(|mean|), sign(mean), log(sigma+eps)].

        Returns:
            [B, dim] distribution embedding.
        """
        shape_emb = self.shape_encoder(quantiles)
        ls_emb = self.loc_scale_encoder(loc_scale)
        gamma_beta = self.film_proj(ls_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * shape_emb + beta


class RandomnessEncoder(nn.Module):
    """Encodes a scalar Gaussian noise sample into a latent vector."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, 1] ~ N(0,1) -> [B, dim]."""
        return self.net(z)


class _FiLMMLPBlock(nn.Module):
    """Single MLP layer with FiLM conditioning from a condition vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.film_proj = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.ln(x)
        gamma, beta = self.film_proj(cond).chunk(2, dim=-1)
        return F.gelu(gamma * x + beta)


class Generator(nn.Module):
    """Conditional generator: (distribution description, noise) -> sample.

    Architecture:
        noise -> RandomnessEncoder -> FiLM-MLP(layer1, cond=distr_emb)
        -> FiLM-MLP(layer2, cond=distr_emb) -> Linear -> scalar sample.
    """

    def __init__(self, dim: int, K: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.distr_encoder = DistrEncoder(K, dim, dropout)
        self.randomness_encoder = RandomnessEncoder(dim, dropout)
        self.block1 = _FiLMMLPBlock(dim)
        self.block2 = _FiLMMLPBlock(dim)
        self.output_head = nn.Linear(dim, 1)

    def forward(
        self,
        quantiles: torch.Tensor,
        loc_scale: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            quantiles: [B, K]
            loc_scale: [B, 3]
            noise: [B, 1] ~ N(0, 1)

        Returns:
            [B, 1] generated sample value.
        """
        distr_emb = self.distr_encoder(quantiles, loc_scale)
        x = self.randomness_encoder(noise)
        x = self.block1(x, distr_emb)
        x = self.block2(x, distr_emb)
        return self.output_head(x)


class Critic(nn.Module):
    """Conditional critic: (distribution description, sample) -> scalar score.

    Uses a **shared** DistrEncoder (passed in from Generator).
    """

    def __init__(self, dim: int, distr_encoder: DistrEncoder, dropout: float = 0.0) -> None:
        super().__init__()
        self.distr_encoder = distr_encoder  # shared reference
        self.sample_encoder = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim),
        )
        self.block1 = _FiLMMLPBlock(dim)
        self.block2 = _FiLMMLPBlock(dim)
        self.output_head = nn.Linear(dim, 1)

    def forward(
        self,
        quantiles: torch.Tensor,
        loc_scale: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            quantiles: [B, K]
            loc_scale: [B, 3]
            sample: [B, 1] real or fake sample value.

        Returns:
            [B, 1] critic score.
        """
        distr_emb = self.distr_encoder(quantiles, loc_scale)
        x = self.sample_encoder(sample)
        x = self.block1(x, distr_emb)
        x = self.block2(x, distr_emb)
        return self.output_head(x)
