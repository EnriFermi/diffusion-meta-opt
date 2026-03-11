from __future__ import annotations

import torch
import torch.nn as nn


class ResidualPatchTokenizer(nn.Module):
    """
    Simple FC tokenizer with strong residual from raw weights.

    At initialization, approximately passes through w_patch as-is,
    giving the big-VAE transformer a reasonable starting point.

    Input:  w_patch [B, p], dist_patch_embed [B, d_dist]
    Output: patch_token [B, d_patch]
    """

    def __init__(
        self,
        p: int,
        d_dist: int,
        d_patch: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.p = int(p)
        self.d_patch = int(d_patch)

        residual_in_dim = self.p + int(d_dist)
        layers: list[nn.Module] = []
        in_dim = residual_in_dim
        for _ in range(max(1, int(num_layers)) - 1):
            layers.extend([
                nn.Linear(in_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.d_patch))
        self.transform = nn.Sequential(*layers)
        self.weight_residual_proj = nn.Linear(self.p, self.d_patch, bias=False)

        self._init_near_identity()

    def _init_near_identity(self) -> None:
        """Initialize so output is close to a linear projection of w_patch."""
        with torch.no_grad():
            last_linear: nn.Linear | None = None
            for module in reversed(list(self.transform.modules())):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

            min_dim = min(self.p, self.d_patch)
            nn.init.zeros_(self.weight_residual_proj.weight)
            self.weight_residual_proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)

    def forward(self, w_patch: torch.Tensor, dist_patch_embed: torch.Tensor) -> torch.Tensor:
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        if dist_patch_embed.ndim != 2:
            raise ValueError(f"dist_patch_embed must be [B, d_dist], got {tuple(dist_patch_embed.shape)}")

        residual = self.weight_residual_proj(w_patch)
        x = torch.cat([w_patch, dist_patch_embed], dim=-1)
        delta = self.transform(x)
        return residual + delta


class MixerPatchTokenizer(nn.Module):
    """
    MLP-Mixer style patch tokenizer.

    Input:  w_patch [B, p], dist_var_tokens [B, p, d_var], dist_patch_embed [B, d_dist]
    Output: patch_token [B, d_patch]
    """

    def __init__(
        self,
        p: int,
        d_var: int,
        d_dist: int,
        d_patch: int,
        d_hidden: int = 128,
        num_mixer_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.p = int(p)
        self.d_patch = int(d_patch)

        self.weight_embed = nn.Sequential(
            nn.Linear(1 + int(d_var), d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        self.weight_mlps = nn.ModuleList()
        self.patch_mlps = nn.ModuleList()
        for _ in range(max(1, int(num_mixer_layers))):
            self.weight_mlps.append(
                nn.Sequential(
                    nn.LayerNorm(d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    nn.GELU(),
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(float(dropout)),
                )
            )
            self.patch_mlps.append(
                nn.Sequential(
                    nn.LayerNorm(self.p),
                    nn.Linear(self.p, self.p),
                    nn.GELU(),
                    nn.Linear(self.p, self.p),
                    nn.Dropout(float(dropout)),
                )
            )

        d_reduction = (self.d_patch + self.p - 1) // self.p
        self.reduction_proj = nn.Linear(d_hidden, d_reduction)
        self.final_proj = nn.Sequential(
            nn.Linear(self.p * d_reduction + int(d_dist), d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_patch),
            nn.GELU(),
            nn.Linear(d_patch, d_patch),
        )

    def forward(
        self,
        w_patch: torch.Tensor,
        dist_var_tokens: torch.Tensor,
        dist_patch_embed: torch.Tensor,
    ) -> torch.Tensor:
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        if dist_var_tokens.ndim != 3:
            raise ValueError(f"dist_var_tokens must be [B, p, d_var], got {tuple(dist_var_tokens.shape)}")
        if dist_patch_embed.ndim != 2:
            raise ValueError(f"dist_patch_embed must be [B, d_dist], got {tuple(dist_patch_embed.shape)}")

        x = torch.cat([w_patch.unsqueeze(-1), dist_var_tokens], dim=-1)
        x = self.weight_embed(x)

        for weight_mlp, patch_mlp in zip(self.weight_mlps, self.patch_mlps):
            x = x + weight_mlp(x)
            x_t = x.transpose(1, 2)
            x_t = x_t + patch_mlp(x_t)
            x = x_t.transpose(1, 2)

        x_pool = self.reduction_proj(x).flatten(start_dim=1)
        x_cat = torch.cat([x_pool, dist_patch_embed], dim=-1)
        return self.final_proj(x_cat)


__all__ = [
    "MixerPatchTokenizer",
    "ResidualPatchTokenizer",
]
