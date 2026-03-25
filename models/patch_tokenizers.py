from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_init_last_linear(module: nn.Module) -> None:
    last_linear: nn.Linear | None = None
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            last_linear = submodule
            break
    if last_linear is None:
        return
    nn.init.zeros_(last_linear.weight)
    if last_linear.bias is not None:
        nn.init.zeros_(last_linear.bias)


class AlignedConditionedPatchBlock(nn.Module):
    """Aligned per-position Y/X residual update block for patch-token refinement."""

    def __init__(
        self,
        *,
        d_model: int,
        d_hidden: int,
        d_cond: int,
        d_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.base_fc1 = nn.Linear(d_model, d_hidden)
        self.base_fc2 = nn.Linear(d_hidden, d_model)

        self.u_proj = nn.Linear(d_model, d_rank)
        self.x_proj = nn.Linear(d_cond, d_rank)
        self.cond_fc1 = nn.Linear(d_rank, d_hidden)
        self.cond_fc2 = nn.Linear(d_hidden, d_model)
        self.gate_fc = nn.Linear(d_rank, d_model)
        self.alpha = nn.Parameter(torch.zeros(1, 1, d_model))

        nn.init.zeros_(self.cond_fc2.weight)
        nn.init.zeros_(self.cond_fc2.bias)
        nn.init.zeros_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, u: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        if u.ndim != 3:
            raise ValueError(f"u must be [B, P, D], got {tuple(u.shape)}")
        if x_c.ndim != 3:
            raise ValueError(f"x_c must be [B, P, Dc], got {tuple(x_c.shape)}")
        if tuple(u.shape[:2]) != tuple(x_c.shape[:2]):
            raise ValueError(f"u and x_c must align on [B, P], got {tuple(u.shape)} vs {tuple(x_c.shape)}")

        h0 = self.norm(u)
        base = self.base_fc2(F.gelu(self.base_fc1(h0)))

        p_u = self.u_proj(h0)
        p_x = self.x_proj(x_c)
        inter = p_u * p_x

        hidden = F.gelu(self.cond_fc1(inter))
        delta = self.cond_fc2(hidden)
        gate = torch.sigmoid(self.gate_fc(inter))
        return u + base + self.alpha * gate * delta


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

    def forward(self, 
        w_patch: torch.Tensor, 
        dist_var_tokens: torch.Tensor = None,
        dist_patch_embed: torch.Tensor = None
    ) -> torch.Tensor:
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


class PatchConditionedMLPTokenizer(nn.Module):
    """
    Patch token encoder with aligned per-position Y/X conditioning.

    Input:  w_patch [B, p], dist_var_tokens [B, p, d_var]
    Output: patch_token [B, d_patch]
    """

    def __init__(
        self,
        p: int,
        d_var: int,
        d_patch: int,
        hidden_dim: int | None = None,
        cond_proj_dim: int | None = None,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.p = int(p)
        self.d_patch = int(d_patch)
        hidden = int(hidden_dim) if hidden_dim is not None else int(4 * d_patch)
        c_proj = int(cond_proj_dim) if cond_proj_dim is not None else min(64, max(16, d_patch // 2))
        rank_dim = max(8, min(self.d_patch, c_proj))

        self.y_proj = nn.Linear(1, self.d_patch)
        self.x_proj = nn.Linear(int(d_var), c_proj)
        self.blocks = nn.ModuleList(
            [
                AlignedConditionedPatchBlock(
                    d_model=self.d_patch,
                    d_hidden=hidden,
                    d_cond=c_proj,
                    d_rank=rank_dim,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_blocks)))
            ]
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(self.p * self.d_patch),
            nn.Linear(self.p * self.d_patch, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.d_patch),
        )

    def forward(
        self,
        w_patch: torch.Tensor,
        dist_var_tokens: torch.Tensor,
        dist_patch_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        if dist_var_tokens.ndim != 3:
            raise ValueError(f"dist_var_tokens must be [B, p, d_var], got {tuple(dist_var_tokens.shape)}")
        if int(w_patch.shape[0]) != int(dist_var_tokens.shape[0]) or int(w_patch.shape[1]) != int(dist_var_tokens.shape[1]):
            raise ValueError(
                "w_patch and dist_var_tokens must agree on [B, p], got "
                f"{tuple(w_patch.shape)} vs {tuple(dist_var_tokens.shape)}"
            )

        u = self.y_proj(w_patch.unsqueeze(-1))
        x_small = self.x_proj(dist_var_tokens)
        for block in self.blocks:
            u = block(u, x_small)
        return self.summary(u.flatten(start_dim=1))


ConditionedMLPBlock = AlignedConditionedPatchBlock


__all__ = [
    "AlignedConditionedPatchBlock",
    "ConditionedMLPBlock",
    "MixerPatchTokenizer",
    "PatchConditionedMLPTokenizer",
    "ResidualPatchTokenizer",
]
