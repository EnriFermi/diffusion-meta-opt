from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.vae_shared import _masked_quantile_over_samples, _quantile_over_samples


class CrossLayer(nn.Module):
    """Minimal DCNv2-style cross layer: x_{l+1} = x_l + x_0 * W(x_l) + b."""

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return xl + x0 * self.linear(xl) + self.bias


class DCNv2(nn.Module):
    """Minimal DCNv2: stack of cross layers + optional deep tower."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_cross_layers: int = 3,
        deep_hidden: int = 0,
        deep_layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(d_in) for _ in range(max(1, num_cross_layers))])

        self.has_deep = deep_hidden > 0 and deep_layers > 0
        if self.has_deep:
            deep: list[nn.Module] = []
            in_dim = d_in
            for _ in range(deep_layers):
                deep.append(nn.Linear(in_dim, deep_hidden))
                deep.append(nn.GELU())
                deep.append(nn.Dropout(dropout))
                in_dim = deep_hidden
            self.deep = nn.Sequential(*deep)
            self.out = nn.Linear(d_in + deep_hidden, d_out)
        else:
            self.deep = None
            self.out = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        if self.has_deep and self.deep is not None:
            xd = self.deep(x)
            return self.out(torch.cat([xl, xd], dim=-1))

        return self.out(xl)


@dataclass(slots=True)
class DistributionConfig:
    k_s: int = 16
    Kq: int = 32
    d_var: int = 128
    d_dist: int = 128
    num_var_attn_layers: int = 2
    var_attn_heads: int = 4
    dcn_num_cross_layers: int = 3
    dcn_deep_hidden: int = 0
    dcn_deep_layers: int = 0
    dropout: float = 0.0
    use_covariance: bool = True
    patch_size_for_cov: int = 16


class InputDistributionEncodingModule(nn.Module):
    """
    Encodes empirical input distribution for one input patch.

    Inputs:
    - X: [B, n, d_in]
    - patch_idx: [B, p] (indices along d_in)

    Outputs:
    - dist_var_tokens: [B, p, d_var]
    - dist_patch_embed: [B, d_dist]
    """

    def __init__(self, cfg: DistributionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.d_var % cfg.var_attn_heads != 0:
            raise ValueError(f"d_var ({cfg.d_var}) must be divisible by var_attn_heads ({cfg.var_attn_heads})")

        self.var_mlp = nn.Sequential(
            nn.Linear(cfg.k_s + 2, cfg.d_var),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_var, cfg.d_var),
            nn.Dropout(cfg.dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_var,
            nhead=cfg.var_attn_heads,
            dim_feedforward=max(4 * cfg.d_var, cfg.d_var),
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.var_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, cfg.num_var_attn_layers))

        self.dcn = DCNv2(
            d_in=cfg.d_var,
            d_out=cfg.d_dist,
            num_cross_layers=cfg.dcn_num_cross_layers,
            deep_hidden=cfg.dcn_deep_hidden,
            deep_layers=cfg.dcn_deep_layers,
            dropout=cfg.dropout,
        )

        self.use_covariance = bool(cfg.use_covariance)
        if self.use_covariance:
            cov_p = max(1, int(cfg.patch_size_for_cov))
            upper_tri_size = cov_p * (cov_p + 1) // 2
            self.cov_encoder = nn.Sequential(
                nn.Linear(upper_tri_size, cfg.d_var),
                nn.GELU(),
                nn.Linear(cfg.d_var, cfg.d_var),
            )
        else:
            self.cov_encoder = None

        probs = torch.linspace(0.0, 1.0, cfg.k_s, dtype=torch.float32)
        self.register_buffer("quantile_probs", probs, persistent=False)

    def forward(
        self,
        X: torch.Tensor,
        patch_idx: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if X.ndim != 3:
            raise ValueError(f"X must be rank-3 [B, n, d_in], got {tuple(X.shape)}")
        if patch_idx.ndim != 2:
            raise ValueError(f"patch_idx must be rank-2 [B, p], got {tuple(patch_idx.shape)}")

        B, n, d_in = X.shape
        B_idx, p = patch_idx.shape
        if B_idx != B:
            raise ValueError(f"patch_idx batch ({B_idx}) must match X batch ({B})")
        mask_bool: torch.Tensor | None = None
        if sample_mask is not None:
            if sample_mask.ndim != 2:
                raise ValueError(f"sample_mask must be rank-2 [B, n], got {tuple(sample_mask.shape)}")
            if tuple(sample_mask.shape) != (B, n):
                raise ValueError(f"sample_mask shape must be {(B, n)}, got {tuple(sample_mask.shape)}")
            mask_bool = sample_mask.to(device=X.device, dtype=torch.bool)

        idx = patch_idx.to(dtype=torch.long, device=X.device).clamp(min=0, max=max(0, d_in - 1))
        X_I = X.gather(dim=2, index=idx.unsqueeze(1).expand(-1, n, -1))

        X_q = X_I.to(torch.float32)
        if mask_bool is None:
            q_raw = _quantile_over_samples(X_q, self.quantile_probs.to(X_q.device))
        else:
            q_raw = _masked_quantile_over_samples(X_q, self.quantile_probs.to(X_q.device), mask_bool)
        q = q_raw.permute(1, 0, 2).contiguous()

        q_first = q[:, 0:1, :]
        q_last = q[:, -1:, :]
        q_span = (q_last - q_first).clamp_min(1e-2)
        q = 2.0 * (q - q_first) / q_span - 1.0
        q = q.clamp(-5.0, 5.0)
        q_idx = torch.arange(q.shape[1], device=q.device, dtype=torch.long).view(1, -1, 1)
        q = torch.where(q_idx == 0, q.new_full((), -1.0), q)
        q = torch.where(q_idx == q.shape[1] - 1, q.new_full((), 1.0), q)

        if mask_bool is None:
            mu = X_q.mean(dim=1)
            sigma = X_q.std(dim=1, unbiased=False)
            valid_counts = None
        else:
            mask_float = mask_bool.to(dtype=X_q.dtype).unsqueeze(-1)
            valid_counts = mask_float.sum(dim=1).clamp_min(1.0)
            mu = (X_q * mask_float).sum(dim=1) / valid_counts
            var = ((X_q - mu.unsqueeze(1)).pow(2) * mask_float).sum(dim=1) / valid_counts
            sigma = torch.sqrt(var.clamp_min(0.0))
        eps = 1e-6
        mu_log = torch.sign(mu) * torch.log1p(mu.abs())
        sigma_log = torch.log(sigma + eps)

        q_var_major = q.permute(0, 2, 1).contiguous()
        features = torch.cat([q_var_major, mu_log.unsqueeze(-1), sigma_log.unsqueeze(-1)], dim=-1)
        v = self.var_mlp(features.to(dtype=X.dtype if X.dtype.is_floating_point else torch.float32))
        dist_var_tokens = self.var_encoder(v)

        v_pool = dist_var_tokens.mean(dim=1)
        if self.use_covariance and self.cov_encoder is not None:
            cov_in_features = int(self.cov_encoder[0].in_features)
            expected_cov_in_features = p * (p + 1) // 2
            if cov_in_features != expected_cov_in_features:
                raise ValueError(
                    "Covariance encoder input size mismatch: "
                    f"patch_idx width p={p} implies {expected_cov_in_features} covariance features, "
                    f"but cov_encoder expects {cov_in_features}. "
                    "Set distribution.patch_size_for_cov to match model.patch_size."
                )
            if mask_bool is None:
                X_I_centered = X_I - X_I.mean(dim=1, keepdim=True)
                cov_denom = X_I.new_full((B, 1, 1), float(max(1, n - 1)))
            else:
                assert valid_counts is not None
                mask_float_cov = mask_bool.to(dtype=X_I.dtype).unsqueeze(-1)
                X_I_centered = (X_I - mu.unsqueeze(1)) * mask_float_cov
                cov_denom = (valid_counts - 1.0).clamp_min(1.0).unsqueeze(-1)
            cov = torch.bmm(
                X_I_centered.transpose(1, 2).to(torch.float32),
                X_I_centered.to(torch.float32),
            ) / cov_denom.to(torch.float32)
            tri_idx = torch.triu_indices(p, p, device=cov.device)
            cov_upper = cov[:, tri_idx[0], tri_idx[1]]
            cov_upper = torch.sign(cov_upper) * torch.log1p(cov_upper.abs())
            v_pool = v_pool + self.cov_encoder(cov_upper)

        dist_patch_embed = self.dcn(v_pool)
        return dist_var_tokens, dist_patch_embed


__all__ = [
    "CrossLayer",
    "DCNv2",
    "DistributionConfig",
    "InputDistributionEncodingModule",
]
