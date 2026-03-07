from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quantile_over_samples(X_q: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compile-friendly quantile over sample axis.

    X_q: [B, n, p]
    q: [k]
    returns: [k, B, p]
    """
    if X_q.ndim != 3:
        raise ValueError(f"X_q must be [B,n,p], got {tuple(X_q.shape)}")
    if q.ndim != 1:
        raise ValueError(f"q must be rank-1, got {tuple(q.shape)}")

    sorted_vals, _ = torch.sort(X_q, dim=1)
    n = sorted_vals.shape[1]
    k = q.shape[0]

    q = q.to(device=X_q.device, dtype=X_q.dtype).clamp(0.0, 1.0)
    pos = q * (n - 1)

    lower = torch.floor(pos).to(dtype=torch.long)
    upper = torch.ceil(pos).to(dtype=torch.long)
    alpha = (pos - lower.to(dtype=X_q.dtype)).view(1, k, 1)

    B, _, p = sorted_vals.shape
    lower_idx = lower.view(1, k, 1).expand(B, k, p)
    upper_idx = upper.view(1, k, 1).expand(B, k, p)

    lower_vals = sorted_vals.gather(dim=1, index=lower_idx)
    upper_vals = sorted_vals.gather(dim=1, index=upper_idx)
    q_vals = lower_vals + (upper_vals - lower_vals) * alpha
    return q_vals.permute(1, 0, 2).contiguous()


def sinusoidal_embedding(indices: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Deterministic sinusoidal embedding for arbitrary-sized integer grids."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    x = indices.to(dtype=torch.float32)
    half = dim // 2
    if half == 0:
        return x.unsqueeze(-1)

    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / max(half - 1, 1))
    angles = x.unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
    """
    Apply RoPE to attention projections.

    x: [B, H, T, D_h]
    positions: [T]
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B,H,T,D_h], got {tuple(x.shape)}")
    if positions.ndim != 1:
        raise ValueError(f"positions must be rank-1 [T], got {tuple(positions.shape)}")
    if int(positions.shape[0]) != int(x.shape[2]):
        raise ValueError(f"positions length ({int(positions.shape[0])}) must match sequence length ({int(x.shape[2])})")

    _, _, _, d_h = x.shape
    rope_dim = int(d_h) if int(d_h) % 2 == 0 else int(d_h) - 1
    if rope_dim <= 0:
        return x

    pos = positions.to(device=x.device, dtype=torch.float32)
    inv_freq = torch.exp(
        -math.log(max_period) * torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / float(rope_dim)
    )
    angles = pos[:, None] * inv_freq[None, :]  # [T, rope_dim/2]
    sin = torch.sin(angles).to(dtype=x.dtype).view(1, 1, x.shape[2], rope_dim // 2)
    cos = torch.cos(angles).to(dtype=x.dtype).view(1, 1, x.shape[2], rope_dim // 2)

    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    x_rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    if x_pass.numel() == 0:
        return x_rotated
    return torch.cat([x_rotated, x_pass], dim=-1)


def _rope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """
    RoPE + scaled dot-product attention.

    q/k/v: [B, H, T, D_h] and [B, H, S, D_h]
    q_pos: [T]
    k_pos: [S]
    returns: [B, H, T, D_h]
    """
    q_rot = _apply_rope(q, q_pos)
    k_rot = _apply_rope(k, k_pos)
    attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=training)
    return torch.matmul(attn_probs, v)


class RoPeMixed2D(nn.Module):
    """
    Learned mixed 2D RoPE (arXiv:2403.13298, Eq. 14).

    angle[h, t] = freq1[h, t] * pos1 + freq2[h, t] * pos2

    Every head dimension encodes both positional axes simultaneously.
    Per-head learnable frequencies, negligible parameter overhead.
    """

    def __init__(self, n_heads: int, head_dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        half = head_dim // 2
        if half <= 0:
            raise ValueError(f"head_dim must be >= 2 for RoPE, got {head_dim}")
        self.half = half
        # Axis-1 frequencies: standard RoPE initialization (base = max_period).
        base_freq1 = torch.exp(
            -math.log(max_period)
            * torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
        )  # [half]
        # Axis-2 frequencies: sqrt(max_period) base for initial axis diversity.
        base_freq2 = torch.exp(
            -0.5 * math.log(max_period)
            * torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
        )  # [half]
        self.freq1 = nn.Parameter(base_freq1.unsqueeze(0).expand(n_heads, -1).clone())
        self.freq2 = nn.Parameter(base_freq2.unsqueeze(0).expand(n_heads, -1).clone())

    def compute_angles(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """
        pos1, pos2: [T]
        returns: [n_heads, T, half]
        """
        p1 = pos1.to(device=self.freq1.device, dtype=torch.float32)
        p2 = pos2.to(device=self.freq2.device, dtype=torch.float32)
        return p1[None, :, None] * self.freq1[:, None, :] + p2[None, :, None] * self.freq2[:, None, :]


def _apply_rope_with_angles(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE using precomputed per-head angles (from RoPeMixed2D).

    x: [B, H, T, D_h]
    angles: [H, T, D_h/2]
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B,H,T,D_h], got {tuple(x.shape)}")
    D_h = x.shape[-1]
    rope_dim = int(D_h) if int(D_h) % 2 == 0 else int(D_h) - 1
    if rope_dim <= 0:
        return x

    sin = torch.sin(angles).to(dtype=x.dtype).unsqueeze(0)  # [1, H, T, half]
    cos = torch.cos(angles).to(dtype=x.dtype).unsqueeze(0)  # [1, H, T, half]

    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    x_rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    if x_pass.numel() == 0:
        return x_rotated
    return torch.cat([x_rotated, x_pass], dim=-1)


def _rope_attention_with_angles(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_angles: torch.Tensor,
    k_angles: torch.Tensor,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """
    Mixed 2D RoPE + scaled dot-product attention.

    q/k/v: [B, H, T, D_h] and [B, H, S, D_h]
    q_angles: [H, T, D_h/2]
    k_angles: [H, S, D_h/2]
    """
    q_rot = _apply_rope_with_angles(q, q_angles)
    k_rot = _apply_rope_with_angles(k, k_angles)
    attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=training)
    return torch.matmul(attn_probs, v)


def _init_vae_module_weights(
    module: nn.Module,
    base_std: float = 0.02,
    init_style: str = "llm",
) -> None:
    """Explicit initialization for mini-VAE modules (real and stub variants)."""
    style = str(init_style).strip().lower()
    if style not in {"xavier", "llm"}:
        raise ValueError(f"init_style must be 'xavier' or 'llm', got {init_style!r}")
    if float(base_std) <= 0.0:
        raise ValueError(f"base_std must be positive, got {base_std}")

    if isinstance(module, nn.Linear):
        if style == "xavier":
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=float(base_std))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            if style == "xavier":
                nn.init.xavier_uniform_(module.in_proj_weight)
            else:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=float(base_std))
        else:
            if getattr(module, "q_proj_weight", None) is not None:
                if style == "xavier":
                    nn.init.xavier_uniform_(module.q_proj_weight)
                else:
                    nn.init.normal_(module.q_proj_weight, mean=0.0, std=float(base_std))
            if getattr(module, "k_proj_weight", None) is not None:
                if style == "xavier":
                    nn.init.xavier_uniform_(module.k_proj_weight)
                else:
                    nn.init.normal_(module.k_proj_weight, mean=0.0, std=float(base_std))
            if getattr(module, "v_proj_weight", None) is not None:
                if style == "xavier":
                    nn.init.xavier_uniform_(module.v_proj_weight)
                else:
                    nn.init.normal_(module.v_proj_weight, mean=0.0, std=float(base_std))
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def _find_ffn_down_projection(module: nn.Module) -> nn.Linear | None:
    # Preferred explicit naming.
    ffn_down = getattr(module, "ffn_down", None)
    if isinstance(ffn_down, nn.Linear):
        return ffn_down

    # MLP helper used in BigVAE blocks.
    ffn = getattr(module, "ffn", None)
    if isinstance(ffn, MLP):
        return ffn.fc2

    # Sequential FFN: use last Linear as down-proj.
    if isinstance(ffn, nn.Sequential):
        for child in reversed(ffn):
            if isinstance(child, nn.Linear):
                return child
    return None


def _apply_residual_scaled_init(root_module: nn.Module, L_stack: int, base_std: float = 0.02) -> None:
    """
    Downscale residual-branch output projections:
    std_out = base_std / sqrt(2 * L_stack)
    """
    if int(L_stack) <= 0:
        raise ValueError(f"L_stack must be positive, got {L_stack}")
    if float(base_std) <= 0.0:
        raise ValueError(f"base_std must be positive, got {base_std}")
    std_out = float(base_std) / math.sqrt(2.0 * float(L_stack))

    with torch.no_grad():
        for module in root_module.modules():
            # MHA residual projection.
            is_mha = isinstance(module, nn.MultiheadAttention)
            if isinstance(module, nn.MultiheadAttention):
                out_proj = getattr(module, "out_proj", None)
                if isinstance(out_proj, nn.Linear):
                    nn.init.normal_(out_proj.weight, mean=0.0, std=std_out)
                    if out_proj.bias is not None:
                        nn.init.zeros_(out_proj.bias)

            # Manual attention residual projections.
            for attr in ("out_proj", "cross_out_proj", "self_out_proj"):
                if is_mha and attr == "out_proj":
                    continue
                proj = getattr(module, attr, None)
                if isinstance(proj, nn.Linear):
                    nn.init.normal_(proj.weight, mean=0.0, std=std_out)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

            # FFN down-projection.
            down_proj = _find_ffn_down_projection(module)
            if isinstance(down_proj, nn.Linear):
                nn.init.normal_(down_proj.weight, mean=0.0, std=std_out)
                if down_proj.bias is not None:
                    nn.init.zeros_(down_proj.bias)
def _init_vae_latent_parameters(module: nn.Module) -> None:
    """Initialize standalone latent parameters that are not covered by module.apply()."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("resampler_latents"):
                nn.init.normal_(param, mean=0.0, std=0.005)


def _decode_direction_and_logscale(
    u_hat: torch.Tensor,
    s_hat: torch.Tensor,
    eps: float,
    s_min: float,
    s_max: float | None = None,
) -> torch.Tensor:
    """
    Convert decoder outputs into weights:
    U = u_hat / (||u_hat||_2 + eps), s = exp(s_hat), W_hat = s * U.
    """
    if u_hat.ndim != 2:
        raise ValueError(f"u_hat must be [B, p], got {tuple(u_hat.shape)}")
    if s_hat.ndim == 1:
        s = s_hat.unsqueeze(-1)
    elif s_hat.ndim == 2 and s_hat.shape[1] == 1:
        s = s_hat
    else:
        raise ValueError(f"s_hat must be [B] or [B,1], got {tuple(s_hat.shape)}")
    if tuple(s.shape[:1]) != tuple(u_hat.shape[:1]):
        raise ValueError(f"Batch mismatch: u_hat={tuple(u_hat.shape)}, s_hat={tuple(s_hat.shape)}")

    u_norm = u_hat.norm(dim=1, keepdim=True).clamp_min(float(eps))
    u = u_hat / u_norm

    # Convert log-scale to scale.
    s = torch.exp(s)

    # Smooth lower barrier: keeps s >= s_min while preserving gradients below the threshold.
    if s_min != 0:
        s = float(s_min) + F.softplus(s - float(s_min))
    # Optional smooth upper barrier to avoid exp overflow and skipped non-finite steps.
    if s_max is not None and s_max != 0:
        s = float(s_max) - F.softplus(float(s_max) - s)

    return s * u


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CrossLayer(nn.Module):
    """Minimal DCNv2-style cross layer: x_{l+1} = x_l + x_0 * W(x_l) + b."""

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0: [B, d_in], xl: [B, d_in]
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
        # x: [B, d_in]
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        if self.has_deep and self.deep is not None:
            xd = self.deep(x)
            x_cat = torch.cat([xl, xd], dim=-1)
            return self.out(x_cat)

        return self.out(xl)


@dataclass(slots=True)
class DistributionConfig:
    k_s: int = 16
    Kq: int = 32  # Legacy field; not used in current quantile-conv path.
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
        if cfg.k_s < 3:
            raise ValueError(
                f"k_s must be >= 3 for kernel_size=3 quantile conv (output length k_s-2), got {cfg.k_s}"
            )

        if cfg.d_var % cfg.var_attn_heads != 0:
            raise ValueError(f"d_var ({cfg.d_var}) must be divisible by var_attn_heads ({cfg.var_attn_heads})")

        # Single-channel conv along quantile axis (no padding).
        # [B*p, 1, k_s] -> [B*p, 1, k_s - 2]
        self.quantile_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=0)

        # Shared per-variable MLP: ((k_s - 2) + 2) -> d_var.
        self.var_mlp = nn.Sequential(
            nn.Linear(cfg.k_s, cfg.d_var),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_var, cfg.d_var),
            nn.Dropout(cfg.dropout),
        )

        # 2-layer variable self-attention (configurable layer count).
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

        # Covariance conditioning: encode upper triangle of patch-local X_I^T @ X_I.
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

    def forward(self, X: torch.Tensor, patch_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # X: [B, n, d_in]
        # patch_idx: [B, p]
        if X.ndim != 3:
            raise ValueError(f"X must be rank-3 [B, n, d_in], got {tuple(X.shape)}")
        if patch_idx.ndim != 2:
            raise ValueError(f"patch_idx must be rank-2 [B, p], got {tuple(patch_idx.shape)}")

        B, n, d_in = X.shape
        B_idx, p = patch_idx.shape
        if B_idx != B:
            raise ValueError(f"patch_idx batch ({B_idx}) must match X batch ({B})")

        idx = patch_idx.to(dtype=torch.long, device=X.device).clamp(min=0, max=max(0, d_in - 1))

        # Gather patch variables.
        # X_I: [B, n, p]
        X_I = X.gather(dim=2, index=idx.unsqueeze(1).expand(-1, n, -1))

        # Quantiles over sample axis n.
        # q_raw: [k_s, B, p] -> q: [B, k_s, p]
        X_q = X_I.to(torch.float32)
        q_raw = _quantile_over_samples(X_q, self.quantile_probs.to(X_q.device))
        q = q_raw.permute(1, 0, 2).contiguous()

        # Normalize quantiles per variable into [-1, 1] scale.
        # q_first/q_last: [B, 1, p]
        q_first = q[:, 0:1, :]
        q_last = q[:, -1:, :]
        q_span = (q_last - q_first).clamp_min(1e-2)  # Raised floor to prevent extreme amplification.
        q = 2.0 * (q - q_first) / q_span - 1.0  # [B, k_s, p]
        q = q.clamp(-5.0, 5.0)  # Safety clamp on normalized quantiles.
        q_idx = torch.arange(q.shape[1], device=q.device, dtype=torch.long).view(1, -1, 1)
        q = torch.where(q_idx == 0, q.new_full((), -1.0), q)
        q = torch.where(q_idx == q.shape[1] - 1, q.new_full((), 1.0), q)

        # Mu/sigma (unconvolved): [B, p]
        mu = X_q.mean(dim=1)
        sigma = X_q.std(dim=1, unbiased=False)
        eps = 1e-6

        # Log-versions of moments:
        # mu_log uses signed log to preserve sign information for negative means.
        # sigma_log is standard log on positive sigma.
        mu_log = torch.sign(mu) * torch.log1p(mu.abs())
        sigma_log = torch.log(sigma + eps)

        # Convolution along quantile axis only.
        # q_var_major: [B, p, k_s]
        # q_conv_in: [B*p, 1, k_s]
        q_var_major = q.permute(0, 2, 1).contiguous()
        q_conv_in = q_var_major.view(B * p, 1, self.cfg.k_s)

        # q_conv_seq: [B*p, 1, k_s - 2] -> q_conv: [B, p, k_s - 2]
        q_conv_seq = self.quantile_conv(q_conv_in)
        q_conv = q_conv_seq.squeeze(1).reshape(B, p, self.cfg.k_s - 2)

        # Per-variable features: [B, p, (k_s - 2) + 2] = [B, p, k_s]
        f = torch.cat([q_conv, mu_log.unsqueeze(-1), sigma_log.unsqueeze(-1)], dim=-1)

        # Shared var MLP: [B, p, d_var]
        v = self.var_mlp(f.to(dtype=X.dtype if X.dtype.is_floating_point else torch.float32))

        # Variable context via self-attention: [B, p, d_var]
        dist_var_tokens = self.var_encoder(v)

        # Mean pool over variables: [B, d_var]
        v_pool = dist_var_tokens.mean(dim=1)

        # Covariance conditioning: patch-local X_I^T @ X_I.
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
            X_I_centered = X_I - X_I.mean(dim=1, keepdim=True)
            # cov: [B, p, p]
            cov = torch.bmm(
                X_I_centered.transpose(1, 2).to(torch.float32),
                X_I_centered.to(torch.float32),
            ) / max(1, n - 1)
            # Extract upper triangle (including diagonal): [B, p*(p+1)//2]
            tri_idx = torch.triu_indices(p, p, device=cov.device)
            cov_upper = cov[:, tri_idx[0], tri_idx[1]]
            # Log-scale for numerical stability.
            cov_upper = torch.sign(cov_upper) * torch.log1p(cov_upper.abs())
            cov_embed = self.cov_encoder(cov_upper)  # [B, d_var]
            v_pool = v_pool + cov_embed

        # DCNv2 patch embedding: [B, d_dist]
        dist_patch_embed = self.dcn(v_pool)

        return dist_var_tokens, dist_patch_embed


@dataclass(slots=True)
class MiniVAEConfig:
    z_dim: int = 64
    d_e: int = 128
    # Real-only: Perceiver latent width in MiniPatchEncoder.
    # If 0, defaults to d_e.
    encoder_latent_dim: int = 0
    # Real-only: Fourier positional width for latent slot anchoring.
    # If 0, defaults to encoder latent width.
    pos_lat_dim: int = 0
    pos_dim: int = 32
    num_attn_layers_encoder: int = 2
    num_layers_decoder: int = 2
    # Legacy field from bilinear decoder path; ignored by current decoder.
    decoder_bilinear_rank: int = 0
    decoder_L_latents: int = 8
    decoder_use_dist_conditioning: bool = True
    decoder_dist_mode: str = "add"  # {"add", "concat"}
    use_latent_sampling: bool = True
    implementation: str = "real"  # {"real", "mlp_stub"}
    mlp_stub_hidden_dim: int = 256
    stub_mlp_use_batchnorm: bool = False
    # Stub-only: if > 0, Perceiver resampler width in TransformerNoCompressionPatchEncoder.
    # If 0, defaults to d_e.
    stub_resampler_d_model: int = 0
    init_style: str = "llm"  # {"xavier", "llm"}
    n_heads: int = 4
    d_patch: int = 64
    dropout: float = 0.0


class MiniPatchEncoder(nn.Module):
    """Encoder used for both MiniPatchVAE pretraining and BigWeightVAE tokenization."""

    def __init__(self, d_var: int, cfg: MiniVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_var = d_var

        if cfg.d_e % cfg.n_heads != 0:
            raise ValueError(f"mini d_e ({cfg.d_e}) must be divisible by n_heads ({cfg.n_heads})")
        if cfg.pos_dim <= 0:
            raise ValueError(f"mini pos_dim must be positive, got {cfg.pos_dim}")
        if int(cfg.encoder_latent_dim) < 0:
            raise ValueError(f"mini encoder_latent_dim must be >= 0, got {int(cfg.encoder_latent_dim)}")
        if int(cfg.pos_lat_dim) < 0:
            raise ValueError(f"mini pos_lat_dim must be >= 0, got {int(cfg.pos_lat_dim)}")

        # Per-element embed: (1 + d_var + pos_dim) -> d_e.
        self.elem_embed = nn.Sequential(
            nn.Linear(1 + d_var + cfg.pos_dim, cfg.d_e),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_e, cfg.d_e),
            nn.Dropout(cfg.dropout),
        )
        self.latent_d_model = int(cfg.encoder_latent_dim) if int(cfg.encoder_latent_dim) > 0 else int(cfg.d_e)
        if self.latent_d_model % cfg.n_heads != 0:
            raise ValueError(
                f"mini encoder latent d_model ({self.latent_d_model}) must be divisible by n_heads ({cfg.n_heads})"
            )
        self.pos_lat_dim = int(cfg.pos_lat_dim) if int(cfg.pos_lat_dim) > 0 else self.latent_d_model

        self.num_latents = max(1, int(cfg.decoder_L_latents))
        self.slot_pos_to_latent = nn.Linear(self.pos_lat_dim, self.latent_d_model)
        slot_tau = (torch.arange(self.num_latents, dtype=torch.float32) + 0.5) / float(self.num_latents)
        self.register_buffer("slot_tau", slot_tau, persistent=False)
        self.resampler = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    d_latent=self.latent_d_model,
                    d_token=int(cfg.d_e) + 1,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                )
                for _ in range(max(1, cfg.num_attn_layers_encoder))
            ]
        )
        self.flat_latent_dim = self.num_latents * self.latent_d_model
        self.latent_norm = nn.LayerNorm(self.latent_d_model)

        self.to_mu = nn.Linear(self.flat_latent_dim, cfg.z_dim)
        self.to_logvar = nn.Linear(self.flat_latent_dim, cfg.z_dim)

        if cfg.d_patch == cfg.z_dim:
            self.patch_proj = nn.Identity()
        else:
            self.patch_proj = nn.Linear(cfg.z_dim, cfg.d_patch)

    def encode(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # w_patch: [B, p]
        # dist_var_tokens: [B, p, d_var]
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        if dist_var_tokens.ndim != 3:
            raise ValueError(f"dist_var_tokens must be [B, p, d_var], got {tuple(dist_var_tokens.shape)}")

        B, p = w_patch.shape
        Bv, pv, d_var = dist_var_tokens.shape
        if Bv != B or pv != p:
            raise ValueError(f"Shape mismatch: w_patch={tuple(w_patch.shape)}, dist_var_tokens={tuple(dist_var_tokens.shape)}")
        if d_var != self.d_var:
            raise ValueError(f"dist_var_tokens last dim must be {self.d_var}, got {d_var}")

        # Deterministic per-position embedding on normalized token coordinates t_i=(i+0.5)/p.
        token_tau = (torch.arange(p, device=w_patch.device, dtype=torch.float32) + 0.5) / float(p)
        pos = sinusoidal_embedding(token_tau, dim=self.cfg.pos_dim).to(dtype=w_patch.dtype)
        pos_expand = pos.unsqueeze(0).expand(B, -1, -1)

        # T: [B, p, 1 + d_var + pos_dim]
        t = torch.cat([w_patch.unsqueeze(-1), dist_var_tokens, pos_expand], dim=-1)

        # E: [B, p, d_e]
        e = self.elem_embed(t)

        # Perceiver resampling: latent queries cross-attend to patch tokens.
        slot_pos = sinusoidal_embedding(self.slot_tau.to(device=w_patch.device), dim=self.pos_lat_dim).to(dtype=e.dtype)
        latents0 = self.slot_pos_to_latent(slot_pos)
        latents = latents0.unsqueeze(0).expand(B, -1, -1)
        tokens_with_w = torch.cat([e, w_patch.unsqueeze(-1).to(dtype=e.dtype)], dim=-1)
        latent_pos = self.slot_tau.to(device=w_patch.device)
        for block in self.resampler:
            latents = block(
                latents=latents,
                tokens=tokens_with_w,
                latent_pos=latent_pos,
                token_pos=token_tau,
            )
        # latents = self.latent_norm(latents)
        h = latents.reshape(B, self.flat_latent_dim)

        # mu/logvar: [B, z_dim]
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def encode_patch(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> torch.Tensor:
        # patch_token: [B, d_patch]
        mu, _ = self.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        patch_token = self.patch_proj(mu)
        return patch_token


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope_2d: bool = False) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.use_rope_2d = bool(use_rope_2d)
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_prob_dropout_p = float(dropout)
        self.attn_out_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_q_proj = nn.Linear(d_model, d_model)
        self.self_k_proj = nn.Linear(d_model, d_model)
        self.self_v_proj = nn.Linear(d_model, d_model)
        self.self_out_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        if self.use_rope_2d:
            self.rope_2d_cross = RoPeMixed2D(n_heads, self.head_dim)
            self.rope_2d_self = RoPeMixed2D(n_heads, self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_pos: torch.Tensor,
        kv_pos: torch.Tensor,
        q_pos2: torch.Tensor | None = None,
        kv_pos2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_2d = self.use_rope_2d and q_pos2 is not None
        q_attn = self.q_norm(q)
        kv_attn = self.kv_norm(kv)

        B, Tq, _ = q_attn.shape
        Tk = kv_attn.shape[1]

        q_proj = self.q_proj(q_attn).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(kv_attn).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(kv_attn).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        # Cross-attention with RoPE (1D or mixed 2D).
        if use_2d:
            _kv_pos2 = kv_pos2 if kv_pos2 is not None else kv_pos
            q_angles = self.rope_2d_cross.compute_angles(q_pos, q_pos2)
            k_angles = self.rope_2d_cross.compute_angles(kv_pos, _kv_pos2)
            attn_out = _rope_attention_with_angles(
                q=q_proj, k=k_proj, v=v_proj,
                q_angles=q_angles, k_angles=k_angles,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )
        else:
            attn_out = _rope_attention(
                q=q_proj, k=k_proj, v=v_proj,
                q_pos=q_pos, k_pos=kv_pos,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        attn_out = self.out_proj(attn_out)

        q = q + self.attn_out_dropout(attn_out)
        q_self = self.self_attn_norm(q)
        self_q = self.self_q_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_k = self.self_k_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_v = self.self_v_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)

        # Self-attention with RoPE (1D or mixed 2D).
        if use_2d:
            self_angles = self.rope_2d_self.compute_angles(q_pos, q_pos2)
            self_out = _rope_attention_with_angles(
                q=self_q, k=self_k, v=self_v,
                q_angles=self_angles, k_angles=self_angles,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )
        else:
            self_out = _rope_attention(
                q=self_q, k=self_k, v=self_v,
                q_pos=q_pos, k_pos=q_pos,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )
        self_out = self_out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        self_out = self.self_out_proj(self_out)
        q = q + self.self_attn_out_dropout(self_out)
        q = q + self.ffn(self.ffn_norm(q))
        return q


class CrossAttnPatchDecoder(nn.Module):
    """
    Cross-attention decoder with forced latent dependence:
    queries = positional tokens, keys/values = latent tokens from z (+ optional dist conditioning).
    """

    def __init__(self, d_dist: int, cfg: MiniVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_e)
        self.n_heads = int(cfg.n_heads)
        self.num_layers = max(1, int(cfg.num_layers_decoder))
        self.L_latents = max(1, int(cfg.decoder_L_latents))
        self.use_dist_conditioning = bool(cfg.decoder_use_dist_conditioning)
        self.dist_mode = str(cfg.decoder_dist_mode).strip().lower()
        self.d_dist = int(d_dist)

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"mini d_e ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.dist_mode not in {"add", "concat"}:
            raise ValueError(f"decoder_dist_mode must be 'add' or 'concat', got {self.dist_mode}")

        self.z_to_latents = nn.Linear(int(cfg.z_dim), self.L_latents * self.d_model)
        if self.use_dist_conditioning:
            self.dist_to_latents = nn.Linear(self.d_dist, self.L_latents * self.d_model)
        else:
            self.dist_to_latents = None

        self.blocks = nn.ModuleList(
            [CrossAttnBlock(d_model=self.d_model, n_heads=self.n_heads, dropout=float(cfg.dropout)) for _ in range(self.num_layers)]
        )
        self.query_seed = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.query_pos_proj = nn.Linear(self.d_model, self.d_model)
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0
        self.direction_seq_norm_eps = 1e-6

        self.direction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1)
        )
        self.scale_head = nn.Sequential(
            nn.Linear(self.d_model, 1),
        )

        # Direct z-to-output shortcut: bypasses attention for immediate z dependence.
        # Position-conditioned MLP: (z, pos_embed) -> scalar per position.
        self.z_shortcut = nn.Sequential(
            nn.Linear(int(cfg.z_dim) + self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    @staticmethod
    def _channel_norm_over_sequence(x: torch.Tensor, eps: float) -> torch.Tensor:
        # x: [B, T, C] -> normalize each channel over sequence length T.
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,C], got {tuple(x.shape)}")
        if int(x.shape[1]) <= 1:
            return x
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + float(eps))

    def forward(
        self,
        z: torch.Tensor,
        patch_size: int,
        dist_patch_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # z: [B, z_dim]
        # dist_patch_embed: [B, d_dist] (optional)
        # patch_size: p
        if z.ndim != 2:
            raise ValueError(f"z must be [B, z_dim], got {tuple(z.shape)}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        B = z.shape[0]
        p = int(patch_size)

        lat = self.z_to_latents(z).view(B, self.L_latents, self.d_model)
        if self.use_dist_conditioning and dist_patch_embed is not None and self.dist_to_latents is not None:
            dist_lat = self.dist_to_latents(dist_patch_embed).view(B, self.L_latents, self.d_model)
            if self.dist_mode == "add":
                lat = lat + dist_lat
            else:
                lat = torch.cat([lat, dist_lat], dim=1)

        q_seed = self.query_seed.to(dtype=z.dtype).view(1, 1, self.d_model).expand(B, p, self.d_model)
        q_pos_emb = sinusoidal_embedding(
            torch.arange(p, device=z.device, dtype=torch.float32),
            dim=self.d_model,
        ).to(dtype=self.query_pos_proj.weight.dtype)
        q_pos_emb = self.query_pos_proj(q_pos_emb).to(dtype=z.dtype)
        q = q_seed + q_pos_emb.unsqueeze(0)
        q_pos = torch.arange(p, device=z.device, dtype=torch.float32)
        kv_pos = torch.arange(lat.shape[1], device=z.device, dtype=torch.float32)

        for block in self.blocks:
            q = block(q, lat, q_pos=q_pos, kv_pos=kv_pos)

        q_dir = q
        # q_dir = self._channel_norm_over_sequence(q, eps=self.direction_seq_norm_eps)
        u_hat_attn = self.direction_head(q_dir).squeeze(-1)  # [B, p]

        # Direct shortcut: z + positional embedding -> per-position scalar.
        z_exp = z.unsqueeze(1).expand(B, p, -1)  # [B, p, z_dim]
        pos_exp = q_pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, p, d_model]
        shortcut_in = torch.cat([z_exp, pos_exp], dim=-1)  # [B, p, z_dim + d_model]
        u_hat_shortcut = self.z_shortcut(shortcut_in).squeeze(-1)  # [B, p]

        u_hat = u_hat_attn + u_hat_shortcut
        
        s_hat = self.scale_head(q.mean(dim=1)).squeeze(-1)  # [B]
        return _decode_direction_and_logscale(
            u_hat=u_hat,
            s_hat=s_hat,
            eps=self.output_eps,
            s_min=-6,
            s_max=self.output_s_max,
        )


class MiniPatchDecoder(CrossAttnPatchDecoder):
    """Backward-compatible alias for cross-attention patch decoder."""

    def __init__(self, d_var: int, cfg: MiniVAEConfig, d_dist: int | None = None) -> None:
        super().__init__(d_dist=int(d_dist) if d_dist is not None else int(d_var), cfg=cfg)


class MiniPatchVAE(nn.Module):
    """
    Patch-local VAE.

    Methods:
    - encode(w_patch, dist_var_tokens) -> mu, logvar
    - decode(z, patch_size, dist_patch_embed=None) -> w_hat
    - encode_patch(w_patch, dist_var_tokens) -> patch_token (for BigWeightVAE)
    """

    def __init__(self, d_var: int, cfg: MiniVAEConfig, d_dist: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = MiniPatchEncoder(d_var=d_var, cfg=cfg)
        self.decoder = CrossAttnPatchDecoder(d_dist=int(d_dist) if d_dist is not None else int(d_var), cfg=cfg)
        init_style = str(cfg.init_style).strip().lower()
        if init_style not in {"xavier", "llm"}:
            raise ValueError(f"mini init_style must be 'xavier' or 'llm', got {cfg.init_style!r}")
        self.apply(lambda module: _init_vae_module_weights(module, base_std=0.02, init_style=init_style))
        _init_vae_latent_parameters(self)
        if init_style == "llm":
            _apply_residual_scaled_init(
                self.encoder,
                L_stack=max(1, int(cfg.num_attn_layers_encoder)),
                base_std=0.02,
            )
            _apply_residual_scaled_init(
                self.decoder,
                L_stack=max(1, int(cfg.num_layers_decoder)),
                base_std=0.02,
            )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mu/logvar: [B, z_dim] -> z: [B, z_dim]
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mu/logvar: [B, z_dim] -> scalar
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
        return kl.mean()

    def encode(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)

    def decode(
        self,
        z: torch.Tensor,
        patch_size: int,
        dist_patch_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(z=z, patch_size=patch_size, dist_patch_embed=dist_patch_embed)

    def encode_patch(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_patch(w_patch=w_patch, dist_var_tokens=dist_var_tokens)

    def forward(
        self,
        w_patch: torch.Tensor,
        dist_var_tokens: torch.Tensor,
        dist_patch_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # w_patch: [B, p], dist_var_tokens: [B, p, d_var], dist_patch_embed: [B, d_dist] (optional)
        mu, logvar = self.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        if bool(self.cfg.use_latent_sampling):
            z = self.reparameterize(mu=mu, logvar=logvar)
        else:
            z = mu
        w_hat = self.decode(
            z=z,
            patch_size=w_patch.shape[1],
            dist_patch_embed=dist_patch_embed,
        )
        return w_hat, mu, logvar, z


class PerceiverResamplerBlock(nn.Module):
    """Perceiver-style resampler block over latent queries and input patch tokens."""

    def __init__(self, d_latent: int, d_token: int, n_heads: int, dropout: float, use_rope_2d: bool = False) -> None:
        super().__init__()
        if d_latent % n_heads != 0:
            raise ValueError(f"d_latent ({d_latent}) must be divisible by n_heads ({n_heads})")
        self.d_latent = int(d_latent)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_latent // n_heads)
        self.use_rope_2d = bool(use_rope_2d)
        self.norm_cross_q = nn.LayerNorm(d_latent)
        self.norm_cross_kv = nn.LayerNorm(d_token)
        self.cross_q_proj = nn.Linear(d_latent, d_latent)
        self.cross_k_proj = nn.Linear(d_token, d_latent)
        self.cross_v_proj = nn.Linear(d_token, d_latent)
        self.cross_out_proj = nn.Linear(d_latent, d_latent)
        self.norm_self = nn.LayerNorm(d_latent)
        self.self_q_proj = nn.Linear(d_latent, d_latent)
        self.self_k_proj = nn.Linear(d_latent, d_latent)
        self.self_v_proj = nn.Linear(d_latent, d_latent)
        self.self_out_proj = nn.Linear(d_latent, d_latent)
        self.norm_ffn = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, 4 * d_latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_latent, d_latent),
            nn.Dropout(dropout),
        )
        self.attn_prob_dropout_p = float(dropout)
        self.dropout = nn.Dropout(dropout)
        if self.use_rope_2d:
            self.rope_2d_cross = RoPeMixed2D(n_heads, self.head_dim)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        latent_pos: torch.Tensor,
        token_pos: torch.Tensor,
        token_pos2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_2d = self.use_rope_2d and token_pos2 is not None
        q = self.norm_cross_q(latents)
        kv = self.norm_cross_kv(tokens)
        B, L_lat, _ = q.shape
        L_tok = kv.shape[1]
        q_cross = self.cross_q_proj(q).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        k_cross = self.cross_k_proj(kv).view(B, L_tok, self.n_heads, self.head_dim).transpose(1, 2)
        v_cross = self.cross_v_proj(kv).view(B, L_tok, self.n_heads, self.head_dim).transpose(1, 2)

        # Cross-attention: latents ← tokens.
        # When 2D: K angles from (token_pos, token_pos2); Q angles from (latent_pos, latent_pos) — degenerate.
        if use_2d:
            q_angles = self.rope_2d_cross.compute_angles(latent_pos, latent_pos)
            k_angles = self.rope_2d_cross.compute_angles(token_pos, token_pos2)
            cross_out = _rope_attention_with_angles(
                q=q_cross, k=k_cross, v=v_cross,
                q_angles=q_angles, k_angles=k_angles,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )
        else:
            cross_out = _rope_attention(
                q=q_cross, k=k_cross, v=v_cross,
                q_pos=latent_pos, k_pos=token_pos,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )
        cross_out = cross_out.transpose(1, 2).contiguous().view(B, L_lat, self.d_latent)
        cross_out = self.cross_out_proj(cross_out)
        latents = latents + self.dropout(cross_out)

        # Self-attention among latents: always 1D (latents have no 2D spatial structure).
        lat_norm = self.norm_self(latents)
        q_self = self.self_q_proj(lat_norm).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        k_self = self.self_k_proj(lat_norm).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        v_self = self.self_v_proj(lat_norm).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        self_out = _rope_attention(
            q=q_self,
            k=k_self,
            v=v_self,
            q_pos=latent_pos,
            k_pos=latent_pos,
            dropout_p=self.attn_prob_dropout_p,
            training=self.training,
        )
        self_out = self_out.transpose(1, 2).contiguous().view(B, L_lat, self.d_latent)
        self_out = self.self_out_proj(self_out)
        latents = latents + self.dropout(self_out)

        latents = latents + self.dropout(self.ffn(self.norm_ffn(latents)))
        return latents


class TransformerNoCompressionPatchEncoder(nn.Module):
    """Debug encoder: Perceiver Resampler over patch tokens (no distribution conditioning)."""

    def __init__(self, d_var: int, cfg: MiniVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_var = int(d_var)
        if cfg.d_e % cfg.n_heads != 0:
            raise ValueError(f"mini d_e ({cfg.d_e}) must be divisible by n_heads ({cfg.n_heads})")
        if cfg.pos_dim <= 0:
            raise ValueError(f"mini pos_dim must be positive, got {cfg.pos_dim}")
        if int(cfg.stub_resampler_d_model) < 0:
            raise ValueError(
                f"mini stub_resampler_d_model must be >= 0, got {int(cfg.stub_resampler_d_model)}"
            )

        self.resampler_d_model = int(cfg.stub_resampler_d_model) if int(cfg.stub_resampler_d_model) > 0 else int(cfg.d_e)
        if self.resampler_d_model % cfg.n_heads != 0:
            raise ValueError(
                f"mini stub resampler d_model ({self.resampler_d_model}) must be divisible by n_heads ({cfg.n_heads})"
            )

        self.elem_embed = nn.Sequential(
            nn.Linear(1 + cfg.pos_dim, cfg.d_e),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_e, cfg.d_e),
            nn.Dropout(cfg.dropout),
        )
        self.num_latents = max(1, int(cfg.decoder_L_latents))
        self.resampler_latents = nn.Parameter(torch.randn(self.num_latents, self.resampler_d_model) * 0.02)
        latent_tau = (torch.arange(self.num_latents, dtype=torch.float32) + 0.5) / float(self.num_latents)
        self.register_buffer("latent_tau", latent_tau, persistent=False)
        self.resampler = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    d_latent=self.resampler_d_model,
                    d_token=int(cfg.d_e) + 1,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                )
                for _ in range(max(1, cfg.num_attn_layers_encoder))
            ]
        )
        self.flat_latent_dim = self.num_latents * self.resampler_d_model
        if int(cfg.z_dim) != self.flat_latent_dim:
            raise ValueError(
                "mini z_dim must equal flattened stub latent size: "
                "z_dim="
                f"{int(cfg.z_dim)} vs stub_resampler_d_model*decoder_L_latents="
                f"{self.resampler_d_model}*{self.num_latents}={self.flat_latent_dim}"
            )
        self.latent_norm = nn.LayerNorm(self.flat_latent_dim)
        self.to_mu = nn.Linear(self.flat_latent_dim, cfg.z_dim)
        self.to_logvar = nn.Linear(self.flat_latent_dim, cfg.z_dim)
        if cfg.d_patch == cfg.z_dim:
            self.patch_proj = nn.Identity()
        else:
            self.patch_proj = nn.Linear(cfg.z_dim, cfg.d_patch)

    def encode(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        if dist_var_tokens.ndim != 3:
            raise ValueError(f"dist_var_tokens must be [B, p, d_var], got {tuple(dist_var_tokens.shape)}")

        B, p = w_patch.shape
        if tuple(dist_var_tokens.shape[:2]) != (B, p):
            raise ValueError(f"Shape mismatch: w_patch={tuple(w_patch.shape)} dist_var_tokens={tuple(dist_var_tokens.shape)}")
        if int(dist_var_tokens.shape[2]) != self.d_var:
            raise ValueError(f"dist_var_tokens last dim must be {self.d_var}, got {int(dist_var_tokens.shape[2])}")

        # Debug path: intentionally ignore distribution tokens to isolate model-only behavior.
        token_tau = (torch.arange(p, device=w_patch.device, dtype=torch.float32) + 0.5) / float(p)
        pos = sinusoidal_embedding(torch.arange(p, device=w_patch.device), dim=self.cfg.pos_dim).to(dtype=w_patch.dtype)
        pos_expand = pos.unsqueeze(0).expand(B, -1, -1)

        token_in = torch.cat([w_patch.unsqueeze(-1), pos_expand], dim=-1)
        e = self.elem_embed(token_in)

        latents = self.resampler_latents.unsqueeze(0).expand(B, -1, -1)
        tokens_with_w = torch.cat([e, w_patch.unsqueeze(-1).to(dtype=e.dtype)], dim=-1)
        latent_pos = self.latent_tau.to(device=w_patch.device)
        for block in self.resampler:
            latents = block(
                latents=latents,
                tokens=tokens_with_w,
                latent_pos=latent_pos,
                token_pos=token_tau,
            )
        h = self.latent_norm(latents.reshape(B, self.flat_latent_dim))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def encode_patch(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        return self.patch_proj(mu)


def _build_stub_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    num_linear_layers: int = 6,
    dropout: float = 0.0,
    use_batchnorm: bool = False,
) -> nn.Sequential:
    """Build an MLP with an explicit number of Linear layers."""
    if num_linear_layers < 2:
        raise ValueError(f"num_linear_layers must be >= 2, got {num_linear_layers}")
    layers: list[nn.Module] = []
    cur_dim = int(in_dim)
    for idx in range(int(num_linear_layers)):
        is_last = idx == int(num_linear_layers) - 1
        next_dim = int(out_dim) if is_last else int(hidden_dim)
        layers.append(nn.Linear(cur_dim, next_dim))
        if not is_last:
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
        cur_dim = next_dim
    return nn.Sequential(*layers)


def _align_feature_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or trim feature dimension to target_dim."""
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor [B, d], got {tuple(x.shape)}")
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    current_dim = int(x.shape[1])
    if current_dim == target_dim:
        return x
    if current_dim > target_dim:
        return x[:, :target_dim]
    pad = x.new_zeros((x.shape[0], target_dim - current_dim))
    return torch.cat([x, pad], dim=1)


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

        # Transform branch: processes concatenated w_patch + dist_patch_embed.
        residual_in_dim = self.p + int(d_dist)
        layers: list[nn.Module] = []
        in_dim = residual_in_dim
        for i in range(max(1, int(num_layers)) - 1):
            layers.extend([
                nn.Linear(in_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.d_patch))
        self.transform = nn.Sequential(*layers)

        # Linear projection of raw weights for residual path.
        self.weight_residual_proj = nn.Linear(self.p, self.d_patch, bias=False)

        self._init_near_identity()

    def _init_near_identity(self) -> None:
        """Initialize so output ≈ linear projection of w_patch (transform starts near zero)."""
        with torch.no_grad():
            # Zero out the last layer of transform so residual dominates at init.
            last_linear: nn.Linear | None = None
            for module in reversed(list(self.transform.modules())):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

            # weight_residual_proj: identity-like (truncated/padded if p != d_patch).
            min_dim = min(self.p, self.d_patch)
            nn.init.zeros_(self.weight_residual_proj.weight)
            self.weight_residual_proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)

    def forward(self, w_patch: torch.Tensor, dist_patch_embed: torch.Tensor) -> torch.Tensor:
        # w_patch: [B, p], dist_patch_embed: [B, d_dist]
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        if dist_patch_embed.ndim != 2:
            raise ValueError(f"dist_patch_embed must be [B, d_dist], got {tuple(dist_patch_embed.shape)}")

        residual = self.weight_residual_proj(w_patch)  # [B, d_patch]
        x = torch.cat([w_patch, dist_patch_embed], dim=-1)  # [B, p + d_dist]
        delta = self.transform(x)  # [B, d_patch]
        return residual + delta


class MLPNoCompressionPatchEncoder(nn.Module):
    """Stub encoder: 6-layer MLP from patch weights to latent stats."""

    def __init__(self, d_var: int, cfg: MiniVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_var = int(d_var)
        self.in_dim = max(1, int(cfg.d_patch))
        hidden = max(8, int(cfg.mlp_stub_hidden_dim))
        self.latent_dim = int(cfg.z_dim)
        self.mlp = _build_stub_mlp(
            in_dim=self.in_dim,
            hidden_dim=hidden,
            out_dim=2 * self.latent_dim,
            num_linear_layers=6,
            dropout=float(cfg.dropout),
            use_batchnorm=bool(cfg.stub_mlp_use_batchnorm),
        )
        if cfg.d_patch == cfg.z_dim:
            self.patch_proj = nn.Identity()
        else:
            self.patch_proj = nn.Linear(cfg.z_dim, cfg.d_patch)

    def encode(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        if dist_var_tokens.ndim != 3:
            raise ValueError(f"dist_var_tokens must be [B, p, d_var], got {tuple(dist_var_tokens.shape)}")
        B, p = w_patch.shape
        if tuple(dist_var_tokens.shape[:2]) != (B, p):
            raise ValueError(f"Shape mismatch: w_patch={tuple(w_patch.shape)} dist_var_tokens={tuple(dist_var_tokens.shape)}")
        if int(dist_var_tokens.shape[2]) != self.d_var:
            raise ValueError(f"dist_var_tokens last dim must be {self.d_var}, got {int(dist_var_tokens.shape[2])}")

        w_in = _align_feature_dim(w_patch, target_dim=self.in_dim)
        stats = self.mlp(w_in)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def encode_patch(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        return self.patch_proj(mu)


class MLPNoCompressionPatchDecoder(nn.Module):
    """Stub decoder: 6-layer MLP from latent vector to full patch."""

    def __init__(self, d_dist: int, cfg: MiniVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_dist_conditioning = bool(cfg.decoder_use_dist_conditioning)

        self.latent_dim = int(cfg.z_dim)
        self.out_dim = max(1, int(cfg.d_patch))
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0
        hidden = max(4, int(cfg.mlp_stub_hidden_dim))
        in_dim = self.latent_dim + (1 if self.use_dist_conditioning else 0)
        self.dist_scalar = nn.Linear(int(d_dist), 1) if self.use_dist_conditioning else None
        self.mlp = _build_stub_mlp(
            in_dim=in_dim,
            hidden_dim=hidden,
            out_dim=self.out_dim + 1,
            num_linear_layers=6,
            dropout=float(cfg.dropout),
            use_batchnorm=bool(cfg.stub_mlp_use_batchnorm),
        )

    def forward(
        self,
        z: torch.Tensor,
        patch_size: int,
        dist_patch_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be [B, z_dim_like], got {tuple(z.shape)}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        B, z_dim = z.shape
        p = int(patch_size)
        if z_dim >= self.latent_dim:
            z_aligned = z[:, : self.latent_dim]
        else:
            pad = z.new_zeros((B, self.latent_dim - z_dim))
            z_aligned = torch.cat([z, pad], dim=1)

        dec_in = z_aligned
        
        if self.use_dist_conditioning:
            if self.dist_scalar is None or dist_patch_embed is None:
                dist_scalar = dec_in.new_zeros((B, 1))
            else:
                dist_scalar = self.dist_scalar(dist_patch_embed)
            dec_in = torch.cat([dec_in, dist_scalar], dim=-1)

        out = self.mlp(dec_in)
        u_hat = out[:, : self.out_dim]
        s_hat = out[:, self.out_dim]

        if self.out_dim == p:
            u_hat_p = u_hat
        elif self.out_dim > p:
            u_hat_p = u_hat[:, :p]
        else:
            pad = u_hat.new_zeros((B, p - self.out_dim))
            u_hat_p = torch.cat([u_hat, pad], dim=1)
        return _decode_direction_and_logscale(
            u_hat=u_hat_p,
            s_hat=s_hat,
            eps=self.output_eps,
            s_min=self.output_s_min,
            s_max=self.output_s_max,
        )


class MiniPatchVAEStub(nn.Module):
    """
    Debug Mini-VAE replacement:
    - encoder: Perceiver Resampler over patch tokens (without distribution conditioning)
    - decoder: 6-layer MLP from latent to full patch
    """

    def __init__(self, d_var: int, cfg: MiniVAEConfig, d_dist: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        decoder_d_dist = int(d_dist) if d_dist is not None else int(d_var)
        self.encoder = TransformerNoCompressionPatchEncoder(d_var=d_var, cfg=cfg)
        self.decoder = MLPNoCompressionPatchDecoder(d_dist=decoder_d_dist, cfg=cfg)
        init_style = str(cfg.init_style).strip().lower()
        if init_style not in {"xavier", "llm"}:
            raise ValueError(f"mini init_style must be 'xavier' or 'llm', got {cfg.init_style!r}")
        self.apply(lambda module: _init_vae_module_weights(module, base_std=0.02, init_style=init_style))
        _init_vae_latent_parameters(self)
        if init_style == "llm":
            _apply_residual_scaled_init(
                self.encoder,
                L_stack=max(1, int(cfg.num_attn_layers_encoder)),
                base_std=0.02,
            )
            _apply_residual_scaled_init(
                self.decoder,
                L_stack=max(1, int(cfg.num_layers_decoder)),
                base_std=0.02,
            )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu.new_zeros(())

    def encode(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)

    def decode(
        self,
        z: torch.Tensor,
        patch_size: int,
        dist_patch_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(z=z, patch_size=patch_size, dist_patch_embed=dist_patch_embed)

    def encode_patch(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_patch(w_patch=w_patch, dist_var_tokens=dist_var_tokens)

    def forward(
        self,
        w_patch: torch.Tensor,
        dist_var_tokens: torch.Tensor,
        dist_patch_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        if bool(self.cfg.use_latent_sampling):
            z = self.reparameterize(mu=mu, logvar=logvar)
        else:
            z = mu
        w_hat = self.decode(
            z=z,
            patch_size=w_patch.shape[1],
            dist_patch_embed=dist_patch_embed,
        )
        return w_hat, mu, logvar, z


@dataclass(slots=True)
class EncoderConfig:
    self_attn_mode: str = "full"  # {"full", "cls_only"}
    cross_attend_only_cls: bool = True


@dataclass(slots=True)
class BigVAEConfig:
    d_model: int = 256
    d_lat: int = 256
    num_latents: int = 32
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    n_heads: int = 8
    ffn_mult: float = 4.0
    dropout: float = 0.0
    pos_fourier_dim: int = 64
    use_latent_sampling: bool = True
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


# Backward-compat placeholder (not used by BigWeightVAE internals).
@dataclass(slots=True)
class ResamplerConfig:
    n_layers: int = 0


@dataclass(slots=True)
class ModelConfig:
    patch_size: int = 16
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    mini_vae: MiniVAEConfig = field(default_factory=MiniVAEConfig)
    big_vae: BigVAEConfig = field(default_factory=BigVAEConfig)
    beta: float = 1e-3


class LocalOutputSelfAttentionBlock(nn.Module):
    """Local attention within one output-column token group, with RoPE on patch positions."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float, self_attn_mode: str) -> None:
        super().__init__()
        if self_attn_mode not in {"full", "cls_only"}:
            raise ValueError(f"encoder.self_attn_mode must be 'full' or 'cls_only', got {self_attn_mode}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.self_attn_mode = self_attn_mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_prob_dropout_p = float(dropout)

        # Shared q/k/v projections for self-attention with RoPE.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        hidden = max(1, int(d_model * ffn_mult))
        self.ffn = MLP(d_model, hidden, d_model, dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B_group, 1 + T, d_model]
        B, S, _ = tokens.shape
        h = self.norm_attn(tokens)

        # Positions: CLS=0, patch_0=1, ..., patch_{T-1}=T.
        pos = torch.arange(S, device=tokens.device, dtype=torch.float32)

        q_all = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k_all = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v_all = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if self.self_attn_mode == "full":
            attn_out = _rope_attention(
                q=q_all, k=k_all, v=v_all,
                q_pos=pos, k_pos=pos,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )
        else:
            # CLS query attends to all tokens.
            cls_out = _rope_attention(
                q=q_all[:, :, 0:1, :], k=k_all, v=v_all,
                q_pos=pos[0:1], k_pos=pos,
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )  # [B, H, 1, D_h]
            # Patch queries attend only to CLS.
            patch_out = _rope_attention(
                q=q_all[:, :, 1:, :], k=k_all[:, :, 0:1, :], v=v_all[:, :, 0:1, :],
                q_pos=pos[1:], k_pos=pos[0:1],
                dropout_p=self.attn_prob_dropout_p, training=self.training,
            )  # [B, H, T, D_h]
            attn_out = torch.cat([cls_out, patch_out], dim=2)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        tokens = tokens + self.dropout(self.out_proj(attn_out))
        tokens = tokens + self.dropout(self.ffn(self.norm_ffn(tokens)))
        return tokens


class LatentEncoderLayer(nn.Module):
    """
    One big-encoder block:
    1) local per-output attention over [CLS_o, patches_o]
    2) Perceiver Resampler: latents cross-attend to tokens with RoPE,
       then latent self-attention with RoPE, then FFN
    """

    def __init__(self, d_model: int, d_lat: int, n_heads: int, ffn_mult: float, dropout: float, self_attn_mode: str, use_rope_2d: bool = False) -> None:
        super().__init__()
        if d_lat % n_heads != 0:
            raise ValueError(f"d_lat ({d_lat}) must be divisible by n_heads ({n_heads})")

        self.local_block = LocalOutputSelfAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            self_attn_mode=self_attn_mode,
        )

        self.perceiver_block = PerceiverResamplerBlock(
            d_latent=d_lat,
            d_token=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope_2d=use_rope_2d,
        )

    def forward(
        self,
        tokens_by_output: torch.Tensor,
        latents: torch.Tensor,
        cross_attend_only_cls: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens_by_output: [B, d_out, 1 + T, d_model]
        # latents: [B, L, d_lat]
        B, d_out, L_local, d_model = tokens_by_output.shape
        T = L_local - 1  # number of patches per output

        # Local self-attention per output group (1D RoPE with t-positions, o is constant within group).
        local_in = tokens_by_output.view(B * d_out, L_local, d_model)  # [B*d_out, 1+T, d_model]
        local_out = self.local_block(local_in)  # [B*d_out, 1+T, d_model]
        tokens = local_out.view(B, d_out, L_local, d_model)  # [B, d_out, 1+T, d_model]

        # K/V source for Perceiver cross-attention.
        device = tokens.device
        if cross_attend_only_cls:
            kv = tokens[:, :, 0, :]  # [B, d_out, d_model]
            # Each CLS token has (o=output_index, t=0).
            token_pos_o = torch.arange(d_out, device=device, dtype=torch.float32)
            token_pos_t = torch.zeros(d_out, device=device, dtype=torch.float32)
        else:
            kv = tokens.reshape(B, d_out * L_local, d_model)  # [B, d_out*(1+T), d_model]
            # Tokens ordered by output: [CLS_0, patch_0_0, ..., CLS_1, patch_1_0, ...]
            token_pos_o = torch.arange(d_out, device=device, dtype=torch.float32).repeat_interleave(L_local)
            token_pos_t = torch.arange(L_local, device=device, dtype=torch.float32).repeat(d_out)

        # Latent positions (abstract, 1D).
        L = latents.shape[1]
        latent_pos = (torch.arange(L, device=device, dtype=torch.float32) + 0.5) / max(float(L), 1.0)

        # Perceiver Resampler with 2D RoPE: cross-attn keys get (o, t) positions.
        latents = self.perceiver_block(
            latents=latents,
            tokens=kv,
            latent_pos=latent_pos,
            token_pos=token_pos_o,
            token_pos2=token_pos_t,
        )
        return tokens, latents


class DecoderCrossBlock(nn.Module):
    """Decoder block with query->latent cross-attention only (no q-q self-attention)."""

    def __init__(self, d_model: int, d_lat: int, n_heads: int, ffn_mult: float, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_lat)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=d_lat,
            vdim=d_lat,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = max(1, int(d_model * ffn_mult))
        self.ffn = MLP(d_model, hidden, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens: torch.Tensor, z_latents: torch.Tensor) -> torch.Tensor:
        # q_tokens: [B, Q, d_model]
        # z_latents: [B, L, d_lat]
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(z_latents)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q_tokens = q_tokens + self.dropout(cross_out)
        q_tokens = q_tokens + self.dropout(self.ffn(self.norm_ffn(q_tokens)))
        return q_tokens


class BigWeightVAE(nn.Module):
    """
    Full matrix VAE.

    Input:
    - W: [B, d_in, d_out] or [d_in, d_out]
    - X: [B, n, d_in] or [n, d_in]

    Output:
    - W_hat: same rank as W input
    - mu: [B, z_dim] (or [z_dim] for unbatched input)
    - logvar: [B, z_dim] (or [z_dim] for unbatched input)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        p = cfg.patch_size
        d_var = cfg.distribution.d_var
        d_dist = cfg.distribution.d_dist

        self.distribution_encoder = InputDistributionEncodingModule(cfg.distribution)

        # Simple residual tokenizer.
        d_patch = cfg.mini_vae.d_patch
        self.patch_tokenizer = ResidualPatchTokenizer(
            p=p,
            d_dist=d_dist,
            d_patch=d_patch,
            hidden_dim=max(64, int(cfg.mini_vae.mlp_stub_hidden_dim)),
            num_layers=3,
            dropout=cfg.big_vae.dropout,
        )

        d_model = cfg.big_vae.d_model
        d_lat = cfg.big_vae.d_lat
        n_heads = cfg.big_vae.n_heads
        num_latents = cfg.big_vae.num_latents
        num_enc_layers = max(1, cfg.big_vae.num_encoder_layers)
        num_dec_layers = max(1, cfg.big_vae.num_decoder_layers)
        dropout = cfg.big_vae.dropout

        if d_model % n_heads != 0:
            raise ValueError(f"big_vae.d_model ({d_model}) must be divisible by big_vae.n_heads ({n_heads})")
        if d_lat % n_heads != 0:
            raise ValueError(f"big_vae.d_lat ({d_lat}) must be divisible by big_vae.n_heads ({n_heads})")

        # Patch token projection.
        self.patch_token_proj = nn.Linear(d_patch, d_model)

        # One learned CLS token expanded per output column.
        self.cls_token = nn.Parameter(torch.zeros(d_model))

        # --- Encoder layers with per-layer dist_var injection ---
        self.encoder_layers = nn.ModuleList(
            [
                LatentEncoderLayer(
                    d_model=d_model,
                    d_lat=d_lat,
                    n_heads=n_heads,
                    ffn_mult=cfg.big_vae.ffn_mult,
                    dropout=dropout,
                    self_attn_mode=cfg.big_vae.encoder.self_attn_mode,
                    use_rope_2d=True,
                )
                for _ in range(num_enc_layers)
            ]
        )
        # Per-layer: concat dist_var to patch tokens and project back to d_model.
        self.enc_dist_inject_projs = nn.ModuleList(
            [nn.Linear(d_model + d_var, d_model) for _ in range(num_enc_layers)]
        )
        # Per-layer: non-trivial head from dist_var → additive vector on flattened latents.
        self.flat_lat_dim = num_latents * d_lat
        self.enc_dist_to_latent_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_var, d_var * 2),
                    nn.GELU(),
                    nn.Linear(d_var * 2, d_var * 2),
                    nn.GELU(),
                    nn.Linear(d_var * 2, self.flat_lat_dim),
                )
                for _ in range(num_enc_layers)
            ]
        )

        # Global latent base: [L, d_lat]
        self.latent_base = nn.Parameter(torch.randn(num_latents, d_lat) * 0.02)

        # Flatten latents → mu/logvar (like mini-VAE).
        z_dim = cfg.mini_vae.z_dim
        self.z_dim = z_dim
        self.latent_norm = nn.LayerNorm(self.flat_lat_dim)
        self.to_mu = nn.Linear(self.flat_lat_dim, z_dim)
        self.to_logvar = nn.Linear(self.flat_lat_dim, z_dim)

        # --- Decoder: CrossAttnBlock (cross-attn + self-attn + FFN, all with RoPE) ---
        # z → decoder latent tokens for KV.
        self.dec_L_latents = max(1, int(cfg.mini_vae.decoder_L_latents))
        self.z_to_latents = nn.Linear(z_dim, self.dec_L_latents * d_model)

        # Decoder query construction with (o, t) positional encoding.
        self.pos_proj = nn.Linear(2 * cfg.big_vae.pos_fourier_dim, d_model)
        self.query_proj = nn.Linear(d_model + d_dist, d_model)
        self.query_pos_proj = nn.Linear(d_model, d_model)

        self.decoder_layers = nn.ModuleList(
            [
                CrossAttnBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, use_rope_2d=True)
                for _ in range(num_dec_layers)
            ]
        )

        # Direction/scale decomposition output (like mini-VAE).
        # Each decoder query = one patch (o,t), so heads output p-dimensional direction.
        self.direction_head = nn.Linear(d_model, p)
        self.scale_head = nn.Linear(d_model, 1)
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0
        self.direction_seq_norm_eps = 1e-6

        # Z-shortcut: direct z + positional → p-dimensional direction, bypassing cross-attn.
        self.z_shortcut = nn.Sequential(
            nn.Linear(z_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, p),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
        return kl.mean()

    @staticmethod
    def structural_recon_loss(W: torch.Tensor, W_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if W.ndim == 2:
            W = W.unsqueeze(0)
            W_hat = W_hat.unsqueeze(0)
        w_scale = W.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(float(eps))
        diff = (W_hat - W) / w_scale
        return diff.pow(2).mean()

    @staticmethod
    def operator_recon_loss(X: torch.Tensor, W: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        d_in = W.shape[-2]
        if X.ndim == 2:
            return (F.mse_loss(X @ W_hat, X @ W) / d_in).sqrt()
        return (F.mse_loss(torch.matmul(X, W_hat), torch.matmul(X, W)) / d_in).sqrt()

    @staticmethod
    def _channel_norm_over_sequence(x: torch.Tensor, eps: float) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,C], got {tuple(x.shape)}")
        if int(x.shape[1]) <= 1:
            return x
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + float(eps))

    @staticmethod
    def _build_patch_indices(d_in: int, patch_size: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
        T = (d_in + patch_size - 1) // patch_size
        d_in_pad = T * patch_size
        patch_idx_t = torch.arange(d_in_pad, device=device, dtype=torch.long).view(T, patch_size)
        patch_idx_t = patch_idx_t.clamp(max=max(0, d_in - 1))
        return patch_idx_t, T, d_in_pad

    def forward(self, W: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if W.ndim not in {2, 3}:
            raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
        if X.ndim not in {2, 3}:
            raise ValueError(f"X must be rank-2 or rank-3, got {tuple(X.shape)}")

        squeeze_batch = W.ndim == 2
        if squeeze_batch != (X.ndim == 2):
            raise ValueError(f"W and X must be both batched or both unbatched, got W={tuple(W.shape)}, X={tuple(X.shape)}")

        if squeeze_batch:
            W = W.unsqueeze(0)
            X = X.unsqueeze(0)

        B, d_in, d_out = W.shape
        Bx, n, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        device = W.device
        dtype = W.dtype
        p = self.cfg.patch_size
        d_model = self.cfg.big_vae.d_model
        patch_idx_t, T, d_in_pad = self._build_patch_indices(d_in=d_in, patch_size=p, device=device)

        # ---------------------------------------------------------------------
        # 1) Distribution encoding per input patch t.
        # ---------------------------------------------------------------------
        patch_idx_bt = patch_idx_t.unsqueeze(0).expand(B, -1, -1).contiguous()
        X_rep = X.unsqueeze(1).expand(B, T, n, d_in).reshape(B * T, n, d_in)
        patch_idx_flat = patch_idx_bt.reshape(B * T, p)

        dist_var_flat, dist_patch_flat = self.distribution_encoder(X_rep, patch_idx_flat)

        d_var = self.cfg.distribution.d_var
        d_dist = self.cfg.distribution.d_dist
        dist_var_by_patch = dist_var_flat.view(B, T, p, d_var)
        dist_patch_by_patch = dist_patch_flat.view(B, T, d_dist)

        # Mean-pool dist_var over patch variables: [B, T, d_var]
        dist_var_pooled = dist_var_by_patch.mean(dim=2)

        # ---------------------------------------------------------------------
        # 2) Patchify W.
        # ---------------------------------------------------------------------
        W_pad = torch.zeros(B, d_in_pad, d_out, device=device, dtype=dtype)
        W_pad[:, :d_in, :] = W
        w_patches = W_pad.transpose(1, 2).contiguous().view(B, d_out, T, p)

        # ---------------------------------------------------------------------
        # 3) Patch embedding using residual tokenizer.
        # ---------------------------------------------------------------------
        dist_patch_expanded = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1)
        w_flat = w_patches.reshape(B * d_out * T, p)
        dist_patch_flat_expanded = dist_patch_expanded.reshape(B * d_out * T, d_dist)

        d_patch = self.cfg.mini_vae.d_patch
        patch_token_raw_flat = self.patch_tokenizer(
            w_patch=w_flat, dist_patch_embed=dist_patch_flat_expanded,
        )
        patch_token_raw = patch_token_raw_flat.view(B, d_out, T, d_patch)
        patch_tokens = self.patch_token_proj(patch_token_raw)

        cls_tokens = self.cls_token.view(1, 1, 1, -1).expand(B, d_out, 1, -1)
        tokens_by_output = torch.cat([cls_tokens, patch_tokens], dim=2)

        # ---------------------------------------------------------------------
        # 4) Encoder with dist_var injection + dist→latent heads per layer.
        # ---------------------------------------------------------------------
        # dist_var for patch token injection: [B, d_out, T, d_var]
        dist_var_for_inject = dist_var_pooled.unsqueeze(1).expand(-1, d_out, -1, -1)
        # Global dist_var for latent head: [B, d_var]
        dist_var_global = dist_var_pooled.mean(dim=1)

        latents = self.latent_base.unsqueeze(0).expand(B, -1, -1)
        num_latents = self.cfg.big_vae.num_latents
        d_lat = self.cfg.big_vae.d_lat

        for layer_idx, enc_layer in enumerate(self.encoder_layers):
            # Inject dist_var into patch tokens (not CLS) before each layer.
            cls_part = tokens_by_output[:, :, :1, :]
            patch_part = tokens_by_output[:, :, 1:, :]
            patch_with_dist = torch.cat([patch_part, dist_var_for_inject], dim=-1)
            patch_part = self.enc_dist_inject_projs[layer_idx](patch_with_dist)
            tokens_by_output = torch.cat([cls_part, patch_part], dim=2)

            # Dist→latent additive head.
            dist_lat_delta = self.enc_dist_to_latent_heads[layer_idx](dist_var_global)
            latents = latents + dist_lat_delta.view(B, num_latents, d_lat)

            tokens_by_output, latents = enc_layer(
                tokens_by_output=tokens_by_output,
                latents=latents,
                cross_attend_only_cls=self.cfg.big_vae.encoder.cross_attend_only_cls,
            )

        # Flatten latents → mu/logvar.
        latents_flat = self.latent_norm(latents.reshape(B, self.flat_lat_dim))
        mu = self.to_mu(latents_flat)
        logvar = self.to_logvar(latents_flat)
        if bool(self.cfg.big_vae.use_latent_sampling):
            z = self.reparameterize(mu=mu, logvar=logvar)
        else:
            z = mu

        # ---------------------------------------------------------------------
        # 5) Decoder: CrossAttnBlock + direction/scale decomposition.
        # ---------------------------------------------------------------------
        lat = self.z_to_latents(z).view(B, self.dec_L_latents, d_model)

        # Build (o, t) positional queries.
        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")

        pos_dim = self.cfg.big_vae.pos_fourier_dim
        pos_o = sinusoidal_embedding(o_grid.to(torch.float32), pos_dim)
        pos_t = sinusoidal_embedding(t_grid.to(torch.float32), pos_dim)
        pos_ot = torch.cat([pos_o, pos_t], dim=-1)  # [d_out, T, 2*pos_dim]

        q_base = self.pos_proj(pos_ot)  # [d_out, T, d_model]
        q_pos_emb = self.query_pos_proj(q_base)  # [d_out, T, d_model]
        q_base_expanded = q_base.unsqueeze(0).expand(B, -1, -1, -1)
        q_cond_cat = torch.cat([q_base_expanded, dist_patch_expanded], dim=-1)
        q_tokens = self.query_proj(q_cond_cat).reshape(B, d_out * T, d_model)

        Q = d_out * T
        # 2D RoPE positions for decoder queries: (o, t) per patch.
        q_pos_o = o_grid.flatten().to(dtype=torch.float32)  # [Q]
        q_pos_t = t_grid.flatten().to(dtype=torch.float32)  # [Q]
        # KV (latent tokens) have abstract 1D positions — use same for both dims (degenerate 2D).
        kv_pos = torch.arange(self.dec_L_latents, device=device, dtype=torch.float32)

        for dec_layer in self.decoder_layers:
            q_tokens = dec_layer(
                q=q_tokens, kv=lat,
                q_pos=q_pos_o, kv_pos=kv_pos,
                q_pos2=q_pos_t, kv_pos2=kv_pos,
            )

        # Direction head: [B, Q, d_model] → [B, Q, p] (p-dim direction per patch).
        q_dir = self._channel_norm_over_sequence(q_tokens, eps=self.direction_seq_norm_eps)
        u_hat_attn = self.direction_head(q_dir)  # [B, Q, p]

        # Z-shortcut: z + positional → p-dim direction per patch.
        z_exp = z.unsqueeze(1).expand(B, Q, -1)
        pos_exp = q_pos_emb.reshape(1, Q, d_model).expand(B, -1, -1)
        shortcut_in = torch.cat([z_exp, pos_exp], dim=-1)
        u_hat_shortcut = self.z_shortcut(shortcut_in)  # [B, Q, p]

        u_hat = (u_hat_attn + u_hat_shortcut).reshape(B * Q, p)  # [B*Q, p]

        # Scale: one scalar per patch from mean-pooled decoder output.
        s_hat = self.scale_head(q_tokens).reshape(B * Q)  # [B*Q]

        # Direction/scale → w_hat per patch.
        w_hat_flat = _decode_direction_and_logscale(
            u_hat=u_hat, s_hat=s_hat,
            eps=self.output_eps, s_min=self.output_s_min, s_max=self.output_s_max,
        )  # [B*Q, p]

        # Stitch patches back to matrix.
        w_hat_patches = w_hat_flat.view(B, d_out, T, p)
        W_hat_pad = w_hat_patches.view(B, d_out, d_in_pad).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]

        if squeeze_batch:
            return W_hat.squeeze(0), mu.squeeze(0), logvar.squeeze(0)

        return W_hat, mu, logvar


# Backward-compat alias used in other files.
WeightQuantileVAE = BigWeightVAE


def _smoke_test() -> None:
    torch.manual_seed(0)
    B, p, z_dim, d_dist = 4, 64, 32, 128
    cfg = MiniVAEConfig(
        z_dim=z_dim,
        d_e=128,
        num_layers_decoder=2,
        n_heads=4,
        decoder_L_latents=8,
        decoder_use_dist_conditioning=True,
        decoder_dist_mode="add",
    )
    decoder = CrossAttnPatchDecoder(d_dist=d_dist, cfg=cfg)
    z = torch.randn(B, z_dim, requires_grad=True)
    dist_patch_embed = torch.randn(B, d_dist)
    w_hat = decoder(z=z, patch_size=p, dist_patch_embed=dist_patch_embed)
    assert tuple(w_hat.shape) == (B, p), f"unexpected shape: {tuple(w_hat.shape)}"
    loss = w_hat.pow(2).mean()
    loss.backward()
    grad_sum = 0.0
    for param in decoder.z_to_latents.parameters():
        if param.grad is not None:
            grad_sum += float(param.grad.abs().sum().item())
    assert grad_sum > 0.0, "expected non-zero grads for z_to_latents parameters"


def smoke_test_big_weight_vae() -> None:
    torch.manual_seed(0)

    cfg = ModelConfig(
        patch_size=16,
        distribution=DistributionConfig(
            k_s=16, Kq=32, d_var=128, d_dist=128,
            use_covariance=True, patch_size_for_cov=16,
        ),
        mini_vae=MiniVAEConfig(
            z_dim=64,
            d_e=128,
            encoder_latent_dim=8,
            num_attn_layers_encoder=2,
            num_layers_decoder=2,
            n_heads=4,
            d_patch=64,
            mlp_stub_hidden_dim=256,
        ),
        big_vae=BigVAEConfig(
            d_model=256,
            d_lat=256,
            num_latents=16,
            num_encoder_layers=2,
            num_decoder_layers=2,
            n_heads=8,
            encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
        ),
    )

    model = BigWeightVAE(cfg)

    B, n, d_in, d_out = 2, 32, 65, 7
    X = torch.randn(B, n, d_in)
    W = torch.randn(B, d_in, d_out)

    W_hat, mu, logvar = model(W, X)

    assert tuple(W_hat.shape) == (B, d_in, d_out), f"W_hat shape mismatch: {tuple(W_hat.shape)}"
    assert tuple(mu.shape) == (B, 64), f"mu shape mismatch: {tuple(mu.shape)}"

    # Test backward pass and gradient flow.
    behavioral_loss = BigWeightVAE.operator_recon_loss(X, W, W_hat)
    structural_loss = BigWeightVAE.structural_recon_loss(W, W_hat)
    kl_loss = BigWeightVAE.kl_loss(mu, logvar)
    total_loss = behavioral_loss + 0.5 * structural_loss + 0.001 * kl_loss
    total_loss.backward()

    # Check no NaN grads.
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

    print("smoke_test_big_weight_vae PASSED")
    print(f"  W_hat: {tuple(W_hat.shape)}, behavioral={behavioral_loss.item():.4f}, "
          f"structural={structural_loss.item():.4f}, kl={kl_loss.item():.4f}")


if __name__ == "__main__":
    smoke_test_big_weight_vae()
