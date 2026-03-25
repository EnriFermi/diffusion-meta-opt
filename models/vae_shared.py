from __future__ import annotations

import copy
import math

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


def _masked_quantile_over_samples(X_q: torch.Tensor, q: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """
    Mask-aware quantile over sample axis.

    X_q: [B, n, p]
    sample_mask: [B, n] with True for valid rows
    q: [k]
    returns: [k, B, p]
    """
    if X_q.ndim != 3:
        raise ValueError(f"X_q must be [B,n,p], got {tuple(X_q.shape)}")
    if q.ndim != 1:
        raise ValueError(f"q must be rank-1, got {tuple(q.shape)}")
    if sample_mask.ndim != 2:
        raise ValueError(f"sample_mask must be [B,n], got {tuple(sample_mask.shape)}")
    if tuple(sample_mask.shape) != tuple(X_q.shape[:2]):
        raise ValueError(
            f"sample_mask shape must match X_q sample axes, got {tuple(sample_mask.shape)} vs {tuple(X_q.shape[:2])}"
        )

    mask = sample_mask.to(device=X_q.device, dtype=torch.bool)
    valid_counts = mask.sum(dim=1).clamp_min(1)
    fill_value = torch.full((), float("inf"), device=X_q.device, dtype=X_q.dtype)
    masked_values = torch.where(mask.unsqueeze(-1), X_q, fill_value)
    sorted_vals, _ = torch.sort(masked_values, dim=1)

    q = q.to(device=X_q.device, dtype=X_q.dtype).clamp(0.0, 1.0)
    pos = q.unsqueeze(0) * (valid_counts.to(dtype=X_q.dtype).unsqueeze(1) - 1.0)

    lower = torch.floor(pos).to(dtype=torch.long)
    upper = torch.ceil(pos).to(dtype=torch.long)
    alpha = (pos - lower.to(dtype=X_q.dtype)).unsqueeze(-1)

    B, _, p = sorted_vals.shape
    k = q.shape[0]
    lower_idx = lower.unsqueeze(-1).expand(B, k, p)
    upper_idx = upper.unsqueeze(-1).expand(B, k, p)

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
    angles = pos[:, None] * inv_freq[None, :]
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
    attn_dropout = float(dropout_p) if training else 0.0
    return F.scaled_dot_product_attention(
        q_rot,
        k_rot,
        v,
        attn_mask=None,
        dropout_p=attn_dropout,
        is_causal=False,
    )


class RoPeMixed2D(nn.Module):
    """
    Learned mixed 2D RoPE (arXiv:2403.13298, Eq. 14).

    angle[h, t] = freq1[h, t] * pos1 + freq2[h, t] * pos2

    Every head dimension encodes both positional axes simultaneously.
    """

    def __init__(self, n_heads: int, head_dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        half = head_dim // 2
        if half <= 0:
            raise ValueError(f"head_dim must be >= 2 for RoPE, got {head_dim}")
        self.half = half
        base_freq1 = torch.exp(
            -math.log(max_period) * torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
        )
        base_freq2 = torch.exp(
            -0.5 * math.log(max_period) * torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
        )
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
    Apply RoPE using precomputed per-head angles.

    x: [B, H, T, D_h]
    angles: [H, T, D_h/2]
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B,H,T,D_h], got {tuple(x.shape)}")
    D_h = x.shape[-1]
    rope_dim = int(D_h) if int(D_h) % 2 == 0 else int(D_h) - 1
    if rope_dim <= 0:
        return x

    sin = torch.sin(angles).to(dtype=x.dtype).unsqueeze(0)
    cos = torch.cos(angles).to(dtype=x.dtype).unsqueeze(0)

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
    attn_dropout = float(dropout_p) if training else 0.0
    return F.scaled_dot_product_attention(
        q_rot,
        k_rot,
        v,
        attn_mask=None,
        dropout_p=attn_dropout,
        is_causal=False,
    )


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


def _find_ffn_down_projection(module: nn.Module) -> nn.Linear | None:
    ffn_down = getattr(module, "ffn_down", None)
    if isinstance(ffn_down, nn.Linear):
        return ffn_down

    ffn = getattr(module, "ffn", None)
    if isinstance(ffn, MLP):
        return ffn.fc2

    if isinstance(ffn, nn.Sequential):
        for child in reversed(ffn):
            if isinstance(child, nn.Linear):
                return child
    return None


def _init_vae_module_weights(module: nn.Module, base_std: float = 0.02, init_style: str = "llm") -> None:
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
            for attr in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
                proj = getattr(module, attr, None)
                if proj is None:
                    continue
                if style == "xavier":
                    nn.init.xavier_uniform_(proj)
                else:
                    nn.init.normal_(proj, mean=0.0, std=float(base_std))
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


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
            is_mha = isinstance(module, nn.MultiheadAttention)
            if is_mha:
                out_proj = getattr(module, "out_proj", None)
                if isinstance(out_proj, nn.Linear):
                    nn.init.normal_(out_proj.weight, mean=0.0, std=std_out)
                    if out_proj.bias is not None:
                        nn.init.zeros_(out_proj.bias)

            for attr in ("out_proj", "cross_out_proj", "self_out_proj"):
                if is_mha and attr == "out_proj":
                    continue
                proj = getattr(module, attr, None)
                if isinstance(proj, nn.Linear):
                    nn.init.normal_(proj.weight, mean=0.0, std=std_out)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

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
    U = u_hat / (||u_hat||_2 + eps), log_s = bounded(s_hat), s = exp(log_s), W_hat = s * U.

    The bounds are applied in log-space before exp(). This avoids masked overflow
    in the forward graph when s_hat becomes very large.
    """
    if u_hat.ndim != 2:
        raise ValueError(f"u_hat must be [B, p], got {tuple(u_hat.shape)}")
    if s_hat.ndim == 1:
        log_s = s_hat.unsqueeze(-1)
    elif s_hat.ndim == 2 and s_hat.shape[1] == 1:
        log_s = s_hat
    else:
        raise ValueError(f"s_hat must be [B] or [B,1], got {tuple(s_hat.shape)}")
    if tuple(log_s.shape[:1]) != tuple(u_hat.shape[:1]):
        raise ValueError(f"Batch mismatch: u_hat={tuple(u_hat.shape)}, s_hat={tuple(log_s.shape)}")

    u_norm = u_hat.norm(dim=1, keepdim=True).clamp_min(float(eps))
    u = u_hat / u_norm

    if s_min != 0:
        log_s = float(s_min) + F.softplus(log_s - float(s_min))
    if s_max is not None and s_max != 0:
        log_s = float(s_max) - F.softplus(float(s_max) - log_s)
    s = torch.exp(log_s)

    return s * u


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

        if use_2d:
            q_angles = self.rope_2d_cross.compute_angles(latent_pos, latent_pos)
            k_angles = self.rope_2d_cross.compute_angles(token_pos, token_pos2)
            cross_out = _rope_attention_with_angles(
                q=q_cross,
                k=k_cross,
                v=v_cross,
                q_angles=q_angles,
                k_angles=k_angles,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
        else:
            cross_out = _rope_attention(
                q=q_cross,
                k=k_cross,
                v=v_cross,
                q_pos=latent_pos,
                k_pos=token_pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
        cross_out = cross_out.transpose(1, 2).contiguous().view(B, L_lat, self.d_latent)
        cross_out = self.cross_out_proj(cross_out)
        latents = latents + self.dropout(cross_out)

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


class TokenSummarizer(nn.Module):
    """Summarize a variable-length token set into a fixed number of output tokens."""

    def __init__(
        self,
        dim: int,
        out_tokens: int,
        mode: str = "mlp",
        hidden_mult: float = 2.0,
        dropout: float = 0.0,
        use_input_norm: bool = True,
    ) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if int(out_tokens) <= 0:
            raise ValueError(f"out_tokens must be positive, got {out_tokens}")

        mode_value = str(mode).strip().lower()
        if mode_value not in {"mlp", "latent_query"}:
            raise ValueError(f"mode must be 'mlp' or 'latent_query', got {mode!r}")

        self.dim = int(dim)
        self.out_tokens = int(out_tokens)
        self.mode = mode_value
        self.input_norm = nn.LayerNorm(self.dim) if bool(use_input_norm) else nn.Identity()

        if self.mode == "mlp":
            hidden_dim = max(1, int(float(hidden_mult) * self.dim))
            self.hidden = nn.Linear(self.dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, self.out_tokens)
            self.dropout = nn.Dropout(dropout)
            self.lat_queries = None
        else:
            self.hidden = None
            self.out_proj = None
            self.dropout = nn.Dropout(0.0)
            self.lat_queries = nn.Parameter(torch.empty(self.out_tokens, self.dim))
            nn.init.normal_(self.lat_queries, mean=0.0, std=0.02)

    def forward(self, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # V: [B, P, D]
        if V.ndim != 3:
            raise ValueError(f"V must be [B, P, D], got {tuple(V.shape)}")
        if int(V.shape[-1]) != self.dim:
            raise ValueError(f"V last dim ({int(V.shape[-1])}) must equal dim ({self.dim})")

        V_norm = self.input_norm(V)
        if self.mode == "mlp":
            assert self.hidden is not None and self.out_proj is not None
            hidden = self.dropout(F.gelu(self.hidden(V_norm)))
            logits = self.out_proj(hidden).permute(0, 2, 1)
        else:
            assert self.lat_queries is not None
            scale = float(self.dim) ** -0.5
            logits = torch.einsum("kd,bpd->bkp", self.lat_queries, V_norm) * scale

        A = torch.softmax(logits, dim=-1)
        Z = torch.matmul(A, V)
        return Z, A


class _TransformerProcessLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        hidden_dim = max(1, int(float(mlp_ratio) * int(dim)))
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B, R, D]
        attn_in = self.norm_attn(Z)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        Z = Z + self.attn_dropout(attn_out)
        Z = Z + self.mlp(self.norm_mlp(Z))
        return Z


class TransformerProcess(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerProcessLayer(
                    dim=int(dim),
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(max(0, int(depth)))
            ]
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B, R, D]
        for layer in self.layers:
            Z = layer(Z)
        return Z


class TTMMemoryBlock(nn.Module):
    """Token Turing Machine latent-memory block with summarization read/write."""

    def __init__(
        self,
        dim: int,
        mem_tokens: int,
        proc_tokens: int,
        process_depth: int,
        num_heads: int,
        summarizer_mode: str = "mlp",
        summarizer_hidden_mult: float = 2.0,
        dropout: float = 0.0,
        use_type_embeddings: bool = True,
        use_positional_embeddings: bool = True,
        memory_init: str = "learned",
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        memory_init_value = str(memory_init).strip().lower()
        if memory_init_value not in {"learned", "zeros"}:
            raise ValueError(f"memory_init must be 'learned' or 'zeros', got {memory_init!r}")

        self.dim = int(dim)
        self.mem_tokens = int(mem_tokens)
        self.proc_tokens = int(proc_tokens)
        self.use_type_embeddings = bool(use_type_embeddings)
        self.use_positional_embeddings = bool(use_positional_embeddings)
        self.memory_init = memory_init_value
        self.return_aux = bool(return_aux)
        self.max_input_tokens = 4096

        self.read_summarizer = TokenSummarizer(
            dim=self.dim,
            out_tokens=self.proc_tokens,
            mode=summarizer_mode,
            hidden_mult=float(summarizer_hidden_mult),
            dropout=float(dropout),
            use_input_norm=True,
        )
        self.process = TransformerProcess(
            dim=self.dim,
            depth=int(process_depth),
            num_heads=int(num_heads),
            mlp_ratio=4.0,
            dropout=float(dropout),
        )
        self.write_summarizer = TokenSummarizer(
            dim=self.dim,
            out_tokens=self.mem_tokens,
            mode=summarizer_mode,
            hidden_mult=float(summarizer_hidden_mult),
            dropout=float(dropout),
            use_input_norm=True,
        )

        if self.memory_init == "learned":
            self.memory_seed = nn.Parameter(torch.empty(1, self.mem_tokens, self.dim))
            nn.init.normal_(self.memory_seed, mean=0.0, std=0.02)
        else:
            self.register_parameter("memory_seed", None)

        if self.use_type_embeddings:
            self.type_mem = nn.Parameter(torch.empty(1, 1, self.dim))
            self.type_in = nn.Parameter(torch.empty(1, 1, self.dim))
            self.type_out = nn.Parameter(torch.empty(1, 1, self.dim))
            nn.init.normal_(self.type_mem, mean=0.0, std=0.02)
            nn.init.normal_(self.type_in, mean=0.0, std=0.02)
            nn.init.normal_(self.type_out, mean=0.0, std=0.02)
        else:
            self.register_parameter("type_mem", None)
            self.register_parameter("type_in", None)
            self.register_parameter("type_out", None)

        if self.use_positional_embeddings:
            self.mem_pos = nn.Parameter(torch.empty(1, self.mem_tokens, self.dim))
            self.input_pos_emb = nn.Parameter(torch.empty(1, self.max_input_tokens, self.dim))
            nn.init.normal_(self.mem_pos, mean=0.0, std=0.02)
            nn.init.normal_(self.input_pos_emb, mean=0.0, std=0.02)
        else:
            self.register_parameter("mem_pos", None)
            self.register_parameter("input_pos_emb", None)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if self.memory_init == "learned":
            assert self.memory_seed is not None
            memory = self.memory_seed.expand(int(batch_size), -1, -1)
            return memory.to(device=device if device is not None else memory.device, dtype=dtype or memory.dtype)

        if device is None:
            if self.mem_pos is not None:
                device = self.mem_pos.device
            else:
                device = torch.device("cpu")
        if dtype is None:
            if self.mem_pos is not None:
                dtype = self.mem_pos.dtype
            else:
                dtype = torch.float32
        return torch.zeros((int(batch_size), self.mem_tokens, self.dim), device=device, dtype=dtype)

    def _resolve_input_pos(self, X: torch.Tensor, input_pos: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_positional_embeddings:
            return None

        B, N, _ = X.shape
        if input_pos is not None:
            if input_pos.ndim != 3:
                raise ValueError(f"input_pos must be [1 or B, N, D], got {tuple(input_pos.shape)}")
            if int(input_pos.shape[-1]) != self.dim or int(input_pos.shape[1]) != N:
                raise ValueError(
                    f"input_pos must be [1 or B, {N}, {self.dim}], got {tuple(input_pos.shape)}"
                )
            if int(input_pos.shape[0]) == 1:
                return input_pos.expand(B, -1, -1)
            if int(input_pos.shape[0]) == B:
                return input_pos
            raise ValueError(f"input_pos batch must be 1 or {B}, got {int(input_pos.shape[0])}")

        if N > self.max_input_tokens:
            raise ValueError(
                f"input length ({N}) exceeds learned input_pos_emb capacity ({self.max_input_tokens})"
            )
        assert self.input_pos_emb is not None
        return self.input_pos_emb[:, :N, :].expand(B, -1, -1)

    def forward(
        self,
        X: torch.Tensor,
        M: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # X: [B, N, D], M: [B, K, D]
        if X.ndim != 3:
            raise ValueError(f"X must be [B, N, D], got {tuple(X.shape)}")
        if int(X.shape[-1]) != self.dim:
            raise ValueError(f"X last dim ({int(X.shape[-1])}) must equal dim ({self.dim})")

        B = int(X.shape[0])
        if M is None:
            M = self.init_memory(batch_size=B, device=X.device, dtype=X.dtype)
        if M.ndim != 3:
            raise ValueError(f"M must be [B, K, D], got {tuple(M.shape)}")
        if tuple(M.shape[1:]) != (self.mem_tokens, self.dim):
            raise ValueError(
                f"M must be [B, {self.mem_tokens}, {self.dim}], got {tuple(M.shape)}"
            )
        if int(M.shape[0]) != B:
            raise ValueError(f"M batch ({int(M.shape[0])}) must match X batch ({B})")

        resolved_input_pos = self._resolve_input_pos(X, input_pos)

        X_read = X
        M_read = M
        if self.use_positional_embeddings:
            assert self.mem_pos is not None
            M_read = M_read + self.mem_pos[:, : self.mem_tokens, :]
            if resolved_input_pos is not None:
                X_read = X_read + resolved_input_pos
        if self.use_type_embeddings:
            assert self.type_mem is not None and self.type_in is not None
            M_read = M_read + self.type_mem
            X_read = X_read + self.type_in

        V_read = torch.cat([M_read, X_read], dim=1)
        Z, A_read = self.read_summarizer(V_read)
        O = self.process(Z)

        X_write = X
        M_write = M
        O_write = O
        if self.use_positional_embeddings:
            assert self.mem_pos is not None
            M_write = M_write + self.mem_pos[:, : self.mem_tokens, :]
            if resolved_input_pos is not None:
                X_write = X_write + resolved_input_pos
        if self.use_type_embeddings:
            assert self.type_mem is not None and self.type_in is not None and self.type_out is not None
            M_write = M_write + self.type_mem
            X_write = X_write + self.type_in
            O_write = O_write + self.type_out

        V_write = torch.cat([M_write, O_write, X_write], dim=1)
        M_new, A_write = self.write_summarizer(V_write)

        if not self.return_aux:
            return M_new
        aux = {
            "read_tokens": Z,
            "process_tokens": O,
            "read_weights": A_read,
            "write_weights": A_write,
        }
        return M_new, aux


class TTMMemoryStack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        block: TTMMemoryBlock,
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        if int(num_blocks) <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self.num_blocks = int(num_blocks)
        self.share_weights = bool(share_weights)
        if self.share_weights:
            self.shared_block = block
            self.blocks = None
        else:
            self.shared_block = None
            self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(self.num_blocks)])

    def _get_block(self, index: int) -> TTMMemoryBlock:
        if self.share_weights:
            assert self.shared_block is not None
            return self.shared_block
        assert self.blocks is not None
        return self.blocks[index]

    def forward(
        self,
        X: torch.Tensor,
        M: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[dict[str, torch.Tensor]]]]:
        # X: [B, N, D], M: [B, K, D]
        block0 = self._get_block(0)
        if M is None:
            M = block0.init_memory(batch_size=int(X.shape[0]), device=X.device, dtype=X.dtype)

        aux_list: list[dict[str, torch.Tensor]] = []
        for block_idx in range(self.num_blocks):
            block = self._get_block(block_idx)
            block_out = block(X, M=M, input_pos=input_pos)
            if isinstance(block_out, tuple):
                M, aux = block_out
                aux_list.append(aux)
            else:
                M = block_out

        if not aux_list:
            return M
        return M, {"blocks": aux_list}


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

        if use_2d:
            effective_kv_pos2 = kv_pos2 if kv_pos2 is not None else kv_pos
            q_angles = self.rope_2d_cross.compute_angles(q_pos, q_pos2)
            k_angles = self.rope_2d_cross.compute_angles(kv_pos, effective_kv_pos2)
            attn_out = _rope_attention_with_angles(
                q=q_proj,
                k=k_proj,
                v=v_proj,
                q_angles=q_angles,
                k_angles=k_angles,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
        else:
            attn_out = _rope_attention(
                q=q_proj,
                k=k_proj,
                v=v_proj,
                q_pos=q_pos,
                k_pos=kv_pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        attn_out = self.out_proj(attn_out)

        q = q + self.attn_out_dropout(attn_out)
        q_self = self.self_attn_norm(q)
        self_q = self.self_q_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_k = self.self_k_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_v = self.self_v_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)

        if use_2d:
            self_angles = self.rope_2d_self.compute_angles(q_pos, q_pos2)
            self_out = _rope_attention_with_angles(
                q=self_q,
                k=self_k,
                v=self_v,
                q_angles=self_angles,
                k_angles=self_angles,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
        else:
            self_out = _rope_attention(
                q=self_q,
                k=self_k,
                v=self_v,
                q_pos=q_pos,
                k_pos=q_pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
        self_out = self_out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        self_out = self.self_out_proj(self_out)
        q = q + self.self_attn_out_dropout(self_out)
        q = q + self.ffn(self.ffn_norm(q))
        return q


__all__ = [
    "CrossAttnBlock",
    "MLP",
    "PerceiverResamplerBlock",
    "RoPeMixed2D",
    "TTMMemoryBlock",
    "TTMMemoryStack",
    "TokenSummarizer",
    "TransformerProcess",
    "sinusoidal_embedding",
]
