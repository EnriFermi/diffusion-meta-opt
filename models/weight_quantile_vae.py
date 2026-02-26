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


def _init_vae_module_weights(module: nn.Module) -> None:
    """Explicit initialization for mini-VAE modules (real and stub variants)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def _init_vae_latent_parameters(module: nn.Module) -> None:
    """Initialize standalone latent parameters that are not covered by module.apply()."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("resampler_latents"):
                nn.init.normal_(param, mean=0.0, std=0.02)


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

    # Smooth lower barrier: keeps s >= s_min while preserving gradients below the threshold.
    s = float(s_min) + F.softplus(s - float(s_min))
    # Optional smooth upper barrier to avoid exp overflow and skipped non-finite steps.
    if s_max is not None:
        s = float(s_max) - F.softplus(float(s_max) - s)
    return u * torch.exp(s)


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
        q_span = (q_last - q_first).clamp_min(1e-6)
        q = 2.0 * (q - q_first) / q_span - 1.0  # [B, k_s, p]
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

        # DCNv2 patch embedding: [B, d_dist]
        dist_patch_embed = self.dcn(v_pool)

        return dist_var_tokens, dist_patch_embed


@dataclass(slots=True)
class MiniVAEConfig:
    z_dim: int = 64
    d_e: int = 128
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
    # Stub-only: if > 0, Perceiver resampler width in TransformerNoCompressionPatchEncoder.
    # If 0, defaults to d_e.
    stub_resampler_d_model: int = 0
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

        # Per-element embed: (1 + d_var + pos_dim) -> d_e.
        self.elem_embed = nn.Sequential(
            nn.Linear(1 + d_var + cfg.pos_dim, cfg.d_e),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_e, cfg.d_e),
            nn.Dropout(cfg.dropout),
        )

        self.num_latents = max(1, int(cfg.decoder_L_latents))
        self.resampler_latents = nn.Parameter(torch.randn(self.num_latents, cfg.d_e) * 0.02)
        self.resampler = nn.ModuleList(
            [
                PerceiverResamplerBlock(d_model=cfg.d_e, n_heads=cfg.n_heads, dropout=cfg.dropout)
                for _ in range(max(1, cfg.num_attn_layers_encoder))
            ]
        )
        self.head_norm = nn.LayerNorm(cfg.d_e)

        self.to_mu = nn.Linear(cfg.d_e, cfg.z_dim)
        self.to_logvar = nn.Linear(cfg.d_e, cfg.z_dim)

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

        # Deterministic per-position embedding: [p, pos_dim] -> [B, p, pos_dim]
        pos = sinusoidal_embedding(torch.arange(p, device=w_patch.device), dim=self.cfg.pos_dim).to(dtype=w_patch.dtype)
        pos_expand = pos.unsqueeze(0).expand(B, -1, -1)

        # T: [B, p, 1 + d_var + pos_dim]
        t = torch.cat([w_patch.unsqueeze(-1), dist_var_tokens, pos_expand], dim=-1)

        # E: [B, p, d_e]
        e = self.elem_embed(t)

        # Perceiver resampling: latent queries cross-attend to patch tokens.
        latents = self.resampler_latents.unsqueeze(0).expand(B, -1, -1)
        for block in self.resampler:
            latents = block(latents=latents, tokens=e)
        h = self.head_norm(latents.mean(dim=1))

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
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_attn = self.q_norm(q)
        kv_attn = self.kv_norm(kv)
        attn_out, _ = self.cross_attn(q_attn, kv_attn, kv_attn, need_weights=False)
        q = q + self.attn_dropout(attn_out)
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
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0

        self.direction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, z: torch.Tensor, patch_size: int, dist_patch_embed: torch.Tensor | None = None) -> torch.Tensor:
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

        q_pos = sinusoidal_embedding(torch.arange(p, device=z.device), dim=self.d_model).to(dtype=z.dtype)
        q = q_pos.unsqueeze(0).expand(B, -1, -1)

        for block in self.blocks:
            q = block(q, lat)

        u_hat = self.direction_head(q).squeeze(-1)  # [B, p]
        s_hat = self.scale_head(q.mean(dim=1)).squeeze(-1)  # [B]
        return _decode_direction_and_logscale(
            u_hat=u_hat,
            s_hat=s_hat,
            eps=self.output_eps,
            s_min=self.output_s_min,
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
        self.apply(_init_vae_module_weights)
        _init_vae_latent_parameters(self)

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

    def decode(self, z: torch.Tensor, patch_size: int, dist_patch_embed: torch.Tensor | None = None) -> torch.Tensor:
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
        w_hat = self.decode(z=z, patch_size=w_patch.shape[1], dist_patch_embed=dist_patch_embed)
        return w_hat, mu, logvar, z


class PerceiverResamplerBlock(nn.Module):
    """Perceiver-style resampler block over latent queries and input patch tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_cross_q = nn.LayerNorm(d_model)
        self.norm_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_cross_q(latents)
        kv = self.norm_cross_kv(tokens)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latents = latents + self.dropout(cross_out)

        lat_norm = self.norm_self(latents)
        self_out, _ = self.self_attn(lat_norm, lat_norm, lat_norm, need_weights=False)
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
        if self.resampler_d_model == int(cfg.d_e):
            self.token_to_resampler = nn.Identity()
        else:
            self.token_to_resampler = nn.Linear(cfg.d_e, self.resampler_d_model)

        self.num_latents = max(1, int(cfg.decoder_L_latents))
        self.resampler_latents = nn.Parameter(torch.randn(self.num_latents, self.resampler_d_model) * 0.02)
        self.resampler = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    d_model=self.resampler_d_model,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                )
                for _ in range(max(1, cfg.num_attn_layers_encoder))
            ]
        )
        self.latent_norm = nn.LayerNorm(self.resampler_d_model)
        self.to_mu = nn.Linear(self.resampler_d_model, cfg.z_dim)
        self.to_logvar = nn.Linear(self.resampler_d_model, cfg.z_dim)
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
        pos = sinusoidal_embedding(torch.arange(p, device=w_patch.device), dim=self.cfg.pos_dim).to(dtype=w_patch.dtype)
        pos_expand = pos.unsqueeze(0).expand(B, -1, -1)

        token_in = torch.cat([w_patch.unsqueeze(-1), pos_expand], dim=-1)
        e = self.elem_embed(token_in)

        token_ctx = self.token_to_resampler(e)
        latents = self.resampler_latents.unsqueeze(0).expand(B, -1, -1)
        for block in self.resampler:
            latents = block(latents=latents, tokens=token_ctx)
        h = self.latent_norm(latents.mean(dim=1))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def encode_patch(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        return self.patch_proj(mu)


class MLPNoCompressionPatchDecoder(nn.Module):
    """Debug decoder: MLP from one latent vector to full patch."""

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
        self.patch_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.direction_head = nn.Linear(hidden, self.out_dim)
        self.scale_head = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor, patch_size: int, dist_patch_embed: torch.Tensor | None = None) -> torch.Tensor:
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

        h = self.patch_trunk(dec_in)
        u_hat = self.direction_head(h)
        s_hat = self.scale_head(h).squeeze(-1)

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
    - decoder: MLP from latent to full patch
    """

    def __init__(self, d_var: int, cfg: MiniVAEConfig, d_dist: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        decoder_d_dist = int(d_dist) if d_dist is not None else int(d_var)
        self.encoder = TransformerNoCompressionPatchEncoder(d_var=d_var, cfg=cfg)
        self.decoder = MLPNoCompressionPatchDecoder(d_dist=decoder_d_dist, cfg=cfg)
        self.apply(_init_vae_module_weights)
        _init_vae_latent_parameters(self)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu.new_zeros(())

    def encode(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)

    def decode(self, z: torch.Tensor, patch_size: int, dist_patch_embed: torch.Tensor | None = None) -> torch.Tensor:
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
        w_hat = self.decode(z=z, patch_size=w_patch.shape[1], dist_patch_embed=dist_patch_embed)
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
    mini_encoder_ckpt_path: str = ""


class LocalOutputSelfAttentionBlock(nn.Module):
    """Local attention within one output-column token group."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float, self_attn_mode: str) -> None:
        super().__init__()
        if self_attn_mode not in {"full", "cls_only"}:
            raise ValueError(f"encoder.self_attn_mode must be 'full' or 'cls_only', got {self_attn_mode}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.self_attn_mode = self_attn_mode
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.full_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cls_to_all_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.patch_to_cls_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        hidden = max(1, int(d_model * ffn_mult))
        self.ffn = MLP(d_model, hidden, d_model, dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B_group, 1 + T, d_model]
        h = self.norm_attn(tokens)

        if self.self_attn_mode == "full":
            attn_out, _ = self.full_attn(h, h, h, need_weights=False)
            tokens = tokens + self.dropout(attn_out)
        else:
            # CLS query attends to all tokens.
            # cls_q: [B_group, 1, d_model]
            cls_q = h[:, 0:1, :]
            cls_out, _ = self.cls_to_all_attn(cls_q, h, h, need_weights=False)

            # Patch queries attend only to CLS.
            # patch_q: [B_group, T, d_model]
            patch_q = h[:, 1:, :]
            patch_out, _ = self.patch_to_cls_attn(patch_q, cls_q, cls_q, need_weights=False)

            cls_res = tokens[:, 0:1, :] + self.dropout(cls_out)
            patch_res = tokens[:, 1:, :] + self.dropout(patch_out)
            tokens = torch.cat([cls_res, patch_res], dim=1)

        tokens = tokens + self.dropout(self.ffn(self.norm_ffn(tokens)))
        return tokens


class LatentEncoderLayer(nn.Module):
    """
    One big-encoder block:
    1) local per-output attention over [CLS_o, patches_o]
    2) latents cross-attend to token K/V (CLS only or all tokens)
    3) latent self-attention
    4) latent FFN
    """

    def __init__(self, d_model: int, d_lat: int, n_heads: int, ffn_mult: float, dropout: float, self_attn_mode: str) -> None:
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

        self.norm_cross_q = nn.LayerNorm(d_lat)
        self.norm_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_lat,
            num_heads=n_heads,
            kdim=d_model,
            vdim=d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_lat_self = nn.LayerNorm(d_lat)
        self.lat_self_attn = nn.MultiheadAttention(d_lat, n_heads, dropout=dropout, batch_first=True)

        self.norm_lat_ffn = nn.LayerNorm(d_lat)
        hidden = max(1, int(d_lat * ffn_mult))
        self.lat_ffn = MLP(d_lat, hidden, d_lat, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens_by_output: torch.Tensor,
        latents: torch.Tensor,
        cross_attend_only_cls: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens_by_output: [B, d_out, 1 + T, d_model]
        # latents: [B, L, d_lat]
        B, d_out, L_local, d_model = tokens_by_output.shape

        # Local self-attention per output group.
        local_in = tokens_by_output.view(B * d_out, L_local, d_model)  # [B*d_out, 1+T, d_model]
        local_out = self.local_block(local_in)  # [B*d_out, 1+T, d_model]
        tokens = local_out.view(B, d_out, L_local, d_model)  # [B, d_out, 1+T, d_model]

        # K/V source for latent cross-attention.
        if cross_attend_only_cls:
            kv = tokens[:, :, 0, :]  # [B, d_out, d_model]
        else:
            kv = tokens.reshape(B, d_out * L_local, d_model)  # [B, d_out*(1+T), d_model]

        q = self.norm_cross_q(latents)  # [B, L, d_lat]
        kv_norm = self.norm_cross_kv(kv)  # [B, S, d_model]
        cross_out, _ = self.cross_attn(q, kv_norm, kv_norm, need_weights=False)
        latents = latents + self.dropout(cross_out)

        lat_norm = self.norm_lat_self(latents)
        lat_self_out, _ = self.lat_self_attn(lat_norm, lat_norm, lat_norm, need_weights=False)
        latents = latents + self.dropout(lat_self_out)

        latents = latents + self.dropout(self.lat_ffn(self.norm_lat_ffn(latents)))
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
    - mu: [B, L, d_lat] (or [L, d_lat] for unbatched input)
    - logvar: [B, L, d_lat] (or [L, d_lat] for unbatched input)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        p = cfg.patch_size
        d_var = cfg.distribution.d_var
        d_dist = cfg.distribution.d_dist

        self.distribution_encoder = InputDistributionEncodingModule(cfg.distribution)
        self.mini_patch_encoder = MiniPatchEncoder(d_var=d_var, cfg=cfg.mini_vae)

        d_patch = cfg.mini_vae.d_patch
        d_model = cfg.big_vae.d_model
        d_lat = cfg.big_vae.d_lat

        if d_model % cfg.big_vae.n_heads != 0:
            raise ValueError(f"big_vae.d_model ({d_model}) must be divisible by big_vae.n_heads ({cfg.big_vae.n_heads})")
        if d_lat % cfg.big_vae.n_heads != 0:
            raise ValueError(f"big_vae.d_lat ({d_lat}) must be divisible by big_vae.n_heads ({cfg.big_vae.n_heads})")

        # Patch token projection after concat([mini_patch_token, dist_patch_embed]).
        self.patch_token_proj = nn.Linear(d_patch + d_dist, d_model)

        # One learned CLS token expanded per output column.
        self.cls_token = nn.Parameter(torch.zeros(d_model))

        self.encoder_layers = nn.ModuleList(
            [
                LatentEncoderLayer(
                    d_model=d_model,
                    d_lat=d_lat,
                    n_heads=cfg.big_vae.n_heads,
                    ffn_mult=cfg.big_vae.ffn_mult,
                    dropout=cfg.big_vae.dropout,
                    self_attn_mode=cfg.big_vae.encoder.self_attn_mode,
                )
                for _ in range(max(1, cfg.big_vae.num_encoder_layers))
            ]
        )

        # Global latent base: [L, d_lat]
        self.latent_base = nn.Parameter(torch.randn(cfg.big_vae.num_latents, d_lat) * 0.02)

        self.to_mu = nn.Linear(d_lat, d_lat)
        self.to_logvar = nn.Linear(d_lat, d_lat)

        # Decoder query construction with deterministic (o, t) positional encoding.
        self.pos_proj = nn.Linear(2 * cfg.big_vae.pos_fourier_dim, d_model)
        self.query_proj = nn.Linear(d_model + d_dist, d_model)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderCrossBlock(
                    d_model=d_model,
                    d_lat=d_lat,
                    n_heads=cfg.big_vae.n_heads,
                    ffn_mult=cfg.big_vae.ffn_mult,
                    dropout=cfg.big_vae.dropout,
                )
                for _ in range(max(1, cfg.big_vae.num_decoder_layers))
            ]
        )

        self.patch_decode_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.big_vae.dropout),
            nn.Linear(d_model, p),
        )

        if cfg.mini_encoder_ckpt_path:
            self.load_pretrained_mini_encoder(cfg.mini_encoder_ckpt_path)

    def load_pretrained_mini_encoder(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Mini-VAE checkpoint not found: {checkpoint_path}")

        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]

        if not isinstance(state, dict):
            raise ValueError("Unsupported checkpoint format: expected state dict or {'model_state': state_dict}")

        # Accept either plain encoder state_dict or full MiniPatchVAE with 'encoder.' prefix.
        if any(k.startswith("encoder.") for k in state.keys()):
            enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
        else:
            enc_state = state

        missing, unexpected = self.mini_patch_encoder.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"[BigWeightVAE] mini encoder missing keys: {missing}")
        if unexpected:
            print(f"[BigWeightVAE] mini encoder unexpected keys: {unexpected}")

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mu/logvar: [B, L, d_lat] -> z: [B, L, d_lat]
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mu/logvar: [B, L, d_lat] or [L, d_lat] -> scalar
        if mu.ndim == 2:
            kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)  # [L]
            return kl.mean()
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)  # [B, L]
        return kl.mean()

    @staticmethod
    def operator_recon_loss(X: torch.Tensor, W: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        # Linear operator loss: MSE(X @ W, X @ W_hat)
        # X: [B, n, d_in] or [n, d_in]
        # W/W_hat: [B, d_in, d_out] or [d_in, d_out]
        if X.ndim == 2:
            y = X @ W  # [n, d_out]
            y_hat = X @ W_hat  # [n, d_out]
            return F.mse_loss(y_hat, y)

        y = torch.matmul(X, W)  # [B, n, d_out]
        y_hat = torch.matmul(X, W_hat)  # [B, n, d_out]
        return F.mse_loss(y_hat, y)

    @staticmethod
    def _build_patch_indices(d_in: int, patch_size: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
        # patch_idx_t: [T, p], T=ceil(d_in/p), values clamped into [0, d_in-1]
        T = (d_in + patch_size - 1) // patch_size
        d_in_pad = T * patch_size
        patch_idx_t = torch.arange(d_in_pad, device=device, dtype=torch.long).view(T, patch_size)
        patch_idx_t = patch_idx_t.clamp(max=max(0, d_in - 1))
        return patch_idx_t, T, d_in_pad

    def forward(self, W: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W: [B, d_in, d_out] or [d_in, d_out]
        # X: [B, n, d_in] or [n, d_in]
        if W.ndim not in {2, 3}:
            raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
        if X.ndim not in {2, 3}:
            raise ValueError(f"X must be rank-2 or rank-3, got {tuple(X.shape)}")

        squeeze_batch = W.ndim == 2
        if squeeze_batch != (X.ndim == 2):
            raise ValueError(f"W and X must be both batched or both unbatched, got W={tuple(W.shape)}, X={tuple(X.shape)}")

        if squeeze_batch:
            W = W.unsqueeze(0)  # [1, d_in, d_out]
            X = X.unsqueeze(0)  # [1, n, d_in]

        B, d_in, d_out = W.shape
        Bx, n, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        device = W.device
        dtype = W.dtype

        p = self.cfg.patch_size
        patch_idx_t, T, d_in_pad = self._build_patch_indices(d_in=d_in, patch_size=p, device=device)

        # ---------------------------------------------------------------------
        # 1) Distribution reuse per input patch t (independent of output o).
        # ---------------------------------------------------------------------
        # patch_idx_bt: [B, T, p]
        patch_idx_bt = patch_idx_t.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Flatten (B, T) for one batched pass through distribution encoder.
        # X_rep: [B*T, n, d_in], patch_idx_flat: [B*T, p]
        X_rep = X.unsqueeze(1).expand(B, T, n, d_in).reshape(B * T, n, d_in)
        patch_idx_flat = patch_idx_bt.reshape(B * T, p)

        # dist_var_flat: [B*T, p, d_var]
        # dist_patch_flat: [B*T, d_dist]
        dist_var_flat, dist_patch_flat = self.distribution_encoder(X_rep, patch_idx_flat)

        # dist_var_by_patch: [B, T, p, d_var]
        # dist_patch_by_patch: [B, T, d_dist]
        d_var = self.cfg.distribution.d_var
        d_dist = self.cfg.distribution.d_dist
        dist_var_by_patch = dist_var_flat.view(B, T, p, d_var)
        dist_patch_by_patch = dist_patch_flat.view(B, T, d_dist)

        # ---------------------------------------------------------------------
        # 2) Patchify W over d_in for each output column.
        # ---------------------------------------------------------------------
        # W_pad: [B, d_in_pad, d_out]
        W_pad = torch.zeros(B, d_in_pad, d_out, device=device, dtype=dtype)
        W_pad[:, :d_in, :] = W

        # w_patches: [B, d_out, T, p]
        w_patches = W_pad.transpose(1, 2).contiguous().view(B, d_out, T, p)

        # ---------------------------------------------------------------------
        # 3) Patch embedding using pretrained mini encoder + dist patch embed.
        # ---------------------------------------------------------------------
        # dist_var_expanded: [B, d_out, T, p, d_var]
        dist_var_expanded = dist_var_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1, -1)

        # Flatten per-(o,t) for shared mini encoder pass.
        # w_flat: [B*d_out*T, p]
        # dist_var_token_flat: [B*d_out*T, p, d_var]
        w_flat = w_patches.reshape(B * d_out * T, p)
        dist_var_token_flat = dist_var_expanded.reshape(B * d_out * T, p, d_var)

        # patch_token_raw_flat: [B*d_out*T, d_patch]
        patch_token_raw_flat = self.mini_patch_encoder.encode_patch(w_patch=w_flat, dist_var_tokens=dist_var_token_flat)
        d_patch = self.cfg.mini_vae.d_patch

        # patch_token_raw: [B, d_out, T, d_patch]
        patch_token_raw = patch_token_raw_flat.view(B, d_out, T, d_patch)

        # dist_patch_expanded: [B, d_out, T, d_dist]
        dist_patch_expanded = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1)

        # patch_token_cat: [B, d_out, T, d_patch + d_dist]
        patch_token_cat = torch.cat([patch_token_raw, dist_patch_expanded], dim=-1)

        # patch_tokens: [B, d_out, T, d_model]
        patch_tokens = self.patch_token_proj(patch_token_cat)

        # CLS per output column.
        # cls_tokens: [B, d_out, 1, d_model]
        cls_tokens = self.cls_token.view(1, 1, 1, -1).expand(B, d_out, 1, -1)

        # tokens_by_output: [B, d_out, 1 + T, d_model]
        tokens_by_output = torch.cat([cls_tokens, patch_tokens], dim=2)

        # ---------------------------------------------------------------------
        # 4) Global latent encoder with local group attention + latent bottleneck.
        # ---------------------------------------------------------------------
        # latents: [B, L, d_lat]
        latents = self.latent_base.unsqueeze(0).expand(B, -1, -1)

        for enc_layer in self.encoder_layers:
            tokens_by_output, latents = enc_layer(
                tokens_by_output=tokens_by_output,
                latents=latents,
                cross_attend_only_cls=self.cfg.big_vae.encoder.cross_attend_only_cls,
            )

        # Posterior params on latents.
        # mu/logvar: [B, L, d_lat]
        mu = self.to_mu(latents)
        logvar = self.to_logvar(latents)
        z_latents = self.reparameterize(mu=mu, logvar=logvar)

        # ---------------------------------------------------------------------
        # 5) Decoder with per-(o,t) queries, no query self-attention.
        # ---------------------------------------------------------------------
        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")  # [d_out, T]

        pos_dim = self.cfg.big_vae.pos_fourier_dim
        pos_o = sinusoidal_embedding(o_grid.to(torch.float32), pos_dim)  # [d_out, T, pos_dim]
        pos_t = sinusoidal_embedding(t_grid.to(torch.float32), pos_dim)  # [d_out, T, pos_dim]
        pos_ot = torch.cat([pos_o, pos_t], dim=-1)  # [d_out, T, 2*pos_dim]

        # q_base: [B, d_out, T, d_model]
        q_base = self.pos_proj(pos_ot).unsqueeze(0).expand(B, -1, -1, -1)

        # q_cond_cat: [B, d_out, T, d_model + d_dist]
        q_cond_cat = torch.cat([q_base, dist_patch_expanded], dim=-1)

        # q_tokens: [B, d_out*T, d_model]
        q_tokens = self.query_proj(q_cond_cat).reshape(B, d_out * T, self.cfg.big_vae.d_model)

        for dec_layer in self.decoder_layers:
            q_tokens = dec_layer(q_tokens=q_tokens, z_latents=z_latents)

        # w_hat_patch_flat: [B, d_out*T, p]
        w_hat_patch_flat = self.patch_decode_head(q_tokens)

        # w_hat_patches: [B, d_out, T, p]
        w_hat_patches = w_hat_patch_flat.view(B, d_out, T, p)

        # Stitch back to matrix shape.
        # W_hat_pad: [B, d_in_pad, d_out]
        W_hat_pad = w_hat_patches.view(B, d_out, d_in_pad).transpose(1, 2)

        # W_hat: [B, d_in, d_out]
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
        distribution=DistributionConfig(k_s=16, Kq=32, d_var=128, d_dist=128),
        mini_vae=MiniVAEConfig(z_dim=64, d_e=128, num_attn_layers_encoder=2, num_layers_decoder=2, n_heads=4, d_patch=64),
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

    print("W shape:", tuple(W.shape))
    print("X shape:", tuple(X.shape))
    print("W_hat shape:", tuple(W_hat.shape))
    print("mu shape:", tuple(mu.shape))
    print("logvar shape:", tuple(logvar.shape))


if __name__ == "__main__":
    smoke_test_big_weight_vae()
