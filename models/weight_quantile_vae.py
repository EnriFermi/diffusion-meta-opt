from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(indices: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Deterministic Fourier/sinusoidal embedding for integer-like indices."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    x = indices.to(dtype=torch.float32)
    half = dim // 2
    if half == 0:
        return x.unsqueeze(-1)

    scale = torch.arange(half, device=x.device, dtype=torch.float32)
    scale = scale / max(half - 1, 1)
    freqs = torch.exp(-math.log(max_period) * scale)
    angles = x.unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class InputDistributionEncoder(nn.Module):
    """
    Encodes input distributions from Q_feature_major [d_in, k] once per layer.
    Steps: shared MLP (k -> k_mlp) per feature, one self-attention layer over d_in, then patch mean.
    Output: dist_patches [n_patches, k_mlp].
    """

    def __init__(self, k: int, k_mlp: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.k = k
        self.k_mlp = k_mlp
        hidden = max(1, k_mlp)

        self.feature_mlp = nn.Sequential(
            nn.Linear(k, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k_mlp),
            nn.Dropout(dropout),
        )
        self.norm_attn = nn.LayerNorm(k_mlp)
        self.attn = nn.MultiheadAttention(k_mlp, num_heads=1, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_feature_major: torch.Tensor, P: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # Q_feature_major: [d_in, k]
        if Q_feature_major.ndim != 2:
            raise ValueError(f"Expected Q_feature_major rank-2 [d_in, k], got {tuple(Q_feature_major.shape)}")
        d_in, k = Q_feature_major.shape
        if k != self.k:
            raise ValueError(f"Expected k={self.k}, got k={k}")
        if P <= 0:
            raise ValueError(f"P must be > 0, got {P}")

        x = self.feature_mlp(Q_feature_major)  # [d_in, k_mlp]
        h = self.norm_attn(x).unsqueeze(0)  # [1, d_in, k_mlp]
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out.squeeze(0))  # [d_in, k_mlp]

        n_patches = (d_in + P - 1) // P
        d_in_pad = n_patches * P
        x_pad = torch.zeros(d_in_pad, self.k_mlp, device=x.device, dtype=x.dtype)
        x_pad[:d_in, :] = x
        dist_patches = x_pad.view(n_patches, P, self.k_mlp).mean(dim=1)  # [n_patches, k_mlp]
        dist_flat = dist_patches.reshape(n_patches * self.k_mlp)  # [n_patches_per_output * K_mlp]
        return dist_patches, dist_flat, n_patches, d_in_pad


@dataclass(slots=True)
class EncoderConfig:
    n_row_layers: int = 2
    self_attn_mode: str = "full"  # {"full", "cls_only"}


@dataclass(slots=True)
class ResamplerConfig:
    n_layers: int = 2


@dataclass(slots=True)
class ModelConfig:
    k: int
    k_mlp: int
    patch_size: int  # P
    d_tok: int
    m_lat: int
    d_lat: int
    n_heads: int = 4
    pos_fourier_dim: int = 32
    row_mlp_mult: float = 4.0
    resampler_mlp_mult: float = 4.0
    decoder_mlp_mult: float = 2.0
    dropout: float = 0.0
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    resampler: ResamplerConfig = field(default_factory=ResamplerConfig)


class RowMixerBlock(nn.Module):
    """Row mixer on tokens_rows [d_out, 1+n_patches, d_tok] with per-output independence."""

    def __init__(
        self,
        d_tok: int,
        n_heads: int,
        mlp_mult: float,
        dropout: float,
        self_attn_mode: str,
    ) -> None:
        super().__init__()
        if self_attn_mode not in {"full", "cls_only"}:
            raise ValueError(f"encoder.self_attn_mode must be 'full' or 'cls_only', got {self_attn_mode}")
        if d_tok % n_heads != 0:
            raise ValueError(f"d_tok ({d_tok}) must be divisible by n_heads ({n_heads})")

        self.self_attn_mode = self_attn_mode
        hidden = max(1, int(d_tok * mlp_mult))

        self.attn = nn.MultiheadAttention(d_tok, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.norm_full_attn = nn.LayerNorm(d_tok)
        self.norm_full_ffn = nn.LayerNorm(d_tok)
        self.full_ffn = MLP(d_tok, hidden, d_tok, dropout=dropout)

        self.norm_cls_kv = nn.LayerNorm(d_tok)
        self.norm_cls_ffn = nn.LayerNorm(d_tok)
        self.cls_ffn = MLP(d_tok, hidden, d_tok, dropout=dropout)

        self.norm_patch_ffn = nn.LayerNorm(d_tok)
        self.patch_ffn = MLP(d_tok, hidden, d_tok, dropout=dropout)

    def forward(self, tokens_rows: torch.Tensor) -> torch.Tensor:
        # tokens_rows: [d_out, L, d_tok], L = 1 + n_patches
        if self.self_attn_mode == "full":
            h = self.norm_full_attn(tokens_rows)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            tokens_rows = tokens_rows + self.dropout(attn_out)
            tokens_rows = tokens_rows + self.dropout(self.full_ffn(self.norm_full_ffn(tokens_rows)))
            return tokens_rows

        kv = self.norm_cls_kv(tokens_rows)
        cls_q = kv[:, 0:1, :]
        cls_attn, _ = self.attn(cls_q, kv, kv, need_weights=False)

        cls = tokens_rows[:, 0:1, :] + self.dropout(cls_attn)
        cls = cls + self.dropout(self.cls_ffn(self.norm_cls_ffn(cls)))

        patches = tokens_rows[:, 1:, :]
        patches = patches + self.dropout(self.patch_ffn(self.norm_patch_ffn(patches)))

        return torch.cat([cls, patches], dim=1)


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, d_lat: int, n_heads: int, mlp_mult: float, dropout: float) -> None:
        super().__init__()
        if d_lat % n_heads != 0:
            raise ValueError(f"d_lat ({d_lat}) must be divisible by n_heads ({n_heads})")

        hidden = max(1, int(d_lat * mlp_mult))
        self.dropout = nn.Dropout(dropout)

        self.norm_cross_q = nn.LayerNorm(d_lat)
        self.norm_cross_kv = nn.LayerNorm(d_lat)
        self.cross_attn = nn.MultiheadAttention(d_lat, n_heads, dropout=dropout, batch_first=True)

        self.norm_self = nn.LayerNorm(d_lat)
        self.self_attn = nn.MultiheadAttention(d_lat, n_heads, dropout=dropout, batch_first=True)

        self.norm_ffn = nn.LayerNorm(d_lat)
        self.ffn = MLP(d_lat, hidden, d_lat, dropout=dropout)

    def forward(self, latents: torch.Tensor, cls_context: torch.Tensor) -> torch.Tensor:
        # latents: [1, m_lat, d_lat]
        # cls_context: [1, d_out, d_lat] (keys/values from CLS_all only)
        q = self.norm_cross_q(latents)
        kv = self.norm_cross_kv(cls_context)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latents = latents + self.dropout(cross_out)

        h = self.norm_self(latents)
        self_out, _ = self.self_attn(h, h, h, need_weights=False)
        latents = latents + self.dropout(self_out)

        latents = latents + self.dropout(self.ffn(self.norm_ffn(latents)))
        return latents


class DecoderCrossBlock(nn.Module):
    """Cross-attention query->latents with no query self-attention."""

    def __init__(self, d_lat: int, n_heads: int, mlp_mult: float, dropout: float) -> None:
        super().__init__()
        if d_lat % n_heads != 0:
            raise ValueError(f"d_lat ({d_lat}) must be divisible by n_heads ({n_heads})")

        hidden = max(1, int(d_lat * mlp_mult))
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(d_lat)
        self.norm_kv = nn.LayerNorm(d_lat)
        self.attn = nn.MultiheadAttention(d_lat, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_lat)
        self.ffn = MLP(d_lat, hidden, d_lat, dropout=dropout)

    def forward(self, q_flat: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # q_flat: [d_out*n_patches, 1, d_lat]
        # latents: [m_lat, d_lat]
        b = q_flat.shape[0]
        kv = self.norm_kv(latents).unsqueeze(0).expand(b, -1, -1)
        q = self.norm_q(q_flat)
        cross_out, _ = self.attn(q, kv, kv, need_weights=False)
        q_flat = q_flat + self.dropout(cross_out)
        q_flat = q_flat + self.dropout(self.ffn(self.norm_ffn(q_flat)))
        return q_flat


class WeightQuantileVAE(nn.Module):
    """
    Fast encoder-decoder for W [d_in, d_out], conditioned on quantiles of X [n, d_in].
    Returns: (W_hat [d_in, d_out], kl_loss scalar, aux_dict).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.encoder.self_attn_mode not in {"full", "cls_only"}:
            raise ValueError(
                f"encoder.self_attn_mode must be in {{'full', 'cls_only'}}, got {cfg.encoder.self_attn_mode}"
            )
        if cfg.d_tok % cfg.n_heads != 0:
            raise ValueError(f"d_tok ({cfg.d_tok}) must be divisible by n_heads ({cfg.n_heads})")
        if cfg.d_lat % cfg.n_heads != 0:
            raise ValueError(f"d_lat ({cfg.d_lat}) must be divisible by n_heads ({cfg.n_heads})")

        P = cfg.patch_size
        self.input_distribution_encoder = InputDistributionEncoder(
            k=cfg.k,
            k_mlp=cfg.k_mlp,
            dropout=cfg.dropout,
        )

        # Shared MLPHead over all (j, p): [k_mlp + P] -> d_tok -> d_tok
        self.patch_mlp_head = nn.Sequential(
            nn.Linear(cfg.k_mlp + P, cfg.d_tok),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_tok, cfg.d_tok),
            nn.Dropout(cfg.dropout),
        )

        # CLS token: learned base + deterministic output-index embedding projected to d_tok.
        self.cls_base = nn.Parameter(torch.zeros(cfg.d_tok))
        self.output_index_proj = nn.Linear(cfg.pos_fourier_dim, cfg.d_tok)

        self.row_mixers = nn.ModuleList(
            [
                RowMixerBlock(
                    d_tok=cfg.d_tok,
                    n_heads=cfg.n_heads,
                    mlp_mult=cfg.row_mlp_mult,
                    dropout=cfg.dropout,
                    self_attn_mode=cfg.encoder.self_attn_mode,
                )
                for _ in range(cfg.encoder.n_row_layers)
            ]
        )

        # Perceiver resampler over CLS_all only.
        self.cls_to_lat = nn.Linear(cfg.d_tok, cfg.d_lat)
        self.latent_base = nn.Parameter(torch.randn(cfg.m_lat, cfg.d_lat) * 0.02)
        self.resampler_layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(
                    d_lat=cfg.d_lat,
                    n_heads=cfg.n_heads,
                    mlp_mult=cfg.resampler_mlp_mult,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.resampler.n_layers)
            ]
        )

        # VAE heads on Z [m_lat, d_lat]
        self.to_mu = nn.Linear(cfg.d_lat, cfg.d_lat)
        self.to_logvar = nn.Linear(cfg.d_lat, cfg.d_lat)

        # Decoder query position (patch index + output index): deterministic embedding + learned projection.
        self.decoder_query_base = nn.Parameter(torch.zeros(cfg.d_tok))
        self.decoder_pos_proj = nn.Linear(2 * cfg.pos_fourier_dim, cfg.d_tok)
        self.decoder_q_to_lat = nn.Linear(cfg.d_tok, cfg.d_lat)
        self.decoder_cross = DecoderCrossBlock(
            d_lat=cfg.d_lat,
            n_heads=cfg.n_heads,
            mlp_mult=cfg.decoder_mlp_mult,
            dropout=cfg.dropout,
        )
        dec_hidden = max(1, int(cfg.d_lat * cfg.decoder_mlp_mult))
        self.patch_decoder = nn.Sequential(
            nn.Linear(cfg.d_lat, dec_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dec_hidden, cfg.patch_size),
        )

    def _compute_quantiles(self, X: torch.Tensor) -> torch.Tensor:
        # X: [n, d_in] -> Q: [k, d_in], with tau_i = (i+0.5)/k
        k = self.cfg.k
        q_dtype = torch.float32 if X.dtype in {torch.float16, torch.bfloat16} else X.dtype
        X_q = X.to(dtype=q_dtype)
        tau = (torch.arange(k, device=X.device, dtype=q_dtype) + 0.5) / float(k)
        Q = torch.quantile(X_q, q=tau, dim=0)
        return Q

    def _patchify_weights(self, W: torch.Tensor, d_in: int, d_out: int, n_patches: int) -> torch.Tensor:
        # W: [d_in, d_out] -> W_patches: [d_out, n_patches, P]
        P = self.cfg.patch_size
        d_in_pad = n_patches * P
        W_pad = torch.zeros(d_in_pad, d_out, device=W.device, dtype=W.dtype)
        W_pad[:d_in, :] = W
        W_patches = W_pad.transpose(0, 1).contiguous().view(d_out, n_patches, P)
        return W_patches

    def _build_cls_tokens(self, d_out: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # CLS_j = cls_base + proj(sinusoidal(output_index=j))
        out_idx = torch.arange(d_out, device=device, dtype=torch.float32)
        out_pos = sinusoidal_embedding(out_idx, self.cfg.pos_fourier_dim).to(dtype=dtype)
        cls_pos = self.output_index_proj(out_pos)  # [d_out, d_tok]
        cls = self.cls_base.to(dtype=dtype).unsqueeze(0) + cls_pos
        return cls

    def _build_decoder_queries(
        self, d_out: int, n_patches: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # q_{j,p} = decoder_query_base + proj(pos_embed(p_index, j_index))
        j_idx = torch.arange(d_out, device=device)
        p_idx = torch.arange(n_patches, device=device)
        j_grid, p_grid = torch.meshgrid(j_idx, p_idx, indexing="ij")

        p_emb = sinusoidal_embedding(p_grid.to(torch.float32), self.cfg.pos_fourier_dim)
        j_emb = sinusoidal_embedding(j_grid.to(torch.float32), self.cfg.pos_fourier_dim)
        pos_pair = torch.cat([p_emb, j_emb], dim=-1).to(dtype=dtype)

        q_pos = self.decoder_pos_proj(pos_pair)  # [d_out, n_patches, d_tok]
        q_dec = self.decoder_query_base.to(dtype=dtype).view(1, 1, -1) + q_pos
        return q_dec

    def forward(self, X: torch.Tensor, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        # X: [n, d_in], W: [d_in, d_out]
        if X.ndim != 2 or W.ndim != 2:
            raise ValueError(f"Expected X and W to be rank-2, got X.ndim={X.ndim}, W.ndim={W.ndim}")

        d_in_x = X.shape[1]
        d_in, d_out = W.shape
        if d_in_x != d_in:
            raise ValueError(f"X has d_in={d_in_x}, but W has d_in={d_in}")

        device = self.cls_base.device
        dtype = self.cls_base.dtype
        X = X.to(device=device, dtype=dtype)
        W = W.to(device=device, dtype=dtype)

        P = self.cfg.patch_size
        # Part A: quantiles computed once and input-distribution encoding computed once for all d_out.
        Q = self._compute_quantiles(X).to(dtype=dtype)  # [k, d_in]
        Q_feature_major = Q.transpose(0, 1).contiguous()  # [d_in, k]
        dist_patches, dist_flat, n_patches, d_in_pad = self.input_distribution_encoder(
            Q_feature_major=Q_feature_major,
            P=P,
        )  # [n_patches, k_mlp]

        # Part B: patchify W and build patch tokens with shared MLPHead.
        W_patches = self._patchify_weights(W=W, d_in=d_in, d_out=d_out, n_patches=n_patches)  # [d_out, n_patches, P]
        dist_shared = dist_patches.unsqueeze(0).expand(d_out, -1, -1)  # [d_out, n_patches, k_mlp]
        v = torch.cat([dist_shared, W_patches], dim=-1)  # [d_out, n_patches, k_mlp+P]
        patch_tokens = self.patch_mlp_head(v)  # [d_out, n_patches, d_tok]

        # Part C: row encoder with CLS_j and row-local mixing.
        cls_tokens = self._build_cls_tokens(d_out=d_out, dtype=dtype, device=device)  # [d_out, d_tok]
        tokens_rows = torch.cat([cls_tokens.unsqueeze(1), patch_tokens], dim=1)  # [d_out, 1+n_patches, d_tok]
        for block in self.row_mixers:
            tokens_rows = block(tokens_rows)
        CLS_all = tokens_rows[:, 0, :]  # [d_out, d_tok]

        # Part D: Perceiver resampler over CLS_all only.
        cls_context = self.cls_to_lat(CLS_all).unsqueeze(0)  # [1, d_out, d_lat]
        latents = self.latent_base.unsqueeze(0).to(dtype=dtype)  # [1, m_lat, d_lat]
        for layer in self.resampler_layers:
            latents = layer(latents, cls_context)
        Z = latents.squeeze(0)  # [m_lat, d_lat]

        mu = self.to_mu(Z)  # [m_lat, d_lat]
        logvar = self.to_logvar(Z)  # [m_lat, d_lat]
        eps = torch.randn_like(mu)
        Z_sample = mu + eps * torch.exp(0.5 * logvar)  # reparameterization
        kl_per_token = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
        kl_loss = kl_per_token.mean()

        # Part E: decoder queries (p_index, j_index) -> cross-attn to latents (no query self-attn).
        Q_dec = self._build_decoder_queries(d_out=d_out, n_patches=n_patches, dtype=dtype, device=device)
        q_lat = self.decoder_q_to_lat(Q_dec)  # [d_out, n_patches, d_lat]
        q_flat = q_lat.contiguous().view(d_out * n_patches, 1, self.cfg.d_lat)
        q_flat = self.decoder_cross(q_flat, Z_sample)

        dec_tokens = q_flat.view(d_out, n_patches, self.cfg.d_lat)
        W_patches_hat = self.patch_decoder(dec_tokens)  # [d_out, n_patches, P]

        # Invert patchify: [d_out, n_patches, P] -> [d_in, d_out]
        W_hat_pad = W_patches_hat.contiguous().view(d_out, d_in_pad).transpose(0, 1)
        W_hat = W_hat_pad[:d_in, :]

        aux_dict: dict[str, object] = {
            "self_attn_mode": self.cfg.encoder.self_attn_mode,
            "Q_shape": tuple(Q.shape),
            "Q_feature_major_shape": tuple(Q_feature_major.shape),
            "dist_patches_shape": tuple(dist_patches.shape),
            "dist_flat_shape": tuple(dist_flat.shape),
            "W_patches_shape": tuple(W_patches.shape),
            "tokens_rows_shape": tuple(tokens_rows.shape),
            "CLS_all_shape": tuple(CLS_all.shape),
            "Z_shape": tuple(Z.shape),
            "Q_dec_shape": tuple(Q_dec.shape),
            "W_patches_hat_shape": tuple(W_patches_hat.shape),
            "n_patches": n_patches,
            "d_in_pad": d_in_pad,
        }
        return W_hat, kl_loss, aux_dict


def _minimal_shape_test() -> None:
    torch.manual_seed(0)

    n = 32
    d_in = 65
    d_out = 7
    k = 8
    k_mlp = 24
    P = 16
    d_tok = 64
    m_lat = 32
    d_lat = 64

    X = torch.randn(n, d_in)
    W = torch.randn(d_in, d_out)

    out_shapes: list[tuple[int, int]] = []
    for mode in ("full", "cls_only"):
        cfg = ModelConfig(
            k=k,
            k_mlp=k_mlp,
            patch_size=P,
            d_tok=d_tok,
            m_lat=m_lat,
            d_lat=d_lat,
            n_heads=8,
            encoder=EncoderConfig(n_row_layers=2, self_attn_mode=mode),
            resampler=ResamplerConfig(n_layers=2),
            dropout=0.0,
        )
        model = WeightQuantileVAE(cfg)
        W_hat, kl_loss, aux = model(X, W)

        assert tuple(W_hat.shape) == (d_in, d_out), f"{mode}: bad W_hat shape {tuple(W_hat.shape)}"
        assert kl_loss.ndim == 0, f"{mode}: kl_loss must be scalar, got ndim={kl_loss.ndim}"
        assert aux["Q_shape"] == (k, d_in), f"{mode}: bad Q shape {aux['Q_shape']}"
        assert aux["dist_patches_shape"] == ((d_in + P - 1) // P, k_mlp), (
            f"{mode}: bad dist_patches shape {aux['dist_patches_shape']}"
        )
        out_shapes.append(tuple(W_hat.shape))

    assert out_shapes[0] == out_shapes[1] == (d_in, d_out)


if __name__ == "__main__":
    _minimal_shape_test()
