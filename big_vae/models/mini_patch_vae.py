from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from big_vae.models.vae_shared import (
    CrossAttnBlock,
    PerceiverResamplerBlock,
    _apply_residual_scaled_init,
    _decode_direction_and_logscale,
    _init_vae_latent_parameters,
    _init_vae_module_weights,
    sinusoidal_embedding,
)


@dataclass(slots=True)
class MiniVAEConfig:
    z_dim: int = 64
    d_e: int = 128
    encoder_latent_dim: int = 0
    pos_lat_dim: int = 0
    pos_dim: int = 32
    num_attn_layers_encoder: int = 2
    num_layers_decoder: int = 2
    decoder_bilinear_rank: int = 0
    decoder_L_latents: int = 8
    decoder_use_dist_conditioning: bool = True
    decoder_dist_mode: str = "add"
    use_latent_sampling: bool = True
    implementation: str = "real"
    mlp_stub_hidden_dim: int = 256
    stub_mlp_use_batchnorm: bool = False
    stub_resampler_d_model: int = 0
    init_style: str = "llm"
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

        token_tau = (torch.arange(p, device=w_patch.device, dtype=torch.float32) + 0.5) / float(p)
        pos = sinusoidal_embedding(token_tau, dim=self.cfg.pos_dim).to(dtype=w_patch.dtype)
        pos_expand = pos.unsqueeze(0).expand(B, -1, -1)

        tokens = torch.cat([w_patch.unsqueeze(-1), dist_var_tokens, pos_expand], dim=-1)
        elem = self.elem_embed(tokens)

        slot_pos = sinusoidal_embedding(self.slot_tau.to(device=w_patch.device), dim=self.pos_lat_dim).to(dtype=elem.dtype)
        latents0 = self.slot_pos_to_latent(slot_pos)
        latents = latents0.unsqueeze(0).expand(B, -1, -1)
        tokens_with_w = torch.cat([elem, w_patch.unsqueeze(-1).to(dtype=elem.dtype)], dim=-1)
        latent_pos = self.slot_tau.to(device=w_patch.device)
        for block in self.resampler:
            latents = block(
                latents=latents,
                tokens=tokens_with_w,
                latent_pos=latent_pos,
                token_pos=token_tau,
            )
        h = latents.reshape(B, self.flat_latent_dim)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def encode_patch(self, w_patch: torch.Tensor, dist_var_tokens: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        return self.patch_proj(mu)


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
            nn.Linear(self.d_model, 1),
        )
        self.scale_head = nn.Sequential(nn.Linear(self.d_model, 1))
        self.z_shortcut = nn.Sequential(
            nn.Linear(int(cfg.z_dim) + self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    @staticmethod
    def _channel_norm_over_sequence(x: torch.Tensor, eps: float) -> torch.Tensor:
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

        u_hat_attn = self.direction_head(q).squeeze(-1)

        z_exp = z.unsqueeze(1).expand(B, p, -1)
        pos_exp = q_pos_emb.unsqueeze(0).expand(B, -1, -1)
        shortcut_in = torch.cat([z_exp, pos_exp], dim=-1)
        u_hat_shortcut = self.z_shortcut(shortcut_in).squeeze(-1)

        u_hat = u_hat_attn + u_hat_shortcut
        s_hat = self.scale_head(q.mean(dim=1)).squeeze(-1)
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
    - encode_patch(w_patch, dist_var_tokens) -> patch_token
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
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
            raise ValueError(f"mini stub_resampler_d_model must be >= 0, got {int(cfg.stub_resampler_d_model)}")

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
                f"z_dim={int(cfg.z_dim)} vs "
                f"stub_resampler_d_model*decoder_L_latents={self.resampler_d_model}*{self.num_latents}={self.flat_latent_dim}"
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

        token_tau = (torch.arange(p, device=w_patch.device, dtype=torch.float32) + 0.5) / float(p)
        pos = sinusoidal_embedding(torch.arange(p, device=w_patch.device), dim=self.cfg.pos_dim).to(dtype=w_patch.dtype)
        pos_expand = pos.unsqueeze(0).expand(B, -1, -1)

        token_in = torch.cat([w_patch.unsqueeze(-1), pos_expand], dim=-1)
        elem = self.elem_embed(token_in)

        latents = self.resampler_latents.unsqueeze(0).expand(B, -1, -1)
        tokens_with_w = torch.cat([elem, w_patch.unsqueeze(-1).to(dtype=elem.dtype)], dim=-1)
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
        return stats.chunk(2, dim=-1)

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
            z_aligned = torch.cat([z, z.new_zeros((B, self.latent_dim - z_dim))], dim=1)

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
            u_hat_p = torch.cat([u_hat, u_hat.new_zeros((B, p - self.out_dim))], dim=1)
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
    - encoder: Perceiver Resampler over patch tokens
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


__all__ = [
    "CrossAttnPatchDecoder",
    "MLPNoCompressionPatchDecoder",
    "MLPNoCompressionPatchEncoder",
    "MiniPatchDecoder",
    "MiniPatchEncoder",
    "MiniPatchVAE",
    "MiniPatchVAEStub",
    "MiniVAEConfig",
    "TransformerNoCompressionPatchEncoder",
]
