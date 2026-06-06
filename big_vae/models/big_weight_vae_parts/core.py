from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from big_vae.models.distribution_encoder import DistributionConfig, InputDistributionEncodingModule
from big_vae.models.mini_patch_vae import MiniVAEConfig
from big_vae.models.patch_tokenizers import MixerPatchTokenizer, PatchConditionedMLPTokenizer, ResidualPatchTokenizer
from big_vae.models.vae_shared import (
    CrossAttnBlock,
    MLP,
    PerceiverResamplerBlock,
    _apply_sequence_mask,
    _decode_direction_and_logscale,
    _key_padding_to_attn_bias,
    _rope_attention,
    sinusoidal_embedding,
)

from big_vae.models.big_weight_vae_parts.config import *
from big_vae.models.big_weight_vae_parts.blocks import *
from big_vae.models.big_weight_vae_parts.latent_mixin import BigWeightVAELatentMixin
from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from big_vae.models.big_weight_vae_parts.encoding_mixin import BigWeightVAEEncodingMixin
from big_vae.models.big_weight_vae_parts.decoding_mixin import BigWeightVAEDecodingMixin
from big_vae.models.big_weight_vae_parts.forward_mixin import BigWeightVAEForwardMixin

class BigWeightVAE(
    BigWeightVAELatentMixin,
    BigWeightVAELossMixin,
    BigWeightVAEEncodingMixin,
    BigWeightVAEDecodingMixin,
    BigWeightVAEForwardMixin,
    nn.Module,
):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        p = cfg.patch_size
        d_var = cfg.distribution.d_var
        d_dist = cfg.distribution.d_dist
        d_model = cfg.big_vae.d_model
        d_lat = cfg.big_vae.d_lat
        n_heads = cfg.big_vae.n_heads
        num_latents = cfg.big_vae.num_latents
        num_enc_layers = max(1, cfg.big_vae.num_encoder_layers)
        num_dec_layers = max(1, cfg.big_vae.num_decoder_layers)
        dropout = cfg.big_vae.dropout

        self.use_distribution_encoder = not bool(cfg.big_vae.disable_distribution_encoder)
        patch_tokenizer_cfg = cfg.big_vae.patch_tokenizer
        patch_tokenizer_kind_raw = str(patch_tokenizer_cfg.kind or cfg.big_vae.patch_tokenizer_kind)
        self.patch_tokenizer_kind = self._normalize_patch_tokenizer_kind(patch_tokenizer_kind_raw)
        self.distribution_encoder_conditioning_kind = self._normalize_distribution_encoder_conditioning_kind(
            cfg.big_vae.distribution_encoder_conditioning_kind
        )
        self.use_legacy_distribution_encoder_conditioning = bool(
            self.use_distribution_encoder and self.distribution_encoder_conditioning_kind == "legacy"
        )
        self.use_token_adapter_distribution_encoder_conditioning = bool(
            self.use_distribution_encoder and self.distribution_encoder_conditioning_kind == "token_adapter"
        )
        if self.use_distribution_encoder:
            self.distribution_encoder: InputDistributionEncodingModule | None = InputDistributionEncodingModule(
                cfg.distribution
            )
            patch_tokenizer_d_dist = d_dist
            query_in_dim = d_model + d_dist
        else:
            self.distribution_encoder = None
            patch_tokenizer_d_dist = 0
            query_in_dim = d_model
        d_patch = int(patch_tokenizer_cfg.d_patch) if int(patch_tokenizer_cfg.d_patch) > 0 else int(cfg.mini_vae.d_patch)
        self.patch_token_d_patch = d_patch
        tokenizer_dropout = (
            float(dropout) if patch_tokenizer_cfg.dropout is None else float(patch_tokenizer_cfg.dropout)
        )
        if self.patch_tokenizer_kind == "conditioned_mlp":
            if not self.use_distribution_encoder:
                raise ValueError(
                    "big_vae.patch_tokenizer_kind='conditioned_mlp' requires disable_distribution_encoder=false"
                )
            self.patch_tokenizer = PatchConditionedMLPTokenizer(
                p=p,
                d_var=d_var,
                d_patch=d_patch,
                hidden_dim=patch_tokenizer_cfg.hidden_dim,
                cond_proj_dim=patch_tokenizer_cfg.cond_proj_dim,
                num_blocks=int(patch_tokenizer_cfg.num_blocks),
                dropout=tokenizer_dropout,
            )
        else:
            # self.patch_tokenizer = MixerPatchTokenizer(
            #     p=p,
            #     d_var=d_var,
            #     d_dist=d_dist,
            #     d_patch=d_patch,
            #     d_hidden=d_var,
            #     num_mixer_layers=3,
            #     dropout=cfg.big_vae.dropout,
            # )
            self.patch_tokenizer = ResidualPatchTokenizer(
                p=p,
                d_dist=patch_tokenizer_d_dist,
                d_patch=d_patch,
                hidden_dim=int(patch_tokenizer_cfg.residual_hidden_dim),
                num_layers=int(patch_tokenizer_cfg.residual_num_layers),
                dropout=float(patch_tokenizer_cfg.residual_dropout),
            )

        if d_model % n_heads != 0:
            raise ValueError(f"big_vae.d_model ({d_model}) must be divisible by big_vae.n_heads ({n_heads})")
        if d_lat % n_heads != 0:
            raise ValueError(f"big_vae.d_lat ({d_lat}) must be divisible by big_vae.n_heads ({n_heads})")

        self.patch_token_proj = nn.Linear(d_patch, d_model)
        self.cls_token = nn.Parameter(torch.zeros(d_model))

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
                    rope_2d_coord_kind=getattr(cfg.big_vae, "rope_2d_coord_kind", "normalized_center"),
                )
                for _ in range(num_enc_layers)
            ]
        )
        if self.use_legacy_distribution_encoder_conditioning:
            self.enc_dist_inject_projs: nn.ModuleList | None = nn.ModuleList(
                [nn.Linear(d_model + d_var, d_model) for _ in range(num_enc_layers)]
            )
        else:
            self.enc_dist_inject_projs = None
        self.flat_lat_dim = num_latents * d_lat
        if self.use_legacy_distribution_encoder_conditioning:
            self.enc_dist_to_latent_heads: nn.ModuleList | None = nn.ModuleList(
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
        else:
            self.enc_dist_to_latent_heads = None
        if self.use_token_adapter_distribution_encoder_conditioning:
            self.encoder_conditioning_adapters: nn.ModuleList | None = nn.ModuleList(
                [
                    TokenConditioningAdapter(
                        d_model=d_model,
                        d_y=d_dist,
                        dropout=dropout,
                    )
                    for _ in range(num_enc_layers)
                ]
            )
        else:
            self.encoder_conditioning_adapters = None

        self.latent_base = nn.Parameter(torch.randn(num_latents, d_lat) * 0.02)
        self.z_dim = self.flat_lat_dim
        self.latent_norm = nn.LayerNorm(self.flat_lat_dim)
        self.latent_prior_kind = self._normalize_latent_prior_kind(cfg.big_vae.latent_prior_kind)
        self.decoder_query_conditioning_kind = self._normalize_decoder_query_conditioning_kind(
            cfg.big_vae.decoder_query_conditioning_kind
        )
        self.rope_2d_coord_kind = _normalize_rope_2d_coord_kind(
            getattr(cfg.big_vae, "rope_2d_coord_kind", "normalized_center")
        )
        if self.latent_prior_kind == "vamp" and bool(cfg.big_vae.use_latent_sampling):
            vamp_prior_K = max(1, int(cfg.big_vae.vamp_prior_K))
            self.vamp_prior_base: nn.Parameter | None = nn.Parameter(torch.randn(vamp_prior_K, num_latents, d_lat) * 0.02)
        else:
            self.vamp_prior_base = None
        self.register_buffer("latent_sampling_gate", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        if bool(cfg.big_vae.use_latent_sampling) or bool(cfg.big_vae.use_encoder_mu_head):
            self.to_mu: nn.Linear | None = nn.Linear(d_lat, d_lat)
            self.to_logvar: nn.Linear | None = (
                nn.Linear(d_lat, d_lat) if bool(cfg.big_vae.use_latent_sampling) else None
            )
            self._init_latent_sampling_heads()
        else:
            self.to_mu = None
            self.to_logvar = None

        self.dec_L_latents = num_latents
        self.latent_to_decoder = nn.Linear(d_lat, d_model)
        self.pos_proj = nn.Linear(2 * cfg.big_vae.pos_fourier_dim, d_model)
        if self.decoder_query_conditioning_kind == "mlp":
            query_hidden = max(
                int(d_model),
                int(round(float(cfg.big_vae.decoder_query_conditioning_hidden_mult) * float(d_model))),
            )
            self.query_proj = nn.Sequential(
                nn.Linear(query_in_dim, query_hidden),
                nn.GELU(),
                nn.Linear(query_hidden, d_model),
            )
        else:
            self.query_proj = nn.Linear(query_in_dim, d_model)
        self.query_pos_proj = nn.Linear(d_model, d_model)
        self.decoder_layers = nn.ModuleList(
            [CrossAttnBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, use_rope_2d=True) for _ in range(num_dec_layers)]
        )

        self.q_tokens_norm = nn.LayerNorm(d_model)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, p),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.debug_query_hint_proj = nn.Linear(d_model, d_model)
        self.debug_encoder_direct_direction_head = nn.Linear(d_model, p)
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0
        self.direction_seq_norm_eps = 1e-6

        self.z_shortcut_proj = nn.Linear(self.flat_lat_dim, d_model)
        self.z_shortcut = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, p),
        )
        if bool(cfg.big_vae.disable_z_shortcut):
            self.z_shortcut_proj.requires_grad_(False)
            self.z_shortcut.requires_grad_(False)

__all__ = [
    'BigWeightVAE',
]
