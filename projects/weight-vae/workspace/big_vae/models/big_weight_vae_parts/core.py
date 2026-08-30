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
    CleanContentReadoutV8,
    CrossAttnBlock,
    DecoderSelfAttentionBlock,
    FixedPositionOnlyFiLM,
    FourTrunkComplementEncoderV11,
    HybridNonlinearContentReadoutV9,
    MandatoryLatentBridge,
    MLP,
    NonAffineRMSNorm,
    OrthogonalComplementEncoderV10,
    PerceiverResamplerBlock,
    _apply_sequence_mask,
    _apply_residual_scaled_init,
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
        architecture_version = str(getattr(cfg.big_vae, "architecture_version", "legacy_v1")).strip().lower()
        if architecture_version not in {
            "legacy_v1",
            "latent_feedback_postnorm_v2",
            "latent_feedback_prenorm_v3",
            "latent_feedback_perirms_qknorm_v4",
            "latent_mandatory_bridge_prenorm_v5",
            "latent_mandatory_bridge_posfilm_v6",
            "latent_mandatory_cross_refresh_posfilm_v7",
            "clean_content_readout_posfilm_v8",
            "hybrid_nonlinear_content_posfilm_v9",
            "carrier_mean_content_posfilm_v9a",
            "orthogonal_complement_tied_posfilm_v10",
            "four_trunk_complement_v11",
        }:
            raise ValueError(
                "big_vae.architecture_version must be 'legacy_v1', "
                "'latent_feedback_postnorm_v2', 'latent_feedback_prenorm_v3', or "
                "'latent_feedback_perirms_qknorm_v4', or "
                "'latent_mandatory_bridge_prenorm_v5', or "
                "'latent_mandatory_bridge_posfilm_v6', "
                "'latent_mandatory_cross_refresh_posfilm_v7', "
                "'clean_content_readout_posfilm_v8', "
                "'hybrid_nonlinear_content_posfilm_v9', "
                "'carrier_mean_content_posfilm_v9a', "
                "'orthogonal_complement_tied_posfilm_v10', "
                "'four_trunk_complement_v11', "
                f"got {architecture_version!r}"
            )
        self.architecture_version = architecture_version
        self.use_latent_feedback_postnorm_v2 = architecture_version in {
            "latent_feedback_postnorm_v2",
            "latent_feedback_prenorm_v3",
            "latent_feedback_perirms_qknorm_v4",
            "latent_mandatory_bridge_prenorm_v5",
            "latent_mandatory_bridge_posfilm_v6",
            "latent_mandatory_cross_refresh_posfilm_v7",
            "clean_content_readout_posfilm_v8",
            "hybrid_nonlinear_content_posfilm_v9",
            "carrier_mean_content_posfilm_v9a",
            "orthogonal_complement_tied_posfilm_v10",
            "four_trunk_complement_v11",
        }
        self.use_outer_postnorm_v2 = architecture_version == "latent_feedback_postnorm_v2"
        self.use_prenorm_v3 = architecture_version == "latent_feedback_prenorm_v3"
        self.use_mandatory_latent_bridge_v5 = architecture_version in {
            "latent_mandatory_bridge_prenorm_v5",
            "latent_mandatory_bridge_posfilm_v6",
            "latent_mandatory_cross_refresh_posfilm_v7",
            "clean_content_readout_posfilm_v8",
            "hybrid_nonlinear_content_posfilm_v9",
            "carrier_mean_content_posfilm_v9a",
            "orthogonal_complement_tied_posfilm_v10",
            "four_trunk_complement_v11",
        }
        self.use_position_only_film_v6 = architecture_version in {
            "latent_mandatory_bridge_posfilm_v6",
            "latent_mandatory_cross_refresh_posfilm_v7",
            "clean_content_readout_posfilm_v8",
            "hybrid_nonlinear_content_posfilm_v9",
            "carrier_mean_content_posfilm_v9a",
            "orthogonal_complement_tied_posfilm_v10",
            "four_trunk_complement_v11",
        }
        self.use_mandatory_cross_refresh_v7 = architecture_version == "latent_mandatory_cross_refresh_posfilm_v7"
        self.use_clean_content_readout_v8 = architecture_version == "clean_content_readout_posfilm_v8"
        self.use_hybrid_nonlinear_content_v9 = architecture_version == "hybrid_nonlinear_content_posfilm_v9"
        self.use_carrier_mean_content_v9a = architecture_version == "carrier_mean_content_posfilm_v9a"
        self.use_orthogonal_complement_v10 = (
            architecture_version == "orthogonal_complement_tied_posfilm_v10"
        )
        self.use_four_trunk_complement_v11 = architecture_version == "four_trunk_complement_v11"
        self.use_clean_content_carrier = (
            self.use_clean_content_readout_v8
            or self.use_hybrid_nonlinear_content_v9
            or self.use_carrier_mean_content_v9a
        )
        self.use_direct_content_encoder = (
            self.use_clean_content_carrier
            or self.use_orthogonal_complement_v10
            or self.use_four_trunk_complement_v11
        )
        if self.use_direct_content_encoder and bool(cfg.big_vae.normalize_latent_slots_before_mu):
            raise ValueError(
                "direct content V8/V9/V10/V11 requires normalize_latent_slots_before_mu=false "
                "to preserve its raw serialized latent coordinates"
            )
        if self.use_orthogonal_complement_v10:
            frozen_contract = (
                p,
                d_lat,
                num_latents,
                int(cfg.big_vae.v10_protected_dim),
                int(cfg.big_vae.v10_adaptive_dim),
                int(cfg.big_vae.v10_num_experts),
            )
            if frozen_contract != (16, 384, 32, 320, 64, 40):
                raise ValueError(
                    "V10 frozen contract is p16/d_lat384/L32/protected320/adaptive64/experts40; "
                    f"got {frozen_contract}"
                )
            if float(dropout) != 0.0:
                raise ValueError("V10 decoder must use dropout=0 for structural zero preservation")
        if self.use_four_trunk_complement_v11:
            frozen_contract = (
                p,
                d_lat,
                num_latents,
                int(cfg.big_vae.v10_protected_dim),
                int(cfg.big_vae.v10_adaptive_dim),
                int(cfg.big_vae.v11_num_trunks),
                int(cfg.big_vae.v11_blocks_per_trunk),
                int(cfg.big_vae.v11_trunk_dim),
            )
            if frozen_contract != (16, 384, 32, 320, 64, 4, 10, 16):
                raise ValueError(
                    "V11 frozen contract is p16/d_lat384/L32/protected320/adaptive64/"
                    "trunks4/depth10/trunk_dim16; "
                    f"got {frozen_contract}"
                )
            if float(dropout) != 0.0:
                raise ValueError("V11 decoder must use dropout=0 for structural zero preservation")
        self.mandatory_cross_refresh_rms = 1.0
        # V5/V6 intentionally keep the V4 encoder/tokenizer unchanged; only
        # the decoder conditioning interface differs across these experiments.
        self.use_peri_rms_qk_norm_v4 = architecture_version in {
            "latent_feedback_perirms_qknorm_v4",
            "latent_mandatory_bridge_prenorm_v5",
            "latent_mandatory_bridge_posfilm_v6",
            "latent_mandatory_cross_refresh_posfilm_v7",
            "clean_content_readout_posfilm_v8",
            "hybrid_nonlinear_content_posfilm_v9",
            "carrier_mean_content_posfilm_v9a",
            "orthogonal_complement_tied_posfilm_v10",
            "four_trunk_complement_v11",
        }
        self.use_single_gate_conditioning = self.use_prenorm_v3 or self.use_peri_rms_qk_norm_v4

        self.use_distribution_encoder = not bool(cfg.big_vae.disable_distribution_encoder)
        patch_tokenizer_cfg = cfg.big_vae.patch_tokenizer
        patch_tokenizer_kind_raw = str(patch_tokenizer_cfg.kind or cfg.big_vae.patch_tokenizer_kind)
        self.patch_tokenizer_kind = self._normalize_patch_tokenizer_kind(patch_tokenizer_kind_raw)
        self.distribution_encoder_conditioning_kind = self._normalize_distribution_encoder_conditioning_kind(
            cfg.big_vae.distribution_encoder_conditioning_kind
        )
        self.weight_input_normalization_kind = str(
            cfg.big_vae.weight_input_normalization_kind
        ).strip().lower()
        if self.weight_input_normalization_kind not in {"none", "per_output_maxabs_q7"}:
            raise ValueError(
                "big_vae.weight_input_normalization_kind must be 'none' or "
                f"'per_output_maxabs_q7', got {self.weight_input_normalization_kind!r}"
            )
        self.weight_input_scale_qmax = float(cfg.big_vae.weight_input_scale_qmax)
        self.weight_input_log2_scale_mean = float(
            cfg.big_vae.weight_input_log2_scale_mean
        )
        self.weight_input_log2_scale_std = float(
            cfg.big_vae.weight_input_log2_scale_std
        )
        if self.weight_input_normalization_kind != "none":
            if not self.use_peri_rms_qk_norm_v4:
                raise ValueError("normalized weight input is currently isolated to the V4-family graph")
            if self.weight_input_scale_qmax <= 0.0:
                raise ValueError("weight_input_scale_qmax must be positive")
            if self.weight_input_log2_scale_std <= 0.0:
                raise ValueError("weight_input_log2_scale_std must be positive")
        self.use_legacy_distribution_encoder_conditioning = bool(
            self.use_distribution_encoder and self.distribution_encoder_conditioning_kind == "legacy"
        )
        self.use_token_adapter_distribution_encoder_conditioning = bool(
            self.use_distribution_encoder and self.distribution_encoder_conditioning_kind == "token_adapter"
        )
        if self.use_mandatory_cross_refresh_v7 and self.use_legacy_distribution_encoder_conditioning:
            raise ValueError(
                "latent_mandatory_cross_refresh_posfilm_v7 forbids legacy direct distribution-to-latent additions"
            )
        if self.use_peri_rms_qk_norm_v4:
            feedback_start = int(cfg.big_vae.latent_feedback_start_layer)
            if not 1 <= feedback_start < num_enc_layers:
                raise ValueError(
                    "big_vae.latent_feedback_start_layer must be in [1, num_encoder_layers), "
                    f"got {feedback_start} for {num_enc_layers} layers"
                )
            weight_additions = 2 * num_enc_layers + (num_enc_layers - feedback_start)
            if self.use_token_adapter_distribution_encoder_conditioning:
                weight_additions += num_enc_layers
            latent_additions = 3 * num_enc_layers
            if self.use_legacy_distribution_encoder_conditioning:
                latent_additions += num_enc_layers
            decoder_additions = (2 if self.use_mandatory_latent_bridge_v5 else 3) * num_dec_layers
            self.peri_rms_v4_residual_scales = {
                "weight_additions": int(weight_additions),
                "latent_additions": int(latent_additions),
                "decoder_additions": int(decoder_additions),
                "tokenizer_additions": 2 * max(1, int(patch_tokenizer_cfg.num_blocks))
                if self.patch_tokenizer_kind == "conditioned_mlp"
                else 1,
                "weight_scale": 1.0 / math.sqrt(float(weight_additions)),
                "latent_scale": 1.0 / math.sqrt(float(latent_additions)),
                "decoder_scale": 1.0 / math.sqrt(float(decoder_additions)),
                "tokenizer_scale": 1.0
                / math.sqrt(
                    float(
                        2 * max(1, int(patch_tokenizer_cfg.num_blocks))
                        if self.patch_tokenizer_kind == "conditioned_mlp"
                        else 1
                    )
                ),
            }
        else:
            self.peri_rms_v4_residual_scales = None
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
                single_gate_v3=self.use_single_gate_conditioning,
                peri_rms_residual=self.use_peri_rms_qk_norm_v4,
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
                peri_rms_residual=self.use_peri_rms_qk_norm_v4,
            )

        if d_model % n_heads != 0:
            raise ValueError(f"big_vae.d_model ({d_model}) must be divisible by big_vae.n_heads ({n_heads})")
        if d_lat % n_heads != 0:
            raise ValueError(f"big_vae.d_lat ({d_lat}) must be divisible by big_vae.n_heads ({n_heads})")

        self.patch_token_proj = nn.Linear(d_patch, d_model)
        self.weight_input_scale_mlp: nn.Module | None = None
        if self.weight_input_normalization_kind != "none":
            # This module is the only additional parameter set in the
            # normalized arm.  Keep its initialization from advancing the
            # global CPU RNG so every pre-existing V4 parameter retains the
            # exact same seed-42 start as the raw graph.
            with torch.random.fork_rng(devices=[]):
                self.weight_input_scale_mlp = nn.Sequential(
                    nn.Linear(1, d_patch),
                    nn.SiLU(),
                    nn.Linear(d_patch, d_patch),
                )
                for module in self.weight_input_scale_mlp.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
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
                    qk_norm=self.use_peri_rms_qk_norm_v4,
                    peri_rms_residual=self.use_peri_rms_qk_norm_v4,
                    weight_residual_scale=(
                        float(self.peri_rms_v4_residual_scales["weight_scale"])
                        if self.peri_rms_v4_residual_scales is not None
                        else 1.0
                    ),
                    latent_residual_scale=(
                        float(self.peri_rms_v4_residual_scales["latent_scale"])
                        if self.peri_rms_v4_residual_scales is not None
                        else 1.0
                    ),
                    mandatory_cross_refresh=self.use_mandatory_cross_refresh_v7,
                    cross_refresh_rms=self.mandatory_cross_refresh_rms,
                )
                for _ in range(num_enc_layers)
            ]
        )
        if self.use_latent_feedback_postnorm_v2:
            feedback_start_layer = int(cfg.big_vae.latent_feedback_start_layer)
            if not 1 <= feedback_start_layer < num_enc_layers:
                raise ValueError(
                    "big_vae.latent_feedback_start_layer must be in [1, num_encoder_layers), "
                    f"got {feedback_start_layer} for {num_enc_layers} layers"
                )
            self.latent_feedback_start_layer = feedback_start_layer
            self.latent_to_weight_feedback: LatentToWeightFeedback | None = LatentToWeightFeedback(
                d_model=d_model,
                d_lat=d_lat,
                attn_dim=int(cfg.big_vae.latent_feedback_attn_dim),
                n_heads=int(cfg.big_vae.latent_feedback_n_heads),
                dropout=dropout,
                qk_norm=self.use_peri_rms_qk_norm_v4,
                peri_rms_residual=self.use_peri_rms_qk_norm_v4,
                residual_scale=(
                    float(self.peri_rms_v4_residual_scales["weight_scale"])
                    if self.peri_rms_v4_residual_scales is not None
                    else 1.0
                ),
            )
            if self.use_mandatory_cross_refresh_v7:
                self.latent_to_weight_feedback.requires_grad_(False)
            self.encoder_weight_post_norms: nn.ModuleList | None = (
                nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_enc_layers)])
                if self.use_outer_postnorm_v2
                else None
            )
            self.encoder_latent_post_norms: nn.ModuleList | None = (
                nn.ModuleList([nn.LayerNorm(d_lat) for _ in range(num_enc_layers)])
                if self.use_outer_postnorm_v2
                else None
            )
        else:
            self.latent_feedback_start_layer = num_enc_layers
            self.latent_to_weight_feedback = None
            self.encoder_weight_post_norms = None
            self.encoder_latent_post_norms = None
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
            self.enc_dist_to_latent_return_norms: nn.ModuleList | None = (
                nn.ModuleList([NonAffineRMSNorm(d_lat) for _ in range(num_enc_layers)])
                if self.use_peri_rms_qk_norm_v4
                else None
            )
        else:
            self.enc_dist_to_latent_heads = None
            self.enc_dist_to_latent_return_norms = None
        if self.use_token_adapter_distribution_encoder_conditioning:
            self.encoder_conditioning_adapters: nn.ModuleList | None = nn.ModuleList(
                [
                    TokenConditioningAdapter(
                        d_model=d_model,
                        d_y=d_dist,
                        dropout=dropout,
                        single_gate_v3=self.use_single_gate_conditioning,
                        peri_rms_residual=self.use_peri_rms_qk_norm_v4,
                        residual_scale=(
                            float(self.peri_rms_v4_residual_scales["weight_scale"])
                            if self.peri_rms_v4_residual_scales is not None
                            else 1.0
                        ),
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

        self.clean_content_readout_v8: CleanContentReadoutV8 | None = None
        self.hybrid_content_readout_v9: HybridNonlinearContentReadoutV9 | None = None
        self.orthogonal_complement_encoder_v10: OrthogonalComplementEncoderV10 | None = None
        self.four_trunk_complement_encoder_v11: FourTrunkComplementEncoderV11 | None = None
        if self.use_clean_content_carrier:
            if not self.use_distribution_encoder:
                raise ValueError("clean_content_readout_posfilm_v8 requires the distribution encoder for X routing")
            # Module construction and explicit initialization are isolated from
            # the global CPU RNG so the shared V6 decoder/head initialization
            # remains bitwise identical at the same model seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(8008)
                self.clean_content_readout_v8 = CleanContentReadoutV8(
                    patch_size=p,
                    d_context=d_dist,
                    d_latent=d_lat,
                    num_latents=num_latents,
                    position_bands=8,
                    init_seed=8008,
                )
            if self.use_hybrid_nonlinear_content_v9 or self.use_carrier_mean_content_v9a:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(9009)
                    self.hybrid_content_readout_v9 = HybridNonlinearContentReadoutV9(
                        patch_size=p,
                        d_context=d_dist,
                        d_latent=d_lat,
                        num_latents=num_latents,
                        num_blocks=int(cfg.big_vae.v9_refinement_blocks),
                        router_width=int(cfg.big_vae.v9_router_width),
                        router_ffn_width=int(cfg.big_vae.v9_router_ffn_width),
                        score_cap=float(cfg.big_vae.v9_score_cap),
                        residual_scale=float(cfg.big_vae.v9_residual_scale),
                        anchor_sigma_keys=float(cfg.big_vae.v9_anchor_sigma_keys),
                        bypass_refinement=bool(cfg.big_vae.v9_bypass_refinement),
                        init_seed=9009,
                        carrier_mean_content=self.use_carrier_mean_content_v9a,
                        carrier_mix=float(cfg.big_vae.v9a_carrier_mix),
                        protected_anchor_floor=float(
                            cfg.big_vae.v9a_protected_anchor_floor
                        ),
                        carrier_content_aggregation=str(
                            cfg.big_vae.v9a_aggregation
                        ),
                    )

        if self.use_orthogonal_complement_v10:
            if not self.use_distribution_encoder:
                raise ValueError("V10 requires distribution patch context for expert routing")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(10101)
                self.orthogonal_complement_encoder_v10 = OrthogonalComplementEncoderV10(
                    patch_size=p,
                    d_context=d_dist,
                    d_latent=d_lat,
                    num_latents=num_latents,
                    protected_dim=int(cfg.big_vae.v10_protected_dim),
                    adaptive_dim=int(cfg.big_vae.v10_adaptive_dim),
                    num_experts=int(cfg.big_vae.v10_num_experts),
                    router_width=int(cfg.big_vae.v9_router_width),
                    router_ffn_width=int(cfg.big_vae.v9_router_ffn_width),
                    score_cap=float(cfg.big_vae.v9_score_cap),
                    residual_scale=float(cfg.big_vae.v9_residual_scale),
                    anchor_sigma_keys=float(cfg.big_vae.v9_anchor_sigma_keys),
                    protected_anchor_floor=float(cfg.big_vae.v9a_protected_anchor_floor),
                    init_seed=10101,
                )

        if self.use_four_trunk_complement_v11:
            if not self.use_distribution_encoder:
                raise ValueError("V11 requires distribution patch context for trunk routing")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(11101)
                self.four_trunk_complement_encoder_v11 = FourTrunkComplementEncoderV11(
                    patch_size=p,
                    d_context=d_dist,
                    d_latent=d_lat,
                    num_latents=num_latents,
                    protected_dim=int(cfg.big_vae.v10_protected_dim),
                    adaptive_dim=int(cfg.big_vae.v10_adaptive_dim),
                    num_trunks=int(cfg.big_vae.v11_num_trunks),
                    blocks_per_trunk=int(cfg.big_vae.v11_blocks_per_trunk),
                    trunk_dim=int(cfg.big_vae.v11_trunk_dim),
                    router_width=int(cfg.big_vae.v9_router_width),
                    router_ffn_width=int(cfg.big_vae.v9_router_ffn_width),
                    score_cap=float(cfg.big_vae.v9_score_cap),
                    residual_scale=float(cfg.big_vae.v9_residual_scale),
                    anchor_sigma_keys=float(cfg.big_vae.v9_anchor_sigma_keys),
                    protected_anchor_floor=float(cfg.big_vae.v9a_protected_anchor_floor),
                    init_seed=11101,
                )

        self.dec_L_latents = num_latents
        self.latent_to_decoder: nn.Linear | None = (
            None if self.use_mandatory_latent_bridge_v5 else nn.Linear(d_lat, d_model)
        )
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
        if self.use_mandatory_latent_bridge_v5:
            self.mandatory_latent_bridge: MandatoryLatentBridge | None = MandatoryLatentBridge(
                d_query=d_model,
                d_latent=(
                    int(cfg.big_vae.v10_adaptive_dim)
                    if self.use_orthogonal_complement_v10
                    or self.use_four_trunk_complement_v11
                    else d_lat
                ),
                d_output=d_model,
                attn_dim=int(cfg.big_vae.decoder_bridge_attn_dim),
                n_heads=int(cfg.big_vae.decoder_bridge_n_heads),
                dropout=dropout,
                qk_norm=True,
            )
            self.position_only_film_v6: FixedPositionOnlyFiLM | None = (
                FixedPositionOnlyFiLM(d_model) if self.use_position_only_film_v6 else None
            )
            self.decoder_layers = nn.ModuleList(
                [
                    DecoderSelfAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        use_rope_2d=True,
                        qk_norm=True,
                    )
                    for _ in range(num_dec_layers)
                ]
            )
            # Standard GPT-style depth scaling for the two decoder residual
            # returns per layer; biases are zeroed by the helper.
            _apply_residual_scaled_init(self.decoder_layers, L_stack=num_dec_layers)
        else:
            self.mandatory_latent_bridge = None
            self.position_only_film_v6 = None
            self.decoder_layers = nn.ModuleList(
                [
                    CrossAttnBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        use_rope_2d=True,
                        qk_norm=self.use_peri_rms_qk_norm_v4,
                        peri_rms_residual=self.use_peri_rms_qk_norm_v4,
                        residual_scale=(
                            float(self.peri_rms_v4_residual_scales["decoder_scale"])
                            if self.peri_rms_v4_residual_scales is not None
                            else 1.0
                        ),
                    )
                    for _ in range(num_dec_layers)
                ]
            )
        self.decoder_post_norms: nn.ModuleList | None = (
            nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_dec_layers)])
            if self.use_outer_postnorm_v2
            else None
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
        self.v10_residual_head: nn.Linear | None = None
        if self.use_orthogonal_complement_v10:
            self.v10_residual_head = nn.Linear(d_model, p, bias=False)
            with torch.no_grad():
                nn.init.normal_(
                    self.v10_residual_head.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(float(d_model)),
                )
        self.v11_residual_head: nn.Linear | None = None
        if self.use_four_trunk_complement_v11:
            self.v11_residual_head = nn.Linear(d_model, p, bias=False)
            with torch.no_grad():
                nn.init.normal_(
                    self.v11_residual_head.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(float(d_model)),
                )
        self.debug_query_hint_proj = nn.Linear(d_model, d_model)
        self.debug_encoder_direct_direction_head = nn.Linear(d_model, p)
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0
        self.direction_seq_norm_eps = 1e-6

        self.z_shortcut_proj: nn.Linear | None = (
            None if self.use_mandatory_latent_bridge_v5 else nn.Linear(self.flat_lat_dim, d_model)
        )
        self.z_shortcut: nn.Sequential | None = (
            None
            if self.use_mandatory_latent_bridge_v5
            else nn.Sequential(
                nn.Linear(d_model + d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, p),
            )
        )
        if bool(cfg.big_vae.disable_z_shortcut) and self.z_shortcut_proj is not None and self.z_shortcut is not None:
            self.z_shortcut_proj.requires_grad_(False)
            self.z_shortcut.requires_grad_(False)

        if self.use_prenorm_v3:
            self._apply_prenorm_v3_residual_initialization(
                num_enc_layers=num_enc_layers,
                num_dec_layers=num_dec_layers,
            )
        if self.use_direct_content_encoder:
            self._freeze_v8_bypassed_modules()
        if self.use_orthogonal_complement_v10:
            self._freeze_v10_bypassed_output_modules()
            self._zero_and_freeze_v10_decoder_offsets()
        if self.use_four_trunk_complement_v11:
            self._freeze_v10_bypassed_output_modules()
            self._zero_and_freeze_v10_decoder_offsets()

    def _freeze_v10_bypassed_output_modules(self) -> None:
        """Freeze legacy output maps that V10 structurally bypasses."""

        for module_name in ("q_tokens_norm", "direction_head", "scale_head"):
            module = getattr(self, module_name)
            module.requires_grad_(False)

    def _zero_and_freeze_v10_decoder_offsets(self) -> None:
        """Make the V10 za-to-U path exactly zero preserving."""

        modules: tuple[nn.Module, ...] = (
            self.mandatory_latent_bridge,
            self.position_only_film_v6,
            self.decoder_layers,
        )
        for root in modules:
            if root is None:
                continue
            for module in root.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
                    with torch.no_grad():
                        module.bias.zero_()
                    module.bias.requires_grad_(False)

    def _freeze_v8_bypassed_modules(self) -> None:
        """Freeze every constructed module/parameter bypassed by V8 encoding."""

        module_names = (
            "patch_tokenizer",
            "patch_token_proj",
            "encoder_layers",
            "latent_to_weight_feedback",
            "encoder_weight_post_norms",
            "encoder_latent_post_norms",
            "enc_dist_inject_projs",
            "enc_dist_to_latent_heads",
            "enc_dist_to_latent_return_norms",
            "encoder_conditioning_adapters",
            "latent_norm",
            "debug_query_hint_proj",
            "debug_encoder_direct_direction_head",
            "query_pos_proj",
        )
        frozen: list[str] = []
        parameters = dict(self.named_parameters())
        for module_name in module_names:
            module = getattr(self, module_name, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(False)
                frozen.extend(name for name in parameters if name.startswith(f"{module_name}."))
        for parameter_name in ("cls_token", "latent_base", "vamp_prior_base"):
            parameter = getattr(self, parameter_name, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad_(False)
                frozen.append(parameter_name)
        for module_name in ("to_mu", "to_logvar"):
            module = getattr(self, module_name, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(False)
                frozen.extend(name for name in parameters if name.startswith(f"{module_name}."))
        self.v8_bypassed_frozen_parameter_names = tuple(sorted(set(frozen)))

    @staticmethod
    def _scale_residual_projection(projection: nn.Linear, scale: float) -> None:
        with torch.no_grad():
            projection.weight.mul_(float(scale))
            if projection.bias is not None:
                projection.bias.zero_()

    def _apply_prenorm_v3_residual_initialization(
        self,
        *,
        num_enc_layers: int,
        num_dec_layers: int,
    ) -> None:
        """Scale only projections whose outputs are added to residual streams."""

        weight_additions = 2 * int(num_enc_layers) + max(
            0,
            int(num_enc_layers) - int(self.latent_feedback_start_layer),
        )
        if self.encoder_conditioning_adapters is not None:
            weight_additions += int(num_enc_layers)
        latent_additions = 3 * int(num_enc_layers)
        decoder_additions = 3 * int(num_dec_layers)
        weight_scale = 1.0 / math.sqrt(float(weight_additions))
        latent_scale = 1.0 / math.sqrt(float(latent_additions))
        decoder_scale = 1.0 / math.sqrt(float(decoder_additions))
        self.prenorm_v3_residual_init = {
            "weight_additions": int(weight_additions),
            "latent_additions": int(latent_additions),
            "decoder_additions": int(decoder_additions),
            "weight_scale": float(weight_scale),
            "latent_scale": float(latent_scale),
            "decoder_scale": float(decoder_scale),
        }

        for layer in self.encoder_layers:
            self._scale_residual_projection(layer.local_block.out_proj, weight_scale)
            self._scale_residual_projection(layer.local_block.ffn.fc2, weight_scale)
            self._scale_residual_projection(layer.perceiver_block.cross_out_proj, latent_scale)
            self._scale_residual_projection(layer.perceiver_block.self_out_proj, latent_scale)
            latent_ffn_out = layer.perceiver_block.ffn[-2]
            if not isinstance(latent_ffn_out, nn.Linear):
                raise TypeError("Perceiver FFN return projection must be nn.Linear")
            self._scale_residual_projection(latent_ffn_out, latent_scale)

        if self.latent_to_weight_feedback is not None:
            self._scale_residual_projection(self.latent_to_weight_feedback.out, weight_scale)
        if self.encoder_conditioning_adapters is not None:
            for adapter in self.encoder_conditioning_adapters:
                self._scale_residual_projection(adapter.mix_out, weight_scale)

        patch_blocks = getattr(self.patch_tokenizer, "blocks", None)
        if patch_blocks is not None:
            tokenizer_scale = 1.0 / math.sqrt(float(2 * len(patch_blocks)))
            for block in patch_blocks:
                self._scale_residual_projection(block.base_fc2, tokenizer_scale)
                self._scale_residual_projection(block.cond_fc2, tokenizer_scale)

        for layer in self.decoder_layers:
            self._scale_residual_projection(layer.out_proj, decoder_scale)
            self._scale_residual_projection(layer.self_out_proj, decoder_scale)
            decoder_ffn_out = layer.ffn[-2]
            if not isinstance(decoder_ffn_out, nn.Linear):
                raise TypeError("Decoder FFN return projection must be nn.Linear")
            self._scale_residual_projection(decoder_ffn_out, decoder_scale)

__all__ = [
    'BigWeightVAE',
]
