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

@dataclass(slots=True)
class EncoderConfig:
    self_attn_mode: str = "full"
    cross_attend_only_cls: bool = True


@dataclass(slots=True)
class TTMMemoryConfig:
    proc_tokens: int = 8
    process_depth: int = 2
    summarizer_mode: str = "mlp"
    summarizer_hidden_mult: float = 2.0
    num_blocks: int = 1
    share_weights: bool = False
    dropout: float = 0.0
    use_type_embeddings: bool = True
    use_positional_embeddings: bool = True
    memory_init: str = "learned"
    return_aux: bool = False


@dataclass(slots=True)
class PatchTokenizerConfig:
    kind: str = ""
    d_patch: int = 0
    hidden_dim: int | None = None
    cond_proj_dim: int | None = None
    num_blocks: int = 2
    dropout: float | None = None
    residual_hidden_dim: int = 256
    residual_num_layers: int = 3
    residual_dropout: float = 0.0


@dataclass(slots=True)
class BigVAEConfig:
    # ``legacy_v1`` keeps the historical graph. V2 adds feedback plus outer
    # postnorm, v3 removes only that outer postnorm, and v4 adds parameter-free
    # Peri-RMS residual returns plus projected per-head QK normalization to the
    # main AE stack. The distribution encoder remains outside the v4 contract.
    # V5 adds the mandatory z-value bridge; V6 adds only a fixed, multiplicative
    # position FiLM immediately after that bridge. V7 keeps the V6 decoder and
    # replaces each latent state with a fixed-RMS cross-attention refresh. V8
    # bypasses that encoder with one clean bias-free W-value/X-routing readout
    # and feeds its z directly to the unchanged V6 bridge/PosFiLM decoder.
    # ``latent_feedback_prenorm_v3`` keeps the feedback and normalized output
    # heads, but removes the repeated outer post-norms and uses depth-scaled
    # residual initialization with single-gate conditioning.
    architecture_version: str = "legacy_v1"
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
    use_encoder_mu_head: bool = False
    # Compatibility switch for the historical global flattened LayerNorm at
    # the encoder output.  False gives the VAE path raw Perceiver latents for
    # posterior heads and decoder input.
    normalize_latent_slots_before_mu: bool = True
    latent_prior_kind: str = "gaussian"
    vamp_prior_K: int = 64
    decoder_query_conditioning_kind: str = "linear"
    decoder_query_conditioning_hidden_mult: float = 2.0
    rope_2d_coord_kind: str = "normalized_center"
    latent_sampling_min_std: float = 1e-4
    latent_sampling_logvar_min: float = -20.0
    latent_sampling_logvar_max: float = 10.0
    disable_z_shortcut: bool = False
    disable_distribution_encoder: bool = False
    patch_tokenizer_kind: str = "residual"
    distribution_encoder_conditioning_kind: str = "legacy"
    # Optional encoder-only representation change.  The decoder and losses
    # always stay in the original weight domain.  ``per_output_maxabs_q7``
    # divides every output column by one local max-abs/7 scale and injects the
    # standardized log2 scale into the ordinary patch-token stream.
    weight_input_normalization_kind: str = "none"
    weight_input_scale_qmax: float = 7.0
    weight_input_log2_scale_mean: float = 0.0
    weight_input_log2_scale_std: float = 1.0
    latent_bottleneck_kind: str = "perceiver_resampler"
    latent_feedback_start_layer: int = 1
    latent_feedback_attn_dim: int = 512
    latent_feedback_n_heads: int = 16
    decoder_bridge_attn_dim: int = 512
    decoder_bridge_n_heads: int = 8
    v9_refinement_blocks: int = 40
    v9_router_width: int = 384
    v9_router_ffn_width: int = 1536
    v9_score_cap: float = 0.1
    v9_residual_scale: float = 0.11180339887498948
    v9_anchor_sigma_keys: float = 0.6
    v9_bypass_refinement: bool = False
    v9a_carrier_mix: float = 0.1
    v9a_protected_anchor_floor: float = 0.25
    v9a_aggregation: str = "sqrt_depth_sum"
    v10_protected_dim: int = 320
    v10_adaptive_dim: int = 64
    v10_num_experts: int = 40
    v11_num_trunks: int = 4
    v11_blocks_per_trunk: int = 10
    v11_trunk_dim: int = 16
    patch_tokenizer: PatchTokenizerConfig = field(default_factory=PatchTokenizerConfig)
    ttm: TTMMemoryConfig = field(default_factory=TTMMemoryConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


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


def _normalize_rope_2d_coord_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value in {"raw", "legacy"}:
        return "raw"
    if value in {"normalized_center", "normalized", "centered"}:
        return "normalized_center"
    raise ValueError(
        "big_vae.rope_2d_coord_kind must be one of "
        "'raw', 'normalized_center', "
        f"got {kind!r}"
    )


def _make_rope_positions(indices: torch.Tensor, *, axis_size: int, coord_kind: str) -> torch.Tensor:
    pos = indices.to(dtype=torch.float32)
    if _normalize_rope_2d_coord_kind(coord_kind) == "normalized_center":
        return (pos + 0.5) / max(float(axis_size), 1.0)
    return pos


def _make_rope_axis_positions(length: int, *, device: torch.device, coord_kind: str) -> torch.Tensor:
    base = torch.arange(int(length), device=device, dtype=torch.float32)
    return _make_rope_positions(base, axis_size=int(length), coord_kind=coord_kind)

__all__ = [
    'EncoderConfig',
    'TTMMemoryConfig',
    'PatchTokenizerConfig',
    'BigVAEConfig',
    'ResamplerConfig',
    'ModelConfig',
    '_normalize_rope_2d_coord_kind',
    '_make_rope_positions',
    '_make_rope_axis_positions',
]
