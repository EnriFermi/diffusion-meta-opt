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
    latent_bottleneck_kind: str = "perceiver_resampler"
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
