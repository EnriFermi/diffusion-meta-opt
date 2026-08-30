from __future__ import annotations

from dataset.big_vae_latent_diffusion_offline_parts import (
    BigVAELatentDiffusionOfflineWriter,
    OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
    OfflineBigVAELatentDiffusionDataset,
    collate_big_vae_latent_diffusion_batch,
    offline_big_vae_latent_diffusion_data_pipeline,
)
from dataset.big_vae_latent_diffusion_offline_parts import builder as _builder
from dataset.big_vae_latent_diffusion_offline_parts.source_slicing import (
    _ExplicitSliceTarget,
    _allocate_capped_proportional_quotas,
    _build_latent_diffusion_batch_from_source_states,
    _directory_size_bytes,
    _distribution_match_group_key_for_sample,
    _latent_diffusion_source_states_total_remaining_slices,
    _make_latent_diffusion_source_state,
    _prepare_cpu_tensor,
    _prune_exhausted_latent_diffusion_source_states_with_offset,
)
from training.big_vae_latent_diffusion import encode_big_vae_layer_batch


def build_big_vae_latent_diffusion_offline_dataset(*args, **kwargs):
    _builder.encode_big_vae_layer_batch = encode_big_vae_layer_batch
    return _builder.build_big_vae_latent_diffusion_offline_dataset(*args, **kwargs)


__all__ = [
    "BigVAELatentDiffusionOfflineWriter",
    "OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION",
    "OfflineBigVAELatentDiffusionDataset",
    "build_big_vae_latent_diffusion_offline_dataset",
    "collate_big_vae_latent_diffusion_batch",
    "offline_big_vae_latent_diffusion_data_pipeline",
    "_ExplicitSliceTarget",
    "_allocate_capped_proportional_quotas",
    "_build_latent_diffusion_batch_from_source_states",
    "_directory_size_bytes",
    "_distribution_match_group_key_for_sample",
    "_latent_diffusion_source_states_total_remaining_slices",
    "_make_latent_diffusion_source_state",
    "_prepare_cpu_tensor",
    "_prune_exhausted_latent_diffusion_source_states_with_offset",
    "encode_big_vae_layer_batch",
]
