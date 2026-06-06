from __future__ import annotations

from big_vae.models.big_weight_vae import *  # noqa: F401,F403
from big_vae.models.big_weight_vae import __all__ as _big_weight_vae_all
from big_vae.models.layer_latent_diffusion_prior import (
    DiffusionSchedule,
    DiffusionScheduleConfig,
    LayerLatentDiffusionPrior,
    LayerLatentDiffusionPriorConfig,
    compute_layer_latent_diffusion_loss,
)
from big_vae.models.mini_patch_vae import (
    CrossAttnPatchDecoder,
    MLPNoCompressionPatchDecoder,
    MLPNoCompressionPatchEncoder,
    MiniPatchDecoder,
    MiniPatchEncoder,
    MiniPatchVAE,
    MiniPatchVAEStub,
    TransformerNoCompressionPatchEncoder,
)
from big_vae.models.vae_shared import (
    CrossAttnBlock,
    MLP,
    PerceiverResamplerBlock,
    RoPeMixed2D,
    _decode_direction_and_logscale,
    sinusoidal_embedding,
)

__all__ = list(_big_weight_vae_all) + [
    "CrossAttnBlock",
    "CrossAttnPatchDecoder",
    "DiffusionSchedule",
    "DiffusionScheduleConfig",
    "LayerLatentDiffusionPrior",
    "LayerLatentDiffusionPriorConfig",
    "MLP",
    "MLPNoCompressionPatchDecoder",
    "MLPNoCompressionPatchEncoder",
    "MiniPatchDecoder",
    "MiniPatchEncoder",
    "MiniPatchVAE",
    "MiniPatchVAEStub",
    "PerceiverResamplerBlock",
    "RoPeMixed2D",
    "TransformerNoCompressionPatchEncoder",
    "_decode_direction_and_logscale",
    "compute_layer_latent_diffusion_loss",
    "sinusoidal_embedding",
]

del _big_weight_vae_all
