from __future__ import annotations

from big_vae.models.big_weight_vae_parts.config import *
from big_vae.models.big_weight_vae_parts.blocks import *
from big_vae.models.big_weight_vae_parts.core import BigWeightVAE
from big_vae.models.big_weight_vae_parts.smoke import smoke_test_big_weight_vae
from big_vae.models.distribution_encoder import CrossLayer, DCNv2, DistributionConfig, InputDistributionEncodingModule
from big_vae.models.mini_patch_vae import MiniVAEConfig
from big_vae.models.patch_tokenizers import MixerPatchTokenizer, PatchConditionedMLPTokenizer, ResidualPatchTokenizer

WeightQuantileVAE = BigWeightVAE


def build_weight_quantile_vae(model_cfg: ModelConfig) -> BigWeightVAE:
    return BigWeightVAE(model_cfg)

for _name in [
    "EncoderConfig",
    "TTMMemoryConfig",
    "BigVAEConfig",
    "ResamplerConfig",
    "ModelConfig",
    "LocalOutputSelfAttentionBlock",
    "TokenConditioningAdapter",
    "LatentEncoderLayer",
    "DecoderCrossBlock",
    "BigWeightVAE",
]:
    if _name in globals():
        globals()[_name].__module__ = __name__

del _name

__all__ = [
    "BigVAEConfig",
    "BigWeightVAE",
    "CrossLayer",
    "DCNv2",
    "DecoderCrossBlock",
    "DistributionConfig",
    "EncoderConfig",
    "InputDistributionEncodingModule",
    "LatentEncoderLayer",
    "LocalOutputSelfAttentionBlock",
    "MiniVAEConfig",
    "MixerPatchTokenizer",
    "ModelConfig",
    "PatchConditionedMLPTokenizer",
    "ResidualPatchTokenizer",
    "ResamplerConfig",
    "TTMMemoryConfig",
    "WeightQuantileVAE",
    "build_weight_quantile_vae",
    "smoke_test_big_weight_vae",
]


if __name__ == "__main__":
    smoke_test_big_weight_vae()
