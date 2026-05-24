from __future__ import annotations

from models.big_weight_vae_parts.config import *
from models.big_weight_vae_parts.blocks import *
from models.big_weight_vae_parts.core import BigWeightVAE
from models.big_weight_vae_parts.smoke import smoke_test_big_weight_vae
from models.patch_tokenizers import MixerPatchTokenizer, PatchConditionedMLPTokenizer, ResidualPatchTokenizer

WeightQuantileVAE = BigWeightVAE

for _name in ['EncoderConfig', 'TTMMemoryConfig', 'BigVAEConfig', 'ResamplerConfig', 'ModelConfig', 'LocalOutputSelfAttentionBlock', 'TokenConditioningAdapter', 'LatentEncoderLayer', 'DecoderCrossBlock', 'BigWeightVAE']:
    if _name in globals():
        globals()[_name].__module__ = __name__

del _name

__all__ = [
    'BigVAEConfig',
    'BigWeightVAE',
    'DecoderCrossBlock',
    'EncoderConfig',
    'LatentEncoderLayer',
    'LocalOutputSelfAttentionBlock',
    'ModelConfig',
    'ResamplerConfig',
    'TTMMemoryConfig',
    'WeightQuantileVAE',
    'smoke_test_big_weight_vae',
    'MixerPatchTokenizer',
    'ResidualPatchTokenizer',
]
