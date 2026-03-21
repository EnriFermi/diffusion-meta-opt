from __future__ import annotations

from models.big_weight_vae import (
    BigVAEConfig,
    BigWeightVAE,
    DecoderCrossBlock,
    EncoderConfig,
    LatentEncoderLayer,
    LocalOutputSelfAttentionBlock,
    ModelConfig,
    ResamplerConfig,
    TTMMemoryConfig,
    WeightQuantileVAE,
    smoke_test_big_weight_vae,
)
from models.distribution_encoder import CrossLayer, DCNv2, DistributionConfig, InputDistributionEncodingModule
from models.mini_patch_vae import (
    CrossAttnPatchDecoder,
    MLPNoCompressionPatchDecoder,
    MLPNoCompressionPatchEncoder,
    MiniPatchDecoder,
    MiniPatchEncoder,
    MiniPatchVAE,
    MiniPatchVAEStub,
    MiniVAEConfig,
    TransformerNoCompressionPatchEncoder,
)
from models.patch_tokenizers import MixerPatchTokenizer, ResidualPatchTokenizer
from models.simple_big_weight_vae import SimpleDirectBigWeightVAE, build_weight_quantile_vae
from models.vae_shared import CrossAttnBlock, MLP, PerceiverResamplerBlock, RoPeMixed2D, sinusoidal_embedding

__all__ = [
    "BigVAEConfig",
    "BigWeightVAE",
    "SimpleDirectBigWeightVAE",
    "CrossAttnBlock",
    "CrossAttnPatchDecoder",
    "CrossLayer",
    "DCNv2",
    "DecoderCrossBlock",
    "DistributionConfig",
    "EncoderConfig",
    "InputDistributionEncodingModule",
    "LatentEncoderLayer",
    "LocalOutputSelfAttentionBlock",
    "MLP",
    "MLPNoCompressionPatchDecoder",
    "MLPNoCompressionPatchEncoder",
    "MiniPatchDecoder",
    "MiniPatchEncoder",
    "MiniPatchVAE",
    "MiniPatchVAEStub",
    "MiniVAEConfig",
    "MixerPatchTokenizer",
    "ModelConfig",
    "PerceiverResamplerBlock",
    "ResidualPatchTokenizer",
    "ResamplerConfig",
    "RoPeMixed2D",
    "TTMMemoryConfig",
    "TransformerNoCompressionPatchEncoder",
    "WeightQuantileVAE",
    "build_weight_quantile_vae",
    "sinusoidal_embedding",
    "smoke_test_big_weight_vae",
]


if __name__ == "__main__":
    smoke_test_big_weight_vae()
