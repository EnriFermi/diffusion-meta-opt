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
from big_vae.models.big_weight_vae_parts.core import BigWeightVAE

def smoke_test_big_weight_vae() -> None:
    torch.manual_seed(0)

    p = 16
    num_latents = 16
    d_lat = 256
    z_dim_expected = num_latents * d_lat

    cfg = ModelConfig(
        patch_size=p,
        distribution=DistributionConfig(
            k_s=16,
            Kq=32,
            d_var=128,
            d_dist=128,
            use_covariance=True,
            patch_size_for_cov=p,
        ),
        mini_vae=MiniVAEConfig(
            z_dim=64,
            d_e=128,
            encoder_latent_dim=8,
            num_attn_layers_encoder=2,
            num_layers_decoder=2,
            n_heads=4,
            d_patch=64,
            mlp_stub_hidden_dim=256,
        ),
        big_vae=BigVAEConfig(
            d_model=256,
            d_lat=d_lat,
            num_latents=num_latents,
            num_encoder_layers=2,
            num_decoder_layers=2,
            n_heads=8,
            use_latent_sampling=False,
            encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
        ),
    )

    model = BigWeightVAE(cfg)
    B, n, d_in, d_out = 2, 32, 65, 7
    X = torch.randn(B, n, d_in)
    W = torch.randn(B, d_in, d_out)

    W_hat, z, logvar, pred_dirs = model(W, X)

    assert tuple(W_hat.shape) == (B, d_in, d_out), f"W_hat shape mismatch: {tuple(W_hat.shape)}"
    assert tuple(z.shape) == (B, z_dim_expected), f"z shape mismatch: {tuple(z.shape)}, expected {(B, z_dim_expected)}"
    assert (logvar == 0).all(), "AE-mode logvar should be all zeros"

    behavioral_loss = BigWeightVAE.operator_recon_loss(X, W, W_hat)
    structural_loss, struct_details = BigWeightVAE.patch_structure_loss(W, W_hat, patch_size=cfg.patch_size, pred_dirs=pred_dirs)
    total_loss = behavioral_loss + 0.5 * structural_loss
    total_loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

    print("smoke_test_big_weight_vae PASSED")
    print(
        f"  W_hat: {tuple(W_hat.shape)}, z: {tuple(z.shape)}, "
        f"behavioral={behavioral_loss.item():.4f}, structural={structural_loss.item():.4f}"
    )
    print(f"  struct details: " + ", ".join(f"{k}={v.item():.4f}" for k, v in struct_details.items()))

__all__ = [
    'smoke_test_big_weight_vae',
]
