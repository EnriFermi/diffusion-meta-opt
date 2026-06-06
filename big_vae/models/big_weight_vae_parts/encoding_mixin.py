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

class BigWeightVAEEncodingMixin:
    @staticmethod
    def _build_batched_patch_indices(
        *,
        d_in_mask: torch.Tensor,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if d_in_mask.ndim != 2:
            raise ValueError(f"d_in_mask must be [B, d_in], got {tuple(d_in_mask.shape)}")
        B, d_in = d_in_mask.shape
        if int(d_in) <= 0:
            raise ValueError(f"d_in must be > 0, got {d_in}")

        valid_d_in = d_in_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1)
        T = (int(d_in) + int(patch_size) - 1) // int(patch_size)
        d_in_pad = T * int(patch_size)
        base_idx = torch.arange(d_in_pad, device=d_in_mask.device, dtype=torch.long).view(1, T, int(patch_size))
        patch_idx = base_idx.expand(B, -1, -1).clamp(max=(valid_d_in - 1).view(B, 1, 1))

        patch_grid = torch.arange(T, device=d_in_mask.device, dtype=torch.long).view(1, T)
        valid_patch_counts = torch.div(valid_d_in + int(patch_size) - 1, int(patch_size), rounding_mode="floor")
        full_patch_counts = torch.div(valid_d_in, int(patch_size), rounding_mode="floor")
        patch_mask = patch_grid < valid_patch_counts.view(B, 1)
        structural_patch_mask = patch_grid < full_patch_counts.view(B, 1)
        return patch_idx, patch_mask, structural_patch_mask, valid_d_in, T, d_in_pad

    def _encode_distribution_context(
        self,
        X: torch.Tensor,
        *,
        d_in: int | None = None,
        x_mask: torch.Tensor | None = None,
        d_in_mask: torch.Tensor | None = None,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if X.ndim != 3:
            raise ValueError(f"X must be rank-3 [B, n, d_in], got {tuple(X.shape)}")

        B, n, d_in_x = X.shape
        legacy_return_shape = d_in is not None
        if d_in is not None and int(d_in) != int(d_in_x):
            raise ValueError(f"d_in must match X.shape[-1], got d_in={d_in} vs X.shape[-1]={d_in_x}")
        if x_mask is not None:
            if x_mask.ndim != 2:
                raise ValueError(f"x_mask must be rank-2 [B, n], got {tuple(x_mask.shape)}")
            if tuple(x_mask.shape) != (B, n):
                raise ValueError(f"x_mask shape must be {(B, n)}, got {tuple(x_mask.shape)}")

        device = X.device
        p = self.cfg.patch_size
        validated_d_in_mask = self._validate_d_in_mask(
            d_in_mask,
            batch_size=B,
            d_in=d_in_x,
            device=device,
        )
        patch_idx_bt, patch_mask, structural_patch_mask, _valid_d_in, T, d_in_pad = self._build_batched_patch_indices(
            d_in_mask=validated_d_in_mask,
            patch_size=p,
        )
        if not self.use_distribution_encoder:
            if legacy_return_shape:
                return T, d_in_pad, None, None, None
            return T, d_in_pad, patch_mask, structural_patch_mask, None, None, None
        if self.distribution_encoder is None:
            raise RuntimeError("distribution_encoder is not initialized")

        X_rep = X.unsqueeze(1).expand(B, T, n, d_in_x).reshape(B * T, n, d_in_x)
        x_mask_rep = (
            x_mask.unsqueeze(1).expand(B, T, n).reshape(B * T, n)
            if x_mask is not None
            else None
        )
        patch_idx_flat = patch_idx_bt.reshape(B * T, p)

        dist_var_flat, dist_patch_flat = self.distribution_encoder(X_rep, patch_idx_flat, sample_mask=x_mask_rep)

        d_var = self.cfg.distribution.d_var
        d_dist = self.cfg.distribution.d_dist
        dist_var_by_patch = dist_var_flat.view(B, T, p, d_var)
        dist_patch_by_patch = dist_patch_flat.view(B, T, d_dist)
        patch_mask_float = patch_mask.to(device=device, dtype=dist_var_by_patch.dtype)
        dist_var_by_patch = dist_var_by_patch * patch_mask_float.unsqueeze(-1).unsqueeze(-1)
        dist_patch_by_patch = dist_patch_by_patch * patch_mask_float.unsqueeze(-1)
        dist_var_pooled = dist_var_by_patch.mean(dim=2)
        if legacy_return_shape:
            return T, d_in_pad, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled
        return T, d_in_pad, patch_mask, structural_patch_mask, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled

    def _encode_latent_slots(
        self,
        W: torch.Tensor,
        *,
        T: int,
        d_in_pad: int,
        patch_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        dist_var_by_patch: torch.Tensor | None,
        dist_patch_by_patch: torch.Tensor | None,
        dist_var_pooled: torch.Tensor | None,
        return_debug_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if W.ndim != 3:
            raise ValueError(f"W must be rank-3 [B, d_in, d_out], got {tuple(W.shape)}")

        B, d_in, d_out = W.shape
        p = self.cfg.patch_size
        d_var = self.cfg.distribution.d_var
        d_dist = self.cfg.distribution.d_dist
        if patch_mask.ndim != 2 or tuple(patch_mask.shape) != (B, T):
            raise ValueError(f"patch_mask must be {(B, T)}, got {tuple(patch_mask.shape)}")
        if d_out_mask.ndim != 2 or tuple(d_out_mask.shape) != (B, d_out):
            raise ValueError(f"d_out_mask must be {(B, d_out)}, got {tuple(d_out_mask.shape)}")
        patch_mask_float = patch_mask.to(device=W.device, dtype=W.dtype)
        d_out_mask_float = d_out_mask.to(device=W.device, dtype=W.dtype)

        if self.use_distribution_encoder:
            if dist_var_by_patch is None or dist_patch_by_patch is None or dist_var_pooled is None:
                raise ValueError("distribution tensors are required when distribution encoder is enabled")
            if tuple(dist_var_by_patch.shape[:3]) != (B, T, p):
                raise ValueError(
                    "dist_var_by_patch must be [B, T, p, d_var], got "
                    f"{tuple(dist_var_by_patch.shape)} for expected {(B, T, p, d_var)}"
                )
            if int(dist_var_by_patch.shape[3]) != d_var:
                raise ValueError(f"dist_var_by_patch last dim must be {d_var}, got {int(dist_var_by_patch.shape[3])}")
            if tuple(dist_patch_by_patch.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_patch_by_patch must be [B, T, d_dist], got "
                    f"{tuple(dist_patch_by_patch.shape)} for expected {(B, T, d_dist)}"
                )
            if int(dist_patch_by_patch.shape[2]) != d_dist:
                raise ValueError(
                    f"dist_patch_by_patch last dim must be {d_dist}, got {int(dist_patch_by_patch.shape[2])}"
                )
            if tuple(dist_var_pooled.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_var_pooled must be [B, T, d_var], got "
                    f"{tuple(dist_var_pooled.shape)} for expected {(B, T, d_var)}"
                )
        elif dist_var_by_patch is not None or dist_patch_by_patch is not None or dist_var_pooled is not None:
            raise ValueError("distribution tensors must be None when distribution encoder is disabled")

        W_pad = torch.zeros(B, d_in_pad, d_out, device=W.device, dtype=W.dtype)
        W_pad[:, :d_in, :] = W * d_out_mask_float.unsqueeze(1)
        w_patches = W_pad.transpose(1, 2).contiguous().view(B, d_out, T, p)

        w_flat = w_patches.reshape(B * d_out * T, p)
        if self.use_distribution_encoder:
            assert dist_var_by_patch is not None
            assert dist_patch_by_patch is not None
            assert dist_var_pooled is not None
            needs_dist_patch_embed = not isinstance(self.patch_tokenizer, PatchConditionedMLPTokenizer)
            needs_dist_var_patch_tokens = isinstance(self.patch_tokenizer, (PatchConditionedMLPTokenizer, MixerPatchTokenizer))

            if needs_dist_patch_embed:
                dist_patch_expanded = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1)
                dist_patch_flat_expanded = dist_patch_expanded.reshape(B * d_out * T, d_dist)
            else:
                dist_patch_flat_expanded = None

            if needs_dist_var_patch_tokens:
                dist_var_expanded = dist_var_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1, -1)
                dist_var_flat_expanded: torch.Tensor | None = dist_var_expanded.reshape(B * d_out * T, p, d_var)
            else:
                dist_var_flat_expanded = None

            if self.use_legacy_distribution_encoder_conditioning:
                dist_var_for_inject = dist_var_pooled.unsqueeze(1).expand(-1, d_out, -1, -1)
                valid_patch_counts = patch_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
                dist_var_global = (dist_var_pooled * patch_mask_float.unsqueeze(-1)).sum(dim=1) / valid_patch_counts
            else:
                dist_var_for_inject = None
                dist_var_global = None
        else:
            dist_patch_flat_expanded = w_flat.new_zeros((B * d_out * T, 0))
            dist_var_flat_expanded = None
            dist_var_for_inject = None
            dist_var_global = None

        d_patch = self.cfg.mini_vae.d_patch
        patch_token_raw_flat = self.patch_tokenizer(
            w_patch=w_flat,
            dist_var_tokens=dist_var_flat_expanded,
            dist_patch_embed=dist_patch_flat_expanded,
        )
        patch_token_raw = patch_token_raw_flat.view(B, d_out, T, d_patch)
        patch_token_raw = patch_token_raw * patch_mask_float.unsqueeze(1).unsqueeze(-1) * d_out_mask_float.unsqueeze(-1).unsqueeze(-1)
        patch_tokens = self.patch_token_proj(patch_token_raw)
        patch_tokens = patch_tokens * patch_mask_float.unsqueeze(1).unsqueeze(-1) * d_out_mask_float.unsqueeze(-1).unsqueeze(-1)

        cls_tokens = self.cls_token.view(1, 1, 1, -1).expand(B, d_out, 1, -1)
        cls_tokens = cls_tokens * d_out_mask_float.unsqueeze(-1).unsqueeze(-1)
        tokens_by_output = torch.cat([cls_tokens, patch_tokens], dim=2)
        token_valid_mask = torch.cat(
            [
                torch.ones((B, d_out, 1), device=W.device, dtype=torch.bool),
                patch_mask.unsqueeze(1).expand(-1, d_out, -1),
            ],
            dim=2,
        )
        if self.use_token_adapter_distribution_encoder_conditioning:
            assert dist_patch_by_patch is not None
            assert self.encoder_conditioning_adapters is not None
            dist_patch_ctx_by_output = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1).reshape(B * d_out, T, d_dist)
            patch_map_idx = torch.arange(T, device=W.device, dtype=torch.long).view(1, T).expand(B * d_out, -1)
        else:
            dist_patch_ctx_by_output = None
            patch_map_idx = None

        latents = self.latent_base.unsqueeze(0).expand(B, -1, -1)
        num_latents = self.cfg.big_vae.num_latents
        d_lat = self.cfg.big_vae.d_lat

        for layer_idx, enc_layer in enumerate(self.encoder_layers):
            if self.use_legacy_distribution_encoder_conditioning:
                assert dist_var_for_inject is not None
                assert dist_var_global is not None
                assert self.enc_dist_inject_projs is not None
                assert self.enc_dist_to_latent_heads is not None
                cls_part = tokens_by_output[:, :, :1, :]
                patch_part = tokens_by_output[:, :, 1:, :]
                patch_with_dist = torch.cat([patch_part, dist_var_for_inject], dim=-1)
                patch_part = self.enc_dist_inject_projs[layer_idx](patch_with_dist)
                patch_part = patch_part * patch_mask_float.unsqueeze(1).unsqueeze(-1) * d_out_mask_float.unsqueeze(-1).unsqueeze(-1)
                tokens_by_output = torch.cat([cls_part, patch_part], dim=2)

                dist_lat_delta = self.enc_dist_to_latent_heads[layer_idx](dist_var_global)
                latents = latents + dist_lat_delta.view(B, num_latents, d_lat)

            tokens_by_output, latents = enc_layer(
                tokens_by_output=tokens_by_output,
                latents=latents,
                cross_attend_only_cls=self.cfg.big_vae.encoder.cross_attend_only_cls,
                patch_conditioner=(
                    self.encoder_conditioning_adapters[layer_idx]
                    if self.use_token_adapter_distribution_encoder_conditioning and self.encoder_conditioning_adapters is not None
                    else None
                ),
                patch_conditioning_ctx=dist_patch_ctx_by_output,
                patch_conditioning_map_idx=patch_map_idx,
                token_valid_mask=token_valid_mask,
                output_valid_mask=d_out_mask,
            )
            tokens_by_output = torch.cat(
                [
                    tokens_by_output[:, :, :1, :] * d_out_mask_float.unsqueeze(-1).unsqueeze(-1),
                    tokens_by_output[:, :, 1:, :] * patch_mask_float.unsqueeze(1).unsqueeze(-1) * d_out_mask_float.unsqueeze(-1).unsqueeze(-1),
                ],
                dim=2,
            )

        if return_debug_info:
            return latents, {
                "debug_patch_tokenizer_kind": self.patch_tokenizer_kind,
                "debug_encoder_conditioning_kind": self.distribution_encoder_conditioning_kind,
                "encoder_tokens_by_output": tokens_by_output,
                "encoder_cls_tokens": tokens_by_output[:, :, :1, :],
                "encoder_patch_tokens": tokens_by_output[:, :, 1:, :],
            }
        return latents


__all__ = [
    'BigWeightVAEEncodingMixin',
]
