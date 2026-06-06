from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.distribution_encoder import DistributionConfig, InputDistributionEncodingModule
from models.mini_patch_vae import MiniVAEConfig
from models.patch_tokenizers import MixerPatchTokenizer, PatchConditionedMLPTokenizer, ResidualPatchTokenizer
from models.vae_shared import (
    CrossAttnBlock,
    MLP,
    PerceiverResamplerBlock,
    _apply_sequence_mask,
    _decode_direction_and_logscale,
    _key_padding_to_attn_bias,
    _rope_attention,
    sinusoidal_embedding,
)

from models.big_weight_vae_parts.config import *
from models.big_weight_vae_parts.blocks import *

class BigWeightVAEForwardMixin:
    def forward_debug(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        *,
        x_mask: torch.Tensor | None = None,
        d_in_mask: torch.Tensor | None = None,
        d_out_mask: torch.Tensor | None = None,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
        debug_decoder_kv_source: str = "latents",
        debug_query_hint: str = "none",
        debug_direct_from_encoder_tokens: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if W.ndim not in {2, 3}:
            raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
        if X.ndim not in {2, 3}:
            raise ValueError(f"X must be rank-2 or rank-3, got {tuple(X.shape)}")

        squeeze_batch = W.ndim == 2
        if squeeze_batch != (X.ndim == 2):
            raise ValueError(f"W and X must be both batched or both unbatched, got W={tuple(W.shape)}, X={tuple(X.shape)}")

        if squeeze_batch:
            W = W.unsqueeze(0)
            X = X.unsqueeze(0)
            if x_mask is not None:
                if x_mask.ndim != 1:
                    raise ValueError(f"x_mask must be rank-1 when X is unbatched, got {tuple(x_mask.shape)}")
                x_mask = x_mask.unsqueeze(0)
            if d_in_mask is not None:
                if d_in_mask.ndim != 1:
                    raise ValueError(f"d_in_mask must be rank-1 when W is unbatched, got {tuple(d_in_mask.shape)}")
                d_in_mask = d_in_mask.unsqueeze(0)
            if d_out_mask is not None:
                if d_out_mask.ndim != 1:
                    raise ValueError(f"d_out_mask must be rank-1 when W is unbatched, got {tuple(d_out_mask.shape)}")
                d_out_mask = d_out_mask.unsqueeze(0)
        B, d_in, d_out = W.shape
        Bx, _, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        T, d_in_pad, patch_mask, structural_patch_mask, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled = self._encode_distribution_context(
            X,
            x_mask=x_mask,
            d_in_mask=d_in_mask,
        )
        latents, encode_debug = self._encode_latent_slots(
            W,
            T=T,
            d_in_pad=d_in_pad,
            patch_mask=patch_mask,
            d_out_mask=self._validate_d_out_mask(d_out_mask, batch_size=B, d_out=d_out, device=W.device),
            dist_var_by_patch=dist_var_by_patch,
            dist_patch_by_patch=dist_patch_by_patch,
            dist_var_pooled=dist_var_pooled,
            return_debug_info=True,
        )
        decode_outputs = self._decode_from_latent_slots(
            latents,
            dist_patch_by_patch=dist_patch_by_patch,
            patch_mask=patch_mask,
            d_in_mask=self._validate_d_in_mask(d_in_mask, batch_size=B, d_in=d_in, device=W.device),
            d_out_mask=self._validate_d_out_mask(d_out_mask, batch_size=B, d_out=d_out, device=W.device),
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            T=T,
            return_direction_pre_norms=return_direction_pre_norms,
            disable_z_shortcut=disable_z_shortcut,
            encoder_patch_tokens=encode_debug["encoder_patch_tokens"],
            debug_decoder_kv_source=debug_decoder_kv_source,
            debug_query_hint=debug_query_hint,
            debug_direct_from_encoder_tokens=debug_direct_from_encoder_tokens,
            return_debug_info=True,
        )
        if return_direction_pre_norms:
            W_hat, mu, logvar, pred_dirs, direction_pre_norms, decode_debug = decode_outputs
        else:
            W_hat, mu, logvar, pred_dirs, decode_debug = decode_outputs

        debug_info = {
            "T": int(T),
            "d_in_pad": int(d_in_pad),
            "patch_mask": patch_mask,
            "structural_patch_mask": structural_patch_mask,
            "dist_var_by_patch": dist_var_by_patch,
            "dist_patch_by_patch": dist_patch_by_patch,
            "encoder_tokens_by_output": encode_debug["encoder_tokens_by_output"],
            "encoder_cls_tokens": encode_debug["encoder_cls_tokens"],
            "encoder_patch_tokens": encode_debug["encoder_patch_tokens"],
            **decode_debug,
        }

        if squeeze_batch:
            outputs = (
                W_hat.squeeze(0),
                mu.squeeze(0),
                logvar.squeeze(0),
                pred_dirs.squeeze(0),
            )
            if return_direction_pre_norms:
                return outputs + (direction_pre_norms, debug_info)
            return outputs + (debug_info,)

        outputs = (W_hat, mu, logvar, pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms, debug_info)
        return outputs + (debug_info,)

    def forward(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        *,
        x_mask: torch.Tensor | None = None,
        d_in_mask: torch.Tensor | None = None,
        d_out_mask: torch.Tensor | None = None,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if W.ndim not in {2, 3}:
            raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
        if X.ndim not in {2, 3}:
            raise ValueError(f"X must be rank-2 or rank-3, got {tuple(X.shape)}")

        squeeze_batch = W.ndim == 2
        if squeeze_batch != (X.ndim == 2):
            raise ValueError(f"W and X must be both batched or both unbatched, got W={tuple(W.shape)}, X={tuple(X.shape)}")

        # W = W[0].repeat(W.shape[0], 1, 1)
        # X = X[0].repeat(X.shape[0], 1, 1)

        if squeeze_batch:
            W = W.unsqueeze(0)
            X = X.unsqueeze(0)
            if x_mask is not None:
                if x_mask.ndim != 1:
                    raise ValueError(f"x_mask must be rank-1 when X is unbatched, got {tuple(x_mask.shape)}")
                x_mask = x_mask.unsqueeze(0)
            if d_in_mask is not None:
                if d_in_mask.ndim != 1:
                    raise ValueError(f"d_in_mask must be rank-1 when W is unbatched, got {tuple(d_in_mask.shape)}")
                d_in_mask = d_in_mask.unsqueeze(0)
            if d_out_mask is not None:
                if d_out_mask.ndim != 1:
                    raise ValueError(f"d_out_mask must be rank-1 when W is unbatched, got {tuple(d_out_mask.shape)}")
                d_out_mask = d_out_mask.unsqueeze(0)
        B, d_in, d_out = W.shape
        Bx, _, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        validated_d_in_mask = self._validate_d_in_mask(d_in_mask, batch_size=B, d_in=d_in, device=W.device)
        validated_d_out_mask = self._validate_d_out_mask(d_out_mask, batch_size=B, d_out=d_out, device=W.device)
        T, d_in_pad, patch_mask, _structural_patch_mask, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled = self._encode_distribution_context(
            X,
            x_mask=x_mask,
            d_in_mask=validated_d_in_mask,
        )
        latents, encode_debug = self._encode_latent_slots(
            W,
            T=T,
            d_in_pad=d_in_pad,
            patch_mask=patch_mask,
            d_out_mask=validated_d_out_mask,
            dist_var_by_patch=dist_var_by_patch,
            dist_patch_by_patch=dist_patch_by_patch,
            dist_var_pooled=dist_var_pooled,
            return_debug_info=True
        )
        decode_outputs = self._decode_from_latent_slots(
            latents,
            dist_patch_by_patch=dist_patch_by_patch,
            patch_mask=patch_mask,
            d_in_mask=validated_d_in_mask,
            d_out_mask=validated_d_out_mask,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            encoder_patch_tokens=encode_debug["encoder_patch_tokens"],
            T=T,
            return_direction_pre_norms=return_direction_pre_norms,
            disable_z_shortcut=disable_z_shortcut,
        )
        if return_direction_pre_norms:
            W_hat, mu, logvar, pred_dirs, direction_pre_norms = decode_outputs
        else:
            W_hat, mu, logvar, pred_dirs = decode_outputs

        if squeeze_batch:
            outputs = (
                W_hat.squeeze(0),
                mu.squeeze(0),
                logvar.squeeze(0),
                pred_dirs.squeeze(0),
            )
            if return_direction_pre_norms:
                return outputs + (direction_pre_norms,)
            return outputs

        outputs = (W_hat, mu, logvar, pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs


__all__ = [
    'BigWeightVAEForwardMixin',
]
