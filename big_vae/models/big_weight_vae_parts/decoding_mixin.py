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

class BigWeightVAEDecodingMixin:
    @staticmethod
    def _normalize_debug_decoder_kv_source(source: str) -> str:
        value = str(source).strip().lower()
        if value not in {"latents", "encoder_patch_tokens"}:
            raise ValueError(
                "debug_decoder_kv_source must be 'latents' or 'encoder_patch_tokens', "
                f"got {source!r}"
            )
        return value

    @staticmethod
    def _normalize_debug_query_hint(source: str) -> str:
        value = str(source).strip().lower()
        if value not in {"none", "aligned_encoder_token"}:
            raise ValueError(
                "debug_query_hint must be 'none' or 'aligned_encoder_token', "
                f"got {source!r}"
            )
        return value

    def _build_decoder_query_state(
        self,
        *,
        batch_size: int,
        dist_patch_by_patch: torch.Tensor | None,
        d_out: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_model = int(self.cfg.big_vae.d_model)
        if self.use_distribution_encoder:
            if dist_patch_by_patch is None:
                raise ValueError("dist_patch_by_patch is required when distribution encoder is enabled")
            B = int(dist_patch_by_patch.shape[0])
            device = dist_patch_by_patch.device
        else:
            B = int(batch_size)
            device = self.cls_token.device

        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")

        pos_dim = self.cfg.big_vae.pos_fourier_dim
        pos_o = sinusoidal_embedding(o_grid.to(torch.float32), pos_dim)
        pos_t = sinusoidal_embedding(t_grid.to(torch.float32), pos_dim)
        pos_ot = torch.cat([pos_o, pos_t], dim=-1)

        q_base = self.pos_proj(pos_ot)
        q_pos_emb = self.query_pos_proj(q_base)
        q_base_expanded = q_base.unsqueeze(0).expand(B, -1, -1, -1)
        if self.use_distribution_encoder:
            assert dist_patch_by_patch is not None
            dist_patch_expanded = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1)
            q_inputs = torch.cat([q_base_expanded, dist_patch_expanded], dim=-1)
        else:
            q_inputs = q_base_expanded
        q_tokens = self.query_proj(q_inputs).reshape(B, d_out * T, d_model)
        q_pos_emb_flat = q_pos_emb.reshape(1, d_out * T, d_model).expand(B, -1, -1)
        q_pos_o = _make_rope_positions(
            o_grid.flatten(),
            axis_size=d_out,
            coord_kind=self.rope_2d_coord_kind,
        )
        q_pos_t = _make_rope_positions(
            t_grid.flatten(),
            axis_size=T,
            coord_kind=self.rope_2d_coord_kind,
        )
        return q_tokens, q_pos_emb_flat, q_pos_o, q_pos_t

    def _apply_debug_query_hint(
        self,
        q_tokens: torch.Tensor,
        *,
        encoder_patch_tokens: torch.Tensor | None,
        debug_query_hint: str,
    ) -> torch.Tensor:
        hint_source = self._normalize_debug_query_hint(debug_query_hint)
        if hint_source == "none":
            return q_tokens
        if encoder_patch_tokens is None:
            raise ValueError("encoder_patch_tokens are required when debug_query_hint='aligned_encoder_token'")

        B, Q, d_model = q_tokens.shape
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if int(encoder_patch_tokens.shape[0]) != B or int(encoder_patch_tokens.shape[3]) != d_model:
            raise ValueError(
                "encoder_patch_tokens batch/model dims must match q_tokens, got "
                f"{tuple(encoder_patch_tokens.shape)} vs {tuple(q_tokens.shape)}"
            )

        encoder_patch_flat = encoder_patch_tokens.reshape(B, -1, d_model)
        if int(encoder_patch_flat.shape[1]) != Q:
            raise ValueError(
                "encoder_patch_tokens flattened length must match decoder query length, got "
                f"{tuple(encoder_patch_flat.shape)} vs {tuple(q_tokens.shape)}"
            )
        return q_tokens + self.debug_query_hint_proj(encoder_patch_flat)

    def _build_decoder_kv_state(
        self,
        *,
        lat: torch.Tensor,
        encoder_patch_tokens: torch.Tensor | None,
        patch_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        d_out: int,
        T: int,
        debug_decoder_kv_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        kv_source = self._normalize_debug_decoder_kv_source(debug_decoder_kv_source)
        if kv_source == "latents":
            kv = lat
            kv_pos = _make_rope_axis_positions(
                int(lat.shape[1]),
                device=lat.device,
                coord_kind=self.rope_2d_coord_kind,
            )
            return kv, kv_pos, kv_pos, None

        if encoder_patch_tokens is None:
            raise ValueError("encoder_patch_tokens are required when debug_decoder_kv_source='encoder_patch_tokens'")
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if tuple(encoder_patch_tokens.shape[1:3]) != (d_out, T):
            raise ValueError(
                "encoder_patch_tokens must match decoder grid [d_out, T], got "
                f"{tuple(encoder_patch_tokens.shape)} vs expected (*, {d_out}, {T}, {self.cfg.big_vae.d_model})"
            )

        device = encoder_patch_tokens.device
        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")
        kv = encoder_patch_tokens.reshape(encoder_patch_tokens.shape[0], d_out * T, encoder_patch_tokens.shape[-1])
        kv_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(
            encoder_patch_tokens.shape[0],
            d_out * T,
        )
        return (
            kv,
            _make_rope_positions(
                o_grid.flatten(),
                axis_size=d_out,
                coord_kind=self.rope_2d_coord_kind,
            ),
            _make_rope_positions(
                t_grid.flatten(),
                axis_size=T,
                coord_kind=self.rope_2d_coord_kind,
            ),
            kv_mask,
        )

    def _decode_query_tokens_to_output(
        self,
        q_tokens: torch.Tensor,
        *,
        z: torch.Tensor,
        q_pos_emb: torch.Tensor,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        patch_mask: torch.Tensor,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Q, d_model = q_tokens.shape
        p = self.cfg.patch_size
        expected_d_in_pad = T * p
        if d_in_pad != expected_d_in_pad:
            raise ValueError(f"d_in_pad must equal T*patch_size={expected_d_in_pad}, got {d_in_pad}")
        if patch_mask.ndim != 2 or tuple(patch_mask.shape) != (B, T):
            raise ValueError(f"patch_mask must be {(B, T)}, got {tuple(patch_mask.shape)}")
        if d_in_mask.ndim != 2 or tuple(d_in_mask.shape) != (B, d_in):
            raise ValueError(f"d_in_mask must be {(B, d_in)}, got {tuple(d_in_mask.shape)}")
        if d_out_mask.ndim != 2 or tuple(d_out_mask.shape) != (B, d_out):
            raise ValueError(f"d_out_mask must be {(B, d_out)}, got {tuple(d_out_mask.shape)}")
        query_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(B, Q)

        shortcut_disabled = bool(disable_z_shortcut or self.cfg.big_vae.disable_z_shortcut)
        q_tokens = _apply_sequence_mask(q_tokens, query_mask)
        q_dir = _apply_sequence_mask(self.q_tokens_norm(q_tokens), query_mask)
        u_hat_attn = self.direction_head(q_dir)
        u_hat_attn = _apply_sequence_mask(u_hat_attn, query_mask)

        if shortcut_disabled:
            u_hat_shortcut = torch.zeros_like(u_hat_attn)
        else:
            z_proj = self.z_shortcut_proj(z)
            z_exp = z_proj.unsqueeze(1).expand(B, Q, -1)
            shortcut_in = torch.cat([z_exp, q_pos_emb], dim=-1)
            u_hat_shortcut = self.z_shortcut(shortcut_in)
            u_hat_shortcut = _apply_sequence_mask(u_hat_shortcut, query_mask)

        u_hat = (u_hat_attn + u_hat_shortcut).reshape(B * Q, p)
        direction_pre_norms = u_hat.norm(dim=-1)
        s_hat = _apply_sequence_mask(self.scale_head(q_tokens), query_mask).reshape(B * Q)

        w_hat_flat = _decode_direction_and_logscale(
            u_hat=u_hat,
            s_hat=s_hat,
            eps=self.output_eps,
            s_min=self.output_s_min,
            s_max=self.output_s_max,
        )

        u_hat_norm = u_hat / (u_hat.norm(dim=-1, keepdim=True) + self.output_eps)
        pred_dirs = u_hat_norm.view(B, d_out, T, p)
        pred_dirs = pred_dirs * patch_mask.to(device=pred_dirs.device, dtype=pred_dirs.dtype).unsqueeze(1).unsqueeze(-1)
        pred_dirs = pred_dirs * d_out_mask.to(device=pred_dirs.device, dtype=pred_dirs.dtype).unsqueeze(-1).unsqueeze(-1)

        w_hat_patches = w_hat_flat.view(B, d_out, T, p)
        W_hat_pad = w_hat_patches.view(B, d_out, d_in_pad).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]
        W_hat = W_hat * d_in_mask.to(device=W_hat.device, dtype=W_hat.dtype).unsqueeze(-1)
        W_hat = W_hat * d_out_mask.to(device=W_hat.device, dtype=W_hat.dtype).unsqueeze(1)

        outputs = (W_hat, z, pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs

    def _decode_direct_from_encoder_patch_tokens(
        self,
        encoder_patch_tokens: torch.Tensor,
        *,
        z: torch.Tensor,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        patch_mask: torch.Tensor,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        return_direction_pre_norms: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if tuple(encoder_patch_tokens.shape[1:3]) != (d_out, T):
            raise ValueError(
                "encoder_patch_tokens must match decoder grid [d_out, T], got "
                f"{tuple(encoder_patch_tokens.shape)} vs expected (*, {d_out}, {T}, {self.cfg.big_vae.d_model})"
            )

        B = int(encoder_patch_tokens.shape[0])
        Q = d_out * T
        p = self.cfg.patch_size
        query_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(B, Q)
        direct_hidden = _apply_sequence_mask(
            encoder_patch_tokens.reshape(B, Q, encoder_patch_tokens.shape[-1]),
            query_mask,
        )
        u_hat = self.debug_encoder_direct_direction_head(direct_hidden).reshape(B * Q, p)
        direction_pre_norms = u_hat.norm(dim=-1)
        s_hat = torch.zeros(B * Q, device=u_hat.device, dtype=u_hat.dtype)

        w_hat_flat = _decode_direction_and_logscale(
            u_hat=u_hat,
            s_hat=s_hat,
            eps=self.output_eps,
            s_min=self.output_s_min,
            s_max=self.output_s_max,
        )

        u_hat_norm = u_hat / (u_hat.norm(dim=-1, keepdim=True) + self.output_eps)
        pred_dirs = u_hat_norm.view(B, d_out, T, p)
        pred_dirs = pred_dirs * patch_mask.to(device=pred_dirs.device, dtype=pred_dirs.dtype).unsqueeze(1).unsqueeze(-1)
        pred_dirs = pred_dirs * d_out_mask.to(device=pred_dirs.device, dtype=pred_dirs.dtype).unsqueeze(-1).unsqueeze(-1)

        w_hat_patches = w_hat_flat.view(B, d_out, T, p)
        W_hat_pad = w_hat_patches.view(B, d_out, d_in_pad).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]
        W_hat = W_hat * d_in_mask.to(device=W_hat.device, dtype=W_hat.dtype).unsqueeze(-1)
        W_hat = W_hat * d_out_mask.to(device=W_hat.device, dtype=W_hat.dtype).unsqueeze(1)

        outputs = (W_hat, z, pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs

    def _decode_from_latent_slots(
        self,
        latent_slots: torch.Tensor,
        *,
        dist_patch_by_patch: torch.Tensor | None,
        patch_mask: torch.Tensor,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
        encoder_patch_tokens: torch.Tensor | None = None,
        debug_decoder_kv_source: str = "latents",
        debug_query_hint: str = "none",
        debug_direct_from_encoder_tokens: bool = False,
        return_debug_info: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if latent_slots.ndim != 3:
            raise ValueError(f"latent_slots must be [B, num_latents, d_lat], got {tuple(latent_slots.shape)}")

        B, num_latents, d_lat = latent_slots.shape
        expected_num_latents = int(self.cfg.big_vae.num_latents)
        expected_d_lat = int(self.cfg.big_vae.d_lat)
        if num_latents != expected_num_latents or d_lat != expected_d_lat:
            raise ValueError(
                "latent_slots shape mismatch: "
                f"got {(B, num_latents, d_lat)}, expected (*, {expected_num_latents}, {expected_d_lat})"
            )
        if self.use_distribution_encoder:
            if dist_patch_by_patch is None:
                raise ValueError("dist_patch_by_patch is required when distribution encoder is enabled")
            if tuple(dist_patch_by_patch.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_patch_by_patch must be [B, T, d_dist], got "
                    f"{tuple(dist_patch_by_patch.shape)} for expected {(B, T, self.cfg.distribution.d_dist)}"
                )
        elif dist_patch_by_patch is not None:
            raise ValueError("dist_patch_by_patch must be None when distribution encoder is disabled")
        if patch_mask.ndim != 2 or tuple(patch_mask.shape) != (B, T):
            raise ValueError(f"patch_mask must be {(B, T)}, got {tuple(patch_mask.shape)}")
        if d_in_mask.ndim != 2 or tuple(d_in_mask.shape) != (B, d_in):
            raise ValueError(f"d_in_mask must be {(B, d_in)}, got {tuple(d_in_mask.shape)}")
        if d_out_mask.ndim != 2 or tuple(d_out_mask.shape) != (B, d_out):
            raise ValueError(f"d_out_mask must be {(B, d_out)}, got {tuple(d_out_mask.shape)}")

        p = self.cfg.patch_size
        expected_d_in_pad = T * p
        if d_in_pad != expected_d_in_pad:
            raise ValueError(f"d_in_pad must equal T*patch_size={expected_d_in_pad}, got {d_in_pad}")

        base_z = self.latent_norm(latent_slots.reshape(B, self.flat_lat_dim))
        mu_input_z = self._mu_head_input_z(latent_slots)
        decoder_z, mu, logvar = self._sample_latent_posterior(base_z, mu_input_z=mu_input_z)
        lat = self.latent_to_decoder(decoder_z.view(B, num_latents, d_lat))

        q_tokens_base, q_pos_emb, q_pos_o, q_pos_t = self._build_decoder_query_state(
            batch_size=B,
            dist_patch_by_patch=dist_patch_by_patch,
            d_out=d_out,
            T=T,
        )
        query_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(B, d_out * T)
        q_tokens_base = _apply_sequence_mask(q_tokens_base, query_mask)
        q_tokens = self._apply_debug_query_hint(
            q_tokens_base,
            encoder_patch_tokens=encoder_patch_tokens,
            debug_query_hint=debug_query_hint,
        )
        q_tokens = _apply_sequence_mask(q_tokens, query_mask)

        #DEBUG
        # debug_direct_from_encoder_tokens = True
        if bool(debug_direct_from_encoder_tokens):
            if encoder_patch_tokens is None:
                raise ValueError("encoder_patch_tokens are required when debug_direct_from_encoder_tokens=True")
            outputs = self._decode_direct_from_encoder_patch_tokens(
                encoder_patch_tokens,
                z=decoder_z,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
                patch_mask=patch_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                return_direction_pre_norms=return_direction_pre_norms,
            )
            decoder_kv = None
            kv_pos_o = None
            kv_pos_t = None
            kv_mask = None
        else:
            decoder_kv, kv_pos_o, kv_pos_t, kv_mask = self._build_decoder_kv_state(
                lat=lat,
                encoder_patch_tokens=encoder_patch_tokens,
                patch_mask=patch_mask,
                d_out_mask=d_out_mask,
                d_out=d_out,
                T=T,
                debug_decoder_kv_source=debug_decoder_kv_source,
            )
            q_hidden = q_tokens
            for dec_layer in self.decoder_layers:
                q_hidden = dec_layer(
                    q=q_hidden,
                    kv=decoder_kv,
                    q_pos=q_pos_o,
                    kv_pos=kv_pos_o,
                    q_pos2=q_pos_t,
                    kv_pos2=kv_pos_t,
                    q_mask=query_mask,
                    kv_mask=kv_mask,
                )
            outputs = self._decode_query_tokens_to_output(
                q_hidden,
                z=decoder_z,
                q_pos_emb=q_pos_emb,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
                patch_mask=patch_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                return_direction_pre_norms=return_direction_pre_norms,
                disable_z_shortcut=disable_z_shortcut,
            )

        if return_direction_pre_norms:
            W_hat, _decoder_z_out, pred_dirs, direction_pre_norms = outputs
        else:
            W_hat, _decoder_z_out, pred_dirs = outputs
        outputs_with_posterior = (W_hat, mu, logvar, pred_dirs)

        if return_debug_info:
            debug_info: dict[str, object] = {
                "debug_patch_tokenizer_kind": self.patch_tokenizer_kind,
                "debug_encoder_conditioning_kind": self.distribution_encoder_conditioning_kind,
                "debug_decoder_kv_source": self._normalize_debug_decoder_kv_source(debug_decoder_kv_source),
                "debug_query_hint": self._normalize_debug_query_hint(debug_query_hint),
                "debug_direct_from_encoder_tokens": bool(debug_direct_from_encoder_tokens),
                "latent_sampling_gate": self._latent_sampling_gate_tensor(mu).detach(),
                "latent_base_z": base_z,
                "latent_decoder_z": decoder_z,
                "decoder_queries_base": q_tokens_base,
                "decoder_queries_init": q_tokens,
                "decoder_query_pos_emb": q_pos_emb,
                "decoder_kv": decoder_kv,
                "decoder_kv_pos_o": kv_pos_o,
                "decoder_kv_pos_t": kv_pos_t,
                "decoder_query_mask": query_mask,
                "decoder_kv_mask": kv_mask,
                "pred_dirs": pred_dirs,
            }
            if return_direction_pre_norms:
                return outputs_with_posterior + (direction_pre_norms, debug_info)
            return outputs_with_posterior + (debug_info,)
        if return_direction_pre_norms:
            return outputs_with_posterior + (direction_pre_norms,)
        return outputs_with_posterior

    def _decode_from_decoder_latent(
        self,
        decoder_z: torch.Tensor,
        *,
        dist_patch_by_patch: torch.Tensor | None,
        patch_mask: torch.Tensor,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if decoder_z.ndim == 3:
            B, num_latents, d_lat = decoder_z.shape
            expected_shape = (int(self.cfg.big_vae.num_latents), int(self.cfg.big_vae.d_lat))
            if (num_latents, d_lat) != expected_shape:
                raise ValueError(
                    "decoder_z token shape mismatch: "
                    f"got {(B, num_latents, d_lat)}, expected (*, {expected_shape[0]}, {expected_shape[1]})"
                )
            z = decoder_z.reshape(B, self.flat_lat_dim)
        elif decoder_z.ndim == 2:
            B = int(decoder_z.shape[0])
            if int(decoder_z.shape[1]) != int(self.flat_lat_dim):
                raise ValueError(
                    f"decoder_z flat shape must be [B,{self.flat_lat_dim}], got {tuple(decoder_z.shape)}"
                )
            z = decoder_z
        else:
            raise ValueError(f"decoder_z must be [B,z_dim] or [B,num_latents,d_lat], got {tuple(decoder_z.shape)}")

        if self.use_distribution_encoder:
            if dist_patch_by_patch is None:
                raise ValueError("dist_patch_by_patch is required when distribution encoder is enabled")
            if tuple(dist_patch_by_patch.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_patch_by_patch must be [B, T, d_dist], got "
                    f"{tuple(dist_patch_by_patch.shape)} for expected {(B, T, self.cfg.distribution.d_dist)}"
                )
        elif dist_patch_by_patch is not None:
            raise ValueError("dist_patch_by_patch must be None when distribution encoder is disabled")
        if patch_mask.ndim != 2 or tuple(patch_mask.shape) != (B, T):
            raise ValueError(f"patch_mask must be {(B, T)}, got {tuple(patch_mask.shape)}")
        if d_in_mask.ndim != 2 or tuple(d_in_mask.shape) != (B, d_in):
            raise ValueError(f"d_in_mask must be {(B, d_in)}, got {tuple(d_in_mask.shape)}")
        if d_out_mask.ndim != 2 or tuple(d_out_mask.shape) != (B, d_out):
            raise ValueError(f"d_out_mask must be {(B, d_out)}, got {tuple(d_out_mask.shape)}")

        expected_d_in_pad = int(T) * int(self.cfg.patch_size)
        if int(d_in_pad) != expected_d_in_pad:
            raise ValueError(f"d_in_pad must equal T*patch_size={expected_d_in_pad}, got {d_in_pad}")

        lat = self.latent_to_decoder(z.view(B, int(self.cfg.big_vae.num_latents), int(self.cfg.big_vae.d_lat)))
        q_tokens_base, q_pos_emb, q_pos_o, q_pos_t = self._build_decoder_query_state(
            batch_size=B,
            dist_patch_by_patch=dist_patch_by_patch,
            d_out=d_out,
            T=T,
        )
        query_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(B, d_out * T)
        q_hidden = _apply_sequence_mask(q_tokens_base, query_mask)

        for dec_layer in self.decoder_layers:
            q_hidden = dec_layer(
                q=q_hidden,
                kv=lat,
                q_pos=q_pos_o,
                kv_pos=torch.arange(lat.shape[1], device=lat.device, dtype=torch.float32),
                q_pos2=q_pos_t,
                kv_pos2=torch.arange(lat.shape[1], device=lat.device, dtype=torch.float32),
                q_mask=query_mask,
                kv_mask=None,
            )

        outputs = self._decode_query_tokens_to_output(
            q_hidden,
            z=z,
            q_pos_emb=q_pos_emb,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            T=T,
            patch_mask=patch_mask,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            return_direction_pre_norms=return_direction_pre_norms,
            disable_z_shortcut=disable_z_shortcut,
        )
        if return_direction_pre_norms:
            W_hat, _decoder_z_out, pred_dirs, direction_pre_norms = outputs
            logvar = torch.zeros_like(z)
            return W_hat, z, logvar, pred_dirs, direction_pre_norms

        W_hat, _decoder_z_out, pred_dirs = outputs
        logvar = torch.zeros_like(z)
        return W_hat, z, logvar, pred_dirs


__all__ = [
    'BigWeightVAEDecodingMixin',
]
