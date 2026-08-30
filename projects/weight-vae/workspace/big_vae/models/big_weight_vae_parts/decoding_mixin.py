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

    def _decode_v5_mandatory_bridge(
        self,
        *,
        q_tokens: torch.Tensor,
        latent_slot_values: torch.Tensor,
        q_pos_o: torch.Tensor,
        q_pos_t: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_mandatory_latent_bridge_v5 or self.mandatory_latent_bridge is None:
            raise RuntimeError("V5 mandatory latent bridge is unavailable")
        # X-conditioned q_tokens form routing weights only.  The returned
        # decoder state contains latent values and has no +q_tokens bypass.
        q_hidden = self.mandatory_latent_bridge(
            q_tokens,
            latent_slot_values,
            query_mask=query_mask,
        )
        if self.use_position_only_film_v6:
            if self.position_only_film_v6 is None:
                raise RuntimeError("V6 fixed position-only FiLM is unavailable")
            q_hidden = self.position_only_film_v6(
                q_hidden,
                pos_o=q_pos_o,
                pos_t=q_pos_t,
                query_mask=query_mask,
            )
        for dec_layer in self.decoder_layers:
            q_hidden = dec_layer(
                q_hidden,
                q_pos=q_pos_o,
                q_pos2=q_pos_t,
                q_mask=query_mask,
            )
        return q_hidden

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

        shortcut_disabled = bool(
            self.use_mandatory_latent_bridge_v5
            or disable_z_shortcut
            or self.cfg.big_vae.disable_z_shortcut
        )
        q_tokens = _apply_sequence_mask(q_tokens, query_mask)
        q_dir = _apply_sequence_mask(self.q_tokens_norm(q_tokens), query_mask)
        u_hat_attn = self.direction_head(q_dir)
        u_hat_attn = _apply_sequence_mask(u_hat_attn, query_mask)

        if shortcut_disabled:
            u_hat_shortcut = torch.zeros_like(u_hat_attn)
        else:
            if self.z_shortcut_proj is None or self.z_shortcut is None:
                raise RuntimeError("z shortcut modules are unavailable for this architecture")
            z_proj = self.z_shortcut_proj(z)
            z_exp = z_proj.unsqueeze(1).expand(B, Q, -1)
            shortcut_in = torch.cat([z_exp, q_pos_emb], dim=-1)
            u_hat_shortcut = self.z_shortcut(shortcut_in)
            u_hat_shortcut = _apply_sequence_mask(u_hat_shortcut, query_mask)

        u_hat = (u_hat_attn + u_hat_shortcut).reshape(B * Q, p)
        direction_pre_norms = u_hat.norm(dim=-1)
        scale_head_input = q_dir if self.use_latent_feedback_postnorm_v2 else q_tokens
        s_hat = _apply_sequence_mask(self.scale_head(scale_head_input), query_mask).reshape(B * Q)

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

    def _decode_v10_orthogonal_output(
        self,
        q_hidden: torch.Tensor,
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
    ) -> tuple[torch.Tensor, ...]:
        """Decode V10 as an exact row-space floor plus orthogonal residual."""

        if not self.use_orthogonal_complement_v10:
            raise RuntimeError("V10 orthogonal output is unavailable")
        encoder = self.orthogonal_complement_encoder_v10
        residual_head = self.v10_residual_head
        if encoder is None or residual_head is None:
            raise RuntimeError("V10 encoder/residual head is unavailable")
        B, Q, _ = q_hidden.shape
        p = int(self.cfg.patch_size)
        if Q != d_out * T or d_in_pad != T * p:
            raise ValueError("V10 decoder grid does not match d_out*T and T*patch_size")
        latent_slots = z.reshape(B, int(self.cfg.big_vae.num_latents), int(self.cfg.big_vae.d_lat))
        protected = latent_slots[..., : int(self.cfg.big_vae.v10_protected_dim)]

        query_mask = (
            patch_mask.unsqueeze(1).expand(-1, d_out, -1)
            & d_out_mask.unsqueeze(-1)
        ).reshape(B, Q)
        raw_u_native = _apply_sequence_mask(residual_head(q_hidden), query_mask)
        with torch.autocast(device_type=q_hidden.device.type, enabled=False):
            protected_routes = protected.reshape(B, encoder.basis.route_count, p).float()
            basis = encoder.basis(patch_mask, d_out_mask).float()
            floor = torch.einsum("brs,brp->bsp", basis, protected_routes)
            raw_u = raw_u_native.float()
            row_coefficients = torch.einsum("brs,bsp->brp", basis, raw_u)
            row_component = torch.einsum("brs,brp->bsp", basis, row_coefficients)
            residual = raw_u - row_component
            v_hat = floor + residual
        valid = query_mask.to(device=v_hat.device, dtype=v_hat.dtype).unsqueeze(-1)
        v_hat = v_hat * valid

        flat = v_hat.reshape(B * Q, p)
        direction_pre_norms = flat.norm(dim=-1)
        pred_dirs = (flat / (direction_pre_norms.unsqueeze(-1) + self.output_eps)).view(
            B, d_out, T, p
        )
        pred_dirs = pred_dirs * valid.view(B, d_out, T, 1)
        patches = v_hat.view(B, d_out, T, p)
        W_hat_pad = patches.reshape(B, d_out, d_in_pad).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]
        W_hat = W_hat * d_in_mask.to(device=W_hat.device, dtype=W_hat.dtype).unsqueeze(-1)
        W_hat = W_hat * d_out_mask.to(device=W_hat.device, dtype=W_hat.dtype).unsqueeze(1)
        outputs = (W_hat, z, pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs

    def _decode_v11_complement_output(
        self,
        q_hidden: torch.Tensor,
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
    ) -> tuple[torch.Tensor, ...]:
        """Decode V11 as the exact V10 floor plus a learned complement."""

        if not self.use_four_trunk_complement_v11:
            raise RuntimeError("V11 complement output is unavailable")
        encoder = self.four_trunk_complement_encoder_v11
        residual_head = self.v11_residual_head
        if encoder is None or residual_head is None:
            raise RuntimeError("V11 encoder/residual head is unavailable")
        batch, query_count, _ = q_hidden.shape
        patch_size = int(self.cfg.patch_size)
        if query_count != d_out * T or d_in_pad != T * patch_size:
            raise ValueError("V11 decoder grid does not match d_out*T and T*patch_size")
        latent_slots = z.reshape(
            batch,
            int(self.cfg.big_vae.num_latents),
            int(self.cfg.big_vae.d_lat),
        )
        protected = latent_slots[..., : int(self.cfg.big_vae.v10_protected_dim)]
        query_mask = (
            patch_mask.unsqueeze(1).expand(-1, d_out, -1)
            & d_out_mask.unsqueeze(-1)
        ).reshape(batch, query_count)
        raw_residual_native = _apply_sequence_mask(residual_head(q_hidden), query_mask)
        with torch.autocast(device_type=q_hidden.device.type, enabled=False):
            protected_routes = protected.reshape(
                batch, encoder.basis.route_count, patch_size
            ).float()
            basis = encoder.basis(patch_mask, d_out_mask).float()
            floor = torch.einsum("brs,brp->bsp", basis, protected_routes)
            raw_residual = raw_residual_native.float()
            row_coefficients = torch.einsum("brs,bsp->brp", basis, raw_residual)
            row_component = torch.einsum("brs,brp->bsp", basis, row_coefficients)
            residual = raw_residual - row_component
            output_patches = floor + residual
        valid = query_mask.to(
            device=output_patches.device, dtype=output_patches.dtype
        ).unsqueeze(-1)
        output_patches = output_patches * valid
        flat = output_patches.reshape(batch * query_count, patch_size)
        direction_pre_norms = flat.norm(dim=-1)
        pred_dirs = (
            flat / (direction_pre_norms.unsqueeze(-1) + self.output_eps)
        ).view(batch, d_out, T, patch_size)
        pred_dirs = pred_dirs * valid.view(batch, d_out, T, 1)
        W_hat_pad = output_patches.view(batch, d_out, T, patch_size).reshape(
            batch, d_out, d_in_pad
        ).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]
        W_hat = W_hat * d_in_mask.to(
            device=W_hat.device, dtype=W_hat.dtype
        ).unsqueeze(-1)
        W_hat = W_hat * d_out_mask.to(
            device=W_hat.device, dtype=W_hat.dtype
        ).unsqueeze(1)
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

        raw_latent_z = latent_slots.reshape(B, self.flat_lat_dim)
        base_z = (
            self.latent_norm(raw_latent_z)
            if bool(self.cfg.big_vae.normalize_latent_slots_before_mu)
            else raw_latent_z
        )
        mu_input_z = self._mu_head_input_z(latent_slots)
        decoder_z, mu, logvar = self._sample_latent_posterior(base_z, mu_input_z=mu_input_z)
        latent_slot_values = decoder_z.view(B, num_latents, d_lat)
        lat = (
            None
            if self.use_mandatory_latent_bridge_v5
            else self.latent_to_decoder(latent_slot_values)
        )

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

        if self.use_mandatory_latent_bridge_v5 and (
            self._normalize_debug_decoder_kv_source(debug_decoder_kv_source) != "latents"
            or self._normalize_debug_query_hint(debug_query_hint) != "none"
            or bool(debug_direct_from_encoder_tokens)
        ):
            raise ValueError("V5 mandatory bridge does not allow decoder bypass debug overrides")

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
            if self.use_mandatory_latent_bridge_v5:
                bridge_latent_slot_values = (
                    latent_slot_values[..., int(self.cfg.big_vae.v10_protected_dim) :]
                    if self.use_orthogonal_complement_v10
                    or self.use_four_trunk_complement_v11
                    else latent_slot_values
                )
                decoder_kv = bridge_latent_slot_values
                kv_pos_o = None
                kv_pos_t = None
                kv_mask = None
                q_hidden = self._decode_v5_mandatory_bridge(
                    q_tokens=q_tokens,
                    latent_slot_values=bridge_latent_slot_values,
                    q_pos_o=q_pos_o,
                    q_pos_t=q_pos_t,
                    query_mask=query_mask,
                )
            else:
                if lat is None:
                    raise RuntimeError("legacy decoder latent projection is missing")
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
                for decoder_layer_idx, dec_layer in enumerate(self.decoder_layers):
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
                    if self.decoder_post_norms is not None:
                        q_hidden = _apply_sequence_mask(
                            self.decoder_post_norms[decoder_layer_idx](q_hidden),
                            query_mask,
                        )
            if self.use_four_trunk_complement_v11:
                outputs = self._decode_v11_complement_output(
                    q_hidden,
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
            elif self.use_orthogonal_complement_v10:
                outputs = self._decode_v10_orthogonal_output(
                    q_hidden,
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
            else:
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

        latent_slot_values = z.view(
            B,
            int(self.cfg.big_vae.num_latents),
            int(self.cfg.big_vae.d_lat),
        )
        lat = (
            None
            if self.use_mandatory_latent_bridge_v5
            else self.latent_to_decoder(latent_slot_values)
        )
        q_tokens_base, q_pos_emb, q_pos_o, q_pos_t = self._build_decoder_query_state(
            batch_size=B,
            dist_patch_by_patch=dist_patch_by_patch,
            d_out=d_out,
            T=T,
        )
        query_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(B, d_out * T)
        q_tokens_base = _apply_sequence_mask(q_tokens_base, query_mask)
        if self.use_mandatory_latent_bridge_v5:
            bridge_latent_slot_values = (
                latent_slot_values[..., int(self.cfg.big_vae.v10_protected_dim) :]
                if self.use_orthogonal_complement_v10
                or self.use_four_trunk_complement_v11
                else latent_slot_values
            )
            q_hidden = self._decode_v5_mandatory_bridge(
                q_tokens=q_tokens_base,
                latent_slot_values=bridge_latent_slot_values,
                q_pos_o=q_pos_o,
                q_pos_t=q_pos_t,
                query_mask=query_mask,
            )
        else:
            if lat is None:
                raise RuntimeError("legacy decoder latent projection is missing")
            q_hidden = q_tokens_base
            decoder_kv, kv_pos_o, kv_pos_t, kv_mask = self._build_decoder_kv_state(
                lat=lat,
                encoder_patch_tokens=None,
                patch_mask=patch_mask,
                d_out_mask=d_out_mask,
                d_out=d_out,
                T=T,
                debug_decoder_kv_source="latents",
            )

            for decoder_layer_idx, dec_layer in enumerate(self.decoder_layers):
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
                if self.decoder_post_norms is not None:
                    q_hidden = _apply_sequence_mask(
                        self.decoder_post_norms[decoder_layer_idx](q_hidden),
                        query_mask,
                    )

        if self.use_four_trunk_complement_v11:
            outputs = self._decode_v11_complement_output(
                q_hidden,
                z=z,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
                patch_mask=patch_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                return_direction_pre_norms=return_direction_pre_norms,
            )
        elif self.use_orthogonal_complement_v10:
            outputs = self._decode_v10_orthogonal_output(
                q_hidden,
                z=z,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
                patch_mask=patch_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                return_direction_pre_norms=return_direction_pre_norms,
            )
        else:
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
