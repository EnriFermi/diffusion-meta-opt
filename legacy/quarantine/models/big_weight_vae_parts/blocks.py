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

class LocalOutputSelfAttentionBlock(nn.Module):
    """Local attention within one output-column token group, with RoPE on patch positions."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float, self_attn_mode: str) -> None:
        super().__init__()
        if self_attn_mode not in {"full", "cls_only"}:
            raise ValueError(f"encoder.self_attn_mode must be 'full' or 'cls_only', got {self_attn_mode}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.self_attn_mode = self_attn_mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_prob_dropout_p = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        hidden = max(1, int(d_model * ffn_mult))
        self.ffn = MLP(d_model, hidden, d_model, dropout=dropout)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, _ = tokens.shape
        h = self.norm_attn(tokens)
        pos = torch.arange(S, device=tokens.device, dtype=torch.float32)
        attn_mask = (
            _key_padding_to_attn_bias(token_mask.to(device=tokens.device, dtype=torch.bool), dtype=h.dtype)
            if token_mask is not None
            else None
        )

        q_all = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k_all = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v_all = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if self.self_attn_mode == "full":
            attn_out = _rope_attention(
                q=q_all,
                k=k_all,
                v=v_all,
                q_pos=pos,
                k_pos=pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=attn_mask,
            )
        else:
            cls_out = _rope_attention(
                q=q_all[:, :, 0:1, :],
                k=k_all,
                v=v_all,
                q_pos=pos[0:1],
                k_pos=pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=attn_mask,
            )
            patch_out = _rope_attention(
                q=q_all[:, :, 1:, :],
                k=k_all[:, :, 0:1, :],
                v=v_all[:, :, 0:1, :],
                q_pos=pos[1:],
                k_pos=pos[0:1],
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
            attn_out = torch.cat([cls_out, patch_out], dim=2)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        tokens = tokens + self.dropout(self.out_proj(attn_out))
        tokens = _apply_sequence_mask(tokens, token_mask)
        tokens = tokens + self.dropout(self.ffn(self.norm_ffn(tokens)))
        return _apply_sequence_mask(tokens, token_mask)


class TokenConditioningAdapter(nn.Module):
    """Token-wise residual conditioning for encoder patch tokens."""

    def __init__(
        self,
        *,
        d_model: int,
        d_y: int,
        dropout: float,
        d_hidden: int | None = None,
        d_gate: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_hidden) if d_hidden is not None else int(4 * d_model)
        gate_dim = int(d_gate) if d_gate is not None else int(d_model)

        self.y_proj = nn.Linear(d_y, d_model)
        self.h_norm = nn.LayerNorm(d_model)
        self.c_norm = nn.LayerNorm(d_model)

        self.mix_h = nn.Linear(d_model, hidden_dim)
        self.mix_c = nn.Linear(d_model, hidden_dim)
        self.mix_out = nn.Linear(hidden_dim, d_model)

        self.gate_h = nn.Linear(d_model, gate_dim)
        self.gate_c = nn.Linear(d_model, gate_dim)
        self.gate_out = nn.Linear(gate_dim, d_model)

        self.alpha = nn.Parameter(torch.full((1,), 1e-3))
        self.conditioning_dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.mix_out.bias)
        nn.init.constant_(self.gate_out.bias, -2.0)

    def forward(
        self,
        h: torch.Tensor,
        y_ctx: torch.Tensor,
        map_idx: torch.Tensor,
    ) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must be [B, T_x, d_model], got {tuple(h.shape)}")
        if y_ctx.ndim != 3:
            raise ValueError(f"y_ctx must be [B, T_y, d_y], got {tuple(y_ctx.shape)}")
        if map_idx.ndim != 2:
            raise ValueError(f"map_idx must be [B, T_x], got {tuple(map_idx.shape)}")
        if int(h.shape[0]) != int(y_ctx.shape[0]) or int(h.shape[0]) != int(map_idx.shape[0]):
            raise ValueError(
                "Batch size mismatch between h, y_ctx, and map_idx: "
                f"{tuple(h.shape)}, {tuple(y_ctx.shape)}, {tuple(map_idx.shape)}"
            )
        if int(h.shape[1]) != int(map_idx.shape[1]):
            raise ValueError(
                f"map_idx token length must match h token length, got {tuple(map_idx.shape)} vs {tuple(h.shape)}"
            )

        gather_idx = map_idx.to(device=y_ctx.device, dtype=torch.long).clamp(min=0, max=max(0, int(y_ctx.shape[1]) - 1))
        y_match = y_ctx.gather(dim=1, index=gather_idx.unsqueeze(-1).expand(-1, -1, int(y_ctx.shape[2])))

        c = self.y_proj(y_match)
        c = self.conditioning_dropout(c)

        h_n = self.h_norm(h)
        c_n = self.c_norm(c)

        mixed = F.gelu(self.mix_h(h_n) + self.mix_c(c_n))
        residual = self.mix_out(mixed)

        gate_in = F.gelu(self.gate_h(h_n) + self.gate_c(c_n))
        gate = torch.sigmoid(self.gate_out(gate_in))
        return h + self.alpha * gate * residual


class LatentEncoderLayer(nn.Module):
    """
    One big-encoder block:
    1) local per-output attention over [CLS_o, patches_o]
    2) Perceiver Resampler over the resulting tokens
    """

    def __init__(
        self,
        d_model: int,
        d_lat: int,
        n_heads: int,
        ffn_mult: float,
        dropout: float,
        self_attn_mode: str,
        use_rope_2d: bool = False,
        rope_2d_coord_kind: str = "normalized_center",
    ) -> None:
        super().__init__()
        if d_lat % n_heads != 0:
            raise ValueError(f"d_lat ({d_lat}) must be divisible by n_heads ({n_heads})")
        self.rope_2d_coord_kind = _normalize_rope_2d_coord_kind(rope_2d_coord_kind)

        self.local_block = LocalOutputSelfAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            self_attn_mode=self_attn_mode,
        )
        self.perceiver_block = PerceiverResamplerBlock(
            d_latent=d_lat,
            d_token=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope_2d=use_rope_2d,
        )

    def forward(
        self,
        tokens_by_output: torch.Tensor,
        latents: torch.Tensor,
        cross_attend_only_cls: bool,
        patch_conditioner: TokenConditioningAdapter | None = None,
        patch_conditioning_ctx: torch.Tensor | None = None,
        patch_conditioning_map_idx: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        output_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, d_out, L_local, d_model = tokens_by_output.shape

        local_in = tokens_by_output.reshape(B * d_out, L_local, d_model)
        local_mask = token_valid_mask.reshape(B * d_out, L_local) if token_valid_mask is not None else None
        local_out = self.local_block(local_in, token_mask=local_mask)
        tokens = local_out.reshape(B, d_out, L_local, d_model)
        if patch_conditioner is not None:
            if patch_conditioning_ctx is None or patch_conditioning_map_idx is None:
                raise ValueError("patch conditioning context and map_idx are required when patch_conditioner is set")
            patch_tokens = tokens[:, :, 1:, :].reshape(B * d_out, L_local - 1, d_model)
            patch_tokens = patch_conditioner(
                h=patch_tokens,
                y_ctx=patch_conditioning_ctx,
                map_idx=patch_conditioning_map_idx,
            )
            if local_mask is not None:
                patch_tokens = _apply_sequence_mask(patch_tokens, local_mask[:, 1:])
            tokens = torch.cat(
                [
                    tokens[:, :, :1, :],
                    patch_tokens.reshape(B, d_out, L_local - 1, d_model),
                ],
                dim=2,
            )

        device = tokens.device
        if cross_attend_only_cls:
            kv = tokens[:, :, 0, :]
            token_pos_o = _make_rope_axis_positions(
                d_out,
                device=device,
                coord_kind=self.rope_2d_coord_kind,
            )
            token_pos_t = _make_rope_positions(
                torch.zeros(d_out, device=device, dtype=torch.float32),
                axis_size=L_local,
                coord_kind=self.rope_2d_coord_kind,
            )
            kv_mask = output_valid_mask
        else:
            kv = tokens.reshape(B, d_out * L_local, d_model)
            token_pos_o = _make_rope_positions(
                torch.arange(d_out, device=device, dtype=torch.float32).repeat_interleave(L_local),
                axis_size=d_out,
                coord_kind=self.rope_2d_coord_kind,
            )
            token_pos_t = _make_rope_positions(
                torch.arange(L_local, device=device, dtype=torch.float32).repeat(d_out),
                axis_size=L_local,
                coord_kind=self.rope_2d_coord_kind,
            )
            kv_mask = None if token_valid_mask is None else token_valid_mask.reshape(B, d_out * L_local)
            if kv_mask is not None and output_valid_mask is not None:
                kv_mask = kv_mask & output_valid_mask.unsqueeze(-1).expand(-1, -1, L_local).reshape(B, d_out * L_local)

        L = latents.shape[1]
        latent_pos = (torch.arange(L, device=device, dtype=torch.float32) + 0.5) / max(float(L), 1.0)
        latents = self.perceiver_block(
            latents=latents,
            tokens=kv,
            latent_pos=latent_pos,
            token_pos=token_pos_o,
            token_pos2=token_pos_t,
            token_mask=kv_mask,
        )
        return tokens, latents


class DecoderCrossBlock(nn.Module):
    """Decoder block with query-to-latent cross-attention only."""

    def __init__(self, d_model: int, d_lat: int, n_heads: int, ffn_mult: float, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_lat)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=d_lat,
            vdim=d_lat,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = max(1, int(d_model * ffn_mult))
        self.ffn = MLP(d_model, hidden, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens: torch.Tensor, z_latents: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(z_latents)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q_tokens = q_tokens + self.dropout(cross_out)
        q_tokens = q_tokens + self.dropout(self.ffn(self.norm_ffn(q_tokens)))
        return q_tokens

__all__ = [
    'LocalOutputSelfAttentionBlock',
    'TokenConditioningAdapter',
    'LatentEncoderLayer',
    'DecoderCrossBlock',
]
