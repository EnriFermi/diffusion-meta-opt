from __future__ import annotations

import torch
import torch.nn as nn

from models.big_weight_vae import LocalOutputSelfAttentionBlock, ModelConfig
from models.vae_shared import _decode_direction_and_logscale


class SimpleDirectBigWeightVAE(nn.Module):
    """
    Minimal BigVAE variant for debugging.

    Design goals:
    - no distribution conditioning
    - no latent bottleneck
    - no decoder cross-attention
    - predict directions directly from final encoder patch tokens

    The public forward API matches BigWeightVAE so the existing train loop can
    swap between implementations with a config flag.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        p = int(cfg.patch_size)
        d_model = int(cfg.big_vae.d_model)
        n_heads = int(cfg.big_vae.n_heads)
        num_enc_layers = max(1, int(cfg.big_vae.num_encoder_layers))
        dropout = float(cfg.big_vae.dropout)

        if d_model % n_heads != 0:
            raise ValueError(f"big_vae.d_model ({d_model}) must be divisible by big_vae.n_heads ({n_heads})")

        self.patch_tokenizer = nn.Linear(p, d_model)
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.encoder_layers = nn.ModuleList(
            [
                LocalOutputSelfAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_mult=float(cfg.big_vae.ffn_mult),
                    dropout=dropout,
                    self_attn_mode=cfg.big_vae.encoder.self_attn_mode,
                )
                for _ in range(num_enc_layers)
            ]
        )
        self.direction_head = nn.Linear(d_model, p)
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0

        self.dec_L_latents = int(cfg.big_vae.num_latents)
        self.z_dim = int(cfg.big_vae.num_latents * cfg.big_vae.d_lat)

    @staticmethod
    def _build_patch_geometry(
        d_in: int,
        patch_size: int,
    ) -> tuple[int, int]:
        T = (int(d_in) + int(patch_size) - 1) // int(patch_size)
        d_in_pad = T * int(patch_size)
        return T, d_in_pad

    def _encode_patch_tokens(
        self,
        W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if W.ndim != 3:
            raise ValueError(f"W must be rank-3 [B, d_in, d_out], got {tuple(W.shape)}")

        B, d_in, d_out = W.shape
        p = int(self.cfg.patch_size)
        T, d_in_pad = self._build_patch_geometry(d_in=d_in, patch_size=p)

        W_pad = W.new_zeros((B, d_in_pad, d_out))
        W_pad[:, :d_in, :] = W
        w_patches = W_pad.transpose(1, 2).contiguous().view(B, d_out, T, p)

        patch_tokens = self.patch_tokenizer(w_patches.reshape(B * d_out * T, p))
        patch_tokens = patch_tokens.view(B, d_out, T, -1)
        cls_tokens = self.cls_token.view(1, 1, 1, -1).expand(B, d_out, 1, -1)
        tokens_by_output = torch.cat([cls_tokens, patch_tokens], dim=2)

        for enc_layer in self.encoder_layers:
            local_in = tokens_by_output.view(B * d_out, T + 1, tokens_by_output.shape[-1])
            local_out = enc_layer(local_in)
            tokens_by_output = local_out.view(B, d_out, T + 1, tokens_by_output.shape[-1])

        encoder_patch_tokens = tokens_by_output[:, :, 1:, :]
        return tokens_by_output, encoder_patch_tokens, T, d_in_pad

    def _decode_direct_from_encoder_tokens(
        self,
        encoder_patch_tokens: torch.Tensor,
        *,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        return_direction_pre_norms: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if tuple(encoder_patch_tokens.shape[1:3]) != (d_out, T):
            raise ValueError(
                "encoder_patch_tokens must match decoder grid [d_out, T], got "
                f"{tuple(encoder_patch_tokens.shape)} vs expected (*, {d_out}, {T}, {encoder_patch_tokens.shape[-1]})"
            )

        B = int(encoder_patch_tokens.shape[0])
        p = int(self.cfg.patch_size)
        Q = d_out * T

        direct_hidden = encoder_patch_tokens.reshape(B, Q, encoder_patch_tokens.shape[-1])
        u_hat = self.direction_head(direct_hidden).reshape(B * Q, p)
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

        w_hat_patches = w_hat_flat.view(B, d_out, T, p)
        W_hat_pad = w_hat_patches.view(B, d_out, d_in_pad).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]

        if return_direction_pre_norms:
            return W_hat, pred_dirs, direction_pre_norms
        return W_hat, pred_dirs

    def forward_debug(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        *,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
        debug_decoder_kv_source: str = "latents",
        debug_query_hint: str = "none",
        debug_direct_from_encoder_tokens: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        del disable_z_shortcut, debug_decoder_kv_source, debug_query_hint, debug_direct_from_encoder_tokens

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
        B, d_in, d_out = W.shape
        Bx, _, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        tokens_by_output, encoder_patch_tokens, T, d_in_pad = self._encode_patch_tokens(W)
        decode_outputs = self._decode_direct_from_encoder_tokens(
            encoder_patch_tokens,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            T=T,
            return_direction_pre_norms=return_direction_pre_norms,
        )
        if return_direction_pre_norms:
            W_hat, pred_dirs, direction_pre_norms = decode_outputs
        else:
            W_hat, pred_dirs = decode_outputs

        z = W.new_zeros((B, self.z_dim))
        debug_patch_flat = encoder_patch_tokens.reshape(B, d_out * T, encoder_patch_tokens.shape[-1])
        debug_info = {
            "T": int(T),
            "d_in_pad": int(d_in_pad),
            "encoder_tokens_by_output": tokens_by_output,
            "encoder_cls_tokens": tokens_by_output[:, :, :1, :],
            "encoder_patch_tokens": encoder_patch_tokens,
            "debug_decoder_kv_source": "encoder_patch_tokens",
            "debug_query_hint": "none",
            "debug_direct_from_encoder_tokens": True,
            "decoder_queries_base": debug_patch_flat,
            "decoder_queries_init": debug_patch_flat,
            "decoder_query_pos_emb": torch.zeros_like(debug_patch_flat),
            "decoder_kv": None,
            "decoder_kv_pos_o": None,
            "decoder_kv_pos_t": None,
            "pred_dirs": pred_dirs,
        }

        if squeeze_batch:
            outputs = (
                W_hat.squeeze(0),
                z.squeeze(0),
                z.new_zeros(self.z_dim),
                pred_dirs.squeeze(0),
            )
            if return_direction_pre_norms:
                return outputs + (direction_pre_norms, debug_info)
            return outputs + (debug_info,)

        outputs = (W_hat, z, z.new_zeros(B, self.z_dim), pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms, debug_info)
        return outputs + (debug_info,)

    def forward(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        *,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        del disable_z_shortcut

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
        B, d_in, d_out = W.shape
        Bx, _, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        _tokens_by_output, encoder_patch_tokens, T, d_in_pad = self._encode_patch_tokens(W)
        decode_outputs = self._decode_direct_from_encoder_tokens(
            encoder_patch_tokens,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            T=T,
            return_direction_pre_norms=return_direction_pre_norms,
        )
        if return_direction_pre_norms:
            W_hat, pred_dirs, direction_pre_norms = decode_outputs
        else:
            W_hat, pred_dirs = decode_outputs

        z = W.new_zeros((B, self.z_dim))
        if squeeze_batch:
            outputs = (
                W_hat.squeeze(0),
                z.squeeze(0),
                z.new_zeros(self.z_dim),
                pred_dirs.squeeze(0),
            )
            if return_direction_pre_norms:
                return outputs + (direction_pre_norms,)
            return outputs

        outputs = (W_hat, z, z.new_zeros(B, self.z_dim), pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs


def build_weight_quantile_vae(model_cfg: ModelConfig) -> nn.Module:
    variant = str(getattr(model_cfg, "variant", "full")).strip().lower()
    if variant in {"full", "big_weight_vae", "default"}:
        from models.big_weight_vae import BigWeightVAE

        return BigWeightVAE(model_cfg)
    if variant in {"simple_direct", "simple", "direct"}:
        return SimpleDirectBigWeightVAE(model_cfg)
    raise ValueError(f"Unknown model.variant={variant!r}; expected 'full' or 'simple_direct'")


__all__ = [
    "SimpleDirectBigWeightVAE",
    "build_weight_quantile_vae",
]
