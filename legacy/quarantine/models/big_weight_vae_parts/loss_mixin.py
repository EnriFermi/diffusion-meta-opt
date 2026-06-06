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

class BigWeightVAELossMixin:
    @staticmethod
    def patch_structure_loss(
        W: torch.Tensor,
        W_hat: torch.Tensor,
        patch_size: int,
        eps: float = 1e-8,
        gamma: float = 0.5,
        lambda_dir: float = 1.0,
        lambda_scale: float = 0.25,
        lambda_rec: float = 0.5,
        lambda_rel: float = 0.1,
        huber_delta: float = 0.1,
        pred_dirs: torch.Tensor | None = None,
        d_in_mask: torch.Tensor | None = None,
        d_out_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if W.ndim == 2:
            W = W.unsqueeze(0)
            W_hat = W_hat.unsqueeze(0)
            if d_in_mask is not None:
                if d_in_mask.ndim != 1:
                    raise ValueError(f"d_in_mask must be [d_in] for unbatched W, got {tuple(d_in_mask.shape)}")
                d_in_mask = d_in_mask.unsqueeze(0)
            if d_out_mask is not None:
                if d_out_mask.ndim != 1:
                    raise ValueError(f"d_out_mask must be [d_out] for unbatched W, got {tuple(d_out_mask.shape)}")
                d_out_mask = d_out_mask.unsqueeze(0)

        B, d_in, d_out = W.shape
        p = int(patch_size)
        if d_out_mask is not None:
            if d_out_mask.ndim != 2 or tuple(d_out_mask.shape) != (B, d_out):
                raise ValueError(f"d_out_mask must be {(B, d_out)}, got {tuple(d_out_mask.shape)}")
            d_out_mask_bool = d_out_mask.to(device=W.device, dtype=torch.bool)
        else:
            d_out_mask_bool = None
        if d_in_mask is not None:
            if d_in_mask.ndim != 2 or tuple(d_in_mask.shape) != (B, d_in):
                raise ValueError(f"d_in_mask must be {(B, d_in)}, got {tuple(d_in_mask.shape)}")
            valid_d_in = d_in_mask.to(device=W.device, dtype=torch.long).sum(dim=1)
            valid_T = torch.div(valid_d_in, p, rounding_mode="floor")
            T = int(valid_T.max().item()) if valid_T.numel() > 0 else 0
            patch_mask = (
                torch.arange(T, device=W.device, dtype=torch.long).view(1, T) < valid_T.view(B, 1)
                if T > 0
                else None
            )
        else:
            valid_T = None
            T = d_in // p
            patch_mask = None

        if T <= 0:
            zero = W.new_zeros(())
            return zero, {"L_dir": zero, "L_scale": zero, "L_rec": zero, "L_rel": zero}

        used = T * p
        X = W[:, :used, :].transpose(1, 2).contiguous().view(B, d_out, T, p)
        X_hat = W_hat[:, :used, :].transpose(1, 2).contiguous().view(B, d_out, T, p)

        N = B * d_out
        X = X.reshape(N, T, p)
        X_hat = X_hat.reshape(N, T, p)
        patch_mask_flat = (
            patch_mask.unsqueeze(1).expand(B, d_out, T).reshape(N, T).to(device=W.device, dtype=W.dtype)
            if patch_mask is not None
            else None
        )
        if d_out_mask_bool is not None:
            d_out_mask_flat = d_out_mask_bool.reshape(N).to(device=W.device, dtype=W.dtype).unsqueeze(-1)
            if patch_mask_flat is None:
                patch_mask_flat = d_out_mask_flat.expand(-1, T)
            else:
                patch_mask_flat = patch_mask_flat * d_out_mask_flat

        r = X.norm(dim=-1)
        u = X / (r.unsqueeze(-1) + eps)
        log_r = torch.log(r + eps)

        use_dir = float(lambda_dir) != 0.0
        use_scale = float(lambda_scale) != 0.0
        use_rec = float(lambda_rec) != 0.0
        use_rel = float(lambda_rel) != 0.0

        zero = X.new_zeros(())
        L_dir = zero
        L_scale = zero
        L_rec = zero
        L_rel = zero

        r_hat: torch.Tensor | None = None
        u_hat: torch.Tensor | None = None
        log_r_hat: torch.Tensor | None = None

        def _ensure_r_hat() -> torch.Tensor:
            nonlocal r_hat
            if r_hat is None:
                r_hat = X_hat.norm(dim=-1)
            return r_hat

        def _ensure_u_hat() -> torch.Tensor:
            nonlocal u_hat
            if u_hat is None:
                current_r_hat = _ensure_r_hat()
                u_hat = X_hat / (current_r_hat.unsqueeze(-1) + eps)
            return u_hat

        def _ensure_log_r_hat() -> torch.Tensor:
            nonlocal log_r_hat
            if log_r_hat is None:
                log_r_hat = torch.log(_ensure_r_hat() + eps)
            return log_r_hat

        if use_dir:
            if pred_dirs is not None:
                u_hat_dir = pred_dirs[:, :, :T, :].reshape(N, T, p)
            else:
                u_hat_dir = _ensure_u_hat()

            cos = (u_hat_dir * u).sum(dim=-1)
            dir_loss = 1.0 - cos
            dir_weight = (r + eps).pow(float(gamma))
            if patch_mask_flat is not None:
                dir_weight = dir_weight * patch_mask_flat
            dir_weight = dir_weight / dir_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            L_dir = (dir_loss * dir_weight).sum(dim=1).mean()

        if use_scale:
            d = _ensure_log_r_hat() - log_r
            abs_d = d.abs()
            huber = torch.where(
                abs_d <= huber_delta,
                0.5 * d.pow(2),
                huber_delta * (abs_d - 0.5 * huber_delta),
            )
            if patch_mask_flat is None:
                L_scale = huber.mean()
            else:
                L_scale = (huber * patch_mask_flat).sum() / patch_mask_flat.sum().clamp_min(1.0)

        if use_rec:
            rec_num = (X_hat - X).pow(2).sum(dim=-1)
            rec_den = r.pow(2) + eps
            rec_loss = rec_num / rec_den
            if patch_mask_flat is None:
                L_rec = rec_loss.mean()
            else:
                L_rec = (rec_loss * patch_mask_flat).sum() / patch_mask_flat.sum().clamp_min(1.0)

        if use_rel:
            current_u_hat = _ensure_u_hat()
            G = torch.bmm(u, u.transpose(1, 2))
            G_hat = torch.bmm(current_u_hat, current_u_hat.transpose(1, 2))
            rel_sq = (G_hat - G).pow(2)
            if patch_mask_flat is None:
                L_rel = rel_sq.mean()
            else:
                pair_mask = patch_mask_flat.unsqueeze(1) * patch_mask_flat.unsqueeze(2)
                L_rel = (rel_sq * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

        total = lambda_dir * L_dir + lambda_scale * L_scale + lambda_rec * L_rec + lambda_rel * L_rel
        details = {
            "L_dir": L_dir.detach(),
            "L_scale": L_scale.detach(),
            "L_rec": L_rec.detach(),
            "L_rel": L_rel.detach(),
        }
        return total, details

    @staticmethod
    def operator_recon_loss(
        X: torch.Tensor,
        W: torch.Tensor,
        W_hat: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        d_in_mask: torch.Tensor | None = None,
        d_out_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = X @ W_hat if X.ndim == 2 else torch.matmul(X, W_hat)
        target = X @ W if X.ndim == 2 else torch.matmul(X, W)
        if X.ndim == 2:
            if x_mask is not None:
                if x_mask.ndim != 1 or int(x_mask.shape[0]) != int(X.shape[0]):
                    raise ValueError(f"x_mask must be [n] for unbatched X, got {tuple(x_mask.shape)}")
                sample_mask = x_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(0)
            else:
                sample_mask = torch.ones((1, int(X.shape[0])), device=pred.device, dtype=pred.dtype)
            if d_in_mask is not None:
                if d_in_mask.ndim != 1 or int(d_in_mask.shape[0]) != int(W.shape[0]):
                    raise ValueError(f"d_in_mask must be [d_in] for unbatched W, got {tuple(d_in_mask.shape)}")
                valid_d_in = d_in_mask.to(device=pred.device, dtype=pred.dtype).sum().view(1).clamp_min(1.0)
            else:
                valid_d_in = pred.new_full((1,), float(W.shape[0]))
            if d_out_mask is not None:
                if d_out_mask.ndim != 1 or int(d_out_mask.shape[0]) != int(W.shape[1]):
                    raise ValueError(f"d_out_mask must be [d_out] for unbatched W, got {tuple(d_out_mask.shape)}")
                output_mask = d_out_mask.to(device=pred.device, dtype=pred.dtype).view(1, -1)
                valid_d_out = output_mask.sum(dim=1).clamp_min(1.0)
            else:
                output_mask = torch.ones((1, int(pred.shape[-1])), device=pred.device, dtype=pred.dtype)
                valid_d_out = pred.new_full((1,), float(pred.shape[-1]))
            diff_sq = (pred - target).pow(2).unsqueeze(0) * sample_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        else:
            if x_mask is not None:
                if x_mask.ndim != 2 or tuple(x_mask.shape) != tuple(X.shape[:2]):
                    raise ValueError(f"x_mask must be [B,n] for batched X, got {tuple(x_mask.shape)} vs {tuple(X.shape[:2])}")
                sample_mask = x_mask.to(device=pred.device, dtype=pred.dtype)
            else:
                sample_mask = torch.ones(tuple(X.shape[:2]), device=pred.device, dtype=pred.dtype)
            if d_in_mask is not None:
                if d_in_mask.ndim != 2 or tuple(d_in_mask.shape) != (int(W.shape[0]), int(W.shape[1])):
                    raise ValueError(
                        f"d_in_mask must be [B,d_in] for batched W, got {tuple(d_in_mask.shape)} vs {(int(W.shape[0]), int(W.shape[1]))}"
                    )
                valid_d_in = d_in_mask.to(device=pred.device, dtype=pred.dtype).sum(dim=1).clamp_min(1.0)
            else:
                valid_d_in = pred.new_full((int(W.shape[0]),), float(W.shape[1]))
            if d_out_mask is not None:
                if d_out_mask.ndim != 2 or tuple(d_out_mask.shape) != (int(W.shape[0]), int(W.shape[2])):
                    raise ValueError(
                        f"d_out_mask must be [B,d_out] for batched W, got {tuple(d_out_mask.shape)} vs {(int(W.shape[0]), int(W.shape[2]))}"
                    )
                output_mask = d_out_mask.to(device=pred.device, dtype=pred.dtype)
                valid_d_out = output_mask.sum(dim=1).clamp_min(1.0)
            else:
                output_mask = torch.ones((int(W.shape[0]), int(pred.shape[-1])), device=pred.device, dtype=pred.dtype)
                valid_d_out = pred.new_full((int(W.shape[0]),), float(pred.shape[-1]))
            diff_sq = (pred - target).pow(2) * sample_mask.unsqueeze(-1) * output_mask.unsqueeze(1)

        denom = sample_mask.sum(dim=1).clamp_min(1.0) * valid_d_out
        mse = diff_sq.sum(dim=(1, 2)) / denom
        return torch.sqrt(mse / valid_d_in).mean()

    @staticmethod
    def operator_direction_scale_loss(
        X: torch.Tensor,
        W: torch.Tensor,
        W_hat: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        d_out_mask: torch.Tensor | None = None,
        eps: float = 1e-8,
        gamma: float = 0.5,
        huber_delta: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if X.ndim == 2:
            if W.ndim != 2 or W_hat.ndim != 2:
                raise ValueError(
                    "unbatched operator_direction_scale_loss expects X=[n,d_in], W=[d_in,d_out], "
                    f"W_hat=[d_in,d_out], got X={tuple(X.shape)} W={tuple(W.shape)} W_hat={tuple(W_hat.shape)}"
                )
            X = X.unsqueeze(0)
            W = W.unsqueeze(0)
            W_hat = W_hat.unsqueeze(0)
            squeeze_batch = True
            if x_mask is not None:
                if x_mask.ndim != 1:
                    raise ValueError(f"x_mask must be [n] for unbatched X, got {tuple(x_mask.shape)}")
                x_mask = x_mask.unsqueeze(0)
            if d_out_mask is not None:
                if d_out_mask.ndim != 1:
                    raise ValueError(f"d_out_mask must be [d_out] for unbatched W, got {tuple(d_out_mask.shape)}")
                d_out_mask = d_out_mask.unsqueeze(0)

        if X.ndim != 3 or W.ndim != 3 or W_hat.ndim != 3:
            raise ValueError(
                "operator_direction_scale_loss expects batched tensors X=[B,n,d_in], W=[B,d_in,d_out], "
                f"W_hat=[B,d_in,d_out], got X={tuple(X.shape)} W={tuple(W.shape)} W_hat={tuple(W_hat.shape)}"
            )
        if tuple(W.shape) != tuple(W_hat.shape):
            raise ValueError(f"W and W_hat shapes must match, got {tuple(W.shape)} vs {tuple(W_hat.shape)}")
        if int(X.shape[0]) != int(W.shape[0]) or int(X.shape[2]) != int(W.shape[1]):
            raise ValueError(
                "X/W shapes are inconsistent, expected X=[B,n,d_in] and W=[B,d_in,d_out], "
                f"got X={tuple(X.shape)} W={tuple(W.shape)}"
            )

        B, n, _ = X.shape
        d_out = int(W.shape[2])
        pred = torch.matmul(X, W_hat)
        target = torch.matmul(X, W)

        if x_mask is not None:
            if x_mask.ndim != 2 or tuple(x_mask.shape) != (B, n):
                prefix = "unbatched " if squeeze_batch else ""
                raise ValueError(f"{prefix}x_mask must be {(B, n)}, got {tuple(x_mask.shape)}")
            sample_mask = x_mask.to(device=pred.device, dtype=pred.dtype)
        else:
            sample_mask = torch.ones((B, n), device=pred.device, dtype=pred.dtype)

        if d_out_mask is not None:
            if d_out_mask.ndim != 2 or tuple(d_out_mask.shape) != (B, d_out):
                prefix = "unbatched " if squeeze_batch else ""
                raise ValueError(f"{prefix}d_out_mask must be {(B, d_out)}, got {tuple(d_out_mask.shape)}")
            output_mask = d_out_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(1)
            pred = pred * output_mask
            target = target * output_mask

        pred_norm = pred.norm(dim=-1)
        target_norm = target.norm(dim=-1)
        active = target_norm > float(eps)
        row_weight = sample_mask * active.to(device=pred.device, dtype=pred.dtype)

        dot = (pred * target).sum(dim=-1)
        cos = dot / (pred_norm * target_norm).clamp_min(float(eps))
        dir_loss = 1.0 - cos.clamp(min=-1.0, max=1.0)
        dir_weight = row_weight * (target_norm + float(eps)).pow(float(gamma))
        L_dir = (dir_loss * dir_weight).sum() / dir_weight.sum().clamp_min(1.0)

        d = torch.log(pred_norm + float(eps)) - torch.log(target_norm + float(eps))
        abs_d = d.abs()
        huber = torch.where(
            abs_d <= huber_delta,
            0.5 * d.pow(2),
            huber_delta * (abs_d - 0.5 * huber_delta),
        )
        L_scale = (huber * row_weight).sum() / row_weight.sum().clamp_min(1.0)
        return L_dir, L_scale

    @staticmethod
    def _channel_norm_over_sequence(x: torch.Tensor, eps: float) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,C], got {tuple(x.shape)}")
        if int(x.shape[1]) <= 1:
            return x
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + float(eps))

    @staticmethod
    def _validate_d_in_mask(
        d_in_mask: torch.Tensor | None,
        *,
        batch_size: int,
        d_in: int,
        device: torch.device,
    ) -> torch.Tensor:
        if d_in_mask is None:
            return torch.ones((batch_size, d_in), device=device, dtype=torch.bool)
        if d_in_mask.ndim != 2:
            raise ValueError(f"d_in_mask must be rank-2 [B, d_in], got {tuple(d_in_mask.shape)}")
        if tuple(d_in_mask.shape) != (batch_size, d_in):
            raise ValueError(f"d_in_mask shape must be {(batch_size, d_in)}, got {tuple(d_in_mask.shape)}")
        return d_in_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _validate_d_out_mask(
        d_out_mask: torch.Tensor | None,
        *,
        batch_size: int,
        d_out: int,
        device: torch.device,
    ) -> torch.Tensor:
        if d_out_mask is None:
            return torch.ones((batch_size, d_out), device=device, dtype=torch.bool)
        if d_out_mask.ndim != 2:
            raise ValueError(f"d_out_mask must be rank-2 [B, d_out], got {tuple(d_out_mask.shape)}")
        if tuple(d_out_mask.shape) != (batch_size, d_out):
            raise ValueError(f"d_out_mask shape must be {(batch_size, d_out)}, got {tuple(d_out_mask.shape)}")
        return d_out_mask.to(device=device, dtype=torch.bool)


__all__ = [
    'BigWeightVAELossMixin',
]
