from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.weight_quantile_vae import DistributionConfig, InputDistributionEncodingModule, MiniPatchVAE, MiniVAEConfig


def build_distribution_config(cfg: DictConfig, section: str = "mini_model") -> DistributionConfig:
    model_cfg = cfg.get(section, {})
    dist_cfg = model_cfg.get("distribution", {})

    return DistributionConfig(
        k_s=int(dist_cfg.get("k_s", 16)),
        Kq=int(dist_cfg.get("Kq", 32)),
        d_var=int(dist_cfg.get("d_var", 128)),
        d_dist=int(dist_cfg.get("d_dist", 128)),
        num_var_attn_layers=int(dist_cfg.get("num_var_attn_layers", 2)),
        var_attn_heads=int(dist_cfg.get("var_attn_heads", 4)),
        dcn_num_cross_layers=int(dist_cfg.get("dcn_num_cross_layers", 3)),
        dcn_deep_hidden=int(dist_cfg.get("dcn_deep_hidden", 0)),
        dcn_deep_layers=int(dist_cfg.get("dcn_deep_layers", 0)),
        dropout=float(dist_cfg.get("dropout", 0.0)),
    )


def build_mini_vae_config(cfg: DictConfig, section: str = "mini_model") -> MiniVAEConfig:
    model_cfg = cfg.get(section, {})
    mini_cfg = model_cfg.get("mini_vae", {})

    return MiniVAEConfig(
        z_dim=int(mini_cfg.get("z_dim", 64)),
        d_e=int(mini_cfg.get("d_e", 128)),
        pos_dim=int(mini_cfg.get("pos_dim", 32)),
        num_attn_layers_encoder=int(mini_cfg.get("num_attn_layers_encoder", 2)),
        num_layers_decoder=int(mini_cfg.get("num_layers_decoder", 2)),
        decoder_bilinear_rank=int(mini_cfg.get("decoder_bilinear_rank", 0)),
        decoder_L_latents=int(mini_cfg.get("decoder_L_latents", 8)),
        decoder_use_dist_conditioning=bool(mini_cfg.get("decoder_use_dist_conditioning", True)),
        decoder_dist_mode=str(mini_cfg.get("decoder_dist_mode", "add")),
        n_heads=int(mini_cfg.get("n_heads", 4)),
        d_patch=int(mini_cfg.get("d_patch", 64)),
        dropout=float(mini_cfg.get("dropout", 0.0)),
    )


class MiniPatchTrainingModel(nn.Module):
    """Joint module for MiniPatchVAE pretraining with structure/behavior/contrastive objectives."""

    def __init__(self, distribution_cfg: DistributionConfig, mini_cfg: MiniVAEConfig) -> None:
        super().__init__()
        self.distribution_encoder = InputDistributionEncodingModule(distribution_cfg)
        self.mini_vae = MiniPatchVAE(d_var=distribution_cfg.d_var, cfg=mini_cfg, d_dist=distribution_cfg.d_dist)

    @staticmethod
    def normalize_patch_weights(
        w_patch: torch.Tensor,
        eps: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize patch weights per batch item using RMS scale.

        w_patch: [B, p]
        returns:
        - w_norm: [B, p]
        - scale: [B, 1]
        """
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        scale = w_patch.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(float(eps))
        w_norm = w_patch / scale
        return w_norm, scale

    @staticmethod
    def project_patch_weights_to_unit_sphere(
        w_patch: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Project patch weight vectors to unit L2 sphere per batch item.

        w_patch: [B, p]
        returns:
        - w_unit: [B, p], ||w_unit[i]||_2 ~= 1
        """
        if w_patch.ndim != 2:
            raise ValueError(f"w_patch must be [B, p], got {tuple(w_patch.shape)}")
        return F.normalize(w_patch, p=2, dim=1, eps=float(eps))

    @staticmethod
    def patch_behavioral_mse(
        X_patch: torch.Tensor,
        w_patch: torch.Tensor,
        w_hat: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        # X_patch: [B_p, n, p], w_patch/w_hat: [B_p, p]
        if X_patch.ndim != 3:
            raise ValueError(f"X_patch must be [B_p,n,p], got {tuple(X_patch.shape)}")
        # Normalize X_patch per patch-sample so behavioral loss is less scale-sensitive to input activations.
        x_scale = X_patch.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(float(eps))
        X_patch_norm = X_patch / x_scale
        y = torch.einsum("bnp,bp->bn", X_patch_norm, w_patch)
        y_hat = torch.einsum("bnp,bp->bn", X_patch_norm, w_hat)
        return F.mse_loss(y_hat, y)

    @staticmethod
    def nt_xent_loss(z_view1: torch.Tensor, z_view2: torch.Tensor, temperature: float) -> torch.Tensor:
        if z_view1.ndim != 2 or z_view2.ndim != 2:
            raise ValueError(
                f"NT-Xent expects rank-2 tensors, got {tuple(z_view1.shape)} and {tuple(z_view2.shape)}"
            )
        if z_view1.shape != z_view2.shape:
            raise ValueError(
                f"NT-Xent expects same shapes, got {tuple(z_view1.shape)} and {tuple(z_view2.shape)}"
            )

        batch = int(z_view1.shape[0])
        if batch < 2:
            return z_view1.new_zeros(())

        temp = max(float(temperature), 1e-6)
        reps = torch.cat(
            [
                F.normalize(z_view1, p=2, dim=-1),
                F.normalize(z_view2, p=2, dim=-1),
            ],
            dim=0,
        )  # [2B, d]
        logits = torch.matmul(reps, reps.transpose(0, 1)) / temp

        diag_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diag_mask, torch.finfo(logits.dtype).min)

        pos_idx = torch.arange(batch, device=logits.device, dtype=torch.long)
        pos_idx = torch.cat([pos_idx + batch, pos_idx], dim=0)  # [2B]

        log_prob = F.log_softmax(logits, dim=1)
        row_idx = torch.arange(2 * batch, device=logits.device, dtype=torch.long)
        return -log_prob[row_idx, pos_idx].mean()

    @staticmethod
    def function_preserving_linear_view(
        x_layer: torch.Tensor,
        W_layer: torch.Tensor,
        *,
        permute_inputs: bool,
        sign_flip_inputs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_layer: [n, d_in], W_layer: [d_in, d_out]
        if x_layer.ndim != 2 or W_layer.ndim != 2:
            raise ValueError(
                f"Expected x_layer=[n,d_in], W_layer=[d_in,d_out], got {tuple(x_layer.shape)} and {tuple(W_layer.shape)}"
            )
        if x_layer.shape[1] != W_layer.shape[0]:
            raise ValueError(f"Shape mismatch: x_layer={tuple(x_layer.shape)} W_layer={tuple(W_layer.shape)}")

        x_aug = x_layer
        W_aug = W_layer
        d_in = int(x_layer.shape[1])
        orig_to_view_idx = torch.arange(d_in, device=x_layer.device, dtype=torch.long)

        if permute_inputs and d_in > 1:
            perm = torch.randperm(d_in, device=x_layer.device)
            x_aug = x_aug.index_select(1, perm)
            W_aug = W_aug.index_select(0, perm)
            # orig_to_view_idx[i] gives the index in x_aug/W_aug for original input coordinate i.
            orig_to_view_idx = torch.argsort(perm)

        if sign_flip_inputs:
            sign_bits = torch.randint(0, 2, (d_in,), device=x_layer.device)
            signs = sign_bits.to(dtype=x_aug.dtype) * 2.0 - 1.0  # {-1, +1}
            x_aug = x_aug * signs.unsqueeze(0)
            W_aug = W_aug * signs.unsqueeze(1)

        return x_aug.contiguous(), W_aug.contiguous(), orig_to_view_idx

    @staticmethod
    def build_patch_batch_from_layer(
        x_layer: torch.Tensor,
        W_layer: torch.Tensor,
        patch_idx: torch.Tensor,
        out_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_layer: [n, d_in], W_layer: [d_in, d_out]
        # patch_idx: [B_p, p], out_idx: [B_p]
        if x_layer.ndim != 2 or W_layer.ndim != 2:
            raise ValueError(
                f"Expected x_layer=[n,d_in], W_layer=[d_in,d_out], got {tuple(x_layer.shape)} and {tuple(W_layer.shape)}"
            )
        if patch_idx.ndim != 2:
            raise ValueError(f"patch_idx must be [B_p,p], got {tuple(patch_idx.shape)}")
        if out_idx.ndim != 1:
            raise ValueError(f"out_idx must be [B_p], got {tuple(out_idx.shape)}")

        n, d_in = x_layer.shape
        d_in_w, d_out = W_layer.shape
        if d_in_w != d_in:
            raise ValueError(f"Shape mismatch: x_layer={tuple(x_layer.shape)} W_layer={tuple(W_layer.shape)}")

        batch_patches, _ = patch_idx.shape
        if out_idx.shape[0] != batch_patches:
            raise ValueError(
                f"out_idx length ({int(out_idx.shape[0])}) must match patch_idx batch ({batch_patches})"
            )

        patch_idx_safe = patch_idx.to(dtype=torch.long, device=x_layer.device).clamp(min=0, max=max(0, d_in - 1))
        out_idx_safe = out_idx.to(dtype=torch.long, device=x_layer.device).clamp(min=0, max=max(0, d_out - 1))

        X_full = x_layer.unsqueeze(0).expand(batch_patches, -1, -1).contiguous()
        X_patch = X_full.gather(dim=2, index=patch_idx_safe.unsqueeze(1).expand(-1, n, -1))

        W_col = W_layer.transpose(0, 1).index_select(0, out_idx_safe).contiguous()  # [B_p, d_in]
        w_patch = W_col.gather(dim=1, index=patch_idx_safe)
        return X_full, X_patch, w_patch

    def contrastive_loss(
        self,
        *,
        X_full: torch.Tensor,
        patch_idx: torch.Tensor,
        out_idx: torch.Tensor | None,
        W_full: torch.Tensor | None,
        alpha: float,
        temperature: float,
        permute_inputs: bool,
        sign_flip_inputs: bool,
    ) -> torch.Tensor:
        if float(alpha) <= 0.0:
            return X_full.new_zeros(())

        if W_full is None or out_idx is None:
            return X_full.new_zeros(())

        if X_full.ndim != 3:
            raise ValueError(f"X_full must be [B_p,n,d_in], got {tuple(X_full.shape)}")
        if W_full.ndim != 2:
            raise ValueError(f"W_full must be [d_in,d_out], got {tuple(W_full.shape)}")

        # sample_patch_batch currently replicates one layer input x across patch-batch.
        x_layer = X_full[0]

        d_in = int(W_full.shape[0])
        patch_idx_base = patch_idx.to(dtype=torch.long, device=x_layer.device).clamp(min=0, max=max(0, d_in - 1))

        x_view1, W_view1, orig_to_view1 = self.function_preserving_linear_view(
            x_layer=x_layer,
            W_layer=W_full,
            permute_inputs=permute_inputs,
            sign_flip_inputs=sign_flip_inputs,
        )
        x_view2, W_view2, orig_to_view2 = self.function_preserving_linear_view(
            x_layer=x_layer,
            W_layer=W_full,
            permute_inputs=permute_inputs,
            sign_flip_inputs=sign_flip_inputs,
        )
        patch_idx_view1 = orig_to_view1[patch_idx_base]
        patch_idx_view2 = orig_to_view2[patch_idx_base]

        X_full_view1, _, w_patch_view1 = self.build_patch_batch_from_layer(
            x_layer=x_view1,
            W_layer=W_view1,
            patch_idx=patch_idx_view1,
            out_idx=out_idx,
        )
        X_full_view2, _, w_patch_view2 = self.build_patch_batch_from_layer(
            x_layer=x_view2,
            W_layer=W_view2,
            patch_idx=patch_idx_view2,
            out_idx=out_idx,
        )

        w_patch_view1_norm, _ = self.normalize_patch_weights(
            w_patch_view1,
        )
        w_patch_view2_norm, _ = self.normalize_patch_weights(
            w_patch_view2,
        )
        w_patch_view1_unit = self.project_patch_weights_to_unit_sphere(w_patch_view1_norm)
        w_patch_view2_unit = self.project_patch_weights_to_unit_sphere(w_patch_view2_norm)

        dist_var_view1, _ = self.distribution_encoder(X=X_full_view1, patch_idx=patch_idx_view1)
        dist_var_view2, _ = self.distribution_encoder(X=X_full_view2, patch_idx=patch_idx_view2)

        mu_view1, _ = self.mini_vae.encode(w_patch=w_patch_view1_unit, dist_var_tokens=dist_var_view1)
        mu_view2, _ = self.mini_vae.encode(w_patch=w_patch_view2_unit, dist_var_tokens=dist_var_view2)
        return self.nt_xent_loss(mu_view1, mu_view2, temperature=temperature)

    def forward(
        self,
        X_full: torch.Tensor,
        X_patch: torch.Tensor,
        w_patch: torch.Tensor,
        patch_idx: torch.Tensor,
        kl_beta: float,
        alpha: float,
        beta: float,
        contrastive_temperature: float,
        contrastive_permute_inputs: bool,
        contrastive_sign_flip_inputs: bool,
        W_full: torch.Tensor | None = None,
        out_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # X_full: [B_p, n, d_in]
        # X_patch: [B_p, n, p]
        # w_patch: [B_p, p]
        # patch_idx: [B_p, p]
        w_patch_norm, _ = self.normalize_patch_weights(
            w_patch,
        )
        w_patch_unit = self.project_patch_weights_to_unit_sphere(w_patch_norm)
        dist_var_tokens, dist_patch_embed = self.distribution_encoder(X=X_full, patch_idx=patch_idx)
        w_hat_raw, mu, logvar, _ = self.mini_vae(
            w_patch=w_patch_unit,
            dist_var_tokens=dist_var_tokens,
            dist_patch_embed=dist_patch_embed,
        )
        w_hat_unit = self.project_patch_weights_to_unit_sphere(w_hat_raw)

        structural_loss = F.mse_loss(w_hat_unit, w_patch_unit)
        behavioral_loss = self.patch_behavioral_mse(X_patch=X_patch, w_patch=w_patch_unit, w_hat=w_hat_unit)
        recon_mix_loss = float(beta) * structural_loss + (1.0 - float(beta)) * behavioral_loss

        contrastive_loss = self.contrastive_loss(
            X_full=X_full,
            patch_idx=patch_idx,
            out_idx=out_idx,
            W_full=W_full,
            alpha=alpha,
            temperature=contrastive_temperature,
            permute_inputs=contrastive_permute_inputs,
            sign_flip_inputs=contrastive_sign_flip_inputs,
        )

        kl_loss = self.mini_vae.kl_loss(mu=mu, logvar=logvar)
        total_core_loss = float(alpha) * contrastive_loss + (1.0 - float(alpha)) * recon_mix_loss
        total_loss = total_core_loss + float(kl_beta) * kl_loss
        return total_loss, structural_loss, behavioral_loss, contrastive_loss, recon_mix_loss, kl_loss

    def ablation_losses(
        self,
        *,
        X_full: torch.Tensor,
        X_patch: torch.Tensor,
        w_patch: torch.Tensor,
        patch_idx: torch.Tensor,
        beta: float,
        kl_beta: float,
    ) -> dict[str, torch.Tensor]:
        """
        Auxiliary ablation losses for evaluating information content:
        1) decoder_random_latent_*: decode from random z ~ N(0, I).
        2) random_dist_*: full mini-VAE path with random distribution conditioning
           (both dist_var_tokens and dist_patch_embed) instead of encoder outputs.
        """
        w_patch_norm, _ = self.normalize_patch_weights(
            w_patch,
        )
        w_patch_unit = self.project_patch_weights_to_unit_sphere(w_patch_norm)
        dist_var_tokens, dist_patch_embed = self.distribution_encoder(X=X_full, patch_idx=patch_idx)
        mu_ref, _ = self.mini_vae.encode(w_patch=w_patch_unit, dist_var_tokens=dist_var_tokens)

        patch_size = int(w_patch.shape[1])
        beta_value = float(beta)
        kl_beta_value = float(kl_beta)

        # Decoder-only ablation: random latents.
        z_random = torch.randn_like(mu_ref)
        w_hat_rand_latent = self.mini_vae.decode(
            z=z_random,
            patch_size=patch_size,
            dist_patch_embed=dist_patch_embed,
        )
        w_hat_rand_latent_unit = self.project_patch_weights_to_unit_sphere(w_hat_rand_latent)
        structural_rand_latent = F.mse_loss(w_hat_rand_latent_unit, w_patch_unit)
        behavioral_rand_latent = self.patch_behavioral_mse(
            X_patch=X_patch,
            w_patch=w_patch_unit,
            w_hat=w_hat_rand_latent_unit,
        )
        recon_mix_rand_latent = beta_value * structural_rand_latent + (1.0 - beta_value) * behavioral_rand_latent

        # Distribution ablation: random tokens replace distribution encoder output.
        rand_dist_var_tokens = torch.randn_like(dist_var_tokens)
        rand_dist_patch_embed = torch.randn_like(dist_patch_embed)
        w_hat_rand_dist_raw, mu_rand_dist, logvar_rand_dist, _ = self.mini_vae(
            w_patch=w_patch_unit,
            dist_var_tokens=rand_dist_var_tokens,
            dist_patch_embed=rand_dist_patch_embed,
        )
        w_hat_rand_dist_unit = self.project_patch_weights_to_unit_sphere(w_hat_rand_dist_raw)
        structural_rand_dist = F.mse_loss(w_hat_rand_dist_unit, w_patch_unit)
        behavioral_rand_dist = self.patch_behavioral_mse(
            X_patch=X_patch,
            w_patch=w_patch_unit,
            w_hat=w_hat_rand_dist_unit,
        )
        recon_mix_rand_dist = beta_value * structural_rand_dist + (1.0 - beta_value) * behavioral_rand_dist
        kl_rand_dist = self.mini_vae.kl_loss(mu=mu_rand_dist, logvar=logvar_rand_dist)
        total_rand_dist = recon_mix_rand_dist + kl_beta_value * kl_rand_dist

        return {
            "decoder_random_latent_structural": structural_rand_latent,
            "decoder_random_latent_behavioral": behavioral_rand_latent,
            "decoder_random_latent_recon_mix": recon_mix_rand_latent,
            "random_dist_structural": structural_rand_dist,
            "random_dist_behavioral": behavioral_rand_dist,
            "random_dist_recon_mix": recon_mix_rand_dist,
            "random_dist_kl": kl_rand_dist,
            "random_dist_total": total_rand_dist,
        }
