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

class BigWeightVAELatentMixin:
    def _init_latent_sampling_heads(self) -> None:
        if self.to_mu is None:
            return
        with torch.no_grad():
            nn.init.eye_(self.to_mu.weight)
            self.to_mu.bias.zero_()
            if self.to_logvar is not None:
                self.to_logvar.weight.zero_()
                self.to_logvar.bias.zero_()

    def set_latent_sampling_gate(self, gate: float) -> None:
        gate_value = max(0.0, min(1.0, float(gate)))
        self.latent_sampling_gate.fill_(gate_value)

    def _latent_sampling_gate_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        return self.latent_sampling_gate.to(device=ref.device, dtype=ref.dtype).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_patch_tokenizer_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value in {"residual", "default", "legacy"}:
            return "residual"
        if value in {"conditioned_mlp", "patch_conditioned_mlp"}:
            return "conditioned_mlp"
        raise ValueError(
            "big_vae.patch_tokenizer_kind must be one of "
            "'residual', 'conditioned_mlp', "
            f"got {kind!r}"
        )

    @staticmethod
    def _normalize_distribution_encoder_conditioning_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value not in {"legacy", "token_adapter"}:
            raise ValueError(
                "big_vae.distribution_encoder_conditioning_kind must be one of "
                "'legacy', 'token_adapter', "
                f"got {kind!r}"
            )
        return value

    @staticmethod
    def _normalize_latent_prior_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value not in {"gaussian", "vamp"}:
            raise ValueError(
                "big_vae.latent_prior_kind must be one of "
                "'gaussian', 'vamp', "
                f"got {kind!r}"
            )
        return value

    @staticmethod
    def _normalize_decoder_query_conditioning_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value not in {"linear", "mlp"}:
            raise ValueError(
                "big_vae.decoder_query_conditioning_kind must be one of "
                "'linear', 'mlp', "
                f"got {kind!r}"
            )
        return value

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        mu_f = mu.to(dtype=torch.float32)
        logvar_f = logvar.to(dtype=torch.float32).clamp(-30.0, 20.0)
        # Keep beta independent of the chosen latent width.  The previous sum
        # made the effective regularization grow linearly with
        # num_latents*d_lat (12,288 coordinates in the production model).
        kl = 0.5 * torch.mean(torch.exp(logvar_f) + mu_f.pow(2) - 1.0 - logvar_f, dim=-1)
        return kl.mean()

    @staticmethod
    def _diag_gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        z_f = z.to(dtype=torch.float32)
        mu_f = mu.to(dtype=torch.float32)
        logvar_f = logvar.to(dtype=torch.float32).clamp(-30.0, 20.0)
        inv_var = torch.exp(-logvar_f)
        return -0.5 * (
            math.log(2.0 * math.pi)
            + logvar_f
            + (z_f - mu_f).pow(2) * inv_var
        ).sum(dim=-1)

    def _mu_head_input_z(self, latent_slots: torch.Tensor) -> torch.Tensor:
        if latent_slots.ndim != 3:
            raise ValueError(f"latent_slots must be [B, num_latents, d_lat], got {tuple(latent_slots.shape)}")
        flat = latent_slots.reshape(int(latent_slots.shape[0]), self.flat_lat_dim)
        if bool(self.cfg.big_vae.normalize_latent_slots_before_mu):
            return self.latent_norm(flat)
        return flat

    def _vamp_prior_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vamp_prior_base is None:
            raise RuntimeError("Vamp prior is not initialized for this model")
        if self.to_mu is None or self.to_logvar is None:
            raise RuntimeError("Vamp prior requires both to_mu and to_logvar heads")

        K = int(self.vamp_prior_base.shape[0])
        mu_input_z = self._mu_head_input_z(self.vamp_prior_base)
        mu_input_slots = mu_input_z.view(K, int(self.cfg.big_vae.num_latents), int(self.cfg.big_vae.d_lat))
        mu_slots = self.to_mu(mu_input_slots)
        logvar_min = float(self.cfg.big_vae.latent_sampling_logvar_min)
        logvar_max = float(self.cfg.big_vae.latent_sampling_logvar_max)
        logvar_slots = self.to_logvar(mu_input_slots).clamp(logvar_min, logvar_max)
        return mu_slots.reshape(K, self.z_dim), logvar_slots.reshape(K, self.z_dim)

    def latent_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not bool(self.cfg.big_vae.use_latent_sampling):
            return mu.new_zeros(())
        if self.latent_prior_kind == "gaussian":
            return self.kl_loss(mu, logvar)
        if self.latent_prior_kind != "vamp":
            raise RuntimeError(f"Unsupported latent prior kind: {self.latent_prior_kind!r}")

        mu_f = mu.to(dtype=torch.float32)
        logvar_f = logvar.to(dtype=torch.float32).clamp(-30.0, 20.0)
        if self.training:
            std_f = torch.exp(0.5 * logvar_f)
            z_sample = mu_f + torch.randn_like(std_f) * std_f
        else:
            z_sample = mu_f

        log_q = self._diag_gaussian_log_prob(z_sample, mu_f, logvar_f)
        prior_mu, prior_logvar = self._vamp_prior_params()
        prior_mu = prior_mu.to(device=z_sample.device, dtype=z_sample.dtype)
        prior_logvar = prior_logvar.to(device=z_sample.device, dtype=z_sample.dtype)
        component_log_probs = self._diag_gaussian_log_prob(
            z_sample.unsqueeze(1),
            prior_mu.unsqueeze(0),
            prior_logvar.unsqueeze(0),
        )
        log_p = torch.logsumexp(component_log_probs, dim=1) - math.log(float(prior_mu.shape[0]))
        return (log_q - log_p).mean()

    def _sample_latent_posterior(
        self,
        base_z: torch.Tensor,
        *,
        mu_input_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base_z.ndim != 2 or int(base_z.shape[1]) != self.z_dim:
            raise ValueError(f"base_z must be [B, {self.z_dim}], got {tuple(base_z.shape)}")
        if mu_input_z is None:
            mu_input_z = base_z
        if mu_input_z.ndim != 2 or tuple(mu_input_z.shape) != tuple(base_z.shape):
            raise ValueError(f"mu_input_z must match base_z shape {tuple(base_z.shape)}, got {tuple(mu_input_z.shape)}")

        B = int(base_z.shape[0])
        num_latents = int(self.cfg.big_vae.num_latents)
        d_lat = int(self.cfg.big_vae.d_lat)
        base_slots = base_z.view(B, num_latents, d_lat)
        mu_input_slots = mu_input_z.view(B, num_latents, d_lat)
        gate = self._latent_sampling_gate_tensor(base_z)
        use_latent_sampling = bool(self.cfg.big_vae.use_latent_sampling)
        use_encoder_mu_head = bool(self.cfg.big_vae.use_encoder_mu_head)

        if use_latent_sampling:
            if self.to_mu is None:
                raise RuntimeError("Latent sampling is enabled, but to_mu is not initialized")
            # The sampling schedule controls stochastic noise only.  It must
            # not change q(z|x), otherwise the reported KL describes a
            # different posterior at every gate value.
            mu_slots = self.to_mu(mu_input_slots)
        elif use_encoder_mu_head:
            if self.to_mu is None:
                raise RuntimeError("Encoder mu head is enabled, but to_mu is not initialized")
            mu_slots = (1.0 - gate) * base_slots + gate * self.to_mu(mu_input_slots)
        else:
            mu_slots = base_slots

        if not use_latent_sampling:
            mu_z = mu_slots.reshape(B, self.z_dim)
            return mu_z, mu_z, base_z.new_zeros(B, self.z_dim)
        if self.to_logvar is None:
            raise RuntimeError("Latent sampling is enabled, but to_logvar is not initialized")

        logvar_min = float(self.cfg.big_vae.latent_sampling_logvar_min)
        logvar_max = float(self.cfg.big_vae.latent_sampling_logvar_max)
        if logvar_min > logvar_max:
            raise ValueError(
                "big_vae.latent_sampling_logvar_min must be <= "
                "big_vae.latent_sampling_logvar_max"
            )
        raw_logvar_slots = self.to_logvar(mu_input_slots).clamp(logvar_min, logvar_max)
        posterior_std_slots = torch.exp(0.5 * raw_logvar_slots)

        min_std = max(0.0, float(self.cfg.big_vae.latent_sampling_min_std))
        if min_std > 0.0:
            min_std_tensor = base_z.new_tensor(min_std)
            posterior_std_slots = posterior_std_slots.clamp_min(min_std_tensor)
        logvar_slots = 2.0 * torch.log(
            posterior_std_slots.clamp_min(torch.finfo(posterior_std_slots.dtype).tiny)
        )

        if self.training:
            sampled_slots = mu_slots + gate * torch.randn_like(posterior_std_slots) * posterior_std_slots
        else:
            sampled_slots = mu_slots

        return sampled_slots.reshape(B, self.z_dim), mu_slots.reshape(B, self.z_dim), logvar_slots.reshape(B, self.z_dim)


__all__ = [
    'BigWeightVAELatentMixin',
]
