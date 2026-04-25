from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from models.layer_latent_diffusion_prior import (
    LayerLatentDiffusionPrior,
    build_layer_latent_diffusion_prior_config,
)
from models.weight_quantile_vae import (
    BigVAEConfig,
    BigWeightVAE,
    DistributionConfig,
    EncoderConfig,
    MiniVAEConfig,
    ModelConfig,
    build_weight_quantile_vae,
)


def _strip_state_prefix(name: str) -> str:
    out = str(name)
    changed = True
    while changed:
        changed = False
        if out.startswith("module."):
            out = out[len("module.") :]
            changed = True
        if out.startswith("_orig_mod."):
            out = out[len("_orig_mod.") :]
            changed = True
    return out


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        normalized[_strip_state_prefix(str(raw_key))] = value
    return normalized


def build_big_vae_model_cfg(raw_cfg: Mapping[str, Any]) -> ModelConfig:
    model_cfg = raw_cfg.get("model", raw_cfg)
    if not isinstance(model_cfg, Mapping):
        raise TypeError("checkpoint config must expose a mapping at 'model'")
    dist_cfg = model_cfg.get("distribution", {})
    mini_cfg = model_cfg.get("mini_vae", {})
    big_cfg = model_cfg.get("big_vae", {})
    enc_cfg = big_cfg.get("encoder", {}) if isinstance(big_cfg, Mapping) else {}
    if not isinstance(dist_cfg, Mapping) or not isinstance(mini_cfg, Mapping) or not isinstance(big_cfg, Mapping):
        raise TypeError("checkpoint model config sections must be mappings")

    return ModelConfig(
        patch_size=int(model_cfg.get("patch_size", 16)),
        beta=float(model_cfg.get("beta", 1e-3)),
        variant=str(model_cfg.get("variant", "full")),
        distribution=DistributionConfig(
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
            use_covariance=bool(dist_cfg.get("use_covariance", True)),
            patch_size_for_cov=int(dist_cfg.get("patch_size_for_cov", int(model_cfg.get("patch_size", 16)))),
        ),
        mini_vae=MiniVAEConfig(
            z_dim=int(mini_cfg.get("z_dim", 64)),
            d_e=int(mini_cfg.get("d_e", 128)),
            encoder_latent_dim=int(mini_cfg.get("encoder_latent_dim", 0)),
            pos_lat_dim=int(mini_cfg.get("pos_lat_dim", 0)),
            pos_dim=int(mini_cfg.get("pos_dim", 64)),
            num_attn_layers_encoder=int(mini_cfg.get("num_attn_layers_encoder", 2)),
            num_layers_decoder=int(mini_cfg.get("num_layers_decoder", 2)),
            decoder_bilinear_rank=int(mini_cfg.get("decoder_bilinear_rank", 0)),
            stub_resampler_d_model=int(mini_cfg.get("stub_resampler_d_model", 0)),
            mlp_stub_hidden_dim=int(mini_cfg.get("mlp_stub_hidden_dim", 256)),
            init_style=str(mini_cfg.get("init_style", "llm")),
            n_heads=int(mini_cfg.get("n_heads", 4)),
            d_patch=int(mini_cfg.get("d_patch", 64)),
            dropout=float(mini_cfg.get("dropout", 0.0)),
        ),
        big_vae=BigVAEConfig(
            d_model=int(big_cfg.get("d_model", 256)),
            d_lat=int(big_cfg.get("d_lat", 256)),
            num_latents=int(big_cfg.get("num_latents", 32)),
            num_encoder_layers=int(big_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(big_cfg.get("num_decoder_layers", 4)),
            n_heads=int(big_cfg.get("n_heads", 8)),
            ffn_mult=float(big_cfg.get("ffn_mult", 4.0)),
            dropout=float(big_cfg.get("dropout", 0.0)),
            pos_fourier_dim=int(big_cfg.get("pos_fourier_dim", 64)),
            use_latent_sampling=bool(big_cfg.get("use_latent_sampling", True)),
            latent_sampling_min_std=float(big_cfg.get("latent_sampling_min_std", 1e-4)),
            latent_sampling_logvar_min=float(big_cfg.get("latent_sampling_logvar_min", -20.0)),
            latent_sampling_logvar_max=float(big_cfg.get("latent_sampling_logvar_max", 10.0)),
            disable_z_shortcut=bool(big_cfg.get("disable_z_shortcut", False)),
            disable_distribution_encoder=bool(big_cfg.get("disable_distribution_encoder", False)),
            patch_tokenizer_kind=str(big_cfg.get("patch_tokenizer_kind", "residual")),
            distribution_encoder_conditioning_kind=str(big_cfg.get("distribution_encoder_conditioning_kind", "legacy")),
            encoder=EncoderConfig(
                self_attn_mode=str(enc_cfg.get("self_attn_mode", "full")),
                cross_attend_only_cls=bool(enc_cfg.get("cross_attend_only_cls", True)),
            ),
        ),
    )


def load_frozen_big_vae_from_checkpoint(checkpoint_path: str | Path, *, device: torch.device) -> BigWeightVAE:
    path = Path(str(checkpoint_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"BigVAE checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"BigVAE checkpoint must contain a mapping payload, got {type(payload)!r}")
    raw_cfg = payload.get("config")
    if not isinstance(raw_cfg, Mapping):
        raise KeyError(f"BigVAE checkpoint has no mapping 'config': {path}")
    state = payload.get("model_state", payload.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError(f"BigVAE checkpoint has no mapping model_state/state_dict: {path}")

    model_cfg = build_big_vae_model_cfg(raw_cfg)
    model = build_weight_quantile_vae(model_cfg)
    if not isinstance(model, BigWeightVAE):
        raise TypeError(
            "Expected full BigWeightVAE when loading frozen BigVAE checkpoint, "
            f"got {type(model).__name__}"
        )
    missing, unexpected = model.load_state_dict(_normalize_state_dict_keys(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "BigVAE checkpoint state_dict mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_frozen_layer_latent_diffusion_prior(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> LayerLatentDiffusionPrior:
    path = Path(str(checkpoint_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Layer latent diffusion prior checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Diffusion prior checkpoint must contain a mapping payload, got {type(payload)!r}")
    raw_cfg = payload.get("config")
    if not isinstance(raw_cfg, Mapping):
        raise KeyError(f"Diffusion prior checkpoint has no mapping 'config': {path}")
    model_cfg_raw = raw_cfg.get("model", raw_cfg)
    if not isinstance(model_cfg_raw, Mapping):
        raise TypeError("Diffusion prior checkpoint config.model must be a mapping")
    prior_cfg_raw = model_cfg_raw.get("latent_diffusion_prior", model_cfg_raw)
    if not isinstance(prior_cfg_raw, Mapping):
        raise TypeError("Diffusion prior checkpoint config must contain model.latent_diffusion_prior mapping")
    state = payload.get("model_state", payload.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError(f"Diffusion prior checkpoint has no mapping model_state/state_dict: {path}")

    prior_cfg = build_layer_latent_diffusion_prior_config(prior_cfg_raw)
    model = LayerLatentDiffusionPrior(prior_cfg)
    missing, unexpected = model.load_state_dict(_normalize_state_dict_keys(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Diffusion prior checkpoint state_dict mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def encode_big_vae_layer_batch(
    big_vae: BigWeightVAE,
    *,
    W: torch.Tensor,
    X: torch.Tensor,
    x_mask: torch.Tensor | None = None,
    d_in_mask: torch.Tensor | None = None,
    d_out_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int | None]:
    if W.ndim not in {2, 3}:
        raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
    if X.ndim not in {2, 3}:
        raise ValueError(f"X must be rank-2 or rank-3, got {tuple(X.shape)}")
    squeeze_batch = W.ndim == 2
    if squeeze_batch != (X.ndim == 2):
        raise ValueError(f"W and X must be both batched or both unbatched, got {tuple(W.shape)} and {tuple(X.shape)}")

    if squeeze_batch:
        W = W.unsqueeze(0)
        X = X.unsqueeze(0)
        if x_mask is not None:
            x_mask = x_mask.unsqueeze(0)
        if d_in_mask is not None:
            d_in_mask = d_in_mask.unsqueeze(0)
        if d_out_mask is not None:
            d_out_mask = d_out_mask.unsqueeze(0)

    batch, d_in, d_out = W.shape
    if tuple(X.shape[:1]) != (batch,) or int(X.shape[2]) != d_in:
        raise ValueError(f"X must match W on [B,d_in], got W={tuple(W.shape)} X={tuple(X.shape)}")

    validated_d_in_mask = big_vae._validate_d_in_mask(d_in_mask, batch_size=batch, d_in=d_in, device=W.device)
    validated_d_out_mask = big_vae._validate_d_out_mask(d_out_mask, batch_size=batch, d_out=d_out, device=W.device)
    T, d_in_pad, patch_mask, structural_patch_mask, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled = (
        big_vae._encode_distribution_context(
            X,
            x_mask=x_mask,
            d_in_mask=validated_d_in_mask,
        )
    )
    latent_slots = big_vae._encode_latent_slots(
        W,
        T=T,
        d_in_pad=d_in_pad,
        patch_mask=patch_mask,
        d_out_mask=validated_d_out_mask,
        dist_var_by_patch=dist_var_by_patch,
        dist_patch_by_patch=dist_patch_by_patch,
        dist_var_pooled=dist_var_pooled,
    )
    if not isinstance(latent_slots, torch.Tensor):
        raise TypeError(f"Expected tensor latent_slots, got {type(latent_slots)!r}")
    base_z = big_vae.latent_norm(latent_slots.reshape(batch, -1))
    with torch.no_grad():
        decoder_z, latent_mu, latent_logvar = big_vae._sample_latent_posterior(base_z)

    return {
        "latent_slots": latent_slots.squeeze(0) if squeeze_batch else latent_slots,
        "base_z": base_z.squeeze(0) if squeeze_batch else base_z,
        "decoder_z": decoder_z.squeeze(0) if squeeze_batch else decoder_z,
        "latent_mu": latent_mu.squeeze(0) if squeeze_batch else latent_mu,
        "latent_logvar": latent_logvar.squeeze(0) if squeeze_batch else latent_logvar,
        "cond_patch": dist_patch_by_patch.squeeze(0) if squeeze_batch and dist_patch_by_patch is not None else dist_patch_by_patch,
        "patch_mask": patch_mask.squeeze(0) if squeeze_batch else patch_mask,
        "structural_patch_mask": structural_patch_mask.squeeze(0) if squeeze_batch else structural_patch_mask,
        "d_in_mask": validated_d_in_mask.squeeze(0) if squeeze_batch else validated_d_in_mask,
        "d_out_mask": validated_d_out_mask.squeeze(0) if squeeze_batch else validated_d_out_mask,
        "T": int(T),
        "d_in_pad": int(d_in_pad),
    }


def load_checkpoint_config(path: str | Path) -> Any:
    payload = torch.load(Path(str(path)).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must contain a mapping payload, got {type(payload)!r}")
    if "config" not in payload:
        raise KeyError(f"checkpoint has no 'config' field: {path}")
    return OmegaConf.create(payload["config"])


__all__ = [
    "build_big_vae_model_cfg",
    "encode_big_vae_layer_batch",
    "load_checkpoint_config",
    "load_frozen_big_vae_from_checkpoint",
    "load_frozen_layer_latent_diffusion_prior",
]
