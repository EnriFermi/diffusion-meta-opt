from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from big_vae.models import (
    BigVAEConfig,
    DistributionConfig,
    EncoderConfig,
    MiniVAEConfig,
    ModelConfig,
    PatchTokenizerConfig,
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "items"):
        return value
    raise TypeError(f"{name} must be a mapping")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return float(value)


def _section(container: Mapping[str, Any], key: str, *, name: str) -> Mapping[str, Any]:
    return _mapping(container.get(key, {}), name=name)


def build_big_vae_model_config(raw_cfg: Mapping[str, Any]) -> ModelConfig:
    model_cfg = _mapping(raw_cfg.get("model", raw_cfg), name="model")
    big_cfg = _section(model_cfg, "big_vae", name="model.big_vae")
    enc_cfg = _section(big_cfg, "encoder", name="model.big_vae.encoder")

    dist_cfg = _section(model_cfg, "distribution", name="model.distribution")
    nested_dist_cfg = _section(big_cfg, "distribution_encoder", name="model.big_vae.distribution_encoder")
    if nested_dist_cfg:
        dist_cfg = nested_dist_cfg

    mini_cfg = _section(model_cfg, "mini_vae", name="model.mini_vae")
    tokenizer_cfg = _section(big_cfg, "patch_tokenizer", name="model.big_vae.patch_tokenizer")

    patch_size = int(model_cfg.get("patch_size", 16))
    d_patch = int(tokenizer_cfg.get("d_patch", mini_cfg.get("d_patch", 64)))
    tokenizer_kind = str(tokenizer_cfg.get("kind", big_cfg.get("patch_tokenizer_kind", "residual")))
    tokenizer_dropout = _optional_float(tokenizer_cfg.get("dropout", None))

    return ModelConfig(
        patch_size=patch_size,
        beta=float(model_cfg.get("beta", 1e-3)),
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
            patch_size_for_cov=int(dist_cfg.get("patch_size_for_cov", patch_size)),
        ),
        mini_vae=MiniVAEConfig(
            z_dim=int(mini_cfg.get("z_dim", 64)),
            d_e=int(mini_cfg.get("d_e", 128)),
            encoder_latent_dim=int(mini_cfg.get("encoder_latent_dim", 0)),
            pos_lat_dim=int(mini_cfg.get("pos_lat_dim", 0)),
            pos_dim=int(mini_cfg.get("pos_dim", 32)),
            num_attn_layers_encoder=int(mini_cfg.get("num_attn_layers_encoder", 2)),
            num_layers_decoder=int(mini_cfg.get("num_layers_decoder", 2)),
            decoder_bilinear_rank=int(mini_cfg.get("decoder_bilinear_rank", 0)),
            stub_resampler_d_model=int(mini_cfg.get("stub_resampler_d_model", 0)),
            mlp_stub_hidden_dim=int(mini_cfg.get("mlp_stub_hidden_dim", 256)),
            init_style=str(mini_cfg.get("init_style", "llm")),
            n_heads=int(mini_cfg.get("n_heads", 4)),
            d_patch=d_patch,
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
            use_encoder_mu_head=bool(big_cfg.get("use_encoder_mu_head", False)),
            normalize_latent_slots_before_mu=bool(big_cfg.get("normalize_latent_slots_before_mu", True)),
            latent_prior_kind=str(big_cfg.get("latent_prior_kind", "gaussian")),
            vamp_prior_K=int(big_cfg.get("vamp_prior_K", 64)),
            decoder_query_conditioning_kind=str(big_cfg.get("decoder_query_conditioning_kind", "linear")),
            decoder_query_conditioning_hidden_mult=float(big_cfg.get("decoder_query_conditioning_hidden_mult", 2.0)),
            rope_2d_coord_kind=str(big_cfg.get("rope_2d_coord_kind", "normalized_center")),
            latent_sampling_min_std=float(big_cfg.get("latent_sampling_min_std", 1e-4)),
            latent_sampling_logvar_min=float(big_cfg.get("latent_sampling_logvar_min", -20.0)),
            latent_sampling_logvar_max=float(big_cfg.get("latent_sampling_logvar_max", 10.0)),
            disable_z_shortcut=bool(big_cfg.get("disable_z_shortcut", False)),
            disable_distribution_encoder=bool(big_cfg.get("disable_distribution_encoder", False)),
            patch_tokenizer_kind=tokenizer_kind,
            distribution_encoder_conditioning_kind=str(big_cfg.get("distribution_encoder_conditioning_kind", "legacy")),
            patch_tokenizer=PatchTokenizerConfig(
                kind=tokenizer_kind,
                d_patch=d_patch,
                hidden_dim=_optional_int(tokenizer_cfg.get("hidden_dim", None)),
                cond_proj_dim=_optional_int(tokenizer_cfg.get("cond_proj_dim", None)),
                num_blocks=int(tokenizer_cfg.get("num_blocks", 2)),
                dropout=tokenizer_dropout,
                residual_hidden_dim=int(tokenizer_cfg.get("residual_hidden_dim", 256)),
                residual_num_layers=int(tokenizer_cfg.get("residual_num_layers", 3)),
                residual_dropout=float(tokenizer_cfg.get("residual_dropout", 0.0)),
            ),
            encoder=EncoderConfig(
                self_attn_mode=str(enc_cfg.get("self_attn_mode", "full")),
                cross_attend_only_cls=bool(enc_cfg.get("cross_attend_only_cls", True)),
            ),
        ),
    )


__all__ = ["build_big_vae_model_config"]
