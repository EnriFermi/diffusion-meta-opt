from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

from post_train_research.vit_latent_scaling.run_vit_latent_scaling import (
    ScalingRunConfig,
    ViTTinyConfig,
    train_once,
    validate_scaling_run_config,
)


def _as_mapping(section: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(section, (dict, DictConfig)):
        raise TypeError(f"{name} must be a mapping")
    return section


def build_vit_latent_scaling_configs(cfg: DictConfig) -> tuple[ScalingRunConfig, ViTTinyConfig]:
    run_cfg_raw = _as_mapping(cfg.get("vit_latent_scaling"), name="vit_latent_scaling")
    model_cfg_raw = _as_mapping(cfg.get("vit_model"), name="vit_model")

    run_cfg = ScalingRunConfig(
        output_dir=str(run_cfg_raw.get("output_dir", "")),
        dataset=str(run_cfg_raw.get("dataset", "")),
        model_size=str(run_cfg_raw.get("model_size", "")),
        setup=str(run_cfg_raw.get("setup", "")),
        data_dir=str(run_cfg_raw.get("data_dir", "")),
        download=bool(run_cfg_raw.get("download", True)),
        device=str(run_cfg_raw.get("device", "auto")),
        seed=int(run_cfg_raw.get("seed", 42)),
        epochs=int(run_cfg_raw.get("epochs", 20)),
        max_steps=int(run_cfg_raw.get("max_steps", 0)),
        batch_size=int(run_cfg_raw.get("batch_size", 128)),
        eval_batch_size=int(run_cfg_raw.get("eval_batch_size", 256)),
        num_workers=int(run_cfg_raw.get("num_workers", 4)),
        train_subset=int(run_cfg_raw.get("train_subset", 0)),
        test_subset=int(run_cfg_raw.get("test_subset", 0)),
        lr=float(run_cfg_raw.get("lr", 1e-3)),
        weight_decay=float(run_cfg_raw.get("weight_decay", 0.05)),
        adam_beta1=float(run_cfg_raw.get("adam_beta1", 0.9)),
        adam_beta2=float(run_cfg_raw.get("adam_beta2", 0.999)),
        adam_eps=float(run_cfg_raw.get("adam_eps", 1e-8)),
        label_smoothing=float(run_cfg_raw.get("label_smoothing", 0.0)),
        grad_clip_norm=float(run_cfg_raw.get("grad_clip_norm", 0.0)),
        amp=bool(run_cfg_raw.get("amp", True)),
        tf32=bool(run_cfg_raw.get("tf32", True)),
        compile=bool(run_cfg_raw.get("compile", False)),
        log_every_steps=int(run_cfg_raw.get("log_every_steps", 50)),
        eval_every_steps=int(run_cfg_raw.get("eval_every_steps", 500)),
        save_checkpoints=bool(run_cfg_raw.get("save_checkpoints", True)),
        raw_checkpoint=str(run_cfg_raw.get("raw_checkpoint", "")),
        big_vae_checkpoint=str(run_cfg_raw.get("big_vae_checkpoint", "")),
        big_vae_latent_init=str(run_cfg_raw.get("big_vae_latent_init", "encoded")),
        big_vae_diffusion_prior_checkpoint=str(run_cfg_raw.get("big_vae_diffusion_prior_checkpoint", "")),
        big_vae_diffusion_prior_steps=int(run_cfg_raw.get("big_vae_diffusion_prior_steps", 50)),
        big_vae_diffusion_prior_sampler=str(run_cfg_raw.get("big_vae_diffusion_prior_sampler", "ddim")),
        big_vae_diffusion_prior_eta=float(run_cfg_raw.get("big_vae_diffusion_prior_eta", 0.0)),
        big_vae_decode=str(run_cfg_raw.get("big_vae_decode", "all")),
        big_vae_tile_T_patches=int(run_cfg_raw.get("big_vae_tile_T_patches", 16)),
        big_vae_tile_d_out=int(run_cfg_raw.get("big_vae_tile_d_out", 8)),
        big_vae_encoder_context_rows=int(run_cfg_raw.get("big_vae_encoder_context_rows", 64)),
        big_vae_encoder_context_std=float(run_cfg_raw.get("big_vae_encoder_context_std", 1.0)),
        big_vae_encoder_batch_size=int(run_cfg_raw.get("big_vae_encoder_batch_size", 16)),
        latent_weight_decay=float(run_cfg_raw.get("latent_weight_decay", 0.0)),
        raw_init_steps=int(run_cfg_raw.get("raw_init_steps", 0)),
    )
    validate_scaling_run_config(run_cfg)

    vit_cfg = ViTTinyConfig(
        image_size=int(model_cfg_raw.get("image_size", 32)),
        patch_size=int(model_cfg_raw.get("patch_size", 4)),
        in_channels=int(model_cfg_raw.get("in_channels", 3)),
        num_classes=int(model_cfg_raw.get("num_classes", 10)),
        hidden_dim=int(model_cfg_raw.get("hidden_dim", 192)),
        depth=int(model_cfg_raw.get("depth", 12)),
        num_heads=int(model_cfg_raw.get("num_heads", 3)),
        mlp_ratio=float(model_cfg_raw.get("mlp_ratio", 4.0)),
        dropout=float(model_cfg_raw.get("dropout", 0.0)),
        attention_dropout=float(model_cfg_raw.get("attention_dropout", 0.0)),
    )
    return run_cfg, vit_cfg


@hydra.main(version_base=None, config_path="../../conf", config_name="config_vit_latent_scaling")
def main(cfg: DictConfig) -> None:
    run_cfg, vit_cfg = build_vit_latent_scaling_configs(cfg)
    print(
        "[run_vit_latent_scaling_hydra] "
        f"dataset={run_cfg.dataset} model_size={run_cfg.model_size} setup={run_cfg.setup} "
        f"latent_init={run_cfg.big_vae_latent_init} output_dir={Path(run_cfg.output_dir).expanduser()}",
        flush=True,
    )
    train_once(run_cfg, vit_cfg)


if __name__ == "__main__":
    main()
