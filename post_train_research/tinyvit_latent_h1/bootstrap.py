from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.compare_vit_tiny_latent_optimization import (
    BigVAELatentTensorStore,
    FunctionalViTTiny,
    load_frozen_big_vae_decoder,
    make_initial_tensors,
)
from post_train_research.tinyvit_latent_h1.config import RunConfig
from post_train_research.tinyvit_latent_h1.source import (
    SourceContext,
    clone_named_tensors,
    conditioning_state_dict,
    evaluate_train_and_test,
    load_checkpoint_payload,
    resolve_source_checkpoint_for_experiment,
    _apply_source_overrides,
    resolve_source_context,
)
from post_train_research.vit_latent_scaling.init import collect_calibration_images, export_named_tensors
from training.big_vae_latent_diffusion import (
    load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint,
    load_frozen_layer_latent_diffusion_prior,
)


def bootstrap_uses_source_checkpoint(cfg: RunConfig) -> bool:
    return str(cfg.source.bootstrap_kind).strip().lower() == "source"


def prepare_config_from_source_checkpoint(cfg: RunConfig, logger: logging.Logger) -> Path | None:
    if not bootstrap_uses_source_checkpoint(cfg):
        if not cfg.setup.big_vae_checkpoint:
            raise ValueError("setup.big_vae_checkpoint is required when source.bootstrap_kind!='source'")
        return None
    checkpoint_path = resolve_source_checkpoint_for_experiment(cfg)
    payload = load_checkpoint_payload(checkpoint_path)
    if not isinstance(payload.get("latent_slots"), dict):
        raise ValueError(f"Source checkpoint must contain latent_slots: {checkpoint_path}")
    _apply_source_overrides(cfg, payload, logger)
    return checkpoint_path


def _resolve_bootstrap_source_context(
    cfg: RunConfig,
    logger: logging.Logger,
    *,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> SourceContext:
    bootstrap_kind = str(cfg.source.bootstrap_kind).strip().lower()
    initial_tensors = make_initial_tensors(cfg.vit_cfg, seed=int(cfg.train.seed))
    big_vae = load_frozen_big_vae_decoder(cfg.setup.big_vae_checkpoint, device=device)
    prior = None
    latent_init = bootstrap_kind
    if bootstrap_kind == "diffusion_prior":
        prior = load_frozen_layer_latent_diffusion_prior(cfg.source.diffusion_prior_checkpoint, device=device)
        loaded = load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint(
            cfg.source.diffusion_prior_checkpoint,
            big_vae=big_vae,
        )
        if loaded:
            logger.info("Loaded finetuned distribution encoder state from diffusion prior checkpoint")
    model = FunctionalViTTiny(
        cfg.vit_cfg,
        clone_named_tensors(initial_tensors),
        parameter_mode="bigvae_latent",
        big_vae=big_vae,
        big_vae_latent_init=latent_init,
        big_vae_diffusion_prior=prior,
        big_vae_diffusion_prior_steps=int(cfg.source.diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(cfg.source.diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(cfg.source.diffusion_prior_eta),
        big_vae_random_init_std=float(cfg.source.random_init_std),
        big_vae_latent_noise_std=0.0,
        big_vae_latent_parameterization=str(cfg.setup.big_vae_latent_parameterization),
        big_vae_decode=str(cfg.setup.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.setup.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.setup.big_vae_tile_d_out),
        big_vae_encoder_context_rows=64,
        big_vae_encoder_context_std=1.0,
        big_vae_encoder_batch_size=16,
    ).to(device)
    if bootstrap_kind == "diffusion_prior":
        calibration_images = collect_calibration_images(train_loader, num_batches=int(cfg.source.calibration_batches))
        if calibration_images is None:
            raise RuntimeError("Could not collect calibration images for diffusion-prior bootstrap")
        model.initialize_bigvae_diffusion_prior(calibration_images.to(device))
    store = getattr(model, "store")
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("Bootstrap source must use BigVAELatentTensorStore")
    z_star_state = store.materialized_latent_slots_state_dict()
    z_star_conditioning_state = conditioning_state_dict(store)
    z_star_named = export_named_tensors(model, list(initial_tensors.keys()))
    train_metrics, test_metrics = evaluate_train_and_test(model, train_loader, test_loader, device=device, amp_enabled=bool(cfg.train.amp))
    decoded_names = list(store.decoded_tensor_names())
    raw_param_count = int(sum(int(z_star_named[name].numel()) for name in decoded_names))
    logger.info(
        "Bootstrapped source kind=%s train_loss=%.6f test_acc=%.4f",
        bootstrap_kind,
        float(train_metrics.loss),
        float(test_metrics.accuracy),
    )
    return SourceContext(
        checkpoint_path=Path(f"{bootstrap_kind}_bootstrap_init.pt"),
        payload={},
        vit_cfg=cfg.vit_cfg,
        initial_tensors=initial_tensors,
        all_tensor_names=list(initial_tensors.keys()),
        decoded_names=decoded_names,
        latent_space=str(store.latent_space),
        big_vae=big_vae,
        z_star_state=z_star_state,
        z_star_conditioning_state=z_star_conditioning_state,
        z_star_named_tensors=z_star_named,
        z_star_train_metrics=train_metrics,
        z_star_test_metrics=test_metrics,
        latent_param_count=int(store.latent_numel()),
        raw_param_count=raw_param_count,
    )


def resolve_source_context_from_config(
    cfg: RunConfig,
    logger: logging.Logger,
    *,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_path: Path | None = None,
) -> SourceContext:
    if bootstrap_uses_source_checkpoint(cfg):
        return resolve_source_context(
            cfg,
            logger,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            checkpoint_path=checkpoint_path,
        )
    return _resolve_bootstrap_source_context(cfg, logger, device=device, train_loader=train_loader, test_loader=test_loader)
