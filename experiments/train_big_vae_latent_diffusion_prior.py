from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from dataset.logging_utils import configure_process_logging
from dataset.big_vae_latent_diffusion_offline import (
    collate_big_vae_latent_diffusion_batch,
    offline_big_vae_latent_diffusion_data_pipeline,
)
from experiments.train_big_vae import (
    _find_latest_resume_state_checkpoint,
    _load_training_state_from_checkpoint,
    _resume_state_load_policy,
    _save_checkpoint,
    _save_resume_state_checkpoint,
    compute_grad_stats,
)
from models.layer_latent_diffusion_prior import (
    LayerLatentDiffusionPrior,
    build_layer_latent_diffusion_prior_config,
    compute_layer_latent_diffusion_loss,
)
from training.optim import build_adamw_optimizer, build_cosine_scheduler
from training.big_vae_latent_diffusion import load_frozen_big_vae_from_checkpoint
from training.runtime import (
    autocast_context,
    configure_per_run_artifacts,
    create_grad_scaler,
    maybe_compile_model,
    resolve_amp,
    set_speed_optimizations,
)


LOGGER = logging.getLogger("train_big_vae_latent_diffusion_prior")


def _flatten_for_tracking(
    payload: Any,
    *,
    prefix: str = "",
    out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = {} if out is None else out
    if isinstance(payload, (dict, DictConfig)):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_for_tracking(value, prefix=next_prefix, out=target)
        return target
    if isinstance(payload, (list, tuple, ListConfig)):
        target[prefix] = list(payload)
        return target
    target[prefix] = payload
    return target


class CometTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        self.logger = logger
        self.experiment: Any | None = None
        self.enabled = False
        telemetry_cfg = cfg.train.get("telemetry", {})
        comet_cfg = telemetry_cfg.get("comet", {}) if isinstance(telemetry_cfg, (dict, DictConfig)) else {}
        if not bool(comet_cfg.get("enabled", False)):
            return
        try:
            from comet_ml import Experiment, OfflineExperiment  # type: ignore
        except Exception as exc:
            self.logger.warning("Comet is enabled but comet_ml is unavailable: %s", exc)
            return

        api_key = str(comet_cfg.get("api_key", "")).strip()
        workspace = str(comet_cfg.get("workspace", "")).strip()
        project_name = str(comet_cfg.get("project_name", "big_vae_latent_diffusion_prior")).strip() or (
            "big_vae_latent_diffusion_prior"
        )
        experiment_name = str(comet_cfg.get("experiment_name", "")).strip()
        offline_dir = str(comet_cfg.get("offline_directory", "")).strip()
        log_code = bool(comet_cfg.get("log_code", False))
        try:
            if api_key:
                exp = Experiment(
                    api_key=api_key,
                    project_name=project_name,
                    workspace=workspace or None,
                    auto_output_logging="simple",
                    log_code=log_code,
                )
            else:
                exp = OfflineExperiment(
                    project_name=project_name,
                    workspace=workspace or None,
                    auto_output_logging="simple",
                    log_code=log_code,
                    offline_directory=offline_dir or None,
                )
            if experiment_name:
                exp.set_name(experiment_name)
            tags = comet_cfg.get("tags", [])
            if isinstance(tags, (list, tuple, ListConfig)):
                for tag in tags:
                    exp.add_tag(str(tag))
            exp.log_parameters(_flatten_for_tracking(OmegaConf.to_container(cfg, resolve=True)))
            self.experiment = exp
            self.enabled = True
        except Exception as exc:
            self.logger.warning("Failed to initialize Comet tracker: %s", exc)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_metrics(metrics, step=int(step))
        except Exception as exc:
            self.logger.warning("Comet metric log failed at step=%s: %s", step, exc)

    def end(self) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.end()
        except Exception:
            pass


class WandbTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        self.logger = logger
        self.run: Any | None = None
        self.enabled = False
        telemetry_cfg = cfg.train.get("telemetry", {})
        wandb_cfg = telemetry_cfg.get("wandb", {}) if isinstance(telemetry_cfg, (dict, DictConfig)) else {}
        if not bool(wandb_cfg.get("enabled", False)):
            return
        try:
            import wandb  # type: ignore
        except Exception as exc:
            self.logger.warning("W&B is enabled but wandb is unavailable: %s", exc)
            return

        project_name = str(wandb_cfg.get("project_name", "big_vae_latent_diffusion_prior")).strip() or (
            "big_vae_latent_diffusion_prior"
        )
        entity = str(wandb_cfg.get("entity", "")).strip()
        run_name = str(wandb_cfg.get("run_name", "")).strip()
        run_mode = str(wandb_cfg.get("mode", "offline")).strip().lower() or "offline"
        run_dir = str(wandb_cfg.get("dir", "")).strip()
        tags = wandb_cfg.get("tags", [])
        tags_list = [str(tag) for tag in tags] if isinstance(tags, (list, tuple, ListConfig)) else None
        try:
            self.run = wandb.init(
                project=project_name,
                entity=entity or None,
                name=run_name or None,
                mode=run_mode,
                dir=run_dir or None,
                config=_flatten_for_tracking(OmegaConf.to_container(cfg, resolve=True)),
                tags=tags_list,
            )
            if self.run is not None:
                self.enabled = True
        except Exception as exc:
            self.logger.warning("Failed to initialize W&B tracker: %s", exc)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=int(step))
        except Exception as exc:
            self.logger.warning("W&B metric log failed at step=%s: %s", step, exc)

    def end(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass


def _promote_run_profile_to_root(cfg: DictConfig) -> None:
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return
    expected_sections = ("train", "model", "training_artifacts", "logging", "hf")
    with open_dict(cfg):
        for section in expected_sections:
            if section not in cfg and section in run_profiles_cfg:
                cfg[section] = run_profiles_cfg[section]


def _resolve_prior_cfg(cfg: DictConfig, *, dataset_summary: dict[str, Any]) -> DictConfig:
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, (dict, DictConfig)):
        raise TypeError("model must be a mapping")
    prior_cfg = model_cfg.get("latent_diffusion_prior", {})
    if not isinstance(prior_cfg, (dict, DictConfig)):
        raise TypeError("model.latent_diffusion_prior must be a mapping")
    with open_dict(cfg):
        if "latent_diffusion_prior" not in cfg.model:
            cfg.model.latent_diffusion_prior = {}
        if int(prior_cfg.get("z_dim", 0)) <= 0:
            cfg.model.latent_diffusion_prior.z_dim = int(dataset_summary.get("z_dim", 0))
        if int(prior_cfg.get("cond_dim", 0)) <= 0:
            cfg.model.latent_diffusion_prior.cond_dim = int(dataset_summary.get("cond_dim", 0))
    return cfg.model.latent_diffusion_prior


def _resolve_checkpoint_layout(cfg: DictConfig) -> int:
    train_cfg = cfg.train
    stage = max(1, int(train_cfg.get("stage", 1)))
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "./artifacts/training/checkpoints/big_vae_latent_diffusion_prior")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return stage


def _dummy_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


def _next_batch(iterator: Iterator[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    items = [next(iterator) for _ in range(max(1, int(batch_size)))]
    return collate_big_vae_latent_diffusion_batch(items)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "latent_mu",
        "latent_logvar",
        "cond_patch",
        "patch_mask",
        "d_in",
        "d_out",
        "X",
        "W",
        "x_mask",
        "d_in_mask",
        "d_out_mask",
    ):
        value = moved.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


@hydra.main(version_base=None, config_path="../conf", config_name="config_big_vae_latent_diffusion_prior")
def main(cfg: DictConfig) -> None:
    _promote_run_profile_to_root(cfg)
    artifacts = configure_per_run_artifacts(cfg, run_label="train_big_vae_latent_diffusion_prior")
    with open_dict(cfg):
        cfg.train.checkpoint_dir = str(Path(artifacts["root_dir"]) / "checkpoints" / "big_vae_latent_diffusion_prior")

    log_path = configure_process_logging(cfg=cfg, role="train_big_vae_latent_diffusion_prior", rank=0, force=True)
    logger = LOGGER
    logger.info("Starting latent diffusion prior training")
    logger.info("Run log file: %s", log_path)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device(str(cfg.train.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    set_speed_optimizations(cfg, device, section="train")
    amp_enabled, amp_dtype = resolve_amp(cfg, device, section="train")
    stage = _resolve_checkpoint_layout(cfg)

    comet_tracker: CometTracker | None = None
    wandb_tracker: WandbTracker | None = None
    with offline_big_vae_latent_diffusion_data_pipeline(cfg, logger=logger) as dataset:
        dataset_summary = dataset.summary()
        dataset_stats = dataset.latent_stats()
        prior_cfg_raw = _resolve_prior_cfg(cfg, dataset_summary=dataset_summary)
        prior_cfg = build_layer_latent_diffusion_prior_config(prior_cfg_raw)

        logger.info(
            "Latent dataset summary: root=%s accepted_records=%s z_dim=%s cond_dim=%s has_decoder_aux_tensors=%s",
            dataset_summary.get("root_dir", ""),
            int(dataset_summary.get("accepted_records", 0)),
            int(dataset_summary.get("z_dim", 0)),
            int(dataset_summary.get("cond_dim", 0)),
            bool(dataset_summary.get("has_decoder_aux_tensors", False)),
        )
        logger.info(
            "Model config: z_dim=%s num_latent_tokens=%s cond_dim=%s d_model=%s n_layers=%s n_heads=%s prediction_type=%s",
            prior_cfg.z_dim,
            prior_cfg.num_latent_tokens,
            prior_cfg.cond_dim,
            prior_cfg.d_model,
            prior_cfg.n_layers,
            prior_cfg.n_heads,
            prior_cfg.prediction_type,
        )

        model = LayerLatentDiffusionPrior(prior_cfg).to(device)
        model.set_latent_normalization_stats(
            latent_mean=dataset_stats["latent_mean"],
            latent_std=dataset_stats["latent_std"],
        )
        model = maybe_compile_model(model, cfg, logger, section="train", label="latent_diffusion_prior")
        decoder_aux_model = None
        if bool(prior_cfg.use_decoder_aux):
            if not bool(dataset_summary.get("has_decoder_aux_tensors", False)):
                raise ValueError(
                    "decoder auxiliary loss requires offline latent diffusion dataset with stored reconstruction targets. "
                    "Rebuild the dataset with latent_diffusion_dataset.builder.store_decoder_aux_tensors=true."
                )
            aux_checkpoint = str(cfg.get("latent_diffusion_prior", {}).get("big_vae_checkpoint", "")).strip()
            if not aux_checkpoint:
                raise ValueError(
                    "latent_diffusion_prior.big_vae_checkpoint must be set when model.latent_diffusion_prior.use_decoder_aux=true"
                )
            decoder_aux_model = load_frozen_big_vae_from_checkpoint(aux_checkpoint, device=device)
            logger.info("Loaded frozen BigVAE decoder for auxiliary loss: %s", aux_checkpoint)

        optimizer = build_adamw_optimizer(
            model=model,
            cfg=cfg,
            device=device,
            section="train",
            default_lr=3e-4,
            default_weight_decay=0.01,
        )
        scheduler = build_cosine_scheduler(
            optimizer=optimizer,
            cfg=cfg,
            section="train",
            default_max_steps=1000,
            default_warmup_steps=100,
            default_min_lr_ratio=0.1,
        )
        scheduler_is_dummy = scheduler is None
        if scheduler is None:
            scheduler = _dummy_scheduler(optimizer)
        scaler = create_grad_scaler(device=device, enabled=amp_enabled)

        comet_tracker = CometTracker(cfg, logger)
        wandb_tracker = WandbTracker(cfg, logger)

        checkpoint_every = max(1, int(cfg.train.get("checkpoint_every", 1000)))
        log_every = max(1, int(cfg.train.get("log_every", 50)))
        grad_accum_steps = max(1, int(cfg.train.get("grad_accum_steps", 1)))
        batch_size = max(1, int(cfg.train.get("batch_size", 8)))
        max_steps = max(1, int(cfg.train.get("max_steps", 1000)))
        grad_clip_norm = float(cfg.train.get("grad_clip_norm", 0.0))
        resume_state_cfg = cfg.train.get("resume_state", {})
        if resume_state_cfg is None:
            resume_state_cfg = {}
        if not isinstance(resume_state_cfg, (dict, DictConfig)):
            raise TypeError("train.resume_state must be a mapping")
        resume_state_enabled = bool(resume_state_cfg.get("enabled", True))
        resume_state_auto_resume = bool(resume_state_cfg.get("auto_resume", True))
        resume_state_save_every = max(1, int(resume_state_cfg.get("save_every", checkpoint_every)))
        resume_state_load_policy = _resume_state_load_policy(resume_state_cfg)
        default_resume_state_dir = Path(str(cfg.train.get("checkpoint_dir"))) / f"stage_{stage}" / "resume_state"
        resume_state_dir = Path(str(resume_state_cfg.get("dir", default_resume_state_dir)))
        resume_state_dir.mkdir(parents=True, exist_ok=True)

        resumed_step = 0
        if resume_state_enabled and resume_state_auto_resume:
            candidate = _find_latest_resume_state_checkpoint(resume_state_dir)
            if candidate is not None:
                logger.info("Auto-resume candidate found: %s", candidate)
                resumed_step = _load_training_state_from_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    path=candidate,
                    logger=logger,
                    load_model_state=resume_state_load_policy["load_model_state"],
                    load_optimizer_state=resume_state_load_policy["load_optimizer_state"],
                    load_scheduler_state=resume_state_load_policy["load_scheduler_state"],
                    load_scaler_state=resume_state_load_policy["load_scaler_state"],
                    load_step=resume_state_load_policy["load_step"],
                )

        iterator = iter(dataset)
        optimizer.zero_grad(set_to_none=True)

        for global_step in range(resumed_step + 1, max_steps + 1):
            step_start = time.perf_counter()
            step_loss_sum = 0.0
            step_patch_tokens = 0.0
            step_patch_tokens_max = 0.0
            step_alpha_sum = 0.0
            step_sigma_sum = 0.0
            step_clean_std = 0.0
            step_pred_std = 0.0
            step_target_std = 0.0
            step_diffusion_loss = 0.0
            step_decoder_aux_loss = 0.0
            step_decoder_aux_behavioral = 0.0
            step_decoder_aux_structural = 0.0
            step_decoder_aux_applied_fraction = 0.0
            step_decoder_aux_weight_mean = 0.0

            for micro_idx in range(grad_accum_steps):
                batch = _move_batch_to_device(_next_batch(iterator, batch_size=batch_size), device=device)
                timesteps = model.schedule.sample_timesteps(batch["latent_mu"].shape[0], device=device)
                alpha_t, sigma_t = model.schedule.alpha_sigma(timesteps, x_ndim=2)
                step_alpha_sum += float(alpha_t.mean().detach().item())
                step_sigma_sum += float(sigma_t.mean().detach().item())
                patch_lengths = batch["patch_mask"].to(dtype=torch.long).sum(dim=1)
                step_patch_tokens += float(patch_lengths.float().mean().item())
                step_patch_tokens_max = max(step_patch_tokens_max, float(patch_lengths.max().item()))

                with autocast_context(amp_enabled, amp_dtype):
                    loss_payload = compute_layer_latent_diffusion_loss(
                        model,
                        clean_latents=batch["latent_mu"],
                        cond_patch=batch["cond_patch"],
                        timesteps=timesteps,
                        patch_mask=batch["patch_mask"],
                        decoder_aux_model=decoder_aux_model,
                        decoder_aux_W=batch.get("W"),
                        decoder_aux_X=batch.get("X"),
                        decoder_aux_x_mask=batch.get("x_mask"),
                        decoder_aux_d_in_mask=batch.get("d_in_mask"),
                        decoder_aux_d_out_mask=batch.get("d_out_mask"),
                    )
                    loss = loss_payload["loss"]
                    step_diffusion_loss += float(loss_payload["diffusion_loss"].detach().item())
                    step_decoder_aux_loss += float(loss_payload["decoder_aux_loss"].detach().item())
                    step_decoder_aux_behavioral += float(loss_payload["decoder_aux_behavioral_loss"].detach().item())
                    step_decoder_aux_structural += float(loss_payload["decoder_aux_structural_loss"].detach().item())
                    step_decoder_aux_applied_fraction += float(loss_payload["decoder_aux_applied_fraction"].detach().item())
                    step_decoder_aux_weight_mean += float(loss_payload["decoder_aux_weight_mean"].detach().item())
                    step_clean_std += float(loss_payload["clean_tokens"].detach().float().std(unbiased=False).item())
                    step_pred_std += float(loss_payload["pred_tokens"].detach().float().std(unbiased=False).item())
                    step_target_std += float(loss_payload["target_tokens"].detach().float().std(unbiased=False).item())
                step_loss_sum += float(loss.detach().item())
                scaler.scale(loss / float(grad_accum_steps)).backward()

            scaler.unscale_(optimizer)
            grad_stats = compute_grad_stats(model)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not scheduler_is_dummy:
                scheduler.step()

            lr = float(optimizer.param_groups[0]["lr"])
            step_time = time.perf_counter() - step_start
            metrics = {
                "loss/total": step_loss_sum / float(grad_accum_steps),
                "loss/diffusion": step_diffusion_loss / float(grad_accum_steps),
                "loss/decoder_aux": step_decoder_aux_loss / float(grad_accum_steps),
                "loss/decoder_aux_behavioral": step_decoder_aux_behavioral / float(grad_accum_steps),
                "loss/decoder_aux_structural": step_decoder_aux_structural / float(grad_accum_steps),
                "schedule/alpha_mean": step_alpha_sum / float(grad_accum_steps),
                "schedule/sigma_mean": step_sigma_sum / float(grad_accum_steps),
                "data/patch_tokens_mean": step_patch_tokens / float(grad_accum_steps),
                "data/patch_tokens_max": step_patch_tokens_max,
                "data/decoder_aux_applied_fraction": step_decoder_aux_applied_fraction / float(grad_accum_steps),
                "data/decoder_aux_weight_mean": step_decoder_aux_weight_mean / float(grad_accum_steps),
                "latent/clean_std": step_clean_std / float(grad_accum_steps),
                "latent/pred_std": step_pred_std / float(grad_accum_steps),
                "latent/target_std": step_target_std / float(grad_accum_steps),
                "optim/lr": lr,
                "time/step_s": step_time,
                "grad/global_norm": float(grad_stats.get("grad/global_norm", 0.0)),
                "grad/rms": float(grad_stats.get("grad/rms", 0.0)),
                "grad/param_rms": float(grad_stats.get("param/rms", 0.0)),
                "grad/grad_to_param_rms_ratio": float(grad_stats.get("grad_to_param_rms_ratio", 0.0)),
            }

            if device.type == "cuda":
                metrics["gpu/memory_allocated_mb"] = float(torch.cuda.memory_allocated(device) / (1024.0 ** 2))
                metrics["gpu/memory_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024.0 ** 2))
                metrics["gpu/max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

            if global_step == 1 or global_step % log_every == 0:
                logger.info(
                    "step=%s loss=%.6f lr=%.3e alpha=%.4f sigma=%.4f patch_tokens=%.2f grad_norm=%.3e step_s=%.3f",
                    global_step,
                    metrics["loss/total"],
                    metrics["optim/lr"],
                    metrics["schedule/alpha_mean"],
                    metrics["schedule/sigma_mean"],
                    metrics["data/patch_tokens_mean"],
                    metrics["grad/global_norm"],
                    metrics["time/step_s"],
                )

            if comet_tracker is not None and comet_tracker.enabled:
                comet_tracker.log_metrics(metrics, step=global_step)
            if wandb_tracker is not None and wandb_tracker.enabled:
                wandb_tracker.log_metrics(metrics, step=global_step)

            if global_step % checkpoint_every == 0 or global_step == max_steps:
                _save_checkpoint(model=model, cfg=cfg, step_idx=global_step, logger=logger, stage=stage)
            if resume_state_enabled and (global_step % resume_state_save_every == 0 or global_step == max_steps):
                _save_resume_state_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cfg=cfg,
                    step_idx=global_step,
                    logger=logger,
                    stage=stage,
                    state_dir=resume_state_dir,
                )

    if comet_tracker is not None:
        comet_tracker.end()
    if wandb_tracker is not None:
        wandb_tracker.end()


if __name__ == "__main__":
    main()
