from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import data_pipeline, setup_logging
from big_vae.datasets.offline import (
    ensure_presliced_big_vae_dataset,
    offline_big_vae_data_pipeline,
    presliced_big_vae_data_pipeline,
)
from dataset.logging_utils import LOG_PATH_ENV, configure_process_logging, resolve_process_log_path
from experiments.background_prefetch import BackgroundPrefetcher
from big_vae.models import (
    BigVAEConfig,
    DistributionConfig,
    EncoderConfig,
    MiniVAEConfig,
    ModelConfig,
    WeightQuantileVAE,
    build_weight_quantile_vae,
)
from training.optim import build_adamw_optimizer, build_cosine_scheduler
from training.forensics import (
    apply_nccl_forensics_env,
    emit_fatal_report,
    maybe_enable_core_dumps,
    maybe_redirect_stdio,
    monitor_send_event,
    start_process_monitor,
    stop_process_monitor,
)
from training.runtime import (
    autocast_context as runtime_autocast_context,
    configure_per_run_artifacts as runtime_configure_per_run_artifacts,
    create_grad_scaler as runtime_create_grad_scaler,
    find_free_port as runtime_find_free_port,
    get_rank_logger,
    maybe_compile_model,
    resolve_amp as runtime_resolve_amp,
    resolve_backend as runtime_resolve_backend,
    resolve_device as runtime_resolve_device,
    resolve_world_size as runtime_resolve_world_size,
    seed_everything as runtime_seed_everything,
    set_speed_optimizations as runtime_set_speed_optimizations,
)
from training.big_vae.grad_stats import _grad_stat_group_prefixes



def _build_external_tracking_params(cfg: DictConfig) -> dict[str, Any]:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    big_cfg = model_cfg.get("big_vae", {})
    patch_tokenizer_cfg = big_cfg.get("patch_tokenizer", {}) if isinstance(big_cfg, (dict, DictConfig)) else {}
    if patch_tokenizer_cfg is None:
        patch_tokenizer_cfg = {}
    if not isinstance(patch_tokenizer_cfg, (dict, DictConfig)):
        patch_tokenizer_cfg = {}
    streaming_cfg = cfg.get("streaming", {})
    collector_cfg = cfg.get("collector", {})
    batch_source_mixing_cfg = train_cfg.get("batch_source_mixing", {})
    if batch_source_mixing_cfg is None:
        batch_source_mixing_cfg = {}
    if not isinstance(batch_source_mixing_cfg, (dict, DictConfig)):
        batch_source_mixing_cfg = {}
    behavioral_loss_cfg = train_cfg.get("behavioral_loss", {})
    if behavioral_loss_cfg is None:
        behavioral_loss_cfg = {}
    if not isinstance(behavioral_loss_cfg, (dict, DictConfig)):
        behavioral_loss_cfg = {}
    latent_sampling_gate_cfg = train_cfg.get("latent_sampling_gate", {})
    if latent_sampling_gate_cfg is None:
        latent_sampling_gate_cfg = {}
    if not isinstance(latent_sampling_gate_cfg, (dict, DictConfig)):
        latent_sampling_gate_cfg = {}
    raw_max_source_samples = batch_source_mixing_cfg.get("max_source_samples", 0)
    if raw_max_source_samples is None:
        parsed_max_source_samples = 0
    elif isinstance(raw_max_source_samples, str) and raw_max_source_samples.strip().lower() in {"", "none", "null"}:
        parsed_max_source_samples = 0
    else:
        parsed_max_source_samples = int(raw_max_source_samples)
    offline_batch_prefetch_cfg = train_cfg.get("offline_batch_prefetch", {})
    if offline_batch_prefetch_cfg is None:
        offline_batch_prefetch_cfg = {}
    if not isinstance(offline_batch_prefetch_cfg, (dict, DictConfig)):
        offline_batch_prefetch_cfg = {}
    preslicing_cfg = train_cfg.get("preslicing", {})
    if preslicing_cfg is None:
        preslicing_cfg = {}
    if not isinstance(preslicing_cfg, (dict, DictConfig)):
        preslicing_cfg = {}

    tracking_params: dict[str, Any] = {
        "train.max_steps": int(train_cfg.get("max_steps", 0)),
        "train.lr": float(train_cfg.get("lr", 0.0)),
        "train.patch_tokenizer_alpha_lr": float(train_cfg.get("patch_tokenizer_alpha_lr", 0.0) or 0.0),
        "train.patch_tokenizer_alpha_weight_decay": float(
            train_cfg.get("patch_tokenizer_alpha_weight_decay", 0.0) or 0.0
        ),
        "train.grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
        "train.kl_beta": float(train_cfg.get("kl_beta", 0.0)),
        "train.kl_schedule.enabled": bool(train_cfg.get("kl_schedule", {}).get("enabled", False)),
        "train.kl_schedule.start_beta": float(train_cfg.get("kl_schedule", {}).get("start_beta", 0.0)),
        "train.kl_schedule.warmup_steps": int(train_cfg.get("kl_schedule", {}).get("warmup_steps", 0)),
        "train.kl_schedule.ramp_steps": int(train_cfg.get("kl_schedule", {}).get("ramp_steps", 0)),
        "train.latent_sampling_gate.enabled": bool(latent_sampling_gate_cfg.get("enabled", True)),
        "train.latent_sampling_gate.start_step": int(latent_sampling_gate_cfg.get("start_step", 0)),
        "train.latent_sampling_gate.ramp_steps": int(latent_sampling_gate_cfg.get("ramp_steps", 0)),
        "train.latent_sampling_gate.start_value": float(latent_sampling_gate_cfg.get("start_value", 1e-4)),
        "train.latent_sampling_gate.end_value": float(latent_sampling_gate_cfg.get("end_value", 1.0)),
        "train.behavioral_coef": float(train_cfg.get("behavioral_coef", 0.0)),
        "train.structural_coef": float(train_cfg.get("structural_coef", 0.0)),
        "train.behavioral_loss.lambda_operator": float(behavioral_loss_cfg.get("lambda_operator", 1.0)),
        "train.behavioral_loss.lambda_dir": float(behavioral_loss_cfg.get("lambda_dir", 0.0)),
        "train.behavioral_loss.lambda_scale": float(behavioral_loss_cfg.get("lambda_scale", 0.0)),
        "train.slice_batch_size": int(train_cfg.get("slice_batch_size", 1)),
        "train.steps_per_sample": int(train_cfg.get("steps_per_sample", 1)),
        "train.batch_source_mixing.enabled": bool(batch_source_mixing_cfg.get("enabled", False)),
        "train.batch_source_mixing.strategy": str(batch_source_mixing_cfg.get("strategy", "round_robin")),
        "train.batch_source_mixing.uniqueness": str(batch_source_mixing_cfg.get("uniqueness", "none")),
        "train.batch_source_mixing.max_source_samples": int(parsed_max_source_samples),
        "train.batch_source_mixing.consume_slices_without_replacement": bool(
            batch_source_mixing_cfg.get("consume_slices_without_replacement", False)
        ),
        "train.offline_dataset.enabled": bool(train_cfg.get("offline_dataset", {}).get("enabled", False)),
        "train.offline_dataset.root_dir": str(train_cfg.get("offline_dataset", {}).get("root_dir", "")),
        "train.offline_dataset.shard_by_rank": bool(train_cfg.get("offline_dataset", {}).get("shard_by_rank", True)),
        "train.offline_dataset.runtime_enforce_stage_compatibility": bool(
            train_cfg.get("offline_dataset", {}).get("runtime_enforce_stage_compatibility", False)
        ),
        "train.offline_dataset.loader_workers": int(train_cfg.get("offline_dataset", {}).get("loader_workers", 0)),
        "train.offline_dataset.loader_batch_size": int(train_cfg.get("offline_dataset", {}).get("loader_batch_size", 1)),
        "train.offline_dataset.loader_prefetch_factor": int(
            train_cfg.get("offline_dataset", {}).get("loader_prefetch_factor", 2)
        ),
        "train.offline_dataset.loader_persistent_workers": bool(
            train_cfg.get("offline_dataset", {}).get("loader_persistent_workers", True)
        ),
        "train.offline_batch_prefetch.enabled": bool(offline_batch_prefetch_cfg.get("enabled", True)),
        "train.offline_batch_prefetch.queue_size": int(offline_batch_prefetch_cfg.get("queue_size", 2)),
        "train.offline_batch_prefetch.pin_memory": bool(offline_batch_prefetch_cfg.get("pin_memory", True)),
        "train.offline_batch_prefetch.cpu_threads": int(offline_batch_prefetch_cfg.get("cpu_threads", 0)),
        "train.offline_batch_prefetch.refill_fetch_batch_size": int(
            offline_batch_prefetch_cfg.get("refill_fetch_batch_size", 0)
        ),
        "train.preslicing.enabled": bool(preslicing_cfg.get("enabled", False)),
        "train.preslicing.root_dir": str(preslicing_cfg.get("root_dir", "")),
        "train.preslicing.num_slices": int(preslicing_cfg.get("num_slices", 0)),
        "train.preslicing.chunk_size_slices": int(preslicing_cfg.get("chunk_size_slices", 0)),
        "model.patch_size": int(model_cfg.get("patch_size", 16)),
        "model.big_vae.use_latent_sampling": bool(big_cfg.get("use_latent_sampling", True)),
        "model.big_vae.use_encoder_mu_head": bool(big_cfg.get("use_encoder_mu_head", False)),
        "model.big_vae.normalize_latent_slots_before_mu": bool(big_cfg.get("normalize_latent_slots_before_mu", True)),
        "model.big_vae.latent_prior_kind": str(big_cfg.get("latent_prior_kind", "gaussian")),
        "model.big_vae.vamp_prior_K": int(big_cfg.get("vamp_prior_K", 64)),
        "model.big_vae.decoder_query_conditioning_kind": str(
            big_cfg.get("decoder_query_conditioning_kind", "linear")
        ),
        "model.big_vae.decoder_query_conditioning_hidden_mult": float(
            big_cfg.get("decoder_query_conditioning_hidden_mult", 2.0)
        ),
        "model.big_vae.rope_2d_coord_kind": str(big_cfg.get("rope_2d_coord_kind", "raw")),
        "model.big_vae.latent_sampling_min_std": float(big_cfg.get("latent_sampling_min_std", 1e-4)),
        "model.big_vae.latent_sampling_logvar_min": float(big_cfg.get("latent_sampling_logvar_min", -20.0)),
        "model.big_vae.latent_sampling_logvar_max": float(big_cfg.get("latent_sampling_logvar_max", 10.0)),
        "model.big_vae.disable_distribution_encoder": bool(big_cfg.get("disable_distribution_encoder", False)),
        "model.big_vae.patch_tokenizer.kind": str(
            patch_tokenizer_cfg.get("kind", big_cfg.get("patch_tokenizer_kind", "residual"))
        ),
        "model.big_vae.patch_tokenizer.d_patch": int(patch_tokenizer_cfg.get("d_patch", 0)),
        "model.big_vae.patch_tokenizer.num_blocks": int(patch_tokenizer_cfg.get("num_blocks", 2)),
        "model.big_vae.distribution_encoder_conditioning_kind": str(
            big_cfg.get("distribution_encoder_conditioning_kind", "legacy")
        ),
        "streaming.mode": str(streaming_cfg.get("mode", "none")),
        "collector.mode": str(collector_cfg.get("mode", "auto")),
        "collector.device": str(collector_cfg.get("device", "")),
        "collector.jobs_per_selected_model": int(collector_cfg.get("jobs_per_selected_model", 0)),
        "collector.parallel_inference_workers": int(collector_cfg.get("parallel_inference_workers", 1)),
        "collector.parallel_model_pool_max_loaded_models": int(
            collector_cfg.get("parallel_model_pool_max_loaded_models", 1)
        ),
        "train.device": str(train_cfg.get("device", "")),
        "train.compile_stable_batch_shapes": bool(train_cfg.get("compile_stable_batch_shapes", False)),
    }
    resume_state_cfg = train_cfg.get("resume_state", {})
    if isinstance(resume_state_cfg, (dict, DictConfig)):
        tracking_params["train.resume_state.enabled"] = bool(resume_state_cfg.get("enabled", False))
        tracking_params["train.resume_state.auto_resume"] = bool(resume_state_cfg.get("auto_resume", True))
        tracking_params["train.resume_state.save_every"] = int(
            resume_state_cfg.get("save_every", train_cfg.get("checkpoint_every", 0))
        )
        tracking_params["train.resume_state.load_model_state"] = bool(
            resume_state_cfg.get("load_model_state", True)
        )
        tracking_params["train.resume_state.load_optimizer_state"] = bool(
            resume_state_cfg.get("load_optimizer_state", True)
        )
        tracking_params["train.resume_state.load_scheduler_state"] = bool(
            resume_state_cfg.get("load_scheduler_state", True)
        )
        tracking_params["train.resume_state.load_scaler_state"] = bool(
            resume_state_cfg.get("load_scaler_state", True)
        )
        tracking_params["train.resume_state.load_rng_state"] = bool(
            resume_state_cfg.get("load_rng_state", True)
        )
        tracking_params["train.resume_state.load_step"] = bool(resume_state_cfg.get("load_step", True))
    clip_by_part_cfg = train_cfg.get("grad_clip_norm_by_part", {})
    if isinstance(clip_by_part_cfg, (dict, DictConfig)):
        for group_name in _grad_stat_group_prefixes():
            if group_name not in clip_by_part_cfg:
                continue
            tracking_params[f"train.grad_clip_norm_by_part.{group_name}"] = float(clip_by_part_cfg[group_name])
    return tracking_params



class CometTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger, rank: int) -> None:
        self.logger = logger
        self.rank = rank
        self.experiment: Any | None = None
        self.enabled = False

        telemetry_cfg = cfg.train.get("telemetry", {})
        comet_cfg = telemetry_cfg.get("comet", {})
        if not bool(comet_cfg.get("enabled", False)):
            return
        if rank != 0:
            return

        try:
            from comet_ml import ExistingExperiment, Experiment, OfflineExperiment  # type: ignore
        except Exception as exc:
            message = str(exc)
            if "rpds.rpds" in message:
                self.logger.warning(
                    "Comet is enabled but comet_ml is unavailable: %s. "
                    "Install missing dependency in active env: `python -m pip install -U rpds-py`",
                    exc,
                )
            else:
                self.logger.warning("Comet is enabled but comet_ml is unavailable: %s", exc)
            return

        api_key = str(comet_cfg.get("api_key", "")).strip() or os.environ.get("COMET_API_KEY", "").strip()
        workspace = str(comet_cfg.get("workspace", "")).strip() or os.environ.get("COMET_WORKSPACE", "").strip()
        project_name = str(comet_cfg.get("project_name", "big_weight_vae")).strip() or "big_weight_vae"
        experiment_name = str(comet_cfg.get("experiment_name", "")).strip()
        existing_experiment_key = str(comet_cfg.get("existing_experiment_key", "")).strip()
        offline_dir = str(comet_cfg.get("offline_directory", "")).strip()
        log_code = bool(comet_cfg.get("log_code", False))

        try:
            tracking_params = _build_external_tracking_params(cfg)
            if api_key and existing_experiment_key:
                exp = ExistingExperiment(
                    api_key=api_key,
                    previous_experiment=existing_experiment_key,
                    project_name=project_name,
                    workspace=workspace or None,
                    auto_output_logging="simple",
                    log_code=log_code,
                )
            elif api_key:
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

            exp.log_parameters(tracking_params)

            self.experiment = exp
            self.enabled = True
            self.logger.info("Comet tracking enabled: project=%s workspace=%s", project_name, workspace or "<default>")
        except Exception as exc:
            self.logger.warning("Failed to initialize Comet tracker: %s", exc)
            self.experiment = None
            self.enabled = False

    def log_parameters(self, params: dict[str, Any]) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_parameters(params)
        except Exception as exc:
            self.logger.warning("Comet parameter log failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_metrics(metrics, step=int(step))
        except Exception as exc:
            self.logger.warning("Comet metrics log failed at step=%s: %s", step, exc)

    def end(self) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.end()
        except Exception:
            pass



class WandbTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger, rank: int) -> None:
        self.logger = logger
        self.rank = rank
        self.run: Any | None = None
        self.enabled = False

        telemetry_cfg = cfg.train.get("telemetry", {})
        wandb_cfg = telemetry_cfg.get("wandb", {})
        if not bool(wandb_cfg.get("enabled", False)):
            return
        if rank != 0:
            return

        try:
            import wandb  # type: ignore
        except Exception as exc:
            self.logger.warning("W&B is enabled but wandb is unavailable: %s", exc)
            return

        project_name = str(wandb_cfg.get("project_name", "big_weight_vae")).strip() or "big_weight_vae"
        entity = str(wandb_cfg.get("entity", "")).strip()
        run_name = str(wandb_cfg.get("run_name", "")).strip()
        run_mode = str(wandb_cfg.get("mode", "offline")).strip().lower() or "offline"
        run_dir = str(wandb_cfg.get("dir", "")).strip()
        log_code = bool(wandb_cfg.get("log_code", False))
        if run_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"train.telemetry.wandb.mode must be one of {{'online','offline','disabled'}}, got {run_mode!r}")

        tags_raw = wandb_cfg.get("tags", [])
        tags: list[str] = []
        if isinstance(tags_raw, (list, tuple, ListConfig)):
            tags = [str(tag) for tag in tags_raw]

        try:
            tracking_params = _build_external_tracking_params(cfg)
            run = wandb.init(
                project=project_name,
                entity=entity or None,
                name=run_name or None,
                mode=run_mode,
                dir=run_dir or None,
                config=tracking_params,
                tags=tags or None,
            )
            if run is None:
                return
            if log_code:
                try:
                    run.log_code(".")
                except Exception as exc:
                    self.logger.warning("W&B code log failed: %s", exc)
            self.run = run
            self.enabled = True
            self.logger.info(
                "W&B tracking enabled: project=%s entity=%s mode=%s",
                project_name,
                entity or "<default>",
                run_mode,
            )
        except Exception as exc:
            self.logger.warning("Failed to initialize W&B tracker: %s", exc)
            self.run = None
            self.enabled = False

    def log_parameters(self, params: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.config.update(params, allow_val_change=True)
        except Exception as exc:
            self.logger.warning("W&B parameter log failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=int(step))
        except Exception as exc:
            self.logger.warning("W&B metrics log failed at step=%s: %s", step, exc)

    def end(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass

__all__ = [
    '_build_external_tracking_params',
    'CometTracker',
    'WandbTracker',
]
