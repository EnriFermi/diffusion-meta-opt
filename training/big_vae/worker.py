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

from training.big_vae.checkpointing import *
from training.big_vae.data import *
from training.big_vae.grad_monitoring import *
from training.big_vae.runtime import *
from training.big_vae.tracking import *



def _run_worker(
    rank: int,
    world_size: int,
    cfg_dict: dict[str, Any],
    master_addr: str,
    master_port: int,
    monitor_queue: Any | None = None,
) -> None:
    print(f"[train_big_vae rank{rank}] worker bootstrapping", flush=True)
    maybe_redirect_stdio(cfg_dict, role="train_worker", section="train", rank=rank)
    cfg = OmegaConf.create(cfg_dict)
    _promote_run_profile_to_root(cfg)
    setup_logging(cfg, rank=rank)
    logger = _logger("train", rank=rank)
    maybe_enable_core_dumps(cfg_dict, section="train", logger=logger)
    monitor_send_event(
        monitor_queue,
        {
            "type": "register",
            "pid": int(os.getpid()),
            "role": f"train_rank_{rank}",
            "metadata": {
                "rank": int(rank),
                "world_size": int(world_size),
            },
        },
    )
    if rank == 0:
        logger.info("Starting training entrypoint")
        logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = _resolve_device(cfg, rank=rank, world_size=world_size)
    backend = _resolve_backend(cfg, device=device)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    seed = int(cfg.data.get("seed", 42)) + rank
    _seed_everything(seed)
    _set_speed_optimizations(cfg, device=device)

    is_distributed = world_size > 1
    streaming_mode = str(cfg.streaming.get("mode", "none")).lower()
    dataset_sharding = bool(cfg.train.get("use_dataset_sharding", True)) and is_distributed and streaming_mode != "none"
    use_broadcast = is_distributed and not dataset_sharding
    synthetic_layer_cfg = cfg.train.get("synthetic_layer_source", {})
    if synthetic_layer_cfg is None:
        synthetic_layer_cfg = {}
    if not isinstance(synthetic_layer_cfg, (dict, DictConfig)):
        raise TypeError("train.synthetic_layer_source must be a mapping")
    synthetic_layer_enabled = bool(synthetic_layer_cfg.get("enabled", False))
    synthetic_n_rows = max(1, int(synthetic_layer_cfg.get("n_rows", 256)))
    synthetic_d_in = max(1, int(synthetic_layer_cfg.get("d_in", 1024)))
    synthetic_d_out = max(1, int(synthetic_layer_cfg.get("d_out", 1024)))
    synthetic_x_std = float(synthetic_layer_cfg.get("x_std", 1.0))
    synthetic_w_std = float(synthetic_layer_cfg.get("w_std", 1.0))
    fixed_batch_cfg = cfg.train.get("fixed_training_batch", {})
    if fixed_batch_cfg is None:
        fixed_batch_cfg = {}
    if not isinstance(fixed_batch_cfg, (dict, DictConfig)):
        raise TypeError("train.fixed_training_batch must be a mapping")
    fixed_training_batch_enabled = bool(fixed_batch_cfg.get("enabled", False))
    offline_dataset_cfg = cfg.train.get("offline_dataset", {})
    if offline_dataset_cfg is None:
        offline_dataset_cfg = {}
    if not isinstance(offline_dataset_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset must be a mapping")
    offline_dataset_enabled = bool(offline_dataset_cfg.get("enabled", False))
    offline_dataset_root = str(offline_dataset_cfg.get("root_dir", "") or "").strip()
    offline_dataset_shard_by_rank = bool(offline_dataset_cfg.get("shard_by_rank", True))
    preslicing_cfg = cfg.train.get("preslicing", {})
    if preslicing_cfg is None:
        preslicing_cfg = {}
    if not isinstance(preslicing_cfg, (dict, DictConfig)):
        raise TypeError("train.preslicing must be a mapping")
    preslicing_enabled = bool(preslicing_cfg.get("enabled", False))
    preslicing_root = str(preslicing_cfg.get("root_dir", "") or "").strip()
    preslicing_shard_by_rank = bool(preslicing_cfg.get("shard_by_rank", True))
    if synthetic_x_std <= 0.0:
        raise ValueError(f"train.synthetic_layer_source.x_std must be > 0, got {synthetic_x_std}")
    if synthetic_w_std <= 0.0:
        raise ValueError(f"train.synthetic_layer_source.w_std must be > 0, got {synthetic_w_std}")
    if offline_dataset_enabled and not offline_dataset_root:
        raise ValueError("train.offline_dataset.root_dir must be set when train.offline_dataset.enabled=true")
    if preslicing_enabled and not offline_dataset_enabled:
        raise ValueError("train.preslicing.enabled=true requires train.offline_dataset.enabled=true")
    if offline_dataset_enabled and synthetic_layer_enabled:
        raise ValueError(
            "train.offline_dataset.enabled=true is incompatible with train.synthetic_layer_source.enabled=true"
        )
    if preslicing_enabled and fixed_training_batch_enabled:
        raise ValueError("train.preslicing.enabled=true is not supported together with train.fixed_training_batch.enabled=true")
    if synthetic_layer_enabled:
        dataset_sharding = False
        use_broadcast = False
    elif offline_dataset_enabled:
        streaming_mode = "presliced_big_vae" if preslicing_enabled else "offline_big_vae"
        active_offline_shard_by_rank = preslicing_shard_by_rank if preslicing_enabled else offline_dataset_shard_by_rank
        dataset_sharding = bool(cfg.train.get("use_dataset_sharding", True)) and is_distributed and active_offline_shard_by_rank
        use_broadcast = is_distributed and not dataset_sharding
    if preslicing_enabled and use_broadcast:
        raise ValueError("train.preslicing.enabled=true requires sharded loading in distributed mode")

    if dataset_sharding and not offline_dataset_enabled:
        cfg.streaming.distributed.enabled = True
        cfg.streaming.distributed.rank_env = "RANK"
        cfg.streaming.distributed.world_size_env = "WORLD_SIZE"
        cfg.streaming.distributed.shard_by = str(cfg.streaming.distributed.get("shard_by", "chunk"))
        base_cache_dir = str(
            cfg.streaming.consumer.get(
                "cache_dir",
                "./data/streaming/cache/consumer",
            )
        )
        cfg.streaming.consumer.cache_dir = str(Path(base_cache_dir) / f"rank_{rank}")

    logger.info(
        "Runtime: device=%s distributed=%s world_size=%s streaming_mode=%s dataset_sharding=%s",
        device,
        is_distributed,
        world_size,
        streaming_mode,
        dataset_sharding,
    )
    if offline_dataset_enabled:
        logger.info(
            "Offline BigVAE dataset enabled: root=%s shard_by_rank=%s use_broadcast=%s",
            offline_dataset_root,
            offline_dataset_shard_by_rank,
            use_broadcast,
        )
    if preslicing_enabled:
        logger.info(
            "Presliced BigVAE dataset enabled: root=%s num_slices=%s shard_by_rank=%s",
            preslicing_root or "<default>",
            int(preslicing_cfg.get("num_slices", 0)),
            preslicing_shard_by_rank,
        )
    if synthetic_layer_enabled:
        logger.info(
            "Synthetic layer source enabled: collectors disabled, x~N(0,%.4f), W~N(0,%.4f), n_rows=%s d_in=%s d_out=%s",
            synthetic_x_std,
            synthetic_w_std,
            synthetic_n_rows,
            synthetic_d_in,
            synthetic_d_out,
        )
    if fixed_training_batch_enabled:
        logger.info(
            "Fixed training batch enabled: one curriculum-sliced batch will be captured once and reused for the whole run"
        )

    model = None
    optimizer = None
    scheduler = None
    scaler = None
    comet_tracker: CometTracker | None = None
    wandb_tracker: WandbTracker | None = None

    failed = False
    try:
        collector: Any | None = None
        dataset: Any | None = None
        dataset_iter: Iterator[Any] | None = None
        dataset_loader: Any | None = None

        with contextlib.ExitStack() as stack:
            if not synthetic_layer_enabled:
                if offline_dataset_enabled:
                    offline_dataset_cfg = cfg.train.get("offline_dataset", {})
                    if offline_dataset_cfg is None:
                        offline_dataset_cfg = {}
                    if not isinstance(offline_dataset_cfg, (dict, DictConfig)):
                        raise TypeError("train.offline_dataset must be a mapping")
                    preslicing_cfg = cfg.train.get("preslicing", {})
                    if preslicing_cfg is None:
                        preslicing_cfg = {}
                    if not isinstance(preslicing_cfg, (dict, DictConfig)):
                        raise TypeError("train.preslicing must be a mapping")
                    active_loader_cfg = preslicing_cfg if preslicing_enabled else offline_dataset_cfg
                    loader_workers = max(0, int(active_loader_cfg.get("loader_workers", offline_dataset_cfg.get("loader_workers", 0))))
                    loader_batch_size = max(1, int(active_loader_cfg.get("loader_batch_size", offline_dataset_cfg.get("loader_batch_size", 1))))
                    loader_prefetch_factor = max(
                        1,
                        int(active_loader_cfg.get("loader_prefetch_factor", offline_dataset_cfg.get("loader_prefetch_factor", 2))),
                    )
                    loader_persistent_workers = bool(
                        active_loader_cfg.get("loader_persistent_workers", offline_dataset_cfg.get("loader_persistent_workers", True))
                    )
                    if preslicing_enabled:
                        if rank == 0:
                            logger.info("Ensuring presliced BigVAE dataset before opening training loader")
                            print(
                                "[train_big_vae rank0] ensuring presliced BigVAE dataset",
                                flush=True,
                            )
                            ensure_presliced_big_vae_dataset(cfg, logger=logger)
                        if is_distributed:
                            dist.barrier()
                    if rank == 0 or dataset_sharding:
                        offline_rank = rank if dataset_sharding else 0
                        offline_world_size = world_size if dataset_sharding else 1
                        if preslicing_enabled:
                            dataset, collector = stack.enter_context(
                                presliced_big_vae_data_pipeline(
                                    cfg,
                                    logger=logger,
                                    rank=offline_rank,
                                    world_size=offline_world_size,
                                )
                            )
                        else:
                            dataset, collector = stack.enter_context(
                                offline_big_vae_data_pipeline(
                                    cfg,
                                    logger=logger,
                                    rank=offline_rank,
                                    world_size=offline_world_size,
                                )
                            )
                        if loader_workers > 0:
                            effective_loader_batch_size = int(loader_batch_size)
                            dataset_loader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=(None if effective_loader_batch_size <= 1 else effective_loader_batch_size),
                                num_workers=loader_workers,
                                collate_fn=(
                                    _identity_sample_collate
                                    if effective_loader_batch_size <= 1
                                    else _sample_list_collate
                                ),
                                prefetch_factor=loader_prefetch_factor,
                                persistent_workers=loader_persistent_workers,
                                pin_memory=False,
                                worker_init_fn=_offline_loader_worker_init_fn,
                            )
                            dataset_loader_iter = iter(dataset_loader)
                            dataset_iter = (
                                dataset_loader_iter
                                if effective_loader_batch_size <= 1
                                else _flatten_loader_batches(dataset_loader_iter)
                            )
                            shutdown_workers = getattr(dataset_loader_iter, "_shutdown_workers", None)
                            if callable(shutdown_workers):
                                stack.callback(shutdown_workers)
                            if rank == 0:
                                logger.info(
                                    "%s DataLoader enabled: num_workers=%s batch_size=%s "
                                    "prefetch_factor=%s persistent_workers=%s",
                                    "Presliced dataset" if preslicing_enabled else "Offline dataset",
                                    loader_workers,
                                    effective_loader_batch_size,
                                    loader_prefetch_factor,
                                    loader_persistent_workers,
                                )
                        else:
                            dataset_iter = iter(dataset)
                else:
                    if rank == 0:
                        dataset, collector = stack.enter_context(
                            data_pipeline(
                                cfg,
                                logger=logger,
                                emit_run_report=True,
                                rank=rank,
                            )
                        )
                        dataset_iter = iter(dataset)
                    elif dataset_sharding:
                        # Consumer-only wrapper over shared chunk stream.
                        dataset, collector = stack.enter_context(
                            data_pipeline(
                                cfg,
                                start_collector=False,
                                predownload_models=False,
                                logger=logger,
                                emit_run_report=False,
                                rank=rank,
                            )
                        )
                        dataset_iter = iter(dataset)

            model_cfg = _build_model_cfg(cfg)
            model = build_weight_quantile_vae(model_cfg).to(device)
            total_params, trainable_params, frozen_params = _parameter_count_summary(model)
            if rank == 0:
                logger.info(
                    "Model params: total=%s trainable=%s frozen=%s",
                    f"{total_params:,}",
                    f"{trainable_params:,}",
                    f"{frozen_params:,}",
                )
            model = _maybe_compile(model, cfg=cfg, logger=logger)

            if is_distributed:
                if device.type == "cuda":
                    model = DDP(
                        model,
                        device_ids=[device.index],
                        output_device=device.index,
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                        gradient_as_bucket_view=True,
                    )
                else:
                    model = DDP(
                        model,
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                        gradient_as_bucket_view=True,
                    )

            optimizer = _build_optimizer(model=model, cfg=cfg, device=device)
            if rank == 0:
                group_summaries = []
                for group_idx, param_group in enumerate(optimizer.param_groups):
                    group_name = (
                        str(param_group.get("group_name", f"group_{group_idx}")).strip() or f"group_{group_idx}"
                    )
                    group_param_count = sum(int(param.numel()) for param in param_group.get("params", []))
                    group_summaries.append(
                        f"{group_name}:params={group_param_count:,},lr={float(param_group['lr']):.6e},"
                        f"wd={float(param_group.get('weight_decay', 0.0)):.6e}"
                    )
                logger.info("Optimizer param groups: %s", "; ".join(group_summaries))
            scheduler = _build_scheduler(optimizer=optimizer, cfg=cfg)
            comet_tracker = CometTracker(cfg=cfg, logger=logger, rank=rank)
            wandb_tracker = WandbTracker(cfg=cfg, logger=logger, rank=rank)
            param_count_payload = {
                "model.param_count.total": int(total_params),
                "model.param_count.trainable": int(trainable_params),
                "model.param_count.frozen": int(frozen_params),
            }
            if comet_tracker is not None and comet_tracker.enabled:
                comet_tracker.log_parameters(param_count_payload)
            if wandb_tracker is not None and wandb_tracker.enabled:
                wandb_tracker.log_parameters(param_count_payload)

            stage_num = max(1, int(cfg.train.get("stage", 1)))
            checkpoint_every = max(1, int(cfg.train.get("checkpoint_every", 200)))
            resume_state_cfg = cfg.train.get("resume_state", {})
            if resume_state_cfg is None:
                resume_state_cfg = {}
            if not isinstance(resume_state_cfg, (dict, DictConfig)):
                raise TypeError("train.resume_state must be a mapping")
            resume_state_enabled = bool(resume_state_cfg.get("enabled", False))
            resume_state_auto_resume = bool(resume_state_cfg.get("auto_resume", True))
            resume_state_save_every = max(1, int(resume_state_cfg.get("save_every", checkpoint_every)))
            resume_state_load_policy = _resume_state_load_policy(resume_state_cfg)
            default_resume_state_dir = (
                Path(str(cfg.train.get("checkpoint_dir", "./checkpoints/weight_quantile_vae")))
                / f"stage_{stage_num}"
                / "resume_state"
            )
            resume_state_dir = Path(str(resume_state_cfg.get("dir", str(default_resume_state_dir))))

            amp_enabled, amp_dtype = _resolve_amp(cfg=cfg, device=device)
            scaler = runtime_create_grad_scaler(
                device=device,
                enabled=(amp_enabled and amp_dtype == torch.float16),
            )
            resume_checkpoint = str(cfg.train.get("resume_checkpoint", "")).strip()
            resumed_training_step = 0
            resume_state_path: Path | None = None
            if resume_state_enabled and resume_state_auto_resume:
                logger.info("Auto-resume probe: dir=%s", resume_state_dir)
                resume_state_path = _find_latest_resume_state_checkpoint(resume_state_dir)
                if resume_state_path is not None:
                    logger.info("Auto-resume candidate found: %s", resume_state_path)
                else:
                    logger.info("Auto-resume candidate not found in %s; starting from scratch", resume_state_dir)
            if resume_state_path is not None:
                resumed_training_step = _load_training_state_from_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    path=resume_state_path,
                    logger=logger,
                    load_model_state=resume_state_load_policy["load_model_state"],
                    load_optimizer_state=resume_state_load_policy["load_optimizer_state"],
                    load_scheduler_state=resume_state_load_policy["load_scheduler_state"],
                    load_scaler_state=resume_state_load_policy["load_scaler_state"],
                    load_step=resume_state_load_policy["load_step"],
                )
            elif resume_checkpoint:
                logger.info(
                    "Resume-state unavailable; loading model weights only from resume_checkpoint=%s",
                    resume_checkpoint,
                )
                _load_model_weights_from_checkpoint(model, resume_checkpoint, logger)
            else:
                logger.info("No resume source configured or found; starting training from step=0")
            cudagraph_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
            use_cudagraph_step_begin = (
                bool(cfg.train.get("compile", False))
                and device.type == "cuda"
                and callable(cudagraph_step_begin)
            )

            max_steps = max(1, int(cfg.train.get("max_steps", 1000)))
            grad_accum_steps = max(1, int(cfg.train.get("grad_accum_steps", 1)))
            kl_beta = float(cfg.train.get("kl_beta", 1e-3))
            kl_schedule_cfg = cfg.train.get("kl_schedule", {})
            if kl_schedule_cfg is None:
                kl_schedule_cfg = {}
            if not isinstance(kl_schedule_cfg, (dict, DictConfig)):
                raise TypeError("train.kl_schedule must be a mapping")
            kl_schedule_enabled = bool(kl_schedule_cfg.get("enabled", False))
            kl_schedule_start_beta = float(kl_schedule_cfg.get("start_beta", 0.0))
            kl_schedule_warmup_steps = max(0, int(kl_schedule_cfg.get("warmup_steps", 0)))
            kl_schedule_ramp_steps = max(0, int(kl_schedule_cfg.get("ramp_steps", 0)))
            model_unwrapped = model.module if isinstance(model, DDP) else model
            cfg_holder = model_unwrapped
            if not hasattr(cfg_holder, "cfg") and hasattr(cfg_holder, "_orig_mod"):
                cfg_holder = getattr(cfg_holder, "_orig_mod")
            if not hasattr(cfg_holder, "cfg"):
                raise AttributeError(f"Model does not expose cfg: type={type(model_unwrapped)}")
            use_latent_sampling = bool(cfg_holder.cfg.big_vae.use_latent_sampling)
            latent_prior_kind = str(getattr(cfg_holder.cfg.big_vae, "latent_prior_kind", "gaussian"))
            latent_sampling_gate_cfg = cfg.train.get("latent_sampling_gate", {})
            if latent_sampling_gate_cfg is None:
                latent_sampling_gate_cfg = {}
            if not isinstance(latent_sampling_gate_cfg, (dict, DictConfig)):
                raise TypeError("train.latent_sampling_gate must be a mapping when provided")
            latent_sampling_gate_enabled = bool(latent_sampling_gate_cfg.get("enabled", True))
            latent_sampling_gate_start_step = max(
                0,
                int(latent_sampling_gate_cfg.get("start_step", kl_schedule_warmup_steps)),
            )
            latent_sampling_gate_ramp_steps = max(
                0,
                int(latent_sampling_gate_cfg.get("ramp_steps", kl_schedule_ramp_steps)),
            )
            latent_sampling_gate_start_value = float(latent_sampling_gate_cfg.get("start_value", 1e-4))
            latent_sampling_gate_end_value = float(latent_sampling_gate_cfg.get("end_value", 1.0))
            latent_sampling_gate_start_value = max(0.0, min(1.0, latent_sampling_gate_start_value))
            latent_sampling_gate_end_value = max(0.0, min(1.0, latent_sampling_gate_end_value))
            behavioral_coef = float(cfg.train.get("behavioral_coef", 1.0))
            structural_coef = float(cfg.train.get("structural_coef", 0.5))
            behavioral_loss_cfg = cfg.train.get("behavioral_loss", {})
            if behavioral_loss_cfg is None:
                behavioral_loss_cfg = {}
            if not isinstance(behavioral_loss_cfg, (dict, DictConfig)):
                raise TypeError("train.behavioral_loss must be a mapping when provided")
            behavioral_lambda_operator = float(behavioral_loss_cfg.get("lambda_operator", 1.0))
            behavioral_lambda_dir = float(behavioral_loss_cfg.get("lambda_dir", 0.0))
            behavioral_lambda_scale = float(behavioral_loss_cfg.get("lambda_scale", 0.0))
            behavioral_gamma = float(behavioral_loss_cfg.get("gamma", 0.5))
            behavioral_huber_delta = float(behavioral_loss_cfg.get("huber_delta", 0.1))
            struct_loss_cfg = cfg.train.get("struct_loss", {})
            if struct_loss_cfg is None:
                struct_loss_cfg = {}
            struct_gamma = float(struct_loss_cfg.get("gamma", 0.5))
            struct_lambda_dir = float(struct_loss_cfg.get("lambda_dir", 1.0))
            struct_lambda_scale = float(struct_loss_cfg.get("lambda_scale", 0.25))
            struct_lambda_rec = float(struct_loss_cfg.get("lambda_rec", 0.5))
            struct_lambda_rel = float(struct_loss_cfg.get("lambda_rel", 0.1))
            struct_huber_delta = float(struct_loss_cfg.get("huber_delta", 0.1))
            grad_clip_norm = float(cfg.train.get("grad_clip_norm", 1.0))
            grad_clip_norm_by_part_raw = cfg.train.get("grad_clip_norm_by_part", {})
            if grad_clip_norm_by_part_raw is None:
                grad_clip_norm_by_part_raw = {}
            if not isinstance(grad_clip_norm_by_part_raw, (dict, DictConfig)):
                raise TypeError("train.grad_clip_norm_by_part must be a mapping when provided")
            grad_group_prefixes = _grad_stat_group_prefixes()
            grad_clip_norm_by_part: dict[str, float] = {
                group_name: float(grad_clip_norm) for group_name in grad_group_prefixes
            }
            partwise_grad_clip_enabled = False
            unknown_clip_groups: list[str] = []
            for key, value in grad_clip_norm_by_part_raw.items():
                group_name = str(key).strip()
                if not group_name:
                    continue
                if group_name not in grad_clip_norm_by_part:
                    unknown_clip_groups.append(group_name)
                    continue
                grad_clip_norm_by_part[group_name] = float(value)
                partwise_grad_clip_enabled = True
            max_x_rows = int(cfg.train.get("max_x_rows", 0))

            patch_size_for_slice = int(cfg.model.get("patch_size", 16))
            curriculum_max_T, curriculum_max_d_out = _compute_curriculum_slice_sizes(cfg)
            slice_batch_size = max(1, int(cfg.train.get("slice_batch_size", 1)))
            stable_batch_target_x_rows, stable_batch_target_d_in, stable_batch_target_d_out = _stable_batch_shape_targets(
                cfg=cfg,
                patch_size=patch_size_for_slice,
                max_T_patches=curriculum_max_T,
                max_d_out=curriculum_max_d_out,
                max_x_rows=max_x_rows,
            )
            batch_source_mixing_cfg = cfg.train.get("batch_source_mixing", {})
            if batch_source_mixing_cfg is None:
                batch_source_mixing_cfg = {}
            if not isinstance(batch_source_mixing_cfg, (dict, DictConfig)):
                raise TypeError("train.batch_source_mixing must be a mapping")
            batch_source_mixing_enabled = bool(batch_source_mixing_cfg.get("enabled", False))
            batch_source_mixing_strategy = str(batch_source_mixing_cfg.get("strategy", "round_robin")).strip().lower()
            if batch_source_mixing_strategy != "round_robin":
                raise ValueError(
                    "train.batch_source_mixing.strategy must be 'round_robin', "
                    f"got {batch_source_mixing_strategy!r}"
                )
            batch_source_mixing_uniqueness = _normalize_batch_source_uniqueness(
                batch_source_mixing_cfg.get("uniqueness", "none")
            )
            consume_slices_without_replacement = bool(
                batch_source_mixing_cfg.get("consume_slices_without_replacement", False)
            )
            raw_max_source_samples = batch_source_mixing_cfg.get("max_source_samples", 0)
            if raw_max_source_samples is None:
                parsed_max_source_samples = 0
            elif isinstance(raw_max_source_samples, str) and raw_max_source_samples.strip().lower() in {"", "none", "null"}:
                parsed_max_source_samples = 0
            else:
                parsed_max_source_samples = int(raw_max_source_samples)
            if parsed_max_source_samples < 0:
                raise ValueError("train.batch_source_mixing.max_source_samples must be >= 0")
            if batch_source_mixing_enabled:
                requested_source_samples_per_refresh = (
                    slice_batch_size if parsed_max_source_samples == 0 else min(slice_batch_size, parsed_max_source_samples)
                )
            else:
                requested_source_samples_per_refresh = 1
            batch_source_mixing_active = batch_source_mixing_enabled and requested_source_samples_per_refresh > 1
            max_active_source_pool_size = max(slice_batch_size, requested_source_samples_per_refresh)
            min_mixed_source_d_in = int(patch_size_for_slice) if batch_source_mixing_active else 0
            min_mixed_source_d_out = (
                1
                if batch_source_mixing_active and consume_slices_without_replacement
                else (int(curriculum_max_d_out) if batch_source_mixing_active else 0)
            )
            deferred_source_cache_size = max(16, int(requested_source_samples_per_refresh) * 8)
            if batch_source_mixing_uniqueness != "none" and use_broadcast:
                raise ValueError(
                    "train.batch_source_mixing.uniqueness != 'none' is not supported together with distributed "
                    "broadcast-based data loading"
                )
            if rank == 0:
                logger.info(
                    "Curriculum slicing: stage=%s max_T_patches=%s max_d_out=%s patch_size=%s slice_batch_size=%s",
                    stage_num, curriculum_max_T, curriculum_max_d_out, patch_size_for_slice, slice_batch_size,
                )
                if (
                    stable_batch_target_x_rows is not None
                    or stable_batch_target_d_in is not None
                    or stable_batch_target_d_out is not None
                ):
                    logger.info(
                        "Compile-stable batch shapes enabled: target_x_rows=%s target_d_in=%s target_d_out=%s",
                        stable_batch_target_x_rows,
                        stable_batch_target_d_in,
                        stable_batch_target_d_out,
                    )
                if batch_source_mixing_enabled:
                    logger.info(
                        "Batch source mixing: enabled=%s strategy=%s requested_source_samples=%s "
                        "uniqueness=%s min_source_shape_for_full_mixing=(d_in>=%s,d_out>=%s) "
                        "collector_pressure_vs_single~%sx",
                        batch_source_mixing_active,
                        batch_source_mixing_strategy,
                        requested_source_samples_per_refresh,
                        batch_source_mixing_uniqueness,
                        min_mixed_source_d_in,
                        min_mixed_source_d_out,
                        requested_source_samples_per_refresh,
                    )
                if consume_slices_without_replacement:
                    logger.info(
                        "Source slice consumption without replacement is enabled: source snapshots remain in the "
                        "local pool until exhausted, and train.steps_per_sample is ignored for refresh decisions"
                    )
            offline_batch_prefetch_cfg = cfg.train.get("offline_batch_prefetch", {})
            if offline_batch_prefetch_cfg is None:
                offline_batch_prefetch_cfg = {}
            if not isinstance(offline_batch_prefetch_cfg, (dict, DictConfig)):
                raise TypeError("train.offline_batch_prefetch must be a mapping")
            offline_batch_prefetch_requested = bool(offline_batch_prefetch_cfg.get("enabled", True))
            offline_batch_prefetch_queue_size = max(1, int(offline_batch_prefetch_cfg.get("queue_size", 2)))
            offline_batch_prefetch_pin_memory = bool(offline_batch_prefetch_cfg.get("pin_memory", True))
            raw_offline_batch_prefetch_cpu_threads = int(offline_batch_prefetch_cfg.get("cpu_threads", 0))
            offline_batch_prefetch_cpu_threads = (
                max(1, min(16, int(os.cpu_count() or 1)))
                if raw_offline_batch_prefetch_cpu_threads <= 0
                else max(1, raw_offline_batch_prefetch_cpu_threads)
            )
            raw_refill_fetch_batch_size = int(offline_batch_prefetch_cfg.get("refill_fetch_batch_size", 0))
            offline_batch_prefetch_refill_fetch_batch_size = (
                max(1, min(16, int(requested_source_samples_per_refresh)))
                if raw_refill_fetch_batch_size <= 0
                else max(1, raw_refill_fetch_batch_size)
            )
            offline_batch_prefetch_reasons: list[str] = []
            if not offline_batch_prefetch_requested:
                offline_batch_prefetch_reasons.append("disabled_by_config")
            if not offline_dataset_enabled:
                offline_batch_prefetch_reasons.append("offline_dataset_disabled")
            if fixed_training_batch_enabled:
                offline_batch_prefetch_reasons.append("fixed_training_batch_enabled")
            if synthetic_layer_enabled:
                offline_batch_prefetch_reasons.append("synthetic_layer_enabled")
            if use_broadcast:
                offline_batch_prefetch_reasons.append("distributed_broadcast_mode")
            offline_batch_prefetch_active = len(offline_batch_prefetch_reasons) == 0
            if offline_batch_prefetch_active:
                torch.set_num_threads(offline_batch_prefetch_cpu_threads)
            if rank == 0:
                if offline_batch_prefetch_active:
                    logger.info(
                        "Offline batch prefetch enabled: queue_size=%s pin_memory=%s cpu_threads=%s refill_fetch_batch_size=%s",
                        offline_batch_prefetch_queue_size,
                        offline_batch_prefetch_pin_memory,
                        offline_batch_prefetch_cpu_threads,
                        offline_batch_prefetch_refill_fetch_batch_size,
                    )
                else:
                    logger.info(
                        "Offline batch prefetch disabled: reasons=%s",
                        ",".join(offline_batch_prefetch_reasons) or "none",
                    )

            log_every = max(1, int(cfg.train.get("log_every", 10)))
            log_worker_status_every = max(
                1,
                int(cfg.train.get("log_worker_status_every", log_every)),
            )
            forensics_cfg = cfg.train.get("forensics", {})
            forensics_heartbeat_steps = max(
                1,
                int(forensics_cfg.get("heartbeat_steps", log_every)),
            )
            telemetry_cfg = cfg.train.get("telemetry", {})
            if not isinstance(telemetry_cfg, (dict, DictConfig)):
                raise TypeError("train.telemetry must be a mapping")
            grad_layer_monitor_cfg = telemetry_cfg.get("grad_layer_monitor", {})
            if not isinstance(grad_layer_monitor_cfg, (dict, DictConfig)):
                raise TypeError("train.telemetry.grad_layer_monitor must be a mapping")
            grad_layer_monitor_enabled = bool(grad_layer_monitor_cfg.get("enabled", True)) and rank == 0
            grad_layer_monitor_every_steps = max(
                1,
                int(grad_layer_monitor_cfg.get("every_steps", log_every)),
            )
            grad_layer_monitor_weights_only = bool(grad_layer_monitor_cfg.get("weights_only", True))
            grad_layer_monitor_topk_layers = max(1, int(grad_layer_monitor_cfg.get("topk_layers", 24)))
            grad_layer_monitor_log_scale = bool(grad_layer_monitor_cfg.get("log_scale", True))
            grad_layer_monitor_save_csv = bool(grad_layer_monitor_cfg.get("save_csv", True))
            grad_layer_monitor_reset_csv_on_start = bool(grad_layer_monitor_cfg.get("reset_csv_on_start", True))
            grad_layer_monitor_save_plot = bool(grad_layer_monitor_cfg.get("save_plot", True))
            grad_layer_monitor_plot_every_steps = max(0, int(grad_layer_monitor_cfg.get("plot_every_steps", 200)))
            grad_layer_monitor_save_heatmap = bool(grad_layer_monitor_cfg.get("save_heatmap", True))
            grad_layer_monitor_heatmap_max_layers = max(1, int(grad_layer_monitor_cfg.get("heatmap_max_layers", 96)))
            grad_layer_monitor_include_prefixes_raw = grad_layer_monitor_cfg.get(
                "include_prefixes",
                [],
            )
            grad_layer_monitor_include_prefixes_list: list[str] = []
            if isinstance(grad_layer_monitor_include_prefixes_raw, (list, tuple, ListConfig)):
                for item in grad_layer_monitor_include_prefixes_raw:
                    text = str(item).strip()
                    if text:
                        grad_layer_monitor_include_prefixes_list.append(text)
            else:
                text = str(grad_layer_monitor_include_prefixes_raw).strip()
                if text:
                    grad_layer_monitor_include_prefixes_list.append(text)
            grad_layer_monitor_include_prefixes = tuple(grad_layer_monitor_include_prefixes_list)
            monitor_base_dir = Path(str(cfg.train.get("checkpoint_dir", "./checkpoints/weight_quantile_vae"))) / f"stage_{stage_num}"
            grad_layer_monitor_csv_path = Path(
                str(grad_layer_monitor_cfg.get("csv_path", str(monitor_base_dir / "grad_layer_rms.csv")))
            )
            grad_layer_monitor_plot_path = Path(
                str(grad_layer_monitor_cfg.get("plot_path", str(monitor_base_dir / "grad_layer_rms.png")))
            )
            grad_layer_monitor_heatmap_path = Path(
                str(grad_layer_monitor_cfg.get("heatmap_path", str(monitor_base_dir / "grad_layer_rms_heatmap.png")))
            )
            params_by_grad_group: dict[str, list[nn.Parameter]] = {}
            if partwise_grad_clip_enabled:
                params_by_grad_group = _collect_params_by_grad_group(model=model, groups=grad_group_prefixes)
            if grad_layer_monitor_enabled and grad_layer_monitor_save_csv and grad_layer_monitor_reset_csv_on_start:
                try:
                    if grad_layer_monitor_csv_path.exists():
                        grad_layer_monitor_csv_path.unlink()
                except Exception as exc:
                    logger.warning("Could not reset grad-layer CSV at %s: %s", grad_layer_monitor_csv_path, exc)

            steps_per_sample = max(1, int(cfg.train.get("steps_per_sample", 1)))
            if rank == 0:
                logger.info("Steps per sample: %s", steps_per_sample)
                if resume_state_enabled:
                    logger.info(
                        "Resume-state checkpointing: dir=%s save_every=%s auto_resume=%s "
                        "load_model_state=%s load_optimizer_state=%s load_scheduler_state=%s "
                        "load_scaler_state=%s load_step=%s",
                        resume_state_dir,
                        resume_state_save_every,
                        resume_state_auto_resume,
                        resume_state_load_policy["load_model_state"],
                        resume_state_load_policy["load_optimizer_state"],
                        resume_state_load_policy["load_scheduler_state"],
                        resume_state_load_policy["load_scaler_state"],
                        resume_state_load_policy["load_step"],
                    )
                    if (
                        resume_state_load_policy["load_step"]
                        != resume_state_load_policy["load_scheduler_state"]
                    ):
                        logger.warning(
                            "Resume-state config mismatch: load_step=%s but load_scheduler_state=%s. "
                            "This can desync logged global_step from LR schedule state.",
                            resume_state_load_policy["load_step"],
                            resume_state_load_policy["load_scheduler_state"],
                        )
                if kl_schedule_enabled:
                    logger.info(
                        "BigVAE latent mode: %s (use_latent_sampling=%s, prior=%s, kl_beta_target=%s, "
                        "kl_schedule=start@%.6f warmup=%s ramp=%s)",
                        "VAE" if use_latent_sampling else "AE",
                        use_latent_sampling,
                        latent_prior_kind,
                        kl_beta,
                        kl_schedule_start_beta,
                        kl_schedule_warmup_steps,
                        kl_schedule_ramp_steps,
                    )
                else:
                    logger.info(
                        "BigVAE latent mode: %s (use_latent_sampling=%s, prior=%s, kl_beta=%s)",
                        "VAE" if use_latent_sampling else "AE",
                        use_latent_sampling,
                        latent_prior_kind,
                        kl_beta,
                    )
                if use_latent_sampling:
                    initial_latent_sampling_gate = _compute_latent_sampling_gate_for_step(
                        resumed_training_step,
                        schedule_enabled=latent_sampling_gate_enabled,
                        start_step=latent_sampling_gate_start_step,
                        ramp_steps=latent_sampling_gate_ramp_steps,
                        start_value=latent_sampling_gate_start_value,
                        end_value=latent_sampling_gate_end_value,
                    )
                    _set_model_latent_sampling_gate(model, initial_latent_sampling_gate)
                    logger.info(
                        "BigVAE latent sampling gate: enabled=%s start_step=%s ramp_steps=%s "
                        "start_value=%.6g end_value=%.6g current_at_resume=%.6g",
                        latent_sampling_gate_enabled,
                        latent_sampling_gate_start_step,
                        latent_sampling_gate_ramp_steps,
                        latent_sampling_gate_start_value,
                        latent_sampling_gate_end_value,
                        initial_latent_sampling_gate,
                    )
                if unknown_clip_groups:
                    logger.warning(
                        "Ignoring unknown train.grad_clip_norm_by_part groups: %s",
                        sorted(set(unknown_clip_groups)),
                    )
                if partwise_grad_clip_enabled:
                    params_with_grad_per_group = {
                        group_name: int(len(params_by_grad_group.get(group_name, [])))
                        for group_name in grad_group_prefixes
                    }
                    logger.info(
                        "Per-part grad clipping enabled: clip_norms=%s params_per_group=%s",
                        grad_clip_norm_by_part,
                        params_with_grad_per_group,
                    )
                if grad_layer_monitor_enabled:
                    logger.info(
                        "Grad-layer monitor enabled: every_steps=%s topk=%s csv=%s plot=%s heatmap=%s "
                        "weights_only=%s include_prefixes=%s",
                        grad_layer_monitor_every_steps,
                        grad_layer_monitor_topk_layers,
                        str(grad_layer_monitor_csv_path.resolve()) if grad_layer_monitor_save_csv else "<off>",
                        str(grad_layer_monitor_plot_path.resolve()) if grad_layer_monitor_save_plot else "<off>",
                        str(grad_layer_monitor_heatmap_path.resolve())
                        if (grad_layer_monitor_save_plot and grad_layer_monitor_save_heatmap)
                        else "<off>",
                        grad_layer_monitor_weights_only,
                        list(grad_layer_monitor_include_prefixes),
                    )

            loss_window = 0.0
            behavioral_window = 0.0
            behavioral_operator_window = 0.0
            behavioral_dir_window = 0.0
            behavioral_scale_window = 0.0
            structural_window = 0.0
            kl_window = 0.0
            struct_dir_window = 0.0
            struct_scale_window = 0.0
            struct_rec_window = 0.0
            struct_rel_window = 0.0
            data_build_window_s = 0.0
            data_wait_window_s = 0.0
            data_h2d_window_s = 0.0
            data_prefetch_depth_window = 0.0
            source_diversity_window_unique_models_sum = 0.0
            source_diversity_window_unique_models_min = math.inf
            source_diversity_window_unique_models_max = 0.0
            source_diversity_window_target_coverage_sum = 0.0
            source_diversity_window_model_perplexity_sum = 0.0
            source_diversity_window_shortfall_steps = 0
            source_diversity_latest: dict[str, float] | None = None
            window_steps = 0
            t0 = time.time()
            grad_layer_history: dict[str, list[tuple[int, float]]] = {}
            latest_grad_layer_snapshot: dict[str, Any] = {
                "step": 0,
                "num_layers": 0,
                "clip_coef": 1.0,
                "global_grad_rms_pre_clip": 0.0,
                "global_grad_rms_post_clip": 0.0,
                "global_param_rms": 0.0,
                "global_grad_to_param_ratio_pre_clip": 0.0,
                "global_grad_to_param_ratio_post_clip": 0.0,
                "top_layers_pre_clip": [],
                "low_layers_pre_clip": [],
                "top_layers_ratio_pre_clip": [],
            }

            current_source_samples: list[SourceSampleRecord] | None = None
            current_source_states: list[SourceSliceState] | None = None
            deferred_source_samples: deque[SourceSampleRecord] = deque()
            current_source_round_robin_offset = 0
            fixed_batch_x: torch.Tensor | None = None
            fixed_batch_W: torch.Tensor | None = None
            fixed_batch_x_mask: torch.Tensor | None = None
            fixed_batch_d_in_mask: torch.Tensor | None = None
            fixed_batch_d_out_mask: torch.Tensor | None = None
            fixed_batch_diversity_stats: dict[str, float] | None = None
            direction_pre_norm_stats_latest: dict[str, Any] | None = None
            source_mixing_shortfall_logged = False
            prefetch_request_iter = iter(
                (step_idx, micro_idx)
                for step_idx in range(resumed_training_step, max_steps)
                for micro_idx in range(grad_accum_steps)
            )

            def _prepare_training_micro_batch_cpu(*, step_idx: int, micro_idx: int) -> PreparedTrainingBatch:
                nonlocal current_source_samples
                nonlocal current_source_states
                nonlocal current_source_round_robin_offset
                nonlocal source_mixing_shortfall_logged

                build_t0 = time.perf_counter()
                if preslicing_enabled:
                    batch = _fetch_presliced_training_batch_cpu(
                        dataset_iter=dataset_iter,
                        batch_size=slice_batch_size,
                        logger=logger,
                    )
                elif consume_slices_without_replacement:
                    current_source_states, current_source_round_robin_offset = _prune_exhausted_source_states_with_offset(
                        current_source_states,
                        current_source_round_robin_offset,
                    )
                    current_source_states = _ensure_source_state_pool_capacity(
                        current_source_states=current_source_states,
                        required_remaining_slices=slice_batch_size,
                        target_source_pool_size=requested_source_samples_per_refresh,
                        max_active_source_pool_size=max_active_source_pool_size,
                        rank=rank,
                        device=device,
                        dataset_iter=dataset_iter,
                        use_broadcast=use_broadcast,
                        max_x_rows=max_x_rows,
                        logger=logger,
                        uniqueness=batch_source_mixing_uniqueness if batch_source_mixing_active else "none",
                        deferred_samples=deferred_source_samples,
                        max_deferred_samples=deferred_source_cache_size,
                        refill_fetch_batch_size=offline_batch_prefetch_refill_fetch_batch_size,
                        min_d_in=min_mixed_source_d_in,
                        min_d_out=min_mixed_source_d_out,
                        max_T_patches=curriculum_max_T,
                        curriculum_max_d_out=curriculum_max_d_out,
                        patch_size=patch_size_for_slice,
                        synthetic_layer_enabled=synthetic_layer_enabled,
                        synthetic_n_rows=synthetic_n_rows,
                        synthetic_d_in=synthetic_d_in,
                        synthetic_d_out=synthetic_d_out,
                        synthetic_x_std=synthetic_x_std,
                        synthetic_w_std=synthetic_w_std,
                    )
                    if not current_source_states:
                        raise RuntimeError("training step requires a loaded source sample")
                    if (
                        batch_source_mixing_active
                        and len(current_source_states) < requested_source_samples_per_refresh
                        and not source_mixing_shortfall_logged
                        and rank == 0
                    ):
                        logger.info(
                            "Batch source mixing shortfall: requested=%s compatible source samples, fetched=%s. "
                            "Training continues with reduced diversity for this refresh.",
                            requested_source_samples_per_refresh,
                            len(current_source_states),
                        )
                        source_mixing_shortfall_logged = True
                    batch_payload = _build_training_batch_from_source_states(
                        current_source_states,
                        batch_size=slice_batch_size,
                        start_offset=current_source_round_robin_offset,
                        target_x_rows=stable_batch_target_x_rows,
                        target_d_in=stable_batch_target_d_in,
                        target_d_out=stable_batch_target_d_out,
                    )
                    current_batch_source_diversity = _compute_consumed_batch_source_diversity_stats(
                        current_source_states,
                        used_source_indices=batch_payload.used_source_indices,
                        source_pool_remaining_slices_pre=batch_payload.source_pool_remaining_slices_pre,
                        source_pool_remaining_slices_post=batch_payload.source_pool_remaining_slices_post,
                    )
                    current_source_states, current_source_round_robin_offset = _prune_exhausted_source_states_with_offset(
                        current_source_states,
                        batch_payload.next_start_offset,
                    )
                    batch = PreparedTrainingBatch(
                        W=batch_payload.W,
                        x=batch_payload.x,
                        x_mask=batch_payload.x_mask,
                        d_in_mask=batch_payload.d_in_mask,
                        d_out_mask=batch_payload.d_out_mask,
                        source_diversity=current_batch_source_diversity,
                        build_time_s=time.perf_counter() - build_t0,
                    )
                else:
                    if current_source_samples is None or (
                        micro_idx == 0 and step_idx % steps_per_sample == 0
                    ):
                        if synthetic_layer_enabled:
                            synthetic_source_device = torch.device("cpu")
                            current_source_samples = []
                            for idx in range(requested_source_samples_per_refresh):
                                x_syn, W_syn = _sample_synthetic_layer(
                                    device=synthetic_source_device,
                                    n_rows=synthetic_n_rows,
                                    d_in=synthetic_d_in,
                                    d_out=synthetic_d_out,
                                    x_std=synthetic_x_std,
                                    w_std=synthetic_w_std,
                                    max_x_rows=max_x_rows,
                                )
                                current_source_samples.append(
                                    SourceSampleRecord(
                                        x=x_syn,
                                        W=W_syn,
                                        model_name=f"synthetic_{idx}",
                                    )
                                )
                        else:
                            if batch_source_mixing_active:
                                current_source_samples = _fetch_source_samples(
                                    rank=rank,
                                    device=device,
                                    dataset_iter=dataset_iter,
                                    use_broadcast=use_broadcast,
                                    max_x_rows=max_x_rows,
                                    logger=logger,
                                    num_samples=requested_source_samples_per_refresh,
                                    uniqueness=batch_source_mixing_uniqueness,
                                    deferred_samples=deferred_source_samples,
                                    max_deferred_samples=deferred_source_cache_size,
                                    min_d_in=min_mixed_source_d_in,
                                    min_d_out=min_mixed_source_d_out,
                                )
                            else:
                                current_source_samples = [
                                    _fetch_source_sample_record_cpu(
                                        rank=rank,
                                        device=device,
                                        dataset_iter=dataset_iter,
                                        use_broadcast=use_broadcast,
                                        max_x_rows=max_x_rows,
                                        logger=logger,
                                    )
                                ]
                        current_source_round_robin_offset = 0
                        if (
                            batch_source_mixing_active
                            and current_source_samples is not None
                            and len(current_source_samples) < requested_source_samples_per_refresh
                            and not source_mixing_shortfall_logged
                            and rank == 0
                        ):
                            logger.info(
                                "Batch source mixing shortfall: requested=%s compatible source samples, fetched=%s. "
                                "Training continues with reduced diversity for this refresh.",
                                requested_source_samples_per_refresh,
                                len(current_source_samples),
                            )
                            source_mixing_shortfall_logged = True

                    if not current_source_samples:
                        raise RuntimeError("training step requires a loaded source sample")
                    current_batch_source_diversity = _compute_batch_source_diversity_stats(
                        current_source_samples,
                        batch_size=slice_batch_size,
                        start_offset=current_source_round_robin_offset,
                    )
                    W_s, x_s, x_mask_s, d_in_mask_s, d_out_mask_s = _build_training_batch_from_source_samples(
                        current_source_samples,
                        max_T_patches=curriculum_max_T,
                        max_d_out=curriculum_max_d_out,
                        patch_size=patch_size_for_slice,
                        batch_size=slice_batch_size,
                        start_offset=current_source_round_robin_offset,
                        target_x_rows=stable_batch_target_x_rows,
                        target_d_in=stable_batch_target_d_in,
                        target_d_out=stable_batch_target_d_out,
                    )
                    current_source_round_robin_offset = (
                        current_source_round_robin_offset + slice_batch_size
                    ) % len(current_source_samples)
                    batch = PreparedTrainingBatch(
                        W=W_s,
                        x=x_s,
                        x_mask=x_mask_s,
                        d_in_mask=d_in_mask_s,
                        d_out_mask=d_out_mask_s,
                        source_diversity=current_batch_source_diversity,
                        build_time_s=time.perf_counter() - build_t0,
                    )
                if offline_batch_prefetch_pin_memory and device.type == "cuda":
                    batch = _pin_prepared_training_batch(batch)
                return batch

            def _prepare_training_micro_batch_cpu_for_prefetch() -> PreparedTrainingBatch:
                step_idx, micro_idx = next(prefetch_request_iter)
                return _prepare_training_micro_batch_cpu(step_idx=step_idx, micro_idx=micro_idx)

            offline_batch_prefetcher: BackgroundPrefetcher[PreparedTrainingBatch] | None = None
            if offline_batch_prefetch_active:
                offline_batch_prefetcher = BackgroundPrefetcher(
                    build_fn=_prepare_training_micro_batch_cpu_for_prefetch,
                    queue_size=offline_batch_prefetch_queue_size,
                    name=f"offline_big_vae_prefetch_rank_{rank}",
                )
                offline_batch_prefetcher.start()

            for step_idx in range(resumed_training_step, max_steps):
                global_step = step_idx + 1
                current_kl_beta = _compute_kl_beta_for_step(
                    global_step,
                    target_beta=kl_beta,
                    schedule_enabled=kl_schedule_enabled,
                    start_beta=kl_schedule_start_beta,
                    warmup_steps=kl_schedule_warmup_steps,
                    ramp_steps=kl_schedule_ramp_steps,
                )
                current_latent_sampling_gate = (
                    _compute_latent_sampling_gate_for_step(
                        global_step,
                        schedule_enabled=latent_sampling_gate_enabled,
                        start_step=latent_sampling_gate_start_step,
                        ramp_steps=latent_sampling_gate_ramp_steps,
                        start_value=latent_sampling_gate_start_value,
                        end_value=latent_sampling_gate_end_value,
                    )
                    if use_latent_sampling
                    else 0.0
                )
                if use_latent_sampling:
                    _set_model_latent_sampling_gate(model, current_latent_sampling_gate)
                model.train()
                optimizer.zero_grad(set_to_none=True)

                if (
                    rank == 0
                    and collector is not None
                    and dataset is not None
                    and not collector.is_async_mode
                    and (
                        not fixed_training_batch_enabled
                        or fixed_batch_x is None
                        or fixed_batch_W is None
                        or fixed_batch_x_mask is None
                        or fixed_batch_d_in_mask is None
                        or fixed_batch_d_out_mask is None
                    )
                ):
                    dataset.maybe_collect(step_idx)

                if fixed_training_batch_enabled:
                    should_refresh_source_sample = False
                    if consume_slices_without_replacement:
                        current_source_states, current_source_round_robin_offset = _prune_exhausted_source_states_with_offset(
                            current_source_states,
                            current_source_round_robin_offset,
                        )
                        if fixed_training_batch_enabled:
                            should_refresh_source_sample = (
                                (
                                    fixed_batch_x is None
                                    or fixed_batch_W is None
                                    or fixed_batch_x_mask is None
                                    or fixed_batch_d_in_mask is None
                                    or fixed_batch_d_out_mask is None
                                )
                                and not current_source_states
                            )
                        else:
                            should_refresh_source_sample = current_source_states is None or not current_source_states
                    else:
                        if fixed_training_batch_enabled:
                            should_refresh_source_sample = (
                                (
                                    fixed_batch_x is None
                                    or fixed_batch_W is None
                                    or fixed_batch_x_mask is None
                                    or fixed_batch_d_in_mask is None
                                    or fixed_batch_d_out_mask is None
                                )
                                and not current_source_samples
                            )
                        else:
                            should_refresh_source_sample = current_source_samples is None or step_idx % steps_per_sample == 0

                    if should_refresh_source_sample:
                        if consume_slices_without_replacement:
                            current_source_states = _ensure_source_state_pool_capacity(
                                current_source_states=current_source_states,
                                required_remaining_slices=slice_batch_size,
                                target_source_pool_size=requested_source_samples_per_refresh,
                                max_active_source_pool_size=max_active_source_pool_size,
                                rank=rank,
                                device=device,
                                dataset_iter=dataset_iter,
                                use_broadcast=use_broadcast,
                                max_x_rows=max_x_rows,
                                logger=logger,
                                uniqueness=batch_source_mixing_uniqueness if batch_source_mixing_active else "none",
                                deferred_samples=deferred_source_samples,
                                max_deferred_samples=deferred_source_cache_size,
                                refill_fetch_batch_size=offline_batch_prefetch_refill_fetch_batch_size,
                                min_d_in=min_mixed_source_d_in,
                                min_d_out=min_mixed_source_d_out,
                                max_T_patches=curriculum_max_T,
                                curriculum_max_d_out=curriculum_max_d_out,
                                patch_size=patch_size_for_slice,
                                synthetic_layer_enabled=synthetic_layer_enabled,
                                synthetic_n_rows=synthetic_n_rows,
                                synthetic_d_in=synthetic_d_in,
                                synthetic_d_out=synthetic_d_out,
                                synthetic_x_std=synthetic_x_std,
                                synthetic_w_std=synthetic_w_std,
                            )
                            current_source_round_robin_offset = (
                                current_source_round_robin_offset % len(current_source_states)
                                if current_source_states
                                else 0
                            )
                            if (
                                batch_source_mixing_active
                                and current_source_states is not None
                                and len(current_source_states) < requested_source_samples_per_refresh
                                and not source_mixing_shortfall_logged
                                and rank == 0
                            ):
                                logger.info(
                                    "Batch source mixing shortfall: requested=%s compatible source samples, fetched=%s. "
                                    "Training continues with reduced diversity for this refresh.",
                                    requested_source_samples_per_refresh,
                                    len(current_source_states),
                                )
                                source_mixing_shortfall_logged = True
                        else:
                            if synthetic_layer_enabled:
                                synthetic_source_device = torch.device("cpu") if batch_source_mixing_active else device
                                current_source_samples = []
                                for idx in range(requested_source_samples_per_refresh):
                                    x_syn, W_syn = _sample_synthetic_layer(
                                        device=synthetic_source_device,
                                        n_rows=synthetic_n_rows,
                                        d_in=synthetic_d_in,
                                        d_out=synthetic_d_out,
                                        x_std=synthetic_x_std,
                                        w_std=synthetic_w_std,
                                        max_x_rows=max_x_rows,
                                    )
                                    current_source_samples.append(
                                        SourceSampleRecord(
                                            x=x_syn,
                                            W=W_syn,
                                            model_name=f"synthetic_{idx}",
                                        )
                                    )
                            else:
                                if batch_source_mixing_active:
                                    current_source_samples = _fetch_source_samples(
                                        rank=rank,
                                        device=device,
                                        dataset_iter=dataset_iter,
                                        use_broadcast=use_broadcast,
                                        max_x_rows=max_x_rows,
                                        logger=logger,
                                        num_samples=requested_source_samples_per_refresh,
                                        uniqueness=batch_source_mixing_uniqueness,
                                        deferred_samples=deferred_source_samples,
                                        max_deferred_samples=deferred_source_cache_size,
                                        refill_fetch_batch_size=offline_batch_prefetch_refill_fetch_batch_size,
                                        min_d_in=min_mixed_source_d_in,
                                        min_d_out=min_mixed_source_d_out,
                                    )
                                else:
                                    current_source_samples = [
                                        _source_sample_record_to_device(
                                            _fetch_source_sample_record_cpu(
                                                rank=rank,
                                                device=device,
                                                dataset_iter=dataset_iter,
                                                use_broadcast=use_broadcast,
                                                max_x_rows=max_x_rows,
                                                logger=logger,
                                            ),
                                            device=device,
                                        )
                                    ]
                            current_source_round_robin_offset = 0
                            if (
                                batch_source_mixing_active
                                and current_source_samples is not None
                                and len(current_source_samples) < requested_source_samples_per_refresh
                                and not source_mixing_shortfall_logged
                                and rank == 0
                            ):
                                logger.info(
                                    "Batch source mixing shortfall: requested=%s compatible source samples, fetched=%s. "
                                    "Training continues with reduced diversity for this refresh.",
                                    requested_source_samples_per_refresh,
                                    len(current_source_samples),
                                )
                                source_mixing_shortfall_logged = True

                loss_acc = 0.0
                behavioral_acc = 0.0
                behavioral_operator_acc = 0.0
                behavioral_dir_acc = 0.0
                behavioral_scale_acc = 0.0
                structural_acc = 0.0
                kl_acc = 0.0
                struct_dir_acc = 0.0
                struct_scale_acc = 0.0
                struct_rec_acc = 0.0
                struct_rel_acc = 0.0
                step_source_diversity_sum: dict[str, float] = {}
                step_source_diversity_micro_count = 0
                step_is_finite = True
                step_invalid_reason: str | None = None

                for micro_idx in range(grad_accum_steps):
                    sync_grad = micro_idx == grad_accum_steps - 1
                    current_batch_build_s = 0.0
                    current_prefetch_wait_s = 0.0
                    current_h2d_enqueue_s = 0.0
                    current_prefetch_depth = 0.0

                    if fixed_training_batch_enabled:
                        if (
                            fixed_batch_x is None
                            or fixed_batch_W is None
                            or fixed_batch_x_mask is None
                            or fixed_batch_d_in_mask is None
                            or fixed_batch_d_out_mask is None
                        ):
                            if consume_slices_without_replacement:
                                current_source_states = _ensure_source_state_pool_capacity(
                                    current_source_states=current_source_states,
                                    required_remaining_slices=slice_batch_size,
                                    target_source_pool_size=requested_source_samples_per_refresh,
                                    max_active_source_pool_size=max_active_source_pool_size,
                                    rank=rank,
                                    device=device,
                                    dataset_iter=dataset_iter,
                                    use_broadcast=use_broadcast,
                                    max_x_rows=max_x_rows,
                                    logger=logger,
                                    uniqueness=batch_source_mixing_uniqueness if batch_source_mixing_active else "none",
                                    deferred_samples=deferred_source_samples,
                                    max_deferred_samples=deferred_source_cache_size,
                                    refill_fetch_batch_size=offline_batch_prefetch_refill_fetch_batch_size,
                                    min_d_in=min_mixed_source_d_in,
                                    min_d_out=min_mixed_source_d_out,
                                    max_T_patches=curriculum_max_T,
                                    curriculum_max_d_out=curriculum_max_d_out,
                                    patch_size=patch_size_for_slice,
                                    synthetic_layer_enabled=synthetic_layer_enabled,
                                    synthetic_n_rows=synthetic_n_rows,
                                    synthetic_d_in=synthetic_d_in,
                                    synthetic_d_out=synthetic_d_out,
                                    synthetic_x_std=synthetic_x_std,
                                    synthetic_w_std=synthetic_w_std,
                                )
                                if not current_source_states:
                                    raise RuntimeError("fixed training batch capture requires a loaded source sample")
                                batch_payload = _build_training_batch_from_source_states(
                                    current_source_states,
                                    batch_size=slice_batch_size,
                                    start_offset=current_source_round_robin_offset,
                                    target_x_rows=stable_batch_target_x_rows,
                                    target_d_in=stable_batch_target_d_in,
                                    target_d_out=stable_batch_target_d_out,
                                )
                                fixed_batch_diversity_stats = _compute_consumed_batch_source_diversity_stats(
                                    current_source_states,
                                    used_source_indices=batch_payload.used_source_indices,
                                    source_pool_remaining_slices_pre=batch_payload.source_pool_remaining_slices_pre,
                                    source_pool_remaining_slices_post=batch_payload.source_pool_remaining_slices_post,
                                )
                                fixed_batch_W, fixed_batch_x, fixed_batch_x_mask, fixed_batch_d_in_mask, fixed_batch_d_out_mask = (
                                    batch_payload.W,
                                    batch_payload.x,
                                    batch_payload.x_mask,
                                    batch_payload.d_in_mask,
                                    batch_payload.d_out_mask,
                                )
                            else:
                                if not current_source_samples:
                                    raise RuntimeError("fixed training batch capture requires a loaded source sample")
                                fixed_batch_diversity_stats = _compute_batch_source_diversity_stats(
                                    current_source_samples,
                                    batch_size=slice_batch_size,
                                    start_offset=current_source_round_robin_offset,
                                )
                                fixed_batch_W, fixed_batch_x, fixed_batch_x_mask, fixed_batch_d_in_mask, fixed_batch_d_out_mask = _build_training_batch_from_source_samples(
                                    current_source_samples,
                                    max_T_patches=curriculum_max_T,
                                    max_d_out=curriculum_max_d_out,
                                    patch_size=patch_size_for_slice,
                                    batch_size=slice_batch_size,
                                    start_offset=current_source_round_robin_offset,
                                    target_x_rows=stable_batch_target_x_rows,
                                    target_d_in=stable_batch_target_d_in,
                                    target_d_out=stable_batch_target_d_out,
                                )
                            fixed_batch_W = fixed_batch_W.to(device=device, non_blocking=True)
                            fixed_batch_x = fixed_batch_x.to(device=device, non_blocking=True)
                            fixed_batch_x_mask = fixed_batch_x_mask.to(device=device, non_blocking=True)
                            fixed_batch_d_in_mask = fixed_batch_d_in_mask.to(device=device, non_blocking=True)
                            fixed_batch_d_out_mask = fixed_batch_d_out_mask.to(device=device, non_blocking=True)
                            current_source_samples = None
                            current_source_states = None
                            current_source_round_robin_offset = 0
                            if rank == 0:
                                logger.info(
                                    "Captured fixed training batch at step=%s: W=%s x=%s",
                                    global_step,
                                    tuple(fixed_batch_W.shape),
                                    tuple(fixed_batch_x.shape),
                                )
                                _maybe_dump_fixed_training_batch(
                                    W_s=fixed_batch_W,
                                    x_s=fixed_batch_x,
                                    x_mask_s=fixed_batch_x_mask,
                                    d_in_mask_s=fixed_batch_d_in_mask,
                                    d_out_mask_s=fixed_batch_d_out_mask,
                                    cfg=cfg,
                                    logger=logger,
                                    global_step=global_step,
                                    stage=stage_num,
                                    patch_size=patch_size_for_slice,
                                    max_T_patches=curriculum_max_T,
                                    max_d_out=curriculum_max_d_out,
                                    slice_batch_size=slice_batch_size,
                                )
                            if dataset is not None:
                                dataset.close()
                                dataset = None
                            if collector is not None:
                                collector.shutdown()
                                collector = None
                            dataset_iter = None
                        if fixed_batch_diversity_stats is None:
                            raise RuntimeError("fixed training batch requires cached diversity stats")
                        current_batch_source_diversity = dict(fixed_batch_diversity_stats)
                        W_s, x_s, x_mask_s, d_in_mask_s, d_out_mask_s = (
                            fixed_batch_W,
                            fixed_batch_x,
                            fixed_batch_x_mask,
                            fixed_batch_d_in_mask,
                            fixed_batch_d_out_mask,
                        )
                    else:
                        if offline_batch_prefetcher is not None:
                            prefetched = offline_batch_prefetcher.get()
                            prepared_batch = prefetched.value
                            current_prefetch_wait_s = float(prefetched.wait_time_s)
                            current_prefetch_depth = float(offline_batch_prefetcher.qsize)
                        else:
                            prepared_batch = _prepare_training_micro_batch_cpu(step_idx=step_idx, micro_idx=micro_idx)
                        current_batch_build_s = float(prepared_batch.build_time_s)
                        current_batch_source_diversity = dict(prepared_batch.source_diversity)
                        h2d_t0 = time.perf_counter()
                        W_s = prepared_batch.W.to(device=device, non_blocking=True)
                        x_s = prepared_batch.x.to(device=device, non_blocking=True)
                        x_mask_s = prepared_batch.x_mask.to(device=device, non_blocking=True)
                        d_in_mask_s = prepared_batch.d_in_mask.to(device=device, non_blocking=True)
                        d_out_mask_s = prepared_batch.d_out_mask.to(device=device, non_blocking=True)
                        current_h2d_enqueue_s = time.perf_counter() - h2d_t0
                    data_build_window_s += float(current_batch_build_s)
                    data_wait_window_s += float(current_prefetch_wait_s)
                    data_h2d_window_s += float(current_h2d_enqueue_s)
                    data_prefetch_depth_window += float(current_prefetch_depth)
                    current_batch_source_diversity["target_models"] = float(max(1, requested_source_samples_per_refresh))
                    current_batch_source_diversity["batch_model_target_coverage"] = (
                        current_batch_source_diversity["batch_unique_models"]
                        / max(1.0, current_batch_source_diversity["target_models"])
                    )
                    current_batch_source_diversity["batch_model_shortfall"] = max(
                        0.0,
                        current_batch_source_diversity["target_models"] - current_batch_source_diversity["batch_unique_models"],
                    )
                    step_source_diversity_micro_count += 1
                    for key, value in current_batch_source_diversity.items():
                        step_source_diversity_sum[key] = step_source_diversity_sum.get(key, 0.0) + float(value)

                    no_sync_ctx = contextlib.nullcontext()
                    if is_distributed and not sync_grad:
                        no_sync_ctx = model.no_sync()  # type: ignore[union-attr]

                    with no_sync_ctx:
                        if use_cudagraph_step_begin:
                            cudagraph_step_begin()
                        with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                            direction_pre_norms: torch.Tensor | None = None
                            if fixed_training_batch_enabled:
                                W_hat, mu, logvar, pred_dirs, direction_pre_norms = model(
                                    W_s,
                                    x_s,
                                    x_mask=x_mask_s,
                                    d_in_mask=d_in_mask_s,
                                    d_out_mask=d_out_mask_s,
                                    return_direction_pre_norms=True,
                                )
                            else:
                                W_hat, mu, logvar, pred_dirs = model(
                                    W_s,
                                    x_s,
                                    x_mask=x_mask_s,
                                    d_in_mask=d_in_mask_s,
                                    d_out_mask=d_out_mask_s,
                                )
                            behavioral_operator_loss = WeightQuantileVAE.operator_recon_loss(
                                x_s,
                                W_s,
                                W_hat,
                                x_mask=x_mask_s,
                                d_in_mask=d_in_mask_s,
                                d_out_mask=d_out_mask_s,
                            )
                            if behavioral_lambda_dir != 0.0 or behavioral_lambda_scale != 0.0:
                                behavioral_dir_loss, behavioral_scale_loss = WeightQuantileVAE.operator_direction_scale_loss(
                                    x_s,
                                    W_s,
                                    W_hat,
                                    x_mask=x_mask_s,
                                    d_out_mask=d_out_mask_s,
                                    gamma=behavioral_gamma,
                                    huber_delta=behavioral_huber_delta,
                                )
                            else:
                                behavioral_dir_loss = behavioral_operator_loss.new_zeros(())
                                behavioral_scale_loss = behavioral_operator_loss.new_zeros(())
                            behavioral_loss = (
                                behavioral_lambda_operator * behavioral_operator_loss
                                + behavioral_lambda_dir * behavioral_dir_loss
                                + behavioral_lambda_scale * behavioral_scale_loss
                            )
                            structural_loss, struct_details = WeightQuantileVAE.patch_structure_loss(
                                W_s, W_hat, patch_size=patch_size_for_slice,
                                gamma=struct_gamma,
                                lambda_dir=struct_lambda_dir,
                                lambda_scale=struct_lambda_scale,
                                lambda_rec=struct_lambda_rec,
                                lambda_rel=struct_lambda_rel,
                                huber_delta=struct_huber_delta,
                                pred_dirs=pred_dirs,
                                d_in_mask=d_in_mask_s,
                                d_out_mask=d_out_mask_s,
                            )
                            if use_latent_sampling:
                                kl_loss = _compute_model_latent_kl(model, mu, logvar)
                            else:
                                kl_loss = mu.new_zeros(())
                            total_loss = mu.new_zeros(())
                            if behavioral_coef != 0.0:
                                total_loss = total_loss + behavioral_coef * behavioral_loss
                            if structural_coef != 0.0:
                                total_loss = total_loss + structural_coef * structural_loss
                            if current_kl_beta != 0.0:
                                total_loss = total_loss + current_kl_beta * kl_loss
                            loss_for_backward = total_loss / grad_accum_steps

                        if rank == 0 and fixed_training_batch_enabled and direction_pre_norms is not None:
                            direction_pre_norm_stats_latest = _tensor_debug_stats(direction_pre_norms)

                        local_loss_is_finite = bool(torch.isfinite(loss_for_backward.detach()).item())
                        if not local_loss_is_finite:
                            non_finite_payload = {
                                "step": int(global_step),
                                "micro_step": int(micro_idx),
                                "rank": int(rank),
                                "fixed_training_batch": bool(fixed_training_batch_enabled),
                                "synthetic_layer_source": bool(synthetic_layer_enabled),
                                "loss": {
                                    "total": _scalar_debug_value(total_loss),
                                    "behavioral": _scalar_debug_value(behavioral_loss),
                                    "behavioral_operator": _scalar_debug_value(behavioral_operator_loss),
                                    "behavioral_dir": _scalar_debug_value(behavioral_dir_loss),
                                    "behavioral_scale": _scalar_debug_value(behavioral_scale_loss),
                                    "structural": _scalar_debug_value(structural_loss),
                                    "kl": _scalar_debug_value(kl_loss),
                                    "kl_beta": float(current_kl_beta),
                                    "latent_sampling_gate": float(current_latent_sampling_gate),
                                    "struct_dir": _scalar_debug_value(struct_details["L_dir"]),
                                    "struct_scale": _scalar_debug_value(struct_details["L_scale"]),
                                    "struct_rec": _scalar_debug_value(struct_details["L_rec"]),
                                    "struct_rel": _scalar_debug_value(struct_details["L_rel"]),
                                },
                                "tensors": {
                                    "x_s": _tensor_debug_stats(x_s),
                                    "W_s": _tensor_debug_stats(W_s),
                                    "W_hat": _tensor_debug_stats(W_hat),
                                    "mu": _tensor_debug_stats(mu),
                                    "logvar": _tensor_debug_stats(logvar),
                                    "pred_dirs": _tensor_debug_stats(pred_dirs),
                                    "direction_pre_norms": _tensor_debug_stats(direction_pre_norms),
                                },
                            }
                            logger.warning(
                                "Non-finite local loss detected: %s",
                                json.dumps(non_finite_payload, ensure_ascii=False),
                            )

                        finite_flag = torch.tensor(
                            1 if local_loss_is_finite else 0,
                            dtype=torch.int32,
                            device=device,
                        )
                        if is_distributed:
                            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

                        if int(finite_flag.item()) == 0:
                            step_is_finite = False
                            step_invalid_reason = "non-finite loss"
                        else:
                            if scaler.is_enabled():
                                scaler.scale(loss_for_backward).backward()
                            else:
                                loss_for_backward.backward()

                            local_bad_grad_report = _collect_nonfinite_grad_report(model, max_items=12)
                            local_grads_finite = local_bad_grad_report is None
                            grad_finite_flag = torch.tensor(
                                1 if local_grads_finite else 0,
                                dtype=torch.int32,
                                device=device,
                            )
                            if is_distributed:
                                dist.all_reduce(grad_finite_flag, op=dist.ReduceOp.MIN)

                            if not local_grads_finite:
                                grad_payload = {
                                    "step": int(global_step),
                                    "micro_step": int(micro_idx),
                                    "rank": int(rank),
                                    "amp_enabled": bool(scaler.is_enabled()),
                                    "fixed_training_batch": bool(fixed_training_batch_enabled),
                                    "synthetic_layer_source": bool(synthetic_layer_enabled),
                                    "report": local_bad_grad_report,
                                }
                                logger.warning(
                                    "Non-finite gradients detected: %s",
                                    json.dumps(grad_payload, ensure_ascii=False),
                                )

                            if int(grad_finite_flag.item()) == 0:
                                step_is_finite = False
                                step_invalid_reason = "non-finite gradients"
                                if local_grads_finite and rank == 0:
                                    logger.warning(
                                        "Non-finite gradients detected on another rank: step=%s micro_step=%s",
                                        global_step,
                                        micro_idx,
                                    )

                    loss_acc += float(total_loss.detach().item())
                    behavioral_acc += float(behavioral_loss.detach().item())
                    behavioral_operator_acc += float(behavioral_operator_loss.detach().item())
                    behavioral_dir_acc += float(behavioral_dir_loss.detach().item())
                    behavioral_scale_acc += float(behavioral_scale_loss.detach().item())
                    structural_acc += float(structural_loss.detach().item())
                    kl_acc += float(kl_loss.detach().item())
                    struct_dir_acc += float(struct_details["L_dir"].detach().item())
                    struct_scale_acc += float(struct_details["L_scale"].detach().item())
                    struct_rec_acc += float(struct_details["L_rec"].detach().item())
                    struct_rel_acc += float(struct_details["L_rel"].detach().item())

                    if not step_is_finite:
                        break

                if not step_is_finite:
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        logger.warning(
                            "Skipping step %s due to %s",
                            global_step,
                            step_invalid_reason or "non-finite values",
                        )
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                monitor_layer_snapshot_this_step = grad_layer_monitor_enabled and (
                    global_step == 1 or (global_step % grad_layer_monitor_every_steps == 0)
                )
                collect_clip_diagnostics_this_step = rank == 0 and (
                    monitor_layer_snapshot_this_step or (global_step % log_every == 0)
                )
                clip_use_foreach = device.type == "cuda"
                layer_rms_pre_clip_snapshot: dict[str, float] | None = None
                if monitor_layer_snapshot_this_step:
                    layer_rms_pre_clip_snapshot = collect_grad_rms_per_layer(
                        model=model,
                        include_prefixes=grad_layer_monitor_include_prefixes,
                        weights_only=grad_layer_monitor_weights_only,
                    )

                grad_clip_coef = 1.0
                grad_global_before_clip = 0.0
                grad_global_after_clip = 0.0
                grad_clip_group_before: dict[str, float] = {}
                grad_clip_group_after: dict[str, float] = {}
                grad_clip_group_coef: dict[str, float] = {}
                if partwise_grad_clip_enabled:
                    for group_name in grad_group_prefixes:
                        params = params_by_grad_group.get(group_name, [])
                        clip_limit = float(grad_clip_norm_by_part.get(group_name, grad_clip_norm))
                        clip_return = None

                        if params and clip_limit > 0.0:
                            clip_return = _clip_grad_norm_with_optional_foreach(
                                get_params=lambda params=params: params,
                                max_norm=clip_limit,
                                use_foreach=clip_use_foreach,
                            )

                        if not collect_clip_diagnostics_this_step:
                            continue

                        if not params:
                            before_norm = 0.0
                            after_norm = 0.0
                            clip_coef = 1.0
                        elif clip_limit > 0.0:
                            before_norm = _clip_return_to_float(clip_return) if clip_return is not None else 0.0
                            clip_coef = min(1.0, clip_limit / max(1e-12, before_norm))
                            after_norm = before_norm * clip_coef
                        else:
                            before_norm = _grad_l2_norm_for_params(params)
                            after_norm = before_norm
                            clip_coef = 1.0
                        grad_clip_group_before[group_name] = float(before_norm)
                        grad_clip_group_after[group_name] = float(after_norm)
                        grad_clip_group_coef[group_name] = float(clip_coef)

                    if collect_clip_diagnostics_this_step:
                        grad_global_before_clip = math.sqrt(
                            sum(float(value) * float(value) for value in grad_clip_group_before.values())
                        )
                        grad_global_after_clip = math.sqrt(
                            sum(float(value) * float(value) for value in grad_clip_group_after.values())
                        )
                        grad_clip_coef = (
                            grad_global_after_clip / max(1e-12, grad_global_before_clip)
                            if grad_global_before_clip > 0.0
                            else 1.0
                        )
                elif grad_clip_norm > 0.0:
                    clip_return = _clip_grad_norm_with_optional_foreach(
                        get_params=model.parameters,
                        max_norm=grad_clip_norm,
                        use_foreach=clip_use_foreach,
                    )
                    if collect_clip_diagnostics_this_step:
                        grad_global_before_clip = _clip_return_to_float(clip_return)
                        grad_clip_coef = min(1.0, float(grad_clip_norm) / max(1e-12, grad_global_before_clip))
                        grad_global_after_clip = grad_global_before_clip * grad_clip_coef
                else:
                    grad_global_before_clip = 0.0
                    grad_global_after_clip = 0.0

                grad_stats: dict[str, float] = {}
                collect_grad_stats_this_step = rank == 0 and (global_step % log_every == 0)
                if collect_grad_stats_this_step:
                    grad_stats = compute_grad_stats(model)
                    if (not partwise_grad_clip_enabled) and grad_clip_norm <= 0.0:
                        grad_global_before_clip = float(grad_stats.get("grad/global_norm", 0.0))
                        grad_global_after_clip = grad_global_before_clip
                    grad_stats["grad/global_norm_before_clip"] = float(grad_global_before_clip)
                    grad_stats["grad/global_norm_after_clip_est"] = float(grad_global_after_clip)
                    grad_stats["grad/clip_coef"] = float(grad_clip_coef)
                    if partwise_grad_clip_enabled:
                        for group_name in grad_group_prefixes:
                            grad_stats[f"grad/{group_name}_global_norm_before_clip"] = float(
                                grad_clip_group_before.get(group_name, 0.0)
                            )
                            grad_stats[f"grad/{group_name}_global_norm_after_clip"] = float(
                                grad_clip_group_after.get(group_name, 0.0)
                            )
                            grad_stats[f"grad/{group_name}_clip_coef"] = float(
                                grad_clip_group_coef.get(group_name, 1.0)
                            )
                            grad_stats[f"grad/{group_name}_clip_norm_limit"] = float(
                                grad_clip_norm_by_part.get(group_name, grad_clip_norm)
                            )

                if monitor_layer_snapshot_this_step:
                    layer_rms_post_clip = collect_grad_rms_per_layer(
                        model=model,
                        include_prefixes=grad_layer_monitor_include_prefixes,
                        weights_only=grad_layer_monitor_weights_only,
                    )
                    if layer_rms_pre_clip_snapshot:
                        layer_rms_pre_clip = {str(key): float(value) for key, value in layer_rms_pre_clip_snapshot.items()}
                    else:
                        clip_coef_safe = max(1e-12, float(grad_clip_coef))
                        layer_rms_pre_clip = {
                            key: (float(value) / clip_coef_safe if float(grad_clip_coef) < 1.0 else float(value))
                            for key, value in layer_rms_post_clip.items()
                        }
                    layer_param_rms = collect_param_rms_per_layer(
                        model=model,
                        include_prefixes=grad_layer_monitor_include_prefixes,
                        weights_only=grad_layer_monitor_weights_only,
                    )
                    layer_grad_to_param_ratio_pre_clip: dict[str, float] = {}
                    layer_grad_to_param_ratio_post_clip: dict[str, float] = {}
                    layer_keys = sorted(set(layer_rms_pre_clip.keys()) | set(layer_rms_post_clip.keys()) | set(layer_param_rms.keys()))
                    for layer in layer_keys:
                        param_rms = float(layer_param_rms.get(layer, 0.0))
                        pre_rms = float(layer_rms_pre_clip.get(layer, 0.0))
                        post_rms = float(layer_rms_post_clip.get(layer, 0.0))
                        denom = max(1e-12, param_rms)
                        layer_grad_to_param_ratio_pre_clip[layer] = pre_rms / denom
                        layer_grad_to_param_ratio_post_clip[layer] = post_rms / denom

                    for layer, value in layer_rms_pre_clip.items():
                        grad_layer_history.setdefault(layer, []).append((int(global_step), float(value)))

                    ranked_layers = [
                        (layer, value)
                        for layer, value in layer_rms_pre_clip.items()
                        if not layer.startswith("__")
                    ]
                    ranked_layers.sort(key=lambda item: float(item[1]), reverse=True)
                    ranked_low_layers = sorted(ranked_layers, key=lambda item: float(item[1]))
                    ranked_by_ratio = sorted(
                        (
                            (layer, float(layer_grad_to_param_ratio_pre_clip.get(layer, 0.0)))
                            for layer, _ in ranked_layers
                        ),
                        key=lambda item: float(item[1]),
                        reverse=True,
                    )
                    topk = max(1, int(grad_layer_monitor_topk_layers))
                    top_layers_pre_clip = [
                        [
                            str(layer),
                            float(value),
                            float(layer_param_rms.get(layer, 0.0)),
                            float(layer_grad_to_param_ratio_pre_clip.get(layer, 0.0)),
                        ]
                        for layer, value in ranked_layers[:topk]
                    ]
                    low_layers_pre_clip = [
                        [
                            str(layer),
                            float(value),
                            float(layer_param_rms.get(layer, 0.0)),
                            float(layer_grad_to_param_ratio_pre_clip.get(layer, 0.0)),
                        ]
                        for layer, value in ranked_low_layers[:topk]
                    ]
                    top_layers_ratio_pre_clip = [
                        [
                            str(layer),
                            float(value),
                            float(layer_rms_pre_clip.get(layer, 0.0)),
                            float(layer_param_rms.get(layer, 0.0)),
                        ]
                        for layer, value in ranked_by_ratio[:topk]
                    ]
                    global_grad_rms_pre_clip = float(layer_rms_pre_clip.get("__global__", 0.0))
                    global_grad_rms_post_clip = float(layer_rms_post_clip.get("__global__", 0.0))
                    global_param_rms = float(layer_param_rms.get("__global__", 0.0))
                    global_grad_to_param_ratio_pre_clip = global_grad_rms_pre_clip / max(1e-12, global_param_rms)
                    global_grad_to_param_ratio_post_clip = global_grad_rms_post_clip / max(1e-12, global_param_rms)

                    latest_grad_layer_snapshot = {
                        "step": int(global_step),
                        "num_layers": int(len(ranked_layers)),
                        "clip_coef": float(grad_clip_coef),
                        "global_grad_rms_pre_clip": global_grad_rms_pre_clip,
                        "global_grad_rms_post_clip": global_grad_rms_post_clip,
                        "global_param_rms": global_param_rms,
                        "global_grad_to_param_ratio_pre_clip": global_grad_to_param_ratio_pre_clip,
                        "global_grad_to_param_ratio_post_clip": global_grad_to_param_ratio_post_clip,
                        "top_layers_pre_clip": top_layers_pre_clip,
                        "low_layers_pre_clip": low_layers_pre_clip,
                        "top_layers_ratio_pre_clip": top_layers_ratio_pre_clip,
                    }

                    if grad_layer_monitor_save_csv:
                        _append_grad_layer_rms_csv(
                            save_path=grad_layer_monitor_csv_path,
                            step=int(global_step),
                            layer_rms_pre_clip=layer_rms_pre_clip,
                            layer_rms_post_clip=layer_rms_post_clip,
                            layer_param_rms=layer_param_rms,
                            layer_grad_to_param_ratio_pre_clip=layer_grad_to_param_ratio_pre_clip,
                            layer_grad_to_param_ratio_post_clip=layer_grad_to_param_ratio_post_clip,
                            clip_coef=float(grad_clip_coef),
                        )
                    if grad_layer_monitor_save_plot and (
                        global_step == 1
                        or (
                            grad_layer_monitor_plot_every_steps > 0
                            and global_step % grad_layer_monitor_plot_every_steps == 0
                        )
                    ):
                        plot_saved = _save_grad_rms_layer_plot(
                            history=grad_layer_history,
                            save_path=grad_layer_monitor_plot_path,
                            topk_layers=grad_layer_monitor_topk_layers,
                            log_scale=grad_layer_monitor_log_scale,
                        )
                        if not plot_saved:
                            logger.warning("Grad-layer monitor plot skipped: matplotlib is unavailable")
                        if grad_layer_monitor_save_heatmap:
                            _save_grad_rms_layer_heatmap(
                                history=grad_layer_history,
                                save_path=grad_layer_monitor_heatmap_path,
                                max_layers=grad_layer_monitor_heatmap_max_layers,
                                log_scale=grad_layer_monitor_log_scale,
                            )

                    logger.info(
                        "grad_layer_monitor step=%s layers=%s global_pre=%.3e global_post=%.3e "
                        "global_ratio_pre=%.3e top_pre=%s low_pre=%s",
                        global_step,
                        int(latest_grad_layer_snapshot.get("num_layers", 0)),
                        float(latest_grad_layer_snapshot.get("global_grad_rms_pre_clip", 0.0)),
                        float(latest_grad_layer_snapshot.get("global_grad_rms_post_clip", 0.0)),
                        float(latest_grad_layer_snapshot.get("global_grad_to_param_ratio_pre_clip", 0.0)),
                        latest_grad_layer_snapshot.get("top_layers_pre_clip", [])[: min(5, grad_layer_monitor_topk_layers)],
                        latest_grad_layer_snapshot.get("low_layers_pre_clip", [])[: min(5, grad_layer_monitor_topk_layers)],
                    )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                step_loss = loss_acc / grad_accum_steps
                step_behavioral = behavioral_acc / grad_accum_steps
                step_behavioral_operator = behavioral_operator_acc / grad_accum_steps
                step_behavioral_dir = behavioral_dir_acc / grad_accum_steps
                step_behavioral_scale = behavioral_scale_acc / grad_accum_steps
                step_structural = structural_acc / grad_accum_steps
                step_kl = kl_acc / grad_accum_steps
                step_struct_dir = struct_dir_acc / grad_accum_steps
                step_struct_scale = struct_scale_acc / grad_accum_steps
                step_struct_rec = struct_rec_acc / grad_accum_steps
                step_struct_rel = struct_rel_acc / grad_accum_steps

                stats = torch.tensor(
                    [
                        step_loss,
                        step_behavioral,
                        step_behavioral_operator,
                        step_behavioral_dir,
                        step_behavioral_scale,
                        step_structural,
                        step_kl,
                        step_struct_dir,
                        step_struct_scale,
                        step_struct_rec,
                        step_struct_rel,
                    ],
                    dtype=torch.float32, device=device,
                )
                if is_distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    stats /= float(world_size)

                loss_window += float(stats[0].item())
                behavioral_window += float(stats[1].item())
                behavioral_operator_window += float(stats[2].item())
                behavioral_dir_window += float(stats[3].item())
                behavioral_scale_window += float(stats[4].item())
                structural_window += float(stats[5].item())
                kl_window += float(stats[6].item())
                struct_dir_window += float(stats[7].item())
                struct_scale_window += float(stats[8].item())
                struct_rec_window += float(stats[9].item())
                struct_rel_window += float(stats[10].item())
                if step_source_diversity_micro_count > 0:
                    step_source_diversity_stats = {
                        key: float(value) / float(step_source_diversity_micro_count)
                        for key, value in step_source_diversity_sum.items()
                    }
                    source_diversity_latest = step_source_diversity_stats
                    source_diversity_window_unique_models_sum += float(step_source_diversity_stats.get("batch_unique_models", 0.0))
                    source_diversity_window_unique_models_min = min(
                        source_diversity_window_unique_models_min,
                        float(step_source_diversity_stats.get("batch_unique_models", 0.0)),
                    )
                    source_diversity_window_unique_models_max = max(
                        source_diversity_window_unique_models_max,
                        float(step_source_diversity_stats.get("batch_unique_models", 0.0)),
                    )
                    source_diversity_window_target_coverage_sum += float(
                        step_source_diversity_stats.get("batch_model_target_coverage", 0.0)
                    )
                    source_diversity_window_model_perplexity_sum += float(
                        step_source_diversity_stats.get("batch_model_perplexity", 0.0)
                    )
                    if float(step_source_diversity_stats.get("batch_model_shortfall", 0.0)) > 0.0:
                        source_diversity_window_shortfall_steps += 1
                window_steps += 1

                if rank == 0 and global_step % log_every == 0:
                    dt = max(1e-6, time.time() - t0)
                    avg_loss = loss_window / max(1, window_steps)
                    avg_behavioral = behavioral_window / max(1, window_steps)
                    avg_behavioral_operator = behavioral_operator_window / max(1, window_steps)
                    avg_behavioral_dir = behavioral_dir_window / max(1, window_steps)
                    avg_behavioral_scale = behavioral_scale_window / max(1, window_steps)
                    avg_structural = structural_window / max(1, window_steps)
                    avg_kl = kl_window / max(1, window_steps)
                    avg_struct_dir = struct_dir_window / max(1, window_steps)
                    avg_struct_scale = struct_scale_window / max(1, window_steps)
                    avg_struct_rec = struct_rec_window / max(1, window_steps)
                    avg_struct_rel = struct_rel_window / max(1, window_steps)
                    lr = float(optimizer.param_groups[0]["lr"])
                    lr_by_group: dict[str, float] = {}
                    for group_idx, param_group in enumerate(optimizer.param_groups):
                        if len(optimizer.param_groups) <= 1 and "group_name" not in param_group:
                            continue
                        group_name = str(param_group.get("group_name", f"group_{group_idx}")).strip()
                        if not group_name:
                            group_name = f"group_{group_idx}"
                        lr_by_group[group_name] = float(param_group["lr"])
                    speed = window_steps / dt
                    window_micro_steps = max(1, window_steps * grad_accum_steps)
                    avg_data_build_ms = 1000.0 * data_build_window_s / float(window_micro_steps)
                    avg_data_wait_ms = 1000.0 * data_wait_window_s / float(window_micro_steps)
                    avg_data_h2d_enqueue_ms = 1000.0 * data_h2d_window_s / float(window_micro_steps)
                    avg_data_prefetch_depth = data_prefetch_depth_window / float(window_micro_steps)
                    diversity_unique_mean = source_diversity_window_unique_models_sum / max(1, window_steps)
                    diversity_unique_min = (
                        source_diversity_window_unique_models_min
                        if source_diversity_window_unique_models_min != math.inf
                        else 0.0
                    )
                    diversity_unique_max = source_diversity_window_unique_models_max
                    diversity_target_coverage_mean = source_diversity_window_target_coverage_sum / max(1, window_steps)
                    diversity_model_perplexity_mean = source_diversity_window_model_perplexity_sum / max(1, window_steps)

                    cache_metric = dataset.cache_size() if dataset is not None else 0
                    encoder_alpha_values = _get_encoder_conditioning_alpha_values(model)
                    patch_tokenizer_alpha_stats = _get_patch_tokenizer_block_alpha_stats(model)
                    patch_latent_variance_stats = _get_patch_latent_variance_stats(
                        model,
                        W_s[:1],
                        x_s[:1],
                        x_mask=x_mask_s[:1],
                        d_in_mask=d_in_mask_s[:1],
                        d_out_mask=d_out_mask_s[:1],
                    )
                    logger.info(
                        "step=%s/%s loss=%.6f behav=%.6f struct=%.6f "
                        "b_op=%.6f b_dir=%.6f b_scl=%.6f "
                        "s_dir=%.6f s_scl=%.6f s_rec=%.6f s_rel=%.6f "
                        "kl=%.6f kl_beta=%.6f latent_gate=%.6f lr=%.6e steps/s=%.2f cache=%s "
                        "data_build_ms=%.2f data_wait_ms=%.2f data_h2d_enqueue_ms=%.2f prefetch_depth=%.2f",
                        global_step,
                        max_steps,
                        avg_loss,
                        avg_behavioral,
                        avg_structural,
                        avg_behavioral_operator,
                        avg_behavioral_dir,
                        avg_behavioral_scale,
                        avg_struct_dir,
                        avg_struct_scale,
                        avg_struct_rec,
                        avg_struct_rel,
                        avg_kl,
                        current_kl_beta,
                        current_latent_sampling_gate,
                        lr,
                        speed,
                        cache_metric,
                        avg_data_build_ms,
                        avg_data_wait_ms,
                        avg_data_h2d_enqueue_ms,
                        avg_data_prefetch_depth,
                    )
                    if encoder_alpha_values:
                        logger.info(
                            "encoder_alpha step=%s values=[%s]",
                            global_step,
                            ",".join(f"{value:.6f}" for value in encoder_alpha_values),
                        )
                    if patch_tokenizer_alpha_stats:
                        logger.info(
                            "patch_tokenizer_alpha step=%s %s",
                            global_step,
                            "; ".join(
                                (
                                    f"block{block_idx}:mean={stats['mean']:.6f},"
                                    f"abs_mean={stats['abs_mean']:.6f},max_abs={stats['max_abs']:.6f}"
                                )
                                for block_idx, stats in enumerate(patch_tokenizer_alpha_stats)
                            ),
                        )
                    if patch_latent_variance_stats is not None:
                        logger.info(
                            "patch_latent_var step=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
                            global_step,
                            float(patch_latent_variance_stats["mean"]),
                            float(patch_latent_variance_stats["std"]),
                            float(patch_latent_variance_stats["min"]),
                            float(patch_latent_variance_stats["max"]),
                        )
                    if source_diversity_latest is not None:
                        logger.info(
                            "batch_diversity step=%s target=%.0f pool=%.2f remaining_slices_pre=%.2f "
                            "remaining_slices_post=%.2f latest_unique_models=%.2f "
                            "latest_sources_used=%.2f latest_coverage=%.3f latest_perplexity=%.3f "
                            "window_unique_models=%.2f[min=%.2f max=%.2f] window_coverage=%.3f shortfall_steps=%s/%s",
                            global_step,
                            float(source_diversity_latest.get("target_models", 0.0)),
                            float(source_diversity_latest.get("source_pool_size", 0.0)),
                            float(source_diversity_latest.get("source_pool_remaining_slices_pre", 0.0)),
                            float(source_diversity_latest.get("source_pool_remaining_slices_post", 0.0)),
                            float(source_diversity_latest.get("batch_unique_models", 0.0)),
                            float(source_diversity_latest.get("batch_sources_used", 0.0)),
                            float(source_diversity_latest.get("batch_model_target_coverage", 0.0)),
                            float(source_diversity_latest.get("batch_model_perplexity", 0.0)),
                            diversity_unique_mean,
                            diversity_unique_min,
                            diversity_unique_max,
                            diversity_target_coverage_mean,
                            source_diversity_window_shortfall_steps,
                            window_steps,
                        )
                    if fixed_training_batch_enabled and direction_pre_norm_stats_latest is not None:
                        logger.info(
                            "direction_pre_norm step=%s mean=%.6f std=%.6f min=%.6f max=%.6f "
                            "finite=%s/%s nan=%s inf=%s",
                            global_step,
                            float(direction_pre_norm_stats_latest.get("mean", 0.0)),
                            float(direction_pre_norm_stats_latest.get("std", 0.0)),
                            float(direction_pre_norm_stats_latest.get("min", 0.0)),
                            float(direction_pre_norm_stats_latest.get("max", 0.0)),
                            int(direction_pre_norm_stats_latest.get("finite_count", 0)),
                            int(direction_pre_norm_stats_latest.get("numel", 0)),
                            int(direction_pre_norm_stats_latest.get("nan_count", 0)),
                            int(direction_pre_norm_stats_latest.get("inf_count", 0)),
                        )
                    if (
                        (comet_tracker is not None and comet_tracker.enabled)
                        or (wandb_tracker is not None and wandb_tracker.enabled)
                    ):
                        comet_metrics: dict[str, float] = {
                            "train/loss": float(avg_loss),
                            "train/behavioral_loss": float(avg_behavioral),
                            "train/behavioral_operator": float(avg_behavioral_operator),
                            "train/behavioral_dir": float(avg_behavioral_dir),
                            "train/behavioral_scale": float(avg_behavioral_scale),
                            "train/structural_loss": float(avg_structural),
                            "train/kl_loss": float(avg_kl),
                            "train/struct_dir": float(avg_struct_dir),
                            "train/struct_scale": float(avg_struct_scale),
                            "train/struct_rec": float(avg_struct_rec),
                            "train/struct_rel": float(avg_struct_rel),
                            "train/lr": float(lr),
                            "train/kl_beta": float(current_kl_beta),
                            "train/latent_sampling_gate": float(current_latent_sampling_gate),
                            "train/steps_per_sec": float(speed),
                            "data/batch_build_ms": float(avg_data_build_ms),
                            "data/batch_wait_ms": float(avg_data_wait_ms),
                            "data/batch_h2d_enqueue_ms": float(avg_data_h2d_enqueue_ms),
                            "data/prefetch_depth_mean": float(avg_data_prefetch_depth),
                            "data/cache_size": float(cache_metric),
                            "grad/global_norm_before_clip": float(grad_stats.get("grad/global_norm_before_clip", 0.0)),
                            "grad/global_norm_after_clip_est": float(grad_stats.get("grad/global_norm_after_clip_est", 0.0)),
                            "grad/global_norm": float(grad_stats.get("grad/global_norm", 0.0)),
                            "grad/rms": float(grad_stats.get("grad/rms", 0.0)),
                            "grad/abs_mean": float(grad_stats.get("grad/abs_mean", 0.0)),
                            "grad/max_abs": float(grad_stats.get("grad/max_abs", 0.0)),
                            "grad/clip_coef": float(grad_stats.get("grad/clip_coef", 1.0)),
                            "grad/distribution_encoder_rms": float(grad_stats.get("grad/distribution_encoder_rms", 0.0)),
                            "grad/patch_tokenizer_rms": float(grad_stats.get("grad/patch_tokenizer_rms", 0.0)),
                            "grad/encoder_rms": float(grad_stats.get("grad/encoder_rms", 0.0)),
                            "grad/decoder_rms": float(grad_stats.get("grad/decoder_rms", 0.0)),
                            "grad/big_vae_other_rms": float(grad_stats.get("grad/big_vae_other_rms", 0.0)),
                            "grad/distribution_encoder_numel": float(grad_stats.get("grad/distribution_encoder_numel", 0.0)),
                            "grad/patch_tokenizer_numel": float(grad_stats.get("grad/patch_tokenizer_numel", 0.0)),
                            "grad/encoder_numel": float(grad_stats.get("grad/encoder_numel", 0.0)),
                            "grad/decoder_numel": float(grad_stats.get("grad/decoder_numel", 0.0)),
                            "grad/big_vae_other_numel": float(grad_stats.get("grad/big_vae_other_numel", 0.0)),
                            "grad/distribution_encoder_params_with_grad": float(
                                grad_stats.get("grad/distribution_encoder_params_with_grad", 0.0)
                            ),
                            "grad/patch_tokenizer_params_with_grad": float(
                                grad_stats.get("grad/patch_tokenizer_params_with_grad", 0.0)
                            ),
                            "grad/encoder_params_with_grad": float(grad_stats.get("grad/encoder_params_with_grad", 0.0)),
                            "grad/decoder_params_with_grad": float(grad_stats.get("grad/decoder_params_with_grad", 0.0)),
                            "grad/big_vae_other_params_with_grad": float(
                                grad_stats.get("grad/big_vae_other_params_with_grad", 0.0)
                            ),
                            "param/rms": float(grad_stats.get("param/rms", 0.0)),
                            "grad_to_param_rms_ratio": float(grad_stats.get("grad_to_param_rms_ratio", 0.0)),
                        }
                        for group_name, group_lr in lr_by_group.items():
                            comet_metrics[f"train/lr/{group_name}"] = float(group_lr)
                        if encoder_alpha_values:
                            comet_metrics["encoder_conditioning/alpha_mean"] = float(
                                sum(encoder_alpha_values) / max(1, len(encoder_alpha_values))
                            )
                            comet_metrics["encoder_conditioning/alpha_max_abs"] = float(
                                max(abs(value) for value in encoder_alpha_values)
                            )
                            for layer_idx, value in enumerate(encoder_alpha_values):
                                comet_metrics[f"encoder_conditioning/alpha_layer_{layer_idx}"] = float(value)
                        if patch_tokenizer_alpha_stats:
                            patch_abs_means = [float(stats["abs_mean"]) for stats in patch_tokenizer_alpha_stats]
                            patch_max_abs = [float(stats["max_abs"]) for stats in patch_tokenizer_alpha_stats]
                            comet_metrics["patch_tokenizer/alpha_abs_mean"] = float(
                                sum(patch_abs_means) / max(1, len(patch_abs_means))
                            )
                            comet_metrics["patch_tokenizer/alpha_max_abs"] = float(max(patch_max_abs))
                            for block_idx, stats in enumerate(patch_tokenizer_alpha_stats):
                                comet_metrics[f"patch_tokenizer/alpha_block_{block_idx}_mean"] = float(stats["mean"])
                                comet_metrics[f"patch_tokenizer/alpha_block_{block_idx}_abs_mean"] = float(
                                    stats["abs_mean"]
                                )
                                comet_metrics[f"patch_tokenizer/alpha_block_{block_idx}_max_abs"] = float(
                                    stats["max_abs"]
                                )
                        if patch_latent_variance_stats is not None:
                            comet_metrics["patch_latent_variance/mean"] = float(patch_latent_variance_stats["mean"])
                            comet_metrics["patch_latent_variance/std"] = float(patch_latent_variance_stats["std"])
                            comet_metrics["patch_latent_variance/min"] = float(patch_latent_variance_stats["min"])
                            comet_metrics["patch_latent_variance/max"] = float(patch_latent_variance_stats["max"])
                        if source_diversity_latest is not None:
                            comet_metrics["data/source_diversity/target_models"] = float(
                                source_diversity_latest.get("target_models", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_pool_size"] = float(
                                source_diversity_latest.get("source_pool_size", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_pool_unique_named_models"] = float(
                                source_diversity_latest.get("source_pool_unique_named_models", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_pool_missing_model_names"] = float(
                                source_diversity_latest.get("source_pool_missing_model_names", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_pool_remaining_slices_pre"] = float(
                                source_diversity_latest.get("source_pool_remaining_slices_pre", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_pool_remaining_slices_post"] = float(
                                source_diversity_latest.get("source_pool_remaining_slices_post", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_batch_sources_used"] = float(
                                source_diversity_latest.get("batch_sources_used", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_batch_unique_models"] = float(
                                source_diversity_latest.get("batch_unique_models", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_batch_unique_named_models"] = float(
                                source_diversity_latest.get("batch_unique_named_models", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_batch_missing_model_sources"] = float(
                                source_diversity_latest.get("batch_missing_model_sources", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_batch_model_entropy"] = float(
                                source_diversity_latest.get("batch_model_entropy", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_batch_model_perplexity"] = float(
                                source_diversity_latest.get("batch_model_perplexity", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_target_coverage"] = float(
                                source_diversity_latest.get("batch_model_target_coverage", 0.0)
                            )
                            comet_metrics["data/source_diversity/latest_model_shortfall"] = float(
                                source_diversity_latest.get("batch_model_shortfall", 0.0)
                            )
                            comet_metrics["data/source_diversity/window_batch_unique_models_mean"] = float(
                                diversity_unique_mean
                            )
                            comet_metrics["data/source_diversity/window_batch_unique_models_min"] = float(
                                diversity_unique_min
                            )
                            comet_metrics["data/source_diversity/window_batch_unique_models_max"] = float(
                                diversity_unique_max
                            )
                            comet_metrics["data/source_diversity/window_target_coverage_mean"] = float(
                                diversity_target_coverage_mean
                            )
                            comet_metrics["data/source_diversity/window_model_perplexity_mean"] = float(
                                diversity_model_perplexity_mean
                            )
                            comet_metrics["data/source_diversity/window_shortfall_steps"] = float(
                                source_diversity_window_shortfall_steps
                            )
                        if partwise_grad_clip_enabled:
                            for group_name in grad_group_prefixes:
                                comet_metrics[f"grad/{group_name}_global_norm_before_clip"] = float(
                                    grad_stats.get(f"grad/{group_name}_global_norm_before_clip", 0.0)
                                )
                                comet_metrics[f"grad/{group_name}_global_norm_after_clip"] = float(
                                    grad_stats.get(f"grad/{group_name}_global_norm_after_clip", 0.0)
                                )
                                comet_metrics[f"grad/{group_name}_clip_coef"] = float(
                                    grad_stats.get(f"grad/{group_name}_clip_coef", 1.0)
                                )
                                comet_metrics[f"grad/{group_name}_clip_norm_limit"] = float(
                                    grad_stats.get(f"grad/{group_name}_clip_norm_limit", 0.0)
                                )
                        if grad_layer_monitor_enabled:
                            comet_metrics["grad/layer_monitor_num_layers"] = float(
                                latest_grad_layer_snapshot.get("num_layers", 0)
                            )
                            comet_metrics["grad/layer_monitor_global_rms_pre_clip"] = float(
                                latest_grad_layer_snapshot.get("global_grad_rms_pre_clip", 0.0)
                            )
                            comet_metrics["grad/layer_monitor_global_rms_post_clip"] = float(
                                latest_grad_layer_snapshot.get("global_grad_rms_post_clip", 0.0)
                            )
                            comet_metrics["grad/layer_monitor_global_param_rms"] = float(
                                latest_grad_layer_snapshot.get("global_param_rms", 0.0)
                            )
                            comet_metrics["grad/layer_monitor_global_ratio_pre_clip"] = float(
                                latest_grad_layer_snapshot.get("global_grad_to_param_ratio_pre_clip", 0.0)
                            )
                            comet_metrics["grad/layer_monitor_global_ratio_post_clip"] = float(
                                latest_grad_layer_snapshot.get("global_grad_to_param_ratio_post_clip", 0.0)
                            )
                        if device.type == "cuda":
                            comet_metrics["gpu/memory_allocated_mb"] = float(
                                torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                            )
                            comet_metrics["gpu/memory_reserved_mb"] = float(
                                torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
                            )
                            comet_metrics["gpu/max_memory_allocated_mb"] = float(
                                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                            )
                        if comet_tracker is not None and comet_tracker.enabled:
                            comet_tracker.log_metrics(comet_metrics, step=global_step)
                        if wandb_tracker is not None and wandb_tracker.enabled:
                            wandb_tracker.log_metrics(comet_metrics, step=global_step)
                    if collector is not None and (global_step % log_worker_status_every == 0):
                        try:
                            collector_status = _build_collector_status_snapshot(collector)
                            logger.info(
                                "collector_status step=%s status=%s",
                                global_step,
                                json.dumps(collector_status, ensure_ascii=False),
                            )
                        except Exception:
                            logger.exception("Failed to capture collector_status at step=%s", global_step)

                    loss_window = 0.0
                    behavioral_window = 0.0
                    behavioral_operator_window = 0.0
                    behavioral_dir_window = 0.0
                    behavioral_scale_window = 0.0
                    structural_window = 0.0
                    kl_window = 0.0
                    struct_dir_window = 0.0
                    struct_scale_window = 0.0
                    struct_rec_window = 0.0
                    struct_rel_window = 0.0
                    data_build_window_s = 0.0
                    data_wait_window_s = 0.0
                    data_h2d_window_s = 0.0
                    data_prefetch_depth_window = 0.0
                    source_diversity_window_unique_models_sum = 0.0
                    source_diversity_window_unique_models_min = math.inf
                    source_diversity_window_unique_models_max = 0.0
                    source_diversity_window_target_coverage_sum = 0.0
                    source_diversity_window_model_perplexity_sum = 0.0
                    source_diversity_window_shortfall_steps = 0
                    window_steps = 0
                    t0 = time.time()

                if global_step % forensics_heartbeat_steps == 0:
                    heartbeat_payload: dict[str, Any] = {
                        "type": "heartbeat",
                        "pid": int(os.getpid()),
                        "role": f"train_rank_{rank}",
                        "metadata": {
                            "rank": int(rank),
                            "world_size": int(world_size),
                            "global_step": int(global_step),
                            "max_steps": int(max_steps),
                            "loss": float(step_loss),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        },
                    }
                    if rank == 0 and collector is not None:
                        heartbeat_payload["tracked_children"] = _collector_tracked_children(collector)
                    monitor_send_event(monitor_queue, heartbeat_payload)

                should_save_model_checkpoint = (global_step % checkpoint_every == 0 or global_step == max_steps)
                should_save_resume_state = resume_state_enabled and (
                    global_step % resume_state_save_every == 0 or global_step == max_steps
                )
                if rank == 0 and should_save_model_checkpoint:
                    _save_checkpoint(
                        model=model,
                        cfg=cfg,
                        step_idx=global_step,
                        logger=logger,
                        stage=stage_num,
                    )
                if rank == 0 and should_save_resume_state:
                    _save_resume_state_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        cfg=cfg,
                        step_idx=global_step,
                        logger=logger,
                        stage=stage_num,
                        state_dir=resume_state_dir,
                    )

            if offline_batch_prefetcher is not None:
                offline_batch_prefetcher.close()
            if rank == 0:
                logger.info("Training completed successfully: steps=%s", max_steps)

    except BaseException as exc:
        failed = True
        prefetcher = locals().get("offline_batch_prefetcher")
        if isinstance(prefetcher, BackgroundPrefetcher):
            prefetcher.close()
        traceback_text = traceback.format_exc()
        report_path = emit_fatal_report(
            cfg_dict,
            role=f"train_rank_{rank}",
            error=str(exc),
            traceback_text=traceback_text,
            extra={
                "rank": int(rank),
                "world_size": int(world_size),
            },
            section="train",
        )
        monitor_send_event(
            monitor_queue,
            {
                "type": "fatal",
                "pid": int(os.getpid()),
                "role": f"train_rank_{rank}",
                "error": str(exc),
                "traceback": traceback_text,
                "metadata": {
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "fatal_report_path": report_path,
                },
            },
        )
        logger.exception(
            "Training worker exiting due to unhandled exception (rank=%s, world_size=%s)",
            rank,
            world_size,
        )
        raise
    finally:
        if comet_tracker is not None:
            comet_tracker.end()
        if wandb_tracker is not None:
            wandb_tracker.end()
        monitor_send_event(
            monitor_queue,
            {
                "type": "exit",
                "pid": int(os.getpid()),
                "role": f"train_rank_{rank}",
                "metadata": {
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "failed": bool(failed),
                },
            },
        )
        if is_distributed and dist.is_initialized():
            if not failed:
                try:
                    dist.barrier()
                except Exception:
                    pass
            else:
                logger.warning("Skipping dist.barrier during shutdown because this rank failed")
            dist.destroy_process_group()
