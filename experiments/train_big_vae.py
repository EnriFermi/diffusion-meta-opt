from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


from dataset import data_pipeline, setup_logging
from dataset.logging_utils import LOG_PATH_ENV, resolve_log_path
from models.weight_quantile_vae import (
    BigVAEConfig,
    DistributionConfig,
    EncoderConfig,
    MiniVAEConfig,
    ModelConfig,
    WeightQuantileVAE,
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


def _promote_run_profile_to_root(cfg: DictConfig) -> None:
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return

    expected_sections = (
        "data",
        "collector",
        "streaming",
        "train",
        "model",
        "training_artifacts",
        "logging",
        "hf",
        "models",
    )
    with open_dict(cfg):
        for section in expected_sections:
            if section in cfg:
                continue
            if section in run_profiles_cfg:
                cfg[section] = run_profiles_cfg[section]


def _logger(name: str, rank: int) -> logging.Logger:
    return get_rank_logger(name, rank)


def _seed_everything(seed: int) -> None:
    runtime_seed_everything(seed)


def _find_free_port() -> int:
    return runtime_find_free_port()


def _resolve_world_size(cfg: DictConfig) -> int:
    return runtime_resolve_world_size(cfg, section="train", error_prefix="train")


def _resolve_backend(cfg: DictConfig, device: torch.device) -> str:
    return runtime_resolve_backend(cfg, device, section="train")


def _resolve_device(cfg: DictConfig, rank: int, world_size: int) -> torch.device:
    return runtime_resolve_device(cfg, rank, world_size, section="train")


def _set_speed_optimizations(cfg: DictConfig, device: torch.device) -> None:
    runtime_set_speed_optimizations(cfg, device, section="train")


def _build_model_cfg(cfg: DictConfig) -> ModelConfig:
    model_cfg = cfg.get("model", {})

    dist_cfg = model_cfg.get("distribution", {})
    mini_cfg = model_cfg.get("mini_vae", {})
    big_cfg = model_cfg.get("big_vae", {})
    enc_cfg = big_cfg.get("encoder", {})

    return ModelConfig(
        patch_size=int(model_cfg.get("patch_size", 16)),
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
            patch_size_for_cov=int(dist_cfg.get("patch_size_for_cov", int(model_cfg.get("patch_size", 16)))),
        ),
        mini_vae=MiniVAEConfig(
            z_dim=int(mini_cfg.get("z_dim", 64)),
            d_e=int(mini_cfg.get("d_e", 128)),
            num_attn_layers_encoder=int(mini_cfg.get("num_attn_layers_encoder", 2)),
            num_layers_decoder=int(mini_cfg.get("num_layers_decoder", 2)),
            decoder_bilinear_rank=int(mini_cfg.get("decoder_bilinear_rank", 0)),
            stub_resampler_d_model=int(mini_cfg.get("stub_resampler_d_model", 0)),
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
            encoder=EncoderConfig(
                self_attn_mode=str(enc_cfg.get("self_attn_mode", "full")),
                cross_attend_only_cls=bool(enc_cfg.get("cross_attend_only_cls", True)),
            ),
        ),
    )


def _maybe_compile(model: torch.nn.Module, cfg: DictConfig, logger: logging.Logger) -> torch.nn.Module:
    return maybe_compile_model(model, cfg, logger, section="train", label="model")


def _build_optimizer(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    return build_adamw_optimizer(
        model=model,
        cfg=cfg,
        device=device,
        section="train",
        default_lr=3e-4,
        default_weight_decay=0.01,
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LambdaLR:
    return build_cosine_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        section="train",
        default_max_steps=1000,
        default_warmup_steps=100,
        default_min_lr_ratio=0.1,
    )


def _resolve_amp(cfg: DictConfig, device: torch.device) -> tuple[bool, torch.dtype | None]:
    return runtime_resolve_amp(cfg, device, section="train")


def _autocast_context(enabled: bool, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    return runtime_autocast_context(enabled, dtype)


def _configure_run_artifacts(cfg: DictConfig) -> dict[str, str]:
    return runtime_configure_per_run_artifacts(cfg, run_label="train_big_vae")


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
            from comet_ml import Experiment, OfflineExperiment  # type: ignore
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

        api_key = str(comet_cfg.get("api_key", "")).strip()
        workspace = str(comet_cfg.get("workspace", "")).strip()
        project_name = str(comet_cfg.get("project_name", "big_weight_vae")).strip() or "big_weight_vae"
        experiment_name = str(comet_cfg.get("experiment_name", "")).strip()
        offline_dir = str(comet_cfg.get("offline_directory", "")).strip()
        log_code = bool(comet_cfg.get("log_code", False))

        try:
            train_cfg = cfg.get("train", {})
            model_cfg = cfg.get("model", {})
            big_cfg = model_cfg.get("big_vae", {})
            streaming_cfg = cfg.get("streaming", {})
            collector_cfg = cfg.get("collector", {})
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
            if isinstance(tags, (list, tuple)):
                for tag in tags:
                    exp.add_tag(str(tag))

            exp.log_parameters(
                {
                    "train.max_steps": int(train_cfg.get("max_steps", 0)),
                    "train.lr": float(train_cfg.get("lr", 0.0)),
                    "train.grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
                    "train.kl_beta": float(train_cfg.get("kl_beta", 0.0)),
                    "train.behavioral_coef": float(train_cfg.get("behavioral_coef", 0.0)),
                    "train.structural_coef": float(train_cfg.get("structural_coef", 0.0)),
                    "model.patch_size": int(model_cfg.get("patch_size", 16)),
                    "model.big_vae.use_latent_sampling": bool(big_cfg.get("use_latent_sampling", True)),
                    "streaming.mode": str(streaming_cfg.get("mode", "none")),
                    "collector.mode": str(collector_cfg.get("mode", "auto")),
                    "collector.device": str(collector_cfg.get("device", "")),
                    "train.device": str(train_cfg.get("device", "")),
                }
            )

            self.experiment = exp
            self.enabled = True
            self.logger.info("Comet tracking enabled: project=%s workspace=%s", project_name, workspace or "<default>")
        except Exception as exc:
            self.logger.warning("Failed to initialize Comet tracker: %s", exc)
            self.experiment = None
            self.enabled = False

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


def _build_collector_status_snapshot(collector: Any) -> dict[str, Any]:
    stats = collector.stats()
    payload: dict[str, Any] = {
        "mode": stats.get("mode"),
        "streaming_mode": stats.get("streaming_mode"),
        "cache_size": int(stats.get("cache_size", 0)),
        "jobs_total": int(stats.get("jobs_total", 0)),
        "items_emitted": int(stats.get("items_emitted", 0)),
        "async_process_alive": stats.get("async_process_alive"),
        "async_events_dropped": stats.get("async_events_dropped"),
        "worker_status_dir": stats.get("worker_status_dir"),
    }

    async_last_status = stats.get("async_last_status")
    if not isinstance(async_last_status, dict):
        return payload

    worker_total = 0
    worker_alive = 0
    worker_disabled = 0
    degraded_workers: list[dict[str, Any]] = []
    raw_workers = async_last_status.get("raw_pool_workers")
    if isinstance(raw_workers, dict):
        for dataset_name, info in raw_workers.items():
            if not isinstance(info, dict):
                continue
            worker_total += 1
            is_alive = bool(info.get("worker_alive", False))
            is_disabled = bool(info.get("disabled", False) or info.get("worker_permanently_stopped", False))
            if is_alive:
                worker_alive += 1
            if is_disabled:
                worker_disabled += 1
            if (not is_alive) or is_disabled or bool(info.get("worker_last_error")):
                degraded_workers.append(
                    {
                        "dataset": str(dataset_name),
                        "worker_pid": info.get("worker_pid"),
                        "worker_alive": is_alive,
                        "disabled": is_disabled,
                        "worker_restarts": int(info.get("worker_restarts", 0) or 0),
                        "worker_last_error": info.get("worker_last_error"),
                    }
                )

    if worker_total > 0:
        payload["dataset_workers_total"] = int(worker_total)
        payload["dataset_workers_alive"] = int(worker_alive)
        payload["dataset_workers_disabled_or_stopped"] = int(worker_disabled)
    if degraded_workers:
        payload["dataset_workers_degraded"] = degraded_workers[:8]

    return payload


def _collector_tracked_children(collector: Any) -> list[dict[str, Any]]:
    tracked: list[dict[str, Any]] = []
    try:
        stats = collector.stats()
    except Exception:
        return tracked

    async_last_status = stats.get("async_last_status")
    if not isinstance(async_last_status, dict):
        return tracked

    collector_pid = async_last_status.get("collector_pid")
    if isinstance(collector_pid, int) and collector_pid > 0:
        tracked.append(
            {
                "pid": int(collector_pid),
                "role": "collector_process",
                "metadata": {
                    "mode": stats.get("mode"),
                    "streaming_mode": stats.get("streaming_mode"),
                },
            }
        )

    raw_workers = async_last_status.get("raw_pool_workers")
    if isinstance(raw_workers, dict):
        for dataset_name, info in raw_workers.items():
            if not isinstance(info, dict):
                continue
            worker_pid = info.get("worker_pid")
            if not isinstance(worker_pid, int) or worker_pid <= 0:
                continue
            tracked.append(
                {
                    "pid": int(worker_pid),
                    "role": "dataset_worker",
                    "metadata": {
                        "dataset": str(dataset_name),
                        "worker_alive": bool(info.get("worker_alive", False)),
                        "worker_restarts": int(info.get("worker_restarts", 0) or 0),
                        "worker_last_error": info.get("worker_last_error"),
                    },
                }
            )
    return tracked


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
    stage: int = 1,
) -> None:
    base_dir = Path(str(cfg.train.get("checkpoint_dir", "./checkpoints/weight_quantile_vae")))
    checkpoint_dir = base_dir / f"stage_{stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step_idx,
        "stage": stage,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    step_path = checkpoint_dir / f"step_{step_idx:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    logger.info("Checkpoint saved: %s", step_path)


def _next_valid_sample(
    dataset_iter: Iterator[Any],
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    attempts = 0
    while True:
        sample = next(dataset_iter)
        x = sample.x
        W = sample.weight

        valid = (
            torch.is_tensor(x)
            and torch.is_tensor(W)
            and x.ndim == 2
            and W.ndim == 2
            and x.shape[1] == W.shape[0]
            and x.shape[0] > 0
            and W.shape[0] > 0
            and W.shape[1] > 0
        )
        if not valid:
            attempts += 1
            if attempts % 100 == 0:
                logger.warning("Skipping invalid sample repeatedly; attempts=%s", attempts)
            continue

        x = x.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
        W = W.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()

        if max_x_rows > 0 and x.shape[0] > max_x_rows:
            keep = torch.randperm(x.shape[0])[:max_x_rows]
            x = x[keep]

        return x, W


def _broadcast_tensor_2d(
    tensor: torch.Tensor | None,
    device: torch.device,
    src: int = 0,
) -> torch.Tensor:
    if not dist.is_initialized():
        if tensor is None:
            raise ValueError("tensor cannot be None when distributed is not initialized")
        return tensor.to(device=device, non_blocking=True)

    rank = dist.get_rank()
    shape = torch.zeros(2, dtype=torch.long, device=device)
    if rank == src:
        if tensor is None:
            raise ValueError("Source rank must provide tensor")
        if tensor.ndim != 2:
            raise ValueError(f"Expected rank-2 tensor, got shape {tuple(tensor.shape)}")
        shape[0] = tensor.shape[0]
        shape[1] = tensor.shape[1]

    dist.broadcast(shape, src=src)

    if rank == src:
        payload = tensor.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    else:
        payload = torch.empty((int(shape[0].item()), int(shape[1].item())), device=device, dtype=torch.float32)

    dist.broadcast(payload, src=src)
    return payload


def _fetch_batch(
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_broadcast:
        x_cpu: torch.Tensor | None = None
        W_cpu: torch.Tensor | None = None
        if rank == 0:
            if dataset_iter is None:
                raise RuntimeError("rank0 requires dataset iterator in broadcast mode")
            x_cpu, W_cpu = _next_valid_sample(
                dataset_iter=dataset_iter,
                max_x_rows=max_x_rows,
                logger=logger,
            )

        x = _broadcast_tensor_2d(x_cpu, device=device, src=0)
        W = _broadcast_tensor_2d(W_cpu, device=device, src=0)
        return x, W

    if dataset_iter is None:
        raise RuntimeError("dataset iterator is required for sharded mode")

    x_cpu, W_cpu = _next_valid_sample(
        dataset_iter=dataset_iter,
        max_x_rows=max_x_rows,
        logger=logger,
    )
    return (
        x_cpu.to(device=device, non_blocking=True),
        W_cpu.to(device=device, non_blocking=True),
    )


def _compute_curriculum_slice_sizes(cfg: DictConfig) -> tuple[int, int]:
    train_cfg = cfg.get("train", {})
    stage = max(1, int(train_cfg.get("stage", 1)))
    base_T = int(train_cfg.get("stage_base_T_patches", 4))
    base_d_out = int(train_cfg.get("stage_base_d_out", 16))
    scale = int(train_cfg.get("stage_scale_factor", 2))
    max_T = base_T * (scale ** (stage - 1))
    max_d_out = base_d_out * (scale ** (stage - 1))
    return max_T, max_d_out


def _slice_sample(
    W: torch.Tensor,
    x: torch.Tensor,
    max_T_patches: int,
    max_d_out: int,
    patch_size: int,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_in, d_out = W.shape
    T_total = d_in // patch_size
    T_use = min(max_T_patches, T_total)
    d_out_use = min(max_d_out, d_out)

    need_row_slice = T_use < T_total
    need_col_slice = d_out_use < d_out
    dev = W.device

    if not need_row_slice and not need_col_slice:
        return W.unsqueeze(0).expand(batch_size, -1, -1), x.unsqueeze(0).expand(batch_size, -1, -1)

    offsets = torch.arange(patch_size, device=dev) if need_row_slice else None
    W_slices = []
    x_slices = []
    for _ in range(batch_size):
        W_i = W
        x_i = x
        if need_row_slice:
            patch_idx = torch.randperm(T_total, device=dev)[:T_use].sort().values
            row_idx = (patch_idx.unsqueeze(1) * patch_size + offsets.unsqueeze(0)).flatten()
            row_idx = row_idx.clamp(max=d_in - 1)
            W_i = W_i[row_idx, :]
            x_i = x_i[:, row_idx]
        if need_col_slice:
            col_idx = torch.randperm(d_out, device=dev)[:d_out_use].sort().values
            W_i = W_i[:, col_idx]
        W_slices.append(W_i)
        x_slices.append(x_i)

    return torch.stack(W_slices), torch.stack(x_slices)


def _load_model_weights_from_checkpoint(
    model: torch.nn.Module,
    path: str,
    logger: logging.Logger,
) -> None:
    logger.info("Loading model weights from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"]
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(state_dict, strict=True)
    logger.info("Model weights loaded successfully (step=%s)", ckpt.get("step", "?"))


def _run_worker(
    rank: int,
    world_size: int,
    cfg_dict: dict[str, Any],
    master_addr: str,
    master_port: int,
    monitor_queue: Any | None = None,
) -> None:
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

    if dataset_sharding:
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

    model = None
    optimizer = None
    scheduler = None
    scaler = None
    comet_tracker: CometTracker | None = None

    failed = False
    try:
        collector: Any | None = None
        dataset: Any | None = None
        dataset_iter: Iterator[Any] | None = None

        with contextlib.ExitStack() as stack:
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
            model = WeightQuantileVAE(model_cfg).to(device)
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

            resume_checkpoint = str(cfg.train.get("resume_checkpoint", "")).strip()
            if resume_checkpoint:
                _load_model_weights_from_checkpoint(model, resume_checkpoint, logger)

            optimizer = _build_optimizer(model=model, cfg=cfg, device=device)
            scheduler = _build_scheduler(optimizer=optimizer, cfg=cfg)
            comet_tracker = CometTracker(cfg=cfg, logger=logger, rank=rank)

            amp_enabled, amp_dtype = _resolve_amp(cfg=cfg, device=device)
            scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
            cudagraph_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
            use_cudagraph_step_begin = (
                bool(cfg.train.get("compile", False))
                and device.type == "cuda"
                and callable(cudagraph_step_begin)
            )

            max_steps = max(1, int(cfg.train.get("max_steps", 1000)))
            grad_accum_steps = max(1, int(cfg.train.get("grad_accum_steps", 1)))
            kl_beta = float(cfg.train.get("kl_beta", 1e-3))
            model_unwrapped = model.module if isinstance(model, DDP) else model
            cfg_holder = model_unwrapped
            if not hasattr(cfg_holder, "cfg") and hasattr(cfg_holder, "_orig_mod"):
                cfg_holder = getattr(cfg_holder, "_orig_mod")
            if not hasattr(cfg_holder, "cfg"):
                raise AttributeError(f"Model does not expose cfg: type={type(model_unwrapped)}")
            use_latent_sampling = bool(cfg_holder.cfg.big_vae.use_latent_sampling)
            behavioral_coef = float(cfg.train.get("behavioral_coef", 1.0))
            structural_coef = float(cfg.train.get("structural_coef", 0.5))
            grad_clip_norm = float(cfg.train.get("grad_clip_norm", 1.0))
            max_x_rows = int(cfg.train.get("max_x_rows", 0))

            patch_size_for_slice = int(cfg.model.get("patch_size", 16))
            curriculum_max_T, curriculum_max_d_out = _compute_curriculum_slice_sizes(cfg)
            stage_num = max(1, int(cfg.train.get("stage", 1)))
            slice_batch_size = max(1, int(cfg.train.get("slice_batch_size", 1)))
            if rank == 0:
                logger.info(
                    "Curriculum slicing: stage=%s max_T_patches=%s max_d_out=%s patch_size=%s slice_batch_size=%s",
                    stage_num, curriculum_max_T, curriculum_max_d_out, patch_size_for_slice, slice_batch_size,
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
            checkpoint_every = max(1, int(cfg.train.get("checkpoint_every", 200)))

            steps_per_sample = max(1, int(cfg.train.get("steps_per_sample", 1)))
            if rank == 0:
                logger.info("Steps per sample: %s", steps_per_sample)
                logger.info(
                    "BigVAE latent mode: %s (use_latent_sampling=%s, kl_beta=%s)",
                    "VAE" if use_latent_sampling else "AE",
                    use_latent_sampling,
                    kl_beta,
                )

            loss_window = 0.0
            behavioral_window = 0.0
            structural_window = 0.0
            kl_window = 0.0
            window_steps = 0
            t0 = time.time()

            current_x: torch.Tensor | None = None
            current_W: torch.Tensor | None = None

            for step_idx in range(max_steps):
                global_step = step_idx + 1
                model.train()
                optimizer.zero_grad(set_to_none=True)

                if rank == 0 and collector is not None and dataset is not None and not collector.is_async_mode:
                    dataset.maybe_collect(step_idx)

                if current_x is None or step_idx % steps_per_sample == 0:
                    current_x, current_W = _fetch_batch(
                        rank=rank,
                        device=device,
                        dataset_iter=dataset_iter,
                        use_broadcast=use_broadcast,
                        max_x_rows=max_x_rows,
                        logger=logger,
                    )

                loss_acc = 0.0
                behavioral_acc = 0.0
                structural_acc = 0.0
                kl_acc = 0.0
                step_is_finite = True

                for micro_idx in range(grad_accum_steps):
                    sync_grad = micro_idx == grad_accum_steps - 1

                    W_s, x_s = _slice_sample(current_W, current_x, curriculum_max_T, curriculum_max_d_out, patch_size_for_slice, batch_size=slice_batch_size)

                    no_sync_ctx = contextlib.nullcontext()
                    if is_distributed and not sync_grad:
                        no_sync_ctx = model.no_sync()  # type: ignore[union-attr]

                    with no_sync_ctx:
                        if use_cudagraph_step_begin:
                            cudagraph_step_begin()
                        with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                            W_hat, mu, logvar = model(W_s, x_s)
                            behavioral_loss = WeightQuantileVAE.operator_recon_loss(x_s, W_s, W_hat)
                            structural_loss = WeightQuantileVAE.structural_recon_loss(W_s, W_hat)
                            if use_latent_sampling:
                                kl_loss = WeightQuantileVAE.kl_loss(mu, logvar)
                            else:
                                kl_loss = mu.new_zeros(())
                            total_loss = (
                                behavioral_coef * behavioral_loss
                                + structural_coef * structural_loss
                                + kl_beta * kl_loss
                            )
                            loss_for_backward = total_loss / grad_accum_steps

                        finite_flag = torch.tensor(
                            1 if torch.isfinite(loss_for_backward.detach()) else 0,
                            dtype=torch.int32,
                            device=device,
                        )
                        if is_distributed:
                            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

                        if int(finite_flag.item()) == 0:
                            step_is_finite = False
                        else:
                            if scaler.is_enabled():
                                scaler.scale(loss_for_backward).backward()
                            else:
                                loss_for_backward.backward()

                    loss_acc += float(total_loss.detach().item())
                    behavioral_acc += float(behavioral_loss.detach().item())
                    structural_acc += float(structural_loss.detach().item())
                    kl_acc += float(kl_loss.detach().item())

                    if not step_is_finite:
                        break

                if not step_is_finite:
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        logger.warning("Skipping step %s due to non-finite loss", global_step)
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()

                step_loss = loss_acc / grad_accum_steps
                step_behavioral = behavioral_acc / grad_accum_steps
                step_structural = structural_acc / grad_accum_steps
                step_kl = kl_acc / grad_accum_steps

                stats = torch.tensor(
                    [step_loss, step_behavioral, step_structural, step_kl],
                    dtype=torch.float32, device=device,
                )
                if is_distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    stats /= float(world_size)

                loss_window += float(stats[0].item())
                behavioral_window += float(stats[1].item())
                structural_window += float(stats[2].item())
                kl_window += float(stats[3].item())
                window_steps += 1

                if rank == 0 and global_step % log_every == 0:
                    dt = max(1e-6, time.time() - t0)
                    avg_loss = loss_window / max(1, window_steps)
                    avg_behavioral = behavioral_window / max(1, window_steps)
                    avg_structural = structural_window / max(1, window_steps)
                    avg_kl = kl_window / max(1, window_steps)
                    lr = float(optimizer.param_groups[0]["lr"])
                    speed = window_steps / dt

                    cache_metric = dataset.cache_size() if dataset is not None else 0
                    logger.info(
                        "step=%s/%s loss=%.6f behav=%.6f struct=%.6f kl=%.6f lr=%.6e steps/s=%.2f cache=%s",
                        global_step,
                        max_steps,
                        avg_loss,
                        avg_behavioral,
                        avg_structural,
                        avg_kl,
                        lr,
                        speed,
                        cache_metric,
                    )
                    if comet_tracker is not None and comet_tracker.enabled:
                        comet_metrics: dict[str, float] = {
                            "train/loss": float(avg_loss),
                            "train/behavioral_loss": float(avg_behavioral),
                            "train/structural_loss": float(avg_structural),
                            "train/kl_loss": float(avg_kl),
                            "train/lr": float(lr),
                            "train/steps_per_sec": float(speed),
                            "data/cache_size": float(cache_metric),
                        }
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
                        comet_tracker.log_metrics(comet_metrics, step=global_step)
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
                    structural_window = 0.0
                    kl_window = 0.0
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

                if rank == 0 and (global_step % checkpoint_every == 0 or global_step == max_steps):
                    _save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        cfg=cfg,
                        step_idx=global_step,
                        logger=logger,
                        stage=stage_num,
                    )

            if rank == 0:
                logger.info("Training completed successfully: steps=%s", max_steps)

    except BaseException as exc:
        failed = True
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


def _spawn_entry(
    rank: int,
    world_size: int,
    cfg_dict: dict[str, Any],
    master_addr: str,
    master_port: int,
    monitor_queue: Any | None = None,
) -> None:
    _run_worker(
        rank=rank,
        world_size=world_size,
        cfg_dict=cfg_dict,
        master_addr=master_addr,
        master_port=master_port,
        monitor_queue=monitor_queue,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _promote_run_profile_to_root(cfg)
    run_artifacts = _configure_run_artifacts(cfg)
    maybe_redirect_stdio(cfg, role="train_launcher", section="train")
    maybe_enable_core_dumps(cfg, section="train", logger=None)
    print(
        f"[train_big_vae] run_id={run_artifacts.get('run_id')} artifacts_root={run_artifacts.get('root_dir')}",
        flush=True,
    )
    train_cfg = cfg.get("train", {})
    print(
        "[train_big_vae] effective train config: "
        f"distributed={train_cfg.get('distributed')} "
        f"num_gpus={train_cfg.get('num_gpus')} "
        f"device={train_cfg.get('device')} "
        f"compile={train_cfg.get('compile')} "
        f"compile_mode={train_cfg.get('compile_mode')} "
        f"compile_dynamic={train_cfg.get('compile_dynamic')}",
        flush=True,
    )
    # Freeze one shared log path before spawning worker processes.
    if not os.environ.get(LOG_PATH_ENV):
        os.environ[LOG_PATH_ENV] = str(resolve_log_path(cfg))

    world_size = _resolve_world_size(cfg)
    print(f"[train_big_vae] resolved world_size={world_size}", flush=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    apply_nccl_forensics_env(cfg_dict, section="train")
    monitor_process, monitor_queue, monitor_stop_event = start_process_monitor(
        cfg_dict,
        section="train",
        label="train_big_vae",
    )
    monitor_send_event(
        monitor_queue,
        {
            "type": "register",
            "pid": int(os.getpid()),
            "role": "train_launcher",
            "metadata": {
                "world_size": int(world_size),
            },
        },
    )

    try:
        if world_size <= 1:
            _run_worker(
                rank=0,
                world_size=1,
                cfg_dict=cfg_dict,
                master_addr="127.0.0.1",
                master_port=0,
                monitor_queue=monitor_queue,
            )
            return

        master_addr = str(cfg.train.get("master_addr", "127.0.0.1"))
        requested_port = int(cfg.train.get("master_port", 0))
        master_port = requested_port if requested_port > 0 else _find_free_port()

        mp.spawn(
            _spawn_entry,
            args=(world_size, cfg_dict, master_addr, master_port, monitor_queue),
            nprocs=world_size,
            join=True,
        )
    except BaseException as exc:
        traceback_text = traceback.format_exc()
        report_path = emit_fatal_report(
            cfg_dict,
            role="train_launcher",
            error=str(exc),
            traceback_text=traceback_text,
            extra={"world_size": int(world_size)},
            section="train",
        )
        monitor_send_event(
            monitor_queue,
            {
                "type": "fatal",
                "pid": int(os.getpid()),
                "role": "train_launcher",
                "error": str(exc),
                "traceback": traceback_text,
                "metadata": {
                    "world_size": int(world_size),
                    "fatal_report_path": report_path,
                },
            },
        )
        raise
    finally:
        monitor_send_event(
            monitor_queue,
            {
                "type": "exit",
                "pid": int(os.getpid()),
                "role": "train_launcher",
                "metadata": {
                    "world_size": int(world_size),
                },
            },
        )
        stop_process_monitor(monitor_process, monitor_queue, monitor_stop_event)


if __name__ == "__main__":
    main()
