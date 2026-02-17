from __future__ import annotations

import contextlib
import inspect
import json
import logging
import math
import os
import random
import socket
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset
from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
from dataset.shared.streaming.factory import build_chunk_store, resolve_streaming_cfg
from dataset.shared.types import SharedSample
from mini_vae_pipeline.model import MiniPatchTrainingModel, build_distribution_config, build_mini_vae_config


# ---------------------------
# Logging / telemetry helpers
# ---------------------------
def setup_logging(cfg: DictConfig, rank: int = 0) -> None:
    level_name = str(cfg.data.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    if rank != 0:
        level = max(level, logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def get_logger(name: str, rank: int) -> logging.Logger:
    return logging.getLogger(f"{name}.rank{rank}")


class CometTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger, rank: int) -> None:
        self.logger = logger
        self.rank = rank
        self.experiment: Any | None = None
        self.enabled = False

        telemetry_cfg = cfg.mini_train.get("telemetry", {})
        comet_cfg = telemetry_cfg.get("comet", {})
        if not bool(comet_cfg.get("enabled", False)):
            return
        if rank != 0:
            return

        try:
            from comet_ml import Experiment, OfflineExperiment  # type: ignore
        except Exception as exc:
            self.logger.warning("Comet is enabled but comet_ml is unavailable: %s", exc)
            return

        api_key = str(comet_cfg.get("api_key", "")).strip()
        workspace = str(comet_cfg.get("workspace", "")).strip()
        project_name = str(comet_cfg.get("project_name", "mini_patch_vae")).strip() or "mini_patch_vae"
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
            if isinstance(tags, (list, tuple)):
                for tag in tags:
                    exp.add_tag(str(tag))

            exp.log_parameters(
                {
                    "mini_train.max_steps": int(cfg.mini_train.get("max_steps", 0)),
                    "mini_train.lr": float(cfg.mini_train.get("lr", 0.0)),
                    "mini_train.grad_accum_steps": int(cfg.mini_train.get("grad_accum_steps", 1)),
                    "mini_train.patches_per_sample": int(cfg.mini_train.get("patches_per_sample", 16)),
                    "mini_model.patch_size": int(cfg.mini_model.get("patch_size", 64)),
                    "streaming.mode": str(cfg.streaming.get("mode", "none")),
                    "collector.mode": str(cfg.collector.get("mode", "auto")),
                    "collector.device": str(cfg.collector.get("device", "")),
                    "mini_train.device": str(cfg.mini_train.get("device", "")),
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

    def log_text(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_text(text, metadata=metadata or {})
        except Exception as exc:
            self.logger.warning("Comet text log failed: %s", exc)

    def end(self) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.end()
        except Exception:
            pass


class JsonlStatusWriter:
    def __init__(self, path: str | Path, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ---------------------------
# Metrics / status helpers
# ---------------------------
def extract_sample_debug_info(sample: SharedSample) -> dict[str, Any]:
    meta = sample.meta if isinstance(sample.meta, dict) else {}
    image_meta = meta.get("image_meta", [])
    datasets: set[str] = set()
    if isinstance(image_meta, list):
        for item in image_meta:
            if isinstance(item, dict) and item.get("dataset_name") is not None:
                datasets.add(str(item.get("dataset_name")))

    row_preview = meta.get("selected_row_indices_preview")
    row_preview_trimmed: list[int] = []
    if isinstance(row_preview, list):
        for item in row_preview[:8]:
            try:
                row_preview_trimmed.append(int(item))
            except Exception:
                continue

    model_run_id = meta.get("model_run_id")
    model_run_id_int: int | None = None
    if model_run_id is not None:
        try:
            model_run_id_int = int(model_run_id)
        except Exception:
            model_run_id_int = None

    selected_row_count = meta.get("selected_row_count")
    selected_row_count_int: int | None = None
    if selected_row_count is not None:
        try:
            selected_row_count_int = int(selected_row_count)
        except Exception:
            selected_row_count_int = None

    return {
        "model_name": str(sample.model_name),
        "layer_name": str(sample.layer_name),
        "datasets": sorted(datasets),
        "model_run_id": model_run_id_int,
        "xy_sampling_mode": meta.get("xy_sampling_mode"),
        "selected_row_count": selected_row_count_int,
        "selected_row_indices_preview": row_preview_trimmed,
        "x_rows": int(sample.x.shape[0]) if torch.is_tensor(sample.x) and sample.x.ndim >= 1 else None,
        "d_in": int(sample.x.shape[1]) if torch.is_tensor(sample.x) and sample.x.ndim == 2 else None,
        "d_out": int(sample.weight.shape[1]) if torch.is_tensor(sample.weight) and sample.weight.ndim == 2 else None,
    }


def compute_grad_stats(model: nn.Module) -> dict[str, float]:
    groups = {
        "distribution_encoder": ("distribution_encoder.",),
        "mini_encoder": ("mini_vae.encoder.",),
        "mini_decoder": ("mini_vae.decoder.",),
        "mini_vae_other": ("mini_vae.",),
    }
    group_sums = {name: 0.0 for name in groups}
    group_numel = {name: 0 for name in groups}

    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    grad_numel = 0
    param_sum_sq = 0.0
    param_numel = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_data = param.detach()
        param_sum_sq += float(param_data.pow(2).sum().item())
        param_numel += int(param_data.numel())

        grad = param.grad
        if grad is None:
            continue

        g = grad.detach()
        abs_g = g.abs()
        sum_sq += float((g * g).sum().item())
        sum_abs += float(abs_g.sum().item())
        max_abs = max(max_abs, float(abs_g.max().item()))
        numel = int(g.numel())
        grad_numel += numel

        for group_name, prefixes in groups.items():
            if group_name == "mini_vae_other":
                if name.startswith("mini_vae.") and not name.startswith("mini_vae.encoder.") and not name.startswith(
                    "mini_vae.decoder."
                ):
                    group_sums[group_name] += float((g * g).sum().item())
                    group_numel[group_name] += numel
                continue

            if any(name.startswith(prefix) for prefix in prefixes):
                group_sums[group_name] += float((g * g).sum().item())
                group_numel[group_name] += numel
                break

    grad_rms = math.sqrt(sum_sq / max(1, grad_numel))
    param_rms = math.sqrt(param_sum_sq / max(1, param_numel))
    payload: dict[str, float] = {
        "grad/global_norm": math.sqrt(sum_sq),
        "grad/rms": grad_rms,
        "grad/abs_mean": (sum_abs / max(1, grad_numel)),
        "grad/max_abs": max_abs,
        "grad/numel": float(grad_numel),
        "param/rms": param_rms,
        "grad_to_param_rms_ratio": grad_rms / max(1e-12, param_rms),
    }

    for group_name in groups:
        payload[f"grad/{group_name}_rms"] = (
            math.sqrt(group_sums[group_name] / max(1, group_numel[group_name])) if group_numel[group_name] > 0 else 0.0
        )

    return payload


def counter_top(counter: Counter[str], limit: int) -> list[list[Any]]:
    return [[name, int(count)] for name, count in counter.most_common(max(1, int(limit)))]


def build_chunk_snapshot(
    collector: CollectorService,
    dataset: SharedModelDataset | None,
    chunk_store: Any | None,
    chunk_preview_count: int,
) -> dict[str, Any]:
    preview_count = max(1, int(chunk_preview_count))
    now = time.time()
    payload: dict[str, Any] = {}

    collector_stats = collector.stats()
    payload["collector"] = {
        "mode": collector_stats.get("mode"),
        "streaming_mode": collector_stats.get("streaming_mode"),
        "cache_size": int(collector_stats.get("cache_size", 0)),
        "jobs_total": int(collector_stats.get("jobs_total", 0)),
        "items_emitted": int(collector_stats.get("items_emitted", 0)),
        "jobs_by_model_top": counter_top(Counter({k: int(v) for k, v in (collector_stats.get("jobs_by_model") or {}).items()}), 8),
        "scheduler": collector_stats.get("scheduler"),
        "sink": collector_stats.get("sink"),
        "async_process_alive": collector_stats.get("async_process_alive"),
        "async_events_dropped": collector_stats.get("async_events_dropped"),
        "async_recent_jobs": list(collector_stats.get("async_recent_jobs") or [])[:preview_count],
    }

    sink_stats = payload["collector"].get("sink")
    if isinstance(sink_stats, dict):
        ready_chunks = sink_stats.get("backend_ready_chunks")
        max_ready_chunks = sink_stats.get("backend_max_ready_chunks")
        if ready_chunks is not None and max_ready_chunks is not None:
            try:
                payload["collector"]["ready_fill_ratio"] = float(ready_chunks) / max(1.0, float(max_ready_chunks))
            except Exception:
                pass

        spool_pending = sink_stats.get("writer_spool_pending_chunks")
        spool_max = sink_stats.get("writer_spool_max_pending_chunks")
        if spool_pending is not None and spool_max is not None:
            try:
                payload["collector"]["spool_fill_ratio"] = float(spool_pending) / max(1.0, float(spool_max))
            except Exception:
                pass

    if dataset is not None:
        payload["dataset"] = dataset.debug_snapshot(preview=preview_count)

    if chunk_store is None:
        payload["chunk_store"] = {"mode": "none"}
        return payload

    ready_count = int(chunk_store.count_ready())
    refs = chunk_store.list_ready(limit=max(preview_count * 8, preview_count))
    refs = refs[:preview_count]
    preview: list[dict[str, Any]] = []
    for ref in refs:
        item: dict[str, Any] = {
            "chunk_id": ref.chunk_id,
            "created_at": float(ref.created_at),
            "age_s": max(0.0, now - float(ref.created_at)),
            "size_bytes": int(ref.size_bytes),
        }
        if isinstance(chunk_store, LocalDiskChunkStore):
            meta_payload = chunk_store.read_ready_chunk_meta(ref.chunk_id)
            if isinstance(meta_payload, dict):
                meta = meta_payload.get("meta")
                if isinstance(meta, dict):
                    item["meta"] = {
                        "num_samples": meta.get("num_samples"),
                        "window_id": meta.get("window_id"),
                        "window_slot": meta.get("window_slot"),
                        "sample_summary": meta.get("sample_summary"),
                    }
        preview.append(item)

    payload["chunk_store"] = {
        "mode": str(collector.streaming_mode),
        "ready_count": ready_count,
        "ready_preview": preview,
        "backend_snapshot": chunk_store.debug_snapshot(limit=preview_count)
        if hasattr(chunk_store, "debug_snapshot")
        else None,
    }
    return payload


# ---------------------------
# Data helpers
# ---------------------------
@contextlib.contextmanager
def data_pipeline(
    cfg: DictConfig,
    *,
    start_collector: bool = True,
    predownload_models: bool | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[tuple[SharedModelDataset, CollectorService]]:
    logger_local = logger or logging.getLogger("mini_vae_train")

    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)

    train_cfg = cfg.mini_train
    should_predownload = bool(train_cfg.get("predownload_models", False))
    if predownload_models is not None:
        should_predownload = bool(predownload_models)

    if start_collector and should_predownload:
        logger_local.info("Predownloading model artifacts before collector start")
        collector.predownload_models()

    if start_collector:
        collector.start()

    logger_local.info(
        "Data pipeline initialized: collector_mode=%s streaming_mode=%s cache_metric=%s",
        collector.collector_mode,
        collector.streaming_mode,
        dataset.cache_size(),
    )

    try:
        yield dataset, collector
    finally:
        dataset.close()
        collector.shutdown()
        logger_local.info("Mini-VAE training shutdown complete")


def next_valid_sample(
    dataset_iter: Iterator[SharedSample],
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
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

        sample_info = extract_sample_debug_info(sample)
        return x, W, sample_info


def broadcast_tensor_2d(
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


def fetch_batch(
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[SharedSample] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    if use_broadcast:
        x_cpu: torch.Tensor | None = None
        W_cpu: torch.Tensor | None = None
        sample_info: dict[str, Any] | None = None
        if rank == 0:
            if dataset_iter is None:
                raise RuntimeError("rank0 requires dataset iterator in broadcast mode")
            x_cpu, W_cpu, sample_info = next_valid_sample(
                dataset_iter=dataset_iter,
                max_x_rows=max_x_rows,
                logger=logger,
            )

        x = broadcast_tensor_2d(x_cpu, device=device, src=0)
        W = broadcast_tensor_2d(W_cpu, device=device, src=0)
        return x, W, sample_info if rank == 0 else None

    if dataset_iter is None:
        raise RuntimeError("dataset iterator is required for sharded mode")

    x_cpu, W_cpu, sample_info = next_valid_sample(
        dataset_iter=dataset_iter,
        max_x_rows=max_x_rows,
        logger=logger,
    )
    return (
        x_cpu.to(device=device, non_blocking=True),
        W_cpu.to(device=device, non_blocking=True),
        sample_info,
    )


def sample_patch_batch(
    x: torch.Tensor,
    W: torch.Tensor,
    patch_size: int,
    patches_per_sample: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample patch-local training batch from one layer sample.

    Inputs:
    - x: [n, d_in]
    - W: [d_in, d_out]

    Outputs:
    - X_full: [B_p, n, d_in]
    - X_patch: [B_p, n, p]
    - w_patch: [B_p, p]
    - patch_idx: [B_p, p]
    - out_idx: [B_p]
    """
    if x.ndim != 2 or W.ndim != 2:
        raise ValueError(f"Expected x=[n,d_in], W=[d_in,d_out], got x={tuple(x.shape)} W={tuple(W.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if patches_per_sample <= 0:
        raise ValueError(f"patches_per_sample must be > 0, got {patches_per_sample}")

    n, d_in = x.shape
    d_in_w, d_out = W.shape
    if d_in_w != d_in:
        raise ValueError(f"Shape mismatch: x={tuple(x.shape)} W={tuple(W.shape)}")
    if d_out <= 0:
        raise ValueError(f"d_out must be positive, got {d_out}")

    device = x.device
    batch_patches = int(patches_per_sample)
    num_patch_slots = max(1, (d_in + patch_size - 1) // patch_size)

    # Random output and patch index per patch-sample item.
    out_idx = torch.randint(0, d_out, (batch_patches,), device=device)
    patch_t = torch.randint(0, num_patch_slots, (batch_patches,), device=device)

    offsets = torch.arange(patch_size, device=device).unsqueeze(0)  # [1, p]
    patch_idx = patch_t.unsqueeze(1) * patch_size + offsets  # [B_p, p]
    patch_idx = patch_idx.clamp(max=max(0, d_in - 1))

    # X_full: [B_p, n, d_in]
    X_full = x.unsqueeze(0).expand(batch_patches, -1, -1).contiguous()

    # X_patch: [B_p, n, p]
    X_patch = X_full.gather(dim=2, index=patch_idx.unsqueeze(1).expand(-1, n, -1))

    # Gather target patch weights from random output columns.
    # W_col: [B_p, d_in]
    W_col = W.transpose(0, 1).index_select(0, out_idx).contiguous()
    # w_patch: [B_p, p]
    w_patch = W_col.gather(dim=1, index=patch_idx)

    return X_full, X_patch, w_patch, patch_idx, out_idx


# ---------------------------
# Runtime helpers
# ---------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_world_size(cfg: DictConfig) -> int:
    train_cfg = cfg.mini_train
    distributed_mode = str(train_cfg.get("distributed", "auto")).strip().lower()
    num_gpus_req = int(train_cfg.get("num_gpus", 0))

    if distributed_mode in {"false", "0", "off", "no"}:
        return 1

    if not torch.cuda.is_available():
        return 1

    num_available = int(torch.cuda.device_count())
    if num_available <= 1:
        return 1

    if num_gpus_req > 0:
        num_available = min(num_available, num_gpus_req)

    if distributed_mode in {"true", "1", "on", "yes"} and num_available < 2:
        raise ValueError("mini_train.distributed=true requires at least 2 visible CUDA devices")

    return max(1, num_available)


def resolve_backend(cfg: DictConfig, device: torch.device) -> str:
    raw = str(cfg.mini_train.get("backend", "auto")).lower()
    if raw != "auto":
        return raw
    if device.type == "cuda":
        return "nccl"
    return "gloo"


def resolve_device(cfg: DictConfig, rank: int, world_size: int) -> torch.device:
    if torch.cuda.is_available():
        if world_size > 1:
            dev = torch.device(f"cuda:{rank}")
        else:
            wanted = str(cfg.mini_train.get("device", "cuda:0"))
            if wanted.startswith("cuda"):
                dev = torch.device(wanted)
            else:
                dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
        return dev
    return torch.device("cpu")


def set_speed_optimizations(cfg: DictConfig, device: torch.device) -> None:
    train_cfg = cfg.mini_train
    tf32 = bool(train_cfg.get("tf32", True))
    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", True))

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark

        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def maybe_compile_model(model: torch.nn.Module, cfg: DictConfig, logger: logging.Logger) -> torch.nn.Module:
    if not bool(cfg.mini_train.get("compile", False)):
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable in this PyTorch version; continuing without compile")
        return model

    compile_mode = str(cfg.mini_train.get("compile_mode", "max-autotune"))
    logger.info("Compiling mini-VAE model with torch.compile(mode=%s)", compile_mode)
    return torch.compile(model, mode=compile_mode, dynamic=True)


def build_optimizer(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    train_cfg = cfg.mini_train
    lr = float(train_cfg.get("lr", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    betas_cfg = train_cfg.get("betas", [0.9, 0.95])
    beta1 = float(betas_cfg[0])
    beta2 = float(betas_cfg[1])
    eps = float(train_cfg.get("eps", 1e-8))

    kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (beta1, beta2),
        "eps": eps,
    }

    params = inspect.signature(torch.optim.AdamW).parameters
    if "fused" in params and device.type == "cuda":
        kwargs["fused"] = True
    if "foreach" in params:
        kwargs["foreach"] = True

    return torch.optim.AdamW(model.parameters(), **kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LambdaLR:
    train_cfg = cfg.mini_train
    max_steps = max(1, int(train_cfg.get("max_steps", 1000)))
    warmup_steps = max(0, int(train_cfg.get("warmup_steps", 100)))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        progress = (step_idx - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_amp(cfg: DictConfig, device: torch.device) -> tuple[bool, torch.dtype | None]:
    mode = str(cfg.mini_train.get("amp", "auto")).lower()
    if device.type != "cuda" or mode in {"off", "false", "0", "none"}:
        return False, None

    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        return True, torch.float16

    if mode in {"bf16", "bfloat16"}:
        return True, torch.bfloat16

    if mode in {"fp16", "float16", "half"}:
        return True, torch.float16

    raise ValueError(f"Unsupported mini_train.amp={mode}")


def autocast_context(enabled: bool, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    if not enabled or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
) -> None:
    checkpoint_dir = Path(str(cfg.mini_train.get("checkpoint_dir", "./checkpoints/mini_patch_vae")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model
    if not isinstance(model_to_save, MiniPatchTrainingModel):
        raise TypeError(f"Expected MiniPatchTrainingModel, got {type(model_to_save)}")

    payload = {
        "step": step_idx,
        "model_state": model_to_save.state_dict(),
        "mini_vae_state": model_to_save.mini_vae.state_dict(),
        "mini_encoder_state": model_to_save.mini_vae.encoder.state_dict(),
        "distribution_encoder_state": model_to_save.distribution_encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    step_path = checkpoint_dir / f"step_{step_idx:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    encoder_latest_path = checkpoint_dir / "mini_encoder_latest.pt"

    torch.save(payload, step_path)
    torch.save(payload, latest_path)

    # Convenience checkpoint for BigWeightVAE mini encoder loading.
    torch.save(
        {
            "step": step_idx,
            "model_state": model_to_save.mini_vae.encoder.state_dict(),
            "config": payload["config"],
        },
        encoder_latest_path,
    )
    logger.info("Checkpoint saved: %s", step_path)


# ---------------------------
# Training loop
# ---------------------------
def run_worker(rank: int, world_size: int, cfg_dict: dict[str, Any], master_addr: str, master_port: int) -> None:
    cfg = OmegaConf.create(cfg_dict)
    setup_logging(cfg, rank=rank)
    logger = get_logger("mini_vae_train", rank=rank)
    if rank == 0:
        logger.info("Starting MiniPatchVAE training entrypoint")
        logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = resolve_device(cfg, rank=rank, world_size=world_size)
    backend = resolve_backend(cfg, device=device)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    seed = int(cfg.data.get("seed", 42)) + rank
    seed_everything(seed)
    set_speed_optimizations(cfg, device=device)

    is_distributed = world_size > 1
    streaming_mode = str(cfg.streaming.get("mode", "none")).lower()
    dataset_sharding = bool(cfg.mini_train.get("use_dataset_sharding", True)) and is_distributed and streaming_mode != "none"
    use_broadcast = is_distributed and not dataset_sharding

    if dataset_sharding:
        cfg.streaming.distributed.enabled = True
        cfg.streaming.distributed.rank_env = "RANK"
        cfg.streaming.distributed.world_size_env = "WORLD_SIZE"
        cfg.streaming.distributed.shard_by = str(cfg.streaming.distributed.get("shard_by", "chunk"))
        base_cache_dir = str(cfg.streaming.consumer.get("local_cache_dir", "./data/streaming/cache/consumer"))
        cfg.streaming.consumer.local_cache_dir = str(Path(base_cache_dir) / f"rank_{rank}")

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

    try:
        collector: CollectorService | None = None
        dataset: SharedModelDataset | None = None
        dataset_iter: Iterator[SharedSample] | None = None

        with contextlib.ExitStack() as stack:
            if rank == 0:
                dataset, collector = stack.enter_context(data_pipeline(cfg, logger=logger))
                dataset_iter = iter(dataset)
            elif dataset_sharding:
                dataset, collector = stack.enter_context(
                    data_pipeline(
                        cfg,
                        start_collector=False,
                        predownload_models=False,
                        logger=logger,
                    )
                )
                dataset_iter = iter(dataset)

            distribution_cfg = build_distribution_config(cfg)
            mini_cfg = build_mini_vae_config(cfg)
            model = MiniPatchTrainingModel(distribution_cfg=distribution_cfg, mini_cfg=mini_cfg).to(device)
            model = maybe_compile_model(model, cfg=cfg, logger=logger)

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

            optimizer = build_optimizer(model=model, cfg=cfg, device=device)
            scheduler = build_scheduler(optimizer=optimizer, cfg=cfg)

            amp_enabled, amp_dtype = resolve_amp(cfg=cfg, device=device)
            scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

            telemetry_cfg = cfg.mini_train.get("telemetry", {})
            local_telemetry_cfg = telemetry_cfg.get("local", {})
            local_status_every_steps = max(1, int(local_telemetry_cfg.get("status_every_steps", 10)))
            local_status_every_seconds = max(1.0, float(local_telemetry_cfg.get("status_every_seconds", 30.0)))
            status_window_steps = max(10, int(local_telemetry_cfg.get("status_window_steps", 100)))
            chunk_preview_count = max(1, int(local_telemetry_cfg.get("chunk_preview_count", 5)))
            recent_job_preview_count = max(1, int(local_telemetry_cfg.get("recent_job_preview_count", 10)))
            status_writer = JsonlStatusWriter(
                path=str(local_telemetry_cfg.get("jsonl_path", "./checkpoints/mini_patch_vae/runtime_status.jsonl")),
                enabled=bool(local_telemetry_cfg.get("save_jsonl", True)) and rank == 0,
            )
            comet_tracker = CometTracker(cfg=cfg, logger=logger, rank=rank)

            chunk_store = None
            if rank == 0 and collector is not None and str(collector.streaming_mode).lower() != "none":
                try:
                    chunk_store = build_chunk_store(resolve_streaming_cfg(OmegaConf.to_container(cfg.streaming, resolve=True)))
                except Exception as exc:
                    logger.warning("Failed to initialize chunk store debug snapshot: %s", exc)
                    chunk_store = None

            max_steps = max(1, int(cfg.mini_train.get("max_steps", 1000)))
            grad_accum_steps = max(1, int(cfg.mini_train.get("grad_accum_steps", 1)))
            kl_beta = float(cfg.mini_train.get("kl_beta", 1e-3))
            loss_cfg = cfg.mini_train.get("loss", {})
            alpha = float(loss_cfg.get("alpha", 0.0))
            beta = float(loss_cfg.get("beta", 0.5))
            contrastive_temperature = float(loss_cfg.get("contrastive_temperature", 0.07))
            contrastive_permute_inputs = bool(loss_cfg.get("contrastive_permute_inputs", True))
            contrastive_sign_flip_inputs = bool(loss_cfg.get("contrastive_sign_flip_inputs", True))
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"mini_train.loss.alpha must be in [0,1], got {alpha}")
            if not (0.0 <= beta <= 1.0):
                raise ValueError(f"mini_train.loss.beta must be in [0,1], got {beta}")
            if contrastive_temperature <= 0.0:
                raise ValueError(
                    f"mini_train.loss.contrastive_temperature must be > 0, got {contrastive_temperature}"
                )

            grad_clip_norm = float(cfg.mini_train.get("grad_clip_norm", 1.0))
            max_x_rows = int(cfg.mini_train.get("max_x_rows", 0))
            patches_per_sample = int(cfg.mini_train.get("patches_per_sample", 16))
            patch_size = int(cfg.mini_model.get("patch_size", 64))
            log_every = max(1, int(cfg.mini_train.get("log_every", 10)))
            checkpoint_every = max(1, int(cfg.mini_train.get("checkpoint_every", 200)))

            loss_window = 0.0
            structural_window = 0.0
            behavioral_window = 0.0
            contrastive_window = 0.0
            recon_mix_window = 0.0
            kl_window = 0.0
            window_steps = 0
            t0 = time.time()
            last_status_log_time = time.time()
            collector_prev_jobs_total = 0
            collector_prev_items_emitted = 0
            collector_prev_status_ts = time.time()
            step_fetch_ms_window = 0.0
            step_patch_ms_window = 0.0
            step_forward_ms_window = 0.0
            step_backward_ms_window = 0.0
            step_opt_ms_window = 0.0

            sample_window: deque[dict[str, Any]] = deque(maxlen=status_window_steps)
            patch_window: deque[dict[str, float]] = deque(maxlen=status_window_steps)

            for step_idx in range(max_steps):
                global_step = step_idx + 1
                model.train()
                optimizer.zero_grad(set_to_none=True)

                if rank == 0 and collector is not None and dataset is not None and not collector.is_async_mode:
                    dataset.maybe_collect(step_idx)

                loss_acc = 0.0
                structural_acc = 0.0
                behavioral_acc = 0.0
                contrastive_acc = 0.0
                recon_mix_acc = 0.0
                kl_acc = 0.0
                step_is_finite = True
                fetch_ms_step = 0.0
                patch_ms_step = 0.0
                forward_ms_step = 0.0
                backward_ms_step = 0.0
                sample_info_latest: dict[str, Any] | None = None
                patch_idx_min = float("nan")
                patch_idx_max = float("nan")
                patch_unique_ratio = float("nan")
                out_idx_unique_ratio = float("nan")
                patch_idx_head: list[int] = []
                out_idx_head: list[int] = []

                for micro_idx in range(grad_accum_steps):
                    sync_grad = micro_idx == grad_accum_steps - 1

                    t_fetch = time.perf_counter()
                    x, W, sample_info = fetch_batch(
                        rank=rank,
                        device=device,
                        dataset_iter=dataset_iter,
                        use_broadcast=use_broadcast,
                        max_x_rows=max_x_rows,
                        logger=logger,
                    )
                    fetch_ms_step += (time.perf_counter() - t_fetch) * 1000.0
                    if rank == 0 and sample_info is not None:
                        sample_info_latest = sample_info
                        sample_window.append(sample_info)

                    t_patch = time.perf_counter()
                    X_full, X_patch, w_patch, patch_idx, out_idx = sample_patch_batch(
                        x=x,
                        W=W,
                        patch_size=patch_size,
                        patches_per_sample=patches_per_sample,
                    )
                    patch_ms_step += (time.perf_counter() - t_patch) * 1000.0
                    if rank == 0:
                        patch_idx_min = float(patch_idx.min().item())
                        patch_idx_max = float(patch_idx.max().item())
                        patch_unique_ratio = float(patch_idx.unique().numel()) / float(max(1, patch_idx.numel()))
                        out_idx_unique_ratio = float(out_idx.unique().numel()) / float(max(1, out_idx.numel()))
                        patch_idx_head = [int(v) for v in patch_idx[0, : min(8, patch_idx.shape[1])].tolist()]
                        out_idx_head = [int(v) for v in out_idx[:8].tolist()]
                        patch_window.append(
                            {
                                "patch_unique_ratio": patch_unique_ratio,
                                "out_idx_unique_ratio": out_idx_unique_ratio,
                                "patch_idx_min": patch_idx_min,
                                "patch_idx_max": patch_idx_max,
                            }
                        )

                    no_sync_ctx = contextlib.nullcontext()
                    if is_distributed and not sync_grad:
                        no_sync_ctx = model.no_sync()  # type: ignore[union-attr]

                    with no_sync_ctx:
                        t_forward = time.perf_counter()
                        with autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                            total_loss, structural_loss, behavioral_loss, contrastive_loss, recon_mix_loss, kl_loss = model(
                                X_full=X_full,
                                X_patch=X_patch,
                                w_patch=w_patch,
                                patch_idx=patch_idx,
                                kl_beta=kl_beta,
                                alpha=alpha,
                                beta=beta,
                                contrastive_temperature=contrastive_temperature,
                                contrastive_permute_inputs=contrastive_permute_inputs,
                                contrastive_sign_flip_inputs=contrastive_sign_flip_inputs,
                                W_full=W,
                                out_idx=out_idx,
                            )
                            loss_for_backward = total_loss / grad_accum_steps
                        forward_ms_step += (time.perf_counter() - t_forward) * 1000.0

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
                            t_backward = time.perf_counter()
                            if scaler.is_enabled():
                                scaler.scale(loss_for_backward).backward()
                            else:
                                loss_for_backward.backward()
                            backward_ms_step += (time.perf_counter() - t_backward) * 1000.0

                    loss_acc += float(total_loss.detach().item())
                    structural_acc += float(structural_loss.detach().item())
                    behavioral_acc += float(behavioral_loss.detach().item())
                    contrastive_acc += float(contrastive_loss.detach().item())
                    recon_mix_acc += float(recon_mix_loss.detach().item())
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

                grad_stats = compute_grad_stats(model)
                grad_global_before_clip = float(grad_stats.get("grad/global_norm", 0.0))
                grad_clip_coef = 1.0
                if grad_clip_norm > 0.0:
                    clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    grad_global_before_clip = float(clip_return)
                    grad_clip_coef = min(1.0, float(grad_clip_norm) / max(1e-12, grad_global_before_clip))
                    grad_stats = compute_grad_stats(model)
                grad_stats["grad/global_norm_before_clip"] = grad_global_before_clip
                grad_stats["grad/clip_coef"] = grad_clip_coef

                t_opt = time.perf_counter()
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                step_opt_ms = (time.perf_counter() - t_opt) * 1000.0

                scheduler.step()

                step_loss = loss_acc / grad_accum_steps
                step_structural = structural_acc / grad_accum_steps
                step_behavioral = behavioral_acc / grad_accum_steps
                step_contrastive = contrastive_acc / grad_accum_steps
                step_recon_mix = recon_mix_acc / grad_accum_steps
                step_kl = kl_acc / grad_accum_steps

                stats = torch.tensor(
                    [
                        step_loss,
                        step_structural,
                        step_behavioral,
                        step_contrastive,
                        step_recon_mix,
                        step_kl,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                if is_distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    stats /= float(world_size)

                loss_window += float(stats[0].item())
                structural_window += float(stats[1].item())
                behavioral_window += float(stats[2].item())
                contrastive_window += float(stats[3].item())
                recon_mix_window += float(stats[4].item())
                kl_window += float(stats[5].item())
                window_steps += 1
                step_fetch_ms_window += fetch_ms_step
                step_patch_ms_window += patch_ms_step
                step_forward_ms_window += forward_ms_step
                step_backward_ms_window += backward_ms_step
                step_opt_ms_window += step_opt_ms

                if rank == 0 and global_step % log_every == 0:
                    dt = max(1e-6, time.time() - t0)
                    avg_loss = loss_window / max(1, window_steps)
                    avg_structural = structural_window / max(1, window_steps)
                    avg_behavioral = behavioral_window / max(1, window_steps)
                    avg_contrastive = contrastive_window / max(1, window_steps)
                    avg_recon_mix = recon_mix_window / max(1, window_steps)
                    avg_kl = kl_window / max(1, window_steps)
                    lr = float(optimizer.param_groups[0]["lr"])
                    speed = window_steps / dt

                    cache_metric = dataset.cache_size() if dataset is not None else 0
                    avg_fetch_ms = step_fetch_ms_window / max(1, window_steps)
                    avg_patch_ms = step_patch_ms_window / max(1, window_steps)
                    avg_forward_ms = step_forward_ms_window / max(1, window_steps)
                    avg_backward_ms = step_backward_ms_window / max(1, window_steps)
                    avg_opt_ms = step_opt_ms_window / max(1, window_steps)

                    model_counter: Counter[str] = Counter()
                    layer_counter: Counter[str] = Counter()
                    dataset_counter: Counter[str] = Counter()
                    for item in sample_window:
                        model_name = item.get("model_name")
                        layer_name = item.get("layer_name")
                        if model_name is not None:
                            model_counter[str(model_name)] += 1
                        if layer_name is not None:
                            layer_counter[str(layer_name)] += 1
                        for ds_name in item.get("datasets", []):
                            dataset_counter[str(ds_name)] += 1

                    patch_avg_unique_ratio = (
                        float(sum(x["patch_unique_ratio"] for x in patch_window) / max(1, len(patch_window)))
                        if patch_window
                        else 0.0
                    )
                    out_avg_unique_ratio = (
                        float(sum(x["out_idx_unique_ratio"] for x in patch_window) / max(1, len(patch_window)))
                        if patch_window
                        else 0.0
                    )

                    logger.info(
                        "step=%s/%s loss=%.6f str=%.6f beh=%.6f con=%.6f recon_mix=%.6f kl=%.6f "
                        "alpha=%.3f beta=%.3f lr=%.6e steps/s=%.2f patches=%s cache=%s "
                        "t_fetch=%.2fms t_patch=%.2fms t_fwd=%.2fms t_bwd=%.2fms t_opt=%.2fms "
                        "grad_norm=%.4f grad_rms=%.6f clip=%.3f",
                        global_step,
                        max_steps,
                        avg_loss,
                        avg_structural,
                        avg_behavioral,
                        avg_contrastive,
                        avg_recon_mix,
                        avg_kl,
                        alpha,
                        beta,
                        lr,
                        speed,
                        patches_per_sample,
                        cache_metric,
                        avg_fetch_ms,
                        avg_patch_ms,
                        avg_forward_ms,
                        avg_backward_ms,
                        avg_opt_ms,
                        float(grad_stats.get("grad/global_norm", 0.0)),
                        float(grad_stats.get("grad/rms", 0.0)),
                        float(grad_stats.get("grad/clip_coef", 1.0)),
                    )
                    logger.info(
                        "sample_mix window=%s models=%s datasets=%s layers=%s",
                        len(sample_window),
                        counter_top(model_counter, 5),
                        counter_top(dataset_counter, 5),
                        counter_top(layer_counter, 5),
                    )
                    logger.info(
                        "patch_debug patch_idx_head=%s out_idx_head=%s patch_unique=%.4f out_unique=%.4f",
                        patch_idx_head,
                        out_idx_head,
                        patch_avg_unique_ratio,
                        out_avg_unique_ratio,
                    )

                    collector_snapshot: dict[str, Any] | None = None
                    now = time.time()
                    should_status_log = (
                        global_step % local_status_every_steps == 0
                        or (now - last_status_log_time) >= local_status_every_seconds
                    )
                    if should_status_log and collector is not None:
                        collector_snapshot = build_chunk_snapshot(
                            collector=collector,
                            dataset=dataset,
                            chunk_store=chunk_store,
                            chunk_preview_count=chunk_preview_count,
                        )
                        collector_payload = collector_snapshot.get("collector", {})
                        if isinstance(collector_payload, dict):
                            jobs_total = int(collector_payload.get("jobs_total", 0))
                            items_total = int(collector_payload.get("items_emitted", 0))
                            elapsed_status = max(1e-6, now - collector_prev_status_ts)
                            collector_payload["jobs_per_sec"] = float(
                                (jobs_total - collector_prev_jobs_total) / elapsed_status
                            )
                            collector_payload["samples_per_sec"] = float(
                                (items_total - collector_prev_items_emitted) / elapsed_status
                            )
                            collector_payload["avg_samples_per_job"] = float(items_total / max(1, jobs_total))
                            collector_prev_jobs_total = jobs_total
                            collector_prev_items_emitted = items_total
                            collector_prev_status_ts = now

                        async_recent_jobs = collector_snapshot.get("collector", {}).get("async_recent_jobs", [])
                        if isinstance(async_recent_jobs, list):
                            collector_snapshot["collector"]["async_recent_jobs"] = async_recent_jobs[:recent_job_preview_count]

                        logger.info("collector_status step=%s snapshot=%s", global_step, json.dumps(collector_snapshot, ensure_ascii=False))
                        status_writer.write(
                            {
                                "step": int(global_step),
                                "timestamp": float(now),
                                "collector_status": collector_snapshot,
                            }
                        )
                        if comet_tracker is not None and comet_tracker.enabled:
                            comet_tracker.log_text(
                                text=f"collector_status_step_{global_step}",
                                metadata=collector_snapshot,
                            )
                        last_status_log_time = now

                    comet_metrics: dict[str, float] = {
                        "train/loss": float(avg_loss),
                        "train/structural": float(avg_structural),
                        "train/behavioral": float(avg_behavioral),
                        "train/contrastive": float(avg_contrastive),
                        "train/recon_mix": float(avg_recon_mix),
                        "train/recon": float(avg_recon_mix),
                        "train/kl": float(avg_kl),
                        "train/loss_alpha": float(alpha),
                        "train/loss_beta": float(beta),
                        "train/lr": float(lr),
                        "train/steps_per_sec": float(speed),
                        "timing/fetch_ms": float(avg_fetch_ms),
                        "timing/patch_ms": float(avg_patch_ms),
                        "timing/forward_ms": float(avg_forward_ms),
                        "timing/backward_ms": float(avg_backward_ms),
                        "timing/optimizer_ms": float(avg_opt_ms),
                        "data/cache_metric": float(cache_metric),
                        "data/patch_unique_ratio": float(patch_avg_unique_ratio),
                        "data/out_idx_unique_ratio": float(out_avg_unique_ratio),
                        "data/patch_idx_min": float(patch_idx_min) if not math.isnan(patch_idx_min) else 0.0,
                        "data/patch_idx_max": float(patch_idx_max) if not math.isnan(patch_idx_max) else 0.0,
                        "grad/global_norm_before_clip": float(grad_stats.get("grad/global_norm_before_clip", 0.0)),
                        "grad/global_norm": float(grad_stats.get("grad/global_norm", 0.0)),
                        "grad/rms": float(grad_stats.get("grad/rms", 0.0)),
                        "grad/abs_mean": float(grad_stats.get("grad/abs_mean", 0.0)),
                        "grad/max_abs": float(grad_stats.get("grad/max_abs", 0.0)),
                        "grad/clip_coef": float(grad_stats.get("grad/clip_coef", 1.0)),
                        "grad/distribution_encoder_rms": float(grad_stats.get("grad/distribution_encoder_rms", 0.0)),
                        "grad/mini_encoder_rms": float(grad_stats.get("grad/mini_encoder_rms", 0.0)),
                        "grad/mini_decoder_rms": float(grad_stats.get("grad/mini_decoder_rms", 0.0)),
                        "grad/mini_vae_other_rms": float(grad_stats.get("grad/mini_vae_other_rms", 0.0)),
                        "param/rms": float(grad_stats.get("param/rms", 0.0)),
                        "grad_to_param_rms_ratio": float(grad_stats.get("grad_to_param_rms_ratio", 0.0)),
                    }
                    if sample_info_latest is not None:
                        comet_metrics["data/x_rows"] = float(sample_info_latest.get("x_rows") or 0)
                        comet_metrics["data/d_in"] = float(sample_info_latest.get("d_in") or 0)
                        comet_metrics["data/d_out"] = float(sample_info_latest.get("d_out") or 0)

                    if device.type == "cuda":
                        comet_metrics["gpu/memory_allocated_mb"] = float(
                            torch.cuda.memory_allocated(device=device) / (1024.0 * 1024.0)
                        )
                        comet_metrics["gpu/memory_reserved_mb"] = float(
                            torch.cuda.memory_reserved(device=device) / (1024.0 * 1024.0)
                        )
                        comet_metrics["gpu/max_memory_allocated_mb"] = float(
                            torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0)
                        )

                    if comet_tracker is not None and comet_tracker.enabled:
                        comet_tracker.log_metrics(comet_metrics, step=global_step)

                    status_writer.write(
                        {
                            "step": int(global_step),
                            "timestamp": float(time.time()),
                            "metrics": comet_metrics,
                            "patch_debug": {
                                "patch_idx_head": patch_idx_head,
                                "out_idx_head": out_idx_head,
                            },
                            "sample_mix": {
                                "models_top": counter_top(model_counter, 10),
                                "datasets_top": counter_top(dataset_counter, 10),
                                "layers_top": counter_top(layer_counter, 10),
                            },
                        }
                    )

                    loss_window = 0.0
                    structural_window = 0.0
                    behavioral_window = 0.0
                    contrastive_window = 0.0
                    recon_mix_window = 0.0
                    kl_window = 0.0
                    window_steps = 0
                    step_fetch_ms_window = 0.0
                    step_patch_ms_window = 0.0
                    step_forward_ms_window = 0.0
                    step_backward_ms_window = 0.0
                    step_opt_ms_window = 0.0
                    t0 = time.time()

                if rank == 0 and (global_step % checkpoint_every == 0 or global_step == max_steps):
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        cfg=cfg,
                        step_idx=global_step,
                        logger=logger,
                    )

            if rank == 0:
                logger.info("Mini-VAE training completed successfully: steps=%s", max_steps)

    finally:
        if comet_tracker is not None:
            comet_tracker.end()

        if is_distributed:
            try:
                dist.barrier()
            except Exception:
                pass

        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def spawn_entry(rank: int, world_size: int, cfg_dict: dict[str, Any], master_addr: str, master_port: int) -> None:
    run_worker(rank=rank, world_size=world_size, cfg_dict=cfg_dict, master_addr=master_addr, master_port=master_port)


@hydra.main(version_base=None, config_path="../conf", config_name="mini_vae_train")
def main(cfg: DictConfig) -> None:
    world_size = resolve_world_size(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    if world_size <= 1:
        run_worker(rank=0, world_size=1, cfg_dict=cfg_dict, master_addr="127.0.0.1", master_port=0)
        return

    master_addr = str(cfg.mini_train.get("master_addr", "127.0.0.1"))
    requested_port = int(cfg.mini_train.get("master_port", 0))
    master_port = requested_port if requested_port > 0 else find_free_port()

    mp.spawn(
        spawn_entry,
        args=(world_size, cfg_dict, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
