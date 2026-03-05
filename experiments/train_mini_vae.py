from __future__ import annotations

import contextlib
import faulthandler
import inspect
import json
import logging
import math
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset
from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
from dataset.shared.streaming.factory import build_chunk_store, resolve_streaming_cfg
from dataset.shared.types import SharedSample
from training.optim import build_cosine_scheduler
from training.mini_patch_training_model import MiniPatchTrainingModel, build_distribution_config, build_mini_vae_config
from training.runtime import (
    autocast_context as runtime_autocast_context,
    configure_per_run_artifacts as runtime_configure_per_run_artifacts,
    find_free_port as runtime_find_free_port,
    get_rank_logger,
    maybe_compile_model as runtime_maybe_compile_model,
    resolve_amp as runtime_resolve_amp,
    resolve_backend as runtime_resolve_backend,
    resolve_device as runtime_resolve_device,
    resolve_world_size as runtime_resolve_world_size,
    seed_everything as runtime_seed_everything,
    set_speed_optimizations as runtime_set_speed_optimizations,
)

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass


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
    return get_rank_logger(name, rank)


def promote_run_profile_to_root(cfg: DictConfig) -> None:
    """
    Backward-compatible fallback for Hydra package/layout mismatches.

    Some compositions may place runtime sections under `run_profiles.*` instead
    of root keys expected by training entrypoints.
    """
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return

    expected_sections = (
        "data",
        "collector",
        "streaming",
        "mini_train",
        "mini_model",
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


def _strip_ddp_prefix(name: str) -> str:
    if name.startswith("module."):
        return name[len("module.") :]
    return name


def _resolve_group_lr(train_cfg: DictConfig, key: str, fallback_lr: float) -> float:
    value = train_cfg.get(key, None)
    if value is None:
        return float(fallback_lr)
    lr = float(value)
    if lr <= 0.0:
        raise ValueError(f"mini_train.{key} must be > 0 when set, got {lr}")
    return lr


def _get_optimizer_group_lr(optimizer: torch.optim.Optimizer, group_name: str, fallback_lr: float) -> float:
    for group in optimizer.param_groups:
        if str(group.get("group_name", "")) == group_name:
            return float(group["lr"])
    return float(fallback_lr)


def _set_optimizer_group_grads_to_none(optimizer: torch.optim.Optimizer, group_name: str) -> int:
    cleared = 0
    for group in optimizer.param_groups:
        if str(group.get("group_name", "")) != group_name:
            continue
        for param in group.get("params", []):
            if isinstance(param, torch.nn.Parameter) and param.grad is not None:
                param.grad = None
                cleared += 1
    return cleared


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
                    "mini_train.lr_encoder": (
                        float(cfg.mini_train.get("lr_encoder"))
                        if cfg.mini_train.get("lr_encoder", None) is not None
                        else float(cfg.mini_train.get("lr", 0.0))
                    ),
                    "mini_train.lr_decoder": (
                        float(cfg.mini_train.get("lr_decoder"))
                        if cfg.mini_train.get("lr_decoder", None) is not None
                        else float(cfg.mini_train.get("lr", 0.0))
                    ),
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
# Test-eval helpers
# ---------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_data_profile_path(profile_name: str) -> Path:
    raw = str(profile_name).strip()
    if not raw:
        raise ValueError("mini_train.test_eval.data_profile must be non-empty")

    root = _project_root()
    with_ext = raw if raw.endswith(".yaml") else f"{raw}.yaml"
    candidate_paths: list[Path] = []

    given = Path(raw)
    if given.is_absolute():
        candidate_paths.append(given if given.suffix else given.with_suffix(".yaml"))
    else:
        candidate_paths.append(root / "conf" / "data_collection_runtime" / "data_profiles" / with_ext)
        candidate_paths.append(root / "conf" / with_ext)
        candidate_paths.append(root / with_ext)

    for path in candidate_paths:
        if path.exists():
            return path

    options = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Could not find mini_train.test_eval.data_profile='{raw}'. Tried: {options}")


def build_test_eval_runtime_cfg(cfg: DictConfig, logger: logging.Logger) -> DictConfig | None:
    test_eval_cfg = cfg.mini_train.get("test_eval", {})
    if not isinstance(test_eval_cfg, (dict, DictConfig)):
        raise TypeError("mini_train.test_eval must be a mapping")
    if not bool(test_eval_cfg.get("enabled", False)):
        return None

    profile_name = str(test_eval_cfg.get("data_profile", "data_profile_for_hf_assets_test")).strip()
    profile_path = _resolve_data_profile_path(profile_name)
    profile_cfg = OmegaConf.load(profile_path)
    profile_payload = OmegaConf.to_container(profile_cfg, resolve=False)
    if not isinstance(profile_payload, dict):
        raise TypeError(f"Invalid test data profile payload in {profile_path}: {type(profile_payload)}")

    enabled_datasets = profile_payload.get("enabled_datasets", [])
    if not isinstance(enabled_datasets, list) or not enabled_datasets:
        raise ValueError(f"{profile_path} must define a non-empty enabled_datasets list")

    runtime_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(runtime_cfg):
        runtime_cfg.data.enabled_datasets = [str(name) for name in enabled_datasets]
        runtime_cfg.data.dataset_overrides = profile_payload.get("dataset_overrides", {}) or {}

        profile_data_path = profile_payload.get("path")
        if profile_data_path not in (None, ""):
            runtime_cfg.data.path = profile_data_path

        override_data_path = test_eval_cfg.get("data_path")
        if override_data_path not in (None, ""):
            runtime_cfg.data.path = str(override_data_path)

        collector_mode = test_eval_cfg.get("collector_mode")
        if collector_mode is None or str(collector_mode).strip().lower() in {"", "none", "null"}:
            runtime_cfg.collector.mode = "interleaved"
        else:
            runtime_cfg.collector.mode = str(collector_mode)

        collector_device_default: object = object()
        collector_device = test_eval_cfg.get("collector_device", collector_device_default)
        train_collector_device = cfg.collector.get("device")
        collector_device_text = (
            str(collector_device).strip().lower() if collector_device is not collector_device_default else ""
        )
        if collector_device is collector_device_default or collector_device is None or collector_device_text in {
            "",
            "none",
            "null",
        }:
            runtime_cfg.collector.device = train_collector_device
        else:
            runtime_cfg.collector.device = collector_device

        runtime_cfg.collector.max_loaded_models = 1
        runtime_cfg.collector.jobs_per_selected_model = max(1, int(test_eval_cfg.get("jobs_per_selected_model", 1)))
        runtime_cfg.collector.num_inflight_jobs = max(1, int(test_eval_cfg.get("num_inflight_jobs", 1)))

        in_memory_cfg = test_eval_cfg.get("in_memory_buffer", {})
        if not isinstance(in_memory_cfg, dict):
            in_memory_cfg = {}
        if runtime_cfg.collector.get("in_memory_buffer") is None:
            runtime_cfg.collector.in_memory_buffer = {}
        capacity_samples = max(1, int(in_memory_cfg.get("capacity_samples", 256)))
        fill_target_samples = max(1, int(in_memory_cfg.get("fill_target_samples", 64)))
        low_watermark_samples = max(1, int(in_memory_cfg.get("low_watermark_samples", 32)))
        fill_target_samples = min(fill_target_samples, capacity_samples)
        low_watermark_samples = min(low_watermark_samples, fill_target_samples)
        runtime_cfg.collector.in_memory_buffer.capacity_samples = capacity_samples
        runtime_cfg.collector.in_memory_buffer.fill_target_samples = fill_target_samples
        runtime_cfg.collector.in_memory_buffer.low_watermark_samples = low_watermark_samples

        if runtime_cfg.collector.get("interleaved_schedule") is None:
            runtime_cfg.collector.interleaved_schedule = {}
        collect_every_n_steps = max(1, int(test_eval_cfg.get("collect_every_n_steps", 1)))
        runtime_cfg.collector.interleaved_schedule.collect_every_n_train_steps = collect_every_n_steps
        # Keep both key variants in runtime config for compatibility across readers.
        runtime_cfg.collector.interleaved_schedule.collect_every_n_steps = collect_every_n_steps
        runtime_cfg.collector.interleaved_schedule.collector_jobs_per_cycle = max(
            1, int(test_eval_cfg.get("collector_jobs_per_cycle", 1))
        )

        streaming_mode = test_eval_cfg.get("streaming_mode")
        if streaming_mode is None or str(streaming_mode).strip().lower() in {"", "none", "null"}:
            runtime_cfg.streaming.mode = "none"
        else:
            runtime_cfg.streaming.mode = str(streaming_mode)

    logger.info(
        "Configured test-eval runtime: profile=%s datasets=%s collector_mode=%s collector_device=%s streaming_mode=%s",
        profile_path,
        runtime_cfg.data.enabled_datasets,
        runtime_cfg.collector.mode,
        runtime_cfg.collector.device,
        runtime_cfg.streaming.mode,
    )
    return runtime_cfg


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
    group_param_count = {name: 0 for name in groups}

    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    grad_numel = 0
    param_sum_sq = 0.0
    param_numel = 0

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
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
                    group_param_count[group_name] += 1
                    group_sums[group_name] += float((g * g).sum().item())
                    group_numel[group_name] += numel
                continue

            if any(name.startswith(prefix) for prefix in prefixes):
                group_param_count[group_name] += 1
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
        payload[f"grad/{group_name}_numel"] = float(group_numel[group_name])
        payload[f"grad/{group_name}_params_with_grad"] = float(group_param_count[group_name])

    return payload


def _layer_key_from_param_name(name: str) -> str:
    parts = name.split(".")
    if not parts:
        return name
    last = parts[-1]
    if last in {"weight", "bias"} or last.endswith("_weight") or last.endswith("_bias"):
        if len(parts) > 1:
            return ".".join(parts[:-1])
    return name


def collect_grad_rms_per_layer(
    model: nn.Module,
    *,
    include_prefixes: tuple[str, ...],
    weights_only: bool,
) -> dict[str, float]:
    sumsq_by_layer: dict[str, float] = {}
    count_by_layer: dict[str, int] = {}
    global_sumsq = 0.0
    global_count = 0

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
        if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
            continue

        grad = param.grad
        if grad is None:
            continue
        if weights_only and param.ndim < 2:
            continue

        g = grad.detach()
        if g.numel() == 0:
            continue

        sumsq = float((g * g).sum().item())
        count = int(g.numel())
        layer = _layer_key_from_param_name(name)
        sumsq_by_layer[layer] = sumsq_by_layer.get(layer, 0.0) + sumsq
        count_by_layer[layer] = count_by_layer.get(layer, 0) + count

        global_sumsq += sumsq
        global_count += count

    out: dict[str, float] = {}
    for layer, sumsq in sumsq_by_layer.items():
        count = max(1, count_by_layer[layer])
        out[layer] = math.sqrt(sumsq / float(count))

    if global_count > 0:
        out["__global__"] = math.sqrt(global_sumsq / float(global_count))
    return out


def _append_grad_layer_rms_csv(
    save_path: Path,
    *,
    step: int,
    layer_rms_pre_clip: dict[str, float],
    layer_rms_post_clip: dict[str, float],
    layer_param_rms: dict[str, float],
    layer_grad_to_param_ratio_pre_clip: dict[str, float],
    layer_grad_to_param_ratio_post_clip: dict[str, float],
    clip_coef: float,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        save_path.write_text(
            "step,layer,grad_rms_pre_clip,grad_rms_post_clip,param_rms,grad_to_param_ratio_pre_clip,"
            "grad_to_param_ratio_post_clip,clip_coef\n",
            encoding="utf-8",
        )

    layers = sorted(
        set(layer_rms_pre_clip.keys())
        | set(layer_rms_post_clip.keys())
        | set(layer_param_rms.keys())
        | set(layer_grad_to_param_ratio_pre_clip.keys())
        | set(layer_grad_to_param_ratio_post_clip.keys())
    )
    with save_path.open("a", encoding="utf-8") as handle:
        for layer in layers:
            pre_val = float(layer_rms_pre_clip.get(layer, 0.0))
            post_val = float(layer_rms_post_clip.get(layer, 0.0))
            param_val = float(layer_param_rms.get(layer, 0.0))
            ratio_pre_val = float(layer_grad_to_param_ratio_pre_clip.get(layer, 0.0))
            ratio_post_val = float(layer_grad_to_param_ratio_post_clip.get(layer, 0.0))
            handle.write(
                f"{int(step)},{layer},{pre_val:.12e},{post_val:.12e},{param_val:.12e},"
                f"{ratio_pre_val:.12e},{ratio_post_val:.12e},{float(clip_coef):.12e}\n"
            )


def collect_param_rms_per_layer(
    model: nn.Module,
    *,
    include_prefixes: tuple[str, ...],
    weights_only: bool,
) -> dict[str, float]:
    sumsq_by_layer: dict[str, float] = {}
    count_by_layer: dict[str, int] = {}
    global_sumsq = 0.0
    global_count = 0

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
        if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
            continue
        if weights_only and param.ndim < 2:
            continue

        p = param.detach()
        if p.numel() == 0:
            continue

        sumsq = float((p * p).sum().item())
        count = int(p.numel())
        layer = _layer_key_from_param_name(name)
        sumsq_by_layer[layer] = sumsq_by_layer.get(layer, 0.0) + sumsq
        count_by_layer[layer] = count_by_layer.get(layer, 0) + count

        global_sumsq += sumsq
        global_count += count

    out: dict[str, float] = {}
    for layer, sumsq in sumsq_by_layer.items():
        count = max(1, count_by_layer[layer])
        out[layer] = math.sqrt(sumsq / float(count))

    if global_count > 0:
        out["__global__"] = math.sqrt(global_sumsq / float(global_count))
    return out


def _save_grad_rms_layer_plot(
    history: dict[str, list[tuple[int, float]]],
    save_path: Path,
    *,
    topk_layers: int,
    log_scale: bool,
) -> bool:
    if not history:
        return False

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return False

    layers = [layer for layer in history.keys() if layer != "__global__"]
    layers.sort(
        key=lambda key: sum(val for _, val in history[key]) / max(1, len(history[key])),
        reverse=True,
    )
    layers = layers[: max(1, int(topk_layers))]

    plt.figure(figsize=(13, 6))
    if "__global__" in history:
        steps = [step for step, _ in history["__global__"]]
        values = [value for _, value in history["__global__"]]
        plt.plot(steps, values, label="__global__", linewidth=2.5, color="black")

    for layer in layers:
        steps = [step for step, _ in history[layer]]
        values = [value for _, value in history[layer]]
        plt.plot(steps, values, label=layer, linewidth=1.1, alpha=0.9)

    plt.title("MiniVAE Gradient RMS Per Layer (pre-clip)")
    plt.xlabel("Step")
    plt.ylabel("Gradient RMS")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.2)
    if layers or "__global__" in history:
        plt.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()
    return True


def _save_grad_rms_layer_heatmap(
    history: dict[str, list[tuple[int, float]]],
    save_path: Path,
    *,
    max_layers: int,
    log_scale: bool,
    min_value: float = 1e-20,
) -> bool:
    if not history:
        return False

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return False

    layers = [layer for layer in history.keys() if layer != "__global__"]
    if not layers:
        return False

    layers.sort(
        key=lambda key: sum(val for _, val in history[key]) / max(1, len(history[key])),
    )
    layers = layers[: max(1, int(max_layers))]

    step_values = sorted({int(step) for points in history.values() for step, _ in points})
    if not step_values:
        return False
    step_to_idx = {step: idx for idx, step in enumerate(step_values)}

    matrix: list[list[float]] = []
    for layer in layers:
        row = [float("nan")] * len(step_values)
        for step, value in history[layer]:
            idx = step_to_idx.get(int(step))
            if idx is None:
                continue
            safe_value = max(float(min_value), float(value))
            row[idx] = math.log10(safe_value) if log_scale else safe_value
        matrix.append(row)

    fig_h = max(4.0, 0.24 * len(layers) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("MiniVAE Gradient RMS Heatmap (pre-clip)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Step")

    ax.set_yticks(list(range(len(layers))))
    ax.set_yticklabels(layers, fontsize=7)

    xtick_count = min(12, len(step_values))
    if xtick_count >= 2:
        xtick_idx = sorted(
            set(int(round(i * (len(step_values) - 1) / float(xtick_count - 1))) for i in range(xtick_count))
        )
    else:
        xtick_idx = [0]
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([str(step_values[idx]) for idx in xtick_idx], rotation=45, ha="right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(grad RMS)" if log_scale else "grad RMS")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close(fig)
    return True


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
    def _prepare_sample(sample: SharedSample) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
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
            return None

        x_out = x.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
        W_out = W.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
        if max_x_rows > 0 and x_out.shape[0] > max_x_rows:
            keep = torch.randperm(x_out.shape[0])[:max_x_rows]
            x_out = x_out[keep]

        return x_out, W_out, extract_sample_debug_info(sample)

    attempts = 0
    while True:
        sample = next(dataset_iter)
        prepared = _prepare_sample(sample)
        if prepared is None:
            attempts += 1
            if attempts % 100 == 0:
                logger.warning("Skipping invalid sample repeatedly; attempts=%s", attempts)
            continue
        return prepared


def next_valid_sample_for_eval(
    dataset: SharedModelDataset,
    collector: CollectorService,
    max_x_rows: int,
    timeout_seconds: float,
    poll_sleep_seconds: float,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    deadline = time.time() + max(1.0, float(timeout_seconds))
    attempts = 0
    probe_step = 0

    while True:
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for test-eval sample")

        if not collector.is_async_mode:
            dataset.maybe_collect(probe_step)

        sample = dataset.try_next_sample()
        probe_step += 1
        if sample is None:
            time.sleep(max(0.001, float(poll_sleep_seconds)))
            continue

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
                logger.warning("Skipping invalid test sample repeatedly; attempts=%s", attempts)
            continue

        x_out = x.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
        W_out = W.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
        if max_x_rows > 0 and x_out.shape[0] > max_x_rows:
            keep = torch.randperm(x_out.shape[0])[:max_x_rows]
            x_out = x_out[keep]

        return x_out, W_out, extract_sample_debug_info(sample)


ABLATION_METRIC_KEYS: tuple[str, ...] = (
    "decoder_random_latent_structural",
    "decoder_random_latent_behavioral",
    "decoder_random_latent_recon_mix",
    "random_dist_structural",
    "random_dist_behavioral",
    "random_dist_recon_mix",
    "random_dist_kl",
    "random_dist_total",
)


def compute_ablation_metrics_for_batch(
    eval_model: nn.Module,
    *,
    X_full: torch.Tensor,
    X_patch: torch.Tensor,
    w_patch: torch.Tensor,
    patch_idx: torch.Tensor,
    structural_coef: float,
    behavioral_coef: float,
    kl_coef: float,
) -> dict[str, float]:
    ablation_fn = getattr(eval_model, "ablation_losses", None)
    if not callable(ablation_fn):
        raise TypeError(
            f"Expected model with callable ablation_losses(...), got {type(eval_model)}"
        )

    raw_payload = ablation_fn(
        X_full=X_full,
        X_patch=X_patch,
        w_patch=w_patch,
        patch_idx=patch_idx,
        structural_coef=structural_coef,
        behavioral_coef=behavioral_coef,
        kl_coef=kl_coef,
    )
    if not isinstance(raw_payload, dict):
        raise TypeError(f"ablation_losses must return dict, got {type(raw_payload)}")

    payload: dict[str, float] = {}
    for key in ABLATION_METRIC_KEYS:
        if key not in raw_payload:
            raise KeyError(f"Missing ablation metric '{key}' in ablation_losses output")
        value = raw_payload[key]
        if torch.is_tensor(value):
            payload[key] = float(value.detach().item())
        else:
            payload[key] = float(value)
    return payload


def run_test_eval(
    model: nn.Module,
    dataset: SharedModelDataset,
    collector: CollectorService,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    patch_size: int,
    patches_per_sample: int,
    structural_coef: float,
    behavioral_coef: float,
    contrastive_coef: float,
    kl_coef: float,
    contrastive_temperature: float,
    contrastive_permute_inputs: bool,
    contrastive_sign_flip_inputs: bool,
    num_batches: int,
    max_x_rows: int,
    timeout_seconds: float,
    poll_sleep_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    eval_model = model.module if isinstance(model, DDP) else model
    was_training = bool(eval_model.training)
    eval_model.eval()

    loss_sum = 0.0
    structural_sum = 0.0
    behavioral_sum = 0.0
    contrastive_sum = 0.0
    recon_mix_sum = 0.0
    kl_sum = 0.0
    x_rows_sum = 0.0
    ablation_sums: dict[str, float] = {key: 0.0 for key in ABLATION_METRIC_KEYS}

    model_counter: Counter[str] = Counter()
    dataset_counter: Counter[str] = Counter()
    layer_counter: Counter[str] = Counter()

    started = time.time()
    try:
        with torch.no_grad():
            for _ in range(max(1, int(num_batches))):
                x_cpu, W_cpu, sample_info = next_valid_sample_for_eval(
                    dataset=dataset,
                    collector=collector,
                    max_x_rows=max_x_rows,
                    timeout_seconds=timeout_seconds,
                    poll_sleep_seconds=poll_sleep_seconds,
                    logger=logger,
                )
                x = x_cpu.to(device=device, non_blocking=True)
                W = W_cpu.to(device=device, non_blocking=True)

                X_full, X_patch, w_patch, patch_idx, out_idx = sample_patch_batch(
                    x=x,
                    W=W,
                    patch_size=patch_size,
                    patches_per_sample=patches_per_sample,
                )
                with autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                    total_loss, structural_loss, behavioral_loss, contrastive_loss, recon_mix_loss, kl_loss = eval_model(
                        X_full=X_full,
                        X_patch=X_patch,
                        w_patch=w_patch,
                        patch_idx=patch_idx,
                        structural_coef=structural_coef,
                        behavioral_coef=behavioral_coef,
                        contrastive_coef=contrastive_coef,
                        kl_coef=kl_coef,
                        contrastive_temperature=contrastive_temperature,
                        contrastive_permute_inputs=contrastive_permute_inputs,
                        contrastive_sign_flip_inputs=contrastive_sign_flip_inputs,
                        W_full=W,
                        out_idx=out_idx,
                    )
                    ablation_metrics_batch = compute_ablation_metrics_for_batch(
                        eval_model=eval_model,
                        X_full=X_full,
                        X_patch=X_patch,
                        w_patch=w_patch,
                        patch_idx=patch_idx,
                        structural_coef=structural_coef,
                        behavioral_coef=behavioral_coef,
                        kl_coef=kl_coef,
                    )

                loss_sum += float(total_loss.detach().item())
                structural_sum += float(structural_loss.detach().item())
                behavioral_sum += float(behavioral_loss.detach().item())
                contrastive_sum += float(contrastive_loss.detach().item())
                recon_mix_sum += float(recon_mix_loss.detach().item())
                kl_sum += float(kl_loss.detach().item())
                x_rows_sum += float(sample_info.get("x_rows") or 0)
                for key in ABLATION_METRIC_KEYS:
                    ablation_sums[key] += float(ablation_metrics_batch[key])

                model_name = sample_info.get("model_name")
                layer_name = sample_info.get("layer_name")
                if model_name is not None:
                    model_counter[str(model_name)] += 1
                if layer_name is not None:
                    layer_counter[str(layer_name)] += 1
                for ds_name in sample_info.get("datasets", []):
                    dataset_counter[str(ds_name)] += 1
    finally:
        if was_training:
            eval_model.train()

    n = float(max(1, int(num_batches)))
    duration_s = max(1e-6, time.time() - started)
    return {
        "num_batches": int(num_batches),
        "duration_s": float(duration_s),
        "metrics": {
            "loss": loss_sum / n,
            "structural": structural_sum / n,
            "behavioral": behavioral_sum / n,
            "contrastive": contrastive_sum / n,
            "recon_mix": recon_mix_sum / n,
            "kl": kl_sum / n,
            "x_rows": x_rows_sum / n,
            "batches_per_sec": n / duration_s,
            "decoder_random_latent_structural": ablation_sums["decoder_random_latent_structural"] / n,
            "decoder_random_latent_behavioral": ablation_sums["decoder_random_latent_behavioral"] / n,
            "decoder_random_latent_recon_mix": ablation_sums["decoder_random_latent_recon_mix"] / n,
            "random_dist_structural": ablation_sums["random_dist_structural"] / n,
            "random_dist_behavioral": ablation_sums["random_dist_behavioral"] / n,
            "random_dist_recon_mix": ablation_sums["random_dist_recon_mix"] / n,
            "random_dist_kl": ablation_sums["random_dist_kl"] / n,
            "random_dist_total": ablation_sums["random_dist_total"] / n,
        },
        "sample_mix": {
            "models_top": counter_top(model_counter, 10),
            "datasets_top": counter_top(dataset_counter, 10),
            "layers_top": counter_top(layer_counter, 10),
        },
    }


def run_train_ablation_eval(
    model: nn.Module,
    dataset: SharedModelDataset,
    collector: CollectorService,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    patch_size: int,
    patches_per_sample: int,
    structural_coef: float,
    behavioral_coef: float,
    kl_coef: float,
    num_batches: int,
    max_x_rows: int,
    timeout_seconds: float,
    poll_sleep_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    eval_model = model.module if isinstance(model, DDP) else model
    was_training = bool(eval_model.training)
    eval_model.eval()

    ablation_sums: dict[str, float] = {key: 0.0 for key in ABLATION_METRIC_KEYS}
    x_rows_sum = 0.0

    model_counter: Counter[str] = Counter()
    dataset_counter: Counter[str] = Counter()
    layer_counter: Counter[str] = Counter()

    started = time.time()
    try:
        with torch.no_grad():
            for _ in range(max(1, int(num_batches))):
                x_cpu, W_cpu, sample_info = next_valid_sample_for_eval(
                    dataset=dataset,
                    collector=collector,
                    max_x_rows=max_x_rows,
                    timeout_seconds=timeout_seconds,
                    poll_sleep_seconds=poll_sleep_seconds,
                    logger=logger,
                )
                x = x_cpu.to(device=device, non_blocking=True)
                W = W_cpu.to(device=device, non_blocking=True)

                X_full, X_patch, w_patch, patch_idx, _ = sample_patch_batch(
                    x=x,
                    W=W,
                    patch_size=patch_size,
                    patches_per_sample=patches_per_sample,
                )
                with autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                    ablation_metrics_batch = compute_ablation_metrics_for_batch(
                        eval_model=eval_model,
                        X_full=X_full,
                        X_patch=X_patch,
                        w_patch=w_patch,
                        patch_idx=patch_idx,
                        structural_coef=structural_coef,
                        behavioral_coef=behavioral_coef,
                        kl_coef=kl_coef,
                    )

                for key in ABLATION_METRIC_KEYS:
                    ablation_sums[key] += float(ablation_metrics_batch[key])
                x_rows_sum += float(sample_info.get("x_rows") or 0)

                model_name = sample_info.get("model_name")
                layer_name = sample_info.get("layer_name")
                if model_name is not None:
                    model_counter[str(model_name)] += 1
                if layer_name is not None:
                    layer_counter[str(layer_name)] += 1
                for ds_name in sample_info.get("datasets", []):
                    dataset_counter[str(ds_name)] += 1
    finally:
        if was_training:
            eval_model.train()

    n = float(max(1, int(num_batches)))
    duration_s = max(1e-6, time.time() - started)
    metrics: dict[str, float] = {
        "x_rows": x_rows_sum / n,
        "batches_per_sec": n / duration_s,
    }
    for key in ABLATION_METRIC_KEYS:
        metrics[key] = ablation_sums[key] / n

    return {
        "num_batches": int(num_batches),
        "duration_s": float(duration_s),
        "metrics": metrics,
        "sample_mix": {
            "models_top": counter_top(model_counter, 10),
            "datasets_top": counter_top(dataset_counter, 10),
            "layers_top": counter_top(layer_counter, 10),
        },
    }


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


def sample_synthetic_layer(
    *,
    device: torch.device,
    n_rows: int,
    d_in: int,
    d_out: int,
    x_std: float,
    w_std: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    x = torch.randn((int(n_rows), int(d_in)), device=device, dtype=torch.float32) * float(x_std)
    W = torch.randn((int(d_in), int(d_out)), device=device, dtype=torch.float32) * float(w_std)
    sample_info = {
        "model_name": "synthetic_random_normal",
        "layer_name": "synthetic_patch_source",
        "datasets": ["synthetic_random_normal"],
        "x_rows": int(n_rows),
        "d_in": int(d_in),
        "d_out": int(d_out),
    }
    return x, W, sample_info


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
    runtime_seed_everything(seed)


def find_free_port() -> int:
    return runtime_find_free_port()


def resolve_world_size(cfg: DictConfig) -> int:
    return runtime_resolve_world_size(cfg, section="mini_train", error_prefix="mini_train")


def resolve_backend(cfg: DictConfig, device: torch.device) -> str:
    return runtime_resolve_backend(cfg, device, section="mini_train")


def resolve_device(cfg: DictConfig, rank: int, world_size: int) -> torch.device:
    return runtime_resolve_device(cfg, rank, world_size, section="mini_train")


def set_speed_optimizations(cfg: DictConfig, device: torch.device) -> None:
    runtime_set_speed_optimizations(cfg, device, section="mini_train")


def maybe_compile_model(model: torch.nn.Module, cfg: DictConfig, logger: logging.Logger) -> torch.nn.Module:
    return runtime_maybe_compile_model(
        model=model,
        cfg=cfg,
        logger=logger,
        section="mini_train",
        label="mini-VAE model",
    )


def build_optimizer(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    train_cfg = cfg["mini_train"]
    lr = float(train_cfg.get("lr", 2e-4))
    if lr <= 0.0:
        raise ValueError(f"mini_train.lr must be > 0, got {lr}")
    lr_encoder = _resolve_group_lr(train_cfg, "lr_encoder", lr)
    lr_decoder = _resolve_group_lr(train_cfg, "lr_decoder", lr)
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    betas_cfg = train_cfg.get("betas", [0.9, 0.95])
    beta1 = float(betas_cfg[0])
    beta2 = float(betas_cfg[1])
    eps = float(train_cfg.get("eps", 1e-8))

    named_params = list(model.named_parameters())
    default_params: list[torch.nn.Parameter] = []
    mini_encoder_params: list[torch.nn.Parameter] = []
    mini_decoder_params: list[torch.nn.Parameter] = []
    for raw_name, param in named_params:
        if not param.requires_grad:
            continue
        name = _strip_ddp_prefix(raw_name)
        if name.startswith("mini_vae.encoder."):
            mini_encoder_params.append(param)
        elif name.startswith("mini_vae.decoder."):
            mini_decoder_params.append(param)
        else:
            default_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if default_params:
        param_groups.append({"params": default_params, "lr": lr, "group_name": "default"})
    if mini_encoder_params:
        param_groups.append({"params": mini_encoder_params, "lr": lr_encoder, "group_name": "mini_encoder"})
    if mini_decoder_params:
        param_groups.append({"params": mini_decoder_params, "lr": lr_decoder, "group_name": "mini_decoder"})
    if not param_groups:
        raise ValueError("No trainable parameters were found for optimizer construction")
    # print('EWWW')
    # print(len(mini_encoder_params), lr_encoder, len(mini_decoder_params), lr_decoder)
    # 1/0
    kwargs: dict[str, Any] = {
        "weight_decay": weight_decay,
        "betas": (beta1, beta2),
        "eps": eps,
    }
    adamw_signature = inspect.signature(torch.optim.AdamW).parameters
    use_fused = "fused" in adamw_signature and device.type == "cuda"
    use_foreach = "foreach" in adamw_signature and not use_fused
    if "fused" in adamw_signature:
        kwargs["fused"] = use_fused
    if "foreach" in adamw_signature:
        kwargs["foreach"] = use_foreach

    return torch.optim.AdamW(param_groups, **kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LambdaLR:
    freeze_decoder_steps = max(0, int(cfg.mini_train.get("freeze_decoder_steps", 0)))
    # Keep decoder LR schedule at its initial warmup point while decoder grads are frozen.
    step_delay_by_group_name = {"mini_decoder": freeze_decoder_steps} if freeze_decoder_steps > 0 else None
    return build_cosine_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        section="mini_train",
        default_max_steps=1000,
        default_warmup_steps=100,
        default_min_lr_ratio=0.1,
        step_delay_by_group_name=step_delay_by_group_name,
    )


def resolve_amp(cfg: DictConfig, device: torch.device) -> tuple[bool, torch.dtype | None]:
    return runtime_resolve_amp(cfg, device, section="mini_train")


def autocast_context(enabled: bool, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    return runtime_autocast_context(enabled, dtype)


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
    promote_run_profile_to_root(cfg)
    setup_logging(cfg, rank=rank)
    logger = get_logger("mini_vae_train", rank=rank)
    is_distributed = world_size > 1
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

    streaming_mode = str(cfg.streaming.get("mode", "none")).lower()
    dataset_sharding = bool(cfg.mini_train.get("use_dataset_sharding", True)) and is_distributed and streaming_mode != "none"
    use_broadcast = is_distributed and not dataset_sharding
    synthetic_patch_cfg = cfg.mini_train.get("synthetic_patch_source", {})
    if synthetic_patch_cfg is None:
        synthetic_patch_cfg = {}
    if not isinstance(synthetic_patch_cfg, (dict, DictConfig)):
        raise TypeError("mini_train.synthetic_patch_source must be a mapping")
    synthetic_patch_enabled = bool(synthetic_patch_cfg.get("enabled", False))
    synthetic_n_rows = max(1, int(synthetic_patch_cfg.get("n_rows", 256)))
    synthetic_d_in = max(1, int(synthetic_patch_cfg.get("d_in", 1024)))
    synthetic_d_out = max(1, int(synthetic_patch_cfg.get("d_out", 1024)))
    synthetic_x_std = float(synthetic_patch_cfg.get("x_std", 1.0))
    synthetic_w_std = float(synthetic_patch_cfg.get("w_std", 1.0))
    if synthetic_x_std <= 0.0:
        raise ValueError(f"mini_train.synthetic_patch_source.x_std must be > 0, got {synthetic_x_std}")
    if synthetic_w_std <= 0.0:
        raise ValueError(f"mini_train.synthetic_patch_source.w_std must be > 0, got {synthetic_w_std}")
    if synthetic_patch_enabled:
        dataset_sharding = False
        use_broadcast = False

    if dataset_sharding:
        cfg.streaming.distributed.enabled = True
        cfg.streaming.distributed.rank_env = "RANK"
        cfg.streaming.distributed.world_size_env = "WORLD_SIZE"
        cfg.streaming.distributed.shard_by = str(cfg.streaming.distributed.get("shard_by", "chunk"))
        base_cache_dir = str(
            cfg.streaming.consumer.get(
                "cache_dir",
                cfg.streaming.consumer.get("local_cache_dir", "./data/streaming/cache/consumer"),
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
    if synthetic_patch_enabled:
        logger.info(
            "Synthetic patch source enabled: collectors disabled, x~N(0,%.4f), W~N(0,%.4f), n_rows=%s d_in=%s d_out=%s",
            synthetic_x_std,
            synthetic_w_std,
            synthetic_n_rows,
            synthetic_d_in,
            synthetic_d_out,
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
        test_eval_runtime_cfg: DictConfig | None = None
        test_eval_predownload = False

        with contextlib.ExitStack() as stack:
            if not synthetic_patch_enabled:
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

            if rank == 0 and not synthetic_patch_enabled:
                test_eval_runtime_cfg = build_test_eval_runtime_cfg(cfg=cfg, logger=logger)
                if test_eval_runtime_cfg is not None:
                    test_eval_cfg_local = cfg.mini_train.get("test_eval", {})
                    test_eval_predownload = bool(test_eval_cfg_local.get("predownload_models", False))

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
            comet_cfg = telemetry_cfg.get("comet", {})
            comet_weight_snapshot_enabled = bool(comet_cfg.get("log_weight_snapshots", True))
            comet_weight_snapshot_every_steps = max(1, int(comet_cfg.get("weight_snapshot_every_steps", 100)))
            comet_weight_snapshot_num_values = max(1, int(comet_cfg.get("weight_snapshot_num_values", 16)))

            chunk_store = None
            if rank == 0 and collector is not None and str(collector.streaming_mode).lower() != "none":
                try:
                    chunk_store = build_chunk_store(resolve_streaming_cfg(OmegaConf.to_container(cfg.streaming, resolve=True)))
                except Exception as exc:
                    logger.warning("Failed to initialize chunk store debug snapshot: %s", exc)
                    chunk_store = None

            max_steps = max(1, int(cfg.mini_train.get("max_steps", 1000)))
            grad_accum_steps = max(1, int(cfg.mini_train.get("grad_accum_steps", 1)))
            loss_cfg = cfg.mini_train.get("loss", {})
            legacy_kl_beta = float(cfg.mini_train.get("kl_beta", 1e-3))
            legacy_alpha = float(loss_cfg.get("alpha", 0.0))
            legacy_beta = float(loss_cfg.get("beta", 0.5))
            has_explicit_loss_coefs = any(
                key in loss_cfg for key in ("structural_coef", "behavioral_coef", "contrastive_coef", "kl_coef")
            )
            if not has_explicit_loss_coefs:
                if not (0.0 <= legacy_alpha <= 1.0):
                    raise ValueError(f"mini_train.loss.alpha must be in [0,1], got {legacy_alpha}")
                if not (0.0 <= legacy_beta <= 1.0):
                    raise ValueError(f"mini_train.loss.beta must be in [0,1], got {legacy_beta}")

            structural_coef = float(loss_cfg.get("structural_coef", (1.0 - legacy_alpha) * legacy_beta))
            behavioral_coef = float(loss_cfg.get("behavioral_coef", (1.0 - legacy_alpha) * (1.0 - legacy_beta)))
            contrastive_coef = float(loss_cfg.get("contrastive_coef", legacy_alpha))
            kl_coef = float(loss_cfg.get("kl_coef", legacy_kl_beta))
            contrastive_temperature = float(loss_cfg.get("contrastive_temperature", 0.07))
            contrastive_permute_inputs = bool(loss_cfg.get("contrastive_permute_inputs", True))
            contrastive_sign_flip_inputs = bool(loss_cfg.get("contrastive_sign_flip_inputs", True))
            if structural_coef < 0.0:
                raise ValueError(f"mini_train.loss.structural_coef must be >= 0, got {structural_coef}")
            if behavioral_coef < 0.0:
                raise ValueError(f"mini_train.loss.behavioral_coef must be >= 0, got {behavioral_coef}")
            if contrastive_coef < 0.0:
                raise ValueError(f"mini_train.loss.contrastive_coef must be >= 0, got {contrastive_coef}")
            if kl_coef < 0.0:
                raise ValueError(f"mini_train.loss.kl_coef must be >= 0, got {kl_coef}")
            if contrastive_temperature <= 0.0:
                raise ValueError(
                    f"mini_train.loss.contrastive_temperature must be > 0, got {contrastive_temperature}"
                )

            grad_clip_norm = float(cfg.mini_train.get("grad_clip_norm", 10.0))
            freeze_decoder_steps = max(0, int(cfg.mini_train.get("freeze_decoder_steps", 0)))
            max_x_rows = int(cfg.mini_train.get("max_x_rows", 0))
            patches_per_sample = int(cfg.mini_train.get("patches_per_sample", 16))
            patch_size = int(cfg.mini_model.get("patch_size", 64))
            log_every = max(1, int(cfg.mini_train.get("log_every", 10)))
            checkpoint_every = max(1, int(cfg.mini_train.get("checkpoint_every", 200)))
            checkpoint_dir = Path(str(cfg.mini_train.get("checkpoint_dir", "./checkpoints/mini_patch_vae")))

            grad_layer_monitor_cfg = telemetry_cfg.get("grad_layer_monitor", {})
            if not isinstance(grad_layer_monitor_cfg, (dict, DictConfig)):
                raise TypeError("mini_train.telemetry.grad_layer_monitor must be a mapping")
            grad_layer_monitor_enabled = bool(grad_layer_monitor_cfg.get("enabled", False)) and rank == 0
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
                ["mini_vae.encoder.", "mini_vae.decoder."],
            )
            grad_layer_monitor_include_prefixes_list: list[str] = []
            if isinstance(grad_layer_monitor_include_prefixes_raw, (list, tuple)):
                for item in grad_layer_monitor_include_prefixes_raw:
                    text = str(item).strip()
                    if text:
                        grad_layer_monitor_include_prefixes_list.append(text)
            else:
                text = str(grad_layer_monitor_include_prefixes_raw).strip()
                if text:
                    grad_layer_monitor_include_prefixes_list.append(text)
            if not grad_layer_monitor_include_prefixes_list:
                grad_layer_monitor_include_prefixes_list = ["mini_vae.encoder.", "mini_vae.decoder."]
            grad_layer_monitor_include_prefixes = tuple(grad_layer_monitor_include_prefixes_list)

            grad_layer_monitor_csv_path = Path(
                str(grad_layer_monitor_cfg.get("csv_path", str(checkpoint_dir / "grad_layer_rms.csv")))
            )
            grad_layer_monitor_plot_path = Path(
                str(grad_layer_monitor_cfg.get("plot_path", str(checkpoint_dir / "grad_layer_rms.png")))
            )
            grad_layer_monitor_heatmap_path = Path(
                str(grad_layer_monitor_cfg.get("heatmap_path", str(checkpoint_dir / "grad_layer_rms_heatmap.png")))
            )

            if grad_layer_monitor_enabled and grad_layer_monitor_save_csv and grad_layer_monitor_reset_csv_on_start:
                try:
                    if grad_layer_monitor_csv_path.exists():
                        grad_layer_monitor_csv_path.unlink()
                except Exception as exc:
                    logger.warning("Could not reset grad-layer CSV at %s: %s", grad_layer_monitor_csv_path, exc)

            if grad_layer_monitor_enabled:
                csv_abs = str(grad_layer_monitor_csv_path.resolve())
                plot_abs = str(grad_layer_monitor_plot_path.resolve())
                heatmap_abs = str(grad_layer_monitor_heatmap_path.resolve())
                logger.info(
                    "Grad-layer monitor enabled: every_steps=%s topk=%s csv=%s plot=%s heatmap=%s prefixes=%s cwd=%s rank=%s",
                    grad_layer_monitor_every_steps,
                    grad_layer_monitor_topk_layers,
                    csv_abs if grad_layer_monitor_save_csv else "<off>",
                    plot_abs if grad_layer_monitor_save_plot else "<off>",
                    heatmap_abs if (grad_layer_monitor_save_plot and grad_layer_monitor_save_heatmap) else "<off>",
                    list(grad_layer_monitor_include_prefixes),
                    str(Path.cwd()),
                    int(rank),
                )
            if freeze_decoder_steps > 0 and rank == 0:
                logger.info("Decoder freeze enabled: freeze_decoder_steps=%s", freeze_decoder_steps)

            test_eval_cfg = cfg.mini_train.get("test_eval", {})
            if not isinstance(test_eval_cfg, (dict, DictConfig)):
                raise TypeError("mini_train.test_eval must be a mapping")
            test_eval_requested = bool(test_eval_cfg.get("enabled", False))
            if synthetic_patch_enabled and test_eval_requested:
                if rank == 0:
                    logger.warning(
                        "mini_train.test_eval.enabled=true ignored because mini_train.synthetic_patch_source.enabled=true"
                    )
                test_eval_requested = False
            test_eval_every_steps = max(1, int(test_eval_cfg.get("every_steps", 1000)))
            test_eval_num_batches = max(1, int(test_eval_cfg.get("num_batches", 8)))
            test_eval_timeout_seconds = max(1.0, float(test_eval_cfg.get("timeout_seconds", 300.0)))
            test_eval_poll_sleep_seconds = max(0.001, float(test_eval_cfg.get("poll_sleep_seconds", 0.05)))
            test_eval_run_on_last_step = bool(test_eval_cfg.get("run_on_last_step", True))
            raw_test_eval_max_x_rows = test_eval_cfg.get("max_x_rows")
            text_test_eval_max_x_rows = str(raw_test_eval_max_x_rows).strip().lower() if raw_test_eval_max_x_rows is not None else ""
            if raw_test_eval_max_x_rows is None or text_test_eval_max_x_rows in {"", "none", "null"}:
                test_eval_max_x_rows = max_x_rows
            else:
                test_eval_max_x_rows = int(raw_test_eval_max_x_rows)
            if test_eval_max_x_rows < 0:
                test_eval_max_x_rows = 0

            if test_eval_requested and rank == 0 and test_eval_runtime_cfg is None:
                raise RuntimeError("mini_train.test_eval is enabled but test-eval runtime config was not initialized")

            if test_eval_requested and rank == 0:
                logger.info(
                    "Test-eval enabled: every_steps=%s num_batches=%s run_on_last_step=%s timeout=%.1fs max_x_rows=%s",
                    test_eval_every_steps,
                    test_eval_num_batches,
                    test_eval_run_on_last_step,
                    test_eval_timeout_seconds,
                    test_eval_max_x_rows,
                )

            train_ablation_cfg = cfg.mini_train.get("train_ablation_eval", {})
            if not isinstance(train_ablation_cfg, (dict, DictConfig)):
                raise TypeError("mini_train.train_ablation_eval must be a mapping")
            train_ablation_requested = bool(train_ablation_cfg.get("enabled", True))
            if synthetic_patch_enabled and train_ablation_requested:
                if rank == 0:
                    logger.warning(
                        "mini_train.train_ablation_eval.enabled=true ignored because mini_train.synthetic_patch_source.enabled=true"
                    )
                train_ablation_requested = False
            train_ablation_every_steps = max(1, int(train_ablation_cfg.get("every_steps", 100)))
            train_ablation_num_batches = max(1, int(train_ablation_cfg.get("num_batches", 4)))
            train_ablation_timeout_seconds = max(1.0, float(train_ablation_cfg.get("timeout_seconds", 120.0)))
            train_ablation_poll_sleep_seconds = max(
                0.001,
                float(train_ablation_cfg.get("poll_sleep_seconds", 0.05)),
            )
            train_ablation_run_on_last_step = bool(train_ablation_cfg.get("run_on_last_step", True))
            raw_train_ablation_max_x_rows = train_ablation_cfg.get("max_x_rows")
            text_train_ablation_max_x_rows = (
                str(raw_train_ablation_max_x_rows).strip().lower()
                if raw_train_ablation_max_x_rows is not None
                else ""
            )
            if raw_train_ablation_max_x_rows is None or text_train_ablation_max_x_rows in {"", "none", "null"}:
                train_ablation_max_x_rows = max_x_rows
            else:
                train_ablation_max_x_rows = int(raw_train_ablation_max_x_rows)
            if train_ablation_max_x_rows < 0:
                train_ablation_max_x_rows = 0

            if train_ablation_requested and rank == 0 and (dataset is None or collector is None):
                raise RuntimeError("mini_train.train_ablation_eval is enabled but train data pipeline is not initialized")

            if train_ablation_requested and rank == 0:
                logger.info(
                    "Train-ablation-eval enabled: every_steps=%s num_batches=%s run_on_last_step=%s timeout=%.1fs max_x_rows=%s",
                    train_ablation_every_steps,
                    train_ablation_num_batches,
                    train_ablation_run_on_last_step,
                    train_ablation_timeout_seconds,
                    train_ablation_max_x_rows,
                )

            loss_window = 0.0
            structural_window = 0.0
            behavioral_window = 0.0
            contrastive_window = 0.0
            recon_mix_window = 0.0
            kl_window = 0.0
            latent_z_grad_norm_window = 0.0
            latent_mu_grad_norm_window = 0.0
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
                "decoder_frozen": False,
            }

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
                latent_z_grad_norm_acc = 0.0
                latent_mu_grad_norm_acc = 0.0
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
                decoder_frozen_this_step = bool(global_step <= freeze_decoder_steps)
                decoder_grads_cleared_this_step = 0

                for micro_idx in range(grad_accum_steps):
                    sync_grad = micro_idx == grad_accum_steps - 1

                    t_fetch = time.perf_counter()
                    if synthetic_patch_enabled:
                        x, W, sample_info = sample_synthetic_layer(
                            device=device,
                            n_rows=synthetic_n_rows,
                            d_in=synthetic_d_in,
                            d_out=synthetic_d_out,
                            x_std=synthetic_x_std,
                            w_std=synthetic_w_std,
                        )
                    else:
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
                            (
                                total_loss,
                                structural_loss,
                                behavioral_loss,
                                contrastive_loss,
                                recon_mix_loss,
                                kl_loss,
                                latent_tensors,
                            ) = model(
                                X_full=X_full,
                                X_patch=X_patch,
                                w_patch=w_patch,
                                patch_idx=patch_idx,
                                structural_coef=structural_coef,
                                behavioral_coef=behavioral_coef,
                                contrastive_coef=contrastive_coef,
                                kl_coef=kl_coef,
                                contrastive_temperature=contrastive_temperature,
                                contrastive_permute_inputs=contrastive_permute_inputs,
                                contrastive_sign_flip_inputs=contrastive_sign_flip_inputs,
                                W_full=W,
                                out_idx=out_idx,
                                return_latent_tensors=True,
                            )
                            loss_for_backward = total_loss / grad_accum_steps
                        forward_ms_step += (time.perf_counter() - t_forward) * 1000.0

                        finite_flag = torch.tensor(
                            1 if torch.isfinite(loss_for_backward.detach()) else 0,
                            dtype=torch.int32,
                            device=device,
                        )
                        print(finite_flag, "FINITE FLAG")
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

                            scale_inv = 1.0
                            if scaler.is_enabled():
                                scale_value = float(scaler.get_scale())
                                if scale_value > 0.0:
                                    scale_inv = 1.0 / scale_value

                            z_ref = latent_tensors.get("z")
                            if torch.is_tensor(z_ref) and z_ref.grad is not None:
                                latent_z_grad_norm_acc += float(z_ref.grad.detach().norm().item()) * scale_inv

                            mu_ref = latent_tensors.get("mu")
                            if torch.is_tensor(mu_ref) and mu_ref.grad is not None:
                                latent_mu_grad_norm_acc += float(mu_ref.grad.detach().norm().item()) * scale_inv

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

                if decoder_frozen_this_step:
                    decoder_grads_cleared_this_step = _set_optimizer_group_grads_to_none(
                        optimizer=optimizer,
                        group_name="mini_decoder",
                    )

                grad_clip_coef = 1.0
                if grad_clip_norm > 0.0:
                    print('We are here')
                    clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    grad_global_before_clip = float(clip_return)
                    grad_clip_coef = min(1.0, float(grad_clip_norm) / max(1e-12, grad_global_before_clip))
                    grad_stats = compute_grad_stats(model)
                else:
                    grad_stats = compute_grad_stats(model)
                    grad_global_before_clip = float(grad_stats.get("grad/global_norm", 0.0))
                grad_stats["grad/global_norm_before_clip"] = grad_global_before_clip
                grad_stats["grad/clip_coef"] = grad_clip_coef

                print("SCK MDK")
                for name, param in dict(model.named_parameters()).items():
                    if param.grad is not None:
                        print(name, torch.norm(param.grad))
                    else:
                        print(name, None)
                if grad_layer_monitor_enabled and (
                    global_step == 1 or (global_step % grad_layer_monitor_every_steps == 0)
                ):
                    layer_rms_post_clip = collect_grad_rms_per_layer(
                        model=model,
                        include_prefixes=grad_layer_monitor_include_prefixes,
                        weights_only=grad_layer_monitor_weights_only,
                    )
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
                    ranked_low_layers = sorted(
                        ranked_layers,
                        key=lambda item: float(item[1]),
                    )
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
                        "decoder_frozen": decoder_frozen_this_step,
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
                        logger.info(
                            "grad_layer_monitor_write step=%s csv=%s rows=%s",
                            global_step,
                            str(grad_layer_monitor_csv_path.resolve()),
                            len(layer_keys),
                        )

                    plot_saved = False
                    heatmap_saved = False
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
                            heatmap_saved = True
                    if grad_layer_monitor_save_plot and (
                        global_step == 1
                        or (
                            grad_layer_monitor_plot_every_steps > 0
                            and global_step % grad_layer_monitor_plot_every_steps == 0
                        )
                    ):
                        logger.info(
                            "grad_layer_monitor_plot step=%s plot_saved=%s plot=%s heatmap_saved=%s heatmap=%s",
                            global_step,
                            plot_saved,
                            str(grad_layer_monitor_plot_path.resolve()),
                            heatmap_saved if grad_layer_monitor_save_heatmap else False,
                            str(grad_layer_monitor_heatmap_path.resolve()) if grad_layer_monitor_save_heatmap else "<off>",
                        )

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
                step_latent_z_grad_norm = latent_z_grad_norm_acc / grad_accum_steps
                step_latent_mu_grad_norm = latent_mu_grad_norm_acc / grad_accum_steps

                stats = torch.tensor(
                    [
                        step_loss,
                        step_structural,
                        step_behavioral,
                        step_contrastive,
                        step_recon_mix,
                        step_kl,
                        step_latent_z_grad_norm,
                        step_latent_mu_grad_norm,
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
                latent_z_grad_norm_window += float(stats[6].item())
                latent_mu_grad_norm_window += float(stats[7].item())
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
                    avg_latent_z_grad_norm = latent_z_grad_norm_window / max(1, window_steps)
                    avg_latent_mu_grad_norm = latent_mu_grad_norm_window / max(1, window_steps)
                    lr_default = _get_optimizer_group_lr(optimizer, "default", float(optimizer.param_groups[0]["lr"]))
                    lr_encoder = _get_optimizer_group_lr(optimizer, "mini_encoder", lr_default)
                    lr_decoder = _get_optimizer_group_lr(optimizer, "mini_decoder", lr_default)
                    lr = lr_default
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
                        "coef_str=%.3f coef_beh=%.3f coef_con=%.3f coef_kl=%.4f "
                        "lr=%.6e lr_enc=%.6e lr_dec=%.6e steps/s=%.2f patches=%s cache=%s "
                        "t_fetch=%.2fms t_patch=%.2fms t_fwd=%.2fms t_bwd=%.2fms t_opt=%.2fms "
                        "grad_norm=%.4f grad_rms=%.6f clip=%.3f "
                        "grad_enc=%.6f grad_dec=%.6f enc_params_with_grad=%.0f dec_params_with_grad=%.0f "
                        "dL_dz=%.6f dL_dmu=%.6f dec_frozen=%s dec_grads_cleared=%s",
                        global_step,
                        max_steps,
                        avg_loss,
                        avg_structural,
                        avg_behavioral,
                        avg_contrastive,
                        avg_recon_mix,
                        avg_kl,
                        structural_coef,
                        behavioral_coef,
                        contrastive_coef,
                        kl_coef,
                        lr,
                        lr_encoder,
                        lr_decoder,
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
                        float(grad_stats.get("grad/mini_encoder_rms", 0.0)),
                        float(grad_stats.get("grad/mini_decoder_rms", 0.0)),
                        float(grad_stats.get("grad/mini_encoder_params_with_grad", 0.0)),
                        float(grad_stats.get("grad/mini_decoder_params_with_grad", 0.0)),
                        float(avg_latent_z_grad_norm),
                        float(avg_latent_mu_grad_norm),
                        decoder_frozen_this_step,
                        decoder_grads_cleared_this_step,
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
                    if grad_layer_monitor_enabled:
                        logger.info(
                            "grad_layer_monitor step=%s layers=%s dec_frozen=%s clip=%.3f global_pre=%.3e global_post=%.3e "
                            "param=%.3e ratio_pre=%.3e ratio_post=%.3e top_pre=%s low_pre=%s top_ratio=%s",
                            int(latest_grad_layer_snapshot.get("step", 0)),
                            int(latest_grad_layer_snapshot.get("num_layers", 0)),
                            bool(latest_grad_layer_snapshot.get("decoder_frozen", False)),
                            float(latest_grad_layer_snapshot.get("clip_coef", 1.0)),
                            float(latest_grad_layer_snapshot.get("global_grad_rms_pre_clip", 0.0)),
                            float(latest_grad_layer_snapshot.get("global_grad_rms_post_clip", 0.0)),
                            float(latest_grad_layer_snapshot.get("global_param_rms", 0.0)),
                            float(latest_grad_layer_snapshot.get("global_grad_to_param_ratio_pre_clip", 0.0)),
                            float(latest_grad_layer_snapshot.get("global_grad_to_param_ratio_post_clip", 0.0)),
                            latest_grad_layer_snapshot.get("top_layers_pre_clip", [])[: min(5, grad_layer_monitor_topk_layers)],
                            latest_grad_layer_snapshot.get("low_layers_pre_clip", [])[: min(5, grad_layer_monitor_topk_layers)],
                            latest_grad_layer_snapshot.get("top_layers_ratio_pre_clip", [])[: min(5, grad_layer_monitor_topk_layers)],
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
                        "train/recon": float(avg_structural),
                        "train/kl": float(avg_kl),
                        "train/latent_z_grad_norm": float(avg_latent_z_grad_norm),
                        "train/latent_mu_grad_norm": float(avg_latent_mu_grad_norm),
                        "train/loss_structural_coef": float(structural_coef),
                        "train/loss_behavioral_coef": float(behavioral_coef),
                        "train/loss_contrastive_coef": float(contrastive_coef),
                        "train/loss_kl_coef": float(kl_coef),
                        "train/decoder_frozen": float(1.0 if decoder_frozen_this_step else 0.0),
                        "train/decoder_grads_cleared": float(decoder_grads_cleared_this_step),
                        "train/decoder_freeze_remaining_steps": float(max(0, freeze_decoder_steps - global_step)),
                        "train/lr": float(lr),
                        "train/lr_default": float(lr_default),
                        "train/lr_encoder": float(lr_encoder),
                        "train/lr_decoder": float(lr_decoder),
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
                        "grad/distribution_encoder_numel": float(grad_stats.get("grad/distribution_encoder_numel", 0.0)),
                        "grad/mini_encoder_numel": float(grad_stats.get("grad/mini_encoder_numel", 0.0)),
                        "grad/mini_decoder_numel": float(grad_stats.get("grad/mini_decoder_numel", 0.0)),
                        "grad/mini_vae_other_numel": float(grad_stats.get("grad/mini_vae_other_numel", 0.0)),
                        "grad/distribution_encoder_params_with_grad": float(
                            grad_stats.get("grad/distribution_encoder_params_with_grad", 0.0)
                        ),
                        "grad/mini_encoder_params_with_grad": float(
                            grad_stats.get("grad/mini_encoder_params_with_grad", 0.0)
                        ),
                        "grad/mini_decoder_params_with_grad": float(
                            grad_stats.get("grad/mini_decoder_params_with_grad", 0.0)
                        ),
                        "grad/mini_vae_other_params_with_grad": float(
                            grad_stats.get("grad/mini_vae_other_params_with_grad", 0.0)
                        ),
                        "param/rms": float(grad_stats.get("param/rms", 0.0)),
                        "grad_to_param_rms_ratio": float(grad_stats.get("grad_to_param_rms_ratio", 0.0)),
                    }
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
                            "grad_layer_monitor": latest_grad_layer_snapshot if grad_layer_monitor_enabled else None,
                        }
                    )

                    loss_window = 0.0
                    structural_window = 0.0
                    behavioral_window = 0.0
                    contrastive_window = 0.0
                    recon_mix_window = 0.0
                    kl_window = 0.0
                    latent_z_grad_norm_window = 0.0
                    latent_mu_grad_norm_window = 0.0
                    window_steps = 0
                    step_fetch_ms_window = 0.0
                    step_patch_ms_window = 0.0
                    step_forward_ms_window = 0.0
                    step_backward_ms_window = 0.0
                    step_opt_ms_window = 0.0
                    patch_window.clear()
                    t0 = time.time()

                run_comet_weight_snapshot = (
                    rank == 0
                    and comet_tracker is not None
                    and comet_tracker.enabled
                    and comet_weight_snapshot_enabled
                    and (global_step % comet_weight_snapshot_every_steps == 0)
                )
                if run_comet_weight_snapshot:
                    try:
                        train_model = model.module if isinstance(model, DDP) else model
                        with torch.no_grad():
                            with autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                                w_target_norm, w_scale = train_model.normalize_patch_weights(w_patch)
                                w_target_unit = train_model.project_patch_weights_to_unit_sphere(w_target_norm)
                                dist_var_tokens_dbg, dist_patch_embed_dbg = train_model.distribution_encoder(
                                    X=X_full, patch_idx=patch_idx
                                )
                                w_pred_raw_dbg, _, _, _ = train_model.mini_vae(
                                    w_patch=w_target_unit,
                                    dist_var_tokens=dist_var_tokens_dbg,
                                    dist_patch_embed=dist_patch_embed_dbg,
                                )
                                w_pred_unit = train_model.project_patch_weights_to_unit_sphere(w_pred_raw_dbg)

                        w_target_fp32 = w_target_unit.detach().float()
                        w_pred_fp32 = w_pred_unit.detach().float()
                        w_scale_fp32 = w_scale.detach().float()
                        w_patch_norm_fp32 = w_target_norm.detach().float()
                        sphere_scale_fp32 = w_patch_norm_fp32.norm(dim=1, keepdim=True).clamp_min(1e-4)
                        w_target_raw_fp32 = w_patch.detach().float()
                        w_pred_raw_fp32 = w_pred_fp32 * w_scale_fp32 * sphere_scale_fp32
                        pred_target_mse = float((w_pred_fp32 - w_target_fp32).pow(2).mean().item())
                        pred_target_mse_raw = float((w_pred_raw_fp32 - w_target_raw_fp32).pow(2).mean().item())

                        head_len = min(comet_weight_snapshot_num_values, int(w_target_fp32.shape[1]))
                        pred_head = [float(v) for v in w_pred_fp32[0, :head_len].cpu().tolist()]
                        target_head = [float(v) for v in w_target_fp32[0, :head_len].cpu().tolist()]
                        diff_head = [float(p - t) for p, t in zip(pred_head, target_head)]
                        pred_raw_head = [float(v) for v in w_pred_raw_fp32[0, :head_len].cpu().tolist()]
                        target_raw_head = [float(v) for v in w_target_raw_fp32[0, :head_len].cpu().tolist()]
                        diff_raw_head = [float(p - t) for p, t in zip(pred_raw_head, target_raw_head)]

                        comet_tracker.log_metrics(
                            {
                                "train/weights_snapshot/mse_norm": pred_target_mse,
                                "train/weights_snapshot/mse_raw": pred_target_mse_raw,
                                "train/weights_snapshot/pred_mean": float(w_pred_fp32.mean().item()),
                                "train/weights_snapshot/target_mean": float(w_target_fp32.mean().item()),
                                "train/weights_snapshot/pred_std": float(w_pred_fp32.std(unbiased=False).item()),
                                "train/weights_snapshot/target_std": float(w_target_fp32.std(unbiased=False).item()),
                            },
                            step=global_step,
                        )
                        comet_tracker.log_text(
                            text=f"weights_snapshot_step_{global_step}",
                            metadata={
                                "step": int(global_step),
                                "num_values": int(head_len),
                                "pred_head_norm": pred_head,
                                "target_head_norm": target_head,
                                "diff_head_norm": diff_head,
                                "pred_head_raw": pred_raw_head,
                                "target_head_raw": target_raw_head,
                                "diff_head_raw": diff_raw_head,
                                "patch_idx_head": patch_idx_head,
                                "out_idx_head": out_idx_head,
                                "model_name": (sample_info_latest or {}).get("model_name"),
                                "layer_name": (sample_info_latest or {}).get("layer_name"),
                            },
                        )
                        status_writer.write(
                            {
                                "step": int(global_step),
                                "timestamp": float(time.time()),
                                "weights_snapshot": {
                                    "mse_norm": pred_target_mse,
                                    "mse_raw": pred_target_mse_raw,
                                    "num_values": int(head_len),
                                    "pred_head_norm": pred_head,
                                    "target_head_norm": target_head,
                                    "diff_head_norm": diff_head,
                                    "pred_head_raw": pred_raw_head,
                                    "target_head_raw": target_raw_head,
                                    "diff_head_raw": diff_raw_head,
                                },
                            }
                        )
                    except Exception as exc:
                        logger.warning("Comet weight snapshot log failed at step=%s: %s", global_step, exc)

                run_train_ablation_this_step = train_ablation_requested and (
                    (global_step % train_ablation_every_steps == 0)
                    or (train_ablation_run_on_last_step and global_step == max_steps)
                )
                if run_train_ablation_this_step and is_distributed:
                    dist.barrier()

                if run_train_ablation_this_step and rank == 0:
                    if dataset is None or collector is None:
                        raise RuntimeError("train_ablation_eval requires initialized train dataset and collector")
                    train_ablation_started = time.time()
                    try:
                        train_ablation_report = run_train_ablation_eval(
                            model=model,
                            dataset=dataset,
                            collector=collector,
                            device=device,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                            patch_size=patch_size,
                            patches_per_sample=patches_per_sample,
                            structural_coef=structural_coef,
                            behavioral_coef=behavioral_coef,
                            kl_coef=kl_coef,
                            num_batches=train_ablation_num_batches,
                            max_x_rows=train_ablation_max_x_rows,
                            timeout_seconds=train_ablation_timeout_seconds,
                            poll_sleep_seconds=train_ablation_poll_sleep_seconds,
                            logger=logger,
                        )
                        train_ablation_metrics = train_ablation_report.get("metrics", {})
                        train_ablation_mix = train_ablation_report.get("sample_mix", {})
                        train_ablation_duration_s = float(
                            train_ablation_report.get("duration_s", time.time() - train_ablation_started)
                        )
                        logger.info(
                            "train_ablation_eval step=%s batches=%s rand_z_recon=%.6f rand_dist_recon=%.6f rand_dist_kl=%.6f "
                            "x_rows=%.2f bps=%.2f duration=%.2fs",
                            global_step,
                            int(train_ablation_report.get("num_batches", train_ablation_num_batches)),
                            float(train_ablation_metrics.get("decoder_random_latent_recon_mix", 0.0)),
                            float(train_ablation_metrics.get("random_dist_recon_mix", 0.0)),
                            float(train_ablation_metrics.get("random_dist_kl", 0.0)),
                            float(train_ablation_metrics.get("x_rows", 0.0)),
                            float(train_ablation_metrics.get("batches_per_sec", 0.0)),
                            train_ablation_duration_s,
                        )
                        logger.info(
                            "train_ablation_eval_mix step=%s models=%s datasets=%s layers=%s",
                            global_step,
                            train_ablation_mix.get("models_top", []),
                            train_ablation_mix.get("datasets_top", []),
                            train_ablation_mix.get("layers_top", []),
                        )
                        status_writer.write(
                            {
                                "step": int(global_step),
                                "timestamp": float(time.time()),
                                "train_ablation_eval": train_ablation_report,
                            }
                        )
                        if comet_tracker is not None and comet_tracker.enabled:
                            comet_tracker.log_metrics(
                                {
                                    "train_eval/decoder_random_latent_structural": float(
                                        train_ablation_metrics.get("decoder_random_latent_structural", 0.0)
                                    ),
                                    "train_eval/decoder_random_latent_behavioral": float(
                                        train_ablation_metrics.get("decoder_random_latent_behavioral", 0.0)
                                    ),
                                    "train_eval/decoder_random_latent_recon_mix": float(
                                        train_ablation_metrics.get("decoder_random_latent_recon_mix", 0.0)
                                    ),
                                    "train_eval/random_dist_structural": float(
                                        train_ablation_metrics.get("random_dist_structural", 0.0)
                                    ),
                                    "train_eval/random_dist_behavioral": float(
                                        train_ablation_metrics.get("random_dist_behavioral", 0.0)
                                    ),
                                    "train_eval/random_dist_recon_mix": float(
                                        train_ablation_metrics.get("random_dist_recon_mix", 0.0)
                                    ),
                                    "train_eval/random_dist_kl": float(train_ablation_metrics.get("random_dist_kl", 0.0)),
                                    "train_eval/random_dist_total": float(
                                        train_ablation_metrics.get("random_dist_total", 0.0)
                                    ),
                                    "train_eval/x_rows": float(train_ablation_metrics.get("x_rows", 0.0)),
                                    "train_eval/batches_per_sec": float(
                                        train_ablation_metrics.get("batches_per_sec", 0.0)
                                    ),
                                    "train_eval/duration_s": float(train_ablation_duration_s),
                                },
                                step=global_step,
                            )
                    except Exception as exc:
                        logger.exception("train_ablation_eval failed at step=%s: %s", global_step, exc)
                        status_writer.write(
                            {
                                "step": int(global_step),
                                "timestamp": float(time.time()),
                                "train_ablation_eval_error": str(exc),
                            }
                        )

                if run_train_ablation_this_step and is_distributed:
                    dist.barrier()

                run_test_eval_this_step = test_eval_requested and (
                    (global_step % test_eval_every_steps == 0)
                    or (test_eval_run_on_last_step and global_step == max_steps)
                )
                if run_test_eval_this_step and is_distributed:
                    dist.barrier()

                if run_test_eval_this_step and rank == 0:
                    eval_started = time.time()
                    try:
                        if test_eval_runtime_cfg is None:
                            raise RuntimeError("test_eval runtime config is None")

                        with data_pipeline(
                            test_eval_runtime_cfg,
                            start_collector=True,
                            predownload_models=test_eval_predownload,
                            logger=logger,
                        ) as (test_dataset, test_collector):
                            test_eval_report = run_test_eval(
                                model=model,
                                dataset=test_dataset,
                                collector=test_collector,
                                device=device,
                                amp_enabled=amp_enabled,
                                amp_dtype=amp_dtype,
                                patch_size=patch_size,
                                patches_per_sample=patches_per_sample,
                                structural_coef=structural_coef,
                                behavioral_coef=behavioral_coef,
                                contrastive_coef=contrastive_coef,
                                kl_coef=kl_coef,
                                contrastive_temperature=contrastive_temperature,
                                contrastive_permute_inputs=contrastive_permute_inputs,
                                contrastive_sign_flip_inputs=contrastive_sign_flip_inputs,
                                num_batches=test_eval_num_batches,
                                max_x_rows=test_eval_max_x_rows,
                                timeout_seconds=test_eval_timeout_seconds,
                                poll_sleep_seconds=test_eval_poll_sleep_seconds,
                                logger=logger,
                            )

                        metrics = test_eval_report.get("metrics", {})
                        sample_mix = test_eval_report.get("sample_mix", {})
                        duration_s = float(test_eval_report.get("duration_s", time.time() - eval_started))
                        logger.info(
                            "test_eval step=%s batches=%s loss=%.6f str=%.6f beh=%.6f con=%.6f recon_mix=%.6f kl=%.6f "
                            "x_rows=%.2f bps=%.2f duration=%.2fs",
                            global_step,
                            int(test_eval_report.get("num_batches", test_eval_num_batches)),
                            float(metrics.get("loss", 0.0)),
                            float(metrics.get("structural", 0.0)),
                            float(metrics.get("behavioral", 0.0)),
                            float(metrics.get("contrastive", 0.0)),
                            float(metrics.get("recon_mix", 0.0)),
                            float(metrics.get("kl", 0.0)),
                            float(metrics.get("x_rows", 0.0)),
                            float(metrics.get("batches_per_sec", 0.0)),
                            duration_s,
                        )
                        logger.info(
                            "test_eval_ablation step=%s rand_z_recon=%.6f rand_dist_recon=%.6f rand_dist_kl=%.6f rand_dist_total=%.6f",
                            global_step,
                            float(metrics.get("decoder_random_latent_recon_mix", 0.0)),
                            float(metrics.get("random_dist_recon_mix", 0.0)),
                            float(metrics.get("random_dist_kl", 0.0)),
                            float(metrics.get("random_dist_total", 0.0)),
                        )
                        logger.info(
                            "test_eval_mix step=%s models=%s datasets=%s layers=%s",
                            global_step,
                            sample_mix.get("models_top", []),
                            sample_mix.get("datasets_top", []),
                            sample_mix.get("layers_top", []),
                        )

                        status_writer.write(
                            {
                                "step": int(global_step),
                                "timestamp": float(time.time()),
                                "test_eval": test_eval_report,
                            }
                        )
                        if comet_tracker is not None and comet_tracker.enabled:
                            comet_tracker.log_metrics(
                                {
                                    "test/loss": float(metrics.get("loss", 0.0)),
                                    "test/structural": float(metrics.get("structural", 0.0)),
                                    "test/behavioral": float(metrics.get("behavioral", 0.0)),
                                    "test/contrastive": float(metrics.get("contrastive", 0.0)),
                                    "test/recon_mix": float(metrics.get("recon_mix", 0.0)),
                                    "test/kl": float(metrics.get("kl", 0.0)),
                                    "test/decoder_random_latent_structural": float(
                                        metrics.get("decoder_random_latent_structural", 0.0)
                                    ),
                                    "test/decoder_random_latent_behavioral": float(
                                        metrics.get("decoder_random_latent_behavioral", 0.0)
                                    ),
                                    "test/decoder_random_latent_recon_mix": float(
                                        metrics.get("decoder_random_latent_recon_mix", 0.0)
                                    ),
                                    "test/random_dist_structural": float(metrics.get("random_dist_structural", 0.0)),
                                    "test/random_dist_behavioral": float(metrics.get("random_dist_behavioral", 0.0)),
                                    "test/random_dist_recon_mix": float(metrics.get("random_dist_recon_mix", 0.0)),
                                    "test/random_dist_kl": float(metrics.get("random_dist_kl", 0.0)),
                                    "test/random_dist_total": float(metrics.get("random_dist_total", 0.0)),
                                    "test/x_rows": float(metrics.get("x_rows", 0.0)),
                                    "test/batches_per_sec": float(metrics.get("batches_per_sec", 0.0)),
                                    "test/duration_s": float(duration_s),
                                },
                                step=global_step,
                            )
                    except Exception as exc:
                        logger.exception("test_eval failed at step=%s: %s", global_step, exc)
                        status_writer.write(
                            {
                                "step": int(global_step),
                                "timestamp": float(time.time()),
                                "test_eval_error": str(exc),
                            }
                        )

                if run_test_eval_this_step and is_distributed:
                    dist.barrier()

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

        if is_distributed and dist.is_initialized():
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
    promote_run_profile_to_root(cfg)
    run_artifacts = runtime_configure_per_run_artifacts(cfg, run_label="train_mini_vae")
    print(
        f"[train_mini_vae] run_id={run_artifacts.get('run_id')} artifacts_root={run_artifacts.get('root_dir')}",
        flush=True,
    )
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
