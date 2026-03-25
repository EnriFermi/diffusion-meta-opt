from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
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


SourceSample = tuple[torch.Tensor, torch.Tensor]


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


def _compute_kl_beta_for_step(
    global_step: int,
    *,
    target_beta: float,
    schedule_enabled: bool,
    start_beta: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    if not schedule_enabled:
        return float(target_beta)
    if global_step <= max(0, int(warmup_steps)):
        return float(start_beta)
    if ramp_steps <= 0:
        return float(target_beta)
    progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(ramp_steps)))
    cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(start_beta + (target_beta - start_beta) * cosine_progress)


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
            if isinstance(tags, (list, tuple, ListConfig)):
                for tag in tags:
                    exp.add_tag(str(tag))

            comet_params: dict[str, Any] = {
                "train.max_steps": int(train_cfg.get("max_steps", 0)),
                "train.lr": float(train_cfg.get("lr", 0.0)),
                "train.grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
                "train.kl_beta": float(train_cfg.get("kl_beta", 0.0)),
                "train.kl_schedule.enabled": bool(train_cfg.get("kl_schedule", {}).get("enabled", False)),
                "train.kl_schedule.start_beta": float(train_cfg.get("kl_schedule", {}).get("start_beta", 0.0)),
                "train.kl_schedule.warmup_steps": int(train_cfg.get("kl_schedule", {}).get("warmup_steps", 0)),
                "train.kl_schedule.ramp_steps": int(train_cfg.get("kl_schedule", {}).get("ramp_steps", 0)),
                "train.behavioral_coef": float(train_cfg.get("behavioral_coef", 0.0)),
                "train.structural_coef": float(train_cfg.get("structural_coef", 0.0)),
                "model.patch_size": int(model_cfg.get("patch_size", 16)),
                "model.big_vae.use_latent_sampling": bool(big_cfg.get("use_latent_sampling", True)),
                "model.big_vae.disable_distribution_encoder": bool(big_cfg.get("disable_distribution_encoder", False)),
                "model.big_vae.patch_tokenizer_kind": str(big_cfg.get("patch_tokenizer_kind", "residual")),
                "model.big_vae.distribution_encoder_conditioning_kind": str(
                    big_cfg.get("distribution_encoder_conditioning_kind", "legacy")
                ),
                "streaming.mode": str(streaming_cfg.get("mode", "none")),
                "collector.mode": str(collector_cfg.get("mode", "auto")),
                "collector.device": str(collector_cfg.get("device", "")),
                "train.device": str(train_cfg.get("device", "")),
            }
            resume_state_cfg = train_cfg.get("resume_state", {})
            if isinstance(resume_state_cfg, (dict, DictConfig)):
                comet_params["train.resume_state.enabled"] = bool(resume_state_cfg.get("enabled", False))
                comet_params["train.resume_state.auto_resume"] = bool(resume_state_cfg.get("auto_resume", True))
                comet_params["train.resume_state.save_every"] = int(
                    resume_state_cfg.get("save_every", train_cfg.get("checkpoint_every", 0))
                )
            clip_by_part_cfg = train_cfg.get("grad_clip_norm_by_part", {})
            if isinstance(clip_by_part_cfg, (dict, DictConfig)):
                for group_name in _grad_stat_group_prefixes():
                    if group_name not in clip_by_part_cfg:
                        continue
                    comet_params[f"train.grad_clip_norm_by_part.{group_name}"] = float(clip_by_part_cfg[group_name])
            exp.log_parameters(comet_params)

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


def _strip_ddp_prefix(name: str) -> str:
    out = str(name)
    # DDP + torch.compile wrappers can both prepend module qualifiers.
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


def _layer_key_from_param_name(name: str) -> str:
    parts = name.split(".")
    if not parts:
        return name
    last = parts[-1]
    if last in {"weight", "bias"} or last.endswith("_weight") or last.endswith("_bias"):
        if len(parts) > 1:
            return ".".join(parts[:-1])
    return name


def _grad_stat_group_prefixes() -> dict[str, tuple[str, ...]]:
    return {
        "distribution_encoder": ("distribution_encoder.",),
        "patch_tokenizer": ("patch_tokenizer.", "patch_token_proj.", "cls_token"),
        "encoder": (
            "encoder_layers.",
            "latent_resampler_layers.",
            "enc_dist_inject_projs.",
            "enc_dist_to_latent_heads.",
            "encoder_conditioning_adapters.",
            "latent_base",
            "latent_norm.",
            "to_mu.",
            "to_logvar.",
        ),
        "decoder": (
            "decoder_layers.",
            "latent_to_decoder.",
            "pos_proj.",
            "query_pos_proj.",
            "query_proj.",
            "direction_head.",
            "z_shortcut_proj.",
            "z_shortcut.",
            "scale_head.",
        ),
        "big_vae_other": tuple(),
    }


def _grad_group_name_for_param(name: str, groups: dict[str, tuple[str, ...]]) -> str:
    for group_name, prefixes in groups.items():
        if group_name == "big_vae_other":
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            return group_name
    return "big_vae_other"


def _parameter_count_summary(model: nn.Module) -> tuple[int, int, int]:
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        count = int(param.numel())
        total_params += count
        if param.requires_grad:
            trainable_params += count
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


def _collect_params_by_grad_group(model: nn.Module, groups: dict[str, tuple[str, ...]]) -> dict[str, list[nn.Parameter]]:
    out: dict[str, list[nn.Parameter]] = {group_name: [] for group_name in groups}
    for raw_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name = _strip_ddp_prefix(raw_name)
        group_name = _grad_group_name_for_param(name, groups)
        out[group_name].append(param)
    return out


def _grad_l2_norm_for_params(params: list[nn.Parameter]) -> float:
    sum_sq = 0.0
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        g = grad.detach()
        sum_sq += float((g * g).sum().item())
    return math.sqrt(sum_sq)


def _clip_grad_norm_with_optional_foreach(
    get_params: Callable[[], Any],
    max_norm: float,
    *,
    use_foreach: bool,
) -> Any:
    """Use foreach clipping when available; gracefully fallback on older torch."""
    try:
        return torch.nn.utils.clip_grad_norm_(get_params(), max_norm, foreach=use_foreach)
    except TypeError:
        return torch.nn.utils.clip_grad_norm_(get_params(), max_norm)


def _clip_return_to_float(clip_return: Any) -> float:
    if torch.is_tensor(clip_return):
        return float(clip_return.detach().item())
    return float(clip_return)


def compute_grad_stats(model: nn.Module) -> dict[str, float]:
    groups = _grad_stat_group_prefixes()
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

        p = param.detach()
        param_sum_sq += float(p.pow(2).sum().item())
        param_numel += int(p.numel())

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

        group_name = _grad_group_name_for_param(name, groups)
        group_param_count[group_name] += 1
        group_sums[group_name] += float((g * g).sum().item())
        group_numel[group_name] += numel

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


def _collect_nonfinite_grad_report(
    model: nn.Module,
    *,
    max_items: int = 12,
) -> dict[str, Any] | None:
    groups = _grad_stat_group_prefixes()
    total_bad_tensors = 0
    total_nan = 0
    total_inf = 0
    items: list[dict[str, Any]] = []

    for raw_name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        grad = param.grad.detach()
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all().item()):
            continue

        total_bad_tensors += 1
        nan_count = int(torch.isnan(grad).sum().item())
        inf_count = int(torch.isinf(grad).sum().item())
        total_nan += nan_count
        total_inf += inf_count

        if len(items) >= int(max_items):
            continue

        name = _strip_ddp_prefix(raw_name)
        item: dict[str, Any] = {
            "name": name,
            "layer": _layer_key_from_param_name(name),
            "group": _grad_group_name_for_param(name, groups),
            "shape": list(grad.shape),
            "dtype": str(grad.dtype),
            "device": str(grad.device),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "finite_count": int(finite_mask.sum().item()),
        }
        if item["finite_count"] > 0:
            finite_vals = grad[finite_mask]
            item["finite_abs_max"] = float(finite_vals.abs().max().item())
            item["finite_mean"] = float(finite_vals.mean().item())
            item["finite_std"] = float(finite_vals.std(unbiased=False).item())
        items.append(item)

    if total_bad_tensors == 0:
        return None

    return {
        "bad_tensors": int(total_bad_tensors),
        "total_nan": int(total_nan),
        "total_inf": int(total_inf),
        "items": items,
    }


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
        if weights_only and param.ndim < 2:
            continue

        grad = param.grad
        if grad is None:
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
        out[layer] = math.sqrt(sumsq / float(max(1, count_by_layer[layer])))
    if global_count > 0:
        out["__global__"] = math.sqrt(global_sumsq / float(global_count))
    return out


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
        out[layer] = math.sqrt(sumsq / float(max(1, count_by_layer[layer])))
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

    plt.title("BigVAE Gradient RMS Per Layer (pre-clip)")
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
    layers.sort(key=lambda key: sum(val for _, val in history[key]) / max(1, len(history[key])))
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
    ax.set_title("BigVAE Gradient RMS Heatmap (pre-clip)")
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


def _save_checkpoint_payload(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _save_checkpoint(
    model: torch.nn.Module,
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
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    step_path = checkpoint_dir / f"step_{step_idx:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    _save_checkpoint_payload(payload, step_path)
    _save_checkpoint_payload(payload, latest_path)
    logger.info("Model checkpoint saved: %s", step_path)


def _save_resume_state_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
    stage: int,
    state_dir: Path,
) -> None:
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

    state_path = state_dir / f"step_{step_idx:07d}.pt"
    _save_checkpoint_payload(payload, state_path)

    for stale_path in sorted(state_dir.glob("step_*.pt")):
        if stale_path == state_path:
            continue
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not delete stale resume-state checkpoint %s: %s", stale_path, exc)
    logger.info("Resume-state checkpoint saved: %s", state_path)


def _maybe_dump_fixed_training_batch(
    *,
    W_s: torch.Tensor,
    x_s: torch.Tensor,
    cfg: DictConfig,
    logger: logging.Logger,
    global_step: int,
    stage: int,
    patch_size: int,
    max_T_patches: int,
    max_d_out: int,
    slice_batch_size: int,
) -> Path | None:
    fixed_batch_cfg = cfg.train.get("fixed_training_batch", {})
    if fixed_batch_cfg is None:
        return None
    if not isinstance(fixed_batch_cfg, (dict, DictConfig)):
        raise TypeError("train.fixed_training_batch must be a mapping")

    dump_path_text = str(fixed_batch_cfg.get("dump_path", "")).strip()
    if not dump_path_text:
        return None

    dump_path = Path(dump_path_text)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fixed_batch_W": W_s.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
        "fixed_batch_x": x_s.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
        "meta": {
            "capture_step": int(global_step),
            "stage": int(stage),
            "patch_size": int(patch_size),
            "max_T_patches": int(max_T_patches),
            "max_d_out": int(max_d_out),
            "slice_batch_size": int(slice_batch_size),
            "W_shape": list(W_s.shape),
            "x_shape": list(x_s.shape),
            "data_seed": int(cfg.data.get("seed", 42)),
        },
    }
    torch.save(payload, dump_path)
    logger.info("Fixed training batch dump saved: %s", dump_path)
    return dump_path


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

        x = _prepare_cpu_sample_tensor(x)
        W = _prepare_cpu_sample_tensor(W)

        if max_x_rows > 0 and x.shape[0] > max_x_rows:
            keep = torch.randperm(x.shape[0])[:max_x_rows]
            x = x[keep]

        return x, W


def _prepare_cpu_sample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        return tensor.to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


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


def _fetch_batch_cpu(
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
        return _prepare_cpu_sample_tensor(x), _prepare_cpu_sample_tensor(W)

    if dataset_iter is None:
        raise RuntimeError("dataset iterator is required for sharded mode")

    return _next_valid_sample(
        dataset_iter=dataset_iter,
        max_x_rows=max_x_rows,
        logger=logger,
    )


def _fetch_batch(
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_cpu, W_cpu = _fetch_batch_cpu(
        rank=rank,
        device=device,
        dataset_iter=dataset_iter,
        use_broadcast=use_broadcast,
        max_x_rows=max_x_rows,
        logger=logger,
    )
    return (
        x_cpu.to(device=device, non_blocking=True),
        W_cpu.to(device=device, non_blocking=True),
    )


def _fetch_source_samples(
    *,
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
    num_samples: int,
    min_d_in: int = 0,
    min_d_out: int = 0,
) -> list[SourceSample]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    selected: list[SourceSample] = []
    fallback_sample: SourceSample | None = None
    attempts = 0
    max_attempts = max(int(num_samples) * 8, int(num_samples))

    while len(selected) < num_samples and attempts < max_attempts:
        x_cpu, W_cpu = _fetch_batch_cpu(
            rank=rank,
            device=device,
            dataset_iter=dataset_iter,
            use_broadcast=use_broadcast,
            max_x_rows=max_x_rows,
            logger=logger,
        )
        attempts += 1

        if fallback_sample is None:
            fallback_sample = (x_cpu, W_cpu)

        if min_d_in > 0 and int(W_cpu.shape[0]) < int(min_d_in):
            continue
        if min_d_out > 0 and int(W_cpu.shape[1]) < int(min_d_out):
            continue

        selected.append((x_cpu, W_cpu))

    if not selected:
        if fallback_sample is None:
            raise RuntimeError("failed to fetch any source samples")
        selected.append(fallback_sample)

    return selected


def _sample_synthetic_layer(
    *,
    device: torch.device,
    n_rows: int,
    d_in: int,
    d_out: int,
    x_std: float,
    w_std: float,
    max_x_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_n_rows = int(n_rows)
    if max_x_rows > 0:
        effective_n_rows = min(effective_n_rows, int(max_x_rows))
    x = torch.randn((effective_n_rows, int(d_in)), device=device, dtype=torch.float32) * float(x_std)
    W = torch.randn((int(d_in), int(d_out)), device=device, dtype=torch.float32) * float(w_std)
    return x, W


def _subsample_x_rows(x: torch.Tensor, target_rows: int) -> torch.Tensor:
    if target_rows <= 0:
        raise ValueError(f"target_rows must be > 0, got {target_rows}")
    if int(x.shape[0]) <= int(target_rows):
        return x
    keep = torch.randperm(int(x.shape[0]), device=x.device)[: int(target_rows)]
    return x[keep]


def _round_robin_source_indices(num_sources: int, batch_size: int, start_offset: int = 0) -> list[int]:
    if num_sources <= 0:
        raise ValueError(f"num_sources must be > 0, got {num_sources}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    start = int(start_offset) % int(num_sources)
    return [int((start + batch_idx) % num_sources) for batch_idx in range(batch_size)]


def _build_training_batch_from_source_samples(
    source_samples: Sequence[SourceSample],
    *,
    max_T_patches: int,
    max_d_out: int,
    patch_size: int,
    batch_size: int,
    start_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not source_samples:
        raise ValueError("source_samples must not be empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    common_x_rows = min(int(x.shape[0]) for x, _ in source_samples)
    normalized_sources = [
        (_subsample_x_rows(x, common_x_rows), W)
        for x, W in source_samples
    ]

    assignment = _round_robin_source_indices(
        num_sources=len(normalized_sources),
        batch_size=batch_size,
        start_offset=start_offset,
    )
    counts = [0] * len(normalized_sources)
    for source_idx in assignment:
        counts[source_idx] += 1

    source_batches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    expected_w_shape: tuple[int, int] | None = None
    expected_x_shape: tuple[int, int] | None = None
    for source_idx, count in enumerate(counts):
        if count <= 0:
            continue
        x_src, W_src = normalized_sources[source_idx]
        W_part, x_part = _slice_sample(
            W_src,
            x_src,
            max_T_patches=max_T_patches,
            max_d_out=max_d_out,
            patch_size=patch_size,
            batch_size=count,
        )
        w_shape = (int(W_part.shape[1]), int(W_part.shape[2]))
        x_shape = (int(x_part.shape[1]), int(x_part.shape[2]))
        if expected_w_shape is None:
            expected_w_shape = w_shape
            expected_x_shape = x_shape
        elif w_shape != expected_w_shape or x_shape != expected_x_shape:
            raise ValueError(
                "round-robin batch mixing requires all selected source samples to slice to the same shape, got "
                f"W={w_shape}/x={x_shape} vs expected W={expected_w_shape}/x={expected_x_shape}"
            )
        source_batches[source_idx] = (W_part, x_part)

    source_offsets = [0] * len(normalized_sources)
    ordered_W: list[torch.Tensor] = []
    ordered_x: list[torch.Tensor] = []
    for source_idx in assignment:
        W_part, x_part = source_batches[source_idx]
        cursor = source_offsets[source_idx]
        ordered_W.append(W_part[cursor: cursor + 1])
        ordered_x.append(x_part[cursor: cursor + 1])
        source_offsets[source_idx] += 1

    return torch.cat(ordered_W, dim=0), torch.cat(ordered_x, dim=0)


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
def _tensor_debug_stats(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"is_none": True}

    detached = tensor.detach()
    flat = detached.reshape(-1)
    numel = int(flat.numel())
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    payload: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": numel,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    if finite_count > 0:
        finite_values = flat[finite_mask]
        payload["min"] = float(finite_values.min().item())
        payload["max"] = float(finite_values.max().item())
        payload["abs_max"] = float(finite_values.abs().max().item())
        payload["mean"] = float(finite_values.mean().item())
        payload["std"] = float(finite_values.std(unbiased=False).item())
    return payload


def _scalar_debug_value(tensor: torch.Tensor) -> float | str:
    value = tensor.detach()
    if value.numel() != 1:
        return f"<non_scalar shape={tuple(value.shape)}>"
    scalar = value.item()
    return float(scalar) if isinstance(scalar, (int, float)) else str(scalar)


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


def _find_latest_resume_state_checkpoint(state_dir: Path) -> Path | None:
    if not state_dir.exists() or not state_dir.is_dir():
        return None
    step_paths = sorted(path for path in state_dir.glob("step_*.pt") if path.is_file())
    if step_paths:
        return step_paths[-1]
    latest_path = state_dir / "latest.pt"
    if latest_path.is_file():
        return latest_path
    return None


def _load_training_state_from_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    path: Path,
    logger: logging.Logger,
) -> int:
    logger.info("Loading training state from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(ckpt["model_state"], strict=True)

    optimizer_state = ckpt.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = ckpt.get("scheduler_state")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = ckpt.get("scaler_state")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    step = int(ckpt.get("step", 0) or 0)
    stage = ckpt.get("stage", "?")
    logger.info("Training state loaded successfully (step=%s stage=%s)", step, stage)
    return step


def _get_encoder_conditioning_alpha_values(model: nn.Module) -> list[float]:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    adapters = getattr(target, "encoder_conditioning_adapters", None)
    if adapters is None:
        return []

    alpha_values: list[float] = []
    for adapter in adapters:
        alpha = getattr(adapter, "alpha", None)
        if alpha is None or not torch.is_tensor(alpha) or alpha.numel() == 0:
            continue
        alpha_values.append(float(alpha.detach().reshape(-1)[0].item()))
    return alpha_values


def _get_patch_tokenizer_block_alpha_stats(model: nn.Module) -> list[dict[str, float]]:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    patch_tokenizer = getattr(target, "patch_tokenizer", None)
    blocks = getattr(patch_tokenizer, "blocks", None)
    if blocks is None:
        return []

    out: list[dict[str, float]] = []
    for block in blocks:
        alpha = getattr(block, "alpha", None)
        if alpha is None or not torch.is_tensor(alpha):
            continue
        alpha_flat = alpha.detach().reshape(-1).to(dtype=torch.float32)
        if alpha_flat.numel() == 0:
            continue
        out.append(
            {
                "mean": float(alpha_flat.mean().item()),
                "abs_mean": float(alpha_flat.abs().mean().item()),
                "max_abs": float(alpha_flat.abs().max().item()),
            }
        )
    return out


def _get_patch_latent_variance_stats(
    model: nn.Module,
    W: torch.Tensor,
    X: torch.Tensor,
) -> dict[str, float] | None:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    forward_debug = getattr(target, "forward_debug", None)
    if not callable(forward_debug):
        return None

    was_training = bool(target.training)
    try:
        target.eval()
        with torch.no_grad():
            outputs = forward_debug(W, X)
        if not outputs or not isinstance(outputs[-1], dict):
            return None
        debug_info = outputs[-1]
        dist_var_by_patch = debug_info.get("dist_var_by_patch")
        if not torch.is_tensor(dist_var_by_patch):
            return None
        dist_var_flat = dist_var_by_patch.detach().to(dtype=torch.float32).flatten(start_dim=2)
        if dist_var_flat.numel() == 0:
            return None
        patch_var = dist_var_flat.var(dim=-1, unbiased=False)
        return {
            "mean": float(patch_var.mean().item()),
            "std": float(patch_var.std(unbiased=False).item()),
            "min": float(patch_var.min().item()),
            "max": float(patch_var.max().item()),
        }
    finally:
        target.train(was_training)


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
    if synthetic_x_std <= 0.0:
        raise ValueError(f"train.synthetic_layer_source.x_std must be > 0, got {synthetic_x_std}")
    if synthetic_w_std <= 0.0:
        raise ValueError(f"train.synthetic_layer_source.w_std must be > 0, got {synthetic_w_std}")
    if synthetic_layer_enabled:
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

    failed = False
    try:
        collector: Any | None = None
        dataset: Any | None = None
        dataset_iter: Iterator[Any] | None = None

        with contextlib.ExitStack() as stack:
            if not synthetic_layer_enabled:
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
                    "Model params: total=%s trainable=%s frozen=%s variant=%s",
                    f"{total_params:,}",
                    f"{trainable_params:,}",
                    f"{frozen_params:,}",
                    getattr(model_cfg, "variant", "full"),
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
            scheduler = _build_scheduler(optimizer=optimizer, cfg=cfg)
            comet_tracker = CometTracker(cfg=cfg, logger=logger, rank=rank)
            if rank == 0 and comet_tracker is not None and comet_tracker.experiment is not None:
                try:
                    comet_tracker.experiment.log_parameters(
                        {
                            "model.param_count.total": int(total_params),
                            "model.param_count.trainable": int(trainable_params),
                            "model.param_count.frozen": int(frozen_params),
                        }
                    )
                except Exception as exc:
                    logger.warning("Comet parameter-count log failed: %s", exc)

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
            default_resume_state_dir = (
                Path(str(cfg.train.get("checkpoint_dir", "./checkpoints/weight_quantile_vae")))
                / f"stage_{stage_num}"
                / "resume_state"
            )
            resume_state_dir = Path(str(resume_state_cfg.get("dir", str(default_resume_state_dir))))

            amp_enabled, amp_dtype = _resolve_amp(cfg=cfg, device=device)
            scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
            resume_checkpoint = str(cfg.train.get("resume_checkpoint", "")).strip()
            resumed_training_step = 0
            resume_state_path: Path | None = None
            if resume_state_enabled and resume_state_auto_resume:
                resume_state_path = _find_latest_resume_state_checkpoint(resume_state_dir)
            if resume_state_path is not None:
                resumed_training_step = _load_training_state_from_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    path=resume_state_path,
                    logger=logger,
                )
            elif resume_checkpoint:
                _load_model_weights_from_checkpoint(model, resume_checkpoint, logger)
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
            behavioral_coef = float(cfg.train.get("behavioral_coef", 1.0))
            structural_coef = float(cfg.train.get("structural_coef", 0.5))
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
            min_mixed_source_d_in = int(curriculum_max_T * patch_size_for_slice) if batch_source_mixing_active else 0
            min_mixed_source_d_out = int(curriculum_max_d_out) if batch_source_mixing_active else 0
            if rank == 0:
                logger.info(
                    "Curriculum slicing: stage=%s max_T_patches=%s max_d_out=%s patch_size=%s slice_batch_size=%s",
                    stage_num, curriculum_max_T, curriculum_max_d_out, patch_size_for_slice, slice_batch_size,
                )
                if batch_source_mixing_enabled:
                    logger.info(
                        "Batch source mixing: enabled=%s strategy=%s requested_source_samples=%s "
                        "min_source_shape_for_full_mixing=(d_in>=%s,d_out>=%s) collector_pressure_vs_single~%sx",
                        batch_source_mixing_active,
                        batch_source_mixing_strategy,
                        requested_source_samples_per_refresh,
                        min_mixed_source_d_in,
                        min_mixed_source_d_out,
                        requested_source_samples_per_refresh,
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
                        "Resume-state checkpointing: dir=%s save_every=%s auto_resume=%s",
                        resume_state_dir,
                        resume_state_save_every,
                        resume_state_auto_resume,
                    )
                if kl_schedule_enabled:
                    logger.info(
                        "BigVAE latent mode: %s (use_latent_sampling=%s, kl_beta_target=%s, "
                        "kl_schedule=start@%.6f warmup=%s ramp=%s)",
                        "VAE" if use_latent_sampling else "AE",
                        use_latent_sampling,
                        kl_beta,
                        kl_schedule_start_beta,
                        kl_schedule_warmup_steps,
                        kl_schedule_ramp_steps,
                    )
                else:
                    logger.info(
                        "BigVAE latent mode: %s (use_latent_sampling=%s, kl_beta=%s)",
                        "VAE" if use_latent_sampling else "AE",
                        use_latent_sampling,
                        kl_beta,
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
            structural_window = 0.0
            kl_window = 0.0
            struct_dir_window = 0.0
            struct_scale_window = 0.0
            struct_rec_window = 0.0
            struct_rel_window = 0.0
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

            current_source_samples: list[SourceSample] | None = None
            current_source_round_robin_offset = 0
            fixed_batch_x: torch.Tensor | None = None
            fixed_batch_W: torch.Tensor | None = None
            direction_pre_norm_stats_latest: dict[str, Any] | None = None
            source_mixing_shortfall_logged = False

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
                model.train()
                optimizer.zero_grad(set_to_none=True)

                if (
                    rank == 0
                    and collector is not None
                    and dataset is not None
                    and not collector.is_async_mode
                    and (not fixed_training_batch_enabled or fixed_batch_x is None or fixed_batch_W is None)
                ):
                    dataset.maybe_collect(step_idx)

                should_refresh_source_sample = False
                if fixed_training_batch_enabled:
                    should_refresh_source_sample = (
                        (fixed_batch_x is None or fixed_batch_W is None)
                        and not current_source_samples
                    )
                else:
                    should_refresh_source_sample = current_source_samples is None or step_idx % steps_per_sample == 0

                if should_refresh_source_sample:
                    if synthetic_layer_enabled:
                        synthetic_source_device = torch.device("cpu") if batch_source_mixing_active else device
                        current_source_samples = [
                            _sample_synthetic_layer(
                                device=synthetic_source_device,
                                n_rows=synthetic_n_rows,
                                d_in=synthetic_d_in,
                                d_out=synthetic_d_out,
                                x_std=synthetic_x_std,
                                w_std=synthetic_w_std,
                                max_x_rows=max_x_rows,
                            )
                            for _ in range(requested_source_samples_per_refresh)
                        ]
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
                                min_d_in=min_mixed_source_d_in,
                                min_d_out=min_mixed_source_d_out,
                            )
                        else:
                            current_x, current_W = _fetch_batch(
                                rank=rank,
                                device=device,
                                dataset_iter=dataset_iter,
                                use_broadcast=use_broadcast,
                                max_x_rows=max_x_rows,
                                logger=logger,
                            )
                            current_source_samples = [(current_x, current_W)]
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
                structural_acc = 0.0
                kl_acc = 0.0
                struct_dir_acc = 0.0
                struct_scale_acc = 0.0
                struct_rec_acc = 0.0
                struct_rel_acc = 0.0
                step_is_finite = True
                step_invalid_reason: str | None = None

                for micro_idx in range(grad_accum_steps):
                    sync_grad = micro_idx == grad_accum_steps - 1

                    if fixed_training_batch_enabled:
                        if fixed_batch_x is None or fixed_batch_W is None:
                            if not current_source_samples:
                                raise RuntimeError("fixed training batch capture requires a loaded source sample")
                            fixed_batch_W, fixed_batch_x = _build_training_batch_from_source_samples(
                                current_source_samples,
                                max_T_patches=curriculum_max_T,
                                max_d_out=curriculum_max_d_out,
                                patch_size=patch_size_for_slice,
                                batch_size=slice_batch_size,
                                start_offset=current_source_round_robin_offset,
                            )
                            fixed_batch_W = fixed_batch_W.to(device=device, non_blocking=True)
                            fixed_batch_x = fixed_batch_x.to(device=device, non_blocking=True)
                            current_source_samples = None
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
                        W_s, x_s = fixed_batch_W, fixed_batch_x
                    else:
                        if not current_source_samples:
                            raise RuntimeError("training step requires a loaded source sample")
                        W_s, x_s = _build_training_batch_from_source_samples(
                            current_source_samples,
                            max_T_patches=curriculum_max_T,
                            max_d_out=curriculum_max_d_out,
                            patch_size=patch_size_for_slice,
                            batch_size=slice_batch_size,
                            start_offset=current_source_round_robin_offset,
                        )
                        W_s = W_s.to(device=device, non_blocking=True)
                        x_s = x_s.to(device=device, non_blocking=True)
                        current_source_round_robin_offset = (
                            current_source_round_robin_offset + slice_batch_size
                        ) % len(current_source_samples)

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
                                    return_direction_pre_norms=True,
                                )
                            else:
                                W_hat, mu, logvar, pred_dirs = model(W_s, x_s)
                            behavioral_loss = WeightQuantileVAE.operator_recon_loss(x_s, W_s, W_hat)
                            structural_loss, struct_details = WeightQuantileVAE.patch_structure_loss(
                                W_s, W_hat, patch_size=patch_size_for_slice,
                                gamma=struct_gamma,
                                lambda_dir=struct_lambda_dir,
                                lambda_scale=struct_lambda_scale,
                                lambda_rec=struct_lambda_rec,
                                lambda_rel=struct_lambda_rel,
                                huber_delta=struct_huber_delta,
                                pred_dirs=pred_dirs,
                            )
                            if use_latent_sampling:
                                kl_loss = WeightQuantileVAE.kl_loss(mu, logvar)
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
                                    "structural": _scalar_debug_value(structural_loss),
                                    "kl": _scalar_debug_value(kl_loss),
                                    "kl_beta": float(current_kl_beta),
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

                scheduler.step()

                step_loss = loss_acc / grad_accum_steps
                step_behavioral = behavioral_acc / grad_accum_steps
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
                structural_window += float(stats[2].item())
                kl_window += float(stats[3].item())
                struct_dir_window += float(stats[4].item())
                struct_scale_window += float(stats[5].item())
                struct_rec_window += float(stats[6].item())
                struct_rel_window += float(stats[7].item())
                window_steps += 1

                if rank == 0 and global_step % log_every == 0:
                    dt = max(1e-6, time.time() - t0)
                    avg_loss = loss_window / max(1, window_steps)
                    avg_behavioral = behavioral_window / max(1, window_steps)
                    avg_structural = structural_window / max(1, window_steps)
                    avg_kl = kl_window / max(1, window_steps)
                    avg_struct_dir = struct_dir_window / max(1, window_steps)
                    avg_struct_scale = struct_scale_window / max(1, window_steps)
                    avg_struct_rec = struct_rec_window / max(1, window_steps)
                    avg_struct_rel = struct_rel_window / max(1, window_steps)
                    lr = float(optimizer.param_groups[0]["lr"])
                    speed = window_steps / dt

                    cache_metric = dataset.cache_size() if dataset is not None else 0
                    encoder_alpha_values = _get_encoder_conditioning_alpha_values(model)
                    patch_tokenizer_alpha_stats = _get_patch_tokenizer_block_alpha_stats(model)
                    patch_latent_variance_stats = _get_patch_latent_variance_stats(model, W_s[:1], x_s[:1])
                    logger.info(
                        "step=%s/%s loss=%.6f behav=%.6f struct=%.6f "
                        "s_dir=%.6f s_scl=%.6f s_rec=%.6f s_rel=%.6f "
                        "kl=%.6f kl_beta=%.6f lr=%.6e steps/s=%.2f cache=%s",
                        global_step,
                        max_steps,
                        avg_loss,
                        avg_behavioral,
                        avg_structural,
                        avg_struct_dir,
                        avg_struct_scale,
                        avg_struct_rec,
                        avg_struct_rel,
                        avg_kl,
                        current_kl_beta,
                        lr,
                        speed,
                        cache_metric,
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
                    if comet_tracker is not None and comet_tracker.enabled:
                        comet_metrics: dict[str, float] = {
                            "train/loss": float(avg_loss),
                            "train/behavioral_loss": float(avg_behavioral),
                            "train/structural_loss": float(avg_structural),
                            "train/kl_loss": float(avg_kl),
                            "train/struct_dir": float(avg_struct_dir),
                            "train/struct_scale": float(avg_struct_scale),
                            "train/struct_rec": float(avg_struct_rec),
                            "train/struct_rel": float(avg_struct_rel),
                            "train/lr": float(lr),
                            "train/kl_beta": float(current_kl_beta),
                            "train/steps_per_sec": float(speed),
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
                    struct_dir_window = 0.0
                    struct_scale_window = 0.0
                    struct_rec_window = 0.0
                    struct_rel_window = 0.0
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
