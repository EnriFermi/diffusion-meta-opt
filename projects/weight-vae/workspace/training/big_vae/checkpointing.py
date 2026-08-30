from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
import numpy as np
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
    resolve_cuda_physical_identity,
    resolve_device as runtime_resolve_device,
    resolve_world_size as runtime_resolve_world_size,
    seed_everything as runtime_seed_everything,
    set_speed_optimizations as runtime_set_speed_optimizations,
)


_VAE_POSTERIOR_HEAD_PREFIXES = ("to_mu.", "to_logvar.", "vamp_prior_base")

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



def _unwrap_model_for_state_io(model: torch.nn.Module) -> torch.nn.Module:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if isinstance(compiled_target, torch.nn.Module):
        target = compiled_target
    return target



def _normalize_model_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        key = _strip_ddp_prefix(str(raw_key))
        existing = normalized.get(key)
        if existing is not None and existing is not value:
            raise ValueError(
                f"State dict key collision after normalization: raw_key={raw_key!r} normalized_key={key!r}"
            )
        normalized[key] = value
    return normalized



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

    model_to_save = _unwrap_model_for_state_io(model)
    payload = {
        "step": step_idx,
        "stage": stage,
        "model_state": model_to_save.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    step_path = checkpoint_dir / f"step_{step_idx:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    _save_checkpoint_payload(payload, step_path)
    step_path.chmod(0o444)
    if bool(cfg.train.get("checkpoint_latest_copy", True)):
        _save_checkpoint_payload(payload, latest_path)
    logger.info("Model checkpoint saved: %s", step_path)



def _save_resume_state_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: GradScaler,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
    stage: int,
    state_dir: Path,
    rank: int = 0,
    world_size: int = 1,
    active_device: torch.device | None = None,
) -> None:
    model_to_save = _unwrap_model_for_state_io(model)
    operator_enabled = bool(cfg.train.get("operator_bank", {}).get("enabled", False))
    stream_contract = (
        _operator_stream_contract(cfg, rank=rank, world_size=world_size) if operator_enabled else None
    )
    parameter_name_by_identity = {
        id(parameter): str(name) for name, parameter in model_to_save.named_parameters()
    }
    optimizer_param_names: list[list[str]] = []
    for group_idx, group in enumerate(optimizer.param_groups):
        names: list[str] = []
        for parameter in group.get("params", []):
            name = parameter_name_by_identity.get(id(parameter))
            if name is None:
                raise RuntimeError(
                    "optimizer parameter is absent from the unwrapped model name inventory: "
                    f"group={group_idx} shape={tuple(parameter.shape)}"
                )
            names.append(name)
        optimizer_param_names.append(names)
    payload = {
        "step": step_idx,
        "stage": stage,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_param_names": optimizer_param_names,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "rng_state": _capture_training_rng_state(
            active_cuda_device=active_device,
            data_stream={
                "exactly_restorable": operator_enabled,
                "kind": "operator_bank_committed_step_cursor"
                if operator_enabled
                else "generic_iterable",
                "committed_training_step": int(step_idx),
                "logical_sample_index": (
                    int(cfg.train.get("operator_bank", {}).get("logical_index_offset", 0))
                    + int(step_idx)
                    * max(1, int(cfg.train.get("grad_accum_steps", 1)))
                    * max(1, int(cfg.train.get("slice_batch_size", 1)))
                )
                if operator_enabled
                else None,
                "reason": "step_addressed_sampler_regenerates_from_committed_optimizer_step"
                if operator_enabled
                else "multiprocess_iterable_prefetch_queue_state_is_not_serialized",
                "operator_stream_contract": stream_contract,
            }
        ),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    state_path = state_dir / f"step_{step_idx:07d}.pt"
    _save_checkpoint_payload(payload, state_path)
    state_path.chmod(0o444)

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
    x_mask_s: torch.Tensor | None,
    d_in_mask_s: torch.Tensor | None,
    d_out_mask_s: torch.Tensor | None,
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
        "fixed_batch_x_mask": (
            x_mask_s.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous()
            if x_mask_s is not None
            else None
        ),
        "fixed_batch_d_in_mask": (
            d_in_mask_s.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous()
            if d_in_mask_s is not None
            else None
        ),
        "fixed_batch_d_out_mask": (
            d_out_mask_s.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous()
            if d_out_mask_s is not None
            else None
        ),
        "meta": {
            "capture_step": int(global_step),
            "stage": int(stage),
            "patch_size": int(patch_size),
            "max_T_patches": int(max_T_patches),
            "max_d_out": int(max_d_out),
            "slice_batch_size": int(slice_batch_size),
            "W_shape": list(W_s.shape),
            "x_shape": list(x_s.shape),
            "x_mask_shape": list(x_mask_s.shape) if x_mask_s is not None else None,
            "d_in_mask_shape": list(d_in_mask_s.shape) if d_in_mask_s is not None else None,
            "d_out_mask_shape": list(d_out_mask_s.shape) if d_out_mask_s is not None else None,
            "data_seed": int(cfg.data.get("seed", 42)),
        },
    }
    torch.save(payload, dump_path)
    logger.info("Fixed training batch dump saved: %s", dump_path)
    return dump_path



def _load_frozen_training_batch_dump(
    dump_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(dump_path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen training batch dump not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Frozen training batch dump must contain a dict payload, got {type(payload)!r}: {path}")

    W_s = payload.get("fixed_batch_W", payload.get("W_s"))
    x_s = payload.get("fixed_batch_x", payload.get("x_s"))
    if not torch.is_tensor(W_s) or not torch.is_tensor(x_s):
        raise KeyError(
            f"Frozen training batch dump must contain tensor keys 'fixed_batch_W'/'fixed_batch_x' or 'W_s'/'x_s': {path}"
        )

    W_s = W_s.detach().to(device=device, dtype=torch.float32, copy=True).contiguous()
    x_s = x_s.detach().to(device=device, dtype=torch.float32, copy=True).contiguous()
    if W_s.ndim != 3 or x_s.ndim != 3:
        raise ValueError(
            "Frozen training batch tensors must be rank-3 batched tensors, got "
            f"W={tuple(W_s.shape)} x={tuple(x_s.shape)} from {path}"
        )
    if int(W_s.shape[0]) != int(x_s.shape[0]) or int(W_s.shape[1]) != int(x_s.shape[2]):
        raise ValueError(
            "Frozen training batch tensor shapes are inconsistent, got "
            f"W={tuple(W_s.shape)} x={tuple(x_s.shape)} from {path}"
        )

    raw_meta = payload.get("meta", {})
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {"raw_meta": raw_meta}
    meta.setdefault("loaded_W_shape", list(W_s.shape))
    meta.setdefault("loaded_x_shape", list(x_s.shape))
    meta.setdefault("dump_path", str(path))
    return W_s, x_s, meta



def _model_uses_latent_sampling(model: torch.nn.Module) -> bool:
    cfg = getattr(model, "cfg", None)
    big_vae_cfg = getattr(cfg, "big_vae", None)
    return bool(getattr(big_vae_cfg, "use_latent_sampling", False))



def _model_uses_encoder_mu_head(model: torch.nn.Module) -> bool:
    cfg = getattr(model, "cfg", None)
    big_vae_cfg = getattr(cfg, "big_vae", None)
    return bool(getattr(big_vae_cfg, "use_encoder_mu_head", False))



def _is_vae_posterior_head_key(key: str) -> bool:
    return any(str(key).startswith(prefix) for prefix in _VAE_POSTERIOR_HEAD_PREFIXES)



def _is_allowed_optional_latent_head_key(*, model: torch.nn.Module, key: str) -> bool:
    key_str = str(key)
    if key_str.startswith("to_mu."):
        return _model_uses_latent_sampling(model) or _model_uses_encoder_mu_head(model)
    if key_str.startswith("to_logvar."):
        return _model_uses_latent_sampling(model)
    if key_str.startswith("vamp_prior_base"):
        return True
    return False



def _load_model_state_allowing_vae_head_migration(
    *,
    target: torch.nn.Module,
    state_dict: dict[str, Any],
    logger: logging.Logger,
    source: str | Path,
) -> bool:
    target_state = target.state_dict()
    missing_keys = sorted(key for key in target_state.keys() if key not in state_dict)
    unexpected_keys = sorted(key for key in state_dict.keys() if key not in target_state)

    if not missing_keys and not unexpected_keys:
        target.load_state_dict(state_dict, strict=True)
        return False

    allowed_missing = sorted(key for key in missing_keys if _is_allowed_optional_latent_head_key(model=target, key=key))
    allowed_unexpected = sorted(
        key for key in unexpected_keys if _is_allowed_optional_latent_head_key(model=target, key=key)
    )
    should_allow_latent_head_migration = (
        bool(allowed_missing or allowed_unexpected)
        and allowed_missing == missing_keys
        and allowed_unexpected == unexpected_keys
    )
    if not should_allow_latent_head_migration:
        target.load_state_dict(state_dict, strict=True)
        return False

    incompatible = target.load_state_dict(state_dict, strict=False)
    unexpected_after_load = list(getattr(incompatible, "unexpected_keys", []))
    missing_after_load = sorted(getattr(incompatible, "missing_keys", []))
    disallowed_missing = [key for key in missing_after_load if not _is_allowed_optional_latent_head_key(model=target, key=key)]
    disallowed_unexpected = [
        key for key in unexpected_after_load if not _is_allowed_optional_latent_head_key(model=target, key=key)
    ]
    if disallowed_unexpected or disallowed_missing:
        raise RuntimeError(
            "Unexpected checkpoint incompatibility while migrating optional latent heads: "
            f"source={source} missing={missing_after_load} unexpected={unexpected_after_load}"
        )

    logger.warning(
        "Loaded BigVAE checkpoint with optional latent-head migration. "
        "source=%s missing_head_keys=%s dropped_head_keys=%s",
        source,
        missing_after_load,
        unexpected_after_load,
    )
    return True



def _load_model_weights_from_checkpoint(
    model: torch.nn.Module,
    path: str,
    logger: logging.Logger,
) -> bool:
    logger.info("Loading model weights from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _normalize_model_state_dict_keys(ckpt["model_state"])
    target = _unwrap_model_for_state_io(model)
    migrated_ae_to_vae = _load_model_state_allowing_vae_head_migration(
        target=target,
        state_dict=state_dict,
        logger=logger,
        source=path,
    )
    logger.info("Model weights loaded successfully (step=%s)", ckpt.get("step", "?"))
    return migrated_ae_to_vae



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



def _resume_state_load_policy(resume_state_cfg: dict[str, Any] | DictConfig) -> dict[str, bool]:
    return {
        "load_model_state": bool(resume_state_cfg.get("load_model_state", True)),
        "load_optimizer_state": bool(resume_state_cfg.get("load_optimizer_state", True)),
        "load_scheduler_state": bool(resume_state_cfg.get("load_scheduler_state", True)),
        "load_scaler_state": bool(resume_state_cfg.get("load_scaler_state", True)),
        "load_rng_state": bool(resume_state_cfg.get("load_rng_state", True)),
        "load_step": bool(resume_state_cfg.get("load_step", True)),
    }


def _operator_stream_contract(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    operator = cfg.train.get("operator_bank", {})
    if not isinstance(operator, (dict, DictConfig)) or not bool(operator.get("enabled", False)):
        raise RuntimeError("operator stream contract requires train.operator_bank.enabled=true")
    payload = {
        "schema": "operator_bank_locality_stream_v1",
        "planning_algorithm": "global_stratum_sequence_exact_within_stratum_grouped_v1",
        "pair_manifest_sha256": str(operator.get("pair_manifest_sha256", "")),
        "effective_seed": int(cfg.data.get("seed", 42)) + int(rank),
        "rank": int(rank),
        "world_size": int(world_size),
        "repeat": bool(operator.get("repeat", True)),
        "permutation_views": bool(operator.get("permutation_views", True)),
        "canonical_probability": float(operator.get("canonical_probability", 1.0 / 6.0)),
        "max_active_strata": int(operator.get("max_active_strata", 256)),
        "max_active_bundle_bytes": int(operator.get("max_active_bundle_bytes", 1536 * 1024 * 1024)),
        "slice_batch_size": int(cfg.train.get("slice_batch_size", 1)),
        "grad_accum_steps": int(cfg.train.get("grad_accum_steps", 1)),
    }
    overfit = operator.get("two_operator_overfit", {})
    if isinstance(overfit, (dict, DictConfig)) and bool(overfit.get("enabled", False)):
        payload["planning_algorithm"] = "two_full_operator_interleaved_exact_cycle_v1"
        payload["two_operator_overfit"] = {
            "schema": "two_full_operator_fixed_cycle_v1",
            "selected_operators": [
                {
                    "checkpoint_sha256": str(row["checkpoint_sha256"]),
                    "layer_key": str(row["layer_key"]),
                }
                for row in overfit.get("selected_operators", [])
            ],
        }
    operator_set = operator.get("operator_set_overfit", {})
    if isinstance(operator_set, (dict, DictConfig)) and bool(operator_set.get("enabled", False)):
        payload["planning_algorithm"] = "v9_exact64_round_robin_62_of_63_v1"
        payload["operator_set_overfit"] = {
            "schema": "v9_exact64_operator_set_v1",
            "selected_operators": [
                {
                    "checkpoint_sha256": str(row["checkpoint_sha256"]),
                    "layer_key": str(row["layer_key"]),
                }
                for row in operator_set.get("selected_operators", [])
            ],
            "training_round_count": 62,
            "heldout_derangement_round_index_zero_based": 62,
            "heldout_derangement_round_number_one_based": 63,
            "selection_sha256": str(operator_set.get("selection_sha256", "")),
            "schedule_sha256": str(operator_set.get("schedule_sha256", "")),
        }
    logical_index_offset = int(operator.get("logical_index_offset", 0))
    if logical_index_offset != 0:
        payload["logical_index_offset"] = logical_index_offset
    if len(payload["pair_manifest_sha256"]) != 64:
        raise RuntimeError("operator stream contract requires a SHA-bound pair manifest")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "operator_bank_stream_contract_v1",
        "sha256": hashlib.sha256(body).hexdigest(),
        "payload": payload,
    }



def _load_training_state_from_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: GradScaler,
    path: Path,
    logger: logging.Logger,
    load_model_state: bool = True,
    load_optimizer_state: bool = True,
    load_scheduler_state: bool = True,
    load_scaler_state: bool = True,
    load_rng_state: bool = True,
    load_step: bool = True,
    expected_data_stream_contract: dict[str, Any] | None = None,
    allowed_data_stream_transition: dict[str, Any] | DictConfig | None = None,
    active_device: torch.device | None = None,
) -> int:
    logger.info("Loading training state from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if expected_data_stream_contract is not None:
        rng_state_for_contract = ckpt.get("rng_state")
        saved_contract = (
            rng_state_for_contract.get("data_stream", {}).get("operator_stream_contract")
            if isinstance(rng_state_for_contract, dict)
            else None
        )
        if not isinstance(saved_contract, dict):
            raise RuntimeError(
                "operator-bank resume checkpoint predates the exact locality stream contract; refusing unsafe resume"
            )
        if saved_contract != expected_data_stream_contract:
            transition = allowed_data_stream_transition or {}
            if not bool(transition.get("enabled", False)):
                raise RuntimeError(
                    "operator-bank resume stream contract mismatch: "
                    f"saved={saved_contract} expected={expected_data_stream_contract}"
                )
            saved_payload = dict(saved_contract.get("payload", {}))
            expected_payload = dict(expected_data_stream_contract.get("payload", {}))
            source_accum = int(transition.get("source_grad_accum_steps", -1))
            target_accum = int(transition.get("target_grad_accum_steps", -1))
            source_step = int(transition.get("source_step", -1))
            start_logical_index = int(transition.get("start_logical_index", -1))
            if int(ckpt.get("step", -1)) != source_step:
                raise RuntimeError("operator stream transition source step mismatch")
            if int(rng_state_for_contract.get("data_stream", {}).get("logical_sample_index", -1)) != start_logical_index:
                raise RuntimeError("operator stream transition logical cursor mismatch")
            if int(saved_payload.get("grad_accum_steps", -1)) != source_accum:
                raise RuntimeError("operator stream transition source accumulation mismatch")
            if int(expected_payload.get("grad_accum_steps", -1)) != target_accum:
                raise RuntimeError("operator stream transition target accumulation mismatch")
            expected_offset = start_logical_index - source_step * int(
                expected_payload.get("slice_batch_size", 0)
            ) * target_accum
            if int(expected_payload.get("logical_index_offset", 0)) != expected_offset:
                raise RuntimeError("operator stream transition offset does not preserve the saved cursor")
            for payload in (saved_payload, expected_payload):
                payload.pop("grad_accum_steps", None)
                payload.pop("logical_index_offset", None)
            if saved_payload != expected_payload:
                raise RuntimeError(
                    "operator stream transition changes fields other than accumulation/cursor offset: "
                    f"saved={saved_payload} expected={expected_payload}"
                )
            logger.info(
                "Validated intentional operator-stream accumulation transition: step=%s cursor=%s accum=%s->%s",
                source_step,
                start_logical_index,
                source_accum,
                target_accum,
            )
    migrated_ae_to_vae = False
    if load_model_state:
        target = _unwrap_model_for_state_io(model)
        state_dict = _normalize_model_state_dict_keys(ckpt["model_state"])
        migrated_ae_to_vae = _load_model_state_allowing_vae_head_migration(
            target=target,
            state_dict=state_dict,
            logger=logger,
            source=path,
        )

    if migrated_ae_to_vae and load_optimizer_state:
        raise RuntimeError(
            "Cannot load optimizer_state while migrating an AE checkpoint without posterior heads "
            "into use_latent_sampling=true BigVAE. Set train.resume_state.load_optimizer_state=false "
            "and resume model-only for AE -> VAE fine-tuning."
        )
    optimizer_state = ckpt.get("optimizer_state")
    if load_optimizer_state and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = ckpt.get("scheduler_state")
    if load_scheduler_state and scheduler_state is not None:
        if scheduler is None:
            raise RuntimeError("resume checkpoint contains scheduler state but current scheduler is disabled")
        scheduler.load_state_dict(scheduler_state)
    scaler_state = ckpt.get("scaler_state")
    if load_scaler_state and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if load_rng_state:
        rng_state = ckpt.get("rng_state")
        if rng_state is None:
            logger.warning("Resume checkpoint has no RNG state; continuation is not RNG-exact: %s", path)
        else:
            _restore_training_rng_state(rng_state, active_cuda_device=active_device)
            data_stream = rng_state.get("data_stream", {})
            if not bool(data_stream.get("exactly_restorable", False)):
                logger.warning(
                    "Process RNGs restored, but data-stream continuation is not exact: %s",
                    data_stream.get("reason", "unspecified"),
                )

    step = int(ckpt.get("step", 0) or 0) if load_step else 0
    stage = ckpt.get("stage", "?")
    logger.info(
        "Training state loaded successfully (path=%s step=%s stage=%s load_model_state=%s "
        "load_optimizer_state=%s load_scheduler_state=%s load_scaler_state=%s load_rng_state=%s load_step=%s)",
        path,
        step,
        stage,
        load_model_state,
        load_optimizer_state,
        load_scheduler_state,
        load_scaler_state,
        load_rng_state,
        load_step,
    )
    return step


def _active_cuda_rng_payload(device: torch.device | None) -> dict[str, Any] | None:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    logical_index = int(device.index or 0)
    logical_device = torch.device("cuda", logical_index)
    identity = resolve_cuda_physical_identity(logical_device)
    return {
        "schema": "active_cuda_rng_v1",
        "logical_device": str(logical_device),
        "physical_uuid": identity["physical_uuid"],
        "device_name": identity["physical_name"],
        "cuda_visible_devices": str(identity["cuda_visible_devices"] or ""),
        "state": torch.cuda.get_rng_state(logical_device),
    }


def _capture_training_rng_state(
    *,
    data_stream: dict[str, Any] | None = None,
    active_cuda_device: torch.device | None = None,
) -> dict[str, Any]:
    """Capture process RNGs stored in the rolling resume checkpoint.

    The explicit data-stream marker is important: a multiprocess DataLoader can
    have prefetched records beyond the last optimizer step, so its queue cursor
    is not recoverable from process RNG state alone.
    """

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_active": _active_cuda_rng_payload(active_cuda_device),
        "data_stream": data_stream or {
            "exactly_restorable": False,
            "reason": "multiprocess_iterable_prefetch_queue_state_is_not_serialized",
        },
    }


def _restore_training_rng_state(
    state: dict[str, Any], *, active_cuda_device: torch.device | None = None
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_payload = state.get("torch_cuda_active")
    if cuda_payload is not None:
        if not isinstance(cuda_payload, dict) or active_cuda_device is None:
            raise RuntimeError("active-device CUDA RNG payload cannot be restored")
        logical_device = torch.device(str(cuda_payload.get("logical_device", "")))
        if logical_device != active_cuda_device:
            raise RuntimeError("active-device CUDA RNG logical device differs at restore")
        identity = resolve_cuda_physical_identity(active_cuda_device)
        if identity["physical_uuid"] != cuda_payload.get("physical_uuid"):
            raise RuntimeError("active-device CUDA RNG physical UUID differs at restore")
        torch.cuda.set_rng_state(cuda_payload["state"], active_cuda_device)

__all__ = [
    '_strip_ddp_prefix',
    '_unwrap_model_for_state_io',
    '_normalize_model_state_dict_keys',
    '_save_checkpoint_payload',
    '_save_checkpoint',
    '_save_resume_state_checkpoint',
    '_maybe_dump_fixed_training_batch',
    '_load_frozen_training_batch_dump',
    '_model_uses_latent_sampling',
    '_model_uses_encoder_mu_head',
    '_is_vae_posterior_head_key',
    '_is_allowed_optional_latent_head_key',
    '_load_model_state_allowing_vae_head_migration',
    '_load_model_weights_from_checkpoint',
    '_find_latest_resume_state_checkpoint',
    '_resume_state_load_policy',
    '_operator_stream_contract',
    '_capture_training_rng_state',
    '_restore_training_rng_state',
    '_load_training_state_from_checkpoint',
]
