from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import logging
import math
import os
import statistics
import subprocess
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

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
from big_vae.datasets.operator_bank import BalancedOperatorBankMixer, operator_bank_data_pipeline
from big_vae.weightclip_benchmark.ae_candidate_config import (
    ae_source_implementation_seal,
    normalize_launcher_runtime_config,
)
from big_vae.weightclip_benchmark.manifests import sha256_file, write_json_immutable, write_records_immutable
from big_vae.weightclip_benchmark.metadata import redact_secrets
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

from training.big_vae.checkpointing import *
from training.big_vae.data import *
from training.big_vae.epsilon_fork import (
    apply_loaded_optimizer_epsilon_fork,
    validate_control_replay_metrics,
)
from training.big_vae.v6_causal_bundle import (
    evaluation_due as v6_causal_evaluation_due,
    prepare_model as prepare_v6_causal_model,
    transform_training_tensors as transform_v6_causal_training_tensors,
    validate_config as validate_v6_causal_config,
    validate_v8_step0_preflight,
)
from training.big_vae.grad_monitoring import *
from training.big_vae.presliced import _prepared_batch_prefetch_blockers
from training.big_vae.runtime import *
from training.big_vae.tracking import *
from training.weightclip_benchmark.gradient_noise_monitor import GradientNoiseMonitor


_BOUNDED_RUNTIME_PROFILE_SCHEMA = "weightclip_ae_runtime_v2"
_BOUNDED_RUNTIME_PROFILE_MAX_STEPS = 32
_BOUNDED_RUNTIME_PROFILE_PRODUCTION_STEPS = 500_000
_BOUNDED_RUNTIME_PROFILE_REQUIRED_WARMUP_STEPS = 4
_BOUNDED_RUNTIME_PROFILE_REQUIRED_MEASURED_STEPS = 28
_BOUNDED_RUNTIME_PROFILE_MODES = ("production_exact", "producer_stress")
_BOUNDED_RUNTIME_PROFILE_QUEUE_BY_MODE = {"production_exact": 24, "producer_stress": 4}
_BOUNDED_RUNTIME_PROFILE_REQUEST_STOP_BY_MODE = {"production_exact": 84, "producer_stress": 64}
_BOUNDED_RUNTIME_PROFILE_TARGET_PARAMETERS = 700_000_000
_BOUNDED_RUNTIME_PROFILE_PARAMETER_TOLERANCE = 0.01
_OPERATOR_SET_OVERFIT_ARCHITECTURES = frozenset(
    {
        "latent_feedback_perirms_qknorm_v4",
        "hybrid_nonlinear_content_posfilm_v9",
        "carrier_mean_content_posfilm_v9a",
        "orthogonal_complement_tied_posfilm_v10",
        "four_trunk_complement_v11",
    }
)


def _validate_operator_set_overfit_architecture(architecture_version: object) -> str:
    value = str(architecture_version)
    if value not in _OPERATOR_SET_OVERFIT_ARCHITECTURES:
        raise ValueError(
            "operator_set_overfit requires the exact64 V4, V9, V9-A, V10, or V11 architecture; "
            f"got {value!r}"
        )
    return value


def _enforce_v10_strict_fp32_backend(
    cfg: DictConfig,
    *,
    device: torch.device,
) -> dict[str, object] | None:
    if str(cfg.model.big_vae.architecture_version) not in {
        "orthogonal_complement_tied_posfilm_v10",
        "four_trunk_complement_v11",
    }:
        return None
    if "tf32" not in cfg.train or bool(cfg.train.get("tf32")):
        raise ValueError("V10/V11 requires explicit train.tf32=false")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else False
    if precision != "highest" or matmul_tf32 or cudnn_tf32:
        raise RuntimeError(
            "V10/V11 strict FP32 backend setup failed: "
            f"precision={precision!r} matmul_tf32={matmul_tf32} cudnn_tf32={cudnn_tf32}"
        )
    return {
        "device_type": device.type,
        "float32_matmul_precision": precision,
        "cuda_matmul_allow_tf32": matmul_tf32,
        "cudnn_allow_tf32": cudnn_tf32,
    }


def _profile_config_contract_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable profile config modulo launcher-created artifact paths.

    ``configure_per_run_artifacts`` expands the run-local logging/artifact tree
    after Hydra composition.  Those paths are the only permitted active-config
    mutation; every model/data/loss/optimizer/scheduler field remains exact.
    """

    normalized = normalize_launcher_runtime_config(value)
    redacted = redact_secrets(normalized)
    if not isinstance(redacted, dict):
        raise TypeError("bounded runtime profile normalized config must remain a mapping")
    return redacted


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_config_contract_diff(
    expected: Any,
    actual: Any,
    *,
    path: str = "",
) -> list[dict[str, Any]]:
    """Return a secret-safe structural diff for a failed profile parity gate."""

    def digest(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    if type(expected) is not type(actual):
        return [
            {
                "path": path or "<root>",
                "kind": "type_mismatch",
                "expected_type": type(expected).__name__,
                "actual_type": type(actual).__name__,
                "expected_value_sha256": digest(expected),
                "actual_value_sha256": digest(actual),
            }
        ]
    if isinstance(expected, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                rows.append(
                    {
                        "path": child_path,
                        "kind": "only_in_active",
                        "actual_type": type(actual[key]).__name__,
                        "actual_value_sha256": digest(actual[key]),
                    }
                )
            elif key not in actual:
                rows.append(
                    {
                        "path": child_path,
                        "kind": "only_in_resolved",
                        "expected_type": type(expected[key]).__name__,
                        "expected_value_sha256": digest(expected[key]),
                    }
                )
            else:
                rows.extend(
                    _profile_config_contract_diff(
                        expected[key],
                        actual[key],
                        path=child_path,
                    )
                )
        return rows
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [
                {
                    "path": path or "<root>",
                    "kind": "length_mismatch",
                    "expected_length": len(expected),
                    "actual_length": len(actual),
                }
            ]
        rows = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            rows.extend(
                _profile_config_contract_diff(
                    expected_item,
                    actual_item,
                    path=f"{path}[{index}]",
                )
            )
        return rows
    if expected != actual:
        return [
            {
                "path": path or "<root>",
                "kind": "value_mismatch",
                "value_type": type(expected).__name__,
                "expected_value_sha256": digest(expected),
                "actual_value_sha256": digest(actual),
            }
        ]
    return []


def _validate_profile_parameter_budget(trainable_parameters: int) -> dict[str, int | float]:
    target = _BOUNDED_RUNTIME_PROFILE_TARGET_PARAMETERS
    tolerance_parameters = int(target * _BOUNDED_RUNTIME_PROFILE_PARAMETER_TOLERANCE)
    delta = int(trainable_parameters) - target
    if abs(delta) > tolerance_parameters:
        raise RuntimeError(
            "bounded runtime profile trainable-parameter count is outside the approved 700M +/-1% budget: "
            f"actual={trainable_parameters} target={target} tolerance={tolerance_parameters}"
        )
    return {
        "target_trainable_parameters": target,
        "relative_tolerance": _BOUNDED_RUNTIME_PROFILE_PARAMETER_TOLERANCE,
        "absolute_tolerance_parameters": tolerance_parameters,
        "actual_trainable_parameters": int(trainable_parameters),
        "delta_parameters": delta,
        "relative_delta": float(delta / target),
    }


def _consumed_logical_range(indices: Sequence[int], *, expected_count: int) -> tuple[int, int]:
    if len(indices) != int(expected_count) or not indices:
        raise RuntimeError("bounded runtime profile consumed logical-index inventory is incomplete")
    start = int(indices[0])
    end = start + int(expected_count)
    if [int(value) for value in indices] != list(range(start, end)):
        raise RuntimeError(f"bounded runtime profile consumed non-contiguous logical indices: {list(indices)}")
    return start, end


def _resolve_profile_gpu_identity(
    device: torch.device,
    *,
    environ: Mapping[str, str] | None = None,
    properties: Any | None = None,
    nvml_index: int | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Compatibility wrapper around the shared Torch-to-NVML resolver."""

    return resolve_cuda_physical_identity(
        device,
        environ=environ,
        properties=properties,
        nvml_index=nvml_index,
        runner=runner,
    )


@dataclass(frozen=True, slots=True)
class _BoundedRuntimeProfileSpec:
    profile_mode: str
    warmup_steps: int
    measured_steps: int
    output_dir: Path
    checkpoint_dir: Path

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.measured_steps


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_bounded_runtime_profile(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
) -> _BoundedRuntimeProfileSpec | None:
    """Validate the unoverrideable safety envelope for the short AE profiler."""

    raw = cfg.train.get("bounded_runtime_profile", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, (dict, DictConfig)):
        raise TypeError("train.bounded_runtime_profile must be a mapping")
    if not bool(raw.get("enabled", False)):
        return None
    if str(raw.get("schema", "")) != _BOUNDED_RUNTIME_PROFILE_SCHEMA:
        raise RuntimeError("bounded runtime profile schema mismatch")
    profile_mode = str(raw.get("mode", ""))
    if profile_mode not in _BOUNDED_RUNTIME_PROFILE_MODES:
        raise RuntimeError(
            "bounded runtime profile mode must be one of "
            f"{list(_BOUNDED_RUNTIME_PROFILE_MODES)}, got {profile_mode!r}"
        )
    warmup_steps = int(raw.get("warmup_steps", 0))
    measured_steps = int(raw.get("measured_steps", 0))
    total_steps = warmup_steps + measured_steps
    if warmup_steps < 1 or measured_steps < 1:
        raise RuntimeError("bounded runtime profile requires at least one warmup and measured step")
    if total_steps > _BOUNDED_RUNTIME_PROFILE_MAX_STEPS:
        raise RuntimeError(
            f"bounded runtime profile hard cap is {_BOUNDED_RUNTIME_PROFILE_MAX_STEPS} optimizer steps, "
            f"requested {total_steps}"
        )
    if int(cfg.train.get("max_steps", 0)) != _BOUNDED_RUNTIME_PROFILE_PRODUCTION_STEPS:
        raise RuntimeError("bounded runtime profile must retain the production 500000-step scheduler horizon")
    if rank != 0 or world_size != 1 or bool(cfg.train.get("distributed", False)):
        raise RuntimeError("bounded runtime profile is single-process/single-GPU only")
    if str(cfg.train.get("device", "")) != "cuda:0" or int(cfg.train.get("num_gpus", 0)) != 1:
        raise RuntimeError("bounded runtime profile requires exactly cuda:0 and one GPU")
    operator = cfg.train.get("operator_bank", {})
    if not isinstance(operator, (dict, DictConfig)) or not bool(operator.get("enabled", False)):
        raise RuntimeError("bounded runtime profile requires the production operator-bank path")
    if not str(operator.get("pair_manifest_sha256", "")):
        raise RuntimeError("bounded runtime profile requires a hash-bound operator-bank pair manifest")
    if bool(cfg.train.get("offline_dataset", {}).get("enabled", False)):
        raise RuntimeError("bounded runtime profile forbids the legacy offline dataset path")
    if bool(cfg.train.get("preslicing", {}).get("enabled", False)):
        raise RuntimeError("bounded runtime profile forbids preslicing")
    if bool(cfg.train.get("synthetic_layer_source", {}).get("enabled", False)):
        raise RuntimeError("bounded runtime profile forbids synthetic data")
    if bool(cfg.train.get("fixed_training_batch", {}).get("enabled", False)):
        raise RuntimeError("bounded runtime profile forbids fixed-batch substitution")
    resume = cfg.train.get("resume_state", {})
    if not isinstance(resume, (dict, DictConfig)):
        raise TypeError("train.resume_state must be a mapping")
    if bool(resume.get("enabled", False)) or bool(resume.get("auto_resume", False)):
        raise RuntimeError("bounded runtime profile forbids resume and auto-resume")
    if str(cfg.train.get("resume_checkpoint", "") or "").strip():
        raise RuntimeError("bounded runtime profile forbids loading a resume checkpoint")
    prefetch = cfg.train.get("offline_batch_prefetch", {})
    expected_queue = _BOUNDED_RUNTIME_PROFILE_QUEUE_BY_MODE[profile_mode]
    if not isinstance(prefetch, (dict, DictConfig)) or int(prefetch.get("queue_size", -1)) != expected_queue:
        raise RuntimeError(
            f"bounded runtime profile mode={profile_mode} requires exact prefetch queue_size={expected_queue}"
        )
    telemetry = cfg.train.get("telemetry", {})
    if not isinstance(telemetry, (dict, DictConfig)):
        raise TypeError("train.telemetry must be a mapping")
    for tracker_name in ("comet", "wandb"):
        tracker = telemetry.get(tracker_name, {})
        if isinstance(tracker, (dict, DictConfig)) and bool(tracker.get("enabled", False)):
            raise RuntimeError(f"bounded runtime profile forbids external tracker {tracker_name}")
    output_text = str(raw.get("output_dir", "") or "").strip()
    if not output_text:
        raise RuntimeError("bounded runtime profile output_dir is required")
    output_dir = Path(output_text).expanduser()
    if not output_dir.is_absolute():
        raise RuntimeError("bounded runtime profile output_dir must be absolute")
    checkpoint_dir = Path(str(cfg.train.get("checkpoint_dir", ""))).expanduser()
    if not checkpoint_dir.is_absolute():
        raise RuntimeError("bounded runtime profile checkpoint_dir must be absolute")
    if not _path_is_within(checkpoint_dir, output_dir):
        raise RuntimeError("bounded runtime profile checkpoint_dir must be isolated inside its output_dir")
    return _BoundedRuntimeProfileSpec(
        profile_mode=profile_mode,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        output_dir=output_dir.resolve(),
        checkpoint_dir=checkpoint_dir.resolve(),
    )


def _profile_percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measured_producer_refill_ledger(
    measured_rows: Sequence[Mapping[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    if not measured_rows:
        raise RuntimeError("bounded runtime profile has no measured rows for producer-refill validation")
    cursors = [int(row["produced_logical_index"]) for row in measured_rows]
    monotonic_nondecreasing = all(
        current >= previous for previous, current in zip(cursors, cursors[1:])
    )
    if not monotonic_nondecreasing:
        raise RuntimeError(
            "bounded runtime profile producer cursor regressed inside the measured window"
        )
    advance = max(cursors) - min(cursors)
    if required and advance <= 0:
        raise RuntimeError(
            "bounded runtime profile measured window never observed producer refill; "
            "input-starvation evidence is not established"
        )
    return {
        "required_by_frozen_protocol": bool(required),
        "observed": advance > 0,
        "first_produced_logical_index": cursors[0],
        "last_produced_logical_index": cursors[-1],
        "min_produced_logical_index": min(cursors),
        "max_produced_logical_index": max(cursors),
        "advance_tiles": advance,
        "monotonic_nondecreasing": True,
    }


def _bounded_training_stop_step(
    max_steps: int,
    spec: _BoundedRuntimeProfileSpec | None,
    stop_after_step: int = 0,
) -> int:
    if spec is not None:
        return int(spec.total_steps)
    requested = int(stop_after_step)
    if requested == 0:
        return int(max_steps)
    if requested < 1 or requested > int(max_steps):
        raise ValueError(
            f"train.stop_after_step must be within [1, {int(max_steps)}], got {requested}"
        )
    return requested


def _prefetch_request_stop_step(
    consumed_stop_step: int,
    *,
    spec: _BoundedRuntimeProfileSpec | None,
    prefetch_active: bool,
    queue_size: int,
) -> int:
    if spec is None or not prefetch_active:
        return int(consumed_stop_step)
    requested_stop_step = int(consumed_stop_step) + int(queue_size) + int(spec.measured_steps)
    frozen_protocol = int(spec.warmup_steps) == 4 and int(spec.measured_steps) == 28
    expected_queue = _BOUNDED_RUNTIME_PROFILE_QUEUE_BY_MODE[spec.profile_mode]
    expected_stop = _BOUNDED_RUNTIME_PROFILE_REQUEST_STOP_BY_MODE[spec.profile_mode]
    if frozen_protocol and (
        int(consumed_stop_step) != 32
        or int(queue_size) != expected_queue
        or requested_stop_step != expected_stop
    ):
        raise RuntimeError(
            f"frozen bounded profile mode={spec.profile_mode} requires consumed_stop_step=32, "
            f"queue_size={expected_queue}, and requested_stop_step={expected_stop}; "
            f"got consumed={consumed_stop_step} queue={queue_size} requested={requested_stop_step}"
        )
    return requested_stop_step


_PROFILE_PREFETCH_CLOSE_TIMEOUT_SECONDS = 5 * 60.0


def _close_training_prefetcher(
    prefetcher: BackgroundPrefetcher[Any],
    *,
    bounded_profile_spec: _BoundedRuntimeProfileSpec | None,
    profile_timeout_s: float = _PROFILE_PREFETCH_CLOSE_TIMEOUT_SECONDS,
) -> None:
    """Close normally, but make profile producer-state snapshots race-free."""

    prefetcher.close()
    if bounded_profile_spec is not None and not prefetcher.wait_closed(profile_timeout_s):
        raise RuntimeError(
            "bounded runtime profile prefetch producer did not terminate before the "
            f"{profile_timeout_s:.1f}s close barrier"
        )


def _producer_close_ledger(
    *,
    committed_logical_index_end_exclusive: int,
    produced_logical_index: int,
    requested_horizon_steps: int,
    requested_horizon_tiles: int,
    tiles_per_optimizer_step: int,
    profile_mode: str,
    require_unconsumed_tail: bool,
) -> dict[str, Any]:
    expected_horizon_tiles = int(requested_horizon_steps) * int(tiles_per_optimizer_step)
    if int(requested_horizon_tiles) != expected_horizon_tiles:
        raise RuntimeError(
            "bounded profile producer horizon unit mismatch: "
            f"steps={requested_horizon_steps} tiles_per_step={tiles_per_optimizer_step} "
            f"expected_tiles={expected_horizon_tiles} actual_tiles={requested_horizon_tiles}"
        )
    expected_steps = _BOUNDED_RUNTIME_PROFILE_REQUEST_STOP_BY_MODE.get(profile_mode)
    if expected_steps is None:
        raise RuntimeError(f"unknown bounded profile mode in producer close ledger: {profile_mode!r}")
    if require_unconsumed_tail and (
        int(requested_horizon_steps) != expected_steps
        or int(tiles_per_optimizer_step) != 32
        or int(requested_horizon_tiles) != expected_steps * 32
    ):
        raise RuntimeError(
            f"frozen bounded profile mode={profile_mode} requires requested horizon "
            f"{expected_steps} optimizer steps = {expected_steps * 32} logical tiles "
            "at 32 tiles/optimizer-step"
        )
    if produced_logical_index < committed_logical_index_end_exclusive:
        raise RuntimeError("bounded profile producer close cursor trails committed consumption")
    discarded = produced_logical_index - committed_logical_index_end_exclusive
    if require_unconsumed_tail and discarded <= 0:
        raise RuntimeError(
            "bounded runtime profile producer did not materialize beyond committed consumption; "
            "concurrent-refill evidence is not established"
        )
    return {
        "committed_logical_index_end_exclusive": int(committed_logical_index_end_exclusive),
        "produced_logical_index_at_close": int(produced_logical_index),
        "profile_mode": profile_mode,
        "requested_horizon_steps": int(requested_horizon_steps),
        "requested_horizon_tiles": int(requested_horizon_tiles),
        "tiles_per_optimizer_step": int(tiles_per_optimizer_step),
        "horizon_units": {
            "requested_horizon_steps": "optimizer_steps",
            "requested_horizon_tiles": "logical_tiles",
            "tiles_per_optimizer_step": "logical_tiles_per_optimizer_step",
        },
        "unconsumed_discarded_tiles": int(discarded),
        "unconsumed_tail_required": bool(require_unconsumed_tail),
        "unconsumed_tail_observed": discarded > 0,
        "unconsumed_tail_commit_policy": "discard_without_commit",
    }


def _checkpoint_writes_allowed(spec: _BoundedRuntimeProfileSpec | None) -> bool:
    return spec is None


class _BoundedRuntimeProfileRecorder:
    """Low-volume timing recorder attached to the real production worker loop."""

    def __init__(
        self,
        spec: _BoundedRuntimeProfileSpec,
        *,
        device: torch.device,
        candidate: str,
        model_fingerprint_sha256: str,
        source_implementation_seal_sha256: str,
        candidate_artifact_set_sha256: str,
        candidate_report_sha256: str,
        candidate_index_sha256: str,
        pair_manifest: str,
        pair_manifest_sha256: str,
        resolved_config_sha256: str,
        profile_contract_sha256: str,
        active_config: Mapping[str, Any],
        actual_model_config: Mapping[str, Any],
        actual_trainable_parameters: int,
        actual_total_parameters: int,
        input_path_ledger: Mapping[str, Any],
        worker_started_s: float,
        gpu_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if not candidate.strip():
            raise RuntimeError("bounded runtime profile requires a non-empty AE candidate")
        for field_name, field_value in (
            ("model_fingerprint_sha256", model_fingerprint_sha256),
            ("source_implementation_seal_sha256", source_implementation_seal_sha256),
            ("candidate_artifact_set_sha256", candidate_artifact_set_sha256),
            ("candidate_report_sha256", candidate_report_sha256),
            ("candidate_index_sha256", candidate_index_sha256),
            ("pair_manifest_sha256", pair_manifest_sha256),
            ("resolved_config_sha256", resolved_config_sha256),
            ("profile_contract_sha256", profile_contract_sha256),
        ):
            if len(field_value) != 64 or any(character not in "0123456789abcdef" for character in field_value.lower()):
                raise RuntimeError(f"bounded runtime profile requires a valid {field_name}")
        pair_manifest_path = Path(pair_manifest).expanduser()
        if not pair_manifest_path.is_file() or sha256_file(pair_manifest_path) != pair_manifest_sha256:
            raise RuntimeError("bounded runtime profile operator-bank pair manifest does not match its SHA-256")
        resolved_config_path = spec.output_dir / "resolved_profile_config.json"
        if not resolved_config_path.is_file() or sha256_file(resolved_config_path) != resolved_config_sha256:
            raise RuntimeError("bounded runtime profile resolved config does not match its immutable SHA-256")
        profile_contract_path = spec.output_dir / "profile_contract.json"
        if not profile_contract_path.is_file() or sha256_file(profile_contract_path) != profile_contract_sha256:
            raise RuntimeError("bounded runtime profile contract does not match its immutable SHA-256")
        resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        profile_contract = json.loads(profile_contract_path.read_text(encoding="utf-8"))
        required_profile_contract = {
            "profile_schema": _BOUNDED_RUNTIME_PROFILE_SCHEMA,
            "profile_mode": spec.profile_mode,
            "prefetch_queue_size": _BOUNDED_RUNTIME_PROFILE_QUEUE_BY_MODE[spec.profile_mode],
            "candidate": candidate,
            "model_fingerprint_sha256": model_fingerprint_sha256,
            "source_implementation_seal_sha256": source_implementation_seal_sha256,
            "candidate_artifact_set_sha256": candidate_artifact_set_sha256,
            "candidate_report_sha256": candidate_report_sha256,
            "candidate_index_sha256": candidate_index_sha256,
            "pair_manifest": pair_manifest,
            "pair_manifest_sha256": pair_manifest_sha256,
            "production_max_steps": _BOUNDED_RUNTIME_PROFILE_PRODUCTION_STEPS,
            "hard_cap_optimizer_steps": _BOUNDED_RUNTIME_PROFILE_MAX_STEPS,
            "warmup_steps": spec.warmup_steps,
            "measured_steps": spec.measured_steps,
            "external_tracking": False,
            "resume": False,
            "checkpoint_writes": False,
            "resolved_profile_config": str(resolved_config_path.resolve()),
            "resolved_profile_config_sha256": resolved_config_sha256,
        }
        for key, expected in required_profile_contract.items():
            if profile_contract.get(key) != expected:
                raise RuntimeError(
                    f"bounded runtime profile contract mismatch for {key}: "
                    f"expected={expected!r} actual={profile_contract.get(key)!r}"
                )
        actual_source_seal = ae_source_implementation_seal()
        if actual_source_seal["source_implementation_seal_sha256"] != source_implementation_seal_sha256:
            raise RuntimeError("bounded runtime profile worker source differs from its implementation seal")
        active_contract = _profile_config_contract_payload(active_config)
        resolved_contract = _profile_config_contract_payload(resolved_config)
        if active_contract != resolved_contract:
            mismatch_path = spec.output_dir / "profile_config_mismatch.json"
            mismatches = _profile_config_contract_diff(resolved_contract, active_contract)
            write_json_immutable(
                mismatch_path,
                {
                    "schema": "weightclip_ae_profile_config_mismatch_v1",
                    "status": "failed_closed",
                    "resolved_contract_sha256": _canonical_payload_sha256(resolved_contract),
                    "active_contract_sha256": _canonical_payload_sha256(active_contract),
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches,
                },
            )
            raise RuntimeError(
                "bounded runtime profile active worker config differs from immutable resolved profile config: "
                f"mismatch_count={len(mismatches)} report={mismatch_path}"
            )
        actual_model_config = dict(actual_model_config)
        actual_model_fingerprint = _canonical_payload_sha256(actual_model_config)
        expected_model_fingerprint = _canonical_payload_sha256(dict(resolved_config["model"]))
        if actual_model_fingerprint != expected_model_fingerprint:
            raise RuntimeError("bounded runtime profile active model config differs from its resolved config")
        if actual_model_fingerprint != model_fingerprint_sha256:
            raise RuntimeError("bounded runtime profile active model config differs from the approved candidate fingerprint")
        if actual_trainable_parameters < 1 or actual_total_parameters < actual_trainable_parameters:
            raise RuntimeError("bounded runtime profile received an invalid actual model parameter ledger")
        parameter_budget = _validate_profile_parameter_budget(actual_trainable_parameters)
        self.spec = spec
        self.device = device
        self.candidate = candidate
        self.model_fingerprint_sha256 = model_fingerprint_sha256
        self.source_implementation_seal_sha256 = source_implementation_seal_sha256
        self.candidate_artifact_set_sha256 = candidate_artifact_set_sha256
        self.candidate_report_sha256 = candidate_report_sha256
        self.candidate_index_sha256 = candidate_index_sha256
        self.pair_manifest = pair_manifest
        self.pair_manifest_sha256 = pair_manifest_sha256
        self.resolved_config_sha256 = resolved_config_sha256
        self.resolved_config_path = str(resolved_config_path.resolve())
        self.profile_contract_sha256 = profile_contract_sha256
        self.profile_contract_path = str(profile_contract_path.resolve())
        self.active_config_contract_sha256 = _canonical_payload_sha256(active_contract)
        self.actual_model_config = actual_model_config
        self.actual_model_config_sha256 = actual_model_fingerprint
        self.actual_trainable_parameters = int(actual_trainable_parameters)
        self.actual_total_parameters = int(actual_total_parameters)
        self.parameter_budget = parameter_budget
        self.input_path_ledger = dict(input_path_ledger)
        self.gpu_identity = dict(gpu_identity or _resolve_profile_gpu_identity(device))
        expected_gpu_uuid = str(os.environ.get("WEIGHTCLIP_AE_PROFILE_GPU_UUID", "") or "")
        if expected_gpu_uuid and self.gpu_identity.get("physical_uuid") != expected_gpu_uuid:
            raise RuntimeError(
                "bounded runtime profile worker GPU UUID differs from the launcher binding: "
                f"worker={self.gpu_identity.get('physical_uuid')} launcher={expected_gpu_uuid}"
            )
        self.worker_started_s = worker_started_s
        self.worker_loop_ready_s = time.perf_counter()
        self.rows: list[dict[str, Any]] = []
        self._step_started_s = 0.0
        self._measurement_started_s: float | None = None
        self._measurement_finished_s: float | None = None
        self._cuda_start_events: list[torch.cuda.Event] = []
        self._cuda_end_events: list[torch.cuda.Event] = []
        self._cuda_start_recorded = False
        self._nvml_process: subprocess.Popen[str] | None = None
        self._nvml_stdout = ""
        self._nvml_stderr = ""
        self._nvml_startup_s = 0.0
        self.input_validation: dict[str, Any] | None = None
        self.producer_close_ledger: dict[str, Any] | None = None

    def _start_nvml(self) -> None:
        if self._nvml_process is not None:
            return
        started = time.perf_counter()
        command = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_identity['physical_uuid']}",
            "-lms",
            "100",
        ]
        self._nvml_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._nvml_startup_s = time.perf_counter() - started

    def _stop_nvml(self) -> None:
        process = self._nvml_process
        if process is None:
            return
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5.0)
        self._nvml_stdout = stdout
        self._nvml_stderr = stderr

    def abort(self) -> None:
        self._stop_nvml()

    @staticmethod
    def _parse_nvml_rows(text: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            values = [value.strip() for value in raw_line.split(",")]
            if len(values) != 7:
                continue
            try:
                rows.append(
                    {
                        "timestamp": values[0],
                        "gpu_index": int(values[1]),
                        "gpu_uuid": values[2],
                        "gpu_utilization_percent": float(values[3]),
                        "memory_utilization_percent": float(values[4]),
                        "memory_used_mib": float(values[5]),
                        "power_draw_w": float(values[6]),
                    }
                )
            except ValueError:
                continue
        return rows

    def begin_step(self, global_step: int) -> None:
        if global_step == self.spec.warmup_steps + 1:
            if self._nvml_process is None:
                raise RuntimeError("bounded runtime profile NVML sampler was not warmed before measurement")
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._measurement_started_s = time.perf_counter()
        self._step_started_s = time.perf_counter()
        if global_step > self.spec.warmup_steps:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self._cuda_start_events.append(start)
            self._cuda_end_events.append(end)
            self._cuda_start_recorded = False

    def begin_cuda_work(self, global_step: int) -> None:
        if global_step <= self.spec.warmup_steps or self._cuda_start_recorded:
            return
        self._cuda_start_events[-1].record()
        self._cuda_start_recorded = True

    def validate_input_batch(self, batch: PreparedTrainingBatch, *, expected_batch_size: int) -> None:
        if self.input_validation is not None:
            return
        expected_shapes = {
            "W": (expected_batch_size, 128, 128),
            "x": (expected_batch_size, 512, 128),
            "x_mask": (expected_batch_size, 512),
            "d_in_mask": (expected_batch_size, 128),
            "d_out_mask": (expected_batch_size, 128),
        }
        tensors = {
            "W": batch.W,
            "x": batch.x,
            "x_mask": batch.x_mask,
            "d_in_mask": batch.d_in_mask,
            "d_out_mask": batch.d_out_mask,
        }
        actual_shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"bounded runtime profile input shape mismatch: expected={expected_shapes} actual={actual_shapes}"
            )
        if not torch.isfinite(batch.W).all() or not torch.isfinite(batch.x).all():
            raise RuntimeError("bounded runtime profile received non-finite operator-bank tensors")
        for name in ("x_mask", "d_in_mask", "d_out_mask"):
            if tensors[name].dtype != torch.bool:
                raise RuntimeError(f"bounded runtime profile mask {name} is not boolean")
            if not bool(tensors[name].any()):
                raise RuntimeError(f"bounded runtime profile mask {name} is empty")
        tensor_bytes = {
            name: int(value.numel() * value.element_size()) for name, value in tensors.items()
        }
        prepared_batch_bytes = sum(tensor_bytes.values())
        microbatch_ledger = self.input_path_ledger.get("microbatch", {})
        memory_ledger = self.input_path_ledger.get("memory_bounds", {})
        if not isinstance(microbatch_ledger, dict) or not isinstance(memory_ledger, dict):
            raise RuntimeError("bounded runtime profile input path lacks memory contract ledgers")
        queue_size = int(microbatch_ledger.get("prepared_batch_queue_size", -1))
        cpu_threads = int(microbatch_ledger.get("cpu_threads", -1))
        if queue_size < 1 or cpu_threads < 1:
            raise RuntimeError("bounded runtime profile has invalid prefetch queue/cpu-thread bounds")
        # Beyond the bounded queue, conservatively include all simultaneous
        # transient materializations: fetch input records while stacking,
        # unpinned input while pinning, and the consumer-owned batch.
        prepared_batches_in_memory_bound = queue_size + 3
        prepared_batch_tensor_bytes_bound = prepared_batches_in_memory_bound * prepared_batch_bytes
        active_bytes = int(memory_ledger.get("active_strata_tensor_bytes", -1))
        inflight_bytes = int(memory_ledger.get("inflight_tensor_bytes", -1))
        if active_bytes < 0 or inflight_bytes < 0:
            raise RuntimeError("bounded runtime profile has invalid operator-bundle memory bounds")
        memory_ledger.update(
            {
                "prepared_batch_bytes": prepared_batch_bytes,
                "prepared_batch_tensor_bytes_by_field": tensor_bytes,
                "prepared_batches_in_memory_bound": prepared_batches_in_memory_bound,
                "prepared_batch_tensor_bytes_bound": prepared_batch_tensor_bytes_bound,
                "process_tensor_bytes_bound": active_bytes + inflight_bytes + prepared_batch_tensor_bytes_bound,
            }
        )
        self.input_validation = {
            "status": "passed",
            "shapes": {name: list(shape) for name, shape in actual_shapes.items()},
            "dtypes": {name: str(value.dtype) for name, value in tensors.items()},
            "tensor_bytes": tensor_bytes,
            "prepared_batch_bytes": prepared_batch_bytes,
            "prefetch_memory_bound": {
                "queue_size": queue_size,
                "cpu_threads": cpu_threads,
                "queue_plus_fetch_pin_consumer_batches": prepared_batches_in_memory_bound,
                "prepared_batch_tensor_bytes_bound": prepared_batch_tensor_bytes_bound,
                "process_tensor_bytes_bound": memory_ledger["process_tensor_bytes_bound"],
            },
            "finite_weight_and_context": True,
            "nonempty_boolean_masks": True,
        }

    def finish_step(
        self,
        *,
        global_step: int,
        loss: float,
        data_wait_s: float,
        data_build_s: float,
        pin_s: float,
        h2d_enqueue_s: float,
        prefetch_depth: float,
        h2d_bytes: int,
        batch_size: int,
        grad_accum_steps: int,
        committed_logical_index_start: int,
        committed_logical_index_end_exclusive: int,
        produced_logical_index: int,
        mixer_telemetry: Mapping[str, Any],
    ) -> None:
        measured = global_step > self.spec.warmup_steps
        if measured:
            if not self._cuda_start_recorded:
                raise RuntimeError("bounded runtime profile did not observe CUDA work for a measured step")
            self._cuda_end_events[-1].record()
        expected_tiles = int(batch_size * grad_accum_steps)
        if committed_logical_index_end_exclusive - committed_logical_index_start != expected_tiles:
            raise RuntimeError("bounded runtime profile mixer cursor disagrees with consumed optimizer-step tiles")
        if produced_logical_index < committed_logical_index_end_exclusive:
            raise RuntimeError("bounded runtime profile producer cursor trails consumed logical tiles")
        self.rows.append(
            {
                "global_step": int(global_step),
                "phase": "measured" if measured else "warmup",
                "host_step_ms": 1000.0 * (time.perf_counter() - self._step_started_s),
                "input_wait_ms": 1000.0 * float(data_wait_s),
                "fetch_pre_pin_ms": 1000.0 * float(data_build_s),
                "pin_memory_ms": 1000.0 * float(pin_s),
                "total_cpu_producer_ms": 1000.0 * float(data_build_s + pin_s),
                "h2d_enqueue_ms": 1000.0 * float(h2d_enqueue_s),
                "prefetch_depth_sum": float(prefetch_depth),
                "prefetch_depth_mean": float(prefetch_depth) / max(1, int(grad_accum_steps)),
                "h2d_bytes": int(h2d_bytes),
                "loss": float(loss),
                "batch_size": int(batch_size),
                "grad_accum_steps": int(grad_accum_steps),
                "examples": int(batch_size * grad_accum_steps),
                "committed_logical_index_start": int(committed_logical_index_start),
                "committed_logical_index_end_exclusive": int(committed_logical_index_end_exclusive),
                "produced_logical_index": int(produced_logical_index),
                "producer_ahead_tiles": int(produced_logical_index - committed_logical_index_end_exclusive),
                "mixer_telemetry": dict(mixer_telemetry),
            }
        )
        if global_step == self.spec.warmup_steps:
            # Start and warm the external sampler before the measured wall
            # clock.  Its process-start overhead is reported separately.
            self._start_nvml()
        if global_step == self.spec.total_steps:
            torch.cuda.synchronize(self.device)
            self._measurement_finished_s = time.perf_counter()
            self._stop_nvml()

    def record_producer_close(
        self,
        *,
        produced_logical_index: int,
        requested_horizon_steps: int,
        requested_horizon_tiles: int,
        tiles_per_optimizer_step: int,
    ) -> None:
        if not self.rows:
            raise RuntimeError("bounded runtime profile producer closed before any consumed step")
        frozen_protocol = (
            self.spec.warmup_steps == _BOUNDED_RUNTIME_PROFILE_REQUIRED_WARMUP_STEPS
            and self.spec.measured_steps == _BOUNDED_RUNTIME_PROFILE_REQUIRED_MEASURED_STEPS
        )
        self.producer_close_ledger = _producer_close_ledger(
            committed_logical_index_end_exclusive=int(
                self.rows[-1]["committed_logical_index_end_exclusive"]
            ),
            produced_logical_index=produced_logical_index,
            requested_horizon_steps=requested_horizon_steps,
            requested_horizon_tiles=requested_horizon_tiles,
            tiles_per_optimizer_step=tiles_per_optimizer_step,
            profile_mode=self.spec.profile_mode,
            require_unconsumed_tail=frozen_protocol,
        )

    def finalize(self) -> dict[str, Any]:
        if self._measurement_started_s is None or self._measurement_finished_s is None:
            raise RuntimeError("bounded runtime profile did not complete its measured window")
        if len(self._cuda_start_events) != self.spec.measured_steps:
            raise RuntimeError("bounded runtime profile CUDA-event inventory is incomplete")
        measured_rows = [row for row in self.rows if row["phase"] == "measured"]
        if len(measured_rows) != self.spec.measured_steps:
            raise RuntimeError("bounded runtime profile measured-step inventory is incomplete")
        if self.input_validation is None:
            raise RuntimeError("bounded runtime profile never validated a production input batch")
        if self.producer_close_ledger is None:
            raise RuntimeError("bounded runtime profile never recorded producer close/discard state")
        for previous, current in zip(self.rows, self.rows[1:]):
            if int(previous["committed_logical_index_end_exclusive"]) != int(
                current["committed_logical_index_start"]
            ):
                raise RuntimeError("bounded runtime profile consumed logical history is not contiguous")
        for row, start, end in zip(measured_rows, self._cuda_start_events, self._cuda_end_events, strict=True):
            row["cuda_stream_ms"] = float(start.elapsed_time(end))
        nvml_rows = self._parse_nvml_rows(self._nvml_stdout)
        if not nvml_rows:
            raise RuntimeError(f"bounded runtime profile received no valid nvidia-smi samples: {self._nvml_stderr}")
        if any(row["gpu_uuid"] != self.gpu_identity["physical_uuid"] for row in nvml_rows):
            raise RuntimeError("bounded runtime profile NVML samples came from a different physical GPU UUID")
        if any(not math.isfinite(float(row["loss"])) for row in measured_rows):
            raise RuntimeError("bounded runtime profile observed a non-finite loss")
        frozen_refill_protocol = (
            self.spec.warmup_steps == _BOUNDED_RUNTIME_PROFILE_REQUIRED_WARMUP_STEPS
            and self.spec.measured_steps == _BOUNDED_RUNTIME_PROFILE_REQUIRED_MEASURED_STEPS
        )
        measured_refill_ledger = _measured_producer_refill_ledger(
            measured_rows,
            required=frozen_refill_protocol,
        )
        wall_s = self._measurement_finished_s - self._measurement_started_s
        measured_consumed_span = int(
            measured_rows[-1]["committed_logical_index_end_exclusive"]
        ) - int(measured_rows[0]["committed_logical_index_end_exclusive"])
        measured_refill_ledger.update(
            {
                "consumed_span_tiles": measured_consumed_span,
                "producer_tiles_per_second": float(
                    measured_refill_ledger["advance_tiles"]
                ) / wall_s,
                "consumer_span_tiles_per_second": measured_consumed_span / wall_s,
                "producer_to_consumer_span_ratio": (
                    float(measured_refill_ledger["advance_tiles"]) / measured_consumed_span
                    if measured_consumed_span > 0
                    else None
                ),
                "no_starvation_claim": "not_automatic_review_input_wait_queue_and_rates",
            }
        )
        host_ms = [float(row["host_step_ms"]) for row in measured_rows]
        wait_ms = [float(row["input_wait_ms"]) for row in measured_rows]
        build_ms = [float(row["fetch_pre_pin_ms"]) for row in measured_rows]
        pin_ms = [float(row["pin_memory_ms"]) for row in measured_rows]
        producer_ms = [float(row["total_cpu_producer_ms"]) for row in measured_rows]
        h2d_ms = [float(row["h2d_enqueue_ms"]) for row in measured_rows]
        prefetch_depth = [float(row["prefetch_depth_mean"]) for row in measured_rows]
        h2d_bytes = [int(row["h2d_bytes"]) for row in measured_rows]
        cuda_ms = [float(row["cuda_stream_ms"]) for row in measured_rows]
        gpu_util = [float(row["gpu_utilization_percent"]) for row in nvml_rows]
        input_wait_fraction = sum(wait_ms) / (1000.0 * wall_s)
        input_wait_p95_ms = _profile_percentile(wait_ms, 0.95)
        median_host_step_ms = statistics.median(host_ms)
        stress_gate = {
            "applicable": self.spec.profile_mode == "producer_stress",
            "status": "not_applicable",
            "thresholds": {
                "min_measured_producer_advance_tiles": 832,
                "min_producer_to_consumer_span_ratio": 26.0 / 27.0,
                "max_input_wait_fraction_of_measured_wall": 0.01,
                "max_input_wait_p95_fraction_of_median_host_step": 0.05,
            },
            "observed": {
                "measured_producer_advance_tiles": int(measured_refill_ledger["advance_tiles"]),
                "producer_to_consumer_span_ratio": measured_refill_ledger[
                    "producer_to_consumer_span_ratio"
                ],
                "input_wait_fraction_of_measured_wall": input_wait_fraction,
                "input_wait_p95_ms": input_wait_p95_ms,
                "median_host_step_ms": median_host_step_ms,
                "input_wait_p95_fraction_of_median_host_step": (
                    input_wait_p95_ms / median_host_step_ms if median_host_step_ms > 0 else None
                ),
            },
        }
        if self.spec.profile_mode == "producer_stress":
            observed_ratio = float(measured_refill_ledger["producer_to_consumer_span_ratio"])
            wait_p95_fraction = input_wait_p95_ms / median_host_step_ms if median_host_step_ms > 0 else math.inf
            failures = []
            if int(measured_refill_ledger["advance_tiles"]) < 832:
                failures.append("measured_producer_advance_tiles")
            if observed_ratio < 26.0 / 27.0:
                failures.append("producer_to_consumer_span_ratio")
            if input_wait_fraction > 0.01:
                failures.append("input_wait_fraction_of_measured_wall")
            if wait_p95_fraction > 0.05:
                failures.append("input_wait_p95_fraction_of_median_host_step")
            stress_gate["status"] = "passed" if not failures else "failed"
            stress_gate["failed_metrics"] = failures
            if failures:
                raise RuntimeError(
                    "producer_stress bounded profile failed ingress gates: "
                    f"{failures}; observed={stress_gate['observed']}"
                )
        if max(gpu_util) <= 0.0:
            raise RuntimeError("bounded runtime profile did not observe any active GPU sample")
        new_checkpoint_files = sorted(
            str(path.resolve())
            for suffix in ("*.pt", "*.ckpt", "*.safetensors")
            for path in self.spec.checkpoint_dir.rglob(suffix)
        ) if self.spec.checkpoint_dir.exists() else []
        if new_checkpoint_files:
            raise RuntimeError(f"bounded runtime profile unexpectedly wrote checkpoints: {new_checkpoint_files}")
        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        step_paths = write_records_immutable(self.spec.output_dir / "step_timings", self.rows)
        nvml_paths = write_records_immutable(self.spec.output_dir / "nvml_samples", nvml_rows)
        summary = {
            "schema_version": 1,
            "profile_schema": _BOUNDED_RUNTIME_PROFILE_SCHEMA,
            "profile_mode": self.spec.profile_mode,
            "termination_reason": "bounded_profile_complete",
            "candidate": self.candidate,
            "model_fingerprint_sha256": self.model_fingerprint_sha256,
            "source_implementation_seal_sha256": self.source_implementation_seal_sha256,
            "candidate_artifact_set_sha256": self.candidate_artifact_set_sha256,
            "candidate_report_sha256": self.candidate_report_sha256,
            "candidate_index_sha256": self.candidate_index_sha256,
            "pair_manifest": self.pair_manifest,
            "pair_manifest_sha256": self.pair_manifest_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "resolved_config_path": self.resolved_config_path,
            "profile_contract_sha256": self.profile_contract_sha256,
            "profile_contract_path": self.profile_contract_path,
            "gpu_identity": self.gpu_identity,
            "active_config_contract_sha256": self.active_config_contract_sha256,
            "actual_model": {
                "config": self.actual_model_config,
                "config_sha256": self.actual_model_config_sha256,
                "trainable_parameters": self.actual_trainable_parameters,
                "total_parameters": self.actual_total_parameters,
                "approved_parameter_budget": self.parameter_budget,
            },
            "production_max_steps": _BOUNDED_RUNTIME_PROFILE_PRODUCTION_STEPS,
            "hard_cap_optimizer_steps": _BOUNDED_RUNTIME_PROFILE_MAX_STEPS,
            "warmup_steps": self.spec.warmup_steps,
            "measured_steps": self.spec.measured_steps,
            "measured_producer_refill": measured_refill_ledger,
            "producer_stress_gate": stress_gate,
            "producer_close": self.producer_close_ledger,
            "executed_optimizer_steps": self.spec.total_steps,
            "measured_wall_seconds": wall_s,
            "optimizer_steps_per_second": self.spec.measured_steps / wall_s,
            "examples_per_second": sum(int(row["examples"]) for row in measured_rows) / wall_s,
            "operator_tiles_per_second": sum(int(row["examples"]) for row in measured_rows) / wall_s,
            "h2d_bytes_per_second": sum(int(row["h2d_bytes"]) for row in measured_rows) / wall_s,
            "host_step_ms": {
                "mean": statistics.fmean(host_ms),
                "median": statistics.median(host_ms),
                "p10": _profile_percentile(host_ms, 0.10),
                "p90": _profile_percentile(host_ms, 0.90),
            },
            "input_wait_ms": {
                "mean": statistics.fmean(wait_ms),
                "p50": statistics.median(wait_ms),
                "p95": input_wait_p95_ms,
                "fraction_of_measured_wall": input_wait_fraction,
            },
            "fetch_pre_pin_ms": {
                "mean": statistics.fmean(build_ms),
                "p50": statistics.median(build_ms),
                "p95": _profile_percentile(build_ms, 0.95),
            },
            "pin_memory_ms": {
                "mean": statistics.fmean(pin_ms),
                "p50": statistics.median(pin_ms),
                "p95": _profile_percentile(pin_ms, 0.95),
            },
            "total_cpu_producer_ms": {
                "mean": statistics.fmean(producer_ms),
                "p50": statistics.median(producer_ms),
                "p95": _profile_percentile(producer_ms, 0.95),
            },
            "h2d_enqueue_ms": {
                "mean": statistics.fmean(h2d_ms),
                "p50": statistics.median(h2d_ms),
                "p95": _profile_percentile(h2d_ms, 0.95),
            },
            "h2d_bytes": {
                "total": sum(h2d_bytes),
                "mean": statistics.fmean(h2d_bytes),
                "p50": statistics.median(h2d_bytes),
                "p95": _profile_percentile(h2d_bytes, 0.95),
            },
            "prefetch_queue_depth": {
                "mean": statistics.fmean(prefetch_depth),
                "p50": statistics.median(prefetch_depth),
                "p95": _profile_percentile(prefetch_depth, 0.95),
            },
            "cuda_stream_ms": {
                "mean": statistics.fmean(cuda_ms),
                "p50": statistics.median(cuda_ms),
                "p95": _profile_percentile(cuda_ms, 0.95),
            },
            "gpu_utilization_percent": {
                "samples": len(gpu_util),
                "active_samples": sum(value > 0.0 for value in gpu_util),
                "active_fraction": sum(value > 0.0 for value in gpu_util) / len(gpu_util),
                "mean": statistics.fmean(gpu_util),
                "p50": statistics.median(gpu_util),
                "p95": _profile_percentile(gpu_util, 0.95),
            },
            "peak_vram_mib": {
                "allocated": torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0),
                "reserved": torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0),
                "nvml_used": max(float(row["memory_used_mib"]) for row in nvml_rows),
            },
            "worker_setup_seconds_before_loop": max(0.0, self.worker_loop_ready_s - self.worker_started_s),
            "nvml_sampler_startup_ms_excluded_from_measured_wall": 1000.0 * self._nvml_startup_s,
            "external_tracking_enabled": False,
            "checkpoint_files_found": new_checkpoint_files,
            "committed_index_contract": {
                "start": int(self.rows[0]["committed_logical_index_start"]),
                "end_exclusive": int(self.rows[-1]["committed_logical_index_end_exclusive"]),
                "kind": "consumed_optimizer_step_logical_range",
                "source": "consumed_prepared_batch_logical_ranges",
                "produced_cursor_source": "balanced_operator_bank_mixer_prefetch_cursor",
            },
            "input_validation": self.input_validation,
            "production_input_path": self.input_path_ledger,
            "final_mixer_telemetry": dict(self.rows[-1]["mixer_telemetry"]),
            "artifacts": {
                "step_timings": step_paths,
                "nvml_samples": nvml_paths,
            },
        }
        summary_path = self.spec.output_dir / "summary.json"
        write_json_immutable(summary_path, summary)
        summary["summary_path"] = str(summary_path.resolve())
        summary["summary_sha256"] = sha256_file(summary_path)
        return summary



def _run_worker(
    rank: int,
    world_size: int,
    cfg_dict: dict[str, Any],
    master_addr: str,
    master_port: int,
    monitor_queue: Any | None = None,
) -> None:
    worker_started_s = time.perf_counter()
    print(f"[train_big_vae rank{rank}] worker bootstrapping", flush=True)
    maybe_redirect_stdio(cfg_dict, role="train_worker", section="train", rank=rank)
    cfg = OmegaConf.create(cfg_dict)
    _promote_run_profile_to_root(cfg)
    bounded_profile_spec = _resolve_bounded_runtime_profile(cfg, rank=rank, world_size=world_size)
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
    _seed_everything(seed, active_cuda_device=device)
    _set_speed_optimizations(cfg, device=device)
    v10_fp32_backend = _enforce_v10_strict_fp32_backend(cfg, device=device)
    if v10_fp32_backend is not None and rank == 0:
        logger.info("V10 strict FP32 backend: %s", json.dumps(v10_fp32_backend, sort_keys=True))

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
    operator_bank_cfg = cfg.train.get("operator_bank", {})
    if operator_bank_cfg is None:
        operator_bank_cfg = {}
    if not isinstance(operator_bank_cfg, (dict, DictConfig)):
        raise TypeError("train.operator_bank must be a mapping")
    operator_bank_enabled = bool(operator_bank_cfg.get("enabled", False))
    operator_bank_pair_manifest = str(operator_bank_cfg.get("pair_manifest", "") or "").strip()
    operator_bank_pair_manifest_sha256 = str(operator_bank_cfg.get("pair_manifest_sha256", "") or "").strip()
    operator_bank_locality_audit_path = str(operator_bank_cfg.get("locality_audit_path", "") or "").strip()
    operator_bank_locality_audit_sha256 = str(
        operator_bank_cfg.get("locality_audit_sha256", "") or ""
    ).strip()
    operator_bank_shard_by_rank = bool(operator_bank_cfg.get("shard_by_rank", True))
    operator_bank_repeat = bool(operator_bank_cfg.get("repeat", True))
    operator_bank_permutation_views = bool(operator_bank_cfg.get("permutation_views", True))
    operator_bank_canonical_probability = float(operator_bank_cfg.get("canonical_probability", 1.0 / 6.0))
    operator_bank_hot_shards = max(1, int(operator_bank_cfg.get("hot_shards", 8)))
    operator_bank_max_active_strata = max(1, int(operator_bank_cfg.get("max_active_strata", 256)))
    operator_bank_max_active_bundle_bytes = max(
        1,
        int(operator_bank_cfg.get("max_active_bundle_bytes", 1536 * 1024 * 1024)),
    )
    operator_bank_max_inflight_bundles = max(1, int(operator_bank_cfg.get("max_inflight_bundles", 17)))
    operator_bank_max_inflight_bundle_bytes = max(
        1,
        int(operator_bank_cfg.get("max_inflight_bundle_bytes", 128 * 1024 * 1024)),
    )
    two_operator_overfit_cfg = operator_bank_cfg.get("two_operator_overfit", {})
    if two_operator_overfit_cfg is None:
        two_operator_overfit_cfg = {}
    if not isinstance(two_operator_overfit_cfg, (dict, DictConfig)):
        raise TypeError("train.operator_bank.two_operator_overfit must be a mapping")
    two_operator_overfit_enabled = bool(two_operator_overfit_cfg.get("enabled", False))
    two_operator_overfit_selection = list(two_operator_overfit_cfg.get("selected_operators", []))
    if two_operator_overfit_enabled:
        if len(two_operator_overfit_selection) != 2:
            raise ValueError("two-operator overfit requires exactly two selected full operators")
        if operator_bank_permutation_views or operator_bank_canonical_probability != 1.0:
            raise ValueError(
                "two-operator overfit requires fixed canonical data: "
                "permutation_views=false and canonical_probability=1"
            )
        if world_size != 1:
            raise ValueError("two-operator overfit is a single-process diagnostic")
    operator_set_overfit_cfg = operator_bank_cfg.get("operator_set_overfit", {})
    if operator_set_overfit_cfg is None:
        operator_set_overfit_cfg = {}
    if not isinstance(operator_set_overfit_cfg, (dict, DictConfig)):
        raise TypeError("train.operator_bank.operator_set_overfit must be a mapping")
    operator_set_overfit_enabled = bool(operator_set_overfit_cfg.get("enabled", False))
    operator_set_overfit_selection = list(operator_set_overfit_cfg.get("selected_operators", []))
    if operator_set_overfit_enabled:
        if two_operator_overfit_enabled:
            raise ValueError("two_operator_overfit and operator_set_overfit are mutually exclusive")
        if len(operator_set_overfit_selection) != 64:
            raise ValueError("V9 operator-set diagnostic requires exactly 64 selected operators")
        if operator_bank_permutation_views or operator_bank_canonical_probability != 1.0:
            raise ValueError(
                "V9 operator-set diagnostic requires fixed canonical data: "
                "permutation_views=false and canonical_probability=1"
            )
        if world_size != 1:
            raise ValueError("V9 operator-set diagnostic is a single-process run")
        operator_set_architecture = _validate_operator_set_overfit_architecture(
            cfg.model.big_vae.architecture_version
        )
        normalized_v4_operator_set = (
            operator_set_architecture == "latent_feedback_perirms_qknorm_v4"
            and str(cfg.model.big_vae.get("weight_input_normalization_kind", "none"))
            == "per_output_maxabs_q7"
        )
        if (
            operator_set_architecture == "latent_feedback_perirms_qknorm_v4"
            and not normalized_v4_operator_set
        ):
            raise ValueError(
                "exact64 V4 operator-set mode is isolated to the normalized-input experiment"
            )
        expected_operator_set_steps = 5000 if normalized_v4_operator_set else 1984
        if int(cfg.train.get("max_steps", -1)) != expected_operator_set_steps or int(
            cfg.train.get("stop_after_step", -1)
        ) != expected_operator_set_steps:
            raise ValueError(
                "operator-set diagnostic requires max_steps=stop_after_step="
                f"{expected_operator_set_steps} for architecture {operator_set_architecture}"
            )
        if int(cfg.train.get("slice_batch_size", -1)) != 6 or int(
            cfg.train.get("grad_accum_steps", -1)
        ) != 3:
            raise ValueError("V9 operator-set diagnostic requires physical B6 x accumulation3")
        if int(cfg.train.get("checkpoint_every", -1)) != expected_operator_set_steps or bool(
            cfg.train.get("checkpoint_latest_copy", True)
        ):
            raise ValueError(
                "operator-set diagnostic permits only the final checkpoint at step "
                f"{expected_operator_set_steps} and no latest copy"
            )
        resume_cfg = cfg.train.get("resume_state", {}) or {}
        if bool(resume_cfg.get("enabled", False)) or bool(resume_cfg.get("auto_resume", False)):
            raise ValueError("V9 operator-set diagnostic forbids resume-state checkpoints")
    v6_causal_raw = cfg.train.get("v6_causal_bundle", {})
    if v6_causal_raw is None:
        v6_causal_raw = {}
    if not isinstance(v6_causal_raw, (dict, DictConfig)):
        raise TypeError("train.v6_causal_bundle must be a mapping")
    v6_causal_cfg = validate_v6_causal_config(v6_causal_raw)
    v6_causal_enabled = bool(v6_causal_cfg.get("enabled", False))
    if v6_causal_enabled:
        if not two_operator_overfit_enabled:
            raise ValueError("V6 causal bundle requires the fixed two-operator diagnostic")
        if world_size != 1:
            raise ValueError("V6 causal bundle is a single-process diagnostic")
        expected_causal_architecture = (
            "latent_mandatory_cross_refresh_posfilm_v7"
            if v6_causal_cfg["arm"] == "mandatory_cross_refresh"
            else (
                "clean_content_readout_posfilm_v8"
                if v6_causal_cfg["arm"] == "clean_content_readout"
                else "latent_mandatory_bridge_posfilm_v6"
            )
        )
        if str(cfg.model.big_vae.architecture_version) != expected_causal_architecture:
            raise ValueError(f"causal arm requires architecture {expected_causal_architecture}")
        if bool(cfg.train.get("compile", False)):
            raise ValueError("V6 causal bundle requires train.compile=false for hook/intervention fidelity")
        expected_steps = 504 if v6_causal_cfg["arm"] == "cyclic_singleton" else 500
        if (
            int(cfg.train.get("max_steps", -1)) != expected_steps
            or int(cfg.train.get("stop_after_step", -1)) != expected_steps
        ):
            raise ValueError(f"V6 causal bundle arm {v6_causal_cfg['arm']} requires exactly {expected_steps} fresh steps")
        resume_cfg = cfg.train.get("resume_state", {}) or {}
        if bool(resume_cfg.get("enabled", False)) or bool(resume_cfg.get("auto_resume", False)):
            raise ValueError("V6 causal bundle forbids resume and rolling resume checkpoints")
    epsilon_fork_cfg = cfg.train.get("exact_optimizer_epsilon_fork", {})
    if epsilon_fork_cfg is None:
        epsilon_fork_cfg = {}
    if not isinstance(epsilon_fork_cfg, (dict, DictConfig)):
        raise TypeError("train.exact_optimizer_epsilon_fork must be a mapping")
    epsilon_fork_enabled = bool(epsilon_fork_cfg.get("enabled", False))
    if epsilon_fork_enabled and not two_operator_overfit_enabled:
        raise ValueError("exact optimizer epsilon fork requires the fixed two-operator diagnostic")
    if not 0.0 <= operator_bank_canonical_probability <= 1.0:
        raise ValueError("train.operator_bank.canonical_probability must lie in [0, 1]")
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
    if operator_bank_enabled and not operator_bank_pair_manifest:
        raise ValueError("train.operator_bank.pair_manifest must be set when train.operator_bank.enabled=true")
    if bool(operator_bank_cfg.get("approved_weightclip_contract", False)) and not operator_bank_pair_manifest_sha256:
        raise ValueError("approved operator-bank contract requires train.operator_bank.pair_manifest_sha256")
    if bool(operator_bank_cfg.get("approved_weightclip_contract", False)) and (
        not operator_bank_locality_audit_path or not operator_bank_locality_audit_sha256
    ):
        raise ValueError("approved operator-bank contract requires a SHA-bound locality audit")
    if operator_bank_enabled and (offline_dataset_enabled or synthetic_layer_enabled or preslicing_enabled):
        raise ValueError(
            "train.operator_bank.enabled=true is mutually exclusive with offline_dataset, preslicing, and synthetic_layer_source"
        )
    if operator_bank_enabled and is_distributed and not operator_bank_shard_by_rank:
        raise ValueError("distributed operator-bank mode requires train.operator_bank.shard_by_rank=true")
    if operator_bank_enabled:
        streaming_mode = "operator_bank"
        dataset_sharding = is_distributed
        use_broadcast = False
    elif synthetic_layer_enabled:
        dataset_sharding = False
        use_broadcast = False
    elif offline_dataset_enabled:
        streaming_mode = "presliced_big_vae" if preslicing_enabled else "offline_big_vae"
        active_offline_shard_by_rank = preslicing_shard_by_rank if preslicing_enabled else offline_dataset_shard_by_rank
        dataset_sharding = bool(cfg.train.get("use_dataset_sharding", True)) and is_distributed and active_offline_shard_by_rank
        use_broadcast = is_distributed and not dataset_sharding
    if preslicing_enabled and use_broadcast:
        raise ValueError("train.preslicing.enabled=true requires sharded loading in distributed mode")

    if dataset_sharding and not offline_dataset_enabled and not operator_bank_enabled:
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
    if operator_bank_enabled:
        logger.info(
            "Operator-bank dataset enabled: pair_manifest=%s pair_manifest_sha256=%s repeat=%s five_graph_gauge_views=%s "
            "canonical_probability=%.6f hot_shards=%s max_active_strata=%s shard_by_rank=%s use_broadcast=%s",
            operator_bank_pair_manifest,
            operator_bank_pair_manifest_sha256 or "<unchecked>",
            operator_bank_repeat,
            operator_bank_permutation_views,
            operator_bank_canonical_probability,
            operator_bank_hot_shards,
            operator_bank_max_active_strata,
            operator_bank_shard_by_rank,
            use_broadcast,
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
        operator_index_sampler: Any | None = None
        operator_tile_mixer: BalancedOperatorBankMixer | None = None

        with contextlib.ExitStack() as stack:
            if not synthetic_layer_enabled:
                if operator_bank_enabled:
                    operator_rank = rank if dataset_sharding else 0
                    operator_world_size = world_size if dataset_sharding else 1
                    if operator_set_overfit_enabled:
                        from training.big_vae.operator_set_overfit import (
                            canonical_operator_set_data_pipeline,
                        )

                        operator_context = canonical_operator_set_data_pipeline(
                            operator_bank_pair_manifest,
                            selected_operators=operator_set_overfit_selection,
                            expected_operator_count=64,
                            seed=seed,
                            hot_shards=operator_bank_hot_shards,
                            expected_pair_manifest_sha256=operator_bank_pair_manifest_sha256,
                            expected_selection_sha256=str(
                                operator_set_overfit_cfg.get("selection_sha256", "")
                            ),
                            expected_schedule_sha256=str(
                                operator_set_overfit_cfg.get("schedule_sha256", "")
                            ),
                            max_active_strata=operator_bank_max_active_strata,
                            max_active_bundle_bytes=operator_bank_max_active_bundle_bytes,
                        )
                    elif two_operator_overfit_enabled:
                        from training.big_vae.two_operator_overfit import two_operator_data_pipeline

                        operator_context = two_operator_data_pipeline(
                            operator_bank_pair_manifest,
                            selected_operators=two_operator_overfit_selection,
                            seed=seed,
                            hot_shards=operator_bank_hot_shards,
                            expected_pair_manifest_sha256=operator_bank_pair_manifest_sha256,
                            max_active_strata=operator_bank_max_active_strata,
                            max_active_bundle_bytes=operator_bank_max_active_bundle_bytes,
                        )
                    else:
                        operator_context = operator_bank_data_pipeline(
                            operator_bank_pair_manifest,
                            seed=seed,
                            repeat=operator_bank_repeat,
                            permutation_views=operator_bank_permutation_views,
                            canonical_probability=operator_bank_canonical_probability,
                            hot_shards=operator_bank_hot_shards,
                            expected_pair_manifest_sha256=operator_bank_pair_manifest_sha256,
                            rank=operator_rank,
                            world_size=operator_world_size,
                            max_active_strata=operator_bank_max_active_strata,
                            max_active_bundle_bytes=operator_bank_max_active_bundle_bytes,
                            logger=logger,
                        )
                    dataset, operator_index_sampler = stack.enter_context(operator_context)
                    if bool(operator_bank_cfg.get("approved_weightclip_contract", False)):
                        locality_audit_path = Path(operator_bank_locality_audit_path)
                        if (
                            not locality_audit_path.is_file()
                            or sha256_file(locality_audit_path) != operator_bank_locality_audit_sha256
                        ):
                            raise RuntimeError("approved operator-bank locality audit is missing or not SHA-bound")
                        sealed_locality_audit = json.loads(locality_audit_path.read_text(encoding="utf-8"))
                        for key, value in dataset.locality_audit.items():
                            if sealed_locality_audit.get(key) != value:
                                raise RuntimeError(
                                    f"approved operator-bank locality audit disagrees on {key}: "
                                    f"sealed={sealed_locality_audit.get(key)} runtime={value}"
                                )
                        if sealed_locality_audit.get("pair_manifest_sha256") != operator_bank_pair_manifest_sha256:
                            raise RuntimeError("approved operator-bank locality audit pair SHA mismatch")
                    loader_workers = max(0, int(operator_bank_cfg.get("loader_workers", 8)))
                    loader_batch_size = max(1, int(operator_bank_cfg.get("loader_batch_size", 4)))
                    loader_prefetch_factor = max(1, int(operator_bank_cfg.get("loader_prefetch_factor", 4)))
                    loader_persistent_workers = bool(operator_bank_cfg.get("loader_persistent_workers", True))
                    operator_loader_generator = torch.Generator(device="cpu").manual_seed(seed + 7_919)
                    inflight_bundle_bound = loader_batch_size * (
                        (loader_workers * loader_prefetch_factor + 1) if loader_workers > 0 else 1
                    )
                    inflight_bundle_bytes_bound = inflight_bundle_bound * int(dataset.max_bundle_tensor_bytes)
                    if inflight_bundle_bound > operator_bank_max_inflight_bundles:
                        raise RuntimeError(
                            "operator-bank DataLoader exceeds max_inflight_bundles: "
                            f"bound={inflight_bundle_bound} max={operator_bank_max_inflight_bundles}"
                        )
                    if inflight_bundle_bytes_bound > operator_bank_max_inflight_bundle_bytes:
                        raise RuntimeError(
                            "operator-bank DataLoader exceeds max_inflight_bundle_bytes: "
                            f"bound={inflight_bundle_bytes_bound} max={operator_bank_max_inflight_bundle_bytes}"
                        )
                    total_bundle_tensor_bytes_bound = (
                        int(dataset.active_tensor_bytes_bound) + inflight_bundle_bytes_bound
                    )
                    dataset_loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=(None if loader_batch_size <= 1 else loader_batch_size),
                        sampler=operator_index_sampler,
                        num_workers=loader_workers,
                        collate_fn=(_identity_sample_collate if loader_batch_size <= 1 else _sample_list_collate),
                        prefetch_factor=(loader_prefetch_factor if loader_workers > 0 else None),
                        persistent_workers=(loader_persistent_workers if loader_workers > 0 else False),
                        pin_memory=False,
                        worker_init_fn=_offline_loader_worker_init_fn,
                        # Keep DataLoader iterator creation from consuming the
                        # model/dropout process RNG after a resume restore.
                        generator=operator_loader_generator,
                    )
                    logger.info(
                        "Operator-bank step-addressed DataLoader prepared: workers=%s ipc_batch=%s "
                        "prefetch_factor=%s persistent=%s inflight_bundle_bound=%s "
                        "inflight_tensor_mib_bound=%.2f active_strata_tensor_mib_bound=%.2f "
                        "total_tensor_mib_bound=%.2f; iterator starts after resume cursor resolution",
                        loader_workers,
                        loader_batch_size,
                        loader_prefetch_factor,
                        loader_persistent_workers if loader_workers > 0 else False,
                        inflight_bundle_bound,
                        inflight_bundle_bytes_bound / (1024.0 * 1024.0),
                        dataset.active_tensor_bytes_bound / (1024.0 * 1024.0),
                        total_bundle_tensor_bytes_bound / (1024.0 * 1024.0),
                    )
                elif offline_dataset_enabled:
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
            v6_causal_state = prepare_v6_causal_model(model, v6_causal_cfg)
            v6_causal_selection_counts = [0] * 18
            v6_causal_presentation_counts = [0] * 18
            if v6_causal_enabled and rank == 0:
                write_json_immutable(
                    Path(str(v6_causal_cfg["ledger_path"])),
                    v6_causal_state.startup_ledger,
                )
            total_params, trainable_params, frozen_params = _parameter_count_summary(model)
            if operator_set_overfit_enabled:
                expected_total = int(operator_set_overfit_cfg.get("expected_total_parameters", -1))
                expected_trainable = int(
                    operator_set_overfit_cfg.get("expected_trainable_parameters", -1)
                )
                if (total_params, trainable_params) != (expected_total, expected_trainable):
                    raise RuntimeError(
                        "V9 exact64 parameter contract mismatch: "
                        f"actual_total={total_params} expected_total={expected_total} "
                        f"actual_trainable={trainable_params} expected_trainable={expected_trainable}"
                    )
                expected_active_encoder = int(
                    operator_set_overfit_cfg.get("expected_active_encoder_parameters", -1)
                )
                architecture = str(cfg.model.big_vae.architecture_version)
                active_encoder_prefixes = (
                    ("distribution_encoder.", "orthogonal_complement_encoder_v10.")
                    if architecture == "orthogonal_complement_tied_posfilm_v10"
                    else (
                        ("distribution_encoder.", "four_trunk_complement_encoder_v11.")
                        if architecture == "four_trunk_complement_v11"
                        else (
                            "distribution_encoder.",
                            "clean_content_readout_v8.",
                            "hybrid_content_readout_v9.",
                        )
                    )
                )
                active_encoder_params = sum(
                    parameter.numel()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and name.startswith(active_encoder_prefixes)
                )
                if (
                    expected_active_encoder >= 0
                    and active_encoder_params != expected_active_encoder
                ):
                    raise RuntimeError(
                        "exact64 active-encoder parameter contract mismatch: "
                        f"actual={active_encoder_params} expected={expected_active_encoder}"
                    )
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
            epsilon_fork_ledger: dict[str, Any] | None = None
            explicit_resume_state = str(
                resume_state_cfg.get("explicit_checkpoint", "") or ""
            ).strip()
            if explicit_resume_state and resume_state_auto_resume:
                raise RuntimeError(
                    "resume_state.explicit_checkpoint and auto_resume are mutually exclusive"
                )
            if explicit_resume_state and not resume_state_enabled:
                raise RuntimeError(
                    "resume_state.explicit_checkpoint requires resume_state.enabled=true"
                )
            if explicit_resume_state:
                resume_state_path = Path(explicit_resume_state)
                if not resume_state_path.is_file():
                    raise RuntimeError(
                        "explicit resume-state checkpoint is missing: "
                        f"{resume_state_path}"
                    )
                logger.info("Exact resume-state checkpoint selected: %s", resume_state_path)
            elif resume_state_enabled and resume_state_auto_resume:
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
                    load_rng_state=resume_state_load_policy["load_rng_state"],
                    load_step=resume_state_load_policy["load_step"],
                    expected_data_stream_contract=(
                        _operator_stream_contract(cfg, rank=rank, world_size=world_size)
                        if operator_bank_enabled
                        else None
                    ),
                    allowed_data_stream_transition=resume_state_cfg.get(
                        "operator_stream_transition", {}
                    ),
                    active_device=device,
                )
                scheduler_name = str(cfg.train.get("scheduler_name", "cosine")).strip().lower()
                if scheduler_name in {"constant", "constant_lr"}:
                    configured_lr = float(cfg.train.get("lr", 3e-4))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = configured_lr
                        param_group["initial_lr"] = configured_lr
                    logger.info(
                        "Constant scheduler resume: reset all optimizer group LRs to %.6e",
                        configured_lr,
                    )
                if epsilon_fork_enabled:
                    if resume_state_auto_resume or not explicit_resume_state:
                        raise RuntimeError(
                            "exact optimizer epsilon fork requires one explicit resume checkpoint and auto_resume=false"
                        )
                    if not all(resume_state_load_policy.values()):
                        raise RuntimeError(
                            "exact optimizer epsilon fork requires full model/optimizer/scheduler/scaler/RNG/step restore"
                        )
                    stream_contract = _operator_stream_contract(cfg, rank=rank, world_size=world_size)
                    epsilon_fork_ledger = apply_loaded_optimizer_epsilon_fork(
                        optimizer,
                        raw_config=epsilon_fork_cfg,
                        resume_checkpoint=resume_state_path,
                        resumed_step=resumed_training_step,
                        stop_after_step=int(cfg.train.get("stop_after_step", 0)),
                        slice_batch_size=int(cfg.train.get("slice_batch_size", 1)),
                        grad_accum_steps=int(cfg.train.get("grad_accum_steps", 1)),
                        stream_contract=stream_contract,
                    )
                    if rank == 0:
                        ledger_path = Path(str(epsilon_fork_cfg["startup_ledger_path"]))
                        write_json_immutable(ledger_path, epsilon_fork_ledger)
                        logger.info(
                            "Exact optimizer epsilon fork armed: source_step=%s target_eps=%s "
                            "logical_range=[%s,%s) ledger=%s",
                            resumed_training_step,
                            epsilon_fork_ledger["target_optimizer_group_epsilons"],
                            epsilon_fork_ledger["committed_logical_index_start"],
                            epsilon_fork_ledger["committed_logical_index_end_exclusive"],
                            ledger_path,
                        )
            elif resume_checkpoint:
                logger.info(
                    "Resume-state unavailable; loading model weights only from resume_checkpoint=%s",
                    resume_checkpoint,
                )
                _load_model_weights_from_checkpoint(model, resume_checkpoint, logger)
            else:
                logger.info("No resume source configured or found; starting training from step=0")
            if operator_bank_enabled:
                if operator_index_sampler is None or dataset_loader is None:
                    raise RuntimeError("operator-bank sampler/loader was not initialized")
                committed_samples = (
                    int(operator_bank_cfg.get("logical_index_offset", 0))
                    + int(resumed_training_step)
                    * max(1, int(cfg.train.get("grad_accum_steps", 1)))
                    * max(1, int(cfg.train.get("slice_batch_size", 1)))
                )
                operator_index_sampler.set_start_index(committed_samples)
                dataset_loader_iter = iter(dataset_loader)
                loader_batch_size = max(1, int(operator_bank_cfg.get("loader_batch_size", 4)))
                bundle_iter = (
                    dataset_loader_iter
                    if loader_batch_size <= 1
                    else _flatten_loader_batches(dataset_loader_iter)
                )
                operator_tile_mixer = BalancedOperatorBankMixer(
                    dataset,
                    bundle_iter,
                    start_index=committed_samples,
                )
                dataset_iter = operator_tile_mixer
                shutdown_workers = getattr(dataset_loader_iter, "_shutdown_workers", None)
                if callable(shutdown_workers):
                    stack.callback(shutdown_workers)
                logger.info(
                    "Operator-bank committed cursor restored: training_step=%s logical_sample_index=%s",
                    resumed_training_step,
                    committed_samples,
                )
            cudagraph_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
            use_cudagraph_step_begin = (
                bool(cfg.train.get("compile", False))
                and device.type == "cuda"
                and callable(cudagraph_step_begin)
            )

            max_steps = max(1, int(cfg.train.get("max_steps", 1000)))
            training_stop_step = _bounded_training_stop_step(
                max_steps,
                bounded_profile_spec,
                int(cfg.train.get("stop_after_step", 0) or 0),
            )
            if resumed_training_step >= training_stop_step:
                raise RuntimeError(
                    f"resume step {resumed_training_step} must be below training stop step "
                    f"{training_stop_step}"
                )
            if training_stop_step != max_steps:
                logger.info(
                    "Finite training stop enabled: stop_after_step=%s scheduler_horizon=%s",
                    training_stop_step,
                    max_steps,
                )
            overfit_success_cfg = two_operator_overfit_cfg.get("early_success", {})
            if overfit_success_cfg is None:
                overfit_success_cfg = {}
            if not isinstance(overfit_success_cfg, (dict, DictConfig)):
                raise TypeError("two_operator_overfit.early_success must be a mapping")
            overfit_success_enabled = two_operator_overfit_enabled and bool(
                overfit_success_cfg.get("enabled", False)
            )
            overfit_success_min_step = int(overfit_success_cfg.get("min_step", 100))
            overfit_success_consecutive = int(overfit_success_cfg.get("consecutive_evals", 2))
            overfit_success_dir_threshold = float(overfit_success_cfg.get("struct_dir_max", 0.05))
            overfit_success_scale_threshold = float(overfit_success_cfg.get("struct_scale_max", 0.01))
            overfit_success_nrmse_threshold = float(overfit_success_cfg.get("nrmse_max", 0.1))
            overfit_success_swap_delta = float(
                overfit_success_cfg.get(
                    "swap_dir_delta_min",
                    overfit_success_cfg.get("swap_total_delta_min", 0.1),
                )
            )
            overfit_success_mean_dir_threshold = overfit_success_cfg.get("matched_mean_dir_max")
            if overfit_success_enabled and (
                overfit_success_min_step < 1
                or overfit_success_consecutive < 1
                or overfit_success_dir_threshold < 0.0
                or overfit_success_scale_threshold < 0.0
                or overfit_success_nrmse_threshold < 0.0
                or overfit_success_swap_delta <= 0.0
                or (
                    overfit_success_mean_dir_threshold is not None
                    and float(overfit_success_mean_dir_threshold) <= 0.0
                )
            ):
                raise ValueError("invalid two-operator early-success contract")
            overfit_success_streak = 0
            if bounded_profile_spec is not None and resumed_training_step != 0:
                raise RuntimeError("bounded runtime profile must start from optimizer step zero")
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
            v11_enabled = (
                str(cfg.model.big_vae.architecture_version)
                == "four_trunk_complement_v11"
            )
            v11_complement_coef = float(
                cfg.train.get("v11_complement_loss_weight", 0.0)
            )
            if v11_enabled and v11_complement_coef != 0.1:
                raise ValueError("V11 freezes train.v11_complement_loss_weight=0.1")
            if not v11_enabled and v11_complement_coef != 0.0:
                raise ValueError("V11 complement loss may only be enabled for V11")
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
            two_operator_evaluator = None
            v11_step0_metrics_baseline: dict[str, float] | None = None
            if operator_set_overfit_enabled:
                if dataset is None or not hasattr(dataset, "source"):
                    raise RuntimeError("V9 operator-set diagnostic lost its source")
                from training.big_vae.operator_set_overfit import OperatorSetOverfitEvaluator

                two_operator_evaluator = OperatorSetOverfitEvaluator(
                    source=dataset.source,
                    output_path=str(operator_set_overfit_cfg["metrics_path"]),
                    every_steps=int(operator_set_overfit_cfg.get("eval_every_steps", 256)),
                    patch_size=patch_size_for_slice,
                    gamma=struct_gamma,
                    lambda_dir=struct_lambda_dir,
                    lambda_scale=struct_lambda_scale,
                    huber_delta=struct_huber_delta,
                )
                with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                    initial_operator_set_metrics = two_operator_evaluator.evaluate(
                        model=model,
                        step=0,
                        eval_batch_size=int(operator_set_overfit_cfg.get("eval_batch_size", 8)),
                    )
                logger.info(
                    "operator_set_eval %s",
                    json.dumps(initial_operator_set_metrics, sort_keys=True),
                )
                if str(cfg.model.big_vae.architecture_version) == "carrier_mean_content_posfilm_v9a":
                    from training.big_vae.operator_set_overfit import (
                        validate_v9a_step0_metrics,
                    )

                    validate_v9a_step0_metrics(initial_operator_set_metrics)
                elif str(cfg.model.big_vae.architecture_version) == "orthogonal_complement_tied_posfilm_v10":
                    from training.big_vae.operator_set_overfit import (
                        validate_v10_step0_metrics,
                    )

                    validate_v10_step0_metrics(initial_operator_set_metrics)
                elif str(cfg.model.big_vae.architecture_version) == "four_trunk_complement_v11":
                    from training.big_vae.operator_set_overfit import (
                        validate_v11_step0_metrics,
                    )

                    validate_v11_step0_metrics(initial_operator_set_metrics)
                    v11_step0_metrics_baseline = {
                        "v11_complement_nrmse_mean": float(
                            initial_operator_set_metrics["v11_complement_nrmse_mean"]
                        ),
                        "v11_complement_normalized_mse_mean": float(
                            initial_operator_set_metrics[
                                "v11_complement_normalized_mse_mean"
                            ]
                        ),
                        "v11_floor_mean_dir": float(
                            initial_operator_set_metrics["v11_floor_mean_dir"]
                        ),
                    }
                    two_operator_evaluator.v11_step0_complement_normalized_mse = float(
                        v11_step0_metrics_baseline[
                            "v11_complement_normalized_mse_mean"
                        ]
                    )
            elif two_operator_overfit_enabled:
                if slice_batch_size * grad_accum_steps != 18 or len(dataset.source) != 18:
                    raise ValueError(
                        "two-operator overfit requires exactly one 18-tile full-set optimizer batch"
                    )
                if dataset is None or not hasattr(dataset, "source"):
                    raise RuntimeError("two-operator overfit lost its diagnostic operator source")
                from training.big_vae.two_operator_overfit import TwoOperatorOverfitEvaluator

                two_operator_evaluator = TwoOperatorOverfitEvaluator(
                    source=dataset.source,
                    output_path=str(two_operator_overfit_cfg["metrics_path"]),
                    every_steps=int(two_operator_overfit_cfg.get("eval_every_steps", 500)),
                    patch_size=patch_size_for_slice,
                    gamma=struct_gamma,
                    lambda_dir=struct_lambda_dir,
                    lambda_scale=struct_lambda_scale,
                    huber_delta=struct_huber_delta,
                    free_weight_steps=int(two_operator_overfit_cfg.get("free_weight_control_steps", 50)),
                    free_weight_lr=float(two_operator_overfit_cfg.get("free_weight_control_lr", 1.0e-2)),
                    include_identity_diagnostics=(
                        epsilon_fork_enabled
                        or bool(two_operator_overfit_cfg.get("include_identity_diagnostics", False))
                    ),
                )
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
            offline_batch_prefetch_reasons = _prepared_batch_prefetch_blockers(
                requested=offline_batch_prefetch_requested,
                prepared_source_enabled=offline_dataset_enabled or operator_bank_enabled,
                fixed_training_batch_enabled=fixed_training_batch_enabled,
                synthetic_layer_enabled=synthetic_layer_enabled,
                use_broadcast=use_broadcast,
            )
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
                        "load_scaler_state=%s load_rng_state=%s load_step=%s",
                        resume_state_dir,
                        resume_state_save_every,
                        resume_state_auto_resume,
                        resume_state_load_policy["load_model_state"],
                        resume_state_load_policy["load_optimizer_state"],
                        resume_state_load_policy["load_scheduler_state"],
                        resume_state_load_policy["load_scaler_state"],
                        resume_state_load_policy["load_rng_state"],
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
            v11_complement_window = 0.0
            kl_window = 0.0
            mu_rms_window = 0.0
            mu_abs_mean_window = 0.0
            posterior_std_mean_window = 0.0
            posterior_std_rms_window = 0.0
            posterior_logvar_mean_window = 0.0
            mu_batch_var_window = 0.0
            mu_active_fraction_window = 0.0
            kl_mean_part_window = 0.0
            kl_variance_part_window = 0.0
            logvar_min_fraction_window = 0.0
            logvar_max_fraction_window = 0.0
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
            prefetch_stop_step = _prefetch_request_stop_step(
                training_stop_step,
                spec=bounded_profile_spec,
                prefetch_active=offline_batch_prefetch_active,
                queue_size=offline_batch_prefetch_queue_size,
            )
            prefetch_request_iter = iter(
                (step_idx, micro_idx)
                for step_idx in range(resumed_training_step, prefetch_stop_step)
                for micro_idx in range(grad_accum_steps)
            )

            def _prepare_training_micro_batch_cpu(*, step_idx: int, micro_idx: int) -> PreparedTrainingBatch:
                nonlocal current_source_samples
                nonlocal current_source_states
                nonlocal current_source_round_robin_offset
                nonlocal source_mixing_shortfall_logged

                build_t0 = time.perf_counter()
                if operator_bank_enabled or preslicing_enabled:
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
                    pin_t0 = time.perf_counter()
                    batch = _pin_prepared_training_batch(batch)
                    batch.pin_time_s = time.perf_counter() - pin_t0
                return batch

            def _prepare_training_micro_batch_cpu_for_prefetch() -> PreparedTrainingBatch:
                step_idx, micro_idx = next(prefetch_request_iter)
                return _prepare_training_micro_batch_cpu(step_idx=step_idx, micro_idx=micro_idx)

            gradient_noise_monitor: GradientNoiseMonitor | None = None
            gradient_noise_cfg = cfg.train.get("telemetry", {}).get("gradient_noise_monitor", {})
            if gradient_noise_cfg is None:
                gradient_noise_cfg = {}
            if not isinstance(gradient_noise_cfg, (dict, DictConfig)):
                raise TypeError("train.telemetry.gradient_noise_monitor must be a mapping")
            if bool(gradient_noise_cfg.get("enabled", False)):
                if rank != 0 or world_size != 1:
                    raise RuntimeError("gradient-noise monitor currently requires rank0/world_size1")
                gradient_noise_monitor = GradientNoiseMonitor(
                    model=model,
                    cfg=cfg,
                    device=device,
                    logger=logger,
                    csv_path=Path(str(gradient_noise_cfg["csv_path"])),
                    sample_manifest_path=Path(str(gradient_noise_cfg["sample_manifest_path"])),
                    every_steps=int(gradient_noise_cfg.get("every_steps", 1_000)),
                    start_logical_index=int(gradient_noise_cfg.get("start_logical_index", 7_000_000)),
                    panel_size=int(gradient_noise_cfg.get("panel_size", 65_536)),
                )
                logger.info(
                    "Gradient-noise monitor armed: every_steps=%s B=128 blocks=4 pairwise=6 csv=%s",
                    int(gradient_noise_cfg.get("every_steps", 1_000)),
                    Path(str(gradient_noise_cfg["csv_path"])),
                )
                if bool(gradient_noise_cfg.get("run_at_start", False)):
                    noise_metrics = gradient_noise_monitor.run(step=resumed_training_step)
                    if comet_tracker is not None and comet_tracker.enabled:
                        comet_tracker.log_metrics(noise_metrics, step=resumed_training_step)

            offline_batch_prefetcher: BackgroundPrefetcher[PreparedTrainingBatch] | None = None
            if offline_batch_prefetch_active:
                offline_batch_prefetcher = BackgroundPrefetcher(
                    build_fn=_prepare_training_micro_batch_cpu_for_prefetch,
                    queue_size=offline_batch_prefetch_queue_size,
                    name=f"offline_big_vae_prefetch_rank_{rank}",
                )
                offline_batch_prefetcher.start()

            profile_recorder: _BoundedRuntimeProfileRecorder | None = None
            if bounded_profile_spec is not None:
                preflight = cfg.get("weightclip_runtime_profile_preflight", {})
                if not isinstance(preflight, (dict, DictConfig)):
                    raise TypeError("weightclip_runtime_profile_preflight must be a mapping")
                resolved_config_sha256 = str(
                    os.environ.get("WEIGHTCLIP_AE_PROFILE_RESOLVED_SHA256", "")
                )
                profile_contract_sha256 = str(
                    os.environ.get("WEIGHTCLIP_AE_PROFILE_CONTRACT_SHA256", "")
                )
                if len(resolved_config_sha256) != 64:
                    raise RuntimeError("bounded runtime profile requires the immutable resolved-config SHA-256")
                if len(profile_contract_sha256) != 64:
                    raise RuntimeError("bounded runtime profile requires the immutable profile-contract SHA-256")
                if operator_tile_mixer is None or dataset is None:
                    raise RuntimeError("bounded runtime profile requires the active production locality mixer")
                active_config = OmegaConf.to_container(cfg, resolve=True)
                actual_model_config = OmegaConf.to_container(cfg.model, resolve=True)
                if not isinstance(active_config, dict) or not isinstance(actual_model_config, dict):
                    raise RuntimeError("bounded runtime profile could not canonicalize the active worker config")
                input_path_ledger = {
                    "schema": "operator_bank_production_input_path_v1",
                    "profile_mode": bounded_profile_spec.profile_mode,
                    "pair_manifest": operator_bank_pair_manifest,
                    "pair_manifest_sha256": operator_bank_pair_manifest_sha256,
                    "locality_audit_path": operator_bank_locality_audit_path,
                    "locality_audit_sha256": operator_bank_locality_audit_sha256,
                    "loader": {
                        "workers": int(operator_bank_cfg.get("loader_workers", 8)),
                        "ipc_batch_size": int(operator_bank_cfg.get("loader_batch_size", 1)),
                        "prefetch_factor": int(operator_bank_cfg.get("loader_prefetch_factor", 2)),
                        "collate": "identity" if int(operator_bank_cfg.get("loader_batch_size", 1)) == 1 else "list",
                    },
                    "memory_bounds": {
                        "active_strata_tensor_bytes": int(dataset.active_tensor_bytes_bound),
                        "max_active_tensor_bytes": int(operator_bank_max_active_bundle_bytes),
                        "inflight_bundle_count": int(inflight_bundle_bound),
                        "inflight_tensor_bytes": int(inflight_bundle_bytes_bound),
                        "max_inflight_bundles": int(operator_bank_max_inflight_bundles),
                        "max_inflight_tensor_bytes": int(operator_bank_max_inflight_bundle_bytes),
                        "total_active_plus_inflight_tensor_bytes": int(total_bundle_tensor_bytes_bound),
                    },
                    "mixer": operator_tile_mixer.telemetry(),
                    "microbatch": {
                        "slice_batch_size": int(slice_batch_size),
                        "grad_accum_steps": int(grad_accum_steps),
                        "pin_memory": bool(offline_batch_prefetch_pin_memory),
                        "background_prefetch": bool(offline_batch_prefetch_active),
                        "prepared_batch_queue_size": int(offline_batch_prefetch_queue_size),
                        "cpu_threads": int(offline_batch_prefetch_cpu_threads),
                        "consumed_stop_step": int(training_stop_step),
                        "profile_prefetch_request_stop_step": int(prefetch_stop_step),
                        "profile_prefetch_tail_steps": int(prefetch_stop_step - training_stop_step),
                    },
                    "stages": [
                        "operator_bank_bundle_dataset",
                        "full_operator_gauge_bundle_worker_materialization",
                        "dataloader_ipc",
                        "main_process_stratum_preserving_locality_mixer",
                        "fetch_presliced_training_batch_cpu",
                        "pin_memory",
                        "five_tensor_nonblocking_h2d_enqueue",
                        "production_model_loss_backward_optimizer_scheduler",
                    ],
                }
                profile_recorder = _BoundedRuntimeProfileRecorder(
                    bounded_profile_spec,
                    device=device,
                    candidate=str(preflight.get("candidate", "")),
                    model_fingerprint_sha256=str(preflight.get("model_fingerprint_sha256", "")),
                    source_implementation_seal_sha256=str(
                        preflight.get("source_implementation_seal_sha256", "")
                    ),
                    candidate_artifact_set_sha256=str(
                        preflight.get("candidate_artifact_set_sha256", "")
                    ),
                    candidate_report_sha256=str(preflight.get("candidate_report_sha256", "")),
                    candidate_index_sha256=str(preflight.get("candidate_index_sha256", "")),
                    pair_manifest=operator_bank_pair_manifest,
                    pair_manifest_sha256=operator_bank_pair_manifest_sha256,
                    resolved_config_sha256=resolved_config_sha256,
                    profile_contract_sha256=profile_contract_sha256,
                    active_config=active_config,
                    actual_model_config=actual_model_config,
                    actual_trainable_parameters=sum(
                        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
                    ),
                    actual_total_parameters=sum(int(parameter.numel()) for parameter in model.parameters()),
                    input_path_ledger=input_path_ledger,
                    worker_started_s=worker_started_s,
                )
                logger.info(
                    "Bounded AE runtime profile armed: warmup=%s measured=%s hard_cap=%s "
                    "production_scheduler_horizon=%s output=%s",
                    bounded_profile_spec.warmup_steps,
                    bounded_profile_spec.measured_steps,
                    _BOUNDED_RUNTIME_PROFILE_MAX_STEPS,
                    max_steps,
                    bounded_profile_spec.output_dir,
                )

            for step_idx in range(resumed_training_step, training_stop_step):
                global_step = step_idx + 1
                if (
                    global_step == 1
                    and str(cfg.model.big_vae.architecture_version)
                    == "four_trunk_complement_v11"
                ):
                    raw_v11_model = model.module if hasattr(model, "module") else model
                    raw_v11_model = (
                        raw_v11_model._orig_mod
                        if hasattr(raw_v11_model, "_orig_mod")
                        else raw_v11_model
                    )
                    for trunk in raw_v11_model.four_trunk_complement_encoder_v11.trunks:
                        trunk.code_tensors_for_gradient.clear()
                        trunk.retain_code_gradient = True
                overfit_eval_microbatches: list[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]
                ] = []
                collect_overfit_eval = bool(
                    two_operator_evaluator is not None
                    and (
                        v6_causal_evaluation_due(
                            step=global_step,
                            every_steps=int(two_operator_overfit_cfg.get("eval_every_steps", 500)),
                            final_step=training_stop_step,
                        )
                        if v6_causal_enabled
                        else (
                            two_operator_evaluator.should_run(
                                global_step,
                                final_step=training_stop_step,
                            )
                            if operator_set_overfit_enabled
                            else two_operator_evaluator.should_run(global_step)
                        )
                    )
                )
                if profile_recorder is not None:
                    profile_recorder.begin_step(global_step)
                    if operator_tile_mixer is None:
                        raise RuntimeError("bounded runtime profile lost the production locality mixer")
                profile_step_consumed_logical_indices: list[int] = []
                profile_step_data_wait_s = 0.0
                profile_step_data_build_s = 0.0
                profile_step_pin_s = 0.0
                profile_step_h2d_enqueue_s = 0.0
                profile_step_prefetch_depth = 0.0
                profile_step_h2d_bytes = 0
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
                v11_complement_acc = 0.0
                kl_acc = 0.0
                mu_rms_acc = 0.0
                mu_abs_mean_acc = 0.0
                posterior_std_mean_acc = 0.0
                posterior_std_rms_acc = 0.0
                posterior_logvar_mean_acc = 0.0
                mu_batch_var_acc = 0.0
                mu_active_fraction_acc = 0.0
                kl_mean_part_acc = 0.0
                kl_variance_part_acc = 0.0
                logvar_min_fraction_acc = 0.0
                logvar_max_fraction_acc = 0.0
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
                    current_batch_pin_s = 0.0
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
                        if profile_recorder is not None:
                            profile_recorder.validate_input_batch(
                                prepared_batch,
                                expected_batch_size=slice_batch_size,
                            )
                            if len(prepared_batch.logical_indices) != slice_batch_size:
                                raise RuntimeError(
                                    "bounded runtime profile requires exact logical identities on every consumed tile"
                                )
                            profile_step_consumed_logical_indices.extend(prepared_batch.logical_indices)
                        current_batch_build_s = float(prepared_batch.build_time_s)
                        current_batch_pin_s = float(prepared_batch.pin_time_s)
                        current_batch_source_diversity = dict(prepared_batch.source_diversity)
                        profile_step_h2d_bytes += sum(
                            int(tensor.numel() * tensor.element_size())
                            for tensor in (
                                prepared_batch.W,
                                prepared_batch.x,
                                prepared_batch.x_mask,
                                prepared_batch.d_in_mask,
                                prepared_batch.d_out_mask,
                            )
                        )
                        h2d_t0 = time.perf_counter()
                        if profile_recorder is not None and micro_idx == 0:
                            profile_recorder.begin_cuda_work(global_step)
                        W_s = prepared_batch.W.to(device=device, non_blocking=True)
                        x_s = prepared_batch.x.to(device=device, non_blocking=True)
                        x_mask_s = prepared_batch.x_mask.to(device=device, non_blocking=True)
                        d_in_mask_s = prepared_batch.d_in_mask.to(device=device, non_blocking=True)
                        d_out_mask_s = prepared_batch.d_out_mask.to(device=device, non_blocking=True)
                        if collect_overfit_eval and not operator_set_overfit_enabled:
                            if len(prepared_batch.logical_indices) != slice_batch_size:
                                raise RuntimeError("two-operator evaluation requires every logical tile identity")
                            overfit_eval_microbatches.append(
                                (
                                    W_s.detach(),
                                    x_s.detach(),
                                    x_mask_s.detach(),
                                    d_in_mask_s.detach(),
                                    d_out_mask_s.detach(),
                                    tuple(prepared_batch.logical_indices),
                                )
                            )
                        if v6_causal_enabled and v6_causal_cfg["arm"] in {
                            "alternating_homogeneous",
                            "cyclic_singleton",
                        }:
                            transformed, homogeneous_ledger = transform_v6_causal_training_tensors(
                                (W_s, x_s, x_mask_s, d_in_mask_s, d_out_mask_s),
                                prepared_batch.logical_indices,
                                step=global_step,
                                arm=str(v6_causal_cfg["arm"]),
                                first_operator=int(v6_causal_cfg["homogeneous_first_operator"]),
                            )
                            W_s, x_s, x_mask_s, d_in_mask_s, d_out_mask_s = transformed
                            if rank == 0 and global_step in {1, 2}:
                                logger.info(
                                    "v6_causal_homogeneous %s",
                                    json.dumps(homogeneous_ledger, sort_keys=True),
                                )
                            if v6_causal_cfg["arm"] == "cyclic_singleton":
                                selected_position = int(homogeneous_ledger["selected_source_position"])
                                v6_causal_selection_counts[selected_position] += 1
                                v6_causal_presentation_counts[selected_position] += 18
                            else:
                                for selected_position in homogeneous_ledger["source_positions"]:
                                    v6_causal_selection_counts[int(selected_position)] += 1
                                    v6_causal_presentation_counts[int(selected_position)] += 2
                        current_h2d_enqueue_s = time.perf_counter() - h2d_t0
                    data_build_window_s += float(current_batch_build_s)
                    data_wait_window_s += float(current_prefetch_wait_s)
                    data_h2d_window_s += float(current_h2d_enqueue_s)
                    data_prefetch_depth_window += float(current_prefetch_depth)
                    profile_step_data_build_s += float(current_batch_build_s)
                    profile_step_pin_s += float(current_batch_pin_s)
                    profile_step_data_wait_s += float(current_prefetch_wait_s)
                    profile_step_h2d_enqueue_s += float(current_h2d_enqueue_s)
                    profile_step_prefetch_depth += float(current_prefetch_depth)
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
                            if (
                                v6_causal_enabled
                                and v6_causal_state.centered_value is not None
                                and v6_causal_state.centered_value.calibration is not None
                                and not getattr(v6_causal_state, "_gain_ledger_written", False)
                            ):
                                gain_path = Path(str(v6_causal_cfg["ledger_path"])).with_name(
                                    "v6_causal_centered_value_gain.json"
                                )
                                write_json_immutable(
                                    gain_path,
                                    {
                                        "schema": "weightclip_ae_v6_centered_value_gain_v1",
                                        "arm": "centered_value",
                                        **v6_causal_state.centered_value.calibration,
                                    },
                                )
                                setattr(v6_causal_state, "_gain_ledger_written", True)
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
                            if v11_enabled:
                                from training.big_vae.operator_set_overfit import (
                                    v11_complement_normalized_mse,
                                )

                                v11_complement_loss = v11_complement_normalized_mse(
                                    W_s,
                                    W_hat,
                                    mu,
                                    d_in_mask=d_in_mask_s,
                                    d_out_mask=d_out_mask_s,
                                )
                            else:
                                v11_complement_loss = structural_loss.new_zeros(())
                            if use_latent_sampling:
                                kl_loss = _compute_model_latent_kl(model, mu, logvar)
                            else:
                                kl_loss = mu.new_zeros(())
                            total_loss = mu.new_zeros(())
                            if behavioral_coef != 0.0:
                                total_loss = total_loss + behavioral_coef * behavioral_loss
                            if structural_coef != 0.0:
                                total_loss = total_loss + structural_coef * structural_loss
                            if v11_complement_coef != 0.0:
                                total_loss = (
                                    total_loss
                                    + v11_complement_coef * v11_complement_loss
                                )
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
                                    "v11_complement": _scalar_debug_value(
                                        v11_complement_loss
                                    ),
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
                    v11_complement_acc += float(v11_complement_loss.detach().item())
                    kl_acc += float(kl_loss.detach().item())
                    mu_f = mu.detach().to(dtype=torch.float32)
                    logvar_f = logvar.detach().to(dtype=torch.float32)
                    posterior_std_f = torch.exp(0.5 * logvar_f)
                    mu_rms_acc += float(mu_f.square().mean().sqrt().item())
                    mu_abs_mean_acc += float(mu_f.abs().mean().item())
                    posterior_std_mean_acc += float(posterior_std_f.mean().item())
                    posterior_std_rms_acc += float(posterior_std_f.square().mean().sqrt().item())
                    posterior_logvar_mean_acc += float(logvar_f.mean().item())
                    mu_coordinate_var = mu_f.var(dim=0, unbiased=False)
                    mu_batch_var_acc += float(mu_coordinate_var.mean().item())
                    mu_active_fraction_acc += float((mu_coordinate_var > 1.0e-4).float().mean().item())
                    kl_mean_part_acc += float((0.5 * mu_f.square()).mean().item())
                    kl_variance_part_acc += float(
                        (0.5 * (torch.exp(logvar_f) - 1.0 - logvar_f)).mean().item()
                    )
                    configured_logvar_min = float(cfg_holder.cfg.big_vae.latent_sampling_logvar_min)
                    configured_logvar_max = float(cfg_holder.cfg.big_vae.latent_sampling_logvar_max)
                    logvar_min_fraction_acc += float((logvar_f <= configured_logvar_min).float().mean().item())
                    logvar_max_fraction_acc += float((logvar_f >= configured_logvar_max).float().mean().item())
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

                if (
                    operator_set_overfit_enabled
                    and str(cfg.model.big_vae.architecture_version)
                    == "carrier_mean_content_posfilm_v9a"
                    and global_step == 1
                ):
                    from training.big_vae.operator_set_overfit import (
                        validate_v9a_step1_gradients,
                    )

                    adam_eps_values = {
                        float(group.get("eps", float("nan")))
                        for group in optimizer.param_groups
                    }
                    if len(adam_eps_values) != 1:
                        raise RuntimeError(
                            f"V9-A step-1 requires one exact Adam eps, got {adam_eps_values}"
                        )
                    v9a_grad_ledger = validate_v9a_step1_gradients(
                        model,
                        adam_eps=next(iter(adam_eps_values)),
                    )
                    v9a_grad_path = Path(
                        str(operator_set_overfit_cfg["metrics_path"])
                    ).with_name("v9a_exact_b18_step1_gradients.json")
                    v9a_grad_path.write_text(
                        json.dumps(v9a_grad_ledger, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    logger.info("V9-A exact B18 step-1 gradients PASS: %s", v9a_grad_path)

                if (
                    operator_set_overfit_enabled
                    and str(cfg.model.big_vae.architecture_version) in {
                        "orthogonal_complement_tied_posfilm_v10",
                        "four_trunk_complement_v11",
                    }
                    and global_step == 1
                ):
                    from training.big_vae.operator_set_overfit import (
                        validate_v10_step1_gradients,
                        validate_v11_step1_gradients,
                    )

                    adam_eps_values = {
                        float(group.get("eps", float("nan")))
                        for group in optimizer.param_groups
                    }
                    learning_rate_values = {
                        float(group.get("lr", float("nan")))
                        for group in optimizer.param_groups
                    }
                    weight_decay_values = {
                        float(group.get("weight_decay", float("nan")))
                        for group in optimizer.param_groups
                    }
                    if (
                        len(adam_eps_values) != 1
                        or len(learning_rate_values) != 1
                        or len(weight_decay_values) != 1
                    ):
                        raise RuntimeError(
                            "V10 step-1 requires one exact optimizer contract, got "
                            f"eps={adam_eps_values} lr={learning_rate_values} "
                            f"weight_decay={weight_decay_values}"
                        )
                    is_v11 = (
                        str(cfg.model.big_vae.architecture_version)
                        == "four_trunk_complement_v11"
                    )
                    gradient_validator = (
                        validate_v11_step1_gradients
                        if is_v11
                        else validate_v10_step1_gradients
                    )
                    v10_grad_ledger = gradient_validator(
                        model,
                        adam_eps=next(iter(adam_eps_values)),
                        learning_rate=next(iter(learning_rate_values)),
                        weight_decay=next(iter(weight_decay_values)),
                    )
                    v10_grad_path = Path(
                        str(operator_set_overfit_cfg["metrics_path"])
                    ).with_name(
                        "v11_exact_b18_step1_gradients_v1.json"
                        if is_v11
                        else "v10_exact_b18_step1_gradients_v2.json"
                    )
                    v10_grad_path.write_text(
                        json.dumps(v10_grad_ledger, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if not bool(v10_grad_ledger["passed"]):
                        raise RuntimeError(
                            "V10/V11 exact B18 step-1 gradient gate failed after complete ledger: "
                            f"{v10_grad_ledger['failures'][:16]}"
                        )
                    logger.info("V10/V11 exact B18 step-1 gradients PASS: %s", v10_grad_path)

                if (
                    v6_causal_enabled
                    and v6_causal_cfg["arm"] == "clean_content_readout"
                    and global_step == 1
                ):
                    if two_operator_evaluator is None or len(overfit_eval_microbatches) != grad_accum_steps:
                        raise RuntimeError("V8 step-0 preflight requires the exact captured B18 microbatches")
                    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                        v8_step0_metrics = two_operator_evaluator.evaluate(
                            model=model,
                            step=0,
                            microbatches=overfit_eval_microbatches,
                        )
                    adam_eps_values = {
                        float(group.get("eps", float("nan"))) for group in optimizer.param_groups
                    }
                    if len(adam_eps_values) != 1:
                        raise RuntimeError(f"V8 step-0 requires one exact Adam eps, got {adam_eps_values}")
                    v8_step0_ledger = validate_v8_step0_preflight(
                        model,
                        v8_step0_metrics,
                        adam_eps=next(iter(adam_eps_values)),
                    )
                    v8_step0_path = Path(str(v6_causal_cfg["ledger_path"])).with_name(
                        "v8_exact_b18_step0_preflight.json"
                    )
                    v8_step0_path.parent.mkdir(parents=True, exist_ok=True)
                    v8_step0_path.write_text(
                        json.dumps(v8_step0_ledger, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    logger.info(
                        "V8 exact B18 step-0 preflight PASS before optimizer: %s",
                        v8_step0_path,
                    )

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

                if gradient_noise_monitor is not None and gradient_noise_monitor.should_run(global_step):
                    noise_metrics = gradient_noise_monitor.run(step=global_step)
                    if comet_tracker is not None and comet_tracker.enabled:
                        comet_tracker.log_metrics(noise_metrics, step=global_step)

                step_loss = loss_acc / grad_accum_steps
                step_behavioral = behavioral_acc / grad_accum_steps
                step_behavioral_operator = behavioral_operator_acc / grad_accum_steps
                step_behavioral_dir = behavioral_dir_acc / grad_accum_steps
                step_behavioral_scale = behavioral_scale_acc / grad_accum_steps
                step_structural = structural_acc / grad_accum_steps
                step_v11_complement = v11_complement_acc / grad_accum_steps
                step_kl = kl_acc / grad_accum_steps
                step_mu_rms = mu_rms_acc / grad_accum_steps
                step_mu_abs_mean = mu_abs_mean_acc / grad_accum_steps
                step_posterior_std_mean = posterior_std_mean_acc / grad_accum_steps
                step_posterior_std_rms = posterior_std_rms_acc / grad_accum_steps
                step_posterior_logvar_mean = posterior_logvar_mean_acc / grad_accum_steps
                step_mu_batch_var = mu_batch_var_acc / grad_accum_steps
                step_mu_active_fraction = mu_active_fraction_acc / grad_accum_steps
                step_kl_mean_part = kl_mean_part_acc / grad_accum_steps
                step_kl_variance_part = kl_variance_part_acc / grad_accum_steps
                step_logvar_min_fraction = logvar_min_fraction_acc / grad_accum_steps
                step_logvar_max_fraction = logvar_max_fraction_acc / grad_accum_steps
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
                        step_mu_rms,
                        step_mu_abs_mean,
                        step_posterior_std_mean,
                        step_posterior_std_rms,
                        step_posterior_logvar_mean,
                        step_mu_batch_var,
                        step_mu_active_fraction,
                        step_kl_mean_part,
                        step_kl_variance_part,
                        step_logvar_min_fraction,
                        step_logvar_max_fraction,
                        step_v11_complement,
                    ],
                    dtype=torch.float32, device=device,
                )
                if is_distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    stats /= float(world_size)

                overfit_success_reached = False
                if collect_overfit_eval:
                    if two_operator_evaluator is None:
                        raise AssertionError("two-operator evaluator disappeared")
                    if (
                        not operator_set_overfit_enabled
                        and len(overfit_eval_microbatches) != grad_accum_steps
                    ):
                        raise RuntimeError("two-operator evaluator did not capture every optimizer microbatch")
                    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                        overfit_metrics = two_operator_evaluator.evaluate(
                            model=model,
                            step=global_step,
                            microbatches=(
                                None if operator_set_overfit_enabled else overfit_eval_microbatches
                            ),
                            **(
                                {
                                    "eval_batch_size": int(
                                        operator_set_overfit_cfg.get("eval_batch_size", 8)
                                    )
                                }
                                if operator_set_overfit_enabled
                                else {}
                            ),
                        )
                    logger.info("two_operator_eval %s", json.dumps(overfit_metrics, sort_keys=True))
                    if (
                        operator_set_overfit_enabled
                        and str(cfg.model.big_vae.architecture_version) in {
                            "orthogonal_complement_tied_posfilm_v10",
                            "four_trunk_complement_v11",
                        }
                    ):
                        from training.big_vae.operator_set_overfit import (
                            validate_v10_geometry_metrics,
                            validate_v11_geometry_metrics,
                        )

                        if (
                            str(cfg.model.big_vae.architecture_version)
                            == "four_trunk_complement_v11"
                        ):
                            validate_v11_geometry_metrics(overfit_metrics)
                            if global_step == 512:
                                if v11_step0_metrics_baseline is None:
                                    raise RuntimeError("V11 step-512 gate lost its step-0 baseline")
                                from training.big_vae.operator_set_overfit import (
                                    v11_scientific_contract_failures,
                                )

                                failures = v11_scientific_contract_failures(
                                    overfit_metrics,
                                    step0_complement_normalized_mse=float(
                                        v11_step0_metrics_baseline[
                                            "v11_complement_normalized_mse_mean"
                                        ]
                                    ),
                                    final=False,
                                )
                                if failures:
                                    raise RuntimeError(
                                        "V11 precommitted step-512 scientific gate failed: "
                                        + "; ".join(failures)
                                    )
                        else:
                            validate_v10_geometry_metrics(overfit_metrics)
                    if comet_tracker is not None and comet_tracker.enabled:
                        comet_tracker.log_metrics(
                            {f"two_operator/{key}": value for key, value in overfit_metrics.items()},
                            step=global_step,
                        )
                    if overfit_success_enabled and global_step >= overfit_success_min_step:
                        from training.big_vae.two_operator_overfit import overfit_success_passed

                        passed = overfit_success_passed(overfit_metrics, overfit_success_cfg)
                        overfit_success_streak = overfit_success_streak + 1 if passed else 0
                        overfit_success_reached = overfit_success_streak >= overfit_success_consecutive
                        if overfit_success_reached and rank == 0:
                            success_path = Path(str(two_operator_overfit_cfg["success_path"]))
                            success_path.parent.mkdir(parents=True, exist_ok=True)
                            success_path.write_text(
                                json.dumps(
                                    {
                                        "schema": "two_full_operator_overfit_success_v1",
                                        "step": global_step,
                                        "consecutive_passing_evals": overfit_success_streak,
                                        "thresholds": dict(overfit_success_cfg),
                                        "metrics": overfit_metrics,
                                    },
                                    indent=2,
                                    sort_keys=True,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            logger.info("Two-operator overfit early success: %s", success_path)

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
                mu_rms_window += float(stats[11].item())
                mu_abs_mean_window += float(stats[12].item())
                posterior_std_mean_window += float(stats[13].item())
                posterior_std_rms_window += float(stats[14].item())
                posterior_logvar_mean_window += float(stats[15].item())
                mu_batch_var_window += float(stats[16].item())
                mu_active_fraction_window += float(stats[17].item())
                kl_mean_part_window += float(stats[18].item())
                kl_variance_part_window += float(stats[19].item())
                logvar_min_fraction_window += float(stats[20].item())
                logvar_max_fraction_window += float(stats[21].item())
                v11_complement_window += float(stats[22].item())
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
                    avg_v11_complement = v11_complement_window / max(1, window_steps)
                    avg_kl = kl_window / max(1, window_steps)
                    avg_struct_dir = struct_dir_window / max(1, window_steps)
                    avg_struct_scale = struct_scale_window / max(1, window_steps)
                    avg_struct_rec = struct_rec_window / max(1, window_steps)
                    avg_struct_rel = struct_rel_window / max(1, window_steps)
                    avg_mu_rms = mu_rms_window / max(1, window_steps)
                    avg_mu_abs_mean = mu_abs_mean_window / max(1, window_steps)
                    avg_posterior_std_mean = posterior_std_mean_window / max(1, window_steps)
                    avg_posterior_std_rms = posterior_std_rms_window / max(1, window_steps)
                    avg_posterior_logvar_mean = posterior_logvar_mean_window / max(1, window_steps)
                    avg_mu_batch_var = mu_batch_var_window / max(1, window_steps)
                    avg_mu_active_fraction = mu_active_fraction_window / max(1, window_steps)
                    avg_kl_mean_part = kl_mean_part_window / max(1, window_steps)
                    avg_kl_variance_part = kl_variance_part_window / max(1, window_steps)
                    avg_logvar_min_fraction = logvar_min_fraction_window / max(1, window_steps)
                    avg_logvar_max_fraction = logvar_max_fraction_window / max(1, window_steps)
                    if epsilon_fork_ledger is not None:
                        replay_payload = validate_control_replay_metrics(
                            epsilon_fork_ledger,
                            global_step=global_step,
                            actual_metrics={
                                "loss": avg_loss,
                                "struct_dir": avg_struct_dir,
                                "struct_scale": avg_struct_scale,
                                "mu_rms": avg_mu_rms,
                            },
                        )
                        if replay_payload is not None:
                            replay_path = Path(str(epsilon_fork_cfg["control_replay_path"]))
                            write_json_immutable(replay_path, replay_payload)
                            logger.info("Exact epsilon=1e-8 step2510 control replay passed: %s", replay_path)
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
                    operator_mixer_metrics = (
                        operator_tile_mixer.telemetry() if operator_tile_mixer is not None else None
                    )
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
                        "step=%s/%s loss=%.6f behav=%.6f struct=%.6f v11_comp=%.6f "
                        "b_op=%.6f b_dir=%.6f b_scl=%.6f "
                        "s_dir=%.6f s_scl=%.6f s_rec=%.6f s_rel=%.6f "
                        "kl=%.6f kl_beta=%.6f latent_gate=%.6f mu_rms=%.6f post_std=%.6f "
                        "logvar_mean=%.6f lr=%.6e steps/s=%.2f cache=%s "
                        "data_build_ms=%.2f data_wait_ms=%.2f data_h2d_enqueue_ms=%.2f prefetch_depth=%.2f",
                        global_step,
                        max_steps,
                        avg_loss,
                        avg_behavioral,
                        avg_structural,
                        avg_v11_complement,
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
                        avg_mu_rms,
                        avg_posterior_std_mean,
                        avg_posterior_logvar_mean,
                        lr,
                        speed,
                        cache_metric,
                        avg_data_build_ms,
                        avg_data_wait_ms,
                        avg_data_h2d_enqueue_ms,
                        avg_data_prefetch_depth,
                    )
                    if operator_mixer_metrics is not None:
                        logger.info(
                            "operator_locality step=%s logical_index=%s materialized_bundles=%s "
                            "emitted_tiles=%s max_active_bundles=%s max_active_tensor_mib=%.2f",
                            global_step,
                            operator_mixer_metrics["logical_index"],
                            operator_mixer_metrics["materialized_bundles"],
                            operator_mixer_metrics["emitted_tiles"],
                            operator_mixer_metrics["max_active_bundles"],
                            operator_mixer_metrics["max_active_tensor_bytes"] / (1024.0 * 1024.0),
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
                            "train/v11_complement_loss": float(avg_v11_complement),
                            "train/kl_loss": float(avg_kl),
                            "train/struct_dir": float(avg_struct_dir),
                            "train/struct_scale": float(avg_struct_scale),
                            "train/struct_rec": float(avg_struct_rec),
                            "train/struct_rel": float(avg_struct_rel),
                            "train/lr": float(lr),
                            "train/kl_beta": float(current_kl_beta),
                            "train/latent_sampling_gate": float(current_latent_sampling_gate),
                            "latent/mu_rms": float(avg_mu_rms),
                            "latent/mu_abs_mean": float(avg_mu_abs_mean),
                            "latent/posterior_std_mean": float(avg_posterior_std_mean),
                            "latent/posterior_std_rms": float(avg_posterior_std_rms),
                            "latent/logvar_mean": float(avg_posterior_logvar_mean),
                            "latent/weighted_kl": float(current_kl_beta * avg_kl),
                            "latent/mu_batch_variance": float(avg_mu_batch_var),
                            "latent/mu_active_fraction_var_gt_1e-4": float(avg_mu_active_fraction),
                            "latent/kl_mean_part": float(avg_kl_mean_part),
                            "latent/kl_variance_part": float(avg_kl_variance_part),
                            "latent/logvar_min_clamp_fraction": float(avg_logvar_min_fraction),
                            "latent/logvar_max_clamp_fraction": float(avg_logvar_max_fraction),
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
                        if operator_mixer_metrics is not None:
                            comet_metrics.update(
                                {
                                    "data/operator_materialized_bundles": float(
                                        operator_mixer_metrics["materialized_bundles"]
                                    ),
                                    "data/operator_emitted_tiles": float(operator_mixer_metrics["emitted_tiles"]),
                                    "data/operator_max_active_bundles": float(
                                        operator_mixer_metrics["max_active_bundles"]
                                    ),
                                    "data/operator_max_active_tensor_mib": float(
                                        operator_mixer_metrics["max_active_tensor_bytes"] / (1024.0 * 1024.0)
                                    ),
                                }
                            )
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
                    v11_complement_window = 0.0
                    kl_window = 0.0
                    mu_rms_window = 0.0
                    mu_abs_mean_window = 0.0
                    posterior_std_mean_window = 0.0
                    posterior_std_rms_window = 0.0
                    posterior_logvar_mean_window = 0.0
                    mu_batch_var_window = 0.0
                    mu_active_fraction_window = 0.0
                    kl_mean_part_window = 0.0
                    kl_variance_part_window = 0.0
                    logvar_min_fraction_window = 0.0
                    logvar_max_fraction_window = 0.0
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

                checkpoint_writes_allowed = _checkpoint_writes_allowed(bounded_profile_spec)
                should_save_model_checkpoint = checkpoint_writes_allowed and (
                    global_step % checkpoint_every == 0
                    or global_step == max_steps
                    or global_step == training_stop_step
                    or overfit_success_reached
                )
                should_save_resume_state = (
                    checkpoint_writes_allowed
                    and resume_state_enabled
                    and not (
                        epsilon_fork_ledger is not None
                        and bool(epsilon_fork_cfg.get("suppress_resume_writes", False))
                    )
                    and (
                        global_step % resume_state_save_every == 0
                        or global_step == max_steps
                        or global_step == training_stop_step
                        or overfit_success_reached
                    )
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
                        rank=rank,
                        world_size=world_size,
                        active_device=device,
                    )
                if profile_recorder is not None:
                    if operator_tile_mixer is None:
                        raise RuntimeError("bounded runtime profile lost the production locality mixer")
                    expected_consumed = slice_batch_size * grad_accum_steps
                    consumed_start, consumed_end = _consumed_logical_range(
                        profile_step_consumed_logical_indices,
                        expected_count=expected_consumed,
                    )
                    profile_recorder.finish_step(
                        global_step=global_step,
                        loss=step_loss,
                        data_wait_s=profile_step_data_wait_s,
                        data_build_s=profile_step_data_build_s,
                        pin_s=profile_step_pin_s,
                        h2d_enqueue_s=profile_step_h2d_enqueue_s,
                        prefetch_depth=profile_step_prefetch_depth,
                        h2d_bytes=profile_step_h2d_bytes,
                        batch_size=slice_batch_size,
                        grad_accum_steps=grad_accum_steps,
                        committed_logical_index_start=consumed_start,
                        committed_logical_index_end_exclusive=consumed_end,
                        produced_logical_index=int(operator_tile_mixer.logical_index),
                        mixer_telemetry=operator_tile_mixer.telemetry(),
                    )
                if overfit_success_reached:
                    break

            if offline_batch_prefetcher is not None:
                _close_training_prefetcher(
                    offline_batch_prefetcher,
                    bounded_profile_spec=bounded_profile_spec,
                )
            if rank == 0:
                if v6_causal_enabled and v6_causal_cfg["arm"] in {
                    "alternating_homogeneous",
                    "cyclic_singleton",
                }:
                    if v6_causal_cfg["arm"] == "cyclic_singleton":
                        if v6_causal_selection_counts != [28] * 18:
                            raise RuntimeError(
                                "cyclic_singleton did not select every source row exactly 28 times: "
                                f"{v6_causal_selection_counts}"
                            )
                        if v6_causal_presentation_counts != [504] * 18:
                            raise RuntimeError(
                                "cyclic_singleton did not present every source row exactly 504 times: "
                                f"{v6_causal_presentation_counts}"
                            )
                    exposure_path = Path(str(v6_causal_cfg["ledger_path"])).with_name(
                        "v6_causal_exposure_ledger.json"
                    )
                    write_json_immutable(
                        exposure_path,
                        {
                            "schema": "weightclip_ae_v6_causal_exposure_v1",
                            "arm": str(v6_causal_cfg["arm"]),
                            "completed_steps": int(training_stop_step),
                            "selection_counts_by_source_position": v6_causal_selection_counts,
                            "presentation_counts_by_source_position": v6_causal_presentation_counts,
                            "physical_batch": 18,
                            "optimizer_steps": int(training_stop_step),
                        },
                    )
                if profile_recorder is not None:
                    if operator_tile_mixer is None:
                        raise RuntimeError("bounded runtime profile lost its operator mixer before close")
                    profile_recorder.record_producer_close(
                        produced_logical_index=int(operator_tile_mixer.logical_index),
                        requested_horizon_steps=int(prefetch_stop_step),
                        requested_horizon_tiles=int(
                            prefetch_stop_step * grad_accum_steps * slice_batch_size
                        ),
                        tiles_per_optimizer_step=int(grad_accum_steps * slice_batch_size),
                    )
                    profile_summary = profile_recorder.finalize()
                    logger.info("Bounded AE runtime profile complete: %s", json.dumps(profile_summary, sort_keys=True))
                else:
                    logger.info("Training completed successfully: steps=%s", max_steps)

    except BaseException as exc:
        failed = True
        active_profile_recorder = locals().get("profile_recorder")
        if isinstance(active_profile_recorder, _BoundedRuntimeProfileRecorder):
            active_profile_recorder.abort()
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
