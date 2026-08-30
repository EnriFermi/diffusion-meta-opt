from __future__ import annotations

import copy
import hashlib
import json
import math
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import torch


EPSILON_FORK_SCHEMA = "two_operator_exact_resume_epsilon_fork_v1"
EPSILON_FORK_SOURCE_STEP = 2_500
EPSILON_FORK_ADDITIONAL_STEPS = 250
EPSILON_FORK_SOURCE_EPSILON = 1.0e-8
EPSILON_FORK_TARGET_EPSILONS = (1.0e-8, 1.0e-12)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plain_json_value(value: Any) -> Any:
    """Recursively detach OmegaConf/container objects without stringifying data."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json_value(child) for child in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"epsilon-fork ledger contains a non-JSON value: {type(value).__name__}")


def _stat_identity(path: Path) -> dict[str, int]:
    info = path.stat()
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
        "mode": int(stat.S_IMODE(info.st_mode)),
    }


def stable_resume_checkpoint_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    before = _stat_identity(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = _stat_identity(resolved)
    if before != after:
        raise RuntimeError(f"epsilon fork source checkpoint changed while hashing: before={before} after={after}")
    if before["mode"] & 0o222:
        raise RuntimeError("epsilon fork source resume checkpoint must be immutable (no write bits)")
    return {"path": str(resolved), "sha256": digest.hexdigest(), "stat": before}


def _state_identity(optimizer: torch.optim.Optimizer) -> tuple[Any, ...]:
    rows: list[Any] = []
    for parameter, state in optimizer.state.items():
        fields: list[Any] = []
        for key in sorted(state):
            value = state[key]
            if torch.is_tensor(value):
                fields.append(
                    (
                        str(key),
                        id(value),
                        int(value._version),
                        tuple(int(dim) for dim in value.shape),
                        str(value.dtype),
                        str(value.device),
                    )
                )
            else:
                fields.append((str(key), id(value), copy.deepcopy(value)))
        rows.append((id(parameter), id(state), tuple(fields)))
    return tuple(rows)


def _group_snapshot(optimizer: torch.optim.Optimizer) -> tuple[list[list[int]], list[dict[str, Any]]]:
    parameter_ids: list[list[int]] = []
    metadata: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        parameter_ids.append([id(parameter) for parameter in group["params"]])
        metadata.append(
            {
                str(key): copy.deepcopy(value)
                for key, value in group.items()
                if key not in {"params", "eps"}
            }
        )
    return parameter_ids, metadata


def validate_epsilon_fork_config(
    raw: Mapping[str, Any],
    *,
    resume_checkpoint: Path,
    resumed_step: int,
    stop_after_step: int,
    slice_batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    allowed = {
        "schema",
        "enabled",
        "source_resume_checkpoint",
        "source_resume_sha256",
        "source_resume_stat",
        "expected_source_step",
        "additional_optimizer_steps",
        "source_optimizer_epsilon",
        "target_optimizer_epsilon",
        "expected_optimizer_group_count",
        "expected_optimizer_parameter_counts",
        "expected_optimizer_state_count",
        "startup_ledger_path",
        "control_replay_path",
        "control_reference",
        "suppress_resume_writes",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown exact epsilon-fork config keys: {unknown}")
    missing = sorted(allowed - set(raw))
    if missing:
        raise ValueError(f"missing exact epsilon-fork config keys: {missing}")
    cfg = dict(raw)
    if cfg["schema"] != EPSILON_FORK_SCHEMA or not bool(cfg["enabled"]):
        raise ValueError(f"epsilon fork must be enabled with schema {EPSILON_FORK_SCHEMA!r}")
    if int(cfg["expected_source_step"]) != EPSILON_FORK_SOURCE_STEP:
        raise ValueError("epsilon fork source step is frozen to exactly 2500")
    if int(cfg["additional_optimizer_steps"]) != EPSILON_FORK_ADDITIONAL_STEPS:
        raise ValueError("epsilon fork is hard-capped at exactly 250 additional optimizer steps")
    if int(resumed_step) != EPSILON_FORK_SOURCE_STEP:
        raise RuntimeError(f"epsilon fork loaded step {resumed_step}, expected exactly 2500")
    if int(stop_after_step) != EPSILON_FORK_SOURCE_STEP + EPSILON_FORK_ADDITIONAL_STEPS:
        raise RuntimeError("epsilon fork stop_after_step must be exactly 2750")
    if int(slice_batch_size) != 18 or int(grad_accum_steps) != 1:
        raise RuntimeError("epsilon fork requires the exact two-operator B18 x accum1 stream")

    expected_path = Path(str(cfg["source_resume_checkpoint"])).expanduser().resolve(strict=True)
    actual_path = resume_checkpoint.expanduser().resolve(strict=True)
    if actual_path != expected_path:
        raise RuntimeError(f"epsilon fork resume path mismatch: expected={expected_path} actual={actual_path}")
    actual_stat = _stat_identity(actual_path)
    expected_stat = {str(key): int(value) for key, value in dict(cfg["source_resume_stat"]).items()}
    if actual_stat != expected_stat:
        raise RuntimeError(f"epsilon fork source checkpoint stat drift: expected={expected_stat} actual={actual_stat}")
    if actual_stat["mode"] & 0o222:
        raise RuntimeError("epsilon fork source resume checkpoint must be immutable (no write bits)")
    source_sha = str(cfg["source_resume_sha256"])
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("epsilon fork requires a lowercase source resume SHA256")

    source_eps = float(cfg["source_optimizer_epsilon"])
    target_eps = float(cfg["target_optimizer_epsilon"])
    if source_eps != EPSILON_FORK_SOURCE_EPSILON:
        raise ValueError("epsilon fork source optimizer epsilon is frozen to 1e-8")
    if target_eps not in EPSILON_FORK_TARGET_EPSILONS:
        raise ValueError("epsilon fork target optimizer epsilon must be exactly 1e-8 or 1e-12")
    if not bool(cfg["suppress_resume_writes"]):
        raise ValueError("bounded epsilon fork must suppress rolling resume-state writes")

    control = cfg["control_reference"]
    if not isinstance(control, Mapping):
        raise TypeError("epsilon fork control_reference must be a mapping")
    expected_control_keys = {"step", "absolute_tolerance", "metrics"}
    if set(control) != expected_control_keys:
        raise ValueError(f"epsilon fork control_reference keys must be {sorted(expected_control_keys)}")
    if int(control["step"]) != 2_510:
        raise ValueError("epsilon fork control replay is frozen to source continuation step 2510")
    tolerance = float(control["absolute_tolerance"])
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0e-4:
        raise ValueError("epsilon fork control replay tolerance must lie in (0, 1e-4]")
    metrics = control["metrics"]
    expected_metric_names = {"loss", "struct_dir", "struct_scale", "mu_rms"}
    if not isinstance(metrics, Mapping) or set(metrics) != expected_metric_names:
        raise ValueError(f"epsilon fork control metrics must be exactly {sorted(expected_metric_names)}")
    for name, value in metrics.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"epsilon fork control metric {name} must be finite")
    return cfg


def apply_loaded_optimizer_epsilon_fork(
    optimizer: torch.optim.Optimizer,
    *,
    raw_config: Mapping[str, Any],
    resume_checkpoint: Path,
    resumed_step: int,
    stop_after_step: int,
    slice_batch_size: int,
    grad_accum_steps: int,
    stream_contract: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = validate_epsilon_fork_config(
        raw_config,
        resume_checkpoint=resume_checkpoint,
        resumed_step=resumed_step,
        stop_after_step=stop_after_step,
        slice_batch_size=slice_batch_size,
        grad_accum_steps=grad_accum_steps,
    )
    expected_group_count = int(cfg["expected_optimizer_group_count"])
    expected_parameter_counts = [int(value) for value in cfg["expected_optimizer_parameter_counts"]]
    expected_state_count = int(cfg["expected_optimizer_state_count"])
    actual_parameter_counts = [len(group["params"]) for group in optimizer.param_groups]
    if len(optimizer.param_groups) != expected_group_count or actual_parameter_counts != expected_parameter_counts:
        raise RuntimeError(
            "epsilon fork optimizer group inventory mismatch: "
            f"groups={len(optimizer.param_groups)} params={actual_parameter_counts}"
        )
    if len(optimizer.state) != expected_state_count:
        raise RuntimeError(
            f"epsilon fork optimizer state count mismatch: {len(optimizer.state)} != {expected_state_count}"
        )

    source_eps = float(cfg["source_optimizer_epsilon"])
    before_eps = [float(group["eps"]) for group in optimizer.param_groups]
    if before_eps != [source_eps] * expected_group_count:
        raise RuntimeError(f"loaded optimizer epsilon mismatch: expected={source_eps} actual={before_eps}")
    before_parameter_ids, before_metadata = _group_snapshot(optimizer)
    before_state_identity = _state_identity(optimizer)
    before_defaults = copy.deepcopy(optimizer.defaults)

    target_eps = float(cfg["target_optimizer_epsilon"])
    for group in optimizer.param_groups:
        group["eps"] = target_eps

    after_parameter_ids, after_metadata = _group_snapshot(optimizer)
    if before_parameter_ids != after_parameter_ids or before_metadata != after_metadata:
        raise AssertionError("epsilon fork changed optimizer parameter groups beyond eps")
    if before_state_identity != _state_identity(optimizer):
        raise AssertionError("epsilon fork changed optimizer moment/state objects")
    if before_defaults != optimizer.defaults:
        raise AssertionError("epsilon fork changed optimizer defaults; only loaded group eps is permitted")
    after_eps = [float(group["eps"]) for group in optimizer.param_groups]
    if after_eps != [target_eps] * expected_group_count:
        raise AssertionError("epsilon fork failed to set every loaded optimizer group epsilon")

    stream_payload = dict(stream_contract.get("payload", {}))
    logical_index_offset = int(stream_payload.get("logical_index_offset", 0))
    logical_start = (
        logical_index_offset
        + EPSILON_FORK_SOURCE_STEP * int(slice_batch_size) * int(grad_accum_steps)
    )
    logical_end = logical_index_offset + stop_after_step * int(slice_batch_size) * int(grad_accum_steps)
    control_step = int(cfg["control_reference"]["step"])
    control_window_steps = 10
    control_start = (
        logical_index_offset
        + (control_step - control_window_steps) * int(slice_batch_size) * int(grad_accum_steps)
    )
    control_end = logical_index_offset + control_step * int(slice_batch_size) * int(grad_accum_steps)
    return {
        "schema": EPSILON_FORK_SCHEMA,
        "source_resume_checkpoint": str(resume_checkpoint.resolve()),
        "source_resume_sha256": str(cfg["source_resume_sha256"]),
        "source_resume_stat": _plain_json_value(cfg["source_resume_stat"]),
        "source_step": int(resumed_step),
        "stop_after_step": int(stop_after_step),
        "additional_optimizer_steps": EPSILON_FORK_ADDITIONAL_STEPS,
        "source_optimizer_group_epsilons": before_eps,
        "target_optimizer_group_epsilons": after_eps,
        "optimizer_defaults_unchanged": True,
        "optimizer_defaults_epsilon": float(optimizer.defaults["eps"]),
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_parameter_counts": actual_parameter_counts,
        "optimizer_state_count": len(optimizer.state),
        "optimizer_non_epsilon_metadata_sha256": _canonical_sha256(before_metadata),
        "optimizer_state_object_identity_verified": True,
        "scheduler_rng_and_model_mutation_by_hook": "none",
        "operator_stream_contract": _plain_json_value(stream_contract),
        "committed_logical_index_start": logical_start,
        "committed_logical_index_end_exclusive": logical_end,
        "first_step_logical_range": [logical_start, logical_start + int(slice_batch_size)],
        "control_log_window_steps": control_window_steps,
        "control_log_window_logical_range": [control_start, control_end],
        "final_step_logical_range": [logical_end - int(slice_batch_size), logical_end],
        "control_reference": _plain_json_value(cfg["control_reference"]),
        "resume_state_writes_suppressed": True,
    }


def validate_control_replay_metrics(
    ledger: Mapping[str, Any],
    *,
    global_step: int,
    actual_metrics: Mapping[str, float],
) -> dict[str, Any] | None:
    target_eps = [float(value) for value in ledger["target_optimizer_group_epsilons"]]
    reference = ledger["control_reference"]
    if target_eps != [EPSILON_FORK_SOURCE_EPSILON] or int(global_step) != int(reference["step"]):
        return None
    expected = {str(key): float(value) for key, value in dict(reference["metrics"]).items()}
    actual = {name: float(actual_metrics[name]) for name in expected}
    tolerance = float(reference["absolute_tolerance"])
    errors = {name: abs(actual[name] - expected[name]) for name in expected}
    objective_metric_names = ("loss", "struct_dir", "struct_scale")
    tolerances = {name: tolerance for name in objective_metric_names}
    if any(not math.isfinite(value) for value in actual.values()):
        raise RuntimeError(f"epsilon=1e-8 control replay produced non-finite metrics: {actual}")
    failed = {
        name: errors[name]
        for name in objective_metric_names
        if errors[name] > tolerances[name]
    }
    if failed:
        raise RuntimeError(
            "epsilon=1e-8 exact continuation failed the source step2510 replay gate: "
            f"expected={expected} actual={actual} errors={errors} tolerances={tolerances} "
            f"failed={failed}"
        )
    return {
        "schema": "two_operator_epsilon_fork_control_replay_v1",
        "step": int(global_step),
        "expected": expected,
        "actual": actual,
        "absolute_errors": errors,
        "objective_absolute_tolerances": tolerances,
        "objective_absolute_tolerance": tolerance,
        "diagnostic_only_metrics": ["mu_rms"],
        "passed": True,
        "consumed_logical_range": list(ledger["control_log_window_logical_range"]),
        "stream_contract_sha256": str(ledger["operator_stream_contract"]["sha256"]),
    }


__all__ = [
    "EPSILON_FORK_ADDITIONAL_STEPS",
    "EPSILON_FORK_SCHEMA",
    "EPSILON_FORK_SOURCE_EPSILON",
    "EPSILON_FORK_SOURCE_STEP",
    "EPSILON_FORK_TARGET_EPSILONS",
    "apply_loaded_optimizer_epsilon_fork",
    "stable_resume_checkpoint_identity",
    "validate_control_replay_metrics",
    "validate_epsilon_fork_config",
]
