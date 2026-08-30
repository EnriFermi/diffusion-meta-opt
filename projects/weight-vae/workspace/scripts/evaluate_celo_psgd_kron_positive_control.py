#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import random
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    FlatSpec,
    TaskTensorSet,
    celo_meta_mlp_spec,
    logits_from_flat,
    spec_from_payload,
    tiny_cnn_spec,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.progress import make_progress


ARTIFACT_ROOT = Path("artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing").resolve()
DEFAULT_METHODS = ("raw_adam", "raw_sgd_momentum", "kron_whiten_momentum")
SUPPORTED_METHODS = ("raw_adam", "raw_sgd_momentum", "kron_whiten_momentum")
IMPLEMENTED_METHODS = frozenset({"raw_adam", "raw_sgd_momentum", "kron_whiten_momentum"})
PRIMARY_ENDPOINT = "post0_train_aulc"
KRON_TORCH_DISTRIBUTION = "kron-torch"
KRON_TORCH_REQUIRED_VERSION = "0.3.3"
DEFAULT_RAW_ADAM_LR_GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
DEFAULT_RAW_SGD_MOMENTUM_LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
DEFAULT_KRON_LR_GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
DEFAULT_KRON_UPDATE_PROBABILITY_GRID = (0.03, 0.1, 1.0)
KRON_PACKAGE_DEFAULT_SCHEDULE = "package_default"
NOT_APPLICABLE_SCHEDULE = "not_applicable"
CANDIDATE_CONFIG_COLUMNS = [
    "method",
    "lr",
    "kron_update_schedule",
    "kron_precond_update_probability",
    "lr_selection_mode",
    "schedule_selection_mode",
]
KRON_IMPORT_MESSAGE = (
    f"kron_whiten_momentum requires the pinned optional dependency "
    f"{KRON_TORCH_DISTRIBUTION}=={KRON_TORCH_REQUIRED_VERSION}. Provision that exact "
    "version in the experiment environment before review. This script never installs "
    "packages. It applies Kron to shaped CELO parameter tensors and is not exact Li c3."
)


def _log(message: str) -> None:
    print(f"[celo_psgd_kron_positive_control] {message}", flush=True)


def _run_dir(reference_run: str) -> Path:
    path = Path(reference_run).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / str(reference_run)).resolve()


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    return _json_safe(records)


def _stable_seed(*parts: Any) -> int:
    return int(_canonical_hash(list(parts))[:15], 16) % (2**31 - 1)


def _tensor_sha256(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return _canonical_hash({"kind": "full_batch"})
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _torch_load_readonly(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_cfg(
    run_dir: Path,
    *,
    device: str,
    downstream_steps: int,
    eval_every: int,
    batch_size: int,
) -> ExperimentConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    cfg = ExperimentConfig(**raw_cfg)
    return replace(
        cfg,
        device=str(device),
        show_progress=True,
        progress_backend="text",
        tune_starts=0,
        downstream_steps=int(downstream_steps),
        downstream_eval_every=max(1, int(eval_every)),
        downstream_batch_size=int(batch_size),
    )


def _spec_for_cfg(cfg: ExperimentConfig) -> FlatSpec:
    value = str(cfg.weight_distribution).strip().lower()
    if value in {"celo", "celo_meta", "celo_meta_mlp", "paper_meta_mlp"}:
        return celo_meta_mlp_spec(cfg)
    if value in {"tiny", "tiny_cnn", "cnn", "fashion_mnist_tiny_cnn"}:
        return tiny_cnn_spec()
    raise ValueError(f"unsupported weight_distribution={cfg.weight_distribution!r}")


def _load_weight_pool(run_dir: Path, cfg: ExperimentConfig) -> tuple[torch.Tensor, pd.DataFrame, str, FlatSpec]:
    payload = _torch_load_readonly(run_dir / "weight_pool.pt")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("weights"), torch.Tensor):
        raise RuntimeError(f"weight_pool.pt in {run_dir} does not contain a tensor under key 'weights'")
    weights = payload["weights"].detach().cpu()

    raw_records = payload.get("records", [])
    records = pd.DataFrame(raw_records)
    records_path = run_dir / "weight_pool_records.csv"
    if records.empty:
        if not records_path.is_file():
            raise FileNotFoundError(records_path)
        records = pd.read_csv(records_path)

    raw_spec = payload.get("spec")
    spec = spec_from_payload(raw_spec) if isinstance(raw_spec, Mapping) else _spec_for_cfg(cfg)
    if int(weights.shape[1]) != int(spec.dim):
        raise RuntimeError(f"weight dimension mismatch: weights={int(weights.shape[1])} spec={int(spec.dim)}")
    return weights, records.reset_index(drop=True), str(payload.get("cache_key", "")), spec


def _load_val_indices_if_available(run_dir: Path) -> torch.Tensor | None:
    checkpoint_path = run_dir / "vae_checkpoint.pt"
    if not checkpoint_path.is_file():
        return None
    payload = _torch_load_readonly(checkpoint_path)
    if not isinstance(payload, Mapping):
        return None
    val_indices = payload.get("val_indices")
    if isinstance(val_indices, torch.Tensor):
        return val_indices.detach().cpu().long()
    return None


def _heldout_final_indices(
    *,
    val_indices: torch.Tensor | None,
    weight_records: pd.DataFrame,
    weights_count: int,
    required: int,
    seed: int,
) -> tuple[list[int], str]:
    required = int(required)
    if required <= 0:
        return [], "none"
    if weight_records.empty or "step" not in weight_records.columns:
        if val_indices is not None and int(val_indices.numel()) >= required:
            return [int(v) for v in val_indices[:required].tolist()], "checkpoint_val_indices"
        return list(range(min(required, int(weights_count)))), "first_weight_indices_no_records"

    record_steps = pd.to_numeric(weight_records["step"], errors="coerce")
    final_step = int(record_steps.max())
    final_mask = record_steps == final_step
    if val_indices is not None and int(val_indices.numel()) > 0:
        step_tensor = torch.as_tensor(record_steps.fillna(-1).astype("int64").to_numpy(copy=True), dtype=torch.long)
        clamped = val_indices.clamp(min=0, max=max(0, int(weights_count) - 1))
        selected = val_indices[step_tensor.index_select(0, clamped) == final_step]
        if int(selected.numel()) >= required:
            return [int(v) for v in selected[:required].tolist()], f"checkpoint_val_indices_final_step_{final_step}"

    final_all = weight_records.index[final_mask].to_numpy(copy=True, dtype=np.int64)
    if int(final_all.shape[0]) >= required:
        rng = np.random.default_rng(int(seed))
        shuffled = np.array(final_all, copy=True)
        rng.shuffle(shuffled)
        return [int(v) for v in shuffled[:required].tolist()], f"seeded_final_step_{final_step}"

    if val_indices is not None and int(val_indices.numel()) >= required:
        return [int(v) for v in val_indices[:required].tolist()], "checkpoint_val_indices_any_step"
    return list(range(min(required, int(weights_count)))), "first_weight_indices_fallback"


def _normalize_start_bank(rows: pd.DataFrame, *, selection: str) -> pd.DataFrame:
    out = rows.copy().reset_index(drop=True)
    if "source_weight_index" not in out.columns:
        raise ValueError("start bank must contain source_weight_index")
    if "start_bank_position" in out.columns:
        if "input_start_bank_position" not in out.columns:
            out["input_start_bank_position"] = out["start_bank_position"]
        out["start_bank_position"] = np.arange(len(out), dtype=np.int64)
    else:
        out.insert(0, "start_bank_position", np.arange(len(out), dtype=np.int64))
    out["source_weight_index"] = pd.to_numeric(out["source_weight_index"], errors="raise").astype("int64")
    if "start_role" not in out.columns:
        out["start_role"] = "positive_control_eval"
    else:
        out["start_role"] = out["start_role"].fillna("positive_control_eval").astype(str)
    out["selection"] = str(selection)
    return out


def _start_bank(
    *,
    start_bank_csv: Path | None,
    eval_starts: int,
    val_indices: torch.Tensor | None,
    weight_records: pd.DataFrame,
    weights_count: int,
    seed: int,
) -> pd.DataFrame:
    eval_starts = int(eval_starts)
    if eval_starts <= 0:
        raise ValueError("--eval-starts must be positive")

    if start_bank_csv is not None:
        rows = pd.read_csv(start_bank_csv)
        if len(rows) < eval_starts:
            raise RuntimeError(f"{start_bank_csv} has {len(rows)} rows but --eval-starts={eval_starts}")
        return _normalize_start_bank(rows.head(eval_starts), selection=f"csv:{start_bank_csv}")

    positions, selection = _heldout_final_indices(
        val_indices=val_indices,
        weight_records=weight_records,
        weights_count=int(weights_count),
        required=eval_starts,
        seed=int(seed),
    )
    if len(positions) < eval_starts:
        raise RuntimeError(f"not enough starts: got={len(positions)} required={eval_starts}")
    if weight_records.empty:
        rows = pd.DataFrame({"source_weight_index": [int(v) for v in positions]})
    else:
        rows = weight_records.iloc[positions].copy().reset_index(drop=True)
        rows["source_weight_index"] = [int(v) for v in positions]
    return _normalize_start_bank(rows, selection=selection)


def _validate_start_bank(start_bank: pd.DataFrame, *, weights_count: int) -> None:
    source_indices = start_bank["source_weight_index"].astype("int64").to_numpy(copy=True)
    bad = source_indices[(source_indices < 0) | (source_indices >= int(weights_count))]
    if bad.size:
        raise ValueError(f"start bank source_weight_index out of range for weight pool: {bad[:10].tolist()}")
    duplicates = start_bank["source_weight_index"].duplicated(keep=False)
    if bool(duplicates.any()):
        values = start_bank.loc[duplicates, "source_weight_index"].astype(int).tolist()
        raise ValueError(f"start bank has duplicate source_weight_index values: {values[:20]}")


def _normalize_methods(values: Iterable[str]) -> list[str]:
    methods: list[str] = []
    for value in values:
        for item in str(value).split(","):
            method = item.strip()
            if method:
                methods.append(method)
    if not methods:
        methods = list(DEFAULT_METHODS)
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"unknown methods={unknown}; supported={list(SUPPORTED_METHODS)}")
    return methods


def _validate_kron_dependency(
    *,
    expected_version: str = KRON_TORCH_REQUIRED_VERSION,
    run_cpu_smoke_step: bool = True,
    run_celo_integration_smoke: bool = True,
) -> dict[str, Any]:
    # kron-torch decorates core CPU helpers with torch.compile. Force eager execution
    # so the reviewed protocol does not depend on a host compiler cache or toolchain.
    torch._dynamo.config.disable = True
    try:
        installed_version = importlib.metadata.version(KRON_TORCH_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(KRON_IMPORT_MESSAGE) from exc
    if str(installed_version) != str(expected_version):
        raise RuntimeError(
            f"unsupported {KRON_TORCH_DISTRIBUTION} version={installed_version!r}; "
            f"required exact version={expected_version!r}"
        )
    try:
        module = importlib.import_module("kron_torch")
    except ImportError as exc:
        raise RuntimeError(f"distribution is installed but import kron_torch failed: {exc}") from exc
    Kron = getattr(module, "Kron", None)
    if not inspect.isclass(Kron):
        raise RuntimeError("kron_torch.Kron is missing or is not a class")

    signature = inspect.signature(Kron.__init__)
    required_parameters = {
        "params",
        "lr",
        "b1",
        "weight_decay",
        "precond_lr",
        "preconditioner_update_probability",
        "memory_save_mode",
    }
    missing = sorted(required_parameters - set(signature.parameters))
    if missing:
        raise RuntimeError(f"kron_torch.Kron API mismatch; missing constructor parameters={missing}; signature={signature}")

    smoke_step_completed = False
    smoke_preconditioner_numel = 0
    smoke_prob_step = 0
    smoke_q_sha256_before = ""
    smoke_q_sha256_after = ""
    smoke_q_hash_changed = False
    if run_cpu_smoke_step:
        parameter = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32))
        try:
            optimizer = Kron(
                [parameter],
                lr=1e-3,
                b1=0.9,
                weight_decay=0.0,
                precond_lr=0.1,
                preconditioner_update_probability=1.0,
                memory_save_mode=None,
            )
            loss = parameter.square().sum()
            loss.backward()
            before = parameter.detach().clone()
            q_before = _optimizer_state_metrics(optimizer, [parameter], method="kron_whiten_momentum")
            optimizer.step()
            q_after = _optimizer_state_metrics(optimizer, [parameter], method="kron_whiten_momentum")
        except Exception as exc:
            raise RuntimeError(f"kron_torch.Kron CPU API smoke step failed: {type(exc).__name__}: {exc}") from exc
        if not bool(torch.isfinite(parameter).all()) or torch.equal(before, parameter.detach()):
            raise RuntimeError("kron_torch.Kron CPU API smoke step produced no finite parameter update")
        smoke_q_sha256_before = str(q_before["kron_q_sha256"])
        smoke_q_sha256_after = str(q_after["kron_q_sha256"])
        smoke_q_hash_changed = bool(smoke_q_sha256_before != smoke_q_sha256_after)
        state = optimizer.state.get(parameter, {})
        q_values = state.get("Q", [])
        smoke_preconditioner_numel = sum(
            int(value.numel()) for value in q_values if isinstance(value, torch.Tensor)
        )
        prob_step = getattr(optimizer, "_prob_step", None)
        smoke_prob_step = int(prob_step.detach().cpu().item()) if isinstance(prob_step, torch.Tensor) else 0
        if smoke_preconditioner_numel <= 0 or smoke_prob_step < 1:
            raise RuntimeError(
                "kron_torch.Kron CPU API smoke step did not expose an initialized Q state "
                "and an observed preconditioner update"
            )
        if not smoke_q_hash_changed:
            raise RuntimeError("kron_torch.Kron CPU API smoke step did not mutate Q at its expected update event")
        smoke_step_completed = True

    try:
        schedule_parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        schedule_optimizer = Kron([schedule_parameter], preconditioner_update_probability=None)
        package_schedule = schedule_optimizer.param_groups[0]["preconditioner_update_probability"]
        if not callable(package_schedule):
            raise RuntimeError("package-default preconditioner_update_probability is not callable")
        schedule_probe_steps = [0, 499, 500, 501, 4000]
        observed_schedule = {
            str(step): float(package_schedule(torch.tensor(float(step), dtype=torch.float32)).item())
            for step in schedule_probe_steps
        }
        expected_schedule = {str(step): _package_default_probability(step) for step in schedule_probe_steps}
        if any(
            not math.isclose(observed_schedule[key], expected_schedule[key], rel_tol=0.0, abs_tol=1e-7)
            for key in expected_schedule
        ):
            raise RuntimeError(
                f"package-default schedule mismatch: observed={observed_schedule} expected={expected_schedule}"
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"kron_torch.Kron package-default schedule validation failed: {exc}") from exc

    module_file = Path(str(getattr(module, "__file__", "")))
    implementation_file = Path(str(inspect.getsourcefile(Kron) or ""))
    celo_integration = _run_celo_kron_cpu_smoke(Kron) if run_celo_integration_smoke else {"status": "skipped"}
    return {
        "distribution": KRON_TORCH_DISTRIBUTION,
        "torch_compile_disabled_for_reproducibility": True,
        "expected_version": str(expected_version),
        "installed_version": str(installed_version),
        "module": str(getattr(module, "__name__", "kron_torch")),
        "module_file": _file_signature(module_file) if module_file.is_file() else str(module_file),
        "kron_implementation_file": (
            _file_signature(implementation_file) if implementation_file.is_file() else str(implementation_file)
        ),
        "kron_signature": str(signature),
        "cpu_smoke_step_completed": bool(smoke_step_completed),
        "cpu_smoke_preconditioner_numel": int(smoke_preconditioner_numel),
        "cpu_smoke_prob_step": int(smoke_prob_step),
        "cpu_smoke_q_sha256_before": smoke_q_sha256_before,
        "cpu_smoke_q_sha256_after": smoke_q_sha256_after,
        "cpu_smoke_q_hash_changed": bool(smoke_q_hash_changed),
        "package_default_schedule_callable": True,
        "package_default_schedule_probe": observed_schedule,
        "celo_shaped_cpu_integration": celo_integration,
    }


def _kron_optimizer_cls():
    module = importlib.import_module("kron_torch")
    return module.Kron


def _format_lr_grid(values: list[float] | None) -> list[float]:
    if values is None:
        return []
    out = [float(v) for v in values]
    bad = [v for v in out if not math.isfinite(v) or v <= 0.0]
    if bad:
        raise ValueError(f"LR grid must contain only positive finite values; bad={bad}")
    return sorted(set(out))


def _format_probability_grid(values: list[float] | None) -> list[float]:
    if values is None:
        return []
    out = [float(v) for v in values]
    bad = [v for v in out if not math.isfinite(v) or v <= 0.0 or v > 1.0]
    if bad:
        raise ValueError(f"probability grid must be finite and in (0, 1]; bad={bad}")
    return sorted(set(out))


def _constant_schedule_name(probability: float) -> str:
    return f"constant_{float(probability):.12g}"


def _candidate_config(row: Mapping[str, Any]) -> dict[str, Any]:
    probability = row.get("kron_precond_update_probability")
    return {
        "method": str(row["method"]),
        "lr": float(row["lr"]),
        "kron_update_schedule": str(row["kron_update_schedule"]),
        "kron_precond_update_probability": None if pd.isna(probability) else float(probability),
        "lr_selection_mode": str(row["lr_selection_mode"]),
        "schedule_selection_mode": str(row["schedule_selection_mode"]),
    }


def _candidate_grid(
    *,
    methods: list[str],
    raw_adam_lrs: list[float],
    raw_sgd_momentum_lrs: list[float],
    kron_lrs: list[float],
    kron_update_probabilities: list[float],
    include_kron_package_default: bool = True,
    raw_adam_lr_fixed: bool = False,
    raw_sgd_momentum_lr_fixed: bool = False,
    kron_lr_fixed: bool = False,
    kron_schedule_fixed: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    method_values: dict[str, list[tuple[float, str, float | None]]] = {
        "raw_adam": [(lr, NOT_APPLICABLE_SCHEDULE, None) for lr in raw_adam_lrs],
        "raw_sgd_momentum": [(lr, NOT_APPLICABLE_SCHEDULE, None) for lr in raw_sgd_momentum_lrs],
        "kron_whiten_momentum": (
            [(lr, KRON_PACKAGE_DEFAULT_SCHEDULE, None) for lr in kron_lrs]
            if include_kron_package_default
            else []
        )
        + [
            (lr, _constant_schedule_name(probability), probability)
            for lr in kron_lrs
            for probability in kron_update_probabilities
        ],
    }
    lr_fixed_by_method = {
        "raw_adam": bool(raw_adam_lr_fixed),
        "raw_sgd_momentum": bool(raw_sgd_momentum_lr_fixed),
        "kron_whiten_momentum": bool(kron_lr_fixed),
    }
    for method in methods:
        values = method_values[method]
        if not values:
            raise ValueError(f"method={method} has an empty predeclared candidate grid")
        for lr, schedule, probability in values:
            config = {
                "method": str(method),
                "lr": float(lr),
                "kron_update_schedule": str(schedule),
                "kron_precond_update_probability": float(probability) if probability is not None else None,
                "lr_selection_mode": "fixed_cli" if lr_fixed_by_method[method] else "tuned_grid",
                "schedule_selection_mode": (
                    "fixed_cli"
                    if method == "kron_whiten_momentum" and kron_schedule_fixed
                    else "tuned_grid" if method == "kron_whiten_momentum" else "not_applicable"
                ),
            }
            rows.append({**config, "candidate_id": _canonical_hash(config)[:16]})
    frame = pd.DataFrame(rows)
    if frame["candidate_id"].duplicated().any():
        raise RuntimeError("candidate hash collision or duplicate candidate config")
    return frame.sort_values(
        ["method", "kron_update_schedule", "lr", "kron_precond_update_probability"],
        na_position="first",
    ).reset_index(drop=True)


def _split_protocol_start_bank(
    start_bank: pd.DataFrame,
    *,
    tune_starts: int,
    eval_starts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tune_starts = int(tune_starts)
    eval_starts = int(eval_starts)
    if tune_starts <= 0 or eval_starts <= 0:
        raise ValueError("tune_starts and eval_starts must both be positive")
    required = tune_starts + eval_starts
    if len(start_bank) != required:
        raise ValueError(f"protocol start bank must have exactly {required} rows; got={len(start_bank)}")
    tune = start_bank.iloc[:tune_starts].copy().reset_index(drop=True)
    evaluation = start_bank.iloc[tune_starts:].copy().reset_index(drop=True)
    tune["protocol_split"] = "tune"
    tune["split_start_index"] = np.arange(len(tune), dtype=np.int64)
    tune["global_stream_index"] = tune["start_bank_position"].astype("int64")
    tune["start_role"] = "positive_control_tune"
    evaluation["protocol_split"] = "eval"
    evaluation["split_start_index"] = np.arange(len(evaluation), dtype=np.int64)
    evaluation["global_stream_index"] = evaluation["start_bank_position"].astype("int64")
    evaluation["start_role"] = "positive_control_eval"
    tune_sources = set(tune["source_weight_index"].astype(int).tolist())
    eval_sources = set(evaluation["source_weight_index"].astype(int).tolist())
    overlap = sorted(tune_sources & eval_sources)
    if overlap:
        raise ValueError(f"tune/eval source overlap={overlap}")
    return tune, evaluation


def _validate_result_grid(
    results: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    start_bank: pd.DataFrame,
    protocol_split: str,
) -> dict[str, Any]:
    config_columns = ["candidate_id", *CANDIDATE_CONFIG_COLUMNS]
    required_columns = {"protocol_split", "source_weight_index", *config_columns}
    missing = sorted(required_columns - set(results.columns))
    if missing:
        raise ValueError(f"result grid missing columns={missing}")
    if set(results["protocol_split"].astype(str)) != {str(protocol_split)}:
        raise ValueError(
            f"result grid contains split leakage: expected={protocol_split!r} "
            f"observed={sorted(set(results['protocol_split'].astype(str)))}"
        )
    candidate_rows = candidates[config_columns].drop_duplicates()
    for candidate in _json_safe_records(candidate_rows):
        config = _candidate_config(candidate)
        expected_id = _canonical_hash(config)[:16]
        if str(candidate["candidate_id"]) != expected_id:
            raise ValueError(
                f"candidate config hash mismatch: candidate_id={candidate['candidate_id']} expected={expected_id}"
            )
    observed_config = results[config_columns].merge(
        candidate_rows,
        on="candidate_id",
        how="left",
        suffixes=("_observed", "_expected"),
        validate="many_to_one",
    )
    string_fields = ["method", "kron_update_schedule", "lr_selection_mode", "schedule_selection_mode"]
    string_match = pd.Series(True, index=observed_config.index)
    for field in string_fields:
        string_match &= observed_config[f"{field}_observed"].astype(str) == observed_config[
            f"{field}_expected"
        ].astype(str)
    lr_match = pd.to_numeric(observed_config["lr_observed"], errors="coerce") == pd.to_numeric(
        observed_config["lr_expected"], errors="coerce"
    )
    probability_observed = pd.to_numeric(
        observed_config["kron_precond_update_probability_observed"], errors="coerce"
    )
    probability_expected = pd.to_numeric(
        observed_config["kron_precond_update_probability_expected"], errors="coerce"
    )
    probability_match = (probability_observed.isna() & probability_expected.isna()) | (
        probability_observed == probability_expected
    )
    config_match = string_match & lr_match & probability_match
    if not bool(config_match.all()):
        bad = observed_config.loc[~config_match].head(5).to_dict(orient="records")
        raise ValueError(f"result rows do not match frozen candidate configs: {bad}")

    expected = candidate_rows.assign(_join=1).merge(
        start_bank[["source_weight_index"]].drop_duplicates().assign(_join=1), on="_join"
    ).drop(columns="_join")
    key_columns = ["candidate_id", "method", "source_weight_index"]
    observed = results[key_columns].copy()
    expected_keys = set(map(tuple, expected[key_columns].itertuples(index=False, name=None)))
    observed_keys = set(map(tuple, observed.itertuples(index=False, name=None)))
    duplicates = int(observed.duplicated().sum())
    missing_keys = sorted(expected_keys - observed_keys)
    extra_keys = sorted(observed_keys - expected_keys)
    if duplicates or missing_keys or extra_keys or len(observed) != len(expected):
        raise ValueError(
            "result grid is not exact: "
            f"expected_rows={len(expected)} observed_rows={len(observed)} duplicates={duplicates} "
            f"missing={missing_keys[:5]} extra={extra_keys[:5]}"
        )
    return {
        "split": str(protocol_split),
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(observed)),
        "candidate_configs_sha256": _frame_sha256(candidate_rows, config_columns),
        "source_indices_sha256": _canonical_hash(sorted(start_bank["source_weight_index"].astype(int).tolist())),
        "accepted": True,
    }


def _validate_artifact_grid(
    frame: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    start_bank: pd.DataFrame,
    protocol_split: str,
    artifact_name: str,
    step_values: list[int] | None,
) -> dict[str, Any]:
    key_columns = ["protocol_split", "candidate_id", "source_weight_index"]
    if step_values is not None:
        key_columns.append("step")
    context_columns = [
        *CANDIDATE_CONFIG_COLUMNS,
        "global_stream_index",
        "start_bank_position",
    ]
    required = set(key_columns + context_columns)
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{artifact_name} missing columns={missing_columns}")
    if set(frame["protocol_split"].astype(str)) != {str(protocol_split)}:
        raise ValueError(
            f"{artifact_name} split leakage: expected={protocol_split!r} "
            f"observed={sorted(set(frame['protocol_split'].astype(str)))}"
        )

    candidate_context = candidates[["candidate_id", *CANDIDATE_CONFIG_COLUMNS]].copy()
    start_context = start_bank[
        ["source_weight_index", "global_stream_index", "start_bank_position"]
    ].copy()
    expected = candidate_context.assign(_join=1).merge(start_context.assign(_join=1), on="_join").drop(
        columns="_join"
    )
    expected.insert(0, "protocol_split", str(protocol_split))
    if step_values is not None:
        expected = expected.assign(_join=1).merge(
            pd.DataFrame({"step": [int(value) for value in step_values], "_join": 1}),
            on="_join",
        ).drop(columns="_join")

    observed_keys = frame[key_columns].copy()
    expected_keys = expected[key_columns].copy()
    duplicate_count = int(observed_keys.duplicated().sum())
    key_join = expected_keys.merge(observed_keys, on=key_columns, how="outer", indicator=True)
    missing_keys = key_join[key_join["_merge"] == "left_only"].head(5).to_dict(orient="records")
    unexpected_keys = key_join[key_join["_merge"] == "right_only"].head(5).to_dict(orient="records")
    if duplicate_count or missing_keys or unexpected_keys or len(frame) != len(expected):
        raise ValueError(
            f"{artifact_name} Cartesian keys are not exact: expected_rows={len(expected)} "
            f"observed_rows={len(frame)} duplicates={duplicate_count} "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )

    observed_context = frame[key_columns + context_columns].merge(
        expected[key_columns + context_columns],
        on=key_columns,
        how="left",
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    context_match = pd.Series(True, index=observed_context.index)
    string_fields = ["method", "kron_update_schedule", "lr_selection_mode", "schedule_selection_mode"]
    for field in string_fields:
        context_match &= observed_context[f"{field}_observed"].astype(str) == observed_context[
            f"{field}_expected"
        ].astype(str)
    for field in ["lr", "global_stream_index", "start_bank_position"]:
        context_match &= pd.to_numeric(observed_context[f"{field}_observed"], errors="coerce") == pd.to_numeric(
            observed_context[f"{field}_expected"], errors="coerce"
        )
    observed_probability = pd.to_numeric(
        observed_context["kron_precond_update_probability_observed"], errors="coerce"
    )
    expected_probability = pd.to_numeric(
        observed_context["kron_precond_update_probability_expected"], errors="coerce"
    )
    context_match &= (observed_probability.isna() & expected_probability.isna()) | (
        observed_probability == expected_probability
    )
    if not bool(context_match.all()):
        bad = observed_context.loc[~context_match].head(5).to_dict(orient="records")
        raise ValueError(f"{artifact_name} context join mismatch: {bad}")
    return {
        "artifact": artifact_name,
        "split": str(protocol_split),
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(frame)),
        "key_sha256": _frame_sha256(frame.sort_values(key_columns), key_columns),
        "accepted": True,
    }


def _recompute_post0_train_aulc(
    *,
    curves: pd.DataFrame,
    results: pd.DataFrame,
    finite_penalty: float,
) -> dict[str, Any]:
    key_columns = ["protocol_split", "candidate_id", "source_weight_index"]
    post0 = curves[pd.to_numeric(curves["step"], errors="coerce") > 0].copy()
    post0["_clipped_train_loss"] = np.minimum(
        pd.to_numeric(post0["train_loss"], errors="coerce").to_numpy(dtype=np.float64),
        float(finite_penalty),
    )
    recomputed = (
        post0.groupby(key_columns, sort=False)["_clipped_train_loss"]
        .mean()
        .reset_index(name="recomputed_post0_train_aulc")
    )
    joined = results[key_columns + [PRIMARY_ENDPOINT]].merge(
        recomputed,
        on=key_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    reported = pd.to_numeric(joined[PRIMARY_ENDPOINT], errors="coerce").to_numpy(dtype=np.float64)
    recalculated = pd.to_numeric(
        joined["recomputed_post0_train_aulc"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    differences = np.abs(reported - recalculated)
    accepted = bool(
        set(joined["_merge"].astype(str)) == {"both"}
        and np.isfinite(differences).all()
        and bool((differences <= 1e-12).all())
    )
    return {
        "accepted": accepted,
        "rows": int(len(joined)),
        "max_abs_diff": float(np.max(differences)) if differences.size else float("inf"),
    }


def _validate_kron_trajectories(
    diagnostics: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    downstream_steps: int,
) -> dict[str, Any]:
    kron = diagnostics[diagnostics["method"].astype(str) == "kron_whiten_momentum"].copy()
    if kron.empty:
        return {"accepted": True, "status": "not_requested", "trajectories": 0, "failures": []}
    q_columns = {
        "kron_q_sha256_before",
        "kron_q_sha256_after",
        "kron_q_hash_changed",
        "kron_expected_q_update_event",
        "kron_q_hash_change_matches_expected",
    }
    missing_q_columns = sorted(q_columns - set(kron.columns))
    if missing_q_columns:
        return {
            "accepted": False,
            "status": "checked",
            "trajectories": 0,
            "failures": [{"trajectory": "all", "reasons": [f"missing_q_mutation_columns:{missing_q_columns}"]}],
        }
    candidate_lookup = candidates.set_index("candidate_id", drop=False)
    failures: list[dict[str, Any]] = []
    trajectory_count = 0
    group_columns = ["protocol_split", "candidate_id", "source_weight_index"]
    for key, trajectory in kron.groupby(group_columns, sort=False):
        trajectory_count += 1
        trajectory = trajectory.sort_values("step").reset_index(drop=True)
        candidate_id = str(key[1])
        if candidate_id not in candidate_lookup.index:
            failures.append({"trajectory": list(key), "reason": "unknown_candidate"})
            continue
        candidate = candidate_lookup.loc[candidate_id]
        probability_value = candidate["kron_precond_update_probability"]
        probability = None if pd.isna(probability_value) else float(probability_value)
        expected = pd.DataFrame(
            _expected_kron_update_trace(
                schedule=str(candidate["kron_update_schedule"]),
                probability=probability,
                steps=int(downstream_steps),
            )
        )
        reasons: list[str] = []
        if trajectory["step"].astype(int).tolist() != expected["step"].astype(int).tolist():
            reasons.append("optimizer_step_grid")
        else:
            checks = {
                "effective_probability": np.allclose(
                    pd.to_numeric(
                        trajectory["kron_effective_update_probability"], errors="coerce"
                    ).to_numpy(dtype=np.float64),
                    expected["kron_effective_update_probability"].to_numpy(dtype=np.float64),
                    rtol=0.0,
                    atol=1e-7,
                ),
                "counter_before": np.array_equal(
                    pd.to_numeric(trajectory["kron_update_counter_before"], errors="coerce").to_numpy(
                        dtype=np.int64
                    ),
                    expected["kron_update_counter_before"].to_numpy(dtype=np.int64),
                ),
                "counter_after": np.array_equal(
                    pd.to_numeric(trajectory["kron_update_counter_after"], errors="coerce").to_numpy(
                        dtype=np.int64
                    ),
                    expected["kron_update_counter_after"].to_numpy(dtype=np.int64),
                ),
                "update_events": np.array_equal(
                    trajectory["kron_update_event"].astype(bool).to_numpy(),
                    expected["kron_update_event"].astype(bool).to_numpy(),
                ),
                "cumulative_updates": np.array_equal(
                    pd.to_numeric(trajectory["kron_cumulative_update_count"], errors="coerce").to_numpy(
                        dtype=np.int64
                    ),
                    expected["kron_cumulative_update_count"].to_numpy(dtype=np.int64),
                ),
            }
            reasons.extend(name for name, accepted in checks.items() if not bool(accepted))
        if not bool(trajectory["kron_counter_transition_matches"].astype(bool).all()):
            reasons.append("runtime_counter_transition_gate")
        if not bool(trajectory["kron_probability_matches"].astype(bool).all()):
            reasons.append("runtime_probability_gate")
        if not bool(trajectory["kron_q_finite"].astype(bool).all()):
            reasons.append("nonfinite_q")
        factor_counts = pd.to_numeric(
            trajectory["kron_preconditioner_tensors"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not bool(
            np.isfinite(factor_counts).all()
            and (factor_counts > 0).all()
            and np.unique(factor_counts).size == 1
        ):
            reasons.append("missing_q_factors")
        shapes = trajectory["kron_q_factor_shapes"].astype(str)
        if bool((shapes.str.len() == 0).any()) or int(shapes.nunique()) != 1:
            reasons.append("q_factor_shapes")
        hashes = trajectory["kron_q_sha256"].astype(str)
        if not bool(hashes.str.fullmatch(r"[0-9a-f]{64}").all()):
            reasons.append("q_hash")
        q_before = trajectory["kron_q_sha256_before"].astype(str)
        q_after = trajectory["kron_q_sha256_after"].astype(str)
        if not bool((hashes.to_numpy() == q_after.to_numpy()).all()):
            reasons.append("q_hash_after_mismatch")
        q_after_valid = q_after.str.fullmatch(r"[0-9a-f]{64}")
        q_before_valid = q_before.eq("") | q_before.str.fullmatch(r"[0-9a-f]{64}")
        if not bool(q_after_valid.all()) or not bool(q_before_valid.all()):
            reasons.append("q_before_after_hash")
        if not q_before.empty and q_before.iloc[0] != "":
            reasons.append("initial_q_state_not_empty")
        if len(trajectory) > 1 and not bool((q_after.iloc[:-1].to_numpy() == q_before.iloc[1:].to_numpy()).all()):
            reasons.append("q_hash_trajectory_discontinuity")
        observed_q_changed = trajectory["kron_q_hash_changed"].astype(bool).to_numpy()
        computed_q_changed = (q_before.to_numpy() != q_after.to_numpy())
        if not np.array_equal(observed_q_changed, computed_q_changed):
            reasons.append("q_hash_change_record")
        recorded_q_events = trajectory["kron_expected_q_update_event"].astype(bool).to_numpy()
        if not np.array_equal(recorded_q_events, expected["kron_q_update_event"].astype(bool).to_numpy()):
            reasons.append("q_expected_event_record")
        if not bool(trajectory["kron_q_hash_change_matches_expected"].astype(bool).all()):
            reasons.append("q_hash_expected_event_runtime_gate")
        expected_updates = int(expected["kron_update_event"].astype(bool).sum())
        if expected_updates <= 0:
            reasons.append("no_expected_preconditioner_update")
        else:
            expected_events = expected["kron_q_update_event"].astype(bool).to_numpy()
            if not np.array_equal(observed_q_changed, expected_events):
                reasons.append("q_hash_mutation_does_not_match_expected_update_events")
        if reasons:
            failures.append({"trajectory": [str(value) for value in key], "reasons": sorted(set(reasons))})
    return {
        "accepted": not failures,
        "status": "checked",
        "trajectories": int(trajectory_count),
        "failures": failures[:20],
    }


def _select_frozen_candidates(
    tuning_results: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    tune_start_bank: pd.DataFrame,
    finite_penalty: float,
) -> pd.DataFrame:
    _validate_result_grid(
        tuning_results,
        candidates=candidates,
        start_bank=tune_start_bank,
        protocol_split="tune",
    )
    if PRIMARY_ENDPOINT not in tuning_results.columns:
        raise ValueError(f"tuning results missing primary endpoint={PRIMARY_ENDPOINT}")
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict(orient="records"):
        sub = tuning_results[tuning_results["candidate_id"].astype(str) == str(candidate["candidate_id"])].copy()
        values = pd.to_numeric(sub[PRIMARY_ENDPOINT], errors="coerce").to_numpy(dtype=np.float64)
        diverged = sub["diverged"].astype(bool).to_numpy()
        valid = bool(values.size == len(tune_start_bank) and np.isfinite(values).all() and not diverged.any())
        median = float(np.median(values)) if values.size and np.isfinite(values).all() else float(finite_penalty)
        score = median if valid else float(finite_penalty)
        rows.append(
            {
                **candidate,
                "primary_endpoint": PRIMARY_ENDPOINT,
                "tune_starts": int(len(tune_start_bank)),
                "tuning_median_primary": float(median),
                "selection_score": float(score),
                "diverged_count": int(diverged.sum()),
                "candidate_valid": bool(valid),
                "selected": 0,
            }
        )
    selection = pd.DataFrame(rows)
    for method, sub in selection.groupby("method", sort=False):
        valid = sub[sub["candidate_valid"].astype(bool)].copy()
        if valid.empty:
            raise RuntimeError(f"no finite non-diverged tuning candidate for method={method}")
        best = valid.sort_values(
            ["selection_score", "kron_update_schedule", "lr", "candidate_id"],
            na_position="first",
        ).index[0]
        selection.loc[best, "selected"] = 1
    if not bool((selection.groupby("method")["selected"].sum() == 1).all()):
        raise RuntimeError("selection must freeze exactly one candidate per method")
    selection["lr_boundary_adequate"] = False
    selection["lr_boundary_status"] = "not_selected"
    for method, selected_rows in selection[selection["selected"].astype(int) == 1].groupby("method"):
        selected_index = selected_rows.index[0]
        chosen = selection.loc[selected_index]
        if str(chosen["lr_selection_mode"]) == "fixed_cli":
            selection.loc[selected_index, "lr_boundary_adequate"] = True
            selection.loc[selected_index, "lr_boundary_status"] = "fixed_cli_no_lr_selection_claim"
            continue
        peers = selection[
            (selection["method"].astype(str) == str(method))
            & (selection["kron_update_schedule"].astype(str) == str(chosen["kron_update_schedule"]))
            & selection["candidate_valid"].astype(bool)
        ]
        lower = bool((pd.to_numeric(peers["lr"], errors="coerce") < float(chosen["lr"])).any())
        higher = bool((pd.to_numeric(peers["lr"], errors="coerce") > float(chosen["lr"])).any())
        adequate = bool(lower and higher)
        selection.loc[selected_index, "lr_boundary_adequate"] = adequate
        missing_sides = [name for name, present in (("lower", lower), ("higher", higher)) if not present]
        selection.loc[selected_index, "lr_boundary_status"] = (
            "finite_nondiverged_bracket" if adequate else "missing_" + "_and_".join(missing_sides)
        )
    return selection.sort_values(["method", "selected", "selection_score"], ascending=[True, False, True]).reset_index(drop=True)


def _task_tensor_set(task_tensors: Mapping[str, TaskTensorSet], task_name: str) -> TaskTensorSet:
    if task_name in task_tensors:
        return task_tensors[task_name]
    if len(task_tensors) == 1:
        return next(iter(task_tensors.values()))
    raise KeyError(f"task {task_name!r} not found in loaded task tensors")


def _batch_indices(
    task_set: TaskTensorSet,
    *,
    batch_size: int,
    step: int,
    start_index: int,
) -> torch.Tensor | None:
    train_count = int(task_set.train_labels.shape[0])
    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size >= train_count:
        return None
    # Keep this exactly aligned with downstream.py so raw_adam can be replayed
    # against existing downstream artifacts. This is the per-split start_index,
    # not the absolute start_bank_position that includes LR-tuning rows.
    offset = (int(start_index) * 1009 + int(step) * batch_size) % train_count
    return (torch.arange(batch_size, device=task_set.train_labels.device) + offset).remainder(train_count).long()


def _loss_acc(
    flat: torch.Tensor,
    *,
    task_set: TaskTensorSet,
    spec: FlatSpec,
    split: str,
    tau: float,
    batch_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        images, labels = task_set.train_images, task_set.train_labels
        if batch_indices is not None:
            images = images.index_select(0, batch_indices)
            labels = labels.index_select(0, batch_indices)
    elif split == "test":
        images, labels = task_set.test_images, task_set.test_labels
    else:
        raise ValueError(f"unknown split={split!r}")
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc


def _finite_float(value: torch.Tensor | float, *, penalty: float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    value = float(value)
    if math.isfinite(value):
        return value
    return float(penalty)


def _safe_acc(value: torch.Tensor, *, finite: bool) -> float:
    if not finite:
        return 0.0
    out = float(value.detach().cpu().item())
    return out if math.isfinite(out) else 0.0


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().double().flatten()
    b64 = b.detach().double().flatten()
    denom = a64.norm() * b64.norm()
    if float(denom.detach().cpu().item()) <= 1e-30:
        return float("nan")
    return float((torch.dot(a64, b64) / denom).detach().cpu().item())


def _package_default_probability(optimizer_step: int) -> float:
    step = torch.tensor(float(optimizer_step), dtype=torch.float32)
    probability = torch.exp(torch.tensor(-0.001, dtype=torch.float32) * (step - 500.0))
    return float(probability.clamp_(min=0.03, max=1.0).item())


def _effective_kron_probability(
    *,
    schedule: str,
    probability: float | None,
    optimizer_step: int,
) -> float:
    if schedule == KRON_PACKAGE_DEFAULT_SCHEDULE:
        if probability is not None:
            raise ValueError("package_default schedule must use probability=None")
        return _package_default_probability(optimizer_step)
    if schedule.startswith("constant_"):
        if probability is None:
            raise ValueError(f"constant schedule={schedule} requires a numeric probability")
        return float(probability)
    raise ValueError(f"unsupported Kron update schedule={schedule!r}")


def _expected_kron_update_trace(
    *,
    schedule: str,
    probability: float | None,
    steps: int,
) -> list[dict[str, Any]]:
    counter = 0
    cumulative = 0
    rows: list[dict[str, Any]] = []
    for optimizer_step in range(int(steps)):
        effective_probability = _effective_kron_probability(
            schedule=schedule,
            probability=probability,
            optimizer_step=optimizer_step,
        )
        before = counter
        counter += 1
        event = bool(counter >= 1.0 / effective_probability)
        if event:
            counter = 0
            cumulative += 1
        rows.append(
            {
                "step": int(optimizer_step),
                "kron_effective_update_probability": float(effective_probability),
                "kron_update_counter_before": int(before),
                "kron_update_counter_after": int(counter),
                "kron_update_event": bool(event),
                "kron_q_update_event": bool(event or optimizer_step == 0),
                "kron_cumulative_update_count": int(cumulative),
            }
        )
    return rows


def _metadata_value(row: Mapping[str, Any], key: str, default: Any) -> Any:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value


def _optimizer_for_method(
    *,
    method: str,
    params: list[torch.nn.Parameter],
    lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    sgd_momentum: float,
    kron_b1: float,
    kron_weight_decay: float,
    kron_precond_lr: float,
    kron_precond_update_probability: float | None,
    kron_memory_save_mode: str | None,
) -> torch.optim.Optimizer:
    if method == "raw_adam":
        return torch.optim.Adam(params, lr=float(lr), betas=(float(adam_beta1), float(adam_beta2)), eps=float(adam_eps))
    if method == "raw_sgd_momentum":
        return torch.optim.SGD(params, lr=float(lr), momentum=float(sgd_momentum))
    if method == "kron_whiten_momentum":
        Kron = _kron_optimizer_cls()
        kwargs: dict[str, Any] = {
            "lr": float(lr),
            "b1": float(kron_b1),
            "weight_decay": float(kron_weight_decay),
            "precond_lr": float(kron_precond_lr),
            "memory_save_mode": kron_memory_save_mode,
        }
        if kron_precond_update_probability is not None:
            kwargs["preconditioner_update_probability"] = float(kron_precond_update_probability)
        return Kron(params, **kwargs)
    raise ValueError(f"unknown implemented method={method!r}")


def _optimizer_state_metrics(
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
    *,
    method: str,
) -> dict[str, Any]:
    if not params:
        return {
            "momentum_norm": float("nan"),
            "second_moment_norm": float("nan"),
            "kron_preconditioner_numel": 0.0,
            "kron_preconditioner_tensors": 0.0,
            "kron_prob_step": float("nan"),
            "kron_q_finite": False,
            "kron_q_factor_shapes": "",
            "kron_q_sha256": "",
        }
    state = optimizer.state.get(params[0], {})
    if method == "raw_adam":
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        return {
            "momentum_norm": float(exp_avg.detach().float().norm().cpu().item()) if isinstance(exp_avg, torch.Tensor) else float("nan"),
            "second_moment_norm": float(exp_avg_sq.detach().float().norm().cpu().item()) if isinstance(exp_avg_sq, torch.Tensor) else float("nan"),
            "kron_preconditioner_numel": 0.0,
            "kron_preconditioner_tensors": 0.0,
            "kron_prob_step": float("nan"),
            "kron_q_finite": False,
            "kron_q_factor_shapes": "",
            "kron_q_sha256": "",
        }
    if method == "raw_sgd_momentum":
        momentum = state.get("momentum_buffer")
        return {
            "momentum_norm": float(momentum.detach().float().norm().cpu().item()) if isinstance(momentum, torch.Tensor) else float("nan"),
            "second_moment_norm": float("nan"),
            "kron_preconditioner_numel": 0.0,
            "kron_preconditioner_tensors": 0.0,
            "kron_prob_step": float("nan"),
            "kron_q_finite": False,
            "kron_q_factor_shapes": "",
            "kron_q_sha256": "",
        }
    if method == "kron_whiten_momentum":
        momentum_norm_sq = 0.0
        precond_numel = 0
        precond_tensors = 0
        q_finite = True
        q_shapes: list[list[int]] = []
        q_digest = hashlib.sha256()
        for param_index, param in enumerate(params):
            p_state = optimizer.state.get(param, {})
            momentum = p_state.get("momentum_buffer")
            if isinstance(momentum, torch.Tensor):
                momentum_norm_sq += float(momentum.detach().float().pow(2).sum().cpu().item())
            q_values = p_state.get("Q", [])
            if isinstance(q_values, list):
                for factor_index, q in enumerate(q_values):
                    if isinstance(q, torch.Tensor):
                        precond_numel += int(q.numel())
                        precond_tensors += 1
                        q_finite = bool(q_finite and torch.isfinite(q).all().detach().cpu().item())
                        q_shapes.append([int(value) for value in q.shape])
                        q_digest.update(f"{param_index}:{factor_index}:".encode("ascii"))
                        q_digest.update(_tensor_sha256(q).encode("ascii"))
        prob_step = getattr(optimizer, "_prob_step", None)
        return {
            "momentum_norm": float(math.sqrt(momentum_norm_sq)) if momentum_norm_sq > 0.0 else float("nan"),
            "second_moment_norm": float("nan"),
            "kron_preconditioner_numel": float(precond_numel),
            "kron_preconditioner_tensors": float(precond_tensors),
            "kron_prob_step": float(prob_step.detach().cpu().item()) if isinstance(prob_step, torch.Tensor) else float("nan"),
            "kron_q_finite": bool(q_finite and precond_tensors > 0),
            "kron_q_factor_shapes": json.dumps(q_shapes, separators=(",", ":")),
            "kron_q_sha256": q_digest.hexdigest() if precond_tensors > 0 else "",
        }
    return {
        "momentum_norm": float("nan"),
        "second_moment_norm": float("nan"),
        "kron_preconditioner_numel": 0.0,
        "kron_preconditioner_tensors": 0.0,
        "kron_prob_step": float("nan"),
        "kron_q_finite": False,
        "kron_q_factor_shapes": "",
        "kron_q_sha256": "",
    }


def _run_celo_kron_cpu_smoke(Kron: type[torch.optim.Optimizer]) -> dict[str, Any]:
    spec = celo_meta_mlp_spec(ExperimentConfig())
    summaries: dict[str, Any] = {}
    arms = [
        (KRON_PACKAGE_DEFAULT_SCHEDULE, None, 3),
        (_constant_schedule_name(0.1), 0.1, 12),
        (_constant_schedule_name(1.0), 1.0, 3),
    ]
    for schedule, probability, steps in arms:
        flat = torch.linspace(-0.1, 0.1, int(spec.dim), dtype=torch.float32)
        params = _params_from_flat(flat, spec)
        optimizer = Kron(
            params,
            lr=1e-4,
            b1=0.9,
            weight_decay=0.0,
            precond_lr=0.1,
            preconditioner_update_probability=probability,
            memory_save_mode=None,
        )
        expected = _expected_kron_update_trace(
            schedule=schedule,
            probability=probability,
            steps=steps,
        )
        actual_events: list[bool] = []
        q_hash_changes: list[bool] = []
        q_hash_trace: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(parameter.square().mean() for parameter in params)
            loss.backward()
            q_before = _optimizer_state_metrics(optimizer, params, method="kron_whiten_momentum")
            counter_before = int(optimizer._update_counter.detach().cpu().item())
            configured_probability = optimizer.param_groups[0]["preconditioner_update_probability"]
            if callable(configured_probability):
                configured_probability = configured_probability(optimizer._prob_step.to(dtype=torch.float32))
            effective_probability = float(torch.as_tensor(configured_probability).detach().cpu().item())
            optimizer.step()
            counter_after = int(optimizer._update_counter.detach().cpu().item())
            event = bool(counter_after == 0 and (counter_before > 0 or effective_probability >= 1.0))
            actual_events.append(event)
            expected_row = expected[step]
            expected_q_event = bool(expected_row["kron_q_update_event"])
            snapshot = _optimizer_state_metrics(optimizer, params, method="kron_whiten_momentum")
            q_hash_changed = bool(q_before["kron_q_sha256"] != snapshot["kron_q_sha256"])
            q_hash_changes.append(q_hash_changed)
            q_hash_trace.append(
                {
                    "step": int(step),
                    "before": str(q_before["kron_q_sha256"]),
                    "after": str(snapshot["kron_q_sha256"]),
                    "changed": bool(q_hash_changed),
                    "expected_update_event": bool(expected_q_event),
                }
            )
            if not bool(snapshot["kron_q_finite"]):
                raise RuntimeError(f"CELO-shaped Kron smoke produced nonfinite Q for schedule={schedule} step={step}")
            if q_hash_changed != expected_q_event:
                raise RuntimeError(
                    f"CELO-shaped Kron smoke Q mutation mismatch schedule={schedule} step={step} "
                    f"expected_q_event={expected_q_event} q_hash_changed={q_hash_changed} "
                    f"q_before={q_before['kron_q_sha256']} q_after={snapshot['kron_q_sha256']}"
                )
            snapshots.append(snapshot)
            if not (
                counter_before == int(expected_row["kron_update_counter_before"])
                and counter_after == int(expected_row["kron_update_counter_after"])
                and event == bool(expected_row["kron_update_event"])
                and math.isclose(
                    effective_probability,
                    float(expected_row["kron_effective_update_probability"]),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
            ):
                raise RuntimeError(
                    f"CELO-shaped Kron smoke counter/schedule mismatch schedule={schedule} step={step}"
                )
        final = snapshots[-1]
        expected_updates = int(sum(bool(row["kron_update_event"]) for row in expected))
        expected_q_updates = int(sum(bool(row["kron_q_update_event"]) for row in expected))
        if int(sum(actual_events)) != expected_updates or int(sum(q_hash_changes)) != expected_q_updates:
            raise RuntimeError(
                f"CELO-shaped Kron smoke update count mismatch schedule={schedule} "
                f"actual_events={sum(actual_events)} q_hash_changes={sum(q_hash_changes)} "
                f"expected_scheduled={expected_updates} expected_q_updates={expected_q_updates}"
            )
        summaries[schedule] = {
            "steps": int(steps),
            "update_count": int(sum(actual_events)),
            "q_hash_change_count": int(sum(q_hash_changes)),
            "q_sha256_initial": str(q_hash_trace[0]["before"]),
            "q_sha256_final": str(final["kron_q_sha256"]),
            "q_sha256_trace": q_hash_trace,
            "q_hash_changed_on_expected_events": bool(
                all(
                    changed == bool(row["kron_q_update_event"])
                    for changed, row in zip(q_hash_changes, expected, strict=True)
                )
            ),
            "q_finite_every_step": True,
            "q_factor_count": int(final["kron_preconditioner_tensors"]),
            "q_factor_shapes": str(final["kron_q_factor_shapes"]),
            "q_sha256": str(final["kron_q_sha256"]),
        }
    return {
        "status": "accepted",
        "model_kind": str(spec.model_kind),
        "parameter_dim": int(spec.dim),
        "parameter_shapes": [[int(value) for value in shape] for shape in spec.shapes],
        "arms": summaries,
    }


def _params_from_flat(flat: torch.Tensor, spec: FlatSpec) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    offset = 0
    for shape, size in zip(spec.shapes, spec.sizes, strict=True):
        tensor = flat[offset : offset + int(size)].detach().clone().reshape(shape)
        params.append(torch.nn.Parameter(tensor))
        offset += int(size)
    return params


def _flat_from_params(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([param.reshape(-1) for param in params], dim=0)


def _flat_grad_from_params(params: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for param in params:
        if param.grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(param.grad.reshape(-1))
    return torch.cat(pieces, dim=0)


def _eval_full_split(
    theta: torch.Tensor,
    *,
    task_set: TaskTensorSet,
    spec: FlatSpec,
    tau: float,
    penalty: float,
) -> dict[str, float]:
    with torch.no_grad():
        train_loss, train_acc = _loss_acc(theta.detach(), task_set=task_set, spec=spec, split="train", tau=float(tau))
        test_loss, test_acc = _loss_acc(theta.detach(), task_set=task_set, spec=spec, split="test", tau=float(tau))
    train_finite = bool(torch.isfinite(train_loss).detach().cpu().item())
    test_finite = bool(torch.isfinite(test_loss).detach().cpu().item())
    return {
        "train_loss": _finite_float(train_loss, penalty=penalty),
        "train_acc": _safe_acc(train_acc, finite=train_finite),
        "test_loss": _finite_float(test_loss, penalty=penalty),
        "test_acc": _safe_acc(test_acc, finite=test_finite),
        "finite": bool(train_finite and test_finite),
    }


def _run_raw_curve(
    *,
    cfg: ExperimentConfig,
    spec: FlatSpec,
    task_tensors: Mapping[str, TaskTensorSet],
    w0: torch.Tensor,
    start_metadata: Mapping[str, Any],
    method: str,
    lr: float,
    start_index: int,
    protocol_split: str,
    candidate_id: str,
    curve_seed: int,
    kron_update_schedule: str,
    lr_selection_mode: str,
    schedule_selection_mode: str,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    sgd_momentum: float,
    kron_b1: float,
    kron_weight_decay: float,
    kron_precond_lr: float,
    kron_precond_update_probability: float | None,
    kron_memory_save_mode: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    task_name = str(_metadata_value(start_metadata, "task_name", "tiny_cnn"))
    tau = float(_metadata_value(start_metadata, "tau", 1.0))
    task_set = _task_tensor_set(task_tensors, task_name)
    batch_size = int(getattr(cfg, "downstream_batch_size", 0))
    eval_every = max(1, int(getattr(cfg, "downstream_eval_every", 1)))
    steps = int(cfg.downstream_steps)
    penalty = float(cfg.finite_penalty)

    source_weight_index = int(_metadata_value(start_metadata, "source_weight_index", start_index))
    start_bank_position = int(_metadata_value(start_metadata, "start_bank_position", start_index))
    start_role = str(_metadata_value(start_metadata, "start_role", "positive_control_eval"))
    source_run = int(_metadata_value(start_metadata, "run", -1))
    source_step = int(_metadata_value(start_metadata, "step", -1))
    source_lr = float(_metadata_value(start_metadata, "source_lr", _metadata_value(start_metadata, "lr", float("nan"))))
    input_start_sha256 = _tensor_sha256(w0)
    torch.manual_seed(int(curve_seed))

    if method == "kron_whiten_momentum":
        params = _params_from_flat(w0, spec)
    else:
        params = [torch.nn.Parameter(w0.detach().clone())]

    def current_flat() -> torch.Tensor:
        return _flat_from_params(params)

    initial_state_sha256 = _tensor_sha256(current_flat())
    if initial_state_sha256 != input_start_sha256:
        raise RuntimeError(
            f"optimizer parameter construction changed the literal start for method={method} "
            f"input_hash={input_start_sha256} parameter_hash={initial_state_sha256}"
        )

    optimizer = _optimizer_for_method(
        method=method,
        params=params,
        lr=float(lr),
        adam_beta1=float(adam_beta1),
        adam_beta2=float(adam_beta2),
        adam_eps=float(adam_eps),
        sgd_momentum=float(sgd_momentum),
        kron_b1=float(kron_b1),
        kron_weight_decay=float(kron_weight_decay),
        kron_precond_lr=float(kron_precond_lr),
        kron_precond_update_probability=kron_precond_update_probability,
        kron_memory_save_mode=kron_memory_save_mode,
    )
    expected_kron_trace = (
        _expected_kron_update_trace(
            schedule=str(kron_update_schedule),
            probability=kron_precond_update_probability,
            steps=steps,
        )
        if method == "kron_whiten_momentum"
        else []
    )

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    diverged = False
    threshold_step = -1
    kron_cumulative_update_count = 0
    kron_initial_q_sha256 = ""
    kron_final_q_sha256 = ""
    kron_q_hash_change_count = 0
    w0_norm = float(w0.detach().float().norm().cpu().item())
    started = time.perf_counter()

    for step in range(steps + 1):
        batch_indices = _batch_indices(
            task_set,
            batch_size=batch_size,
            step=int(step),
            start_index=int(start_index),
        )
        batch_indices_sha256 = _tensor_sha256(batch_indices)
        with torch.enable_grad():
            train_step_loss, _ = _loss_acc(
                current_flat(),
                task_set=task_set,
                spec=spec,
                split="train",
                tau=float(tau),
                batch_indices=batch_indices,
            )
        train_step_finite = bool(torch.isfinite(train_step_loss).detach().cpu().item())
        if not train_step_finite:
            diverged = True

        should_record = step == 0 or step % eval_every == 0 or step == steps or diverged
        if should_record:
            evals = _eval_full_split(current_flat().detach(), task_set=task_set, spec=spec, tau=float(tau), penalty=penalty)
            if threshold_step < 0 and evals["train_loss"] <= float(cfg.success_threshold):
                threshold_step = int(step)
            if not bool(evals["finite"]):
                diverged = True
            rows.append(
                {
                    "split": str(protocol_split),
                    "protocol_split": str(protocol_split),
                    "method": str(method),
                    "candidate_id": str(candidate_id),
                    "curve_seed": int(curve_seed),
                    "start_index": int(start_index),
                    "global_stream_index": int(start_index),
                    "start_bank_position": int(start_bank_position),
                    "start_role": start_role,
                    "source_weight_index": int(source_weight_index),
                    "source_run": int(source_run),
                    "source_step": int(source_step),
                    "source_lr": float(source_lr),
                    "task_name": task_name,
                    "tau": float(tau),
                    "lr": float(lr),
                    "kron_update_schedule": str(kron_update_schedule),
                    "lr_selection_mode": str(lr_selection_mode),
                    "schedule_selection_mode": str(schedule_selection_mode),
                    "kron_precond_update_probability": float(kron_precond_update_probability)
                    if kron_precond_update_probability is not None
                    else float("nan"),
                    "step": int(step),
                    "initial_state_sha256": str(initial_state_sha256),
                    "batch_indices_sha256": str(batch_indices_sha256),
                    "train_loss": float(evals["train_loss"]),
                    "train_acc": float(evals["train_acc"]),
                    "test_loss": float(evals["test_loss"]),
                    "test_acc": float(evals["test_acc"]),
                    "reconstruction_rel_l2": 0.0,
                    "diverged": bool(diverged),
                    "elapsed_sec": float(time.perf_counter() - started),
                }
            )

        if step == steps or diverged:
            break

        optimizer.zero_grad(set_to_none=True)
        train_step_loss.backward()
        grad = _flat_grad_from_params(params).detach().clone()
        grad_finite = bool(torch.isfinite(grad).all().detach().cpu().item())
        grad_norm = float(grad.detach().float().norm().cpu().item()) if grad_finite else float("inf")
        param_before = current_flat().detach().clone()
        before_norm = float(param_before.float().norm().cpu().item())
        if not grad_finite:
            diverged = True
            diag_rows.append(
                {
                    "protocol_split": str(protocol_split),
                    "method": str(method),
                    "candidate_id": str(candidate_id),
                    "curve_seed": int(curve_seed),
                    "start_index": int(start_index),
                    "global_stream_index": int(start_index),
                    "start_bank_position": int(start_bank_position),
                    "source_weight_index": int(source_weight_index),
                    "task_name": task_name,
                    "tau": float(tau),
                    "lr": float(lr),
                    "kron_update_schedule": str(kron_update_schedule),
                    "lr_selection_mode": str(lr_selection_mode),
                    "schedule_selection_mode": str(schedule_selection_mode),
                    "kron_precond_update_probability": float(kron_precond_update_probability)
                    if kron_precond_update_probability is not None
                    else float("nan"),
                    "step": int(step),
                    "initial_state_sha256": str(initial_state_sha256),
                    "batch_indices_sha256": str(batch_indices_sha256),
                    "batch_size": int(batch_size),
                    "batch_loss_before": _finite_float(train_step_loss, penalty=penalty),
                    "batch_loss_finite": bool(train_step_finite),
                    "gradient_finite": False,
                    "gradient_norm": float(grad_norm),
                    "parameter_norm_before": float(before_norm),
                    "update_norm": float("nan"),
                    "update_rel_w0": float("nan"),
                    "gradient_update_cosine": float("nan"),
                    "momentum_norm": float("nan"),
                    "second_moment_norm": float("nan"),
                    "kron_preconditioner_numel": float("nan"),
                    "kron_preconditioner_tensors": float("nan"),
                    "kron_prob_step": float("nan"),
                    "kron_effective_update_probability": float("nan"),
                    "kron_update_counter_before": float("nan"),
                    "kron_update_counter_after": float("nan"),
                    "kron_update_event": False,
                    "kron_expected_update_event": False,
                    "kron_expected_q_update_event": False,
                    "kron_q_hash_change_matches_expected": False,
                    "kron_counter_transition_matches": False,
                    "kron_probability_matches": False,
                    "kron_cumulative_update_count": float("nan"),
                    "kron_expected_cumulative_update_count": float("nan"),
                    "kron_q_finite": False,
                    "kron_q_factor_shapes": "",
                    "kron_q_sha256": "",
                    "kron_q_sha256_before": "",
                    "kron_q_sha256_after": "",
                    "kron_q_hash_changed": False,
                    "diverged": True,
                    "elapsed_sec": float(time.perf_counter() - started),
                }
            )
            break

        if method == "kron_whiten_momentum":
            q_before_metrics = _optimizer_state_metrics(optimizer, params, method=method)
            q_sha256_before = str(q_before_metrics["kron_q_sha256"])
            if step == 0:
                kron_initial_q_sha256 = q_sha256_before
            update_counter = getattr(optimizer, "_update_counter", None)
            counter_before = int(update_counter.detach().cpu().item()) if isinstance(update_counter, torch.Tensor) else -1
            configured_probability = optimizer.param_groups[0]["preconditioner_update_probability"]
            if callable(configured_probability):
                prob_step = getattr(optimizer, "_prob_step", None)
                if not isinstance(prob_step, torch.Tensor):
                    raise RuntimeError("Kron callable schedule is missing tensor _prob_step")
                configured_probability = configured_probability(prob_step.to(dtype=torch.float32))
            actual_effective_probability = float(torch.as_tensor(configured_probability).detach().cpu().item())
        else:
            counter_before = -1
            actual_effective_probability = float("nan")
        optimizer.step()
        if method == "kron_whiten_momentum":
            update_counter = getattr(optimizer, "_update_counter", None)
            counter_after = int(update_counter.detach().cpu().item()) if isinstance(update_counter, torch.Tensor) else -1
            expected_update = expected_kron_trace[step]
            probability_matches = bool(
                math.isclose(
                    actual_effective_probability,
                    float(expected_update["kron_effective_update_probability"]),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
            )
            actual_update_event = bool(
                counter_after == 0 and (counter_before > 0 or actual_effective_probability >= 1.0)
            )
            kron_cumulative_update_count += int(actual_update_event)
            counter_transition_matches = bool(
                counter_before == int(expected_update["kron_update_counter_before"])
                and counter_after == int(expected_update["kron_update_counter_after"])
                and actual_update_event == bool(expected_update["kron_update_event"])
                and probability_matches
            )
        else:
            counter_after = -1
            expected_update = {}
            probability_matches = True
            actual_update_event = False
            counter_transition_matches = True
        update = current_flat().detach() - param_before
        update_norm = float(update.detach().float().norm().cpu().item())
        state_metrics = _optimizer_state_metrics(optimizer, params, method=method)
        if method == "kron_whiten_momentum":
            q_sha256_after = str(state_metrics["kron_q_sha256"])
            q_hash_changed = bool(q_sha256_before != q_sha256_after)
            kron_final_q_sha256 = q_sha256_after
            kron_q_hash_change_count += int(q_hash_changed)
        else:
            q_sha256_before = ""
            q_sha256_after = ""
            q_hash_changed = False
        diag_rows.append(
            {
                "protocol_split": str(protocol_split),
                "method": str(method),
                "candidate_id": str(candidate_id),
                "curve_seed": int(curve_seed),
                "start_index": int(start_index),
                "global_stream_index": int(start_index),
                "start_bank_position": int(start_bank_position),
                "source_weight_index": int(source_weight_index),
                "task_name": task_name,
                "tau": float(tau),
                "lr": float(lr),
                "kron_update_schedule": str(kron_update_schedule),
                "lr_selection_mode": str(lr_selection_mode),
                "schedule_selection_mode": str(schedule_selection_mode),
                "kron_precond_update_probability": float(kron_precond_update_probability)
                if kron_precond_update_probability is not None
                else float("nan"),
                "step": int(step),
                "initial_state_sha256": str(initial_state_sha256),
                "batch_indices_sha256": str(batch_indices_sha256),
                "batch_size": int(batch_size),
                "batch_loss_before": _finite_float(train_step_loss, penalty=penalty),
                "batch_loss_finite": bool(train_step_finite),
                "gradient_finite": bool(grad_finite),
                "gradient_norm": float(grad_norm),
                "parameter_norm_before": float(before_norm),
                "update_norm": float(update_norm),
                "update_rel_w0": float(update_norm / max(w0_norm, 1e-12)),
                "gradient_update_cosine": _cosine(grad, update),
                "momentum_norm": float(state_metrics["momentum_norm"]),
                "second_moment_norm": float(state_metrics["second_moment_norm"]),
                "kron_preconditioner_numel": float(state_metrics["kron_preconditioner_numel"]),
                "kron_preconditioner_tensors": float(state_metrics["kron_preconditioner_tensors"]),
                "kron_prob_step": float(state_metrics["kron_prob_step"]),
                "kron_effective_update_probability": float(actual_effective_probability),
                "kron_update_counter_before": int(counter_before) if method == "kron_whiten_momentum" else float("nan"),
                "kron_update_counter_after": int(counter_after) if method == "kron_whiten_momentum" else float("nan"),
                "kron_update_event": bool(actual_update_event),
                "kron_expected_update_event": bool(expected_update.get("kron_update_event", False)),
                "kron_expected_q_update_event": bool(expected_update.get("kron_q_update_event", False)),
                "kron_q_hash_change_matches_expected": bool(
                    q_hash_changed == bool(expected_update.get("kron_q_update_event", False))
                ),
                "kron_counter_transition_matches": bool(counter_transition_matches),
                "kron_probability_matches": bool(probability_matches),
                "kron_cumulative_update_count": int(kron_cumulative_update_count)
                if method == "kron_whiten_momentum"
                else float("nan"),
                "kron_expected_cumulative_update_count": int(expected_update["kron_cumulative_update_count"])
                if method == "kron_whiten_momentum"
                else float("nan"),
                "kron_q_finite": bool(state_metrics["kron_q_finite"]),
                "kron_q_factor_shapes": str(state_metrics["kron_q_factor_shapes"]),
                "kron_q_sha256": str(state_metrics["kron_q_sha256"]),
                "kron_q_sha256_before": str(q_sha256_before),
                "kron_q_sha256_after": str(q_sha256_after),
                "kron_q_hash_changed": bool(q_hash_changed),
                "diverged": False,
                "elapsed_sec": float(time.perf_counter() - started),
            }
        )

    train_losses = np.array([float(row["train_loss"]) for row in rows], dtype=np.float64)
    test_losses = np.array([float(row["test_loss"]) for row in rows], dtype=np.float64)
    test_accs = np.array([float(row["test_acc"]) for row in rows], dtype=np.float64)
    update_norms = np.array([float(row["update_norm"]) for row in diag_rows if math.isfinite(float(row["update_norm"]))], dtype=np.float64)
    grad_norms = np.array([float(row["gradient_norm"]) for row in diag_rows if math.isfinite(float(row["gradient_norm"]))], dtype=np.float64)
    final = rows[-1]
    metrics = {
        "split": str(protocol_split),
        "protocol_split": str(protocol_split),
        "method": str(method),
        "candidate_id": str(candidate_id),
        "curve_seed": int(curve_seed),
        "start_index": int(start_index),
        "global_stream_index": int(start_index),
        "start_bank_position": int(start_bank_position),
        "start_role": start_role,
        "source_weight_index": int(source_weight_index),
        "source_run": int(source_run),
        "source_step": int(source_step),
        "source_lr": float(source_lr),
        "task_name": task_name,
        "tau": float(tau),
        "lr": float(lr),
        "kron_update_schedule": str(kron_update_schedule),
        "lr_selection_mode": str(lr_selection_mode),
        "schedule_selection_mode": str(schedule_selection_mode),
        "kron_precond_update_probability": float(kron_precond_update_probability)
        if kron_precond_update_probability is not None
        else float("nan"),
        "aulc": float(np.mean(np.minimum(train_losses, penalty))),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "step0_train_loss": float(train_losses[0]),
        "post0_train_aulc": float(np.mean(np.minimum(train_losses[1:], penalty)))
        if len(train_losses) > 1
        else float("nan"),
        "train_mean_loss": float(np.mean(train_losses)),
        "test_mean_loss": float(np.mean(test_losses)),
        "test_trapz_loss": float(np.trapezoid(test_losses) / max(1, len(test_losses) - 1)),
        "step0_test_loss": float(test_losses[0]),
        "post0_test_loss_mean": float(np.mean(test_losses[1:])) if len(test_losses) > 1 else float("nan"),
        "final_train_loss": float(final["train_loss"]),
        "best_train_loss": float(np.min(train_losses)),
        "final_test_loss": float(final["test_loss"]),
        "best_test_loss": float(np.min(test_losses)),
        "final_test_acc": float(final["test_acc"]),
        "best_test_acc": float(np.max(test_accs)),
        "steps_to_threshold": int(threshold_step),
        "reconstruction_rel_l2": 0.0,
        "initial_state_sha256": str(initial_state_sha256),
        "diverged": bool(diverged),
        "optimizer_steps_completed": int(len(diag_rows)),
        "mean_update_norm": float(np.mean(update_norms)) if update_norms.size else float("nan"),
        "median_update_norm": float(np.median(update_norms)) if update_norms.size else float("nan"),
        "mean_gradient_norm": float(np.mean(grad_norms)) if grad_norms.size else float("nan"),
        "median_gradient_norm": float(np.median(grad_norms)) if grad_norms.size else float("nan"),
        "kron_initial_q_sha256": str(kron_initial_q_sha256) if method == "kron_whiten_momentum" else "",
        "kron_final_q_sha256": str(kron_final_q_sha256) if method == "kron_whiten_momentum" else "",
        "kron_q_hash_change_count": int(kron_q_hash_change_count) if method == "kron_whiten_momentum" else 0,
        "kron_q_update_event_count": int(kron_q_hash_change_count) if method == "kron_whiten_momentum" else 0,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    return rows, metrics, diag_rows


def _summary_table(results: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "post0_train_aulc",
        "step0_train_loss",
        "aulc",
        "train_mean_loss",
        "test_mean_loss",
        "test_trapz_loss",
        "step0_test_loss",
        "post0_test_loss_mean",
        "final_test_loss",
        "best_test_loss",
        "final_test_acc",
        "best_test_acc",
        "mean_update_norm",
        "median_gradient_norm",
    ]
    rows: list[dict[str, Any]] = []
    for method, sub in results.groupby("method", sort=True):
        for metric in metric_names:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            rows.append(
                {
                    "method": str(method),
                    "metric": str(metric),
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
        rows.append(
            {
                "method": str(method),
                "metric": "diverged_fraction",
                "n": int(len(sub)),
                "mean": float(pd.to_numeric(sub["diverged"], errors="coerce").fillna(0).astype(float).mean()),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        )
        reached = pd.to_numeric(sub["steps_to_threshold"], errors="coerce").fillna(-1).astype(int) >= 0
        rows.append(
            {
                "method": str(method),
                "metric": "success_fraction",
                "n": int(len(sub)),
                "mean": float(reached.astype(float).mean()),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _paired_eval_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["method"].astype(str) == "raw_adam"].copy()
    if baseline.empty:
        return pd.DataFrame()
    pair_keys = ["protocol_split", "source_weight_index", "global_stream_index"]
    metrics = [PRIMARY_ENDPOINT, "final_train_loss", "final_test_loss", "final_test_acc"]
    baseline_columns = pair_keys + ["candidate_id", *metrics]
    baseline = baseline[baseline_columns].rename(
        columns={
            "candidate_id": "baseline_candidate_id",
            **{metric: f"baseline_{metric}" for metric in metrics},
        }
    )
    rows: list[pd.DataFrame] = []
    for method, method_rows in results.groupby("method", sort=False):
        if str(method) == "raw_adam":
            continue
        compared = method_rows[pair_keys + ["candidate_id", *metrics]].merge(
            baseline,
            on=pair_keys,
            how="inner",
            validate="one_to_one",
        )
        if len(compared) != len(baseline) or len(compared) != len(method_rows):
            raise RuntimeError(
                f"paired eval contrast is incomplete for method={method}: "
                f"baseline={len(baseline)} method_rows={len(method_rows)} pairs={len(compared)}"
            )
        compared.insert(1, "method", str(method))
        for metric in metrics:
            compared[f"delta_{metric}_vs_raw_adam"] = pd.to_numeric(
                compared[metric], errors="coerce"
            ) - pd.to_numeric(compared[f"baseline_{metric}"], errors="coerce")
        rows.append(compared)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_paired_contrast_plot(contrasts: pd.DataFrame, path: Path) -> None:
    if contrasts.empty:
        return
    import matplotlib.pyplot as plt

    metric = f"delta_{PRIMARY_ENDPOINT}_vs_raw_adam"
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for method, rows in contrasts.groupby("method", sort=False):
        ordered = rows.sort_values("global_stream_index")
        ax.plot(
            ordered["global_stream_index"],
            ordered[metric],
            marker="o",
            markersize=2.8,
            linewidth=1.0,
            label=str(method),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Global eval stream index")
    ax.set_ylabel("Post0 train AULC delta vs raw Adam")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    _log(f"wrote {path}")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    _log(f"wrote {path} rows={len(frame)}")


def _set_reproducibility(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _protocol_validation(
    *,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    tune_bank: pd.DataFrame,
    eval_bank: pd.DataFrame,
    tuning_results: pd.DataFrame,
    eval_results: pd.DataFrame,
    tuning_curves: pd.DataFrame,
    eval_curves: pd.DataFrame,
    tuning_diagnostics: pd.DataFrame,
    eval_diagnostics: pd.DataFrame,
    downstream_steps: int,
    checkpoint_steps: list[int],
    finite_penalty: float,
) -> dict[str, Any]:
    selected_candidates = selected[selected["selected"].astype(int) == 1][
        ["candidate_id", *CANDIDATE_CONFIG_COLUMNS]
    ].copy()
    gates: dict[str, Any] = {}
    try:
        gates["tuning_grid"] = _validate_result_grid(
            tuning_results,
            candidates=candidates,
            start_bank=tune_bank,
            protocol_split="tune",
        )
        gates["eval_grid"] = _validate_result_grid(
            eval_results,
            candidates=selected_candidates,
            start_bank=eval_bank,
            protocol_split="eval",
        )
        gates["tuning_result_keys"] = _validate_artifact_grid(
            tuning_results,
            candidates=candidates,
            start_bank=tune_bank,
            protocol_split="tune",
            artifact_name="tuning_results",
            step_values=None,
        )
        gates["eval_result_keys"] = _validate_artifact_grid(
            eval_results,
            candidates=selected_candidates,
            start_bank=eval_bank,
            protocol_split="eval",
            artifact_name="eval_results",
            step_values=None,
        )
        gates["tuning_curve_keys"] = _validate_artifact_grid(
            tuning_curves,
            candidates=candidates,
            start_bank=tune_bank,
            protocol_split="tune",
            artifact_name="tuning_curves",
            step_values=checkpoint_steps,
        )
        gates["eval_curve_keys"] = _validate_artifact_grid(
            eval_curves,
            candidates=selected_candidates,
            start_bank=eval_bank,
            protocol_split="eval",
            artifact_name="eval_curves",
            step_values=checkpoint_steps,
        )
        optimizer_steps = list(range(int(downstream_steps)))
        gates["tuning_diagnostic_keys"] = _validate_artifact_grid(
            tuning_diagnostics,
            candidates=candidates,
            start_bank=tune_bank,
            protocol_split="tune",
            artifact_name="tuning_diagnostics",
            step_values=optimizer_steps,
        )
        gates["eval_diagnostic_keys"] = _validate_artifact_grid(
            eval_diagnostics,
            candidates=selected_candidates,
            start_bank=eval_bank,
            protocol_split="eval",
            artifact_name="eval_diagnostics",
            step_values=optimizer_steps,
        )
        gates["artifact_grids_exact"] = True
    except ValueError as exc:
        gates["artifact_grids_exact"] = False
        gates["artifact_grid_error"] = str(exc)

    tune_sources = set(tune_bank["source_weight_index"].astype(int).tolist())
    eval_sources = set(eval_bank["source_weight_index"].astype(int).tolist())
    gates["tune_eval_disjoint"] = not bool(tune_sources & eval_sources)
    gates["tune_eval_overlap"] = sorted(tune_sources & eval_sources)
    tune_streams = set(tune_bank["global_stream_index"].astype(int).tolist())
    eval_streams = set(eval_bank["global_stream_index"].astype(int).tolist())
    expected_tune_streams = set(range(len(tune_bank)))
    expected_eval_streams = set(range(len(tune_bank), len(tune_bank) + len(eval_bank)))
    gates["global_batch_streams_disjoint"] = bool(
        not (tune_streams & eval_streams)
        and tune_streams == expected_tune_streams
        and eval_streams == expected_eval_streams
    )
    gates["tune_global_stream_indices"] = sorted(tune_streams)
    gates["eval_global_stream_indices"] = sorted(eval_streams)
    gates["primary_endpoint"] = PRIMARY_ENDPOINT
    gates["primary_endpoint_predeclared"] = bool(
        set(tuning_results.get("primary_endpoint", pd.Series(dtype=str)).astype(str)) == {PRIMARY_ENDPOINT}
        and set(eval_results.get("primary_endpoint", pd.Series(dtype=str)).astype(str)) == {PRIMARY_ENDPOINT}
    )
    gates["tuning_primary_recomputed"] = _recompute_post0_train_aulc(
        curves=tuning_curves,
        results=tuning_results,
        finite_penalty=float(finite_penalty),
    )
    gates["eval_primary_recomputed"] = _recompute_post0_train_aulc(
        curves=eval_curves,
        results=eval_results,
        finite_penalty=float(finite_penalty),
    )
    gates["primary_endpoint_recomputed_exact"] = bool(
        gates["tuning_primary_recomputed"]["accepted"]
        and gates["eval_primary_recomputed"]["accepted"]
    )

    combined_results = pd.concat([tuning_results, eval_results], ignore_index=True)
    combined_curves = pd.concat([tuning_curves, eval_curves], ignore_index=True)
    combined_diagnostics = pd.concat([tuning_diagnostics, eval_diagnostics], ignore_index=True)
    gates["all_results_finite"] = bool(
        np.isfinite(pd.to_numeric(combined_results[PRIMARY_ENDPOINT], errors="coerce").to_numpy(dtype=np.float64)).all()
    )
    gates["no_divergence"] = not bool(combined_results["diverged"].astype(bool).any())

    initial_counts = combined_results.groupby(["protocol_split", "source_weight_index"])[
        "initial_state_sha256"
    ].nunique()
    gates["literal_start_hashes_match"] = bool((initial_counts == 1).all())
    curve_seed_counts = combined_results.groupby(["protocol_split", "source_weight_index"])["curve_seed"].nunique()
    gates["curve_seeds_match"] = bool((curve_seed_counts == 1).all())
    step0 = combined_curves[pd.to_numeric(combined_curves["step"], errors="coerce").astype(int) == 0]
    step0_ranges = step0.groupby(["protocol_split", "source_weight_index"])[["train_loss", "test_loss"]].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    step0_max = float(step0_ranges.to_numpy(dtype=np.float64).max()) if not step0_ranges.empty else float("inf")
    gates["step0_loss_max_abs_diff"] = step0_max
    gates["step0_losses_exact"] = bool(step0_max == 0.0)

    batch_counts = combined_diagnostics.groupby(["protocol_split", "source_weight_index", "step"])[
        "batch_indices_sha256"
    ].nunique()
    gates["batch_hashes_match"] = bool((batch_counts == 1).all())
    gates["batch_hash_group_count"] = int(len(batch_counts))

    gates["kron_trajectory_validation"] = _validate_kron_trajectories(
        combined_diagnostics,
        candidates=candidates,
        downstream_steps=int(downstream_steps),
    )
    gates["kron_trajectories_exact"] = bool(gates["kron_trajectory_validation"]["accepted"])
    selected_rows = selected[selected["selected"].astype(int) == 1]
    gates["lr_boundary_adequate"] = bool(selected_rows["lr_boundary_adequate"].astype(bool).all())
    gates["lr_boundary_status"] = selected_rows[
        ["method", "candidate_id", "lr", "kron_update_schedule", "lr_selection_mode", "lr_boundary_status"]
    ].to_dict(orient="records")

    expected_tune_results = int(len(candidates) * len(tune_bank))
    expected_eval_results = int(len(selected_candidates) * len(eval_bank))
    checkpoint_count = int(len(checkpoint_steps))
    gates["row_counts"] = {
        "tuning_results": {"expected": expected_tune_results, "observed": int(len(tuning_results))},
        "eval_results": {"expected": expected_eval_results, "observed": int(len(eval_results))},
        "tuning_curves": {"expected": expected_tune_results * checkpoint_count, "observed": int(len(tuning_curves))},
        "eval_curves": {"expected": expected_eval_results * checkpoint_count, "observed": int(len(eval_curves))},
        "tuning_diagnostics": {
            "expected": expected_tune_results * int(downstream_steps),
            "observed": int(len(tuning_diagnostics)),
        },
        "eval_diagnostics": {
            "expected": expected_eval_results * int(downstream_steps),
            "observed": int(len(eval_diagnostics)),
        },
    }
    gates["row_counts_exact"] = all(
        int(value["expected"]) == int(value["observed"]) for value in gates["row_counts"].values()
    )
    eval_ids = set(eval_results["candidate_id"].astype(str).tolist())
    selected_ids = set(selected_candidates["candidate_id"].astype(str).tolist())
    gates["frozen_selection_only_on_eval"] = bool(eval_ids == selected_ids)
    gates["candidate_grid_sha256"] = _frame_sha256(
        candidates, ["candidate_id", *CANDIDATE_CONFIG_COLUMNS]
    )
    gates["selected_configs_sha256"] = _frame_sha256(
        selected_candidates, ["candidate_id", *CANDIDATE_CONFIG_COLUMNS]
    )
    gates["tune_bank_sha256"] = _frame_sha256(tune_bank)
    gates["eval_bank_sha256"] = _frame_sha256(eval_bank)
    required_boolean_gates = [
        "artifact_grids_exact",
        "tune_eval_disjoint",
        "global_batch_streams_disjoint",
        "primary_endpoint_predeclared",
        "primary_endpoint_recomputed_exact",
        "all_results_finite",
        "no_divergence",
        "literal_start_hashes_match",
        "curve_seeds_match",
        "step0_losses_exact",
        "batch_hashes_match",
        "kron_trajectories_exact",
        "lr_boundary_adequate",
        "row_counts_exact",
        "frozen_selection_only_on_eval",
    ]
    gates["accepted"] = bool(all(gates.get(key) is True for key in required_boolean_gates))
    gates["required_boolean_gates"] = required_boolean_gates
    return gates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible raw-weight downstream optimizer positive control on a CELO reference run. "
            "Implemented methods are raw_adam, raw_sgd_momentum, and optional kron_whiten_momentum via kron-torch."
        )
    )
    parser.add_argument("--reference-run", required=True, help="Reference run label under the CELO artifact root, or an absolute run directory.")
    parser.add_argument("--start-bank-csv", default="", help="Optional CSV with source_weight_index rows to reuse.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Validate pinned dependency/API and protocol grids on CPU, then exit before loading data.")
    parser.add_argument("--tune-starts", type=int, default=8)
    parser.add_argument("--eval-starts", type=int, default=64)
    parser.add_argument("--downstream-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0, help="Runner seed for fallback start selection when --start-bank-csv is absent.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help=f"Methods to run. Supported: {', '.join(SUPPORTED_METHODS)}")
    parser.add_argument("--raw-adam-lr", type=float, default=None, help="Predeclare one fixed raw_adam LR instead of tuning its grid.")
    parser.add_argument(
        "--raw-adam-lr-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_RAW_ADAM_LR_GRID),
        help="Predeclared raw_adam tune-only LR grid.",
    )
    parser.add_argument(
        "--raw-sgd-momentum-lr",
        type=float,
        default=None,
        help="Predeclare one fixed raw_sgd_momentum LR instead of tuning its grid.",
    )
    parser.add_argument(
        "--raw-sgd-momentum-lr-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_RAW_SGD_MOMENTUM_LR_GRID),
        help="Predeclared raw_sgd_momentum tune-only LR grid.",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--kron-lr", type=float, default=None, help="Predeclare one fixed Kron parameter LR instead of tuning its LR grid.")
    parser.add_argument(
        "--kron-lr-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_KRON_LR_GRID),
        help="Predeclared Kron tune-only parameter LR grid.",
    )
    parser.add_argument("--kron-b1", type=float, default=0.9)
    parser.add_argument("--kron-weight-decay", type=float, default=0.0)
    parser.add_argument("--kron-precond-lr", type=float, default=0.1)
    parser.add_argument(
        "--kron-precond-update-probability",
        type=float,
        default=None,
        help="Predeclare one fixed numeric Kron update probability instead of tuning schedule arms.",
    )
    parser.add_argument(
        "--kron-fixed-package-default-schedule",
        action="store_true",
        help="Use only the package-default callable schedule and make no schedule-selection claim.",
    )
    parser.add_argument(
        "--kron-precond-update-probability-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_KRON_UPDATE_PROBABILITY_GRID),
        help="Predeclared numeric Kron tune-only arms; package_default is also included unless a schedule is fixed.",
    )
    parser.add_argument(
        "--kron-memory-save-mode",
        default="",
        choices=["", "smart_one_diag", "one_diag", "all_diag"],
        help="Optional kron-torch memory_save_mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        if int(args.tune_starts) <= 0 or int(args.eval_starts) <= 0:
            raise ValueError("tune_starts and eval_starts must both be positive")
        if int(args.downstream_steps) <= 0:
            raise ValueError("downstream_steps must be positive so the post0 endpoint is defined")
        if int(args.eval_every) <= 0:
            raise ValueError("eval_every must be positive")
        if int(args.batch_size) < 0:
            raise ValueError("batch_size must be non-negative; zero means full batch")
        if args.kron_precond_update_probability is not None and bool(args.kron_fixed_package_default_schedule):
            raise ValueError(
                "--kron-precond-update-probability and --kron-fixed-package-default-schedule are mutually exclusive"
            )
        methods = _normalize_methods(args.methods)
        if "kron_whiten_momentum" in methods:
            missing_controls = {"raw_adam", "raw_sgd_momentum"} - set(methods)
            if missing_controls:
                raise ValueError(
                    "Kron review protocol requires meaningful raw controls; "
                    f"missing methods={sorted(missing_controls)}"
                )
        raw_adam_lrs = _format_lr_grid(
            [float(args.raw_adam_lr)] if args.raw_adam_lr is not None else args.raw_adam_lr_grid
        )
        raw_sgd_momentum_lrs = _format_lr_grid(
            [float(args.raw_sgd_momentum_lr)]
            if args.raw_sgd_momentum_lr is not None
            else args.raw_sgd_momentum_lr_grid
        )
        kron_lrs = _format_lr_grid([float(args.kron_lr)] if args.kron_lr is not None else args.kron_lr_grid)
        if bool(args.kron_fixed_package_default_schedule):
            kron_probabilities = []
            include_kron_package_default = True
        elif args.kron_precond_update_probability is not None:
            kron_probabilities = _format_probability_grid([float(args.kron_precond_update_probability)])
            include_kron_package_default = False
        else:
            kron_probabilities = _format_probability_grid(args.kron_precond_update_probability_grid)
            include_kron_package_default = True
        candidates = _candidate_grid(
            methods=methods,
            raw_adam_lrs=raw_adam_lrs,
            raw_sgd_momentum_lrs=raw_sgd_momentum_lrs,
            kron_lrs=kron_lrs,
            kron_update_probabilities=kron_probabilities,
            include_kron_package_default=include_kron_package_default,
            raw_adam_lr_fixed=args.raw_adam_lr is not None,
            raw_sgd_momentum_lr_fixed=args.raw_sgd_momentum_lr is not None,
            kron_lr_fixed=args.kron_lr is not None,
            kron_schedule_fixed=bool(
                args.kron_fixed_package_default_schedule or args.kron_precond_update_probability is not None
            ),
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None

    dependency_validation: dict[str, Any] = {"status": "not_requested"}
    if "kron_whiten_momentum" in methods:
        try:
            dependency_validation = {
                "status": "accepted",
                **_validate_kron_dependency(expected_version=KRON_TORCH_REQUIRED_VERSION),
            }
        except RuntimeError as exc:
            raise SystemExit(f"ERROR: strict Kron dependency/API gate failed: {exc}") from None

    protocol_preview = {
        "causal_scope": "practical raw-space kron_torch.Kron positive control; not exact Li c3",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "tune_starts": int(args.tune_starts),
        "eval_starts": int(args.eval_starts),
        "methods": methods,
        "candidate_grid_sha256": _frame_sha256(
            candidates, ["candidate_id", *CANDIDATE_CONFIG_COLUMNS]
        ),
        "candidate_rows": _json_safe_records(candidates),
        "dependency_validation": dependency_validation,
    }
    _log("strict_protocol_preview " + json.dumps(protocol_preview, sort_keys=True, allow_nan=False))
    if bool(args.dry_run):
        _log("dry_run=accepted dependency/API/grid validation complete; no data loaded and no artifacts written")
        return

    started = time.perf_counter()
    _set_reproducibility(int(args.seed))

    run_dir = _run_dir(str(args.reference_run))
    out_dir = Path(args.output_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"ERROR: output directory must be new or empty to prevent stale artifact mixing: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(
        "startup "
        f"reference_run={args.reference_run} run_dir={run_dir} output_dir={out_dir} "
        f"device={args.device} tune_starts={args.tune_starts} eval_starts={args.eval_starts} "
        f"downstream_steps={args.downstream_steps} primary_endpoint={PRIMARY_ENDPOINT} "
        f"eval_every={args.eval_every} batch_size={args.batch_size} fallback_start_seed={args.seed} methods={methods}"
    )
    _log("cache_mode=read_only_reference_artifacts no_result_cache=true")

    cfg = _load_cfg(
        run_dir,
        device=str(args.device),
        downstream_steps=int(args.downstream_steps),
        eval_every=int(args.eval_every),
        batch_size=int(args.batch_size),
    )
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    _log(
        "resolved_config "
        f"device={device} dtype={dtype} reference_seed={cfg.seed} fallback_start_seed={args.seed} "
        f"data_root={cfg.data_root} download={cfg.download} output_dir={out_dir}"
    )
    _log("candidate_grid\n" + candidates.to_string(index=False))

    _log("stage=load_data")
    task_tensors = _load_task_tensors_for_pipeline(cfg, device=device, dtype=dtype)
    _log(
        "loaded_tasks "
        + ", ".join(
            f"{name}:train={tuple(task.train_images.shape)} test={tuple(task.test_images.shape)}"
            for name, task in task_tensors.items()
        )
    )

    _log("stage=load_reference_weight_pool")
    weights_cpu, weight_records, weight_key, spec = _load_weight_pool(run_dir, cfg)
    if str(spec.model_kind) != "celo_meta_mlp":
        _log(f"WARNING reference spec model_kind={spec.model_kind!r}; this runner is intended for CELO MLP positive controls")
    _log(
        f"reference_weight_pool path={run_dir / 'weight_pool.pt'} cache_key={weight_key} "
        f"weights_shape={tuple(weights_cpu.shape)} records={len(weight_records)} spec_model={spec.model_kind}"
    )

    _log("stage=start_bank")
    val_indices = None if str(args.start_bank_csv).strip() else _load_val_indices_if_available(run_dir)
    start_bank_csv = Path(args.start_bank_csv).expanduser().resolve() if str(args.start_bank_csv).strip() else None
    start_bank = _start_bank(
        start_bank_csv=start_bank_csv,
        eval_starts=int(args.tune_starts) + int(args.eval_starts),
        val_indices=val_indices,
        weight_records=weight_records,
        weights_count=int(weights_cpu.shape[0]),
        seed=int(args.seed),
    )
    _validate_start_bank(start_bank, weights_count=int(weights_cpu.shape[0]))
    tune_bank, eval_bank = _split_protocol_start_bank(
        start_bank,
        tune_starts=int(args.tune_starts),
        eval_starts=int(args.eval_starts),
    )
    _write_csv(start_bank, out_dir / "positive_control_start_bank.csv")
    _write_csv(tune_bank, out_dir / "positive_control_tune_start_bank.csv")
    _write_csv(eval_bank, out_dir / "positive_control_eval_start_bank.csv")
    _write_csv(candidates, out_dir / "candidate_grid.csv")
    _log(
        "start_bank_summary "
        f"total_rows={len(start_bank)} tune_sources={tune_bank['source_weight_index'].astype(int).tolist()} "
        f"eval_sources={eval_bank['source_weight_index'].astype(int).tolist()}"
    )

    _log("stage=tune_grid_evaluation")
    weights_device = weights_cpu.to(device=device, dtype=dtype)

    def run_grid(grid: pd.DataFrame, bank: pd.DataFrame, *, protocol_split: str):
        curve_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        total = int(len(grid) * len(bank))
        progress = make_progress(cfg, total=total, desc=f"positive-control {protocol_split}")
        try:
            for candidate in grid.to_dict(orient="records"):
                method = str(candidate["method"])
                lr = float(candidate["lr"])
                probability_value = candidate.get("kron_precond_update_probability")
                probability = None if pd.isna(probability_value) else float(probability_value)
                _log(
                    f"stage={protocol_split}_candidate method={method} candidate_id={candidate['candidate_id']} "
                    f"lr={lr:g} kron_schedule={candidate['kron_update_schedule']} "
                    f"kron_update_probability={probability} starts={len(bank)}"
                )
                for row_index in range(len(bank)):
                    metadata = bank.iloc[row_index].to_dict()
                    source_index = int(metadata["source_weight_index"])
                    curve_seed = _stable_seed(
                        "kron_positive_control",
                        int(args.seed),
                        protocol_split,
                        source_index,
                    )
                    rows, metrics, diags = _run_raw_curve(
                        cfg=cfg,
                        spec=spec,
                        task_tensors=task_tensors,
                        w0=weights_device[source_index],
                        start_metadata=metadata,
                        method=method,
                        lr=lr,
                        start_index=int(metadata["global_stream_index"]),
                        protocol_split=protocol_split,
                        candidate_id=str(candidate["candidate_id"]),
                        curve_seed=int(curve_seed),
                        kron_update_schedule=str(candidate["kron_update_schedule"]),
                        lr_selection_mode=str(candidate["lr_selection_mode"]),
                        schedule_selection_mode=str(candidate["schedule_selection_mode"]),
                        adam_beta1=float(args.adam_beta1),
                        adam_beta2=float(args.adam_beta2),
                        adam_eps=float(args.adam_eps),
                        sgd_momentum=float(args.sgd_momentum),
                        kron_b1=float(args.kron_b1),
                        kron_weight_decay=float(args.kron_weight_decay),
                        kron_precond_lr=float(args.kron_precond_lr),
                        kron_precond_update_probability=probability,
                        kron_memory_save_mode=str(args.kron_memory_save_mode).strip() or None,
                    )
                    curve_rows.extend(rows)
                    result_rows.append(metrics)
                    diagnostic_rows.extend(diags)
                    progress.set_postfix(
                        {
                            "split": protocol_split,
                            "method": method,
                            "candidate": str(candidate["candidate_id"])[:8],
                            "start": int(row_index),
                            PRIMARY_ENDPOINT: f"{float(metrics[PRIMARY_ENDPOINT]):.4g}",
                        }
                    )
                    progress.update(1)
        finally:
            progress.close()
        return pd.DataFrame(curve_rows), pd.DataFrame(result_rows), pd.DataFrame(diagnostic_rows)

    tuning_curves, tuning_results, tuning_diagnostics = run_grid(candidates, tune_bank, protocol_split="tune")
    _write_csv(tuning_curves, out_dir / "tuning_curves.csv")
    _write_csv(tuning_results, out_dir / "tuning_results.csv")
    _write_csv(tuning_diagnostics, out_dir / "tuning_optimizer_diagnostics.csv")
    selected = _select_frozen_candidates(
        tuning_results,
        candidates=candidates,
        tune_start_bank=tune_bank,
        finite_penalty=float(cfg.finite_penalty),
    )
    _write_csv(selected, out_dir / "selected_configs.csv")
    _log("frozen_selection\n" + selected[selected["selected"].astype(int) == 1].to_string(index=False))

    selected_grid = selected[selected["selected"].astype(int) == 1][
        ["candidate_id", *CANDIDATE_CONFIG_COLUMNS]
    ].copy().reset_index(drop=True)
    _log("stage=frozen_eval_evaluation")
    curves, results, diagnostics = run_grid(selected_grid, eval_bank, protocol_split="eval")
    summary = _summary_table(results)
    paired_contrasts = _paired_eval_contrasts(results)

    _log("stage=output_writing")
    _write_csv(curves, out_dir / "positive_control_curves.csv")
    _write_csv(results, out_dir / "positive_control_results.csv")
    _write_csv(summary, out_dir / "positive_control_summary.csv")
    _write_csv(diagnostics, out_dir / "optimizer_diagnostics.csv")
    _write_csv(paired_contrasts, out_dir / "paired_eval_contrasts.csv")
    _write_paired_contrast_plot(paired_contrasts, out_dir / "paired_eval_post0_aulc.png")

    checkpoint_steps = list(range(0, int(cfg.downstream_steps) + 1, int(cfg.downstream_eval_every)))
    if checkpoint_steps[-1] != int(cfg.downstream_steps):
        checkpoint_steps.append(int(cfg.downstream_steps))
    validation = _protocol_validation(
        candidates=candidates,
        selected=selected,
        tune_bank=tune_bank,
        eval_bank=eval_bank,
        tuning_results=tuning_results,
        eval_results=results,
        tuning_curves=tuning_curves,
        eval_curves=curves,
        tuning_diagnostics=tuning_diagnostics,
        eval_diagnostics=diagnostics,
        downstream_steps=int(cfg.downstream_steps),
        checkpoint_steps=checkpoint_steps,
        finite_penalty=float(cfg.finite_penalty),
    )
    validation_path = out_dir / "protocol_validation.json"
    validation_path.write_text(
        json.dumps(_json_safe(validation), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _log(f"wrote {validation_path} accepted={validation['accepted']}")

    emitted_artifacts = {
        "start_bank": out_dir / "positive_control_start_bank.csv",
        "tune_start_bank": out_dir / "positive_control_tune_start_bank.csv",
        "eval_start_bank": out_dir / "positive_control_eval_start_bank.csv",
        "candidate_grid": out_dir / "candidate_grid.csv",
        "tuning_curves": out_dir / "tuning_curves.csv",
        "tuning_results": out_dir / "tuning_results.csv",
        "tuning_optimizer_diagnostics": out_dir / "tuning_optimizer_diagnostics.csv",
        "selected_configs": out_dir / "selected_configs.csv",
        "protocol_validation": validation_path,
        "curves": out_dir / "positive_control_curves.csv",
        "results": out_dir / "positive_control_results.csv",
        "summary": out_dir / "positive_control_summary.csv",
        "optimizer_diagnostics": out_dir / "optimizer_diagnostics.csv",
        "paired_eval_contrasts": out_dir / "paired_eval_contrasts.csv",
        **(
            {"paired_eval_post0_aulc_plot": out_dir / "paired_eval_post0_aulc.png"}
            if (out_dir / "paired_eval_post0_aulc.png").is_file()
            else {}
        ),
    }
    manifest = {
        "script": _file_signature(Path(__file__).resolve()),
        "cli_args": _json_safe(vars(args)),
        "reference_run": str(args.reference_run),
        "reference_run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "artifact_paths": {
            "manifest": str(out_dir / "manifest.json"),
            **{name: str(path) for name, path in emitted_artifacts.items()},
        },
        "artifact_signatures": {name: _file_signature(path) for name, path in emitted_artifacts.items()},
        "reference_signatures": {
            "config": _file_signature(run_dir / "config.json"),
            "weight_pool": _file_signature(run_dir / "weight_pool.pt"),
            "weight_pool_records": _file_signature(run_dir / "weight_pool_records.csv")
            if (run_dir / "weight_pool_records.csv").is_file()
            else None,
            "selected_lrs": _file_signature(run_dir / "selected_lrs.csv") if (run_dir / "selected_lrs.csv").is_file() else None,
            "vae_checkpoint": _file_signature(run_dir / "vae_checkpoint.pt") if (run_dir / "vae_checkpoint.pt").is_file() else None,
            "start_bank_csv": _file_signature(start_bank_csv) if start_bank_csv is not None else None,
        },
        "resolved_config": {
            "device": str(device),
            "dtype": str(dtype),
            "reference_config_seed": int(cfg.seed),
            "fallback_start_seed": int(args.seed),
            "batch_schedule": (
                "downstream.py:_train_batch_indices offset=global_stream_index*1009+step*batch_size; "
                "tune streams precede disjoint eval streams"
            ),
            "download": bool(cfg.download),
            "data_root": str(cfg.data_root),
            "downstream_steps": int(cfg.downstream_steps),
            "downstream_eval_every": int(cfg.downstream_eval_every),
            "downstream_batch_size": int(cfg.downstream_batch_size),
            "success_threshold": float(cfg.success_threshold),
            "finite_penalty": float(cfg.finite_penalty),
            "weight_distribution": str(cfg.weight_distribution),
            "spec_model_kind": str(spec.model_kind),
            "weight_pool_cache_key": str(weight_key),
        },
        "methods_requested": list(methods),
        "methods_implemented": sorted(IMPLEMENTED_METHODS),
        "causal_scope": "practical raw-space kron_torch.Kron positive control; not exact Li c3",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "dependency_validation": dependency_validation,
        "candidate_grid": _json_safe_records(candidates),
        "candidate_grid_sha256": validation["candidate_grid_sha256"],
        "selected_configs": _json_safe_records(selected[selected["selected"].astype(int) == 1]),
        "selected_configs_sha256": validation["selected_configs_sha256"],
        "optional_dependency_notes": {"kron_whiten_momentum": KRON_IMPORT_MESSAGE},
        "optimizer_hyperparameters": {
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "adam_eps": float(args.adam_eps),
            "sgd_momentum": float(args.sgd_momentum),
            "kron_b1": float(args.kron_b1),
            "kron_weight_decay": float(args.kron_weight_decay),
            "kron_precond_lr": float(args.kron_precond_lr),
            "kron_precond_update_probability": (
                "package_default callable or numeric arm selected from predeclared tune-only grid"
            ),
            "kron_memory_save_mode": str(args.kron_memory_save_mode).strip() or None,
        },
        "start_bank": {
            "rows": int(len(start_bank)),
            "tune_rows": int(len(tune_bank)),
            "eval_rows": int(len(eval_bank)),
            "source_weight_indices": [int(v) for v in start_bank["source_weight_index"].astype(int).tolist()],
            "tune_source_weight_indices": [int(v) for v in tune_bank["source_weight_index"].astype(int).tolist()],
            "eval_source_weight_indices": [int(v) for v in eval_bank["source_weight_index"].astype(int).tolist()],
            "tune_global_stream_indices": [int(v) for v in tune_bank["global_stream_index"].astype(int).tolist()],
            "eval_global_stream_indices": [int(v) for v in eval_bank["global_stream_index"].astype(int).tolist()],
            "selection": sorted(str(v) for v in start_bank["selection"].astype(str).unique().tolist()),
        },
        "row_counts": {
            "tuning_curves": int(len(tuning_curves)),
            "tuning_results": int(len(tuning_results)),
            "tuning_optimizer_diagnostics": int(len(tuning_diagnostics)),
            "curves": int(len(curves)),
            "results": int(len(results)),
            "summary": int(len(summary)),
            "optimizer_diagnostics": int(len(diagnostics)),
            "paired_eval_contrasts": int(len(paired_contrasts)),
        },
        "protocol_validation": validation,
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _log(f"wrote {out_dir / 'manifest.json'}")

    key_rows = summary[summary["metric"].isin([PRIMARY_ENDPOINT, "test_mean_loss", "final_test_acc", "diverged_fraction"])].copy()
    if not key_rows.empty:
        _log("summary\n" + key_rows.to_string(index=False))
    _log(f"done elapsed_sec={time.perf_counter() - started:.2f} output_dir={out_dir}")
    if not bool(validation["accepted"]):
        raise SystemExit(f"ERROR: protocol validation failed; inspect {validation_path}")


if __name__ == "__main__":
    main()
