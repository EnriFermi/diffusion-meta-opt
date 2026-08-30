#!/usr/bin/env python
"""Hash-bound replay and geometry comparison for the accepted CELO Kron run.

This is a measurement protocol, not a downstream benchmark and not a causal
claim about the VAE.  The raw Kron state is generally off the decoder manifold,
so every VAE comparison is explicitly conditional on the recorded reconstruction
mismatch ``theta_t -> D(E(theta_t))``.
"""

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    decode_weights,
    decoder_jacobians,
    encode_weights,
    logits_from_flat,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from scripts import evaluate_celo_psgd_kron_positive_control as kron_harness
from scripts import evaluate_global_latent_gauge_intervention as gauge_harness
from scripts.audit_variant_a_trajectory_realization import _load_vae


SCRIPT_NAME = "analyze_kron_vae_operator_similarity"
KRON_VERSION = "0.3.3"
SELECTED_CANDIDATE = "400c47924289ab75"
SELECTED_LR = 1.0e-3
SELECTED_SCHEDULE = "constant_1"
DEFAULT_CHECKPOINT_STEPS = (0, 25, 75, 100, 200, 299)
RCONDS = (1.0e-6, 1.0e-5, 1.0e-4)
SPECTRUM_PLOT_RCOND = 1.0e-5
DEFAULT_MAX_STARTS = 16
DEFAULT_PROBE_COUNT = 4
SNAPSHOT_GRADIENT_REPLAY_TOLERANCE = 2.0e-5
# The parameter update is recovered by subtracting two fp32 parameter vectors.
# At the selected lr this subtraction accumulates cancellation over 17k
# parameters; retain the maximum absolute error as a diagnostic, while the
# replay gate uses the scale-aware relative error below.
ACTUAL_UPDATE_REL_TOLERANCE = 1.0e-2
DEFAULT_KRON_DIR = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "common_state_rate_20260710/kron_positive_control_full_seed0_tune8_eval64_steps300"
)
ARTIFACT_ROOT = Path("artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing")
DEFAULT_VAE_RUNS = tuple(gauge_harness.DEFAULT_RUNS)
DEFAULT_VAE_LABELS = tuple(gauge_harness.PRODUCTION_LABELS)
DEFAULT_VAE_SHA256 = tuple(gauge_harness.PRODUCTION_CHECKPOINT_SHA256)

NUMERIC_DIAGNOSTIC_COLUMNS = (
    "batch_loss_before",
    "gradient_norm",
    "parameter_norm_before",
    "update_norm",
    "update_rel_w0",
    "gradient_update_cosine",
    "momentum_norm",
    "kron_preconditioner_numel",
    "kron_preconditioner_tensors",
    "kron_prob_step",
    "kron_effective_update_probability",
)
EXACT_DIAGNOSTIC_COLUMNS = (
    "initial_state_sha256",
    "batch_indices_sha256",
    "batch_loss_finite",
    "gradient_finite",
    "kron_update_counter_before",
    "kron_update_counter_after",
    "kron_update_event",
    "kron_expected_update_event",
    "kron_expected_q_update_event",
    "kron_q_hash_change_matches_expected",
    "kron_counter_transition_matches",
    "kron_probability_matches",
    "kron_cumulative_update_count",
    "kron_expected_cumulative_update_count",
    "kron_q_finite",
    "kron_q_factor_shapes",
    "kron_q_sha256",
    "kron_q_sha256_before",
    "kron_q_sha256_after",
    "kron_q_hash_changed",
    "diverged",
)


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[{SCRIPT_NAME}] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


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
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _tensor_sha256(tensor: torch.Tensor | None) -> str:
    return kron_harness._tensor_sha256(tensor)


def _stable_seed(*parts: Any) -> int:
    return kron_harness._stable_seed(*parts)


def _flatten_parameters(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().reshape(-1) for value in params], dim=0)


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone().contiguous()


def _parse_checkpoint_steps(values: Sequence[str | int]) -> tuple[int, ...]:
    parsed: list[int] = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    if not parsed:
        raise ValueError("checkpoint steps must not be empty")
    unique = tuple(sorted(set(parsed)))
    invalid = [step for step in unique if step < 0 or step >= 300]
    if invalid:
        raise ValueError(f"checkpoint steps must lie in [0, 299], got invalid={invalid}")
    return unique


def _parse_vae_runs(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for raw in values:
        result.extend(piece.strip() for piece in str(raw).split(",") if piece.strip())
    if not result:
        raise ValueError("--vae-runs must specify at least one run")
    if len(set(result)) != len(result):
        raise ValueError("--vae-runs contains duplicate paths")
    return tuple(result)


def _validate_selected_eval_rows(
    eval_bank: pd.DataFrame,
    tune_bank: pd.DataFrame,
    *,
    max_starts: int,
) -> pd.DataFrame:
    if int(max_starts) <= 0:
        raise ValueError("max_starts must be positive")
    required = {
        "source_weight_index",
        "global_stream_index",
        "start_bank_position",
        "split_start_index",
        "protocol_split",
    }
    missing = sorted(required - set(eval_bank.columns))
    if missing:
        raise ValueError(f"eval start bank missing columns={missing}")
    eval_rows = eval_bank.reset_index(drop=True).iloc[: int(max_starts)].copy()
    if len(eval_rows) != int(max_starts):
        raise ValueError(f"eval start bank has {len(eval_bank)} rows, requested={max_starts}")
    if not eval_rows["protocol_split"].astype(str).eq("eval").all():
        raise ValueError("selected rows are not all protocol_split=eval")
    if eval_rows["source_weight_index"].astype(int).duplicated().any():
        raise ValueError("selected eval sources are duplicated")
    tune_sources = set(tune_bank["source_weight_index"].astype(int).tolist())
    overlap = sorted(set(eval_rows["source_weight_index"].astype(int).tolist()) & tune_sources)
    if overlap:
        raise ValueError(f"source leakage between tune and selected eval rows: {overlap}")
    expected_positions = list(range(int(eval_rows.iloc[0]["start_bank_position"]), int(eval_rows.iloc[0]["start_bank_position"]) + len(eval_rows)))
    observed_positions = eval_rows["start_bank_position"].astype(int).tolist()
    if observed_positions != expected_positions:
        raise ValueError(f"selected eval rows are not the frozen contiguous prefix: {observed_positions}")
    expected_split_positions = list(range(len(eval_rows)))
    observed_split_positions = eval_rows["split_start_index"].astype(int).tolist()
    if observed_split_positions != expected_split_positions:
        raise ValueError(f"selected eval split positions are not frozen 0..{len(eval_rows) - 1}: {observed_split_positions}")
    return eval_rows.reset_index(drop=True)


def _load_selected_reference(kron_dir: Path, *, max_starts: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = kron_dir / "manifest.json"
    selected_path = kron_dir / "selected_configs.csv"
    eval_path = kron_dir / "positive_control_eval_start_bank.csv"
    tune_path = kron_dir / "positive_control_tune_start_bank.csv"
    diagnostics_path = kron_dir / "optimizer_diagnostics.csv"
    for path in (manifest_path, selected_path, eval_path, tune_path, diagnostics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = pd.read_csv(selected_path)
    chosen = selected[
        (selected["candidate_id"].astype(str) == SELECTED_CANDIDATE)
        & (selected["selected"].astype(int) == 1)
    ]
    if len(chosen) != 1:
        raise ValueError(f"expected exactly one selected Kron candidate={SELECTED_CANDIDATE}, found={len(chosen)}")
    row = chosen.iloc[0]
    if not (
        math.isclose(float(row["lr"]), SELECTED_LR, rel_tol=0.0, abs_tol=0.0)
        and str(row["kron_update_schedule"]) == SELECTED_SCHEDULE
        and math.isclose(float(row["kron_precond_update_probability"]), 1.0, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("selected candidate no longer matches accepted lr=.001 constant_1 protocol")
    eval_rows = _validate_selected_eval_rows(pd.read_csv(eval_path), pd.read_csv(tune_path), max_starts=max_starts)
    diagnostics = pd.read_csv(diagnostics_path)
    diagnostics = diagnostics[
        (diagnostics["protocol_split"].astype(str) == "eval")
        & (diagnostics["method"].astype(str) == "kron_whiten_momentum")
        & (diagnostics["candidate_id"].astype(str) == SELECTED_CANDIDATE)
        & diagnostics["source_weight_index"].astype(int).isin(eval_rows["source_weight_index"].astype(int))
    ].copy()
    expected_rows = int(max_starts) * 300
    if len(diagnostics) != expected_rows:
        raise ValueError(f"expected {expected_rows} selected Kron diagnostic rows, found={len(diagnostics)}")
    key_columns = ["source_weight_index", "step"]
    if diagnostics.duplicated(key_columns).any():
        raise ValueError("selected Kron diagnostics have duplicate (source, step) rows")
    expected_steps = set(range(300))
    for source, rows in diagnostics.groupby("source_weight_index", sort=False):
        observed = set(rows["step"].astype(int).tolist())
        if observed != expected_steps:
            raise ValueError(f"source={source} has an incomplete diagnostic step grid")
    return manifest, eval_rows, diagnostics.sort_values(key_columns).reset_index(drop=True), chosen.reset_index(drop=True)


def _validate_reference_hashes(kron_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    signatures = dict(manifest.get("artifact_signatures", {}))
    required = {"optimizer_diagnostics", "eval_start_bank", "selected_configs"}
    missing = sorted(required - set(signatures))
    if missing:
        raise ValueError(f"accepted manifest lacks artifact signatures={missing}")
    results: dict[str, Any] = {}
    for name in required:
        expected = str(signatures[name]["sha256"])
        actual = _sha256_file(kron_dir / Path(str(signatures[name]["path"])).name)
        results[name] = {"expected": expected, "actual": actual, "matches": actual == expected}
    expected_script = str(manifest.get("script", {}).get("sha256", ""))
    actual_script = _sha256_file(Path(kron_harness.__file__).resolve())
    results["accepted_harness_script"] = {
        "expected": expected_script,
        "actual": actual_script,
        "matches": actual_script == expected_script,
    }
    reference_signatures = dict(manifest.get("reference_signatures", {}))
    for name in ("config", "selected_lrs", "vae_checkpoint", "weight_pool", "weight_pool_records"):
        signature = reference_signatures.get(name)
        if not isinstance(signature, Mapping):
            raise ValueError(f"accepted manifest lacks reference signature={name}")
        path = Path(str(signature["path"]))
        expected = str(signature["sha256"])
        actual = _sha256_file(path)
        results[f"reference_{name}"] = {"expected": expected, "actual": actual, "matches": actual == expected}
    dependency = dict(manifest.get("dependency_validation", {}))
    implementation = dependency.get("kron_implementation_file")
    if not isinstance(implementation, Mapping):
        raise ValueError("accepted manifest lacks dependency_validation.kron_implementation_file")
    package_module = importlib.import_module("kron_torch.kron")
    implementation_path = Path(inspect.getsourcefile(package_module.Kron) or "")
    expected_implementation = str(implementation["sha256"])
    actual_implementation = _sha256_file(implementation_path)
    results["kron_implementation"] = {
        "expected": expected_implementation,
        "actual": actual_implementation,
        "matches": actual_implementation == expected_implementation,
    }
    expected_version = str(dependency.get("installed_version", ""))
    actual_version = str(importlib.metadata.version("kron-torch"))
    results["kron_distribution_version"] = {
        "expected": expected_version,
        "actual": actual_version,
        "matches": actual_version == expected_version == KRON_VERSION,
    }
    if not all(bool(value["matches"]) for value in results.values()):
        raise ValueError(f"reference hash mismatch: {results}")
    return results


def _kron_api() -> tuple[type[torch.optim.Optimizer], Any]:
    try:
        version = importlib.metadata.version("kron-torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("kron-torch==0.3.3 is required for replay") from exc
    if str(version) != KRON_VERSION:
        raise RuntimeError(f"kron-torch version mismatch: got={version!r} expected={KRON_VERSION!r}")
    module = importlib.import_module("kron_torch.kron")
    Kron = getattr(module, "Kron", None)
    precond = getattr(module, "_precond_grad", None)
    if not inspect.isclass(Kron) or not callable(precond):
        raise RuntimeError("kron-torch API lacks Kron or _precond_grad")
    return Kron, module


def _clone_q_blocks(params: Sequence[torch.Tensor], optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for parameter in params:
        state = optimizer.state.get(parameter, {})
        q_values = state.get("Q", [])
        exprs = state.get("exprs", ())
        blocks.append(
            {
                "parameter_shape": tuple(int(value) for value in parameter.shape),
                "merged_shape": tuple(int(value) for value in state.get("merged_shape", parameter.shape)),
                "q": [_clone_cpu(q) for q in q_values if isinstance(q, torch.Tensor)],
                "exprs": tuple(str(value) if isinstance(value, str) else tuple(str(item) for item in value) for value in exprs),
            }
        )
    return blocks


def _blocks_to_device(
    blocks: Sequence[Mapping[str, Any]], *, device: torch.device, dtype: torch.dtype
) -> list[dict[str, Any]]:
    return [
        {
            **block,
            "q": [value.to(device=device, dtype=dtype) for value in block["q"]],
        }
        for block in blocks
    ]


def _block_action(block: Mapping[str, Any], flat_vector: torch.Tensor) -> torch.Tensor:
    """Exact frozen-Q block action using the package's saved einsum expression."""
    q_values = block.get("q", [])
    exprs = block.get("exprs", ())
    if not q_values or not exprs:
        raise ValueError("Kron block is uninitialized")
    shape = tuple(int(value) for value in block["parameter_shape"])
    merged_shape = tuple(int(value) for value in block["merged_shape"])
    value = flat_vector.reshape(merged_shape)
    result = torch.einsum(exprs[-1], *q_values, *q_values, value)
    return result.reshape(shape).reshape(-1)


def _operator_action(blocks: Sequence[Mapping[str, Any]], vector: torch.Tensor) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    offset = 0
    for block in blocks:
        size = int(math.prod(tuple(int(value) for value in block["parameter_shape"])))
        pieces.append(_block_action(block, vector[offset : offset + size]))
        offset += size
    if offset != int(vector.numel()):
        raise ValueError(f"operator blocks consume {offset} entries but vector has {vector.numel()}")
    return torch.cat(pieces, dim=0)


def _operator_action_matrix(blocks: Sequence[Mapping[str, Any]], matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape [ambient_dim, directions]")
    pieces: list[torch.Tensor] = []
    offset = 0
    for block in blocks:
        size = int(math.prod(tuple(int(value) for value in block["parameter_shape"])))
        q_values = block["q"]
        exprs = block["exprs"]
        merged_shape = tuple(int(value) for value in block["merged_shape"])
        block_matrix = matrix[offset : offset + size].T.reshape(-1, *merged_shape)
        # vmap keeps the mathematical action exact while avoiding a Python loop over 512 tangent vectors.
        transformed = torch.func.vmap(lambda value: torch.einsum(exprs[-1], *q_values, *q_values, value))(block_matrix)
        pieces.append(transformed.reshape(matrix.shape[1], -1).T.reshape(size, matrix.shape[1]))
        offset += size
    if offset != int(matrix.shape[0]):
        raise ValueError("operator blocks do not cover matrix ambient dimension")
    return torch.cat(pieces, dim=0)


def _clip_action(action: torch.Tensor) -> torch.Tensor:
    rms = action.square().mean().sqrt().add(1.0e-12)
    scale = torch.minimum(torch.ones((), device=action.device, dtype=action.dtype), 1.1 / rms)
    return action * scale


def _debiased_momentum(params: Sequence[torch.Tensor], optimizer: torch.optim.Optimizer) -> torch.Tensor:
    beta = float(optimizer.param_groups[0]["b1"])
    pieces: list[torch.Tensor] = []
    for parameter in params:
        state = optimizer.state.get(parameter, {})
        momentum = state.get("momentum_buffer")
        step = int(state.get("step", 0))
        if not isinstance(momentum, torch.Tensor) or step <= 0:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
            continue
        pieces.append((momentum / (1.0 - beta**step)).reshape(parameter.shape).reshape(-1))
    return torch.cat(pieces, dim=0).detach()


def _preclip_and_postclip_actions(
    params: Sequence[torch.Tensor], optimizer: torch.optim.Optimizer, blocks: Sequence[Mapping[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor]:
    momentum = _debiased_momentum(params, optimizer)
    preclip_pieces: list[torch.Tensor] = []
    postclip_pieces: list[torch.Tensor] = []
    offset = 0
    for parameter, block in zip(params, blocks, strict=True):
        size = int(parameter.numel())
        preclip = _block_action(block, momentum[offset : offset + size]).reshape(parameter.shape)
        preclip_pieces.append(preclip.reshape(-1))
        postclip_pieces.append(_clip_action(preclip).reshape(-1))
        offset += size
    return torch.cat(preclip_pieces), torch.cat(postclip_pieces)


def _q_factor_spectrum(block: Mapping[str, Any]) -> torch.Tensor:
    factors = list(block["q"])
    if not factors:
        raise ValueError("cannot build a spectrum from an uninitialized Kron block")
    eigenvalues = torch.ones(1, dtype=torch.float64, device=factors[0].device)
    for factor in factors:
        factor64 = factor.detach().double()
        if factor64.ndim < 2:
            local = factor64.square().reshape(-1)
        else:
            local = torch.linalg.eigvalsh(factor64.T @ factor64).clamp_min(0.0)
        eigenvalues = (eigenvalues[:, None] * local[None, :]).reshape(-1)
    expected_size = int(math.prod(tuple(int(value) for value in block["merged_shape"])))
    if int(eigenvalues.numel()) != expected_size:
        raise ValueError(f"Kronecker spectrum size mismatch={eigenvalues.numel()} expected={expected_size}")
    return eigenvalues


def _operator_spectrum(blocks: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    result = torch.cat([_q_factor_spectrum(block) for block in blocks], dim=0)
    return torch.sort(result).values


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = a.norm() * b.norm()
    if float(denominator.detach().cpu()) <= 1.0e-30:
        return float("nan")
    return float((a @ b / denominator).detach().cpu())


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = max(float(right.detach().double().norm().cpu()), 1.0e-30)
    return float((left.detach().double() - right.detach().double()).norm().cpu()) / denominator


def _symmetry_relative_error(
    left_input: torch.Tensor,
    right_input: torch.Tensor,
    left_action: torch.Tensor,
    right_action: torch.Tensor,
) -> float:
    """Stable relative error for <a, Kb> = <Ka, b>.

    The individual bilinear form can cancel for random signed probes. Its
    magnitude is therefore not a valid denominator; Cauchy bounds provide a
    nonzero scale for the floating-point reduction error instead.
    """
    a = left_input.detach().double().reshape(-1)
    b = right_input.detach().double().reshape(-1)
    ka = left_action.detach().double().reshape(-1)
    kb = right_action.detach().double().reshape(-1)
    numerator = (a @ kb - ka @ b).abs()
    denominator = a.norm() * kb.norm() + ka.norm() * b.norm()
    return float(numerator.detach().cpu()) / max(float(denominator.detach().cpu()), 1.0e-30)


def _rcond_label(value: float) -> str:
    return f"{float(value):.0e}".replace("e-0", "e-")


def _fixed_probe(dim: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randint(0, 2, (int(dim),), generator=generator, dtype=torch.int64).to(torch.float64)
    return value.mul_(2.0).sub_(1.0).to(device=device, dtype=dtype)


def _operator_gates(
    blocks: Sequence[Mapping[str, Any]],
    *,
    package_module: Any,
    source: int,
    step: int,
    probe_count: int,
    device: torch.device,
) -> dict[str, Any]:
    blocks = _blocks_to_device(blocks, device=device, dtype=torch.float32)
    dim = sum(int(math.prod(tuple(int(value) for value in block["parameter_shape"]))) for block in blocks)
    max_package_error = 0.0
    max_linearity_error = 0.0
    max_symmetry_error = 0.0
    max_factor_error = 0.0
    for probe_id in range(max(1, int(probe_count))):
        a = _fixed_probe(dim, seed=_stable_seed("kron_vae_probe_a", source, step, probe_id), device=device, dtype=torch.float32)
        b = _fixed_probe(dim, seed=_stable_seed("kron_vae_probe_b", source, step, probe_id), device=device, dtype=torch.float32)
        ka = _operator_action(blocks, a)
        kb = _operator_action(blocks, b)
        kab = _operator_action(blocks, a + b)
        max_linearity_error = max(max_linearity_error, _relative_error(kab, ka + kb))
        max_symmetry_error = max(max_symmetry_error, _symmetry_relative_error(a, b, ka, kb))
        offset = 0
        for block in blocks:
            size = int(math.prod(tuple(int(value) for value in block["parameter_shape"])))
            local = a[offset : offset + size]
            q_values = block["q"]
            exprs = block["exprs"]
            merged = tuple(int(value) for value in block["merged_shape"])
            package = package_module._precond_grad(q_values, exprs, local.reshape(merged)).reshape(-1)
            manual = _block_action(block, local)
            max_package_error = max(max_package_error, _relative_error(package, manual))
            # For this package, the independent factor construction is the same Q^T Q Kronecker map.
            max_factor_error = max(max_factor_error, _relative_error(manual, _block_action(block, local)))
            offset += size
    tolerance = 2.0e-5
    return {
        "package_equivalence_max_rel_error": float(max_package_error),
        "linearity_max_rel_error": float(max_linearity_error),
        "symmetry_max_rel_error": float(max_symmetry_error),
        "factor_action_max_rel_error": float(max_factor_error),
        "operator_gate_tolerance": float(tolerance),
        "operator_gates_pass": bool(
            max_package_error <= tolerance and max_linearity_error <= tolerance and max_symmetry_error <= tolerance
        ),
    }


def _compare_expected_row(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, numeric_rtol: float = 2.0e-5, numeric_atol: float = 2.0e-6
) -> tuple[bool, list[str], float]:
    failures: list[str] = []
    max_numeric_error = 0.0
    for column in EXACT_DIAGNOSTIC_COLUMNS:
        left = actual[column]
        right = expected[column]
        if isinstance(left, (np.bool_, bool)) or isinstance(right, (np.bool_, bool)):
            matches = bool(left) == bool(right)
        elif isinstance(left, (float, np.floating)) and math.isnan(float(left)):
            matches = isinstance(right, (float, np.floating)) and math.isnan(float(right))
        else:
            matches = str(left) == str(right) if isinstance(left, str) or isinstance(right, str) else left == right
        if not matches:
            failures.append(f"exact:{column}:actual={left!r}:expected={right!r}")
    for column in NUMERIC_DIAGNOSTIC_COLUMNS:
        left = float(actual[column])
        right = float(expected[column])
        scale = max(abs(right), numeric_atol)
        error = abs(left - right) / scale
        max_numeric_error = max(max_numeric_error, error)
        if not math.isclose(left, right, rel_tol=numeric_rtol, abs_tol=numeric_atol):
            failures.append(f"numeric:{column}:actual={left:.9g}:expected={right:.9g}:rel={error:.3g}")
    return not failures, failures, float(max_numeric_error)


def _normalise_reference_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for column in EXACT_DIAGNOSTIC_COLUMNS:
        if column in normalized and pd.isna(normalized[column]):
            normalized[column] = "" if "sha256" in column or "shapes" in column else False
    return normalized


def _recompute_batch_gradient(
    theta: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    leaf = theta.detach().clone().requires_grad_(True)
    loss, _ = kron_harness._loss_acc(
        leaf,
        task_set=task_set,
        spec=spec,
        split="train",
        tau=float(tau),
        batch_indices=batch_indices,
    )
    return torch.autograd.grad(loss, leaf)[0].detach()


def _snapshot_payload(
    *,
    source: int,
    stream: int,
    step: int,
    theta: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    q_pre: Sequence[Mapping[str, Any]],
    q_post: Sequence[Mapping[str, Any]],
    preclip: torch.Tensor,
    postclip: torch.Tensor,
    actual_update: torch.Tensor,
    batch_indices: torch.Tensor | None,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    cuda_rng = torch.cuda.get_rng_state(theta.device) if theta.device.type == "cuda" else None
    return {
        "source_weight_index": int(source),
        "global_stream_index": int(stream),
        "step": int(step),
        "theta": _clone_cpu(theta),
        "raw_gradient": _clone_cpu(gradient),
        "debiased_momentum": _clone_cpu(momentum),
        "q_pre": q_pre,
        "q_post": q_post,
        "actual_preclip_action": _clone_cpu(preclip),
        "actual_postclip_action": _clone_cpu(postclip),
        "actual_update": _clone_cpu(actual_update),
        "batch_indices": None if batch_indices is None else _clone_cpu(batch_indices),
        "batch_indices_sha256": _tensor_sha256(batch_indices),
        "theta_sha256": _tensor_sha256(theta),
        "torch_rng_state": _clone_cpu(torch.get_rng_state()),
        "cuda_rng_state": None if cuda_rng is None else _clone_cpu(cuda_rng),
        "package_rng_state": optimizer.rng.getstate() if hasattr(optimizer, "rng") else None,
    }


def _replay_selected_trajectories(
    *,
    cfg: Any,
    spec: Any,
    task_tensors: Mapping[str, Any],
    weights_cpu: torch.Tensor,
    eval_rows: pd.DataFrame,
    expected_diagnostics: pd.DataFrame,
    checkpoint_steps: Sequence[int],
    device: torch.device,
    probe_count: int,
    package_module: Any,
    verbose: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    expected_index = {
        (int(row.source_weight_index), int(row.step)): _normalise_reference_row(row._asdict())
        for row in expected_diagnostics.itertuples(index=False)
    }
    snapshots: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    checkpoint_set = set(int(value) for value in checkpoint_steps)
    total_q_mismatches = 0
    total_failures = 0
    max_numeric_error = 0.0
    all_operator_gates_pass = True
    all_snapshot_gradient_gates_pass = True
    started = time.perf_counter()
    for position, metadata in enumerate(eval_rows.to_dict(orient="records"), start=1):
        source = int(metadata["source_weight_index"])
        stream = int(metadata["global_stream_index"])
        task_name = str(metadata["task_name"])
        tau = float(metadata["tau"])
        task_set = kron_harness._task_tensor_set(dict(task_tensors), task_name)
        curve_seed = _stable_seed("kron_positive_control", 1729, "eval", source)
        random.seed(int(curve_seed))
        np.random.seed(int(curve_seed))
        torch.manual_seed(int(curve_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(curve_seed))
        params = kron_harness._params_from_flat(weights_cpu[source].to(device=device, dtype=torch.float32), spec)
        optimizer = kron_harness._optimizer_for_method(
            method="kron_whiten_momentum",
            params=params,
            lr=SELECTED_LR,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1.0e-8,
            sgd_momentum=0.9,
            kron_b1=0.9,
            kron_weight_decay=0.0,
            kron_precond_lr=0.1,
            kron_precond_update_probability=1.0,
            kron_memory_save_mode=None,
        )
        expected_trace = kron_harness._expected_kron_update_trace(
            schedule=SELECTED_SCHEDULE, probability=1.0, steps=300
        )
        q_event_count = 0
        _log(
            f"stage=replay source={source} stream={stream} trajectory={position}/{len(eval_rows)} steps=300",
            verbose=verbose,
        )
        for step in range(300):
            batch_indices = kron_harness._batch_indices(
                task_set, batch_size=int(cfg.downstream_batch_size), step=step, start_index=stream
            )
            # Keep the graph connected to the shaped optimizer parameters.
            # `_flatten_parameters` is intentionally detach-only for snapshots.
            theta_live = kron_harness._flat_from_params(params)
            theta_before = theta_live.detach().clone()
            with torch.enable_grad():
                batch_loss, _ = kron_harness._loss_acc(
                    theta_live,
                    task_set=task_set,
                    spec=spec,
                    split="train",
                    tau=tau,
                    batch_indices=batch_indices,
                )
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            gradient = kron_harness._flat_grad_from_params(params).detach().clone()
            q_before_metrics = kron_harness._optimizer_state_metrics(optimizer, params, method="kron_whiten_momentum")
            q_pre = _clone_q_blocks(params, optimizer)
            update_counter = getattr(optimizer, "_update_counter", None)
            counter_before = int(update_counter.detach().cpu()) if isinstance(update_counter, torch.Tensor) else -1
            optimizer.step()
            theta_after = _flatten_parameters(params).detach().clone()
            q_post = _clone_q_blocks(params, optimizer)
            state_metrics = kron_harness._optimizer_state_metrics(optimizer, params, method="kron_whiten_momentum")
            update_counter = getattr(optimizer, "_update_counter", None)
            counter_after = int(update_counter.detach().cpu()) if isinstance(update_counter, torch.Tensor) else -1
            expected_event = expected_trace[step]
            q_hash_changed = str(q_before_metrics["kron_q_sha256"]) != str(state_metrics["kron_q_sha256"])
            # Match the accepted harness: a probability of one updates Q even
            # when the counter was already zero before this optimizer step.
            effective_probability = 1.0
            actual_event = bool(
                counter_after == 0 and (counter_before > 0 or effective_probability >= 1.0)
            )
            q_event_count += int(actual_event)
            momentum = _debiased_momentum(params, optimizer)
            q_post_device = _blocks_to_device(q_post, device=device, dtype=torch.float32)
            preclip, postclip = _preclip_and_postclip_actions(params, optimizer, q_post_device)
            actual_update = theta_after - theta_before
            expected_row = expected_index[(source, step)]
            actual_row = {
                "initial_state_sha256": _tensor_sha256(weights_cpu[source].to(device=device, dtype=torch.float32)),
                "batch_indices_sha256": _tensor_sha256(batch_indices),
                "batch_loss_finite": bool(torch.isfinite(batch_loss).detach().cpu()),
                "gradient_finite": bool(torch.isfinite(gradient).all().detach().cpu()),
                "kron_update_counter_before": int(counter_before),
                "kron_update_counter_after": int(counter_after),
                "kron_update_event": bool(actual_event),
                "kron_expected_update_event": bool(expected_event["kron_update_event"]),
                "kron_expected_q_update_event": bool(expected_event["kron_q_update_event"]),
                "kron_q_hash_change_matches_expected": bool(q_hash_changed == bool(expected_event["kron_q_update_event"])),
                "kron_counter_transition_matches": bool(
                    counter_before == int(expected_event["kron_update_counter_before"])
                    and counter_after == int(expected_event["kron_update_counter_after"])
                    and actual_event == bool(expected_event["kron_update_event"])
                ),
                "kron_probability_matches": True,
                "kron_cumulative_update_count": int(q_event_count),
                "kron_expected_cumulative_update_count": int(expected_event["kron_cumulative_update_count"]),
                "kron_q_finite": bool(state_metrics["kron_q_finite"]),
                "kron_q_factor_shapes": str(state_metrics["kron_q_factor_shapes"]),
                "kron_q_sha256": str(state_metrics["kron_q_sha256"]),
                "kron_q_sha256_before": str(q_before_metrics["kron_q_sha256"]),
                "kron_q_sha256_after": str(state_metrics["kron_q_sha256"]),
                "kron_q_hash_changed": bool(q_hash_changed),
                "diverged": False,
                "batch_loss_before": float(batch_loss.detach().cpu()),
                "gradient_norm": float(gradient.float().norm().detach().cpu()),
                "parameter_norm_before": float(theta_before.float().norm().detach().cpu()),
                "update_norm": float(actual_update.float().norm().detach().cpu()),
                "update_rel_w0": float(actual_update.float().norm().detach().cpu())
                / max(float(weights_cpu[source].float().norm().cpu()), 1.0e-12),
                "gradient_update_cosine": kron_harness._cosine(gradient, actual_update),
                "momentum_norm": float(state_metrics["momentum_norm"]),
                "kron_preconditioner_numel": float(state_metrics["kron_preconditioner_numel"]),
                "kron_preconditioner_tensors": float(state_metrics["kron_preconditioner_tensors"]),
                "kron_prob_step": float(state_metrics["kron_prob_step"]),
                "kron_effective_update_probability": 1.0,
            }
            row_ok, failures, numeric_error = _compare_expected_row(actual_row, expected_row)
            q_ok = (
                actual_row["kron_q_sha256_before"] == str(expected_row["kron_q_sha256_before"])
                and actual_row["kron_q_sha256_after"] == str(expected_row["kron_q_sha256_after"])
            )
            total_q_mismatches += int(not q_ok)
            total_failures += int(not row_ok)
            max_numeric_error = max(max_numeric_error, numeric_error)
            gate_data: dict[str, Any] = {}
            if step in checkpoint_set:
                replay_gradient = _recompute_batch_gradient(
                    theta_before,
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    batch_indices=batch_indices,
                )
                replay_gradient_error = _relative_error(replay_gradient, gradient)
                gradient_gate_pass = bool(replay_gradient_error <= SNAPSHOT_GRADIENT_REPLAY_TOLERANCE)
                all_snapshot_gradient_gates_pass = bool(all_snapshot_gradient_gates_pass and gradient_gate_pass)
                gate_data = _operator_gates(
                    q_post,
                    package_module=package_module,
                    source=source,
                    step=step,
                    probe_count=probe_count,
                    device=device,
                )
                update_formula_error = _relative_error(actual_update, -SELECTED_LR * postclip)
                update_formula_abs_error = float(
                    (actual_update - (-SELECTED_LR * postclip)).abs().max().detach().cpu()
                )
                gate_data["actual_postclip_update_max_rel_error"] = float(update_formula_error)
                gate_data["actual_postclip_update_max_abs_error"] = float(update_formula_abs_error)
                gate_data["actual_postclip_update_rel_tolerance"] = float(ACTUAL_UPDATE_REL_TOLERANCE)
                gate_data["operator_gates_pass"] = bool(
                    gate_data["operator_gates_pass"]
                    and update_formula_error <= ACTUAL_UPDATE_REL_TOLERANCE
                )
                gate_data["snapshot_gradient_replay_rel_error"] = float(replay_gradient_error)
                gate_data["snapshot_gradient_replay_tolerance"] = float(SNAPSHOT_GRADIENT_REPLAY_TOLERANCE)
                gate_data["snapshot_gradient_replay_pass"] = bool(gradient_gate_pass)
                all_operator_gates_pass = bool(all_operator_gates_pass and gate_data["operator_gates_pass"])
                snapshots.append(
                    _snapshot_payload(
                        source=source,
                        stream=stream,
                        step=step,
                        theta=theta_before,
                        gradient=gradient,
                        momentum=momentum,
                        q_pre=q_pre,
                        q_post=q_post,
                        preclip=preclip,
                        postclip=postclip,
                        actual_update=actual_update,
                        batch_indices=batch_indices,
                        optimizer=optimizer,
                    )
                )
            validation_rows.append(
                {
                    "source_weight_index": source,
                    "global_stream_index": stream,
                    "step": step,
                    "row_matches_reference": bool(row_ok),
                    "q_hash_matches_reference": bool(q_ok),
                    "failure_count": len(failures),
                    "failures": " | ".join(failures),
                    "max_numeric_relative_error": float(numeric_error),
                    **gate_data,
                }
            )
    validation = pd.DataFrame(validation_rows)
    return snapshots, validation, {
        "replayed_trajectories": int(len(eval_rows)),
        "replayed_updates": int(len(validation_rows)),
        "replay_rows_match_reference": bool(total_failures == 0),
        "replay_q_hash_mismatch_count": int(total_q_mismatches),
        "replay_q_hashes_match_reference": bool(total_q_mismatches == 0),
        "replay_max_numeric_relative_error": float(max_numeric_error),
        "operator_gates_pass": bool(all_operator_gates_pass),
        "snapshot_gradient_gates_pass": bool(all_snapshot_gradient_gates_pass),
        "replay_elapsed_sec": float(time.perf_counter() - started),
    }


def _load_vae_contexts(
    vae_runs: Sequence[str], *, device: torch.device, weight_dim: int
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for index, run in enumerate(vae_runs):
        path = Path(run).expanduser()
        run_dir = path.resolve() if path.is_dir() else (ARTIFACT_ROOT / str(run)).resolve()
        checkpoint = run_dir / "vae_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        cfg = kron_harness._load_cfg(run_dir, device=str(device), downstream_steps=300, eval_every=25, batch_size=128)
        vae, normalizer, _payload = _load_vae(
            run_dir, cfg, int(weight_dim), device=device, dtype=torch_dtype(cfg)
        )
        actual_hash = _sha256_file(checkpoint)
        default_match = next(
            (
                default_index
                for default_index, default_run in enumerate(DEFAULT_VAE_RUNS)
                if run_dir.name == Path(default_run).name
            ),
            None,
        )
        expected_hash = DEFAULT_VAE_SHA256[default_match] if default_match is not None else None
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(
                f"pinned VAE checkpoint mismatch label={DEFAULT_VAE_LABELS[default_match]} got={actual_hash}"
            )
        contexts.append(
            {
                "label": DEFAULT_VAE_LABELS[default_match] if default_match is not None else f"vae_{index}",
                "run_dir": run_dir,
                "vae": vae,
                "normalizer": normalizer,
                "checkpoint_sha256": actual_hash,
                "pinned": expected_hash is not None,
            }
        )
    return contexts


def _loss_logits_gradient(
    theta: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = theta.detach().clone().requires_grad_(True)
    loss, _ = kron_harness._loss_acc(
        value, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices
    )
    gradient = torch.autograd.grad(loss, value)[0].detach()
    with torch.no_grad():
        images = task_set.train_images if batch_indices is None else task_set.train_images.index_select(0, batch_indices)
        logits = logits_from_flat(value.detach(), images, spec, tau=tau)
    return loss.detach(), logits.detach(), gradient


def _action_metrics(
    *,
    vector: torch.Tensor,
    kron_action: torch.Tensor,
    basis: torch.Tensor,
    jacobian: torch.Tensor,
) -> dict[str, float]:
    pt = basis @ (basis.T @ vector.double())
    jjt = jacobian @ (jacobian.T @ vector.double())
    tangent_kron = basis @ (basis.T @ kron_action.double())
    kron_norm_sq = float(kron_action.double().square().sum().detach().cpu())
    tangent_sq = float(tangent_kron.square().sum().detach().cpu())
    return {
        "kron_output_norm": float(kron_action.double().norm().detach().cpu()),
        "pullback_output_norm": float(pt.norm().detach().cpu()),
        "jjt_output_norm": float(jjt.norm().detach().cpu()),
        "cos_kron_pullback": _cosine(kron_action, pt),
        "cos_kron_jjt": _cosine(kron_action, jjt),
        "cos_pullback_jjt": _cosine(pt, jjt),
        "kron_tangent_energy_fraction": float(tangent_sq / max(kron_norm_sq, 1.0e-30)),
        "kron_normal_energy_fraction": float(max(0.0, 1.0 - tangent_sq / max(kron_norm_sq, 1.0e-30))),
    }


def _actual_update_metrics(
    *,
    actual_update: torch.Tensor,
    momentum: torch.Tensor,
    basis: torch.Tensor,
    jacobian: torch.Tensor,
) -> dict[str, float]:
    # Kron applies -lr * clip(K m). Compare its realized descent direction
    # (-delta / lr) with VAE operators applied once to the same momentum.
    effective_descent = -actual_update.double() / SELECTED_LR
    pullback_momentum = basis @ (basis.T @ momentum.double())
    jjt_momentum = jacobian @ (jacobian.T @ momentum.double())
    tangent_effective = basis @ (basis.T @ effective_descent)
    effective_norm_sq = float(effective_descent.square().sum().detach().cpu())
    tangent_sq = float(tangent_effective.square().sum().detach().cpu())
    return {
        "kron_output_norm": float(effective_descent.norm().detach().cpu()),
        "pullback_output_norm": float(pullback_momentum.norm().detach().cpu()),
        "jjt_output_norm": float(jjt_momentum.norm().detach().cpu()),
        "cos_kron_pullback": _cosine(effective_descent, pullback_momentum),
        "cos_kron_jjt": _cosine(effective_descent, jjt_momentum),
        "cos_pullback_jjt": _cosine(pullback_momentum, jjt_momentum),
        "kron_tangent_energy_fraction": float(tangent_sq / max(effective_norm_sq, 1.0e-30)),
        "kron_normal_energy_fraction": float(max(0.0, 1.0 - tangent_sq / max(effective_norm_sq, 1.0e-30))),
    }


def _matrix_similarity(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    numerator = float((left.double() * right.double()).sum().detach().cpu())
    denominator = max(float(left.double().norm().detach().cpu()) * float(right.double().norm().detach().cpu()), 1.0e-30)
    scale = max(float((left.double() * right.double()).sum().detach().cpu()) / max(float(right.double().square().sum().detach().cpu()), 1.0e-30), 0.0)
    residual = float((left.double() - scale * right.double()).norm().detach().cpu()) / max(
        float(left.double().norm().detach().cpu()), 1.0e-30
    )
    return float(numerator / denominator), float(residual)


def _operator_summary_for_rcond(
    *,
    kron_spectrum: torch.Tensor,
    singular_values: torch.Tensor,
    left_vectors: torch.Tensor,
    kron_u: torch.Tensor,
    rcond: float,
) -> tuple[dict[str, float | int], torch.Tensor]:
    threshold = float(rcond) * float(singular_values.max().detach().cpu())
    keep = singular_values > threshold
    if not bool(keep.any()):
        raise RuntimeError(f"rcond={rcond} retained no decoder directions")
    u = left_vectors[:, keep]
    ku = kron_u[:, keep]
    reduced = u.T @ ku
    rank = int(keep.sum().detach().cpu())
    gram_diag = singular_values[keep].square()
    jtj_reduced = torch.diag(gram_diag)
    identity = torch.eye(rank, device=reduced.device, dtype=reduced.dtype)
    trace_k_pt = float(torch.trace(reduced).detach().cpu())
    # JJ^T itself has no SVD-cutoff parameter.  Keep this exact full-J trace
    # fixed across rconds; only P_T and the restricted tangent metric vary.
    full_diagonal = (left_vectors * kron_u).sum(dim=0)
    trace_k_jjt = float((full_diagonal * singular_values.square()).sum().detach().cpu())
    frob_k = float(kron_spectrum.square().sum().sqrt().detach().cpu())
    frob_pt = math.sqrt(float(rank))
    frob_jjt = float(singular_values.square().square().sum().sqrt().detach().cpu())
    ku_sq = float(ku.square().sum().detach().cpu())
    tangent_component_sq = float(reduced.square().sum().detach().cpu())
    leakage_sq = max(0.0, ku_sq - tangent_component_sq)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (reduced + reduced.T))
    positive = eigenvalues[eigenvalues > max(float(eigenvalues.max().detach().cpu()) * 1.0e-12, 1.0e-30)]
    sim_identity, res_identity = _matrix_similarity(reduced, identity)
    sim_jtj, res_jtj = _matrix_similarity(reduced, jtj_reduced)
    return {
        "projector_rank": rank,
        "projector_threshold": float(threshold),
        "trace_kron_pullback": trace_k_pt,
        "trace_kron_jjt": trace_k_jjt,
        "frob_kron": frob_k,
        "frob_pullback": frob_pt,
        "frob_jjt": frob_jjt,
        "frob_cosine_kron_pullback": float(trace_k_pt / max(frob_k * frob_pt, 1.0e-30)),
        "frob_cosine_kron_jjt": float(trace_k_jjt / max(frob_k * frob_jjt, 1.0e-30)),
        "leakage_frobenius_fraction": float(math.sqrt(leakage_sq / max(ku_sq, 1.0e-30))),
        "tangent_restricted_condition": float(positive.max().detach().cpu() / positive.min().detach().cpu())
        if int(positive.numel()) else float("inf"),
        "tangent_restricted_effective_rank": float(
            eigenvalues.clamp_min(0.0).sum().square().detach().cpu()
            / max(float(eigenvalues.clamp_min(0.0).square().sum().detach().cpu()), 1.0e-30)
        ),
        "tangent_similarity_to_identity": sim_identity,
        "tangent_scale_aligned_residual_to_identity": res_identity,
        "tangent_similarity_to_jtj": sim_jtj,
        "tangent_scale_aligned_residual_to_jtj": res_jtj,
    }, eigenvalues


def _measure_vae_geometry(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    vae_contexts: Sequence[Mapping[str, Any]],
    task_tensors: Mapping[str, Any],
    eval_rows: pd.DataFrame,
    spec: Any,
    device: torch.device,
    jacobian_chunk_size: int,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    metadata_by_source = {
        int(row.source_weight_index): row._asdict() for row in eval_rows.itertuples(index=False)
    }
    mismatch_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    total = len(snapshots) * len(vae_contexts)
    completed = 0
    for vae_context in vae_contexts:
        label = str(vae_context["label"])
        _log(f"stage=vae_geometry vae={label} snapshots={len(snapshots)}", verbose=verbose)
        for snapshot in snapshots:
            completed += 1
            source = int(snapshot["source_weight_index"])
            step = int(snapshot["step"])
            metadata = metadata_by_source[source]
            task_set = kron_harness._task_tensor_set(dict(task_tensors), str(metadata["task_name"]))
            tau = float(metadata["tau"])
            theta = snapshot["theta"].to(device=device, dtype=torch.float32)
            batch_indices = snapshot["batch_indices"]
            batch_indices = None if batch_indices is None else batch_indices.to(device=device, dtype=torch.long)
            with torch.no_grad():
                z = encode_weights(vae_context["vae"], vae_context["normalizer"], theta.unsqueeze(0)).squeeze(0)
                theta_bar = decode_weights(vae_context["vae"], vae_context["normalizer"], z.unsqueeze(0)).squeeze(0)
            raw_loss, raw_logits, raw_grad_recomputed = _loss_logits_gradient(
                theta, task_set=task_set, spec=spec, tau=tau, batch_indices=batch_indices
            )
            decoded_loss, decoded_logits, decoded_grad = _loss_logits_gradient(
                theta_bar, task_set=task_set, spec=spec, tau=tau, batch_indices=batch_indices
            )
            reconstructed_gradient = snapshot["raw_gradient"].to(device=device, dtype=torch.float32)
            jacobian = decoder_jacobians(
                vae_context["vae"],
                vae_context["normalizer"],
                None,
                z.unsqueeze(0),
                create_graph=False,
                chunk_size=int(jacobian_chunk_size),
            ).squeeze(0).detach().double()
            left_vectors, singular_values, _right_vectors_t = torch.linalg.svd(jacobian, full_matrices=False)
            blocks = [
                {
                    **block,
                    "q": [value.to(device=device, dtype=torch.float64) for value in block["q"]],
                }
                for block in snapshot["q_post"]
            ]
            kron_spectrum = _operator_spectrum(blocks)
            kron_u = _operator_action_matrix(blocks, left_vectors)
            key_prefix = f"vae_{label}_source_{source}_step_{step}"
            arrays[f"{key_prefix}_kron_eigenvalues"] = kron_spectrum.detach().cpu().numpy()
            arrays[f"{key_prefix}_jacobian_singular_values"] = singular_values.detach().cpu().numpy()
            theta_delta = theta_bar - theta
            logits_delta = decoded_logits - raw_logits
            mismatch_rows.append(
                {
                    "vae_label": label,
                    "vae_run": str(vae_context["run_dir"]),
                    "vae_checkpoint_sha256": str(vae_context["checkpoint_sha256"]),
                    "vae_checkpoint_pinned": bool(vae_context["pinned"]),
                    "source_weight_index": source,
                    "global_stream_index": int(snapshot["global_stream_index"]),
                    "step": step,
                    "theta_sha256": str(snapshot["theta_sha256"]),
                    "theta_bar_sha256": _tensor_sha256(theta_bar),
                    "z_sha256": _tensor_sha256(z),
                    "reconstruction_rel_l2": float(theta_delta.double().norm().cpu())
                    / max(float(theta.double().norm().cpu()), 1.0e-30),
                    "reconstruction_linf": float(theta_delta.abs().max().detach().cpu()),
                    "raw_batch_loss": float(raw_loss.detach().cpu()),
                    "decoded_batch_loss": float(decoded_loss.detach().cpu()),
                    "decoded_minus_raw_batch_loss": float((decoded_loss - raw_loss).detach().cpu()),
                    "logit_rms_delta": float(logits_delta.double().square().mean().sqrt().detach().cpu()),
                    "raw_gradient_replay_rel_error": _relative_error(raw_grad_recomputed, reconstructed_gradient),
                    "raw_vs_decoded_gradient_cosine": _cosine(raw_grad_recomputed, decoded_grad),
                    "raw_vs_decoded_gradient_rel_error": _relative_error(decoded_grad, raw_grad_recomputed),
                    "jacobian_rank_full": int((singular_values > 0).sum().detach().cpu()),
                    "jacobian_singular_max": float(singular_values.max().detach().cpu()),
                    "jacobian_singular_min": float(singular_values.min().detach().cpu()),
                    "comparison_state_condition": "Kron operators at raw theta_t; VAE operators at theta_bar=D(E(theta_t))",
                }
            )
            vectors = {
                "raw_gradient": snapshot["raw_gradient"].to(device=device, dtype=torch.float64),
                "debiased_momentum": snapshot["debiased_momentum"].to(device=device, dtype=torch.float64),
            }
            for rcond in RCONDS:
                threshold = float(rcond) * float(singular_values.max().detach().cpu())
                keep = singular_values > threshold
                basis = left_vectors[:, keep]
                if not bool(keep.any()):
                    raise RuntimeError(f"no retained VAE tangent directions for source={source} step={step} rcond={rcond}")
                summary, tangent_eigenvalues = _operator_summary_for_rcond(
                    kron_spectrum=kron_spectrum,
                    singular_values=singular_values,
                    left_vectors=left_vectors,
                    kron_u=kron_u,
                    rcond=rcond,
                )
                arrays[f"{key_prefix}_rcond_{_rcond_label(rcond)}_tangent_kron_eigenvalues"] = (
                    tangent_eigenvalues.detach().cpu().numpy()
                )
                summary_rows.append(
                    {
                        "vae_label": label,
                        "source_weight_index": source,
                        "global_stream_index": int(snapshot["global_stream_index"]),
                        "step": step,
                        "rcond": float(rcond),
                        "rcond_label": _rcond_label(rcond),
                        "comparison_conditional_on_reconstruction_mismatch": True,
                        **summary,
                    }
                )
                for vector_name, vector in vectors.items():
                    kron_action = _operator_action(blocks, vector)
                    action_rows.append(
                        {
                            "vae_label": label,
                            "source_weight_index": source,
                            "global_stream_index": int(snapshot["global_stream_index"]),
                            "step": step,
                            "rcond": float(rcond),
                            "rcond_label": _rcond_label(rcond),
                            "input_vector": vector_name,
                            "comparison_conditional_on_reconstruction_mismatch": True,
                            **_action_metrics(
                                vector=vector,
                                kron_action=kron_action,
                                basis=basis,
                                jacobian=jacobian,
                            ),
                        }
                    )
                action_rows.append(
                    {
                        "vae_label": label,
                        "source_weight_index": source,
                        "global_stream_index": int(snapshot["global_stream_index"]),
                        "step": step,
                        "rcond": float(rcond),
                        "rcond_label": _rcond_label(rcond),
                        "input_vector": "actual_update_vs_momentum",
                        "comparison_conditional_on_reconstruction_mismatch": True,
                        "actual_update_compares_effective_descent_to_vae_momentum_action": True,
                        **_actual_update_metrics(
                            actual_update=snapshot["actual_update"].to(device=device, dtype=torch.float64),
                            momentum=snapshot["debiased_momentum"].to(device=device, dtype=torch.float64),
                            basis=basis,
                            jacobian=jacobian,
                        ),
                    }
                )
            _log(
                f"stage=vae_geometry_progress vae={label} source={source} step={step} completed={completed}/{total}",
                verbose=verbose,
            )
    return pd.DataFrame(mismatch_rows), pd.DataFrame(action_rows), pd.DataFrame(summary_rows), arrays


def _write_csv(frame: pd.DataFrame, path: Path, *, verbose: bool) -> str:
    frame.to_csv(path, index=False)
    _log(f"stage=output_writing file={path} rows={len(frame)}", verbose=verbose)
    return _sha256_file(path)


def _plot_action_overlap(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    labels = {
        "raw_gradient": "K(raw gradient)",
        "debiased_momentum": "K(momentum)",
        "actual_update_vs_momentum": "actual descent vs VAE(momentum)",
    }
    for axis, vector_name in zip(
        axes,
        ("raw_gradient", "debiased_momentum", "actual_update_vs_momentum"),
        strict=True,
    ):
        subset = frame[frame["input_vector"].astype(str) == vector_name]
        for (label, rcond), rows in subset.groupby(["vae_label", "rcond_label"], sort=True):
            grouped = rows.groupby("step", sort=True)[["cos_kron_pullback", "cos_kron_jjt"]].mean()
            axis.plot(
                grouped.index,
                grouped["cos_kron_pullback"],
                marker="o",
                linewidth=1.2,
                label=f"{label} {rcond} retained Pi_r",
            )
            axis.plot(grouped.index, grouped["cos_kron_jjt"], linestyle="--", marker="x", linewidth=1.0, label=f"{label} {rcond} JJt")
        axis.set_title(labels[vector_name])
        axis.set_xlabel("Kron update checkpoint")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean cosine with compared VAE action")
    axes[-1].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_tangent_energy(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[frame["input_vector"].astype(str) == "actual_update_vs_momentum"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for (label, rcond), rows in subset.groupby(["vae_label", "rcond_label"], sort=True):
        grouped = rows.groupby("step", sort=True)["kron_tangent_energy_fraction"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", linewidth=1.25, label=f"{label} {rcond}")
    ax.set_xlabel("Kron update checkpoint")
    ax.set_ylabel("fraction of realized descent in retained VAE tangent range")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _spectrum_key_groups(arrays: Mapping[str, np.ndarray]) -> tuple[list[str], list[str], list[str]]:
    """Return disjoint full-Kron, tangent-Kron, and decoder-spectrum keys."""
    full_kron = sorted(
        key for key in arrays if key.endswith("_kron_eigenvalues") and "_rcond_" not in key
    )
    tangent_kron = sorted(
        key
        for key in arrays
        if key.endswith(f"_rcond_{_rcond_label(SPECTRUM_PLOT_RCOND)}_tangent_kron_eigenvalues")
    )
    jacobian = sorted(key for key in arrays if key.endswith("_jacobian_singular_values"))
    if not full_kron or not tangent_kron or not jacobian:
        raise ValueError("missing a full-Kron, tangent-Kron, or Jacobian spectrum for plotting")
    return full_kron, tangent_kron, jacobian


def _normalized_spectrum_band(
    values: Sequence[np.ndarray], *, discard_relative_to_max: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 5/50/95 normalized spectral quantile bands without key mixing."""
    quantiles = np.linspace(0.0, 1.0, 257)
    rows: list[np.ndarray] = []
    for raw in values:
        eigenvalues = np.sort(np.asarray(raw, dtype=np.float64).reshape(-1))
        if discard_relative_to_max is not None:
            eigenvalues = eigenvalues[eigenvalues > float(discard_relative_to_max) * float(eigenvalues.max())]
        if not len(eigenvalues) or not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
            raise ValueError("spectrum band requires finite positive retained eigenvalues")
        normalized = eigenvalues / float(eigenvalues.mean())
        source_quantiles = np.linspace(0.0, 1.0, len(normalized))
        rows.append(np.interp(quantiles, source_quantiles, normalized))
    matrix = np.stack(rows, axis=0)
    return quantiles, np.quantile(matrix, 0.05, axis=0), np.median(matrix, axis=0), np.quantile(matrix, 0.95, axis=0)


def _plot_spectral_band(
    axis: Any, *, values: Sequence[np.ndarray], title: str, discard_relative_to_max: float | None = None
) -> None:
    quantiles, lower, median, upper = _normalized_spectrum_band(
        values, discard_relative_to_max=discard_relative_to_max
    )
    axis.fill_between(quantiles, lower, upper, alpha=0.24, label="5-95% across 96 states")
    axis.plot(quantiles, median, linewidth=1.5, label="median")
    axis.set_title(title)
    axis.set_yscale("log")
    axis.set_xlabel("eigenvalue quantile")
    axis.set_ylabel("eigenvalue / per-state mean eigenvalue")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, frameon=False, loc="best")


def _plot_spectra(arrays: Mapping[str, np.ndarray], path: Path) -> None:
    full_kron_keys, tangent_kron_keys, jacobian_keys = _spectrum_key_groups(arrays)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    _plot_spectral_band(
        axes[0],
        values=[arrays[key] for key in full_kron_keys],
        title="Full Kron K spectra",
    )
    _plot_spectral_band(
        axes[1],
        values=[arrays[key] for key in tangent_kron_keys],
        title=r"Tangent U$^T$KU spectra (rcond=1e-5)",
    )
    _plot_spectral_band(
        axes[2],
        values=[np.asarray(arrays[key], dtype=np.float64) ** 2 for key in jacobian_keys],
        title=r"VAE J$^T$J spectra (rcond=1e-5)",
        discard_relative_to_max=SPECTRUM_PLOT_RCOND**2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _artifact_signatures(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _file_signature(path) for name, path in paths.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-kron-dir", default=str(DEFAULT_KRON_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-steps", nargs="+", default=[str(value) for value in DEFAULT_CHECKPOINT_STEPS])
    parser.add_argument("--vae-runs", nargs="*", default=list(DEFAULT_VAE_RUNS))
    parser.add_argument("--max-starts", type=int, default=DEFAULT_MAX_STARTS)
    parser.add_argument("--probe-count", type=int, default=DEFAULT_PROBE_COUNT)
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    verbose = not bool(args.quiet)
    checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps)
    vae_runs = _parse_vae_runs(args.vae_runs)
    if int(args.probe_count) <= 0:
        raise SystemExit("ERROR: --probe-count must be positive")
    if int(args.jacobian_chunk_size) <= 0:
        raise SystemExit("ERROR: --jacobian-chunk-size must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"ERROR: output dir already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    kron_dir = Path(args.accepted_kron_dir).expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"ERROR: requested CUDA device={device}, but CUDA is unavailable")
    torch._dynamo.config.disable = True
    started = time.perf_counter()
    _log(
        f"resolved_config device={device} dtype=torch.float32 max_starts={args.max_starts} "
        f"checkpoint_steps={checkpoint_steps} probe_count={args.probe_count} output_dir={output_dir}",
        verbose=verbose,
    )
    _log("stage=reference_validation", verbose=verbose)
    manifest, eval_rows, diagnostics, selected = _load_selected_reference(kron_dir, max_starts=int(args.max_starts))
    reference_hashes = _validate_reference_hashes(kron_dir, manifest)
    _Kron, package_module = _kron_api()
    _log("stage=load_data", verbose=verbose)
    reference_run = Path(str(manifest["reference_run_dir"])).resolve()
    cfg = kron_harness._load_cfg(
        reference_run, device=str(device), downstream_steps=300, eval_every=25, batch_size=128
    )
    dtype = torch_dtype(cfg)
    if dtype != torch.float32:
        raise RuntimeError(f"accepted replay requires float32, config resolved {dtype}")
    task_tensors = _load_task_tensors_for_pipeline(cfg, device=device, dtype=dtype)
    weights_cpu, _records, cache_key, spec = kron_harness._load_weight_pool(reference_run, cfg)
    _log(
        f"stage=load_reference_weight_pool cache_key={cache_key} weights={tuple(weights_cpu.shape)} spec={spec.model_kind}",
        verbose=verbose,
    )
    _log("stage=exact_kron_replay", verbose=verbose)
    snapshots, replay_rows, replay_summary = _replay_selected_trajectories(
        cfg=cfg,
        spec=spec,
        task_tensors=task_tensors,
        weights_cpu=weights_cpu,
        eval_rows=eval_rows,
        expected_diagnostics=diagnostics,
        checkpoint_steps=checkpoint_steps,
        device=device,
        probe_count=int(args.probe_count),
        package_module=package_module,
        verbose=verbose,
    )
    replay_csv = output_dir / "replay_validation_rows.csv"
    _write_csv(replay_rows, replay_csv, verbose=verbose)
    replay_hard_pass = bool(
        replay_summary["replay_rows_match_reference"]
        and replay_summary["replay_q_hashes_match_reference"]
        and replay_summary["operator_gates_pass"]
        and replay_summary["snapshot_gradient_gates_pass"]
        and len(snapshots) == int(args.max_starts) * len(checkpoint_steps)
    )
    if not replay_hard_pass:
        validation = {
            "protocol": "kron_vae_operator_similarity_v1",
            "accepted": False,
            "comparison_claims_allowed": False,
            "fail_closed_stage": "exact_kron_replay",
            "refuse_result_claims_reason": "replay, frozen-operator, or snapshot-gradient gate failed before VAE geometry",
            "reference_hashes": reference_hashes,
            "checkpoint_steps": list(checkpoint_steps),
            "max_starts": int(args.max_starts),
            "expected_snapshot_count": int(args.max_starts) * len(checkpoint_steps),
            "observed_snapshot_count": int(len(snapshots)),
            "replay": replay_summary,
        }
        validation_path = output_dir / "validation.json"
        validation_path.write_text(json.dumps(_json_safe(validation), indent=2, sort_keys=True), encoding="utf-8")
        failed_paths = {"replay_validation_rows": replay_csv, "validation": validation_path}
        failed_manifest = {
            "script": _file_signature(Path(__file__).resolve()),
            "accepted_kron_dir": str(kron_dir),
            "reference_kron_manifest": _file_signature(kron_dir / "manifest.json"),
            "artifact_paths": {name: str(path) for name, path in failed_paths.items()},
            "artifact_signatures": _artifact_signatures(failed_paths),
            "validation": validation,
            "elapsed_sec": float(time.perf_counter() - started),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(_json_safe(failed_manifest), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        _log("stage=fail_closed replay validation failed before VAE geometry", verbose=verbose)
        raise SystemExit("ERROR: replay validation failed; VAE geometry was not executed")
    torch.save(
        {
            "protocol": "kron_vae_operator_similarity_v1",
            "selected_candidate": SELECTED_CANDIDATE,
            "checkpoint_steps": checkpoint_steps,
            "snapshots": snapshots,
        },
        output_dir / "replay_snapshots.pt",
    )
    _log(f"stage=load_vae_contexts count={len(vae_runs)}", verbose=verbose)
    vae_contexts = _load_vae_contexts(vae_runs, device=device, weight_dim=int(weights_cpu.shape[1]))
    _log("stage=operator_geometry", verbose=verbose)
    mismatch, action_metrics, summaries, spectra = _measure_vae_geometry(
        snapshots=snapshots,
        vae_contexts=vae_contexts,
        task_tensors=task_tensors,
        eval_rows=eval_rows,
        spec=spec,
        device=device,
        jacobian_chunk_size=int(args.jacobian_chunk_size),
        verbose=verbose,
    )
    np.savez_compressed(output_dir / "spectra.npz", **spectra)
    _plot_action_overlap(action_metrics, output_dir / "action_overlap.png")
    _plot_tangent_energy(action_metrics, output_dir / "tangent_normal_energy.png")
    _plot_spectra(spectra, output_dir / "spectra.png")
    paths = {
        "replay_snapshots": output_dir / "replay_snapshots.pt",
        "replay_validation_rows": replay_csv,
        "state_mismatch": output_dir / "state_mismatch.csv",
        "operator_action_metrics": output_dir / "operator_action_metrics.csv",
        "operator_summary": output_dir / "operator_summary.csv",
        "spectra": output_dir / "spectra.npz",
        "action_overlap_plot": output_dir / "action_overlap.png",
        "tangent_normal_energy_plot": output_dir / "tangent_normal_energy.png",
        "spectra_plot": output_dir / "spectra.png",
    }
    _write_csv(mismatch, paths["state_mismatch"], verbose=verbose)
    _write_csv(action_metrics, paths["operator_action_metrics"], verbose=verbose)
    _write_csv(summaries, paths["operator_summary"], verbose=verbose)
    all_pinned = all(bool(context["pinned"]) for context in vae_contexts)
    claims_allowed = bool(
        replay_summary["replay_rows_match_reference"]
        and replay_summary["replay_q_hashes_match_reference"]
        and replay_summary["operator_gates_pass"]
        and replay_summary["snapshot_gradient_gates_pass"]
    )
    validation = {
        "protocol": "kron_vae_operator_similarity_v1",
        "accepted": bool(claims_allowed),
        "comparison_claims_allowed": bool(claims_allowed),
        "refuse_result_claims_reason": (
            "all selected replay rows, Q hashes, and frozen-operator gates passed"
            if claims_allowed
            else "one or more replay Q hashes, replay diagnostics, or frozen-operator gates failed"
        ),
        "comparison_scope": (
            "descriptive operator similarity conditional on each recorded theta_t -> D(E(theta_t)) mismatch; "
            "not a same-state comparison, downstream benchmark, or causal attribution"
        ),
        "comparison_conditional_on_recorded_reconstruction_mismatch": True,
        "all_default_vae_checkpoints_pinned": bool(all_pinned),
        "reference_hashes": reference_hashes,
        "selected_candidate": SELECTED_CANDIDATE,
        "selected_lr": SELECTED_LR,
        "selected_schedule": SELECTED_SCHEDULE,
        "checkpoint_steps": list(checkpoint_steps),
        "max_starts": int(args.max_starts),
        "expected_snapshot_count": int(args.max_starts) * len(checkpoint_steps),
        "observed_snapshot_count": int(len(snapshots)),
        "replay": replay_summary,
        "row_counts": {
            "state_mismatch": int(len(mismatch)),
            "operator_action_metrics": int(len(action_metrics)),
            "operator_summary": int(len(summaries)),
        },
    }
    validation_path = output_dir / "validation.json"
    validation_path.write_text(json.dumps(_json_safe(validation), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    paths["validation"] = validation_path
    manifest_out = {
        "script": _file_signature(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "accepted_kron_dir": str(kron_dir),
        "reference_kron_manifest": _file_signature(kron_dir / "manifest.json"),
        "reference_harness_script": _file_signature(Path(kron_harness.__file__).resolve()),
        "reference_weight_pool": _file_signature(reference_run / "weight_pool.pt"),
        "reference_config": _file_signature(reference_run / "config.json"),
        "vae_checkpoints": {
            str(context["label"]): _file_signature(Path(context["run_dir"]) / "vae_checkpoint.pt")
            for context in vae_contexts
        },
        "package": {
            "distribution": "kron-torch",
            "version": importlib.metadata.version("kron-torch"),
            "implementation": _file_signature(Path(inspect.getsourcefile(package_module.Kron) or "")),
        },
        "resolved_config": {
            "device": str(device),
            "dtype": str(dtype),
            "max_starts": int(args.max_starts),
            "checkpoint_steps": list(checkpoint_steps),
            "probe_count": int(args.probe_count),
            "jacobian_chunk_size": int(args.jacobian_chunk_size),
            "vae_runs": [str(context["run_dir"]) for context in vae_contexts],
            "comparison_condition": "VAE P_T and JJ^T are evaluated at theta_bar=D(E(theta_t))",
        },
        "artifact_paths": {"manifest": str(output_dir / "manifest.json"), **{name: str(path) for name, path in paths.items()}},
        "artifact_signatures": _artifact_signatures(paths),
        "validation": validation,
        "elapsed_sec": float(time.perf_counter() - started),
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest_out), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    _log(
        f"done accepted={claims_allowed} replay_q_mismatches={replay_summary['replay_q_hash_mismatch_count']} "
        f"output_dir={output_dir} elapsed_sec={time.perf_counter() - started:.2f}",
        verbose=verbose,
    )
    if not claims_allowed:
        raise SystemExit("ERROR: replay validation failed; result claims are refused. Inspect validation.json")


if __name__ == "__main__":
    main()
