#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    decode_weights,
    decoder_jacobians,
    encode_weights,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.progress import make_progress
from scripts.audit_latent_raw_common_state_rate import (
    AdamState,
    TRUST_SCAN_SCALES,
    _adam_delta,
    _build_projector,
    _cosine,
    _norm,
    _sha256_file,
    _stable_json_hash,
)
from scripts.audit_variant_a_trajectory_realization import (
    _batch_indices,
    _load_cfg,
    _load_vae,
    _load_weight_pool,
    _loss_acc,
    _selected_lr,
    _task_tensor_set,
    _tensor_sha256,
)
from scripts.evaluate_pullback_deployment_intervention import (
    _eval,
    _executed_preimage_residuals,
    _first_upward_crossings,
    _latent_grad,
    _raw_reference,
    _task_tensors_fingerprint,
    _theta_grad,
)


PROTOCOL_VERSION = "global_latent_gauge_v2"
PRODUCTION_STEPS = 25
METRIC_STATE_STEPS = (0, 1, 5, 25)
OCCUPANCY_STEPS = (0, 1, 5, 25)
REAL_SMOKE_STEPS = 2
REAL_SMOKE_METRIC_STATE_STEPS = (0, 1, 2)
REAL_SMOKE_OCCUPANCY_STEPS = (0, 1, 2)
RCONDS = (1.0e-6, 1.0e-5, 1.0e-4)
PRIMARY_RCOND = 1.0e-5
PROJECTOR_RCOND = 1.0e-5
DECODED_START_ATOL = 2.0e-5
DECODED_START_RTOL = 2.0e-6
PRODUCTION_LABELS = ("control_seed0", "control_seed1", "control_seed2")
PRODUCTION_START_BANK_SHA256 = "b59f2c50fd389dd78097b30f1c8240c123ba9c66c7fea73c66ee99f4a4cace29"
PRODUCTION_EVAL_ROWS_SHA256 = "79afed5cb9e7d904f015ed11d8604d842d72ec27f1be4c60af8369acd86bd4c9"
PRODUCTION_FIT_ROWS_SHA256 = "f088b5ac53a20db592c505c6529d8c20d771af1a2f523b602a43edfc58064264"
PRODUCTION_EVAL_SOURCES = (
    15374, 10003, 9470, 1434, 6026, 12176, 942, 8978,
    4632, 14718, 10946, 13160, 122, 14431, 3402, 10659,
)
PRODUCTION_FIT_SOURCES = (
    12094, 2295, 9757, 2869, 6682, 2541, 1721, 14882,
    6805, 8035, 14062, 14964, 14185, 10331, 4673, 40,
)
PRODUCTION_CHECKPOINT_SHA256 = (
    "3fbb53ef5cb290b82c4e2a51788f54951544e59e8c78b1d50aff6edfaa0978bb",
    "10e5e0ffbc83b4eeb0993717490f0a469b93d4f4c89d1b48c89641f819b0a0f5",
    "400545a5bf876c36a439b55d0ba63e1a4d641666822933d0b545e51760dfb5ae",
)
PROTOCOL_PATH = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "common_state_rate_20260710/global_latent_gauge_protocol.md"
)
DEFAULT_START_BANK = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "extended_downstream/c0p001_unclipped_fixedlr_64eval/downstream_start_bank.csv"
)
DEFAULT_RUNS = (
    "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_"
    "marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
    "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_"
    "marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_methodrep_seed1",
    "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_"
    "marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_methodrep_seed2",
)
ARTIFACT_ROOT = Path("artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing")


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[global_latent_gauge] {message}", flush=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return _sha256_bytes(frame.reset_index(drop=True).to_csv(index=False).encode("utf-8"))


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _request_hash(payload: Mapping[str, Any]) -> str:
    if "elapsed_sec" in payload:
        raise ValueError("deterministic request payload must exclude elapsed_sec")
    return _stable_json_hash(dict(payload))


def _rcond_label(value: float | None) -> str:
    if value is None:
        return "none"
    return {1.0e-6: "1e-6", 1.0e-5: "1e-5", 1.0e-4: "1e-4"}[float(value)]


@dataclass(frozen=True)
class Arm:
    arm_id: str
    optimizer: str
    gauge: str
    rcond: float | None
    family: str

    @property
    def rcond_label(self) -> str:
        return _rcond_label(self.rcond)


@dataclass(frozen=True)
class Gauge:
    name: str
    rcond: float | None
    matrix: torch.Tensor
    inverse: torch.Tensor
    normalized_metric: torch.Tensor
    raw_eigenvalues: torch.Tensor
    floored_eigenvalues: torch.Tensor
    clipped_count: int
    clipped_mass: float

    def to_gauge(self, z: torch.Tensor) -> torch.Tensor:
        return self.matrix @ z

    def from_gauge(self, y: torch.Tensor) -> torch.Tensor:
        return self.inverse @ y

    def gradient_to_gauge(self, grad_z: torch.Tensor) -> torch.Tensor:
        return self.inverse.T @ grad_z

    def metric_to_gauge(self, metric_z: torch.Tensor) -> torch.Tensor:
        return self.inverse.T @ metric_z @ self.inverse


@dataclass(frozen=True)
class RadiusMatch:
    scale: float
    value: torch.Tensor
    radius: float
    relative_error: float
    bracket_found: bool
    crossing_count: int
    boundary_hit: bool
    termination_reason: str


@dataclass
class ProductionVAEContext:
    label: str
    run_dir: Path
    cfg: Any
    vae: torch.nn.Module
    normalizer: Any
    checkpoint_payload: Mapping[str, Any]
    raw_lr: float
    latent_lr: float
    checkpoint_sha256: str
    config_sha256: str
    selected_lrs_sha256: str
    normalizer_sha256: str


@dataclass
class RuntimeCounters:
    decoder_forward_calls: int = 0
    jacobian_calls: int = 0
    jacobian_latent_directions: int = 0
    full_split_eval_calls: int = 0
    batch_gradient_calls: int = 0
    trust_candidate_calls: int = 0


CONTRAST_SPECS = (
    ("primary_full_1e-5_minus_identity_sgd", "z_sgd:full:1e-5", "z_sgd:identity:none", "1e-5", "global_gauge_primary"),
    ("orthogonal_minus_identity_sgd", "z_sgd:orthogonal:none", "z_sgd:identity:none", "none", "algebraic_negative_control"),
    ("full_minus_diagonal_sgd_1e-6", "z_sgd:full:1e-6", "z_sgd:diagonal:1e-6", "1e-6", "off_diagonal_structure"),
    ("full_minus_diagonal_sgd_1e-5", "z_sgd:full:1e-5", "z_sgd:diagonal:1e-5", "1e-5", "off_diagonal_structure"),
    ("full_minus_diagonal_sgd_1e-4", "z_sgd:full:1e-4", "z_sgd:diagonal:1e-4", "1e-4", "off_diagonal_structure"),
    ("pullback_minus_full_sgd_1e-6", "pullback", "z_sgd:full:1e-6", "1e-6", "local_vs_global_metric"),
    ("pullback_minus_full_sgd_1e-5", "pullback", "z_sgd:full:1e-5", "1e-5", "local_vs_global_metric"),
    ("pullback_minus_full_sgd_1e-4", "pullback", "z_sgd:full:1e-4", "1e-4", "local_vs_global_metric"),
    ("task_normal_minus_pullback", "augmented_task_normal", "pullback", "1e-5", "same_total_radius_reallocation"),
    ("task_normal_minus_placebo", "augmented_task_normal", "augmented_placebo", "1e-5", "clean_paired_normal_access"),
)


def _arm_grid() -> tuple[Arm, ...]:
    arms = [Arm("raw_adam", "raw_adam", "identity", None, "raw")]
    for optimizer in ("z_sgd", "z_adam"):
        for gauge in ("identity", "orthogonal"):
            arms.append(Arm(f"{optimizer}:{gauge}:none", optimizer, gauge, None, "pure_gauge"))
        for gauge in ("diagonal", "full"):
            for rcond in RCONDS:
                label = _rcond_label(rcond)
                arms.append(Arm(f"{optimizer}:{gauge}:{label}", optimizer, gauge, rcond, "pure_gauge"))
    arms.extend(
        [
            Arm("pullback", "pullback", "local", PROJECTOR_RCOND, "local_access"),
            Arm("augmented_task_normal", "augmented", "task_normal", PROJECTOR_RCOND, "normal_access"),
            Arm("augmented_placebo", "augmented", "placebo", PROJECTOR_RCOND, "normal_access"),
        ]
    )
    if len(arms) != 20 or len({arm.arm_id for arm in arms}) != len(arms):
        raise AssertionError("protocol arm grid must contain exactly 20 unique arms")
    return tuple(arms)


def _partition_protocol_banks(bank: pd.DataFrame, *, require_exact: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(bank) < 32:
        raise ValueError(f"start bank has {len(bank)} rows; protocol requires at least 32")
    if "source_weight_index" not in bank.columns or "start_bank_position" not in bank.columns:
        raise ValueError("start bank is missing source_weight_index/start_bank_position")
    evaluation = bank.iloc[0:16].copy().reset_index(drop=True)
    metric_fit = bank.iloc[16:32].copy().reset_index(drop=True)
    eval_sources = tuple(evaluation["source_weight_index"].astype(int))
    fit_sources = tuple(metric_fit["source_weight_index"].astype(int))
    overlap = sorted(set(eval_sources) & set(fit_sources))
    if overlap:
        raise ValueError(f"metric-fit/evaluation source leakage: {overlap}")
    if require_exact:
        violations = []
        if eval_sources != PRODUCTION_EVAL_SOURCES:
            violations.append(f"evaluation sources={eval_sources}")
        if fit_sources != PRODUCTION_FIT_SOURCES:
            violations.append(f"metric-fit sources={fit_sources}")
        if tuple(evaluation["start_bank_position"].astype(int)) != tuple(range(16)):
            violations.append("evaluation positions are not rows 0:15")
        if tuple(metric_fit["start_bank_position"].astype(int)) != tuple(range(16, 32)):
            violations.append("metric-fit positions are not rows 16:31")
        if _frame_sha256(evaluation) != PRODUCTION_EVAL_ROWS_SHA256:
            violations.append(f"evaluation row hash={_frame_sha256(evaluation)}")
        if _frame_sha256(metric_fit) != PRODUCTION_FIT_ROWS_SHA256:
            violations.append(f"metric-fit row hash={_frame_sha256(metric_fit)}")
        if violations:
            raise ValueError("exact bank protocol mismatch: " + "; ".join(violations))
    evaluation["protocol_split"] = "evaluation"
    metric_fit["protocol_split"] = "metric_fit"
    return evaluation, metric_fit


def _load_protocol_banks(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    resolved = path.expanduser().resolve()
    file_hash = _sha256_file(resolved)
    if file_hash != PRODUCTION_START_BANK_SHA256:
        raise ValueError(f"start-bank SHA256 mismatch: got={file_hash} expected={PRODUCTION_START_BANK_SHA256}")
    raw = pd.read_csv(resolved)
    evaluation, metric_fit = _partition_protocol_banks(raw, require_exact=True)
    hashes = {
        "file_sha256": file_hash,
        "evaluation_rows_sha256": _frame_sha256(evaluation.drop(columns="protocol_split")),
        "metric_fit_rows_sha256": _frame_sha256(metric_fit.drop(columns="protocol_split")),
    }
    return evaluation, metric_fit, hashes


def _normalized_metric(metric: torch.Tensor) -> torch.Tensor:
    value = 0.5 * (metric.detach().double() + metric.detach().double().T)
    scale = torch.trace(value) / int(value.shape[0])
    if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        raise ValueError(f"metric trace normalization is invalid: {float(scale)}")
    return value / scale


def _whitening_gauge(metric: torch.Tensor, *, kind: str, rcond: float) -> Gauge:
    if float(rcond) not in RCONDS:
        raise ValueError(f"rcond must be one of {RCONDS}, got={rcond}")
    normalized = _normalized_metric(metric)
    if kind == "full":
        eigenvalues, eigenvectors = torch.linalg.eigh(normalized)
    elif kind == "diagonal":
        eigenvalues = torch.diagonal(normalized).clone()
        eigenvectors = torch.eye(normalized.shape[0], dtype=torch.float64, device=normalized.device)
    else:
        raise ValueError(f"whitening kind must be full or diagonal, got={kind!r}")
    largest = float(eigenvalues.max().detach().cpu())
    if not math.isfinite(largest) or largest <= 0.0:
        raise ValueError(f"metric has no positive finite maximum eigenvalue: {largest}")
    floor = float(rcond) * largest
    floored = torch.clamp(eigenvalues, min=floor)
    clipped = eigenvalues < floor
    clipped_mass = float((floored[clipped] - eigenvalues[clipped]).sum().detach().cpu())
    matrix = (eigenvectors * floored.sqrt().unsqueeze(0)) @ eigenvectors.T
    inverse = (eigenvectors * floored.rsqrt().unsqueeze(0)) @ eigenvectors.T
    return Gauge(
        name=kind,
        rcond=float(rcond),
        matrix=matrix,
        inverse=inverse,
        normalized_metric=normalized,
        raw_eigenvalues=eigenvalues,
        floored_eigenvalues=floored,
        clipped_count=int(clipped.sum().detach().cpu()),
        clipped_mass=clipped_mass,
    )


def _identity_gauge(metric: torch.Tensor) -> Gauge:
    normalized = _normalized_metric(metric)
    identity = torch.eye(metric.shape[0], dtype=torch.float64, device=metric.device)
    eigenvalues = torch.ones(metric.shape[0], dtype=torch.float64, device=metric.device)
    return Gauge("identity", None, identity, identity, normalized, eigenvalues, eigenvalues, 0, 0.0)


def _orthogonal_gauge(metric: torch.Tensor, *, seed: int) -> Gauge:
    normalized = _normalized_metric(metric)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sample = torch.randn(metric.shape[0], metric.shape[0], generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(sample)
    signs = torch.where(torch.diagonal(r) < 0.0, -1.0, 1.0)
    q = q * signs.unsqueeze(0)
    q = q.to(metric.device)
    eigenvalues = torch.ones(metric.shape[0], dtype=torch.float64, device=metric.device)
    return Gauge("orthogonal", None, q, q.T, normalized, eigenvalues, eigenvalues, 0, 0.0)


def _gauges_for_metric(metric: torch.Tensor, *, seed: int) -> dict[tuple[str, str], Gauge]:
    gauges = {
        ("identity", "none"): _identity_gauge(metric),
        ("orthogonal", "none"): _orthogonal_gauge(metric, seed=seed),
    }
    for kind in ("diagonal", "full"):
        for rcond in RCONDS:
            gauges[(kind, _rcond_label(rcond))] = _whitening_gauge(metric, kind=kind, rcond=rcond)
    return gauges


def _celo_literal_start(
    vae: torch.nn.Module,
    normalizer: Any,
    weights: torch.Tensor,
    *,
    counters: RuntimeCounters | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        z0 = encode_weights(vae, normalizer, weights.reshape(1, -1)).squeeze(0).float()
        theta0 = decode_weights(vae, normalizer, z0.reshape(1, -1)).squeeze(0).float()
    if counters is not None:
        counters.decoder_forward_calls += 1
    return z0.detach(), theta0.detach()


def _celo_decode_gauged(
    vae: torch.nn.Module,
    normalizer: Any,
    y: torch.Tensor,
    gauge: Gauge,
    *,
    counters: RuntimeCounters | None = None,
    trust_candidate: bool = False,
) -> torch.Tensor:
    z = gauge.from_gauge(y.double()).to(device=y.device, dtype=torch.float32)
    value = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0).float()
    if counters is not None:
        counters.decoder_forward_calls += 1
        counters.trust_candidate_calls += int(trust_candidate)
    return value


def _celo_explicit_fp32_jacobian(
    vae: torch.nn.Module,
    normalizer: Any,
    z: torch.Tensor,
    *,
    chunk_size: int,
    counters: RuntimeCounters | None = None,
) -> torch.Tensor:
    z32 = z.detach().to(dtype=torch.float32)
    jacobian = decoder_jacobians(
        vae,
        normalizer,
        None,
        z32.reshape(1, -1),
        create_graph=False,
        chunk_size=int(chunk_size),
    ).squeeze(0)
    if jacobian.dtype != torch.float32:
        raise RuntimeError(f"protocol requires an explicit fp32 decoder Jacobian, got={jacobian.dtype}")
    if counters is not None:
        counters.jacobian_calls += 1
        counters.jacobian_latent_directions += int(z32.numel())
    return jacobian.detach()


def _tensor_mapping_sha256(values: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for key, value in sorted(values.items()):
        payload[str(key)] = _tensor_sha256(value) if isinstance(value, torch.Tensor) else value
    return _stable_json_hash(payload)


def _task_hashes(task_tensors: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, task in sorted(task_tensors.items()):
        result[str(name)] = {
            "train_images_sha256": _tensor_sha256(task.train_images),
            "train_labels_sha256": _tensor_sha256(task.train_labels),
            "test_images_sha256": _tensor_sha256(task.test_images),
            "test_labels_sha256": _tensor_sha256(task.test_labels),
        }
    return result


def _gauge_loss_grad(
    y: torch.Tensor,
    *,
    vae: torch.nn.Module,
    normalizer: Any,
    gauge: Gauge,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    counters: RuntimeCounters | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    leaf = y.detach().clone().requires_grad_(True)
    theta = _celo_decode_gauged(vae, normalizer, leaf, gauge, counters=counters)
    loss = loss_fn(theta)
    grad = torch.autograd.grad(loss, leaf)[0].detach()
    return grad, theta.detach(), float(loss.detach().cpu())


def _trajectory_execution_flags(diagnostics: pd.DataFrame) -> pd.DataFrame:
    keys = ["vae_seed", "source_weight_index", "arm_id"]
    required = set(keys) | {"trust_executable", "operator_executable", "retraction_executable", "treatment_executable"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise ValueError(f"execution diagnostics missing columns: {sorted(missing)}")
    return diagnostics.groupby(keys, as_index=False).agg(
        trajectory_trust_executable=("trust_executable", "all"),
        trajectory_operator_executable=("operator_executable", "all"),
        trajectory_retraction_executable=("retraction_executable", "all"),
        trajectory_treatment_executable=("treatment_executable", "all"),
        trajectory_trust_failure=("trust_executable", lambda values: bool((~values.astype(bool)).any())),
        trajectory_singularity_stratum=("singularity_state_local", "any"),
    )


def _propagate_execution_flags(
    curves: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flags = _trajectory_execution_flags(diagnostics)
    keys = ["vae_seed", "source_weight_index", "arm_id"]
    curves_out = curves.merge(flags, on=keys, how="left", validate="many_to_one")
    diagnostics_out = diagnostics.merge(flags, on=keys, how="left", validate="many_to_one")
    if curves_out["trajectory_treatment_executable"].isna().any():
        raise RuntimeError("failed to propagate treatment execution status to every curve")
    return curves_out, diagnostics_out, flags


def _validate_same_batches(
    diagnostics: pd.DataFrame,
    *,
    expected_arm_ids: Sequence[str],
) -> dict[str, Any]:
    keys = ["vae_seed", "source_weight_index", "proposal_step"]
    required = set(keys) | {"arm_id", "batch_indices_sha256"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise ValueError(f"same-batch validation missing columns: {sorted(missing)}")
    counts = diagnostics.groupby(keys).agg(
        arm_count=("arm_id", "nunique"),
        batch_hash_count=("batch_indices_sha256", "nunique"),
    )
    bad_arms = counts[counts["arm_count"] != len(expected_arm_ids)]
    bad_hashes = counts[counts["batch_hash_count"] != 1]
    if not bad_arms.empty or not bad_hashes.empty:
        raise ValueError(
            "same-batch protocol mismatch: "
            f"arm_grid_cells={len(bad_arms)} batch_hash_cells={len(bad_hashes)}"
        )
    return {"accepted": True, "cells": int(len(counts)), "max_batch_hash_count": 1}


def _validate_coupled_normal_tangents(diagnostics: pd.DataFrame) -> dict[str, Any]:
    arm_ids = ("augmented_task_normal", "augmented_placebo")
    subset = diagnostics[diagnostics["arm_id"].isin(arm_ids)].copy()
    keys = ["vae_seed", "source_weight_index", "proposal_step"]
    required = set(keys) | {
        "arm_id",
        "state_z_sha256",
        "paired_parent_theta_sha256",
        "tangent_proposal_sha256",
        "executed_latent_delta_sha256",
        "trust_scale",
        "full_realized_radius_error_task",
        "full_realized_radius_error_placebo",
        "pair_trust_executable",
        "pair_operator_executable",
        "pair_retraction_executable",
        "treatment_executable",
        "trust_bracket_found",
        "trust_scale_boundary_hit",
        "task_normal_norm",
        "placebo_normal_norm",
    }
    missing = required - set(subset.columns)
    if missing:
        raise ValueError(f"normal-pair validation missing columns: {sorted(missing)}")
    grouped = subset.groupby(keys).agg(
        arms=("arm_id", "nunique"),
        states_z=("state_z_sha256", "nunique"),
        parents=("paired_parent_theta_sha256", "nunique"),
        tangents=("tangent_proposal_sha256", "nunique"),
        latent_deltas=("executed_latent_delta_sha256", "nunique"),
        trust_scales=("trust_scale", "nunique"),
        task_errors=("full_realized_radius_error_task", "nunique"),
        placebo_errors=("full_realized_radius_error_placebo", "nunique"),
        pair_trust=("pair_trust_executable", "nunique"),
        treatments=("treatment_executable", "nunique"),
    )
    bad = grouped[
        (grouped["arms"] != 2)
        | (grouped["states_z"] != 1)
        | (grouped["parents"] != 1)
        | (grouped["tangents"] != 1)
        | (grouped["latent_deltas"] != 1)
        | (grouped["trust_scales"] != 1)
        | (grouped["task_errors"] != 1)
        | (grouped["placebo_errors"] != 1)
        | (grouped["pair_trust"] != 1)
        | (grouped["treatments"] != 1)
    ]
    if not bad.empty:
        raise ValueError(f"normal pair does not hold tangent state fixed in {len(bad)} cells")
    inconsistent_gates = 0
    unequal_norms = 0
    for _keys, rows in subset.groupby(keys, sort=False):
        first = rows.iloc[0]
        expected_trust = _pair_trust_gate(
            task_error=float(first["full_realized_radius_error_task"]),
            placebo_error=float(first["full_realized_radius_error_placebo"]),
            bracket_found=bool(first["trust_bracket_found"]),
            boundary_hit=bool(first["trust_scale_boundary_hit"]),
        )
        expected_treatment = bool(
            expected_trust
            and bool(first["pair_operator_executable"])
            and bool(first["pair_retraction_executable"])
        )
        if bool(first["pair_trust_executable"]) != expected_trust or bool(first["treatment_executable"]) != expected_treatment:
            inconsistent_gates += 1
        task_norm = float(first["task_normal_norm"])
        placebo_norm = float(first["placebo_normal_norm"])
        if abs(task_norm - placebo_norm) / max(task_norm, 1.0e-30) > 1.0e-6:
            unequal_norms += 1
    if inconsistent_gates or unequal_norms:
        raise ValueError(
            "normal pair full-radius trust validation failed: "
            f"inconsistent_gates={inconsistent_gates} unequal_norms={unequal_norms}"
        )
    pair_status = subset.groupby(keys)["pair_trust_executable"].first().astype(bool)
    return {
        "accepted": True,
        "cells": int(len(grouped)),
        "task_radius_error_max": float(subset["full_realized_radius_error_task"].max()),
        "placebo_radius_error_max": float(subset["full_realized_radius_error_placebo"].max()),
        "pair_trust_failure_cells": int((~pair_status).sum()),
    }


def _verify_artifact_hashes(root: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    missing = []
    mismatched = []
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif _sha256_file(path) != str(digest):
            mismatched.append(name)
    if missing or mismatched:
        raise ValueError(f"artifact hash verification failed: missing={missing} mismatched={mismatched}")
    return {"accepted": True, "files": len(expected)}


def _write_hash_index(
    root: Path,
    *,
    file_hashes: Mapping[str, str],
    index_name: str = "global_latent_gauge_output_hashes.json",
) -> Path:
    _verify_artifact_hashes(root, file_hashes)
    path = root / index_name
    path.write_text(
        json.dumps(
            {
                "hash_index_self_excluded": True,
                "files": dict(file_hashes),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    values = {
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    if device.type == "cuda":
        values.update(
            {
                "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return values


def _validate_real_smoke_selection(
    *,
    run_values: Sequence[str],
    labels: Sequence[str],
    fit_source: int,
    eval_source: int,
) -> dict[str, Any]:
    if len(run_values) != 1 or len(labels) != 1:
        raise ValueError("real_smoke requires exactly one --run and one --label")
    if int(fit_source) not in PRODUCTION_FIT_SOURCES:
        raise ValueError(f"real_smoke fit source must come from pinned rows16:31, got={fit_source}")
    if int(eval_source) not in PRODUCTION_EVAL_SOURCES:
        raise ValueError(f"real_smoke eval source must come from pinned rows0:15, got={eval_source}")
    if int(fit_source) == int(eval_source):
        raise ValueError("real_smoke fit/eval sources must be disjoint")
    return {
        "accepted": True,
        "run": str(run_values[0]),
        "label": str(labels[0]),
        "fit_source": int(fit_source),
        "eval_source": int(eval_source),
        "production_grid": False,
    }


def _fit_global_metric(
    records: pd.DataFrame,
    *,
    expected_fit_sources: Sequence[int],
    evaluation_sources: Sequence[int],
) -> tuple[torch.Tensor, pd.DataFrame]:
    required = {"source_weight_index", "state_sha256", "gram"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"metric records missing columns: {sorted(missing)}")
    observed_sources = tuple(dict.fromkeys(records["source_weight_index"].astype(int).tolist()))
    if observed_sources != tuple(int(value) for value in expected_fit_sources):
        raise ValueError(f"metric-fit source order mismatch: got={observed_sources}")
    leaked = sorted(set(observed_sources) & {int(value) for value in evaluation_sources})
    if leaked:
        raise ValueError(f"evaluation sources leaked into metric fitting: {leaked}")
    start_means = []
    kept_rows: list[dict[str, Any]] = []
    for source in expected_fit_sources:
        source_rows = records[records["source_weight_index"].astype(int) == int(source)]
        unique_grams = []
        for state_hash, duplicates in source_rows.groupby("state_sha256", sort=False):
            tensors = [torch.as_tensor(value, dtype=torch.float64) for value in duplicates["gram"]]
            if any(not torch.equal(tensors[0], other) for other in tensors[1:]):
                raise ValueError(f"duplicate state hash has unequal JtJ tensors: source={source} hash={state_hash}")
            unique_grams.append(tensors[0])
            row = duplicates.iloc[0].to_dict()
            row["duplicate_count"] = int(len(duplicates))
            kept_rows.append(row)
        if not unique_grams:
            raise ValueError(f"metric-fit source has no states: {source}")
        start_means.append(torch.stack(unique_grams).mean(dim=0))
    metric = torch.stack(start_means).mean(dim=0)
    if metric.dtype != torch.float64 or not bool(torch.isfinite(metric).all()):
        raise RuntimeError("global metric accumulation must produce finite fp64")
    return metric, pd.DataFrame(kept_rows)


def _real_metric_fit(
    *,
    context: ProductionVAEContext,
    weights_cpu: torch.Tensor,
    fit_bank: pd.DataFrame,
    eval_bank: pd.DataFrame,
    task_tensors: Mapping[str, Any],
    task_hashes: Mapping[str, Mapping[str, str]],
    spec: Any,
    device: torch.device,
    batch_size: int,
    jacobian_chunk_size: int,
    progress_enabled: bool,
    steps: int = PRODUCTION_STEPS,
    metric_state_steps: Sequence[int] = METRIC_STATE_STEPS,
    expected_fit_sources: Sequence[int] = PRODUCTION_FIT_SOURCES,
    evaluation_sources: Sequence[int] = PRODUCTION_EVAL_SOURCES,
    counters: RuntimeCounters | None = None,
) -> tuple[torch.Tensor, dict[tuple[str, str], Gauge], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    state_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    checkpoint_steps = {int(value) for value in metric_state_steps}
    progress_cfg = context.cfg
    if not progress_enabled:
        progress_cfg.show_progress = False
    progress = make_progress(progress_cfg, total=len(fit_bank), desc=f"gauge metric fit {context.label}")
    try:
        for row in fit_bank.to_dict(orient="records"):
            source = int(row["source_weight_index"])
            stream_index = int(row.get("start_index", row["start_bank_position"]))
            task_name = str(row["task_name"])
            tau = float(row["tau"])
            task_set = _task_tensor_set(dict(task_tensors), task_name)
            w0 = weights_cpu[source].to(device=device, dtype=torch.float32)
            z, theta0 = _celo_literal_start(
                context.vae,
                context.normalizer,
                w0,
                counters=counters,
            )
            adam_state = AdamState.zeros_like(z)
            for step in range(int(steps) + 1):
                state_hash = _tensor_sha256(z)
                theta = decode_weights(context.vae, context.normalizer, z.reshape(1, -1)).squeeze(0).detach().float()
                if counters is not None:
                    counters.decoder_forward_calls += 1
                incoming_batch = None if step == 0 else _batch_indices(
                    task_set,
                    batch_size=int(batch_size),
                    step=step - 1,
                    start_bank_position=stream_index,
                )
                next_batch = None if step == int(steps) else _batch_indices(
                    task_set,
                    batch_size=int(batch_size),
                    step=step,
                    start_bank_position=stream_index,
                )
                if step in checkpoint_steps:
                    jacobian32 = _celo_explicit_fp32_jacobian(
                        context.vae,
                        context.normalizer,
                        z,
                        chunk_size=int(jacobian_chunk_size),
                        counters=counters,
                    )
                    gram64 = jacobian32.double().T @ jacobian32.double()
                    gram_key = f"gram_source{source}_step{step}"
                    arrays[gram_key] = gram64.detach().cpu().numpy()
                    record = {
                        "vae_seed": context.label,
                        "source_weight_index": source,
                        "start_bank_position": int(row["start_bank_position"]),
                        "stream_start_index": stream_index,
                        "task_name": task_name,
                        "tau": tau,
                        "metric_state_step": int(step),
                        "state_sha256": state_hash,
                        "theta_sha256": _tensor_sha256(theta),
                        "theta0_sha256": _tensor_sha256(theta0),
                        "incoming_batch_sha256": _tensor_sha256(incoming_batch),
                        "next_batch_sha256": "terminal" if next_batch is None and step == int(steps) else _tensor_sha256(next_batch),
                        "jacobian_dtype": str(jacobian32.dtype),
                        "jacobian_sha256": _tensor_sha256(jacobian32),
                        "gram_dtype": str(gram64.dtype),
                        "gram_sha256": _tensor_sha256(gram64),
                        "gram_artifact_key": gram_key,
                        "task_train_images_sha256": task_hashes[task_name]["train_images_sha256"],
                        "task_train_labels_sha256": task_hashes[task_name]["train_labels_sha256"],
                        "task_test_images_sha256": task_hashes[task_name]["test_images_sha256"],
                        "task_test_labels_sha256": task_hashes[task_name]["test_labels_sha256"],
                    }
                    state_records.append(record)
                    fit_records.append({**record, "gram": gram64})
                if step == int(steps):
                    break
                batch_indices = next_batch
                grad_z, theta_check, batch_loss = _latent_grad(
                    z,
                    vae=context.vae,
                    normalizer=context.normalizer,
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    batch_indices=batch_indices,
                )
                if counters is not None:
                    counters.decoder_forward_calls += 1
                    counters.batch_gradient_calls += 1
                if _tensor_sha256(theta_check) != _tensor_sha256(theta):
                    raise RuntimeError(f"metric-fit decode mismatch for {context.label} source={source} step={step}")
                trajectory_rows.append(
                    {
                        "vae_seed": context.label,
                        "source_weight_index": source,
                        "start_bank_position": int(row["start_bank_position"]),
                        "stream_start_index": stream_index,
                        "task_name": task_name,
                        "tau": tau,
                        "optimizer_step": int(step),
                        "state_sha256": state_hash,
                        "theta_sha256": _tensor_sha256(theta),
                        "batch_indices_sha256": _tensor_sha256(batch_indices),
                        "batch_loss": batch_loss,
                        "gradient_sha256": _tensor_sha256(grad_z),
                        "adam_state_step_before": int(adam_state.step),
                    }
                )
                z = (z + _adam_delta(adam_state, grad_z, lr=float(context.latent_lr))).detach()
            progress.set_postfix({"source": source, "states": len(state_records)})
            progress.update(1)
    finally:
        progress.close()
    raw_states = pd.DataFrame(state_records)
    metric, deduplicated = _fit_global_metric(
        pd.DataFrame(fit_records),
        expected_fit_sources=tuple(int(value) for value in expected_fit_sources),
        evaluation_sources=tuple(int(value) for value in evaluation_sources),
    )
    duplicate_counts = deduplicated[["source_weight_index", "state_sha256", "duplicate_count"]]
    raw_states = raw_states.merge(
        duplicate_counts,
        on=["source_weight_index", "state_sha256"],
        how="left",
        validate="many_to_one",
    )
    gauges = _gauges_for_metric(metric, seed=_stable_seed("production_orthogonal", context.label))
    arrays["global_metric_fp64"] = metric.detach().cpu().numpy()
    for (gauge_name, rcond_label), gauge in gauges.items():
        key = f"gauge_{gauge_name}_{rcond_label}".replace("-", "m")
        arrays[f"{key}_A"] = gauge.matrix.detach().cpu().numpy()
        arrays[f"{key}_A_inv"] = gauge.inverse.detach().cpu().numpy()
        arrays[f"{key}_raw_eigenvalues"] = gauge.raw_eigenvalues.detach().cpu().numpy()
        arrays[f"{key}_floored_eigenvalues"] = gauge.floored_eigenvalues.detach().cpu().numpy()
    return metric, gauges, raw_states, pd.DataFrame(trajectory_rows), arrays


def _matrix_metadata(
    *,
    vae_seed: str,
    metric: torch.Tensor,
    gauges: Mapping[tuple[str, str], Gauge],
    matrix_file: Path,
) -> pd.DataFrame:
    rows = []
    identity = torch.eye(metric.shape[0], dtype=torch.float64, device=metric.device)
    for (gauge_name, rcond_label), gauge in gauges.items():
        inverse_error = _norm(gauge.inverse @ gauge.matrix - identity)
        rows.append(
            {
                "vae_seed": vae_seed,
                "gauge": gauge_name,
                "rcond_label": rcond_label,
                "rcond": gauge.rcond,
                "global_metric_sha256": _tensor_sha256(metric),
                "normalized_metric_sha256": _tensor_sha256(gauge.normalized_metric),
                "matrix_sha256": _tensor_sha256(gauge.matrix),
                "inverse_sha256": _tensor_sha256(gauge.inverse),
                "matrix_file": matrix_file.name,
                "inverse_error": inverse_error,
                "raw_condition": float(torch.linalg.cond(gauge.normalized_metric).detach().cpu()),
                "floored_condition": float((gauge.floored_eigenvalues.max() / gauge.floored_eigenvalues.min()).detach().cpu()),
                "clipped_count": gauge.clipped_count,
                "clipped_mass": gauge.clipped_mass,
            }
        )
    return pd.DataFrame(rows)


def _normal_placebo(
    projector: Any,
    task_normal: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    norm = _norm(task_normal)
    if norm <= 1.0e-30:
        return torch.zeros_like(task_normal)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    task64 = task_normal.detach().double()
    for _ in range(32):
        candidate = torch.randn(task_normal.numel(), generator=generator, dtype=torch.float64).to(task_normal.device)
        candidate = candidate - projector.project64(candidate)
        candidate = candidate - task64 * (torch.dot(candidate, task64) / torch.dot(task64, task64))
        candidate_norm = _norm(candidate)
        if candidate_norm > 1.0e-12:
            return (candidate * (norm / candidate_norm)).to(dtype=task_normal.dtype)
    raise RuntimeError("could not construct a deterministic orthogonal normal-space placebo")


def _augmented_components(
    jacobian: torch.Tensor,
    ambient_proposal: torch.Tensor,
    *,
    rcond: float = PROJECTOR_RCOND,
    placebo_seed: int,
) -> dict[str, torch.Tensor | Any]:
    projector = _build_projector(jacobian.float(), rcond=float(rcond))
    tangent = projector.project(ambient_proposal)
    task_normal = ambient_proposal - tangent
    placebo = _normal_placebo(projector, task_normal, seed=int(placebo_seed))
    return {
        "projector": projector,
        "latent_preimage": projector.latent_preimage(ambient_proposal),
        "tangent": tangent,
        "task_normal": task_normal,
        "placebo": placebo,
    }


def _match_exogenous_radius(
    candidate: Callable[[float], torch.Tensor],
    *,
    current: torch.Tensor,
    target: float,
) -> RadiusMatch:
    if not math.isfinite(float(target)) or float(target) < 0.0:
        raise ValueError(f"target radius must be finite and nonnegative, got={target}")
    observations = [(0.0, 0.0)]
    values: dict[float, torch.Tensor] = {0.0: current.detach().clone()}
    for scale in TRUST_SCAN_SCALES[1:]:
        value = candidate(float(scale)).detach()
        values[float(scale)] = value
        observations.append((float(scale), _norm(value - current)))
    crossings = _first_upward_crossings(observations, float(target))
    if crossings:
        (lo, lo_radius), (hi, hi_radius) = crossings[0]
        scale = lo if abs(lo_radius - target) <= abs(hi_radius - target) else hi
        value = values[float(scale)]
        radius = _norm(value - current)
        reason = "scan_endpoint"
        for _ in range(40):
            if abs(radius - target) / max(target, 1.0e-30) <= 0.0025:
                reason = "matched"
                break
            mid = 0.5 * (lo + hi)
            mid_value = candidate(float(mid)).detach()
            mid_radius = _norm(mid_value - current)
            if abs(mid_radius - target) < abs(radius - target):
                scale, value, radius = float(mid), mid_value, float(mid_radius)
            if mid_radius >= target:
                hi = float(mid)
            else:
                lo = float(mid)
        else:
            reason = "max_iterations"
        bracket = True
    else:
        scale, radius = min(observations, key=lambda item: abs(item[1] - target))
        value = values[float(scale)]
        reason = "no_upward_crossing"
        bracket = False
    relative_error = abs(float(radius) - float(target)) / max(float(target), 1.0e-30)
    return RadiusMatch(
        float(scale), value, float(radius), float(relative_error), bracket,
        len(crossings), bool(float(scale) >= float(TRUST_SCAN_SCALES[-1])), reason,
    )


def _real_gauge_curve(
    *,
    context: ProductionVAEContext,
    arm: Arm,
    gauge: Gauge,
    z0: torch.Tensor,
    theta0: torch.Tensor,
    target_radii: Sequence[float],
    task_set: Any,
    spec: Any,
    tau: float,
    stream_start_index: int,
    batch_size: int,
    jacobian_chunk_size: int,
    common: Mapping[str, Any],
    steps: int = PRODUCTION_STEPS,
    occupancy_steps: Sequence[int] = OCCUPANCY_STEPS,
    counters: RuntimeCounters | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = gauge.to_gauge(z0.detach().double()).detach()
    adam_state = AdamState.zeros_like(y)
    curves: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    occupancy: list[dict[str, Any]] = []
    cumulative = 0.0
    occupancy_step_set = {int(value) for value in occupancy_steps}
    for step in range(int(steps) + 1):
        with torch.no_grad():
            z = gauge.from_gauge(y).to(device=z0.device, dtype=torch.float32)
            theta = _celo_decode_gauged(
                context.vae,
                context.normalizer,
                y,
                gauge,
                counters=counters,
            ).detach()
        metrics = _eval(theta, task_set=task_set, spec=spec, tau=float(tau))
        if counters is not None:
            counters.full_split_eval_calls += 1
        curves.append(
            {
                **common,
                "arm_id": arm.arm_id,
                "optimizer": arm.optimizer,
                "gauge": arm.gauge,
                "rcond_label": arm.rcond_label,
                "rcond": arm.rcond,
                "step": int(step),
                "cumulative_weight_path": cumulative,
                "state_theta_sha256": _tensor_sha256(theta),
                "state_z_sha256": _tensor_sha256(z),
                "state_y_sha256": _tensor_sha256(y),
                "decoded_start_reference_sha256": _tensor_sha256(theta0),
                "decoded_start_max_abs_error": float((theta - theta0).abs().max().detach().cpu()) if step == 0 else float("nan"),
                **metrics,
            }
        )
        if step in occupancy_step_set:
            jacobian32 = _celo_explicit_fp32_jacobian(
                context.vae,
                context.normalizer,
                z,
                chunk_size=int(jacobian_chunk_size),
                counters=counters,
            )
            metric64 = jacobian32.double().T @ jacobian32.double()
            transformed = gauge.metric_to_gauge(metric64)
            occupancy.append(
                {
                    **common,
                    "arm_id": arm.arm_id,
                    "optimizer": arm.optimizer,
                    "gauge": arm.gauge,
                    "rcond_label": arm.rcond_label,
                    "occupancy_step": int(step),
                    "state_z_sha256": _tensor_sha256(z),
                    "state_y_sha256": _tensor_sha256(y),
                    "jacobian_sha256": _tensor_sha256(jacobian32),
                    "metric_sha256": _tensor_sha256(metric64),
                    "transformed_metric_sha256": _tensor_sha256(transformed),
                    "metric_rank": int(torch.linalg.matrix_rank(metric64).detach().cpu()),
                    "metric_condition": float(torch.linalg.cond(metric64).detach().cpu()),
                    "transformed_metric_condition": float(torch.linalg.cond(transformed).detach().cpu()),
                }
            )
        if step == int(steps):
            break
        batch_indices = _batch_indices(
            task_set,
            batch_size=int(batch_size),
            step=step,
            start_bank_position=int(stream_start_index),
        )

        def batch_loss_fn(value: torch.Tensor) -> torch.Tensor:
            return _loss_acc(
                value,
                task_set=task_set,
                spec=spec,
                split="train",
                tau=float(tau),
                batch_indices=batch_indices,
            )[0]

        grad_y, theta_check, batch_loss = _gauge_loss_grad(
            y,
            vae=context.vae,
            normalizer=context.normalizer,
            gauge=gauge,
            loss_fn=batch_loss_fn,
            counters=counters,
        )
        if counters is not None:
            counters.batch_gradient_calls += 1
        if _tensor_sha256(theta_check) != _tensor_sha256(theta):
            raise RuntimeError(f"gauge gradient decode mismatch arm={arm.arm_id} step={step}")
        optimizer_step_before = int(adam_state.step)
        if arm.optimizer == "z_sgd":
            dy = -grad_y * float(context.latent_lr)
        elif arm.optimizer == "z_adam":
            dy = _adam_delta(adam_state, grad_y, lr=float(context.latent_lr))
        else:
            raise ValueError(f"unsupported gauge optimizer={arm.optimizer}")
        match = _match_exogenous_radius(
            lambda scale: _celo_decode_gauged(
                context.vae,
                context.normalizer,
                y + dy * float(scale),
                gauge,
                counters=counters,
                trust_candidate=True,
            ),
            current=theta,
            target=float(target_radii[step]),
        )
        y_new = (y + dy * float(match.scale)).detach()
        z_new = gauge.from_gauge(y_new).to(device=z0.device, dtype=torch.float32)
        realized_theta = match.value.detach()
        trust_executable = bool(
            match.bracket_found
            and not match.boundary_hit
            and math.isfinite(match.relative_error)
            and match.relative_error <= 0.01
        )
        cumulative += match.radius
        diagnostics.append(
            {
                **common,
                "arm_id": arm.arm_id,
                "optimizer": arm.optimizer,
                "gauge": arm.gauge,
                "rcond_label": arm.rcond_label,
                "proposal_step": int(step),
                "batch_indices_sha256": _tensor_sha256(batch_indices),
                "batch_loss": batch_loss,
                "state_theta_sha256": _tensor_sha256(theta),
                "next_theta_sha256": _tensor_sha256(realized_theta),
                "state_z_sha256": _tensor_sha256(z),
                "next_z_sha256": _tensor_sha256(z_new),
                "gradient_y_sha256": _tensor_sha256(grad_y),
                "proposal_y_sha256": _tensor_sha256(dy),
                "optimizer_state_step_before": optimizer_step_before,
                "target_raw_radius": float(target_radii[step]),
                "realized_radius": match.radius,
                "trust_match_rel_error": match.relative_error,
                "trust_scale": match.scale,
                "trust_scale_boundary_hit": match.boundary_hit,
                "trust_bracket_found": match.bracket_found,
                "trust_crossing_count": match.crossing_count,
                "trust_termination_reason": match.termination_reason,
                "trust_executable": trust_executable,
                "operator_executable": True,
                "retraction_executable": True,
                "treatment_executable": trust_executable,
                "singularity_state_local": False,
                "paired_parent_theta_sha256": "not_applicable",
                "tangent_proposal_sha256": "not_applicable",
                "executed_latent_delta_sha256": _tensor_sha256(z_new - z),
                "retraction_rel_error": float("nan"),
                "retraction_cosine": float("nan"),
            }
        )
        y = y_new
    return curves, diagnostics, occupancy


def _pair_trust_gate(
    *,
    task_error: float,
    placebo_error: float,
    bracket_found: bool,
    boundary_hit: bool,
) -> bool:
    return bool(
        bracket_found
        and not boundary_hit
        and math.isfinite(float(task_error))
        and math.isfinite(float(placebo_error))
        and float(task_error) <= 0.01
        and float(placebo_error) <= 0.01
    )


def _local_projector_stats(projector: Any) -> dict[str, Any]:
    kept = projector.singular_values[projector.keep]
    condition = float(kept.max().detach().cpu() / kept.min().detach().cpu()) if kept.numel() else float("inf")
    singular_max = float(projector.singular_values.max().detach().cpu())
    ranks = {
        rcond: int((projector.singular_values > singular_max * rcond).sum().detach().cpu())
        for rcond in RCONDS
    }
    return {
        "projector_rank": int(projector.rank),
        "projector_condition": condition,
        "projector_rank_rcond_1e6": ranks[1.0e-6],
        "projector_rank_rcond_1e5": ranks[1.0e-5],
        "projector_rank_rcond_1e4": ranks[1.0e-4],
        "projector_singular_max": singular_max,
        "projector_singular_min_kept": float(kept.min().detach().cpu()) if kept.numel() else float("nan"),
        "singularity_state_local": bool(condition > 1.0e5 or ranks[1.0e-6] != ranks[1.0e-5]),
    }


def _real_pullback_curve(
    *,
    context: ProductionVAEContext,
    z0: torch.Tensor,
    theta0: torch.Tensor,
    target_radii: Sequence[float],
    task_set: Any,
    spec: Any,
    tau: float,
    stream_start_index: int,
    batch_size: int,
    jacobian_chunk_size: int,
    common: Mapping[str, Any],
    steps: int,
    counters: RuntimeCounters | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    z = z0.detach().clone()
    cumulative = 0.0
    curves: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    def decode_latent(value: torch.Tensor, *, trust_candidate: bool = False) -> torch.Tensor:
        decoded = decode_weights(context.vae, context.normalizer, value.reshape(1, -1)).squeeze(0).float()
        if counters is not None:
            counters.decoder_forward_calls += 1
            counters.trust_candidate_calls += int(trust_candidate)
        return decoded

    for step in range(int(steps) + 1):
        with torch.no_grad():
            theta = decode_latent(z).detach()
        metrics = _eval(theta, task_set=task_set, spec=spec, tau=float(tau))
        if counters is not None:
            counters.full_split_eval_calls += 1
        curves.append(
            {
                **common,
                "arm_id": "pullback",
                "optimizer": "pullback",
                "gauge": "local",
                "rcond_label": "1e-5",
                "rcond": PROJECTOR_RCOND,
                "step": int(step),
                "cumulative_weight_path": cumulative,
                "state_theta_sha256": _tensor_sha256(theta),
                "state_z_sha256": _tensor_sha256(z),
                "state_residual_sha256": _tensor_sha256(torch.zeros_like(theta)),
                "trajectory_semantics": "independent_tangent_only_rollout",
                "decoded_start_reference_sha256": _tensor_sha256(theta0),
                "decoded_start_max_abs_error": float((theta - theta0).abs().max().detach().cpu()) if step == 0 else float("nan"),
                **metrics,
            }
        )
        if step == int(steps):
            break
        batch_indices = _batch_indices(
            task_set,
            batch_size=int(batch_size),
            step=step,
            start_bank_position=int(stream_start_index),
        )
        grad_theta, batch_loss = _theta_grad(
            theta,
            task_set=task_set,
            spec=spec,
            tau=float(tau),
            batch_indices=batch_indices,
        )
        if counters is not None:
            counters.batch_gradient_calls += 1
        jacobian32 = _celo_explicit_fp32_jacobian(
            context.vae,
            context.normalizer,
            z,
            chunk_size=int(jacobian_chunk_size),
            counters=counters,
        )
        ambient = -grad_theta
        projector = _build_projector(jacobian32, rcond=PROJECTOR_RCOND)
        dz = projector.latent_preimage(ambient)
        tangent = projector.project(ambient)
        match = _match_exogenous_radius(
            lambda scale: decode_latent(z + dz * float(scale), trust_candidate=True),
            current=theta,
            target=float(target_radii[step]),
        )
        z_new = (z + dz * float(match.scale)).detach()
        theta_new = match.value.detach()
        dz_executed = z_new - z
        tangent_linear = jacobian32 @ dz_executed
        delta = theta_new - theta
        retraction_error = _norm(delta - tangent_linear) / max(_norm(tangent_linear), 1.0e-30)
        retraction_cosine = _cosine(delta, tangent_linear)
        preimage = _executed_preimage_residuals(projector, ambient, dz_executed, scale=float(match.scale))
        trust_executable = bool(
            match.bracket_found
            and not match.boundary_hit
            and math.isfinite(match.relative_error)
            and match.relative_error <= 0.01
        )
        operator_executable = bool(preimage["algebraic"] <= 1.0e-10 and preimage["executable"] <= 1.0e-4)
        retraction_executable = bool(
            math.isfinite(retraction_error)
            and math.isfinite(retraction_cosine)
            and retraction_error <= 0.25
            and retraction_cosine >= 0.95
        )
        cumulative += match.radius
        diagnostics.append(
            {
                **common,
                "arm_id": "pullback",
                "optimizer": "pullback",
                "gauge": "local",
                "rcond_label": "1e-5",
                "proposal_step": int(step),
                "batch_indices_sha256": _tensor_sha256(batch_indices),
                "batch_loss": batch_loss,
                "state_theta_sha256": _tensor_sha256(theta),
                "next_theta_sha256": _tensor_sha256(theta_new),
                "state_z_sha256": _tensor_sha256(z),
                "next_z_sha256": _tensor_sha256(z_new),
                "paired_parent_theta_sha256": "not_applicable",
                "tangent_proposal_sha256": _tensor_sha256(tangent),
                "executed_latent_delta_sha256": _tensor_sha256(dz_executed),
                "proposal_policy": "independent_tangent_only_full_radius_match",
                "target_raw_radius": float(target_radii[step]),
                "realized_radius": match.radius,
                "trust_match_rel_error": match.relative_error,
                "full_realized_radius_error_task": float("nan"),
                "full_realized_radius_error_placebo": float("nan"),
                "pair_trust_executable": False,
                "trust_scale": match.scale,
                "trust_scale_boundary_hit": match.boundary_hit,
                "trust_bracket_found": match.bracket_found,
                "trust_crossing_count": match.crossing_count,
                "trust_termination_reason": match.termination_reason,
                "preimage_algebraic_residual": preimage["algebraic"],
                "preimage_quantization_residual": preimage["quantization"],
                "preimage_executable_residual": preimage["executable"],
                "retraction_rel_error": retraction_error,
                "retraction_cosine": retraction_cosine,
                "trust_executable": trust_executable,
                "operator_executable": operator_executable,
                "retraction_executable": retraction_executable,
                "treatment_executable": bool(trust_executable and operator_executable and retraction_executable),
                **_local_projector_stats(projector),
            }
        )
        z = z_new
    return curves, diagnostics


def _real_augmented_pair_curves(
    *,
    context: ProductionVAEContext,
    z0: torch.Tensor,
    theta0: torch.Tensor,
    target_radii: Sequence[float],
    task_set: Any,
    spec: Any,
    tau: float,
    stream_start_index: int,
    batch_size: int,
    jacobian_chunk_size: int,
    common: Mapping[str, Any],
    placebo_seed: int,
    steps: int,
    counters: RuntimeCounters | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arm_ids = ("augmented_task_normal", "augmented_placebo")
    arm_map = {arm.arm_id: arm for arm in _arm_grid()}
    z = z0.detach().clone()
    residual = torch.zeros_like(theta0)
    task_state = theta0.detach().clone()
    placebo_state = theta0.detach().clone()
    cumulative = {arm_id: 0.0 for arm_id in arm_ids}
    curves: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    def decode_latent(value: torch.Tensor, *, trust_candidate: bool = False) -> torch.Tensor:
        decoded = decode_weights(context.vae, context.normalizer, value.reshape(1, -1)).squeeze(0).float()
        if counters is not None:
            counters.decoder_forward_calls += 1
            counters.trust_candidate_calls += int(trust_candidate)
        return decoded

    def append_curve(
        arm_id: str,
        state: torch.Tensor,
        residual_state: torch.Tensor,
        *,
        step: int,
    ) -> None:
        arm = arm_map[arm_id]
        metrics = _eval(state, task_set=task_set, spec=spec, tau=float(tau))
        if counters is not None:
            counters.full_split_eval_calls += 1
        curves.append(
            {
                **common,
                "arm_id": arm_id,
                "optimizer": arm.optimizer,
                "gauge": arm.gauge,
                "rcond_label": arm.rcond_label,
                "rcond": arm.rcond,
                "step": int(step),
                "cumulative_weight_path": cumulative[arm_id],
                "state_theta_sha256": _tensor_sha256(state),
                "state_z_sha256": _tensor_sha256(z),
                "state_residual_sha256": _tensor_sha256(residual_state),
                "trajectory_semantics": "sequential_task_parent_paired_counterfactual_fork",
                "decoded_start_reference_sha256": _tensor_sha256(theta0),
                "decoded_start_max_abs_error": float((state - theta0).abs().max().detach().cpu()) if step == 0 else float("nan"),
                **metrics,
            }
        )

    append_curve("augmented_task_normal", task_state, residual, step=0)
    append_curve("augmented_placebo", placebo_state, residual, step=0)
    for step in range(int(steps)):
        parent_theta = task_state.detach()
        batch_indices = _batch_indices(
            task_set,
            batch_size=int(batch_size),
            step=step,
            start_bank_position=int(stream_start_index),
        )
        grad_theta, batch_loss = _theta_grad(
            parent_theta,
            task_set=task_set,
            spec=spec,
            tau=float(tau),
            batch_indices=batch_indices,
        )
        if counters is not None:
            counters.batch_gradient_calls += 1
        jacobian32 = _celo_explicit_fp32_jacobian(
            context.vae,
            context.normalizer,
            z,
            chunk_size=int(jacobian_chunk_size),
            counters=counters,
        )
        ambient = -grad_theta
        components = _augmented_components(
            jacobian32,
            ambient,
            rcond=PROJECTOR_RCOND,
            placebo_seed=_stable_seed(placebo_seed, common["source_weight_index"], step),
        )
        projector = components["projector"]
        dz = torch.as_tensor(components["latent_preimage"])
        tangent = torch.as_tensor(components["tangent"])
        task_normal = torch.as_tensor(components["task_normal"])
        placebo = torch.as_tensor(components["placebo"])
        task_normal_norm = _norm(task_normal)
        placebo_norm = _norm(placebo)
        normal_norm_rel_error = abs(task_normal_norm - placebo_norm) / max(task_normal_norm, 1.0e-30)
        if normal_norm_rel_error > 1.0e-6:
            raise RuntimeError(f"paired normal norms differ at source={common['source_weight_index']} step={step}")

        def full_task_candidate(scale: float) -> torch.Tensor:
            base = decode_latent(z + dz * float(scale), trust_candidate=True)
            return base + residual + task_normal * float(scale)

        match = _match_exogenous_radius(
            full_task_candidate,
            current=parent_theta,
            target=float(target_radii[step]),
        )
        scale = float(match.scale)
        z_new = (z + dz * scale).detach()
        task_next = match.value.detach()
        base_new = (task_next - residual - task_normal * scale).detach()
        placebo_next = (base_new + residual + placebo * scale).detach()
        dz_executed = z_new - z
        task_delta = task_next - parent_theta
        placebo_delta = placebo_next - parent_theta
        task_radius = _norm(task_delta)
        placebo_radius = _norm(placebo_delta)
        target_radius = float(target_radii[step])
        task_error = abs(task_radius - target_radius) / max(target_radius, 1.0e-30)
        placebo_error = abs(placebo_radius - target_radius) / max(target_radius, 1.0e-30)
        pair_trust_executable = _pair_trust_gate(
            task_error=task_error,
            placebo_error=placebo_error,
            bracket_found=match.bracket_found,
            boundary_hit=match.boundary_hit,
        )
        task_linear = tangent * scale + task_normal * scale
        placebo_linear = tangent * scale + placebo * scale
        task_retraction_error = _norm(task_delta - task_linear) / max(_norm(task_linear), 1.0e-30)
        placebo_retraction_error = _norm(placebo_delta - placebo_linear) / max(_norm(placebo_linear), 1.0e-30)
        task_retraction_cosine = _cosine(task_delta, task_linear)
        placebo_retraction_cosine = _cosine(placebo_delta, placebo_linear)
        task_retraction_executable = bool(
            math.isfinite(task_retraction_error)
            and math.isfinite(task_retraction_cosine)
            and task_retraction_error <= 0.25
            and task_retraction_cosine >= 0.95
        )
        placebo_retraction_executable = bool(
            math.isfinite(placebo_retraction_error)
            and math.isfinite(placebo_retraction_cosine)
            and placebo_retraction_error <= 0.25
            and placebo_retraction_cosine >= 0.95
        )
        pair_retraction_executable = bool(task_retraction_executable and placebo_retraction_executable)
        preimage = _executed_preimage_residuals(projector, ambient, dz_executed, scale=scale)
        operator_executable = bool(preimage["algebraic"] <= 1.0e-10 and preimage["executable"] <= 1.0e-4)
        pair_treatment_executable = bool(pair_trust_executable and operator_executable and pair_retraction_executable)
        cumulative["augmented_task_normal"] += task_radius
        cumulative["augmented_placebo"] += placebo_radius
        branch_values = {
            "augmented_task_normal": (task_next, task_normal, task_radius, task_error, task_retraction_error, task_retraction_cosine),
            "augmented_placebo": (placebo_next, placebo, placebo_radius, placebo_error, placebo_retraction_error, placebo_retraction_cosine),
        }
        stats = _local_projector_stats(projector)
        for arm_id, (next_state, normal, radius, error, retract_error, retract_cosine) in branch_values.items():
            diagnostics.append(
                {
                    **common,
                    "arm_id": arm_id,
                    "optimizer": "augmented",
                    "gauge": "task_normal" if arm_id == "augmented_task_normal" else "placebo",
                    "rcond_label": "1e-5",
                    "proposal_step": int(step),
                    "batch_indices_sha256": _tensor_sha256(batch_indices),
                    "batch_loss": batch_loss,
                    "state_theta_sha256": _tensor_sha256(parent_theta),
                    "next_theta_sha256": _tensor_sha256(next_state),
                    "state_z_sha256": _tensor_sha256(z),
                    "next_z_sha256": _tensor_sha256(z_new),
                    "state_residual_sha256": _tensor_sha256(residual),
                    "next_residual_sha256": _tensor_sha256(residual + normal * scale),
                    "paired_parent_theta_sha256": _tensor_sha256(parent_theta),
                    "tangent_proposal_sha256": _tensor_sha256(tangent),
                    "executed_latent_delta_sha256": _tensor_sha256(dz_executed),
                    "normal_component_sha256": _tensor_sha256(normal),
                    "proposal_policy": "full_task_normal_radius_common_parent",
                    "target_raw_radius": target_radius,
                    "realized_radius": radius,
                    "trust_match_rel_error": error,
                    "full_realized_radius_task": task_radius,
                    "full_realized_radius_placebo": placebo_radius,
                    "full_realized_radius_error_task": task_error,
                    "full_realized_radius_error_placebo": placebo_error,
                    "pair_trust_executable": pair_trust_executable,
                    "pair_operator_executable": operator_executable,
                    "pair_retraction_executable": pair_retraction_executable,
                    "normal_norm_relative_error": normal_norm_rel_error,
                    "trust_scale": scale,
                    "trust_scale_boundary_hit": match.boundary_hit,
                    "trust_bracket_found": match.bracket_found,
                    "trust_crossing_count": match.crossing_count,
                    "trust_termination_reason": match.termination_reason,
                    "jacobian_sha256": _tensor_sha256(jacobian32),
                    "ambient_proposal_sha256": _tensor_sha256(ambient),
                    "task_normal_norm": task_normal_norm,
                    "placebo_normal_norm": placebo_norm,
                    "normal_component_norm": _norm(normal),
                    "placebo_task_normal_dot": float(torch.dot(placebo.double(), task_normal.double()).detach().cpu()),
                    "placebo_jt_norm": _norm(jacobian32.double().T @ placebo.double()),
                    "preimage_algebraic_residual": preimage["algebraic"],
                    "preimage_quantization_residual": preimage["quantization"],
                    "preimage_executable_residual": preimage["executable"],
                    "retraction_rel_error": retract_error,
                    "retraction_cosine": retract_cosine,
                    "trust_executable": pair_trust_executable,
                    "operator_executable": operator_executable,
                    "retraction_executable": pair_retraction_executable,
                    "treatment_executable": pair_treatment_executable,
                    **stats,
                }
            )
        z = z_new
        task_next_residual = (residual + task_normal * scale).detach()
        placebo_next_residual = (residual + placebo * scale).detach()
        residual = task_next_residual
        task_state = task_next
        placebo_state = placebo_next
        append_curve("augmented_task_normal", task_state, task_next_residual, step=step + 1)
        append_curve("augmented_placebo", placebo_state, placebo_next_residual, step=step + 1)
    return curves, diagnostics


def _real_coupled_local_curves(
    *,
    context: ProductionVAEContext,
    z0: torch.Tensor,
    theta0: torch.Tensor,
    target_radii: Sequence[float],
    task_set: Any,
    spec: Any,
    tau: float,
    stream_start_index: int,
    batch_size: int,
    jacobian_chunk_size: int,
    common: Mapping[str, Any],
    placebo_seed: int,
    steps: int = PRODUCTION_STEPS,
    counters: RuntimeCounters | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pullback_curves, pullback_diagnostics = _real_pullback_curve(
        context=context,
        z0=z0,
        theta0=theta0,
        target_radii=target_radii,
        task_set=task_set,
        spec=spec,
        tau=tau,
        stream_start_index=stream_start_index,
        batch_size=batch_size,
        jacobian_chunk_size=jacobian_chunk_size,
        common=common,
        steps=int(steps),
        counters=counters,
    )
    pair_curves, pair_diagnostics = _real_augmented_pair_curves(
        context=context,
        z0=z0,
        theta0=theta0,
        target_radii=target_radii,
        task_set=task_set,
        spec=spec,
        tau=tau,
        stream_start_index=stream_start_index,
        batch_size=batch_size,
        jacobian_chunk_size=jacobian_chunk_size,
        common=common,
        placebo_seed=placebo_seed,
        steps=int(steps),
        counters=counters,
    )
    return pullback_curves + pair_curves, pullback_diagnostics + pair_diagnostics


def _validate_exact_grid(
    frame: pd.DataFrame,
    *,
    dimensions: Mapping[str, Sequence[Any]],
    key_columns: Sequence[str],
    artifact_name: str,
) -> dict[str, Any]:
    missing_columns = set(key_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{artifact_name} missing key columns: {sorted(missing_columns)}")
    expected_frame = pd.MultiIndex.from_product(
        [list(dimensions[column]) for column in key_columns], names=list(key_columns)
    ).to_frame(index=False)
    expected = {tuple(row) for row in expected_frame.itertuples(index=False, name=None)}
    actual_rows = [tuple(row) for row in frame[list(key_columns)].itertuples(index=False, name=None)]
    actual = set(actual_rows)
    duplicates = len(actual_rows) - len(actual)
    missing = expected - actual
    unexpected = actual - expected
    if duplicates or missing or unexpected or len(frame) != len(expected):
        raise ValueError(
            f"{artifact_name} Cartesian keys are not exact: rows={len(frame)} expected={len(expected)} "
            f"duplicates={duplicates} missing={len(missing)} unexpected={len(unexpected)}"
        )
    return {"accepted": True, "rows": len(frame), "expected_rows": len(expected)}


def _primary_aggregation(
    curves: pd.DataFrame,
    *,
    steps: int = PRODUCTION_STEPS,
    expected_vae_seeds: Sequence[str] = PRODUCTION_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"vae_seed", "source_weight_index", "arm_id", "step", "train_loss"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"primary aggregation missing columns: {sorted(missing)}")
    if curves.duplicated(["vae_seed", "source_weight_index", "arm_id", "step"]).any():
        raise ValueError("primary aggregation received duplicate curve checkpoints")
    rows = []
    for keys, group in curves.groupby(["vae_seed", "source_weight_index", "arm_id"], sort=False):
        ordered = group.sort_values("step")
        if tuple(ordered["step"].astype(int)) != tuple(range(int(steps) + 1)):
            raise ValueError(f"primary curve has non-exact steps for {keys}")
        start_loss = float(ordered.iloc[0]["train_loss"])
        post0_aulc = float(ordered.iloc[1:]["train_loss"].mean())
        rows.append(
            {
                "vae_seed": str(keys[0]),
                "source_weight_index": int(keys[1]),
                "arm_id": str(keys[2]),
                "train_step0_loss": start_loss,
                "train_post0_aulc": post0_aulc,
                "train_post0_aulc_progress": start_loss - post0_aulc,
            }
        )
    per_start = pd.DataFrame(rows)
    per_seed = (
        per_start.groupby(["vae_seed", "arm_id"], as_index=False)["train_post0_aulc_progress"]
        .mean()
        .rename(columns={"train_post0_aulc_progress": "vae_seed_mean_progress"})
    )
    seed_counts = per_seed.groupby("arm_id")["vae_seed"].nunique()
    if not seed_counts.eq(len(expected_vae_seeds)).all():
        raise ValueError(
            f"primary inference requires exactly {len(expected_vae_seeds)} VAE seeds per arm: "
            f"{seed_counts.to_dict()}"
        )
    aggregate = (
        per_seed.groupby("arm_id", as_index=False)["vae_seed_mean_progress"]
        .mean()
        .rename(columns={"vae_seed_mean_progress": "equal_weight_vae_seed_mean_progress"})
    )
    aggregate["n_vae_seeds"] = len(expected_vae_seeds)
    return per_start, per_seed, aggregate


def _primary_with_execution(
    curves: pd.DataFrame,
    flags: pd.DataFrame,
    *,
    steps: int = PRODUCTION_STEPS,
    expected_vae_seeds: Sequence[str] = PRODUCTION_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_start, per_seed, aggregate = _primary_aggregation(
        curves,
        steps=int(steps),
        expected_vae_seeds=expected_vae_seeds,
    )
    per_start = per_start.merge(
        flags,
        on=["vae_seed", "source_weight_index", "arm_id"],
        how="left",
        validate="one_to_one",
    )
    if per_start["trajectory_treatment_executable"].isna().any():
        raise RuntimeError("primary summaries are missing execution flags")
    return per_start, per_seed, aggregate


def _paired_contrast_outputs(
    per_start: pd.DataFrame,
    *,
    expected_vae_seeds: Sequence[str] = PRODUCTION_LABELS,
    expected_sources: Sequence[int] = PRODUCTION_EVAL_SOURCES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    equal_seed_rows: list[dict[str, Any]] = []
    for contrast, left_id, right_id, cutoff, estimand in CONTRAST_SPECS:
        left = per_start[per_start["arm_id"] == left_id]
        right = per_start[per_start["arm_id"] == right_id]
        joined = left.merge(
            right,
            on=["vae_seed", "source_weight_index"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        if len(joined) != len(expected_vae_seeds) * len(expected_sources):
            raise ValueError(f"contrast {contrast} has incomplete paired starts: {len(joined)}")
        joined["contrast_value_left_better_positive"] = (
            joined["train_post0_aulc_progress_left"] - joined["train_post0_aulc_progress_right"]
        )
        joined["pair_executable"] = (
            joined["trajectory_treatment_executable_left"].astype(bool)
            & joined["trajectory_treatment_executable_right"].astype(bool)
        )
        tangent_held = bool(contrast == "task_normal_minus_placebo")
        same_total_radius_reallocation = bool(contrast == "task_normal_minus_pullback")
        for row in joined.itertuples(index=False):
            pair_rows.append(
                {
                    "contrast": contrast,
                    "estimand": estimand,
                    "cutoff": cutoff,
                    "left_arm_id": left_id,
                    "right_arm_id": right_id,
                    "vae_seed": str(row.vae_seed),
                    "source_weight_index": int(row.source_weight_index),
                    "endpoint": "train_post0_aulc_progress",
                    "contrast_value_left_better_positive": float(row.contrast_value_left_better_positive),
                    "left_treatment_executable": bool(row.trajectory_treatment_executable_left),
                    "right_treatment_executable": bool(row.trajectory_treatment_executable_right),
                    "pair_executable": bool(row.pair_executable),
                    "tangent_held_equal_by_design": tangent_held,
                    "same_total_radius_reallocation": same_total_radius_reallocation,
                }
            )
        for analysis_set in ("all_start_itt", "executable_sensitivity"):
            eligible = joined if analysis_set == "all_start_itt" else joined[joined["pair_executable"]]
            for vae_seed in expected_vae_seeds:
                values = eligible.loc[
                    eligible["vae_seed"] == vae_seed,
                    "contrast_value_left_better_positive",
                ].to_numpy(dtype=np.float64)
                seed_rows.append(
                    {
                        "contrast": contrast,
                        "estimand": estimand,
                        "cutoff": cutoff,
                        "left_arm_id": left_id,
                        "right_arm_id": right_id,
                        "analysis_set": analysis_set,
                        "vae_seed": vae_seed,
                        "endpoint": "train_post0_aulc_progress",
                        "n_starts": int(values.size),
                        "vae_seed_mean_left_better_positive": float(values.mean()) if values.size else float("nan"),
                        "vae_seed_median_left_better_positive": float(np.median(values)) if values.size else float("nan"),
                        "tangent_held_equal_by_design": tangent_held,
                        "same_total_radius_reallocation": same_total_radius_reallocation,
                    }
                )
            current_seed_rows = seed_rows[-len(expected_vae_seeds):]
            finite_means = np.asarray(
                [row["vae_seed_mean_left_better_positive"] for row in current_seed_rows],
                dtype=np.float64,
            )
            finite_means = finite_means[np.isfinite(finite_means)]
            equal_seed_rows.append(
                {
                    "contrast": contrast,
                    "estimand": estimand,
                    "cutoff": cutoff,
                    "left_arm_id": left_id,
                    "right_arm_id": right_id,
                    "analysis_set": analysis_set,
                    "endpoint": "train_post0_aulc_progress",
                    "n_vae_seeds": int(finite_means.size),
                    "equal_weight_vae_seed_mean_left_better_positive": float(finite_means.mean()) if finite_means.size else float("nan"),
                    "tangent_held_equal_by_design": tangent_held,
                    "same_total_radius_reallocation": same_total_radius_reallocation,
                }
            )
    return pd.DataFrame(pair_rows), pd.DataFrame(seed_rows), pd.DataFrame(equal_seed_rows)


def _normalize_raw_outputs(
    curves: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve_rows = []
    for source in curves:
        row = dict(source)
        row.update(
            {
                "arm_id": "raw_adam",
                "optimizer": "raw_adam",
                "gauge": "identity",
                "rcond_label": "none",
                "rcond": None,
                "decoded_start_reference_sha256": row["theta0_sha256"],
                "decoded_start_max_abs_error": 0.0 if int(row["step"]) == 0 else float("nan"),
            }
        )
        row.pop("method", None)
        row.pop("trust_fraction", None)
        curve_rows.append(row)
    diagnostic_rows = []
    for source in diagnostics:
        row = dict(source)
        row.update(
            {
                "arm_id": "raw_adam",
                "optimizer": "raw_adam",
                "gauge": "identity",
                "rcond_label": "none",
                "paired_parent_theta_sha256": "not_applicable",
                "tangent_proposal_sha256": "not_applicable",
                "executed_latent_delta_sha256": "not_applicable",
                "trust_scale_boundary_hit": bool(row.get("trust_scale_boundary_hit", False)),
            }
        )
        row.pop("method", None)
        row.pop("trust_fraction", None)
        diagnostic_rows.append(row)
    return curve_rows, diagnostic_rows


class _SyntheticDecoder:
    def __init__(self, *, seed: int, latent_dim: int = 6, ambient_dim: int = 11) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.weight = torch.randn(ambient_dim, latent_dim, generator=generator, dtype=torch.float32) / math.sqrt(latent_dim)
        self.wave = torch.randn(ambient_dim, latent_dim, generator=generator, dtype=torch.float32) / math.sqrt(latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.weight @ z.float() + 0.03 * torch.sin(self.wave @ z.float())

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        phase = self.wave @ z.float()
        return self.weight + 0.03 * torch.cos(phase).unsqueeze(1) * self.wave


def _synthetic_z0(source: int, latent_dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_stable_seed("z0", source))
    return torch.randn(latent_dim, generator=generator, dtype=torch.float32) * 0.35


def _synthetic_target(source: int, ambient_dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_stable_seed("target", source))
    return torch.randn(ambient_dim, generator=generator, dtype=torch.float32) * 0.25


def _batch_hash(source: int, step: int) -> str:
    values = torch.tensor(
        [(_stable_seed("batch", source, step) + 17 * index) % 4096 for index in range(32)],
        dtype=torch.int64,
    )
    return _tensor_sha256(values)


def _context_hash(row: Mapping[str, Any]) -> str:
    payload = {
        "source_weight_index": int(row["source_weight_index"]),
        "start_bank_position": int(row["start_bank_position"]),
        "task_name": str(row["task_name"]),
        "tau": float(row["tau"]),
    }
    return _stable_json_hash(payload)


def _metric_fit_smoke(
    decoder: _SyntheticDecoder,
    fit_bank: pd.DataFrame,
    eval_bank: pd.DataFrame,
    *,
    vae_seed: str,
    latent_lr: float,
) -> tuple[torch.Tensor, pd.DataFrame, pd.DataFrame]:
    records = []
    trajectory = []
    checkpoints = set(METRIC_STATE_STEPS)
    for index, row in enumerate(fit_bank.to_dict(orient="records"), start=1):
        source = int(row["source_weight_index"])
        z = _synthetic_z0(source, decoder.weight.shape[1])
        target = _synthetic_target(source, decoder.weight.shape[0])
        state = AdamState.zeros_like(z)
        for step in range(PRODUCTION_STEPS + 1):
            state_hash = _tensor_sha256(z)
            if step in checkpoints:
                jacobian32 = decoder.jacobian(z).to(dtype=torch.float32)
                gram64 = jacobian32.double().T @ jacobian32.double()
                records.append(
                    {
                        "vae_seed": vae_seed,
                        "source_weight_index": source,
                        "metric_state_step": step,
                        "state_sha256": state_hash,
                        "jacobian_sha256": _tensor_sha256(jacobian32),
                        "gram_sha256": _tensor_sha256(gram64),
                        "gram": gram64,
                    }
                )
            if step == PRODUCTION_STEPS:
                break
            theta = decoder.decode(z)
            jacobian = decoder.jacobian(z)
            grad_z = jacobian.T @ (theta - target)
            trajectory.append(
                {
                    "vae_seed": vae_seed,
                    "source_weight_index": source,
                    "optimizer_step": step,
                    "state_sha256": state_hash,
                    "batch_indices_sha256": _batch_hash(source, step),
                    "train_loss": 0.5 * float(torch.dot(theta - target, theta - target)),
                }
            )
            z = (z + _adam_delta(state, grad_z, lr=float(latent_lr))).detach()
    record_frame = pd.DataFrame(records)
    metric, deduplicated = _fit_global_metric(
        record_frame,
        expected_fit_sources=PRODUCTION_FIT_SOURCES,
        evaluation_sources=eval_bank["source_weight_index"].astype(int).tolist(),
    )
    serializable = deduplicated.drop(columns="gram").copy()
    serializable["gram_flat_fp64"] = [
        json.dumps(torch.as_tensor(value).reshape(-1).tolist(), separators=(",", ":"))
        for value in deduplicated["gram"]
    ]
    return metric, serializable, pd.DataFrame(trajectory)


def _raw_smoke_curve(
    decoder: _SyntheticDecoder,
    z0: torch.Tensor,
    target: torch.Tensor,
    *,
    source: int,
    raw_lr: float,
) -> tuple[list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    theta = decoder.decode(z0).detach()
    state = AdamState.zeros_like(theta)
    radii: list[float] = []
    curves = []
    diagnostics = []
    for step in range(PRODUCTION_STEPS + 1):
        curves.append({"step": step, "train_loss": 0.5 * float(torch.dot(theta - target, theta - target)), "state_theta_sha256": _tensor_sha256(theta)})
        if step == PRODUCTION_STEPS:
            break
        proposal = _adam_delta(state, theta - target, lr=float(raw_lr))
        theta_new = theta + proposal
        radius = _norm(theta_new - theta)
        radii.append(radius)
        diagnostics.append({"proposal_step": step, "target_raw_radius": radius, "realized_radius": radius, "trust_scale": 1.0, "trust_match_rel_error": 0.0, "trust_bracket_found": True, "trust_boundary_hit": False, "trust_crossing_count": 1, "trust_termination_reason": "raw_reference", "batch_indices_sha256": _batch_hash(source, step)})
        theta = theta_new.detach()
    return radii, curves, diagnostics


def _pure_gauge_smoke_curve(
    decoder: _SyntheticDecoder,
    z0: torch.Tensor,
    target: torch.Tensor,
    radii: Sequence[float],
    *,
    arm: Arm,
    gauge: Gauge,
    latent_lr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = gauge.to_gauge(z0.double()).detach()
    state = AdamState.zeros_like(y)
    curves = []
    diagnostics = []
    occupancy = []
    reference_theta0 = decoder.decode(z0)
    for step in range(PRODUCTION_STEPS + 1):
        z = gauge.from_gauge(y).float()
        theta = decoder.decode(z)
        curves.append(
            {
                "step": step,
                "train_loss": 0.5 * float(torch.dot(theta - target, theta - target)),
                "state_theta_sha256": _tensor_sha256(theta),
                "decoded_start_reference_sha256": _tensor_sha256(reference_theta0),
                "decoded_start_max_abs_error": float((theta - reference_theta0).abs().max()) if step == 0 else float("nan"),
            }
        )
        if step in OCCUPANCY_STEPS:
            jacobian32 = decoder.jacobian(z).float()
            metric64 = jacobian32.double().T @ jacobian32.double()
            transformed = gauge.metric_to_gauge(metric64)
            occupancy.append(
                {
                    "occupancy_step": step,
                    "state_z_sha256": _tensor_sha256(z),
                    "jacobian_sha256": _tensor_sha256(jacobian32),
                    "metric_sha256": _tensor_sha256(metric64),
                    "transformed_metric_sha256": _tensor_sha256(transformed),
                    "metric_condition": float(torch.linalg.cond(metric64)),
                    "transformed_metric_condition": float(torch.linalg.cond(transformed)),
                }
            )
        if step == PRODUCTION_STEPS:
            break
        jacobian = decoder.jacobian(z)
        grad_z = jacobian.T @ (theta - target)
        grad_y = gauge.gradient_to_gauge(grad_z.double())
        if arm.optimizer == "z_sgd":
            dy = -grad_y * float(latent_lr)
        elif arm.optimizer == "z_adam":
            dy = _adam_delta(state, grad_y, lr=float(latent_lr))
        else:
            raise ValueError(f"not a pure gauge optimizer: {arm.optimizer}")
        current = theta.detach()
        match = _match_exogenous_radius(
            lambda scale: decoder.decode(gauge.from_gauge(y + dy * float(scale)).float()),
            current=current,
            target=float(radii[step]),
        )
        y_new = (y + dy * match.scale).detach()
        diagnostics.append(
            {
                "proposal_step": step,
                "target_raw_radius": float(radii[step]),
                "realized_radius": match.radius,
                "trust_scale": match.scale,
                "trust_match_rel_error": match.relative_error,
                "trust_bracket_found": match.bracket_found,
                "trust_boundary_hit": match.boundary_hit,
                "trust_crossing_count": match.crossing_count,
                "trust_termination_reason": match.termination_reason,
            }
        )
        y = y_new
    return curves, diagnostics, occupancy


def _coupled_local_smoke_curves(
    decoder: _SyntheticDecoder,
    z0: torch.Tensor,
    target: torch.Tensor,
    radii: Sequence[float],
    *,
    source: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    theta0 = decoder.decode(z0)

    z_pullback = z0.detach().clone()
    for step in range(PRODUCTION_STEPS + 1):
        state = decoder.decode(z_pullback)
        curves.append(
            {
                "arm_id": "pullback",
                "step": step,
                "train_loss": 0.5 * float(torch.dot(state - target, state - target)),
                "state_theta_sha256": _tensor_sha256(state),
                "decoded_start_reference_sha256": _tensor_sha256(theta0),
                "decoded_start_max_abs_error": float((state - theta0).abs().max()) if step == 0 else float("nan"),
            }
        )
        if step == PRODUCTION_STEPS:
            break
        jacobian = decoder.jacobian(z_pullback).float()
        ambient = -(state - target)
        components = _augmented_components(
            jacobian,
            ambient,
            placebo_seed=_stable_seed("pullback_smoke", source, step),
        )
        dz = torch.as_tensor(components["latent_preimage"])
        tangent = torch.as_tensor(components["tangent"])
        match = _match_exogenous_radius(
            lambda scale: decoder.decode(z_pullback + dz * float(scale)),
            current=state,
            target=float(radii[step]),
        )
        z_new = (z_pullback + dz * match.scale).detach()
        diagnostics.append(
            {
                "arm_id": "pullback",
                "proposal_step": step,
                "target_raw_radius": float(radii[step]),
                "realized_radius": match.radius,
                "trust_scale": match.scale,
                "trust_match_rel_error": match.relative_error,
                "trust_bracket_found": match.bracket_found,
                "trust_boundary_hit": match.boundary_hit,
                "trust_scale_boundary_hit": match.boundary_hit,
                "trust_crossing_count": match.crossing_count,
                "trust_termination_reason": match.termination_reason,
                "paired_parent_theta_sha256": "not_applicable",
                "tangent_proposal_sha256": _tensor_sha256(tangent),
                "executed_latent_delta_sha256": _tensor_sha256(z_new - z_pullback),
            }
        )
        z_pullback = z_new

    z = z0.detach().clone()
    residual = torch.zeros_like(target)
    task_state = theta0.detach().clone()
    placebo_state = theta0.detach().clone()

    def append_pair_curves(step: int) -> None:
        for arm_id, state in (("augmented_task_normal", task_state), ("augmented_placebo", placebo_state)):
            curves.append(
                {
                    "arm_id": arm_id,
                    "step": step,
                    "train_loss": 0.5 * float(torch.dot(state - target, state - target)),
                    "state_theta_sha256": _tensor_sha256(state),
                    "decoded_start_reference_sha256": _tensor_sha256(theta0),
                    "decoded_start_max_abs_error": float((state - theta0).abs().max()) if step == 0 else float("nan"),
                }
            )

    append_pair_curves(0)
    for step in range(PRODUCTION_STEPS):
        parent = task_state
        jacobian = decoder.jacobian(z).float()
        ambient = -(parent - target)
        components = _augmented_components(
            jacobian,
            ambient,
            placebo_seed=_stable_seed("coupled_smoke_placebo", source, step),
        )
        dz = torch.as_tensor(components["latent_preimage"])
        tangent = torch.as_tensor(components["tangent"])
        task_normal = torch.as_tensor(components["task_normal"])
        placebo = torch.as_tensor(components["placebo"])
        match = _match_exogenous_radius(
            lambda scale: decoder.decode(z + dz * float(scale)) + residual + task_normal * float(scale),
            current=parent,
            target=float(radii[step]),
        )
        z_new = (z + dz * match.scale).detach()
        task_next = match.value.detach()
        base_new = (task_next - residual - task_normal * match.scale).detach()
        placebo_next = (base_new + residual + placebo * match.scale).detach()
        executed_dz_hash = _tensor_sha256(z_new - z)
        task_radius = _norm(task_next - parent)
        placebo_radius = _norm(placebo_next - parent)
        target_radius = float(radii[step])
        task_error = abs(task_radius - target_radius) / max(target_radius, 1.0e-30)
        placebo_error = abs(placebo_radius - target_radius) / max(target_radius, 1.0e-30)
        pair_trust = _pair_trust_gate(
            task_error=task_error,
            placebo_error=placebo_error,
            bracket_found=match.bracket_found,
            boundary_hit=match.boundary_hit,
        )
        task_norm = _norm(task_normal)
        placebo_norm = _norm(placebo)
        for arm_id, state, normal, realized, error in (
            ("augmented_task_normal", task_next, task_normal, task_radius, task_error),
            ("augmented_placebo", placebo_next, placebo, placebo_radius, placebo_error),
        ):
            diagnostics.append(
                {
                    "arm_id": arm_id,
                    "proposal_step": step,
                    "target_raw_radius": float(radii[step]),
                    "realized_radius": realized,
                    "trust_scale": match.scale,
                    "trust_match_rel_error": error,
                    "trust_bracket_found": match.bracket_found,
                    "trust_boundary_hit": match.boundary_hit,
                    "trust_scale_boundary_hit": match.boundary_hit,
                    "trust_crossing_count": match.crossing_count,
                    "trust_termination_reason": match.termination_reason,
                    "paired_parent_theta_sha256": _tensor_sha256(parent),
                    "state_z_sha256": _tensor_sha256(z),
                    "tangent_proposal_sha256": _tensor_sha256(tangent),
                    "executed_latent_delta_sha256": executed_dz_hash,
                    "task_normal_norm": task_norm,
                    "placebo_normal_norm": placebo_norm,
                    "normal_component_norm": _norm(normal),
                    "full_realized_radius_task": task_radius,
                    "full_realized_radius_placebo": placebo_radius,
                    "full_realized_radius_error_task": task_error,
                    "full_realized_radius_error_placebo": placebo_error,
                    "pair_trust_executable": pair_trust,
                    "pair_operator_executable": True,
                    "pair_retraction_executable": True,
                    "treatment_executable": pair_trust,
                    "placebo_task_normal_dot": float(torch.dot(placebo.double(), task_normal.double())),
                    "placebo_jt_norm": _norm(jacobian.double().T @ placebo.double()),
                }
            )
        z = z_new
        residual = (residual + task_normal * match.scale).detach()
        task_state = task_next
        placebo_state = placebo_next
        append_pair_curves(step + 1)
    return curves, diagnostics


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False)
    return _sha256_file(path)


def _run_smoke(
    *,
    output_dir: Path,
    eval_bank: pd.DataFrame,
    fit_bank: pd.DataFrame,
    bank_hashes: Mapping[str, str],
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log("stage=synthetic_model_build device=cpu dtype=float32 cache_mode=disabled", verbose=verbose)
    arms = _arm_grid()
    all_curves = []
    all_diagnostics = []
    all_occupancy = []
    all_metric_states = []
    all_metric_trajectory = []
    matrix_rows = []
    raw_lr = 0.015
    latent_lr = 0.02
    for seed_index, vae_seed in enumerate(PRODUCTION_LABELS):
        _log(f"stage=metric_fit vae_seed={vae_seed} fit_starts=16 checkpoints={METRIC_STATE_STEPS}", verbose=verbose)
        decoder = _SyntheticDecoder(seed=_stable_seed(seed, "decoder", seed_index))
        metric, metric_states, metric_trajectory = _metric_fit_smoke(
            decoder, fit_bank, eval_bank, vae_seed=vae_seed, latent_lr=latent_lr
        )
        all_metric_states.append(metric_states)
        all_metric_trajectory.append(metric_trajectory)
        gauges = _gauges_for_metric(metric, seed=_stable_seed(seed, "orthogonal", seed_index))
        matrix_path = output_dir / f"gauge_matrices_{vae_seed}.npz"
        arrays: dict[str, np.ndarray] = {"global_metric": metric.cpu().numpy()}
        for (gauge_name, rcond_label), gauge in gauges.items():
            key = f"{gauge_name}_{rcond_label}".replace("-", "m")
            arrays[f"{key}_A"] = gauge.matrix.cpu().numpy()
            arrays[f"{key}_A_inv"] = gauge.inverse.cpu().numpy()
            arrays[f"{key}_raw_eigenvalues"] = gauge.raw_eigenvalues.cpu().numpy()
            arrays[f"{key}_floored_eigenvalues"] = gauge.floored_eigenvalues.cpu().numpy()
            inverse_error = _norm(gauge.inverse @ gauge.matrix - torch.eye(metric.shape[0], dtype=torch.float64))
            matrix_rows.append(
                {
                    "vae_seed": vae_seed,
                    "gauge": gauge_name,
                    "rcond_label": rcond_label,
                    "rcond": gauge.rcond,
                    "matrix_sha256": _tensor_sha256(gauge.matrix),
                    "inverse_sha256": _tensor_sha256(gauge.inverse),
                    "inverse_error": inverse_error,
                    "raw_condition": float(torch.linalg.cond(gauge.normalized_metric)),
                    "floored_condition": float(gauge.floored_eigenvalues.max() / gauge.floored_eigenvalues.min()),
                    "clipped_count": gauge.clipped_count,
                    "clipped_mass": gauge.clipped_mass,
                }
            )
        np.savez(matrix_path, **arrays)
        _log(f"stage=evaluation vae_seed={vae_seed} starts=16 arms={len(arms)} steps={PRODUCTION_STEPS}", verbose=verbose)
        for start_number, start_row in enumerate(eval_bank.to_dict(orient="records"), start=1):
            source = int(start_row["source_weight_index"])
            context_hash = _context_hash(start_row)
            z0 = _synthetic_z0(source, decoder.weight.shape[1])
            target = _synthetic_target(source, decoder.weight.shape[0])
            radii, raw_curves, raw_diagnostics = _raw_smoke_curve(
                decoder, z0, target, source=source, raw_lr=raw_lr
            )
            arm = arms[0]
            for row in raw_curves:
                all_curves.append({"vae_seed": vae_seed, "source_weight_index": source, "arm_id": arm.arm_id, "optimizer": arm.optimizer, "gauge": arm.gauge, "rcond_label": arm.rcond_label, "context_sha256": context_hash, "decoded_start_reference_sha256": _tensor_sha256(decoder.decode(z0)), "decoded_start_max_abs_error": 0.0, **row})
            for row in raw_diagnostics:
                all_diagnostics.append({"vae_seed": vae_seed, "source_weight_index": source, "arm_id": arm.arm_id, "context_sha256": context_hash, **row})
            for arm in (value for value in arms if value.family == "pure_gauge"):
                gauge = gauges[(arm.gauge, arm.rcond_label)]
                curves, diagnostics, occupancy = _pure_gauge_smoke_curve(
                    decoder, z0, target, radii, arm=arm, gauge=gauge, latent_lr=latent_lr
                )
                for row in occupancy:
                    all_occupancy.append({"vae_seed": vae_seed, "source_weight_index": source, "arm_id": arm.arm_id, "context_sha256": context_hash, **row})
                for row in curves:
                    all_curves.append({"vae_seed": vae_seed, "source_weight_index": source, "arm_id": arm.arm_id, "optimizer": arm.optimizer, "gauge": arm.gauge, "rcond_label": arm.rcond_label, "context_sha256": context_hash, **row})
                for row in diagnostics:
                    row.setdefault("batch_indices_sha256", _batch_hash(source, int(row["proposal_step"])))
                    all_diagnostics.append({"vae_seed": vae_seed, "source_weight_index": source, "arm_id": arm.arm_id, "context_sha256": context_hash, **row})
            local_curves, local_diagnostics = _coupled_local_smoke_curves(
                decoder,
                z0,
                target,
                radii,
                source=source,
            )
            arm_by_id = {value.arm_id: value for value in arms}
            for row in local_curves:
                arm = arm_by_id[str(row["arm_id"])]
                all_curves.append(
                    {
                        "vae_seed": vae_seed,
                        "source_weight_index": source,
                        "optimizer": arm.optimizer,
                        "gauge": arm.gauge,
                        "rcond_label": arm.rcond_label,
                        "context_sha256": context_hash,
                        **row,
                    }
                )
            for row in local_diagnostics:
                row.setdefault("batch_indices_sha256", _batch_hash(source, int(row["proposal_step"])))
                all_diagnostics.append(
                    {
                        "vae_seed": vae_seed,
                        "source_weight_index": source,
                        "context_sha256": context_hash,
                        **row,
                    }
                )
            if start_number % 4 == 0 or start_number == len(eval_bank):
                _log(f"progress vae_seed={vae_seed} eval_start={start_number}/16 elapsed_sec={time.perf_counter()-started:.2f}", verbose=verbose)
    curves = pd.DataFrame(all_curves)
    diagnostics = pd.DataFrame(all_diagnostics)
    occupancy = pd.DataFrame(all_occupancy)
    metric_states = pd.concat(all_metric_states, ignore_index=True)
    metric_trajectory = pd.concat(all_metric_trajectory, ignore_index=True)
    matrices = pd.DataFrame(matrix_rows)
    _log("stage=exact_grid_validation", verbose=verbose)
    dimensions = {
        "vae_seed": PRODUCTION_LABELS,
        "source_weight_index": PRODUCTION_EVAL_SOURCES,
        "arm_id": [arm.arm_id for arm in arms],
        "step": tuple(range(PRODUCTION_STEPS + 1)),
        "proposal_step": tuple(range(PRODUCTION_STEPS)),
        "occupancy_step": OCCUPANCY_STEPS,
    }
    curve_grid = _validate_exact_grid(curves, dimensions=dimensions, key_columns=("vae_seed", "source_weight_index", "arm_id", "step"), artifact_name="smoke curves")
    diagnostic_grid = _validate_exact_grid(diagnostics, dimensions=dimensions, key_columns=("vae_seed", "source_weight_index", "arm_id", "proposal_step"), artifact_name="smoke diagnostics")
    pure_arm_ids = [arm.arm_id for arm in arms if arm.family == "pure_gauge"]
    occupancy_grid = _validate_exact_grid(
        occupancy,
        dimensions={**dimensions, "arm_id": pure_arm_ids},
        key_columns=("vae_seed", "source_weight_index", "arm_id", "occupancy_step"),
        artifact_name="smoke occupancy",
    )
    coupled_normal_validation = _validate_coupled_normal_tangents(diagnostics)
    if diagnostics.groupby(["vae_seed", "source_weight_index", "proposal_step"])["batch_indices_sha256"].nunique().max() != 1:
        raise RuntimeError("same-start arms did not use the same deterministic batch stream")
    start_rows = curves[curves["step"] == 0]
    if float(start_rows["decoded_start_max_abs_error"].max()) > DECODED_START_ATOL:
        raise RuntimeError("literal gauge transform changed a decoded start beyond fp32 tolerance")
    per_start, per_seed, primary = _primary_aggregation(curves)
    artifact_hashes = {}
    for name, frame in (
        ("global_latent_gauge_curves.csv", curves),
        ("global_latent_gauge_step_diagnostics.csv", diagnostics),
        ("global_latent_gauge_occupancy.csv", occupancy),
        ("global_latent_gauge_metric_states.csv", metric_states),
        ("global_latent_gauge_metric_trajectory.csv", metric_trajectory),
        ("global_latent_gauge_matrices.csv", matrices),
        ("global_latent_gauge_primary_per_start.csv", per_start),
        ("global_latent_gauge_primary_per_seed.csv", per_seed),
        ("global_latent_gauge_primary.csv", primary),
    ):
        artifact_hashes[name] = _write_csv(frame, output_dir / name)
    for path in sorted(output_dir.glob("gauge_matrices_*.npz")):
        artifact_hashes[path.name] = _sha256_file(path)
    primary_id = "z_sgd:full:1e-5"
    identity_id = "z_sgd:identity:none"
    primary_values = primary.set_index("arm_id")["equal_weight_vae_seed_mean_progress"]
    validation = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "smoke",
        "device": "cpu",
        "dtype": "float32_jacobian_float64_gram",
        "seed": int(seed),
        "cache_mode": "disabled",
        "bank_hashes": dict(bank_hashes),
        "curve_grid": curve_grid,
        "diagnostic_grid": diagnostic_grid,
        "occupancy_grid": occupancy_grid,
        "coupled_normal_tangent": coupled_normal_validation,
        "arm_count": len(arms),
        "fit_state_rows": len(metric_states),
        "decoded_start_max_abs_error": float(start_rows["decoded_start_max_abs_error"].max()),
        "trust_match_rel_error_max": float(diagnostics["trust_match_rel_error"].max()),
        "trust_bracket_missing_count": int((~diagnostics["trust_bracket_found"].astype(bool)).sum()),
        "trust_boundary_hit_count": int(diagnostics["trust_boundary_hit"].astype(bool).sum()),
        "primary_contrast_full_whitened_z_sgd_minus_identity": float(primary_values[primary_id] - primary_values[identity_id]),
        "artifact_sha256": artifact_hashes,
    }
    validation["acceptance_checks"] = {
        "exact_banks": bank_hashes["evaluation_rows_sha256"] == PRODUCTION_EVAL_ROWS_SHA256 and bank_hashes["metric_fit_rows_sha256"] == PRODUCTION_FIT_ROWS_SHA256,
        "exact_grids": all(
            item["accepted"]
            for item in (curve_grid, diagnostic_grid, occupancy_grid, coupled_normal_validation)
        ),
        "decoded_starts": validation["decoded_start_max_abs_error"] <= DECODED_START_ATOL,
        "three_seed_hierarchy": bool(primary["n_vae_seeds"].eq(3).all()),
        "fresh_factorial_present": len(arms) == 20,
    }
    validation["acceptance_pass"] = bool(all(validation["acceptance_checks"].values()))
    validation_path = output_dir / "global_latent_gauge_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "smoke",
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH.resolve()),
        "request_hash": _stable_json_hash({"seed": seed, "bank_hashes": dict(bank_hashes), "arms": [arm.arm_id for arm in arms]}),
        "validation_sha256": _sha256_file(validation_path),
        "elapsed_sec": time.perf_counter() - started,
        "artifact_sha256": artifact_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"stage=output_writing acceptance_pass={validation['acceptance_pass']} curves={len(curves)} diagnostics={len(diagnostics)} output_dir={output_dir}", verbose=verbose)
    _log(f"summary primary_contrast={validation['primary_contrast_full_whitened_z_sgd_minus_identity']:.8g} manifest={manifest_path} validation={validation_path}", verbose=verbose)
    return validation


def _resolve_run(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() or candidate.exists() else (ARTIFACT_ROOT / candidate).resolve()


def _production_preflight(
    *,
    output_dir: Path,
    run_values: Sequence[str],
    labels: Sequence[str],
    bank_hashes: Mapping[str, str],
    verbose: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    if tuple(labels) != PRODUCTION_LABELS:
        raise ValueError(f"production labels must be exactly {PRODUCTION_LABELS}, got={tuple(labels)}")
    if len(run_values) != 3:
        raise ValueError("production preflight requires exactly three per-VAE runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [_resolve_run(value) for value in run_values]
    required_files = ("vae_checkpoint.pt", "config.json", "selected_lrs.csv", "weight_pool.pt", "weight_pool_records.csv")
    _log(f"stage=production_preflight device=none dtype=declared_fp32 seed=none cache_mode=read_only output_dir={output_dir}", verbose=verbose)
    rows = []
    for label, run_dir, expected_checkpoint in zip(labels, run_dirs, PRODUCTION_CHECKPOINT_SHA256, strict=True):
        missing = [name for name in required_files if not (run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"production run {run_dir} missing files: {missing}")
        _log(f"stage=input_hashing vae_seed={label} run={run_dir}", verbose=verbose)
        hashes = {name: _sha256_file(run_dir / name) for name in required_files}
        if hashes["vae_checkpoint.pt"] != expected_checkpoint:
            raise ValueError(f"checkpoint SHA256 mismatch for {label}: {hashes['vae_checkpoint.pt']}")
        rows.append(
            {
                "vae_seed": label,
                "run_dir": str(run_dir),
                "raw_lr": float(_selected_lr(run_dir, "raw")),
                "latent_lr": float(_selected_lr(run_dir, "decoder_latent")),
                **{name.replace(".", "_") + "_sha256": digest for name, digest in hashes.items()},
            }
        )
    frame = pd.DataFrame(rows)
    if frame["weight_pool_pt_sha256"].nunique() != 1 or frame["weight_pool_records_csv_sha256"].nunique() != 1:
        raise ValueError("production VAE seeds do not share exact weight-pool inputs")
    if frame["raw_lr"].nunique() != 1 or frame["latent_lr"].nunique() != 1:
        raise ValueError("production VAE seeds do not share frozen selected raw/latent LRs")
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "steps": PRODUCTION_STEPS,
        "metric_state_steps": METRIC_STATE_STEPS,
        "occupancy_steps": OCCUPANCY_STEPS,
        "rconds": RCONDS,
        "primary_rcond": PRIMARY_RCOND,
        "projector_rcond": PROJECTOR_RCOND,
        "labels": list(PRODUCTION_LABELS),
        "evaluation_sources": list(PRODUCTION_EVAL_SOURCES),
        "metric_fit_sources": list(PRODUCTION_FIT_SOURCES),
        "arms": [arm.__dict__ | {"rcond_label": arm.rcond_label} for arm in _arm_grid()],
        "bank_hashes": dict(bank_hashes),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH.resolve()),
        "run_inputs": frame.to_dict(orient="records"),
    }
    request_hash = _request_hash(request)
    context = {
        **request,
        "status": "PASS_PRODUCTION_PREFLIGHT",
        "production_execution_implemented": True,
        "production_execution_launched": False,
        "request": request,
        "request_hash": request_hash,
        "request_rehash_matches": bool(_request_hash(request) == request_hash),
        "elapsed_sec": time.perf_counter() - started,
    }
    inputs_path = output_dir / "global_latent_gauge_production_inputs.csv"
    _write_csv(frame, inputs_path)
    context["production_inputs_sha256"] = _sha256_file(inputs_path)
    preflight_path = output_dir / "global_latent_gauge_production_preflight.json"
    preflight_path.write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"stage=preflight_complete status={context['status']} request_hash={context['request_hash']} artifact={preflight_path}", verbose=verbose)
    return context


def _real_smoke_preflight(
    *,
    output_dir: Path,
    run_values: Sequence[str],
    labels: Sequence[str],
    fit_source: int,
    eval_source: int,
    bank_hashes: Mapping[str, str],
    device_name: str,
    jacobian_chunk_size: int,
    verbose: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    selection = _validate_real_smoke_selection(
        run_values=run_values,
        labels=labels,
        fit_source=int(fit_source),
        eval_source=int(eval_source),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _resolve_run(run_values[0])
    known_runs = {
        Path(name).name: (label, digest)
        for name, label, digest in zip(DEFAULT_RUNS, PRODUCTION_LABELS, PRODUCTION_CHECKPOINT_SHA256, strict=True)
    }
    if run_dir.name not in known_runs:
        raise ValueError(f"real_smoke run is not one of the three pinned control VAEs: {run_dir}")
    expected_label, expected_checkpoint = known_runs[run_dir.name]
    if str(labels[0]) != expected_label:
        raise ValueError(f"real_smoke label mismatch for {run_dir.name}: got={labels[0]} expected={expected_label}")
    required_files = ("vae_checkpoint.pt", "config.json", "selected_lrs.csv", "weight_pool.pt", "weight_pool_records.csv")
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"real_smoke run {run_dir} missing files: {missing}")
    _log(f"stage=real_smoke_input_hashing run={run_dir}", verbose=verbose)
    hashes = {name: _sha256_file(run_dir / name) for name in required_files}
    if hashes["vae_checkpoint.pt"] != expected_checkpoint:
        raise ValueError(f"real_smoke checkpoint SHA256 mismatch: {hashes['vae_checkpoint.pt']}")
    input_row = {
        "vae_seed": expected_label,
        "run_dir": str(run_dir),
        "fit_source": int(fit_source),
        "eval_source": int(eval_source),
        "raw_lr": float(_selected_lr(run_dir, "raw")),
        "latent_lr": float(_selected_lr(run_dir, "decoder_latent")),
        **{name.replace(".", "_") + "_sha256": digest for name, digest in hashes.items()},
    }
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "real_smoke",
        "non_production": True,
        "non_evidence": True,
        "production_acceptance_eligible": False,
        "steps": REAL_SMOKE_STEPS,
        "metric_state_steps": list(REAL_SMOKE_METRIC_STATE_STEPS),
        "occupancy_steps": list(REAL_SMOKE_OCCUPANCY_STEPS),
        "device": str(device_name),
        "jacobian_chunk_size": int(jacobian_chunk_size),
        "selection": selection,
        "bank_hashes": dict(bank_hashes),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH.resolve()),
        "run_input": input_row,
        "arms": [arm.arm_id for arm in _arm_grid()],
    }
    request_hash = _request_hash(request)
    input_path = output_dir / "global_latent_gauge_real_smoke_inputs.csv"
    _write_csv(pd.DataFrame([input_row]), input_path)
    context = {
        **request,
        "status": "PASS_REAL_SMOKE_PREFLIGHT",
        "request": request,
        "request_hash": request_hash,
        "request_rehash_matches": bool(_request_hash(request) == request_hash),
        "real_smoke_inputs_sha256": _sha256_file(input_path),
        "production_execution_launched": False,
        "elapsed_sec": time.perf_counter() - started,
    }
    path = output_dir / "global_latent_gauge_real_smoke_preflight.json"
    path.write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")
    _log(
        f"stage=real_smoke_preflight_complete request_hash={request_hash} artifact={path}",
        verbose=verbose,
    )
    return context


def _validate_context_hashes(
    frame: pd.DataFrame,
    bank: pd.DataFrame,
    *,
    labels: Sequence[str] = PRODUCTION_LABELS,
) -> dict[str, Any]:
    expected = {
        (label, int(row["source_weight_index"])): _context_hash(row)
        for label in labels
        for row in bank.to_dict(orient="records")
    }
    observed = frame[["vae_seed", "source_weight_index", "context_sha256"]].drop_duplicates()
    mismatches = 0
    for row in observed.itertuples(index=False):
        key = (str(row.vae_seed), int(row.source_weight_index))
        if key not in expected or str(row.context_sha256) != expected[key]:
            mismatches += 1
    if mismatches or len(observed) != len(expected):
        raise ValueError(
            f"evaluation context mismatch: observed={len(observed)} expected={len(expected)} mismatches={mismatches}"
        )
    return {"accepted": True, "contexts": len(observed)}


def _real_eval_seed(
    *,
    context: ProductionVAEContext,
    weights_cpu: torch.Tensor,
    eval_bank: pd.DataFrame,
    task_tensors: Mapping[str, Any],
    spec: Any,
    device: torch.device,
    batch_size: int,
    gauges: Mapping[tuple[str, str], Gauge],
    jacobian_chunk_size: int,
    progress_enabled: bool,
    steps: int = PRODUCTION_STEPS,
    occupancy_steps: Sequence[int] = OCCUPANCY_STEPS,
    counters: RuntimeCounters | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms = _arm_grid()
    curve_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    if not progress_enabled:
        context.cfg.show_progress = False
    progress = make_progress(context.cfg, total=len(eval_bank), desc=f"global gauge eval {context.label}")
    try:
        for start_number, row in enumerate(eval_bank.to_dict(orient="records"), start=1):
            source = int(row["source_weight_index"])
            position = int(row["start_bank_position"])
            stream_index = int(row.get("start_index", position))
            task_name = str(row["task_name"])
            tau = float(row["tau"])
            task_set = _task_tensor_set(dict(task_tensors), task_name)
            w0 = weights_cpu[source].to(device=device, dtype=torch.float32)
            z0, theta0 = _celo_literal_start(
                context.vae,
                context.normalizer,
                w0,
                counters=counters,
            )
            common = {
                "vae_seed": context.label,
                "label": context.label,
                "run_name": context.run_dir.name,
                "source_weight_index": source,
                "start_bank_position": position,
                "stream_start_index": stream_index,
                "task_name": task_name,
                "tau": tau,
                "raw_lr": context.raw_lr,
                "latent_lr": context.latent_lr,
                "theta0_sha256": _tensor_sha256(theta0),
                "z0_sha256": _tensor_sha256(z0),
                "context_sha256": _context_hash(row),
            }
            raw_curves, target_radii, raw_diagnostics = _raw_reference(
                theta0=theta0,
                raw_lr=float(context.raw_lr),
                steps=int(steps),
                eval_every=1,
                task_set=task_set,
                spec=spec,
                tau=tau,
                stream_start_index=stream_index,
                batch_size=int(batch_size),
                common=common,
            )
            if counters is not None:
                counters.full_split_eval_calls += int(steps) + 1
                counters.batch_gradient_calls += int(steps)
            normalized_curves, normalized_diagnostics = _normalize_raw_outputs(raw_curves, raw_diagnostics)
            curve_rows.extend(normalized_curves)
            diagnostic_rows.extend(normalized_diagnostics)
            for arm in (value for value in arms if value.family == "pure_gauge"):
                gauge = gauges[(arm.gauge, arm.rcond_label)]
                curves, diagnostics, occupancy = _real_gauge_curve(
                    context=context,
                    arm=arm,
                    gauge=gauge,
                    z0=z0,
                    theta0=theta0,
                    target_radii=target_radii,
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    stream_start_index=stream_index,
                    batch_size=int(batch_size),
                    jacobian_chunk_size=int(jacobian_chunk_size),
                    common=common,
                    steps=int(steps),
                    occupancy_steps=occupancy_steps,
                    counters=counters,
                )
                curve_rows.extend(curves)
                diagnostic_rows.extend(diagnostics)
                occupancy_rows.extend(occupancy)
            local_curves, local_diagnostics = _real_coupled_local_curves(
                context=context,
                z0=z0,
                theta0=theta0,
                target_radii=target_radii,
                task_set=task_set,
                spec=spec,
                tau=tau,
                stream_start_index=stream_index,
                batch_size=int(batch_size),
                jacobian_chunk_size=int(jacobian_chunk_size),
                common=common,
                placebo_seed=_stable_seed("production_placebo", context.label),
                steps=int(steps),
                counters=counters,
            )
            curve_rows.extend(local_curves)
            diagnostic_rows.extend(local_diagnostics)
            progress.set_postfix(
                {
                    "source": source,
                    "start": f"{start_number}/{len(eval_bank)}",
                    "arms": len(arms),
                }
            )
            progress.update(1)
    finally:
        progress.close()
    return pd.DataFrame(curve_rows), pd.DataFrame(diagnostic_rows), pd.DataFrame(occupancy_rows)


def _run_real_smoke(
    *,
    output_dir: Path,
    run_value: str,
    label: str,
    fit_bank: pd.DataFrame,
    eval_bank: pd.DataFrame,
    bank_hashes: Mapping[str, str],
    preflight: Mapping[str, Any],
    device_name: str,
    jacobian_chunk_size: int,
    seed: int,
    verbose: bool,
    allow_cpu_for_tests: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(device_name)
    if device.type != "cuda" and not allow_cpu_for_tests:
        raise ValueError(f"real_smoke requires a CUDA device, got={device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("real_smoke requires an available CUDA runtime")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = RuntimeCounters()
    stage_runtime: dict[str, float] = {}
    memory = {"startup": _memory_snapshot(device)}
    run_dir = _resolve_run(run_value)
    _log(
        f"stage=real_smoke_load run={run_dir} label={label} device={device} dtype=float32 "
        f"steps={REAL_SMOKE_STEPS} fit_starts=1 eval_starts=1 arms={len(_arm_grid())}",
        verbose=verbose,
    )
    stage_started = time.perf_counter()
    cfg = _load_cfg(run_dir, device=str(device), downstream_steps=REAL_SMOKE_STEPS)
    dtype = torch_dtype(cfg)
    if dtype != torch.float32:
        raise RuntimeError(f"real_smoke requires float32 model execution, got={dtype}")
    weights_cpu, weight_records, weight_key, spec = _load_weight_pool(run_dir)
    task_tensors = _load_task_tensors_for_pipeline(cfg, device=device, dtype=dtype)
    vae, normalizer, checkpoint_payload = _load_vae(
        run_dir,
        cfg,
        int(weights_cpu.shape[1]),
        device=device,
        dtype=dtype,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_runtime["load_inputs_model_sec"] = time.perf_counter() - stage_started
    memory["after_load"] = _memory_snapshot(device)
    task_fingerprint = _task_tensors_fingerprint(dict(task_tensors))
    detailed_task_hashes = _task_hashes(task_tensors)
    context = ProductionVAEContext(
        label=str(label),
        run_dir=run_dir,
        cfg=cfg,
        vae=vae,
        normalizer=normalizer,
        checkpoint_payload=checkpoint_payload,
        raw_lr=float(_selected_lr(run_dir, "raw")),
        latent_lr=float(_selected_lr(run_dir, "decoder_latent")),
        checkpoint_sha256=_sha256_file(run_dir / "vae_checkpoint.pt"),
        config_sha256=_sha256_file(run_dir / "config.json"),
        selected_lrs_sha256=_sha256_file(run_dir / "selected_lrs.csv"),
        normalizer_sha256=_tensor_mapping_sha256(checkpoint_payload["normalizer"]),
    )
    execution_request = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "real_smoke",
        "non_production": True,
        "non_evidence": True,
        "production_acceptance_eligible": False,
        "preflight_request_hash": str(preflight["request_hash"]),
        "run": str(run_dir),
        "label": str(label),
        "fit_source": int(fit_bank.iloc[0]["source_weight_index"]),
        "eval_source": int(eval_bank.iloc[0]["source_weight_index"]),
        "steps": REAL_SMOKE_STEPS,
        "metric_state_steps": list(REAL_SMOKE_METRIC_STATE_STEPS),
        "occupancy_steps": list(REAL_SMOKE_OCCUPANCY_STEPS),
        "device": str(device),
        "dtype": str(dtype),
        "seed": int(seed),
        "jacobian_chunk_size": int(jacobian_chunk_size),
        "task_tensor_fingerprint": task_fingerprint,
        "task_tensor_hashes": detailed_task_hashes,
        "weight_pool_cache_key": weight_key,
        "weight_pool_records_rows": int(len(weight_records)),
        "normalizer_sha256": context.normalizer_sha256,
        "arms": [arm.arm_id for arm in _arm_grid()],
        "contrasts": [item[0] for item in CONTRAST_SPECS],
        "bank_hashes": dict(bank_hashes),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH.resolve()),
    }
    request_hash = _request_hash(execution_request)
    _log(
        f"stage=real_smoke_metric_fit source={execution_request['fit_source']} "
        f"checkpoints={REAL_SMOKE_METRIC_STATE_STEPS}",
        verbose=verbose,
    )
    stage_started = time.perf_counter()
    metric, gauges, metric_states, metric_trajectory, arrays = _real_metric_fit(
        context=context,
        weights_cpu=weights_cpu,
        fit_bank=fit_bank,
        eval_bank=eval_bank,
        task_tensors=task_tensors,
        task_hashes=detailed_task_hashes,
        spec=spec,
        device=device,
        batch_size=int(cfg.downstream_batch_size),
        jacobian_chunk_size=int(jacobian_chunk_size),
        progress_enabled=verbose,
        steps=REAL_SMOKE_STEPS,
        metric_state_steps=REAL_SMOKE_METRIC_STATE_STEPS,
        expected_fit_sources=(execution_request["fit_source"],),
        evaluation_sources=(execution_request["eval_source"],),
        counters=counters,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_runtime["metric_fit_sec"] = time.perf_counter() - stage_started
    memory["after_metric_fit"] = _memory_snapshot(device)
    matrix_path = output_dir / f"global_latent_gauge_real_smoke_matrices_{label}.npz"
    np.savez(matrix_path, **arrays)
    matrices = _matrix_metadata(
        vae_seed=label,
        metric=metric,
        gauges=gauges,
        matrix_file=matrix_path,
    )
    matrices["matrix_file_sha256"] = _sha256_file(matrix_path)
    _log(
        f"stage=real_smoke_evaluation source={execution_request['eval_source']} "
        f"arms={len(_arm_grid())} steps={REAL_SMOKE_STEPS}",
        verbose=verbose,
    )
    stage_started = time.perf_counter()
    curves, diagnostics, occupancy = _real_eval_seed(
        context=context,
        weights_cpu=weights_cpu,
        eval_bank=eval_bank,
        task_tensors=task_tensors,
        spec=spec,
        device=device,
        batch_size=int(cfg.downstream_batch_size),
        gauges=gauges,
        jacobian_chunk_size=int(jacobian_chunk_size),
        progress_enabled=verbose,
        steps=REAL_SMOKE_STEPS,
        occupancy_steps=REAL_SMOKE_OCCUPANCY_STEPS,
        counters=counters,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_runtime["evaluation_sec"] = time.perf_counter() - stage_started
    memory["after_evaluation"] = _memory_snapshot(device)
    curves, diagnostics, execution_flags = _propagate_execution_flags(curves, diagnostics)
    per_start, per_seed, primary = _primary_with_execution(
        curves,
        execution_flags,
        steps=REAL_SMOKE_STEPS,
        expected_vae_seeds=(label,),
    )
    paired_starts, paired_seeds, paired_equal = _paired_contrast_outputs(
        per_start,
        expected_vae_seeds=(label,),
        expected_sources=(execution_request["eval_source"],),
    )
    arms = _arm_grid()
    pure_arm_ids = [arm.arm_id for arm in arms if arm.family == "pure_gauge"]
    gauge_ids = [f"{name}:{cutoff}" for name, cutoff in gauges]
    dimensions = {
        "vae_seed": (label,),
        "source_weight_index": (execution_request["eval_source"],),
        "arm_id": [arm.arm_id for arm in arms],
        "step": tuple(range(REAL_SMOKE_STEPS + 1)),
        "proposal_step": tuple(range(REAL_SMOKE_STEPS)),
        "occupancy_step": REAL_SMOKE_OCCUPANCY_STEPS,
    }
    validations = {
        "metric_states": _validate_exact_grid(
            metric_states,
            dimensions={
                "vae_seed": (label,),
                "source_weight_index": (execution_request["fit_source"],),
                "metric_state_step": REAL_SMOKE_METRIC_STATE_STEPS,
            },
            key_columns=("vae_seed", "source_weight_index", "metric_state_step"),
            artifact_name="real_smoke metric states",
        ),
        "metric_trajectory": _validate_exact_grid(
            metric_trajectory,
            dimensions={
                "vae_seed": (label,),
                "source_weight_index": (execution_request["fit_source"],),
                "optimizer_step": tuple(range(REAL_SMOKE_STEPS)),
            },
            key_columns=("vae_seed", "source_weight_index", "optimizer_step"),
            artifact_name="real_smoke metric trajectory",
        ),
        "matrices": _validate_exact_grid(
            matrices.assign(gauge_id=matrices["gauge"] + ":" + matrices["rcond_label"]),
            dimensions={"vae_seed": (label,), "gauge_id": gauge_ids},
            key_columns=("vae_seed", "gauge_id"),
            artifact_name="real_smoke matrices",
        ),
        "curves": _validate_exact_grid(
            curves,
            dimensions=dimensions,
            key_columns=("vae_seed", "source_weight_index", "arm_id", "step"),
            artifact_name="real_smoke curves",
        ),
        "diagnostics": _validate_exact_grid(
            diagnostics,
            dimensions=dimensions,
            key_columns=("vae_seed", "source_weight_index", "arm_id", "proposal_step"),
            artifact_name="real_smoke diagnostics",
        ),
        "occupancy": _validate_exact_grid(
            occupancy,
            dimensions={**dimensions, "arm_id": pure_arm_ids},
            key_columns=("vae_seed", "source_weight_index", "arm_id", "occupancy_step"),
            artifact_name="real_smoke occupancy",
        ),
        "paired_starts": _validate_exact_grid(
            paired_starts,
            dimensions={
                "contrast": [item[0] for item in CONTRAST_SPECS],
                "vae_seed": (label,),
                "source_weight_index": (execution_request["eval_source"],),
            },
            key_columns=("contrast", "vae_seed", "source_weight_index"),
            artifact_name="real_smoke paired starts",
        ),
        "paired_seeds": _validate_exact_grid(
            paired_seeds,
            dimensions={
                "contrast": [item[0] for item in CONTRAST_SPECS],
                "analysis_set": ("all_start_itt", "executable_sensitivity"),
                "vae_seed": (label,),
            },
            key_columns=("contrast", "analysis_set", "vae_seed"),
            artifact_name="real_smoke paired seeds",
        ),
        "paired_equal_seed": _validate_exact_grid(
            paired_equal,
            dimensions={
                "contrast": [item[0] for item in CONTRAST_SPECS],
                "analysis_set": ("all_start_itt", "executable_sensitivity"),
            },
            key_columns=("contrast", "analysis_set"),
            artifact_name="real_smoke equal-seed contrasts",
        ),
        "same_batches": _validate_same_batches(diagnostics, expected_arm_ids=[arm.arm_id for arm in arms]),
        "coupled_normal_pair": _validate_coupled_normal_tangents(diagnostics),
        "contexts": _validate_context_hashes(curves, eval_bank, labels=(label,)),
    }
    initial = curves[curves["step"] == 0]
    initial_hashes = initial.groupby(["vae_seed", "source_weight_index"])["state_theta_sha256"].nunique()
    initial_declared = initial["state_theta_sha256"].eq(initial["decoded_start_reference_sha256"])
    if int(initial_hashes.max()) != 1 or not bool(initial_declared.all()):
        raise RuntimeError("real_smoke literal decoded starts differ")
    if set(metric_states["source_weight_index"].astype(int)) & set(eval_bank["source_weight_index"].astype(int)):
        raise RuntimeError("real_smoke evaluation source leaked into metric fitting")
    stage_runtime["total_before_writing_sec"] = time.perf_counter() - started
    memory["before_writing"] = _memory_snapshot(device)
    runtime_payload = {
        "protocol_mode": "real_smoke",
        "non_production": True,
        "non_evidence": True,
        "stage_runtime_sec": stage_runtime,
        "counters": asdict(counters),
        "decoder_forward_count_scope": "explicit decode API calls; Jacobian directional work reported separately",
        "memory": memory,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu_test_override",
        "cpu_test_override": bool(allow_cpu_for_tests),
    }
    output_frames = {
        "global_latent_gauge_metric_states.csv": metric_states,
        "global_latent_gauge_metric_trajectory.csv": metric_trajectory,
        "global_latent_gauge_matrices.csv": matrices,
        "global_latent_gauge_curves.csv": curves,
        "global_latent_gauge_step_diagnostics.csv": diagnostics,
        "global_latent_gauge_occupancy.csv": occupancy,
        "global_latent_gauge_execution_flags.csv": execution_flags,
        "global_latent_gauge_primary_per_start.csv": per_start,
        "global_latent_gauge_primary_per_seed.csv": per_seed,
        "global_latent_gauge_primary.csv": primary,
        "global_latent_gauge_paired_per_start.csv": paired_starts,
        "global_latent_gauge_paired_per_seed.csv": paired_seeds,
        "global_latent_gauge_paired_equal_seed.csv": paired_equal,
    }
    output_write_started = time.perf_counter()
    eval_bank_path = output_dir / "global_latent_gauge_real_smoke_eval_bank.csv"
    fit_bank_path = output_dir / "global_latent_gauge_real_smoke_metric_fit_bank.csv"
    eval_bank.to_csv(eval_bank_path, index=False)
    fit_bank.to_csv(fit_bank_path, index=False)
    artifact_hashes = {
        eval_bank_path.name: _sha256_file(eval_bank_path),
        fit_bank_path.name: _sha256_file(fit_bank_path),
        matrix_path.name: _sha256_file(matrix_path),
    }
    for name, frame in output_frames.items():
        artifact_hashes[name] = _write_csv(frame, output_dir / name)
    stage_runtime["output_frames_sec"] = time.perf_counter() - output_write_started
    memory["after_output_frames"] = _memory_snapshot(device)
    runtime_path = output_dir / "global_latent_gauge_real_smoke_runtime.json"
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True), encoding="utf-8")
    artifact_hashes[runtime_path.name] = _sha256_file(runtime_path)
    validation = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "real_smoke",
        "non_production": True,
        "non_evidence": True,
        "production_acceptance_eligible": False,
        "production_acceptance_pass": False,
        "request_hash": request_hash,
        "request_rehash_matches": bool(_request_hash(execution_request) == request_hash),
        "validations": validations,
        "runtime": runtime_payload,
        "trust_failure_step_count": int((~diagnostics["trust_executable"].astype(bool)).sum()),
        "trust_failure_trajectory_count": int((~execution_flags["trajectory_trust_executable"].astype(bool)).sum()),
        "artifact_sha256": artifact_hashes,
    }
    validation["real_smoke_acceptance_checks"] = {
        "request_hash": validation["request_rehash_matches"],
        "exact_limited_grids": all(item["accepted"] for item in validations.values()),
        "literal_same_start": int(initial_hashes.max()) == 1 and bool(initial_declared.all()),
        "fit_eval_disjoint": True,
        "all_20_arms": curves["arm_id"].nunique() == 20,
        "artifact_hashes": _verify_artifact_hashes(output_dir, artifact_hashes)["accepted"],
    }
    validation["real_smoke_acceptance_pass"] = bool(all(validation["real_smoke_acceptance_checks"].values()))
    validation_path = output_dir / "global_latent_gauge_real_smoke_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "real_smoke",
        "non_production": True,
        "non_evidence": True,
        "production_acceptance_eligible": False,
        "production_acceptance_pass": False,
        "request": execution_request,
        "request_hash": request_hash,
        "validation_sha256": _sha256_file(validation_path),
        "artifact_sha256": artifact_hashes,
        "elapsed_sec": time.perf_counter() - started,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    hash_files = {
        **artifact_hashes,
        validation_path.name: _sha256_file(validation_path),
        manifest_path.name: _sha256_file(manifest_path),
    }
    for name in ("global_latent_gauge_real_smoke_preflight.json", "global_latent_gauge_real_smoke_inputs.csv"):
        path = output_dir / name
        if path.is_file():
            hash_files[name] = _sha256_file(path)
    hash_index_path = _write_hash_index(output_dir, file_hashes=hash_files)
    _log(
        f"done mode=real_smoke non_evidence=true production_acceptance_eligible=false "
        f"real_smoke_acceptance_pass={validation['real_smoke_acceptance_pass']} "
        f"curves={len(curves)} diagnostics={len(diagnostics)} jacobians={counters.jacobian_calls} "
        f"decoder_forwards={counters.decoder_forward_calls} elapsed_sec={time.perf_counter()-started:.2f}",
        verbose=verbose,
    )
    _log(
        f"artifacts manifest={manifest_path} validation={validation_path} runtime={runtime_path} "
        f"hash_index={hash_index_path}",
        verbose=verbose,
    )
    return validation


def _run_production(
    *,
    output_dir: Path,
    run_values: Sequence[str],
    labels: Sequence[str],
    eval_bank: pd.DataFrame,
    fit_bank: pd.DataFrame,
    bank_hashes: Mapping[str, str],
    preflight: Mapping[str, Any],
    device_name: str,
    jacobian_chunk_size: int,
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [_resolve_run(value) for value in run_values]
    _log(
        f"stage=production_load device={device_name} dtype=float32 seed={seed} cache_mode=read_only "
        f"jacobian_chunk_size={jacobian_chunk_size} output_dir={output_dir}",
        verbose=verbose,
    )
    base_cfg = _load_cfg(run_dirs[0], device=str(device_name), downstream_steps=PRODUCTION_STEPS)
    device = torch.device(base_cfg.device)
    dtype = torch_dtype(base_cfg)
    if dtype != torch.float32:
        raise RuntimeError(f"production protocol requires float32 model execution, got={dtype}")
    weights_cpu, weight_records, weight_key, spec = _load_weight_pool(run_dirs[0])
    task_tensors = _load_task_tensors_for_pipeline(base_cfg, device=device, dtype=dtype)
    task_fingerprint = _task_tensors_fingerprint(dict(task_tensors))
    detailed_task_hashes = _task_hashes(task_tensors)
    batch_size = int(base_cfg.downstream_batch_size)
    for run_dir in run_dirs[1:]:
        other_cfg = _load_cfg(run_dir, device=str(device_name), downstream_steps=PRODUCTION_STEPS)
        other_tasks = _load_task_tensors_for_pipeline(other_cfg, device=device, dtype=dtype)
        if _task_tensors_fingerprint(other_tasks) != task_fingerprint:
            raise RuntimeError(f"real production task tensor mismatch for {run_dir}")
        if int(other_cfg.downstream_batch_size) != batch_size:
            raise RuntimeError(f"real production batch-size mismatch for {run_dir}")
        del other_tasks
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "production",
        "device": str(device),
        "dtype": str(dtype),
        "seed": int(seed),
        "cache_mode": "read_only",
        "steps": PRODUCTION_STEPS,
        "metric_state_steps": list(METRIC_STATE_STEPS),
        "occupancy_steps": list(OCCUPANCY_STEPS),
        "jacobian_chunk_size": int(jacobian_chunk_size),
        "bank_hashes": dict(bank_hashes),
        "preflight_request_hash": str(preflight["request_hash"]),
        "task_tensor_fingerprint": task_fingerprint,
        "task_tensor_hashes": detailed_task_hashes,
        "weight_pool_cache_key": weight_key,
        "weight_pool_records_rows": int(len(weight_records)),
        "arms": [arm.arm_id for arm in _arm_grid()],
        "contrasts": [specification[0] for specification in CONTRAST_SPECS],
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH.resolve()),
    }
    request_hash = _request_hash(request)
    all_metric_states = []
    all_metric_trajectories = []
    all_matrices = []
    all_curves = []
    all_diagnostics = []
    all_occupancy = []
    matrix_paths: list[Path] = []
    provenance_rows = []
    preflight_inputs = {
        str(row["vae_seed"]): row
        for row in preflight.get("run_inputs", [])
    }
    for label, run_dir, checkpoint_hash in zip(labels, run_dirs, PRODUCTION_CHECKPOINT_SHA256, strict=True):
        _log(f"stage=vae_load vae_seed={label} run={run_dir}", verbose=verbose)
        cfg = _load_cfg(run_dir, device=str(device_name), downstream_steps=PRODUCTION_STEPS)
        vae, normalizer, checkpoint_payload = _load_vae(
            run_dir,
            cfg,
            int(weights_cpu.shape[1]),
            device=device,
            dtype=dtype,
        )
        context = ProductionVAEContext(
            label=str(label),
            run_dir=run_dir,
            cfg=cfg,
            vae=vae,
            normalizer=normalizer,
            checkpoint_payload=checkpoint_payload,
            raw_lr=float(_selected_lr(run_dir, "raw")),
            latent_lr=float(_selected_lr(run_dir, "decoder_latent")),
            checkpoint_sha256=checkpoint_hash,
            config_sha256=_sha256_file(run_dir / "config.json"),
            selected_lrs_sha256=_sha256_file(run_dir / "selected_lrs.csv"),
            normalizer_sha256=_tensor_mapping_sha256(checkpoint_payload["normalizer"]),
        )
        provenance_rows.append(
            {
                "vae_seed": label,
                "run_dir": str(run_dir),
                "checkpoint_sha256": context.checkpoint_sha256,
                "config_sha256": context.config_sha256,
                "selected_lrs_sha256": context.selected_lrs_sha256,
                "normalizer_sha256": context.normalizer_sha256,
                "raw_lr": context.raw_lr,
                "latent_lr": context.latent_lr,
                "weight_pool_sha256": str(preflight_inputs[label]["weight_pool_pt_sha256"]),
                "weight_pool_records_sha256": str(preflight_inputs[label]["weight_pool_records_csv_sha256"]),
                "task_tensor_fingerprint": task_fingerprint,
            }
        )
        _log(
            f"stage=metric_fit vae_seed={label} starts={len(fit_bank)} checkpoints={METRIC_STATE_STEPS} "
            f"latent_lr={context.latent_lr}",
            verbose=verbose,
        )
        metric, gauges, metric_states, metric_trajectory, arrays = _real_metric_fit(
            context=context,
            weights_cpu=weights_cpu,
            fit_bank=fit_bank,
            eval_bank=eval_bank,
            task_tensors=task_tensors,
            task_hashes=detailed_task_hashes,
            spec=spec,
            device=device,
            batch_size=batch_size,
            jacobian_chunk_size=int(jacobian_chunk_size),
            progress_enabled=verbose,
        )
        matrix_path = output_dir / f"global_latent_gauge_matrices_{label}.npz"
        np.savez(matrix_path, **arrays)
        matrix_paths.append(matrix_path)
        matrix_metadata = _matrix_metadata(
            vae_seed=label,
            metric=metric,
            gauges=gauges,
            matrix_file=matrix_path,
        )
        matrix_metadata["matrix_file_sha256"] = _sha256_file(matrix_path)
        all_metric_states.append(metric_states)
        all_metric_trajectories.append(metric_trajectory)
        all_matrices.append(matrix_metadata)
        _log(
            f"stage=evaluation vae_seed={label} starts={len(eval_bank)} arms={len(_arm_grid())} "
            f"steps={PRODUCTION_STEPS} raw_lr={context.raw_lr}",
            verbose=verbose,
        )
        curves, diagnostics, occupancy = _real_eval_seed(
            context=context,
            weights_cpu=weights_cpu,
            eval_bank=eval_bank,
            task_tensors=task_tensors,
            spec=spec,
            device=device,
            batch_size=batch_size,
            gauges=gauges,
            jacobian_chunk_size=int(jacobian_chunk_size),
            progress_enabled=verbose,
        )
        all_curves.append(curves)
        all_diagnostics.append(diagnostics)
        all_occupancy.append(occupancy)
        del vae, normalizer, checkpoint_payload, metric, gauges, arrays
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metric_states = pd.concat(all_metric_states, ignore_index=True)
    metric_trajectory = pd.concat(all_metric_trajectories, ignore_index=True)
    matrices = pd.concat(all_matrices, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    occupancy = pd.concat(all_occupancy, ignore_index=True)
    curves, diagnostics, execution_flags = _propagate_execution_flags(curves, diagnostics)
    per_start, per_seed, primary = _primary_with_execution(curves, execution_flags)
    paired_starts, paired_seeds, paired_equal = _paired_contrast_outputs(per_start)
    _log("stage=exact_grid_validation", verbose=verbose)
    arms = _arm_grid()
    pure_arm_ids = [arm.arm_id for arm in arms if arm.family == "pure_gauge"]
    dimensions = {
        "vae_seed": PRODUCTION_LABELS,
        "source_weight_index": PRODUCTION_EVAL_SOURCES,
        "arm_id": [arm.arm_id for arm in arms],
        "step": tuple(range(PRODUCTION_STEPS + 1)),
        "proposal_step": tuple(range(PRODUCTION_STEPS)),
        "occupancy_step": OCCUPANCY_STEPS,
    }
    validations = {
        "metric_states": _validate_exact_grid(
            metric_states,
            dimensions={"vae_seed": PRODUCTION_LABELS, "source_weight_index": PRODUCTION_FIT_SOURCES, "metric_state_step": METRIC_STATE_STEPS},
            key_columns=("vae_seed", "source_weight_index", "metric_state_step"),
            artifact_name="production metric states",
        ),
        "metric_trajectory": _validate_exact_grid(
            metric_trajectory,
            dimensions={"vae_seed": PRODUCTION_LABELS, "source_weight_index": PRODUCTION_FIT_SOURCES, "optimizer_step": tuple(range(PRODUCTION_STEPS))},
            key_columns=("vae_seed", "source_weight_index", "optimizer_step"),
            artifact_name="production metric trajectories",
        ),
        "matrices": _validate_exact_grid(
            matrices.assign(gauge_id=matrices["gauge"] + ":" + matrices["rcond_label"]),
            dimensions={"vae_seed": PRODUCTION_LABELS, "gauge_id": [f"{name}:{cutoff}" for name, cutoff in _gauges_for_metric(torch.eye(2, dtype=torch.float64), seed=1)]},
            key_columns=("vae_seed", "gauge_id"),
            artifact_name="production matrices",
        ),
        "curves": _validate_exact_grid(
            curves,
            dimensions=dimensions,
            key_columns=("vae_seed", "source_weight_index", "arm_id", "step"),
            artifact_name="production curves",
        ),
        "diagnostics": _validate_exact_grid(
            diagnostics,
            dimensions=dimensions,
            key_columns=("vae_seed", "source_weight_index", "arm_id", "proposal_step"),
            artifact_name="production diagnostics",
        ),
        "occupancy": _validate_exact_grid(
            occupancy,
            dimensions={**dimensions, "arm_id": pure_arm_ids},
            key_columns=("vae_seed", "source_weight_index", "arm_id", "occupancy_step"),
            artifact_name="production occupancy",
        ),
        "paired_starts": _validate_exact_grid(
            paired_starts,
            dimensions={"contrast": [item[0] for item in CONTRAST_SPECS], "vae_seed": PRODUCTION_LABELS, "source_weight_index": PRODUCTION_EVAL_SOURCES},
            key_columns=("contrast", "vae_seed", "source_weight_index"),
            artifact_name="production paired starts",
        ),
        "paired_seeds": _validate_exact_grid(
            paired_seeds,
            dimensions={"contrast": [item[0] for item in CONTRAST_SPECS], "analysis_set": ("all_start_itt", "executable_sensitivity"), "vae_seed": PRODUCTION_LABELS},
            key_columns=("contrast", "analysis_set", "vae_seed"),
            artifact_name="production paired seeds",
        ),
        "paired_equal_seed": _validate_exact_grid(
            paired_equal,
            dimensions={"contrast": [item[0] for item in CONTRAST_SPECS], "analysis_set": ("all_start_itt", "executable_sensitivity")},
            key_columns=("contrast", "analysis_set"),
            artifact_name="production equal-seed contrasts",
        ),
        "same_batches": _validate_same_batches(diagnostics, expected_arm_ids=[arm.arm_id for arm in arms]),
        "coupled_normal_tangent": _validate_coupled_normal_tangents(diagnostics),
        "contexts": _validate_context_hashes(curves, eval_bank),
    }
    initial = curves[curves["step"] == 0]
    initial_hash_counts = initial.groupby(["vae_seed", "source_weight_index"])["state_theta_sha256"].nunique()
    initial_declared = initial["state_theta_sha256"].eq(initial["decoded_start_reference_sha256"])
    if int(initial_hash_counts.max()) != 1 or not bool(initial_declared.all()):
        raise RuntimeError(
            f"literal decoded starts differ: max_hash_count={int(initial_hash_counts.max())} "
            f"declared_mismatch={int((~initial_declared).sum())}"
        )
    if set(metric_states["source_weight_index"].astype(int)) & set(PRODUCTION_EVAL_SOURCES):
        raise RuntimeError("evaluation source leaked into stored metric states")
    identity_sgd = curves[curves["arm_id"] == "z_sgd:identity:none"]
    orthogonal_sgd = curves[curves["arm_id"] == "z_sgd:orthogonal:none"]
    orthogonal_pair = identity_sgd.merge(
        orthogonal_sgd,
        on=["vae_seed", "source_weight_index", "step"],
        suffixes=("_identity", "_orthogonal"),
        validate="one_to_one",
    )
    orthogonal_execution = {
        "paired_rows": int(len(orthogonal_pair)),
        "theta_hash_mismatch_count": int(
            (~orthogonal_pair["state_theta_sha256_identity"].eq(orthogonal_pair["state_theta_sha256_orthogonal"])).sum()
        ),
        "train_loss_abs_discrepancy_max": float(
            (orthogonal_pair["train_loss_identity"] - orthogonal_pair["train_loss_orthogonal"]).abs().max()
        ),
        "test_loss_abs_discrepancy_max": float(
            (orthogonal_pair["test_loss_identity"] - orthogonal_pair["test_loss_orthogonal"]).abs().max()
        ),
    }
    _log("stage=output_writing", verbose=verbose)
    eval_bank.to_csv(output_dir / "global_latent_gauge_eval_bank.csv", index=False)
    fit_bank.to_csv(output_dir / "global_latent_gauge_metric_fit_bank.csv", index=False)
    output_frames = {
        "global_latent_gauge_metric_states.csv": metric_states,
        "global_latent_gauge_metric_trajectory.csv": metric_trajectory,
        "global_latent_gauge_matrices.csv": matrices,
        "global_latent_gauge_curves.csv": curves,
        "global_latent_gauge_step_diagnostics.csv": diagnostics,
        "global_latent_gauge_occupancy.csv": occupancy,
        "global_latent_gauge_execution_flags.csv": execution_flags,
        "global_latent_gauge_primary_per_start.csv": per_start,
        "global_latent_gauge_primary_per_seed.csv": per_seed,
        "global_latent_gauge_primary.csv": primary,
        "global_latent_gauge_paired_per_start.csv": paired_starts,
        "global_latent_gauge_paired_per_seed.csv": paired_seeds,
        "global_latent_gauge_paired_equal_seed.csv": paired_equal,
        "global_latent_gauge_provenance.csv": pd.DataFrame(provenance_rows),
    }
    artifact_hashes = {
        "global_latent_gauge_eval_bank.csv": _sha256_file(output_dir / "global_latent_gauge_eval_bank.csv"),
        "global_latent_gauge_metric_fit_bank.csv": _sha256_file(output_dir / "global_latent_gauge_metric_fit_bank.csv"),
    }
    for name, frame in output_frames.items():
        artifact_hashes[name] = _write_csv(frame, output_dir / name)
    for path in matrix_paths:
        artifact_hashes[path.name] = _sha256_file(path)
    validation = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "production",
        "request_hash": request_hash,
        "request_rehash_matches": bool(_request_hash(request) == request_hash),
        "validations": validations,
        "orthogonal_sgd_execution_discrepancy": orthogonal_execution,
        "decoded_start_hash_max_unique": int(initial_hash_counts.max()),
        "decoded_start_hash_mismatch_count": int((~initial_declared).sum()),
        "trust_failure_step_count": int((~diagnostics["trust_executable"].astype(bool)).sum()),
        "trust_failure_trajectory_count": int((~execution_flags["trajectory_trust_executable"].astype(bool)).sum()),
        "treatment_executable_trajectory_count": int(execution_flags["trajectory_treatment_executable"].astype(bool).sum()),
        "artifact_sha256": artifact_hashes,
    }
    validation["acceptance_checks"] = {
        "request_hash": validation["request_rehash_matches"],
        "exact_grids": all(item["accepted"] for item in validations.values()),
        "literal_same_start": validation["decoded_start_hash_mismatch_count"] == 0,
        "no_metric_eval_leakage": True,
        "artifact_hashes": _verify_artifact_hashes(output_dir, artifact_hashes)["accepted"],
    }
    validation["acceptance_pass"] = bool(all(validation["acceptance_checks"].values()))
    validation_path = output_dir / "global_latent_gauge_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": "production",
        "request": request,
        "request_hash": request_hash,
        "validation_sha256": _sha256_file(validation_path),
        "artifact_sha256": artifact_hashes,
        "elapsed_sec": time.perf_counter() - started,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    hash_index = {
        **artifact_hashes,
        validation_path.name: _sha256_file(validation_path),
        manifest_path.name: _sha256_file(manifest_path),
    }
    preflight_path = output_dir / "global_latent_gauge_production_preflight.json"
    inputs_path = output_dir / "global_latent_gauge_production_inputs.csv"
    for path in (preflight_path, inputs_path):
        if path.is_file():
            hash_index[path.name] = _sha256_file(path)
    hash_index_path = _write_hash_index(output_dir, file_hashes=hash_index)
    _log(
        f"done acceptance_pass={validation['acceptance_pass']} curves={len(curves)} "
        f"diagnostics={len(diagnostics)} metric_states={len(metric_states)} "
        f"trust_failure_trajectories={validation['trust_failure_trajectory_count']} "
        f"elapsed_sec={time.perf_counter()-started:.2f} output_dir={output_dir}",
        verbose=verbose,
    )
    _log(
        f"artifacts manifest={manifest_path} validation={validation_path} hash_index={hash_index_path}",
        verbose=verbose,
    )
    return validation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewed v2 global latent-gauge intervention executor.")
    parser.add_argument("--protocol-mode", choices=("smoke", "real_smoke", "production"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-bank-csv", default=str(DEFAULT_START_BANK))
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--real-smoke-fit-source", type=int, default=PRODUCTION_FIT_SOURCES[0])
    parser.add_argument("--real-smoke-eval-source", type=int, default=PRODUCTION_EVAL_SOURCES[0])
    parser.add_argument(
        "--execute-production",
        action="store_true",
        help="After exact preflight, launch the full real 3x16 production protocol.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    verbose = not bool(args.quiet)
    if int(args.jacobian_chunk_size) <= 0:
        raise ValueError("--jacobian-chunk-size must be positive")
    _log(
        f"startup protocol_version={PROTOCOL_VERSION} mode={args.protocol_mode} "
        f"device={'cpu' if args.protocol_mode == 'smoke' else args.device} seed={args.seed} "
        f"execute_production={bool(args.execute_production)} output_dir={Path(args.output_dir).resolve()}",
        verbose=verbose,
    )
    eval_bank, fit_bank, bank_hashes = _load_protocol_banks(Path(args.start_bank_csv))
    _log(f"stage=bank_validation eval_rows={len(eval_bank)} fit_rows={len(fit_bank)} eval_hash={bank_hashes['evaluation_rows_sha256']} fit_hash={bank_hashes['metric_fit_rows_sha256']}", verbose=verbose)
    if args.protocol_mode == "smoke":
        if args.execute_production:
            raise ValueError("smoke mode rejects --execute-production")
        if args.run or args.label:
            raise ValueError("smoke mode rejects --run/--label; it uses exact synthetic three-seed coverage")
        _run_smoke(
            output_dir=Path(args.output_dir).expanduser().resolve(),
            eval_bank=eval_bank,
            fit_bank=fit_bank,
            bank_hashes=bank_hashes,
            seed=int(args.seed),
            verbose=verbose,
        )
    elif args.protocol_mode == "real_smoke":
        if args.execute_production:
            raise ValueError("real_smoke rejects --execute-production")
        real_smoke_device = torch.device(str(args.device))
        if real_smoke_device.type != "cuda":
            raise ValueError(f"real_smoke requires --device cuda:N, got={real_smoke_device}")
        if not torch.cuda.is_available():
            raise RuntimeError("real_smoke requires an available CUDA runtime")
        run_values = tuple(args.run or (DEFAULT_RUNS[0],))
        labels = tuple(args.label or (PRODUCTION_LABELS[0],))
        selection = _validate_real_smoke_selection(
            run_values=run_values,
            labels=labels,
            fit_source=int(args.real_smoke_fit_source),
            eval_source=int(args.real_smoke_eval_source),
        )
        selected_fit = fit_bank[
            fit_bank["source_weight_index"].astype(int) == int(selection["fit_source"])
        ].copy().reset_index(drop=True)
        selected_eval = eval_bank[
            eval_bank["source_weight_index"].astype(int) == int(selection["eval_source"])
        ].copy().reset_index(drop=True)
        if len(selected_fit) != 1 or len(selected_eval) != 1:
            raise RuntimeError("real_smoke pinned source selection did not resolve to exactly one row per split")
        preflight = _real_smoke_preflight(
            output_dir=Path(args.output_dir).expanduser().resolve(),
            run_values=run_values,
            labels=labels,
            fit_source=int(selection["fit_source"]),
            eval_source=int(selection["eval_source"]),
            bank_hashes=bank_hashes,
            device_name=str(args.device),
            jacobian_chunk_size=int(args.jacobian_chunk_size),
            verbose=verbose,
        )
        _run_real_smoke(
            output_dir=Path(args.output_dir).expanduser().resolve(),
            run_value=run_values[0],
            label=labels[0],
            fit_bank=selected_fit,
            eval_bank=selected_eval,
            bank_hashes=bank_hashes,
            preflight=preflight,
            device_name=str(args.device),
            jacobian_chunk_size=int(args.jacobian_chunk_size),
            seed=int(args.seed),
            verbose=verbose,
        )
    else:
        run_values = tuple(args.run or DEFAULT_RUNS)
        labels = tuple(args.label or PRODUCTION_LABELS)
        preflight = _production_preflight(
            output_dir=Path(args.output_dir).expanduser().resolve(),
            run_values=run_values,
            labels=labels,
            bank_hashes=bank_hashes,
            verbose=verbose,
        )
        if args.execute_production:
            _run_production(
                output_dir=Path(args.output_dir).expanduser().resolve(),
                run_values=run_values,
                labels=labels,
                eval_bank=eval_bank,
                fit_bank=fit_bank,
                bank_hashes=bank_hashes,
                preflight=preflight,
                device_name=str(args.device),
                jacobian_chunk_size=int(args.jacobian_chunk_size),
                seed=int(args.seed),
                verbose=verbose,
            )
        else:
            _log("production executor not launched; pass --execute-production after reviewer GO", verbose=verbose)


if __name__ == "__main__":
    main()
