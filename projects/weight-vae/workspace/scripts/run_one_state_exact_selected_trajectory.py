from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    encode_weights,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _latent_hvp,
    _task_set_for_record,
)
from scripts import audit_one_state_exact_a_proposal3_cross as exact_helpers
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _add_,
    _cosine,
    _dot,
    _named_tensor_hash,
    _negative_normalized,
    _norm,
    _vector_is_finite,
    _zeros_like,
)
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _load_run,
    _probe_cfg,
    sha256_file,
    sha256_tensor,
)
from scripts.run_one_state_a_full_burg_armijo import _set_parameters
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    STATE_BANK,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_selected_trajectory_i6_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_6_exact_selected_trajectory/protocol.md"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration6_exact_selected_trajectory_production"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_6_frozen_dependency_manifest.json"
EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = "042d7605633a76c408cd86c1dc01150d9bba3210da0c890ad6f62e78ba99c260"
EXPECTED_NORMALIZED_SOURCE_SHA256 = "20ad776cc9a7e2d513aab11ee527bd284e679a4d80558f5919f00813d9bb93b3"

ITERATION3 = OUTPUT_ROOT / "iteration3_p32_unit_common_production"
ITERATION3_CHECKPOINT = ITERATION3 / "final_checkpoint.pt"
ITERATION3_STATES = ITERATION3 / "state_metrics.csv"
ITERATION3_SPECTRA = ITERATION3 / "state_spectra.csv"
ITERATION5_ENDPOINTS = OUTPUT_ROOT / "iteration5_exact_a_low_common_production/endpoint_metrics.csv"
EXPECTED_ITERATION3_CHECKPOINT_SHA256 = (
    "ccafbad0e0fcf25aaabffe0a406558937d1b448ac06c80b7c40a6637209dc5ad"
)
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "31abec6439564c392a99b6c82dd1386aed714cb1b2a1ce1c4709088686a0caff"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"

SOURCE_WEIGHT_INDEX = 378
STATE_POSITION = 2
EPSILON = 1e-4
HESSIAN_CHUNK_SIZE = 64
BLOCK_SIZE = 64
LOW_THRESHOLD = 0.1
TARGET_NORM = 0.04892722657548397
CANCELLATION_NORM_MIN = 0.5
SELECTION_ALPHA = 1.0 / 128.0
ARMIJO_C1 = 1e-4
MAX_ACCEPTED_UPDATES = 100
LINE_ALPHAS = (
    1.0,
    1.0 / 2.0,
    1.0 / 4.0,
    1.0 / 8.0,
    1.0 / 16.0,
    1.0 / 32.0,
    1.0 / 64.0,
    1.0 / 128.0,
    1.0 / 256.0,
    1.0 / 512.0,
    1.0 / 1024.0,
    1.0 / 2048.0,
)
RETAINED_MEMORY_RANGE_MAX = 64 * 1024 * 1024
BLOCK_PEAK_RANGE_MAX = 128 * 1024 * 1024
BURG_GRADIENT_FORMULA_MAX_ERROR = 5e-6
FLOORS = {
    "A": 1e-9,
    "B": 1e-9,
    "L_low": 1e-10,
    "A_low90": 1e-9,
    "A_gt1": 1e-9,
    "m_max": 1e-8,
    "m_p50": 1e-10,
    "effective_rank": 1e-8,
}
REPEAT_MAX_ERROR = 1e-10
SUCCESS_A_MAX = 0.90
SUCCESS_BULK_FRACTION_MIN = 0.25
SUCCESS_TAIL_BULK_UPDATES_MIN = 4

Vector = list[torch.Tensor]


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _source_sha256(path: Path | None = None) -> str:
    return hashlib.sha256((path or Path(__file__)).read_bytes()).hexdigest()


def _normalized_source_sha256(path: Path | None = None) -> str:
    source = path or Path(__file__)
    masked = (
        "EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    normalized: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        prefix = next((candidate for candidate in masked if line.startswith(candidate)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _validate_dependencies() -> dict[str, bool]:
    if "TO_BE_FROZEN" in (
        EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        EXPECTED_NORMALIZED_SOURCE_SHA256,
    ):
        raise RuntimeError("runner has not been frozen")
    if _normalized_source_sha256() != EXPECTED_NORMALIZED_SOURCE_SHA256:
        raise RuntimeError("normalized source hash mismatch")
    if sha256_file(FROZEN_DEPENDENCY_MANIFEST) != EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256:
        raise RuntimeError("dependency manifest hash mismatch")
    expected = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    matches = {
        relative: (ROOT / relative).is_file() and sha256_file(ROOT / relative) == digest
        for relative, digest in expected.items()
    }
    failed = [relative for relative, ok in matches.items() if not ok]
    if failed:
        raise RuntimeError(f"frozen dependency mismatch: {failed}")
    return matches


def _numeric_finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def _row_numeric_values_finite(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _blocked_three_gradients(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    cotangents: Mapping[str, torch.Tensor],
    update: int,
) -> tuple[dict[str, Vector], dict[str, Any]]:
    names = ("A", "B", "low")
    if set(cotangents) != set(names):
        raise ValueError(f"cotangent names must be {names}")
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    dim = int(z.numel())
    accumulated = {name: _zeros_like(active, dtype=torch.float64) for name in names}
    unused = {name: np.zeros(len(active), dtype=np.int64) for name in names}
    retained_trace: list[int] = []
    peak_trace: list[int] = []
    block_count = math.ceil(dim / BLOCK_SIZE)
    started = time.perf_counter()
    for start in range(0, dim, BLOCK_SIZE):
        torch.cuda.reset_peak_memory_stats(z.device)
        stop = min(start + BLOCK_SIZE, dim)
        scalars = {name: [] for name in names}
        for index in range(start, stop):
            basis = torch.zeros_like(z)
            basis[index] = 1.0
            hvp = _latent_hvp(
                cfg,
                run.vae,
                run.normalizer,
                z,
                basis,
                task_set=task_set,
                spec=run.spec,
                tau=tau,
                batch_indices=None,
            )
            for name in names:
                scalars[name].append(
                    torch.dot(cotangents[name][index].to(dtype=hvp.dtype), hvp)
                )
        totals = {name: torch.stack(scalars[name]).sum() for name in names}
        gradients: dict[str, Sequence[torch.Tensor | None]] = {}
        gradients["A"] = torch.autograd.grad(
            totals["A"], active, retain_graph=True, allow_unused=True
        )
        gradients["B"] = torch.autograd.grad(
            totals["B"], active, retain_graph=True, allow_unused=True
        )
        gradients["low"] = torch.autograd.grad(
            totals["low"], active, retain_graph=False, allow_unused=True
        )
        for name in names:
            for index, value in enumerate(gradients[name]):
                if value is None:
                    unused[name][index] += 1
            _add_(accumulated[name], gradients[name])
        retained_trace.append(int(torch.cuda.memory_allocated(z.device)))
        peak_trace.append(int(torch.cuda.max_memory_allocated(z.device)))
        print(
            f"[exact-selected-i6] update={update} stage=blocked-three-vjp "
            f"rows={stop}/{dim} blocks={(stop + BLOCK_SIZE - 1) // BLOCK_SIZE}/"
            f"{block_count} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del scalars, totals, gradients
    edge = min(2, len(retained_trace))
    retained_range = max(retained_trace) - min(retained_trace)
    peak_range = max(peak_trace) - min(peak_trace)
    metadata = {
        "basis_count": dim,
        "block_count": block_count,
        "block_size": BLOCK_SIZE,
        "retained_memory_by_block_bytes": retained_trace,
        "peak_memory_by_block_bytes": peak_trace,
        "retained_memory_range_bytes": retained_range,
        "block_peak_range_bytes": peak_range,
        "memory_early_median_bytes": float(np.median(retained_trace[:edge])),
        "memory_late_median_bytes": float(np.median(retained_trace[-edge:])),
        "memory_gate_pass": bool(
            retained_range <= RETAINED_MEMORY_RANGE_MAX
            and peak_range <= BLOCK_PEAK_RANGE_MAX
        ),
        "elapsed_sec": time.perf_counter() - started,
    }
    for name in names:
        metadata[f"{name}_unused_parameter_tensors_all_blocks"] = int(
            (unused[name] == block_count).sum()
        )
        metadata[f"{name}_all_accumulators_float64"] = bool(
            all(value.dtype == torch.float64 for value in accumulated[name])
        )
    return accumulated, metadata


def _low_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    eig_raw, eigvec = torch.linalg.eigh(matrix.double())
    eig = eig_raw.clamp_min(0.0)
    mask = eig < LOW_THRESHOLD
    count = int(mask.sum().cpu())
    if count <= 0:
        raise RuntimeError("current low subspace is empty")
    basis = eigvec[:, mask].detach()
    values = eig_raw[mask]
    orthogonality = float(
        (
            basis.transpose(0, 1) @ basis
            - torch.eye(count, device=basis.device, dtype=torch.float64)
        )
        .abs()
        .max()
        .cpu()
    )
    residual = float(
        (matrix.double() @ basis - basis * values.unsqueeze(0)).norm().cpu()
        / max(float(matrix.double().norm().cpu()), 1e-30)
    )
    return basis, eig, {
        "low_count": float(count),
        "low_basis_orthogonality_max_abs": orthogonality,
        "low_basis_eigen_residual_relative": residual,
        "low_basis_hash": sha256_tensor(basis @ basis.transpose(0, 1)),
    }


def _augment_metrics(
    metrics: dict[str, float],
    matrix: torch.Tensor,
    eig: torch.Tensor,
    frozen_low_basis: torch.Tensor,
) -> dict[str, float]:
    result = dict(metrics)
    result.update(
        {
            "frozen_low_energy": float(
                torch.trace(
                    frozen_low_basis.transpose(0, 1)
                    @ matrix.double()
                    @ frozen_low_basis
                ).cpu()
                / float(frozen_low_basis.shape[1])
            ),
            "count_lt_1e_4": float((eig < 1e-4).sum().cpu()),
            "count_lt_1e_2": float((eig < 1e-2).sum().cpu()),
            "count_lt_0p1": float((eig < 0.1).sum().cpu()),
            "a_gt1": float(result["a_high_gt_1_abs_per_dim"]),
        }
    )
    return result


def _evaluate(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    frozen_low_basis: torch.Tensor | None,
) -> dict[str, Any]:
    metrics, hessian, matrix, eig_from_helper, burg_gradient = exact_helpers._evaluate_dense(
        run=run,
        z=z,
        record=record,
        epsilon=EPSILON,
        hessian_chunk_size=HESSIAN_CHUNK_SIZE,
    )
    current_basis, eig, basis_meta = _low_basis(matrix)
    basis = current_basis if frozen_low_basis is None else frozen_low_basis
    metrics = _augment_metrics(metrics, matrix, eig, basis)
    metrics["canonical_low_energy"] = float(
        torch.trace(current_basis.transpose(0, 1) @ matrix.double() @ current_basis).cpu()
        / float(current_basis.shape[1])
    )
    metrics.update(basis_meta)
    eig_helper_error = float((eig - eig_from_helper.double()).abs().max().cpu())
    metrics["helper_eigenvalue_max_abs_error"] = eig_helper_error
    return {
        "metrics": metrics,
        "hessian": hessian.detach().double().clone(),
        "matrix": matrix.detach().double().clone(),
        "eig": eig.detach().double().clone(),
        "current_low_basis": current_basis,
        "burg_gradient": burg_gradient.detach().double().clone(),
    }


def _repeat_errors(primary: Mapping[str, Any], repeat: Mapping[str, Any]) -> dict[str, float]:
    discrete = {"count_lt_1e_4", "count_lt_1e_2", "count_lt_0p1"}
    continuous = sorted(
        (set(primary["metrics"]) & set(repeat["metrics"]))
        - discrete
        - {"hessian_sec", "low_basis_hash"}
    )
    result = {
        f"repeat_{key}_abs_error": abs(
            float(primary["metrics"][key]) - float(repeat["metrics"][key])
        )
        for key in continuous
    }
    for key in sorted(discrete):
        result[f"repeat_{key}_abs_error"] = abs(
            float(primary["metrics"][key]) - float(repeat["metrics"][key])
        )
    result["repeat_spectrum_max_abs_error"] = float(
        (primary["eig"] - repeat["eig"]).abs().max().cpu()
    )
    result["repeat_hessian_max_abs_error"] = float(
        (primary["hessian"] - repeat["hessian"]).abs().max().cpu()
    )
    return result


def _tolerances(base_repeat_errors: Mapping[str, float]) -> dict[str, float]:
    metric_to_repeat = {
        "A": "repeat_exact_a_per_dim_abs_error",
        "B": "repeat_damped_full_burg_per_dim_abs_error",
        "L_low": "repeat_frozen_low_energy_abs_error",
        "A_low90": "repeat_a_low90_abs_per_dim_abs_error",
        "A_gt1": "repeat_a_gt1_abs_error",
        "m_max": "repeat_m_max_abs_error",
        "m_p50": "repeat_m_p50_abs_error",
        "effective_rank": "repeat_effective_rank_abs_error",
    }
    return {
        name: max(5.0 * float(base_repeat_errors[column]), FLOORS[name])
        for name, column in metric_to_repeat.items()
    }


def _unit_direction(
    gradient_a: Sequence[torch.Tensor],
    gradient_x: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
) -> tuple[Vector | None, dict[str, float | bool]]:
    norm_a = _norm(gradient_a)
    norm_x = _norm(gradient_x)
    cosine = _cosine(gradient_a, gradient_x)
    common = exact_helpers._unit_common_gradient(gradient_a, gradient_x, active)
    source_norm = _norm(common)
    eligible = bool(
        math.isfinite(source_norm)
        and norm_a > 0.0
        and norm_x > 0.0
        and source_norm >= CANCELLATION_NORM_MIN
    )
    direction = (
        _negative_normalized(common, target_norm=TARGET_NORM, active=active)
        if eligible
        else None
    )
    return direction, {
        "gradient_a_norm": norm_a,
        "gradient_x_norm": norm_x,
        "gradient_cosine": cosine,
        "unit_common_source_norm": source_norm,
        "cancellation_gate_pass": eligible,
        "direction_norm": 0.0 if direction is None else _norm(direction),
    }


def _objective_values(metrics: Mapping[str, float]) -> dict[str, float]:
    return {
        "A": float(metrics["exact_a_per_dim"]),
        "B": float(metrics["damped_full_burg_per_dim"]),
        "L_low": -float(metrics["frozen_low_energy"]),
    }


def _candidate_gates(
    *,
    current: Mapping[str, float],
    candidate: Mapping[str, float],
    slopes: Mapping[str, float],
    alpha: float,
    tolerances: Mapping[str, float],
    require_low90: bool,
) -> dict[str, bool]:
    current_objectives = _objective_values(current)
    candidate_objectives = _objective_values(candidate)
    gates: dict[str, bool] = {}
    for name in ("A", "B", "L_low"):
        gates[f"{name}_slope_negative"] = float(slopes[name]) < 0.0
        gates[f"{name}_armijo"] = candidate_objectives[name] <= (
            current_objectives[name] + ARMIJO_C1 * alpha * float(slopes[name])
        )
        gates[f"{name}_actual_decrease"] = (
            current_objectives[name] - candidate_objectives[name]
            > float(tolerances[name])
        )
    gates["high_tail_nonincrease"] = float(candidate["a_gt1"]) <= (
        float(current["a_gt1"]) + float(tolerances["A_gt1"])
    )
    gates["low90_decreases"] = (
        not require_low90
        or float(current["a_low90_abs_per_dim"])
        - float(candidate["a_low90_abs_per_dim"])
        > float(tolerances["A_low90"])
    )
    gates["exact_a_closes"] = bool(
        float(candidate["a_direct_abs_error"]) <= 1e-9
        and float(candidate["a_trace_abs_error"]) <= 1e-9
    )
    gates["spectrum_finite_nonnegative"] = bool(
        math.isfinite(float(candidate["m_raw_eig_min"]))
        and float(candidate["m_raw_eig_min"]) >= -1e-8
    )
    gates["metrics_finite"] = all(
        math.isfinite(float(value))
        for key, value in candidate.items()
        if key != "low_basis_hash"
    )
    return gates


def _historical_transition_audit(
    *,
    state_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    line_rows: Sequence[Mapping[str, Any]],
    tolerances: Mapping[str, float],
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    accepted = [row for row in proposal_rows if int(row.get("accepted", 0)) == 1]
    states_by_update = {int(row["accepted_update"]): row for row in state_rows}
    audit_rows: list[dict[str, Any]] = []
    sequence_pass = [int(row["target_update"]) for row in accepted] == list(
        range(1, len(accepted) + 1)
    )
    for proposal in accepted:
        update = int(proposal["target_update"])
        alpha = float(proposal["selected_alpha"])
        slopes = {
            "A": float(proposal["slope_A"]),
            "B": float(proposal["slope_B"]),
            "L_low": float(proposal["slope_L_low"]),
        }
        current = {
            "A": float(proposal["current_A"]),
            "B": float(proposal["current_B"]),
            "L_low": -float(proposal["current_low_energy"]),
        }
        candidate = {
            "A": float(proposal["candidate_A"]),
            "B": float(proposal["candidate_B"]),
            "L_low": -float(proposal["candidate_low_energy"]),
        }
        component_gates: dict[str, bool] = {}
        for name in ("A", "B", "L_low"):
            component_gates[f"{name}_slope_negative"] = slopes[name] < 0.0
            component_gates[f"{name}_armijo"] = candidate[name] <= (
                current[name] + ARMIJO_C1 * alpha * slopes[name]
            )
            component_gates[f"{name}_actual_decrease"] = (
                current[name] - candidate[name] > float(tolerances[name])
            )
        high_tail_pass = float(proposal["candidate_a_gt1"]) <= (
            float(proposal["current_a_gt1"]) + float(tolerances["A_gt1"])
        )
        target_lines = [
            row for row in line_rows if int(row["target_update"]) == update
        ]
        matching_lines = [
            row
            for row in target_lines
            if math.isclose(float(row["alpha"]), alpha, rel_tol=0.0, abs_tol=0.0)
        ]
        line_pass = bool(
            len(matching_lines) == 1
            and int(matching_lines[0]["passes"]) == 1
            and sum(int(row["passes"]) for row in target_lines) == 1
            and target_lines
            and target_lines[-1] is matching_lines[0]
        )
        if line_pass:
            gate_columns = [
                key for key in matching_lines[0] if key.startswith("gate_")
            ]
            line_pass = bool(all(int(matching_lines[0][key]) == 1 for key in gate_columns))
        state = states_by_update.get(update)
        state_matches = bool(
            state is not None
            and abs(float(state["exact_a_per_dim"]) - candidate["A"]) <= 1e-9
            and abs(float(state["damped_full_burg_per_dim"]) - candidate["B"]) <= 1e-9
            and abs(float(state["a_gt1"]) - float(proposal["candidate_a_gt1"])) <= 1e-9
        )
        row_pass = bool(
            all(component_gates.values()) and high_tail_pass and line_pass and state_matches
        )
        audit_rows.append(
            {
                "accepted_update": update,
                "alpha": alpha,
                **{key: int(value) for key, value in component_gates.items()},
                "high_tail_nonincrease": int(high_tail_pass),
                "selected_line_row_passes": int(line_pass),
                "committed_state_matches": int(state_matches),
                "transition_passes": int(row_pass),
            }
        )
    summary = {
        "accepted_proposal_count_matches_states": len(accepted) == len(state_rows) - 1,
        "accepted_update_sequence_exact": sequence_pass,
        "all_historical_transitions_pass": bool(
            all(int(row["transition_passes"]) == 1 for row in audit_rows)
        ),
        "all_historical_a_armijo_pass": bool(
            all(
                int(row["A_slope_negative"]) == 1
                and int(row["A_armijo"]) == 1
                and int(row["A_actual_decrease"]) == 1
                for row in audit_rows
            )
        ),
    }
    return summary, audit_rows


def _gradient_payload(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    state: Mapping[str, Any],
    update: int,
) -> tuple[dict[str, Vector], dict[str, Any]]:
    hessian = state["hessian"].double()
    matrix = state["matrix"].double()
    low_basis = state["current_low_basis"].double()
    dim = int(hessian.shape[0])
    identity = torch.eye(dim, device=hessian.device, dtype=torch.float64)
    k_a = (4.0 * (matrix - identity) @ hessian / float(dim)).detach()
    g_b_matrix = identity / (float(dim) * (1.0 + EPSILON)) - torch.linalg.inv(
        matrix + EPSILON * identity
    ) / float(dim)
    k_b = (2.0 * g_b_matrix @ hessian).detach()
    k_low = (
        -2.0
        * low_basis
        @ (low_basis.transpose(0, 1) @ hessian)
        / float(low_basis.shape[1])
    ).detach()
    burg_gradient_error = float(
        (g_b_matrix - state["burg_gradient"].double()).abs().max().cpu()
    )
    gradients, metadata = _blocked_three_gradients(
        cfg=cfg,
        run=run,
        z=z,
        record=record,
        active=active,
        cotangents={"A": k_a, "B": k_b, "low": k_low},
        update=update,
    )
    metadata.update(
        {
            "k_a_norm": float(k_a.norm().cpu()),
            "k_b_norm": float(k_b.norm().cpu()),
            "k_low_norm": float(k_low.norm().cpu()),
            "burg_gradient_formula_max_abs_error": burg_gradient_error,
        }
    )
    valid = bool(
        metadata["basis_count"] == 512
        and metadata["memory_gate_pass"]
        and burg_gradient_error <= BURG_GRADIENT_FORMULA_MAX_ERROR
        and all(
            int(metadata[f"{name}_unused_parameter_tensors_all_blocks"]) == 0
            and bool(metadata[f"{name}_all_accumulators_float64"])
            and _vector_is_finite(gradients[name])
            and _norm(gradients[name]) > 0.0
            for name in ("A", "B", "low")
        )
    )
    metadata["gradient_payload_valid"] = valid
    if not valid:
        raise RuntimeError(f"exact three-gradient gate failed: {metadata}")
    return gradients, metadata


def _line_endpoint(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active_names: Sequence[str],
    active: Sequence[torch.nn.Parameter],
    base: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    alpha: float,
    frozen_low_basis: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, float], str]:
    _set_parameters(list(active), list(base), list(direction), alpha)
    endpoint_hash = _named_tensor_hash(active_names, active)
    primary = _evaluate(
        run=run,
        z=z,
        record=record,
        frozen_low_basis=frozen_low_basis,
    )
    repeat = _evaluate(
        run=run,
        z=z,
        record=record,
        frozen_low_basis=frozen_low_basis,
    )
    errors = _repeat_errors(primary, repeat)
    errors["repeat_parameter_hash_unchanged"] = float(
        _named_tensor_hash(active_names, active) == endpoint_hash
    )
    return primary, errors, endpoint_hash


def _repeat_passes(
    errors: Mapping[str, float], tolerances: Mapping[str, float]
) -> bool:
    metric_limits = {
        "repeat_exact_a_per_dim_abs_error": float(tolerances["A"]),
        "repeat_damped_full_burg_per_dim_abs_error": float(tolerances["B"]),
        "repeat_frozen_low_energy_abs_error": float(tolerances["L_low"]),
        "repeat_a_low90_abs_per_dim_abs_error": float(tolerances["A_low90"]),
        "repeat_a_gt1_abs_error": float(tolerances["A_gt1"]),
        "repeat_m_max_abs_error": float(tolerances["m_max"]),
        "repeat_m_p50_abs_error": float(tolerances["m_p50"]),
        "repeat_effective_rank_abs_error": float(tolerances["effective_rank"]),
    }
    exact_keys = {
        "repeat_count_lt_1e_4_abs_error",
        "repeat_count_lt_1e_2_abs_error",
        "repeat_count_lt_0p1_abs_error",
        "repeat_parameter_hash_unchanged",
    }
    if float(errors["repeat_parameter_hash_unchanged"]) != 1.0:
        return False
    if any(float(errors[key]) != 0.0 for key in exact_keys - {"repeat_parameter_hash_unchanged"}):
        return False
    if any(float(errors[key]) > limit for key, limit in metric_limits.items()):
        return False
    remaining = set(errors) - set(metric_limits) - exact_keys
    return bool(all(float(errors[key]) <= REPEAT_MAX_ERROR for key in remaining))


def _state_row(update: int, metrics: Mapping[str, Any], parameter_hash: str) -> dict[str, Any]:
    return {"accepted_update": update, "parameter_hash": parameter_hash, **metrics}


def _spectrum_rows(update: int, eig: torch.Tensor) -> list[dict[str, float | int]]:
    return [
        {
            "accepted_update": update,
            "rank": rank,
            "m_eigenvalue": float(value),
            "a_contribution": float((value - 1.0) ** 2),
        }
        for rank, value in enumerate(eig.detach().cpu().numpy())
    ]


def _write_progress_tables(
    staging: Path,
    state_rows: Sequence[Mapping[str, Any]],
    spectrum_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    line_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_csv(staging / "state_metrics.csv", state_rows)
    _atomic_csv(staging / "state_spectra.csv", spectrum_rows)
    _atomic_csv(staging / "proposal_diagnostics.csv", proposal_rows)
    _atomic_csv(staging / "line_search.csv", line_rows)
    _atomic_csv(staging / "arm_selection.csv", selection_rows)


def _write_progress_checkpoint(
    *,
    staging: Path,
    active_names: Sequence[str],
    active: Sequence[torch.nn.Parameter],
    accepted_updates: int,
    selected_arm: str,
    state_rows: Sequence[Mapping[str, Any]],
    spectrum_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    line_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    initial_metrics: Mapping[str, Any],
    tolerances: Mapping[str, float],
    terminal: bool,
    termination: str,
    proposal0_payload: Mapping[str, Any],
) -> None:
    checkpoint = {
        "protocol_id": PROTOCOL_ID,
        "normalized_source_sha256": EXPECTED_NORMALIZED_SOURCE_SHA256,
        "dependency_manifest_sha256": EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        "accepted_updates": accepted_updates,
        "selected_arm": selected_arm,
        "terminal": bool(terminal),
        "termination": termination,
        "proposal0_payload": dict(proposal0_payload),
        "active_parameter_hash": _named_tensor_hash(active_names, active),
        "active_model_state": {
            name: parameter.detach().cpu().clone()
            for name, parameter in zip(active_names, active, strict=True)
        },
        "state_rows": list(state_rows),
        "spectrum_rows": list(spectrum_rows),
        "proposal_rows": list(proposal_rows),
        "line_rows": list(line_rows),
        "selection_rows": list(selection_rows),
        "initial_metrics": dict(initial_metrics),
        "tolerances": dict(tolerances),
    }
    _atomic_torch_save(staging / "progress_checkpoint.pt", checkpoint)
    _write_progress_tables(
        staging,
        state_rows,
        spectrum_rows,
        proposal_rows,
        line_rows,
        selection_rows,
    )


def _plot(output: Path, states: pd.DataFrame, spectra: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    panels = (
        ("exact_a_per_dim", "Exact dense A"),
        ("damped_full_burg_per_dim", "Damped full Burg B"),
        ("a_low90_abs_per_dim", "Lower-90 A contribution"),
        ("a_gt1", "High-tail A contribution"),
        ("m_p50", "Median M eigenvalue"),
        ("effective_rank", "Effective rank"),
    )
    for axis, (column, title) in zip(axes.flat, panels, strict=True):
        axis.plot(states["accepted_update"], states[column], linewidth=2.0)
        axis.set(title=title, xlabel="accepted update", ylabel=column)
        axis.grid(alpha=0.25)
    fig.suptitle("Iteration 6 exact selected-arm trajectory")
    fig.savefig(output / "exact_selected_trajectory.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    updates = sorted(set(int(value) for value in spectra["accepted_update"]))
    chosen = sorted(set([updates[0], *[value for value in (25, 50, 75) if value in updates], updates[-1]]))
    for update in chosen:
        rows = spectra.loc[spectra["accepted_update"].eq(update)].sort_values("rank")
        axis.plot(
            rows["rank"],
            np.clip(rows["m_eigenvalue"], 1e-14, None),
            label=f"update {update}",
        )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_yscale("log")
    axis.set(title="Ordered M spectra", xlabel="rank", ylabel="eigenvalue")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "exact_selected_spectra.png", dpi=180)
    plt.close(fig)


def _manifest(output: Path, source_snapshot: Path) -> dict[str, Any]:
    excluded = {
        "artifact_manifest.json",
        "FINALIZED.json",
        "INCOMPLETE",
        "run.log",
    }
    artifacts = sorted(
        path for path in output.iterdir() if path.is_file() and path.name not in excluded
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "executed_source_sha256": _source_sha256(source_snapshot),
        "executed_normalized_source_sha256": _normalized_source_sha256(source_snapshot),
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
    }


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("production trajectory accepts no CLI overrides")
    dependency_matches = _validate_dependencies()
    final_output = DEFAULT_OUTPUT.resolve()
    staging = Path(str(final_output) + ".incomplete")
    if final_output.exists():
        if (final_output / "FINALIZED.json").is_file() and (
            final_output / "INCOMPLETE"
        ).is_file():
            recovered = json.loads((final_output / "FINALIZED.json").read_text())
            if (
                recovered.get("protocol_id") != PROTOCOL_ID
                or sha256_file(final_output / "decision.json")
                != recovered.get("decision_sha256")
                or sha256_file(final_output / "artifact_manifest.json")
                != recovered.get("artifact_manifest_sha256")
            ):
                raise RuntimeError("published finalization recovery hash mismatch")
            (final_output / "INCOMPLETE").unlink()
            print(
                f"[exact-selected-i6] recovered published finalization marker: "
                f"{final_output}",
                flush=True,
            )
            return
        raise FileExistsError(f"refusing to overwrite {final_output}")
    resume = staging.exists()
    if resume:
        if not (staging / "INCOMPLETE").is_file() or not (
            staging / "progress_checkpoint.pt"
        ).is_file():
            raise RuntimeError("incomplete staging exists without resumable checkpoint")
        source_snapshot = staging / "executed_source_snapshot.py"
        if _normalized_source_sha256(source_snapshot) != EXPECTED_NORMALIZED_SOURCE_SHA256:
            raise RuntimeError("resume source snapshot mismatch")
    else:
        staging.mkdir(parents=True)
        (staging / "INCOMPLETE").write_text(PROTOCOL_ID + "\n", encoding="utf-8")
        source_snapshot = staging / "executed_source_snapshot.py"
        shutil.copy2(Path(__file__), source_snapshot)
        shutil.copy2(PROTOCOL_PATH, staging / "protocol_snapshot.md")
        shutil.copy2(
            FROZEN_DEPENDENCY_MANIFEST,
            staging / "frozen_dependency_manifest_snapshot.json",
        )

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_handle = (staging / "run.log").open(
        "a" if resume else "w", encoding="utf-8", buffering=1
    )
    sys.stdout = _Tee(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, log_handle)  # type: ignore[assignment]
    started = time.perf_counter()
    device = torch.device("cuda:0")
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "resume": resume,
        "device": str(device),
        "dtype": "float32 model/HVP; FP64 gradient accumulation and diagnostics",
        "seed": "none; exact full-basis gradients",
        "cache_mode": "accepted h2048 run + serialized Iteration-3 checkpoint",
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "state_position": STATE_POSITION,
        "epsilon": EPSILON,
        "target_norm": TARGET_NORM,
        "selection_alpha": SELECTION_ALPHA,
        "armijo_c1": ARMIJO_C1,
        "line_alphas": list(LINE_ALPHAS),
        "max_accepted_updates": MAX_ACCEPTED_UPDATES,
        "burg_gradient_formula_max_error": BURG_GRADIENT_FORMULA_MAX_ERROR,
        "floors": FLOORS,
        "output_dir": str(final_output),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_sha256": _source_sha256(source_snapshot),
        "normalized_source_sha256": _normalized_source_sha256(source_snapshot),
        "dependency_manifest_sha256": sha256_file(FROZEN_DEPENDENCY_MANIFEST),
        "dependency_matches": dependency_matches,
    }
    if not resume:
        _atomic_json(staging / "resolved_config.json", resolved)
    else:
        stored_resolved = json.loads((staging / "resolved_config.json").read_text())
        for key in (
            "protocol_id",
            "source_sha256",
            "normalized_source_sha256",
            "dependency_manifest_sha256",
            "line_alphas",
            "max_accepted_updates",
        ):
            if stored_resolved[key] != resolved[key]:
                raise RuntimeError(f"resume resolved-config mismatch: {key}")
    print(f"[exact-selected-i6] startup {json.dumps(resolved, sort_keys=True)}", flush=True)

    try:
        print("[exact-selected-i6] stage=load-fresh-run", flush=True)
        accepted_checkpoint = DEFAULT_RUN_DIR / "vae_checkpoint.pt"
        if sha256_file(accepted_checkpoint) != EXPECTED_CHECKPOINT:
            raise RuntimeError("accepted h2048 checkpoint hash mismatch")
        if sha256_file(ITERATION3_CHECKPOINT) != EXPECTED_ITERATION3_CHECKPOINT_SHA256:
            raise RuntimeError("Iteration-3 checkpoint hash mismatch")
        run = _load_run(DEFAULT_RUN_DIR, device=device)
        cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
        cfg = replace(cfg, vae_precond_hvp_mode="autograd")
        bank = pd.read_csv(STATE_BANK)
        selected_state = bank.loc[bank["state_position"].eq(STATE_POSITION)]
        if len(selected_state) != 1 or int(
            selected_state.iloc[0]["source_weight_index"]
        ) != SOURCE_WEIGHT_INDEX:
            raise RuntimeError("state-bank identity mismatch")
        record = run.records.iloc[SOURCE_WEIGHT_INDEX].to_dict()
        record["source_weight_index"] = SOURCE_WEIGHT_INDEX
        weight = run.weights[[SOURCE_WEIGHT_INDEX]].to(
            device=device, dtype=torch_dtype(run.cfg)
        )
        with torch.no_grad():
            z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
        if sha256_tensor(z) != EXPECTED_Z_SHA256:
            raise RuntimeError("z fingerprint mismatch")
        task_set = _task_set_for_record(run.task_tensors, record)
        if not all(
            _batch_indices(
                task_set,
                batch_size=int(cfg.vae_precond_batch_size),
                step=10,
                sample_key=SOURCE_WEIGHT_INDEX,
                pair_key=pair_key,
            )
            is None
            for pair_key in range(8)
        ):
            raise RuntimeError("full CE batch gate failed")
        active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
        named = dict(run.vae.named_parameters())
        for name, parameter in named.items():
            parameter.requires_grad_(name in active_names)
        active = [named[name] for name in active_names]
        active_parameter_count = sum(parameter.numel() for parameter in active)
        if active_parameter_count != 11_685_120:
            raise RuntimeError(f"active parameter count mismatch: {active_parameter_count}")
        iteration3 = torch.load(ITERATION3_CHECKPOINT, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name in active_names:
                named[name].copy_(iteration3["active_model_state"][name].to(device=device))
        if _named_tensor_hash(active_names, active) != EXPECTED_INITIAL_PARAMETER_SHA256:
            raise RuntimeError("Iteration-3 active parameter hash mismatch")

        state_rows: list[dict[str, Any]] = []
        spectrum_rows: list[dict[str, Any]] = []
        proposal_rows: list[dict[str, Any]] = []
        line_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        selected_arm: str | None = None
        accepted_updates = 0
        resumed_terminal = False
        resumed_termination = ""
        initial_metrics: dict[str, Any]
        tolerances: dict[str, float]
        proposal0_payload: dict[str, Any] = {}

        if resume:
            print("[exact-selected-i6] stage=resume-checkpoint", flush=True)
            progress = torch.load(
                staging / "progress_checkpoint.pt", map_location="cpu", weights_only=False
            )
            if progress["protocol_id"] != PROTOCOL_ID:
                raise RuntimeError("resume protocol mismatch")
            if progress["normalized_source_sha256"] != EXPECTED_NORMALIZED_SOURCE_SHA256:
                raise RuntimeError("resume source fingerprint mismatch")
            if (
                progress["dependency_manifest_sha256"]
                != EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256
            ):
                raise RuntimeError("resume dependency fingerprint mismatch")
            with torch.no_grad():
                for name in active_names:
                    named[name].copy_(progress["active_model_state"][name].to(device=device))
            if _named_tensor_hash(active_names, active) != progress["active_parameter_hash"]:
                raise RuntimeError("resume active parameter hash mismatch")
            accepted_updates = int(progress["accepted_updates"])
            selected_arm = str(progress["selected_arm"])
            resumed_terminal = bool(progress["terminal"])
            resumed_termination = str(progress["termination"])
            state_rows = list(progress["state_rows"])
            spectrum_rows = list(progress["spectrum_rows"])
            proposal_rows = list(progress["proposal_rows"])
            line_rows = list(progress["line_rows"])
            selection_rows = list(progress["selection_rows"])
            initial_metrics = dict(progress["initial_metrics"])
            tolerances = dict(progress["tolerances"])
            proposal0_payload = dict(progress["proposal0_payload"])
            proposal0_path = staging / "proposal0_decision.json"
            if proposal0_path.exists():
                if json.loads(proposal0_path.read_text(encoding="utf-8")) != proposal0_payload:
                    raise RuntimeError("resume proposal-0 payload mismatch")
            else:
                _atomic_json(proposal0_path, proposal0_payload)
        else:
            print("[exact-selected-i6] stage=base-replay", flush=True)
            base_state = _evaluate(run=run, z=z, record=record, frozen_low_basis=None)
            base_repeat = _evaluate(
                run=run,
                z=z,
                record=record,
                frozen_low_basis=base_state["current_low_basis"],
            )
            base_repeat_errors = _repeat_errors(base_state, base_repeat)
            if max(base_repeat_errors.values()) > REPEAT_MAX_ERROR:
                raise RuntimeError("base repeatability gate failed")
            tolerances = _tolerances(base_repeat_errors)
            initial_metrics = dict(base_state["metrics"])
            stored_state = pd.read_csv(ITERATION3_STATES).loc[
                lambda frame: frame["proposal"].eq(100)
            ].iloc[0]
            stored_spectrum = (
                pd.read_csv(ITERATION3_SPECTRA)
                .loc[lambda frame: frame["proposal"].eq(100)]
                .sort_values("rank")["m_eigenvalue"]
                .to_numpy(dtype=np.float64)
            )
            replay_errors = {
                key: abs(float(initial_metrics[key]) - float(stored_state[key]))
                for key in (
                    "exact_a_per_dim",
                    "damped_full_burg_per_dim",
                    "m_max",
                    "m_p50",
                    "m_lt_1e_4_fraction",
                    "m_lt_0p01_fraction",
                    "m_lt_0p1_fraction",
                )
            }
            replay_spectrum_error = float(
                np.max(np.abs(base_state["eig"].cpu().numpy() - stored_spectrum))
            )
            if max([*replay_errors.values(), replay_spectrum_error]) > 1e-9:
                raise RuntimeError("base does not replay Iteration-3 state 100")

            print("[exact-selected-i6] stage=proposal0-three-gradients", flush=True)
            gradients, gradient_meta = _gradient_payload(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                active=active,
                state=base_state,
                update=0,
            )
            directions: dict[str, Vector | None] = {}
            direction_meta: dict[str, dict[str, float | bool]] = {}
            directions["B"], direction_meta["B"] = _unit_direction(
                gradients["A"], gradients["B"], active
            )
            directions["low"], direction_meta["low"] = _unit_direction(
                gradients["A"], gradients["low"], active
            )
            start_parameters = [parameter.detach().clone() for parameter in active]
            start_hash = _named_tensor_hash(active_names, active)
            iteration5_reference = pd.read_csv(ITERATION5_ENDPOINTS).loc[
                lambda frame: np.isclose(frame["alpha"], SELECTION_ALPHA)
            ]
            if len(iteration5_reference) != 1:
                raise RuntimeError("Iteration-5 low-arm reference row is not unique")
            iteration5_reference_row = iteration5_reference.iloc[0]
            iteration5_replay_errors: dict[str, float] = {}
            arm_results: list[dict[str, Any]] = []
            for arm in ("B", "low"):
                direction = directions[arm]
                slopes = {
                    "A": 0.0 if direction is None else _dot(gradients["A"], direction),
                    "B": 0.0 if direction is None else _dot(gradients["B"], direction),
                    "L_low": 0.0
                    if direction is None
                    else _dot(gradients["low"], direction),
                }
                if direction is None:
                    if arm == "low":
                        raise RuntimeError(
                            "low arm failed cancellation gate despite Iteration-5 reference"
                        )
                    row = {
                        "arm": arm,
                        "alpha": SELECTION_ALPHA,
                        **direction_meta[arm],
                        **{f"slope_{key}": value for key, value in slopes.items()},
                        "eligible": False,
                        "failure": "cancellation_gate",
                        "iteration5_reference_max_abs_error": 0.0,
                        "iteration5_reference_hash_matches": 1,
                    }
                    selection_rows.append(row)
                    arm_results.append(row)
                    continue
                try:
                    endpoint, repeat_errors, endpoint_hash = _line_endpoint(
                        run=run,
                        z=z,
                        record=record,
                        active_names=active_names,
                        active=active,
                        base=start_parameters,
                        direction=direction,
                        alpha=SELECTION_ALPHA,
                        frozen_low_basis=base_state["current_low_basis"],
                    )
                    gates = _candidate_gates(
                        current=initial_metrics,
                        candidate=endpoint["metrics"],
                        slopes=slopes,
                        alpha=SELECTION_ALPHA,
                        tolerances=tolerances,
                        require_low90=True,
                    )
                    repeat_pass = _repeat_passes(repeat_errors, tolerances)
                    eligible = bool(all(gates.values()) and repeat_pass)
                    row = {
                        "arm": arm,
                        "alpha": SELECTION_ALPHA,
                        **direction_meta[arm],
                        **{f"slope_{key}": value for key, value in slopes.items()},
                        **endpoint["metrics"],
                        **{f"gate_{key}": int(value) for key, value in gates.items()},
                        **repeat_errors,
                        "endpoint_parameter_hash": endpoint_hash,
                        "repeat_pass": int(repeat_pass),
                        "low90_decrease": float(
                            initial_metrics["a_low90_abs_per_dim"]
                            - endpoint["metrics"]["a_low90_abs_per_dim"]
                        ),
                        "a_decrease": float(
                            initial_metrics["exact_a_per_dim"]
                            - endpoint["metrics"]["exact_a_per_dim"]
                        ),
                        "eligible": eligible,
                        "failure": "" if eligible else "selection_gates",
                        "iteration5_reference_max_abs_error": 0.0,
                        "iteration5_reference_hash_matches": 1,
                    }
                    if arm == "low":
                        iteration5_replay_errors = {
                            key: abs(
                                float(endpoint["metrics"][key])
                                - float(iteration5_reference_row[key])
                            )
                            for key in (
                                "exact_a_per_dim",
                                "damped_full_burg_per_dim",
                                "frozen_low_energy",
                                "a_low90_abs_per_dim",
                                "a_high_gt_1_abs_per_dim",
                                "m_max",
                                "m_p50",
                            )
                        }
                        iteration5_hash_matches = endpoint_hash == str(
                            iteration5_reference_row["endpoint_parameter_hash"]
                        )
                        row["iteration5_reference_max_abs_error"] = max(
                            iteration5_replay_errors.values()
                        )
                        row["iteration5_reference_hash_matches"] = int(
                            iteration5_hash_matches
                        )
                        if (
                            max(iteration5_replay_errors.values()) > 1e-9
                            or not iteration5_hash_matches
                        ):
                            raise RuntimeError("Iteration-5 low-arm endpoint replay failed")
                    selection_rows.append(row)
                    arm_results.append(row)
                finally:
                    _set_parameters(active, start_parameters, list(direction), 0.0)
                    if _named_tensor_hash(active_names, active) != start_hash:
                        raise RuntimeError(f"proposal-0 restoration failed for arm={arm}")
            eligible_arms = [row for row in arm_results if bool(row["eligible"])]
            if eligible_arms:
                chosen = sorted(
                    eligible_arms,
                    key=lambda row: (
                        -float(row["low90_decrease"]),
                        -float(row["a_decrease"]),
                        str(row["arm"]),
                    ),
                )[0]
                selected_arm = str(chosen["arm"])
            else:
                selected_arm = "none"
            initial_state_metrics = dict(initial_metrics)
            initial_state_metrics["parent_frozen_low_energy"] = float(
                initial_state_metrics["frozen_low_energy"]
            )
            state_rows = [_state_row(0, initial_state_metrics, start_hash)]
            spectrum_rows = _spectrum_rows(0, base_state["eig"])
            proposal0_payload = {
                "selected_arm": selected_arm,
                "eligible_arms": [str(row["arm"]) for row in eligible_arms],
                "gradient_metadata": gradient_meta,
                "direction_metadata": direction_meta,
                "tolerances": tolerances,
                "replay_errors": replay_errors,
                "replay_spectrum_error": replay_spectrum_error,
                "iteration5_low_arm_replay_errors": iteration5_replay_errors,
            }
            _write_progress_checkpoint(
                staging=staging,
                active_names=active_names,
                active=active,
                accepted_updates=0,
                selected_arm=selected_arm,
                state_rows=state_rows,
                spectrum_rows=spectrum_rows,
                proposal_rows=proposal_rows,
                line_rows=line_rows,
                selection_rows=selection_rows,
                initial_metrics=initial_metrics,
                tolerances=tolerances,
                terminal=selected_arm == "none",
                termination=(
                    "proposal0_no_eligible_arm" if selected_arm == "none" else "running"
                ),
                proposal0_payload=proposal0_payload,
            )
            _atomic_json(staging / "proposal0_decision.json", proposal0_payload)
            del gradients, directions, base_repeat, base_repeat_errors

        if selected_arm not in {"B", "low"}:
            termination = "proposal0_no_eligible_arm"
            print("[exact-selected-i6] proposal0 no eligible arm; terminating", flush=True)
        elif resumed_terminal:
            termination = resumed_termination
            print(
                f"[exact-selected-i6] terminal checkpoint replayed; "
                f"accepted={accepted_updates} termination={termination}",
                flush=True,
            )
        else:
            termination = "max_updates_reached"
            print(
                f"[exact-selected-i6] selected_arm={selected_arm} "
                f"resume_update={accepted_updates}",
                flush=True,
            )
            print("[exact-selected-i6] stage=exact-trajectory", flush=True)
            while accepted_updates < MAX_ACCEPTED_UPDATES:
                current_hash = _named_tensor_hash(active_names, active)
                current_parameters = [parameter.detach().clone() for parameter in active]
                current_state = _evaluate(
                    run=run, z=z, record=record, frozen_low_basis=None
                )
                current_metrics = current_state["metrics"]
                stored_current = state_rows[-1]
                replay_columns = (
                    "exact_a_per_dim",
                    "damped_full_burg_per_dim",
                    "a_low90_abs_per_dim",
                    "a_gt1",
                    "m_max",
                    "m_p50",
                    "effective_rank",
                    "frozen_low_energy",
                    "canonical_low_energy",
                )
                current_replay_error = max(
                    abs(float(current_metrics[key]) - float(stored_current[key]))
                    for key in replay_columns
                )
                stored_spectrum = np.array(
                    [
                        float(row["m_eigenvalue"])
                        for row in spectrum_rows
                        if int(row["accepted_update"]) == accepted_updates
                    ],
                    dtype=np.float64,
                )
                spectrum_replay_error = float(
                    np.max(np.abs(current_state["eig"].cpu().numpy() - stored_spectrum))
                )
                if (
                    current_hash != str(stored_current["parameter_hash"])
                    or current_replay_error > 1e-9
                    or spectrum_replay_error > 1e-9
                ):
                    raise RuntimeError("current state/checkpoint replay failed")

                gradients, gradient_meta = _gradient_payload(
                    cfg=cfg,
                    run=run,
                    z=z,
                    record=record,
                    active=active,
                    state=current_state,
                    update=accepted_updates + 1,
                )
                auxiliary = gradients["B"] if selected_arm == "B" else gradients["low"]
                direction, direction_meta = _unit_direction(
                    gradients["A"], auxiliary, active
                )
                slopes = {
                    "A": 0.0 if direction is None else _dot(gradients["A"], direction),
                    "B": 0.0 if direction is None else _dot(gradients["B"], direction),
                    "L_low": 0.0
                    if direction is None
                    else _dot(gradients["low"], direction),
                }
                proposal = accepted_updates + 1
                proposal_record: dict[str, Any] = {
                    "target_update": proposal,
                    "selected_arm": selected_arm,
                    **direction_meta,
                    **{f"slope_{key}": value for key, value in slopes.items()},
                    "current_A": float(current_metrics["exact_a_per_dim"]),
                    "current_B": float(current_metrics["damped_full_burg_per_dim"]),
                    "current_low_energy": float(current_metrics["frozen_low_energy"]),
                    "current_a_gt1": float(current_metrics["a_gt1"]),
                    "gradient_metadata": json.dumps(gradient_meta, sort_keys=True),
                    "selected_alpha": 0.0,
                    "accepted": 0,
                    "failure": "",
                }
                if direction is None or not all(value < 0.0 for value in slopes.values()):
                    proposal_record["failure"] = (
                        "cancellation_gate" if direction is None else "nonnegative_joint_slope"
                    )
                    proposal_rows.append(proposal_record)
                    termination = str(proposal_record["failure"])
                    print(
                        f"[exact-selected-i6] stalled update={proposal} "
                        f"reason={termination}",
                        flush=True,
                    )
                    break

                accepted_payload: dict[str, Any] | None = None
                accepted_hash: str | None = None
                selected_alpha: float | None = None
                try:
                    for alpha in LINE_ALPHAS:
                        endpoint, repeat_errors, endpoint_hash = _line_endpoint(
                            run=run,
                            z=z,
                            record=record,
                            active_names=active_names,
                            active=active,
                            base=current_parameters,
                            direction=direction,
                            alpha=alpha,
                            frozen_low_basis=current_state["current_low_basis"],
                        )
                        gates = _candidate_gates(
                            current=current_metrics,
                            candidate=endpoint["metrics"],
                            slopes=slopes,
                            alpha=alpha,
                            tolerances=tolerances,
                            require_low90=False,
                        )
                        repeat_pass = _repeat_passes(repeat_errors, tolerances)
                        passes = bool(all(gates.values()) and repeat_pass)
                        line_rows.append(
                            {
                                "target_update": proposal,
                                "selected_arm": selected_arm,
                                "alpha": alpha,
                                **endpoint["metrics"],
                                **{f"gate_{key}": int(value) for key, value in gates.items()},
                                **repeat_errors,
                                "endpoint_parameter_hash": endpoint_hash,
                                "passes": int(passes),
                            }
                        )
                        print(
                            f"[exact-selected-i6] update={proposal} alpha={alpha:.7g} "
                            f"pass={int(passes)} A={endpoint['metrics']['exact_a_per_dim']:.7g} "
                            f"B={endpoint['metrics']['damped_full_burg_per_dim']:.7g} "
                            f"Elow={endpoint['metrics']['frozen_low_energy']:.7g} "
                            f"Agt1={endpoint['metrics']['a_gt1']:.7g}",
                            flush=True,
                        )
                        if passes:
                            accepted_payload = endpoint
                            accepted_hash = endpoint_hash
                            selected_alpha = alpha
                            break
                        _set_parameters(active, current_parameters, direction, 0.0)
                        if _named_tensor_hash(active_names, active) != current_hash:
                            raise RuntimeError("line-search restoration failed")
                finally:
                    _set_parameters(active, current_parameters, direction, 0.0)

                if accepted_payload is None or accepted_hash is None or selected_alpha is None:
                    proposal_record["failure"] = "backtracking_exhausted"
                    proposal_rows.append(proposal_record)
                    termination = "backtracking_exhausted"
                    print(
                        f"[exact-selected-i6] stalled update={proposal} "
                        "reason=backtracking_exhausted",
                        flush=True,
                    )
                    break

                _set_parameters(active, current_parameters, direction, selected_alpha)
                committed_hash = _named_tensor_hash(active_names, active)
                if committed_hash != accepted_hash:
                    raise RuntimeError("committed endpoint hash mismatch")
                accepted_updates += 1
                proposal_record["selected_alpha"] = selected_alpha
                proposal_record["accepted"] = 1
                proposal_record["candidate_A"] = float(
                    accepted_payload["metrics"]["exact_a_per_dim"]
                )
                proposal_record["candidate_B"] = float(
                    accepted_payload["metrics"]["damped_full_burg_per_dim"]
                )
                proposal_record["candidate_low_energy"] = float(
                    accepted_payload["metrics"]["frozen_low_energy"]
                )
                proposal_record["candidate_a_gt1"] = float(
                    accepted_payload["metrics"]["a_gt1"]
                )
                proposal_rows.append(proposal_record)
                committed_metrics = dict(accepted_payload["metrics"])
                committed_metrics["parent_frozen_low_energy"] = float(
                    committed_metrics["frozen_low_energy"]
                )
                committed_metrics["frozen_low_energy"] = float(
                    committed_metrics["canonical_low_energy"]
                )
                state_rows.append(
                    _state_row(accepted_updates, committed_metrics, committed_hash)
                )
                spectrum_rows.extend(
                    _spectrum_rows(accepted_updates, accepted_payload["eig"])
                )
                _write_progress_checkpoint(
                    staging=staging,
                    active_names=active_names,
                    active=active,
                    accepted_updates=accepted_updates,
                    selected_arm=selected_arm,
                    state_rows=state_rows,
                    spectrum_rows=spectrum_rows,
                    proposal_rows=proposal_rows,
                    line_rows=line_rows,
                    selection_rows=selection_rows,
                    initial_metrics=initial_metrics,
                    tolerances=tolerances,
                    terminal=False,
                    termination="running",
                    proposal0_payload=proposal0_payload,
                )
                print(
                    f"[exact-selected-i6] accepted={accepted_updates}/{MAX_ACCEPTED_UPDATES} "
                    f"arm={selected_arm} alpha={selected_alpha:.7g} "
                    f"A={accepted_payload['metrics']['exact_a_per_dim']:.7g} "
                    f"B={accepted_payload['metrics']['damped_full_burg_per_dim']:.7g} "
                    f"low90={accepted_payload['metrics']['a_low90_abs_per_dim']:.7g} "
                    f"mmax={accepted_payload['metrics']['m_max']:.7g} "
                    f"elapsed={time.perf_counter() - started:.1f}s checkpoint="
                    f"{staging / 'progress_checkpoint.pt'}",
                    flush=True,
                )
                del gradients, direction, current_state, accepted_payload

        _write_progress_checkpoint(
            staging=staging,
            active_names=active_names,
            active=active,
            accepted_updates=accepted_updates,
            selected_arm=str(selected_arm),
            state_rows=state_rows,
            spectrum_rows=spectrum_rows,
            proposal_rows=proposal_rows,
            line_rows=line_rows,
            selection_rows=selection_rows,
            initial_metrics=initial_metrics,
            tolerances=tolerances,
            terminal=True,
            termination=termination,
            proposal0_payload=proposal0_payload,
        )
        states = pd.DataFrame(state_rows).sort_values("accepted_update")
        spectra = pd.DataFrame(spectrum_rows).sort_values(["accepted_update", "rank"])
        proposals = pd.DataFrame(proposal_rows)
        lines = pd.DataFrame(line_rows)
        selections = pd.DataFrame(selection_rows)
        _write_progress_tables(
            staging,
            state_rows,
            spectrum_rows,
            proposal_rows,
            line_rows,
            selection_rows,
        )
        history_summary, history_rows = _historical_transition_audit(
            state_rows=state_rows,
            proposal_rows=proposal_rows,
            line_rows=line_rows,
            tolerances=tolerances,
        )
        _atomic_csv(staging / "historical_transition_audit.csv", history_rows)
        initial = states.iloc[0]
        final = states.iloc[-1]
        total_a_reduction = float(initial["exact_a_per_dim"] - final["exact_a_per_dim"])
        lower90_reduction = float(
            initial["a_low90_abs_per_dim"] - final["a_low90_abs_per_dim"]
        )
        bulk_fraction = lower90_reduction / max(total_a_reduction, 1e-30)
        state_low90_decrease = -states["a_low90_abs_per_dim"].diff()
        tail_mask = states["accepted_update"].between(81, 100)
        tail_bulk_updates = int(
            (state_low90_decrease.loc[tail_mask] > tolerances["A_low90"]).sum()
        )
        state_a_diffs = states["exact_a_per_dim"].diff().iloc[1:]
        state_high_diffs = states["a_gt1"].diff().iloc[1:]
        success_gates = {
            "one_hundred_accepted_updates": accepted_updates == MAX_ACCEPTED_UPDATES,
            "tail_bulk_activity": tail_bulk_updates >= SUCCESS_TAIL_BULK_UPDATES_MIN,
            "final_a_at_most_0p90": float(final["exact_a_per_dim"]) <= SUCCESS_A_MAX,
            "accepted_a_strictly_decreases": bool(
                (state_a_diffs < -tolerances["A"]).all()
            ),
            "all_historical_a_armijo_transitions": bool(
                history_summary["all_historical_a_armijo_pass"]
            ),
            "final_b_below_initial": float(final["damped_full_burg_per_dim"])
            < float(initial["damped_full_burg_per_dim"]) - tolerances["B"],
            "non_top_only_fraction_at_least_0p25": bulk_fraction
            >= SUCCESS_BULK_FRACTION_MIN,
            "final_p50_above_initial": float(final["m_p50"])
            > float(initial["m_p50"]) + tolerances["m_p50"],
            "final_effective_rank_above_initial": float(final["effective_rank"])
            > float(initial["effective_rank"]) + tolerances["effective_rank"],
            "final_count_lt_1e_4_strictly_lower": int(final["count_lt_1e_4"])
            < int(initial["count_lt_1e_4"]),
            "final_count_lt_1e_2_strictly_lower": int(final["count_lt_1e_2"])
            < int(initial["count_lt_1e_2"]),
            "final_count_lt_0p1_strictly_lower": int(final["count_lt_0p1"])
            < int(initial["count_lt_0p1"]),
            "final_mmax_not_above_initial": float(final["m_max"])
            <= float(initial["m_max"]) + tolerances["m_max"],
            "high_tail_never_increases_beyond_floor": bool(
                (state_high_diffs <= tolerances["A_gt1"]).all()
            ),
            "all_state_values_finite": _numeric_finite(states),
            "all_spectrum_values_finite": _numeric_finite(spectra),
            "all_raw_spectrum_minima_valid": bool(
                states["m_raw_eig_min"].ge(-1e-8).all()
            ),
            "all_exact_a_closures_valid": bool(
                states["a_direct_abs_error"].le(1e-9).all()
                and states["a_trace_abs_error"].le(1e-9).all()
            ),
        }
        scientific_success = bool(all(success_gates.values()))

        print("[exact-selected-i6] stage=serialize-final-checkpoint", flush=True)
        live_final_hash = _named_tensor_hash(active_names, active)
        final_checkpoint = {
            "protocol_id": PROTOCOL_ID,
            "selected_arm": selected_arm,
            "accepted_updates": accepted_updates,
            "termination": termination,
            "active_parameter_hash": live_final_hash,
            "active_model_state": {
                name: parameter.detach().cpu().clone()
                for name, parameter in zip(active_names, active, strict=True)
            },
            "source_weight_index": SOURCE_WEIGHT_INDEX,
            "z_sha256": EXPECTED_Z_SHA256,
            "stored_state_metrics": dict(state_rows[-1]),
        }
        _atomic_torch_save(staging / "final_checkpoint.pt", final_checkpoint)

        print("[exact-selected-i6] stage=disk-final-checkpoint-replay", flush=True)
        disk_checkpoint = torch.load(
            staging / "final_checkpoint.pt", map_location="cpu", weights_only=False
        )
        if (
            disk_checkpoint["protocol_id"] != PROTOCOL_ID
            or int(disk_checkpoint["accepted_updates"]) != accepted_updates
            or str(disk_checkpoint["selected_arm"]) != str(selected_arm)
            or str(disk_checkpoint["termination"]) != termination
            or int(disk_checkpoint["source_weight_index"]) != SOURCE_WEIGHT_INDEX
            or str(disk_checkpoint["z_sha256"]) != EXPECTED_Z_SHA256
            or set(disk_checkpoint["active_model_state"]) != set(active_names)
        ):
            raise RuntimeError("serialized final checkpoint metadata mismatch")
        with torch.no_grad():
            for name in active_names:
                named[name].copy_(disk_checkpoint["active_model_state"][name].to(device=device))
        final_hash = _named_tensor_hash(active_names, active)
        final_state = _evaluate(run=run, z=z, record=record, frozen_low_basis=None)
        final_replay_errors = {
            key: abs(float(final_state["metrics"][key]) - float(final[key]))
            for key in (
                "exact_a_per_dim",
                "damped_full_burg_per_dim",
                "a_low90_abs_per_dim",
                "a_gt1",
                "m_max",
                "m_p50",
                "effective_rank",
                "count_lt_1e_4",
                "count_lt_1e_2",
                "count_lt_0p1",
                "frozen_low_energy",
                "canonical_low_energy",
            )
        }
        final_stored_spectrum = spectra.loc[
            spectra["accepted_update"].eq(accepted_updates)
        ].sort_values("rank")["m_eigenvalue"].to_numpy(dtype=np.float64)
        final_spectrum_replay_error = float(
            np.max(
                np.abs(final_state["eig"].cpu().numpy() - final_stored_spectrum)
            )
        )
        final_replay_pass = bool(
            final_hash == live_final_hash
            and final_hash == str(disk_checkpoint["active_parameter_hash"])
            and final_hash == str(final["parameter_hash"])
            and max(final_replay_errors.values()) <= 1e-9
            and final_spectrum_replay_error <= 1e-9
        )
        if not final_replay_pass:
            raise RuntimeError("final checkpoint replay failed")
        success_gates["final_checkpoint_replay"] = final_replay_pass
        scientific_success = bool(all(success_gates.values()))

        final_checkpoint_file_sha256 = sha256_file(staging / "final_checkpoint.pt")
        final_dependency_matches = _validate_dependencies()
        validity_gates = {
            "accepted_checkpoint_matches": sha256_file(accepted_checkpoint)
            == EXPECTED_CHECKPOINT,
            "iteration3_checkpoint_matches": sha256_file(ITERATION3_CHECKPOINT)
            == EXPECTED_ITERATION3_CHECKPOINT_SHA256,
            "active_parameter_count_matches": active_parameter_count == 11_685_120,
            "selected_arm_well_formed": selected_arm in {"B", "low", "none"},
            "proposal0_payload_persisted": bool(
                (staging / "proposal0_decision.json").is_file()
                and json.loads(
                    (staging / "proposal0_decision.json").read_text(encoding="utf-8")
                )
                == proposal0_payload
            ),
            "state_rows_complete": len(states) == accepted_updates + 1,
            "spectrum_rows_complete": len(spectra) == (accepted_updates + 1) * 512,
            "accepted_history_is_bijective": bool(
                history_summary["accepted_proposal_count_matches_states"]
                and history_summary["accepted_update_sequence_exact"]
                and history_summary["all_historical_transitions_pass"]
            ),
            "final_checkpoint_replays": final_replay_pass,
            "source_snapshot_matches": _normalized_source_sha256(source_snapshot)
            == EXPECTED_NORMALIZED_SOURCE_SHA256,
            "live_source_unchanged": _normalized_source_sha256()
            == EXPECTED_NORMALIZED_SOURCE_SHA256,
            "dependencies_unchanged": all(final_dependency_matches.values()),
            "tables_finite": _numeric_finite(states)
            and _numeric_finite(spectra)
            and _row_numeric_values_finite(proposal_rows)
            and _row_numeric_values_finite(line_rows)
            and _row_numeric_values_finite(selection_rows),
        }
        valid = bool(all(validity_gates.values()))
        decision = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "scientific_success": scientific_success if valid else None,
            "selected_arm": selected_arm,
            "accepted_updates": accepted_updates,
            "termination": termination,
            "initial_metrics": initial_metrics,
            "final_metrics": dict(final_state["metrics"]),
            "total_a_reduction": total_a_reduction,
            "lower90_a_reduction": lower90_reduction,
            "non_top_only_fraction": bulk_fraction,
            "tail_bulk_updates": tail_bulk_updates,
            "historical_transition_summary": history_summary,
            "tolerances": tolerances,
            "success_gates": success_gates,
            "validity_gates": validity_gates,
            "final_replay_errors": final_replay_errors,
            "final_spectrum_replay_error": final_spectrum_replay_error,
            "final_checkpoint_file_sha256": final_checkpoint_file_sha256,
            "elapsed_sec": time.perf_counter() - started,
        }
        _atomic_json(staging / "decision.json", decision)
        if not valid:
            raise RuntimeError("refusing to finalize invalid trajectory")
        _plot(staging, states, spectra)
        artifact_manifest = _manifest(staging, source_snapshot)
        _atomic_json(staging / "artifact_manifest.json", artifact_manifest)
        finalized = {
            "protocol_id": PROTOCOL_ID,
            "status": "complete_awaiting_independent_review",
            "valid": valid,
            "scientific_success": scientific_success,
            "selected_arm": selected_arm,
            "accepted_updates": accepted_updates,
            "termination": termination,
            "decision_sha256": sha256_file(staging / "decision.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        _atomic_json(staging / "FINALIZED.json", finalized)
        print(
            f"[exact-selected-i6] complete valid={valid} success={scientific_success} "
            f"arm={selected_arm} accepted={accepted_updates} termination={termination} "
            f"A={float(initial['exact_a_per_dim']):.7g}->{float(final['exact_a_per_dim']):.7g} "
            f"bulk_fraction={bulk_fraction:.4g} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        os.replace(staging, final_output)
        (final_output / "INCOMPLETE").unlink()
        print(f"[exact-selected-i6] published output={final_output}", flush=True)
    except Exception:
        print("[exact-selected-i6] FAILED; resumable staging retained", flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
