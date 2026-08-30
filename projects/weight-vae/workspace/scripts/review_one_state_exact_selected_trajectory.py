from __future__ import annotations

import os
import sys

# Direct path execution otherwise lets scripts/inspect shadow stdlib modules.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _SCRIPT_DIR:
    sys.path.pop(0)
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import argparse
import copy
import hashlib
import json
import math
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    decode_weights,
    encode_weights,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _hessian_via_batched_hvp,
    _latent_hvp,
    _task_loss_from_flat,
    _task_set_for_record,
)
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _load_run,
    _probe_cfg,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_selected_trajectory_i6_v1"
REVIEW_PROTOCOL_ID = "one_state_exact_selected_trajectory_i6_independent_review_v1"
GEOMETRY_REPLAY_SCHEMA = "one_state_exact_selected_trajectory_i6_geometry_replay_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration6_exact_selected_trajectory_production"
PRODUCER_SOURCE = ROOT / "scripts/run_one_state_exact_selected_trajectory.py"
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_6_exact_selected_trajectory/protocol.md"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_6_frozen_dependency_manifest.json"
ITERATION3 = OUTPUT_ROOT / "iteration3_p32_unit_common_production"
ITERATION3_CHECKPOINT = ITERATION3 / "final_checkpoint.pt"
ITERATION3_STATES = ITERATION3 / "state_metrics.csv"
ITERATION3_SPECTRA = ITERATION3 / "state_spectra.csv"
ITERATION3_DECISION = ITERATION3 / "decision.json"
ITERATION3_REVIEW = ITERATION3 / "independent_review.json"
ITERATION5 = OUTPUT_ROOT / "iteration5_exact_a_low_common_production"
ITERATION5_ENDPOINTS = ITERATION5 / "endpoint_metrics.csv"
ITERATION5_DECISION = ITERATION5 / "decision.json"
ITERATION5_REVIEW = ITERATION5 / "independent_review.json"
ACTIVE_PARAMETERS = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
)
STATE_BANK = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_fixed_state_probe_variance_h2048/state_bank.csv"
)

EXPECTED_PRODUCER_SHA256 = (
    "632cae7129452e8acba129e12340cf5bdab11d55fbd0115dc9275eb72eb6b731"
)
EXPECTED_NORMALIZED_PRODUCER_SHA256 = (
    "20ad776cc9a7e2d513aab11ee527bd284e679a4d80558f5919f00813d9bb93b3"
)
EXPECTED_PROTOCOL_SHA256 = (
    "6216a14d445cc996de7dcdda16c28fda382d2d1493df4eebac188a8b36a68e90"
)
EXPECTED_DEPENDENCY_MANIFEST_SHA256 = (
    "042d7605633a76c408cd86c1dc01150d9bba3210da0c890ad6f62e78ba99c260"
)
EXPECTED_ITERATION3_CHECKPOINT_SHA256 = (
    "ccafbad0e0fcf25aaabffe0a406558937d1b448ac06c80b7c40a6637209dc5ad"
)
EXPECTED_ACCEPTED_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "31abec6439564c392a99b6c82dd1386aed714cb1b2a1ce1c4709088686a0caff"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"

DIMENSION = 512
ACTIVE_PARAMETER_COUNT = 11_685_120
SOURCE_WEIGHT_INDEX = 378
STATE_POSITION = 2
EPSILON = 1e-4
LOW_THRESHOLD = 0.1
HESSIAN_CHUNK_SIZE = 64
GRADIENT_BLOCK_SIZE = 64
TARGET_NORM = 0.04892722657548397
CANCELLATION_NORM_MIN = 0.5
SELECTION_ALPHA = 1.0 / 128.0
ARMIJO_C1 = 1e-4
MAX_ACCEPTED_UPDATES = 100
OLD_BETA = 22.536727828943093
LINE_ALPHAS = tuple(2.0**-index for index in range(12))
REPEAT_MAX_ERROR = 1e-10
BURG_GRADIENT_FORMULA_MAX_ERROR = 5e-6
DIRECTION_IDENTITY_ATOL = 2e-9
DIRECTION_IDENTITY_RTOL = 5e-7
DIRECTION_RADIUS_ATOL = 5e-9
GPU_GRADIENT_REPLAY_ATOL = 2e-7
GPU_GRADIENT_REPLAY_RTOL = 5e-7
GPU_GEOMETRY_REPLAY_ATOL = 1e-8
GPU_GEOMETRY_REPLAY_RTOL = 1e-8
RETAINED_MEMORY_RANGE_MAX = 64 * 1024 * 1024
BLOCK_PEAK_RANGE_MAX = 128 * 1024 * 1024
SUCCESS_A_MAX = 0.90
SUCCESS_BULK_FRACTION_MIN = 0.25
SUCCESS_TAIL_BULK_UPDATES_MIN = 4
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

EXPECTED_MANIFEST_ARTIFACTS = {
    "arm_selection.csv",
    "decision.json",
    "exact_selected_spectra.png",
    "exact_selected_trajectory.png",
    "executed_source_snapshot.py",
    "final_checkpoint.pt",
    "frozen_dependency_manifest_snapshot.json",
    "historical_transition_audit.csv",
    "line_search.csv",
    "progress_checkpoint.pt",
    "proposal0_decision.json",
    "proposal_diagnostics.csv",
    "protocol_snapshot.md",
    "resolved_config.json",
    "state_metrics.csv",
    "state_spectra.csv",
}
UNMANIFESTED_FILES = {
    "artifact_manifest.json",
    "FINALIZED.json",
    "independent_review.json",
    "independent_review.log",
    "run.log",
}
OPTIONAL_TEXT_MISSING_COLUMNS = {"failure"}


class ReviewError(RuntimeError):
    pass


class Review:
    def __init__(self) -> None:
        self.gates: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.errors: list[str] = []
        self.limitations: list[str] = []
        self.warnings: list[str] = []

    def gate(self, name: str, value: Any) -> bool:
        if name in self.gates:
            raise KeyError(f"duplicate gate: {name}")
        result = bool(value)
        self.gates[name] = result
        return result

    def phase(self, name: str, function: Callable[[], None]) -> bool:
        try:
            function()
        except Exception as error:
            self.gates[f"{name}_completed"] = False
            self.errors.append(f"{name}: {type(error).__name__}: {error}")
            self.details[f"{name}_traceback"] = traceback.format_exc(limit=12)
            return False
        self.gates[f"{name}_completed"] = True
        return True


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ReviewError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _normalized_producer_sha256(path: Path) -> str:
    masked = (
        "EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    normalized: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        prefix = next((item for item in masked if line.startswith(item)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} is not a JSON object")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(path)


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ReviewError(f"nonfinite numeric value: {value!r}")
    return result


def _int(value: Any) -> int:
    number = _float(value)
    result = int(number)
    if number != float(result):
        raise ReviewError(f"nonintegral value: {value!r}")
    return result


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = _float(value)
        if number in (0.0, 1.0):
            return bool(int(number))
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ReviewError(f"invalid boolean value: {value!r}")


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _close(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> bool:
    try:
        return bool(
            np.isclose(
                float(left), float(right), atol=atol, rtol=rtol, equal_nan=False
            )
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _equivalent(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return (
            isinstance(left, (bool, np.bool_))
            and isinstance(right, (bool, np.bool_))
            and bool(left) == bool(right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    left_sequence = isinstance(left, Sequence) and not isinstance(left, (str, bytes))
    right_sequence = isinstance(right, Sequence) and not isinstance(right, (str, bytes))
    if left_sequence or right_sequence:
        if not left_sequence or not right_sequence or len(left) != len(right):
            return False
        return all(
            _equivalent(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    try:
        return _close(left, right, atol=atol, rtol=rtol)
    except Exception:
        return left == right


def _frame_numeric_finite(frame: pd.DataFrame, *, excluded: set[str]) -> bool:
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            return False
    return True


def _cell_equivalent(
    expected: Any,
    observed: Any,
    *,
    empty_text_is_missing: bool = False,
) -> bool:
    def missing(value: Any) -> bool:
        return value is None or (empty_text_is_missing and value == "") or (
            isinstance(value, (float, np.floating)) and math.isnan(float(value))
        )

    expected_missing = missing(expected)
    observed_missing = missing(observed)
    if expected_missing or observed_missing:
        return expected_missing and observed_missing
    if isinstance(expected, (str, bytes)):
        return str(observed) == str(expected)
    if isinstance(expected, (bool, np.bool_)):
        try:
            return _bool(observed) is bool(expected)
        except ReviewError:
            return False
    return _close(expected, observed, atol=1e-12, rtol=1e-12)


def _checkpoint_rows_match_csv(
    records: Sequence[Mapping[str, Any]], frame: pd.DataFrame
) -> bool:
    expected = pd.DataFrame(list(records))
    if expected.empty and frame.empty:
        return True
    if list(expected.columns) != list(frame.columns) or len(expected) != len(frame):
        return False
    for position in range(len(expected)):
        for column in expected.columns:
            if not _cell_equivalent(
                expected.iloc[position][column],
                frame.iloc[position][column],
                empty_text_is_missing=column in OPTIONAL_TEXT_MISSING_COLUMNS,
            ):
                return False
    return True


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    _require(data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not PNG")
    _require(data[12:16] == b"IHDR", f"{path.name} has no IHDR")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _spectral_metrics(eigenvalues: np.ndarray) -> dict[str, float]:
    eig = np.asarray(eigenvalues, dtype=np.float64)
    _require(eig.shape == (DIMENSION,), f"expected {DIMENSION} eigenvalues")
    _require(np.isfinite(eig).all(), "spectrum contains nonfinite values")
    _require(bool(np.all(np.diff(eig) >= -1e-12)), "spectrum is not ordered")
    contribution = np.square(eig - 1.0)
    total_a = max(float(contribution.sum()), 1e-30)
    trace = float(eig.sum())
    square_sum = max(float(np.square(eig).sum()), 1e-30)
    r_eig = (eig + EPSILON) / (1.0 + EPSILON)
    _require(bool(np.all(r_eig > 0.0)), "damped spectrum is not positive")
    g_eig = (1.0 - 1.0 / r_eig) / (DIMENSION * (1.0 + EPSILON))
    quantiles = np.quantile(
        eig,
        [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
        method="linear",
    )
    low_mask = eig < LOW_THRESHOLD
    _require(bool(low_mask.any()), "low eigenspace is empty")
    a = float(contribution.mean())
    burg_trace = float(r_eig.mean())
    logdet = float(np.log(r_eig).mean())
    b = burg_trace - logdet - 1.0
    lower90_stop = int(math.floor(0.9 * DIMENSION))
    return {
        "exact_a_per_dim": a,
        "a_constant_term": 1.0,
        "a_linear_trace_term": -2.0 * float(eig.mean()),
        "a_quartic_term": float(np.square(eig).mean()),
        "trace_m_per_dim": float(eig.mean()),
        "damped_full_burg_per_dim": b,
        "burg_trace_r_term": burg_trace,
        "burg_neg_logdet_r_term": -logdet,
        "logdet_r_per_dim": logdet,
        "m_min": float(quantiles[0]),
        "m_p01": float(quantiles[1]),
        "m_p10": float(quantiles[2]),
        "m_p25": float(quantiles[3]),
        "m_p50": float(quantiles[4]),
        "m_p75": float(quantiles[5]),
        "m_p90": float(quantiles[6]),
        "m_p95": float(quantiles[7]),
        "m_p99": float(quantiles[8]),
        "m_max": float(quantiles[9]),
        "m_lt_1e_4_fraction": float(np.mean(eig < 1e-4)),
        "m_lt_0p01_fraction": float(np.mean(eig < 1e-2)),
        "m_lt_0p1_fraction": float(np.mean(low_mask)),
        "m_lt_0p5_fraction": float(np.mean(eig < 0.5)),
        "m_near_1_10pct_fraction": float(np.mean(np.abs(eig - 1.0) <= 0.1)),
        "m_gt_1_fraction": float(np.mean(eig > 1.0)),
        "m_gt_2_fraction": float(np.mean(eig > 2.0)),
        "a_from_m_lt_0p1_share": float(contribution[low_mask].sum() / total_a),
        "a_from_m_gt_1_share": float(contribution[eig > 1.0].sum() / total_a),
        "burg_matrix_gradient_norm": float(np.linalg.norm(g_eig)),
        "burg_matrix_gradient_eig_min": float(np.min(g_eig)),
        "burg_matrix_gradient_eig_p50": float(np.sort(g_eig)[(DIMENSION - 1) // 2]),
        "burg_matrix_gradient_eig_max": float(np.max(g_eig)),
        "true_objective": a + OLD_BETA * b,
        "a_direct_matrix": a,
        "a_trace_closure": 1.0 - 2.0 * float(eig.mean()) + float(np.square(eig).mean()),
        "effective_rank": trace * trace / square_sum,
        "effective_rank_fraction": trace * trace / (square_sum * DIMENSION),
        "li_gap_per_dim": float(np.mean(np.square(np.sqrt(eig) - 1.0))),
        "a_low90_abs_per_dim": float(contribution[:lower90_stop].sum() / DIMENSION),
        "a_low_lt_0p1_abs_per_dim": float(contribution[low_mask].sum() / DIMENSION),
        "a_high_gt_1_abs_per_dim": float(contribution[eig > 1.0].sum() / DIMENSION),
        "a_top1_share": float(contribution[-1] / total_a),
        "a_top10_share": float(contribution[-10:].sum() / total_a),
        "top1_trace_share": float(eig[-1] / max(trace, 1e-30)),
        "top10_trace_share": float(eig[-10:].sum() / max(trace, 1e-30)),
        "frozen_low_energy": float(eig[low_mask].mean()),
        "canonical_low_energy": float(eig[low_mask].mean()),
        "count_lt_1e_4": float(np.sum(eig < 1e-4)),
        "count_lt_1e_2": float(np.sum(eig < 1e-2)),
        "count_lt_0p1": float(np.sum(low_mask)),
        "a_gt1": float(contribution[eig > 1.0].sum() / DIMENSION),
        "low_count": float(np.sum(low_mask)),
    }


def _validate_spectrum_tables(
    states: pd.DataFrame, spectra: pd.DataFrame
) -> dict[str, Any]:
    _require(not states.empty, "state table is empty")
    required_state = {
        "accepted_update",
        "parameter_hash",
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
        "parent_frozen_low_energy",
        "m_raw_eig_min",
        "a_direct_abs_error",
        "a_trace_abs_error",
        "low_basis_hash",
    }
    _require(required_state.issubset(states.columns), "state table columns missing")
    _require(
        {"accepted_update", "rank", "m_eigenvalue", "a_contribution"}.issubset(
            spectra.columns
        ),
        "spectrum table columns missing",
    )
    _require(
        _frame_numeric_finite(
            states, excluded={"parameter_hash", "low_basis_hash"}
        ),
        "state table contains nonfinite/nonnumeric values",
    )
    _require(
        _frame_numeric_finite(spectra, excluded=set()),
        "spectrum table contains nonfinite/nonnumeric values",
    )
    updates = [_int(value) for value in states["accepted_update"]]
    _require(updates == list(range(len(states))), "state update sequence is not exact")
    _require(
        len(spectra) == len(states) * DIMENSION,
        "spectrum row count does not match states x dimension",
    )
    parameter_hashes = [_text(value) for value in states["parameter_hash"]]
    _require(all(_is_sha256(value) for value in parameter_hashes), "bad state hash")
    _require(
        len(set(parameter_hashes)) == len(parameter_hashes),
        "accepted state parameter hashes are not globally unique",
    )

    worst_metric_error = 0.0
    worst_contribution_error = 0.0
    per_update: list[dict[str, Any]] = []
    integral_metrics = {"count_lt_1e_4", "count_lt_1e_2", "count_lt_0p1", "low_count"}
    for position, update in enumerate(updates):
        state = states.iloc[position]
        block = spectra.loc[spectra["accepted_update"].eq(update)].sort_values("rank")
        ranks = [_int(value) for value in block["rank"]]
        _require(ranks == list(range(DIMENSION)), f"bad rank grid at update {update}")
        eig = block["m_eigenvalue"].to_numpy(dtype=np.float64)
        expected = _spectral_metrics(eig)
        contribution = block["a_contribution"].to_numpy(dtype=np.float64)
        contribution_error = float(np.max(np.abs(contribution - np.square(eig - 1.0))))
        _require(
            contribution_error <= 1e-9,
            f"a_contribution mismatch at update {update}: {contribution_error}",
        )
        worst_contribution_error = max(worst_contribution_error, contribution_error)
        errors: dict[str, float] = {}
        for metric, reference in expected.items():
            _require(metric in states.columns, f"missing state metric {metric}")
            observed = _float(state[metric])
            if metric in integral_metrics:
                _require(observed == reference, f"{metric} mismatch at update {update}")
                error = abs(observed - reference)
            else:
                error = abs(observed - reference)
                _require(
                    _close(observed, reference, atol=2e-9, rtol=2e-9),
                    f"{metric} mismatch at update {update}: {observed} vs {reference}",
                )
            errors[metric] = error
            worst_metric_error = max(worst_metric_error, error)
        raw_min = _float(state["m_raw_eig_min"])
        _require(raw_min >= -1e-8, f"raw negative eigenvalue too large at {update}")
        _require(
            _close(max(raw_min, 0.0), eig[0], atol=2e-9, rtol=2e-9),
            f"raw/clamped minimum mismatch at update {update}",
        )
        _require(
            _float(state["a_direct_abs_error"]) <= 1e-9
            and _float(state["a_trace_abs_error"]) <= 1e-9,
            f"exact-A closure failed at update {update}",
        )
        _require(
            _float(state["low_basis_orthogonality_max_abs"]) <= 1e-10,
            f"low-basis orthogonality failed at update {update}",
        )
        _require(
            _float(state["low_basis_eigen_residual_relative"]) <= 1e-10,
            f"low-basis residual failed at update {update}",
        )
        _require(
            _float(state["helper_eigenvalue_max_abs_error"]) <= 1e-10,
            f"helper eigenvalue mismatch at update {update}",
        )
        _require(_is_sha256(_text(state["low_basis_hash"])), "bad low-basis hash")
        per_update.append(
            {
                "accepted_update": update,
                "max_spectrum_metric_abs_error": max(errors.values()),
                "a_contribution_max_abs_error": contribution_error,
            }
        )
    return {
        "state_count": len(states),
        "spectrum_rows": len(spectra),
        "worst_spectrum_metric_abs_error": worst_metric_error,
        "worst_a_contribution_abs_error": worst_contribution_error,
        "per_update": per_update,
    }


def _line_metric_consistency(row: Mapping[str, Any]) -> None:
    a = _float(row["exact_a_per_dim"])
    _require(
        _close(
            a,
            _float(row["a_constant_term"])
            + _float(row["a_linear_trace_term"])
            + _float(row["a_quartic_term"]),
            atol=1e-9,
            rtol=1e-9,
        ),
        "line exact-A trace identity failed",
    )
    _require(
        _close(a, row["a_direct_matrix"], atol=1e-9, rtol=1e-9)
        and _close(a, row["a_trace_closure"], atol=1e-9, rtol=1e-9),
        "line exact-A closure values failed",
    )
    b = _float(row["damped_full_burg_per_dim"])
    _require(
        _close(
            b,
            _float(row["burg_trace_r_term"])
            + _float(row["burg_neg_logdet_r_term"])
            - 1.0,
            atol=1e-9,
            rtol=1e-9,
        )
        and _close(
            _float(row["burg_neg_logdet_r_term"]),
            -_float(row["logdet_r_per_dim"]),
            atol=1e-9,
            rtol=1e-9,
        ),
        "line Burg identity failed",
    )
    _require(
        _close(row["a_gt1"], row["a_high_gt_1_abs_per_dim"], atol=1e-12, rtol=1e-12),
        "line high-tail aliases disagree",
    )
    _require(
        _int(row["count_lt_0p1"]) == _int(row["low_count"]),
        "line low count aliases disagree",
    )
    for count, fraction in (
        ("count_lt_1e_4", "m_lt_1e_4_fraction"),
        ("count_lt_1e_2", "m_lt_0p01_fraction"),
        ("count_lt_0p1", "m_lt_0p1_fraction"),
    ):
        _require(
            _close(_int(row[count]) / DIMENSION, row[fraction], atol=1e-12, rtol=0.0),
            f"line {count}/{fraction} mismatch",
        )
    _require(
        _float(row["a_direct_abs_error"]) <= 1e-9
        and _float(row["a_trace_abs_error"]) <= 1e-9
        and _float(row["m_raw_eig_min"]) >= -1e-8,
        "line closure or raw spectrum gate failed",
    )


def _objective_values(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "A": _float(row["exact_a_per_dim"]),
        "B": _float(row["damped_full_burg_per_dim"]),
        "L_low": -_float(row["frozen_low_energy"]),
    }


def _candidate_gates(
    *,
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    slopes: Mapping[str, float],
    alpha: float,
    tolerances: Mapping[str, float],
    require_low90: bool,
) -> dict[str, bool]:
    current_objectives = _objective_values(current)
    candidate_objectives = _objective_values(candidate)
    gates: dict[str, bool] = {}
    for name in ("A", "B", "L_low"):
        gates[f"{name}_slope_negative"] = slopes[name] < 0.0
        gates[f"{name}_armijo"] = candidate_objectives[name] <= (
            current_objectives[name] + ARMIJO_C1 * alpha * slopes[name]
        )
        gates[f"{name}_actual_decrease"] = (
            current_objectives[name] - candidate_objectives[name] > tolerances[name]
        )
    gates["high_tail_nonincrease"] = _float(candidate["a_gt1"]) <= (
        _float(current["a_gt1"]) + tolerances["A_gt1"]
    )
    gates["low90_decreases"] = (
        not require_low90
        or _float(current["a_low90_abs_per_dim"])
        - _float(candidate["a_low90_abs_per_dim"])
        > tolerances["A_low90"]
    )
    gates["exact_a_closes"] = (
        _float(candidate["a_direct_abs_error"]) <= 1e-9
        and _float(candidate["a_trace_abs_error"]) <= 1e-9
    )
    gates["spectrum_finite_nonnegative"] = _float(candidate["m_raw_eig_min"]) >= -1e-8
    required_finite = {
        "task_loss",
        "exact_a_per_dim",
        "damped_full_burg_per_dim",
        "frozen_low_energy",
        "a_low90_abs_per_dim",
        "a_gt1",
        "m_max",
        "m_p50",
        "effective_rank",
        "count_lt_1e_4",
        "count_lt_1e_2",
        "count_lt_0p1",
        "a_direct_abs_error",
        "a_trace_abs_error",
        "m_raw_eig_min",
    }
    try:
        gates["metrics_finite"] = all(
            key in candidate and math.isfinite(float(candidate[key]))
            for key in required_finite
        )
    except (TypeError, ValueError, OverflowError):
        gates["metrics_finite"] = False
    return gates


def _repeat_passes(row: Mapping[str, Any], tolerances: Mapping[str, float]) -> bool:
    limits = {
        "repeat_exact_a_per_dim_abs_error": tolerances["A"],
        "repeat_damped_full_burg_per_dim_abs_error": tolerances["B"],
        "repeat_frozen_low_energy_abs_error": tolerances["L_low"],
        "repeat_a_low90_abs_per_dim_abs_error": tolerances["A_low90"],
        "repeat_a_gt1_abs_error": tolerances["A_gt1"],
        "repeat_m_max_abs_error": tolerances["m_max"],
        "repeat_m_p50_abs_error": tolerances["m_p50"],
        "repeat_effective_rank_abs_error": tolerances["effective_rank"],
    }
    exact = {
        "repeat_count_lt_1e_4_abs_error",
        "repeat_count_lt_1e_2_abs_error",
        "repeat_count_lt_0p1_abs_error",
    }
    try:
        if _float(row["repeat_parameter_hash_unchanged"]) != 1.0:
            return False
        if any(_float(row[key]) != 0.0 for key in exact):
            return False
        if any(_float(row[key]) > limit for key, limit in limits.items()):
            return False
        repeat_columns = {
            key
            for key in row.keys()
            if key.startswith("repeat_")
            and key not in set(limits)
            and key not in exact
            and key not in {"repeat_parameter_hash_unchanged", "repeat_pass"}
        }
        return all(_float(row[key]) <= REPEAT_MAX_ERROR for key in repeat_columns)
    except (KeyError, ReviewError, TypeError, ValueError):
        return False


def _direction_identity(row: Mapping[str, Any], auxiliary: str) -> dict[str, float]:
    norm_a = _float(row["gradient_a_norm"])
    norm_x = _float(row["gradient_x_norm"])
    cosine = _float(row["gradient_cosine"])
    source = _float(row["unit_common_source_norm"])
    expected_source = math.sqrt(max(0.0, 2.0 + 2.0 * cosine))
    _require(
        _close(
            source,
            expected_source,
            atol=DIRECTION_IDENTITY_ATOL,
            rtol=DIRECTION_IDENTITY_RTOL,
        ),
        f"{auxiliary} normalized-common source norm identity failed",
    )
    expected_a_slope = -TARGET_NORM * norm_a * (1.0 + cosine) / expected_source
    expected_x_slope = -TARGET_NORM * norm_x * (1.0 + cosine) / expected_source
    _require(
        _close(
            row["slope_A"],
            expected_a_slope,
            atol=DIRECTION_IDENTITY_ATOL,
            rtol=DIRECTION_IDENTITY_RTOL,
        ),
        f"{auxiliary} A slope identity failed",
    )
    slope_column = "slope_B" if auxiliary == "B" else "slope_L_low"
    _require(
        _close(
            row[slope_column],
            expected_x_slope,
            atol=DIRECTION_IDENTITY_ATOL,
            rtol=DIRECTION_IDENTITY_RTOL,
        ),
        f"{auxiliary} own slope identity failed",
    )
    return {
        "expected_source_norm": expected_source,
        "expected_slope_A": expected_a_slope,
        "expected_slope_X": expected_x_slope,
    }


def _validate_direction_record(
    row: Mapping[str, Any],
    *,
    auxiliary: str,
) -> dict[str, Any]:
    norm_a = _float(row["gradient_a_norm"])
    norm_x = _float(row["gradient_x_norm"])
    cosine = _float(row["gradient_cosine"])
    _require(norm_a > 0.0 and norm_x > 0.0, f"{auxiliary} gradient norm is not positive")
    _require(-1.0 <= cosine <= 1.0, f"{auxiliary} gradient cosine is outside [-1,1]")
    expected_source = math.sqrt(2.0 + 2.0 * cosine)
    observed_source = _float(row["unit_common_source_norm"])
    _require(
        _close(
            observed_source,
            expected_source,
            atol=DIRECTION_IDENTITY_ATOL,
            rtol=DIRECTION_IDENTITY_RTOL,
        ),
        f"{auxiliary} normalized-common source norm identity failed",
    )
    expected_cancellation_pass = expected_source >= CANCELLATION_NORM_MIN
    realized_cancellation_pass = observed_source >= CANCELLATION_NORM_MIN
    _require(
        realized_cancellation_pass is expected_cancellation_pass,
        f"{auxiliary} FP32 source norm changes the cancellation side",
    )
    _require(
        _bool(row["cancellation_gate_pass"]) is realized_cancellation_pass,
        f"{auxiliary} cancellation decision mismatch",
    )
    if expected_cancellation_pass:
        _require(
            _close(
                row["direction_norm"],
                TARGET_NORM,
                atol=DIRECTION_RADIUS_ATOL,
                rtol=0.0,
            ),
            f"{auxiliary} direction radius mismatch",
        )
        direction = _direction_identity(row, auxiliary)
    else:
        _require(_float(row["direction_norm"]) == 0.0, f"{auxiliary} cancelled direction nonzero")
        for slope in ("slope_A", "slope_B", "slope_L_low"):
            _require(_float(row[slope]) == 0.0, f"{auxiliary} cancelled direction has nonzero slope")
        direction = {
            "expected_source_norm": expected_source,
            "expected_slope_A": 0.0,
            "expected_slope_X": 0.0,
        }
    return {
        "cancellation_gate_pass": expected_cancellation_pass,
        **direction,
    }


def _validate_gradient_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    _require(_int(metadata["basis_count"]) == DIMENSION, "gradient basis count mismatch")
    _require(_int(metadata["block_count"]) == 8, "gradient block count mismatch")
    _require(_int(metadata["block_size"]) == 64, "gradient block size mismatch")
    retained = [_int(value) for value in metadata["retained_memory_by_block_bytes"]]
    peaks = [_int(value) for value in metadata["peak_memory_by_block_bytes"]]
    _require(len(retained) == 8 and len(peaks) == 8, "gradient memory trace length mismatch")
    _require(
        all(value >= 0 for value in retained)
        and all(value >= 0 for value in peaks)
        and all(peak >= live for peak, live in zip(peaks, retained, strict=True)),
        "gradient memory trace is invalid",
    )
    retained_range = max(retained) - min(retained)
    peak_range = max(peaks) - min(peaks)
    _require(
        _int(metadata["retained_memory_range_bytes"]) == retained_range,
        "retained memory range mismatch",
    )
    _require(
        _int(metadata["block_peak_range_bytes"]) == peak_range,
        "peak memory range mismatch",
    )
    expected_memory_gate = (
        retained_range <= RETAINED_MEMORY_RANGE_MAX
        and peak_range <= BLOCK_PEAK_RANGE_MAX
    )
    _require(
        _bool(metadata["memory_gate_pass"]) is expected_memory_gate,
        "gradient memory gate mismatch",
    )
    edge = min(2, len(retained))
    _require(
        _close(
            metadata["memory_early_median_bytes"],
            float(np.median(retained[:edge])),
            atol=0.0,
            rtol=0.0,
        )
        and _close(
            metadata["memory_late_median_bytes"],
            float(np.median(retained[-edge:])),
            atol=0.0,
            rtol=0.0,
        ),
        "gradient memory medians mismatch",
    )
    _require(_float(metadata["elapsed_sec"]) >= 0.0, "gradient elapsed time is invalid")
    payload_valid = expected_memory_gate
    for name in ("A", "B", "low"):
        unused = _int(metadata[f"{name}_unused_parameter_tensors_all_blocks"])
        fp64 = _bool(metadata[f"{name}_all_accumulators_float64"])
        _require(
            unused == 0,
            f"{name} has globally unused active tensors",
        )
        _require(
            fp64,
            f"{name} accumulator dtype gate failed",
        )
        payload_valid = payload_valid and unused == 0 and fp64
    for key in ("k_a_norm", "k_b_norm", "k_low_norm"):
        positive = _float(metadata[key]) > 0.0
        _require(positive, f"{key} is not positive")
        payload_valid = payload_valid and positive
    burg_error = _float(metadata["burg_gradient_formula_max_abs_error"])
    _require(
        burg_error <= BURG_GRADIENT_FORMULA_MAX_ERROR,
        "Burg gradient formula cross-check failed",
    )
    payload_valid = payload_valid and burg_error <= BURG_GRADIENT_FORMULA_MAX_ERROR
    _require(
        _bool(metadata["gradient_payload_valid"]) is payload_valid,
        "producer gradient payload validity flag mismatch",
    )
    return {
        "retained_memory_range_bytes": retained_range,
        "block_peak_range_bytes": peak_range,
        "burg_gradient_formula_max_abs_error": burg_error,
    }


def _validate_proposal0(
    *,
    states: pd.DataFrame,
    selections: pd.DataFrame,
    payload: Mapping[str, Any],
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    _require(len(selections) == 2, "proposal-0 must contain exactly two arms")
    _require(list(selections["arm"].astype(str)) == ["B", "low"], "arm order/grid mismatch")
    base = states.iloc[0]
    direction_payload = payload.get("direction_metadata")
    _require(isinstance(direction_payload, Mapping), "missing direction metadata")
    independently_eligible: list[str] = []
    arm_details: dict[str, Any] = {}
    for _, row in selections.iterrows():
        arm = str(row["arm"])
        _require(_close(row["alpha"], SELECTION_ALPHA, atol=0.0, rtol=0.0), "bad selection alpha")
        meta = direction_payload.get(arm)
        _require(isinstance(meta, Mapping), f"missing direction payload for {arm}")
        for key in (
            "gradient_a_norm",
            "gradient_x_norm",
            "gradient_cosine",
            "unit_common_source_norm",
            "direction_norm",
        ):
            _require(_close(row[key], meta[key], atol=1e-12, rtol=1e-12), f"{arm} metadata mismatch")
        _require(
            _bool(row["cancellation_gate_pass"]) is _bool(meta["cancellation_gate_pass"]),
            f"{arm} cancellation payload mismatch",
        )
        direction_details = _validate_direction_record(row, auxiliary=arm)
        cancellation_pass = bool(direction_details["cancellation_gate_pass"])
        if not cancellation_pass:
            _require(not _bool(row["eligible"]), f"cancelled {arm} marked eligible")
            _require(_text(row["failure"]) == "cancellation_gate", f"bad {arm} failure")
            arm_details[arm] = {
                "eligible": False,
                "cancelled": True,
                **direction_details,
            }
            continue
        _line_metric_consistency(row)
        slopes = {
            "A": _float(row["slope_A"]),
            "B": _float(row["slope_B"]),
            "L_low": _float(row["slope_L_low"]),
        }
        gates = _candidate_gates(
            current=base,
            candidate=row,
            slopes=slopes,
            alpha=SELECTION_ALPHA,
            tolerances=tolerances,
            require_low90=True,
        )
        for name, expected in gates.items():
            _require(
                _bool(row[f"gate_{name}"]) is expected,
                f"{arm} stored gate_{name} mismatch",
            )
        repeat_pass = _repeat_passes(row, tolerances)
        _require(_bool(row["repeat_pass"]) is repeat_pass, f"{arm} repeat gate mismatch")
        eligible = all(gates.values()) and repeat_pass
        _require(_bool(row["eligible"]) is eligible, f"{arm} eligibility mismatch")
        _require(
            _close(
                row["low90_decrease"],
                _float(base["a_low90_abs_per_dim"]) - _float(row["a_low90_abs_per_dim"]),
                atol=1e-12,
                rtol=1e-12,
            ),
            f"{arm} low90 decrease mismatch",
        )
        _require(
            _close(
                row["a_decrease"],
                _float(base["exact_a_per_dim"]) - _float(row["exact_a_per_dim"]),
                atol=1e-12,
                rtol=1e-12,
            ),
            f"{arm} A decrease mismatch",
        )
        _require(_is_sha256(_text(row["endpoint_parameter_hash"])), f"bad {arm} endpoint hash")
        if eligible:
            independently_eligible.append(arm)
        arm_details[arm] = {
            "eligible": eligible,
            "cancelled": False,
            "gates": gates,
            "repeat_pass": repeat_pass,
            **direction_details,
        }
    _require(
        list(payload.get("eligible_arms", [])) == independently_eligible,
        "proposal-0 eligible-arm list mismatch",
    )
    if independently_eligible:
        rows = [
            selections.loc[selections["arm"].eq(arm)].iloc[0]
            for arm in independently_eligible
        ]
        chosen = sorted(
            rows,
            key=lambda row: (
                -_float(row["low90_decrease"]),
                -_float(row["a_decrease"]),
                str(row["arm"]),
            ),
        )[0]
        selected = str(chosen["arm"])
    else:
        selected = "none"
    _require(payload.get("selected_arm") == selected, "proposal-0 arm selection mismatch")
    return {
        "selected_arm": selected,
        "eligible_arms": independently_eligible,
        "arms": arm_details,
    }


def _validate_history(
    *,
    states: pd.DataFrame,
    proposals: pd.DataFrame,
    lines: pd.DataFrame,
    selected_arm: str,
    tolerances: Mapping[str, float],
    termination: str,
) -> dict[str, Any]:
    accepted_updates = len(states) - 1
    if proposals.empty:
        _require(accepted_updates == 0 and selected_arm == "none", "missing proposal history")
        _require(lines.empty, "line rows exist without proposals")
        return {"accepted_updates": 0, "audit_rows": [], "terminal_proposal": None}
    targets = [_int(value) for value in proposals["target_update"]]
    _require(targets == list(range(1, len(proposals) + 1)), "proposal sequence is not exact")
    accepted_flags = [_int(value) for value in proposals["accepted"]]
    _require(
        accepted_flags[:accepted_updates] == [1] * accepted_updates,
        "accepted proposal prefix mismatch",
    )
    _require(
        accepted_flags[accepted_updates:] in ([], [0]),
        "only one final rejected proposal is allowed",
    )
    _require(
        all(str(value) == selected_arm for value in proposals["selected_arm"]),
        "selected arm changed during trajectory",
    )
    _require(selected_arm in {"B", "low"}, "proposal history has invalid selected arm")
    line_targets = [] if lines.empty else [_int(value) for value in lines["target_update"]]
    _require(set(line_targets).issubset(set(targets)), "orphan line-search rows")

    audit_rows: list[dict[str, Any]] = []
    line_gate_error_count = 0
    for index, proposal in proposals.iterrows():
        update = _int(proposal["target_update"])
        parent = states.loc[states["accepted_update"].eq(update - 1)].iloc[0]
        _require(
            _close(proposal["current_A"], parent["exact_a_per_dim"], atol=1e-9, rtol=1e-9)
            and _close(proposal["current_B"], parent["damped_full_burg_per_dim"], atol=1e-9, rtol=1e-9)
            and _close(proposal["current_low_energy"], parent["frozen_low_energy"], atol=1e-9, rtol=1e-9)
            and _close(proposal["current_a_gt1"], parent["a_gt1"], atol=1e-9, rtol=1e-9),
            f"proposal {update} current state mismatch",
        )
        gradient_metadata = json.loads(str(proposal["gradient_metadata"]))
        _validate_gradient_metadata(gradient_metadata)
        direction_details = _validate_direction_record(
            proposal,
            auxiliary=selected_arm,
        )
        cancellation = bool(direction_details["cancellation_gate_pass"])
        slopes = {
            "A": _float(proposal["slope_A"]),
            "B": _float(proposal["slope_B"]),
            "L_low": _float(proposal["slope_L_low"]),
        }
        target_lines = lines.loc[lines["target_update"].eq(update)] if not lines.empty else lines
        accepted = _int(proposal["accepted"]) == 1
        if not cancellation or not all(value < 0.0 for value in slopes.values()):
            _require(not accepted and target_lines.empty, f"proposal {update} invalid slope has lines")
            expected_failure = "cancellation_gate" if not cancellation else "nonnegative_joint_slope"
            _require(_text(proposal["failure"]) == expected_failure, f"proposal {update} bad failure")
            _require(_float(proposal["selected_alpha"]) == 0.0, f"proposal {update} stalled alpha nonzero")
            for candidate_column in (
                "candidate_A",
                "candidate_B",
                "candidate_low_energy",
                "candidate_a_gt1",
            ):
                _require(
                    pd.isna(proposal[candidate_column]),
                    f"proposal {update} stalled candidate payload is not empty",
                )
            continue

        attempted = [_float(value) for value in target_lines["alpha"]]
        _require(
            attempted == list(LINE_ALPHAS[: len(attempted)]),
            f"proposal {update} line grid is not an exact prefix",
        )
        recomputed_passes: list[bool] = []
        for _, line in target_lines.iterrows():
            _require(
                _is_sha256(_text(line["endpoint_parameter_hash"])),
                f"proposal {update} line endpoint hash malformed",
            )
            _line_metric_consistency(line)
            gates = _candidate_gates(
                current=parent,
                candidate=line,
                slopes=slopes,
                alpha=_float(line["alpha"]),
                tolerances=tolerances,
                require_low90=False,
            )
            for name, expected in gates.items():
                observed = _bool(line[f"gate_{name}"])
                if observed is not expected:
                    line_gate_error_count += 1
                _require(observed is expected, f"proposal {update} gate_{name} mismatch")
            repeat_pass = _repeat_passes(line, tolerances)
            passes = all(gates.values()) and repeat_pass
            _require(_bool(line["passes"]) is passes, f"proposal {update} pass flag mismatch")
            recomputed_passes.append(passes)
        if accepted:
            _require(target_lines.shape[0] > 0, f"accepted proposal {update} has no lines")
            _require(
                recomputed_passes[-1] and not any(recomputed_passes[:-1]),
                f"proposal {update} did not accept the first passing alpha",
            )
            selected = target_lines.iloc[-1]
            alpha = _float(selected["alpha"])
            _require(
                _close(proposal["selected_alpha"], alpha, atol=0.0, rtol=0.0),
                f"proposal {update} selected alpha mismatch",
            )
            state = states.loc[states["accepted_update"].eq(update)].iloc[0]
            _require(
                _text(selected["endpoint_parameter_hash"]) == _text(state["parameter_hash"]),
                f"proposal {update} endpoint/state hash mismatch",
            )
            candidate_pairs = (
                ("candidate_A", "exact_a_per_dim"),
                ("candidate_B", "damped_full_burg_per_dim"),
                ("candidate_low_energy", "frozen_low_energy"),
                ("candidate_a_gt1", "a_gt1"),
            )
            _require(
                all(
                    _close(proposal[left], selected[right], atol=1e-9, rtol=1e-9)
                    for left, right in candidate_pairs
                ),
                f"proposal {update} candidate/line mismatch",
            )
            _require(
                _close(state["exact_a_per_dim"], selected["exact_a_per_dim"], atol=1e-9, rtol=1e-9)
                and _close(state["damped_full_burg_per_dim"], selected["damped_full_burg_per_dim"], atol=1e-9, rtol=1e-9)
                and _close(state["a_low90_abs_per_dim"], selected["a_low90_abs_per_dim"], atol=1e-9, rtol=1e-9)
                and _close(state["a_gt1"], selected["a_gt1"], atol=1e-9, rtol=1e-9)
                and _close(state["m_max"], selected["m_max"], atol=1e-9, rtol=1e-9)
                and _close(state["m_p50"], selected["m_p50"], atol=1e-9, rtol=1e-9)
                and _close(state["effective_rank"], selected["effective_rank"], atol=1e-9, rtol=1e-9)
                and _int(state["count_lt_1e_4"]) == _int(selected["count_lt_1e_4"])
                and _int(state["count_lt_1e_2"]) == _int(selected["count_lt_1e_2"])
                and _int(state["count_lt_0p1"]) == _int(selected["count_lt_0p1"])
                and _close(state["task_loss"], selected["task_loss"], atol=1e-9, rtol=1e-9)
                and _close(state["parent_frozen_low_energy"], selected["frozen_low_energy"], atol=1e-9, rtol=1e-9),
                f"proposal {update} selected line/committed state mismatch",
            )
            component_gates: dict[str, bool] = {}
            current_objectives = _objective_values(parent)
            candidate_objectives = _objective_values(selected)
            for name in ("A", "B", "L_low"):
                component_gates[f"{name}_slope_negative"] = slopes[name] < 0.0
                component_gates[f"{name}_armijo"] = candidate_objectives[name] <= (
                    current_objectives[name] + ARMIJO_C1 * alpha * slopes[name]
                )
                component_gates[f"{name}_actual_decrease"] = (
                    current_objectives[name] - candidate_objectives[name] > tolerances[name]
                )
            high_tail = _float(selected["a_gt1"]) <= (
                _float(parent["a_gt1"]) + tolerances["A_gt1"]
            )
            transition_pass = all(component_gates.values()) and high_tail
            _require(transition_pass, f"accepted proposal {update} transition gates fail")
            audit_rows.append(
                {
                    "accepted_update": update,
                    "alpha": alpha,
                    **{key: int(value) for key, value in component_gates.items()},
                    "high_tail_nonincrease": int(high_tail),
                    "selected_line_row_passes": 1,
                    "committed_state_matches": 1,
                    "transition_passes": 1,
                }
            )
        else:
            _require(
                len(attempted) == len(LINE_ALPHAS) and not any(recomputed_passes),
                f"stalled proposal {update} did not exhaust the line",
            )
            _require(
                _text(proposal["failure"]) == "backtracking_exhausted",
                f"proposal {update} rejected for wrong reason",
            )
    terminal = None
    if len(proposals) > accepted_updates:
        terminal = _text(proposals.iloc[-1]["failure"])
    expected_termination = (
        "proposal0_no_eligible_arm"
        if selected_arm == "none"
        else "max_updates_reached"
        if accepted_updates == MAX_ACCEPTED_UPDATES
        else terminal
    )
    _require(termination == expected_termination, "termination does not match history")
    return {
        "accepted_updates": accepted_updates,
        "proposal_count": len(proposals),
        "line_count": len(lines),
        "audit_rows": audit_rows,
        "terminal_proposal": terminal,
        "line_gate_error_count": line_gate_error_count,
    }


def _success_from_raw(
    states: pd.DataFrame,
    *,
    tolerances: Mapping[str, float],
    history: Mapping[str, Any],
) -> dict[str, Any]:
    initial = states.iloc[0]
    final = states.iloc[-1]
    accepted_updates = len(states) - 1
    total_a_reduction = _float(initial["exact_a_per_dim"]) - _float(final["exact_a_per_dim"])
    lower90_reduction = _float(initial["a_low90_abs_per_dim"]) - _float(
        final["a_low90_abs_per_dim"]
    )
    bulk_fraction = lower90_reduction / max(total_a_reduction, 1e-30)
    low90_decreases = -states["a_low90_abs_per_dim"].astype(float).diff()
    tail_mask = states["accepted_update"].astype(int).between(81, 100)
    tail_bulk_updates = int((low90_decreases.loc[tail_mask] > tolerances["A_low90"]).sum())
    a_diffs = states["exact_a_per_dim"].astype(float).diff().iloc[1:]
    high_diffs = states["a_gt1"].astype(float).diff().iloc[1:]
    audit_rows = history["audit_rows"]
    all_a_armijo = all(
        row["A_slope_negative"] == 1
        and row["A_armijo"] == 1
        and row["A_actual_decrease"] == 1
        for row in audit_rows
    )
    gates = {
        "one_hundred_accepted_updates": accepted_updates == MAX_ACCEPTED_UPDATES,
        "tail_bulk_activity": tail_bulk_updates >= SUCCESS_TAIL_BULK_UPDATES_MIN,
        "final_a_at_most_0p90": _float(final["exact_a_per_dim"]) <= SUCCESS_A_MAX,
        "accepted_a_strictly_decreases": bool((a_diffs < -tolerances["A"]).all()),
        "all_historical_a_armijo_transitions": all_a_armijo,
        "final_b_below_initial": _float(final["damped_full_burg_per_dim"])
        < _float(initial["damped_full_burg_per_dim"]) - tolerances["B"],
        "non_top_only_fraction_at_least_0p25": bulk_fraction
        >= SUCCESS_BULK_FRACTION_MIN,
        "final_p50_above_initial": _float(final["m_p50"])
        > _float(initial["m_p50"]) + tolerances["m_p50"],
        "final_effective_rank_above_initial": _float(final["effective_rank"])
        > _float(initial["effective_rank"]) + tolerances["effective_rank"],
        "final_count_lt_1e_4_strictly_lower": _int(final["count_lt_1e_4"])
        < _int(initial["count_lt_1e_4"]),
        "final_count_lt_1e_2_strictly_lower": _int(final["count_lt_1e_2"])
        < _int(initial["count_lt_1e_2"]),
        "final_count_lt_0p1_strictly_lower": _int(final["count_lt_0p1"])
        < _int(initial["count_lt_0p1"]),
        "final_mmax_not_above_initial": _float(final["m_max"])
        <= _float(initial["m_max"]) + tolerances["m_max"],
        "high_tail_never_increases_beyond_floor": bool(
            (high_diffs <= tolerances["A_gt1"]).all()
        ),
        "all_state_values_finite": _frame_numeric_finite(
            states, excluded={"parameter_hash", "low_basis_hash"}
        ),
        "all_spectrum_values_finite": True,
        "all_raw_spectrum_minima_valid": bool(
            states["m_raw_eig_min"].astype(float).ge(-1e-8).all()
        ),
        "all_exact_a_closures_valid": bool(
            states["a_direct_abs_error"].astype(float).le(1e-9).all()
            and states["a_trace_abs_error"].astype(float).le(1e-9).all()
        ),
        "final_checkpoint_replay": True,
    }
    return {
        "scientific_success": bool(all(gates.values())),
        "success_gates": gates,
        "total_a_reduction": total_a_reduction,
        "lower90_a_reduction": lower90_reduction,
        "non_top_only_fraction": bulk_fraction,
        "tail_bulk_updates": tail_bulk_updates,
    }


def _named_tensor_hash(
    names: Sequence[str], mapping: Mapping[str, torch.Tensor]
) -> tuple[str, int]:
    _require(set(mapping) == set(names), "checkpoint active tensor key set mismatch")
    digest = hashlib.sha256()
    parameter_count = 0
    for name in names:
        tensor = mapping[name]
        _require(isinstance(tensor, torch.Tensor), f"checkpoint value {name} is not tensor")
        _require(tensor.device.type == "cpu", f"checkpoint tensor {name} is not on CPU")
        _require(bool(torch.isfinite(tensor).all()), f"checkpoint tensor {name} is nonfinite")
        value = tensor.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
        parameter_count += value.numel()
    return digest.hexdigest(), parameter_count


def _load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, dict), f"{path.name} checkpoint is not a mapping")
    return value


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _live_parameter_hash(
    names: Sequence[str], named: Mapping[str, torch.nn.Parameter]
) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name in names:
        value = named[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
        count += value.numel()
    return digest.hexdigest(), count


def _copy_active_mapping(
    names: Sequence[str],
    named: Mapping[str, torch.nn.Parameter],
    source: Mapping[str, torch.Tensor],
) -> None:
    _require(set(source) == set(names), "replay checkpoint active tensor keys mismatch")
    with torch.no_grad():
        for name in names:
            named[name].copy_(source[name].to(device=named[name].device, dtype=named[name].dtype))


def _independent_geometry(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    frozen_low_basis: torch.Tensor | None,
) -> dict[str, Any]:
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))

    def loss_from_z(latent: torch.Tensor) -> torch.Tensor:
        decoded = decode_weights(run.vae, run.normalizer, latent.reshape(1, -1)).squeeze(0)
        return _task_loss_from_flat(
            decoded,
            task_set=task_set,
            spec=run.spec,
            tau=tau,
            batch_indices=None,
        )

    with torch.no_grad():
        task_loss = float(loss_from_z(z).detach().cpu())
    started = time.perf_counter()
    hessian = _hessian_via_batched_hvp(
        loss_from_z,
        z,
        chunk_size=HESSIAN_CHUNK_SIZE,
    ).detach().double()
    hessian_sec = time.perf_counter() - started
    symmetry_relative = float(
        ((hessian - hessian.T).norm() / hessian.norm().clamp_min(1e-30)).cpu()
    )
    matrix = hessian @ hessian.T
    matrix = 0.5 * (matrix + matrix.T)
    raw_eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = raw_eigenvalues.clamp_min(0.0)
    low_mask = eigenvalues < LOW_THRESHOLD
    low_count = int(low_mask.sum().cpu())
    _require(low_count > 0, "independent replay low eigenspace is empty")
    canonical_basis = eigenvectors[:, low_mask].detach()
    low_basis = canonical_basis if frozen_low_basis is None else frozen_low_basis
    _require(low_basis.ndim == 2 and low_basis.shape[0] == DIMENSION, "bad frozen low basis")

    metrics = _spectral_metrics(eigenvalues.detach().cpu().numpy())
    identity = torch.eye(DIMENSION, device=matrix.device, dtype=matrix.dtype)
    direct_a = float(((matrix - identity).square().sum() / DIMENSION).cpu())
    trace_a = float(
        (1.0 - 2.0 * eigenvalues.mean() + eigenvalues.square().mean()).cpu()
    )
    canonical_energy = float(
        (
            torch.trace(canonical_basis.T @ matrix @ canonical_basis)
            / float(canonical_basis.shape[1])
        ).cpu()
    )
    frozen_energy = float(
        (torch.trace(low_basis.T @ matrix @ low_basis) / float(low_basis.shape[1])).cpu()
    )
    projector = canonical_basis @ canonical_basis.T
    orthogonality = float(
        (
            canonical_basis.T @ canonical_basis
            - torch.eye(low_count, device=matrix.device, dtype=matrix.dtype)
        )
        .abs()
        .max()
        .cpu()
    )
    residual = float(
        (matrix @ canonical_basis - canonical_basis * raw_eigenvalues[low_mask].unsqueeze(0))
        .norm()
        .cpu()
        / max(float(matrix.norm().cpu()), 1e-30)
    )
    metrics.update(
        {
            "task_loss": task_loss,
            "hessian_sec": hessian_sec,
            "hessian_symmetry_rel": symmetry_relative,
            "a_direct_matrix": direct_a,
            "a_trace_closure": trace_a,
            "a_direct_abs_error": abs(direct_a - metrics["exact_a_per_dim"]),
            "a_trace_abs_error": abs(trace_a - metrics["exact_a_per_dim"]),
            "m_raw_eig_min": float(raw_eigenvalues.min().cpu()),
            "frozen_low_energy": frozen_energy,
            "canonical_low_energy": canonical_energy,
            "low_count": float(low_count),
            "low_basis_orthogonality_max_abs": orthogonality,
            "low_basis_eigen_residual_relative": residual,
            "low_basis_hash": _tensor_sha256(projector),
        }
    )
    return {
        "metrics": metrics,
        "hessian": hessian,
        "matrix": matrix,
        "raw_eigenvalues": raw_eigenvalues,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "canonical_low_basis": canonical_basis,
    }


def _compare_geometry_to_row(
    geometry: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    stored_spectrum: np.ndarray | None,
    label: str,
) -> dict[str, Any]:
    metrics = geometry["metrics"]
    replay_eigenvalues = geometry["eigenvalues"].detach().cpu().numpy()
    derivative_diagnostics = {
        "burg_matrix_gradient_norm",
        "burg_matrix_gradient_eig_min",
        "burg_matrix_gradient_eig_p50",
        "burg_matrix_gradient_eig_max",
    }
    independent_keys = (set(_spectral_metrics(replay_eigenvalues)) - derivative_diagnostics) | {
        "task_loss",
        "hessian_symmetry_rel",
        "a_direct_matrix",
        "a_trace_closure",
        "m_raw_eig_min",
        "frozen_low_energy",
        "canonical_low_energy",
        "low_count",
    }
    errors: dict[str, float] = {}
    derivative_diagnostic_errors = {
        key: abs(_float(row[key]) - _float(metrics[key]))
        for key in derivative_diagnostics
    }
    integral = {"count_lt_1e_4", "count_lt_1e_2", "count_lt_0p1", "low_count"}
    for key in sorted(independent_keys):
        _require(key in row, f"{label} is missing replay metric {key}")
        observed = _float(row[key])
        expected = _float(metrics[key])
        if key in integral:
            _require(observed == expected, f"{label} integral metric {key} mismatch")
        else:
            _require(
                _close(
                    observed,
                    expected,
                    atol=GPU_GEOMETRY_REPLAY_ATOL,
                    rtol=GPU_GEOMETRY_REPLAY_RTOL,
                ),
                f"{label} metric {key} mismatch: {observed} vs {expected}",
            )
        errors[key] = abs(observed - expected)
    _require(
        _float(metrics["a_direct_abs_error"]) <= 1e-9
        and _float(metrics["a_trace_abs_error"]) <= 1e-9
        and _float(metrics["low_basis_orthogonality_max_abs"]) <= 1e-10
        and _float(metrics["low_basis_eigen_residual_relative"]) <= 1e-10,
        f"{label} independent closure or low-basis diagnostic failed",
    )
    spectrum_error = None
    if stored_spectrum is not None:
        replay = geometry["eigenvalues"].detach().cpu().numpy()
        _require(stored_spectrum.shape == (DIMENSION,), f"{label} stored spectrum shape mismatch")
        spectrum_error = float(np.max(np.abs(replay - stored_spectrum)))
        _require(
            spectrum_error <= GPU_GEOMETRY_REPLAY_ATOL,
            f"{label} spectrum replay mismatch",
        )
    return {
        "metric_max_abs_error": max(errors.values()),
        "metric_errors": errors,
        "fresh_burg_derivative_diagnostic_errors": derivative_diagnostic_errors,
        "spectrum_max_abs_error": spectrum_error,
        "stored_low_basis_hash": _text(row["low_basis_hash"]),
        "replay_low_basis_hash": metrics["low_basis_hash"],
        "low_basis_hash_bitwise_match": (
            _text(row["low_basis_hash"]) == metrics["low_basis_hash"]
        ),
    }


def _independent_cotangents(geometry: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    hessian = geometry["hessian"]
    matrix = geometry["matrix"]
    low_basis = geometry["canonical_low_basis"]
    identity = torch.eye(DIMENSION, device=matrix.device, dtype=torch.float64)
    gradient_b = identity / (DIMENSION * (1.0 + EPSILON)) - torch.linalg.solve(
        matrix + EPSILON * identity,
        identity,
    ) / DIMENSION
    return {
        "A": 4.0 * ((matrix - identity) @ hessian) / DIMENSION,
        "B": 2.0 * gradient_b @ hessian,
        "low": -2.0 * (low_basis @ low_basis.T @ hessian) / float(low_basis.shape[1]),
    }


def _blocked_parameter_gradients(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    cotangents: Mapping[str, torch.Tensor],
    label: str,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, Any]]:
    names = ("A", "B", "low")
    _require(set(cotangents) == set(names), "independent cotangent name set mismatch")
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    accumulated = {
        name: [torch.zeros_like(parameter, dtype=torch.float64) for parameter in active]
        for name in names
    }
    block_count = math.ceil(DIMENSION / GRADIENT_BLOCK_SIZE)
    unused = {name: np.zeros(len(active), dtype=np.int64) for name in names}
    started = time.perf_counter()
    for start in range(0, DIMENSION, GRADIENT_BLOCK_SIZE):
        stop = min(start + GRADIENT_BLOCK_SIZE, DIMENSION)
        scalar_rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
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
                scalar_rows[name].append(
                    torch.dot(cotangents[name][index].to(dtype=hvp.dtype), hvp)
                )
        totals = {name: torch.stack(scalar_rows[name]).sum() for name in names}
        block_gradients = {
            "A": torch.autograd.grad(
                totals["A"], active, retain_graph=True, allow_unused=True
            ),
            "B": torch.autograd.grad(
                totals["B"], active, retain_graph=True, allow_unused=True
            ),
            "low": torch.autograd.grad(
                totals["low"], active, retain_graph=False, allow_unused=True
            ),
        }
        with torch.no_grad():
            for name in names:
                for index, value in enumerate(block_gradients[name]):
                    if value is None:
                        unused[name][index] += 1
                    else:
                        accumulated[name][index].add_(value.detach().double())
        print(
            f"[exact-selected-i6-review] stage=gpu-gradient-replay label={label} "
            f"rows={stop}/{DIMENSION} block={(stop + GRADIENT_BLOCK_SIZE - 1) // GRADIENT_BLOCK_SIZE}/"
            f"{block_count} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del scalar_rows, totals, block_gradients
    unused_all_blocks = {
        name: int((unused[name] == block_count).sum()) for name in names
    }
    _require(
        all(value == 0 for value in unused_all_blocks.values()),
        f"independent {label} replay has globally unused active tensors",
    )
    return accumulated, {
        "block_count": block_count,
        "unused_parameter_tensors_all_blocks": unused_all_blocks,
        "elapsed_sec": time.perf_counter() - started,
    }


def _vector_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    _require(len(left) == len(right) and len(left) > 0, "gradient vector shape mismatch")
    total = torch.zeros((), device=left[0].device, dtype=torch.float64)
    for a, b in zip(left, right, strict=True):
        total += (a.detach().double() * b.detach().double()).sum()
    return float(total.cpu())


def _vector_norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, _vector_dot(values, values)))


def _independent_direction(
    gradient_a: Sequence[torch.Tensor],
    gradient_x: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
) -> tuple[list[torch.Tensor] | None, dict[str, Any]]:
    norm_a = _vector_norm(gradient_a)
    norm_x = _vector_norm(gradient_x)
    cosine = _vector_dot(gradient_a, gradient_x) / max(norm_a * norm_x, 1e-300)
    common = [
        (
            a.detach().to(device=parameter.device, dtype=parameter.dtype) / norm_a
            + x.detach().to(device=parameter.device, dtype=parameter.dtype) / norm_x
        ).clone()
        for a, x, parameter in zip(gradient_a, gradient_x, active, strict=True)
    ]
    source_norm = _vector_norm(common)
    cancellation_pass = bool(
        norm_a > 0.0
        and norm_x > 0.0
        and math.isfinite(source_norm)
        and source_norm >= CANCELLATION_NORM_MIN
    )
    direction = None
    if cancellation_pass:
        direction = [
            (
                value.detach().to(device=parameter.device, dtype=parameter.dtype)
                * (-TARGET_NORM / source_norm)
            ).clone()
            for value, parameter in zip(common, active, strict=True)
        ]
    return direction, {
        "gradient_a_norm": norm_a,
        "gradient_x_norm": norm_x,
        "gradient_cosine": cosine,
        "unit_common_source_norm": source_norm,
        "cancellation_gate_pass": cancellation_pass,
        "direction_norm": 0.0 if direction is None else _vector_norm(direction),
    }


def _direction_slopes(
    gradients: Mapping[str, Sequence[torch.Tensor]],
    direction: Sequence[torch.Tensor],
) -> dict[str, float]:
    return {
        "A": _vector_dot(gradients["A"], direction),
        "B": _vector_dot(gradients["B"], direction),
        "L_low": _vector_dot(gradients["low"], direction),
    }


def _set_direction_endpoint(
    active: Sequence[torch.nn.Parameter],
    base: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    alpha: float,
) -> None:
    with torch.no_grad():
        for parameter, origin, delta in zip(active, base, direction, strict=True):
            parameter.copy_(origin + float(alpha) * delta)


def _gradient_replay_close(observed: Any, expected: Any) -> bool:
    return _close(
        observed,
        expected,
        atol=GPU_GRADIENT_REPLAY_ATOL,
        rtol=GPU_GRADIENT_REPLAY_RTOL,
    )


def _cotangent_norms(cotangents: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        "k_a_norm": float(cotangents["A"].norm().cpu()),
        "k_b_norm": float(cotangents["B"].norm().cpu()),
        "k_low_norm": float(cotangents["low"].norm().cpu()),
    }


def _run_independent_gpu_replay(
    *,
    output: Path,
    device_name: str,
    states: pd.DataFrame,
    spectra: pd.DataFrame,
    proposals: pd.DataFrame,
    selections: pd.DataFrame,
    proposal0: Mapping[str, Any],
    final_checkpoint: Mapping[str, Any],
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is unavailable for required independent replay")
    device = torch.device(device_name)
    _require(device.type == "cuda", "independent replay device must be CUDA")
    print(
        f"[exact-selected-i6-review] stage=gpu-replay-load device={device} "
        f"checkpoint={output / 'final_checkpoint.pt'}",
        flush=True,
    )
    accepted_checkpoint = DEFAULT_RUN_DIR / "vae_checkpoint.pt"
    _require(
        _sha256_file(accepted_checkpoint) == EXPECTED_ACCEPTED_CHECKPOINT_SHA256,
        "independent replay accepted checkpoint hash mismatch",
    )
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
    cfg = replace(cfg, vae_precond_hvp_mode="autograd")
    bank = pd.read_csv(STATE_BANK)
    selected_state = bank.loc[bank["state_position"].eq(STATE_POSITION)]
    _require(
        len(selected_state) == 1
        and _int(selected_state.iloc[0]["source_weight_index"]) == SOURCE_WEIGHT_INDEX,
        "independent replay state-bank identity mismatch",
    )
    record = run.records.iloc[SOURCE_WEIGHT_INDEX].to_dict()
    record["source_weight_index"] = SOURCE_WEIGHT_INDEX
    weight = run.weights[[SOURCE_WEIGHT_INDEX]].to(
        device=device,
        dtype=torch_dtype(run.cfg),
    )
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
    _require(_tensor_sha256(z) == EXPECTED_Z_SHA256, "independent replay z hash mismatch")
    task_set = _task_set_for_record(run.task_tensors, record)
    full_ce = all(
        _batch_indices(
            task_set,
            batch_size=int(cfg.vae_precond_batch_size),
            step=10,
            sample_key=SOURCE_WEIGHT_INDEX,
            pair_key=pair_key,
        )
        is None
        for pair_key in range(8)
    )
    _require(full_ce, "independent replay is not using full CE data")
    active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
    named = dict(run.vae.named_parameters())
    _require(set(active_names).issubset(named), "independent replay active names missing")
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in active_names]
    _require(
        sum(parameter.numel() for parameter in active) == ACTIVE_PARAMETER_COUNT,
        "independent replay active parameter count mismatch",
    )

    iteration3 = _load_checkpoint(ITERATION3_CHECKPOINT)
    _copy_active_mapping(active_names, named, iteration3["active_model_state"])
    start_hash, start_count = _live_parameter_hash(active_names, named)
    _require(
        start_count == ACTIVE_PARAMETER_COUNT
        and start_hash == EXPECTED_INITIAL_PARAMETER_SHA256
        and start_hash == _text(states.iloc[0]["parameter_hash"]),
        "independent replay start parameter hash mismatch",
    )
    print("[exact-selected-i6-review] stage=gpu-replay-start-geometry", flush=True)
    start_geometry = _independent_geometry(
        run=run,
        z=z,
        record=record,
        frozen_low_basis=None,
    )
    start_spectrum = (
        spectra.loc[spectra["accepted_update"].eq(0)]
        .sort_values("rank")["m_eigenvalue"]
        .to_numpy(dtype=np.float64)
    )
    start_geometry_review = _compare_geometry_to_row(
        start_geometry,
        states.iloc[0],
        stored_spectrum=start_spectrum,
        label="start disk checkpoint",
    )
    start_cotangents = _independent_cotangents(start_geometry)
    start_cotangent_norms = _cotangent_norms(start_cotangents)
    stored_start_gradient = proposal0.get("gradient_metadata")
    _require(isinstance(stored_start_gradient, Mapping), "proposal-0 gradient metadata missing")
    _require(
        all(
            _gradient_replay_close(stored_start_gradient[key], value)
            for key, value in start_cotangent_norms.items()
        ),
        "proposal-0 cotangent norms do not replay",
    )
    print("[exact-selected-i6-review] stage=gpu-replay-start-gradients", flush=True)
    start_gradients, start_gradient_meta = _blocked_parameter_gradients(
        cfg=cfg,
        run=run,
        z=z,
        record=record,
        active=active,
        cotangents=start_cotangents,
        label="proposal0",
    )
    base = [parameter.detach().clone() for parameter in active]
    arm_replay: dict[str, Any] = {}
    independently_eligible: list[str] = []
    replay_directions: dict[str, list[torch.Tensor]] = {}
    for arm in ("B", "low"):
        row_frame = selections.loc[selections["arm"].eq(arm)]
        _require(len(row_frame) == 1, f"independent replay missing proposal-0 arm {arm}")
        row = row_frame.iloc[0]
        auxiliary_key = arm if arm == "B" else "low"
        direction, direction_meta = _independent_direction(
            start_gradients["A"],
            start_gradients[auxiliary_key],
            active,
        )
        _require(direction is not None, f"independent proposal-0 arm {arm} cancelled")
        replay_directions[arm] = direction
        for key in (
            "gradient_a_norm",
            "gradient_x_norm",
            "gradient_cosine",
            "unit_common_source_norm",
            "direction_norm",
        ):
            _require(
                _gradient_replay_close(row[key], direction_meta[key]),
                f"independent proposal-0 {arm} {key} mismatch",
            )
        slopes = _direction_slopes(start_gradients, direction)
        for key, column in (
            ("A", "slope_A"),
            ("B", "slope_B"),
            ("L_low", "slope_L_low"),
        ):
            _require(
                _gradient_replay_close(row[column], slopes[key]),
                f"independent proposal-0 {arm} {column} mismatch",
            )
        try:
            _set_direction_endpoint(active, base, direction, SELECTION_ALPHA)
            endpoint_hash, _ = _live_parameter_hash(active_names, named)
            _require(
                endpoint_hash == _text(row["endpoint_parameter_hash"]),
                f"independent proposal-0 {arm} endpoint hash mismatch",
            )
            endpoint = _independent_geometry(
                run=run,
                z=z,
                record=record,
                frozen_low_basis=start_geometry["canonical_low_basis"],
            )
            endpoint_review = _compare_geometry_to_row(
                endpoint,
                row,
                stored_spectrum=None,
                label=f"proposal-0 {arm} endpoint",
            )
            repeat = _independent_geometry(
                run=run,
                z=z,
                record=record,
                frozen_low_basis=start_geometry["canonical_low_basis"],
            )
            repeat_spectrum_error = float(
                (endpoint["eigenvalues"] - repeat["eigenvalues"]).abs().max().cpu()
            )
            repeat_hessian_error = float(
                (endpoint["hessian"] - repeat["hessian"]).abs().max().cpu()
            )
            repeat_hash, _ = _live_parameter_hash(active_names, named)
            _require(
                repeat_spectrum_error <= REPEAT_MAX_ERROR
                and repeat_hessian_error <= REPEAT_MAX_ERROR
                and repeat_hash == endpoint_hash,
                f"independent proposal-0 {arm} repeat failed",
            )
            gates = _candidate_gates(
                current=states.iloc[0],
                candidate=endpoint["metrics"],
                slopes=slopes,
                alpha=SELECTION_ALPHA,
                tolerances=tolerances,
                require_low90=True,
            )
            eligible = all(gates.values())
            _require(
                eligible is _bool(row["eligible"]),
                f"independent proposal-0 {arm} eligibility mismatch",
            )
            if eligible:
                independently_eligible.append(arm)
            arm_replay[arm] = {
                "direction_metadata": direction_meta,
                "slopes": slopes,
                "endpoint_parameter_hash": endpoint_hash,
                "endpoint_review": endpoint_review,
                "repeat_spectrum_max_abs_error": repeat_spectrum_error,
                "repeat_hessian_max_abs_error": repeat_hessian_error,
                "gates": gates,
                "eligible": eligible,
                "low90_decrease": _float(states.iloc[0]["a_low90_abs_per_dim"])
                - _float(endpoint["metrics"]["a_low90_abs_per_dim"]),
                "a_decrease": _float(states.iloc[0]["exact_a_per_dim"])
                - _float(endpoint["metrics"]["exact_a_per_dim"]),
            }
        finally:
            _set_direction_endpoint(active, base, direction, 0.0)
            restored_hash, _ = _live_parameter_hash(active_names, named)
            _require(restored_hash == start_hash, f"proposal-0 {arm} restoration hash mismatch")
    chosen = sorted(
        independently_eligible,
        key=lambda arm: (
            -float(arm_replay[arm]["low90_decrease"]),
            -float(arm_replay[arm]["a_decrease"]),
            arm,
        ),
    )[0] if independently_eligible else "none"
    _require(chosen == proposal0.get("selected_arm"), "independent arm selection mismatch")

    del start_gradients, start_cotangents, replay_directions, base
    torch.cuda.empty_cache()
    _copy_active_mapping(active_names, named, final_checkpoint["active_model_state"])
    final_hash, final_count = _live_parameter_hash(active_names, named)
    _require(
        final_count == ACTIVE_PARAMETER_COUNT
        and final_hash == final_checkpoint.get("active_parameter_hash")
        and final_hash == _text(states.iloc[-1]["parameter_hash"]),
        "independent final disk checkpoint hash mismatch",
    )
    print("[exact-selected-i6-review] stage=gpu-replay-final-geometry", flush=True)
    final_geometry = _independent_geometry(
        run=run,
        z=z,
        record=record,
        frozen_low_basis=None,
    )
    final_update = _int(states.iloc[-1]["accepted_update"])
    final_spectrum = (
        spectra.loc[spectra["accepted_update"].eq(final_update)]
        .sort_values("rank")["m_eigenvalue"]
        .to_numpy(dtype=np.float64)
    )
    final_geometry_review = _compare_geometry_to_row(
        final_geometry,
        states.iloc[-1],
        stored_spectrum=final_spectrum,
        label="final disk checkpoint",
    )
    terminal_rows = proposals.loc[proposals["target_update"].eq(final_update + 1)]
    _require(len(terminal_rows) == 1, "terminal cancellation proposal is not unique")
    terminal = terminal_rows.iloc[0]
    _require(
        _text(terminal["failure"]) == "cancellation_gate",
        "terminal proposal is not the recorded cancellation gate",
    )
    final_cotangents = _independent_cotangents(final_geometry)
    final_cotangent_norms = _cotangent_norms(final_cotangents)
    terminal_gradient_metadata = json.loads(str(terminal["gradient_metadata"]))
    _require(
        all(
            _gradient_replay_close(terminal_gradient_metadata[key], value)
            for key, value in final_cotangent_norms.items()
        ),
        "terminal cotangent norms do not replay",
    )
    print("[exact-selected-i6-review] stage=gpu-replay-terminal-gradients", flush=True)
    final_gradients, final_gradient_meta = _blocked_parameter_gradients(
        cfg=cfg,
        run=run,
        z=z,
        record=record,
        active=active,
        cotangents=final_cotangents,
        label=f"proposal{final_update + 1}",
    )
    terminal_direction, terminal_meta = _independent_direction(
        final_gradients["A"],
        final_gradients["low" if chosen == "low" else "B"],
        active,
    )
    for key in (
        "gradient_a_norm",
        "gradient_x_norm",
        "gradient_cosine",
        "unit_common_source_norm",
        "direction_norm",
    ):
        _require(
            _gradient_replay_close(terminal[key], terminal_meta[key]),
            f"terminal proposal independent {key} mismatch",
        )
    _require(
        terminal_direction is None
        and terminal_meta["cancellation_gate_pass"] is False
        and terminal_meta["unit_common_source_norm"] < CANCELLATION_NORM_MIN
        and _bool(terminal["cancellation_gate_pass"]) is False,
        "proposal 18 cancellation gate does not independently replay",
    )
    terminal_margin = CANCELLATION_NORM_MIN - float(
        terminal_meta["unit_common_source_norm"]
    )
    _require(
        terminal_margin > 1000.0 * DIRECTION_IDENTITY_ATOL,
        "terminal cancellation is numerically ambiguous",
    )
    del final_gradients, final_cotangents
    torch.cuda.empty_cache()
    return {
        "performed": True,
        "device": str(device),
        "full_ce": full_ce,
        "z_sha256": _tensor_sha256(z),
        "initial_parameter_sha256": start_hash,
        "final_parameter_sha256": final_hash,
        "start_geometry": start_geometry_review,
        "proposal0_cotangent_norms": start_cotangent_norms,
        "proposal0_gradient_replay": start_gradient_meta,
        "proposal0_arms": arm_replay,
        "independently_eligible_arms": independently_eligible,
        "independently_selected_arm": chosen,
        "final_geometry": final_geometry_review,
        "terminal_cotangent_norms": final_cotangent_norms,
        "terminal_gradient_replay": final_gradient_meta,
        "terminal_direction_metadata": terminal_meta,
        "terminal_cancellation_margin": terminal_margin,
    }


def _validate_geometry_replay_payload(
    payload: Mapping[str, Any],
    *,
    final_checkpoint_sha256: str,
    active_parameter_hash: str,
    final_state: Mapping[str, Any],
    final_spectrum: np.ndarray,
) -> dict[str, Any]:
    _require(payload.get("schema_id") == GEOMETRY_REPLAY_SCHEMA, "geometry replay schema mismatch")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "geometry replay protocol mismatch")
    _require(
        payload.get("final_checkpoint_file_sha256") == final_checkpoint_sha256,
        "geometry replay checkpoint hash mismatch",
    )
    _require(
        payload.get("active_parameter_hash") == active_parameter_hash,
        "geometry replay parameter hash mismatch",
    )
    _require(payload.get("source_weight_index") == SOURCE_WEIGHT_INDEX, "geometry replay state mismatch")
    _require(payload.get("z_sha256") == EXPECTED_Z_SHA256, "geometry replay z mismatch")
    replay_spectrum = np.asarray(payload.get("m_eigenvalues"), dtype=np.float64)
    _require(replay_spectrum.shape == (DIMENSION,), "geometry replay spectrum shape mismatch")
    spectrum_error = float(np.max(np.abs(replay_spectrum - final_spectrum)))
    _require(spectrum_error <= 1e-9, "geometry replay spectrum mismatch")
    replay_metrics = payload.get("metrics")
    _require(isinstance(replay_metrics, Mapping), "geometry replay metrics missing")
    required_metrics = (
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
    metric_errors = {
        key: abs(_float(replay_metrics[key]) - _float(final_state[key]))
        for key in required_metrics
    }
    _require(max(metric_errors.values()) <= 1e-9, "geometry replay metric mismatch")
    spectrum_metrics = _spectral_metrics(replay_spectrum)
    _require(
        all(
            _close(replay_metrics[key], spectrum_metrics[key], atol=2e-9, rtol=2e-9)
            for key in required_metrics
            if key in spectrum_metrics
        ),
        "geometry replay metrics do not close against replay spectrum",
    )
    return {
        "performed": True,
        "spectrum_max_abs_error": spectrum_error,
        "metric_max_abs_error": max(metric_errors.values()),
    }


def _expect_rejected(function: Callable[[], Any]) -> bool:
    try:
        function()
    except (ReviewError, KeyError, ValueError, TypeError, AssertionError):
        return True
    return False


def _run_tamper_tests(
    *,
    states: pd.DataFrame,
    spectra: pd.DataFrame,
    proposals: pd.DataFrame,
    lines: pd.DataFrame,
    selected_arm: str,
    tolerances: Mapping[str, float],
    termination: str,
) -> dict[str, bool]:
    spectrum_tamper = spectra.copy(deep=True)
    spectrum_tamper.loc[spectrum_tamper.index[-1], "m_eigenvalue"] += 1e-3
    spectrum_rejected = _expect_rejected(
        lambda: _validate_spectrum_tables(states.copy(deep=True), spectrum_tamper)
    )

    history_rejected = True
    if not proposals.empty and bool(proposals["accepted"].astype(int).eq(1).any()):
        proposal_tamper = proposals.copy(deep=True)
        accepted_index = proposal_tamper.index[proposal_tamper["accepted"].astype(int).eq(1)][0]
        proposal_tamper.loc[accepted_index, "candidate_A"] += 1e-4
        history_rejected = _expect_rejected(
            lambda: _validate_history(
                states=states.copy(deep=True),
                proposals=proposal_tamper,
                lines=lines.copy(deep=True),
                selected_arm=selected_arm,
                tolerances=tolerances,
                termination=termination,
            )
        )

    line_rejected = True
    line_nan_rejected = True
    repeat_rejected = True
    if not lines.empty:
        line_tamper = lines.copy(deep=True)
        failing = line_tamper.index[line_tamper["passes"].astype(int).eq(0)]
        target_index = failing[0] if len(failing) else line_tamper.index[0]
        line_tamper.loc[target_index, "passes"] = 1 - int(line_tamper.loc[target_index, "passes"])
        line_rejected = _expect_rejected(
            lambda: _validate_history(
                states=states.copy(deep=True),
                proposals=proposals.copy(deep=True),
                lines=line_tamper,
                selected_arm=selected_arm,
                tolerances=tolerances,
                termination=termination,
            )
        )
        line_nan_tamper = lines.copy(deep=True)
        line_nan_tamper.loc[line_nan_tamper.index[0], "task_loss"] = np.nan
        line_nan_rejected = _expect_rejected(
            lambda: _validate_history(
                states=states.copy(deep=True),
                proposals=proposals.copy(deep=True),
                lines=line_nan_tamper,
                selected_arm=selected_arm,
                tolerances=tolerances,
                termination=termination,
            )
        )
        passing = lines.index[lines["passes"].astype(int).eq(1)]
        if len(passing):
            repeat_tamper = lines.copy(deep=True)
            repeat_tamper.loc[passing[0], "repeat_hessian_max_abs_error"] = 1e-6
            repeat_rejected = _expect_rejected(
                lambda: _validate_history(
                    states=states.copy(deep=True),
                    proposals=proposals.copy(deep=True),
                    lines=repeat_tamper,
                    selected_arm=selected_arm,
                    tolerances=tolerances,
                    termination=termination,
                )
            )

    cancellation_rejected = True
    radius_rejected = True
    if not proposals.empty:
        terminal_indices = proposals.index[proposals["accepted"].astype(int).eq(0)]
        if len(terminal_indices):
            cancellation_tamper = proposals.copy(deep=True)
            cancellation_tamper.loc[
                terminal_indices[-1], "unit_common_source_norm"
            ] += 1e-4
            cancellation_rejected = _expect_rejected(
                lambda: _validate_history(
                    states=states.copy(deep=True),
                    proposals=cancellation_tamper,
                    lines=lines.copy(deep=True),
                    selected_arm=selected_arm,
                    tolerances=tolerances,
                    termination=termination,
                )
            )
        accepted_indices = proposals.index[proposals["accepted"].astype(int).eq(1)]
        if len(accepted_indices):
            radius_tamper = proposals.copy(deep=True)
            radius_tamper.loc[accepted_indices[0], "direction_norm"] += 1e-6
            radius_rejected = _expect_rejected(
                lambda: _validate_history(
                    states=states.copy(deep=True),
                    proposals=radius_tamper,
                    lines=lines.copy(deep=True),
                    selected_arm=selected_arm,
                    tolerances=tolerances,
                    termination=termination,
                )
            )

    hash_bijection_tamper = states.copy(deep=True)
    hash_bijection_tamper.loc[
        hash_bijection_tamper.index[-1], "parameter_hash"
    ] = hash_bijection_tamper.iloc[0]["parameter_hash"]
    hash_bijection_rejected = _expect_rejected(
        lambda: _validate_spectrum_tables(hash_bijection_tamper, spectra.copy(deep=True))
    )
    return {
        "spectrum_value_tamper_rejected": spectrum_rejected,
        "accepted_proposal_tamper_rejected": history_rejected,
        "line_pass_flag_tamper_rejected": line_rejected,
        "line_nan_tamper_rejected": line_nan_rejected,
        "selected_repeat_tamper_rejected": repeat_rejected,
        "terminal_cancellation_tamper_rejected": cancellation_rejected,
        "direction_radius_tamper_rejected": radius_rejected,
        "state_hash_bijection_tamper_rejected": hash_bijection_rejected,
    }


def _write_report(
    output: Path,
    review: Review,
    *,
    scientific_success: bool | None,
    selected_arm: str | None,
    accepted_updates: int | None,
    termination: str | None,
) -> bool:
    valid = bool(review.gates and all(review.gates.values()) and not review.errors)
    report = {
        "review_protocol_id": REVIEW_PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "review_scope": (
            "independent artifact/formula validation plus fresh GPU replay from disk checkpoints"
        ),
        "reviewer_source_sha256": _sha256_file(Path(__file__)),
        "valid": valid,
        "scientific_success": scientific_success if valid else None,
        "recomputed_scientific_success": scientific_success,
        "selected_arm": selected_arm,
        "accepted_updates": accepted_updates,
        "termination": termination,
        "gates": review.gates,
        "failed_gates": sorted(name for name, passed in review.gates.items() if not passed),
        "recomputed": review.details,
        "warnings": review.warnings,
        "limitations": review.limitations,
        "errors": review.errors,
    }
    path = output / "independent_review.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(
        f"[exact-selected-i6-review] report={path} valid={valid} "
        f"scientific_success={report['scientific_success']}",
        flush=True,
    )
    if not valid:
        print(
            f"[exact-selected-i6-review] failed_gates={report['failed_gates']}",
            flush=True,
        )
        for error in review.errors:
            print(f"[exact-selected-i6-review] error={error}", flush=True)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--geometry-replay-json",
        type=Path,
        default=None,
        help="Optional independently generated GPU geometry replay payload.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for mandatory independent checkpoint/gradient replay.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    staging = Path(str(output) + ".incomplete")
    if (
        not output.is_dir()
        or not (output / "FINALIZED.json").is_file()
        or (output / "INCOMPLETE").exists()
        or staging.exists()
    ):
        print(
            f"[exact-selected-i6-review] finalized production unavailable: "
            f"output={output} staging_exists={staging.exists()}",
            flush=True,
        )
        return 1
    print(
        f"[exact-selected-i6-review] start output={output} "
        f"cpu_artifact_review=true gpu_replay_device={args.device}",
        flush=True,
    )

    review = Review()
    context: dict[str, Any] = {}
    scientific_success: bool | None = None
    selected_arm: str | None = None
    accepted_updates: int | None = None
    termination: str | None = None

    def metadata_phase() -> None:
        finalized = _load_json(output / "FINALIZED.json")
        resolved = _load_json(output / "resolved_config.json")
        decision = _load_json(output / "decision.json")
        manifest = _load_json(output / "artifact_manifest.json")
        proposal0 = _load_json(output / "proposal0_decision.json")
        context.update(
            finalized=finalized,
            resolved=resolved,
            decision=decision,
            manifest=manifest,
            proposal0=proposal0,
        )
        review.gate(
            "protocol_ids_match",
            {
                finalized.get("protocol_id"),
                resolved.get("protocol_id"),
                decision.get("protocol_id"),
                manifest.get("protocol_id"),
            }
            == {PROTOCOL_ID},
        )
        review.gate(
            "finalization_complete",
            finalized.get("status") == "complete_awaiting_independent_review"
            and not (output / "INCOMPLETE").exists(),
        )
        review.gate(
            "finalization_hashes_match",
            finalized.get("decision_sha256") == _sha256_file(output / "decision.json")
            and finalized.get("artifact_manifest_sha256")
            == _sha256_file(output / "artifact_manifest.json"),
        )
        artifacts = manifest.get("artifacts")
        artifact_matches = {
            name: (
                isinstance(artifacts, Mapping)
                and _is_sha256(artifacts.get(name))
                and (output / name).is_file()
                and _sha256_file(output / name) == artifacts.get(name)
            )
            for name in EXPECTED_MANIFEST_ARTIFACTS
        }
        review.gate(
            "manifest_artifact_set_exact",
            isinstance(artifacts, Mapping)
            and set(artifacts) == EXPECTED_MANIFEST_ARTIFACTS,
        )
        review.gate("all_artifact_hashes_match", all(artifact_matches.values()))
        actual_files = {path.name for path in output.iterdir() if path.is_file()}
        allowed = EXPECTED_MANIFEST_ARTIFACTS | UNMANIFESTED_FILES
        review.gate(
            "no_unaccounted_production_files",
            actual_files.issubset(allowed)
            and (EXPECTED_MANIFEST_ARTIFACTS | {"artifact_manifest.json", "FINALIZED.json", "run.log"}).issubset(actual_files),
        )

        source_snapshot = output / "executed_source_snapshot.py"
        source_hash = _sha256_file(source_snapshot)
        normalized_source_hash = _normalized_producer_sha256(source_snapshot)
        review.gate(
            "executed_source_fingerprint_matches",
            source_hash == EXPECTED_PRODUCER_SHA256
            and normalized_source_hash == EXPECTED_NORMALIZED_PRODUCER_SHA256
            and source_hash == resolved.get("source_sha256")
            and normalized_source_hash == resolved.get("normalized_source_sha256")
            and source_hash == manifest.get("executed_source_sha256")
            and normalized_source_hash
            == manifest.get("executed_normalized_source_sha256"),
        )
        review.gate(
            "live_producer_matches_executed_source",
            PRODUCER_SOURCE.is_file()
            and _sha256_file(PRODUCER_SOURCE) == source_hash
            and _normalized_producer_sha256(PRODUCER_SOURCE) == normalized_source_hash,
        )

        protocol_snapshot = output / "protocol_snapshot.md"
        protocol_hash = _sha256_file(protocol_snapshot)
        review.gate(
            "protocol_snapshot_and_live_match",
            protocol_hash == EXPECTED_PROTOCOL_SHA256
            and protocol_hash == resolved.get("protocol_sha256")
            and PROTOCOL_PATH.is_file()
            and _sha256_file(PROTOCOL_PATH) == protocol_hash,
        )
        dependency_snapshot = output / "frozen_dependency_manifest_snapshot.json"
        dependency_hash = _sha256_file(dependency_snapshot)
        dependencies = _load_json(dependency_snapshot)
        dependency_matches = {
            relative: (
                isinstance(relative, str)
                and _is_sha256(digest)
                and (ROOT / relative).is_file()
                and _sha256_file(ROOT / relative) == digest
            )
            for relative, digest in dependencies.items()
        }
        review.gate(
            "dependency_manifest_fingerprint_matches",
            dependency_hash == EXPECTED_DEPENDENCY_MANIFEST_SHA256
            and dependency_hash == resolved.get("dependency_manifest_sha256")
            and FROZEN_DEPENDENCY_MANIFEST.is_file()
            and _sha256_file(FROZEN_DEPENDENCY_MANIFEST) == dependency_hash,
        )
        review.gate(
            "all_frozen_dependencies_match",
            len(dependencies) == 24
            and all(dependency_matches.values())
            and _equivalent(resolved.get("dependency_matches"), dependency_matches),
        )
        accepted_relative = (
            "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing/"
            "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0/"
            "vae_checkpoint.pt"
        )
        iteration3_relative = (
            "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
            "one_state_exact_a_h2048/iteration3_p32_unit_common_production/final_checkpoint.pt"
        )
        review.gate(
            "checkpoint_dependencies_are_expected",
            dependencies.get(accepted_relative) == EXPECTED_ACCEPTED_CHECKPOINT_SHA256
            and dependencies.get(iteration3_relative) == EXPECTED_ITERATION3_CHECKPOINT_SHA256,
        )
        review.gate(
            "resolved_constants_match_protocol",
            resolved.get("device") == "cuda:0"
            and resolved.get("dtype")
            == "float32 model/HVP; FP64 gradient accumulation and diagnostics"
            and resolved.get("seed") == "none; exact full-basis gradients"
            and resolved.get("source_weight_index") == SOURCE_WEIGHT_INDEX
            and resolved.get("state_position") == STATE_POSITION
            and _close(resolved.get("epsilon"), EPSILON, atol=0.0, rtol=0.0)
            and _close(resolved.get("target_norm"), TARGET_NORM, atol=0.0, rtol=0.0)
            and _close(resolved.get("selection_alpha"), SELECTION_ALPHA, atol=0.0, rtol=0.0)
            and _close(resolved.get("armijo_c1"), ARMIJO_C1, atol=0.0, rtol=0.0)
            and _equivalent(resolved.get("line_alphas"), list(LINE_ALPHAS), atol=0.0, rtol=0.0)
            and resolved.get("max_accepted_updates") == MAX_ACCEPTED_UPDATES
            and _close(
                resolved.get("burg_gradient_formula_max_error"),
                BURG_GRADIENT_FORMULA_MAX_ERROR,
                atol=0.0,
                rtol=0.0,
            )
            and _equivalent(resolved.get("floors"), FLOORS, atol=0.0, rtol=0.0)
            and Path(str(resolved.get("output_dir"))).resolve() == DEFAULT_OUTPUT.resolve(),
        )
        widths = {
            name: _png_dimensions(output / name)
            for name in ("exact_selected_trajectory.png", "exact_selected_spectra.png")
        }
        review.gate(
            "plots_are_nontrivial_pngs",
            all(width >= 1000 and height >= 600 for width, height in widths.values()),
        )
        run_log = (output / "run.log").read_text(encoding="utf-8", errors="replace")
        review.gate(
            "run_log_records_clean_publication",
            "[exact-selected-i6] stage=load-fresh-run" in run_log
            and "[exact-selected-i6] stage=serialize-final-checkpoint" in run_log
            and "[exact-selected-i6] stage=disk-final-checkpoint-replay" in run_log
            and "[exact-selected-i6] complete valid=True" in run_log
            and "[exact-selected-i6] published output=" in run_log
            and "[exact-selected-i6] FAILED" not in run_log,
        )
        review.details["immutable_hash_review"] = {
            "executed_source_sha256": source_hash,
            "executed_normalized_source_sha256": normalized_source_hash,
            "protocol_sha256": protocol_hash,
            "dependency_manifest_sha256": dependency_hash,
            "artifact_matches": artifact_matches,
            "dependency_matches": dependency_matches,
            "plot_dimensions": {key: list(value) for key, value in widths.items()},
            "run_log_sha256_unmanifested": _sha256_file(output / "run.log"),
        }

    if not review.phase("metadata", metadata_phase):
        return 0 if _write_report(
            output,
            review,
            scientific_success=None,
            selected_arm=None,
            accepted_updates=None,
            termination=None,
        ) else 1

    def tables_phase() -> None:
        states = _read_csv(output / "state_metrics.csv")
        spectra = _read_csv(output / "state_spectra.csv")
        proposals = _read_csv(output / "proposal_diagnostics.csv")
        lines = _read_csv(output / "line_search.csv")
        selections = _read_csv(output / "arm_selection.csv")
        historical = _read_csv(output / "historical_transition_audit.csv")
        spectrum_review = _validate_spectrum_tables(states, spectra)
        context.update(
            states=states,
            spectra=spectra,
            proposals=proposals,
            lines=lines,
            selections=selections,
            historical=historical,
        )
        review.gate("state_and_spectrum_recompute", True)
        review.details["spectrum_review"] = spectrum_review

    if not review.phase("tables", tables_phase):
        return 0 if _write_report(
            output,
            review,
            scientific_success=None,
            selected_arm=None,
            accepted_updates=None,
            termination=None,
        ) else 1

    def checkpoint_phase() -> None:
        states: pd.DataFrame = context["states"]
        spectra: pd.DataFrame = context["spectra"]
        proposals: pd.DataFrame = context["proposals"]
        lines: pd.DataFrame = context["lines"]
        selections: pd.DataFrame = context["selections"]
        proposal0: dict[str, Any] = context["proposal0"]
        decision: dict[str, Any] = context["decision"]
        progress = _load_checkpoint(output / "progress_checkpoint.pt")
        final_checkpoint = _load_checkpoint(output / "final_checkpoint.pt")
        active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
        _require(len(active_names) == 31, "active name count mismatch")
        progress_hash, progress_count = _named_tensor_hash(
            active_names, progress["active_model_state"]
        )
        final_hash, final_count = _named_tensor_hash(
            active_names, final_checkpoint["active_model_state"]
        )
        _require(progress_count == final_count == ACTIVE_PARAMETER_COUNT, "active parameter count mismatch")
        _require(progress_hash == final_hash, "progress/final tensor hashes differ")
        _require(
            all(
                torch.equal(
                    progress["active_model_state"][name],
                    final_checkpoint["active_model_state"][name],
                )
                for name in active_names
            ),
            "progress/final tensors are not byte-equivalent",
        )
        final_state = states.iloc[-1]
        _require(
            progress_hash
            == progress.get("active_parameter_hash")
            == final_checkpoint.get("active_parameter_hash")
            == _text(final_state["parameter_hash"]),
            "final checkpoint/state parameter hash chain failed",
        )
        _require(
            progress.get("protocol_id") == final_checkpoint.get("protocol_id") == PROTOCOL_ID,
            "checkpoint protocol mismatch",
        )
        _require(
            progress.get("normalized_source_sha256") == EXPECTED_NORMALIZED_PRODUCER_SHA256
            and progress.get("dependency_manifest_sha256")
            == EXPECTED_DEPENDENCY_MANIFEST_SHA256,
            "progress immutable fingerprint mismatch",
        )
        _require(_bool(progress.get("terminal")), "final progress checkpoint is not terminal")
        _require(
            _int(progress.get("accepted_updates")) == len(states) - 1
            and _int(final_checkpoint.get("accepted_updates")) == len(states) - 1,
            "checkpoint accepted count mismatch",
        )
        _require(
            progress.get("selected_arm") == final_checkpoint.get("selected_arm"),
            "checkpoint selected arm mismatch",
        )
        _require(
            progress.get("termination") == final_checkpoint.get("termination"),
            "checkpoint termination mismatch",
        )
        _require(
            final_checkpoint.get("source_weight_index") == SOURCE_WEIGHT_INDEX
            and final_checkpoint.get("z_sha256") == EXPECTED_Z_SHA256,
            "final checkpoint state identity mismatch",
        )
        _require(
            _checkpoint_rows_match_csv(progress["state_rows"], states)
            and _checkpoint_rows_match_csv(progress["spectrum_rows"], spectra)
            and _checkpoint_rows_match_csv(progress["proposal_rows"], proposals)
            and _checkpoint_rows_match_csv(progress["line_rows"], lines)
            and _checkpoint_rows_match_csv(progress["selection_rows"], selections),
            "progress checkpoint table copies do not match CSV",
        )
        _require(
            _equivalent(progress.get("proposal0_payload"), proposal0, atol=1e-12, rtol=1e-12),
            "proposal-0 JSON/checkpoint payload mismatch",
        )
        _require(
            _equivalent(progress.get("initial_metrics"), decision.get("initial_metrics"), atol=1e-9, rtol=1e-9),
            "progress/decision initial metrics mismatch",
        )
        stored_final = final_checkpoint.get("stored_state_metrics")
        _require(isinstance(stored_final, Mapping), "final checkpoint stored metrics missing")
        _require(
            all(
                column in stored_final
                and _cell_equivalent(stored_final[column], final_state[column])
                for column in states.columns
            ),
            "final checkpoint stored state metrics mismatch",
        )
        final_file_hash = _sha256_file(output / "final_checkpoint.pt")
        _require(
            decision.get("final_checkpoint_file_sha256") == final_file_hash,
            "decision final checkpoint file hash mismatch",
        )

        i3_checkpoint = _load_checkpoint(ITERATION3_CHECKPOINT)
        i3_names = list(i3_checkpoint.get("active_names", []))
        _require(i3_names == active_names, "Iteration-3 active names mismatch")
        i3_hash, i3_count = _named_tensor_hash(active_names, i3_checkpoint["active_model_state"])
        _require(
            i3_count == ACTIVE_PARAMETER_COUNT
            and i3_hash
            == i3_checkpoint.get("final_parameter_hash")
            == EXPECTED_INITIAL_PARAMETER_SHA256
            == _text(states.iloc[0]["parameter_hash"]),
            "Iteration-3/start parameter hash chain failed",
        )
        review.gate("checkpoint_tensor_and_table_chain_recomputes", True)
        review.details["checkpoint_review"] = {
            "active_tensor_count": len(active_names),
            "active_parameter_count": final_count,
            "initial_parameter_sha256": i3_hash,
            "final_parameter_sha256": final_hash,
            "final_checkpoint_file_sha256": final_file_hash,
            "progress_checkpoint_file_sha256": _sha256_file(output / "progress_checkpoint.pt"),
            "cpu_tensor_equality_progress_vs_final": True,
        }
        context.update(
            progress=progress,
            final_checkpoint=final_checkpoint,
            final_parameter_hash=final_hash,
            final_checkpoint_file_sha256=final_file_hash,
        )

    if not review.phase("checkpoint", checkpoint_phase):
        return 0 if _write_report(
            output,
            review,
            scientific_success=None,
            selected_arm=None,
            accepted_updates=None,
            termination=None,
        ) else 1

    def protocol_phase() -> None:
        nonlocal selected_arm, accepted_updates, termination
        states: pd.DataFrame = context["states"]
        spectra: pd.DataFrame = context["spectra"]
        proposals: pd.DataFrame = context["proposals"]
        lines: pd.DataFrame = context["lines"]
        selections: pd.DataFrame = context["selections"]
        historical: pd.DataFrame = context["historical"]
        proposal0: dict[str, Any] = context["proposal0"]
        progress: dict[str, Any] = context["progress"]
        decision: dict[str, Any] = context["decision"]
        finalized: dict[str, Any] = context["finalized"]
        tolerances_raw = progress.get("tolerances")
        _require(isinstance(tolerances_raw, Mapping), "progress tolerances missing")
        tolerances = {key: _float(tolerances_raw[key]) for key in FLOORS}
        _require(
            all(tolerances[key] >= FLOORS[key] for key in FLOORS),
            "tolerance below protocol floor",
        )
        review.gate(
            "tolerances_consistent_across_artifacts",
            _equivalent(tolerances, proposal0.get("tolerances"), atol=0.0, rtol=0.0)
            and _equivalent(tolerances, decision.get("tolerances"), atol=0.0, rtol=0.0),
        )
        review.gate(
            "this_run_uses_frozen_tolerance_floors",
            _equivalent(tolerances, FLOORS, atol=0.0, rtol=0.0),
        )
        gradient_review = _validate_gradient_metadata(proposal0["gradient_metadata"])
        proposal0_review = _validate_proposal0(
            states=states,
            selections=selections,
            payload=proposal0,
            tolerances=tolerances,
        )
        selected_arm = proposal0_review["selected_arm"]
        accepted_updates = len(states) - 1
        termination = str(progress["termination"])
        history_review = _validate_history(
            states=states,
            proposals=proposals,
            lines=lines,
            selected_arm=selected_arm,
            tolerances=tolerances,
            termination=termination,
        )
        expected_historical = pd.DataFrame(history_review["audit_rows"])
        historical_match = (
            expected_historical.empty
            and historical.empty
            or _checkpoint_rows_match_csv(history_review["audit_rows"], historical)
        )
        review.gate("historical_audit_recomputes_exactly", historical_match)
        review.gate("proposal0_and_history_validate", True)

        i3_decision = _load_json(ITERATION3_DECISION)
        i3_review = _load_json(ITERATION3_REVIEW)
        i3_states = pd.read_csv(ITERATION3_STATES)
        i3_spectra = pd.read_csv(ITERATION3_SPECTRA)
        i3_state100 = i3_states.loc[i3_states["proposal"].eq(100)]
        i3_spectrum100 = i3_spectra.loc[i3_spectra["proposal"].eq(100)].sort_values("rank")
        _require(len(i3_state100) == 1 and len(i3_spectrum100) == DIMENSION, "Iteration-3 terminal state missing")
        initial = states.iloc[0]
        replay_keys = (
            "exact_a_per_dim",
            "damped_full_burg_per_dim",
            "m_max",
            "m_p50",
            "m_lt_1e_4_fraction",
            "m_lt_0p01_fraction",
            "m_lt_0p1_fraction",
        )
        i3_metric_error = max(
            abs(_float(initial[key]) - _float(i3_state100.iloc[0][key])) for key in replay_keys
        )
        initial_spectrum = spectra.loc[spectra["accepted_update"].eq(0)].sort_values("rank")
        i3_spectrum_error = float(
            np.max(
                np.abs(
                    initial_spectrum["m_eigenvalue"].to_numpy(dtype=np.float64)
                    - i3_spectrum100["m_eigenvalue"].to_numpy(dtype=np.float64)
                )
            )
        )
        review.gate(
            "iteration3_start_replays",
            i3_decision.get("valid") is True
            and i3_review.get("valid") is True
            and i3_metric_error <= 1e-9
            and i3_spectrum_error <= 1e-9,
        )

        i5_decision = _load_json(ITERATION5_DECISION)
        i5_review = _load_json(ITERATION5_REVIEW)
        i5_rows = pd.read_csv(ITERATION5_ENDPOINTS)
        i5_reference = i5_rows.loc[np.isclose(i5_rows["alpha"], SELECTION_ALPHA)]
        _require(len(i5_reference) == 1, "Iteration-5 low endpoint is not unique")
        low = selections.loc[selections["arm"].eq("low")]
        _require(len(low) == 1, "proposal-0 low arm missing")
        low = low.iloc[0]
        i5 = i5_reference.iloc[0]
        i5_pairs = (
            ("exact_a_per_dim", "exact_a_per_dim"),
            ("damped_full_burg_per_dim", "damped_full_burg_per_dim"),
            ("frozen_low_energy", "frozen_low_energy"),
            ("a_low90_abs_per_dim", "a_low90_abs_per_dim"),
            ("a_high_gt_1_abs_per_dim", "a_high_gt_1_abs_per_dim"),
            ("m_max", "m_max"),
            ("m_p50", "m_p50"),
        )
        i5_errors = {key: abs(_float(low[key]) - _float(i5[reference])) for key, reference in i5_pairs}
        i5_hash_match = _text(low["endpoint_parameter_hash"]) == _text(i5["endpoint_parameter_hash"])
        reported_i5_error = _float(low["iteration5_reference_max_abs_error"])
        review.gate(
            "iteration5_low_arm_replays",
            i5_decision.get("valid") is True
            and i5_review.get("valid") is True
            and max(i5_errors.values()) <= 1e-9
            and i5_hash_match
            and reported_i5_error <= 1e-9
            and abs(reported_i5_error - max(i5_errors.values())) <= 1e-12
            and _bool(low["iteration5_reference_hash_matches"]),
        )
        review.gate(
            "proposal_and_final_metadata_agree",
            decision.get("selected_arm") == selected_arm
            and finalized.get("selected_arm") == selected_arm
            and decision.get("accepted_updates") == accepted_updates
            and finalized.get("accepted_updates") == accepted_updates
            and decision.get("termination") == termination
            and finalized.get("termination") == termination,
        )
        review.details["proposal0_review"] = proposal0_review
        review.details["proposal0_gradient_review"] = gradient_review
        review.details["history_review"] = {
            key: value for key, value in history_review.items() if key != "audit_rows"
        }
        review.details["history_review"]["audit_rows"] = history_review["audit_rows"]
        review.details["upstream_replay_review"] = {
            "iteration3_metric_max_abs_error": i3_metric_error,
            "iteration3_spectrum_max_abs_error": i3_spectrum_error,
            "iteration5_low_endpoint_errors": i5_errors,
            "iteration5_low_endpoint_hash_matches": i5_hash_match,
        }
        context.update(tolerances=tolerances, history_review=history_review)

    if not review.phase("protocol", protocol_phase):
        return 0 if _write_report(
            output,
            review,
            scientific_success=None,
            selected_arm=selected_arm,
            accepted_updates=accepted_updates,
            termination=termination,
        ) else 1

    def outcome_phase() -> None:
        nonlocal scientific_success
        states: pd.DataFrame = context["states"]
        spectra: pd.DataFrame = context["spectra"]
        proposals: pd.DataFrame = context["proposals"]
        lines: pd.DataFrame = context["lines"]
        decision: dict[str, Any] = context["decision"]
        finalized: dict[str, Any] = context["finalized"]
        tolerances: dict[str, float] = context["tolerances"]
        history_review: dict[str, Any] = context["history_review"]
        outcome = _success_from_raw(states, tolerances=tolerances, history=history_review)
        scientific_success = bool(outcome["scientific_success"])
        review.gate(
            "success_gates_recompute",
            _equivalent(outcome["success_gates"], decision.get("success_gates"), atol=1e-12, rtol=1e-12)
            and _close(outcome["total_a_reduction"], decision.get("total_a_reduction"), atol=1e-12, rtol=1e-12)
            and _close(outcome["lower90_a_reduction"], decision.get("lower90_a_reduction"), atol=1e-12, rtol=1e-12)
            and _close(outcome["non_top_only_fraction"], decision.get("non_top_only_fraction"), atol=1e-12, rtol=1e-12)
            and outcome["tail_bulk_updates"] == decision.get("tail_bulk_updates"),
        )
        review.gate(
            "scientific_outcome_recomputes",
            decision.get("scientific_success") is scientific_success
            and finalized.get("scientific_success") is scientific_success,
        )
        producer_validity = decision.get("validity_gates")
        review.gate(
            "producer_declared_valid_but_not_used_as_evidence",
            decision.get("valid") is True
            and finalized.get("valid") is True
            and isinstance(producer_validity, Mapping)
            and all(value is True for value in producer_validity.values()),
        )

        tamper = _run_tamper_tests(
            states=states,
            spectra=spectra,
            proposals=proposals,
            lines=lines,
            selected_arm=str(selected_arm),
            tolerances=tolerances,
            termination=str(termination),
        )
        review.gate("tamper_tests_fail_closed", all(tamper.values()))
        review.details["tamper_tests"] = tamper
        review.details["recomputed_outcome"] = outcome

        gpu_replay = _run_independent_gpu_replay(
            output=output,
            device_name=str(args.device),
            states=states,
            spectra=spectra,
            proposals=proposals,
            selections=context["selections"],
            proposal0=context["proposal0"],
            final_checkpoint=context["final_checkpoint"],
            tolerances=tolerances,
        )
        review.gate("independent_gpu_disk_and_cancellation_replay_matches", True)
        review.details["independent_gpu_replay"] = gpu_replay

        final_update = len(states) - 1
        final_spectrum = (
            spectra.loc[spectra["accepted_update"].eq(final_update)]
            .sort_values("rank")["m_eigenvalue"]
            .to_numpy(dtype=np.float64)
        )
        if args.geometry_replay_json is not None:
            replay_payload = _load_json(args.geometry_replay_json.resolve())
            replay_review = _validate_geometry_replay_payload(
                replay_payload,
                final_checkpoint_sha256=context["final_checkpoint_file_sha256"],
                active_parameter_hash=context["final_parameter_hash"],
                final_state=states.iloc[-1],
                final_spectrum=final_spectrum,
            )
            review.gate("external_gpu_geometry_payload_matches", True)
        else:
            replay_review = {
                "performed": False,
                "required_schema": GEOMETRY_REPLAY_SCHEMA,
                "cli_hook": "--geometry-replay-json PATH",
            }
        review.details["external_geometry_replay"] = replay_review
        review.limitations.append(
            "The independent GPU pass recomputes proposal-0 and terminal proposal-18 gradients. "
            "Intermediate accepted-state gradient vectors were not stored and their parameter "
            "states cannot be reconstructed from endpoint hashes alone, so their cross-component "
            "slopes are checked from stored scalars plus direction identities and Armijo outcomes."
        )
        review.limitations.append(
            "The producer did not store the two untouched-base repeat evaluations used to choose "
            "tolerances. This run records exactly the frozen floors, and those floors are checked, "
            "but the branch condition that selected them is not independently reconstructible."
        )

    if not review.phase("outcome", outcome_phase):
        return 0 if _write_report(
            output,
            review,
            scientific_success=scientific_success,
            selected_arm=selected_arm,
            accepted_updates=accepted_updates,
            termination=termination,
        ) else 1

    valid = _write_report(
        output,
        review,
        scientific_success=scientific_success,
        selected_arm=selected_arm,
        accepted_updates=accepted_updates,
        termination=termination,
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
