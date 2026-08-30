from __future__ import annotations

import os
import sys

# Direct path execution otherwise lets scripts/inspect shadow the stdlib module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _SCRIPT_DIR:
    sys.path.pop(0)
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import argparse
import hashlib
import json
import math
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_a_realized_path_i2_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration2_realized_path_production"
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_2_realized_path/protocol.md"
PRODUCER_SOURCE = ROOT / "scripts/audit_one_state_exact_a_realized_path.py"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_2_frozen_dependency_manifest.json"
ITERATION1_DECISION = OUTPUT_ROOT / "iteration1_proposal3_cross_production/decision.json"
ITERATION5 = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration5_p32_training_production"
)
ITERATION5_STATES = ITERATION5 / "state_objective_curve.csv"
ITERATION5_PROPOSALS = ITERATION5 / "proposal_diagnostics.csv"
ACTIVE_PARAMETERS = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
)
CHECKPOINT = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
    / "vae_checkpoint.pt"
)

OLD_BETA = 22.536727828943093
EPSILON = 1e-4
TARGET_NORM = 0.04892722657548397
BLOCK_SIZE = 64
DIMENSION = 512
FD_RADII = (1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0, 1.0 / 512.0)
COMMON_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
DIRECTIONS = ("exact_a_raw", "exact_oldbeta_raw", "exact_unit_common_raw")
SIGNS = ("plus", "minus")
EXPECTED_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
EXPECTED_ARTIFACTS = {
    "resolved_config.json",
    "executed_source_snapshot.py",
    "frozen_dependency_manifest_snapshot.json",
    "reconstruction.csv",
    "base_repeatability.csv",
    "h_space_fd.csv",
    "gradient_method_comparison.csv",
    "realized_path.csv",
    "endpoint_metrics.csv",
    "parameter_block_realization.csv",
    "hessian_endpoints.npz",
    "exact_common_line.csv",
    "realized_path_and_common_line.png",
    "decision.json",
}

NOISE_COLUMNS = {
    "a": "exact_a_per_dim",
    "b": "damped_full_burg_per_dim",
    "m_max": "m_max",
    "m_p50": "m_p50",
    "m_lt_0p1_fraction": "m_lt_0p1_fraction",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _close(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> bool:
    try:
        return bool(
            np.isclose(float(left), float(right), atol=atol, rtol=rtol, equal_nan=False)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _equivalent(left: Any, right: Any, *, atol: float = 1e-10, rtol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return isinstance(left, (bool, np.bool_)) and isinstance(right, (bool, np.bool_)) and bool(left) == bool(right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
    ) or (
        isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        if not (
            isinstance(left, Sequence)
            and not isinstance(left, (str, bytes))
            and isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
            and len(left) == len(right)
        ):
            return False
        return all(
            _equivalent(l_value, r_value, atol=atol, rtol=rtol)
            for l_value, r_value in zip(left, right)
        )
    if isinstance(left, Real) and isinstance(right, Real):
        return _close(left, right, atol=atol, rtol=rtol)
    return left == right


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _float_grid_matches(values: pd.Series, expected: Sequence[float]) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if len(numeric) != len(expected) or not np.isfinite(numeric).all():
        return False
    matches = np.isclose(
        numeric[:, None],
        np.asarray(expected, dtype=np.float64)[None, :],
        rtol=0.0,
        atol=1e-12,
    )
    return bool(matches.sum(axis=0).tolist() == [1] * len(expected) and matches.sum(axis=1).tolist() == [1] * len(numeric))


def _one_at(frame: pd.DataFrame, column: str, value: float) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    selected = frame.loc[np.isclose(numeric, value, rtol=0.0, atol=1e-12)]
    if len(selected) != 1:
        raise ValueError(f"expected one row with {column}={value}, found {len(selected)}")
    return selected.iloc[0]


def _bool_values(series: pd.Series) -> np.ndarray:
    values: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
            continue
        normalized = str(value).strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"invalid boolean value {value!r}")
        values.append(normalized == "true")
    return np.asarray(values, dtype=bool)


def _all_numeric_finite(frames: Mapping[str, pd.DataFrame]) -> bool:
    for frame in frames.values():
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] and not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            return False
    return True


def _metric_algebra_is_consistent(frame: pd.DataFrame) -> bool:
    required = {
        "exact_a_per_dim",
        "a_constant_term",
        "a_linear_trace_term",
        "a_quartic_term",
        "trace_m_per_dim",
        "damped_full_burg_per_dim",
        "burg_trace_r_term",
        "burg_neg_logdet_r_term",
        "logdet_r_per_dim",
        "m_min",
        "m_p01",
        "m_p10",
        "m_p25",
        "m_p50",
        "m_p75",
        "m_p90",
        "m_p95",
        "m_p99",
        "m_max",
        "m_lt_1e_4_fraction",
        "m_lt_0p01_fraction",
        "m_lt_0p1_fraction",
        "m_lt_0p5_fraction",
        "m_near_1_10pct_fraction",
        "m_gt_1_fraction",
        "m_gt_2_fraction",
        "a_from_m_lt_0p1_share",
        "a_from_m_gt_1_share",
        "true_objective",
        "a_direct_matrix",
        "a_trace_closure",
        "a_direct_abs_error",
        "a_trace_abs_error",
        "effective_rank",
        "effective_rank_fraction",
        "li_gap_per_dim",
        "a_low90_abs_per_dim",
        "a_low_lt_0p1_abs_per_dim",
        "a_high_gt_1_abs_per_dim",
        "a_top1_share",
        "a_top10_share",
        "top1_trace_share",
        "top10_trace_share",
    }
    _require_columns(frame, required, "dense metric frame")
    if frame.empty:
        return False

    def values(column: str) -> np.ndarray:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)

    finite = np.isfinite(frame[list(required)].to_numpy(dtype=np.float64)).all()
    a = values("exact_a_per_dim")
    a_terms = values("a_constant_term") + values("a_linear_trace_term") + values("a_quartic_term")
    burg = values("burg_trace_r_term") + values("burg_neg_logdet_r_term") - 1.0
    trace_r = (values("trace_m_per_dim") + EPSILON) / (1.0 + EPSILON)
    objective = a + OLD_BETA * values("damped_full_burg_per_dim")
    closure = (
        np.isclose(a, a_terms, rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("damped_full_burg_per_dim"), burg, rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("burg_trace_r_term"), trace_r, rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("burg_neg_logdet_r_term"), -values("logdet_r_per_dim"), rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("true_objective"), objective, rtol=1e-9, atol=1e-9).all()
        and np.isclose(values("a_direct_abs_error"), np.abs(values("a_direct_matrix") - a), rtol=1e-7, atol=1e-12).all()
        and np.isclose(values("a_trace_abs_error"), np.abs(values("a_trace_closure") - a), rtol=1e-7, atol=1e-12).all()
        and np.isclose(values("effective_rank_fraction") * DIMENSION, values("effective_rank"), rtol=1e-9, atol=1e-9).all()
    )
    quantile_columns = (
        "m_min",
        "m_p01",
        "m_p10",
        "m_p25",
        "m_p50",
        "m_p75",
        "m_p90",
        "m_p95",
        "m_p99",
        "m_max",
    )
    quantiles = frame[list(quantile_columns)].to_numpy(dtype=np.float64)
    quantiles_ordered = bool((np.diff(quantiles, axis=1) >= -1e-10).all() and (quantiles[:, 0] >= -1e-10).all())
    fraction_columns = (
        "m_lt_1e_4_fraction",
        "m_lt_0p01_fraction",
        "m_lt_0p1_fraction",
        "m_lt_0p5_fraction",
        "m_near_1_10pct_fraction",
        "m_gt_1_fraction",
        "m_gt_2_fraction",
    )
    fractions = frame[list(fraction_columns)].to_numpy(dtype=np.float64)
    fractions_valid = bool(
        (fractions >= -1e-12).all()
        and (fractions <= 1.0 + 1e-12).all()
        and np.isclose(fractions * DIMENSION, np.rint(fractions * DIMENSION), rtol=0.0, atol=1e-8).all()
        and (fractions[:, 0] <= fractions[:, 1] + 1e-12).all()
        and (fractions[:, 1] <= fractions[:, 2] + 1e-12).all()
        and (fractions[:, 2] <= fractions[:, 3] + 1e-12).all()
        and (fractions[:, 6] <= fractions[:, 5] + 1e-12).all()
    )
    bounded_columns = (
        "a_from_m_lt_0p1_share",
        "a_from_m_gt_1_share",
        "a_top1_share",
        "a_top10_share",
        "top1_trace_share",
        "top10_trace_share",
        "effective_rank_fraction",
    )
    bounded = frame[list(bounded_columns)].to_numpy(dtype=np.float64)
    spectrum_bounds = bool(
        (a >= -1e-10).all()
        and (values("damped_full_burg_per_dim") >= -1e-10).all()
        and (values("li_gap_per_dim") >= -1e-10).all()
        and (values("a_low90_abs_per_dim") >= -1e-10).all()
        and (values("a_low90_abs_per_dim") <= a + 1e-8).all()
        and (values("a_low_lt_0p1_abs_per_dim") >= -1e-10).all()
        and (values("a_high_gt_1_abs_per_dim") >= -1e-10).all()
        and (bounded >= -1e-10).all()
        and (bounded <= 1.0 + 1e-8).all()
        and (values("a_top1_share") <= values("a_top10_share") + 1e-10).all()
        and (values("top1_trace_share") <= values("top10_trace_share") + 1e-10).all()
    )
    return bool(finite and closure and quantiles_ordered and fractions_valid and spectrum_bounds)


def _dense_metrics_from_hessian(hessian: np.ndarray) -> dict[str, float]:
    h64 = np.asarray(hessian, dtype=np.float64)
    matrix = h64 @ h64.T
    matrix = 0.5 * (matrix + matrix.T)
    raw_eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    dimension = int(eigenvalues.size)
    identity = np.eye(dimension, dtype=np.float64)
    contribution = np.square(eigenvalues - 1.0)
    contribution_total = max(float(contribution.sum()), 1e-30)
    trace = float(eigenvalues.sum())
    square_sum = max(float(np.square(eigenvalues).sum()), 1e-30)
    r_eigenvalues = (eigenvalues + EPSILON) / (1.0 + EPSILON)
    g_eigenvalues = (1.0 - np.reciprocal(r_eigenvalues)) / (
        float(dimension) * (1.0 + EPSILON)
    )
    quantiles = np.quantile(
        eigenvalues,
        [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
        method="linear",
    )
    a_value = float(contribution.mean())
    a_direct = float(np.square(matrix - identity).sum() / float(dimension))
    a_trace = float(1.0 - 2.0 * eigenvalues.mean() + np.square(eigenvalues).mean())
    trace_r = float(r_eigenvalues.mean())
    neg_logdet_r = float(-np.log(r_eigenvalues).mean())
    sorted_g = np.sort(g_eigenvalues)
    h_norm = max(float(np.linalg.norm(h64)), 1e-30)
    return {
        "hessian_symmetry_rel": float(np.linalg.norm(h64 - h64.T) / h_norm),
        "exact_a_per_dim": a_value,
        "a_constant_term": 1.0,
        "a_linear_trace_term": float(-2.0 * eigenvalues.mean()),
        "a_quartic_term": float(np.square(eigenvalues).mean()),
        "trace_m_per_dim": float(eigenvalues.mean()),
        "damped_full_burg_per_dim": trace_r + neg_logdet_r - 1.0,
        "burg_trace_r_term": trace_r,
        "burg_neg_logdet_r_term": neg_logdet_r,
        "logdet_r_per_dim": float(np.log(r_eigenvalues).mean()),
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
        "m_lt_1e_4_fraction": float(np.mean(eigenvalues < 1e-4)),
        "m_lt_0p01_fraction": float(np.mean(eigenvalues < 0.01)),
        "m_lt_0p1_fraction": float(np.mean(eigenvalues < 0.1)),
        "m_lt_0p5_fraction": float(np.mean(eigenvalues < 0.5)),
        "m_near_1_10pct_fraction": float(np.mean(np.abs(eigenvalues - 1.0) <= 0.1)),
        "m_gt_1_fraction": float(np.mean(eigenvalues > 1.0)),
        "m_gt_2_fraction": float(np.mean(eigenvalues > 2.0)),
        "a_from_m_lt_0p1_share": float(contribution[eigenvalues < 0.1].sum() / contribution_total),
        "a_from_m_gt_1_share": float(contribution[eigenvalues > 1.0].sum() / contribution_total),
        "burg_matrix_gradient_norm": float(np.linalg.norm(g_eigenvalues)),
        "burg_matrix_gradient_eig_min": float(g_eigenvalues.min()),
        "burg_matrix_gradient_eig_p50": float(sorted_g[(dimension - 1) // 2]),
        "burg_matrix_gradient_eig_max": float(g_eigenvalues.max()),
        "true_objective": a_value + OLD_BETA * (trace_r + neg_logdet_r - 1.0),
        "a_direct_matrix": a_direct,
        "a_trace_closure": a_trace,
        "a_direct_abs_error": abs(a_direct - a_value),
        "a_trace_abs_error": abs(a_trace - a_value),
        "m_raw_eig_min": float(raw_eigenvalues.min()),
        "effective_rank": trace * trace / square_sum,
        "effective_rank_fraction": trace * trace / (square_sum * float(dimension)),
        "li_gap_per_dim": float(np.square(np.sqrt(eigenvalues) - 1.0).mean()),
        "a_low90_abs_per_dim": float(
            contribution[: int(math.floor(0.9 * dimension))].sum() / float(dimension)
        ),
        "a_low_lt_0p1_abs_per_dim": float(
            contribution[eigenvalues < 0.1].sum() / float(dimension)
        ),
        "a_high_gt_1_abs_per_dim": float(
            contribution[eigenvalues > 1.0].sum() / float(dimension)
        ),
        "a_top1_share": float(contribution[-1] / contribution_total),
        "a_top10_share": float(contribution[-10:].sum() / contribution_total),
        "top1_trace_share": float(eigenvalues[-1] / max(trace, 1e-30)),
        "top10_trace_share": float(eigenvalues[-10:].sum() / max(trace, 1e-30)),
    }


def _scalarization_stats(inputs: Mapping[str, Any], beta: float) -> dict[str, Any]:
    norm_a = float(inputs["norm_a"])
    norm_b = float(inputs["norm_b"])
    dot_ab = float(inputs["dot_ab"])
    s_value = norm_a * norm_a + beta * dot_ab
    beta_crit = None if dot_ab >= 0.0 else -(norm_a * norm_a) / dot_ab
    return {
        "norm_a": norm_a,
        "norm_b": norm_b,
        "dot_ab": dot_ab,
        "cosine_ab": dot_ab / max(norm_a * norm_b, 1e-30),
        "s_value": s_value,
        "beta_crit": beta_crit,
        "conflict": bool(
            dot_ab < 0.0
            and beta_crit is not None
            and beta > beta_crit
            and s_value < 0.0
        ),
    }


def _candidate(
    line: pd.DataFrame,
    tolerances: Mapping[str, float],
) -> tuple[dict[str, Any], list[float]]:
    eligible = line.loc[pd.to_numeric(line["alpha"], errors="coerce").ge(0.125)].copy()
    passes = (
        eligible["delta_a"].lt(-float(tolerances["a"]))
        & eligible["delta_b"].lt(-float(tolerances["b"]))
        & eligible["delta_m_max"].le(float(tolerances["m_max"]))
        & eligible["delta_m_p50"].ge(-float(tolerances["m_p50"]))
        & eligible["delta_m_lt_0p1_fraction"].le(
            float(tolerances["m_lt_0p1_fraction"])
        )
    )
    passing = eligible.loc[passes].sort_values("alpha", ascending=False)
    passing_alphas = [float(value) for value in passing["alpha"].tolist()]
    return {
        "selected": bool(passing_alphas),
        "largest_passing_alpha": passing_alphas[0] if passing_alphas else None,
        "passing_count": len(passing_alphas),
        "tolerances": {key: float(value) for key, value in tolerances.items()},
    }, passing_alphas


class Review:
    def __init__(self) -> None:
        self.gates: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.errors: list[str] = []

    def gate(self, name: str, value: Any) -> bool:
        result = bool(value)
        self.gates[name] = result
        return result

    def phase(self, name: str, function: Callable[[], None]) -> bool:
        try:
            function()
        except Exception as error:  # Keep a durable failure report for malformed packets.
            self.gates[f"{name}_completed"] = False
            self.errors.append(f"{name}: {type(error).__name__}: {error}")
            self.details[f"{name}_traceback"] = traceback.format_exc(limit=8)
            return False
        self.gates[f"{name}_completed"] = True
        return True


def _write_report(output: Path, review: Review) -> bool:
    valid = bool(review.gates and all(review.gates.values()) and not review.errors)
    report = {
        "protocol_id": PROTOCOL_ID,
        "valid": valid,
        "gates": review.gates,
        "recomputed": review.details,
        "errors": review.errors,
    }
    report_path = output / "independent_review.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[exact-a-i2-review] report={report_path} valid={valid}", flush=True)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    review = Review()
    if not output.is_dir():
        print(f"[exact-a-i2-review] missing output directory: {output}", flush=True)
        return 1
    print(
        f"[exact-a-i2-review] start protocol={PROTOCOL_ID} output={output}",
        flush=True,
    )

    context: dict[str, Any] = {}

    def metadata_phase() -> None:
        finalized = json.loads((output / "FINALIZED.json").read_text(encoding="utf-8"))
        resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
        if not all(isinstance(value, dict) for value in (finalized, resolved, decision, manifest)):
            raise TypeError("metadata JSON roots must be objects")
        context.update(
            finalized=finalized,
            resolved=resolved,
            decision=decision,
            manifest=manifest,
        )

        review.gate("finalized_marker_and_no_incomplete", not (output / "INCOMPLETE").exists())
        review.gate(
            "protocol_ids_match",
            {finalized.get("protocol_id"), resolved.get("protocol_id"), decision.get("protocol_id")}
            == {PROTOCOL_ID},
        )
        producer_gates = decision.get("validity_gates")
        review.gate(
            "producer_declared_valid",
            finalized.get("valid") is True and decision.get("valid") is True,
        )
        review.gate(
            "decision_valid_is_producer_gate_conjunction",
            isinstance(producer_gates, dict)
            and bool(decision.get("valid")) == all(bool(value) for value in producer_gates.values()),
        )
        review.gate(
            "producer_validity_gates_all_true",
            isinstance(producer_gates, dict)
            and bool(producer_gates)
            and all(value is True for value in producer_gates.values()),
        )
        review.gate(
            "decision_hash_matches_finalized",
            finalized.get("decision_sha256") == _sha256_file(output / "decision.json"),
        )
        review.gate(
            "manifest_hash_matches_finalized",
            finalized.get("artifact_manifest_sha256")
            == _sha256_file(output / "artifact_manifest.json"),
        )
        review.gate(
            "manifest_has_exact_artifact_set",
            set(manifest) == EXPECTED_ARTIFACTS
            and all(_is_sha256(value) for value in manifest.values()),
        )
        review.gate(
            "all_manifest_artifacts_match",
            set(manifest) == EXPECTED_ARTIFACTS
            and all(
                (output / name).is_file() and _sha256_file(output / name) == digest
                for name, digest in manifest.items()
            ),
        )
        source_snapshot_hash = _sha256_file(output / "executed_source_snapshot.py")
        review.gate(
            "executed_source_hashes_match",
            source_snapshot_hash == resolved.get("source_sha256")
            and source_snapshot_hash == decision.get("source_sha256"),
        )
        review.gate(
            "current_producer_matches_executed_snapshot",
            PRODUCER_SOURCE.is_file() and _sha256_file(PRODUCER_SOURCE) == source_snapshot_hash,
        )
        dependency_snapshot = output / "frozen_dependency_manifest_snapshot.json"
        snapshot_hash = _sha256_file(dependency_snapshot)
        frozen_dependencies = json.loads(dependency_snapshot.read_text(encoding="utf-8"))
        if not isinstance(frozen_dependencies, dict):
            raise TypeError("frozen dependency snapshot must be a JSON object")
        frozen_dependency_results = {
            relative: (
                isinstance(relative, str)
                and _is_sha256(digest)
                and (ROOT / relative).is_file()
                and _sha256_file(ROOT / relative) == digest
            )
            for relative, digest in frozen_dependencies.items()
        }
        review.gate(
            "frozen_dependency_snapshot_hash_matches",
            snapshot_hash == resolved.get("frozen_dependency_manifest_sha256")
            and FROZEN_DEPENDENCY_MANIFEST.is_file()
            and _sha256_file(FROZEN_DEPENDENCY_MANIFEST) == snapshot_hash,
        )
        review.gate(
            "all_frozen_dependencies_still_match",
            bool(frozen_dependency_results)
            and all(frozen_dependency_results.values())
            and _equivalent(
                resolved.get("frozen_dependency_matches"),
                frozen_dependency_results,
            ),
        )
        review.details["frozen_dependency_matches"] = frozen_dependency_results
        protocol_hash = _sha256_file(PROTOCOL_PATH)
        review.gate(
            "protocol_hashes_match",
            protocol_hash == resolved.get("protocol_sha256")
            and protocol_hash == decision.get("protocol_sha256"),
        )
        dependency_checks = {
            "iteration1_decision": (
                ITERATION1_DECISION,
                resolved.get("iteration1_decision_sha256"),
            ),
            "iteration5_states": (
                ITERATION5_STATES,
                resolved.get("iteration5_states_sha256"),
            ),
            "iteration5_proposals": (
                ITERATION5_PROPOSALS,
                resolved.get("iteration5_proposals_sha256"),
            ),
        }
        dependency_results = {
            name: path.is_file() and _is_sha256(expected) and _sha256_file(path) == expected
            for name, (path, expected) in dependency_checks.items()
        }
        review.details["dependency_hashes_match"] = dependency_results
        review.gate("frozen_dependency_hashes_match", all(dependency_results.values()))
        checkpoint_matches = CHECKPOINT.is_file() and _sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256
        review.gate("checkpoint_hash_matches", checkpoint_matches)
        review.gate(
            "resolved_protocol_constants_match",
            resolved.get("state_position") == 2
            and resolved.get("source_weight_index") == 378
            and _close(resolved.get("beta"), OLD_BETA)
            and _close(resolved.get("burg_epsilon"), EPSILON)
            and _close(resolved.get("target_norm"), TARGET_NORM)
            and resolved.get("block_size") == BLOCK_SIZE
            and _equivalent(resolved.get("fd_radii"), list(FD_RADII), atol=1e-15, rtol=0.0)
            and _equivalent(resolved.get("common_alphas"), list(COMMON_ALPHAS), atol=1e-15, rtol=0.0),
        )
        review.gate(
            "plot_is_nonempty",
            (output / "realized_path_and_common_line.png").stat().st_size > 10_000,
        )

    if not review.phase("metadata", metadata_phase):
        return 0 if _write_report(output, review) else 1

    def table_phase() -> None:
        frames = {
            "reconstruction": pd.read_csv(output / "reconstruction.csv"),
            "base_repeatability": pd.read_csv(output / "base_repeatability.csv"),
            "h_space": pd.read_csv(output / "h_space_fd.csv"),
            "gradient_comparison": pd.read_csv(output / "gradient_method_comparison.csv"),
            "realized": pd.read_csv(output / "realized_path.csv"),
            "endpoints": pd.read_csv(output / "endpoint_metrics.csv"),
            "blocks": pd.read_csv(output / "parameter_block_realization.csv"),
            "common": pd.read_csv(output / "exact_common_line.csv"),
        }
        context["frames"] = frames
        reconstruction = frames["reconstruction"]
        base_repeatability = frames["base_repeatability"]
        h_space = frames["h_space"]
        gradient = frames["gradient_comparison"]
        realized = frames["realized"]
        endpoints = frames["endpoints"]
        blocks = frames["blocks"]
        common = frames["common"]

        _require_columns(reconstruction, {"proposal"}, "reconstruction.csv")
        _require_columns(base_repeatability, {"evaluation", *NOISE_COLUMNS.values()}, "base_repeatability.csv")
        _require_columns(h_space, {"objective", "delta", "relative_error"}, "h_space_fd.csv")
        _require_columns(gradient, {"component", "relative_error", "cosine"}, "gradient_method_comparison.csv")
        _require_columns(
            realized,
            {
                "direction",
                "radius",
                "nominal_a_slope",
                "nominal_b_slope",
                "nominal_f_slope",
                "cauchy_scale_a",
                "cauchy_scale_f",
                "nominal_direction_norm",
                "effective_direction_norm",
                "effective_to_nominal_norm_ratio",
                "effective_to_nominal_cosine",
                "midpoint_drift_norm",
                "midpoint_drift_relative_to_step",
                "slope_theta",
                "slope_theta_f",
                "slope_hessian",
                "slope_a_direct",
                "slope_b",
                "slope_hessian_f",
                "slope_f_direct",
                "theta_to_h_normalized_error",
                "h_to_a_normalized_error",
                "theta_to_h_f_normalized_error",
                "h_to_f_normalized_error",
            },
            "realized_path.csv",
        )
        _require_columns(
            endpoints,
            {
                "direction",
                "radius",
                "sign",
                "parameter_hash",
                "hessian_hash",
                "repeat_hessian_hash",
                "repeat_matches",
                "repeat_a_abs_error",
                "repeat_b_abs_error",
                "a_direct_matrix",
                "damped_full_burg_per_dim",
            },
            "endpoint_metrics.csv",
        )
        _require_columns(
            blocks,
            {
                "direction",
                "radius",
                "parameter",
                "nominal_norm",
                "effective_norm",
                "effective_minus_nominal_norm",
                "midpoint_drift_norm",
            },
            "parameter_block_realization.csv",
        )
        _require_columns(
            common,
            {
                "alpha",
                "delta_a",
                "delta_b",
                "delta_m_max",
                "delta_m_p50",
                "delta_m_lt_0p1_fraction",
                *NOISE_COLUMNS.values(),
            },
            "exact_common_line.csv",
        )
        review.gate("all_csv_numeric_values_finite", _all_numeric_finite(frames))
        review.gate(
            "reconstruction_rows_complete_unique",
            len(reconstruction) == 2
            and not reconstruction.duplicated(["proposal"]).any()
            and set(pd.to_numeric(reconstruction["proposal"], errors="coerce")) == {1, 2},
        )
        review.gate(
            "base_repeatability_rows_complete_unique",
            len(base_repeatability) == 2
            and base_repeatability["evaluation"].is_unique
            and set(base_repeatability["evaluation"].astype(str)) == {"primary", "repeat"},
        )
        review.gate(
            "h_space_rows_complete_unique",
            len(h_space) == 4
            and not h_space.duplicated(["objective", "delta"]).any()
            and set(h_space["objective"].astype(str)) == {"a", "oldbeta"}
            and all(
                _float_grid_matches(group["delta"], (1e-5, 5e-6))
                for _, group in h_space.groupby("objective")
            ),
        )
        review.gate(
            "gradient_rows_complete_unique",
            len(gradient) == 3
            and gradient["component"].is_unique
            and set(gradient["component"].astype(str)) == {"a", "b", "oldbeta"},
        )
        realized_grid = bool(
            len(realized) == len(DIRECTIONS) * len(FD_RADII)
            and not realized.duplicated(["direction", "radius"]).any()
            and set(realized["direction"].astype(str)) == set(DIRECTIONS)
            and all(
                _float_grid_matches(group["radius"], FD_RADII)
                for _, group in realized.groupby("direction")
            )
        )
        review.gate("realized_path_keys_complete_unique", realized_grid)
        endpoint_grid = bool(
            len(endpoints) == len(DIRECTIONS) * len(FD_RADII) * len(SIGNS)
            and not endpoints.duplicated(["direction", "radius", "sign"]).any()
            and set(endpoints["direction"].astype(str)) == set(DIRECTIONS)
            and set(endpoints["sign"].astype(str)) == set(SIGNS)
            and all(
                len(group) == len(FD_RADII) * len(SIGNS)
                and all(
                    _float_grid_matches(sign_group["radius"], FD_RADII)
                    for _, sign_group in group.groupby("sign")
                )
                for _, group in endpoints.groupby("direction")
            )
        )
        review.gate("endpoint_keys_complete_unique", endpoint_grid)
        active = pd.read_csv(ACTIVE_PARAMETERS)
        _require_columns(active, {"parameter"}, str(ACTIVE_PARAMETERS))
        expected_parameters = set(active["parameter"].astype(str))
        block_grid = bool(
            expected_parameters
            and set(blocks["direction"].astype(str)) == set(DIRECTIONS)
            and set(blocks["parameter"].astype(str)) == expected_parameters
            and not blocks.duplicated(["direction", "radius", "parameter"]).any()
            and len(blocks) == len(DIRECTIONS) * len(FD_RADII) * len(expected_parameters)
            and all(
                len(group) == len(FD_RADII) * len(expected_parameters)
                and set(group["parameter"].astype(str)) == expected_parameters
                and all(
                    set(radius_group["parameter"].astype(str)) == expected_parameters
                    for _, radius_group in group.groupby("radius")
                )
                and _float_grid_matches(group.drop_duplicates("radius")["radius"], FD_RADII)
                for _, group in blocks.groupby("direction")
            )
        )
        review.gate("parameter_block_keys_complete_unique", block_grid)
        review.details["active_parameter_count"] = len(expected_parameters)
        review.gate(
            "common_line_keys_complete_unique",
            len(common) == len(COMMON_ALPHAS)
            and not common.duplicated(["alpha"]).any()
            and _float_grid_matches(common["alpha"], COMMON_ALPHAS),
        )
        hash_columns = endpoints[["parameter_hash", "hessian_hash", "repeat_hessian_hash"]]
        review.gate(
            "endpoint_hash_fields_well_formed",
            all(_is_sha256(value) for value in hash_columns.to_numpy().ravel().tolist()),
        )

    if not review.phase("tables", table_phase):
        return 0 if _write_report(output, review) else 1

    frames: dict[str, pd.DataFrame] = context["frames"]
    decision: dict[str, Any] = context["decision"]
    resolved: dict[str, Any] = context["resolved"]
    recomputed_validity: dict[str, bool] = {}

    def basic_numeric_phase() -> None:
        reconstruction = frames["reconstruction"]
        base_repeatability = frames["base_repeatability"]
        h_space = frames["h_space"]
        gradient = frames["gradient_comparison"]
        endpoints = frames["endpoints"]
        common = frames["common"]

        abs_columns = [column for column in reconstruction if column.endswith("_abs_error")]
        relative_columns = [column for column in reconstruction if column.endswith("_relative_error")]
        replay_pass = bool(
            abs_columns
            and relative_columns
            and reconstruction[abs_columns].to_numpy(dtype=np.float64).max() <= 1e-5
            and reconstruction[relative_columns].to_numpy(dtype=np.float64).max() <= 1e-6
        )
        review.gate("reconstruction_errors_within_tolerance", replay_pass)
        recomputed_validity["replay_matches"] = replay_pass

        primary = base_repeatability.loc[base_repeatability["evaluation"].eq("primary")].iloc[0]
        repeat = base_repeatability.loc[base_repeatability["evaluation"].eq("repeat")].iloc[0]
        base_noise = {
            key: abs(float(primary[column]) - float(repeat[column]))
            for key, column in NOISE_COLUMNS.items()
        }
        tolerances = {
            key: max(5.0 * value, 1e-8 if key in {"a", "b", "m_max"} else 1e-12)
            for key, value in base_noise.items()
        }
        context["base_noise"] = base_noise
        context["tolerances"] = tolerances
        base_repeatable = max(base_noise.values()) <= 1e-6
        review.gate("base_repeatability_within_tolerance", base_repeatable)
        review.gate("base_noise_recomputes", _equivalent(base_noise, decision["base_noise"], atol=1e-12, rtol=1e-9))
        review.gate(
            "base_metrics_match_primary_row",
            all(
                key in primary.index and _close(value, primary[key], atol=1e-10, rtol=1e-9)
                for key, value in decision["base_metrics"].items()
                if isinstance(value, Real) and not isinstance(value, bool)
            ),
        )
        recomputed_validity["base_repeatable"] = base_repeatable
        review.details["base_noise"] = base_noise
        review.details["candidate_tolerances"] = tolerances

        h_space_pass = bool(h_space["relative_error"].max() <= 1e-5)
        review.gate("h_space_fd_within_tolerance", h_space_pass)
        recomputed_validity["h_space_fd_passes"] = h_space_pass

        gradient_map = {
            str(row.component): {
                "relative_error": float(row.relative_error),
                "cosine": float(row.cosine),
            }
            for row in gradient.itertuples(index=False)
        }
        gradient_pass = all(
            metrics["relative_error"] <= 1e-4 and metrics["cosine"] >= 0.999999
            for metrics in gradient_map.values()
        )
        review.gate("gradient_methods_within_tolerance", gradient_pass)
        review.gate(
            "gradient_comparison_matches_decision",
            _equivalent(gradient_map, decision["gradient_comparison"], atol=1e-12, rtol=1e-9),
        )
        recomputed_validity["gradient_methods_agree"] = gradient_pass

        repeat_flags = _bool_values(endpoints["repeat_matches"])
        endpoint_repeat_pass = bool(
            repeat_flags.all()
            and (endpoints["hessian_hash"].astype(str) == endpoints["repeat_hessian_hash"].astype(str)).all()
            and endpoints["repeat_a_abs_error"].max() <= 1e-10
            and endpoints["repeat_b_abs_error"].max() <= 1e-10
        )
        review.gate("endpoint_repeatability_recomputes", endpoint_repeat_pass)
        recomputed_validity["endpoint_repeats_match"] = endpoint_repeat_pass

        metric_frames = {
            "base_repeatability": base_repeatability,
            "endpoints": endpoints,
            "common": common,
        }
        metric_algebra_pass = all(_metric_algebra_is_consistent(frame) for frame in metric_frames.values())
        review.gate("dense_metric_and_spectrum_algebra_recomputes", metric_algebra_pass)
        review.gate(
            "a_matrix_and_trace_closures",
            all(
                frame[["a_direct_abs_error", "a_trace_abs_error"]].to_numpy(dtype=np.float64).max()
                <= 1e-10
                for frame in metric_frames.values()
            ),
        )

        base = decision["base_metrics"]
        delta_columns = {
            "delta_a": "exact_a_per_dim",
            "delta_b": "damped_full_burg_per_dim",
            "delta_m_max": "m_max",
            "delta_m_p50": "m_p50",
            "delta_m_lt_0p1_fraction": "m_lt_0p1_fraction",
        }
        delta_checks = [
            _close(
                row[delta_column],
                float(row[metric_column]) - float(base[metric_column]),
                atol=1e-10,
                rtol=1e-9,
            )
            for _, row in common.iterrows()
            for delta_column, metric_column in delta_columns.items()
        ]
        review.gate("common_line_deltas_recompute", all(delta_checks))
        candidate, passing_alphas = _candidate(common, tolerances)
        context["candidate"] = candidate
        review.details["common_candidate"] = candidate
        review.details["passing_common_alphas"] = passing_alphas
        review.gate(
            "common_candidate_recomputes",
            _equivalent(
                candidate if decision["valid"] else None,
                decision["common_candidate"],
                atol=1e-12,
                rtol=1e-9,
            ),
        )
        deployment = decision["deployment_rule"]
        review.gate(
            "deployment_largest_passing_alpha_recomputes",
            deployment.get("update_map") == "raw_exact_unit_common_without_adam_or_clip"
            and deployment.get("alpha_rule") == "largest_passing_alpha"
            and _equivalent(
                deployment.get("selected_alpha"),
                candidate["largest_passing_alpha"] if decision["valid"] else None,
                atol=1e-12,
                rtol=0.0,
            ),
        )

    review.phase("basic_numeric", basic_numeric_phase)

    def scalarization_phase() -> None:
        blocked = _scalarization_stats(decision["scalarization_blocked"], float(resolved["beta"]))
        sequential = _scalarization_stats(decision["scalarization_sequential"], float(resolved["beta"]))
        disagreement = abs(float(blocked["s_value"]) - float(sequential["s_value"]))
        margin_pass = bool(
            blocked["conflict"]
            and sequential["conflict"]
            and min(-float(blocked["s_value"]), -float(sequential["s_value"]))
            > 5.0 * disagreement
        )
        review.gate(
            "blocked_scalarization_algebra_recomputes",
            _equivalent(blocked, decision["scalarization_blocked"], atol=1e-10, rtol=1e-9),
        )
        review.gate(
            "sequential_scalarization_algebra_recomputes",
            _equivalent(sequential, decision["scalarization_sequential"], atol=1e-10, rtol=1e-9),
        )
        review.gate("scalarization_disagreement_recomputes", _close(disagreement, decision["s_disagreement"], atol=1e-12, rtol=1e-9))
        review.gate("scalarization_conflict_margin_passes", margin_pass)
        review.gate(
            "scalarization_decision_recomputes",
            decision.get("scalarization_conflict_confirmed")
            == (margin_pass if decision["valid"] else None),
        )
        recomputed_validity["scalarization_margin_passes"] = margin_pass
        review.details["scalarization"] = {
            "blocked": blocked,
            "sequential": sequential,
            "s_disagreement": disagreement,
            "margin_pass": margin_pass,
        }

        norm_a = float(blocked["norm_a"])
        norm_b = float(blocked["norm_b"])
        dot_ab = float(blocked["dot_ab"])
        cosine = float(blocked["cosine_ab"])
        common_source_norm = math.sqrt(max(0.0, 2.0 + 2.0 * cosine))
        review.gate(
            "common_source_norm_recomputes",
            _close(common_source_norm, decision["common_source_norm"], atol=5e-8, rtol=1e-8),
        )
        cancellation_pass = common_source_norm >= 0.5
        review.gate("common_cancellation_margin_passes", cancellation_pass)
        recomputed_validity["common_cancellation_margin_passes"] = cancellation_pass

        slopes = decision["common_nominal_slopes"]
        blocked_common_a = -TARGET_NORM * (norm_a + dot_ab / norm_b) / common_source_norm
        blocked_common_b = -TARGET_NORM * (dot_ab / norm_a + norm_b) / common_source_norm
        review.gate(
            "blocked_common_nominal_slopes_recompute",
            _close(blocked_common_a, slopes["blocked_a"], atol=1e-8, rtol=5e-5)
            and _close(blocked_common_b, slopes["blocked_b"], atol=1e-8, rtol=5e-5),
        )
        common_descends = all(float(value) < 0.0 for value in slopes.values())
        review.gate("common_nominal_descends_both_methods", common_descends)
        recomputed_validity["common_nominal_descends_both"] = common_descends

        old_norm = math.sqrt(max(0.0, norm_a * norm_a + 2.0 * OLD_BETA * dot_ab + OLD_BETA * OLD_BETA * norm_b * norm_b))
        expected_nominal = {
            "exact_a_raw": (
                -TARGET_NORM * norm_a,
                -TARGET_NORM * dot_ab / norm_a,
            ),
            "exact_oldbeta_raw": (
                -TARGET_NORM * (norm_a * norm_a + OLD_BETA * dot_ab) / old_norm,
                -TARGET_NORM * (dot_ab + OLD_BETA * norm_b * norm_b) / old_norm,
            ),
            "exact_unit_common_raw": (blocked_common_a, blocked_common_b),
        }
        nominal_checks: list[bool] = []
        realized = frames["realized"]
        for direction, (expected_a, expected_b) in expected_nominal.items():
            rows = realized.loc[realized["direction"].eq(direction)]
            nominal_checks.extend(
                [
                    np.isclose(rows["nominal_direction_norm"], TARGET_NORM, atol=1e-6, rtol=0.0).all(),
                    np.isclose(rows["nominal_a_slope"], expected_a, atol=1e-8, rtol=5e-5).all(),
                    np.isclose(rows["nominal_b_slope"], expected_b, atol=1e-8, rtol=5e-5).all(),
                    np.isclose(
                        rows["nominal_f_slope"],
                        rows["nominal_a_slope"] + OLD_BETA * rows["nominal_b_slope"],
                        atol=1e-7,
                        rtol=1e-8,
                    ).all(),
                ]
            )
        review.gate("realized_nominal_slopes_match_scalarization_inputs", all(nominal_checks))

    review.phase("scalarization", scalarization_phase)

    def realized_phase() -> None:
        realized = frames["realized"]
        endpoints = frames["endpoints"]
        blocks = frames["blocks"]
        geometry_checks: list[bool] = []
        secant_checks: list[bool] = []
        normalized_error_checks: list[bool] = []
        for direction in DIRECTIONS:
            direction_realized = realized.loc[realized["direction"].eq(direction)]
            direction_endpoints = endpoints.loc[endpoints["direction"].eq(direction)]
            direction_blocks = blocks.loc[blocks["direction"].eq(direction)]
            for radius in FD_RADII:
                row = _one_at(direction_realized, "radius", radius)
                block_rows = direction_blocks.loc[
                    np.isclose(direction_blocks["radius"], radius, rtol=0.0, atol=1e-12)
                ]
                nominal2 = float(np.square(block_rows["nominal_norm"]).sum())
                effective2 = float(np.square(block_rows["effective_norm"]).sum())
                difference2 = float(np.square(block_rows["effective_minus_nominal_norm"]).sum())
                midpoint2 = float(np.square(block_rows["midpoint_drift_norm"]).sum())
                nominal_norm = math.sqrt(max(0.0, nominal2))
                effective_norm = math.sqrt(max(0.0, effective2))
                dot = 0.5 * (nominal2 + effective2 - difference2)
                cosine = dot / max(nominal_norm * effective_norm, 1e-30)
                midpoint_norm = math.sqrt(max(0.0, midpoint2))
                geometry_checks.extend(
                    [
                        _close(row["nominal_direction_norm"], nominal_norm, atol=1e-9, rtol=1e-8),
                        _close(row["effective_direction_norm"], effective_norm, atol=1e-9, rtol=1e-8),
                        _close(row["effective_to_nominal_norm_ratio"], effective_norm / max(nominal_norm, 1e-30), atol=1e-9, rtol=1e-8),
                        _close(row["effective_to_nominal_cosine"], cosine, atol=1e-8, rtol=1e-8),
                        _close(row["midpoint_drift_norm"], midpoint_norm, atol=1e-9, rtol=1e-8),
                        _close(row["midpoint_drift_relative_to_step"], midpoint_norm / max(radius * nominal_norm, 1e-30), atol=1e-8, rtol=1e-8),
                    ]
                )
                radius_endpoints = direction_endpoints.loc[
                    np.isclose(direction_endpoints["radius"], radius, rtol=0.0, atol=1e-12)
                ]
                plus = radius_endpoints.loc[radius_endpoints["sign"].eq("plus")].iloc[0]
                minus = radius_endpoints.loc[radius_endpoints["sign"].eq("minus")].iloc[0]
                slope_a = (float(plus["a_direct_matrix"]) - float(minus["a_direct_matrix"])) / (2.0 * radius)
                slope_b = (
                    float(plus["damped_full_burg_per_dim"])
                    - float(minus["damped_full_burg_per_dim"])
                ) / (2.0 * radius)
                slope_f = slope_a + OLD_BETA * slope_b
                secant_checks.extend(
                    [
                        _close(row["slope_a_direct"], slope_a, atol=1e-9, rtol=1e-9),
                        _close(row["slope_b"], slope_b, atol=1e-9, rtol=1e-9),
                        _close(row["slope_f_direct"], slope_f, atol=1e-8, rtol=1e-9),
                    ]
                )
                recomputed_errors = {
                    "theta_to_h_normalized_error": abs(float(row["slope_theta"]) - float(row["slope_hessian"])) / max(float(row["cauchy_scale_a"]), 1e-30),
                    "h_to_a_normalized_error": abs(float(row["slope_hessian"]) - float(row["slope_a_direct"])) / max(float(row["cauchy_scale_a"]), 1e-30),
                    "theta_to_h_f_normalized_error": abs(float(row["slope_theta_f"]) - float(row["slope_hessian_f"])) / max(float(row["cauchy_scale_f"]), 1e-30),
                    "h_to_f_normalized_error": abs(float(row["slope_hessian_f"]) - float(row["slope_f_direct"])) / max(float(row["cauchy_scale_f"]), 1e-30),
                }
                normalized_error_checks.extend(
                    _close(row[key], value, atol=1e-10, rtol=1e-9)
                    for key, value in recomputed_errors.items()
                )
        review.gate("parameter_block_geometry_recomputes", all(geometry_checks))
        review.gate("endpoint_direct_secants_recompute", all(secant_checks))
        review.gate("normalized_chain_errors_recompute", all(normalized_error_checks))

        old = realized.loc[realized["direction"].eq("exact_oldbeta_raw")]
        coarse = old.loc[
            np.isclose(old["radius"].to_numpy(dtype=np.float64)[:, None], np.asarray(FD_RADII[:3])[None, :], rtol=0.0, atol=1e-12).any(axis=1)
        ]
        a_sign_pass = bool(
            len(coarse) == 3
            and (coarse[["nominal_a_slope", "slope_theta", "slope_hessian", "slope_a_direct"]] > 0.0).all().all()
        )
        f_sign_pass = bool(
            len(coarse) == 3
            and (coarse[["nominal_f_slope", "slope_theta_f", "slope_hessian_f", "slope_f_direct"]] < 0.0).all().all()
        )
        chain_fidelity_pass = bool(
            len(coarse) == 3
            and coarse[
                [
                    "theta_to_h_normalized_error",
                    "h_to_a_normalized_error",
                    "theta_to_h_f_normalized_error",
                    "h_to_f_normalized_error",
                ]
            ].to_numpy(dtype=np.float64).max()
            <= 0.01
        )
        parameter_fidelity_pass = bool(
            len(coarse) == 3
            and coarse["effective_to_nominal_cosine"].min() >= 0.95
            and coarse["effective_to_nominal_norm_ratio"].between(0.9, 1.1).all()
            and coarse["midpoint_drift_relative_to_step"].max() <= 0.25
        )
        realized_pass = a_sign_pass and f_sign_pass and chain_fidelity_pass and parameter_fidelity_pass
        review.gate("realized_oldbeta_a_signs_pass", a_sign_pass)
        review.gate("realized_oldbeta_f_signs_pass", f_sign_pass)
        review.gate("realized_oldbeta_chain_fidelity_pass", chain_fidelity_pass)
        review.gate("realized_oldbeta_parameter_fidelity_pass", parameter_fidelity_pass)
        review.gate(
            "realized_oldbeta_decision_recomputes",
            decision.get("realized_path_confirmed")
            == (realized_pass if decision["valid"] else None),
        )
        recomputed_validity["realized_exact_oldbeta_sign_passes"] = realized_pass
        review.details["realized_oldbeta"] = {
            "a_sign_pass": a_sign_pass,
            "f_sign_pass": f_sign_pass,
            "chain_fidelity_pass": chain_fidelity_pass,
            "parameter_fidelity_pass": parameter_fidelity_pass,
            "combined_pass": realized_pass,
            "max_chain_error": float(
                coarse[
                    [
                        "theta_to_h_normalized_error",
                        "h_to_a_normalized_error",
                        "theta_to_h_f_normalized_error",
                        "h_to_f_normalized_error",
                    ]
                ].to_numpy(dtype=np.float64).max()
            ),
        }

    review.phase("realized_path", realized_phase)

    def hessian_archive_phase() -> None:
        endpoints = frames["endpoints"]
        expected_keys = {
            f"{direction}_h{int(round(1.0 / radius))}_{sign}"
            for direction in DIRECTIONS
            for radius in FD_RADII
            for sign in SIGNS
        }
        hash_checks: list[bool] = []
        spectrum_checks: list[bool] = []
        max_spectrum_error = 0.0
        failed_spectrum_fields: list[str] = []
        started = time.perf_counter()
        with np.load(output / "hessian_endpoints.npz", allow_pickle=False) as archive:
            review.gate("hessian_archive_keys_complete_unique", set(archive.files) == expected_keys)
            for index, key in enumerate(sorted(expected_keys), start=1):
                hessian = np.asarray(archive[key])
                parts = key.rsplit("_h", 1)
                direction = parts[0]
                denominator_text, sign = parts[1].rsplit("_", 1)
                radius = 1.0 / float(denominator_text)
                endpoint_rows = endpoints.loc[
                    endpoints["direction"].eq(direction)
                    & endpoints["sign"].eq(sign)
                    & np.isclose(endpoints["radius"], radius, rtol=0.0, atol=1e-12)
                ]
                if len(endpoint_rows) != 1:
                    raise ValueError(f"archive key {key} does not map to one endpoint row")
                endpoint = endpoint_rows.iloc[0]
                array_hash = _sha256_array(hessian)
                hash_checks.extend(
                    [
                        hessian.shape == (DIMENSION, DIMENSION),
                        np.isfinite(hessian).all(),
                        array_hash == str(endpoint["hessian_hash"]),
                        array_hash == str(endpoint["repeat_hessian_hash"]),
                    ]
                )
                recomputed = _dense_metrics_from_hessian(hessian)
                for field, expected in recomputed.items():
                    if field not in endpoint.index:
                        spectrum_checks.append(False)
                        failed_spectrum_fields.append(f"{key}:{field}:missing")
                        continue
                    actual = float(endpoint[field])
                    difference = abs(actual - expected)
                    max_spectrum_error = max(max_spectrum_error, difference)
                    matches = _close(actual, expected, atol=5e-8, rtol=2e-7)
                    spectrum_checks.append(matches)
                    if not matches and len(failed_spectrum_fields) < 30:
                        failed_spectrum_fields.append(
                            f"{key}:{field}:stored={actual:.12g}:recomputed={expected:.12g}"
                        )
                if not args.quiet:
                    print(
                        f"[exact-a-i2-review] stage=spectrum endpoint={index}/{len(expected_keys)} "
                        f"key={key} elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
        review.gate("archived_hessian_hashes_match_endpoint_rows", all(hash_checks))
        review.gate("endpoint_spectra_recompute_from_archived_hessians", all(spectrum_checks))
        review.details["endpoint_spectrum_review"] = {
            "endpoint_count": len(expected_keys),
            "recomputed_field_count": len(spectrum_checks),
            "max_absolute_error": max_spectrum_error,
            "failed_fields": failed_spectrum_fields,
            "elapsed_sec": time.perf_counter() - started,
        }

    review.phase("hessian_archive", hessian_archive_phase)

    def validity_reconciliation_phase() -> None:
        producer_gates = decision["validity_gates"]
        review.details["recomputed_validity_gates"] = recomputed_validity
        review.gate(
            "recomputed_validity_gates_match_decision",
            all(key in producer_gates and bool(producer_gates[key]) == value for key, value in recomputed_validity.items()),
        )
        review.gate(
            "producer_only_validity_gates_present",
            {
                "checkpoint_matches",
                "full_ce",
                "all_exact_basis_counts_512",
                "no_fully_unused_tensors",
                "memory_growth_passes",
                "parameters_restored",
                "cuda_rng_unchanged",
            }.issubset(producer_gates),
        )
        review.gate(
            "producer_checkpoint_gate_matches_independent_hash",
            bool(producer_gates["checkpoint_matches"]) == review.gates["checkpoint_hash_matches"],
        )
        numeric_finite = bool(
            review.gates["all_csv_numeric_values_finite"]
            and review.gates.get("archived_hessian_hashes_match_endpoint_rows", False)
        )
        recomputed_validity["numeric_finite"] = numeric_finite
        review.gate(
            "numeric_finite_gate_matches_decision",
            bool(producer_gates["numeric_finite"]) == numeric_finite,
        )

    review.phase("validity_reconciliation", validity_reconciliation_phase)
    return 0 if _write_report(output, review) else 1


if __name__ == "__main__":
    raise SystemExit(main())
