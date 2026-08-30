from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_v1"
REVIEW_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_cpu_review_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "postgoal_relaxed_cancellation_production"
PARENT_OUTPUT = OUTPUT_ROOT / "iteration6_exact_selected_trajectory_production"
PARENT_PROGRESS = PARENT_OUTPUT / "progress_checkpoint.pt"
PARENT_FINAL = PARENT_OUTPUT / "final_checkpoint.pt"
PARENT_REVIEW = PARENT_OUTPUT / "independent_review.json"
PRODUCER_SOURCE = (
    ROOT / "scripts/run_one_state_exact_selected_trajectory_continuation.py"
)
PROTOCOL_PATH = OUTPUT_ROOT / "postgoal_relaxed_cancellation/protocol.md"
DEPENDENCY_MANIFEST = (
    OUTPUT_ROOT / "postgoal_relaxed_cancellation/frozen_dependency_manifest.json"
)

EXPECTED_PRODUCER_SHA256 = (
    "d79462816ff95b2c3f1a94bed06f8a2c12d6a25610dd9385d9361da2e970f39f"
)
EXPECTED_NORMALIZED_PRODUCER_SHA256 = (
    "a6232ce8f6a0b8385d5977fb82fa4410602b1b2ef383c80835db43a16b42ec04"
)
EXPECTED_PROTOCOL_SHA256 = (
    "870272310940873a8fb939434979cbef3b42d1a7c68ebde127afe0dfe8453efb"
)
EXPECTED_DEPENDENCY_MANIFEST_SHA256 = (
    "9437b54d9a6379adaab9a7e2bd578caf55e3c879bf861f8a0150f124b577eb32"
)
EXPECTED_PARENT_FINAL_SHA256 = (
    "7063f393332f2640d599fa41928006f74342c3d681dd9b77a575e5e9e40716e5"
)
EXPECTED_PARENT_PROGRESS_SHA256 = (
    "010d34d51c21ec139fca3e93f51b91c7563f43a524a59b521dcec50de9c752ce"
)
EXPECTED_PARENT_ACTIVE_HASH = (
    "5b6fc545a74f397c4df4a023e2cccc83609633f9e1a738e717832c833ac05b73"
)
EXPECTED_PARENT_REVIEW_SHA256 = (
    "7677a64b38755bed2d553996f103d943ac7d43c7dae748f546e93ba5441b9703"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"

DIMENSION = 512
ACTIVE_PARAMETER_COUNT = 11_685_120
ACTIVE_TENSOR_COUNT = 31
SOURCE_WEIGHT_INDEX = 378
START_UPDATE = 17
MAX_ACCEPTED_UPDATES = 100
SUSTAINED_NEW_UPDATES = 10
TARGET_NORM = 0.04892722657548397
LEGACY_CANCELLATION_NORM_MIN = 0.5
ARMIJO_C1 = 1e-4
EPSILON = 1e-4
LOW_THRESHOLD = 0.1
OLD_BETA = 22.536727828943093
LINE_ALPHAS = tuple(2.0**-index for index in range(12))
REPEAT_MAX_ERROR = 1e-10
SUCCESS_A_MAX = 0.90
SUCCESS_BULK_FRACTION_MIN = 0.25
SUCCESS_TAIL_BULK_UPDATES_MIN = 4
DIRECTION_RADIUS_ATOL = 5e-9
PARENT_REPLAY_MAX_ERROR = 1e-9
PROPOSAL18_REPLAY_ATOL = 5e-9
PROPOSAL18_REPLAY_RTOL = 5e-7
BURG_GRADIENT_FORMULA_MAX_ERROR = 5e-6
RETAINED_MEMORY_RANGE_MAX = 64 * 1024 * 1024
BLOCK_PEAK_RANGE_MAX = 128 * 1024 * 1024

EXPECTED_MANIFEST_ARTIFACTS = {
    "arm_selection.csv",
    "decision.json",
    "executed_source_snapshot.py",
    "final_checkpoint.pt",
    "frozen_dependency_manifest_snapshot.json",
    "historical_transition_audit.csv",
    "intervention_origin.json",
    "intervention_preflight.json",
    "line_search.csv",
    "progress_checkpoint.pt",
    "proposal_diagnostics.csv",
    "protocol_snapshot.md",
    "relaxed_cancellation_direction_diagnostics.png",
    "relaxed_cancellation_spectra.png",
    "relaxed_cancellation_trajectory.png",
    "resolved_config.json",
    "state_metrics.csv",
    "state_spectra.csv",
}
UNMANIFESTED_PRODUCER_FILES = {
    "artifact_manifest.json",
    "FINALIZED.json",
    "run.log",
}
OPTIONAL_TEXT_COLUMNS = {"failure"}
GATE_NAMES = (
    "A_slope_negative",
    "A_armijo",
    "A_actual_decrease",
    "B_slope_negative",
    "B_armijo",
    "B_actual_decrease",
    "L_low_slope_negative",
    "L_low_armijo",
    "L_low_actual_decrease",
    "high_tail_nonincrease",
    "low90_decreases",
    "exact_a_closes",
    "spectrum_finite_nonnegative",
    "metrics_finite",
)
CONTINUATION_PROPOSAL_KEYS = {
    "phase",
    "target_update",
    "local_proposal",
    "selected_arm",
    "base_parameter_hash",
    "current_A",
    "current_B",
    "current_low_energy",
    "current_a_gt1",
    "gradient_a_norm",
    "gradient_x_norm",
    "gradient_cosine",
    "unit_common_source_norm",
    "fp64_common_source_norm",
    "theoretical_common_source_norm",
    "fp32_source_vs_theory_abs_error",
    "fp64_source_vs_theory_abs_error",
    "legacy_cancellation_gate_pass",
    "relaxed_nonzero_gate_pass",
    "fp64_shadow_valid",
    "common_amplification",
    "common_amplification_unbounded",
    "target_over_common_source",
    "fp32_fp64_direction_cosine",
    "fp32_fp64_direction_relative_error",
    "direction_norm",
    "slope_A",
    "slope_B",
    "slope_L_low",
    "theoretical_slope_A",
    "theoretical_slope_L_low",
    "slope_A_vs_theory_abs_error",
    "slope_L_low_vs_theory_abs_error",
    "gradient_metadata",
    "selected_alpha",
    "realized_path_length",
    "accepted",
    "candidate_A",
    "candidate_B",
    "candidate_low_energy",
    "candidate_a_gt1",
    "endpoint_parameter_hash",
    "transition_chain_sha256",
    "failure",
}
EXPECTED_VALIDITY_GATE_NAMES = {
    "parent_packet_valid",
    "parent_final_checkpoint_hash_matches",
    "parent_independent_review_valid",
    "intervention_preflight_complete",
    "state_rows_complete",
    "spectrum_rows_complete",
    "accepted_history_is_bijective",
    "disk_checkpoint_metadata_replays",
    "final_checkpoint_replays",
    "source_snapshot_matches",
    "live_source_unchanged",
    "dependencies_unchanged",
    "progress_checkpoint_fail_closed_audit",
    "state_and_spectra_finite",
    "proposal_and_line_values_finite",
}
EXPECTED_PROGRESS_AUDIT_GATE_NAMES = {
    "parent_state_prefix_exact",
    "parent_spectrum_prefix_exact",
    "parent_proposal_prefix_exact",
    "parent_line_prefix_exact",
    "selection_rows_parent_exact",
    "intervention_origin_parent_exact",
    "state_updates_exact",
    "state_count_exact",
    "accepted_proposals_exact",
    "no_parent_rejected_proposal_in_main_table",
    "state_phase_boundary_exact",
    "spectrum_phase_boundary_exact",
    "proposal_phase_boundary_exact",
    "line_phase_boundary_exact",
    "spectrum_update_keys_exact",
    "spectrum_ranks_exact",
    "spectrum_physical_order_exact",
    "terminal_structure_exact",
    "proposal_row_order_exact",
    "continuation_line_targets_known",
    "continuation_line_prefix_repeat_pass",
    "continuation_gate_map_recomputes",
    "continuation_repeat_values_valid",
    "continuation_pass_bits_recompute",
    "continuation_row_identity_exact",
    "continuation_diagnostics_recompute",
    "continuation_metric_schema_exact",
    "continuation_control_flow_recomputes",
    "continuation_gradient_metadata_valid",
    "continuation_parameter_hash_links_pass",
    "continuation_parent_low_links_pass",
    "continuation_proposal_current_links_pass",
    "continuation_endpoint_metric_links_pass",
    "continuation_spectrum_metrics_recompute",
    "transition_chain_recomputes",
    "progress_protocol_matches",
    "progress_source_matches",
    "progress_dependency_manifest_matches",
    "progress_parent_checkpoint_matches",
    "progress_parent_progress_matches",
    "progress_parent_active_hash_matches",
    "progress_selected_arm_is_low",
    "progress_update_range_valid",
    "progress_new_update_count_matches",
    "progress_tolerances_parent_exact",
    "progress_initial_metrics_parent_exact",
    "progress_active_names_exact",
    "progress_active_hash_recomputes",
    "progress_active_hash_matches_last_state",
    "progress_preflight_valid",
}
EXPECTED_CONFIG_KEYS = {
    "armijo_c1",
    "cache_mode",
    "continuation_cancellation_rule",
    "dependency_manifest_sha256",
    "dependency_matches",
    "device",
    "dtype",
    "legacy_cancellation_norm_min",
    "line_alphas",
    "max_new_accepted_updates",
    "max_total_accepted_updates",
    "normalized_source_sha256",
    "output_dir",
    "parent_checkpoint",
    "parent_checkpoint_sha256",
    "parent_progress_checkpoint_sha256",
    "protocol_id",
    "protocol_sha256",
    "resume",
    "seed",
    "selected_arm",
    "source_sha256",
    "start_update",
    "target_norm",
}
EXPECTED_MANIFEST_KEYS = {
    "protocol_id",
    "executed_source_sha256",
    "executed_normalized_source_sha256",
    "artifacts",
}
EXPECTED_FINALIZED_KEYS = {
    "protocol_id",
    "status",
    "valid",
    "scientific_success",
    "immediate_premature_cutoff",
    "sustained_continuation",
    "accepted_updates",
    "new_accepted_updates",
    "termination",
    "decision_sha256",
    "artifact_manifest_sha256",
}
EXPECTED_DECISION_KEYS = {
    "protocol_id",
    "valid",
    "scientific_success",
    "accepted_updates",
    "new_accepted_updates",
    "selected_arm",
    "termination",
    "intervention_preflight",
    "outcome",
    "initial_metrics",
    "intervention_start_metrics",
    "final_metrics",
    "historical_transition_summary",
    "progress_audit_gates",
    "tolerances",
    "validity_gates",
    "final_replay_errors",
    "final_spectrum_replay_error",
    "final_checkpoint_file_sha256",
    "elapsed_sec",
}


class ReviewError(RuntimeError):
    pass


class Review:
    def __init__(self) -> None:
        self.gates: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.errors: list[str] = []
        self.limitations: list[str] = []

    def phase(self, name: str, callback: Callable[[], Any]) -> Any | None:
        try:
            value = callback()
        except Exception as error:
            self.gates[f"{name}_completed"] = False
            self.errors.append(f"{name}: {type(error).__name__}: {error}")
            self.details[f"{name}_traceback"] = traceback.format_exc(limit=12)
            return None
        self.gates[f"{name}_completed"] = True
        return value


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
    normalized: list[str] = []
    prefix = "EXPECTED_NORMALIZED_SOURCE_SHA256 = "
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        normalized.append(f'{prefix}"<FROZEN>"\n' if line.startswith(prefix) else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} is not a JSON object")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, dict), f"{path.name} is not a checkpoint mapping")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    _require(path.read_text(encoding="utf-8").strip(), f"{path.name} is empty")
    return pd.read_csv(path)


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ReviewError(f"nonfinite numeric value: {value!r}")
    return result


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
        return None
    return _float(value)


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
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
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
            np.isclose(float(left), float(right), atol=atol, rtol=rtol, equal_nan=False)
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
    return _close(left, right, atol=atol, rtol=rtol)


def _cell_equivalent(
    expected: Any,
    observed: Any,
    *,
    empty_text_is_missing: bool = False,
) -> bool:
    def missing(value: Any) -> bool:
        return (
            value is None
            or (empty_text_is_missing and value == "")
            or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
        )

    if missing(expected) or missing(observed):
        return missing(expected) and missing(observed)
    if isinstance(expected, str):
        return str(observed) == expected
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
                empty_text_is_missing=column in OPTIONAL_TEXT_COLUMNS,
            ):
                return False
    return True


def _frame_numeric_finite(frame: pd.DataFrame, *, excluded: set[str]) -> bool:
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            return False
    return True


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    _require(data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not PNG")
    _require(data[12:16] == b"IHDR", f"{path.name} has no IHDR")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    _require(width >= 400 and height >= 300, f"{path.name} is unexpectedly small")
    return width, height


def _named_tensor_hash(mapping: Mapping[str, torch.Tensor]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name in sorted(mapping):
        tensor = mapping[name]
        _require(isinstance(tensor, torch.Tensor), f"{name} is not a tensor")
        _require(tensor.device.type == "cpu", f"{name} is not stored on CPU")
        _require(bool(torch.isfinite(tensor).all()), f"{name} is nonfinite")
        value = tensor.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
        count += value.numel()
    return digest.hexdigest(), count


def _chain_hash(
    previous: str,
    *,
    update: int,
    base_hash: str,
    endpoint_hash: str,
    alpha: float,
) -> str:
    payload = json.dumps(
        {
            "previous": previous,
            "update": update,
            "base_parameter_hash": base_hash,
            "endpoint_parameter_hash": endpoint_hash,
            "alpha": float(alpha),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _spectral_metrics(eigenvalues: np.ndarray) -> dict[str, float]:
    eig = np.asarray(eigenvalues, dtype=np.float64)
    _require(eig.shape == (DIMENSION,), f"expected {DIMENSION} eigenvalues")
    _require(np.isfinite(eig).all(), "spectrum contains nonfinite values")
    _require(bool(np.all(eig >= 0.0)), "clamped spectrum contains negatives")
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
    _require(
        {"accepted_update", "parameter_hash", "phase"}.issubset(states.columns),
        "state table columns missing",
    )
    _require(
        {"accepted_update", "rank", "m_eigenvalue", "a_contribution", "phase"}.issubset(
            spectra.columns
        ),
        "spectrum table columns missing",
    )
    _require(
        _frame_numeric_finite(
            states, excluded={"parameter_hash", "low_basis_hash", "phase"}
        ),
        "state table contains nonfinite values",
    )
    _require(
        _frame_numeric_finite(spectra, excluded={"phase"}),
        "spectrum table contains nonfinite values",
    )
    updates = [_int(value) for value in states["accepted_update"]]
    _require(updates == list(range(len(states))), "state update sequence is not exact")
    _require(
        len(spectra) == len(states) * DIMENSION,
        "spectrum row count does not equal states x dimension",
    )
    physical_grid = [
        (_int(row.accepted_update), _int(row.rank))
        for row in spectra.itertuples(index=False)
    ]
    expected_grid = [(update, rank) for update in updates for rank in range(DIMENSION)]
    _require(physical_grid == expected_grid, "spectrum physical row order mismatch")
    hashes = [_text(value) for value in states["parameter_hash"]]
    _require(all(_is_sha256(value) for value in hashes), "malformed state hash")
    _require(len(set(hashes)) == len(hashes), "state hashes are not unique")
    worst_metric_error = 0.0
    worst_contribution_error = 0.0
    integral = {"count_lt_1e_4", "count_lt_1e_2", "count_lt_0p1", "low_count"}
    per_update: list[dict[str, Any]] = []
    for position, update in enumerate(updates):
        state = states.iloc[position]
        block = spectra.iloc[update * DIMENSION : (update + 1) * DIMENSION]
        eig = block["m_eigenvalue"].to_numpy(dtype=np.float64)
        expected = _spectral_metrics(eig)
        observed_contribution = block["a_contribution"].to_numpy(dtype=np.float64)
        contribution_error = float(
            np.max(np.abs(observed_contribution - np.square(eig - 1.0)))
        )
        _require(
            contribution_error <= 1e-9,
            f"a_contribution mismatch at update {update}",
        )
        errors: dict[str, float] = {}
        for name, reference in expected.items():
            _require(name in states.columns, f"missing state metric {name}")
            observed = _float(state[name])
            error = abs(observed - reference)
            if name in integral:
                _require(observed == reference, f"{name} mismatch at update {update}")
            else:
                _require(
                    _close(observed, reference, atol=2e-9, rtol=2e-9),
                    f"{name} mismatch at update {update}",
                )
            errors[name] = error
        raw_min = _float(state["m_raw_eig_min"])
        _require(raw_min >= -1e-8, f"raw spectrum minimum invalid at {update}")
        _require(
            _close(max(raw_min, 0.0), eig[0], atol=2e-9, rtol=2e-9),
            f"raw/clamped spectrum minimum mismatch at {update}",
        )
        _require(
            _float(state["a_direct_abs_error"]) <= 1e-9
            and _float(state["a_trace_abs_error"]) <= 1e-9,
            f"exact-A closure failed at update {update}",
        )
        _require(
            _float(state["low_basis_orthogonality_max_abs"]) <= 1e-10
            and _float(state["low_basis_eigen_residual_relative"]) <= 1e-10
            and _float(state["helper_eigenvalue_max_abs_error"]) <= 1e-10,
            f"low eigenspace audit failed at update {update}",
        )
        _require(_is_sha256(_text(state["low_basis_hash"])), "bad low-basis hash")
        worst_metric_error = max(worst_metric_error, max(errors.values()))
        worst_contribution_error = max(worst_contribution_error, contribution_error)
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
        "line exact-A closure aliases failed",
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
            row["burg_neg_logdet_r_term"],
            -_float(row["logdet_r_per_dim"]),
            atol=1e-9,
            rtol=1e-9,
        ),
        "line Burg identity failed",
    )
    _require(
        _close(row["a_gt1"], row["a_high_gt_1_abs_per_dim"], atol=1e-12),
        "line high-tail aliases disagree",
    )
    _require(
        _int(row["count_lt_0p1"]) == _int(row["low_count"]),
        "line low-count aliases disagree",
    )
    for count, fraction in (
        ("count_lt_1e_4", "m_lt_1e_4_fraction"),
        ("count_lt_1e_2", "m_lt_0p01_fraction"),
        ("count_lt_0p1", "m_lt_0p1_fraction"),
    ):
        _require(
            _close(_int(row[count]) / DIMENSION, row[fraction], atol=1e-12, rtol=0),
            f"line {count}/{fraction} mismatch",
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
    gates["low90_decreases"] = True
    gates["exact_a_closes"] = (
        _float(candidate["a_direct_abs_error"]) <= 1e-9
        and _float(candidate["a_trace_abs_error"]) <= 1e-9
    )
    gates["spectrum_finite_nonnegative"] = _float(candidate["m_raw_eig_min"]) >= -1e-8
    try:
        gates["metrics_finite"] = all(
            key == "low_basis_hash" or math.isfinite(float(value))
            for key, value in candidate.items()
        )
    except (TypeError, ValueError, OverflowError):
        gates["metrics_finite"] = False
    return gates


def _validate_gate_map(row: Mapping[str, Any], expected: Mapping[str, bool]) -> None:
    observed_keys = {key for key in row if key.startswith("gate_")}
    expected_keys = {f"gate_{name}" for name in GATE_NAMES}
    _require(observed_keys == expected_keys, "line gate schema mismatch")
    for name, value in expected.items():
        _require(
            _bool(row[f"gate_{name}"]) is value,
            f"gate_{name} does not recompute",
        )


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
    if _float(row["repeat_parameter_hash_unchanged"]) != 1.0:
        return False
    if any(_float(row[key]) != 0.0 for key in exact):
        return False
    if any(_float(row[key]) > limit for key, limit in limits.items()):
        return False
    remaining = {
        key
        for key in row
        if key.startswith("repeat_")
        and key not in set(limits)
        and key not in exact
        and key != "repeat_parameter_hash_unchanged"
    }
    return all(_float(row[key]) <= REPEAT_MAX_ERROR for key in remaining)


def _validate_repeat_payload(
    row: Mapping[str, Any],
    tolerances: Mapping[str, float],
    expected_keys: set[str],
) -> bool:
    observed = {key for key in row if key.startswith("repeat_")}
    _require(observed == expected_keys, "repeat payload schema mismatch")
    for key in observed:
        try:
            value = float(row[key])
        except (TypeError, ValueError, OverflowError) as error:
            raise ReviewError(f"invalid repeat error: {key}") from error
        _require(
            math.isfinite(value) and value >= 0.0,
            f"nonfinite or negative repeat error: {key}",
        )
    _require(
        _float(row["repeat_parameter_hash_unchanged"]) == 1.0,
        "repeat parameter hash changed",
    )
    return _repeat_passes(row, tolerances)


def _validate_gradient_metadata(metadata: Mapping[str, Any]) -> None:
    _require(_int(metadata["basis_count"]) == DIMENSION, "gradient basis mismatch")
    _require(_int(metadata["block_count"]) == 8, "gradient block count mismatch")
    _require(_int(metadata["block_size"]) == 64, "gradient block size mismatch")
    retained = [_int(value) for value in metadata["retained_memory_by_block_bytes"]]
    peaks = [_int(value) for value in metadata["peak_memory_by_block_bytes"]]
    _require(len(retained) == 8 and len(peaks) == 8, "gradient trace length mismatch")
    _require(
        all(value >= 0 for value in retained)
        and all(value >= 0 for value in peaks)
        and all(peak >= live for peak, live in zip(peaks, retained, strict=True)),
        "invalid gradient memory trace",
    )
    retained_range = max(retained) - min(retained)
    peak_range = max(peaks) - min(peaks)
    _require(
        _int(metadata["retained_memory_range_bytes"]) == retained_range
        and _int(metadata["block_peak_range_bytes"]) == peak_range,
        "gradient memory range mismatch",
    )
    memory_gate = (
        retained_range <= RETAINED_MEMORY_RANGE_MAX
        and peak_range <= BLOCK_PEAK_RANGE_MAX
    )
    _require(_bool(metadata["memory_gate_pass"]) is memory_gate, "memory gate mismatch")
    edge = min(2, len(retained))
    _require(
        _close(
            metadata["memory_early_median_bytes"],
            float(np.median(retained[:edge])),
            atol=0,
            rtol=0,
        )
        and _close(
            metadata["memory_late_median_bytes"],
            float(np.median(retained[-edge:])),
            atol=0,
            rtol=0,
        ),
        "gradient memory median mismatch",
    )
    _require(_float(metadata["elapsed_sec"]) >= 0.0, "gradient elapsed invalid")
    valid = memory_gate
    for name in ("A", "B", "low"):
        unused = _int(metadata[f"{name}_unused_parameter_tensors_all_blocks"])
        accumulator = _bool(metadata[f"{name}_all_accumulators_float64"])
        _require(unused == 0 and accumulator, f"{name} gradient metadata invalid")
        valid = valid and unused == 0 and accumulator
    for name in ("k_a_norm", "k_b_norm", "k_low_norm"):
        positive = _float(metadata[name]) > 0.0
        _require(positive, f"{name} is not positive")
        valid = valid and positive
    burg_error = _float(metadata["burg_gradient_formula_max_abs_error"])
    _require(
        burg_error <= BURG_GRADIENT_FORMULA_MAX_ERROR,
        "Burg gradient formula error too large",
    )
    valid = valid and burg_error <= BURG_GRADIENT_FORMULA_MAX_ERROR
    _require(
        _bool(metadata["gradient_payload_valid"]) is valid,
        "gradient payload validity flag mismatch",
    )


def _validate_continuation_direction(
    row: Mapping[str, Any], *, expected_metadata_keys: set[str] | None = None
) -> dict[str, Any]:
    norm_a = _float(row["gradient_a_norm"])
    norm_low = _float(row["gradient_x_norm"])
    cosine = _float(row["gradient_cosine"])
    source = _float(row["unit_common_source_norm"])
    fp64_source = _float(row["fp64_common_source_norm"])
    _require(norm_a > 0.0 and norm_low > 0.0, "gradient norm is not positive")
    _require(-1.0 - 1e-9 <= cosine <= 1.0 + 1e-9, "gradient cosine invalid")
    _require(source >= 0.0 and fp64_source >= 0.0, "common source norm is negative")
    theoretical = math.sqrt(max(0.0, 2.0 + 2.0 * cosine))
    relaxed = source > 0.0
    shadow_valid = fp64_source > 0.0
    _require(
        _bool(row["legacy_cancellation_gate_pass"])
        is (source >= LEGACY_CANCELLATION_NORM_MIN),
        "legacy cancellation bit mismatch",
    )
    _require(
        _bool(row["relaxed_nonzero_gate_pass"]) is relaxed,
        "relaxed cancellation bit mismatch",
    )
    _require(
        _bool(row["fp64_shadow_valid"]) is shadow_valid,
        "FP64 shadow bit mismatch",
    )
    _require(
        _close(row["theoretical_common_source_norm"], theoretical, atol=2e-12),
        "theoretical source norm mismatch",
    )
    _require(
        _close(
            row["fp32_source_vs_theory_abs_error"],
            abs(source - theoretical),
            atol=2e-12,
        )
        and _close(
            row["fp64_source_vs_theory_abs_error"],
            abs(fp64_source - theoretical),
            atol=2e-12,
        ),
        "source error diagnostic mismatch",
    )
    observed_amplification = _optional_float(row["common_amplification"])
    observed_target_ratio = _optional_float(row["target_over_common_source"])
    if relaxed:
        _require(
            observed_amplification is not None
            and _close(observed_amplification, 1.0 / source, atol=1e-12),
            "normalization amplification mismatch",
        )
        _require(
            observed_target_ratio is not None
            and _close(observed_target_ratio, TARGET_NORM / source, atol=1e-12),
            "target/source ratio mismatch",
        )
        _require(
            not _bool(row["common_amplification_unbounded"]),
            "finite source marked as unbounded",
        )
        _require(
            _close(
                row["direction_norm"], TARGET_NORM, atol=DIRECTION_RADIUS_ATOL, rtol=0
            ),
            "direction radius mismatch",
        )
        theoretical_a = -TARGET_NORM * norm_a * theoretical / 2.0
        theoretical_low = -TARGET_NORM * norm_low * theoretical / 2.0
        _require(
            _close(row["theoretical_slope_A"], theoretical_a, atol=2e-12)
            and _close(row["theoretical_slope_L_low"], theoretical_low, atol=2e-12),
            "theoretical directional slope mismatch",
        )
        _require(
            _close(
                row["slope_A_vs_theory_abs_error"],
                abs(_float(row["slope_A"]) - theoretical_a),
                atol=2e-12,
            )
            and _close(
                row["slope_L_low_vs_theory_abs_error"],
                abs(_float(row["slope_L_low"]) - theoretical_low),
                atol=2e-12,
            ),
            "directional slope error diagnostic mismatch",
        )
    else:
        _require(source == 0.0, "ineligible finite source is not exact zero")
        _require(
            observed_amplification is None and observed_target_ratio is None,
            "zero source serialized with finite amplification",
        )
        _require(
            _bool(row["common_amplification_unbounded"]),
            "zero source not marked as unbounded",
        )
        _require(_float(row["direction_norm"]) == 0.0, "zero source has direction")
        for name in ("slope_A", "slope_B", "slope_L_low"):
            _require(_float(row[name]) == 0.0, "zero source has nonzero slope")
    if shadow_valid and relaxed:
        shadow_cosine = _float(row["fp32_fp64_direction_cosine"])
        shadow_error = _float(row["fp32_fp64_direction_relative_error"])
        _require(-1.0 - 1e-9 <= shadow_cosine <= 1.0 + 1e-9, "shadow cosine invalid")
        _require(shadow_error >= 0.0, "shadow relative error negative")
    else:
        _require(
            _float(row["fp32_fp64_direction_cosine"]) == 0.0
            and _float(row["fp32_fp64_direction_relative_error"]) == 0.0,
            "invalid shadow has nonzero diagnostics",
        )
    metadata = json.loads(_text(row["gradient_metadata"]))
    _require(isinstance(metadata, Mapping), "gradient metadata is not a mapping")
    if expected_metadata_keys is not None:
        _require(
            set(metadata) == expected_metadata_keys,
            "gradient metadata schema mismatch",
        )
    _validate_gradient_metadata(metadata)
    selected_alpha = _float(row["selected_alpha"])
    _require(
        _close(
            row["realized_path_length"],
            selected_alpha * TARGET_NORM,
            atol=2e-12,
        ),
        "proposal realized path mismatch",
    )
    return {
        "relaxed": relaxed,
        "shadow_valid": shadow_valid,
        "source": source,
        "theoretical_source": theoretical,
    }


def _validate_continuation_row_identity(
    row: Mapping[str, Any], *, target: int, kind: str
) -> None:
    _require(row.get("phase") == "relaxed_continuation", f"{kind} phase mismatch")
    _require(row.get("selected_arm") == "low", f"{kind} selected arm mismatch")
    _require(_int(row["target_update"]) == target, f"{kind} target mismatch")
    _require(
        _int(row["local_proposal"]) == target - START_UPDATE,
        f"{kind} local proposal mismatch",
    )


def _validate_exact_manifest(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    expected_artifacts: set[str] | None = None,
    unmanifested: set[str] | None = None,
) -> dict[str, str]:
    expected = (
        EXPECTED_MANIFEST_ARTIFACTS
        if expected_artifacts is None
        else expected_artifacts
    )
    outside = UNMANIFESTED_PRODUCER_FILES if unmanifested is None else unmanifested
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, Mapping), "artifact manifest payload missing")
    _require(set(artifacts) == expected, "artifact manifest name set mismatch")
    entries = list(output.iterdir())
    _require(
        all(path.is_file() and not path.is_symlink() for path in entries),
        "published output contains a directory, symlink, or special file",
    )
    actual = {path.name for path in entries}
    _require(actual == expected | outside, "published output file set mismatch")
    result: dict[str, str] = {}
    for name, digest in artifacts.items():
        _require(_is_sha256(digest), f"malformed artifact digest: {name}")
        observed = _sha256_file(output / name)
        _require(observed == digest, f"artifact hash mismatch: {name}")
        result[name] = observed
    return result


def _validate_static_and_publication(output: Path) -> dict[str, Any]:
    _require(output.is_dir(), f"production output does not exist: {output}")
    _require(not (output / "INCOMPLETE").exists(), "production is still incomplete")
    _require(
        _sha256_file(PRODUCER_SOURCE) == EXPECTED_PRODUCER_SHA256,
        "live producer raw hash mismatch",
    )
    _require(
        _normalized_producer_sha256(PRODUCER_SOURCE)
        == EXPECTED_NORMALIZED_PRODUCER_SHA256,
        "live producer normalized hash mismatch",
    )
    _require(
        _sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256,
        "protocol hash mismatch",
    )
    _require(
        _sha256_file(DEPENDENCY_MANIFEST) == EXPECTED_DEPENDENCY_MANIFEST_SHA256,
        "dependency manifest hash mismatch",
    )
    dependency_payload = _load_json(DEPENDENCY_MANIFEST)
    for relative, expected in dependency_payload.items():
        path = ROOT / relative
        _require(path.is_file(), f"frozen dependency missing: {relative}")
        _require(
            _sha256_file(path) == expected, f"frozen dependency changed: {relative}"
        )
    manifest = _load_json(output / "artifact_manifest.json")
    _require(
        set(manifest) == EXPECTED_MANIFEST_KEYS, "artifact manifest schema mismatch"
    )
    hashes = _validate_exact_manifest(output, manifest)
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "manifest protocol mismatch")
    source_snapshot = output / "executed_source_snapshot.py"
    _require(
        _sha256_file(source_snapshot) == EXPECTED_PRODUCER_SHA256,
        "executed producer raw hash mismatch",
    )
    _require(
        _normalized_producer_sha256(source_snapshot)
        == EXPECTED_NORMALIZED_PRODUCER_SHA256,
        "executed producer normalized hash mismatch",
    )
    _require(
        manifest.get("executed_source_sha256") == EXPECTED_PRODUCER_SHA256,
        "manifest executed raw source mismatch",
    )
    _require(
        manifest.get("executed_normalized_source_sha256")
        == EXPECTED_NORMALIZED_PRODUCER_SHA256,
        "manifest executed normalized source mismatch",
    )
    _require(
        _sha256_file(output / "protocol_snapshot.md") == EXPECTED_PROTOCOL_SHA256,
        "protocol snapshot mismatch",
    )
    _require(
        _sha256_file(output / "frozen_dependency_manifest_snapshot.json")
        == EXPECTED_DEPENDENCY_MANIFEST_SHA256,
        "dependency manifest snapshot mismatch",
    )
    config = _load_json(output / "resolved_config.json")
    _require(set(config) == EXPECTED_CONFIG_KEYS, "resolved config schema mismatch")
    expected_config = {
        "protocol_id": PROTOCOL_ID,
        "parent_checkpoint_sha256": EXPECTED_PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": EXPECTED_PARENT_PROGRESS_SHA256,
        "start_update": START_UPDATE,
        "max_total_accepted_updates": MAX_ACCEPTED_UPDATES,
        "max_new_accepted_updates": MAX_ACCEPTED_UPDATES - START_UPDATE,
        "selected_arm": "low",
        "legacy_cancellation_norm_min": LEGACY_CANCELLATION_NORM_MIN,
        "continuation_cancellation_rule": "finite and strictly nonzero only",
        "target_norm": TARGET_NORM,
        "armijo_c1": ARMIJO_C1,
        "line_alphas": list(LINE_ALPHAS),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_sha256": EXPECTED_PRODUCER_SHA256,
        "normalized_source_sha256": EXPECTED_NORMALIZED_PRODUCER_SHA256,
        "dependency_manifest_sha256": EXPECTED_DEPENDENCY_MANIFEST_SHA256,
    }
    for name, expected in expected_config.items():
        _require(
            _equivalent(config.get(name), expected, atol=0, rtol=0),
            f"resolved config mismatch: {name}",
        )
    _require(
        set(config.get("dependency_matches", {})) == set(dependency_payload)
        and all(config["dependency_matches"].values()),
        "resolved dependency gate mismatch",
    )
    _require(
        _text(config.get("device")) == "cuda:0"
        and config.get("dtype")
        == "float32 model/HVP and executed direction; FP64 gradients/diagnostics"
        and config.get("seed") == "none; exact complete-basis gradients"
        and config.get("cache_mode")
        == "accepted h2048 run + immutable I6 final checkpoint"
        and config.get("parent_checkpoint") == str(PARENT_FINAL)
        and config.get("output_dir") == str(DEFAULT_OUTPUT)
        and isinstance(config.get("resume"), bool),
        "resolved execution metadata mismatch",
    )
    finalized = _load_json(output / "FINALIZED.json")
    decision = _load_json(output / "decision.json")
    _require(set(finalized) == EXPECTED_FINALIZED_KEYS, "FINALIZED schema mismatch")
    _require(set(decision) == EXPECTED_DECISION_KEYS, "decision schema mismatch")
    _require(finalized.get("protocol_id") == PROTOCOL_ID, "FINALIZED protocol mismatch")
    _require(
        finalized.get("status") == "complete_awaiting_independent_review",
        "FINALIZED status mismatch",
    )
    _require(finalized.get("valid") is True, "producer did not finalize valid")
    _require(decision.get("protocol_id") == PROTOCOL_ID, "decision protocol mismatch")
    _require(decision.get("valid") is True, "producer decision is invalid")
    _require(
        _float(decision.get("elapsed_sec")) >= 0.0, "decision elapsed time invalid"
    )
    _require(
        finalized.get("decision_sha256") == _sha256_file(output / "decision.json"),
        "FINALIZED decision hash mismatch",
    )
    _require(
        finalized.get("artifact_manifest_sha256")
        == _sha256_file(output / "artifact_manifest.json"),
        "FINALIZED manifest hash mismatch",
    )
    _require((output / "run.log").stat().st_size > 0, "run log is empty")
    pngs = {
        name: _png_dimensions(output / name)
        for name in EXPECTED_MANIFEST_ARTIFACTS
        if name.endswith(".png")
    }
    return {
        "manifest": manifest,
        "artifact_hashes": hashes,
        "config": config,
        "finalized": finalized,
        "decision": decision,
        "png_dimensions": pngs,
        "run_log_sha256": _sha256_file(output / "run.log"),
    }


def _seeded_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_rows": [{**dict(row), "phase": "i6"} for row in parent["state_rows"]],
        "spectrum_rows": [
            {**dict(row), "phase": "i6"} for row in parent["spectrum_rows"]
        ],
        "proposal_rows": [
            {**dict(row), "phase": "i6"}
            for row in parent["proposal_rows"]
            if _int(row["accepted"]) == 1
        ],
        "line_rows": [{**dict(row), "phase": "i6"} for row in parent["line_rows"]],
        "selection_rows": [
            {**dict(row), "phase": "i6"} for row in parent["selection_rows"]
        ],
        "intervention_origin": dict(parent["proposal_rows"][-1]),
    }


def _validate_preflight(
    progress: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    preflight = progress["intervention_preflight"]
    origin = progress["intervention_origin"]
    _require(origin == parent["proposal_rows"][-1], "intervention origin mismatch")
    continuation = [
        row
        for row in progress["proposal_rows"]
        if _int(row["target_update"]) > START_UPDATE
    ]
    _require(continuation, "continuation has no proposal")
    first = continuation[0]
    _require(
        _int(first["target_update"]) == START_UPDATE + 1,
        "first continuation target mismatch",
    )
    replay_names = (
        "gradient_a_norm",
        "gradient_x_norm",
        "gradient_cosine",
        "unit_common_source_norm",
    )
    errors = {
        name: abs(_float(first[name]) - _float(origin[name])) for name in replay_names
    }
    stored_errors = preflight.get("proposal18_replay_errors")
    _require(
        isinstance(stored_errors, Mapping)
        and set(stored_errors) == set(errors)
        and all(
            _close(stored_errors[name], errors[name], atol=0, rtol=0) for name in errors
        ),
        "proposal-18 replay error payload mismatch",
    )
    replay_pass = all(
        math.isclose(
            _float(first[name]),
            _float(origin[name]),
            rel_tol=PROPOSAL18_REPLAY_RTOL,
            abs_tol=PROPOSAL18_REPLAY_ATOL,
        )
        for name in replay_names
    )
    _require(
        preflight.get("proposal18_replay_pass") is replay_pass and replay_pass,
        "proposal-18 replay failed",
    )
    parent_state = parent["state_rows"][-1]
    expected_error_names = {
        name
        for name, value in parent_state.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and name not in {"accepted_update", "parent_frozen_low_energy", "hessian_sec"}
    }
    metric_errors = preflight.get("parent_state_metric_errors")
    _require(isinstance(metric_errors, Mapping), "parent metric errors missing")
    _require(
        set(metric_errors) == expected_error_names,
        "parent metric error schema mismatch",
    )
    values = [_float(value) for value in metric_errors.values()]
    _require(
        values and all(value >= 0.0 for value in values), "invalid parent metric errors"
    )
    maximum = max(values)
    _require(maximum <= PARENT_REPLAY_MAX_ERROR, "parent metric replay too large")
    _require(
        _close(preflight["parent_state_metric_max_abs_error"], maximum, atol=0, rtol=0),
        "parent metric error maximum mismatch",
    )
    spectrum_error = _float(preflight["parent_spectrum_max_abs_error"])
    _require(
        0.0 <= spectrum_error <= PARENT_REPLAY_MAX_ERROR,
        "parent spectrum replay failed",
    )
    _require(
        preflight.get("parent_parameter_hash_matches") is True
        and first.get("base_parameter_hash") == EXPECTED_PARENT_ACTIVE_HASH
        and parent_state.get("parameter_hash") == EXPECTED_PARENT_ACTIVE_HASH,
        "parent parameter lineage mismatch",
    )
    _require(
        preflight.get("parent_low_basis_hash_matches") is True,
        "parent low basis replay failed",
    )
    _require(
        preflight.get("parent_independent_review_valid") is True,
        "parent review preflight failed",
    )
    old_rejects = (
        _float(first["unit_common_source_norm"]) < LEGACY_CANCELLATION_NORM_MIN
    )
    relaxed_accepts = _float(first["unit_common_source_norm"]) > 0.0
    _require(
        preflight.get("old_gate_rejects") is old_rejects and old_rejects,
        "old cutoff discriminator failed",
    )
    _require(
        preflight.get("relaxed_gate_accepts") is relaxed_accepts and relaxed_accepts,
        "relaxed cutoff discriminator failed",
    )
    return {
        "proposal18_replay_errors": errors,
        "parent_state_metric_max_abs_error": maximum,
        "parent_spectrum_max_abs_error": spectrum_error,
    }


def _validate_checkpoint_and_tables(output: Path) -> dict[str, Any]:
    _require(
        _sha256_file(PARENT_FINAL) == EXPECTED_PARENT_FINAL_SHA256,
        "parent final changed",
    )
    _require(
        _sha256_file(PARENT_PROGRESS) == EXPECTED_PARENT_PROGRESS_SHA256,
        "parent progress changed",
    )
    _require(
        _sha256_file(PARENT_REVIEW) == EXPECTED_PARENT_REVIEW_SHA256,
        "parent review changed",
    )
    parent_review = _load_json(PARENT_REVIEW)
    _require(
        parent_review.get("valid") is True
        and parent_review.get("scientific_success") is False
        and parent_review.get("failed_gates") == []
        and parent_review.get("errors") == [],
        "parent independent review is not accepted",
    )
    parent = _load_checkpoint(PARENT_PROGRESS)
    progress = _load_checkpoint(output / "progress_checkpoint.pt")
    final = _load_checkpoint(output / "final_checkpoint.pt")
    accepted = _int(progress.get("accepted_updates", -1))
    expected_progress_keys = {
        "protocol_id",
        "normalized_source_sha256",
        "dependency_manifest_sha256",
        "parent_checkpoint_sha256",
        "parent_progress_checkpoint_sha256",
        "parent_active_parameter_hash",
        "accepted_updates",
        "new_accepted_updates",
        "selected_arm",
        "terminal",
        "termination",
        "transition_chain_sha256",
        "active_parameter_hash",
        "active_model_state",
        "state_rows",
        "spectrum_rows",
        "proposal_rows",
        "line_rows",
        "selection_rows",
        "intervention_origin",
        "initial_metrics",
        "tolerances",
        "intervention_preflight",
    }
    _require(set(progress) == expected_progress_keys, "progress schema mismatch")
    expected_final_keys = {
        "protocol_id",
        "parent_checkpoint_sha256",
        "parent_progress_checkpoint_sha256",
        "selected_arm",
        "accepted_updates",
        "new_accepted_updates",
        "termination",
        "transition_chain_sha256",
        "active_parameter_hash",
        "active_model_state",
        "source_weight_index",
        "z_sha256",
        "stored_state_metrics",
    }
    _require(set(final) == expected_final_keys, "final checkpoint schema mismatch")
    common = {
        "protocol_id": PROTOCOL_ID,
        "parent_checkpoint_sha256": EXPECTED_PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": EXPECTED_PARENT_PROGRESS_SHA256,
        "selected_arm": "low",
        "accepted_updates": accepted,
        "new_accepted_updates": accepted - START_UPDATE,
        "termination": progress["termination"],
        "transition_chain_sha256": progress["transition_chain_sha256"],
    }
    for name, expected in common.items():
        _require(progress.get(name) == expected, f"progress metadata mismatch: {name}")
        _require(final.get(name) == expected, f"final metadata mismatch: {name}")
    _require(
        START_UPDATE <= accepted <= MAX_ACCEPTED_UPDATES,
        "accepted update count invalid",
    )
    _require(progress.get("terminal") is True, "final progress is not terminal")
    _require(
        progress.get("normalized_source_sha256") == EXPECTED_NORMALIZED_PRODUCER_SHA256
        and progress.get("dependency_manifest_sha256")
        == EXPECTED_DEPENDENCY_MANIFEST_SHA256
        and progress.get("parent_active_parameter_hash") == EXPECTED_PARENT_ACTIVE_HASH,
        "progress frozen lineage mismatch",
    )
    _require(
        final.get("source_weight_index") == SOURCE_WEIGHT_INDEX
        and final.get("z_sha256") == EXPECTED_Z_SHA256,
        "final state identity mismatch",
    )
    progress_state = progress.get("active_model_state")
    final_state = final.get("active_model_state")
    _require(
        isinstance(progress_state, Mapping) and isinstance(final_state, Mapping),
        "active state missing",
    )
    _require(set(progress_state) == set(final_state), "checkpoint tensor key mismatch")
    parent_active_state = parent.get("active_model_state")
    _require(
        isinstance(parent_active_state, Mapping)
        and set(progress_state) == set(parent_active_state),
        "checkpoint tensor names differ from frozen parent",
    )
    _require(len(progress_state) == ACTIVE_TENSOR_COUNT, "active tensor count mismatch")
    for name in progress_state:
        _require(
            progress_state[name].shape == parent_active_state[name].shape
            and progress_state[name].dtype == parent_active_state[name].dtype,
            f"active tensor metadata differs from parent: {name}",
        )
        _require(
            torch.equal(progress_state[name], final_state[name]),
            f"final/progress tensor mismatch: {name}",
        )
    progress_hash, progress_count = _named_tensor_hash(progress_state)
    final_hash, final_count = _named_tensor_hash(final_state)
    _require(
        progress_count == final_count == ACTIVE_PARAMETER_COUNT,
        "active parameter count mismatch",
    )
    _require(
        progress_hash
        == final_hash
        == progress["active_parameter_hash"]
        == final["active_parameter_hash"],
        "active parameter hash mismatch",
    )
    states = _read_csv(output / "state_metrics.csv")
    spectra = _read_csv(output / "state_spectra.csv")
    proposals = _read_csv(output / "proposal_diagnostics.csv")
    lines = _read_csv(output / "line_search.csv")
    selections = _read_csv(output / "arm_selection.csv")
    for records, frame, label in (
        (progress["state_rows"], states, "state"),
        (progress["spectrum_rows"], spectra, "spectrum"),
        (progress["proposal_rows"], proposals, "proposal"),
        (progress["line_rows"], lines, "line"),
        (progress["selection_rows"], selections, "selection"),
    ):
        _require(
            _checkpoint_rows_match_csv(records, frame),
            f"{label} CSV/checkpoint mismatch",
        )
    _require(
        _equivalent(
            _load_json(output / "intervention_origin.json"),
            progress["intervention_origin"],
            atol=0,
            rtol=0,
        ),
        "intervention origin JSON mismatch",
    )
    _require(
        _equivalent(
            _load_json(output / "intervention_preflight.json"),
            progress["intervention_preflight"],
            atol=0,
            rtol=0,
        ),
        "intervention preflight JSON mismatch",
    )
    seeded = _seeded_parent(parent)
    _require(
        progress["state_rows"][: START_UPDATE + 1] == seeded["state_rows"],
        "parent state prefix mismatch",
    )
    _require(
        progress["spectrum_rows"][: (START_UPDATE + 1) * DIMENSION]
        == seeded["spectrum_rows"],
        "parent spectrum prefix mismatch",
    )
    _require(
        progress["proposal_rows"][:START_UPDATE] == seeded["proposal_rows"],
        "parent proposal prefix mismatch",
    )
    _require(
        progress["line_rows"][: len(seeded["line_rows"])] == seeded["line_rows"],
        "parent line prefix mismatch",
    )
    _require(
        progress["selection_rows"] == seeded["selection_rows"],
        "parent selection rows mismatch",
    )
    _require(
        progress["intervention_origin"] == seeded["intervention_origin"],
        "parent terminal origin mismatch",
    )
    _require(
        progress["initial_metrics"] == parent["initial_metrics"],
        "initial metric lineage mismatch",
    )
    _require(
        progress["tolerances"] == parent["tolerances"], "tolerance lineage mismatch"
    )
    _require(
        final["stored_state_metrics"] == progress["state_rows"][-1],
        "stored final metrics mismatch",
    )
    _require(
        progress_hash == progress["state_rows"][-1]["parameter_hash"],
        "active hash/last state mismatch",
    )
    preflight = _validate_preflight(progress, parent)
    return {
        "parent": parent,
        "progress": progress,
        "final": final,
        "states": states,
        "spectra": spectra,
        "proposals": proposals,
        "lines": lines,
        "selections": selections,
        "preflight": preflight,
    }


def _metric_keys_from_parent(parent: Mapping[str, Any]) -> set[str]:
    excluded = {
        "phase",
        "target_update",
        "selected_arm",
        "alpha",
        "endpoint_parameter_hash",
        "passes",
    }
    return {
        key
        for key in parent["line_rows"][-1]
        if key not in excluded
        and not key.startswith("gate_")
        and not key.startswith("repeat_")
    }


def _selected_line_matches_state(
    line: Mapping[str, Any], state: Mapping[str, Any], metric_keys: set[str]
) -> bool:
    for key in metric_keys:
        state_key = key
        if key == "frozen_low_energy":
            state_key = "parent_frozen_low_energy"
        if not _equivalent(line[key], state[state_key], atol=1e-9, rtol=2e-9):
            return False
    return _equivalent(
        line["canonical_low_energy"], state["frozen_low_energy"], atol=1e-9, rtol=2e-9
    ) and _equivalent(
        line["canonical_low_energy"],
        state["canonical_low_energy"],
        atol=1e-9,
        rtol=2e-9,
    )


def _validate_history(
    progress: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    states = list(progress["state_rows"])
    spectra = list(progress["spectrum_rows"])
    proposals = list(progress["proposal_rows"])
    lines = list(progress["line_rows"])
    accepted_updates = _int(progress["accepted_updates"])
    tolerances = {key: _float(value) for key, value in progress["tolerances"].items()}
    state_updates = [_int(row["accepted_update"]) for row in states]
    _require(
        state_updates == list(range(accepted_updates + 1)), "state sequence mismatch"
    )
    _require(
        len(spectra) == (accepted_updates + 1) * DIMENSION, "spectrum count mismatch"
    )
    accepted = [row for row in proposals if _int(row["accepted"]) == 1]
    rejected = [row for row in proposals if _int(row["accepted"]) == 0]
    _require(
        [_int(row["target_update"]) for row in accepted]
        == list(range(1, accepted_updates + 1)),
        "accepted proposal sequence mismatch",
    )
    expected_targets = list(range(1, accepted_updates + 1))
    if accepted_updates < MAX_ACCEPTED_UPDATES:
        expected_targets.append(accepted_updates + 1)
    _require(
        [_int(row["target_update"]) for row in proposals] == expected_targets,
        "proposal physical order mismatch",
    )
    if accepted_updates == MAX_ACCEPTED_UPDATES:
        _require(
            not rejected and progress["termination"] == "max_updates_reached",
            "max-update termination mismatch",
        )
    else:
        _require(len(rejected) == 1, "terminal proposal count mismatch")
        _require(
            _text(rejected[0]["failure"]) == progress["termination"],
            "terminal failure mismatch",
        )
        _require(
            progress["termination"]
            in {
                "exact_cancellation",
                "nonnegative_joint_slope",
                "backtracking_exhausted",
            },
            "unknown termination",
        )
    state_by_update = {_int(row["accepted_update"]): row for row in states}
    lines_by_target: dict[int, list[Mapping[str, Any]]] = {}
    for line in lines:
        lines_by_target.setdefault(_int(line["target_update"]), []).append(line)
    _require(
        set(lines_by_target).issubset(set(range(1, accepted_updates + 2))),
        "orphan line target",
    )
    parent_line_count = len(parent["line_rows"])
    continuation_lines = lines[parent_line_count:]
    _require(
        [_int(row["target_update"]) for row in continuation_lines]
        == sorted(_int(row["target_update"]) for row in continuation_lines),
        "continuation line target order mismatch",
    )
    parent_metric_keys = _metric_keys_from_parent(parent)
    expected_repeat_keys = {
        key for key in parent["line_rows"][-1] if key.startswith("repeat_")
    }
    expected_gate_keys = {f"gate_{name}" for name in GATE_NAMES}
    parent_gradient_metadata = json.loads(
        _text(parent["proposal_rows"][-1]["gradient_metadata"])
    )
    _require(
        isinstance(parent_gradient_metadata, Mapping),
        "parent gradient metadata is malformed",
    )
    expected_gradient_metadata_keys = set(parent_gradient_metadata)
    parent_line_schema = set(parent["line_rows"][-1]) | {"phase"}
    continuation_line_schema = parent_line_schema | {
        "base_parameter_hash",
        "local_proposal",
        "realized_path_length",
    }
    parent_state_schema = set(parent["state_rows"][-1]) | {"phase"}
    parent_spectrum_schema = set(parent["spectrum_rows"][-1]) | {"phase"}
    for row in states[START_UPDATE + 1 :]:
        _require(set(row) == parent_state_schema, "continuation state schema mismatch")
        _require(
            row.get("phase") == "relaxed_continuation",
            "continuation state phase mismatch",
        )
    for row in spectra[(START_UPDATE + 1) * DIMENSION :]:
        _require(
            set(row) == parent_spectrum_schema,
            "continuation spectrum schema mismatch",
        )
        _require(
            row.get("phase") == "relaxed_continuation",
            "continuation spectrum phase mismatch",
        )
    audit_rows: list[dict[str, Any]] = []
    chain = EXPECTED_PARENT_FINAL_SHA256
    for proposal in proposals:
        target = _int(proposal["target_update"])
        parent_state = state_by_update[target - 1]
        target_lines = lines_by_target.get(target, [])
        is_continuation = target > START_UPDATE
        if is_continuation:
            _require(
                set(proposal) == CONTINUATION_PROPOSAL_KEYS,
                "continuation proposal schema mismatch",
            )
            _validate_continuation_row_identity(
                proposal, target=target, kind="continuation proposal"
            )
            _require(
                proposal.get("base_parameter_hash") == parent_state["parameter_hash"],
                "proposal base hash mismatch",
            )
            _validate_continuation_direction(
                proposal,
                expected_metadata_keys=expected_gradient_metadata_keys,
            )
            for line in target_lines:
                _require(
                    set(line) == continuation_line_schema,
                    "continuation line schema mismatch",
                )
                _validate_continuation_row_identity(
                    line, target=target, kind="continuation line"
                )
                _require(
                    line.get("base_parameter_hash") == parent_state["parameter_hash"],
                    "line base hash mismatch",
                )
                _require(
                    _close(
                        line["realized_path_length"],
                        _float(line["alpha"]) * TARGET_NORM,
                        atol=2e-12,
                    ),
                    "line path length mismatch",
                )
        _require(
            _close(proposal["current_A"], parent_state["exact_a_per_dim"], atol=1e-9)
            and _close(
                proposal["current_B"],
                parent_state["damped_full_burg_per_dim"],
                atol=1e-9,
            )
            and _close(
                proposal["current_low_energy"],
                parent_state["frozen_low_energy"],
                atol=1e-9,
            )
            and _close(proposal["current_a_gt1"], parent_state["a_gt1"], atol=1e-9),
            f"proposal {target} current-state link mismatch",
        )
        slopes = {
            name: _float(proposal[f"slope_{name}"]) for name in ("A", "B", "L_low")
        }
        attempted = [_float(line["alpha"]) for line in target_lines]
        _require(
            attempted == list(LINE_ALPHAS[: len(attempted)]),
            f"proposal {target} alpha prefix mismatch",
        )
        recomputed_passes: list[bool] = []
        for line in target_lines:
            _require(
                {key for key in line if key.startswith("gate_")} == expected_gate_keys,
                "line gate schema mismatch",
            )
            _line_metric_consistency(line)
            candidate = {key: line[key] for key in parent_metric_keys}
            gates = _candidate_gates(
                current=parent_state,
                candidate=candidate,
                slopes=slopes,
                alpha=_float(line["alpha"]),
                tolerances=tolerances,
            )
            _validate_gate_map(line, gates)
            repeat = _validate_repeat_payload(line, tolerances, expected_repeat_keys)
            passes = all(gates.values()) and repeat
            _require(
                _bool(line["passes"]) is passes, f"proposal {target} pass bit mismatch"
            )
            _require(
                _is_sha256(_text(line["endpoint_parameter_hash"])),
                "line endpoint hash malformed",
            )
            recomputed_passes.append(passes)
        accepted_flag = _int(proposal["accepted"]) == 1
        if accepted_flag:
            _require(
                target_lines
                and recomputed_passes[-1]
                and not any(recomputed_passes[:-1]),
                f"proposal {target} did not accept first passing alpha",
            )
            selected = target_lines[-1]
            state = state_by_update[target]
            alpha = _float(selected["alpha"])
            _require(
                _close(proposal["selected_alpha"], alpha, atol=0, rtol=0),
                "selected alpha mismatch",
            )
            _require(
                _text(selected["endpoint_parameter_hash"])
                == _text(state["parameter_hash"]),
                "selected endpoint/state hash mismatch",
            )
            _require(
                _close(proposal["candidate_A"], selected["exact_a_per_dim"], atol=1e-9)
                and _close(
                    proposal["candidate_B"],
                    selected["damped_full_burg_per_dim"],
                    atol=1e-9,
                )
                and _close(
                    proposal["candidate_low_energy"],
                    selected["frozen_low_energy"],
                    atol=1e-9,
                )
                and _close(proposal["candidate_a_gt1"], selected["a_gt1"], atol=1e-9),
                "proposal candidate/selected-line mismatch",
            )
            _require(
                _selected_line_matches_state(selected, state, parent_metric_keys),
                "selected line/committed state mismatch",
            )
            if is_continuation:
                _require(
                    proposal["endpoint_parameter_hash"] == state["parameter_hash"],
                    "proposal endpoint hash mismatch",
                )
                chain = _chain_hash(
                    chain,
                    update=target,
                    base_hash=_text(proposal["base_parameter_hash"]),
                    endpoint_hash=_text(proposal["endpoint_parameter_hash"]),
                    alpha=alpha,
                )
                _require(
                    proposal["transition_chain_sha256"] == chain,
                    "transition chain mismatch",
                )
            component = {}
            current_objectives = _objective_values(parent_state)
            candidate_objectives = _objective_values(selected)
            for name in ("A", "B", "L_low"):
                component[f"{name}_slope_negative"] = slopes[name] < 0.0
                component[f"{name}_armijo"] = (
                    candidate_objectives[name]
                    <= current_objectives[name] + ARMIJO_C1 * alpha * slopes[name]
                )
                component[f"{name}_actual_decrease"] = (
                    current_objectives[name] - candidate_objectives[name]
                    > tolerances[name]
                )
            high_tail = (
                _float(selected["a_gt1"])
                <= _float(parent_state["a_gt1"]) + tolerances["A_gt1"]
            )
            _require(
                all(component.values()) and high_tail, "accepted transition gate failed"
            )
            audit_rows.append(
                {
                    "accepted_update": target,
                    "alpha": alpha,
                    **{key: int(value) for key, value in component.items()},
                    "high_tail_nonincrease": int(high_tail),
                    "selected_line_row_passes": 1,
                    "committed_state_matches": 1,
                    "transition_passes": 1,
                }
            )
        else:
            failure = _text(proposal["failure"])
            _require(
                _float(proposal["selected_alpha"]) == 0.0,
                "terminal selected alpha is nonzero",
            )
            for name in (
                "candidate_A",
                "candidate_B",
                "candidate_low_energy",
                "candidate_a_gt1",
            ):
                _require(proposal[name] is None, f"terminal {name} is populated")
            if failure == "exact_cancellation":
                _require(
                    not target_lines
                    and not _bool(proposal["relaxed_nonzero_gate_pass"]),
                    "exact cancellation control flow mismatch",
                )
            elif failure == "nonnegative_joint_slope":
                _require(
                    not target_lines
                    and _bool(proposal["relaxed_nonzero_gate_pass"])
                    and not all(value < 0.0 for value in slopes.values()),
                    "nonnegative slope control flow mismatch",
                )
            elif failure == "backtracking_exhausted":
                _require(
                    len(target_lines) == len(LINE_ALPHAS)
                    and all(value < 0.0 for value in slopes.values())
                    and not any(recomputed_passes),
                    "backtracking control flow mismatch",
                )
            else:
                raise ReviewError(f"unknown terminal failure: {failure}")
            _require(
                proposal["transition_chain_sha256"] == chain, "terminal chain mismatch"
            )
    _require(progress["transition_chain_sha256"] == chain, "progress chain mismatch")
    summary = {
        "accepted_proposal_count_matches_states": len(accepted) == len(states) - 1,
        "accepted_update_sequence_exact": [
            _int(row["target_update"]) for row in accepted
        ]
        == list(range(1, len(accepted) + 1)),
        "all_historical_transitions_pass": all(
            row["transition_passes"] == 1 for row in audit_rows
        ),
        "all_historical_a_armijo_pass": all(
            row["A_slope_negative"] == 1
            and row["A_armijo"] == 1
            and row["A_actual_decrease"] == 1
            for row in audit_rows
        ),
    }
    _require(all(summary.values()), "historical transition summary failed")
    return {
        "summary": summary,
        "audit_rows": audit_rows,
        "transition_chain_sha256": chain,
    }


def _near_cancellation_diagnostics(proposals: pd.DataFrame) -> dict[str, Any]:
    rows = proposals.loc[proposals["target_update"].astype(int).gt(START_UPDATE)]
    if rows.empty:
        return {
            "minimum_source_norm": 0.0,
            "final_source_norm": 0.0,
            "maximum_amplification": 0.0,
            "maximum_amplification_unbounded": False,
            "minimum_fp32_fp64_direction_cosine": 0.0,
            "maximum_fp32_fp64_direction_relative_error": 0.0,
            "minimum_accepted_alpha": 0.0,
            "descriptive_only": True,
        }
    sources = rows["unit_common_source_norm"].astype(float)
    unbounded = bool(sources.eq(0.0).any())
    maximum_amplification = (
        None if unbounded else float(rows["common_amplification"].astype(float).max())
    )
    shadow = rows.loc[rows["fp64_shadow_valid"].map(_bool)]
    accepted = rows.loc[rows["accepted"].astype(int).eq(1)]
    return {
        "minimum_source_norm": float(sources.min()),
        "final_source_norm": float(sources.iloc[-1]),
        "maximum_amplification": maximum_amplification,
        "maximum_amplification_unbounded": unbounded,
        "minimum_fp32_fp64_direction_cosine": float(
            shadow["fp32_fp64_direction_cosine"].astype(float).min()
        )
        if not shadow.empty
        else 0.0,
        "maximum_fp32_fp64_direction_relative_error": float(
            shadow["fp32_fp64_direction_relative_error"].astype(float).max()
        )
        if not shadow.empty
        else 0.0,
        "minimum_accepted_alpha": float(accepted["selected_alpha"].astype(float).min())
        if not accepted.empty
        else 0.0,
        "descriptive_only": True,
    }


def _derive_outcome(
    *,
    states: pd.DataFrame,
    spectra: pd.DataFrame,
    proposals: pd.DataFrame,
    tolerances: Mapping[str, float],
    accepted_updates: int,
    termination: str,
    history_summary: Mapping[str, bool],
) -> dict[str, Any]:
    initial = states.iloc[0]
    intervention = states.loc[
        states["accepted_update"].astype(int).eq(START_UPDATE)
    ].iloc[0]
    final = states.iloc[-1]
    total_a = _float(initial["exact_a_per_dim"]) - _float(final["exact_a_per_dim"])
    total_low = _float(initial["a_low90_abs_per_dim"]) - _float(
        final["a_low90_abs_per_dim"]
    )
    continuation_a = _float(intervention["exact_a_per_dim"]) - _float(
        final["exact_a_per_dim"]
    )
    continuation_low = _float(intervention["a_low90_abs_per_dim"]) - _float(
        final["a_low90_abs_per_dim"]
    )
    low90_decreases = -states["a_low90_abs_per_dim"].astype(float).diff()
    tail = states["accepted_update"].astype(int).between(81, 100)
    tail_bulk = int((low90_decreases.loc[tail] > tolerances["A_low90"]).sum())
    a_diffs = states["exact_a_per_dim"].astype(float).diff().iloc[1:]
    high_diffs = states["a_gt1"].astype(float).diff().iloc[1:]
    bulk_fraction = total_low / max(total_a, 1e-30)
    success_gates = {
        "one_hundred_accepted_updates": accepted_updates == MAX_ACCEPTED_UPDATES,
        "tail_bulk_activity": tail_bulk >= SUCCESS_TAIL_BULK_UPDATES_MIN,
        "final_a_at_most_0p90": _float(final["exact_a_per_dim"]) <= SUCCESS_A_MAX,
        "accepted_a_strictly_decreases": bool((a_diffs < -tolerances["A"]).all()),
        "all_historical_a_armijo_transitions": bool(
            history_summary["all_historical_a_armijo_pass"]
        ),
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
            states, excluded={"parameter_hash", "low_basis_hash", "phase"}
        ),
        "all_spectrum_values_finite": _frame_numeric_finite(
            spectra, excluded={"phase"}
        ),
        "all_raw_spectrum_minima_valid": bool(
            states["m_raw_eig_min"].astype(float).ge(-1e-8).all()
        ),
        "all_exact_a_closures_valid": bool(
            states["a_direct_abs_error"].astype(float).le(1e-9).all()
            and states["a_trace_abs_error"].astype(float).le(1e-9).all()
        ),
        "final_checkpoint_replay": True,
    }
    continuation_rows = proposals.loc[
        proposals["target_update"].astype(int).gt(START_UPDATE)
    ]
    immediate = bool(
        not continuation_rows.empty
        and _int(continuation_rows.iloc[0]["target_update"]) == START_UPDATE + 1
        and _int(continuation_rows.iloc[0]["accepted"]) == 1
    )
    terminal = (
        continuation_rows.iloc[-1]
        if not continuation_rows.empty
        and _int(continuation_rows.iloc[-1]["accepted"]) == 0
        else None
    )
    b_non_descent = bool(
        termination == "nonnegative_joint_slope"
        and terminal is not None
        and _float(terminal["slope_A"]) < 0.0
        and _float(terminal["slope_L_low"]) < 0.0
        and _float(terminal["slope_B"]) >= 0.0
    )
    grid_exhaustion = bool(
        termination == "backtracking_exhausted"
        and terminal is not None
        and all(_float(terminal[f"slope_{name}"]) < 0.0 for name in ("A", "B", "L_low"))
    )
    new_updates = accepted_updates - START_UPDATE
    scalar_only = bool(
        immediate
        and continuation_a > tolerances["A"]
        and bulk_fraction < SUCCESS_BULK_FRACTION_MIN
    )
    return {
        "immediate_premature_cutoff": immediate,
        "sustained_continuation": new_updates >= SUSTAINED_NEW_UPDATES,
        "b_non_descent": b_non_descent,
        "finite_grid_exhaustion": grid_exhaustion,
        "finite_grid_conflict": bool(b_non_descent or grid_exhaustion),
        "scalar_only_repair": scalar_only,
        "new_accepted_updates": new_updates,
        "scientific_success": bool(all(success_gates.values())),
        "success_gates": success_gates,
        "termination": termination,
        "total_a_reduction": total_a,
        "total_lower90_reduction": total_low,
        "total_non_top_only_fraction": bulk_fraction,
        "continuation_a_reduction": continuation_a,
        "continuation_lower90_reduction": continuation_low,
        "continuation_non_top_only_fraction": continuation_low
        / max(continuation_a, 1e-30),
        "tail_bulk_updates": tail_bulk,
        "near_cancellation_diagnostics": _near_cancellation_diagnostics(proposals),
    }


def _validate_decision(
    publication: Mapping[str, Any],
    packet: Mapping[str, Any],
    history: Mapping[str, Any],
) -> dict[str, Any]:
    decision = publication["decision"]
    finalized = publication["finalized"]
    progress = packet["progress"]
    states = packet["states"]
    spectra = packet["spectra"]
    proposals = packet["proposals"]
    accepted = _int(progress["accepted_updates"])
    tolerances = {key: _float(value) for key, value in progress["tolerances"].items()}
    outcome = _derive_outcome(
        states=states,
        spectra=spectra,
        proposals=proposals,
        tolerances=tolerances,
        accepted_updates=accepted,
        termination=_text(progress["termination"]),
        history_summary=history["summary"],
    )
    _require(
        _equivalent(decision.get("outcome"), outcome, atol=2e-9, rtol=2e-9),
        "decision outcome does not independently recompute",
    )
    _require(
        decision.get("scientific_success") is outcome["scientific_success"],
        "decision scientific-success mismatch",
    )
    _require(
        finalized.get("scientific_success") is outcome["scientific_success"],
        "FINALIZED scientific-success mismatch",
    )
    for name, expected in (
        ("accepted_updates", accepted),
        ("new_accepted_updates", accepted - START_UPDATE),
        ("termination", progress["termination"]),
    ):
        _require(
            decision.get(name) == expected and finalized.get(name) == expected,
            f"decision/finalized mismatch: {name}",
        )
    _require(decision.get("selected_arm") == "low", "decision selected arm mismatch")
    _require(
        finalized.get("immediate_premature_cutoff")
        is outcome["immediate_premature_cutoff"],
        "FINALIZED immediate outcome mismatch",
    )
    _require(
        finalized.get("sustained_continuation") is outcome["sustained_continuation"],
        "FINALIZED sustained outcome mismatch",
    )
    _require(
        decision.get("intervention_preflight") == progress["intervention_preflight"],
        "decision preflight mismatch",
    )
    _require(
        decision.get("initial_metrics") == progress["initial_metrics"],
        "decision initial metrics mismatch",
    )
    _require(
        decision.get("tolerances") == progress["tolerances"],
        "decision tolerances mismatch",
    )
    _require(
        decision.get("historical_transition_summary") == history["summary"],
        "decision history summary mismatch",
    )
    validity_gates = decision.get("validity_gates")
    _require(
        isinstance(validity_gates, Mapping)
        and set(validity_gates) == EXPECTED_VALIDITY_GATE_NAMES
        and all(value is True for value in validity_gates.values()),
        "producer validity gate schema/value mismatch",
    )
    progress_gates = decision.get("progress_audit_gates")
    _require(
        isinstance(progress_gates, Mapping)
        and set(progress_gates) == EXPECTED_PROGRESS_AUDIT_GATE_NAMES
        and all(value is True for value in progress_gates.values()),
        "producer progress-audit gate schema/value mismatch",
    )
    replay_errors = decision.get("final_replay_errors")
    _require(
        isinstance(replay_errors, Mapping) and replay_errors,
        "final replay errors missing",
    )
    expected_replay_names = {
        name
        for name, value in packet["progress"]["state_rows"][-1].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and name
        not in {
            "accepted_update",
            "parent_frozen_low_energy",
            "hessian_sec",
        }
    }
    _require(
        set(replay_errors) == expected_replay_names,
        "final replay error schema mismatch",
    )
    replay_values = [_float(value) for value in replay_errors.values()]
    _require(
        all(value >= 0.0 for value in replay_values)
        and max(replay_values) <= PARENT_REPLAY_MAX_ERROR,
        "final metric replay error too large",
    )
    spectrum_error = _float(decision["final_spectrum_replay_error"])
    _require(
        0.0 <= spectrum_error <= PARENT_REPLAY_MAX_ERROR,
        "final spectrum replay error too large",
    )
    _require(
        decision.get("final_checkpoint_file_sha256")
        == publication["artifact_hashes"]["final_checkpoint.pt"],
        "decision checkpoint file hash mismatch",
    )
    start_row = (
        states.loc[states["accepted_update"].astype(int).eq(START_UPDATE)]
        .iloc[0]
        .to_dict()
    )
    _require(
        _equivalent(
            decision.get("intervention_start_metrics"),
            start_row,
            atol=1e-12,
            rtol=1e-12,
        ),
        "decision intervention-start metric mismatch",
    )
    final_row = states.iloc[-1]
    final_metrics = decision.get("final_metrics")
    _require(isinstance(final_metrics, Mapping), "decision final metrics missing")
    expected_final_metric_names = set(packet["progress"]["state_rows"][-1]) - {
        "accepted_update",
        "parameter_hash",
        "parent_frozen_low_energy",
        "phase",
    }
    _require(
        set(final_metrics) == expected_final_metric_names,
        "decision final metric schema mismatch",
    )
    for name, value in final_metrics.items():
        if name == "hessian_sec":
            _require(_float(value) >= 0.0, "final hessian timing invalid")
            continue
        _require(
            name in final_row
            and _equivalent(value, final_row[name], atol=2e-9, rtol=2e-9),
            f"decision final metric mismatch: {name}",
        )
    return {
        "outcome": outcome,
        "final_replay_max_abs_error": max(replay_values),
        "final_spectrum_replay_error": spectrum_error,
    }


def review_output(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    started = time.perf_counter()
    review = Review()
    publication = review.phase(
        "static_publication", lambda: _validate_static_and_publication(output)
    )
    packet = (
        review.phase(
            "checkpoint_tables", lambda: _validate_checkpoint_and_tables(output)
        )
        if publication is not None
        else None
    )
    spectrum = (
        review.phase(
            "state_spectrum",
            lambda: _validate_spectrum_tables(packet["states"], packet["spectra"]),
        )
        if packet is not None
        else None
    )
    history = (
        review.phase(
            "control_flow",
            lambda: _validate_history(packet["progress"], packet["parent"]),
        )
        if packet is not None
        else None
    )
    if packet is not None and history is not None:

        def validate_history_csv() -> dict[str, Any]:
            frame = _read_csv(output / "historical_transition_audit.csv")
            _require(
                _checkpoint_rows_match_csv(history["audit_rows"], frame),
                "historical audit CSV mismatch",
            )
            return {"rows": len(frame)}

        review.phase("historical_csv", validate_history_csv)
    decision_result = None
    if publication is not None and packet is not None and history is not None:
        publication = dict(publication)
        publication["output"] = str(output)
        decision_result = review.phase(
            "decision_outcome", lambda: _validate_decision(publication, packet, history)
        )
    valid = bool(review.gates and all(review.gates.values()) and not review.errors)
    outcome = decision_result["outcome"] if decision_result is not None else None
    review.limitations.extend(
        [
            "CPU review does not rerun model HVPs or decoder gradients; it validates their frozen lineage, repeat payload semantics, spectral consequences, and serialized replay bounds.",
            "The scientific interpretation remains one-state, one-VAE, one normalized-bisector-path evidence.",
        ]
    )
    return {
        "review_protocol_id": REVIEW_PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "reviewer_source_sha256": _sha256_file(Path(__file__)),
        "reviewed_output": str(output.resolve()),
        "producer_source_sha256": EXPECTED_PRODUCER_SHA256,
        "producer_normalized_source_sha256": EXPECTED_NORMALIZED_PRODUCER_SHA256,
        "valid": valid,
        "scientific_success": outcome["scientific_success"]
        if valid and outcome
        else None,
        "outcome": outcome if valid else None,
        "failed_gates": [name for name, passed in review.gates.items() if not passed],
        "gates": review.gates,
        "details": {
            **review.details,
            "publication": (
                {
                    "artifact_hashes": publication["artifact_hashes"],
                    "png_dimensions": publication["png_dimensions"],
                    "run_log_sha256": publication["run_log_sha256"],
                }
                if publication is not None
                else None
            ),
            "spectrum": spectrum,
            "preflight": packet["preflight"] if packet is not None else None,
            "history_summary": history["summary"] if history is not None else None,
            "decision_review": decision_result,
        },
        "errors": review.errors,
        "limitations": review.limitations,
        "elapsed_sec": time.perf_counter() - started,
        "read_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only CPU review of relaxed-cancellation continuation artifacts"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = review_output(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
