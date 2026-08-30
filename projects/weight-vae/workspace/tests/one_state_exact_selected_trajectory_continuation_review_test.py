from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.review_one_state_exact_selected_trajectory_continuation import (
    DIMENSION,
    START_UPDATE,
    TARGET_NORM,
    ReviewError,
    _candidate_gates,
    _checkpoint_rows_match_csv,
    _derive_outcome,
    _near_cancellation_diagnostics,
    _normalized_producer_sha256,
    _repeat_passes,
    _spectral_metrics,
    _validate_continuation_direction,
    _validate_continuation_row_identity,
    _validate_exact_manifest,
    _validate_gate_map,
    _validate_repeat_payload,
    _validate_spectrum_tables,
)


TOLERANCES = {
    "A": 1e-9,
    "B": 1e-9,
    "L_low": 1e-10,
    "A_low90": 1e-9,
    "A_gt1": 1e-9,
    "m_max": 1e-8,
    "m_p50": 1e-10,
    "effective_rank": 1e-8,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_repeat_payload() -> dict[str, float]:
    names = {
        "repeat_exact_a_per_dim_abs_error",
        "repeat_damped_full_burg_per_dim_abs_error",
        "repeat_frozen_low_energy_abs_error",
        "repeat_a_low90_abs_per_dim_abs_error",
        "repeat_a_gt1_abs_error",
        "repeat_m_max_abs_error",
        "repeat_m_p50_abs_error",
        "repeat_effective_rank_abs_error",
        "repeat_count_lt_1e_4_abs_error",
        "repeat_count_lt_1e_2_abs_error",
        "repeat_count_lt_0p1_abs_error",
        "repeat_spectrum_max_abs_error",
        "repeat_parameter_hash_unchanged",
    }
    result = {name: 0.0 for name in names}
    result["repeat_parameter_hash_unchanged"] = 1.0
    return result


def _gradient_metadata() -> dict[str, object]:
    retained = [1024] * 8
    peaks = [2048] * 8
    return {
        "basis_count": 512,
        "block_count": 8,
        "block_size": 64,
        "retained_memory_by_block_bytes": retained,
        "peak_memory_by_block_bytes": peaks,
        "retained_memory_range_bytes": 0,
        "block_peak_range_bytes": 0,
        "memory_gate_pass": True,
        "memory_early_median_bytes": 1024.0,
        "memory_late_median_bytes": 1024.0,
        "elapsed_sec": 1.0,
        "A_unused_parameter_tensors_all_blocks": 0,
        "B_unused_parameter_tensors_all_blocks": 0,
        "low_unused_parameter_tensors_all_blocks": 0,
        "A_all_accumulators_float64": True,
        "B_all_accumulators_float64": True,
        "low_all_accumulators_float64": True,
        "k_a_norm": 1.0,
        "k_b_norm": 2.0,
        "k_low_norm": 3.0,
        "burg_gradient_formula_max_abs_error": 1e-7,
        "gradient_payload_valid": True,
    }


def _zero_cancellation_proposal() -> dict[str, object]:
    return {
        "gradient_a_norm": 2.0,
        "gradient_x_norm": 3.0,
        "gradient_cosine": -1.0,
        "unit_common_source_norm": 0.0,
        "fp64_common_source_norm": 0.0,
        "theoretical_common_source_norm": 0.0,
        "fp32_source_vs_theory_abs_error": 0.0,
        "fp64_source_vs_theory_abs_error": 0.0,
        "legacy_cancellation_gate_pass": False,
        "relaxed_nonzero_gate_pass": False,
        "fp64_shadow_valid": False,
        "common_amplification": None,
        "common_amplification_unbounded": True,
        "target_over_common_source": None,
        "fp32_fp64_direction_cosine": 0.0,
        "fp32_fp64_direction_relative_error": 0.0,
        "direction_norm": 0.0,
        "slope_A": 0.0,
        "slope_B": 0.0,
        "slope_L_low": 0.0,
        "theoretical_slope_A": 0.0,
        "theoretical_slope_L_low": 0.0,
        "slope_A_vs_theory_abs_error": 0.0,
        "slope_L_low_vs_theory_abs_error": 0.0,
        "gradient_metadata": json.dumps(_gradient_metadata()),
        "selected_alpha": 0.0,
        "realized_path_length": 0.0,
    }


def _spectrum_packet() -> tuple[pd.DataFrame, pd.DataFrame]:
    eig = np.linspace(1e-6, 2.0, DIMENSION, dtype=np.float64)
    metrics = _spectral_metrics(eig)
    state = {
        **metrics,
        "accepted_update": 0,
        "parameter_hash": "a" * 64,
        "phase": "i6",
        "m_raw_eig_min": float(eig[0]),
        "a_direct_abs_error": 0.0,
        "a_trace_abs_error": 0.0,
        "low_basis_orthogonality_max_abs": 0.0,
        "low_basis_eigen_residual_relative": 0.0,
        "helper_eigenvalue_max_abs_error": 0.0,
        "low_basis_hash": "b" * 64,
        "parent_frozen_low_energy": metrics["frozen_low_energy"],
    }
    spectra = pd.DataFrame(
        {
            "accepted_update": np.zeros(DIMENSION, dtype=np.int64),
            "rank": np.arange(DIMENSION, dtype=np.int64),
            "m_eigenvalue": eig,
            "a_contribution": np.square(eig - 1.0),
            "phase": ["i6"] * DIMENSION,
        }
    )
    return pd.DataFrame([state]), spectra


def _outcome_states(last_update: int) -> pd.DataFrame:
    rows = []
    for update in range(last_update + 1):
        rows.append(
            {
                "accepted_update": update,
                "exact_a_per_dim": 1.1 - 0.001 * update,
                "a_low90_abs_per_dim": 0.9,
                "a_gt1": 0.1 - 0.0001 * update,
                "damped_full_burg_per_dim": 2.0 - 0.001 * update,
                "m_p50": 0.01 + 0.0001 * update,
                "effective_rank": 10.0 + 0.01 * update,
                "count_lt_1e_4": 100,
                "count_lt_1e_2": 200,
                "count_lt_0p1": 300,
                "m_max": 2.0,
                "m_raw_eig_min": 0.0,
                "a_direct_abs_error": 0.0,
                "a_trace_abs_error": 0.0,
                "parameter_hash": f"{update:064x}",
                "low_basis_hash": "a" * 64,
                "phase": "i6" if update <= START_UPDATE else "relaxed_continuation",
            }
        )
    return pd.DataFrame(rows)


def _outcome_spectra() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "accepted_update": 0,
                "rank": 0,
                "m_eigenvalue": 0.1,
                "a_contribution": 0.81,
                "phase": "i6",
            }
        ]
    )


def _outcome_proposal(
    *, accepted: int, slope_b: float, source: float = 0.2
) -> dict[str, object]:
    return {
        "target_update": START_UPDATE + 1,
        "accepted": accepted,
        "slope_A": -1.0,
        "slope_B": slope_b,
        "slope_L_low": -1.0,
        "unit_common_source_norm": source,
        "common_amplification": 1.0 / source if source else None,
        "fp64_shadow_valid": True,
        "fp32_fp64_direction_cosine": 1.0,
        "fp32_fp64_direction_relative_error": 0.0,
        "selected_alpha": 0.5 if accepted else 0.0,
    }


def test_normalized_hash_masks_only_self_hash_assignment(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text(
        'EXPECTED_NORMALIZED_SOURCE_SHA256 = "old"\nVALUE = 1\n',
        encoding="utf-8",
    )
    second.write_text(
        'EXPECTED_NORMALIZED_SOURCE_SHA256 = "new"\nVALUE = 1\n',
        encoding="utf-8",
    )
    assert _normalized_producer_sha256(first) == _normalized_producer_sha256(second)
    second.write_text(
        'EXPECTED_NORMALIZED_SOURCE_SHA256 = "new"\nVALUE = 2\n',
        encoding="utf-8",
    )
    assert _normalized_producer_sha256(first) != _normalized_producer_sha256(second)


def test_manifest_rejects_missing_extra_and_tampered_artifacts(
    tmp_path: Path,
) -> None:
    for name, value in (("a.bin", b"a"), ("b.bin", b"b"), ("FINALIZED.json", b"{}")):
        (tmp_path / name).write_bytes(value)
    manifest = {
        "artifacts": {
            "a.bin": _sha256(tmp_path / "a.bin"),
            "b.bin": _sha256(tmp_path / "b.bin"),
        }
    }
    _validate_exact_manifest(
        tmp_path,
        manifest,
        expected_artifacts={"a.bin", "b.bin"},
        unmanifested={"FINALIZED.json"},
    )

    (tmp_path / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ReviewError, match="file set"):
        _validate_exact_manifest(
            tmp_path,
            manifest,
            expected_artifacts={"a.bin", "b.bin"},
            unmanifested={"FINALIZED.json"},
        )
    (tmp_path / "extra.bin").unlink()
    (tmp_path / "extra_dir").mkdir()
    with pytest.raises(ReviewError, match="directory"):
        _validate_exact_manifest(
            tmp_path,
            manifest,
            expected_artifacts={"a.bin", "b.bin"},
            unmanifested={"FINALIZED.json"},
        )
    (tmp_path / "extra_dir").rmdir()
    (tmp_path / "a.bin").write_bytes(b"tampered")
    with pytest.raises(ReviewError, match="hash mismatch"):
        _validate_exact_manifest(
            tmp_path,
            manifest,
            expected_artifacts={"a.bin", "b.bin"},
            unmanifested={"FINALIZED.json"},
        )


def test_gate_bits_are_recomputed_from_metrics_and_slopes() -> None:
    current = {
        "exact_a_per_dim": 2.0,
        "damped_full_burg_per_dim": 3.0,
        "frozen_low_energy": 0.1,
        "a_gt1": 0.4,
    }
    candidate = {
        "exact_a_per_dim": 1.9,
        "damped_full_burg_per_dim": 2.9,
        "frozen_low_energy": 0.2,
        "a_gt1": 0.3,
        "a_low90_abs_per_dim": 0.5,
        "a_direct_abs_error": 0.0,
        "a_trace_abs_error": 0.0,
        "m_raw_eig_min": 0.0,
        "low_basis_hash": "a" * 64,
    }
    slopes = {"A": -1.0, "B": -1.0, "L_low": -1.0}
    gates = _candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=1 / 128,
        tolerances=TOLERANCES,
    )
    assert all(gates.values())
    row = {f"gate_{name}": int(value) for name, value in gates.items()}
    _validate_gate_map(row, gates)
    row["gate_A_armijo"] = 0
    with pytest.raises(ReviewError, match="gate_A_armijo"):
        _validate_gate_map(row, gates)


@pytest.mark.parametrize("bad", [-1.0, math.nan])
def test_repeat_payload_rejects_negative_and_nonfinite_values(bad: float) -> None:
    row = _valid_repeat_payload()
    assert _validate_repeat_payload(row, TOLERANCES, set(row))
    row["repeat_spectrum_max_abs_error"] = bad
    with pytest.raises(ReviewError, match="repeat"):
        _validate_repeat_payload(row, TOLERANCES, set(row))


def test_original_repeat_predicate_is_used_after_payload_validation() -> None:
    row = _valid_repeat_payload()
    row["repeat_exact_a_per_dim_abs_error"] = 2 * TOLERANCES["A"]
    assert not _repeat_passes(row, TOLERANCES)
    assert not _validate_repeat_payload(row, TOLERANCES, set(row))


def test_spectrum_metrics_recompute_and_physical_reordering_is_rejected() -> None:
    states, spectra = _spectrum_packet()
    result = _validate_spectrum_tables(states, spectra)
    assert result["state_count"] == 1
    assert result["worst_a_contribution_abs_error"] == 0.0

    reordered = spectra.copy()
    reordered.iloc[[0, 1]] = reordered.iloc[[1, 0]].to_numpy()
    with pytest.raises(ReviewError, match="physical row order"):
        _validate_spectrum_tables(states, reordered)


def test_spectrum_metric_and_contribution_tampering_is_rejected() -> None:
    states, spectra = _spectrum_packet()
    bad_state = states.copy()
    bad_state.loc[0, "m_p50"] += 1e-3
    with pytest.raises(ReviewError, match="m_p50 mismatch"):
        _validate_spectrum_tables(bad_state, spectra)

    bad_spectrum = spectra.copy()
    bad_spectrum.loc[10, "a_contribution"] += 1e-3
    with pytest.raises(ReviewError, match="a_contribution"):
        _validate_spectrum_tables(states, bad_spectrum)


def test_exact_zero_cancellation_is_serialized_as_unbounded() -> None:
    proposal = _zero_cancellation_proposal()
    details = _validate_continuation_direction(proposal)
    assert details["relaxed"] is False
    frame = pd.DataFrame(
        [
            {
                **proposal,
                "target_update": START_UPDATE + 1,
                "accepted": 0,
            }
        ]
    )
    outcome = _near_cancellation_diagnostics(frame)
    assert outcome["minimum_source_norm"] == 0.0
    assert outcome["final_source_norm"] == 0.0
    assert outcome["maximum_amplification"] is None
    assert outcome["maximum_amplification_unbounded"] is True


def test_zero_cancellation_with_finite_amplification_is_rejected() -> None:
    proposal = _zero_cancellation_proposal()
    proposal["common_amplification"] = 0.0
    with pytest.raises(ReviewError, match="finite amplification"):
        _validate_continuation_direction(proposal)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_arm", "B", "selected arm"),
        ("local_proposal", 2, "local proposal"),
        ("phase", "i6", "phase"),
    ],
)
def test_continuation_identity_tampering_is_rejected(
    field: str, value: object, message: str
) -> None:
    row = {
        "phase": "relaxed_continuation",
        "selected_arm": "low",
        "target_update": START_UPDATE + 1,
        "local_proposal": 1,
    }
    _validate_continuation_row_identity(row, target=START_UPDATE + 1, kind="proposal")
    tampered = copy.deepcopy(row)
    tampered[field] = value
    with pytest.raises(ReviewError, match=message):
        _validate_continuation_row_identity(
            tampered, target=START_UPDATE + 1, kind="proposal"
        )


def test_checkpoint_csv_comparison_rejects_semantic_row_tamper() -> None:
    records = [{"target_update": 18, "accepted": 1, "failure": ""}]
    frame = pd.DataFrame([{"target_update": 18, "accepted": 1, "failure": np.nan}])
    assert _checkpoint_rows_match_csv(records, frame)
    frame.loc[0, "accepted"] = 0
    assert not _checkpoint_rows_match_csv(records, frame)


def test_nonzero_direction_diagnostic_rejects_false_radius() -> None:
    cosine = -0.5
    source = math.sqrt(2.0 + 2.0 * cosine)
    norm_a = 2.0
    norm_low = 3.0
    slope_a = -TARGET_NORM * norm_a * source / 2.0
    slope_low = -TARGET_NORM * norm_low * source / 2.0
    row = {
        **_zero_cancellation_proposal(),
        "gradient_a_norm": norm_a,
        "gradient_x_norm": norm_low,
        "gradient_cosine": cosine,
        "unit_common_source_norm": source,
        "fp64_common_source_norm": source,
        "theoretical_common_source_norm": source,
        "legacy_cancellation_gate_pass": True,
        "relaxed_nonzero_gate_pass": True,
        "fp64_shadow_valid": True,
        "common_amplification": 1.0 / source,
        "common_amplification_unbounded": False,
        "target_over_common_source": TARGET_NORM / source,
        "fp32_fp64_direction_cosine": 1.0,
        "direction_norm": TARGET_NORM,
        "slope_A": slope_a,
        "slope_B": -0.1,
        "slope_L_low": slope_low,
        "theoretical_slope_A": slope_a,
        "theoretical_slope_L_low": slope_low,
        "selected_alpha": 0.5,
        "realized_path_length": 0.5 * TARGET_NORM,
    }
    _validate_continuation_direction(row)
    row["direction_norm"] = TARGET_NORM + 1e-4
    with pytest.raises(ReviewError, match="radius"):
        _validate_continuation_direction(row)


def test_outcome_distinguishes_b_non_descent_from_grid_exhaustion() -> None:
    common = {
        "states": _outcome_states(START_UPDATE),
        "spectra": _outcome_spectra(),
        "tolerances": TOLERANCES,
        "accepted_updates": START_UPDATE,
        "history_summary": {"all_historical_a_armijo_pass": True},
    }
    b_conflict = _derive_outcome(
        **common,
        proposals=pd.DataFrame([_outcome_proposal(accepted=0, slope_b=0.1)]),
        termination="nonnegative_joint_slope",
    )
    assert b_conflict["b_non_descent"] is True
    assert b_conflict["finite_grid_exhaustion"] is False
    assert b_conflict["finite_grid_conflict"] is True

    grid_conflict = _derive_outcome(
        **common,
        proposals=pd.DataFrame([_outcome_proposal(accepted=0, slope_b=-0.1)]),
        termination="backtracking_exhausted",
    )
    assert grid_conflict["b_non_descent"] is False
    assert grid_conflict["finite_grid_exhaustion"] is True
    assert grid_conflict["finite_grid_conflict"] is True


def test_outcome_recomputes_immediate_sustained_and_scalar_only() -> None:
    accepted_updates = START_UPDATE + 10
    proposals = pd.DataFrame(
        [
            {
                **_outcome_proposal(accepted=1, slope_b=-0.1),
                "target_update": target,
            }
            for target in range(START_UPDATE + 1, accepted_updates + 1)
        ]
    )
    outcome = _derive_outcome(
        states=_outcome_states(accepted_updates),
        spectra=_outcome_spectra(),
        proposals=proposals,
        tolerances=TOLERANCES,
        accepted_updates=accepted_updates,
        termination="synthetic_nonterminal",
        history_summary={"all_historical_a_armijo_pass": True},
    )
    assert outcome["immediate_premature_cutoff"] is True
    assert outcome["sustained_continuation"] is True
    assert outcome["scalar_only_repair"] is True
