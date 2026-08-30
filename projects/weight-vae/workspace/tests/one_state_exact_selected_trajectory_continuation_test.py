from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import run_one_state_exact_selected_trajectory_continuation as continuation


def _near_cancelled_gradients() -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.nn.Parameter]
]:
    cosine = -0.92
    gradient_a = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    gradient_low = [
        torch.tensor([cosine, math.sqrt(1.0 - cosine**2)], dtype=torch.float64)
    ]
    active = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))]
    return gradient_a, gradient_low, active


def test_relaxed_direction_accepts_sublegacy_source_and_targets_radius() -> None:
    gradient_a, gradient_low, active = _near_cancelled_gradients()

    direction, diagnostics = continuation._relaxed_unit_direction(
        gradient_a, gradient_low, active
    )

    assert direction is not None
    assert diagnostics["unit_common_source_norm"] == pytest.approx(0.4, abs=2e-8)
    assert (
        diagnostics["unit_common_source_norm"] < continuation.i6.CANCELLATION_NORM_MIN
    )
    assert diagnostics["legacy_cancellation_gate_pass"] is False
    assert diagnostics["relaxed_nonzero_gate_pass"] is True
    assert continuation.i6._norm(direction) == pytest.approx(
        continuation.i6.TARGET_NORM, rel=5e-7, abs=1e-9
    )
    assert continuation.i6._dot(gradient_a, direction) < 0.0
    assert continuation.i6._dot(gradient_low, direction) < 0.0


def test_relaxed_direction_rejects_exact_cancellation() -> None:
    active = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))]
    gradient_a = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    gradient_low = [torch.tensor([-1.0, 0.0], dtype=torch.float64)]

    direction, diagnostics = continuation._relaxed_unit_direction(
        gradient_a, gradient_low, active
    )

    assert direction is None
    assert diagnostics["gradient_cosine"] == pytest.approx(-1.0)
    assert diagnostics["unit_common_source_norm"] == 0.0
    assert diagnostics["fp64_common_source_norm"] == 0.0
    assert diagnostics["relaxed_nonzero_gate_pass"] is False
    assert diagnostics["common_amplification"] is None
    assert diagnostics["common_amplification_unbounded"] is True
    assert diagnostics["target_over_common_source"] is None
    assert diagnostics["direction_norm"] == 0.0


def test_relaxed_direction_fp32_fp64_diagnostics_are_finite_and_close() -> None:
    gradient_a, gradient_low, active = _near_cancelled_gradients()

    direction, diagnostics = continuation._relaxed_unit_direction(
        gradient_a, gradient_low, active
    )

    assert direction is not None
    numeric_keys = (
        "gradient_a_norm",
        "gradient_x_norm",
        "gradient_cosine",
        "unit_common_source_norm",
        "fp64_common_source_norm",
        "theoretical_common_source_norm",
        "fp32_source_vs_theory_abs_error",
        "fp64_source_vs_theory_abs_error",
        "common_amplification",
        "target_over_common_source",
        "fp32_fp64_direction_cosine",
        "fp32_fp64_direction_relative_error",
        "direction_norm",
    )
    assert all(math.isfinite(float(diagnostics[key])) for key in numeric_keys)
    assert diagnostics["unit_common_source_norm"] == pytest.approx(
        diagnostics["fp64_common_source_norm"], rel=0.0, abs=2e-8
    )
    assert diagnostics["fp64_common_source_norm"] == pytest.approx(
        diagnostics["theoretical_common_source_norm"], rel=0.0, abs=1e-12
    )
    assert diagnostics["fp32_fp64_direction_cosine"] >= 1.0 - 1e-6
    assert diagnostics["fp32_fp64_direction_relative_error"] < 1e-6


def _proposal18_payload() -> dict[str, float]:
    return {
        "gradient_a_norm": 12.259394818120903,
        "gradient_x_norm": 0.2277393895882087,
        "gradient_cosine": -0.8957795409263385,
        "unit_common_source_norm": 0.456553312923466,
    }


def _producer_direction_payload() -> dict[str, float | bool]:
    payload = _proposal18_payload()
    source = float(payload["unit_common_source_norm"])
    theoretical = math.sqrt(2.0 + 2.0 * float(payload["gradient_cosine"]))
    fp64_source = theoretical
    return {
        **payload,
        "fp64_common_source_norm": fp64_source,
        "theoretical_common_source_norm": theoretical,
        "fp32_source_vs_theory_abs_error": abs(source - theoretical),
        "fp64_source_vs_theory_abs_error": abs(fp64_source - theoretical),
        "legacy_cancellation_gate_pass": False,
        "relaxed_nonzero_gate_pass": True,
        "fp64_shadow_valid": True,
        "common_amplification": 1.0 / source,
        "common_amplification_unbounded": False,
        "target_over_common_source": continuation.i6.TARGET_NORM / source,
        "fp32_fp64_direction_cosine": 1.0,
        "fp32_fp64_direction_relative_error": 0.0,
        "direction_norm": continuation.i6.TARGET_NORM,
    }


def test_proposal18_replay_accepts_values_within_frozen_tolerance() -> None:
    stored = _proposal18_payload()
    observed = {
        key: value
        + 0.25
        * (
            continuation.PROPOSAL18_REPLAY_ATOL
            + continuation.PROPOSAL18_REPLAY_RTOL * abs(value)
        )
        for key, value in stored.items()
    }

    assert continuation._proposal18_replay_passes(observed, stored)
    errors = continuation._proposal18_replay_errors(observed, stored)
    assert set(errors) == set(stored)
    assert all(error > 0.0 for error in errors.values())


@pytest.mark.parametrize("tampered_key", tuple(_proposal18_payload()))
def test_proposal18_replay_rejects_each_out_of_tolerance_field(
    tampered_key: str,
) -> None:
    stored = _proposal18_payload()
    observed = dict(stored)
    value = stored[tampered_key]
    observed[tampered_key] += 10.0 * (
        continuation.PROPOSAL18_REPLAY_ATOL
        + continuation.PROPOSAL18_REPLAY_RTOL * abs(value)
    )

    assert not continuation._proposal18_replay_passes(observed, stored)


def test_seed_parent_rows_keeps_global_history_but_excludes_rejected_proposal18() -> (
    None
):
    progress: dict[str, Any] = {
        "state_rows": [
            {"accepted_update": update, "parameter_hash": f"state-{update}"}
            for update in range(continuation.START_UPDATE + 1)
        ],
        "spectrum_rows": [
            {"accepted_update": update, "rank": 0, "m_eigenvalue": float(update)}
            for update in range(continuation.START_UPDATE + 1)
        ],
        "proposal_rows": [
            {"target_update": update, "accepted": 1, "failure": ""}
            for update in range(1, continuation.START_UPDATE + 1)
        ]
        + [
            {
                "target_update": continuation.START_UPDATE + 1,
                "accepted": 0,
                "failure": "cancellation_gate",
            }
        ],
        "line_rows": [],
        "selection_rows": [],
    }

    seeded = continuation._seed_parent_rows(progress)

    assert [row["accepted_update"] for row in seeded["state_rows"]] == list(
        range(continuation.START_UPDATE + 1)
    )
    assert [row["target_update"] for row in seeded["proposal_rows"]] == list(
        range(1, continuation.START_UPDATE + 1)
    )
    assert all(row["phase"] == "i6" for row in seeded["state_rows"])
    assert all(row["phase"] == "i6" for row in seeded["proposal_rows"])
    assert (
        seeded["intervention_origin"]["target_update"] == continuation.START_UPDATE + 1
    )
    assert seeded["intervention_origin"]["accepted"] == 0
    assert seeded["intervention_origin"]["failure"] == "cancellation_gate"
    assert not any(
        row["target_update"] == continuation.START_UPDATE + 1
        for row in seeded["proposal_rows"]
    )
    next_global_proposal = seeded["state_rows"][-1]["accepted_update"] + 1
    assert next_global_proposal == continuation.START_UPDATE + 1
    assert next_global_proposal - continuation.START_UPDATE == 1


def test_chain_hash_is_deterministic_and_tamper_sensitive() -> None:
    arguments = {
        "previous": "parent-chain",
        "update": 18,
        "base_hash": "base-17",
        "endpoint_hash": "endpoint-18",
        "alpha": 1.0 / 128.0,
    }

    expected = continuation._chain_hash(**arguments)

    assert continuation._chain_hash(**arguments) == expected
    assert len(expected) == 64
    int(expected, 16)
    tampered = (
        {**arguments, "previous": "different-parent"},
        {**arguments, "update": 19},
        {**arguments, "base_hash": "different-base"},
        {**arguments, "endpoint_hash": "different-endpoint"},
        {**arguments, "alpha": 1.0 / 64.0},
    )
    tampered_hashes = {continuation._chain_hash(**payload) for payload in tampered}
    assert expected not in tampered_hashes
    assert len(tampered_hashes) == len(tampered)


def test_termination_after_loop_promotes_running_at_exact_maximum() -> None:
    assert continuation._termination_after_loop(99, "running") == "running"
    assert (
        continuation._termination_after_loop(
            continuation.MAX_TOTAL_ACCEPTED_UPDATES, "running"
        )
        == "max_updates_reached"
    )
    assert (
        continuation._termination_after_loop(
            continuation.MAX_TOTAL_ACCEPTED_UPDATES, "max_updates_reached"
        )
        == "max_updates_reached"
    )


def test_termination_after_loop_rejects_failure_at_exact_maximum() -> None:
    with pytest.raises(RuntimeError, match="maximum update count conflicts"):
        continuation._termination_after_loop(
            continuation.MAX_TOTAL_ACCEPTED_UPDATES, "backtracking_exhausted"
        )


def _audit_tolerances() -> dict[str, float]:
    return {
        "A": 1e-9,
        "B": 1e-9,
        "L_low": 1e-10,
        "A_low90": 1e-9,
        "A_gt1": 1e-9,
        "m_max": 1e-8,
        "m_p50": 1e-10,
        "effective_rank": 1e-8,
    }


def _passing_repeat_errors() -> dict[str, float]:
    return {
        "repeat_exact_a_per_dim_abs_error": 0.0,
        "repeat_damped_full_burg_per_dim_abs_error": 0.0,
        "repeat_frozen_low_energy_abs_error": 0.0,
        "repeat_a_low90_abs_per_dim_abs_error": 0.0,
        "repeat_a_gt1_abs_error": 0.0,
        "repeat_m_max_abs_error": 0.0,
        "repeat_m_p50_abs_error": 0.0,
        "repeat_effective_rank_abs_error": 0.0,
        "repeat_count_lt_1e_4_abs_error": 0.0,
        "repeat_count_lt_1e_2_abs_error": 0.0,
        "repeat_count_lt_0p1_abs_error": 0.0,
        "repeat_parameter_hash_unchanged": 1.0,
        "repeat_spectrum_max_abs_error": 0.0,
        "repeat_hessian_max_abs_error": 0.0,
        "repeat_task_loss_abs_error": 0.0,
    }


def _producer_gradient_metadata() -> dict[str, Any]:
    retained = [1_000] * 8
    peaks = [2_000] * 8
    return {
        "basis_count": 512,
        "block_count": 8,
        "block_size": 64,
        "retained_memory_by_block_bytes": retained,
        "peak_memory_by_block_bytes": peaks,
        "retained_memory_range_bytes": 0,
        "block_peak_range_bytes": 0,
        "memory_early_median_bytes": 1_000.0,
        "memory_late_median_bytes": 1_000.0,
        "memory_gate_pass": True,
        "elapsed_sec": 0.0,
        "A_unused_parameter_tensors_all_blocks": 0,
        "B_unused_parameter_tensors_all_blocks": 0,
        "low_unused_parameter_tensors_all_blocks": 0,
        "A_all_accumulators_float64": True,
        "B_all_accumulators_float64": True,
        "low_all_accumulators_float64": True,
        "k_a_norm": 1.0,
        "k_b_norm": 1.0,
        "k_low_norm": 1.0,
        "burg_gradient_formula_max_abs_error": 0.0,
        "gradient_payload_valid": True,
    }


@pytest.mark.parametrize("invalid_value", (-1.0, float("nan")))
def test_strict_repeat_passes_rejects_negative_and_nan(
    invalid_value: float,
) -> None:
    errors = _passing_repeat_errors()
    assert continuation._repeat_payload_values_valid(errors)
    assert continuation._strict_repeat_passes(errors, _audit_tolerances())
    assert continuation._strict_repeat_passes(
        errors, _audit_tolerances()
    ) == continuation.i6._repeat_passes(errors, _audit_tolerances())

    errors["repeat_spectrum_max_abs_error"] = invalid_value

    assert not continuation._repeat_payload_values_valid(errors)
    assert not continuation._strict_repeat_passes(errors, _audit_tolerances())


def _passing_gate_columns() -> dict[str, int]:
    return {
        f"gate_{name}": 1
        for name in (
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
    }


def _spectral_candidate(eigenvalues: np.ndarray, *, basis_hash: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        **continuation.i6_review._spectral_metrics(eigenvalues),
        "a_direct_abs_error": 0.0,
        "a_trace_abs_error": 0.0,
        "m_raw_eig_min": float(eigenvalues[0]),
        "low_basis_hash": basis_hash,
    }
    return metrics


@pytest.fixture
def audit_packets(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    parent_active_state = {
        "decoder.weight": torch.tensor([1.0, -2.0], dtype=torch.float32)
    }
    parent_active_hash = continuation._active_state_hash(parent_active_state)
    monkeypatch.setattr(continuation, "EXPECTED_NORMALIZED_SOURCE_SHA256", "source")
    monkeypatch.setattr(
        continuation, "EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256", "manifest"
    )
    monkeypatch.setattr(
        continuation, "EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256", "parent-final"
    )
    monkeypatch.setattr(
        continuation,
        "EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256",
        "parent-progress",
    )
    monkeypatch.setattr(continuation, "EXPECTED_PARENT_ACTIVE_HASH", parent_active_hash)

    eig18 = np.linspace(0.0, 0.8, 512, dtype=np.float64)
    metrics18 = _spectral_candidate(eig18, basis_hash="basis-18")
    parent_metrics = {
        "exact_a_per_dim": float(metrics18["exact_a_per_dim"]) + 0.1,
        "damped_full_burg_per_dim": float(metrics18["damped_full_burg_per_dim"]) + 0.1,
        "frozen_low_energy": float(metrics18["frozen_low_energy"]) - 0.01,
        "canonical_low_energy": float(metrics18["frozen_low_energy"]) - 0.01,
        "a_low90_abs_per_dim": float(metrics18["a_low90_abs_per_dim"]) + 0.1,
        "a_gt1": 0.05,
        "a_direct_abs_error": 0.0,
        "a_trace_abs_error": 0.0,
        "m_raw_eig_min": 0.0,
        "low_basis_hash": "basis-17",
    }
    parent_states = [
        {
            "accepted_update": update,
            "parameter_hash": (
                parent_active_hash
                if update == continuation.START_UPDATE
                else f"parent-state-{update}"
            ),
        }
        for update in range(continuation.START_UPDATE + 1)
    ]
    parent_states[-1].update(parent_metrics)
    parent_spectra = [
        {
            "accepted_update": update,
            "rank": rank,
            "m_eigenvalue": float(rank) / 512.0,
            "a_contribution": (float(rank) / 512.0 - 1.0) ** 2,
        }
        for update in range(continuation.START_UPDATE + 1)
        for rank in range(512)
    ]
    rejected_proposal18 = {
        "target_update": continuation.START_UPDATE + 1,
        "accepted": 0,
        "failure": "cancellation_gate",
        **_proposal18_payload(),
    }
    parent_progress: dict[str, Any] = {
        "state_rows": parent_states,
        "spectrum_rows": parent_spectra,
        "proposal_rows": [
            {"target_update": update, "accepted": 1, "failure": ""}
            for update in range(1, continuation.START_UPDATE + 1)
        ]
        + [rejected_proposal18],
        "line_rows": [
            {
                "target_update": continuation.START_UPDATE,
                "alpha": 1.0,
                "passes": 1,
                **metrics18,
                **_passing_gate_columns(),
                **_passing_repeat_errors(),
            }
        ],
        "selection_rows": [],
        "initial_metrics": {"exact_a_per_dim": 1.2},
        "tolerances": _audit_tolerances(),
        "active_model_state": parent_active_state,
    }
    seeded = continuation._seed_parent_rows(parent_progress)
    initial_progress: dict[str, Any] = {
        "protocol_id": continuation.PROTOCOL_ID,
        "normalized_source_sha256": "source",
        "dependency_manifest_sha256": "manifest",
        "parent_checkpoint_sha256": "parent-final",
        "parent_progress_checkpoint_sha256": "parent-progress",
        "parent_active_parameter_hash": parent_active_hash,
        "accepted_updates": continuation.START_UPDATE,
        "new_accepted_updates": 0,
        "selected_arm": "low",
        "terminal": False,
        "termination": "running",
        "transition_chain_sha256": "parent-final",
        "active_parameter_hash": parent_active_hash,
        "active_model_state": copy.deepcopy(parent_active_state),
        **{
            key: copy.deepcopy(seeded[key])
            for key in (
                "state_rows",
                "spectrum_rows",
                "proposal_rows",
                "line_rows",
                "selection_rows",
                "intervention_origin",
            )
        },
        "initial_metrics": copy.deepcopy(parent_progress["initial_metrics"]),
        "tolerances": copy.deepcopy(parent_progress["tolerances"]),
        "intervention_preflight": {},
    }

    def append_transition(
        progress: dict[str, Any],
        *,
        target: int,
        eigenvalues: np.ndarray,
        active_tensor: torch.Tensor,
    ) -> None:
        parent_state = progress["state_rows"][-1]
        base_hash = str(parent_state["parameter_hash"])
        endpoint_state = {"decoder.weight": active_tensor}
        endpoint_hash = continuation._active_state_hash(endpoint_state)
        alpha = float(continuation.i6.LINE_ALPHAS[0])
        metrics = _spectral_candidate(eigenvalues, basis_hash=f"basis-{target}")
        direction = _producer_direction_payload()
        theoretical_slope_a = (
            -continuation.i6.TARGET_NORM
            * float(direction["gradient_a_norm"])
            * float(direction["theoretical_common_source_norm"])
            / 2.0
        )
        theoretical_slope_low = (
            -continuation.i6.TARGET_NORM
            * float(direction["gradient_x_norm"])
            * float(direction["theoretical_common_source_norm"])
            / 2.0
        )
        slopes = {
            "A": theoretical_slope_a,
            "B": -1.0,
            "L_low": theoretical_slope_low,
        }
        gate_map = continuation.i6._candidate_gates(
            current=parent_state,
            candidate=metrics,
            slopes=slopes,
            alpha=alpha,
            tolerances=progress["tolerances"],
            require_low90=False,
        )
        assert all(gate_map.values())
        chain_hash = continuation._chain_hash(
            str(progress["transition_chain_sha256"]),
            update=target,
            base_hash=base_hash,
            endpoint_hash=endpoint_hash,
            alpha=alpha,
        )
        proposal = {
            "target_update": target,
            "local_proposal": target - continuation.START_UPDATE,
            "phase": "relaxed_continuation",
            "selected_arm": "low",
            "accepted": 1,
            "failure": "",
            "base_parameter_hash": base_hash,
            "endpoint_parameter_hash": endpoint_hash,
            "selected_alpha": alpha,
            "candidate_A": float(metrics["exact_a_per_dim"]),
            "candidate_B": float(metrics["damped_full_burg_per_dim"]),
            "candidate_low_energy": float(metrics["frozen_low_energy"]),
            "candidate_a_gt1": float(metrics["a_gt1"]),
            "current_A": float(parent_state["exact_a_per_dim"]),
            "current_B": float(parent_state["damped_full_burg_per_dim"]),
            "current_low_energy": float(parent_state["frozen_low_energy"]),
            "current_a_gt1": float(parent_state["a_gt1"]),
            "slope_A": slopes["A"],
            "slope_B": slopes["B"],
            "slope_L_low": slopes["L_low"],
            "transition_chain_sha256": chain_hash,
            **direction,
            "theoretical_slope_A": theoretical_slope_a,
            "theoretical_slope_L_low": theoretical_slope_low,
            "slope_A_vs_theory_abs_error": 0.0,
            "slope_L_low_vs_theory_abs_error": 0.0,
            "realized_path_length": alpha * continuation.i6.TARGET_NORM,
            "gradient_metadata": json.dumps(
                _producer_gradient_metadata(), sort_keys=True
            ),
        }
        line = {
            "target_update": target,
            "local_proposal": target - continuation.START_UPDATE,
            "phase": "relaxed_continuation",
            "selected_arm": "low",
            "base_parameter_hash": base_hash,
            "endpoint_parameter_hash": endpoint_hash,
            "alpha": alpha,
            "realized_path_length": alpha * continuation.i6.TARGET_NORM,
            "passes": 1,
            **{f"gate_{key}": int(value) for key, value in gate_map.items()},
            **metrics,
            **_passing_repeat_errors(),
        }
        committed_metrics = dict(metrics)
        committed_metrics["parent_frozen_low_energy"] = float(
            committed_metrics["frozen_low_energy"]
        )
        committed_metrics["frozen_low_energy"] = float(
            committed_metrics["canonical_low_energy"]
        )
        progress["proposal_rows"].append(proposal)
        progress["line_rows"].append(line)
        progress["state_rows"].append(
            {
                "accepted_update": target,
                "parameter_hash": endpoint_hash,
                "phase": "relaxed_continuation",
                **committed_metrics,
            }
        )
        progress["spectrum_rows"].extend(
            {
                "accepted_update": target,
                "rank": rank,
                "m_eigenvalue": float(value),
                "a_contribution": float((value - 1.0) ** 2),
                "phase": "relaxed_continuation",
            }
            for rank, value in enumerate(eigenvalues)
        )
        progress.update(
            {
                "accepted_updates": target,
                "new_accepted_updates": target - continuation.START_UPDATE,
                "transition_chain_sha256": chain_hash,
                "active_parameter_hash": endpoint_hash,
                "active_model_state": endpoint_state,
            }
        )

    continued = copy.deepcopy(initial_progress)
    append_transition(
        continued,
        target=continuation.START_UPDATE + 1,
        eigenvalues=eig18,
        active_tensor=torch.tensor([2.0, -2.0], dtype=torch.float32),
    )
    parent_error_keys = {
        key
        for key, value in parent_states[-1].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key not in {"accepted_update", "parent_frozen_low_energy", "hessian_sec"}
    }
    continued["intervention_preflight"] = {
        "proposal18_replay_pass": True,
        "proposal18_replay_errors": {key: 0.0 for key in _proposal18_payload()},
        "parent_state_metric_errors": {key: 0.0 for key in parent_error_keys},
        "parent_state_metric_max_abs_error": 0.0,
        "parent_spectrum_max_abs_error": 0.0,
        "parent_parameter_hash_matches": True,
        "parent_low_basis_hash_matches": True,
        "old_gate_rejects": True,
        "relaxed_gate_accepts": True,
        "parent_independent_review_valid": True,
    }

    continued_two = copy.deepcopy(continued)
    append_transition(
        continued_two,
        target=continuation.START_UPDATE + 2,
        eigenvalues=np.linspace(0.01, 0.85, 512, dtype=np.float64),
        active_tensor=torch.tensor([3.0, -2.0], dtype=torch.float32),
    )
    return {
        "parent": parent_progress,
        "initial": initial_progress,
        "continued": continued,
        "continued_two": continued_two,
    }


def _row_payload(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        key: progress[key]
        for key in (
            "state_rows",
            "spectrum_rows",
            "proposal_rows",
            "line_rows",
            "selection_rows",
            "intervention_origin",
        )
    }


def _audit_row_gates(
    progress: dict[str, Any], parent_progress: dict[str, Any]
) -> dict[str, bool]:
    gates, _ = continuation._audit_rows(
        rows=_row_payload(progress),
        accepted_updates=int(progress["accepted_updates"]),
        terminal=bool(progress["terminal"]),
        termination=str(progress["termination"]),
        tolerances=progress["tolerances"],
        parent_progress=parent_progress,
        stored_chain_hash=str(progress["transition_chain_sha256"]),
    )
    return gates


def test_audit_accepts_valid_seeded_initial_progress(
    audit_packets: dict[str, Any],
) -> None:
    gates = continuation._audit_progress_checkpoint(
        audit_packets["initial"], audit_packets["parent"]
    )

    assert gates
    assert all(gates.values())


def test_audit_rows_accepts_valid_one_step_continuation(
    audit_packets: dict[str, Any],
) -> None:
    gates = _audit_row_gates(audit_packets["continued"], audit_packets["parent"])

    assert all(gates.values())
    assert all(
        continuation._audit_progress_checkpoint(
            audit_packets["continued"], audit_packets["parent"]
        ).values()
    )


def test_audit_rejects_accepted_proposal_with_nonempty_failure(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress["proposal_rows"][-1]["failure"] = "backtracking_exhausted"

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_control_flow_recomputes"]
    with pytest.raises(RuntimeError, match="continuation_control_flow_recomputes"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_exact_cancellation_with_relaxed_negative_direction(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["initial"])
    proposal = copy.deepcopy(audit_packets["continued"]["proposal_rows"][-1])
    proposal.update(
        {
            "accepted": 0,
            "failure": "exact_cancellation",
            "selected_alpha": 0.0,
            "realized_path_length": 0.0,
        }
    )
    assert proposal["relaxed_nonzero_gate_pass"] is True
    assert all(proposal[f"slope_{name}"] < 0.0 for name in ("A", "B", "L_low"))
    progress["proposal_rows"].append(proposal)
    progress["terminal"] = True
    progress["termination"] = "exact_cancellation"
    progress["intervention_preflight"] = copy.deepcopy(
        audit_packets["continued"]["intervention_preflight"]
    )

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert gates["terminal_structure_exact"]
    assert not gates["continuation_control_flow_recomputes"]
    with pytest.raises(RuntimeError, match="continuation_control_flow_recomputes"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_missing_selected_line_metric(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    del progress["line_rows"][-1]["low_count"]

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_metric_schema_exact"]
    with pytest.raises(RuntimeError, match="continuation_metric_schema_exact"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_empty_gradient_metadata(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress["proposal_rows"][-1]["gradient_metadata"] = "{}"

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_gradient_metadata_valid"]
    with pytest.raises(RuntimeError, match="continuation_gradient_metadata_valid"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize(
    ("row_kind", "field"),
    (
        ("proposal_rows", "direction_norm"),
        ("proposal_rows", "realized_path_length"),
        ("line_rows", "realized_path_length"),
    ),
)
def test_audit_rejects_direction_and_path_diagnostic_tamper(
    audit_packets: dict[str, Any], row_kind: str, field: str
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress[row_kind][-1][field] += 1.0

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_diagnostics_recompute"]
    with pytest.raises(RuntimeError, match="continuation_diagnostics_recompute"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_gate_metric_tamper_with_stale_gate_bit(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress["line_rows"][-1]["a_direct_abs_error"] = 1.0

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_gate_map_recomputes"]
    assert not gates["continuation_pass_bits_recompute"]
    with pytest.raises(RuntimeError, match="continuation_gate_map_recomputes"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize("invalid_value", (-1.0, float("nan")))
def test_audit_rejects_negative_or_nan_repeat_error(
    audit_packets: dict[str, Any], invalid_value: float
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress["line_rows"][-1]["repeat_spectrum_max_abs_error"] = invalid_value

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_repeat_values_valid"]
    assert not gates["continuation_pass_bits_recompute"]
    with pytest.raises(RuntimeError, match="continuation_repeat_values_valid"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_intermediate_committed_state_metric_tamper_before_outcome(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued_two"])
    intermediate = next(
        row
        for row in progress["state_rows"]
        if int(row["accepted_update"]) == continuation.START_UPDATE + 1
    )
    intermediate["exact_a_per_dim"] += 0.01

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_endpoint_metric_links_pass"]
    assert not gates["continuation_proposal_current_links_pass"]
    assert not gates["continuation_spectrum_metrics_recompute"]
    with pytest.raises(RuntimeError, match="continuation_endpoint_metric_links_pass"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_spectrum_physical_row_reorder(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    first = (continuation.START_UPDATE + 1) * 512
    rows = progress["spectrum_rows"]
    rows[first], rows[first + 1] = rows[first + 1], rows[first]

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert gates["spectrum_ranks_exact"]
    assert not gates["spectrum_physical_order_exact"]
    with pytest.raises(RuntimeError, match="spectrum_physical_order_exact"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize("row_kind", ("proposal_rows", "line_rows"))
@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (("selected_arm", "B"), ("local_proposal", 99)),
)
def test_audit_rejects_continuation_row_identity_tamper(
    audit_packets: dict[str, Any],
    row_kind: str,
    field: str,
    tampered_value: str | int,
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress[row_kind][-1][field] = tampered_value

    gates = _audit_row_gates(progress, audit_packets["parent"])

    assert not gates["continuation_row_identity_exact"]
    with pytest.raises(RuntimeError, match="continuation_row_identity_exact"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize("mode", ("duplicate", "missing"))
def test_audit_rejects_duplicate_or_missing_state_update(
    audit_packets: dict[str, Any], mode: str
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    if mode == "duplicate":
        progress["state_rows"].append(copy.deepcopy(progress["state_rows"][-1]))
    else:
        progress["state_rows"].pop()

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["state_updates_exact"]
    assert not gates["state_count_exact"]
    with pytest.raises(RuntimeError, match="state_(updates|count)_exact"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_duplicate_spectrum_rank(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress["spectrum_rows"][-1]["rank"] = 510

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["spectrum_ranks_exact"]
    with pytest.raises(RuntimeError, match="spectrum_ranks_exact"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_reinserted_parent_rejected_proposal18(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    rejected = copy.deepcopy(progress["intervention_origin"])
    rejected["phase"] = "relaxed_continuation"
    rejected["legacy_cancellation_gate_pass"] = False
    rejected["relaxed_nonzero_gate_pass"] = True
    progress["proposal_rows"].insert(continuation.START_UPDATE, rejected)

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["terminal_structure_exact"]
    assert not gates["proposal_row_order_exact"]
    with pytest.raises(RuntimeError, match="proposal_row_order_exact"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_nonprefix_alpha(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    alpha = float(continuation.i6.LINE_ALPHAS[1])
    progress["line_rows"][-1]["alpha"] = alpha
    progress["proposal_rows"][-1]["selected_alpha"] = alpha
    chain_hash = continuation._chain_hash(
        continuation.EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256,
        update=continuation.START_UPDATE + 1,
        base_hash=progress["proposal_rows"][-1]["base_parameter_hash"],
        endpoint_hash=progress["proposal_rows"][-1]["endpoint_parameter_hash"],
        alpha=alpha,
    )
    progress["proposal_rows"][-1]["transition_chain_sha256"] = chain_hash
    progress["transition_chain_sha256"] = chain_hash

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["continuation_line_prefix_repeat_pass"]
    with pytest.raises(RuntimeError, match="continuation_line_prefix_repeat_pass"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_repeat_pass_mismatch(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress["line_rows"][-1]["repeat_spectrum_max_abs_error"] = 1e-3

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["continuation_line_prefix_repeat_pass"]
    with pytest.raises(RuntimeError, match="continuation_line_prefix_repeat_pass"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize("target", ("base", "endpoint", "state"))
def test_audit_rejects_parameter_hash_link_tamper(
    audit_packets: dict[str, Any], target: str
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    if target == "base":
        progress["proposal_rows"][-1]["base_parameter_hash"] = "tampered-base"
    elif target == "endpoint":
        progress["line_rows"][-1]["endpoint_parameter_hash"] = "tampered-endpoint"
    else:
        progress["state_rows"][-1]["parameter_hash"] = "tampered-state"

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["continuation_parameter_hash_links_pass"]
    with pytest.raises(RuntimeError, match="continuation_parameter_hash_links_pass"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def test_audit_rejects_parent_frozen_low_link_tamper(
    audit_packets: dict[str, Any],
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    progress["state_rows"][-1]["parent_frozen_low_energy"] = 1.3

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["continuation_parent_low_links_pass"]
    with pytest.raises(RuntimeError, match="continuation_parent_low_links_pass"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize("target", ("proposal", "checkpoint"))
def test_audit_rejects_transition_chain_tamper(
    audit_packets: dict[str, Any], target: str
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    if target == "proposal":
        progress["proposal_rows"][-1]["transition_chain_sha256"] = "tampered-chain"
    else:
        progress["transition_chain_sha256"] = "tampered-chain"

    gates = _audit_row_gates(progress, audit_packets["parent"])
    assert not gates["transition_chain_recomputes"]
    with pytest.raises(RuntimeError, match="transition_chain_recomputes"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize(
    ("target", "expected_gate"),
    (
        ("selected_arm", "progress_selected_arm_is_low"),
        ("new_count", "progress_new_update_count_matches"),
        ("parent_hash", "progress_parent_checkpoint_matches"),
        ("active_state", "progress_active_hash_recomputes"),
    ),
)
def test_progress_audit_rejects_header_and_active_state_tamper(
    audit_packets: dict[str, Any], target: str, expected_gate: str
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    if target == "selected_arm":
        progress["selected_arm"] = "B"
    elif target == "new_count":
        progress["new_accepted_updates"] = 2
    elif target == "parent_hash":
        progress["parent_checkpoint_sha256"] = "tampered-parent"
    else:
        progress["active_model_state"]["decoder.weight"][0] += 1.0

    with pytest.raises(RuntimeError, match=expected_gate):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


@pytest.mark.parametrize("mode", ("invalid_schema", "negative", "nan", "max_mismatch"))
def test_progress_audit_rejects_invalid_parent_replay_error_payload(
    audit_packets: dict[str, Any], mode: str
) -> None:
    progress = copy.deepcopy(audit_packets["continued"])
    preflight = progress["intervention_preflight"]
    errors = preflight["parent_state_metric_errors"]
    key = next(iter(errors))
    if mode == "invalid_schema":
        errors["unexpected_metric"] = 0.0
    elif mode == "negative":
        errors[key] = -1.0
    elif mode == "nan":
        errors[key] = float("nan")
    else:
        preflight["parent_state_metric_max_abs_error"] = 1e-12

    with pytest.raises(RuntimeError, match="progress_preflight_valid"):
        continuation._audit_progress_checkpoint(progress, audit_packets["parent"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _valid_published_recovery(output: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output.mkdir()
    source_snapshot = output / "executed_source_snapshot.py"
    source_snapshot.write_bytes(Path(continuation.__file__).read_bytes())
    expected_normalized = continuation._normalized_source_sha256(source_snapshot)
    monkeypatch.setattr(
        continuation, "EXPECTED_NORMALIZED_SOURCE_SHA256", expected_normalized
    )

    (output / "INCOMPLETE").write_text("incomplete\n", encoding="utf-8")
    final_checkpoint = output / "final_checkpoint.pt"
    final_checkpoint.write_bytes(b"synthetic-cpu-checkpoint")
    outcome = {
        "termination": "backtracking_exhausted",
        "scientific_success": False,
        "immediate_premature_cutoff": True,
        "sustained_continuation": False,
    }
    decision = {
        "protocol_id": continuation.PROTOCOL_ID,
        "valid": True,
        "selected_arm": "low",
        "accepted_updates": continuation.START_UPDATE + 1,
        "new_accepted_updates": 1,
        "termination": outcome["termination"],
        "scientific_success": outcome["scientific_success"],
        "outcome": outcome,
        "final_checkpoint_file_sha256": continuation.sha256_file(final_checkpoint),
    }
    _write_json(output / "decision.json", decision)
    manifest = continuation._artifact_manifest(output, source_snapshot)
    _write_json(output / "artifact_manifest.json", manifest)
    finalized = {
        "protocol_id": continuation.PROTOCOL_ID,
        "status": "complete_awaiting_independent_review",
        "valid": True,
        "accepted_updates": decision["accepted_updates"],
        "new_accepted_updates": decision["new_accepted_updates"],
        "termination": decision["termination"],
        "scientific_success": decision["scientific_success"],
        "immediate_premature_cutoff": outcome["immediate_premature_cutoff"],
        "sustained_continuation": outcome["sustained_continuation"],
        "decision_sha256": continuation.sha256_file(output / "decision.json"),
        "artifact_manifest_sha256": continuation.sha256_file(
            output / "artifact_manifest.json"
        ),
    }
    _write_json(output / "FINALIZED.json", finalized)
    return output


def test_published_recovery_accepts_valid_cpu_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _valid_published_recovery(tmp_path / "published", monkeypatch)

    gates = continuation._validate_published_recovery(output)

    assert gates
    assert all(gates.values())


def test_published_recovery_rejects_missing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _valid_published_recovery(tmp_path / "published", monkeypatch)
    (output / "final_checkpoint.pt").unlink()

    with pytest.raises(RuntimeError, match="recovery files missing"):
        continuation._validate_published_recovery(output)


def test_published_recovery_rejects_tampered_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _valid_published_recovery(tmp_path / "published", monkeypatch)
    with (output / "final_checkpoint.pt").open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(RuntimeError, match="manifest_artifact_hashes_match"):
        continuation._validate_published_recovery(output)


def _synthetic_states(
    final_update: int,
    *,
    final_a: float,
    final_low90: float,
) -> pd.DataFrame:
    updates = [0, continuation.START_UPDATE] + list(
        range(continuation.START_UPDATE + 1, final_update + 1)
    )
    rows: list[dict[str, float | int]] = []
    for update in updates:
        if update == 0:
            progress = 0.0
        elif final_update == continuation.START_UPDATE:
            progress = 1.0
        else:
            progress = (update - continuation.START_UPDATE) / (
                final_update - continuation.START_UPDATE
            )
        if update == 0:
            exact_a = 1.2
            low90 = 0.9
            stage = 0.0
        else:
            exact_a = 1.0 + progress * (final_a - 1.0)
            low90 = 0.86 + progress * (final_low90 - 0.86)
            stage = 0.5 + 0.5 * progress
        rows.append(
            {
                "accepted_update": update,
                "exact_a_per_dim": exact_a,
                "a_low90_abs_per_dim": low90,
                "a_gt1": 0.2 - 0.1 * stage,
                "damped_full_burg_per_dim": 7.0 - stage,
                "m_p50": 0.01 + 0.01 * stage,
                "effective_rank": 10.0 + stage,
                "count_lt_1e_4": int(500 - 10 * stage),
                "count_lt_1e_2": int(505 - 10 * stage),
                "count_lt_0p1": int(510 - 10 * stage),
                "m_max": 4.0 - 0.1 * stage,
                "m_raw_eig_min": 0.0,
                "a_direct_abs_error": 0.0,
                "a_trace_abs_error": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_proposals(
    final_update: int, *, first_accepted: bool = True
) -> pd.DataFrame:
    if final_update == continuation.START_UPDATE:
        updates = [continuation.START_UPDATE + 1]
    else:
        updates = list(range(continuation.START_UPDATE + 1, final_update + 1))
    return pd.DataFrame(
        [
            {
                "target_update": update,
                "accepted": int(first_accepted or index > 0),
                "relaxed_nonzero_gate_pass": True,
                "unit_common_source_norm": 0.4 - 0.001 * index,
                "fp32_fp64_direction_cosine": 0.999999,
                "fp32_fp64_direction_relative_error": 1e-7,
                "common_amplification": 2.5 + 0.01 * index,
                "selected_alpha": (1.0 / 128.0 if first_accepted or index > 0 else 0.0),
                "slope_A": -1.0,
                "slope_B": -1.0,
                "slope_L_low": -1.0,
            }
            for index, update in enumerate(updates)
        ]
    )


def _outcome_tolerances() -> dict[str, float]:
    return {
        "A": 1e-9,
        "B": 1e-9,
        "A_low90": 1e-9,
        "A_gt1": 1e-9,
        "m_p50": 1e-10,
        "effective_rank": 1e-8,
        "m_max": 1e-8,
    }


def _synthetic_spectra(states: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "accepted_update": int(update),
                "rank": 0,
                "m_eigenvalue": 0.5,
                "a_contribution": 0.25,
            }
            for update in states["accepted_update"]
        ]
    )


def _classify_outcome(
    *,
    states: pd.DataFrame,
    proposals: pd.DataFrame,
    accepted_updates: int,
    termination: str,
) -> dict[str, Any]:
    return continuation._outcome(
        states=states,
        spectra=_synthetic_spectra(states),
        proposals=proposals,
        tolerances=_outcome_tolerances(),
        accepted_updates=accepted_updates,
        termination=termination,
        history_summary={"all_historical_a_armijo_pass": True},
    )


def test_outcome_classifies_no_immediate_continuation() -> None:
    states = _synthetic_states(continuation.START_UPDATE, final_a=1.0, final_low90=0.86)
    proposals = _synthetic_proposals(continuation.START_UPDATE, first_accepted=False)

    outcome = _classify_outcome(
        states=states,
        proposals=proposals,
        accepted_updates=continuation.START_UPDATE,
        termination="backtracking_exhausted",
    )

    assert outcome["immediate_premature_cutoff"] is False
    assert outcome["sustained_continuation"] is False
    assert outcome["b_non_descent"] is False
    assert outcome["finite_grid_conflict"] is True
    assert outcome["scalar_only_repair"] is False
    assert outcome["new_accepted_updates"] == 0


def test_outcome_classifies_immediate_but_not_sustained_continuation() -> None:
    final_update = continuation.START_UPDATE + 1
    outcome = _classify_outcome(
        states=_synthetic_states(final_update, final_a=0.98, final_low90=0.855),
        proposals=_synthetic_proposals(final_update),
        accepted_updates=final_update,
        termination="backtracking_exhausted",
    )

    assert outcome["immediate_premature_cutoff"] is True
    assert outcome["sustained_continuation"] is False
    assert outcome["b_non_descent"] is False
    assert outcome["finite_grid_conflict"] is False
    assert outcome["scalar_only_repair"] is True
    assert outcome["new_accepted_updates"] == 1


def test_outcome_classifies_b_non_descent_terminal_mechanism() -> None:
    states = _synthetic_states(continuation.START_UPDATE, final_a=1.0, final_low90=0.86)
    proposals = _synthetic_proposals(continuation.START_UPDATE, first_accepted=False)
    proposals.loc[proposals.index[-1], "slope_B"] = 0.0

    outcome = _classify_outcome(
        states=states,
        proposals=proposals,
        accepted_updates=continuation.START_UPDATE,
        termination="nonnegative_joint_slope",
    )

    assert outcome["b_non_descent"] is True
    assert outcome["finite_grid_exhaustion"] is False
    assert outcome["finite_grid_conflict"] is True
    assert outcome["scalar_only_repair"] is False


def test_outcome_marks_exact_cancellation_amplification_unbounded() -> None:
    states = _synthetic_states(continuation.START_UPDATE, final_a=1.0, final_low90=0.86)
    proposals = _synthetic_proposals(continuation.START_UPDATE, first_accepted=False)
    proposals.loc[proposals.index[-1], "unit_common_source_norm"] = 0.0
    proposals.loc[proposals.index[-1], "common_amplification"] = None

    outcome = _classify_outcome(
        states=states,
        proposals=proposals,
        accepted_updates=continuation.START_UPDATE,
        termination="exact_cancellation",
    )

    diagnostics = outcome["near_cancellation_diagnostics"]
    assert diagnostics["minimum_source_norm"] == 0.0
    assert diagnostics["final_source_norm"] == 0.0
    assert diagnostics["maximum_amplification"] is None
    assert diagnostics["maximum_amplification_unbounded"] is True


@pytest.mark.parametrize(
    ("final_low90", "expected_fraction", "bulk_gate"),
    [
        (0.84, 0.15, False),
        (0.70, 0.50, True),
    ],
)
def test_outcome_classifies_sustained_scalar_only_vs_bulk_continuation(
    final_low90: float,
    expected_fraction: float,
    bulk_gate: bool,
) -> None:
    final_update = continuation.START_UPDATE + continuation.SUSTAINED_NEW_UPDATES
    outcome = _classify_outcome(
        states=_synthetic_states(final_update, final_a=0.8, final_low90=final_low90),
        proposals=_synthetic_proposals(final_update),
        accepted_updates=final_update,
        termination="backtracking_exhausted",
    )

    assert outcome["immediate_premature_cutoff"] is True
    assert outcome["sustained_continuation"] is True
    assert outcome["b_non_descent"] is False
    assert outcome["finite_grid_conflict"] is False
    assert outcome["scalar_only_repair"] is (not bulk_gate)
    assert outcome["new_accepted_updates"] == continuation.SUSTAINED_NEW_UPDATES
    assert outcome["total_non_top_only_fraction"] == pytest.approx(expected_fraction)
    assert outcome["success_gates"]["non_top_only_fraction_at_least_0p25"] is bulk_gate
    if not bulk_gate:
        assert outcome["continuation_a_reduction"] > 0.0
        assert outcome["continuation_non_top_only_fraction"] < 0.25
