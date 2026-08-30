from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts import run_one_state_exact_selected_trajectory as trajectory


def test_blocked_three_gradients_share_hvp_graph_and_report_fp64_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = [
        torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor(-1.5, dtype=torch.float32)),
    ]
    z = torch.zeros(3, dtype=torch.float32)
    coefficient_0 = torch.tensor(
        [[1.0, 0.0, 2.0], [-1.0, 3.0, 1.0], [2.0, 2.0, -2.0]]
    )
    coefficient_1 = torch.tensor(
        [[0.0, 1.0, -1.0], [4.0, -2.0, 0.0], [1.0, 3.0, 2.0]]
    )
    cotangents = {
        "A": torch.tensor(
            [[1.0, 2.0, 0.0], [0.0, -1.0, 3.0], [2.0, 1.0, -2.0]],
            dtype=torch.float64,
        ),
        "B": torch.tensor(
            [[-1.0, 0.0, 2.0], [3.0, 1.0, 0.0], [0.0, -2.0, 1.0]],
            dtype=torch.float64,
        ),
        "low": torch.tensor(
            [[2.0, -1.0, 1.0], [-2.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float64,
        ),
    }
    task_set = object()
    hvp_rows: list[int] = []
    reset_devices: list[torch.device] = []

    monkeypatch.setattr(trajectory, "BLOCK_SIZE", 2)
    monkeypatch.setattr(
        trajectory, "_task_set_for_record", lambda *_args, **_kwargs: task_set
    )

    def fake_hvp(*args: object, **kwargs: object) -> torch.Tensor:
        basis = args[4]
        assert isinstance(basis, torch.Tensor)
        row = int(torch.argmax(basis).item())
        hvp_rows.append(row)
        assert kwargs["task_set"] is task_set
        assert kwargs["tau"] == pytest.approx(2.5)
        assert kwargs["batch_indices"] is None
        return active[0] * coefficient_0[row] + active[1] * coefficient_1[row]

    retained = iter([100, 112])
    peaks = iter([200, 232])
    monkeypatch.setattr(trajectory, "_latent_hvp", fake_hvp)
    monkeypatch.setattr(
        trajectory.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_devices.append(device),
    )
    monkeypatch.setattr(
        trajectory.torch.cuda, "memory_allocated", lambda _device: next(retained)
    )
    monkeypatch.setattr(
        trajectory.torch.cuda, "max_memory_allocated", lambda _device: next(peaks)
    )

    gradients, metadata = trajectory._blocked_three_gradients(
        cfg=SimpleNamespace(),
        run=SimpleNamespace(task_tensors={}, vae=None, normalizer=None, spec=None),
        z=z,
        record={"tau": 2.5},
        active=active,
        cotangents=cotangents,
        update=7,
    )

    assert hvp_rows == [0, 1, 2]
    assert reset_devices == [z.device, z.device]
    for name, cotangent in cotangents.items():
        expected = [
            torch.sum(cotangent.float() * coefficient).double()
            for coefficient in (coefficient_0, coefficient_1)
        ]
        assert len(gradients[name]) == 2
        for observed, reference in zip(gradients[name], expected, strict=True):
            torch.testing.assert_close(observed, reference, rtol=0.0, atol=0.0)
            assert observed.dtype == torch.float64
        assert metadata[f"{name}_unused_parameter_tensors_all_blocks"] == 0
        assert metadata[f"{name}_all_accumulators_float64"] is True

    assert metadata["basis_count"] == 3
    assert metadata["block_count"] == 2
    assert metadata["block_size"] == 2
    assert metadata["retained_memory_by_block_bytes"] == [100, 112]
    assert metadata["peak_memory_by_block_bytes"] == [200, 232]
    assert metadata["retained_memory_range_bytes"] == 12
    assert metadata["block_peak_range_bytes"] == 32
    assert metadata["memory_gate_pass"] is True
    assert metadata["elapsed_sec"] >= 0.0


def test_unit_direction_normalizes_components_before_combining_and_targets_radius() -> None:
    active = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))]
    direction, metadata = trajectory._unit_direction(
        [torch.tensor([3.0, 0.0])],
        [torch.tensor([0.0, 4.0])],
        active,
    )
    rescaled_direction, _ = trajectory._unit_direction(
        [torch.tensor([300.0, 0.0])],
        [torch.tensor([0.0, 0.25])],
        active,
    )

    assert direction is not None
    assert rescaled_direction is not None
    torch.testing.assert_close(direction[0], rescaled_direction[0], rtol=1e-6, atol=1e-8)
    expected_component = -trajectory.TARGET_NORM / (2.0**0.5)
    torch.testing.assert_close(
        direction[0],
        torch.tensor([expected_component, expected_component], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-8,
    )
    assert metadata["gradient_a_norm"] == pytest.approx(3.0)
    assert metadata["gradient_x_norm"] == pytest.approx(4.0)
    assert metadata["gradient_cosine"] == pytest.approx(0.0)
    assert metadata["unit_common_source_norm"] == pytest.approx(2.0**0.5)
    assert metadata["cancellation_gate_pass"] is True
    assert metadata["direction_norm"] == pytest.approx(trajectory.TARGET_NORM, rel=1e-6)


def test_unit_direction_rejects_cancelled_unit_components() -> None:
    active = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))]

    direction, metadata = trajectory._unit_direction(
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 0.0])],
        active,
    )

    assert direction is None
    assert metadata["gradient_cosine"] == pytest.approx(-1.0)
    assert metadata["unit_common_source_norm"] == pytest.approx(0.0)
    assert metadata["cancellation_gate_pass"] is False
    assert metadata["direction_norm"] == 0.0


def _gate_metrics() -> dict[str, float | str]:
    return {
        "exact_a_per_dim": 10.0,
        "damped_full_burg_per_dim": 20.0,
        "frozen_low_energy": 1.0,
        "a_gt1": 2.0,
        "a_low90_abs_per_dim": 4.0,
        "a_direct_abs_error": 0.0,
        "a_trace_abs_error": 0.0,
        "m_raw_eig_min": 0.0,
        "low_basis_hash": "not-a-numeric-metric",
    }


def _passing_candidate() -> dict[str, float | str]:
    return {
        **_gate_metrics(),
        "exact_a_per_dim": 9.99,
        "damped_full_burg_per_dim": 19.99,
        "frozen_low_energy": 1.01,
        "a_low90_abs_per_dim": 3.0,
    }


def _gate_tolerances() -> dict[str, float]:
    return {
        "A": 1e-9,
        "B": 1e-9,
        "L_low": 1e-9,
        "A_gt1": 1e-9,
        "A_low90": 1e-9,
    }


@pytest.mark.parametrize(
    ("target", "metric", "metric_sign"),
    [
        ("A", "exact_a_per_dim", -1.0),
        ("B", "damped_full_burg_per_dim", -1.0),
        ("L_low", "frozen_low_energy", 1.0),
    ],
)
def test_candidate_gates_keep_component_armijo_and_slope_signs_independent(
    target: str,
    metric: str,
    metric_sign: float,
) -> None:
    current = _gate_metrics()
    candidate = _passing_candidate()
    alpha = 0.5
    slopes = {"A": -1.0, "B": -1.0, "L_low": -1.0}
    insufficient_reduction = trajectory.ARMIJO_C1 * alpha / 2.0
    candidate[metric] = float(current[metric]) + metric_sign * insufficient_reduction

    gates = trajectory._candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=alpha,
        tolerances=_gate_tolerances(),
        require_low90=False,
    )

    for name in ("A", "B", "L_low"):
        assert gates[f"{name}_slope_negative"] is True
        assert gates[f"{name}_armijo"] is (name != target)
        assert gates[f"{name}_actual_decrease"] is True

    nonnegative_target_slopes = {**slopes, target: 0.0}
    sign_gates = trajectory._candidate_gates(
        current=current,
        candidate=_passing_candidate(),
        slopes=nonnegative_target_slopes,
        alpha=alpha,
        tolerances=_gate_tolerances(),
        require_low90=False,
    )
    for name in ("A", "B", "L_low"):
        assert sign_gates[f"{name}_slope_negative"] is (name != target)


@pytest.mark.parametrize(
    ("target", "metric", "boundary_value", "passing_value"),
    [
        ("A", "exact_a_per_dim", 9.75, 9.749),
        ("B", "damped_full_burg_per_dim", 19.5, 19.499),
        ("L_low", "frozen_low_energy", 1.75, 1.751),
    ],
)
def test_candidate_actual_decrease_is_strictly_greater_than_each_floor(
    target: str,
    metric: str,
    boundary_value: float,
    passing_value: float,
) -> None:
    current = _gate_metrics()
    candidate = {
        **_passing_candidate(),
        "exact_a_per_dim": 9.5,
        "damped_full_burg_per_dim": 19.0,
        "frozen_low_energy": 2.0,
        metric: boundary_value,
    }
    tolerances = {
        **_gate_tolerances(),
        "A": 0.25,
        "B": 0.5,
        "L_low": 0.75,
    }
    slopes = {"A": -1.0, "B": -1.0, "L_low": -1.0}

    boundary = trajectory._candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=0.5,
        tolerances=tolerances,
        require_low90=False,
    )
    assert boundary[f"{target}_armijo"] is True
    assert boundary[f"{target}_actual_decrease"] is False
    assert all(
        boundary[f"{name}_actual_decrease"] is True
        for name in ("A", "B", "L_low")
        if name != target
    )

    candidate[metric] = passing_value
    passing = trajectory._candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=0.5,
        tolerances=tolerances,
        require_low90=False,
    )
    assert passing[f"{target}_actual_decrease"] is True


def test_candidate_tail_boundary_and_selection_only_low90_gate() -> None:
    current = _gate_metrics()
    candidate = _passing_candidate()
    tolerances = {
        **_gate_tolerances(),
        "A_gt1": 0.125,
        "A_low90": 0.25,
    }
    slopes = {"A": -1.0, "B": -1.0, "L_low": -1.0}
    candidate["a_gt1"] = float(current["a_gt1"]) + tolerances["A_gt1"]
    candidate["a_low90_abs_per_dim"] = 10.0

    proposal_gates = trajectory._candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=0.5,
        tolerances=tolerances,
        require_low90=False,
    )
    selection_gates = trajectory._candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=0.5,
        tolerances=tolerances,
        require_low90=True,
    )
    assert proposal_gates["high_tail_nonincrease"] is True
    assert proposal_gates["low90_decreases"] is True
    assert selection_gates["low90_decreases"] is False

    candidate["a_low90_abs_per_dim"] = (
        float(current["a_low90_abs_per_dim"]) - tolerances["A_low90"]
    )
    at_low90_floor = trajectory._candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=0.5,
        tolerances=tolerances,
        require_low90=True,
    )
    assert at_low90_floor["low90_decreases"] is False

    candidate["a_low90_abs_per_dim"] = float(candidate["a_low90_abs_per_dim"]) - 0.001
    candidate["a_gt1"] = float(candidate["a_gt1"]) + 0.001
    beyond_boundaries = trajectory._candidate_gates(
        current=current,
        candidate=candidate,
        slopes=slopes,
        alpha=0.5,
        tolerances=tolerances,
        require_low90=True,
    )
    assert beyond_boundaries["low90_decreases"] is True
    assert beyond_boundaries["high_tail_nonincrease"] is False


def test_tolerances_use_five_repeat_errors_or_metric_specific_floors() -> None:
    repeat_errors = {
        "repeat_exact_a_per_dim_abs_error": 4e-10,
        "repeat_damped_full_burg_per_dim_abs_error": 1e-12,
        "repeat_frozen_low_energy_abs_error": 4e-11,
        "repeat_a_low90_abs_per_dim_abs_error": 1e-12,
        "repeat_a_gt1_abs_error": 3e-10,
        "repeat_m_max_abs_error": 3e-9,
        "repeat_m_p50_abs_error": 1e-12,
        "repeat_effective_rank_abs_error": 4e-9,
    }

    assert trajectory._tolerances(repeat_errors) == pytest.approx(
        {
            "A": 2e-9,
            "B": 1e-9,
            "L_low": 2e-10,
            "A_low90": 1e-9,
            "A_gt1": 1.5e-9,
            "m_max": 1.5e-8,
            "m_p50": 1e-10,
            "effective_rank": 2e-8,
        }
    )


def test_repeat_gate_uses_metric_tolerances_but_exact_counts_and_spectrum() -> None:
    tolerances = {
        "A": 1e-8,
        "B": 2e-8,
        "L_low": 3e-8,
        "A_low90": 4e-8,
        "A_gt1": 5e-8,
        "m_max": 6e-8,
        "m_p50": 7e-8,
        "effective_rank": 8e-8,
    }
    errors = {
        "repeat_exact_a_per_dim_abs_error": 5e-9,
        "repeat_damped_full_burg_per_dim_abs_error": 1e-8,
        "repeat_frozen_low_energy_abs_error": 1e-8,
        "repeat_a_low90_abs_per_dim_abs_error": 1e-8,
        "repeat_a_gt1_abs_error": 1e-8,
        "repeat_m_max_abs_error": 1e-8,
        "repeat_m_p50_abs_error": 1e-8,
        "repeat_effective_rank_abs_error": 1e-8,
        "repeat_count_lt_1e_4_abs_error": 0.0,
        "repeat_count_lt_1e_2_abs_error": 0.0,
        "repeat_count_lt_0p1_abs_error": 0.0,
        "repeat_parameter_hash_unchanged": 1.0,
        "repeat_spectrum_max_abs_error": 0.0,
        "repeat_hessian_max_abs_error": 0.0,
        "repeat_task_loss_abs_error": 0.0,
    }
    assert trajectory._repeat_passes(errors, tolerances)
    assert not trajectory._repeat_passes(
        {**errors, "repeat_spectrum_max_abs_error": 2e-10}, tolerances
    )
    assert not trajectory._repeat_passes(
        {**errors, "repeat_count_lt_1e_2_abs_error": 1.0}, tolerances
    )


def test_historical_transition_audit_replays_armijo_line_and_commit() -> None:
    state_rows = [
        {
            "accepted_update": 0,
            "exact_a_per_dim": 10.0,
            "damped_full_burg_per_dim": 20.0,
            "a_gt1": 2.0,
        },
        {
            "accepted_update": 1,
            "exact_a_per_dim": 9.9,
            "damped_full_burg_per_dim": 19.9,
            "a_gt1": 1.9,
        },
    ]
    proposal_rows = [
        {
            "target_update": 1,
            "accepted": 1,
            "selected_alpha": 0.5,
            "slope_A": -1.0,
            "slope_B": -1.0,
            "slope_L_low": -1.0,
            "current_A": 10.0,
            "current_B": 20.0,
            "current_low_energy": 1.0,
            "current_a_gt1": 2.0,
            "candidate_A": 9.9,
            "candidate_B": 19.9,
            "candidate_low_energy": 1.1,
            "candidate_a_gt1": 1.9,
        }
    ]
    line_rows = [
        {"target_update": 1, "alpha": 1.0, "passes": 0, "gate_A_armijo": 0},
        {"target_update": 1, "alpha": 0.5, "passes": 1, "gate_A_armijo": 1},
    ]
    tolerances = {"A": 1e-9, "B": 1e-9, "L_low": 1e-9, "A_gt1": 1e-9}

    summary, rows = trajectory._historical_transition_audit(
        state_rows=state_rows,
        proposal_rows=proposal_rows,
        line_rows=line_rows,
        tolerances=tolerances,
    )

    assert all(summary.values())
    assert rows[0]["transition_passes"] == 1
    broken_summary, broken_rows = trajectory._historical_transition_audit(
        state_rows=state_rows,
        proposal_rows=[{**proposal_rows[0], "candidate_A": 9.99999}],
        line_rows=line_rows,
        tolerances=tolerances,
    )
    assert broken_summary["all_historical_transitions_pass"] is False
    assert broken_rows[0]["transition_passes"] == 0


def test_state_and_spectrum_rows_preserve_update_hash_order_and_contributions() -> None:
    metrics = {"exact_a_per_dim": 1.25, "low_basis_hash": "basis-hash"}

    state = trajectory._state_row(7, metrics, "parameter-hash")
    spectrum = trajectory._spectrum_rows(
        7,
        torch.tensor([0.0, 0.25, 2.0], dtype=torch.float64, requires_grad=True),
    )

    assert state == {
        "accepted_update": 7,
        "parameter_hash": "parameter-hash",
        **metrics,
    }
    assert spectrum == [
        {"accepted_update": 7, "rank": 0, "m_eigenvalue": 0.0, "a_contribution": 1.0},
        {
            "accepted_update": 7,
            "rank": 1,
            "m_eigenvalue": 0.25,
            "a_contribution": 0.5625,
        },
        {"accepted_update": 7, "rank": 2, "m_eigenvalue": 2.0, "a_contribution": 1.0},
    ]


def test_sparse_diagnostic_rows_are_finite_without_dataframe_nan_artifacts() -> None:
    assert trajectory._row_numeric_values_finite(
        [{"failure": "cancellation", "selected_alpha": 0.0}, {"candidate_A": 1.0}]
    )
    assert not trajectory._row_numeric_values_finite([{"candidate_A": float("nan")}])


def test_progress_checkpoint_and_tables_roundtrip_through_atomic_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_names = ["decoder.weight", "decoder.bias"]
    active = [
        torch.nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64)),
    ]
    state_rows = [
        {"accepted_update": 0, "parameter_hash": "initial", "exact_a_per_dim": 1.25}
    ]
    spectrum_rows = [
        {"accepted_update": 0, "rank": 0, "m_eigenvalue": 0.5, "a_contribution": 0.25}
    ]
    proposal_rows = [{"proposal": 1, "accepted": True}]
    line_rows = [{"proposal": 1, "arm": "B", "alpha": 0.5}]
    selection_rows = [{"accepted_update": 1, "selected_arm": "B"}]
    initial_metrics = {"exact_a_per_dim": 1.25}
    tolerances = {"A": 1e-9, "B": 2e-9}
    parameter_hash = trajectory._named_tensor_hash(active_names, active)
    real_replace = trajectory.os.replace
    replacements: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        replacements.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(trajectory.os, "replace", recording_replace)

    trajectory._write_progress_checkpoint(
        staging=tmp_path,
        active_names=active_names,
        active=active,
        accepted_updates=1,
        selected_arm="B",
        state_rows=state_rows,
        spectrum_rows=spectrum_rows,
        proposal_rows=proposal_rows,
        line_rows=line_rows,
        selection_rows=selection_rows,
        initial_metrics=initial_metrics,
        tolerances=tolerances,
        terminal=False,
        termination="running",
        proposal0_payload={"selected_arm": "B", "gradient_metadata": {"basis_count": 512}},
    )

    expected_files = [
        "progress_checkpoint.pt",
        "state_metrics.csv",
        "state_spectra.csv",
        "proposal_diagnostics.csv",
        "line_search.csv",
        "arm_selection.csv",
    ]
    assert [destination.name for _, destination in replacements] == expected_files
    assert all(
        source == destination.with_suffix(destination.suffix + ".tmp")
        for source, destination in replacements
    )
    assert not list(tmp_path.glob("*.tmp"))

    checkpoint = torch.load(
        tmp_path / "progress_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["protocol_id"] == trajectory.PROTOCOL_ID
    assert checkpoint["normalized_source_sha256"] == trajectory.EXPECTED_NORMALIZED_SOURCE_SHA256
    assert checkpoint["dependency_manifest_sha256"] == (
        trajectory.EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256
    )
    assert checkpoint["accepted_updates"] == 1
    assert checkpoint["selected_arm"] == "B"
    assert checkpoint["terminal"] is False
    assert checkpoint["termination"] == "running"
    assert checkpoint["proposal0_payload"] == {
        "selected_arm": "B",
        "gradient_metadata": {"basis_count": 512},
    }
    assert checkpoint["active_parameter_hash"] == parameter_hash
    assert checkpoint["state_rows"] == state_rows
    assert checkpoint["spectrum_rows"] == spectrum_rows
    assert checkpoint["proposal_rows"] == proposal_rows
    assert checkpoint["line_rows"] == line_rows
    assert checkpoint["selection_rows"] == selection_rows
    assert checkpoint["initial_metrics"] == initial_metrics
    assert checkpoint["tolerances"] == tolerances
    for name, parameter in zip(active_names, active, strict=True):
        saved = checkpoint["active_model_state"][name]
        torch.testing.assert_close(saved, parameter.detach().cpu())
        assert saved.device.type == "cpu"
        assert saved.requires_grad is False

    table_rows = {
        "state_metrics.csv": state_rows,
        "state_spectra.csv": spectrum_rows,
        "proposal_diagnostics.csv": proposal_rows,
        "line_search.csv": line_rows,
        "arm_selection.csv": selection_rows,
    }
    for filename, rows in table_rows.items():
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / filename),
            pd.DataFrame(rows),
            check_dtype=False,
        )
