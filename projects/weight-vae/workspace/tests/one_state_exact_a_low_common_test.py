from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts import audit_one_state_exact_a_low_common as audit


def test_blocked_two_gradients_reuses_one_hvp_graph_for_both_cotangents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = [
        torch.nn.Parameter(torch.tensor(2.0)),
        torch.nn.Parameter(torch.tensor(3.0)),
    ]
    z = torch.zeros(2)

    def fake_hvp(*args: object, **kwargs: object) -> torch.Tensor:
        basis = args[4]
        assert isinstance(basis, torch.Tensor)
        p0, p1 = active
        return torch.stack(
            [p0 * basis[0] + p1 * basis[1], p1 * basis[0] + p0 * basis[1]]
        )

    monkeypatch.setattr(audit, "_task_set_for_record", lambda *args: None)
    monkeypatch.setattr(audit, "_latent_hvp", fake_hvp)
    monkeypatch.setattr(audit.torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(audit.torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(audit.torch.cuda, "max_memory_allocated", lambda *args: 0)
    run = SimpleNamespace(task_tensors=None, vae=None, normalizer=None, spec=None)

    gradient_a, gradient_low, metadata = audit._blocked_two_gradients(
        cfg=SimpleNamespace(),
        run=run,
        z=z,
        record={},
        active=active,
        k_a=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        k_low=torch.tensor([[-1.0, 1.0], [2.0, -2.0]]),
    )

    assert [float(value) for value in gradient_a] == pytest.approx([5.0, 5.0])
    assert [float(value) for value in gradient_low] == pytest.approx([-3.0, 3.0])
    assert all(value.dtype == torch.float64 for value in gradient_a + gradient_low)
    assert metadata["basis_count"] == 2
    assert metadata["a_unused_parameter_tensors_all_blocks"] == 0
    assert metadata["low_unused_parameter_tensors_all_blocks"] == 0
    assert metadata["retained_memory_by_block_bytes"] == [0]
    assert metadata["peak_memory_by_block_bytes"] == [0]


def test_projected_metrics_match_known_diagonal_spectrum_and_ranks() -> None:
    matrix = torch.diag(torch.tensor([100.0, 1.0, 4.0, 9.0], dtype=torch.float64))
    low_basis = torch.eye(4, dtype=torch.float64)[:, [1, 2, 3]]

    metrics, eigenvalues = audit._projected_metrics(matrix, low_basis)

    assert torch.equal(eigenvalues, torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64))
    assert metrics["projected_low_raw_min"] == pytest.approx(1.0)
    assert metrics["projected_low_mean"] == pytest.approx(14.0 / 3.0)
    assert metrics["projected_low_max"] == pytest.approx(9.0)
    assert metrics["projected_low_p01"] == pytest.approx(1.06)
    assert metrics["projected_low_p10"] == pytest.approx(1.6)
    assert metrics["projected_low_p25"] == pytest.approx(2.5)
    assert metrics["projected_low_p50"] == pytest.approx(4.0)
    assert metrics["projected_low_p75"] == pytest.approx(6.5)
    assert metrics["projected_low_p90"] == pytest.approx(8.0)
    assert metrics["projected_low_p95"] == pytest.approx(8.5)
    assert metrics["projected_low_p99"] == pytest.approx(8.9)
    assert metrics["projected_low_participation_rank"] == pytest.approx(2.0)

    probabilities = (1.0 / 14.0, 4.0 / 14.0, 9.0 / 14.0)
    entropy_rank = math.exp(-sum(value * math.log(value) for value in probabilities))
    assert metrics["projected_low_entropy_rank"] == pytest.approx(entropy_rank)


def test_projected_metrics_report_raw_negative_eigenvalue_but_clip_spectrum() -> None:
    matrix = torch.diag(torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64))

    metrics, eigenvalues = audit._projected_metrics(
        matrix, torch.eye(3, dtype=torch.float64)
    )

    assert torch.equal(eigenvalues, torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64))
    assert metrics["projected_low_raw_min"] == pytest.approx(-2.0)
    assert metrics["projected_low_mean"] == pytest.approx(1.0)
    assert metrics["projected_low_participation_rank"] == pytest.approx(1.0)
    assert metrics["projected_low_entropy_rank"] == pytest.approx(1.0)


def _valid_parameter_stats() -> dict[str, float]:
    return {
        "effective_direction_cosine": 0.995,
        "effective_direction_norm_ratio": 1.0,
        "midpoint_drift_relative": 0.01,
    }


def test_chain_row_passes_consistent_negative_realized_chain() -> None:
    row = audit._chain_row(
        radius=1.0 / 512.0,
        objective="exact_a",
        nominal=-2.0,
        s_theta=-1.98,
        s_h=-1.96,
        direct=-1.94,
        parameter_stats=_valid_parameter_stats(),
    )

    assert row["radius"] == pytest.approx(1.0 / 512.0)
    assert row["objective"] == "exact_a"
    assert row["expected_signs"] is True
    assert row["transitions_pass"] is True
    assert row["realization_pass"] is True
    assert row["row_pass"] is True


@pytest.mark.parametrize(
    ("values", "parameter_overrides", "failed_gate"),
    [
        ((0.0, 0.0, 0.0, 0.0), {}, "expected_signs"),
        ((-1.0, -1.0, -1.0, -0.8), {}, "transitions_pass"),
        (
            (-1.0, -1.0, -1.0, -1.0),
            {"effective_direction_cosine": audit.EFFECTIVE_COSINE_MIN - 0.001},
            "realization_pass",
        ),
    ],
    ids=["sign", "transition", "realization"],
)
def test_chain_row_fails_when_one_required_gate_fails(
    values: tuple[float, float, float, float],
    parameter_overrides: dict[str, float],
    failed_gate: str,
) -> None:
    parameter_stats = {**_valid_parameter_stats(), **parameter_overrides}
    row = audit._chain_row(
        radius=1.0 / 1024.0,
        objective="low_energy",
        nominal=values[0],
        s_theta=values[1],
        s_h=values[2],
        direct=values[3],
        parameter_stats=parameter_stats,
    )

    gate_names = {"expected_signs", "transitions_pass", "realization_pass"}
    assert row[failed_gate] is False
    assert all(row[name] is True for name in gate_names - {failed_gate})
    assert row["row_pass"] is False


def test_metric_tolerances_use_five_times_max_repeat_error_or_floor() -> None:
    observed_maxima = {
        "exact_a_per_dim": 4e-10,
        "damped_full_burg_per_dim": 1e-12,
        "frozen_low_energy": 4e-11,
        "m_max": 3e-9,
        "m_p50": 1e-12,
        "effective_rank": 4e-9,
        "projected_low_p50": 4e-13,
        "projected_low_participation_rank": 1e-10,
        "projected_low_entropy_rank": 4e-9,
    }
    endpoints = pd.DataFrame(
        [
            {
                f"repeat_{key}_abs_error": value
                for key, value in observed_maxima.items()
            },
            {
                f"repeat_{key}_abs_error": value / 2.0
                for key, value in observed_maxima.items()
            },
        ]
    )

    assert audit._metric_tolerances(endpoints) == pytest.approx(
        {
            "exact_a_per_dim": 2e-9,
            "damped_full_burg_per_dim": 1e-9,
            "frozen_low_energy": 2e-10,
            "m_max": 1.5e-8,
            "m_p50": 1e-10,
            "effective_rank": 2e-8,
            "projected_low_p50": 2e-12,
            "projected_low_participation_rank": 1e-8,
            "projected_low_entropy_rank": 2e-8,
        }
    )


def _base_endpoint() -> dict[str, float | str]:
    return {
        "case": "base",
        "alpha": 0.0,
        "exact_a_per_dim": 10.0,
        "damped_full_burg_per_dim": 20.0,
        "frozen_low_energy": 1.0,
        "m_max": 5.0,
        "m_p50": 2.0,
        "effective_rank": 8.0,
        "projected_low_p50": 1.5,
        "projected_low_participation_rank": 6.0,
        "projected_low_entropy_rank": 7.0,
        "count_lt_1e_4": 3.0,
        "count_lt_1e_2": 5.0,
        "m_raw_eig_min": 0.0,
        "projected_low_raw_min": 0.0,
    }


def _passing_candidate(
    *, alpha: float, case: str = "candidate"
) -> dict[str, float | str]:
    return {
        "case": case,
        "alpha": alpha,
        "exact_a_per_dim": 9.0,
        "damped_full_burg_per_dim": 19.0,
        "frozen_low_energy": 2.0,
        "m_max": 5.5,
        "m_p50": 3.0,
        "effective_rank": 7.5,
        "projected_low_p50": 2.5,
        "projected_low_participation_rank": 5.5,
        "projected_low_entropy_rank": 6.5,
        "count_lt_1e_4": 3.0,
        "count_lt_1e_2": 5.0,
        "m_raw_eig_min": -1e-8,
        "projected_low_raw_min": -1e-8,
    }


def _candidate_tolerances() -> dict[str, float]:
    return {metric: 0.5 for metric in audit.METRIC_FLOORS}


def test_candidate_rows_enforces_alpha_eligibility_and_accepts_noncollapse_boundaries(
) -> None:
    endpoints = pd.DataFrame(
        [
            _base_endpoint(),
            _passing_candidate(alpha=1.0 / 8.0 - 1e-6, case="below-threshold"),
            _passing_candidate(alpha=1.0 / 8.0, case="at-threshold"),
            _passing_candidate(alpha=1.0, case="above-threshold"),
        ]
    )

    result = audit._candidate_rows(endpoints, _candidate_tolerances())

    assert result["case"].tolist() == ["at-threshold", "above-threshold"]
    assert result["candidate_pass"].tolist() == [True, True]


@pytest.mark.parametrize(
    ("column", "failing_value"),
    [
        ("exact_a_per_dim", 9.5),
        ("damped_full_burg_per_dim", 19.5),
        ("frozen_low_energy", 1.5),
        ("m_max", 5.5001),
        ("m_p50", 2.5),
        ("projected_low_p50", 2.0),
        ("effective_rank", 7.4999),
        ("projected_low_participation_rank", 5.4999),
        ("projected_low_entropy_rank", 6.4999),
        ("count_lt_1e_4", 4.0),
        ("count_lt_1e_2", 6.0),
        ("m_raw_eig_min", -1.0001e-8),
        ("projected_low_raw_min", -1.0001e-8),
    ],
)
def test_candidate_rows_rejects_each_required_gate(
    column: str, failing_value: float
) -> None:
    candidate = _passing_candidate(alpha=1.0 / 8.0)
    candidate[column] = failing_value
    endpoints = pd.DataFrame([_base_endpoint(), candidate])

    result = audit._candidate_rows(endpoints, _candidate_tolerances())

    assert len(result) == 1
    assert result.iloc[0]["case"] == "candidate"
    assert not bool(result.iloc[0]["candidate_pass"])
