from __future__ import annotations

import math

from scripts.run_one_state_exact_a_p32_unit_common import (
    _acceptance_failures,
    _acceptance_tolerances,
    _state_row,
    _tail_state_pass,
)


def _metric_row() -> dict[str, float]:
    return {
        "exact_a_per_dim": 1.2,
        "damped_full_burg_per_dim": 6.0,
        "m_max": 8.0,
        "m_p50": 0.01,
        "m_lt_0p1_fraction": 0.8,
        "m_lt_1e_4_fraction": 0.09,
        "m_lt_0p01_fraction": 0.49,
        "trace_m_per_dim": 0.06,
        "a_low90_abs_per_dim": 0.8,
    }


def test_acceptance_requires_both_components_and_spectral_guards() -> None:
    current = _metric_row()
    tolerances = _acceptance_tolerances(
        {"a": 0.0, "b": 0.0, "m_max": 0.0, "m_p50": 0.0, "m_lt_0p1_fraction": 0.0}
    )
    passing = {**current, "exact_a_per_dim": 1.1, "damped_full_burg_per_dim": 5.9, "m_max": 7.9}
    assert _acceptance_failures(current, passing, tolerances) == []

    bad = {
        **passing,
        "damped_full_burg_per_dim": 6.1,
        "m_max": 8.1,
        "m_p50": 0.009,
        "m_lt_0p1_fraction": 0.81,
    }
    assert set(_acceptance_failures(current, bad, tolerances)) == {
        "full_b_not_lower",
        "m_max_increased",
        "m_p50_decreased",
        "low_fraction_increased",
    }


def test_state_row_lower90_ratio_uses_cumulative_a_reduction() -> None:
    initial = _metric_row()
    current = {**initial, "exact_a_per_dim": 0.8, "a_low90_abs_per_dim": 0.7}
    row = _state_row(proposal=5, accepted=True, alpha=0.5, metrics=current, initial=initial)
    assert math.isclose(float(row["rho_lower90"]), 0.25)
    assert math.isclose(float(row["accepted_radius"]), 0.5 * 0.04892722657548397)


def test_tail_gate_counts_modes_not_only_fractions() -> None:
    passing = _metric_row()
    passing["rho_lower90"] = 0.3
    assert _tail_state_pass(passing)

    failing = {**passing, "m_lt_1e_4_fraction": 52.0 / 512.0}
    assert not _tail_state_pass(failing)
