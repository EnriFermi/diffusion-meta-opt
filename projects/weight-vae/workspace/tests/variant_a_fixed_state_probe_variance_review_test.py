from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from scripts.review_variant_a_fixed_state_probe_variance import (
    ReviewError,
    _validate_nonalignment,
    primary_mechanism_decision,
    recompute_bootstrap_tables,
    recompute_nonalignment_tables,
    require_exact_key_grid,
    summarize_bridge_diagnostics,
    summarize_directional_components,
    validate_bootstrap_contract,
)


def test_exact_key_grid_rejects_duplicate_or_missing_rows() -> None:
    expected = set(itertools.product(range(2), ("raw", "unit")))
    frame = pd.DataFrame(expected, columns=["bootstrap", "vector_kind"])
    assert require_exact_key_grid(
        frame, ("bootstrap", "vector_kind"), expected, "synthetic.csv"
    ) == {"rows": 4, "unique_keys": 4}

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ReviewError, match="duplicate keys"):
        require_exact_key_grid(
            duplicate, ("bootstrap", "vector_kind"), expected, "synthetic.csv"
        )

    with pytest.raises(ReviewError, match="key grid mismatch"):
        require_exact_key_grid(
            frame.iloc[:-1], ("bootstrap", "vector_kind"), expected, "synthetic.csv"
        )


def test_directional_summary_uses_fixed_panel_percentiles_and_decision_rule() -> None:
    delta = pd.DataFrame(
        [
            {
                "scope": "primary_mean_tasks",
                "vector_kind": "unit",
                "V_probe": 0.4,
                "E_state_within": 0.1,
                "E_task_panel": 0.05,
                "V_state_total": 0.15,
                "Delta_total": 0.25,
            }
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "vector_kind": "unit",
                "V_probe": value + 0.15,
                "E_state_within": 0.1,
                "E_task_panel": 0.05,
                "V_state_total": 0.15,
                "Delta_total": value,
            }
            for value in (0.10, 0.20, 0.30, 0.40)
        ]
    )

    summary = summarize_directional_components(delta, bootstrap).set_index("component")
    assert summary.loc["Delta_total", "point"] == 0.25
    assert summary.loc["Delta_total", "ci95_low"] == pytest.approx(
        np.quantile([0.10, 0.20, 0.30, 0.40], 0.025)
    )
    assert summary.loc["Delta_total", "ci95_high"] == pytest.approx(
        np.quantile([0.10, 0.20, 0.30, 0.40], 0.975)
    )
    assert (
        primary_mechanism_decision(
            delta_dir_ci_low=0.10,
            delta_dir_ci_high=0.40,
            within_state_cosine_median_ci_high=0.79,
            state_nonalignment_gate_passed=False,
        )
        == "probe-direction-major"
    )


def test_bridge_summary_is_task_scoped_and_uses_no_epsilon_scalar_error() -> None:
    bridge = pd.DataFrame(
        [
            {
                "state_position": position,
                "task_name": task,
                "gradient_cosine_b128_to_full": cosine,
                "gradient_relative_error_b128_to_full": 1.0 - cosine,
                "gradient_norm_ratio_b128_to_full": ratio,
                "a_scalar_b128": left,
                "a_scalar_full": right,
            }
            for position, task, cosine, ratio, left, right in (
                (0, "fashion_mnist", 0.9, 0.8, 0.0, 0.0),
                (4, "fashion_mnist", 0.7, 1.2, 1e-300, 0.0),
                (16, "mnist", 0.8, 0.9, 2.0, 2.0),
                (20, "mnist", 0.6, 1.1, 1.0, 3.0),
            )
        ]
    )

    summary = summarize_bridge_diagnostics(bridge).set_index("scope")
    assert summary.loc["fashion_mnist", "state_count"] == 2
    assert summary.loc["fashion_mnist", "gradient_cosine_median"] == pytest.approx(0.8)
    assert summary.loc["fashion_mnist", "scalar_symmetric_error_median"] == pytest.approx(1.0)
    assert summary.loc["all_bridge_states", "paired_draw_count"] == 4


def test_bootstrap_review_rejects_algebraically_consistent_tampering() -> None:
    identity = np.eye(16 * 64, dtype=np.float64)
    task_grams = {
        "fashion_mnist": 2.0 * identity,
        "mnist": identity,
    }
    cross_task = {
        "raw": np.zeros((64, 64), dtype=np.float64),
        "unit": np.zeros((64, 64), dtype=np.float64),
    }
    primary, panel = recompute_bootstrap_tables(task_grams, cross_task, draws=2)
    detail, _reviewed_primary, _reviewed_panel = validate_bootstrap_contract(
        primary, panel, task_grams, cross_task, expected_draws=2
    )
    assert detail["bootstrap_draws"] == 2

    tampered = primary.copy()
    row = (tampered["bootstrap"] == 0) & (tampered["vector_kind"] == "unit")
    tampered.loc[row, "V_probe"] += 0.5
    tampered.loc[row, "Delta"] += 0.5
    tampered.loc[row, "Delta_total"] += 0.5
    tampered.loc[row, "fashion_delta"] += 0.5
    tampered.loc[row, "fashion_minus_mnist_delta"] += 0.5
    with pytest.raises(ReviewError, match="mismatch"):
        validate_bootstrap_contract(
            tampered, panel, task_grams, cross_task, expected_draws=2
        )


def test_nonalignment_review_rejects_self_consistent_cosine_tampering() -> None:
    state_gram = np.ones((64, 64), dtype=np.float64)
    task_gram = np.ones((16 * 64, 16 * 64), dtype=np.float64)
    reconstruction_keys = {
        *(f"pair_{left:02d}_{left + 16:02d}" for left in range(16)),
        *(
            f"pair_{left:02d}_{left + 1:02d}"
            for start in (0, 16)
            for left in range(start, start + 16, 2)
        ),
    }
    archives = {
        "state_grams": {
            f"state_{position:02d}_p4": state_gram.copy() for position in range(32)
        },
        "task_grams": {
            "fashion_mnist": task_gram.copy(),
            "mnist": task_gram.copy(),
        },
        "reconstruction_cross": {
            key: state_gram.copy() for key in reconstruction_keys
        },
        "cross_task": {
            "raw": 256.0 * state_gram,
            "unit": 256.0 * state_gram,
        },
    }
    crossfit, task_crossfit, gates, detail = recompute_nonalignment_tables(archives)
    assert detail["conditional_mean_gate_passing_states"] == 32
    state_summary = pd.DataFrame(
        {
            "state_position": range(32),
            "prefix": 4,
            "split_gate_pass": True,
        }
    )
    reviewed = _validate_nonalignment(
        archives, state_summary, crossfit, task_crossfit, gates
    )
    assert reviewed["within_task_state_nonalignment_gate"] is False

    tampered = crossfit.copy()
    tampered.loc[0, "crossfit_cosine_forward"] = 0.5
    tampered.loc[0, "crossfit_cosine"] = 0.75
    with pytest.raises(ReviewError, match="mismatch"):
        _validate_nonalignment(
            archives, state_summary, tampered, task_crossfit, gates
        )
