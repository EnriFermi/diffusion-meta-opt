from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.review_one_state_exact_selected_trajectory import (
    DEFAULT_OUTPUT,
    ReviewError,
    _cell_equivalent,
    _checkpoint_rows_match_csv,
    _direction_identity,
    _repeat_passes,
    _validate_direction_record,
)


def test_empty_optional_text_survives_csv_missing_roundtrip() -> None:
    assert not _cell_equivalent("", np.nan)
    assert _cell_equivalent("", np.nan, empty_text_is_missing=True)
    assert _cell_equivalent(np.nan, "", empty_text_is_missing=True)
    assert not _cell_equivalent(
        "cancellation_gate", np.nan, empty_text_is_missing=True
    )


def test_checkpoint_rows_accept_only_semantically_empty_failure() -> None:
    rows = [{"target_update": 1, "accepted": 1, "failure": ""}]
    parsed = pd.DataFrame(
        [{"target_update": 1, "accepted": 1, "failure": np.nan}]
    )
    assert _checkpoint_rows_match_csv(rows, parsed)

    nonempty = pd.DataFrame(
        [{"target_update": 1, "accepted": 1, "failure": "cancellation_gate"}]
    )
    assert not _checkpoint_rows_match_csv(rows, nonempty)


def test_fp32_materialized_direction_identity_is_tight_but_not_exact() -> None:
    rows = pd.read_csv(DEFAULT_OUTPUT / "arm_selection.csv")
    for _, row in rows.iterrows():
        _direction_identity(row, str(row["arm"]))

    tampered = rows.iloc[0].copy()
    tampered["slope_A"] = float(tampered["slope_A"]) + 1e-5
    with pytest.raises(ReviewError, match="A slope identity failed"):
        _direction_identity(tampered, str(tampered["arm"]))


def test_repeat_gate_iterates_series_column_names() -> None:
    rows = pd.read_csv(DEFAULT_OUTPUT / "arm_selection.csv")
    tolerances = {
        "A": 1e-9,
        "B": 1e-9,
        "L_low": 1e-10,
        "A_low90": 1e-9,
        "A_gt1": 1e-9,
        "m_max": 1e-8,
        "m_p50": 1e-10,
        "effective_rank": 1e-8,
    }
    assert all(_repeat_passes(row, tolerances) for _, row in rows.iterrows())

    tampered = rows.iloc[0].copy()
    tampered["repeat_exact_a_per_dim_abs_error"] = 1e-4
    assert not _repeat_passes(tampered, tolerances)


def test_fp32_direction_radius_allows_observed_roundoff_only() -> None:
    rows = pd.read_csv(DEFAULT_OUTPUT / "proposal_diagnostics.csv")
    for _, row in rows.iterrows():
        _validate_direction_record(row, auxiliary="low")

    tampered = rows.iloc[0].copy()
    tampered["direction_norm"] = float(tampered["direction_norm"]) + 1e-6
    with pytest.raises(ReviewError, match="direction radius mismatch"):
        _validate_direction_record(tampered, auxiliary="low")


def test_terminal_cancellation_source_tamper_is_rejected() -> None:
    rows = pd.read_csv(DEFAULT_OUTPUT / "proposal_diagnostics.csv")
    terminal = rows.loc[rows["accepted"].eq(0)].iloc[0].copy()
    details = _validate_direction_record(terminal, auxiliary="low")
    assert details["cancellation_gate_pass"] is False

    terminal["unit_common_source_norm"] = (
        float(terminal["unit_common_source_norm"]) + 1e-4
    )
    with pytest.raises(ReviewError, match="source norm identity failed"):
        _validate_direction_record(terminal, auxiliary="low")
