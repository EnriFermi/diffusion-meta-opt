from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.review_variant_a_estimator_stability import (
    GRID,
    ReviewError,
    apply_reference_gated_decisions,
    exact_symmetric_scalar_error,
    require_finite,
)


def test_exact_symmetric_scalar_error_has_no_epsilon() -> None:
    assert exact_symmetric_scalar_error(0.0, -0.0) == 0.0
    assert exact_symmetric_scalar_error(3.0, 3.0) == 0.0
    assert exact_symmetric_scalar_error(-1.0, 1.0) == 2.0
    assert exact_symmetric_scalar_error(0.0, 1e-300) == 2.0
    assert exact_symmetric_scalar_error(1.0, 3.0) == 1.0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_require_finite_rejects_any_nonfinite_metric(bad_value: float) -> None:
    frame = pd.DataFrame({"metric": [1.0, bad_value], "other": [2.0, 3.0]})
    with pytest.raises(ReviewError, match="NaN/inf/nonnumeric"):
        require_finite(frame, ["metric", "other"], "synthetic.csv")


def test_reference_failure_makes_cells_not_evaluable_without_budget_censoring() -> None:
    summary = pd.DataFrame(
        [
            {
                "states": states_count,
                "pairs": pairs_count,
                "hvp_count": 2 * states_count * pairs_count,
                "raw_cell_thresholds_pass": True,
            }
            for states_count in GRID
            for pairs_count in GRID
        ]
    )

    reviewed, decision = apply_reference_gated_decisions(summary, reference_valid=False)

    assert set(reviewed["cell_verdict"]) == {"not_evaluable"}
    assert set(reviewed["state_doubling_verdict"]) == {"not_evaluable"}
    assert set(reviewed["pair_doubling_verdict"]) == {"not_evaluable"}
    assert set(reviewed["axial_persistence_verdict"]) == {"not_evaluable"}
    assert decision["selected_axially_confirmed_cell"] is None
    assert decision["budget_censored"] is False
    assert decision["requires_reference_extension"] is True
    assert decision["requires_budget_extension"] is False
    assert decision["current_2x4"]["cell_verdict"] == "not_evaluable"
