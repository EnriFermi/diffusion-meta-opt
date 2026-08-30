from __future__ import annotations

import pandas as pd
import pytest
import torch

from scripts import audit_one_state_exact_a_proposal3_cross as audit


def test_exact_a_hessian_cotangent_matches_central_difference() -> None:
    generator = torch.Generator().manual_seed(7)
    hessian = torch.randn(7, 7, generator=generator, dtype=torch.float64)
    direction = torch.randn(7, 7, generator=generator, dtype=torch.float64)
    dim = hessian.shape[0]
    metric = hessian @ hessian.T
    identity = torch.eye(dim, dtype=torch.float64)
    cotangent = 4.0 * (metric - identity) @ hessian / float(dim)
    analytic = torch.sum(cotangent * direction)

    def objective(value: torch.Tensor) -> torch.Tensor:
        matrix = value @ value.T
        return (matrix - identity).square().sum() / float(dim)

    step = 1e-6
    finite_difference = (
        objective(hessian + step * direction) - objective(hessian - step * direction)
    ) / (2.0 * step)
    assert float(finite_difference) == pytest.approx(float(analytic), rel=1e-8, abs=1e-8)


def test_exact_burg_hessian_cotangent_matches_central_difference() -> None:
    generator = torch.Generator().manual_seed(11)
    hessian = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    direction = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    dim = hessian.shape[0]
    epsilon = 1e-4
    identity = torch.eye(dim, dtype=torch.float64)
    metric = hessian @ hessian.T
    matrix_gradient = identity / (float(dim) * (1.0 + epsilon)) - torch.linalg.inv(
        metric + epsilon * identity
    ) / float(dim)
    cotangent = 2.0 * matrix_gradient @ hessian
    analytic = torch.sum(cotangent * direction)

    def objective(value: torch.Tensor) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(value @ value.T)
        ratio = (eigenvalues + epsilon) / (1.0 + epsilon)
        return (ratio - ratio.log() - 1.0).mean()

    step = 1e-6
    finite_difference = (
        objective(hessian + step * direction) - objective(hessian - step * direction)
    ) / (2.0 * step)
    assert float(finite_difference) == pytest.approx(float(analytic), rel=1e-7, abs=1e-7)


def test_mechanism_classifier_uses_crossed_direction_signs_and_radius() -> None:
    slopes = {
        name: {
            "slope_a_h128": -1.0,
            "slope_a_h256": -1.1,
        }
        for name in audit.DIRECTION_NAMES
    }
    slopes["p32_oldbeta_raw"] = {"slope_a_h128": 0.8, "slope_a_h256": 0.9}
    slopes["exact_oldbeta_raw"] = {"slope_a_h128": 0.7, "slope_a_h256": 0.8}
    slopes["p32_oldbeta_carried_adam"] = {
        "slope_a_h128": -0.5,
        "slope_a_h256": -0.4,
    }
    rows = [
        {
            "direction": "p32_oldbeta_carried_adam",
            "alpha": alpha,
            "exact_a_per_dim": value,
            "m_max": 13.0 if alpha == 1.0 else 9.0,
            "delta_m_max": 3.0 if alpha == 1.0 else -1.0,
            "a_high_gt_1_abs_per_dim": 0.7 if alpha == 1.0 else 0.3,
        }
        for alpha, value in ((1.0, 1.3), (0.5, 0.9), (0.25, 0.95), (0.125, 0.98),
                             (0.0625, 1.01), (0.03125, 1.01), (0.015625, 1.01))
    ]
    result = audit._classify_mechanisms(
        slopes,
        pd.DataFrame(rows),
        base={
            "exact_a_per_dim": 1.0,
            "m_max": 10.0,
            "a_high_gt_1_abs_per_dim": 0.4,
        },
    )
    assert result == {
        "finite_p32_estimator_supported": False,
        "exact_oldbeta_scalarization_conflict_supported": True,
        "p32_composite_estimator_supported": False,
        "carried_adam_transform_supported": False,
        "finite_radius_spike_overshoot_supported": True,
    }


def test_directional_reliability_rejects_a_slope_below_fd_uncertainty() -> None:
    reliable = audit._directional_derivative_reliability(
        -1.0,
        {
            "slope_a_h128": -0.99,
            "slope_a_h256": -1.0,
            "slope_a_repeat_h256": -1.0,
            "slope_a_endpoint_noise_bound_h256": 0.001,
        },
    )
    assert reliable["slope_a_reliable"] is True

    unresolved = audit._directional_derivative_reliability(
        -0.01,
        {
            "slope_a_h128": -0.01,
            "slope_a_h256": -0.011,
            "slope_a_repeat_h256": -0.009,
            "slope_a_endpoint_noise_bound_h256": 0.003,
        },
    )
    assert unresolved["slope_a_reliable"] is False


def test_candidate_requires_joint_component_and_spectrum_improvement() -> None:
    base = {
        "exact_a_per_dim": 1.4,
        "damped_full_burg_per_dim": 6.7,
        "m_max": 12.0,
        "m_p50": 0.01,
        "m_lt_0p1_fraction": 0.75,
    }
    frame = pd.DataFrame(
        [
            {
                "direction": "candidate",
                "kind": "line",
                "alpha": 0.5,
                "exact_a_per_dim": 1.2,
                "damped_full_burg_per_dim": 6.5,
                "m_max": 11.5,
                "m_p50": 0.011,
                "m_lt_0p1_fraction": 0.74,
            },
            {
                "direction": "candidate",
                "kind": "line",
                "alpha": 1.0,
                "exact_a_per_dim": 1.1,
                "damped_full_burg_per_dim": 6.4,
                "m_max": 13.0,
                "m_p50": 0.012,
                "m_lt_0p1_fraction": 0.73,
            },
        ]
    )
    tolerances = {
        "a": 1e-8,
        "b": 1e-8,
        "m_max": 1e-8,
        "m_p50": 1e-12,
        "m_lt_0p1_fraction": 1e-12,
    }
    result = audit._candidate_decision(frame, "candidate", base, tolerances)
    assert result == {
        "direction": "candidate",
        "selected": True,
        "largest_passing_alpha": 0.5,
        "passing_alpha_count": 1,
        "tolerances": tolerances,
    }


def test_p32_module_state_reset_clears_all_mutable_diagnostics() -> None:
    audit.p32.DRAW_ROWS.append({"proposal": 99})
    audit.p32.PAIR_ROWS.append({"proposal": 99})
    audit.p32.POOLING_PREFLIGHT["valid"] = True
    audit.p32.SEED_PREFLIGHT["valid"] = True
    audit.p32.TREATMENT_BUILD_CALLS = 3
    audit.p32.TREATMENT_CLIP_CALLS = 3
    audit.p32.TREATMENT_ADAM_CALLS = 3
    audit._reset_p32_module_state()
    assert audit.p32.DRAW_ROWS == []
    assert audit.p32.PAIR_ROWS == []
    assert audit.p32.POOLING_PREFLIGHT == {}
    assert audit.p32.SEED_PREFLIGHT == {}
    assert audit.p32.TREATMENT_BUILD_CALLS == 0
    assert audit.p32.TREATMENT_CLIP_CALLS == 0
    assert audit.p32.TREATMENT_ADAM_CALLS == 0
