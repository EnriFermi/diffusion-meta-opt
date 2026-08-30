from __future__ import annotations

import torch

from scripts.audit_one_state_exact_low_mode_pullback import (
    _h_space_fd,
    _loss_low,
    _low_energy,
    _repeat_errors,
)


def test_frozen_low_objective_cotangent_sign_and_formula() -> None:
    generator = torch.Generator().manual_seed(17)
    hessian = torch.randn(7, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    basis, _ = torch.linalg.qr(torch.randn(7, 3, generator=generator, dtype=torch.float64))
    loss = _loss_low(hessian, basis)
    observed = torch.autograd.grad(loss, hessian)[0]
    expected = -2.0 * basis @ (basis.T @ hessian.detach()) / 3.0
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_h_space_fd_matches_analytic_cotangent() -> None:
    generator = torch.Generator().manual_seed(23)
    hessian = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(torch.randn(8, 4, generator=generator, dtype=torch.float64))
    cotangent = -2.0 * basis @ (basis.T @ hessian) / 4.0
    rows = _h_space_fd(hessian, basis, cotangent)
    assert all(bool(row["sign_agrees"]) for row in rows)
    assert max(float(row["relative_error"]) for row in rows) < 1e-9


def test_low_energy_is_projected_metric_trace() -> None:
    hessian = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    matrix = hessian @ hessian.T
    basis = torch.eye(3, dtype=torch.float64)[:, :2]
    assert _low_energy(matrix, basis) == 2.5


def test_repeat_errors_cover_all_metrics_except_runtime() -> None:
    primary = {
        "task_loss": 1.0,
        "a_trace_closure": 2.0,
        "hessian_sec": 3.0,
        "count_lt_0p1": 4.0,
    }
    repeat = {
        "task_loss": 1.1,
        "a_trace_closure": 2.2,
        "hessian_sec": 99.0,
        "count_lt_0p1": 5.0,
    }
    eig = torch.tensor([1.0], dtype=torch.float64)
    errors = _repeat_errors(primary, repeat, eig, eig)
    assert errors["repeat_task_loss_abs_error"] > 0.0
    assert errors["repeat_a_trace_closure_abs_error"] > 0.0
    assert errors["repeat_count_lt_0p1_abs_error"] == 1.0
    assert "repeat_hessian_sec_abs_error" not in errors
