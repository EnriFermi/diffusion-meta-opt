from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import audit_one_state_exact_a_realized_path as audit


def test_blocked_basis_gradient_sums_one_backward_per_block(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.7, 0.5]))
    z = torch.zeros(4)
    cotangent = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.5, 0.2, 0.3],
            [0.4, -0.7, 1.3, 2.1],
            [2.2, 0.1, -0.8, 0.9],
        ]
    )

    monkeypatch.setattr(audit, "_task_set_for_record", lambda *_args, **_kwargs: None)

    def fake_hvp(_cfg, _vae, _normalizer, _z, basis, **_kwargs):
        return parameter * (1.0 + torch.argmax(basis).to(parameter.dtype))

    monkeypatch.setattr(audit, "_latent_hvp", fake_hvp)
    gradient, metadata = audit._blocked_basis_gradient(
        cfg=None,
        run=SimpleNamespace(task_tensors={}, vae=None, normalizer=None, spec=None),
        z=z,
        record={},
        active=[parameter],
        k_cotangent=cotangent,
        block_size=2,
    )
    expected = sum(
        (index + 1.0) * cotangent[index]
        for index in range(cotangent.shape[0])
    )
    assert torch.allclose(gradient[0], expected.double())
    assert metadata["basis_count"] == 4
    assert metadata["block_count"] == 2
    assert metadata["unused_parameter_tensors_all_blocks"] == 0


def test_scalarization_stats_identifies_critical_beta_crossing(monkeypatch) -> None:
    monkeypatch.setattr(audit, "OLD_BETA", 3.0)
    gradient_a = [torch.tensor([2.0, 0.0])]
    gradient_b = [torch.tensor([-1.0, 1.0])]
    result = audit._scalarization_stats(gradient_a, gradient_b)
    assert result["dot_ab"] == pytest.approx(-2.0)
    assert result["beta_crit"] == pytest.approx(2.0)
    assert result["s_value"] == pytest.approx(-2.0)
    assert result["conflict"] is True


def test_realized_direction_metrics_use_stored_float32_endpoints() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    base = [parameter.detach().clone()]
    nominal = [torch.tensor([0.2, -0.4], dtype=torch.float32)]
    radius = 0.25
    plus = [(base[0] + radius * nominal[0]).float()]
    minus = [(base[0] - radius * nominal[0]).float()]
    gradient_a = [torch.tensor([3.0, -2.0], dtype=torch.float64)]
    metrics, rows = audit._realized_direction_metrics(
        active_names=["decoder.weight"],
        active=[parameter],
        base=base,
        plus=plus,
        minus=minus,
        nominal=nominal,
        gradient_a=gradient_a,
        gradient_f=gradient_a,
        radius=radius,
    )
    expected_slope = float((gradient_a[0] * nominal[0].double()).sum())
    assert metrics["slope_theta"] == pytest.approx(expected_slope, rel=1e-6)
    assert metrics["effective_to_nominal_cosine"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["effective_to_nominal_norm_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert rows[0]["parameter"] == "decoder.weight"
