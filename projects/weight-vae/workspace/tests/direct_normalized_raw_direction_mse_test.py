from __future__ import annotations

from unittest.mock import patch

import torch

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    _operator_scale_only_loss,
    _polar_raw_direction_mse_production_loss,
    _raw_p16_direction_mse,
)


TARGET_RADIUS = 0.08615882694721222  # Seed-42 initialization-scale median.


def _full_masks(batch: int, d_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.ones(batch, 128, dtype=torch.bool),
        torch.ones(batch, d_out, dtype=torch.bool),
    )


def test_raw_direction_mse_is_zero_at_radius_matched_target() -> None:
    generator = torch.Generator().manual_seed(17)
    target_patches = torch.randn(2, 3, 8, 16, generator=generator)
    W = target_patches.reshape(2, 3, 128).transpose(1, 2).contiguous()
    d_in_mask, d_out_mask = _full_masks(2, 3)
    target_direction = target_patches / target_patches.norm(dim=-1, keepdim=True)
    logits = (TARGET_RADIUS * target_direction).requires_grad_()

    loss, stats = _raw_p16_direction_mse(
        logits,
        W,
        d_in_mask,
        d_out_mask,
        target_radius=TARGET_RADIUS,
    )

    assert float(loss.detach()) < 1.0e-12
    assert abs(float(stats["raw_direction_pred_radius_mean"]) - TARGET_RADIUS) < 1.0e-7
    loss.backward()
    assert float(logits.grad.abs().max()) < 1.0e-7


def test_raw_direction_mse_matches_unit_scale_at_orthogonal_equal_radius() -> None:
    target_patches = torch.zeros(1, 2, 8, 16)
    target_patches[..., 0] = torch.linspace(0.5, 2.0, 16).reshape(1, 2, 8)
    W = target_patches.reshape(1, 2, 128).transpose(1, 2).contiguous()
    logits = torch.zeros_like(target_patches)
    logits[..., 1] = TARGET_RADIUS
    d_in_mask, d_out_mask = _full_masks(1, 2)

    loss, _ = _raw_p16_direction_mse(
        logits,
        W,
        d_in_mask,
        d_out_mask,
        target_radius=TARGET_RADIUS,
    )

    assert torch.allclose(loss, torch.ones_like(loss), atol=1.0e-6, rtol=0.0)


def test_scale_only_helper_matches_old_scale_scalar() -> None:
    generator = torch.Generator().manual_seed(23)
    X = torch.randn(2, 5, 128, generator=generator)
    W = torch.randn(2, 128, 4, generator=generator)
    W_hat = torch.randn(2, 128, 4, generator=generator)
    x_mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )
    d_out_mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True]]
    )

    scale_only = _operator_scale_only_loss(X, W, W_hat, x_mask, d_out_mask)
    _, old_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
        W_hat,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )

    assert torch.allclose(scale_only, old_scale, atol=1.0e-7, rtol=1.0e-6)


def test_k1_two_loss_objective_omits_behavioral_scale_graph() -> None:
    generator = torch.Generator().manual_seed(29)
    batch, d_out = 2, 3
    X = torch.randn(batch, 6, 128, generator=generator)
    W = torch.randn(batch, 128, d_out, generator=generator)
    raw_logits = (0.02 * torch.randn(batch, d_out, 8, 16, generator=generator)).requires_grad_()
    pred_dirs = torch.randn(batch, d_out, 8, 16, generator=generator)
    pred_dirs = (pred_dirs / pred_dirs.norm(dim=-1, keepdim=True)).requires_grad_()
    pred_log_scales = torch.zeros(batch, d_out, 8, requires_grad=True)
    x_mask = torch.ones(batch, 6, dtype=torch.bool)
    d_in_mask, d_out_mask = _full_masks(batch, d_out)
    loss_cfg = {
        "raw_direction_mse": 1.0,
        "behavioral_coef": 0.0,
        "behavioral_operator": 0.0,
        "behavioral_direction": 0.0,
        "behavioral_scale": 0.0,
        "structural_coef": 1.0,
        "structural_direction": 0.0,
        "structural_scale": 10.0,
        "structural_reconstruction": 0.0,
        "structural_relational": 0.0,
    }
    regression_cfg = {"target_radius": TARGET_RADIUS}

    with patch(
        "training.weightclip_benchmark."
        "run_direct_normalized_scaled_700m_production._operator_scale_only_loss",
        side_effect=AssertionError("behavioral scale must not be constructed"),
    ):
        total, parts, direction_objective, scale_objective = (
            _polar_raw_direction_mse_production_loss(
                X,
                W,
                raw_logits,
                pred_dirs,
                pred_log_scales,
                x_mask,
                d_in_mask,
                d_out_mask,
                loss_cfg,
                regression_cfg,
            )
        )

    assert torch.allclose(total, direction_objective + scale_objective)
    assert float(parts["behavioral"]) == 0.0
    assert float(parts["behavioral_direction"]) == 0.0
    assert float(parts["behavioral_scale"]) == 0.0
    assert float(parts["structural_direction"]) == 0.0
    total.backward()
    assert raw_logits.grad is not None and float(raw_logits.grad.norm()) > 0.0
    assert pred_log_scales.grad is not None and float(pred_log_scales.grad.norm()) > 0.0
    assert pred_dirs.grad is None
