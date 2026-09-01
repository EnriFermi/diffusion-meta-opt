from __future__ import annotations

from copy import deepcopy

import torch

from big_vae.datasets.operator_bank import OperatorBankSample
from training.weightclip_benchmark.run_mini_polar_regression_production import (
    EXPECTED_PARAMETERS,
    MiniPolarConfig,
    MiniPolarWeightBottleneck,
    SubtileCursor,
    ValidSubtileStream,
    _behavioral_direction_scale_routed,
    _polar_routed_loss,
    _prepare_normalized_inputs,
)
from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin


def _parent_sample() -> OperatorBankSample:
    weight = torch.arange(128 * 128, dtype=torch.float32).view(128, 128)
    x = torch.arange(8 * 128, dtype=torch.float32).view(8, 128)
    d_in = torch.zeros(128, dtype=torch.bool)
    d_out = torch.zeros(128, dtype=torch.bool)
    d_in[:48] = True
    d_out[:65] = True
    return OperatorBankSample(
        x=x,
        weight=weight,
        meta={
            "logical_index": 7,
            "tile_row": 2,
            "tile_col": 1,
            "tile_row_start": 256,
            "tile_col_start": 128,
            "d_in_mask": d_in,
            "d_out_mask": d_out,
            "x_mask": torch.ones(8, dtype=torch.bool),
        },
        model_name="test-model",
        layer_name="test-layer",
    )


def test_valid_subtile_stream_skips_padding_and_resumes_exactly() -> None:
    parent = _parent_sample()
    stream = ValidSubtileStream(iter([parent]), SubtileCursor(7, 0, 19))
    samples = [next(stream) for _ in range(6)]
    # Two valid input blocks x three valid output blocks.
    assert [sample.meta["subpatch_index"] for sample in samples] == [0, 1, 2, 4, 5, 6]
    assert [sample.meta["logical_index"] for sample in samples] == list(range(19, 25))
    assert samples[0].meta["tile_row"] == 8
    assert samples[0].meta["tile_col"] == 4
    assert torch.equal(samples[5].weight, parent.weight[32:64, 64:96])
    assert torch.equal(samples[5].x, parent.x[:, 32:64])
    assert stream.cursor() == SubtileCursor(7, 7, 25)
    assert stream.skipped_empty_subtiles == 1

    resumed = ValidSubtileStream(iter([deepcopy(parent)]), stream.cursor())
    try:
        next(resumed)
    except StopIteration:
        pass
    else:
        raise AssertionError("the remainder of this partial parent must be empty")


def test_mini_model_parameter_count_shapes_and_all_gradients() -> None:
    torch.manual_seed(42)
    model = MiniPolarWeightBottleneck(MiniPolarConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    batch = 2
    W = torch.randn(batch, 32, 32) * 0.02
    X = torch.randn(batch, 24, 32)
    d_in = torch.ones(batch, 32, dtype=torch.bool)
    d_out = torch.ones(batch, 32, dtype=torch.bool)
    x_mask = torch.ones(batch, 24, dtype=torch.bool)
    duplicate_X = X[[0, 0, 1, 1]]
    duplicate_mask = x_mask[[0, 0, 1, 1]]
    direct_distribution = model._encode_distribution_context(
        duplicate_X, duplicate_mask
    )
    deduplicated_distribution = model._encode_distribution_context(
        duplicate_X,
        duplicate_mask,
        torch.tensor([0, 2]),
        torch.tensor([0, 0, 1, 1]),
    )
    assert torch.allclose(direct_distribution, deduplicated_distribution, atol=2e-6)
    content, log_scale, token_valid = _prepare_normalized_inputs(
        W,
        d_in,
        d_out,
        scale_mean=-8.0,
        scale_std=1.0,
    )
    prediction, latent, depth, pred_dirs, pred_log_scales = model(
        content,
        log_scale,
        torch.tensor([0, 71]),
        X,
        d_in_mask=d_in,
        d_out_mask=d_out,
        tile_col=torch.tensor([0, 7]),
        activation_sample_mask=x_mask,
        token_valid_mask=token_valid,
        capture_depth=True,
    )
    assert prediction.shape == (batch, 32, 32)
    assert latent.shape == (batch, 16, 48)
    assert len(depth) == 10
    assert pred_dirs.shape == (batch, 32, 2, 16)
    assert pred_log_scales.shape == (batch, 32, 2)
    assert torch.allclose(pred_dirs.norm(dim=-1), torch.ones(batch, 32, 2), atol=2e-5)
    loss, parts = _polar_routed_loss(
        X,
        W,
        pred_dirs,
        pred_log_scales,
        x_mask,
        d_in,
        d_out,
        {
            "behavioral_coef": 1.0,
            "behavioral_operator": 0.0,
            "behavioral_direction": 1.0,
            "behavioral_scale": 10.0,
            "structural_coef": 1.0,
            "structural_direction": 1.0,
            "structural_scale": 10.0,
            "structural_reconstruction": 0.0,
            "structural_relational": 0.0,
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(parts["behavioral_operator"]) == 0.0
    assert not [name for name, parameter in model.named_parameters() if parameter.grad is None]


def test_fused_behavioral_routing_matches_historical_loss_terms() -> None:
    torch.manual_seed(7)
    X = torch.randn(3, 11, 32)
    W = torch.randn(3, 32, 32)
    direction_prediction = torch.randn_like(W)
    scale_prediction = torch.randn_like(W)
    x_mask = torch.rand(3, 11) > 0.2
    d_out_mask = torch.rand(3, 32) > 0.15
    expected_direction, _ = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
        direction_prediction,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )
    _, expected_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
        scale_prediction,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )
    actual_direction, actual_scale = _behavioral_direction_scale_routed(
        X,
        W,
        direction_prediction,
        scale_prediction,
        x_mask,
        d_out_mask,
    )
    assert torch.equal(actual_direction, expected_direction)
    assert torch.equal(actual_scale, expected_scale)
