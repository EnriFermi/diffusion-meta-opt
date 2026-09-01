from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from big_vae.datasets.operator_bank import OperatorBankSample
from training.weightclip_benchmark.run_mini_polar_latent_preference_production import (
    EXPECTED_PROJECTOR_PARAMETERS,
    HardPairResolver,
    LatentRepresentationProjector,
    _atomic_hardlink,
    _binary_matched_infonce,
    _decoder_preference_loss,
    _load_config,
    _objective_gradient_groups,
    _pair_weight_distances,
    _polar_routed_loss_per_example,
)
from training.weightclip_benchmark.run_mini_polar_regression_production import (
    MiniPolarConfig,
    MiniPolarWeightBottleneck,
    _polar_routed_loss,
)
from training.weightclip_benchmark.stream_direct_normalized_production_to_comet import (
    _gradient_row_metrics,
)


LOSS_CONFIG = {
    "behavioral_coef": 1.0,
    "behavioral_operator": 0.0,
    "behavioral_direction": 1.0,
    "behavioral_scale": 10.0,
    "structural_coef": 1.0,
    "structural_direction": 1.0,
    "structural_scale": 10.0,
    "structural_reconstruction": 0.0,
    "structural_relational": 0.0,
}


def test_retained_checkpoint_hardlink_is_atomic_and_storage_shared(tmp_path: Path) -> None:
    source = tmp_path / "resume_latest.pt"
    source.write_bytes(b"full-resume-payload")
    destination = tmp_path / "retained" / "resume_step_000006500.pt"
    _atomic_hardlink(source, destination)
    assert destination.read_bytes() == b"full-resume-payload"
    assert source.stat().st_ino == destination.stat().st_ino


def test_revised_preference_calibration_contract_is_exact() -> None:
    config = _load_config(
        Path(
            "conf/weightclip_benchmark/"
            "mini_polar_latent_preference_10m_p16_tile32_"
            "pref025_ramp10k_production_500k.yaml"
        )
    )
    calibration = config["calibration"]
    assert calibration["preference_target_ratio"] == 0.025
    assert calibration["representation_target_encoder_ratio"] == 0.10
    assert calibration["ramp_steps"] == 10000
    assert calibration["auxiliary_gradient_ratio_stop"] == 0.75
    assert config["seed"] == 42
    assert config["steps"] == 500000
    assert config["batch_size"] == 128


def test_beta1_099_arm_changes_only_optimizer_and_runtime_paths() -> None:
    reference = _load_config(
        Path(
            "conf/weightclip_benchmark/"
            "mini_polar_latent_preference_10m_p16_tile32_"
            "pref025_ramp10k_production_500k.yaml"
        )
    )
    beta1_arm = _load_config(
        Path(
            "conf/weightclip_benchmark/"
            "mini_polar_latent_preference_10m_p16_tile32_"
            "pref025_ramp10k_beta1_099_production_500k.yaml"
        )
    )
    assert beta1_arm["betas"] == [0.99, 0.999]
    assert beta1_arm["device"] == "cuda:1"
    allowed = {
        "betas",
        "device",
        "output_root",
        "resume_checkpoint",
        "persistent_model_checkpoint",
    }
    assert {
        key
        for key in reference
        if reference.get(key) != beta1_arm.get(key)
    } == allowed


def _normalized_directions(batch: int) -> torch.Tensor:
    directions = torch.randn(batch, 32, 2, 16)
    return F.normalize(directions, dim=-1)


def test_per_example_preference_loss_is_exact_for_single_example() -> None:
    torch.manual_seed(19)
    batch = 1
    X = torch.randn(batch, 17, 32)
    W = torch.randn(batch, 32, 32)
    pred_dirs = _normalized_directions(batch)
    pred_scales = torch.randn(batch, 32, 2).clamp(-2.5, 2.5)
    x_mask = torch.rand(batch, 17) > 0.2
    d_in_mask = torch.ones(batch, 32, dtype=torch.bool)
    d_out_mask = torch.ones(batch, 32, dtype=torch.bool)
    d_in_mask[:, 27:] = False
    d_out_mask[:, 29:] = False
    pred_dirs = pred_dirs * (
        d_out_mask[:, :, None, None]
        & d_in_mask.view(batch, 2, 16)[:, None]
    )
    pred_scales = pred_scales * (
        d_out_mask[:, :, None]
        & d_in_mask.view(batch, 2, 16).any(dim=-1)[:, None]
    )
    historical, parts = _polar_routed_loss(
        X,
        W,
        pred_dirs,
        pred_scales,
        x_mask,
        d_in_mask,
        d_out_mask,
        LOSS_CONFIG,
    )
    per_example = _polar_routed_loss_per_example(
        X,
        W,
        pred_dirs,
        pred_scales,
        x_mask,
        d_in_mask,
        d_out_mask,
        LOSS_CONFIG,
    )
    assert torch.allclose(per_example["total"].squeeze(0), historical, atol=2e-6, rtol=2e-6)
    for name in (
        "behavioral_direction",
        "behavioral_scale",
        "structural_direction",
        "structural_scale",
    ):
        assert torch.allclose(per_example[name].squeeze(0), parts[name], atol=2e-6, rtol=2e-6)


def test_symmetric_binary_infonce_and_preference_have_expected_limits() -> None:
    torch.manual_seed(23)
    projector = LatentRepresentationProjector(MiniPolarConfig())
    assert sum(parameter.numel() for parameter in projector.parameters()) == EXPECTED_PROJECTOR_PARAMETERS
    identical = F.normalize(torch.ones(4, 8), dim=-1)
    loss, stats = _binary_matched_infonce(identical, identical, temperature=0.1)
    assert torch.allclose(loss, loss.new_tensor(math.log(2.0)), atol=1e-6)
    assert float(stats["pairwise_accuracy"]) == 0.0

    positive = {"total": torch.ones(4)}
    negative = {"total": torch.full((4,), 1.5)}
    preference, preference_stats = _decoder_preference_loss(
        positive,
        negative,
        margin=0.5,
        temperature=0.1,
        eps=1.0e-6,
    )
    assert float(preference) < 0.07
    assert float(preference_stats["margin_satisfied"]) == 0.0  # eps makes the gap infinitesimally below 0.5


def _sample(*, lineage: str, checkpoint: str) -> OperatorBankSample:
    meta = {
        "dataset": "dataset",
        "checkpoint_index_zero_based": 43,
        "checkpoint_sha256": checkpoint,
        "lineage_id": lineage,
        "layer_key": "layer1.0.conv1.weight",
        "operator": {"operation": "conv2d", "depth_index": 1, "role": "residual_conv1"},
        "parent_tile_row": 0,
        "parent_tile_col": 0,
        "subpatch_index": 0,
        "subtile_input_index": 0,
        "subtile_output_index": 0,
        "tile_row": 0,
        "tile_col": 0,
        "gauge_id": "canonical",
        "d_in_mask": torch.ones(32, dtype=torch.bool),
        "d_out_mask": torch.ones(32, dtype=torch.bool),
        "x_mask": torch.ones(8, dtype=torch.bool),
    }
    return OperatorBankSample(
        x=torch.randn(8, 32),
        weight=torch.randn(32, 32),
        meta=meta,
        model_name=lineage,
        layer_name=str(meta["layer_key"]),
    )


def test_hard_pair_validation_rejects_false_matches() -> None:
    left = _sample(lineage="dataset:seed=1", checkpoint="a")
    right = _sample(lineage="dataset:seed=2", checkpoint="b")
    HardPairResolver._validate(left, right)
    same_lineage = deepcopy(right)
    same_lineage.meta["lineage_id"] = left.meta["lineage_id"]
    try:
        HardPairResolver._validate(left, same_lineage)
    except RuntimeError as exc:
        assert "different lineages" in str(exc)
    else:
        raise AssertionError("same-lineage false negative must be rejected")
    wrong_coordinate = deepcopy(right)
    wrong_coordinate.meta["tile_col"] = 1
    try:
        HardPairResolver._validate(left, wrong_coordinate)
    except RuntimeError as exc:
        assert "tile_col" in str(exc)
    else:
        raise AssertionError("coordinate mismatch must be rejected")


def test_negative_decoder_branch_detaches_conditioning_but_trains_decoder() -> None:
    torch.manual_seed(29)
    model = MiniPolarWeightBottleneck(MiniPolarConfig())
    z = torch.randn(1, 16, 48, requires_grad=True)
    dist_patch = torch.randn(1, 2, 64, requires_grad=True)
    d_in = torch.ones(1, 32, dtype=torch.bool)
    d_out = torch.ones(1, 32, dtype=torch.bool)
    token_valid = torch.ones(1, 64, dtype=torch.bool)
    prediction, _directions, _scales = model.decode_polar(
        z.detach(),
        torch.tensor([1]),
        torch.tensor([1]),
        d_in,
        d_out,
        token_valid,
        dist_patch,
        detach_direct_conditioning=True,
    )
    prediction.square().mean().backward()
    assert z.grad is None
    assert dist_patch.grad is None
    assert model.output_queries.grad is None
    assert model.tile_row_embedding.weight.grad is None
    assert model.tile_col_embedding.weight.grad is None
    assert not [
        name
        for name, parameter in model.decoder_query_conditioner.named_parameters()
        if parameter.grad is not None
    ]
    assert model.from_latent.weight.grad is not None
    assert model.decoder_blocks[0].qkv.weight.grad is not None
    assert model.direction_head.weight.grad is not None
    assert model.scale_head.weight.grad is not None


def test_objective_gradient_groups_and_sidecar_accept_new_schema() -> None:
    named = [
        ("model.encoder_blocks.0.qkv.weight", torch.zeros(3, 3)),
        ("model.from_latent.weight", torch.zeros(2, 2)),
        ("model.decoder_blocks.0.qkv.weight", torch.zeros(4, 4)),
        ("model.direction_head.weight", torch.zeros(2, 3)),
        ("model.scale_head.weight", torch.zeros(1, 3)),
        ("projector.first.weight", torch.zeros(5, 5)),
    ]
    grads = [torch.ones_like(parameter) for _name, parameter in named]
    grouped = _objective_gradient_groups(named, grads)
    assert grouped["encoder"]["gradient_l2"] == 3.0
    assert grouped["latent_bridge"]["gradient_l2"] == 2.0
    assert grouped["representation_head"]["gradient_l2"] == 5.0
    row = {
        "schema": "weightclip_mini_polar_latent_preference_production_v1",
        "step": 1,
        "auxiliary_gradient_ratio": 0.2,
        "model_groups": {
            "decoder_block_1": {"gradient_rms": 0.03, "numel": 16}
        },
        "projector_groups": {
            "input_and_queries": {"gradient_rms": 0.04, "numel": 25}
        },
        "objective_gradient_groups": {
            "base": grouped,
            "preference": grouped,
            "representation": grouped,
        },
    }
    metrics = _gradient_row_metrics(row, raw_direction_mse=False)
    assert metrics["grad_rms/model/decoder_block_1"] == 0.03
    assert metrics["grad_rms/projector/input_and_queries"] == 0.04
    assert metrics["objective_gradient/base/encoder/gradient_l2"] == 3.0


def test_pair_weight_distance_is_zero_only_for_identical_pairs() -> None:
    W = torch.zeros(4, 32, 32)
    W[1] = 1.0
    W[2] = 2.0
    W[3] = 2.0
    mask = torch.ones(4, 32, dtype=torch.bool)
    absolute, relative = _pair_weight_distances(W, mask, mask)
    assert torch.allclose(absolute, torch.tensor([1.0, 0.0]))
    assert float(relative[0]) > 0.0
    assert float(relative[1]) == 0.0
