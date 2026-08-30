from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    _batch_from_stream,
    _exact_layout_groups,
    _model_config,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_parallel_categorical_gptq_700m_production import (
    ParallelCategoricalGPTQWeightBottleneck,
    _batch_stream,
    _load_config,
    _mask_aware_gptq_targets,
    _ordered_cumulative_log_loss,
    _scale_bin_centers,
    _scale_class_targets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only physical scale audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _quantile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.float(), q))


def _masked_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().mean().sqrt() * right.square().mean().sqrt()
    return float((left * right).mean() / denominator.clamp_min(1.0e-30))


def _physical_metrics(
    prediction_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_scales: torch.Tensor,
    valid: torch.Tensor,
    centers: torch.Tensor,
) -> dict[str, float]:
    predicted_log2 = centers.index_select(0, prediction_ids.reshape(-1)).view_as(
        prediction_ids
    )
    target_log2_continuous = torch.log2(target_scales.float())
    target_log2_binned = centers.index_select(0, target_ids.reshape(-1)).view_as(
        target_ids
    )
    signed_log2_error = (predicted_log2 - target_log2_continuous)[valid]
    absolute_log2_error = signed_log2_error.abs()
    binning_log2_error = (target_log2_binned - target_log2_continuous)[valid].abs()
    prediction_scale = torch.exp2(predicted_log2)[valid]
    continuous_scale = target_scales.float()[valid]
    absolute_relative_error = (
        (prediction_scale - continuous_scale).abs()
        / continuous_scale.clamp_min(1.0e-30)
    )
    multiplicative_factor = torch.exp2(absolute_log2_error)
    return {
        "log2_error_mean_signed": float(signed_log2_error.mean()),
        "log2_error_mae": float(absolute_log2_error.mean()),
        "log2_error_rmse": float(signed_log2_error.square().mean().sqrt()),
        "log2_error_p50": _quantile(absolute_log2_error, 0.50),
        "log2_error_p90": _quantile(absolute_log2_error, 0.90),
        "log2_error_p99": _quantile(absolute_log2_error, 0.99),
        "multiplicative_factor_p50": _quantile(multiplicative_factor, 0.50),
        "multiplicative_factor_p90": _quantile(multiplicative_factor, 0.90),
        "multiplicative_factor_p99": _quantile(multiplicative_factor, 0.99),
        "absolute_relative_error_mean": float(absolute_relative_error.mean()),
        "absolute_relative_error_p50": _quantile(absolute_relative_error, 0.50),
        "absolute_relative_error_p90": _quantile(absolute_relative_error, 0.90),
        "absolute_relative_error_p99": _quantile(absolute_relative_error, 0.99),
        "target_binning_log2_error_p50": _quantile(binning_log2_error, 0.50),
        "target_binning_log2_error_p99": _quantile(binning_log2_error, 0.99),
        "prediction_target_log2_correlation": _masked_correlation(
            predicted_log2[valid], target_log2_continuous[valid]
        ),
    }


def _evaluate_scale(
    name: str,
    model: ParallelCategoricalGPTQWeightBottleneck,
    latent: torch.Tensor,
    tile_row: torch.Tensor,
    tile_col: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    token_valid: torch.Tensor,
    target_ids: torch.Tensor,
    target_scales: torch.Tensor,
    valid: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _code_logits, scale_logits, _ = model.decode_categorical(
            latent,
            tile_row,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid,
        )
    scale_loss, stats = _ordered_cumulative_log_loss(scale_logits, target_ids, valid)
    prediction_ids = scale_logits.float().argmax(dim=-1)
    result: dict[str, Any] = {
        "variant": name,
        "ordinal_loss": float(scale_loss),
        "accuracy": float(stats["accuracy"]),
        "off_by_one_accuracy": float(stats["off_by_one_accuracy"]),
        "mean_absolute_bin_error": float(stats["mean_absolute_bin_error"]),
        "mean_squared_bin_error": float(stats["mean_squared_bin_error"]),
        "prediction_entropy_normalized": float(stats["prediction_entropy"]),
        **_physical_metrics(prediction_ids, target_ids, target_scales, valid, centers),
    }
    return result, prediction_ids


def _constant_baseline(
    name: str,
    prediction_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_scales: torch.Tensor,
    valid: torch.Tensor,
    centers: torch.Tensor,
) -> dict[str, Any]:
    errors = (prediction_ids - target_ids).abs().float()[valid]
    return {
        "variant": name,
        "accuracy": float((errors == 0).float().mean()),
        "off_by_one_accuracy": float((errors <= 1).float().mean()),
        "mean_absolute_bin_error": float(errors.mean()),
        "mean_squared_bin_error": float(errors.square().mean()),
        **_physical_metrics(prediction_ids, target_ids, target_scales, valid, centers),
    }


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    print(f"[scale-audit] stage=checkpoint-open path={checkpoint_path}", flush=True)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    checkpoint_step = int(checkpoint["step"])
    cursor = int(checkpoint["committed_logical_index"])
    if checkpoint.get("schema") != config["schema"]:
        raise RuntimeError("checkpoint/config schema mismatch")
    if cursor != checkpoint_step * int(config["batch_size"]):
        raise RuntimeError("checkpoint cursor mismatch")

    device = torch.device(config["device"])
    model = ParallelCategoricalGPTQWeightBottleneck(
        _model_config(config),
        "normalized_float",
        scale_vocab_size=int(config["scale_bins"]["vocabulary_size"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    gc.collect()
    model.to(device).eval()

    probe_config = dict(config)
    probe_config["operator_bank"] = dict(config["operator_bank"])
    probe_config["operator_bank"]["loader_workers"] = 0
    logger = logging.getLogger("scale-audit")
    logger.setLevel(logging.WARNING)
    print(
        f"[scale-audit] stage=data batch={args.batch_size} cursor={cursor}", flush=True
    )
    with ExitStack() as stack:
        stream = _batch_stream(probe_config, cursor, logger, stack)
        cpu_batch = _batch_from_stream(stream, int(args.batch_size))

    weights = cpu_batch["W"].to(device)
    activation = cpu_batch["X"].to(device)
    x_mask = cpu_batch["x_mask"].to(device)
    d_in_mask = cpu_batch["d_in_mask"].to(device)
    d_out_mask = cpu_batch["d_out_mask"].to(device)
    tile_row = cpu_batch["tile_row"].to(device)
    tile_col = cpu_batch["tile_col"].to(device)
    _target_codes, target_scales, _ = _mask_aware_gptq_targets(
        weights,
        activation,
        x_mask,
        d_in_mask,
        d_out_mask,
        bits=int(config["gptq"]["bits"]),
        damp_fraction=float(config["gptq"]["damp_fraction"]),
    )
    centers = _scale_bin_centers(
        int(config["scale_bins"]["vocabulary_size"]),
        float(config["scale_bins"]["log2_min"]),
        float(config["scale_bins"]["log2_max"]),
        device=device,
    )
    target_ids = _scale_class_targets(target_scales, centers)
    valid = (
        d_out_mask
        & d_in_mask.any(dim=-1)[:, None]
        & (target_scales > 1.0e-8 / 7.0)
    )
    content, input_log_scale, token_valid = _prepare_normalized_inputs(
        weights,
        d_in_mask,
        d_out_mask,
        scale_mean=float(config["normalization"]["log2_scale_mean"]),
        scale_std=float(config["normalization"]["log2_scale_std"]),
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _code_logits, _scale_logits, latent, _ = model(
            content,
            input_log_scale,
            tile_row,
            activation,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            activation_sample_mask=x_mask,
            token_valid_mask=token_valid,
        )

    groups = _exact_layout_groups(
        cpu_batch["tile_row"],
        cpu_batch["tile_col"],
        cpu_batch["d_in_mask"],
        cpu_batch["d_out_mask"],
        minimum_group_size=2,
    )
    eligible = torch.zeros(int(args.batch_size), dtype=torch.bool, device=device)
    swap_index = torch.arange(int(args.batch_size), device=device)
    for group in groups:
        eligible[group] = True
        swap_index[group] = torch.tensor(group[1:] + group[:1], device=device)

    evaluations: list[dict[str, Any]] = []
    prediction_ids: dict[str, torch.Tensor] = {}
    for name, variant_latent, variant_valid in (
        ("original_z_all", latent, valid),
        ("original_z_exact_layout_subset", latent, valid & eligible[:, None]),
        (
            "exact_layout_swapped_z",
            latent.index_select(0, swap_index),
            valid & eligible[:, None],
        ),
        ("zero_z_exact_layout_subset", torch.zeros_like(latent), valid & eligible[:, None]),
    ):
        result, ids = _evaluate_scale(
            name,
            model,
            variant_latent,
            tile_row,
            tile_col,
            d_in_mask,
            d_out_mask,
            token_valid,
            target_ids,
            target_scales,
            variant_valid,
            centers,
        )
        evaluations.append(result)
        prediction_ids[name] = ids

    # Sealed fit-split global scale mode from the existing prior artifact.
    global_mode_ids = torch.full_like(target_ids, 122)
    baselines = [
        _constant_baseline(
            "sealed_fit_global_mode_class_122",
            global_mode_ids,
            target_ids,
            target_scales,
            valid,
            centers,
        )
    ]

    sample_oracle_ids = torch.zeros_like(target_ids)
    for sample in range(target_ids.shape[0]):
        sample_valid = valid[sample]
        sample_median = target_ids[sample][sample_valid].float().median().long()
        sample_oracle_ids[sample].fill_(sample_median)
    baselines.append(
        _constant_baseline(
            "oracle_per_sample_median_broadcast",
            sample_oracle_ids,
            target_ids,
            target_scales,
            valid,
            centers,
        )
    )

    original_ids = prediction_ids["original_z_all"]
    target_log2 = torch.log2(target_scales.float())
    predicted_log2 = centers.index_select(0, original_ids.reshape(-1)).view_as(
        original_ids
    )
    target_sample_means: list[torch.Tensor] = []
    predicted_sample_means: list[torch.Tensor] = []
    target_within: list[torch.Tensor] = []
    predicted_within: list[torch.Tensor] = []
    for sample in range(target_ids.shape[0]):
        sample_valid = valid[sample]
        target_values = target_log2[sample][sample_valid]
        predicted_values = predicted_log2[sample][sample_valid]
        target_sample_means.append(target_values.mean())
        predicted_sample_means.append(predicted_values.mean())
        target_within.append(target_values - target_values.mean())
        predicted_within.append(predicted_values - predicted_values.mean())
    target_means = torch.stack(target_sample_means)
    predicted_means = torch.stack(predicted_sample_means)
    target_within_flat = torch.cat(target_within)
    predicted_within_flat = torch.cat(predicted_within)
    decomposition = {
        "sample_mean_log2_correlation": _masked_correlation(
            predicted_means, target_means
        ),
        "sample_mean_log2_mae": float((predicted_means - target_means).abs().mean()),
        "within_sample_log2_correlation": _masked_correlation(
            predicted_within_flat, target_within_flat
        ),
        "within_sample_log2_rmse": float(
            (predicted_within_flat - target_within_flat).square().mean().sqrt()
        ),
        "target_sample_mean_variance_fraction": float(
            target_means.var(unbiased=False)
            / target_log2[valid].var(unbiased=False).clamp_min(1.0e-30)
        ),
    }

    bin_width = float(centers[1] - centers[0])
    payload: dict[str, Any] = {
        "schema": "parallel_categorical_scale_checkpoint_audit_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_cursor": cursor,
        "probe_logical_indices": cpu_batch["logical_indices"],
        "batch_size": int(args.batch_size),
        "valid_scale_targets": int(valid.sum()),
        "exact_layout_group_sizes": [len(group) for group in groups],
        "exact_layout_eligible_samples": int(eligible.sum()),
        "scale_bin_width_log2": bin_width,
        "scale_bin_adjacent_ratio": math.pow(2.0, bin_width),
        "loss_max_distance_denominator": int((centers.numel() - 1) ** 2),
        "evaluations": evaluations,
        "baselines": baselines,
        "prediction_decomposition": decomposition,
        "implementation_facts": {
            "teacher_scale_equals_encoder_input_scale_before_standardization": True,
            "teacher_scale_formula": "per-output max(abs(W))/7",
            "encoder_scale_tokenization": "standardized log2(scale), repeated across four p32 chunks per output",
            "scale_head": "bias-free Linear(1536,256) after pooling eight p16 children per output",
            "ordinal_component_reduction": "mean over valid scale outputs; total objective adds code_loss + scale_loss",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[scale-audit] stage=complete output={args.output}", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
