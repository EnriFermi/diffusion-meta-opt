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
    parser = argparse.ArgumentParser(description="Read-only scale-path checkpoint probe")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _safe_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.float().reshape(-1)
    y = right.float().reshape(-1)
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denominator) <= 1.0e-30:
        return float("nan")
    return float((x * y).sum() / denominator)


def _effective_rank(matrix: torch.Tensor) -> dict[str, float | int]:
    value = matrix.float()
    if min(value.shape) == 0:
        return {"rank_at_1e-3": 0, "effective_rank": 0.0, "top1_energy_fraction": 0.0}
    singular = torch.linalg.svdvals(value)
    energy = singular.square()
    total = energy.sum().clamp_min(1.0e-30)
    probability = energy / total
    entropy = -(probability * probability.clamp_min(1.0e-30).log()).sum()
    return {
        "rank_at_1e-3": int((singular > singular.max() * 1.0e-3).sum()),
        "effective_rank": float(entropy.exp()),
        "top1_energy_fraction": float(energy[0] / total),
    }


def _point_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    truth = target.float()[valid]
    estimate = prediction.float()[valid]
    residual = estimate - truth
    centered_truth = truth - truth.mean()
    return {
        "target_mean": float(truth.mean()),
        "prediction_mean": float(estimate.mean()),
        "target_std": float(truth.std(unbiased=False)),
        "prediction_std": float(estimate.std(unbiased=False)),
        "bias": float(residual.mean()),
        "mae_log2": float(residual.abs().mean()),
        "rmse_log2": float(residual.square().mean().sqrt()),
        "pearson": _safe_corr(truth, estimate),
        "r2": float(1.0 - residual.square().sum() / centered_truth.square().sum().clamp_min(1.0e-30)),
        "median_multiplicative_error": float(torch.exp2(residual.abs()).median()),
        "p90_multiplicative_error": float(torch.exp2(torch.quantile(residual.abs(), 0.90))),
    }


def _group_decomposition(
    target: torch.Tensor,
    prediction: torch.Tensor,
    groups: list[list[int]],
    valid: torch.Tensor,
) -> dict[str, Any]:
    sample_truth: list[torch.Tensor] = []
    sample_prediction: list[torch.Tensor] = []
    row_truth: list[torch.Tensor] = []
    row_prediction: list[torch.Tensor] = []
    interaction_truth: list[torch.Tensor] = []
    interaction_prediction: list[torch.Tensor] = []
    group_records: list[dict[str, Any]] = []
    truth_energy = {"sample": 0.0, "row": 0.0, "interaction": 0.0}
    prediction_energy = {"sample": 0.0, "row": 0.0, "interaction": 0.0}
    energy_weight = 0

    for group in groups:
        indices = torch.tensor(group, device=target.device)
        group_valid = valid.index_select(0, indices)
        active_rows = group_valid.all(dim=0)
        if not bool(active_rows.any()):
            continue
        truth = target.index_select(0, indices)[:, active_rows].float()
        estimate = prediction.index_select(0, indices)[:, active_rows].float()
        truth_global = truth.mean()
        estimate_global = estimate.mean()
        truth_sample = truth.mean(dim=1, keepdim=True) - truth_global
        estimate_sample = estimate.mean(dim=1, keepdim=True) - estimate_global
        truth_row = truth.mean(dim=0, keepdim=True) - truth_global
        estimate_row = estimate.mean(dim=0, keepdim=True) - estimate_global
        truth_interaction = truth - truth.mean(dim=1, keepdim=True) - truth.mean(dim=0, keepdim=True) + truth_global
        estimate_interaction = estimate - estimate.mean(dim=1, keepdim=True) - estimate.mean(dim=0, keepdim=True) + estimate_global

        sample_truth.append(truth.mean(dim=1))
        sample_prediction.append(estimate.mean(dim=1))
        row_truth.append(truth - truth.mean(dim=1, keepdim=True))
        row_prediction.append(estimate - estimate.mean(dim=1, keepdim=True))
        interaction_truth.append(truth_interaction)
        interaction_prediction.append(estimate_interaction)
        elements = truth.numel()
        energy_weight += elements
        for key, component_truth, component_prediction in (
            ("sample", truth_sample.expand_as(truth), estimate_sample.expand_as(estimate)),
            ("row", truth_row.expand_as(truth), estimate_row.expand_as(estimate)),
            ("interaction", truth_interaction, estimate_interaction),
        ):
            truth_energy[key] += float(component_truth.square().sum())
            prediction_energy[key] += float(component_prediction.square().sum())

        group_records.append(
            {
                "size": len(group),
                "active_rows": int(active_rows.sum()),
                "target_matrix_rank": _effective_rank(truth_interaction),
                "prediction_matrix_rank": _effective_rank(estimate_interaction),
                "sample_mean_correlation": _safe_corr(truth.mean(dim=1), estimate.mean(dim=1)),
                "row_centered_correlation": _safe_corr(
                    truth - truth.mean(dim=1, keepdim=True),
                    estimate - estimate.mean(dim=1, keepdim=True),
                ),
                "interaction_correlation": _safe_corr(truth_interaction, estimate_interaction),
            }
        )

    if not sample_truth:
        raise RuntimeError("no usable exact-layout groups")
    sample_truth_flat = torch.cat(sample_truth)
    sample_prediction_flat = torch.cat(sample_prediction)
    row_truth_flat = torch.cat([x.reshape(-1) for x in row_truth])
    row_prediction_flat = torch.cat([x.reshape(-1) for x in row_prediction])
    interaction_truth_flat = torch.cat([x.reshape(-1) for x in interaction_truth])
    interaction_prediction_flat = torch.cat([x.reshape(-1) for x in interaction_prediction])
    target_total = sum(truth_energy.values())
    prediction_total = sum(prediction_energy.values())
    return {
        "eligible_group_count": len(group_records),
        "eligible_sample_count": int(sample_truth_flat.numel()),
        "sample_mean_target_std": float(sample_truth_flat.std(unbiased=False)),
        "sample_mean_prediction_std": float(sample_prediction_flat.std(unbiased=False)),
        "sample_mean_correlation": _safe_corr(sample_truth_flat, sample_prediction_flat),
        "row_centered_target_rms": float(row_truth_flat.square().mean().sqrt()),
        "row_centered_prediction_rms": float(row_prediction_flat.square().mean().sqrt()),
        "row_centered_correlation": _safe_corr(row_truth_flat, row_prediction_flat),
        "interaction_target_rms": float(interaction_truth_flat.square().mean().sqrt()),
        "interaction_prediction_rms": float(interaction_prediction_flat.square().mean().sqrt()),
        "interaction_correlation": _safe_corr(interaction_truth_flat, interaction_prediction_flat),
        "target_centered_variance_fraction": {
            key: value / max(target_total, 1.0e-30) for key, value in truth_energy.items()
        },
        "prediction_centered_variance_fraction": {
            key: value / max(prediction_total, 1.0e-30) for key, value in prediction_energy.items()
        },
        "groups": group_records,
        "energy_weight": energy_weight,
    }


@torch.no_grad()
def _decode_scale(
    model: ParallelCategoricalGPTQWeightBottleneck,
    latent: torch.Tensor,
    tile_row: torch.Tensor,
    tile_col: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    token_valid: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _code_logits, scale_logits, children = model.decode_categorical(
            latent,
            tile_row,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid,
        )
    probabilities = scale_logits.float().softmax(dim=-1)
    hard_ids = scale_logits.float().argmax(dim=-1)
    hard_log2 = centers.index_select(0, hard_ids.reshape(-1)).view_as(hard_ids)
    expected_log2 = (probabilities * centers).sum(dim=-1)
    cdf = probabilities.cumsum(dim=-1)
    median_ids = (cdf < 0.5).sum(dim=-1).clamp_max(centers.numel() - 1)
    median_log2 = centers.index_select(0, median_ids.reshape(-1)).view_as(median_ids)
    return scale_logits, hard_log2, expected_log2, median_log2, children


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    print(f"[scale-probe] stage=checkpoint-open path={checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("schema") != config["schema"]:
        raise RuntimeError("checkpoint/config schema mismatch")
    checkpoint_step = int(checkpoint["step"])
    cursor = int(checkpoint["committed_logical_index"])
    if checkpoint_step * int(config["batch_size"]) != cursor:
        raise RuntimeError("checkpoint cursor is inconsistent with its step")
    print(f"[scale-probe] stage=checkpoint-gate step={checkpoint_step} cursor={cursor}", flush=True)

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
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])

    probe_config = dict(config)
    probe_config["operator_bank"] = dict(config["operator_bank"])
    logger = logging.getLogger("categorical-scale-checkpoint-probe")
    logger.setLevel(logging.WARNING)
    print(f"[scale-probe] stage=data batch={args.batch_size} start_cursor={cursor}", flush=True)
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
    target_codes, target_scales, _teacher = _mask_aware_gptq_targets(
        weights,
        activation,
        x_mask,
        d_in_mask,
        d_out_mask,
        bits=int(config["gptq"]["bits"]),
        damp_fraction=float(config["gptq"]["damp_fraction"]),
    )
    del target_codes
    centers = _scale_bin_centers(
        int(config["scale_bins"]["vocabulary_size"]),
        float(config["scale_bins"]["log2_min"]),
        float(config["scale_bins"]["log2_max"]),
        device=device,
    )
    scale_targets = _scale_class_targets(target_scales, centers)
    scale_valid = d_out_mask & d_in_mask.any(dim=-1)[:, None] & (target_scales > 1.0e-8 / 7.0)
    target_log2 = torch.log2(target_scales.float().clamp_min(1.0e-30))
    content, input_log_scale, token_valid = _prepare_normalized_inputs(
        weights,
        d_in_mask,
        d_out_mask,
        scale_mean=float(config["normalization"]["log2_scale_mean"]),
        scale_std=float(config["normalization"]["log2_scale_std"]),
    )
    groups = _exact_layout_groups(
        cpu_batch["tile_row"],
        cpu_batch["tile_col"],
        cpu_batch["d_in_mask"],
        cpu_batch["d_out_mask"],
        minimum_group_size=2,
    )
    eligible = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
    swap_index = torch.arange(args.batch_size, device=device)
    for group in groups:
        eligible[group] = True
        swap_index[group] = torch.tensor(group[1:] + group[:1], device=device)
    if not bool(eligible.any()):
        raise RuntimeError("probe batch contains no exact-layout groups")

    print("[scale-probe] stage=encode", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dist_patch_by_patch = model._encode_distribution_context(
            activation,
            sample_mask=x_mask,
        )
        latent, _telemetry = model.encode(
            content,
            input_log_scale,
            tile_row,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=dist_patch_by_patch,
        )

    global_mean = latent.mean(dim=(0, 1), keepdim=True)
    sample_mean = latent.mean(dim=1, keepdim=True)
    slot_mean = latent.mean(dim=0, keepdim=True)
    interaction = latent - sample_mean - slot_mean + global_mean
    variants = {
        "original_z": latent,
        "exact_layout_swapped_z": latent.index_select(0, swap_index),
        "sample_main_only_z": sample_mean.expand_as(latent),
        "slot_main_only_z": slot_mean.expand_as(latent),
        "no_interaction_z": sample_mean + slot_mean - global_mean,
        "interaction_plus_global_z": global_mean + interaction,
        "global_mean_z": global_mean.expand_as(latent),
        "zero_z": torch.zeros_like(latent),
    }

    records: list[dict[str, Any]] = []
    original_hard: torch.Tensor | None = None
    original_expected: torch.Tensor | None = None
    original_children: torch.Tensor | None = None
    print(f"[scale-probe] stage=decode variants={len(variants)}", flush=True)
    for name, variant_latent in variants.items():
        logits, hard_log2, expected_log2, median_log2, children = _decode_scale(
            model,
            variant_latent,
            tile_row,
            tile_col,
            d_in_mask,
            d_out_mask,
            token_valid,
            centers,
        )
        ordinal_loss, ordinal_stats = _ordered_cumulative_log_loss(logits, scale_targets, scale_valid)
        record: dict[str, Any] = {
            "variant": name,
            "ordinal_loss": float(ordinal_loss),
            "class_accuracy": float(ordinal_stats["accuracy"]),
            "off_by_one_accuracy": float(ordinal_stats["off_by_one_accuracy"]),
            "class_mae_bins": float(ordinal_stats["mean_absolute_bin_error"]),
            "normalized_entropy": float(ordinal_stats["prediction_entropy"]),
            "hard_argmax": _point_metrics(target_log2, hard_log2, scale_valid),
            "probability_mean": _point_metrics(target_log2, expected_log2, scale_valid),
            "probability_median": _point_metrics(target_log2, median_log2, scale_valid),
            "exact_layout_hard_decomposition": _group_decomposition(
                target_log2, hard_log2, groups, scale_valid
            ),
            "exact_layout_expected_decomposition": _group_decomposition(
                target_log2, expected_log2, groups, scale_valid
            ),
        }
        if original_hard is None:
            original_hard = hard_log2
            original_expected = expected_log2
            original_children = children
            record["change_from_original"] = None
        else:
            assert original_expected is not None
            record["change_from_original"] = {
                "hard_class_change_fraction_exact_layout": float(
                    (hard_log2[scale_valid & eligible[:, None]] != original_hard[scale_valid & eligible[:, None]])
                    .float()
                    .mean()
                ),
                "hard_log2_change_rms_exact_layout": float(
                    (hard_log2 - original_hard)[scale_valid & eligible[:, None]].square().mean().sqrt()
                ),
                "expected_log2_change_rms_exact_layout": float(
                    (expected_log2 - original_expected)[scale_valid & eligible[:, None]].square().mean().sqrt()
                ),
            }
        records.append(record)
        del logits, hard_log2, expected_log2, median_log2, children

    assert original_children is not None
    p16_component_valid = d_in_mask.view(args.batch_size, 8, 16)
    counts = p16_component_valid.sum(dim=-1).float()
    pooled = (
        original_children.float() * counts[:, None, :, None]
    ).sum(dim=2) / counts.sum(dim=-1).clamp_min(1.0)[:, None, None]
    pooled = pooled * d_out_mask[:, :, None].to(pooled.dtype)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        segment_logits = model.scale_head(model.scale_output_norm(original_children))
    segment_probability = segment_logits.float().softmax(dim=-1)
    segment_expected = (segment_probability * centers).sum(dim=-1)
    local_max_abs = weights.float().abs().view(args.batch_size, 8, 16, 128).amax(dim=2).permute(0, 2, 1)
    local_scale_log2 = torch.log2((local_max_abs / 7.0).clamp_min(1.0e-30))
    segment_valid = d_out_mask[:, :, None] & p16_component_valid.any(dim=-1)[:, None, :]
    repeated_global_target = target_log2[:, :, None].expand_as(segment_expected)
    pooling_analysis = {
        "pooled_hidden_rms": float(pooled[scale_valid].square().mean().sqrt()),
        "within_row_segment_hidden_rms": float(
            (original_children.float() - original_children.float().mean(dim=2, keepdim=True))[segment_valid]
            .square()
            .mean()
            .sqrt()
        ),
        "segment_expected_vs_local_p16_scale": _point_metrics(
            local_scale_log2, segment_expected, segment_valid
        ),
        "segment_expected_vs_global_row_scale": _point_metrics(
            repeated_global_target, segment_expected, segment_valid
        ),
        "within_row_segment_prediction_rms": float(
            (segment_expected - segment_expected.mean(dim=2, keepdim=True))[segment_valid]
            .square()
            .mean()
            .sqrt()
        ),
    }

    payload: dict[str, Any] = {
        "schema": "weightclip_parallel_categorical_scale_checkpoint_probe_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_cursor": cursor,
        "probe_logical_indices": cpu_batch["logical_indices"],
        "batch_size": args.batch_size,
        "scale_bin_width_log2": float(centers[1] - centers[0]),
        "exact_layout_group_sizes": [len(group) for group in groups],
        "exact_layout_eligible_samples": int(eligible.sum()),
        "variants": records,
        "pooling_analysis": pooling_analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[scale-probe] stage=complete output={args.output}", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
