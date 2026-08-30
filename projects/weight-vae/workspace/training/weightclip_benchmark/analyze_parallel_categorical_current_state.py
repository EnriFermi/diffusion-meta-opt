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
import torch.nn.functional as F

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.weightclip_benchmark.analyze_parallel_categorical_checkpoint import (
    _latent_anova,
    _rms,
)
from training.weightclip_benchmark.analyze_parallel_categorical_scale_checkpoint import (
    _group_decomposition,
    _point_metrics,
)
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    _batch_from_stream,
    _exact_layout_groups,
    _model_config,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_parallel_categorical_gptq_700m_production import (
    CODE_VOCAB_SIZE,
    CODE_ZERO_INDEX,
    ParallelCategoricalGPTQWeightBottleneck,
    _batch_stream,
    _load_config,
    _mask_aware_gptq_targets,
    _ordered_cumulative_log_loss,
    _relative_raw_nrmse,
    _scale_bin_centers,
    _scale_class_targets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only current-state mechanism probe for categorical GPTQ Weight-AE"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.to(value.device, dtype=torch.bool)
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    if not bool(expanded.any()):
        return 0.0
    return float(value.float()[expanded].square().mean().sqrt())


@torch.no_grad()
def _group_sample_rms(
    value: torch.Tensor,
    groups: list[list[int]],
    valid: torch.Tensor,
) -> tuple[float, float]:
    raw_sum = value.new_zeros((), dtype=torch.float32)
    centered_sum = value.new_zeros((), dtype=torch.float32)
    count = 0
    for group in groups:
        group_value = value[group].float()
        group_valid = valid[group].bool()
        expanded = group_valid
        while expanded.ndim < group_value.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(group_value)
        centered = group_value - group_value.mean(dim=0, keepdim=True)
        raw_sum += group_value[expanded].square().sum()
        centered_sum += centered[expanded].square().sum()
        count += int(expanded.sum())
    if count == 0:
        return 0.0, 0.0
    return float((raw_sum / count).sqrt()), float((centered_sum / count).sqrt())


def _rank_record(matrix: torch.Tensor) -> dict[str, float | int]:
    value = matrix.float()
    if value.ndim != 2 or min(value.shape) == 0:
        return {
            "rows": int(value.shape[0]),
            "columns": int(value.shape[1]),
            "rank_at_relative_1e_3": 0,
            "effective_rank": 0.0,
            "top1_energy_fraction": 0.0,
            "rank_90pct_energy": 0,
            "rank_95pct_energy": 0,
            "rank_99pct_energy": 0,
        }
    singular = torch.linalg.svdvals(value)
    energy = singular.square()
    total = energy.sum().clamp_min(1.0e-30)
    probability = energy / total
    cumulative = probability.cumsum(dim=0)

    def rank_for(fraction: float) -> int:
        return int(torch.searchsorted(cumulative, cumulative.new_tensor(fraction)).item() + 1)

    entropy = -(probability * probability.clamp_min(1.0e-30).log()).sum()
    return {
        "rows": int(value.shape[0]),
        "columns": int(value.shape[1]),
        "rank_at_relative_1e_3": int((singular > singular.max() * 1.0e-3).sum()),
        "effective_rank": float(entropy.exp()),
        "top1_energy_fraction": float(probability[0]),
        "rank_90pct_energy": rank_for(0.90),
        "rank_95pct_energy": rank_for(0.95),
        "rank_99pct_energy": rank_for(0.99),
    }


@torch.no_grad()
def _latent_rank_payload(latent: torch.Tensor) -> dict[str, Any]:
    z = latent.float()
    global_mean = z.mean(dim=(0, 1), keepdim=True)
    sample_main = z.mean(dim=1) - global_mean.squeeze(0).squeeze(0)
    slot_main = z.mean(dim=0) - global_mean.squeeze(0)
    interaction = (
        z
        - z.mean(dim=1, keepdim=True)
        - z.mean(dim=0, keepdim=True)
        + global_mean
    )
    centered = z - global_mean
    return {
        "all_centered": _rank_record(centered.reshape(-1, centered.shape[-1])),
        "sample_main": _rank_record(sample_main),
        "slot_main": _rank_record(slot_main),
        "sample_slot_interaction": _rank_record(
            interaction.reshape(-1, interaction.shape[-1])
        ),
    }


@torch.no_grad()
def _attention_record(
    name: str,
    block: torch.nn.Module,
    query_state: torch.Tensor,
    memory: torch.Tensor,
    query_mask: torch.Tensor,
    groups: list[list[int]],
) -> dict[str, float | str]:
    batch, query_count, dim = query_state.shape
    query_normalized = block.attn_norm(query_state)
    memory_normalized = block.attn_norm(memory)
    weight = block.qkv.weight
    query = F.linear(query_normalized, weight[:dim])
    key = F.linear(memory_normalized, weight[dim : 2 * dim])
    value = F.linear(memory_normalized, weight[2 * dim :])
    query = query.view(batch, query_count, block.heads, block.head_dim).transpose(1, 2)
    key = key.view(batch, memory.shape[1], block.heads, block.head_dim).transpose(1, 2)
    value = value.view(batch, memory.shape[1], block.heads, block.head_dim).transpose(1, 2)
    if block.bounded_cosine_attention:
        query = F.normalize(query.float(), dim=-1, eps=1.0e-6).to(query.dtype)
        key = F.normalize(key.float(), dim=-1, eps=1.0e-6).to(key.dtype)
        scores = (
            torch.matmul(query.float(), key.float().transpose(-1, -2))
            * float(block.attention_logit_scale)
        )
    else:
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(
            block.head_dim
        )
    probabilities = scores.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(float(memory.shape[1]))
    valid_attention = query_mask[:, None, :].expand(-1, block.heads, -1)
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        scale=(block.attention_logit_scale if block.bounded_cosine_attention else None),
    )
    write = block._bounded_write(
        block.attn_out(attended.transpose(1, 2).reshape(batch, query_count, dim))
    )

    query_raw, query_sample = _group_sample_rms(query_state, groups, query_mask)
    write_raw, write_sample = _group_sample_rms(write, groups, query_mask)
    probability_valid = query_mask[:, None, :].expand(-1, block.heads, -1)
    probability_raw, probability_sample = _group_sample_rms(
        probabilities,
        groups,
        probability_valid,
    )
    memory_valid = torch.ones(
        (batch, block.heads, memory.shape[1]), device=memory.device, dtype=torch.bool
    )
    key_raw, key_sample = _group_sample_rms(key, groups, memory_valid)
    value_raw, value_sample = _group_sample_rms(value, groups, memory_valid)

    return {
        "block": name,
        "attention_entropy_normalized": float(normalized_entropy[valid_attention].mean()),
        "attention_max_probability": float(
            probabilities.max(dim=-1).values[valid_attention].mean()
        ),
        "attention_logit_std": float(scores[valid_attention].std()),
        "query_rms": _rms(query_state, query_mask),
        "attention_write_rms": _rms(write, query_mask),
        "attention_write_over_query_rms": _rms(write, query_mask)
        / max(_rms(query_state, query_mask), 1.0e-30),
        "exact_layout_query_sample_centered_rms": query_sample,
        "exact_layout_query_sample_centered_variance_fraction": (
            query_sample / max(query_raw, 1.0e-30)
        )
        ** 2,
        "exact_layout_write_sample_centered_rms": write_sample,
        "exact_layout_write_sample_centered_variance_fraction": (
            write_sample / max(write_raw, 1.0e-30)
        )
        ** 2,
        "exact_layout_attention_map_sample_centered_rms": probability_sample,
        "exact_layout_attention_map_sample_centered_variance_fraction": (
            probability_sample / max(probability_raw, 1.0e-30)
        )
        ** 2,
        "exact_layout_key_sample_centered_variance_fraction": (
            key_sample / max(key_raw, 1.0e-30)
        )
        ** 2,
        "exact_layout_value_sample_centered_variance_fraction": (
            value_sample / max(value_raw, 1.0e-30)
        )
        ** 2,
    }


def _histogram(ids: torch.Tensor, valid: torch.Tensor, classes: int) -> dict[str, Any]:
    selected = ids[valid].long()
    counts = torch.bincount(selected, minlength=classes)
    fractions = counts.float() / counts.sum().clamp_min(1)
    top = torch.argsort(counts, descending=True)[: min(10, classes)]
    return {
        "counts": [int(item) for item in counts.cpu()],
        "fractions": [float(item) for item in fractions.cpu()],
        "unique_classes": int((counts > 0).sum()),
        "top_classes": [
            {
                "class": int(index),
                "count": int(counts[index]),
                "fraction": float(fractions[index]),
            }
            for index in top
        ],
    }


def _trim_group_decomposition(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "groups"}


def _code_soft_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    logits32 = logits.float()
    probabilities = logits32.softmax(dim=-1)
    class_axis = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
    expected = (probabilities * class_axis).sum(dim=-1)
    target_probability = probabilities.gather(-1, targets[..., None]).squeeze(-1)
    error = (expected - targets.float())[valid]
    return {
        "target_class_nll": float(-logits32.log_softmax(dim=-1).gather(
            -1, targets[..., None]
        ).squeeze(-1)[valid].mean()),
        "target_class_probability_mean": float(target_probability[valid].mean()),
        "expected_bin_mae": float(error.abs().mean()),
        "expected_bin_rmse": float(error.square().mean().sqrt()),
    }


def _weight_from_components(
    signed_codes: torch.Tensor,
    scales: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    batch = signed_codes.shape[0]
    component_valid = (
        d_out_mask[:, :, None, None]
        & d_in_mask.view(batch, 1, 8, 16)
    )
    rows = signed_codes.float() * scales.float()[:, :, None, None]
    rows = rows * component_valid.to(rows.dtype)
    return rows.reshape(batch, 128, 128).transpose(1, 2).contiguous()


def _operator_split_metrics(
    prediction: torch.Tensor,
    activation: torch.Tensor,
    weights: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> dict[str, float]:
    calibration = x_mask.clone()
    calibration[:, 256:] = False
    heldout = x_mask.clone()
    heldout[:, :256] = False
    result: dict[str, float] = {}
    for name, mask in (
        ("full", x_mask),
        ("calibration", calibration),
        ("heldout", heldout),
    ):
        result[name] = float(
            BigWeightVAELossMixin.operator_recon_loss(
                activation,
                weights,
                prediction,
                x_mask=mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
            )
        )
    return result


@torch.no_grad()
def _decode_variant(
    name: str,
    model: ParallelCategoricalGPTQWeightBottleneck,
    latent: torch.Tensor,
    tile_row: torch.Tensor,
    tile_col: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    token_valid: torch.Tensor,
    code_targets: torch.Tensor,
    scale_targets: torch.Tensor,
    code_valid: torch.Tensor,
    scale_valid: torch.Tensor,
    eligible: torch.Tensor,
    scale_centers: torch.Tensor,
    target_log2_scale: torch.Tensor,
    groups: list[list[int]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        code_logits, scale_logits, _children = model.decode_categorical(
            latent,
            tile_row,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid,
        )
    eligible_code = code_valid & eligible[:, None, None, None]
    eligible_scale = scale_valid & eligible[:, None]
    code_loss, code_stats = _ordered_cumulative_log_loss(
        code_logits, code_targets, eligible_code
    )
    scale_loss, scale_stats = _ordered_cumulative_log_loss(
        scale_logits, scale_targets, eligible_scale
    )
    code_probability = code_logits.float().softmax(dim=-1)
    code_axis = torch.arange(CODE_VOCAB_SIZE, device=latent.device, dtype=torch.float32)
    code_expected_ids = (code_probability * code_axis).sum(dim=-1)
    code_hard_ids = code_logits.float().argmax(dim=-1)
    scale_probability = scale_logits.float().softmax(dim=-1)
    scale_hard_ids = scale_logits.float().argmax(dim=-1)
    scale_hard_log2 = scale_centers[scale_hard_ids]
    scale_expected_log2 = (scale_probability * scale_centers).sum(dim=-1)
    scale_median_ids = (scale_probability.cumsum(dim=-1) < 0.5).sum(dim=-1).clamp_max(
        scale_centers.numel() - 1
    )
    scale_median_log2 = scale_centers[scale_median_ids]

    hard_scale_decomposition = _trim_group_decomposition(
        _group_decomposition(target_log2_scale, scale_hard_log2, groups, scale_valid)
    )
    expected_scale_decomposition = _trim_group_decomposition(
        _group_decomposition(target_log2_scale, scale_expected_log2, groups, scale_valid)
    )
    record: dict[str, Any] = {
        "variant": name,
        "code_ordinal_loss": float(code_loss),
        "scale_ordinal_loss": float(scale_loss),
        "total_ordinal_loss": float(code_loss + scale_loss),
        "code_accuracy": float(code_stats["accuracy"]),
        "code_off_by_one_accuracy": float(code_stats["off_by_one_accuracy"]),
        "code_hard_mae_bins": float(code_stats["mean_absolute_bin_error"]),
        "code_entropy_normalized": float(code_stats["prediction_entropy"]),
        "code_soft": _code_soft_metrics(code_logits, code_targets, eligible_code),
        "scale_accuracy": float(scale_stats["accuracy"]),
        "scale_off_by_one_accuracy": float(scale_stats["off_by_one_accuracy"]),
        "scale_hard_mae_bins": float(scale_stats["mean_absolute_bin_error"]),
        "scale_entropy_normalized": float(scale_stats["prediction_entropy"]),
        "scale_hard_log2": _point_metrics(
            target_log2_scale, scale_hard_log2, eligible_scale
        ),
        "scale_expected_log2": _point_metrics(
            target_log2_scale, scale_expected_log2, eligible_scale
        ),
        "scale_median_log2": _point_metrics(
            target_log2_scale, scale_median_log2, eligible_scale
        ),
        "scale_hard_exact_layout_decomposition": hard_scale_decomposition,
        "scale_expected_exact_layout_decomposition": expected_scale_decomposition,
    }
    tensors = {
        "code_hard_ids": code_hard_ids,
        "code_expected_ids": code_expected_ids,
        "scale_hard_ids": scale_hard_ids,
        "scale_hard_log2": scale_hard_log2,
        "scale_expected_log2": scale_expected_log2,
        "scale_median_log2": scale_median_log2,
    }
    return record, tensors


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    config = _load_config(config_path)
    checkpoint_stat = checkpoint_path.stat()
    print(
        "[current-state-probe] stage=checkpoint-open "
        f"path={checkpoint_path} size={checkpoint_stat.st_size} "
        f"mtime_ns={checkpoint_stat.st_mtime_ns}",
        flush=True,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("schema") != config["schema"]:
        raise RuntimeError("checkpoint/config schema mismatch")
    checkpoint_step = int(checkpoint["step"])
    checkpoint_cursor = int(checkpoint["committed_logical_index"])
    if checkpoint_step * int(config["batch_size"]) != checkpoint_cursor:
        raise RuntimeError("checkpoint cursor is inconsistent with step")
    print(
        "[current-state-probe] stage=checkpoint-gate "
        f"step={checkpoint_step} cursor={checkpoint_cursor} device={config['device']} "
        f"dtype=bfloat16 batch={args.batch_size} output={output_path}",
        flush=True,
    )

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
    probe_config["operator_bank"]["loader_workers"] = 0
    logger = logging.getLogger("parallel-categorical-current-state-probe")
    logger.setLevel(logging.WARNING)
    print(
        "[current-state-probe] stage=data "
        f"start_cursor={checkpoint_cursor} count={args.batch_size}",
        flush=True,
    )
    with ExitStack() as stack:
        stream = _batch_stream(probe_config, checkpoint_cursor, logger, stack)
        cpu_batch = _batch_from_stream(stream, int(args.batch_size))

    weights = cpu_batch["W"].to(device)
    activation = cpu_batch["X"].to(device)
    x_mask = cpu_batch["x_mask"].to(device)
    d_in_mask = cpu_batch["d_in_mask"].to(device)
    d_out_mask = cpu_batch["d_out_mask"].to(device)
    tile_row = cpu_batch["tile_row"].to(device)
    tile_col = cpu_batch["tile_col"].to(device)
    target_codes_signed, target_scales, _teacher = _mask_aware_gptq_targets(
        weights,
        activation,
        x_mask,
        d_in_mask,
        d_out_mask,
        bits=int(config["gptq"]["bits"]),
        damp_fraction=float(config["gptq"]["damp_fraction"]),
    )
    batch = int(args.batch_size)
    code_targets = target_codes_signed.long().view(batch, 128, 8, 16) + CODE_ZERO_INDEX
    scale_centers = _scale_bin_centers(
        int(config["scale_bins"]["vocabulary_size"]),
        float(config["scale_bins"]["log2_min"]),
        float(config["scale_bins"]["log2_max"]),
        device=device,
    )
    scale_targets = _scale_class_targets(target_scales, scale_centers)
    code_valid = d_out_mask[:, :, None, None] & d_in_mask.view(batch, 1, 8, 16)
    scale_valid = (
        d_out_mask
        & d_in_mask.any(dim=-1)[:, None]
        & (target_scales > 1.0e-8 / 7.0)
    )
    target_log2_scale = torch.log2(target_scales.float().clamp_min(1.0e-30))
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
    eligible = torch.zeros(batch, dtype=torch.bool, device=device)
    swap_index = torch.arange(batch, device=device)
    for group in groups:
        eligible[group] = True
        swap_index[group] = torch.tensor(group[1:] + group[:1], device=device)
    if not bool(eligible.any()):
        raise RuntimeError("probe batch contains no exact-layout groups")

    print(
        "[current-state-probe] stage=encode-and-attention "
        f"groups={len(groups)} eligible={int(eligible.sum())}",
        flush=True,
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dist = model._encode_distribution_context(activation, sample_mask=x_mask)
        latent, _telemetry = model.encode(
            content,
            input_log_scale,
            tile_row,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=dist,
        )

    block_names = {
        id(block): f"decoder_block_{index + 1}"
        for index, block in enumerate(model.decoder_blocks)
    }
    block_names[id(model.categorical_tail)] = "categorical_tail"
    attention_records: list[dict[str, float | str]] = []
    original_runner = model._run_cross_attention_block

    def instrumented_runner(
        block: torch.nn.Module,
        query_state: torch.Tensor,
        memory: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_records.append(
            _attention_record(
                block_names[id(block)],
                block,
                query_state,
                memory,
                query_mask,
                groups,
            )
        )
        return original_runner(block, query_state, memory, query_mask)

    model._run_cross_attention_block = instrumented_runner  # type: ignore[method-assign]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.decode_categorical(
            latent,
            tile_row,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid,
        )
    model._run_cross_attention_block = original_runner  # type: ignore[method-assign]

    global_mean = latent.mean(dim=(0, 1), keepdim=True)
    sample_mean = latent.mean(dim=1, keepdim=True)
    slot_mean = latent.mean(dim=0, keepdim=True)
    interaction = latent - sample_mean - slot_mean + global_mean
    variants = {
        "original_z": latent,
        "exact_layout_swapped_z": latent.index_select(0, swap_index),
        "sample_mean_broadcast_z": sample_mean.expand_as(latent),
        "no_interaction_z": sample_mean + slot_mean - global_mean,
        "interaction_plus_global_z": global_mean + interaction,
        "global_mean_z": global_mean.expand_as(latent),
        "zero_z": torch.zeros_like(latent),
    }
    print(
        f"[current-state-probe] stage=ablation-decode variants={len(variants)}",
        flush=True,
    )
    variant_records: list[dict[str, Any]] = []
    variant_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for index, (name, variant_latent) in enumerate(variants.items(), start=1):
        print(
            f"[current-state-probe] stage=ablation-decode variant={name} "
            f"index={index}/{len(variants)}",
            flush=True,
        )
        record, tensors = _decode_variant(
            name,
            model,
            variant_latent,
            tile_row,
            tile_col,
            d_in_mask,
            d_out_mask,
            token_valid,
            code_targets,
            scale_targets,
            code_valid,
            scale_valid,
            eligible,
            scale_centers,
            target_log2_scale,
            groups,
        )
        variant_records.append(record)
        variant_tensors[name] = tensors

    original_tensors = variant_tensors["original_z"]
    original_codes = original_tensors["code_hard_ids"]
    original_scales = original_tensors["scale_hard_ids"]
    eligible_code = code_valid & eligible[:, None, None, None]
    eligible_scale = scale_valid & eligible[:, None]
    hard_changes: dict[str, dict[str, float]] = {}
    for name, tensors in variant_tensors.items():
        if name == "original_z":
            continue
        hard_changes[name] = {
            "code_argmax_change_fraction": float(
                (tensors["code_hard_ids"][eligible_code] != original_codes[eligible_code])
                .float()
                .mean()
            ),
            "scale_argmax_change_fraction": float(
                (tensors["scale_hard_ids"][eligible_scale] != original_scales[eligible_scale])
                .float()
                .mean()
            ),
            "code_expected_id_change_rms": float(
                (
                    tensors["code_expected_ids"] - original_tensors["code_expected_ids"]
                )[eligible_code]
                .square()
                .mean()
                .sqrt()
            ),
            "scale_expected_log2_change_rms": float(
                (
                    tensors["scale_expected_log2"]
                    - original_tensors["scale_expected_log2"]
                )[eligible_scale]
                .square()
                .mean()
                .sqrt()
            ),
        }

    prediction_histograms = {
        "code_target_exact_layout": _histogram(
            code_targets, eligible_code, CODE_VOCAB_SIZE
        ),
        "code_prediction_exact_layout": _histogram(
            original_codes, eligible_code, CODE_VOCAB_SIZE
        ),
        "scale_target_exact_layout": _histogram(
            scale_targets, eligible_scale, scale_centers.numel()
        ),
        "scale_prediction_exact_layout": _histogram(
            original_scales, eligible_scale, scale_centers.numel()
        ),
        "code_target_nonzero_fraction": float(
            (code_targets[eligible_code] != CODE_ZERO_INDEX).float().mean()
        ),
        "code_prediction_nonzero_fraction": float(
            (original_codes[eligible_code] != CODE_ZERO_INDEX).float().mean()
        ),
    }

    pred_hard_codes = original_codes.float() - float(CODE_ZERO_INDEX)
    pred_expected_codes = original_tensors["code_expected_ids"] - float(CODE_ZERO_INDEX)
    target_codes_patched = target_codes_signed.float().view(batch, 128, 8, 16)
    pred_hard_scales = torch.exp2(original_tensors["scale_hard_log2"])
    pred_expected_scales = torch.exp2(original_tensors["scale_expected_log2"])
    pred_median_scales = torch.exp2(original_tensors["scale_median_log2"])
    target_binned_scales = torch.exp2(scale_centers[scale_targets])
    crossed_predictions = {
        "pred_hard_code_pred_hard_scale": _weight_from_components(
            pred_hard_codes, pred_hard_scales, d_in_mask, d_out_mask
        ),
        "pred_hard_code_pred_expected_scale": _weight_from_components(
            pred_hard_codes, pred_expected_scales, d_in_mask, d_out_mask
        ),
        "pred_hard_code_pred_median_scale": _weight_from_components(
            pred_hard_codes, pred_median_scales, d_in_mask, d_out_mask
        ),
        "pred_expected_code_pred_expected_scale": _weight_from_components(
            pred_expected_codes, pred_expected_scales, d_in_mask, d_out_mask
        ),
        "pred_hard_code_oracle_continuous_scale": _weight_from_components(
            pred_hard_codes, target_scales, d_in_mask, d_out_mask
        ),
        "pred_expected_code_oracle_continuous_scale": _weight_from_components(
            pred_expected_codes, target_scales, d_in_mask, d_out_mask
        ),
        "oracle_code_pred_hard_scale": _weight_from_components(
            target_codes_patched, pred_hard_scales, d_in_mask, d_out_mask
        ),
        "oracle_code_pred_expected_scale": _weight_from_components(
            target_codes_patched, pred_expected_scales, d_in_mask, d_out_mask
        ),
        "oracle_code_pred_median_scale": _weight_from_components(
            target_codes_patched, pred_median_scales, d_in_mask, d_out_mask
        ),
        "oracle_code_oracle_binned_scale": _weight_from_components(
            target_codes_patched, target_binned_scales, d_in_mask, d_out_mask
        ),
        "oracle_code_oracle_continuous_scale": _weight_from_components(
            target_codes_patched, target_scales, d_in_mask, d_out_mask
        ),
        "zero_baseline": torch.zeros_like(weights),
    }
    value_mask = d_in_mask[:, :, None] & d_out_mask[:, None, :]
    crossed_metrics: dict[str, Any] = {}
    print(
        f"[current-state-probe] stage=crossed-oracle variants={len(crossed_predictions)}",
        flush=True,
    )
    operator_names = {
        "pred_hard_code_pred_hard_scale",
        "pred_hard_code_oracle_continuous_scale",
        "oracle_code_pred_expected_scale",
        "oracle_code_oracle_continuous_scale",
        "zero_baseline",
    }
    for name, prediction in crossed_predictions.items():
        record: dict[str, Any] = {
            "raw_nrmse": float(_relative_raw_nrmse(prediction, weights, value_mask))
        }
        if name in operator_names:
            record["operator_split"] = _operator_split_metrics(
                prediction,
                activation,
                weights,
                x_mask,
                d_in_mask,
                d_out_mask,
            )
        crossed_metrics[name] = record

    latent_payload = _latent_anova(latent)
    latent_payload["rank"] = _latent_rank_payload(latent)
    latent_raw, latent_sample = _group_sample_rms(
        latent,
        groups,
        torch.ones((batch, latent.shape[1]), device=device, dtype=torch.bool),
    )
    latent_payload["exact_layout_sample_centered_rms"] = latent_sample
    latent_payload["exact_layout_sample_centered_variance_fraction"] = (
        latent_sample / max(latent_raw, 1.0e-30)
    ) ** 2

    payload: dict[str, Any] = {
        "schema": "weightclip_parallel_categorical_current_state_probe_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_stat_at_open": {
            "size": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
        },
        "checkpoint_step": checkpoint_step,
        "checkpoint_cursor": checkpoint_cursor,
        "cursor_contract_valid": True,
        "config": str(config_path),
        "device": str(device),
        "dtype": "bfloat16_autocast_float32_metrics",
        "batch_size": batch,
        "probe_logical_indices": cpu_batch["logical_indices"],
        "exact_layout_group_sizes": [len(group) for group in groups],
        "exact_layout_eligible_samples": int(eligible.sum()),
        "latent": latent_payload,
        "attention": attention_records,
        "ablation_variants": variant_records,
        "hard_prediction_changes": hard_changes,
        "prediction_histograms": prediction_histograms,
        "crossed_oracle_reconstruction": crossed_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "[current-state-probe] stage=complete "
        f"output={output_path} checkpoint_step={checkpoint_step}",
        flush=True,
    )
    compact = {
        "checkpoint_step": checkpoint_step,
        "latent": latent_payload,
        "attention": attention_records,
        "ablation_variants": variant_records,
        "hard_prediction_changes": hard_changes,
        "prediction_histograms": prediction_histograms,
        "crossed_oracle_reconstruction": crossed_metrics,
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
