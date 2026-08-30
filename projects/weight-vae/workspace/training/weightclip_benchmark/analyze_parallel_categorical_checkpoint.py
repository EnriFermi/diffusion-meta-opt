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

from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    _batch_from_stream,
    _exact_layout_groups,
    _model_config,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_parallel_categorical_gptq_700m_production import (
    CODE_ZERO_INDEX,
    ParallelCategoricalGPTQWeightBottleneck,
    _batch_stream,
    _decode_categorical_argmax,
    _load_config,
    _mask_aware_gptq_targets,
    _ordered_cumulative_log_loss,
    _relative_raw_nrmse,
    _scale_bin_centers,
    _scale_class_targets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only categorical checkpoint probe")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    value32 = value.float()
    if mask is None:
        return float(value32.square().mean().sqrt())
    expanded = mask.to(value32.device, dtype=torch.bool)
    while expanded.ndim < value32.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value32)
    return float(value32[expanded].square().mean().sqrt())


@torch.no_grad()
def _within_group_centered_rms(
    value: torch.Tensor,
    groups: list[list[int]],
    query_mask: torch.Tensor,
) -> float:
    squares = value.new_zeros((), dtype=torch.float32)
    count = 0
    for group in groups:
        group_value = value[group].float()
        centered = group_value - group_value.mean(dim=0, keepdim=True)
        valid = query_mask[group].bool()
        while valid.ndim < centered.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.expand_as(centered)
        squares += centered[valid].square().sum()
        count += int(valid.sum())
    if count == 0:
        return 0.0
    return float((squares / count).sqrt())


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
    )
    write = block._bounded_write(
        block.attn_out(attended.transpose(1, 2).reshape(batch, query_count, dim))
    )
    query_rms = _rms(query_state, query_mask)
    write_rms = _rms(write, query_mask)
    within_query = _within_group_centered_rms(query_state, groups, query_mask)
    within_write = _within_group_centered_rms(write, groups, query_mask)
    within_row_scores = scores - scores.mean(dim=-1, keepdim=True)
    return {
        "block": name,
        "attention_entropy_normalized": float(normalized_entropy[valid_attention].mean()),
        "attention_max_probability": float(
            probabilities.max(dim=-1).values[valid_attention].mean()
        ),
        "attention_logit_std": float(scores[valid_attention].std()),
        "attention_logit_within_row_std": float(
            within_row_scores[valid_attention].std()
        ),
        "query_rms": query_rms,
        "attention_write_rms": write_rms,
        "attention_write_over_query_rms": write_rms / max(query_rms, 1.0e-30),
        "exact_layout_query_centered_rms": within_query,
        "exact_layout_attention_write_centered_rms": within_write,
        "exact_layout_attention_write_centered_fraction": within_write
        / max(write_rms, 1.0e-30),
    }


@torch.no_grad()
def _latent_anova(latent: torch.Tensor) -> dict[str, float]:
    z = latent.float()
    global_mean = z.mean(dim=(0, 1), keepdim=True)
    sample_main = z.mean(dim=1, keepdim=True) - global_mean
    slot_main = z.mean(dim=0, keepdim=True) - global_mean
    interaction = z - z.mean(dim=1, keepdim=True) - z.mean(dim=0, keepdim=True) + global_mean
    sample_energy = sample_main.square().mean()
    slot_energy = slot_main.square().mean()
    interaction_energy = interaction.square().mean()
    centered_energy = sample_energy + slot_energy + interaction_energy
    return {
        "latent_rms": float(z.square().mean().sqrt()),
        "sample_main_rms": float(sample_energy.sqrt()),
        "slot_main_rms": float(slot_energy.sqrt()),
        "sample_slot_interaction_rms": float(interaction_energy.sqrt()),
        "sample_main_centered_variance_fraction": float(
            sample_energy / centered_energy.clamp_min(1.0e-30)
        ),
        "slot_main_centered_variance_fraction": float(
            slot_energy / centered_energy.clamp_min(1.0e-30)
        ),
        "sample_slot_interaction_centered_variance_fraction": float(
            interaction_energy / centered_energy.clamp_min(1.0e-30)
        ),
    }


@torch.no_grad()
def _evaluate_variant(
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
    weights: torch.Tensor,
    scale_centers: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[dict[str, float | str], torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        code_logits, scale_logits, _ = model.decode_categorical(
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
    prediction, hard_codes, hard_scales = _decode_categorical_argmax(
        code_logits, scale_logits, scale_centers, d_in_mask, d_out_mask
    )
    value_mask = (
        d_in_mask[:, :, None]
        & d_out_mask[:, None, :]
        & eligible[:, None, None]
    )
    result: dict[str, float | str] = {
        "variant": name,
        "code_ordinal_loss": float(code_loss),
        "scale_ordinal_loss": float(scale_loss),
        "total_ordinal_loss": float(code_loss + scale_loss),
        "code_accuracy": float(code_stats["accuracy"]),
        "code_off_by_one_accuracy": float(code_stats["off_by_one_accuracy"]),
        "code_mean_absolute_bin_error": float(code_stats["mean_absolute_bin_error"]),
        "scale_accuracy": float(scale_stats["accuracy"]),
        "scale_off_by_one_accuracy": float(scale_stats["off_by_one_accuracy"]),
        "scale_mean_absolute_bin_error": float(scale_stats["mean_absolute_bin_error"]),
        "hard_decode_raw_nrmse": float(_relative_raw_nrmse(prediction, weights, value_mask)),
    }
    return result, hard_codes, hard_scales


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config.resolve())
    print(f"[categorical-probe] stage=checkpoint-open path={args.checkpoint}", flush=True)
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("schema") != config["schema"]:
        raise RuntimeError("checkpoint/config schema mismatch")
    checkpoint_step = int(checkpoint["step"])
    cursor = int(checkpoint["committed_logical_index"])
    if checkpoint_step * int(config["batch_size"]) != cursor:
        raise RuntimeError("checkpoint cursor is inconsistent with its step")
    print(
        f"[categorical-probe] stage=checkpoint-gate step={checkpoint_step} cursor={cursor}",
        flush=True,
    )

    device = torch.device(config["device"])
    model_cfg = _model_config(config)
    model = ParallelCategoricalGPTQWeightBottleneck(
        model_cfg,
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
    logger = logging.getLogger("categorical-checkpoint-probe")
    logger.setLevel(logging.WARNING)
    print(
        f"[categorical-probe] stage=data batch={args.batch_size} start_cursor={cursor}",
        flush=True,
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
    target_codes, target_scales, _ = _mask_aware_gptq_targets(
        weights,
        activation,
        x_mask,
        d_in_mask,
        d_out_mask,
        bits=int(config["gptq"]["bits"]),
        damp_fraction=float(config["gptq"]["damp_fraction"]),
    )
    batch = int(args.batch_size)
    code_targets = target_codes.long().view(batch, 128, 8, 16) + CODE_ZERO_INDEX
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
        raise RuntimeError("probe batch contains no exact-layout group")

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
    print("[categorical-probe] stage=encode-and-attention", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        code_logits, scale_logits, latent, _ = model(
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
    model._run_cross_attention_block = original_runner  # type: ignore[method-assign]

    original_result, original_codes, original_scales = _evaluate_variant(
        "original_z",
        model,
        latent,
        tile_row,
        tile_col,
        d_in_mask,
        d_out_mask,
        token_valid,
        code_targets,
        scale_targets,
        code_valid,
        scale_valid,
        weights,
        scale_centers,
        eligible,
    )
    variants: list[dict[str, float | str]] = [original_result]
    hard_changes: dict[str, dict[str, float]] = {}
    for name, variant_latent in (
        ("exact_layout_swapped_z", latent.index_select(0, swap_index)),
        ("zero_z", torch.zeros_like(latent)),
    ):
        result, hard_codes, hard_scales = _evaluate_variant(
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
            weights,
            scale_centers,
            eligible,
        )
        variants.append(result)
        hard_changes[name] = {
            "code_argmax_change_fraction": float(
                (hard_codes[code_valid & eligible[:, None, None, None]]
                != original_codes[code_valid & eligible[:, None, None, None]])
                .float()
                .mean()
            ),
            "scale_argmax_change_fraction": float(
                (hard_scales[scale_valid & eligible[:, None]]
                != original_scales[scale_valid & eligible[:, None]])
                .float()
                .mean()
            ),
        }

    latent_stats = _latent_anova(latent)
    latent_stats["exact_layout_sample_centered_rms"] = _within_group_centered_rms(
        latent, groups, torch.ones((batch, latent.shape[1]), dtype=torch.bool, device=device)
    )
    latent_stats["exact_layout_sample_centered_over_latent_rms"] = (
        latent_stats["exact_layout_sample_centered_rms"]
        / max(latent_stats["latent_rms"], 1.0e-30)
    )

    payload: dict[str, Any] = {
        "schema": "weightclip_parallel_categorical_gptq_checkpoint_probe_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "checkpoint_cursor": cursor,
        "probe_logical_indices": cpu_batch["logical_indices"],
        "batch_size": batch,
        "exact_layout_group_sizes": [len(group) for group in groups],
        "exact_layout_eligible_samples": int(eligible.sum()),
        "latent_anova": latent_stats,
        "attention": attention_records,
        "ablation_variants": variants,
        "hard_prediction_changes": hard_changes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[categorical-probe] stage=complete output={args.output}", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
