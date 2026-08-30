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

from training.weightclip_benchmark.analyze_parallel_categorical_checkpoint import (
    _latent_anova,
)
from training.weightclip_benchmark.analyze_parallel_categorical_current_state import (
    _latent_rank_payload,
    _weight_from_components,
)
from training.weightclip_benchmark.analyze_parallel_categorical_scale_checkpoint import (
    _group_decomposition,
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
    _decode_categorical_argmax,
    _detached_old_loss_diagnostics,
    _load_config,
    _mask_aware_gptq_targets,
    _ordered_cumulative_log_loss,
    _relative_raw_nrmse,
    _scale_bin_centers,
    _scale_class_targets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired read-only 90k/100k mechanism probe for categorical GPTQ"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-90k", type=Path, required=True)
    parser.add_argument("--checkpoint-100k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-cursor", type=int, default=3_136_000)
    return parser.parse_args()


def _ordinal_per_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits32 = logits.float()
    classes = logits32.shape[-1]
    log_normalizer = torch.logsumexp(logits32, dim=-1, keepdim=True)
    log_le = torch.logcumsumexp(logits32, dim=-1)[..., :-1] - log_normalizer
    log_gt = (
        torch.flip(
            torch.logcumsumexp(torch.flip(logits32, dims=(-1,)), dim=-1),
            dims=(-1,),
        )[..., 1:]
        - log_normalizer
    )
    thresholds = torch.arange(classes - 1, device=logits.device)
    target_expanded = targets[..., None]
    target_is_le = target_expanded <= thresholds
    threshold_nll = -torch.where(target_is_le, log_le, log_gt)
    threshold_weight = (
        (2 * (thresholds - target_expanded) + 1).abs().float()
        / float((classes - 1) ** 2)
    )
    return (threshold_weight * threshold_nll).sum(dim=-1)


def _fraction_record(value: torch.Tensor) -> dict[str, Any]:
    values = value.detach().float().cpu()
    total = values.sum().clamp_min(1.0e-30)
    fractions = values / total
    entropy = -(fractions * fractions.clamp_min(1.0e-30).log()).sum()
    return {
        "counts_or_mass": [float(item) for item in values],
        "fractions": [float(item) for item in fractions],
        "used_nonzero": int((values > 0).sum()),
        "used_ge_0p1pct": int((fractions >= 1.0e-3).sum()),
        "effective_slots": float(entropy.exp()),
        "max_fraction": float(fractions.max()),
    }


@torch.no_grad()
def _attention_owner_record(
    name: str,
    block: torch.nn.Module,
    query_state: torch.Tensor,
    memory: torch.Tensor,
    query_mask: torch.Tensor,
    groups: list[list[int]],
) -> dict[str, Any]:
    batch, query_count, dim = query_state.shape
    query_normalized = block.attn_norm(query_state)
    memory_normalized = block.attn_norm(memory)
    weight = block.qkv.weight
    query = F.linear(query_normalized, weight[:dim])
    key = F.linear(memory_normalized, weight[dim : 2 * dim])
    query = query.view(batch, query_count, block.heads, block.head_dim).transpose(1, 2)
    key = key.view(batch, memory.shape[1], block.heads, block.head_dim).transpose(1, 2)
    if block.bounded_cosine_attention:
        query = F.normalize(query.float(), dim=-1, eps=1.0e-6).to(query.dtype)
        key = F.normalize(key.float(), dim=-1, eps=1.0e-6).to(key.dtype)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * float(
            block.attention_logit_scale
        )
    else:
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(
            block.head_dim
        )
    probabilities = scores.softmax(dim=-1)
    winners = probabilities.argmax(dim=-1)
    valid = query_mask[:, None, :].expand(-1, block.heads, -1)
    winner_counts = torch.bincount(winners[valid], minlength=memory.shape[1]).float()
    soft_mass = (probabilities * valid[..., None].float()).sum(dim=(0, 1, 2))
    entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum(dim=-1)

    per_sample_used = []
    for sample in range(batch):
        sample_valid = valid[sample]
        per_sample_used.append(float(torch.unique(winners[sample][sample_valid]).numel()))
    per_head_used = []
    for head in range(block.heads):
        head_valid = valid[:, head]
        per_head_used.append(float(torch.unique(winners[:, head][head_valid]).numel()))

    group_agreement_numerator = scores.new_zeros(())
    group_agreement_denominator = 0
    for group in groups:
        group_winners = winners[group]
        group_valid = valid[group]
        mode = torch.mode(group_winners, dim=0).values
        comparable = group_valid.all(dim=0)
        group_agreement_numerator += (
            (group_winners == mode[None]) & comparable[None]
        ).float().sum()
        group_agreement_denominator += int(comparable.sum()) * len(group)

    return {
        "block": name,
        "latent_slots": int(memory.shape[1]),
        "valid_head_queries": int(valid.sum()),
        "attention_entropy_normalized": float(
            (entropy[valid] / math.log(float(memory.shape[1]))).mean()
        ),
        "attention_max_probability": float(probabilities.max(dim=-1).values[valid].mean()),
        "attention_logit_std": float(scores[valid].std()),
        "winner_load": _fraction_record(winner_counts),
        "soft_mass_load": _fraction_record(soft_mass),
        "per_sample_winner_slots_mean": float(torch.tensor(per_sample_used).mean()),
        "per_sample_winner_slots_min": float(min(per_sample_used)),
        "per_sample_winner_slots_max": float(max(per_sample_used)),
        "per_head_winner_slots_mean": float(torch.tensor(per_head_used).mean()),
        "per_head_winner_slots_min": float(min(per_head_used)),
        "per_head_winner_slots_max": float(max(per_head_used)),
        "exact_layout_winner_agreement_to_group_mode": float(
            group_agreement_numerator / max(group_agreement_denominator, 1)
        ),
    }


def _run_block_with_temperature(
    block: torch.nn.Module,
    query_state: torch.Tensor,
    memory: torch.Tensor,
    query_valid_mask: torch.Tensor,
    factor: float,
) -> torch.Tensor:
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
        scale = float(block.attention_logit_scale) * factor
    else:
        scale = factor / math.sqrt(block.head_dim)
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        scale=scale,
    )
    attention_write = block.attn_out(
        attended.transpose(1, 2).reshape(batch, query_count, dim)
    )
    query_state = query_state + block._bounded_write(attention_write)
    left, right = block.mlp_in(block.mlp_norm(query_state)).chunk(2, dim=-1)
    hidden = F.silu(left) * right
    if block.bounded_swiglu_hidden:
        hidden_rms_sq = hidden.float().square().mean(dim=-1, keepdim=True)
        hidden = hidden * torch.rsqrt(1.0 + hidden_rms_sq).to(hidden.dtype)
    query_state = query_state + block._bounded_write(block.mlp_out(hidden))
    return query_state * query_valid_mask.unsqueeze(-1).to(query_state.dtype)


def _decode(
    model: ParallelCategoricalGPTQWeightBottleneck,
    latent: torch.Tensor,
    shared: dict[str, Any],
    *,
    factor_by_block: dict[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_runner = model._run_cross_attention_block
    if factor_by_block:
        def temperature_runner(
            block: torch.nn.Module,
            query_state: torch.Tensor,
            memory: torch.Tensor,
            query_mask: torch.Tensor,
        ) -> torch.Tensor:
            return _run_block_with_temperature(
                block,
                query_state,
                memory,
                query_mask,
                factor_by_block.get(id(block), 1.0),
            )

        model._run_cross_attention_block = temperature_runner  # type: ignore[method-assign]
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            code_logits, scale_logits, _ = model.decode_categorical(
                latent,
                shared["tile_row"],
                d_in_mask=shared["d_in_mask"],
                d_out_mask=shared["d_out_mask"],
                tile_col=shared["tile_col"],
                token_valid_mask=shared["token_valid"],
            )
    finally:
        model._run_cross_attention_block = original_runner  # type: ignore[method-assign]
    return code_logits, scale_logits


@torch.no_grad()
def _decode_metrics(
    code_logits: torch.Tensor,
    scale_logits: torch.Tensor,
    shared: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    eligible_code = shared["code_valid"] & shared["eligible"][:, None, None, None]
    eligible_scale = shared["scale_valid"] & shared["eligible"][:, None]
    code_loss, code_stats = _ordered_cumulative_log_loss(
        code_logits, shared["code_targets"], eligible_code
    )
    scale_loss, scale_stats = _ordered_cumulative_log_loss(
        scale_logits, shared["scale_targets"], eligible_scale
    )
    hard_prediction, hard_code_ids, hard_scale_ids = _decode_categorical_argmax(
        code_logits,
        scale_logits,
        shared["scale_centers"],
        shared["d_in_mask"],
        shared["d_out_mask"],
    )
    code_probabilities = code_logits.float().softmax(dim=-1)
    code_axis = torch.arange(CODE_VOCAB_SIZE, device=code_logits.device).float()
    expected_codes = (code_probabilities * code_axis).sum(dim=-1) - CODE_ZERO_INDEX
    scale_probabilities = scale_logits.float().softmax(dim=-1)
    expected_log2_scale = (
        scale_probabilities * shared["scale_centers"]
    ).sum(dim=-1)
    expected_scales = torch.exp2(expected_log2_scale)
    expected_prediction = _weight_from_components(
        expected_codes,
        expected_scales,
        shared["d_in_mask"],
        shared["d_out_mask"],
    )
    oracle_code_expected_scale = _weight_from_components(
        shared["target_codes_signed"].view(-1, 128, 8, 16),
        expected_scales,
        shared["d_in_mask"],
        shared["d_out_mask"],
    )
    hard_code_oracle_scale = _weight_from_components(
        hard_code_ids.float() - CODE_ZERO_INDEX,
        shared["target_scales"],
        shared["d_in_mask"],
        shared["d_out_mask"],
    )
    value_mask_full = shared["d_in_mask"][:, :, None] & shared["d_out_mask"][:, None, :]
    value_mask_eligible = value_mask_full & shared["eligible"][:, None, None]
    target_probability = code_probabilities.gather(
        -1, shared["code_targets"][..., None]
    ).squeeze(-1)
    diagnostics = _detached_old_loss_diagnostics(
        shared["activation"],
        shared["weights"],
        hard_prediction,
        shared["latent_for_diagnostics"],
        shared["x_mask"],
        shared["d_in_mask"],
        shared["d_out_mask"],
        shared["layout_groups"],
        shared["config"],
    )
    scale_decomposition = _group_decomposition(
        shared["target_log2_scale"],
        expected_log2_scale,
        shared["groups"],
        shared["scale_valid"],
    )
    scale_decomposition = {
        key: value for key, value in scale_decomposition.items() if key != "groups"
    }
    record = {
        "code_ordinal_loss": float(code_loss),
        "scale_ordinal_loss": float(scale_loss),
        "total_ordinal_loss": float(code_loss + scale_loss),
        "code_accuracy": float(code_stats["accuracy"]),
        "code_off_by_one_accuracy": float(code_stats["off_by_one_accuracy"]),
        "code_hard_mae_bins": float(code_stats["mean_absolute_bin_error"]),
        "code_entropy_normalized": float(code_stats["prediction_entropy"]),
        "code_target_probability": float(target_probability[eligible_code].mean()),
        "scale_accuracy": float(scale_stats["accuracy"]),
        "scale_off_by_one_accuracy": float(scale_stats["off_by_one_accuracy"]),
        "scale_hard_mae_bins": float(scale_stats["mean_absolute_bin_error"]),
        "scale_entropy_normalized": float(scale_stats["prediction_entropy"]),
        "hard_nrmse_full": float(
            _relative_raw_nrmse(hard_prediction, shared["weights"], value_mask_full)
        ),
        "hard_nrmse_exact_layout": float(
            _relative_raw_nrmse(hard_prediction, shared["weights"], value_mask_eligible)
        ),
        "expected_nrmse_full": float(
            _relative_raw_nrmse(expected_prediction, shared["weights"], value_mask_full)
        ),
        "expected_nrmse_exact_layout": float(
            _relative_raw_nrmse(expected_prediction, shared["weights"], value_mask_eligible)
        ),
        "hard_code_oracle_scale_nrmse_full": float(
            _relative_raw_nrmse(hard_code_oracle_scale, shared["weights"], value_mask_full)
        ),
        "oracle_code_expected_scale_nrmse_full": float(
            _relative_raw_nrmse(oracle_code_expected_scale, shared["weights"], value_mask_full)
        ),
        "scale_expected_exact_layout_decomposition": scale_decomposition,
        "old_diagnostics": diagnostics,
    }
    tensors = {
        "hard_code_ids": hard_code_ids,
        "hard_scale_ids": hard_scale_ids,
        "hard_prediction": hard_prediction,
        "code_probabilities": code_probabilities,
        "scale_probabilities": scale_probabilities,
        "code_per_target_ordinal": _ordinal_per_target(
            code_logits, shared["code_targets"]
        ),
        "scale_per_target_ordinal": _ordinal_per_target(
            scale_logits, shared["scale_targets"]
        ),
    }
    return record, tensors


def _load_model(checkpoint_path: Path, config: dict[str, Any], device: torch.device):
    stat = checkpoint_path.stat()
    print(
        f"[paired-probe] stage=checkpoint-open path={checkpoint_path} "
        f"size={stat.st_size} mtime_ns={stat.st_mtime_ns}",
        flush=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    step = int(checkpoint["step"])
    cursor_field_present = "committed_logical_index" in checkpoint
    cursor = int(
        checkpoint.get(
            "committed_logical_index",
            step * int(config["batch_size"]),
        )
    )
    if checkpoint.get("schema") != config["schema"]:
        raise RuntimeError("checkpoint/config schema mismatch")
    if cursor != step * int(config["batch_size"]):
        raise RuntimeError("checkpoint cursor contract failed")
    model = ParallelCategoricalGPTQWeightBottleneck(
        _model_config(config),
        "normalized_float",
        scale_vocab_size=int(config["scale_bins"]["vocabulary_size"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    gc.collect()
    model.to(device).eval()
    return model, {
        "path": str(checkpoint_path),
        "step": step,
        "cursor": cursor,
        "cursor_field_present": cursor_field_present,
        "cursor_derived_from_step_for_persistent_model": not cursor_field_present,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _paired_change(
    old: dict[str, torch.Tensor],
    new: dict[str, torch.Tensor],
    shared: dict[str, Any],
) -> dict[str, float]:
    code_valid = shared["code_valid"] & shared["eligible"][:, None, None, None]
    scale_valid = shared["scale_valid"] & shared["eligible"][:, None]
    old_code_error = (
        old["hard_code_ids"] - shared["code_targets"]
    ).abs().float()[code_valid]
    new_code_error = (
        new["hard_code_ids"] - shared["code_targets"]
    ).abs().float()[code_valid]
    old_scale_error = (
        old["hard_scale_ids"] - shared["scale_targets"]
    ).abs().float()[scale_valid]
    new_scale_error = (
        new["hard_scale_ids"] - shared["scale_targets"]
    ).abs().float()[scale_valid]
    old_code_ordinal = old["code_per_target_ordinal"][code_valid]
    new_code_ordinal = new["code_per_target_ordinal"][code_valid]
    old_scale_ordinal = old["scale_per_target_ordinal"][scale_valid]
    new_scale_ordinal = new["scale_per_target_ordinal"][scale_valid]
    code_probability_l1 = (
        old["code_probabilities"] - new["code_probabilities"]
    ).abs().sum(dim=-1)[code_valid]
    scale_probability_l1 = (
        old["scale_probabilities"] - new["scale_probabilities"]
    ).abs().sum(dim=-1)[scale_valid]
    return {
        "code_argmax_change_fraction": float(
            (old["hard_code_ids"][code_valid] != new["hard_code_ids"][code_valid])
            .float()
            .mean()
        ),
        "scale_argmax_change_fraction": float(
            (old["hard_scale_ids"][scale_valid] != new["hard_scale_ids"][scale_valid])
            .float()
            .mean()
        ),
        "code_hard_error_improved_fraction": float((new_code_error < old_code_error).float().mean()),
        "code_hard_error_worsened_fraction": float((new_code_error > old_code_error).float().mean()),
        "scale_hard_error_improved_fraction": float((new_scale_error < old_scale_error).float().mean()),
        "scale_hard_error_worsened_fraction": float((new_scale_error > old_scale_error).float().mean()),
        "code_ordinal_improved_fraction": float((new_code_ordinal < old_code_ordinal).float().mean()),
        "code_ordinal_worsened_fraction": float((new_code_ordinal > old_code_ordinal).float().mean()),
        "scale_ordinal_improved_fraction": float((new_scale_ordinal < old_scale_ordinal).float().mean()),
        "scale_ordinal_worsened_fraction": float((new_scale_ordinal > old_scale_ordinal).float().mean()),
        "code_probability_l1_mean": float(code_probability_l1.mean()),
        "scale_probability_l1_mean": float(scale_probability_l1.mean()),
    }


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    config = _load_config(config_path)
    device = torch.device(config["device"])
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    print(
        "[paired-probe] stage=start "
        f"device={device} dtype=bfloat16 seed={config['seed']} batch={args.batch_size} "
        f"probe_cursor={args.probe_cursor} output={output_path}",
        flush=True,
    )

    probe_config = dict(config)
    probe_config["operator_bank"] = dict(config["operator_bank"])
    probe_config["operator_bank"]["loader_workers"] = 0
    logger = logging.getLogger("parallel-categorical-paired-probe")
    logger.setLevel(logging.WARNING)
    with ExitStack() as stack:
        stream = _batch_stream(probe_config, int(args.probe_cursor), logger, stack)
        cpu_batch = _batch_from_stream(stream, int(args.batch_size))
    weights = cpu_batch["W"].to(device)
    activation = cpu_batch["X"].to(device)
    x_mask = cpu_batch["x_mask"].to(device)
    d_in_mask = cpu_batch["d_in_mask"].to(device)
    d_out_mask = cpu_batch["d_out_mask"].to(device)
    tile_row = cpu_batch["tile_row"].to(device)
    tile_col = cpu_batch["tile_col"].to(device)
    print("[paired-probe] stage=gptq-targets", flush=True)
    target_codes_signed, target_scales, _ = _mask_aware_gptq_targets(
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
    scale_valid = d_out_mask & d_in_mask.any(dim=-1)[:, None] & (target_scales > 1.0e-8 / 7.0)
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
        raise RuntimeError("fixed batch has no exact-layout groups")

    shared: dict[str, Any] = {
        "config": config,
        "weights": weights,
        "activation": activation,
        "x_mask": x_mask,
        "d_in_mask": d_in_mask,
        "d_out_mask": d_out_mask,
        "tile_row": tile_row,
        "tile_col": tile_col,
        "token_valid": token_valid,
        "target_codes_signed": target_codes_signed,
        "target_scales": target_scales,
        "target_log2_scale": target_log2_scale,
        "code_targets": code_targets,
        "scale_targets": scale_targets,
        "scale_centers": scale_centers,
        "code_valid": code_valid,
        "scale_valid": scale_valid,
        "groups": groups,
        "layout_groups": groups,
        "eligible": eligible,
    }

    checkpoint_specs = [
        ("step90000", args.checkpoint_90k.resolve(), 90_000),
        ("step100000", args.checkpoint_100k.resolve(), 100_000),
    ]
    checkpoint_records: dict[str, Any] = {}
    retained: dict[str, dict[str, torch.Tensor]] = {}
    for label, checkpoint_path, expected_step in checkpoint_specs:
        print(f"[paired-probe] stage=model label={label}", flush=True)
        model, checkpoint_record = _load_model(checkpoint_path, config, device)
        if checkpoint_record["step"] != expected_step:
            raise RuntimeError(
                f"{label} expected step {expected_step}, got {checkpoint_record['step']}"
            )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            distribution = model._encode_distribution_context(activation, sample_mask=x_mask)
            latent, _ = model.encode(
                content,
                input_log_scale,
                tile_row,
                tile_col=tile_col,
                token_valid_mask=token_valid,
                dist_patch_by_patch=distribution,
            )
        shared["latent_for_diagnostics"] = latent

        attention: list[dict[str, Any]] = []
        block_names = {
            id(block): f"decoder_block_{index + 1}"
            for index, block in enumerate(model.decoder_blocks)
        }
        block_names[id(model.categorical_tail)] = "categorical_tail"
        original_runner = model._run_cross_attention_block

        def instrumented_runner(
            block: torch.nn.Module,
            query_state: torch.Tensor,
            memory: torch.Tensor,
            query_mask: torch.Tensor,
        ) -> torch.Tensor:
            attention.append(
                _attention_owner_record(
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
        try:
            code_logits, scale_logits = _decode(model, latent, shared)
        finally:
            model._run_cross_attention_block = original_runner  # type: ignore[method-assign]
        base_metrics, base_tensors = _decode_metrics(code_logits, scale_logits, shared)
        swapped_logits = _decode(model, latent.index_select(0, swap_index), shared)
        shared["latent_for_diagnostics"] = latent.index_select(0, swap_index)
        swap_metrics, swap_tensors = _decode_metrics(*swapped_logits, shared)
        shared["latent_for_diagnostics"] = latent
        eligible_code = code_valid & eligible[:, None, None, None]
        eligible_scale = scale_valid & eligible[:, None]
        z_swap = {
            "metrics": swap_metrics,
            "code_loss_ratio": swap_metrics["code_ordinal_loss"] / max(base_metrics["code_ordinal_loss"], 1e-30),
            "scale_loss_ratio": swap_metrics["scale_ordinal_loss"] / max(base_metrics["scale_ordinal_loss"], 1e-30),
            "code_argmax_change_fraction": float(
                (swap_tensors["hard_code_ids"][eligible_code] != base_tensors["hard_code_ids"][eligible_code]).float().mean()
            ),
            "scale_argmax_change_fraction": float(
                (swap_tensors["hard_scale_ids"][eligible_scale] != base_tensors["hard_scale_ids"][eligible_scale]).float().mean()
            ),
        }

        temperature: dict[str, Any] = {}
        if expected_step == 100_000:
            variants = {
                "tail_x0p5": {id(model.categorical_tail): 0.5},
                "tail_x2": {id(model.categorical_tail): 2.0},
                "tail_x4": {id(model.categorical_tail): 4.0},
                "all_blocks_x2": {
                    id(block): 2.0 for block in [*model.decoder_blocks, model.categorical_tail]
                },
            }
            for variant_name, factor_by_block in variants.items():
                print(
                    f"[paired-probe] stage=temperature variant={variant_name}", flush=True
                )
                variant_logits = _decode(
                    model,
                    latent,
                    shared,
                    factor_by_block=factor_by_block,
                )
                variant_metrics, _ = _decode_metrics(*variant_logits, shared)
                temperature[variant_name] = variant_metrics

        latent_payload = _latent_anova(latent)
        latent_payload["rank"] = _latent_rank_payload(latent)
        checkpoint_records[label] = {
            "checkpoint": checkpoint_record,
            "latent": latent_payload,
            "attention_owner": attention,
            "baseline": base_metrics,
            "z_swap": z_swap,
            "temperature": temperature,
        }
        retained[label] = {
            key: value.detach().cpu()
            for key, value in base_tensors.items()
        }
        del model, distribution, latent, code_logits, scale_logits, swapped_logits
        gc.collect()
        torch.cuda.empty_cache()

    paired_change = _paired_change(
        {key: value.to(device) for key, value in retained["step90000"].items()},
        {key: value.to(device) for key, value in retained["step100000"].items()},
        shared,
    )
    payload = {
        "schema": "weightclip_parallel_categorical_paired_90k_100k_probe_v1",
        "config": str(config_path),
        "device": str(device),
        "dtype": "bfloat16_autocast_float32_metrics",
        "batch_size": batch,
        "probe_cursor": int(args.probe_cursor),
        "probe_logical_indices": cpu_batch["logical_indices"],
        "exact_layout_group_sizes": [len(group) for group in groups],
        "exact_layout_eligible_samples": int(eligible.sum()),
        "checkpoints": checkpoint_records,
        "paired_change_step90000_to_step100000": paired_change,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"[paired-probe] stage=complete output={output_path}",
        flush=True,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
