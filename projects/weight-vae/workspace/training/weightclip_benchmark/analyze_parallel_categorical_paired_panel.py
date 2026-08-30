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

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.weightclip_benchmark.analyze_parallel_categorical_current_state import (
    _weight_from_components,
)
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    _batch_from_stream,
    _model_config,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_parallel_categorical_gptq_700m_production import (
    CODE_VOCAB_SIZE,
    CODE_ZERO_INDEX,
    ParallelCategoricalGPTQWeightBottleneck,
    _batch_stream,
    _decode_categorical_argmax,
    _load_config,
    _mask_aware_gptq_targets,
    _ordered_cumulative_log_loss,
    _scale_bin_centers,
    _scale_class_targets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core paired checkpoint panel")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-90k", type=Path, required=True)
    parser.add_argument("--checkpoint-100k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-cursor", type=int, default=3_136_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=8)
    return parser.parse_args()


def _load_model(path: Path, config: dict[str, Any], device: torch.device, expected: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if int(checkpoint["step"]) != expected or checkpoint.get("schema") != config["schema"]:
        raise RuntimeError(f"bad checkpoint contract: {path}")
    model = ParallelCategoricalGPTQWeightBottleneck(
        _model_config(config),
        "normalized_float",
        scale_vocab_size=int(config["scale_bins"]["vocabulary_size"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    gc.collect()
    return model.to(device).eval()


@torch.no_grad()
def _evaluate(
    model: ParallelCategoricalGPTQWeightBottleneck,
    data: dict[str, torch.Tensor],
    config: dict[str, Any],
    scale_centers: torch.Tensor,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    weights = data["W"]
    activation = data["X"]
    x_mask = data["x_mask"]
    d_in_mask = data["d_in_mask"]
    d_out_mask = data["d_out_mask"]
    batch = weights.shape[0]
    target_codes_signed, target_scales, _ = _mask_aware_gptq_targets(
        weights,
        activation,
        x_mask,
        d_in_mask,
        d_out_mask,
        bits=int(config["gptq"]["bits"]),
        damp_fraction=float(config["gptq"]["damp_fraction"]),
    )
    code_targets = target_codes_signed.long().view(batch, 128, 8, 16) + CODE_ZERO_INDEX
    scale_targets = _scale_class_targets(target_scales, scale_centers)
    code_valid = d_out_mask[:, :, None, None] & d_in_mask.view(batch, 1, 8, 16)
    scale_valid = d_out_mask & d_in_mask.any(dim=-1)[:, None] & (target_scales > 1e-8 / 7.0)
    content, input_log_scale, token_valid = _prepare_normalized_inputs(
        weights,
        d_in_mask,
        d_out_mask,
        scale_mean=float(config["normalization"]["log2_scale_mean"]),
        scale_std=float(config["normalization"]["log2_scale_std"]),
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        distribution = model._encode_distribution_context(activation, sample_mask=x_mask)
        latent, _ = model.encode(
            content,
            input_log_scale,
            data["tile_row"],
            tile_col=data["tile_col"],
            token_valid_mask=token_valid,
            dist_patch_by_patch=distribution,
        )
        code_logits, scale_logits, _ = model.decode_categorical(
            latent,
            data["tile_row"],
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=data["tile_col"],
            token_valid_mask=token_valid,
        )
    code_loss, code_stats = _ordered_cumulative_log_loss(code_logits, code_targets, code_valid)
    scale_loss, scale_stats = _ordered_cumulative_log_loss(scale_logits, scale_targets, scale_valid)
    hard_prediction, hard_codes, hard_scales = _decode_categorical_argmax(
        code_logits, scale_logits, scale_centers, d_in_mask, d_out_mask
    )
    code_probability = code_logits.float().softmax(dim=-1)
    code_axis = torch.arange(CODE_VOCAB_SIZE, device=weights.device).float()
    expected_codes = (code_probability * code_axis).sum(dim=-1) - CODE_ZERO_INDEX
    scale_probability = scale_logits.float().softmax(dim=-1)
    expected_log2_scale = (scale_probability * scale_centers).sum(dim=-1)
    expected_prediction = _weight_from_components(
        expected_codes,
        torch.exp2(expected_log2_scale),
        d_in_mask,
        d_out_mask,
    )
    oracle_code_expected_scale = _weight_from_components(
        target_codes_signed.view(batch, 128, 8, 16),
        torch.exp2(expected_log2_scale),
        d_in_mask,
        d_out_mask,
    )
    valid_values = d_in_mask[:, :, None] & d_out_mask[:, None, :]
    target_probability = code_probability.gather(-1, code_targets[..., None]).squeeze(-1)
    target_nll = -code_logits.float().log_softmax(dim=-1).gather(
        -1, code_targets[..., None]
    ).squeeze(-1)

    def sums(prediction: torch.Tensor) -> tuple[float, float]:
        error = ((prediction.float() - weights.float()) * valid_values).square().sum()
        energy = (weights.float() * valid_values).square().sum()
        return float(error), float(energy)

    hard_sse, target_energy = sums(hard_prediction)
    expected_sse, _ = sums(expected_prediction)
    oracle_code_scale_sse, _ = sums(oracle_code_expected_scale)
    operator = BigWeightVAELossMixin.operator_recon_loss(
        activation,
        weights,
        hard_prediction,
        x_mask=x_mask,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
    )
    record = {
        "code_ordinal_loss": float(code_loss),
        "scale_ordinal_loss": float(scale_loss),
        "code_accuracy": float(code_stats["accuracy"]),
        "code_mae_bins": float(code_stats["mean_absolute_bin_error"]),
        "scale_accuracy": float(scale_stats["accuracy"]),
        "scale_mae_bins": float(scale_stats["mean_absolute_bin_error"]),
        "code_target_probability": float(target_probability[code_valid].mean()),
        "code_target_nll": float(target_nll[code_valid].mean()),
        "code_prediction_zero_fraction": float((hard_codes[code_valid] == CODE_ZERO_INDEX).float().mean()),
        "code_target_zero_fraction": float((code_targets[code_valid] == CODE_ZERO_INDEX).float().mean()),
        "hard_sse": hard_sse,
        "expected_sse": expected_sse,
        "oracle_code_expected_scale_sse": oracle_code_scale_sse,
        "target_energy": target_energy,
        "hard_nrmse": math.sqrt(hard_sse / target_energy),
        "expected_nrmse": math.sqrt(expected_sse / target_energy),
        "oracle_code_expected_scale_nrmse": math.sqrt(oracle_code_scale_sse / target_energy),
        "operator_metric": float(operator),
        "code_count": int(code_valid.sum()),
        "scale_count": int(scale_valid.sum()),
    }
    tensors = {
        "hard_codes": hard_codes.cpu(),
        "hard_scales": hard_scales.cpu(),
        "code_valid": code_valid.cpu(),
        "scale_valid": scale_valid.cpu(),
    }
    return record, tensors


def _aggregate(records: list[dict[str, float]]) -> dict[str, Any]:
    code_count = sum(int(row["code_count"]) for row in records)
    scale_count = sum(int(row["scale_count"]) for row in records)
    target_energy = sum(row["target_energy"] for row in records)
    result: dict[str, Any] = {
        "batches": len(records),
        "samples": 0,
        "code_count": code_count,
        "scale_count": scale_count,
    }
    for key in (
        "code_ordinal_loss",
        "code_accuracy",
        "code_mae_bins",
        "code_target_probability",
        "code_target_nll",
        "code_prediction_zero_fraction",
        "code_target_zero_fraction",
    ):
        result[key] = sum(row[key] * row["code_count"] for row in records) / code_count
    for key in ("scale_ordinal_loss", "scale_accuracy", "scale_mae_bins"):
        result[key] = sum(row[key] * row["scale_count"] for row in records) / scale_count
    for prefix in ("hard", "expected", "oracle_code_expected_scale"):
        result[f"{prefix}_nrmse"] = math.sqrt(
            sum(row[f"{prefix}_sse"] for row in records) / target_energy
        )
    result["operator_metric_mean"] = sum(row["operator_metric"] for row in records) / len(records)
    return result


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config.resolve())
    device = torch.device(config["device"])
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    print(
        f"[paired-panel] stage=start batches={args.batches} batch={args.batch_size} "
        f"cursor={args.start_cursor} workers={config['operator_bank']['loader_workers']} "
        f"device={device} dtype=bfloat16 output={args.output.resolve()}",
        flush=True,
    )
    logger = logging.getLogger("paired-panel")
    logger.setLevel(logging.WARNING)
    cpu_batches = []
    with ExitStack() as stack:
        stream = _batch_stream(config, int(args.start_cursor), logger, stack)
        for index in range(int(args.batches)):
            cpu_batches.append(_batch_from_stream(stream, int(args.batch_size)))
            print(f"[paired-panel] stage=data batch={index + 1}/{args.batches}", flush=True)
    print("[paired-panel] stage=models", flush=True)
    models = {
        "step90000": _load_model(args.checkpoint_90k.resolve(), config, device, 90_000),
        "step100000": _load_model(args.checkpoint_100k.resolve(), config, device, 100_000),
    }
    scale_centers = _scale_bin_centers(
        int(config["scale_bins"]["vocabulary_size"]),
        float(config["scale_bins"]["log2_min"]),
        float(config["scale_bins"]["log2_max"]),
        device=device,
    )
    by_model: dict[str, list[dict[str, float]]] = {key: [] for key in models}
    changes = []
    for batch_index, cpu_batch in enumerate(cpu_batches):
        data = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in cpu_batch.items()
        }
        outputs = {}
        for label, model in models.items():
            record, tensors = _evaluate(model, data, config, scale_centers)
            record["batch_index"] = batch_index
            record["logical_start"] = int(cpu_batch["logical_indices"][0])
            record["logical_end"] = int(cpu_batch["logical_indices"][-1])
            by_model[label].append(record)
            outputs[label] = tensors
        code_valid = outputs["step90000"]["code_valid"]
        scale_valid = outputs["step90000"]["scale_valid"]
        changes.append({
            "batch_index": batch_index,
            "code_argmax_change_fraction": float(
                (outputs["step90000"]["hard_codes"][code_valid] != outputs["step100000"]["hard_codes"][code_valid]).float().mean()
            ),
            "scale_argmax_change_fraction": float(
                (outputs["step90000"]["hard_scales"][scale_valid] != outputs["step100000"]["hard_scales"][scale_valid]).float().mean()
            ),
        })
        print(f"[paired-panel] stage=evaluate batch={batch_index + 1}/{args.batches}", flush=True)
    aggregate = {label: _aggregate(records) for label, records in by_model.items()}
    for value in aggregate.values():
        value["samples"] = int(args.batches) * int(args.batch_size)
    delta = {
        key: aggregate["step100000"][key] - aggregate["step90000"][key]
        for key in (
            "code_ordinal_loss",
            "scale_ordinal_loss",
            "hard_nrmse",
            "expected_nrmse",
            "oracle_code_expected_scale_nrmse",
            "operator_metric_mean",
            "code_target_probability",
            "code_target_nll",
            "code_prediction_zero_fraction",
        )
    }
    metric_keys = ("code_ordinal_loss", "scale_ordinal_loss", "hard_nrmse", "expected_nrmse", "operator_metric")
    delta["batch_sign_counts"] = {
        key: {
            "step100_better": sum(by_model["step100000"][i][key] < by_model["step90000"][i][key] for i in range(len(cpu_batches))),
            "step100_worse": sum(by_model["step100000"][i][key] > by_model["step90000"][i][key] for i in range(len(cpu_batches))),
        }
        for key in metric_keys
    }
    payload = {
        "schema": "weightclip_parallel_categorical_paired_panel_v1",
        "config": str(args.config.resolve()),
        "checkpoint_90k": str(args.checkpoint_90k.resolve()),
        "checkpoint_100k": str(args.checkpoint_100k.resolve()),
        "device": str(device),
        "dtype": "bfloat16_autocast_float32_metrics",
        "start_cursor": int(args.start_cursor),
        "batch_size": int(args.batch_size),
        "batches": int(args.batches),
        "per_batch": by_model,
        "aggregate": aggregate,
        "paired_argmax_changes": changes,
        "delta_step100000_minus_step90000": delta,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[paired-panel] stage=complete output={output}", flush=True)
    print(json.dumps({"aggregate": aggregate, "delta": delta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
