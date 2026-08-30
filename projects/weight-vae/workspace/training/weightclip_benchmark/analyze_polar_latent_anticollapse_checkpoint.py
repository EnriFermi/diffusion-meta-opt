from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from training.weightclip_benchmark.analyze_polar_tail_mechanism import (
    _direction_target_summary,
    _effective_rank,
    _intervention_summary,
    _load_batch,
    _make_component_mask,
    _pairwise_cosine_summary,
)
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA,
    PolarTailedUnifiedWeightBottleneck,
    _latent_anticollapse_objective,
    _load_config,
    _model_config,
    _polar_direction_scale_objectives,
    _prepare_normalized_inputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only latent geometry and same-layout swap probe."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--required-step", type=int, default=1000)
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _layout_key(batch: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        int(batch["tile_row"][index]),
        int(batch["tile_col"][index]),
        bytes(batch["d_in_mask"][index].to(torch.uint8).tolist()),
        bytes(batch["d_out_mask"][index].to(torch.uint8).tolist()),
    )


def _select_same_layout_batch(
    batch: dict[str, Any], maximum: int
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index in range(int(batch["W"].shape[0])):
        groups[_layout_key(batch, index)].append(index)

    selected: list[int] = []
    selected_groups: list[list[int]] = []
    for indices in groups.values():
        if len(indices) < 2 or len(selected) >= maximum:
            continue
        take = indices[: maximum - len(selected)]
        if len(take) < 2:
            continue
        positions = list(range(len(selected), len(selected) + len(take)))
        selected.extend(take)
        selected_groups.append(positions)
    if len(selected) < 2:
        raise RuntimeError("candidate batch has no repeated exact layout")

    permutation = list(range(len(selected)))
    for positions in selected_groups:
        rolled = positions[-1:] + positions[:-1]
        for destination, source in zip(positions, rolled, strict=True):
            permutation[destination] = source
    if any(destination == source for destination, source in enumerate(permutation)):
        raise RuntimeError("same-layout swap contains a fixed point")
    return (
        torch.tensor(selected, dtype=torch.long),
        torch.tensor(permutation, dtype=torch.long),
        selected_groups,
    )


def _latent_geometry(z: torch.Tensor) -> dict[str, Any]:
    value = z.detach().float()
    rms = value.square().mean().sqrt().clamp_min(1.0e-12)
    grand = value.mean(dim=(0, 1), keepdim=True)
    sample_effect = value.mean(dim=1, keepdim=True) - grand
    slot_effect = value.mean(dim=0, keepdim=True) - grand
    interaction = value - value.mean(1, keepdim=True) - value.mean(0, keepdim=True) + grand

    within_rank = [_effective_rank(sample) for sample in value]
    within_cosine = [
        _pairwise_cosine_summary(sample)["mean"] for sample in value
    ]
    within_centered_ratio = [
        float(
            (sample - sample.mean(0, keepdim=True))
            .square()
            .mean()
            .sqrt()
            .div(sample.square().mean().sqrt().clamp_min(1.0e-12))
        )
        for sample in value
    ]
    anti_loss, anti_stats = _latent_anticollapse_objective(value)
    return {
        "shape": list(value.shape),
        "rms": float(rms),
        "cross_sample": {
            "effective_rank_flattened_sample": _effective_rank(value.flatten(1)),
            "pairwise_cosine_flattened_sample": _pairwise_cosine_summary(
                value.flatten(1)
            ),
            "sample_effect_rms_over_total": float(
                sample_effect.square().mean().sqrt() / rms
            ),
        },
        "within_sample_slots": {
            "effective_rank": _summary(within_rank),
            "pairwise_cosine": _summary(within_cosine),
            "centered_rms_over_sample_rms": _summary(within_centered_ratio),
            "slot_effect_rms_over_total": float(slot_effect.square().mean().sqrt() / rms),
        },
        "sample_slot_interaction": {
            "rms_over_total": float(interaction.square().mean().sqrt() / rms),
            "objective": float(anti_loss),
            **{key: float(value) for key, value in anti_stats.items()},
        },
    }


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    run_root = args.run_root.resolve()
    config = _load_config(config_path)
    if config["schema"] != POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA:
        raise ValueError("probe requires the polar latent-anticollapse schema")
    checkpoint_stat = checkpoint_path.stat()
    payload = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    step = int(payload["step"])
    cursor = int(payload["committed_logical_index"])
    if step != int(args.required_step):
        raise RuntimeError(f"required checkpoint step {args.required_step}, found {step}")
    if cursor != step * int(config["batch_size"]):
        raise RuntimeError("checkpoint cursor does not match step and batch size")

    output_root = run_root / f"latent_geometry_step_{step:07d}_v1"
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    _atomic_json(
        output_root / "RUNNING.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "config": str(config_path),
            "step": step,
        },
    )

    candidate = _load_batch(config, cursor, int(args.candidate_batch_size))
    selected, swap, groups = _select_same_layout_batch(
        candidate, int(args.probe_batch_size)
    )
    cpu = {
        key: candidate[key].index_select(0, selected)
        for key in (
            "W",
            "X",
            "x_mask",
            "d_in_mask",
            "d_out_mask",
            "tile_row",
            "tile_col",
        )
    }
    logical_indices = [candidate["logical_indices"][int(index)] for index in selected]

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    torch.set_float32_matmul_precision("high")
    model = PolarTailedUnifiedWeightBottleneck(
        _model_config(config), "normalized_float"
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    del payload
    model.eval()

    W = cpu["W"].to(device)
    X = cpu["X"].to(device)
    x_mask = cpu["x_mask"].to(device)
    d_in = cpu["d_in_mask"].to(device)
    d_out = cpu["d_out_mask"].to(device)
    tile_row = cpu["tile_row"].to(device)
    tile_col = cpu["tile_col"].to(device)
    content, log_scale, token_valid = _prepare_normalized_inputs(
        W,
        d_in,
        d_out,
        scale_mean=float(config["normalization"]["log2_scale_mean"]),
        scale_std=float(config["normalization"]["log2_scale_std"]),
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        prediction, z, _depth, pred_dirs, pred_scales = model(
            content,
            log_scale,
            tile_row,
            X,
            d_in_mask=d_in,
            d_out_mask=d_out,
            tile_col=tile_col,
            activation_sample_mask=x_mask,
            token_valid_mask=token_valid,
        )
        distribution = model._encode_distribution_context(X, sample_mask=x_mask)
        swapped_prediction, swapped_dirs, swapped_scales = model.decode_polar(
            z.index_select(0, swap.to(device)),
            tile_row,
            d_in_mask=d_in,
            d_out_mask=d_out,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=distribution,
        )
        matched_direction, matched_scale = _polar_direction_scale_objectives(
            X,
            W,
            pred_dirs,
            pred_scales,
            x_mask,
            d_in,
            d_out,
            config["loss"],
        )
        swapped_direction, swapped_scale = _polar_direction_scale_objectives(
            X,
            W,
            swapped_dirs,
            swapped_scales,
            x_mask,
            d_in,
            d_out,
            config["loss"],
        )

    component_mask = _make_component_mask(d_in, d_out)
    direction_summary = _direction_target_summary(W, pred_dirs, component_mask)
    sensitivity = _intervention_summary(
        prediction,
        pred_dirs,
        pred_scales,
        swapped_prediction,
        swapped_dirs,
        swapped_scales,
        component_mask,
    )
    swap_cpu = swap.tolist()
    report = {
        "schema": "weightclip_polar_latent_anticollapse_checkpoint_probe_v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "step": step,
            "committed_logical_index": cursor,
            "size": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
        },
        "batch": {
            "candidate_size": int(args.candidate_batch_size),
            "probe_size": len(logical_indices),
            "logical_indices": logical_indices,
            "same_layout_group_sizes": [len(group) for group in groups],
            "swap_logical_indices": [logical_indices[index] for index in swap_cpu],
            "all_swap_pairs_exact_layout_match": all(
                _layout_key(candidate, int(selected[destination]))
                == _layout_key(candidate, int(selected[source]))
                for destination, source in enumerate(swap_cpu)
            ),
        },
        "latent_geometry": _latent_geometry(z),
        "direction_representation": direction_summary,
        "same_layout_latent_swap": {
            "weighted_direction": {
                "matched": float(matched_direction),
                "swapped": float(swapped_direction),
                "swapped_minus_matched": float(swapped_direction - matched_direction),
            },
            "weighted_scale": {
                "matched": float(matched_scale),
                "swapped": float(swapped_scale),
                "swapped_minus_matched": float(swapped_scale - matched_scale),
            },
            "decoder_sensitivity": sensitivity,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_root / "report.json", report)
    (output_root / "RUNNING.json").unlink()
    _atomic_json(
        output_root / "COMPLETE.json",
        {
            "complete": True,
            "step": step,
            "report": str(output_root / "report.json"),
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
