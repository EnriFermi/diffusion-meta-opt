from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    BOUNDED_BALANCED_SCHEMA,
    DEFAULT_CONFIG,
    _load_config,
    _model_config,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    DEFAULT_RESOLVED,
    DEFAULT_SELECTION,
    UnifiedWeightBottleneck,
    _load_exact64_tiles,
    _seed_everything,
)


DEFAULT_CHECKPOINT = Path(
    "/home/coder/project/projects/shared/storage/artifacts/weightclip_benchmark/"
    "direct_normalized_scaled_700m_p32_no_operator_500k_v1/model_latest.pt"
)
DEFAULT_OUTPUT = Path(
    "/mnt/shared/weightclip_benchmark/"
    "direct_normalized_scaled_700m_p32_no_operator_500k_v1/"
    "analysis_checkpoint_050000_latent_use.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--resolved-config", type=Path, default=DEFAULT_RESOLVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-operators", type=int, default=2)
    parser.add_argument("--production-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--initial", action="store_true")
    return parser.parse_args()


def _condition_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    *,
    structural_scale_weight: float,
) -> dict[str, float]:
    structural, structural_parts = BigWeightVAELossMixin.patch_structure_loss(
        target.flatten(0, 1),
        prediction.flatten(0, 1),
        patch_size=16,
        gamma=0.5,
        lambda_dir=1.0,
        lambda_scale=structural_scale_weight,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=0.1,
    )
    flat_context = context.flatten(0, 1)
    flat_target = target.flatten(0, 1)
    flat_prediction = prediction.flatten(0, 1)
    behavioral_dir, behavioral_scale = (
        BigWeightVAELossMixin.operator_direction_scale_loss(
            flat_context,
            flat_target,
            flat_prediction,
            gamma=0.5,
            huber_delta=0.1,
        )
    )
    predicted_action = torch.einsum(
        "btri,btic->btrc", context.float(), prediction.float()
    ).sum(dim=1)
    target_action = torch.einsum(
        "btri,btic->btrc", context.float(), target.float()
    ).sum(dim=1)
    raw_num = (prediction.float() - target.float()).square().sum()
    raw_den = target.float().square().sum().clamp_min(1.0e-12)
    action_num = (predicted_action - target_action).square().sum()
    action_den = target_action.square().sum().clamp_min(1.0e-12)
    cosine = F.cosine_similarity(
        prediction.float().flatten(1), target.float().flatten(1), dim=1
    ).mean()
    return {
        "structural": float(structural.item()),
        "structural_direction": float(structural_parts["L_dir"].item()),
        "structural_scale": float(structural_parts["L_scale"].item()),
        "behavioral_direction": float(behavioral_dir.item()),
        "behavioral_scale": float(behavioral_scale.item()),
        "raw_nrmse": float(torch.sqrt(raw_num / raw_den).item()),
        "action_nrmse": float(torch.sqrt(action_num / action_den).item()),
        "mean_tile_cosine": float(cosine.item()),
    }


def main() -> None:
    args = _parse_args()
    if args.batch_operators < 2:
        raise ValueError("batch-operators must be at least two for paired shuffles")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.initial:
        checkpoint_step = 0
        checkpoint_label = "seed-42 initialization"
        production_config = _load_config(args.production_config.resolve())
        model_state = None
    else:
        print(f"[latent-use] stage=load-checkpoint path={args.checkpoint}", flush=True)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_step = int(payload["step"])
        checkpoint_label = str(args.checkpoint.resolve())
        production_config = payload["config"]
        model_state = payload["model_state"]
        del payload
    cfg = _model_config(production_config)
    _seed_everything(cfg.seed)
    model = UnifiedWeightBottleneck(cfg, "normalized_float")
    if model_state is not None:
        model.load_state_dict(model_state, strict=True)

    print("[latent-use] stage=load-exact64", flush=True)
    data = _load_exact64_tiles(args.selection.resolve(), args.resolved_config.resolve())
    weights = data["weights"].float()
    contexts = data["contexts"].float()
    tile_rows = data["tile_rows"].long()
    if tuple(weights.shape) != (64, 9, 128, 128):
        raise RuntimeError(f"unexpected exact64 weight shape: {tuple(weights.shape)}")
    if tuple(contexts.shape) != (64, 9, 512, 128):
        raise RuntimeError(f"unexpected exact64 context shape: {tuple(contexts.shape)}")

    device = torch.device("cuda:0")
    model = model.to(device).eval()
    conditions = ("matched", "zero_latent", "shuffled_latent", "shuffled_weight_input")
    totals: dict[str, dict[str, float]] = {
        condition: defaultdict(float) for condition in conditions
    }
    counts = {condition: 0 for condition in conditions}
    latent_sums = defaultdict(float)
    latent_rows: list[torch.Tensor] = []
    operator_count = 0
    print(
        f"[latent-use] stage=evaluate checkpoint_step={checkpoint_step} operators=64",
        flush=True,
    )
    with torch.no_grad():
        for start in range(0, 64, args.batch_operators):
            stop = min(start + args.batch_operators, 64)
            if stop - start < 2:
                break
            batch_weights = weights[start:stop].to(device)
            batch_contexts = contexts[start:stop].to(device)
            batch_rows = tile_rows[start:stop].to(device)
            operators = stop - start
            flat_weights = batch_weights.flatten(0, 1)
            flat_contexts = batch_contexts.flatten(0, 1)
            flat_rows = batch_rows.flatten(0, 1)
            flat_cols = torch.zeros_like(flat_rows)
            full_mask = torch.ones(
                flat_weights.shape[0], 128, device=device, dtype=torch.bool
            )
            content, log_scale, token_valid = _prepare_normalized_inputs(
                flat_weights,
                full_mask,
                full_mask,
                scale_mean=float(
                    production_config["normalization"]["log2_scale_mean"]
                ),
                scale_std=float(
                    production_config["normalization"]["log2_scale_std"]
                ),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                dist = model._encode_distribution_context(flat_contexts)
                z, _ = model.encode(
                    content,
                    log_scale,
                    flat_rows,
                    tile_col=flat_cols,
                    token_valid_mask=token_valid,
                    dist_patch_by_patch=dist,
                )
                matched = model.decode(
                    z,
                    flat_rows,
                    tile_col=flat_cols,
                    token_valid_mask=token_valid,
                    dist_patch_by_patch=dist,
                )
                zero_latent = model.decode(
                    torch.zeros_like(z),
                    flat_rows,
                    tile_col=flat_cols,
                    token_valid_mask=token_valid,
                    dist_patch_by_patch=dist,
                )
                z_by_operator = z.view(operators, 9, cfg.latent_slots, cfg.latent_dim)
                shuffled_z = z_by_operator.roll(1, dims=0).reshape_as(z)
                shuffled_latent = model.decode(
                    shuffled_z,
                    flat_rows,
                    tile_col=flat_cols,
                    token_valid_mask=token_valid,
                    dist_patch_by_patch=dist,
                )
                content_by_operator = content.view(operators, 9, *content.shape[1:])
                scale_by_operator = log_scale.view(operators, 9, *log_scale.shape[1:])
                shuffled_content = content_by_operator.roll(1, dims=0).reshape_as(content)
                shuffled_scale = scale_by_operator.roll(1, dims=0).reshape_as(log_scale)
                shuffled_input_z, _ = model.encode(
                    shuffled_content,
                    shuffled_scale,
                    flat_rows,
                    tile_col=flat_cols,
                    token_valid_mask=token_valid,
                    dist_patch_by_patch=dist,
                )
                shuffled_weight_input = model.decode(
                    shuffled_input_z,
                    flat_rows,
                    tile_col=flat_cols,
                    token_valid_mask=token_valid,
                    dist_patch_by_patch=dist,
                )

            outputs = {
                "matched": matched.float().view(operators, 9, 128, 128),
                "zero_latent": zero_latent.float().view(operators, 9, 128, 128),
                "shuffled_latent": shuffled_latent.float().view(operators, 9, 128, 128),
                "shuffled_weight_input": shuffled_weight_input.float().view(
                    operators, 9, 128, 128
                ),
            }
            for condition, prediction in outputs.items():
                metrics = _condition_metrics(
                    prediction,
                    batch_weights,
                    batch_contexts,
                    structural_scale_weight=(
                        1.0
                        if production_config["schema"] == BOUNDED_BALANCED_SCHEMA
                        else 10.0
                    ),
                )
                for key, value in metrics.items():
                    totals[condition][key] += value * operators
                counts[condition] += operators
            latent_sums["matched_rms"] += float(z.float().square().mean().sqrt().item()) * operators
            latent_sums["weight_shuffle_relative_delta"] += float(
                (shuffled_input_z.float() - z.float()).square().mean().sqrt().div(
                    z.float().square().mean().sqrt().clamp_min(1.0e-12)
                ).item()
            ) * operators
            latent_rows.append(
                z.float()
                .view(operators, 9, cfg.latent_slots, cfg.latent_dim)
                .flatten(1)
                .cpu()
            )
            target_rms = batch_weights.square().mean().sqrt().clamp_min(1.0e-12)
            latent_sums["matched_vs_zero_output_delta_over_target"] += float(
                (outputs["matched"] - outputs["zero_latent"])
                .square()
                .mean()
                .sqrt()
                .div(target_rms)
                .item()
            ) * operators
            operator_count += operators
            print(f"[latent-use] progress={stop}/64", flush=True)

    latent_all = torch.cat(latent_rows, dim=0)
    normalized_latent = F.normalize(latent_all, dim=1)
    latent_cosine = normalized_latent @ normalized_latent.T
    off_diagonal = latent_cosine[~torch.eye(operator_count, dtype=torch.bool)]
    centered_latent = latent_all - latent_all.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered_latent @ centered_latent.T).clamp_min(0.0)
    result = {
        "schema": "direct_normalized_production_latent_use_v1",
        "checkpoint": checkpoint_label,
        "checkpoint_step": checkpoint_step,
        "selection": str(args.selection.resolve()),
        "operators": operator_count,
        "conditions": {
            condition: {
                key: value / counts[condition]
                for key, value in sorted(totals[condition].items())
            }
            for condition in conditions
        },
        "latent": {
            key: value / operator_count for key, value in sorted(latent_sums.items())
        },
        "latent_geometry": {
            "raw_pairwise_cosine_mean": float(off_diagonal.mean().item()),
            "raw_pairwise_cosine_p05": float(torch.quantile(off_diagonal, 0.05).item()),
            "raw_pairwise_cosine_p95": float(torch.quantile(off_diagonal, 0.95).item()),
            "centered_participation_rank": float(
                eigenvalues.sum().square().div(
                    eigenvalues.square().sum().clamp_min(1.0e-20)
                ).item()
            ),
            "centered_top1_energy_fraction": float(
                eigenvalues.max().div(eigenvalues.sum().clamp_min(1.0e-20)).item()
            ),
        },
    }
    for condition in conditions:
        for value in result["conditions"][condition].values():
            if not math.isfinite(float(value)):
                raise RuntimeError("nonfinite intervention metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[latent-use] stage=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
