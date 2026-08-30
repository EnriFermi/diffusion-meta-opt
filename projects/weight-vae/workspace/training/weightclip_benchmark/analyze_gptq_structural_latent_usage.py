from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import torch

from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    ExperimentConfig,
    UnifiedWeightBottleneck,
    _arm_content,
    _arm_log_scale,
    _atomic_json,
    _big_vae_structural_loss,
    _load_exact64_tiles,
    _prepare_representations,
)


SCHEMA = "weightclip_gptq_structural_latent_usage_v1"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure causal decoder use of the saved encoder latents."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-operators", type=int, default=6)
    return parser.parse_args()


def _operator_tile_indices(operators: list[int]) -> torch.Tensor:
    return torch.tensor(
        [operator * 9 + tile for operator in operators for tile in range(9)],
        dtype=torch.long,
    )


@torch.no_grad()
def _capture_latents(
    model: UnifiedWeightBottleneck,
    prepared: dict[str, Any],
    operators: list[int],
    batch_operators: int,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(operators), batch_operators):
        batch = operators[start : start + batch_operators]
        indices = _operator_tile_indices(batch)
        content = _arm_content(prepared, model.arm, indices).to(device)
        scale = _arm_log_scale(prepared, model.arm, indices).to(device)
        tile_rows = prepared["flat_tile_rows"].index_select(0, indices).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent, _ = model.encode(content, scale, tile_rows)
        chunks.append(
            latent.float()
            .view(len(batch), 9, model.cfg.latent_slots, model.cfg.latent_dim)
            .cpu()
        )
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _decode_metrics(
    model: UnifiedWeightBottleneck,
    prepared: dict[str, Any],
    operators: list[int],
    latent: torch.Tensor,
    matched_predictions: torch.Tensor | None,
    batch_operators: int,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    totals: dict[str, float] = defaultdict(float)
    prediction_chunks: list[torch.Tensor] = []
    for start in range(0, len(operators), batch_operators):
        batch = operators[start : start + batch_operators]
        indices = _operator_tile_indices(batch)
        tile_rows = prepared["flat_tile_rows"].index_select(0, indices).to(device)
        target = prepared["flat_weights"].index_select(0, indices).to(device).float()
        z = latent[start : start + len(batch)].reshape(
            -1, model.cfg.latent_slots, model.cfg.latent_dim
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.decode(z, tile_rows)
        prediction = prediction.float()
        loss, details = _big_vae_structural_loss(prediction, target, model.cfg)
        count = int(target.shape[0])
        totals["loss"] += float(loss.item()) * count
        totals["direction"] += float(details["L_dir"].item()) * count
        totals["scale"] += float(details["L_scale"].item()) * count
        totals["count"] += count
        totals["target_sq"] += float(target.square().sum().item())
        totals["prediction_sq"] += float(prediction.square().sum().item())
        if matched_predictions is not None:
            matched = matched_predictions[start : start + len(batch)].reshape_as(
                prediction
            ).to(device)
            totals["matched_delta_sq"] += float(
                (prediction - matched).square().sum().item()
            )
        prediction_chunks.append(
            prediction.view(len(batch), 9, 128, 128).cpu()
        )
    count = totals["count"]
    metrics = {
        "structural_loss": totals["loss"] / count,
        "direction_loss": totals["direction"] / count,
        "scale_loss": totals["scale"] / count,
        "prediction_over_target_rms": math.sqrt(
            totals["prediction_sq"] / max(totals["target_sq"], 1.0e-30)
        ),
    }
    if matched_predictions is not None:
        metrics["output_delta_over_target_rms"] = math.sqrt(
            totals["matched_delta_sq"] / max(totals["target_sq"], 1.0e-30)
        )
    return metrics, torch.cat(prediction_chunks, dim=0)


def main() -> None:
    args = _parse_args()
    root = args.run_root.expanduser().resolve()
    startup = json.loads((root / "config.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**startup["config"])
    arms = tuple(str(arm) for arm in startup["arms"])
    operators: list[int]
    device = torch.device("cuda:0")
    print(
        f"[latent-usage] stage=load root={root} device={device} arms={arms}",
        flush=True,
    )
    data = _load_exact64_tiles(
        Path(startup["selection"]), Path(startup["resolved_config"])
    )
    prepared = _prepare_representations(data, cfg, device)
    operators = list(prepared["train_operator_indices"])
    saved = torch.load(root / "final_models.pt", map_location="cpu", weights_only=True)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "run_root": str(root),
        "split": "48 training operators",
        "operators": operators,
        "interventions": {
            "zero": "replace every latent coordinate by zero",
            "operator_shuffle": "cyclically shift complete nine-tile latent groups across operators",
            "tile_shuffle": "cyclically shift latent groups across the nine tiles within each operator",
            "mean": "replace each operator latent by the train-set mean latent for the same tile",
            "alpha": list(ALPHAS),
        },
        "arms": {},
    }
    for arm in arms:
        print(f"[latent-usage] stage=arm arm={arm}", flush=True)
        model = UnifiedWeightBottleneck(cfg, arm).to(device)
        model.load_state_dict(saved[arm], strict=True)
        model.eval()
        latent = _capture_latents(
            model, prepared, operators, args.batch_operators, device
        )
        matched_metrics, matched_predictions = _decode_metrics(
            model,
            prepared,
            operators,
            latent,
            None,
            args.batch_operators,
            device,
        )
        interventions = {
            "zero": torch.zeros_like(latent),
            "operator_shuffle": torch.roll(latent, shifts=1, dims=0),
            "tile_shuffle": torch.roll(latent, shifts=1, dims=1),
            "mean": latent.mean(dim=0, keepdim=True).expand_as(latent),
        }
        intervention_metrics: dict[str, dict[str, float]] = {
            "matched": matched_metrics
        }
        for name, intervened in interventions.items():
            metrics, _ = _decode_metrics(
                model,
                prepared,
                operators,
                intervened,
                matched_predictions,
                args.batch_operators,
                device,
            )
            intervention_metrics[name] = metrics
        alpha_metrics: dict[str, dict[str, float]] = {}
        for alpha in ALPHAS:
            metrics, _ = _decode_metrics(
                model,
                prepared,
                operators,
                latent * alpha,
                matched_predictions,
                args.batch_operators,
                device,
            )
            alpha_metrics[str(alpha)] = metrics
        zero_loss = intervention_metrics["zero"]["structural_loss"]
        matched_loss = matched_metrics["structural_loss"]
        operator_shuffle_loss = intervention_metrics["operator_shuffle"][
            "structural_loss"
        ]
        denominator = max(zero_loss - matched_loss, 1.0e-12)
        result["arms"][arm] = {
            "latent_rms": float(latent.square().mean().sqrt().item()),
            "interventions": intervention_metrics,
            "alpha_sweep": alpha_metrics,
            "summary": {
                "loss_reduction_vs_zero_fraction": (zero_loss - matched_loss)
                / max(zero_loss, 1.0e-12),
                "operator_specificity_fraction": (
                    operator_shuffle_loss - matched_loss
                )
                / denominator,
                "zero_minus_matched_loss": zero_loss - matched_loss,
                "operator_shuffle_minus_matched_loss": operator_shuffle_loss
                - matched_loss,
            },
        }
        del model, latent, matched_predictions
        torch.cuda.empty_cache()
    _atomic_json(root / "structural_latent_usage.json", result)
    print(
        f"[latent-usage] stage=complete artifact={root / 'structural_latent_usage.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
