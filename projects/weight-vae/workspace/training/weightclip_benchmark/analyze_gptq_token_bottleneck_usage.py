from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    ExperimentConfig,
    UnifiedWeightBottleneck,
    _arm_content,
    _arm_log_scale,
    _atomic_json,
    _load_exact64_tiles,
    _prepare_representations,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether the decoder uses the saved bottleneck.")
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def _action_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
) -> tuple[float, list[float]]:
    predicted_action = torch.einsum("btri,btic->btrc", context, prediction).sum(dim=1)
    target_action = torch.einsum("btri,btic->btrc", context, target).sum(dim=1)
    numerator = (predicted_action - target_action).square().sum(dim=(1, 2))
    denominator = target_action.square().sum(dim=(1, 2)).clamp_min(1.0e-12)
    values = torch.sqrt(numerator / denominator)
    global_value = float(torch.sqrt(numerator.sum() / denominator.sum()).item())
    return global_value, values.cpu().tolist()


def main() -> None:
    args = _parse_args()
    root = args.run_root.expanduser().resolve()
    startup = json.loads((root / "config.json").read_text(encoding="utf-8"))
    run_arms = tuple(str(arm) for arm in startup["arms"])
    cfg = ExperimentConfig(**startup["config"])
    device = torch.device("cuda:0")
    print(f"[gptq-usage] stage=load run_root={root} device={device}", flush=True)
    data = _load_exact64_tiles(Path(startup["selection"]), Path(startup["resolved_config"]))
    prepared = _prepare_representations(data, cfg, device)
    # Keep all checkpoints on CPU and materialize only the current arm.
    # Loading all optimizer-sized models onto CUDA would waste most of the H100.
    saved = torch.load(root / "final_models.pt", map_location="cpu", weights_only=True)
    heldout = list(prepared["heldout_operator_indices"])
    flat_indices = torch.tensor(
        [operator * 9 + tile for operator in heldout for tile in range(9)], dtype=torch.long
    )
    tile_rows = prepared["flat_tile_rows"].index_select(0, flat_indices).to(device)
    target = prepared["flat_weights"].index_select(0, flat_indices).to(device).view(
        len(heldout), 9, 128, 128
    )
    context = prepared["flat_contexts"].index_select(0, flat_indices)[
        :, cfg.train_activation_rows :
    ].to(device).view(len(heldout), 9, cfg.test_activation_rows, 128)
    result: dict[str, object] = {
        "schema": "weightclip_gptq_token_bottleneck_usage_v1",
        "run_root": str(root),
        "heldout_operator_indices": heldout,
        "arms": {},
    }
    rng = np.random.default_rng(cfg.seed)
    for arm in run_arms:
        print(f"[gptq-usage] stage=arm arm={arm}", flush=True)
        model = UnifiedWeightBottleneck(cfg, arm).to(device)
        model.load_state_dict(saved[arm], strict=True)
        model.eval()
        content = _arm_content(prepared, arm, flat_indices).to(device)
        scale = _arm_log_scale(prepared, arm, flat_indices).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _telemetry = model.encode(content, scale, tile_rows)
            matched = model.decode(z, tile_rows)
            zero = model.decode(torch.zeros_like(z), tile_rows)
            z_by_operator = z.view(len(heldout), 9, cfg.latent_slots, cfg.latent_dim)
            shuffled_z = torch.roll(z_by_operator, shifts=1, dims=0).reshape_as(z)
            shuffled = model.decode(shuffled_z, tile_rows)
        predictions = {
            "matched": matched.float().view(len(heldout), 9, 128, 128),
            "operator_cyclic_shuffle": shuffled.float().view(len(heldout), 9, 128, 128),
            "zero_latent": zero.float().view(len(heldout), 9, 128, 128),
        }
        metrics: dict[str, object] = {}
        for name, prediction in predictions.items():
            global_nrmse, per_operator = _action_metrics(
                prediction, target.float(), context.float()
            )
            metrics[name] = {
                "global_action_nrmse": global_nrmse,
                "operator_action_nrmse": per_operator,
            }
        matched_values = np.asarray(metrics["matched"]["operator_action_nrmse"])
        shuffled_values = np.asarray(
            metrics["operator_cyclic_shuffle"]["operator_action_nrmse"]
        )
        zero_values = np.asarray(metrics["zero_latent"]["operator_action_nrmse"])
        bootstrap_shuffle: list[float] = []
        bootstrap_zero: list[float] = []
        for _ in range(20_000):
            indices = rng.integers(0, len(heldout), len(heldout))
            bootstrap_shuffle.append(float((shuffled_values - matched_values)[indices].mean()))
            bootstrap_zero.append(float((zero_values - matched_values)[indices].mean()))
        metrics["causal_gaps"] = {
            "shuffle_minus_matched_mean": float((shuffled_values - matched_values).mean()),
            "shuffle_minus_matched_median": float(np.median(shuffled_values - matched_values)),
            "shuffle_minus_matched_bootstrap95": np.quantile(
                bootstrap_shuffle, [0.025, 0.975]
            ).tolist(),
            "zero_minus_matched_mean": float((zero_values - matched_values).mean()),
            "zero_minus_matched_median": float(np.median(zero_values - matched_values)),
            "zero_minus_matched_bootstrap95": np.quantile(
                bootstrap_zero, [0.025, 0.975]
            ).tolist(),
            "latent_operator_shuffle_relative_delta": float(
                (z_by_operator - torch.roll(z_by_operator, shifts=1, dims=0))
                .float()
                .square()
                .mean()
                .sqrt()
                .div(z_by_operator.float().square().mean().sqrt().clamp_min(1.0e-12))
                .item()
            ),
        }
        result["arms"][arm] = metrics
        del model, content, scale, z, matched, zero, shuffled, predictions
        torch.cuda.empty_cache()
    _atomic_json(root / "bottleneck_usage.json", result)
    print(f"[gptq-usage] stage=complete artifact={root / 'bottleneck_usage.json'}", flush=True)


if __name__ == "__main__":
    main()
