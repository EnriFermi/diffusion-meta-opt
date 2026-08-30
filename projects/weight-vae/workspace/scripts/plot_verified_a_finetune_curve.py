#!/usr/bin/env python3
"""Plot saved loss metrics for the verified hidden_dim=2048 A fine-tune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / (
    "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing/"
    "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_verified_nocap_v1_trainseed1/"
    "vae_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "verified_a_finetune_h2048/training_curves"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def endpoint_stats(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[a_finetune_curve] checkpoint={checkpoint}", flush=True)
    print(f"[a_finetune_curve] output_dir={output_dir}", flush=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = pd.DataFrame(payload["metrics"]).sort_values("step").reset_index(drop=True)

    columns = [
        "step",
        "train_loss",
        "train_recon_mse",
        "train_precond_a_loss",
        "train_precond_effective_loss",
        "train_precond_scale",
        "val_recon_mse",
    ]
    curve = metrics.loc[:, columns].copy()
    a_steps = curve.loc[curve["train_precond_a_loss"].notna() & (curve["step"] >= 10)].copy()
    if len(a_steps) != 500:
        raise ValueError(f"expected 500 logged A steps, found {len(a_steps)}")

    bin_width = 250
    max_step = int(a_steps["step"].max())
    edges = np.arange(0, max_step + bin_width + 1, bin_width)
    a_steps["step_bin"] = pd.cut(a_steps["step"], bins=edges, right=True, include_lowest=True)
    binned = (
        a_steps.groupby("step_bin", observed=True)
        .agg(
            step_start=("step", "min"),
            step_end=("step", "max"),
            count=("train_precond_a_loss", "size"),
            a_mean=("train_precond_a_loss", "mean"),
            a_median=("train_precond_a_loss", "median"),
            a_p90=("train_precond_a_loss", lambda x: x.quantile(0.90)),
            a_p95=("train_precond_a_loss", lambda x: x.quantile(0.95)),
            a_max=("train_precond_a_loss", "max"),
            recon_median=("train_recon_mse", "median"),
            total_median=("train_loss", "median"),
        )
        .reset_index(drop=True)
    )
    binned["step_center"] = (binned["step_start"] + binned["step_end"]) / 2.0

    curve_csv = output_dir / "training_curve_metrics.csv"
    binned_csv = output_dir / "training_curve_binned_250_steps.csv"
    curve.to_csv(curve_csv, index=False)
    binned.to_csv(binned_csv, index=False)

    x = a_steps["step"].to_numpy()
    raw_a = a_steps["train_precond_a_loss"].to_numpy()
    rolling_median = a_steps["train_precond_a_loss"].rolling(25, min_periods=10).median()
    rolling_mean = a_steps["train_precond_a_loss"].rolling(25, min_periods=10).mean()

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)

    ax = axes[0]
    ax.scatter(x, raw_a, s=12, alpha=0.22, color="#2673b8", label="sampled A (every 10 steps)")
    ax.plot(x, rolling_median, color="#111111", linewidth=2.2, label="rolling median (25 A steps)")
    ax.plot(x, rolling_mean, color="#d1495b", linewidth=1.8, label="rolling mean (25 A steps)")
    ax.set_yscale("log")
    ax.set_ylabel("raw sampled A-surrogate")
    ax.set_title("A-surrogate used for training (log scale)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right")

    ax = axes[1]
    centers = binned["step_center"].to_numpy()
    ax.plot(centers, binned["a_median"], marker="o", linewidth=2.2, color="#111111", label="median")
    ax.plot(centers, binned["a_p90"], marker="o", linewidth=1.8, color="#f28e2b", label="p90")
    ax.plot(centers, binned["a_mean"], marker="o", linewidth=1.8, color="#d1495b", label="mean")
    ax.set_yscale("log")
    ax.set_ylabel("A-surrogate per 250-step bin")
    ax.set_title("Binned A statistics: heavy-tail behavior is shown explicitly")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.scatter(
        a_steps["step"],
        a_steps["train_recon_mse"],
        s=10,
        alpha=0.18,
        color="#59a14f",
        label="train reconstruction batch",
    )
    ax.plot(
        a_steps["step"],
        a_steps["train_recon_mse"].rolling(25, min_periods=10).median(),
        linewidth=2.0,
        color="#2f7d32",
        label="train reconstruction rolling median",
    )
    val = curve.loc[curve["val_recon_mse"].notna()]
    ax.plot(
        val["step"],
        val["val_recon_mse"],
        marker="o",
        linewidth=2.0,
        color="#8f3f97",
        label="validation reconstruction",
    )
    ax.set_xlabel("fine-tune step")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("VAE reconstruction during the same fine-tune")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")

    figure_path = output_dir / "finetune_loss_curves.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    first = a_steps.iloc[:100]["train_precond_a_loss"]
    last = a_steps.iloc[-100:]["train_precond_a_loss"]
    summary = {
        "checkpoint": str(checkpoint),
        "logged_a_steps": int(len(a_steps)),
        "a_step_interval": 10,
        "first_100_a_steps": endpoint_stats(first),
        "last_100_a_steps": endpoint_stats(last),
        "validation_recon_start": float(val.iloc[0]["val_recon_mse"]),
        "validation_recon_end": float(val.iloc[-1]["val_recon_mse"]),
        "artifacts": {
            "figure": str(figure_path),
            "curve_csv": str(curve_csv),
            "binned_csv": str(binned_csv),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[a_finetune_curve] wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
