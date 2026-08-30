#!/usr/bin/env python3
"""Review two WeightCLIP-style ResNet18Slim lineage training runs.

The probe distinguishes *45 training epochs with two retained terminal
snapshots* from the incorrect interpretation "two-epoch training".  It also
quantifies whether epochs 44--45 are representative of the late trajectory or
whether the five-snapshot paper population contains materially more movement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


METRIC_COLUMNS = (
    "train_loss",
    "val_loss",
    "test_loss",
    "train_acc",
    "val_acc",
    "test_acc",
    "learning_rate",
    "time_per_epoch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--late-start", type=int, default=40, help="Zero-based first checkpoint of the five-snapshot diagnostic window.")
    return parser.parse_args()


def read_progress(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def flatten_state(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    chunks = []
    for key in sorted(payload):
        value = payload[key]
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            # A 2.8M-parameter cosine accumulated in float32 can round above
            # one, yielding an impossible negative ``1 - cosine`` diagnostic.
            # Use float64 for the review metric; training/checkpoints stay FP32.
            chunks.append(value.detach().to(dtype=torch.float64).reshape(-1))
    if not chunks:
        raise ValueError(f"No floating tensors in {path}")
    return torch.cat(chunks)


def finite_slope(rows: list[dict[str, float]], key: str) -> float:
    xs = np.asarray([row["epoch"] for row in rows], dtype=np.float64)
    ys = np.asarray([row[key] for row in rows], dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(xs[mask], ys[mask], 1)[0])


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and abs(den) > 0 else float("nan")


def analyze_seed(seed_dir: Path, late_start: int) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], dict[str, object]]:
    seed = int(seed_dir.name.rsplit("=", 1)[1])
    metrics = read_progress(seed_dir / "progress.csv")
    epoch_rows = [{"seed": seed, **row} for row in metrics]
    last_five = [row for row in metrics if int(row["epoch"]) >= late_start]
    last_two = [row for row in metrics if int(row["epoch"]) >= max(row["epoch"] for row in metrics) - 1]

    drift_rows: list[dict[str, float | int]] = []
    vectors: dict[int, torch.Tensor] = {}
    for epoch in range(late_start, int(max(row["epoch"] for row in metrics)) + 1):
        state_path = seed_dir / f"checkpoint_{epoch:06d}" / "checkpoints"
        if state_path.exists():
            vectors[epoch] = flatten_state(state_path)
    for previous, current in zip(sorted(vectors)[:-1], sorted(vectors)[1:]):
        before, after = vectors[previous], vectors[current]
        delta = after - before
        drift_rows.append(
            {
                "seed": seed,
                "from_epoch": previous,
                "to_epoch": current,
                "absolute_l2": float(torch.linalg.vector_norm(delta)),
                "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(before)),
                "cosine": float(torch.nn.functional.cosine_similarity(before, after, dim=0).clamp(-1.0, 1.0)),
            }
        )

    first_epoch, last_epoch = min(vectors), max(vectors)
    late_span_relative_l2 = float(torch.linalg.vector_norm(vectors[last_epoch] - vectors[first_epoch]) / torch.linalg.vector_norm(vectors[first_epoch]))
    terminal_step = next((float(row["relative_l2"]) for row in drift_rows if int(row["to_epoch"]) == last_epoch), float("nan"))
    summary: dict[str, object] = {
        "seed": seed,
        "epochs_logged": len(metrics),
        "first_epoch_zero_based": int(min(row["epoch"] for row in metrics)),
        "last_epoch_zero_based": int(max(row["epoch"] for row in metrics)),
        "late_checkpoint_epochs_zero_based": sorted(vectors),
        "terminal_step_relative_l2": terminal_step,
        "five_snapshot_span_relative_l2": late_span_relative_l2,
        "terminal_step_over_five_snapshot_span": safe_ratio(terminal_step, late_span_relative_l2),
        "last_five": {},
        "last_two": {},
    }
    for key in METRIC_COLUMNS:
        five_values = [row[key] for row in last_five if math.isfinite(row[key])]
        two_values = [row[key] for row in last_two if math.isfinite(row[key])]
        summary["last_five"][key] = {
            "slope_per_epoch": finite_slope(last_five, key),
            "range": float(max(five_values) - min(five_values)) if five_values else float("nan"),
            "final": float(five_values[-1]) if five_values else float("nan"),
        }
        summary["last_two"][key] = {
            "delta": float(two_values[-1] - two_values[0]) if len(two_values) >= 2 else float("nan"),
            "range": float(max(two_values) - min(two_values)) if two_values else float("nan"),
        }
    return epoch_rows, drift_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(epoch_rows: list[dict[str, object]], output: Path, late_start: int) -> None:
    seeds = sorted({int(row["seed"]) for row in epoch_rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for seed in seeds:
        rows = [row for row in epoch_rows if int(row["seed"]) == seed]
        epochs = [int(row["epoch"]) + 1 for row in rows]
        for key in ("train_loss", "val_loss", "test_loss"):
            axes[0, 0].plot(epochs, [float(row[key]) for row in rows], label=f"s{seed} {key}")
        for key in ("train_acc", "val_acc", "test_acc"):
            axes[0, 1].plot(epochs, [float(row[key]) for row in rows], label=f"s{seed} {key}")
        axes[1, 0].plot(epochs, [float(row["learning_rate"]) for row in rows], label=f"seed {seed}")
        late = [row for row in rows if int(row["epoch"]) >= late_start]
        axes[1, 1].plot([int(row["epoch"]) + 1 for row in late], [float(row["test_acc"]) for row in late], marker="o", label=f"seed {seed}")
    axes[0, 0].set(title="Loss curves", xlabel="one-based epoch", ylabel="cross entropy")
    axes[0, 1].set(title="Accuracy curves", xlabel="one-based epoch", ylabel="accuracy")
    axes[1, 0].set(title="OneCycle learning rate", xlabel="one-based epoch", ylabel="LR")
    axes[1, 1].set(title="Late-window test accuracy", xlabel="one-based epoch", ylabel="accuracy")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("WeightCLIP ResNet18Slim lineage probe: full 45-epoch runs")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_drift(drift_rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in drift_rows}):
        rows = [row for row in drift_rows if int(row["seed"]) == seed]
        x = [int(row["to_epoch"]) + 1 for row in rows]
        axes[0].plot(x, [float(row["relative_l2"]) for row in rows], marker="o", label=f"seed {seed}")
        axes[1].plot(x, [1.0 - float(row["cosine"]) for row in rows], marker="o", label=f"seed {seed}")
    axes[0].set(title="Consecutive checkpoint displacement", xlabel="one-based destination epoch", ylabel="||W_t-W_(t-1)|| / ||W_(t-1)||")
    axes[1].set(title="Consecutive checkpoint angular change", xlabel="one-based destination epoch", ylabel="1 - cosine(W_t, W_(t-1))")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def review_lr_sweep(path: Path, output: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    candidates = payload["candidates"]
    lrs = np.asarray([float(item["lr"]) for item in candidates])
    means = np.asarray([float(item["mean_final_test_acc"]) for item in candidates])
    stds = np.asarray([float(item["std_final_test_acc"]) for item in candidates])
    robust = np.asarray([float(item["robust_final_test_acc"]) for item in candidates])
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.errorbar(lrs, means, yerr=stds, marker="o", capsize=4, label="mean ± std (2 seeds)")
    axis.plot(lrs, robust, marker="s", linestyle="--", label="released selector: mean - std")
    for item in candidates:
        for run in item.get("runs", []):
            axis.scatter(float(item["lr"]), float(run["final_test_acc"]), color="black", alpha=0.55, s=18)
    chosen = float(payload["chosen_lr"])
    axis.axvline(chosen, color="tab:red", linestyle=":", label=f"chosen LR={chosen:g}")
    axis.set(xlabel="OneCycle max LR", ylabel="final test accuracy", title="Released test-selected LR sweep")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "chosen_lr": chosen,
        "selection_strategy": payload.get("selection_strategy"),
        "chosen_is_grid_boundary": bool(chosen in {float(lrs.min()), float(lrs.max())}),
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(args.zoo_dir.glob("NN_tune_trainable_seed=*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No final seed directories under {args.zoo_dir}")
    print(f"[probe] zoo_dir={args.zoo_dir}")
    print(f"[probe] output_dir={args.out_dir}")
    print(f"[probe] seeds={[path.name for path in seed_dirs]}")
    all_epochs: list[dict[str, object]] = []
    all_drift: list[dict[str, object]] = []
    summaries = []
    for seed_dir in seed_dirs:
        print(f"[probe] analyzing {seed_dir.name}")
        epoch_rows, drift_rows, summary = analyze_seed(seed_dir, args.late_start)
        all_epochs.extend(epoch_rows)
        all_drift.extend(drift_rows)
        summaries.append(summary)
    write_csv(args.out_dir / "epoch_metrics.csv", all_epochs)
    write_csv(args.out_dir / "checkpoint_drift.csv", all_drift)
    sweep_review = review_lr_sweep(args.zoo_dir / "lr_sweep" / "summary.json", args.out_dir / "lr_sweep.png")
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump({"late_start_zero_based": args.late_start, "lr_sweep": sweep_review, "lineages": summaries}, handle, indent=2, allow_nan=True)
    plot_curves(all_epochs, args.out_dir / "training_curves.png", args.late_start)
    plot_drift(all_drift, args.out_dir / "late_checkpoint_drift.png")
    print(f"[probe] wrote {args.out_dir / 'summary.json'}")
    print(f"[probe] wrote {args.out_dir / 'training_curves.png'}")
    print(f"[probe] wrote {args.out_dir / 'late_checkpoint_drift.png'}")
    if sweep_review is not None:
        print(f"[probe] wrote {args.out_dir / 'lr_sweep.png'}")


if __name__ == "__main__":
    main()
