#!/usr/bin/env python3
"""Read-only trajectory audit for the parallel categorical GPTQ production run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

# This repository has ``scripts/inspect``. When a script is executed from the
# scripts directory, that package otherwise shadows Python's stdlib ``inspect``
# and breaks matplotlib/scipy imports.
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _SCRIPT_DIRECTORY:
    sys.path.pop(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


PFX = "diagnostic_old_direction_contrastive_"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--plateau-start", type=int, default=510)
    return parser.parse_args()


def _read_complete_rows(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                # The trainer appends one compact line at a time. A partial final
                # line can only be the concurrently-written tail and is skipped.
                if handle.readline() == "":
                    break
                raise RuntimeError(f"invalid JSON at line {line_number}") from error
    frame = pd.DataFrame(rows).sort_values("step")
    return frame.drop_duplicates("step", keep="last").reset_index(drop=True)


def _sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def _fixed_signature_slope(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    signature_columns = [
        PFX + "eligible_samples",
        PFX + "group_count",
        PFX + "group_size_min",
        PFX + "group_size_median",
        PFX + "group_size_max",
    ]
    work = frame[["step", metric, *signature_columns]].dropna().copy()
    work["signature"] = list(map(tuple, work[signature_columns].to_numpy()))
    counts = work.groupby("signature")["signature"].transform("size")
    work = work[counts >= 2].copy()
    x = work["step"] / 1000.0
    y = work[metric]
    x_centered = x - work.groupby("signature")["step"].transform("mean") / 1000.0
    y_centered = y - work.groupby("signature")[metric].transform("mean")
    denominator = float(np.square(x_centered).sum())
    slope = float((x_centered * y_centered).sum() / denominator)
    residual = y_centered - slope * x_centered
    group_count = int(work["signature"].nunique())
    degrees_of_freedom = max(len(work) - group_count - 1, 1)
    standard_error = math.sqrt(
        float(np.square(residual).sum()) / degrees_of_freedom / denominator
    )
    return {
        "slope_per_1000_steps": slope,
        "standard_error": standard_error,
        "rows": int(len(work)),
        "signatures": group_count,
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_stat = args.metrics.stat()
    frame = _read_complete_rows(args.metrics)
    if frame.empty:
        raise RuntimeError("metrics file is empty")

    frame["contrastive_chance_top1"] = (
        frame[PFX + "group_count"] / frame[PFX + "eligible_samples"]
    )
    frame["contrastive_top1_advantage"] = (
        frame[PFX + "top1_accuracy"] - frame["contrastive_chance_top1"]
    )
    frame["contrastive_diagonal_minus_offdiagonal"] = (
        frame[PFX + "diagonal_similarity_mean"]
        - frame[PFX + "offdiagonal_similarity_mean"]
    )
    frame["behavioral_direction_alignment"] = (
        1.0 - frame["diagnostic_old_behavioral_direction"]
    )
    frame["structural_direction_alignment"] = (
        1.0 - frame["diagnostic_old_structural_direction"]
    )

    latest_step = int(frame["step"].max())
    windows = [
        (510, 1000, "early_prior_plateau"),
        (1010, 1500, "escape_onset"),
        (1510, 2000, "early_escape"),
        (2010, 2500, "mid_escape"),
        (2510, 3000, "late_escape"),
        (3010, latest_step, "latest"),
    ]
    summary_metrics = [
        "loss",
        "code_ordinal_loss",
        "scale_ordinal_loss",
        "code_accuracy",
        "code_off_by_one_accuracy",
        "code_mean_absolute_bin_error",
        "hard_decode_raw_nrmse",
        "diagnostic_old_legacy_task_loss",
        "diagnostic_old_behavioral_direction",
        "diagnostic_old_structural_direction",
        "behavioral_direction_alignment",
        "structural_direction_alignment",
        PFX + "loss",
        PFX + "top1_accuracy",
        "contrastive_chance_top1",
        "contrastive_top1_advantage",
        PFX + "diagonal_similarity_mean",
        PFX + "offdiagonal_similarity_mean",
        "contrastive_diagonal_minus_offdiagonal",
        PFX + "diagonal_minus_hardest_negative_mean",
        "diagnostic_old_behavioral_scale",
        "diagnostic_old_structural_scale",
        "diagnostic_old_behavioral_operator_actual",
        "diagnostic_old_latent_interaction_rms_ratio",
        "diagnostic_old_latent_rms",
        "teacher_continuous_scale_raw_nrmse",
        "teacher_continuous_scale_operator_metric",
        "code_target_low_boundary_fraction",
        "code_target_high_boundary_fraction",
        "teacher_scale_log2_min",
        "teacher_scale_log2_max",
    ]
    window_rows: list[dict[str, float | int | str]] = []
    for start, stop, label in windows:
        current = frame[(frame["step"] >= start) & (frame["step"] <= stop)]
        if current.empty:
            continue
        for metric in summary_metrics:
            window_rows.append(
                {
                    "window": label,
                    "step_start": int(current["step"].min()),
                    "step_stop": int(current["step"].max()),
                    "rows": int(len(current)),
                    "metric": metric,
                    "mean": float(current[metric].mean()),
                    "sem": _sem(current[metric]),
                    "std": float(current[metric].std(ddof=1)),
                }
            )
    pd.DataFrame(window_rows).to_csv(args.output_dir / "window_summary.csv", index=False)

    plateau = frame[frame["step"] >= int(args.plateau_start)].copy()
    slope_metrics = [
        "loss",
        "code_ordinal_loss",
        "scale_ordinal_loss",
        "hard_decode_raw_nrmse",
        "diagnostic_old_behavioral_direction",
        "diagnostic_old_structural_direction",
        PFX + "loss",
        PFX + "top1_accuracy",
        "contrastive_chance_top1",
        "contrastive_top1_advantage",
        PFX + "diagonal_similarity_mean",
        PFX + "offdiagonal_similarity_mean",
        "contrastive_diagonal_minus_offdiagonal",
        PFX + "diagonal_minus_hardest_negative_mean",
        "diagnostic_old_behavioral_scale",
        "diagnostic_old_structural_scale",
        "diagnostic_old_behavioral_operator_actual",
        "diagnostic_old_latent_interaction_rms_ratio",
        "diagnostic_old_latent_rms",
    ]
    slope_rows = []
    for metric in slope_metrics:
        fit = linregress(plateau["step"] / 1000.0, plateau[metric])
        row: dict[str, float | int | str] = {
            "metric": metric,
            "slope_per_1000_steps": float(fit.slope),
            "standard_error": float(fit.stderr),
            "p_value": float(fit.pvalue),
            "rows": int(len(plateau)),
        }
        if metric.startswith(PFX) or metric.startswith("contrastive_"):
            matched = _fixed_signature_slope(plateau, metric)
            row.update({f"signature_fixed_{key}": value for key, value in matched.items()})
        slope_rows.append(row)
    pd.DataFrame(slope_rows).to_csv(args.output_dir / "plateau_slopes.csv", index=False)

    metadata = {
        "schema": "parallel_categorical_gptq_diagnostic_trajectory_v1",
        "source": str(args.metrics),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "rows": int(len(frame)),
        "first_step": int(frame["step"].min()),
        "latest_step": latest_step,
        "plateau_start": int(args.plateau_start),
        "notes": [
            "All old losses are detached diagnostics and do not contribute to backward.",
            "Contrastive chance top-1 is exact for random ranking: group_count / eligible_samples.",
            "Signature-fixed slopes condition on eligible count, group count, and min/median/max group size.",
        ],
    }
    (args.output_dir / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    plot_frame = frame[frame["step"] >= int(args.plateau_start)].copy()
    rolling = plot_frame.set_index("step").select_dtypes(include=[np.number]).rolling(
        20, min_periods=5
    ).mean()
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 13.0), sharex=True)
    axes[0].plot(
        plot_frame["step"], plot_frame["code_ordinal_loss"], alpha=0.16, color="C0"
    )
    axes[0].plot(rolling.index, rolling["code_ordinal_loss"], label="code ordinal", color="C0")
    axes[0].plot(rolling.index, rolling["loss"], label="total ordinal", color="C1")
    axes[0].axhline(0.083400, color="black", linestyle="--", linewidth=1, label="sealed code prior")
    axes[0].set_ylabel("optimized loss")
    axes[0].legend(loc="best")

    axes[1].plot(rolling.index, rolling["behavioral_direction_alignment"], label="1 - behavioral dir loss")
    axes[1].plot(rolling.index, rolling["structural_direction_alignment"], label="1 - structural dir loss")
    axes[1].set_ylabel("absolute cosine alignment")
    axes[1].legend(loc="best")

    axes[2].plot(rolling.index, rolling[PFX + "top1_accuracy"], label="same-layout top-1")
    axes[2].plot(rolling.index, rolling["contrastive_chance_top1"], linestyle="--", label="composition chance")
    twin = axes[2].twinx()
    twin.plot(
        rolling.index,
        rolling[PFX + "diagonal_minus_hardest_negative_mean"],
        color="C2",
        label="diag - hardest negative",
    )
    axes[2].set_ylabel("contrastive top-1")
    twin.set_ylabel("similarity gap")
    handles, labels = axes[2].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[2].legend(handles + handles2, labels + labels2, loc="best")

    axes[3].plot(
        rolling.index,
        rolling["diagnostic_old_latent_interaction_rms_ratio"],
        label="latent interaction ratio",
        color="C3",
    )
    twin = axes[3].twinx()
    twin.plot(
        rolling.index,
        rolling["hard_decode_raw_nrmse"] - 1.0,
        label="hard NRMSE - 1",
        color="C4",
    )
    axes[3].set_ylabel("latent interaction ratio")
    twin.set_ylabel("NRMSE excess over zero")
    axes[3].set_xlabel("optimizer step")
    handles, labels = axes[3].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[3].legend(handles + handles2, labels + labels2, loc="best")

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"Categorical production after prior convergence: steps {args.plateau_start}--{latest_step}"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "trajectory.png", dpi=170)
    plt.close(fig)

    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output_dir / 'window_summary.csv'}")
    print(f"wrote {args.output_dir / 'plateau_slopes.csv'}")
    print(f"wrote {args.output_dir / 'trajectory.png'}")


if __name__ == "__main__":
    main()
