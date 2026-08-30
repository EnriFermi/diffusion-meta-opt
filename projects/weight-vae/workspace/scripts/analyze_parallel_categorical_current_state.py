#!/usr/bin/env python3
"""Read-only robust trajectory audit for the live categorical GPTQ run.

The trainer may append while this script is reading.  Only complete JSONL rows
are admitted, and the source ``size + mtime_ns`` before/after the read is
recorded instead of copying or hashing the live files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _SCRIPT_DIRECTORY:
    # ``scripts/inspect`` otherwise shadows the Python stdlib module.
    sys.path.pop(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes


CONTRASTIVE_PREFIX = "diagnostic_old_direction_contrastive_"
BIN_WIDTH = 1_000
MIN_SEGMENT_BINS = 5
SEGMENTATION_PENALTIES = (25.0, 30.0, 35.0, 40.0)
SEGMENTATION_METRICS = (
    "code_ordinal_loss",
    "scale_ordinal_loss",
    "hard_decode_raw_nrmse",
    "diagnostic_old_behavioral_direction",
    "diagnostic_old_structural_direction",
    "diagnostic_old_direction_contrastive_loss",
    "diagnostic_old_latent_interaction_rms_ratio",
    "diagnostic_old_latent_rms",
    "diagnostic_old_behavioral_operator_actual",
    "grad_norm_pre_clip",
)

SUMMARY_METRICS = (
    "loss",
    "code_ordinal_loss",
    "scale_ordinal_loss",
    "scale_loss_fraction",
    "code_accuracy",
    "code_off_by_one_accuracy",
    "code_mean_absolute_bin_error",
    "scale_accuracy",
    "scale_off_by_one_accuracy",
    "scale_mean_absolute_bin_error",
    "hard_decode_raw_nrmse",
    "diagnostic_old_behavioral_operator_actual",
    "operator_over_teacher_floor",
    "diagnostic_old_behavioral_direction",
    "diagnostic_old_structural_direction",
    "diagnostic_old_direction_contrastive_loss",
    "diagnostic_old_direction_contrastive_top1_accuracy",
    "contrastive_chance_top1",
    "contrastive_top1_advantage",
    "diagnostic_old_direction_contrastive_diagonal_minus_hardest_negative_mean",
    "diagnostic_old_latent_interaction_rms_ratio",
    "latent_interaction_rms_estimate",
    "diagnostic_old_latent_rms",
    "grad_norm_pre_clip",
    "teacher_continuous_scale_raw_nrmse",
    "teacher_continuous_scale_operator_metric",
    "steps_per_second",
    "gptq_tokenizer_seconds",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("gradients", type=Path)
    parser.add_argument("resolved_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_complete_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    partial_tail_skipped = False
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError as error:
                # A concurrently appended final line is allowed to be partial.
                remainder = handle.read(1)
                if remainder == b"":
                    partial_tail_skipped = True
                    break
                raise RuntimeError(
                    f"invalid non-final JSON in {path} at line {line_number}"
                ) from error
    return rows, {"partial_tail_skipped": partial_tail_skipped}


def _augment_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["contrastive_chance_top1"] = (
        frame[CONTRASTIVE_PREFIX + "group_count"]
        / frame[CONTRASTIVE_PREFIX + "eligible_samples"]
    )
    frame["contrastive_top1_advantage"] = (
        frame[CONTRASTIVE_PREFIX + "top1_accuracy"]
        - frame["contrastive_chance_top1"]
    )
    frame["latent_interaction_rms_estimate"] = (
        frame["diagnostic_old_latent_interaction_rms_ratio"]
        * frame["diagnostic_old_latent_rms"]
    )
    frame["scale_loss_fraction"] = frame["scale_ordinal_loss"] / frame["loss"]
    frame["operator_over_teacher_floor"] = (
        frame["diagnostic_old_behavioral_operator_actual"]
        / frame["teacher_continuous_scale_operator_metric"]
    )
    frame["behavioral_direction_alignment"] = (
        1.0 - frame["diagnostic_old_behavioral_direction"]
    )
    frame["structural_direction_alignment"] = (
        1.0 - frame["diagnostic_old_structural_direction"]
    )
    frame["local_steps_per_second"] = frame["step"].diff() / frame[
        "elapsed_seconds"
    ].diff()
    return frame


def _flatten_gradient_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {"step": int(row["step"]), "schema": row["schema"]}
        flat.update(row["latent_code_scale_gradients"])
        for group, values in row["groups"].items():
            if isinstance(values, dict) and "gradient_rms" in values:
                flat[group] = float(values["gradient_rms"])
            elif group in {"none_parameter_tensors"}:
                flat[group] = int(values)
        flattened.append(flat)
    return pd.DataFrame(flattened)


def _robust_bins(frame: pd.DataFrame, width: int = BIN_WIDTH) -> pd.DataFrame:
    work = frame.copy()
    work["bin_end"] = ((work["step"] - 1) // width + 1) * width
    numeric = work.select_dtypes(include=[np.number]).columns.tolist()
    if "bin_end" in numeric:
        numeric.remove("bin_end")
    median = work.groupby("bin_end", sort=True)[numeric].median()
    counts = work.groupby("bin_end", sort=True).size().rename("rows")
    step_min = work.groupby("bin_end", sort=True)["step"].min().rename("step_min")
    step_max = work.groupby("bin_end", sort=True)["step"].max().rename("step_max")
    result = pd.concat([counts, step_min, step_max, median], axis=1).reset_index()
    result["complete"] = result["step_max"] >= result["bin_end"]
    return result


def _linear_segment_costs(values: np.ndarray, minimum: int) -> np.ndarray:
    count = len(values)
    costs = np.full((count + 1, count + 1), np.inf, dtype=np.float64)
    x_all = np.arange(count, dtype=np.float64)
    for start in range(count):
        for stop in range(start + minimum, count + 1):
            x = x_all[start:stop]
            y = values[start:stop]
            x_centered = x - x.mean()
            y_mean = y.mean(axis=0)
            denominator = float(np.square(x_centered).sum())
            slopes = (
                (x_centered[:, None] * (y - y_mean)).sum(axis=0) / denominator
            )
            residual = y - (y_mean + x_centered[:, None] * slopes)
            costs[start, stop] = float(np.square(residual).sum())
    return costs


def _segment_end_indices(costs: np.ndarray, penalty: float, minimum: int) -> list[int]:
    count = costs.shape[0] - 1
    score = np.full(count + 1, np.inf)
    previous = np.full(count + 1, -1, dtype=np.int64)
    score[0] = -penalty
    for stop in range(minimum, count + 1):
        candidates = [
            start
            for start in range(0, stop - minimum + 1)
            if math.isfinite(float(score[start]))
        ]
        candidate_scores = [
            float(score[start] + costs[start, stop] + penalty)
            for start in candidates
        ]
        if candidate_scores:
            best = int(np.argmin(candidate_scores))
            score[stop] = candidate_scores[best]
            previous[stop] = candidates[best]
    ends: list[int] = []
    cursor = count
    while cursor > 0:
        ends.append(cursor)
        cursor = int(previous[cursor])
        if cursor < 0:
            raise RuntimeError("segmentation backtracking failed")
    return list(reversed(ends))


def _detect_phase_changes(complete_bins: pd.DataFrame) -> dict[str, Any]:
    raw = complete_bins.loc[:, SEGMENTATION_METRICS].to_numpy(dtype=np.float64)
    medians = np.median(raw, axis=0)
    mad = np.median(np.abs(raw - medians), axis=0)
    scale = 1.4826 * mad
    scale[scale < 1.0e-12] = 1.0
    standardized = (raw - medians) / scale
    costs = _linear_segment_costs(standardized, MIN_SEGMENT_BINS)
    terminal = len(complete_bins)
    by_penalty: dict[str, list[int]] = {}
    support: dict[int, int] = {}
    for penalty in SEGMENTATION_PENALTIES:
        ends = _segment_end_indices(costs, penalty, MIN_SEGMENT_BINS)
        step_ends = [int(complete_bins.iloc[index - 1]["bin_end"]) for index in ends]
        by_penalty[str(penalty)] = step_ends
        for index in ends:
            if index != terminal:
                step = int(complete_bins.iloc[index - 1]["bin_end"])
                support[step] = support.get(step, 0) + 1
    stable = sorted(
        step for step, count in support.items() if count >= len(SEGMENTATION_PENALTIES) - 1
    )
    return {
        "method": "joint robust-1k-bin piecewise-linear dynamic programming",
        "metrics": list(SEGMENTATION_METRICS),
        "standardization": "median / (1.4826 * MAD)",
        "minimum_segment_bins": MIN_SEGMENT_BINS,
        "penalties": list(SEGMENTATION_PENALTIES),
        "segment_ends_by_penalty": by_penalty,
        "nonterminal_breakpoint_support": {
            str(step): count for step, count in sorted(support.items())
        },
        "stable_breakpoints": stable,
    }


def _metric_summary_rows(
    frame: pd.DataFrame,
    windows: list[tuple[int, int, str]],
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start, stop, label in windows:
        current = frame[(frame["step"] >= start) & (frame["step"] <= stop)]
        if current.empty:
            continue
        for metric in metrics:
            values = current[metric].dropna()
            median = float(values.median())
            rows.append(
                {
                    "window": label,
                    "step_start": int(current["step"].min()),
                    "step_stop": int(current["step"].max()),
                    "rows": int(len(current)),
                    "metric": metric,
                    "median": median,
                    "mean": float(values.mean()),
                    "p10": float(values.quantile(0.10)),
                    "p90": float(values.quantile(0.90)),
                    "mad": float((values - median).abs().median()),
                }
            )
    return rows


def _phase_summary_rows(
    complete_bins: pd.DataFrame, phase_boundaries: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase_start = 1
    for phase_index, phase_stop in enumerate(phase_boundaries, start=1):
        current = complete_bins[
            (complete_bins["bin_end"] >= phase_start)
            & (complete_bins["bin_end"] <= phase_stop)
        ]
        if current.empty:
            continue
        x = current["bin_end"].to_numpy(dtype=np.float64) / 1_000.0
        for metric in SUMMARY_METRICS:
            y = current[metric].to_numpy(dtype=np.float64)
            slope = float(theilslopes(y, x).slope) if len(current) >= 2 else float("nan")
            rows.append(
                {
                    "phase": phase_index,
                    "step_start": int(current["step_min"].min()),
                    "step_stop": int(current["step_max"].max()),
                    "complete_1k_bins": int(len(current)),
                    "metric": metric,
                    "median_of_bin_medians": float(np.median(y)),
                    "first_bin_median": float(y[0]),
                    "last_bin_median": float(y[-1]),
                    "theil_sen_slope_per_1000_steps": slope,
                }
            )
        phase_start = phase_stop + 1
    return rows


def _gradient_phase_rows(
    gradient_frame: pd.DataFrame, phase_boundaries: list[int]
) -> list[dict[str, Any]]:
    metrics = [
        column
        for column in gradient_frame.columns
        if column not in {"step", "schema"}
        and np.issubdtype(gradient_frame[column].dtype, np.number)
    ]
    rows: list[dict[str, Any]] = []
    phase_start = 1
    for phase_index, phase_stop in enumerate(phase_boundaries, start=1):
        current = gradient_frame[
            (gradient_frame["step"] >= phase_start)
            & (gradient_frame["step"] <= phase_stop)
        ]
        for metric in metrics:
            values = current[metric].dropna()
            rows.append(
                {
                    "phase": phase_index,
                    "step_start": int(current["step"].min()),
                    "step_stop": int(current["step"].max()),
                    "rows": int(len(current)),
                    "metric": metric,
                    "median": float(values.median()),
                    "p10": float(values.quantile(0.10)),
                    "p90": float(values.quantile(0.90)),
                }
            )
        phase_start = phase_stop + 1
    return rows


def _add_phase_lines(axes: np.ndarray, breakpoints: list[int]) -> None:
    for axis in axes.flat:
        for step in breakpoints:
            axis.axvline(step, color="0.35", linestyle=":", linewidth=1.0)
        axis.grid(alpha=0.22)


def _plot_overview(
    bins: pd.DataFrame, breakpoints: list[int], output_path: Path
) -> None:
    x = bins["bin_end"]
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 14.0), sharex=True)

    axes[0, 0].plot(x, bins["code_ordinal_loss"], label="code ordinal", color="C0")
    twin = axes[0, 0].twinx()
    twin.plot(x, bins["scale_ordinal_loss"], label="scale ordinal", color="C1")
    axes[0, 0].set_ylabel("code loss")
    twin.set_ylabel("scale loss")
    axes[0, 0].set_title("Optimized objective")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[0, 0].legend(handles + handles2, labels + labels2)

    axes[0, 1].plot(x, bins["code_mean_absolute_bin_error"], label="code MAE")
    axes[0, 1].plot(x, bins["scale_mean_absolute_bin_error"], label="scale MAE")
    axes[0, 1].set_ylabel("bins")
    axes[0, 1].set_title("Hard categorical errors")
    axes[0, 1].legend()

    axes[1, 0].plot(x, bins["hard_decode_raw_nrmse"], label="model hard NRMSE")
    axes[1, 0].plot(
        x,
        bins["teacher_continuous_scale_raw_nrmse"],
        label="GPTQ teacher floor",
        alpha=0.8,
    )
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="zero output")
    axes[1, 0].set_ylabel("NRMSE")
    axes[1, 0].set_title("Hard reconstruction")
    axes[1, 0].legend()

    axes[1, 1].plot(
        x,
        bins["diagnostic_old_behavioral_direction"],
        label="behavioral direction",
    )
    axes[1, 1].plot(
        x,
        bins["diagnostic_old_structural_direction"],
        label="structural direction",
    )
    axes[1, 1].plot(
        x,
        bins["diagnostic_old_direction_contrastive_loss"],
        label="contrastive CE",
    )
    axes[1, 1].set_ylabel("loss (lower is better)")
    axes[1, 1].set_title("Detached directional diagnostics")
    axes[1, 1].legend()

    axes[2, 0].plot(
        x,
        bins["diagnostic_old_direction_contrastive_top1_accuracy"],
        label="top-1",
    )
    axes[2, 0].plot(x, bins["contrastive_chance_top1"], label="batch-exact chance")
    axes[2, 0].plot(x, bins["contrastive_top1_advantage"], label="advantage")
    axes[2, 0].set_ylim(0.0, 1.0)
    axes[2, 0].set_ylabel("fraction")
    axes[2, 0].set_title("Same-layout identity")
    axes[2, 0].legend()

    axes[2, 1].plot(
        x,
        bins["diagnostic_old_latent_interaction_rms_ratio"],
        label="interaction / total RMS",
    )
    axes[2, 1].plot(
        x,
        bins["latent_interaction_rms_estimate"],
        label="interaction RMS estimate",
    )
    twin = axes[2, 1].twinx()
    twin.plot(x, bins["diagnostic_old_latent_rms"], color="C2", label="latent RMS")
    axes[2, 1].set_ylabel("interaction")
    twin.set_ylabel("total latent RMS")
    axes[2, 1].set_title("Latent geometry")
    handles, labels = axes[2, 1].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[2, 1].legend(handles + handles2, labels + labels2)

    axes[3, 0].plot(
        x,
        bins["diagnostic_old_behavioral_operator_actual"],
        label="model operator",
    )
    axes[3, 0].plot(
        x,
        bins["teacher_continuous_scale_operator_metric"],
        label="GPTQ teacher floor",
    )
    axes[3, 0].set_yscale("log")
    axes[3, 0].set_ylabel("operator metric")
    axes[3, 0].set_title("Activation-weighted reconstruction")
    axes[3, 0].legend()

    axes[3, 1].plot(x, bins["grad_norm_pre_clip"], label="pre-clip norm")
    axes[3, 1].set_yscale("log")
    axes[3, 1].set_ylabel("global grad norm")
    axes[3, 1].set_title("Optimization health; clip threshold = 5")
    axes[3, 1].legend()

    _add_phase_lines(axes, breakpoints)
    for axis in axes[-1, :]:
        axis.set_xlabel("optimizer step")
    fig.suptitle(
        f"Parallel categorical GPTQ trajectory through step {int(bins.step_max.max())}\n"
        "points are robust medians in complete 1k-step bins"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_gradients(
    gradient_bins: pd.DataFrame, breakpoints: list[int], output_path: Path
) -> None:
    x = gradient_bins["bin_end"]
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 11.0), sharex=True)
    axes[0, 0].plot(x, gradient_bins["code_gradient_rms"], label="d code / dz")
    axes[0, 0].plot(x, gradient_bins["scale_gradient_rms"], label="d scale / dz")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("activation gradient RMS")
    axes[0, 0].set_title("Latent credit signal")
    axes[0, 0].legend()

    axes[0, 1].plot(
        x, gradient_bins["scale_over_code_gradient_rms"], label="scale / code RMS"
    )
    axes[0, 1].plot(
        x, gradient_bins["code_scale_gradient_cosine"], label="code-scale cosine"
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_ylabel("ratio / cosine")
    axes[0, 1].set_title("Objective interaction at z")
    axes[0, 1].legend()

    for depth in range(1, 6):
        axes[1, 0].plot(
            x, gradient_bins[f"decoder_block_{depth}"], label=f"decoder {depth}"
        )
    axes[1, 0].plot(x, gradient_bins["categorical_tail"], label="categorical tail")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("parameter grad RMS")
    axes[1, 0].set_title("Decoder depth")
    axes[1, 0].legend(ncol=2, fontsize=8)

    for depth in (1, 2, 5, 9, 13):
        axes[1, 1].plot(
            x, gradient_bins[f"encoder_block_{depth}"], label=f"encoder {depth}"
        )
    axes[1, 1].plot(
        x, gradient_bins["distribution_encoder"], label="distribution encoder"
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("parameter grad RMS")
    axes[1, 1].set_title("Encoder depth")
    axes[1, 1].legend(ncol=2, fontsize=8)

    for column, label in (
        ("input_and_queries", "inputs + queries"),
        ("bottleneck_projections", "bottleneck projections"),
        ("code_head", "code head"),
        ("scale_head", "scale head"),
    ):
        axes[2, 0].plot(x, gradient_bins[column], label=label)
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_ylabel("parameter grad RMS")
    axes[2, 0].set_title("Boundary paths")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(
        x,
        gradient_bins["decoder_block_5"] / gradient_bins["decoder_block_1"],
        label="decoder 5 / decoder 1",
    )
    axes[2, 1].plot(
        x,
        gradient_bins["encoder_block_13"] / gradient_bins["encoder_block_1"],
        label="encoder 13 / encoder 1",
    )
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_ylabel("gradient RMS ratio")
    axes[2, 1].set_title("Depth attenuation (not aggregate energy)")
    axes[2, 1].legend()

    _add_phase_lines(axes, breakpoints)
    for axis in axes[-1, :]:
        axis.set_xlabel("optimizer step")
    fig.suptitle("Gradient telemetry; robust medians in complete 1k-step bins")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_before = {
        "metrics": _source_stat(args.metrics),
        "gradients": _source_stat(args.gradients),
        "resolved_config": _source_stat(args.resolved_config),
    }
    metric_rows, metric_read = _read_complete_jsonl(args.metrics)
    gradient_rows, gradient_read = _read_complete_jsonl(args.gradients)
    source_after = {
        "metrics": _source_stat(args.metrics),
        "gradients": _source_stat(args.gradients),
        "resolved_config": _source_stat(args.resolved_config),
    }
    config = json.loads(args.resolved_config.read_text(encoding="utf-8"))

    raw_metrics = pd.DataFrame(metric_rows)
    raw_gradients = _flatten_gradient_rows(gradient_rows)
    duplicate_metric_steps = int(raw_metrics["step"].duplicated().sum())
    duplicate_gradient_steps = int(raw_gradients["step"].duplicated().sum())
    metrics = raw_metrics.sort_values("step").drop_duplicates("step", keep="last")
    gradients = raw_gradients.sort_values("step").drop_duplicates("step", keep="last")
    metrics = _augment_metrics(metrics).reset_index(drop=True)
    gradients = gradients.reset_index(drop=True)

    numeric_metrics = metrics.select_dtypes(include=[np.number])
    finite_metric_columns = [
        column for column in numeric_metrics.columns if column != "local_steps_per_second"
    ]
    numeric_gradients = gradients.select_dtypes(include=[np.number])
    expected_metric_steps = {1, *range(10, int(metrics["step"].max()) + 1, 10)}
    expected_gradient_steps = {
        1,
        *range(100, int(gradients["step"].max()) + 1, 100),
    }
    missing_metric_steps = sorted(expected_metric_steps - set(metrics["step"].astype(int)))
    missing_gradient_steps = sorted(
        expected_gradient_steps - set(gradients["step"].astype(int))
    )
    loss_identity_error = (
        metrics["loss"] - metrics["code_ordinal_loss"] - metrics["scale_ordinal_loss"]
    ).abs()
    cursor_error = (
        metrics["committed_logical_index"] - metrics["step"] * int(config["model_config"]["batch_size"])
    ).abs()
    clip_threshold = float(config["resolved_config"]["grad_clip_norm"])

    metric_bins = _robust_bins(metrics)
    complete_metric_bins = metric_bins[metric_bins["complete"]].copy().reset_index(drop=True)
    gradient_bins = _robust_bins(gradients)
    complete_gradient_bins = gradient_bins[gradient_bins["complete"]].copy().reset_index(drop=True)
    phase_detection = _detect_phase_changes(complete_metric_bins)
    stable_breakpoints = [int(step) for step in phase_detection["stable_breakpoints"]]
    last_complete_step = int(complete_metric_bins["bin_end"].max())
    phase_boundaries = [*stable_breakpoints, last_complete_step]

    window_specs = [
        (1, 500, "initial_1_500"),
        (510, 1_000, "prior_plateau_510_1000"),
        (3_010, 8_000, "directional_escape_3k_8k"),
        (8_010, 17_000, "conditional_breakout_8k_17k"),
        (17_010, 32_000, "reconstruction_growth_17k_32k"),
        (32_010, last_complete_step, "late_plateau_32k_latest_complete"),
        (max(1, last_complete_step - 4_990), last_complete_step, "latest_complete_5k"),
    ]
    pd.DataFrame(_metric_summary_rows(metrics, window_specs)).to_csv(
        args.output_dir / "window_summary.csv", index=False
    )
    pd.DataFrame(_phase_summary_rows(complete_metric_bins, phase_boundaries)).to_csv(
        args.output_dir / "phase_summary.csv", index=False
    )
    pd.DataFrame(_gradient_phase_rows(gradients, phase_boundaries)).to_csv(
        args.output_dir / "gradient_phase_summary.csv", index=False
    )
    metric_bins.to_csv(args.output_dir / "robust_1k_metric_bins.csv", index=False)
    gradient_bins.to_csv(args.output_dir / "robust_1k_gradient_bins.csv", index=False)

    latest_gradient_start = max(1, int(gradients["step"].max()) - 4_900)
    latest_gradients = gradients[gradients["step"] >= latest_gradient_start]
    latest_gradient_rows = []
    for metric in latest_gradients.select_dtypes(include=[np.number]).columns:
        if metric == "step":
            continue
        values = latest_gradients[metric].dropna()
        latest_gradient_rows.append(
            {
                "step_start": int(latest_gradients["step"].min()),
                "step_stop": int(latest_gradients["step"].max()),
                "rows": int(len(latest_gradients)),
                "metric": metric,
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p90": float(values.quantile(0.90)),
            }
        )
    pd.DataFrame(latest_gradient_rows).to_csv(
        args.output_dir / "latest_5k_gradient_summary.csv", index=False
    )

    phase_detection_path = args.output_dir / "phase_changes.json"
    phase_detection_path.write_text(
        json.dumps(phase_detection, indent=2) + "\n", encoding="utf-8"
    )

    _plot_overview(
        complete_metric_bins,
        stable_breakpoints,
        args.output_dir / "trajectory_overview.png",
    )
    _plot_gradients(
        complete_gradient_bins,
        stable_breakpoints,
        args.output_dir / "gradient_trajectory.png",
    )

    latest_local = metrics[metrics["step"] >= int(metrics["step"].max()) - 5_000]
    metadata = {
        "schema": "parallel_categorical_current_state_trajectory_v1",
        "source_before": source_before,
        "source_after": source_after,
        "read": {"metrics": metric_read, "gradients": gradient_read},
        "rows": {"metrics": int(len(metrics)), "gradients": int(len(gradients))},
        "steps": {
            "first_metric": int(metrics["step"].min()),
            "latest_metric": int(metrics["step"].max()),
            "latest_gradient": int(gradients["step"].max()),
            "latest_complete_1k_bin": last_complete_step,
        },
        "validity": {
            "duplicate_metric_steps": duplicate_metric_steps,
            "duplicate_gradient_steps": duplicate_gradient_steps,
            "missing_expected_metric_steps": missing_metric_steps,
            "missing_expected_gradient_steps": missing_gradient_steps,
            "all_logged_metric_numeric_finite": bool(
                np.isfinite(numeric_metrics[finite_metric_columns].to_numpy()).all()
            ),
            "all_defined_local_rates_finite": bool(
                np.isfinite(metrics["local_steps_per_second"].dropna().to_numpy()).all()
            ),
            "all_gradient_numeric_finite": bool(
                np.isfinite(numeric_gradients.to_numpy()).all()
            ),
            "max_loss_identity_abs_error": float(loss_identity_error.max()),
            "max_cursor_abs_error": int(cursor_error.max()),
            "learning_rates": sorted(metrics["learning_rate"].unique().tolist()),
            "metric_schemas": sorted(metrics["schema"].unique().tolist()),
            "gradient_schemas": sorted(gradients["schema"].unique().tolist()),
            "elapsed_monotonic": bool(metrics["elapsed_seconds"].is_monotonic_increasing),
        },
        "clipping": {
            "threshold": clip_threshold,
            "steps_at_or_above_threshold": int(
                (metrics["grad_norm_pre_clip"] >= clip_threshold).sum()
            ),
            "max_pre_clip_norm": float(metrics["grad_norm_pre_clip"].max()),
            "late_5k_p99_pre_clip_norm": float(
                latest_local["grad_norm_pre_clip"].quantile(0.99)
            ),
        },
        "runtime": {
            "latest_cumulative_steps_per_second": float(metrics.iloc[-1]["steps_per_second"]),
            "late_5k_local_steps_per_second_median": float(
                latest_local["local_steps_per_second"].median()
            ),
            "late_5k_tokenizer_seconds_median": float(
                latest_local["gptq_tokenizer_seconds"].median()
            ),
            "cuda_peak_gib": float(metrics["cuda_peak_gib"].max()),
        },
        "phase_detection": phase_detection,
        "notes": [
            "All diagnostic_old_* quantities are detached and do not contribute to backward.",
            "Robust plots use medians in complete 1000-step bins; the live partial tail is excluded.",
            "latent_interaction_rms_estimate is ratio * total latent RMS; the logged ratio alone can fall when the denominator inflates.",
            "Per-parameter gradient RMS values across groups are not aggregate gradient-energy shares.",
        ],
    }
    (args.output_dir / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    for artifact in sorted(args.output_dir.iterdir()):
        print(f"wrote {artifact}")


if __name__ == "__main__":
    main()
