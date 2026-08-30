#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finite_frame(frame: pd.DataFrame, columns: list[str]) -> bool:
    existing = [col for col in columns if col in frame.columns]
    if not existing:
        return False
    return bool(np.isfinite(frame[existing].to_numpy(dtype=np.float64)).all())


def _bootstrap_ci(values: np.ndarray, *, seed: int, reps: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        draws[idx] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _summary(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    lo, hi = _bootstrap_ci(values, seed=seed)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "worse_count_delta_gt0": int(np.sum(values > 0.0)),
        "better_count_delta_lt0": int(np.sum(values < 0.0)),
        "bootstrap95_low": lo,
        "bootstrap95_high": hi,
        "min": float(np.min(values)) if values.size else float("nan"),
        "max": float(np.max(values)) if values.size else float("nan"),
    }


def _method_step_summaries(paired_curves: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "test_loss_delta",
        "train_loss_delta",
        "theta_rel_w0_delta",
        "theta_rel_dec0_delta",
        "path_length_rel_w0_delta",
    ]
    seed = 500
    for (method, step), group in paired_curves.groupby(["method", "step"], sort=True):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            rows.append({"method": str(method), "step": int(step), "metric": metric, **_summary(values, seed=seed)})
            seed += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "projected_method_step_summary.csv", index=False)
    step25 = frame[frame["step"].astype(int) == int(paired_curves["step"].max())].copy()
    step25.to_csv(out_dir / "projected_step25_method_summary.csv", index=False)
    return frame


def _method_gap_summaries(paired_curves: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["source_weight_index", "start_bank_position", "task_name", "tau", "step"]
    base = paired_curves[paired_curves["method"].astype(str) == "raw_from_decoded"].copy()
    rows: list[dict[str, Any]] = []
    for method, method_frame in paired_curves.groupby("method", sort=True):
        method_name = str(method)
        if method_name == "raw_from_decoded":
            continue
        merged = method_frame.merge(base, on=keys, suffixes=("_method", "_rawdecoded"), validate="one_to_one")
        for _, row in merged.iterrows():
            gap_control = float(row["test_loss_control_method"] - row["test_loss_control_rawdecoded"])
            gap_a = float(row["test_loss_a_method"] - row["test_loss_a_rawdecoded"])
            rows.append(
                {
                    "method": method_name,
                    **{key: row[key] for key in keys},
                    "test_loss_gap_control": gap_control,
                    "test_loss_gap_a": gap_a,
                    "test_loss_gap_delta": gap_a - gap_control,
                    "method_test_loss_delta": float(row["test_loss_delta_method"]),
                    "raw_from_decoded_test_loss_delta": float(row["test_loss_delta_rawdecoded"]),
                }
            )
    gaps = pd.DataFrame(rows)
    gaps.to_csv(out_dir / "projected_method_gaps.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    seed = 900
    for (method, step), group in gaps.groupby(["method", "step"], sort=True):
        for metric in ["test_loss_gap_delta", "method_test_loss_delta", "raw_from_decoded_test_loss_delta"]:
            values = group[metric].to_numpy(dtype=np.float64)
            summary_rows.append({"method": str(method), "step": int(step), "metric": metric, **_summary(values, seed=seed)})
            seed += 1
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "projected_method_gap_summary.csv", index=False)
    absolute_rows: list[dict[str, Any]] = []
    for (method, step), group in gaps.groupby(["method", "step"], sort=True):
        for column, metric in [
            ("test_loss_gap_control", "control_method_minus_rawdecoded"),
            ("test_loss_gap_a", "A_method_minus_rawdecoded"),
        ]:
            values = group[column].to_numpy(dtype=np.float64)
            absolute_rows.append({"method": str(method), "step": int(step), "metric": metric, **_summary(values, seed=1500 + len(absolute_rows))})
    absolute = pd.DataFrame(absolute_rows)
    absolute.to_csv(out_dir / "projected_absolute_method_gap_summary.csv", index=False)
    return gaps, summary, absolute


def _geometry_summary(geometry: pd.DataFrame, paired_geom: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "normal_residual_fraction",
        "projection_norm_fraction",
        "projection_cos_with_neg_grad",
        "metric_condition",
        "fresh_raw_adam_projection_residual_fraction",
    ]
    for (label, method, step), group in geometry.groupby(["label", "method", "step"], sort=True):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "label": str(label),
                    "method": str(method),
                    "step": int(step),
                    "metric": metric,
                    "n": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
            )
    for (method, step), group in paired_geom.groupby(["method", "step"], sort=True):
        for metric in ["normal_residual_fraction_delta", "projection_norm_fraction_delta", "projection_cos_with_neg_grad_delta"]:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=np.float64)
            rows.append({"label": "A_minus_control", "method": str(method), "step": int(step), "metric": metric, **_summary(values, seed=1200 + len(rows))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "projected_geometry_summary.csv", index=False)
    return frame


def _load_optional_natural(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(file_path)
    frame["source"] = str(file_path)
    return frame


def _chart_info_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return {
            "latent_dim": None,
            "vae_hidden_dim": None,
            "run_name": "",
            "description": "the current frozen decoder chart",
        }
    run_dir = Path(str(runs[0])).expanduser()
    config_path = run_dir / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = payload.get("config", payload)
        latent_dim = cfg.get("latent_dim")
        hidden_dim = cfg.get("vae_hidden_dim")
    except Exception:
        latent_dim = None
        hidden_dim = None
    details = []
    if latent_dim is not None:
        details.append(f"latent_dim={int(latent_dim)}")
    if hidden_dim is not None:
        details.append(f"vae_hidden_dim={int(hidden_dim)}")
    description = "the current frozen decoder chart"
    if details:
        description = f"{description} ({', '.join(details)})"
    return {
        "latent_dim": int(latent_dim) if latent_dim is not None else None,
        "vae_hidden_dim": int(hidden_dim) if hidden_dim is not None else None,
        "run_name": run_dir.name,
        "description": description,
    }


def _plot_outputs(paired_curves: pd.DataFrame, geometry: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        paired_curves.groupby(["method", "step"], as_index=False)["test_loss_delta"]
        .agg(["mean", "median"])
        .reset_index()
    )
    methods = sorted(summary["method"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method in methods:
        sub = summary[summary["method"].astype(str) == method]
        ax.plot(sub["step"], sub["mean"], marker="o", label=method)
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("A - control test loss")
    ax.set_title("Projected Discriminator: Test-Loss Delta By Method")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "projected_test_delta_by_step.png", dpi=180)
    plt.close(fig)

    step_max = int(paired_curves["step"].max())
    step25 = paired_curves[paired_curves["step"].astype(int) == step_max].copy()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(methods))
    for idx, method in enumerate(methods):
        vals = step25[step25["method"].astype(str) == method]["test_loss_delta"].to_numpy(dtype=np.float64)
        jitter = np.linspace(-0.18, 0.18, max(len(vals), 1))
        ax.scatter(np.full_like(vals, positions[idx], dtype=np.float64) + jitter[: len(vals)], vals, alpha=0.8)
        ax.plot([positions[idx] - 0.25, positions[idx] + 0.25], [np.mean(vals), np.mean(vals)], color="black", lw=2)
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("A - control test loss")
    ax.set_title(f"Projected Discriminator: Step {step_max} Paired Deltas")
    fig.savefig(out_dir / "projected_step25_test_delta_scatter.png", dpi=180)
    plt.close(fig)

    geom = (
        geometry.groupby(["label", "method", "step"], as_index=False)["normal_residual_fraction"]
        .median()
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for (label, method), sub in geom.groupby(["label", "method"], sort=True):
        ax.plot(sub["step"], sub["normal_residual_fraction"], marker="o", label=f"{label}:{method}")
    ax.set_xlabel("step")
    ax.set_ylabel("median normal residual")
    ax.set_title("Normal Residual By Chart Method")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "projected_normal_residual_by_method.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    curves = pd.read_csv(out_dir / "trajectory_curves.csv")
    geometry = pd.read_csv(out_dir / "trajectory_geometry.csv")
    cg = pd.read_csv(out_dir / "trajectory_cg.csv")
    paired_curves = pd.read_csv(out_dir / "trajectory_paired_curve_deltas.csv")
    paired_geom = pd.read_csv(out_dir / "trajectory_paired_geometry_deltas.csv")
    paired_cg = pd.read_csv(out_dir / "trajectory_paired_cg_deltas.csv")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    chart_info = _chart_info_from_manifest(manifest)

    method_summary = _method_step_summaries(paired_curves, out_dir)
    gaps, gap_summary, absolute_gap_summary = _method_gap_summaries(paired_curves, out_dir)
    geometry_summary = _geometry_summary(geometry, paired_geom, out_dir)
    natural = _load_optional_natural(args.natural_summary_csv)
    if not natural.empty:
        natural.to_csv(out_dir / "projected_existing_natural_step25_summary.csv", index=False)
    _plot_outputs(paired_curves, geometry, out_dir)

    labels = sorted(curves["label"].astype(str).unique().tolist())
    methods = sorted(curves["method"].astype(str).unique().tolist())
    steps = sorted(int(v) for v in curves["step"].unique().tolist())
    sources = sorted(int(v) for v in curves["source_weight_index"].unique().tolist())
    raw = paired_curves[paired_curves["method"].astype(str) == "raw"].copy()
    projected_final = curves[
        (curves["method"].astype(str) == "projected_raw_adam_tangent")
        & (curves["step"].astype(int) == max(steps))
    ].copy()
    validation = {
        "labels": labels,
        "methods": methods,
        "steps": steps,
        "source_indices": sources,
        "curve_rows": int(len(curves)),
        "paired_curve_rows": int(len(paired_curves)),
        "geometry_rows": int(len(geometry)),
        "paired_geometry_rows": int(len(paired_geom)),
        "cg_rows": int(len(cg)),
        "paired_cg_rows": int(len(paired_cg)),
        "expected_curve_rows": int(len(labels) * len(methods) * len(sources) * len(steps)),
        "expected_paired_curve_rows": int(len(methods) * len(sources) * len(steps)),
        "same_curve_batch_hash_all": bool(
            (
                paired_curves["batch_indices_sha256_control"].astype(str)
                == paired_curves["batch_indices_sha256_a"].astype(str)
            ).all()
        ),
        "same_cg_batch_hash_all": bool(
            (paired_cg["batch_indices_sha256_control"].astype(str) == paired_cg["batch_indices_sha256_a"].astype(str)).all()
        ),
        "same_probe_hash_all": bool(paired_cg["same_probe_sha256"].astype(bool).all()) if "same_probe_sha256" in paired_cg else False,
        "raw_test_loss_delta_max_abs": float(raw["test_loss_delta"].abs().max()),
        "raw_train_loss_delta_max_abs": float(raw["train_loss_delta"].abs().max()),
        "raw_theta_rel_w0_delta_max_abs": float(raw["theta_rel_w0_delta"].abs().max()),
        "raw_path_length_rel_w0_delta_max_abs": float(raw["path_length_rel_w0_delta"].abs().max()),
        "curves_core_finite": _finite_frame(curves, ["train_loss", "test_loss", "theta_rel_w0", "path_length_rel_w0"]),
        "geometry_core_finite": _finite_frame(geometry, ["normal_residual_fraction", "metric_condition", "projection_norm_fraction"]),
        "cg_core_finite": _finite_frame(cg, ["c3_forward", "c3_inverse", "c3_total", "cg_final_rel_residual"]),
        "cg_residual_p90": float(cg["cg_final_rel_residual"].quantile(0.90)),
        "cg_residual_max": float(cg["cg_final_rel_residual"].max()),
        "projected_final_path_length_rel_w0_min": float(projected_final["path_length_rel_w0"].min()),
        "projected_final_path_length_rel_w0_median": float(projected_final["path_length_rel_w0"].median()),
        "projected_final_last_linear_step_norm_median": float(projected_final["last_projected_linear_step_norm"].median()),
        "projected_projection_residual_step_positive_median": float(
            curves[
                (curves["method"].astype(str) == "projected_raw_adam_tangent")
                & (curves["step"].astype(int) > 0)
            ]["last_projected_raw_adam_projection_residual_fraction"].median()
        ),
        "chart_latent_dim": chart_info["latent_dim"],
        "chart_vae_hidden_dim": chart_info["vae_hidden_dim"],
        "chart_reference_run": chart_info["run_name"],
        "manifest": manifest,
    }
    validation["accepted"] = bool(
        validation["curve_rows"] == validation["expected_curve_rows"]
        and validation["paired_curve_rows"] == validation["expected_paired_curve_rows"]
        and validation["same_curve_batch_hash_all"]
        and validation["same_cg_batch_hash_all"]
        and validation["same_probe_hash_all"]
        and validation["raw_test_loss_delta_max_abs"] == 0.0
        and validation["raw_train_loss_delta_max_abs"] == 0.0
        and validation["raw_theta_rel_w0_delta_max_abs"] == 0.0
        and validation["raw_path_length_rel_w0_delta_max_abs"] == 0.0
        and validation["curves_core_finite"]
        and validation["geometry_core_finite"]
        and validation["cg_core_finite"]
        and validation["cg_residual_p90"] < 1.0e-4
        and validation["cg_residual_max"] < 3.0e-4
        and validation["projected_final_path_length_rel_w0_min"] > 0.0
    )
    (out_dir / "projected_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    final_step = max(steps)
    final_summary = method_summary[
        (method_summary["step"].astype(int) == final_step)
        & (method_summary["metric"].astype(str) == "test_loss_delta")
    ].copy()
    final_gap = gap_summary[
        (gap_summary["step"].astype(int) == final_step)
        & (gap_summary["metric"].astype(str) == "test_loss_gap_delta")
    ].copy()
    final_absolute_gap = absolute_gap_summary[
        (absolute_gap_summary["step"].astype(int) == final_step)
        & (absolute_gap_summary["metric"].astype(str).isin(["control_method_minus_rawdecoded", "A_method_minus_rawdecoded"]))
    ].copy()
    geom_final = geometry_summary[
        (geometry_summary["step"].astype(int) == final_step)
        & (geometry_summary["metric"].astype(str).isin(["normal_residual_fraction", "normal_residual_fraction_delta"]))
    ].copy()

    review_lines = [
        "# Projected Raw-Adam Trajectory Discriminator Review",
        "",
        "## Validation",
        "",
        f"- Accepted: `{validation['accepted']}`.",
        f"- Rows: curves `{validation['curve_rows']}` / expected `{validation['expected_curve_rows']}`, paired curves `{validation['paired_curve_rows']}` / expected `{validation['expected_paired_curve_rows']}`, geometry `{validation['geometry_rows']}`, CG `{validation['cg_rows']}`.",
        f"- Same batch hashes across labels: curves `{validation['same_curve_batch_hash_all']}`, CG `{validation['same_cg_batch_hash_all']}`; same probe hashes `{validation['same_probe_hash_all']}`.",
        f"- Raw no-op max abs deltas: test `{validation['raw_test_loss_delta_max_abs']:.6g}`, train `{validation['raw_train_loss_delta_max_abs']:.6g}`, theta_rel_w0 `{validation['raw_theta_rel_w0_delta_max_abs']:.6g}`, path `{validation['raw_path_length_rel_w0_delta_max_abs']:.6g}`.",
        f"- CG residual p90 `{validation['cg_residual_p90']:.6g}`, max `{validation['cg_residual_max']:.6g}`.",
        f"- Projected path nonzero: final min path_rel_w0 `{validation['projected_final_path_length_rel_w0_min']:.6g}`, median `{validation['projected_final_path_length_rel_w0_median']:.6g}`.",
        f"- Projected raw Adam proposal residual median at positive checkpoints `{validation['projected_projection_residual_step_positive_median']:.6g}`.",
        "",
        "## Step 25 A-Control Test-Loss Deltas",
        "",
    ]
    for _, row in final_summary.sort_values("method").iterrows():
        review_lines.append(
            f"- `{row['method']}`: mean `{float(row['mean']):+.6g}`, median `{float(row['median']):+.6g}`, "
            f"CI `[{float(row['bootstrap95_low']):+.6g}, {float(row['bootstrap95_high']):+.6g}]`, "
            f"worse `{int(row['worse_count_delta_gt0'])}/{int(row['n'])}`."
        )
    review_lines.extend(["", "## Step 25 Method Gap Versus raw_from_decoded", ""])
    for _, row in final_gap.sort_values("method").iterrows():
        review_lines.append(
            f"- `{row['method']} - raw_from_decoded`, A-control delta of gap: mean `{float(row['mean']):+.6g}`, "
            f"median `{float(row['median']):+.6g}`, CI `[{float(row['bootstrap95_low']):+.6g}, {float(row['bootstrap95_high']):+.6g}]`, "
            f"worse `{int(row['worse_count_delta_gt0'])}/{int(row['n'])}`."
        )
    review_lines.extend(["", "## Step 25 Absolute Method Gaps", ""])
    for _, row in final_absolute_gap.sort_values(["method", "metric"]).iterrows():
        review_lines.append(
            f"- `{row['method']}:{row['metric']}` mean `{float(row['mean']):+.6g}`, "
            f"median `{float(row['median']):+.6g}`, CI `[{float(row['bootstrap95_low']):+.6g}, {float(row['bootstrap95_high']):+.6g}]`, "
            f"gap>0 `{int(row['worse_count_delta_gt0'])}/{int(row['n'])}`."
        )
    review_lines.extend(["", "## Geometry At Step 25", ""])
    for _, row in geom_final.sort_values(["label", "method", "metric"]).iterrows():
        if str(row["metric"]).endswith("_delta"):
            review_lines.append(
                f"- `{row['label']}:{row['method']}:{row['metric']}` mean `{float(row['mean']):+.6g}`, median `{float(row['median']):+.6g}`."
            )
        else:
            review_lines.append(
                f"- `{row['label']}:{row['method']}:{row['metric']}` mean `{float(row['mean']):.6g}`, median `{float(row['median']):.6g}`."
            )
    review_lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            f"This audit tests {chart_info['description']} only. `projected_raw_adam_tangent` is a stateful theta-space Adam proposal projected into the decoder tangent with no line search; it is not a PSGD/Kron baseline.",
            "",
            "If projected tangent updates remain worse than or fail to outperform `raw_from_decoded` while normal residuals stay high, this supports CH1/rank-tangent bottleneck over the residual CH2 claim that latent Adam alone hides an A win.",
        ]
    )
    (out_dir / "projected_trajectory_review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the projected raw-Adam tangent trajectory discriminator.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--natural-summary-csv", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
