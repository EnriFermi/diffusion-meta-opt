#!/usr/bin/env python
"""Plot additive VAE training-loss components from existing metrics only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
CONTROL_RUN = (
    "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_"
    "marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0"
)
A_RUN = (
    "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_clip20_"
    "fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_"
    "clean_harness_v1_seed0"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "training_loss_components_h2048_m512"
)

COMPONENT_COLORS = {
    "reconstruction": "#2563eb",
    "weighted KL": "#7c3aed",
    "fc2 reconstruction": "#059669",
    "function anchor": "#d97706",
    "A/preconditioner": "#dc2626",
    "geometry": "#0891b2",
    "row direction": "#6b7280",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(np.float64)


def _load_history(run_dir: Path, *, label: str) -> tuple[pd.DataFrame, dict[str, object]]:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "vae_metrics.csv"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = config_payload.get("config", config_payload)
    if int(config["vae_hidden_dim"]) != 2048 or int(config["latent_dim"]) != 512:
        raise ValueError(
            f"{label}: expected hidden_dim=2048, latent_dim=512; "
            f"got {config['vae_hidden_dim']}, {config['latent_dim']}"
        )
    if str(config["vae_loss_kind"]).lower() not in {"mse", "normalized_mse"}:
        raise ValueError(f"{label}: additive reconstruction decomposition expects MSE VAE loss")

    frame = pd.read_csv(metrics_path, low_memory=False)
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame = frame[frame["step"].notna()].copy().sort_values("step").reset_index(drop=True)
    beta_kl = float(config["beta_kl"])

    for split in ("train", "val"):
        frame[f"{split}_component_reconstruction"] = _numeric(frame, f"{split}_recon_mse")
        frame[f"{split}_component_weighted_kl"] = beta_kl * _numeric(frame, f"{split}_kl")
        frame[f"{split}_component_block_recon"] = _numeric(frame, f"{split}_block_recon_effective_loss")
        frame[f"{split}_component_anchor"] = _numeric(frame, f"{split}_function_anchor_effective_loss")
        frame[f"{split}_component_preconditioner"] = _numeric(frame, f"{split}_precond_effective_loss")
        frame[f"{split}_component_geometry"] = _numeric(frame, f"{split}_geometry_reg") * _numeric(
            frame, f"{split}_geometry_reg_coeff"
        )
        frame[f"{split}_component_direction"] = _numeric(frame, f"{split}_block_direction_effective_loss")
        component_columns = [column for column in frame if column.startswith(f"{split}_component_")]
        frame[f"{split}_recomposed_loss"] = frame[component_columns].sum(axis=1)
        logged = pd.to_numeric(frame.get(f"{split}_loss"), errors="coerce")
        frame[f"{split}_loss_residual"] = logged - frame[f"{split}_recomposed_loss"]

    metadata = {
        "label": label,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "config_sha256": _sha256(config_path),
        "metrics_sha256": _sha256(metrics_path),
        "history_rows": int(len(frame)),
        "step_min": float(frame["step"].min()),
        "step_max": float(frame["step"].max()),
        "beta_kl": beta_kl,
        "vae_hidden_dim": int(config["vae_hidden_dim"]),
        "latent_dim": int(config["latent_dim"]),
        "max_abs_train_recomposition_residual": float(frame["train_loss_residual"].abs().max()),
        "max_abs_val_recomposition_residual": float(frame["val_loss_residual"].dropna().abs().max()),
    }
    frame.insert(0, "run_label_short", label)
    return frame, metadata


def _plot_total(axis: plt.Axes, control: pd.DataFrame, variant_a: pd.DataFrame) -> None:
    axis.plot(variant_a["step"], variant_a["train_loss"], color="#ea580c", linewidth=1.0, alpha=0.8, label="A train")
    axis.plot(
        control["step"],
        control["train_loss"],
        color="#1d4ed8",
        linewidth=1.3,
        marker="o",
        markersize=3,
        label="control train (sparse log)",
    )
    for frame, color, label in ((variant_a, "#fb923c", "A validation"), (control, "#60a5fa", "control validation")):
        valid = frame["val_loss"].notna()
        axis.plot(frame.loc[valid, "step"], frame.loc[valid, "val_loss"], color=color, linestyle="--", marker=".", label=label)
    axis.set_title("Logged total VAE loss")
    axis.set_ylabel("loss")
    axis.legend(fontsize=8, frameon=False, ncol=2)


def _plot_components(axis: plt.Axes, frame: pd.DataFrame, *, title: str, sparse: bool) -> None:
    columns = {
        "reconstruction": "train_component_reconstruction",
        "weighted KL": "train_component_weighted_kl",
        "fc2 reconstruction": "train_component_block_recon",
        "function anchor": "train_component_anchor",
        "A/preconditioner": "train_component_preconditioner",
        "geometry": "train_component_geometry",
        "row direction": "train_component_direction",
    }
    for label, column in columns.items():
        values = frame[column]
        if float(values.abs().max()) == 0.0:
            continue
        axis.plot(
            frame["step"],
            values,
            color=COMPONENT_COLORS[label],
            linewidth=1.1,
            marker="o" if sparse else None,
            markersize=3 if sparse else None,
            alpha=0.9,
            label=label,
        )
    axis.set_yscale("symlog", linthresh=1.0e-6, linscale=0.8)
    axis.set_title(title)
    axis.set_ylabel("effective additive contribution")
    axis.legend(fontsize=8, frameon=False, ncol=2)


def _plot_preconditioner(axis: plt.Axes, variant_a: pd.DataFrame) -> None:
    raw = _numeric(variant_a, "train_precond_a_loss")
    effective = variant_a["train_component_preconditioner"]
    axis.plot(variant_a["step"], raw, color="#9f1239", linewidth=1.0, label="raw sampled A surrogate")
    axis.plot(variant_a["step"], effective, color="#dc2626", linewidth=1.2, label="effective A contribution")
    axis.set_yscale("symlog", linthresh=1.0e-4, linscale=0.8)
    axis.set_title("A surrogate: raw estimator versus effective contribution")
    axis.set_ylabel("loss")
    axis.legend(fontsize=8, frameon=False)
    scale_axis = axis.twinx()
    scale_axis.plot(
        variant_a["step"],
        _numeric(variant_a, "train_precond_grad_scale"),
        color="#111827",
        linewidth=0.9,
        alpha=0.55,
        label="gradient scale",
    )
    scale_axis.set_ylabel("preconditioner gradient scale")
    scale_axis.set_ylim(-0.03, 1.03)


def _plot_residual(axis: plt.Axes, control: pd.DataFrame, variant_a: pd.DataFrame) -> None:
    axis.plot(variant_a["step"], variant_a["train_loss_residual"], color="#ea580c", linewidth=1.0, label="A train")
    axis.plot(
        control["step"],
        control["train_loss_residual"],
        color="#1d4ed8",
        marker="o",
        markersize=3,
        linewidth=1.0,
        label="control train",
    )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_yscale("symlog", linthresh=1.0e-10)
    axis.set_title("Logged total minus recomposed components")
    axis.set_ylabel("residual")
    axis.legend(fontsize=8, frameon=False)


def _make_figure(control: pd.DataFrame, variant_a: pd.DataFrame, path: Path) -> None:
    panels = (
        ("Logged total train loss", "train_loss"),
        ("Reconstruction MSE", "train_component_reconstruction"),
        ("Weighted KL: beta_kl * KL", "train_component_weighted_kl"),
        ("Effective fc2 reconstruction", "train_component_block_recon"),
        ("Effective function anchor", "train_component_anchor"),
        ("Effective A / preconditioner term", "train_component_preconditioner"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 11.2), sharex=True)
    for panel_index, (axis, (title, column)) in enumerate(zip(axes.flat, panels, strict=True)):
        axis.plot(
            variant_a["step"],
            variant_a[column],
            color="#ea580c",
            linewidth=1.0,
            alpha=0.85,
            label="A_clip20 (logged every 10 steps)",
        )
        axis.plot(
            control["step"],
            control[column],
            color="#1d4ed8",
            linewidth=1.25,
            marker="o",
            markersize=3.5,
            label="control (21 logged checkpoints)",
        )
        axis.set_title(title)
        axis.set_ylabel("loss contribution")
        axis.set_xlabel("VAE training step")
        axis.grid(alpha=0.22)
        if panel_index == 0:
            axis.legend(fontsize=8, frameon=False)
        if column not in {"train_loss", "train_component_reconstruction"}:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.suptitle(
        "VAE train-loss components: hidden_dim=2048, latent_dim=512\n"
        "Existing metrics only; each panel has its own linear y-scale",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_a_surrogate_figure(
    variant_a: pd.DataFrame,
    *,
    plot_path: Path,
    summary_csv_path: Path,
) -> dict[str, float | int]:
    sample_count = pd.to_numeric(variant_a["train_precond_sample_count"], errors="coerce").fillna(0.0)
    points = variant_a.loc[sample_count > 0.0, ["step", "train_precond_a_loss", "train_precond_a_train_loss"]].copy()
    points = points.rename(
        columns={
            "train_precond_a_loss": "raw_a_surrogate",
            "train_precond_a_train_loss": "clipped_a_surrogate",
        }
    )
    points["raw_a_surrogate"] = pd.to_numeric(points["raw_a_surrogate"], errors="raise")
    points["clipped_a_surrogate"] = pd.to_numeric(points["clipped_a_surrogate"], errors="raise")
    points = points.loc[points["raw_a_surrogate"] > 0.0].reset_index(drop=True)
    points["rolling_raw_median_101"] = points["raw_a_surrogate"].rolling(101, center=True, min_periods=51).median()
    points["rolling_clipped_mean_101"] = (
        points["clipped_a_surrogate"].rolling(101, center=True, min_periods=51).mean()
    )
    points["bin_end"] = (((points["step"] - 1.0) // 500.0) + 1.0) * 500.0

    grouped = points.groupby("bin_end", sort=True)
    binned = grouped.agg(
        sample_count=("raw_a_surrogate", "size"),
        raw_mean=("raw_a_surrogate", "mean"),
        raw_median=("raw_a_surrogate", "median"),
        clipped_mean=("clipped_a_surrogate", "mean"),
        clipped_median=("clipped_a_surrogate", "median"),
    ).reset_index()
    binned["raw_p90"] = grouped["raw_a_surrogate"].quantile(0.90).to_numpy()
    binned["raw_max"] = grouped["raw_a_surrogate"].max().to_numpy()
    binned.to_csv(summary_csv_path, index=False)

    rank_correlation = float(points["step"].rank().corr(points["raw_a_surrogate"].rank()))
    first = binned.iloc[0]
    last = binned.iloc[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), constrained_layout=True)
    axes[0].scatter(
        points["step"],
        points["raw_a_surrogate"],
        s=12,
        color="#94a3b8",
        alpha=0.42,
        edgecolors="none",
        label="raw sampled A surrogate",
    )
    axes[0].plot(
        points["step"],
        points["rolling_raw_median_101"],
        color="#dc2626",
        linewidth=2.4,
        label="rolling median (101 A updates)",
    )
    axes[0].plot(
        points["step"],
        points["rolling_clipped_mean_101"],
        color="#ea580c",
        linewidth=2.0,
        label="rolling mean after clip20",
    )
    axes[0].axhline(1.0, color="#111827", linestyle="--", linewidth=1.1, label="A = 1 reference")
    axes[0].set_yscale("log")
    axes[0].set_title("All sampled A values")
    axes[0].set_xlabel("VAE fine-tuning step")
    axes[0].set_ylabel("A surrogate")
    axes[0].grid(alpha=0.22, which="both")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        binned["bin_end"],
        binned["raw_median"],
        color="#dc2626",
        linewidth=2.4,
        marker="o",
        label="median",
    )
    axes[1].plot(
        binned["bin_end"],
        binned["clipped_mean"],
        color="#ea580c",
        linewidth=2.0,
        marker="o",
        label="mean after clip20",
    )
    axes[1].plot(
        binned["bin_end"],
        binned["raw_p90"],
        color="#64748b",
        linewidth=1.6,
        marker="o",
        label="raw p90",
    )
    axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=1.1)
    axes[1].set_yscale("log")
    axes[1].set_title("500-step summaries")
    axes[1].set_xlabel("end of VAE fine-tuning bin")
    axes[1].set_ylabel("A surrogate")
    axes[1].grid(alpha=0.22, which="both")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(
        0.03,
        0.04,
        f"median: {first['raw_median']:.4f} -> {last['raw_median']:.4f}\n"
        f"Spearman rho: {rank_correlation:+.3f}",
        transform=axes[1].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    fig.suptitle(
        "A-surrogate during VAE fine-tuning (A_clip20)\n"
        "Existing metrics only; raw estimator and clipped training scalar",
        fontsize=14,
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "sample_count": int(len(points)),
        "first_500_raw_median": float(first["raw_median"]),
        "last_500_raw_median": float(last["raw_median"]),
        "first_500_clipped_mean": float(first["clipped_mean"]),
        "last_500_clipped_mean": float(last["clipped_mean"]),
        "raw_step_spearman": rank_correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", default=CONTROL_RUN)
    parser.add_argument("--a-run", default=A_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    control_dir = ARTIFACT_ROOT / args.control_run
    a_dir = ARTIFACT_ROOT / args.a_run
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[plot_vae_loss_components] mode=read_existing_metrics_only "
        f"control={control_dir} a={a_dir} output={output_dir}",
        flush=True,
    )

    control, control_meta = _load_history(control_dir, label="control")
    variant_a, a_meta = _load_history(a_dir, label="A_clip20")
    combined = pd.concat([control, variant_a], ignore_index=True, sort=False)
    csv_path = output_dir / "vae_loss_component_decomposition.csv"
    plot_path = output_dir / "vae_train_loss_components.png"
    a_surrogate_plot_path = output_dir / "a_surrogate_finetune_trend.png"
    a_surrogate_summary_csv_path = output_dir / "a_surrogate_500step_summary.csv"
    manifest_path = output_dir / "manifest.json"
    combined.to_csv(csv_path, index=False)
    _make_figure(control, variant_a, plot_path)
    a_surrogate_summary = _make_a_surrogate_figure(
        variant_a,
        plot_path=a_surrogate_plot_path,
        summary_csv_path=a_surrogate_summary_csv_path,
    )
    manifest = {
        "mode": "read_existing_metrics_only",
        "training_executed": False,
        "control": control_meta,
        "variant_a": a_meta,
        "outputs": {
            "decomposition_csv": str(csv_path),
            "decomposition_csv_sha256": _sha256(csv_path),
            "plot": str(plot_path),
            "plot_sha256": _sha256(plot_path),
            "a_surrogate_plot": str(a_surrogate_plot_path),
            "a_surrogate_plot_sha256": _sha256(a_surrogate_plot_path),
            "a_surrogate_500step_summary_csv": str(a_surrogate_summary_csv_path),
            "a_surrogate_500step_summary_csv_sha256": _sha256(a_surrogate_summary_csv_path),
        },
        "a_surrogate_summary": a_surrogate_summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[plot_vae_loss_components] done "
        f"control_rows={len(control)} a_rows={len(variant_a)} "
        f"control_max_residual={control_meta['max_abs_train_recomposition_residual']:.3e} "
        f"a_max_residual={a_meta['max_abs_train_recomposition_residual']:.3e} "
        f"plot={plot_path} a_surrogate_plot={a_surrogate_plot_path} "
        f"csv={csv_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
