from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAIR_PREFIX = "marginhuber_c0p001_m0_d0p05_toprec1_topex0p1"
PATH_PREFIX = "marginhuber_c0p001"
BASE = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
PAIR_DIR = BASE / "causal_pair_analysis"
MARGIN_DIR = BASE / "block_recon_analysis/margin_mechanism"
EST_DIR = BASE / "estimator_variance"
OUT_DIR = BASE / "arch_invariant_causal_analysis"


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _summarize(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"n": 0.0}
    return {
        "n": float(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _estimator_summary() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    runs = {
        "A_pairs4": EST_DIR / "current_A_pairs4/a_estimator_samples.csv",
        "A_pairs16": EST_DIR / "current_A_pairs16/a_estimator_samples.csv",
        "control_pairs4": EST_DIR / "current_control_pairs4/a_estimator_samples.csv",
    }
    for label, path in runs.items():
        if not path.is_file():
            continue
        frame = _read(path)
        for column in ["precond_a_paired_loss", "precond_a_global_audit_loss", "precond_trace_m_per_dim"]:
            row: dict[str, float | str] = {"estimator_run": label, "metric": column}
            row.update(_summarize(frame[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def _path_component_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    components = _read(PAIR_DIR / f"{PATH_PREFIX}_path_component_paired_deltas.csv")
    joined = _read(PAIR_DIR / f"{PATH_PREFIX}_path_component_joined_downstream.csv")
    metrics = [
        "delta_test_loss_decoded_raw_minus_raw_A_minus_control",
        "delta_test_loss_latent_minus_decoded_raw_A_minus_control",
        "delta_test_loss_latent_minus_raw_A_minus_control",
        "start_reconstruction_rel_l2_A_minus_control",
        "latent_tortuosity_A_minus_control",
        "change_cos_raw_vs_latent_from_dec0_A_minus_control",
    ]
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        row: dict[str, float | str] = {"metric": metric}
        row.update(_summarize(components[metric]))
        for target in ["step0_test_loss_delta", "test_mean_aulc_delta", "post0_test_loss_mean_delta"]:
            row[f"corr_vs_{target}"] = float(joined[metric].corr(joined[target]))
        rows.append(row)
    return pd.DataFrame(rows), joined


def _latent_metric_summary() -> pd.DataFrame:
    summary = _read(PAIR_DIR / "latent_metric_marginhuber_c0p001/latent_metric_summary.csv")
    keep = summary[
        summary["direction_method"].isin(
            ["latent_adam_actual", "raw_decoded_adam_actual", "latent_pullback_natural_grad"]
        )
    ].copy()
    return keep.sort_values(["variant_label", "direction_method", "scale_anchor"])


def _function_tail_summary() -> pd.DataFrame:
    margin = _read(MARGIN_DIR / f"{PAIR_PREFIX}_margin_mechanism_summary.csv")
    return margin


def _downstream_summary() -> pd.DataFrame:
    return _read(PAIR_DIR / f"{PAIR_PREFIX}_downstream_summary.csv")


def _plot_path_decomposition(joined: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = joined["task_name"].map({"mnist": "#4c78a8", "fashion_mnist": "#f58518"}).fillna("#777777")
    axes[0].scatter(
        joined["delta_test_loss_decoded_raw_minus_raw_A_minus_control"],
        joined["step0_test_loss_delta"],
        c=colors,
        s=70,
        alpha=0.85,
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].axvline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Decoded-start component is not A-specific")
    axes[0].set_xlabel("A-control decoded_raw_minus_raw")
    axes[0].set_ylabel("A-control step0 test loss")
    axes[1].scatter(
        joined["delta_test_loss_latent_minus_decoded_raw_A_minus_control"],
        joined["post0_test_loss_mean_delta"],
        c=colors,
        s=70,
        alpha=0.85,
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].axvline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Latent residual is not A-specific")
    axes[1].set_xlabel("A-control latent_minus_decoded_raw")
    axes[1].set_ylabel("A-control post0 test loss mean")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#4c78a8", label="mnist"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#f58518", label="fashion_mnist"),
    ]
    axes[1].legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_estimator(summary: pd.DataFrame, output: Path) -> None:
    source_paths = {
        "A_pairs4": EST_DIR / "current_A_pairs4/a_estimator_samples.csv",
        "A_pairs16": EST_DIR / "current_A_pairs16/a_estimator_samples.csv",
        "control_pairs4": EST_DIR / "current_control_pairs4/a_estimator_samples.csv",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, path in source_paths.items():
        if not path.is_file():
            continue
        frame = _read(path)
        values = pd.to_numeric(frame["precond_a_paired_loss"], errors="coerce").dropna()
        axes[0].hist(values.clip(lower=-5, upper=100), bins=70, alpha=0.50, label=label)
    axes[0].set_title("Sampled A estimator distribution")
    axes[0].set_xlabel("paired A estimator, clipped to [-5, 100] for display")
    axes[0].set_ylabel("count")
    axes[0].legend()
    pivot = summary[summary["metric"] == "precond_a_paired_loss"].set_index("estimator_run")
    for stat, color in [("median", "#4c78a8"), ("p95", "#f58518"), ("p99", "#e45756"), ("max", "#72b7b2")]:
        if stat in pivot:
            axes[1].plot(pivot.index, pivot[stat], marker="o", label=stat, color=color)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_title("Estimator tails")
    axes[1].set_ylabel("paired A estimator")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    downstream = _downstream_summary()
    path_summary, path_joined = _path_component_summary()
    latent = _latent_metric_summary()
    margin = _function_tail_summary()
    estimator = _estimator_summary()

    downstream.to_csv(OUT_DIR / "downstream_summary.csv", index=False)
    path_summary.to_csv(OUT_DIR / "path_component_summary.csv", index=False)
    path_joined.to_csv(OUT_DIR / "path_component_joined_downstream.csv", index=False)
    latent.to_csv(OUT_DIR / "latent_metric_selected_summary.csv", index=False)
    margin.to_csv(OUT_DIR / "function_tail_summary.csv", index=False)
    estimator.to_csv(OUT_DIR / "estimator_variance_summary.csv", index=False)
    _plot_path_decomposition(path_joined, OUT_DIR / "path_decomposition_vs_downstream.png")
    if not estimator.empty:
        _plot_estimator(estimator, OUT_DIR / "estimator_variance_tails.png")
    payload = {
        "outputs": {
            "downstream_summary": str(OUT_DIR / "downstream_summary.csv"),
            "path_component_summary": str(OUT_DIR / "path_component_summary.csv"),
            "path_component_joined_downstream": str(OUT_DIR / "path_component_joined_downstream.csv"),
            "latent_metric_selected_summary": str(OUT_DIR / "latent_metric_selected_summary.csv"),
            "function_tail_summary": str(OUT_DIR / "function_tail_summary.csv"),
            "estimator_variance_summary": str(OUT_DIR / "estimator_variance_summary.csv"),
            "path_plot": str(OUT_DIR / "path_decomposition_vs_downstream.png"),
            "estimator_plot": str(OUT_DIR / "estimator_variance_tails.png"),
        }
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
