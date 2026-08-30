from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
OUT_DIR = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_analysis/lam0p03"

RUNS = {
    "clean_control": "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
    "clean_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
    "headce_control": "sage_cnn_vae_smoothing_celo_meta_control_headce_fc2_c0p003_clean_harness_v1_seed0",
    "headce_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_headce_fc2_c0p003_clean_harness_v1_seed0",
    "block_control": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0",
    "block_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p03_clean_harness_v1_seed0",
}

KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]


def _run_dir(label: str) -> Path:
    return ARTIFACT_ROOT / RUNS[label]


def _require_complete(label: str) -> None:
    run_dir = _run_dir(label)
    required = [
        "downstream_results.csv",
        "downstream_curves.csv",
        "selected_lrs.csv",
        "preconditioning_diagnostics.csv",
        "vae_metrics.csv",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{label} missing {missing}")


def _selected_lr(label: str, method: str) -> float:
    rows = pd.read_csv(_run_dir(label) / "selected_lrs.csv")
    hit = rows[(rows["method"].astype(str) == method) & (pd.to_numeric(rows["selected"], errors="coerce") == 1)]
    if len(hit) != 1:
        return float("nan")
    return float(hit["candidate_lr"].iloc[0])


def _quality(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    quality = rows[record_type == "vae_quality"].copy()
    return quality


def _train_history(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    train = rows[record_type != "vae_quality"].copy()
    train["step"] = pd.to_numeric(train["step"], errors="coerce")
    return train[train["step"].notna()].copy()


def _summary_row(label: str) -> dict[str, float | str]:
    downstream = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    dec = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "decoder_latent")]
    raw = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "raw")]
    curves = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    step0 = curves[
        (curves["split"].astype(str) == "eval")
        & (curves["method"].astype(str) == "decoder_latent")
        & (pd.to_numeric(curves["step"], errors="coerce") == 0)
    ]
    quality = _quality(label)
    train = _train_history(label)
    diag = pd.read_csv(_run_dir(label) / "preconditioning_diagnostics.csv")
    row: dict[str, float | str] = {
        "label": label,
        "run_label": RUNS[label],
        "selected_decoder_lr": _selected_lr(label, "decoder_latent"),
        "selected_raw_lr": _selected_lr(label, "raw"),
        "decoder_aulc_mean": float(dec["aulc"].mean()),
        "decoder_aulc_median": float(dec["aulc"].median()),
        "raw_aulc_mean": float(raw["aulc"].mean()),
        "raw_aulc_median": float(raw["aulc"].median()),
        "decoder_step0_test_loss_mean": float(step0["test_loss"].mean()),
        "decoder_step0_test_loss_median": float(step0["test_loss"].median()),
        "reconstruction_rel_l2_median": float(pd.to_numeric(quality["reconstruction_rel_l2"], errors="coerce").median()),
        "decoded_test_loss_median": float(pd.to_numeric(quality["decoded_test_loss"], errors="coerce").median()),
        "decoded_test_acc_median": float(pd.to_numeric(quality["decoded_test_acc"], errors="coerce").median()),
        "val_recon_mse_final": float(pd.to_numeric(train["val_recon_mse"], errors="coerce").dropna().iloc[-1]),
        "li_A_full_per_dim_p95": float(pd.to_numeric(diag["li_A_full_per_dim"], errors="coerce").quantile(0.95)),
        "hvp_probe_trace_m_per_dim_median": float(pd.to_numeric(diag["hvp_probe_trace_m_per_dim"], errors="coerce").median()),
    }
    for col in [
        "train_block_recon_effective_loss",
        "train_block_recon_to_full_mse",
        "train_block_recon_rel_l2",
        "train_block_recon_grad_ratio",
        "train_precond_effective_loss",
        "train_precond_grad_scale",
    ]:
        if col in train:
            values = pd.to_numeric(train[col], errors="coerce").dropna()
            if col == "train_block_recon_grad_ratio":
                values = values[values > 0.0]
            row[f"{col}_median"] = float(values.median()) if not values.empty else float("nan")
            row[f"{col}_final"] = float(values.iloc[-1]) if not values.empty else float("nan")
    return row


def _load_eval_curves(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == "decoder_latent")].copy()
    rows["label"] = label
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    return rows


def _paired_step0_delta(control_label: str, a_label: str, tag: str) -> pd.DataFrame:
    control = _load_eval_curves(control_label)
    variant = _load_eval_curves(a_label)
    control = control[control["step"] == 0][KEY_COLS + ["test_loss", "test_acc"]].copy()
    variant = variant[variant["step"] == 0][KEY_COLS + ["test_loss", "test_acc"]].copy()
    merged = control.merge(variant, on=KEY_COLS, suffixes=("_control", "_a"))
    merged["tag"] = tag
    merged["a_minus_control_step0_test_loss"] = merged["test_loss_a"] - merged["test_loss_control"]
    merged["a_minus_control_step0_test_acc"] = merged["test_acc_a"] - merged["test_acc_control"]
    return merged


def _plot_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    order = ["clean_control", "clean_a", "headce_control", "headce_a", "block_control", "block_a"]
    colors = ["#5068a9", "#8b4d9c", "#9b5b3e", "#b34b4b", "#3d7f63", "#2b8a8a"]
    for ax, col, title in [
        (axes[0], "decoder_aulc_mean", "Decoder eval AULC mean"),
        (axes[1], "reconstruction_rel_l2_median", "VAE quality rel L2 median"),
        (axes[2], "li_A_full_per_dim_p95", "li_A full per dim p95"),
    ]:
        vals = [float(summary.loc[summary["label"] == label, col].iloc[0]) for label in order if label in set(summary["label"])]
        labels = [label.replace("_", "\n") for label in order if label in set(summary["label"])]
        ax.bar(labels, vals, color=colors[: len(vals)])
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_bars.png", dpi=180)
    plt.close(fig)


def _plot_eval_curves() -> None:
    labels = ["clean_control", "clean_a", "headce_control", "headce_a", "block_control", "block_a"]
    frames = []
    for label in labels:
        if _run_dir(label).exists():
            frames.append(_load_eval_curves(label))
    curves = pd.concat(frames, ignore_index=True)
    agg = curves.groupby(["label", "step"], as_index=False)["test_loss"].agg(["mean", "median"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in labels:
        sub = agg[agg["label"] == label].sort_values("step")
        if sub.empty:
            continue
        ax.plot(sub["step"], sub["mean"], marker="o", linewidth=1.6, label=label)
    ax.set_xlabel("Downstream step")
    ax.set_ylabel("Eval test loss, mean over starts")
    ax.set_title("Decoder-latent downstream curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decoder_eval_curves_mean.png", dpi=180)
    plt.close(fig)


def _plot_step0_deltas() -> None:
    clean = _paired_step0_delta("clean_control", "clean_a", "clean")
    block = _paired_step0_delta("block_control", "block_a", "block_lam0p03")
    rows = pd.concat([clean, block], ignore_index=True)
    rows.to_csv(OUT_DIR / "paired_step0_deltas.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [rows[rows["tag"] == tag]["a_minus_control_step0_test_loss"].to_numpy() for tag in ["clean", "block_lam0p03"]]
    ax.boxplot(data, labels=["clean", "block_lam0p03"], showmeans=True)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("A - control step0 test loss")
    ax.set_title("Step0 decoded-start gap")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "step0_gap_boxplot.png", dpi=180)
    plt.close(fig)


def _plot_block_training() -> None:
    frames = []
    for label in ["block_control", "block_a"]:
        train = _train_history(label)
        train["label"] = label
        frames.append(train)
    rows = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    cols = [
        ("train_recon_mse", "Train recon MSE"),
        ("val_recon_mse", "Val recon MSE"),
        ("train_block_recon_effective_loss", "Effective block penalty"),
        ("train_block_recon_grad_ratio", "Block grad / base grad"),
    ]
    for ax, (col, title) in zip(axes.ravel(), cols, strict=True):
        if col not in rows:
            ax.set_axis_off()
            continue
        for label in ["block_control", "block_a"]:
            sub = rows[rows["label"] == label].sort_values("step")
            vals = pd.to_numeric(sub[col], errors="coerce")
            if col == "train_block_recon_grad_ratio":
                vals = vals.where(vals > 0.0)
            ax.plot(sub["step"], vals, marker="o", linewidth=1.3, label=label)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("VAE step")
    axes[-1, 1].set_xlabel("VAE step")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "block_training_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in RUNS:
        if _run_dir(label).exists():
            _require_complete(label)
    summary = pd.DataFrame([_summary_row(label) for label in RUNS if _run_dir(label).exists()])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    _plot_summary(summary)
    _plot_eval_curves()
    if {"block_control", "block_a"}.issubset(set(summary["label"])):
        _plot_step0_deltas()
        _plot_block_training()
    print(f"[block_recon_analysis] wrote {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
