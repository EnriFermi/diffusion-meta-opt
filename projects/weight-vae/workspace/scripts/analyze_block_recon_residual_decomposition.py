from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
OUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_analysis/lam0p03/residual_decomposition"
)

RUNS = {
    "clean_control": "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
    "clean_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
    "block_control": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0",
    "block_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p03_clean_harness_v1_seed0",
}

PAIRS = {
    "clean": ("clean_control", "clean_a"),
    "block_lam0p03": ("block_control", "block_a"),
}

KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]


def _run_dir(label: str) -> Path:
    return ARTIFACT_ROOT / RUNS[label]


def _load_results(label: str, method: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _load_curves(label: str, method: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    for col in ["step", "train_loss", "test_loss", "train_acc", "test_acc"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def _pair_results(tag: str, method: str) -> pd.DataFrame:
    control_label, a_label = PAIRS[tag]
    control = _load_results(control_label, method)
    variant = _load_results(a_label, method)
    keep = KEY_COLS + [
        "aulc",
        "final_train_loss",
        "final_test_loss",
        "final_test_acc",
        "reconstruction_rel_l2",
    ]
    merged = control[keep].merge(variant[keep], on=KEY_COLS, suffixes=("_control", "_a"))
    merged["tag"] = tag
    merged["method"] = method
    for col in ["aulc", "final_train_loss", "final_test_loss", "final_test_acc", "reconstruction_rel_l2"]:
        merged[f"delta_{col}"] = merged[f"{col}_a"] - merged[f"{col}_control"]
    return merged


def _pair_curves(tag: str, method: str) -> pd.DataFrame:
    control_label, a_label = PAIRS[tag]
    control = _load_curves(control_label, method)
    variant = _load_curves(a_label, method)
    keep = KEY_COLS + ["step", "train_loss", "test_loss", "train_acc", "test_acc"]
    merged = control[keep].merge(variant[keep], on=KEY_COLS + ["step"], suffixes=("_control", "_a"))
    merged["tag"] = tag
    merged["method"] = method
    for col in ["train_loss", "test_loss", "train_acc", "test_acc"]:
        merged[f"delta_{col}"] = merged[f"{col}_a"] - merged[f"{col}_control"]
    return merged


def _decomposition(tag: str, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_delta = _pair_results(tag, method)
    curve_delta = _pair_curves(tag, method)
    step0 = curve_delta[curve_delta["step"] == 0][
        KEY_COLS + ["delta_train_loss", "delta_test_loss", "delta_train_acc", "delta_test_acc"]
    ].rename(
        columns={
            "delta_train_loss": "step0_delta_train_loss",
            "delta_test_loss": "step0_delta_test_loss",
            "delta_train_acc": "step0_delta_train_acc",
            "delta_test_acc": "step0_delta_test_acc",
        }
    )
    curve_mean = (
        curve_delta.groupby(KEY_COLS, as_index=False)
        .agg(
            curve_mean_delta_train_loss=("delta_train_loss", "mean"),
            curve_mean_delta_test_loss=("delta_test_loss", "mean"),
            curve_post0_delta_train_loss=("delta_train_loss", lambda x: float(np.mean(np.asarray(x)[1:]))),
            curve_post0_delta_test_loss=("delta_test_loss", lambda x: float(np.mean(np.asarray(x)[1:]))),
            curve_final_delta_train_loss=("delta_train_loss", "last"),
            curve_final_delta_test_loss=("delta_test_loss", "last"),
        )
        .reset_index(drop=True)
    )
    per_start = result_delta.merge(step0, on=KEY_COLS).merge(curve_mean, on=KEY_COLS)
    per_start["curve_delta_minus_step0_train_loss"] = (
        per_start["curve_mean_delta_train_loss"] - per_start["step0_delta_train_loss"]
    )
    per_start["curve_delta_minus_step0_test_loss"] = (
        per_start["curve_mean_delta_test_loss"] - per_start["step0_delta_test_loss"]
    )
    summary_rows = []
    for task_name, sub in [("all", per_start), *list(per_start.groupby("task_name"))]:
        step0_train = pd.to_numeric(sub["step0_delta_train_loss"], errors="coerce")
        step0_test = pd.to_numeric(sub["step0_delta_test_loss"], errors="coerce")
        aulc = pd.to_numeric(sub["delta_aulc"], errors="coerce")
        summary_rows.append(
            {
                "tag": tag,
                "method": method,
                "task_name": task_name,
                "starts": int(len(sub)),
                "mean_delta_aulc": float(aulc.mean()),
                "median_delta_aulc": float(aulc.median()),
                "worse_aulc_count": int((aulc > 0).sum()),
                "mean_step0_delta_train_loss": float(step0_train.mean()),
                "median_step0_delta_train_loss": float(step0_train.median()),
                "mean_step0_delta_test_loss": float(step0_test.mean()),
                "median_step0_delta_test_loss": float(step0_test.median()),
                "mean_curve_delta_train_loss": float(sub["curve_mean_delta_train_loss"].mean()),
                "mean_curve_delta_test_loss": float(sub["curve_mean_delta_test_loss"].mean()),
                "mean_post0_delta_train_loss": float(sub["curve_post0_delta_train_loss"].mean()),
                "mean_post0_delta_test_loss": float(sub["curve_post0_delta_test_loss"].mean()),
                "corr_step0_train_vs_aulc": float(step0_train.corr(aulc)) if len(sub) >= 3 else float("nan"),
                "corr_step0_test_vs_aulc": float(step0_test.corr(aulc)) if len(sub) >= 3 else float("nan"),
            }
        )
    return per_start, pd.DataFrame(summary_rows)


def _plot_curve_deltas(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, col, title in [
        (axes[0], "delta_train_loss", "A - control train loss by downstream step"),
        (axes[1], "delta_test_loss", "A - control test loss by downstream step"),
    ]:
        for tag in PAIRS:
            sub = (
                curves[(curves["tag"] == tag) & (curves["method"] == "decoder_latent")]
                .groupby("step", as_index=False)[col]
                .mean()
                .sort_values("step")
            )
            ax.plot(sub["step"], sub[col], marker="o", linewidth=1.6, label=tag)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Downstream step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean delta over eval starts")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decoder_latent_curve_deltas.png", dpi=190)
    plt.close(fig)


def _plot_step0_vs_aulc(per_start: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"mnist": "#4c78a8", "fashion_mnist": "#f58518"}
    for ax, xcol, title in [
        (axes[0], "step0_delta_train_loss", "Official AULC delta vs step0 train delta"),
        (axes[1], "step0_delta_test_loss", "Official AULC delta vs step0 test delta"),
    ]:
        for tag, marker in [("clean", "o"), ("block_lam0p03", "^")]:
            sub = per_start[(per_start["tag"] == tag) & (per_start["method"] == "decoder_latent")]
            for task_name, task_rows in sub.groupby("task_name"):
                ax.scatter(
                    task_rows[xcol],
                    task_rows["delta_aulc"],
                    marker=marker,
                    s=45,
                    color=colors.get(str(task_name), "#666666"),
                    alpha=0.8,
                    label=f"{tag} {task_name}",
                )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel(xcol)
        ax.set_ylabel("A - control official AULC")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles, strict=False))
    axes[1].legend(
        dedup.values(),
        dedup.keys(),
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "step0_vs_aulc_scatter.png", dpi=190)
    plt.close(fig)


def _plot_top_residual_curves(curves: pd.DataFrame, per_start: pd.DataFrame) -> None:
    block = per_start[(per_start["tag"] == "block_lam0p03") & (per_start["method"] == "decoder_latent")]
    top = block.nlargest(4, "delta_aulc")[KEY_COLS]
    keys = [tuple(row[col] for col in KEY_COLS) for _, row in top.iterrows()]
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 2.8 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = np.array([axes])
    for row_idx, key in enumerate(keys):
        key_mask = np.ones(len(curves), dtype=bool)
        for col, value in zip(KEY_COLS, key, strict=True):
            key_mask &= curves[col].to_numpy() == value
        sub = curves[key_mask & (curves["tag"] == "block_lam0p03") & (curves["method"] == "decoder_latent")]
        title = f"src={key[0]} start={key[1]} task={key[2]}"
        for ax, loss_col, ylabel in [
            (axes[row_idx, 0], "train_loss", "train loss"),
            (axes[row_idx, 1], "test_loss", "test loss"),
        ]:
            for suffix, color in [("control", "#4c78a8"), ("a", "#e45756")]:
                ax.plot(
                    sub["step"],
                    sub[f"{loss_col}_{suffix}"],
                    marker="o",
                    linewidth=1.4,
                    color=color,
                    label=suffix,
                )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Downstream step")
    axes[-1, 1].set_xlabel("Downstream step")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_block_residual_start_curves.png", dpi=190)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[residual_decomp] output_dir={OUT_DIR}", flush=True)
    per_start_frames = []
    summary_frames = []
    curve_frames = []
    for tag in PAIRS:
        for method in ["raw", "decoder_latent"]:
            print(f"[residual_decomp] pair={tag} method={method}", flush=True)
            per_start, summary = _decomposition(tag, method)
            per_start_frames.append(per_start)
            summary_frames.append(summary)
            curve_frames.append(_pair_curves(tag, method))
    per_start_all = pd.concat(per_start_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    curves_all = pd.concat(curve_frames, ignore_index=True)
    per_start_all.to_csv(OUT_DIR / "per_start_residual_decomposition.csv", index=False)
    summary_all.to_csv(OUT_DIR / "residual_decomposition_summary.csv", index=False)
    curves_all.to_csv(OUT_DIR / "paired_curve_deltas.csv", index=False)
    _plot_curve_deltas(curves_all)
    _plot_step0_vs_aulc(per_start_all)
    _plot_top_residual_curves(curves_all, per_start_all)
    print("[residual_decomp] wrote:")
    for path in [
        "per_start_residual_decomposition.csv",
        "residual_decomposition_summary.csv",
        "paired_curve_deltas.csv",
        "decoder_latent_curve_deltas.png",
        "step0_vs_aulc_scatter.png",
        "top_block_residual_start_curves.png",
    ]:
        print(f"  - {OUT_DIR / path}", flush=True)
    print(summary_all.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
