from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
OUT_DIR = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_analysis/lam_cap_sweep"

RUNS = {
    "clean_control": "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
    "clean_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
    "lam0p03_control": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0",
    "lam0p03_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p03_clean_harness_v1_seed0",
    "lam0p03_cap0p25_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_clean_harness_v1_seed0",
    "lam0p06_control": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p06_clean_harness_v1_seed0",
    "lam0p06_a": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p06_clean_harness_v1_seed0",
}

PAIRS = {
    "clean": ("clean_control", "clean_a", 0.0),
    "fc2_lam0p03": ("lam0p03_control", "lam0p03_a", 0.03),
    "fc2_lam0p03_cap0p25": ("lam0p03_control", "lam0p03_cap0p25_a", 0.03),
    "fc2_lam0p06": ("lam0p06_control", "lam0p06_a", 0.06),
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
    hit = rows[
        (rows["method"].astype(str) == method)
        & (pd.to_numeric(rows["selected"], errors="coerce") == 1)
    ]
    return float(hit["candidate_lr"].iloc[0]) if len(hit) == 1 else float("nan")


def _quality(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    return rows[record_type == "vae_quality"].copy()


def _train_history(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    train = rows[record_type != "vae_quality"].copy()
    train["step"] = pd.to_numeric(train["step"], errors="coerce")
    return train[train["step"].notna()].copy()


def _load_results(label: str, method: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    rows = rows[
        (rows["split"].astype(str) == "eval")
        & (rows["method"].astype(str) == method)
    ].copy()
    rows["label"] = label
    for col in ["aulc", "final_train_loss", "final_test_loss", "final_test_acc", "reconstruction_rel_l2"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def _load_curves(label: str, method: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    rows = rows[
        (rows["split"].astype(str) == "eval")
        & (rows["method"].astype(str) == method)
    ].copy()
    rows["label"] = label
    for col in ["step", "train_loss", "test_loss", "train_acc", "test_acc"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def _summary_row(label: str) -> dict[str, float | str]:
    downstream = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    dec = downstream[
        (downstream["split"].astype(str) == "eval")
        & (downstream["method"].astype(str) == "decoder_latent")
    ].copy()
    raw = downstream[
        (downstream["split"].astype(str) == "eval")
        & (downstream["method"].astype(str) == "raw")
    ].copy()
    dec["aulc"] = pd.to_numeric(dec["aulc"], errors="coerce")
    raw["aulc"] = pd.to_numeric(raw["aulc"], errors="coerce")
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
        "reconstruction_rel_l2_median": float(
            pd.to_numeric(quality["reconstruction_rel_l2"], errors="coerce").median()
        ),
        "decoded_test_loss_median": float(
            pd.to_numeric(quality["decoded_test_loss"], errors="coerce").median()
        ),
        "decoded_test_acc_median": float(
            pd.to_numeric(quality["decoded_test_acc"], errors="coerce").median()
        ),
        "val_recon_mse_final": float(pd.to_numeric(train["val_recon_mse"], errors="coerce").dropna().iloc[-1]),
        "li_A_full_per_dim_p95": float(pd.to_numeric(diag["li_A_full_per_dim"], errors="coerce").quantile(0.95)),
        "hvp_probe_trace_m_per_dim_median": float(
            pd.to_numeric(diag["hvp_probe_trace_m_per_dim"], errors="coerce").median()
        ),
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


def _pair_results(tag: str, method: str) -> pd.DataFrame:
    control_label, a_label, lam = PAIRS[tag]
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
    merged["lam"] = lam
    merged["method"] = method
    for col in ["aulc", "final_train_loss", "final_test_loss", "final_test_acc", "reconstruction_rel_l2"]:
        merged[f"delta_{col}"] = merged[f"{col}_a"] - merged[f"{col}_control"]
    return merged


def _pair_curves(tag: str, method: str) -> pd.DataFrame:
    control_label, a_label, lam = PAIRS[tag]
    control = _load_curves(control_label, method)
    variant = _load_curves(a_label, method)
    keep = KEY_COLS + ["step", "train_loss", "test_loss", "train_acc", "test_acc"]
    merged = control[keep].merge(variant[keep], on=KEY_COLS + ["step"], suffixes=("_control", "_a"))
    merged["tag"] = tag
    merged["lam"] = lam
    merged["method"] = method
    for col in ["train_loss", "test_loss", "train_acc", "test_acc"]:
        merged[f"delta_{col}"] = merged[f"{col}_a"] - merged[f"{col}_control"]
    return merged


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _decompose_pair(tag: str, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        aulc = pd.to_numeric(sub["delta_aulc"], errors="coerce")
        step0_train = pd.to_numeric(sub["step0_delta_train_loss"], errors="coerce")
        step0_test = pd.to_numeric(sub["step0_delta_test_loss"], errors="coerce")
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
                "mean_post0_delta_train_loss": float(sub["curve_post0_delta_train_loss"].mean()),
                "mean_post0_delta_test_loss": float(sub["curve_post0_delta_test_loss"].mean()),
                "corr_step0_train_vs_aulc": _safe_corr(step0_train, aulc),
                "corr_step0_test_vs_aulc": _safe_corr(step0_test, aulc),
            }
        )
    return per_start, curve_delta, pd.DataFrame(summary_rows)


def _plot_summary(summary: pd.DataFrame) -> None:
    labels = list(RUNS.keys())
    colors = ["#536d9e", "#8e5aa7", "#3f8c6b", "#2b8a8a", "#7b6bbd", "#c27a35", "#b35c35"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [
        ("decoder_aulc_mean", "Decoder eval AULC mean"),
        ("decoder_aulc_median", "Decoder eval AULC median"),
        ("li_A_full_per_dim_p95", "li_A full per dim p95"),
        ("reconstruction_rel_l2_median", "VAE rel L2 median"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        vals = [float(summary.loc[summary["label"] == label, col].iloc[0]) for label in labels]
        ax.bar([label.replace("_", "\n") for label in labels], vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matched_summary_bars.png", dpi=190)
    plt.close(fig)


def _plot_lam_tradeoff(pair_summary: pd.DataFrame, summary: pd.DataFrame) -> None:
    dec_all = pair_summary[
        (pair_summary["method"] == "decoder_latent")
        & (pair_summary["task_name"] == "all")
        & (pair_summary["tag"].isin(["fc2_lam0p03", "fc2_lam0p06"]))
    ].copy()
    dec_all["lam"] = dec_all["tag"].map({"fc2_lam0p03": 0.03, "fc2_lam0p06": 0.06})

    rows = []
    for tag, control_label, a_label, lam in [
        ("fc2_lam0p03", "lam0p03_control", "lam0p03_a", 0.03),
        ("fc2_lam0p06", "lam0p06_control", "lam0p06_a", 0.06),
    ]:
        c = summary[summary["label"] == control_label].iloc[0]
        a = summary[summary["label"] == a_label].iloc[0]
        pair = dec_all[dec_all["tag"] == tag].iloc[0]
        rows.append(
            {
                "lam": lam,
                "mean_delta_aulc": float(pair["mean_delta_aulc"]),
                "median_delta_aulc": float(pair["median_delta_aulc"]),
                "mean_step0_delta_train_loss": float(pair["mean_step0_delta_train_loss"]),
                "li_A_p95_a_minus_control": float(a["li_A_full_per_dim_p95"] - c["li_A_full_per_dim_p95"]),
                "trace_a_minus_control": float(
                    a["hvp_probe_trace_m_per_dim_median"] - c["hvp_probe_trace_m_per_dim_median"]
                ),
            }
        )
    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(OUT_DIR / "lam_tradeoff_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in [
        (axes[0], "mean_delta_aulc", "A - control mean AULC"),
        (axes[1], "median_delta_aulc", "A - control median AULC"),
        (axes[2], "li_A_p95_a_minus_control", "A - control li_A p95"),
    ]:
        ax.plot(tradeoff["lam"], tradeoff[col], marker="o", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("fc2.weight block coeff")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lam_tradeoff.png", dpi=190)
    plt.close(fig)


def _plot_pair_tradeoff(pair_summary: pd.DataFrame, summary: pd.DataFrame) -> None:
    tags = ["clean", "fc2_lam0p03", "fc2_lam0p03_cap0p25", "fc2_lam0p06"]
    rows = []
    for tag in tags:
        control_label, a_label, lam = PAIRS[tag]
        c = summary[summary["label"] == control_label].iloc[0]
        a = summary[summary["label"] == a_label].iloc[0]
        pair = pair_summary[
            (pair_summary["method"] == "decoder_latent")
            & (pair_summary["task_name"] == "all")
            & (pair_summary["tag"] == tag)
        ].iloc[0]
        rows.append(
            {
                "tag": tag,
                "lam": float(lam),
                "cap": 0.25 if tag == "fc2_lam0p03_cap0p25" else (0.0 if tag == "clean" else 1.0),
                "mean_delta_aulc": float(pair["mean_delta_aulc"]),
                "median_delta_aulc": float(pair["median_delta_aulc"]),
                "mean_step0_delta_test_loss": float(pair["mean_step0_delta_test_loss"]),
                "median_step0_delta_test_loss": float(pair["median_step0_delta_test_loss"]),
                "mean_post0_delta_test_loss": float(pair["mean_post0_delta_test_loss"]),
                "li_A_p95_control": float(c["li_A_full_per_dim_p95"]),
                "li_A_p95_a": float(a["li_A_full_per_dim_p95"]),
                "li_A_p95_a_minus_control": float(a["li_A_full_per_dim_p95"] - c["li_A_full_per_dim_p95"]),
                "recon_rel_l2_a_minus_control": float(a["reconstruction_rel_l2_median"] - c["reconstruction_rel_l2_median"]),
            }
        )
    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(OUT_DIR / "pair_tradeoff_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    x = np.arange(len(tradeoff))
    labels = [str(v).replace("_", "\n") for v in tradeoff["tag"]]
    colors = ["#6d6d6d", "#2b8a8a", "#7b6bbd", "#b35c35"]
    for ax, col, title in [
        (axes[0], "mean_delta_aulc", "A - control mean AULC"),
        (axes[1], "mean_step0_delta_test_loss", "A - control step0 test"),
        (axes[2], "mean_post0_delta_test_loss", "A - control post-step test"),
        (axes[3], "li_A_p95_a_minus_control", "A - control li_A p95"),
    ]:
        ax.bar(x, tradeoff[col], color=colors)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pair_tradeoff_bars.png", dpi=190)
    plt.close(fig)


def _plot_curve_deltas(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True)
    colors = {
        "clean": "#6d6d6d",
        "fc2_lam0p03": "#2b8a8a",
        "fc2_lam0p03_cap0p25": "#7b6bbd",
        "fc2_lam0p06": "#b35c35",
    }
    for ax, col, title in [
        (axes[0], "delta_train_loss", "A - control train loss by downstream step"),
        (axes[1], "delta_test_loss", "A - control test loss by downstream step"),
    ]:
        for tag in PAIRS:
            sub = curves[(curves["method"] == "decoder_latent") & (curves["tag"] == tag)]
            agg = sub.groupby("step", as_index=False)[col].mean().sort_values("step")
            ax.plot(agg["step"], agg[col], marker="o", linewidth=1.6, label=tag, color=colors[tag])
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    colors = {"mnist": "#4c78a8", "fashion_mnist": "#f58518"}
    markers = {"clean": "o", "fc2_lam0p03": "^", "fc2_lam0p03_cap0p25": "D", "fc2_lam0p06": "s"}
    for ax, xcol, title in [
        (axes[0], "step0_delta_train_loss", "AULC delta vs step0 train delta"),
        (axes[1], "step0_delta_test_loss", "AULC delta vs step0 test delta"),
    ]:
        for tag in PAIRS:
            sub = per_start[(per_start["method"] == "decoder_latent") & (per_start["tag"] == tag)]
            for task_name, task_rows in sub.groupby("task_name"):
                ax.scatter(
                    task_rows[xcol],
                    task_rows["delta_aulc"],
                    marker=markers[tag],
                    color=colors.get(str(task_name), "#666666"),
                    s=48,
                    alpha=0.82,
                    label=f"{tag} {task_name}",
                )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel(xcol)
        ax.set_ylabel("A - control AULC")
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
    fig.savefig(OUT_DIR / "step0_vs_aulc_lam_sweep.png", dpi=190)
    plt.close(fig)


def _plot_aulc_delta_by_start(per_start: pd.DataFrame) -> None:
    dec = per_start[
        (per_start["method"] == "decoder_latent")
        & (per_start["tag"].isin(["fc2_lam0p03", "fc2_lam0p03_cap0p25", "fc2_lam0p06"]))
    ].copy()
    dec["start_key"] = (
        dec["source_weight_index"].astype(str)
        + "/"
        + dec["start_index"].astype(str)
        + "\n"
        + dec["task_name"].astype(str)
    )
    pivot = dec.pivot_table(index="start_key", columns="tag", values="delta_aulc", aggfunc="first")
    sort_col = "fc2_lam0p03_cap0p25" if "fc2_lam0p03_cap0p25" in pivot else "fc2_lam0p06"
    pivot = pivot.sort_values(sort_col, ascending=False)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(pivot))
    width = 0.26
    ax.bar(x - width, pivot["fc2_lam0p03"], width, label="fc2_lam0p03 cap1", color="#2b8a8a")
    ax.bar(x, pivot["fc2_lam0p03_cap0p25"], width, label="fc2_lam0p03 cap0.25", color="#7b6bbd")
    ax.bar(x + width, pivot["fc2_lam0p06"], width, label="fc2_lam0p06 cap1", color="#b35c35")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("A - control decoder AULC")
    ax.set_title(f"Per-start AULC deltas sorted by {sort_col} damage")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aulc_delta_by_start.png", dpi=190)
    plt.close(fig)


def _plot_top_curves(curves: pd.DataFrame, per_start: pd.DataFrame, *, tag: str, output_name: str) -> None:
    block = per_start[
        (per_start["tag"] == tag)
        & (per_start["method"] == "decoder_latent")
    ].copy()
    top = block.nlargest(4, "delta_aulc")[KEY_COLS]
    keys = [tuple(row[col] for col in KEY_COLS) for _, row in top.iterrows()]
    fig, axes = plt.subplots(len(keys), 2, figsize=(10.5, 2.8 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = np.array([axes])
    for row_idx, key in enumerate(keys):
        mask = np.ones(len(curves), dtype=bool)
        for col, value in zip(KEY_COLS, key, strict=True):
            mask &= curves[col].to_numpy() == value
        sub = curves[
            mask
            & (curves["tag"] == tag)
            & (curves["method"] == "decoder_latent")
        ]
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
    fig.savefig(OUT_DIR / output_name, dpi=190)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[lam_sweep] output_dir={OUT_DIR}", flush=True)
    for label in RUNS:
        _require_complete(label)

    summary = pd.DataFrame([_summary_row(label) for label in RUNS])
    summary.to_csv(OUT_DIR / "run_summary.csv", index=False)

    per_start_frames = []
    curve_frames = []
    pair_summary_frames = []
    for tag in PAIRS:
        for method in ["raw", "decoder_latent"]:
            print(f"[lam_sweep] decompose tag={tag} method={method}", flush=True)
            per_start, curves, pair_summary = _decompose_pair(tag, method)
            per_start_frames.append(per_start)
            curve_frames.append(curves)
            pair_summary_frames.append(pair_summary)
    per_start = pd.concat(per_start_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    pair_summary = pd.concat(pair_summary_frames, ignore_index=True)
    per_start.to_csv(OUT_DIR / "per_start_deltas.csv", index=False)
    curves.to_csv(OUT_DIR / "paired_curve_deltas.csv", index=False)
    pair_summary.to_csv(OUT_DIR / "pair_delta_summary.csv", index=False)

    _plot_summary(summary)
    _plot_lam_tradeoff(pair_summary, summary)
    _plot_pair_tradeoff(pair_summary, summary)
    _plot_curve_deltas(curves)
    _plot_step0_vs_aulc(per_start)
    _plot_aulc_delta_by_start(per_start)
    _plot_top_curves(curves, per_start, tag="fc2_lam0p06", output_name="top_lam0p06_start_curves.png")
    _plot_top_curves(curves, per_start, tag="fc2_lam0p03_cap0p25", output_name="top_lam0p03_cap0p25_start_curves.png")

    print("[lam_sweep] run_summary")
    print(summary.to_string(index=False))
    print("[lam_sweep] pair_delta_summary")
    print(pair_summary.to_string(index=False))
    print(f"[lam_sweep] wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
