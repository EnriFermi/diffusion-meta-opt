from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
OUT_DIR = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/clean_harness_analysis"

RUNS = {
    "control_clean": "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
    "a_cap1_clean": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
}

KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]
CURVE_KEY_COLS = KEY_COLS + ["step"]
REQUIRED_FILES = [
    "artifact_manifest.json",
    "config.json",
    "downstream_curves.csv",
    "downstream_results.csv",
    "downstream_start_bank.csv",
    "geometry.csv",
    "preconditioning_diagnostics.csv",
    "selected_lrs.csv",
    "vae_metrics.csv",
]


def _run_dir(label: str) -> Path:
    return ARTIFACT_ROOT / RUNS[label]


def _check_complete(label: str) -> None:
    run_dir = _run_dir(label)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{label} missing required files: {missing}")
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    status = manifest.get("summary", {}).get("status")
    if status != "complete":
        raise RuntimeError(f"{label} manifest status={status!r}, expected complete")


def _load_results(label: str, method: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _load_curves(label: str, method: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _selected_lr(label: str, method: str) -> float:
    rows = pd.read_csv(_run_dir(label) / "selected_lrs.csv")
    hit = rows[(rows["method"].astype(str) == method) & (pd.to_numeric(rows["selected"], errors="coerce") == 1)]
    if len(hit) != 1:
        raise RuntimeError(f"{label} method={method} has {len(hit)} selected LR rows, expected exactly one")
    return float(hit["candidate_lr"].iloc[0])


def _tuning_lrs(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "selected_lrs.csv")
    rows["label"] = label
    return rows


def _diag_summary(label: str) -> dict[str, float]:
    rows = pd.read_csv(_run_dir(label) / "preconditioning_diagnostics.csv")
    out: dict[str, float] = {}
    for col in [
        "li_A_full_per_dim",
        "hvp_probe_a_loss_per_dim",
        "hvp_probe_trace_m_per_dim",
        "hessian_log_abs_var",
        "hessian_log_abs_spread",
        "hessian_negative_fraction",
    ]:
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        out[f"{col}_median"] = float(vals.median())
        out[f"{col}_p95"] = float(vals.quantile(0.95))
    return out


def _geometry_summary(label: str) -> dict[str, float]:
    row = pd.read_csv(_run_dir(label) / "geometry.csv").iloc[0]
    return {
        "isometry_objective": float(row["isometry_objective"]),
        "trace_g_median": float(row["trace_g_median"]),
        "trace_g2_median": float(row["trace_g2_median"]),
        "condition_median": float(row["condition_median"]),
        "condition_p90": float(row["condition_p90"]),
        "log_eig_spread_median": float(row["log_eig_spread_median"]),
    }


def _train_history(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    rows = rows[record_type != "vae_quality"].copy()
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    return rows[rows["step"].notna()].copy()


def _quality_rows(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    rows = rows[record_type == "vae_quality"].copy()
    rows["label"] = label
    return rows


def _hash_check() -> dict[str, float | int | bool]:
    control = _train_history("control_clean")[["step", "train_vae_batch_index_hash"]].dropna()
    variant = _train_history("a_cap1_clean")[["step", "train_vae_batch_index_hash"]].dropna()
    merged = control.merge(variant, on="step", suffixes=("_control", "_a"))
    mismatches = merged["train_vae_batch_index_hash_control"] != merged["train_vae_batch_index_hash_a"]
    return {
        "control_logged_hash_steps": int(len(control)),
        "a_logged_hash_steps": int(len(variant)),
        "common_logged_hash_steps": int(len(merged)),
        "common_hash_mismatches": int(mismatches.sum()),
        "common_hashes_match": bool(not mismatches.any()),
        "control_only_logged_hash_steps": int(len(set(control["step"]) - set(variant["step"]))),
        "a_only_logged_hash_steps": int(len(set(variant["step"]) - set(control["step"]))),
    }


def _start_bank_check() -> dict[str, bool | int]:
    control = pd.read_csv(_run_dir("control_clean") / "downstream_start_bank.csv")
    variant = pd.read_csv(_run_dir("a_cap1_clean") / "downstream_start_bank.csv")
    cols = [col for col in control.columns if col in variant.columns]
    return {
        "control_start_bank_rows": int(len(control)),
        "a_start_bank_rows": int(len(variant)),
        "start_bank_common_cols_equal": bool(control[cols].equals(variant[cols])),
    }


def _build_run_summary() -> pd.DataFrame:
    rows = []
    for label in RUNS:
        decoder = _load_results(label, "decoder_latent")
        raw = _load_results(label, "raw")
        dec_step0 = _load_curves(label, "decoder_latent").query("step == 0")
        history = _train_history(label)
        quality = _quality_rows(label)
        row = {
            "label": label,
            "run_label": RUNS[label],
            "selected_raw_lr": _selected_lr(label, "raw"),
            "selected_decoder_lr": _selected_lr(label, "decoder_latent"),
            "decoder_aulc_mean": float(decoder["aulc"].mean()),
            "decoder_aulc_median": float(decoder["aulc"].median()),
            "decoder_step0_test_loss_mean": float(dec_step0["test_loss"].mean()),
            "decoder_step0_test_loss_median": float(dec_step0["test_loss"].median()),
            "decoder_final_test_loss_mean": float(decoder["final_test_loss"].mean()),
            "decoder_final_test_loss_median": float(decoder["final_test_loss"].median()),
            "decoder_reconstruction_rel_l2_mean": float(decoder["reconstruction_rel_l2"].mean()),
            "decoder_reconstruction_rel_l2_median": float(decoder["reconstruction_rel_l2"].median()),
            "raw_aulc_mean": float(raw["aulc"].mean()),
            "raw_aulc_median": float(raw["aulc"].median()),
            "train_recon_mse_final": float(pd.to_numeric(history["train_recon_mse"], errors="coerce").dropna().iloc[-1]),
            "val_recon_mse_final": float(pd.to_numeric(history["val_recon_mse"], errors="coerce").dropna().iloc[-1]),
            "quality_recon_rel_l2_median": float(pd.to_numeric(quality["reconstruction_rel_l2"], errors="coerce").median()),
        }
        row.update(_diag_summary(label))
        row.update(_geometry_summary(label))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "clean_run_summary.csv", index=False)
    return summary


def _assert_same_eval_keys(method: str) -> None:
    control = _load_results("control_clean", method)[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    variant = _load_results("a_cap1_clean", method)[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    if not control.equals(variant):
        raise RuntimeError(f"eval result keys differ for method={method}")


def _paired_tables(method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _assert_same_eval_keys(method)
    control_results = _load_results("control_clean", method)
    variant_results = _load_results("a_cap1_clean", method)
    control_key = control_results[
        KEY_COLS + ["aulc", "final_test_loss", "final_test_acc", "reconstruction_rel_l2", "lr"]
    ].rename(
        columns={
            "aulc": "control_aulc",
            "final_test_loss": "control_final_test_loss",
            "final_test_acc": "control_final_test_acc",
            "reconstruction_rel_l2": "control_reconstruction_rel_l2",
            "lr": "control_lr",
        }
    )
    results = variant_results.merge(control_key, on=KEY_COLS, how="left", validate="one_to_one")
    results["method_analyzed"] = method
    results["delta_aulc"] = results["aulc"] - results["control_aulc"]
    results["delta_final_test_loss"] = results["final_test_loss"] - results["control_final_test_loss"]
    results["delta_final_test_acc"] = results["final_test_acc"] - results["control_final_test_acc"]
    results["delta_reconstruction_rel_l2"] = results["reconstruction_rel_l2"] - results["control_reconstruction_rel_l2"]

    control_curves = _load_curves("control_clean", method)
    variant_curves = _load_curves("a_cap1_clean", method)
    control_curve_key = control_curves[CURVE_KEY_COLS + ["train_loss", "test_loss", "test_acc"]].rename(
        columns={
            "train_loss": "control_train_loss",
            "test_loss": "control_test_loss",
            "test_acc": "control_test_acc",
        }
    )
    curves = variant_curves.merge(control_curve_key, on=CURVE_KEY_COLS, how="left", validate="one_to_one")
    curves["method_analyzed"] = method
    curves["delta_train_loss"] = curves["train_loss"] - curves["control_train_loss"]
    curves["delta_test_loss"] = curves["test_loss"] - curves["control_test_loss"]
    curves["delta_test_acc"] = curves["test_acc"] - curves["control_test_acc"]
    step0 = curves[curves["step"] == 0][KEY_COLS + ["delta_train_loss", "delta_test_loss", "delta_test_acc"]].rename(
        columns={
            "delta_train_loss": "delta_step0_train_loss",
            "delta_test_loss": "delta_step0_test_loss",
            "delta_test_acc": "delta_step0_test_acc",
        }
    )
    results = results.merge(step0, on=KEY_COLS, how="left", validate="one_to_one")
    curves = curves.merge(step0, on=KEY_COLS, how="left", validate="many_to_one")
    curves["step0_gap_closed_test_loss"] = curves["delta_step0_test_loss"] - curves["delta_test_loss"]
    return results, curves


def _pair_summary(results: pd.DataFrame, curves: pd.DataFrame, method: str) -> pd.DataFrame:
    step0 = pd.to_numeric(results["delta_step0_test_loss"], errors="coerce")
    aulc = pd.to_numeric(results["delta_aulc"], errors="coerce")
    row: dict[str, float | int | str] = {
        "method": method,
        "n_eval_starts": int(len(results)),
        "mean_delta_aulc": float(aulc.mean()),
        "median_delta_aulc": float(aulc.median()),
        "frac_starts_a_better_aulc": float((aulc < 0).mean()),
        "mean_delta_step0_test_loss": float(step0.mean()),
        "median_delta_step0_test_loss": float(step0.median()),
        "frac_starts_a_better_step0": float((step0 < 0).mean()),
        "mean_delta_final_test_loss": float(pd.to_numeric(results["delta_final_test_loss"], errors="coerce").mean()),
        "median_delta_final_test_loss": float(pd.to_numeric(results["delta_final_test_loss"], errors="coerce").median()),
        "mean_delta_reconstruction_rel_l2": float(pd.to_numeric(results["delta_reconstruction_rel_l2"], errors="coerce").mean()),
        "median_delta_reconstruction_rel_l2": float(pd.to_numeric(results["delta_reconstruction_rel_l2"], errors="coerce").median()),
        "corr_step0_delta_vs_aulc_delta": float(step0.corr(aulc)) if len(results) >= 3 else float("nan"),
    }
    by_step = curves.groupby("step", as_index=False).agg(
        mean_delta_test_loss=("delta_test_loss", "mean"),
        median_delta_test_loss=("delta_test_loss", "median"),
        mean_step0_gap_closed=("step0_gap_closed_test_loss", "mean"),
    )
    for step in [0, 25, 50, 100, 200, 300]:
        hit = by_step[by_step["step"] == step]
        if hit.empty:
            row[f"mean_delta_test_loss_step{step}"] = float("nan")
            row[f"mean_step0_gap_closed_step{step}"] = float("nan")
        else:
            row[f"mean_delta_test_loss_step{step}"] = float(hit["mean_delta_test_loss"].iloc[0])
            row[f"mean_step0_gap_closed_step{step}"] = float(hit["mean_step0_gap_closed"].iloc[0])
    return pd.DataFrame([row])


def _save_pair_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_frames = []
    curve_frames = []
    summary_frames = []
    for method in ["raw", "decoder_latent"]:
        results, curves = _paired_tables(method)
        results.to_csv(OUT_DIR / f"clean_{method}_per_start_deltas.csv", index=False)
        curves.to_csv(OUT_DIR / f"clean_{method}_curve_deltas.csv", index=False)
        result_frames.append(results)
        curve_frames.append(curves)
        summary_frames.append(_pair_summary(results, curves, method))
    all_results = pd.concat(result_frames, ignore_index=True)
    all_curves = pd.concat(curve_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(OUT_DIR / "clean_pair_summary.csv", index=False)
    return all_results, all_curves, summary


def _plot_summary_bars(run_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    run = run_summary.set_index("label")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    labels = ["control_clean", "a_cap1_clean"]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [run.loc[l, "li_A_full_per_dim_p95"] for l in labels], width=0.34, label="li_A p95")
    axes[0].bar(x + 0.18, [run.loc[l, "hvp_probe_trace_m_per_dim_median"] for l in labels], width=0.34, label="trace(M) median")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, ["control", "A"])
    axes[0].set_ylabel("log scale")
    axes[0].set_title("A-objective diagnostics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - 0.18, [run.loc[l, "decoder_aulc_mean"] for l in labels], width=0.34, label="mean AULC")
    axes[1].bar(x + 0.18, [run.loc[l, "decoder_step0_test_loss_mean"] for l in labels], width=0.34, label="mean step0 test loss")
    axes[1].set_xticks(x, ["control", "A"])
    axes[1].set_title("decoder downstream")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    dec = pair_summary[pair_summary["method"] == "decoder_latent"].iloc[0]
    vals = [
        dec["mean_delta_step0_test_loss"],
        dec["mean_delta_aulc"],
        dec["mean_delta_final_test_loss"],
        dec["mean_delta_reconstruction_rel_l2"],
    ]
    names = ["step0 loss", "AULC", "final loss", "recon rel-L2"]
    colors = ["#4c78a8" if v <= 0 else "#f58518" for v in vals]
    axes[2].bar(np.arange(len(vals)), vals, color=colors)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xticks(np.arange(len(vals)), names, rotation=20, ha="right")
    axes[2].set_title("A - control deltas")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "clean_summary_bars.png", dpi=190)
    plt.close(fig)


def _plot_step0_vs_aulc(all_results: pd.DataFrame) -> None:
    dec = all_results[all_results["method_analyzed"] == "decoder_latent"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    colors = np.where(dec["delta_aulc"] <= 0, "#4c78a8", "#f58518")
    ax.scatter(dec["delta_step0_test_loss"], dec["delta_aulc"], s=58, c=colors, alpha=0.85)
    for _, row in dec.iterrows():
        ax.annotate(str(int(row["source_weight_index"])), (row["delta_step0_test_loss"], row["delta_aulc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("A - control decoded step0 test loss")
    ax.set_ylabel("A - control decoder AULC")
    ax.set_title("Paired starts: step0 damage vs downstream")
    ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "clean_step0_vs_aulc_delta.png", dpi=190)
    plt.close(fig)


def _plot_curve_deltas(all_curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3), constrained_layout=True)
    for ax, method, title in [
        (axes[0], "decoder_latent", "decoder latent"),
        (axes[1], "raw", "raw control check"),
    ]:
        rows = all_curves[all_curves["method_analyzed"] == method].copy()
        grouped = rows.groupby("step", as_index=False).agg(
            mean_delta_test_loss=("delta_test_loss", "mean"),
            sem_delta_test_loss=("delta_test_loss", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=1) / np.sqrt(max(1, s.notna().sum())))),
            mean_gap_closed=("step0_gap_closed_test_loss", "mean"),
        )
        ax.plot(grouped["step"], grouped["mean_delta_test_loss"], marker="o", markersize=3, label="mean test-loss delta")
        lo = grouped["mean_delta_test_loss"] - grouped["sem_delta_test_loss"]
        hi = grouped["mean_delta_test_loss"] + grouped["sem_delta_test_loss"]
        ax.fill_between(grouped["step"].to_numpy(dtype=float), lo.to_numpy(dtype=float), hi.to_numpy(dtype=float), alpha=0.18)
        if method == "decoder_latent":
            ax.plot(grouped["step"], grouped["mean_gap_closed"], marker="s", markersize=3, label="step0 gap closed")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("downstream step")
        ax.set_ylabel("A - control")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "clean_mean_curve_deltas.png", dpi=190)
    plt.close(fig)


def _plot_tail_start_curves(all_results: pd.DataFrame) -> None:
    dec = all_results[all_results["method_analyzed"] == "decoder_latent"].copy()
    chosen = pd.concat([dec.nsmallest(2, "delta_aulc"), dec.nlargest(3, "delta_aulc")]).drop_duplicates(KEY_COLS)
    control_curves = _load_curves("control_clean", "decoder_latent")
    variant_curves = _load_curves("a_cap1_clean", "decoder_latent")
    fig, axes = plt.subplots(len(chosen), 1, figsize=(8.5, max(3.0, 2.2 * len(chosen))), sharex=True, constrained_layout=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, chosen.iterrows(), strict=True):
        mask_control = np.logical_and.reduce([control_curves[col] == row[col] for col in KEY_COLS])
        mask_variant = np.logical_and.reduce([variant_curves[col] == row[col] for col in KEY_COLS])
        c = control_curves[mask_control].sort_values("step")
        a = variant_curves[mask_variant].sort_values("step")
        ax.plot(c["step"], c["test_loss"], marker="o", markersize=3, label="control")
        ax.plot(a["step"], a["test_loss"], marker="o", markersize=3, label="A")
        ax.set_ylabel("test loss")
        ax.set_title(
            f"weight {int(row['source_weight_index'])} {row['task_name']} "
            f"delta AULC={row['delta_aulc']:+.4f} step0={row['delta_step0_test_loss']:+.4f}"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("downstream step")
    fig.savefig(OUT_DIR / "clean_tail_start_curves.png", dpi=190)
    plt.close(fig)


def _plot_vae_training() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), constrained_layout=True)
    for label, color in [("control_clean", "#4c78a8"), ("a_cap1_clean", "#f58518")]:
        hist = _train_history(label).sort_values("step")
        pretty = "control" if label == "control_clean" else "A"
        axes[0].plot(hist["step"], hist["train_recon_mse"], color=color, alpha=0.7, label=f"{pretty} train")
        axes[0].plot(hist["step"], hist["val_recon_mse"], color=color, linestyle="--", label=f"{pretty} val")
    axes[0].set_xlabel("VAE step")
    axes[0].set_ylabel("normalized reconstruction MSE")
    axes[0].set_title("VAE reconstruction")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    a_hist = _train_history("a_cap1_clean").sort_values("step")
    a_pre = a_hist[pd.to_numeric(a_hist["train_precond_effective_loss"], errors="coerce").notna()].copy()
    axes[1].plot(a_pre["step"], a_pre["train_precond_effective_loss"], label="effective A loss", color="#f58518")
    axes[1].plot(a_pre["step"], a_pre["train_precond_a_loss"], label="raw sampled A loss", color="#4c78a8", alpha=0.8)
    axes[1].set_xlabel("VAE step")
    axes[1].set_title("A regularizer during training")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.savefig(OUT_DIR / "clean_vae_training.png", dpi=190)
    plt.close(fig)


def _plot_li_distributions() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    cols = [
        ("li_A_full_per_dim", "exact li_A / dim"),
        ("hvp_probe_trace_m_per_dim", "trace(M) / dim"),
        ("hessian_log_abs_var", "Hessian log-abs variance"),
    ]
    for ax, (col, title) in zip(axes, cols, strict=True):
        for label, color in [("control_clean", "#4c78a8"), ("a_cap1_clean", "#f58518")]:
            rows = pd.read_csv(_run_dir(label) / "preconditioning_diagnostics.csv")
            vals = pd.to_numeric(rows[col], errors="coerce").dropna().to_numpy()
            pretty = "control" if label == "control_clean" else "A"
            ax.hist(vals, bins=18, alpha=0.45, color=color, label=pretty)
            ax.axvline(np.median(vals), color=color, linewidth=2)
        if col != "hessian_log_abs_var":
            ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "clean_li_distributions.png", dpi=190)
    plt.close(fig)


def _plot_lr_tuning() -> None:
    rows = pd.concat([_tuning_lrs("control_clean"), _tuning_lrs("a_cap1_clean")], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), constrained_layout=True)
    for ax, method in zip(axes, ["raw", "decoder_latent"], strict=True):
        sub = rows[rows["method"].astype(str) == method].copy()
        for label, color in [("control_clean", "#4c78a8"), ("a_cap1_clean", "#f58518")]:
            run = sub[sub["label"] == label].sort_values("candidate_lr")
            pretty = "control" if label == "control_clean" else "A"
            ax.plot(run["candidate_lr"], run["tuning_median_aulc"], marker="o", markersize=3, color=color, label=pretty)
            selected = run[pd.to_numeric(run["selected"], errors="coerce") == 1]
            if not selected.empty:
                ax.scatter(selected["candidate_lr"], selected["tuning_median_aulc"], s=80, color=color, edgecolor="black", zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("candidate LR")
        ax.set_ylabel("tuning median AULC")
        ax.set_title(method)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "clean_lr_tuning.png", dpi=190)
    plt.close(fig)


def _plot_reconstruction_vs_downstream(all_results: pd.DataFrame) -> None:
    dec = all_results[all_results["method_analyzed"] == "decoder_latent"].copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.6), constrained_layout=True)
    ax.scatter(dec["delta_reconstruction_rel_l2"], dec["delta_aulc"], s=58, alpha=0.85)
    for _, row in dec.iterrows():
        ax.annotate(str(int(row["source_weight_index"])), (row["delta_reconstruction_rel_l2"], row["delta_aulc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("A - control reconstruction rel-L2")
    ax.set_ylabel("A - control decoder AULC")
    ax.set_title("Reconstruction change vs downstream change")
    ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "clean_reconstruction_vs_aulc.png", dpi=190)
    plt.close(fig)


def _write_notes(run_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    dec = pair_summary[pair_summary["method"] == "decoder_latent"].iloc[0]
    raw = pair_summary[pair_summary["method"] == "raw"].iloc[0]
    run = run_summary.set_index("label")
    lines = [
        "# Variant A Clean Harness Analysis",
        "",
        "## Technical Checks",
        "",
        f"- Batch-hash check: `{_hash_check()}`.",
        f"- Start-bank check: `{_start_bank_check()}`.",
        f"- Selected decoder LR: control `{run.loc['control_clean', 'selected_decoder_lr']}`, A `{run.loc['a_cap1_clean', 'selected_decoder_lr']}`.",
        f"- Raw paired mean delta AULC: `{raw['mean_delta_aulc']:.12g}`; this should be exactly zero or numerical noise.",
        "",
        "## Main Paired Result",
        "",
        f"- A reduced exact `li_A_full_per_dim` p95 from `{run.loc['control_clean', 'li_A_full_per_dim_p95']:.6g}` to `{run.loc['a_cap1_clean', 'li_A_full_per_dim_p95']:.6g}`.",
        f"- A reduced `hvp_probe_trace_m_per_dim` median from `{run.loc['control_clean', 'hvp_probe_trace_m_per_dim_median']:.6g}` to `{run.loc['a_cap1_clean', 'hvp_probe_trace_m_per_dim_median']:.6g}`.",
        f"- Decoder mean AULC delta A-control: `{dec['mean_delta_aulc']:+.6g}`; median delta `{dec['median_delta_aulc']:+.6g}`.",
        f"- Decoder mean step0 test-loss delta A-control: `{dec['mean_delta_step0_test_loss']:+.6g}`; median delta `{dec['median_delta_step0_test_loss']:+.6g}`.",
        f"- Fraction of eval starts where A is better by AULC: `{dec['frac_starts_a_better_aulc']:.3f}`.",
        f"- Corr(step0 delta, AULC delta): `{dec['corr_step0_delta_vs_aulc_delta']:.6g}`.",
        "",
        "## Plots",
        "",
        "- `clean_summary_bars.png`",
        "- `clean_step0_vs_aulc_delta.png`",
        "- `clean_mean_curve_deltas.png`",
        "- `clean_tail_start_curves.png`",
        "- `clean_vae_training.png`",
        "- `clean_li_distributions.png`",
        "- `clean_lr_tuning.png`",
        "- `clean_reconstruction_vs_aulc.png`",
    ]
    (OUT_DIR / "clean_harness_analysis_notes.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in RUNS:
        _check_complete(label)
    run_summary = _build_run_summary()
    all_results, all_curves, pair_summary = _save_pair_tables()
    technical = {**_hash_check(), **_start_bank_check()}
    pd.DataFrame([technical]).to_csv(OUT_DIR / "clean_technical_checks.csv", index=False)
    _plot_summary_bars(run_summary, pair_summary)
    _plot_step0_vs_aulc(all_results)
    _plot_curve_deltas(all_curves)
    _plot_tail_start_curves(all_results)
    _plot_vae_training()
    _plot_li_distributions()
    _plot_lr_tuning()
    _plot_reconstruction_vs_downstream(all_results)
    _write_notes(run_summary, pair_summary)
    print(f"[analyze_variant_a_clean_harness] wrote {OUT_DIR}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
