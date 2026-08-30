from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
DEFAULT_OUT = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/function_anchor_analysis"

RUNS = {
    "original_control": "sage_cnn_vae_smoothing_celo_meta_finetune_control_current_v1_seed0",
    "original_A_cap1": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_frontier_alpha0p0100_cap1p000_v1_seed0",
    "anchor_control_c1e-3": "sage_cnn_vae_smoothing_celo_meta_control_logit_anchor_c0p0010_v1_seed0",
    "anchor_A_cap1_c1e-3": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_logit_anchor_c0p0010_v1_seed0",
}

MATCHED_CONTROLS = {
    "original_control": "original_control",
    "original_A_cap1": "original_control",
    "anchor_control_c1e-3": "anchor_control_c1e-3",
    "anchor_A_cap1_c1e-3": "anchor_control_c1e-3",
}

VS_ORIGINAL_LABELS = ["original_A_cap1", "anchor_control_c1e-3", "anchor_A_cap1_c1e-3"]

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
        raise RuntimeError(f"{label} manifest status is {status!r}, expected complete")


def _load_eval_results(label: str, method: str = "decoder_latent") -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _load_eval_curves(label: str, method: str = "decoder_latent") -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _selected_lr(label: str, method: str) -> float:
    rows = pd.read_csv(_run_dir(label) / "selected_lrs.csv")
    hit = rows[(rows["method"].astype(str) == method) & (pd.to_numeric(rows["selected"], errors="coerce") == 1)]
    if hit.empty:
        return float("nan")
    return float(hit["candidate_lr"].iloc[0])


def _li_summary(label: str) -> dict[str, float]:
    rows = pd.read_csv(_run_dir(label) / "preconditioning_diagnostics.csv")
    return {
        "li_A_full_per_dim_median": float(rows["li_A_full_per_dim"].median()),
        "li_A_full_per_dim_p95": float(rows["li_A_full_per_dim"].quantile(0.95)),
        "hvp_probe_a_loss_per_dim_median": float(rows["hvp_probe_a_loss_per_dim"].median()),
        "hvp_probe_trace_m_per_dim_median": float(rows["hvp_probe_trace_m_per_dim"].median()),
    }


def _vae_summary(label: str) -> dict[str, float]:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    last = rows.tail(1).iloc[0]
    out = {
        "val_recon_mse_final": float(last.get("val_recon_mse", np.nan)),
        "function_anchor_ce_delta_final": float(last.get("function_anchor_ce_delta", np.nan)),
        "function_anchor_loss_final": float(last.get("function_anchor_loss", np.nan)),
        "function_anchor_effective_loss_final": float(last.get("function_anchor_effective_loss", np.nan)),
    }
    numeric = rows.select_dtypes(include=[np.number])
    for col in [
        "function_anchor_ce_delta",
        "function_anchor_loss",
        "function_anchor_effective_loss",
        "train_precond_effective_loss",
        "train_precond_grad_norm",
        "train_precond_base_grad_norm",
    ]:
        out[f"{col}_median"] = float(numeric[col].median()) if col in numeric else float("nan")
    return out


def _geometry_summary(label: str) -> dict[str, float]:
    rows = pd.read_csv(_run_dir(label) / "geometry.csv")
    row = rows.iloc[0]
    return {
        "isometry_objective": float(row["isometry_objective"]),
        "trace_g_median": float(row["trace_g_median"]),
        "trace_g2_median": float(row["trace_g2_median"]),
        "log_eig_spread_median": float(row["log_eig_spread_median"]),
    }


def _assert_same_keys(label: str, control_label: str) -> None:
    rows = _load_eval_results(label)[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    control = _load_eval_results(control_label)[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    if not rows.equals(control):
        raise RuntimeError(f"{label} eval start keys do not match {control_label}")


def _build_summary(out_dir: Path) -> pd.DataFrame:
    rows = []
    for label in RUNS:
        decoder = _load_eval_results(label, "decoder_latent")
        raw = _load_eval_results(label, "raw")
        row = {
            "label": label,
            "run_label": RUNS[label],
            "n_eval_decoder": int(len(decoder)),
            "decoder_aulc_mean": float(decoder["aulc"].mean()),
            "decoder_aulc_median": float(decoder["aulc"].median()),
            "raw_aulc_mean": float(raw["aulc"].mean()),
            "raw_aulc_median": float(raw["aulc"].median()),
            "decoder_step0_test_loss_mean": float(_load_eval_curves(label).query("step == 0")["test_loss"].mean()),
            "decoder_step0_test_loss_median": float(_load_eval_curves(label).query("step == 0")["test_loss"].median()),
            "decoder_final_test_loss_mean": float(decoder["final_test_loss"].mean()),
            "decoder_final_test_loss_median": float(decoder["final_test_loss"].median()),
            "decoder_reconstruction_rel_l2_median": float(decoder["reconstruction_rel_l2"].median()),
            "selected_raw_lr": _selected_lr(label, "raw"),
            "selected_decoder_lr": _selected_lr(label, "decoder_latent"),
        }
        row.update(_li_summary(label))
        row.update(_vae_summary(label))
        row.update(_geometry_summary(label))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "anchor_fix_run_summary.csv", index=False)
    return summary


def _build_paired_tables(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for label, control_label in MATCHED_CONTROLS.items():
        _assert_same_keys(label, control_label)
        rows = _load_eval_results(label)
        control_rows = _load_eval_results(control_label)
        control_key = control_rows[
            KEY_COLS + ["aulc", "final_test_loss", "final_test_acc", "reconstruction_rel_l2"]
        ].rename(
            columns={
                "aulc": "matched_control_aulc",
                "final_test_loss": "matched_control_final_test_loss",
                "final_test_acc": "matched_control_final_test_acc",
                "reconstruction_rel_l2": "matched_control_reconstruction_rel_l2",
            }
        )
        merged = rows.merge(control_key, on=KEY_COLS, how="left", validate="one_to_one")
        if int(merged["matched_control_aulc"].notna().sum()) != int(len(merged)):
            raise RuntimeError(f"{label} result mismatch vs {control_label}")
        merged["matched_control_label"] = control_label
        merged["delta_aulc_vs_matched_control"] = merged["aulc"] - merged["matched_control_aulc"]
        merged["delta_final_test_loss_vs_matched_control"] = (
            merged["final_test_loss"] - merged["matched_control_final_test_loss"]
        )
        merged["delta_final_test_acc_vs_matched_control"] = (
            merged["final_test_acc"] - merged["matched_control_final_test_acc"]
        )
        merged["delta_reconstruction_rel_l2_vs_matched_control"] = (
            merged["reconstruction_rel_l2"] - merged["matched_control_reconstruction_rel_l2"]
        )
        result_frames.append(merged)

        curves = _load_eval_curves(label)
        control_curves = _load_eval_curves(control_label)
        control_curve_key = control_curves[CURVE_KEY_COLS + ["train_loss", "test_loss", "test_acc"]].rename(
            columns={
                "train_loss": "matched_control_train_loss",
                "test_loss": "matched_control_test_loss",
                "test_acc": "matched_control_test_acc",
            }
        )
        curve_merged = curves.merge(control_curve_key, on=CURVE_KEY_COLS, how="left", validate="one_to_one")
        if int(curve_merged["matched_control_test_loss"].notna().sum()) != int(len(curve_merged)):
            raise RuntimeError(f"{label} curve mismatch vs {control_label}")
        curve_merged["matched_control_label"] = control_label
        curve_merged["delta_test_loss_vs_matched_control"] = (
            curve_merged["test_loss"] - curve_merged["matched_control_test_loss"]
        )
        curve_merged["delta_train_loss_vs_matched_control"] = (
            curve_merged["train_loss"] - curve_merged["matched_control_train_loss"]
        )
        curve_merged["delta_test_acc_vs_matched_control"] = (
            curve_merged["test_acc"] - curve_merged["matched_control_test_acc"]
        )
        curve_frames.append(curve_merged)

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    curves = pd.concat(curve_frames, ignore_index=True, sort=False)
    step0 = curves[curves["step"] == 0][
        KEY_COLS + ["label", "delta_test_loss_vs_matched_control", "delta_train_loss_vs_matched_control"]
    ].rename(
        columns={
            "delta_test_loss_vs_matched_control": "delta_step0_test_loss_vs_matched_control",
            "delta_train_loss_vs_matched_control": "delta_step0_train_loss_vs_matched_control",
        }
    )
    results = results.merge(step0, on=KEY_COLS + ["label"], how="left", validate="one_to_one")
    curves = curves.merge(step0, on=KEY_COLS + ["label"], how="left", validate="many_to_one")
    curves["step0_gap_closed_test_loss"] = (
        curves["delta_step0_test_loss_vs_matched_control"] - curves["delta_test_loss_vs_matched_control"]
    )
    results.to_csv(out_dir / "anchor_fix_per_start_deltas.csv", index=False)
    curves.to_csv(out_dir / "anchor_fix_curve_deltas.csv", index=False)
    return results, curves


def _build_pair_summary(results: pd.DataFrame, curves: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for label, sub in results.groupby("label", sort=False):
        step0 = pd.to_numeric(sub["delta_step0_test_loss_vs_matched_control"], errors="coerce")
        aulc = pd.to_numeric(sub["delta_aulc_vs_matched_control"], errors="coerce")
        corr = float(step0.corr(aulc)) if len(sub) >= 3 else float("nan")
        curve_sub = curves[curves["label"] == label]
        by_step = curve_sub.groupby("step", as_index=False).agg(
            mean_delta_test_loss=("delta_test_loss_vs_matched_control", "mean"),
            median_delta_test_loss=("delta_test_loss_vs_matched_control", "median"),
            mean_step0_gap_closed=("step0_gap_closed_test_loss", "mean"),
        )
        row = {
            "label": label,
            "matched_control_label": str(sub["matched_control_label"].iloc[0]),
            "mean_delta_aulc": float(aulc.mean()),
            "median_delta_aulc": float(aulc.median()),
            "mean_delta_step0_test_loss": float(step0.mean()),
            "median_delta_step0_test_loss": float(step0.median()),
            "corr_step0_delta_vs_aulc_delta": corr,
            "worse_aulc_count": int((aulc > 0).sum()),
            "n_eval_starts": int(len(sub)),
            "mean_delta_final_test_loss": float(
                pd.to_numeric(sub["delta_final_test_loss_vs_matched_control"], errors="coerce").mean()
            ),
            "mean_delta_reconstruction_rel_l2": float(
                pd.to_numeric(sub["delta_reconstruction_rel_l2_vs_matched_control"], errors="coerce").mean()
            ),
        }
        for step in [0, 25, 50, 100, 200, 300]:
            hit = by_step[by_step["step"] == step]
            if hit.empty:
                row[f"mean_delta_test_loss_step{step}"] = float("nan")
                row[f"mean_step0_gap_closed_step{step}"] = float("nan")
            else:
                row[f"mean_delta_test_loss_step{step}"] = float(hit["mean_delta_test_loss"].iloc[0])
                row[f"mean_step0_gap_closed_step{step}"] = float(hit["mean_step0_gap_closed"].iloc[0])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "anchor_fix_pair_summary.csv", index=False)
    return summary


def _build_vs_original_tables(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    control_label = "original_control"
    control_results = _load_eval_results(control_label)
    control_curves = _load_eval_curves(control_label)
    control_result_key = control_results[
        KEY_COLS + ["aulc", "final_test_loss", "final_test_acc", "reconstruction_rel_l2"]
    ].rename(
        columns={
            "aulc": "original_control_aulc",
            "final_test_loss": "original_control_final_test_loss",
            "final_test_acc": "original_control_final_test_acc",
            "reconstruction_rel_l2": "original_control_reconstruction_rel_l2",
        }
    )
    control_curve_key = control_curves[CURVE_KEY_COLS + ["train_loss", "test_loss", "test_acc"]].rename(
        columns={
            "train_loss": "original_control_train_loss",
            "test_loss": "original_control_test_loss",
            "test_acc": "original_control_test_acc",
        }
    )
    result_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for label in VS_ORIGINAL_LABELS:
        _assert_same_keys(label, control_label)
        rows = _load_eval_results(label).merge(control_result_key, on=KEY_COLS, how="left", validate="one_to_one")
        if int(rows["original_control_aulc"].notna().sum()) != int(len(rows)):
            raise RuntimeError(f"{label} result mismatch vs original_control")
        rows["delta_aulc_vs_original_control"] = rows["aulc"] - rows["original_control_aulc"]
        rows["delta_final_test_loss_vs_original_control"] = rows["final_test_loss"] - rows["original_control_final_test_loss"]
        rows["delta_final_test_acc_vs_original_control"] = rows["final_test_acc"] - rows["original_control_final_test_acc"]
        rows["delta_reconstruction_rel_l2_vs_original_control"] = (
            rows["reconstruction_rel_l2"] - rows["original_control_reconstruction_rel_l2"]
        )
        result_frames.append(rows)

        curves = _load_eval_curves(label).merge(control_curve_key, on=CURVE_KEY_COLS, how="left", validate="one_to_one")
        if int(curves["original_control_test_loss"].notna().sum()) != int(len(curves)):
            raise RuntimeError(f"{label} curve mismatch vs original_control")
        curves["delta_test_loss_vs_original_control"] = curves["test_loss"] - curves["original_control_test_loss"]
        curves["delta_train_loss_vs_original_control"] = curves["train_loss"] - curves["original_control_train_loss"]
        curves["delta_test_acc_vs_original_control"] = curves["test_acc"] - curves["original_control_test_acc"]
        curve_frames.append(curves)

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    curves = pd.concat(curve_frames, ignore_index=True, sort=False)
    step0 = curves[curves["step"] == 0][
        KEY_COLS + ["label", "delta_test_loss_vs_original_control", "delta_train_loss_vs_original_control"]
    ].rename(
        columns={
            "delta_test_loss_vs_original_control": "delta_step0_test_loss_vs_original_control",
            "delta_train_loss_vs_original_control": "delta_step0_train_loss_vs_original_control",
        }
    )
    results = results.merge(step0, on=KEY_COLS + ["label"], how="left", validate="one_to_one")
    curves = curves.merge(step0, on=KEY_COLS + ["label"], how="left", validate="many_to_one")
    curves["step0_gap_closed_test_loss_vs_original_control"] = (
        curves["delta_step0_test_loss_vs_original_control"] - curves["delta_test_loss_vs_original_control"]
    )
    summary_rows = []
    for label, sub in results.groupby("label", sort=False):
        step0_delta = pd.to_numeric(sub["delta_step0_test_loss_vs_original_control"], errors="coerce")
        aulc_delta = pd.to_numeric(sub["delta_aulc_vs_original_control"], errors="coerce")
        curve_sub = curves[curves["label"] == label]
        by_step = curve_sub.groupby("step", as_index=False).agg(
            mean_delta_test_loss=("delta_test_loss_vs_original_control", "mean"),
            mean_step0_gap_closed=("step0_gap_closed_test_loss_vs_original_control", "mean"),
        )
        row = {
            "label": label,
            "mean_delta_aulc_vs_original_control": float(aulc_delta.mean()),
            "median_delta_aulc_vs_original_control": float(aulc_delta.median()),
            "mean_delta_step0_test_loss_vs_original_control": float(step0_delta.mean()),
            "median_delta_step0_test_loss_vs_original_control": float(step0_delta.median()),
            "corr_step0_delta_vs_aulc_delta": float(step0_delta.corr(aulc_delta)) if len(sub) >= 3 else float("nan"),
            "worse_aulc_count": int((aulc_delta > 0).sum()),
            "n_eval_starts": int(len(sub)),
            "mean_delta_final_test_loss_vs_original_control": float(
                pd.to_numeric(sub["delta_final_test_loss_vs_original_control"], errors="coerce").mean()
            ),
            "mean_delta_reconstruction_rel_l2_vs_original_control": float(
                pd.to_numeric(sub["delta_reconstruction_rel_l2_vs_original_control"], errors="coerce").mean()
            ),
        }
        for step in [0, 25, 50, 100, 200, 300]:
            hit = by_step[by_step["step"] == step]
            row[f"mean_delta_test_loss_step{step}_vs_original_control"] = (
                float(hit["mean_delta_test_loss"].iloc[0]) if not hit.empty else float("nan")
            )
            row[f"mean_step0_gap_closed_step{step}_vs_original_control"] = (
                float(hit["mean_step0_gap_closed"].iloc[0]) if not hit.empty else float("nan")
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    results.to_csv(out_dir / "anchor_fix_vs_original_control_per_start.csv", index=False)
    curves.to_csv(out_dir / "anchor_fix_vs_original_control_curves.csv", index=False)
    summary.to_csv(out_dir / "anchor_fix_vs_original_control_summary.csv", index=False)
    return results, curves, summary


def _plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    order = list(RUNS)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), constrained_layout=True)
    panels = [
        ("decoder_aulc_median", "Decoder AULC median", False),
        ("li_A_full_per_dim_p95", "Exact li_A / dim p95", True),
        ("decoder_reconstruction_rel_l2_median", "Recon rel-L2 median", False),
        ("isometry_objective", "Latent isometry objective", True),
    ]
    for ax, (col, title, logy) in zip(axes.ravel(), panels, strict=True):
        vals = [float(summary.loc[summary["label"] == label, col].iloc[0]) for label in order]
        ax.bar(range(len(order)), vals, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=18, ha="right", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if logy:
            ax.set_yscale("log")
        for idx, value in enumerate(vals):
            ax.text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_dir / "anchor_fix_summary_bars.png", dpi=190)
    plt.close(fig)


def _plot_gap_closure(pair_summary: pd.DataFrame, curves: pd.DataFrame, out_dir: Path) -> None:
    labels = ["original_A_cap1", "anchor_A_cap1_c1e-3", "anchor_control_c1e-3"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(labels):
        rows = curves[curves["label"] == label]
        if rows.empty:
            continue
        grouped = rows.groupby("step", as_index=False).agg(
            mean_delta=("delta_test_loss_vs_matched_control", "mean"),
            sem_delta=(
                "delta_test_loss_vs_matched_control",
                lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=1) / np.sqrt(max(1, s.notna().sum()))),
            ),
            mean_closed=("step0_gap_closed_test_loss", "mean"),
        )
        color = cmap(idx)
        axes[0].plot(grouped["step"], grouped["mean_delta"], marker="o", markersize=3, label=label, color=color)
        lo = grouped["mean_delta"] - grouped["sem_delta"]
        hi = grouped["mean_delta"] + grouped["sem_delta"]
        axes[0].fill_between(grouped["step"].to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), color=color, alpha=0.13)
        axes[1].plot(grouped["step"], grouped["mean_closed"], marker="o", markersize=3, label=label, color=color)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("downstream step")
    axes[0].set_ylabel("mean test-loss delta vs matched control")
    axes[0].set_title("Matched-control loss gap")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("downstream step")
    axes[1].set_ylabel("mean step0 gap closed")
    axes[1].set_title("Post-step0 catch-up")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.savefig(out_dir / "anchor_fix_gap_closure.png", dpi=190)
    plt.close(fig)


def _plot_step0_scatter(results: pd.DataFrame, out_dir: Path) -> None:
    labels = ["original_A_cap1", "anchor_A_cap1_c1e-3", "anchor_control_c1e-3"]
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(labels):
        rows = results[results["label"] == label].copy()
        ax.scatter(
            rows["delta_step0_test_loss_vs_matched_control"],
            rows["delta_aulc_vs_matched_control"],
            s=60,
            alpha=0.82,
            label=label,
            color=cmap(idx),
        )
        for _, row in rows.nlargest(3, "delta_aulc_vs_matched_control").iterrows():
            ax.annotate(
                str(int(row["source_weight_index"])),
                (
                    float(row["delta_step0_test_loss_vs_matched_control"]),
                    float(row["delta_aulc_vs_matched_control"]),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("decoded step0 test-loss delta vs matched control")
    ax.set_ylabel("decoder AULC delta vs matched control")
    ax.set_title("Step0 damage predicts downstream AULC delta")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "anchor_fix_step0_vs_aulc.png", dpi=190)
    plt.close(fig)


def _plot_tail_starts(curves: pd.DataFrame, out_dir: Path) -> None:
    score = curves[(curves["label"] == "anchor_A_cap1_c1e-3") & (curves["step"] == 0)].copy()
    source_indices = score.nlargest(4, "delta_test_loss_vs_matched_control")["source_weight_index"].astype(int).tolist()
    labels = ["original_control", "original_A_cap1", "anchor_control_c1e-3", "anchor_A_cap1_c1e-3"]
    ncols = 2
    nrows = int(np.ceil(len(source_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows), constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    all_curves = []
    for label in labels:
        all_curves.append(_load_eval_curves(label).assign(label=label))
    all_curves_df = pd.concat(all_curves, ignore_index=True)
    for ax, source_idx in zip(axes.ravel(), source_indices, strict=False):
        for idx, label in enumerate(labels):
            rows = all_curves_df[
                (all_curves_df["label"] == label)
                & (all_curves_df["source_weight_index"].astype(int) == int(source_idx))
            ].sort_values("step")
            if rows.empty:
                continue
            task = str(rows["task_name"].iloc[0])
            ax.plot(rows["step"], rows["test_loss"], marker="o", markersize=3, label=label, color=cmap(idx))
            ax.set_title(f"source_weight_index={source_idx}, task={task}")
        ax.set_xlabel("downstream step")
        ax.set_ylabel("decoder test loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(source_indices) :]:
        ax.axis("off")
    fig.savefig(out_dir / "anchor_fix_tail_start_curves.png", dpi=190)
    plt.close(fig)


def _plot_vs_original(vs_summary: pd.DataFrame, vs_curves: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(VS_ORIGINAL_LABELS):
        rows = vs_curves[vs_curves["label"] == label]
        grouped = rows.groupby("step", as_index=False).agg(
            mean_delta=("delta_test_loss_vs_original_control", "mean"),
            sem_delta=(
                "delta_test_loss_vs_original_control",
                lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=1) / np.sqrt(max(1, s.notna().sum()))),
            ),
            mean_closed=("step0_gap_closed_test_loss_vs_original_control", "mean"),
        )
        color = cmap(idx)
        axes[0].plot(grouped["step"], grouped["mean_delta"], marker="o", markersize=3, label=label, color=color)
        lo = grouped["mean_delta"] - grouped["sem_delta"]
        hi = grouped["mean_delta"] + grouped["sem_delta"]
        axes[0].fill_between(grouped["step"].to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), color=color, alpha=0.13)
        axes[1].plot(grouped["step"], grouped["mean_closed"], marker="o", markersize=3, label=label, color=color)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("downstream step")
    axes[0].set_ylabel("mean test-loss delta vs original control")
    axes[0].set_title("Absolute degradation vs no-anchor control")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("downstream step")
    axes[1].set_ylabel("mean step0 gap closed vs original control")
    axes[1].set_title("Catch-up from absolute step0 damage")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.savefig(out_dir / "anchor_fix_vs_original_control_gap.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(VS_ORIGINAL_LABELS))
    step0 = [
        float(vs_summary.loc[vs_summary["label"] == label, "mean_delta_step0_test_loss_vs_original_control"].iloc[0])
        for label in VS_ORIGINAL_LABELS
    ]
    aulc = [
        float(vs_summary.loc[vs_summary["label"] == label, "mean_delta_aulc_vs_original_control"].iloc[0])
        for label in VS_ORIGINAL_LABELS
    ]
    axes[0].bar(x, step0, color=["#f58518", "#54a24b", "#e45756"])
    axes[1].bar(x, aulc, color=["#f58518", "#54a24b", "#e45756"])
    for ax, vals, title, ylabel in [
        (axes[0], step0, "Step0 decoded test-loss delta", "mean delta vs original control"),
        (axes[1], aulc, "Decoder AULC delta", "mean delta vs original control"),
    ]:
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(VS_ORIGINAL_LABELS, rotation=18, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(vals):
            va = "bottom" if value >= 0 else "top"
            ax.text(idx, value, f"{value:.3g}", ha="center", va=va, fontsize=8)
    fig.savefig(out_dir / "anchor_fix_vs_original_control_bars.png", dpi=190)
    plt.close(fig)


def main() -> None:
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for label in RUNS:
        _check_complete(label)
    summary = _build_summary(out_dir)
    results, curves = _build_paired_tables(out_dir)
    pair_summary = _build_pair_summary(results, curves, out_dir)
    _, vs_curves, vs_summary = _build_vs_original_tables(out_dir)
    _plot_summary_bars(summary, out_dir)
    _plot_gap_closure(pair_summary, curves, out_dir)
    _plot_step0_scatter(results, out_dir)
    _plot_tail_starts(curves, out_dir)
    _plot_vs_original(vs_summary, vs_curves, out_dir)
    print(f"[analyze_variant_a_anchor_fix] wrote {out_dir}")
    print(pair_summary.to_string(index=False))
    print(vs_summary.to_string(index=False))


if __name__ == "__main__":
    main()
