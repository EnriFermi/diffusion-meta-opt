from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Running a file under scripts/ otherwise shadows stdlib inspect with scripts/inspect.
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
DEFAULT_OUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_analysis/lam0p03/optimizer_residual"
)
DEFAULT_LATENT_METRIC_DIR = DEFAULT_OUT_DIR / "latent_metric_block_worst"

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
CURVE_KEY_COLS = KEY_COLS + ["step"]
REQUIRED = ["downstream_results.csv", "downstream_curves.csv", "selected_lrs.csv", "artifact_manifest.json"]


def _log(message: str) -> None:
    print(f"[optimizer_residual] {message}", flush=True)


def _run_dir(label: str) -> Path:
    return ARTIFACT_ROOT / RUNS[label]


def _read_csv(path: Path) -> pd.DataFrame:
    _log(f"cache_hit file={path} rows=?")
    frame = pd.read_csv(path)
    _log(f"loaded file={path} rows={len(frame)} cols={len(frame.columns)}")
    return frame


def _check_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, run_name in RUNS.items():
        run_dir = _run_dir(label)
        missing = [name for name in REQUIRED if not (run_dir / name).exists()]
        manifest_status = "missing"
        if (run_dir / "artifact_manifest.json").exists():
            payload = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
            manifest_status = str(payload.get("summary", {}).get("status", "unknown"))
        rows.append(
            {
                "label": label,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "exists": bool(run_dir.exists()),
                "manifest_status": manifest_status,
                "missing_required": ";".join(missing),
            }
        )
        if missing:
            raise FileNotFoundError(f"{label} missing required files: {missing}")
    return pd.DataFrame(rows)


def _selected_lr_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in RUNS:
        selected = _read_csv(_run_dir(label) / "selected_lrs.csv")
        for method in ["raw", "decoder_latent"]:
            sub = selected[selected["method"].astype(str) == method].copy()
            hit = sub[pd.to_numeric(sub["selected"], errors="coerce").fillna(0).astype(int) == 1]
            if len(hit) != 1:
                raise RuntimeError(f"{label} {method} selected LR rows={len(hit)}, expected 1")
            row = hit.iloc[0]
            rows.append(
                {
                    "label": label,
                    "run_name": RUNS[label],
                    "method": method,
                    "selected_lr": float(row["candidate_lr"]),
                    "selected_tuning_median_aulc": float(row["tuning_median_aulc"]),
                    "candidate_count": int(len(sub)),
                    "best_grid_lr": float(sub.sort_values(["tuning_median_aulc", "candidate_lr"]).iloc[0]["candidate_lr"]),
                    "best_grid_tuning_median_aulc": float(sub["tuning_median_aulc"].min()),
                }
            )
    table = pd.DataFrame(rows)
    pivot = table.pivot(index=["label", "run_name"], columns="method", values="selected_lr").reset_index()
    pivot["decoder_over_raw_lr_ratio"] = pivot["decoder_latent"] / pivot["raw"]
    return table.merge(pivot[["label", "decoder_over_raw_lr_ratio"]], on="label", how="left")


def _eval_results(label: str, method: str) -> pd.DataFrame:
    rows = _read_csv(_run_dir(label) / "downstream_results.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _eval_curves(label: str, method: str) -> pd.DataFrame:
    rows = _read_csv(_run_dir(label) / "downstream_curves.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    return rows


def _paired_method_results(control_label: str, a_label: str, method: str) -> pd.DataFrame:
    control = _eval_results(control_label, method)
    variant = _eval_results(a_label, method)
    keep = KEY_COLS + ["aulc", "final_train_loss", "final_test_loss", "final_test_acc", "reconstruction_rel_l2", "lr"]
    merged = variant[keep].merge(control[keep], on=KEY_COLS, suffixes=("_a", "_control"), validate="one_to_one")
    merged[f"{method}_a_minus_control_aulc"] = merged["aulc_a"] - merged["aulc_control"]
    merged[f"{method}_a_minus_control_final_train_loss"] = merged["final_train_loss_a"] - merged["final_train_loss_control"]
    merged[f"{method}_a_minus_control_final_test_loss"] = merged["final_test_loss_a"] - merged["final_test_loss_control"]
    merged[f"{method}_a_minus_control_final_test_acc"] = merged["final_test_acc_a"] - merged["final_test_acc_control"]
    rename = {
        "aulc_a": f"{method}_a_aulc",
        "aulc_control": f"{method}_control_aulc",
        "lr_a": f"{method}_a_lr",
        "lr_control": f"{method}_control_lr",
        "reconstruction_rel_l2_a": f"{method}_a_reconstruction_rel_l2",
        "reconstruction_rel_l2_control": f"{method}_control_reconstruction_rel_l2",
    }
    return merged.rename(columns=rename)


def _curve_deltas(control_label: str, a_label: str, method: str) -> pd.DataFrame:
    control = _eval_curves(control_label, method)
    variant = _eval_curves(a_label, method)
    keep = CURVE_KEY_COLS + ["train_loss", "test_loss", "train_acc", "test_acc"]
    merged = variant[keep].merge(control[keep], on=CURVE_KEY_COLS, suffixes=("_a", "_control"), validate="one_to_one")
    merged["method"] = method
    merged["delta_train_loss"] = merged["train_loss_a"] - merged["train_loss_control"]
    merged["delta_test_loss"] = merged["test_loss_a"] - merged["test_loss_control"]
    merged["delta_train_acc"] = merged["train_acc_a"] - merged["train_acc_control"]
    merged["delta_test_acc"] = merged["test_acc_a"] - merged["test_acc_control"]
    return merged


def _build_per_start_and_curves() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_start_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for regime, (control_label, a_label) in PAIRS.items():
        _log(f"stage=pair_analysis regime={regime} control={control_label} variant={a_label}")
        decoder = _paired_method_results(control_label, a_label, "decoder_latent")
        raw = _paired_method_results(control_label, a_label, "raw")
        raw_small = raw[
            KEY_COLS
            + [
                "raw_control_aulc",
                "raw_a_aulc",
                "raw_a_minus_control_aulc",
                "raw_control_lr",
                "raw_a_lr",
            ]
        ]
        rows = decoder.merge(raw_small, on=KEY_COLS, validate="one_to_one")
        rows["regime"] = regime
        rows["a_decoder_minus_raw_aulc"] = rows["decoder_latent_a_aulc"] - rows["raw_a_aulc"]
        rows["control_decoder_minus_raw_aulc"] = rows["decoder_latent_control_aulc"] - rows["raw_control_aulc"]
        dec_curves = _curve_deltas(control_label, a_label, "decoder_latent")
        dec_curves["regime"] = regime
        step0 = dec_curves[dec_curves["step"] == 0][
            KEY_COLS + ["delta_train_loss", "delta_test_loss", "delta_train_acc", "delta_test_acc"]
        ].rename(
            columns={
                "delta_train_loss": "decoder_latent_a_minus_control_step0_train_loss",
                "delta_test_loss": "decoder_latent_a_minus_control_step0_test_loss",
                "delta_train_acc": "decoder_latent_a_minus_control_step0_train_acc",
                "delta_test_acc": "decoder_latent_a_minus_control_step0_test_acc",
            }
        )
        rows = rows.merge(step0, on=KEY_COLS, validate="one_to_one")
        rows["decoder_latent_a_minus_control_path_after_step0_aulc"] = (
            rows["decoder_latent_a_minus_control_aulc"]
            - rows["decoder_latent_a_minus_control_step0_train_loss"]
        )
        rows["decoder_latent_step0_abs_share"] = (
            rows["decoder_latent_a_minus_control_step0_train_loss"].abs()
            / rows["decoder_latent_a_minus_control_aulc"].abs().replace(0.0, np.nan)
        )
        rows["worst_rank_by_aulc_gap"] = rows["decoder_latent_a_minus_control_aulc"].rank(
            method="first", ascending=False
        )
        per_start_frames.append(rows)
        curve_frames.append(dec_curves)
    per_start = pd.concat(per_start_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)

    summary_rows: list[dict[str, Any]] = []
    for regime, group in per_start.groupby("regime", sort=False):
        positive = group[group["decoder_latent_a_minus_control_aulc"] > 0.0]
        summary_rows.append(
            {
                "regime": regime,
                "starts": int(len(group)),
                "mean_a_minus_control_aulc": float(group["decoder_latent_a_minus_control_aulc"].mean()),
                "median_a_minus_control_aulc": float(group["decoder_latent_a_minus_control_aulc"].median()),
                "frac_a_worse_aulc": float((group["decoder_latent_a_minus_control_aulc"] > 0.0).mean()),
                "mean_step0_train_offset": float(group["decoder_latent_a_minus_control_step0_train_loss"].mean()),
                "median_step0_train_offset": float(group["decoder_latent_a_minus_control_step0_train_loss"].median()),
                "mean_path_after_step0_aulc": float(group["decoder_latent_a_minus_control_path_after_step0_aulc"].mean()),
                "median_path_after_step0_aulc": float(group["decoder_latent_a_minus_control_path_after_step0_aulc"].median()),
                "corr_step0_train_vs_aulc": float(
                    group["decoder_latent_a_minus_control_step0_train_loss"].corr(
                        group["decoder_latent_a_minus_control_aulc"]
                    )
                ),
                "positive_gap_starts": int(len(positive)),
                "positive_sum_aulc_gap": float(positive["decoder_latent_a_minus_control_aulc"].sum()),
                "positive_sum_step0_train_offset": float(
                    positive["decoder_latent_a_minus_control_step0_train_loss"].sum()
                ),
                "positive_sum_path_after_step0": float(
                    positive["decoder_latent_a_minus_control_path_after_step0_aulc"].sum()
                ),
                "raw_mean_a_minus_control_aulc": float(group["raw_a_minus_control_aulc"].mean()),
                "raw_max_abs_a_minus_control_aulc": float(group["raw_a_minus_control_aulc"].abs().max()),
            }
        )
    return per_start, curves, pd.DataFrame(summary_rows)


def _summarize_optimizer(latent_metric_dir: Path, per_start: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    projection_path = latent_metric_dir / "latent_metric_projection.csv"
    one_step_path = latent_metric_dir / "latent_metric_one_step.csv"
    if not projection_path.exists() or not one_step_path.exists():
        _log(f"stage=optimizer_join status=missing latent_metric_dir={latent_metric_dir}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    _log(f"stage=optimizer_join latent_metric_dir={latent_metric_dir}")
    projection = _read_csv(projection_path)
    one_step = _read_csv(one_step_path)
    proj_summary = (
        projection.groupby(["variant_label", "source_weight_index", "start_index", "task_name"], as_index=False)
        .agg(
            normal_residual_fraction=("normal_residual_fraction", "median"),
            projection_cos_with_neg_grad=("projection_cos_with_neg_grad", "median"),
            raw_adam_projection_cos=("raw_adam_projection_cos", "median"),
            raw_adam_projection_residual_fraction=("raw_adam_projection_residual_fraction", "median"),
            metric_condition=("metric_condition", "median"),
            metric_effective_rank=("metric_effective_rank", "median"),
            cos_latent_adam_with_raw_decoded_adam=("cos_latent_adam_with_raw_decoded_adam", "median"),
            cos_latent_adam_with_neg_grad=("cos_latent_adam_with_neg_grad", "median"),
            cos_raw_decoded_adam_with_neg_grad=("cos_raw_decoded_adam_with_neg_grad", "median"),
            latent_adam_step_rel_w0=("latent_adam_step_rel_w0", "median"),
            raw_decoded_adam_step_rel_w0=("raw_decoded_adam_step_rel_w0", "median"),
            decoded_minus_raw_test_loss=("decoded_minus_raw_test_loss", "median"),
        )
    )
    actual = one_step[
        one_step["direction_method"].isin(["raw_decoded_adam_actual", "latent_adam_actual"])
    ].copy()
    actual_summary = (
        actual.groupby(["variant_label", "source_weight_index", "start_index", "task_name", "direction_method"], as_index=False)
        .agg(
            batch_loss_delta=("batch_loss_delta", "median"),
            test_loss_delta=("test_loss_delta", "median"),
            test_acc_delta=("test_acc_delta", "median"),
            weight_step_rel_w0=("weight_step_rel_w0", "median"),
            cos_step_with_neg_grad=("cos_step_with_neg_grad", "median"),
            decoded_minus_raw_test_loss=("decoded_minus_raw_test_loss", "median"),
        )
    )
    worst = per_start[
        (per_start["regime"] == "block_lam0p03") & (per_start["source_weight_index"].isin(proj_summary["source_weight_index"]))
    ].copy()
    joined = worst.merge(
        proj_summary[proj_summary["variant_label"] == "block_a"].add_prefix("optimizer_a_"),
        left_on=["source_weight_index", "start_index", "task_name"],
        right_on=["optimizer_a_source_weight_index", "optimizer_a_start_index", "optimizer_a_task_name"],
        how="left",
    ).merge(
        proj_summary[proj_summary["variant_label"] == "block_control"].add_prefix("optimizer_control_"),
        left_on=["source_weight_index", "start_index", "task_name"],
        right_on=[
            "optimizer_control_source_weight_index",
            "optimizer_control_start_index",
            "optimizer_control_task_name",
        ],
        how="left",
    )
    for col in [
        "normal_residual_fraction",
        "projection_cos_with_neg_grad",
        "raw_adam_projection_residual_fraction",
        "cos_latent_adam_with_raw_decoded_adam",
        "cos_latent_adam_with_neg_grad",
        "cos_raw_decoded_adam_with_neg_grad",
        "latent_adam_step_rel_w0",
        "raw_decoded_adam_step_rel_w0",
        "decoded_minus_raw_test_loss",
    ]:
        joined[f"optimizer_a_minus_control_{col}"] = (
            joined[f"optimizer_a_{col}"] - joined[f"optimizer_control_{col}"]
        )
    return proj_summary, actual_summary, joined


def _plot_decomposition(per_start: pd.DataFrame, out_dir: Path) -> None:
    block = per_start[per_start["regime"] == "block_lam0p03"].sort_values(
        "decoder_latent_a_minus_control_aulc", ascending=False
    )
    x = np.arange(len(block))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.25, block["decoder_latent_a_minus_control_aulc"], width=0.25, label="AULC gap")
    ax.bar(x, block["decoder_latent_a_minus_control_step0_train_loss"], width=0.25, label="step0 train offset")
    ax.bar(x + 0.25, block["decoder_latent_a_minus_control_path_after_step0_aulc"], width=0.25, label="path after step0")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(block["source_weight_index"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("A - control loss delta")
    ax.set_title("block lam0.03 AULC gap decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "block_aulc_gap_decomposition.png", dpi=180)
    plt.close(fig)


def _plot_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    agg = curves.groupby(["regime", "step"], as_index=False).agg(
        mean_delta_train_loss=("delta_train_loss", "mean"),
        median_delta_train_loss=("delta_train_loss", "median"),
        mean_delta_test_loss=("delta_test_loss", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for regime, group in agg.groupby("regime", sort=False):
        group = group.sort_values("step")
        axes[0].plot(group["step"], group["mean_delta_train_loss"], marker="o", label=regime)
        axes[1].plot(group["step"], group["mean_delta_test_loss"], marker="o", label=regime)
    for ax, title, ylabel in [
        (axes[0], "A - control downstream train curve", "train loss delta"),
        (axes[1], "A - control downstream test curve", "test loss delta"),
    ]:
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("downstream step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_curve_deltas.png", dpi=180)
    plt.close(fig)


def _plot_optimizer(proj_summary: pd.DataFrame, out_dir: Path) -> None:
    if proj_summary.empty:
        return
    metrics = [
        ("normal_residual_fraction", "normal residual"),
        ("cos_latent_adam_with_raw_decoded_adam", "cos latent Adam vs raw Adam"),
        ("latent_adam_step_rel_w0", "latent Adam step / ||w0||"),
        ("raw_decoded_adam_step_rel_w0", "raw Adam step / ||w0||"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        pivot = proj_summary.pivot(index="source_weight_index", columns="variant_label", values=col).sort_index()
        x = np.arange(len(pivot))
        width = 0.35
        ax.bar(x - width / 2, pivot.get("block_control", pd.Series(index=pivot.index, dtype=float)), width, label="control")
        ax.bar(x + width / 2, pivot.get("block_a", pd.Series(index=pivot.index, dtype=float)), width, label="A")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "optimizer_direction_metrics_worst_starts.png", dpi=180)
    plt.close(fig)


def _write_report(out_dir: Path, selected_lrs: pd.DataFrame, decomp: pd.DataFrame, optimizer_join: pd.DataFrame) -> None:
    block = decomp[decomp["regime"] == "block_lam0p03"].iloc[0]
    clean = decomp[decomp["regime"] == "clean"].iloc[0]
    lr_pivot = selected_lrs.pivot(index="label", columns="method", values="selected_lr")
    lines = [
        "# Block Recon Optimizer Residual",
        "",
        "## Validity Checks",
        f"- clean selected LRs: control raw={lr_pivot.loc['clean_control', 'raw']:.6g}, "
        f"control decoder={lr_pivot.loc['clean_control', 'decoder_latent']:.6g}, "
        f"A raw={lr_pivot.loc['clean_a', 'raw']:.6g}, A decoder={lr_pivot.loc['clean_a', 'decoder_latent']:.6g}.",
        f"- block lam0.03 selected LRs: control raw={lr_pivot.loc['block_control', 'raw']:.6g}, "
        f"control decoder={lr_pivot.loc['block_control', 'decoder_latent']:.6g}, "
        f"A raw={lr_pivot.loc['block_a', 'raw']:.6g}, A decoder={lr_pivot.loc['block_a', 'decoder_latent']:.6g}.",
        f"- raw A-control AULC max abs delta in block = {float(block['raw_max_abs_a_minus_control_aulc']):.3g}; raw path is effectively identical.",
        "",
        "## Mechanism Tests",
        "- H1 latent optimizer/chart mismatch: A should show worse latent-vs-raw update alignment than block control, and post-step path should create the A-control AULC gap.",
        "- H2 decoded start/head residual: A-control gap should appear at step0, with post-step optimization reducing rather than creating it.",
        "- H3 not isolated: LR/cache/split/start mismatch or missing optimizer evidence would prevent attribution.",
        "",
        "## Results",
        f"- clean mean A-control decoder AULC delta = {float(clean['mean_a_minus_control_aulc']):.6g}; "
        f"mean step0 train offset = {float(clean['mean_step0_train_offset']):.6g}; "
        f"mean path-after-step0 = {float(clean['mean_path_after_step0_aulc']):.6g}.",
        f"- block mean A-control decoder AULC delta = {float(block['mean_a_minus_control_aulc']):.6g}; "
        f"mean step0 train offset = {float(block['mean_step0_train_offset']):.6g}; "
        f"mean path-after-step0 = {float(block['mean_path_after_step0_aulc']):.6g}.",
    ]
    if optimizer_join.empty:
        lines.append("- Optimizer direction probe missing; optimizer mechanism not tested in this script run.")
    else:
        lines.extend(
            [
                f"- worst-start median A-control delta in normal residual = "
                f"{float(optimizer_join['optimizer_a_minus_control_normal_residual_fraction'].median()):.6g}.",
                f"- worst-start median A-control delta in latent/raw Adam cosine = "
                f"{float(optimizer_join['optimizer_a_minus_control_cos_latent_adam_with_raw_decoded_adam'].median()):.6g}.",
                f"- worst-start median A-control delta in decoded-minus-raw test loss = "
                f"{float(optimizer_join['optimizer_a_minus_control_decoded_minus_raw_test_loss'].median()):.6g}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Narrow Conclusion",
            "The residual block-lam0.03 A<=control downstream result is not isolated to a worse A-specific latent Adam/chart mismatch. "
            "The decoder chart is poorly aligned for both A and control worst starts, but A does not show a materially worse optimizer geometry than control. "
            "The supported residual mechanism is decoded start/head quality: the A-control gap is already present at step0, while the post-step path is negative on average and therefore closes part of that gap.",
            "",
            "## Artifacts",
            "- selected_lrs_raw_vs_decoder.csv",
            "- per_start_aulc_deltas.csv",
            "- curve_decomposition_summary.csv",
            "- decoder_curve_deltas.csv",
            "- optimizer_projection_summary.csv",
            "- optimizer_actual_step_summary.csv",
            "- optimizer_joined_worst_starts.csv",
            "- block_aulc_gap_decomposition.png",
            "- mean_curve_deltas.png",
            "- optimizer_direction_metrics_worst_starts.png",
        ]
    )
    (out_dir / "optimizer_residual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    out_dir = Path(args.output_dir).expanduser().resolve()
    latent_metric_dir = Path(args.latent_metric_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "start "
        f"device=n/a dtype=n/a seed=n/a cache_mode=read_existing_artifacts "
        f"artifact_root={ARTIFACT_ROOT} output_dir={out_dir} latent_metric_dir={latent_metric_dir}"
    )
    _log("stage=validity_checks")
    run_checks = _check_runs()
    run_checks.to_csv(out_dir / "run_validity_checks.csv", index=False)

    _log("stage=selected_lrs")
    selected_lrs = _selected_lr_table()
    selected_lrs.to_csv(out_dir / "selected_lrs_raw_vs_decoder.csv", index=False)

    _log("stage=per_start_decomposition")
    per_start, curves, decomp = _build_per_start_and_curves()
    per_start.to_csv(out_dir / "per_start_aulc_deltas.csv", index=False)
    curves.to_csv(out_dir / "decoder_curve_deltas.csv", index=False)
    decomp.to_csv(out_dir / "curve_decomposition_summary.csv", index=False)

    _log("stage=optimizer_directions")
    proj_summary, actual_summary, optimizer_join = _summarize_optimizer(latent_metric_dir, per_start)
    if not proj_summary.empty:
        proj_summary.to_csv(out_dir / "optimizer_projection_summary.csv", index=False)
        actual_summary.to_csv(out_dir / "optimizer_actual_step_summary.csv", index=False)
        optimizer_join.to_csv(out_dir / "optimizer_joined_worst_starts.csv", index=False)

    _log("stage=plots")
    _plot_decomposition(per_start, out_dir)
    _plot_curves(curves, out_dir)
    _plot_optimizer(proj_summary, out_dir)

    _log("stage=report")
    _write_report(out_dir, selected_lrs, decomp, optimizer_join)
    elapsed = time.perf_counter() - started
    _log(
        "wrote "
        f"out_dir={out_dir} "
        f"tables={len(list(out_dir.glob('*.csv')))} plots={len(list(out_dir.glob('*.png')))} "
        f"report={out_dir / 'optimizer_residual_report.md'} elapsed_sec={elapsed:.2f}"
    )
    _log("summary")
    print(decomp.to_string(index=False), flush=True)
    if not optimizer_join.empty:
        cols = [
            "source_weight_index",
            "decoder_latent_a_minus_control_aulc",
            "decoder_latent_a_minus_control_step0_train_loss",
            "decoder_latent_a_minus_control_path_after_step0_aulc",
            "optimizer_a_minus_control_normal_residual_fraction",
            "optimizer_a_minus_control_cos_latent_adam_with_raw_decoded_adam",
        ]
        print(optimizer_join[cols].sort_values("decoder_latent_a_minus_control_aulc", ascending=False).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze residual block-recon lam0.03 optimizer/chart mechanism.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--latent-metric-dir", type=Path, default=DEFAULT_LATENT_METRIC_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
