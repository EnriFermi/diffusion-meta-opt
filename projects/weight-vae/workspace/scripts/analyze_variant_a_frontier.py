from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import ExperimentConfig
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import _all_outputs_exist


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
DEFAULT_OUT = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/frontier_analysis"
BLOCK_SPLICE_DIR = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_splice_fixed_nocap_full16"


BASE_RUNS: dict[str, str] = {
    "control": "sage_cnn_vae_smoothing_celo_meta_finetune_control_current_v1_seed0",
    "old_control_stale": "sage_cnn_vae_smoothing_celo_meta_finetune_control_v1_seed0",
    "fixed_cap_a001": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_gradratio025_v1_seed0",
    "fixed_cap_a005": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_gradratio025_alpha005_v1_seed0",
    "local_nocap_a001": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_nogradcap_alpha001_ablation_v1_seed0",
    "old_global_a001": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_finetune_global_coeff001_v2_seed0",
    "old_global_a005": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_finetune_global_v1_seed0",
}


def _short_label(label: str) -> str:
    fixed = {
        "control": "control",
        "fixed_cap_a001": "fixed\na=.01\ncap=.25",
        "fixed_cap_a005": "fixed\na=.05\ncap=.25",
        "local_nocap_a001": "no cap\na=.01",
        "old_global_a001": "old global\na=.01",
        "old_global_a005": "old global\na=.05",
    }
    if label in fixed:
        return fixed[label]
    match = re.fullmatch(r"frontier_alpha0p(\d+)_cap(\d+)p(\d+)", label)
    if match:
        alpha_digits = match.group(1)
        cap_int = int(match.group(2))
        cap_frac = match.group(3)
        alpha = float(f"0.{alpha_digits}")
        cap = float(f"{cap_int}.{cap_frac}")
        return f"frontier\na={alpha:g}\ncap={cap:g}"
    match = re.fullmatch(r"frontier_alpha0p(\d+)_nocap", label)
    if match:
        alpha = float(f"0.{match.group(1)}")
        return f"frontier\na={alpha:g}\nno cap"
    return label.replace("_", "\n")


def _finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else float("nan")


def _finite_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(q)) if not values.empty else float("nan")


def _run_config(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise ValueError(f"bad config payload: {run_dir / 'config.json'}")
    return config


def _manifest_status(run_dir: Path) -> str:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return "missing"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return "bad_summary"
    return str(summary.get("status", "missing"))


def _complete_current_artifact(run_dir: Path) -> bool:
    if not (run_dir / "config.json").is_file():
        return False
    try:
        cfg = ExperimentConfig(**_run_config(run_dir))
    except Exception as exc:
        print(f"[variant_a_analyze] skip {run_dir.name}: bad config for current validation: {exc}", flush=True)
        return False
    if _manifest_status(run_dir) != "complete":
        print(f"[variant_a_analyze] skip {run_dir.name}: manifest status={_manifest_status(run_dir)!r}", flush=True)
        return False
    if not _all_outputs_exist(run_dir, cfg):
        print(f"[variant_a_analyze] skip {run_dir.name}: incomplete or stale cache keys", flush=True)
        return False
    return True


def _summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    cfg = _run_config(run_dir)
    downstream = pd.read_csv(run_dir / "downstream_results.csv")
    vae_metrics = pd.read_csv(run_dir / "vae_metrics.csv")
    diagnostics = pd.read_csv(run_dir / "preconditioning_diagnostics.csv")
    quality = vae_metrics[vae_metrics.get("record_type", "") == "vae_quality"].copy()
    train = vae_metrics[vae_metrics.get("record_type", "").fillna("") != "vae_quality"].copy()
    eval_dec = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "decoder_latent")]
    eval_raw = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "raw")]
    train_pre = train[pd.to_numeric(train.get("train_precond_a_loss"), errors="coerce").notna()].copy()
    return {
        "label": label,
        "run_label": run_dir.name,
        "run_dir": str(run_dir),
        "exists": bool(run_dir.is_dir()),
        "alpha": float(cfg.get("vae_precond_coeff") or 0.0),
        "cap": float(cfg.get("vae_precond_max_grad_ratio") or 0.0),
        "scope": str(cfg.get("vae_precond_estimator_scope") or ""),
        "kind": str(cfg.get("vae_precond_loss_kind") or "none"),
        "decoder_eval_rows": int(len(eval_dec)),
        "decoder_eval_starts": int(eval_dec["source_weight_index"].nunique()) if "source_weight_index" in eval_dec else int(len(eval_dec)),
        "raw_eval_rows": int(len(eval_raw)),
        "decoder_aulc_median": _finite_median(eval_dec["aulc"]),
        "decoder_aulc_mean": float(pd.to_numeric(eval_dec["aulc"], errors="coerce").mean()) if not eval_dec.empty else float("nan"),
        "raw_aulc_median": _finite_median(eval_raw["aulc"]),
        "decoded_test_loss_median": _finite_median(quality["decoded_test_loss"]) if "decoded_test_loss" in quality else float("nan"),
        "decoded_test_acc_median": _finite_median(quality["decoded_test_acc"]) if "decoded_test_acc" in quality else float("nan"),
        "raw_test_loss_median": _finite_median(quality["raw_test_loss"]) if "raw_test_loss" in quality else float("nan"),
        "raw_test_acc_median": _finite_median(quality["raw_test_acc"]) if "raw_test_acc" in quality else float("nan"),
        "reconstruction_rel_l2_median": _finite_median(quality["reconstruction_rel_l2"]) if "reconstruction_rel_l2" in quality else float("nan"),
        "li_A_full_per_dim_median": _finite_median(diagnostics["li_A_full_per_dim"]) if "li_A_full_per_dim" in diagnostics else float("nan"),
        "li_A_full_per_dim_p95": _finite_quantile(diagnostics["li_A_full_per_dim"], 0.95) if "li_A_full_per_dim" in diagnostics else float("nan"),
        "hvp_probe_a_loss_per_dim_median": _finite_median(diagnostics["hvp_probe_a_loss_per_dim"])
        if "hvp_probe_a_loss_per_dim" in diagnostics
        else float("nan"),
        "hvp_probe_trace_m_per_dim_median": _finite_median(diagnostics["hvp_probe_trace_m_per_dim"])
        if "hvp_probe_trace_m_per_dim" in diagnostics
        else float("nan"),
        "train_precond_a_loss_median": _finite_median(train_pre["train_precond_a_loss"]) if "train_precond_a_loss" in train_pre else float("nan"),
        "train_precond_grad_scale_median": _finite_median(train_pre["train_precond_grad_scale"]) if "train_precond_grad_scale" in train_pre else float("nan"),
        "train_precond_effective_loss_median": _finite_median(train_pre["train_precond_effective_loss"])
        if "train_precond_effective_loss" in train_pre
        else float("nan"),
        "final_val_recon_mse": float(pd.to_numeric(train["val_recon_mse"], errors="coerce").dropna().iloc[-1])
        if "val_recon_mse" in train and pd.to_numeric(train["val_recon_mse"], errors="coerce").notna().any()
        else float("nan"),
    }


def _downstream_eval(run_dir: Path, label: str) -> pd.DataFrame:
    downstream = pd.read_csv(run_dir / "downstream_results.csv")
    rows = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "decoder_latent")].copy()
    rows["label"] = label
    return rows


def _frontier_run_dirs() -> dict[str, str]:
    result: dict[str, str] = {}
    for run_dir in sorted(ARTIFACT_ROOT.glob("sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_frontier_*_v1_seed0")):
        suffix = run_dir.name.replace("sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_frontier_", "").replace("_v1_seed0", "")
        result[f"frontier_{suffix}"] = run_dir.name
    return result


def _plot_frontier(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary[summary["label"] != "control"].copy()
    plot_df = plot_df[np.isfinite(plot_df["li_A_full_per_dim_p95"]) & np.isfinite(plot_df["delta_aulc_mean_vs_control"])]
    plot_df = plot_df.reset_index(drop=True)
    plot_df["cap_label"] = np.where(plot_df["cap"] <= 0.0, "no cap", "cap=" + plot_df["cap"].map(lambda v: f"{v:.2g}"))
    plot_df["marker"] = np.where(plot_df["cap"] <= 0.0, "x", "o")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    for (cap_label, mark), sub in plot_df.groupby(["cap_label", "marker"], sort=True):
        axes[0].scatter(sub["li_A_full_per_dim_p95"], sub["delta_aulc_mean_vs_control"], label=cap_label, marker=mark, s=70, alpha=0.85)
    axes[0].set_xscale("log")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("diagnostic exact A p95 / dim (log)")
    axes[0].set_ylabel("decoder AULC mean delta vs control")
    axes[0].set_title("A proxy vs downstream")
    axes[0].legend(fontsize=8)

    cap_line = plot_df[
        (plot_df["alpha"].round(8) == 0.01)
        & (plot_df["cap"] > 0.0)
        & (plot_df["label"].str.contains("frontier|fixed_cap", regex=True))
    ].copy()
    cap_line = cap_line.sort_values("cap")
    axes[1].plot(cap_line["cap"], cap_line["li_A_full_per_dim_p95"], marker="o", label="exact A p95")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("positive grad-ratio cap")
    axes[1].set_ylabel("exact A p95 / dim")
    axes[1].set_title("cap frontier at alpha=0.01")
    ax2 = axes[1].twinx()
    ax2.plot(cap_line["cap"], cap_line["delta_aulc_mean_vs_control"], color="tab:red", marker="s", label="AULC delta")
    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.set_ylabel("AULC mean delta vs control")
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, fontsize=8, loc="best")

    alpha_cap = plot_df[(plot_df["cap"].round(8) == 0.25) & (plot_df["kind"] == "li_a_hvp")].copy()
    alpha_cap = alpha_cap.sort_values("alpha")
    axes[2].plot(alpha_cap["alpha"], alpha_cap["train_precond_grad_scale_median"], marker="o", label="median grad scale")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("median grad scale")
    axes[2].set_title("fixed cap alpha sensitivity")
    ax3 = axes[2].twinx()
    ax3.plot(alpha_cap["alpha"], alpha_cap["delta_aulc_mean_vs_control"], marker="s", color="tab:red", label="AULC delta")
    ax3.axhline(0.0, color="black", linewidth=1)
    ax3.set_ylabel("AULC mean delta vs control")
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    path = out_dir / "variant_a_frontier_pareto.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_quality(summary: pd.DataFrame, out_dir: Path) -> None:
    keep = summary[summary["label"].isin(["control", "fixed_cap_a001", "fixed_cap_a005", "local_nocap_a001", "old_global_a005"]) | summary["label"].str.startswith("frontier_")].copy()
    keep = keep.sort_values(["kind", "cap", "alpha", "label"])
    x = np.arange(len(keep))
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(keep) * 0.65), 10), constrained_layout=True, sharex=True)
    axes[0].bar(x, keep["delta_aulc_mean_vs_control"], color=np.where(keep["delta_aulc_mean_vs_control"] <= 0, "tab:green", "tab:red"), alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("AULC mean delta")
    axes[0].set_title("downstream: lower than 0 would beat control")
    axes[1].bar(x, keep["reconstruction_rel_l2_median"], color="tab:blue", alpha=0.8)
    axes[1].axhline(float(summary.loc[summary["label"] == "control", "reconstruction_rel_l2_median"].iloc[0]), color="black", linewidth=1, linestyle="--")
    axes[1].set_ylabel("median recon rel-L2")
    axes[2].bar(x, keep["li_A_full_per_dim_p95"], color="tab:purple", alpha=0.8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("exact A p95 / dim")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([_short_label(label) for label in keep["label"]], rotation=0, ha="center", fontsize=7)
    path = out_dir / "variant_a_quality_frontier.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_block_splice(out_dir: Path) -> None:
    summary_path = BLOCK_SPLICE_DIR / "decoder_block_splice_summary.csv"
    if not summary_path.is_file():
        return
    rows = pd.read_csv(summary_path)
    keep = rows[
        (rows["block_group"].isin(["classifier_head", "all_weight_tensors"]))
        & (rows["action"] == "rescue_raw_block_into_decoded")
        & (rows["start_group"].isin(["fashion_eval", "worst_fashion"]))
    ].copy()
    keep["residual_fraction"] = keep["candidate_minus_raw_test_loss_median"] / keep["decoded_minus_raw_test_loss_median"].replace(0, np.nan)
    labels = []
    values = []
    colors = []
    for _, row in keep.sort_values(["start_group", "variant_label", "block_group"]).iterrows():
        variant = _short_label(str(row["variant_label"])).replace("\n", " ")
        labels.append(f"{variant} | {row['start_group']} | {row['block_group']}")
        values.append(float(row["rescue_removed_fraction_of_decoded_test_gap_median"]))
        colors.append("tab:orange" if row["block_group"] == "classifier_head" else "tab:blue")
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, max(6.0, len(labels) * 0.34)), constrained_layout=True)
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.axvline(0.9, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1.05)
    ax.set_ylabel("variant / start slice / rescued block")
    ax.set_title("raw-block rescue intervention")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("fraction removed")
    path = out_dir / "variant_a_block_splice_rescue.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_downstream_deltas(run_map: dict[str, str], out_dir: Path) -> pd.DataFrame:
    control = _downstream_eval(ARTIFACT_ROOT / run_map["control"], "control")
    key_cols = ["source_weight_index", "start_index", "task_name", "tau"]
    control_key = control[key_cols + ["aulc", "final_test_loss", "final_test_acc"]].rename(
        columns={
            "aulc": "control_aulc",
            "final_test_loss": "control_final_test_loss",
            "final_test_acc": "control_final_test_acc",
        }
    )
    frames = []
    for label, run_name in run_map.items():
        rows = _downstream_eval(ARTIFACT_ROOT / run_name, label)
        merged = rows.merge(control_key, on=key_cols, how="left", validate="one_to_one")
        if int(merged["control_aulc"].notna().sum()) != int(len(merged)):
            raise RuntimeError(
                f"downstream start mismatch for {label}: matched "
                f"{int(merged['control_aulc'].notna().sum())}/{int(len(merged))} rows"
            )
        merged["delta_aulc_vs_control"] = merged["aulc"] - merged["control_aulc"]
        merged["delta_final_test_loss_vs_control"] = merged["final_test_loss"] - merged["control_final_test_loss"]
        merged["delta_final_test_acc_vs_control"] = merged["final_test_acc"] - merged["control_final_test_acc"]
        frames.append(merged)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame.to_csv(out_dir / "variant_a_frontier_per_start_deltas.csv", index=False)
    coverage = frame.groupby("label", as_index=False).agg(
        rows=("source_weight_index", "size"),
        starts=("source_weight_index", "nunique"),
        matched_control_rows=("control_aulc", lambda s: int(s.notna().sum())),
        mean_delta_aulc=("delta_aulc_vs_control", "mean"),
        median_delta_aulc=("delta_aulc_vs_control", "median"),
        max_delta_aulc=("delta_aulc_vs_control", "max"),
        min_delta_aulc=("delta_aulc_vs_control", "min"),
    )
    coverage.to_csv(out_dir / "variant_a_frontier_downstream_coverage.csv", index=False)
    task = frame.groupby(["label", "task_name"], as_index=False).agg(
        starts=("source_weight_index", "nunique"),
        mean_delta_aulc=("delta_aulc_vs_control", "mean"),
        median_delta_aulc=("delta_aulc_vs_control", "median"),
        mean_delta_final_test_loss=("delta_final_test_loss_vs_control", "mean"),
        mean_delta_final_test_acc=("delta_final_test_acc_vs_control", "mean"),
    )
    task.to_csv(out_dir / "variant_a_frontier_task_deltas.csv", index=False)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Variant A frontier runs and block-splice interventions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_map = {**BASE_RUNS, **_frontier_run_dirs()}
    available = {label: run_name for label, run_name in run_map.items() if _complete_current_artifact(ARTIFACT_ROOT / run_name)}
    missing = sorted(set(run_map) - set(available))
    print(f"[variant_a_analyze] available_runs={len(available)} missing={missing} out_dir={out_dir}", flush=True)
    if "control" not in available:
        raise RuntimeError("current-code control is missing or stale; run scripts/run_variant_a_current_control.py first")
    summary = pd.DataFrame([_summarize_run(label, ARTIFACT_ROOT / run_name) for label, run_name in available.items()])
    control_mean = float(summary.loc[summary["label"] == "control", "decoder_aulc_mean"].iloc[0])
    control_median = float(summary.loc[summary["label"] == "control", "decoder_aulc_median"].iloc[0])
    summary["delta_aulc_mean_vs_control"] = summary["decoder_aulc_mean"] - control_mean
    summary["delta_aulc_median_vs_control"] = summary["decoder_aulc_median"] - control_median
    summary = summary.sort_values(["kind", "cap", "alpha", "label"]).reset_index(drop=True)
    summary.to_csv(out_dir / "variant_a_frontier_summary.csv", index=False)
    coverage = _write_downstream_deltas(available, out_dir)
    _plot_frontier(summary, out_dir)
    _plot_quality(summary, out_dir)
    _plot_block_splice(out_dir)
    print(
        "[variant_a_analyze] wrote "
        f"summary={out_dir / 'variant_a_frontier_summary.csv'} "
        f"coverage={out_dir / 'variant_a_frontier_downstream_coverage.csv'} "
        f"plots={[str(p) for p in sorted(out_dir.glob('*.png'))]}",
        flush=True,
    )
    bad_coverage = coverage[(coverage["rows"] != 16) | (coverage["matched_control_rows"] != coverage["rows"])]
    if not bad_coverage.empty:
        print("[variant_a_analyze] WARNING bad downstream coverage")
        print(bad_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
