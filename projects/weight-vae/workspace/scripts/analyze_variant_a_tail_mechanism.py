from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analyze_variant_a_frontier import (
    ARTIFACT_ROOT,
    BASE_RUNS,
    _complete_current_artifact,
    _frontier_run_dirs,
    _short_label,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/frontier_analysis_current_control"
KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]
CURVE_KEY_COLS = KEY_COLS + ["step"]


PLOT_LABELS = [
    "frontier_alpha0p0100_cap0p050",
    "frontier_alpha0p0100_cap0p100",
    "fixed_cap_a001",
    "frontier_alpha0p0100_cap0p500",
    "frontier_alpha0p0100_cap1p000",
    "local_nocap_a001",
]


def _available_runs() -> dict[str, str]:
    run_map = {**BASE_RUNS, **_frontier_run_dirs()}
    return {label: run_name for label, run_name in run_map.items() if _complete_current_artifact(ARTIFACT_ROOT / run_name)}


def _decoder_eval_results(run_name: str, label: str) -> pd.DataFrame:
    rows = pd.read_csv(ARTIFACT_ROOT / run_name / "downstream_results.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == "decoder_latent")].copy()
    rows["label"] = label
    return rows


def _decoder_eval_curves(run_name: str, label: str) -> pd.DataFrame:
    rows = pd.read_csv(ARTIFACT_ROOT / run_name / "downstream_curves.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == "decoder_latent")].copy()
    rows["label"] = label
    return rows


def _build_step_delta_tables(available: dict[str, str], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "control" not in available:
        raise RuntimeError("current-code control is missing")
    control_results = _decoder_eval_results(available["control"], "control")
    control_curves = _decoder_eval_curves(available["control"], "control")
    control_result_key = control_results[KEY_COLS + ["aulc", "final_test_loss", "reconstruction_rel_l2"]].rename(
        columns={
            "aulc": "control_aulc",
            "final_test_loss": "control_final_test_loss",
            "reconstruction_rel_l2": "control_reconstruction_rel_l2",
        }
    )
    control_curve_key = control_curves[CURVE_KEY_COLS + ["train_loss", "test_loss", "test_acc"]].rename(
        columns={
            "train_loss": "control_train_loss",
            "test_loss": "control_test_loss",
            "test_acc": "control_test_acc",
        }
    )
    result_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for label, run_name in available.items():
        rows = _decoder_eval_results(run_name, label).merge(control_result_key, on=KEY_COLS, how="left", validate="one_to_one")
        if int(rows["control_aulc"].notna().sum()) != int(len(rows)):
            raise RuntimeError(f"result start mismatch for {label}")
        rows["delta_aulc_vs_control"] = rows["aulc"] - rows["control_aulc"]
        rows["delta_final_test_loss_vs_control"] = rows["final_test_loss"] - rows["control_final_test_loss"]
        rows["delta_reconstruction_rel_l2_vs_control"] = rows["reconstruction_rel_l2"] - rows["control_reconstruction_rel_l2"]
        result_frames.append(rows)

        curves = _decoder_eval_curves(run_name, label).merge(control_curve_key, on=CURVE_KEY_COLS, how="left", validate="one_to_one")
        if int(curves["control_test_loss"].notna().sum()) != int(len(curves)):
            raise RuntimeError(f"curve start/step mismatch for {label}")
        curves["delta_test_loss_vs_control"] = curves["test_loss"] - curves["control_test_loss"]
        curves["delta_train_loss_vs_control"] = curves["train_loss"] - curves["control_train_loss"]
        curve_frames.append(curves)

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    curves = pd.concat(curve_frames, ignore_index=True, sort=False)
    step0 = curves[curves["step"] == 0][KEY_COLS + ["label", "delta_test_loss_vs_control", "delta_train_loss_vs_control"]].rename(
        columns={
            "delta_test_loss_vs_control": "delta_step0_test_loss_vs_control",
            "delta_train_loss_vs_control": "delta_step0_train_loss_vs_control",
        }
    )
    results = results.merge(step0, on=KEY_COLS + ["label"], how="left", validate="one_to_one")
    curves = curves.merge(
        step0[KEY_COLS + ["label", "delta_step0_test_loss_vs_control", "delta_step0_train_loss_vs_control"]],
        on=KEY_COLS + ["label"],
        how="left",
        validate="many_to_one",
    )
    curves["step0_gap_closed_test_loss"] = curves["delta_step0_test_loss_vs_control"] - curves["delta_test_loss_vs_control"]
    curves["step0_gap_closed_train_loss"] = curves["delta_step0_train_loss_vs_control"] - curves["delta_train_loss_vs_control"]
    results.to_csv(out_dir / "variant_a_step0_result_deltas.csv", index=False)
    curves.to_csv(out_dir / "variant_a_curve_deltas.csv", index=False)
    return results, curves


def _summary_table(results: pd.DataFrame, curves: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summaries = []
    for label, sub in results.groupby("label"):
        if label == "control":
            continue
        step0 = pd.to_numeric(sub["delta_step0_test_loss_vs_control"], errors="coerce")
        aulc = pd.to_numeric(sub["delta_aulc_vs_control"], errors="coerce")
        corr = float(step0.corr(aulc)) if step0.notna().sum() >= 3 and aulc.notna().sum() >= 3 else float("nan")
        curve_sub = curves[curves["label"] == label]
        by_step = curve_sub.groupby("step", as_index=False).agg(
            mean_delta_test_loss=("delta_test_loss_vs_control", "mean"),
            mean_step0_gap_closed=("step0_gap_closed_test_loss", "mean"),
        )
        row = {
            "label": label,
            "n_eval_starts": int(sub["source_weight_index"].nunique()),
            "mean_delta_aulc": float(aulc.mean()),
            "median_delta_aulc": float(aulc.median()),
            "mean_delta_step0_test_loss": float(step0.mean()),
            "median_delta_step0_test_loss": float(step0.median()),
            "corr_step0_delta_vs_aulc_delta": corr,
            "mean_delta_final_test_loss": float(pd.to_numeric(sub["delta_final_test_loss_vs_control"], errors="coerce").mean()),
            "mean_delta_reconstruction_rel_l2": float(pd.to_numeric(sub["delta_reconstruction_rel_l2_vs_control"], errors="coerce").mean()),
        }
        for step in [25, 50, 100, 200, 300]:
            hit = by_step[by_step["step"] == step]
            if hit.empty:
                row[f"mean_delta_test_loss_step{step}"] = float("nan")
                row[f"mean_step0_gap_closed_step{step}"] = float("nan")
            else:
                row[f"mean_delta_test_loss_step{step}"] = float(hit["mean_delta_test_loss"].iloc[0])
                row[f"mean_step0_gap_closed_step{step}"] = float(hit["mean_step0_gap_closed"].iloc[0])
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("mean_delta_step0_test_loss").reset_index(drop=True)
    summary.to_csv(out_dir / "variant_a_step0_mechanism_summary.csv", index=False)
    return summary


def _plot_step0_scatter(results: pd.DataFrame, out_dir: Path) -> None:
    keep = results[results["label"].isin(PLOT_LABELS)].copy()
    keep = keep[np.isfinite(keep["delta_step0_test_loss_vs_control"]) & np.isfinite(keep["delta_aulc_vs_control"])]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), constrained_layout=True)
    panels = [
        ("capped variants", keep[keep["label"] != "local_nocap_a001"]),
        ("no-cap ablation", keep[keep["label"] == "local_nocap_a001"]),
    ]
    cmap = plt.get_cmap("tab10")
    for ax, (title, sub) in zip(axes, panels, strict=True):
        for idx, (label, rows) in enumerate(sub.groupby("label", sort=True)):
            ax.scatter(
                rows["delta_step0_test_loss_vs_control"],
                rows["delta_aulc_vs_control"],
                s=58,
                alpha=0.82,
                label=_short_label(label).replace("\n", " "),
                color=cmap(idx % 10),
            )
            for _, row in rows.nlargest(2, "delta_aulc_vs_control").iterrows():
                ax.annotate(
                    str(int(row["source_weight_index"])),
                    (float(row["delta_step0_test_loss_vs_control"]), float(row["delta_aulc_vs_control"])),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("decoded step0 test-loss delta vs control")
        ax.set_ylabel("decoder AULC delta vs control")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    path = out_dir / "variant_a_step0_vs_aulc_delta.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _plot_gap_closure(curves: pd.DataFrame, out_dir: Path) -> None:
    keep = curves[curves["label"].isin(PLOT_LABELS)].copy()
    grouped = keep.groupby(["label", "step"], as_index=False).agg(
        mean_delta_test_loss=("delta_test_loss_vs_control", "mean"),
        sem_delta_test_loss=("delta_test_loss_vs_control", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=1) / np.sqrt(max(1, s.notna().sum())))),
        mean_gap_closed=("step0_gap_closed_test_loss", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(PLOT_LABELS):
        rows = grouped[grouped["label"] == label].sort_values("step")
        if rows.empty:
            continue
        color = cmap(idx % 10)
        legend_label = _short_label(label).replace("\n", " ")
        axes[0].plot(rows["step"], rows["mean_delta_test_loss"], marker="o", markersize=3, label=legend_label, color=color)
        lo = rows["mean_delta_test_loss"] - rows["sem_delta_test_loss"]
        hi = rows["mean_delta_test_loss"] + rows["sem_delta_test_loss"]
        axes[0].fill_between(rows["step"].to_numpy(dtype=float), lo.to_numpy(dtype=float), hi.to_numpy(dtype=float), color=color, alpha=0.13)
        axes[1].plot(rows["step"], rows["mean_gap_closed"], marker="o", markersize=3, label=legend_label, color=color)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("downstream step")
    axes[0].set_ylabel("mean test-loss delta vs control")
    axes[0].set_title("absolute gap during downstream")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("downstream step")
    axes[1].set_ylabel("mean step0 gap closed")
    axes[1].set_title("post-step0 catch-up")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)
    path = out_dir / "variant_a_gap_closure_curves.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _plot_tail_starts(curves: pd.DataFrame, out_dir: Path) -> None:
    labels = ["control", "fixed_cap_a001", "frontier_alpha0p0100_cap1p000", "local_nocap_a001"]
    candidates = curves[curves["label"] == "local_nocap_a001"].copy()
    candidates = (
        candidates[candidates["step"] == 0]
        .sort_values("delta_step0_test_loss_vs_control", ascending=False)
        .head(4)["source_weight_index"]
        .astype(int)
        .tolist()
    )
    if not candidates:
        return
    ncols = 2
    nrows = int(np.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, max(5.0, 4.1 * nrows)), constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, source_idx in zip(axes.ravel(), candidates, strict=False):
        for idx, label in enumerate(labels):
            rows = curves[(curves["label"] == label) & (curves["source_weight_index"].astype(int) == int(source_idx))].sort_values("step")
            if rows.empty:
                continue
            task = str(rows["task_name"].iloc[0])
            ax.plot(rows["step"], rows["test_loss"], marker="o", markersize=3, label=_short_label(label).replace("\n", " "), color=cmap(idx % 10))
            ax.set_title(f"source_weight_index={source_idx} task={task}")
        ax.set_xlabel("downstream step")
        ax.set_ylabel("decoder test loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(candidates) :]:
        ax.axis("off")
    path = out_dir / "variant_a_tail_start_curves.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Variant A downstream tail mechanism against current-code control.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    available = _available_runs()
    results, curves = _build_step_delta_tables(available, out_dir)
    summary = _summary_table(results, curves, out_dir)
    _plot_step0_scatter(results, out_dir)
    _plot_gap_closure(curves, out_dir)
    _plot_tail_starts(curves, out_dir)
    print(f"[variant_a_tail_mechanism] wrote out_dir={out_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(
        "[variant_a_tail_mechanism] plots="
        + ",".join(
            str(out_dir / name)
            for name in [
                "variant_a_step0_vs_aulc_delta.png",
                "variant_a_gap_closure_curves.png",
                "variant_a_tail_start_curves.png",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
