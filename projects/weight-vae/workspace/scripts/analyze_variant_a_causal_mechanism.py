from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/clean_harness_analysis"
ANCHOR_DIR = BASE / "anchor_sweep"
PATH_DIR = BASE / "weight_paths_anchor_sweep"
SPLICE_DIR = BASE / "block_splice_anchor_sweep"
OUT_DIR = BASE / "causal_mechanism"

VARIANT_ORDER = [
    "control c=0",
    "A c=0",
    "control c=1e-4",
    "A c=1e-4",
    "control c=3e-4",
    "A c=3e-4",
]
TAIL_SOURCES = [10946, 13160, 942, 122, 6026, 14718, 14431]


def _trapz_auc(group: pd.DataFrame, column: str) -> float:
    ordered = group.sort_values("step")
    x = pd.to_numeric(ordered["step"], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(ordered[column], errors="raise").to_numpy(dtype=float)
    if len(x) < 2 or float(x[-1] - x[0]) <= 0.0:
        raise ValueError(f"cannot compute AULC for {column}: steps={x.tolist()}")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _load_path_decomposition() -> pd.DataFrame:
    steps_path = PATH_DIR / "downstream_weight_path_steps.csv"
    summary_path = PATH_DIR / "downstream_weight_path_summary.csv"
    if not steps_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"missing path diagnostic outputs in {PATH_DIR}")
    steps = pd.read_csv(steps_path)
    summary = pd.read_csv(summary_path)
    expected_rows = len(VARIANT_ORDER) * 16
    if len(summary) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} path summary rows, got {len(summary)}")
    required_steps = {0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300}
    observed_steps = set(pd.to_numeric(steps["step"], errors="raise").astype(int).unique().tolist())
    if observed_steps != required_steps:
        raise RuntimeError(f"path steps mismatch: observed={sorted(observed_steps)}")

    rows: list[dict[str, object]] = []
    for (variant, source, task), group in steps.groupby(["variant_label", "source_weight_index", "task_name"]):
        step0 = group[group["step"] == 0]
        if len(step0) != 1:
            raise RuntimeError(f"{variant} source={source} has {len(step0)} step0 rows")
        rows.append(
            {
                "variant_label": str(variant),
                "source_weight_index": int(source),
                "task_name": str(task),
                "raw_auc": _trapz_auc(group, "raw_test_loss"),
                "decoded_raw_auc": _trapz_auc(group, "decoded_raw_test_loss"),
                "latent_auc": _trapz_auc(group, "latent_test_loss"),
                "step0_decoded_minus_raw": float(step0["delta_test_loss_decoded_raw_minus_raw"].iloc[0]),
            }
        )
    result = pd.DataFrame(rows)
    result["decoded_raw_minus_raw_auc"] = result["decoded_raw_auc"] - result["raw_auc"]
    result["latent_minus_raw_auc"] = result["latent_auc"] - result["raw_auc"]
    result["latent_minus_decoded_raw_auc"] = result["latent_auc"] - result["decoded_raw_auc"]
    result["is_tail"] = result["source_weight_index"].isin(TAIL_SOURCES)
    return result


def _plot_path_decomposition(path_df: pd.DataFrame) -> None:
    summary = (
        path_df.groupby("variant_label", as_index=False)
        .agg(
            step0_decoded_minus_raw=("step0_decoded_minus_raw", "mean"),
            decoded_start_auc=("decoded_raw_minus_raw_auc", "mean"),
            latent_only_auc=("latent_minus_decoded_raw_auc", "mean"),
            latent_total_auc=("latent_minus_raw_auc", "mean"),
        )
        .set_index("variant_label")
        .loc[VARIANT_ORDER]
        .reset_index()
    )
    tail = (
        path_df[path_df["is_tail"]]
        .groupby("variant_label", as_index=False)
        .agg(
            decoded_start_auc=("decoded_raw_minus_raw_auc", "mean"),
            latent_only_auc=("latent_minus_decoded_raw_auc", "mean"),
        )
        .set_index("variant_label")
        .loc[VARIANT_ORDER]
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "path_decomposition_summary.csv", index=False)
    tail.to_csv(OUT_DIR / "path_decomposition_tail_summary.csv", index=False)

    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    axes[0].bar(x, summary["decoded_start_auc"], label="raw Adam from decoded start - raw Adam", color="#4c78a8")
    axes[0].bar(
        x,
        summary["latent_only_auc"],
        bottom=summary["decoded_start_auc"],
        label="latent Adam - raw Adam from decoded start",
        color="#f58518",
    )
    axes[0].set_title("test-loss AULC gap decomposition")
    axes[0].set_ylabel("gap vs raw Adam from original weights")
    axes[0].set_xticks(x, summary["variant_label"], rotation=25, ha="right")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, summary["step0_decoded_minus_raw"], color="#54a24b")
    axes[1].set_title("decoded step0 test-loss offset")
    axes[1].set_ylabel("decoded - raw test loss at step 0")
    axes[1].set_xticks(x, summary["variant_label"], rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3g", fontsize=8)
    fig.savefig(OUT_DIR / "path_decomposition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(x, tail["decoded_start_auc"], label="decoded-start component", color="#4c78a8")
    ax.bar(x, tail["latent_only_auc"], bottom=tail["decoded_start_auc"], label="latent-only component", color="#f58518")
    ax.set_title("tail sources: test-loss AULC gap decomposition")
    ax.set_ylabel("mean gap on selected fashion tail starts")
    ax.set_xticks(x, tail["variant_label"], rotation=25, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3g", fontsize=8)
    fig.savefig(OUT_DIR / "path_decomposition_tail.png", dpi=180)
    plt.close(fig)


def _load_splice() -> pd.DataFrame:
    rows_path = SPLICE_DIR / "decoder_block_splice_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    rows = pd.read_csv(rows_path)
    expected = len(VARIANT_ORDER) * len(TAIL_SOURCES) * 9 * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} splice rows, got {len(rows)}")
    return rows


def _plot_splice(splice: pd.DataFrame) -> None:
    focus_blocks = ["classifier_head", "fc2.weight", "input_hidden_block", "all_weight_tensors"]
    rescue = splice[
        (splice["action"] == "rescue_raw_block_into_decoded") & (splice["block_group"].isin(focus_blocks))
    ].copy()
    rescue_summary = (
        rescue.groupby(["variant_label", "block_group"], as_index=False)
        .agg(
            decoded_gap_median=("decoded_minus_raw_test_loss", "median"),
            rescue_removed_fraction_median=("rescue_removed_fraction_of_decoded_test_gap", "median"),
            candidate_gap_median=("candidate_minus_raw_test_loss", "median"),
        )
    )
    rescue_summary.to_csv(OUT_DIR / "block_splice_rescue_summary.csv", index=False)

    pivot = (
        rescue_summary.pivot(index="variant_label", columns="block_group", values="rescue_removed_fraction_median")
        .loc[VARIANT_ORDER, focus_blocks]
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    width = 0.18
    x = np.arange(len(pivot.index))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for i, block in enumerate(focus_blocks):
        vals = pivot[block].to_numpy(dtype=float)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=block, color=colors[i])
    ax.axhline(1.0, color="black", lw=1, alpha=0.5)
    ax.axhline(0.0, color="black", lw=1, alpha=0.5)
    ax.set_title("raw-block rescue fraction of decoded test-loss gap")
    ax.set_ylabel("fraction removed; 1 = full rescue")
    ax.set_xticks(x, pivot.index, rotation=25, ha="right")
    ax.legend(fontsize=9, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "block_splice_rescue_fractions.png", dpi=180)
    plt.close(fig)

    per_source = splice[
        (splice["variant_label"] == "A c=3e-4")
        & (splice["action"] == "rescue_raw_block_into_decoded")
        & (splice["block_group"].isin(["classifier_head", "input_hidden_block", "all_weight_tensors"]))
    ].copy()
    decoded = splice[
        (splice["variant_label"] == "A c=3e-4")
        & (splice["action"] == "rescue_raw_block_into_decoded")
        & (splice["block_group"] == "all_tensors")
    ][["source_weight_index", "decoded_minus_raw_test_loss", "raw_test_acc", "decoded_test_acc"]]
    wide = per_source.pivot_table(
        index="source_weight_index",
        columns="block_group",
        values="rescue_removed_fraction_of_decoded_test_gap",
        aggfunc="median",
    ).reset_index()
    per_source_summary = decoded.merge(wide, on="source_weight_index", how="left").sort_values(
        "decoded_minus_raw_test_loss", ascending=False
    )
    per_source_summary.to_csv(OUT_DIR / "block_splice_a_c3_tail_per_source.csv", index=False)

    x = np.arange(len(per_source_summary))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    axes[0].bar(x, per_source_summary["decoded_minus_raw_test_loss"], color="#e45756")
    axes[0].set_title("A c=3e-4 tail decoded test-loss gap")
    axes[0].set_ylabel("decoded - raw test loss")
    axes[0].set_xticks(x, per_source_summary["source_weight_index"].astype(str), rotation=30, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    for block, color in [
        ("classifier_head", "#4c78a8"),
        ("input_hidden_block", "#54a24b"),
        ("all_weight_tensors", "#f58518"),
    ]:
        axes[1].plot(
            x,
            per_source_summary[block],
            marker="o",
            label=block,
            color=color,
        )
    axes[1].axhline(1.0, color="black", lw=1, alpha=0.5)
    axes[1].axhline(0.0, color="black", lw=1, alpha=0.5)
    axes[1].set_title("A c=3e-4 per-source rescue fraction")
    axes[1].set_ylabel("fraction of decoded gap removed")
    axes[1].set_xticks(x, per_source_summary["source_weight_index"].astype(str), rotation=30, ha="right")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "block_splice_a_c3_tail.png", dpi=180)
    plt.close(fig)


def _write_notes(path_df: pd.DataFrame, splice: pd.DataFrame) -> None:
    pair_summary = pd.read_csv(ANCHOR_DIR / "anchor_sweep_pair_summary.csv")
    path_summary = pd.read_csv(OUT_DIR / "path_decomposition_summary.csv")
    tail_path = pd.read_csv(OUT_DIR / "path_decomposition_tail_summary.csv")
    rescue = pd.read_csv(OUT_DIR / "block_splice_rescue_summary.csv")
    a_c3_rescue = rescue[
        (rescue["variant_label"] == "A c=3e-4") & (rescue["block_group"] == "classifier_head")
    ].iloc[0]
    a_c3_path = path_summary[path_summary["variant_label"] == "A c=3e-4"].iloc[0]
    c3_path = path_summary[path_summary["variant_label"] == "control c=3e-4"].iloc[0]
    a_c3_tail = tail_path[tail_path["variant_label"] == "A c=3e-4"].iloc[0]
    c3_tail = tail_path[tail_path["variant_label"] == "control c=3e-4"].iloc[0]
    c3_pair = pair_summary[pair_summary["pair"] == "c=3e-4"].iloc[0]

    notes = [
        "# Variant A Causal Mechanism Diagnostics",
        "",
        "## Checks",
        f"- Path rows: `{len(path_df)}` source/run rows, expected `{len(VARIANT_ORDER) * 16}`.",
        f"- Block-splice rows: `{len(splice)}` intervention rows, expected `{len(VARIANT_ORDER) * len(TAIL_SOURCES) * 9 * 2}`.",
        "",
        "## Main Findings",
        (
            "- Matched `A c=3e-4` is worse than matched control: "
            f"test-curve AULC delta `{c3_pair['mean_delta_test_curve_aulc']:.6g}`, "
            f"step0 test-loss delta `{c3_pair['mean_delta_step0_test_loss']:.6g}`."
        ),
        (
            "- Path decomposition attributes most absolute `A c=3e-4` damage to decoded-start quality: "
            f"decoded-start AULC component `{a_c3_path['decoded_start_auc']:.6g}`, "
            f"latent-only component `{a_c3_path['latent_only_auc']:.6g}`."
        ),
        (
            "- Relative to `control c=3e-4`, `A c=3e-4` increases decoded-start component "
            f"by `{a_c3_path['decoded_start_auc'] - c3_path['decoded_start_auc']:.6g}` "
            f"and latent-only component by `{a_c3_path['latent_only_auc'] - c3_path['latent_only_auc']:.6g}`."
        ),
        (
            "- On tail sources, `A c=3e-4` decoded-start component is "
            f"`{a_c3_tail['decoded_start_auc']:.6g}` vs control `{c3_tail['decoded_start_auc']:.6g}`; "
            f"latent-only is `{a_c3_tail['latent_only_auc']:.6g}` vs control `{c3_tail['latent_only_auc']:.6g}`."
        ),
        (
            "- Block-splice localizes decoded test-loss gap to the classifier head: "
            f"`A c=3e-4` classifier-head rescue removes median "
            f"`{a_c3_rescue['rescue_removed_fraction_median']:.4g}` of tail decoded gap."
        ),
        "",
        "## Figures",
        "- `path_decomposition.png`",
        "- `path_decomposition_tail.png`",
        "- `block_splice_rescue_fractions.png`",
        "- `block_splice_a_c3_tail.png`",
    ]
    (OUT_DIR / "causal_mechanism_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    print(f"[causal_mechanism] input anchor_dir={ANCHOR_DIR}")
    print(f"[causal_mechanism] input path_dir={PATH_DIR}")
    print(f"[causal_mechanism] input splice_dir={SPLICE_DIR}")
    print(f"[causal_mechanism] output_dir={OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_df = _load_path_decomposition()
    print(f"[causal_mechanism] loaded path rows={len(path_df)}")
    _plot_path_decomposition(path_df)
    print("[causal_mechanism] wrote path decomposition figures")
    splice = _load_splice()
    print(f"[causal_mechanism] loaded splice rows={len(splice)}")
    _plot_splice(splice)
    print("[causal_mechanism] wrote block-splice figures")
    _write_notes(path_df, splice)
    print(f"[causal_mechanism] wrote notes={OUT_DIR / 'causal_mechanism_notes.md'}")


if __name__ == "__main__":
    main()
