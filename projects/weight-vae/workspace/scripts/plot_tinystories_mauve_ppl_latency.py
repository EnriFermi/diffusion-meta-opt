#!/usr/bin/env python3
"""Plot TinyStories quality/latency curves with explicit NFE point labels.

All latency rows consumed by this revision are measured on the same H100 NVL
with the same batch-one, length, warmup/repeat, and upstream timing protocol.
Precision/compile policy is recorded per point and stated on every figure.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import pandas as pd


CFM_MODELS = {
    "M1-100k-source": "CFM M1 (100k target; best@68k)",
    "M1-step200000": "CFM M1 (200k)",
    "M2-source-fast": "CFM M2",
    "M3-step100000-fast": "CFM M3 / BCFM (fast)",
}

COLORS = {
    "CFM M1 (100k target; best@68k)": "#2458A6",
    "CFM M1 (100k exact)": "#C43C82",
    "CFM M1 (200k)": "#D14B3E",
    "CFM M2": "#D99100",
    "CFM M3 / BCFM (fast)": "#14866D",
    "BD3-LM": "#7651A8",
    "MDLM": "#8B5E3C",
}

MARKERS = {
    "CFM M1 (100k target; best@68k)": "o",
    "CFM M1 (100k exact)": "X",
    "CFM M1 (200k)": "h",
    "CFM M2": "^",
    "CFM M3 / BCFM (fast)": "D",
    "BD3-LM": "P",
    "MDLM": "s",
}

BASELINE_MODELS = {"BD3-LM", "MDLM"}
FIGURE_BG = "#F5F1E8"
AXIS_BG = "#FFFCF7"
INK = "#292724"
GRID = "#D8D1C5"
FIRST_HITTING_COLOR = "#B83B5E"

ANNOTATION_DY = {
    "CFM M1 (100k target; best@68k)": 6,
    "CFM M1 (100k exact)": 15,
    "CFM M1 (200k)": -13,
    "CFM M2": 6,
    "CFM M3 / BCFM (fast)": -13,
    "BD3-LM": 6,
    "MDLM": -13,
}

# Dense low-NFE regions need point-specific label placement.  These offsets
# are intentionally explicit so regenerated figures remain deterministic.
ANNOTATION_OVERRIDES = {
    ("mauve", "CFM M2", 1): (-2, 11),
    ("mauve", "CFM M2", 2): (-13, 11),
    ("mauve", "CFM M2", 4): (6, 10),
    ("mauve", "BD3-LM", 1): (10, -15),
    ("mauve", "BD3-LM", 2): (9, -15),
    ("mauve", "CFM M1 (200k)", 32): (9, 9),
    ("mauve", "CFM M3 / BCFM (fast)", 16): (8, -20),
    ("gen_ppl", "CFM M2", 1): (8, 10),
    ("gen_ppl", "CFM M2", 2): (8, 9),
    ("gen_ppl", "CFM M3 / BCFM (fast)", 8): (-12, -15),
    ("gen_ppl", "CFM M3 / BCFM (fast)", 16): (7, -15),
    ("gen_ppl", "MDLM", 16): (7, -14),
}


def _cfm_rows(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    metadata = document["metadata"]
    rows = []
    for source in document["rows"]:
        if source["model"] not in CFM_MODELS or source["discretize"] != "argmax":
            continue
        rows.append(
            {
                "model": CFM_MODELS[source["model"]],
                "model_family": source["model"],
                "sampler": source["sampler"],
                "point": source["point"],
                "nfe": int(source["nfe"]),
                "forwards_per_sequence": int(source["forwards_per_sequence"]),
                "quality_n_samples": int(source["quality_n_samples"]),
                "mauve": float(source["mauve"]),
                "gen_ppl": float(source["gen_ppl"]),
                "latency_ms": float(source["sequence_latency_ms"]),
                "latency_p10_ms": float(source["sequence_latency_p10_ms"]),
                "latency_p90_ms": float(source["sequence_latency_p90_ms"]),
                "latency_repeats": int(source["sequence_latency_repeats"]),
                "hardware": metadata["gpu_name"],
                "hardware_group": "H100",
                "dtype": metadata["dtype"],
                "batch_size": int(metadata["batch_size"]),
                "length": int(metadata["length"]),
                "quality_protocol": "gpt2_external_quality_v2 / gpt2_gptj6b_mauve_max256",
                "latency_protocol": f"upstream_{metadata['upstream_commit'][:8]}_sequence_latency_v1",
                "quality_source": source["label"],
                "latency_source": str(path.resolve()),
            }
        )
    return rows


def _baseline_rows(quality_path: Path, latency_path: Path) -> list[dict]:
    quality = pd.read_csv(quality_path)
    latency = pd.read_csv(latency_path)
    specifications = (
        ("BD3-LM", "BD3LM_best_sweep", "bd3lm_ancestral"),
        ("MDLM", "MDLM_best_sweep", "mdlm"),
    )
    rows = []
    for family, evaluation_label, sampler in specifications:
        quality_part = quality[
            (quality.model_family == family)
            & (quality.model_variant == "best_canonical_100k")
            & (quality.evaluation_label == evaluation_label)
            & (quality.sampler == sampler)
        ]
        latency_part = latency[
            (latency.model_family == family) & (latency.sampler == sampler)
        ]
        joined = quality_part.merge(
            latency_part,
            on=["model_family", "sampler", "point", "nfe"],
            how="inner",
            suffixes=("_quality", "_latency"),
            validate="one_to_one",
        )
        if len(joined) != len(quality_part):
            missing = sorted(set(quality_part.nfe) - set(joined.nfe))
            raise RuntimeError(f"{family} quality/latency join missing NFE={missing}")
        for _, source in joined.iterrows():
            rows.append(
                {
                    "model": family,
                    "model_family": family,
                    "sampler": sampler,
                    "point": source["point"],
                    "nfe": int(source["nfe"]),
                    "forwards_per_sequence": int(source["forwards_per_sequence_latency"]),
                    "quality_n_samples": int(source["n_samples"]),
                    "mauve": float(source["mauve"]),
                    "gen_ppl": float(source["gen_ppl"]),
                    "latency_ms": float(source["sequence_latency_ms"]),
                    "latency_p10_ms": float(source["sequence_latency_p10_ms"]),
                    "latency_p90_ms": float(source["sequence_latency_p90_ms"]),
                    "latency_repeats": int(source["sequence_latency_repeats"]),
                    "hardware": source["latency_gpu_name"],
                    "hardware_group": "H100",
                    "dtype": source["dtype"],
                    "batch_size": int(source["batch_size"]),
                    "length": int(source["length"]),
                    "quality_protocol": source["protocol_id_quality"],
                    "latency_protocol": source["protocol_id_latency"],
                    "quality_source": source["source_file"],
                    "latency_source": str(latency_path.resolve()),
                }
            )
    return rows


def _fused_rows(path: Path) -> list[dict]:
    """Load matched BF16/max-autotune operating points for optimized models."""
    frame = pd.read_csv(path)
    model_names = {
        "M1-68k": "CFM M1 (100k target; best@68k)",
        "M1-100k": "CFM M1 (100k exact)",
        "M1-200k": "CFM M1 (200k)",
        "M2": "CFM M2",
        "M3": "CFM M3 / BCFM (fast)",
        "BD3-LM": "BD3-LM",
        "MDLM": "MDLM",
    }
    frame = frame[frame.model.isin(model_names)].copy()
    rows = []
    for _, source in frame.iterrows():
        family = str(source["model"])
        first_hitting = "first_hitting" in str(source["sampler"])
        rows.append(
            {
                "model": model_names[family],
                "model_family": family,
                "sampler": source["sampler"],
                "point": source["point"],
                "nfe": (
                    int(source["total_denoising_nfe"])
                    if first_hitting else int(source["steps_per_block"])
                ),
                "forwards_per_sequence": int(source["backbone_calls_per_sequence"]),
                "quality_n_samples": int(source["quality_n_samples"]),
                "mauve": float(source["mauve"]),
                "gen_ppl": float(source["gen_ppl"]),
                "latency_ms": float(source["sequence_latency_ms"]),
                "latency_p10_ms": float(source["sequence_latency_p10_ms"]),
                "latency_p90_ms": float(source["sequence_latency_p90_ms"]),
                "latency_repeats": int(source["sequence_latency_repeats"]),
                "hardware": source["latency_gpu_name"],
                "hardware_group": "H100",
                "dtype": source["dtype"],
                "batch_size": int(source["batch_size"]),
                "length": int(source["length"]),
                "quality_protocol": "gpt2_external_quality_v2 / gpt2_gptj6b_mauve_max256",
                "latency_protocol": (
                    "upstream_2e21c577_fused_h100_"
                    f"{source['dtype']}_b1_l256_{source.get('compile_mode', 'default')}"
                ),
                "quality_source": source["quality_source"],
                "latency_source": str(path.resolve()),
            }
        )
    return rows


def _draw_curves(ax, frame: pd.DataFrame, metric: str, *, legend: bool = True) -> None:
    ax.set_facecolor(AXIS_BG)
    for model in COLORS:
        points = frame[
            (frame.model == model)
            & ~frame.sampler.str.contains("first_hitting")
        ].sort_values("nfe")
        if points.empty:
            continue
        x = points.latency_ms.to_numpy()
        y = points[metric].to_numpy()
        xerr = [
            x - points.latency_p10_ms.to_numpy(),
            points.latency_p90_ms.to_numpy() - x,
        ]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            label=model,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle="--" if model in BASELINE_MODELS else "-",
            linewidth=2.45 if model in BASELINE_MODELS else 2.25,
            markersize=7.2,
            markeredgecolor=AXIS_BG,
            markeredgewidth=0.8,
            elinewidth=0.9,
            capsize=2.5,
            alpha=0.96,
        )
        for _, point in points.iterrows():
            dx, dy = ANNOTATION_OVERRIDES.get(
                (metric, model, int(point.nfe)),
                (5, ANNOTATION_DY[model]),
            )
            annotation = ax.annotate(
                f"{int(point.nfe)}",
                (point.latency_ms, point[metric]),
                xytext=(dx, dy),
                textcoords="offset points",
                color=COLORS[model],
                fontsize=8.2,
                fontweight="bold",
                annotation_clip=False,
            )
            annotation.set_path_effects(
                [path_effects.withStroke(linewidth=2.5, foreground="white")]
            )
    first_hitting = frame[frame.sampler.str.contains("first_hitting")]
    if not first_hitting.empty:
        if len(first_hitting) != 1:
            raise RuntimeError("expected exactly one first-hitting point")
        point = first_hitting.iloc[0]
        xerr = [[point.latency_ms - point.latency_p10_ms],
                [point.latency_p90_ms - point.latency_ms]]
        ax.errorbar(
            [point.latency_ms], [point[metric]], xerr=xerr,
            color=FIRST_HITTING_COLOR, marker="*", linestyle="none",
            markersize=12, markeredgecolor=AXIS_BG, markeredgewidth=0.8,
            elinewidth=0.9, capsize=2.5, label="BD3-LM first-hitting",
            zorder=6,
        )
        dx, dy = (7, -16) if metric == "mauve" else (7, 7)
        annotation = ax.annotate(
            "FH", (point.latency_ms, point[metric]), xytext=(dx, dy),
            textcoords="offset points", color=FIRST_HITTING_COLOR,
            fontsize=8.2, fontweight="bold", annotation_clip=False, zorder=7,
        )
        annotation.set_path_effects(
            [path_effects.withStroke(linewidth=2.5, foreground="white")]
        )
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.9, alpha=0.72)
    ax.grid(True, which="minor", color=GRID, linewidth=0.45, alpha=0.35)
    ax.set_xlabel("batch=1 sequence latency, median ms (log scale)")
    ax.tick_params(colors=INK, labelsize=9.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8A8379")
    if legend:
        ax.legend(fontsize=8, loc="best")


def _plot_combined_dashboard(
    frame: pd.DataFrame, output: Path, protocol_caption: str
) -> None:
    """One publication-style canvas containing all requested model curves."""
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.4))
    fig.patch.set_facecolor(FIGURE_BG)

    panels = (
        (axes[0], "mauve", "MAUVE", "Distributional quality · higher is better"),
        (axes[1], "gen_ppl", "gen-PPL", "Fluency · lower is better (log scale)"),
    )
    for ax, metric, ylabel, title in panels:
        _draw_curves(ax, frame, metric, legend=False)
        ax.set_ylabel(ylabel, fontweight="semibold")
        ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="semibold", pad=12)
    axes[0].set_ylim(-0.025, 1.025)
    axes[1].set_yscale("log")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle="--" if model in BASELINE_MODELS else "-",
            linewidth=2.5,
            markersize=7.5,
            markeredgecolor=FIGURE_BG,
            label=model,
        )
        for model in COLORS
    ] + [
        Line2D(
            [0], [0], color=FIRST_HITTING_COLOR, marker="*", linestyle="none",
            markersize=10, markeredgecolor=FIGURE_BG,
            label="BD3-LM first-hitting",
        )
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=2.6,
        columnspacing=2.0,
    )
    fig.suptitle(
        "TinyStories · quality versus generation latency",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.902,
        "Numbers next to points are NFE  ·  FH = first-hitting  ·  solid = CFM argmax  ·  dashed = canonical diffusion samplers",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#5E5952",
    )
    fig.text(
        0.5,
        0.035,
        protocol_caption,
        ha="center",
        fontsize=9.5,
        color="#9A3D32",
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.14, top=0.705, wspace=0.20)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_by_family(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    output: Path,
    protocol_caption: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    facets = (
        (set(CFM_MODELS.values()), "CFM family — NVIDIA H100 NVL"),
        (BASELINE_MODELS, "Discrete baselines — NVIDIA H100 NVL"),
    )
    for ax, (models, title) in zip(axes, facets):
        _draw_curves(ax, frame[frame.model.isin(models)], metric)
        ax.set_title(title)
    axes[0].set_ylabel(ylabel)
    if metric == "mauve":
        axes[0].set_ylim(-0.02, 1.02)
    else:
        axes[0].set_yscale("log")
    fig.suptitle(
        f"TinyStories: {ylabel} vs latency by NFE (numbers next to points are NFE)",
        y=0.975,
    )
    fig.text(
        0.5,
        0.025,
        protocol_caption + " · Quality sample count is matched (512).",
        ha="center",
        fontsize=9,
    )
    # Explicit margins are more stable than tight_layout for shared log-y axes;
    # tight_layout can lift subplot titles into the suptitle band.
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.82, wspace=0.08)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    output: Path,
    protocol_caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    _draw_curves(ax, frame, metric)
    ax.set_ylabel(ylabel)
    if metric == "mauve":
        ax.set_ylim(-0.02, 1.02)
    else:
        ax.set_yscale("log")
    ax.set_title(
        f"TinyStories: {ylabel} vs latency by NFE — matched H100 NVL"
    )
    fig.text(
        0.5,
        0.012,
        "CFM: argmax endpoints. BD3-LM/MDLM: canonical stochastic samplers. "
        + protocol_caption,
        ha="center",
        color="#5E5952",
        fontsize=9,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pareto_flags(frame: pd.DataFrame, metric: str, *, higher_is_better: bool) -> pd.Series:
    flags = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("hardware_group"):
        for index, point in group.iterrows():
            no_slower = group.latency_ms <= point.latency_ms
            if higher_is_better:
                no_worse = group[metric] >= point[metric]
                strict = (group.latency_ms < point.latency_ms) | (group[metric] > point[metric])
            else:
                no_worse = group[metric] <= point[metric]
                strict = (group.latency_ms < point.latency_ms) | (group[metric] < point[metric])
            flags.loc[index] = not bool((no_slower & no_worse & strict).any())
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfm-grid", type=Path, required=True)
    parser.add_argument("--external-quality", type=Path, required=True)
    parser.add_argument("--baseline-latency", type=Path, required=True)
    parser.add_argument("--fused-quality-latency", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[config] "
        + json.dumps(
            {
                "cfm_grid": str(args.cfm_grid.resolve()),
                "external_quality": str(args.external_quality.resolve()),
                "baseline_latency": str(args.baseline_latency.resolve()),
                "fused_quality_latency": (
                    str(args.fused_quality_latency.resolve())
                    if args.fused_quality_latency else None
                ),
                "output_dir": str(output_dir),
                "device": "plotting=cpu; all measurements=NVIDIA H100 NVL",
                "dtype": "read from stored measurement rows",
                "seed": 0,
                "cache_mode": "reuse stored evaluated points",
            }
        ),
        flush=True,
    )

    print("[stage=load] loading and joining stored quality/latency points", flush=True)
    rows = _cfm_rows(args.cfm_grid)
    rows.extend(_baseline_rows(args.external_quality, args.baseline_latency))
    if args.fused_quality_latency:
        fused_rows = _fused_rows(args.fused_quality_latency)
        replaced = {row["model"] for row in fused_rows}
        rows = [row for row in rows if row["model"] not in replaced]
        rows.extend(fused_rows)
    frame = pd.DataFrame(rows).sort_values(["model", "nfe"]).reset_index(drop=True)

    expected_models = set(COLORS)
    if set(frame.model) != expected_models:
        raise RuntimeError(
            f"model coverage mismatch: got={sorted(set(frame.model))} "
            f"expected={sorted(expected_models)}"
        )
    expected_points = 42 if args.fused_quality_latency else 35
    if len(frame) != expected_points:
        raise RuntimeError(f"expected {expected_points} matched points, got {len(frame)}")
    if frame[["mauve", "gen_ppl", "latency_ms"]].isna().any().any():
        raise RuntimeError("a plotted metric is missing")
    if not (frame.quality_n_samples == 512).all():
        raise RuntimeError("quality sample counts are not matched at 512")
    if not (frame.latency_repeats == 30).all():
        raise RuntimeError("latency repeat counts are not matched at 30")
    if set(frame.hardware_group) != {"H100"}:
        raise RuntimeError(f"latency hardware is not matched: {sorted(set(frame.hardware_group))}")
    if not (
        (frame.latency_p10_ms <= frame.latency_ms)
        & (frame.latency_ms <= frame.latency_p90_ms)
    ).all():
        raise RuntimeError("latency quantiles do not bracket the median")
    if args.fused_quality_latency and set(frame.dtype) == {"bf16"}:
        protocol_caption = (
            "H100 NVL · batch 1 · length 256 · BF16 + max-autotune "
            "· 5 warmups · median of 30 repeats"
        )
    else:
        protocol_caption = (
            "H100 NVL · batch 1 · length 256 · FP32/TF32 · 5 warmups "
            "· median of 30 repeats"
        )
    frame["pareto_mauve_within_hardware"] = _pareto_flags(
        frame, "mauve", higher_is_better=True,
    )
    frame["pareto_gen_ppl_within_hardware"] = _pareto_flags(
        frame, "gen_ppl", higher_is_better=False,
    )
    print(
        f"[validity] points={len(frame)} models={frame.model.nunique()} "
        f"quality_n=512 latency_batch=1 repeats=30 finite=True "
        f"mauve_front={int(frame.pareto_mauve_within_hardware.sum())} "
        f"ppl_front={int(frame.pareto_gen_ppl_within_hardware.sum())}",
        flush=True,
    )

    points_path = output_dir / "tinystories_quality_latency_by_nfe.csv"
    frame.to_csv(points_path, index=False)
    print(f"[stage=write_data] {points_path}", flush=True)

    outputs = {
        "combined_dashboard": output_dir / "quality_vs_latency_all_models_argmax.png",
        "mauve_by_family": output_dir / "mauve_vs_latency_by_nfe_h100_by_family.png",
        "gen_ppl_by_family": output_dir / "genppl_vs_latency_by_nfe_h100_by_family.png",
        "mauve_all_models": output_dir / "mauve_vs_latency_by_nfe_h100_all_models.png",
        "gen_ppl_all_models": output_dir / "genppl_vs_latency_by_nfe_h100_all_models.png",
    }
    print("[stage=plot] rendering MAUVE and gen-PPL figures", flush=True)
    _plot_combined_dashboard(frame, outputs["combined_dashboard"], protocol_caption)
    _plot_by_family(
        frame, "mauve", "MAUVE (higher is better)",
        outputs["mauve_by_family"], protocol_caption,
    )
    _plot_by_family(
        frame, "gen_ppl", "gen-PPL (lower is better, log scale)",
        outputs["gen_ppl_by_family"], protocol_caption,
    )
    _plot_overlay(
        frame, "mauve", "MAUVE (higher is better)",
        outputs["mauve_all_models"], protocol_caption,
    )
    _plot_overlay(
        frame, "gen_ppl", "gen-PPL (lower is better, log scale)",
        outputs["gen_ppl_all_models"], protocol_caption,
    )

    metadata = {
        "rows": len(frame),
        "models": sorted(frame.model.unique()),
        "nfe_by_model": {
            model: sorted(int(value) for value in group.nfe.unique())
            for model, group in frame.groupby("model")
        },
        "quality_n_samples": 512,
        "latency_batch_size": 1,
        "latency_repeats": 30,
        "latency_hardware": "NVIDIA H100 NVL for every plotted model",
        "latency_protocol": protocol_caption + "; error bars=p10-p90",
        "bd3lm_selection": (
            "BD3LM_best_sweep quality paired with the exact-parity fused ancestral sampler; "
            "the exact-parity fused first-hitting NFE256 result is shown as a separate star "
            "rather than connected to the ancestral steps/block curve."
        ),
        "cfm_decoding": "argmax only; CFM sample-decoding rows are excluded",
        "baseline_decoding": (
            "canonical stochastic MDLM sampling and exact-token-parity fused BD3-LM "
            "ancestral sampling; no argmax baseline rows are mixed into the figure"
        ),
        "m3_selection": "M3-step100000 fused-fast argmax with independently rescored quality",
        "m2_selection": "M2-source fused-fast argmax with independently rescored quality",
        "m1_selection": "M1-68k, exact M1-100k, and M1-200k fused full-flow argmax with independently rescored BF16 quality",
        "mdlm_selection": "canonical stochastic MDLM with compiled denoiser and independently rescored BF16 quality",
        "fused_quality_latency_source": (
            str(args.fused_quality_latency.resolve())
            if args.fused_quality_latency else None
        ),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "data": str(points_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[done] elapsed={time.perf_counter() - started:.1f}s "
        f"data={points_path} metadata={metadata_path} "
        f"plots={','.join(str(path) for path in outputs.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
