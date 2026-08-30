#!/usr/bin/env python3
"""Build the compact two-panel TinyStories figure used by the BCFM paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import pandas as pd


MODEL_NAMES = {
    "CFM M1 (100k target; best@68k)": "M1 source (68k)",
    "CFM M1 (100k exact)": "M1 (100k exact)",
    "CFM M1 (200k)": "M1 (200k)",
    "CFM M2": "M2",
    "CFM M3 / BCFM (fast)": "M3 / BCFM",
    "BD3-LM": "BD3-LM",
    "MDLM": "MDLM",
}

COLORS = {
    "M1 source (68k)": "#2458A6",
    "M1 (100k exact)": "#C43C82",
    "M1 (200k)": "#D14B3E",
    "M2": "#D99100",
    "M3 / BCFM": "#14866D",
    "BD3-LM": "#7651A8",
    "MDLM": "#8B5E3C",
}

MARKERS = {
    "M1 source (68k)": "o",
    "M1 (100k exact)": "X",
    "M1 (200k)": "h",
    "M2": "^",
    "M3 / BCFM": "D",
    "BD3-LM": "P",
    "MDLM": "s",
}

BASELINES = {"BD3-LM", "MDLM"}
BLOCKWISE = {"M2", "M3 / BCFM", "BD3-LM"}
FIRST_HITTING_COLOR = "#B83B5E"
DY = {
    "M1 source (68k)": 4,
    "M1 (100k exact)": 13,
    "M1 (200k)": -8,
    "M2": 4,
    "M3 / BCFM": -8,
    "BD3-LM": 4,
    "MDLM": -8,
}

ANNOTATION_OVERRIDES = {
    ("mauve", "BD3-LM", 16): (-17, -11),
    ("mauve", "MDLM", 128): (-24, 7),
    ("mauve", "MDLM", 256): (4, -11),
    ("gen_ppl", "BD3-LM", 16): (4, 7),
    ("gen_ppl", "MDLM", 128): (-16, -11),
    ("gen_ppl", "MDLM", 256): (4, -11),
}


def draw(ax: plt.Axes, frame: pd.DataFrame, metric: str) -> None:
    for model in COLORS:
        points = frame[
            (frame.paper_model == model)
            & ~frame.sampler.str.contains("first_hitting")
        ].sort_values("nfe")
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
            color=COLORS[model],
            linestyle="--" if model in BASELINES else "-",
            marker=MARKERS[model],
            linewidth=1.45,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=0.55,
            capsize=1.2,
            alpha=0.98,
            zorder=3,
        )
        for _, point in points.iterrows():
            dx, dy = 2.3, DY[model]
            if metric == "mauve" and model == "BD3-LM" and point.nfe in (1, 2):
                dy = -8
            if metric == "mauve" and model == "M2" and point.nfe == 2:
                dx = -7
            if metric == "gen_ppl" and model == "M3 / BCFM" and point.nfe == 8:
                dx = -7
            dx, dy = ANNOTATION_OVERRIDES.get(
                (metric, model, int(point.nfe)), (dx, dy)
            )
            prefix = "S" if model in BLOCKWISE else "N"
            label = ax.annotate(
                f"{prefix}{int(point.nfe)}",
                (point.latency_ms, point[metric]),
                xytext=(dx, dy),
                textcoords="offset points",
                color=COLORS[model],
                fontsize=5.6,
                fontweight="bold",
                annotation_clip=False,
                zorder=5,
            )
            label.set_path_effects(
                [path_effects.withStroke(linewidth=1.5, foreground="white")]
            )

    first_hitting = frame[frame.sampler.str.contains("first_hitting")]
    for _, point in first_hitting.iterrows():
        ax.errorbar(
            point.latency_ms,
            point[metric],
            xerr=[
                [point.latency_ms - point.latency_p10_ms],
                [point.latency_p90_ms - point.latency_ms],
            ],
            color=FIRST_HITTING_COLOR,
            marker="*",
            linestyle="none",
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=0.55,
            capsize=1.2,
            zorder=4,
        )
        label = ax.annotate(
            "FH",
            (point.latency_ms, point[metric]),
            xytext=(5, -16 if metric == "mauve" else 8),
            textcoords="offset points",
            color=FIRST_HITTING_COLOR,
            fontsize=5.6,
            fontweight="bold",
            annotation_clip=False,
            zorder=5,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=1.5, foreground="white")]
        )

    ax.set_xscale("log")
    ax.grid(True, which="major", color="#D8D1C5", linewidth=0.55, alpha=0.8)
    ax.grid(True, which="minor", color="#D8D1C5", linewidth=0.28, alpha=0.45)
    ax.tick_params(labelsize=6.4, width=0.6, length=2.5)
    ax.set_xlabel("Sequence latency (ms, log scale)", fontsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.65)
        ax.spines[side].set_color("#6F685F")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "axes.labelcolor": "#292724",
            "text.color": "#292724",
        }
    )
    frame = pd.read_csv(args.input)
    frame["paper_model"] = frame.model.map(MODEL_NAMES)
    if frame.paper_model.isna().any() or len(frame) != 42:
        raise RuntimeError("unexpected model labels or point count")

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25))
    draw(axes[0], frame, "mauve")
    draw(axes[1], frame, "gen_ppl")
    axes[0].set_ylabel(r"MAUVE $\uparrow$", fontsize=7.2)
    axes[0].set_ylim(-0.025, 1.025)
    axes[0].set_title("(a) Distributional quality", loc="left", fontsize=8.2, fontweight="bold")
    axes[1].set_ylabel(r"gen-PPL $\downarrow$", fontsize=7.2)
    axes[1].set_yscale("log")
    axes[1].set_title("(b) External-model perplexity", loc="left", fontsize=8.2, fontweight="bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linestyle="--" if model in BASELINES else "-",
            marker=MARKERS[model],
            linewidth=1.5,
            markersize=4.4,
            markeredgecolor="white",
            label=model,
        )
        for model in COLORS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=FIRST_HITTING_COLOR,
            linestyle="none",
            marker="*",
            markersize=6.0,
            markeredgecolor="white",
            label="BD3-LM first-hitting",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=6.0,
        columnspacing=1.0,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.78, wspace=0.24)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"wrote {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
