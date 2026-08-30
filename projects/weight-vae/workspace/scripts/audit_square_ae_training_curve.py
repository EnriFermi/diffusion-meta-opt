from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_RE = re.compile(
    r"step=(?P<step>\d+)/(?P<max_steps>\d+) "
    r"loss=(?P<loss>[-+0-9.eE]+) "
    r"behav=(?P<behavioral>[-+0-9.eE]+) "
    r"struct=(?P<structural>[-+0-9.eE]+) "
    r"b_op=(?P<behavioral_operator>[-+0-9.eE]+) "
    r"b_dir=(?P<behavioral_dir>[-+0-9.eE]+) "
    r"b_scl=(?P<behavioral_scale>[-+0-9.eE]+) "
    r"s_dir=(?P<struct_dir>[-+0-9.eE]+) "
    r"s_scl=(?P<struct_scale>[-+0-9.eE]+) "
    r"s_rec=(?P<struct_rec>[-+0-9.eE]+) "
    r"s_rel=(?P<struct_rel>[-+0-9.eE]+) "
    r"kl=(?P<kl>[-+0-9.eE]+) "
    r"kl_beta=(?P<kl_beta>[-+0-9.eE]+) "
    r"lr=(?P<lr>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the exact square-AE training lineage from local logs.")
    parser.add_argument("--initial-log", type=Path, required=True)
    parser.add_argument("--resume-log", type=Path, required=True)
    parser.add_argument("--heldout-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, default=60_000)
    parser.add_argument("--checkpoint-step", type=int, default=480_000)
    parser.add_argument("--bin-size", type=int, default=10_000)
    return parser.parse_args()


def parse_log(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = METRIC_RE.search(line)
            if match is None:
                continue
            row: dict[str, float | int | str] = {"source_log": str(path)}
            for key, value in match.groupdict().items():
                row[key] = int(value) if key in {"step", "max_steps"} else float(value)
            rows.append(row)
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def summarize_rows(rows: list[dict[str, float | int | str]], label: str) -> dict[str, float | int | str]:
    metric_names = [
        "loss",
        "behavioral",
        "behavioral_operator",
        "behavioral_dir",
        "behavioral_scale",
        "structural",
        "struct_dir",
        "struct_scale",
    ]
    payload: dict[str, float | int | str] = {
        "label": label,
        "row_count": len(rows),
        "step_min": min(int(row["step"]) for row in rows) if rows else 0,
        "step_max": max(int(row["step"]) for row in rows) if rows else 0,
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        payload[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
        payload[f"{metric}_median"] = statistics.median(values) if values else math.nan
        payload[f"{metric}_p10"] = percentile(values, 0.10)
        payload[f"{metric}_p90"] = percentile(values, 0.90)
    return payload


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = parse_log(args.initial_log)
    resumed = parse_log(args.resume_log)

    # The production run restarted from the persisted step-60k resume state.  The
    # initial log continued beyond that point, but those later updates are not in
    # the final checkpoint lineage and must not be mixed into the curve.
    lineage = [row for row in initial if int(row["step"]) <= args.resume_step]
    lineage.extend(
        row for row in resumed if args.resume_step < int(row["step"]) <= args.checkpoint_step
    )
    lineage.sort(key=lambda row: int(row["step"]))

    duplicate_steps = len(lineage) - len({int(row["step"]) for row in lineage})
    expected_steps = set(range(10, args.checkpoint_step + 1, 10))
    observed_steps = {int(row["step"]) for row in lineage}
    missing_steps = sorted(expected_steps - observed_steps)

    write_csv(args.output_dir / "training_curve.csv", lineage)

    binned_rows: list[dict[str, object]] = []
    for bin_start in range(0, args.checkpoint_step, args.bin_size):
        bin_end = min(args.checkpoint_step, bin_start + args.bin_size)
        selected = [row for row in lineage if bin_start < int(row["step"]) <= bin_end]
        binned_rows.append(summarize_rows(selected, f"({bin_start},{bin_end}]"))
    write_csv(args.output_dir / "training_curve_10k_bins.csv", binned_rows)

    comparison_windows = [
        ("first_10k", 0, 10_000),
        ("early_50k", 0, 50_000),
        ("pre_resume_50k_60k", 50_000, 60_000),
        ("late_430k_440k", 430_000, 440_000),
        ("final_470k_480k", 470_000, 480_000),
    ]
    window_rows: list[dict[str, object]] = []
    for label, start, end in comparison_windows:
        selected = [row for row in lineage if start < int(row["step"]) <= end]
        window_rows.append(summarize_rows(selected, label))
    write_csv(args.output_dir / "training_window_summary.csv", window_rows)

    heldout = json.loads(args.heldout_summary.read_text(encoding="utf-8"))
    heldout_extract = {
        "checkpoint": heldout.get("checkpoint", {}),
        "coverage": heldout.get("coverage", {}),
        "global": heldout.get("global", {}),
        "macro": heldout.get("macro", {}),
    }
    (args.output_dir / "heldout_step480k_extract.json").write_text(
        json.dumps(heldout_extract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # The summary rows identify bins by label, so use the bin end as x coordinate.
    x = [min(args.checkpoint_step, (idx + 1) * args.bin_size) for idx in range(len(binned_rows))]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_specs = [
        ("behavioral_median", "Behavioral objective", axes[0, 0]),
        ("behavioral_operator_median", "Operator RMSE term", axes[0, 1]),
        ("behavioral_dir_median", "Operator direction term", axes[1, 0]),
        ("behavioral_scale_median", "Operator scale term", axes[1, 1]),
    ]
    for key, title, axis in plot_specs:
        y = [float(row[key]) for row in binned_rows]
        lower = [float(row[key.replace("_median", "_p10")]) for row in binned_rows]
        upper = [float(row[key.replace("_median", "_p90")]) for row in binned_rows]
        axis.plot(x, y, linewidth=1.7, label="10k-bin median")
        axis.fill_between(x, lower, upper, alpha=0.18, label="10th--90th percentile")
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="zero/random direction baseline")
    for axis in axes.flat:
        axis.legend(fontsize=8)
    figure.suptitle("Square AE checkpoint lineage (local train logs)")
    figure.savefig(args.output_dir / "training_curve.png", dpi=170)
    plt.close(figure)

    audit = {
        "initial_log": str(args.initial_log),
        "resume_log": str(args.resume_log),
        "heldout_summary": str(args.heldout_summary),
        "resume_step": args.resume_step,
        "checkpoint_step": args.checkpoint_step,
        "lineage_rows": len(lineage),
        "duplicate_steps": duplicate_steps,
        "missing_logged_steps_count": len(missing_steps),
        "missing_logged_steps_head": missing_steps[:20],
        "first_step": int(lineage[0]["step"]),
        "last_step": int(lineage[-1]["step"]),
        "artifacts": {
            "curve_csv": str(args.output_dir / "training_curve.csv"),
            "bins_csv": str(args.output_dir / "training_curve_10k_bins.csv"),
            "windows_csv": str(args.output_dir / "training_window_summary.csv"),
            "plot": str(args.output_dir / "training_curve.png"),
            "heldout_extract": str(args.output_dir / "heldout_step480k_extract.json"),
        },
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
