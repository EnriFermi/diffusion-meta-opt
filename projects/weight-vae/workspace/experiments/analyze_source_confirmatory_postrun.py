#!/usr/bin/env python3
"""Post-run diagnosis for the source-only G0/G1 confirmatory experiment.

This script never imports the Weight-AE or any target-domain asset.  It works
only from the sufficient statistics already written by
``source_confirmatory_g01_gate.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
METHODS = (
    "ae_global_c0",
    "ae_cell_mean_c0",
    "ae_cell_medoid_c0",
    "ae_native_c",
    "ae_zero_c",
    "identity",
    "zero",
)
FIXED_METHODS = ("ae_global_c0", "ae_cell_mean_c0", "ae_cell_medoid_c0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=26_081_699)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio_error(group: pd.DataFrame, prefix: str, gain: float) -> float:
    target = float(group[f"{prefix}_target"].sum())
    pred = float(group[f"{prefix}_pred"].sum())
    dot = float(group[f"{prefix}_dot"].sum())
    return (target - 2.0 * gain * dot + gain * gain * pred) / target


def cosine(group: pd.DataFrame, prefix: str) -> float:
    target = float(group[f"{prefix}_target"].sum())
    pred = float(group[f"{prefix}_pred"].sum())
    dot = float(group[f"{prefix}_dot"].sum())
    return dot / math.sqrt(target * pred)


def aggregate_draws(
    group: pd.DataFrame,
    draw_indices: np.ndarray,
    error_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    indexed = group.set_index(["depth", "role"])
    role_num = []
    role_den = []
    for role in ROLES:
        numerators = np.asarray([indexed.loc[(depth, role), error_key] for depth in range(12)])
        denominators = np.asarray([indexed.loc[(depth, role), "x_target"] for depth in range(12)])
        role_num.append(numerators[draw_indices].sum(axis=1))
        role_den.append(denominators[draw_indices].sum(axis=1))
    num = np.stack(role_num, axis=1)
    den = np.stack(role_den, axis=1)
    macro = (num / den).mean(axis=1)
    micro = num.sum(axis=1) / den.sum(axis=1)
    return macro, micro


def point_aggregates(group: pd.DataFrame, error_key: str) -> tuple[float, float]:
    role_values = []
    for role in ROLES:
        role_group = group[group["role"] == role]
        role_values.append(float(role_group[error_key].sum() / role_group["x_target"].sum()))
    macro = float(np.mean(role_values))
    micro = float(group[error_key].sum() / group["x_target"].sum())
    return macro, micro


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("source-postrun")
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.run_dir / "matrix_metrics.csv"
    decisions_path = args.run_dir / "decisions.json"
    log.info(
        "resolved_config run_dir=%s output_dir=%s device=cpu dtype=float64 seed=%s "
        "bootstrap_draws=%s cache_mode=read_existing_sufficient_stats_only",
        args.run_dir,
        args.output_dir,
        args.seed,
        args.bootstrap_draws,
    )

    log.info("stage=input_loading matrix_metrics=%s", matrix_path)
    frame = pd.read_csv(matrix_path)
    with decisions_path.open() as handle:
        decisions = json.load(handle)

    log.info("stage=validity_checks rows=%s", len(frame))
    expected_rows = 2 * len(METHODS) * 72
    duplicate_count = int(frame.duplicated(["tiling_seed", "method", "depth", "role"]).sum())
    numeric = frame.select_dtypes(include=[np.number])
    nonfinite_count = int((~np.isfinite(numeric.to_numpy())).sum())
    seeds = sorted(int(value) for value in frame["tiling_seed"].unique())
    validity = {
        "expected_rows": expected_rows,
        "observed_rows": int(len(frame)),
        "duplicate_key_rows": duplicate_count,
        "nonfinite_numeric_values": nonfinite_count,
        "tiling_seeds": seeds,
        "methods": sorted(str(value) for value in frame["method"].unique()),
        "roles": sorted(str(value) for value in frame["role"].unique()),
        "depths": sorted(int(value) for value in frame["depth"].unique()),
    }
    if (
        len(frame) != expected_rows
        or duplicate_count
        or nonfinite_count
        or tuple(validity["methods"]) != tuple(sorted(METHODS))
        or validity["depths"] != list(range(12))
        or tuple(validity["roles"]) != tuple(sorted(ROLES))
    ):
        raise RuntimeError(f"matrix-metric validity failure: {validity}")

    log.info("stage=gain_transfer_diagnostics")
    gain_rows: list[dict[str, float | int | str]] = []
    for tiling_seed in seeds:
        for method in FIXED_METHODS + ("ae_native_c", "ae_zero_c"):
            method_group = frame[(frame["tiling_seed"] == tiling_seed) & (frame["method"] == method)]
            for role in ROLES:
                group = method_group[method_group["role"] == role]
                source_gain = float(group["gain"].iloc[0])
                heldout_x_gain = float(group["x_dot"].sum() / group["x_pred"].sum())
                heldout_w_gain = float(group["w_dot"].sum() / group["w_pred"].sum())
                x_cosine = cosine(group, "x")
                w_cosine = cosine(group, "w")
                gain_rows.append(
                    {
                        "tiling_seed": tiling_seed,
                        "method": method,
                        "role": role,
                        "source_operator_gain": source_gain,
                        "heldout_oracle_operator_gain_diagnostic": heldout_x_gain,
                        "heldout_oracle_weight_gain_diagnostic": heldout_w_gain,
                        "source_over_heldout_operator_gain": source_gain / heldout_x_gain,
                        "raw_E_X": ratio_error(group, "x", 1.0),
                        "source_gain_E_X": ratio_error(group, "x", source_gain),
                        "heldout_oracle_E_X_diagnostic": ratio_error(group, "x", heldout_x_gain),
                        "raw_E_W": ratio_error(group, "w", 1.0),
                        "source_operator_gain_E_W": ratio_error(group, "w", source_gain),
                        "heldout_oracle_E_W_diagnostic": ratio_error(group, "w", heldout_w_gain),
                        "operator_cosine": x_cosine,
                        "weight_cosine": w_cosine,
                        "operator_direction_floor": 1.0 - x_cosine * x_cosine,
                        "weight_direction_floor": 1.0 - w_cosine * w_cosine,
                    }
                )
    gain_frame = pd.DataFrame(gain_rows)
    gain_frame.to_csv(args.output_dir / "gain_transfer_diagnostics.csv", index=False)

    log.info("stage=paired_block_bootstrap draws=%s", args.bootstrap_draws)
    rng = np.random.default_rng(args.seed)
    draw_indices = rng.integers(0, 12, size=(args.bootstrap_draws, 12))
    pair_rows: list[dict[str, float | int | str]] = []
    references = ("zero", "ae_zero_c", "ae_global_c0", "ae_cell_medoid_c0", "ae_native_c")
    for tiling_seed in seeds:
        primary = frame[(frame["tiling_seed"] == tiling_seed) & (frame["method"] == "ae_cell_mean_c0")]
        for calibration, error_key in (("raw", "x_error"), ("source_gain", "x_error_scaled")):
            primary_macro, primary_micro = aggregate_draws(primary, draw_indices, error_key)
            primary_point = point_aggregates(primary, error_key)
            for reference_method in references:
                reference = frame[
                    (frame["tiling_seed"] == tiling_seed) & (frame["method"] == reference_method)
                ]
                reference_macro, reference_micro = aggregate_draws(reference, draw_indices, error_key)
                reference_point = point_aggregates(reference, error_key)
                for aggregation, numerator, denominator, point_num, point_den in (
                    ("macro", primary_macro, reference_macro, primary_point[0], reference_point[0]),
                    ("micro", primary_micro, reference_micro, primary_point[1], reference_point[1]),
                ):
                    samples = numerator / denominator
                    pair_rows.append(
                        {
                            "tiling_seed": tiling_seed,
                            "calibration": calibration,
                            "primary_method": "ae_cell_mean_c0",
                            "reference_method": reference_method,
                            "aggregation": aggregation,
                            "point_primary_over_reference": point_num / point_den,
                            "bootstrap_mean": float(samples.mean()),
                            "l95": float(np.quantile(samples, 0.05)),
                            "u95": float(np.quantile(samples, 0.95)),
                            "draws": args.bootstrap_draws,
                        }
                    )
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(args.output_dir / "paired_operator_comparisons.csv", index=False)

    log.info("stage=tiling_stability")
    stability_rows: list[dict[str, float | str]] = []
    seed_a, seed_b = seeds
    for method in METHODS:
        left = frame[(frame["tiling_seed"] == seed_a) & (frame["method"] == method)].set_index(
            ["depth", "role"]
        )
        right = frame[(frame["tiling_seed"] == seed_b) & (frame["method"] == method)].set_index(
            ["depth", "role"]
        )
        for metric in (
            "raw_E_W",
            "raw_E_X",
            "calibrated_E_W",
            "calibrated_E_X",
            "raw_weight_cosine",
            "raw_operator_cosine",
            "raw_weight_norm_ratio",
            "raw_operator_norm_ratio",
        ):
            difference = (left[metric] - right[metric]).abs()
            correlation_defined = bool(left[metric].std() > 0.0 and right[metric].std() > 0.0)
            pearson_r = float(left[metric].corr(right[metric])) if correlation_defined else 1.0
            stability_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "pearson_r": pearson_r,
                    "pearson_defined": correlation_defined,
                    "mean_absolute_difference": float(difference.mean()),
                    "max_absolute_difference": float(difference.max()),
                }
            )
    stability_frame = pd.DataFrame(stability_rows)
    stability_frame.to_csv(args.output_dir / "tiling_stability.csv", index=False)

    log.info("stage=plotting")
    primary_gain = gain_frame[gain_frame["method"] == "ae_cell_mean_c0"]
    role_summary = primary_gain.groupby("role", sort=False).mean(numeric_only=True).loc[list(ROLES)]
    x = np.arange(len(ROLES))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    width = 0.25
    axes[0].bar(x - width / 2, role_summary["operator_cosine"], width, label="operator cosine")
    axes[0].bar(x + width / 2, role_summary["weight_cosine"], width, label="weight cosine")
    axes[0].set_title("Direction retained by fixed-C decode")
    axes[0].set_ylabel("cosine")
    axes[0].legend()
    axes[1].bar(x - width, role_summary["raw_E_X"], width, label="raw")
    axes[1].bar(x, role_summary["source_gain_E_X"], width, label="source-gain")
    axes[1].bar(
        x + width,
        role_summary["heldout_oracle_E_X_diagnostic"],
        width,
        label="heldout oracle (diagnostic)",
    )
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1].set_title("Operator error: amplitude diagnosis")
    axes[1].set_ylabel("relative error")
    axes[1].legend()
    axes[2].bar(x - width, role_summary["raw_E_W"], width, label="raw")
    axes[2].bar(x, role_summary["source_operator_gain_E_W"], width, label="source operator-gain")
    axes[2].bar(
        x + width,
        role_summary["heldout_oracle_E_W_diagnostic"],
        width,
        label="heldout weight oracle (diagnostic)",
    )
    axes[2].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[2].set_title("Weight error: shared-gain mismatch")
    axes[2].set_ylabel("relative error")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, [role.replace("attn_", "a_").replace("ffn_", "f_") for role in ROLES], rotation=35)
        axis.grid(axis="y", alpha=0.2)
    plot_path = args.output_dir / "scale_direction_diagnosis.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    primary_stability = stability_frame[
        (stability_frame["method"] == "ae_cell_mean_c0")
        & (stability_frame["metric"].isin(["raw_E_X", "calibrated_E_X", "raw_E_W", "calibrated_E_W"]))
    ]
    paired_source = pair_frame[
        (pair_frame["calibration"] == "source_gain")
        & (pair_frame["reference_method"].isin(["zero", "ae_zero_c", "ae_native_c", "ae_global_c0"]))
    ]
    summary = {
        "status": "DIAGNOSTIC_ONLY_TARGET_REMAINS_SEALED",
        "input_sha256": {
            "matrix_metrics.csv": sha256(matrix_path),
            "decisions.json": sha256(decisions_path),
        },
        "validity": validity,
        "original_decision": decisions,
        "primary_gain_diagnostics": gain_frame[gain_frame["method"] == "ae_cell_mean_c0"].to_dict("records"),
        "primary_tiling_stability": primary_stability.to_dict("records"),
        "paired_source_gain_operator_comparisons": paired_source.to_dict("records"),
    }
    summary_path = args.output_dir / "diagnostic_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    readme = f"""# Post-run diagnosis: held-out source G0/G1

Status: **diagnostic only; target remains sealed**.

The completed evaluator decision remains `G1=false`; this analysis does not
retroactively change its gate.  It decomposes the failure using only stored
float64 sufficient statistics.  Input `matrix_metrics.csv` SHA256:
`{sha256(matrix_path)}`.

## Mechanical validity

- rows: {len(frame)}/{expected_rows}; duplicate keys: {duplicate_count}; non-finite numeric values: {nonfinite_count};
- two tilings, seven methods, 12 blocks, and six roles are complete;
- no Weight-AE forward and no target/data2vec read occurs in this analysis.

## Interpretation boundary

`gain_transfer_diagnostics.csv` compares the frozen source-fitted operator
gain with held-out oracle gains.  Oracle columns are mechanism diagnostics,
not admissible paper endpoints.  `paired_operator_comparisons.csv` uses paired
block resampling and keeps all roles within each sampled block.

Artifacts:

- `gain_transfer_diagnostics.csv`;
- `paired_operator_comparisons.csv`;
- `tiling_stability.csv`;
- `scale_direction_diagnosis.png`;
- `diagnostic_summary.json`.
"""
    readme_path = args.output_dir / "README.md"
    readme_path.write_text(readme)
    log.info(
        "stage=output_writing artifacts=%s elapsed_s=%.1f",
        [str(path) for path in (readme_path, summary_path, plot_path)],
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main()
