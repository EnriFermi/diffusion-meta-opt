#!/usr/bin/env python3
"""Independent post-run review for the frozen Iteration-4 crossed audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / (
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_a_full_burg_h2048/iteration4_probe_adam_cross_production_v2"
)
BLIND_SEEDS = tuple(range(1001, 1017))
P32_BANKS = {
    "p32_a": tuple(range(1001, 1009)),
    "p32_b": tuple(range(1009, 1017)),
}
TRANSFORMS = ("raw", "zero_moment_adam", "carried_adam")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(row: pd.Series, slope_tolerances: dict[float, float]) -> str:
    analytic_margin = max(slope_tolerances.values())
    analytic = float(row["analytic_slope"])
    slope_h128 = float(row["slope_h128"])
    slope_h256 = float(row["slope_h256"])
    if (
        analytic < -analytic_margin
        and slope_h128 < -slope_tolerances[1.0 / 128.0]
        and slope_h256 < -slope_tolerances[1.0 / 256.0]
    ):
        return "stable_downhill"
    if (
        analytic > analytic_margin
        and slope_h128 > slope_tolerances[1.0 / 128.0]
        and slope_h256 > slope_tolerances[1.0 / 256.0]
    ):
        return "stable_uphill"
    return "ambiguous"


def direction_state(directions: pd.DataFrame, source: str, transform: str) -> str:
    rows = directions.loc[
        directions["source"].eq(source) & directions["transform"].eq(transform)
    ]
    if len(rows) != 1:
        raise AssertionError(f"non-unique direction key: {source}/{transform}")
    return str(rows.iloc[0]["posthoc_sign_state"])


def make_plot(
    gradients: pd.DataFrame,
    directions: pd.DataFrame,
    slope_tolerances: dict[float, float],
    output: Path,
) -> None:
    blind = directions.loc[
        directions["source"].isin([f"p4_{seed}" for seed in BLIND_SEEDS])
        & directions["transform"].eq("carried_adam")
    ].sort_values("analytic_slope")
    colors = np.where(blind["posthoc_sign_state"].eq("stable_downhill"), "#147d3f", "#c53832")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(blind))
    axes[0].bar(x, blind["analytic_slope"], color=colors, alpha=0.82, label="analytic")
    axes[0].scatter(x, blind["slope_h128"], marker="o", color="black", s=20, label="FD h=1/128")
    axes[0].scatter(x, blind["slope_h256"], marker="x", color="#f59e0b", s=28, label="FD h=1/256")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axhspan(
        -max(slope_tolerances.values()),
        max(slope_tolerances.values()),
        color="#9ca3af",
        alpha=0.18,
        label="analytic ambiguity band",
    )
    axes[0].set_xticks(x, blind["source"], rotation=65, ha="right", fontsize=8)
    axes[0].set(title="Blind P4 carried-Adam true-F slopes", ylabel="dF / d alpha")
    axes[0].legend(fontsize=8)

    sources = [f"p4_{seed}" for seed in BLIND_SEEDS] + ["p32_a", "p32_b"]
    rows = gradients.set_index("source").loc[sources].reset_index()
    colors = ["#2563eb"] * len(BLIND_SEEDS) + ["#c53832", "#c53832"]
    x = np.arange(len(rows))
    axes[1].bar(x, rows["cosine_to_exact"], color=colors)
    axes[1].set_xticks(x, rows["source"], rotation=65, ha="right", fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set(title="Gradient agreement with complete-basis exact", ylabel="cosine")

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_dir.resolve()

    decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    gradients = pd.read_csv(run / "gradient_sources.csv")
    directions = pd.read_csv(run / "direction_diagnostics.csv")
    fd = pd.read_csv(run / "direction_finite_differences.csv")
    lines = pd.read_csv(run / "extended_line_profiles.csv")

    checks: dict[str, bool] = {}
    checks["manifest_hashes_match"] = all(
        sha256_file(run / name) == expected
        for name, expected in manifest["artifacts"].items()
    )
    checks["production_valid_and_all_gates_true"] = bool(
        decision["valid"] and all(decision["validity_gates"].values())
    )

    expected_sources = {
        "exact",
        "p4_59",
        "p4_67",
        *(f"p4_{seed}" for seed in BLIND_SEEDS),
        *P32_BANKS,
    }
    checks["gradient_key_coverage"] = bool(
        len(gradients) == 21
        and set(gradients["source"]) == expected_sources
        and not gradients["source"].duplicated().any()
    )
    expected_direction_keys = {
        (source, transform) for source in expected_sources for transform in TRANSFORMS
    }
    checks["direction_key_coverage"] = bool(
        len(directions) == 63
        and set(zip(directions["source"], directions["transform"], strict=True))
        == expected_direction_keys
        and not directions.duplicated(["source", "transform"]).any()
    )
    checks["fd_key_coverage"] = bool(
        len(fd) == 126
        and not fd.duplicated(["source", "transform", "alpha"]).any()
        and set(np.round(fd["alpha"], 12))
        == {round(1.0 / 128.0, 12), round(1.0 / 256.0, 12)}
    )

    numeric_frames = (gradients, directions, fd, lines)
    checks["all_numeric_finite"] = all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        for frame in numeric_frames
    )

    f_tolerance = float(decision["f_tolerance"])
    slope_tolerances = {
        1.0 / 128.0: f_tolerance / (1.0 / 128.0),
        1.0 / 256.0: f_tolerance / (1.0 / 256.0),
    }
    directions["posthoc_sign_state"] = directions.apply(
        classify, axis=1, slope_tolerances=slope_tolerances
    )
    checks["all_sign_states_reproduced"] = bool(
        directions["posthoc_sign_state"].eq(directions["sign_state"]).all()
    )
    checks["fd_rows_match_direction_summaries"] = bool(
        all(
            np.isclose(
                float(row["central_slope"]),
                float(
                    directions.loc[
                        directions["direction"].eq(row["direction"]),
                        "slope_h128" if np.isclose(row["alpha"], 1.0 / 128.0) else "slope_h256",
                    ].iloc[0]
                ),
                rtol=0.0,
                atol=1e-12,
            )
            for _, row in fd.iterrows()
        )
    )

    expected_line_directions = set(
        directions.loc[directions["all_slopes_negative"].eq(1), "direction"]
    )
    observed_line_counts = lines.groupby("direction").size().to_dict()
    checks["line_coverage"] = bool(
        set(observed_line_counts) == expected_line_directions
        and all(count == 13 for count in observed_line_counts.values())
    )

    pool_closer: dict[str, bool] = {}
    pool_metrics: dict[str, dict[str, float]] = {}
    for pool, seeds in P32_BANKS.items():
        pool_row = gradients.loc[gradients["source"].eq(pool)].iloc[0]
        block = gradients.loc[gradients["source"].isin([f"p4_{seed}" for seed in seeds])]
        pool_metrics[pool] = {
            "pool_cosine": float(pool_row["cosine_to_exact"]),
            "block_median_cosine": float(block["cosine_to_exact"].median()),
            "pool_relative_error": float(pool_row["relative_error_to_exact"]),
            "block_median_relative_error": float(block["relative_error_to_exact"].median()),
        }
        pool_closer[pool] = bool(
            pool_metrics[pool]["pool_cosine"] > pool_metrics[pool]["block_median_cosine"]
            and pool_metrics[pool]["pool_relative_error"]
            < pool_metrics[pool]["block_median_relative_error"]
        )
    checks["both_pools_closer_to_exact"] = all(pool_closer.values())

    blind_sources = [f"p4_{seed}" for seed in BLIND_SEEDS]
    finite_probe: dict[str, bool] = {}
    for transform in TRANSFORMS:
        blind_states = set(
            directions.loc[
                directions["source"].isin(blind_sources)
                & directions["transform"].eq(transform),
                "posthoc_sign_state",
            ]
        )
        finite_probe[transform] = bool(
            direction_state(directions, "exact", transform) == "stable_downhill"
            and all(
                direction_state(directions, pool, transform) == "stable_downhill"
                for pool in P32_BANKS
            )
            and {"stable_downhill", "stable_uphill"}.issubset(blind_states)
            and all(pool_closer.values())
        )

    history_patterns = {
        source: bool(
            direction_state(directions, source, "raw") == "stable_downhill"
            and direction_state(directions, source, "zero_moment_adam") == "stable_downhill"
            and direction_state(directions, source, "carried_adam") == "stable_uphill"
        )
        for source in ("exact", *P32_BANKS)
    }
    carried_history = history_patterns["exact"] or all(
        history_patterns[pool] for pool in P32_BANKS
    )
    diagonal_patterns = {
        pool: bool(
            direction_state(directions, pool, "raw") == "stable_downhill"
            and direction_state(directions, pool, "zero_moment_adam") == "stable_uphill"
        )
        for pool in P32_BANKS
    }
    adam_diagonal = bool(
        direction_state(directions, "exact", "zero_moment_adam") == "stable_downhill"
        and all(diagonal_patterns.values())
    )

    radius_directions: list[str] = []
    for direction, group in lines.groupby("direction"):
        row = directions.loc[directions["direction"].eq(direction)].iloc[0]
        original = bool(
            ((group["in_original_grid"].eq(1)) & group["strict_decrease"].eq(1)).any()
        )
        extended = bool(
            ((group["in_original_grid"].eq(0)) & group["strict_decrease"].eq(1)).any()
        )
        if bool(row["all_slopes_negative"]) and not original and extended:
            radius_directions.append(str(direction))
    radius_residual = "p4_59__carried_adam" in radius_directions

    exact_raw = directions.loc[
        directions["source"].eq("exact") & directions["transform"].eq("raw")
    ].iloc[0]
    exact_raw_lines = lines.loc[lines["direction"].eq("exact__raw")]
    stationarity = bool(
        abs(float(exact_raw["analytic_slope"])) <= max(slope_tolerances.values())
        and abs(float(exact_raw["slope_h128"])) <= slope_tolerances[1.0 / 128.0]
        and abs(float(exact_raw["slope_h256"])) <= slope_tolerances[1.0 / 256.0]
        and float(exact_raw_lines["delta_f"].min()) >= -f_tolerance
    )

    primary_ambiguous = any(
        direction_state(directions, source, transform) == "ambiguous"
        for source in ("exact", *P32_BANKS)
        for transform in TRANSFORMS
    )
    candidate: str | None = None
    if not primary_ambiguous:
        if adam_diagonal and finite_probe["raw"]:
            candidate = "p32_plus_raw_sgd_training"
        elif adam_diagonal:
            candidate = "raw_sgd_training"
        elif finite_probe["raw"] and carried_history:
            candidate = "p32_plus_zero_moment_training"
        elif carried_history:
            candidate = "zero_moment_optimizer_training"
        elif finite_probe["carried_adam"]:
            candidate = "p32_variance_controlled_training"
        elif (
            direction_state(directions, "exact", "carried_adam") == "stable_downhill"
            and all(
                direction_state(directions, pool, "carried_adam") == "stable_uphill"
                for pool in P32_BANKS
            )
        ):
            candidate = "complete_basis_training"
        elif radius_residual:
            candidate = "extended_line_search"

    mechanisms = {
        "finite_probe_by_transform": finite_probe,
        "carried_history_supported": bool(carried_history),
        "adam_diagonal_supported": adam_diagonal,
        "radius_supported_directions": sorted(radius_directions),
        "radius_supported_for_residual": radius_residual,
        "stationarity_supported": stationarity,
        "primary_sign_ambiguity": primary_ambiguous,
    }
    checks["mechanism_decision_reproduced"] = bool(
        finite_probe == decision["mechanisms"]["finite_probe_by_transform"]
        and bool(carried_history) == decision["mechanisms"]["carried_history_supported"]
        and adam_diagonal == decision["mechanisms"]["adam_diagonal_supported"]
        and sorted(radius_directions)
        == sorted(decision["mechanisms"]["radius_supported_directions"])
        and radius_residual == decision["mechanisms"]["radius_supported_for_residual"]
        and stationarity == decision["mechanisms"]["stationarity_supported"]
        and primary_ambiguous == decision["mechanisms"]["primary_sign_ambiguity"]
    )
    checks["iteration5_candidate_reproduced"] = candidate == decision["iteration5_candidate"]

    blind_carried = directions.loc[
        directions["source"].isin(blind_sources)
        & directions["transform"].eq("carried_adam")
    ]
    blind_counts = {
        str(key): int(value)
        for key, value in blind_carried["posthoc_sign_state"].value_counts().items()
    }
    primary_slopes = directions.loc[
        directions["source"].isin(["exact", *P32_BANKS])
    ][
        ["source", "transform", "analytic_slope", "slope_h128", "slope_h256", "posthoc_sign_state"]
    ].to_dict(orient="records")

    make_plot(
        gradients,
        directions,
        slope_tolerances,
        run / "posthoc_key_discriminator.png",
    )
    result: dict[str, Any] = {
        "valid": bool(all(checks.values())),
        "checks": checks,
        "check_count": len(checks),
        "blind_p4_carried_sign_counts": blind_counts,
        "pool_metrics": pool_metrics,
        "primary_slopes": primary_slopes,
        "mechanisms": mechanisms,
        "iteration5_candidate": candidate,
        "plot": str(run / "posthoc_key_discriminator.png"),
    }
    (run / "posthoc_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
