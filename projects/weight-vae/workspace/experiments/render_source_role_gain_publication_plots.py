#!/usr/bin/env python3
"""Render publication-readable plots from the sealed role-gain CSV artifacts.

This script is intentionally plot-only.  It validates every artifact recorded in
the formal analysis manifest before and after rendering, reads three sealed CSVs,
and writes figures to a separate sibling directory.  It never writes inside the
formal analysis directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/coder/project")
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "artifacts/crossmodal_united_structure/source_role_gain_mechanism_20260816"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/crossmodal_united_structure/"
    "source_role_gain_mechanism_publication_plots_20260816_v2"
)

ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
ROLE_SHORT = {
    "attn_query": "Q",
    "attn_key": "K",
    "attn_value": "V",
    "attn_output": "O",
    "ffn_up": "FFN up",
    "ffn_down": "FFN down",
}
CODES = ("correct", "permuted_within_row", "zero")
CODE_COLORS = {
    "correct": "#2F6DAE",
    "permuted_within_row": "#D97721",
    "zero": "#74849A",
}
CODE_LABELS = {
    "correct": "correct latent code",
    "permuted_within_row": "within-row code permutation",
    "zero": "zero_code: decode(z=0, C), nonzero decoder prior",
}
LITERAL_ZERO_LABEL = r"literal-zero output ($\hat{Y}=0$; $E_X=1$)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_one(frame: pd.DataFrame, **conditions: Any) -> pd.Series:
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one row for {conditions}; found {len(selected)}")
    return selected.iloc[0]


def validate_manifest(source: Path) -> tuple[Path, str, list[dict[str, Any]]]:
    manifest_path = source / "artifact_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    payload = json.loads(manifest_path.read_text())
    entries = payload.get("artifacts")
    if not isinstance(entries, list) or payload.get("count") != len(entries):
        raise RuntimeError("Malformed formal artifact_manifest.json")

    validated: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for entry in entries:
        formal_path = Path(entry["path"])
        name = formal_path.name
        if name in seen_names:
            raise RuntimeError(f"Duplicate basename in formal manifest: {name}")
        seen_names.add(name)
        candidate = source / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Formal artifact missing through source path: {candidate}")
        actual_sha = sha256_file(candidate)
        actual_bytes = candidate.stat().st_size
        if actual_sha != entry["sha256"] or actual_bytes != int(entry["bytes"]):
            raise RuntimeError(
                f"Formal artifact hash/size mismatch for {candidate}: "
                f"expected {entry['sha256']} / {entry['bytes']}, "
                f"got {actual_sha} / {actual_bytes}"
            )
        validated.append(
            {
                "name": name,
                "path": str(candidate.resolve()),
                "sha256": actual_sha,
                "bytes": actual_bytes,
            }
        )
    return manifest_path, manifest_sha, validated


def load_csv(source: Path, name: str, validated: Iterable[dict[str, Any]]) -> pd.DataFrame:
    validated_by_name = {row["name"]: row for row in validated}
    if name not in validated_by_name:
        raise RuntimeError(f"CSV is not sealed by the formal manifest: {name}")
    frame = pd.read_csv(source / name)
    if frame.empty:
        raise RuntimeError(f"Sealed CSV unexpectedly empty: {name}")
    return frame


def build_presentation_aggregates(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Disambiguate pooled energy sums from the reported aggregation statistic."""
    presentation = aggregates.rename(
        columns={
            "x_target": "sum_x_target_energy",
            "x_error": "sum_x_error_energy",
            "w_target": "sum_w_target_energy",
            "w_error": "sum_w_error_energy",
        }
    ).copy()
    presentation["energy_ratio_from_sums_E_X"] = (
        presentation["sum_x_error_energy"] / presentation["sum_x_target_energy"]
    )
    presentation["energy_ratio_from_sums_E_W"] = (
        presentation["sum_w_error_energy"] / presentation["sum_w_target_energy"]
    )
    presentation["E_X_aggregation_definition"] = presentation["aggregation"].map(
        lambda value: (
            "unweighted_mean_of_six_role_level_E_X"
            if value == "macro"
            else (
                "pooled_energy_ratio_over_all_72_matrices"
                if value == "micro"
                else "pooled_energy_ratio_over_12_depths_for_this_role"
            )
        )
    )
    presentation["E_W_aggregation_definition"] = presentation["aggregation"].map(
        lambda value: (
            "unweighted_mean_of_six_role_level_E_W"
            if value == "macro"
            else (
                "pooled_energy_ratio_over_all_72_matrices"
                if value == "micro"
                else "pooled_energy_ratio_over_12_depths_for_this_role"
            )
        )
    )

    key = [
        "tiling_seed",
        "encoder_condition",
        "code_condition",
        "decoder_condition",
        "calibration",
    ]
    for _, group in presentation.groupby(key, sort=False):
        macro = select_one(group, aggregation="macro")
        micro = select_one(group, aggregation="micro")
        roles = group[group["aggregation"].isin(ROLES)]
        if len(roles) != len(ROLES):
            raise RuntimeError("Presentation aggregate validation found an incomplete role group")
        checks = (
            (float(macro["E_X"]), float(roles["E_X"].mean()), "macro E_X"),
            (float(macro["E_W"]), float(roles["E_W"].mean()), "macro E_W"),
            (
                float(macro["energy_ratio_from_sums_E_X"]),
                float(micro["E_X"]),
                "macro-row X sum ratio vs micro E_X",
            ),
            (
                float(macro["energy_ratio_from_sums_E_W"]),
                float(micro["E_W"]),
                "macro-row W sum ratio vs micro E_W",
            ),
        )
        for actual, expected, label in checks:
            if not np.isclose(actual, expected, rtol=2e-13, atol=2e-13):
                raise RuntimeError(f"Presentation aggregate validation failed: {label}")
    nonmacro = presentation[presentation["aggregation"] != "macro"]
    if not np.allclose(
        nonmacro["energy_ratio_from_sums_E_X"], nonmacro["E_X"], rtol=2e-13, atol=2e-13
    ):
        raise RuntimeError("Non-macro E_X does not match its explicitly labeled energy ratio")
    if not np.allclose(
        nonmacro["energy_ratio_from_sums_E_W"], nonmacro["E_W"], rtol=2e-13, atol=2e-13
    ):
        raise RuntimeError("Non-macro E_W does not match its explicitly labeled energy ratio")
    return presentation


def build_extended_gain_sensitivity(
    aggregates: pd.DataFrame, sensitivity: pd.DataFrame, seeds: list[int]
) -> tuple[pd.DataFrame, float]:
    """Evaluate the exact quadratic error identity on a range containing g=1."""
    formal = sensitivity[sensitivity["context"] == "cell/cell"].copy()
    rows: list[dict[str, Any]] = []
    max_formal_identity_error = 0.0
    for tiling_seed in seeds:
        for role in ROLES:
            formal_role = formal[
                (formal["tiling_seed"] == tiling_seed) & (formal["role"] == role)
            ].sort_values("gain")
            if formal_role.empty:
                raise RuntimeError(f"Missing formal sensitivity rows for {tiling_seed}/{role}")
            aggregate = select_one(
                aggregates,
                tiling_seed=tiling_seed,
                encoder_condition="cell",
                decoder_condition="cell",
                code_condition="correct",
                calibration="raw",
                aggregation=role,
            )
            cosine = float(aggregate["operator_cosine"])
            raw_norm_ratio = float(aggregate["operator_norm_ratio"])
            angular_floor = 1.0 - cosine**2
            formal_gains = formal_role["gain"].to_numpy(dtype=np.float64)
            expected_formal_error = angular_floor + (formal_gains * raw_norm_ratio - cosine) ** 2
            max_formal_identity_error = max(
                max_formal_identity_error,
                float(np.max(np.abs(expected_formal_error - formal_role["E_X"].to_numpy()))),
            )
            source_gain = float(formal_role["source_gain"].iloc[0])
            oracle_gain = float(formal_role["heldout_oracle_gain"].iloc[0])
            formal_max = float(formal_role["gain"].max())
            presentation_max = max(formal_max, 1.10)
            for gain in np.linspace(0.0, presentation_max, 161):
                radial_penalty = float((gain * raw_norm_ratio - cosine) ** 2)
                rows.append(
                    {
                        "tiling_seed": tiling_seed,
                        "context": "cell/cell",
                        "role": role,
                        "gain": float(gain),
                        "gain_multiplier": float(gain / source_gain),
                        "source_gain": source_gain,
                        "heldout_oracle_gain": oracle_gain,
                        "formal_curve_max_gain": formal_max,
                        "presentation_curve_max_gain": presentation_max,
                        "E_X": angular_floor + radial_penalty,
                        "angular_floor": angular_floor,
                        "radial_penalty": radial_penalty,
                        "derivation": "exact_quadratic_identity_from_sealed_raw_cosine_and_norm_ratio",
                    }
                )
    if max_formal_identity_error > 5e-13:
        raise RuntimeError(
            "Sealed sensitivity CSV does not match the exact quadratic identity: "
            f"max_abs_error={max_formal_identity_error:.3e}"
        )
    return pd.DataFrame(rows), max_formal_identity_error


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    if not path.is_file() or path.stat().st_size < 10_000:
        raise RuntimeError(f"Rendered PNG is missing or suspiciously small: {path}")


def plot_fixed_context_error_rescue(
    aggregates: pd.DataFrame, seeds: list[int], output: Path, dpi: int
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), sharey=True)
    calibrations = ("raw", "source_role_matched")
    calibration_labels = ("raw decoder output (g=1)", "source-frozen role gains")
    x = np.arange(2, dtype=float)
    width = 0.24
    for axis, tiling_seed in zip(axes, seeds):
        base = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["aggregation"] == "macro")
        ]
        for index, code in enumerate(CODES):
            values = [
                float(
                    select_one(
                        base,
                        code_condition=code,
                        calibration=calibration,
                    )["E_X"]
                )
                for calibration in calibrations
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                color=CODE_COLORS[code],
                label=CODE_LABELS[code],
            )
        axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1.1, label=LITERAL_ZERO_LABEL)
        axis.set_xticks(x, calibration_labels)
        axis.set_title(f"Tiling seed {tiling_seed}")
        axis.set_xlabel("output scaling")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(r"operator error $E_X$ (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Fixed-context operator error under frozen scaling controls", y=0.98)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.25, wspace=0.12)
    path = output / "fixed_context_error_rescue.png"
    save_figure(fig, path, dpi)
    return path


def plot_fixed_context_role_rescue(
    aggregates: pd.DataFrame, seeds: list[int], output: Path, dpi: int
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), sharey=True)
    x = np.arange(len(ROLES), dtype=float)
    width = 0.25
    specs = (
        ("correct", "raw", "correct code, raw output (g=1)", "#9CC7DE"),
        ("correct", "source_role_matched", "correct code + source-frozen role gains", "#2F6DAE"),
        (
            "zero",
            "source_role_matched",
            "zero_code + same gains (nonzero decoder prior)",
            "#74849A",
        ),
    )
    for axis, tiling_seed in zip(axes, seeds):
        base = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["aggregation"].isin(ROLES))
        ]
        for index, (code, calibration, label, color) in enumerate(specs):
            values = [
                float(
                    select_one(
                        base,
                        code_condition=code,
                        calibration=calibration,
                        aggregation=role,
                    )["E_X"]
                )
                for role in ROLES
            ]
            axis.bar(x + (index - 1) * width, values, width, color=color, label=label)
        axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1.1, label=LITERAL_ZERO_LABEL)
        axis.set_xticks(x, [ROLE_SHORT[role] for role in ROLES], rotation=14, ha="right")
        axis.set_title(f"Tiling seed {tiling_seed}")
        axis.set_xlabel("matrix role")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(r"operator error $E_X$ (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Role-wise fixed-context operator error", y=0.98)
    fig.subplots_adjust(left=0.068, right=0.99, top=0.85, bottom=0.25, wspace=0.10)
    path = output / "fixed_context_role_rescue.png"
    save_figure(fig, path, dpi)
    return path


def plot_radial_angular_decomposition(
    aggregates: pd.DataFrame, seeds: list[int], output: Path, dpi: int
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)
    radial_role_short = {
        "attn_query": "Q",
        "attn_key": "K",
        "attn_value": "V",
        "attn_output": "O",
        "ffn_up": "Up",
        "ffn_down": "Down",
    }
    for axis, tiling_seed in zip(axes, seeds):
        base = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["code_condition"] == "correct")
            & (aggregates["aggregation"].isin(ROLES))
        ]
        positions: list[float] = []
        tick_labels: list[str] = []
        angular: list[float] = []
        radial: list[float] = []
        for role_index, role in enumerate(ROLES):
            for cal_index, calibration in enumerate(("raw", "source_role_matched")):
                row = select_one(base, aggregation=role, calibration=calibration)
                positions.append(role_index * 2.6 + cal_index)
                tick_labels.append(
                    f"{radial_role_short[role]}\n{'raw' if cal_index == 0 else 'role gain'}"
                )
                angular.append(float(row["operator_angular_floor"]))
                radial.append(float(row["operator_radial_penalty"]))
        axis.bar(positions, angular, color="#4E79A7", label=r"angular floor $1-c^2$")
        axis.bar(
            positions,
            radial,
            bottom=angular,
            color="#F28E2B",
            label=r"radial penalty $(gr-c)^2$",
        )
        axis.set_xticks(positions, tick_labels, fontsize=8.5)
        axis.set_title(f"Tiling seed {tiling_seed}")
        axis.set_xlabel("matrix role and output scaling")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(r"components of operator error $E_X$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Angular and radial components under fixed context", y=0.98)
    fig.subplots_adjust(left=0.068, right=0.99, top=0.85, bottom=0.20, wspace=0.10)
    path = output / "radial_angular_decomposition.png"
    save_figure(fig, path, dpi)
    return path


def plot_directional_operator_cosine(
    aggregates: pd.DataFrame, seeds: list[int], output: Path, dpi: int
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.7), sharey=True)
    x = np.arange(len(ROLES), dtype=float)
    width = 0.25
    for axis, tiling_seed in zip(axes, seeds):
        base = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["calibration"] == "raw")
            & (aggregates["aggregation"].isin(ROLES))
        ]
        for index, code in enumerate(CODES):
            values = [
                float(select_one(base, code_condition=code, aggregation=role)["operator_cosine"])
                for role in ROLES
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                color=CODE_COLORS[code],
                label=CODE_LABELS[code],
            )
        axis.axhline(0.0, color="#222222", linewidth=0.9)
        axis.set_xticks(x, [ROLE_SHORT[role] for role in ROLES])
        axis.set_title(f"Tiling seed {tiling_seed}")
        axis.set_xlabel("matrix role")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("operator cosine (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Directional agreement by latent-code condition", y=0.98)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.85, bottom=0.19, wspace=0.11)
    path = output / "directional_operator_cosine.png"
    save_figure(fig, path, dpi)
    return path


def plot_gain_sensitivity(
    sensitivity: pd.DataFrame, seeds: list[int], output: Path, dpi: int
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 10.4))
    colors = {seeds[0]: "#2F6DAE", seeds[1]: "#D97721"}
    base = sensitivity[sensitivity["context"] == "cell/cell"]
    for axis, role in zip(axes.flat, ROLES):
        role_rows = base[base["role"] == role]
        for tiling_seed in seeds:
            rows = role_rows[role_rows["tiling_seed"] == tiling_seed].sort_values("gain")
            if rows.empty:
                raise RuntimeError(f"Missing gain-sensitivity rows for {role}, tiling {tiling_seed}")
            axis.plot(rows["gain"], rows["E_X"], color=colors[tiling_seed], linewidth=2.0)
            oracle = float(rows["heldout_oracle_gain"].iloc[0])
            axis.axvline(oracle, color=colors[tiling_seed], linestyle=":", linewidth=1.8, alpha=0.85)
        source_gain = float(role_rows["source_gain"].iloc[0])
        axis.axvline(source_gain, color="#222222", linestyle="--", linewidth=1.4)
        axis.axvline(1.0, color="#74849A", linestyle="-.", linewidth=1.2)
        axis.set_title(ROLE_SHORT[role])
        axis.set_xlabel("scalar output gain g")
        axis.set_ylabel(r"operator error $E_X$")
        axis.grid(alpha=0.20)

    legend_handles = [
        Line2D([0], [0], color=colors[seed], linewidth=2.2, label=f"error curve: tiling {seed}")
        for seed in seeds
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.5, label="source-frozen role gain"),
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=":",
                linewidth=1.8,
                label="heldout-fitted per-role oracle (descriptive only)",
            ),
            Line2D([0], [0], color="#74849A", linestyle="-.", linewidth=1.3, label="raw decoder output (g=1)"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Gain sensitivity under fixed context", y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.14, hspace=0.35, wspace=0.25)
    path = output / "gain_sensitivity_curves.png"
    save_figure(fig, path, dpi)
    return path


def plot_role_gain_permutation_control(
    permutations: pd.DataFrame, seeds: list[int], output: Path, dpi: int
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))
    for row_index, tiling_seed in enumerate(seeds):
        for column_index, aggregation in enumerate(("macro", "micro")):
            axis = axes[row_index, column_index]
            selected = permutations[
                (permutations["tiling_seed"] == tiling_seed)
                & (permutations["context"] == "cell/cell")
                & (permutations["code_condition"] == "correct")
                & (permutations["aggregation"] == aggregation)
            ]
            if len(selected) != 720 or int(selected["is_identity"].sum()) != 1:
                raise RuntimeError(
                    f"Unexpected permutation panel for tiling={tiling_seed}, aggregation={aggregation}: "
                    f"rows={len(selected)}, identity={int(selected['is_identity'].sum())}"
                )
            identity = float(selected.loc[selected["is_identity"], "E_X"].iloc[0])
            axis.hist(selected["E_X"], bins=35, color="#9AAAC0", edgecolor="white")
            axis.axvline(
                identity,
                color="#C43C39",
                linewidth=2.2,
                label=rf"correct role mapping: $E_X={identity:.3f}$",
            )
            axis.set_title(f"Tiling seed {tiling_seed} · {aggregation}")
            axis.set_xlabel(r"$E_X$ across all 720 gain-to-role mappings")
            axis.set_ylabel("number of mappings")
            axis.legend(frameon=False, loc="upper right")
            axis.grid(axis="y", alpha=0.20)
    fig.suptitle("Role-gain assignment permutation control", y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.08, hspace=0.34, wspace=0.20)
    path = output / "role_gain_permutation_control.png"
    save_figure(fig, path, dpi)
    return path


def write_supplement_notes(
    output: Path,
    source: Path,
    max_sensitivity_identity_error: float,
) -> tuple[Path, Path]:
    readme = output / "README.md"
    errata = output / "ERRATA.md"
    readme.write_text(
        "\n".join(
            [
                "# Source role-gain publication plot supplement",
                "",
                "Status: **plot-only rerender from sealed formal CSVs**.",
                "",
                f"Formal source: `{source}`.",
                "The renderer validated every entry in the formal `artifact_manifest.json` before and after rendering.",
                "No formal metrics, decisions, analyzer code, or target-domain assets were changed or accessed.",
                "",
                "## Semantic controls",
                "",
                "- `zero_code` means `decode(z=0, C)` and is a nonzero, context-conditioned decoder prior.",
                "- literal-zero output means `Y_hat=0`, for which `E_X=1` by definition.",
                "- heldout-fitted per-role oracle is descriptive only; it is not a source-frozen control.",
                "",
                "## Files",
                "",
                "- Six publication-readable PNGs with non-overlapping titles and legends.",
                "- `aggregate_metrics_presentation.csv`: the sealed aggregate values with pooled energy sums explicitly labeled.",
                "- `gain_sensitivity_presentation.csv`: exact quadratic curves extended to include `g=1` plus margin.",
                "- `ERRATA.md`: precise presentation issues addressed by this supplement.",
                "- `output_manifest.json`: source hashes, renderer hash, and output hashes.",
                "",
                "This supplement does not alter the formal scientific verdict.",
                "",
            ]
        )
    )
    errata.write_text(
        "\n".join(
            [
                "# Presentation errata for the sealed source role-gain analysis",
                "",
                "The sealed analysis remains authoritative for metrics and decisions. These are presentation/schema clarifications only.",
                "",
                "1. Several original PNGs placed a figure-level legend and title in the same upper band, causing overlap. The rerender separates them.",
                "2. The original gain-sensitivity grid spans `0 <= g <= 2 * source_gain`. For V, O, and FFN-down this excludes the raw `g=1` reference from the curve support. The supplemental curves use the exact identity `E_X(g) = (1-c^2) + (g*r-c)^2` from sealed raw cosine `c` and norm ratio `r`, and extend every role to at least `g=1.10`. This is an exact sufficient-statistic evaluation, not a fitted extrapolation.",
                f"   The maximum discrepancy between that identity and all overlapping sealed curve points was `{max_sensitivity_identity_error:.3e}`.",
                "3. In the formal `aggregate_metrics.csv`, macro `E_X` is the unweighted mean of six role-level errors, while the adjacent `x_error` and `x_target` fields are pooled sums whose ratio equals micro `E_X`, not macro `E_X`. The same distinction holds for `E_W`. `aggregate_metrics_presentation.csv` renames these fields to `sum_*_energy`, adds their explicit ratio, and states each row's aggregation definition. No numeric value was changed.",
                "4. Plot labels now distinguish `zero_code = decode(z=0,C)` (a nonzero decoder prior) from literal-zero output `Y_hat=0`, and label the heldout-fitted oracle as descriptive only.",
                "",
                "No target assets were accessed.",
                "",
            ]
        )
    )
    return readme, errata


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents:
        raise RuntimeError("Output must be a separate sibling, not the formal artifact directory or its child")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    events: list[str] = []

    def log(stage: str, message: str) -> None:
        line = f"[{utc_now()}] [{stage}] {message}"
        events.append(line)
        print(line, flush=True)

    log("config", f"source={source}")
    log("config", f"output={output}")
    log("config", f"device=cpu dtype=float64-for-CSV dpi={args.dpi} cache_mode=sealed-read-only")
    log("validate", "checking every formal artifact against artifact_manifest.json")
    manifest_path, manifest_sha, validated_before = validate_manifest(source)
    log("validate", f"PASS: {len(validated_before)} formal artifacts match hash and size")

    required_inputs = (
        "aggregate_metrics.csv",
        "gain_sensitivity.csv",
        "role_gain_permutations.csv",
    )
    log("load", f"loading sealed CSVs: {', '.join(required_inputs)}")
    aggregates = load_csv(source, required_inputs[0], validated_before)
    sensitivity = load_csv(source, required_inputs[1], validated_before)
    permutations = load_csv(source, required_inputs[2], validated_before)
    seeds = sorted(int(seed) for seed in aggregates["tiling_seed"].unique())
    if len(seeds) != 2:
        raise RuntimeError(f"Expected two tiling seeds; found {seeds}")
    missing_roles = set(ROLES) - set(aggregates["aggregation"].unique())
    if missing_roles:
        raise RuntimeError(f"aggregate_metrics.csv is missing roles: {sorted(missing_roles)}")
    log(
        "load",
        f"rows: aggregate={len(aggregates)}, sensitivity={len(sensitivity)}, "
        f"permutations={len(permutations)}; tiling_seeds={seeds}",
    )

    log("presentation", "building schema-disambiguated aggregate table")
    presentation_aggregates = build_presentation_aggregates(aggregates)
    aggregate_presentation_path = output / "aggregate_metrics_presentation.csv"
    presentation_aggregates.to_csv(aggregate_presentation_path, index=False)
    log(
        "presentation",
        f"wrote {aggregate_presentation_path} ({len(presentation_aggregates)} rows)",
    )
    log("presentation", "extending exact gain curves to include g=1 plus margin")
    extended_sensitivity, max_identity_error = build_extended_gain_sensitivity(
        aggregates, sensitivity, seeds
    )
    sensitivity_presentation_path = output / "gain_sensitivity_presentation.csv"
    extended_sensitivity.to_csv(sensitivity_presentation_path, index=False)
    log(
        "presentation",
        f"wrote {sensitivity_presentation_path} ({len(extended_sensitivity)} rows); "
        f"formal-overlap max_abs_error={max_identity_error:.3e}",
    )

    set_plot_style()
    plotters = (
        ("fixed-context error rescue", plot_fixed_context_error_rescue, aggregates),
        ("fixed-context role rescue", plot_fixed_context_role_rescue, aggregates),
        ("radial/angular decomposition", plot_radial_angular_decomposition, aggregates),
        ("directional operator cosine", plot_directional_operator_cosine, aggregates),
        ("gain sensitivity", plot_gain_sensitivity, extended_sensitivity),
        ("role-gain permutation control", plot_role_gain_permutation_control, permutations),
    )
    written: list[Path] = []
    for index, (label, plotter, frame) in enumerate(plotters, start=1):
        log("render", f"[{index}/{len(plotters)}] {label}")
        written.append(plotter(frame, seeds, output, args.dpi))

    readme_path, errata_path = write_supplement_notes(
        output, source, max_identity_error
    )
    written.extend(
        [
            aggregate_presentation_path,
            sensitivity_presentation_path,
            readme_path,
            errata_path,
        ]
    )
    log("documentation", f"wrote {readme_path} and {errata_path}")

    log("validate", "rechecking formal source artifacts after rendering")
    _, manifest_sha_after, validated_after = validate_manifest(source)
    if manifest_sha_after != manifest_sha or validated_after != validated_before:
        raise RuntimeError("Formal source artifacts changed during plot-only rendering")
    log("validate", "PASS: formal source artifacts unchanged")
    for path in written:
        log("output", f"wrote {path} ({path.stat().st_size} bytes)")

    png_count = sum(path.suffix.lower() == ".png" for path in written)
    log("done", f"rendered {png_count} PNGs; visual inspection is required separately")
    run_log = output / "run.log"
    run_log.write_text("\n".join(events) + "\n")

    validated_by_name = {row["name"]: row for row in validated_before}
    output_rows = [
        {"name": path.name, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in written + [run_log]
    ]
    renderer_path = Path(__file__).resolve()
    output_manifest = {
        "created_at_utc": utc_now(),
        "analysis_class": "PLOT_ONLY_RERENDER_FROM_SEALED_FORMAL_CSVS",
        "formal_artifact_directory": str(source),
        "formal_artifact_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "formal_artifacts_validated": len(validated_before),
            "unchanged_after_render": True,
        },
        "renderer": {"path": str(renderer_path), "sha256": sha256_file(renderer_path)},
        "render_config": {"device": "cpu", "dpi": args.dpi, "tiling_seeds": seeds},
        "sealed_csv_inputs": [validated_by_name[name] for name in required_inputs],
        "semantic_labels": {
            "zero_code": "decode(z=0, C), a nonzero C-conditioned decoder prior",
            "literal_zero_output": "Y_hat=0, giving E_X=1",
            "heldout_oracle": "heldout-fitted per-role oracle (descriptive only)",
        },
        "outputs": output_rows,
        "count": len(output_rows),
        "target_assets_accessed": False,
    }
    manifest_out_path = output / "output_manifest.json"
    manifest_out_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(f"[{utc_now()}] [manifest] wrote {manifest_out_path}", flush=True)


if __name__ == "__main__":
    main()
