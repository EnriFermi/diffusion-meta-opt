#!/usr/bin/env python3
"""Frozen post-run review for the source-only latent-code factorial.

This program is intentionally standalone and has no model, held-out dataset,
or target-domain imports.  It validates the sufficient-statistic artifacts,
recomputes the preregistered success screen, and writes the fixed diagnostic
tables and plots required before the factorial can be interpreted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
ENC_CONDITIONS = ("cell", "native")
CODE_CONDITIONS = ("correct", "permuted_within_row", "deranged_block", "zero")
DEC_CONDITIONS = ("cell", "native", "wrong_role_same_depth", "same_role_wrong_depth")
TILING_SEEDS = (26_081_601, 26_081_602)
AGGREGATIONS = ("macro", "micro", *ROLES)
NUM_MATRICES = 72
EXPECTED_MATRIX_ROWS = len(TILING_SEEDS) * NUM_MATRICES * len(ENC_CONDITIONS) * len(
    CODE_CONDITIONS
) * len(DEC_CONDITIONS)
EXPECTED_EFFECT_ROWS = (
    len(TILING_SEEDS)
    * len(ENC_CONDITIONS)
    * len(DEC_CONDITIONS)
    * 3
    * 2
    * len(AGGREGATIONS)
)
EXPECTED_LATENT_TILE_ROWS = len(TILING_SEEDS) * 20_736 * len(ENC_CONDITIONS) * 2
EXPECTED_LATENT_SUMMARY_ROWS = len(TILING_SEEDS) * NUM_MATRICES * len(ENC_CONDITIONS) * 2
EXPECTED_COSINE_EFFECT_ROWS = EXPECTED_EFFECT_ROWS
EXPECTED_C_EFFECT_ROWS = len(TILING_SEEDS) * 4 * 5 * len(AGGREGATIONS)

MATRIX_FILE = "factorial_matrix_sufficient_stats.csv"
AGGREGATE_FILE = "factorial_aggregate_metrics.csv"
EFFECT_FILE = "paired_code_effects.csv"
COSINE_EFFECT_FILE = "paired_code_cosine_effects.csv"
C_EFFECT_FILE = "paired_c_effects.csv"
LATENT_TILE_FILE = "latent_correct_donor_tiles.csv"
LATENT_SUMMARY_FILE = "latent_correct_donor_summary.csv"
NATIVE_ROW_FILE = "native_row_condition_invariance.csv"
TEMPLATE_DISTANCE_FILE = "decoder_template_distance_diagnostics.csv"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/postrun_analysis; must remain inside RUN_DIR.",
    )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def finite_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[:20].tolist()
        raise RuntimeError(f"{label} contains non-finite values at {bad}")


def validate_runner_artifact_hashes(run_dir: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    required = {
        MATRIX_FILE,
        AGGREGATE_FILE,
        EFFECT_FILE,
        COSINE_EFFECT_FILE,
        C_EFFECT_FILE,
        LATENT_TILE_FILE,
        LATENT_SUMMARY_FILE,
        NATIVE_ROW_FILE,
        TEMPLATE_DISTANCE_FILE,
        "decoder_path_preflight.json",
        "parent_baseline_parity.json",
        "zero_code_reuse_invariance.json",
        "factorial_grid_validity.json",
        "mechanism_screen.json",
        "resolved_config.json",
        "preexecution_contract.json",
        "target_access_seal.json",
    }
    records = run_manifest.get("artifacts")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise RuntimeError("run manifest does not contain the sealed artifact path/SHA/size records")
    by_name: dict[str, dict[str, Any]] = {}
    for row in records:
        if set(row) != {"path", "sha256", "bytes"}:
            raise RuntimeError(f"malformed artifact seal row: {row}")
        path = Path(str(row["path"]))
        if path.name in by_name:
            raise RuntimeError(f"duplicate artifact seal basename: {path.name}")
        if path.resolve(strict=True).parent != run_dir:
            raise RuntimeError(f"artifact seal escapes source run directory: {path}")
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != int(row["bytes"]) or actual_sha != str(row["sha256"]):
            raise RuntimeError(
                f"sealed runner artifact changed: {path.name} "
                f"size={actual_size}/{row['bytes']} sha={actual_sha}/{row['sha256']}"
            )
        by_name[path.name] = row
    missing = sorted(required - set(by_name))
    if missing:
        raise RuntimeError(f"run manifest does not seal required analyzer inputs: {missing}")
    return {
        "pass": True,
        "sealed_artifact_count": len(by_name),
        "required_input_count": len(required),
        "required_inputs": sorted(required),
    }


def validate_frozen_identity(run_dir: Path) -> dict[str, Any]:
    run_manifest = read_json(run_dir / "run_manifest.json")
    artifact_seal = validate_runner_artifact_hashes(run_dir, run_manifest)
    resolved = read_json(run_dir / "resolved_config.json")
    preexecution = read_json(run_dir / "preexecution_contract.json")
    current_sha = sha256_file(Path(__file__).resolve())
    expected_shas = {
        str(run_manifest["postrun_analyzer"]["sha256"]),
        str(resolved["postrun_analyzer_sha256"]),
        str(preexecution["postrun_analyzer_sha256"]),
    }
    if expected_shas != {current_sha}:
        raise RuntimeError(
            f"post-run analyzer is not the frozen preexecution version: current={current_sha} "
            f"expected={sorted(expected_shas)}"
        )
    if run_manifest.get("status") != "COMPLETE_SOURCE_ONLY_FACTORIAL_AWAITING_FROZEN_POSTRUN_ANALYZER_AND_PLOT_REVIEW":
        raise RuntimeError(f"factorial run is not awaiting this analyzer: {run_manifest.get('status')}")
    if bool(run_manifest.get("target_access")) or bool(run_manifest.get("target_data2vec_access")):
        raise RuntimeError("source-only run manifest reports target access")
    if resolved.get("target_access") is not False or preexecution.get("target_data2vec_access") is not False:
        raise RuntimeError("source-only preexecution target seal is absent")
    counts = run_manifest["counts"]
    expected_counts = {
        "heldout_matrices": NUM_MATRICES,
        "tiling_seeds": len(TILING_SEEDS),
        "formal_arms": len(ENC_CONDITIONS) * len(CODE_CONDITIONS) * len(DEC_CONDITIONS),
        "unique_decoder_arms": 28,
        "matrix_rows": EXPECTED_MATRIX_ROWS,
        "latent_pair_summary_rows": EXPECTED_LATENT_SUMMARY_ROWS,
        "latent_pair_tile_rows": EXPECTED_LATENT_TILE_ROWS,
        "native_row_condition_rows": len(TILING_SEEDS) * NUM_MATRICES,
        "paired_code_effect_rows": EXPECTED_EFFECT_ROWS,
        "paired_code_cosine_effect_rows": EXPECTED_COSINE_EFFECT_ROWS,
        "paired_c_effect_rows": EXPECTED_C_EFFECT_ROWS,
    }
    mismatches = {
        key: {"actual": counts.get(key), "expected": expected}
        for key, expected in expected_counts.items()
        if int(counts.get(key, -1)) != expected
    }
    if mismatches:
        raise RuntimeError(f"run-manifest count mismatch: {mismatches}")
    return {
        "run_manifest": run_manifest,
        "resolved_config": resolved,
        "preexecution": preexecution,
        "analyzer_sha256": current_sha,
        "artifact_seal": artifact_seal,
    }


def validate_matrix_grid(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "tiling_seed",
        "depth",
        "role",
        "encoder_condition",
        "code_condition",
        "decoder_condition",
        "computed_arm",
        "prediction_sha256",
        "w_target",
        "w_pred",
        "w_dot",
        "w_error",
        "x_target",
        "x_pred",
        "x_dot",
        "x_error",
        "raw_E_W",
        "raw_E_X",
        "raw_weight_cosine",
        "raw_operator_cosine",
    }
    require_columns(frame, required, label=MATRIX_FILE)
    if len(frame) != EXPECTED_MATRIX_ROWS:
        raise RuntimeError(f"matrix row count mismatch: actual={len(frame)} expected={EXPECTED_MATRIX_ROWS}")
    key_columns = [
        "tiling_seed",
        "depth",
        "role",
        "encoder_condition",
        "code_condition",
        "decoder_condition",
    ]
    if frame.duplicated(key_columns).any():
        raise RuntimeError("factorial matrix grid contains duplicate formal cells")
    observed = {
        tuple(row)
        for row in frame.loc[:, key_columns].itertuples(index=False, name=None)
    }
    expected = {
        (seed, depth, role, enc, code, dec)
        for seed in TILING_SEEDS
        for depth in range(12)
        for role in ROLES
        for enc in ENC_CONDITIONS
        for code in CODE_CONDITIONS
        for dec in DEC_CONDITIONS
    }
    if observed != expected:
        raise RuntimeError(
            f"factorial matrix grid differs from preregistration: missing={len(expected - observed)} "
            f"extra={len(observed - expected)}"
        )
    finite_columns(
        frame,
        (
            "w_target",
            "w_pred",
            "w_dot",
            "w_error",
            "x_target",
            "x_pred",
            "x_dot",
            "x_error",
            "raw_E_W",
            "raw_E_X",
            "raw_weight_cosine",
            "raw_operator_cosine",
        ),
        label=MATRIX_FILE,
    )
    if (frame[["w_target", "x_target"]].astype(float) <= 0).any().any():
        raise RuntimeError("factorial matrix grid contains a nonpositive target norm")
    if set(frame["role"]) != set(ROLES):
        raise RuntimeError(f"role grid mismatch: {sorted(set(frame['role']))}")

    computed_counts = frame.groupby(["tiling_seed", "depth", "role"])["computed_arm"].nunique()
    if set(computed_counts.astype(int)) != {28}:
        raise RuntimeError(f"unique computed-arm count mismatch: {computed_counts.value_counts().to_dict()}")
    zero = frame.loc[frame["code_condition"] == "zero"]
    zero_hash_counts = zero.groupby(["tiling_seed", "depth", "role", "decoder_condition"])[
        "prediction_sha256"
    ].nunique()
    if set(zero_hash_counts.astype(int)) != {1}:
        raise RuntimeError("zero-code predictions are not invariant to the formal encoder label")
    return {
        "rows": len(frame),
        "unique_formal_cells": len(observed),
        "unique_computed_arms_per_matrix": 28,
        "zero_reuse_pairs": len(zero_hash_counts),
    }


def compare_numeric_tables(
    *,
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    keys: list[str],
    numeric: list[str],
    exact: list[str],
    label: str,
    atol: float = 1e-12,
) -> dict[str, Any]:
    if stored.duplicated(keys).any() or recomputed.duplicated(keys).any():
        raise RuntimeError(f"{label} contains duplicate comparison keys")
    left_keys = {tuple(row) for row in stored[keys].itertuples(index=False, name=None)}
    right_keys = {tuple(row) for row in recomputed[keys].itertuples(index=False, name=None)}
    if left_keys != right_keys:
        raise RuntimeError(
            f"{label} key mismatch: stored_only={len(left_keys - right_keys)} "
            f"recomputed_only={len(right_keys - left_keys)}"
        )
    merged = stored.merge(recomputed, on=keys, suffixes=("__stored", "__recomputed"), validate="one_to_one")
    maximum = 0.0
    worst: dict[str, Any] | None = None
    for column in numeric:
        left = pd.to_numeric(merged[f"{column}__stored"], errors="raise").to_numpy(dtype=np.float64)
        right = pd.to_numeric(merged[f"{column}__recomputed"], errors="raise").to_numpy(dtype=np.float64)
        difference = np.abs(left - right)
        if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.isfinite(difference).all():
            raise RuntimeError(f"{label} has non-finite values in {column}")
        index = int(np.argmax(difference))
        value = float(difference[index])
        if value > maximum:
            maximum = value
            worst = {
                "column": column,
                "difference": value,
                "stored": float(left[index]),
                "recomputed": float(right[index]),
                "key": {key: merged.iloc[index][key] for key in keys},
            }
    if maximum > atol:
        raise RuntimeError(f"{label} independent recomputation mismatch: max={maximum} atol={atol} worst={worst}")
    exact_mismatches: dict[str, int] = {}
    for column in exact:
        left = merged[f"{column}__stored"].astype(str).to_numpy()
        right = merged[f"{column}__recomputed"].astype(str).to_numpy()
        count = int(np.count_nonzero(left != right))
        if count:
            exact_mismatches[column] = count
    if exact_mismatches:
        raise RuntimeError(f"{label} categorical recomputation mismatch: {exact_mismatches}")
    return {"pass": True, "rows": len(merged), "max_abs_difference": maximum, "atol": atol}


def recompute_aggregate_metrics(matrix: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    group_columns = ["tiling_seed", "encoder_condition", "code_condition", "decoder_condition"]
    grouped = matrix.groupby(group_columns, sort=True, dropna=False)
    for (seed, enc, code, dec), values in grouped:
        records = values.to_dict("records")
        scopes: list[tuple[str, list[dict[str, Any]]]] = [("micro", records)]
        scopes.extend((role, [row for row in records if row["role"] == role]) for role in ROLES)
        scope_rows: dict[str, dict[str, Any]] = {}
        for scope, selected in scopes:
            w_target = sum(float(row["w_target"]) for row in selected)
            w_pred = sum(float(row["w_pred"]) for row in selected)
            w_dot = sum(float(row["w_dot"]) for row in selected)
            x_target = sum(float(row["x_target"]) for row in selected)
            x_pred = sum(float(row["x_pred"]) for row in selected)
            x_dot = sum(float(row["x_dot"]) for row in selected)
            w_error = sum(float(row["w_error"]) for row in selected)
            x_error = sum(float(row["x_error"]) for row in selected)
            row = {
                "tiling_seed": int(seed),
                "encoder_condition": str(enc),
                "code_condition": str(code),
                "decoder_condition": str(dec),
                "aggregation": scope,
                "matrices": len(selected),
                "raw_E_W": w_error / w_target,
                "raw_E_X": x_error / x_target,
                "raw_weight_cosine": w_dot / math.sqrt(max(w_target * w_pred, 1e-300)),
                "raw_operator_cosine": x_dot / math.sqrt(max(x_target * x_pred, 1e-300)),
            }
            scope_rows[scope] = row
            output.append(row)
        output.append(
            {
                "tiling_seed": int(seed),
                "encoder_condition": str(enc),
                "code_condition": str(code),
                "decoder_condition": str(dec),
                "aggregation": "macro",
                "matrices": len(records),
                "raw_E_W": float(np.mean([scope_rows[role]["raw_E_W"] for role in ROLES])),
                "raw_E_X": float(np.mean([scope_rows[role]["raw_E_X"] for role in ROLES])),
                "raw_weight_cosine": float(np.mean([scope_rows[role]["raw_weight_cosine"] for role in ROLES])),
                "raw_operator_cosine": float(
                    np.mean([scope_rows[role]["raw_operator_cosine"] for role in ROLES])
                ),
            }
        )
    return pd.DataFrame(output)


def validate_aggregate_metrics(stored: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, Any]:
    expected_rows = len(TILING_SEEDS) * len(ENC_CONDITIONS) * len(CODE_CONDITIONS) * len(
        DEC_CONDITIONS
    ) * len(AGGREGATIONS)
    if len(stored) != expected_rows:
        raise RuntimeError(f"aggregate row count mismatch: actual={len(stored)} expected={expected_rows}")
    recomputed = recompute_aggregate_metrics(matrix)
    return compare_numeric_tables(
        stored=stored,
        recomputed=recomputed,
        keys=["tiling_seed", "encoder_condition", "code_condition", "decoder_condition", "aggregation"],
        numeric=["raw_E_W", "raw_E_X", "raw_weight_cosine", "raw_operator_cosine"],
        exact=["matrices"],
        label="factorial aggregate metrics",
    )


def bootstrap_error_scopes(
    indexed: dict[tuple[int, str], dict[str, Any]],
    sampled_depths: np.ndarray,
    *,
    metric: str,
) -> dict[str, np.ndarray]:
    prefix = "x" if metric == "E_X" else "w"
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    for role in ROLES:
        role_numerator = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_error"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        role_denominator = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_target"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        numerators.append(role_numerator)
        denominators.append(role_denominator)
    numerator = np.stack(numerators, axis=1)
    denominator = np.stack(denominators, axis=1)
    role_values = numerator / denominator
    output = {role: role_values[:, index] for index, role in enumerate(ROLES)}
    output["macro"] = role_values.mean(axis=1)
    output["micro"] = numerator.sum(axis=1) / denominator.sum(axis=1)
    return output


def bootstrap_cosine_scopes(
    indexed: dict[tuple[int, str], dict[str, Any]],
    sampled_depths: np.ndarray,
    *,
    metric: str,
) -> dict[str, np.ndarray]:
    prefix = "x" if metric == "operator_cosine" else "w"
    role_values: list[np.ndarray] = []
    role_dots: list[np.ndarray] = []
    role_targets: list[np.ndarray] = []
    role_predictions: list[np.ndarray] = []
    for role in ROLES:
        dots = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_dot"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        targets = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_target"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        predictions = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_pred"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        role_values.append(dots / np.sqrt(np.maximum(targets * predictions, 1e-300)))
        role_dots.append(dots)
        role_targets.append(targets)
        role_predictions.append(predictions)
    values = np.stack(role_values, axis=1)
    dots = np.stack(role_dots, axis=1)
    targets = np.stack(role_targets, axis=1)
    predictions = np.stack(role_predictions, axis=1)
    output = {role: values[:, index] for index, role in enumerate(ROLES)}
    output["macro"] = values.mean(axis=1)
    output["micro"] = dots.sum(axis=1) / np.sqrt(
        np.maximum(targets.sum(axis=1) * predictions.sum(axis=1), 1e-300)
    )
    return output


def recompute_paired_code_effects(matrix: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_depths = rng.integers(0, 12, size=(draws, 12))
    sampled_orbits = rng.integers(0, 6, size=(draws, 6))
    sampled_depth_orbits = np.concatenate([sampled_orbits, sampled_orbits + 6], axis=1)
    exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
    output: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        for enc in ENC_CONDITIONS:
            for dec in DEC_CONDITIONS:
                indexed: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
                for code in CODE_CONDITIONS:
                    selected = matrix[
                        (matrix["tiling_seed"] == tiling_seed)
                        & (matrix["encoder_condition"] == enc)
                        & (matrix["code_condition"] == code)
                        & (matrix["decoder_condition"] == dec)
                    ]
                    indexed[code] = {
                        (int(row["depth"]), str(row["role"])): row
                        for row in selected.to_dict("records")
                    }
                    if len(indexed[code]) != NUM_MATRICES:
                        raise RuntimeError(f"independent effect cell incomplete: {tiling_seed}/{enc}/{code}/{dec}")
                for comparator in ("permuted_within_row", "deranged_block", "zero"):
                    depth_draws = sampled_depth_orbits if comparator == "deranged_block" else sampled_depths
                    bootstrap_unit = (
                        "six_depth_plus_6_orbits" if comparator == "deranged_block" else "12_transformer_blocks"
                    )
                    clusters = 6 if comparator == "deranged_block" else 12
                    for metric in ("E_X", "E_W"):
                        correct = bootstrap_error_scopes(indexed["correct"], depth_draws, metric=metric)
                        control = bootstrap_error_scopes(indexed[comparator], depth_draws, metric=metric)
                        correct_point = bootstrap_error_scopes(indexed["correct"], exact_depths, metric=metric)
                        control_point = bootstrap_error_scopes(indexed[comparator], exact_depths, metric=metric)
                        for aggregation in AGGREGATIONS:
                            delta = correct[aggregation] - control[aggregation]
                            ratio = control[aggregation] / correct[aggregation]
                            point_correct = float(correct_point[aggregation][0])
                            point_control = float(control_point[aggregation][0])
                            point_ratio = point_control / point_correct
                            ratio_l95 = float(np.quantile(ratio, 0.05))
                            output.append(
                                {
                                    "tiling_seed": tiling_seed,
                                    "encoder_condition": enc,
                                    "decoder_condition": dec,
                                    "metric": metric,
                                    "aggregation": aggregation,
                                    "comparator": comparator,
                                    "point_correct": point_correct,
                                    "point_comparator": point_control,
                                    "point_delta_correct_minus_comparator": point_correct - point_control,
                                    "delta_l95": float(np.quantile(delta, 0.05)),
                                    "delta_u95": float(np.quantile(delta, 0.95)),
                                    "point_ratio_comparator_over_correct": point_ratio,
                                    "ratio_l95": ratio_l95,
                                    "ratio_u95": float(np.quantile(ratio, 0.95)),
                                    "passes_point_ratio_1p05_and_ratio_l95_gt_1": bool(
                                        point_ratio >= 1.05 and ratio_l95 > 1.0
                                    ),
                                    "draws": draws,
                                    "bootstrap_unit": bootstrap_unit,
                                    "effective_clusters": clusters,
                                }
                            )
    return pd.DataFrame(output)


def factorial_cell_index(
    matrix: pd.DataFrame,
    *,
    tiling_seed: int,
    encoder_condition: str,
    code_condition: str,
    decoder_condition: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    selected = matrix[
        (matrix["tiling_seed"] == tiling_seed)
        & (matrix["encoder_condition"] == encoder_condition)
        & (matrix["code_condition"] == code_condition)
        & (matrix["decoder_condition"] == decoder_condition)
    ]
    indexed = {
        (int(row["depth"]), str(row["role"])): row
        for row in selected.to_dict("records")
    }
    if len(indexed) != NUM_MATRICES:
        raise RuntimeError(
            "independent factorial cell incomplete: "
            f"{tiling_seed}/{encoder_condition}/{code_condition}/{decoder_condition}"
        )
    return indexed


def recompute_paired_cosine_effects(matrix: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_depths = rng.integers(0, 12, size=(draws, 12))
    sampled_orbits = rng.integers(0, 6, size=(draws, 6))
    sampled_depth_orbits = np.concatenate([sampled_orbits, sampled_orbits + 6], axis=1)
    exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
    output: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        for enc in ENC_CONDITIONS:
            for dec in DEC_CONDITIONS:
                indexed = {
                    code: factorial_cell_index(
                        matrix,
                        tiling_seed=tiling_seed,
                        encoder_condition=enc,
                        code_condition=code,
                        decoder_condition=dec,
                    )
                    for code in CODE_CONDITIONS
                }
                for comparator in ("permuted_within_row", "deranged_block", "zero"):
                    depth_draws = sampled_depth_orbits if comparator == "deranged_block" else sampled_depths
                    bootstrap_unit = (
                        "six_depth_plus_6_orbits" if comparator == "deranged_block" else "12_transformer_blocks"
                    )
                    clusters = 6 if comparator == "deranged_block" else 12
                    for metric in ("operator_cosine", "weight_cosine"):
                        correct = bootstrap_cosine_scopes(indexed["correct"], depth_draws, metric=metric)
                        control = bootstrap_cosine_scopes(indexed[comparator], depth_draws, metric=metric)
                        correct_point = bootstrap_cosine_scopes(indexed["correct"], exact_depths, metric=metric)
                        control_point = bootstrap_cosine_scopes(indexed[comparator], exact_depths, metric=metric)
                        for aggregation in AGGREGATIONS:
                            delta = correct[aggregation] - control[aggregation]
                            output.append(
                                {
                                    "tiling_seed": tiling_seed,
                                    "encoder_condition": enc,
                                    "decoder_condition": dec,
                                    "metric": metric,
                                    "aggregation": aggregation,
                                    "comparator": comparator,
                                    "point_correct": float(correct_point[aggregation][0]),
                                    "point_comparator": float(control_point[aggregation][0]),
                                    "point_delta_correct_minus_comparator": float(
                                        correct_point[aggregation][0] - control_point[aggregation][0]
                                    ),
                                    "delta_l95": float(np.quantile(delta, 0.05)),
                                    "delta_u95": float(np.quantile(delta, 0.95)),
                                    "draws": draws,
                                    "bootstrap_unit": bootstrap_unit,
                                    "effective_clusters": clusters,
                                }
                            )
    return pd.DataFrame(output)


def recompute_paired_c_effects(matrix: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_depths = rng.integers(0, 12, size=(draws, 12))
    exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
    cells = {
        "CC": ("cell", "cell"),
        "NC": ("native", "cell"),
        "CN": ("cell", "native"),
        "NN": ("native", "native"),
    }
    contrasts = {
        "encoder_native_minus_cell_at_cell_decoder": {"NC": 1.0, "CC": -1.0},
        "encoder_native_minus_cell_at_native_decoder": {"NN": 1.0, "CN": -1.0},
        "decoder_native_minus_cell_at_cell_encoder": {"CN": 1.0, "CC": -1.0},
        "decoder_native_minus_cell_at_native_encoder": {"NN": 1.0, "NC": -1.0},
        "interaction_nn_minus_nc_minus_cn_plus_cc": {
            "NN": 1.0,
            "NC": -1.0,
            "CN": -1.0,
            "CC": 1.0,
        },
    }
    output: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        indexed = {
            cell: factorial_cell_index(
                matrix,
                tiling_seed=tiling_seed,
                encoder_condition=enc,
                code_condition="correct",
                decoder_condition=dec,
            )
            for cell, (enc, dec) in cells.items()
        }
        for metric in ("E_X", "E_W", "operator_cosine", "weight_cosine"):
            aggregator = bootstrap_error_scopes if metric in {"E_X", "E_W"} else bootstrap_cosine_scopes
            sampled = {
                cell: aggregator(value, sampled_depths, metric=metric)
                for cell, value in indexed.items()
            }
            points = {
                cell: aggregator(value, exact_depths, metric=metric)
                for cell, value in indexed.items()
            }
            for contrast, coefficients in contrasts.items():
                for aggregation in AGGREGATIONS:
                    delta = sum(
                        coefficient * sampled[cell][aggregation]
                        for cell, coefficient in coefficients.items()
                    )
                    point = sum(
                        coefficient * points[cell][aggregation][0]
                        for cell, coefficient in coefficients.items()
                    )
                    output.append(
                        {
                            "tiling_seed": tiling_seed,
                            "metric": metric,
                            "aggregation": aggregation,
                            "contrast": contrast,
                            "coefficients": json.dumps(coefficients, sort_keys=True),
                            "point_contrast": float(point),
                            "l95": float(np.quantile(delta, 0.05)),
                            "u95": float(np.quantile(delta, 0.95)),
                            "draws": draws,
                            "bootstrap_unit": "12_transformer_blocks",
                            "effective_clusters": 12,
                        }
                    )
    return pd.DataFrame(output)


def validate_validity_artifacts(run_dir: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in (
        "decoder_path_preflight.json",
        "parent_baseline_parity.json",
        "zero_code_reuse_invariance.json",
        "factorial_grid_validity.json",
    ):
        value = read_json(run_dir / name)
        if value.get("pass") is not True:
            raise RuntimeError(f"required validity artifact failed: {name}: {value}")
        checks[name] = value
    parity = checks["parent_baseline_parity.json"]
    if int(parity["prediction_sha256_equal"]) != 288 or int(parity["prediction_sha256_total"]) != 288:
        raise RuntimeError(f"parent parity did not cover exactly 288 prediction hashes: {parity}")
    if float(parity["max_sufficient_stat_abs_diff"]) > 1e-9:
        raise RuntimeError(f"parent sufficient-stat absolute parity failed: {parity}")
    if float(parity["max_sufficient_stat_rel_diff"]) > 1e-12:
        raise RuntimeError(f"parent sufficient-stat relative parity failed: {parity}")
    return {
        "decoder_path_preflight": True,
        "parent_baseline_parity": True,
        "parent_prediction_hashes_equal": 288,
        "zero_code_reuse": True,
        "factorial_grid": True,
    }


def validate_effects(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    required = {
        "tiling_seed",
        "encoder_condition",
        "decoder_condition",
        "metric",
        "aggregation",
        "comparator",
        "point_correct",
        "point_comparator",
        "point_delta_correct_minus_comparator",
        "delta_l95",
        "delta_u95",
        "point_ratio_comparator_over_correct",
        "ratio_l95",
        "ratio_u95",
        "passes_point_ratio_1p05_and_ratio_l95_gt_1",
        "bootstrap_unit",
        "effective_clusters",
    }
    require_columns(frame, required, label=EFFECT_FILE)
    if len(frame) != EXPECTED_EFFECT_ROWS:
        raise RuntimeError(f"effect row count mismatch: actual={len(frame)} expected={EXPECTED_EFFECT_ROWS}")
    key = [
        "tiling_seed",
        "encoder_condition",
        "decoder_condition",
        "metric",
        "aggregation",
        "comparator",
    ]
    if frame.duplicated(key).any():
        raise RuntimeError("paired effect table contains duplicate estimands")
    if not (
        frame["contrast"]
        == "correct_minus_" + frame["comparator"].astype(str)
    ).all():
        raise RuntimeError("paired effect contrast labels do not match their comparator")
    finite_columns(
        frame,
        (
            "point_correct",
            "point_comparator",
            "point_delta_correct_minus_comparator",
            "delta_l95",
            "delta_u95",
            "point_ratio_comparator_over_correct",
            "ratio_l95",
            "ratio_u95",
        ),
        label=EFFECT_FILE,
    )
    recomputed = (
        (frame["point_ratio_comparator_over_correct"].astype(float) >= 1.05)
        & (frame["ratio_l95"].astype(float) > 1.0)
    )
    stored = frame["passes_point_ratio_1p05_and_ratio_l95_gt_1"].astype(str).str.lower().eq("true")
    if not np.array_equal(recomputed.to_numpy(), stored.to_numpy()):
        raise RuntimeError("stored success flags do not match the exact preregistered ratio rule")
    block = frame[frame["comparator"] == "deranged_block"]
    ordinary = frame[frame["comparator"] != "deranged_block"]
    if set(block["bootstrap_unit"]) != {"six_depth_plus_6_orbits"} or set(
        block["effective_clusters"].astype(int)
    ) != {6}:
        raise RuntimeError("depth+6 comparator did not use the six-orbit clustered bootstrap")
    if set(ordinary["bootstrap_unit"]) != {"12_transformer_blocks"} or set(
        ordinary["effective_clusters"].astype(int)
    ) != {12}:
        raise RuntimeError("primary/zero comparator did not use the 12-block bootstrap")
    recomputed = recompute_paired_code_effects(matrix, draws=draws, seed=seed)
    independent = compare_numeric_tables(
        stored=frame,
        recomputed=recomputed,
        keys=key,
        numeric=[
            "point_correct",
            "point_comparator",
            "point_delta_correct_minus_comparator",
            "delta_l95",
            "delta_u95",
            "point_ratio_comparator_over_correct",
            "ratio_l95",
            "ratio_u95",
        ],
        exact=[
            "passes_point_ratio_1p05_and_ratio_l95_gt_1",
            "draws",
            "bootstrap_unit",
            "effective_clusters",
        ],
        label="paired code effects",
    )
    return {
        "rows": len(frame),
        "exact_success_flags_recomputed": True,
        "bootstrap_units_valid": True,
        "independent_from_matrix_recomputation": independent,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def validate_cosine_effects(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    required = {
        "tiling_seed",
        "encoder_condition",
        "decoder_condition",
        "metric",
        "aggregation",
        "comparator",
        "point_correct",
        "point_comparator",
        "point_delta_correct_minus_comparator",
        "delta_l95",
        "delta_u95",
        "draws",
        "bootstrap_unit",
        "effective_clusters",
    }
    require_columns(frame, required, label=COSINE_EFFECT_FILE)
    if len(frame) != EXPECTED_COSINE_EFFECT_ROWS:
        raise RuntimeError(
            f"cosine effect row count mismatch: actual={len(frame)} expected={EXPECTED_COSINE_EFFECT_ROWS}"
        )
    keys = [
        "tiling_seed",
        "encoder_condition",
        "decoder_condition",
        "metric",
        "aggregation",
        "comparator",
    ]
    if frame.duplicated(keys).any():
        raise RuntimeError("paired cosine effect table contains duplicate estimands")
    finite_columns(
        frame,
        (
            "point_correct",
            "point_comparator",
            "point_delta_correct_minus_comparator",
            "delta_l95",
            "delta_u95",
        ),
        label=COSINE_EFFECT_FILE,
    )
    if set(frame["metric"]) != {"operator_cosine", "weight_cosine"} or set(
        frame["aggregation"]
    ) != set(AGGREGATIONS):
        raise RuntimeError("paired cosine effect metric/aggregation grid mismatch")
    if set(frame["draws"].astype(int)) != {draws}:
        raise RuntimeError("paired cosine effects use the wrong bootstrap draw count")
    block = frame[frame["comparator"] == "deranged_block"]
    ordinary = frame[frame["comparator"] != "deranged_block"]
    if set(block["bootstrap_unit"]) != {"six_depth_plus_6_orbits"} or set(
        block["effective_clusters"].astype(int)
    ) != {6}:
        raise RuntimeError("paired cosine depth+6 effects did not use six-orbit clustering")
    if set(ordinary["bootstrap_unit"]) != {"12_transformer_blocks"} or set(
        ordinary["effective_clusters"].astype(int)
    ) != {12}:
        raise RuntimeError("paired cosine primary effects did not use 12-block clustering")
    recomputed = recompute_paired_cosine_effects(matrix, draws=draws, seed=seed)
    independent = compare_numeric_tables(
        stored=frame,
        recomputed=recomputed,
        keys=keys,
        numeric=[
            "point_correct",
            "point_comparator",
            "point_delta_correct_minus_comparator",
            "delta_l95",
            "delta_u95",
        ],
        exact=["draws", "bootstrap_unit", "effective_clusters"],
        label="paired code cosine effects",
    )
    return {
        "rows": len(frame),
        "bootstrap_units_valid": True,
        "independent_from_matrix_recomputation": independent,
        "bootstrap_seed": seed,
    }


def validate_c_effects(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    required = {
        "tiling_seed",
        "metric",
        "aggregation",
        "contrast",
        "coefficients",
        "point_contrast",
        "l95",
        "u95",
        "draws",
        "bootstrap_unit",
        "effective_clusters",
    }
    require_columns(frame, required, label=C_EFFECT_FILE)
    if len(frame) != EXPECTED_C_EFFECT_ROWS:
        raise RuntimeError(f"C-effect row count mismatch: actual={len(frame)} expected={EXPECTED_C_EFFECT_ROWS}")
    keys = ["tiling_seed", "metric", "aggregation", "contrast"]
    if frame.duplicated(keys).any():
        raise RuntimeError("paired C-effect table contains duplicate estimands")
    finite_columns(frame, ("point_contrast", "l95", "u95"), label=C_EFFECT_FILE)
    expected_contrasts = {
        "encoder_native_minus_cell_at_cell_decoder",
        "encoder_native_minus_cell_at_native_decoder",
        "decoder_native_minus_cell_at_cell_encoder",
        "decoder_native_minus_cell_at_native_encoder",
        "interaction_nn_minus_nc_minus_cn_plus_cc",
    }
    if (
        set(frame["metric"]) != {"E_X", "E_W", "operator_cosine", "weight_cosine"}
        or set(frame["aggregation"]) != set(AGGREGATIONS)
        or set(frame["contrast"]) != expected_contrasts
    ):
        raise RuntimeError("paired C-effect factorial grid mismatch")
    if (
        set(frame["draws"].astype(int)) != {draws}
        or set(frame["bootstrap_unit"]) != {"12_transformer_blocks"}
        or set(frame["effective_clusters"].astype(int)) != {12}
    ):
        raise RuntimeError("paired C effects use the wrong bootstrap contract")
    recomputed = recompute_paired_c_effects(matrix, draws=draws, seed=seed)
    independent = compare_numeric_tables(
        stored=frame,
        recomputed=recomputed,
        keys=keys,
        numeric=["point_contrast", "l95", "u95"],
        exact=["coefficients", "draws", "bootstrap_unit", "effective_clusters"],
        label="paired C effects",
    )
    return {
        "rows": len(frame),
        "contrasts": sorted(expected_contrasts),
        "bootstrap_units_valid": True,
        "independent_from_matrix_recomputation": independent,
        "bootstrap_seed": seed,
    }


def validate_latent_diagnostics(tiles: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    require_columns(
        tiles,
        {
            "tiling_seed",
            "target_depth",
            "target_role",
            "tile_ordinal",
            "encoder_condition",
            "comparator",
            "correct_norm",
            "donor_norm",
            "difference_norm",
            "relative_distance",
            "cosine",
        },
        label=LATENT_TILE_FILE,
    )
    require_columns(
        summary,
        {
            "tiling_seed",
            "target_depth",
            "target_role",
            "encoder_condition",
            "comparator",
            "correct_sha256",
            "donor_sha256",
            "hashes_unequal",
            "all_tile_differences_nonzero",
            "finite",
        },
        label=LATENT_SUMMARY_FILE,
    )
    if len(tiles) != EXPECTED_LATENT_TILE_ROWS or len(summary) != EXPECTED_LATENT_SUMMARY_ROWS:
        raise RuntimeError(
            f"latent diagnostic count mismatch: tiles={len(tiles)}/{EXPECTED_LATENT_TILE_ROWS} "
            f"summary={len(summary)}/{EXPECTED_LATENT_SUMMARY_ROWS}"
        )
    finite_columns(
        tiles,
        ("correct_norm", "donor_norm", "difference_norm", "relative_distance", "cosine"),
        label=LATENT_TILE_FILE,
    )
    if (tiles[["correct_norm", "donor_norm", "difference_norm", "relative_distance"]].astype(float) <= 0).any().any():
        raise RuntimeError("latent diagnostics contain a zero/nonpositive norm or intervention distance")
    bool_columns = ("hashes_unequal", "all_tile_differences_nonzero", "finite")
    if not all(summary[name].astype(str).str.lower().eq("true").all() for name in bool_columns):
        raise RuntimeError("latent pair summary contains a failed separation assertion")
    if (summary["correct_sha256"] == summary["donor_sha256"]).any():
        raise RuntimeError("latent pair summary contains equal correct/donor tensor hashes")
    keys = ["tiling_seed", "target_depth", "target_role", "encoder_condition", "comparator"]
    if summary.duplicated(keys).any():
        raise RuntimeError("latent pair summary contains duplicate matrix/condition rows")
    return {"tile_rows": len(tiles), "summary_rows": len(summary), "all_tile_differences_nonzero": True}


def validate_native_row_invariance(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "tiling_seed",
        "depth",
        "role",
        "source_key",
        "row_groups",
        "columns_per_row_group",
        "x_row_templates_sha256",
        "c_var_identical_within_row_group",
        "c_var_max_abs_within_row_group",
        "c_var_row_templates_sha256",
        "c_patch_identical_within_row_group",
        "c_patch_max_abs_within_row_group",
        "c_patch_row_templates_sha256",
        "c_pooled_identical_within_row_group",
        "c_pooled_max_abs_within_row_group",
        "c_pooled_row_templates_sha256",
        "native_context_basis",
    }
    require_columns(frame, required, label=NATIVE_ROW_FILE)
    expected_rows = len(TILING_SEEDS) * NUM_MATRICES
    if len(frame) != expected_rows or frame.duplicated(["tiling_seed", "depth", "role"]).any():
        raise RuntimeError(f"native row-invariance census mismatch: {len(frame)}/{expected_rows}")
    expected_grid = {
        (seed, depth, role)
        for seed in TILING_SEEDS
        for depth in range(12)
        for role in ROLES
    }
    observed_grid = {
        tuple(row)
        for row in frame[["tiling_seed", "depth", "role"]].itertuples(index=False, name=None)
    }
    if observed_grid != expected_grid:
        raise RuntimeError("native row-invariance table is not the exact two-tiling 12x6 grid")
    bool_columns = (
        "c_var_identical_within_row_group",
        "c_patch_identical_within_row_group",
        "c_pooled_identical_within_row_group",
    )
    if not all(frame[column].astype(str).str.lower().eq("true").all() for column in bool_columns):
        raise RuntimeError("native C changed across column tiles sharing the same activation row group")
    max_columns = (
        "c_var_max_abs_within_row_group",
        "c_patch_max_abs_within_row_group",
        "c_pooled_max_abs_within_row_group",
    )
    finite_columns(frame, max_columns, label=NATIVE_ROW_FILE)
    if not (frame[list(max_columns)].astype(float) == 0.0).all().all():
        raise RuntimeError("native row-invariance max-absolute diagnostics are not exactly zero")
    if (frame[["row_groups", "columns_per_row_group"]].astype(int) <= 0).any().any():
        raise RuntimeError("native row-invariance geometry is nonpositive")
    hash_columns = (
        "x_row_templates_sha256",
        "c_var_row_templates_sha256",
        "c_patch_row_templates_sha256",
        "c_pooled_row_templates_sha256",
    )
    if not all(frame[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all() for column in hash_columns):
        raise RuntimeError("native row-invariance table contains malformed tensor hashes")
    return {
        "rows": len(frame),
        "all_c_var_c_patch_c_pooled_bit_identical_within_row_group": True,
        "max_abs": 0.0,
    }


def validate_template_distances(frame: pd.DataFrame) -> dict[str, Any]:
    require_columns(
        frame,
        {
            "decoder_condition",
            "target_role",
            "target_depth",
            "template_role",
            "template_depth",
            "target_sha256",
            "wrong_sha256",
            "l2_distance",
            "relative_l2_distance",
            "cosine",
        },
        label=TEMPLATE_DISTANCE_FILE,
    )
    if len(frame) != 144:
        raise RuntimeError(f"decoder-template distance table must contain 144 rows, got {len(frame)}")
    if frame.duplicated(["decoder_condition", "target_role", "target_depth"]).any():
        raise RuntimeError("decoder-template distance table contains duplicate target cells")
    finite_columns(frame, ("l2_distance", "relative_l2_distance", "cosine"), label=TEMPLATE_DISTANCE_FILE)
    if (frame[["l2_distance", "relative_l2_distance"]].astype(float) <= 0).any().any():
        raise RuntimeError("wrong decoder-template intervention has zero distance")
    if (frame["target_sha256"] == frame["wrong_sha256"]).any():
        raise RuntimeError("wrong decoder-template intervention has an identical tensor hash")
    expected_conditions = {"wrong_role_same_depth", "same_role_wrong_depth"}
    if set(frame["decoder_condition"]) != expected_conditions:
        raise RuntimeError(f"wrong decoder condition set mismatch: {set(frame['decoder_condition'])}")
    return {"rows": len(frame), "conditions": sorted(expected_conditions), "all_distances_positive": True}


def correct_code_2x2(aggregate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        aggregate,
        {
            "tiling_seed",
            "encoder_condition",
            "code_condition",
            "decoder_condition",
            "aggregation",
            "raw_E_X",
            "raw_E_W",
            "raw_operator_cosine",
            "raw_weight_cosine",
        },
        label=AGGREGATE_FILE,
    )
    finite_columns(
        aggregate,
        ("raw_E_X", "raw_E_W", "raw_operator_cosine", "raw_weight_cosine"),
        label=AGGREGATE_FILE,
    )
    selected = aggregate[
        (aggregate["code_condition"] == "correct")
        & aggregate["encoder_condition"].isin(ENC_CONDITIONS)
        & aggregate["decoder_condition"].isin(("cell", "native"))
    ].copy()
    expected = len(TILING_SEEDS) * len(ENC_CONDITIONS) * 2 * len(AGGREGATIONS)
    if len(selected) != expected:
        raise RuntimeError(f"correct-code 2x2 C table incomplete: actual={len(selected)} expected={expected}")
    selected["cell_label"] = selected["encoder_condition"].str[0].str.upper() + selected[
        "decoder_condition"
    ].str[0].str.upper()
    rows: list[dict[str, Any]] = []
    metrics = ("raw_E_X", "raw_E_W", "raw_operator_cosine", "raw_weight_cosine")
    for (seed, aggregation), values in selected.groupby(["tiling_seed", "aggregation"], sort=True):
        indexed = values.set_index(["encoder_condition", "decoder_condition"])
        if len(indexed) != 4:
            raise RuntimeError(f"duplicate/incomplete 2x2 C cell: seed={seed} aggregation={aggregation}")
        for metric in metrics:
            cc = float(indexed.loc[("cell", "cell"), metric])
            nc = float(indexed.loc[("native", "cell"), metric])
            cn = float(indexed.loc[("cell", "native"), metric])
            nn = float(indexed.loc[("native", "native"), metric])
            rows.append(
                {
                    "tiling_seed": int(seed),
                    "aggregation": aggregation,
                    "metric": metric,
                    "cell_cell": cc,
                    "native_cell": nc,
                    "cell_native": cn,
                    "native_native": nn,
                    "encoder_native_minus_cell_at_cell_decoder": nc - cc,
                    "encoder_native_minus_cell_at_native_decoder": nn - cn,
                    "decoder_native_minus_cell_at_cell_encoder": cn - cc,
                    "decoder_native_minus_cell_at_native_encoder": nn - nc,
                    "difference_in_differences_nn_minus_nc_minus_cn_plus_cc": nn - nc - cn + cc,
                }
            )
    return selected, pd.DataFrame(rows)


def mechanism_criterion(
    effect: pd.DataFrame,
    cosine_effect: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tier_rows = effect[
        (effect["metric"] == "E_X")
        & (effect["aggregation"].isin(("macro", "micro")))
        & (effect["comparator"].isin(("permuted_within_row", "zero")))
        & (
            ((effect["encoder_condition"] == "cell") & (effect["decoder_condition"] == "cell"))
            | ((effect["encoder_condition"] == "native") & (effect["decoder_condition"] == "native"))
        )
    ].copy()
    if len(tier_rows) != 16:
        raise RuntimeError(f"fixed plus native criterion tiers must contain 16 rows, got {len(tier_rows)}")
    tier_rows["criterion_pass_recomputed"] = (
        (tier_rows["point_ratio_comparator_over_correct"].astype(float) >= 1.05)
        & (tier_rows["ratio_l95"].astype(float) > 1.0)
    )
    tier_rows["excludes_ratio_at_least_1p05"] = tier_rows["ratio_u95"].astype(float) < 1.05
    tier_rows["tier"] = np.where(
        tier_rows["encoder_condition"] == "cell",
        "PRIMARY_FIXED_C",
        "NATIVE_NATIVE_ROBUSTNESS",
    )
    fixed = tier_rows[
        (tier_rows["encoder_condition"] == "cell") & (tier_rows["decoder_condition"] == "cell")
    ]
    native = tier_rows[
        (tier_rows["encoder_condition"] == "native") & (tier_rows["decoder_condition"] == "native")
    ]
    if len(fixed) != 8 or len(native) != 8:
        raise RuntimeError(f"mechanism tier census mismatch: fixed={len(fixed)} native={len(native)}")
    directional = cosine_effect[
        (cosine_effect["metric"] == "operator_cosine")
        & (cosine_effect["aggregation"].isin(("macro", "micro")))
        & (cosine_effect["comparator"] == "permuted_within_row")
        & (cosine_effect["encoder_condition"] == "cell")
        & (cosine_effect["decoder_condition"] == "cell")
    ].copy()
    if len(directional) != 4:
        raise RuntimeError(f"directional operator tier must contain four rows, got {len(directional)}")
    directional["directional_pass_recomputed"] = directional["delta_l95"].astype(float) > 0.0
    fixed_pass = bool(fixed["criterion_pass_recomputed"].all())
    native_pass = bool(native["criterion_pass_recomputed"].all())
    directional_pass = bool(directional["directional_pass_recomputed"].all())
    any_exclusion = bool(fixed["excludes_ratio_at_least_1p05"].any())
    if fixed_pass:
        primary_status = "SUPPORTED_ON_FIXED_C_SOURCE_PANEL"
    elif any_exclusion:
        primary_status = "PREREGISTERED_UNIVERSAL_8_ROW_CLAIM_EXCLUDED_AT_ONE_OR_MORE_ENDPOINTS"
    else:
        primary_status = "INCONCLUSIVE_FOR_5_PERCENT_EFFECT"
    result = {
        "primary_fixed_c_criterion": (
            "raw E_X comparator/correct point ratio >= 1.05 AND paired one-sided ratio L95 > 1, "
            "macro AND micro, within-row permutation AND zero, both locked tilings (8 rows)"
        ),
        "native_native_robustness_criterion": "the analogous 8 rows under native/native context",
        "directional_operator_structure_criterion": (
            "cell/cell correct-minus-within-row operator cosine L95 > 0 for macro and micro, both tilings"
        ),
        "primary_fixed_c_load_bearing_pass": fixed_pass,
        "native_native_robustness_pass": native_pass,
        "strong_context_robust_all_16_pass": bool(fixed_pass and native_pass),
        "directional_operator_cosine_pass": directional_pass,
        "fixed_c_tile_specific_with_directional_structure_pass": bool(fixed_pass and directional_pass),
        "strong_weight_code_dependence_screen": fixed_pass,
        "primary_fixed_c_passed_rows": int(fixed["criterion_pass_recomputed"].sum()),
        "primary_fixed_c_required_rows": len(fixed),
        "native_robustness_passed_rows": int(native["criterion_pass_recomputed"].sum()),
        "native_robustness_required_rows": len(native),
        "directional_passed_rows": int(directional["directional_pass_recomputed"].sum()),
        "directional_required_rows": len(directional),
        "primary_fixed_c_status": primary_status,
        "fixed_c_universal_claim_excluded_by_any_endpoint": any_exclusion,
        "interpretation_if_not_pass": (
            "a ratio U95 below 1.05 excludes the >=5% effect at that endpoint; otherwise non-pass is "
            "inconclusive. Smaller/local/tiling-specific effects remain exploratory"
        ),
        "scope_limit": (
            "pass establishes causal W/tile-specific z contribution on this source panel, not absolute "
            "reconstruction quality, transfer, or a global weight manifold"
        ),
    }
    return tier_rows, directional, result


def validate_stored_mechanism_screen(run_dir: Path, recomputed: dict[str, Any]) -> dict[str, Any]:
    stored = read_json(run_dir / "mechanism_screen.json")
    keys = (
        "strong_tile_specific_code_under_fixed_c",
        "strong_tile_specific_code_under_native_c",
        "strong_weight_code_dependence_screen",
        "primary_fixed_c_load_bearing_pass",
        "native_native_robustness_pass",
        "strong_context_robust_all_16_pass",
        "directional_operator_cosine_pass",
        "fixed_c_tile_specific_with_directional_structure_pass",
        "fixed_c_universal_claim_excluded_by_any_endpoint",
    )
    expected = {
        "strong_tile_specific_code_under_fixed_c": recomputed["primary_fixed_c_load_bearing_pass"],
        "strong_tile_specific_code_under_native_c": recomputed["native_native_robustness_pass"],
        "strong_weight_code_dependence_screen": recomputed["strong_weight_code_dependence_screen"],
        "primary_fixed_c_load_bearing_pass": recomputed["primary_fixed_c_load_bearing_pass"],
        "native_native_robustness_pass": recomputed["native_native_robustness_pass"],
        "strong_context_robust_all_16_pass": recomputed["strong_context_robust_all_16_pass"],
        "directional_operator_cosine_pass": recomputed["directional_operator_cosine_pass"],
        "fixed_c_tile_specific_with_directional_structure_pass": recomputed[
            "fixed_c_tile_specific_with_directional_structure_pass"
        ],
        "fixed_c_universal_claim_excluded_by_any_endpoint": recomputed[
            "fixed_c_universal_claim_excluded_by_any_endpoint"
        ],
    }
    mismatches = {
        key: {"stored": stored.get(key), "recomputed": expected[key]}
        for key in keys
        if bool(stored.get(key)) != bool(expected[key])
    }
    if mismatches:
        raise RuntimeError(f"stored mechanism screen differs from frozen recomputation: {mismatches}")
    return {"pass": True, "stored_equals_recomputed": expected}


def role_depth_effects(matrix: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for comparator in ("permuted_within_row", "deranged_block"):
        selected = matrix[
            (matrix["code_condition"].isin(("correct", comparator)))
            & (
                ((matrix["encoder_condition"] == "cell") & (matrix["decoder_condition"] == "cell"))
                | ((matrix["encoder_condition"] == "native") & (matrix["decoder_condition"] == "native"))
            )
        ].copy()
        index = ["tiling_seed", "depth", "role", "encoder_condition", "decoder_condition"]
        wide = selected.pivot(
            index=index,
            columns="code_condition",
            values=["raw_E_X", "raw_operator_cosine"],
        )
        wide.columns = [f"{metric}_{code}" for metric, code in wide.columns]
        wide = wide.reset_index()
        if len(wide) != len(TILING_SEEDS) * NUM_MATRICES * 2:
            raise RuntimeError(f"role-depth paired table incomplete for {comparator}: {len(wide)}")
        wide["comparator"] = comparator
        wide["correct_minus_control_raw_E_X"] = wide["raw_E_X_correct"] - wide[f"raw_E_X_{comparator}"]
        wide["correct_minus_control_operator_cosine"] = (
            wide["raw_operator_cosine_correct"] - wide[f"raw_operator_cosine_{comparator}"]
        )
        wide["context"] = np.where(wide["encoder_condition"] == "cell", "cell_cell", "native_native")
        outputs.append(wide)
    output = pd.concat(outputs, ignore_index=True)
    expected = len(TILING_SEEDS) * NUM_MATRICES * 2 * 2
    if len(output) != expected:
        raise RuntimeError(f"combined role-depth code-effect table incomplete: {len(output)}/{expected}")
    return output


def wrong_context_effects(matrix: pd.DataFrame, distances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = matrix[
        (matrix["encoder_condition"] == "cell")
        & (matrix["code_condition"] == "correct")
        & matrix["decoder_condition"].isin(("cell", "wrong_role_same_depth", "same_role_wrong_depth"))
    ].copy()
    index = ["tiling_seed", "depth", "role"]
    wide = selected.pivot(
        index=index,
        columns="decoder_condition",
        values=["raw_E_X", "raw_E_W", "raw_operator_cosine", "raw_weight_cosine"],
    )
    wide.columns = [f"{metric}__{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    rows: list[pd.DataFrame] = []
    for condition in ("wrong_role_same_depth", "same_role_wrong_depth"):
        value = wide[index].copy()
        value["decoder_condition"] = condition
        for metric in ("raw_E_X", "raw_E_W", "raw_operator_cosine", "raw_weight_cosine"):
            value[f"wrong_minus_cell__{metric}"] = (
                wide[f"{metric}__{condition}"] - wide[f"{metric}__cell"]
            )
            value[f"cell__{metric}"] = wide[f"{metric}__cell"]
            value[f"wrong__{metric}"] = wide[f"{metric}__{condition}"]
        rows.append(value)
    effects = pd.concat(rows, ignore_index=True)
    merged = effects.merge(
        distances,
        left_on=["decoder_condition", "role", "depth"],
        right_on=["decoder_condition", "target_role", "target_depth"],
        validate="many_to_one",
    )
    if len(merged) != len(TILING_SEEDS) * NUM_MATRICES * 2:
        raise RuntimeError(f"wrong-template effect merge incomplete: {len(merged)}")
    correlations: list[dict[str, Any]] = []
    for (condition, seed), values in merged.groupby(["decoder_condition", "tiling_seed"], sort=True):
        correlations.append(
            {
                "decoder_condition": condition,
                "tiling_seed": int(seed),
                "n": len(values),
                "pearson_relative_distance_vs_delta_E_X": float(
                    values["relative_l2_distance"].corr(values["wrong_minus_cell__raw_E_X"], method="pearson")
                ),
                "spearman_relative_distance_vs_delta_E_X": float(
                    values["relative_l2_distance"].corr(values["wrong_minus_cell__raw_E_X"], method="spearman")
                ),
                "pearson_relative_distance_vs_delta_operator_cosine": float(
                    values["relative_l2_distance"].corr(
                        values["wrong_minus_cell__raw_operator_cosine"], method="pearson"
                    )
                ),
                "spearman_relative_distance_vs_delta_operator_cosine": float(
                    values["relative_l2_distance"].corr(
                        values["wrong_minus_cell__raw_operator_cosine"], method="spearman"
                    )
                ),
            }
        )
    correlation_frame = pd.DataFrame(correlations)
    finite_columns(
        correlation_frame,
        (
            "pearson_relative_distance_vs_delta_E_X",
            "spearman_relative_distance_vs_delta_E_X",
            "pearson_relative_distance_vs_delta_operator_cosine",
            "spearman_relative_distance_vs_delta_operator_cosine",
        ),
        label="wrong-template correlations",
    )
    return merged, correlation_frame


def save_2x2_plot(selected: pd.DataFrame, output: Path) -> None:
    metrics = (("raw_E_X", "raw E_X"), ("raw_operator_cosine", "operator cosine"))
    scopes = ("macro", "micro")
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for row_idx, (metric, metric_label) in enumerate(metrics):
        subset_metric = selected[selected["aggregation"].isin(scopes)]
        all_values = subset_metric[metric].astype(float).to_numpy()
        vmin, vmax = float(all_values.min()), float(all_values.max())
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        for seed_idx, seed in enumerate(TILING_SEEDS):
            for scope_idx, scope in enumerate(scopes):
                axis = axes[row_idx, seed_idx * 2 + scope_idx]
                values = selected[
                    (selected["tiling_seed"] == seed) & (selected["aggregation"] == scope)
                ].set_index(["encoder_condition", "decoder_condition"])
                matrix = np.asarray(
                    [
                        [values.loc[("cell", "cell"), metric], values.loc[("cell", "native"), metric]],
                        [values.loc[("native", "cell"), metric], values.loc[("native", "native"), metric]],
                    ],
                    dtype=float,
                )
                image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
                for i in range(2):
                    for j in range(2):
                        axis.text(j, i, f"{matrix[i, j]:.4g}", ha="center", va="center", color="white")
                axis.set_xticks((0, 1), ("Cdec cell", "Cdec native"))
                axis.set_yticks((0, 1), ("Cenc cell", "Cenc native"))
                axis.set_title(f"{metric_label} | {scope} | seed {seed}")
                figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Correct-code 2×2 encoder/decoder-context factorial (raw metrics)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_c_effect_plot(frame: pd.DataFrame, output: Path) -> None:
    contrasts = (
        "encoder_native_minus_cell_at_cell_decoder",
        "encoder_native_minus_cell_at_native_decoder",
        "decoder_native_minus_cell_at_cell_encoder",
        "decoder_native_minus_cell_at_native_encoder",
        "interaction_nn_minus_nc_minus_cn_plus_cc",
    )
    labels = ("enc@Cdec cell", "enc@Cdec native", "dec@Cenc cell", "dec@Cenc native", "interaction")
    metrics = (("E_X", "raw E_X contrast"), ("operator_cosine", "operator cosine contrast"))
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    colors = {"macro": "tab:blue", "micro": "tab:orange"}
    offsets = {"macro": -0.1, "micro": 0.1}
    x = np.arange(len(contrasts), dtype=float)
    for row_idx, (metric, metric_label) in enumerate(metrics):
        for col_idx, seed in enumerate(TILING_SEEDS):
            axis = axes[row_idx, col_idx]
            values = frame[(frame["metric"] == metric) & (frame["tiling_seed"] == seed)]
            for aggregation in ("macro", "micro"):
                indexed = values[values["aggregation"] == aggregation].set_index("contrast").loc[
                    list(contrasts)
                ]
                point = indexed["point_contrast"].astype(float).to_numpy()
                low = indexed["l95"].astype(float).to_numpy()
                high = indexed["u95"].astype(float).to_numpy()
                x_value = x + offsets[aggregation]
                axis.vlines(x_value, low, high, color=colors[aggregation], linewidth=1.5)
                axis.scatter(x_value, point, color=colors[aggregation], label=aggregation, s=32, zorder=3)
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
            axis.set_xticks(x, labels, rotation=28, ha="right")
            axis.set_ylabel(f"{metric_label} (5–95% paired bootstrap)")
            axis.set_title(f"correct code | seed {seed}")
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
    figure.suptitle("Correct-code Cenc×Cdec main effects and NN−NC−CN+CC interaction")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_paired_effect_plot(effect: pd.DataFrame, output: Path) -> None:
    selected = effect[
        (effect["metric"] == "E_X")
        & (effect["aggregation"].isin(AGGREGATIONS))
        & (
            ((effect["encoder_condition"] == "cell") & (effect["decoder_condition"] == "cell"))
            | ((effect["encoder_condition"] == "native") & (effect["decoder_condition"] == "native"))
        )
    ].copy()
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, constrained_layout=True)
    colors = {"permuted_within_row": "tab:blue", "deranged_block": "tab:orange", "zero": "tab:red"}
    offsets = {"permuted_within_row": -0.22, "deranged_block": 0.0, "zero": 0.22}
    x = np.arange(len(AGGREGATIONS), dtype=float)
    for row_idx, context in enumerate(("cell", "native")):
        for col_idx, seed in enumerate(TILING_SEEDS):
            axis = axes[row_idx, col_idx]
            values = selected[
                (selected["tiling_seed"] == seed)
                & (selected["encoder_condition"] == context)
                & (selected["decoder_condition"] == context)
            ]
            for comparator in ("permuted_within_row", "deranged_block", "zero"):
                indexed = values[values["comparator"] == comparator].set_index("aggregation").loc[
                    list(AGGREGATIONS)
                ]
                point = indexed["point_ratio_comparator_over_correct"].astype(float).to_numpy()
                low = indexed["ratio_l95"].astype(float).to_numpy()
                high = indexed["ratio_u95"].astype(float).to_numpy()
                x_value = x + offsets[comparator]
                axis.vlines(x_value, low, high, color=colors[comparator], linewidth=1.5)
                axis.scatter(x_value, point, color=colors[comparator], s=28, label=comparator, zorder=3)
            axis.axhline(1.0, color="black", linewidth=1, linestyle="--", label="no effect")
            axis.axhline(1.05, color="gray", linewidth=1, linestyle=":", label="5% point threshold")
            axis.set_xticks(x, AGGREGATIONS, rotation=35, ha="right")
            axis.set_title(f"{context}/{context} context | seed {seed}")
            axis.set_ylabel("raw E_X comparator / correct (5–95% bootstrap)")
            axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    figure.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(unique))
    figure.suptitle("Paired code effects; primary claim requires macro and micro on both tilings")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_paired_cosine_effect_plot(effect: pd.DataFrame, output: Path) -> None:
    selected = effect[
        (effect["metric"] == "operator_cosine")
        & (effect["aggregation"].isin(AGGREGATIONS))
        & (
            ((effect["encoder_condition"] == "cell") & (effect["decoder_condition"] == "cell"))
            | ((effect["encoder_condition"] == "native") & (effect["decoder_condition"] == "native"))
        )
    ].copy()
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, constrained_layout=True)
    colors = {"permuted_within_row": "tab:blue", "deranged_block": "tab:orange", "zero": "tab:red"}
    offsets = {"permuted_within_row": -0.22, "deranged_block": 0.0, "zero": 0.22}
    x = np.arange(len(AGGREGATIONS), dtype=float)
    for row_idx, context in enumerate(("cell", "native")):
        for col_idx, seed in enumerate(TILING_SEEDS):
            axis = axes[row_idx, col_idx]
            values = selected[
                (selected["tiling_seed"] == seed)
                & (selected["encoder_condition"] == context)
                & (selected["decoder_condition"] == context)
            ]
            for comparator in ("permuted_within_row", "deranged_block", "zero"):
                indexed = values[values["comparator"] == comparator].set_index("aggregation").loc[
                    list(AGGREGATIONS)
                ]
                point = indexed["point_delta_correct_minus_comparator"].astype(float).to_numpy()
                low = indexed["delta_l95"].astype(float).to_numpy()
                high = indexed["delta_u95"].astype(float).to_numpy()
                x_value = x + offsets[comparator]
                axis.vlines(x_value, low, high, color=colors[comparator], linewidth=1.5)
                axis.scatter(x_value, point, color=colors[comparator], s=28, label=comparator, zorder=3)
            axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
            axis.set_xticks(x, AGGREGATIONS, rotation=35, ha="right")
            axis.set_title(f"{context}/{context} context | seed {seed}")
            axis.set_ylabel("operator cosine(correct) − cosine(control), 5–95%")
            axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    figure.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    figure.suptitle("Paired operator-cosine code effects; descriptive companion to raw E_X decision")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_role_depth_heatmaps(frame: pd.DataFrame, output: Path) -> None:
    frame = frame[frame["comparator"] == "permuted_within_row"]
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    bound = float(np.nanmax(np.abs(frame["correct_minus_control_raw_E_X"].astype(float))))
    bound = max(bound, 1e-12)
    for row_idx, context in enumerate(("cell_cell", "native_native")):
        for col_idx, seed in enumerate(TILING_SEEDS):
            values = frame[(frame["context"] == context) & (frame["tiling_seed"] == seed)]
            pivot = values.pivot(index="role", columns="depth", values="correct_minus_control_raw_E_X").loc[
                list(ROLES), list(range(12))
            ]
            axis = axes[row_idx, col_idx]
            image = axis.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
            axis.set_xticks(range(12), range(12))
            axis.set_yticks(range(len(ROLES)), ROLES)
            axis.set_xlabel("ViT block depth")
            axis.set_title(f"{context} | seed {seed}")
            figure.colorbar(image, ax=axis, fraction=0.03)
    figure.suptitle("raw E_X(correct) − raw E_X(within-row permuted); negative means correct code is better")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_secondary_block_heatmaps(frame: pd.DataFrame, output: Path) -> None:
    frame = frame[(frame["comparator"] == "deranged_block") & (frame["context"] == "cell_cell")]
    metrics = (
        ("correct_minus_control_raw_E_X", "raw E_X correct − depth+6"),
        ("correct_minus_control_operator_cosine", "operator cosine correct − depth+6"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for row_idx, (metric, label) in enumerate(metrics):
        bound = max(float(np.nanmax(np.abs(frame[metric].astype(float)))), 1e-12)
        for col_idx, seed in enumerate(TILING_SEEDS):
            values = frame[frame["tiling_seed"] == seed]
            pivot = values.pivot(index="role", columns="depth", values=metric).loc[
                list(ROLES), list(range(12))
            ]
            axis = axes[row_idx, col_idx]
            image = axis.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
            axis.set_xticks(range(12), range(12))
            axis.set_yticks(range(len(ROLES)), ROLES)
            axis.set_xlabel("ViT block depth")
            axis.set_title(f"{label} | fixed C | seed {seed}")
            figure.colorbar(image, ax=axis, fraction=0.03)
    figure.suptitle("Secondary depth+6 wrong-layer control; six-orbit inference is reported separately")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_latent_distribution_plot(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for row_idx, comparator in enumerate(("permuted_within_row", "deranged_block")):
        for encoder_idx, encoder in enumerate(ENC_CONDITIONS):
            cosine_axis = axes[row_idx, encoder_idx * 2]
            norm_axis = axes[row_idx, encoder_idx * 2 + 1]
            values = frame[(frame["comparator"] == comparator) & (frame["encoder_condition"] == encoder)].copy()
            for seed in TILING_SEEDS:
                selected = values[values["tiling_seed"] == seed]
                cosine_axis.hist(
                    selected["cosine"].astype(float),
                    bins=80,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=f"seed {seed}",
                )
                norm_axis.hist(
                    selected["donor_norm"].astype(float) / selected["correct_norm"].astype(float),
                    bins=80,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=f"norm ratio seed {seed}",
                )
                median_distance = float(selected["relative_distance"].astype(float).median())
                norm_axis.axvline(
                    float((selected["donor_norm"].astype(float) / selected["correct_norm"].astype(float)).median()),
                    linewidth=1,
                    linestyle=":",
                    label=f"median rel-L2={median_distance:.3g}",
                )
            cosine_axis.set_title(f"{comparator} | Cenc={encoder} | cosine")
            cosine_axis.set_xlabel("correct/control latent cosine")
            cosine_axis.set_ylabel("density")
            cosine_axis.grid(alpha=0.2)
            cosine_axis.legend(fontsize=7)
            norm_axis.set_title(f"{comparator} | Cenc={encoder} | norm")
            norm_axis.set_xlabel("control / correct latent L2 norm")
            norm_axis.set_ylabel("density")
            norm_axis.grid(alpha=0.2)
            norm_axis.legend(fontsize=7)
    figure.suptitle("Latent intervention separation: cosine, norm-ratio, and relative-distance diagnostics")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_wrong_distance_plot(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    role_colors = {role: plt.get_cmap("tab10")(index) for index, role in enumerate(ROLES)}
    for axis, condition in zip(
        axes,
        ("wrong_role_same_depth", "same_role_wrong_depth"),
        strict=True,
    ):
        values = frame[frame["decoder_condition"] == condition]
        for role in ROLES:
            selected = values[values["role"] == role]
            axis.scatter(
                selected["relative_l2_distance"],
                selected["wrong_minus_cell__raw_E_X"],
                s=24,
                alpha=0.75,
                color=role_colors[role],
                label=role,
            )
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_xlabel("wrong-vs-correct decoder C_patch relative L2 distance")
        axis.set_ylabel("raw E_X(wrong Cdec) − raw E_X(cell Cdec)")
        axis.set_title(condition)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(ROLES))
    figure.suptitle("Decoder-context effect versus actual source-template distance")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def validate_plot(path: Path) -> dict[str, Any]:
    values = plt.imread(path)
    if values.ndim not in (2, 3) or values.shape[0] < 500 or values.shape[1] < 500:
        raise RuntimeError(f"plot has invalid dimensions: {path}: {values.shape}")
    if not np.isfinite(values).all() or float(np.std(values.astype(np.float64))) <= 1e-6:
        raise RuntimeError(f"plot is blank or non-finite: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "shape": list(values.shape)}


def write_review(
    *,
    output: Path,
    identity: dict[str, Any],
    validity: dict[str, Any],
    mechanism: dict[str, Any],
    interaction: pd.DataFrame,
    correlations: pd.DataFrame,
    plots: list[dict[str, Any]],
) -> None:
    interaction_primary = interaction[
        (interaction["aggregation"].isin(("macro", "micro")))
        & (interaction["metric"].isin(("raw_E_X", "raw_operator_cosine")))
    ]
    lines = [
        "# Frozen post-run review: source latent-code factorial",
        "",
        f"Analyzer SHA256: `{identity['analyzer_sha256']}`.",
        "",
        "## Validity",
        "",
        "All mandatory preinterpretation gates passed: exact 4,608-row factorial grid, 28 unique "
        "decoder calls per matrix, 288/288 exact parent prediction hashes, sufficient-stat parity, "
        "runner artifact SHA/size seals, independent aggregate/bootstrap recomputation, decoder-path "
        "preflight, zero-code reuse, bit-identical native C within each row group, nonzero "
        "correct/control latent differences, and separate nondegenerate wrong-role and wrong-depth "
        "C_patch interventions.",
        "",
        "## Preregistered result",
        "",
        mechanism["primary_fixed_c_criterion"] + ".",
        "",
        f"Primary fixed-C rows: {mechanism['primary_fixed_c_passed_rows']}/"
        f"{mechanism['primary_fixed_c_required_rows']}; pass={mechanism['primary_fixed_c_load_bearing_pass']}.",
        f"Native/native robustness rows: {mechanism['native_robustness_passed_rows']}/"
        f"{mechanism['native_robustness_required_rows']}; pass={mechanism['native_native_robustness_pass']}.",
        f"Context-robust all-16 tier: {mechanism['strong_context_robust_all_16_pass']}.",
        f"Directional operator-cosine rows: {mechanism['directional_passed_rows']}/"
        f"{mechanism['directional_required_rows']}; pass={mechanism['directional_operator_cosine_pass']}.",
        f"Primary status: {mechanism['primary_fixed_c_status']}.",
        "",
    ]
    if not mechanism["primary_fixed_c_load_bearing_pass"]:
        lines.extend([mechanism["interpretation_if_not_pass"] + ".", ""])
    lines.extend([mechanism["scope_limit"] + ".", ""])
    lines.extend(
        [
            "## Fixed diagnostic outputs",
            "",
            "- `mechanism_criterion_rows.csv`: every row entering the exact success decision.",
            "- `directional_operator_criterion_rows.csv`: the four rows gating directional wording.",
            "- `correct_code_2x2.csv` and `correct_code_c_interactions.csv`: raw Cenc×Cdec cells and "
            "difference-in-differences, without causal interpretation from scale alone.",
            "- `role_depth_code_effects.csv`: primary within-row and secondary depth+6 localization.",
            "- `latent_intervention_distribution_summary.csv`: latent cosine, norm, and distance summaries.",
            "- `wrong_context_effect_vs_template_distance.csv`: separate wrong-role and wrong-depth effects.",
            "",
            "The 2×2 interactions and template-distance correlations are descriptive mechanism diagnostics. "
            "They do not replace the preregistered paired raw-E_X rule.",
            "",
            "## Compact descriptive values",
            "",
            "```text",
            interaction_primary.to_string(index=False),
            "```",
            "",
            "```text",
            correlations.to_string(index=False),
            "```",
            "",
            f"Validated plot files: {len(plots)}.",
            f"Validity summary keys: {sorted(validity)}.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    output = (args.output_dir or (run_dir / "postrun_analysis")).resolve()
    if not output.is_relative_to(run_dir) or output == run_dir:
        raise RuntimeError(f"analysis output must be a child of the immutable run directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"analysis output directory must be fresh and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "analysis.log", mode="w")],
        force=True,
    )
    logger = logging.getLogger("source_latent_code_factorial_analysis")
    started = time.monotonic()
    logger.info("stage=identity run_dir=%s output=%s analyzer=%s", run_dir, output, Path(__file__).resolve())
    identity = validate_frozen_identity(run_dir)

    required_files = (
        MATRIX_FILE,
        AGGREGATE_FILE,
        EFFECT_FILE,
        COSINE_EFFECT_FILE,
        C_EFFECT_FILE,
        LATENT_TILE_FILE,
        LATENT_SUMMARY_FILE,
        NATIVE_ROW_FILE,
        TEMPLATE_DISTANCE_FILE,
    )
    missing = [str(run_dir / name) for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required factorial artifacts are missing: {missing}")
    logger.info("stage=read_tables")
    matrix = pd.read_csv(run_dir / MATRIX_FILE)
    aggregate = pd.read_csv(run_dir / AGGREGATE_FILE)
    effect = pd.read_csv(run_dir / EFFECT_FILE)
    cosine_effect = pd.read_csv(run_dir / COSINE_EFFECT_FILE)
    c_effect = pd.read_csv(run_dir / C_EFFECT_FILE)
    latent_tiles = pd.read_csv(run_dir / LATENT_TILE_FILE)
    latent_summary = pd.read_csv(run_dir / LATENT_SUMMARY_FILE)
    native_rows = pd.read_csv(run_dir / NATIVE_ROW_FILE)
    distances = pd.read_csv(run_dir / TEMPLATE_DISTANCE_FILE)

    logger.info("stage=validity_checks")
    draws = int(identity["resolved_config"]["bootstrap_draws"])
    paired_seed = int(identity["resolved_config"]["bootstrap_seeds"]["paired_code"])
    c_seed = int(identity["resolved_config"]["bootstrap_seeds"]["paired_c"])
    factorial_seed = int(identity["resolved_config"]["factorial_seed"])
    if paired_seed != factorial_seed + 99 or c_seed != factorial_seed + 199:
        raise RuntimeError("resolved bootstrap seeds do not match the frozen derivation rule")
    validity = {
        "identity": {
            "analyzer_sha256": identity["analyzer_sha256"],
            "source_only_target_access": False,
        },
        "stored_validity": validate_validity_artifacts(run_dir),
        "matrix_grid": validate_matrix_grid(matrix),
        "aggregate_recomputation": validate_aggregate_metrics(aggregate, matrix),
        "effects": validate_effects(effect, matrix, draws=draws, seed=paired_seed),
        "cosine_effects": validate_cosine_effects(
            cosine_effect, matrix, draws=draws, seed=paired_seed
        ),
        "c_effects": validate_c_effects(c_effect, matrix, draws=draws, seed=c_seed),
        "latent_diagnostics": validate_latent_diagnostics(latent_tiles, latent_summary),
        "native_row_condition_invariance": validate_native_row_invariance(native_rows),
        "template_distances": validate_template_distances(distances),
    }

    logger.info("stage=fixed_estimands")
    two_by_two, interactions = correct_code_2x2(aggregate)
    criterion_rows, directional_rows, mechanism = mechanism_criterion(effect, cosine_effect)
    validity["stored_mechanism_screen"] = validate_stored_mechanism_screen(run_dir, mechanism)
    role_depth = role_depth_effects(matrix)
    wrong_effects, correlations = wrong_context_effects(matrix, distances)
    latent_distribution_summary = (
        latent_tiles.groupby(["tiling_seed", "encoder_condition", "comparator"])[
            ["correct_norm", "donor_norm", "difference_norm", "relative_distance", "cosine"]
        ]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    latent_distribution_summary.columns = [
        "__".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in latent_distribution_summary.columns
    ]

    two_by_two.to_csv(output / "correct_code_2x2.csv", index=False)
    interactions.to_csv(output / "correct_code_c_interactions.csv", index=False)
    criterion_rows.to_csv(output / "mechanism_criterion_rows.csv", index=False)
    directional_rows.to_csv(output / "directional_operator_criterion_rows.csv", index=False)
    role_depth.to_csv(output / "role_depth_code_effects.csv", index=False)
    latent_distribution_summary.to_csv(output / "latent_intervention_distribution_summary.csv", index=False)
    wrong_effects.to_csv(output / "wrong_context_effect_vs_template_distance.csv", index=False)
    correlations.to_csv(output / "wrong_context_template_distance_correlations.csv", index=False)
    write_json(output / "mechanism_screen_recomputed.json", mechanism)

    logger.info("stage=fixed_plots")
    plot_paths = [
        output / "correct_code_2x2.png",
        output / "correct_code_c_effects_and_interaction.png",
        output / "paired_code_effect_ratios.png",
        output / "paired_operator_cosine_effects.png",
        output / "role_depth_primary_within_row.png",
        output / "role_depth_secondary_depth_plus_6.png",
        output / "latent_cosine_norm_distributions.png",
        output / "wrong_context_effect_vs_template_distance.png",
    ]
    save_2x2_plot(two_by_two, plot_paths[0])
    save_c_effect_plot(c_effect, plot_paths[1])
    save_paired_effect_plot(effect, plot_paths[2])
    save_paired_cosine_effect_plot(cosine_effect, plot_paths[3])
    save_role_depth_heatmaps(role_depth, plot_paths[4])
    save_secondary_block_heatmaps(role_depth, plot_paths[5])
    save_latent_distribution_plot(latent_tiles, plot_paths[6])
    save_wrong_distance_plot(wrong_effects, plot_paths[7])
    plots = [validate_plot(path) for path in plot_paths]
    validity["plots"] = plots
    validity["pass"] = True
    write_json(output / "review_validity.json", validity)
    write_review(
        output=output,
        identity=identity,
        validity=validity,
        mechanism=mechanism,
        interaction=interactions,
        correlations=correlations,
        plots=plots,
    )

    artifacts = sorted(
        {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in output.iterdir()
            if path.is_file() and path.name not in {"postrun_analysis_manifest.json", "analysis.log"}
        }.values(),
        key=lambda row: row["path"],
    )
    write_json(
        output / "postrun_analysis_manifest.json",
        {
            "status": "COMPLETE_FROZEN_POSTRUN_ANALYSIS_REQUIRES_HUMAN_PLOT_AND_RESULT_REVIEW",
            "elapsed_seconds": time.monotonic() - started,
            "source_run_dir": str(run_dir),
            "analyzer_sha256": identity["analyzer_sha256"],
            "target_access": False,
            "validity_pass": True,
            "mechanism_screen": mechanism,
            "artifacts": artifacts,
        },
    )
    logger.info(
        "completed status=REQUIRES_HUMAN_REVIEW elapsed=%.1fs screen=%s artifacts=%s",
        time.monotonic() - started,
        mechanism["strong_weight_code_dependence_screen"],
        output,
    )


if __name__ == "__main__":
    main()
