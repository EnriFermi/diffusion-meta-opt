#!/usr/bin/env python3
"""Post-hoc source-only direction-versus-scale mechanism diagnostic.

This analysis consumes only sealed sufficient statistics and source-fitted
role gains.  It performs no model forward pass and must not access target
domain assets.  Its result cannot alter the locked raw-factorial verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Iterable

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
TILING_SEEDS = (26_081_601, 26_081_602)
PRIMARY_CONTEXT = ("cell", "cell")
ROBUSTNESS_CONTEXT = ("native", "native")
CONTEXTS = (PRIMARY_CONTEXT, ROBUSTNESS_CONTEXT)
CODES = ("correct", "permuted_within_row", "zero")
COMPARATORS = ("permuted_within_row", "zero")
AGGREGATIONS = ("macro", "micro", *ROLES)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 26_081_776
REL_TOL = 1e-12

EXPECTED_GAINS = {
    "attn_query": 0.5270182885626962,
    "attn_key": 0.5562907139098529,
    "attn_value": 0.3990051254514762,
    "attn_output": 0.3564476277035192,
    "ffn_up": 1.0280206811266708,
    "ffn_down": 0.3048084709039325,
}
SOURCE_MEDIAN_GAIN = float(np.median(list(EXPECTED_GAINS.values())))
SOURCE_MEAN_GAIN = float(np.mean(list(EXPECTED_GAINS.values())))

PROJECT = Path("/home/coder/project")
FACTORIAL_DIR = PROJECT / "artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816"
PARENT_DIR = PROJECT / "artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2"
DEFAULT_OUTPUT = PROJECT / "artifacts/crossmodal_united_structure/source_role_gain_mechanism_20260816"
DESIGN_PATH = PROJECT / "docs/notes/source_role_gain_mechanism_diagnostic_design_20260816.md"

INPUTS = {
    "factorial_matrix": (
        FACTORIAL_DIR / "factorial_matrix_sufficient_stats.csv",
        "3ea4e95e5570547d17e6ab718e1a1f9b1e97a35e99266aa36571440604e5143a",
    ),
    "source_gains": (
        PARENT_DIR / "source_fit_role_gains.csv",
        "e05f2339ba33eb65feee00df2f095655c9319ecf118cd81d7159f6e14f70c4f3",
    ),
    "gain_manifest": (
        PARENT_DIR / "gain_sampling_manifest.json",
        "2512a92f848d27ab589fb69ebc5989ed0cd15b46545d6f763dc3aac068648791",
    ),
    "heldout_manifest": (
        PARENT_DIR / "heldout_panel_manifest.json",
        "10020aeabfcce6fe2df3003a7a626f21735cd0763c664e178f27ebcbf42ff985",
    ),
    "parent_matrix": (
        PARENT_DIR / "matrix_metrics.csv",
        "d9d42484c55bd74a51da773edd3596306ad7f4b7d7f6055da90e1f284e7d19a1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def relative_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-300)


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("source_role_gain_mechanism")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def validate_input_hashes() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, (path, expected) in INPUTS.items():
        actual = sha256_file(path.resolve(strict=True))
        if actual != expected:
            raise RuntimeError(f"input hash mismatch for {label}: {actual} != {expected}")
        output[label] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}
    output["design"] = {
        "path": str(DESIGN_PATH.resolve(strict=True)),
        "sha256": sha256_file(DESIGN_PATH),
        "bytes": DESIGN_PATH.stat().st_size,
    }
    output["script"] = {
        "path": str(Path(__file__).resolve(strict=True)),
        "sha256": sha256_file(Path(__file__).resolve()),
        "bytes": Path(__file__).stat().st_size,
    }
    return output


def validate_target_seal() -> dict[str, Any]:
    seal = read_json(FACTORIAL_DIR / "target_access_seal.json")
    run_manifest = read_json(FACTORIAL_DIR / "run_manifest.json")
    resolved = read_json(FACTORIAL_DIR / "resolved_config.json")
    checks = {
        "factorial_seal_false": seal.get("target_data2vec_access") is False,
        "factorial_manifest_target_false": run_manifest.get("target_access") is False,
        "factorial_manifest_data2vec_false": run_manifest.get("target_data2vec_access") is False,
        "factorial_config_target_false": resolved.get("target_access") is False,
        "bootstrap_draws_match": int(resolved.get("bootstrap_draws", -1)) == BOOTSTRAP_DRAWS,
        "bootstrap_seed_match": int(resolved.get("bootstrap_seeds", {}).get("paired_code", -1))
        == BOOTSTRAP_SEED,
    }
    if not all(checks.values()):
        raise RuntimeError(f"target/bootstrap seal failed: {checks}")
    return {"pass": True, "checks": checks}


def validate_disjointness() -> dict[str, Any]:
    gain_rows = read_json(INPUTS["gain_manifest"][0])
    heldout_raw = read_json(INPUTS["heldout_manifest"][0])
    heldout_rows = [row["context_a"] for row in heldout_raw]
    if len(gain_rows) != 72 or len(heldout_rows) != 72:
        raise RuntimeError(f"unexpected gain/heldout sizes: {len(gain_rows)}/{len(heldout_rows)}")

    def field_set(rows: list[dict[str, Any]], field: str) -> set[str]:
        return {str(row[field]) for row in rows}

    intersections = {
        field: sorted(field_set(gain_rows, field) & field_set(heldout_rows, field))
        for field in ("model_name", "primary_dataset", "source_key")
    }
    role_counts = pd.Series([row["role"] for row in gain_rows]).value_counts().to_dict()
    heldout_role_counts = pd.Series([row["role"] for row in heldout_rows]).value_counts().to_dict()
    checks = {
        "gain_rows_72": len(gain_rows) == 72,
        "heldout_rows_72": len(heldout_rows) == 72,
        "gain_source_keys_unique": len(field_set(gain_rows, "source_key")) == 72,
        "heldout_source_keys_unique": len(field_set(heldout_rows, "source_key")) == 72,
        "gain_12_per_role": all(int(role_counts.get(role, 0)) == 12 for role in ROLES),
        "heldout_12_per_role": all(int(heldout_role_counts.get(role, 0)) == 12 for role in ROLES),
        "model_disjoint": not intersections["model_name"],
        "dataset_disjoint": not intersections["primary_dataset"],
        "source_key_disjoint": not intersections["source_key"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"gain/heldout disjointness failed: {checks}; intersections={intersections}")
    return {
        "pass": True,
        "checks": checks,
        "intersections": intersections,
        "gain_models": sorted(field_set(gain_rows, "model_name")),
        "gain_datasets": sorted(field_set(gain_rows, "primary_dataset")),
        "heldout_models": sorted(field_set(heldout_rows, "model_name")),
        "heldout_datasets": sorted(field_set(heldout_rows, "primary_dataset")),
        "gain_role_counts": role_counts,
        "heldout_role_counts": heldout_role_counts,
    }


def load_and_validate_gains() -> dict[str, float]:
    frame = pd.read_csv(INPUTS["source_gains"][0])
    selected = frame[frame["method"] == "ae_cell_mean_c0"]
    if len(selected) != len(ROLES) or set(selected["role"]) != set(ROLES):
        raise RuntimeError("source gain table does not contain exactly six primary role gains")
    actual = {str(row.role): float(row.gain) for row in selected.itertuples(index=False)}
    for role in ROLES:
        if not math.isclose(actual[role], EXPECTED_GAINS[role], rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"frozen gain mismatch for {role}: {actual[role]} != {EXPECTED_GAINS[role]}")
        if not (0.0 < actual[role] < 4.0):
            raise RuntimeError(f"invalid positive gain for {role}: {actual[role]}")
    if not math.isclose(SOURCE_MEDIAN_GAIN, 0.46301170700708616, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("source median gain contract drift")
    if not math.isclose(SOURCE_MEAN_GAIN, 0.5285984846096913, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("source mean gain contract drift")
    return actual


def validate_matrix(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "tiling_seed",
        "encoder_condition",
        "code_condition",
        "decoder_condition",
        "depth",
        "role",
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"factorial matrix is missing columns: {missing}")
    if len(frame) != 4608:
        raise RuntimeError(f"factorial row count is {len(frame)}, expected 4608")
    key = ["tiling_seed", "encoder_condition", "code_condition", "decoder_condition", "depth", "role"]
    if frame.duplicated(key).any():
        raise RuntimeError("factorial matrix contains duplicate formal cells")
    numeric = [
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
    ]
    values = frame[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("factorial matrix contains nonfinite sufficient statistics")
    if (frame[["w_target", "w_pred", "x_target", "x_pred"]].to_numpy() <= 0).any():
        raise RuntimeError("factorial matrix contains nonpositive target/prediction energies")
    expected_cells = 2 * 2 * 4 * 4 * 12 * 6
    if expected_cells != len(frame):
        raise RuntimeError("internal factorial size contract is inconsistent")
    exact_sets = {
        "tiling_seed": set(TILING_SEEDS),
        "encoder_condition": {"cell", "native"},
        "code_condition": {"correct", "permuted_within_row", "deranged_block", "zero"},
        "decoder_condition": {"cell", "native", "wrong_role_same_depth", "same_role_wrong_depth"},
        "depth": set(range(12)),
        "role": set(ROLES),
    }
    for column, expected in exact_sets.items():
        observed = set(frame[column].tolist())
        if observed != expected:
            raise RuntimeError(f"factorial grid values differ for {column}: {observed} != {expected}")
    arm_sizes = frame.groupby(
        ["tiling_seed", "encoder_condition", "code_condition", "decoder_condition"], sort=False
    ).size()
    if len(arm_sizes) != 64 or set(arm_sizes.tolist()) != {72}:
        raise RuntimeError(f"factorial formal-arm completeness failed: groups={len(arm_sizes)} sizes={set(arm_sizes)}")
    for prefix, metric in (("x", "raw_E_X"), ("w", "raw_E_W")):
        recomputed_error = frame[f"{prefix}_target"] - 2 * frame[f"{prefix}_dot"] + frame[f"{prefix}_pred"]
        error_rel = relative_error(recomputed_error.to_numpy(), frame[f"{prefix}_error"].to_numpy())
        recomputed_metric = recomputed_error / frame[f"{prefix}_target"]
        metric_rel = relative_error(recomputed_metric.to_numpy(), frame[metric].to_numpy())
        if float(error_rel.max()) > REL_TOL or float(metric_rel.max()) > REL_TOL:
            raise RuntimeError(
                f"g=1 parity failed for {prefix}: error={error_rel.max()} metric={metric_rel.max()}"
            )
    return {
        "pass": True,
        "rows": len(frame),
        "unique_cells": int(frame[key].drop_duplicates().shape[0]),
        "g1_max_relative_error_X": float(
            relative_error(
                (
                    frame["x_target"] - 2 * frame["x_dot"] + frame["x_pred"]
                ).to_numpy(),
                frame["x_error"].to_numpy(),
            ).max()
        ),
        "g1_max_relative_error_W": float(
            relative_error(
                (
                    frame["w_target"] - 2 * frame["w_dot"] + frame["w_pred"]
                ).to_numpy(),
                frame["w_error"].to_numpy(),
            ).max()
        ),
    }


def validate_parent_parity(frame: pd.DataFrame, gains: dict[str, float]) -> dict[str, Any]:
    current = frame[
        (frame["encoder_condition"] == "cell")
        & (frame["code_condition"] == "correct")
        & (frame["decoder_condition"] == "cell")
    ].copy()
    parent = pd.read_csv(INPUTS["parent_matrix"][0])
    parent = parent[parent["method"] == "ae_cell_mean_c0"].copy()
    if len(current) != 144 or len(parent) != 144:
        raise RuntimeError(f"parent parity expects 144/144 rows, got {len(current)}/{len(parent)}")
    keys = ["tiling_seed", "depth", "role"]
    merged = current.merge(parent, on=keys, suffixes=("_factorial", "_parent"), validate="one_to_one")
    if len(merged) != 144:
        raise RuntimeError("parent parity merge is incomplete")
    columns = ("w_target", "w_pred", "w_dot", "w_error", "x_target", "x_pred", "x_dot", "x_error")
    max_rel = 0.0
    for column in columns:
        rel = relative_error(
            merged[f"{column}_factorial"].to_numpy(dtype=np.float64),
            merged[f"{column}_parent"].to_numpy(dtype=np.float64),
        )
        max_rel = max(max_rel, float(rel.max()))
    if not (merged["prediction_sha256_factorial"] == merged["prediction_sha256_parent"]).all():
        raise RuntimeError("parent parity prediction hashes differ")
    for prefix, parent_metric in (("x", "calibrated_E_X"), ("w", "calibrated_E_W")):
        g = merged["role"].map(gains).to_numpy(dtype=np.float64)
        target = merged[f"{prefix}_target_factorial"].to_numpy(dtype=np.float64)
        pred = merged[f"{prefix}_pred_factorial"].to_numpy(dtype=np.float64)
        dot = merged[f"{prefix}_dot_factorial"].to_numpy(dtype=np.float64)
        derived = (target - 2 * g * dot + g * g * pred) / target
        rel = relative_error(derived, merged[parent_metric].to_numpy(dtype=np.float64))
        max_rel = max(max_rel, float(rel.max()))
    if max_rel > REL_TOL:
        raise RuntimeError(f"parent calibrated parity exceeds tolerance: {max_rel}")
    return {"pass": True, "rows": len(merged), "max_relative_error": max_rel, "prediction_hashes_exact": True}


def gain_for(calibration: str, role: str, gains: dict[str, float]) -> float:
    if calibration == "raw":
        return 1.0
    if calibration == "source_role_matched":
        return gains[role]
    if calibration == "source_median_common":
        return SOURCE_MEDIAN_GAIN
    if calibration == "source_mean_common":
        return SOURCE_MEAN_GAIN
    raise KeyError(calibration)


CALIBRATIONS = ("raw", "source_role_matched", "source_median_common", "source_mean_common")


def scaled_components(target: float, pred: float, dot: float, gain: float) -> dict[str, float]:
    if gain < 0.0:
        raise RuntimeError(f"negative gain violates the frozen nonnegative-scaling contract: {gain}")
    error = target - 2.0 * gain * dot + gain * gain * pred
    tolerance = 1e-12 * max(target, gain * gain * pred, 1.0)
    if error < -tolerance:
        raise RuntimeError(f"derived SSE is materially negative: {error}")
    error = max(error, 0.0)
    scaled_pred = gain * gain * pred
    scaled_dot = gain * dot
    base_norm_ratio = math.sqrt(max(pred / target, 0.0))
    cosine = dot / math.sqrt(max(target * pred, 1e-300))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    norm_ratio = gain * base_norm_ratio
    angular_floor = 1.0 - cosine * cosine
    radial_penalty = (gain * base_norm_ratio - cosine) ** 2
    normalized_error = error / target
    if not math.isclose(normalized_error, angular_floor + radial_penalty, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("radial/angular decomposition parity failed")
    return {
        "error": error,
        "E": normalized_error,
        "pred_scaled": scaled_pred,
        "dot_scaled": scaled_dot,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "angular_floor": angular_floor,
        "radial_penalty": radial_penalty,
    }


def build_calibrated_matrix(frame: pd.DataFrame, gains: dict[str, float]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    id_columns = [
        "tiling_seed",
        "encoder_condition",
        "code_condition",
        "decoder_condition",
        "depth",
        "role",
        "prediction_sha256",
    ]
    for row in frame.to_dict("records"):
        for calibration in CALIBRATIONS:
            gain = gain_for(calibration, str(row["role"]), gains)
            x = scaled_components(row["x_target"], row["x_pred"], row["x_dot"], gain)
            w = scaled_components(row["w_target"], row["w_pred"], row["w_dot"], gain)
            record = {column: row[column] for column in id_columns}
            record.update(
                {
                    "calibration": calibration,
                    "gain": gain,
                    "x_target": row["x_target"],
                    "x_pred": row["x_pred"],
                    "x_dot": row["x_dot"],
                    "E_X": x["E"],
                    "x_error_scaled": x["error"],
                    "operator_cosine": x["cosine"],
                    "operator_norm_ratio": x["norm_ratio"],
                    "operator_angular_floor": x["angular_floor"],
                    "operator_radial_penalty": x["radial_penalty"],
                    "w_target": row["w_target"],
                    "w_pred": row["w_pred"],
                    "w_dot": row["w_dot"],
                    "E_W": w["E"],
                    "w_error_scaled": w["error"],
                    "weight_cosine": w["cosine"],
                    "weight_norm_ratio": w["norm_ratio"],
                    "weight_angular_floor": w["angular_floor"],
                    "weight_radial_penalty": w["radial_penalty"],
                }
            )
            output.append(record)
    result = pd.DataFrame(output)
    numeric = result.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("calibrated matrix contains nonfinite values")
    return result


def aggregate_calibrated(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["tiling_seed", "encoder_condition", "code_condition", "decoder_condition", "calibration"]
    output: list[dict[str, Any]] = []
    for key, selected in frame.groupby(groups, sort=True):
        role_rows: list[dict[str, Any]] = []
        for role in ROLES:
            rows = selected[selected["role"] == role]
            if len(rows) != 12:
                raise RuntimeError(f"aggregate cell {key}/{role} has {len(rows)} rows")
            x = scaled_components(
                float(rows["x_target"].sum()),
                float((rows["gain"] ** 2 * rows["x_pred"]).sum()),
                float((rows["gain"] * rows["x_dot"]).sum()),
                1.0,
            )
            w = scaled_components(
                float(rows["w_target"].sum()),
                float((rows["gain"] ** 2 * rows["w_pred"]).sum()),
                float((rows["gain"] * rows["w_dot"]).sum()),
                1.0,
            )
            role_rows.append(
                {
                    "aggregation": role,
                    "matrices": 12,
                    "x_target": float(rows["x_target"].sum()),
                    "x_error": x["error"],
                    "E_X": x["E"],
                    "operator_cosine": x["cosine"],
                    "operator_norm_ratio": x["norm_ratio"],
                    "operator_angular_floor": x["angular_floor"],
                    "operator_radial_penalty": x["radial_penalty"],
                    "w_target": float(rows["w_target"].sum()),
                    "w_error": w["error"],
                    "E_W": w["E"],
                    "weight_cosine": w["cosine"],
                    "weight_norm_ratio": w["norm_ratio"],
                    "weight_angular_floor": w["angular_floor"],
                    "weight_radial_penalty": w["radial_penalty"],
                    "gain_min": float(rows["gain"].min()),
                    "gain_max": float(rows["gain"].max()),
                }
            )
        base = dict(zip(groups, key))
        for row in role_rows:
            output.append({**base, **row})
        output.append(
            {
                **base,
                "aggregation": "macro",
                "matrices": 72,
                "x_target": float(sum(row["x_target"] for row in role_rows)),
                "x_error": float(sum(row["x_error"] for row in role_rows)),
                "E_X": float(np.mean([row["E_X"] for row in role_rows])),
                "operator_cosine": float(np.mean([row["operator_cosine"] for row in role_rows])),
                "operator_norm_ratio": float(np.mean([row["operator_norm_ratio"] for row in role_rows])),
                "operator_angular_floor": float(
                    np.mean([row["operator_angular_floor"] for row in role_rows])
                ),
                "operator_radial_penalty": float(
                    np.mean([row["operator_radial_penalty"] for row in role_rows])
                ),
                "w_target": float(sum(row["w_target"] for row in role_rows)),
                "w_error": float(sum(row["w_error"] for row in role_rows)),
                "E_W": float(np.mean([row["E_W"] for row in role_rows])),
                "weight_cosine": float(np.mean([row["weight_cosine"] for row in role_rows])),
                "weight_norm_ratio": float(np.mean([row["weight_norm_ratio"] for row in role_rows])),
                "weight_angular_floor": float(np.mean([row["weight_angular_floor"] for row in role_rows])),
                "weight_radial_penalty": float(np.mean([row["weight_radial_penalty"] for row in role_rows])),
                "gain_min": float(min(row["gain_min"] for row in role_rows)),
                "gain_max": float(max(row["gain_max"] for row in role_rows)),
            }
        )
        x_target = float(sum(row["x_target"] for row in role_rows))
        x_error = float(sum(row["x_error"] for row in role_rows))
        x_pred = float((selected["gain"] ** 2 * selected["x_pred"]).sum())
        x_dot = float((selected["gain"] * selected["x_dot"]).sum())
        x = scaled_components(x_target, x_pred, x_dot, 1.0)
        w_target = float(sum(row["w_target"] for row in role_rows))
        w_error = float(sum(row["w_error"] for row in role_rows))
        w_pred = float((selected["gain"] ** 2 * selected["w_pred"]).sum())
        w_dot = float((selected["gain"] * selected["w_dot"]).sum())
        w = scaled_components(w_target, w_pred, w_dot, 1.0)
        if not math.isclose(x["error"], x_error, rel_tol=1e-12, abs_tol=1e-8):
            raise RuntimeError("micro X aggregate SSE mismatch")
        if not math.isclose(w["error"], w_error, rel_tol=1e-12, abs_tol=1e-8):
            raise RuntimeError("micro W aggregate SSE mismatch")
        output.append(
            {
                **base,
                "aggregation": "micro",
                "matrices": 72,
                "x_target": x_target,
                "x_error": x_error,
                "E_X": x_error / x_target,
                "operator_cosine": x["cosine"],
                "operator_norm_ratio": x["norm_ratio"],
                "operator_angular_floor": x["angular_floor"],
                "operator_radial_penalty": x["radial_penalty"],
                "w_target": w_target,
                "w_error": w_error,
                "E_W": w_error / w_target,
                "weight_cosine": w["cosine"],
                "weight_norm_ratio": w["norm_ratio"],
                "weight_angular_floor": w["angular_floor"],
                "weight_radial_penalty": w["radial_penalty"],
                "gain_min": float(selected["gain"].min()),
                "gain_max": float(selected["gain"].max()),
            }
        )
    result = pd.DataFrame(output)
    if not np.isfinite(result.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all():
        raise RuntimeError("aggregate metrics contain nonfinite values")
    return result


def indexed_cell(
    frame: pd.DataFrame,
    *,
    tiling_seed: int,
    encoder_condition: str,
    decoder_condition: str,
    code_condition: str,
    calibration: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    selected = frame[
        (frame["tiling_seed"] == tiling_seed)
        & (frame["encoder_condition"] == encoder_condition)
        & (frame["decoder_condition"] == decoder_condition)
        & (frame["code_condition"] == code_condition)
        & (frame["calibration"] == calibration)
    ]
    result = {
        (int(row["depth"]), str(row["role"])): row
        for row in selected.to_dict("records")
    }
    if len(result) != 72:
        raise RuntimeError(
            f"incomplete cell {tiling_seed}/{encoder_condition}/{decoder_condition}/"
            f"{code_condition}/{calibration}: {len(result)}"
        )
    return result


def bootstrap_error_scopes(
    indexed: dict[tuple[int, str], dict[str, Any]], sampled_depths: np.ndarray
) -> dict[str, np.ndarray]:
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    for role in ROLES:
        errors = np.asarray(
            [float(indexed[(depth, role)]["x_error_scaled"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        targets = np.asarray(
            [float(indexed[(depth, role)]["x_target"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        numerators.append(errors)
        denominators.append(targets)
    numerator = np.stack(numerators, axis=1)
    denominator = np.stack(denominators, axis=1)
    values = numerator / denominator
    output = {role: values[:, index] for index, role in enumerate(ROLES)}
    output["macro"] = values.mean(axis=1)
    output["micro"] = numerator.sum(axis=1) / denominator.sum(axis=1)
    return output


def bootstrap_cosine_scopes(
    indexed: dict[tuple[int, str], dict[str, Any]], sampled_depths: np.ndarray
) -> dict[str, np.ndarray]:
    role_values: list[np.ndarray] = []
    role_dots: list[np.ndarray] = []
    role_targets: list[np.ndarray] = []
    role_predictions: list[np.ndarray] = []
    for role in ROLES:
        dots = np.asarray(
            [float(indexed[(depth, role)]["x_dot"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        targets = np.asarray(
            [float(indexed[(depth, role)]["x_target"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        predictions = np.asarray(
            [float(indexed[(depth, role)]["x_pred"]) for depth in range(12)], dtype=np.float64
        )[sampled_depths].sum(axis=1)
        role_dots.append(dots)
        role_targets.append(targets)
        role_predictions.append(predictions)
        role_values.append(dots / np.sqrt(np.maximum(targets * predictions, 1e-300)))
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


def paired_bootstraps(calibrated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_depths = rng.integers(0, 12, size=(BOOTSTRAP_DRAWS, 12))
    exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
    effect_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    did_rows: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        for encoder_condition, decoder_condition in CONTEXTS:
            cell: dict[tuple[str, str], dict[tuple[int, str], dict[str, Any]]] = {}
            for calibration in CALIBRATIONS:
                for code in CODES:
                    cell[(calibration, code)] = indexed_cell(
                        calibrated,
                        tiling_seed=tiling_seed,
                        encoder_condition=encoder_condition,
                        decoder_condition=decoder_condition,
                        code_condition=code,
                        calibration=calibration,
                    )
            for comparator in COMPARATORS:
                for calibration in CALIBRATIONS:
                    correct = bootstrap_error_scopes(cell[(calibration, "correct")], sampled_depths)
                    control = bootstrap_error_scopes(cell[(calibration, comparator)], sampled_depths)
                    correct_point = bootstrap_error_scopes(cell[(calibration, "correct")], exact_depths)
                    control_point = bootstrap_error_scopes(cell[(calibration, comparator)], exact_depths)
                    for aggregation in AGGREGATIONS:
                        delta = correct[aggregation] - control[aggregation]
                        ratio = control[aggregation] / correct[aggregation]
                        point_correct = float(correct_point[aggregation][0])
                        point_control = float(control_point[aggregation][0])
                        point_ratio = point_control / point_correct
                        effect_rows.append(
                            {
                                "tiling_seed": tiling_seed,
                                "context": f"{encoder_condition}/{decoder_condition}",
                                "encoder_condition": encoder_condition,
                                "decoder_condition": decoder_condition,
                                "calibration": calibration,
                                "comparator": comparator,
                                "aggregation": aggregation,
                                "point_correct": point_correct,
                                "point_comparator": point_control,
                                "point_ratio_comparator_over_correct": point_ratio,
                                "ratio_l05": float(np.quantile(ratio, 0.05)),
                                "ratio_u95": float(np.quantile(ratio, 0.95)),
                                "point_delta_correct_minus_comparator": point_correct - point_control,
                                "delta_l05": float(np.quantile(delta, 0.05)),
                                "delta_u95": float(np.quantile(delta, 0.95)),
                                "passes_5pct_screen": bool(point_ratio >= 1.05 and np.quantile(ratio, 0.05) > 1.0),
                                "draws": BOOTSTRAP_DRAWS,
                                "bootstrap_seed": BOOTSTRAP_SEED,
                                "bootstrap_unit": "12_transformer_blocks_roles_carried_together",
                            }
                        )

                correct_cos = bootstrap_cosine_scopes(cell[("raw", "correct")], sampled_depths)
                control_cos = bootstrap_cosine_scopes(cell[("raw", comparator)], sampled_depths)
                correct_cos_point = bootstrap_cosine_scopes(cell[("raw", "correct")], exact_depths)
                control_cos_point = bootstrap_cosine_scopes(cell[("raw", comparator)], exact_depths)
                for aggregation in AGGREGATIONS:
                    delta = correct_cos[aggregation] - control_cos[aggregation]
                    cosine_rows.append(
                        {
                            "tiling_seed": tiling_seed,
                            "context": f"{encoder_condition}/{decoder_condition}",
                            "encoder_condition": encoder_condition,
                            "decoder_condition": decoder_condition,
                            "comparator": comparator,
                            "aggregation": aggregation,
                            "point_correct": float(correct_cos_point[aggregation][0]),
                            "point_comparator": float(control_cos_point[aggregation][0]),
                            "point_delta_correct_minus_comparator": float(
                                correct_cos_point[aggregation][0] - control_cos_point[aggregation][0]
                            ),
                            "delta_l05": float(np.quantile(delta, 0.05)),
                            "delta_u95": float(np.quantile(delta, 0.95)),
                            "passes_directional_screen": bool(np.quantile(delta, 0.05) > 0.0),
                            "draws": BOOTSTRAP_DRAWS,
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "bootstrap_unit": "12_transformer_blocks_roles_carried_together",
                        }
                    )

                raw_correct = bootstrap_error_scopes(cell[("raw", "correct")], sampled_depths)
                raw_control = bootstrap_error_scopes(cell[("raw", comparator)], sampled_depths)
                gain_correct = bootstrap_error_scopes(
                    cell[("source_role_matched", "correct")], sampled_depths
                )
                gain_control = bootstrap_error_scopes(
                    cell[("source_role_matched", comparator)], sampled_depths
                )
                raw_correct_point = bootstrap_error_scopes(cell[("raw", "correct")], exact_depths)
                raw_control_point = bootstrap_error_scopes(cell[("raw", comparator)], exact_depths)
                gain_correct_point = bootstrap_error_scopes(
                    cell[("source_role_matched", "correct")], exact_depths
                )
                gain_control_point = bootstrap_error_scopes(
                    cell[("source_role_matched", comparator)], exact_depths
                )
                for aggregation in AGGREGATIONS:
                    did = (
                        (gain_correct[aggregation] - gain_control[aggregation])
                        - (raw_correct[aggregation] - raw_control[aggregation])
                    )
                    point = float(
                        (gain_correct_point[aggregation][0] - gain_control_point[aggregation][0])
                        - (raw_correct_point[aggregation][0] - raw_control_point[aggregation][0])
                    )
                    did_rows.append(
                        {
                            "tiling_seed": tiling_seed,
                            "context": f"{encoder_condition}/{decoder_condition}",
                            "encoder_condition": encoder_condition,
                            "decoder_condition": decoder_condition,
                            "comparator": comparator,
                            "aggregation": aggregation,
                            "point_difference_in_differences": point,
                            "did_l05": float(np.quantile(did, 0.05)),
                            "did_u95": float(np.quantile(did, 0.95)),
                            "differential_rescue_u95_below_zero": bool(np.quantile(did, 0.95) < 0.0),
                            "draws": BOOTSTRAP_DRAWS,
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "bootstrap_unit": "12_transformer_blocks_roles_carried_together",
                        }
                    )
    return pd.DataFrame(effect_rows), pd.DataFrame(cosine_rows), pd.DataFrame(did_rows)


def summed_by_role(
    matrix: pd.DataFrame,
    *,
    tiling_seed: int,
    encoder_condition: str,
    decoder_condition: str,
    code_condition: str,
) -> pd.DataFrame:
    selected = matrix[
        (matrix["tiling_seed"] == tiling_seed)
        & (matrix["encoder_condition"] == encoder_condition)
        & (matrix["decoder_condition"] == decoder_condition)
        & (matrix["code_condition"] == code_condition)
    ]
    if len(selected) != 72:
        raise RuntimeError("role-sum source cell is incomplete")
    return selected.groupby("role", sort=False)[
        ["x_target", "x_pred", "x_dot", "w_target", "w_pred", "w_dot"]
    ].sum().loc[list(ROLES)]


def error_for_role_sums(role_sums: pd.DataFrame, gain_values: np.ndarray) -> tuple[float, float]:
    target = role_sums["x_target"].to_numpy(dtype=np.float64)
    pred = role_sums["x_pred"].to_numpy(dtype=np.float64)
    dot = role_sums["x_dot"].to_numpy(dtype=np.float64)
    error = target - 2 * gain_values * dot + gain_values * gain_values * pred
    return float(np.mean(error / target)), float(error.sum() / target.sum())


def role_permutation_controls(matrix: pd.DataFrame, gains: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_roles = tuple(ROLES)
    gain_values = tuple(gains[role] for role in source_roles)
    permutations = list(itertools.permutations(range(len(ROLES))))
    rows: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        for encoder_condition, decoder_condition in CONTEXTS:
            for code_condition in CODES:
                role_sums = summed_by_role(
                    matrix,
                    tiling_seed=tiling_seed,
                    encoder_condition=encoder_condition,
                    decoder_condition=decoder_condition,
                    code_condition=code_condition,
                )
                for permutation_id, permutation in enumerate(permutations):
                    assigned = np.asarray([gain_values[index] for index in permutation], dtype=np.float64)
                    macro, micro = error_for_role_sums(role_sums, assigned)
                    mapping = ";".join(
                        f"{target_role}<-{source_roles[source_index]}"
                        for target_role, source_index in zip(ROLES, permutation)
                    )
                    for aggregation, value in (("macro", macro), ("micro", micro)):
                        rows.append(
                            {
                                "tiling_seed": tiling_seed,
                                "context": f"{encoder_condition}/{decoder_condition}",
                                "encoder_condition": encoder_condition,
                                "decoder_condition": decoder_condition,
                                "code_condition": code_condition,
                                "aggregation": aggregation,
                                "permutation_id": permutation_id,
                                "is_identity": permutation == tuple(range(len(ROLES))),
                                "mapping": mapping,
                                "E_X": value,
                            }
                        )
    frame = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    group_columns = [
        "tiling_seed",
        "context",
        "encoder_condition",
        "decoder_condition",
        "code_condition",
        "aggregation",
    ]
    for key, selected in frame.groupby(group_columns, sort=True):
        if len(selected) != 720 or int(selected["is_identity"].sum()) != 1:
            raise RuntimeError(f"role permutation group malformed: {key}")
        identity = float(selected.loc[selected["is_identity"], "E_X"].iloc[0])
        values = selected["E_X"].to_numpy(dtype=np.float64)
        better = int(np.sum(values < identity - 1e-14))
        not_worse = int(np.sum(values <= identity + 1e-14))
        summary.append(
            {
                **dict(zip(group_columns, key)),
                "identity_E_X": identity,
                "identity_rank_1_best": better + 1,
                "identity_exact_lower_tail_p": not_worse / 720.0,
                "best_E_X": float(np.min(values)),
                "median_E_X": float(np.median(values)),
                "worst_E_X": float(np.max(values)),
                "identity_top_5pct": bool((better + 1) <= 36),
                "permutations": 720,
            }
        )
    return frame, pd.DataFrame(summary)


def common_and_oracle_controls(matrix: pd.DataFrame, gains: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_rows: list[dict[str, Any]] = []
    oracle_role_rows: list[dict[str, Any]] = []
    matched_values = np.asarray([gains[role] for role in ROLES], dtype=np.float64)
    for tiling_seed in TILING_SEEDS:
        for encoder_condition, decoder_condition in CONTEXTS:
            role_sums_by_code: dict[str, pd.DataFrame] = {}
            for code_condition in CODES:
                role_sums = summed_by_role(
                    matrix,
                    tiling_seed=tiling_seed,
                    encoder_condition=encoder_condition,
                    decoder_condition=decoder_condition,
                    code_condition=code_condition,
                )
                role_sums_by_code[code_condition] = role_sums
                target = role_sums["x_target"].to_numpy(dtype=np.float64)
                pred = role_sums["x_pred"].to_numpy(dtype=np.float64)
                dot = role_sums["x_dot"].to_numpy(dtype=np.float64)
                common_candidates: list[tuple[str, str, float]] = [
                    ("raw", "source_fixed", 1.0),
                    ("source_median_common", "source_fixed", SOURCE_MEDIAN_GAIN),
                    ("source_mean_common", "source_fixed", SOURCE_MEAN_GAIN),
                ]
                macro_oracle = max(0.0, float(np.sum(dot / target) / np.sum(pred / target)))
                micro_oracle = max(0.0, float(np.sum(dot) / np.sum(pred)))
                for name, fit_scope, gain in common_candidates:
                    macro, micro = error_for_role_sums(role_sums, np.full(len(ROLES), gain))
                    for aggregation, value in (("macro", macro), ("micro", micro)):
                        common_rows.append(
                            {
                                "tiling_seed": tiling_seed,
                                "context": f"{encoder_condition}/{decoder_condition}",
                                "code_condition": code_condition,
                                "calibration": name,
                                "fit_scope": fit_scope,
                                "aggregation": aggregation,
                                "gain": gain,
                                "E_X": value,
                            }
                        )
                macro, _ = error_for_role_sums(role_sums, np.full(len(ROLES), macro_oracle))
                _, micro = error_for_role_sums(role_sums, np.full(len(ROLES), micro_oracle))
                common_rows.extend(
                    [
                        {
                            "tiling_seed": tiling_seed,
                            "context": f"{encoder_condition}/{decoder_condition}",
                            "code_condition": code_condition,
                            "calibration": "heldout_oracle_common",
                            "fit_scope": "heldout_fitted_descriptive_only",
                            "aggregation": "macro",
                            "gain": macro_oracle,
                            "E_X": macro,
                        },
                        {
                            "tiling_seed": tiling_seed,
                            "context": f"{encoder_condition}/{decoder_condition}",
                            "code_condition": code_condition,
                            "calibration": "heldout_oracle_common",
                            "fit_scope": "heldout_fitted_descriptive_only",
                            "aggregation": "micro",
                            "gain": micro_oracle,
                            "E_X": micro,
                        },
                    ]
                )
                oracle_gains = np.maximum(0.0, dot / pred)
                oracle_macro, oracle_micro = error_for_role_sums(role_sums, oracle_gains)
                oracle_role_rows.extend(
                    {
                        "tiling_seed": tiling_seed,
                        "context": f"{encoder_condition}/{decoder_condition}",
                        "code_condition": code_condition,
                        "aggregation": aggregation,
                        "fit_scope": "heldout_fitted_per_role_descriptive_only",
                        "E_X": value,
                        "gains_json": json.dumps(
                            {role: float(gain) for role, gain in zip(ROLES, oracle_gains)}, sort_keys=True
                        ),
                    }
                    for aggregation, value in (("macro", oracle_macro), ("micro", oracle_micro))
                )
            correct_sums = role_sums_by_code["correct"]
            correct_macro, correct_micro = error_for_role_sums(correct_sums, matched_values)
            for code_condition in COMPARATORS:
                selected_rows = [
                    row
                    for row in oracle_role_rows
                    if row["tiling_seed"] == tiling_seed
                    and row["context"] == f"{encoder_condition}/{decoder_condition}"
                    and row["code_condition"] == code_condition
                ]
                for row in selected_rows:
                    denominator = correct_macro if row["aggregation"] == "macro" else correct_micro
                    row["ratio_oracle_control_over_source_matched_correct"] = row["E_X"] / denominator
            for row in oracle_role_rows:
                if (
                    row["tiling_seed"] == tiling_seed
                    and row["context"] == f"{encoder_condition}/{decoder_condition}"
                    and row["code_condition"] == "correct"
                ):
                    denominator = correct_macro if row["aggregation"] == "macro" else correct_micro
                    row["ratio_oracle_control_over_source_matched_correct"] = row["E_X"] / denominator
    common = pd.DataFrame(common_rows)
    oracle_role = pd.DataFrame(oracle_role_rows)
    if oracle_role["ratio_oracle_control_over_source_matched_correct"].isna().any():
        raise RuntimeError("oracle role stress is missing ratios")
    return common, oracle_role


def gain_sensitivity(matrix: pd.DataFrame, gains: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    multipliers = np.linspace(0.0, 2.0, 81)
    for tiling_seed in TILING_SEEDS:
        for encoder_condition, decoder_condition in CONTEXTS:
            sums = summed_by_role(
                matrix,
                tiling_seed=tiling_seed,
                encoder_condition=encoder_condition,
                decoder_condition=decoder_condition,
                code_condition="correct",
            )
            for role in ROLES:
                target, pred, dot = [float(sums.loc[role, column]) for column in ("x_target", "x_pred", "x_dot")]
                oracle_gain = max(0.0, dot / pred)
                for multiplier in multipliers:
                    gain = float(multiplier * gains[role])
                    components = scaled_components(target, pred, dot, gain)
                    rows.append(
                        {
                            "tiling_seed": tiling_seed,
                            "context": f"{encoder_condition}/{decoder_condition}",
                            "role": role,
                            "gain_multiplier": float(multiplier),
                            "gain": gain,
                            "source_gain": gains[role],
                            "heldout_oracle_gain": oracle_gain,
                            "E_X": components["E"],
                            "angular_floor": components["angular_floor"],
                            "radial_penalty": components["radial_penalty"],
                        }
                    )
    return pd.DataFrame(rows)


def select_rows(frame: pd.DataFrame, **conditions: Any) -> pd.DataFrame:
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column] == value]
    return selected


def evaluate_screen(
    aggregates: pd.DataFrame,
    effects: pd.DataFrame,
    cosine: pd.DataFrame,
    did: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    criterion_rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        criterion_rows.append({"criterion": name, "pass": bool(passed), "detail": detail})

    raw_primary = select_rows(
        effects,
        context="cell/cell",
        calibration="raw",
    )
    raw_primary = raw_primary[
        raw_primary["aggregation"].isin(("macro", "micro"))
        & raw_primary["comparator"].isin(COMPARATORS)
    ]
    if len(raw_primary) != 8:
        raise RuntimeError("raw primary screen does not contain 8 rows")
    raw_pass_count = int(raw_primary["passes_5pct_screen"].sum())
    add(
        "locked_raw_verdict_remains_nonpass",
        raw_pass_count == 6,
        f"raw fixed-C pass count={raw_pass_count}/8; authority remains parent raw screen",
    )

    calibrated_primary = select_rows(
        effects,
        context="cell/cell",
        calibration="source_role_matched",
    )
    calibrated_primary = calibrated_primary[
        calibrated_primary["aggregation"].isin(("macro", "micro"))
        & calibrated_primary["comparator"].isin(COMPARATORS)
    ]
    calibrated_pass_count = int(calibrated_primary["passes_5pct_screen"].sum())
    add(
        "calibrated_fixed_context_8_of_8",
        len(calibrated_primary) == 8 and calibrated_pass_count == 8,
        f"source-role-gain pass count={calibrated_pass_count}/{len(calibrated_primary)}",
    )

    directional = select_rows(cosine, context="cell/cell")
    directional = directional[
        directional["aggregation"].isin(("macro", "micro"))
        & directional["comparator"].isin(COMPARATORS)
    ]
    directional_pass_count = int(directional["passes_directional_screen"].sum())
    add(
        "directional_correct_beats_permutation_and_zero_code",
        len(directional) == 8 and directional_pass_count == 8,
        f"operator-cosine L05>0 count={directional_pass_count}/{len(directional)}",
    )

    zero_did = select_rows(did, context="cell/cell", comparator="zero")
    zero_did = zero_did[zero_did["aggregation"].isin(("macro", "micro"))]
    did_pass_count = int(zero_did["differential_rescue_u95_below_zero"].sum())
    add(
        "zero_code_differential_rescue",
        len(zero_did) == 4 and did_pass_count == 4,
        f"difference-in-differences U95<0 count={did_pass_count}/{len(zero_did)}",
    )

    radial_checks: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        for role in ("attn_value", "attn_output", "ffn_down"):
            raw = select_rows(
                aggregates,
                tiling_seed=tiling_seed,
                encoder_condition="cell",
                decoder_condition="cell",
                code_condition="correct",
                calibration="raw",
                aggregation=role,
            )
            calibrated = select_rows(
                aggregates,
                tiling_seed=tiling_seed,
                encoder_condition="cell",
                decoder_condition="cell",
                code_condition="correct",
                calibration="source_role_matched",
                aggregation=role,
            )
            if len(raw) != 1 or len(calibrated) != 1:
                raise RuntimeError("radial screen cell is missing")
            raw_value = float(raw["operator_radial_penalty"].iloc[0])
            calibrated_value = float(calibrated["operator_radial_penalty"].iloc[0])
            radial_checks.append(
                {
                    "tiling_seed": tiling_seed,
                    "role": role,
                    "raw_radial_penalty": raw_value,
                    "calibrated_radial_penalty": calibrated_value,
                    "pass": calibrated_value < raw_value,
                }
            )
    add(
        "radial_penalty_reduced_in_raw_failing_roles",
        all(row["pass"] for row in radial_checks),
        json.dumps(radial_checks, sort_keys=True),
    )

    generic_checks: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        for aggregation in ("macro", "micro"):
            matched = select_rows(
                aggregates,
                tiling_seed=tiling_seed,
                encoder_condition="cell",
                decoder_condition="cell",
                code_condition="correct",
                calibration="source_role_matched",
                aggregation=aggregation,
            )
            generic = select_rows(
                aggregates,
                tiling_seed=tiling_seed,
                encoder_condition="cell",
                decoder_condition="cell",
                code_condition="correct",
                calibration="source_median_common",
                aggregation=aggregation,
            )
            if len(matched) != 1 or len(generic) != 1:
                raise RuntimeError("generic shrink screen cell is missing")
            generic_checks.append(
                {
                    "tiling_seed": tiling_seed,
                    "aggregation": aggregation,
                    "matched_E_X": float(matched["E_X"].iloc[0]),
                    "median_common_E_X": float(generic["E_X"].iloc[0]),
                    "pass": float(matched["E_X"].iloc[0]) < float(generic["E_X"].iloc[0]),
                }
            )
    add(
        "role_matched_beats_source_median_common",
        all(row["pass"] for row in generic_checks),
        json.dumps(generic_checks, sort_keys=True),
    )

    mapping = select_rows(permutation_summary, context="cell/cell", code_condition="correct")
    mapping = mapping[mapping["aggregation"].isin(("macro", "micro"))]
    mapping_pass_count = int(mapping["identity_top_5pct"].sum())
    add(
        "correct_role_mapping_top_5pct",
        len(mapping) == 4 and mapping_pass_count == 4,
        f"identity top-5% count={mapping_pass_count}/{len(mapping)}; ranks="
        + json.dumps(mapping[["tiling_seed", "aggregation", "identity_rank_1_best"]].to_dict("records")),
    )

    role_zero = select_rows(
        effects,
        context="cell/cell",
        calibration="source_role_matched",
        comparator="zero",
    )
    role_zero = role_zero[role_zero["aggregation"].isin(ROLES)]
    all_role_point_better = bool((role_zero["point_ratio_comparator_over_correct"] > 1.0).all())
    role_5pct_count = int(
        (role_zero["point_ratio_comparator_over_correct"] >= 1.05).sum()
    )
    add(
        "all_roles_point_better_than_zero_code",
        len(role_zero) == 12 and all_role_point_better,
        f"all point ratios>1={all_role_point_better}; >=1.05 count={role_5pct_count}/{len(role_zero)}",
    )

    criteria = pd.DataFrame(criterion_rows)
    mechanism_names = [
        "calibrated_fixed_context_8_of_8",
        "directional_correct_beats_permutation_and_zero_code",
        "zero_code_differential_rescue",
        "radial_penalty_reduced_in_raw_failing_roles",
        "role_matched_beats_source_median_common",
        "correct_role_mapping_top_5pct",
    ]
    mechanism = bool(criteria[criteria["criterion"].isin(mechanism_names)]["pass"].all())
    decision = {
        "analysis_class": "POST_HOC_SOURCE_ONLY_MECHANISM_DIAGNOSTIC",
        "locked_raw_primary_pass_count": raw_pass_count,
        "locked_raw_primary_total": 8,
        "locked_raw_status": "INCONCLUSIVE_FOR_5_PERCENT_EFFECT",
        "raw_verdict_changed": False,
        "calibrated_primary_pass_count": calibrated_pass_count,
        "calibrated_primary_total": 8,
        "full_scale_mechanism_screen_pass": mechanism,
        "interpretation": (
            "Latent operator signal plus material radial/scale miscalibration on this source panel"
            if mechanism
            else "Mechanisms remain mixed or the full scale-rescue discriminator did not pass"
        ),
        "scope_limits": [
            "Does not establish a standalone weight codec",
            "Does not establish universal role quality",
            "Does not establish cross-domain transfer",
            "Cannot retroactively pass the locked raw gate",
            "Gain-panel uncertainty is not included in bootstrap intervals",
        ],
        "target_access": False,
        "target_unseal_authorized": False,
        "next_action": (
            "Freeze and run independent prospective source replications before any confirmatory target access"
        ),
    }
    return decision, criteria


def plot_fixed_context_rescue(aggregates: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True, constrained_layout=True)
    colors = {"correct": "#2b6cb0", "permuted_within_row": "#dd6b20", "zero": "#718096"}
    labels = {"correct": "correct code", "permuted_within_row": "within-row permutation", "zero": "zero_code"}
    calibrations = ("raw", "source_role_matched")
    for axis, tiling_seed in zip(axes, TILING_SEEDS):
        selected = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["aggregation"] == "macro")
            & (aggregates["code_condition"].isin(CODES))
            & (aggregates["calibration"].isin(calibrations))
        ]
        x = np.arange(len(calibrations), dtype=float)
        width = 0.24
        for index, code in enumerate(CODES):
            values = [
                float(select_rows(selected, calibration=calibration, code_condition=code)["E_X"].iloc[0])
                for calibration in calibrations
            ]
            axis.bar(x + (index - 1) * width, values, width, color=colors[code], label=labels[code])
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="literal-zero output")
        axis.set_xticks(x, ["raw g=1", "source role gains"])
        axis.set_title(f"Tiling {tiling_seed}: fixed-context macro")
        axis.set_ylabel("operator error E_X")
        axis.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=4, frameon=False)
    fig.suptitle("Fixed-context operator error before and after source-frozen calibration", y=0.93, fontsize=14)
    fig.savefig(output / "fixed_context_error_rescue.png", dpi=180)
    plt.close(fig)


def plot_role_rescue(aggregates: pd.DataFrame, output: Path) -> None:
    short = {
        "attn_query": "Q",
        "attn_key": "K",
        "attn_value": "V",
        "attn_output": "O",
        "ffn_up": "FFN up",
        "ffn_down": "FFN down",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True, constrained_layout=True)
    x = np.arange(len(ROLES), dtype=float)
    width = 0.25
    for axis, tiling_seed in zip(axes, TILING_SEEDS):
        base = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["aggregation"].isin(ROLES))
        ]
        specifications = (
            ("correct", "raw", "correct raw", "#9ecae1"),
            ("correct", "source_role_matched", "correct + source gain", "#2b6cb0"),
            ("zero", "source_role_matched", "zero_code + same gain", "#718096"),
        )
        for index, (code, calibration, label, color) in enumerate(specifications):
            values = [
                float(
                    select_rows(base, code_condition=code, calibration=calibration, aggregation=role)["E_X"].iloc[0]
                )
                for role in ROLES
            ]
            axis.bar(x + (index - 1) * width, values, width, color=color, label=label)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="literal-zero output")
        axis.set_xticks(x, [short[role] for role in ROLES], rotation=18, ha="right")
        axis.set_title(f"Tiling {tiling_seed}")
        axis.set_ylabel("operator error E_X")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.suptitle("Role-wise fixed-context operator error (all roles shown)", y=0.93, fontsize=14)
    fig.savefig(output / "fixed_context_role_rescue.png", dpi=180)
    plt.close(fig)


def plot_radial_angular(aggregates: pd.DataFrame, output: Path) -> None:
    short = ["Q", "K", "V", "O", "Up", "Down"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True, constrained_layout=True)
    for axis, tiling_seed in zip(axes, TILING_SEEDS):
        selected = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["code_condition"] == "correct")
            & (aggregates["aggregation"].isin(ROLES))
            & (aggregates["calibration"].isin(("raw", "source_role_matched")))
        ]
        positions: list[float] = []
        labels: list[str] = []
        angular: list[float] = []
        radial: list[float] = []
        for role_index, role in enumerate(ROLES):
            for cal_index, calibration in enumerate(("raw", "source_role_matched")):
                row = select_rows(selected, aggregation=role, calibration=calibration).iloc[0]
                positions.append(role_index * 2.5 + cal_index)
                labels.append(f"{short[role_index]}\n{'raw' if cal_index == 0 else 'gain'}")
                angular.append(float(row["operator_angular_floor"]))
                radial.append(float(row["operator_radial_penalty"]))
        axis.bar(positions, angular, color="#4c78a8", label="angular floor 1-c²")
        axis.bar(positions, radial, bottom=angular, color="#f58518", label="radial penalty (gr-c)²")
        axis.set_xticks(positions, labels, fontsize=8)
        axis.set_title(f"Tiling {tiling_seed}")
        axis.set_ylabel("E_X decomposition")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.suptitle("Radial/angular decomposition before and after calibration", y=0.93, fontsize=14)
    fig.savefig(output / "radial_angular_decomposition.png", dpi=180)
    plt.close(fig)


def plot_role_permutations(permutations: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for row_index, tiling_seed in enumerate(TILING_SEEDS):
        for column_index, aggregation in enumerate(("macro", "micro")):
            axis = axes[row_index, column_index]
            selected = select_rows(
                permutations,
                tiling_seed=tiling_seed,
                context="cell/cell",
                code_condition="correct",
                aggregation=aggregation,
            )
            identity = float(selected.loc[selected["is_identity"], "E_X"].iloc[0])
            axis.hist(selected["E_X"], bins=35, color="#a0aec0", edgecolor="white")
            axis.axvline(identity, color="#c53030", linewidth=2.0, label=f"identity={identity:.3f}")
            axis.set_title(f"Tiling {tiling_seed} · {aggregation}")
            axis.set_xlabel("E_X across all 720 gain-to-role mappings")
            axis.set_ylabel("count")
            axis.legend(frameon=False)
            axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Exact role-gain assignment control under fixed context", fontsize=14)
    fig.savefig(output / "role_gain_permutation_control.png", dpi=180)
    plt.close(fig)


def plot_directional_cosine(aggregates: pd.DataFrame, output: Path) -> None:
    short = ["Q", "K", "V", "O", "Up", "Down"]
    colors = {"correct": "#2b6cb0", "permuted_within_row": "#dd6b20", "zero": "#718096"}
    labels = {"correct": "correct", "permuted_within_row": "within-row permutation", "zero": "zero_code"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True, constrained_layout=True)
    x = np.arange(len(ROLES), dtype=float)
    width = 0.25
    for axis, tiling_seed in zip(axes, TILING_SEEDS):
        selected = aggregates[
            (aggregates["tiling_seed"] == tiling_seed)
            & (aggregates["encoder_condition"] == "cell")
            & (aggregates["decoder_condition"] == "cell")
            & (aggregates["calibration"] == "raw")
            & (aggregates["aggregation"].isin(ROLES))
        ]
        for index, code in enumerate(CODES):
            values = [
                float(select_rows(selected, code_condition=code, aggregation=role)["operator_cosine"].iloc[0])
                for role in ROLES
            ]
            axis.bar(x + (index - 1) * width, values, width, color=colors[code], label=labels[code])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, short)
        axis.set_title(f"Tiling {tiling_seed}")
        axis.set_ylabel("operator cosine")
        axis.grid(axis="y", alpha=0.25)
    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="outside upper center", ncol=3, frameon=False)
    fig.suptitle("Operator cosine by code condition and role", y=0.93, fontsize=14)
    fig.savefig(output / "directional_operator_cosine.png", dpi=180)
    plt.close(fig)


def plot_gain_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    short = {
        "attn_query": "Q",
        "attn_key": "K",
        "attn_value": "V",
        "attn_output": "O",
        "ffn_up": "FFN up",
        "ffn_down": "FFN down",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    colors = {TILING_SEEDS[0]: "#2b6cb0", TILING_SEEDS[1]: "#dd6b20"}
    selected = sensitivity[sensitivity["context"] == "cell/cell"]
    for axis, role in zip(axes.flat, ROLES):
        role_rows = selected[selected["role"] == role]
        for tiling_seed in TILING_SEEDS:
            rows = role_rows[role_rows["tiling_seed"] == tiling_seed].sort_values("gain")
            axis.plot(rows["gain"], rows["E_X"], color=colors[tiling_seed], label=f"tiling {tiling_seed}")
            oracle = float(rows["heldout_oracle_gain"].iloc[0])
            axis.axvline(oracle, color=colors[tiling_seed], linestyle=":", alpha=0.75)
        source_gain = float(role_rows["source_gain"].iloc[0])
        axis.axvline(source_gain, color="black", linestyle="--", linewidth=1.2, label="source role gain")
        axis.axvline(1.0, color="#718096", linestyle="-.", linewidth=1.0, label="raw g=1")
        axis.set_title(short[role])
        axis.set_xlabel("gain g")
        axis.set_ylabel("operator error E_X")
        axis.grid(alpha=0.2)
    handles, labels_list = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="outside upper center", ncol=4, frameon=False)
    fig.suptitle(
        "Fixed-context gain sensitivity (dotted: heldout-fitted oracle, descriptive only)",
        y=0.96,
        fontsize=14,
    )
    fig.savefig(output / "gain_sensitivity_curves.png", dpi=180)
    plt.close(fig)


def artifact_manifest(output: Path, exclusions: Iterable[str] = ("artifact_manifest.json",)) -> list[dict[str, Any]]:
    excluded = set(exclusions)
    rows: list[dict[str, Any]] = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name in excluded:
            continue
        rows.append({"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def build_readme(
    output: Path,
    decision: dict[str, Any],
    effects: pd.DataFrame,
    cosine: pd.DataFrame,
    criteria: pd.DataFrame,
) -> None:
    primary = effects[
        (effects["context"] == "cell/cell")
        & (effects["calibration"] == "source_role_matched")
        & (effects["aggregation"].isin(("macro", "micro")))
    ].copy()
    direction = cosine[
        (cosine["context"] == "cell/cell")
        & (cosine["aggregation"].isin(("macro", "micro")))
    ].copy()
    lines = [
        "# Source role-gain mechanism diagnostic",
        "",
        f"Status: **{decision['analysis_class']}**.",
        "",
        "The locked raw gate remains 6/8 and non-passing. This analysis does not change it.",
        f"The full post-hoc scale-mechanism screen passed: **{decision['full_scale_mechanism_screen_pass']}**.",
        "Target-domain access remained false and is not authorized by this result.",
        "",
        "## Primary fixed-context calibrated comparisons",
        "",
        "| Tiling | Comparator | Aggregate | Correct E_X | Control E_X | Ratio | L05 | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in primary.sort_values(["tiling_seed", "comparator", "aggregation"]).itertuples(index=False):
        comparator_label = "zero_code" if row.comparator == "zero" else row.comparator
        lines.append(
            f"| {row.tiling_seed} | {comparator_label} | {row.aggregation} | {row.point_correct:.4f} | "
            f"{row.point_comparator:.4f} | {row.point_ratio_comparator_over_correct:.3f} | "
            f"{row.ratio_l05:.3f} | {row.passes_5pct_screen} |"
        )
    lines.extend(
        [
            "",
            "## Directional controls",
            "",
            "| Tiling | Comparator | Aggregate | Cosine delta | L05 | Pass |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for row in direction.sort_values(["tiling_seed", "comparator", "aggregation"]).itertuples(index=False):
        comparator_label = "zero_code" if row.comparator == "zero" else row.comparator
        lines.append(
            f"| {row.tiling_seed} | {comparator_label} | {row.aggregation} | "
            f"{row.point_delta_correct_minus_comparator:.3f} | {row.delta_l05:.3f} | "
            f"{row.passes_directional_screen} |"
        )
    lines.extend(["", "## Mechanism criteria", "", "| Criterion | Pass |", "|---|---|"])
    for row in criteria.to_dict("records"):
        lines.append(f"| {row['criterion']} | {row['pass']} |")
    lines.extend(
        [
            "",
            "## Narrow conclusion",
            "",
            decision["interpretation"] + ".",
            "The evidence does not establish cross-domain transfer or a high-fidelity standalone weight codec.",
            "",
            "## Key artifacts",
            "",
            "- `decision.json` and `mechanism_criteria.csv`",
            "- `paired_calibrated_effects.csv`, `paired_directional_effects.csv`, and `difference_in_differences.csv`",
            "- `aggregate_metrics.csv` and `calibrated_matrix_metrics.csv`",
            "- `role_gain_permutation_summary.csv` and `heldout_oracle_role_controls.csv`",
            "- publication-facing PNG diagnostics in this directory",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output)
    started = time.monotonic()
    logger.info(
        "startup label=source_role_gain_mechanism device=cpu dtype=float64 seed=%s draws=%s output=%s",
        BOOTSTRAP_SEED,
        BOOTSTRAP_DRAWS,
        output,
    )
    logger.info("scope=POST_HOC_SOURCE_ONLY target_access=false raw_verdict_mutable=false")

    logger.info("stage=provenance_and_seal_validation")
    provenance = {
        "input_hashes": validate_input_hashes(),
        "target_seal": validate_target_seal(),
        "gain_heldout_disjointness": validate_disjointness(),
    }
    gains = load_and_validate_gains()
    resolved = {
        "label": "source_role_gain_mechanism",
        "analysis_class": "post_hoc_source_only_mechanism_diagnostic",
        "device": "cpu",
        "dtype": "float64",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_unit": "12_transformer_blocks_roles_carried_together",
        "tiling_seeds": list(TILING_SEEDS),
        "primary_context": "cell/cell",
        "robustness_context": "native/native",
        "source_role_gains": gains,
        "source_median_common_gain": SOURCE_MEDIAN_GAIN,
        "source_mean_common_gain": SOURCE_MEAN_GAIN,
        "code_condition_labels": {
            "correct": "correct latent code",
            "permuted_within_row": "same-matrix within-row latent-code permutation",
            "zero": "zero_code = decode(z=0, C), a nonzero C-conditioned decoder prior",
        },
        "target_access": False,
        "raw_verdict_mutable": False,
        "output_dir": str(output),
    }
    write_json(output / "resolved_config.json", resolved)
    write_json(
        output / "literal_zero_output_reference.json",
        {
            "definition": "Y_hat=0 (distinct from zero_code=decode(z=0,C))",
            "E_X": 1.0,
            "E_W": 1.0,
            "operator_norm_ratio": 0.0,
            "weight_norm_ratio": 0.0,
            "operator_cosine": None,
            "weight_cosine": None,
        },
    )

    logger.info("stage=matrix_load_and_parity")
    matrix = pd.read_csv(INPUTS["factorial_matrix"][0])
    provenance["factorial_matrix_validity"] = validate_matrix(matrix)
    provenance["parent_correct_calibrated_parity"] = validate_parent_parity(matrix, gains)
    write_json(output / "provenance_audit.json", provenance)

    logger.info("stage=apply_frozen_calibrations rows=%s calibrations=%s", len(matrix), len(CALIBRATIONS))
    calibrated = build_calibrated_matrix(matrix, gains)
    aggregates = aggregate_calibrated(calibrated)
    write_csv(output / "calibrated_matrix_metrics.csv", calibrated)
    write_csv(output / "aggregate_metrics.csv", aggregates)

    logger.info("stage=paired_bootstrap draws=%s contexts=%s", BOOTSTRAP_DRAWS, len(CONTEXTS))
    effects, cosine, did = paired_bootstraps(calibrated)
    write_csv(output / "paired_calibrated_effects.csv", effects)
    write_csv(output / "paired_directional_effects.csv", cosine)
    write_csv(output / "difference_in_differences.csv", did)

    logger.info("stage=role_permutation_control permutations=720 cells=%s", len(TILING_SEEDS) * len(CONTEXTS) * len(CODES))
    permutations, permutation_summary = role_permutation_controls(matrix, gains)
    write_csv(output / "role_gain_permutations.csv", permutations)
    write_csv(output / "role_gain_permutation_summary.csv", permutation_summary)

    logger.info("stage=common_and_heldout_oracle_controls")
    common_controls, oracle_role = common_and_oracle_controls(matrix, gains)
    write_csv(output / "common_shrink_controls.csv", common_controls)
    write_csv(output / "heldout_oracle_role_controls.csv", oracle_role)

    logger.info("stage=gain_sensitivity curves=%s", len(TILING_SEEDS) * len(CONTEXTS) * len(ROLES))
    sensitivity = gain_sensitivity(matrix, gains)
    write_csv(output / "gain_sensitivity.csv", sensitivity)

    logger.info("stage=mechanism_screen raw_verdict_remains_authoritative")
    decision, criteria = evaluate_screen(aggregates, effects, cosine, did, permutation_summary)
    write_json(output / "decision.json", decision)
    write_csv(output / "mechanism_criteria.csv", criteria)

    logger.info("stage=plotting")
    plot_fixed_context_rescue(aggregates, output)
    plot_role_rescue(aggregates, output)
    plot_radial_angular(aggregates, output)
    plot_role_permutations(permutations, output)
    plot_directional_cosine(aggregates, output)
    plot_gain_sensitivity(sensitivity, output)
    build_readme(output, decision, effects, cosine, criteria)

    logger.info("stage=artifact_manifest")
    elapsed = time.monotonic() - started
    run_summary = {
        "status": "COMPLETE_POST_HOC_SOURCE_ONLY_MECHANISM_DIAGNOSTIC",
        "elapsed_seconds": elapsed,
        "matrix_rows": len(matrix),
        "calibrated_matrix_rows": len(calibrated),
        "aggregate_rows": len(aggregates),
        "paired_effect_rows": len(effects),
        "directional_rows": len(cosine),
        "difference_in_differences_rows": len(did),
        "role_permutation_rows": len(permutations),
        "full_scale_mechanism_screen_pass": decision["full_scale_mechanism_screen_pass"],
        "locked_raw_status": decision["locked_raw_status"],
        "target_access": False,
        "target_unseal_authorized": False,
    }
    write_json(output / "run_summary.json", run_summary)
    logger.info(
        "complete elapsed=%.2fs raw=%s calibrated=%s/%s full_mechanism=%s target_access=false",
        elapsed,
        decision["locked_raw_status"],
        decision["calibrated_primary_pass_count"],
        decision["calibrated_primary_total"],
        decision["full_scale_mechanism_screen_pass"],
    )
    logger.info("artifacts output=%s manifest=%s readme=%s", output, output / "artifact_manifest.json", output / "README.md")
    manifest = artifact_manifest(output)
    write_json(output / "artifact_manifest.json", {"artifacts": manifest, "count": len(manifest)})


if __name__ == "__main__":
    main()
