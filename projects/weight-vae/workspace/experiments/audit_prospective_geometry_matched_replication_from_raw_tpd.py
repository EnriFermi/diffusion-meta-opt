#!/usr/bin/env python3
"""Independent decision audit from the original sealed FP64 T/P/D tables.

This reviewer intentionally does not import the frozen formal analyzer.  It
reimplements the frozen aggregation, bootstrap, role-gain permutation, and
seven-condition decision rules and then compares its results to the formal CPU
analyzer output.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = (
    PROJECT
    / "artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_20260816"
)
FORMAL_DEFAULT = (
    PROJECT
    / "artifacts/crossmodal_united_structure/"
    "prospective_geometry_matched_replication_analysis_project_root_compatibility_repair_20260816"
)
OUTPUT_DEFAULT = (
    PROJECT
    / "artifacts/crossmodal_united_structure/"
    "prospective_geometry_matched_replication_independent_decision_audit_v3_20260816"
)

PANELS = ("beans", "trocr_sroie")
SPLITS = ("A", "B")
TILINGS = (26_081_801, 26_081_802)
DEPTHS = tuple(range(12))
ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
ARMS = ("correct", "permuted_within_row", "zero_code")
COMPARATORS = ("permuted_within_row", "zero_code")
AGGREGATIONS = ("macro", "micro")
CALIBRATIONS = ("raw", "source_role_gain", "source_median_common")
RADIAL_ROLES = ("attn_value", "attn_output", "ffn_down")
GAINS = {
    "attn_query": 0.5270182885626962,
    "attn_key": 0.5562907139098529,
    "attn_value": 0.3990051254514762,
    "attn_output": 0.3564476277035192,
    "ffn_up": 1.0280206811266708,
    "ffn_down": 0.3048084709039325,
}
COMMON_GAIN = 0.46301170700708616
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEEDS = {"beans": 26_081_903, "trocr_sroie": 26_081_902}
TIE_ATOL = 1e-12

EXPECTED_RAW_MANIFEST = "b30b8064ccd98e6d83b0e557e1267be5cd4abfc0d6e9b5430b2f4c1f5f95b80e"
EXPECTED_FORMAL_MANIFEST = "e92d532261dd126324fbcecfbe90713763f74611ef2338772113fa13a81729aa"
EXPECTED_WEIGHT_SHA = "9a20de3ecf920081e736a6c35d218abd4e7798a20853954b2c6ee54108a969e8"
EXPECTED_OPERATOR_SHA = "4cf673372ae2a58b146c21b1e451e3e1ef5704c1b826f9032e8a32c962bd977a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--formal-dir", type=Path, default=FORMAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "schema_version": "prospective_replication_independent_decision_audit_manifest_v1",
        "manifest_self_excluded": True,
        "count": len(rows),
        "artifacts": rows,
    }


def verify_manifest(root: Path, expected_sha: str) -> dict[str, Any]:
    path = root / "artifact_manifest.json"
    if path.is_symlink() or sha256_file(path) != expected_sha:
        raise RuntimeError(f"manifest identity mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["artifacts"]
    declared = set()
    for row in rows:
        relative = Path(str(row["path"]))
        member = root / relative
        if member.is_symlink() or not member.is_file():
            raise RuntimeError(f"manifest member type failure: {member}")
        if member.stat().st_size != int(row["bytes"]) or sha256_file(member) != row["sha256"]:
            raise RuntimeError(f"manifest member hash failure: {member}")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if declared != actual or len(declared) != len(rows):
        raise RuntimeError(f"manifest completeness failure: {root}")
    return {"path": str(path.resolve()), "sha256": expected_sha, "count": len(rows), "pass": True}


def gains(calibration: str, role_names: Sequence[str]) -> np.ndarray:
    if calibration == "raw":
        return np.ones(len(role_names), dtype=np.float64)
    if calibration == "source_role_gain":
        return np.asarray([GAINS[role] for role in role_names], dtype=np.float64)
    if calibration == "source_median_common":
        return np.full(len(role_names), COMMON_GAIN, dtype=np.float64)
    raise KeyError(calibration)


def metrics(
    target: np.ndarray | float,
    pred: np.ndarray | float,
    dot: np.ndarray | float,
    gain: np.ndarray | float,
) -> dict[str, np.ndarray]:
    target, pred, dot, gain = np.broadcast_arrays(
        np.asarray(target, dtype=np.float64),
        np.asarray(pred, dtype=np.float64),
        np.asarray(dot, dtype=np.float64),
        np.asarray(gain, dtype=np.float64),
    )
    pred_scaled = gain * gain * pred
    dot_scaled = gain * dot
    sse = np.maximum(target - 2.0 * dot_scaled + pred_scaled, 0.0)
    cosine = np.clip(dot_scaled / np.sqrt(target * pred_scaled), -1.0, 1.0)
    norm_ratio = np.sqrt(pred_scaled / target)
    angular_floor = 1.0 - cosine * cosine
    radial_penalty = (norm_ratio - cosine) ** 2
    error = sse / target
    if not np.allclose(error, angular_floor + radial_penalty, rtol=1e-10, atol=1e-12):
        raise RuntimeError("independent radial/angular identity failed")
    return {
        "T": target,
        "P_scaled": pred_scaled,
        "D_scaled": dot_scaled,
        "SSE": sse,
        "E": error,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "angular_floor": angular_floor,
        "radial_penalty": radial_penalty,
    }


METRIC_COLUMNS = (
    "target_energy",
    "scaled_prediction_energy",
    "scaled_target_prediction_dot",
    "sse",
    "E",
    "cosine",
    "norm_ratio",
    "angular_floor",
    "radial_penalty",
)


def metric_record(metric: Mapping[str, np.ndarray], index: int | None = None) -> dict[str, float]:
    mapping = {
        "target_energy": "T",
        "scaled_prediction_energy": "P_scaled",
        "scaled_target_prediction_dot": "D_scaled",
        "sse": "SSE",
        "E": "E",
        "cosine": "cosine",
        "norm_ratio": "norm_ratio",
        "angular_floor": "angular_floor",
        "radial_penalty": "radial_penalty",
    }
    result = {}
    for output_name, input_name in mapping.items():
        value = np.asarray(metric[input_name])
        result[output_name] = float(value.reshape(-1)[0] if index is None else value[index])
    return result


def build_aggregates(frame: pd.DataFrame, space: str) -> pd.DataFrame:
    group_columns = ["panel_id"]
    if space == "operator":
        group_columns.append("score_split")
    group_columns.extend(["tiling_seed", "arm"])
    rows: list[dict[str, Any]] = []
    for key, selected in frame.groupby(group_columns, sort=True):
        if len(selected) != 72:
            raise RuntimeError(f"incomplete aggregate group {space}/{key}")
        base = dict(zip(group_columns, key, strict=True))
        sums = selected.groupby("role")[["T", "P", "D"]].sum().loc[list(ROLES)]
        for calibration in CALIBRATIONS:
            gain = gains(calibration, ROLES)
            role_metrics = metrics(sums["T"], sums["P"], sums["D"], gain)
            for index, role in enumerate(ROLES):
                rows.append(
                    {
                        **base,
                        "space": space,
                        "calibration": calibration,
                        "aggregation": role,
                        "matrices": 12,
                        "gain_min": float(gain[index]),
                        "gain_max": float(gain[index]),
                        **metric_record(role_metrics, index),
                    }
                )
            macro = {
                **base,
                "space": space,
                "calibration": calibration,
                "aggregation": "macro",
                "matrices": 72,
                "gain_min": float(gain.min()),
                "gain_max": float(gain.max()),
                "target_energy": float(role_metrics["T"].sum()),
                "scaled_prediction_energy": float(role_metrics["P_scaled"].sum()),
                "scaled_target_prediction_dot": float(role_metrics["D_scaled"].sum()),
                "sse": float(role_metrics["SSE"].sum()),
            }
            for field in ("E", "cosine", "norm_ratio", "angular_floor", "radial_penalty"):
                macro[field] = float(np.mean(role_metrics[field]))
            rows.append(macro)
            micro_metrics = metrics(
                float(role_metrics["T"].sum()),
                float(role_metrics["P_scaled"].sum()),
                float(role_metrics["D_scaled"].sum()),
                1.0,
            )
            rows.append(
                {
                    **base,
                    "space": space,
                    "calibration": calibration,
                    "aggregation": "micro",
                    "matrices": 72,
                    "gain_min": float(gain.min()),
                    "gain_max": float(gain.max()),
                    **metric_record(micro_metrics),
                }
            )
    return pd.DataFrame(rows)


def cell_arrays(
    operator: pd.DataFrame,
    panel: str,
    split: str,
    tiling: int,
    arm: str,
) -> dict[str, np.ndarray]:
    selected = operator[
        (operator.panel_id == panel)
        & (operator.score_split == split)
        & (operator.tiling_seed == tiling)
        & (operator.arm == arm)
    ].set_index(["depth", "role"])
    if len(selected) != 72:
        raise RuntimeError(f"incomplete bootstrap source {panel}/{split}/{tiling}/{arm}")
    return {
        field: np.asarray(
            [[selected.loc[(depth, role), field] for role in ROLES] for depth in DEPTHS],
            dtype=np.float64,
        )
        for field in ("T", "P", "D")
    }


def draw_metrics(
    arrays: Mapping[str, np.ndarray],
    sampled_depths: np.ndarray,
    calibration: str,
) -> dict[str, dict[str, np.ndarray]]:
    target = arrays["T"][sampled_depths, :].sum(axis=1)
    pred = arrays["P"][sampled_depths, :].sum(axis=1)
    dot = arrays["D"][sampled_depths, :].sum(axis=1)
    role_metrics = metrics(target, pred, dot, gains(calibration, ROLES).reshape(1, 6))
    result = {
        field: {"macro": np.mean(role_metrics[field], axis=1)}
        for field in ("E", "cosine", "norm_ratio", "angular_floor", "radial_penalty")
    }
    micro_metrics = metrics(
        role_metrics["T"].sum(axis=1),
        role_metrics["P_scaled"].sum(axis=1),
        role_metrics["D_scaled"].sum(axis=1),
        1.0,
    )
    for field in result:
        result[field]["micro"] = micro_metrics[field]
    return result


def q(value: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), probability, method="linear"))


def build_bootstraps(
    operator: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    error_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    did_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    absolute_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"draws": BOOTSTRAP_DRAWS, "panels": {}}
    exact = np.arange(12, dtype=np.int64).reshape(1, 12)
    for panel in PANELS:
        rng = np.random.default_rng(BOOTSTRAP_SEEDS[panel])
        sampled = rng.integers(0, 12, size=(BOOTSTRAP_DRAWS, 12), dtype=np.int64)
        manifest["panels"][panel] = {
            "seed": BOOTSTRAP_SEEDS[panel],
            "index_shape": list(sampled.shape),
            "index_sha256": hashlib.sha256(
                np.ascontiguousarray(sampled, dtype="<i8").tobytes(order="C")
            ).hexdigest(),
        }
        print(f"stage=bootstrap panel={panel} draws={BOOTSTRAP_DRAWS}", flush=True)
        for split in SPLITS:
            for tiling in TILINGS:
                arrays = {
                    arm: cell_arrays(operator, panel, split, tiling, arm) for arm in ARMS
                }
                draws = {}
                points = {}
                for calibration in CALIBRATIONS:
                    for arm in ARMS:
                        draws[(calibration, arm)] = draw_metrics(
                            arrays[arm], sampled, calibration
                        )
                        points[(calibration, arm)] = draw_metrics(
                            arrays[arm], exact, calibration
                        )
                base = {
                    "panel_id": panel,
                    "score_split": split,
                    "tiling_seed": tiling,
                    "draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEEDS[panel],
                }
                for comparator in COMPARATORS:
                    for aggregation in AGGREGATIONS:
                        correct_e = draws[("source_role_gain", "correct")]["E"][aggregation]
                        comparator_e = draws[("source_role_gain", comparator)]["E"][aggregation]
                        point_correct = float(
                            points[("source_role_gain", "correct")]["E"][aggregation][0]
                        )
                        point_comparator = float(
                            points[("source_role_gain", comparator)]["E"][aggregation][0]
                        )
                        ratio = comparator_e / correct_e
                        point_ratio = point_comparator / point_correct
                        error_rows.append(
                            {
                                **base,
                                "comparator": comparator,
                                "aggregation": aggregation,
                                "calibration": "source_role_gain",
                                "point_correct_E_X": point_correct,
                                "point_comparator_E_X": point_comparator,
                                "point_ratio_comparator_over_correct": point_ratio,
                                "ratio_l05": q(ratio, 0.05),
                                "ratio_u95": q(ratio, 0.95),
                                "passes": bool(point_ratio >= 1.05 and q(ratio, 0.05) > 1.0),
                            }
                        )
                        correct_cos = draws[("source_role_gain", "correct")]["cosine"][aggregation]
                        comparator_cos = draws[("source_role_gain", comparator)]["cosine"][aggregation]
                        delta = correct_cos - comparator_cos
                        point_delta = float(
                            points[("source_role_gain", "correct")]["cosine"][aggregation][0]
                            - points[("source_role_gain", comparator)]["cosine"][aggregation][0]
                        )
                        cosine_rows.append(
                            {
                                **base,
                                "comparator": comparator,
                                "aggregation": aggregation,
                                "calibration": "source_role_gain",
                                "point_delta_correct_minus_comparator": point_delta,
                                "delta_l05": q(delta, 0.05),
                                "delta_u95": q(delta, 0.95),
                                "passes": bool(q(delta, 0.05) > 0.0),
                            }
                        )
                for aggregation in AGGREGATIONS:
                    gain_difference = (
                        draws[("source_role_gain", "correct")]["E"][aggregation]
                        - draws[("source_role_gain", "zero_code")]["E"][aggregation]
                    )
                    raw_difference = (
                        draws[("raw", "correct")]["E"][aggregation]
                        - draws[("raw", "zero_code")]["E"][aggregation]
                    )
                    did_draw = gain_difference - raw_difference
                    point_did = float(
                        (
                            points[("source_role_gain", "correct")]["E"][aggregation][0]
                            - points[("source_role_gain", "zero_code")]["E"][aggregation][0]
                        )
                        - (
                            points[("raw", "correct")]["E"][aggregation][0]
                            - points[("raw", "zero_code")]["E"][aggregation][0]
                        )
                    )
                    did_rows.append(
                        {
                            **base,
                            "aggregation": aggregation,
                            "point_difference_in_differences": point_did,
                            "did_l05": q(did_draw, 0.05),
                            "did_u95": q(did_draw, 0.95),
                            "passes": bool(q(did_draw, 0.95) < 0.0),
                        }
                    )
                    role_e = draws[("source_role_gain", "correct")]["E"][aggregation]
                    common_e = draws[("source_median_common", "correct")]["E"][aggregation]
                    role_minus_common = role_e - common_e
                    point_role = float(
                        points[("source_role_gain", "correct")]["E"][aggregation][0]
                    )
                    point_common = float(
                        points[("source_median_common", "correct")]["E"][aggregation][0]
                    )
                    common_rows.append(
                        {
                            **base,
                            "aggregation": aggregation,
                            "point_role_E_X": point_role,
                            "point_common_E_X": point_common,
                            "point_role_minus_common": point_role - point_common,
                            "delta_l05": q(role_minus_common, 0.05),
                            "delta_u95": q(role_minus_common, 0.95),
                            "passes": bool(
                                point_role - point_common < 0.0
                                and q(role_minus_common, 0.95) < 0.0
                            ),
                        }
                    )
                    absolute_rows.append(
                        {
                            **base,
                            "aggregation": aggregation,
                            "point_E_X": point_role,
                            "E_X_l05": q(role_e, 0.05),
                            "E_X_u95": q(role_e, 0.95),
                            "literal_zero_E_X": 1.0,
                            "passes": bool(point_role < 1.0 and q(role_e, 0.95) < 1.0),
                        }
                    )
    return (
        pd.DataFrame(error_rows),
        pd.DataFrame(cosine_rows),
        pd.DataFrame(did_rows),
        pd.DataFrame(common_rows),
        pd.DataFrame(absolute_rows),
        manifest,
    )


def build_role_mappings(operator: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_gains = np.asarray([GAINS[role] for role in ROLES], dtype=np.float64)
    identity_tuple = tuple(range(6))
    permutations = list(itertools.permutations(range(6)))
    rows = []
    for panel in PANELS:
        print(f"stage=role_gain_permutations panel={panel} mappings=720", flush=True)
        for split in SPLITS:
            for tiling in TILINGS:
                selected = operator[
                    (operator.panel_id == panel)
                    & (operator.score_split == split)
                    & (operator.tiling_seed == tiling)
                    & (operator.arm == "correct")
                ]
                sums = selected.groupby("role")[["T", "P", "D"]].sum().loc[list(ROLES)]
                for mapping_id, permutation in enumerate(permutations):
                    assigned = source_gains[np.asarray(permutation, dtype=np.int64)]
                    metric = metrics(sums["T"], sums["P"], sums["D"], assigned)
                    values = {
                        "macro": float(np.mean(metric["E"])),
                        "micro": float(np.sum(metric["SSE"]) / np.sum(metric["T"])),
                    }
                    mapping = ";".join(
                        f"{role}<-{ROLES[source]}"
                        for role, source in zip(ROLES, permutation, strict=True)
                    )
                    for aggregation, value in values.items():
                        rows.append(
                            {
                                "panel_id": panel,
                                "score_split": split,
                                "tiling_seed": tiling,
                                "aggregation": aggregation,
                                "mapping_id": mapping_id,
                                "mapping": mapping,
                                "is_identity": permutation == identity_tuple,
                                "E_X": value,
                            }
                        )
    mappings = pd.DataFrame(rows)
    summaries = []
    keys = ["panel_id", "score_split", "tiling_seed", "aggregation"]
    for key, selected in mappings.groupby(keys, sort=True):
        identity = float(selected.loc[selected.is_identity, "E_X"].iloc[0])
        values = selected.E_X.to_numpy(dtype=np.float64)
        better = int(np.sum(values < identity - TIE_ATOL))
        tied = int(np.sum(np.abs(values - identity) <= TIE_ATOL))
        rank = better + tied
        summaries.append(
            {
                **dict(zip(keys, key, strict=True)),
                "identity_E_X": identity,
                "strictly_better_count": better,
                "identity_tie_count_atol_1e_12": tied,
                "identity_conservative_worst_rank": rank,
                "identity_top_5pct": bool(rank <= 36),
                "best_E_X": float(values.min()),
                "median_E_X": float(np.median(values)),
                "worst_E_X": float(values.max()),
                "permutations": 720,
                "tie_atol": TIE_ATOL,
            }
        )
    return mappings, pd.DataFrame(summaries)


def one(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {filters}, got {len(selected)}")
    return selected.iloc[0]


def build_criteria_and_decisions(
    operator_aggregates: pd.DataFrame,
    error: pd.DataFrame,
    cosine: pd.DataFrame,
    did: pd.DataFrame,
    common: pd.DataFrame,
    absolute: pd.DataFrame,
    mapping_summary: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows = [
        {"criterion": 1, "cell_type": "calibrated_error_ratio", **row}
        for row in error.to_dict("records")
    ]
    rows.extend(
        {"criterion": 2, "cell_type": "operator_cosine_effect", **row}
        for row in cosine.to_dict("records")
    )
    for panel in PANELS:
        for split in SPLITS:
            for tiling in TILINGS:
                for role in ROLES:
                    correct = one(
                        operator_aggregates,
                        panel_id=panel,
                        score_split=split,
                        tiling_seed=tiling,
                        arm="correct",
                        calibration="source_role_gain",
                        aggregation=role,
                    )
                    zero = one(
                        operator_aggregates,
                        panel_id=panel,
                        score_split=split,
                        tiling_seed=tiling,
                        arm="zero_code",
                        calibration="source_role_gain",
                        aggregation=role,
                    )
                    rows.append(
                        {
                            "criterion": 3,
                            "cell_type": "role_correct_below_zero_code_and_literal_zero",
                            "panel_id": panel,
                            "score_split": split,
                            "tiling_seed": tiling,
                            "aggregation": role,
                            "point_correct_E_X": float(correct.E),
                            "point_zero_code_E_X": float(zero.E),
                            "literal_zero_E_X": 1.0,
                            "passes": bool(correct.E < zero.E and correct.E < 1.0),
                        }
                    )
    rows.extend(
        {"criterion": 4, "cell_type": "calibration_difference_in_differences", **row}
        for row in did.to_dict("records")
    )
    for panel in PANELS:
        for split in SPLITS:
            for tiling in TILINGS:
                for role in RADIAL_ROLES:
                    calibrated = one(
                        operator_aggregates,
                        panel_id=panel,
                        score_split=split,
                        tiling_seed=tiling,
                        arm="correct",
                        calibration="source_role_gain",
                        aggregation=role,
                    )
                    raw = one(
                        operator_aggregates,
                        panel_id=panel,
                        score_split=split,
                        tiling_seed=tiling,
                        arm="correct",
                        calibration="raw",
                        aggregation=role,
                    )
                    rows.append(
                        {
                            "criterion": 5,
                            "cell_type": "role_radial_penalty_reduction",
                            "panel_id": panel,
                            "score_split": split,
                            "tiling_seed": tiling,
                            "aggregation": role,
                            "raw_radial_penalty": float(raw.radial_penalty),
                            "calibrated_radial_penalty": float(calibrated.radial_penalty),
                            "point_delta_calibrated_minus_raw": float(
                                calibrated.radial_penalty - raw.radial_penalty
                            ),
                            "passes": bool(calibrated.radial_penalty < raw.radial_penalty),
                        }
                    )
    rows.extend(
        {"criterion": 6, "cell_type": "source_role_gain_vs_common", **row}
        for row in common.to_dict("records")
    )
    rows.extend(
        {
            "criterion": 6,
            "cell_type": "identity_role_gain_mapping_rank",
            **row,
            "passes": bool(row["identity_top_5pct"]),
        }
        for row in mapping_summary.to_dict("records")
    )
    rows.extend(
        {"criterion": 7, "cell_type": "absolute_operator_quality", **row}
        for row in absolute.to_dict("records")
    )
    criteria = pd.DataFrame(rows)
    criteria["passes"] = criteria.passes.astype(bool)
    panel_metadata = {str(row["panel_id"]): row for row in metadata["panels"]}
    decisions = {}
    for panel in PANELS:
        selected = criteria[criteria.panel_id == panel]
        conditions = {
            str(index): bool(selected[selected.criterion == index].passes.all())
            for index in range(1, 8)
        }
        counts = {
            str(index): {
                "passed": int(selected[selected.criterion == index].passes.sum()),
                "total": int(len(selected[selected.criterion == index])),
            }
            for index in range(1, 8)
        }
        validity = all(
            bool(panel_metadata[panel][field])
            for field in ("provenance_pass", "resource_hash_pass", "quality_pass", "validity_pass")
        )
        full = validity and all(conditions.values())
        directional_error = error[
            (error.panel_id == panel) & (error.comparator == "permuted_within_row")
        ]
        directional_cosine = cosine[
            (cosine.panel_id == panel) & (cosine.comparator == "permuted_within_row")
        ]
        directional = bool(
            (directional_error.point_correct_E_X < directional_error.point_comparator_E_X).all()
            and (directional_cosine.delta_l05 > 0.0).all()
        )
        if not validity:
            outcome = "INVALID_PANEL"
        elif full:
            outcome = "FULL_REPLICATION_PASS"
        elif directional:
            outcome = "DIRECTIONAL_ONLY_OR_MIXED"
        else:
            outcome = "MECHANISM_FAIL"
        decisions[panel] = {
            "outcome": outcome,
            "validity_pass": validity,
            "conditions": conditions,
            "condition_cell_counts": counts,
            "full_replication_pass": full,
            "directional_screen_pass": directional,
            "outcome_precedence": [
                "INVALID_PANEL",
                "FULL_REPLICATION_PASS",
                "DIRECTIONAL_ONLY_OR_MIXED",
                "MECHANISM_FAIL",
            ],
        }
    experiment = {
        "panel_outcomes": {panel: decisions[panel]["outcome"] for panel in PANELS},
        "both_panels_full_replication_pass": all(
            decisions[panel]["outcome"] == "FULL_REPLICATION_PASS" for panel in PANELS
        ),
        "later_confirmatory_target_protocol_may_be_frozen": all(
            decisions[panel]["outcome"] == "FULL_REPLICATION_PASS" for panel in PANELS
        ),
        "target_data_unsealed": False,
        "scope": "two unseen ViT-style vision encoders; not cross-domain evidence",
    }
    return criteria, decisions, experiment


def compare_frames(
    independent: pd.DataFrame,
    formal: pd.DataFrame,
    keys: Sequence[str],
    label: str,
) -> dict[str, Any]:
    if len(independent) != len(formal):
        raise RuntimeError(f"{label} row mismatch {len(independent)} != {len(formal)}")
    independent = independent.sort_values(list(keys), kind="stable").reset_index(drop=True)
    formal = formal.sort_values(list(keys), kind="stable").reset_index(drop=True)
    for key in keys:
        left = independent[key].fillna("<NA>").astype(str)
        right = formal[key].fillna("<NA>").astype(str)
        if not left.equals(right):
            raise RuntimeError(f"{label} key mismatch: {key}")
    shared = sorted(set(independent.columns) & set(formal.columns) - set(keys))
    numeric_differences = {}
    numeric_relative_differences = {}
    numeric_tolerance_violations = {}
    exact_differences = {}
    for column in shared:
        left = independent[column]
        right = formal[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = left.to_numpy(dtype=np.float64)
            right_values = right.to_numpy(dtype=np.float64)
            both_nan = np.isnan(left_values) & np.isnan(right_values)
            mismatch_nan = np.isnan(left_values) ^ np.isnan(right_values)
            if mismatch_nan.any():
                numeric_differences[column] = math.inf
                numeric_relative_differences[column] = math.inf
                numeric_tolerance_violations[column] = int(mismatch_nan.sum())
            else:
                difference = np.abs(left_values - right_values)
                difference[both_nan] = 0.0
                numeric_differences[column] = float(np.max(difference, initial=0.0))
                scale = np.maximum.reduce(
                    [np.abs(left_values), np.abs(right_values), np.ones_like(left_values)]
                )
                scale[both_nan] = 1.0
                relative = difference / scale
                numeric_relative_differences[column] = float(
                    np.max(relative, initial=0.0)
                )
                numeric_tolerance_differences = difference > (1e-12 + 5e-15 * scale)
                numeric_tolerance_violations[column] = int(
                    numeric_tolerance_differences.sum()
                )
        else:
            equal = left.fillna("<NA>").astype(str) == right.fillna("<NA>").astype(str)
            exact_differences[column] = int((~equal).sum())
    maximum_numeric = max(numeric_differences.values(), default=0.0)
    maximum_relative = max(numeric_relative_differences.values(), default=0.0)
    exact_mismatches = sum(exact_differences.values())
    tolerance_violations = sum(numeric_tolerance_violations.values())
    passed = bool(tolerance_violations == 0 and exact_mismatches == 0)
    if not passed:
        raise RuntimeError(
            f"{label} comparison failed max_numeric={maximum_numeric} "
            f"max_relative={maximum_relative} numeric_violations={tolerance_violations} "
            f"exact={exact_mismatches}"
        )
    return {
        "label": label,
        "rows": len(independent),
        "keys": list(keys),
        "max_abs_numeric_difference": maximum_numeric,
        "max_scaled_numeric_difference": maximum_relative,
        "numeric_column_max_abs": numeric_differences,
        "numeric_column_max_scaled": numeric_relative_differences,
        "numeric_tolerance_violations": numeric_tolerance_violations,
        "exact_mismatches": exact_differences,
        "pass": passed,
    }


def condition_margin_summary(criteria: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        selected = criteria[criteria.panel_id == panel]
        c1 = selected[selected.criterion == 1]
        c2 = selected[selected.criterion == 2]
        c3 = selected[selected.criterion == 3]
        c4 = selected[selected.criterion == 4]
        c5 = selected[selected.criterion == 5]
        common = selected[selected.cell_type == "source_role_gain_vs_common"]
        mapping = selected[selected.cell_type == "identity_role_gain_mapping_rank"]
        c7 = selected[selected.criterion == 7]
        rows.extend(
            [
                {
                    "panel_id": panel,
                    "criterion": 1,
                    "minimum_point_ratio_minus_1p05": float(
                        (c1.point_ratio_comparator_over_correct - 1.05).min()
                    ),
                    "minimum_bootstrap_l05_minus_1": float((c1.ratio_l05 - 1.0).min()),
                    "failed_cells": int((~c1.passes).sum()),
                },
                {
                    "panel_id": panel,
                    "criterion": 2,
                    "minimum_cosine_delta_l05": float(c2.delta_l05.min()),
                    "failed_cells": int((~c2.passes).sum()),
                },
                {
                    "panel_id": panel,
                    "criterion": 3,
                    "minimum_min_control_literal_zero_minus_correct": float(
                        (
                            np.minimum(c3.point_zero_code_E_X, c3.literal_zero_E_X)
                            - c3.point_correct_E_X
                        ).min()
                    ),
                    "failed_cells": int((~c3.passes).sum()),
                },
                {
                    "panel_id": panel,
                    "criterion": 4,
                    "maximum_did_u95": float(c4.did_u95.max()),
                    "failed_cells": int((~c4.passes).sum()),
                },
                {
                    "panel_id": panel,
                    "criterion": 5,
                    "minimum_raw_minus_calibrated_radial": float(
                        (c5.raw_radial_penalty - c5.calibrated_radial_penalty).min()
                    ),
                    "failed_cells": int((~c5.passes).sum()),
                },
                {
                    "panel_id": panel,
                    "criterion": 6,
                    "minimum_common_minus_role_point_E": float(
                        (common.point_common_E_X - common.point_role_E_X).min()
                    ),
                    "maximum_role_minus_common_u95": float(common.delta_u95.max()),
                    "maximum_identity_rank": int(mapping.identity_conservative_worst_rank.max()),
                    "failed_cells": int((~selected[selected.criterion == 6].passes).sum()),
                },
                {
                    "panel_id": panel,
                    "criterion": 7,
                    "minimum_one_minus_point_E": float((1.0 - c7.point_E_X).min()),
                    "minimum_one_minus_u95_E": float((1.0 - c7.E_X_u95).min()),
                    "failed_cells": int((~c7.passes).sum()),
                },
            ]
        )
    return pd.DataFrame(rows)


def role_calibration_diagnostics(operator: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = operator[operator.arm == "correct"]
    for key, group in selected.groupby(
        ["panel_id", "score_split", "tiling_seed", "role"], sort=True
    ):
        panel, split, tiling, role = key
        target = float(group["T"].sum())
        pred = float(group["P"].sum())
        dot = float(group["D"].sum())
        raw = metrics(target, pred, dot, 1.0)
        source = metrics(target, pred, dot, GAINS[role])
        common = metrics(target, pred, dot, COMMON_GAIN)
        optimal_gain = dot / pred
        optimal = metrics(target, pred, dot, optimal_gain)
        rows.append(
            {
                "panel_id": panel,
                "score_split": split,
                "tiling_seed": tiling,
                "role": role,
                "T": target,
                "P": pred,
                "D": dot,
                "source_gain": GAINS[role],
                "common_gain": COMMON_GAIN,
                "operator_optimal_gain": optimal_gain,
                "raw_E": float(raw["E"]),
                "source_gain_E": float(source["E"]),
                "common_gain_E": float(common["E"]),
                "optimal_gain_E": float(optimal["E"]),
                "raw_radial_penalty": float(raw["radial_penalty"]),
                "source_gain_radial_penalty": float(source["radial_penalty"]),
                "common_gain_radial_penalty": float(common["radial_penalty"]),
                "cosine": float(raw["cosine"]),
            }
        )
    return pd.DataFrame(rows)


def depth_calibration_diagnostics(operator: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = operator[operator.arm == "correct"]
    for key, group in selected.groupby(
        ["panel_id", "score_split", "tiling_seed", "depth"], sort=True
    ):
        panel, split, tiling, depth = key
        sums = group.set_index("role").loc[list(ROLES)]
        source = metrics(
            sums["T"], sums["P"], sums["D"], gains("source_role_gain", ROLES)
        )
        common = metrics(
            sums["T"], sums["P"], sums["D"], gains("source_median_common", ROLES)
        )
        source_micro = float(source["SSE"].sum() / source["T"].sum())
        common_micro = float(common["SSE"].sum() / common["T"].sum())
        rows.extend(
            [
                {
                    "panel_id": panel,
                    "score_split": split,
                    "tiling_seed": tiling,
                    "depth": depth,
                    "aggregation": "macro",
                    "source_role_E": float(source["E"].mean()),
                    "common_E": float(common["E"].mean()),
                    "source_minus_common": float(source["E"].mean() - common["E"].mean()),
                },
                {
                    "panel_id": panel,
                    "score_split": split,
                    "tiling_seed": tiling,
                    "depth": depth,
                    "aggregation": "micro",
                    "source_role_E": source_micro,
                    "common_E": common_micro,
                    "source_minus_common": source_micro - common_micro,
                },
            ]
        )
    return pd.DataFrame(rows)


def compact_key_metrics(
    criteria: pd.DataFrame,
    operator_aggregates: pd.DataFrame,
    weight_aggregates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        for split in SPLITS:
            for tiling in TILINGS:
                for aggregation in AGGREGATIONS:
                    base = {
                        "panel_id": panel,
                        "score_split": split,
                        "tiling_seed": tiling,
                        "aggregation": aggregation,
                    }
                    values = {**base}
                    for arm in ARMS:
                        record = one(
                            operator_aggregates,
                            **base,
                            arm=arm,
                            calibration="source_role_gain",
                        )
                        values[f"{arm}_E_X"] = float(record.E)
                        values[f"{arm}_cosine_X"] = float(record.cosine)
                    rows.append(values)
    operator_summary = pd.DataFrame(rows)
    weight_rows = []
    for panel in PANELS:
        for tiling in TILINGS:
            for aggregation in AGGREGATIONS:
                values = {
                    "panel_id": panel,
                    "score_split": "weight_space",
                    "tiling_seed": tiling,
                    "aggregation": aggregation,
                }
                for arm in ARMS:
                    record = one(
                        weight_aggregates,
                        panel_id=panel,
                        tiling_seed=tiling,
                        aggregation=aggregation,
                        arm=arm,
                        calibration="source_role_gain",
                    )
                    values[f"{arm}_E_W"] = float(record.E)
                    values[f"{arm}_cosine_W"] = float(record.cosine)
                weight_rows.append(values)
    return pd.concat([operator_summary, pd.DataFrame(weight_rows)], ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    raw = args.raw_dir.resolve(strict=True)
    formal = args.formal_dir.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(
        "resolved_config "
        f"raw={raw} formal={formal} output={output} device=cpu dtype=FP64 "
        f"bootstrap_draws={BOOTSTRAP_DRAWS} seeds={BOOTSTRAP_SEEDS} verbose=true",
        flush=True,
    )

    print("stage=input_manifest_verification", flush=True)
    input_audit = {
        "raw": verify_manifest(raw, EXPECTED_RAW_MANIFEST),
        "formal": verify_manifest(formal, EXPECTED_FORMAL_MANIFEST),
    }
    weight_path = raw / "weight_sufficient_stats.csv"
    operator_path = raw / "operator_sufficient_stats.csv"
    if sha256_file(weight_path) != EXPECTED_WEIGHT_SHA or sha256_file(operator_path) != EXPECTED_OPERATOR_SHA:
        raise RuntimeError("original T/P/D table identity mismatch")
    weight = pd.read_csv(weight_path)
    operator = pd.read_csv(operator_path)
    for frame in (weight, operator):
        for column in ("T", "P", "D"):
            frame[column] = frame[column].astype(np.float64)
        frame["tiling_seed"] = frame.tiling_seed.astype(np.int64)
        frame["depth"] = frame.depth.astype(np.int64)
    if len(weight) != 864 or len(operator) != 1728:
        raise RuntimeError("raw T/P/D row-count mismatch")
    if not np.isfinite(weight[["T", "P", "D"]]).all().all() or not np.isfinite(
        operator[["T", "P", "D"]]
    ).all().all():
        raise RuntimeError("nonfinite raw T/P/D")

    print("stage=independent_aggregates", flush=True)
    weight_aggregates = build_aggregates(weight, "weight")
    operator_aggregates = build_aggregates(operator, "operator")
    write_csv(output / "independent_weight_aggregates.csv", weight_aggregates)
    write_csv(output / "independent_operator_aggregates.csv", operator_aggregates)

    print("stage=independent_bootstrap", flush=True)
    error, cosine, did, common, absolute, bootstrap_manifest = build_bootstraps(operator)
    write_csv(output / "independent_criterion1_error_bootstrap.csv", error)
    write_csv(output / "independent_criterion2_cosine_bootstrap.csv", cosine)
    write_csv(output / "independent_criterion4_did_bootstrap.csv", did)
    write_csv(output / "independent_criterion6_common_bootstrap.csv", common)
    write_csv(output / "independent_criterion7_absolute_bootstrap.csv", absolute)
    write_json(output / "independent_bootstrap_manifest.json", bootstrap_manifest)

    mappings, mapping_summary = build_role_mappings(operator)
    write_csv(output / "independent_role_gain_mappings.csv", mappings)
    write_csv(output / "independent_role_gain_mapping_summary.csv", mapping_summary)

    print("stage=independent_exact_decision", flush=True)
    metadata = json.loads((raw / "runner_metadata.json").read_text(encoding="utf-8"))
    criteria, decisions, experiment = build_criteria_and_decisions(
        operator_aggregates,
        error,
        cosine,
        did,
        common,
        absolute,
        mapping_summary,
        metadata,
    )
    write_csv(output / "independent_criterion_cells.csv", criteria)
    write_json(output / "independent_panel_decisions.json", decisions)
    write_json(output / "independent_experiment_decision.json", experiment)
    failures = criteria[~criteria.passes].copy()
    write_csv(output / "failed_criterion_cells.csv", failures)

    print("stage=diagnostic_tables", flush=True)
    margins = condition_margin_summary(criteria)
    role_diagnostics = role_calibration_diagnostics(operator)
    depth_diagnostics = depth_calibration_diagnostics(operator)
    key_metrics = compact_key_metrics(criteria, operator_aggregates, weight_aggregates)
    write_csv(output / "condition_margin_summary.csv", margins)
    write_csv(output / "role_calibration_diagnostics.csv", role_diagnostics)
    write_csv(output / "depth_calibration_diagnostics.csv", depth_diagnostics)
    write_csv(output / "key_metrics.csv", key_metrics)

    print("stage=formal_output_comparison", flush=True)
    comparisons = {}
    table_specs = {
        "weight_aggregates": (
            weight_aggregates,
            "weight_aggregates.csv",
            ["panel_id", "tiling_seed", "arm", "calibration", "aggregation"],
        ),
        "operator_aggregates": (
            operator_aggregates,
            "operator_aggregates.csv",
            ["panel_id", "score_split", "tiling_seed", "arm", "calibration", "aggregation"],
        ),
        "criterion1": (
            error,
            "criterion1_error_bootstrap.csv",
            ["panel_id", "score_split", "tiling_seed", "comparator", "aggregation"],
        ),
        "criterion2": (
            cosine,
            "criterion2_cosine_bootstrap.csv",
            ["panel_id", "score_split", "tiling_seed", "comparator", "aggregation"],
        ),
        "criterion4": (
            did,
            "criterion4_did_bootstrap.csv",
            ["panel_id", "score_split", "tiling_seed", "aggregation"],
        ),
        "criterion6_common": (
            common,
            "criterion6_common_bootstrap.csv",
            ["panel_id", "score_split", "tiling_seed", "aggregation"],
        ),
        "criterion7": (
            absolute,
            "criterion7_absolute_bootstrap.csv",
            ["panel_id", "score_split", "tiling_seed", "aggregation"],
        ),
        "mapping_summary": (
            mapping_summary,
            "role_gain_mapping_summary.csv",
            ["panel_id", "score_split", "tiling_seed", "aggregation"],
        ),
        "mappings_full": (
            mappings,
            "role_gain_mappings.csv",
            ["panel_id", "score_split", "tiling_seed", "aggregation", "mapping_id"],
        ),
    }
    for label, (independent, filename, keys) in table_specs.items():
        comparisons[label] = compare_frames(
            independent,
            pd.read_csv(formal / filename),
            keys,
            label,
        )
    formal_criteria = pd.read_csv(formal / "criterion_cells.csv")
    criterion_specs = {
        "criterion1_cells": "calibrated_error_ratio",
        "criterion2_cells": "operator_cosine_effect",
        "criterion3_cells": "role_correct_below_zero_code_and_literal_zero",
        "criterion4_cells": "calibration_difference_in_differences",
        "criterion5_cells": "role_radial_penalty_reduction",
        "criterion6_common_cells": "source_role_gain_vs_common",
        "criterion6_mapping_cells": "identity_role_gain_mapping_rank",
        "criterion7_cells": "absolute_operator_quality",
    }
    for label, cell_type in criterion_specs.items():
        left = criteria[criteria.cell_type == cell_type]
        right = formal_criteria[formal_criteria.cell_type == cell_type]
        keys = ["panel_id", "score_split", "tiling_seed", "aggregation"]
        if "comparator" in left.columns and left.comparator.notna().any():
            keys.append("comparator")
        comparisons[label] = compare_frames(left, right, keys, label)
    formal_decisions = json.loads((formal / "panel_decisions.json").read_text(encoding="utf-8"))
    formal_experiment = json.loads((formal / "experiment_decision.json").read_text(encoding="utf-8"))
    if decisions != formal_decisions or experiment != formal_experiment:
        raise RuntimeError("independent JSON decisions differ from formal analyzer")
    formal_bootstrap = json.loads((formal / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    for panel in PANELS:
        if bootstrap_manifest["panels"][panel] != formal_bootstrap["panels"][panel]:
            raise RuntimeError(f"bootstrap index identity mismatch: {panel}")

    diagnostic_summary = {
        "schema_version": "prospective_replication_independent_decision_audit_v1",
        "status": "PASS_EXACT_RECOMPUTATION",
        "scope": "original sealed raw T/P/D; independent formulas; formal outputs comparison",
        "raw_inputs": {
            "weight": {"path": str(weight_path), "sha256": EXPECTED_WEIGHT_SHA, "rows": len(weight)},
            "operator": {
                "path": str(operator_path),
                "sha256": EXPECTED_OPERATOR_SHA,
                "rows": len(operator),
            },
        },
        "input_audit": input_audit,
        "bootstrap_manifest": bootstrap_manifest,
        "comparisons": comparisons,
        "decisions_exact_match": True,
        "panel_decisions": decisions,
        "experiment_decision": experiment,
        "failed_cells": int(len(failures)),
        "failed_cells_by_panel_condition": {
            panel: {
                str(condition): int(
                    len(
                        failures[
                            (failures.panel_id == panel) & (failures.criterion == condition)
                        ]
                    )
                )
                for condition in range(1, 8)
            }
            for panel in PANELS
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "audit_summary.json", diagnostic_summary)
    (output / "README.md").write_text(
        "# Independent raw-T/P/D decision audit\n\n"
        "The seven-condition decision was reimplemented without importing the formal analyzer.\n\n"
        f"- Beans: `{decisions['beans']['outcome']}`; failed cells: "
        f"{len(failures[failures.panel_id == 'beans'])}.\n"
        f"- TrOCR/SROIE: `{decisions['trocr_sroie']['outcome']}`; failed cells: "
        f"{len(failures[failures.panel_id == 'trocr_sroie'])}.\n"
        "- All formal aggregate, bootstrap, mapping, criterion, and decision outputs match the "
        "independent recomputation within FP64 tolerance.\n"
        "- Diagnostic interpretation and visual plot audit are recorded separately by the reviewer.\n",
        encoding="utf-8",
    )
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    print(
        "stage=complete "
        f"elapsed={time.monotonic() - started:.2f}s failed_cells={len(failures)} "
        f"outcomes={experiment['panel_outcomes']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
