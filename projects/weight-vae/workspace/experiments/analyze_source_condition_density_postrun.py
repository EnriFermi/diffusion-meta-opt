from __future__ import annotations

"""CPU-only post-run review for the source condition-density experiment.

The analyzer intentionally accepts only the frozen source-only density output
and its frozen sparse source reference.  It performs no model forward and has
no target-domain input.  It refuses to create review artifacts until the dense
run has written its COMPLETE run manifest and every required result artifact.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch


DEFAULT_DENSE_DIR = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_condition_density_gate_20260816"
)
DEFAULT_SPARSE_DIR = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_confirmatory_gate_20260816_clean2"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_condition_density_postrun_review_20260816"
)

ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
ROLE_LABELS = {
    "attn_query": "Q",
    "attn_key": "K",
    "attn_value": "V",
    "attn_output": "O",
    "ffn_up": "FFN-up",
    "ffn_down": "FFN-down",
}
VARIANTS = ("d2_k1", "d2_k4", "d2_k8", "all_k1", "all_k4", "all_k8")
CONTROLS = ("ae_native_c", "ae_zero_c", "identity", "zero")
METHODS = VARIANTS + CONTROLS
TILING_SEEDS = (26081601, 26081602)

MATRIX_NUMERIC = (
    "tiling_seed",
    "depth",
    "canonical_depth",
    "num_tiles",
    "gain",
    "w_target",
    "w_pred",
    "w_dot",
    "w_error",
    "x_target",
    "x_pred",
    "x_dot",
    "x_error",
    "w_error_scaled",
    "x_error_scaled",
    "w_pred_scaled",
    "x_pred_scaled",
    "w_dot_scaled",
    "x_dot_scaled",
    "raw_E_W",
    "raw_E_X",
    "calibrated_E_W",
    "calibrated_E_X",
    "raw_weight_cosine",
    "raw_operator_cosine",
    "raw_weight_norm_ratio",
    "raw_operator_norm_ratio",
)
AGGREGATE_METRICS = (
    "raw_E_W",
    "raw_E_X",
    "calibrated_E_W",
    "calibrated_E_X",
    "raw_weight_cosine",
    "raw_operator_cosine",
    "raw_weight_norm_ratio",
    "raw_operator_norm_ratio",
)
BOOTSTRAP_METRICS = (
    "raw_E_W",
    "raw_E_X",
    "calibrated_E_W",
    "calibrated_E_X",
    "raw_weight_cosine",
    "raw_operator_cosine",
)
SUM_KEYS = (
    "w_target",
    "w_pred",
    "w_dot",
    "w_error",
    "x_target",
    "x_pred",
    "x_dot",
    "x_error",
    "w_error_scaled",
    "x_error_scaled",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="CPU-only post-run review of the frozen source condition-density gate"
    )
    value.add_argument("--dense-dir", type=Path, default=DEFAULT_DENSE_DIR)
    value.add_argument("--sparse-dir", type=Path, default=DEFAULT_SPARSE_DIR)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def number(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite {key}={row[key]!r} in row {row}")
    return value


def close_enough(left: float, right: float, *, atol: float = 1e-12, rtol: float = 1e-12) -> bool:
    return abs(left - right) <= atol + rtol * max(abs(left), abs(right))


def require_files(directory: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            "dense run is not complete; required files are absent: " + ", ".join(missing)
        )


def require_completed_run(dense: Path, sparse: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if dense.resolve().name != DEFAULT_DENSE_DIR.name:
        raise RuntimeError(f"unexpected dense source directory: {dense.resolve()}")
    if sparse.resolve().name != DEFAULT_SPARSE_DIR.name:
        raise RuntimeError(f"unexpected sparse source directory: {sparse.resolve()}")
    forbidden = ("data2vec", "wav2vec", "whisper", "llama")
    for path in (dense.resolve(), sparse.resolve()):
        if any(token in str(path).lower() for token in forbidden):
            raise RuntimeError(f"forbidden non-source path: {path}")
    require_files(
        dense,
        (
            "run_manifest.json",
            "density_decision.json",
            "execution_manifest.json",
            "resolved_config.json",
            "checkpoint_info.json",
            "target_seal_assertions.json",
            "condition_vector_selection_manifest.json",
            "common_family_eligibility.json",
            "sparse_template_reproduction.json",
            "sparse_prediction_reproduction.json",
            "source_condition_density_templates.pt",
            "condition_template_summary.csv",
            "condition_hierarchy_manifest.json",
            "condition_record_dataset_variance.csv",
            "source_gain_sufficient_stats.csv",
            "source_fit_role_gains.csv",
            "source_gain_validity.json",
            "tiling_coverage.csv",
            "matrix_metrics.csv",
            "aggregate_metrics.csv",
            "block_bootstrap_absolute.csv",
            "block_bootstrap_paired_contrasts.csv",
            "condition_density_factorial.png",
            "run.log",
        ),
    )
    require_files(
        sparse,
        (
            "run_manifest.json",
            "source_condition_templates.pt",
            "condition_template_summary.csv",
            "source_fit_role_gains.csv",
            "matrix_metrics.csv",
            "aggregate_metrics.csv",
            "block_bootstrap.csv",
        ),
    )
    run_manifest = load_json(dense / "run_manifest.json")
    decision = load_json(dense / "density_decision.json")
    if run_manifest.get("status") != "COMPLETE_SOURCE_CONDITION_DENSITY":
        raise RuntimeError(
            "dense run has not reached COMPLETE_SOURCE_CONDITION_DENSITY: "
            f"{run_manifest.get('status')!r}"
        )
    if run_manifest.get("target_access") is not False:
        raise RuntimeError("completed run does not assert target_access=false")
    if "REMAINS_SEALED" not in str(decision.get("target_action", "")):
        raise RuntimeError("completed decision does not preserve the target seal")
    execution = load_json(dense / "execution_manifest.json")
    if execution.get("target_access") is not False:
        raise RuntimeError("execution manifest target_access is not false")
    return run_manifest, decision


def assert_unique(rows: Sequence[dict[str, Any]], fields: Sequence[str], label: str) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        counts: dict[tuple[Any, ...], int] = defaultdict(int)
        for key in keys:
            counts[key] += 1
        duplicates = [key for key, count in counts.items() if count > 1][:10]
        raise RuntimeError(f"duplicate {label} keys for {fields}: {duplicates}")


def assert_finite(rows: Sequence[dict[str, Any]], fields: Sequence[str], label: str) -> None:
    for index, row in enumerate(rows):
        for field in fields:
            if field not in row or row[field] == "":
                raise RuntimeError(f"missing numeric field {label}[{index}].{field}")
            number(row, field)


def aggregate_from_matrix(matrix_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for tiling_seed in TILING_SEEDS:
        for method in METHODS:
            group = [
                row
                for row in matrix_rows
                if int(row["tiling_seed"]) == tiling_seed and row["method"] == method
            ]
            if len(group) != 72:
                raise RuntimeError(f"incomplete aggregate group: {tiling_seed}/{method}/{len(group)}")
            role_values: dict[str, dict[str, float]] = {}
            for role in ROLES:
                role_group = [row for row in group if row["role"] == role]
                sums = {key: sum(number(row, key) for row in role_group) for key in SUM_KEYS}
                values = {
                    "raw_E_W": sums["w_error"] / sums["w_target"],
                    "raw_E_X": sums["x_error"] / sums["x_target"],
                    "calibrated_E_W": sums["w_error_scaled"] / sums["w_target"],
                    "calibrated_E_X": sums["x_error_scaled"] / sums["x_target"],
                    "raw_weight_cosine": sums["w_dot"]
                    / math.sqrt(max(sums["w_target"] * sums["w_pred"], 1e-300)),
                    "raw_operator_cosine": sums["x_dot"]
                    / math.sqrt(max(sums["x_target"] * sums["x_pred"], 1e-300)),
                    "raw_weight_norm_ratio": math.sqrt(sums["w_pred"] / sums["w_target"]),
                    "raw_operator_norm_ratio": math.sqrt(sums["x_pred"] / sums["x_target"]),
                }
                role_values[role] = values
                output.append(
                    {"tiling_seed": tiling_seed, "method": method, "aggregation": role, **values}
                )
            macro = {
                metric: sum(role_values[role][metric] for role in ROLES) / len(ROLES)
                for metric in AGGREGATE_METRICS
            }
            output.append(
                {"tiling_seed": tiling_seed, "method": method, "aggregation": "macro", **macro}
            )
            sums = {key: sum(number(row, key) for row in group) for key in SUM_KEYS}
            micro = {
                "raw_E_W": sums["w_error"] / sums["w_target"],
                "raw_E_X": sums["x_error"] / sums["x_target"],
                "calibrated_E_W": sums["w_error_scaled"] / sums["w_target"],
                "calibrated_E_X": sums["x_error_scaled"] / sums["x_target"],
                "raw_weight_cosine": sums["w_dot"]
                / math.sqrt(max(sums["w_target"] * sums["w_pred"], 1e-300)),
                "raw_operator_cosine": sums["x_dot"]
                / math.sqrt(max(sums["x_target"] * sums["x_pred"], 1e-300)),
                "raw_weight_norm_ratio": math.sqrt(sums["w_pred"] / sums["w_target"]),
                "raw_operator_norm_ratio": math.sqrt(sums["x_pred"] / sums["x_target"]),
            }
            output.append(
                {"tiling_seed": tiling_seed, "method": method, "aggregation": "micro", **micro}
            )
    return output


def bootstrap_samples(
    group: Sequence[dict[str, Any]], draw_indices: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    by_depth_role = {(int(row["depth"]), row["role"]): row for row in group}
    if len(by_depth_role) != 72:
        raise RuntimeError("bootstrap matrix grid is not 12x6")
    keys = (
        "w_target",
        "w_pred",
        "w_dot",
        "w_error",
        "w_error_scaled",
        "x_target",
        "x_pred",
        "x_dot",
        "x_error",
        "x_error_scaled",
    )
    role_arrays: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for role in ROLES:
        for key in keys:
            values = np.asarray(
                [number(by_depth_role[(depth, role)], key) for depth in range(12)], dtype=np.float64
            )
            role_arrays[role][key] = values[draw_indices].sum(axis=1)
    result: dict[tuple[str, str], np.ndarray] = {}
    per_role: dict[str, dict[str, np.ndarray]] = {}
    for role in ROLES:
        values = role_arrays[role]
        per_role[role] = {
            "raw_E_W": values["w_error"] / values["w_target"],
            "raw_E_X": values["x_error"] / values["x_target"],
            "calibrated_E_W": values["w_error_scaled"] / values["w_target"],
            "calibrated_E_X": values["x_error_scaled"] / values["x_target"],
            "raw_weight_cosine": values["w_dot"]
            / np.sqrt(np.maximum(values["w_target"] * values["w_pred"], 1e-300)),
            "raw_operator_cosine": values["x_dot"]
            / np.sqrt(np.maximum(values["x_target"] * values["x_pred"], 1e-300)),
        }
    for metric in BOOTSTRAP_METRICS:
        result[("macro", metric)] = np.stack(
            [per_role[role][metric] for role in ROLES], axis=1
        ).mean(axis=1)
    sums = {key: sum(role_arrays[role][key] for role in ROLES) for key in keys}
    result[("micro", "raw_E_W")] = sums["w_error"] / sums["w_target"]
    result[("micro", "raw_E_X")] = sums["x_error"] / sums["x_target"]
    result[("micro", "calibrated_E_W")] = sums["w_error_scaled"] / sums["w_target"]
    result[("micro", "calibrated_E_X")] = sums["x_error_scaled"] / sums["x_target"]
    result[("micro", "raw_weight_cosine")] = sums["w_dot"] / np.sqrt(
        np.maximum(sums["w_target"] * sums["w_pred"], 1e-300)
    )
    result[("micro", "raw_operator_cosine")] = sums["x_dot"] / np.sqrt(
        np.maximum(sums["x_target"] * sums["x_pred"], 1e-300)
    )
    return result


def recompute_bootstrap(
    matrix_rows: Sequence[dict[str, Any]], draws: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, 12, size=(draws, 12))
    absolute: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    comparisons = (
        ("d2_k4_minus_d2_k1", "d2_k4", "d2_k1", "within_dataset_K1_to_K4_at_D2"),
        ("all_k4_minus_all_k1", "all_k4", "all_k1", "within_dataset_K1_to_K4_at_allD"),
        ("all_k1_minus_d2_k1", "all_k1", "d2_k1", "dataset_breadth_at_K1"),
        ("all_k4_minus_d2_k4", "all_k4", "d2_k4", "dataset_breadth_at_K4"),
        ("all_k8_minus_d2_k8", "all_k8", "d2_k8", "dataset_breadth_at_K8"),
        ("all_k4_minus_d2_k1", "all_k4", "d2_k1", "predeclared_dense_primary_vs_sparse"),
        (
            "all_k8_minus_d2_k1",
            "all_k8",
            "d2_k1",
            "predeclared_max_density_primary_vs_sparse",
        ),
        ("d2_k8_minus_d2_k4", "d2_k8", "d2_k4", "K4_to_K8_saturation_at_D2"),
        ("all_k8_minus_all_k4", "all_k8", "all_k4", "K4_to_K8_saturation_at_allD"),
    )
    interactions = (
        (
            "breadth_x_K1_to_K4_interaction",
            ("all_k4", "all_k1"),
            ("d2_k4", "d2_k1"),
            "(all_k4-all_k1) - (d2_k4-d2_k1)",
        ),
        (
            "breadth_x_K4_to_K8_interaction",
            ("all_k8", "all_k4"),
            ("d2_k8", "d2_k4"),
            "(all_k8-all_k4) - (d2_k8-d2_k4)",
        ),
    )
    for tiling_seed in TILING_SEEDS:
        samples: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        for method in METHODS:
            group = [
                row
                for row in matrix_rows
                if int(row["tiling_seed"]) == tiling_seed and row["method"] == method
            ]
            samples[method] = bootstrap_samples(group, draw_indices)
            for (aggregation, metric), values in samples[method].items():
                absolute.append(
                    {
                        "tiling_seed": tiling_seed,
                        "method": method,
                        "aggregation": aggregation,
                        "metric": metric,
                        "draws": draws,
                        "mean": float(values.mean()),
                        "l95": float(np.quantile(values, 0.05)),
                        "u95": float(np.quantile(values, 0.95)),
                    }
                )
        for contrast_name, candidate, reference, interpretation in comparisons:
            for key in samples[candidate]:
                delta = samples[candidate][key] - samples[reference][key]
                aggregation, metric = key
                lower_is_better = metric in {
                    "raw_E_W",
                    "raw_E_X",
                    "calibrated_E_W",
                    "calibrated_E_X",
                }
                contrasts.append(
                    {
                        "tiling_seed": tiling_seed,
                        "contrast": contrast_name,
                        "candidate": candidate,
                        "reference": reference,
                        "interpretation": interpretation,
                        "aggregation": aggregation,
                        "metric": metric,
                        "lower_is_better": lower_is_better,
                        "draws": draws,
                        "mean_delta": float(delta.mean()),
                        "l95_delta": float(np.quantile(delta, 0.05)),
                        "u95_delta": float(np.quantile(delta, 0.95)),
                        "probability_improves": float(
                            np.mean(delta < 0) if lower_is_better else np.mean(delta > 0)
                        ),
                    }
                )
        for contrast_name, (all_high, all_low), (d2_high, d2_low), formula in interactions:
            for key in samples[all_high]:
                delta = (
                    samples[all_high][key]
                    - samples[all_low][key]
                    - samples[d2_high][key]
                    + samples[d2_low][key]
                )
                aggregation, metric = key
                lower_is_better = metric in {
                    "raw_E_W",
                    "raw_E_X",
                    "calibrated_E_W",
                    "calibrated_E_X",
                }
                contrasts.append(
                    {
                        "tiling_seed": tiling_seed,
                        "contrast": contrast_name,
                        "candidate": "difference_in_differences",
                        "reference": "zero_interaction",
                        "interpretation": formula,
                        "aggregation": aggregation,
                        "metric": metric,
                        "lower_is_better": lower_is_better,
                        "draws": draws,
                        "mean_delta": float(delta.mean()),
                        "l95_delta": float(np.quantile(delta, 0.05)),
                        "u95_delta": float(np.quantile(delta, 0.95)),
                        "probability_improves": "",
                    }
                )
    return absolute, contrasts


def compare_rows(
    stored: Sequence[dict[str, Any]],
    recomputed: Sequence[dict[str, Any]],
    *,
    key_fields: Sequence[str],
    numeric_fields: Sequence[str],
    label: str,
) -> dict[str, Any]:
    stored_map = {tuple(str(row[field]) for field in key_fields): row for row in stored}
    recomputed_map = {tuple(str(row[field]) for field in key_fields): row for row in recomputed}
    if set(stored_map) != set(recomputed_map):
        raise RuntimeError(
            f"{label} key mismatch: stored_only={len(set(stored_map)-set(recomputed_map))} "
            f"recomputed_only={len(set(recomputed_map)-set(stored_map))}"
        )
    max_abs = 0.0
    max_rel = 0.0
    mismatches = 0
    for key in stored_map:
        for field in numeric_fields:
            left = number(stored_map[key], field)
            right = number(recomputed_map[key], field)
            delta = abs(left - right)
            max_abs = max(max_abs, delta)
            max_rel = max(max_rel, delta / max(abs(left), abs(right), 1e-300))
            if not close_enough(left, right):
                mismatches += 1
    return {
        "label": label,
        "rows": len(stored_map),
        "numeric_fields_per_row": len(numeric_fields),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mismatched_values_at_1e-12": mismatches,
        "exact_within_1e-12": mismatches == 0,
    }


def validate_cache(dense: Path) -> dict[str, Any]:
    selection = load_json(dense / "condition_vector_selection_manifest.json")
    if not isinstance(selection, list) or len(selection) != 12180:
        raise RuntimeError(f"condition selection has wrong size: {type(selection)}/{len(selection)}")
    expected_indices = [int(row["vector_index"]) for row in selection]
    if expected_indices != list(range(12180)):
        raise RuntimeError("condition selection vector indices are not exact 0..12179")
    cache_dir = dense / "condition_vector_cache"
    tensor_paths = sorted(cache_dir.glob("condition_vectors_*.pt"))
    meta_paths = sorted(cache_dir.glob("condition_vectors_*.json"))
    if len(tensor_paths) != 96 or len(meta_paths) != 96:
        raise RuntimeError(
            f"condition cache is incomplete: pt={len(tensor_paths)} json={len(meta_paths)} expected=96"
        )
    seen: list[int] = []
    records = 0
    for tensor_path, meta_path in zip(tensor_paths, meta_paths, strict=True):
        if tensor_path.stem != meta_path.stem:
            raise RuntimeError(f"cache shard pairing mismatch: {tensor_path}/{meta_path}")
        meta = load_json(meta_path)
        if sha256_file(tensor_path) != meta.get("tensor_file_sha256"):
            raise RuntimeError(f"cache shard file hash mismatch: {tensor_path}")
        payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
        c_var = payload["c_var"].to(torch.float32).contiguous()
        c_patch = payload["c_patch"].to(torch.float32).contiguous()
        indices = [int(value) for value in payload["vector_indices"].tolist()]
        if indices != [int(value) for value in meta["vector_indices"]]:
            raise RuntimeError(f"cache vector index metadata mismatch: {tensor_path}")
        if tuple(c_var.shape) != tuple(c_patch.shape) or tuple(c_var.shape)[1:] != (256,):
            raise RuntimeError(f"cache vector shape mismatch: {tensor_path}/{c_var.shape}/{c_patch.shape}")
        if not bool(torch.isfinite(c_var).all()) or not bool(torch.isfinite(c_patch).all()):
            raise RuntimeError(f"nonfinite cache vector: {tensor_path}")
        if tensor_sha256(c_var) != meta.get("c_var_sha256"):
            raise RuntimeError(f"cache c_var tensor hash mismatch: {tensor_path}")
        if tensor_sha256(c_patch) != meta.get("c_patch_sha256"):
            raise RuntimeError(f"cache c_patch tensor hash mismatch: {tensor_path}")
        seen.extend(indices)
        records += int(c_var.shape[0])
    if seen != list(range(12180)) or records != 12180:
        raise RuntimeError(f"cache coverage mismatch: records={records}, indices={len(seen)}")
    return {
        "selection_rows": len(selection),
        "cache_shards": len(tensor_paths),
        "cache_records": records,
        "vector_indices_exact_0_to_12179": True,
        "all_files_and_tensor_hashes_exact": True,
        "all_vectors_finite": True,
    }


def validate_tables(
    dense: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    tables = {
        "matrix": read_csv(dense / "matrix_metrics.csv"),
        "aggregate": read_csv(dense / "aggregate_metrics.csv"),
        "absolute": read_csv(dense / "block_bootstrap_absolute.csv"),
        "contrasts": read_csv(dense / "block_bootstrap_paired_contrasts.csv"),
        "gains": read_csv(dense / "source_fit_role_gains.csv"),
        "gain_stats": read_csv(dense / "source_gain_sufficient_stats.csv"),
        "templates": read_csv(dense / "condition_template_summary.csv"),
        "variance": read_csv(dense / "condition_record_dataset_variance.csv"),
        "tiling": read_csv(dense / "tiling_coverage.csv"),
    }
    matrix = tables["matrix"]
    assert_unique(matrix, ("tiling_seed", "method", "depth", "role"), "matrix")
    assert_finite(matrix, MATRIX_NUMERIC, "matrix")
    expected_matrix = {
        (str(seed), method, str(depth), role)
        for seed in TILING_SEEDS
        for method in METHODS
        for depth in range(12)
        for role in ROLES
    }
    actual_matrix = {
        (row["tiling_seed"], row["method"], row["depth"], row["role"]) for row in matrix
    }
    if actual_matrix != expected_matrix or len(matrix) != 1440:
        raise RuntimeError(
            f"matrix grid mismatch rows={len(matrix)} missing={len(expected_matrix-actual_matrix)} "
            f"extra={len(actual_matrix-expected_matrix)}"
        )
    for seed in TILING_SEEDS:
        for method in METHODS:
            tile_count = sum(
                int(row["num_tiles"])
                for row in matrix
                if int(row["tiling_seed"]) == seed and row["method"] == method
            )
            if tile_count != 20736:
                raise RuntimeError(f"matrix tile count mismatch: {seed}/{method}/{tile_count}")
    aggregate = tables["aggregate"]
    assert_unique(aggregate, ("tiling_seed", "method", "aggregation"), "aggregate")
    assert_finite(aggregate, ("tiling_seed",) + AGGREGATE_METRICS, "aggregate")
    expected_aggregate = {
        (str(seed), method, aggregation)
        for seed in TILING_SEEDS
        for method in METHODS
        for aggregation in ROLES + ("macro", "micro")
    }
    actual_aggregate = {
        (row["tiling_seed"], row["method"], row["aggregation"]) for row in aggregate
    }
    if actual_aggregate != expected_aggregate or len(aggregate) != 160:
        raise RuntimeError("aggregate grid is not exact 2x10x8")
    absolute = tables["absolute"]
    assert_unique(absolute, ("tiling_seed", "method", "aggregation", "metric"), "bootstrap absolute")
    assert_finite(absolute, ("tiling_seed", "draws", "mean", "l95", "u95"), "bootstrap absolute")
    expected_absolute = {
        (str(seed), method, aggregation, metric)
        for seed in TILING_SEEDS
        for method in METHODS
        for aggregation in ("macro", "micro")
        for metric in BOOTSTRAP_METRICS
    }
    actual_absolute = {
        (row["tiling_seed"], row["method"], row["aggregation"], row["metric"])
        for row in absolute
    }
    if actual_absolute != expected_absolute or len(absolute) != 240:
        raise RuntimeError("absolute bootstrap grid is not exact 2x10x2x6")
    contrasts = tables["contrasts"]
    assert_unique(contrasts, ("tiling_seed", "contrast", "aggregation", "metric"), "paired contrasts")
    assert_finite(
        contrasts,
        ("tiling_seed", "draws", "mean_delta", "l95_delta", "u95_delta"),
        "paired contrasts",
    )
    if len(contrasts) != 264:
        raise RuntimeError(f"paired contrast row count is not 264: {len(contrasts)}")
    gains = tables["gains"]
    assert_unique(gains, ("method", "role"), "gains")
    assert_finite(gains, ("gain", "fit_matrices"), "gains")
    if {(row["method"], row["role"]) for row in gains} != {
        (method, role) for method in VARIANTS for role in ROLES
    }:
        raise RuntimeError("gain grid is not exact 6x6")
    gain_stats = tables["gain_stats"]
    assert_unique(gain_stats, ("method", "source_key"), "gain sufficient stats")
    assert_finite(
        gain_stats,
        (
            "depth",
            "canonical_depth",
            "tiling_seed",
            "num_tiles",
            "w_target",
            "w_pred",
            "w_dot",
            "w_error",
            "x_target",
            "x_pred",
            "x_dot",
            "x_error",
        ),
        "gain sufficient stats",
    )
    if len(gain_stats) != 432:
        raise RuntimeError(f"gain sufficient-stat rows are not 6x72: {len(gain_stats)}")
    templates = tables["templates"]
    assert_unique(templates, ("variant", "role", "canonical_depth"), "template summary")
    assert_finite(templates, ("canonical_depth", "c_var_norm", "c_patch_norm"), "templates")
    if len(templates) != 432:
        raise RuntimeError(f"template summary is not 6x72: {len(templates)}")
    variance = tables["variance"]
    assert_unique(variance, ("source_key", "vector_kind"), "condition variance")
    assert_finite(
        variance,
        (
            "native_ordinal",
            "native_layers",
            "datasets",
            "records_total_up_to_8",
            "equal_dataset_within_record_mse",
            "between_dataset_mean_mse",
        ),
        "condition variance",
    )
    if len(variance) != 1896:
        raise RuntimeError(f"condition variance rows are not 948x2: {len(variance)}")
    tiling = tables["tiling"]
    assert_unique(tiling, ("tiling_seed", "source_key"), "tiling coverage")
    if len(tiling) != 144:
        raise RuntimeError(f"tiling coverage rows are not 2x72: {len(tiling)}")
    for seed in TILING_SEEDS:
        tile_count = sum(
            int(row["num_tiles"]) for row in tiling if int(row["tiling_seed"]) == seed
        )
        if tile_count != 20736:
            raise RuntimeError(f"tiling coverage tile count mismatch: {seed}/{tile_count}")

    recomputed_aggregate = aggregate_from_matrix(matrix)
    aggregate_check = compare_rows(
        aggregate,
        recomputed_aggregate,
        key_fields=("tiling_seed", "method", "aggregation"),
        numeric_fields=AGGREGATE_METRICS,
        label="aggregate_metrics_from_matrix_metrics",
    )
    config = load_json(dense / "resolved_config.json")
    draws = int(config["bootstrap_draws"])
    seed = int(config["seed"]) + 99
    recomputed_absolute, recomputed_contrasts = recompute_bootstrap(matrix, draws, seed)
    absolute_check = compare_rows(
        absolute,
        recomputed_absolute,
        key_fields=("tiling_seed", "method", "aggregation", "metric"),
        numeric_fields=("draws", "mean", "l95", "u95"),
        label="absolute_bootstrap_from_matrix_metrics",
    )
    contrast_check = compare_rows(
        contrasts,
        recomputed_contrasts,
        key_fields=("tiling_seed", "contrast", "aggregation", "metric"),
        numeric_fields=("draws", "mean_delta", "l95_delta", "u95_delta"),
        label="paired_bootstrap_from_matrix_metrics",
    )
    if not all(
        check["exact_within_1e-12"]
        for check in (aggregate_check, absolute_check, contrast_check)
    ):
        raise RuntimeError(
            f"stored aggregate/bootstrap arithmetic failed independent recomputation: "
            f"{aggregate_check}/{absolute_check}/{contrast_check}"
        )
    cache_check = validate_cache(dense)
    summary = {
        "row_counts": {key: len(value) for key, value in tables.items()},
        "matrix_grid_exact_2x10x12x6": True,
        "aggregate_grid_exact_2x10x8": True,
        "bootstrap_absolute_grid_exact_2x10x2x6": True,
        "paired_contrast_rows_exact": True,
        "gain_grid_exact_6x6": True,
        "gain_stat_grid_exact_6x72": True,
        "template_grid_exact_6x72": True,
        "condition_variance_grid_exact_948x2": True,
        "tiling_coverage_exact": True,
        "all_checked_values_finite": True,
        "independent_arithmetic": [aggregate_check, absolute_check, contrast_check],
        "cache": cache_check,
    }
    return tables, summary


def sparse_reproduction(
    dense: Path,
    sparse: Path,
    tables: dict[str, list[dict[str, str]]],
    output: Path,
) -> dict[str, Any]:
    old_matrix = [
        row for row in read_csv(sparse / "matrix_metrics.csv") if row["method"] == "ae_cell_mean_c0"
    ]
    new_matrix = [row for row in tables["matrix"] if row["method"] == "d2_k1"]
    old_map = {
        (row["tiling_seed"], row["depth"], row["role"]): row for row in old_matrix
    }
    new_map = {
        (row["tiling_seed"], row["depth"], row["role"]): row for row in new_matrix
    }
    if set(old_map) != set(new_map) or len(old_map) != 144:
        raise RuntimeError("old/new sparse matrix key sets differ")
    comparison_rows: list[dict[str, Any]] = []
    numeric_fields = tuple(field for field in MATRIX_NUMERIC if field not in {"tiling_seed"})
    for key in sorted(old_map, key=lambda value: (int(value[0]), int(value[1]), ROLES.index(value[2]))):
        old = old_map[key]
        new = new_map[key]
        differences = [abs(number(old, field) - number(new, field)) for field in numeric_fields]
        relative = [
            difference / max(abs(number(old, field)), abs(number(new, field)), 1e-300)
            for field, difference in zip(numeric_fields, differences, strict=True)
        ]
        comparison_rows.append(
            {
                "tiling_seed": key[0],
                "depth": key[1],
                "role": key[2],
                "prediction_hash_exact": old["prediction_sha256"] == new["prediction_sha256"],
                "old_prediction_sha256": old["prediction_sha256"],
                "new_prediction_sha256": new["prediction_sha256"],
                "max_numeric_abs_diff": max(differences),
                "max_numeric_rel_diff": max(relative),
                "raw_E_X_abs_diff": abs(number(old, "raw_E_X") - number(new, "raw_E_X")),
                "calibrated_E_X_abs_diff": abs(
                    number(old, "calibrated_E_X") - number(new, "calibrated_E_X")
                ),
            }
        )
    write_csv(output / "sparse_prediction_numeric_reproduction.csv", comparison_rows)

    old_gains = {
        row["role"]: row
        for row in read_csv(sparse / "source_fit_role_gains.csv")
        if row["method"] == "ae_cell_mean_c0"
    }
    new_gains = {
        row["role"]: row for row in tables["gains"] if row["method"] == "d2_k1"
    }
    gain_rows = [
        {
            "role": role,
            "old_gain": number(old_gains[role], "gain"),
            "new_gain": number(new_gains[role], "gain"),
            "abs_diff": abs(number(old_gains[role], "gain") - number(new_gains[role], "gain")),
        }
        for role in ROLES
    ]
    write_csv(output / "sparse_gain_reproduction.csv", gain_rows)

    old_aggregate = {
        (row["tiling_seed"], row["aggregation"]): row
        for row in read_csv(sparse / "aggregate_metrics.csv")
        if row["method"] == "ae_cell_mean_c0"
    }
    new_aggregate = {
        (row["tiling_seed"], row["aggregation"]): row
        for row in tables["aggregate"]
        if row["method"] == "d2_k1"
    }
    aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(old_aggregate):
        row: dict[str, Any] = {"tiling_seed": key[0], "aggregation": key[1]}
        for metric in AGGREGATE_METRICS:
            row[f"old_{metric}"] = number(old_aggregate[key], metric)
            row[f"new_{metric}"] = number(new_aggregate[key], metric)
            row[f"abs_diff_{metric}"] = abs(
                number(old_aggregate[key], metric) - number(new_aggregate[key], metric)
            )
        aggregate_rows.append(row)
    write_csv(output / "sparse_aggregate_reproduction.csv", aggregate_rows)

    old_bootstrap = [
        row
        for row in read_csv(sparse / "block_bootstrap.csv")
        if row["method"] == "ae_cell_mean_c0"
    ]
    old_boot_map: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in old_bootstrap:
        aggregation, endpoint = row["metric"].split("_", 1)
        metric = f"{row['calibration']}_{endpoint}"
        old_boot_map[(row["tiling_seed"], aggregation, metric)] = row
    new_boot_map = {
        (row["tiling_seed"], row["aggregation"], row["metric"]): row
        for row in tables["absolute"]
        if row["method"] == "d2_k1" and row["metric"] in {
            "raw_E_W",
            "raw_E_X",
            "calibrated_E_W",
            "calibrated_E_X",
        }
    }
    if set(old_boot_map) != set(new_boot_map):
        raise RuntimeError("old/new sparse bootstrap key sets differ")
    bootstrap_rows: list[dict[str, Any]] = []
    for key in sorted(old_boot_map):
        old = old_boot_map[key]
        new = new_boot_map[key]
        bootstrap_rows.append(
            {
                "tiling_seed": key[0],
                "aggregation": key[1],
                "metric": key[2],
                **{
                    f"old_{field}": number(old, field)
                    for field in ("draws", "mean", "l95", "u95")
                },
                **{
                    f"new_{field}": number(new, field)
                    for field in ("draws", "mean", "l95", "u95")
                },
                **{
                    f"abs_diff_{field}": abs(number(old, field) - number(new, field))
                    for field in ("draws", "mean", "l95", "u95")
                },
            }
        )
    write_csv(output / "sparse_bootstrap_reproduction.csv", bootstrap_rows)

    old_templates = torch.load(
        sparse / "source_condition_templates.pt", map_location="cpu", weights_only=False
    )["cell_mean"]
    new_templates = torch.load(
        dense / "source_condition_density_templates.pt", map_location="cpu", weights_only=False
    )["d2_k1"]
    template_exact = set(old_templates) == set(new_templates) and all(
        torch.equal(old_templates[key][kind], new_templates[key][kind])
        for key in old_templates
        for kind in ("c_var", "c_patch")
    )
    summary = {
        "prediction_rows": len(comparison_rows),
        "all_144_prediction_hashes_exact": all(
            bool(row["prediction_hash_exact"]) for row in comparison_rows
        ),
        "max_matrix_numeric_abs_diff": max(
            float(row["max_numeric_abs_diff"]) for row in comparison_rows
        ),
        "max_matrix_numeric_rel_diff": max(
            float(row["max_numeric_rel_diff"]) for row in comparison_rows
        ),
        "all_144_template_c_var_c_patch_tensors_bit_exact": template_exact,
        "max_gain_abs_diff": max(float(row["abs_diff"]) for row in gain_rows),
        "max_aggregate_abs_diff": max(
            float(value)
            for row in aggregate_rows
            for key, value in row.items()
            if key.startswith("abs_diff_")
        ),
        "max_bootstrap_abs_diff": max(
            float(value)
            for row in bootstrap_rows
            for key, value in row.items()
            if key.startswith("abs_diff_")
        ),
        "runner_sparse_template_lock": load_json(dense / "sparse_template_reproduction.json"),
        "runner_sparse_prediction_lock": load_json(dense / "sparse_prediction_reproduction.json"),
    }
    if not summary["all_144_prediction_hashes_exact"] or not template_exact:
        raise RuntimeError(f"sparse reproduction anchor failed: {summary}")
    if any(
        summary[key] > 1e-12
        for key in (
            "max_matrix_numeric_abs_diff",
            "max_gain_abs_diff",
            "max_aggregate_abs_diff",
            "max_bootstrap_abs_diff",
        )
    ):
        raise RuntimeError(f"sparse numeric reproduction failed at 1e-12: {summary}")
    write_json(output / "sparse_reproduction_summary.json", summary)
    return summary


def lookup(
    rows: Sequence[dict[str, Any]], **conditions: Any
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in conditions.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"lookup expected one row, got {len(matches)} for {conditions}")
    return matches[0]


def build_k_trajectories(
    tables: dict[str, list[dict[str, str]]], output: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in TILING_SEEDS:
        for aggregation in ("macro", "micro"):
            for scope in ("d2", "all"):
                for k in (1, 4, 8):
                    method = f"{scope}_k{k}"
                    aggregate = lookup(
                        tables["aggregate"],
                        tiling_seed=seed,
                        method=method,
                        aggregation=aggregation,
                    )
                    row: dict[str, Any] = {
                        "tiling_seed": seed,
                        "aggregation": aggregation,
                        "scope": scope,
                        "K": k,
                        "method": method,
                    }
                    for metric in ("raw_E_X", "calibrated_E_X"):
                        boot = lookup(
                            tables["absolute"],
                            tiling_seed=seed,
                            method=method,
                            aggregation=aggregation,
                            metric=metric,
                        )
                        row[metric] = number(aggregate, metric)
                        row[f"{metric}_bootstrap_l95"] = number(boot, "l95")
                        row[f"{metric}_bootstrap_u95"] = number(boot, "u95")
                    rows.append(row)
    write_csv(output / "k_trajectories.csv", rows)
    return rows


def build_role_depth_deltas(
    tables: dict[str, list[dict[str, str]]], output: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in TILING_SEEDS:
        for depth in range(12):
            for role in ROLES:
                sparse = lookup(
                    tables["matrix"],
                    tiling_seed=seed,
                    method="d2_k1",
                    depth=depth,
                    role=role,
                )
                dense = lookup(
                    tables["matrix"],
                    tiling_seed=seed,
                    method="all_k8",
                    depth=depth,
                    role=role,
                )
                row: dict[str, Any] = {
                    "tiling_seed": seed,
                    "depth": depth,
                    "role": role,
                }
                for metric in ("raw_E_X", "calibrated_E_X", "raw_operator_cosine"):
                    row[f"d2_k1_{metric}"] = number(sparse, metric)
                    row[f"all_k8_{metric}"] = number(dense, metric)
                    row[f"delta_all_k8_minus_d2_k1_{metric}"] = number(dense, metric) - number(
                        sparse, metric
                    )
                rows.append(row)
    write_csv(output / "role_depth_density_deltas.csv", rows)
    return rows


def template_drift(
    dense: Path, output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    templates = torch.load(
        dense / "source_condition_density_templates.pt", map_location="cpu", weights_only=False
    )
    if tuple(templates) != VARIANTS:
        raise RuntimeError(f"template variant order/content mismatch: {tuple(templates)}")
    comparisons = (
        ("d2_k1_to_d2_k4", "d2_k1", "d2_k4"),
        ("d2_k4_to_d2_k8", "d2_k4", "d2_k8"),
        ("all_k1_to_all_k4", "all_k1", "all_k4"),
        ("all_k4_to_all_k8", "all_k4", "all_k8"),
        ("d2_k1_to_all_k8", "d2_k1", "all_k8"),
        ("d2_k4_to_all_k8", "d2_k4", "all_k8"),
        ("d2_k8_to_all_k8", "d2_k8", "all_k8"),
        ("all_k1_to_all_k8", "all_k1", "all_k8"),
    )
    cells: list[dict[str, Any]] = []
    for comparison, start, end in comparisons:
        for role in ROLES:
            for depth in range(12):
                for kind in ("c_var", "c_patch", "joint"):
                    if kind == "joint":
                        left = torch.cat(
                            [templates[start][(role, depth)][part] for part in ("c_var", "c_patch")]
                        ).to(torch.float64)
                        right = torch.cat(
                            [templates[end][(role, depth)][part] for part in ("c_var", "c_patch")]
                        ).to(torch.float64)
                    else:
                        left = templates[start][(role, depth)][kind].to(torch.float64)
                        right = templates[end][(role, depth)][kind].to(torch.float64)
                    left_norm = float(left.norm())
                    right_norm = float(right.norm())
                    cosine = float(torch.dot(left, right) / max(left_norm * right_norm, 1e-300))
                    relative_l2 = float((right - left).norm()) / max(left_norm, 1e-300)
                    cells.append(
                        {
                            "comparison": comparison,
                            "start": start,
                            "end": end,
                            "role": role,
                            "canonical_depth": depth,
                            "vector_kind": kind,
                            "cosine": cosine,
                            "relative_l2_from_start": relative_l2,
                            "norm_ratio_end_over_start": right_norm / max(left_norm, 1e-300),
                        }
                    )
    summary: list[dict[str, Any]] = []
    for comparison, start, end in comparisons:
        for kind in ("c_var", "c_patch", "joint"):
            for role in ROLES + ("all",):
                group = [
                    row
                    for row in cells
                    if row["comparison"] == comparison
                    and row["vector_kind"] == kind
                    and (role == "all" or row["role"] == role)
                ]
                cosine = np.asarray([row["cosine"] for row in group], dtype=np.float64)
                relative = np.asarray(
                    [row["relative_l2_from_start"] for row in group], dtype=np.float64
                )
                summary.append(
                    {
                        "comparison": comparison,
                        "start": start,
                        "end": end,
                        "vector_kind": kind,
                        "aggregation": role,
                        "cells": len(group),
                        "cosine_mean": float(cosine.mean()),
                        "cosine_min": float(cosine.min()),
                        "cosine_p10": float(np.quantile(cosine, 0.10)),
                        "relative_l2_mean": float(relative.mean()),
                        "relative_l2_median": float(np.median(relative)),
                        "relative_l2_p90": float(np.quantile(relative, 0.90)),
                        "relative_l2_max": float(relative.max()),
                    }
                )
    write_csv(output / "template_drift_cells.csv", cells)
    write_csv(output / "template_drift_summary.csv", summary)
    return cells, summary


def interpolation_source_weights(rows: Sequence[dict[str, Any]], depth: int) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: int(row["native_ordinal"]))
    expected = list(range(int(ordered[0]["native_layers"])))
    if [int(row["native_ordinal"]) for row in ordered] != expected:
        raise RuntimeError(
            f"incomplete native depth support for {ordered[0]['model_name']}/{ordered[0]['role']}"
        )
    # Match the runner's torch.float32 linspace/tensor interpolation arithmetic.
    target = float(torch.linspace(0.0, 1.0, 12, dtype=torch.float32)[depth])
    native_u = np.asarray(
        [int(row["native_ordinal"]) / (int(row["native_layers"]) - 1) for row in ordered],
        dtype=np.float32,
    )
    right = int(np.searchsorted(native_u, target, side="left"))
    if right <= 0:
        return {ordered[0]["source_key"]: 1.0}
    if right >= len(ordered):
        return {ordered[-1]["source_key"]: 1.0}
    left = right - 1
    span = float(native_u[right] - native_u[left])
    alpha = 0.0 if span <= 0 else float((target - native_u[left]) / span)
    weights = {
        ordered[left]["source_key"]: 1.0 - alpha,
        ordered[right]["source_key"]: alpha,
    }
    return {key: value for key, value in weights.items() if value > 1e-15}


def effective_support(dense: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hierarchy = load_json(dense / "condition_hierarchy_manifest.json")
    eligibility = load_json(dense / "common_family_eligibility.json")
    by_variant_role_model: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], dict[str, Any]] = {}
    for row in hierarchy:
        by_variant_role_model[(row["variant"], row["role"], row["model_name"])].append(row)
        by_source[(row["variant"], row["source_key"])] = row
    cells: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for role in ROLES:
            families = list(eligibility["eligible_family_roles"][role])
            family_count = len(families)
            for depth in range(12):
                source_weights: dict[str, float] = defaultdict(float)
                for family in families:
                    family_weights = interpolation_source_weights(
                        by_variant_role_model[(variant, role, family)], depth
                    )
                    for source_key, weight in family_weights.items():
                        source_weights[source_key] += weight / family_count
                if not close_enough(sum(source_weights.values()), 1.0, atol=1e-10, rtol=1e-10):
                    raise RuntimeError(f"source weights do not sum to one: {variant}/{role}/{depth}")
                dataset_weights: list[float] = []
                record_weights: list[float] = []
                records = 0
                datasets = 0
                for source_key, source_weight in source_weights.items():
                    source = by_source[(variant, source_key)]
                    counts = [int(value) for value in source["records_per_dataset"]]
                    datasets += len(counts)
                    records += sum(counts)
                    for count in counts:
                        dataset_weight = source_weight / len(counts)
                        dataset_weights.append(dataset_weight)
                        record_weights.extend([dataset_weight / count] * count)
                if not close_enough(sum(record_weights), 1.0, atol=1e-10, rtol=1e-10):
                    raise RuntimeError(f"record weights do not sum to one: {variant}/{role}/{depth}")
                cells.append(
                    {
                        "variant": variant,
                        "role": role,
                        "canonical_depth": depth,
                        "family_count": family_count,
                        "contributing_source_units": len(source_weights),
                        "contributing_dataset_units": datasets,
                        "contributing_records": records,
                        "kish_effective_source_units": 1.0
                        / sum(weight * weight for weight in source_weights.values()),
                        "kish_effective_dataset_units": 1.0
                        / sum(weight * weight for weight in dataset_weights),
                        "kish_effective_records": 1.0
                        / sum(weight * weight for weight in record_weights),
                    }
                )
    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        group = [row for row in cells if row["variant"] == variant]
        hierarchy_group = [
            row
            for row in hierarchy
            if row["variant"] == variant
            and row["model_name"] in eligibility["eligible_family_roles"][row["role"]]
        ]
        summary.append(
            {
                "variant": variant,
                "eligible_source_units": len(hierarchy_group),
                "selected_dataset_units": sum(len(row["datasets"]) for row in hierarchy_group),
                "selected_records": sum(
                    sum(int(value) for value in row["records_per_dataset"])
                    for row in hierarchy_group
                ),
                "kish_effective_records_mean_over_72_cells": statistics.fmean(
                    float(row["kish_effective_records"]) for row in group
                ),
                "kish_effective_records_min_over_72_cells": min(
                    float(row["kish_effective_records"]) for row in group
                ),
                "kish_effective_records_max_over_72_cells": max(
                    float(row["kish_effective_records"]) for row in group
                ),
                "kish_effective_datasets_mean_over_72_cells": statistics.fmean(
                    float(row["kish_effective_dataset_units"]) for row in group
                ),
                "kish_effective_sources_mean_over_72_cells": statistics.fmean(
                    float(row["kish_effective_source_units"]) for row in group
                ),
            }
        )
    write_csv(output / "effective_support_cells.csv", cells)
    write_csv(output / "effective_support_summary.csv", summary)
    return cells, summary


def role_heterogeneity(
    tables: dict[str, list[dict[str, str]]], output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in TILING_SEEDS:
        for role in ROLES:
            sparse = lookup(
                tables["aggregate"], tiling_seed=seed, method="d2_k1", aggregation=role
            )
            dense = lookup(
                tables["aggregate"], tiling_seed=seed, method="all_k8", aggregation=role
            )
            row: dict[str, Any] = {"tiling_seed": seed, "role": role}
            for metric in ("raw_E_X", "calibrated_E_X", "raw_operator_cosine"):
                sparse_value = number(sparse, metric)
                dense_value = number(dense, metric)
                row[f"d2_k1_{metric}"] = sparse_value
                row[f"all_k8_{metric}"] = dense_value
                row[f"delta_all_k8_minus_d2_k1_{metric}"] = dense_value - sparse_value
            row["relative_raw_E_X_reduction"] = (
                number(sparse, "raw_E_X") - number(dense, "raw_E_X")
            ) / max(number(sparse, "raw_E_X"), 1e-300)
            rows.append(row)
    for seed in TILING_SEEDS:
        group = [row for row in rows if row["tiling_seed"] == seed]
        ranked = sorted(group, key=lambda row: row["delta_all_k8_minus_d2_k1_raw_E_X"])
        for rank, row in enumerate(ranked, start=1):
            row["raw_E_X_improvement_rank_1_best"] = rank
    summary: dict[str, Any] = {}
    for seed in TILING_SEEDS:
        seed_summary: dict[str, Any] = {}
        group = [row for row in rows if row["tiling_seed"] == seed]
        for metric in ("raw_E_X", "calibrated_E_X", "raw_operator_cosine"):
            key = f"delta_all_k8_minus_d2_k1_{metric}"
            values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            lower_better = metric != "raw_operator_cosine"
            seed_summary[metric] = {
                "mean_delta": float(values.mean()),
                "std_delta_across_roles": float(values.std(ddof=0)),
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
                "roles_improved": int(np.sum(values < 0) if lower_better else np.sum(values > 0)),
                "roles_total": len(ROLES),
            }
        summary[str(seed)] = seed_summary
    write_csv(output / "role_heterogeneity.csv", rows)
    write_json(output / "role_heterogeneity_summary.json", summary)
    return rows, summary


def effect_and_gate(
    tables: dict[str, list[dict[str, str]]], output: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in TILING_SEEDS:
        for aggregation in ("macro", "micro"):
            for metric in ("raw_E_X", "calibrated_E_X"):
                sparse = lookup(
                    tables["aggregate"],
                    tiling_seed=seed,
                    method="d2_k1",
                    aggregation=aggregation,
                )
                dense = lookup(
                    tables["aggregate"],
                    tiling_seed=seed,
                    method="all_k8",
                    aggregation=aggregation,
                )
                native = lookup(
                    tables["aggregate"],
                    tiling_seed=seed,
                    method="ae_native_c",
                    aggregation=aggregation,
                )
                zero = lookup(
                    tables["aggregate"],
                    tiling_seed=seed,
                    method="zero",
                    aggregation=aggregation,
                )
                dense_boot = lookup(
                    tables["absolute"],
                    tiling_seed=seed,
                    method="all_k8",
                    aggregation=aggregation,
                    metric=metric,
                )
                contrast = lookup(
                    tables["contrasts"],
                    tiling_seed=seed,
                    contrast="all_k8_minus_d2_k1",
                    aggregation=aggregation,
                    metric=metric,
                )
                sparse_value = number(sparse, metric)
                dense_value = number(dense, metric)
                zero_value = number(zero, metric)
                old_gap = sparse_value - zero_value
                rows.append(
                    {
                        "tiling_seed": seed,
                        "aggregation": aggregation,
                        "metric": metric,
                        "d2_k1_point": sparse_value,
                        "all_k8_point": dense_value,
                        "point_delta_all_k8_minus_d2_k1": dense_value - sparse_value,
                        "relative_error_reduction": (sparse_value - dense_value)
                        / max(sparse_value, 1e-300),
                        "zero_reconstruction_point": zero_value,
                        "all_k8_margin_below_zero_point": zero_value - dense_value,
                        "d2_k1_gap_above_zero_point": old_gap,
                        "fraction_sparse_to_zero_gap_closed": (
                            (sparse_value - dense_value) / old_gap if old_gap > 0 else ""
                        ),
                        "native_c_point": number(native, metric),
                        "all_k8_excess_over_native_c": dense_value - number(native, metric),
                        "all_k8_absolute_bootstrap_u95": number(dense_boot, "u95"),
                        "old_g1_operator_absolute_u95_threshold": 0.90,
                        "all_k8_absolute_u95_margin_below_old_g1_operator_gate": 0.90
                        - number(dense_boot, "u95"),
                        "all_k8_old_g1_operator_gate_pass": number(dense_boot, "u95") < 0.90,
                        "paired_delta_mean": number(contrast, "mean_delta"),
                        "paired_delta_l95": number(contrast, "l95_delta"),
                        "paired_delta_u95": number(contrast, "u95_delta"),
                        "paired_density_effect_pass": number(contrast, "u95_delta") < 0.0,
                    }
                )
    write_csv(output / "effect_size_and_gate_distance.csv", rows)
    return rows


def tiling_stability(
    tables: dict[str, list[dict[str, str]]], output: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = (
        "raw_E_X",
        "calibrated_E_X",
        "raw_operator_cosine",
        "raw_E_W",
        "calibrated_E_W",
    )
    for method in METHODS:
        for metric in metrics:
            first = {
                (int(row["depth"]), row["role"]): number(row, metric)
                for row in tables["matrix"]
                if int(row["tiling_seed"]) == TILING_SEEDS[0] and row["method"] == method
            }
            second = {
                (int(row["depth"]), row["role"]): number(row, metric)
                for row in tables["matrix"]
                if int(row["tiling_seed"]) == TILING_SEEDS[1] and row["method"] == method
            }
            keys = sorted(first)
            left = np.asarray([first[key] for key in keys], dtype=np.float64)
            right = np.asarray([second[key] for key in keys], dtype=np.float64)
            correlation = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else float("nan")
            if not math.isfinite(correlation):
                correlation = 1.0 if np.array_equal(left, right) else 0.0
            rows.append(
                {
                    "level": "matrix_72",
                    "method": method,
                    "aggregation": "matrix",
                    "metric": metric,
                    "pearson_across_tilings": correlation,
                    "mean_absolute_difference": float(np.mean(np.abs(right - left))),
                    "max_absolute_difference": float(np.max(np.abs(right - left))),
                    "mean_signed_seed2_minus_seed1": float(np.mean(right - left)),
                }
            )
        for aggregation in ROLES + ("macro", "micro"):
            for metric in metrics:
                first = lookup(
                    tables["aggregate"],
                    tiling_seed=TILING_SEEDS[0],
                    method=method,
                    aggregation=aggregation,
                )
                second = lookup(
                    tables["aggregate"],
                    tiling_seed=TILING_SEEDS[1],
                    method=method,
                    aggregation=aggregation,
                )
                rows.append(
                    {
                        "level": "aggregate",
                        "method": method,
                        "aggregation": aggregation,
                        "metric": metric,
                        "pearson_across_tilings": "",
                        "mean_absolute_difference": abs(number(second, metric) - number(first, metric)),
                        "max_absolute_difference": abs(number(second, metric) - number(first, metric)),
                        "mean_signed_seed2_minus_seed1": number(second, metric) - number(first, metric),
                    }
                )
    write_csv(output / "tiling_stability.csv", rows)
    return rows


def configure_plots() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 190,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def plot_k_trajectories(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = configure_plots()
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 14), sharex=True)
    panel_specs = (
        ("macro", "raw_E_X", "Raw macro $E_X$"),
        ("micro", "raw_E_X", "Raw micro $E_X$"),
        ("macro", "calibrated_E_X", "Calibrated macro $E_X$"),
        ("micro", "calibrated_E_X", "Calibrated micro $E_X$"),
    )
    colors = {"d2": "#1f77b4", "all": "#d62728"}
    for column, seed in enumerate(TILING_SEEDS):
        for row_index, (aggregation, metric, title) in enumerate(panel_specs):
            axis = axes[row_index, column]
            for scope in ("d2", "all"):
                group = sorted(
                    [
                        row
                        for row in rows
                        if row["tiling_seed"] == seed
                        and row["aggregation"] == aggregation
                        and row["scope"] == scope
                    ],
                    key=lambda row: row["K"],
                )
                x = np.asarray([row["K"] for row in group])
                y = np.asarray([row[metric] for row in group])
                low = np.asarray([row[f"{metric}_bootstrap_l95"] for row in group])
                high = np.asarray([row[f"{metric}_bootstrap_u95"] for row in group])
                axis.plot(x, y, marker="o", linewidth=2, color=colors[scope], label=scope)
                axis.fill_between(x, low, high, color=colors[scope], alpha=0.12)
                for xv, yv in zip(x, y, strict=True):
                    axis.annotate(f"{yv:.3f}", (xv, yv), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
            axis.axhline(1.0, color="black", linestyle=":", linewidth=1, label="zero reconstruction")
            axis.axhline(0.90, color="black", linestyle="--", linewidth=1, label="old G1 U95 gate")
            axis.set_title(f"{title}; tiling {seed}")
            axis.set_xticks((1, 4, 8))
            axis.set_xlabel("records per source×dataset pair, up to K")
            axis.set_ylabel("relative error")
            if row_index == 0:
                axis.legend(fontsize=8)
    figure.suptitle("Condition-density K trajectories (d2 and all-dataset lines kept separate)")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path)
    plt.close(figure)


def plot_role_depth_heatmaps(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = configure_plots()
    metrics = (
        ("delta_all_k8_minus_d2_k1_raw_E_X", r"$\Delta$ raw $E_X$ (negative improves)"),
        (
            "delta_all_k8_minus_d2_k1_calibrated_E_X",
            r"$\Delta$ calibrated $E_X$ (negative improves)",
        ),
        (
            "delta_all_k8_minus_d2_k1_raw_operator_cosine",
            r"$\Delta$ operator cosine (positive improves)",
        ),
    )
    figure, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, sharey=True)
    for row_index, (metric, title) in enumerate(metrics):
        values_all = np.asarray([float(row[metric]) for row in rows])
        limit = max(float(np.max(np.abs(values_all))), 1e-8)
        for column, seed in enumerate(TILING_SEEDS):
            axis = axes[row_index, column]
            matrix = np.asarray(
                [
                    [
                        next(
                            float(row[metric])
                            for row in rows
                            if row["tiling_seed"] == seed
                            and row["role"] == role
                            and row["depth"] == depth
                        )
                        for depth in range(12)
                    ]
                    for role in ROLES
                ]
            )
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
            axis.set_title(f"{title}; tiling {seed}")
            axis.set_yticks(range(len(ROLES)), [ROLE_LABELS[role] for role in ROLES])
            axis.set_xticks(range(12))
            axis.set_xlabel("ViT-B depth")
            axis.set_ylabel("role")
            figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle("all_k8 − d2_k1 held-out matrix effects by role and depth")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path)
    plt.close(figure)


def plot_template_drift(summary: Sequence[dict[str, Any]], path: Path) -> None:
    plt = configure_plots()
    comparisons = []
    for row in summary:
        if row["aggregation"] == "all" and row["vector_kind"] == "joint":
            comparisons.append(row["comparison"])
    comparisons = list(dict.fromkeys(comparisons))
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for column, kind in enumerate(("c_var", "c_patch", "joint")):
        group = [
            row
            for row in summary
            if row["aggregation"] == "all" and row["vector_kind"] == kind
        ]
        by_name = {row["comparison"]: row for row in group}
        x = np.arange(len(comparisons))
        cosine = np.asarray([by_name[name]["cosine_mean"] for name in comparisons])
        cosine_low = np.asarray([by_name[name]["cosine_min"] for name in comparisons])
        relative = np.asarray([by_name[name]["relative_l2_mean"] for name in comparisons])
        relative_high = np.asarray([by_name[name]["relative_l2_p90"] for name in comparisons])
        axes[0, column].plot(x, cosine, marker="o", color="#2c7fb8")
        axes[0, column].fill_between(x, cosine_low, cosine, color="#2c7fb8", alpha=0.15)
        axes[0, column].set_title(f"{kind}: template cosine")
        axes[0, column].set_ylabel("mean (band: min to mean)")
        axes[1, column].plot(x, relative, marker="o", color="#d95f0e")
        axes[1, column].fill_between(x, relative, relative_high, color="#d95f0e", alpha=0.15)
        axes[1, column].set_title(f"{kind}: relative L2 drift")
        axes[1, column].set_ylabel("mean (band: mean to p90)")
        axes[1, column].set_xticks(x, comparisons, rotation=42, ha="right")
    figure.suptitle("Condition-template convergence and total drift (72 role×depth cells)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path)
    plt.close(figure)


def plot_gains_support(
    gains: Sequence[dict[str, Any]], support: Sequence[dict[str, Any]], path: Path
) -> None:
    plt = configure_plots()
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = dict(zip(ROLES, plt.cm.tab10.colors[: len(ROLES)], strict=True))
    styles = {"d2": "-", "all": "--"}
    for role in ROLES:
        for scope in ("d2", "all"):
            group = sorted(
                [
                    row
                    for row in gains
                    if row["role"] == role and row["method"].startswith(f"{scope}_")
                ],
                key=lambda row: int(row["method"].split("k")[-1]),
            )
            axes[0].plot(
                [int(row["method"].split("k")[-1]) for row in group],
                [number(row, "gain") for row in group],
                color=colors[role],
                linestyle=styles[scope],
                marker="o",
                label=f"{ROLE_LABELS[role]} {scope}",
            )
    gain_values = [number(row, "gain") for row in gains]
    gain_low, gain_high = min(gain_values), max(gain_values)
    gain_padding = max(0.06 * (gain_high - gain_low), 0.025)
    axes[0].set_ylim(max(0.0, gain_low - gain_padding), gain_high + gain_padding)
    if gain_low - gain_padding <= 0.25 <= gain_high + gain_padding:
        axes[0].axhline(0.25, color="black", linestyle=":", linewidth=1)
    axes[0].set_title("Source-fit role gains")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("operator gain")
    axes[0].set_xticks((1, 4, 8))
    axes[0].legend(ncol=2, fontsize=7)
    support_map = {row["variant"]: row for row in support}
    for scope, color in (("d2", "#1f77b4"), ("all", "#d62728")):
        methods = [f"{scope}_k{k}" for k in (1, 4, 8)]
        axes[1].plot(
            (1, 4, 8),
            [support_map[method]["kish_effective_records_mean_over_72_cells"] for method in methods],
            marker="o",
            color=color,
            label=scope,
        )
        axes[1].fill_between(
            (1, 4, 8),
            [support_map[method]["kish_effective_records_min_over_72_cells"] for method in methods],
            [support_map[method]["kish_effective_records_max_over_72_cells"] for method in methods],
            color=color,
            alpha=0.12,
        )
        axes[2].plot(
            (1, 4, 8),
            [support_map[method]["selected_records"] for method in methods],
            marker="o",
            color=color,
            label=f"{scope} records",
        )
    axes[1].set_title("Kish-effective record support per cell")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("effective records (mean; band min–max)")
    axes[1].set_xticks((1, 4, 8))
    axes[1].legend()
    axes[2].set_title("Actual selected records in eligible cohort")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("records")
    axes[2].set_xticks((1, 4, 8))
    axes[2].legend()
    figure.suptitle("Gain movement versus hierarchical conditioning support")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)


def plot_tiling_stability(
    tables: dict[str, list[dict[str, str]]], stability: Sequence[dict[str, Any]], path: Path
) -> None:
    plt = configure_plots()
    metrics = (
        ("raw_E_X", "Raw matrix $E_X$"),
        ("calibrated_E_X", "Calibrated matrix $E_X$"),
        ("raw_operator_cosine", "Matrix operator cosine"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    colors = {"d2_k1": "#1f77b4", "all_k8": "#d62728"}
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        limits: list[float] = []
        for method in ("d2_k1", "all_k8"):
            first = {
                (row["depth"], row["role"]): number(row, metric)
                for row in tables["matrix"]
                if int(row["tiling_seed"]) == TILING_SEEDS[0] and row["method"] == method
            }
            second = {
                (row["depth"], row["role"]): number(row, metric)
                for row in tables["matrix"]
                if int(row["tiling_seed"]) == TILING_SEEDS[1] and row["method"] == method
            }
            keys = sorted(first)
            x = np.asarray([first[key] for key in keys])
            y = np.asarray([second[key] for key in keys])
            limits.extend(x.tolist() + y.tolist())
            audit = lookup(
                stability,
                level="matrix_72",
                method=method,
                aggregation="matrix",
                metric=metric,
            )
            axis.scatter(
                x,
                y,
                s=17,
                alpha=0.72,
                color=colors[method],
                label=f"{method}, r={float(audit['pearson_across_tilings']):.3f}",
            )
        low, high = min(limits), max(limits)
        padding = max((high - low) * 0.04, 1e-5)
        axis.plot((low - padding, high + padding), (low - padding, high + padding), "k:", linewidth=1)
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        axis.set_title(title)
        axis.set_xlabel(str(TILING_SEEDS[0]))
        axis.set_ylabel(str(TILING_SEEDS[1]))
        axis.legend(fontsize=8)
    figure.suptitle("Tiling stability across the 72 held-out role×depth matrices")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)


def plot_role_heterogeneity(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = configure_plots()
    metrics = (
        ("delta_all_k8_minus_d2_k1_raw_E_X", r"$\Delta$ raw $E_X$", True),
        ("delta_all_k8_minus_d2_k1_calibrated_E_X", r"$\Delta$ calibrated $E_X$", True),
        (
            "delta_all_k8_minus_d2_k1_raw_operator_cosine",
            r"$\Delta$ operator cosine",
            False,
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    x = np.arange(len(ROLES))
    width = 0.36
    for axis, (metric, title, lower_better) in zip(axes, metrics, strict=True):
        for index, seed in enumerate(TILING_SEEDS):
            values = [
                next(
                    float(row[metric])
                    for row in rows
                    if row["tiling_seed"] == seed and row["role"] == role
                )
                for role in ROLES
            ]
            axis.bar(x + (index - 0.5) * width, values, width=width, label=str(seed), alpha=0.82)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x, [ROLE_LABELS[role] for role in ROLES], rotation=25)
        axis.set_title(title + (" (negative improves)" if lower_better else " (positive improves)"))
        axis.set_ylabel("all_k8 − d2_k1")
        axis.legend(fontsize=8)
    figure.suptitle("Role heterogeneity of the maximum-density effect")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)


def write_readme(
    *,
    output: Path,
    dense: Path,
    sparse: Path,
    decision: dict[str, Any],
    validation: dict[str, Any],
    reproduction: dict[str, Any],
    effects: Sequence[dict[str, Any]],
    contrasts: Sequence[dict[str, Any]],
    drift_summary: Sequence[dict[str, Any]],
    heterogeneity: dict[str, Any],
    gain_validity: dict[str, Any],
) -> None:
    primary = [
        row
        for row in effects
        if row["aggregation"] == "macro" and row["metric"] == "raw_E_X"
    ]
    saturation = [
        row
        for row in contrasts
        if row["contrast"] == "all_k8_minus_all_k4"
        and row["aggregation"] == "macro"
        and row["metric"] == "raw_E_X"
    ]
    total_drift = lookup(
        drift_summary,
        comparison="d2_k1_to_all_k8",
        vector_kind="joint",
        aggregation="all",
    )
    lines = [
        "# Source condition-density post-run review",
        "",
        "This report is generated only after the source-only dense run reached its COMPLETE manifest. "
        "The analyzer is CPU-only, performed no model forward, and did not read a target-domain artifact.",
        "",
        "## Validity and sparse reproduction",
        "",
        f"- Dense source artifact: `{dense}`.",
        f"- Locked sparse source reference: `{sparse}`.",
        f"- Matrix grid: {validation['row_counts']['matrix']} finite rows; exact 2 tilings × 10 methods × 12 depths × 6 roles.",
        f"- Cache: {validation['cache']['cache_records']} finite vectors in {validation['cache']['cache_shards']} hash-validated shards.",
        f"- Sparse prediction hashes exact: {reproduction['all_144_prediction_hashes_exact']} (144/144).",
        f"- Sparse template tensors bit-exact: {reproduction['all_144_template_c_var_c_patch_tensors_bit_exact']} (72 cells × 2 tensors).",
        f"- Maximum old/new sparse numeric difference: matrix={reproduction['max_matrix_numeric_abs_diff']:.3g}, "
        f"gain={reproduction['max_gain_abs_diff']:.3g}, aggregate={reproduction['max_aggregate_abs_diff']:.3g}, "
        f"bootstrap={reproduction['max_bootstrap_abs_diff']:.3g}.",
        "- Aggregate metrics and both absolute and paired bootstraps independently recompute from `matrix_metrics.csv` within 1e-12.",
        "",
        "## Frozen primary density contrast",
        "",
        f"Runner decision: **{decision['interpretation']}**. This is a source-only post-hoc mechanism diagnostic; it cannot retroactively pass the old G1.",
        "",
        "| tiling | d2_k1 raw macro E_X | all_k8 | point delta | paired U95(delta) | all_k8 absolute U95 | old G1 U95<.90 |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['tiling_seed']} | {row['d2_k1_point']:.6f} | {row['all_k8_point']:.6f} | "
            f"{row['point_delta_all_k8_minus_d2_k1']:.6f} | {row['paired_delta_u95']:.6f} | "
            f"{row['all_k8_absolute_bootstrap_u95']:.6f} | {row['all_k8_old_g1_operator_gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "The paired primary asks whether density changes raw macro operator error, not whether the new arm by itself clears the old absolute gate. "
            "`effect_size_and_gate_distance.csv` reports both quantities, the zero-reconstruction baseline, native-C distance, and the fraction of any sparse-to-zero gap closed.",
            "",
            "## Saturation, template movement, and heterogeneity",
            "",
        ]
    )
    for row in saturation:
        lines.append(
            f"- Tiling {row['tiling_seed']}: all K8−K4 raw macro E_X mean delta "
            f"{number(row, 'mean_delta'):.6f}, bootstrap 5th/95th quantiles "
            f"[{number(row, 'l95_delta'):.6f}, {number(row, 'u95_delta'):.6f}]."
        )
    lines.extend(
        [
            f"- Joint-template d2_k1→all_k8 drift over 72 cells: mean cosine {number(total_drift, 'cosine_mean'):.6f}, "
            f"mean relative L2 {number(total_drift, 'relative_l2_mean'):.6f}, p90 relative L2 {number(total_drift, 'relative_l2_p90'):.6f}.",
            f"- Calibrated secondaries valid under the runner's [0.25, 4] gain rule: {gain_validity['calibrated_secondary_valid']}.",
        ]
    )
    for seed in TILING_SEEDS:
        raw = heterogeneity[str(seed)]["raw_E_X"]
        lines.append(
            f"- Tiling {seed}: raw E_X improves in {raw['roles_improved']}/{raw['roles_total']} roles; "
            f"role-delta range [{raw['min_delta']:.6f}, {raw['max_delta']:.6f}]."
        )
    lines.extend(
        [
            "",
            "## Artifact guide",
            "",
            "- `k_trajectories.csv`, `k_trajectories.png`: separate d2/all K=1,4,8 lines for raw/calibrated macro/micro E_X and bootstrap bands.",
            "- `role_depth_density_deltas.csv`, `role_depth_density_heatmaps.png`: all_k8−d2_k1 matrix effects by role/depth/tiling.",
            "- `template_drift_cells.csv`, `template_drift_summary.csv`, `template_drift.png`: K1→K4, K4→K8 and convergence to all_k8.",
            "- `effective_support_cells.csv`, `effective_support_summary.csv`, `gains_and_effective_support.png`: exact hierarchical/Kish support and fitted gains.",
            "- `tiling_stability.csv`, `tiling_stability.png`: matrix correlations and aggregate differences across the two tilings.",
            "- `role_heterogeneity.csv`, `role_heterogeneity.png`: role-level density effects.",
            "- `effect_size_and_gate_distance.csv`: point/paired effect sizes, zero baseline, native-C distance, and absolute-gate margins.",
            "- `sparse_*_reproduction.csv`, `sparse_reproduction_summary.json`: old/new d2_k1 parity.",
            "- `validation_summary.json`: completeness, finiteness, cache hashes, and independent arithmetic checks.",
            "",
            "## Scope caveats",
            "",
            "- `all_k8` means every available source dataset identity with up to eight records per source×dataset pair; it does not mean all 290,966 available records.",
            "- Breadth is supported only by the minority of source units with more than two datasets, so role/family heterogeneity matters.",
            "- The experiment is one held-out source architecture and diagnoses conditioning estimation. It does not establish cross-domain transfer.",
            "- The target remains sealed regardless of this result.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    dense = args.dense_dir.resolve()
    sparse = args.sparse_dir.resolve()
    output = args.output_dir.resolve()
    started = time.monotonic()
    print(
        json.dumps(
            {
                "stage": "startup",
                "experiment": "source_condition_density_postrun_review",
                "device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "dtype": "float64 metrics / float32 artifact validation",
                "seed": "read from completed dense resolved_config; no new stochastic analysis",
                "cache_mode": "read-only validation",
                "dense_dir": str(dense),
                "sparse_dir": str(sparse),
                "output_dir": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("stage=completion_gate", flush=True)
    run_manifest, decision = require_completed_run(dense, sparse)
    output.mkdir(parents=True, exist_ok=True)

    print("stage=table_cache_validation", flush=True)
    tables, validation = validate_tables(dense)
    validation["completed_run_manifest"] = {
        "status": run_manifest["status"],
        "target_access": run_manifest["target_access"],
        "matrix_metric_rows": run_manifest["counts"]["matrix_metric_rows"],
    }
    write_json(output / "validation_summary.json", validation)

    print("stage=sparse_exact_reproduction", flush=True)
    reproduction = sparse_reproduction(dense, sparse, tables, output)

    print("stage=derived_metrics", flush=True)
    trajectories = build_k_trajectories(tables, output)
    role_depth = build_role_depth_deltas(tables, output)
    _drift_cells, drift_summary = template_drift(dense, output)
    _support_cells, support_summary = effective_support(dense, output)
    heterogeneity_rows, heterogeneity_summary = role_heterogeneity(tables, output)
    effects = effect_and_gate(tables, output)
    stability = tiling_stability(tables, output)

    print("stage=plots", flush=True)
    plot_k_trajectories(trajectories, output / "k_trajectories.png")
    plot_role_depth_heatmaps(role_depth, output / "role_depth_density_heatmaps.png")
    plot_template_drift(drift_summary, output / "template_drift.png")
    plot_gains_support(tables["gains"], support_summary, output / "gains_and_effective_support.png")
    plot_tiling_stability(tables, stability, output / "tiling_stability.png")
    plot_role_heterogeneity(heterogeneity_rows, output / "role_heterogeneity.png")

    print("stage=report", flush=True)
    gain_validity = load_json(dense / "source_gain_validity.json")
    write_readme(
        output=output,
        dense=dense,
        sparse=sparse,
        decision=decision,
        validation=validation,
        reproduction=reproduction,
        effects=effects,
        contrasts=tables["contrasts"],
        drift_summary=drift_summary,
        heterogeneity=heterogeneity_summary,
        gain_validity=gain_validity,
    )
    manifest = {
        "status": "COMPLETE_CPU_ONLY_SOURCE_DENSITY_POSTRUN_REVIEW",
        "elapsed_seconds": time.monotonic() - started,
        "device": "cpu",
        "model_forward": False,
        "target_access": False,
        "dense_run_manifest_sha256": sha256_file(dense / "run_manifest.json"),
        "dense_matrix_metrics_sha256": sha256_file(dense / "matrix_metrics.csv"),
        "sparse_matrix_metrics_sha256": sha256_file(sparse / "matrix_metrics.csv"),
        "decision": decision,
        "artifacts": sorted(str(path) for path in output.iterdir()),
    }
    write_json(output / "review_manifest.json", manifest)
    print(
        json.dumps(
            {
                "stage": "complete",
                "elapsed_seconds": manifest["elapsed_seconds"],
                "density_interpretation": decision["interpretation"],
                "sparse_prediction_hashes_exact": reproduction[
                    "all_144_prediction_hashes_exact"
                ],
                "artifacts": manifest["artifacts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
