#!/usr/bin/env python3
"""Frozen CPU analysis for the global-context latent-geometry ablation.

This module deliberately contains no Weight-AE import or model execution path.
It consumes only the sealed runner artifacts, validates their complete grids,
recomputes every registered statistic from stored codes/features, and writes
the 25 tables/reports and 11 plots fixed in the prospective design.

Use ``--self-test`` to exercise the analysis primitives on synthetic arrays.
The self-test never opens the formal runner directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (PROJECT / "artifacts/crossmodal_united_structure").resolve()
DESIGN_PATH = PROJECT / "docs/notes/global_context_latent_geometry_ablation_frozen_design_20260816.md"
DESIGN_SHA256 = "6cacb7bb269fd22a7652b07d37e026c8e8e24d6e8023e437656ddad794f1d56f"
CONTRACT_PATH = PROJECT / "docs/notes/global_context_latent_geometry_ablation_recovery_contract_20260816.json"
DEFAULT_INPUT = ARTIFACT_ROOT / "global_context_latent_geometry_ablation_run_20260816"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "global_context_latent_geometry_ablation_analysis_20260816"
RUNNER_PATH = PROJECT / "experiments/run_global_context_latent_geometry_ablation.py"
PANEL_BUILDER_PATH = PROJECT / "experiments/build_prospective_geometry_matched_panels.py"
ANALYZER_TRANSACTION_MARKER = ".analyzer_transaction.json"
ANALYZER_FAILURE_RECORD = "failure_record.json"

PANELS = ("source_vit_b_flickr", "beans", "trocr_sroie")
ROLES = ("attn_query", "attn_key", "attn_value", "attn_output", "ffn_up", "ffn_down")
ATTENTION_ROLES = ROLES[:4]
DEPTHS = tuple(range(12))
TILING_SEEDS = (26_081_901, 26_081_902)
UNTRAINED_SEEDS = (26_081_971, 26_081_972, 26_081_973)
COUNTSKETCH_SEEDS = (26_081_941, 26_081_942, 26_081_943, 26_081_944, 26_081_945)
BASE_SEED = 26_081_931
QAP_BASE_SEED = 26_081_951
STABILITY_BASE_SEED = 26_081_961
N_RESAMPLES = 10_000

ANALYSIS_RUNTIME_KEYS = (
    "python",
    "python_major_minor",
    "executable",
    "torch",
    "numpy",
    "scikit-learn",
    "scipy",
    "pandas",
    "matplotlib",
)
FROZEN_ANALYSIS_RUNTIME = {
    "python_major_minor": "3.12",
    "torch": "2.10.0+cu128",
    "numpy": "2.3.5",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.0",
    "pandas": "3.0.0",
    "matplotlib": "3.10.8",
}
CONTRACT_RUNTIME_KEYS = (
    *ANALYSIS_RUNTIME_KEYS,
    "omegaconf",
    "hydra-core",
    "cuda",
)

LEARNED_REPRESENTATIONS = ("learned_cell", "learned_global", "learned_zero")
UNTRAINED_REPRESENTATIONS = tuple(f"untrained_global_{seed}" for seed in UNTRAINED_SEEDS)
COUNTSKETCH_REPRESENTATIONS = tuple(f"countsketch_{seed}" for seed in COUNTSKETCH_SEEDS)
REPRESENTATIONS = (
    *LEARNED_REPRESENTATIONS,
    *UNTRAINED_REPRESENTATIONS,
    "raw_simple",
    *COUNTSKETCH_REPRESENTATIONS,
)
CODE_REPRESENTATIONS = (*LEARNED_REPRESENTATIONS, *UNTRAINED_REPRESENTATIONS)
RELIABILITY_REPRESENTATIONS = tuple(rep for rep in REPRESENTATIONS if rep != "raw_simple")
COMPARATORS = ("untrained_global_endpoint", "raw_simple", "countsketch_endpoint")
UNTRAINED_MODEL_KIND = "untrained_model"

GLOBAL_C_VAR_SHA256 = "3681bfd5033ade56c65f6d090ce41dea33866295580b1c85d679c6b3ae06303c"
GLOBAL_C_PATCH_SHA256 = "b8755c89a668b4bd844f4bb53511951e5cd85b56d91bd537f28c5655df0829e4"
ZERO_C_SHA256 = "96225d4f7e38e1ce50382fb47280587f7f6cf27782234c35b772e18f027d406b"
SOURCE_TEMPLATE_FILE_SHA256 = "8fc6c61bb6baae4e7b1d618133ec651a91386d90c66f540182faa7dfb1655f99"

RUNNER_FILES = (
    "run.log",
    "resolved_config.json",
    "preexecution_contract.json",
    "preexecution_binding.json",
    "input_audit.json",
    "target_access_seal.json",
    "source_template_audit.json",
    "model_contract.json",
    "known_source_numeric_preflight.json",
    "dataflow_audit.json",
    "tiling_manifest.csv",
    "tiling_indices.pt",
    "code_manifest.csv",
    "latent_codes.pt",
    "raw_simple_features.csv",
    "countsketch_maps.json",
    "aggregate_feature_manifest.csv",
    "aggregate_features.npz",
    "zero_weight_sanity.json",
    "input_immutability_recheck.json",
    "runner_metadata.json",
)
CONTRACT_SCHEMA_KEYS = (
    "schema_version", "contract_schema_keys", "created_utc", "source_only_seal",
    "frozen_design_path", "frozen_design_sha256", "runner_path", "runner_sha256",
    "analyzer_path", "analyzer_sha256", "runtime", "execution_config", "frozen_grids",
    "scripts", "model_dependencies", "input_audit", "static_call_graph_audit",
    "expected_counts", "artifact_contract", "dataflow_contract", "analysis_contract",
)
TOP_LEVEL_FILES = (
    "analysis.log",
    "resolved_analysis_config.json",
    "input_manifest_audit.json",
    "feature_audit.json",
    "latent_pair_effects.csv",
    "tiling_reliability.csv",
    "tiling_bootstrap.csv",
    "rsa.csv",
    "rsa_qap_permutations.csv",
    "role_probe_predictions.csv",
    "role_probe_metrics.csv",
    "role_bootstrap.csv",
    "depth_probe_predictions.csv",
    "depth_probe_metrics.csv",
    "depth_bootstrap.csv",
    "level_b_role_deltas.csv",
    "level_b_role_bootstrap.csv",
    "level_b_secondary_noninferiority.csv",
    "variance_decomposition.csv",
    "pca_scores.csv",
    "pca_scaler_and_loadings.npz",
    "decision_cells.csv",
    "decision.json",
    "plot_audit.json",
    "README.md",
)
PLOT_FILES = (
    "global_pc1_pc2_by_panel.png",
    "global_pc1_pc2_by_role.png",
    "global_pc1_pc2_by_depth.png",
    "global_role_facets_by_panel.png",
    "global_depth_facets_by_panel.png",
    "cell_global_zero_projected_pc1_pc2.png",
    "tiling_reliability_by_panel_and_representation.png",
    "rsa_by_panel_pair_and_representation.png",
    "attention_role_balanced_accuracy.png",
    "depth_spearman_by_panel_and_representation.png",
    "level_b_attention_paired_deltas.png",
)

TILING_COLUMNS = (
    "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
    "d_in", "d_out", "row_groups", "col_groups", "num_tiles",
    "row_index_sha256", "column_index_sha256", "partition_sha256",
    "weight_shape_bytes_sha256", "coverage_min", "coverage_max",
    "weight_reassembly_bit_exact", "coordinate_reassembly_bit_exact",
    "shape_global_identity_bit_exact",
)
CODE_COLUMNS = (
    "representation", "panel_id", "tiling_seed", "tiling_index", "depth", "role",
    "matrix_key", "d_in", "d_out", "num_tiles", "code_width", "code_key",
    "code_tensor_sha256", "partition_sha256", "weight_shape_bytes_sha256",
    "c_var_sha256", "c_patch_sha256", "model_kind", "model_seed", "condition_kind",
    "activation_consumers", "metadata_label_consumers", "finite", "nonzero",
)
FEATURE_COLUMNS = (
    "representation", "panel_id", "tiling_seed", "tiling_index", "depth", "role",
    "matrix_key", "d_in", "d_out", "num_tiles", "feature_dim", "array_name",
    "array_row_index", "source_kind", "source_seed", "feature_tensor_sha256",
    "partition_sha256", "tiling_invariant", "finite", "nonzero",
)
RAW_SIMPLE_KEYS = (
    "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
    "d_in", "d_out", "num_tiles",
)
RAW_SIMPLE_FEATURES = (
    "log_numel", "log_d_in", "log_d_out", "mean", "population_std", "rms",
    "frobenius_norm", "mean_abs", "minimum", "maximum", "q01", "q05", "q10",
    "q25", "q50", "q75", "q90", "q95", "q99", "row_rms_mean",
    "row_rms_population_std", "row_rms_minimum", "row_rms_q10", "row_rms_q25",
    "row_rms_q50", "row_rms_q75", "row_rms_q90", "row_rms_maximum",
    "column_rms_mean", "column_rms_population_std", "column_rms_minimum",
    "column_rms_q10", "column_rms_q25", "column_rms_q50", "column_rms_q75",
    "column_rms_q90", "column_rms_maximum",
)
RAW_SIMPLE_COLUMNS = (
    *RAW_SIMPLE_KEYS,
    *RAW_SIMPLE_FEATURES,
    "feature_tensor_sha256", "weight_shape_bytes_sha256", "tiling_invariant",
)

EXPECTED_COUNTS = {
    "tiling_manifest_rows": 432,
    "tiling_index_entries": 432,
    "code_manifest_rows": 2_592,
    "latent_code_entries": 2_592,
    "latent_code_rows": 746_496,
    "raw_simple_rows": 432,
    "countsketch_map_entries": 20_480,
    "aggregate_feature_rows": 5_184,
    "latent_pair_effect_rows": 864,
    "tiling_reliability_rows": 42,
    "tiling_bootstrap_rows": 30_000,
    "rsa_rows": 42,
    "rsa_qap_rows": 30_000,
    "role_prediction_rows": 1_728,
    "role_metric_rows": 42,
    "role_bootstrap_rows": 30_000,
    "depth_prediction_rows": 2_592,
    "depth_metric_rows": 294,
    "depth_bootstrap_rows": 30_000,
    "level_b_delta_rows": 12,
    "level_b_bootstrap_rows": 30_000,
    "level_b_secondary_rows": 18,
    "variance_rows": 36,
    "pca_rows": 1_296,
    "decision_rows": 51,
    "plot_count": 11,
}
EXPECTED_RUNNER_COUNTS = {
    "panels": 3,
    "tilings": 2,
    "depths": 12,
    "roles": 6,
    "matrices_per_panel": 72,
    "tiles_per_role_matrix": {
        "attn_query": 144,
        "attn_key": 144,
        "attn_value": 144,
        "attn_output": 144,
        "ffn_up": 576,
        "ffn_down": 576,
    },
    "tiles_per_panel_tiling": 20_736,
    "tiling_manifest_rows": 432,
    "tiling_index_entries": 432,
    "code_representations": 6,
    "code_manifest_rows": 2_592,
    "latent_code_entries": 2_592,
    "latent_code_rows": 746_496,
    "latent_code_width": 512,
    "raw_simple_rows": 432,
    "raw_simple_feature_columns": 37,
    "countsketch_maps": 5,
    "countsketch_pairs": 20_480,
    "aggregate_manifest_rows": 5_184,
    "aggregate_arrays": 12,
    "aggregate_2560_arrays": 11,
    "aggregate_37_arrays": 1,
    "declared_raw_files": 21,
    "files_including_self_excluded_manifest": 22,
}
VARIANCE_DECOMPOSITION_CONTRACT = {
    "identifier": "balanced_panel_tiling_cell_feature_ss_v1",
    "input_axes": ["panel=3", "tiling=2", "cell=72", "feature"],
    "grand": "mean(panel,tiling,cell)",
    "cell_effect": "mean(panel,tiling)-grand",
    "panel_effect": "mean(tiling,cell)-grand",
    "interaction_effect": "mean(tiling)-grand-cell_effect-panel_effect",
    "tiling_residual": "value-mean(tiling)",
    "ss_multipliers": {
        "shared_cell": 6,
        "panel_main": 144,
        "panel_by_cell": 2,
        "tiling_residual": 1,
    },
    "required_checks": {
        "finite_total_ss": True,
        "total_ss_strictly_positive": True,
        "fraction_sum_abs_error_lt": 1e-10,
    },
}

HEX64 = __import__("re").compile(r"[0-9a-f]{64}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--design-path", type=Path, default=DESIGN_PATH)
    parser.add_argument("--external-contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output-dir", type=Path)
    return parser.parse_args(argv)


def validate_invocation(argv: Sequence[str] | None, args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed on the one formal CLI and the isolated self-test CLI."""

    raw = list(sys.argv[1:] if argv is None else argv)
    formal_flags = {"--input-dir", "--output-dir", "--design-path", "--external-contract"}
    observed_flags = {token.split("=", 1)[0] for token in raw if token.startswith("--")}
    if args.self_test:
        require(not observed_flags.intersection(formal_flags), "self-test may not carry formal path options")
        require(raw.count("--self-test") == 1, "self-test flag must occur exactly once")
        output_flag_count = sum(
            token == "--self-test-output-dir" or token.startswith("--self-test-output-dir=")
            for token in raw
        )
        require(output_flag_count <= 1, "self-test output flag duplicated")
        cursor = 0
        while cursor < len(raw):
            token = raw[cursor]
            if token == "--self-test":
                cursor += 1
            elif token == "--self-test-output-dir":
                require(cursor + 1 < len(raw) and not raw[cursor + 1].startswith("--"), "self-test output value absent")
                cursor += 2
            elif token.startswith("--self-test-output-dir="):
                require(bool(token.split("=", 1)[1]), "self-test output value absent")
                cursor += 1
            else:
                raise RuntimeError(f"noncanonical self-test argument: {token}")
        return {
            "pass": True,
            "mode": "self_test",
            "formal_path_options_present": False,
            "raw_argv": raw,
        }

    require(raw == [], f"formal analyzer invocation requires empty argv after script: {raw}")
    entrypoint = Path(sys.argv[0]).resolve(strict=True)
    script = Path(__file__).resolve(strict=True)
    require(entrypoint == script, f"formal analyzer entrypoint mismatch: {entrypoint} != {script}")
    return {
        "pass": True,
        "mode": "formal",
        "entrypoint": str(entrypoint),
        "raw_argv": [],
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, *labels: Any) -> int:
    payload = "|".join(str(item) for item in (base, *labels)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**63 - 1)


def tensor_sha256(array: np.ndarray, dtype: np.dtype[Any] | str) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def exact_columns(frame: pd.DataFrame, expected: Sequence[str], label: str) -> None:
    require(tuple(frame.columns) == tuple(expected), f"{label} exact columns mismatch: {tuple(frame.columns)}")


def exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    require(set(value) == set(expected), f"{label} exact keys mismatch: {sorted(value)}")


def ensure_finite(array: np.ndarray, label: str) -> None:
    require(np.isfinite(array).all(), f"{label} contains nonfinite values")


def expected_execution_config() -> dict[str, Any]:
    return {
        "device": "cuda:0",
        "batch_size": 64,
        "log_every_batches": 20,
        "output_dir": str(DEFAULT_INPUT.resolve(strict=False)),
        "analyzer_output_dir": str(DEFAULT_OUTPUT.resolve(strict=False)),
        "contract_path": str(CONTRACT_PATH.resolve(strict=False)),
        "global_seed": BASE_SEED,
        "common_tiling_seeds": list(TILING_SEEDS),
        "untrained_seeds": list(UNTRAINED_SEEDS),
        "countsketch_seeds": list(COUNTSKETCH_SEEDS),
        "panel_order": list(PANELS),
        "role_order": list(ROLES),
        "representation_order": list(REPRESENTATIONS),
        "device_dtype": "FP32 inputs; CUDA BF16 autocast; FP32 codes; FP64 aggregates",
        "cache_mode": "immutable_inputs_revalidated; old codes audit-only; all decision features fresh",
        "latent_sampling": False,
        "rope_2d_coordinates": "raw_integer",
        "decoder_forward": False,
        "distribution_encoder_forward": False,
    }


def expected_dataflow_contract() -> dict[str, Any]:
    return {
        "seal_installed_before_scientific_reads": True,
        "w_only_materialization_before_model_import_and_build": True,
        "encoder_function_positional_arguments": ["model", "W_tiles", "fixed_template"],
        "primary_global_condition_hashes": {
            "c_var": GLOBAL_C_VAR_SHA256,
            "c_patch": GLOBAL_C_PATCH_SHA256,
        },
        "panel_activation_model_consumers": 0,
        "metadata_label_model_consumers": 0,
        "distribution_encoder_calls": 0,
        "decoder_calls": 0,
        "old_code_or_tiling_decision_features": False,
        "analyzer_invoked_by_runner": False,
    }


def validate_static_call_graph_payload(payload: Mapping[str, Any], runner_sha256: str) -> dict[str, Any]:
    exact_keys(
        payload,
        (
            "schema_version", "script_sha256", "encoder_function", "formal_arguments",
            "forbidden_scientific_argument_or_local_names",
            "forbidden_decoder_or_distribution_attribute_calls",
            "encode_z_dec_subscript_call_count",
            "model_receives_only_weight_batch_and_expanded_fixed_condition", "pass",
        ),
        "contract static call graph",
    )
    require(payload["schema_version"] == "global_context_static_call_graph_audit_v1", "static call graph schema mismatch")
    require(payload["script_sha256"] == runner_sha256, "static call graph is not bound to runner SHA")
    require(payload["encoder_function"] == "encode_weight_tiles", "static encoder function mismatch")
    require(payload["formal_arguments"] == ["model", "W_tiles", "fixed_template"], "static encoder arguments mismatch")
    require(payload["forbidden_scientific_argument_or_local_names"] == [], "static encoder exposes scientific metadata")
    require(payload["forbidden_decoder_or_distribution_attribute_calls"] == [], "static encoder exposes forbidden model call")
    require(payload["encode_z_dec_subscript_call_count"] == 1, "static encoder invocation count mismatch")
    require(payload["model_receives_only_weight_batch_and_expanded_fixed_condition"] is True, "static model call contract failed")
    require(payload["pass"] is True, "static call graph did not pass")
    return {"pass": True, "runner_sha256": runner_sha256, "formal_arguments": payload["formal_arguments"]}


def validate_source_template_payload(payload: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, str]]:
    exact_keys(
        payload,
        (
            "schema_version", "template_file_sha256", "global", "cell_mean",
            "cell_medoid_audited_not_used", "global_hashes_match_frozen_design",
            "cell_count", "pass",
        ),
        "source template audit",
    )
    require(payload["schema_version"] == "global_context_source_template_audit_v1", "source template schema mismatch")
    require(payload["template_file_sha256"] == SOURCE_TEMPLATE_FILE_SHA256, "source template file SHA mismatch")
    exact_keys(payload["global"], ("c_var_sha256", "c_patch_sha256"), "source global template")
    require(
        payload["global"]
        == {"c_var_sha256": GLOBAL_C_VAR_SHA256, "c_patch_sha256": GLOBAL_C_PATCH_SHA256},
        "source global template hashes mismatch",
    )
    require(payload["cell_medoid_audited_not_used"] is True, "source cell medoid audit/use contract mismatch")
    require(payload["global_hashes_match_frozen_design"] is True and payload["cell_count"] == 72 and payload["pass"] is True, "source template summary mismatch")
    cell_mean = payload["cell_mean"]
    require(isinstance(cell_mean, Mapping), "source cell_mean must be a mapping")
    expected_keys = [f"depth={depth:02d}|role={role}" for depth in DEPTHS for role in ROLES]
    require(set(cell_mean) == set(expected_keys) and len(cell_mean) == 72, "source cell_mean exact grid mismatch")
    result: dict[tuple[int, str], dict[str, str]] = {}
    for depth in DEPTHS:
        for role in ROLES:
            key = f"depth={depth:02d}|role={role}"
            record = cell_mean[key]
            exact_keys(record, ("c_var_sha256", "c_patch_sha256"), f"source cell template {key}")
            require(all(HEX64.fullmatch(str(value)) for value in record.values()), f"source cell template hash malformed: {key}")
            result[(depth, role)] = {
                "c_var": str(record["c_var_sha256"]),
                "c_patch": str(record["c_patch_sha256"]),
            }
    return result


def validate_cell_template_hash_pair(
    depth: int,
    role: str,
    c_var_sha256: str,
    c_patch_sha256: str,
    cell_template_hashes: Mapping[tuple[int, str], Mapping[str, str]],
    *,
    label: str,
) -> None:
    expected = cell_template_hashes[(depth, role)]
    require(
        {"c_var": str(c_var_sha256), "c_patch": str(c_patch_sha256)} == expected,
        f"{label} differs from contracted full-72 cell-template map: depth={depth} role={role}",
    )


def validate_contracted_w_grid(materialization: Mapping[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    exact_keys(
        materialization,
        (
            "schema_version", "source_runtime", "candidate_runtime", "w_entry_count",
            "w_grid_rows", "w_grid_sha256", "tiling_row_count", "tiling_grid_sha256",
            "source_preflight", "w_only_entry_fields", "activation_tensors_retained",
            "activation_references_retained", "model_modules_loaded_before_completion",
            "garbage_collection_completed", "pass",
        ),
        "contract W-only materialization",
    )
    require(materialization["schema_version"] == "global_context_w_only_materialization_v1", "W materialization schema mismatch")
    require(materialization["w_entry_count"] == 216 and materialization["tiling_row_count"] == 432, "contract W/tiling counts mismatch")
    require(HEX64.fullmatch(str(materialization["tiling_grid_sha256"])) is not None, "contract tiling-grid SHA malformed")
    require(
        materialization["w_only_entry_fields"]
        == ["W", "depth", "panel_id", "role", "tilings", "weight_shape_bytes_sha256", "weight_tensor_sha256"],
        "contract W-only field grid mismatch",
    )
    require(materialization["activation_tensors_retained"] == 0 and materialization["activation_references_retained"] == 0, "contract retained activation")
    require(materialization["model_modules_loaded_before_completion"] == [], "contract loaded model before W-only completion")
    require(materialization["garbage_collection_completed"] is True and materialization["pass"] is True, "contract W materialization failed")
    rows = materialization["w_grid_rows"]
    require(isinstance(rows, list) and len(rows) == 216, "contract W row count mismatch")
    expected_grid = canonical_cells()
    observed_grid: list[tuple[str, int, str]] = []
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        exact_keys(
            row,
            ("panel_id", "depth", "role", "shape", "weight_shape_bytes_sha256", "weight_tensor_sha256"),
            "contract W row",
        )
        identity = (str(row["panel_id"]), int(row["depth"]), str(row["role"]))
        observed_grid.append(identity)
        d_in, d_out, _ = _matrix_shape(identity[2])
        require(list(row["shape"]) == [d_in, d_out], f"contract W shape mismatch: {identity}")
        require(HEX64.fullmatch(str(row["weight_shape_bytes_sha256"])) is not None, "contract W shape/bytes hash malformed")
        require(HEX64.fullmatch(str(row["weight_tensor_sha256"])) is not None, "contract W tensor hash malformed")
        result[identity] = dict(row)
    require(observed_grid == expected_grid and len(result) == 216, "contract W canonical grid/order mismatch")
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    require(digest == materialization["w_grid_sha256"], "contract W grid SHA mismatch")
    return result


def analysis_runtime_snapshot() -> dict[str, str]:
    """Return only interpreter/packages that can affect the CPU analysis."""

    import torch

    return {
        "python": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def validate_analysis_runtime(contract_runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before artifact analysis if the bound numerical runtime drifted."""

    require(isinstance(contract_runtime, Mapping), "contract runtime must be a mapping")
    exact_keys(contract_runtime, CONTRACT_RUNTIME_KEYS, "contract runtime")
    actual = analysis_runtime_snapshot()
    for key in ANALYSIS_RUNTIME_KEYS:
        require(
            actual[key] == contract_runtime[key],
            f"analysis runtime/contract mismatch for {key}: {actual[key]} != {contract_runtime[key]}",
        )
    for key, expected in FROZEN_ANALYSIS_RUNTIME.items():
        require(
            actual[key] == expected,
            f"analysis runtime/frozen-version mismatch for {key}: {actual[key]} != {expected}",
        )
    return {
        "pass": True,
        "exact_contract_match": True,
        "frozen_versions_match": True,
        "runtime": actual,
    }


def canonical_rows() -> list[tuple[str, int, int, int, str]]:
    return [
        (panel, seed, tiling_index, depth, role)
        for panel in PANELS
        for tiling_index, seed in enumerate(TILING_SEEDS, start=1)
        for depth in DEPTHS
        for role in ROLES
    ]


def canonical_cells() -> list[tuple[str, int, str]]:
    return [(panel, depth, role) for panel in PANELS for depth in DEPTHS for role in ROLES]


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger(f"global_context_analyzer.{id(output)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(output / "analysis.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def create_staged_output(final_output: Path, *, allowed_parent: Path) -> Path:
    resolved_parent = final_output.parent.resolve(strict=True)
    expected_parent = allowed_parent.resolve(strict=True)
    require(resolved_parent == expected_parent, f"output parent mismatch: {resolved_parent} != {expected_parent}")
    require(not final_output.exists() and not final_output.is_symlink(), f"final output must be fresh: {final_output}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.staging-pid-{os.getpid()}-",
            dir=resolved_parent,
        )
    ).resolve(strict=True)
    marker = {
        "schema_version": "global_context_analyzer_transaction_owner_v1",
        "owner_pid": os.getpid(),
        "staging_path": str(staging),
        "final_output_path": str(final_output.resolve(strict=False)),
        "formal_path_created": False,
    }
    json_write(staging / ANALYZER_TRANSACTION_MARKER, marker)
    return staging


def quarantine_failed_staging(
    staging: Path,
    final_output: Path,
    error: BaseException,
    *,
    failure_stage: str,
) -> dict[str, Any]:
    """Preserve a failed private tree and keep the frozen final path absent."""

    result: dict[str, Any] = {
        "schema_version": "global_context_analyzer_failure_quarantine_v1",
        "owner_pid": os.getpid(),
        "failure_stage": failure_stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "staging_path": str(staging),
        "final_output_path": str(final_output.resolve(strict=False)),
        "formal_path_absent": not final_output.exists() and not final_output.is_symlink(),
        "quarantine_path": None,
    }
    if not staging.is_dir() or staging.is_symlink():
        result["staging_present"] = False
        return result
    result["staging_present"] = True
    try:
        json_write(staging / ANALYZER_FAILURE_RECORD, result)
    except OSError as record_error:
        result["failure_record_error"] = {
            "type": type(record_error).__name__,
            "message": str(record_error),
        }
    base = final_output.with_name(
        f"{final_output.name}_FAILED_{time.time_ns()}_pid-{os.getpid()}"
    )
    quarantine = base
    suffix = 0
    while quarantine.exists() or quarantine.is_symlink():
        suffix += 1
        quarantine = base.with_name(f"{base.name}_{suffix}")
    os.rename(staging, quarantine)
    result["quarantine_path"] = str(quarantine)
    result["staging_present_after_quarantine"] = False
    result["formal_path_absent"] = not final_output.exists() and not final_output.is_symlink()
    return result


def require_formal_input(path: Path) -> Path:
    require(path.exists() and not path.is_symlink(), f"formal input absent or symlinked: {path}")
    resolved = path.resolve(strict=True)
    require(resolved.parent == ARTIFACT_ROOT, f"formal input must be under artifact root: {resolved}")
    return resolved


def artifact_manifest(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "artifact_manifest.json":
            continue
        rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "schema_version": "global_context_latent_geometry_ablation_analyzer_manifest_v1",
        "manifest_self_excluded": True,
        "artifacts": rows,
        "count": len(rows),
    }


def validate_analyzer_artifact_manifest(root: Path) -> dict[str, Any]:
    """Validate the exact 36-file analyzer publication before/after rename."""

    require(root.is_dir() and not root.is_symlink(), f"analyzer artifact root absent/symlinked: {root}")
    require(not any(path.is_symlink() for path in root.rglob("*")), "analyzer tree contains a symlink")
    manifest_path = root / "artifact_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "analyzer artifact manifest absent/symlinked")
    payload = json_read(manifest_path)
    exact_keys(payload, ("schema_version", "manifest_self_excluded", "artifacts", "count"), "analyzer manifest")
    require(
        payload["schema_version"] == "global_context_latent_geometry_ablation_analyzer_manifest_v1",
        "analyzer manifest schema mismatch",
    )
    require(payload["manifest_self_excluded"] is True, "analyzer manifest is not self-excluded")
    rows = payload["artifacts"]
    require(isinstance(rows, list) and payload["count"] == len(rows) == 36, "analyzer manifest count mismatch")
    expected_names = set(TOP_LEVEL_FILES) | set(PLOT_FILES)
    observed_names: list[str] = []
    for row in rows:
        exact_keys(row, ("path", "sha256", "bytes"), "analyzer manifest row")
        relative = str(row["path"])
        rel = Path(relative)
        require(
            relative == rel.as_posix()
            and not rel.is_absolute()
            and len(rel.parts) == 1
            and ".." not in rel.parts,
            f"unsafe/non-top-level analyzer manifest path: {relative}",
        )
        require(relative not in observed_names and relative != "artifact_manifest.json", f"duplicate/self analyzer artifact: {relative}")
        candidate = root / rel
        require(candidate.is_file() and not candidate.is_symlink(), f"analyzer artifact absent/symlinked: {relative}")
        require(HEX64.fullmatch(str(row["sha256"])) is not None, f"bad analyzer artifact SHA: {relative}")
        require(isinstance(row["bytes"], int) and not isinstance(row["bytes"], bool) and row["bytes"] >= 0, f"bad analyzer artifact size: {relative}")
        require(sha256_file(candidate) == row["sha256"] and candidate.stat().st_size == row["bytes"], f"analyzer artifact manifest mismatch: {relative}")
        observed_names.append(relative)
    require(observed_names == sorted(observed_names), "analyzer manifest rows not in canonical lexical order")
    require(set(observed_names) == expected_names, "analyzer manifest exact 36-file name grid mismatch")
    actual_names = {
        path.name for path in root.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    require(actual_names == expected_names, "analyzer publication contains missing/extra files")
    require(not any(path.is_dir() for path in root.iterdir()), "analyzer publication contains unexpected directory")
    return {
        "pass": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_count": 36,
        "exact_name_grid": True,
        "no_symlinks": True,
    }


def publish_staged_output(staging: Path, final_output: Path, *, allowed_parent: Path) -> dict[str, Any]:
    """Manifest-validate a private tree, then publish it by one same-FS rename."""

    require(staging.parent.resolve(strict=True) == allowed_parent.resolve(strict=True), "staging parent mismatch")
    require(final_output.parent.resolve(strict=True) == allowed_parent.resolve(strict=True), "final parent mismatch")
    require(not final_output.exists() and not final_output.is_symlink(), "final analyzer output appeared before publication")
    marker = staging / ANALYZER_TRANSACTION_MARKER
    require(marker.is_file() and not marker.is_symlink(), "analyzer staging owner marker absent")
    marker_payload = json_read(marker)
    require(marker_payload.get("staging_path") == str(staging), "analyzer staging marker path mismatch")
    require(marker_payload.get("final_output_path") == str(final_output.resolve(strict=False)), "analyzer staging marker final path mismatch")
    marker.unlink()
    json_write(staging / "artifact_manifest.json", artifact_manifest(staging))
    prepublish = validate_analyzer_artifact_manifest(staging)
    require(not final_output.exists() and not final_output.is_symlink(), "final analyzer output appeared during validation")
    os.rename(staging, final_output)
    try:
        require(final_output.is_dir() and not staging.exists(), "atomic analyzer publication postcondition failed")
        postpublish = validate_analyzer_artifact_manifest(final_output)
    except Exception:
        if final_output.is_dir() and not final_output.is_symlink() and not staging.exists():
            os.rename(final_output, staging)
        raise
    require(
        prepublish["manifest_sha256"] == postpublish["manifest_sha256"],
        "analyzer manifest changed across atomic publication",
    )
    return {
        "pass": True,
        "publication": "same_filesystem_atomic_rename",
        "final_output": str(final_output),
        "manifest_sha256": postpublish["manifest_sha256"],
        "artifact_count": 36,
    }


def validate_artifact_manifest(root: Path) -> dict[str, Any]:
    path = root / "artifact_manifest.json"
    require(path.is_file() and not path.is_symlink(), "runner artifact_manifest absent/symlinked")
    payload = json_read(path)
    exact_keys(
        payload,
        (
            "schema_version", "frozen_design_sha256", "runner_sha256", "analyzer_sha256",
            "external_contract_sha256", "self_excluded", "file_count", "files",
        ),
        "runner manifest",
    )
    require(payload["schema_version"] == "global_context_latent_geometry_raw_artifact_manifest_v1", "runner manifest schema mismatch")
    require(payload["frozen_design_sha256"] == DESIGN_SHA256, "runner manifest design hash mismatch")
    require(all(HEX64.fullmatch(str(payload[key])) for key in ("runner_sha256", "analyzer_sha256", "external_contract_sha256")), "runner manifest identity SHA malformed")
    require(payload["self_excluded"] is True, "runner manifest not self-excluded")
    rows = payload["files"]
    require(isinstance(rows, list) and payload["file_count"] == len(rows) == 21, "runner manifest count mismatch")
    seen: set[str] = set()
    for row in rows:
        exact_keys(row, ("path", "sha256", "bytes"), "runner manifest row")
        relative = str(row["path"])
        rel = Path(relative)
        require(relative == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts, f"unsafe manifest path: {relative}")
        require(relative not in seen and relative != "artifact_manifest.json", f"duplicate/self manifest row: {relative}")
        seen.add(relative)
        candidate = root / rel
        cursor = root
        for component in rel.parts:
            cursor = cursor / component
            require(not cursor.is_symlink(), f"manifest follows symlink: {relative}")
        resolved = candidate.resolve(strict=True)
        require(resolved.parent == root and resolved.is_file(), f"manifest artifact escapes/not top-level file: {relative}")
        require(HEX64.fullmatch(str(row["sha256"])) is not None, f"bad manifest SHA: {relative}")
        require(not isinstance(row["bytes"], bool) and int(row["bytes"]) >= 0, f"bad manifest size: {relative}")
        require(sha256_file(resolved) == row["sha256"] and resolved.stat().st_size == row["bytes"], f"manifest mismatch: {relative}")
    require(seen == set(RUNNER_FILES), f"runner artifact name grid mismatch: {sorted(seen)}")
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "artifact_manifest.json"}
    require(actual == seen, f"runner manifest completeness mismatch actual={sorted(actual)}")
    require(not any(path.is_symlink() for path in root.rglob("*")), "runner tree contains symlink")
    return {
        "pass": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "count": 21,
        "files": sorted(seen),
        "runner_sha256": payload["runner_sha256"],
        "analyzer_sha256": payload["analyzer_sha256"],
        "external_contract_sha256": payload["external_contract_sha256"],
    }


def _bool_series(series: pd.Series, label: str) -> np.ndarray:
    allowed = {True, False, 0, 1, "True", "False", "true", "false", "0", "1"}
    require(set(series.unique()).issubset(allowed), f"{label} has non-boolean values")
    return series.map(lambda x: str(x).lower() in {"true", "1"}).to_numpy(dtype=bool)


def _matrix_shape(role: str) -> tuple[int, int, int]:
    if role in ATTENTION_ROLES:
        return (768, 768, 144)
    if role == "ffn_up":
        return (768, 3_072, 576)
    return (3_072, 768, 576)


def canonical_tiling_key(panel: str, seed: int, depth: int, role: str) -> str:
    return f"panel={panel}|seed={seed}|depth={depth:02d}|role={role}"


def derive_frozen_common_tiling(seed: int, d_in: int, d_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Independently regenerate the registered Torch-2.10 CPU partition."""

    require(seed in TILING_SEEDS, f"unregistered common-tiling seed: {seed}")
    require((d_in, d_out) in {_matrix_shape(role)[:2] for role in ROLES}, f"unregistered tiling shape: {d_in}x{d_out}")
    import torch

    identity = f"global_context_common_tiling_v2|shape={d_in}x{d_out}"
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, identity))
    patch_groups = torch.randperm(d_in // 16, generator=generator).reshape(-1, 4)
    offsets = torch.arange(16, dtype=torch.int64)
    rows = torch.stack(
        [
            (group[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values
            for group in patch_groups
        ]
    ).to(torch.int64).contiguous()
    cols = (
        torch.randperm(d_out, generator=generator)
        .reshape(-1, 64)
        .sort(dim=1)
        .values.to(torch.int64)
        .contiguous()
    )
    return rows.numpy(), cols.numpy()


def validate_design_contract_binding(
    design_path: Path,
    external_contract_path: Path,
    input_dir: Path,
) -> dict[str, Any]:
    """Fail closed on the frozen design and the three copied/bound contracts.

    The external contract is intentionally allowed to carry additional audit
    detail, but all identity-bearing fields below are mandatory and must agree
    byte-for-byte across the external copy, runner copy, binding, metadata and
    resolved config. The exact top-level contract schema is itself committed in
    ``contract_schema_keys`` and checked against the payload.
    """

    design = design_path.resolve(strict=True)
    actual_design_sha = sha256_file(design)
    require(actual_design_sha == DESIGN_SHA256, f"frozen design SHA mismatch: {actual_design_sha}")
    external_path = external_contract_path.resolve(strict=True)
    copied_path = (input_dir / "preexecution_contract.json").resolve(strict=True)
    binding_path = (input_dir / "preexecution_binding.json").resolve(strict=True)
    external_bytes = external_path.read_bytes()
    require(copied_path.read_bytes() == external_bytes, "runner contract is not byte-identical to external contract")
    contract_sha = hashlib.sha256(external_bytes).hexdigest()
    contract = json.loads(external_bytes)
    require(isinstance(contract, dict), "external contract must be a mapping")
    schema_keys = contract.get("contract_schema_keys")
    require(isinstance(schema_keys, list) and all(isinstance(key, str) for key in schema_keys), "contract lacks exact contract_schema_keys")
    require(schema_keys == list(CONTRACT_SCHEMA_KEYS) and set(contract) == set(CONTRACT_SCHEMA_KEYS), "external contract top-level schema differs from exact frozen schema")
    require(contract.get("schema_version") == "global_context_latent_geometry_preexecution_contract_v1", "contract schema version mismatch")
    require(contract.get("frozen_design_sha256") == DESIGN_SHA256, "contract design hash mismatch")
    require(Path(str(contract.get("frozen_design_path"))).resolve(strict=True) == design, "contract design path mismatch")
    runtime_audit = validate_analysis_runtime(contract.get("runtime"))
    require(contract.get("analyzer_sha256") == sha256_file(Path(__file__).resolve(strict=True)), "contract analyzer hash mismatch")
    runner_path = Path(str(contract.get("runner_path"))).resolve(strict=True)
    analyzer_path = Path(str(contract.get("analyzer_path"))).resolve(strict=True)
    require(runner_path == RUNNER_PATH.resolve(strict=True), "contract runner path is not the frozen runner entrypoint")
    require(sha256_file(runner_path) == contract.get("runner_sha256"), "contract runner file hash mismatch")
    require(analyzer_path == Path(__file__).resolve(strict=True) and sha256_file(analyzer_path) == contract.get("analyzer_sha256"), "contract analyzer file identity mismatch")
    scripts = contract.get("scripts")
    require(isinstance(scripts, dict) and set(scripts) == {"runner", "analyzer", "panel_builder"}, "contract script snapshot grid mismatch")
    for name, script in scripts.items():
        exact_keys(script, ("path", "sha256", "bytes"), f"contract script {name}")
        script_path = Path(str(script["path"])).resolve(strict=True)
        require(sha256_file(script_path) == script["sha256"] and script_path.stat().st_size == script["bytes"], f"contract script drift: {name}")
    expected_script_paths = {
        "runner": RUNNER_PATH.resolve(strict=True),
        "analyzer": Path(__file__).resolve(strict=True),
        "panel_builder": PANEL_BUILDER_PATH.resolve(strict=True),
    }
    require(
        {name: Path(str(row["path"])).resolve(strict=True) for name, row in scripts.items()}
        == expected_script_paths,
        "contract script paths differ from the frozen three-entrypoint grid",
    )
    require(scripts["runner"]["sha256"] == contract["runner_sha256"] and scripts["analyzer"]["sha256"] == contract["analyzer_sha256"], "contract top-level/script SHA mismatch")
    seal = contract.get("source_only_seal")
    require(isinstance(seal, dict) and seal.get("installed_before_scientific_input_read") is True, "contract source-only seal missing/pre-read false")
    require(all(int(seal.get(key, -1)) == 0 for key in ("target_access_events", "network_connections", "subprocess_launches")), "contract source-only seal recorded forbidden events")
    require(contract.get("expected_counts") == EXPECTED_RUNNER_COUNTS, "contract expected-count grid mismatch")
    frozen_grids = contract.get("frozen_grids")
    require(isinstance(frozen_grids, dict), "contract lacks frozen_grids")
    exact_keys(
        frozen_grids,
        (
            "panel_order", "role_order", "depths", "common_tiling_seeds",
            "code_representation_order", "representation_order", "untrained_seeds",
            "countsketch_seeds", "role_shapes",
        ),
        "contract frozen grids",
    )
    require(frozen_grids.get("panel_order") == list(PANELS), "contract panel order mismatch")
    require(frozen_grids.get("role_order") == list(ROLES), "contract role order mismatch")
    require(frozen_grids.get("depths") == list(DEPTHS), "contract depth grid mismatch")
    require(frozen_grids.get("common_tiling_seeds") == list(TILING_SEEDS), "contract tiling seeds mismatch")
    require(frozen_grids.get("code_representation_order") == list(CODE_REPRESENTATIONS), "contract code representation order mismatch")
    require(frozen_grids.get("representation_order") == list(REPRESENTATIONS), "contract representation order mismatch")
    require(frozen_grids.get("untrained_seeds") == list(UNTRAINED_SEEDS), "contract untrained seed mismatch")
    require(frozen_grids.get("countsketch_seeds") == list(COUNTSKETCH_SEEDS), "contract CountSketch seed mismatch")
    require(
        frozen_grids.get("role_shapes")
        == {role: list(_matrix_shape(role)[:2]) for role in ROLES},
        "contract role-shape grid mismatch",
    )
    artifact_contract = contract.get("artifact_contract")
    require(
        artifact_contract
        == {
            "self_excluded_files": list(RUNNER_FILES),
            "declared_file_count": 21,
            "file_count_including_manifest": 22,
            "artifact_manifest_self_excluded": True,
            "runner_and_analyzer_outputs_disjoint": True,
            "no_symlinks": True,
        },
        "contract runner artifact grid mismatch",
    )
    execution = contract.get("execution_config")
    require(isinstance(execution, dict), "contract lacks execution_config")
    require(execution == expected_execution_config(), "contract exact execution_config mismatch")
    static_call_audit = validate_static_call_graph_payload(
        contract.get("static_call_graph_audit"), str(contract["runner_sha256"])
    )
    require(contract.get("dataflow_contract") == expected_dataflow_contract(), "contract exact dataflow contract mismatch")
    contracted_input = validate_contracted_input_audit(contract.get("input_audit"))
    analysis_contract = contract.get("analysis_contract")
    require(isinstance(analysis_contract, dict), "contract lacks analysis_contract")
    exact_keys(analysis_contract, ("variance_decomposition",), "analysis contract")
    require(
        analysis_contract.get("variance_decomposition") == VARIANCE_DECOMPOSITION_CONTRACT,
        "contract variance-decomposition formula mismatch",
    )

    binding = json_read(binding_path)
    require(isinstance(binding, dict), "preexecution binding must be a mapping")
    exact_keys(
        binding,
        (
            "schema_version", "external_contract_path", "external_contract_sha256",
            "copied_contract_path", "copied_contract_sha256", "frozen_design_path",
            "frozen_design_sha256", "runner_path", "runner_sha256", "analyzer_path",
            "analyzer_sha256", "runner_output_path", "analyzer_output_path",
            "exact_contract_verified_before_model_import",
        ),
        "preexecution binding",
    )
    require(binding["schema_version"] == "global_context_preexecution_binding_v1", "binding schema version mismatch")
    expected_binding = {
        "external_contract_path": str(external_path),
        "external_contract_sha256": contract_sha,
        "copied_contract_path": str(copied_path),
        "copied_contract_sha256": contract_sha,
        "frozen_design_path": str(design),
        "frozen_design_sha256": DESIGN_SHA256,
        "runner_path": str(Path(str(contract["runner_path"])).resolve(strict=True)),
        "runner_sha256": str(contract["runner_sha256"]),
        "analyzer_path": str(Path(__file__).resolve(strict=True)),
        "analyzer_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "runner_output_path": str(input_dir),
        "analyzer_output_path": str(contract["execution_config"]["analyzer_output_dir"]),
        "exact_contract_verified_before_model_import": True,
    }
    for key, expected in expected_binding.items():
        require(binding.get(key) == expected, f"binding identity mismatch for {key}")

    metadata = json_read(input_dir / "runner_metadata.json")
    resolved = json_read(input_dir / "resolved_config.json")
    for label, payload in (("runner_metadata", metadata), ("resolved_config", resolved)):
        require(payload.get("frozen_design_sha256") == DESIGN_SHA256, f"{label} design hash mismatch")
        require(payload.get("external_contract_sha256") == contract_sha, f"{label} contract hash mismatch")
        require(payload.get("runner_sha256") == contract["runner_sha256"], f"{label} runner hash mismatch")
        require(payload.get("analyzer_sha256") == contract["analyzer_sha256"], f"{label} analyzer hash mismatch")
    return {
        "pass": True,
        "design_path": str(design),
        "design_sha256": actual_design_sha,
        "external_contract_path": str(external_path),
        "external_contract_sha256": contract_sha,
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "tiling_identity_scope": contract.get("tiling_identity_scope"),
        "analysis_runtime": runtime_audit,
        "contract_semantic_audit": {
            "pass": True,
            "execution_config_exact": True,
            "static_call_graph": static_call_audit,
            "source_template_cells": len(contracted_input["cell_template_hashes"]),
            "canonical_w_rows": len(contracted_input["w_grid"]),
            "zero_w_links_checked": contracted_input["zero_weight_sanity"]["contracted_w_links_checked"],
        },
        "contract": contract,
    }


def validate_runner_json_audits(input_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    contracted_input = validate_contracted_input_audit(contract["input_audit"])
    expected_w_grid = contracted_input["w_grid"]
    cell_template_hashes = contracted_input["cell_template_hashes"]
    runner_input = json_read(input_dir / "input_audit.json")
    expected_runner_input = {
        **contract["input_audit"],
        "execute_time_w_only_exact_contract_match": True,
        "execute_time_template_exact_contract_match": True,
        "model_modules_loaded_at_materialization_completion": [],
    }
    require(runner_input == expected_runner_input, "execute-time input_audit is not the exact contracted payload plus frozen flags")
    source_template = json_read(input_dir / "source_template_audit.json")
    require(source_template == contract["input_audit"]["source_template_audit"], "execute-time source-template audit differs from contract")
    validate_source_template_payload(source_template)
    zero_payload = json_read(input_dir / "zero_weight_sanity.json")
    require(zero_payload == contract["input_audit"]["zero_weight_sanity_preflight"], "execute-time zero-W audit differs from contract")
    zero_weight = validate_zero_weight_sanity_payload(zero_payload, expected_w_grid)

    seal = json_read(input_dir / "target_access_seal.json")
    require(seal.get("target_access_events") == 0, "runner reports target access")
    require(seal.get("network_connections") == 0, "runner reports network access")
    require(seal.get("subprocess_launches") == 0, "runner reports subprocess launch")
    require(seal.get("pass") is True, "runner source-only seal failed")

    model = json_read(input_dir / "model_contract.json")
    exact_keys(
        model,
        (
            "schema_version", "learned", "untrained", "local_module_closure_after_import",
            "global_seed", "untrained_seed_order", "decoder_calls",
            "distribution_encoder_calls", "three_untrained_state_dict_hashes_distinct",
            "all_untrained_parameter_grids_match_learned",
            "all_untrained_checkpoint_tensor_copy_counts_zero", "pass",
        ),
        "model contract",
    )
    require(model.get("schema_version") == "global_context_model_contract_v1", "model contract schema mismatch")
    require(model.get("pass") is True, "runner model contract failed")
    require(model.get("decoder_calls") == 0, "runner model contract reports decoder call")
    require(model.get("distribution_encoder_calls") == 0, "runner reports distribution encoder call")
    require(model.get("global_seed") == BASE_SEED and model.get("untrained_seed_order") == list(UNTRAINED_SEEDS), "model seed grid mismatch")
    require(
        model.get("three_untrained_state_dict_hashes_distinct") is True
        and model.get("all_untrained_parameter_grids_match_learned") is True
        and model.get("all_untrained_checkpoint_tensor_copy_counts_zero") is True,
        "model aggregate untrained checks failed",
    )
    learned = model.get("learned")
    require(isinstance(learned, dict) and learned.get("pass") is True, "learned model sub-contract failed")
    require(learned.get("kind") == "learned_strict_checkpoint" and learned.get("eval_mode") is True and learned.get("checkpoint_tensors_loaded") is True, "learned model mode/load contract mismatch")
    required_learned = {
        "checkpoint_sha256": "d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00",
        "checkpoint_step": 480000,
        "strict_load": True,
        "rope_2d_coord_kind": "raw",
        "use_latent_sampling": False,
        "use_encoder_mu_head": False,
        "disable_z_shortcut": True,
        "flat_lat_dim": 512,
        "patch_size": 16,
        "locked_tile_shape": [64, 64],
        "amp_enabled": True,
        "amp_dtype": "torch.bfloat16",
    }
    require(learned.get("required_contract") == required_learned, "learned exact model contract mismatch")
    untrained = model.get("untrained")
    require(isinstance(untrained, dict) and list(untrained) == [str(seed) for seed in UNTRAINED_SEEDS], "untrained fingerprint grid mismatch")
    untrained_rows = [untrained[str(seed)] for seed in UNTRAINED_SEEDS]
    require([int(row["seed"]) for row in untrained_rows] == list(UNTRAINED_SEEDS), "untrained seed order mismatch")
    fingerprints = [str(row["fingerprint"]["state_dict_sha256"]) for row in untrained_rows]
    require(all(HEX64.fullmatch(value) for value in fingerprints) and len(set(fingerprints)) == 3, "untrained fingerprints invalid/nonunique")
    require(all(row.get("parameter_name_shape_dtype_grid_matches_learned") is True for row in untrained_rows), "untrained parameter grid mismatch")
    require(all(row.get("checkpoint_tensor_copy_count") == 0 and row.get("checkpoint_state_dict_loaded") is False for row in untrained_rows), "untrained model copied checkpoint tensors")
    require(all(row.get("eval_mode") is True and row.get("state_dict_differs_from_learned") is True for row in untrained_rows), "untrained state/mode contract mismatch")

    preflight = json_read(input_dir / "known_source_numeric_preflight.json")
    exact_keys(
        preflight,
        (
            "schema_version", "pass", "performed_before_new_code_entries",
            "source_activation_passed_to_weight_ae", "source_activation_consumers",
            "metadata_label_model_consumers", "decoder_calls", "distribution_encoder_calls",
            "source_panel", "depth", "role", "old_tiling_seed", "old_partition_sha256",
            "cached_code_key", "execution_batch", "expected_cached_slice_sha256",
            "current_code_sha256", "bit_exact", "max_abs", "relative_l2", "cosine",
            "comparison_method",
        ),
        "known-source numeric preflight",
    )
    require(preflight.get("schema_version") == "global_context_known_source_numeric_preflight_v1", "known-source preflight schema mismatch")
    require(preflight.get("pass") is True, "known-source numeric preflight failed")
    require(preflight.get("bit_exact") is True and preflight.get("performed_before_new_code_entries") is True, "known-source preflight order/exactness failed")
    require(float(preflight.get("max_abs", math.inf)) == 0.0, "known-source numeric preflight not bit exact")
    require(float(preflight.get("relative_l2", math.inf)) == 0.0, "known-source numeric relative L2 nonzero")
    require(float(preflight.get("cosine", -math.inf)) == 1.0, "known-source numeric cosine not exact")
    require(preflight.get("decoder_calls") == 0 and preflight.get("distribution_encoder_calls") == 0, "known-source preflight reports forbidden call")
    require(preflight.get("source_activation_passed_to_weight_ae") is False and preflight.get("source_activation_consumers") == 0 and preflight.get("metadata_label_model_consumers") == 0, "known-source preflight consumed activation/metadata")

    dataflow = json_read(input_dir / "dataflow_audit.json")
    exact_keys(
        dataflow,
        (
            "schema_version", "static_call_graph", "local_module_closure", "w_only_lifecycle",
            "known_source_preflight_call", "runtime_calls", "total_runtime_call_count",
            "primary_global_summary", "zero_summary", "learned_cell_summary", "decoder_calls",
            "distribution_encoder_calls", "panel_activation_consumers",
            "old_source_or_candidate_cache_feature_consumers",
            "old_source_or_candidate_cache_use", "pass",
        ),
        "dataflow audit",
    )
    require(dataflow.get("schema_version") == "global_context_dataflow_audit_v1", "dataflow schema version mismatch")
    require(dataflow.get("pass") is True, "dataflow audit failed")
    static_graph = dataflow.get("static_call_graph", {})
    require(static_graph == contract["static_call_graph_audit"], "runtime static-call audit differs from external contract")
    exact_keys(
        static_graph,
        (
            "schema_version", "script_sha256", "encoder_function", "formal_arguments",
            "forbidden_scientific_argument_or_local_names",
            "forbidden_decoder_or_distribution_attribute_calls",
            "encode_z_dec_subscript_call_count",
            "model_receives_only_weight_batch_and_expanded_fixed_condition", "pass",
        ),
        "static call graph",
    )
    require(static_graph.get("schema_version") == "global_context_static_call_graph_audit_v1", "static call graph schema mismatch")
    require(HEX64.fullmatch(str(static_graph.get("script_sha256"))) is not None, "static call graph script SHA malformed")
    require(static_graph.get("encoder_function") == "encode_weight_tiles", "static call graph encoder function mismatch")
    require(static_graph.get("formal_arguments") == ["model", "W_tiles", "fixed_template"], "encoding formal argument contract mismatch")
    require(static_graph.get("forbidden_scientific_argument_or_local_names") == [] and static_graph.get("forbidden_decoder_or_distribution_attribute_calls") == [], "static encoding path exposes forbidden names/calls")
    require(static_graph.get("encode_z_dec_subscript_call_count") == 1 and static_graph.get("model_receives_only_weight_batch_and_expanded_fixed_condition") is True and static_graph.get("pass") is True, "static call graph encode invocation mismatch")
    closure = dataflow.get("local_module_closure", {})
    exact_keys(closure, ("pass", "allowed_file_count", "loaded_module_count", "loaded_modules"), "local module closure")
    require(closure.get("pass") is True and int(closure.get("allowed_file_count", 0)) >= int(closure.get("loaded_module_count", -1)) >= 1, "local module closure count/pass mismatch")
    loaded_modules = closure.get("loaded_modules")
    require(isinstance(loaded_modules, list) and len(loaded_modules) == closure["loaded_module_count"], "local module closure row count mismatch")
    require(all(set(row) == {"module", "path"} for row in loaded_modules), "local module closure row schema mismatch")
    lifecycle = dataflow.get("w_only_lifecycle", {})
    exact_keys(
        lifecycle,
        (
            "materialization_completed_before_model_import",
            "activation_tensors_retained_at_model_import",
            "activation_references_retained_at_model_import", "exact_allowed_w_only_fields",
            "allowed_metadata_fields",
            "activation_or_sample_target_payload_references_after_materialization",
            "encode_runtime_exact_keys",
            "encode_runtime_contains_panel_role_depth_dataset_activation_or_label",
        ),
        "W-only lifecycle",
    )
    require(lifecycle.get("materialization_completed_before_model_import") is True, "W-only materialization did not precede model import")
    require(lifecycle.get("activation_tensors_retained_at_model_import") == 0, "activation tensors retained at model import")
    require(lifecycle.get("activation_references_retained_at_model_import") == 0, "activation references retained at model import")
    require(
        lifecycle.get("exact_allowed_w_only_fields")
        == [
            "W", "depth", "panel_id", "role", "tilings",
            "weight_shape_bytes_sha256", "weight_tensor_sha256",
        ],
        "W-only field allowlist mismatch",
    )
    require(lifecycle.get("allowed_metadata_fields") == ["panel_id", "depth", "role"], "W-only metadata allowlist mismatch")
    require(lifecycle.get("activation_or_sample_target_payload_references_after_materialization") == 0, "activation/sample-target payload survived W-only materialization")
    require(
        lifecycle.get("encode_runtime_exact_keys")
        == [
            "amp_dtype", "amp_enabled", "batch_size", "call_ordinal", "call_records",
            "device", "factorial", "gate", "log_every_batches", "logger",
        ],
        "encode runtime exact-key contract mismatch",
    )
    require(lifecycle.get("encode_runtime_contains_panel_role_depth_dataset_activation_or_label") is False, "encode runtime accepts forbidden metadata/activation/label")
    primary = dataflow.get("primary_global_summary", {})
    exact_keys(
        primary,
        (
            "representations", "runtime_call_count", "unique_c_var_hashes",
            "unique_c_patch_hashes", "activation_consumers", "metadata_label_consumers",
            "condition_selector_metadata_fields", "decision_eligible",
        ),
        "primary-global summary",
    )
    require(primary.get("runtime_call_count") == 4 * 432, "primary-global call count mismatch")
    require(
        primary.get("representations")
        == ["learned_global", *UNTRAINED_REPRESENTATIONS],
        "primary-global representation grid mismatch",
    )
    require(primary.get("unique_c_var_hashes") == [GLOBAL_C_VAR_SHA256], "primary global c_var is not constant")
    require(primary.get("unique_c_patch_hashes") == [GLOBAL_C_PATCH_SHA256], "primary global c_patch is not constant")
    require(primary.get("activation_consumers") == 0 and primary.get("metadata_label_consumers") == 0, "primary global model path consumed metadata/activations")
    require(primary.get("condition_selector_metadata_fields") == [], "primary global condition selected by metadata")
    require(primary.get("decision_eligible") is True, "primary global path not marked decision eligible")
    zero = dataflow.get("zero_summary", {})
    exact_keys(
        zero,
        (
            "representation", "runtime_call_count", "unique_c_var_hashes",
            "unique_c_patch_hashes", "activation_consumers", "metadata_label_consumers",
            "condition_selector_metadata_fields", "secondary_non_gating",
        ),
        "zero summary",
    )
    require(zero.get("runtime_call_count") == 432 and zero.get("representation") == "learned_zero", "zero call grid mismatch")
    require(zero.get("unique_c_var_hashes") == [ZERO_C_SHA256] and zero.get("unique_c_patch_hashes") == [ZERO_C_SHA256], "zero condition hash mismatch")
    require(zero.get("activation_consumers") == 0 and zero.get("metadata_label_consumers") == 0, "zero path consumed metadata/activations")
    require(zero.get("condition_selector_metadata_fields") == [], "zero condition selected by metadata")
    require(zero.get("secondary_non_gating") is True, "zero path not marked secondary/non-gating")
    cell = dataflow.get("learned_cell_summary", {})
    exact_keys(
        cell,
        (
            "representation", "runtime_call_count", "segregated_call_record_namespace",
            "condition_selection_source", "allowed_selector_metadata_fields",
            "primary_dataflow_claim_eligible", "level_a_or_b_decision_eligible",
            "activation_consumers", "metadata_label_consumers",
        ),
        "learned-cell summary",
    )
    require(cell.get("runtime_call_count") == 432 and cell.get("representation") == "learned_cell", "learned-cell call grid mismatch")
    require(cell.get("segregated_call_record_namespace") is True, "learned-cell calls not segregated")
    require(cell.get("condition_selection_source") == "immutable_source_cell_mean_by_role_depth", "learned-cell selector source mismatch")
    require(cell.get("allowed_selector_metadata_fields") == ["role", "depth"], "learned-cell selector allowlist mismatch")
    require(cell.get("activation_consumers") == 0 and cell.get("metadata_label_consumers") == 0, "learned-cell path consumed activations/sample labels")
    require(cell.get("primary_dataflow_claim_eligible") is False and cell.get("level_a_or_b_decision_eligible") is False, "learned-cell path incorrectly decision eligible")
    calls = dataflow.get("runtime_calls")
    require(isinstance(calls, list) and len(calls) == 2_592, "runtime call record total mismatch")
    runtime_call_audit = validate_runtime_call_records(calls, cell_template_hashes)
    preflight_call = dataflow.get("known_source_preflight_call")
    require(isinstance(preflight_call, dict), "known-source preflight call record missing")
    exact_keys(
        preflight_call,
        (
            "call_ordinal", "tile_rows", "code_rows", "code_width", "template_hashes",
            "batches", "activation_consumers", "metadata_label_consumers",
            "distribution_encoder_calls", "decoder_calls", "representation",
            "panel_id_metadata_write_only", "role_metadata_write_only",
            "depth_metadata_write_only", "tiling_seed_metadata_write_only",
            "condition_kind", "call_class", "outer_condition_selector_metadata",
            "decision_eligible",
        ),
        "known-source preflight runtime call",
    )
    require(preflight_call.get("call_ordinal") == 1 and preflight_call.get("tile_rows") == 64 and preflight_call.get("code_rows") == 64 and preflight_call.get("code_width") == 512, "preflight call numeric grid mismatch")
    require(preflight_call.get("representation") == "known_source_numeric_preflight", "preflight call representation mismatch")
    require(preflight_call.get("call_class") == "preflight_not_a_new_code_entry", "preflight call class mismatch")
    require(preflight_call.get("condition_kind") == "source_cell_mean", "preflight call condition mismatch")
    require(preflight_call.get("outer_condition_selector_metadata") == ["fixed_preflight_identity"], "preflight selector mismatch")
    require(preflight_call.get("decision_eligible") is False, "preflight call incorrectly decision eligible")
    require(preflight_call.get("activation_consumers") == 0 and preflight_call.get("metadata_label_consumers") == 0 and preflight_call.get("decoder_calls") == 0 and preflight_call.get("distribution_encoder_calls") == 0, "preflight call consumed forbidden path")
    require(dataflow.get("total_runtime_call_count") == 2_592, "dataflow total runtime call count mismatch")
    require(dataflow.get("decoder_calls") == 0 and dataflow.get("distribution_encoder_calls") == 0, "forbidden model forward in dataflow")
    require(dataflow.get("panel_activation_consumers") == 0, "dataflow panel activation consumer found")
    require(dataflow.get("old_source_or_candidate_cache_feature_consumers") == 0, "old cache feature consumer found")
    require(
        dataflow.get("old_source_or_candidate_cache_use")
        == "known-source hash preflight and provenance audit only",
        "old source/candidate cache use exceeded preflight/provenance scope",
    )

    source_cell_hashes = cell_template_hashes[(0, "attn_query")]
    require(
        preflight_call.get("template_hashes")
        == {
            "c_var": source_cell_hashes["c_var"],
            "c_patch": source_cell_hashes["c_patch"],
        },
        "preflight runtime/source-cell template hash mismatch",
    )
    require(isinstance(preflight_call.get("batches"), list) and len(preflight_call["batches"]) == 1, "preflight runtime batch grid mismatch")
    immutability = json_read(input_dir / "input_immutability_recheck.json")
    exact_keys(
        immutability,
        (
            "schema_version", "resource_snapshot_exact", "script_snapshot_exact",
            "dependency_snapshot_exact", "in_memory_w_grid_sha256",
            "expected_w_grid_sha256", "in_memory_w_grid_exact",
            "global_template_hashes", "global_template_hashes_exact",
            "source_only_seal_clean", "pass",
        ),
        "input immutability recheck",
    )
    require(immutability["schema_version"] == "global_context_input_immutability_recheck_v1", "immutability schema mismatch")
    require(
        immutability["in_memory_w_grid_sha256"]
        == immutability["expected_w_grid_sha256"]
        == contracted_input["w_grid_sha256"],
        "immutability W-grid SHA does not bind contracted 216-W map",
    )
    require(
        immutability["global_template_hashes"]
        == {"c_var": GLOBAL_C_VAR_SHA256, "c_patch": GLOBAL_C_PATCH_SHA256},
        "immutability global template hashes mismatch",
    )
    for key in (
        "resource_snapshot_exact", "script_snapshot_exact", "dependency_snapshot_exact",
        "in_memory_w_grid_exact", "global_template_hashes_exact", "source_only_seal_clean", "pass",
    ):
        require(immutability[key] is True, f"immutability recheck failed: {key}")
    return {
        "pass": True,
        "source_seal": seal,
        "untrained_fingerprints": fingerprints,
        "known_source_preflight": preflight,
        "primary_global_summary": primary,
        "zero_summary": zero,
        "learned_cell_summary": cell,
        "runtime_call_audit": runtime_call_audit,
        "w_only_lifecycle": lifecycle,
        "zero_weight_sanity": zero_weight,
        "contract_semantic_binding": {
            "pass": True,
            "input_audit_exact": True,
            "source_template_exact": True,
            "zero_weight_exact": True,
            "cell_template_hashes_checked": 72,
            "runtime_cell_calls_checked": 432,
            "runtime_batch_template_hashes_checked": runtime_call_audit["batches"],
            "canonical_w_rows_checked": 216,
            "immutability_w_grid_exact": True,
            "static_call_graph_exact": True,
        },
        "_expected_w_grid": expected_w_grid,
        "_cell_template_hashes": cell_template_hashes,
    }


def validate_runtime_call_records(
    calls: Sequence[Mapping[str, Any]],
    cell_template_hashes: Mapping[tuple[int, str], Mapping[str, str]],
) -> dict[str, Any]:
    call_keys = (
        "call_ordinal", "tile_rows", "code_rows", "code_width", "template_hashes",
        "batches", "activation_consumers", "metadata_label_consumers",
        "distribution_encoder_calls", "decoder_calls", "representation",
        "panel_id_metadata_write_only", "role_metadata_write_only",
        "depth_metadata_write_only", "tiling_seed_metadata_write_only",
        "condition_kind", "call_class", "outer_condition_selector_metadata",
        "decision_eligible",
    )
    batch_keys = (
        "batch_index", "batch_rows", "c_var_template_sha256",
        "c_patch_template_sha256", "activation_consumers", "metadata_label_consumers",
        "distribution_encoder_calls", "decoder_calls",
    )
    expected_grid = [
        (rep, panel, seed, depth, role)
        for rep in CODE_REPRESENTATIONS
        for panel, seed, _, depth, role in canonical_rows()
    ]
    actual_grid: list[tuple[str, str, int, int, str]] = []
    ordinals: list[int] = []
    batch_count = 0
    for row in calls:
        exact_keys(row, call_keys, "runtime call record")
        rep = str(row["representation"])
        panel = str(row["panel_id_metadata_write_only"])
        seed = int(row["tiling_seed_metadata_write_only"])
        depth = int(row["depth_metadata_write_only"])
        role = str(row["role_metadata_write_only"])
        actual_grid.append((rep, panel, seed, depth, role))
        ordinals.append(int(row["call_ordinal"]))
        _, _, expected_tiles = _matrix_shape(role)
        require(int(row["tile_rows"]) == int(row["code_rows"]) == expected_tiles, "runtime tile/code row count mismatch")
        require(int(row["code_width"]) == 512, "runtime code width mismatch")
        require(
            row["activation_consumers"] == 0
            and row["metadata_label_consumers"] == 0
            and row["distribution_encoder_calls"] == 0
            and row["decoder_calls"] == 0,
            "runtime call consumed forbidden input/path",
        )
        template_hashes = row["template_hashes"]
        exact_keys(template_hashes, ("c_var", "c_patch"), "runtime call template hashes")
        require(all(HEX64.fullmatch(str(value)) for value in template_hashes.values()), "runtime template hash malformed")
        if rep == "learned_cell":
            validate_cell_template_hash_pair(
                depth,
                role,
                str(template_hashes["c_var"]),
                str(template_hashes["c_patch"]),
                cell_template_hashes,
                label="learned-cell runtime hash",
            )
            require(row["condition_kind"] == "source_cell_mean", "learned-cell runtime condition mismatch")
            require(row["call_class"] == "segregated_non_gating_cell_reference", "learned-cell call class mismatch")
            require(row["outer_condition_selector_metadata"] == ["role", "depth"], "learned-cell selector mismatch")
            require(row["decision_eligible"] is False, "learned-cell runtime incorrectly decision eligible")
        elif rep == "learned_zero":
            require(row["condition_kind"] == "exact_zero", "zero runtime condition mismatch")
            require(template_hashes == {"c_var": ZERO_C_SHA256, "c_patch": ZERO_C_SHA256}, "zero runtime template mismatch")
            require(row["call_class"] == "primary_fixed_condition" and row["outer_condition_selector_metadata"] == [], "zero runtime selector/class mismatch")
            require(row["decision_eligible"] is False, "zero runtime incorrectly decision eligible")
        else:
            require(row["condition_kind"] == "source_global", f"global runtime condition mismatch: {rep}")
            require(template_hashes == {"c_var": GLOBAL_C_VAR_SHA256, "c_patch": GLOBAL_C_PATCH_SHA256}, f"global runtime template mismatch: {rep}")
            require(row["call_class"] == "primary_fixed_condition" and row["outer_condition_selector_metadata"] == [], f"global runtime selector/class mismatch: {rep}")
            require(row["decision_eligible"] is True, f"decision path not marked eligible: {rep}")
        batches = row["batches"]
        require(isinstance(batches, list) and len(batches) == math.ceil(expected_tiles / 64), "runtime batch grid mismatch")
        require(sum(int(batch["batch_rows"]) for batch in batches) == expected_tiles, "runtime batch rows do not sum to entry tiles")
        for batch_index, batch in enumerate(batches, start=1):
            exact_keys(batch, batch_keys, "runtime batch record")
            require(int(batch["batch_index"]) == batch_index and 0 < int(batch["batch_rows"]) <= 64, "runtime batch index/size mismatch")
            require(batch["c_var_template_sha256"] == template_hashes["c_var"] and batch["c_patch_template_sha256"] == template_hashes["c_patch"], "runtime batch template drift")
            require(
                batch["activation_consumers"] == 0
                and batch["metadata_label_consumers"] == 0
                and batch["distribution_encoder_calls"] == 0
                and batch["decoder_calls"] == 0,
                "runtime batch consumed forbidden input/path",
            )
            batch_count += 1
    require(actual_grid == expected_grid, "runtime call representation/panel/tiling/depth/role grid mismatch")
    require(ordinals == list(range(2, 2_594)), "runtime call ordinal sequence mismatch")
    return {
        "pass": True,
        "records": len(calls),
        "batches": batch_count,
        "exact_call_keyset": list(call_keys),
        "exact_batch_keyset": list(batch_keys),
        "primary_fixed_condition_metadata_selectors": [],
        "learned_cell_segregated": True,
        "learned_zero_non_gating": True,
    }


def validate_zero_weight_sanity_payload(
    payload: Mapping[str, Any],
    expected_w_grid: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the exact 216-W algebra and bind every row to contracted W."""

    exact_keys(payload, ("schema_version", "matrix_count", "model_forward_calls", "entries", "summary"), "zero_weight_sanity")
    require(payload["schema_version"] == "global_context_zero_weight_sanity_v1", "zero-weight schema mismatch")
    require(payload["matrix_count"] == 216 and payload["model_forward_calls"] == 0, "zero-weight header mismatch")
    entries = payload["entries"]
    require(isinstance(entries, list) and len(entries) == 216, "zero-weight entry count mismatch")
    expected_grid = [(panel, depth, role) for panel in PANELS for depth in DEPTHS for role in ROLES]
    actual_grid: list[tuple[str, int, str]] = []
    for row in entries:
        exact_keys(
            row,
            (
                "panel_id", "depth", "role", "d_in", "d_out",
                "weight_shape_bytes_sha256", "weight_tensor_sha256", "numel",
                "finite_count", "nonzero_count", "sum_squared", "frobenius_norm",
                "rms", "literal_zero_relative_squared_error",
            ),
            "zero-weight entry",
        )
        identity = (str(row["panel_id"]), int(row["depth"]), str(row["role"]))
        actual_grid.append(identity)
        d_in, d_out, _ = _matrix_shape(row["role"])
        numel = d_in * d_out
        require((int(row["d_in"]), int(row["d_out"]), int(row["numel"])) == (d_in, d_out, numel), "zero-weight shape/numel mismatch")
        require(int(row["finite_count"]) == numel and 0 < int(row["nonzero_count"]) <= numel, "zero-weight finite/nonzero count mismatch")
        require(HEX64.fullmatch(str(row["weight_shape_bytes_sha256"])) is not None and HEX64.fullmatch(str(row["weight_tensor_sha256"])) is not None, "zero-weight hash malformed")
        if expected_w_grid is not None:
            require(identity in expected_w_grid, f"zero-weight row absent from contracted W grid: {identity}")
            contracted = expected_w_grid[identity]
            require(list(contracted["shape"]) == [d_in, d_out], f"zero-weight/contract W shape mismatch: {identity}")
            require(
                row["weight_shape_bytes_sha256"] == contracted["weight_shape_bytes_sha256"],
                f"zero-weight/contract W shape-bytes hash mismatch: {identity}",
            )
            require(
                row["weight_tensor_sha256"] == contracted["weight_tensor_sha256"],
                f"zero-weight/contract W tensor hash mismatch: {identity}",
            )
        sum_squared = float(row["sum_squared"])
        frobenius = float(row["frobenius_norm"])
        rms = float(row["rms"])
        require(np.isfinite([sum_squared, frobenius, rms]).all() and sum_squared > 0 and frobenius > 0 and rms > 0, "zero-weight norm invalid")
        require(math.isclose(frobenius, math.sqrt(sum_squared), rel_tol=2e-15, abs_tol=0.0), "zero-weight Frobenius algebra mismatch")
        require(math.isclose(rms, math.sqrt(sum_squared / numel), rel_tol=2e-15, abs_tol=0.0), "zero-weight RMS algebra mismatch")
        require(type(row["literal_zero_relative_squared_error"]) is float and row["literal_zero_relative_squared_error"] == 1.0, "literal-zero ratio not exact FP64 1.0")
    require(actual_grid == expected_grid, "zero-weight canonical matrix grid mismatch")
    summary = payload["summary"]
    exact_keys(summary, ("all_finite", "all_have_nonzero_entry", "all_positive_sum_squared", "all_literal_zero_ratios_exactly_one", "pass"), "zero-weight summary")
    require(all(summary[key] is True for key in summary), "zero-weight summary failed")
    return {
        "pass": True,
        "matrix_count": 216,
        "model_forward_calls": 0,
        "canonical_grid": True,
        "contracted_w_links_checked": 216 if expected_w_grid is not None else 0,
        "literal_zero_ratios_exactly_one": True,
        "validity_only_excluded_from_scientific_features": True,
    }


def validate_zero_weight_sanity(
    path: Path,
    expected_w_grid: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return validate_zero_weight_sanity_payload(json_read(path), expected_w_grid)


def validate_contracted_input_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all contract-side scientific identities before artifact use."""

    exact_keys(
        payload,
        (
            "schema_version", "resources", "source_parent_audit", "old_reference_audit",
            "source_template_audit", "w_only_materialization",
            "zero_weight_sanity_preflight", "weight_ae_model_modules_loaded",
            "weight_ae_forward_count", "decoder_forward_count",
            "distribution_encoder_forward_count", "source_only_seal_clean", "pass",
        ),
        "contract input audit",
    )
    require(payload["schema_version"] == "global_context_input_audit_v1", "contract input-audit schema mismatch")
    require(payload["weight_ae_model_modules_loaded"] == [], "contract input audit loaded Weight-AE modules")
    require(
        payload["weight_ae_forward_count"] == 0
        and payload["decoder_forward_count"] == 0
        and payload["distribution_encoder_forward_count"] == 0,
        "contract input audit reports forbidden model forward",
    )
    require(payload["source_only_seal_clean"] is True and payload["pass"] is True, "contract input audit failed")
    resources = payload["resources"]
    require(isinstance(resources, Mapping), "contract resources must be a mapping")
    require("source_condition_templates" in resources, "contract source-template resource absent")
    source_resource = resources["source_condition_templates"]
    exact_keys(source_resource, ("path", "sha256", "bytes"), "source-template resource")
    require(source_resource["sha256"] == SOURCE_TEMPLATE_FILE_SHA256, "source-template resource SHA mismatch")
    require(
        isinstance(source_resource["bytes"], int)
        and not isinstance(source_resource["bytes"], bool)
        and source_resource["bytes"] > 0,
        "source-template resource size invalid",
    )
    cell_hashes = validate_source_template_payload(payload["source_template_audit"])
    w_grid = validate_contracted_w_grid(payload["w_only_materialization"])
    zero_audit = validate_zero_weight_sanity_payload(payload["zero_weight_sanity_preflight"], w_grid)
    return {
        "pass": True,
        "source_template_resource": dict(source_resource),
        "cell_template_hashes": cell_hashes,
        "w_grid": w_grid,
        "w_grid_sha256": payload["w_only_materialization"]["w_grid_sha256"],
        "zero_weight_sanity": zero_audit,
    }


def validate_tiling_grid(
    input_dir: Path,
    expected_w_grid: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(input_dir / "tiling_manifest.csv")
    exact_columns(frame, TILING_COLUMNS, "tiling_manifest.csv")
    require(len(frame) == 432, "tiling manifest row count mismatch")
    keys = list(frame.loc[:, ["panel_id", "tiling_seed", "tiling_index", "depth", "role"]].itertuples(index=False, name=None))
    require(keys == canonical_rows(), "tiling manifest canonical row order/grid mismatch")
    require(not frame.duplicated(["panel_id", "tiling_seed", "depth", "role"]).any(), "duplicate tiling rows")
    for column in (
        "weight_reassembly_bit_exact",
        "coordinate_reassembly_bit_exact",
        "shape_global_identity_bit_exact",
    ):
        require(_bool_series(frame[column], f"tiling {column}").all(), f"tiling validity flag failed: {column}")
    for row in frame.itertuples(index=False):
        d_in, d_out, n_tiles = _matrix_shape(row.role)
        require((int(row.d_in), int(row.d_out), int(row.num_tiles)) == (d_in, d_out, n_tiles), f"tiling shape mismatch: {row}")
        expected_tiling_index = TILING_SEEDS.index(int(row.tiling_seed)) + 1
        require(int(row.tiling_index) == expected_tiling_index, "tiling index/seed mismatch")
        require(row.matrix_key == f"depth={int(row.depth):02d}|role={row.role}", "tiling matrix key mismatch")
        require(int(row.row_groups) == d_in // 64 and int(row.col_groups) == d_out // 64, "tiling group count mismatch")
        require(int(row.row_groups) * int(row.col_groups) == n_tiles, "tiling group product mismatch")
        require(int(row.coverage_min) == 1 and int(row.coverage_max) == 1, "tiling coverage not exact once")
        for field in ("row_index_sha256", "column_index_sha256", "partition_sha256", "weight_shape_bytes_sha256"):
            require(HEX64.fullmatch(str(getattr(row, field))) is not None, f"bad tiling hash {field}")

        if expected_w_grid is not None:
            contracted = expected_w_grid[(row.panel_id, int(row.depth), row.role)]
            require(list(contracted["shape"]) == [d_in, d_out], "contracted W/tiling shape mismatch")
            require(row.weight_shape_bytes_sha256 == contracted["weight_shape_bytes_sha256"], "contracted W/tiling hash mismatch")

    groups = list(frame.groupby(["tiling_seed", "d_in", "d_out"], sort=False))
    require(len(groups) == 6, "common-tiling seed/shape group count mismatch")
    for _, group in groups:
        require(group["partition_sha256"].nunique() == 1, "shape-only partition differs across role/depth/panel")

    import torch

    payload = torch.load(input_dir / "tiling_indices.pt", map_location="cpu", weights_only=True)
    exact_keys(payload, ("schema_version", "entry_count", "seeds", "panel_order", "entries"), "tiling_indices.pt")
    require(payload["schema_version"] == "global_context_tiling_indices_v1", "tiling index schema mismatch")
    require(int(payload["entry_count"]) == 432 and tuple(payload["seeds"]) == TILING_SEEDS and tuple(payload["panel_order"]) == PANELS, "tiling index header mismatch")
    entries = payload["entries"]
    require(isinstance(entries, dict) and len(entries) == 432, "tiling index entry count mismatch")
    expected_entry_keys = [
        canonical_tiling_key(panel, seed, depth, role)
        for panel, seed, _, depth, role in canonical_rows()
    ]
    require(list(entries) == expected_entry_keys, "tiling index canonical key/order mismatch")
    expected_partitions = {
        (seed, d_in, d_out): derive_frozen_common_tiling(seed, d_in, d_out)
        for seed in TILING_SEEDS
        for d_in, d_out in sorted({_matrix_shape(role)[:2] for role in ROLES})
    }
    checked = 0
    for row in frame.itertuples(index=False):
        entry_key = canonical_tiling_key(row.panel_id, int(row.tiling_seed), int(row.depth), row.role)
        entry = entries[entry_key]
        exact_keys(
            entry,
            (
                "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
                "d_in", "d_out", "rows", "cols", "row_index_sha256",
                "column_index_sha256", "partition_sha256", "weight_shape_bytes_sha256",
            ),
            "tiling index entry",
        )
        rows = entry["rows"]
        cols = entry["cols"]
        require(str(rows.dtype) == "torch.int64" and str(cols.dtype) == "torch.int64", "tiling indices must be int64")
        require(rows.is_contiguous() and cols.is_contiguous(), "tiling indices must be contiguous")
        require(tuple(rows.shape) == (int(row.d_in) // 64, 64), "row tiling tensor shape mismatch")
        require(tuple(cols.shape) == (int(row.d_out) // 64, 64), "column tiling tensor shape mismatch")
        row_flat = np.concatenate([np.asarray(value, dtype=np.int64) for value in rows])
        col_flat = np.concatenate([np.asarray(value, dtype=np.int64) for value in cols])
        require(np.array_equal(np.sort(row_flat), np.arange(int(row.d_in))), "row partition coverage failure")
        require(np.array_equal(np.sort(col_flat), np.arange(int(row.d_out))), "column partition coverage failure")
        require(all(np.all(np.diff(np.asarray(group)) >= 0) for group in rows), "unsorted row group")
        require(all(np.all(np.diff(np.asarray(group)) >= 0) for group in cols), "unsorted column group")
        row_array = np.asarray(rows, dtype=np.int64)
        col_array = np.asarray(cols, dtype=np.int64)
        row_sha = tensor_sha256(row_array, np.int64)
        col_sha = tensor_sha256(col_array, np.int64)
        partition_sha = hashlib.sha256(f"{row_sha}|{col_sha}".encode("utf-8")).hexdigest()
        require(row_sha == row.row_index_sha256 == entry["row_index_sha256"], "row index tensor hash mismatch")
        require(col_sha == row.column_index_sha256 == entry["column_index_sha256"], "column index tensor hash mismatch")
        require(partition_sha == row.partition_sha256 == entry["partition_sha256"], "combined partition hash mismatch")
        expected_rows, expected_cols = expected_partitions[(int(row.tiling_seed), int(row.d_in), int(row.d_out))]
        require(np.array_equal(row_array, expected_rows), "row indices differ from frozen registered RNG partition")
        require(np.array_equal(col_array, expected_cols), "column indices differ from frozen registered RNG partition")
        for field in (
            "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
            "d_in", "d_out", "row_index_sha256", "column_index_sha256",
            "partition_sha256", "weight_shape_bytes_sha256",
        ):
            require(entry[field] == getattr(row, field), f"tiling PT/CSV identity mismatch: {field}")
        checked += 1
    return frame, {
        "pass": True,
        "rows": len(frame),
        "index_entries": checked,
        "tiling_identity": "global_context_common_tiling_v2|shape=<d_in>x<d_out>",
        "shape_global_identity": True,
        "registered_seed_shape_partitions_rederived": len(expected_partitions),
        "combined_partition_hashes_recomputed": checked,
        "contracted_w_links_checked": checked if expected_w_grid is not None else 0,
    }


def aggregate_tiles(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 512 and values.shape[0] > 0, "invalid tile-code shape")
    ensure_finite(values, "tile codes")
    blocks = (
        np.mean(values, axis=0),
        np.std(values, axis=0, ddof=0),
        np.quantile(values, 0.10, axis=0, method="linear"),
        np.quantile(values, 0.50, axis=0, method="linear"),
        np.quantile(values, 0.90, axis=0, method="linear"),
    )
    return np.concatenate(blocks).astype(np.float64, copy=False)


def derive_countsketch_map(seed: int) -> tuple[np.ndarray, np.ndarray]:
    buckets = np.empty(4_096, dtype=np.int64)
    signs = np.empty(4_096, dtype=np.int64)
    for coordinate in range(4_096):
        digest = hashlib.sha256(
            f"global_context_countsketch_v1|{seed}|{coordinate}".encode("utf-8")
        ).digest()
        buckets[coordinate] = int.from_bytes(digest[:8], "big") % 512
        signs[coordinate] = 1 if (digest[8] & 1) == 0 else -1
    return buckets, signs


def validate_and_load_features(
    input_dir: Path,
    tiling: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(input_dir / "aggregate_feature_manifest.csv")
    exact_columns(manifest, FEATURE_COLUMNS, "aggregate_feature_manifest.csv")
    require(len(manifest) == 5_184, "aggregate feature manifest row count mismatch")
    expected_keys = [
        (rep, *row)
        for rep in REPRESENTATIONS
        for row in canonical_rows()
    ]
    actual_keys = list(manifest.loc[:, ["representation", "panel_id", "tiling_seed", "tiling_index", "depth", "role"]].itertuples(index=False, name=None))
    require(actual_keys == expected_keys, "aggregate feature canonical representation/row grid mismatch")
    require(not manifest.duplicated(["representation", "panel_id", "tiling_seed", "depth", "role"]).any(), "duplicate aggregate feature row")
    require(_bool_series(manifest["finite"], "aggregate finite").all(), "runner marked nonfinite aggregate feature")
    require(_bool_series(manifest["nonzero"], "aggregate nonzero").all(), "runner marked zero aggregate feature")

    arrays: dict[str, np.ndarray] = {}
    with np.load(input_dir / "aggregate_features.npz", allow_pickle=False) as archive:
        require(tuple(archive.files) == REPRESENTATIONS, f"aggregate NPZ key/order mismatch: {archive.files}")
        for rep in REPRESENTATIONS:
            expected_dim = 37 if rep == "raw_simple" else 2_560
            array = np.asarray(archive[rep])
            require(array.dtype == np.float64 and array.shape == (432, expected_dim), f"aggregate array contract mismatch for {rep}: {array.dtype}/{array.shape}")
            ensure_finite(array, f"aggregate array {rep}")
            require(np.linalg.norm(array) > 0, f"aggregate array is all zero: {rep}")
            arrays[rep] = np.array(array, copy=True)

    for rep_index, rep in enumerate(REPRESENTATIONS):
        rows = manifest.iloc[rep_index * 432 : (rep_index + 1) * 432]
        require((rows["array_name"] == rep).all(), f"array_name mismatch for {rep}")
        require(np.array_equal(rows["array_row_index"].to_numpy(), np.arange(432)), f"array row indices mismatch for {rep}")
        expected_dim = arrays[rep].shape[1]
        require((rows["feature_dim"] == expected_dim).all(), f"feature_dim mismatch for {rep}")
        for index, expected_hash in enumerate(rows["feature_tensor_sha256"]):
            require(tensor_sha256(arrays[rep][index], np.float64) == expected_hash, f"feature row hash mismatch {rep}/{index}")
        if rep == "raw_simple":
            require(_bool_series(rows["tiling_invariant"], f"{rep} invariant").all(), "raw_simple not marked invariant")
            require((rows["source_kind"] == "raw_simple").all(), "raw_simple source_kind mismatch")
            require(rows["source_seed"].isna().all(), "raw_simple source_seed must be empty")
        elif rep.startswith("countsketch_"):
            seed = int(rep.rsplit("_", 1)[1])
            require((rows["source_kind"] == "countsketch").all(), f"countsketch source kind mismatch {rep}")
            require((pd.to_numeric(rows["source_seed"]) == seed).all(), f"countsketch source seed mismatch {rep}")
            require(not _bool_series(rows["tiling_invariant"], f"{rep} invariant").any(), f"countsketch incorrectly invariant {rep}")
        elif rep.startswith("untrained_global_"):
            seed = int(rep.rsplit("_", 1)[1])
            require((rows["source_kind"] == "untrained_model").all(), f"untrained source kind mismatch {rep}")
            require((pd.to_numeric(rows["source_seed"]) == seed).all(), f"untrained source seed mismatch {rep}")
            require(not _bool_series(rows["tiling_invariant"], f"{rep} invariant").any(), f"untrained representation incorrectly invariant {rep}")
        else:
            require((rows["source_kind"] == "learned_checkpoint").all(), f"learned source kind mismatch {rep}")
            require(rows["source_seed"].isna().all(), f"learned source_seed must be empty {rep}")
            require(not _bool_series(rows["tiling_invariant"], f"{rep} invariant").any(), f"learned representation incorrectly invariant {rep}")

    tiling_partitions = tiling["partition_sha256"].to_numpy()
    for rep_index in range(len(REPRESENTATIONS)):
        rows = manifest.iloc[rep_index * 432 : (rep_index + 1) * 432]
        require(np.array_equal(rows["partition_sha256"].to_numpy(), tiling_partitions), "aggregate partition hash mismatch")
        for field in (
            "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
            "d_in", "d_out", "num_tiles",
        ):
            require(np.array_equal(rows[field].to_numpy(), tiling[field].to_numpy()), f"aggregate/tiling metadata mismatch {rep}/{field}")

    raw = pd.read_csv(input_dir / "raw_simple_features.csv", float_precision="round_trip")
    exact_columns(raw, RAW_SIMPLE_COLUMNS, "raw_simple_features.csv")
    require(len(raw) == 432, "raw simple row count mismatch")
    raw_keys = list(raw.loc[:, ["panel_id", "tiling_seed", "tiling_index", "depth", "role"]].itertuples(index=False, name=None))
    require(raw_keys == canonical_rows(), "raw simple canonical grid mismatch")
    raw_values = raw.loc[:, RAW_SIMPLE_FEATURES].to_numpy(dtype=np.float64)
    ensure_finite(raw_values, "raw simple features")
    require(np.array_equal(raw_values, arrays["raw_simple"]), "raw simple CSV/NPZ mismatch")
    require(_bool_series(raw["tiling_invariant"], "raw simple invariant").all(), "raw simple invariant flag failure")
    for field in RAW_SIMPLE_KEYS:
        require(np.array_equal(raw[field].to_numpy(), tiling[field].to_numpy()), f"raw-simple/tiling metadata mismatch {field}")
    require(np.array_equal(raw["weight_shape_bytes_sha256"].to_numpy(), tiling["weight_shape_bytes_sha256"].to_numpy()), "raw-simple/tiling weight hash mismatch")
    for index, expected_hash in enumerate(raw["feature_tensor_sha256"]):
        require(tensor_sha256(raw_values[index], np.float64) == expected_hash, f"raw simple row hash mismatch {index}")
    for panel in PANELS:
        for depth in DEPTHS:
            for role in ROLES:
                indices = manifest.index[(manifest.representation == "raw_simple") & (manifest.panel_id == panel) & (manifest.depth == depth) & (manifest.role == role)].to_numpy()
                local = indices - int(manifest.index[manifest.representation == "raw_simple"][0])
                require(len(local) == 2 and np.array_equal(arrays["raw_simple"][local[0]], arrays["raw_simple"][local[1]]), "raw_simple differs across technical tilings")

    maps = json_read(input_dir / "countsketch_maps.json")
    exact_keys(
        maps,
        ("schema_version", "coordinate_count_per_seed", "output_width", "seeds", "total_pair_count"),
        "CountSketch maps",
    )
    require(maps.get("schema_version") == "global_context_countsketch_maps_v1", "CountSketch map schema mismatch")
    require(maps.get("coordinate_count_per_seed") == 4_096 and maps.get("output_width") == 512 and maps.get("total_pair_count") == 20_480, "CountSketch header count mismatch")
    records = maps.get("seeds")
    require(isinstance(records, dict) and list(records) == [str(seed) for seed in COUNTSKETCH_SEEDS], "CountSketch seed grid/order mismatch")
    map_entry_count = 0
    for seed in COUNTSKETCH_SEEDS:
        record = records[str(seed)]
        exact_keys(
            record,
            (
                "seed", "pairs", "bucket_tensor_sha256", "sign_tensor_sha256",
                "pair_map_sha256", "all_buckets_in_range",
                "all_signs_exact_plus_or_minus_one",
            ),
            f"CountSketch seed record {seed}",
        )
        require(int(record["seed"]) == seed, "CountSketch record seed mismatch")
        pairs = record["pairs"]
        require(isinstance(pairs, list) and len(pairs) == 4_096, "CountSketch pair grid mismatch")
        require(all(set(pair) == {"coordinate", "bucket", "sign"} for pair in pairs), "CountSketch pair schema mismatch")
        require([int(pair["coordinate"]) for pair in pairs] == list(range(4_096)), "CountSketch coordinate order mismatch")
        buckets = np.asarray([pair["bucket"] for pair in pairs], dtype=np.int64)
        signs = np.asarray([pair["sign"] for pair in pairs], dtype=np.int64)
        require(buckets.shape == signs.shape == (4_096,), "CountSketch map shape mismatch")
        require(np.all((buckets >= 0) & (buckets < 512)) and set(np.unique(signs)).issubset({-1, 1}), "CountSketch range/sign mismatch")
        expected_buckets, expected_signs = derive_countsketch_map(seed)
        require(np.array_equal(buckets, expected_buckets) and np.array_equal(signs, expected_signs), f"CountSketch construction mismatch seed={seed}")
        bucket_hash = tensor_sha256(buckets, np.int64)
        sign_hash = tensor_sha256(signs, np.int64)
        require(record["bucket_tensor_sha256"] == bucket_hash, "CountSketch bucket hash mismatch")
        require(record["sign_tensor_sha256"] == sign_hash, "CountSketch sign hash mismatch")
        require(record["pair_map_sha256"] == hashlib.sha256(f"{bucket_hash}|{sign_hash}".encode("utf-8")).hexdigest(), "CountSketch combined map hash mismatch")
        require(record["all_buckets_in_range"] is True and record["all_signs_exact_plus_or_minus_one"] is True, "CountSketch runner range/sign audit failed")
        map_entry_count += len(buckets)
    require(map_entry_count == 20_480, "CountSketch map entry total mismatch")

    audit = {
        "pass": True,
        "aggregate_rows": len(manifest),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "feature_row_hashes_recomputed": len(manifest),
        "raw_simple_csv_npz_bit_exact": True,
        "raw_simple_tiling_invariant": True,
        "countsketch_maps_rederived": map_entry_count,
    }
    return manifest, arrays, raw, audit


def validate_and_load_codes(
    input_dir: Path,
    feature_manifest: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    tiling: pd.DataFrame,
    cell_template_hashes: Mapping[tuple[int, str], Mapping[str, str]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(input_dir / "code_manifest.csv")
    exact_columns(manifest, CODE_COLUMNS, "code_manifest.csv")
    require(len(manifest) == 2_592, "code manifest row count mismatch")
    expected_keys = [(rep, *row) for rep in CODE_REPRESENTATIONS for row in canonical_rows()]
    actual_keys = list(manifest.loc[:, ["representation", "panel_id", "tiling_seed", "tiling_index", "depth", "role"]].itertuples(index=False, name=None))
    require(actual_keys == expected_keys, "code manifest canonical grid mismatch")
    require(_bool_series(manifest["finite"], "code finite").all() and _bool_series(manifest["nonzero"], "code nonzero").all(), "runner marked invalid code")
    require((manifest["code_width"] == 512).all(), "code width mismatch")
    require((manifest["activation_consumers"] == 0).all() and (manifest["metadata_label_consumers"] == 0).all(), "model call consumed activation/metadata")

    import torch

    payload = torch.load(input_dir / "latent_codes.pt", map_location="cpu", weights_only=True)
    exact_keys(payload, ("schema_version", "representation_order", "entry_count", "total_code_rows", "code_width", "codes"), "latent_codes.pt")
    require(payload["schema_version"] == "global_context_latent_codes_v1", "latent code schema mismatch")
    require(tuple(payload["representation_order"]) == CODE_REPRESENTATIONS, "latent representation order mismatch")
    require(int(payload["entry_count"]) == 2_592 and int(payload["total_code_rows"]) == 746_496 and int(payload["code_width"]) == 512, "latent code header counts mismatch")
    stored = payload["codes"]
    require(isinstance(stored, dict) and len(stored) == 2_592, "latent code mapping count mismatch")
    codes: dict[str, np.ndarray] = {}
    independently_aggregated = 0
    total_rows = 0
    for row_index, row in enumerate(manifest.itertuples(index=False)):
        expected_code_key = (
            f"representation={row.representation}|panel={row.panel_id}|seed={int(row.tiling_seed)}|"
            f"depth={int(row.depth):02d}|role={row.role}"
        )
        require(row.code_key == expected_code_key, f"noncanonical code key: {row.code_key}")
        require(row.code_key in stored and row.code_key not in codes, f"missing/duplicate latent code key: {row.code_key}")
        tensor = stored[row.code_key]
        require(str(tensor.dtype) == "torch.float32" and tuple(tensor.shape) == (int(row.num_tiles), 512), f"stored code shape/dtype mismatch: {row.code_key}")
        array = tensor.detach().cpu().numpy()
        ensure_finite(array, f"latent code {row.code_key}")
        require(np.linalg.norm(array) > 0, f"latent code is zero: {row.code_key}")
        require(tensor_sha256(array, np.float32) == row.code_tensor_sha256, f"latent code hash mismatch: {row.code_key}")
        codes[row.code_key] = array
        total_rows += array.shape[0]
        rep = row.representation
        aggregate_row = row_index % 432
        recomputed = aggregate_tiles(array)
        require(np.array_equal(recomputed, arrays[rep][aggregate_row]), f"independent aggregate mismatch {rep}/{aggregate_row}")
        independently_aggregated += 1

        if rep == "learned_zero":
            require(row.c_var_sha256 == ZERO_C_SHA256 and row.c_patch_sha256 == ZERO_C_SHA256 and row.condition_kind == "exact_zero", "learned_zero condition mismatch")
        elif rep == "learned_cell":
            validate_cell_template_hash_pair(
                int(row.depth),
                str(row.role),
                str(row.c_var_sha256),
                str(row.c_patch_sha256),
                cell_template_hashes,
                label=f"learned_cell code row {row.code_key}",
            )
            require(row.condition_kind == "source_cell_mean", "learned_cell condition kind mismatch")
        else:
            require(row.c_var_sha256 == GLOBAL_C_VAR_SHA256 and row.c_patch_sha256 == GLOBAL_C_PATCH_SHA256 and row.condition_kind == "source_global", f"global condition mismatch {rep}")
        if rep.startswith("untrained_global_"):
            seed = int(rep.rsplit("_", 1)[1])
            require(
                row.model_kind == UNTRAINED_MODEL_KIND and int(row.model_seed) == seed,
                f"untrained model metadata mismatch {rep}",
            )
        else:
            require(row.model_kind == "learned_checkpoint", f"learned model kind mismatch {rep}")

        feature_row = feature_manifest.iloc[row_index]
        require(feature_row.representation == rep and feature_row.partition_sha256 == row.partition_sha256, "code/feature partition association mismatch")
        tiling_row = tiling.iloc[row_index % 432]
        for field in (
            "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
            "d_in", "d_out", "num_tiles", "partition_sha256", "weight_shape_bytes_sha256",
        ):
            require(getattr(row, field) == tiling_row[field], f"code/tiling association mismatch {rep}/{row_index}/{field}")
    require(total_rows == 746_496, f"stored code row total mismatch: {total_rows}")
    require(set(stored) == set(codes), "latent code mapping has stale entries")

    effects: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, str, int, int, str], str] = {}
    for row in manifest.itertuples(index=False):
        by_identity[(row.representation, row.panel_id, int(row.tiling_seed), int(row.depth), row.role)] = row.code_key
    for panel, seed, tiling_index, depth, role in canonical_rows():
        pairs = (
            ("global_vs_cell", "learned_global", "learned_cell"),
            ("global_vs_zero", "learned_global", "learned_zero"),
        )
        for comparison, left_rep, right_rep in pairs:
            left = codes[by_identity[(left_rep, panel, seed, depth, role)]].astype(np.float64).ravel()
            right = codes[by_identity[(right_rep, panel, seed, depth, role)]].astype(np.float64).ravel()
            left_norm = float(np.linalg.norm(left))
            right_norm = float(np.linalg.norm(right))
            delta_norm = float(np.linalg.norm(left - right))
            require(left_norm > 0 and right_norm > 0, "latent pair contains zero norm")
            effects.append(
                {
                    "comparison": comparison,
                    "left_representation": left_rep,
                    "right_representation": right_rep,
                    "panel_id": panel,
                    "tiling_seed": seed,
                    "tiling_index": tiling_index,
                    "depth": depth,
                    "role": role,
                    "num_tiles": int(left.size // 512),
                    "cosine": float(np.dot(left, right) / (left_norm * right_norm)),
                    "relative_l2_to_left": delta_norm / left_norm,
                    "left_l2": left_norm,
                    "right_l2": right_norm,
                    "delta_l2": delta_norm,
                }
            )
    effects_frame = pd.DataFrame(effects)
    require(len(effects_frame) == 864, "latent pair effect count mismatch")
    audit = {
        "pass": True,
        "code_manifest_rows": len(manifest),
        "stored_entries": len(codes),
        "stored_code_rows": total_rows,
        "stored_code_width": 512,
        "code_hashes_recomputed": len(codes),
        "aggregates_independently_recomputed": independently_aggregated,
        "aggregate_recomputation_bit_exact": True,
        "contracted_cell_template_code_rows_checked": 432,
        "latent_pair_effect_rows": len(effects_frame),
    }
    return manifest, codes, effects_frame, audit


def technical_average(features: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metadata_rows = [(panel, depth, role) for panel in PANELS for depth in DEPTHS for role in ROLES]
    metadata = pd.DataFrame(metadata_rows, columns=["panel_id", "depth", "role"])
    outputs: dict[str, np.ndarray] = {}
    for rep, array in features.items():
        reshaped = array.reshape(3, 2, 12, 6, array.shape[1])
        outputs[rep] = reshaped.mean(axis=1).reshape(216, array.shape[1])
    return metadata, outputs


def fit_scaler(train: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    mask = std >= 1e-12
    require(int(mask.sum()) > 0, "scaler dropped every feature")
    return {"mean": mean, "std": std, "mask": mask}


def apply_scaler(values: np.ndarray, scaler: Mapping[str, np.ndarray]) -> np.ndarray:
    mask = scaler["mask"]
    output = (np.asarray(values, dtype=np.float64)[:, mask] - scaler["mean"][mask]) / scaler["std"][mask]
    ensure_finite(output, "standardized features")
    return output


def source_scalers(
    unaveraged: Mapping[str, np.ndarray],
    averaged: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    source_unaveraged = slice(0, 144)
    source_averaged = slice(0, 72)
    unavg: dict[str, dict[str, np.ndarray]] = {}
    avg: dict[str, dict[str, np.ndarray]] = {}
    for rep in REPRESENTATIONS:
        if rep in {"learned_cell", "learned_zero"}:
            continue
        unavg[rep] = fit_scaler(unaveraged[rep][source_unaveraged])
        avg[rep] = fit_scaler(averaged[rep][source_averaged])
    unavg["learned_cell"] = unavg["learned_global"]
    unavg["learned_zero"] = unavg["learned_global"]
    avg["learned_cell"] = avg["learned_global"]
    avg["learned_zero"] = avg["learned_global"]
    return unavg, avg


def _panel_unaveraged(array: np.ndarray, panel: str) -> np.ndarray:
    panel_index = PANELS.index(panel)
    return array[panel_index * 144 : (panel_index + 1) * 144]


def _panel_averaged(array: np.ndarray, panel: str) -> np.ndarray:
    panel_index = PANELS.index(panel)
    return array[panel_index * 72 : (panel_index + 1) * 72]


def reliability_ratio(matched: np.ndarray, wrong: np.ndarray, label: str) -> tuple[float, float, float]:
    matched_median = float(np.median(matched))
    wrong_median = float(np.median(wrong))
    require(np.isfinite([matched_median, wrong_median]).all(), f"nonfinite reliability endpoint: {label}")
    require(wrong_median > 0, f"zero/nonpositive wrong-depth reliability denominator: {label}")
    return matched_median, wrong_median, matched_median / wrong_median


def _role_major(values: np.ndarray) -> np.ndarray:
    require(values.shape[0] == 72, "role-major conversion requires 72 rows")
    indices = [depth * 6 + role_index for role_index in range(6) for depth in range(12)]
    return values[indices]


def build_tiling_reliability(
    arrays: Mapping[str, np.ndarray],
    scalers: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    detail: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for rep in RELIABILITY_REPRESENTATIONS:
        transformed = apply_scaler(arrays[rep], scalers[rep])
        for panel in PANELS:
            local = _panel_unaveraged(transformed, panel).reshape(2, 12, 6, -1)
            matched = np.linalg.norm(local[0] - local[1], axis=-1)
            wrong_a = np.linalg.norm(local[0] - np.roll(local[1], shift=-6, axis=0), axis=-1)
            wrong_b = np.linalg.norm(local[1] - np.roll(local[0], shift=-6, axis=0), axis=-1)
            wrong = (wrong_a + wrong_b) / 2.0
            matched_median, denominator, ratio = reliability_ratio(matched, wrong, f"{rep}/{panel}")
            detail[(rep, panel)] = (matched, wrong)
            raw_rows.append(
                {
                    "representation": rep,
                    "panel_id": panel,
                    "endpoint_kind": "raw_representation",
                    "eligible": True,
                    "matched_median": matched_median,
                    "wrong_depth_median": denominator,
                    "tiling_ratio": ratio,
                    "S05": np.nan,
                    "S95": np.nan,
                    "member_count": 1,
                }
            )
    for panel in PANELS:
        raw_rows.append(
            {
                "representation": "raw_simple",
                "panel_id": panel,
                "endpoint_kind": "ineligible_tiling_invariant",
                "eligible": False,
                "matched_median": 0.0,
                "wrong_depth_median": 0.0,
                "tiling_ratio": np.nan,
                "S05": np.nan,
                "S95": np.nan,
                "member_count": 1,
            }
        )

    bootstrap_rows: list[dict[str, Any]] = []
    stability: dict[str, dict[str, Any]] = {}
    for panel in PANELS:
        matched, wrong = detail[("learned_global", panel)]
        stream_seed = stable_seed(STABILITY_BASE_SEED, "tiling_reliability", "learned_global", panel)
        rng = np.random.Generator(np.random.PCG64(stream_seed))
        values = np.empty(N_RESAMPLES, dtype=np.float64)
        for draw in range(N_RESAMPLES):
            sampled = rng.integers(0, 12, size=12)
            matched_median, wrong_median, values[draw] = reliability_ratio(
                matched[sampled].reshape(-1),
                wrong[sampled].reshape(-1),
                f"learned_global/{panel}/draw={draw}",
            )
            bootstrap_rows.append(
                {
                    "panel_id": panel,
                    "representation": "learned_global",
                    "draw": draw,
                    "stream_seed": stream_seed,
                    "tiling_ratio": values[draw],
                }
            )
        s05, s95 = np.quantile(values, [0.05, 0.95], method="linear")
        stability[panel] = {"stream_seed": stream_seed, "S05": float(s05), "S95": float(s95)}
        for row in raw_rows:
            if row["representation"] == "learned_global" and row["panel_id"] == panel:
                row["S05"] = float(s05)
                row["S95"] = float(s95)

    def add_endpoint(name: str, members: Sequence[str]) -> None:
        for panel in PANELS:
            member_rows = [row for row in raw_rows if row["panel_id"] == panel and row["representation"] in members]
            require(len(member_rows) == len(members), f"missing reliability endpoint members: {name}/{panel}")
            raw_rows.append(
                {
                    "representation": name,
                    "panel_id": panel,
                    "endpoint_kind": "arithmetic_seed_mean",
                    "eligible": True,
                    "matched_median": float(np.mean([row["matched_median"] for row in member_rows])),
                    "wrong_depth_median": float(np.mean([row["wrong_depth_median"] for row in member_rows])),
                    "tiling_ratio": float(np.mean([row["tiling_ratio"] for row in member_rows])),
                    "S05": np.nan,
                    "S95": np.nan,
                    "member_count": len(members),
                }
            )

    add_endpoint("untrained_global_endpoint", UNTRAINED_REPRESENTATIONS)
    add_endpoint("countsketch_endpoint", COUNTSKETCH_REPRESENTATIONS)
    result = pd.DataFrame(raw_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    require(len(result) == 42 and len(bootstrap) == 30_000, "tiling reliability output count mismatch")
    return result, bootstrap, {"pass": True, "level_a_streams": stability}


def distance_matrices(
    averaged: Mapping[str, np.ndarray],
    scalers: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for rep in REPRESENTATIONS:
        transformed = apply_scaler(averaged[rep], scalers[rep])
        for panel in PANELS:
            values = _role_major(_panel_averaged(transformed, panel))
            matrix = squareform(pdist(values, metric="euclidean"))
            ensure_finite(matrix, f"distance matrix {rep}/{panel}")
            require(np.allclose(matrix, matrix.T, atol=0, rtol=0) and np.all(np.diag(matrix) == 0), "distance matrix invalid")
            result[(rep, panel)] = matrix
    return result


def _spearman(x: np.ndarray, y: np.ndarray, *, undefined_zero: bool = False) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    ensure_finite(x_values, "Spearman left input")
    ensure_finite(y_values, "Spearman right input")
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        if undefined_zero:
            return 0.0
        raise RuntimeError("undefined Spearman statistic from constant input")
    value = float(spearmanr(x_values, y_values).statistic)
    if not np.isfinite(value):
        if undefined_zero:
            return 0.0
        raise RuntimeError("undefined Spearman statistic")
    return value


def qap_permutation(rng: np.random.Generator) -> np.ndarray:
    sigma_attention = rng.permutation(4)
    sigma = np.concatenate([sigma_attention, np.array([4, 5], dtype=np.int64)])
    permutation = np.empty(72, dtype=np.int64)
    for role_index in range(6):
        tau = rng.permutation(12)
        for depth in range(12):
            permutation[role_index * 12 + depth] = int(sigma[role_index]) * 12 + int(tau[depth])
    require(np.array_equal(np.sort(permutation), np.arange(72)), "QAP mapping is not a bijection")
    return permutation


def build_rsa(
    matrices: Mapping[tuple[str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel_pairs = ((PANELS[0], PANELS[1]), (PANELS[0], PANELS[2]), (PANELS[1], PANELS[2]))
    triangle = np.triu_indices(72, k=1)
    rows: list[dict[str, Any]] = []
    raw_lookup: dict[tuple[str, str, str], float] = {}
    for rep in REPRESENTATIONS:
        for left, right in panel_pairs:
            rho = _spearman(matrices[(rep, left)][triangle], matrices[(rep, right)][triangle])
            raw_lookup[(rep, left, right)] = rho
            rows.append(
                {
                    "representation": rep,
                    "panel_left": left,
                    "panel_right": right,
                    "endpoint_kind": "raw_representation",
                    "rsa_rho": rho,
                    "qap_p_one_sided": np.nan,
                    "member_count": 1,
                }
            )

    qap_rows: list[dict[str, Any]] = []
    qap_audit: dict[str, Any] = {}
    for left, right in panel_pairs:
        observed = raw_lookup[("learned_global", left, right)]
        stream_seed = stable_seed(QAP_BASE_SEED, "rsa", "learned_global", left, right, "structured_qap")
        rng = np.random.Generator(np.random.PCG64(stream_seed))
        left_vector = matrices[("learned_global", left)][triangle]
        left_ranks = rankdata(left_vector, method="average")
        right_matrix = matrices[("learned_global", right)]
        right_ranks_upper = rankdata(right_matrix[triangle], method="average")
        right_rank_matrix = np.zeros((72, 72), dtype=np.float64)
        right_rank_matrix[triangle] = right_ranks_upper
        right_rank_matrix[(triangle[1], triangle[0])] = right_ranks_upper
        left_centered = left_ranks - left_ranks.mean()
        left_scale = float(np.linalg.norm(left_centered))
        require(left_scale > 0, "QAP left ranks are degenerate")
        null_values = np.empty(N_RESAMPLES, dtype=np.float64)
        for draw in range(N_RESAMPLES):
            permutation = qap_permutation(rng)
            permuted_ranks = right_rank_matrix[permutation][:, permutation][triangle]
            right_centered = permuted_ranks - permuted_ranks.mean()
            right_scale = float(np.linalg.norm(right_centered))
            require(right_scale > 0, "QAP right ranks are degenerate")
            rho = float(np.dot(left_centered, right_centered) / (left_scale * right_scale))
            null_values[draw] = rho
            qap_rows.append(
                {
                    "panel_left": left,
                    "panel_right": right,
                    "representation": "learned_global",
                    "draw": draw,
                    "stream_seed": stream_seed,
                    "permutation_sha256": tensor_sha256(permutation, np.int64),
                    "null_rho": rho,
                }
            )
        p_value = float((1 + np.count_nonzero(null_values >= observed)) / (N_RESAMPLES + 1))
        for row in rows:
            if row["representation"] == "learned_global" and row["panel_left"] == left and row["panel_right"] == right:
                row["qap_p_one_sided"] = p_value
        qap_audit[f"{left}__{right}"] = {
            "stream_seed": stream_seed,
            "observed_rho": observed,
            "p_one_sided": p_value,
            "null_min": float(null_values.min()),
            "null_max": float(null_values.max()),
        }

    def add_endpoint(name: str, members: Sequence[str]) -> None:
        for left, right in panel_pairs:
            values = [raw_lookup[(member, left, right)] for member in members]
            rows.append(
                {
                    "representation": name,
                    "panel_left": left,
                    "panel_right": right,
                    "endpoint_kind": "arithmetic_seed_mean",
                    "rsa_rho": float(np.mean(values)),
                    "qap_p_one_sided": np.nan,
                    "member_count": len(members),
                }
            )

    add_endpoint("untrained_global_endpoint", UNTRAINED_REPRESENTATIONS)
    add_endpoint("countsketch_endpoint", COUNTSKETCH_REPRESENTATIONS)
    result = pd.DataFrame(rows)
    qap = pd.DataFrame(qap_rows)
    require(len(result) == 42 and len(qap) == 30_000, "RSA/QAP row count mismatch")
    for left, right in panel_pairs:
        observed = float(
            result[
                (result.representation == "learned_global")
                & (result.panel_left == left)
                & (result.panel_right == right)
            ].rsa_rho.iloc[0]
        )
        stored_null = qap[(qap.panel_left == left) & (qap.panel_right == right)].null_rho.to_numpy(dtype=np.float64)
        recomputed = float((1 + np.count_nonzero(stored_null >= observed)) / (len(stored_null) + 1))
        recorded = float(
            result[
                (result.representation == "learned_global")
                & (result.panel_left == left)
                & (result.panel_right == right)
            ].qap_p_one_sided.iloc[0]
        )
        require(recomputed == recorded, f"QAP p-value did not recompute from stored null rows: {left}/{right}")
    return result, qap, {"pass": True, "qap_streams": qap_audit, "joint_row_column_action": True}


def _scaler_audit_record(
    scaler: Mapping[str, np.ndarray],
    *,
    representation: str,
    heldout_panel: str,
    probe: str,
    role: str | None,
    train_panels: Sequence[str],
) -> dict[str, Any]:
    return {
        "representation": representation,
        "heldout_panel": heldout_panel,
        "probe": probe,
        "role": role,
        "train_panels": list(train_panels),
        "heldout_rows_in_fit": 0,
        "mean_sha256": tensor_sha256(scaler["mean"], np.float64),
        "std_sha256": tensor_sha256(scaler["std"], np.float64),
        "mask_sha256": tensor_sha256(scaler["mask"].astype(np.uint8), np.uint8),
        "kept_features": int(scaler["mask"].sum()),
    }


def require_probe_rank(values: np.ndarray, components: int, label: str) -> None:
    require(values.shape[1] >= components, f"{label} kept-feature count below frozen PCA dimension")
    rank = int(np.linalg.matrix_rank(values - values.mean(axis=0)))
    require(rank >= components, f"{label} matrix rank {rank} below frozen PCA dimension {components}")


def require_role_classes(labels: np.ndarray, label: str) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    require(np.array_equal(unique, np.arange(4)) and np.all(counts == counts[0]), f"{label} missing/unbalanced role classes")


def require_logistic_convergence(caught: Sequence[Any], iterations: np.ndarray, label: str) -> None:
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    require(not convergence_warnings and int(np.max(iterations)) < 10_000, f"{label} did not converge")


def build_fixed_probes(
    averaged: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    role_predictions: list[dict[str, Any]] = []
    role_metrics_raw: list[dict[str, Any]] = []
    depth_predictions: list[dict[str, Any]] = []
    depth_metrics_raw: list[dict[str, Any]] = []
    scaler_records: list[dict[str, Any]] = []
    convergence_records: list[dict[str, Any]] = []

    for rep in REPRESENTATIONS:
        values = averaged[rep]
        for heldout in PANELS:
            train_panels = tuple(panel for panel in PANELS if panel != heldout)
            train_blocks = [_panel_averaged(values, panel) for panel in train_panels]
            test_block = _panel_averaged(values, heldout)

            attention_indices = np.array(
                [depth * 6 + role_index for depth in DEPTHS for role_index in range(4)],
                dtype=np.int64,
            )
            x_train_raw = np.concatenate([block[attention_indices] for block in train_blocks], axis=0)
            # The concatenated blocks are depth-major with four roles per depth.
            y_train = np.tile(np.tile(np.arange(4), 12), len(train_panels))
            x_test_raw = test_block[attention_indices]
            y_test = np.tile(np.arange(4), 12)
            require_role_classes(y_train, f"role probe train classes {rep}/{heldout}")
            require_role_classes(y_test, f"role probe test classes {rep}/{heldout}")
            scaler = fit_scaler(x_train_raw)
            scaler_records.append(
                _scaler_audit_record(
                    scaler,
                    representation=rep,
                    heldout_panel=heldout,
                    probe="attention_role",
                    role=None,
                    train_panels=train_panels,
                )
            )
            x_train = apply_scaler(x_train_raw, scaler)
            x_test = apply_scaler(x_test_raw, scaler)
            require_probe_rank(x_train, 16, f"role probe {rep}/{heldout}")
            pca = PCA(n_components=16, svd_solver="full", whiten=False)
            train_pc = pca.fit_transform(x_train)
            test_pc = pca.transform(x_test)
            classifier = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                fit_intercept=True,
                class_weight=None,
                tol=1e-8,
                max_iter=10_000,
                random_state=BASE_SEED,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(train_pc, y_train)
            require_logistic_convergence(caught, classifier.n_iter_, f"role probe {rep}/{heldout}")
            prediction = classifier.predict(test_pc)
            probability = classifier.predict_proba(test_pc)
            accuracy = float(balanced_accuracy_score(y_test, prediction))
            role_metrics_raw.append(
                {
                    "representation": rep,
                    "heldout_panel": heldout,
                    "endpoint_kind": "raw_representation",
                    "balanced_accuracy": accuracy,
                    "converged": True,
                    "member_count": 1,
                }
            )
            convergence_records.append(
                {
                    "representation": rep,
                    "heldout_panel": heldout,
                    "probe": "attention_role",
                    "role": None,
                    "iterations": int(np.max(classifier.n_iter_)),
                    "converged": True,
                }
            )
            for local_index, (depth, role_index) in enumerate((depth, role_index) for depth in DEPTHS for role_index in range(4)):
                role_predictions.append(
                    {
                        "representation": rep,
                        "heldout_panel": heldout,
                        "depth": depth,
                        "role": ATTENTION_ROLES[role_index],
                        "true_class": role_index,
                        "predicted_class": int(prediction[local_index]),
                        "correct": bool(prediction[local_index] == role_index),
                        **{f"prob_{name}": float(probability[local_index, index]) for index, name in enumerate(ATTENTION_ROLES)},
                    }
                )

            for role_index, role in enumerate(ROLES):
                role_indices = np.array([depth * 6 + role_index for depth in DEPTHS], dtype=np.int64)
                role_train_raw = np.concatenate([block[role_indices] for block in train_blocks], axis=0)
                role_test_raw = test_block[role_indices]
                depth_scaler = fit_scaler(role_train_raw)
                scaler_records.append(
                    _scaler_audit_record(
                        depth_scaler,
                        representation=rep,
                        heldout_panel=heldout,
                        probe="depth",
                        role=role,
                        train_panels=train_panels,
                    )
                )
                role_train = apply_scaler(role_train_raw, depth_scaler)
                role_test = apply_scaler(role_test_raw, depth_scaler)
                require_probe_rank(role_train, 8, f"depth probe {rep}/{heldout}/{role}")
                depth_pca = PCA(n_components=8, svd_solver="full", whiten=False)
                train_depth_pc = depth_pca.fit_transform(role_train)
                test_depth_pc = depth_pca.transform(role_test)
                target_train = np.tile(np.arange(12, dtype=np.float64) / 11.0, 2)
                target_test = np.arange(12, dtype=np.float64) / 11.0
                regressor = Ridge(alpha=1.0, solver="svd", fit_intercept=True)
                regressor.fit(train_depth_pc, target_train)
                predicted_depth = regressor.predict(test_depth_pc)
                rho_raw = spearmanr(target_test, predicted_depth).statistic
                undefined = not np.isfinite(rho_raw)
                rho = 0.0 if undefined else float(rho_raw)
                mae = float(np.mean(np.abs(target_test - predicted_depth)))
                depth_metrics_raw.append(
                    {
                        "representation": rep,
                        "heldout_panel": heldout,
                        "role": role,
                        "metric_level": "role",
                        "endpoint_kind": "raw_representation",
                        "spearman_rho": rho,
                        "mae": mae,
                        "undefined_spearman": bool(undefined),
                        "member_count": 1,
                    }
                )
                convergence_records.append(
                    {
                        "representation": rep,
                        "heldout_panel": heldout,
                        "probe": "depth",
                        "role": role,
                        "iterations": 1,
                        "converged": True,
                    }
                )
                for depth in DEPTHS:
                    depth_predictions.append(
                        {
                            "representation": rep,
                            "heldout_panel": heldout,
                            "role": role,
                            "depth": depth,
                            "target_depth_scaled": float(target_test[depth]),
                            "predicted_depth_scaled": float(predicted_depth[depth]),
                            "absolute_error": float(abs(target_test[depth] - predicted_depth[depth])),
                        }
                    )

    role_prediction_frame = pd.DataFrame(role_predictions)
    role_metric_frame = pd.DataFrame(role_metrics_raw)
    depth_prediction_frame = pd.DataFrame(depth_predictions)
    depth_rows = list(depth_metrics_raw)
    for rep in REPRESENTATIONS:
        for panel in PANELS:
            subset = [row for row in depth_metrics_raw if row["representation"] == rep and row["heldout_panel"] == panel]
            require(len(subset) == 6, "depth macro missing roles")
            depth_rows.append(
                {
                    "representation": rep,
                    "heldout_panel": panel,
                    "role": "macro",
                    "metric_level": "panel_macro",
                    "endpoint_kind": "raw_representation",
                    "spearman_rho": float(np.mean([row["spearman_rho"] for row in subset])),
                    "mae": float(np.mean([row["mae"] for row in subset])),
                    "undefined_spearman": bool(any(row["undefined_spearman"] for row in subset)),
                    "member_count": 1,
                }
            )

    def add_seed_endpoint(name: str, members: Sequence[str]) -> None:
        for panel in PANELS:
            role_values = role_metric_frame[(role_metric_frame.heldout_panel == panel) & role_metric_frame.representation.isin(members)]
            require(len(role_values) == len(members), f"role endpoint member count mismatch {name}/{panel}")
            nonlocal_role_rows.append(
                {
                    "representation": name,
                    "heldout_panel": panel,
                    "endpoint_kind": "arithmetic_seed_mean",
                    "balanced_accuracy": float(role_values["balanced_accuracy"].mean()),
                    "converged": bool(role_values["converged"].all()),
                    "member_count": len(members),
                }
            )
            for role in ROLES:
                subset = [row for row in depth_metrics_raw if row["heldout_panel"] == panel and row["role"] == role and row["representation"] in members]
                require(len(subset) == len(members), f"depth endpoint member count mismatch {name}/{panel}/{role}")
                depth_rows.append(
                    {
                        "representation": name,
                        "heldout_panel": panel,
                        "role": role,
                        "metric_level": "role",
                        "endpoint_kind": "arithmetic_seed_mean",
                        "spearman_rho": float(np.mean([row["spearman_rho"] for row in subset])),
                        "mae": float(np.mean([row["mae"] for row in subset])),
                        "undefined_spearman": bool(any(row["undefined_spearman"] for row in subset)),
                        "member_count": len(members),
                    }
                )
            subset = [row for row in depth_rows if row["heldout_panel"] == panel and row["role"] in ROLES and row["representation"] == name]
            depth_rows.append(
                {
                    "representation": name,
                    "heldout_panel": panel,
                    "role": "macro",
                    "metric_level": "panel_macro",
                    "endpoint_kind": "arithmetic_seed_mean",
                    "spearman_rho": float(np.mean([row["spearman_rho"] for row in subset])),
                    "mae": float(np.mean([row["mae"] for row in subset])),
                    "undefined_spearman": bool(any(row["undefined_spearman"] for row in subset)),
                    "member_count": len(members),
                }
            )

    nonlocal_role_rows: list[dict[str, Any]] = []
    add_seed_endpoint("untrained_global_endpoint", UNTRAINED_REPRESENTATIONS)
    add_seed_endpoint("countsketch_endpoint", COUNTSKETCH_REPRESENTATIONS)
    role_metric_frame = pd.concat([role_metric_frame, pd.DataFrame(nonlocal_role_rows)], ignore_index=True)
    depth_metric_frame = pd.DataFrame(depth_rows)
    require(len(role_prediction_frame) == 1_728, "role prediction count mismatch")
    require(len(role_metric_frame) == 42, "role metric count mismatch")
    require(len(depth_prediction_frame) == 2_592, "depth prediction count mismatch")
    require(len(depth_metric_frame) == 294, f"depth metric count mismatch: {len(depth_metric_frame)}")
    audit = {
        "pass": True,
        "train_panels_only": True,
        "heldout_rows_in_any_fit": 0,
        "scaler_records": scaler_records,
        "convergence_records": convergence_records,
        "all_probes_converged": all(row["converged"] for row in convergence_records),
        "role_prediction_rows": len(role_prediction_frame),
        "depth_prediction_rows": len(depth_prediction_frame),
    }
    return role_prediction_frame, role_metric_frame, depth_prediction_frame, depth_metric_frame, audit


def _role_correct_array(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(["depth", "true_class"])
    require(len(ordered) == 48, "role prediction block must have 48 rows")
    return ordered["correct"].to_numpy(dtype=np.float64).reshape(12, 4)


def _role_accuracy_from_predictions(frame: pd.DataFrame, sampled_depths: np.ndarray) -> float:
    return float(_role_correct_array(frame)[sampled_depths].mean())


def _depth_macro_from_predictions(frame: pd.DataFrame, sampled_depths: np.ndarray) -> float:
    values: list[float] = []
    for role in ROLES:
        role_frame = frame[frame.role == role].set_index("depth")
        true = role_frame.loc[sampled_depths, "target_depth_scaled"].to_numpy(dtype=np.float64)
        predicted = role_frame.loc[sampled_depths, "predicted_depth_scaled"].to_numpy(dtype=np.float64)
        values.append(_spearman(true, predicted, undefined_zero=True))
    return float(np.mean(values))


def _depth_macro_draws(frame: pd.DataFrame, sampled_depths: np.ndarray) -> np.ndarray:
    require(sampled_depths.ndim == 2 and sampled_depths.shape[1] == 12, "depth draw index shape mismatch")
    target_by_depth = np.arange(12, dtype=np.float64) / 11.0
    true_values = target_by_depth[sampled_depths]
    true_ranks = rankdata(true_values, axis=1, method="average")
    true_centered = true_ranks - true_ranks.mean(axis=1, keepdims=True)
    true_norm = np.linalg.norm(true_centered, axis=1)
    role_values: list[np.ndarray] = []
    for role in ROLES:
        role_frame = frame[frame.role == role].sort_values("depth")
        require(len(role_frame) == 12 and role_frame.depth.tolist() == list(DEPTHS), "depth prediction role grid mismatch")
        predicted_by_depth = role_frame.predicted_depth_scaled.to_numpy(dtype=np.float64)
        predicted_values = predicted_by_depth[sampled_depths]
        predicted_ranks = rankdata(predicted_values, axis=1, method="average")
        predicted_centered = predicted_ranks - predicted_ranks.mean(axis=1, keepdims=True)
        denominator = true_norm * np.linalg.norm(predicted_centered, axis=1)
        correlations = np.zeros(sampled_depths.shape[0], dtype=np.float64)
        valid = denominator > 0
        correlations[valid] = np.sum(true_centered[valid] * predicted_centered[valid], axis=1) / denominator[valid]
        role_values.append(correlations)
    result = np.mean(np.stack(role_values, axis=1), axis=1)
    ensure_finite(result, "vectorized depth stability draws")
    return result


def build_probe_bootstraps(
    role_predictions: pd.DataFrame,
    depth_predictions: pd.DataFrame,
    role_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    role_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    streams: dict[str, Any] = {}
    for panel in PANELS:
        role_panel = role_predictions[(role_predictions.representation == "learned_global") & (role_predictions.heldout_panel == panel)]
        role_correct = _role_correct_array(role_panel)
        role_seed = stable_seed(STABILITY_BASE_SEED, "attention_role", "learned_global", panel)
        role_rng = np.random.Generator(np.random.PCG64(role_seed))
        role_samples = role_rng.integers(0, 12, size=(N_RESAMPLES, 12))
        role_values = role_correct[role_samples].mean(axis=(1, 2))
        depth_panel = depth_predictions[(depth_predictions.representation == "learned_global") & (depth_predictions.heldout_panel == panel)]
        depth_seed = stable_seed(STABILITY_BASE_SEED, "depth", "learned_global", panel)
        depth_rng = np.random.Generator(np.random.PCG64(depth_seed))
        depth_samples = depth_rng.integers(0, 12, size=(N_RESAMPLES, 12))
        depth_values = _depth_macro_draws(depth_panel, depth_samples)
        for draw in range(N_RESAMPLES):
            role_rows.append({"panel_id": panel, "representation": "learned_global", "draw": draw, "stream_seed": role_seed, "balanced_accuracy": role_values[draw]})
            depth_rows.append({"panel_id": panel, "representation": "learned_global", "draw": draw, "stream_seed": depth_seed, "macro_spearman_rho": depth_values[draw]})
        role_s05, role_s95 = np.quantile(role_values, [0.05, 0.95], method="linear")
        depth_s05, depth_s95 = np.quantile(depth_values, [0.05, 0.95], method="linear")
        streams[panel] = {
            "role": {"seed": role_seed, "S05": float(role_s05), "S95": float(role_s95)},
            "depth": {"seed": depth_seed, "S05": float(depth_s05), "S95": float(depth_s95)},
        }

    comparator_members = {
        "untrained_global_endpoint": UNTRAINED_REPRESENTATIONS,
        "raw_simple": ("raw_simple",),
        "countsketch_endpoint": COUNTSKETCH_REPRESENTATIONS,
    }
    delta_points: list[dict[str, Any]] = []
    for comparator, members in comparator_members.items():
        panel_deltas: list[float] = []
        for panel in PANELS:
            learned = float(role_metrics[(role_metrics.representation == "learned_global") & (role_metrics.heldout_panel == panel)].balanced_accuracy.iloc[0])
            member_values = [float(role_metrics[(role_metrics.representation == member) & (role_metrics.heldout_panel == panel)].balanced_accuracy.iloc[0]) for member in members]
            comparator_value = float(np.mean(member_values))
            delta = learned - comparator_value
            panel_deltas.append(delta)
            delta_points.append(
                {
                    "comparator": comparator,
                    "scope": "checkpoint",
                    "heldout_panel": panel,
                    "learned_global_accuracy": learned,
                    "comparator_accuracy": comparator_value,
                    "delta": delta,
                    "S05": np.nan,
                    "S95": np.nan,
                }
            )
        delta_points.append(
            {
                "comparator": comparator,
                "scope": "checkpoint_macro",
                "heldout_panel": "macro",
                "learned_global_accuracy": float(np.mean([row["learned_global_accuracy"] for row in delta_points if row["comparator"] == comparator and row["scope"] == "checkpoint"])),
                "comparator_accuracy": float(np.mean([row["comparator_accuracy"] for row in delta_points if row["comparator"] == comparator and row["scope"] == "checkpoint"])),
                "delta": float(np.mean(panel_deltas)),
                "S05": np.nan,
                "S95": np.nan,
            }
        )

    level_b_rows: list[dict[str, Any]] = []
    for comparator, members in comparator_members.items():
        values = np.empty(N_RESAMPLES, dtype=np.float64)
        stream_labels = {
            panel: ["level_b_attention_delta", "learned_global", panel, comparator]
            for panel in PANELS
        }
        stream_seeds = {
            panel: stable_seed(STABILITY_BASE_SEED, *stream_labels[panel])
            for panel in PANELS
        }
        samples_by_panel = {}
        for panel in PANELS:
            rng = np.random.Generator(np.random.PCG64(stream_seeds[panel]))
            samples_by_panel[panel] = rng.integers(0, 12, size=(N_RESAMPLES, 12))
        learned_correct = {
            panel: _role_correct_array(
                role_predictions[(role_predictions.representation == "learned_global") & (role_predictions.heldout_panel == panel)]
            )
            for panel in PANELS
        }
        member_correct = {
            (panel, member): _role_correct_array(
                role_predictions[(role_predictions.representation == member) & (role_predictions.heldout_panel == panel)]
            )
            for panel in PANELS
            for member in members
        }
        learned_draws = np.column_stack(
            [learned_correct[panel][samples_by_panel[panel]].mean(axis=(1, 2)) for panel in PANELS]
        )
        comparator_draws = np.column_stack(
            [
                np.mean(
                    np.column_stack(
                        [member_correct[(panel, member)][samples_by_panel[panel]].mean(axis=(1, 2)) for member in members]
                    ),
                    axis=1,
                )
                for panel in PANELS
            ]
        )
        learned_macros = learned_draws.mean(axis=1)
        comparator_macros = comparator_draws.mean(axis=1)
        values[:] = learned_macros - comparator_macros
        for draw in range(N_RESAMPLES):
            level_b_rows.append(
                {
                    "comparator": comparator,
                    "draw": draw,
                    **{
                        f"{panel}_stream_label": "|".join(stream_labels[panel])
                        for panel in PANELS
                    },
                    **{
                        f"{panel}_stream_seed": stream_seeds[panel]
                        for panel in PANELS
                    },
                    "learned_global_checkpoint_macro": float(learned_macros[draw]),
                    "comparator_checkpoint_macro": float(comparator_macros[draw]),
                    "delta": values[draw],
                }
            )
        s05, s95 = np.quantile(values, [0.05, 0.95], method="linear")
        for row in delta_points:
            if row["comparator"] == comparator and row["scope"] == "checkpoint_macro":
                row["S05"] = float(s05)
                row["S95"] = float(s95)
        streams[f"level_b__{comparator}"] = {
            "panel_streams": {
                panel: {
                    "labels": stream_labels[panel],
                    "seed": stream_seeds[panel],
                }
                for panel in PANELS
            },
            "S05": float(s05),
            "S95": float(s95),
        }

    role_bootstrap = pd.DataFrame(role_rows)
    depth_bootstrap = pd.DataFrame(depth_rows)
    delta_frame = pd.DataFrame(delta_points)
    level_b_bootstrap = pd.DataFrame(level_b_rows)
    require(len(role_bootstrap) == 30_000 and len(depth_bootstrap) == 30_000, "Level A probe bootstrap count mismatch")
    require(len(delta_frame) == 12 and len(level_b_bootstrap) == 30_000, "Level B delta/bootstrap count mismatch")
    require(
        len(
            {
                streams[f"level_b__{comparator}"]["panel_streams"][panel]["seed"]
                for comparator in COMPARATORS
                for panel in PANELS
            }
        )
        == 9,
        "Level B comparator-by-panel streams are not nine distinct deterministic streams",
    )
    return role_bootstrap, depth_bootstrap, delta_frame, level_b_bootstrap, {"pass": True, "streams": streams}


def build_level_b_secondary(
    depth_metrics: pd.DataFrame,
    rsa: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        for panel in PANELS:
            learned = float(
                depth_metrics[
                    (depth_metrics.representation == "learned_global")
                    & (depth_metrics.heldout_panel == panel)
                    & (depth_metrics.metric_level == "panel_macro")
                ].spearman_rho.iloc[0]
            )
            baseline = float(
                depth_metrics[
                    (depth_metrics.representation == comparator)
                    & (depth_metrics.heldout_panel == panel)
                    & (depth_metrics.metric_level == "panel_macro")
                ].spearman_rho.iloc[0]
            )
            rows.append(
                {
                    "comparator": comparator,
                    "metric": "depth_macro_spearman",
                    "panel_or_pair": panel,
                    "learned_global_value": learned,
                    "comparator_value": baseline,
                    "delta": learned - baseline,
                    "noninferiority_margin": -0.05,
                    "pass": bool(learned - baseline >= -0.05),
                }
            )
        pair_rows = rsa[rsa.representation == "learned_global"]
        for pair in pair_rows.itertuples(index=False):
            baseline = float(
                rsa[
                    (rsa.representation == comparator)
                    & (rsa.panel_left == pair.panel_left)
                    & (rsa.panel_right == pair.panel_right)
                ].rsa_rho.iloc[0]
            )
            delta = float(pair.rsa_rho - baseline)
            rows.append(
                {
                    "comparator": comparator,
                    "metric": "cross_checkpoint_rsa",
                    "panel_or_pair": f"{pair.panel_left}__{pair.panel_right}",
                    "learned_global_value": float(pair.rsa_rho),
                    "comparator_value": baseline,
                    "delta": delta,
                    "noninferiority_margin": -0.05,
                    "pass": bool(delta >= -0.05),
                }
            )
    frame = pd.DataFrame(rows)
    require(len(frame) == 18, "Level B secondary row count mismatch")
    return frame


def validate_seed_mean_endpoints(
    reliability: pd.DataFrame,
    rsa: pd.DataFrame,
    role_metrics: pd.DataFrame,
    depth_metrics: pd.DataFrame,
) -> dict[str, Any]:
    definitions = {
        "untrained_global_endpoint": UNTRAINED_REPRESENTATIONS,
        "countsketch_endpoint": COUNTSKETCH_REPRESENTATIONS,
    }
    checked = 0
    for endpoint, members in definitions.items():
        for panel in PANELS:
            expected = float(
                reliability[
                    (reliability.panel_id == panel) & reliability.representation.isin(members)
                ].tiling_ratio.mean()
            )
            actual = float(
                reliability[
                    (reliability.panel_id == panel) & (reliability.representation == endpoint)
                ].tiling_ratio.iloc[0]
            )
            require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15), f"reliability seed endpoint mismatch {endpoint}/{panel}")
            checked += 1
            expected = float(
                role_metrics[
                    (role_metrics.heldout_panel == panel) & role_metrics.representation.isin(members)
                ].balanced_accuracy.mean()
            )
            actual = float(
                role_metrics[
                    (role_metrics.heldout_panel == panel) & (role_metrics.representation == endpoint)
                ].balanced_accuracy.iloc[0]
            )
            require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15), f"role seed endpoint mismatch {endpoint}/{panel}")
            checked += 1
            for role in (*ROLES, "macro"):
                expected = float(
                    depth_metrics[
                        (depth_metrics.heldout_panel == panel)
                        & (depth_metrics.role == role)
                        & depth_metrics.representation.isin(members)
                    ].spearman_rho.mean()
                )
                actual = float(
                    depth_metrics[
                        (depth_metrics.heldout_panel == panel)
                        & (depth_metrics.role == role)
                        & (depth_metrics.representation == endpoint)
                    ].spearman_rho.iloc[0]
                )
                require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15), f"depth seed endpoint mismatch {endpoint}/{panel}/{role}")
                checked += 1
        for left, right in ((PANELS[0], PANELS[1]), (PANELS[0], PANELS[2]), (PANELS[1], PANELS[2])):
            expected = float(
                rsa[
                    (rsa.panel_left == left)
                    & (rsa.panel_right == right)
                    & rsa.representation.isin(members)
                ].rsa_rho.mean()
            )
            actual = float(
                rsa[
                    (rsa.panel_left == left)
                    & (rsa.panel_right == right)
                    & (rsa.representation == endpoint)
                ].rsa_rho.iloc[0]
            )
            require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15), f"RSA seed endpoint mismatch {endpoint}/{left}/{right}")
            checked += 1
    return {
        "pass": True,
        "endpoint_definitions": {key: list(value) for key, value in definitions.items()},
        "arithmetic_seed_mean_checks": checked,
        "best_seed_selection": False,
    }


def variance_components(values: np.ndarray) -> dict[str, float]:
    """Balanced SS for [panel, tiling, cell, feature]."""

    require(values.ndim == 4 and values.shape[:3] == (3, 2, 72), f"variance tensor shape mismatch: {values.shape}")
    grand = values.mean(axis=(0, 1, 2), keepdims=True)
    cell_effect = values.mean(axis=(0, 1), keepdims=True) - grand
    panel_effect = values.mean(axis=(1, 2), keepdims=True) - grand
    mean_over_tiling = values.mean(axis=1, keepdims=True)
    interaction = mean_over_tiling - grand - cell_effect - panel_effect
    residual = values - mean_over_tiling
    sums = {
        "shared_role_depth_cell": float(3 * 2 * np.square(cell_effect).sum()),
        "checkpoint_main": float(2 * 72 * np.square(panel_effect).sum()),
        "checkpoint_by_cell_interaction": float(2 * np.square(interaction).sum()),
        "tiling_residual": float(np.square(residual).sum()),
    }
    total = sum(sums.values())
    require(total > 0 and np.isfinite(total), "variance decomposition total invalid")
    direct_total = float(np.square(values - grand).sum())
    require(
        math.isclose(total, direct_total, rel_tol=1e-12, abs_tol=1e-10),
        f"balanced SS components are not orthogonal/complete: components={total} direct={direct_total}",
    )
    return {key: value / total for key, value in sums.items()}


def build_pca_and_variance(
    arrays: Mapping[str, np.ndarray],
    source_unaveraged_scalers: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scaler = source_unaveraged_scalers["learned_global"]
    global_standardized = apply_scaler(arrays["learned_global"], scaler)
    source = global_standardized[:144]
    require(source.shape[1] >= 10 and np.linalg.matrix_rank(source - source.mean(axis=0)) >= 10, "exploratory PCA source rank insufficient")
    pca = PCA(n_components=10, svd_solver="full", whiten=False)
    pca.fit(source)
    require(pca.components_.shape == (10, source.shape[1]), "PCA loading shape mismatch")
    np.savez_compressed(
        output_dir / "pca_scaler_and_loadings.npz",
        scaler_mean=np.asarray(scaler["mean"], dtype=np.float64),
        scaler_std=np.asarray(scaler["std"], dtype=np.float64),
        scaler_mask=np.asarray(scaler["mask"], dtype=np.uint8),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        source_training_row_indices=np.arange(144, dtype=np.int64),
    )

    score_rows: list[dict[str, Any]] = []
    standardized: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for rep in LEARNED_REPRESENTATIONS:
        standardized[rep] = apply_scaler(arrays[rep], scaler)
        scores[rep] = pca.transform(standardized[rep])
        ensure_finite(scores[rep], f"PCA scores {rep}")
        for index, (panel, seed, tiling_index, depth, role) in enumerate(canonical_rows()):
            score_rows.append(
                {
                    "representation": rep,
                    "panel_id": panel,
                    "tiling_seed": seed,
                    "tiling_index": tiling_index,
                    "depth": depth,
                    "role": role,
                    **{f"PC{component + 1}": float(scores[rep][index, component]) for component in range(10)},
                }
            )
    pca_frame = pd.DataFrame(score_rows)
    require(len(pca_frame) == 1_296, "PCA score row count mismatch")

    variance_rows: list[dict[str, Any]] = []
    for rep in LEARNED_REPRESENTATIONS:
        spaces = {
            "full_standardized": standardized[rep],
            "global_pca10": scores[rep],
            "global_pc1_pc2": scores[rep][:, :2],
        }
        for space, values in spaces.items():
            tensor = values.reshape(3, 2, 12, 6, values.shape[1]).reshape(3, 2, 72, values.shape[1])
            fractions = variance_components(tensor)
            require(abs(sum(fractions.values()) - 1.0) < 1e-10, "variance fractions do not sum to one")
            for component, fraction in fractions.items():
                variance_rows.append(
                    {
                        "formula_identifier": VARIANCE_DECOMPOSITION_CONTRACT["identifier"],
                        "representation": rep,
                        "space": space,
                        "component": component,
                        "ss_fraction": fraction,
                    }
                )
    variance = pd.DataFrame(variance_rows)
    require(len(variance) == 36, "variance decomposition row count mismatch")
    audit = {
        "pass": True,
        "variance_decomposition_contract": VARIANCE_DECOMPOSITION_CONTRACT,
        "source_only_scaler_fit_rows": 144,
        "source_only_pca_fit_rows": 144,
        "target_fit_rows": 0,
        "pca_components": 10,
        "pca_solver": "full",
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "scaler_mean_sha256": tensor_sha256(scaler["mean"], np.float64),
        "scaler_std_sha256": tensor_sha256(scaler["std"], np.float64),
        "scaler_mask_sha256": tensor_sha256(scaler["mask"].astype(np.uint8), np.uint8),
        "pca_components_sha256": tensor_sha256(pca.components_, np.float64),
    }
    return pca_frame, variance, audit


def _quantile(frame: pd.DataFrame, column: str, q: float) -> float:
    return float(np.quantile(frame[column].to_numpy(dtype=np.float64), q, method="linear"))


def build_decisions(
    validity: Sequence[tuple[str, bool, str]],
    reliability: pd.DataFrame,
    tiling_bootstrap: pd.DataFrame,
    rsa: pd.DataFrame,
    role_metrics: pd.DataFrame,
    role_bootstrap: pd.DataFrame,
    depth_metrics: pd.DataFrame,
    depth_bootstrap: pd.DataFrame,
    level_b_deltas: pd.DataFrame,
    level_b_bootstrap: pd.DataFrame,
    level_b_secondary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require(len(validity) == 8, "validity cell count must be exactly eight")
    rows: list[dict[str, Any]] = []
    for cell_id, passed, evidence in validity:
        rows.append(
            {
                "level": "VALIDITY",
                "criterion": cell_id,
                "scope": "experiment",
                "point_value": float(bool(passed)),
                "secondary_value": np.nan,
                "threshold": "pass=true",
                "pass": bool(passed),
                "evidence": evidence,
            }
        )

    for panel in PANELS:
        endpoint = reliability[(reliability.representation == "learned_global") & (reliability.panel_id == panel)].iloc[0]
        s95 = _quantile(tiling_bootstrap[tiling_bootstrap.panel_id == panel], "tiling_ratio", 0.95)
        rows.append(
            {
                "level": "A",
                "criterion": "A1_tiling_reliability",
                "scope": panel,
                "point_value": float(endpoint.tiling_ratio),
                "secondary_value": s95,
                "threshold": "point<0.75 and S95<1",
                "pass": bool(endpoint.tiling_ratio < 0.75 and s95 < 1.0),
                "evidence": "tiling_reliability.csv;tiling_bootstrap.csv",
            }
        )
    for endpoint in rsa[rsa.representation == "learned_global"].itertuples(index=False):
        rows.append(
            {
                "level": "A",
                "criterion": "A2_cross_checkpoint_rsa",
                "scope": f"{endpoint.panel_left}__{endpoint.panel_right}",
                "point_value": float(endpoint.rsa_rho),
                "secondary_value": float(endpoint.qap_p_one_sided),
                "threshold": "rho>=0.20 and QAP_p<=0.01",
                "pass": bool(endpoint.rsa_rho >= 0.20 and endpoint.qap_p_one_sided <= 0.01),
                "evidence": "rsa.csv;rsa_qap_permutations.csv",
            }
        )
    for panel in PANELS:
        metric = role_metrics[(role_metrics.representation == "learned_global") & (role_metrics.heldout_panel == panel)].iloc[0]
        s05 = _quantile(role_bootstrap[role_bootstrap.panel_id == panel], "balanced_accuracy", 0.05)
        rows.append(
            {
                "level": "A",
                "criterion": "A3_attention_role",
                "scope": panel,
                "point_value": float(metric.balanced_accuracy),
                "secondary_value": s05,
                "threshold": "balanced_accuracy>=0.40 and S05>0.25",
                "pass": bool(metric.balanced_accuracy >= 0.40 and s05 > 0.25),
                "evidence": "role_probe_metrics.csv;role_bootstrap.csv",
            }
        )
    for panel in PANELS:
        metric = depth_metrics[(depth_metrics.representation == "learned_global") & (depth_metrics.heldout_panel == panel) & (depth_metrics.metric_level == "panel_macro")].iloc[0]
        s05 = _quantile(depth_bootstrap[depth_bootstrap.panel_id == panel], "macro_spearman_rho", 0.05)
        rows.append(
            {
                "level": "A",
                "criterion": "A4_ordered_depth",
                "scope": panel,
                "point_value": float(metric.spearman_rho),
                "secondary_value": s05,
                "threshold": "macro_rho>=0.30 and S05>0",
                "pass": bool(metric.spearman_rho >= 0.30 and s05 > 0),
                "evidence": "depth_probe_metrics.csv;depth_bootstrap.csv",
            }
        )

    validity_pass = all(row[1] for row in validity)
    level_a_rows = [row for row in rows if row["level"] == "A"]
    level_a_pass = validity_pass and all(row["pass"] for row in level_a_rows)
    rows.append(
        {
            "level": "B",
            "criterion": "B0_level_A_prerequisite",
            "scope": "experiment",
            "point_value": float(level_a_pass),
            "secondary_value": np.nan,
            "threshold": "Level_A=true",
            "pass": level_a_pass,
            "evidence": "decision_cells.csv",
        }
    )
    for point in level_b_deltas[level_b_deltas.scope == "checkpoint"].itertuples(index=False):
        rows.append(
            {
                "level": "B",
                "criterion": "B1_checkpoint_attention_delta",
                "scope": f"{point.comparator}__{point.heldout_panel}",
                "point_value": float(point.delta),
                "secondary_value": np.nan,
                "threshold": "delta>=0",
                "pass": bool(point.delta >= 0),
                "evidence": "level_b_role_deltas.csv",
            }
        )
    for point in level_b_deltas[level_b_deltas.scope == "checkpoint_macro"].itertuples(index=False):
        boot = level_b_bootstrap[level_b_bootstrap.comparator == point.comparator]
        s05 = _quantile(boot, "delta", 0.05)
        rows.append(
            {
                "level": "B",
                "criterion": "B2_macro_attention_delta",
                "scope": point.comparator,
                "point_value": float(point.delta),
                "secondary_value": s05,
                "threshold": "delta>=0.05 and S05>0",
                "pass": bool(point.delta >= 0.05 and s05 > 0),
                "evidence": "level_b_role_deltas.csv;level_b_role_bootstrap.csv",
            }
        )
    for point in level_b_secondary.to_dict("records"):
        rows.append(
            {
                "level": "B",
                "criterion": "B3_depth_noninferiority" if point["metric"] == "depth_macro_spearman" else "B4_rsa_noninferiority",
                "scope": f"{point['comparator']}__{point['panel_or_pair']}",
                "point_value": float(point["delta"]),
                "secondary_value": np.nan,
                "threshold": "delta>=-0.05",
                "pass": bool(point["pass"]),
                "evidence": "level_b_secondary_noninferiority.csv",
            }
        )
    cells = pd.DataFrame(rows)
    require(len(cells) == 51, f"decision cell count mismatch: {len(cells)}")
    level_b_rows = cells[cells.level == "B"]
    require(len(level_b_rows) == 31, "Level B decision count mismatch")
    level_b_pass = bool(level_a_pass and level_b_rows["pass"].all())
    decision = {
        "schema_version": "global_context_latent_geometry_ablation_decision_v1",
        "frozen_design_sha256": DESIGN_SHA256,
        "validity_pass": validity_pass,
        "level_A_pass": bool(level_a_pass),
        "level_B_pass": level_b_pass,
        "level_A_claim_boundary": "conditional_on_three_fixed_vit_style_vision_checkpoints_and_one_fixed_source_global_condition",
        "level_B_claim_boundary": "learned_encoder_vs_predeclared_raw_and_same_architecture_untrained_controls_on_fixed_checkpoints",
        "cross_domain_claim_established": False,
        "population_generalization_established": False,
        "failed_validity_cells": cells[(cells.level == "VALIDITY") & ~cells["pass"]]["criterion"].tolist(),
        "failed_level_A_cells": cells[(cells.level == "A") & ~cells["pass"]][["criterion", "scope"]].to_dict("records"),
        "failed_level_B_cells": cells[(cells.level == "B") & ~cells["pass"]][["criterion", "scope"]].to_dict("records"),
        "cell_counts": {"validity": 8, "level_A": 12, "level_B": 31, "total": 51},
    }
    return cells, decision


PANEL_COLORS = {"source_vit_b_flickr": "#377eb8", "beans": "#ff7f00", "trocr_sroie": "#4daf4a"}
ROLE_COLORS = {role: plt.get_cmap("tab10")(index) for index, role in enumerate(ROLES)}


def _finish_figure(fig: plt.Figure, path: Path, *, legend: bool = True) -> None:
    if legend:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        if handles:
            unique: dict[str, Any] = {}
            for handle, label in zip(handles, labels, strict=True):
                unique.setdefault(label, handle)
            fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=min(5, len(unique)), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if legend else 1))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _global_scores(pca_scores: pd.DataFrame) -> pd.DataFrame:
    return pca_scores[pca_scores.representation == "learned_global"].copy()


def plot_global_by_panel(pca_scores: pd.DataFrame, path: Path) -> None:
    frame = _global_scores(pca_scores)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for panel in PANELS:
        group = frame[frame.panel_id == panel]
        for tiling, marker in ((1, "o"), (2, "^")):
            part = group[group.tiling_index == tiling]
            ax.scatter(part.PC1, part.PC2, s=24, alpha=0.72, color=PANEL_COLORS[panel], marker=marker, label=f"{panel} / tiling {tiling}")
    ax.set(title="Learned global-C: frozen source PCA", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=0.2)
    _finish_figure(fig, path)


def plot_global_by_role(pca_scores: pd.DataFrame, path: Path) -> None:
    frame = _global_scores(pca_scores)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for role in ROLES:
        group = frame[frame.role == role]
        ax.scatter(group.PC1, group.PC2, s=25, alpha=0.7, color=ROLE_COLORS[role], label=role)
    ax.set(title="Learned global-C by role", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=0.2)
    _finish_figure(fig, path)


def plot_global_by_depth(pca_scores: pd.DataFrame, path: Path) -> None:
    frame = _global_scores(pca_scores)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    scatter = ax.scatter(frame.PC1, frame.PC2, c=frame.depth, cmap="viridis", s=27, alpha=0.75)
    fig.colorbar(scatter, ax=ax, label="depth")
    ax.set(title="Learned global-C by depth", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=0.2)
    _finish_figure(fig, path, legend=False)


def plot_role_facets(pca_scores: pd.DataFrame, path: Path) -> None:
    frame = _global_scores(pca_scores)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, role in zip(axes.ravel(), ROLES, strict=True):
        for panel in PANELS:
            part = frame[(frame.role == role) & (frame.panel_id == panel)]
            for tiling, marker in ((1, "o"), (2, "^")):
                line = part[part.tiling_index == tiling].sort_values("depth")
                ax.plot(line.PC1, line.PC2, color=PANEL_COLORS[panel], alpha=0.45, linewidth=1)
                ax.scatter(line.PC1, line.PC2, color=PANEL_COLORS[panel], marker=marker, s=22, label=f"{panel}/t{tiling}")
        ax.set_title(role)
        ax.grid(alpha=0.2)
    fig.supxlabel("PC1")
    fig.supylabel("PC2")
    fig.suptitle("Role facets: panel trajectories and both tilings", y=1.04)
    _finish_figure(fig, path)


def plot_depth_facets(pca_scores: pd.DataFrame, path: Path) -> None:
    frame = _global_scores(pca_scores)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for ax, depth in zip(axes.ravel(), DEPTHS, strict=True):
        for panel in PANELS:
            for tiling, marker in ((1, "o"), (2, "^")):
                part = frame[(frame.depth == depth) & (frame.panel_id == panel) & (frame.tiling_index == tiling)]
                ax.scatter(part.PC1, part.PC2, color=PANEL_COLORS[panel], marker=marker, s=24, label=f"{panel}/t{tiling}")
        ax.set_title(f"depth {depth}")
        ax.grid(alpha=0.2)
    fig.supxlabel("PC1")
    fig.supylabel("PC2")
    fig.suptitle("Depth facets: six roles, three panels, both tilings", y=1.03)
    _finish_figure(fig, path)


def plot_condition_overlay(pca_scores: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    colors = {"learned_cell": "#984ea3", "learned_global": "#377eb8", "learned_zero": "#999999"}
    for ax, panel in zip(axes, PANELS, strict=True):
        for rep in LEARNED_REPRESENTATIONS:
            part = pca_scores[(pca_scores.panel_id == panel) & (pca_scores.representation == rep)]
            for tiling, marker in ((1, "o"), (2, "^")):
                local = part[part.tiling_index == tiling]
                ax.scatter(local.PC1, local.PC2, s=18, marker=marker, alpha=0.58, color=colors[rep], label=f"{rep}/t{tiling}")
        ax.set_title(panel)
        ax.grid(alpha=0.2)
    fig.supxlabel("PC1")
    fig.supylabel("PC2")
    fig.suptitle("Cell/global/zero conditions in frozen source-global PCA", y=1.05)
    _finish_figure(fig, path)


def _endpoint_order() -> tuple[str, ...]:
    return (
        "learned_cell", "learned_global", "learned_zero", "untrained_global_endpoint",
        "raw_simple", "countsketch_endpoint",
    )


def plot_reliability(reliability: pd.DataFrame, path: Path) -> None:
    frame = reliability[reliability.representation.isin(_endpoint_order())].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.13
    x = np.arange(len(PANELS), dtype=np.float64)
    for index, rep in enumerate(_endpoint_order()):
        values = [float(frame[(frame.representation == rep) & (frame.panel_id == panel)].tiling_ratio.iloc[0]) if rep != "raw_simple" else np.nan for panel in PANELS]
        ax.bar(x + (index - 2.5) * width, values, width=width, label=rep)
    ax.axhline(0.75, color="black", linestyle="--", linewidth=1, label="A1 point threshold")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1, label="wrong-depth parity")
    ax.set_xticks(x, PANELS, rotation=12)
    ax.set_ylabel("median matched / median wrong-depth")
    ax.set_title("Technical tiling reliability (raw_simple is ineligible)")
    ax.grid(axis="y", alpha=0.2)
    _finish_figure(fig, path)


def plot_rsa(rsa: pd.DataFrame, path: Path) -> None:
    frame = rsa[rsa.representation.isin(_endpoint_order())]
    pairs = [f"{left}\nvs\n{right}" for left, right in ((PANELS[0], PANELS[1]), (PANELS[0], PANELS[2]), (PANELS[1], PANELS[2]))]
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.13
    x = np.arange(3)
    for index, rep in enumerate(_endpoint_order()):
        values = []
        for left, right in ((PANELS[0], PANELS[1]), (PANELS[0], PANELS[2]), (PANELS[1], PANELS[2])):
            values.append(float(frame[(frame.representation == rep) & (frame.panel_left == left) & (frame.panel_right == right)].rsa_rho.iloc[0]))
        ax.bar(x + (index - 2.5) * width, values, width=width, label=rep)
    ax.axhline(0.20, color="black", linestyle="--", linewidth=1, label="A2 rho threshold")
    ax.set_xticks(x, pairs)
    ax.set_ylabel("Spearman RSA")
    ax.set_title("Cross-checkpoint role/depth geometry")
    ax.grid(axis="y", alpha=0.2)
    _finish_figure(fig, path)


def plot_role_accuracy(metrics: pd.DataFrame, path: Path) -> None:
    frame = metrics[metrics.representation.isin(_endpoint_order())]
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.13
    x = np.arange(3)
    for index, rep in enumerate(_endpoint_order()):
        values = [float(frame[(frame.representation == rep) & (frame.heldout_panel == panel)].balanced_accuracy.iloc[0]) for panel in PANELS]
        ax.bar(x + (index - 2.5) * width, values, width=width, label=rep)
    ax.axhline(0.25, color="red", linestyle=":", linewidth=1, label="chance")
    ax.axhline(0.40, color="black", linestyle="--", linewidth=1, label="A3 point threshold")
    ax.set_xticks(x, PANELS, rotation=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Fixed train-panels-only attention-role probe")
    ax.grid(axis="y", alpha=0.2)
    _finish_figure(fig, path)


def plot_depth_spearman(metrics: pd.DataFrame, path: Path) -> None:
    frame = metrics[(metrics.metric_level == "panel_macro") & metrics.representation.isin(_endpoint_order())]
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.13
    x = np.arange(3)
    for index, rep in enumerate(_endpoint_order()):
        values = [float(frame[(frame.representation == rep) & (frame.heldout_panel == panel)].spearman_rho.iloc[0]) for panel in PANELS]
        ax.bar(x + (index - 2.5) * width, values, width=width, label=rep)
    ax.axhline(0.30, color="black", linestyle="--", linewidth=1, label="A4 point threshold")
    ax.set_xticks(x, PANELS, rotation=12)
    ax.set_ylabel("macro six-role Spearman rho")
    ax.set_title("Fixed train-panels-only ordered-depth probe")
    ax.grid(axis="y", alpha=0.2)
    _finish_figure(fig, path)


def plot_level_b_deltas(deltas: pd.DataFrame, path: Path) -> None:
    frame = deltas[deltas.scope == "checkpoint_macro"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(COMPARATORS))
    points = frame.set_index("comparator").loc[list(COMPARATORS)]
    values = points.delta.to_numpy(dtype=np.float64)
    lower = values - points.S05.to_numpy(dtype=np.float64)
    upper = points.S95.to_numpy(dtype=np.float64) - values
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="o", capsize=5, color="#377eb8")
    ax.axhline(0.0, color="red", linestyle=":", label="zero delta")
    ax.axhline(0.05, color="black", linestyle="--", label="B macro threshold")
    ax.set_xticks(x, COMPARATORS, rotation=15)
    ax.set_ylabel("learned-global minus comparator accuracy")
    ax.set_title("Level B paired depth-block attention deltas (S05/S95)")
    ax.grid(axis="y", alpha=0.2)
    _finish_figure(fig, path)


def render_plots(
    output: Path,
    pca_scores: pd.DataFrame,
    reliability: pd.DataFrame,
    rsa: pd.DataFrame,
    role_metrics: pd.DataFrame,
    depth_metrics: pd.DataFrame,
    level_b_deltas: pd.DataFrame,
) -> dict[str, Any]:
    renderers = (
        ("global_pc1_pc2_by_panel.png", lambda path: plot_global_by_panel(pca_scores, path)),
        ("global_pc1_pc2_by_role.png", lambda path: plot_global_by_role(pca_scores, path)),
        ("global_pc1_pc2_by_depth.png", lambda path: plot_global_by_depth(pca_scores, path)),
        ("global_role_facets_by_panel.png", lambda path: plot_role_facets(pca_scores, path)),
        ("global_depth_facets_by_panel.png", lambda path: plot_depth_facets(pca_scores, path)),
        ("cell_global_zero_projected_pc1_pc2.png", lambda path: plot_condition_overlay(pca_scores, path)),
        ("tiling_reliability_by_panel_and_representation.png", lambda path: plot_reliability(reliability, path)),
        ("rsa_by_panel_pair_and_representation.png", lambda path: plot_rsa(rsa, path)),
        ("attention_role_balanced_accuracy.png", lambda path: plot_role_accuracy(role_metrics, path)),
        ("depth_spearman_by_panel_and_representation.png", lambda path: plot_depth_spearman(depth_metrics, path)),
        ("level_b_attention_paired_deltas.png", lambda path: plot_level_b_deltas(level_b_deltas, path)),
    )
    records: list[dict[str, Any]] = []
    for name, renderer in renderers:
        path = output / name
        renderer(path)
        require(path.is_file() and path.stat().st_size > 10_000, f"plot absent/too small: {name}")
        image = plt.imread(path)
        ensure_finite(image, f"plot pixels {name}")
        height, width = image.shape[:2]
        require(width >= 900 and height >= 500, f"plot resolution too small: {name}/{width}x{height}")
        require(float(np.std(image)) > 0.01, f"plot appears blank: {name}")
        records.append(
            {
                "filename": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "finite_pixels": True,
                "nonblank": True,
                "mechanical_pass": True,
                "visual_review_status": "PENDING_INDEPENDENT_POSTRUN_REVIEW",
            }
        )
    require(tuple(row["filename"] for row in records) == PLOT_FILES, "plot name/order grid mismatch")
    return {
        "schema_version": "global_context_latent_geometry_ablation_plot_audit_v1",
        "pass": True,
        "mechanical_pass": True,
        "visual_review_required_postrun": True,
        "plots": records,
        "count": len(records),
    }


def assert_analyzer_isolation() -> dict[str, Any]:
    forbidden = (
        "big_vae",
        "weight_quantile_vae",
        "training.big_vae",
        "distribution_encoder",
    )
    loaded = sorted(name for name in sys.modules if any(marker in name.lower() for marker in forbidden))
    require(not loaded, f"CPU analyzer imported forbidden Weight-AE modules: {loaded}")
    return {
        "pass": True,
        "device": "cpu",
        "weight_ae_modules": loaded,
        "weight_ae_forward_calls": 0,
        "decoder_forward_calls": 0,
        "distribution_encoder_calls": 0,
        "target_paths_opened": 0,
    }


def build_readme(output: Path, decision: Mapping[str, Any], counts: Mapping[str, int]) -> None:
    text = f"""# Global-context latent-geometry ablation: frozen CPU analysis

Frozen design SHA-256: `{DESIGN_SHA256}`.

The analyzer imported and executed no Weight-AE, decoder, or distribution encoder. It recomputed the registered geometry, probes, structured QAP nulls, depth-block stability rows, and decisions from sealed FP32 codes and FP64 aggregate features. Both technical tilings were averaged before every diagnostic except repeatability.

## Decision

- validity: `{decision['validity_pass']}`
- Level A fixed-chart claim: `{decision['level_A_pass']}`
- Level B learned-representation-over-controls claim: `{decision['level_B_pass']}`
- cross-domain claim established: `False`
- population generalization established: `False`

Exact failed cells are listed in `decision.json` and all 51 fixed cells are in `decision_cells.csv`.

## Scope

The result is conditional on three fixed ViT-style vision checkpoints and one fixed source-derived global condition. Checkpoint, task, dataset, and model identity remain confounded. `learned_cell` is a segregated positive reference and `learned_zero` is a secondary mechanism diagnostic; neither is eligible for Level A or Level B. The zero-weight sanity artifact is validity-only and supplies no feature, plot, probe, or decision metric.

The QAP tests are dependent pairwise tests and are not combined. S05/S95 values are fixed-depth stability summaries, not population confidence intervals. Random encoder and CountSketch endpoints are arithmetic seed means; no best-seed selection is performed.

## Artifact counts

```json
{json.dumps(dict(counts), indent=2, sort_keys=True)}
```

All 11 PNGs passed mechanical finite/nonblank/resolution checks. Independent visual inspection remains a post-run responsibility and is recorded as pending in `plot_audit.json`.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _runner_observed_counts(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    observed = metadata.get("observed_counts")
    require(isinstance(observed, dict), "runner metadata lacks observed_counts")
    required = {
        "tiling_manifest_rows": 432,
        "tiling_index_entries": 432,
        "code_manifest_rows": 2_592,
        "latent_code_entries": 2_592,
        "latent_code_rows": 746_496,
        "raw_simple_rows": 432,
        "countsketch_pairs": 20_480,
        "aggregate_manifest_rows": 5_184,
    }
    for key, expected in required.items():
        require(int(observed.get(key, -1)) == expected, f"runner observed count mismatch {key}")
    observed_width = observed.get("latent_code_width")
    require(
        type(observed_width) is int
        and observed_width == EXPECTED_RUNNER_COUNTS["latent_code_width"],
        "runner observed latent-code width audit failed",
    )
    expected = metadata.get("expected_counts")
    require(expected == EXPECTED_RUNNER_COUNTS, "runner metadata full expected-count contract mismatch")
    return required


def run_formal(args: argparse.Namespace, invocation_audit: Mapping[str, Any]) -> None:
    input_dir = require_formal_input(args.input_dir)
    require(input_dir == DEFAULT_INPUT, f"formal input path is not frozen: {input_dir}")
    require(args.output_dir.resolve(strict=False) == DEFAULT_OUTPUT, f"formal output path is not frozen: {args.output_dir}")
    require(args.design_path.resolve(strict=True) == DESIGN_PATH.resolve(strict=True), "formal design path is not frozen")
    require(args.external_contract.resolve(strict=True) == CONTRACT_PATH.resolve(strict=True), "formal external contract path is not frozen")
    external_contract = json_read(args.external_contract)
    require(isinstance(external_contract, Mapping), "external contract must be a mapping")
    runtime_preflight = validate_analysis_runtime(external_contract.get("runtime"))
    final_output = args.output_dir.resolve(strict=False)
    output = create_staged_output(final_output, allowed_parent=ARTIFACT_ROOT)
    try:
        logger = setup_logger(output)
    except Exception as error:
        quarantine_failed_staging(output, final_output, error, failure_stage="logger_setup")
        raise
    started = time.monotonic()
    script_path = Path(__file__).resolve(strict=True)
    try:
        resolved_config = {
            "schema_version": "global_context_latent_geometry_ablation_analysis_config_v1",
            "input_dir": str(input_dir),
            "output_dir": str(final_output),
            "design_path": str(args.design_path.resolve(strict=True)),
            "frozen_design_sha256": DESIGN_SHA256,
            "external_contract_path": str(args.external_contract.resolve(strict=True)),
            "analyzer_path": str(script_path),
            "analyzer_sha256": sha256_file(script_path),
            "device": "cpu",
            "feature_dtype": "FP64",
            "stored_code_dtype": "FP32",
            "base_seed": BASE_SEED,
            "qap_base_seed": QAP_BASE_SEED,
            "stability_base_seed": STABILITY_BASE_SEED,
            "resamples_per_stream": N_RESAMPLES,
            "cache_mode": "sealed_runner_artifacts_read_only",
            "verbose": True,
            "weight_ae_imports_or_forwards": 0,
            "runtime": runtime_preflight["runtime"],
            "formal_invocation": dict(invocation_audit),
        }
        json_write(output / "resolved_analysis_config.json", resolved_config)
        logger.info(
            "stage=startup input=%s output=%s device=cpu feature_dtype=FP64 code_dtype=FP32 seed=%s cache_mode=sealed verbose=true",
            input_dir,
            final_output,
            BASE_SEED,
        )

        logger.info("stage=input_manifest_and_contract_audit expected_runner_files=21")
        isolation_start = assert_analyzer_isolation()
        manifest_start = validate_artifact_manifest(input_dir)
        binding = validate_design_contract_binding(args.design_path, args.external_contract, input_dir)
        require(manifest_start["runner_sha256"] == binding["contract"]["runner_sha256"], "manifest/contract runner SHA mismatch")
        require(manifest_start["analyzer_sha256"] == binding["contract"]["analyzer_sha256"], "manifest/contract analyzer SHA mismatch")
        require(manifest_start["external_contract_sha256"] == binding["external_contract_sha256"], "manifest/contract digest mismatch")
        runner_audits = validate_runner_json_audits(input_dir, binding["contract"])
        metadata = json_read(input_dir / "runner_metadata.json")
        runner_counts = _runner_observed_counts(metadata)

        logger.info("stage=tiling_grid_and_index_audit expected_rows=432 identity=shape_only_v2")
        tiling, tiling_audit = validate_tiling_grid(input_dir, runner_audits["_expected_w_grid"])
        logger.info("tiling_audit pass=true rows=%s index_entries=%s", len(tiling), tiling_audit["index_entries"])

        logger.info("stage=feature_grid_and_independent_hash_audit expected_rows=5184")
        feature_manifest, arrays, _, feature_grid_audit = validate_and_load_features(input_dir, tiling)
        logger.info("feature_grid pass=true arrays=%s rows=%s", len(arrays), len(feature_manifest))

        logger.info("stage=code_grid_hash_and_independent_reaggregation expected_entries=2592 expected_rows=746496")
        code_manifest_checked, code_arrays, latent_effects, code_audit = validate_and_load_codes(
            input_dir,
            feature_manifest,
            arrays,
            tiling,
            runner_audits["_cell_template_hashes"],
        )
        del code_manifest_checked, code_arrays
        gc.collect()
        csv_write(output / "latent_pair_effects.csv", latent_effects)
        logger.info("code_audit pass=true entries=%s code_rows=%s pair_effects=%s", code_audit["stored_entries"], code_audit["stored_code_rows"], len(latent_effects))

        logger.info("stage=technical_tiling_averaging rows_per_representation=216")
        _, averaged = technical_average(arrays)
        source_unavg_scalers, source_avg_scalers = source_scalers(arrays, averaged)

        logger.info("stage=source_only_tiling_reliability resamples=%s", N_RESAMPLES)
        reliability, tiling_bootstrap, reliability_audit = build_tiling_reliability(arrays, source_unavg_scalers)
        csv_write(output / "tiling_reliability.csv", reliability)
        csv_write(output / "tiling_bootstrap.csv", tiling_bootstrap)

        logger.info("stage=source_only_rsa_and_joint_row_column_qap pairs=3 permutations_per_pair=%s", N_RESAMPLES)
        matrices = distance_matrices(averaged, source_avg_scalers)
        rsa, qap, rsa_audit = build_rsa(matrices)
        csv_write(output / "rsa.csv", rsa)
        csv_write(output / "rsa_qap_permutations.csv", qap)

        logger.info("stage=train_panels_only_fixed_probes outer_folds=3 representations=12")
        role_predictions, role_metrics, depth_predictions, depth_metrics, probe_audit = build_fixed_probes(averaged)
        csv_write(output / "role_probe_predictions.csv", role_predictions)
        csv_write(output / "role_probe_metrics.csv", role_metrics)
        csv_write(output / "depth_probe_predictions.csv", depth_predictions)
        csv_write(output / "depth_probe_metrics.csv", depth_metrics)
        endpoint_audit = validate_seed_mean_endpoints(reliability, rsa, role_metrics, depth_metrics)

        logger.info("stage=probe_depth_block_stability level_A_rows=60000 level_B_rows=30000")
        role_bootstrap, depth_bootstrap, level_b_deltas, level_b_bootstrap, bootstrap_audit = build_probe_bootstraps(
            role_predictions,
            depth_predictions,
            role_metrics,
        )
        csv_write(output / "role_bootstrap.csv", role_bootstrap)
        csv_write(output / "depth_bootstrap.csv", depth_bootstrap)
        csv_write(output / "level_b_role_deltas.csv", level_b_deltas)
        csv_write(output / "level_b_role_bootstrap.csv", level_b_bootstrap)
        secondary = build_level_b_secondary(depth_metrics, rsa)
        csv_write(output / "level_b_secondary_noninferiority.csv", secondary)

        logger.info("stage=source_global_scaler_pca_and_variance pca_components=10")
        pca_scores, variance, pca_audit = build_pca_and_variance(arrays, source_unavg_scalers, output)
        csv_write(output / "pca_scores.csv", pca_scores)
        csv_write(output / "variance_decomposition.csv", variance)

        logger.info("stage=fixed_plot_rendering count=11")
        plot_audit = render_plots(output, pca_scores, reliability, rsa, role_metrics, depth_metrics, level_b_deltas)
        json_write(output / "plot_audit.json", plot_audit)

        logger.info("stage=decision_cells validity=8 level_A=12 level_B=31")
        isolation_end = assert_analyzer_isolation()
        validity = (
            ("V1_immutable_input_and_source_seal", True, "input_manifest_audit.json;target_access_seal.json"),
            ("V2_W_grid_and_hashes", bool(runner_audits["zero_weight_sanity"]["pass"]), "input_audit.json;zero_weight_sanity.json"),
            ("V3_common_tiling_grid_and_shape_identity", bool(tiling_audit["pass"]), "tiling_manifest.csv;tiling_indices.pt"),
            ("V4_learned_model_and_known_source_preflight", bool(runner_audits["known_source_preflight"]["pass"]), "model_contract.json;known_source_numeric_preflight.json"),
            ("V5_untrained_initialization_fingerprints", len(runner_audits["untrained_fingerprints"]) == 3, "model_contract.json"),
            ("V6_primary_global_dataflow_template_constancy", bool(runner_audits["primary_global_summary"]["runtime_call_count"] == 1728), "dataflow_audit.json"),
            ("V7_code_feature_artifact_grids", bool(feature_grid_audit["pass"] and code_audit["pass"]), "feature_audit.json;artifact_manifest.json"),
            ("V8_analyzer_isolation_probe_convergence_plot_audit", bool(isolation_start["pass"] and isolation_end["pass"] and probe_audit["all_probes_converged"] and plot_audit["pass"]), "feature_audit.json;plot_audit.json"),
        )
        decisions, decision = build_decisions(
            validity,
            reliability,
            tiling_bootstrap,
            rsa,
            role_metrics,
            role_bootstrap,
            depth_metrics,
            depth_bootstrap,
            level_b_deltas,
            level_b_bootstrap,
            secondary,
        )
        csv_write(output / "decision_cells.csv", decisions)
        json_write(output / "decision.json", decision)

        feature_audit = {
            "schema_version": "global_context_latent_geometry_ablation_feature_audit_v1",
            "pass": True,
            "feature_grid": feature_grid_audit,
            "code_grid": code_audit,
            "technical_repeat_handling": {"averaged_before_rsa_and_probes": True, "bootstrap_resamples_tiling": False},
            "source_only_preprocessing": {
                "pass": True,
                "rsa_and_reliability_target_fit_rows": 0,
                "learned_cell_and_zero_use_learned_global_scaler": True,
                "independent_control_coordinate_bases_use_own_source_scalers": True,
            },
            "probe_scaler_scope": probe_audit,
            "reliability": reliability_audit,
            "rsa_qap": rsa_audit,
            "bootstrap": bootstrap_audit,
            "seed_mean_endpoints": endpoint_audit,
            "pca": pca_audit,
            "learned_cell_segregated_non_gating": True,
            "zero_weight_sanity_validity_only": True,
        }
        json_write(output / "feature_audit.json", feature_audit)

        logger.info("stage=runner_immutability_recheck")
        manifest_end = validate_artifact_manifest(input_dir)
        require(manifest_start == manifest_end, "runner manifest-covered tree changed during analysis")
        input_manifest_audit = {
            "schema_version": "global_context_latent_geometry_ablation_input_manifest_audit_v1",
            "pass": True,
            "runner_manifest_initial": manifest_start,
            "runner_manifest_final": manifest_end,
            "immutable_during_analysis": True,
            "contract_binding": {key: value for key, value in binding.items() if key != "contract"},
            "runner_json_audits": {
                key: value for key, value in runner_audits.items() if not key.startswith("_")
            },
            "runner_counts": runner_counts,
            "analyzer_isolation_start": isolation_start,
            "analyzer_isolation_end": isolation_end,
        }
        json_write(output / "input_manifest_audit.json", input_manifest_audit)

        actual_counts = {
            "latent_pair_effect_rows": len(latent_effects),
            "tiling_reliability_rows": len(reliability),
            "tiling_bootstrap_rows": len(tiling_bootstrap),
            "rsa_rows": len(rsa),
            "rsa_qap_rows": len(qap),
            "role_prediction_rows": len(role_predictions),
            "role_metric_rows": len(role_metrics),
            "role_bootstrap_rows": len(role_bootstrap),
            "depth_prediction_rows": len(depth_predictions),
            "depth_metric_rows": len(depth_metrics),
            "depth_bootstrap_rows": len(depth_bootstrap),
            "level_b_delta_rows": len(level_b_deltas),
            "level_b_bootstrap_rows": len(level_b_bootstrap),
            "level_b_secondary_rows": len(secondary),
            "variance_rows": len(variance),
            "pca_rows": len(pca_scores),
            "decision_rows": len(decisions),
            "plot_count": len(plot_audit["plots"]),
        }
        expected_analysis_counts = {key: EXPECTED_COUNTS[key] for key in actual_counts}
        require(actual_counts == expected_analysis_counts, f"final analysis counts mismatch: {actual_counts}")
        build_readme(output, decision, actual_counts)

        expected_names = set(TOP_LEVEL_FILES) | set(PLOT_FILES)
        actual_names = {
            path.name for path in output.iterdir()
            if path.is_file() and path.name not in {"artifact_manifest.json", ANALYZER_TRANSACTION_MARKER}
        }
        require(actual_names == expected_names, f"analyzer top-level file grid mismatch missing={sorted(expected_names-actual_names)} extra={sorted(actual_names-expected_names)}")
        require(not any(path.is_symlink() for path in output.rglob("*")), "analyzer output contains symlink")
        logger.info(
            "stage=complete elapsed_seconds=%.1f validity=%s level_A=%s level_B=%s output=%s",
            time.monotonic() - started,
            decision["validity_pass"],
            decision["level_A_pass"],
            decision["level_B_pass"],
            final_output,
        )
        logger.info("artifacts manifest=%s README=%s decision=%s plots=11", final_output / "artifact_manifest.json", final_output / "README.md", final_output / "decision.json")
    except Exception as error:
        logger.exception("stage=failed elapsed_seconds=%.1f", time.monotonic() - started)
        close_logger(logger)
        quarantine_failed_staging(output, final_output, error, failure_stage="formal_analysis")
        raise
    close_logger(logger)
    try:
        publish_staged_output(output, final_output, allowed_parent=ARTIFACT_ROOT)
    except Exception as error:
        quarantine_failed_staging(output, final_output, error, failure_stage="manifest_or_publication")
        raise


def _expect_rejection(label: str, callback: Any) -> str:
    try:
        callback()
    except (RuntimeError, AssertionError, ValueError, KeyError):
        return label
    raise AssertionError(f"synthetic corruption was not rejected: {label}")


def _synthetic_plot_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260816)
    pca_rows: list[dict[str, Any]] = []
    for rep_index, rep in enumerate(LEARNED_REPRESENTATIONS):
        for panel_index, (panel, seed, tiling_index, depth, role) in enumerate(canonical_rows()):
            role_index = ROLES.index(role)
            pca_rows.append(
                {
                    "representation": rep,
                    "panel_id": panel,
                    "tiling_seed": seed,
                    "tiling_index": tiling_index,
                    "depth": depth,
                    "role": role,
                    "PC1": depth + 0.3 * role_index + 0.5 * panel_index // 144 + 0.1 * rep_index + rng.normal(0, 0.03),
                    "PC2": role_index + 0.2 * panel_index // 144 + 0.1 * tiling_index + rng.normal(0, 0.03),
                    **{f"PC{index}": float(rng.normal()) for index in range(3, 11)},
                }
            )
    endpoint_order = _endpoint_order()
    reliability_rows = [
        {"representation": rep, "panel_id": panel, "tiling_ratio": 0.45 + 0.04 * index}
        for index, rep in enumerate(endpoint_order)
        for panel in PANELS
    ]
    rsa_rows = [
        {"representation": rep, "panel_left": left, "panel_right": right, "rsa_rho": 0.65 - 0.04 * index}
        for index, rep in enumerate(endpoint_order)
        for left, right in ((PANELS[0], PANELS[1]), (PANELS[0], PANELS[2]), (PANELS[1], PANELS[2]))
    ]
    role_rows = [
        {"representation": rep, "heldout_panel": panel, "balanced_accuracy": 0.70 - 0.04 * index}
        for index, rep in enumerate(endpoint_order)
        for panel in PANELS
    ]
    depth_rows = [
        {"representation": rep, "heldout_panel": panel, "metric_level": "panel_macro", "spearman_rho": 0.72 - 0.04 * index}
        for index, rep in enumerate(endpoint_order)
        for panel in PANELS
    ]
    delta_rows = [
        {"comparator": comparator, "scope": "checkpoint_macro", "delta": 0.12, "S05": 0.06, "S95": 0.17}
        for comparator in COMPARATORS
    ]
    return (
        pd.DataFrame(pca_rows),
        pd.DataFrame(reliability_rows),
        pd.DataFrame(rsa_rows),
        pd.DataFrame(role_rows),
        pd.DataFrame(depth_rows),
        pd.DataFrame(delta_rows),
    )


def audit_runner_model_kind_contract(runner_path: Path) -> dict[str, Any]:
    """Cross-script fixture for the producer/consumer code-manifest vocabulary."""

    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_kinds: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = node.func.id if isinstance(node.func, ast.Name) else None
        if function_name != "encode_fixed_condition_representation":
            continue
        keyword = next((item for item in node.keywords if item.arg == "model_kind"), None)
        require(
            keyword is not None
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str),
            "runner fixed-condition call lacks literal model_kind",
        )
        call_kinds.append(keyword.value.value)

    require(
        call_kinds.count("learned_checkpoint") == 2
        and call_kinds.count(UNTRAINED_MODEL_KIND) == 1
        and len(call_kinds) == 3,
        f"runner model_kind producer vocabulary drift: {call_kinds}",
    )
    require("untrained" not in call_kinds, "legacy untrained model_kind reappeared in runner")

    producer = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_code_and_feature_rows"
        ),
        None,
    )
    require(producer is not None, "runner code-manifest producer function absent")
    model_kind_passthrough = any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "model_kind"
            and isinstance(value, ast.Name)
            and value.id == "model_kind"
            for key, value in zip(node.keys, node.values, strict=True)
        )
        for node in ast.walk(producer)
    )
    require(model_kind_passthrough, "runner code manifest does not pass through model_kind")
    return {
        "pass": True,
        "runner_sha256": sha256_file(runner_path),
        "producer_call_model_kinds": call_kinds,
        "untrained_consumer_value": UNTRAINED_MODEL_KIND,
        "code_manifest_model_kind_passthrough": True,
    }


def _synthetic_contracted_input_payload() -> dict[str, Any]:
    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    cell_mean = {
        f"depth={depth:02d}|role={role}": {
            "c_var_sha256": digest(f"synthetic-cell-c-var|{depth}|{role}"),
            "c_patch_sha256": digest(f"synthetic-cell-c-patch|{depth}|{role}"),
        }
        for depth in DEPTHS
        for role in ROLES
    }
    source_template = {
        "schema_version": "global_context_source_template_audit_v1",
        "template_file_sha256": SOURCE_TEMPLATE_FILE_SHA256,
        "global": {
            "c_var_sha256": GLOBAL_C_VAR_SHA256,
            "c_patch_sha256": GLOBAL_C_PATCH_SHA256,
        },
        "cell_mean": cell_mean,
        "cell_medoid_audited_not_used": True,
        "global_hashes_match_frozen_design": True,
        "cell_count": 72,
        "pass": True,
    }
    w_rows = []
    zero_entries = []
    for panel, depth, role in canonical_cells():
        d_in, d_out, _ = _matrix_shape(role)
        shape_hash = digest(f"synthetic-W-shape-bytes|{panel}|{depth}|{role}")
        tensor_hash = digest(f"synthetic-W-tensor|{panel}|{depth}|{role}")
        w_rows.append(
            {
                "panel_id": panel,
                "depth": depth,
                "role": role,
                "shape": [d_in, d_out],
                "weight_shape_bytes_sha256": shape_hash,
                "weight_tensor_sha256": tensor_hash,
            }
        )
        numel = d_in * d_out
        sum_squared = float(numel)
        zero_entries.append(
            {
                "panel_id": panel,
                "depth": depth,
                "role": role,
                "d_in": d_in,
                "d_out": d_out,
                "weight_shape_bytes_sha256": shape_hash,
                "weight_tensor_sha256": tensor_hash,
                "numel": numel,
                "finite_count": numel,
                "nonzero_count": numel,
                "sum_squared": sum_squared,
                "frobenius_norm": math.sqrt(sum_squared),
                "rms": math.sqrt(sum_squared / numel),
                "literal_zero_relative_squared_error": 1.0,
            }
        )
    w_grid_sha256 = hashlib.sha256(
        json.dumps(w_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    zero_sanity = {
        "schema_version": "global_context_zero_weight_sanity_v1",
        "matrix_count": 216,
        "model_forward_calls": 0,
        "entries": zero_entries,
        "summary": {
            "all_finite": True,
            "all_have_nonzero_entry": True,
            "all_positive_sum_squared": True,
            "all_literal_zero_ratios_exactly_one": True,
            "pass": True,
        },
    }
    materialization = {
        "schema_version": "global_context_w_only_materialization_v1",
        "source_runtime": {"synthetic": True},
        "candidate_runtime": {"synthetic": True},
        "w_entry_count": 216,
        "w_grid_rows": w_rows,
        "w_grid_sha256": w_grid_sha256,
        "tiling_row_count": 432,
        "tiling_grid_sha256": digest("synthetic-tiling-grid"),
        "source_preflight": {"synthetic": True},
        "w_only_entry_fields": [
            "W", "depth", "panel_id", "role", "tilings",
            "weight_shape_bytes_sha256", "weight_tensor_sha256",
        ],
        "activation_tensors_retained": 0,
        "activation_references_retained": 0,
        "model_modules_loaded_before_completion": [],
        "garbage_collection_completed": True,
        "pass": True,
    }
    return {
        "schema_version": "global_context_input_audit_v1",
        "resources": {
            "source_condition_templates": {
                "path": "/synthetic/source_condition_templates.pt",
                "sha256": SOURCE_TEMPLATE_FILE_SHA256,
                "bytes": 1,
            }
        },
        "source_parent_audit": {"synthetic": True},
        "old_reference_audit": {"synthetic": True},
        "source_template_audit": source_template,
        "w_only_materialization": materialization,
        "zero_weight_sanity_preflight": zero_sanity,
        "weight_ae_model_modules_loaded": [],
        "weight_ae_forward_count": 0,
        "decoder_forward_count": 0,
        "distribution_encoder_forward_count": 0,
        "source_only_seal_clean": True,
        "pass": True,
    }


def _synthetic_contract_binding_test(temp: Path) -> dict[str, Any]:
    case = temp / "contract_binding"
    case.mkdir()
    external = case / "external_contract.json"
    runner_path = RUNNER_PATH.resolve(strict=True)
    analyzer_path = Path(__file__).resolve(strict=True)
    panel_builder_path = PANEL_BUILDER_PATH.resolve(strict=True)
    model_kind_contract = audit_runner_model_kind_contract(runner_path)
    import torch

    runtime = {
        **analysis_runtime_snapshot(),
        "omegaconf": importlib.metadata.version("omegaconf"),
        "hydra-core": importlib.metadata.version("hydra-core"),
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_0_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    contract: dict[str, Any] = {
        "schema_version": "global_context_latent_geometry_preexecution_contract_v1",
        "contract_schema_keys": list(CONTRACT_SCHEMA_KEYS),
        "created_utc": "synthetic",
        "source_only_seal": {
            "installed_before_scientific_input_read": True,
            "target_access_events": 0,
            "network_connections": 0,
            "subprocess_launches": 0,
        },
        "frozen_design_path": str(DESIGN_PATH.resolve(strict=True)),
        "frozen_design_sha256": DESIGN_SHA256,
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "analyzer_path": str(analyzer_path),
        "analyzer_sha256": sha256_file(analyzer_path),
        "runtime": runtime,
        "execution_config": expected_execution_config(),
        "frozen_grids": {
            "panel_order": list(PANELS),
            "role_order": list(ROLES),
            "depths": list(DEPTHS),
            "common_tiling_seeds": list(TILING_SEEDS),
            "code_representation_order": list(CODE_REPRESENTATIONS),
            "representation_order": list(REPRESENTATIONS),
            "untrained_seeds": list(UNTRAINED_SEEDS),
            "countsketch_seeds": list(COUNTSKETCH_SEEDS),
            "role_shapes": {role: list(_matrix_shape(role)[:2]) for role in ROLES},
        },
        "scripts": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in (
                ("runner", runner_path),
                ("analyzer", analyzer_path),
                ("panel_builder", panel_builder_path),
            )
        },
        "model_dependencies": {"synthetic": True},
        "input_audit": _synthetic_contracted_input_payload(),
        "static_call_graph_audit": {
            "schema_version": "global_context_static_call_graph_audit_v1",
            "script_sha256": sha256_file(runner_path),
            "encoder_function": "encode_weight_tiles",
            "formal_arguments": ["model", "W_tiles", "fixed_template"],
            "forbidden_scientific_argument_or_local_names": [],
            "forbidden_decoder_or_distribution_attribute_calls": [],
            "encode_z_dec_subscript_call_count": 1,
            "model_receives_only_weight_batch_and_expanded_fixed_condition": True,
            "pass": True,
        },
        "expected_counts": EXPECTED_RUNNER_COUNTS,
        "artifact_contract": {
            "self_excluded_files": list(RUNNER_FILES),
            "declared_file_count": 21,
            "file_count_including_manifest": 22,
            "artifact_manifest_self_excluded": True,
            "runner_and_analyzer_outputs_disjoint": True,
            "no_symlinks": True,
        },
        "dataflow_contract": expected_dataflow_contract(),
        "analysis_contract": {"variance_decomposition": VARIANCE_DECOMPOSITION_CONTRACT},
    }

    def write_case(payload: Mapping[str, Any]) -> str:
        json_write(external, payload)
        copied = case / "preexecution_contract.json"
        json_write(copied, payload)
        digest = sha256_file(external)
        require(sha256_file(copied) == digest, "synthetic copied contract is not exact")
        binding = {
            "schema_version": "global_context_preexecution_binding_v1",
            "frozen_design_path": str(DESIGN_PATH.resolve(strict=True)),
            "frozen_design_sha256": DESIGN_SHA256,
            "runner_path": str(runner_path),
            "runner_sha256": sha256_file(runner_path),
            "analyzer_path": str(analyzer_path),
            "analyzer_sha256": sha256_file(analyzer_path),
            "external_contract_path": str(external.resolve(strict=True)),
            "external_contract_sha256": digest,
            "copied_contract_path": str(copied.resolve(strict=True)),
            "copied_contract_sha256": digest,
            "runner_output_path": str(case.resolve(strict=True)),
            "analyzer_output_path": str(DEFAULT_OUTPUT),
            "exact_contract_verified_before_model_import": True,
        }
        json_write(case / "preexecution_binding.json", binding)
        identity = {
            "frozen_design_sha256": DESIGN_SHA256,
            "external_contract_sha256": digest,
            "runner_sha256": sha256_file(runner_path),
            "analyzer_sha256": sha256_file(analyzer_path),
        }
        json_write(case / "runner_metadata.json", identity)
        json_write(case / "resolved_config.json", identity)
        return digest

    digest = write_case(contract)
    audit = validate_design_contract_binding(DESIGN_PATH, external, case.resolve(strict=True))
    require(audit["external_contract_sha256"] == digest, "synthetic contract digest audit mismatch")
    corrupted = json.loads(json.dumps(contract))
    corrupted["expected_counts"]["latent_code_rows"] += 1
    write_case(corrupted)
    rejection = _expect_rejection(
        "nested_contract_count_corruption",
        lambda: validate_design_contract_binding(DESIGN_PATH, external, case.resolve(strict=True)),
    )
    runtime_corrupted = json.loads(json.dumps(contract))
    runtime_corrupted["runtime"]["numpy"] = "0.0.synthetic-corruption"
    write_case(runtime_corrupted)
    runtime_rejection = _expect_rejection(
        "contract_runtime_corruption",
        lambda: validate_design_contract_binding(DESIGN_PATH, external, case.resolve(strict=True)),
    )
    execution_corrupted = json.loads(json.dumps(contract))
    execution_corrupted["execution_config"]["batch_size"] = 63
    write_case(execution_corrupted)
    execution_rejection = _expect_rejection(
        "contract_exact_execution_config_corruption",
        lambda: validate_design_contract_binding(DESIGN_PATH, external, case.resolve(strict=True)),
    )
    static_corrupted = json.loads(json.dumps(contract))
    static_corrupted["static_call_graph_audit"]["script_sha256"] = "f" * 64
    write_case(static_corrupted)
    static_rejection = _expect_rejection(
        "contract_static_runner_sha_corruption",
        lambda: validate_design_contract_binding(DESIGN_PATH, external, case.resolve(strict=True)),
    )
    w_corrupted = json.loads(json.dumps(contract))
    w_rows = w_corrupted["input_audit"]["w_only_materialization"]["w_grid_rows"]
    w_rows[0]["weight_tensor_sha256"] = "e" * 64
    w_corrupted["input_audit"]["w_only_materialization"]["w_grid_sha256"] = hashlib.sha256(
        json.dumps(w_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_case(w_corrupted)
    w_crosslink_rejection = _expect_rejection(
        "contract_W_zero_crosslink_corruption",
        lambda: validate_design_contract_binding(DESIGN_PATH, external, case.resolve(strict=True)),
    )
    contracted_semantics = validate_contracted_input_audit(contract["input_audit"])
    cell_crosslink_rejection = _expect_rejection(
        "contract_cell_hash_consumer_corruption",
        lambda: validate_cell_template_hash_pair(
            0,
            ROLES[0],
            "d" * 64,
            contracted_semantics["cell_template_hashes"][(0, ROLES[0])]["c_patch"],
            contracted_semantics["cell_template_hashes"],
            label="synthetic cell consumer",
        ),
    )
    write_case(contract)
    return {
        "pass": True,
        "valid_exact_binding": True,
        "nested_corruption_rejected": rejection,
        "runtime_corruption_rejected": runtime_rejection,
        "execution_config_corruption_rejected": execution_rejection,
        "static_runner_sha_corruption_rejected": static_rejection,
        "W_zero_crosslink_corruption_rejected": w_crosslink_rejection,
        "cell_consumer_crosslink_corruption_rejected": cell_crosslink_rejection,
        "canonical_W_rows_checked": len(contracted_semantics["w_grid"]),
        "full_cell_hash_map_checked": len(contracted_semantics["cell_template_hashes"]),
        "runner_model_kind_contract": model_kind_contract,
    }


def _synthetic_tiling_validation_test(temp: Path) -> dict[str, Any]:
    case = temp / "tiling_validation"
    case.mkdir()
    import torch

    synthetic_input = _synthetic_contracted_input_payload()
    expected_w_grid = validate_contracted_w_grid(synthetic_input["w_only_materialization"])
    partition_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, str, str, str]] = {}
    csv_rows: list[dict[str, Any]] = []
    entries: dict[str, dict[str, Any]] = {}
    for panel, seed, tiling_index, depth, role in canonical_rows():
        d_in, d_out, num_tiles = _matrix_shape(role)
        cache_key = (seed, d_in, d_out)
        if cache_key not in partition_cache:
            rows, cols = derive_frozen_common_tiling(seed, d_in, d_out)
            row_sha = tensor_sha256(rows, np.int64)
            col_sha = tensor_sha256(cols, np.int64)
            combined = hashlib.sha256(f"{row_sha}|{col_sha}".encode("utf-8")).hexdigest()
            partition_cache[cache_key] = (rows, cols, row_sha, col_sha, combined)
        rows, cols, row_sha, col_sha, combined = partition_cache[cache_key]
        weight_hash = expected_w_grid[(panel, depth, role)]["weight_shape_bytes_sha256"]
        matrix = f"depth={depth:02d}|role={role}"
        csv_rows.append(
            {
                "panel_id": panel,
                "tiling_seed": seed,
                "tiling_index": tiling_index,
                "depth": depth,
                "role": role,
                "matrix_key": matrix,
                "d_in": d_in,
                "d_out": d_out,
                "row_groups": d_in // 64,
                "col_groups": d_out // 64,
                "num_tiles": num_tiles,
                "row_index_sha256": row_sha,
                "column_index_sha256": col_sha,
                "partition_sha256": combined,
                "weight_shape_bytes_sha256": weight_hash,
                "coverage_min": 1,
                "coverage_max": 1,
                "weight_reassembly_bit_exact": True,
                "coordinate_reassembly_bit_exact": True,
                "shape_global_identity_bit_exact": True,
            }
        )
        entries[canonical_tiling_key(panel, seed, depth, role)] = {
            "panel_id": panel,
            "tiling_seed": seed,
            "tiling_index": tiling_index,
            "depth": depth,
            "role": role,
            "matrix_key": matrix,
            "d_in": d_in,
            "d_out": d_out,
            "rows": torch.from_numpy(rows.copy()),
            "cols": torch.from_numpy(cols.copy()),
            "row_index_sha256": row_sha,
            "column_index_sha256": col_sha,
            "partition_sha256": combined,
            "weight_shape_bytes_sha256": weight_hash,
        }
    frame = pd.DataFrame(csv_rows, columns=TILING_COLUMNS)
    csv_write(case / "tiling_manifest.csv", frame)
    payload = {
        "schema_version": "global_context_tiling_indices_v1",
        "entry_count": 432,
        "seeds": list(TILING_SEEDS),
        "panel_order": list(PANELS),
        "entries": entries,
    }
    torch.save(payload, case / "tiling_indices.pt")
    _, valid = validate_tiling_grid(case, expected_w_grid)
    first_key = next(iter(entries))
    corrupted_payload = torch.load(case / "tiling_indices.pt", map_location="cpu", weights_only=True)
    corrupted_payload["entries"][first_key]["rows"] = corrupted_payload["entries"][first_key]["rows"].clone()
    corrupted_payload["entries"][first_key]["rows"][0, 0] = corrupted_payload["entries"][first_key]["rows"][0, 1]
    torch.save(corrupted_payload, case / "tiling_indices.pt")
    index_rejection = _expect_rejection("tiling_index_tensor_corruption", lambda: validate_tiling_grid(case, expected_w_grid))
    torch.save(payload, case / "tiling_indices.pt")
    corrupted_frame = frame.copy()
    corrupted_frame.loc[0, "matrix_key"] = "depth=99|role=attn_query"
    csv_write(case / "tiling_manifest.csv", corrupted_frame)
    identity_rejection = _expect_rejection("tiling_csv_identity_corruption", lambda: validate_tiling_grid(case, expected_w_grid))
    csv_write(case / "tiling_manifest.csv", frame)
    return {
        "pass": True,
        "rows": valid["rows"],
        "registered_seed_shape_partitions_rederived": valid["registered_seed_shape_partitions_rederived"],
        "combined_partition_hashes_recomputed": valid["combined_partition_hashes_recomputed"],
        "contracted_W_links_checked": valid["contracted_w_links_checked"],
        "index_corruption_rejected": index_rejection,
        "identity_corruption_rejected": identity_rejection,
    }


def _synthetic_transaction_and_invocation_test(temp: Path) -> dict[str, Any]:
    failed_final = temp / "analysis_failed_final"
    failed_staging = create_staged_output(failed_final, allowed_parent=temp)
    (failed_staging / "diagnostic.txt").write_text("synthetic failure\n", encoding="utf-8")
    quarantine = quarantine_failed_staging(
        failed_staging,
        failed_final,
        RuntimeError("synthetic forced failure before publication"),
        failure_stage="synthetic_forced_failure",
    )
    quarantine_path = Path(str(quarantine["quarantine_path"]))
    require(not failed_final.exists(), "failed formal output leaked into final path")
    require((quarantine_path / ANALYZER_FAILURE_RECORD).is_file(), "quarantined failure record absent")

    success_final = temp / "analysis_success_final"
    success_staging = create_staged_output(success_final, allowed_parent=temp)
    for name in (*TOP_LEVEL_FILES, *PLOT_FILES):
        (success_staging / name).write_bytes(f"synthetic analyzer artifact:{name}\n".encode("utf-8"))
    publication = publish_staged_output(success_staging, success_final, allowed_parent=temp)
    require(success_final.is_dir() and not success_staging.exists(), "synthetic atomic publication failed")

    formal_audit = validate_invocation([], parse_args([]))
    equal_default_override_rejection = _expect_rejection(
        "formal_equal_default_override",
        lambda: validate_invocation(
            ["--input-dir", str(DEFAULT_INPUT)],
            parse_args(["--input-dir", str(DEFAULT_INPUT)]),
        ),
    )
    mixed_selftest_rejection = _expect_rejection(
        "selftest_formal_path_mixing",
        lambda: validate_invocation(
            ["--self-test", "--output-dir", str(DEFAULT_OUTPUT)],
            parse_args(["--self-test", "--output-dir", str(DEFAULT_OUTPUT)]),
        ),
    )
    original_argv0 = sys.argv[0]
    try:
        sys.argv[0] = str(RUNNER_PATH)
        entrypoint_rejection = _expect_rejection(
            "formal_wrong_entrypoint",
            lambda: validate_invocation([], parse_args([])),
        )
    finally:
        sys.argv[0] = original_argv0
    return {
        "pass": True,
        "failure_quarantine": {
            "final_path_absent": True,
            "failure_record_present": True,
            "quarantine_path": str(quarantine_path),
        },
        "atomic_publication": publication,
        "formal_empty_argv_accepted": formal_audit["pass"],
        "equal_default_override_rejected": equal_default_override_rejection,
        "mixed_selftest_formal_path_rejected": mixed_selftest_rejection,
        "wrong_entrypoint_rejected": entrypoint_rejection,
    }


def _synthetic_full_pipeline_fixture(temp: Path) -> dict[str, Any]:
    rng = np.random.default_rng(26_081_931)
    arrays: dict[str, np.ndarray] = {}
    for rep_index, rep in enumerate(REPRESENTATIONS):
        dimension = 37 if rep == "raw_simple" else 64
        shared = rng.normal(scale=0.35 + 0.01 * rep_index, size=(12, 6, dimension))
        values = np.empty((432, dimension), dtype=np.float64)
        for index, (panel, _, tiling_index, depth, role) in enumerate(canonical_rows()):
            role_index = ROLES.index(role)
            panel_index = PANELS.index(panel)
            vector = shared[depth, role_index].copy()
            vector[:6] += np.eye(6)[role_index] * 3.0
            vector[6] += depth / 11.0 * 3.0
            vector[7] += (depth / 11.0) ** 2 * 2.0
            vector[8] += math.sin(depth / 11.0 * math.pi)
            vector += panel_index * rng.normal(scale=0.03, size=dimension)
            if rep != "raw_simple":
                vector += (tiling_index - 1.5) * rng.normal(scale=0.005, size=dimension)
            values[index] = vector
        if rep == "raw_simple":
            reshaped = values.reshape(3, 2, 12, 6, dimension)
            reshaped[:, 1] = reshaped[:, 0]
            values = reshaped.reshape(432, dimension)
        arrays[rep] = values

    _, averaged = technical_average(arrays)
    unavg_scalers, avg_scalers = source_scalers(arrays, averaged)
    reliability, tiling_bootstrap, _ = build_tiling_reliability(arrays, unavg_scalers)
    matrices = distance_matrices(averaged, avg_scalers)
    rsa, qap, _ = build_rsa(matrices)
    role_predictions, role_metrics, depth_predictions, depth_metrics, probe_audit = build_fixed_probes(averaged)
    role_bootstrap, depth_bootstrap, deltas, delta_bootstrap, bootstrap_audit = build_probe_bootstraps(
        role_predictions,
        depth_predictions,
        role_metrics,
    )
    endpoint_audit = validate_seed_mean_endpoints(reliability, rsa, role_metrics, depth_metrics)
    secondary = build_level_b_secondary(depth_metrics, rsa)
    pca_scores, variance, pca_audit = build_pca_and_variance(arrays, unavg_scalers, temp)
    validity = tuple((f"V{index}", True, "synthetic_full_pipeline") for index in range(1, 9))
    decisions, decision = build_decisions(
        validity,
        reliability,
        tiling_bootstrap,
        rsa,
        role_metrics,
        role_bootstrap,
        depth_metrics,
        depth_bootstrap,
        deltas,
        delta_bootstrap,
        secondary,
    )
    sample_frame = depth_predictions[
        (depth_predictions.representation == "learned_global")
        & (depth_predictions.heldout_panel == PANELS[0])
    ]
    sample_indices = rng.integers(0, 12, size=(20, 12))
    vectorized = _depth_macro_draws(sample_frame, sample_indices)
    scalar = np.asarray(
        [_depth_macro_from_predictions(sample_frame, sample) for sample in sample_indices],
        dtype=np.float64,
    )
    require(np.allclose(vectorized, scalar, atol=1e-14, rtol=1e-14), "vectorized depth stability differs from scalar definition")
    counts = {
        "tiling_reliability": len(reliability),
        "tiling_bootstrap": len(tiling_bootstrap),
        "rsa": len(rsa),
        "qap": len(qap),
        "role_predictions": len(role_predictions),
        "role_metrics": len(role_metrics),
        "role_bootstrap": len(role_bootstrap),
        "depth_predictions": len(depth_predictions),
        "depth_metrics": len(depth_metrics),
        "depth_bootstrap": len(depth_bootstrap),
        "level_b_deltas": len(deltas),
        "level_b_bootstrap": len(delta_bootstrap),
        "level_b_secondary": len(secondary),
        "pca_scores": len(pca_scores),
        "variance": len(variance),
        "decision_cells": len(decisions),
    }
    expected = {
        "tiling_reliability": 42,
        "tiling_bootstrap": 30_000,
        "rsa": 42,
        "qap": 30_000,
        "role_predictions": 1_728,
        "role_metrics": 42,
        "role_bootstrap": 30_000,
        "depth_predictions": 2_592,
        "depth_metrics": 294,
        "depth_bootstrap": 30_000,
        "level_b_deltas": 12,
        "level_b_bootstrap": 30_000,
        "level_b_secondary": 18,
        "pca_scores": 1_296,
        "variance": 36,
        "decision_cells": 51,
    }
    require(counts == expected, f"synthetic full-pipeline count mismatch: {counts}")
    level_b_seeds = {
        bootstrap_audit["streams"][f"level_b__{comparator}"]["panel_streams"][panel]["seed"]
        for comparator in COMPARATORS
        for panel in PANELS
    }
    require(len(level_b_seeds) == 9, "synthetic full pipeline lacks nine Level-B streams")
    for comparator in COMPARATORS:
        rows = delta_bootstrap[delta_bootstrap.comparator == comparator]
        require(len(rows) == N_RESAMPLES, f"Level-B bootstrap row count changed: {comparator}")
        for panel in PANELS:
            expected_seed = stable_seed(
                STABILITY_BASE_SEED,
                "level_b_attention_delta",
                "learned_global",
                panel,
                comparator,
            )
            swapped_order_seed = stable_seed(
                STABILITY_BASE_SEED,
                "level_b_attention_delta",
                "learned_global",
                comparator,
                panel,
            )
            require(
                rows[f"{panel}_stream_seed"].nunique() == 1
                and int(rows[f"{panel}_stream_seed"].iloc[0]) == expected_seed,
                f"Level-B stored panel seed mismatch: {comparator}/{panel}",
            )
            require(
                rows[f"{panel}_stream_label"].nunique() == 1
                and rows[f"{panel}_stream_label"].iloc[0]
                == f"level_b_attention_delta|learned_global|{panel}|{comparator}",
                f"Level-B stored literal label order mismatch: {comparator}/{panel}",
            )
            require(
                expected_seed != swapped_order_seed
                and int(rows[f"{panel}_stream_seed"].iloc[0]) != swapped_order_seed,
                f"Level-B stream silently uses forbidden comparator-before-panel label order: {comparator}/{panel}",
            )
    return {
        "pass": True,
        "counts": counts,
        "all_probes_converged": probe_audit["all_probes_converged"],
        "seed_mean_endpoint_checks": endpoint_audit["arithmetic_seed_mean_checks"],
        "pca_target_fit_rows": pca_audit["target_fit_rows"],
        "depth_vectorized_matches_scalar": True,
        "level_b_distinct_comparator_panel_streams": len(level_b_seeds),
        "level_b_bootstrap_rows_unchanged": len(delta_bootstrap),
        "decision_level_A": decision["level_A_pass"],
        "decision_level_B": decision["level_B_pass"],
    }


def synthetic_self_test(persistent_output: Path | None) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="global_context_analyzer_selftest_") as temporary:
        temp = Path(temporary)

        valid_runner_count_metadata = {
            "expected_counts": EXPECTED_RUNNER_COUNTS,
            "observed_counts": {
                "tiling_manifest_rows": 432,
                "tiling_index_entries": 432,
                "code_manifest_rows": 2_592,
                "latent_code_entries": 2_592,
                "latent_code_rows": 746_496,
                "latent_code_width": 512,
                "raw_simple_rows": 432,
                "countsketch_pairs": 20_480,
                "aggregate_manifest_rows": 5_184,
            },
        }
        _runner_observed_counts(valid_runner_count_metadata)
        width_rejections: dict[str, str] = {}
        for label, bad_width in {
            "legacy_bool_true": True,
            "wrong_511": 511,
            "mixed_list": [511, 512],
            "missing_none": None,
        }.items():
            corrupted = json.loads(json.dumps(valid_runner_count_metadata))
            corrupted["observed_counts"]["latent_code_width"] = bad_width
            width_rejections[label] = _expect_rejection(
                f"runner_latent_width_{label}",
                lambda payload=corrupted: _runner_observed_counts(payload),
            )

        rng = np.random.default_rng(17)
        fp64_values = rng.normal(size=(72, len(RAW_SIMPLE_FEATURES))).astype(np.float64)
        fp64_values[0, :5] = np.asarray(
            [
                0.10000000000000002,
                1.2345678901234567,
                123.45678901234567,
                1.0e-12,
                np.nextafter(1.0, 2.0),
            ],
            dtype=np.float64,
        )
        fp64_csv = temp / "runner_style_fp64_raw_simple.csv"
        with fp64_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RAW_SIMPLE_FEATURES))
            writer.writeheader()
            for row in fp64_values:
                writer.writerow(
                    {
                        name: float(value)
                        for name, value in zip(RAW_SIMPLE_FEATURES, row, strict=True)
                    }
                )
        default_fp64 = pd.read_csv(fp64_csv).loc[:, RAW_SIMPLE_FEATURES].to_numpy(dtype=np.float64)
        roundtrip_fp64 = (
            pd.read_csv(fp64_csv, float_precision="round_trip")
            .loc[:, RAW_SIMPLE_FEATURES]
            .to_numpy(dtype=np.float64)
        )
        default_fp64_mismatches = int(np.count_nonzero(default_fp64.view(np.uint64) != fp64_values.view(np.uint64)))
        require(default_fp64_mismatches > 0, "FP64 CSV fixture does not discriminate the default parser")
        require(np.array_equal(roundtrip_fp64, fp64_values), "round-trip FP64 CSV parser is not bit exact")

        train = rng.normal(size=(144, 24))
        heldout = rng.normal(loc=50, size=(72, 24))
        scaler = fit_scaler(train)
        original_hash = tensor_sha256(scaler["mean"], np.float64)
        corrupted_heldout = heldout * 1_000 + 999
        repeated = fit_scaler(train)
        require(original_hash == tensor_sha256(repeated["mean"], np.float64), "heldout corruption leaked into scaler")
        require(not np.allclose(apply_scaler(heldout, scaler), apply_scaler(corrupted_heldout, scaler)), "heldout corruption fixture ineffective")

        probe_train = rng.normal(size=(96, 32))
        probe_labels = np.tile(np.arange(4), 24)
        probe_train[np.arange(96), probe_labels] += 2.0

        def fit_fixture_probe() -> tuple[str, str, str]:
            local_scaler = fit_scaler(probe_train)
            transformed = apply_scaler(probe_train, local_scaler)
            require_probe_rank(transformed, 16, "synthetic leakage probe")
            local_pca = PCA(n_components=16, svd_solver="full", whiten=False)
            projected = local_pca.fit_transform(transformed)
            classifier = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                fit_intercept=True,
                class_weight=None,
                tol=1e-8,
                max_iter=10_000,
                random_state=BASE_SEED,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(projected, probe_labels)
            require_logistic_convergence(caught, classifier.n_iter_, "synthetic leakage probe")
            return (
                tensor_sha256(local_scaler["mean"], np.float64),
                tensor_sha256(local_pca.components_, np.float64),
                tensor_sha256(classifier.coef_, np.float64),
            )

        probe_hashes_before = fit_fixture_probe()
        # The held-out mutation is deliberately never passed to a fit method.
        corrupted_heldout[:] = rng.normal(loc=-10_000, scale=1_000, size=corrupted_heldout.shape)
        probe_hashes_after = fit_fixture_probe()
        require(probe_hashes_before == probe_hashes_after, "heldout mutation changed scaler/PCA/probe coefficients")

        qap_rng = np.random.Generator(np.random.PCG64(stable_seed(QAP_BASE_SEED, "fixture")))
        qap_hashes = []
        base_matrix = squareform(np.arange(72 * 71 // 2, dtype=np.float64) + 1)
        for _ in range(100):
            permutation = qap_permutation(qap_rng)
            require(np.array_equal(np.sort(permutation), np.arange(72)), "fixture QAP is not bijective")
            joint = base_matrix[permutation][:, permutation]
            require(np.array_equal(joint, base_matrix[np.ix_(permutation, permutation)]), "QAP did not act jointly on rows/columns")
            qap_hashes.append(tensor_sha256(permutation, np.int64))
        require(len(set(qap_hashes)) > 95, "QAP fixture lacks permutation diversity")
        null_fixture = np.linspace(-1.0, 1.0, N_RESAMPLES, dtype=np.float64)
        observed_fixture = 0.25
        qap_p_fixture = float((1 + np.count_nonzero(null_fixture >= observed_fixture)) / (N_RESAMPLES + 1))
        require(qap_p_fixture == (1 + 3_750) / 10_001, "QAP plus-one p-value fixture mismatch")

        seed_values = np.array([0.2, 0.5, 0.8])
        require(float(seed_values.mean()) == 0.5, "untrained endpoint is not arithmetic seed mean")
        sketch_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        require(float(sketch_values.mean()) == 0.3, "CountSketch endpoint is not arithmetic seed mean")
        countsketch_hashes = []
        for seed in COUNTSKETCH_SEEDS:
            buckets, signs = derive_countsketch_map(seed)
            require(buckets.shape == signs.shape == (4_096,), "CountSketch fixture map shape mismatch")
            countsketch_hashes.append(
                {
                    "seed": seed,
                    "bucket_sha256": tensor_sha256(buckets, np.int64),
                    "sign_sha256": tensor_sha256(signs, np.int64),
                }
            )
        old_cache_names = {
            "factorial_code_cache_seed_26081601.pt",
            "factorial_code_cache_seed_26081602.pt",
            "correct_latent_codes.pt",
        }
        require(old_cache_names.isdisjoint(RUNNER_FILES), "old code cache entered formal runner artifact allowlist")

        tile_codes = rng.normal(size=(144, 512)).astype(np.float32)
        aggregate = aggregate_tiles(tile_codes)
        require(aggregate.shape == (2_560,) and aggregate.dtype == np.float64, "aggregate fixture shape/dtype failure")
        code_hash = tensor_sha256(tile_codes, np.float32)
        corrupted_codes = tile_codes.copy()
        corrupted_codes[0, 0] += 1
        require(tensor_sha256(corrupted_codes, np.float32) != code_hash, "code corruption hash was not detected")

        rejection_fixtures = {
            "zero_reliability_denominator": _expect_rejection(
                "zero_reliability_denominator",
                lambda: reliability_ratio(np.ones(12), np.zeros(12), "synthetic"),
            ),
            "constant_rsa": _expect_rejection(
                "constant_rsa",
                lambda: _spearman(np.ones(12), np.arange(12, dtype=np.float64)),
            ),
            "nonfinite_feature": _expect_rejection(
                "nonfinite_feature",
                lambda: ensure_finite(np.array([0.0, np.nan]), "synthetic"),
            ),
            "insufficient_rank": _expect_rejection(
                "insufficient_rank",
                lambda: require_probe_rank(np.ones((24, 16)), 8, "synthetic"),
            ),
            "missing_role_class": _expect_rejection(
                "missing_role_class",
                lambda: require_role_classes(np.array([0, 1, 2] * 4), "synthetic"),
            ),
            "nonconvergence": _expect_rejection(
                "nonconvergence",
                lambda: require_logistic_convergence([], np.array([10_000]), "synthetic"),
            ),
            "csv_schema_extra_column": _expect_rejection(
                "csv_schema_extra_column",
                lambda: exact_columns(
                    pd.DataFrame(columns=[*FEATURE_COLUMNS, "unexpected"]),
                    FEATURE_COLUMNS,
                    "synthetic feature manifest",
                ),
            ),
        }

        synthetic_variance = np.zeros((3, 2, 72, 4), dtype=np.float64)
        cell_signal = np.arange(72, dtype=np.float64) - 35.5
        panel_signal = np.array([-1.0, 0.0, 1.0])
        interaction_cell = np.tile(np.array([-1.0, 1.0]), 36)
        tiling_signal = np.array([-2.0, 2.0])
        synthetic_variance[:, :, :, 0] += cell_signal[None, None, :]
        synthetic_variance[:, :, :, 1] += panel_signal[:, None, None]
        synthetic_variance[:, :, :, 2] += panel_signal[:, None, None] * interaction_cell[None, None, :]
        synthetic_variance[:, :, :, 3] += tiling_signal[None, :, None]
        variance_result = variance_components(synthetic_variance)
        expected_ss = {
            "shared_role_depth_cell": 6.0 * float(np.square(cell_signal).sum()),
            "checkpoint_main": 144.0 * float(np.square(panel_signal).sum()),
            "checkpoint_by_cell_interaction": 2.0 * float(np.square(panel_signal[:, None] * interaction_cell[None, :]).sum()),
            "tiling_residual": float(3 * 72 * np.square(tiling_signal).sum()),
        }
        expected_total = sum(expected_ss.values())
        expected_fractions = {key: value / expected_total for key, value in expected_ss.items()}
        require(
            all(math.isclose(variance_result[key], expected_fractions[key], rel_tol=1e-13, abs_tol=1e-13) for key in expected_fractions),
            "known balanced-SS decomposition fixture mismatch",
        )

        validity = tuple((f"V{index}", True, "synthetic") for index in range(1, 9))
        reliability = pd.DataFrame([{"representation": "learned_global", "panel_id": panel, "tiling_ratio": 0.5} for panel in PANELS])
        tiling_boot = pd.DataFrame([{"panel_id": panel, "tiling_ratio": 0.6} for panel in PANELS for _ in range(20)])
        rsa = pd.DataFrame([{"representation": "learned_global", "panel_left": left, "panel_right": right, "rsa_rho": 0.5, "qap_p_one_sided": 0.001} for left, right in ((PANELS[0], PANELS[1]), (PANELS[0], PANELS[2]), (PANELS[1], PANELS[2]))])
        role_metrics = pd.DataFrame([{"representation": "learned_global", "heldout_panel": panel, "balanced_accuracy": 0.6} for panel in PANELS])
        role_boot = pd.DataFrame([{"panel_id": panel, "balanced_accuracy": 0.55} for panel in PANELS for _ in range(20)])
        depth_metrics = pd.DataFrame([{"representation": "learned_global", "heldout_panel": panel, "metric_level": "panel_macro", "spearman_rho": 0.6} for panel in PANELS])
        depth_boot = pd.DataFrame([{"panel_id": panel, "macro_spearman_rho": 0.5} for panel in PANELS for _ in range(20)])
        delta_rows = []
        for comparator in COMPARATORS:
            for panel in PANELS:
                delta_rows.append({"comparator": comparator, "scope": "checkpoint", "heldout_panel": panel, "delta": 0.1})
            delta_rows.append({"comparator": comparator, "scope": "checkpoint_macro", "heldout_panel": "macro", "delta": 0.1})
        deltas = pd.DataFrame(delta_rows)
        delta_boot = pd.DataFrame([{"comparator": comparator, "delta": 0.08} for comparator in COMPARATORS for _ in range(20)])
        secondary = pd.DataFrame(
            [
                {"comparator": comparator, "metric": metric, "panel_or_pair": scope, "delta": 0.0, "pass": True}
                for comparator in COMPARATORS
                for metric, scopes in (
                    ("depth_macro_spearman", PANELS),
                    ("cross_checkpoint_rsa", ("a__b", "a__c", "b__c")),
                )
                for scope in scopes
            ]
        )
        cells, decision = build_decisions(validity, reliability, tiling_boot, rsa, role_metrics, role_boot, depth_metrics, depth_boot, deltas, delta_boot, secondary)
        require(len(cells) == 51 and decision["level_A_pass"] and decision["level_B_pass"], "synthetic passing decision failed")
        corrupted_secondary = secondary.copy()
        corrupted_secondary.loc[0, "pass"] = False
        _, failed = build_decisions(validity, reliability, tiling_boot, rsa, role_metrics, role_boot, depth_metrics, depth_boot, deltas, delta_boot, corrupted_secondary)
        require(not failed["level_B_pass"], "decision corruption did not fail Level B")

        manifest_dir = temp / "runner_manifest"
        manifest_dir.mkdir()
        for name in RUNNER_FILES:
            (manifest_dir / name).write_bytes(f"synthetic:{name}\n".encode("utf-8"))
        manifest_payload = {
            "schema_version": "global_context_latent_geometry_raw_artifact_manifest_v1",
            "frozen_design_sha256": DESIGN_SHA256,
            "runner_sha256": "1" * 64,
            "analyzer_sha256": "2" * 64,
            "external_contract_sha256": "3" * 64,
            "self_excluded": True,
            "files": [
                {"path": name, "sha256": sha256_file(manifest_dir / name), "bytes": (manifest_dir / name).stat().st_size}
                for name in RUNNER_FILES
            ],
            "file_count": 21,
        }
        json_write(manifest_dir / "artifact_manifest.json", manifest_payload)
        manifest_valid = validate_artifact_manifest(manifest_dir)
        bad_schema_manifest = json.loads(json.dumps(manifest_payload))
        bad_schema_manifest["unexpected_key"] = True
        json_write(manifest_dir / "artifact_manifest.json", bad_schema_manifest)
        manifest_schema_rejection = _expect_rejection("manifest_exact_key_corruption", lambda: validate_artifact_manifest(manifest_dir))
        json_write(manifest_dir / "artifact_manifest.json", manifest_payload)
        (manifest_dir / RUNNER_FILES[0]).write_bytes(b"corrupted\n")
        manifest_rejection = _expect_rejection("manifest_hash_corruption", lambda: validate_artifact_manifest(manifest_dir))

        symlink_dir = temp / "runner_manifest_symlink"
        symlink_dir.mkdir()
        for name in RUNNER_FILES:
            (symlink_dir / name).write_bytes(f"synthetic:{name}\n".encode("utf-8"))
        (symlink_dir / "run.log").unlink()
        (symlink_dir / "run.log").symlink_to("resolved_config.json")
        symlink_manifest = {
            "schema_version": "global_context_latent_geometry_raw_artifact_manifest_v1",
            "frozen_design_sha256": DESIGN_SHA256,
            "runner_sha256": "1" * 64,
            "analyzer_sha256": "2" * 64,
            "external_contract_sha256": "3" * 64,
            "self_excluded": True,
            "files": [
                {"path": name, "sha256": sha256_file(symlink_dir / name), "bytes": (symlink_dir / name).stat().st_size}
                for name in RUNNER_FILES
            ],
            "file_count": 21,
        }
        json_write(symlink_dir / "artifact_manifest.json", symlink_manifest)
        manifest_symlink_rejection = _expect_rejection("manifest_symlink", lambda: validate_artifact_manifest(symlink_dir))
        contract_binding_audit = _synthetic_contract_binding_test(temp)
        tiling_validation_audit = _synthetic_tiling_validation_test(temp)
        transaction_invocation_audit = _synthetic_transaction_and_invocation_test(temp)
        full_pipeline_audit = _synthetic_full_pipeline_fixture(temp)

        plot_dir = persistent_output if persistent_output is not None else temp / "plots"
        if persistent_output is None:
            plot_dir.mkdir()
        plot_inputs = _synthetic_plot_inputs()
        plot_audit = render_plots(plot_dir, *plot_inputs)
        require(plot_audit["pass"] and len(plot_audit["plots"]) == 11, "synthetic plot creation failed")

        result = {
            "status": "PASS",
            "scope": "synthetic_only_no_formal_runner_outcomes_no_weight_ae",
            "elapsed_seconds": time.monotonic() - started,
            "frozen_design_sha256": DESIGN_SHA256,
            "analyzer_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "scaler_scope": {
                "train_rows": 144,
                "heldout_fit_rows": 0,
                "heldout_corruption_did_not_change_scaler": True,
                "heldout_corruption_did_not_change_pca_or_probe_coefficients": True,
                "fit_hashes": probe_hashes_before,
            },
            "qap": {
                "bijections_checked": 100,
                "joint_row_column_action": True,
                "unique_permutations": len(set(qap_hashes)),
                "plus_one_p_value_recomputed": qap_p_fixture,
            },
            "seed_means": {
                "untrained_three_seed_mean": 0.5,
                "countsketch_five_seed_mean": 0.3,
                "best_seed_selection": False,
                "countsketch_maps_independently_derived": countsketch_hashes,
            },
            "feature_code_audit": {"aggregate_dimension": 2_560, "code_hash_corruption_detected": True},
            "runner_latent_code_width_contract": {
                "valid_numeric_512_accepted": True,
                "corruptions_rejected": width_rejections,
                "pass": True,
            },
            "runner_fp64_csv_roundtrip": {
                "writer": "csv.DictWriter_python_float_repr",
                "rows": 72,
                "columns": len(RAW_SIMPLE_FEATURES),
                "default_parser_bit_mismatches": default_fp64_mismatches,
                "round_trip_parser_bit_mismatches": 0,
                "round_trip_bit_exact": True,
            },
            "old_cache_exclusion": {
                "pass": True,
                "old_cell_code_seed_files_in_runner_allowlist": False,
                "old_tiling_index_file_in_runner_allowlist": False,
            },
            "fail_closed_rejections": rejection_fixtures,
            "variance_decomposition": {
                "contract": VARIANCE_DECOMPOSITION_CONTRACT,
                "known_fractions": variance_result,
                "expected_fractions": expected_fractions,
                "orthogonal_complete": True,
            },
            "decisions": {"passing_cells": len(cells), "passing_level_A": True, "passing_level_B": True, "corrupted_level_B_rejected": True},
            "manifest": {
                "valid_count": manifest_valid["count"],
                "hash_corruption_rejected": manifest_rejection,
                "exact_key_corruption_rejected": manifest_schema_rejection,
                "symlink_rejected": manifest_symlink_rejection,
            },
            "contract_binding": contract_binding_audit,
            "exact_tiling_validation": tiling_validation_audit,
            "transaction_and_invocation": transaction_invocation_audit,
            "full_pipeline": full_pipeline_audit,
            "plots": {"count": plot_audit["count"], "mechanical_pass": plot_audit["pass"]},
            "analyzer_isolation": assert_analyzer_isolation(),
        }
        if persistent_output is not None:
            json_write(plot_dir / "self_test_result.json", result)
            (plot_dir / "self_test.log").write_text(
                "status=PASS\nscope=synthetic_only_no_formal_runner_outcomes_no_weight_ae\n"
                f"analyzer_sha256={result['analyzer_sha256']}\n",
                encoding="utf-8",
            )
            json_write(plot_dir / "artifact_manifest.json", artifact_manifest(plot_dir))
            result["persistent_output"] = str(plot_dir)
            result["persistent_manifest_sha256"] = sha256_file(plot_dir / "artifact_manifest.json")
        return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    invocation_audit = validate_invocation(argv, args)
    if args.self_test:
        output: Path | None = None
        if args.self_test_output_dir is not None:
            candidate = args.self_test_output_dir
            parent = candidate.parent.resolve(strict=True)
            require(parent == ARTIFACT_ROOT, "self-test output parent must be artifact root")
            require(not candidate.exists() and not candidate.is_symlink(), "self-test output must be fresh")
            candidate.mkdir()
            output = candidate.resolve(strict=True)
        result = synthetic_self_test(output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    run_formal(args, invocation_audit)


if __name__ == "__main__":
    main()
