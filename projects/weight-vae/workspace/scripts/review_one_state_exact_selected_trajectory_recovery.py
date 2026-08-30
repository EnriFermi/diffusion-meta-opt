from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import stat
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
RECOVERY_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_recovery_v1"
REVIEW_PROTOCOL_ID = (
    "one_state_exact_selected_relaxed_cancellation_recovery_cpu_review_v1"
)
SOURCE_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_v1"
SOURCE_STAGING = OUTPUT_ROOT / "postgoal_relaxed_cancellation_production.incomplete"
DEFAULT_RECOVERY_OUTPUT = (
    OUTPUT_ROOT / "postgoal_relaxed_cancellation_recovery_finalization_v1"
)
RECOVERY_PROTOCOL = OUTPUT_ROOT / "postgoal_relaxed_cancellation/recovery_protocol.md"
FROZEN_INPUT_MANIFEST = (
    OUTPUT_ROOT / "postgoal_relaxed_cancellation/recovery_frozen_input_manifest.json"
)
RECOVERY_TASK_MANIFEST = (
    OUTPUT_ROOT / "postgoal_relaxed_cancellation/recovery_task_manifest.json"
)
RECOVERY_IMPORT_MANIFEST = (
    OUTPUT_ROOT / "postgoal_relaxed_cancellation/recovery_import_manifest.json"
)
ACCEPTED_RUN_DIR = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
ACCEPTED_VAE_CHECKPOINT = ACCEPTED_RUN_DIR / "vae_checkpoint.pt"
PARENT_PROGRESS = (
    OUTPUT_ROOT
    / "iteration6_exact_selected_trajectory_production/progress_checkpoint.pt"
)
PARENT_FINAL = (
    OUTPUT_ROOT / "iteration6_exact_selected_trajectory_production/final_checkpoint.pt"
)
PARENT_REVIEW = (
    OUTPUT_ROOT
    / "iteration6_exact_selected_trajectory_production/independent_review.json"
)

EXPECTED_FROZEN_INPUT_MANIFEST_SHA256 = (
    "8d7f5a72092de0fbaffe67a1c2421dfcb227bf412690f470d552ab0069b99f66"
)
EXPECTED_RECOVERY_PROTOCOL_SHA256 = (
    "e7f941d8a6632e71ffd09ad9c0c5a3ed11490a3f8fadd3d7f9b1f8001d5108c1"
)
EXPECTED_RECOVERY_TASK_MANIFEST_SHA256 = (
    "a4b0b4bc1b50d04e2cd6d6ab35fe05a8200caa222cf84228eaee894504a8054b"
)
EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256 = (
    "976d35e42c4fa491324660e044a4bb6ac224786108e2e81effcef5168753da10"
)
EXPECTED_SOURCE_PROGRESS_SHA256 = (
    "ab3366870d62beb71b4d59fd2d54e3cb7354963a16e17a8b9cc13527a4d25311"
)
EXPECTED_SOURCE_PROTOCOL_SHA256 = (
    "870272310940873a8fb939434979cbef3b42d1a7c68ebde127afe0dfe8453efb"
)
EXPECTED_SOURCE_DEPENDENCY_MANIFEST_SHA256 = (
    "9437b54d9a6379adaab9a7e2bd578caf55e3c879bf861f8a0150f124b577eb32"
)
EXPECTED_PRODUCER_SHA256 = (
    "d79462816ff95b2c3f1a94bed06f8a2c12d6a25610dd9385d9361da2e970f39f"
)
EXPECTED_CONTINUATION_REVIEWER_SHA256 = (
    "57ae23d336cc348548b744c90e41c1560c5867af512bf4a5b635fc344737adbb"
)
EXPECTED_ACTIVE_PARAMETER_HASH = (
    "b4d22a52fdb07a51978d466ef0841bfa92b33a625c84e32433ccb6b334039427"
)
EXPECTED_TRANSITION_CHAIN_SHA256 = (
    "8ebc37fa89b471a722844d553b2a5a70003cd44b3cda3ce987075a6df8e261fe"
)
EXPECTED_PARENT_FINAL_SHA256 = (
    "7063f393332f2640d599fa41928006f74342c3d681dd9b77a575e5e9e40716e5"
)
EXPECTED_PARENT_PROGRESS_SHA256 = (
    "010d34d51c21ec139fca3e93f51b91c7563f43a524a59b521dcec50de9c752ce"
)
EXPECTED_PARENT_REVIEW_SHA256 = (
    "7677a64b38755bed2d553996f103d943ac7d43c7dae748f546e93ba5441b9703"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_FINALIZER_SHA256 = (
    "5407f712f46090eeeed004a76dea1ea10b250398171575586c1c6737b145b325"
)
EXPECTED_FINALIZER_NORMALIZED_SHA256 = (
    "e25be1074055a66ad68c63acc1f722191efbbf8a827ecd73adfeaa090329a599"
)
EXPECTED_ACCEPTED_RUN_CONFIG_SHA256 = (
    "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4"
)
EXPECTED_ACCEPTED_RUN_WEIGHT_POOL_SHA256 = (
    "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef"
)
EXPECTED_ACCEPTED_RUN_RECORDS_SHA256 = (
    "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933"
)
EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
EXPECTED_ACCEPTED_RUN_INPUT_SHA256 = {
    "config.json": EXPECTED_ACCEPTED_RUN_CONFIG_SHA256,
    "weight_pool.pt": EXPECTED_ACCEPTED_RUN_WEIGHT_POOL_SHA256,
    "weight_pool_records.csv": EXPECTED_ACCEPTED_RUN_RECORDS_SHA256,
    "vae_checkpoint.pt": EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
}

DIMENSION = 512
START_UPDATE = 17
MAX_ACCEPTED_UPDATES = 100
NEW_ACCEPTED_UPDATES = 83
ACTIVE_TENSOR_COUNT = 31
ACTIVE_PARAMETER_COUNT = 11_685_120
SOURCE_WEIGHT_INDEX = 378
EXPECTED_TASK_NAME = "fashion_mnist"
EXPECTED_TASK_TAU = 1.1614345407370807
EXPECTED_RUNTIME = {
    "python": "3.12.12",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "torchvision": "0.25.0+cu128",
}
EXPECTED_TASK_SOURCE_DEPENDENCIES = {
    "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/config.py": (
        "aa71d2909f4ac7cde9b7f6fdb2d0caff8bfd3795f2fdea94c1be3e2fe853675e"
    ),
    "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py": (
        "ff67cc37006f5678d67ebaa919d348161d0b030beae95c1e3701ec07da35a43a"
    ),
    "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/pipeline.py": (
        "9cf6c56aa3725920d70926a79769789b8e4d89e8a47d96abd335e84e9550832c"
    ),
}
EXPECTED_TASK_TENSORS = {
    "train_images": {
        "dtype": "torch.float32",
        "shape": [16_384, 1, 16, 16],
        "sha256": "98d8b2c338d98439c570982e78a0856c63a95ce3ee179cdc9fa299efbf7c69f4",
    },
    "train_labels": {
        "dtype": "torch.int64",
        "shape": [16_384],
        "sha256": "1316df8b5cab69fd3601a0cd76df1761d64fddbad8a477824c3e2981701d3a97",
    },
    "test_images": {
        "dtype": "torch.float32",
        "shape": [4_096, 1, 16, 16],
        "sha256": "570d06ea17ffa18e87c287d895c3e5689fbc24b8c855b88acd5ef6036adc57c0",
    },
    "test_labels": {
        "dtype": "torch.int64",
        "shape": [4_096],
        "sha256": "041fefaaf3e04affd53ca0f0bd6497a52ad31af1e9bbcea1da0639250e8e6f6c",
    },
}
RECOVERY_RADIUS_ATOL = 5e-9
ORIGINAL_RADIUS_ATOL = 1e-9
REPLAY_ATOL = 1e-9
GEOMETRY_EPSILON = 1e-4
GEOMETRY_OLD_BETA = 22.536727828943093
GEOMETRY_LOW_THRESHOLD = 0.1
EXPECTED_MAX_RADIUS_ERROR = 2.7241507938313703e-9
EXPECTED_ABOVE_ORIGINAL_RADIUS_COUNT = 37
EXPECTED_ORIGINAL_FAILED_GATES = ["continuation_diagnostics_recompute"]
FORBIDDEN_SOURCE_FINALIZATION_FILES = {
    "artifact_manifest.json",
    "decision.json",
    "final_checkpoint.pt",
    "FINALIZED.json",
}

# The finalizer excludes the manifest, FINALIZED marker, and run log from its
# hash payload to avoid cycles and permit the final publication line in the log.
EXPECTED_MANIFESTED_ARTIFACTS = {
    "arm_selection.csv",
    "decision.json",
    "executed_recovery_finalizer_source_snapshot.py",
    "final_checkpoint.pt",
    "frozen_continuation_reviewer_snapshot.py",
    "frozen_executed_producer_snapshot.py",
    "frozen_producer_dependency_manifest_snapshot.json",
    "frozen_producer_protocol_snapshot.md",
    "historical_transition_audit.csv",
    "line_search.csv",
    "proposal_diagnostics.csv",
    "recovery_audit.json",
    "recovery_direction_radius_audit.png",
    "recovery_frozen_input_manifest_snapshot.json",
    "recovery_import_manifest_snapshot.json",
    "recovery_protocol_snapshot.md",
    "recovery_task_manifest_snapshot.json",
    "recovery_spectra.png",
    "recovery_trajectory.png",
    "replayed_final_geometry.pt",
    "replayed_final_spectrum.csv",
    "resolved_config.json",
    "source_intervention_origin.json",
    "source_intervention_preflight.json",
    "source_progress_checkpoint.pt",
    "source_resolved_config.json",
    "source_run.log",
    "state_metrics.csv",
    "state_spectra.csv",
}
EXPECTED_UNMANIFESTED_ARTIFACTS = {
    "artifact_manifest.json",
    "FINALIZED.json",
    "run.log",
}
EXPECTED_RECOVERY_FILE_SET = (
    EXPECTED_MANIFESTED_ARTIFACTS | EXPECTED_UNMANIFESTED_ARTIFACTS
)

EXPECTED_MANIFEST_KEYS = {
    "protocol_id",
    "recovery_finalizer_source_sha256",
    "recovery_finalizer_normalized_source_sha256",
    "frozen_input_manifest_sha256",
    "recovery_task_manifest_sha256",
    "recovery_import_manifest_sha256",
    "source_progress_checkpoint_sha256",
    "artifacts",
}
EXPECTED_FINALIZED_KEYS = {
    "protocol_id",
    "source_protocol_id",
    "status",
    "recovery_valid",
    "scientific_success",
    "accepted_updates",
    "new_accepted_updates",
    "termination",
    "source_progress_checkpoint_sha256",
    "accepted_vae_checkpoint_sha256",
    "recovery_task_manifest_sha256",
    "recovery_import_manifest_sha256",
    "recovery_finalizer_source_sha256",
    "recovery_finalizer_normalized_source_sha256",
    "decision_sha256",
    "recovery_audit_sha256",
    "artifact_manifest_sha256",
    "final_checkpoint_sha256",
}
EXPECTED_ORIGINAL_AUDIT_KEYS = {
    "pass",
    "failed_gates",
    "exception_type",
    "exception_message",
    "audit_invocations",
    "row_gates",
    "all_unmodified_row_gates_pass",
    "transition_chain_sha256",
}
EXPECTED_RECOVERY_AUDIT_KEYS = {
    "protocol_id",
    "source_protocol_id",
    "recovery_valid",
    "original_runner_audit_pass",
    "original_runner_failed_gates",
    "original_runner_audit",
    "independent_recovery_audit",
    "checkpoint_lineage",
    "geometry_execution",
    "geometry_reconstruction",
    "reconstruction_input_files",
    "final_geometry_replay",
    "immutable_input_preflight",
    "immutable_input_postflight",
    "dependency_import",
    "task_manifest_preflight",
    "task_manifest_postflight",
    "import_manifest_preflight",
    "import_manifest_postflight",
    "runtime_import_closure",
    "startup_self_freeze",
    "copied_self_freeze",
    "recovery_validity_gates",
}
EXPECTED_GEOMETRY_EXECUTION_KEYS = {
    "geometry_evaluation_count",
    "forbidden_operation_attempts",
    "optimization_gradient_evaluations",
    "proposals",
    "line_searches",
    "parameter_updates",
}
EXPECTED_GEOMETRY_RECONSTRUCTION_KEYS = {
    "accepted_checkpoint_sha256",
    "expected_accepted_checkpoint_sha256",
    "z_sha256",
    "active_tensor_count",
    "active_parameter_count",
    "installed_active_parameter_hash",
    "post_replay_active_parameter_hash",
    "base_parameter_gradient_slots",
    "post_replay_parameter_gradient_slots",
    "model_installation_audit",
    "model_replay_nonmutation_audit",
    "model_tensor_snapshots",
    "runtime_provenance",
    "reconstruction_input_files",
    "task_at_load",
    "task_immediately_pre_replay",
    "task_post_replay",
    "full_ce_batch",
}
EXPECTED_REPLAY_KEYS = {
    "pass",
    "metric_absolute_tolerance",
    "metric_errors",
    "metric_max_absolute_error",
    "metric_max_error_name",
    "spectrum_absolute_tolerance",
    "spectrum_eigenvalue_count",
    "spectrum_max_absolute_error",
    "spectrum_max_error_rank",
    "replay_low_basis_hash_matches",
    "hessian_shape",
    "matrix_shape",
    "spectrum_shape",
    "low_basis_shape",
    "low_projector_shape",
    "hessian_sha256",
    "matrix_sha256",
    "spectrum_tensor_sha256",
    "low_basis_sha256",
    "low_projector_sha256",
    "active_parameter_hash_matches",
    "replayed_final_spectrum_sha256",
    "replayed_final_spectrum_rows",
    "geometry_artifact_sha256",
    "geometry_artifact_audit",
}
# The finalizer calls this field ``low_basis_hash_matches``.
EXPECTED_REPLAY_KEYS.remove("replay_low_basis_hash_matches")
EXPECTED_REPLAY_KEYS.add("low_basis_hash_matches")
EXPECTED_FINAL_CHECKPOINT_KEYS = {
    "protocol_id",
    "source_protocol_id",
    "source_progress_checkpoint_sha256",
    "parent_checkpoint_sha256",
    "parent_progress_checkpoint_sha256",
    "selected_arm",
    "accepted_updates",
    "new_accepted_updates",
    "termination",
    "transition_chain_sha256",
    "active_parameter_hash",
    "active_model_state",
    "source_weight_index",
    "z_sha256",
    "accepted_vae_checkpoint_sha256",
    "stored_state_metrics",
}
EXPECTED_DECISION_KEYS = {
    "protocol_id",
    "source_protocol_id",
    "recovery_valid",
    "scientific_success",
    "recovery_validity_gates",
    "accepted_updates",
    "new_accepted_updates",
    "selected_arm",
    "termination",
    "original_runner_audit_pass",
    "original_runner_failed_gates",
    "recovery_radius_audit",
    "outcome",
    "final_metrics",
    "final_geometry_replay",
    "final_checkpoint_sha256",
    "accepted_vae_checkpoint_sha256",
    "recovery_task_manifest_sha256",
    "recovery_import_manifest_sha256",
    "recovery_audit_sha256",
    "elapsed_sec",
}
EXPECTED_CONFIG_KEYS = {
    "protocol_id",
    "source_protocol_id",
    "device",
    "dtype",
    "seed",
    "cache_mode",
    "source_staging",
    "source_staging_mutation_allowed",
    "output_dir",
    "working_dir",
    "source_progress_checkpoint_sha256",
    "frozen_input_manifest_sha256",
    "recovery_protocol_sha256",
    "recovery_task_manifest_sha256",
    "recovery_import_manifest_sha256",
    "producer_source_sha256",
    "continuation_reviewer_source_sha256",
    "finalizer_source_sha256",
    "finalizer_normalized_source_sha256",
    "startup_self_freeze",
    "copied_self_freeze",
    "task_manifest_preflight",
    "import_manifest_preflight",
    "copied_task_manifest_audit",
    "copied_import_manifest_audit",
    "expected_runtime",
    "accepted_run_dir",
    "accepted_run_input_sha256",
    "geometry_evaluation_budget",
    "optimization_gradient_evaluation_budget",
    "proposal_budget",
    "line_search_budget",
    "parameter_update_budget",
    "metric_replay_absolute_tolerance",
    "spectrum_replay_absolute_tolerance",
    "direction_radius_absolute_tolerance",
    "direction_radius_relative_tolerance",
    "snapshot_hashes",
}
EXPECTED_RECOVERY_VALIDITY_GATES = {
    "immutable_input_preflight",
    "immutable_input_postflight",
    "startup_finalizer_self_freeze",
    "copied_finalizer_self_freeze",
    "task_manifest_pre_post_exact",
    "import_manifest_pre_post_exact",
    "runtime_import_closure_exact",
    "frozen_dependencies_verified_before_import",
    "loaded_producer_equals_executed_snapshot",
    "terminal_progress_lineage",
    "original_runner_expected_failure_observed",
    "all_unmodified_original_row_gates_pass",
    "independent_recovery_audit_pass",
    "recovery_radius_audit_pass",
    "checkpoint_cpu_bitwise_lineage_pass",
    "runtime_versions_exact",
    "runtime_task_tensors_exact_immediately_before_replay",
    "runtime_task_tensors_unchanged_after_replay",
    "accepted_checkpoint_exact",
    "reconstruction_inputs_exact_pre_post_load_run",
    "terminal_install_changes_exact_active_set",
    "all_model_tensors_unchanged_during_replay",
    "exactly_one_geometry_evaluation",
    "no_optimization_operations",
    "final_geometry_replay_pass",
    "geometry_artifact_independent_closure_pass",
    "replayed_spectrum_csv_pass",
    "scientific_success_criteria_unchanged",
}
EXPECTED_SUCCESS_GATES = {
    "one_hundred_accepted_updates",
    "tail_bulk_activity",
    "final_a_at_most_0p90",
    "accepted_a_strictly_decreases",
    "all_historical_a_armijo_transitions",
    "final_b_below_initial",
    "non_top_only_fraction_at_least_0p25",
    "final_p50_above_initial",
    "final_effective_rank_above_initial",
    "final_count_lt_1e_4_strictly_lower",
    "final_count_lt_1e_2_strictly_lower",
    "final_count_lt_0p1_strictly_lower",
    "final_mmax_not_above_initial",
    "high_tail_never_increases_beyond_floor",
    "all_state_values_finite",
    "all_spectrum_values_finite",
    "all_raw_spectrum_minima_valid",
    "all_exact_a_closures_valid",
    "final_checkpoint_replay",
}


class ReviewError(RuntimeError):
    pass


class Review:
    def __init__(self) -> None:
        self.gates: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.errors: list[str] = []
        self.limitations: list[str] = []

    def phase(self, name: str, callback: Callable[[], Any]) -> Any | None:
        try:
            value = callback()
        except Exception as error:
            self.gates[f"{name}_completed"] = False
            self.errors.append(f"{name}: {type(error).__name__}: {error}")
            self.details[f"{name}_traceback"] = traceback.format_exc(limit=12)
            return None
        self.gates[f"{name}_completed"] = True
        return value


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ReviewError(message)


def _sha256_file(path: Path) -> str:
    path = _require_regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    _require(isinstance(tensor, torch.Tensor), "fingerprint value is not a tensor")
    value = tensor.detach().cpu().contiguous()
    _require(bool(torch.isfinite(value).all()), "fingerprint tensor is nonfinite")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": value.numel(),
        "sha256": _sha256_tensor(value),
    }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _absolute_without_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_components(
    path: Path, *, allow_missing_leaf: bool = False
) -> Path:
    absolute = _absolute_without_resolve(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        is_leaf = index == len(parts) - 1
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and is_leaf:
                return absolute
            raise ReviewError(f"path component does not exist: {current}") from None
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"symlink path component is forbidden: {current}",
        )
    return absolute


def _require_regular_file(path: Path) -> Path:
    absolute = _require_no_symlink_components(path)
    _require(
        stat.S_ISREG(os.lstat(absolute).st_mode),
        f"expected regular non-symlink file: {absolute}",
    )
    return absolute


def _require_directory(path: Path) -> Path:
    absolute = _require_no_symlink_components(path)
    _require(
        stat.S_ISDIR(os.lstat(absolute).st_mode),
        f"expected non-symlink directory: {absolute}",
    )
    return absolute


def _load_json(path: Path) -> dict[str, Any]:
    path = _require_regular_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} is not a JSON object")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    path = _require_regular_file(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, dict), f"{path.name} is not a checkpoint mapping")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    path = _require_regular_file(path)
    _require(path.stat().st_size > 0, f"{path.name} is empty")
    return pd.read_csv(path)


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ReviewError(f"nonfinite numeric value: {value!r}")
    return result


def _int(value: Any) -> int:
    number = _float(value)
    result = int(number)
    _require(number == float(result), f"nonintegral value: {value!r}")
    return result


def _close(left: Any, right: Any, *, atol: float, rtol: float = 0.0) -> bool:
    try:
        return bool(
            np.isclose(float(left), float(right), atol=atol, rtol=rtol, equal_nan=False)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _equivalent(
    left: Any,
    right: Any,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return (
            isinstance(left, (bool, np.bool_))
            and isinstance(right, (bool, np.bool_))
            and bool(left) == bool(right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    left_sequence = isinstance(left, Sequence) and not isinstance(left, (str, bytes))
    right_sequence = isinstance(right, Sequence) and not isinstance(right, (str, bytes))
    if left_sequence or right_sequence:
        if not left_sequence or not right_sequence or len(left) != len(right):
            return False
        return all(
            _equivalent(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    return _close(left, right, atol=atol, rtol=rtol)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        _absolute_without_resolve(path).relative_to(_absolute_without_resolve(parent))
    except ValueError:
        return False
    return True


def _regular_file_names(directory: Path) -> set[str]:
    directory = _require_directory(directory)
    entries = list(directory.iterdir())
    for path in entries:
        _require_regular_file(path)
    return {path.name for path in entries}


def _tree_fingerprint(directory: Path) -> str:
    directory = _require_directory(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        _require_regular_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_only_fingerprint(path: Path) -> dict[str, Any]:
    path = _absolute_without_resolve(path)
    if not os.path.lexists(path):
        return {"exists": False}
    try:
        _require_no_symlink_components(path)
    except ReviewError as error:
        return {"exists": True, "path_error": str(error)}
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        return {
            "exists": True,
            "is_directory": False,
            "is_symlink": False,
        }
    try:
        return {"exists": True, "tree_sha256": _tree_fingerprint(path)}
    except Exception as error:
        return {
            "exists": True,
            "fingerprint_error": f"{type(error).__name__}: {error}",
        }


def _named_tensor_hash(mapping: Mapping[str, torch.Tensor]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name in sorted(mapping):
        tensor = mapping[name]
        _require(isinstance(tensor, torch.Tensor), f"{name} is not a tensor")
        _require(tensor.device.type == "cpu", f"{name} is not stored on CPU")
        _require(bool(torch.isfinite(tensor).all()), f"{name} is nonfinite")
        value = tensor.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
        count += value.numel()
    return digest.hexdigest(), count


def _load_frozen_manifest(path: Path = FROZEN_INPUT_MANIFEST) -> dict[str, Any]:
    path = _require_regular_file(path)
    _require(
        path == _absolute_without_resolve(FROZEN_INPUT_MANIFEST),
        "unexpected frozen manifest path",
    )
    _require(
        _sha256_file(RECOVERY_PROTOCOL) == EXPECTED_RECOVERY_PROTOCOL_SHA256,
        "recovery protocol hash mismatch",
    )
    _require(
        _sha256_file(path) == EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
        "frozen input manifest hash mismatch",
    )
    manifest = _load_json(path)
    expected_keys = {
        "protocol_id",
        "source_staging_relative_path",
        "expected_exact_file_set",
        "files_sha256",
        "expected_progress",
        "expected_original_audit",
        "recovery_dependencies",
        "recovery_radius_rule",
    }
    _require(set(manifest) == expected_keys, "frozen input manifest schema mismatch")
    _require(
        manifest["protocol_id"] == RECOVERY_PROTOCOL_ID, "manifest protocol mismatch"
    )
    source_relative = Path(manifest["source_staging_relative_path"])
    _require(
        not source_relative.is_absolute() and ".." not in source_relative.parts,
        "frozen source staging path is invalid",
    )
    dependencies = manifest["recovery_dependencies"]
    _require(
        isinstance(dependencies, Mapping)
        and set(dependencies)
        == {"live_producer_source", "independent_continuation_reviewer"},
        "recovery dependency schema mismatch",
    )
    expected_dependencies = {
        "live_producer_source": EXPECTED_PRODUCER_SHA256,
        "independent_continuation_reviewer": EXPECTED_CONTINUATION_REVIEWER_SHA256,
    }
    for name, expected_hash in expected_dependencies.items():
        payload = dependencies[name]
        _require(
            set(payload) == {"relative_path", "sha256"},
            f"bad dependency schema: {name}",
        )
        relative = Path(payload["relative_path"])
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"dependency path invalid: {name}",
        )
        dep_path = _require_regular_file(ROOT / relative)
        _require(payload["sha256"] == expected_hash, f"dependency pin mismatch: {name}")
        _require(_sha256_file(dep_path) == expected_hash, f"dependency changed: {name}")
    radius = manifest["recovery_radius_rule"]
    _require(
        set(radius)
        == {
            "absolute_tolerance",
            "relative_tolerance",
            "expected_max_absolute_error",
            "parent_reviewer_source_relative_path",
            "parent_reviewer_source_sha256",
        },
        "radius rule schema mismatch",
    )
    _require(
        _close(radius["absolute_tolerance"], RECOVERY_RADIUS_ATOL, atol=0)
        and _close(radius["relative_tolerance"], 0.0, atol=0)
        and _close(
            radius["expected_max_absolute_error"], EXPECTED_MAX_RADIUS_ERROR, atol=0
        )
        and radius["parent_reviewer_source_sha256"]
        == EXPECTED_CONTINUATION_REVIEWER_SHA256,
        "radius rule values changed",
    )
    return manifest


def _observed_runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": importlib.import_module("torchvision").__version__,
    }


def _validate_accepted_run_identity(
    run_dir: Any,
    files_sha256: Any,
    *,
    context: str,
) -> None:
    _require(
        run_dir == str(ACCEPTED_RUN_DIR),
        f"{context} accepted run path mismatch",
    )
    _require(
        isinstance(files_sha256, Mapping)
        and dict(files_sha256) == EXPECTED_ACCEPTED_RUN_INPUT_SHA256,
        f"{context} accepted run input hashes mismatch",
    )


def _validate_reconstruction_input_files(record: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"pass", "pre_post_equal", "pre_load_run", "post_load_run"},
        "reconstruction input audit schema mismatch",
    )
    _require(
        record["pass"] is True and record["pre_post_equal"] is True,
        "reconstruction input audit did not pass",
    )
    expected_files = {
        name: {
            "path": str(ACCEPTED_RUN_DIR / name),
            "sha256": digest,
        }
        for name, digest in EXPECTED_ACCEPTED_RUN_INPUT_SHA256.items()
    }
    expected_run_audit = {
        "pass": True,
        "run_dir": str(ACCEPTED_RUN_DIR),
        "file_count": len(expected_files),
        "files": expected_files,
    }
    for label in ("pre_load_run", "post_load_run"):
        audit = record[label]
        _require(
            isinstance(audit, Mapping)
            and set(audit) == {"pass", "run_dir", "file_count", "files"},
            f"reconstruction input {label} schema mismatch",
        )
    _require(
        record["pre_load_run"] == record["post_load_run"],
        "reconstruction inputs differ before and after _load_run",
    )
    for label in ("pre_load_run", "post_load_run"):
        audit = record[label]
        _require(
            audit.get("pass") is True
            and audit.get("run_dir") == str(ACCEPTED_RUN_DIR)
            and audit.get("file_count") == len(expected_files)
            and audit.get("files") == expected_files,
            f"reconstruction input {label} identity mismatch",
        )
    _require(
        record["pre_load_run"] == record["post_load_run"] == expected_run_audit,
        "reconstruction input audit differs from the frozen accepted run",
    )
    return {
        "pass": True,
        "pre_post_equal": True,
        "run_dir": str(ACCEPTED_RUN_DIR),
        "file_count": len(expected_files),
        "files_sha256": dict(EXPECTED_ACCEPTED_RUN_INPUT_SHA256),
    }


def _validate_reconstruction_input_linkage(
    recovery_audit: Mapping[str, Any],
) -> dict[str, Any]:
    top_level = recovery_audit.get("reconstruction_input_files")
    _require(
        isinstance(top_level, Mapping),
        "top-level reconstruction input audit is missing",
    )
    result = _validate_reconstruction_input_files(top_level)
    geometry = recovery_audit.get("geometry_reconstruction")
    _require(
        isinstance(geometry, Mapping)
        and geometry.get("reconstruction_input_files") == top_level,
        "top-level and geometry reconstruction input audits differ",
    )
    gates = recovery_audit.get("recovery_validity_gates")
    _require(
        isinstance(gates, Mapping)
        and gates.get("reconstruction_inputs_exact_pre_post_load_run") is True
        and result["pass"] is True
        and result["pre_post_equal"] is True,
        "reconstruction input validity gate is inconsistent with its evidence",
    )
    return result


def _validate_task_manifest(
    path: Path = RECOVERY_TASK_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _require_regular_file(path)
    _require(
        path == _absolute_without_resolve(RECOVERY_TASK_MANIFEST),
        "unexpected recovery task manifest path",
    )
    observed_hash = _sha256_file(path)
    _require(
        observed_hash == EXPECTED_RECOVERY_TASK_MANIFEST_SHA256,
        "recovery task manifest hash mismatch",
    )
    payload = _load_json(path)
    expected_keys = {
        "protocol_id",
        "source_weight_index",
        "task_name",
        "tau",
        "accepted_vae_checkpoint_sha256",
        "expected_runtime",
        "source_dependencies",
        "selected_task_tensors",
    }
    _require(set(payload) == expected_keys, "recovery task manifest schema mismatch")
    expected_identity = {
        "protocol_id": RECOVERY_PROTOCOL_ID,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "task_name": EXPECTED_TASK_NAME,
        "tau": EXPECTED_TASK_TAU,
        "accepted_vae_checkpoint_sha256": (EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256),
        "expected_runtime": EXPECTED_RUNTIME,
        "source_dependencies": EXPECTED_TASK_SOURCE_DEPENDENCIES,
        "selected_task_tensors": EXPECTED_TASK_TENSORS,
    }
    _require(payload == expected_identity, "recovery task manifest identity mismatch")
    dependency_hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_TASK_SOURCE_DEPENDENCIES.items():
        source = _require_regular_file(ROOT / relative)
        observed = _sha256_file(source)
        _require(observed == expected, f"task source hash mismatch: {relative}")
        dependency_hashes[relative] = observed
    checkpoint = _require_regular_file(ACCEPTED_VAE_CHECKPOINT)
    _require(
        _sha256_file(checkpoint) == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "accepted VAE checkpoint hash mismatch",
    )
    observed_runtime = _observed_runtime()
    _require(observed_runtime == EXPECTED_RUNTIME, "review runtime identity mismatch")
    return payload, {
        "pass": True,
        "manifest_sha256": observed_hash,
        "identity_gates": {name: True for name in expected_identity},
        "source_dependency_count": len(dependency_hashes),
        "source_dependencies_sha256": dict(sorted(dependency_hashes.items())),
        "accepted_vae_checkpoint_sha256": EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "observed_runtime": observed_runtime,
    }


def _validate_import_manifest(
    path: Path = RECOVERY_IMPORT_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _require_regular_file(path)
    _require(
        path == _absolute_without_resolve(RECOVERY_IMPORT_MANIFEST),
        "unexpected recovery import manifest path",
    )
    observed_hash = _sha256_file(path)
    _require(
        observed_hash == EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256,
        "recovery import manifest hash mismatch",
    )
    payload = _load_json(path)
    _require(
        set(payload) == {"protocol_id", "entry_modules", "loaded_repository_modules"},
        "recovery import manifest schema mismatch",
    )
    expected_entries = [
        "scripts.run_one_state_exact_selected_trajectory_continuation",
        "scripts.review_one_state_exact_selected_trajectory_continuation",
    ]
    modules = payload["loaded_repository_modules"]
    _require(
        payload["protocol_id"] == RECOVERY_PROTOCOL_ID
        and payload["entry_modules"] == expected_entries
        and isinstance(modules, Mapping)
        and len(modules) == 49
        and set(expected_entries).issubset(modules),
        "recovery import manifest identity mismatch",
    )
    verified: dict[str, dict[str, str]] = {}
    seen_paths: set[str] = set()
    for name, entry in modules.items():
        _require(
            isinstance(name, str)
            and isinstance(entry, Mapping)
            and set(entry) == {"relative_path", "sha256"},
            f"import closure entry malformed: {name}",
        )
        relative = Path(entry["relative_path"])
        digest = entry["sha256"]
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.suffix == ".py"
            and _is_sha256(digest),
            f"import closure entry invalid: {name}",
        )
        relative_text = str(relative)
        _require(
            relative_text not in seen_paths, f"duplicate import source: {relative}"
        )
        seen_paths.add(relative_text)
        source = _require_regular_file(ROOT / relative)
        observed = _sha256_file(source)
        _require(observed == digest, f"import closure hash mismatch: {name}")
        verified[name] = {"relative_path": relative_text, "sha256": observed}
    for relative, digest in EXPECTED_TASK_SOURCE_DEPENDENCIES.items():
        matches = [
            entry for entry in verified.values() if entry["relative_path"] == relative
        ]
        _require(
            len(matches) == 1 and matches[0]["sha256"] == digest,
            f"task source absent from import closure: {relative}",
        )
    return payload, {
        "pass": True,
        "manifest_sha256": observed_hash,
        "module_count": len(verified),
        "modules": dict(sorted(verified.items())),
    }


def _load_pinned_continuation_reviewer(manifest: Mapping[str, Any]) -> ModuleType:
    payload = manifest["recovery_dependencies"]["independent_continuation_reviewer"]
    path = _require_regular_file(ROOT / payload["relative_path"])
    _require(
        _sha256_file(path) == EXPECTED_CONTINUATION_REVIEWER_SHA256,
        "reviewer dependency changed",
    )
    spec = importlib.util.spec_from_file_location("_pinned_continuation_reviewer", path)
    _require(
        spec is not None and spec.loader is not None,
        "cannot load pinned continuation reviewer",
    )
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    _require(
        module.DIRECTION_RADIUS_ATOL == RECOVERY_RADIUS_ATOL,
        "pinned reviewer radius tolerance mismatch",
    )
    return module


def _validate_source_staging(
    source: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    declared = Path(manifest["source_staging_relative_path"])
    expected_path = _absolute_without_resolve(
        declared if declared.is_absolute() else ROOT / declared
    )
    source = _require_directory(source)
    _require(source == expected_path, "source staging path does not match manifest")
    actual_names = _regular_file_names(source)
    expected_names = set(manifest["expected_exact_file_set"])
    _require(actual_names == expected_names, "source staging exact file set mismatch")
    _require("INCOMPLETE" in actual_names, "source staging lost INCOMPLETE marker")
    _require(
        actual_names.isdisjoint(FORBIDDEN_SOURCE_FINALIZATION_FILES),
        "source staging contains producer finalization files",
    )
    expected_hashes = manifest["files_sha256"]
    _require(set(expected_hashes) == expected_names, "source staging hash set mismatch")
    observed: dict[str, str] = {}
    for name in sorted(expected_names):
        digest = _sha256_file(source / name)
        _require(
            digest == expected_hashes[name], f"source staging hash mismatch: {name}"
        )
        observed[name] = digest
    _require(
        observed["progress_checkpoint.pt"] == EXPECTED_SOURCE_PROGRESS_SHA256,
        "source progress pin mismatch",
    )
    _require(
        observed["executed_source_snapshot.py"] == EXPECTED_PRODUCER_SHA256,
        "source producer snapshot pin mismatch",
    )
    _require(
        observed["protocol_snapshot.md"] == EXPECTED_SOURCE_PROTOCOL_SHA256,
        "source protocol pin mismatch",
    )
    _require(
        observed["frozen_dependency_manifest_snapshot.json"]
        == EXPECTED_SOURCE_DEPENDENCY_MANIFEST_SHA256,
        "source dependency snapshot pin mismatch",
    )
    return {
        "files_sha256": observed,
        "tree_sha256": _tree_fingerprint(source),
    }


def _validate_source_checkpoint_and_tables(
    source: Path,
    manifest: Mapping[str, Any],
    base: ModuleType,
) -> dict[str, Any]:
    _require(
        _sha256_file(PARENT_FINAL) == EXPECTED_PARENT_FINAL_SHA256,
        "parent final changed",
    )
    _require(
        _sha256_file(PARENT_PROGRESS) == EXPECTED_PARENT_PROGRESS_SHA256,
        "parent progress changed",
    )
    _require(
        _sha256_file(PARENT_REVIEW) == EXPECTED_PARENT_REVIEW_SHA256,
        "parent review changed",
    )
    parent_review = _load_json(PARENT_REVIEW)
    _require(
        parent_review.get("valid") is True
        and parent_review.get("failed_gates") == []
        and parent_review.get("errors") == [],
        "parent review is not valid",
    )
    parent = _load_checkpoint(PARENT_PROGRESS)
    progress = _load_checkpoint(source / "progress_checkpoint.pt")
    expected_progress_keys = {
        "protocol_id",
        "normalized_source_sha256",
        "dependency_manifest_sha256",
        "parent_checkpoint_sha256",
        "parent_progress_checkpoint_sha256",
        "parent_active_parameter_hash",
        "accepted_updates",
        "new_accepted_updates",
        "selected_arm",
        "terminal",
        "termination",
        "transition_chain_sha256",
        "active_parameter_hash",
        "active_model_state",
        "state_rows",
        "spectrum_rows",
        "proposal_rows",
        "line_rows",
        "selection_rows",
        "intervention_origin",
        "initial_metrics",
        "tolerances",
        "intervention_preflight",
    }
    _require(set(progress) == expected_progress_keys, "source progress schema mismatch")
    expected = manifest["expected_progress"]
    scalar_expected = {
        "accepted_updates": MAX_ACCEPTED_UPDATES,
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "terminal": True,
        "termination": "max_updates_reached",
        "active_parameter_hash": EXPECTED_ACTIVE_PARAMETER_HASH,
        "transition_chain_sha256": EXPECTED_TRANSITION_CHAIN_SHA256,
    }
    for name, value in scalar_expected.items():
        _require(
            expected[name] == value, f"manifest expected-progress pin changed: {name}"
        )
        _require(progress[name] == value, f"source progress mismatch: {name}")
    row_counts = {
        "state_rows": 101,
        "spectrum_rows": 51_712,
        "proposal_rows": 100,
        "line_rows": 766,
    }
    for name, count in row_counts.items():
        _require(expected[name] == count, f"manifest row pin changed: {name}")
        _require(len(progress[name]) == count, f"source row count mismatch: {name}")
    _require(progress["protocol_id"] == SOURCE_PROTOCOL_ID, "source protocol mismatch")
    _require(progress["selected_arm"] == "low", "source selected arm mismatch")
    _require(
        progress["parent_checkpoint_sha256"] == EXPECTED_PARENT_FINAL_SHA256
        and progress["parent_progress_checkpoint_sha256"]
        == EXPECTED_PARENT_PROGRESS_SHA256,
        "source parent checkpoint lineage mismatch",
    )
    active_state = progress["active_model_state"]
    _require(isinstance(active_state, Mapping), "source active state missing")
    active_hash, active_count = _named_tensor_hash(active_state)
    _require(
        len(active_state) == ACTIVE_TENSOR_COUNT, "source active tensor count mismatch"
    )
    _require(
        active_count == ACTIVE_PARAMETER_COUNT, "source active parameter count mismatch"
    )
    _require(
        active_hash
        == progress["active_parameter_hash"]
        == progress["state_rows"][-1]["parameter_hash"]
        == EXPECTED_ACTIVE_PARAMETER_HASH,
        "source active tensor hash mismatch",
    )
    tables = {
        "states": _read_csv(source / "state_metrics.csv"),
        "spectra": _read_csv(source / "state_spectra.csv"),
        "proposals": _read_csv(source / "proposal_diagnostics.csv"),
        "lines": _read_csv(source / "line_search.csv"),
        "selections": _read_csv(source / "arm_selection.csv"),
    }
    for records_name, table_name in (
        ("state_rows", "states"),
        ("spectrum_rows", "spectra"),
        ("proposal_rows", "proposals"),
        ("line_rows", "lines"),
        ("selection_rows", "selections"),
    ):
        _require(
            base._checkpoint_rows_match_csv(progress[records_name], tables[table_name]),
            f"source {table_name} CSV/checkpoint mismatch",
        )
    _require(
        _equivalent(
            _load_json(source / "intervention_origin.json"),
            progress["intervention_origin"],
        ),
        "source intervention origin mismatch",
    )
    _require(
        _equivalent(
            _load_json(source / "intervention_preflight.json"),
            progress["intervention_preflight"],
        ),
        "source intervention preflight mismatch",
    )
    seeded = base._seeded_parent(parent)
    _require(
        progress["state_rows"][: START_UPDATE + 1] == seeded["state_rows"],
        "source parent state prefix mismatch",
    )
    _require(
        progress["spectrum_rows"][: (START_UPDATE + 1) * DIMENSION]
        == seeded["spectrum_rows"],
        "source parent spectrum prefix mismatch",
    )
    _require(
        progress["proposal_rows"][:START_UPDATE] == seeded["proposal_rows"],
        "source parent proposal prefix mismatch",
    )
    _require(
        progress["line_rows"][: len(seeded["line_rows"])] == seeded["line_rows"],
        "source parent line prefix mismatch",
    )
    _require(
        progress["selection_rows"] == seeded["selection_rows"],
        "source parent selections mismatch",
    )
    _require(
        progress["intervention_origin"] == seeded["intervention_origin"],
        "source origin lineage mismatch",
    )
    preflight = base._validate_preflight(progress, parent)
    spectrum = base._validate_spectrum_tables(tables["states"], tables["spectra"])
    history = base._validate_history(progress, parent)
    return {
        "progress": progress,
        "parent": parent,
        **tables,
        "preflight": preflight,
        "spectrum_review": spectrum,
        "history": history,
    }


def _direction_diagnostics(proposals: pd.DataFrame) -> dict[str, Any]:
    rows = proposals.loc[proposals["target_update"].astype(int).gt(START_UPDATE)]
    _require(len(rows) == NEW_ACCEPTED_UPDATES, "continuation direction count mismatch")
    errors = np.abs(
        rows["direction_norm"].to_numpy(dtype=np.float64) - base_target_norm()
    )
    _require(np.isfinite(errors).all(), "nonfinite direction radius error")
    maximum_position = int(np.argmax(errors))
    maximum = float(errors[maximum_position])
    maximum_proposal = _int(rows.iloc[maximum_position]["target_update"])
    above_original = int(np.sum(errors > ORIGINAL_RADIUS_ATOL))
    above_recovery = int(np.sum(errors > RECOVERY_RADIUS_ATOL))
    _require(
        _close(maximum, EXPECTED_MAX_RADIUS_ERROR, atol=5e-16),
        "maximum direction radius error mismatch",
    )
    _require(maximum_proposal == 74, "maximum radius proposal mismatch")
    _require(
        above_original == EXPECTED_ABOVE_ORIGINAL_RADIUS_COUNT,
        "original radius failure count mismatch",
    )
    _require(above_recovery == 0, "recovery radius tolerance exceeded")
    return {
        "directions_checked": len(rows),
        "maximum_direction_radius_absolute_error": maximum,
        "maximum_direction_radius_proposal": maximum_proposal,
        "directions_above_original_tolerance": above_original,
        "directions_above_recovery_tolerance": above_recovery,
    }


def base_target_norm() -> float:
    return 0.04892722657548397


COPIED_SOURCE_EVIDENCE = {
    "source_progress_checkpoint.pt": "progress_checkpoint.pt",
    "state_metrics.csv": "state_metrics.csv",
    "state_spectra.csv": "state_spectra.csv",
    "proposal_diagnostics.csv": "proposal_diagnostics.csv",
    "line_search.csv": "line_search.csv",
    "arm_selection.csv": "arm_selection.csv",
    "source_intervention_origin.json": "intervention_origin.json",
    "source_intervention_preflight.json": "intervention_preflight.json",
    "source_resolved_config.json": "resolved_config.json",
    "source_run.log": "run.log",
    "frozen_executed_producer_snapshot.py": "executed_source_snapshot.py",
    "frozen_producer_dependency_manifest_snapshot.json": (
        "frozen_dependency_manifest_snapshot.json"
    ),
    "frozen_producer_protocol_snapshot.md": "protocol_snapshot.md",
}


def _validate_copied_source_evidence(
    output: Path,
    source: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for destination, source_name in COPIED_SOURCE_EVIDENCE.items():
        digest = _sha256_file(output / destination)
        _require(
            digest == _sha256_file(source / source_name),
            f"copied source evidence mismatch: {destination}",
        )
        _require(
            digest == manifest["files_sha256"][source_name],
            f"copied source evidence is not frozen: {destination}",
        )
        observed[destination] = digest
    fixed_snapshots = {
        "frozen_continuation_reviewer_snapshot.py": (
            EXPECTED_CONTINUATION_REVIEWER_SHA256
        ),
        "recovery_protocol_snapshot.md": EXPECTED_RECOVERY_PROTOCOL_SHA256,
        "recovery_frozen_input_manifest_snapshot.json": (
            EXPECTED_FROZEN_INPUT_MANIFEST_SHA256
        ),
        "recovery_task_manifest_snapshot.json": (
            EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
        ),
        "recovery_import_manifest_snapshot.json": (
            EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
        ),
    }
    for name, expected in fixed_snapshots.items():
        digest = _sha256_file(output / name)
        _require(digest == expected, f"frozen snapshot mismatch: {name}")
        observed[name] = digest
    finalizer_snapshot = output / "executed_recovery_finalizer_source_snapshot.py"
    finalizer_hash = _sha256_file(finalizer_snapshot)
    _require(
        finalizer_hash == EXPECTED_FINALIZER_SHA256,
        "frozen finalizer snapshot raw hash mismatch",
    )
    _require(
        _normalized_finalizer_sha256(finalizer_snapshot)
        == EXPECTED_FINALIZER_NORMALIZED_SHA256,
        "frozen finalizer snapshot normalized hash mismatch",
    )
    observed[finalizer_snapshot.name] = finalizer_hash
    snapshot_hashes = config.get("snapshot_hashes")
    _require(
        isinstance(snapshot_hashes, Mapping) and set(snapshot_hashes) == set(observed),
        "resolved snapshot-hash schema mismatch",
    )
    _require(
        all(snapshot_hashes[name] == digest for name, digest in observed.items()),
        "resolved snapshot hashes do not match copied evidence",
    )
    return observed


def _normalized_finalizer_sha256(path: Path) -> str:
    path = _require_regular_file(path)
    prefix = "EXPECTED_NORMALIZED_SOURCE_SHA256 = "
    normalized: list[str] = []
    masked = 0
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith(prefix):
            normalized.append(f'{prefix}"<FROZEN>"\n')
            masked += 1
        else:
            normalized.append(line)
    _require(masked == 1, f"finalizer normalization masked {masked} lines")
    return _sha256_bytes("".join(normalized).encode("utf-8"))


def _validate_finalizer_static_source(
    path: Path,
    *,
    expected_raw_sha256: str | None = None,
    expected_normalized_sha256: str | None = None,
) -> dict[str, Any]:
    path = _require_regular_file(path)
    expected_raw = expected_raw_sha256 or EXPECTED_FINALIZER_SHA256
    expected_normalized = (
        expected_normalized_sha256 or EXPECTED_FINALIZER_NORMALIZED_SHA256
    )
    _require(_is_sha256(expected_raw), "reviewed finalizer raw hash is not frozen")
    _require(
        _is_sha256(expected_normalized),
        "reviewed finalizer normalized hash is not frozen",
    )
    raw_hash = _sha256_file(path)
    normalized_hash = _normalized_finalizer_sha256(path)
    _require(raw_hash == expected_raw, "reviewed finalizer raw hash mismatch")
    _require(
        normalized_hash == expected_normalized,
        "reviewed finalizer normalized hash mismatch",
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: set[str] = set()
    call_counts: dict[str, int] = {}
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assigned_literals: dict[str, Any] = {}
    forbidden_calls: list[str] = []
    forbidden_names = {
        "_exact_joint_gradients",
        "_build_direction",
        "backward",
        "step",
        "zero_grad",
        "Adam",
        "AdamW",
        "SGD",
        "Optimizer",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                assigned_literals[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            call_counts[name] = call_counts.get(name, 0) + 1
            if name in forbidden_names:
                forbidden_calls.append(name)
    evaluate_calls = call_counts.get("_evaluate", 0)
    atomic_replace_calls = call_counts.get("replace", 0)
    _require(
        evaluate_calls == 1,
        "finalizer source does not contain exactly one _evaluate call site",
    )
    _require(
        not forbidden_calls,
        f"finalizer source contains forbidden call sites: {forbidden_calls}",
    )
    _require(
        atomic_replace_calls >= 4, "finalizer source lacks atomic replace operations"
    )
    _require(".incomplete." in source, "finalizer source lacks isolated staging name")
    _require(
        assigned_literals.get("PRODUCTION_DEVICE") == "cuda:0",
        "finalizer production device is not exactly cuda:0",
    )
    _require(
        assigned_literals.get("EXPECTED_NORMALIZED_SOURCE_SHA256")
        == expected_normalized,
        "finalizer self-reported normalized hash mismatch",
    )
    _require(
        call_counts.get("_require_default_output", 0) == 1
        and "postgoal_relaxed_cancellation_recovery_finalization_v1" in string_literals
        and "--output" not in string_literals
        and "--device" not in string_literals,
        "finalizer exact-output/device gate is missing",
    )
    for clean_import_call in (
        "_producer_dependency_preimport_audit",
        "_reject_preloaded_import_manifest_modules",
        "_audit_loaded_repository_module_closure",
        "_verify_recovery_import_manifest",
    ):
        _require(
            call_counts.get(clean_import_call, 0) >= 1,
            f"finalizer clean import gate missing: {clean_import_call}",
        )
    _require(
        call_counts.get("_verify_finalizer_source_freeze", 0) >= 3
        and "normalized source must mask exactly one self-freeze constant" in source,
        "finalizer normalized self-freeze gate is incomplete",
    )
    for replay_guard_literal in (
        "eigenvalue_errors = np.abs(observed_eig - stored_eig)",
        "spectrum_eigenvalue_count",
        "low_basis_hash_matches",
        "geometry_evaluation_count",
        "optimization_gradient_evaluations",
        "parameter_updates",
        "replayed_final_geometry.pt",
        "replayed_final_spectrum.csv",
    ):
        _require(
            replay_guard_literal in source,
            f"finalizer replay guard missing: {replay_guard_literal}",
        )
    return {
        "raw_source_sha256": raw_hash,
        "normalized_source_sha256": normalized_hash,
        "imports": sorted(imports),
        "evaluate_call_sites": evaluate_calls,
        "atomic_replace_call_sites": atomic_replace_calls,
    }


def _validate_png(path: Path) -> dict[str, Any]:
    path = _require_regular_file(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        _require(width >= 600 and height >= 400, f"{path.name} is unexpectedly small")
        converted = image.convert("RGB")
        extrema = ImageStat.Stat(converted).extrema
        dynamic_channels = sum(high > low for low, high in extrema)
        _require(dynamic_channels >= 2, f"{path.name} appears blank or one-color")
        colors = converted.resize((128, 128)).getcolors(maxcolors=128 * 128)
        _require(
            colors is None or len(colors) >= 16,
            f"{path.name} has too little visual content",
        )
    return {"width": width, "height": height, "dynamic_channels": dynamic_channels}


def _validate_manifest_and_publication(
    output: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    output = _require_directory(output)
    _require(
        not output.name.endswith(".incomplete"),
        "recovery output still has incomplete name",
    )
    _require(
        not os.path.lexists(output / "INCOMPLETE"),
        "recovery output has INCOMPLETE marker",
    )
    task_manifest, task_manifest_audit = _validate_task_manifest()
    import_manifest, import_manifest_audit = _validate_import_manifest()
    actual_names = _regular_file_names(output)
    _require(
        actual_names == EXPECTED_RECOVERY_FILE_SET, "recovery exact file set mismatch"
    )
    artifact_manifest = _load_json(output / "artifact_manifest.json")
    _require(
        set(artifact_manifest) == EXPECTED_MANIFEST_KEYS,
        "artifact manifest schema mismatch",
    )
    _require(
        artifact_manifest["protocol_id"] == RECOVERY_PROTOCOL_ID,
        "artifact manifest protocol mismatch",
    )
    artifacts = artifact_manifest["artifacts"]
    _require(isinstance(artifacts, Mapping), "artifact manifest payload missing")
    _require(
        set(artifacts) == EXPECTED_MANIFESTED_ARTIFACTS,
        "manifest artifact set mismatch",
    )
    artifact_hashes: dict[str, str] = {}
    for name, digest in artifacts.items():
        _require(_is_sha256(digest), f"malformed artifact hash: {name}")
        observed = _sha256_file(output / name)
        _require(observed == digest, f"artifact hash mismatch: {name}")
        artifact_hashes[name] = observed
    finalizer_hash = _sha256_file(
        output / "executed_recovery_finalizer_source_snapshot.py"
    )
    finalizer_normalized_hash = _normalized_finalizer_sha256(
        output / "executed_recovery_finalizer_source_snapshot.py"
    )
    _require(
        finalizer_hash
        == artifact_manifest["recovery_finalizer_source_sha256"]
        == EXPECTED_FINALIZER_SHA256,
        "finalizer source snapshot mismatch",
    )
    _require(
        finalizer_normalized_hash
        == artifact_manifest["recovery_finalizer_normalized_source_sha256"]
        == EXPECTED_FINALIZER_NORMALIZED_SHA256,
        "finalizer normalized source snapshot mismatch",
    )
    _require(
        artifact_manifest["frozen_input_manifest_sha256"]
        == EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
        "manifest frozen-input hash mismatch",
    )
    _require(
        artifact_manifest["recovery_task_manifest_sha256"]
        == EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
        and artifact_manifest["recovery_import_manifest_sha256"]
        == EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256,
        "manifest task/import hash mismatch",
    )
    _require(
        artifact_manifest["source_progress_checkpoint_sha256"]
        == EXPECTED_SOURCE_PROGRESS_SHA256,
        "manifest source-progress hash mismatch",
    )
    finalized = _load_json(output / "FINALIZED.json")
    _require(set(finalized) == EXPECTED_FINALIZED_KEYS, "FINALIZED schema mismatch")
    _require(
        finalized["protocol_id"] == RECOVERY_PROTOCOL_ID, "FINALIZED protocol mismatch"
    )
    _require(
        finalized["source_protocol_id"] == SOURCE_PROTOCOL_ID,
        "FINALIZED source protocol mismatch",
    )
    _require(
        finalized["status"] == "complete_awaiting_independent_recovery_review",
        "FINALIZED status mismatch",
    )
    _require(finalized["recovery_valid"] is True, "recovery was not finalized valid")
    _require(
        finalized["accepted_vae_checkpoint_sha256"]
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        and finalized["recovery_task_manifest_sha256"]
        == EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
        and finalized["recovery_import_manifest_sha256"]
        == EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
        and finalized["recovery_finalizer_source_sha256"] == EXPECTED_FINALIZER_SHA256
        and finalized["recovery_finalizer_normalized_source_sha256"]
        == EXPECTED_FINALIZER_NORMALIZED_SHA256,
        "FINALIZED frozen dependency identity mismatch",
    )
    _require(
        finalized["source_progress_checkpoint_sha256"]
        == EXPECTED_SOURCE_PROGRESS_SHA256,
        "FINALIZED source-progress mismatch",
    )
    _require(
        finalized["decision_sha256"] == _sha256_file(output / "decision.json"),
        "FINALIZED decision hash mismatch",
    )
    _require(
        finalized["recovery_audit_sha256"]
        == _sha256_file(output / "recovery_audit.json"),
        "FINALIZED recovery-audit hash mismatch",
    )
    _require(
        finalized["artifact_manifest_sha256"]
        == _sha256_file(output / "artifact_manifest.json"),
        "FINALIZED manifest hash mismatch",
    )
    _require(
        finalized["final_checkpoint_sha256"]
        == _sha256_file(output / "final_checkpoint.pt"),
        "FINALIZED checkpoint hash mismatch",
    )
    _require((output / "run.log").stat().st_size > 0, "recovery run log is empty")
    config = _load_json(output / "resolved_config.json")
    _require(set(config) == EXPECTED_CONFIG_KEYS, "recovery config schema mismatch")
    _require(config["protocol_id"] == RECOVERY_PROTOCOL_ID, "config protocol mismatch")
    _require(
        config["source_protocol_id"] == SOURCE_PROTOCOL_ID,
        "config source protocol mismatch",
    )
    _require(
        _absolute_without_resolve(Path(config["source_staging"]))
        == _absolute_without_resolve(SOURCE_STAGING),
        "config source staging mismatch",
    )
    _require(
        _absolute_without_resolve(Path(config["output_dir"])) == output,
        "config output path mismatch",
    )
    _require(
        config["source_progress_checkpoint_sha256"] == EXPECTED_SOURCE_PROGRESS_SHA256,
        "config progress pin mismatch",
    )
    _require(
        config["frozen_input_manifest_sha256"] == EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
        "config manifest pin mismatch",
    )
    _require(
        config["recovery_protocol_sha256"] == EXPECTED_RECOVERY_PROTOCOL_SHA256,
        "config protocol hash mismatch",
    )
    _require(
        config["recovery_task_manifest_sha256"]
        == EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
        and config["recovery_import_manifest_sha256"]
        == EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256,
        "config task/import manifest pin mismatch",
    )
    _require(
        config["producer_source_sha256"] == EXPECTED_PRODUCER_SHA256,
        "config producer pin mismatch",
    )
    _require(
        config["continuation_reviewer_source_sha256"]
        == EXPECTED_CONTINUATION_REVIEWER_SHA256,
        "config continuation reviewer pin mismatch",
    )
    _require(
        config["finalizer_source_sha256"] == finalizer_hash == EXPECTED_FINALIZER_SHA256
        and config["finalizer_normalized_source_sha256"]
        == finalizer_normalized_hash
        == EXPECTED_FINALIZER_NORMALIZED_SHA256,
        "config finalizer freeze mismatch",
    )
    _require(
        config["geometry_evaluation_budget"] == 1
        and config["optimization_gradient_evaluation_budget"] == 0
        and config["proposal_budget"] == 0
        and config["line_search_budget"] == 0
        and config["parameter_update_budget"] == 0,
        "config replay budgets changed",
    )
    _require(
        _close(config["metric_replay_absolute_tolerance"], REPLAY_ATOL, atol=0)
        and _close(config["spectrum_replay_absolute_tolerance"], REPLAY_ATOL, atol=0)
        and _close(
            config["direction_radius_absolute_tolerance"], RECOVERY_RADIUS_ATOL, atol=0
        )
        and _close(config["direction_radius_relative_tolerance"], 0.0, atol=0),
        "config radius rule mismatch",
    )
    _require(
        config["source_staging_mutation_allowed"] is False,
        "source mutation was allowed",
    )
    _require(
        config["device"] == "cuda:0"
        and config["dtype"] == "float32 model/HVP; FP64 dense H/M replay products"
        and config["seed"] == "none; deterministic frozen terminal-state replay"
        and config["cache_mode"] == "accepted h2048 VAE plus frozen terminal progress"
        and config["expected_runtime"] == EXPECTED_RUNTIME,
        "config execution identity mismatch",
    )
    _validate_accepted_run_identity(
        config["accepted_run_dir"],
        config["accepted_run_input_sha256"],
        context="config",
    )
    task_record = {
        key: task_manifest_audit[key]
        for key in (
            "pass",
            "manifest_sha256",
            "identity_gates",
            "source_dependency_count",
            "source_dependencies_sha256",
        )
    }
    import_record = {
        key: import_manifest_audit[key]
        for key in ("pass", "manifest_sha256", "module_count", "modules")
    }
    _require(
        config["task_manifest_preflight"]
        == config["copied_task_manifest_audit"]
        == task_record
        and config["import_manifest_preflight"]
        == config["copied_import_manifest_audit"]
        == import_record,
        "config task/import audit record mismatch",
    )
    _require(
        config["startup_self_freeze"].get("live_source_sha256")
        == EXPECTED_FINALIZER_SHA256
        and config["copied_self_freeze"].get("snapshot_source_sha256")
        == EXPECTED_FINALIZER_SHA256
        and config["startup_self_freeze"].get("live_normalized_source_sha256")
        == EXPECTED_FINALIZER_NORMALIZED_SHA256
        and config["copied_self_freeze"].get("snapshot_normalized_source_sha256")
        == EXPECTED_FINALIZER_NORMALIZED_SHA256,
        "config finalizer self-freeze record mismatch",
    )
    working = Path(config["working_dir"])
    _require(
        _absolute_without_resolve(working) != output
        and ".incomplete." in working.name
        and not os.path.lexists(working),
        "resolved atomic staging directory is invalid or still present",
    )
    static_source = _validate_finalizer_static_source(
        output / "executed_recovery_finalizer_source_snapshot.py"
    )
    return {
        "artifact_manifest": artifact_manifest,
        "artifact_hashes": artifact_hashes,
        "finalized": finalized,
        "config": config,
        "task_manifest": task_manifest,
        "task_manifest_audit": task_manifest_audit,
        "import_manifest": import_manifest,
        "import_manifest_audit": import_manifest_audit,
        "finalizer_static_source": static_source,
    }


def _validate_original_and_recovery_audits(
    output: Path,
    manifest: Mapping[str, Any],
    source_packet: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    recovery = _load_json(output / "recovery_audit.json")
    _require(
        set(recovery) == EXPECTED_RECOVERY_AUDIT_KEYS, "recovery audit schema mismatch"
    )
    _require(
        recovery["protocol_id"] == RECOVERY_PROTOCOL_ID,
        "recovery audit protocol mismatch",
    )
    _require(
        recovery["source_protocol_id"] == SOURCE_PROTOCOL_ID,
        "recovery source protocol mismatch",
    )
    _require(recovery["recovery_valid"] is True, "recovery audit is invalid")
    _require(
        recovery["original_runner_audit_pass"] is False,
        "original audit falsely recorded as pass",
    )
    _require(
        recovery["original_runner_failed_gates"] == EXPECTED_ORIGINAL_FAILED_GATES,
        "top-level original failed-gate set mismatch",
    )
    original = recovery["original_runner_audit"]
    _require(isinstance(original, Mapping), "original audit payload missing")
    _require(
        set(original) == EXPECTED_ORIGINAL_AUDIT_KEYS, "original audit schema mismatch"
    )
    _require(original["pass"] is False, "nested original audit falsely passed")
    _require(
        original["failed_gates"] == EXPECTED_ORIGINAL_FAILED_GATES,
        "original failed-gate set mismatch",
    )
    _require(
        manifest["expected_original_audit"]
        == {"pass": False, "failed_gates": EXPECTED_ORIGINAL_FAILED_GATES},
        "manifest original-audit expectation mismatch",
    )
    _require(
        original["exception_type"] == "RuntimeError",
        "original audit exception type mismatch",
    )
    _require(
        original["exception_message"]
        == "continuation progress audit failed: ['continuation_diagnostics_recompute']",
        "original audit exception message mismatch",
    )
    _require(
        original["audit_invocations"] == 1, "original audit invocation count mismatch"
    )
    row_gates = original["row_gates"]
    _require(isinstance(row_gates, Mapping) and row_gates, "original row gates missing")
    _require(
        [name for name, passed in row_gates.items() if passed is not True]
        == EXPECTED_ORIGINAL_FAILED_GATES,
        "original row-gate failure set mismatch",
    )
    _require(
        original["all_unmodified_row_gates_pass"] is True,
        "unmodified row gates did not pass",
    )
    _require(
        original["transition_chain_sha256"] == EXPECTED_TRANSITION_CHAIN_SHA256,
        "original audit chain mismatch",
    )
    independent = recovery["independent_recovery_audit"]
    expected_independent_keys = {
        "pass",
        "gates",
        "prefix_gates",
        "history_summary",
        "transition_chain_sha256",
        "radius_audit",
        "preflight",
        "spectrum",
    }
    _require(
        isinstance(independent, Mapping)
        and set(independent) == expected_independent_keys,
        "independent recovery audit schema mismatch",
    )
    _require(independent["pass"] is True, "independent recovery audit failed")
    _require(
        isinstance(independent["gates"], Mapping)
        and all(value is True for value in independent["gates"].values()),
        "independent audit gates failed",
    )
    _require(
        isinstance(independent["prefix_gates"], Mapping)
        and all(value is True for value in independent["prefix_gates"].values()),
        "independent prefix gates failed",
    )
    _require(
        independent["history_summary"] == source_packet["history"]["summary"],
        "independent history summary mismatch",
    )
    _require(
        independent["transition_chain_sha256"] == EXPECTED_TRANSITION_CHAIN_SHA256,
        "independent transition chain mismatch",
    )
    _require(
        _equivalent(
            independent["spectrum"],
            source_packet["spectrum_review"],
            atol=2e-9,
            rtol=2e-9,
        ),
        "independent spectrum audit payload mismatch",
    )
    radius = independent["radius_audit"]
    expected_radius = {
        "pass": True,
        "relative_tolerance": 0.0,
        "absolute_tolerance": RECOVERY_RADIUS_ATOL,
        "row_count": diagnostics["directions_checked"],
        "max_absolute_error": diagnostics["maximum_direction_radius_absolute_error"],
        "max_error_target_update": diagnostics["maximum_direction_radius_proposal"],
        "count_exceeding_original_1e_9": diagnostics[
            "directions_above_original_tolerance"
        ],
        "count_exceeding_recovery_5e_9": diagnostics[
            "directions_above_recovery_tolerance"
        ],
    }
    _require(
        _equivalent(radius, expected_radius, atol=1e-15),
        "recovery radius audit mismatch",
    )
    _require(
        _equivalent(independent["preflight"], source_packet["preflight"], atol=1e-15),
        "independent preflight payload mismatch",
    )
    recovery_gates = recovery["recovery_validity_gates"]
    _require(
        isinstance(recovery_gates, Mapping)
        and set(recovery_gates) == EXPECTED_RECOVERY_VALIDITY_GATES
        and all(value is True for value in recovery_gates.values()),
        "recovery validity gates changed or failed",
    )
    _validate_reconstruction_input_linkage(recovery)
    preflight = recovery["immutable_input_preflight"]
    postflight = recovery["immutable_input_postflight"]
    _require(preflight == postflight, "immutable input pre/postflight differ")
    _require(
        preflight.get("manifest_sha256") == EXPECTED_FROZEN_INPUT_MANIFEST_SHA256
        and preflight.get("source_file_count") == 14
        and preflight.get("source_files_sha256") == manifest["files_sha256"],
        "immutable input audit payload mismatch",
    )
    dependency_hashes = preflight.get("dependency_files_sha256")
    _require(
        dependency_hashes
        == {
            "independent_continuation_reviewer": EXPECTED_CONTINUATION_REVIEWER_SHA256,
            "live_producer_source": EXPECTED_PRODUCER_SHA256,
        },
        "immutable dependency audit mismatch",
    )
    dependency_import = recovery["dependency_import"]
    _require(
        isinstance(dependency_import, Mapping)
        and set(dependency_import)
        == {
            "preimport",
            "transitive_preimport",
            "transitive_postimport",
            "task_postimport",
            "import_postimport",
            "loaded_repository_closure",
            "postimport",
            "loaded_paths",
            "executed_snapshot_equals_loaded_producer",
        },
        "dependency-import audit schema mismatch",
    )
    _require(
        dependency_import["preimport"] == dependency_import["postimport"] == preflight
        and dependency_import["executed_snapshot_equals_loaded_producer"] is True,
        "dependency-import pre/postflight mismatch",
    )
    expected_loaded_paths = {
        name: str(_absolute_without_resolve(ROOT / payload["relative_path"]))
        for name, payload in manifest["recovery_dependencies"].items()
    }
    _require(
        dependency_import["loaded_paths"] == expected_loaded_paths,
        "loaded recovery dependency paths mismatch",
    )
    task_manifest, task_audit = _validate_task_manifest()
    import_manifest, import_audit = _validate_import_manifest()
    task_record = {
        key: task_audit[key]
        for key in (
            "pass",
            "manifest_sha256",
            "identity_gates",
            "source_dependency_count",
            "source_dependencies_sha256",
        )
    }
    import_record = {
        key: import_audit[key]
        for key in ("pass", "manifest_sha256", "module_count", "modules")
    }
    _require(
        recovery["task_manifest_preflight"]
        == recovery["task_manifest_postflight"]
        == dependency_import["task_postimport"]
        == task_record,
        "task manifest pre/post/import audit mismatch",
    )
    _require(
        recovery["import_manifest_preflight"]
        == recovery["import_manifest_postflight"]
        == dependency_import["import_postimport"]
        == import_record,
        "import manifest pre/post/import audit mismatch",
    )
    transitive_preimport = dependency_import["transitive_preimport"]
    transitive_postimport = dependency_import["transitive_postimport"]
    _require(
        isinstance(transitive_preimport, Mapping)
        and transitive_preimport.get("preloaded_module_rejection_pass") is True
        and transitive_preimport.get("task_manifest") == task_record
        and transitive_preimport.get("import_manifest") == import_record,
        "clean pre-import gate evidence mismatch",
    )
    transitive_tree_keys = {
        "root_manifest_sha256",
        "manifest_count",
        "verified_file_count",
        "verified_files_sha256",
        "python_paths",
    }
    _require(
        isinstance(transitive_postimport, Mapping)
        and set(transitive_postimport) == transitive_tree_keys
        and {key: transitive_preimport[key] for key in transitive_tree_keys}
        == transitive_postimport,
        "transitive dependency tree changed after import",
    )
    runtime_closure = recovery["runtime_import_closure"]
    expected_runtime_closure = _validate_runtime_import_closure(
        runtime_closure, import_manifest
    )
    _require(
        dependency_import["loaded_repository_closure"] == expected_runtime_closure,
        "dependency import closure differs from top-level runtime closure",
    )
    _require(
        task_manifest["expected_runtime"] == EXPECTED_RUNTIME,
        "task runtime pin mismatch",
    )
    startup_freeze = recovery["startup_self_freeze"]
    copied_freeze = recovery["copied_self_freeze"]
    _require(
        set(startup_freeze)
        == {
            "live_source_sha256",
            "live_normalized_source_sha256",
            "expected_normalized_source_sha256",
            "recovery_protocol_sha256",
        }
        and startup_freeze.get("live_source_sha256") == EXPECTED_FINALIZER_SHA256
        and startup_freeze.get("live_normalized_source_sha256")
        == EXPECTED_FINALIZER_NORMALIZED_SHA256
        and startup_freeze.get("expected_normalized_source_sha256")
        == EXPECTED_FINALIZER_NORMALIZED_SHA256
        and startup_freeze.get("recovery_protocol_sha256")
        == EXPECTED_RECOVERY_PROTOCOL_SHA256,
        "startup finalizer self-freeze mismatch",
    )
    _require(
        set(copied_freeze)
        == {
            "live_source_sha256",
            "live_normalized_source_sha256",
            "expected_normalized_source_sha256",
            "recovery_protocol_sha256",
            "snapshot_source_sha256",
            "snapshot_normalized_source_sha256",
            "snapshot_protocol_sha256",
        }
        and copied_freeze.get("live_source_sha256") == EXPECTED_FINALIZER_SHA256
        and copied_freeze.get("snapshot_source_sha256") == EXPECTED_FINALIZER_SHA256
        and copied_freeze.get("live_normalized_source_sha256")
        == EXPECTED_FINALIZER_NORMALIZED_SHA256
        and copied_freeze.get("snapshot_normalized_source_sha256")
        == EXPECTED_FINALIZER_NORMALIZED_SHA256
        and copied_freeze.get("snapshot_protocol_sha256")
        == EXPECTED_RECOVERY_PROTOCOL_SHA256,
        "copied finalizer self-freeze mismatch",
    )
    history_frame = _read_csv(output / "historical_transition_audit.csv")
    _require(
        source_packet["history"]["audit_rows"] and source_packet["history"]["summary"],
        "independent history is empty",
    )
    base = source_packet["base"]
    _require(
        base._checkpoint_rows_match_csv(
            source_packet["history"]["audit_rows"], history_frame
        ),
        "historical transition audit CSV mismatch",
    )
    return {"original": original, "recovery": recovery}


def _validate_runtime_import_closure(
    record: Mapping[str, Any], import_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "pass": True,
        "module_count": 49,
        "modules": import_manifest["loaded_repository_modules"],
        "allowed_additions": [
            "__main__",
            "scripts.finalize_one_state_exact_selected_trajectory_recovery",
            "__mp_main__",
        ],
    }
    _require(
        record == expected,
        "runtime import closure mismatch",
    )
    return expected


def _validate_geometry_execution(execution: Mapping[str, Any]) -> None:
    _require(
        set(execution) == EXPECTED_GEOMETRY_EXECUTION_KEYS,
        "geometry execution schema mismatch",
    )
    _require(
        execution
        == {
            "geometry_evaluation_count": 1,
            "forbidden_operation_attempts": 0,
            "optimization_gradient_evaluations": 0,
            "proposals": 0,
            "line_searches": 0,
            "parameter_updates": 0,
        },
        "exact-one-evaluate/no-optimization evidence failed",
    )


def _recompute_geometry_metrics(
    *,
    hessian: torch.Tensor,
    matrix: torch.Tensor,
    eig: torch.Tensor,
    low_basis: torch.Tensor,
    low_projector: torch.Tensor,
) -> dict[str, float]:
    dim = int(eig.numel())
    _require(dim == DIMENSION, "geometry dimension mismatch")
    contribution = (eig - 1.0).square()
    total_a = contribution.sum().clamp_min(1e-30)
    trace = eig.sum()
    square_sum = eig.square().sum().clamp_min(1e-30)
    quantiles = torch.quantile(
        eig,
        torch.tensor(
            [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
            dtype=torch.float64,
        ),
    )
    r_eig = (eig + GEOMETRY_EPSILON) / (1.0 + GEOMETRY_EPSILON)
    g_eig = (1.0 - r_eig.reciprocal()) / (float(dim) * (1.0 + GEOMETRY_EPSILON))
    burg_trace = r_eig.mean()
    burg_neg_logdet = -r_eig.log().mean()
    full_burg = burg_trace + burg_neg_logdet - 1.0
    identity = torch.eye(dim, dtype=torch.float64)
    direct_a = (matrix - identity).square().sum() / float(dim)
    trace_a = 1.0 - 2.0 * eig.mean() + eig.square().mean()
    raw_eig = torch.linalg.eigvalsh(matrix)
    low_mask = eig < GEOMETRY_LOW_THRESHOLD
    low_values = raw_eig[low_mask]
    _require(
        low_basis.shape == (DIMENSION, int(low_mask.sum())),
        "low basis count does not match spectrum threshold",
    )
    _require(low_basis.shape[1] > 0, "low basis is unexpectedly empty")
    orthogonality = (
        (low_basis.T @ low_basis - torch.eye(low_basis.shape[1], dtype=torch.float64))
        .abs()
        .max()
    )
    residual = (
        matrix @ low_basis - low_basis * low_values.unsqueeze(0)
    ).norm() / matrix.norm().clamp_min(1e-30)
    canonical_low_energy = torch.trace(low_basis.T @ matrix @ low_basis) / float(
        low_basis.shape[1]
    )
    exact_a = contribution.mean()
    projector_error = float((low_projector - low_basis @ low_basis.T).abs().max())
    _require(
        projector_error <= REPLAY_ATOL,
        "low projector/basis closure exceeds replay tolerance",
    )
    return {
        "hessian_symmetry_rel": float(
            (hessian - hessian.T).norm() / hessian.norm().clamp_min(1e-30)
        ),
        "exact_a_per_dim": float(exact_a),
        "a_constant_term": 1.0,
        "a_linear_trace_term": float(-2.0 * eig.mean()),
        "a_quartic_term": float(eig.square().mean()),
        "trace_m_per_dim": float(eig.mean()),
        "damped_full_burg_per_dim": float(full_burg),
        "burg_trace_r_term": float(burg_trace),
        "burg_neg_logdet_r_term": float(burg_neg_logdet),
        "logdet_r_per_dim": float(r_eig.log().mean()),
        "m_min": float(quantiles[0]),
        "m_p01": float(quantiles[1]),
        "m_p10": float(quantiles[2]),
        "m_p25": float(quantiles[3]),
        "m_p50": float(quantiles[4]),
        "m_p75": float(quantiles[5]),
        "m_p90": float(quantiles[6]),
        "m_p95": float(quantiles[7]),
        "m_p99": float(quantiles[8]),
        "m_max": float(quantiles[9]),
        "m_lt_1e_4_fraction": float((eig < 1e-4).double().mean()),
        "m_lt_0p01_fraction": float((eig < 0.01).double().mean()),
        "m_lt_0p1_fraction": float((eig < 0.1).double().mean()),
        "m_lt_0p5_fraction": float((eig < 0.5).double().mean()),
        "m_near_1_10pct_fraction": float(((eig - 1.0).abs() <= 0.1).double().mean()),
        "m_gt_1_fraction": float((eig > 1.0).double().mean()),
        "m_gt_2_fraction": float((eig > 2.0).double().mean()),
        "a_from_m_lt_0p1_share": float(contribution[eig < 0.1].sum() / total_a),
        "a_from_m_gt_1_share": float(contribution[eig > 1.0].sum() / total_a),
        "burg_matrix_gradient_norm": float(g_eig.norm()),
        "burg_matrix_gradient_eig_min": float(g_eig.min()),
        "burg_matrix_gradient_eig_p50": float(g_eig.median()),
        "burg_matrix_gradient_eig_max": float(g_eig.max()),
        "true_objective": float(exact_a + GEOMETRY_OLD_BETA * full_burg),
        "a_direct_matrix": float(direct_a),
        "a_trace_closure": float(trace_a),
        "a_direct_abs_error": float(abs(direct_a - exact_a)),
        "a_trace_abs_error": float(abs(trace_a - exact_a)),
        "m_raw_eig_min": float(raw_eig.min()),
        "effective_rank": float(trace.square() / square_sum),
        "effective_rank_fraction": float(trace.square() / (square_sum * dim)),
        "li_gap_per_dim": float((eig.sqrt() - 1.0).square().mean()),
        "a_low90_abs_per_dim": float(
            contribution[: int(math.floor(0.9 * dim))].sum() / dim
        ),
        "a_low_lt_0p1_abs_per_dim": float(contribution[eig < 0.1].sum() / dim),
        "a_high_gt_1_abs_per_dim": float(contribution[eig > 1.0].sum() / dim),
        "a_top1_share": float(contribution[-1] / total_a),
        "a_top10_share": float(contribution[-10:].sum() / total_a),
        "top1_trace_share": float(eig[-1] / trace.clamp_min(1e-30)),
        "top10_trace_share": float(eig[-10:].sum() / trace.clamp_min(1e-30)),
        "frozen_low_energy": float(canonical_low_energy),
        "count_lt_1e_4": float((eig < 1e-4).sum()),
        "count_lt_1e_2": float((eig < 1e-2).sum()),
        "count_lt_0p1": float((eig < 0.1).sum()),
        "a_gt1": float(contribution[eig > 1.0].sum() / dim),
        "canonical_low_energy": float(canonical_low_energy),
        "low_count": float(low_mask.sum()),
        "low_basis_orthogonality_max_abs": float(orthogonality),
        "low_basis_eigen_residual_relative": float(residual),
        "helper_eigenvalue_max_abs_error": float(
            (eig - raw_eig.clamp_min(0.0)).abs().max()
        ),
    }


def _validate_replayed_spectrum_csv(
    path: Path,
    *,
    stored_eig: np.ndarray,
    replayed_eig: np.ndarray,
) -> dict[str, Any]:
    path = _require_regular_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames == ["rank", "stored", "replayed", "abs_error"],
            "replayed spectrum CSV schema mismatch",
        )
        rows = list(reader)
    _require(len(rows) == DIMENSION, "replayed spectrum CSV row count mismatch")
    errors = np.empty(DIMENSION, dtype=np.float64)
    for rank, row in enumerate(rows):
        _require(
            set(row) == {"rank", "stored", "replayed", "abs_error"},
            f"replayed spectrum row schema mismatch at rank {rank}",
        )
        observed_rank = _int(row["rank"])
        observed_stored = _float(row["stored"])
        observed_replayed = _float(row["replayed"])
        observed_error = _float(row["abs_error"])
        expected_error = abs(float(replayed_eig[rank]) - float(stored_eig[rank]))
        _require(observed_rank == rank, f"replayed spectrum rank mismatch: {rank}")
        _require(
            observed_stored == float(stored_eig[rank])
            and observed_replayed == float(replayed_eig[rank])
            and observed_error == expected_error,
            f"replayed spectrum value mismatch at rank {rank}",
        )
        _require(
            expected_error <= REPLAY_ATOL,
            f"replayed spectrum exceeds tolerance at rank {rank}",
        )
        errors[rank] = expected_error
    maximum_rank = int(np.argmax(errors))
    return {
        "sha256": _sha256_file(path),
        "row_count": len(rows),
        "errors": errors,
        "max_absolute_error": float(errors[maximum_rank]),
        "max_error_rank": maximum_rank,
    }


def _validate_replayed_geometry(
    output: Path,
    source_packet: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    geometry_path = _require_regular_file(output / "replayed_final_geometry.pt")
    payload = _load_checkpoint(geometry_path)
    expected_payload_keys = {
        "protocol_id",
        "source_protocol_id",
        "source_progress_checkpoint_sha256",
        "accepted_vae_checkpoint_sha256",
        "active_parameter_hash",
        "source_weight_index",
        "task_name",
        "tau",
        "z_sha256",
        "accepted_update",
        "geometry_constants",
        "hessian",
        "matrix",
        "eig",
        "current_low_basis",
        "current_low_projector",
        "metrics",
        "tensor_fingerprints",
        "installed_model_aggregate_sha256",
        "post_replay_model_aggregate_sha256",
    }
    _require(set(payload) == expected_payload_keys, "geometry payload schema mismatch")
    expected_metadata = {
        "protocol_id": RECOVERY_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": EXPECTED_SOURCE_PROGRESS_SHA256,
        "accepted_vae_checkpoint_sha256": EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "active_parameter_hash": EXPECTED_ACTIVE_PARAMETER_HASH,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "task_name": EXPECTED_TASK_NAME,
        "tau": EXPECTED_TASK_TAU,
        "z_sha256": EXPECTED_Z_SHA256,
        "accepted_update": MAX_ACCEPTED_UPDATES,
        "geometry_constants": {
            "dimension": DIMENSION,
            "epsilon": GEOMETRY_EPSILON,
            "old_beta": GEOMETRY_OLD_BETA,
            "low_threshold": GEOMETRY_LOW_THRESHOLD,
            "absolute_tolerance": REPLAY_ATOL,
        },
    }
    for name, expected in expected_metadata.items():
        _require(payload[name] == expected, f"geometry metadata mismatch: {name}")
    _require(
        payload["installed_model_aggregate_sha256"]
        == payload["post_replay_model_aggregate_sha256"],
        "geometry payload model aggregate mutation",
    )
    tensor_names = (
        "hessian",
        "matrix",
        "eig",
        "current_low_basis",
        "current_low_projector",
    )
    tensors = {name: payload[name] for name in tensor_names}
    _require(
        all(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float64
            and bool(torch.isfinite(value).all())
            for value in tensors.values()
        ),
        "geometry tensors must be finite CPU float64",
    )
    hessian = tensors["hessian"]
    matrix = tensors["matrix"]
    eig = tensors["eig"]
    low_basis = tensors["current_low_basis"]
    low_projector = tensors["current_low_projector"]
    _require(hessian.shape == (DIMENSION, DIMENSION), "geometry H shape mismatch")
    _require(matrix.shape == (DIMENSION, DIMENSION), "geometry M shape mismatch")
    _require(eig.shape == (DIMENSION,), "geometry eig shape mismatch")
    _require(
        low_basis.ndim == 2 and low_basis.shape[0] == DIMENSION,
        "geometry low-basis shape mismatch",
    )
    _require(
        low_projector.shape == (DIMENSION, DIMENSION),
        "geometry low-projector shape mismatch",
    )
    fingerprints = {name: _tensor_fingerprint(value) for name, value in tensors.items()}
    _require(
        payload["tensor_fingerprints"] == fingerprints,
        "geometry tensor fingerprint mismatch",
    )
    recomputed_matrix = hessian @ hessian.T
    matrix_errors = (matrix - recomputed_matrix).abs()
    matrix_max_error = float(matrix_errors.max())
    matrix_symmetry_error = float((matrix - matrix.T).abs().max())
    _require(matrix_max_error <= REPLAY_ATOL, "stored M != H @ H.T")
    _require(matrix_symmetry_error <= REPLAY_ATOL, "stored M is not symmetric")
    recomputed_eig = torch.linalg.eigvalsh(recomputed_matrix).clamp_min(0.0)
    eig_errors = (eig - recomputed_eig).abs().numpy()
    _require(
        bool(np.all(eig_errors <= REPLAY_ATOL)),
        "stored eigenvalue tensor does not match eigvalsh(H @ H.T)",
    )
    progress = source_packet["progress"]
    final_row = progress["state_rows"][-1]
    stored_spectrum = (
        source_packet["spectra"]
        .loc[
            source_packet["spectra"]["accepted_update"]
            .astype(int)
            .eq(MAX_ACCEPTED_UPDATES)
        ]
        .sort_values("rank")
    )
    _require(
        stored_spectrum["rank"].astype(int).tolist() == list(range(DIMENSION)),
        "stored endpoint spectrum grid mismatch",
    )
    stored_eig = stored_spectrum["m_eigenvalue"].to_numpy(dtype=np.float64)
    spectrum = _validate_replayed_spectrum_csv(
        output / "replayed_final_spectrum.csv",
        stored_eig=stored_eig,
        replayed_eig=eig.numpy(),
    )
    _require(
        bool(np.all(np.abs(eig.numpy() - stored_eig) <= REPLAY_ATOL)),
        "geometry eig does not match all 512 stored endpoint eigenvalues",
    )
    metrics = payload["metrics"]
    _require(isinstance(metrics, Mapping), "geometry metrics missing")
    closures = _recompute_geometry_metrics(
        hessian=hessian,
        matrix=matrix,
        eig=eig,
        low_basis=low_basis,
        low_projector=low_projector,
    )
    _require(
        set(metrics) == set(closures) | {"task_loss", "hessian_sec", "low_basis_hash"},
        "geometry metric closure schema mismatch",
    )
    closure_errors = {
        name: abs(_float(metrics[name]) - expected)
        for name, expected in closures.items()
    }
    _require(
        all(error <= REPLAY_ATOL for error in closure_errors.values()),
        "A/B/trace/quantile/count/rank metric closure failed",
    )
    source_metric_errors = {
        name: abs(_float(metrics[name]) - _float(final_row[name]))
        for name in set(metrics) - {"hessian_sec", "low_basis_hash"}
    }
    _require(
        all(error <= REPLAY_ATOL for error in source_metric_errors.values()),
        "geometry metrics do not match source endpoint",
    )
    _require(
        _float(metrics["hessian_sec"]) >= 0.0,
        "geometry Hessian timing is invalid",
    )
    projector_hash = _sha256_tensor(low_projector)
    projector_error = float((low_projector - low_basis @ low_basis.T).abs().max())
    _require(
        projector_error <= REPLAY_ATOL
        and projector_hash == metrics["low_basis_hash"] == final_row["low_basis_hash"],
        "low-basis projector/hash closure failed",
    )
    _require(
        closures["low_basis_orthogonality_max_abs"] <= REPLAY_ATOL
        and closures["low_basis_eigen_residual_relative"] <= REPLAY_ATOL,
        "low-basis orthogonality/eigen residual failed",
    )
    expected_artifact_audit = {
        "pass": True,
        "metadata_gates": {
            name: True
            for name in (
                "protocol_id",
                "source_protocol_id",
                "source_progress",
                "accepted_checkpoint",
                "active_parameter_hash",
                "source_weight_index",
                "task_name",
                "tau",
                "z_sha256",
                "accepted_update",
                "model_nonmutation",
                "constants",
            )
        },
        "matrix_from_hessian_max_abs_error": matrix_max_error,
        "matrix_symmetry_max_abs_error": matrix_symmetry_error,
        "eigvalsh_matrix_max_abs_error": float(eig_errors.max()),
        "stored_spectrum_max_abs_error": float(np.abs(eig.numpy() - stored_eig).max()),
        "low_projector_basis_max_abs_error": projector_error,
        "low_projector_sha256": projector_hash,
        "low_basis_orthogonality_max_abs": closures["low_basis_orthogonality_max_abs"],
        "low_basis_eigen_residual_relative": closures[
            "low_basis_eigen_residual_relative"
        ],
        "metric_closure_errors": closure_errors,
        "metric_closure_max_abs_error": max(closure_errors.values()),
        "source_metric_max_abs_error": max(source_metric_errors.values()),
        "tensor_fingerprints": fingerprints,
        "artifact_sha256": _sha256_file(geometry_path),
        "cpu_reload_pass": True,
    }
    producer_audit = replay["geometry_artifact_audit"]
    _require(
        _equivalent(
            producer_audit,
            expected_artifact_audit,
            atol=REPLAY_ATOL,
            rtol=0.0,
        ),
        "self-reported geometry artifact audit differs from independent audit",
    )
    _require(
        replay["geometry_artifact_sha256"] == _sha256_file(geometry_path)
        and replay["replayed_final_spectrum_sha256"] == spectrum["sha256"]
        and replay["replayed_final_spectrum_rows"] == DIMENSION
        and replay["hessian_sha256"] == fingerprints["hessian"]["sha256"]
        and replay["matrix_sha256"] == fingerprints["matrix"]["sha256"]
        and replay["spectrum_tensor_sha256"] == fingerprints["eig"]["sha256"]
        and replay["low_basis_sha256"] == fingerprints["current_low_basis"]["sha256"]
        and replay["low_projector_sha256"] == projector_hash,
        "replay artifact/tensor hash links mismatch",
    )
    _require(
        _close(
            replay["spectrum_max_absolute_error"],
            spectrum["max_absolute_error"],
            atol=1e-15,
        )
        and replay["spectrum_max_error_rank"] == spectrum["max_error_rank"],
        "replay spectrum maximum does not close over all 512 rows",
    )
    return {
        "payload": payload,
        "matrix_max_abs_error": matrix_max_error,
        "matrix_symmetry_max_abs_error": matrix_symmetry_error,
        "eig_max_abs_error": float(eig_errors.max()),
        "spectrum": spectrum,
        "metric_closure_max_abs_error": max(closure_errors.values()),
        "source_metric_max_abs_error": max(source_metric_errors.values()),
        "low_projector_sha256": projector_hash,
    }


def _validate_model_snapshot(snapshot: Mapping[str, Any], *, label: str) -> None:
    expected_keys = {
        "parameters",
        "buffers",
        "parameter_tensor_count",
        "parameter_element_count",
        "buffer_tensor_count",
        "buffer_element_count",
        "all_tensor_count",
        "all_element_count",
        "aggregate_sha256",
    }
    _require(set(snapshot) == expected_keys, f"{label} model snapshot schema mismatch")
    digest = hashlib.sha256()
    total_tensors = 0
    total_elements = 0
    for category in ("parameters", "buffers"):
        entries = snapshot[category]
        _require(
            isinstance(entries, Mapping), f"{label} {category} fingerprints missing"
        )
        category_elements = 0
        for name in sorted(entries):
            fingerprint = entries[name]
            _require(
                isinstance(name, str)
                and isinstance(fingerprint, Mapping)
                and set(fingerprint) == {"dtype", "shape", "numel", "sha256"}
                and isinstance(fingerprint["dtype"], str)
                and isinstance(fingerprint["shape"], list)
                and all(
                    isinstance(value, int) and value >= 0
                    for value in fingerprint["shape"]
                )
                and _int(fingerprint["numel"]) == math.prod(fingerprint["shape"])
                and _is_sha256(fingerprint["sha256"]),
                f"{label} {category} fingerprint invalid: {name}",
            )
            digest.update(category.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(json.dumps(fingerprint, sort_keys=True).encode("utf-8"))
            total_tensors += 1
            category_elements += int(fingerprint["numel"])
        singular = category[:-1]
        _require(
            snapshot[f"{singular}_tensor_count"] == len(entries)
            and snapshot[f"{singular}_element_count"] == category_elements,
            f"{label} {category} count closure mismatch",
        )
        total_elements += category_elements
    _require(
        snapshot["all_tensor_count"] == total_tensors
        and snapshot["all_element_count"] == total_elements
        and snapshot["aggregate_sha256"] == digest.hexdigest(),
        f"{label} aggregate fingerprint closure mismatch",
    )


def _validate_task_reconstruction_record(record: Mapping[str, Any]) -> None:
    expected_keys = {
        "pass",
        "gates",
        "task_name",
        "source_weight_index",
        "tau",
        "train_sample_count",
        "test_sample_count",
        "selected_task_tensors",
        "accepted_vae_checkpoint_sha256",
    }
    _require(set(record) == expected_keys, "runtime task audit schema mismatch")
    expected_gate_names = {
        "manifest_protocol",
        "source_weight_index",
        "record_task_name",
        "task_set_name",
        "tau",
        "full_train_sample_count",
        "full_test_sample_count",
        "selected_task_tensors",
        "accepted_checkpoint",
    }
    _require(
        record["pass"] is True
        and isinstance(record["gates"], Mapping)
        and set(record["gates"]) == expected_gate_names
        and all(value is True for value in record["gates"].values())
        and record["task_name"] == EXPECTED_TASK_NAME
        and record["source_weight_index"] == SOURCE_WEIGHT_INDEX
        and record["tau"] == EXPECTED_TASK_TAU
        and record["train_sample_count"] == 16_384
        and record["test_sample_count"] == 4_096
        and record["selected_task_tensors"] == EXPECTED_TASK_TENSORS
        and record["accepted_vae_checkpoint_sha256"]
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "runtime task audit identity mismatch",
    )


def _validate_geometry_reconstruction(
    reconstruction: Mapping[str, Any], geometry_payload: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        set(reconstruction) == EXPECTED_GEOMETRY_RECONSTRUCTION_KEYS,
        "geometry reconstruction schema mismatch",
    )
    reconstruction_inputs = _validate_reconstruction_input_files(
        reconstruction["reconstruction_input_files"]
    )
    _require(
        reconstruction["accepted_checkpoint_sha256"]
        == reconstruction["expected_accepted_checkpoint_sha256"]
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        and reconstruction["z_sha256"] == EXPECTED_Z_SHA256
        and reconstruction["active_tensor_count"] == ACTIVE_TENSOR_COUNT
        and reconstruction["active_parameter_count"] == ACTIVE_PARAMETER_COUNT
        and reconstruction["installed_active_parameter_hash"]
        == reconstruction["post_replay_active_parameter_hash"]
        == EXPECTED_ACTIVE_PARAMETER_HASH
        and reconstruction["base_parameter_gradient_slots"] == 0
        and reconstruction["post_replay_parameter_gradient_slots"] == 0
        and reconstruction["full_ce_batch"] is True,
        "geometry reconstruction lineage/runtime counters mismatch",
    )
    snapshots = reconstruction["model_tensor_snapshots"]
    _require(
        isinstance(snapshots, Mapping)
        and set(snapshots) == {"base", "installed", "post_replay"},
        "model tensor snapshots schema mismatch",
    )
    for name in ("base", "installed", "post_replay"):
        _validate_model_snapshot(snapshots[name], label=name)
    base = snapshots["base"]
    installed = snapshots["installed"]
    post = snapshots["post_replay"]
    _require(
        set(base["parameters"])
        == set(installed["parameters"])
        == set(post["parameters"])
        and set(base["buffers"]) == set(installed["buffers"]) == set(post["buffers"]),
        "model parameter/buffer names changed",
    )
    changed_parameters = sorted(
        name
        for name in base["parameters"]
        if base["parameters"][name] != installed["parameters"][name]
    )
    changed_buffers = sorted(
        name
        for name in base["buffers"]
        if base["buffers"][name] != installed["buffers"][name]
    )
    _require(
        installed == post,
        "parameter or buffer fingerprint changed during geometry replay",
    )
    installation = reconstruction["model_installation_audit"]
    _require(
        set(installation)
        == {
            "pass",
            "expected_changed_parameter_count",
            "changed_parameter_count",
            "changed_parameters",
            "changed_buffer_count",
            "changed_buffers",
            "base_aggregate_sha256",
            "installed_aggregate_sha256",
            "base_parameter_tensor_count",
            "base_buffer_tensor_count",
        }
        and installation.get("pass") is True
        and installation.get("expected_changed_parameter_count") == ACTIVE_TENSOR_COUNT
        and installation.get("changed_parameter_count")
        == len(changed_parameters)
        == ACTIVE_TENSOR_COUNT
        and installation.get("changed_parameters") == changed_parameters
        and installation.get("changed_buffer_count") == len(changed_buffers) == 0
        and installation.get("changed_buffers") == []
        and installation.get("base_aggregate_sha256") == base["aggregate_sha256"]
        and installation.get("installed_aggregate_sha256")
        == installed["aggregate_sha256"]
        and installation.get("base_parameter_tensor_count")
        == base["parameter_tensor_count"]
        and installation.get("base_buffer_tensor_count") == base["buffer_tensor_count"],
        "model installation audit mismatch",
    )
    nonmutation = reconstruction["model_replay_nonmutation_audit"]
    _require(
        set(nonmutation)
        == {
            "pass",
            "all_parameters_bitwise_unchanged",
            "all_buffers_bitwise_unchanged",
            "parameter_tensor_count",
            "buffer_tensor_count",
            "installed_aggregate_sha256",
            "post_replay_aggregate_sha256",
        }
        and nonmutation.get("pass") is True
        and nonmutation.get("all_parameters_bitwise_unchanged") is True
        and nonmutation.get("all_buffers_bitwise_unchanged") is True
        and nonmutation.get("parameter_tensor_count")
        == installed["parameter_tensor_count"]
        and nonmutation.get("buffer_tensor_count") == installed["buffer_tensor_count"]
        and nonmutation.get("installed_aggregate_sha256")
        == nonmutation.get("post_replay_aggregate_sha256")
        == installed["aggregate_sha256"],
        "model replay nonmutation audit mismatch",
    )
    _require(
        geometry_payload["installed_model_aggregate_sha256"]
        == geometry_payload["post_replay_model_aggregate_sha256"]
        == installed["aggregate_sha256"],
        "geometry payload/model snapshot aggregate link mismatch",
    )
    for name in ("task_at_load", "task_immediately_pre_replay", "task_post_replay"):
        _validate_task_reconstruction_record(reconstruction[name])
    _require(
        reconstruction["task_at_load"]
        == reconstruction["task_immediately_pre_replay"]
        == reconstruction["task_post_replay"],
        "runtime task tensors changed around replay",
    )
    runtime = reconstruction["runtime_provenance"]
    _require(
        set(runtime)
        == {
            "pass",
            "expected_runtime",
            "observed_runtime",
            "requested_device",
            "resolved_cuda_device_index",
            "cuda_device_name",
            "cuda_compute_capability",
            "cuda_total_memory_bytes",
            "cuda_current_device_index",
            "cudnn_version",
        }
        and runtime.get("pass") is True
        and runtime.get("expected_runtime")
        == runtime.get("observed_runtime")
        == EXPECTED_RUNTIME
        and runtime.get("requested_device") == "cuda:0"
        and runtime.get("resolved_cuda_device_index") == 0,
        "runtime provenance mismatch",
    )
    return {
        "parameter_count": installed["parameter_tensor_count"],
        "buffer_count": installed["buffer_tensor_count"],
        "aggregate_sha256": installed["aggregate_sha256"],
        "changed_parameters": changed_parameters,
        "reconstruction_input_files": reconstruction_inputs,
    }


def _validate_replay_and_final_checkpoint(
    output: Path,
    source_packet: Mapping[str, Any],
    recovery_audit: Mapping[str, Any],
) -> dict[str, Any]:
    progress = source_packet["progress"]
    replay = recovery_audit["final_geometry_replay"]
    _require(isinstance(replay, Mapping), "final geometry replay payload missing")
    _require(set(replay) == EXPECTED_REPLAY_KEYS, "final replay schema mismatch")
    execution = recovery_audit["geometry_execution"]
    _require(isinstance(execution, Mapping), "geometry execution payload missing")
    _validate_geometry_execution(execution)
    stored = progress["state_rows"][-1]
    decision = _load_json(output / "decision.json")
    replay_metrics = decision["final_metrics"]
    errors = replay["metric_errors"]
    _require(
        isinstance(replay_metrics, Mapping) and isinstance(errors, Mapping),
        "replay metrics/errors missing",
    )
    expected_metric_names = {
        name for name in replay_metrics if name not in {"hessian_sec", "low_basis_hash"}
    }
    _require(
        set(errors) == expected_metric_names, "replay metric-error schema mismatch"
    )
    recomputed_errors: dict[str, float] = {}
    for name in expected_metric_names:
        recomputed_errors[name] = abs(
            _float(replay_metrics[name]) - _float(stored[name])
        )
        _require(
            _close(errors[name], recomputed_errors[name], atol=1e-15),
            f"replay stored error mismatch: {name}",
        )
    maximum = max(recomputed_errors.values())
    _require(maximum <= REPLAY_ATOL, "final replay metric error too large")
    _require(
        _close(replay["metric_max_absolute_error"], maximum, atol=1e-15),
        "replay metric maximum mismatch",
    )
    _require(
        replay["metric_max_error_name"] in errors
        and errors[replay["metric_max_error_name"]] == max(errors.values()),
        "replay maximum metric name mismatch",
    )
    _require(
        _close(replay["metric_absolute_tolerance"], REPLAY_ATOL, atol=0)
        and _close(replay["spectrum_absolute_tolerance"], REPLAY_ATOL, atol=0),
        "replay tolerance mismatch",
    )
    stored_spectrum = source_packet["spectra"].loc[
        source_packet["spectra"]["accepted_update"].astype(int).eq(MAX_ACCEPTED_UPDATES)
    ]
    stored_eig = stored_spectrum["m_eigenvalue"].to_numpy(dtype=np.float64)
    _require(len(stored_eig) == DIMENSION, "stored endpoint spectrum missing")
    _require(
        replay["spectrum_eigenvalue_count"] == DIMENSION
        and 0 <= _int(replay["spectrum_max_error_rank"]) < DIMENSION
        and 0.0 <= _float(replay["spectrum_max_absolute_error"]) <= REPLAY_ATOL,
        "final 512-eigenvalue replay record is invalid",
    )
    _require(
        replay["low_basis_hash_matches"] is True
        and replay_metrics["low_basis_hash"] == stored["low_basis_hash"],
        "replay low-basis link mismatch",
    )
    _require(
        replay["pass"] is True
        and replay["active_parameter_hash_matches"] is True
        and replay["hessian_shape"] == [DIMENSION, DIMENSION]
        and replay["matrix_shape"] == [DIMENSION, DIMENSION]
        and replay["spectrum_shape"] == [DIMENSION],
        "final H/M/spectrum replay payload mismatch",
    )
    geometry = _validate_replayed_geometry(output, source_packet, replay)
    geometry_payload = geometry["payload"]
    _require(
        _equivalent(replay_metrics, geometry_payload["metrics"], atol=0.0, rtol=0.0),
        "decision metrics differ from replayed geometry metrics",
    )
    _require(
        replay["low_basis_shape"] == list(geometry_payload["current_low_basis"].shape)
        and replay["low_projector_shape"] == [DIMENSION, DIMENSION],
        "replay low-basis/projector shape record mismatch",
    )
    reconstruction = recovery_audit["geometry_reconstruction"]
    _require(isinstance(reconstruction, Mapping), "geometry reconstruction missing")
    reconstruction_review = _validate_geometry_reconstruction(
        reconstruction, geometry_payload
    )
    _require(
        reconstruction_review["changed_parameters"]
        == sorted(progress["active_model_state"]),
        "installed model change-set differs from terminal active tensors",
    )
    final = _load_checkpoint(output / "final_checkpoint.pt")
    _require(
        set(final) == EXPECTED_FINAL_CHECKPOINT_KEYS,
        "recovery final checkpoint schema mismatch",
    )
    expected_common = {
        "protocol_id": RECOVERY_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": EXPECTED_SOURCE_PROGRESS_SHA256,
        "parent_checkpoint_sha256": EXPECTED_PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": EXPECTED_PARENT_PROGRESS_SHA256,
        "selected_arm": "low",
        "accepted_updates": MAX_ACCEPTED_UPDATES,
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "termination": "max_updates_reached",
        "transition_chain_sha256": EXPECTED_TRANSITION_CHAIN_SHA256,
        "active_parameter_hash": EXPECTED_ACTIVE_PARAMETER_HASH,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "z_sha256": EXPECTED_Z_SHA256,
        "accepted_vae_checkpoint_sha256": (EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256),
    }
    for name, value in expected_common.items():
        _require(final[name] == value, f"final checkpoint metadata mismatch: {name}")
    source_state = progress["active_model_state"]
    final_state = final["active_model_state"]
    _require(
        set(source_state) == set(final_state),
        "final/source active tensor keys mismatch",
    )
    for name in source_state:
        _require(
            source_state[name].dtype == final_state[name].dtype
            and source_state[name].shape == final_state[name].shape
            and torch.equal(source_state[name], final_state[name]),
            f"final checkpoint tensor lineage mismatch: {name}",
        )
    final_hash, final_count = _named_tensor_hash(final_state)
    _require(
        final_hash == EXPECTED_ACTIVE_PARAMETER_HASH,
        "final checkpoint active hash mismatch",
    )
    _require(
        final_count == ACTIVE_PARAMETER_COUNT,
        "final checkpoint parameter count mismatch",
    )
    _require(
        final["stored_state_metrics"] == stored,
        "final checkpoint stored metrics mismatch",
    )
    checkpoint_audit = recovery_audit["checkpoint_lineage"]
    expected_checkpoint_gate_names = {
        "protocol_id",
        "source_protocol_id",
        "source_progress_checkpoint_sha256",
        "parent_checkpoint_sha256",
        "parent_progress_checkpoint_sha256",
        "selected_arm",
        "accepted_updates",
        "new_accepted_updates",
        "termination",
        "transition_chain",
        "active_hash",
        "stored_metrics",
        "source_weight_index",
        "z_sha256",
        "accepted_vae_checkpoint_sha256",
    }
    _require(
        set(checkpoint_audit)
        == {
            "pass",
            "bitwise_tensor_equality",
            "active_tensor_count",
            "active_parameter_count",
            "active_parameter_hash",
            "metadata_gates",
            "final_checkpoint_sha256",
            "cpu_reload_pass",
        }
        and checkpoint_audit.get("pass") is True
        and checkpoint_audit.get("bitwise_tensor_equality") is True
        and checkpoint_audit.get("cpu_reload_pass") is True
        and checkpoint_audit.get("active_tensor_count") == ACTIVE_TENSOR_COUNT
        and checkpoint_audit.get("active_parameter_count") == ACTIVE_PARAMETER_COUNT
        and checkpoint_audit.get("active_parameter_hash")
        == EXPECTED_ACTIVE_PARAMETER_HASH
        and isinstance(checkpoint_audit.get("metadata_gates"), Mapping)
        and set(checkpoint_audit["metadata_gates"]) == expected_checkpoint_gate_names
        and all(value is True for value in checkpoint_audit["metadata_gates"].values()),
        "serialized checkpoint-lineage audit mismatch",
    )
    checkpoint_sha256 = _sha256_file(output / "final_checkpoint.pt")
    _require(
        checkpoint_audit.get("final_checkpoint_sha256") == checkpoint_sha256,
        "checkpoint audit file hash mismatch",
    )
    return {
        "replay": replay,
        "metric_max_abs_error": maximum,
        "spectrum_max_abs_error": geometry["spectrum"]["max_absolute_error"],
        "final_checkpoint_sha256": checkpoint_sha256,
        "geometry": geometry,
        "reconstruction": reconstruction_review,
    }


def _validate_plot_data_and_pngs(
    output: Path, source_packet: Mapping[str, Any]
) -> dict[str, Any]:
    states = source_packet["states"]
    spectra = source_packet["spectra"]
    proposals = source_packet["proposals"]
    trajectory_columns = {
        "accepted_update",
        "exact_a_per_dim",
        "damped_full_burg_per_dim",
        "a_low90_abs_per_dim",
        "a_gt1",
        "m_p50",
        "effective_rank",
    }
    _require(
        trajectory_columns.issubset(states.columns),
        "trajectory backing columns missing",
    )
    _require(len(states) == 101, "trajectory backing row count mismatch")
    _require(
        states["accepted_update"].astype(int).tolist() == list(range(101))
        and np.isfinite(
            states[list(trajectory_columns - {"phase"})]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        ).all(),
        "trajectory plot backing values are invalid",
    )
    selected_updates = [0, START_UPDATE, MAX_ACCEPTED_UPDATES]
    spectra_expected = spectra.loc[
        spectra["accepted_update"].astype(int).isin(selected_updates),
        ["accepted_update", "rank", "m_eigenvalue", "a_contribution", "phase"],
    ].reset_index(drop=True)
    _require(
        len(spectra_expected) == 3 * DIMENSION, "spectra plot backing is incomplete"
    )
    expected_grid = [
        (update, rank) for update in selected_updates for rank in range(DIMENSION)
    ]
    observed_grid = list(
        zip(
            spectra_expected["accepted_update"].astype(int),
            spectra_expected["rank"].astype(int),
            strict=True,
        )
    )
    _require(observed_grid == expected_grid, "spectra plot backing order mismatch")
    eig = spectra_expected["m_eigenvalue"].to_numpy(dtype=np.float64)
    contribution = spectra_expected["a_contribution"].to_numpy(dtype=np.float64)
    _require(
        np.isfinite(eig).all()
        and np.isfinite(contribution).all()
        and float(np.max(np.abs(contribution - np.square(eig - 1.0)))) <= 1e-9,
        "spectra plot backing values are invalid",
    )
    continuation = proposals.loc[
        proposals["target_update"].astype(int).gt(START_UPDATE)
    ].copy()
    radius_errors = np.abs(
        continuation["direction_norm"].to_numpy(dtype=np.float64) - base_target_norm()
    )
    _require(
        len(radius_errors) == NEW_ACCEPTED_UPDATES
        and np.isfinite(radius_errors).all()
        and float(radius_errors.max()) <= RECOVERY_RADIUS_ATOL,
        "direction plot backing is invalid",
    )
    finalizer_source = (
        output / "executed_recovery_finalizer_source_snapshot.py"
    ).read_text(encoding="utf-8")
    for required_literal in (
        "recovery_trajectory.png",
        "recovery_spectra.png",
        "recovery_direction_radius_audit.png",
        "exact_a_per_dim",
        "damped_full_burg_per_dim",
        "a_low90_abs_per_dim",
        "direction_norm",
        "m_eigenvalue",
    ):
        _require(
            required_literal in finalizer_source,
            f"plot source literal missing: {required_literal}",
        )
    pngs = {
        name: _validate_png(output / name)
        for name in (
            "recovery_trajectory.png",
            "recovery_spectra.png",
            "recovery_direction_radius_audit.png",
        )
    }
    expected_dimensions = {
        "recovery_trajectory.png": (3060, 1800),
        "recovery_spectra.png": (1800, 1080),
        "recovery_direction_radius_audit.png": (1980, 1080),
    }
    for name, dimensions in expected_dimensions.items():
        _require(
            (pngs[name]["width"], pngs[name]["height"]) == dimensions,
            f"{name} dimensions mismatch",
        )
    return {
        "trajectory_rows": len(states),
        "spectra_rows": len(spectra_expected),
        "direction_rows": len(continuation),
        "direction_radius_max_abs_error": float(radius_errors.max()),
        "pngs": pngs,
    }


def _validate_decision(
    output: Path,
    publication: Mapping[str, Any],
    source_packet: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    decision = _load_json(output / "decision.json")
    _require(set(decision) == EXPECTED_DECISION_KEYS, "decision schema mismatch")
    _require(
        decision["protocol_id"] == RECOVERY_PROTOCOL_ID, "decision protocol mismatch"
    )
    _require(
        decision["source_protocol_id"] == SOURCE_PROTOCOL_ID,
        "decision source protocol mismatch",
    )
    _require(decision["recovery_valid"] is True, "decision recovery invalid")
    expected_scalars = {
        "accepted_updates": MAX_ACCEPTED_UPDATES,
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "selected_arm": "low",
        "termination": "max_updates_reached",
        "original_runner_audit_pass": False,
        "original_runner_failed_gates": EXPECTED_ORIGINAL_FAILED_GATES,
        "final_checkpoint_sha256": replay["final_checkpoint_sha256"],
        "accepted_vae_checkpoint_sha256": (EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256),
        "recovery_task_manifest_sha256": (EXPECTED_RECOVERY_TASK_MANIFEST_SHA256),
        "recovery_import_manifest_sha256": (EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256),
    }
    for name, value in expected_scalars.items():
        _require(
            _equivalent(decision[name], value, atol=1e-15), f"decision mismatch: {name}"
        )
    recovery_audit = _load_json(output / "recovery_audit.json")
    _require(
        decision["recovery_validity_gates"]
        == recovery_audit["recovery_validity_gates"],
        "decision recovery-validity gates mismatch",
    )
    _require(
        _equivalent(
            decision["recovery_radius_audit"],
            recovery_audit["independent_recovery_audit"]["radius_audit"],
            atol=1e-15,
        ),
        "decision radius audit mismatch",
    )
    _require(
        _equivalent(
            decision["final_geometry_replay"],
            recovery_audit["final_geometry_replay"],
            atol=1e-15,
        ),
        "decision final replay mismatch",
    )
    _require(
        decision["recovery_audit_sha256"]
        == _sha256_file(output / "recovery_audit.json"),
        "decision recovery-audit hash mismatch",
    )
    _require(_float(decision["elapsed_sec"]) >= 0.0, "decision elapsed time invalid")
    outcome = source_packet["base"]._derive_outcome(
        states=source_packet["states"],
        spectra=source_packet["spectra"],
        proposals=source_packet["proposals"],
        tolerances={
            key: _float(value)
            for key, value in source_packet["progress"]["tolerances"].items()
        },
        accepted_updates=MAX_ACCEPTED_UPDATES,
        termination="max_updates_reached",
        history_summary=source_packet["history"]["summary"],
    )
    _require(
        _equivalent(decision["outcome"], outcome, atol=REPLAY_ATOL, rtol=0.0),
        "decision scientific outcome does not independently recompute",
    )
    _require(
        decision["scientific_success"] is outcome["scientific_success"],
        "decision scientific-success mismatch",
    )
    _require(
        set(outcome["success_gates"]) == EXPECTED_SUCCESS_GATES,
        "scientific-success gate schema changed",
    )
    finalized = publication["finalized"]
    _require(
        finalized["scientific_success"] is outcome["scientific_success"],
        "FINALIZED scientific-success mismatch",
    )
    for name, value in (
        ("accepted_updates", MAX_ACCEPTED_UPDATES),
        ("new_accepted_updates", NEW_ACCEPTED_UPDATES),
        ("termination", "max_updates_reached"),
    ):
        _require(finalized[name] == value, f"FINALIZED mismatch: {name}")
    stored = source_packet["progress"]["state_rows"][-1]
    expected_final_names = set(stored) - {
        "accepted_update",
        "parameter_hash",
        "parent_frozen_low_energy",
        "phase",
    }
    _require(
        set(decision["final_metrics"]) == expected_final_names,
        "decision final metric schema mismatch",
    )
    for name, value in decision["final_metrics"].items():
        if name == "hessian_sec":
            _require(_float(value) >= 0.0, "decision replay timing invalid")
        else:
            _require(
                _equivalent(value, stored[name], atol=REPLAY_ATOL, rtol=0),
                f"decision final metric mismatch: {name}",
            )
    return {"decision": decision, "outcome": outcome}


def _validate_recovery_packet(
    output: Path,
    source: Path,
    frozen_manifest: Path,
) -> dict[str, Any]:
    output = _require_directory(output)
    _regular_file_names(output)
    source = _require_directory(source)
    frozen_manifest = _require_regular_file(frozen_manifest)
    _require(
        output == _absolute_without_resolve(DEFAULT_RECOVERY_OUTPUT),
        "recovery output path differs from frozen production output",
    )
    _require(
        source == _absolute_without_resolve(SOURCE_STAGING),
        "source staging path differs from frozen source",
    )
    _require(
        frozen_manifest == _absolute_without_resolve(FROZEN_INPUT_MANIFEST),
        "frozen input manifest path differs from protocol",
    )
    _require(
        not _path_is_within(output, source) and not _path_is_within(source, output),
        "source staging and recovery output are not disjoint",
    )
    manifest = _load_frozen_manifest(frozen_manifest)
    source_before = _validate_source_staging(source, manifest)
    base = _load_pinned_continuation_reviewer(manifest)
    source_packet = _validate_source_checkpoint_and_tables(source, manifest, base)
    source_packet["base"] = base
    diagnostics = _direction_diagnostics(source_packet["proposals"])
    publication = _validate_manifest_and_publication(output, manifest)
    copied_evidence = _validate_copied_source_evidence(
        output,
        source,
        manifest,
        publication["config"],
    )
    audits = _validate_original_and_recovery_audits(
        output, manifest, source_packet, diagnostics
    )
    replay = _validate_replay_and_final_checkpoint(
        output, source_packet, audits["recovery"]
    )
    plots = _validate_plot_data_and_pngs(output, source_packet)
    decision = _validate_decision(
        output, publication, source_packet, diagnostics, replay
    )
    source_after = _validate_source_staging(source, manifest)
    _require(source_before == source_after, "source staging changed during review")
    return {
        "manifest": manifest,
        "source": source_before,
        "publication": publication,
        "copied_evidence": copied_evidence,
        "source_packet": source_packet,
        "diagnostics": diagnostics,
        "audits": audits,
        "replay": replay,
        "plots": plots,
        "decision": decision,
    }


def review_recovery(
    output: Path = DEFAULT_RECOVERY_OUTPUT,
    source: Path = SOURCE_STAGING,
    frozen_manifest: Path = FROZEN_INPUT_MANIFEST,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = _absolute_without_resolve(output)
    source = _absolute_without_resolve(source)
    frozen_manifest = _absolute_without_resolve(frozen_manifest)
    review = Review()
    source_fingerprint_before = _read_only_fingerprint(source)
    output_fingerprint_before = _read_only_fingerprint(output)
    packet = review.phase(
        "recovery_packet",
        lambda: _validate_recovery_packet(output, source, frozen_manifest),
    )
    source_fingerprint_after = _read_only_fingerprint(source)
    output_fingerprint_after = _read_only_fingerprint(output)
    read_only = (
        source_fingerprint_before == source_fingerprint_after
        and output_fingerprint_before == output_fingerprint_after
    )
    review.gates["read_only_footprint"] = read_only
    if not read_only:
        review.errors.append("review changed source staging or recovery output")
    valid = bool(review.gates and all(review.gates.values()) and not review.errors)
    outcome = packet["decision"]["outcome"] if packet is not None else None
    review.limitations.extend(
        [
            "The CPU reviewer does not rerun the GPU HVP. It verifies the frozen one-call replay record, exact tensor lineage, full stored trajectory, and replay errors against the endpoint.",
            "Static call-site inspection and runtime counters are strong audit evidence but cannot prove behavior outside the frozen finalizer source and packet.",
            "Scientific interpretation remains limited to one state, one VAE, and one selected normalized-bisector trajectory.",
        ]
    )
    details: dict[str, Any] = {}
    if packet is not None:
        details = {
            "source_tree_sha256": packet["source"]["tree_sha256"],
            "artifact_hashes": packet["publication"]["artifact_hashes"],
            "finalizer_static_source": packet["publication"]["finalizer_static_source"],
            "preflight": packet["source_packet"]["preflight"],
            "spectrum_review": packet["source_packet"]["spectrum_review"],
            "history_summary": packet["source_packet"]["history"]["summary"],
            "direction_diagnostics": packet["diagnostics"],
            "replay": {
                "metric_max_abs_error": packet["replay"]["metric_max_abs_error"],
                "spectrum_max_abs_error": packet["replay"]["spectrum_max_abs_error"],
                "final_checkpoint_sha256": packet["replay"]["final_checkpoint_sha256"],
            },
            "plots": packet["plots"],
        }
    return {
        "review_protocol_id": REVIEW_PROTOCOL_ID,
        "protocol_id": RECOVERY_PROTOCOL_ID,
        "reviewer_source_sha256": _sha256_file(Path(__file__)),
        "frozen_input_manifest_sha256": EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
        "reviewed_output": str(output),
        "source_staging": str(source),
        "valid": valid,
        "recovery_valid": valid,
        "scientific_success": outcome["scientific_success"]
        if valid and outcome
        else None,
        "outcome": outcome if valid else None,
        "failed_gates": [name for name, passed in review.gates.items() if not passed],
        "gates": review.gates,
        "details": details,
        "errors": review.errors,
        "limitations": review.limitations,
        "elapsed_sec": time.perf_counter() - started,
        "read_only": read_only,
    }


def _write_review_json(
    path: Path, result: Mapping[str, Any], protected: Sequence[Path]
) -> None:
    path = _absolute_without_resolve(path)
    _require(path.suffix == ".json", "review output must use .json suffix")
    _require_directory(path.parent)
    _require_no_symlink_components(path, allow_missing_leaf=True)
    _require(not os.path.lexists(path), "review output already exists")
    _require(
        all(not _path_is_within(path, directory) for directory in protected),
        "review JSON must be outside source staging and recovery production",
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict independent read-only review of recovery finalization"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RECOVERY_OUTPUT)
    parser.add_argument("--source-staging", type=Path, default=SOURCE_STAGING)
    parser.add_argument(
        "--frozen-input-manifest", type=Path, default=FROZEN_INPUT_MANIFEST
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        help="Optional new JSON path outside source and recovery directories",
    )
    args = parser.parse_args()
    result = review_recovery(
        args.output.resolve(),
        args.source_staging.resolve(),
        args.frozen_input_manifest.resolve(),
    )
    if args.review_json is not None:
        _write_review_json(
            args.review_json,
            result,
            [args.output.resolve(), args.source_staging.resolve()],
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
