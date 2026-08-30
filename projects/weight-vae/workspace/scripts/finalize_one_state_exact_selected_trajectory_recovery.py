from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import stat
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LIVE_SOURCE = Path(os.path.abspath(__file__))
ROOT = LIVE_SOURCE.parents[1]
PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_recovery_v1"
SOURCE_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
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
DEFAULT_OUTPUT = OUTPUT_ROOT / "postgoal_relaxed_cancellation_recovery_finalization_v1"
PRODUCTION_DEVICE = "cuda:0"

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
EXPECTED_PRODUCER_SHA256 = (
    "d79462816ff95b2c3f1a94bed06f8a2c12d6a25610dd9385d9361da2e970f39f"
)
EXPECTED_REVIEWER_SHA256 = (
    "57ae23d336cc348548b744c90e41c1560c5867af512bf4a5b635fc344737adbb"
)
EXPECTED_PROGRESS_SHA256 = (
    "ab3366870d62beb71b4d59fd2d54e3cb7354963a16e17a8b9cc13527a4d25311"
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
EXPECTED_PRODUCER_DEPENDENCY_MANIFEST_SHA256 = (
    "9437b54d9a6379adaab9a7e2bd578caf55e3c879bf861f8a0150f124b577eb32"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_ACCEPTED_RUN_DIR = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
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
EXPECTED_ACCEPTED_RUN_CHECKPOINT_SHA256 = EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
# This is the only line masked by _normalized_source_sha256.
# fmt: off
EXPECTED_NORMALIZED_SOURCE_SHA256 = "e25be1074055a66ad68c63acc1f722191efbbf8a827ecd73adfeaa090329a599"
# fmt: on
EXPECTED_ORIGINAL_FAILED_GATES = ("continuation_diagnostics_recompute",)

DIMENSION = 512
START_UPDATE = 17
MAX_ACCEPTED_UPDATES = 100
ACTIVE_TENSOR_COUNT = 31
ACTIVE_PARAMETER_COUNT = 11_685_120
SOURCE_WEIGHT_INDEX = 378
EXPECTED_TASK_NAME = "fashion_mnist"
EXPECTED_TASK_TAU = 1.1614345407370807
EXPECTED_TASK_TRAIN_COUNT = 16_384
EXPECTED_TASK_TEST_COUNT = 4_096
EXPECTED_TORCH_VERSION = "2.10.0+cu128"
EXPECTED_TORCHVISION_VERSION = "0.25.0+cu128"
EXPECTED_PYTHON_VERSION = "3.12.12"
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_RUNTIME = {
    "python": EXPECTED_PYTHON_VERSION,
    "torch": EXPECTED_TORCH_VERSION,
    "torch_cuda": EXPECTED_TORCH_CUDA_VERSION,
    "torchvision": EXPECTED_TORCHVISION_VERSION,
}
FINAL_REPLAY_ATOL = 1e-9
RECOVERY_RADIUS_ATOL = 5e-9
ORIGINAL_RADIUS_ATOL = 1e-9
GEOMETRY_EPSILON = 1e-4
GEOMETRY_OLD_BETA = 22.536727828943093
GEOMETRY_LOW_THRESHOLD = 0.1

EXPECTED_PROGRESS_KEYS = {
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
EXPECTED_REPLAYED_GEOMETRY_KEYS = {
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
EXPECTED_STAGED_FILE_SET = EXPECTED_MANIFESTED_ARTIFACTS | {
    "artifact_manifest.json",
    "FINALIZED.json",
    "INCOMPLETE",
    "run.log",
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


class RecoveryError(RuntimeError):
    pass


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def _tee_to_log(path: Path) -> Any:
    original_stdout, original_stderr = sys.stdout, sys.stderr
    handle = path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, handle)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        handle.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _normalized_source_sha256(path: Path = LIVE_SOURCE) -> str:
    prefix = "EXPECTED_NORMALIZED_SOURCE_SHA256 = "
    masked = 0
    normalized: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith(prefix):
            normalized.append(f'{prefix}"<FROZEN>"\n')
            masked += 1
        else:
            normalized.append(line)
    if masked != 1:
        raise RecoveryError(
            f"normalized source must mask exactly one self-freeze constant, got {masked}"
        )
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _absolute_without_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lstat_no_symlink_components(
    path: Path, *, allow_missing_leaf: bool = False
) -> Path:
    absolute = _absolute_without_resolve(path)
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current /= part
        is_leaf = index == len(absolute.parts[1:]) - 1
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and is_leaf:
                return absolute
            raise RecoveryError(
                f"declared path component does not exist: {current}"
            ) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise RecoveryError(f"symlink path component is forbidden: {current}")
    return absolute


def _lstat_regular_file(path: Path) -> Path:
    absolute = _lstat_no_symlink_components(path)
    metadata = os.lstat(absolute)
    if not stat.S_ISREG(metadata.st_mode):
        raise RecoveryError(f"expected regular non-symlink file: {absolute}")
    return absolute


def _lstat_directory(path: Path) -> Path:
    absolute = _lstat_no_symlink_components(path)
    metadata = os.lstat(absolute)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RecoveryError(f"expected non-symlink directory: {absolute}")
    return absolute


def _verify_finalizer_source_freeze(
    *,
    snapshot: Path | None = None,
    protocol_snapshot: Path | None = None,
) -> dict[str, Any]:
    live = _lstat_regular_file(LIVE_SOURCE)
    protocol = _lstat_regular_file(RECOVERY_PROTOCOL)
    live_raw = _sha256_file(live)
    live_normalized = _normalized_source_sha256(live)
    if (
        EXPECTED_NORMALIZED_SOURCE_SHA256 == "TO_BE_FROZEN"
        or live_normalized != EXPECTED_NORMALIZED_SOURCE_SHA256
    ):
        raise RecoveryError("recovery finalizer normalized source hash mismatch")
    if _sha256_file(protocol) != EXPECTED_RECOVERY_PROTOCOL_SHA256:
        raise RecoveryError("recovery protocol hash mismatch")
    result: dict[str, Any] = {
        "live_source_sha256": live_raw,
        "live_normalized_source_sha256": live_normalized,
        "expected_normalized_source_sha256": EXPECTED_NORMALIZED_SOURCE_SHA256,
        "recovery_protocol_sha256": EXPECTED_RECOVERY_PROTOCOL_SHA256,
    }
    if snapshot is not None:
        frozen = _lstat_regular_file(snapshot)
        snapshot_raw = _sha256_file(frozen)
        snapshot_normalized = _normalized_source_sha256(frozen)
        if snapshot_raw != live_raw or snapshot_normalized != live_normalized:
            raise RecoveryError("copied recovery finalizer source identity mismatch")
        result.update(
            {
                "snapshot_source_sha256": snapshot_raw,
                "snapshot_normalized_source_sha256": snapshot_normalized,
            }
        )
    if protocol_snapshot is not None:
        frozen_protocol = _lstat_regular_file(protocol_snapshot)
        snapshot_protocol_hash = _sha256_file(frozen_protocol)
        if snapshot_protocol_hash != EXPECTED_RECOVERY_PROTOCOL_SHA256:
            raise RecoveryError("copied recovery protocol snapshot hash mismatch")
        result["snapshot_protocol_sha256"] = snapshot_protocol_hash
    return result


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pd.DataFrame(list(rows)).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolved_within(root: Path, path: Path) -> Path:
    resolved_root = _lstat_directory(root)
    declared = _absolute_without_resolve(path)
    try:
        declared.relative_to(resolved_root)
    except ValueError as error:
        raise RecoveryError(f"path escapes repository root: {path}") from error
    _lstat_no_symlink_components(declared)
    resolved = declared.resolve(strict=True)
    if resolved != declared:
        raise RecoveryError(f"declared path canonicalization changed: {path}")
    return resolved


@dataclass(frozen=True)
class FrozenInputGuard:
    repository_root: Path
    manifest_path: Path
    expected_manifest_sha256: str
    manifest: Mapping[str, Any]
    source_staging: Path

    @classmethod
    def create(
        cls,
        *,
        repository_root: Path,
        manifest_path: Path,
        expected_manifest_sha256: str,
    ) -> FrozenInputGuard:
        repository_root = _lstat_directory(repository_root)
        manifest_path = _lstat_regular_file(
            _resolved_within(repository_root, manifest_path)
        )
        observed_manifest_hash = _sha256_file(manifest_path)
        if observed_manifest_hash != expected_manifest_sha256:
            raise RecoveryError(
                "frozen recovery input manifest hash mismatch: "
                f"{observed_manifest_hash} != {expected_manifest_sha256}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise RecoveryError("frozen recovery input manifest is not an object")
        required_keys = {
            "protocol_id",
            "source_staging_relative_path",
            "expected_exact_file_set",
            "files_sha256",
            "expected_progress",
            "expected_original_audit",
            "recovery_dependencies",
            "recovery_radius_rule",
        }
        if set(manifest) != required_keys:
            raise RecoveryError("frozen recovery input manifest schema mismatch")
        if manifest.get("protocol_id") != PROTOCOL_ID:
            raise RecoveryError("frozen recovery input protocol mismatch")
        source_relative = manifest.get("source_staging_relative_path")
        source_relative_path = (
            Path(source_relative) if isinstance(source_relative, str) else None
        )
        if (
            source_relative_path is None
            or source_relative_path.is_absolute()
            or ".." in source_relative_path.parts
        ):
            raise RecoveryError("invalid source staging relative path")
        source_staging = _resolved_within(
            repository_root, repository_root / source_relative_path
        )
        guard = cls(
            repository_root=repository_root,
            manifest_path=manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            manifest=dict(manifest),
            source_staging=source_staging,
        )
        guard.verify()
        return guard

    def dependency_paths(self) -> dict[str, Path]:
        dependencies = self.manifest.get("recovery_dependencies")
        if not isinstance(dependencies, Mapping) or set(dependencies) != {
            "live_producer_source",
            "independent_continuation_reviewer",
        }:
            raise RecoveryError("recovery dependency schema mismatch")
        paths: dict[str, Path] = {}
        for name, payload in dependencies.items():
            if not isinstance(payload, Mapping) or set(payload) != {
                "relative_path",
                "sha256",
            }:
                raise RecoveryError(f"recovery dependency entry malformed: {name}")
            relative = payload.get("relative_path")
            digest = payload.get("sha256")
            relative_path = Path(relative) if isinstance(relative, str) else None
            if (
                relative_path is None
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or not _is_sha256(digest)
            ):
                raise RecoveryError(f"recovery dependency entry invalid: {name}")
            paths[name] = _resolved_within(
                self.repository_root, self.repository_root / relative_path
            )
        return paths

    def verify(self) -> dict[str, Any]:
        if _sha256_file(self.manifest_path) != self.expected_manifest_sha256:
            raise RecoveryError("frozen recovery input manifest changed")
        _lstat_directory(self.source_staging)
        expected_names = self.manifest.get("expected_exact_file_set")
        expected_hashes = self.manifest.get("files_sha256")
        if (
            not isinstance(expected_names, list)
            or not isinstance(expected_hashes, Mapping)
            or len(expected_names) != len(set(expected_names))
            or set(expected_names) != set(expected_hashes)
        ):
            raise RecoveryError("source staging file-set manifest is malformed")
        actual_entries = list(self.source_staging.iterdir())
        actual_names = {path.name for path in actual_entries}
        if actual_names != set(expected_names):
            missing = sorted(set(expected_names) - actual_names)
            extra = sorted(actual_names - set(expected_names))
            raise RecoveryError(
                f"source staging exact file set mismatch: missing={missing} extra={extra}"
            )
        observed_hashes: dict[str, str] = {}
        for path in actual_entries:
            _lstat_regular_file(path)
            expected = expected_hashes[path.name]
            if not _is_sha256(expected):
                raise RecoveryError(f"invalid frozen hash for {path.name}")
            observed = _sha256_file(path)
            if observed != expected:
                raise RecoveryError(
                    f"source staging hash mismatch for {path.name}: {observed}"
                )
            observed_hashes[path.name] = observed

        dependency_paths = self.dependency_paths()
        dependency_hashes: dict[str, str] = {}
        dependencies = self.manifest["recovery_dependencies"]
        for name, path in dependency_paths.items():
            _lstat_regular_file(path)
            observed = _sha256_file(path)
            expected = dependencies[name]["sha256"]
            if observed != expected:
                raise RecoveryError(
                    f"recovery dependency hash mismatch for {name}: {observed}"
                )
            dependency_hashes[name] = observed

        producer_hash = dependency_hashes["live_producer_source"]
        reviewer_hash = dependency_hashes["independent_continuation_reviewer"]
        radius = self.manifest.get("recovery_radius_rule")
        if not isinstance(radius, Mapping):
            raise RecoveryError("recovery radius rule is malformed")
        if (
            producer_hash != EXPECTED_PRODUCER_SHA256
            or observed_hashes.get("executed_source_snapshot.py") != producer_hash
            or reviewer_hash != EXPECTED_REVIEWER_SHA256
            or radius.get("parent_reviewer_source_sha256") != reviewer_hash
            or radius.get("parent_reviewer_source_relative_path")
            != dependencies["independent_continuation_reviewer"]["relative_path"]
            or float(radius.get("absolute_tolerance", math.nan)) != RECOVERY_RADIUS_ATOL
            or float(radius.get("relative_tolerance", math.nan)) != 0.0
        ):
            raise RecoveryError("recovery dependency/radius lineage mismatch")
        return {
            "manifest_sha256": self.expected_manifest_sha256,
            "source_staging": str(self.source_staging),
            "source_file_count": len(observed_hashes),
            "source_files_sha256": dict(sorted(observed_hashes.items())),
            "dependency_files_sha256": dict(sorted(dependency_hashes.items())),
        }


def _validate_output_separation(source_staging: Path, output: Path) -> None:
    source = _lstat_directory(source_staging)
    output = _absolute_without_resolve(output)
    _lstat_no_symlink_components(output.parent)
    if source == output:
        raise RecoveryError("recovery output cannot equal source staging")
    if source in output.parents or output in source.parents:
        raise RecoveryError("recovery output and source staging must be disjoint")


def _require_default_output(output: Path) -> Path:
    declared = _absolute_without_resolve(output)
    expected = _absolute_without_resolve(DEFAULT_OUTPUT)
    if declared != expected:
        raise RecoveryError(
            f"production recovery output is frozen to {expected}; got {declared}"
        )
    _lstat_no_symlink_components(declared.parent)
    return declared


def _load_hash_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(_lstat_regular_file(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise RecoveryError(f"frozen dependency manifest is malformed: {path}")
    result: dict[str, str] = {}
    for relative, digest in payload.items():
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not _is_sha256(digest)
        ):
            raise RecoveryError(f"invalid frozen dependency entry: {relative}")
        result[str(relative_path)] = str(digest)
    return result


def _verify_recovery_task_manifest(
    *,
    repository_root: Path = ROOT,
    manifest_path: Path = RECOVERY_TASK_MANIFEST,
    expected_manifest_sha256: str = EXPECTED_RECOVERY_TASK_MANIFEST_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _lstat_directory(repository_root)
    path = _lstat_regular_file(_resolved_within(root, manifest_path))
    observed_manifest_sha256 = _sha256_file(path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise RecoveryError("recovery task manifest hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "protocol_id",
        "source_weight_index",
        "task_name",
        "tau",
        "accepted_vae_checkpoint_sha256",
        "expected_runtime",
        "source_dependencies",
        "selected_task_tensors",
    }:
        raise RecoveryError("recovery task manifest schema mismatch")
    identity = {
        "protocol_id": payload.get("protocol_id") == PROTOCOL_ID,
        "source_weight_index": payload.get("source_weight_index")
        == SOURCE_WEIGHT_INDEX,
        "task_name": payload.get("task_name") == EXPECTED_TASK_NAME,
        "tau": payload.get("tau") == EXPECTED_TASK_TAU,
        "accepted_vae_checkpoint_sha256": payload.get("accepted_vae_checkpoint_sha256")
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "expected_runtime": payload.get("expected_runtime") == EXPECTED_RUNTIME,
        "source_dependencies": payload.get("source_dependencies")
        == EXPECTED_TASK_SOURCE_DEPENDENCIES,
        "selected_task_tensors": payload.get("selected_task_tensors")
        == EXPECTED_TASK_TENSORS,
    }
    if not all(identity.values()):
        raise RecoveryError(
            "recovery task manifest frozen identity mismatch: "
            f"{[name for name, passed in identity.items() if not passed]}"
        )
    dependency_hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_TASK_SOURCE_DEPENDENCIES.items():
        dependency = _lstat_regular_file(_resolved_within(root, root / Path(relative)))
        observed = _sha256_file(dependency)
        if observed != expected:
            raise RecoveryError(f"recovery task source hash mismatch: {relative}")
        dependency_hashes[relative] = observed
    return dict(payload), {
        "pass": True,
        "manifest_sha256": observed_manifest_sha256,
        "identity_gates": identity,
        "source_dependency_count": len(dependency_hashes),
        "source_dependencies_sha256": dict(sorted(dependency_hashes.items())),
    }


def _verify_recovery_import_manifest(
    *,
    repository_root: Path = ROOT,
    manifest_path: Path = RECOVERY_IMPORT_MANIFEST,
    expected_manifest_sha256: str = EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _lstat_directory(repository_root)
    path = _lstat_regular_file(_resolved_within(root, manifest_path))
    observed_manifest_sha256 = _sha256_file(path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise RecoveryError("recovery import manifest hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "protocol_id",
        "entry_modules",
        "loaded_repository_modules",
    }:
        raise RecoveryError("recovery import manifest schema mismatch")
    expected_entries = [
        "scripts.run_one_state_exact_selected_trajectory_continuation",
        "scripts.review_one_state_exact_selected_trajectory_continuation",
    ]
    modules = payload.get("loaded_repository_modules")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("entry_modules") != expected_entries
        or not isinstance(modules, Mapping)
        or len(modules) != 49
        or set(expected_entries) - set(modules)
    ):
        raise RecoveryError("recovery import manifest frozen identity mismatch")
    verified: dict[str, dict[str, str]] = {}
    for module_name, entry in modules.items():
        if (
            not isinstance(module_name, str)
            or not isinstance(entry, Mapping)
            or set(entry) != {"relative_path", "sha256"}
        ):
            raise RecoveryError("recovery import manifest module entry malformed")
        relative = entry.get("relative_path")
        digest = entry.get("sha256")
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix != ".py"
            or not _is_sha256(digest)
        ):
            raise RecoveryError(
                f"recovery import manifest module entry invalid: {module_name}"
            )
        source = _lstat_regular_file(_resolved_within(root, root / relative_path))
        observed = _sha256_file(source)
        if observed != digest:
            raise RecoveryError(f"recovery import source hash mismatch: {module_name}")
        verified[module_name] = {
            "relative_path": str(relative_path),
            "sha256": observed,
        }
    for relative, digest in EXPECTED_TASK_SOURCE_DEPENDENCIES.items():
        matches = [
            entry for entry in verified.values() if entry["relative_path"] == relative
        ]
        if len(matches) != 1 or matches[0]["sha256"] != digest:
            raise RecoveryError(
                f"task source is not pinned by recovery import closure: {relative}"
            )
    return dict(payload), {
        "pass": True,
        "manifest_sha256": observed_manifest_sha256,
        "module_count": len(verified),
        "modules": dict(sorted(verified.items())),
    }


def _reject_preloaded_import_manifest_modules(
    import_manifest: Mapping[str, Any],
) -> None:
    modules = import_manifest.get("loaded_repository_modules")
    if not isinstance(modules, Mapping):
        raise RecoveryError("recovery import manifest modules are missing")
    conflicts = sorted(set(modules).intersection(sys.modules))
    if conflicts:
        raise RecoveryError(
            "frozen import-closure modules were preloaded before verification: "
            f"{conflicts}"
        )


def _audit_loaded_repository_module_closure(
    *,
    repository_root: Path,
    import_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = _lstat_directory(repository_root)
    expected = import_manifest.get("loaded_repository_modules")
    if not isinstance(expected, Mapping):
        raise RecoveryError("recovery import manifest modules are missing")
    observed: dict[str, dict[str, str]] = {}
    extras: dict[str, str] = {}
    live_source = _lstat_regular_file(LIVE_SOURCE)
    allowed_mp_main: ModuleType | None = None
    if "__mp_main__" in sys.modules:
        candidate = sys.modules["__mp_main__"]
        if not isinstance(candidate, ModuleType):
            raise RecoveryError(
                "invalid __mp_main__ repository closure alias: value is not a module"
            )
        if "__main__" not in sys.modules or candidate is not sys.modules["__main__"]:
            raise RecoveryError(
                "invalid __mp_main__ repository closure alias: "
                "__main__ identity mismatch"
            )
        if (
            candidate.__dict__
            is not _audit_loaded_repository_module_closure.__globals__
        ):
            raise RecoveryError(
                "invalid __mp_main__ repository closure alias: "
                "finalizer globals provenance mismatch"
            )
        main_file = getattr(sys.modules["__main__"], "__file__", None)
        if (
            not isinstance(main_file, str)
            or main_file.startswith("<")
            or not Path(main_file).is_absolute()
        ):
            raise RecoveryError(
                "invalid __mp_main__ repository closure alias: "
                "__main__.__file__ must be an absolute filesystem path"
            )
        try:
            main_source = _lstat_regular_file(_resolved_within(root, Path(main_file)))
        except RecoveryError as error:
            raise RecoveryError(
                "invalid __mp_main__ repository closure alias: "
                "__main__.__file__ does not resolve exactly to LIVE_SOURCE"
            ) from error
        if main_source != live_source:
            raise RecoveryError(
                "invalid __mp_main__ repository closure alias: "
                "__main__.__file__ does not resolve exactly to LIVE_SOURCE"
            )
        allowed_mp_main = candidate
    verified_mp_main_alias = False
    for module_name, module in list(sys.modules.items()):
        if module_name == "__mp_main__":
            if module is not allowed_mp_main:
                raise RecoveryError(
                    "invalid __mp_main__ repository closure alias: "
                    "module changed during audit"
                )
            verified_mp_main_alias = True
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or module_file.startswith("<"):
            continue
        module_path = Path(module_file)
        if not module_path.is_absolute():
            continue
        declared = _absolute_without_resolve(module_path)
        try:
            declared.relative_to(root)
        except ValueError:
            continue
        source = _lstat_regular_file(_resolved_within(root, declared))
        if source == live_source and module_name in {
            "__main__",
            "scripts.finalize_one_state_exact_selected_trajectory_recovery",
        }:
            continue
        digest = _sha256_file(source)
        entry = {"relative_path": str(source.relative_to(root)), "sha256": digest}
        if module_name in expected:
            observed[module_name] = entry
        else:
            extras[module_name] = entry["relative_path"]
    missing = sorted(set(expected) - set(observed))
    mismatched = sorted(
        name
        for name in set(expected) & set(observed)
        if observed[name] != expected[name]
    )
    if missing or mismatched or extras:
        raise RecoveryError(
            "loaded repository module closure mismatch: "
            f"missing={missing} mismatched={mismatched} extras={extras}"
        )
    return {
        "pass": True,
        "module_count": len(observed),
        "modules": dict(sorted(observed.items())),
        "allowed_additions": [
            "__main__",
            "scripts.finalize_one_state_exact_selected_trajectory_recovery",
            *(["__mp_main__"] if verified_mp_main_alias else []),
        ],
    }


def _verify_dependency_manifest_tree(
    *,
    repository_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = _lstat_directory(repository_root)
    initial = _lstat_regular_file(manifest_path)
    if _sha256_file(initial) != expected_manifest_sha256:
        raise RecoveryError("frozen producer dependency manifest hash mismatch")
    pending = [(initial, expected_manifest_sha256)]
    visited_manifests: dict[str, str] = {}
    verified_files: dict[str, str] = {}
    python_paths: set[Path] = set()
    while pending:
        current_manifest, expected_hash = pending.pop()
        manifest_key = str(current_manifest)
        if manifest_key in visited_manifests:
            if visited_manifests[manifest_key] != expected_hash:
                raise RecoveryError("conflicting nested dependency manifest hashes")
            continue
        if _sha256_file(current_manifest) != expected_hash:
            raise RecoveryError(
                f"nested dependency manifest hash mismatch: {current_manifest}"
            )
        visited_manifests[manifest_key] = expected_hash
        for relative, digest in _load_hash_manifest(current_manifest).items():
            dependency = _lstat_regular_file(
                _resolved_within(root, root / Path(relative))
            )
            observed = _sha256_file(dependency)
            if observed != digest:
                raise RecoveryError(
                    f"frozen transitive dependency mismatch: {relative}"
                )
            previous = verified_files.get(relative)
            if previous is not None and previous != digest:
                raise RecoveryError(f"conflicting frozen dependency hash: {relative}")
            verified_files[relative] = observed
            if dependency.suffix == ".py":
                python_paths.add(dependency)
            if (
                dependency.suffix == ".json"
                and "frozen_dependency_manifest" in dependency.name
            ):
                pending.append((dependency, digest))
    return {
        "root_manifest_sha256": expected_manifest_sha256,
        "manifest_count": len(visited_manifests),
        "verified_file_count": len(verified_files),
        "verified_files_sha256": dict(sorted(verified_files.items())),
        "python_paths": sorted(str(path) for path in python_paths),
    }


def _frozen_module_names(repository_root: Path, python_paths: set[Path]) -> set[str]:
    names: set[str] = set()
    for path in python_paths:
        relative = path.relative_to(repository_root)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            names.add(".".join(parts))
    return names


def _reject_preloaded_frozen_modules(
    *, repository_root: Path, python_paths: set[Path]
) -> None:
    frozen_paths = {_absolute_without_resolve(path) for path in python_paths}
    frozen_names = _frozen_module_names(repository_root, frozen_paths)
    conflicts: list[str] = []
    for name, module in list(sys.modules.items()):
        if name in frozen_names:
            conflicts.append(name)
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            loaded_path = _absolute_without_resolve(Path(module_file))
        except (TypeError, ValueError):
            continue
        if loaded_path in frozen_paths:
            conflicts.append(name)
    if conflicts:
        raise RecoveryError(
            "frozen repo modules were preloaded before dependency verification: "
            f"{sorted(set(conflicts))}"
        )


def _producer_dependency_preimport_audit(
    guard: FrozenInputGuard,
) -> dict[str, Any]:
    task_manifest, task_audit = _verify_recovery_task_manifest(
        repository_root=guard.repository_root
    )
    import_manifest, import_audit = _verify_recovery_import_manifest(
        repository_root=guard.repository_root
    )
    _reject_preloaded_import_manifest_modules(import_manifest)
    source_manifest = guard.source_staging / "frozen_dependency_manifest_snapshot.json"
    tree = _verify_dependency_manifest_tree(
        repository_root=guard.repository_root,
        manifest_path=source_manifest,
        expected_manifest_sha256=EXPECTED_PRODUCER_DEPENDENCY_MANIFEST_SHA256,
    )
    python_paths = {Path(path) for path in tree["python_paths"]}
    python_paths.update(guard.dependency_paths().values())
    python_paths.update(
        guard.repository_root / Path(relative)
        for relative in task_manifest["source_dependencies"]
    )
    python_paths.update(
        guard.repository_root / Path(entry["relative_path"])
        for entry in import_manifest["loaded_repository_modules"].values()
    )
    _reject_preloaded_frozen_modules(
        repository_root=guard.repository_root,
        python_paths=python_paths,
    )
    return {
        **tree,
        "preloaded_module_rejection_pass": True,
        "guarded_python_paths": sorted(str(path) for path in python_paths),
        "task_manifest": task_audit,
        "import_manifest": import_audit,
    }


def _import_verified_dependencies(
    guard: FrozenInputGuard,
) -> tuple[Any, Any, dict[str, Any]]:
    preimport = guard.verify()
    transitive_preimport = _producer_dependency_preimport_audit(guard)
    paths = guard.dependency_paths()
    producer = importlib.import_module(
        "scripts.run_one_state_exact_selected_trajectory_continuation"
    )
    reviewer = importlib.import_module(
        "scripts.review_one_state_exact_selected_trajectory_continuation"
    )
    loaded = {
        "live_producer_source": Path(producer.__file__).resolve(),
        "independent_continuation_reviewer": Path(reviewer.__file__).resolve(),
    }
    for name, expected_path in paths.items():
        if (
            loaded[name] != expected_path
            or _sha256_file(loaded[name])
            != guard.manifest["recovery_dependencies"][name]["sha256"]
        ):
            raise RecoveryError(f"loaded recovery dependency mismatch: {name}")
    transitive_postimport = _verify_dependency_manifest_tree(
        repository_root=guard.repository_root,
        manifest_path=guard.source_staging / "frozen_dependency_manifest_snapshot.json",
        expected_manifest_sha256=EXPECTED_PRODUCER_DEPENDENCY_MANIFEST_SHA256,
    )
    _, task_postimport = _verify_recovery_task_manifest(
        repository_root=guard.repository_root
    )
    import_manifest, import_postimport = _verify_recovery_import_manifest(
        repository_root=guard.repository_root
    )
    loaded_repository_closure = _audit_loaded_repository_module_closure(
        repository_root=guard.repository_root,
        import_manifest=import_manifest,
    )
    postimport = guard.verify()
    return (
        producer,
        reviewer,
        {
            "preimport": preimport,
            "transitive_preimport": transitive_preimport,
            "transitive_postimport": transitive_postimport,
            "task_postimport": task_postimport,
            "import_postimport": import_postimport,
            "loaded_repository_closure": loaded_repository_closure,
            "postimport": postimport,
            "loaded_paths": {name: str(path) for name, path in loaded.items()},
            "executed_snapshot_equals_loaded_producer": (
                _sha256_file(guard.source_staging / "executed_source_snapshot.py")
                == _sha256_file(loaded["live_producer_source"])
            ),
        },
    )


def _named_tensor_hash(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise RecoveryError(f"active state entry is not a tensor: {name}")
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise RecoveryError(f"active state tensor is nonfinite: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
        count += value.numel()
    return digest.hexdigest(), count


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise RecoveryError("tensor fingerprint input is not a tensor")
    value = tensor.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise RecoveryError("tensor fingerprint input is nonfinite")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": value.numel(),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def _snapshot_model_tensors(model: torch.nn.Module) -> dict[str, Any]:
    categories = {
        "parameters": dict(model.named_parameters()),
        "buffers": dict(model.named_buffers()),
    }
    digest = hashlib.sha256()
    result: dict[str, Any] = {}
    total_tensors = 0
    total_elements = 0
    for category, tensors in categories.items():
        fingerprints: dict[str, Any] = {}
        for name in sorted(tensors):
            fingerprint = _tensor_fingerprint(tensors[name])
            fingerprints[name] = fingerprint
            digest.update(category.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(json.dumps(fingerprint, sort_keys=True).encode("utf-8"))
            total_tensors += 1
            total_elements += int(fingerprint["numel"])
        result[category] = fingerprints
        result[f"{category[:-1]}_tensor_count"] = len(fingerprints)
        result[f"{category[:-1]}_element_count"] = sum(
            int(value["numel"]) for value in fingerprints.values()
        )
    result.update(
        {
            "all_tensor_count": total_tensors,
            "all_element_count": total_elements,
            "aggregate_sha256": digest.hexdigest(),
        }
    )
    return result


def _audit_model_installation(
    *,
    base: Mapping[str, Any],
    installed: Mapping[str, Any],
    expected_changed_parameters: set[str],
) -> dict[str, Any]:
    if len(expected_changed_parameters) != ACTIVE_TENSOR_COUNT:
        raise RecoveryError("expected active parameter name count mismatch")
    for category in ("parameters", "buffers"):
        if not isinstance(base.get(category), Mapping) or not isinstance(
            installed.get(category), Mapping
        ):
            raise RecoveryError("model tensor snapshot schema mismatch")
        if set(base[category]) != set(installed[category]):
            raise RecoveryError(f"model {category} names changed during installation")
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
    if set(changed_parameters) != expected_changed_parameters or changed_buffers:
        raise RecoveryError(
            "terminal installation changed an unexpected model tensor: "
            f"parameters={changed_parameters} buffers={changed_buffers}"
        )
    return {
        "pass": True,
        "expected_changed_parameter_count": ACTIVE_TENSOR_COUNT,
        "changed_parameter_count": len(changed_parameters),
        "changed_parameters": changed_parameters,
        "changed_buffer_count": 0,
        "changed_buffers": [],
        "base_aggregate_sha256": base["aggregate_sha256"],
        "installed_aggregate_sha256": installed["aggregate_sha256"],
        "base_parameter_tensor_count": base["parameter_tensor_count"],
        "base_buffer_tensor_count": base["buffer_tensor_count"],
    }


def _audit_model_replay_nonmutation(
    *, installed: Mapping[str, Any], replayed: Mapping[str, Any]
) -> dict[str, Any]:
    if dict(installed) != dict(replayed):
        changed: dict[str, list[str]] = {}
        for category in ("parameters", "buffers"):
            before = installed.get(category, {})
            after = replayed.get(category, {})
            names = set(before) | set(after)
            changed[category] = sorted(
                name for name in names if before.get(name) != after.get(name)
            )
        raise RecoveryError(f"geometry replay mutated model tensors: {changed}")
    return {
        "pass": True,
        "all_parameters_bitwise_unchanged": True,
        "all_buffers_bitwise_unchanged": True,
        "parameter_tensor_count": installed["parameter_tensor_count"],
        "buffer_tensor_count": installed["buffer_tensor_count"],
        "installed_aggregate_sha256": installed["aggregate_sha256"],
        "post_replay_aggregate_sha256": replayed["aggregate_sha256"],
    }


def _audit_runtime_task_reconstruction(
    *,
    task_manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    task_set: Any,
    accepted_checkpoint: Path,
) -> dict[str, Any]:
    checkpoint = _lstat_regular_file(accepted_checkpoint)
    checkpoint_sha256 = _sha256_file(checkpoint)
    tensors: dict[str, Any] = {}
    for name in EXPECTED_TASK_TENSORS:
        value = getattr(task_set, name, None)
        if not isinstance(value, torch.Tensor):
            raise RecoveryError(f"selected task tensor is missing: {name}")
        tensors[name] = _tensor_fingerprint(value)
        tensors[name].pop("numel")
    gates = {
        "manifest_protocol": task_manifest.get("protocol_id") == PROTOCOL_ID,
        "source_weight_index": record.get("source_weight_index")
        == SOURCE_WEIGHT_INDEX
        == task_manifest.get("source_weight_index"),
        "record_task_name": record.get("task_name")
        == EXPECTED_TASK_NAME
        == task_manifest.get("task_name"),
        "task_set_name": getattr(task_set, "task_name", None) == EXPECTED_TASK_NAME,
        "tau": record.get("tau") == EXPECTED_TASK_TAU == task_manifest.get("tau"),
        "full_train_sample_count": int(task_set.train_images.shape[0])
        == int(task_set.train_labels.shape[0])
        == EXPECTED_TASK_TRAIN_COUNT,
        "full_test_sample_count": int(task_set.test_images.shape[0])
        == int(task_set.test_labels.shape[0])
        == EXPECTED_TASK_TEST_COUNT,
        "selected_task_tensors": tensors
        == task_manifest.get("selected_task_tensors")
        == EXPECTED_TASK_TENSORS,
        "accepted_checkpoint": checkpoint_sha256
        == task_manifest.get("accepted_vae_checkpoint_sha256")
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
    }
    if not all(gates.values()):
        raise RecoveryError(
            "runtime task reconstruction mismatch: "
            f"{[name for name, passed in gates.items() if not passed]}"
        )
    return {
        "pass": True,
        "gates": gates,
        "task_name": EXPECTED_TASK_NAME,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "tau": EXPECTED_TASK_TAU,
        "train_sample_count": EXPECTED_TASK_TRAIN_COUNT,
        "test_sample_count": EXPECTED_TASK_TEST_COUNT,
        "selected_task_tensors": tensors,
        "accepted_vae_checkpoint_sha256": checkpoint_sha256,
    }


def _audit_runtime_provenance(
    *, task_manifest: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    torchvision = importlib.import_module("torchvision")
    observed_runtime = {
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
    }
    if (
        task_manifest.get("expected_runtime") != EXPECTED_RUNTIME
        or observed_runtime != EXPECTED_RUNTIME
    ):
        raise RecoveryError(
            "frozen runtime version mismatch: "
            f"observed={observed_runtime} expected={EXPECTED_RUNTIME}"
        )
    if (
        str(device) != PRODUCTION_DEVICE
        or device.type != "cuda"
        or device.index != 0
        or not torch.cuda.is_available()
    ):
        raise RecoveryError("frozen geometry replay requires CUDA provenance")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "pass": True,
        "expected_runtime": dict(EXPECTED_RUNTIME),
        "observed_runtime": observed_runtime,
        "requested_device": str(device),
        "resolved_cuda_device_index": index,
        "cuda_device_name": properties.name,
        "cuda_compute_capability": [properties.major, properties.minor],
        "cuda_total_memory_bytes": properties.total_memory,
        "cuda_current_device_index": torch.cuda.current_device(),
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _load_source_progress(
    guard: FrozenInputGuard, reviewer: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress_path = guard.source_staging / "progress_checkpoint.pt"
    if _sha256_file(progress_path) != EXPECTED_PROGRESS_SHA256:
        raise RecoveryError("source progress checkpoint hash mismatch")
    progress = torch.load(progress_path, map_location="cpu", weights_only=False)
    if not isinstance(progress, dict) or set(progress) != EXPECTED_PROGRESS_KEYS:
        raise RecoveryError("source progress checkpoint schema mismatch")
    expected = guard.manifest.get("expected_progress")
    if not isinstance(expected, Mapping):
        raise RecoveryError("expected progress manifest entry is malformed")
    observed_header = {
        "accepted_updates": int(progress.get("accepted_updates", -1)),
        "new_accepted_updates": int(progress.get("new_accepted_updates", -1)),
        "state_rows": len(progress.get("state_rows", [])),
        "spectrum_rows": len(progress.get("spectrum_rows", [])),
        "proposal_rows": len(progress.get("proposal_rows", [])),
        "line_rows": len(progress.get("line_rows", [])),
        "terminal": progress.get("terminal"),
        "termination": progress.get("termination"),
        "transition_chain_sha256": progress.get("transition_chain_sha256"),
        "active_parameter_hash": progress.get("active_parameter_hash"),
    }
    if observed_header != dict(expected):
        raise RecoveryError(
            f"source progress header mismatch: observed={observed_header}"
        )
    if (
        progress.get("protocol_id") != SOURCE_PROTOCOL_ID
        or progress.get("selected_arm") != "low"
        or observed_header["accepted_updates"] != MAX_ACCEPTED_UPDATES
        or observed_header["new_accepted_updates"]
        != MAX_ACCEPTED_UPDATES - START_UPDATE
        or observed_header["transition_chain_sha256"]
        != EXPECTED_TRANSITION_CHAIN_SHA256
    ):
        raise RecoveryError("source progress frozen identity mismatch")
    active_state = progress.get("active_model_state")
    if not isinstance(active_state, Mapping):
        raise RecoveryError("source progress active state is missing")
    active_hash, active_count = _named_tensor_hash(active_state)
    if (
        len(active_state) != ACTIVE_TENSOR_COUNT
        or active_count != ACTIVE_PARAMETER_COUNT
        or active_hash != EXPECTED_ACTIVE_PARAMETER_HASH
        or active_hash != progress["active_parameter_hash"]
        or progress["state_rows"][-1]["parameter_hash"] != active_hash
    ):
        raise RecoveryError("source progress active tensor lineage mismatch")

    table_names = {
        "state_rows": "state_metrics.csv",
        "spectrum_rows": "state_spectra.csv",
        "proposal_rows": "proposal_diagnostics.csv",
        "line_rows": "line_search.csv",
        "selection_rows": "arm_selection.csv",
    }
    for key, filename in table_names.items():
        frame = pd.read_csv(guard.source_staging / filename)
        if not reviewer._checkpoint_rows_match_csv(progress[key], frame):
            raise RecoveryError(f"source progress/{filename} mismatch")
    for key, filename in (
        ("intervention_origin", "intervention_origin.json"),
        ("intervention_preflight", "intervention_preflight.json"),
    ):
        payload = json.loads(
            (guard.source_staging / filename).read_text(encoding="utf-8")
        )
        if not reviewer._equivalent(payload, progress[key], atol=0.0, rtol=0.0):
            raise RecoveryError(f"source progress/{filename} mismatch")
    return progress, {
        "progress_checkpoint_sha256": _sha256_file(progress_path),
        "progress_header": observed_header,
        "active_tensor_count": len(active_state),
        "active_parameter_count": active_count,
        "active_parameter_hash": active_hash,
        "checkpoint_csv_lineage_pass": True,
    }


def _parse_original_failed_gates(error: BaseException) -> tuple[str, ...]:
    prefix = "continuation progress audit failed: "
    message = str(error)
    if not message.startswith(prefix):
        raise RecoveryError(f"unexpected original audit error: {message}") from error
    try:
        payload = ast.literal_eval(message.removeprefix(prefix))
    except (SyntaxError, ValueError) as parse_error:
        raise RecoveryError(
            "could not parse original audit failed gates"
        ) from parse_error
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise RecoveryError("original audit failed-gate payload is malformed")
    return tuple(payload)


def _require_original_audit_failure(
    audit: Callable[[], Any], *, expected_failed_gates: Sequence[str]
) -> dict[str, Any]:
    try:
        audit()
    except RuntimeError as error:
        failed = _parse_original_failed_gates(error)
        if failed != tuple(expected_failed_gates):
            raise RecoveryError(
                f"original audit failed set mismatch: {list(failed)}"
            ) from error
        return {
            "pass": False,
            "failed_gates": list(failed),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "audit_invocations": 1,
        }
    raise RecoveryError("original runner audit unexpectedly passed")


def _run_original_runner_audit(
    *, producer: Any, progress: Mapping[str, Any], parent_progress: Mapping[str, Any]
) -> dict[str, Any]:
    expected = tuple(guard_value for guard_value in EXPECTED_ORIGINAL_FAILED_GATES)
    result = _require_original_audit_failure(
        lambda: producer._audit_progress_checkpoint(progress, parent_progress),
        expected_failed_gates=expected,
    )
    rows = {
        key: list(progress[key])
        for key in (
            "state_rows",
            "spectrum_rows",
            "proposal_rows",
            "line_rows",
            "selection_rows",
        )
    }
    rows["intervention_origin"] = dict(progress["intervention_origin"])
    row_gates, recomputed_chain = producer._audit_rows(
        rows=rows,
        accepted_updates=int(progress["accepted_updates"]),
        terminal=bool(progress["terminal"]),
        termination=str(progress["termination"]),
        tolerances=progress["tolerances"],
        parent_progress=parent_progress,
        stored_chain_hash=str(progress["transition_chain_sha256"]),
    )
    failed_row_gates = [name for name, passed in row_gates.items() if not passed]
    if (
        failed_row_gates != list(expected)
        or recomputed_chain != progress["transition_chain_sha256"]
    ):
        raise RecoveryError("original row-gate replay disagrees with original audit")
    result.update(
        {
            "row_gates": row_gates,
            "all_unmodified_row_gates_pass": all(
                passed or name in expected for name, passed in row_gates.items()
            ),
            "transition_chain_sha256": recomputed_chain,
        }
    )
    return result


def _radius_audit(
    proposal_rows: Sequence[Mapping[str, Any]],
    *,
    target_norm: float,
    absolute_tolerance: float,
    expected_max_absolute_error: float | None = None,
) -> dict[str, Any]:
    continuation = [
        row for row in proposal_rows if int(row.get("target_update", -1)) > START_UPDATE
    ]
    if not continuation:
        raise RecoveryError("no continuation rows for direction-radius audit")
    errors: list[tuple[int, float]] = []
    for row in continuation:
        radius = float(row["direction_norm"])
        if not math.isfinite(radius):
            raise RecoveryError("direction radius is nonfinite")
        errors.append((int(row["target_update"]), abs(radius - target_norm)))
    max_target, maximum = max(errors, key=lambda item: item[1])
    if maximum > absolute_tolerance:
        raise RecoveryError(
            f"recovery direction radius exceeds {absolute_tolerance}: {maximum}"
        )
    if expected_max_absolute_error is not None and not math.isclose(
        maximum,
        expected_max_absolute_error,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RecoveryError(
            "recovery direction maximum error differs from frozen expectation"
        )
    return {
        "pass": True,
        "relative_tolerance": 0.0,
        "absolute_tolerance": absolute_tolerance,
        "row_count": len(errors),
        "max_absolute_error": maximum,
        "max_error_target_update": max_target,
        "count_exceeding_original_1e_9": sum(
            error > ORIGINAL_RADIUS_ATOL for _, error in errors
        ),
        "count_exceeding_recovery_5e_9": sum(
            error > absolute_tolerance for _, error in errors
        ),
    }


def _independent_recovery_audit(
    *,
    reviewer: Any,
    progress: Mapping[str, Any],
    parent_progress: Mapping[str, Any],
    guard: FrozenInputGuard,
) -> dict[str, Any]:
    radius_rule = guard.manifest["recovery_radius_rule"]
    if (
        reviewer.DIRECTION_RADIUS_ATOL != float(radius_rule["absolute_tolerance"])
        or reviewer.DIRECTION_RADIUS_ATOL != RECOVERY_RADIUS_ATOL
    ):
        raise RecoveryError("frozen reviewer radius constant mismatch")
    states = pd.DataFrame(progress["state_rows"])
    spectra = pd.DataFrame(progress["spectrum_rows"])
    spectrum_result = reviewer._validate_spectrum_tables(states, spectra)
    preflight_result = reviewer._validate_preflight(progress, parent_progress)
    history = reviewer._validate_history(progress, parent_progress)
    seeded = reviewer._seeded_parent(parent_progress)
    prefix_gates = {
        "parent_state_prefix_exact": progress["state_rows"][: START_UPDATE + 1]
        == seeded["state_rows"],
        "parent_spectrum_prefix_exact": progress["spectrum_rows"][
            : (START_UPDATE + 1) * DIMENSION
        ]
        == seeded["spectrum_rows"],
        "parent_proposal_prefix_exact": progress["proposal_rows"][:START_UPDATE]
        == seeded["proposal_rows"],
        "parent_line_prefix_exact": progress["line_rows"][: len(seeded["line_rows"])]
        == seeded["line_rows"],
        "parent_selection_exact": progress["selection_rows"]
        == seeded["selection_rows"],
        "intervention_origin_exact": progress["intervention_origin"]
        == seeded["intervention_origin"],
        "initial_metrics_exact": progress["initial_metrics"]
        == parent_progress["initial_metrics"],
        "tolerances_exact": progress["tolerances"] == parent_progress["tolerances"],
    }
    if not all(prefix_gates.values()):
        raise RecoveryError(
            "independent recovery parent-prefix audit failed: "
            f"{[name for name, value in prefix_gates.items() if not value]}"
        )
    radius = _radius_audit(
        progress["proposal_rows"],
        target_norm=float(reviewer.TARGET_NORM),
        absolute_tolerance=float(radius_rule["absolute_tolerance"]),
        expected_max_absolute_error=float(radius_rule["expected_max_absolute_error"]),
    )
    gates = {
        "frozen_reviewer_hash_verified": _sha256_file(Path(reviewer.__file__))
        == EXPECTED_REVIEWER_SHA256,
        "reviewer_radius_rule_exact": reviewer.DIRECTION_RADIUS_ATOL
        == RECOVERY_RADIUS_ATOL,
        "parent_prefixes_exact": all(prefix_gates.values()),
        "preflight_recomputed": bool(preflight_result),
        "spectrum_recomputed": bool(spectrum_result),
        "history_recomputed": bool(history),
        "all_historical_transitions_pass": bool(
            history["summary"]["all_historical_transitions_pass"]
        ),
        "all_historical_a_armijo_pass": bool(
            history["summary"]["all_historical_a_armijo_pass"]
        ),
        "transition_chain_recomputed": history["transition_chain_sha256"]
        == progress["transition_chain_sha256"],
        "recovery_radius_audit_pass": radius["pass"],
    }
    if not all(gates.values()):
        raise RecoveryError(
            "independent recovery audit failed: "
            f"{[name for name, value in gates.items() if not value]}"
        )
    return {
        "pass": True,
        "gates": gates,
        "prefix_gates": prefix_gates,
        "history_summary": history["summary"],
        "historical_transition_rows": history["audit_rows"],
        "transition_chain_sha256": history["transition_chain_sha256"],
        "radius_audit": radius,
        "preflight": preflight_result,
        "spectrum": spectrum_result,
    }


def _build_final_checkpoint(
    *,
    progress: Mapping[str, Any],
    source_progress_sha256: str,
) -> dict[str, Any]:
    active_state = progress.get("active_model_state")
    if not isinstance(active_state, Mapping):
        raise RecoveryError("terminal progress active state is missing")
    return {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": progress["protocol_id"],
        "source_progress_checkpoint_sha256": source_progress_sha256,
        "parent_checkpoint_sha256": progress["parent_checkpoint_sha256"],
        "parent_progress_checkpoint_sha256": progress[
            "parent_progress_checkpoint_sha256"
        ],
        "selected_arm": progress["selected_arm"],
        "accepted_updates": int(progress["accepted_updates"]),
        "new_accepted_updates": int(progress["new_accepted_updates"]),
        "termination": progress["termination"],
        "transition_chain_sha256": progress["transition_chain_sha256"],
        "active_parameter_hash": progress["active_parameter_hash"],
        "active_model_state": {
            name: tensor.detach().cpu().clone() for name, tensor in active_state.items()
        },
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "z_sha256": EXPECTED_Z_SHA256,
        "accepted_vae_checkpoint_sha256": EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "stored_state_metrics": dict(progress["state_rows"][-1]),
    }


def _audit_checkpoint_lineage(
    checkpoint: Mapping[str, Any],
    progress: Mapping[str, Any],
    *,
    expected_tensor_count: int,
    expected_parameter_count: int,
    expected_active_hash: str,
) -> dict[str, Any]:
    if set(checkpoint) != EXPECTED_FINAL_CHECKPOINT_KEYS:
        raise RecoveryError("checkpoint exact schema mismatch")
    stored = checkpoint.get("active_model_state")
    source = progress.get("active_model_state")
    if not isinstance(stored, Mapping) or not isinstance(source, Mapping):
        raise RecoveryError("checkpoint active tensor mapping is missing")
    if set(stored) != set(source) or len(stored) != expected_tensor_count:
        raise RecoveryError("checkpoint active tensor names/count mismatch")
    for name in sorted(source):
        source_tensor = source[name]
        stored_tensor = stored[name]
        if (
            source_tensor.device.type != "cpu"
            or stored_tensor.device.type != "cpu"
            or source_tensor.shape != stored_tensor.shape
            or source_tensor.dtype != stored_tensor.dtype
            or not torch.equal(source_tensor, stored_tensor)
        ):
            raise RecoveryError(
                f"checkpoint tensor is not bitwise source-derived: {name}"
            )
    active_hash, parameter_count = _named_tensor_hash(stored)
    metadata_gates = {
        "protocol_id": checkpoint.get("protocol_id") == PROTOCOL_ID,
        "source_protocol_id": checkpoint.get("source_protocol_id")
        == progress.get("protocol_id")
        == SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": checkpoint.get(
            "source_progress_checkpoint_sha256"
        )
        == EXPECTED_PROGRESS_SHA256,
        "parent_checkpoint_sha256": checkpoint.get("parent_checkpoint_sha256")
        == progress.get("parent_checkpoint_sha256")
        == EXPECTED_PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": checkpoint.get(
            "parent_progress_checkpoint_sha256"
        )
        == progress.get("parent_progress_checkpoint_sha256")
        == EXPECTED_PARENT_PROGRESS_SHA256,
        "selected_arm": checkpoint.get("selected_arm")
        == progress.get("selected_arm")
        == "low",
        "accepted_updates": checkpoint.get("accepted_updates")
        == progress.get("accepted_updates")
        == MAX_ACCEPTED_UPDATES,
        "new_accepted_updates": checkpoint.get("new_accepted_updates")
        == progress.get("new_accepted_updates")
        == MAX_ACCEPTED_UPDATES - START_UPDATE,
        "termination": checkpoint.get("termination")
        == progress.get("termination")
        == "max_updates_reached",
        "transition_chain": checkpoint.get("transition_chain_sha256")
        == progress.get("transition_chain_sha256")
        == EXPECTED_TRANSITION_CHAIN_SHA256,
        "active_hash": active_hash
        == checkpoint.get("active_parameter_hash")
        == progress.get("active_parameter_hash")
        == expected_active_hash,
        "stored_metrics": checkpoint.get("stored_state_metrics")
        == progress["state_rows"][-1],
        "source_weight_index": checkpoint.get("source_weight_index")
        == SOURCE_WEIGHT_INDEX,
        "z_sha256": checkpoint.get("z_sha256") == EXPECTED_Z_SHA256,
        "accepted_vae_checkpoint_sha256": checkpoint.get(
            "accepted_vae_checkpoint_sha256"
        )
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
    }
    if parameter_count != expected_parameter_count or not all(metadata_gates.values()):
        raise RecoveryError("checkpoint metadata/hash lineage mismatch")
    return {
        "pass": True,
        "bitwise_tensor_equality": True,
        "active_tensor_count": len(stored),
        "active_parameter_count": parameter_count,
        "active_parameter_hash": active_hash,
        "metadata_gates": metadata_gates,
    }


def _write_and_reload_final_checkpoint(
    *,
    path: Path,
    progress: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = _build_final_checkpoint(
        progress=progress,
        source_progress_sha256=EXPECTED_PROGRESS_SHA256,
    )
    _atomic_torch_save(path, checkpoint)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(reloaded, Mapping):
        raise RecoveryError("reloaded final checkpoint is malformed")
    audit = _audit_checkpoint_lineage(
        reloaded,
        progress,
        expected_tensor_count=ACTIVE_TENSOR_COUNT,
        expected_parameter_count=ACTIVE_PARAMETER_COUNT,
        expected_active_hash=EXPECTED_ACTIVE_PARAMETER_HASH,
    )
    audit["final_checkpoint_sha256"] = _sha256_file(path)
    audit["cpu_reload_pass"] = True
    return dict(reloaded), audit


@dataclass
class GeometryReplayGuard:
    evaluation_count: int = 0
    forbidden_operation_attempts: int = 0

    def begin_evaluation(self) -> None:
        if self.evaluation_count != 0:
            raise RecoveryError("geometry evaluation budget exceeded")
        self.evaluation_count += 1

    def evaluate(
        self, callback: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        self.begin_evaluation()
        return callback(*args, **kwargs)

    def forbid(self, operation: str) -> None:
        self.forbidden_operation_attempts += 1
        raise RecoveryError(f"forbidden recovery optimization operation: {operation}")

    def require_complete(self) -> dict[str, Any]:
        if self.evaluation_count != 1 or self.forbidden_operation_attempts != 0:
            raise RecoveryError("geometry replay execution invariant failed")
        return {
            "geometry_evaluation_count": self.evaluation_count,
            "forbidden_operation_attempts": self.forbidden_operation_attempts,
            "optimization_gradient_evaluations": 0,
            "proposals": 0,
            "line_searches": 0,
            "parameter_updates": 0,
        }


def _audit_reconstruction_input_files(producer: Any) -> dict[str, Any]:
    expected_files = {
        "config.json": EXPECTED_ACCEPTED_RUN_CONFIG_SHA256,
        "weight_pool.pt": EXPECTED_ACCEPTED_RUN_WEIGHT_POOL_SHA256,
        "weight_pool_records.csv": EXPECTED_ACCEPTED_RUN_RECORDS_SHA256,
        "vae_checkpoint.pt": EXPECTED_ACCEPTED_RUN_CHECKPOINT_SHA256,
    }
    if not all(_is_sha256(digest) for digest in expected_files.values()):
        raise RecoveryError("reconstruction input hash constants are malformed")
    declared_run_dir = getattr(producer, "DEFAULT_RUN_DIR", None)
    if not isinstance(declared_run_dir, Path):
        raise RecoveryError("producer reconstruction input path schema mismatch")
    run_dir = _absolute_without_resolve(declared_run_dir)
    expected_run_dir = _absolute_without_resolve(EXPECTED_ACCEPTED_RUN_DIR)
    if run_dir != expected_run_dir:
        raise RecoveryError(
            "producer reconstruction input run path mismatch: "
            f"{run_dir} != {expected_run_dir}"
        )
    run_dir = _lstat_directory(run_dir)
    producer_checkpoint_hash = getattr(producer, "EXPECTED_CHECKPOINT", None)
    if producer_checkpoint_hash != EXPECTED_ACCEPTED_RUN_CHECKPOINT_SHA256:
        raise RecoveryError("producer reconstruction checkpoint constant mismatch")
    files: dict[str, dict[str, str]] = {}
    for filename, expected_hash in expected_files.items():
        expected_path = expected_run_dir / filename
        source = _lstat_regular_file(_resolved_within(ROOT, expected_path))
        if source != expected_path:
            raise RecoveryError(f"reconstruction input path mismatch: {filename}")
        observed_hash = _sha256_file(source)
        if observed_hash != expected_hash:
            raise RecoveryError(f"reconstruction input hash mismatch: {filename}")
        files[filename] = {
            "path": str(source),
            "sha256": observed_hash,
        }
    return {
        "pass": True,
        "run_dir": str(run_dir),
        "file_count": len(files),
        "files": dict(sorted(files.items())),
    }


def _require_reconstruction_input_pre_post_consistency(
    pre_load_run: Mapping[str, Any],
    post_load_run: Mapping[str, Any],
) -> dict[str, Any]:
    expected_names = {
        "config.json",
        "weight_pool.pt",
        "weight_pool_records.csv",
        "vae_checkpoint.pt",
    }
    expected_hashes = {
        "config.json": EXPECTED_ACCEPTED_RUN_CONFIG_SHA256,
        "weight_pool.pt": EXPECTED_ACCEPTED_RUN_WEIGHT_POOL_SHA256,
        "weight_pool_records.csv": EXPECTED_ACCEPTED_RUN_RECORDS_SHA256,
        "vae_checkpoint.pt": EXPECTED_ACCEPTED_RUN_CHECKPOINT_SHA256,
    }
    for label, audit in (
        ("pre_load_run", pre_load_run),
        ("post_load_run", post_load_run),
    ):
        if set(audit) != {"pass", "run_dir", "file_count", "files"}:
            raise RecoveryError(f"reconstruction input {label} audit schema mismatch")
        files = audit.get("files")
        if (
            audit.get("pass") is not True
            or audit.get("run_dir")
            != str(_absolute_without_resolve(EXPECTED_ACCEPTED_RUN_DIR))
            or audit.get("file_count") != len(expected_names)
            or not isinstance(files, Mapping)
            or set(files) != expected_names
        ):
            raise RecoveryError(f"reconstruction input {label} audit identity mismatch")
        for name, entry in files.items():
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"path", "sha256"}
                or entry.get("path")
                != str(_absolute_without_resolve(EXPECTED_ACCEPTED_RUN_DIR / name))
                or entry.get("sha256") != expected_hashes[name]
            ):
                raise RecoveryError(
                    f"reconstruction input {label} file schema/path/hash mismatch: {name}"
                )
    if dict(pre_load_run) != dict(post_load_run):
        raise RecoveryError("reconstruction inputs changed across _load_run")
    return {
        "pass": True,
        "pre_post_equal": True,
        "pre_load_run": dict(pre_load_run),
        "post_load_run": dict(post_load_run),
    }


def _fresh_final_geometry_replay(
    *,
    producer: Any,
    progress: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    device: torch.device,
    replay_guard: GeometryReplayGuard,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RecoveryError("final geometry replay requires an available CUDA device")
    runtime_provenance = _audit_runtime_provenance(
        task_manifest=task_manifest,
        device=device,
    )
    reconstruction_inputs_pre = _audit_reconstruction_input_files(producer)
    run = producer._load_run(producer.DEFAULT_RUN_DIR, device=device)
    reconstruction_inputs_post = _audit_reconstruction_input_files(producer)
    reconstruction_input_files = _require_reconstruction_input_pre_post_consistency(
        reconstruction_inputs_pre,
        reconstruction_inputs_post,
    )
    accepted_checkpoint = Path(
        reconstruction_inputs_post["files"]["vae_checkpoint.pt"]["path"]
    )
    accepted_checkpoint_sha256 = reconstruction_inputs_post["files"][
        "vae_checkpoint.pt"
    ]["sha256"]
    if not (
        producer.EXPECTED_CHECKPOINT
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        == task_manifest.get("accepted_vae_checkpoint_sha256")
        == accepted_checkpoint_sha256
    ):
        raise RecoveryError("accepted h2048 checkpoint hash mismatch")
    cfg = producer._probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
    cfg = replace(cfg, vae_precond_hvp_mode="autograd")
    bank = pd.read_csv(producer.STATE_BANK)
    selected_state = bank.loc[bank["state_position"].eq(producer.STATE_POSITION)]
    if (
        len(selected_state) != 1
        or int(selected_state.iloc[0]["source_weight_index"]) != SOURCE_WEIGHT_INDEX
    ):
        raise RecoveryError("state-bank identity mismatch")
    record = run.records.iloc[SOURCE_WEIGHT_INDEX].to_dict()
    record["source_weight_index"] = SOURCE_WEIGHT_INDEX
    weight = run.weights[[SOURCE_WEIGHT_INDEX]].to(
        device=device, dtype=producer.i6.torch_dtype(run.cfg)
    )
    with torch.no_grad():
        z = producer.encode_weights(run.vae, run.normalizer, weight).detach()[0]
    if producer.sha256_tensor(z) != producer.EXPECTED_Z_SHA256:
        raise RecoveryError("reconstructed z fingerprint mismatch")
    task_set = producer._task_set_for_record(run.task_tensors, record)
    task_at_load = _audit_runtime_task_reconstruction(
        task_manifest=task_manifest,
        record=record,
        task_set=task_set,
        accepted_checkpoint=accepted_checkpoint,
    )
    if not all(
        producer._batch_indices(
            task_set,
            batch_size=int(cfg.vae_precond_batch_size),
            step=10,
            sample_key=SOURCE_WEIGHT_INDEX,
            pair_key=pair_key,
        )
        is None
        for pair_key in range(8)
    ):
        raise RecoveryError("full CE batch gate failed")

    active_names = sorted(
        set(pd.read_csv(producer.ACTIVE_PARAMETERS)["parameter"].astype(str))
    )
    active_state = progress["active_model_state"]
    if len(active_names) != ACTIVE_TENSOR_COUNT or set(active_names) != set(
        active_state
    ):
        raise RecoveryError("reconstructed active tensor names mismatch")
    named = dict(run.vae.named_parameters())
    if set(active_names) - set(named):
        raise RecoveryError("active parameter names are missing from reconstructed VAE")
    base_model_snapshot = _snapshot_model_tensors(run.vae)
    base_gradient_slots = sum(
        parameter.grad is not None for parameter in run.vae.parameters()
    )
    if base_gradient_slots != 0:
        raise RecoveryError("fresh reconstructed VAE contains parameter gradients")
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    with torch.no_grad():
        for name in active_names:
            source_tensor = active_state[name]
            parameter = named[name]
            if (
                source_tensor.shape != parameter.shape
                or source_tensor.dtype != parameter.dtype
            ):
                raise RecoveryError(
                    f"reconstructed active tensor metadata mismatch: {name}"
                )
            parameter.copy_(source_tensor.to(device=device))
    installed_model_snapshot = _snapshot_model_tensors(run.vae)
    installation_audit = _audit_model_installation(
        base=base_model_snapshot,
        installed=installed_model_snapshot,
        expected_changed_parameters=set(active_names),
    )
    active_mapping = {name: named[name] for name in active_names}
    installed_hash, installed_count = _named_tensor_hash(active_mapping)
    if (
        installed_count != ACTIVE_PARAMETER_COUNT
        or installed_hash != EXPECTED_ACTIVE_PARAMETER_HASH
    ):
        raise RecoveryError("installed terminal active tensor hash mismatch")

    task_immediately_pre_replay = _audit_runtime_task_reconstruction(
        task_manifest=task_manifest,
        record=record,
        task_set=task_set,
        accepted_checkpoint=accepted_checkpoint,
    )
    if task_immediately_pre_replay != task_at_load:
        raise RecoveryError("task reconstruction changed before geometry replay")

    def evaluate_geometry() -> dict[str, Any]:
        state = producer.i6._evaluate(
            run=run,
            z=z,
            record=record,
            frozen_low_basis=None,
        )
        basis = state.get("current_low_basis")
        if not isinstance(basis, torch.Tensor):
            raise RecoveryError("geometry replay current low basis is missing")
        state["current_low_projector"] = (
            (basis @ basis.transpose(0, 1)).detach().double().clone()
        )
        return state

    final_state = replay_guard.evaluate(evaluate_geometry)
    post_replay_model_snapshot = _snapshot_model_tensors(run.vae)
    nonmutation_audit = _audit_model_replay_nonmutation(
        installed=installed_model_snapshot,
        replayed=post_replay_model_snapshot,
    )
    post_gradient_slots = sum(
        parameter.grad is not None for parameter in run.vae.parameters()
    )
    if post_gradient_slots != 0:
        raise RecoveryError("geometry replay populated parameter gradient slots")
    post_hash, post_count = _named_tensor_hash(active_mapping)
    if post_hash != installed_hash or post_count != installed_count:
        raise RecoveryError("geometry replay mutated terminal active tensors")
    task_post_replay = _audit_runtime_task_reconstruction(
        task_manifest=task_manifest,
        record=record,
        task_set=task_set,
        accepted_checkpoint=accepted_checkpoint,
    )
    if task_post_replay != task_immediately_pre_replay:
        raise RecoveryError("geometry replay changed task reconstruction fingerprints")
    return final_state, {
        "accepted_checkpoint_sha256": accepted_checkpoint_sha256,
        "expected_accepted_checkpoint_sha256": (
            EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        ),
        "z_sha256": producer.sha256_tensor(z),
        "active_tensor_count": len(active_names),
        "active_parameter_count": installed_count,
        "installed_active_parameter_hash": installed_hash,
        "post_replay_active_parameter_hash": post_hash,
        "base_parameter_gradient_slots": base_gradient_slots,
        "post_replay_parameter_gradient_slots": post_gradient_slots,
        "model_installation_audit": installation_audit,
        "model_replay_nonmutation_audit": nonmutation_audit,
        "model_tensor_snapshots": {
            "base": base_model_snapshot,
            "installed": installed_model_snapshot,
            "post_replay": post_replay_model_snapshot,
        },
        "runtime_provenance": runtime_provenance,
        "reconstruction_input_files": reconstruction_input_files,
        "task_at_load": task_at_load,
        "task_immediately_pre_replay": task_immediately_pre_replay,
        "task_post_replay": task_post_replay,
        "full_ce_batch": True,
    }


def _compare_final_geometry_replay(
    *,
    final_state: Mapping[str, Any],
    progress: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    final_row = progress["state_rows"][-1]
    observed_metrics = final_state.get("metrics")
    if not isinstance(observed_metrics, Mapping):
        raise RecoveryError("final geometry replay metrics are missing")
    expected_metric_keys = set(final_row) - {
        "accepted_update",
        "parameter_hash",
        "parent_frozen_low_energy",
        "phase",
    }
    if set(observed_metrics) != expected_metric_keys:
        raise RecoveryError(
            "final geometry metric schema mismatch: "
            f"missing={sorted(expected_metric_keys - set(observed_metrics))} "
            f"extra={sorted(set(observed_metrics) - expected_metric_keys)}"
        )
    metric_errors: dict[str, float] = {}
    for name in sorted(expected_metric_keys - {"hessian_sec", "low_basis_hash"}):
        observed = float(observed_metrics[name])
        expected = float(final_row[name])
        if not math.isfinite(observed) or not math.isfinite(expected):
            raise RecoveryError(f"nonfinite final replay metric: {name}")
        metric_errors[name] = abs(observed - expected)
    maximum_metric_error = max(metric_errors.values())
    if maximum_metric_error > FINAL_REPLAY_ATOL:
        raise RecoveryError(
            f"final geometry metric replay exceeds tolerance: {maximum_metric_error}"
        )
    if (
        str(observed_metrics["low_basis_hash"]) != str(final_row["low_basis_hash"])
        or not math.isfinite(float(observed_metrics["hessian_sec"]))
        or float(observed_metrics["hessian_sec"]) < 0.0
    ):
        raise RecoveryError("final low-basis hash/timing replay mismatch")

    eig = final_state.get("eig")
    hessian = final_state.get("hessian")
    matrix = final_state.get("matrix")
    low_basis = final_state.get("current_low_basis")
    low_projector = final_state.get("current_low_projector")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (eig, hessian, matrix, low_basis, low_projector)
    ):
        raise RecoveryError("final H/M/spectrum replay payload is incomplete")
    if (
        tuple(eig.shape) != (DIMENSION,)
        or tuple(hessian.shape) != (DIMENSION, DIMENSION)
        or tuple(matrix.shape) != (DIMENSION, DIMENSION)
        or low_basis.ndim != 2
        or tuple(low_basis.shape)[0] != DIMENSION
        or tuple(low_projector.shape) != (DIMENSION, DIMENSION)
        or not bool(torch.isfinite(eig).all())
        or not bool(torch.isfinite(hessian).all())
        or not bool(torch.isfinite(matrix).all())
        or not bool(torch.isfinite(low_basis).all())
        or not bool(torch.isfinite(low_projector).all())
    ):
        raise RecoveryError("final H/M/spectrum replay payload is invalid")
    projector_hash = _sha256_tensor(low_projector)
    if projector_hash != str(observed_metrics["low_basis_hash"]):
        raise RecoveryError("replayed low-basis projector hash mismatch")
    final_update = int(progress["accepted_updates"])
    stored_rows = sorted(
        (
            row
            for row in progress["spectrum_rows"]
            if int(row["accepted_update"]) == final_update
        ),
        key=lambda row: int(row["rank"]),
    )
    if len(stored_rows) != DIMENSION or [
        int(row["rank"]) for row in stored_rows
    ] != list(range(DIMENSION)):
        raise RecoveryError("stored final spectrum is incomplete")
    stored_eig = np.asarray(
        [float(row["m_eigenvalue"]) for row in stored_rows], dtype=np.float64
    )
    observed_eig = eig.detach().cpu().double().numpy()
    eigenvalue_errors = np.abs(observed_eig - stored_eig)
    maximum_spectrum_error = float(eigenvalue_errors.max())
    if maximum_spectrum_error > FINAL_REPLAY_ATOL:
        raise RecoveryError(
            f"final spectrum replay exceeds tolerance: {maximum_spectrum_error}"
        )
    expected_hash = str(progress["active_parameter_hash"])
    if not (
        execution["installed_active_parameter_hash"]
        == execution["post_replay_active_parameter_hash"]
        == checkpoint["active_parameter_hash"]
        == final_row["parameter_hash"]
        == expected_hash
        == EXPECTED_ACTIVE_PARAMETER_HASH
    ):
        raise RecoveryError("final replay active parameter hash mismatch")
    return {
        "pass": True,
        "metric_absolute_tolerance": FINAL_REPLAY_ATOL,
        "metric_errors": metric_errors,
        "metric_max_absolute_error": maximum_metric_error,
        "metric_max_error_name": max(metric_errors, key=metric_errors.get),
        "spectrum_absolute_tolerance": FINAL_REPLAY_ATOL,
        "spectrum_eigenvalue_count": len(observed_eig),
        "spectrum_max_absolute_error": maximum_spectrum_error,
        "spectrum_max_error_rank": int(np.argmax(eigenvalue_errors)),
        "low_basis_hash_matches": True,
        "hessian_shape": list(hessian.shape),
        "matrix_shape": list(matrix.shape),
        "spectrum_shape": list(eig.shape),
        "low_basis_shape": list(low_basis.shape),
        "low_projector_shape": list(low_projector.shape),
        "hessian_sha256": _sha256_tensor(hessian),
        "matrix_sha256": _sha256_tensor(matrix),
        "spectrum_tensor_sha256": _sha256_tensor(eig),
        "low_basis_sha256": _sha256_tensor(low_basis),
        "low_projector_sha256": projector_hash,
        "active_parameter_hash_matches": True,
    }


def _stored_final_spectrum(progress: Mapping[str, Any]) -> np.ndarray:
    final_update = int(progress["accepted_updates"])
    rows = sorted(
        (
            row
            for row in progress["spectrum_rows"]
            if int(row["accepted_update"]) == final_update
        ),
        key=lambda row: int(row["rank"]),
    )
    if len(rows) != DIMENSION or [int(row["rank"]) for row in rows] != list(
        range(DIMENSION)
    ):
        raise RecoveryError("stored final spectrum is incomplete")
    values = np.asarray([float(row["m_eigenvalue"]) for row in rows], dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise RecoveryError("stored final spectrum is nonfinite")
    return values


def _audit_replayed_final_spectrum_file(
    *,
    path: Path,
    stored_eig: np.ndarray,
    replayed_eig: np.ndarray,
) -> dict[str, Any]:
    source = _lstat_regular_file(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["rank", "stored", "replayed", "abs_error"]:
            raise RecoveryError("replayed final spectrum CSV schema mismatch")
        rows = list(reader)
    if len(rows) != DIMENSION:
        raise RecoveryError("replayed final spectrum CSV row count mismatch")
    observed_errors: list[float] = []
    for rank, row in enumerate(rows):
        if set(row) != {"rank", "stored", "replayed", "abs_error"}:
            raise RecoveryError("replayed final spectrum CSV row schema mismatch")
        try:
            observed_rank = int(row["rank"])
            observed_stored = float(row["stored"])
            observed_replayed = float(row["replayed"])
            observed_error = float(row["abs_error"])
        except (TypeError, ValueError) as error:
            raise RecoveryError(
                "replayed final spectrum CSV value malformed"
            ) from error
        expected_stored = float(stored_eig[rank])
        expected_replayed = float(replayed_eig[rank])
        expected_error = abs(expected_replayed - expected_stored)
        if (
            observed_rank != rank
            or not all(
                math.isfinite(value)
                for value in (observed_stored, observed_replayed, observed_error)
            )
            or observed_stored != expected_stored
            or observed_replayed != expected_replayed
            or observed_error != expected_error
        ):
            raise RecoveryError(f"replayed final spectrum CSV mismatch at rank {rank}")
        observed_errors.append(observed_error)
    maximum = max(observed_errors)
    if maximum > FINAL_REPLAY_ATOL:
        raise RecoveryError("replayed final spectrum CSV exceeds tolerance")
    return {
        "pass": True,
        "sha256": _sha256_file(source),
        "row_count": len(rows),
        "max_absolute_error": maximum,
        "max_error_rank": int(np.argmax(np.asarray(observed_errors))),
    }


def _write_replayed_final_spectrum(
    *, path: Path, progress: Mapping[str, Any], eig: torch.Tensor
) -> dict[str, Any]:
    stored = _stored_final_spectrum(progress)
    replayed = eig.detach().cpu().double().numpy()
    if replayed.shape != (DIMENSION,) or not bool(np.isfinite(replayed).all()):
        raise RecoveryError("replayed final spectrum tensor is invalid")
    rows = [
        {
            "rank": rank,
            "stored": float(stored[rank]),
            "replayed": float(replayed[rank]),
            "abs_error": abs(float(replayed[rank]) - float(stored[rank])),
        }
        for rank in range(DIMENSION)
    ]
    _atomic_csv(path, rows)
    return _audit_replayed_final_spectrum_file(
        path=path,
        stored_eig=stored,
        replayed_eig=replayed,
    )


def _recompute_geometry_metric_closures(
    *,
    hessian: torch.Tensor,
    matrix: torch.Tensor,
    eig: torch.Tensor,
    low_basis: torch.Tensor,
    low_projector: torch.Tensor,
) -> dict[str, float]:
    dim = int(eig.numel())
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
    orthogonality = (
        (
            low_basis.transpose(0, 1) @ low_basis
            - torch.eye(low_basis.shape[1], dtype=torch.float64)
        )
        .abs()
        .max()
    )
    residual = (
        matrix @ low_basis - low_basis * low_values.unsqueeze(0)
    ).norm() / matrix.norm().clamp_min(1e-30)
    canonical_low_energy = torch.trace(
        low_basis.transpose(0, 1) @ matrix @ low_basis
    ) / float(low_basis.shape[1])
    exact_a = contribution.mean()
    result = {
        "hessian_symmetry_rel": float(
            ((hessian - hessian.T).norm() / hessian.norm().clamp_min(1e-30))
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
    projector_closure = float(
        (low_projector - low_basis @ low_basis.transpose(0, 1)).abs().max()
    )
    if projector_closure > FINAL_REPLAY_ATOL:
        raise RecoveryError("stored low projector/basis closure exceeds tolerance")
    return result


def _audit_replayed_geometry_payload(
    *,
    payload: Mapping[str, Any],
    final_row: Mapping[str, Any],
    stored_eig: np.ndarray,
) -> dict[str, Any]:
    if set(payload) != EXPECTED_REPLAYED_GEOMETRY_KEYS:
        raise RecoveryError("replayed final geometry exact schema mismatch")
    metadata_gates = {
        "protocol_id": payload.get("protocol_id") == PROTOCOL_ID,
        "source_protocol_id": payload.get("source_protocol_id") == SOURCE_PROTOCOL_ID,
        "source_progress": payload.get("source_progress_checkpoint_sha256")
        == EXPECTED_PROGRESS_SHA256,
        "accepted_checkpoint": payload.get("accepted_vae_checkpoint_sha256")
        == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "active_parameter_hash": payload.get("active_parameter_hash")
        == EXPECTED_ACTIVE_PARAMETER_HASH,
        "source_weight_index": payload.get("source_weight_index")
        == SOURCE_WEIGHT_INDEX,
        "task_name": payload.get("task_name") == EXPECTED_TASK_NAME,
        "tau": payload.get("tau") == EXPECTED_TASK_TAU,
        "z_sha256": payload.get("z_sha256") == EXPECTED_Z_SHA256,
        "accepted_update": payload.get("accepted_update") == MAX_ACCEPTED_UPDATES,
        "model_nonmutation": payload.get("installed_model_aggregate_sha256")
        == payload.get("post_replay_model_aggregate_sha256"),
        "constants": payload.get("geometry_constants")
        == {
            "dimension": DIMENSION,
            "epsilon": GEOMETRY_EPSILON,
            "old_beta": GEOMETRY_OLD_BETA,
            "low_threshold": GEOMETRY_LOW_THRESHOLD,
            "absolute_tolerance": FINAL_REPLAY_ATOL,
        },
    }
    if not all(metadata_gates.values()):
        raise RecoveryError("replayed final geometry metadata mismatch")
    names = (
        "hessian",
        "matrix",
        "eig",
        "current_low_basis",
        "current_low_projector",
    )
    tensors = {name: payload.get(name) for name in names}
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise RecoveryError("replayed final geometry tensor is missing")
    hessian = tensors["hessian"]
    matrix = tensors["matrix"]
    eig = tensors["eig"]
    low_basis = tensors["current_low_basis"]
    low_projector = tensors["current_low_projector"]
    if (
        any(
            value.device.type != "cpu" or value.dtype != torch.float64
            for value in tensors.values()
        )
        or tuple(hessian.shape) != (DIMENSION, DIMENSION)
        or tuple(matrix.shape) != (DIMENSION, DIMENSION)
        or tuple(eig.shape) != (DIMENSION,)
        or low_basis.ndim != 2
        or low_basis.shape[0] != DIMENSION
        or tuple(low_projector.shape) != (DIMENSION, DIMENSION)
        or not all(bool(torch.isfinite(value).all()) for value in tensors.values())
    ):
        raise RecoveryError("replayed final geometry tensor metadata is invalid")
    expected_fingerprints = {
        name: _tensor_fingerprint(value) for name, value in tensors.items()
    }
    if payload.get("tensor_fingerprints") != expected_fingerprints:
        raise RecoveryError("replayed final geometry tensor fingerprint mismatch")
    recomputed_matrix = hessian @ hessian.transpose(0, 1)
    recomputed_matrix = 0.5 * (recomputed_matrix + recomputed_matrix.transpose(0, 1))
    matrix_closure = float((matrix - recomputed_matrix).abs().max())
    matrix_symmetry = float((matrix - matrix.transpose(0, 1)).abs().max())
    recomputed_eig = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
    eig_closure = float((eig - recomputed_eig).abs().max())
    stored_spectrum_error = float(
        np.abs(eig.numpy() - np.asarray(stored_eig, dtype=np.float64)).max()
    )
    projector_closure = float(
        (low_projector - low_basis @ low_basis.transpose(0, 1)).abs().max()
    )
    projector_hash = _sha256_tensor(low_projector)
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RecoveryError("replayed final geometry metrics are missing")
    closure_metrics = _recompute_geometry_metric_closures(
        hessian=hessian,
        matrix=matrix,
        eig=eig,
        low_basis=low_basis,
        low_projector=low_projector,
    )
    expected_closure_names = set(metrics) - {
        "task_loss",
        "hessian_sec",
        "low_basis_hash",
    }
    if set(closure_metrics) != expected_closure_names:
        raise RecoveryError("replayed final geometry metric closure schema mismatch")
    metric_closure_errors = {
        name: abs(float(metrics[name]) - value)
        for name, value in closure_metrics.items()
    }
    source_metric_errors = {
        name: abs(float(metrics[name]) - float(final_row[name]))
        for name in set(metrics) - {"hessian_sec", "low_basis_hash"}
    }
    maximum_metric_closure = max(metric_closure_errors.values())
    maximum_source_metric_error = max(source_metric_errors.values())
    if (
        matrix_closure > FINAL_REPLAY_ATOL
        or matrix_symmetry > FINAL_REPLAY_ATOL
        or eig_closure > FINAL_REPLAY_ATOL
        or stored_spectrum_error > FINAL_REPLAY_ATOL
        or projector_closure > FINAL_REPLAY_ATOL
        or projector_hash != metrics.get("low_basis_hash")
        or projector_hash != final_row.get("low_basis_hash")
        or maximum_metric_closure > FINAL_REPLAY_ATOL
        or maximum_source_metric_error > FINAL_REPLAY_ATOL
    ):
        raise RecoveryError("replayed final geometry scientific closure failed")
    return {
        "pass": True,
        "metadata_gates": metadata_gates,
        "matrix_from_hessian_max_abs_error": matrix_closure,
        "matrix_symmetry_max_abs_error": matrix_symmetry,
        "eigvalsh_matrix_max_abs_error": eig_closure,
        "stored_spectrum_max_abs_error": stored_spectrum_error,
        "low_projector_basis_max_abs_error": projector_closure,
        "low_projector_sha256": projector_hash,
        "low_basis_orthogonality_max_abs": closure_metrics[
            "low_basis_orthogonality_max_abs"
        ],
        "low_basis_eigen_residual_relative": closure_metrics[
            "low_basis_eigen_residual_relative"
        ],
        "metric_closure_errors": metric_closure_errors,
        "metric_closure_max_abs_error": maximum_metric_closure,
        "source_metric_max_abs_error": maximum_source_metric_error,
        "tensor_fingerprints": expected_fingerprints,
    }


def _write_and_reload_replayed_geometry(
    *,
    path: Path,
    final_state: Mapping[str, Any],
    progress: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensor_names = (
        "hessian",
        "matrix",
        "eig",
        "current_low_basis",
        "current_low_projector",
    )
    tensors: dict[str, torch.Tensor] = {}
    for name in tensor_names:
        value = final_state.get(name)
        if not isinstance(value, torch.Tensor):
            raise RecoveryError(f"final geometry tensor is missing: {name}")
        tensors[name] = value.detach().cpu().double().contiguous().clone()
    installed_aggregate = reconstruction["model_tensor_snapshots"]["installed"][
        "aggregate_sha256"
    ]
    post_aggregate = reconstruction["model_tensor_snapshots"]["post_replay"][
        "aggregate_sha256"
    ]
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": EXPECTED_PROGRESS_SHA256,
        "accepted_vae_checkpoint_sha256": EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
        "active_parameter_hash": EXPECTED_ACTIVE_PARAMETER_HASH,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "task_name": EXPECTED_TASK_NAME,
        "tau": EXPECTED_TASK_TAU,
        "z_sha256": EXPECTED_Z_SHA256,
        "accepted_update": int(progress["accepted_updates"]),
        "geometry_constants": {
            "dimension": DIMENSION,
            "epsilon": GEOMETRY_EPSILON,
            "old_beta": GEOMETRY_OLD_BETA,
            "low_threshold": GEOMETRY_LOW_THRESHOLD,
            "absolute_tolerance": FINAL_REPLAY_ATOL,
        },
        **tensors,
        "metrics": dict(final_state["metrics"]),
        "tensor_fingerprints": {
            name: _tensor_fingerprint(value) for name, value in tensors.items()
        },
        "installed_model_aggregate_sha256": installed_aggregate,
        "post_replay_model_aggregate_sha256": post_aggregate,
    }
    _atomic_torch_save(path, payload)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(reloaded, Mapping):
        raise RecoveryError("replayed final geometry artifact is malformed")
    audit = _audit_replayed_geometry_payload(
        payload=reloaded,
        final_row=progress["state_rows"][-1],
        stored_eig=_stored_final_spectrum(progress),
    )
    audit["artifact_sha256"] = _sha256_file(path)
    audit["cpu_reload_pass"] = True
    return dict(reloaded), audit


def _copy_source_evidence(
    *, stage: Path, guard: FrozenInputGuard, reviewer_path: Path
) -> dict[str, str]:
    copies = {
        "source_progress_checkpoint.pt": guard.source_staging
        / "progress_checkpoint.pt",
        "state_metrics.csv": guard.source_staging / "state_metrics.csv",
        "state_spectra.csv": guard.source_staging / "state_spectra.csv",
        "proposal_diagnostics.csv": guard.source_staging / "proposal_diagnostics.csv",
        "line_search.csv": guard.source_staging / "line_search.csv",
        "arm_selection.csv": guard.source_staging / "arm_selection.csv",
        "source_intervention_origin.json": guard.source_staging
        / "intervention_origin.json",
        "source_intervention_preflight.json": guard.source_staging
        / "intervention_preflight.json",
        "source_resolved_config.json": guard.source_staging / "resolved_config.json",
        "source_run.log": guard.source_staging / "run.log",
        "frozen_executed_producer_snapshot.py": guard.source_staging
        / "executed_source_snapshot.py",
        "frozen_producer_dependency_manifest_snapshot.json": guard.source_staging
        / "frozen_dependency_manifest_snapshot.json",
        "frozen_producer_protocol_snapshot.md": guard.source_staging
        / "protocol_snapshot.md",
        "frozen_continuation_reviewer_snapshot.py": reviewer_path,
        "recovery_protocol_snapshot.md": RECOVERY_PROTOCOL,
        "recovery_frozen_input_manifest_snapshot.json": guard.manifest_path,
        "recovery_task_manifest_snapshot.json": RECOVERY_TASK_MANIFEST,
        "recovery_import_manifest_snapshot.json": RECOVERY_IMPORT_MANIFEST,
        "executed_recovery_finalizer_source_snapshot.py": LIVE_SOURCE,
    }
    hashes: dict[str, str] = {}
    for destination_name, source in copies.items():
        frozen_source = _lstat_regular_file(source)
        destination = stage / destination_name
        shutil.copyfile(frozen_source, destination)
        _lstat_regular_file(destination)
        hashes[destination_name] = _sha256_file(destination)
    return hashes


def _plot_recovery(
    stage: Path,
    *,
    states: pd.DataFrame,
    spectra: pd.DataFrame,
    proposals: pd.DataFrame,
    target_norm: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    panels = (
        ("exact_a_per_dim", "Exact dense A"),
        ("damped_full_burg_per_dim", "Damped full Burg B"),
        ("a_low90_abs_per_dim", "Lower-90 A contribution"),
        ("a_gt1", "High-tail A contribution"),
        ("m_p50", "Median M eigenvalue"),
        ("effective_rank", "Effective rank"),
    )
    for axis, (column, title) in zip(axes.flat, panels, strict=True):
        axis.plot(states["accepted_update"], states[column], linewidth=2.0)
        axis.axvline(START_UPDATE, color="black", linestyle="--", linewidth=1.0)
        axis.set(title=title, xlabel="global accepted update", ylabel=column)
        axis.grid(alpha=0.25)
    fig.suptitle("Recovered frozen exact A+low trajectory")
    fig.savefig(stage / "recovery_trajectory.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for update in (0, START_UPDATE, MAX_ACCEPTED_UPDATES):
        rows = spectra.loc[spectra["accepted_update"].eq(update)].sort_values("rank")
        axis.plot(
            rows["rank"],
            np.clip(rows["m_eigenvalue"], 1e-14, None),
            label=f"update {update}",
        )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_yscale("log")
    axis.set(title="Ordered M spectra", xlabel="rank", ylabel="eigenvalue")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(stage / "recovery_spectra.png", dpi=180)
    plt.close(fig)

    continuation = proposals.loc[
        proposals["target_update"].astype(int).gt(START_UPDATE)
    ].copy()
    continuation["radius_absolute_error"] = (
        continuation["direction_norm"].astype(float) - target_norm
    ).abs()
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(
        continuation["target_update"],
        continuation["radius_absolute_error"],
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    axis.axhline(ORIGINAL_RADIUS_ATOL, color="red", linestyle="--", label="1e-9")
    axis.axhline(RECOVERY_RADIUS_ATOL, color="black", linestyle="--", label="5e-9")
    axis.set(
        title="Frozen direction-radius serialization error",
        xlabel="global proposal",
        ylabel="absolute radius error",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(stage / "recovery_direction_radius_audit.png", dpi=180)
    plt.close(fig)


def _validate_exact_regular_file_set(
    directory: Path, expected_names: set[str]
) -> dict[str, Any]:
    root = _lstat_directory(directory)
    entries = list(root.iterdir())
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise RecoveryError(
            "staged exact entry set mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    hashes: dict[str, str] = {}
    for entry in entries:
        regular = _lstat_regular_file(entry)
        hashes[entry.name] = _sha256_file(regular)
    return {
        "pass": True,
        "entry_count": len(entries),
        "entries_sha256": dict(sorted(hashes.items())),
    }


def _artifact_manifest(stage: Path) -> dict[str, Any]:
    current_expected = EXPECTED_MANIFESTED_ARTIFACTS | {"INCOMPLETE", "run.log"}
    _validate_exact_regular_file_set(stage, current_expected)
    artifacts = sorted(stage / name for name in EXPECTED_MANIFESTED_ARTIFACTS)
    return {
        "protocol_id": PROTOCOL_ID,
        "recovery_finalizer_source_sha256": _sha256_file(LIVE_SOURCE),
        "recovery_finalizer_normalized_source_sha256": (
            EXPECTED_NORMALIZED_SOURCE_SHA256
        ),
        "frozen_input_manifest_sha256": EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
        "recovery_task_manifest_sha256": EXPECTED_RECOVERY_TASK_MANIFEST_SHA256,
        "recovery_import_manifest_sha256": EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256,
        "source_progress_checkpoint_sha256": EXPECTED_PROGRESS_SHA256,
        "artifacts": {path.name: _sha256_file(path) for path in artifacts},
    }


def _validate_staged_publication(stage: Path) -> dict[str, bool]:
    exact_files = _validate_exact_regular_file_set(stage, EXPECTED_STAGED_FILE_SET)
    manifest = json.loads((stage / "artifact_manifest.json").read_text())
    finalized = json.loads((stage / "FINALIZED.json").read_text())
    decision = json.loads((stage / "decision.json").read_text())
    audit = json.loads((stage / "recovery_audit.json").read_text())
    actual_artifacts = set(EXPECTED_MANIFESTED_ARTIFACTS)
    artifact_hashes = manifest.get("artifacts", {})
    freeze_audit = _verify_finalizer_source_freeze(
        snapshot=stage / "executed_recovery_finalizer_source_snapshot.py",
        protocol_snapshot=stage / "recovery_protocol_snapshot.md",
    )
    geometry = torch.load(
        stage / "replayed_final_geometry.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(geometry, Mapping):
        raise RecoveryError("staged replayed geometry is malformed")
    states = pd.read_csv(stage / "state_metrics.csv")
    final_states = states.loc[
        states["accepted_update"].astype(int).eq(MAX_ACCEPTED_UPDATES)
    ]
    spectra = pd.read_csv(stage / "state_spectra.csv")
    final_spectra = spectra.loc[
        spectra["accepted_update"].astype(int).eq(MAX_ACCEPTED_UPDATES)
    ].sort_values("rank")
    if len(final_states) != 1 or len(final_spectra) != DIMENSION:
        raise RecoveryError("staged endpoint state/spectrum is incomplete")
    stored_eig = final_spectra["m_eigenvalue"].to_numpy(dtype=np.float64)
    geometry_audit = _audit_replayed_geometry_payload(
        payload=geometry,
        final_row=final_states.iloc[0].to_dict(),
        stored_eig=stored_eig,
    )
    replayed_eig = geometry["eig"].numpy()
    spectrum_audit = _audit_replayed_final_spectrum_file(
        path=stage / "replayed_final_spectrum.csv",
        stored_eig=stored_eig,
        replayed_eig=replayed_eig,
    )
    replay_record = audit.get("final_geometry_replay", {})
    gates = {
        "exact_regular_file_set": exact_files["pass"],
        "manifest_protocol_matches": manifest.get("protocol_id") == PROTOCOL_ID,
        "manifest_names_exact": isinstance(artifact_hashes, Mapping)
        and set(artifact_hashes) == actual_artifacts,
        "manifest_hashes_match": isinstance(artifact_hashes, Mapping)
        and all(
            _sha256_file(stage / name) == digest
            for name, digest in artifact_hashes.items()
        ),
        "finalized_protocol_matches": finalized.get("protocol_id") == PROTOCOL_ID,
        "finalized_status_exact": finalized.get("status")
        == "complete_awaiting_independent_recovery_review",
        "finalized_recovery_valid": finalized.get("recovery_valid") is True,
        "decision_recovery_valid": decision.get("recovery_valid") is True,
        "audit_recovery_valid": audit.get("recovery_valid") is True,
        "original_failure_preserved": audit.get("original_runner_audit_pass") is False
        and audit.get("original_runner_failed_gates")
        == list(EXPECTED_ORIGINAL_FAILED_GATES),
        "geometry_budget_exact": audit.get("geometry_execution", {}).get(
            "geometry_evaluation_count"
        )
        == 1
        and audit.get("geometry_execution", {}).get("forbidden_operation_attempts")
        == 0,
        "decision_hash_matches": finalized.get("decision_sha256")
        == _sha256_file(stage / "decision.json"),
        "audit_hash_matches": finalized.get("recovery_audit_sha256")
        == _sha256_file(stage / "recovery_audit.json"),
        "manifest_hash_matches": finalized.get("artifact_manifest_sha256")
        == _sha256_file(stage / "artifact_manifest.json"),
        "scientific_success_consistent": finalized.get("scientific_success")
        == decision.get("scientific_success"),
        "input_manifest_snapshot_matches": _sha256_file(
            stage / "recovery_frozen_input_manifest_snapshot.json"
        )
        == EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
        "producer_snapshot_matches": _sha256_file(
            stage / "frozen_executed_producer_snapshot.py"
        )
        == EXPECTED_PRODUCER_SHA256,
        "reviewer_snapshot_matches": _sha256_file(
            stage / "frozen_continuation_reviewer_snapshot.py"
        )
        == EXPECTED_REVIEWER_SHA256,
        "task_manifest_snapshot_matches": _sha256_file(
            stage / "recovery_task_manifest_snapshot.json"
        )
        == EXPECTED_RECOVERY_TASK_MANIFEST_SHA256,
        "import_manifest_snapshot_matches": _sha256_file(
            stage / "recovery_import_manifest_snapshot.json"
        )
        == EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256,
        "finalizer_self_freeze_matches": freeze_audit["live_normalized_source_sha256"]
        == EXPECTED_NORMALIZED_SOURCE_SHA256,
        "geometry_artifact_independent_audit": geometry_audit["pass"],
        "spectrum_csv_independent_audit": spectrum_audit["pass"],
        "geometry_artifact_hash_recorded": replay_record.get("geometry_artifact_sha256")
        == _sha256_file(stage / "replayed_final_geometry.pt"),
        "spectrum_csv_hash_recorded": replay_record.get(
            "replayed_final_spectrum_sha256"
        )
        == spectrum_audit["sha256"],
        "hessian_hash_recorded": replay_record.get("hessian_sha256")
        == geometry_audit["tensor_fingerprints"]["hessian"]["sha256"],
        "matrix_hash_recorded": replay_record.get("matrix_sha256")
        == geometry_audit["tensor_fingerprints"]["matrix"]["sha256"],
        "spectrum_tensor_hash_recorded": replay_record.get("spectrum_tensor_sha256")
        == geometry_audit["tensor_fingerprints"]["eig"]["sha256"],
        "runtime_task_audit_recorded": audit.get("geometry_reconstruction", {})
        .get("task_immediately_pre_replay", {})
        .get("pass")
        is True,
        "runtime_version_audit_recorded": audit.get("geometry_reconstruction", {})
        .get("runtime_provenance", {})
        .get("observed_runtime")
        == EXPECTED_RUNTIME,
        "reconstruction_inputs_recorded_pre_post": audit.get(
            "reconstruction_input_files", {}
        ).get("pass")
        is True
        and audit.get("reconstruction_input_files", {}).get("pre_post_equal") is True
        and audit.get("reconstruction_input_files")
        == audit.get("geometry_reconstruction", {}).get("reconstruction_input_files"),
    }
    if not all(gates.values()):
        raise RecoveryError(
            "staged recovery publication audit failed: "
            f"{[name for name, value in gates.items() if not value]}"
        )
    return gates


@dataclass(frozen=True)
class AtomicPublicationResult:
    output: Path
    failed_staging: Path | None
    payload: Any


def _failure_path(output: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return output.with_name(f"{output.name}.failed.{timestamp}.{os.getpid()}")


def _replace_with_regular_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path) and stat.S_ISDIR(os.lstat(path).st_mode):
            displaced = path.with_name(
                f"{path.name}.rejected-directory.{uuid.uuid4().hex}"
            )
            os.replace(path, displaced)
        os.replace(temporary, path)
        _lstat_regular_file(path)
    finally:
        if not descriptor < 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)


def _atomic_directory_publication(
    output: Path,
    builder: Callable[[Path], Any],
    *,
    protocol_id: str = PROTOCOL_ID,
    prepublish: Callable[[Path], None] | None = None,
) -> AtomicPublicationResult:
    output = _absolute_without_resolve(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _lstat_directory(output.parent)
    _lstat_no_symlink_components(output, allow_missing_leaf=True)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite recovery output: {output}")
    stage = output.with_name(
        f".{output.name}.incomplete.{os.getpid()}.{uuid.uuid4().hex}"
    )
    stage.mkdir()
    _lstat_directory(stage)
    _replace_with_regular_text(stage / "INCOMPLETE", protocol_id + "\n")
    _lstat_regular_file(stage / "INCOMPLETE")
    try:
        payload = builder(stage)
        _lstat_regular_file(stage / "FINALIZED.json")
        _lstat_regular_file(stage / "INCOMPLETE")
        if prepublish is not None:
            prepublish(stage)
        _lstat_directory(stage)
        _lstat_regular_file(stage / "FINALIZED.json")
        _lstat_regular_file(stage / "INCOMPLETE")
        if os.path.lexists(output):
            raise RecoveryError("recovery output appeared before atomic publication")
        (stage / "INCOMPLETE").unlink()
        os.replace(stage, output)
        _lstat_directory(output)
        return AtomicPublicationResult(
            output=output, failed_staging=None, payload=payload
        )
    except BaseException as error:
        failure = {
            "protocol_id": protocol_id,
            "status": "failed_recovery_staging",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=30),
        }
        try:
            _replace_with_regular_text(stage / "INCOMPLETE", protocol_id + "\n")
            _atomic_json(stage / "failure.json", failure)
            failed = _failure_path(output)
            os.replace(stage, failed)
        except Exception:
            failed = stage if stage.exists() else None
        try:
            setattr(error, "failed_recovery_staging", failed)
        except Exception:
            pass
        raise


def _write_recovery_packet(
    *,
    stage: Path,
    output: Path,
    guard: FrozenInputGuard,
    device: torch.device,
    startup_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    with _tee_to_log(stage / "run.log"):
        print(
            "[recovery-finalizer] stage=startup "
            f"protocol={PROTOCOL_ID} device={device} output={output}",
            flush=True,
        )
        if _sha256_file(RECOVERY_PROTOCOL) != EXPECTED_RECOVERY_PROTOCOL_SHA256:
            raise RecoveryError("recovery protocol hash mismatch")
        input_preflight = guard.verify()
        producer, reviewer, dependency_import = _import_verified_dependencies(guard)
        task_manifest, task_manifest_preflight = _verify_recovery_task_manifest(
            repository_root=guard.repository_root
        )
        import_manifest, import_manifest_preflight = _verify_recovery_import_manifest(
            repository_root=guard.repository_root
        )
        if not all(producer._validate_dependencies().values()):
            raise RecoveryError("frozen producer transitive dependency audit failed")
        _, parent_progress, parent_review = producer._validate_parent_packet()
        if parent_review.get("valid") is not True:
            raise RecoveryError("frozen parent independent review is not valid")

        print("[recovery-finalizer] stage=load-terminal-progress", flush=True)
        progress, progress_lineage = _load_source_progress(guard, reviewer)
        expected_original = guard.manifest["expected_original_audit"]
        if expected_original != {
            "pass": False,
            "failed_gates": list(EXPECTED_ORIGINAL_FAILED_GATES),
        }:
            raise RecoveryError("frozen expected original audit entry mismatch")

        print("[recovery-finalizer] stage=original-read-only-audit", flush=True)
        original_audit = _run_original_runner_audit(
            producer=producer,
            progress=progress,
            parent_progress=parent_progress,
        )
        print(
            "[recovery-finalizer] original_audit_pass=0 failed="
            f"{original_audit['failed_gates']}",
            flush=True,
        )

        print("[recovery-finalizer] stage=independent-recovery-audit", flush=True)
        independent_audit = _independent_recovery_audit(
            reviewer=reviewer,
            progress=progress,
            parent_progress=parent_progress,
            guard=guard,
        )
        radius = independent_audit["radius_audit"]
        print(
            "[recovery-finalizer] recovery_radius_pass=1 "
            f"max_error={radius['max_absolute_error']:.12g} "
            f"old_violations={radius['count_exceeding_original_1e_9']} "
            f"recovery_violations={radius['count_exceeding_recovery_5e_9']}",
            flush=True,
        )

        snapshot_hashes = _copy_source_evidence(
            stage=stage,
            guard=guard,
            reviewer_path=Path(reviewer.__file__).resolve(),
        )
        copied_freeze = _verify_finalizer_source_freeze(
            snapshot=stage / "executed_recovery_finalizer_source_snapshot.py",
            protocol_snapshot=stage / "recovery_protocol_snapshot.md",
        )
        _, copied_task_manifest = _verify_recovery_task_manifest(
            repository_root=guard.repository_root,
            manifest_path=stage / "recovery_task_manifest_snapshot.json",
        )
        _, copied_import_manifest = _verify_recovery_import_manifest(
            repository_root=guard.repository_root,
            manifest_path=stage / "recovery_import_manifest_snapshot.json",
        )
        resolved = {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "device": str(device),
            "dtype": "float32 model/HVP; FP64 dense H/M replay products",
            "seed": "none; deterministic frozen terminal-state replay",
            "cache_mode": "accepted h2048 VAE plus frozen terminal progress",
            "source_staging": str(guard.source_staging),
            "source_staging_mutation_allowed": False,
            "source_progress_checkpoint_sha256": EXPECTED_PROGRESS_SHA256,
            "frozen_input_manifest_sha256": EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
            "recovery_protocol_sha256": EXPECTED_RECOVERY_PROTOCOL_SHA256,
            "recovery_task_manifest_sha256": (EXPECTED_RECOVERY_TASK_MANIFEST_SHA256),
            "recovery_import_manifest_sha256": (
                EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
            ),
            "producer_source_sha256": EXPECTED_PRODUCER_SHA256,
            "continuation_reviewer_source_sha256": EXPECTED_REVIEWER_SHA256,
            "geometry_evaluation_budget": 1,
            "optimization_gradient_evaluation_budget": 0,
            "proposal_budget": 0,
            "line_search_budget": 0,
            "parameter_update_budget": 0,
            "metric_replay_absolute_tolerance": FINAL_REPLAY_ATOL,
            "spectrum_replay_absolute_tolerance": FINAL_REPLAY_ATOL,
            "direction_radius_absolute_tolerance": RECOVERY_RADIUS_ATOL,
            "direction_radius_relative_tolerance": 0.0,
            "output_dir": str(output),
            "working_dir": str(stage),
            "finalizer_source_sha256": _sha256_file(LIVE_SOURCE),
            "finalizer_normalized_source_sha256": (EXPECTED_NORMALIZED_SOURCE_SHA256),
            "startup_self_freeze": dict(startup_freeze),
            "copied_self_freeze": copied_freeze,
            "task_manifest_preflight": task_manifest_preflight,
            "import_manifest_preflight": import_manifest_preflight,
            "copied_task_manifest_audit": copied_task_manifest,
            "copied_import_manifest_audit": copied_import_manifest,
            "expected_runtime": task_manifest["expected_runtime"],
            "accepted_run_dir": str(EXPECTED_ACCEPTED_RUN_DIR),
            "accepted_run_input_sha256": {
                "config.json": EXPECTED_ACCEPTED_RUN_CONFIG_SHA256,
                "weight_pool.pt": EXPECTED_ACCEPTED_RUN_WEIGHT_POOL_SHA256,
                "weight_pool_records.csv": EXPECTED_ACCEPTED_RUN_RECORDS_SHA256,
                "vae_checkpoint.pt": EXPECTED_ACCEPTED_RUN_CHECKPOINT_SHA256,
            },
            "snapshot_hashes": snapshot_hashes,
        }
        _atomic_json(stage / "resolved_config.json", resolved)

        print("[recovery-finalizer] stage=terminal-checkpoint-lineage", flush=True)
        final_checkpoint, checkpoint_audit = _write_and_reload_final_checkpoint(
            path=stage / "final_checkpoint.pt",
            progress=progress,
        )
        guard.verify()

        print("[recovery-finalizer] stage=single-final-geometry-replay", flush=True)
        replay_guard = GeometryReplayGuard()
        final_state, reconstruction = _fresh_final_geometry_replay(
            producer=producer,
            progress=progress,
            task_manifest=task_manifest,
            device=device,
            replay_guard=replay_guard,
        )
        geometry_execution = replay_guard.require_complete()
        replay = _compare_final_geometry_replay(
            final_state=final_state,
            progress=progress,
            checkpoint=final_checkpoint,
            execution=reconstruction,
        )
        spectrum_replay_audit = _write_replayed_final_spectrum(
            path=stage / "replayed_final_spectrum.csv",
            progress=progress,
            eig=final_state["eig"],
        )
        _, geometry_artifact_audit = _write_and_reload_replayed_geometry(
            path=stage / "replayed_final_geometry.pt",
            final_state=final_state,
            progress=progress,
            reconstruction=reconstruction,
        )
        replay.update(
            {
                "replayed_final_spectrum_sha256": spectrum_replay_audit["sha256"],
                "replayed_final_spectrum_rows": spectrum_replay_audit["row_count"],
                "geometry_artifact_sha256": geometry_artifact_audit["artifact_sha256"],
                "geometry_artifact_audit": geometry_artifact_audit,
            }
        )
        if not (
            replay["hessian_sha256"]
            == geometry_artifact_audit["tensor_fingerprints"]["hessian"]["sha256"]
            and replay["matrix_sha256"]
            == geometry_artifact_audit["tensor_fingerprints"]["matrix"]["sha256"]
            and replay["spectrum_tensor_sha256"]
            == geometry_artifact_audit["tensor_fingerprints"]["eig"]["sha256"]
        ):
            raise RecoveryError("geometry artifact tensor hash lineage mismatch")
        print(
            "[recovery-finalizer] geometry_replay_pass=1 evaluations=1 "
            f"metric_max={replay['metric_max_absolute_error']:.12g} "
            f"spectrum_max={replay['spectrum_max_absolute_error']:.12g}",
            flush=True,
        )

        states = pd.DataFrame(progress["state_rows"]).sort_values("accepted_update")
        spectra = pd.DataFrame(progress["spectrum_rows"]).sort_values(
            ["accepted_update", "rank"]
        )
        proposals = pd.DataFrame(progress["proposal_rows"])
        outcome = producer._outcome(
            states=states,
            spectra=spectra,
            proposals=proposals,
            tolerances=progress["tolerances"],
            accepted_updates=int(progress["accepted_updates"]),
            termination=str(progress["termination"]),
            history_summary=independent_audit["history_summary"],
        )
        outcome["success_gates"]["final_checkpoint_replay"] = replay["pass"]
        if set(outcome["success_gates"]) != EXPECTED_SUCCESS_GATES:
            raise RecoveryError("scientific-success gate schema changed")
        outcome["scientific_success"] = bool(all(outcome["success_gates"].values()))

        _atomic_csv(
            stage / "historical_transition_audit.csv",
            independent_audit["historical_transition_rows"],
        )
        _plot_recovery(
            stage,
            states=states,
            spectra=spectra,
            proposals=proposals,
            target_norm=float(reviewer.TARGET_NORM),
        )
        input_postflight = guard.verify()
        task_manifest_post, task_manifest_postflight = _verify_recovery_task_manifest(
            repository_root=guard.repository_root
        )
        import_manifest_post, import_manifest_postflight = (
            _verify_recovery_import_manifest(repository_root=guard.repository_root)
        )
        runtime_import_closure = _audit_loaded_repository_module_closure(
            repository_root=guard.repository_root,
            import_manifest=import_manifest_post,
        )
        recovery_validity_gates = {
            "immutable_input_preflight": input_preflight["source_file_count"] == 14,
            "immutable_input_postflight": input_postflight == input_preflight,
            "startup_finalizer_self_freeze": startup_freeze[
                "live_normalized_source_sha256"
            ]
            == EXPECTED_NORMALIZED_SOURCE_SHA256,
            "copied_finalizer_self_freeze": copied_freeze[
                "snapshot_normalized_source_sha256"
            ]
            == EXPECTED_NORMALIZED_SOURCE_SHA256,
            "task_manifest_pre_post_exact": task_manifest_post == task_manifest
            and task_manifest_postflight == task_manifest_preflight,
            "import_manifest_pre_post_exact": import_manifest_post == import_manifest
            and import_manifest_postflight == import_manifest_preflight,
            "runtime_import_closure_exact": runtime_import_closure["pass"],
            "frozen_dependencies_verified_before_import": bool(dependency_import),
            "loaded_producer_equals_executed_snapshot": bool(
                dependency_import["executed_snapshot_equals_loaded_producer"]
            ),
            "terminal_progress_lineage": progress_lineage["active_parameter_hash"]
            == EXPECTED_ACTIVE_PARAMETER_HASH,
            "original_runner_expected_failure_observed": original_audit["pass"] is False
            and original_audit["failed_gates"] == list(EXPECTED_ORIGINAL_FAILED_GATES),
            "all_unmodified_original_row_gates_pass": original_audit[
                "all_unmodified_row_gates_pass"
            ],
            "independent_recovery_audit_pass": independent_audit["pass"],
            "recovery_radius_audit_pass": radius["pass"],
            "checkpoint_cpu_bitwise_lineage_pass": checkpoint_audit["pass"],
            "runtime_versions_exact": reconstruction["runtime_provenance"][
                "observed_runtime"
            ]
            == EXPECTED_RUNTIME,
            "runtime_task_tensors_exact_immediately_before_replay": reconstruction[
                "task_immediately_pre_replay"
            ]["pass"],
            "runtime_task_tensors_unchanged_after_replay": reconstruction[
                "task_post_replay"
            ]
            == reconstruction["task_immediately_pre_replay"],
            "accepted_checkpoint_exact": reconstruction["accepted_checkpoint_sha256"]
            == EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256,
            "reconstruction_inputs_exact_pre_post_load_run": reconstruction[
                "reconstruction_input_files"
            ]["pass"]
            and reconstruction["reconstruction_input_files"]["pre_post_equal"],
            "terminal_install_changes_exact_active_set": reconstruction[
                "model_installation_audit"
            ]["pass"],
            "all_model_tensors_unchanged_during_replay": reconstruction[
                "model_replay_nonmutation_audit"
            ]["pass"],
            "exactly_one_geometry_evaluation": geometry_execution[
                "geometry_evaluation_count"
            ]
            == 1,
            "no_optimization_operations": all(
                geometry_execution[name] == 0
                for name in (
                    "forbidden_operation_attempts",
                    "optimization_gradient_evaluations",
                    "proposals",
                    "line_searches",
                    "parameter_updates",
                )
            ),
            "final_geometry_replay_pass": replay["pass"],
            "geometry_artifact_independent_closure_pass": geometry_artifact_audit[
                "pass"
            ],
            "replayed_spectrum_csv_pass": spectrum_replay_audit["pass"],
            "scientific_success_criteria_unchanged": set(outcome["success_gates"])
            == EXPECTED_SUCCESS_GATES,
        }
        recovery_valid = bool(all(recovery_validity_gates.values()))
        if not recovery_valid:
            raise RecoveryError(
                "recovery validity gates failed: "
                f"{[name for name, value in recovery_validity_gates.items() if not value]}"
            )

        recovery_audit = {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "recovery_valid": recovery_valid,
            "original_runner_audit_pass": original_audit["pass"],
            "original_runner_failed_gates": original_audit["failed_gates"],
            "original_runner_audit": original_audit,
            "independent_recovery_audit": {
                key: value
                for key, value in independent_audit.items()
                if key != "historical_transition_rows"
            },
            "checkpoint_lineage": checkpoint_audit,
            "geometry_execution": geometry_execution,
            "geometry_reconstruction": reconstruction,
            "reconstruction_input_files": reconstruction["reconstruction_input_files"],
            "final_geometry_replay": replay,
            "immutable_input_preflight": input_preflight,
            "immutable_input_postflight": input_postflight,
            "dependency_import": dependency_import,
            "task_manifest_preflight": task_manifest_preflight,
            "task_manifest_postflight": task_manifest_postflight,
            "import_manifest_preflight": import_manifest_preflight,
            "import_manifest_postflight": import_manifest_postflight,
            "runtime_import_closure": runtime_import_closure,
            "startup_self_freeze": dict(startup_freeze),
            "copied_self_freeze": copied_freeze,
            "recovery_validity_gates": recovery_validity_gates,
        }
        _atomic_json(stage / "recovery_audit.json", recovery_audit)
        decision = {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "recovery_valid": recovery_valid,
            "scientific_success": outcome["scientific_success"],
            "recovery_validity_gates": recovery_validity_gates,
            "accepted_updates": int(progress["accepted_updates"]),
            "new_accepted_updates": int(progress["new_accepted_updates"]),
            "selected_arm": progress["selected_arm"],
            "termination": progress["termination"],
            "outcome": outcome,
            "final_metrics": dict(final_state["metrics"]),
            "original_runner_audit_pass": False,
            "original_runner_failed_gates": list(EXPECTED_ORIGINAL_FAILED_GATES),
            "recovery_radius_audit": radius,
            "final_geometry_replay": replay,
            "final_checkpoint_sha256": checkpoint_audit["final_checkpoint_sha256"],
            "accepted_vae_checkpoint_sha256": (EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256),
            "recovery_task_manifest_sha256": (EXPECTED_RECOVERY_TASK_MANIFEST_SHA256),
            "recovery_import_manifest_sha256": (
                EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
            ),
            "recovery_audit_sha256": _sha256_file(stage / "recovery_audit.json"),
            "elapsed_sec": time.perf_counter() - started,
        }
        _atomic_json(stage / "decision.json", decision)
        guard.verify()
        manifest = _artifact_manifest(stage)
        _atomic_json(stage / "artifact_manifest.json", manifest)
        finalized = {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "status": "complete_awaiting_independent_recovery_review",
            "recovery_valid": recovery_valid,
            "scientific_success": outcome["scientific_success"],
            "accepted_updates": int(progress["accepted_updates"]),
            "new_accepted_updates": int(progress["new_accepted_updates"]),
            "termination": progress["termination"],
            "source_progress_checkpoint_sha256": EXPECTED_PROGRESS_SHA256,
            "accepted_vae_checkpoint_sha256": (EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256),
            "recovery_task_manifest_sha256": (EXPECTED_RECOVERY_TASK_MANIFEST_SHA256),
            "recovery_import_manifest_sha256": (
                EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
            ),
            "recovery_finalizer_source_sha256": _sha256_file(LIVE_SOURCE),
            "recovery_finalizer_normalized_source_sha256": (
                EXPECTED_NORMALIZED_SOURCE_SHA256
            ),
            "decision_sha256": _sha256_file(stage / "decision.json"),
            "recovery_audit_sha256": _sha256_file(stage / "recovery_audit.json"),
            "artifact_manifest_sha256": _sha256_file(stage / "artifact_manifest.json"),
            "final_checkpoint_sha256": _sha256_file(stage / "final_checkpoint.pt"),
        }
        _atomic_json(stage / "FINALIZED.json", finalized)
        publication_gates = _validate_staged_publication(stage)
        guard.verify()
        print(
            "[recovery-finalizer] stage=ready-to-publish "
            f"recovery_valid=1 scientific_success={int(outcome['scientific_success'])} "
            f"A={float(states.iloc[0]['exact_a_per_dim']):.9g}->"
            f"{float(states.iloc[-1]['exact_a_per_dim']):.9g} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return {
            "recovery_valid": recovery_valid,
            "scientific_success": outcome["scientific_success"],
            "publication_gates": publication_gates,
            "geometry_execution": geometry_execution,
            "final_a": float(states.iloc[-1]["exact_a_per_dim"]),
        }


def finalize_recovery() -> AtomicPublicationResult:
    startup_freeze = _verify_finalizer_source_freeze()
    guard = FrozenInputGuard.create(
        repository_root=ROOT,
        manifest_path=FROZEN_INPUT_MANIFEST,
        expected_manifest_sha256=EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
    )
    output = _require_default_output(DEFAULT_OUTPUT)
    _validate_output_separation(guard.source_staging, output)

    def prepublish(stage: Path) -> None:
        guard.verify()
        _verify_recovery_task_manifest(repository_root=ROOT)
        _verify_recovery_import_manifest(repository_root=ROOT)
        _verify_finalizer_source_freeze(
            snapshot=stage / "executed_recovery_finalizer_source_snapshot.py",
            protocol_snapshot=stage / "recovery_protocol_snapshot.md",
        )
        _validate_staged_publication(stage)

    result = _atomic_directory_publication(
        output,
        lambda stage: _write_recovery_packet(
            stage=stage,
            output=output,
            guard=guard,
            device=torch.device(PRODUCTION_DEVICE),
            startup_freeze=startup_freeze,
        ),
        prepublish=prepublish,
    )
    try:
        guard.verify()
    except Exception as error:
        failed = _failure_path(output)
        if output.exists():
            _replace_with_regular_text(output / "INCOMPLETE", PROTOCOL_ID + "\n")
            _atomic_json(
                output / "failure.json",
                {
                    "protocol_id": PROTOCOL_ID,
                    "status": "failed_postpublication_input_audit",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            os.replace(output, failed)
        raise
    print(f"[recovery-finalizer] published output={result.output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provenance-preserving finalizer for the frozen exact trajectory"
    )
    parser.parse_args()
    finalize_recovery()


if __name__ == "__main__":
    main()
