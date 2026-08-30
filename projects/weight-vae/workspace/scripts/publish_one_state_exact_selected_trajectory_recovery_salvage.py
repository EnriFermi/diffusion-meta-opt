from __future__ import annotations

import csv
import ctypes
import decimal
import errno
import hashlib
import io
import json
import math
import os
import platform
import stat
import struct
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


EXPECTED_NORMALIZED_SOURCE_SHA256 = "29168b8bcf6d3824c2ef8f11af0ea9b5a14edbb11aaaf51397186d30fa3afc00"  # fmt: skip

PRODUCTION_ROOT = Path("/home/coder/project")
PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_no_replay_salvage_v1"
FREEZE_ID = (
    "one_state_exact_selected_relaxed_cancellation_no_replay_salvage_execution_v1"
)
RECOVERY_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_recovery_v1"
SOURCE_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_v1"

PROTOCOL_SHA256 = "7b457796d930111ca1c568266ee002d92588d0cacc64d8ad8abc023617150141"
FAILED_MANIFEST_SHA256 = (
    "944f0683383abf8923cc5fc7eb19421cd8c75c325a8874b80af8f210b1b80fec"
)
DERIVATION_SHA256 = "8550adfbcfd9dfe366c7876f330d71037462cc1854589a61337078c7b6d06e9d"
RECOVERY_PROTOCOL_SHA256 = (
    "e7f941d8a6632e71ffd09ad9c0c5a3ed11490a3f8fadd3d7f9b1f8001d5108c1"
)
FROZEN_INPUT_MANIFEST_SHA256 = (
    "8d7f5a72092de0fbaffe67a1c2421dfcb227bf412690f470d552ab0069b99f66"
)
RECOVERY_TASK_MANIFEST_SHA256 = (
    "a4b0b4bc1b50d04e2cd6d6ab35fe05a8200caa222cf84228eaee894504a8054b"
)
RECOVERY_IMPORT_MANIFEST_SHA256 = (
    "976d35e42c4fa491324660e044a4bb6ac224786108e2e81effcef5168753da10"
)

FAILED_TREE_SHA256 = "7dc8fc68a0e8e8baa1140b43f7fbfb5bf9fc129b9f5929403419c54bd5b92c82"
STAGED_TREE_SHA256 = "b73fceb960f0bec4c0f5ffc308da0bfe90d45d3d6b9d01ffb20d5554480e0514"
PUBLISHED_TREE_SHA256 = (
    "d3d11880b2909bb1cd2681bdc10b840e8b43b9aa84af49ba4f41cedacba5daf7"
)
STARTUP_TREE_SHA256 = "8d5a465e9c44a3dcb02246b544fddd77b777b6d30b29c40aef343c16ee17fa76"
FAILED_TOTAL_SIZE = 114_951_202
STAGED_TOTAL_SIZE = 114_949_714
PUBLISHED_TOTAL_SIZE = 114_949_656
STARTUP_TOTAL_SIZE = 2_075

SOURCE_PROGRESS_SHA256 = (
    "ab3366870d62beb71b4d59fd2d54e3cb7354963a16e17a8b9cc13527a4d25311"
)
FINAL_CHECKPOINT_SHA256 = (
    "6688e9ad4c94e90833537b0aaca6fa19bff15c294789a2f9d97cf9dfdcf1a593"
)
GEOMETRY_ARTIFACT_SHA256 = (
    "2d4d93bc507d09be2ce0ad7d12ffbeb99ccb8192ba794b9288d2c9b803bc168d"
)
FINALIZER_SOURCE_SHA256 = (
    "5407f712f46090eeeed004a76dea1ea10b250398171575586c1c6737b145b325"
)
FINALIZER_NORMALIZED_SHA256 = (
    "e25be1074055a66ad68c63acc1f722191efbbf8a827ecd73adfeaa090329a599"
)
PRODUCER_SOURCE_SHA256 = (
    "d79462816ff95b2c3f1a94bed06f8a2c12d6a25610dd9385d9361da2e970f39f"
)
CONTINUATION_REVIEWER_SHA256 = (
    "57ae23d336cc348548b744c90e41c1560c5867af512bf4a5b635fc344737adbb"
)
SOURCE_DEPENDENCY_MANIFEST_SHA256 = (
    "9437b54d9a6379adaab9a7e2bd578caf55e3c879bf861f8a0150f124b577eb32"
)
SOURCE_PROTOCOL_SHA256 = (
    "870272310940873a8fb939434979cbef3b42d1a7c68ebde127afe0dfe8453efb"
)
PARENT_FINAL_SHA256 = "7063f393332f2640d599fa41928006f74342c3d681dd9b77a575e5e9e40716e5"
PARENT_PROGRESS_SHA256 = (
    "010d34d51c21ec139fca3e93f51b91c7563f43a524a59b521dcec50de9c752ce"
)
PARENT_REVIEW_SHA256 = (
    "7677a64b38755bed2d553996f103d943ac7d43c7dae748f546e93ba5441b9703"
)
ACCEPTED_VAE_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
ACTIVE_PARAMETER_HASH = (
    "b4d22a52fdb07a51978d466ef0841bfa92b33a625c84e32433ccb6b334039427"
)
TRANSITION_CHAIN_SHA256 = (
    "8ebc37fa89b471a722844d553b2a5a70003cd44b3cda3ce987075a6df8e261fe"
)
Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
GEOMETRY_EIG_SHA256 = "ad87003f2638e0acad92f413e5f44d9c1b8176ce7ff030c744c9f4e5c3ec355b"

DIMENSION = 512
LOW_BASIS_COUNT = 451
START_UPDATE = 17
MAX_ACCEPTED_UPDATES = 100
NEW_ACCEPTED_UPDATES = 83
ACTIVE_TENSOR_COUNT = 31
ACTIVE_PARAMETER_COUNT = 11_685_120
SOURCE_WEIGHT_INDEX = 378
REPLAY_ATOL = 1e-9
GEOMETRY_EPSILON = 1e-4
GEOMETRY_OLD_BETA = 22.536727828943093
GEOMETRY_LOW_THRESHOLD = 0.1
TARGET_NORM = 0.04892722657548397

EXPECTED_PACKET_RUNTIME = {
    "python": "3.12.12",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "torchvision": "0.25.0+cu128",
}
EXPECTED_RUNTIME = {"python": "3.12.12", "torch": "2.10.0+cu128"}
EXPECTED_ACCEPTED_RUN_INPUT_SHA256 = {
    "config.json": "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4",
    "vae_checkpoint.pt": ACCEPTED_VAE_CHECKPOINT_SHA256,
    "weight_pool.pt": "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef",
    "weight_pool_records.csv": (
        "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933"
    ),
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

CONTROL_REL = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_exact_a_h2048/postgoal_relaxed_cancellation"
)
FAILED_REL = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_exact_a_h2048/"
    "postgoal_relaxed_cancellation_recovery_finalization_v1.failed."
    "20260716T074002.762548Z.2787575"
)
STARTUP_FAILED_REL = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_exact_a_h2048/"
    "postgoal_relaxed_cancellation_recovery_finalization_v1.failed."
    "20260716T070631.160509Z.2760686"
)
ORIGINAL_SOURCE_REL = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_exact_a_h2048/postgoal_relaxed_cancellation_production.incomplete"
)
TARGET_REL = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_exact_a_h2048/"
    "postgoal_relaxed_cancellation_recovery_finalization_v1"
)
STAGE_REL = TARGET_REL.with_name(f".{TARGET_REL.name}.salvage.incomplete")
PROTOCOL_REL = CONTROL_REL / "salvage_protocol.md"
FAILED_MANIFEST_REL = CONTROL_REL / "salvage_frozen_failed_manifest.json"
DERIVATION_REL = CONTROL_REL / "salvage_frozen_failed_manifest_derivation.md"
EXECUTION_FREEZE_REL = CONTROL_REL / "salvage_execution_freeze.json"
PUBLISHER_SNAPSHOT_REL = CONTROL_REL / "salvage_publisher_source_snapshot.py"
EXECUTION_RECORD_REL = CONTROL_REL / "salvage_execution_record.json"
INDEPENDENT_REVIEW_REL = CONTROL_REL / "salvage_independent_review.json"
INDEPENDENT_REVIEW_TEMP_REL = (
    CONTROL_REL / ".salvage_independent_review.json.incomplete"
)
RECOVERY_PROTOCOL_REL = CONTROL_REL / "recovery_protocol.md"
FROZEN_INPUT_MANIFEST_REL = CONTROL_REL / "recovery_frozen_input_manifest.json"
RECOVERY_TASK_MANIFEST_REL = CONTROL_REL / "recovery_task_manifest.json"
RECOVERY_IMPORT_MANIFEST_REL = CONTROL_REL / "recovery_import_manifest.json"
PUBLISHER_SOURCE_REL = Path(
    "scripts/publish_one_state_exact_selected_trajectory_recovery_salvage.py"
)
REVIEWER_SOURCE_REL = Path(
    "scripts/review_one_state_exact_selected_trajectory_recovery_salvage.py"
)
PUBLISHER_TEST_REL = Path(
    "tests/one_state_exact_selected_trajectory_recovery_salvage_publish_test.py"
)
REVIEWER_TEST_REL = Path(
    "tests/one_state_exact_selected_trajectory_recovery_salvage_review_test.py"
)
PARENT_FINAL_REL = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_exact_a_h2048/iteration6_exact_selected_trajectory_production/"
    "final_checkpoint.pt"
)
PARENT_PROGRESS_REL = PARENT_FINAL_REL.with_name("progress_checkpoint.pt")
PARENT_REVIEW_REL = PARENT_FINAL_REL.with_name("independent_review.json")
ACCEPTED_RUN_REL = Path(
    "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing/"
    "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)

LIVE_SOURCE = Path(os.path.abspath(__file__))
SELF_FREEZE_PREFIX = "EXPECTED_NORMALIZED_SOURCE_SHA256 = "
SELF_FREEZE_MASK = "<FROZEN>"

ACCEPTANCE_GATE_NAMES = (
    "failed_tree_exact",
    "original_source_tree_exact",
    "startup_failure_tree_exact",
    "publisher_source_frozen",
    "protocol_and_manifests_frozen",
    "failure_and_run_log_exact",
    "packet_hash_graph_exact",
    "execution_budget_exact",
    "checkpoint_lineage_exact",
    "task_and_input_provenance_exact",
    "model_nonmutation_exact",
    "geometry_shapes_dtypes_finite",
    "matrix_from_hessian_closure",
    "eigensystem_closure",
    "low_basis_projector_closure",
    "metric_closure",
    "spectrum_bitwise_closure",
    "csv_checkpoint_closure",
    "scientific_outcome_unchanged",
    "stage_path_exclusive",
    "target_absent",
    "record_path_exclusive",
    "same_filesystem",
    "byte_copy_exact",
    "hardlinks_absent",
    "staged_tree_exact",
    "protected_trees_unchanged_prepublication",
    "cuda_uninitialized",
    "scientific_execution_absent",
)
EXECUTION_RECORD_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "created_at_utc",
    "repository_root",
    "protocol_sha256",
    "failed_manifest_sha256",
    "derivation_sha256",
    "execution_freeze_sha256",
    "publisher_source_sha256",
    "publisher_normalized_source_sha256",
    "publisher_snapshot_sha256",
    "reviewer_source_sha256",
    "publisher_test_sha256",
    "reviewer_test_sha256",
    "source",
    "staging",
    "publication",
    "copy_audit",
    "protected_lineage",
    "scientific_execution",
    "numeric_audit",
    "acceptance_gates",
}
FREEZE_KEYS = {
    "schema_version",
    "protocol_id",
    "frozen_at_utc",
    "repository_root",
    "protocol_sha256",
    "failed_manifest_sha256",
    "derivation_sha256",
    "publisher",
    "reviewer",
    "publisher_test",
    "reviewer_test",
    "runtime",
    "import_policy",
    "paths",
    "prereview",
}
ALLOWED_TOP_LEVEL_IMPORTS = [
    "__future__",
    "ast",
    "binascii",
    "collections",
    "csv",
    "ctypes",
    "dataclasses",
    "datetime",
    "decimal",
    "errno",
    "hashlib",
    "io",
    "json",
    "math",
    "os",
    "pathlib",
    "platform",
    "stat",
    "struct",
    "sys",
    "time",
    "torch",
    "typing",
    "zlib",
]
FORBIDDEN_TOKENS = [
    "importlib",
    "runpy",
    "subprocess",
    "exec(",
    "eval(",
    "_load_run",
    "_evaluate",
    "torch.autograd",
    "torch.optim",
    "torch.cuda.",
    "backward(",
    ".grad(",
]
SCIENTIFIC_EXECUTION_ZERO = {
    "automatic_differentiation_calls": 0,
    "cuda_initializations": 0,
    "hessian_vector_products": 0,
    "line_searches": 0,
    "model_loads": 0,
    "new_geometry_evaluations": 0,
    "optimization_gradient_evaluations": 0,
    "optimizer_operations": 0,
    "packet_source_writes": 0,
    "parameter_updates": 0,
    "proposals": 0,
    "protected_tree_writes": 0,
    "task_loads": 0,
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
EXPECTED_FAILED_SUCCESS_GATES = [
    "final_a_at_most_0p90",
    "non_top_only_fraction_at_least_0p25",
]


class SalvageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    root: Path
    dependency_root: Path

    @classmethod
    def for_root(cls, root: Path, *, dependency_root: Path | None = None) -> Layout:
        absolute = _absolute(root)
        dependencies = _absolute(dependency_root or root)
        return cls(root=absolute, dependency_root=dependencies)

    def path(self, relative: Path) -> Path:
        return self.root / relative

    def dependency_path(self, relative: Path) -> Path:
        return self.dependency_root / relative

    @property
    def failed(self) -> Path:
        return self.path(FAILED_REL)

    @property
    def original_source(self) -> Path:
        return self.path(ORIGINAL_SOURCE_REL)

    @property
    def startup_failed(self) -> Path:
        return self.path(STARTUP_FAILED_REL)

    @property
    def stage(self) -> Path:
        return self.path(STAGE_REL)

    @property
    def target(self) -> Path:
        return self.path(TARGET_REL)

    @property
    def record(self) -> Path:
        return self.path(EXECUTION_RECORD_REL)

    @property
    def freeze(self) -> Path:
        return self.path(EXECUTION_FREEZE_REL)

    @property
    def snapshot(self) -> Path:
        return self.path(PUBLISHER_SNAPSHOT_REL)


@dataclass(frozen=True)
class FileSpec:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class FailedManifest:
    payload: dict[str, Any]
    failed_specs: tuple[FileSpec, ...]
    staged_specs: tuple[FileSpec, ...]
    published_specs: tuple[FileSpec, ...]
    startup_specs: tuple[FileSpec, ...]


@dataclass(frozen=True)
class ScientificAudit:
    gates: dict[str, bool]
    numeric_audit: dict[str, Any]
    details: dict[str, Any]


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise SalvageError(message)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


PRODUCTION_LAYOUT = Layout.for_root(PRODUCTION_ROOT)


def _relative_text(path: Path) -> str:
    _require(not path.is_absolute() and ".." not in path.parts, f"invalid path: {path}")
    return path.as_posix()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_float(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SalvageError(f"{label} is not numeric: {value!r}") from error
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def _as_int(value: Any, label: str) -> int:
    number = _as_float(value, label)
    result = int(number)
    _require(number == float(result), f"{label} is nonintegral")
    return result


def _as_csv_int(value: Any, label: str) -> int:
    _require(isinstance(value, str) and value, f"{label} is not integer text")
    try:
        number = decimal.Decimal(value)
    except decimal.InvalidOperation as error:
        raise SalvageError(f"{label} is not integer text: {value!r}") from error
    _require(
        number.is_finite() and number == number.to_integral_value(),
        f"{label} is nonintegral",
    )
    if number.is_zero():
        _require(not number.is_signed(), f"{label} is negative zero")
    return int(number)


def _close(left: Any, right: Any, *, atol: float = REPLAY_ATOL) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


def _equivalent(left: Any, right: Any, *, atol: float = 0.0) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left is right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key], atol=atol) for key in left
        )
    left_sequence = isinstance(left, Sequence) and not isinstance(left, (str, bytes))
    right_sequence = isinstance(right, Sequence) and not isinstance(right, (str, bytes))
    if left_sequence or right_sequence:
        if not left_sequence or not right_sequence or len(left) != len(right):
            return False
        return all(
            _equivalent(a, b, atol=atol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, str) or isinstance(right, str):
        return type(left) is str and type(right) is str and left == right
    return _close(left, right, atol=atol)


def _float_bits(value: Any) -> bytes:
    return struct.pack(">d", _as_float(value, "binary64 value"))


def _bitwise_float_equal(left: Any, right: Any) -> bool:
    try:
        return _float_bits(left) == _float_bits(right)
    except SalvageError:
        return False


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_utc(value: Any, label: str) -> None:
    _require(isinstance(value, str) and value.endswith("Z"), f"{label} is not UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise SalvageError(f"{label} is malformed") from error
    _require(parsed.tzinfo is not None, f"{label} lacks timezone")


def _cuda_uninitialized() -> bool:
    module = sys.modules.get("torch.cuda")
    return not bool(getattr(module, "_initialized", False))


def _require_cuda_uninitialized() -> None:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "device mask changed")
    _require(_cuda_uninitialized(), "CUDA was initialized")


def _require_no_symlink_components(
    path: Path, *, allow_missing_leaf: bool = False
) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current /= part
        is_leaf = index == len(absolute.parts[1:]) - 1
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and is_leaf and not os.path.lexists(current):
                return absolute
            raise SalvageError(f"missing path component: {current}") from None
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"symlink path component is forbidden: {current}",
        )
    return absolute


def _require_under_root(root: Path, path: Path) -> Path:
    root = _require_directory(root)
    absolute = _absolute(path)
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise SalvageError(f"path escapes repository root: {absolute}") from error
    return absolute


def _require_regular_file(
    path: Path, *, immutable: bool = False, single_link: bool = False
) -> Path:
    absolute = _require_no_symlink_components(path)
    metadata = os.lstat(absolute)
    _require(stat.S_ISREG(metadata.st_mode), f"not a regular file: {absolute}")
    if single_link:
        _require(metadata.st_nlink == 1, f"hardlinked file is forbidden: {absolute}")
    if immutable:
        _require(
            stat.S_IMODE(metadata.st_mode) & 0o222 == 0,
            f"immutable file is writable: {absolute}",
        )
    return absolute


def _require_directory(path: Path) -> Path:
    absolute = _require_no_symlink_components(path)
    _require(stat.S_ISDIR(os.lstat(absolute).st_mode), f"not a directory: {absolute}")
    return absolute


def _require_absent(path: Path, label: str) -> Path:
    absolute = _require_no_symlink_components(path, allow_missing_leaf=True)
    _require(not os.path.lexists(absolute), f"{label} exists: {absolute}")
    return absolute


def _open_readonly(path: Path) -> tuple[int, os.stat_result]:
    path = _require_regular_file(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    _require(stat.S_ISREG(metadata.st_mode), f"opened non-regular file: {path}")
    return descriptor, metadata


def _sha256_file(path: Path) -> str:
    descriptor, before = _open_readonly(path)
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    return digest.hexdigest()


def _read_bytes(path: Path) -> bytes:
    descriptor, before = _open_readonly(path)
    parts: list[bytes] = []
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            parts.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while reading: {path}",
    )
    value = b"".join(parts)
    _require(len(value) == before.st_size, f"short read: {path}")
    return value


def _strict_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SalvageError(f"{label} is not UTF-8") from error

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SalvageError(f"{label} has duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SalvageError(f"{label} has nonfinite JSON constant: {value}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except SalvageError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise SalvageError(f"{label} is malformed JSON") from error
    _require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(_read_bytes(path), str(path))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SalvageError("record is not strict JSON") from error
    return (text + "\n").encode("utf-8")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = _read_bytes(path).decode("utf-8")
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            _require(reader.fieldnames is not None, f"CSV has no header: {path}")
            header = list(reader.fieldnames)
            _require(
                header and len(header) == len(set(header)) and all(header),
                f"CSV header is invalid: {path}",
            )
            rows: list[dict[str, str]] = []
            for index, row in enumerate(reader):
                _require(
                    None not in row, f"CSV has extra fields at row {index}: {path}"
                )
                _require(
                    set(row) == set(header)
                    and all(isinstance(row[name], str) for name in header),
                    f"CSV row schema mismatch at row {index}: {path}",
                )
                rows.append(dict(row))
    except (UnicodeDecodeError, csv.Error) as error:
        raise SalvageError(f"malformed CSV: {path}") from error
    return header, rows


def _source_normalized_bytes(data: bytes) -> tuple[bytes, int]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SalvageError("source is not UTF-8") from error
    normalized: list[str] = []
    masked = 0
    for line in text.splitlines(keepends=True):
        if line.startswith(SELF_FREEZE_PREFIX):
            ending = "\n" if line.endswith("\n") else ""
            normalized.append(f'{SELF_FREEZE_PREFIX}"{SELF_FREEZE_MASK}"{ending}')
            masked += 1
        else:
            normalized.append(line)
    _require(masked <= 1, f"source normalization masked {masked} lines")
    return "".join(normalized).encode("utf-8"), masked


def _normalized_source_sha256(path: Path, *, require_marker: bool = False) -> str:
    normalized, masked = _source_normalized_bytes(_read_bytes(path))
    if require_marker:
        _require(masked == 1, "normalized self-hash marker count is not one")
    return hashlib.sha256(normalized).hexdigest()


def _verify_self_freeze(path: Path) -> dict[str, str]:
    path = _require_regular_file(path)
    normalized = _normalized_source_sha256(path, require_marker=True)
    _require(
        _is_sha256(EXPECTED_NORMALIZED_SOURCE_SHA256),
        "publisher normalized source hash is not frozen",
    )
    _require(
        normalized == EXPECTED_NORMALIZED_SOURCE_SHA256,
        "publisher normalized source hash mismatch",
    )
    return {"raw_sha256": _sha256_file(path), "normalized_sha256": normalized}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    _require(isinstance(tensor, torch.Tensor), "value is not a tensor")
    _require(tensor.device.type == "cpu", "tensor is not on CPU")
    _require(tensor.requires_grad is False, "tensor unexpectedly tracks derivatives")
    value = tensor.contiguous().clone()
    byte_count = value.numel() * value.element_size()
    _require(
        value.untyped_storage().nbytes() == byte_count,
        "tensor storage size mismatch",
    )
    raw = b"" if byte_count == 0 else ctypes.string_at(value.data_ptr(), byte_count)
    _require(
        len(raw) == byte_count,
        "tensor storage size mismatch",
    )
    return raw


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    _require(bool(torch.isfinite(tensor).all()), "tensor contains nonfinite values")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "sha256": _sha256_tensor(tensor),
    }


def _named_tensor_hash(
    mapping: Mapping[str, Any], *, label: str = "active-state"
) -> tuple[str, int]:
    _require(isinstance(mapping, Mapping), "active state is not a mapping")
    digest = hashlib.sha256()
    count = 0
    names = sorted(mapping)
    for index, name in enumerate(names, start=1):
        _require(isinstance(name, str), "active tensor name is not text")
        tensor = mapping[name]
        _require(
            isinstance(tensor, torch.Tensor), f"active value is not tensor: {name}"
        )
        _require(tensor.device.type == "cpu", f"active tensor is not CPU: {name}")
        _require(
            bool(torch.isfinite(tensor).all()), f"active tensor is nonfinite: {name}"
        )
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
        count += tensor.numel()
        if index == 1 or index % 8 == 0 or index == len(names):
            _log(
                "tensor-hash-progress",
                f"label={label} tensors={index}/{len(names)} parameters={count}",
            )
    return digest.hexdigest(), count


def _load_checkpoint(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    descriptor, before = _open_readonly(path)
    try:
        if expected_sha256 is not None:
            _require(_is_sha256(expected_sha256), "checkpoint hash pin is malformed")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            _require(
                digest.hexdigest() == expected_sha256,
                f"checkpoint hash mismatch: {path}",
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = torch.load(handle, map_location="cpu", weights_only=True)
            after = os.fstat(handle.fileno())
    except Exception as error:
        raise SalvageError(f"checkpoint load failed: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"checkpoint changed while loading: {path}",
    )
    _require(isinstance(payload, dict), f"checkpoint is not a mapping: {path}")
    _require_cuda_uninitialized()
    return payload


def _tree_line(spec: FileSpec) -> bytes:
    return (
        json.dumps(
            {
                "name": spec.name,
                "sha256": spec.sha256,
                "size_bytes": spec.size_bytes,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _tree_hash(specs: Sequence[FileSpec]) -> str:
    digest = hashlib.sha256()
    for spec in sorted(specs, key=lambda item: os.fsencode(item.name)):
        digest.update(_tree_line(spec))
    return digest.hexdigest()


def _audit_tree(
    path: Path,
    specs: Sequence[FileSpec],
    *,
    expected_tree_sha256: str | None,
    expected_total_size: int | None,
    label: str,
) -> dict[str, Any]:
    path = _require_directory(path)
    expected = {spec.name: spec for spec in specs}
    _require(len(expected) == len(specs), f"duplicate expected name in {label}")
    with os.scandir(path) as iterator:
        entries = list(iterator)
    observed_names = {entry.name for entry in entries}
    _require(observed_names == set(expected), f"{label} file set mismatch")
    observed_specs: list[FileSpec] = []
    files: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: os.fsencode(item.name)):
        child = path / entry.name
        _require_regular_file(child, single_link=True)
        metadata = os.lstat(child)
        spec = expected[entry.name]
        _require(
            metadata.st_size == spec.size_bytes, f"{label} size mismatch: {entry.name}"
        )
        observed_hash = _sha256_file(child)
        _require(observed_hash == spec.sha256, f"{label} hash mismatch: {entry.name}")
        observed = FileSpec(entry.name, metadata.st_size, observed_hash)
        observed_specs.append(observed)
        files.append(
            {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "name": entry.name,
                "nlink": metadata.st_nlink,
                "sha256": observed_hash,
                "size_bytes": metadata.st_size,
            }
        )
    total = sum(spec.size_bytes for spec in observed_specs)
    tree = _tree_hash(observed_specs)
    if expected_tree_sha256 is not None:
        _require(tree == expected_tree_sha256, f"{label} tree hash mismatch")
    if expected_total_size is not None:
        _require(total == expected_total_size, f"{label} total size mismatch")
    return {
        "entry_count": len(observed_specs),
        "files": files,
        "path": str(path),
        "regular_file_count": len(observed_specs),
        "total_size_bytes": total,
        "tree_sha256": tree,
    }


def _specs_from_hashes(
    directory: Path, hashes: Mapping[str, Any]
) -> tuple[FileSpec, ...]:
    specs: list[FileSpec] = []
    for name in sorted(hashes, key=os.fsencode):
        _require(
            isinstance(name, str)
            and Path(name).name == name
            and name not in {"", ".", ".."},
            f"invalid manifest filename: {name!r}",
        )
        digest = hashes[name]
        _require(_is_sha256(digest), f"invalid manifest hash: {name}")
        path = _require_regular_file(directory / name)
        specs.append(FileSpec(name, os.lstat(path).st_size, digest))
    return tuple(specs)


def _validate_failed_manifest_payload(payload: dict[str, Any]) -> FailedManifest:
    _require(
        set(payload)
        == {
            "schema_version",
            "manifest_id",
            "frozen_at_utc",
            "path_basis",
            "immutable_failed_staging",
            "salvage_projection",
            "expected_failure",
            "critical_scientific_audit",
            "preserved_lineage",
        },
        "failed manifest top-level schema mismatch",
    )
    _require(payload["schema_version"] == 1, "failed manifest version mismatch")
    _require(
        payload["manifest_id"]
        == "one_state_exact_selected_relaxed_cancellation_no_replay_salvage_failed_tree_v1",
        "failed manifest id mismatch",
    )
    _validate_utc(payload["frozen_at_utc"], "failed manifest timestamp")
    _require(
        payload["path_basis"] == "repository_relative_to_/home/coder/project",
        "failed manifest path basis mismatch",
    )
    failed = payload["immutable_failed_staging"]
    _require(
        isinstance(failed, dict)
        and set(failed)
        == {
            "path",
            "required_type",
            "mutation_policy",
            "expected_entry_count",
            "expected_regular_file_count",
            "expected_non_regular_entry_count",
            "total_size_bytes",
            "tree_hash",
            "files",
        },
        "failed tree manifest schema mismatch",
    )
    _require(failed["path"] == FAILED_REL.as_posix(), "failed path pin mismatch")
    _require(
        failed["required_type"] == "directory_with_only_regular_non_symlink_files"
        and failed["mutation_policy"] == "never_mutate"
        and failed["expected_entry_count"] == 34
        and failed["expected_regular_file_count"] == 34
        and failed["expected_non_regular_entry_count"] == 0
        and failed["total_size_bytes"] == FAILED_TOTAL_SIZE,
        "failed tree identity mismatch",
    )
    tree_hash = failed["tree_hash"]
    _require(
        isinstance(tree_hash, dict)
        and set(tree_hash) == {"algorithm", "serialization", "sha256"}
        and tree_hash["algorithm"] == "sha256"
        and tree_hash["sha256"] == FAILED_TREE_SHA256,
        "failed tree hash record mismatch",
    )
    files = failed["files"]
    _require(isinstance(files, list) and len(files) == 34, "failed file list mismatch")
    failed_specs: list[FileSpec] = []
    names: list[str] = []
    for item in files:
        _require(
            isinstance(item, dict)
            and set(item) == {"name", "path", "size_bytes", "sha256"},
            "failed file schema mismatch",
        )
        name = item["name"]
        _require(
            isinstance(name, str) and Path(name).name == name,
            "failed file name is invalid",
        )
        _require(
            item["path"] == (FAILED_REL / name).as_posix(),
            f"failed file path mismatch: {name}",
        )
        _require(
            _is_int(item["size_bytes"])
            and item["size_bytes"] >= 0
            and _is_sha256(item["sha256"]),
            f"failed file metadata invalid: {name}",
        )
        names.append(name)
        failed_specs.append(FileSpec(name, item["size_bytes"], item["sha256"]))
    _require(len(set(names)) == len(names), "failed file uniqueness mismatch")
    _require(
        sum(spec.size_bytes for spec in failed_specs) == FAILED_TOTAL_SIZE
        and _tree_hash(failed_specs) == FAILED_TREE_SHA256,
        "failed manifest file/tree closure mismatch",
    )
    projection = payload["salvage_projection"]
    _require(
        isinstance(projection, dict)
        and set(projection)
        == {
            "copy_source",
            "copy_rule",
            "excluded_failure_file",
            "staged_file_count_with_copied_incomplete",
            "staged_total_size_bytes",
            "staged_tree_sha256",
            "remove_after_all_staged_validation",
            "published_file_count",
            "published_total_size_bytes",
            "published_tree_sha256",
            "publication_target",
            "rewrite_policy",
        },
        "salvage projection schema mismatch",
    )
    _require(
        projection
        == {
            "copy_source": "immutable_failed_staging",
            "copy_rule": "copy_every_failed_tree_file_except_failure.json_byte_for_byte",
            "excluded_failure_file": "failure.json",
            "staged_file_count_with_copied_incomplete": 33,
            "staged_total_size_bytes": STAGED_TOTAL_SIZE,
            "staged_tree_sha256": STAGED_TREE_SHA256,
            "remove_after_all_staged_validation": "INCOMPLETE",
            "published_file_count": 32,
            "published_total_size_bytes": PUBLISHED_TOTAL_SIZE,
            "published_tree_sha256": PUBLISHED_TREE_SHA256,
            "publication_target": TARGET_REL.as_posix(),
            "rewrite_policy": "no_packet_file_may_be_rewritten_or_regenerated",
        },
        "salvage projection values mismatch",
    )
    expected_failure = payload["expected_failure"]
    _require(
        isinstance(expected_failure, dict)
        and set(expected_failure)
        == {
            "path",
            "sha256",
            "status",
            "error_type",
            "error",
            "mismatch_rank",
            "failure_scope",
        }
        and expected_failure["path"] == (FAILED_REL / "failure.json").as_posix()
        and expected_failure["sha256"]
        == "606a717bf73aac351595ab0b8f0f9ba47c5cb6a1a13fda691d2cc5ed1753c2b7"
        and expected_failure["status"] == "failed_recovery_staging"
        and expected_failure["error_type"] == "RecoveryError"
        and expected_failure["error"]
        == "replayed final spectrum CSV mismatch at rank 2"
        and expected_failure["mismatch_rank"] == 2
        and expected_failure["failure_scope"]
        == "exact_comparison_publication_validation_only",
        "expected failure record mismatch",
    )
    critical = payload["critical_scientific_audit"]
    _require(
        isinstance(critical, dict)
        and set(critical)
        == {
            "recovery_valid",
            "geometry_artifact",
            "authoritative_spectrum_chain",
            "packet_integrity",
            "scientific_outcome",
        }
        and critical["recovery_valid"] is True,
        "critical scientific audit schema mismatch",
    )
    geometry = critical["geometry_artifact"]
    _require(
        isinstance(geometry, dict)
        and set(geometry)
        == {
            "path",
            "sha256",
            "geometry_evaluation_count",
            "additional_geometry_evaluations_allowed",
            "optimization_gradient_evaluations",
            "parameter_updates",
            "proposals",
            "line_searches",
            "metric_replay_max_absolute_error",
            "spectrum_replay_max_absolute_error",
        }
        and geometry["path"] == (FAILED_REL / "replayed_final_geometry.pt").as_posix()
        and geometry["sha256"] == GEOMETRY_ARTIFACT_SHA256
        and geometry["geometry_evaluation_count"] == 1
        and geometry["additional_geometry_evaluations_allowed"] == 0
        and all(
            geometry[name] == 0
            for name in (
                "optimization_gradient_evaluations",
                "parameter_updates",
                "proposals",
                "line_searches",
            )
        )
        and geometry["metric_replay_max_absolute_error"] == 0.0
        and geometry["spectrum_replay_max_absolute_error"] == 0.0,
        "critical geometry record mismatch",
    )
    spectrum = critical["authoritative_spectrum_chain"]
    _require(
        isinstance(spectrum, dict)
        and set(spectrum)
        == {
            "parser",
            "eigenvalue_count",
            "representations",
            "all_representations_pairwise_bitwise_equal_binary64",
            "max_absolute_error",
            "geometry_eig_tensor_sha256",
            "pandas_is_authoritative",
            "pandas_descriptive_delta",
        }
        and spectrum["parser"]
        == "Python csv.DictReader plus correctly-rounded binary64 conversion"
        and spectrum["eigenvalue_count"] == DIMENSION
        and spectrum["all_representations_pairwise_bitwise_equal_binary64"] is True
        and spectrum["max_absolute_error"] == 0.0
        and spectrum["geometry_eig_tensor_sha256"] == GEOMETRY_EIG_SHA256
        and spectrum["pandas_is_authoritative"] is False,
        "critical spectrum record mismatch",
    )
    parser_delta = spectrum["pandas_descriptive_delta"]
    _require(
        isinstance(parser_delta, dict)
        and parser_delta
        == {
            "pandas_version_observed": "3.0.0",
            "changed_eigenvalue_count": 282,
            "max_absolute_delta": 2.220446049250313e-16,
            "max_ulp_delta": 5341,
            "first_changed_rank": 2,
            "caused_failed_exact_comparison": True,
        },
        "rank-2 parser diagnostic changed",
    )
    packet_integrity = critical["packet_integrity"]
    _require(
        isinstance(packet_integrity, dict)
        and set(packet_integrity)
        == {
            "artifact_manifest_link_count",
            "all_artifact_manifest_links_pass",
            "finalized_file_cross_hash_count",
            "all_finalized_file_cross_hashes_pass",
            "artifact_manifest_metadata_cross_hash_count",
            "all_artifact_manifest_metadata_cross_hashes_pass",
            "csv_row_counts",
        }
        and packet_integrity["artifact_manifest_link_count"] == 29
        and packet_integrity["all_artifact_manifest_links_pass"] is True
        and packet_integrity["finalized_file_cross_hash_count"] == 8
        and packet_integrity["all_finalized_file_cross_hashes_pass"] is True
        and packet_integrity["artifact_manifest_metadata_cross_hash_count"] == 5
        and packet_integrity["all_artifact_manifest_metadata_cross_hashes_pass"] is True
        and packet_integrity["csv_row_counts"]
        == {
            "arm_selection.csv": 2,
            "historical_transition_audit.csv": 100,
            "line_search.csv": 766,
            "proposal_diagnostics.csv": 100,
            "replayed_final_spectrum.csv": 512,
            "state_metrics.csv": 101,
            "state_spectra.csv": 51_712,
        },
        "packet integrity record mismatch",
    )
    outcome = critical["scientific_outcome"]
    _require(
        isinstance(outcome, dict)
        and set(outcome)
        == {
            "scientific_success",
            "failed_success_gates",
            "final_exact_a_per_dim",
            "continuation_non_top_only_fraction",
            "total_non_top_only_fraction",
            "accepted_updates",
            "new_accepted_updates",
            "termination",
        }
        and outcome["scientific_success"] is False
        and outcome["failed_success_gates"] == EXPECTED_FAILED_SUCCESS_GATES
        and outcome["final_exact_a_per_dim"] == 0.9321672207645146
        and outcome["continuation_non_top_only_fraction"] == 0.11301885339451084
        and outcome["total_non_top_only_fraction"] == 0.07733408144475684
        and outcome["accepted_updates"] == MAX_ACCEPTED_UPDATES
        and outcome["new_accepted_updates"] == NEW_ACCEPTED_UPDATES
        and outcome["termination"] == "max_updates_reached",
        "scientific outcome manifest record mismatch",
    )
    lineage = payload["preserved_lineage"]
    _require(
        isinstance(lineage, dict)
        and set(lineage) == {"original_source_staging", "first_startup_failed_staging"},
        "preserved lineage schema mismatch",
    )
    original = lineage["original_source_staging"]
    _require(
        isinstance(original, dict)
        and set(original)
        == {
            "path",
            "expected_file_count",
            "frozen_manifest_path",
            "frozen_manifest_sha256",
            "mutation_policy",
        }
        and original["path"] == ORIGINAL_SOURCE_REL.as_posix()
        and original["expected_file_count"] == 14
        and original["frozen_manifest_path"] == FROZEN_INPUT_MANIFEST_REL.as_posix()
        and original["frozen_manifest_sha256"] == FROZEN_INPUT_MANIFEST_SHA256
        and original["mutation_policy"] == "never_mutate",
        "original source lineage mismatch",
    )
    startup = lineage["first_startup_failed_staging"]
    _require(
        isinstance(startup, dict)
        and set(startup)
        == {
            "path",
            "expected_file_count",
            "total_size_bytes",
            "tree_sha256",
            "files",
            "failure_json_sha256",
            "geometry_evaluations",
            "model_loaded",
            "mutation_policy",
        }
        and startup["path"] == STARTUP_FAILED_REL.as_posix()
        and startup["expected_file_count"] == 3
        and startup["total_size_bytes"] == STARTUP_TOTAL_SIZE
        and startup["tree_sha256"] == STARTUP_TREE_SHA256
        and startup["failure_json_sha256"]
        == "da4c42c3b550b3d764117e967713987e72c712181ee2d65483ce2ae95da1ad9e"
        and startup["geometry_evaluations"] == 0
        and startup["model_loaded"] is False
        and startup["mutation_policy"] == "never_mutate",
        "startup failure lineage mismatch",
    )
    startup_files = startup["files"]
    _require(
        isinstance(startup_files, list) and len(startup_files) == 3,
        "startup files missing",
    )
    startup_specs: list[FileSpec] = []
    for item in startup_files:
        _require(
            isinstance(item, dict)
            and set(item) == {"name", "size_bytes", "sha256"}
            and isinstance(item["name"], str)
            and Path(item["name"]).name == item["name"]
            and _is_int(item["size_bytes"])
            and _is_sha256(item["sha256"]),
            "startup file record mismatch",
        )
        startup_specs.append(FileSpec(item["name"], item["size_bytes"], item["sha256"]))
    _require(
        _tree_hash(startup_specs) == STARTUP_TREE_SHA256
        and sum(item.size_bytes for item in startup_specs) == STARTUP_TOTAL_SIZE,
        "startup tree closure mismatch",
    )
    staged_specs = tuple(spec for spec in failed_specs if spec.name != "failure.json")
    published_specs = tuple(spec for spec in staged_specs if spec.name != "INCOMPLETE")
    _require(
        len(staged_specs) == 33
        and _tree_hash(staged_specs) == STAGED_TREE_SHA256
        and sum(spec.size_bytes for spec in staged_specs) == STAGED_TOTAL_SIZE
        and len(published_specs) == 32
        and _tree_hash(published_specs) == PUBLISHED_TREE_SHA256
        and sum(spec.size_bytes for spec in published_specs) == PUBLISHED_TOTAL_SIZE,
        "salvage projection closure mismatch",
    )
    return FailedManifest(
        payload=payload,
        failed_specs=tuple(failed_specs),
        staged_specs=staged_specs,
        published_specs=published_specs,
        startup_specs=tuple(startup_specs),
    )


def _load_failed_manifest(layout: Layout) -> FailedManifest:
    protocol = _require_regular_file(layout.path(PROTOCOL_REL))
    manifest = _require_regular_file(layout.path(FAILED_MANIFEST_REL))
    derivation = _require_regular_file(layout.path(DERIVATION_REL))
    _require(_sha256_file(protocol) == PROTOCOL_SHA256, "salvage protocol changed")
    _require(
        _sha256_file(manifest) == FAILED_MANIFEST_SHA256,
        "failed-tree manifest changed",
    )
    _require(_sha256_file(derivation) == DERIVATION_SHA256, "derivation changed")
    return _validate_failed_manifest_payload(_load_json(manifest))


def _load_original_manifest(
    layout: Layout,
) -> tuple[dict[str, Any], tuple[FileSpec, ...]]:
    path = _require_regular_file(layout.path(FROZEN_INPUT_MANIFEST_REL))
    _require(
        _sha256_file(path) == FROZEN_INPUT_MANIFEST_SHA256,
        "original frozen-input manifest changed",
    )
    payload = _load_json(path)
    _require(
        set(payload)
        == {
            "protocol_id",
            "source_staging_relative_path",
            "expected_exact_file_set",
            "files_sha256",
            "expected_progress",
            "expected_original_audit",
            "recovery_dependencies",
            "recovery_radius_rule",
        },
        "original frozen-input manifest schema mismatch",
    )
    _require(
        payload["protocol_id"] == RECOVERY_PROTOCOL_ID,
        "input manifest protocol mismatch",
    )
    _require(
        payload["source_staging_relative_path"] == ORIGINAL_SOURCE_REL.as_posix(),
        "input manifest source path mismatch",
    )
    names = payload["expected_exact_file_set"]
    hashes = payload["files_sha256"]
    _require(
        isinstance(names, list)
        and len(names) == 14
        and len(set(names)) == 14
        and isinstance(hashes, dict)
        and set(names) == set(hashes),
        "input manifest file set mismatch",
    )
    specs = _specs_from_hashes(layout.original_source, hashes)
    return payload, specs


def _audit_protected_trees(
    layout: Layout, manifest: FailedManifest
) -> dict[str, dict[str, Any]]:
    _, original_specs = _load_original_manifest(layout)
    return {
        "failed": _audit_tree(
            layout.failed,
            manifest.failed_specs,
            expected_tree_sha256=FAILED_TREE_SHA256,
            expected_total_size=FAILED_TOTAL_SIZE,
            label="failed tree",
        ),
        "original_source": _audit_tree(
            layout.original_source,
            original_specs,
            expected_tree_sha256=None,
            expected_total_size=None,
            label="original source tree",
        ),
        "startup_failure": _audit_tree(
            layout.startup_failed,
            manifest.startup_specs,
            expected_tree_sha256=STARTUP_TREE_SHA256,
            expected_total_size=STARTUP_TOTAL_SIZE,
            label="startup failure tree",
        ),
    }


def _validate_execution_freeze(
    layout: Layout, *, publisher_source: Path
) -> dict[str, Any]:
    path = _require_regular_file(layout.freeze, immutable=True, single_link=True)
    payload = _load_json(path)
    _require(set(payload) == FREEZE_KEYS, "execution freeze schema mismatch")
    _require(
        payload["schema_version"] == 1
        and payload["protocol_id"] == PROTOCOL_ID
        and payload["repository_root"] == str(layout.root),
        "execution freeze identity mismatch",
    )
    _validate_utc(payload["frozen_at_utc"], "execution freeze timestamp")
    _require(
        payload["protocol_sha256"] == PROTOCOL_SHA256
        and payload["failed_manifest_sha256"] == FAILED_MANIFEST_SHA256
        and payload["derivation_sha256"] == DERIVATION_SHA256,
        "execution freeze document pins mismatch",
    )
    _require(
        _sha256_file(layout.path(PROTOCOL_REL)) == PROTOCOL_SHA256
        and _sha256_file(layout.path(FAILED_MANIFEST_REL)) == FAILED_MANIFEST_SHA256
        and _sha256_file(layout.path(DERIVATION_REL)) == DERIVATION_SHA256,
        "frozen live documents changed",
    )
    source_roles = {
        "publisher": PUBLISHER_SOURCE_REL,
        "reviewer": REVIEWER_SOURCE_REL,
    }
    source_audits: dict[str, dict[str, str]] = {}
    for role, relative in source_roles.items():
        pin = payload[role]
        source = _require_regular_file(layout.path(relative))
        _require(
            isinstance(pin, dict)
            and set(pin) == {"path", "raw_sha256", "normalized_sha256"}
            and pin["path"] == str(source)
            and _is_sha256(pin["raw_sha256"])
            and _is_sha256(pin["normalized_sha256"])
            and _sha256_file(source) == pin["raw_sha256"]
            and _normalized_source_sha256(source, require_marker=True)
            == pin["normalized_sha256"],
            f"execution freeze source mismatch: {role}",
        )
        source_audits[role] = dict(pin)
    test_roles = {
        "publisher_test": PUBLISHER_TEST_REL,
        "reviewer_test": REVIEWER_TEST_REL,
    }
    test_audits: dict[str, dict[str, str]] = {}
    for role, relative in test_roles.items():
        pin = payload[role]
        test_path = _require_regular_file(layout.path(relative))
        _require(
            isinstance(pin, dict)
            and set(pin) == {"path", "sha256"}
            and pin["path"] == str(test_path)
            and _is_sha256(pin["sha256"])
            and _sha256_file(test_path) == pin["sha256"],
            f"execution freeze test mismatch: {role}",
        )
        test_audits[role] = dict(pin)
    _require(
        _absolute(publisher_source) == layout.path(PUBLISHER_SOURCE_REL),
        "executed publisher path differs from frozen publisher path",
    )
    self_freeze = _verify_self_freeze(publisher_source)
    _require(
        self_freeze["raw_sha256"] == payload["publisher"]["raw_sha256"]
        and self_freeze["normalized_sha256"]
        == payload["publisher"]["normalized_sha256"],
        "publisher self-freeze differs from execution freeze",
    )
    _require(
        payload["runtime"] == EXPECTED_RUNTIME
        and {"python": platform.python_version(), "torch": torch.__version__}
        == EXPECTED_RUNTIME,
        "execution freeze runtime mismatch",
    )
    _require(
        payload["import_policy"]
        == {
            "allowed_top_level_imports": ALLOWED_TOP_LEVEL_IMPORTS,
            "forbidden_tokens": FORBIDDEN_TOKENS,
        },
        "execution import policy mismatch",
    )
    expected_paths = {
        "failed_source": str(layout.failed),
        "stage": str(layout.stage),
        "target": str(layout.target),
        "publisher_snapshot": str(layout.snapshot),
        "execution_record": str(layout.record),
        "review_record_temp": str(layout.path(INDEPENDENT_REVIEW_TEMP_REL)),
        "review_record": str(layout.path(INDEPENDENT_REVIEW_REL)),
    }
    _require(payload["paths"] == expected_paths, "execution freeze path map mismatch")
    _require(
        payload["prereview"]
        == {"provenance_decision": "GO", "scientific_decision": "GO"},
        "execution freeze prereview decisions are not GO",
    )
    return {
        "payload": payload,
        "sha256": _sha256_file(path),
        "self_freeze": self_freeze,
        "sources": source_audits,
        "tests": test_audits,
    }


def _expected_header(records: Sequence[Mapping[str, Any]]) -> list[str]:
    header: list[str] = []
    seen: set[str] = set()
    for record in records:
        _require(isinstance(record, Mapping), "checkpoint row is not a mapping")
        for name in record:
            _require(isinstance(name, str), "checkpoint column is not text")
            if name not in seen:
                seen.add(name)
                header.append(name)
    return header


def _csv_cell_matches(expected: Any, observed: str) -> bool:
    if expected is None:
        return observed == ""
    if isinstance(expected, bool):
        return observed == str(expected)
    if isinstance(expected, int):
        try:
            return _as_csv_int(observed, "CSV integer") == expected
        except SalvageError:
            return False
    if isinstance(expected, float):
        return _bitwise_float_equal(expected, observed)
    if isinstance(expected, str):
        return observed == expected
    return False


def _audit_checkpoint_csv(
    records: Sequence[Mapping[str, Any]], path: Path, *, label: str
) -> dict[str, Any]:
    header, rows = _read_csv(path)
    expected_header = _expected_header(records)
    _require(header == expected_header, f"{label} CSV header mismatch")
    _require(len(rows) == len(records), f"{label} CSV row count mismatch")
    checked = 0
    for row_index, (expected, observed) in enumerate(zip(records, rows, strict=True)):
        for name in header:
            _require(
                _csv_cell_matches(expected.get(name), observed[name]),
                f"{label} CSV/checkpoint mismatch at row {row_index}, column {name}",
            )
            checked += 1
    return {
        "column_count": len(header),
        "field_comparisons": checked,
        "row_count": len(rows),
        "sha256": _sha256_file(path),
    }


def _validate_failure_and_log(packet: Path) -> dict[str, Any]:
    failure = _load_json(packet / "failure.json")
    _require(
        set(failure) == {"error", "error_type", "protocol_id", "status", "traceback"},
        "failure.json schema mismatch",
    )
    _require(
        failure["error"] == "replayed final spectrum CSV mismatch at rank 2"
        and failure["error_type"] == "RecoveryError"
        and failure["protocol_id"] == RECOVERY_PROTOCOL_ID
        and failure["status"] == "failed_recovery_staging",
        "failure.json identity mismatch",
    )
    traceback_text = failure["traceback"]
    _require(
        isinstance(traceback_text, str)
        and "_audit_replayed_final_spectrum_file" in traceback_text
        and "_validate_staged_publication" in traceback_text
        and traceback_text.endswith(
            "RecoveryError: replayed final spectrum CSV mismatch at rank 2\n"
        ),
        "failure traceback mismatch",
    )
    try:
        log = _read_bytes(packet / "run.log").decode("utf-8")
    except UnicodeDecodeError as error:
        raise SalvageError("run.log is not UTF-8") from error
    expected_fragments = (
        "stage=startup protocol=one_state_exact_selected_relaxed_cancellation_recovery_v1",
        "stage=load-terminal-progress",
        "stage=original-read-only-audit",
        "stage=independent-recovery-audit",
        "stage=terminal-checkpoint-lineage",
        "stage=single-final-geometry-replay",
        "geometry_replay_pass=1 evaluations=1 metric_max=0 spectrum_max=0",
    )
    for fragment in expected_fragments:
        _require(fragment in log, f"run.log is missing: {fragment}")
    forbidden_fragments = (
        "publication complete",
        "published output",
        "stage=published",
        "stage=complete",
    )
    _require(
        not any(fragment in log.lower() for fragment in forbidden_fragments),
        "run.log falsely records successful publication",
    )
    _require(log.count("geometry_replay_pass=1") == 1, "run.log replay count mismatch")
    return {
        "failure_error": failure["error"],
        "mismatch_rank": 2,
        "replay_success_lines": 1,
        "successful_publication_lines": 0,
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
    "recovery_spectra.png",
    "recovery_task_manifest_snapshot.json",
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
EXPECTED_PUBLISHED_NAMES = EXPECTED_MANIFESTED_ARTIFACTS | {
    "artifact_manifest.json",
    "FINALIZED.json",
    "run.log",
}
ARTIFACT_MANIFEST_KEYS = {
    "protocol_id",
    "recovery_finalizer_source_sha256",
    "recovery_finalizer_normalized_source_sha256",
    "frozen_input_manifest_sha256",
    "recovery_task_manifest_sha256",
    "recovery_import_manifest_sha256",
    "source_progress_checkpoint_sha256",
    "artifacts",
}
FINALIZED_KEYS = {
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
DECISION_KEYS = {
    "accepted_updates",
    "accepted_vae_checkpoint_sha256",
    "elapsed_sec",
    "final_checkpoint_sha256",
    "final_geometry_replay",
    "final_metrics",
    "new_accepted_updates",
    "original_runner_audit_pass",
    "original_runner_failed_gates",
    "outcome",
    "protocol_id",
    "recovery_audit_sha256",
    "recovery_import_manifest_sha256",
    "recovery_radius_audit",
    "recovery_task_manifest_sha256",
    "recovery_valid",
    "recovery_validity_gates",
    "scientific_success",
    "selected_arm",
    "source_protocol_id",
    "termination",
}
RECOVERY_AUDIT_KEYS = {
    "checkpoint_lineage",
    "copied_self_freeze",
    "dependency_import",
    "final_geometry_replay",
    "geometry_execution",
    "geometry_reconstruction",
    "immutable_input_postflight",
    "immutable_input_preflight",
    "import_manifest_postflight",
    "import_manifest_preflight",
    "independent_recovery_audit",
    "original_runner_audit",
    "original_runner_audit_pass",
    "original_runner_failed_gates",
    "protocol_id",
    "reconstruction_input_files",
    "recovery_valid",
    "recovery_validity_gates",
    "runtime_import_closure",
    "source_protocol_id",
    "startup_self_freeze",
    "task_manifest_postflight",
    "task_manifest_preflight",
}
RECOVERY_VALIDITY_GATES = {
    "accepted_checkpoint_exact",
    "all_model_tensors_unchanged_during_replay",
    "all_unmodified_original_row_gates_pass",
    "checkpoint_cpu_bitwise_lineage_pass",
    "copied_finalizer_self_freeze",
    "exactly_one_geometry_evaluation",
    "final_geometry_replay_pass",
    "frozen_dependencies_verified_before_import",
    "geometry_artifact_independent_closure_pass",
    "immutable_input_postflight",
    "immutable_input_preflight",
    "import_manifest_pre_post_exact",
    "independent_recovery_audit_pass",
    "loaded_producer_equals_executed_snapshot",
    "no_optimization_operations",
    "original_runner_expected_failure_observed",
    "reconstruction_inputs_exact_pre_post_load_run",
    "recovery_radius_audit_pass",
    "replayed_spectrum_csv_pass",
    "runtime_import_closure_exact",
    "runtime_task_tensors_exact_immediately_before_replay",
    "runtime_task_tensors_unchanged_after_replay",
    "runtime_versions_exact",
    "scientific_success_criteria_unchanged",
    "startup_finalizer_self_freeze",
    "task_manifest_pre_post_exact",
    "terminal_install_changes_exact_active_set",
    "terminal_progress_lineage",
}


def _validate_packet_hash_graph(packet: Path) -> dict[str, Any]:
    artifact_manifest = _load_json(packet / "artifact_manifest.json")
    _require(
        set(artifact_manifest) == ARTIFACT_MANIFEST_KEYS,
        "artifact manifest schema mismatch",
    )
    _require(
        artifact_manifest["protocol_id"] == RECOVERY_PROTOCOL_ID,
        "artifact manifest protocol mismatch",
    )
    artifacts = artifact_manifest["artifacts"]
    _require(
        isinstance(artifacts, dict) and set(artifacts) == EXPECTED_MANIFESTED_ARTIFACTS,
        "artifact manifest file set mismatch",
    )
    for name, expected in artifacts.items():
        _require(_is_sha256(expected), f"invalid artifact hash: {name}")
        _require(
            _sha256_file(packet / name) == expected, f"artifact hash mismatch: {name}"
        )
    expected_metadata = {
        "recovery_finalizer_source_sha256": FINALIZER_SOURCE_SHA256,
        "recovery_finalizer_normalized_source_sha256": FINALIZER_NORMALIZED_SHA256,
        "frozen_input_manifest_sha256": FROZEN_INPUT_MANIFEST_SHA256,
        "recovery_task_manifest_sha256": RECOVERY_TASK_MANIFEST_SHA256,
        "recovery_import_manifest_sha256": RECOVERY_IMPORT_MANIFEST_SHA256,
        "source_progress_checkpoint_sha256": SOURCE_PROGRESS_SHA256,
    }
    for name, expected in expected_metadata.items():
        _require(
            artifact_manifest[name] == expected, f"artifact metadata mismatch: {name}"
        )
    finalizer_snapshot = packet / "executed_recovery_finalizer_source_snapshot.py"
    _require(
        _sha256_file(finalizer_snapshot) == FINALIZER_SOURCE_SHA256
        and _normalized_source_sha256(finalizer_snapshot, require_marker=True)
        == FINALIZER_NORMALIZED_SHA256,
        "executed finalizer source snapshot mismatch",
    )
    finalized = _load_json(packet / "FINALIZED.json")
    _require(set(finalized) == FINALIZED_KEYS, "FINALIZED schema mismatch")
    _require(
        finalized["protocol_id"] == RECOVERY_PROTOCOL_ID
        and finalized["source_protocol_id"] == SOURCE_PROTOCOL_ID
        and finalized["status"] == "complete_awaiting_independent_recovery_review"
        and finalized["recovery_valid"] is True
        and finalized["scientific_success"] is False
        and finalized["accepted_updates"] == MAX_ACCEPTED_UPDATES
        and finalized["new_accepted_updates"] == NEW_ACCEPTED_UPDATES
        and finalized["termination"] == "max_updates_reached",
        "FINALIZED identity mismatch",
    )
    finalized_pins = {
        "source_progress_checkpoint_sha256": SOURCE_PROGRESS_SHA256,
        "accepted_vae_checkpoint_sha256": ACCEPTED_VAE_CHECKPOINT_SHA256,
        "recovery_task_manifest_sha256": RECOVERY_TASK_MANIFEST_SHA256,
        "recovery_import_manifest_sha256": RECOVERY_IMPORT_MANIFEST_SHA256,
        "recovery_finalizer_source_sha256": FINALIZER_SOURCE_SHA256,
        "recovery_finalizer_normalized_source_sha256": FINALIZER_NORMALIZED_SHA256,
        "decision_sha256": _sha256_file(packet / "decision.json"),
        "recovery_audit_sha256": _sha256_file(packet / "recovery_audit.json"),
        "artifact_manifest_sha256": _sha256_file(packet / "artifact_manifest.json"),
        "final_checkpoint_sha256": _sha256_file(packet / "final_checkpoint.pt"),
    }
    for name, expected in finalized_pins.items():
        _require(finalized[name] == expected, f"FINALIZED hash link mismatch: {name}")
    decision = _load_json(packet / "decision.json")
    _require(set(decision) == DECISION_KEYS, "decision schema mismatch")
    _require(
        decision["protocol_id"] == RECOVERY_PROTOCOL_ID
        and decision["source_protocol_id"] == SOURCE_PROTOCOL_ID
        and decision["recovery_valid"] is True
        and decision["scientific_success"] is False
        and decision["accepted_updates"] == MAX_ACCEPTED_UPDATES
        and decision["new_accepted_updates"] == NEW_ACCEPTED_UPDATES
        and decision["selected_arm"] == "low"
        and decision["termination"] == "max_updates_reached"
        and decision["original_runner_audit_pass"] is False
        and decision["original_runner_failed_gates"]
        == ["continuation_diagnostics_recompute"]
        and decision["final_checkpoint_sha256"] == FINAL_CHECKPOINT_SHA256
        and decision["accepted_vae_checkpoint_sha256"] == ACCEPTED_VAE_CHECKPOINT_SHA256
        and decision["recovery_task_manifest_sha256"] == RECOVERY_TASK_MANIFEST_SHA256
        and decision["recovery_import_manifest_sha256"]
        == RECOVERY_IMPORT_MANIFEST_SHA256
        and decision["recovery_audit_sha256"]
        == _sha256_file(packet / "recovery_audit.json")
        and _as_float(decision["elapsed_sec"], "decision elapsed_sec") >= 0.0,
        "decision identity mismatch",
    )
    recovery = _load_json(packet / "recovery_audit.json")
    _require(set(recovery) == RECOVERY_AUDIT_KEYS, "recovery audit schema mismatch")
    gates = recovery["recovery_validity_gates"]
    _require(
        recovery["protocol_id"] == RECOVERY_PROTOCOL_ID
        and recovery["source_protocol_id"] == SOURCE_PROTOCOL_ID
        and recovery["recovery_valid"] is True
        and recovery["original_runner_audit_pass"] is False
        and recovery["original_runner_failed_gates"]
        == ["continuation_diagnostics_recompute"]
        and isinstance(gates, dict)
        and set(gates) == RECOVERY_VALIDITY_GATES
        and all(value is True for value in gates.values()),
        "recovery validity gates mismatch",
    )
    _require(
        decision["recovery_validity_gates"] == gates
        and decision["final_geometry_replay"] == recovery["final_geometry_replay"]
        and decision["recovery_radius_audit"]
        == recovery["independent_recovery_audit"]["radius_audit"],
        "decision/recovery audit graph mismatch",
    )
    config = _load_json(packet / "resolved_config.json")
    _require(
        config.get("protocol_id") == RECOVERY_PROTOCOL_ID
        and config.get("source_protocol_id") == SOURCE_PROTOCOL_ID
        and config.get("source_progress_checkpoint_sha256") == SOURCE_PROGRESS_SHA256
        and config.get("frozen_input_manifest_sha256") == FROZEN_INPUT_MANIFEST_SHA256
        and config.get("recovery_protocol_sha256") == RECOVERY_PROTOCOL_SHA256
        and config.get("recovery_task_manifest_sha256") == RECOVERY_TASK_MANIFEST_SHA256
        and config.get("recovery_import_manifest_sha256")
        == RECOVERY_IMPORT_MANIFEST_SHA256
        and config.get("producer_source_sha256") == PRODUCER_SOURCE_SHA256
        and config.get("continuation_reviewer_source_sha256")
        == CONTINUATION_REVIEWER_SHA256
        and config.get("finalizer_source_sha256") == FINALIZER_SOURCE_SHA256
        and config.get("finalizer_normalized_source_sha256")
        == FINALIZER_NORMALIZED_SHA256
        and config.get("source_staging") == str(PRODUCTION_ROOT / ORIGINAL_SOURCE_REL)
        and config.get("output_dir") == str(PRODUCTION_ROOT / TARGET_REL)
        and config.get("source_staging_mutation_allowed") is False
        and config.get("expected_runtime") == EXPECTED_PACKET_RUNTIME,
        "resolved config frozen identity mismatch",
    )
    snapshot_hashes = config.get("snapshot_hashes")
    _require(isinstance(snapshot_hashes, dict), "resolved snapshot hash graph missing")
    for name, expected in snapshot_hashes.items():
        _require(
            name in EXPECTED_MANIFESTED_ARTIFACTS
            and _is_sha256(expected)
            and _sha256_file(packet / name) == expected,
            f"resolved snapshot hash mismatch: {name}",
        )
    return {
        "artifact_link_count": len(artifacts),
        "artifact_manifest": artifact_manifest,
        "config": config,
        "decision": decision,
        "finalized": finalized,
        "recovery_audit": recovery,
    }


def _validate_execution_budget(graph: Mapping[str, Any]) -> dict[str, int]:
    config = graph["config"]
    expected_budgets = {
        "geometry_evaluation_budget": 1,
        "optimization_gradient_evaluation_budget": 0,
        "proposal_budget": 0,
        "line_search_budget": 0,
        "parameter_update_budget": 0,
    }
    for name, expected in expected_budgets.items():
        _require(config.get(name) == expected, f"config budget mismatch: {name}")
    _require(
        config.get("metric_replay_absolute_tolerance") == REPLAY_ATOL
        and config.get("spectrum_replay_absolute_tolerance") == REPLAY_ATOL
        and config.get("direction_radius_absolute_tolerance") == 5e-9
        and config.get("direction_radius_relative_tolerance") == 0.0,
        "config tolerance mismatch",
    )
    execution = graph["recovery_audit"]["geometry_execution"]
    expected_execution = {
        "geometry_evaluation_count": 1,
        "forbidden_operation_attempts": 0,
        "optimization_gradient_evaluations": 0,
        "proposals": 0,
        "line_searches": 0,
        "parameter_updates": 0,
    }
    _require(execution == expected_execution, "geometry execution record mismatch")
    return expected_execution


def _validate_dependency_hash_manifest(
    layout: Layout, payload: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_text, expected in payload.items():
        _require(
            isinstance(relative_text, str) and _is_sha256(expected),
            f"{label} entry malformed",
        )
        relative = Path(relative_text)
        _relative_text(relative)
        path = _require_regular_file(layout.dependency_path(relative))
        digest = _sha256_file(path)
        _require(digest == expected, f"{label} dependency changed: {relative_text}")
        observed[relative_text] = digest
    return observed


def _validate_task_and_inputs(
    layout: Layout, packet: Path, graph: Mapping[str, Any]
) -> dict[str, Any]:
    live_task = layout.path(RECOVERY_TASK_MANIFEST_REL)
    live_imports = layout.path(RECOVERY_IMPORT_MANIFEST_REL)
    _require(
        _sha256_file(live_task) == RECOVERY_TASK_MANIFEST_SHA256,
        "task manifest changed",
    )
    _require(
        _sha256_file(live_imports) == RECOVERY_IMPORT_MANIFEST_SHA256,
        "import manifest changed",
    )
    _require(
        _read_bytes(live_task)
        == _read_bytes(packet / "recovery_task_manifest_snapshot.json")
        and _read_bytes(live_imports)
        == _read_bytes(packet / "recovery_import_manifest_snapshot.json"),
        "task/import snapshot byte mismatch",
    )
    task = _load_json(live_task)
    _require(
        set(task)
        == {
            "protocol_id",
            "source_weight_index",
            "task_name",
            "tau",
            "accepted_vae_checkpoint_sha256",
            "expected_runtime",
            "source_dependencies",
            "selected_task_tensors",
        }
        and task["protocol_id"] == RECOVERY_PROTOCOL_ID
        and task["source_weight_index"] == SOURCE_WEIGHT_INDEX
        and task["task_name"] == "fashion_mnist"
        and task["tau"] == 1.1614345407370807
        and task["accepted_vae_checkpoint_sha256"] == ACCEPTED_VAE_CHECKPOINT_SHA256
        and task["expected_runtime"] == EXPECTED_PACKET_RUNTIME
        and task["source_dependencies"] == EXPECTED_TASK_SOURCE_DEPENDENCIES
        and task["selected_task_tensors"] == EXPECTED_TASK_TENSORS,
        "task manifest identity mismatch",
    )
    for relative_text, expected in EXPECTED_TASK_SOURCE_DEPENDENCIES.items():
        _require(
            _sha256_file(layout.dependency_path(Path(relative_text))) == expected,
            f"task source dependency changed: {relative_text}",
        )
    accepted_run = layout.dependency_path(ACCEPTED_RUN_REL)
    _require_directory(accepted_run)
    accepted_hashes: dict[str, str] = {}
    for name, expected in EXPECTED_ACCEPTED_RUN_INPUT_SHA256.items():
        digest = _sha256_file(accepted_run / name)
        _require(digest == expected, f"accepted run input changed: {name}")
        accepted_hashes[name] = digest
    imports = _load_json(live_imports)
    _require(
        set(imports) == {"protocol_id", "entry_modules", "loaded_repository_modules"}
        and imports["protocol_id"] == RECOVERY_PROTOCOL_ID
        and imports["entry_modules"]
        == [
            "scripts.run_one_state_exact_selected_trajectory_continuation",
            "scripts.review_one_state_exact_selected_trajectory_continuation",
        ]
        and isinstance(imports["loaded_repository_modules"], dict)
        and len(imports["loaded_repository_modules"]) == 49,
        "import manifest schema mismatch",
    )
    imported_hashes: dict[str, str] = {}
    for module_name, pin in imports["loaded_repository_modules"].items():
        _require(
            isinstance(module_name, str)
            and isinstance(pin, dict)
            and set(pin) == {"relative_path", "sha256"}
            and _is_sha256(pin["sha256"]),
            f"import manifest pin malformed: {module_name}",
        )
        relative = Path(pin["relative_path"])
        _relative_text(relative)
        digest = _sha256_file(layout.dependency_path(relative))
        _require(digest == pin["sha256"], f"import source changed: {module_name}")
        imported_hashes[module_name] = digest
    producer_dependencies = _load_json(
        packet / "frozen_producer_dependency_manifest_snapshot.json"
    )
    _require(
        _sha256_file(packet / "frozen_producer_dependency_manifest_snapshot.json")
        == SOURCE_DEPENDENCY_MANIFEST_SHA256,
        "producer dependency snapshot hash mismatch",
    )
    producer_hashes = _validate_dependency_hash_manifest(
        layout, producer_dependencies, label="producer dependency"
    )
    _require(
        _sha256_file(layout.dependency_path(PARENT_FINAL_REL)) == PARENT_FINAL_SHA256
        and _sha256_file(layout.dependency_path(PARENT_PROGRESS_REL))
        == PARENT_PROGRESS_SHA256
        and _sha256_file(layout.dependency_path(PARENT_REVIEW_REL))
        == PARENT_REVIEW_SHA256,
        "parent checkpoint/review lineage changed",
    )
    parent_review = _load_json(layout.dependency_path(PARENT_REVIEW_REL))
    _require(
        parent_review.get("valid") is True
        and parent_review.get("failed_gates") == []
        and parent_review.get("errors") == [],
        "parent independent review is invalid",
    )
    recovery = graph["recovery_audit"]
    reconstruction_files = recovery["reconstruction_input_files"]
    _require(
        isinstance(reconstruction_files, dict)
        and set(reconstruction_files)
        == {"pass", "pre_post_equal", "pre_load_run", "post_load_run"}
        and reconstruction_files["pass"] is True
        and reconstruction_files["pre_post_equal"] is True
        and reconstruction_files["pre_load_run"]
        == reconstruction_files["post_load_run"],
        "reconstruction input pre/post audit mismatch",
    )
    recorded_run = reconstruction_files["pre_load_run"]
    expected_files = {
        name: {
            "path": str(PRODUCTION_ROOT / ACCEPTED_RUN_REL / name),
            "sha256": digest,
        }
        for name, digest in EXPECTED_ACCEPTED_RUN_INPUT_SHA256.items()
    }
    _require(
        recorded_run
        == {
            "pass": True,
            "run_dir": str(PRODUCTION_ROOT / ACCEPTED_RUN_REL),
            "file_count": 4,
            "files": expected_files,
        },
        "recorded reconstruction inputs mismatch",
    )
    return {
        "accepted_run_hashes": accepted_hashes,
        "import_module_count": len(imported_hashes),
        "producer_dependency_count": len(producer_hashes),
        "task_dependency_count": len(EXPECTED_TASK_SOURCE_DEPENDENCIES),
    }


def _validate_model_snapshot(snapshot: Mapping[str, Any], label: str) -> None:
    _require(
        isinstance(snapshot, Mapping)
        and set(snapshot)
        == {
            "parameters",
            "buffers",
            "parameter_tensor_count",
            "parameter_element_count",
            "buffer_tensor_count",
            "buffer_element_count",
            "all_tensor_count",
            "all_element_count",
            "aggregate_sha256",
        },
        f"{label} model snapshot schema mismatch",
    )
    digest = hashlib.sha256()
    total_tensors = 0
    total_elements = 0
    for category in ("parameters", "buffers"):
        entries = snapshot[category]
        _require(isinstance(entries, Mapping), f"{label} fingerprints missing")
        elements = 0
        for name in sorted(entries):
            fingerprint = entries[name]
            _require(
                isinstance(name, str)
                and isinstance(fingerprint, Mapping)
                and set(fingerprint) == {"dtype", "shape", "numel", "sha256"}
                and isinstance(fingerprint["dtype"], str)
                and isinstance(fingerprint["shape"], list)
                and all(_is_int(value) and value >= 0 for value in fingerprint["shape"])
                and fingerprint["numel"] == math.prod(fingerprint["shape"])
                and _is_sha256(fingerprint["sha256"]),
                f"{label} fingerprint invalid: {name}",
            )
            digest.update(category.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(json.dumps(fingerprint, sort_keys=True).encode("utf-8"))
            total_tensors += 1
            elements += fingerprint["numel"]
        singular = category[:-1]
        _require(
            snapshot[f"{singular}_tensor_count"] == len(entries)
            and snapshot[f"{singular}_element_count"] == elements,
            f"{label} model count closure mismatch",
        )
        total_elements += elements
    _require(
        snapshot["all_tensor_count"] == total_tensors
        and snapshot["all_element_count"] == total_elements
        and snapshot["aggregate_sha256"] == digest.hexdigest(),
        f"{label} aggregate model fingerprint mismatch",
    )


def _validate_task_reconstruction_record(record: Mapping[str, Any]) -> None:
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
        isinstance(record, Mapping)
        and set(record)
        == {
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
        and record["pass"] is True
        and isinstance(record["gates"], Mapping)
        and set(record["gates"]) == expected_gate_names
        and all(value is True for value in record["gates"].values())
        and record["task_name"] == "fashion_mnist"
        and record["source_weight_index"] == SOURCE_WEIGHT_INDEX
        and record["tau"] == 1.1614345407370807
        and record["train_sample_count"] == 16_384
        and record["test_sample_count"] == 4_096
        and record["selected_task_tensors"] == EXPECTED_TASK_TENSORS
        and record["accepted_vae_checkpoint_sha256"] == ACCEPTED_VAE_CHECKPOINT_SHA256,
        "runtime task reconstruction mismatch",
    )


def _validate_model_nonmutation(
    recovery: Mapping[str, Any], geometry_payload: Mapping[str, Any]
) -> dict[str, Any]:
    reconstruction = recovery["geometry_reconstruction"]
    expected_keys = {
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
    _require(
        isinstance(reconstruction, Mapping) and set(reconstruction) == expected_keys,
        "geometry reconstruction schema mismatch",
    )
    _require(
        reconstruction["accepted_checkpoint_sha256"]
        == reconstruction["expected_accepted_checkpoint_sha256"]
        == ACCEPTED_VAE_CHECKPOINT_SHA256
        and reconstruction["z_sha256"] == Z_SHA256
        and reconstruction["active_tensor_count"] == ACTIVE_TENSOR_COUNT
        and reconstruction["active_parameter_count"] == ACTIVE_PARAMETER_COUNT
        and reconstruction["installed_active_parameter_hash"]
        == reconstruction["post_replay_active_parameter_hash"]
        == ACTIVE_PARAMETER_HASH
        and reconstruction["base_parameter_gradient_slots"] == 0
        and reconstruction["post_replay_parameter_gradient_slots"] == 0
        and reconstruction["full_ce_batch"] is True,
        "geometry reconstruction counters/lineage mismatch",
    )
    snapshots = reconstruction["model_tensor_snapshots"]
    _require(
        isinstance(snapshots, Mapping)
        and set(snapshots) == {"base", "installed", "post_replay"},
        "model snapshot set mismatch",
    )
    for name in ("base", "installed", "post_replay"):
        _validate_model_snapshot(snapshots[name], name)
    base = snapshots["base"]
    installed = snapshots["installed"]
    post_replay = snapshots["post_replay"]
    _require(installed == post_replay, "model changed during stored replay")
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
    installation = reconstruction["model_installation_audit"]
    _require(
        installation.get("pass") is True
        and installation.get("expected_changed_parameter_count") == ACTIVE_TENSOR_COUNT
        and installation.get("changed_parameter_count")
        == len(changed_parameters)
        == ACTIVE_TENSOR_COUNT
        and installation.get("changed_parameters") == changed_parameters
        and installation.get("changed_buffer_count") == len(changed_buffers) == 0
        and installation.get("changed_buffers") == []
        and installation.get("base_aggregate_sha256") == base["aggregate_sha256"]
        and installation.get("installed_aggregate_sha256")
        == installed["aggregate_sha256"],
        "model installation audit mismatch",
    )
    nonmutation = reconstruction["model_replay_nonmutation_audit"]
    _require(
        nonmutation.get("pass") is True
        and nonmutation.get("all_parameters_bitwise_unchanged") is True
        and nonmutation.get("all_buffers_bitwise_unchanged") is True
        and nonmutation.get("installed_aggregate_sha256")
        == nonmutation.get("post_replay_aggregate_sha256")
        == installed["aggregate_sha256"]
        and geometry_payload["installed_model_aggregate_sha256"]
        == geometry_payload["post_replay_model_aggregate_sha256"]
        == installed["aggregate_sha256"],
        "model nonmutation aggregate mismatch",
    )
    for name in ("task_at_load", "task_immediately_pre_replay", "task_post_replay"):
        _validate_task_reconstruction_record(reconstruction[name])
    _require(
        reconstruction["task_at_load"]
        == reconstruction["task_immediately_pre_replay"]
        == reconstruction["task_post_replay"],
        "task tensors changed around replay",
    )
    runtime = reconstruction["runtime_provenance"]
    _require(
        runtime.get("pass") is True
        and runtime.get("expected_runtime")
        == runtime.get("observed_runtime")
        == EXPECTED_PACKET_RUNTIME
        and runtime.get("requested_device") == "cuda:0"
        and runtime.get("resolved_cuda_device_index") == 0,
        "stored replay runtime provenance mismatch",
    )
    return {
        "aggregate_sha256": installed["aggregate_sha256"],
        "changed_parameter_count": len(changed_parameters),
        "unchanged_during_replay": True,
    }


PROGRESS_KEYS = {
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
FINAL_CHECKPOINT_KEYS = {
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
GEOMETRY_KEYS = {
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


def _validate_checkpoint_lineage(
    layout: Layout, packet: Path, graph: Mapping[str, Any]
) -> dict[str, Any]:
    progress_path = packet / "source_progress_checkpoint.pt"
    final_path = packet / "final_checkpoint.pt"
    progress = _load_checkpoint(progress_path, expected_sha256=SOURCE_PROGRESS_SHA256)
    final = _load_checkpoint(final_path, expected_sha256=FINAL_CHECKPOINT_SHA256)
    _require(set(progress) == PROGRESS_KEYS, "source progress schema mismatch")
    _require(set(final) == FINAL_CHECKPOINT_KEYS, "final checkpoint schema mismatch")
    expected_progress = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "dependency_manifest_sha256": SOURCE_DEPENDENCY_MANIFEST_SHA256,
        "parent_checkpoint_sha256": PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": PARENT_PROGRESS_SHA256,
        "accepted_updates": MAX_ACCEPTED_UPDATES,
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "selected_arm": "low",
        "terminal": True,
        "termination": "max_updates_reached",
        "transition_chain_sha256": TRANSITION_CHAIN_SHA256,
        "active_parameter_hash": ACTIVE_PARAMETER_HASH,
    }
    for name, expected in expected_progress.items():
        _require(progress[name] == expected, f"source progress mismatch: {name}")
    row_counts = {
        "state_rows": 101,
        "spectrum_rows": 51_712,
        "proposal_rows": 100,
        "line_rows": 766,
        "selection_rows": 2,
    }
    for name, expected in row_counts.items():
        _require(
            isinstance(progress[name], list) and len(progress[name]) == expected,
            f"source progress row count mismatch: {name}",
        )
    active = progress["active_model_state"]
    active_hash, active_count = _named_tensor_hash(
        active, label="source-progress-active-state"
    )
    _require(
        len(active) == ACTIVE_TENSOR_COUNT
        and active_count == ACTIVE_PARAMETER_COUNT
        and active_hash == ACTIVE_PARAMETER_HASH
        and progress["state_rows"][-1]["parameter_hash"] == ACTIVE_PARAMETER_HASH,
        "source active tensor hash/count mismatch",
    )
    expected_final = {
        "protocol_id": RECOVERY_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": SOURCE_PROGRESS_SHA256,
        "parent_checkpoint_sha256": PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": PARENT_PROGRESS_SHA256,
        "selected_arm": "low",
        "accepted_updates": MAX_ACCEPTED_UPDATES,
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "termination": "max_updates_reached",
        "transition_chain_sha256": TRANSITION_CHAIN_SHA256,
        "active_parameter_hash": ACTIVE_PARAMETER_HASH,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "z_sha256": Z_SHA256,
        "accepted_vae_checkpoint_sha256": ACCEPTED_VAE_CHECKPOINT_SHA256,
    }
    for name, expected in expected_final.items():
        _require(final[name] == expected, f"final checkpoint mismatch: {name}")
    final_active = final["active_model_state"]
    _require(set(active) == set(final_active), "active tensor key set mismatch")
    for name in active:
        source_tensor = active[name]
        final_tensor = final_active[name]
        _require(
            isinstance(source_tensor, torch.Tensor)
            and isinstance(final_tensor, torch.Tensor)
            and source_tensor.device.type == final_tensor.device.type == "cpu"
            and source_tensor.dtype == final_tensor.dtype
            and source_tensor.shape == final_tensor.shape
            and torch.equal(source_tensor, final_tensor),
            f"active tensor lineage mismatch: {name}",
        )
    final_hash, final_count = _named_tensor_hash(
        final_active, label="final-checkpoint-active-state"
    )
    _require(
        final_hash == ACTIVE_PARAMETER_HASH and final_count == ACTIVE_PARAMETER_COUNT,
        "final active tensor hash/count mismatch",
    )
    terminal = progress["state_rows"][-1]
    _require(
        final["stored_state_metrics"] == terminal, "stored terminal metric mismatch"
    )
    csv_map = {
        "state_rows": "state_metrics.csv",
        "spectrum_rows": "state_spectra.csv",
        "proposal_rows": "proposal_diagnostics.csv",
        "line_rows": "line_search.csv",
        "selection_rows": "arm_selection.csv",
    }
    csv_audits = {
        records_name: _audit_checkpoint_csv(
            progress[records_name], packet / filename, label=records_name
        )
        for records_name, filename in csv_map.items()
    }
    original_names = {
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
    for packet_name, source_name in original_names.items():
        _require(
            _read_bytes(packet / packet_name)
            == _read_bytes(layout.original_source / source_name),
            f"copied original source evidence mismatch: {packet_name}",
        )
    _require(
        _load_json(packet / "source_intervention_origin.json")
        == progress["intervention_origin"]
        and _load_json(packet / "source_intervention_preflight.json")
        == progress["intervention_preflight"],
        "intervention JSON/checkpoint mismatch",
    )
    parent = _load_checkpoint(
        layout.dependency_path(PARENT_PROGRESS_REL),
        expected_sha256=PARENT_PROGRESS_SHA256,
    )
    seeded_states = [{**dict(row), "phase": "i6"} for row in parent["state_rows"]]
    seeded_spectra = [{**dict(row), "phase": "i6"} for row in parent["spectrum_rows"]]
    seeded_proposals = [
        {**dict(row), "phase": "i6"}
        for row in parent["proposal_rows"]
        if _as_int(row["accepted"], "parent accepted") == 1
    ]
    seeded_lines = [{**dict(row), "phase": "i6"} for row in parent["line_rows"]]
    seeded_selections = [
        {**dict(row), "phase": "i6"} for row in parent["selection_rows"]
    ]
    _require(
        progress["state_rows"][: START_UPDATE + 1] == seeded_states
        and progress["spectrum_rows"][: (START_UPDATE + 1) * DIMENSION]
        == seeded_spectra
        and progress["proposal_rows"][:START_UPDATE] == seeded_proposals
        and progress["line_rows"][: len(seeded_lines)] == seeded_lines
        and progress["selection_rows"] == seeded_selections
        and progress["intervention_origin"] == parent["proposal_rows"][-1],
        "parent progress prefix lineage mismatch",
    )
    checkpoint_audit = graph["recovery_audit"]["checkpoint_lineage"]
    _require(
        checkpoint_audit.get("pass") is True
        and checkpoint_audit.get("bitwise_tensor_equality") is True
        and checkpoint_audit.get("cpu_reload_pass") is True
        and checkpoint_audit.get("active_tensor_count") == ACTIVE_TENSOR_COUNT
        and checkpoint_audit.get("active_parameter_count") == ACTIVE_PARAMETER_COUNT
        and checkpoint_audit.get("active_parameter_hash") == ACTIVE_PARAMETER_HASH
        and checkpoint_audit.get("final_checkpoint_sha256") == FINAL_CHECKPOINT_SHA256
        and isinstance(checkpoint_audit.get("metadata_gates"), Mapping)
        and all(value is True for value in checkpoint_audit["metadata_gates"].values()),
        "checkpoint lineage audit mismatch",
    )
    return {
        "active_parameter_count": active_count,
        "active_tensor_count": len(active),
        "csv_audits": csv_audits,
        "final": final,
        "progress": progress,
        "terminal": terminal,
    }


def _spectral_metrics(eigenvalues: Sequence[float]) -> dict[str, float]:
    _require(len(eigenvalues) == DIMENSION, "spectrum dimension mismatch")
    eig = torch.tensor(list(eigenvalues), dtype=torch.float64)
    _require(bool(torch.isfinite(eig).all()), "spectrum is nonfinite")
    _require(bool((eig >= 0.0).all()), "spectrum has negative value")
    _require(bool((eig[1:] - eig[:-1] >= -1e-12).all()), "spectrum is unordered")
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
    full_burg = r_eig.mean() - r_eig.log().mean() - 1.0
    values = {
        "exact_a_per_dim": float(contribution.mean()),
        "a_constant_term": 1.0,
        "a_linear_trace_term": float(-2.0 * eig.mean()),
        "a_quartic_term": float(eig.square().mean()),
        "trace_m_per_dim": float(eig.mean()),
        "damped_full_burg_per_dim": float(full_burg),
        "burg_trace_r_term": float(r_eig.mean()),
        "burg_neg_logdet_r_term": float(-r_eig.log().mean()),
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
        "effective_rank": float(trace.square() / square_sum),
        "effective_rank_fraction": float(trace.square() / (square_sum * DIMENSION)),
        "li_gap_per_dim": float((eig.sqrt() - 1.0).square().mean()),
        "a_low90_abs_per_dim": float(contribution[:460].sum() / DIMENSION),
        "a_low_lt_0p1_abs_per_dim": float(contribution[eig < 0.1].sum() / DIMENSION),
        "a_high_gt_1_abs_per_dim": float(contribution[eig > 1.0].sum() / DIMENSION),
        "a_top1_share": float(contribution[-1] / total_a),
        "a_top10_share": float(contribution[-10:].sum() / total_a),
        "top1_trace_share": float(eig[-1] / trace.clamp_min(1e-30)),
        "top10_trace_share": float(eig[-10:].sum() / trace.clamp_min(1e-30)),
        "count_lt_1e_4": float((eig < 1e-4).sum()),
        "count_lt_1e_2": float((eig < 1e-2).sum()),
        "count_lt_0p1": float((eig < 0.1).sum()),
        "a_gt1": float(contribution[eig > 1.0].sum() / DIMENSION),
        "low_count": float((eig < 0.1).sum()),
    }
    return values


def _recompute_geometry_metrics(
    hessian: torch.Tensor,
    matrix: torch.Tensor,
    eig: torch.Tensor,
    low_basis: torch.Tensor,
    low_projector: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    spectral = _spectral_metrics(eig.tolist())
    raw_eig = torch.linalg.eigvalsh(matrix)
    low_values = raw_eig[:LOW_BASIS_COUNT]
    identity = torch.eye(DIMENSION, dtype=torch.float64)
    direct_a = (matrix - identity).square().sum() / DIMENSION
    trace_a = 1.0 - 2.0 * eig.mean() + eig.square().mean()
    r_eig = (eig + GEOMETRY_EPSILON) / (1.0 + GEOMETRY_EPSILON)
    gradient_eig = (1.0 - r_eig.reciprocal()) / (DIMENSION * (1.0 + GEOMETRY_EPSILON))
    orthogonality = (
        (low_basis.T @ low_basis - torch.eye(LOW_BASIS_COUNT, dtype=torch.float64))
        .abs()
        .max()
    )
    residual = (matrix @ low_basis - low_basis * low_values.unsqueeze(0)).norm() / (
        matrix.norm().clamp_min(1e-30)
    )
    canonical_low = torch.trace(low_basis.T @ matrix @ low_basis) / LOW_BASIS_COUNT
    spectral.update(
        {
            "hessian_symmetry_rel": float(
                (hessian - hessian.T).norm() / hessian.norm().clamp_min(1e-30)
            ),
            "burg_matrix_gradient_norm": float(gradient_eig.norm()),
            "burg_matrix_gradient_eig_min": float(gradient_eig.min()),
            "burg_matrix_gradient_eig_p50": float(gradient_eig.median()),
            "burg_matrix_gradient_eig_max": float(gradient_eig.max()),
            "true_objective": float(
                spectral["exact_a_per_dim"]
                + GEOMETRY_OLD_BETA * spectral["damped_full_burg_per_dim"]
            ),
            "a_direct_matrix": float(direct_a),
            "a_trace_closure": float(trace_a),
            "a_direct_abs_error": float(
                abs(float(direct_a) - spectral["exact_a_per_dim"])
            ),
            "a_trace_abs_error": float(
                abs(float(trace_a) - spectral["exact_a_per_dim"])
            ),
            "m_raw_eig_min": float(raw_eig.min()),
            "frozen_low_energy": float(canonical_low),
            "canonical_low_energy": float(canonical_low),
            "low_basis_orthogonality_max_abs": float(orthogonality),
            "low_basis_eigen_residual_relative": float(residual),
            "helper_eigenvalue_max_abs_error": float(
                (eig - raw_eig.clamp_min(0.0)).abs().max()
            ),
        }
    )
    closures = {
        "matrix_from_hessian_max_abs_error": float(
            (matrix - hessian @ hessian.T).abs().max()
        ),
        "matrix_symmetry_max_abs_error": float((matrix - matrix.T).abs().max()),
        "eigvalsh_matrix_max_abs_error": float(
            (eig - raw_eig.clamp_min(0.0)).abs().max()
        ),
        "low_basis_orthogonality_max_abs": float(orthogonality),
        "low_basis_eigen_residual_relative": float(residual),
        "low_projector_basis_max_abs_error": float(
            (low_projector - low_basis @ low_basis.T).abs().max()
        ),
    }
    return spectral, closures


def _validate_geometry_payload(
    payload: Mapping[str, Any], terminal: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        isinstance(payload, Mapping) and set(payload) == GEOMETRY_KEYS,
        "geometry schema mismatch",
    )
    expected_metadata = {
        "protocol_id": RECOVERY_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": SOURCE_PROGRESS_SHA256,
        "accepted_vae_checkpoint_sha256": ACCEPTED_VAE_CHECKPOINT_SHA256,
        "active_parameter_hash": ACTIVE_PARAMETER_HASH,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "task_name": "fashion_mnist",
        "tau": 1.1614345407370807,
        "z_sha256": Z_SHA256,
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
    tensor_names = (
        "hessian",
        "matrix",
        "eig",
        "current_low_basis",
        "current_low_projector",
    )
    tensors = {name: payload[name] for name in tensor_names}
    expected_shapes = {
        "hessian": [DIMENSION, DIMENSION],
        "matrix": [DIMENSION, DIMENSION],
        "eig": [DIMENSION],
        "current_low_basis": [DIMENSION, LOW_BASIS_COUNT],
        "current_low_projector": [DIMENSION, DIMENSION],
    }
    for name, tensor in tensors.items():
        _require(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float64
            and tensor.requires_grad is False
            and list(tensor.shape) == expected_shapes[name]
            and bool(torch.isfinite(tensor).all()),
            f"geometry tensor invalid: {name}",
        )
    fingerprints: dict[str, dict[str, Any]] = {}
    for index, (name, tensor) in enumerate(tensors.items(), start=1):
        _log(
            "geometry-fingerprint-progress",
            f"tensor={name} index={index}/{len(tensors)}",
        )
        fingerprints[name] = _tensor_fingerprint(tensor)
    _require(
        payload["tensor_fingerprints"] == fingerprints, "geometry fingerprints mismatch"
    )
    _require(
        fingerprints["eig"]["sha256"] == GEOMETRY_EIG_SHA256,
        "geometry eig hash mismatch",
    )
    metrics, closures = _recompute_geometry_metrics(
        tensors["hessian"],
        tensors["matrix"],
        tensors["eig"],
        tensors["current_low_basis"],
        tensors["current_low_projector"],
    )
    _require(
        closures["matrix_from_hessian_max_abs_error"] <= REPLAY_ATOL
        and closures["matrix_symmetry_max_abs_error"] <= REPLAY_ATOL,
        "M/H closure failed",
    )
    _require(
        closures["eigvalsh_matrix_max_abs_error"] <= REPLAY_ATOL,
        "eigensystem closure failed",
    )
    _require(
        closures["low_basis_orthogonality_max_abs"] <= REPLAY_ATOL
        and closures["low_basis_eigen_residual_relative"] <= REPLAY_ATOL
        and closures["low_projector_basis_max_abs_error"] <= REPLAY_ATOL,
        "low basis/projector closure failed",
    )
    stored_metrics = payload["metrics"]
    _require(
        isinstance(stored_metrics, Mapping)
        and set(stored_metrics)
        == set(metrics) | {"task_loss", "hessian_sec", "low_basis_hash"},
        "geometry metric schema mismatch",
    )
    metric_errors: dict[str, float] = {}
    source_errors: dict[str, float] = {}
    for name, expected in metrics.items():
        observed = _as_float(stored_metrics[name], f"geometry metric {name}")
        error = abs(observed - expected)
        _require(error <= REPLAY_ATOL, f"geometry metric closure failed: {name}")
        metric_errors[name] = error
    for name, value in stored_metrics.items():
        if name == "hessian_sec":
            _require(
                _as_float(value, "geometry hessian_sec") >= 0.0, "invalid replay timing"
            )
        elif name == "low_basis_hash":
            _require(
                value
                == terminal[name]
                == fingerprints["current_low_projector"]["sha256"],
                "low basis hash link mismatch",
            )
        elif isinstance(value, str):
            _require(value == terminal[name], f"source metric text mismatch: {name}")
        else:
            error = abs(_as_float(value, name) - _as_float(terminal[name], name))
            _require(error <= REPLAY_ATOL, f"source terminal metric mismatch: {name}")
            source_errors[name] = error
    return {
        "closures": closures,
        "eig": tensors["eig"],
        "fingerprints": fingerprints,
        "metric_closure_max_abs_error": max(metric_errors.values()),
        "metrics": stored_metrics,
        "source_metric_max_abs_error": max(source_errors.values()),
    }


def _validate_spectrum_chain(
    packet: Path,
    progress: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint = progress["spectrum_rows"][MAX_ACCEPTED_UPDATES * DIMENSION :]
    _require(len(endpoint) == DIMENSION, "terminal progress spectrum count mismatch")
    _require(
        [
            (
                _as_int(row["accepted_update"], "accepted_update"),
                _as_int(row["rank"], "rank"),
            )
            for row in endpoint
        ]
        == [(MAX_ACCEPTED_UPDATES, rank) for rank in range(DIMENSION)],
        "terminal progress spectrum grid mismatch",
    )
    progress_values = [
        _as_float(row["m_eigenvalue"], "progress eigenvalue") for row in endpoint
    ]
    _, state_spectrum_rows = _read_csv(packet / "state_spectra.csv")
    csv_endpoint = state_spectrum_rows[-DIMENSION:]
    _require(
        [
            (
                _as_csv_int(row["accepted_update"], "CSV accepted_update"),
                _as_csv_int(row["rank"], "CSV rank"),
            )
            for row in csv_endpoint
        ]
        == [(MAX_ACCEPTED_UPDATES, rank) for rank in range(DIMENSION)],
        "terminal CSV spectrum grid mismatch",
    )
    csv_values = [
        _as_float(row["m_eigenvalue"], "CSV eigenvalue") for row in csv_endpoint
    ]
    replay_header, replay_rows = _read_csv(packet / "replayed_final_spectrum.csv")
    _require(
        replay_header == ["rank", "stored", "replayed", "abs_error"]
        and len(replay_rows) == DIMENSION,
        "replayed spectrum CSV schema/count mismatch",
    )
    replay_stored: list[float] = []
    replay_replayed: list[float] = []
    for rank, row in enumerate(replay_rows):
        _require(
            _as_csv_int(row["rank"], "replayed rank") == rank,
            "replayed rank mismatch",
        )
        stored = _as_float(row["stored"], "replayed stored")
        replayed = _as_float(row["replayed"], "replayed value")
        error = _as_float(row["abs_error"], "replayed error")
        _require(
            _bitwise_float_equal(stored, replayed)
            and _float_bits(error) == _float_bits(0.0),
            f"replayed spectrum mismatch at rank {rank}",
        )
        replay_stored.append(stored)
        replay_replayed.append(replayed)
    geometry_values = geometry["eig"].tolist()
    representations = (
        progress_values,
        csv_values,
        replay_stored,
        replay_replayed,
        geometry_values,
    )
    for rank in range(DIMENSION):
        bits = {_float_bits(values[rank]) for values in representations}
        _require(len(bits) == 1, f"bitwise spectrum closure failed at rank {rank}")
    _require(
        _float_bits(csv_values[2]).hex() == "3de22a7f787e6c62",
        "rank-2 correctly-rounded binary64 regression failed",
    )
    return {
        "bitwise_equal_count": DIMENSION,
        "max_absolute_error": 0.0,
        "rank_2_binary64_hex": _float_bits(csv_values[2]).hex(),
        "representation_count": len(representations),
    }


def _validate_geometry(
    packet: Path, checkpoint: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    path = packet / "replayed_final_geometry.pt"
    payload = _load_checkpoint(path, expected_sha256=GEOMETRY_ARTIFACT_SHA256)
    geometry = _validate_geometry_payload(payload, checkpoint["terminal"])
    spectrum = _validate_spectrum_chain(packet, checkpoint["progress"], geometry)
    replay = graph["recovery_audit"]["final_geometry_replay"]
    _require(
        replay.get("pass") is True
        and replay.get("metric_absolute_tolerance") == REPLAY_ATOL
        and replay.get("spectrum_absolute_tolerance") == REPLAY_ATOL
        and replay.get("metric_max_absolute_error") == 0.0
        and replay.get("spectrum_max_absolute_error") == 0.0
        and replay.get("spectrum_eigenvalue_count") == DIMENSION
        and replay.get("spectrum_max_error_rank") == 0
        and replay.get("low_basis_hash_matches") is True
        and replay.get("active_parameter_hash_matches") is True
        and replay.get("hessian_shape") == [DIMENSION, DIMENSION]
        and replay.get("matrix_shape") == [DIMENSION, DIMENSION]
        and replay.get("spectrum_shape") == [DIMENSION]
        and replay.get("low_basis_shape") == [DIMENSION, LOW_BASIS_COUNT]
        and replay.get("low_projector_shape") == [DIMENSION, DIMENSION]
        and replay.get("geometry_artifact_sha256") == GEOMETRY_ARTIFACT_SHA256
        and replay.get("replayed_final_spectrum_sha256")
        == _sha256_file(packet / "replayed_final_spectrum.csv")
        and replay.get("replayed_final_spectrum_rows") == DIMENSION,
        "stored final geometry replay record mismatch",
    )
    fingerprint_links = {
        "hessian_sha256": "hessian",
        "matrix_sha256": "matrix",
        "spectrum_tensor_sha256": "eig",
        "low_basis_sha256": "current_low_basis",
        "low_projector_sha256": "current_low_projector",
    }
    for record_name, tensor_name in fingerprint_links.items():
        _require(
            replay.get(record_name) == geometry["fingerprints"][tensor_name]["sha256"],
            f"replay tensor hash link mismatch: {record_name}",
        )
    _require(
        graph["decision"]["final_metrics"] == payload["metrics"],
        "decision/geometry metric payload mismatch",
    )
    model = _validate_model_nonmutation(graph["recovery_audit"], payload)
    return {**geometry, "model": model, "spectrum": spectrum}


def _chain_hash(
    previous: str,
    *,
    update: int,
    base_hash: str,
    endpoint_hash: str,
    alpha: float,
) -> str:
    payload = json.dumps(
        {
            "previous": previous,
            "update": update,
            "base_parameter_hash": base_hash,
            "endpoint_parameter_hash": endpoint_hash,
            "alpha": float(alpha),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _near_cancellation_diagnostics(
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in proposals
        if _as_int(row["target_update"], "proposal target") > START_UPDATE
    ]
    _require(rows, "continuation proposal set is empty")
    sources = [_as_float(row["unit_common_source_norm"], "source norm") for row in rows]
    unbounded = any(value == 0.0 for value in sources)
    shadow = [
        row for row in rows if str(row["fp64_shadow_valid"]).lower() in {"true", "1"}
    ]
    accepted = [
        row for row in rows if _as_int(row["accepted"], "proposal accepted") == 1
    ]
    return {
        "minimum_source_norm": min(sources),
        "final_source_norm": sources[-1],
        "maximum_amplification": None
        if unbounded
        else max(
            _as_float(row["common_amplification"], "common amplification")
            for row in rows
        ),
        "maximum_amplification_unbounded": unbounded,
        "minimum_fp32_fp64_direction_cosine": min(
            _as_float(row["fp32_fp64_direction_cosine"], "direction cosine")
            for row in shadow
        )
        if shadow
        else 0.0,
        "maximum_fp32_fp64_direction_relative_error": max(
            _as_float(row["fp32_fp64_direction_relative_error"], "direction error")
            for row in shadow
        )
        if shadow
        else 0.0,
        "minimum_accepted_alpha": min(
            _as_float(row["selected_alpha"], "selected alpha") for row in accepted
        )
        if accepted
        else 0.0,
        "descriptive_only": True,
    }


def _validate_all_spectrum_metrics(progress: Mapping[str, Any]) -> dict[str, Any]:
    states = progress["state_rows"]
    spectra = progress["spectrum_rows"]
    _require(
        [_as_int(row["accepted_update"], "state update") for row in states]
        == list(range(MAX_ACCEPTED_UPDATES + 1)),
        "state update sequence mismatch",
    )
    hashes = [row["parameter_hash"] for row in states]
    _require(
        all(_is_sha256(value) for value in hashes) and len(set(hashes)) == len(hashes),
        "state parameter hashes malformed or repeated",
    )
    worst_metric_error = 0.0
    worst_contribution_error = 0.0
    for update, state_row in enumerate(states):
        block = spectra[update * DIMENSION : (update + 1) * DIMENSION]
        _require(
            len(block) == DIMENSION
            and [
                (
                    _as_int(row["accepted_update"], "spectrum update"),
                    _as_int(row["rank"], "spectrum rank"),
                )
                for row in block
            ]
            == [(update, rank) for rank in range(DIMENSION)],
            f"spectrum grid mismatch at update {update}",
        )
        eig = [_as_float(row["m_eigenvalue"], "spectrum eigenvalue") for row in block]
        expected = _spectral_metrics(eig)
        contribution_error = max(
            abs(
                _as_float(row["a_contribution"], "spectrum contribution")
                - (eig[rank] - 1.0) ** 2
            )
            for rank, row in enumerate(block)
        )
        _require(
            contribution_error <= REPLAY_ATOL,
            f"spectrum contribution closure failed at update {update}",
        )
        errors: list[float] = []
        for name, reference in expected.items():
            observed = _as_float(state_row[name], f"state metric {name}")
            error = abs(observed - reference)
            if name in {"count_lt_1e_4", "count_lt_1e_2", "count_lt_0p1", "low_count"}:
                _require(
                    observed == reference, f"integral spectrum metric mismatch: {name}"
                )
            else:
                _require(
                    error <= REPLAY_ATOL,
                    f"spectrum metric mismatch: update {update} {name}",
                )
            errors.append(error)
        raw_min = _as_float(state_row["m_raw_eig_min"], "raw spectrum minimum")
        _require(
            raw_min >= -1e-8 and abs(max(raw_min, 0.0) - eig[0]) <= REPLAY_ATOL,
            f"raw spectrum minimum mismatch at update {update}",
        )
        _require(
            _as_float(state_row["a_direct_abs_error"], "direct A error") <= REPLAY_ATOL
            and _as_float(state_row["a_trace_abs_error"], "trace A error")
            <= REPLAY_ATOL
            and _as_float(
                state_row["low_basis_orthogonality_max_abs"], "basis orthogonality"
            )
            <= 1e-10
            and _as_float(
                state_row["low_basis_eigen_residual_relative"], "basis residual"
            )
            <= 1e-10
            and _as_float(
                state_row["helper_eigenvalue_max_abs_error"], "helper eig error"
            )
            <= 1e-10
            and _is_sha256(state_row["low_basis_hash"]),
            f"state closure audit failed at update {update}",
        )
        worst_metric_error = max(worst_metric_error, max(errors))
        worst_contribution_error = max(worst_contribution_error, contribution_error)
    return {
        "state_count": len(states),
        "spectrum_row_count": len(spectra),
        "worst_spectrum_metric_abs_error": worst_metric_error,
        "worst_contribution_abs_error": worst_contribution_error,
    }


def _validate_transition_chain(progress: Mapping[str, Any]) -> dict[str, Any]:
    states = progress["state_rows"]
    proposals = progress["proposal_rows"]
    _require(
        [_as_int(row["target_update"], "proposal target") for row in proposals]
        == list(range(1, MAX_ACCEPTED_UPDATES + 1))
        and all(
            _as_int(row["accepted"], "proposal accepted") == 1 for row in proposals
        ),
        "accepted proposal sequence mismatch",
    )
    state_by_update = {
        _as_int(row["accepted_update"], "state update"): row for row in states
    }
    chain = PARENT_FINAL_SHA256
    checked = 0
    for proposal in proposals[START_UPDATE:]:
        target = _as_int(proposal["target_update"], "continuation target")
        _require(
            target == START_UPDATE + checked + 1, "continuation target order mismatch"
        )
        parent_hash = state_by_update[target - 1]["parameter_hash"]
        endpoint_hash = state_by_update[target]["parameter_hash"]
        _require(
            proposal["base_parameter_hash"] == parent_hash
            and proposal["endpoint_parameter_hash"] == endpoint_hash,
            f"transition endpoint linkage mismatch: {target}",
        )
        chain = _chain_hash(
            chain,
            update=target,
            base_hash=parent_hash,
            endpoint_hash=endpoint_hash,
            alpha=_as_float(proposal["selected_alpha"], "transition alpha"),
        )
        _require(
            proposal["transition_chain_sha256"] == chain,
            f"transition chain mismatch: {target}",
        )
        checked += 1
    _require(
        chain == progress["transition_chain_sha256"] == TRANSITION_CHAIN_SHA256,
        "terminal chain mismatch",
    )
    return {"continuation_links_checked": checked, "transition_chain_sha256": chain}


SUCCESS_GATE_ORDER = (
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
)


def _derive_scientific_outcome(
    progress: Mapping[str, Any], *, all_historical_a_armijo: bool
) -> dict[str, Any]:
    states = progress["state_rows"]
    spectra = progress["spectrum_rows"]
    proposals = progress["proposal_rows"]
    tolerances = {
        name: _as_float(value, f"tolerance {name}")
        for name, value in progress["tolerances"].items()
    }
    initial = states[0]
    intervention = states[START_UPDATE]
    final = states[-1]
    total_a = _as_float(initial["exact_a_per_dim"], "initial A") - _as_float(
        final["exact_a_per_dim"], "final A"
    )
    total_low = _as_float(initial["a_low90_abs_per_dim"], "initial low90") - _as_float(
        final["a_low90_abs_per_dim"], "final low90"
    )
    continuation_a = _as_float(
        intervention["exact_a_per_dim"], "intervention A"
    ) - _as_float(final["exact_a_per_dim"], "final A")
    continuation_low = _as_float(
        intervention["a_low90_abs_per_dim"], "intervention low90"
    ) - _as_float(final["a_low90_abs_per_dim"], "final low90")
    a_diffs = [
        _as_float(states[index]["exact_a_per_dim"], "A")
        - _as_float(states[index - 1]["exact_a_per_dim"], "A")
        for index in range(1, len(states))
    ]
    high_diffs = [
        _as_float(states[index]["a_gt1"], "high tail")
        - _as_float(states[index - 1]["a_gt1"], "high tail")
        for index in range(1, len(states))
    ]
    tail_bulk = sum(
        _as_float(states[index - 1]["a_low90_abs_per_dim"], "previous low90")
        - _as_float(states[index]["a_low90_abs_per_dim"], "current low90")
        > tolerances["A_low90"]
        for index in range(81, 101)
    )
    bulk_fraction = total_low / max(total_a, 1e-30)
    all_state_finite = all(
        math.isfinite(float(value))
        for row in states
        for name, value in row.items()
        if name not in {"parameter_hash", "low_basis_hash", "phase"}
    )
    all_spectrum_finite = all(
        math.isfinite(float(value))
        for row in spectra
        for name, value in row.items()
        if name != "phase"
    )
    success_gates = {
        "one_hundred_accepted_updates": progress["accepted_updates"]
        == MAX_ACCEPTED_UPDATES,
        "tail_bulk_activity": tail_bulk >= 4,
        "final_a_at_most_0p90": _as_float(final["exact_a_per_dim"], "final A") <= 0.90,
        "accepted_a_strictly_decreases": all(
            value < -tolerances["A"] for value in a_diffs
        ),
        "all_historical_a_armijo_transitions": all_historical_a_armijo,
        "final_b_below_initial": _as_float(final["damped_full_burg_per_dim"], "final B")
        < _as_float(initial["damped_full_burg_per_dim"], "initial B") - tolerances["B"],
        "non_top_only_fraction_at_least_0p25": bulk_fraction >= 0.25,
        "final_p50_above_initial": _as_float(final["m_p50"], "final p50")
        > _as_float(initial["m_p50"], "initial p50") + tolerances["m_p50"],
        "final_effective_rank_above_initial": _as_float(
            final["effective_rank"], "final effective rank"
        )
        > _as_float(initial["effective_rank"], "initial effective rank")
        + tolerances["effective_rank"],
        "final_count_lt_1e_4_strictly_lower": _as_int(
            final["count_lt_1e_4"], "final count 1e-4"
        )
        < _as_int(initial["count_lt_1e_4"], "initial count 1e-4"),
        "final_count_lt_1e_2_strictly_lower": _as_int(
            final["count_lt_1e_2"], "final count 1e-2"
        )
        < _as_int(initial["count_lt_1e_2"], "initial count 1e-2"),
        "final_count_lt_0p1_strictly_lower": _as_int(
            final["count_lt_0p1"], "final count 0.1"
        )
        < _as_int(initial["count_lt_0p1"], "initial count 0.1"),
        "final_mmax_not_above_initial": _as_float(final["m_max"], "final mmax")
        <= _as_float(initial["m_max"], "initial mmax") + tolerances["m_max"],
        "high_tail_never_increases_beyond_floor": all(
            value <= tolerances["A_gt1"] for value in high_diffs
        ),
        "all_state_values_finite": all_state_finite,
        "all_spectrum_values_finite": all_spectrum_finite,
        "all_raw_spectrum_minima_valid": all(
            _as_float(row["m_raw_eig_min"], "raw spectrum minimum") >= -1e-8
            for row in states
        ),
        "all_exact_a_closures_valid": all(
            _as_float(row["a_direct_abs_error"], "direct A error") <= REPLAY_ATOL
            and _as_float(row["a_trace_abs_error"], "trace A error") <= REPLAY_ATOL
            for row in states
        ),
        "final_checkpoint_replay": True,
    }
    _require(tuple(success_gates) == SUCCESS_GATE_ORDER, "success gate order mismatch")
    continuation_rows = [
        row
        for row in proposals
        if _as_int(row["target_update"], "proposal target") > START_UPDATE
    ]
    immediate = bool(
        continuation_rows
        and _as_int(continuation_rows[0]["target_update"], "first continuation")
        == START_UPDATE + 1
        and _as_int(continuation_rows[0]["accepted"], "first accepted") == 1
    )
    return {
        "immediate_premature_cutoff": immediate,
        "sustained_continuation": NEW_ACCEPTED_UPDATES >= 10,
        "b_non_descent": False,
        "finite_grid_exhaustion": False,
        "finite_grid_conflict": False,
        "scalar_only_repair": bool(
            immediate and continuation_a > tolerances["A"] and bulk_fraction < 0.25
        ),
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "scientific_success": bool(all(success_gates.values())),
        "success_gates": success_gates,
        "termination": "max_updates_reached",
        "total_a_reduction": total_a,
        "total_lower90_reduction": total_low,
        "total_non_top_only_fraction": bulk_fraction,
        "continuation_a_reduction": continuation_a,
        "continuation_lower90_reduction": continuation_low,
        "continuation_non_top_only_fraction": continuation_low
        / max(continuation_a, 1e-30),
        "tail_bulk_updates": tail_bulk,
        "near_cancellation_diagnostics": _near_cancellation_diagnostics(proposals),
    }


def _validate_scientific_outcome(
    packet: Path, checkpoint: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    progress = checkpoint["progress"]
    spectrum_metrics = _validate_all_spectrum_metrics(progress)
    transition = _validate_transition_chain(progress)
    history_header, history_rows = _read_csv(packet / "historical_transition_audit.csv")
    _require(
        history_header
        == [
            "accepted_update",
            "alpha",
            "A_slope_negative",
            "A_armijo",
            "A_actual_decrease",
            "B_slope_negative",
            "B_armijo",
            "B_actual_decrease",
            "L_low_slope_negative",
            "L_low_armijo",
            "L_low_actual_decrease",
            "high_tail_nonincrease",
            "selected_line_row_passes",
            "committed_state_matches",
            "transition_passes",
        ]
        and len(history_rows) == MAX_ACCEPTED_UPDATES,
        "historical transition audit schema/count mismatch",
    )
    integer_gate_columns = set(history_header) - {"accepted_update", "alpha"}
    _require(
        [
            _as_csv_int(row["accepted_update"], "history accepted_update")
            for row in history_rows
        ]
        == list(range(1, MAX_ACCEPTED_UPDATES + 1))
        and all(_as_float(row["alpha"], "history alpha") > 0.0 for row in history_rows),
        "historical transition update/alpha fields mismatch",
    )
    all_history = all(
        _as_csv_int(row[name], f"history {name}") == 1
        for row in history_rows
        for name in integer_gate_columns
    )
    all_a_armijo = all(
        all(
            _as_csv_int(row[name], f"history {name}") == 1
            for name in ("A_slope_negative", "A_armijo", "A_actual_decrease")
        )
        for row in history_rows
    )
    _require(all_history and all_a_armijo, "historical transition gate failed")
    outcome = _derive_scientific_outcome(progress, all_historical_a_armijo=all_a_armijo)
    decision_outcome = graph["decision"]["outcome"]
    _require(
        isinstance(decision_outcome, Mapping)
        and set(decision_outcome) == set(outcome)
        and _equivalent(decision_outcome, outcome, atol=REPLAY_ATOL),
        "scientific outcome does not independently recompute",
    )
    failed = [name for name in SUCCESS_GATE_ORDER if not outcome["success_gates"][name]]
    _require(
        outcome["scientific_success"] is False
        and graph["decision"]["scientific_success"] is False
        and graph["finalized"]["scientific_success"] is False
        and failed == EXPECTED_FAILED_SUCCESS_GATES
        and outcome["total_non_top_only_fraction"] == 0.07733408144475684
        and outcome["continuation_non_top_only_fraction"] == 0.11301885339451084
        and checkpoint["terminal"]["exact_a_per_dim"] == 0.9321672207645146,
        "scientific outcome or failed gate set changed",
    )
    final_metrics = graph["decision"]["final_metrics"]
    terminal = checkpoint["terminal"]
    expected_final_names = set(terminal) - {
        "accepted_update",
        "parameter_hash",
        "parent_frozen_low_energy",
        "phase",
    }
    _require(set(final_metrics) == expected_final_names, "final metric schema mismatch")
    for name, value in final_metrics.items():
        if name == "hessian_sec":
            _require(
                _as_float(value, "final hessian_sec") >= 0.0, "invalid final timing"
            )
        elif isinstance(value, str):
            _require(value == terminal[name], f"final metric text mismatch: {name}")
        else:
            _require(
                abs(_as_float(value, name) - _as_float(terminal[name], name))
                <= REPLAY_ATOL,
                f"final metric mismatch: {name}",
            )
    return {
        "failed_success_gates": failed,
        "outcome": outcome,
        "spectrum_metrics": spectrum_metrics,
        "transition": transition,
    }


def audit_failed_packet_read_only(
    layout: Layout = PRODUCTION_LAYOUT,
    *,
    manifest: FailedManifest | None = None,
    protected: Mapping[str, Any] | None = None,
) -> ScientificAudit:
    started = time.monotonic()

    def report(step: str, message: str = "") -> None:
        suffix = f" {message}" if message else ""
        _log(
            "scientific-preflight-progress",
            f"step={step} elapsed_seconds={time.monotonic() - started:.3f}{suffix}",
        )

    _require_cuda_uninitialized()
    report(
        "config",
        f"packet={layout.failed} device=cpu dtype=torch.float64 "
        "seed=none cache_mode=read-only",
    )
    manifest = manifest or _load_failed_manifest(layout)
    report("protected-tree-preflight")
    protected_before = dict(protected or _audit_protected_trees(layout, manifest))
    packet = layout.failed
    report("failure-localization")
    _validate_failure_and_log(packet)
    report("packet-hash-graph")
    graph = _validate_packet_hash_graph(packet)
    report("execution-budget")
    _validate_execution_budget(graph)
    report("task-and-input-provenance")
    task_inputs = _validate_task_and_inputs(layout, packet, graph)
    report("checkpoint-lineage")
    checkpoint = _validate_checkpoint_lineage(layout, packet, graph)
    report("geometry-closure")
    geometry = _validate_geometry(packet, checkpoint, graph)
    report("scientific-outcome")
    scientific = _validate_scientific_outcome(packet, checkpoint, graph)
    report("protected-tree-postflight")
    protected_after = _audit_protected_trees(layout, manifest)
    _require(
        protected_before == protected_after, "protected tree changed during preflight"
    )
    _require_cuda_uninitialized()
    closures = geometry["closures"]
    numeric_audit = {
        "absolute_tolerance": REPLAY_ATOL,
        "relative_tolerance": 0.0,
        "matrix_from_hessian_max_abs_error": closures[
            "matrix_from_hessian_max_abs_error"
        ],
        "matrix_symmetry_max_abs_error": closures["matrix_symmetry_max_abs_error"],
        "eigvalsh_matrix_max_abs_error": closures["eigvalsh_matrix_max_abs_error"],
        "low_basis_orthogonality_max_abs": closures["low_basis_orthogonality_max_abs"],
        "low_basis_eigen_residual_relative": closures[
            "low_basis_eigen_residual_relative"
        ],
        "low_projector_basis_max_abs_error": closures[
            "low_projector_basis_max_abs_error"
        ],
        "metric_closure_max_abs_error": geometry["metric_closure_max_abs_error"],
        "source_metric_max_abs_error": geometry["source_metric_max_abs_error"],
        "spectrum_max_abs_error": geometry["spectrum"]["max_absolute_error"],
        "spectrum_bitwise_equal_count": geometry["spectrum"]["bitwise_equal_count"],
        "spectrum_representation_count": geometry["spectrum"]["representation_count"],
        "rank_2_binary64_hex": geometry["spectrum"]["rank_2_binary64_hex"],
        "state_csv_rows": checkpoint["csv_audits"]["state_rows"]["row_count"],
        "spectrum_csv_rows": checkpoint["csv_audits"]["spectrum_rows"]["row_count"],
        "scientific_success": False,
        "failed_scientific_success_gates": EXPECTED_FAILED_SUCCESS_GATES,
        "final_exact_a_per_dim": 0.9321672207645146,
        "continuation_non_top_only_fraction": 0.11301885339451084,
        "total_non_top_only_fraction": 0.07733408144475684,
    }
    gates = {
        "failure_and_run_log_exact": True,
        "packet_hash_graph_exact": True,
        "execution_budget_exact": True,
        "checkpoint_lineage_exact": True,
        "task_and_input_provenance_exact": True,
        "model_nonmutation_exact": geometry["model"]["unchanged_during_replay"],
        "geometry_shapes_dtypes_finite": True,
        "matrix_from_hessian_closure": closures["matrix_from_hessian_max_abs_error"]
        <= REPLAY_ATOL,
        "eigensystem_closure": closures["eigvalsh_matrix_max_abs_error"] <= REPLAY_ATOL,
        "low_basis_projector_closure": max(
            closures["low_basis_orthogonality_max_abs"],
            closures["low_basis_eigen_residual_relative"],
            closures["low_projector_basis_max_abs_error"],
        )
        <= REPLAY_ATOL,
        "metric_closure": geometry["metric_closure_max_abs_error"] <= REPLAY_ATOL,
        "spectrum_bitwise_closure": geometry["spectrum"]["bitwise_equal_count"]
        == DIMENSION,
        "csv_checkpoint_closure": all(
            audit["row_count"] > 0 for audit in checkpoint["csv_audits"].values()
        ),
        "scientific_outcome_unchanged": scientific["failed_success_gates"]
        == EXPECTED_FAILED_SUCCESS_GATES,
        "cuda_uninitialized": _cuda_uninitialized(),
        "scientific_execution_absent": True,
    }
    _require(all(gates.values()), "scientific preflight gate failed")
    report(
        "complete",
        f"gates={len(gates)} spectrum_bitwise_equal_count="
        f"{numeric_audit['spectrum_bitwise_equal_count']} "
        f"scientific_success={int(numeric_audit['scientific_success'])} "
        f"artifact={packet / 'replayed_final_geometry.pt'}",
    )
    return ScientificAudit(
        gates=gates,
        numeric_audit=numeric_audit,
        details={
            "artifact_link_count": graph["artifact_link_count"],
            "checkpoint": {
                "active_parameter_count": checkpoint["active_parameter_count"],
                "active_tensor_count": checkpoint["active_tensor_count"],
            },
            "scientific": scientific,
            "task_inputs": task_inputs,
        },
    )


def _fsync_directory(path: Path) -> None:
    path = _require_directory(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        _require(written > 0, "zero-byte write")
        offset += written


def _create_exclusive_file(path: Path, data: bytes, *, label: str) -> None:
    path = _require_absent(path, label)
    parent = _require_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as error:
        raise SalvageError(f"{label} creation lost exclusivity: {path}") from error
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _require_regular_file(path, immutable=True, single_link=True)
    _require(_read_bytes(path) == data, f"{label} readback mismatch")
    _fsync_directory(parent)


def _files_byte_identical(left: Path, right: Path) -> bool:
    left_fd, left_stat = _open_readonly(left)
    right_fd, right_stat = _open_readonly(right)
    try:
        if left_stat.st_size != right_stat.st_size:
            return False
        while True:
            left_chunk = os.read(left_fd, 1024 * 1024)
            right_chunk = os.read(right_fd, 1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                break
        left_after = os.fstat(left_fd)
        right_after = os.fstat(right_fd)
    finally:
        os.close(left_fd)
        os.close(right_fd)
    _require(
        (left_stat.st_dev, left_stat.st_ino, left_stat.st_size, left_stat.st_mtime_ns)
        == (
            left_after.st_dev,
            left_after.st_ino,
            left_after.st_size,
            left_after.st_mtime_ns,
        ),
        f"file changed during byte comparison: {left}",
    )
    _require(
        (
            right_stat.st_dev,
            right_stat.st_ino,
            right_stat.st_size,
            right_stat.st_mtime_ns,
        )
        == (
            right_after.st_dev,
            right_after.st_ino,
            right_after.st_size,
            right_after.st_mtime_ns,
        ),
        f"file changed during byte comparison: {right}",
    )
    return True


def _create_or_verify_snapshot(source: Path, snapshot: Path) -> dict[str, Any]:
    source = _require_regular_file(source)
    source_bytes = _read_bytes(source)
    if os.path.lexists(snapshot):
        snapshot = _require_regular_file(snapshot, immutable=True, single_link=True)
        _require(
            _files_byte_identical(source, snapshot),
            "existing publisher snapshot is not byte-identical",
        )
        created = False
    else:
        _create_exclusive_file(snapshot, source_bytes, label="publisher snapshot")
        created = True
    source_stat = os.lstat(source)
    snapshot_stat = os.lstat(snapshot)
    _require(
        (source_stat.st_dev, source_stat.st_ino)
        != (snapshot_stat.st_dev, snapshot_stat.st_ino),
        "publisher snapshot is a hardlink",
    )
    digest = _sha256_file(snapshot)
    _require(digest == _sha256_file(source), "publisher snapshot hash mismatch")
    return {
        "created": created,
        "path": str(snapshot),
        "sha256": digest,
        "size_bytes": snapshot_stat.st_size,
    }


def _copy_one(source: Path, destination: Path, spec: FileSpec) -> dict[str, Any]:
    source_fd, source_before = _open_readonly(source)
    _require(
        source_before.st_size == spec.size_bytes and source_before.st_nlink == 1,
        f"copy source size mismatch: {spec.name}",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        destination_fd = os.open(destination, flags, 0o444)
    except FileExistsError as error:
        os.close(source_fd)
        raise SalvageError(f"copy destination exists: {destination}") from error
    digest = hashlib.sha256()
    copied = 0
    try:
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(destination_fd, chunk)
            copied += len(chunk)
        source_after = os.fstat(source_fd)
        os.fsync(destination_fd)
        destination_stat = os.fstat(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    _require(
        (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
        )
        == (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
        ),
        f"copy source changed: {spec.name}",
    )
    _require(
        copied == spec.size_bytes
        and destination_stat.st_size == spec.size_bytes
        and digest.hexdigest() == spec.sha256,
        f"copied byte/hash mismatch: {spec.name}",
    )
    _require(
        destination_stat.st_nlink == 1
        and (source_before.st_dev, source_before.st_ino)
        != (destination_stat.st_dev, destination_stat.st_ino),
        f"copy is hardlinked: {spec.name}",
    )
    _require_regular_file(destination, immutable=True, single_link=True)
    _require(
        _sha256_file(destination) == spec.sha256, f"copy readback mismatch: {spec.name}"
    )
    return {
        "destination_device": destination_stat.st_dev,
        "destination_inode": destination_stat.st_ino,
        "destination_nlink": destination_stat.st_nlink,
        "name": spec.name,
        "sha256": spec.sha256,
        "size_bytes": spec.size_bytes,
        "source_device": source_before.st_dev,
        "source_inode": source_before.st_ino,
        "source_nlink": source_before.st_nlink,
    }


def _copy_projection(
    source: Path,
    stage: Path,
    specs: Sequence[FileSpec],
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    source = _require_directory(source)
    stage = _require_absent(stage, "staging path")
    parent = _require_directory(stage.parent)
    try:
        os.mkdir(stage, 0o700)
    except FileExistsError as error:
        raise SalvageError("staging directory creation lost exclusivity") from error
    _fsync_directory(parent)
    if fault_hook is not None:
        fault_hook("after_stage_created")
    per_file: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        per_file.append(_copy_one(source / spec.name, stage / spec.name, spec))
        _log(
            "byte-copy-progress",
            f"file={spec.name} index={index}/{len(specs)} bytes={spec.size_bytes}",
        )
        if fault_hook is not None:
            fault_hook(f"after_copy:{spec.name}")
    _fsync_directory(stage)
    _fsync_directory(parent)
    return {
        "copied_names": [spec.name for spec in specs],
        "distinct_inode_count": len(per_file),
        "excluded_names": ["failure.json"],
        "hardlink_pair_count": 0,
        "per_file": per_file,
        "removed_after_validation": ["INCOMPLETE"],
        "source_destination_sha256_equal": True,
        "source_destination_size_equal": True,
    }


def _current_copy_evidence(
    source: Path, destination: Path, specs: Sequence[FileSpec]
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for spec in specs:
        source_path = _require_regular_file(source / spec.name)
        destination_path = _require_regular_file(
            destination / spec.name, immutable=True, single_link=True
        )
        source_stat = os.lstat(source_path)
        destination_stat = os.lstat(destination_path)
        _require(
            source_stat.st_size == destination_stat.st_size == spec.size_bytes
            and _sha256_file(source_path)
            == _sha256_file(destination_path)
            == spec.sha256,
            f"published byte identity mismatch: {spec.name}",
        )
        _require(
            source_stat.st_nlink == destination_stat.st_nlink == 1
            and (source_stat.st_dev, source_stat.st_ino)
            != (destination_stat.st_dev, destination_stat.st_ino),
            f"published hardlink detected: {spec.name}",
        )
        evidence.append(
            {
                "destination_device": destination_stat.st_dev,
                "destination_inode": destination_stat.st_ino,
                "destination_nlink": destination_stat.st_nlink,
                "name": spec.name,
                "sha256": spec.sha256,
                "size_bytes": spec.size_bytes,
                "source_device": source_stat.st_dev,
                "source_inode": source_stat.st_ino,
                "source_nlink": source_stat.st_nlink,
            }
        )
    return evidence


def _remove_copied_incomplete(stage: Path) -> None:
    marker = _require_regular_file(
        stage / "INCOMPLETE", immutable=True, single_link=True
    )
    _require(marker.parent == stage, "INCOMPLETE removal path mismatch")
    os.unlink(marker)
    _require(not os.path.lexists(marker), "copied INCOMPLETE marker remains")
    _fsync_directory(stage)
    _fsync_directory(stage.parent)


def _same_filesystem(stage: Path, target: Path) -> bool:
    stage = _require_directory(stage)
    parent = _require_directory(target.parent)
    return os.lstat(stage).st_dev == os.lstat(parent).st_dev


def _rename_noreplace(source: Path, target: Path) -> None:
    source = _require_directory(source)
    target = _require_no_symlink_components(target, allow_missing_leaf=True)
    _require(not os.path.lexists(target), "rename target exists before renameat2")
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    _require(function is not None, "libc renameat2 is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise SalvageError("target appeared during renameat2 no-replace")
        raise SalvageError(
            f"renameat2(RENAME_NOREPLACE) failed: {os.strerror(error_number)}"
        )


def _classify_crash_state(layout: Layout) -> str:
    record_exists = os.path.lexists(layout.record)
    target_exists = os.path.lexists(layout.target)
    stage_exists = os.path.lexists(layout.stage)
    if not record_exists and not target_exists and not stage_exists:
        return "fresh"
    if record_exists and not target_exists and stage_exists:
        return "resume_pending_rename"
    if record_exists and target_exists and not stage_exists:
        return "already_published"
    raise SalvageError(
        "manual quarantine required for invalid crash state: "
        f"record={int(record_exists)} target={int(target_exists)} "
        f"stage={int(stage_exists)}"
    )


def _validate_numeric_record(payload: Any) -> None:
    expected_keys = {
        "absolute_tolerance",
        "relative_tolerance",
        "matrix_from_hessian_max_abs_error",
        "matrix_symmetry_max_abs_error",
        "eigvalsh_matrix_max_abs_error",
        "low_basis_orthogonality_max_abs",
        "low_basis_eigen_residual_relative",
        "low_projector_basis_max_abs_error",
        "metric_closure_max_abs_error",
        "source_metric_max_abs_error",
        "spectrum_max_abs_error",
        "spectrum_bitwise_equal_count",
        "spectrum_representation_count",
        "rank_2_binary64_hex",
        "state_csv_rows",
        "spectrum_csv_rows",
        "scientific_success",
        "failed_scientific_success_gates",
        "final_exact_a_per_dim",
        "continuation_non_top_only_fraction",
        "total_non_top_only_fraction",
    }
    _require(
        isinstance(payload, dict) and set(payload) == expected_keys,
        "numeric audit schema mismatch",
    )
    _require(
        payload["absolute_tolerance"] == REPLAY_ATOL
        and payload["relative_tolerance"] == 0.0
        and payload["matrix_from_hessian_max_abs_error"] == 3.885780586188048e-16
        and payload["matrix_symmetry_max_abs_error"] == 0.0
        and payload["eigvalsh_matrix_max_abs_error"] == 6.661338147750939e-15
        and payload["low_basis_orthogonality_max_abs"] == 2.1163626406917047e-15
        and payload["low_basis_eigen_residual_relative"] == 1.4866823708986403e-15
        and payload["low_projector_basis_max_abs_error"] == 1.9984014443252818e-15
        and 0.0 <= payload["metric_closure_max_abs_error"] <= REPLAY_ATOL
        and 0.0 <= payload["source_metric_max_abs_error"] <= REPLAY_ATOL
        and payload["spectrum_max_abs_error"] == 0.0
        and payload["spectrum_bitwise_equal_count"] == DIMENSION
        and payload["spectrum_representation_count"] == 5
        and payload["rank_2_binary64_hex"] == "3de22a7f787e6c62"
        and payload["state_csv_rows"] == 101
        and payload["spectrum_csv_rows"] == 51_712
        and payload["scientific_success"] is False
        and payload["failed_scientific_success_gates"] == EXPECTED_FAILED_SUCCESS_GATES
        and payload["final_exact_a_per_dim"] == 0.9321672207645146
        and payload["continuation_non_top_only_fraction"] == 0.11301885339451084
        and payload["total_non_top_only_fraction"] == 0.07733408144475684,
        "numeric audit value mismatch",
    )


def _execution_freeze_lineage(
    layout: Layout, freeze: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = freeze["payload"]
    documents = {
        "derivation": {
            "path": str(layout.path(DERIVATION_REL)),
            "sha256": payload["derivation_sha256"],
        },
        "failed_manifest": {
            "path": str(layout.path(FAILED_MANIFEST_REL)),
            "sha256": payload["failed_manifest_sha256"],
        },
        "protocol": {
            "path": str(layout.path(PROTOCOL_REL)),
            "sha256": payload["protocol_sha256"],
        },
    }
    sources = {
        role: dict(payload[role])
        for role in ("publisher", "reviewer", "publisher_test", "reviewer_test")
    }
    return documents, sources


def _build_execution_record(
    layout: Layout,
    manifest: FailedManifest,
    freeze: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    protected_pre_copy: Mapping[str, Any],
    protected_post_copy: Mapping[str, Any],
    protected_prepublication: Mapping[str, Any],
    staged_with_marker: Mapping[str, Any],
    staged_ready: Mapping[str, Any],
    copy_audit: Mapping[str, Any],
    scientific: ScientificAudit,
    gates: Mapping[str, bool],
) -> dict[str, Any]:
    frozen_documents, frozen_sources = _execution_freeze_lineage(layout, freeze)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "validated_ready_for_atomic_publish",
        "created_at_utc": _utc_now(),
        "repository_root": str(layout.root),
        "protocol_sha256": PROTOCOL_SHA256,
        "failed_manifest_sha256": FAILED_MANIFEST_SHA256,
        "derivation_sha256": DERIVATION_SHA256,
        "execution_freeze_sha256": freeze["sha256"],
        "publisher_source_sha256": frozen_sources["publisher"]["raw_sha256"],
        "publisher_normalized_source_sha256": frozen_sources["publisher"][
            "normalized_sha256"
        ],
        "publisher_snapshot_sha256": snapshot["sha256"],
        "reviewer_source_sha256": frozen_sources["reviewer"]["raw_sha256"],
        "publisher_test_sha256": frozen_sources["publisher_test"]["sha256"],
        "reviewer_test_sha256": frozen_sources["reviewer_test"]["sha256"],
        "source": {
            "path": str(layout.failed),
            "pre_copy_audit": protected_pre_copy["failed"],
            "post_copy_audit": protected_post_copy["failed"],
            "prepublication_audit": protected_prepublication["failed"],
        },
        "staging": {
            "path": str(layout.stage),
            "copied_entry_count": 33,
            "copied_total_size_bytes": STAGED_TOTAL_SIZE,
            "copied_tree_sha256": STAGED_TREE_SHA256,
            "copied_audit": staged_with_marker,
            "incomplete_removed": True,
            "ready_entry_count": 32,
            "ready_total_size_bytes": PUBLISHED_TOTAL_SIZE,
            "ready_tree_sha256": PUBLISHED_TREE_SHA256,
            "ready_audit": staged_ready,
            "fsync_complete": True,
        },
        "publication": {
            "target": str(layout.target),
            "target_absent_pre_copy": True,
            "target_absent_prepublication": True,
            "method": "renameat2(RENAME_NOREPLACE)",
            "rename_flags": 1,
            "expected_entry_count": 32,
            "expected_total_size_bytes": PUBLISHED_TOTAL_SIZE,
            "expected_tree_sha256": PUBLISHED_TREE_SHA256,
        },
        "copy_audit": dict(copy_audit),
        "protected_lineage": {
            "failed_manifest_path": str(layout.path(FAILED_MANIFEST_REL)),
            "original_source_pre_copy": protected_pre_copy["original_source"],
            "original_source_prepublication": protected_prepublication[
                "original_source"
            ],
            "startup_failure_pre_copy": protected_pre_copy["startup_failure"],
            "startup_failure_prepublication": protected_prepublication[
                "startup_failure"
            ],
            "frozen_documents": frozen_documents,
            "frozen_sources": frozen_sources,
            "execution_freeze_path": str(layout.freeze),
            "publisher_snapshot_path": str(layout.snapshot),
        },
        "scientific_execution": dict(SCIENTIFIC_EXECUTION_ZERO),
        "numeric_audit": scientific.numeric_audit,
        "acceptance_gates": dict(gates),
    }


def _validate_record_tree_audit(
    payload: Any,
    specs: Sequence[FileSpec],
    *,
    path: Path,
    label: str,
) -> None:
    expected = {spec.name: spec for spec in specs}
    _require(
        isinstance(payload, dict)
        and set(payload)
        == {
            "entry_count",
            "files",
            "path",
            "regular_file_count",
            "total_size_bytes",
            "tree_sha256",
        }
        and payload["entry_count"] == len(specs)
        and payload["regular_file_count"] == len(specs)
        and payload["total_size_bytes"] == sum(spec.size_bytes for spec in specs)
        and payload["tree_sha256"] == _tree_hash(specs)
        and payload["path"] == str(path)
        and isinstance(payload["files"], list)
        and len(payload["files"]) == len(specs),
        f"{label} tree-audit schema mismatch",
    )
    names: set[str] = set()
    for item in payload["files"]:
        _require(
            isinstance(item, dict)
            and set(item)
            == {
                "device",
                "inode",
                "name",
                "nlink",
                "sha256",
                "size_bytes",
            },
            f"{label} tree-audit file schema mismatch",
        )
        name = item["name"]
        _require(
            isinstance(name, str)
            and name in expected
            and name not in names
            and _is_int(item["device"])
            and item["device"] >= 0
            and _is_int(item["inode"])
            and item["inode"] > 0
            and _is_int(item["nlink"])
            and item["nlink"] == 1
            and item["sha256"] == expected[name].sha256
            and _is_int(item["size_bytes"])
            and item["size_bytes"] == expected[name].size_bytes,
            f"{label} tree-audit file mismatch: {name!r}",
        )
        names.add(name)
    _require(names == set(expected), f"{label} tree-audit file set mismatch")


def _validate_execution_record(
    layout: Layout,
    manifest: FailedManifest,
    freeze: Mapping[str, Any],
    protected: Mapping[str, Any],
    destination: Path,
    destination_audit: Mapping[str, Any],
) -> dict[str, Any]:
    record_path = _require_regular_file(layout.record, immutable=True, single_link=True)
    record_bytes = _read_bytes(record_path)
    record = _strict_json_bytes(record_bytes, str(record_path))
    _require(
        record_bytes == _json_bytes(record),
        "execution record serialization is not canonical",
    )
    _require(set(record) == EXECUTION_RECORD_KEYS, "execution record schema mismatch")
    _validate_utc(record["created_at_utc"], "execution record timestamp")
    frozen_documents, frozen_sources = _execution_freeze_lineage(layout, freeze)
    expected_scalars = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "validated_ready_for_atomic_publish",
        "repository_root": str(layout.root),
        "protocol_sha256": PROTOCOL_SHA256,
        "failed_manifest_sha256": FAILED_MANIFEST_SHA256,
        "derivation_sha256": DERIVATION_SHA256,
        "execution_freeze_sha256": freeze["sha256"],
        "publisher_source_sha256": frozen_sources["publisher"]["raw_sha256"],
        "publisher_normalized_source_sha256": frozen_sources["publisher"][
            "normalized_sha256"
        ],
        "publisher_snapshot_sha256": frozen_sources["publisher"]["raw_sha256"],
        "reviewer_source_sha256": frozen_sources["reviewer"]["raw_sha256"],
        "publisher_test_sha256": frozen_sources["publisher_test"]["sha256"],
        "reviewer_test_sha256": frozen_sources["reviewer_test"]["sha256"],
    }
    for name, expected in expected_scalars.items():
        _require(record[name] == expected, f"execution record mismatch: {name}")
    snapshot = _require_regular_file(layout.snapshot, immutable=True, single_link=True)
    _require(
        _sha256_file(snapshot) == record["publisher_snapshot_sha256"]
        and _files_byte_identical(layout.path(PUBLISHER_SOURCE_REL), snapshot),
        "execution record publisher snapshot mismatch",
    )
    source = record["source"]
    _require(
        isinstance(source, dict)
        and set(source)
        == {"path", "pre_copy_audit", "post_copy_audit", "prepublication_audit"}
        and source["path"] == str(layout.failed)
        and source["pre_copy_audit"]
        == source["post_copy_audit"]
        == source["prepublication_audit"]
        == protected["failed"],
        "execution record source audits mismatch",
    )
    staging = record["staging"]
    _require(
        isinstance(staging, dict)
        and set(staging)
        == {
            "path",
            "copied_entry_count",
            "copied_total_size_bytes",
            "copied_tree_sha256",
            "copied_audit",
            "incomplete_removed",
            "ready_entry_count",
            "ready_total_size_bytes",
            "ready_tree_sha256",
            "ready_audit",
            "fsync_complete",
        }
        and staging["path"] == str(layout.stage)
        and staging["copied_entry_count"] == 33
        and staging["copied_total_size_bytes"] == STAGED_TOTAL_SIZE
        and staging["copied_tree_sha256"] == STAGED_TREE_SHA256
        and staging["incomplete_removed"] is True
        and staging["ready_entry_count"] == 32
        and staging["ready_total_size_bytes"] == PUBLISHED_TOTAL_SIZE
        and staging["ready_tree_sha256"] == PUBLISHED_TREE_SHA256
        and staging["fsync_complete"] is True,
        "execution record staging audit mismatch",
    )
    _validate_record_tree_audit(
        staging["copied_audit"],
        manifest.staged_specs,
        path=layout.stage,
        label="copied staging",
    )
    _validate_record_tree_audit(
        staging["ready_audit"],
        manifest.published_specs,
        path=layout.stage,
        label="ready staging",
    )
    ready_historical = dict(staging["ready_audit"])
    ready_historical["path"] = str(destination)
    _require(ready_historical == destination_audit, "ready stage/target audit changed")
    publication = record["publication"]
    _require(
        publication
        == {
            "target": str(layout.target),
            "target_absent_pre_copy": True,
            "target_absent_prepublication": True,
            "method": "renameat2(RENAME_NOREPLACE)",
            "rename_flags": 1,
            "expected_entry_count": 32,
            "expected_total_size_bytes": PUBLISHED_TOTAL_SIZE,
            "expected_tree_sha256": PUBLISHED_TREE_SHA256,
        },
        "execution record publication schema mismatch",
    )
    copy_audit = record["copy_audit"]
    _require(
        isinstance(copy_audit, dict)
        and set(copy_audit)
        == {
            "copied_names",
            "distinct_inode_count",
            "excluded_names",
            "hardlink_pair_count",
            "per_file",
            "removed_after_validation",
            "source_destination_sha256_equal",
            "source_destination_size_equal",
        }
        and copy_audit["copied_names"] == [spec.name for spec in manifest.staged_specs]
        and copy_audit["excluded_names"] == ["failure.json"]
        and copy_audit["removed_after_validation"] == ["INCOMPLETE"]
        and copy_audit["distinct_inode_count"] == 33
        and copy_audit["hardlink_pair_count"] == 0
        and copy_audit["source_destination_sha256_equal"] is True
        and copy_audit["source_destination_size_equal"] is True
        and isinstance(copy_audit["per_file"], list)
        and len(copy_audit["per_file"]) == 33,
        "execution record copy audit mismatch",
    )
    evidence_keys = {
        "destination_device",
        "destination_inode",
        "destination_nlink",
        "name",
        "sha256",
        "size_bytes",
        "source_device",
        "source_inode",
        "source_nlink",
    }
    per_file = copy_audit["per_file"]
    _require(
        all(
            isinstance(item, dict)
            and set(item) == evidence_keys
            and isinstance(item["name"], str)
            and _is_sha256(item["sha256"])
            and all(
                _is_int(item[name]) and item[name] >= 0
                for name in (
                    "destination_device",
                    "destination_inode",
                    "destination_nlink",
                    "size_bytes",
                    "source_device",
                    "source_inode",
                    "source_nlink",
                )
            )
            for item in per_file
        ),
        "copy evidence item schema mismatch",
    )
    current_evidence = _current_copy_evidence(
        layout.failed, destination, manifest.published_specs
    )
    record_evidence = {item["name"]: item for item in per_file}
    _require(
        len(record_evidence) == len(per_file)
        and set(record_evidence) == {spec.name for spec in manifest.staged_specs},
        "copy evidence filename set mismatch",
    )
    for current in current_evidence:
        _require(
            record_evidence[current["name"]] == current,
            f"copy inode/hash evidence changed: {current['name']}",
        )
    incomplete = record_evidence["INCOMPLETE"]
    incomplete_spec = next(
        spec for spec in manifest.staged_specs if spec.name == "INCOMPLETE"
    )
    source_incomplete = os.lstat(layout.failed / "INCOMPLETE")
    _require(
        incomplete["name"] == "INCOMPLETE"
        and incomplete["sha256"] == incomplete_spec.sha256
        and incomplete["size_bytes"] == incomplete_spec.size_bytes
        and incomplete["source_device"] == source_incomplete.st_dev
        and incomplete["source_inode"] == source_incomplete.st_ino
        and incomplete["source_nlink"] == 1
        and incomplete["destination_nlink"] == 1
        and (
            incomplete["source_device"],
            incomplete["source_inode"],
        )
        != (
            incomplete["destination_device"],
            incomplete["destination_inode"],
        ),
        "removed INCOMPLETE copy evidence mismatch",
    )
    lineage = record["protected_lineage"]
    _require(
        isinstance(lineage, dict)
        and set(lineage)
        == {
            "failed_manifest_path",
            "original_source_pre_copy",
            "original_source_prepublication",
            "startup_failure_pre_copy",
            "startup_failure_prepublication",
            "frozen_documents",
            "frozen_sources",
            "execution_freeze_path",
            "publisher_snapshot_path",
        }
        and lineage["failed_manifest_path"] == str(layout.path(FAILED_MANIFEST_REL))
        and lineage["original_source_pre_copy"]
        == lineage["original_source_prepublication"]
        == protected["original_source"]
        and lineage["startup_failure_pre_copy"]
        == lineage["startup_failure_prepublication"]
        == protected["startup_failure"]
        and lineage["frozen_documents"] == frozen_documents
        and lineage["frozen_sources"] == frozen_sources
        and lineage["execution_freeze_path"] == str(layout.freeze)
        and lineage["publisher_snapshot_path"] == str(layout.snapshot),
        "execution record protected lineage mismatch",
    )
    _require(
        record["scientific_execution"] == SCIENTIFIC_EXECUTION_ZERO,
        "execution record scientific counters changed",
    )
    _validate_numeric_record(record["numeric_audit"])
    gates = record["acceptance_gates"]
    _require(
        isinstance(gates, dict)
        and set(gates) == set(ACCEPTANCE_GATE_NAMES)
        and all(value is True for value in gates.values()),
        "execution record acceptance gate map mismatch",
    )
    return record


def _log(stage: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    print(f"[salvage-publisher] stage={stage}{suffix}", flush=True)


def _record_gates(
    scientific: ScientificAudit,
    *,
    protected_unchanged: bool,
) -> dict[str, bool]:
    values = {
        "failed_tree_exact": True,
        "original_source_tree_exact": True,
        "startup_failure_tree_exact": True,
        "publisher_source_frozen": True,
        "protocol_and_manifests_frozen": True,
        **scientific.gates,
        "stage_path_exclusive": True,
        "target_absent": True,
        "record_path_exclusive": True,
        "same_filesystem": True,
        "byte_copy_exact": True,
        "hardlinks_absent": True,
        "staged_tree_exact": True,
        "protected_trees_unchanged_prepublication": protected_unchanged,
        "cuda_uninitialized": _cuda_uninitialized(),
        "scientific_execution_absent": True,
    }
    _require(set(values) == set(ACCEPTANCE_GATE_NAMES), "publisher gate set mismatch")
    ordered = {name: values[name] for name in ACCEPTANCE_GATE_NAMES}
    _require(all(ordered.values()), "publisher acceptance gate failed")
    return ordered


def _postpublication_validate(
    layout: Layout,
    manifest: FailedManifest,
    freeze: Mapping[str, Any],
    *,
    publisher_source: Path,
) -> dict[str, Any]:
    _require(not os.path.lexists(layout.stage), "stage remains after publication")
    target_audit = _audit_tree(
        layout.target,
        manifest.published_specs,
        expected_tree_sha256=PUBLISHED_TREE_SHA256,
        expected_total_size=PUBLISHED_TOTAL_SIZE,
        label="published target",
    )
    protected = _audit_protected_trees(layout, manifest)
    _verify_self_freeze(publisher_source)
    record = _validate_execution_record(
        layout,
        manifest,
        freeze,
        protected,
        layout.target,
        target_audit,
    )
    _require_cuda_uninitialized()
    return {
        "execution_record_sha256": _sha256_file(layout.record),
        "published_tree_sha256": target_audit["tree_sha256"],
        "record_status": record["status"],
        "target": str(layout.target),
    }


def _resume_pending_rename(
    layout: Layout,
    manifest: FailedManifest,
    freeze: Mapping[str, Any],
    *,
    publisher_source: Path,
    fault_hook: Callable[[str], None] | None,
    rename_impl: Callable[[Path, Path], None],
) -> dict[str, Any]:
    _log("resume-validation", f"stage={layout.stage}")
    _require_absent(layout.target, "publication target")
    protected = _audit_protected_trees(layout, manifest)
    stage_audit = _audit_tree(
        layout.stage,
        manifest.published_specs,
        expected_tree_sha256=PUBLISHED_TREE_SHA256,
        expected_total_size=PUBLISHED_TOTAL_SIZE,
        label="resumed staging tree",
    )
    _require(
        _same_filesystem(layout.stage, layout.target),
        "stage/target filesystem mismatch",
    )
    if fault_hook is not None:
        fault_hook("before_rename")
        protected = _audit_protected_trees(layout, manifest)
        stage_audit = _audit_tree(
            layout.stage,
            manifest.published_specs,
            expected_tree_sha256=PUBLISHED_TREE_SHA256,
            expected_total_size=PUBLISHED_TOTAL_SIZE,
            label="resumed staging tree after fault hook",
        )
    _validate_execution_record(
        layout,
        manifest,
        freeze,
        protected,
        layout.stage,
        stage_audit,
    )
    _require_absent(layout.target, "publication target")
    _verify_self_freeze(publisher_source)
    freeze = _validate_execution_freeze(layout, publisher_source=publisher_source)
    _require_cuda_uninitialized()
    _log("atomic-publish", "method=renameat2(RENAME_NOREPLACE) resume=1")
    rename_impl(layout.stage, layout.target)
    _fsync_directory(layout.target.parent)
    if fault_hook is not None:
        fault_hook("after_rename")
    result = _postpublication_validate(
        layout, manifest, freeze, publisher_source=publisher_source
    )
    result["status"] = "resumed_publish"
    return result


def publish_salvage(
    layout: Layout = PRODUCTION_LAYOUT,
    *,
    publisher_source: Path = LIVE_SOURCE,
    fault_hook: Callable[[str], None] | None = None,
    rename_impl: Callable[[Path, Path], None] = _rename_noreplace,
    scientific_auditor: Callable[..., ScientificAudit] = audit_failed_packet_read_only,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    layout = Layout.for_root(layout.root, dependency_root=layout.dependency_root)
    publisher_source = _absolute(publisher_source)
    _require_under_root(layout.root, publisher_source)
    _require_directory(layout.root)
    _require_directory(layout.dependency_root)
    _require_cuda_uninitialized()
    state = _classify_crash_state(layout)
    _log(
        "startup",
        f"protocol={PROTOCOL_ID} state={state} device=cpu dtype=torch.float64 "
        f"seed=none cache_mode=read-only-byte-salvage",
    )
    _log(
        "config",
        f"root={layout.root} source={layout.failed} stage={layout.stage} "
        f"target={layout.target} record={layout.record}",
    )
    manifest = _load_failed_manifest(layout)
    _log("source-freeze", f"freeze={layout.freeze}")
    freeze = _validate_execution_freeze(layout, publisher_source=publisher_source)
    if state == "already_published":
        _log("postpublication-read-only", f"target={layout.target}")
        result = _postpublication_validate(
            layout, manifest, freeze, publisher_source=publisher_source
        )
        result["status"] = "already_published"
        _log(
            "complete",
            f"status=already_published target={layout.target} "
            f"record={layout.record} tree={result['published_tree_sha256']}",
        )
        return result
    if state == "resume_pending_rename":
        result = _resume_pending_rename(
            layout,
            manifest,
            freeze,
            publisher_source=publisher_source,
            fault_hook=fault_hook,
            rename_impl=rename_impl,
        )
        _log(
            "complete",
            f"status=resumed_publish target={layout.target} record={layout.record} "
            f"tree={result['published_tree_sha256']}",
        )
        return result
    _require(state == "fresh", "unreachable publication state")
    _require_absent(layout.stage, "staging path")
    _require_absent(layout.target, "publication target")
    _require_absent(layout.record, "execution record")
    protected_pre_copy = _audit_protected_trees(layout, manifest)
    _log("scientific-preflight", "mode=cpu-read-only")
    scientific = scientific_auditor(
        layout,
        manifest=manifest,
        protected=protected_pre_copy,
    )
    _require_absent(layout.stage, "staging path")
    _require_absent(layout.target, "publication target")
    _require_absent(layout.record, "execution record")
    _verify_self_freeze(publisher_source)
    freeze = _validate_execution_freeze(layout, publisher_source=publisher_source)
    snapshot = _create_or_verify_snapshot(publisher_source, layout.snapshot)
    if fault_hook is not None:
        fault_hook("after_snapshot")
    _log("byte-copy", f"files={len(manifest.staged_specs)}")
    copy_audit = _copy_projection(
        layout.failed,
        layout.stage,
        manifest.staged_specs,
        fault_hook=fault_hook,
    )
    staged_with_marker = _audit_tree(
        layout.stage,
        manifest.staged_specs,
        expected_tree_sha256=STAGED_TREE_SHA256,
        expected_total_size=STAGED_TOTAL_SIZE,
        label="staged projection with marker",
    )
    protected_post_copy = _audit_protected_trees(layout, manifest)
    _require(
        protected_post_copy == protected_pre_copy,
        "protected tree changed during byte copy",
    )
    if fault_hook is not None:
        fault_hook("after_staged_validation")
    _remove_copied_incomplete(layout.stage)
    if fault_hook is not None:
        fault_hook("after_incomplete_removed")
    staged_ready = _audit_tree(
        layout.stage,
        manifest.published_specs,
        expected_tree_sha256=PUBLISHED_TREE_SHA256,
        expected_total_size=PUBLISHED_TOTAL_SIZE,
        label="ready staged projection",
    )
    _require_absent(layout.target, "publication target")
    _require_absent(layout.record, "execution record")
    _require(
        _same_filesystem(layout.stage, layout.target),
        "stage/target filesystem mismatch",
    )
    protected_prepublication = _audit_protected_trees(layout, manifest)
    protected_unchanged = (
        protected_prepublication == protected_post_copy == protected_pre_copy
    )
    _require(protected_unchanged, "protected trees changed before publication")
    current_copy = _current_copy_evidence(
        layout.failed, layout.stage, manifest.published_specs
    )
    historical_by_name = {item["name"]: item for item in copy_audit["per_file"]}
    _require(
        all(historical_by_name[item["name"]] == item for item in current_copy),
        "staged copy inode evidence changed",
    )
    _verify_self_freeze(publisher_source)
    freeze = _validate_execution_freeze(layout, publisher_source=publisher_source)
    snapshot = _create_or_verify_snapshot(publisher_source, layout.snapshot)
    gates = _record_gates(
        scientific,
        protected_unchanged=protected_unchanged,
    )
    record = _build_execution_record(
        layout,
        manifest,
        freeze,
        snapshot,
        protected_pre_copy,
        protected_post_copy,
        protected_prepublication,
        staged_with_marker,
        staged_ready,
        copy_audit,
        scientific,
        gates,
    )
    _log("publication-intent", f"record={layout.record}")
    _create_exclusive_file(
        layout.record,
        _json_bytes(record),
        label="execution record",
    )
    _validate_execution_record(
        layout,
        manifest,
        freeze,
        protected_prepublication,
        layout.stage,
        staged_ready,
    )
    if fault_hook is not None:
        fault_hook("after_record_created")
        fault_hook("before_rename")
        protected_final = _audit_protected_trees(layout, manifest)
        staged_final = _audit_tree(
            layout.stage,
            manifest.published_specs,
            expected_tree_sha256=PUBLISHED_TREE_SHA256,
            expected_total_size=PUBLISHED_TOTAL_SIZE,
            label="ready staged projection after fault hook",
        )
        _validate_execution_record(
            layout,
            manifest,
            freeze,
            protected_final,
            layout.stage,
            staged_final,
        )
    _require_absent(layout.target, "publication target")
    _verify_self_freeze(publisher_source)
    freeze = _validate_execution_freeze(layout, publisher_source=publisher_source)
    _require_cuda_uninitialized()
    _log("atomic-publish", "method=renameat2(RENAME_NOREPLACE) resume=0")
    rename_impl(layout.stage, layout.target)
    _fsync_directory(layout.target.parent)
    if fault_hook is not None:
        fault_hook("after_rename")
    result = _postpublication_validate(
        layout, manifest, freeze, publisher_source=publisher_source
    )
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    result["status"] = "published"
    result["elapsed_seconds"] = elapsed
    _log(
        "complete",
        f"status=published elapsed_seconds={elapsed:.3f} target={layout.target} "
        f"record={layout.record} snapshot={layout.snapshot} "
        f"tree={result['published_tree_sha256']} scientific_success=0",
    )
    return result


def main() -> None:
    _require(len(sys.argv) == 1, "production CLI accepts no arguments or overrides")
    _require(
        LIVE_SOURCE == PRODUCTION_ROOT / PUBLISHER_SOURCE_REL,
        "publisher path is not frozen",
    )
    _require(PRODUCTION_LAYOUT.root == PRODUCTION_ROOT, "production root is not frozen")
    publish_salvage()


if __name__ == "__main__":
    main()
