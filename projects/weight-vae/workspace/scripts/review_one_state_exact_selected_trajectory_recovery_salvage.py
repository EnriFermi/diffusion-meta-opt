from __future__ import annotations

import ast
import binascii
import csv
import ctypes
import datetime as dt
import decimal
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


# This must be set before torch is imported. The reviewer has no accelerator path.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


ROOT = Path("/home/coder/project")
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
CONTROL_DIR = OUTPUT_ROOT / "postgoal_relaxed_cancellation"

PROTOCOL = CONTROL_DIR / "salvage_protocol.md"
FAILED_MANIFEST = CONTROL_DIR / "salvage_frozen_failed_manifest.json"
DERIVATION = CONTROL_DIR / "salvage_frozen_failed_manifest_derivation.md"
EXECUTION_FREEZE = CONTROL_DIR / "salvage_execution_freeze.json"
EXECUTION_RECORD = CONTROL_DIR / "salvage_execution_record.json"
REVIEW_RECORD_TEMP = CONTROL_DIR / ".salvage_independent_review.json.incomplete"
REVIEW_RECORD = CONTROL_DIR / "salvage_independent_review.json"
PUBLISHER_SNAPSHOT = CONTROL_DIR / "salvage_publisher_source_snapshot.py"

FAILED_SOURCE = OUTPUT_ROOT / (
    "postgoal_relaxed_cancellation_recovery_finalization_v1.failed."
    "20260716T074002.762548Z.2787575"
)
STARTUP_FAILURE = OUTPUT_ROOT / (
    "postgoal_relaxed_cancellation_recovery_finalization_v1.failed."
    "20260716T070631.160509Z.2760686"
)
ORIGINAL_SOURCE = OUTPUT_ROOT / "postgoal_relaxed_cancellation_production.incomplete"
STAGE = OUTPUT_ROOT / (
    ".postgoal_relaxed_cancellation_recovery_finalization_v1.salvage.incomplete"
)
TARGET = OUTPUT_ROOT / "postgoal_relaxed_cancellation_recovery_finalization_v1"

PUBLISHER_SOURCE = (
    ROOT / "scripts/publish_one_state_exact_selected_trajectory_recovery_salvage.py"
)
REVIEWER_SOURCE = Path(__file__)
PUBLISHER_TEST = (
    ROOT / "tests/one_state_exact_selected_trajectory_recovery_salvage_publish_test.py"
)
REVIEWER_TEST = (
    ROOT / "tests/one_state_exact_selected_trajectory_recovery_salvage_review_test.py"
)

RECOVERY_INPUT_MANIFEST = CONTROL_DIR / "recovery_frozen_input_manifest.json"
RECOVERY_TASK_MANIFEST = CONTROL_DIR / "recovery_task_manifest.json"
RECOVERY_IMPORT_MANIFEST = CONTROL_DIR / "recovery_import_manifest.json"
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
ACCEPTED_RUN = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)

PROTOCOL_SHA256 = "7b457796d930111ca1c568266ee002d92588d0cacc64d8ad8abc023617150141"
FAILED_MANIFEST_SHA256 = (
    "944f0683383abf8923cc5fc7eb19421cd8c75c325a8874b80af8f210b1b80fec"
)
DERIVATION_SHA256 = "8550adfbcfd9dfe366c7876f330d71037462cc1854589a61337078c7b6d06e9d"
RECOVERY_INPUT_MANIFEST_SHA256 = (
    "8d7f5a72092de0fbaffe67a1c2421dfcb227bf412690f470d552ab0069b99f66"
)
RECOVERY_PROTOCOL_SHA256 = (
    "e7f941d8a6632e71ffd09ad9c0c5a3ed11490a3f8fadd3d7f9b1f8001d5108c1"
)
RECOVERY_TASK_MANIFEST_SHA256 = (
    "a4b0b4bc1b50d04e2cd6d6ab35fe05a8200caa222cf84228eaee894504a8054b"
)
RECOVERY_IMPORT_MANIFEST_SHA256 = (
    "976d35e42c4fa491324660e044a4bb6ac224786108e2e81effcef5168753da10"
)
SOURCE_PROGRESS_SHA256 = (
    "ab3366870d62beb71b4d59fd2d54e3cb7354963a16e17a8b9cc13527a4d25311"
)
PARENT_PROGRESS_SHA256 = (
    "010d34d51c21ec139fca3e93f51b91c7563f43a524a59b521dcec50de9c752ce"
)
PARENT_FINAL_SHA256 = "7063f393332f2640d599fa41928006f74342c3d681dd9b77a575e5e9e40716e5"
PARENT_REVIEW_SHA256 = (
    "7677a64b38755bed2d553996f103d943ac7d43c7dae748f546e93ba5441b9703"
)
ACCEPTED_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
ACTIVE_PARAMETER_SHA256 = (
    "b4d22a52fdb07a51978d466ef0841bfa92b33a625c84e32433ccb6b334039427"
)
TRANSITION_CHAIN_SHA256 = (
    "8ebc37fa89b471a722844d553b2a5a70003cd44b3cda3ce987075a6df8e261fe"
)
Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
FINALIZER_SHA256 = "5407f712f46090eeeed004a76dea1ea10b250398171575586c1c6737b145b325"
FINALIZER_NORMALIZED_SHA256 = (
    "e25be1074055a66ad68c63acc1f722191efbbf8a827ecd73adfeaa090329a599"
)

PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_no_replay_salvage_v1"
REVIEW_PROTOCOL_ID = (
    "one_state_exact_selected_relaxed_cancellation_no_replay_salvage_cpu_review_v1"
)
RECOVERY_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_recovery_v1"
SOURCE_PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_v1"

# This is the only line replaced by normalized_source_sha256().
EXPECTED_NORMALIZED_SOURCE_SHA256 = "9fb5f0828c403a487f7412651c3de128654df65d097fe948529b72331e2272b0"  # fmt: skip

DIMENSION = 512
LOW_COUNT = 451
START_UPDATE = 17
ACCEPTED_UPDATES = 100
NEW_ACCEPTED_UPDATES = 83
ACTIVE_TENSOR_COUNT = 31
ACTIVE_PARAMETER_COUNT = 11_685_120
SOURCE_WEIGHT_INDEX = 378
TASK_NAME = "fashion_mnist"
TASK_TAU = 1.1614345407370807
EPSILON = 1e-4
OLD_BETA = 22.536727828943093
LOW_THRESHOLD = 0.1
TARGET_NORM = 0.04892722657548397
ABSOLUTE_TOLERANCE = 1e-9
RADIUS_TOLERANCE = 5e-9
ARMIJO_C1 = 1e-4
LINE_ALPHAS = tuple(2.0**-index for index in range(12))

FAILED_TREE_SHA256 = "7dc8fc68a0e8e8baa1140b43f7fbfb5bf9fc129b9f5929403419c54bd5b92c82"
STAGED_TREE_SHA256 = "b73fceb960f0bec4c0f5ffc308da0bfe90d45d3d6b9d01ffb20d5554480e0514"
PUBLISHED_TREE_SHA256 = (
    "d3d11880b2909bb1cd2681bdc10b840e8b43b9aa84af49ba4f41cedacba5daf7"
)
ORIGINAL_TREE_SHA256 = (
    "813c191f276dc72ab0816899d1f4d00a73798b4d76ddcadf8f7e2ba2ae13a6b2"
)
STARTUP_TREE_SHA256 = "8d5a465e9c44a3dcb02246b544fddd77b777b6d30b29c40aef343c16ee17fa76"

REVIEW_KEYS = {
    "schema_version",
    "review_protocol_id",
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
    "reviewer_normalized_source_sha256",
    "publisher_test_sha256",
    "reviewer_test_sha256",
    "execution_record_sha256",
    "failed_tree_sha256",
    "published_tree_sha256",
    "reviewed_output",
    "source_staging",
    "valid",
    "recovery_valid",
    "scientific_success",
    "failed_scientific_success_gates",
    "read_only",
    "acceptance_gates",
    "numeric_audit",
    "byte_copy_audit",
    "protected_tree_audits",
    "errors",
    "limitations",
}

REVIEW_GATES = {
    "reviewer_source_frozen",
    "protocol_and_manifests_frozen",
    "execution_freeze_exact",
    "execution_record_exact",
    "failed_tree_exact",
    "original_source_tree_exact",
    "startup_failure_tree_exact",
    "published_tree_exact",
    "byte_identity_exact",
    "hardlinks_absent",
    "artifact_hash_graph_exact",
    "checkpoint_lineage_exact",
    "task_and_input_provenance_exact",
    "model_nonmutation_exact",
    "geometry_closure",
    "metric_closure",
    "spectrum_bitwise_closure",
    "csv_checkpoint_closure",
    "scientific_outcome_unchanged",
    "publisher_static_policy",
    "reviewer_static_policy",
    "protected_trees_read_only",
    "cuda_uninitialized",
    "scientific_execution_absent",
}

PUBLISHER_GATES = {
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
}

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

FREEZE_PATHS = {
    "failed_source": str(FAILED_SOURCE),
    "stage": str(STAGE),
    "target": str(TARGET),
    "publisher_snapshot": str(PUBLISHER_SNAPSHOT),
    "execution_record": str(EXECUTION_RECORD),
    "review_record_temp": str(REVIEW_RECORD_TEMP),
    "review_record": str(REVIEW_RECORD),
}

EXPECTED_RUNTIME = {
    "python": "3.12.12",
    "torch": "2.10.0+cu128",
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
FORBIDDEN_POLICY_TOKENS = [
    "import" + "lib",
    "run" + "py",
    "sub" + "process",
    "ex" + "ec(",
    "ev" + "al(",
    "_load" + "_run",
    "_eval" + "uate",
    "torch.auto" + "grad",
    "torch.op" + "tim",
    "torch.cu" + "da.",
    "back" + "ward(",
    ".gr" + "ad(",
]

FAILED_SCIENTIFIC_GATES = [
    "final_a_at_most_0p90",
    "non_top_only_fraction_at_least_0p25",
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

NUMERIC_AUDIT_KEYS = {
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


class SalvageReviewError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise SalvageReviewError(message)


def absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _path_components(path: Path) -> list[Path]:
    path = absolute(path)
    current = Path(path.anchor)
    components: list[Path] = []
    for part in path.parts[1:]:
        current /= part
        components.append(current)
    return components


def check_path(
    path: Path,
    *,
    root: Path = ROOT,
    allow_missing_leaf: bool = False,
) -> Path:
    path = absolute(path)
    root = absolute(root)
    require(
        path == root or root in path.parents, f"path escapes repository root: {path}"
    )
    components = _path_components(path)
    for position, component in enumerate(components):
        leaf = position == len(components) - 1
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            if allow_missing_leaf and leaf:
                return path
            raise SalvageReviewError(f"missing path component: {component}") from None
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink component: {component}")
    return path


def regular_file(path: Path, *, root: Path = ROOT) -> Path:
    path = check_path(path, root=root)
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode), f"not a regular file: {path}")
    return path


def directory(path: Path, *, root: Path = ROOT) -> Path:
    path = check_path(path, root=root)
    metadata = os.lstat(path)
    require(stat.S_ISDIR(metadata.st_mode), f"not a directory: {path}")
    return path


def _open_read(path: Path, *, root: Path = ROOT) -> io.BufferedReader:
    path = regular_file(path, root=root)
    before = os.lstat(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    after = os.fstat(descriptor)
    try:
        require(stat.S_ISREG(after.st_mode), f"opened non-regular file: {path}")
        require(
            (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
            f"file identity changed while opening: {path}",
        )
    except Exception:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "rb")


def read_bytes(path: Path, *, root: Path = ROOT) -> bytes:
    with _open_read(path, root=root) as handle:
        return handle.read()


def sha256_file(path: Path, *, root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    with _open_read(path, root=root) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> Any:
    raise SalvageReviewError(f"non-finite JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return text.encode("utf-8")


def load_json(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    raw = read_bytes(path, root=root)
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SalvageReviewError(f"invalid JSON {path}: {error}") from None
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_checkpoint(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    with _open_read(path, root=root) as handle:
        value = torch.load(handle, map_location="cpu", weights_only=True)
    require(isinstance(value, dict), f"checkpoint root is not a mapping: {path}")
    return value


def normalized_source_sha256(path: Path = REVIEWER_SOURCE, *, root: Path = ROOT) -> str:
    prefix = "EXPECTED_NORMALIZED_SOURCE_SHA256 = "
    source = read_bytes(path, root=root).decode("utf-8")
    normalized: list[str] = []
    masked = 0
    for line in source.splitlines(keepends=True):
        if line.startswith(prefix):
            normalized.append(f'{prefix}"<FROZEN>"\n')
            masked += 1
        else:
            normalized.append(line)
    require(masked == 1, f"self normalization masked {masked} lines")
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    now = dt.datetime.now(dt.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def validate_timestamp(value: Any, *, field: str) -> None:
    require(isinstance(value, str), f"{field} is not text")
    require(value.endswith("Z"), f"{field} is not UTC")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise SalvageReviewError(f"{field} is not ISO-8601") from None
    require(parsed.tzinfo == dt.UTC, f"{field} timezone mismatch")


def tensor_bytes_sha256(value: torch.Tensor) -> str:
    require(isinstance(value, torch.Tensor), "value is not a tensor")
    require(value.device.type == "cpu", "tensor is not on CPU")
    require(value.layout == torch.strided, "tensor layout is not strided")
    require(not value.is_sparse, "sparse tensor is forbidden")
    require(not value.requires_grad, "stored tensor unexpectedly tracks derivatives")
    contiguous = value.detach().contiguous()
    size = contiguous.numel() * contiguous.element_size()
    digest = hashlib.sha256()
    address = contiguous.data_ptr()
    for offset in range(0, size, 1024 * 1024):
        count = min(1024 * 1024, size - offset)
        digest.update(ctypes.string_at(address + offset, count))
    return digest.hexdigest()


def tensor_fingerprint(value: torch.Tensor) -> dict[str, Any]:
    require(isinstance(value, torch.Tensor), "fingerprint value is not a tensor")
    require(value.device.type == "cpu", "fingerprint tensor is not on CPU")
    require(bool(torch.isfinite(value).all()), "fingerprint tensor is non-finite")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": value.numel(),
        "sha256": tensor_bytes_sha256(value),
    }


def named_tensor_sha256(values: Mapping[str, Any]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name in sorted(values):
        value = values[name]
        require(isinstance(name, str), "tensor name is not text")
        require(isinstance(value, torch.Tensor), f"{name} is not a tensor")
        require(value.device.type == "cpu", f"{name} is not on CPU")
        require(bool(torch.isfinite(value).all()), f"{name} is non-finite")
        digest.update(name.encode("utf-8"))
        contiguous = value.detach().contiguous()
        size = contiguous.numel() * contiguous.element_size()
        address = contiguous.data_ptr()
        for offset in range(0, size, 1024 * 1024):
            length = min(1024 * 1024, size - offset)
            digest.update(ctypes.string_at(address + offset, length))
        count += contiguous.numel()
    return digest.hexdigest(), count


def float_bits(value: Any) -> bytes:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise SalvageReviewError(f"invalid binary64 value: {value!r}") from None
    require(math.isfinite(parsed), f"non-finite binary64 value: {value!r}")
    return struct.pack(">d", parsed)


def finite_float(value: Any) -> float:
    float_bits(value)
    return float(value)


def exact_int(value: Any) -> int:
    if isinstance(value, bool):
        raise SalvageReviewError(f"boolean is not an integer field: {value!r}")
    if isinstance(value, int):
        return value
    try:
        number = decimal.Decimal(str(value))
    except decimal.InvalidOperation:
        raise SalvageReviewError(f"invalid integer field: {value!r}") from None
    require(
        number.is_finite() and number == number.to_integral_value(), "nonintegral field"
    )
    if number.is_zero():
        require(not number.is_signed(), "negative zero in integer field")
    return int(number)


def same_float(left: Any, right: Any) -> bool:
    try:
        return float_bits(left) == float_bits(right)
    except SalvageReviewError:
        return False


def close(left: Any, right: Any, *, atol: float = ABSOLUTE_TOLERANCE) -> bool:
    try:
        return abs(finite_float(left) - finite_float(right)) <= atol
    except SalvageReviewError:
        return False


def deep_exact(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left is right
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and tensor_bytes_sha256(left) == tensor_bytes_sha256(right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            deep_exact(left[k], right[k]) for k in left
        )
    sequence_types = (list, tuple)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if not isinstance(left, sequence_types) or not isinstance(
            right, sequence_types
        ):
            return False
        return len(left) == len(right) and all(
            deep_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) or isinstance(right, float):
        return (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and same_float(left, right)
        )
    return type(left) is type(right) and left == right


def deep_close(left: Any, right: Any, *, atol: float = ABSOLUTE_TOLERANCE) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left is right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            deep_close(left[k], right[k], atol=atol) for k in left
        )
    sequence_types = (list, tuple)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if not isinstance(left, sequence_types) or not isinstance(
            right, sequence_types
        ):
            return False
        return len(left) == len(right) and all(
            deep_close(a, b, atol=atol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    try:
        return close(left, right, atol=atol)
    except (TypeError, ValueError):
        return False


def _tree_line(name: str, digest: str, size: int) -> bytes:
    value = {"name": name, "sha256": digest, "size_bytes": size}
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )


def audit_tree(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    path = directory(path, root=root)
    entries = sorted(os.scandir(path), key=lambda entry: entry.name.encode("utf-8"))
    files: dict[str, dict[str, Any]] = {}
    serialized = hashlib.sha256()
    total = 0
    for entry in entries:
        child = path / entry.name
        metadata = os.lstat(child)
        require(not entry.is_symlink(), f"tree entry is a symlink: {child}")
        require(stat.S_ISREG(metadata.st_mode), f"tree entry is not regular: {child}")
        digest = sha256_file(child, root=root)
        size = metadata.st_size
        serialized.update(_tree_line(entry.name, digest, size))
        total += size
        files[entry.name] = {
            "sha256": digest,
            "size_bytes": size,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "link_count": metadata.st_nlink,
            "mode": stat.S_IMODE(metadata.st_mode),
            "mtime_ns": metadata.st_mtime_ns,
        }
    return {
        "path": str(path),
        "file_count": len(files),
        "total_size_bytes": total,
        "tree_sha256": serialized.hexdigest(),
        "files": files,
    }


def audit_tree_against_entries(
    path: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_tree_sha256: str,
    expected_total_size: int,
    root: Path = ROOT,
) -> dict[str, Any]:
    require(isinstance(entries, Sequence), "manifest file list is not a sequence")
    expected: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        require(
            isinstance(entry, Mapping)
            and set(entry).issuperset({"name", "size_bytes", "sha256"}),
            "malformed tree manifest entry",
        )
        name = entry["name"]
        require(isinstance(name, str) and name and "/" not in name, "invalid file name")
        require(name not in expected, f"duplicate manifest file: {name}")
        require(is_sha256(entry["sha256"]), f"invalid hash for {name}")
        require(
            type(entry["size_bytes"]) is int and entry["size_bytes"] >= 0,
            "invalid size",
        )
        expected[name] = entry
    observed = audit_tree(path, root=root)
    require(set(observed["files"]) == set(expected), "tree file set mismatch")
    for name, entry in expected.items():
        actual = observed["files"][name]
        require(actual["size_bytes"] == entry["size_bytes"], f"size mismatch: {name}")
        require(actual["sha256"] == entry["sha256"], f"hash mismatch: {name}")
    require(
        observed["total_size_bytes"] == expected_total_size, "tree total size mismatch"
    )
    require(observed["tree_sha256"] == expected_tree_sha256, "tree hash mismatch")
    return observed


def _expected_csv_header(records: Sequence[Mapping[str, Any]]) -> list[str]:
    header: list[str] = []
    seen: set[str] = set()
    for record in records:
        require(isinstance(record, Mapping), "checkpoint row is not a mapping")
        for key in record:
            require(isinstance(key, str), "checkpoint column is not text")
            if key not in seen:
                seen.add(key)
                header.append(key)
    return header


def _csv_cell_matches(expected: Any, observed: str) -> bool:
    if expected is None:
        return observed == ""
    if isinstance(expected, bool):
        return observed == ("True" if expected else "False")
    if isinstance(expected, int):
        try:
            return exact_int(observed) == expected
        except SalvageReviewError:
            return False
    if isinstance(expected, float):
        return same_float(expected, observed)
    if isinstance(expected, str):
        return observed == expected
    return False


def validate_rows_csv(
    records: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    raw = read_bytes(path, root=root)
    require(raw, f"empty CSV: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SalvageReviewError(f"CSV is not UTF-8: {path}") from None
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    header = reader.fieldnames
    expected_header = _expected_csv_header(records)
    require(header is not None, f"CSV has no header: {path}")
    require(len(header) == len(set(header)), f"duplicate CSV header: {path}")
    require(header == expected_header, f"CSV header mismatch: {path.name}")
    observed_rows = list(reader)
    require(len(observed_rows) == len(records), f"CSV row count mismatch: {path.name}")
    for position, (expected, observed) in enumerate(
        zip(records, observed_rows, strict=True)
    ):
        require(None not in observed, f"extra CSV fields at row {position}")
        require(set(observed) == set(header), f"CSV row schema mismatch at {position}")
        for column in header:
            require(
                _csv_cell_matches(expected.get(column), observed[column]),
                f"CSV/checkpoint mismatch {path.name} row {position} column {column}",
            )
    return {
        "path": str(absolute(path)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "header": header,
        "row_count": len(observed_rows),
        "binary64_exact": True,
    }


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.AST = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _expression_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _static_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def audit_static_source(
    path: Path,
    policy: Mapping[str, Any],
    *,
    expected_raw: str,
    expected_normalized: str | None,
    root: Path = ROOT,
    reviewer: bool,
) -> dict[str, Any]:
    path = regular_file(path, root=root)
    raw = read_bytes(path, root=root)
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == expected_raw, f"static source raw hash mismatch: {path}")
    source = raw.decode("utf-8")
    tree = ast.parse(source, filename=str(path))
    require(
        isinstance(policy, Mapping)
        and set(policy) == {"allowed_top_level_imports", "forbidden_tokens"},
        "import policy schema mismatch",
    )
    allowed = policy["allowed_top_level_imports"]
    forbidden = policy["forbidden_tokens"]
    require(
        isinstance(allowed, list)
        and allowed
        and all(isinstance(item, str) and item for item in allowed)
        and len(allowed) == len(set(allowed)),
        "allowed import policy is malformed",
    )
    require(
        isinstance(forbidden, list)
        and forbidden
        and all(isinstance(item, str) and item for item in forbidden)
        and len(forbidden) == len(set(forbidden)),
        "forbidden token policy is malformed",
    )
    imports: set[str] = set()
    calls: list[str] = []
    call_nodes: list[ast.Call] = []
    names: set[str] = set()
    attributes: set[str] = set()
    subscripts: list[ast.Subscript] = []
    string_nodes: list[ast.Constant] = []
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    allowed_direct_imports = {
        (name, None)
        for name in (
            "ast",
            "binascii",
            "csv",
            "ctypes",
            "decimal",
            "errno",
            "hashlib",
            "io",
            "json",
            "math",
            "os",
            "platform",
            "stat",
            "struct",
            "sys",
            "time",
            "torch",
        )
    } | {("datetime", "dt")}
    allowed_from_imports = {
        "__future__": {("annotations", None)},
        "collections.abc": {
            ("Callable", None),
            ("Mapping", None),
            ("Sequence", None),
        },
        "dataclasses": {("dataclass", None)},
        "datetime": {("datetime", None), ("timezone", None)},
        "pathlib": {("Path", None)},
        "typing": {("Any", None)},
    }
    allowed_module_expressions = {
        "ast": {
            "ast.AST",
            "ast.Add",
            "ast.Assign",
            "ast.Attribute",
            "ast.BinOp",
            "ast.Call",
            "ast.Constant",
            "ast.Import",
            "ast.ImportFrom",
            "ast.Name",
            "ast.Subscript",
            "ast.iter_child_nodes",
            "ast.parse",
            "ast.walk",
        },
        "binascii": {"binascii.crc32"},
        "csv": {"csv.DictReader", "csv.Error"},
        "ctypes": {
            "ctypes.CDLL",
            "ctypes.c_char_p",
            "ctypes.c_int",
            "ctypes.c_uint",
            "ctypes.get_errno",
            "ctypes.string_at",
        },
        "decimal": {"decimal.Decimal", "decimal.InvalidOperation"},
        "dt": {
            "dt.UTC",
            "dt.datetime",
            "dt.datetime.fromisoformat",
            "dt.datetime.now",
        },
        "errno": {"errno.EEXIST"},
        "hashlib": {"hashlib.sha256"},
        "io": {"io.BufferedReader", "io.StringIO"},
        "json": {"json.JSONDecodeError", "json.dumps", "json.loads"},
        "math": {"math.floor", "math.isfinite", "math.prod"},
        "os": {
            "os.O_CLOEXEC",
            "os.O_CREAT",
            "os.O_DIRECTORY",
            "os.O_EXCL",
            "os.O_NOFOLLOW",
            "os.O_RDONLY",
            "os.O_WRONLY",
            "os.SEEK_SET",
            "os.close",
            "os.environ",
            "os.environ.get",
            "os.fchmod",
            "os.fdopen",
            "os.fsencode",
            "os.fspath",
            "os.fstat",
            "os.fsync",
            "os.lseek",
            "os.lstat",
            "os.mkdir",
            "os.open",
            "os.path",
            "os.path.abspath",
            "os.path.lexists",
            "os.read",
            "os.scandir",
            "os.stat_result",
            "os.strerror",
            "os.unlink",
            "os.write",
        },
        "platform": {"platform.python_version"},
        "stat": {"stat.S_IMODE", "stat.S_ISDIR", "stat.S_ISLNK", "stat.S_ISREG"},
        "struct": {"struct.pack", "struct.unpack"},
        "sys": {"sys.argv", "sys.modules", "sys.modules.get"},
        "time": {"time.monotonic", "time.perf_counter"},
        "torch": {
            "torch.Tensor",
            "torch.__version__",
            "torch.equal",
            "torch.eye",
            "torch.float64",
            "torch.isfinite",
            "torch.linalg",
            "torch.linalg.eigvalsh",
            "torch.load",
            "torch.quantile",
            "torch.strided",
            "torch.tensor",
            "torch.trace",
        },
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(
                all(
                    (alias.name, alias.asname) in allowed_direct_imports
                    for alias in node.names
                ),
                f"unapproved direct import in {path}:{node.lineno}",
            )
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            require(node.level == 0, f"relative import forbidden: {path}")
            if node.module:
                require(
                    node.module in allowed_from_imports
                    and all(
                        (alias.name, alias.asname) in allowed_from_imports[node.module]
                        for alias in node.names
                    ),
                    f"unapproved from-import in {path}:{node.lineno}",
                )
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            calls.append(_call_name(node))
            call_nodes.append(node)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, ast.Subscript):
            subscripts.append(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_nodes.append(node)
    require(
        imports.issubset(set(allowed)),
        f"unapproved imports in {path}: {sorted(imports - set(allowed))}",
    )
    require("torch" in imports, f"torch import missing: {path}")
    dangerous_calls = {
        "compile",
        "__import__",
        "os.system",
        "os.popen",
        "torch.save",
    }
    require(not dangerous_calls.intersection(calls), f"dangerous call in {path}")
    dangerous_names = {
        "__builtins__",
        "__import__",
        "compile",
        "eval",
        "exec",
        "globals",
        "locals",
        "open",
        "vars",
    }
    require(
        dangerous_names.isdisjoint(names),
        f"dangerous identifier in {path}: {sorted(dangerous_names.intersection(names))}",
    )
    dangerous_attributes = {
        "__builtins__",
        "__dict__",
        "__getattr__",
        "__getattribute__",
        "__globals__",
        "__subclasses__",
        "backward",
        "grad",
        "popen",
        "pythonapi",
        "requires_grad_",
        "save",
        "system",
    }
    require(
        dangerous_attributes.isdisjoint(attributes),
        f"dangerous attribute in {path}",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "getattr":
            parent = parents.get(node)
            require(
                isinstance(parent, ast.Call) and parent.func is node,
                f"aliased getattr in {path}:{node.lineno}",
            )
        if isinstance(node, ast.Name) and node.id in allowed_module_expressions:
            parent = parents.get(node)
            direct_module_use = (
                isinstance(parent, ast.Attribute) and parent.value is node
            )
            approved_getattr_use = (
                isinstance(parent, ast.Call)
                and _call_name(parent) == "getattr"
                and parent.args
                and parent.args[0] is node
            )
            require(
                direct_module_use or approved_getattr_use,
                f"aliased sensitive module in {path}:{node.lineno}: {node.id}",
            )
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id == "ctypes":
            allowed_ctypes_attributes = {
                "CDLL",
                "c_char_p",
                "c_int",
                "c_uint",
                "get_errno",
                "string_at",
            }
            require(
                node.attr in allowed_ctypes_attributes,
                f"unapproved ctypes surface in {path}:{node.lineno}: {node.attr}",
            )
        if node.value.id == "os":
            allowed_os_attributes = {
                "O_CLOEXEC",
                "O_CREAT",
                "O_DIRECTORY",
                "O_EXCL",
                "O_NOFOLLOW",
                "O_RDONLY",
                "O_WRONLY",
                "SEEK_SET",
                "close",
                "environ",
                "fchmod",
                "fdopen",
                "fsencode",
                "fspath",
                "fstat",
                "fsync",
                "lseek",
                "lstat",
                "mkdir",
                "open",
                "path",
                "read",
                "scandir",
                "stat_result",
                "strerror",
                "unlink",
                "write",
            }
            require(
                node.attr in allowed_os_attributes,
                f"unapproved os surface in {path}:{node.lineno}: {node.attr}",
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        expression = _expression_name(node)
        filesystem_mutation_attributes = {
            "chmod",
            "hardlink_to",
            "lchmod",
            "mkdir",
            "open",
            "rename",
            "rmdir",
            "symlink_to",
            "touch",
            "truncate",
            "unlink",
            "write",
            "write_bytes",
            "write_text",
            "writelines",
        }
        allowed_filesystem_mutations = {
            "os.mkdir",
            "os.open",
            "os.unlink",
            "os.write",
        }
        if node.attr in filesystem_mutation_attributes:
            require(
                expression in allowed_filesystem_mutations
                and path in {PUBLISHER_SOURCE, REVIEWER_SOURCE},
                f"unapproved filesystem mutation in {path}:{node.lineno}: {expression}",
            )
        module_root = expression.split(".", 1)[0]
        if module_root in allowed_module_expressions:
            require(
                expression in allowed_module_expressions[module_root],
                f"unapproved module surface in {path}:{node.lineno}: {expression}",
            )
        if expression.startswith("torch."):
            allowed_torch_expressions = {
                "torch.Tensor",
                "torch.__version__",
                "torch.equal",
                "torch.eye",
                "torch.float64",
                "torch.isfinite",
                "torch.linalg",
                "torch.linalg.eigvalsh",
                "torch.load",
                "torch.quantile",
                "torch.strided",
                "torch.tensor",
                "torch.trace",
            }
            require(
                expression in allowed_torch_expressions,
                f"unapproved torch surface in {path}:{node.lineno}: {expression}",
            )
        if expression.startswith("sys."):
            require(
                expression in {"sys.argv", "sys.modules", "sys.modules.get"},
                f"unapproved sys surface in {path}:{node.lineno}: {expression}",
            )
        if expression.startswith("platform."):
            require(
                expression == "platform.python_version",
                f"unapproved platform surface in {path}:{node.lineno}: {expression}",
            )
        if expression.startswith("os.path."):
            require(
                expression in {"os.path.abspath", "os.path.lexists"},
                f"unapproved os.path surface in {path}:{node.lineno}: {expression}",
            )
        if expression == "sys.modules":
            parent = parents.get(node)
            require(
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr == "get",
                f"unapproved sys.modules use in {path}:{node.lineno}",
            )
        if expression == "os.path":
            parent = parents.get(node)
            require(
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr in {"abspath", "lexists"},
                f"aliased os.path in {path}:{node.lineno}",
            )
        if expression == "torch.linalg":
            parent = parents.get(node)
            require(
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr == "eigvalsh",
                f"aliased torch.linalg in {path}:{node.lineno}",
            )
        if expression in {"ctypes.CDLL", "ctypes.get_errno", "ctypes.string_at"}:
            parent = parents.get(node)
            require(
                isinstance(parent, ast.Call) and parent.func is node,
                f"aliased ctypes callable in {path}:{node.lineno}: {expression}",
            )
        if expression == "torch.load":
            parent = parents.get(node)
            require(
                isinstance(parent, ast.Call) and parent.func is node,
                f"aliased torch.load in {path}:{node.lineno}",
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "ctypes.CDLL":
            continue
        parent = parents.get(node)
        require(
            isinstance(parent, ast.Assign)
            and parent.value is node
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
            and parent.targets[0].id == "library"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value is None
            and len(node.keywords) == 1
            and node.keywords[0].arg == "use_errno"
            and isinstance(node.keywords[0].value, ast.Constant)
            and node.keywords[0].value.value is True,
            f"unapproved ctypes.CDLL construction in {path}:{node.lineno}",
        )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != "library":
            continue
        parent = parents.get(node)
        allowed_library_use = (
            isinstance(parent, ast.Assign)
            and node in parent.targets
            or isinstance(parent, ast.Call)
            and _call_name(parent) == "getattr"
            and parent.args
            and parent.args[0] is node
        )
        require(
            allowed_library_use,
            f"unapproved ctypes library use in {path}:{node.lineno}",
        )
    for node in call_nodes:
        require(
            isinstance(node.func, (ast.Name, ast.Attribute)),
            f"dynamic call target in {path}:{node.lineno}",
        )
        for keyword in node.keywords:
            if keyword.arg == "device":
                require(
                    _static_string(keyword.value) == "cpu",
                    f"non-CPU device request in {path}:{node.lineno}",
                )
            if keyword.arg == "requires_grad":
                require(
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False,
                    f"autograd request in {path}:{node.lineno}",
                )
        if _call_name(node) == "sys.modules.get":
            require(
                len(node.args) == 1
                and _static_string(node.args[0]) == "torch.cuda"
                and not node.keywords,
                f"unapproved sys.modules lookup in {path}:{node.lineno}",
            )
        if _call_name(node) == "torch.load":
            keyword_values = {keyword.arg: keyword.value for keyword in node.keywords}
            require(
                len(node.args) == 1
                and set(keyword_values) == {"map_location", "weights_only"}
                and _static_string(keyword_values["map_location"]) == "cpu"
                and isinstance(keyword_values["weights_only"], ast.Constant)
                and keyword_values["weights_only"].value is True,
                f"unsafe torch.load signature in {path}:{node.lineno}",
            )
        if _call_name(node) != "getattr":
            continue
        require(
            2 <= len(node.args) <= 3 and not node.keywords,
            f"dynamic getattr signature in {path}:{node.lineno}",
        )
        pair = (_expression_name(node.args[0]), _static_string(node.args[1]))
        allowed_getattrs = {
            ("library", "renameat2"),
            ("module", "_initialized"),
            ("namespace", "_initialized"),
            ("os", "O_DIRECTORY"),
            ("os", "O_NOFOLLOW"),
            ("torch", "cuda"),
        }
        require(
            pair in allowed_getattrs,
            f"unapproved dynamic getattr in {path}:{node.lineno}: {pair}",
        )
    dangerous_lookup_keys = dangerous_names | dangerous_attributes
    for node in subscripts:
        key = _static_string(node.slice)
        require(
            key not in dangerous_lookup_keys,
            f"dangerous dynamic lookup in {path}:{node.lineno}: {key!r}",
        )
    scan_lines = [list(line) for line in source.splitlines(keepends=True)]
    for node in string_nodes:
        end_line = node.end_lineno or node.lineno
        end_column = node.end_col_offset or node.col_offset
        for line_number in range(node.lineno, end_line + 1):
            line = scan_lines[line_number - 1]
            start = node.col_offset if line_number == node.lineno else 0
            stop = end_column if line_number == end_line else len(line)
            for position in range(start, min(stop, len(line))):
                if line[position] not in {"\n", "\r"}:
                    line[position] = " "
    policy_scan_source = "".join("".join(line) for line in scan_lines)
    for token in forbidden:
        require(
            token not in policy_scan_source,
            f"forbidden token in {path.name}: {token!r}",
        )
    if reviewer:
        require(expected_normalized is not None, "reviewer normalized hash is missing")
        normalized = normalized_source_sha256(path, root=root)
        require(normalized == expected_normalized, "reviewer normalized hash mismatch")
    else:
        normalized = expected_normalized
    return {
        "path": str(path),
        "raw_sha256": digest,
        "normalized_sha256": normalized,
        "imports": sorted(imports),
        "call_count": len(calls),
        "policy_pass": True,
    }


def observed_runtime() -> dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__}


def accelerator_uninitialized() -> bool:
    namespace = getattr(torch, "cu" + "da")
    return getattr(namespace, "_initialized", False) is False


def validate_execution_freeze(
    freeze: Mapping[str, Any],
    *,
    root: Path = ROOT,
    verify_live_files: bool = True,
) -> dict[str, Any]:
    require(set(freeze) == FREEZE_KEYS, "execution freeze top-level schema mismatch")
    require(freeze["schema_version"] == 1, "execution freeze version mismatch")
    require(freeze["protocol_id"] == PROTOCOL_ID, "execution freeze protocol mismatch")
    validate_timestamp(freeze["frozen_at_utc"], field="frozen_at_utc")
    require(
        freeze["repository_root"] == str(absolute(root)),
        "execution freeze root mismatch",
    )
    require(
        freeze["protocol_sha256"] == PROTOCOL_SHA256,
        "execution freeze protocol hash mismatch",
    )
    require(
        freeze["failed_manifest_sha256"] == FAILED_MANIFEST_SHA256,
        "execution freeze failed-manifest hash mismatch",
    )
    require(
        freeze["derivation_sha256"] == DERIVATION_SHA256,
        "execution freeze derivation hash mismatch",
    )

    identities = {
        "publisher": PUBLISHER_SOURCE,
        "reviewer": REVIEWER_SOURCE,
    }
    for name, expected_path in identities.items():
        value = freeze[name]
        require(
            isinstance(value, Mapping)
            and set(value) == {"path", "raw_sha256", "normalized_sha256"},
            f"execution freeze {name} schema mismatch",
        )
        require(
            value["path"] == str(expected_path),
            f"execution freeze {name} path mismatch",
        )
        require(
            is_sha256(value["raw_sha256"]), f"execution freeze {name} raw hash invalid"
        )
        require(
            is_sha256(value["normalized_sha256"]),
            f"execution freeze {name} normalized hash invalid",
        )
        if verify_live_files:
            require(
                sha256_file(expected_path, root=root) == value["raw_sha256"],
                f"execution freeze {name} live raw hash mismatch",
            )
            observed_normalized = normalized_source_sha256(expected_path, root=root)
            require(
                observed_normalized == value["normalized_sha256"],
                f"execution freeze {name} live normalized hash mismatch",
            )
            if name == "reviewer":
                require(
                    observed_normalized == EXPECTED_NORMALIZED_SOURCE_SHA256,
                    "reviewer normalized self-freeze mismatch",
                )
    tests = {"publisher_test": PUBLISHER_TEST, "reviewer_test": REVIEWER_TEST}
    for name, expected_path in tests.items():
        value = freeze[name]
        require(
            isinstance(value, Mapping) and set(value) == {"path", "sha256"},
            f"execution freeze {name} schema mismatch",
        )
        require(
            value["path"] == str(expected_path),
            f"execution freeze {name} path mismatch",
        )
        require(is_sha256(value["sha256"]), f"execution freeze {name} hash invalid")
        if verify_live_files:
            require(
                sha256_file(expected_path, root=root) == value["sha256"],
                f"execution freeze {name} live hash mismatch",
            )
    require(
        freeze["runtime"] == EXPECTED_RUNTIME, "execution freeze runtime pin mismatch"
    )
    require(
        observed_runtime() == EXPECTED_RUNTIME,
        "live runtime differs from execution freeze",
    )
    policy = freeze["import_policy"]
    require(
        isinstance(policy, Mapping)
        and set(policy) == {"allowed_top_level_imports", "forbidden_tokens"},
        "execution freeze import policy schema mismatch",
    )
    require(
        policy["allowed_top_level_imports"] == ALLOWED_TOP_LEVEL_IMPORTS
        and policy["forbidden_tokens"] == FORBIDDEN_POLICY_TOKENS,
        "execution freeze import policy values mismatch",
    )
    paths = freeze["paths"]
    require(
        isinstance(paths, Mapping) and dict(paths) == FREEZE_PATHS,
        "execution freeze paths mismatch",
    )
    prereview = freeze["prereview"]
    require(
        isinstance(prereview, Mapping)
        and set(prereview) == {"provenance_decision", "scientific_decision"}
        and prereview["provenance_decision"] == "GO"
        and prereview["scientific_decision"] == "GO",
        "execution freeze prereview decisions are not GO",
    )
    return {
        "sha256": sha256_file(EXECUTION_FREEZE, root=root)
        if verify_live_files
        else None,
        "runtime": dict(freeze["runtime"]),
        "paths": dict(paths),
        "exact": True,
    }


def validate_frozen_documents(*, root: Path = ROOT) -> dict[str, Any]:
    require(
        sha256_file(PROTOCOL, root=root) == PROTOCOL_SHA256, "salvage protocol changed"
    )
    require(
        sha256_file(FAILED_MANIFEST, root=root) == FAILED_MANIFEST_SHA256,
        "failed-tree manifest changed",
    )
    require(
        sha256_file(DERIVATION, root=root) == DERIVATION_SHA256,
        "manifest derivation changed",
    )
    manifest = load_json(FAILED_MANIFEST, root=root)
    require(
        set(manifest)
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
        "failed-tree manifest schema mismatch",
    )
    require(manifest["schema_version"] == 1, "failed-tree manifest version mismatch")
    require(
        manifest["manifest_id"]
        == "one_state_exact_selected_relaxed_cancellation_no_replay_salvage_failed_tree_v1",
        "failed-tree manifest id mismatch",
    )
    require(
        manifest["path_basis"] == "repository_relative_to_/home/coder/project",
        "failed-tree path basis mismatch",
    )
    validate_timestamp(manifest["frozen_at_utc"], field="manifest frozen_at_utc")
    failed = manifest["immutable_failed_staging"]
    require(
        isinstance(failed, Mapping)
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
        "immutable failed-tree schema mismatch",
    )
    expected_relative = str(FAILED_SOURCE.relative_to(root))
    require(failed["path"] == expected_relative, "immutable failed-tree path mismatch")
    require(
        failed["mutation_policy"] == "never_mutate",
        "failed-tree mutation policy changed",
    )
    require(
        failed["expected_entry_count"] == failed["expected_regular_file_count"] == 34
        and failed["expected_non_regular_entry_count"] == 0
        and failed["total_size_bytes"] == 114_951_202,
        "failed-tree count/size pins changed",
    )
    tree_hash = failed["tree_hash"]
    require(
        isinstance(tree_hash, Mapping)
        and set(tree_hash) == {"algorithm", "serialization", "sha256"}
        and tree_hash["algorithm"] == "sha256"
        and tree_hash["sha256"] == FAILED_TREE_SHA256,
        "failed-tree hash declaration mismatch",
    )
    files = failed["files"]
    require(
        isinstance(files, list) and len(files) == 34, "failed-tree file list mismatch"
    )
    names: set[str] = set()
    for item in files:
        require(
            isinstance(item, Mapping)
            and set(item) == {"name", "path", "size_bytes", "sha256"},
            "failed-tree file schema mismatch",
        )
        name = item["name"]
        require(
            isinstance(name, str) and name not in names, "duplicate failed-tree name"
        )
        names.add(name)
        require(
            item["path"] == f"{expected_relative}/{name}",
            f"failed-tree manifest path mismatch: {name}",
        )
    require(names.issuperset({"failure.json", "INCOMPLETE"}), "failed markers missing")

    projection = manifest["salvage_projection"]
    require(
        isinstance(projection, Mapping)
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
    require(
        projection["copy_source"] == "immutable_failed_staging"
        and projection["excluded_failure_file"] == "failure.json"
        and projection["staged_file_count_with_copied_incomplete"] == 33
        and projection["staged_total_size_bytes"] == 114_949_714
        and projection["staged_tree_sha256"] == STAGED_TREE_SHA256
        and projection["remove_after_all_staged_validation"] == "INCOMPLETE"
        and projection["published_file_count"] == 32
        and projection["published_total_size_bytes"] == 114_949_656
        and projection["published_tree_sha256"] == PUBLISHED_TREE_SHA256
        and projection["publication_target"] == str(TARGET.relative_to(root))
        and projection["rewrite_policy"]
        == "no_packet_file_may_be_rewritten_or_regenerated",
        "salvage projection values changed",
    )
    return {
        "manifest": manifest,
        "failed_entries": files,
        "published_names": names - {"failure.json", "INCOMPLETE"},
    }


def _entries_from_hash_map(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": name, "sha256": digest, "size_bytes": 0}
        for name, digest in values.items()
    ]


def validate_original_source(
    manifest: Mapping[str, Any], *, root: Path = ROOT, verify_large_inputs: bool = True
) -> dict[str, Any]:
    require(
        sha256_file(RECOVERY_INPUT_MANIFEST, root=root)
        == RECOVERY_INPUT_MANIFEST_SHA256,
        "recovery input manifest changed",
    )
    frozen = load_json(RECOVERY_INPUT_MANIFEST, root=root)
    require(
        set(frozen)
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
        "recovery input manifest schema mismatch",
    )
    require(
        frozen["protocol_id"] == RECOVERY_PROTOCOL_ID,
        "input manifest protocol mismatch",
    )
    require(
        frozen["source_staging_relative_path"]
        == str(ORIGINAL_SOURCE.relative_to(root)),
        "original source path changed",
    )
    hashes = frozen["files_sha256"]
    require(
        isinstance(hashes, Mapping)
        and set(hashes) == set(frozen["expected_exact_file_set"])
        and len(hashes) == 14,
        "original source manifest file set mismatch",
    )
    observed = audit_tree(ORIGINAL_SOURCE, root=root)
    require(
        set(observed["files"]) == set(hashes), "original source tree file set mismatch"
    )
    for name, digest in hashes.items():
        require(
            observed["files"][name]["sha256"] == digest,
            f"original source hash mismatch: {name}",
        )
    require(
        observed["tree_sha256"] == ORIGINAL_TREE_SHA256,
        "original source tree hash mismatch",
    )
    require(observed["file_count"] == 14, "original source file count mismatch")
    require(
        observed["total_size_bytes"] == 58_934_801,
        "original source total size mismatch",
    )

    dependencies = frozen["recovery_dependencies"]
    require(
        isinstance(dependencies, Mapping)
        and set(dependencies)
        == {"live_producer_source", "independent_continuation_reviewer"},
        "recovery dependency schema mismatch",
    )
    for name, item in dependencies.items():
        require(
            isinstance(item, Mapping) and set(item) == {"relative_path", "sha256"},
            f"dependency schema mismatch: {name}",
        )
        dep_path = root / item["relative_path"]
        require(
            sha256_file(dep_path, root=root) == item["sha256"],
            f"dependency hash mismatch: {name}",
        )

    require(
        sha256_file(PARENT_PROGRESS, root=root) == PARENT_PROGRESS_SHA256,
        "parent progress changed",
    )
    require(
        sha256_file(PARENT_FINAL, root=root) == PARENT_FINAL_SHA256,
        "parent final changed",
    )
    require(
        sha256_file(PARENT_REVIEW, root=root) == PARENT_REVIEW_SHA256,
        "parent review changed",
    )
    parent_review = load_json(PARENT_REVIEW, root=root)
    require(
        parent_review.get("valid") is True
        and parent_review.get("failed_gates") == []
        and parent_review.get("errors") == [],
        "parent independent review is invalid",
    )

    task = validate_task_and_import_provenance(
        root=root, verify_large_inputs=verify_large_inputs
    )
    return {"tree": observed, "frozen_manifest": frozen, "task_and_import": task}


def validate_task_and_import_provenance(
    *, root: Path = ROOT, verify_large_inputs: bool = True
) -> dict[str, Any]:
    require(
        sha256_file(RECOVERY_TASK_MANIFEST, root=root) == RECOVERY_TASK_MANIFEST_SHA256,
        "task manifest changed",
    )
    require(
        sha256_file(RECOVERY_IMPORT_MANIFEST, root=root)
        == RECOVERY_IMPORT_MANIFEST_SHA256,
        "import manifest changed",
    )
    task = load_json(RECOVERY_TASK_MANIFEST, root=root)
    require(
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
        },
        "task manifest schema mismatch",
    )
    require(
        task["protocol_id"] == RECOVERY_PROTOCOL_ID
        and task["source_weight_index"] == SOURCE_WEIGHT_INDEX
        and same_float(task["tau"], TASK_TAU)
        and task["task_name"] == TASK_NAME
        and task["accepted_vae_checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256,
        "task identity mismatch",
    )
    for relative, digest in task["source_dependencies"].items():
        require(
            sha256_file(root / relative, root=root) == digest,
            f"task dependency changed: {relative}",
        )
    selected = task["selected_task_tensors"]
    require(
        isinstance(selected, Mapping)
        and set(selected)
        == {"train_images", "train_labels", "test_images", "test_labels"},
        "task tensor fingerprint schema mismatch",
    )
    for name, item in selected.items():
        require(
            isinstance(item, Mapping)
            and set(item) == {"dtype", "shape", "sha256"}
            and isinstance(item["dtype"], str)
            and isinstance(item["shape"], list)
            and is_sha256(item["sha256"]),
            f"task tensor fingerprint malformed: {name}",
        )

    imported = load_json(RECOVERY_IMPORT_MANIFEST, root=root)
    require(
        set(imported) == {"protocol_id", "entry_modules", "loaded_repository_modules"}
        and imported["protocol_id"] == RECOVERY_PROTOCOL_ID,
        "import manifest schema/protocol mismatch",
    )
    modules = imported["loaded_repository_modules"]
    require(
        isinstance(modules, Mapping) and len(modules) == 49,
        "import closure count mismatch",
    )
    paths: set[str] = set()
    for name, item in modules.items():
        require(
            isinstance(name, str)
            and isinstance(item, Mapping)
            and set(item) == {"relative_path", "sha256"},
            f"import closure entry malformed: {name}",
        )
        relative = item["relative_path"]
        require(
            isinstance(relative, str) and relative not in paths,
            "duplicate import closure path",
        )
        paths.add(relative)
        require(
            sha256_file(root / relative, root=root) == item["sha256"],
            f"import closure changed: {name}",
        )

    expected_inputs = {
        "config.json": "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4",
        "weight_pool.pt": "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef",
        "weight_pool_records.csv": "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933",
        "vae_checkpoint.pt": ACCEPTED_CHECKPOINT_SHA256,
    }
    verified_inputs: dict[str, str] = {}
    if verify_large_inputs:
        for name, digest in expected_inputs.items():
            observed = sha256_file(ACCEPTED_RUN / name, root=root)
            require(
                observed == digest, f"accepted reconstruction input changed: {name}"
            )
            verified_inputs[name] = observed
    return {
        "task_manifest_sha256": RECOVERY_TASK_MANIFEST_SHA256,
        "import_manifest_sha256": RECOVERY_IMPORT_MANIFEST_SHA256,
        "module_count": len(modules),
        "accepted_inputs_sha256": verified_inputs or expected_inputs,
        "task_tensors": selected,
    }


def validate_startup_failure(
    manifest: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    lineage = manifest["preserved_lineage"]
    require(
        isinstance(lineage, Mapping)
        and set(lineage) == {"original_source_staging", "first_startup_failed_staging"},
        "preserved lineage schema mismatch",
    )
    startup = lineage["first_startup_failed_staging"]
    require(
        isinstance(startup, Mapping)
        and startup["path"] == str(STARTUP_FAILURE.relative_to(root))
        and startup["expected_file_count"] == 3
        and startup["total_size_bytes"] == 2075
        and startup["tree_sha256"] == STARTUP_TREE_SHA256
        and startup["geometry_evaluations"] == 0
        and startup["model_loaded"] is False
        and startup["mutation_policy"] == "never_mutate",
        "startup failure identity mismatch",
    )
    observed = audit_tree_against_entries(
        STARTUP_FAILURE,
        startup["files"],
        expected_tree_sha256=STARTUP_TREE_SHA256,
        expected_total_size=2075,
        root=root,
    )
    return observed


def validate_failure_localization(
    source: Path, manifest: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    expected = manifest["expected_failure"]
    failure = load_json(source / "failure.json", root=root)
    require(
        set(failure) == {"error", "error_type", "protocol_id", "status", "traceback"},
        "failure record schema mismatch",
    )
    require(
        failure["status"] == expected["status"] == "failed_recovery_staging"
        and failure["error_type"] == expected["error_type"] == "RecoveryError"
        and failure["error"]
        == expected["error"]
        == "replayed final spectrum CSV mismatch at rank 2"
        and failure["protocol_id"] == RECOVERY_PROTOCOL_ID,
        "failure record identity mismatch",
    )
    require(
        sha256_file(source / "failure.json", root=root) == expected["sha256"],
        "failure hash mismatch",
    )
    log = read_bytes(source / "run.log", root=root).decode("utf-8")
    required_lines = (
        "stage=single-final-geometry-replay",
        "geometry_replay_pass=1 evaluations=1 metric_max=0 spectrum_max=0",
    )
    require(
        all(line in log for line in required_lines),
        "run log lacks successful replay evidence",
    )
    require(
        "published output=" not in log and "publication complete" not in log,
        "run log claims publication",
    )
    return {
        "failure_sha256": expected["sha256"],
        "run_log_sha256": sha256_file(source / "run.log", root=root),
    }


def validate_png(
    path: Path, *, expected_size: tuple[int, int], root: Path = ROOT
) -> dict[str, Any]:
    raw = read_bytes(path, root=root)
    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), f"bad PNG signature: {path.name}")
    position = 8
    names: list[bytes] = []
    width = height = 0
    compressed = 0
    while position < len(raw):
        require(position + 12 <= len(raw), f"truncated PNG chunk: {path.name}")
        length = struct.unpack(">I", raw[position : position + 4])[0]
        kind = raw[position + 4 : position + 8]
        stop = position + 12 + length
        require(stop <= len(raw), f"truncated PNG payload: {path.name}")
        data = raw[position + 8 : position + 8 + length]
        expected_crc = struct.unpack(">I", raw[position + 8 + length : stop])[0]
        actual_crc = binascii.crc32(kind + data) & 0xFFFFFFFF
        require(expected_crc == actual_crc, f"PNG CRC mismatch: {path.name}")
        names.append(kind)
        if kind == b"IHDR":
            require(length == 13 and len(names) == 1, f"bad PNG IHDR: {path.name}")
            width, height = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            compressed += length
        elif kind == b"IEND":
            require(length == 0 and stop == len(raw), f"bad PNG ending: {path.name}")
            position = stop
            break
        position = stop
    require(names and names[-1] == b"IEND", f"PNG has no IEND: {path.name}")
    require((width, height) == expected_size, f"PNG dimensions mismatch: {path.name}")
    require(compressed > 0, f"PNG has no image data: {path.name}")
    return {
        "width": width,
        "height": height,
        "idat_bytes": compressed,
        "crc_exact": True,
    }


def validate_packet_hash_graph(
    packet: Path, expected_names: set[str], *, root: Path = ROOT
) -> dict[str, Any]:
    observed_names = set(audit_tree(packet, root=root)["files"])
    require(
        expected_names.issubset(observed_names), "packet is missing published files"
    )
    artifacts = load_json(packet / "artifact_manifest.json", root=root)
    require(
        set(artifacts)
        == {
            "protocol_id",
            "recovery_finalizer_source_sha256",
            "recovery_finalizer_normalized_source_sha256",
            "frozen_input_manifest_sha256",
            "recovery_task_manifest_sha256",
            "recovery_import_manifest_sha256",
            "source_progress_checkpoint_sha256",
            "artifacts",
        },
        "artifact manifest schema mismatch",
    )
    require(
        artifacts["protocol_id"] == RECOVERY_PROTOCOL_ID, "artifact protocol mismatch"
    )
    links = artifacts["artifacts"]
    require(
        isinstance(links, Mapping) and len(links) == 29, "artifact link count mismatch"
    )
    require(
        set(links)
        == expected_names - {"artifact_manifest.json", "FINALIZED.json", "run.log"},
        "artifact link set mismatch",
    )
    for name, digest in links.items():
        require(is_sha256(digest), f"malformed artifact hash: {name}")
        require(
            sha256_file(packet / name, root=root) == digest,
            f"artifact hash graph mismatch: {name}",
        )
    require(
        artifacts["recovery_finalizer_source_sha256"] == FINALIZER_SHA256
        and artifacts["recovery_finalizer_normalized_source_sha256"]
        == FINALIZER_NORMALIZED_SHA256
        and artifacts["frozen_input_manifest_sha256"] == RECOVERY_INPUT_MANIFEST_SHA256
        and artifacts["recovery_task_manifest_sha256"] == RECOVERY_TASK_MANIFEST_SHA256
        and artifacts["recovery_import_manifest_sha256"]
        == RECOVERY_IMPORT_MANIFEST_SHA256
        and artifacts["source_progress_checkpoint_sha256"] == SOURCE_PROGRESS_SHA256,
        "artifact manifest metadata links mismatch",
    )
    finalized = load_json(packet / "FINALIZED.json", root=root)
    require(
        finalized.get("protocol_id") == RECOVERY_PROTOCOL_ID
        and finalized.get("source_protocol_id") == SOURCE_PROTOCOL_ID
        and finalized.get("status") == "complete_awaiting_independent_recovery_review"
        and finalized.get("recovery_valid") is True
        and finalized.get("scientific_success") is False
        and finalized.get("accepted_updates") == ACCEPTED_UPDATES
        and finalized.get("new_accepted_updates") == NEW_ACCEPTED_UPDATES
        and finalized.get("termination") == "max_updates_reached",
        "FINALIZED identity mismatch",
    )
    cross = {
        "decision_sha256": "decision.json",
        "recovery_audit_sha256": "recovery_audit.json",
        "artifact_manifest_sha256": "artifact_manifest.json",
        "final_checkpoint_sha256": "final_checkpoint.pt",
    }
    for key, name in cross.items():
        require(
            finalized.get(key) == sha256_file(packet / name, root=root),
            f"FINALIZED cross-hash mismatch: {key}",
        )
    pngs = {
        "recovery_trajectory.png": (3060, 1800),
        "recovery_spectra.png": (1800, 1080),
        "recovery_direction_radius_audit.png": (1980, 1080),
    }
    png_audits = {
        name: validate_png(packet / name, expected_size=size, root=root)
        for name, size in pngs.items()
    }
    return {
        "artifact_manifest": artifacts,
        "finalized": finalized,
        "artifact_count": len(links),
        "pngs": png_audits,
    }


def validate_publication_bytes(
    source: Path,
    target: Path,
    published_names: set[str],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    source_audit = audit_tree(source, root=root)
    target_audit = audit_tree(target, root=root)
    require(
        set(target_audit["files"]) == published_names, "published file set mismatch"
    )
    require(target_audit["file_count"] == 32, "published file count mismatch")
    require(
        target_audit["total_size_bytes"] == 114_949_656, "published total size mismatch"
    )
    require(
        target_audit["tree_sha256"] == PUBLISHED_TREE_SHA256,
        "published tree hash mismatch",
    )
    inode_pairs: dict[str, dict[str, int]] = {}
    for name in sorted(published_names):
        left = source_audit["files"][name]
        right = target_audit["files"][name]
        require(left["sha256"] == right["sha256"], f"published bytes differ: {name}")
        require(
            left["size_bytes"] == right["size_bytes"], f"published size differs: {name}"
        )
        require(
            (left["device"], left["inode"]) != (right["device"], right["inode"]),
            f"published file is a source hardlink: {name}",
        )
        require(
            left["link_count"] == 1 and right["link_count"] == 1,
            f"hardlink count is not one: {name}",
        )
        inode_pairs[name] = {
            "source_inode": left["inode"],
            "published_inode": right["inode"],
        }
    return {
        "source_tree_sha256": source_audit["tree_sha256"],
        "published_tree_sha256": target_audit["tree_sha256"],
        "file_count": len(published_names),
        "total_size_bytes": target_audit["total_size_bytes"],
        "all_bytes_identical": True,
        "all_inodes_distinct": True,
        "all_link_counts_one": True,
        "inode_pairs": inode_pairs,
    }


def spectral_metrics(eigenvalues: torch.Tensor) -> dict[str, float]:
    require(isinstance(eigenvalues, torch.Tensor), "spectrum is not a tensor")
    eig = eigenvalues.detach().to(device="cpu", dtype=torch.float64).contiguous()
    require(eig.ndim == 1 and eig.numel() > 0, "spectrum shape is invalid")
    require(bool(torch.isfinite(eig).all()), "spectrum is non-finite")
    require(bool((eig >= 0.0).all()), "spectrum contains negative values")
    if eig.numel() > 1:
        require(bool((eig[1:] >= eig[:-1] - 1e-12).all()), "spectrum is not ordered")
    dimension = eig.numel()
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
    low = eig < LOW_THRESHOLD
    require(bool(low.any()), "low spectrum is empty")
    ratio = (eig + EPSILON) / (1.0 + EPSILON)
    matrix_slope = (1.0 - ratio.reciprocal()) / (dimension * (1.0 + EPSILON))
    burg_trace = ratio.mean()
    logdet = ratio.log().mean()
    exact_a = contribution.mean()
    burg = burg_trace - logdet - 1.0
    lower_stop = math.floor(0.9 * dimension)
    return {
        "exact_a_per_dim": float(exact_a),
        "a_constant_term": 1.0,
        "a_linear_trace_term": float(-2.0 * eig.mean()),
        "a_quartic_term": float(eig.square().mean()),
        "trace_m_per_dim": float(eig.mean()),
        "damped_full_burg_per_dim": float(burg),
        "burg_trace_r_term": float(burg_trace),
        "burg_neg_logdet_r_term": float(-logdet),
        "logdet_r_per_dim": float(logdet),
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
        "m_lt_0p01_fraction": float((eig < 1e-2).double().mean()),
        "m_lt_0p1_fraction": float(low.double().mean()),
        "m_lt_0p5_fraction": float((eig < 0.5).double().mean()),
        "m_near_1_10pct_fraction": float(((eig - 1.0).abs() <= 0.1).double().mean()),
        "m_gt_1_fraction": float((eig > 1.0).double().mean()),
        "m_gt_2_fraction": float((eig > 2.0).double().mean()),
        "a_from_m_lt_0p1_share": float(contribution[low].sum() / total_a),
        "a_from_m_gt_1_share": float(contribution[eig > 1.0].sum() / total_a),
        "burg_matrix_gradient_norm": float(matrix_slope.norm()),
        "burg_matrix_gradient_eig_min": float(matrix_slope.min()),
        "burg_matrix_gradient_eig_p50": float(matrix_slope.median()),
        "burg_matrix_gradient_eig_max": float(matrix_slope.max()),
        "true_objective": float(exact_a + OLD_BETA * burg),
        "a_direct_matrix": float(exact_a),
        "a_trace_closure": float(1.0 - 2.0 * eig.mean() + eig.square().mean()),
        "effective_rank": float(trace.square() / square_sum),
        "effective_rank_fraction": float(trace.square() / (square_sum * dimension)),
        "li_gap_per_dim": float((eig.sqrt() - 1.0).square().mean()),
        "a_low90_abs_per_dim": float(contribution[:lower_stop].sum() / dimension),
        "a_low_lt_0p1_abs_per_dim": float(contribution[low].sum() / dimension),
        "a_high_gt_1_abs_per_dim": float(contribution[eig > 1.0].sum() / dimension),
        "a_top1_share": float(contribution[-1] / total_a),
        "a_top10_share": float(contribution[-10:].sum() / total_a),
        "top1_trace_share": float(eig[-1] / trace.clamp_min(1e-30)),
        "top10_trace_share": float(eig[-10:].sum() / trace.clamp_min(1e-30)),
        "frozen_low_energy": float(eig[low].mean()),
        "canonical_low_energy": float(eig[low].mean()),
        "count_lt_1e_4": float((eig < 1e-4).sum()),
        "count_lt_1e_2": float((eig < 1e-2).sum()),
        "count_lt_0p1": float(low.sum()),
        "a_gt1": float(contribution[eig > 1.0].sum() / dimension),
        "low_count": float(low.sum()),
    }


def validate_spectrum_history(progress: Mapping[str, Any]) -> dict[str, Any]:
    states = progress["state_rows"]
    spectra = progress["spectrum_rows"]
    require(
        isinstance(states, list) and len(states) == 101,
        "state checkpoint count mismatch",
    )
    require(
        isinstance(spectra, list) and len(spectra) == 51_712,
        "spectrum checkpoint count mismatch",
    )
    require(
        [exact_int(row["accepted_update"]) for row in states] == list(range(101)),
        "state update sequence mismatch",
    )
    hashes = [row["parameter_hash"] for row in states]
    require(all(is_sha256(value) for value in hashes), "state parameter hash malformed")
    require(len(set(hashes)) == len(hashes), "duplicate state parameter hash")
    worst_metric = 0.0
    worst_contribution = 0.0
    for update, state_row in enumerate(states):
        block = spectra[update * DIMENSION : (update + 1) * DIMENSION]
        require(
            [
                (exact_int(row["accepted_update"]), exact_int(row["rank"]))
                for row in block
            ]
            == [(update, rank) for rank in range(DIMENSION)],
            f"spectrum physical grid mismatch at update {update}",
        )
        eig = torch.tensor(
            [finite_float(row["m_eigenvalue"]) for row in block], dtype=torch.float64
        )
        contribution = torch.tensor(
            [finite_float(row["a_contribution"]) for row in block], dtype=torch.float64
        )
        contribution_error = float((contribution - (eig - 1.0).square()).abs().max())
        require(
            contribution_error <= ABSOLUTE_TOLERANCE,
            f"spectrum contribution mismatch at {update}",
        )
        expected = spectral_metrics(eig)
        errors: list[float] = []
        for name, value in expected.items():
            require(name in state_row, f"state metric missing: {name}")
            error = abs(finite_float(state_row[name]) - value)
            require(
                error <= ABSOLUTE_TOLERANCE,
                f"state spectrum closure mismatch: {update}/{name}",
            )
            errors.append(error)
        raw_min = finite_float(state_row["m_raw_eig_min"])
        require(raw_min >= -1e-8, f"raw spectrum minimum invalid at {update}")
        require(
            abs(max(raw_min, 0.0) - float(eig[0])) <= ABSOLUTE_TOLERANCE,
            "raw/clamped minimum mismatch",
        )
        require(
            finite_float(state_row["a_direct_abs_error"]) <= ABSOLUTE_TOLERANCE
            and finite_float(state_row["a_trace_abs_error"]) <= ABSOLUTE_TOLERANCE,
            f"exact-A closure failed at {update}",
        )
        require(
            finite_float(state_row["low_basis_orthogonality_max_abs"])
            <= ABSOLUTE_TOLERANCE
            and finite_float(state_row["low_basis_eigen_residual_relative"])
            <= ABSOLUTE_TOLERANCE
            and finite_float(state_row["helper_eigenvalue_max_abs_error"])
            <= ABSOLUTE_TOLERANCE,
            f"stored low-space closure failed at {update}",
        )
        require(
            is_sha256(state_row["low_basis_hash"]),
            f"low-basis hash malformed at {update}",
        )
        worst_metric = max(worst_metric, max(errors))
        worst_contribution = max(worst_contribution, contribution_error)
    return {
        "state_count": len(states),
        "spectrum_row_count": len(spectra),
        "worst_metric_abs_error": worst_metric,
        "worst_contribution_abs_error": worst_contribution,
        "physical_grid_exact": True,
    }


def _seed_parent_rows(parent: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "state_rows": [{**dict(row), "phase": "i6"} for row in parent["state_rows"]],
        "spectrum_rows": [
            {**dict(row), "phase": "i6"} for row in parent["spectrum_rows"]
        ],
        "proposal_rows": [
            {**dict(row), "phase": "i6"}
            for row in parent["proposal_rows"]
            if exact_int(row["accepted"]) == 1
        ],
        "line_rows": [{**dict(row), "phase": "i6"} for row in parent["line_rows"]],
        "selection_rows": [
            {**dict(row), "phase": "i6"} for row in parent["selection_rows"]
        ],
    }


def validate_checkpoint_and_csv(packet: Path, *, root: Path = ROOT) -> dict[str, Any]:
    progress = load_checkpoint(packet / "source_progress_checkpoint.pt", root=root)
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
    require(
        set(progress) == expected_progress_keys, "terminal progress schema mismatch"
    )
    require(
        progress["protocol_id"] == SOURCE_PROTOCOL_ID
        and progress["accepted_updates"] == ACCEPTED_UPDATES
        and progress["new_accepted_updates"] == NEW_ACCEPTED_UPDATES
        and progress["selected_arm"] == "low"
        and progress["terminal"] is True
        and progress["termination"] == "max_updates_reached"
        and progress["transition_chain_sha256"] == TRANSITION_CHAIN_SHA256
        and progress["active_parameter_hash"] == ACTIVE_PARAMETER_SHA256
        and progress["parent_checkpoint_sha256"] == PARENT_FINAL_SHA256
        and progress["parent_progress_checkpoint_sha256"] == PARENT_PROGRESS_SHA256,
        "terminal progress identity mismatch",
    )
    active = progress["active_model_state"]
    require(
        isinstance(active, Mapping) and len(active) == ACTIVE_TENSOR_COUNT,
        "active tensor set mismatch",
    )
    active_hash, active_count = named_tensor_sha256(active)
    require(
        active_hash == ACTIVE_PARAMETER_SHA256
        and active_count == ACTIVE_PARAMETER_COUNT
        and progress["state_rows"][-1]["parameter_hash"] == ACTIVE_PARAMETER_SHA256,
        "active tensor fingerprint mismatch",
    )

    csv_map = {
        "state_rows": "state_metrics.csv",
        "spectrum_rows": "state_spectra.csv",
        "proposal_rows": "proposal_diagnostics.csv",
        "line_rows": "line_search.csv",
        "selection_rows": "arm_selection.csv",
    }
    csv_audits = {
        key: validate_rows_csv(progress[key], packet / name, root=root)
        for key, name in csv_map.items()
    }
    require(
        [audit["row_count"] for audit in csv_audits.values()]
        == [101, 51_712, 100, 766, 2],
        "authoritative CSV row counts mismatch",
    )

    parent = load_checkpoint(PARENT_PROGRESS, root=root)
    seeded = _seed_parent_rows(parent)
    require(
        progress["state_rows"][:18] == seeded["state_rows"],
        "parent state prefix mismatch",
    )
    require(
        progress["spectrum_rows"][: 18 * DIMENSION] == seeded["spectrum_rows"],
        "parent spectrum prefix mismatch",
    )
    require(
        progress["proposal_rows"][:17] == seeded["proposal_rows"],
        "parent proposal prefix mismatch",
    )
    require(
        progress["line_rows"][: len(seeded["line_rows"])] == seeded["line_rows"],
        "parent line prefix mismatch",
    )
    require(
        progress["selection_rows"] == seeded["selection_rows"],
        "parent selection prefix mismatch",
    )
    require(
        progress["intervention_origin"] == parent["proposal_rows"][-1],
        "intervention origin mismatch",
    )
    require(
        progress["initial_metrics"] == parent["initial_metrics"],
        "initial metrics lineage mismatch",
    )
    require(
        progress["tolerances"] == parent["tolerances"], "tolerance lineage mismatch"
    )
    spectra = validate_spectrum_history(progress)
    return {
        "progress": progress,
        "parent": parent,
        "csv": csv_audits,
        "spectrum_history": spectra,
        "active_parameter_count": active_count,
        "active_tensor_count": len(active),
    }


def _geometry_metric_closure(
    hessian: torch.Tensor,
    matrix: torch.Tensor,
    eig: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, float]:
    values = spectral_metrics(eig)
    identity = torch.eye(eig.numel(), dtype=torch.float64)
    direct = (matrix - identity).square().sum() / eig.numel()
    raw_eig = torch.linalg.eigvalsh(matrix)
    low_values = raw_eig[: basis.shape[1]]
    orthogonality = float(
        (basis.T @ basis - torch.eye(basis.shape[1], dtype=torch.float64)).abs().max()
    )
    residual = float(
        (matrix @ basis - basis * low_values.unsqueeze(0)).norm()
        / matrix.norm().clamp_min(1e-30)
    )
    low_energy = float(torch.trace(basis.T @ matrix @ basis) / basis.shape[1])
    values.update(
        {
            "hessian_symmetry_rel": float(
                (hessian - hessian.T).norm() / hessian.norm().clamp_min(1e-30)
            ),
            "a_direct_matrix": float(direct),
            "a_direct_abs_error": abs(float(direct) - values["exact_a_per_dim"]),
            "a_trace_abs_error": abs(
                values["a_trace_closure"] - values["exact_a_per_dim"]
            ),
            "m_raw_eig_min": float(raw_eig.min()),
            "frozen_low_energy": low_energy,
            "canonical_low_energy": low_energy,
            "low_basis_orthogonality_max_abs": orthogonality,
            "low_basis_eigen_residual_relative": residual,
            "helper_eigenvalue_max_abs_error": float(
                (eig - raw_eig.clamp_min(0.0)).abs().max()
            ),
        }
    )
    return values


def validate_replayed_spectrum_csv(
    path: Path,
    authoritative: Sequence[float],
    geometry_eig: torch.Tensor,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    raw = read_bytes(path, root=root)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
    require(
        reader.fieldnames == ["rank", "stored", "replayed", "abs_error"]
        and len(set(reader.fieldnames)) == 4,
        "replayed spectrum CSV header mismatch",
    )
    rows = list(reader)
    require(
        len(rows) == len(authoritative) == geometry_eig.numel(),
        "replayed spectrum count mismatch",
    )
    maximum = 0.0
    maximum_rank = 0
    for rank, row in enumerate(rows):
        require(
            None not in row and set(row) == set(reader.fieldnames),
            f"replayed spectrum row schema: {rank}",
        )
        require(
            exact_int(row["rank"]) == rank, f"replayed spectrum rank mismatch: {rank}"
        )
        expected = authoritative[rank]
        replayed = float(geometry_eig[rank])
        require(
            same_float(row["stored"], expected),
            f"stored spectrum bit mismatch at rank {rank}",
        )
        require(
            same_float(row["replayed"], replayed),
            f"replayed spectrum bit mismatch at rank {rank}",
        )
        require(
            same_float(expected, replayed),
            f"geometry/checkpoint spectrum bit mismatch at rank {rank}",
        )
        error = abs(replayed - expected)
        require(
            same_float(row["abs_error"], error),
            f"spectrum error mismatch at rank {rank}",
        )
        if error > maximum:
            maximum = error
            maximum_rank = rank
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "max_absolute_error": maximum,
        "max_error_rank": maximum_rank,
        "all_binary64_representations_exact": True,
    }


def audit_geometry_payload(
    payload: Mapping[str, Any],
    authoritative_spectrum: Sequence[float],
    final_state: Mapping[str, Any],
    replay_csv: Path,
    *,
    root: Path = ROOT,
    strict_identity: bool = True,
) -> dict[str, Any]:
    tensor_names = (
        "hessian",
        "matrix",
        "eig",
        "current_low_basis",
        "current_low_projector",
    )
    for name in tensor_names:
        require(
            name in payload and isinstance(payload[name], torch.Tensor),
            f"geometry tensor missing: {name}",
        )
        value = payload[name]
        require(
            value.device.type == "cpu" and value.dtype == torch.float64,
            f"geometry dtype/device: {name}",
        )
        require(
            not value.requires_grad and bool(torch.isfinite(value).all()),
            f"geometry tensor invalid: {name}",
        )
    hessian = payload["hessian"].detach().contiguous()
    matrix = payload["matrix"].detach().contiguous()
    eig = payload["eig"].detach().contiguous()
    basis = payload["current_low_basis"].detach().contiguous()
    projector = payload["current_low_projector"].detach().contiguous()
    dimension = eig.numel()
    require(hessian.shape == (dimension, dimension), "H shape mismatch")
    require(matrix.shape == (dimension, dimension), "M shape mismatch")
    require(eig.shape == (dimension,), "eigenvalue shape mismatch")
    require(projector.shape == (dimension, dimension), "projector shape mismatch")
    require(basis.ndim == 2 and basis.shape[0] == dimension, "low basis shape mismatch")
    if strict_identity:
        require(
            dimension == DIMENSION and basis.shape == (DIMENSION, LOW_COUNT),
            "frozen geometry dimensions mismatch",
        )
        metadata = {
            "protocol_id": RECOVERY_PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "source_progress_checkpoint_sha256": SOURCE_PROGRESS_SHA256,
            "accepted_vae_checkpoint_sha256": ACCEPTED_CHECKPOINT_SHA256,
            "active_parameter_hash": ACTIVE_PARAMETER_SHA256,
            "source_weight_index": SOURCE_WEIGHT_INDEX,
            "task_name": TASK_NAME,
            "tau": TASK_TAU,
            "z_sha256": Z_SHA256,
            "accepted_update": ACCEPTED_UPDATES,
        }
        for name, value in metadata.items():
            require(
                deep_exact(payload.get(name), value),
                f"geometry metadata mismatch: {name}",
            )
        require(
            payload.get("geometry_constants")
            == {
                "dimension": DIMENSION,
                "epsilon": EPSILON,
                "old_beta": OLD_BETA,
                "low_threshold": LOW_THRESHOLD,
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
            },
            "geometry constants mismatch",
        )
        require(
            payload.get("installed_model_aggregate_sha256")
            == payload.get("post_replay_model_aggregate_sha256"),
            "stored tensor aggregate changed during replay",
        )

    fingerprints = {name: tensor_fingerprint(payload[name]) for name in tensor_names}
    require(
        payload.get("tensor_fingerprints") == fingerprints,
        "geometry tensor fingerprints mismatch",
    )
    product = hessian @ hessian.T
    matrix_error = float((matrix - product).abs().max())
    symmetry_error = float((matrix - matrix.T).abs().max())
    require(matrix_error <= ABSOLUTE_TOLERANCE, "M != H H^T")
    require(symmetry_error <= ABSOLUTE_TOLERANCE, "M is not symmetric")
    recomputed_eig = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
    eig_error = float((eig - recomputed_eig).abs().max())
    require(eig_error <= ABSOLUTE_TOLERANCE, "stored spectrum != eigvalsh(M)")
    low_mask = eig < LOW_THRESHOLD
    require(int(low_mask.sum()) == basis.shape[1], "low basis count mismatch")
    basis_error = float(
        (basis.T @ basis - torch.eye(basis.shape[1], dtype=torch.float64)).abs().max()
    )
    residual = float(
        (matrix @ basis - basis * recomputed_eig[low_mask].unsqueeze(0)).norm()
        / matrix.norm().clamp_min(1e-30)
    )
    projector_error = float((projector - basis @ basis.T).abs().max())
    require(basis_error <= ABSOLUTE_TOLERANCE, "low basis is not orthonormal")
    require(residual <= ABSOLUTE_TOLERANCE, "low basis eigen residual too large")
    require(projector_error <= ABSOLUTE_TOLERANCE, "projector/basis closure failed")

    replay = validate_replayed_spectrum_csv(
        replay_csv, authoritative_spectrum, eig, root=root
    )
    metrics = payload.get("metrics")
    require(isinstance(metrics, Mapping), "geometry metrics missing")
    closures = _geometry_metric_closure(hessian, matrix, eig, basis)
    expected_metric_names = set(closures) | {
        "task_loss",
        "hessian_sec",
        "low_basis_hash",
    }
    require(set(metrics) == expected_metric_names, "geometry metric schema mismatch")
    errors = {
        name: abs(finite_float(metrics[name]) - value)
        for name, value in closures.items()
    }
    require(
        max(errors.values()) <= ABSOLUTE_TOLERANCE,
        "geometry metric recomputation failed",
    )
    source_errors = {
        name: abs(finite_float(metrics[name]) - finite_float(final_state[name]))
        for name in metrics
        if name not in {"hessian_sec", "low_basis_hash"}
    }
    require(
        max(source_errors.values()) <= ABSOLUTE_TOLERANCE,
        "geometry/source metric closure failed",
    )
    require(finite_float(metrics["hessian_sec"]) >= 0.0, "geometry timing is negative")
    projector_hash = tensor_bytes_sha256(projector)
    require(
        metrics["low_basis_hash"] == final_state["low_basis_hash"] == projector_hash,
        "projector hash chain mismatch",
    )
    return {
        "matrix_from_hessian_max_abs_error": matrix_error,
        "matrix_symmetry_max_abs_error": symmetry_error,
        "eigvalsh_matrix_max_abs_error": eig_error,
        "low_basis_orthogonality_max_abs": basis_error,
        "low_basis_eigen_residual_relative": residual,
        "low_projector_basis_max_abs_error": projector_error,
        "metric_closure_max_abs_error": max(errors.values()),
        "source_metric_max_abs_error": max(source_errors.values()),
        "spectrum_max_abs_error": replay["max_absolute_error"],
        "spectrum": replay,
        "fingerprints": fingerprints,
        "projector_sha256": projector_hash,
    }


def validate_geometry_and_final_checkpoint(
    packet: Path, checkpoint_audit: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    progress = checkpoint_audit["progress"]
    endpoint = [
        row
        for row in progress["spectrum_rows"]
        if exact_int(row["accepted_update"]) == ACCEPTED_UPDATES
    ]
    require(
        len(endpoint) == DIMENSION
        and [exact_int(row["rank"]) for row in endpoint] == list(range(DIMENSION)),
        "endpoint spectrum grid mismatch",
    )
    authoritative = [finite_float(row["m_eigenvalue"]) for row in endpoint]
    geometry_payload = load_checkpoint(packet / "replayed_final_geometry.pt", root=root)
    geometry = audit_geometry_payload(
        geometry_payload,
        authoritative,
        progress["state_rows"][-1],
        packet / "replayed_final_spectrum.csv",
        root=root,
        strict_identity=True,
    )
    final = load_checkpoint(packet / "final_checkpoint.pt", root=root)
    expected_final_keys = {
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
    require(set(final) == expected_final_keys, "final checkpoint schema mismatch")
    expected = {
        "protocol_id": RECOVERY_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": SOURCE_PROGRESS_SHA256,
        "parent_checkpoint_sha256": PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": PARENT_PROGRESS_SHA256,
        "selected_arm": "low",
        "accepted_updates": ACCEPTED_UPDATES,
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "termination": "max_updates_reached",
        "transition_chain_sha256": TRANSITION_CHAIN_SHA256,
        "active_parameter_hash": ACTIVE_PARAMETER_SHA256,
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "z_sha256": Z_SHA256,
        "accepted_vae_checkpoint_sha256": ACCEPTED_CHECKPOINT_SHA256,
    }
    for name, value in expected.items():
        require(
            deep_exact(final[name], value),
            f"final checkpoint metadata mismatch: {name}",
        )
    require(
        deep_exact(final["active_model_state"], progress["active_model_state"]),
        "final active tensors differ",
    )
    final_hash, final_count = named_tensor_sha256(final["active_model_state"])
    require(
        final_hash == ACTIVE_PARAMETER_SHA256 and final_count == ACTIVE_PARAMETER_COUNT,
        "final active hash mismatch",
    )
    require(
        deep_exact(final["stored_state_metrics"], progress["state_rows"][-1]),
        "stored final metrics mismatch",
    )
    return {
        "geometry": geometry,
        "geometry_payload": geometry_payload,
        "final_checkpoint_sha256": sha256_file(
            packet / "final_checkpoint.pt", root=root
        ),
        "bitwise_active_tensor_lineage": True,
    }


def _chain_hash(previous: str, proposal: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "previous": previous,
            "update": exact_int(proposal["target_update"]),
            "base_parameter_hash": proposal["base_parameter_hash"],
            "endpoint_parameter_hash": proposal["endpoint_parameter_hash"],
            "alpha": finite_float(proposal["selected_alpha"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _objective(row: Mapping[str, Any], name: str) -> float:
    key = {
        "A": "exact_a_per_dim",
        "B": "damped_full_burg_per_dim",
        "L_low": "frozen_low_energy",
    }[name]
    value = finite_float(row[key])
    return -value if name == "L_low" else value


def _line_gate_map(
    current: Mapping[str, Any],
    line: Mapping[str, Any],
    proposal: Mapping[str, Any],
    tolerances: Mapping[str, Any],
) -> dict[str, bool]:
    alpha = finite_float(line["alpha"])
    result: dict[str, bool] = {}
    for name in ("A", "B", "L_low"):
        slope = finite_float(proposal[f"slope_{name}"])
        before = _objective(current, name)
        after = _objective(line, name)
        result[f"{name}_slope_negative"] = slope < 0.0
        result[f"{name}_armijo"] = after <= before + ARMIJO_C1 * alpha * slope
        result[f"{name}_actual_decrease"] = before - after > finite_float(
            tolerances[name]
        )
    result["high_tail_nonincrease"] = finite_float(line["a_gt1"]) <= finite_float(
        current["a_gt1"]
    ) + finite_float(tolerances["A_gt1"])
    result["low90_decreases"] = True
    result["exact_a_closes"] = (
        finite_float(line["a_direct_abs_error"]) <= ABSOLUTE_TOLERANCE
        and finite_float(line["a_trace_abs_error"]) <= ABSOLUTE_TOLERANCE
    )
    result["spectrum_finite_nonnegative"] = finite_float(line["m_raw_eig_min"]) >= -1e-8
    numeric = [
        value
        for key, value in line.items()
        if key
        not in {
            "selected_arm",
            "low_basis_hash",
            "endpoint_parameter_hash",
            "phase",
            "base_parameter_hash",
        }
        and value is not None
        and isinstance(value, (int, float))
    ]
    result["metrics_finite"] = all(math.isfinite(float(value)) for value in numeric)
    return result


def _repeat_pass(line: Mapping[str, Any], tolerances: Mapping[str, Any]) -> bool:
    repeated = {key: value for key, value in line.items() if key.startswith("repeat_")}
    require(repeated, "line repeat payload is missing")
    limits = {
        "repeat_exact_a_per_dim_abs_error": finite_float(tolerances["A"]),
        "repeat_damped_full_burg_per_dim_abs_error": finite_float(tolerances["B"]),
        "repeat_frozen_low_energy_abs_error": finite_float(tolerances["L_low"]),
        "repeat_a_low90_abs_per_dim_abs_error": finite_float(tolerances["A_low90"]),
        "repeat_a_gt1_abs_error": finite_float(tolerances["A_gt1"]),
        "repeat_m_max_abs_error": finite_float(tolerances["m_max"]),
        "repeat_m_p50_abs_error": finite_float(tolerances["m_p50"]),
        "repeat_effective_rank_abs_error": finite_float(tolerances["effective_rank"]),
    }
    exact = {
        "repeat_count_lt_1e_4_abs_error",
        "repeat_count_lt_1e_2_abs_error",
        "repeat_count_lt_0p1_abs_error",
    }
    for name, value in repeated.items():
        number = finite_float(value)
        if name == "repeat_parameter_hash_unchanged":
            require(number == 1.0, "repeat parameter fingerprint changed")
        elif name in exact:
            require(number == 0.0, f"repeat integral closure failed: {name}")
        elif name in limits:
            require(0.0 <= number <= limits[name], f"repeat tolerance failed: {name}")
        else:
            require(0.0 <= number <= 1e-10, f"repeat closure failed: {name}")
    return True


def _selected_line_matches_state(
    line: Mapping[str, Any], state_row: Mapping[str, Any]
) -> None:
    excluded = {
        "accepted_update",
        "parameter_hash",
        "parent_frozen_low_energy",
        "phase",
    }
    for name, value in state_row.items():
        if name in excluded or name not in line:
            continue
        if name == "frozen_low_energy":
            continue
        if isinstance(value, str):
            require(line[name] == value, f"selected line/state text mismatch: {name}")
        else:
            require(
                close(line[name], value), f"selected line/state metric mismatch: {name}"
            )
    require(
        close(line["frozen_low_energy"], state_row["parent_frozen_low_energy"]),
        "selected line frozen-low parent link mismatch",
    )
    require(
        close(line["canonical_low_energy"], state_row["frozen_low_energy"])
        and close(line["canonical_low_energy"], state_row["canonical_low_energy"]),
        "selected line canonical-low state link mismatch",
    )


def validate_history_and_outcome(
    packet: Path,
    progress: Mapping[str, Any],
    geometry_payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    states = progress["state_rows"]
    spectra = progress["spectrum_rows"]
    proposals = progress["proposal_rows"]
    lines = progress["line_rows"]
    tolerances = progress["tolerances"]
    require(len(proposals) == ACCEPTED_UPDATES, "proposal count mismatch")
    require(
        [exact_int(row["target_update"]) for row in proposals]
        == list(range(1, ACCEPTED_UPDATES + 1)),
        "proposal physical order mismatch",
    )
    require(
        all(exact_int(row["accepted"]) == 1 for row in proposals),
        "unexpected rejected proposal",
    )
    line_groups: dict[int, list[Mapping[str, Any]]] = {}
    previous_target = 0
    for line in lines:
        target = exact_int(line["target_update"])
        require(target >= previous_target, "line target order mismatch")
        previous_target = target
        line_groups.setdefault(target, []).append(line)
    require(set(line_groups) == set(range(1, 101)), "line target set mismatch")
    chain = PARENT_FINAL_SHA256
    history_rows: list[dict[str, Any]] = []
    radius_errors: list[tuple[int, float]] = []
    for proposal in proposals:
        target = exact_int(proposal["target_update"])
        current = states[target - 1]
        committed = states[target]
        candidates = line_groups[target]
        alphas = [finite_float(row["alpha"]) for row in candidates]
        require(
            alphas == list(LINE_ALPHAS[: len(alphas)]),
            f"line alpha prefix mismatch: {target}",
        )
        pass_bits: list[bool] = []
        for line in candidates:
            gates = _line_gate_map(current, line, proposal, tolerances)
            for name, value in gates.items():
                require(
                    exact_int(line[f"gate_{name}"]) == int(value),
                    f"line gate mismatch: {target}/{name}",
                )
            repeated = _repeat_pass(line, tolerances)
            passes = all(gates.values()) and repeated
            require(
                exact_int(line["passes"]) == int(passes),
                f"line pass bit mismatch: {target}",
            )
            pass_bits.append(passes)
        require(
            pass_bits[-1] and not any(pass_bits[:-1]),
            f"proposal did not take first passing line: {target}",
        )
        selected = candidates[-1]
        alpha = finite_float(selected["alpha"])
        require(
            same_float(proposal["selected_alpha"], alpha),
            f"selected alpha mismatch: {target}",
        )
        require(
            selected["endpoint_parameter_hash"] == committed["parameter_hash"],
            "endpoint/state hash mismatch",
        )
        _selected_line_matches_state(selected, committed)
        require(
            close(proposal["current_A"], current["exact_a_per_dim"])
            and close(proposal["current_B"], current["damped_full_burg_per_dim"])
            and close(proposal["current_low_energy"], current["frozen_low_energy"])
            and close(proposal["current_a_gt1"], current["a_gt1"]),
            f"proposal current-state link mismatch: {target}",
        )
        require(
            close(proposal["candidate_A"], selected["exact_a_per_dim"])
            and close(proposal["candidate_B"], selected["damped_full_burg_per_dim"])
            and close(proposal["candidate_low_energy"], selected["frozen_low_energy"])
            and close(proposal["candidate_a_gt1"], selected["a_gt1"]),
            f"proposal candidate link mismatch: {target}",
        )
        if target > START_UPDATE:
            require(
                proposal["phase"] == "relaxed_continuation",
                "continuation proposal phase mismatch",
            )
            require(
                proposal["base_parameter_hash"] == current["parameter_hash"],
                "base hash mismatch",
            )
            require(
                proposal["endpoint_parameter_hash"] == committed["parameter_hash"],
                "proposal endpoint mismatch",
            )
            require(
                close(
                    proposal["realized_path_length"], alpha * TARGET_NORM, atol=2e-12
                ),
                "realized path length mismatch",
            )
            radius_errors.append(
                (target, abs(finite_float(proposal["direction_norm"]) - TARGET_NORM))
            )
            chain = _chain_hash(chain, proposal)
            require(
                proposal["transition_chain_sha256"] == chain,
                "transition chain mismatch",
            )
        component: dict[str, bool] = {}
        for name in ("A", "B", "L_low"):
            slope = finite_float(proposal[f"slope_{name}"])
            component[f"{name}_slope_negative"] = slope < 0.0
            component[f"{name}_armijo"] = (
                _objective(selected, name)
                <= _objective(current, name) + ARMIJO_C1 * alpha * slope
            )
            component[f"{name}_actual_decrease"] = _objective(
                current, name
            ) - _objective(selected, name) > finite_float(tolerances[name])
        high_tail = finite_float(selected["a_gt1"]) <= finite_float(
            current["a_gt1"]
        ) + finite_float(tolerances["A_gt1"])
        require(
            all(component.values()) and high_tail,
            f"accepted transition failed: {target}",
        )
        history_rows.append(
            {
                "accepted_update": target,
                "alpha": alpha,
                **{name: int(value) for name, value in component.items()},
                "high_tail_nonincrease": int(high_tail),
                "selected_line_row_passes": 1,
                "committed_state_matches": 1,
                "transition_passes": 1,
            }
        )
    require(
        chain == progress["transition_chain_sha256"] == TRANSITION_CHAIN_SHA256,
        "final chain mismatch",
    )
    validate_rows_csv(
        history_rows, packet / "historical_transition_audit.csv", root=root
    )
    require(
        len(radius_errors) == NEW_ACCEPTED_UPDATES, "continuation radius count mismatch"
    )
    radius_max_target, radius_max = max(radius_errors, key=lambda item: item[1])
    require(
        abs(radius_max - 2.7241507938313703e-9) <= 5e-16
        and radius_max_target == 74
        and sum(error > 1e-9 for _, error in radius_errors) == 37
        and sum(error > RADIUS_TOLERANCE for _, error in radius_errors) == 0,
        "continuation direction-radius audit mismatch",
    )

    initial = states[0]
    intervention = states[START_UPDATE]
    final = states[-1]
    total_a = finite_float(initial["exact_a_per_dim"]) - finite_float(
        final["exact_a_per_dim"]
    )
    total_low = finite_float(initial["a_low90_abs_per_dim"]) - finite_float(
        final["a_low90_abs_per_dim"]
    )
    continuation_a = finite_float(intervention["exact_a_per_dim"]) - finite_float(
        final["exact_a_per_dim"]
    )
    continuation_low = finite_float(intervention["a_low90_abs_per_dim"]) - finite_float(
        final["a_low90_abs_per_dim"]
    )
    low_decreases = [
        finite_float(states[index - 1]["a_low90_abs_per_dim"])
        - finite_float(states[index]["a_low90_abs_per_dim"])
        for index in range(1, len(states))
    ]
    tail_bulk = sum(
        low_decreases[index - 1] > finite_float(tolerances["A_low90"])
        for index in range(81, 101)
    )
    a_diffs = [
        finite_float(states[index]["exact_a_per_dim"])
        - finite_float(states[index - 1]["exact_a_per_dim"])
        for index in range(1, len(states))
    ]
    high_diffs = [
        finite_float(states[index]["a_gt1"]) - finite_float(states[index - 1]["a_gt1"])
        for index in range(1, len(states))
    ]
    bulk_fraction = total_low / max(total_a, 1e-30)
    success = {
        "one_hundred_accepted_updates": len(states) - 1 == ACCEPTED_UPDATES,
        "tail_bulk_activity": tail_bulk >= 4,
        "final_a_at_most_0p90": finite_float(final["exact_a_per_dim"]) <= 0.90,
        "accepted_a_strictly_decreases": all(
            value < -finite_float(tolerances["A"]) for value in a_diffs
        ),
        "all_historical_a_armijo_transitions": all(
            row["A_slope_negative"] == row["A_armijo"] == row["A_actual_decrease"] == 1
            for row in history_rows
        ),
        "final_b_below_initial": finite_float(final["damped_full_burg_per_dim"])
        < finite_float(initial["damped_full_burg_per_dim"])
        - finite_float(tolerances["B"]),
        "non_top_only_fraction_at_least_0p25": bulk_fraction >= 0.25,
        "final_p50_above_initial": finite_float(final["m_p50"])
        > finite_float(initial["m_p50"]) + finite_float(tolerances["m_p50"]),
        "final_effective_rank_above_initial": finite_float(final["effective_rank"])
        > finite_float(initial["effective_rank"])
        + finite_float(tolerances["effective_rank"]),
        "final_count_lt_1e_4_strictly_lower": exact_int(final["count_lt_1e_4"])
        < exact_int(initial["count_lt_1e_4"]),
        "final_count_lt_1e_2_strictly_lower": exact_int(final["count_lt_1e_2"])
        < exact_int(initial["count_lt_1e_2"]),
        "final_count_lt_0p1_strictly_lower": exact_int(final["count_lt_0p1"])
        < exact_int(initial["count_lt_0p1"]),
        "final_mmax_not_above_initial": finite_float(final["m_max"])
        <= finite_float(initial["m_max"]) + finite_float(tolerances["m_max"]),
        "high_tail_never_increases_beyond_floor": all(
            value <= finite_float(tolerances["A_gt1"]) for value in high_diffs
        ),
        "all_state_values_finite": all(
            math.isfinite(float(value))
            for row in states
            for key, value in row.items()
            if key not in {"parameter_hash", "low_basis_hash", "phase"}
        ),
        "all_spectrum_values_finite": all(
            math.isfinite(float(value))
            for row in spectra
            for key, value in row.items()
            if key != "phase"
        ),
        "all_raw_spectrum_minima_valid": all(
            finite_float(row["m_raw_eig_min"]) >= -1e-8 for row in states
        ),
        "all_exact_a_closures_valid": all(
            finite_float(row["a_direct_abs_error"]) <= ABSOLUTE_TOLERANCE
            and finite_float(row["a_trace_abs_error"]) <= ABSOLUTE_TOLERANCE
            for row in states
        ),
        "final_checkpoint_replay": True,
    }
    failed = [name for name, passed in success.items() if not passed]
    require(failed == FAILED_SCIENTIFIC_GATES, "scientific failed-gate set changed")
    continuation = proposals[START_UPDATE:]
    valid_shadow = [row for row in continuation if row.get("fp64_shadow_valid") is True]
    accepted_alpha = [finite_float(row["selected_alpha"]) for row in continuation]
    near = {
        "minimum_source_norm": min(
            finite_float(row["unit_common_source_norm"]) for row in continuation
        ),
        "final_source_norm": finite_float(continuation[-1]["unit_common_source_norm"]),
        "maximum_amplification": max(
            finite_float(row["common_amplification"]) for row in continuation
        ),
        "maximum_amplification_unbounded": False,
        "minimum_fp32_fp64_direction_cosine": min(
            finite_float(row["fp32_fp64_direction_cosine"]) for row in valid_shadow
        ),
        "maximum_fp32_fp64_direction_relative_error": max(
            finite_float(row["fp32_fp64_direction_relative_error"])
            for row in valid_shadow
        ),
        "minimum_accepted_alpha": min(accepted_alpha),
        "descriptive_only": True,
    }
    outcome = {
        "immediate_premature_cutoff": True,
        "sustained_continuation": True,
        "b_non_descent": False,
        "finite_grid_exhaustion": False,
        "finite_grid_conflict": False,
        "scalar_only_repair": continuation_a > finite_float(tolerances["A"])
        and bulk_fraction < 0.25,
        "new_accepted_updates": NEW_ACCEPTED_UPDATES,
        "scientific_success": False,
        "success_gates": success,
        "termination": "max_updates_reached",
        "total_a_reduction": total_a,
        "total_lower90_reduction": total_low,
        "total_non_top_only_fraction": bulk_fraction,
        "continuation_a_reduction": continuation_a,
        "continuation_lower90_reduction": continuation_low,
        "continuation_non_top_only_fraction": continuation_low
        / max(continuation_a, 1e-30),
        "tail_bulk_updates": tail_bulk,
        "near_cancellation_diagnostics": near,
    }
    decision = load_json(packet / "decision.json", root=root)
    require(
        decision.get("recovery_valid") is True
        and decision.get("scientific_success") is False,
        "decision identity mismatch",
    )
    require(
        deep_close(decision.get("outcome"), outcome),
        "scientific outcome recomputation mismatch",
    )
    final_metrics = decision.get("final_metrics")
    require(isinstance(final_metrics, Mapping), "decision final metrics missing")
    require(
        deep_close(final_metrics, geometry_payload["metrics"]),
        "decision/geometry metrics mismatch",
    )
    require(
        same_float(final["exact_a_per_dim"], 0.9321672207645146)
        and close(
            outcome["continuation_non_top_only_fraction"],
            0.11301885339451084,
            atol=1e-15,
        )
        and close(
            outcome["total_non_top_only_fraction"], 0.07733408144475684, atol=1e-15
        ),
        "frozen scientific scalar pins changed",
    )
    return {
        "outcome": outcome,
        "failed_scientific_gates": failed,
        "history_rows": len(history_rows),
        "transition_chain_sha256": chain,
        "radius_max_abs_error": radius_max,
        "radius_max_target_update": radius_max_target,
    }


def validate_snapshot_record(snapshot: Mapping[str, Any], *, label: str) -> None:
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
    require(set(snapshot) == expected_keys, f"{label} snapshot schema mismatch")
    digest = hashlib.sha256()
    tensors = 0
    elements = 0
    for category in ("parameters", "buffers"):
        values = snapshot[category]
        require(isinstance(values, Mapping), f"{label} {category} missing")
        category_elements = 0
        for name in sorted(values):
            fingerprint = values[name]
            require(
                isinstance(fingerprint, Mapping)
                and set(fingerprint) == {"dtype", "shape", "numel", "sha256"}
                and isinstance(fingerprint["dtype"], str)
                and isinstance(fingerprint["shape"], list)
                and math.prod(fingerprint["shape"]) == exact_int(fingerprint["numel"])
                and is_sha256(fingerprint["sha256"]),
                f"{label} fingerprint malformed: {name}",
            )
            digest.update(category.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(json.dumps(fingerprint, sort_keys=True).encode("utf-8"))
            tensors += 1
            category_elements += exact_int(fingerprint["numel"])
        singular = category[:-1]
        require(
            snapshot[f"{singular}_tensor_count"] == len(values)
            and snapshot[f"{singular}_element_count"] == category_elements,
            f"{label} category count mismatch",
        )
        elements += category_elements
    require(
        snapshot["all_tensor_count"] == tensors
        and snapshot["all_element_count"] == elements
        and snapshot["aggregate_sha256"] == digest.hexdigest(),
        f"{label} aggregate fingerprint mismatch",
    )


def validate_nonmutation_evidence(
    reconstruction: Mapping[str, Any],
    progress: Mapping[str, Any],
    geometry_payload: Mapping[str, Any],
) -> dict[str, Any]:
    snapshots = reconstruction.get("model_tensor_snapshots")
    require(
        isinstance(snapshots, Mapping)
        and set(snapshots) == {"base", "installed", "post_replay"},
        "tensor snapshot set mismatch",
    )
    for name, snapshot in snapshots.items():
        validate_snapshot_record(snapshot, label=name)
    base = snapshots["base"]
    installed = snapshots["installed"]
    post = snapshots["post_replay"]
    require(installed == post, "stored tensor fingerprints changed during replay")
    require(
        set(base["parameters"]) == set(installed["parameters"]),
        "parameter names changed",
    )
    require(set(base["buffers"]) == set(installed["buffers"]), "buffer names changed")
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
    require(
        changed_parameters == sorted(progress["active_model_state"]),
        "installed active set mismatch",
    )
    require(
        len(changed_parameters) == ACTIVE_TENSOR_COUNT and not changed_buffers,
        "installation change count mismatch",
    )
    installation = reconstruction["model_installation_audit"]
    require(
        installation.get("pass") is True
        and installation.get("changed_parameters") == changed_parameters
        and installation.get("changed_parameter_count") == ACTIVE_TENSOR_COUNT
        and installation.get("changed_buffers") == []
        and installation.get("changed_buffer_count") == 0
        and installation.get("installed_aggregate_sha256")
        == installed["aggregate_sha256"],
        "installation audit mismatch",
    )
    nonmutation = reconstruction["model_replay_nonmutation_audit"]
    require(
        nonmutation.get("pass") is True
        and nonmutation.get("all_parameters_bitwise_unchanged") is True
        and nonmutation.get("all_buffers_bitwise_unchanged") is True
        and nonmutation.get("installed_aggregate_sha256")
        == nonmutation.get("post_replay_aggregate_sha256")
        == installed["aggregate_sha256"],
        "non-mutation audit mismatch",
    )
    require(
        geometry_payload["installed_model_aggregate_sha256"]
        == geometry_payload["post_replay_model_aggregate_sha256"]
        == installed["aggregate_sha256"],
        "geometry aggregate fingerprint link mismatch",
    )
    return {
        "parameter_tensor_count": installed["parameter_tensor_count"],
        "buffer_tensor_count": installed["buffer_tensor_count"],
        "aggregate_sha256": installed["aggregate_sha256"],
        "changed_parameter_count": len(changed_parameters),
    }


def _validate_reconstruction_inputs(record: Mapping[str, Any]) -> dict[str, str]:
    expected_hashes = {
        "config.json": "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4",
        "weight_pool.pt": "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef",
        "weight_pool_records.csv": "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933",
        "vae_checkpoint.pt": ACCEPTED_CHECKPOINT_SHA256,
    }
    pre_load_key = "pre_" + "load_run"
    post_load_key = "post_" + "load_run"
    require(
        set(record) == {"pass", "pre_post_equal", pre_load_key, post_load_key}
        and record["pass"] is True
        and record["pre_post_equal"] is True
        and record[pre_load_key] == record[post_load_key],
        "reconstruction input pre/post record mismatch",
    )
    audit = record[pre_load_key]
    require(
        isinstance(audit, Mapping)
        and set(audit) == {"pass", "run_dir", "file_count", "files"}
        and audit["pass"] is True
        and audit["run_dir"] == str(ACCEPTED_RUN)
        and audit["file_count"] == 4,
        "reconstruction input audit schema mismatch",
    )
    expected_files = {
        name: {"path": str(ACCEPTED_RUN / name), "sha256": digest}
        for name, digest in expected_hashes.items()
    }
    require(audit["files"] == expected_files, "reconstruction input identity mismatch")
    return expected_hashes


def _validate_task_records(
    reconstruction: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    labels = ("task_at_load", "task_immediately_pre_replay", "task_post_replay")
    records = [reconstruction[name] for name in labels]
    require(
        records[0] == records[1] == records[2],
        "task fingerprints changed around replay",
    )
    record = records[0]
    require(
        record.get("pass") is True
        and isinstance(record.get("gates"), Mapping)
        and all(value is True for value in record["gates"].values())
        and record.get("task_name") == TASK_NAME
        and record.get("source_weight_index") == SOURCE_WEIGHT_INDEX
        and same_float(record.get("tau"), TASK_TAU)
        and record.get("train_sample_count") == 16_384
        and record.get("test_sample_count") == 4_096
        and record.get("selected_task_tensors") == task["selected_task_tensors"]
        and record.get("accepted_vae_checkpoint_sha256") == ACCEPTED_CHECKPOINT_SHA256,
        "task reconstruction record mismatch",
    )


def validate_recovery_evidence(
    packet: Path,
    checkpoint: Mapping[str, Any],
    geometry: Mapping[str, Any],
    science: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    recovery = load_json(packet / "recovery_audit.json", root=root)
    expected_keys = {
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
    require(set(recovery) == expected_keys, "recovery evidence schema mismatch")
    require(
        recovery["protocol_id"] == RECOVERY_PROTOCOL_ID
        and recovery["source_protocol_id"] == SOURCE_PROTOCOL_ID
        and recovery["recovery_valid"] is True,
        "recovery evidence identity mismatch",
    )
    original = recovery["original_runner_audit"]
    require(
        recovery["original_runner_audit_pass"] is False
        and recovery["original_runner_failed_gates"]
        == ["continuation_diagnostics_recompute"]
        and original.get("pass") is False
        and original.get("failed_gates") == ["continuation_diagnostics_recompute"]
        and original.get("all_unmodified_row_gates_pass") is True
        and original.get("audit_invocations") == 1
        and original.get("transition_chain_sha256") == TRANSITION_CHAIN_SHA256,
        "original expected failure evidence changed",
    )
    row_gates = original["row_gates"]
    require(
        [name for name, passed in row_gates.items() if passed is not True]
        == ["continuation_diagnostics_recompute"],
        "original failure gate set mismatch",
    )
    independent = recovery["independent_recovery_audit"]
    require(
        independent.get("pass") is True
        and all(value is True for value in independent.get("gates", {}).values())
        and all(value is True for value in independent.get("prefix_gates", {}).values())
        and independent.get("transition_chain_sha256") == TRANSITION_CHAIN_SHA256,
        "stored independent recovery evidence invalid",
    )
    radius = independent["radius_audit"]
    require(
        radius.get("pass") is True
        and same_float(radius.get("absolute_tolerance"), RADIUS_TOLERANCE)
        and same_float(radius.get("relative_tolerance"), 0.0)
        and radius.get("row_count") == NEW_ACCEPTED_UPDATES
        and close(
            radius.get("max_absolute_error"),
            science["radius_max_abs_error"],
            atol=5e-16,
        )
        and radius.get("max_error_target_update") == science["radius_max_target_update"]
        and radius.get("count_exceeding_original_1e_9") == 37
        and radius.get("count_exceeding_recovery_5e_9") == 0,
        "stored radius evidence mismatch",
    )
    execution = recovery["geometry_execution"]
    require(
        execution
        == {
            "geometry_evaluation_count": 1,
            "forbidden_operation_attempts": 0,
            "optimization_gradient_evaluations": 0,
            "proposals": 0,
            "line_searches": 0,
            "parameter_updates": 0,
        },
        "single-replay execution budget mismatch",
    )

    replay = recovery["final_geometry_replay"]
    observed_geometry = geometry["geometry"]
    require(
        replay.get("pass") is True
        and replay.get("active_parameter_hash_matches") is True
        and replay.get("low_basis_hash_matches") is True
        and replay.get("hessian_shape") == [DIMENSION, DIMENSION]
        and replay.get("matrix_shape") == [DIMENSION, DIMENSION]
        and replay.get("spectrum_shape") == [DIMENSION]
        and replay.get("low_basis_shape") == [DIMENSION, LOW_COUNT]
        and replay.get("low_projector_shape") == [DIMENSION, DIMENSION]
        and same_float(replay.get("metric_absolute_tolerance"), ABSOLUTE_TOLERANCE)
        and same_float(replay.get("spectrum_absolute_tolerance"), ABSOLUTE_TOLERANCE)
        and replay.get("spectrum_eigenvalue_count") == DIMENSION
        and same_float(replay.get("spectrum_max_absolute_error"), 0.0),
        "final replay summary mismatch",
    )
    fingerprints = observed_geometry["fingerprints"]
    require(
        replay.get("geometry_artifact_sha256")
        == sha256_file(packet / "replayed_final_geometry.pt", root=root)
        and replay.get("replayed_final_spectrum_sha256")
        == sha256_file(packet / "replayed_final_spectrum.csv", root=root)
        and replay.get("replayed_final_spectrum_rows") == DIMENSION
        and replay.get("hessian_sha256") == fingerprints["hessian"]["sha256"]
        and replay.get("matrix_sha256") == fingerprints["matrix"]["sha256"]
        and replay.get("spectrum_tensor_sha256") == fingerprints["eig"]["sha256"]
        and replay.get("low_basis_sha256")
        == fingerprints["current_low_basis"]["sha256"]
        and replay.get("low_projector_sha256")
        == fingerprints["current_low_projector"]["sha256"],
        "final replay artifact links mismatch",
    )
    self_audit = replay["geometry_artifact_audit"]
    numeric_links = {
        "matrix_from_hessian_max_abs_error": "matrix_from_hessian_max_abs_error",
        "matrix_symmetry_max_abs_error": "matrix_symmetry_max_abs_error",
        "eigvalsh_matrix_max_abs_error": "eigvalsh_matrix_max_abs_error",
        "low_projector_basis_max_abs_error": "low_projector_basis_max_abs_error",
        "low_basis_orthogonality_max_abs": "low_basis_orthogonality_max_abs",
        "low_basis_eigen_residual_relative": "low_basis_eigen_residual_relative",
        "metric_closure_max_abs_error": "metric_closure_max_abs_error",
        "source_metric_max_abs_error": "source_metric_max_abs_error",
        "stored_spectrum_max_abs_error": "spectrum_max_abs_error",
    }
    for stored_name, observed_name in numeric_links.items():
        require(
            close(self_audit[stored_name], observed_geometry[observed_name]),
            f"geometry self-audit mismatch: {stored_name}",
        )
    require(
        self_audit.get("tensor_fingerprints") == fingerprints,
        "self-audit tensor fingerprints mismatch",
    )

    checkpoint_lineage = recovery["checkpoint_lineage"]
    require(
        checkpoint_lineage.get("pass") is True
        and checkpoint_lineage.get("bitwise_tensor_equality") is True
        and checkpoint_lineage.get("cpu_reload_pass") is True
        and checkpoint_lineage.get("active_tensor_count") == ACTIVE_TENSOR_COUNT
        and checkpoint_lineage.get("active_parameter_count") == ACTIVE_PARAMETER_COUNT
        and checkpoint_lineage.get("active_parameter_hash") == ACTIVE_PARAMETER_SHA256
        and checkpoint_lineage.get("final_checkpoint_sha256")
        == geometry["final_checkpoint_sha256"]
        and all(
            value is True
            for value in checkpoint_lineage.get("metadata_gates", {}).values()
        ),
        "checkpoint lineage evidence mismatch",
    )

    reconstruction = recovery["geometry_reconstruction"]
    require(
        reconstruction.get("accepted_checkpoint_sha256")
        == reconstruction.get("expected_accepted_checkpoint_sha256")
        == ACCEPTED_CHECKPOINT_SHA256
        and reconstruction.get("z_sha256") == Z_SHA256
        and reconstruction.get("active_tensor_count") == ACTIVE_TENSOR_COUNT
        and reconstruction.get("active_parameter_count") == ACTIVE_PARAMETER_COUNT
        and reconstruction.get("installed_active_parameter_hash")
        == reconstruction.get("post_replay_active_parameter_hash")
        == ACTIVE_PARAMETER_SHA256
        and reconstruction.get("base_parameter_gradient_slots") == 0
        and reconstruction.get("post_replay_parameter_gradient_slots") == 0
        and reconstruction.get("full_ce_batch") is True,
        "geometry reconstruction identity mismatch",
    )
    task_manifest = load_json(RECOVERY_TASK_MANIFEST, root=root)
    _validate_task_records(reconstruction, task_manifest)
    input_hashes = _validate_reconstruction_inputs(
        reconstruction["reconstruction_input_files"]
    )
    require(
        recovery["reconstruction_input_files"]
        == reconstruction["reconstruction_input_files"],
        "top-level reconstruction input link mismatch",
    )
    nonmutation = validate_nonmutation_evidence(
        reconstruction, checkpoint["progress"], geometry["geometry_payload"]
    )

    before = recovery["immutable_input_preflight"]
    after = recovery["immutable_input_postflight"]
    require(before == after, "immutable input pre/post evidence differs")
    require(
        before.get("manifest_sha256") == RECOVERY_INPUT_MANIFEST_SHA256
        and before.get("source_file_count") == 14
        and before.get("source_staging") == str(ORIGINAL_SOURCE)
        and before.get("source_files_sha256") == source_manifest["files_sha256"],
        "immutable input evidence mismatch",
    )
    gates = recovery["recovery_validity_gates"]
    require(
        isinstance(gates, Mapping)
        and len(gates) == 28
        and all(value is True for value in gates.values()),
        "recovery validity gates failed",
    )
    require(
        set(gates)
        == {
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
            "reconstruction_inputs_exact_pre_post_" + "load_run",
            "terminal_install_changes_exact_active_set",
            "all_model_tensors_unchanged_during_replay",
            "exactly_one_geometry_evaluation",
            "no_optimization_operations",
            "final_geometry_replay_pass",
            "geometry_artifact_independent_closure_pass",
            "replayed_spectrum_csv_pass",
            "scientific_success_criteria_unchanged",
        },
        "recovery validity gate schema changed",
    )
    require(
        sha256_file(
            packet / "executed_recovery_finalizer_source_snapshot.py", root=root
        )
        == FINALIZER_SHA256
        and normalized_source_sha256(
            packet / "executed_recovery_finalizer_source_snapshot.py", root=root
        )
        == FINALIZER_NORMALIZED_SHA256,
        "executed finalizer snapshot identity mismatch",
    )
    return {
        "recovery_audit": recovery,
        "execution": execution,
        "nonmutation": nonmutation,
        "reconstruction_input_sha256": input_hashes,
        "all_recovery_gates_true": True,
    }


def validate_resolved_config(packet: Path, *, root: Path = ROOT) -> dict[str, Any]:
    config = load_json(packet / "resolved_config.json", root=root)
    require(
        config.get("protocol_id") == RECOVERY_PROTOCOL_ID
        and config.get("source_protocol_id") == SOURCE_PROTOCOL_ID
        and config.get("source_staging") == str(ORIGINAL_SOURCE)
        and config.get("output_dir") == str(TARGET)
        and config.get("source_staging_mutation_allowed") is False
        and config.get("source_progress_checkpoint_sha256") == SOURCE_PROGRESS_SHA256
        and config.get("frozen_input_manifest_sha256") == RECOVERY_INPUT_MANIFEST_SHA256
        and config.get("recovery_protocol_sha256") == RECOVERY_PROTOCOL_SHA256
        and config.get("recovery_task_manifest_sha256") == RECOVERY_TASK_MANIFEST_SHA256
        and config.get("recovery_import_manifest_sha256")
        == RECOVERY_IMPORT_MANIFEST_SHA256
        and config.get("geometry_evaluation_budget") == 1
        and config.get("optimization_gradient_evaluation_budget") == 0
        and config.get("proposal_budget") == 0
        and config.get("line_search_budget") == 0
        and config.get("parameter_update_budget") == 0
        and same_float(
            config.get("metric_replay_absolute_tolerance"), ABSOLUTE_TOLERANCE
        )
        and same_float(
            config.get("spectrum_replay_absolute_tolerance"), ABSOLUTE_TOLERANCE
        )
        and same_float(
            config.get("direction_radius_absolute_tolerance"), RADIUS_TOLERANCE
        )
        and same_float(config.get("direction_radius_relative_tolerance"), 0.0),
        "resolved replay config mismatch",
    )
    working = Path(config["working_dir"])
    require(
        absolute(working) != TARGET
        and working.name.startswith(
            ".postgoal_relaxed_cancellation_recovery_finalization_v1.incomplete."
        )
        and not os.path.lexists(working),
        "historical working path is invalid or unexpectedly present",
    )
    return config


COPIED_SOURCE_FILES = {
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
    "frozen_producer_dependency_manifest_snapshot.json": "frozen_dependency_manifest_snapshot.json",
    "frozen_producer_protocol_snapshot.md": "protocol_snapshot.md",
}


def validate_copied_source_files(packet: Path, *, root: Path = ROOT) -> dict[str, str]:
    observed: dict[str, str] = {}
    for destination, source_name in COPIED_SOURCE_FILES.items():
        source_hash = sha256_file(ORIGINAL_SOURCE / source_name, root=root)
        destination_hash = sha256_file(packet / destination, root=root)
        require(
            destination_hash == source_hash,
            f"copied source evidence differs: {destination}",
        )
        observed[destination] = destination_hash
    fixed = {
        "recovery_protocol_snapshot.md": RECOVERY_PROTOCOL_SHA256,
        "recovery_frozen_input_manifest_snapshot.json": RECOVERY_INPUT_MANIFEST_SHA256,
        "recovery_task_manifest_snapshot.json": RECOVERY_TASK_MANIFEST_SHA256,
        "recovery_import_manifest_snapshot.json": RECOVERY_IMPORT_MANIFEST_SHA256,
        "executed_recovery_finalizer_source_snapshot.py": FINALIZER_SHA256,
    }
    for name, digest in fixed.items():
        require(
            sha256_file(packet / name, root=root) == digest,
            f"fixed snapshot differs: {name}",
        )
        observed[name] = digest
    return observed


def audit_failed_packet(
    *, root: Path = ROOT, verify_large_inputs: bool = True
) -> dict[str, Any]:
    started = time.perf_counter()
    require(
        absolute(root) == ROOT, "real failed-packet audit root is not production root"
    )
    require(
        accelerator_uninitialized(),
        "accelerator runtime initialized before failed-packet audit",
    )
    documents = validate_frozen_documents(root=root)
    manifest = documents["manifest"]
    protected_before = {
        "failed": audit_tree_against_entries(
            FAILED_SOURCE,
            documents["failed_entries"],
            expected_tree_sha256=FAILED_TREE_SHA256,
            expected_total_size=114_951_202,
            root=root,
        ),
        "original_source": audit_tree(ORIGINAL_SOURCE, root=root),
        "startup_failure": validate_startup_failure(manifest, root=root),
    }
    original = validate_original_source(
        manifest, root=root, verify_large_inputs=verify_large_inputs
    )
    require(
        protected_before["original_source"] == original["tree"],
        "original source audit disagreement",
    )
    failure = validate_failure_localization(FAILED_SOURCE, manifest, root=root)
    graph = validate_packet_hash_graph(
        FAILED_SOURCE, documents["published_names"], root=root
    )
    copied = validate_copied_source_files(FAILED_SOURCE, root=root)
    config = validate_resolved_config(FAILED_SOURCE, root=root)
    checkpoint = validate_checkpoint_and_csv(FAILED_SOURCE, root=root)
    geometry = validate_geometry_and_final_checkpoint(
        FAILED_SOURCE, checkpoint, root=root
    )
    science = validate_history_and_outcome(
        FAILED_SOURCE,
        checkpoint["progress"],
        geometry["geometry_payload"],
        root=root,
    )
    recovery = validate_recovery_evidence(
        FAILED_SOURCE,
        checkpoint,
        geometry,
        science,
        original["frozen_manifest"],
        root=root,
    )
    protected_after = {
        "failed": audit_tree_against_entries(
            FAILED_SOURCE,
            documents["failed_entries"],
            expected_tree_sha256=FAILED_TREE_SHA256,
            expected_total_size=114_951_202,
            root=root,
        ),
        "original_source": audit_tree(ORIGINAL_SOURCE, root=root),
        "startup_failure": validate_startup_failure(manifest, root=root),
    }
    require(
        protected_before == protected_after,
        "protected trees changed during failed-packet audit",
    )
    require(
        accelerator_uninitialized(),
        "accelerator runtime initialized during failed-packet audit",
    )
    numeric = geometry["geometry"]
    endpoint = checkpoint["progress"]["spectrum_rows"][-DIMENSION:]
    numeric_audit = {
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": 0.0,
        "matrix_from_hessian_max_abs_error": numeric[
            "matrix_from_hessian_max_abs_error"
        ],
        "matrix_symmetry_max_abs_error": numeric["matrix_symmetry_max_abs_error"],
        "eigvalsh_matrix_max_abs_error": numeric["eigvalsh_matrix_max_abs_error"],
        "low_basis_orthogonality_max_abs": numeric["low_basis_orthogonality_max_abs"],
        "low_basis_eigen_residual_relative": numeric[
            "low_basis_eigen_residual_relative"
        ],
        "low_projector_basis_max_abs_error": numeric[
            "low_projector_basis_max_abs_error"
        ],
        "metric_closure_max_abs_error": numeric["metric_closure_max_abs_error"],
        "source_metric_max_abs_error": numeric["source_metric_max_abs_error"],
        "spectrum_max_abs_error": numeric["spectrum_max_abs_error"],
        "spectrum_bitwise_equal_count": DIMENSION,
        "spectrum_representation_count": 5,
        "rank_2_binary64_hex": float_bits(endpoint[2]["m_eigenvalue"]).hex(),
        "state_csv_rows": checkpoint["csv"]["state_rows"]["row_count"],
        "spectrum_csv_rows": checkpoint["csv"]["spectrum_rows"]["row_count"],
        "scientific_success": False,
        "failed_scientific_success_gates": FAILED_SCIENTIFIC_GATES,
        "final_exact_a_per_dim": 0.9321672207645146,
        "continuation_non_top_only_fraction": 0.11301885339451084,
        "total_non_top_only_fraction": 0.07733408144475684,
    }
    gates = {
        "failed_tree_exact": True,
        "original_source_tree_exact": True,
        "startup_failure_tree_exact": True,
        "protocol_and_manifests_frozen": True,
        "failure_and_run_log_exact": True,
        "packet_hash_graph_exact": graph["artifact_count"] == 29,
        "execution_budget_exact": recovery["execution"]["geometry_evaluation_count"]
        == 1,
        "checkpoint_lineage_exact": geometry["bitwise_active_tensor_lineage"],
        "task_and_input_provenance_exact": True,
        "model_nonmutation_exact": recovery["nonmutation"]["changed_parameter_count"]
        == ACTIVE_TENSOR_COUNT,
        "geometry_shapes_dtypes_finite": True,
        "matrix_from_hessian_closure": numeric["matrix_from_hessian_max_abs_error"]
        <= ABSOLUTE_TOLERANCE,
        "eigensystem_closure": numeric["eigvalsh_matrix_max_abs_error"]
        <= ABSOLUTE_TOLERANCE,
        "low_basis_projector_closure": max(
            numeric["low_basis_orthogonality_max_abs"],
            numeric["low_basis_eigen_residual_relative"],
            numeric["low_projector_basis_max_abs_error"],
        )
        <= ABSOLUTE_TOLERANCE,
        "metric_closure": numeric["metric_closure_max_abs_error"] <= ABSOLUTE_TOLERANCE,
        "spectrum_bitwise_closure": numeric["spectrum"][
            "all_binary64_representations_exact"
        ],
        "csv_checkpoint_closure": all(
            value["binary64_exact"] for value in checkpoint["csv"].values()
        ),
        "scientific_outcome_unchanged": science["failed_scientific_gates"]
        == FAILED_SCIENTIFIC_GATES,
        "cuda_uninitialized": accelerator_uninitialized(),
        "scientific_execution_absent": True,
    }
    require(all(gates.values()), "failed-packet audit gate failed")
    return {
        "gates": gates,
        "numeric_audit": numeric_audit,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "failure": failure,
        "graph": graph,
        "copied_source_sha256": copied,
        "config": config,
        "checkpoint": checkpoint,
        "geometry": geometry,
        "science": science,
        "recovery": recovery,
        "elapsed_sec": time.perf_counter() - started,
    }


def _immutable_regular_file(path: Path, *, root: Path = ROOT) -> Path:
    path = regular_file(path, root=root)
    metadata = os.lstat(path)
    require(metadata.st_nlink == 1, f"immutable file link count is not one: {path}")
    require(
        stat.S_IMODE(metadata.st_mode) & 0o222 == 0,
        f"immutable file is writable: {path}",
    )
    return path


def _publisher_tree_projection(
    audit: Mapping[str, Any], *, expected_path: Path
) -> dict[str, Any]:
    expected_path = absolute(expected_path)
    require(
        isinstance(audit, Mapping)
        and set(audit)
        == {
            "path",
            "file_count",
            "total_size_bytes",
            "tree_sha256",
            "files",
        },
        "independent tree audit schema mismatch",
    )
    require(audit["path"] == str(expected_path), "independent tree path mismatch")
    require(is_sha256(audit["tree_sha256"]), "independent tree hash is malformed")
    require(
        type(audit["file_count"]) is int and type(audit["total_size_bytes"]) is int,
        "independent tree count/size type mismatch",
    )
    files = audit["files"]
    require(isinstance(files, Mapping), "independent tree files are not a mapping")
    require(len(files) == audit["file_count"], "independent tree file count mismatch")
    projected: list[dict[str, Any]] = []
    for name in sorted(files, key=os.fsencode):
        item = files[name]
        require(
            isinstance(name, str)
            and Path(name).name == name
            and isinstance(item, Mapping)
            and set(item)
            == {
                "sha256",
                "size_bytes",
                "device",
                "inode",
                "link_count",
                "mode",
                "mtime_ns",
            },
            f"independent tree file audit malformed: {name!r}",
        )
        require(
            is_sha256(item["sha256"])
            and all(
                type(item[field]) is int
                for field in (
                    "size_bytes",
                    "device",
                    "inode",
                    "link_count",
                    "mode",
                    "mtime_ns",
                )
            ),
            f"independent tree metadata malformed: {name}",
        )
        projected.append(
            {
                "device": item["device"],
                "inode": item["inode"],
                "name": name,
                "nlink": item["link_count"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
        )
    require(
        sum(item["size_bytes"] for item in projected) == audit["total_size_bytes"],
        "independent tree total size does not close",
    )
    return {
        "entry_count": audit["file_count"],
        "files": projected,
        "path": str(expected_path),
        "regular_file_count": audit["file_count"],
        "total_size_bytes": audit["total_size_bytes"],
        "tree_sha256": audit["tree_sha256"],
    }


def _manifest_projection(
    documents: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    entries = documents["failed_entries"]
    require(
        isinstance(entries, list) and len(entries) == 34,
        "frozen failed entries are malformed",
    )
    staged = [item for item in entries if item["name"] != "failure.json"]
    published = [item for item in staged if item["name"] != "INCOMPLETE"]
    require(
        len(staged) == 33 and len(published) == 32,
        "frozen salvage projection order/count mismatch",
    )
    return staged, published


def _validate_historical_tree_audit(
    audit: Any,
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_path: Path,
    expected_tree_sha256: str,
    expected_total_size: int,
) -> dict[str, Mapping[str, Any]]:
    require(
        isinstance(audit, Mapping)
        and set(audit)
        == {
            "entry_count",
            "files",
            "path",
            "regular_file_count",
            "total_size_bytes",
            "tree_sha256",
        },
        "historical tree audit schema mismatch",
    )
    require(
        audit["path"] == str(expected_path)
        and audit["entry_count"] == audit["regular_file_count"] == len(entries)
        and audit["total_size_bytes"] == expected_total_size
        and audit["tree_sha256"] == expected_tree_sha256,
        "historical tree audit identity mismatch",
    )
    files = audit["files"]
    require(
        isinstance(files, list) and len(files) == len(entries),
        "historical tree file audit count mismatch",
    )
    expected = {item["name"]: item for item in entries}
    require(
        [item.get("name") for item in files] == sorted(expected, key=os.fsencode),
        "historical tree file order mismatch",
    )
    result: dict[str, Mapping[str, Any]] = {}
    for item in files:
        require(
            isinstance(item, Mapping)
            and set(item)
            == {"device", "inode", "name", "nlink", "sha256", "size_bytes"},
            "historical tree file schema mismatch",
        )
        name = item["name"]
        require(name in expected and name not in result, "historical filename mismatch")
        spec = expected[name]
        require(
            item["sha256"] == spec["sha256"]
            and item["size_bytes"] == spec["size_bytes"]
            and item["nlink"] == 1
            and all(type(item[field]) is int for field in ("device", "inode")),
            f"historical tree metadata mismatch: {name}",
        )
        result[name] = item
    return result


def _current_copy_evidence(
    source: Path,
    target: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for spec in entries:
        name = spec["name"]
        source_path = regular_file(source / name, root=root)
        target_path = regular_file(target / name, root=root)
        source_stat = os.lstat(source_path)
        target_stat = os.lstat(target_path)
        require(
            source_stat.st_size == target_stat.st_size == spec["size_bytes"]
            and sha256_file(source_path, root=root)
            == sha256_file(target_path, root=root)
            == spec["sha256"],
            f"current copy byte identity mismatch: {name}",
        )
        require(
            source_stat.st_nlink == target_stat.st_nlink == 1
            and (source_stat.st_dev, source_stat.st_ino)
            != (target_stat.st_dev, target_stat.st_ino),
            f"current source/target hardlink detected: {name}",
        )
        evidence.append(
            {
                "destination_device": target_stat.st_dev,
                "destination_inode": target_stat.st_ino,
                "destination_nlink": target_stat.st_nlink,
                "name": name,
                "sha256": spec["sha256"],
                "size_bytes": spec["size_bytes"],
                "source_device": source_stat.st_dev,
                "source_inode": source_stat.st_ino,
                "source_nlink": source_stat.st_nlink,
            }
        )
    return evidence


def validate_numeric_audit(
    payload: Any, *, independently_observed: Mapping[str, Any]
) -> dict[str, Any]:
    require(
        isinstance(payload, Mapping) and set(payload) == NUMERIC_AUDIT_KEYS,
        "execution numeric audit schema mismatch",
    )
    require(
        isinstance(independently_observed, Mapping)
        and set(independently_observed) == NUMERIC_AUDIT_KEYS
        and deep_exact(payload, independently_observed),
        "execution numeric audit differs from independent recomputation",
    )
    require(
        payload["absolute_tolerance"] == ABSOLUTE_TOLERANCE
        and payload["relative_tolerance"] == 0.0
        and payload["matrix_from_hessian_max_abs_error"] == 3.885780586188048e-16
        and payload["matrix_symmetry_max_abs_error"] == 0.0
        and payload["eigvalsh_matrix_max_abs_error"] == 6.661338147750939e-15
        and payload["low_basis_orthogonality_max_abs"] == 2.1163626406917047e-15
        and payload["low_basis_eigen_residual_relative"] == 1.4866823708986403e-15
        and payload["low_projector_basis_max_abs_error"] == 1.9984014443252818e-15
        and 0.0 <= payload["metric_closure_max_abs_error"] <= ABSOLUTE_TOLERANCE
        and 0.0 <= payload["source_metric_max_abs_error"] <= ABSOLUTE_TOLERANCE
        and payload["spectrum_max_abs_error"] == 0.0
        and payload["spectrum_bitwise_equal_count"] == DIMENSION
        and payload["spectrum_representation_count"] == 5
        and payload["rank_2_binary64_hex"] == "3de22a7f787e6c62"
        and payload["state_csv_rows"] == 101
        and payload["spectrum_csv_rows"] == 51_712
        and payload["scientific_success"] is False
        and payload["failed_scientific_success_gates"] == FAILED_SCIENTIFIC_GATES
        and payload["final_exact_a_per_dim"] == 0.9321672207645146
        and payload["continuation_non_top_only_fraction"] == 0.11301885339451084
        and payload["total_non_top_only_fraction"] == 0.07733408144475684,
        "execution numeric audit value mismatch",
    )
    return {"exact": True, "field_count": len(payload)}


def validate_execution_record_payload(
    record: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    freeze_sha256: str,
    documents: Mapping[str, Any],
    failed_audit: Mapping[str, Any],
    target_tree_audit: Mapping[str, Any],
    current_copy_evidence: Sequence[Mapping[str, Any]],
    publisher_snapshot_sha256: str,
    publisher_snapshot_matches_source: bool,
) -> dict[str, Any]:
    require(
        isinstance(record, Mapping) and set(record) == EXECUTION_RECORD_KEYS,
        "execution record schema mismatch",
    )
    validate_timestamp(record["created_at_utc"], field="execution created_at_utc")
    require(is_sha256(freeze_sha256), "execution freeze digest is malformed")
    expected_scalars = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "validated_ready_for_atomic_publish",
        "repository_root": str(ROOT),
        "protocol_sha256": PROTOCOL_SHA256,
        "failed_manifest_sha256": FAILED_MANIFEST_SHA256,
        "derivation_sha256": DERIVATION_SHA256,
        "execution_freeze_sha256": freeze_sha256,
        "publisher_source_sha256": freeze["publisher"]["raw_sha256"],
        "publisher_normalized_source_sha256": freeze["publisher"]["normalized_sha256"],
        "publisher_snapshot_sha256": freeze["publisher"]["raw_sha256"],
        "reviewer_source_sha256": freeze["reviewer"]["raw_sha256"],
        "publisher_test_sha256": freeze["publisher_test"]["sha256"],
        "reviewer_test_sha256": freeze["reviewer_test"]["sha256"],
    }
    for name, expected in expected_scalars.items():
        require(record[name] == expected, f"execution record mismatch: {name}")
    require(
        publisher_snapshot_sha256 == record["publisher_snapshot_sha256"]
        and publisher_snapshot_matches_source,
        "execution publisher snapshot mismatch",
    )

    staged_entries, published_entries = _manifest_projection(documents)
    protected = failed_audit["protected_after"]
    require(
        failed_audit["protected_before"] == protected,
        "failed audit did not preserve protected trees",
    )
    source_projection = _publisher_tree_projection(
        protected["failed"], expected_path=FAILED_SOURCE
    )
    original_projection = _publisher_tree_projection(
        protected["original_source"], expected_path=ORIGINAL_SOURCE
    )
    startup_projection = _publisher_tree_projection(
        protected["startup_failure"], expected_path=STARTUP_FAILURE
    )
    target_projection = _publisher_tree_projection(
        target_tree_audit, expected_path=TARGET
    )

    source = record["source"]
    require(
        isinstance(source, Mapping)
        and set(source)
        == {"path", "pre_copy_audit", "post_copy_audit", "prepublication_audit"}
        and source["path"] == str(FAILED_SOURCE)
        and source["pre_copy_audit"]
        == source["post_copy_audit"]
        == source["prepublication_audit"]
        == source_projection,
        "execution source audits mismatch",
    )

    staging = record["staging"]
    require(
        isinstance(staging, Mapping)
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
        and staging["path"] == str(STAGE)
        and staging["copied_entry_count"] == 33
        and staging["copied_total_size_bytes"] == 114_949_714
        and staging["copied_tree_sha256"] == STAGED_TREE_SHA256
        and staging["incomplete_removed"] is True
        and staging["ready_entry_count"] == 32
        and staging["ready_total_size_bytes"] == 114_949_656
        and staging["ready_tree_sha256"] == PUBLISHED_TREE_SHA256
        and staging["fsync_complete"] is True,
        "execution staging identity mismatch",
    )
    copied_files = _validate_historical_tree_audit(
        staging["copied_audit"],
        staged_entries,
        expected_path=STAGE,
        expected_tree_sha256=STAGED_TREE_SHA256,
        expected_total_size=114_949_714,
    )
    ready_files = _validate_historical_tree_audit(
        staging["ready_audit"],
        published_entries,
        expected_path=STAGE,
        expected_tree_sha256=PUBLISHED_TREE_SHA256,
        expected_total_size=114_949_656,
    )
    ready_as_target = dict(staging["ready_audit"])
    ready_as_target["path"] = str(TARGET)
    require(
        ready_as_target == target_projection,
        "historical ready-stage audit differs from canonical target",
    )
    for name, item in ready_files.items():
        require(
            item == copied_files[name],
            f"historical staged inode changed before publication: {name}",
        )

    require(
        record["publication"]
        == {
            "target": str(TARGET),
            "target_absent_pre_copy": True,
            "target_absent_prepublication": True,
            "method": "renameat2(RENAME_NOREPLACE)",
            "rename_flags": 1,
            "expected_entry_count": 32,
            "expected_total_size_bytes": 114_949_656,
            "expected_tree_sha256": PUBLISHED_TREE_SHA256,
        },
        "execution publication record mismatch",
    )

    copy_audit = record["copy_audit"]
    require(
        isinstance(copy_audit, Mapping)
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
        and copy_audit["copied_names"] == [item["name"] for item in staged_entries]
        and copy_audit["excluded_names"] == ["failure.json"]
        and copy_audit["removed_after_validation"] == ["INCOMPLETE"]
        and copy_audit["distinct_inode_count"] == 33
        and copy_audit["hardlink_pair_count"] == 0
        and copy_audit["source_destination_sha256_equal"] is True
        and copy_audit["source_destination_size_equal"] is True,
        "execution copy audit identity mismatch",
    )
    per_file = copy_audit["per_file"]
    require(
        isinstance(per_file, list)
        and len(per_file) == 33
        and [item.get("name") for item in per_file]
        == [item["name"] for item in staged_entries],
        "execution per-file copy audit order/count mismatch",
    )
    evidence_by_name: dict[str, Mapping[str, Any]] = {}
    for item in per_file:
        require(
            isinstance(item, Mapping)
            and set(item)
            == {
                "destination_device",
                "destination_inode",
                "destination_nlink",
                "name",
                "sha256",
                "size_bytes",
                "source_device",
                "source_inode",
                "source_nlink",
            },
            "execution per-file copy schema mismatch",
        )
        name = item["name"]
        require(name not in evidence_by_name, "duplicate execution copy evidence")
        require(
            all(
                type(item[field]) is int
                for field in (
                    "destination_device",
                    "destination_inode",
                    "destination_nlink",
                    "size_bytes",
                    "source_device",
                    "source_inode",
                    "source_nlink",
                )
            )
            and item["source_nlink"] == item["destination_nlink"] == 1
            and (item["source_device"], item["source_inode"])
            != (item["destination_device"], item["destination_inode"]),
            f"execution copy inode evidence mismatch: {name}",
        )
        evidence_by_name[name] = item
    require(
        len(
            {
                (item["destination_device"], item["destination_inode"])
                for item in per_file
            }
        )
        == 33,
        "execution destination inode set is not distinct",
    )
    require(
        isinstance(current_copy_evidence, Sequence)
        and len(current_copy_evidence) == 32
        and all(isinstance(item, Mapping) for item in current_copy_evidence),
        "current copy evidence count mismatch",
    )
    current_names = [item.get("name") for item in current_copy_evidence]
    require(
        len(set(current_names)) == 32
        and set(current_names) == {item["name"] for item in published_entries},
        "current copy evidence filename set mismatch",
    )
    for current in current_copy_evidence:
        require(
            current == evidence_by_name.get(current["name"]),
            f"execution copy evidence differs from current target: {current['name']}",
        )
    source_files = {item["name"]: item for item in source_projection["files"]}
    incomplete = evidence_by_name["INCOMPLETE"]
    incomplete_spec = next(
        item for item in staged_entries if item["name"] == "INCOMPLETE"
    )
    require(
        incomplete["sha256"] == incomplete_spec["sha256"]
        and incomplete["size_bytes"] == incomplete_spec["size_bytes"]
        and incomplete["source_device"] == source_files["INCOMPLETE"]["device"]
        and incomplete["source_inode"] == source_files["INCOMPLETE"]["inode"]
        and incomplete["destination_device"] == copied_files["INCOMPLETE"]["device"]
        and incomplete["destination_inode"] == copied_files["INCOMPLETE"]["inode"],
        "removed INCOMPLETE copy evidence mismatch",
    )

    frozen_documents = {
        "derivation": {"path": str(DERIVATION), "sha256": DERIVATION_SHA256},
        "failed_manifest": {
            "path": str(FAILED_MANIFEST),
            "sha256": FAILED_MANIFEST_SHA256,
        },
        "protocol": {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA256},
    }
    frozen_sources = {
        name: dict(freeze[name])
        for name in ("publisher", "reviewer", "publisher_test", "reviewer_test")
    }
    lineage = record["protected_lineage"]
    require(
        isinstance(lineage, Mapping)
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
        and lineage["failed_manifest_path"] == str(FAILED_MANIFEST)
        and lineage["original_source_pre_copy"]
        == lineage["original_source_prepublication"]
        == original_projection
        and lineage["startup_failure_pre_copy"]
        == lineage["startup_failure_prepublication"]
        == startup_projection
        and lineage["frozen_documents"] == frozen_documents
        and lineage["frozen_sources"] == frozen_sources
        and lineage["execution_freeze_path"] == str(EXECUTION_FREEZE)
        and lineage["publisher_snapshot_path"] == str(PUBLISHER_SNAPSHOT),
        "execution protected lineage mismatch",
    )
    require(
        record["scientific_execution"] == SCIENTIFIC_EXECUTION_ZERO,
        "execution scientific counters are not exact zero",
    )
    validate_numeric_audit(
        record["numeric_audit"],
        independently_observed=failed_audit["numeric_audit"],
    )
    gates = record["acceptance_gates"]
    require(
        isinstance(gates, Mapping)
        and set(gates) == PUBLISHER_GATES
        and len(gates) == 29
        and all(value is True for value in gates.values()),
        "execution publisher gate map changed or failed",
    )
    return {
        "valid": True,
        "gate_count": len(gates),
        "copy_file_count": len(per_file),
        "scientific_execution_absent": True,
    }


def load_and_validate_execution_record(
    *,
    freeze: Mapping[str, Any],
    freeze_sha256: str,
    documents: Mapping[str, Any],
    failed_audit: Mapping[str, Any],
    target_tree_audit: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    record_path = _immutable_regular_file(EXECUTION_RECORD, root=root)
    raw = read_bytes(record_path, root=root)
    record = load_json(record_path, root=root)
    require(raw == json_bytes(record), "execution record serialization mismatch")
    snapshot = _immutable_regular_file(PUBLISHER_SNAPSHOT, root=root)
    snapshot_sha256 = sha256_file(snapshot, root=root)
    publisher_snapshot_matches_source = read_bytes(snapshot, root=root) == read_bytes(
        PUBLISHER_SOURCE, root=root
    )
    _, published_entries = _manifest_projection(documents)
    current_evidence = _current_copy_evidence(
        FAILED_SOURCE, TARGET, published_entries, root=root
    )
    validation = validate_execution_record_payload(
        record,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        documents=documents,
        failed_audit=failed_audit,
        target_tree_audit=target_tree_audit,
        current_copy_evidence=current_evidence,
        publisher_snapshot_sha256=snapshot_sha256,
        publisher_snapshot_matches_source=publisher_snapshot_matches_source,
    )
    require(
        sha256_file(record_path, root=root) == hashlib.sha256(raw).hexdigest(),
        "execution record changed during validation",
    )
    return {
        "record": record,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "validation": validation,
    }


def validate_review_record(record: Mapping[str, Any]) -> dict[str, Any]:
    require(
        set(record) == REVIEW_KEYS and len(record) == 33,
        "review record schema mismatch",
    )
    require(record["schema_version"] == 1, "review record version mismatch")
    require(
        record["review_protocol_id"] == REVIEW_PROTOCOL_ID, "review protocol mismatch"
    )
    require(record["protocol_id"] == PROTOCOL_ID, "review salvage protocol mismatch")
    require(
        record["status"] == "valid_independent_salvage_review",
        "review status mismatch",
    )
    validate_timestamp(record["created_at_utc"], field="review created_at_utc")
    require(record["repository_root"] == str(ROOT), "review repository root mismatch")
    require(
        record["protocol_sha256"] == PROTOCOL_SHA256, "review protocol hash mismatch"
    )
    require(
        record["failed_manifest_sha256"] == FAILED_MANIFEST_SHA256,
        "review failed-manifest hash mismatch",
    )
    require(
        record["derivation_sha256"] == DERIVATION_SHA256,
        "review derivation hash mismatch",
    )
    hash_fields = {
        "execution_freeze_sha256",
        "publisher_source_sha256",
        "publisher_normalized_source_sha256",
        "publisher_snapshot_sha256",
        "reviewer_source_sha256",
        "reviewer_normalized_source_sha256",
        "publisher_test_sha256",
        "reviewer_test_sha256",
        "execution_record_sha256",
        "failed_tree_sha256",
        "published_tree_sha256",
    }
    require(
        all(is_sha256(record[name]) for name in hash_fields),
        "review record contains malformed hash",
    )
    require(
        record["failed_tree_sha256"] == FAILED_TREE_SHA256,
        "review failed-tree hash mismatch",
    )
    require(
        record["published_tree_sha256"] == PUBLISHED_TREE_SHA256,
        "review published-tree hash mismatch",
    )
    require(record["reviewed_output"] == str(TARGET), "review output path mismatch")
    require(
        record["source_staging"] == str(FAILED_SOURCE), "review source path mismatch"
    )
    require(
        record["valid"] is True
        and record["recovery_valid"] is True
        and record["scientific_success"] is False
        and record["read_only"] is True,
        "review verdict mismatch",
    )
    require(
        record["failed_scientific_success_gates"] == FAILED_SCIENTIFIC_GATES,
        "review scientific failed-gate set mismatch",
    )
    gates = record["acceptance_gates"]
    require(
        isinstance(gates, Mapping)
        and set(gates) == REVIEW_GATES
        and len(gates) == 24
        and all(value is True for value in gates.values()),
        "review acceptance gates changed or failed",
    )
    for name in ("numeric_audit", "byte_copy_audit", "protected_tree_audits"):
        require(
            isinstance(record[name], Mapping) and record[name], f"review {name} missing"
        )
    require(record["errors"] == [], "review errors are not empty")
    require(
        isinstance(record["limitations"], list)
        and record["limitations"]
        and all(isinstance(value, str) and value for value in record["limitations"]),
        "review limitations malformed",
    )
    return {"valid": True, "gate_count": len(gates)}


def _fsync_directory(path: Path, *, root: Path) -> None:
    path = directory(path, root=root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    require(function is not None, "renameat2 is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    at_current_directory = -100
    no_replace = 1
    result = function(
        at_current_directory,
        os.fsencode(source),
        at_current_directory,
        os.fsencode(target),
        no_replace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(target))


def _exclusive_record_write(path: Path, payload: bytes, *, root: Path) -> None:
    path = check_path(path, root=root, allow_missing_leaf=True)
    require(not os.path.lexists(path), f"record path already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            require(count > 0, "short record write")
            written += count
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent, root=root)


def _load_and_validate_review(path: Path, *, root: Path) -> dict[str, Any]:
    _immutable_regular_file(path, root=root)
    record = load_json(path, root=root)
    validate_review_record(record)
    require(
        read_bytes(path, root=root) == json_bytes(record),
        "review record serialization mismatch",
    )
    return record


def publish_review_record_atomic(
    pending: Path,
    final: Path,
    record: Mapping[str, Any],
    *,
    filesystem_root: Path,
    rename_operation: Any | None = None,
) -> str:
    filesystem_root = absolute(filesystem_root)
    pending = check_path(pending, root=filesystem_root, allow_missing_leaf=True)
    final = check_path(final, root=filesystem_root, allow_missing_leaf=True)
    require(pending.parent == final.parent, "record paths do not share a directory")
    pending_exists = os.path.lexists(pending)
    final_exists = os.path.lexists(final)
    require(not (pending_exists and final_exists), "both review record paths exist")
    if final_exists:
        _load_and_validate_review(final, root=filesystem_root)
        return "existing_valid"
    if pending_exists:
        _load_and_validate_review(pending, root=filesystem_root)
    else:
        validate_review_record(record)
        _exclusive_record_write(pending, json_bytes(record), root=filesystem_root)
    rename = rename_operation or _rename_noreplace
    rename(pending, final)
    _fsync_directory(final.parent, root=filesystem_root)
    _load_and_validate_review(final, root=filesystem_root)
    require(not os.path.lexists(pending), "pending review record remains after rename")
    return "published"


def _log(stage: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    print(f"[salvage-reviewer] stage={stage}{suffix}", flush=True)


def _audit_protected_and_target(
    documents: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    _, published_entries = _manifest_projection(documents)
    return {
        "failed": audit_tree_against_entries(
            FAILED_SOURCE,
            documents["failed_entries"],
            expected_tree_sha256=FAILED_TREE_SHA256,
            expected_total_size=114_951_202,
            root=root,
        ),
        "original_source": audit_tree(ORIGINAL_SOURCE, root=root),
        "startup_failure": validate_startup_failure(documents["manifest"], root=root),
        "published": audit_tree_against_entries(
            TARGET,
            published_entries,
            expected_tree_sha256=PUBLISHED_TREE_SHA256,
            expected_total_size=114_949_656,
            root=root,
        ),
    }


def collect_production_review_evidence(
    *, root: Path = ROOT, verify_large_inputs: bool = True
) -> dict[str, Any]:
    started = time.perf_counter()
    root = absolute(root)
    require(root == ROOT, "production review root differs from frozen root")
    require(accelerator_uninitialized(), "accelerator initialized before review")
    check_path(STAGE, root=root, allow_missing_leaf=True)
    require(not os.path.lexists(STAGE), "salvage stage remains beside target")
    directory(TARGET, root=root)
    _immutable_regular_file(EXECUTION_RECORD, root=root)
    _immutable_regular_file(PUBLISHER_SNAPSHOT, root=root)

    _log("execution-freeze", f"path={EXECUTION_FREEZE}")
    freeze_file = regular_file(EXECUTION_FREEZE, root=root)
    require(os.lstat(freeze_file).st_nlink == 1, "execution freeze has extra links")
    freeze = load_json(freeze_file, root=root)
    freeze_validation = validate_execution_freeze(freeze, root=root)
    freeze_sha256 = sha256_file(freeze_file, root=root)
    require(
        freeze_validation["sha256"] == freeze_sha256,
        "execution freeze digest changed during validation",
    )
    require(
        normalized_source_sha256(REVIEWER_SOURCE, root=root)
        == EXPECTED_NORMALIZED_SOURCE_SHA256,
        "reviewer normalized source differs from embedded freeze",
    )

    _log("static-policy", "roles=publisher,reviewer")
    policy = freeze["import_policy"]
    publisher_static = audit_static_source(
        PUBLISHER_SOURCE,
        policy,
        expected_raw=freeze["publisher"]["raw_sha256"],
        expected_normalized=freeze["publisher"]["normalized_sha256"],
        root=root,
        reviewer=True,
    )
    reviewer_static = audit_static_source(
        REVIEWER_SOURCE,
        policy,
        expected_raw=freeze["reviewer"]["raw_sha256"],
        expected_normalized=freeze["reviewer"]["normalized_sha256"],
        root=root,
        reviewer=True,
    )

    _log("failed-packet-audit", f"source={FAILED_SOURCE}")
    documents = validate_frozen_documents(root=root)
    failed_audit = audit_failed_packet(
        root=root, verify_large_inputs=verify_large_inputs
    )
    protected_before = {
        **failed_audit["protected_after"],
        "published": audit_tree_against_entries(
            TARGET,
            _manifest_projection(documents)[1],
            expected_tree_sha256=PUBLISHED_TREE_SHA256,
            expected_total_size=114_949_656,
            root=root,
        ),
    }

    _log("publication-byte-audit", f"target={TARGET}")
    publication = validate_publication_bytes(
        FAILED_SOURCE, TARGET, documents["published_names"], root=root
    )
    target_graph = validate_packet_hash_graph(
        TARGET, documents["published_names"], root=root
    )
    target_tree_audit = audit_tree(TARGET, root=root)

    _log("execution-record-audit", f"record={EXECUTION_RECORD}")
    execution = load_and_validate_execution_record(
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        documents=documents,
        failed_audit=failed_audit,
        target_tree_audit=target_tree_audit,
        root=root,
    )
    protected_after = _audit_protected_and_target(documents, root=root)
    require(
        protected_before == protected_after,
        "protected or published tree changed during independent review",
    )
    require(
        accelerator_uninitialized(),
        "accelerator initialized during independent review",
    )
    return {
        "freeze": freeze,
        "freeze_sha256": freeze_sha256,
        "freeze_validation": freeze_validation,
        "documents": documents,
        "publisher_static": publisher_static,
        "reviewer_static": reviewer_static,
        "failed_audit": failed_audit,
        "publication": publication,
        "target_graph": target_graph,
        "target_tree_audit": target_tree_audit,
        "execution": execution,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "elapsed_sec": time.perf_counter() - started,
    }


def _review_gate_map(evidence: Mapping[str, Any]) -> dict[str, bool]:
    failed_gates = evidence["failed_audit"]["gates"]
    publication = evidence["publication"]
    execution = evidence["execution"]
    values = {
        "reviewer_source_frozen": evidence["reviewer_static"]["policy_pass"]
        and evidence["reviewer_static"]["normalized_sha256"]
        == EXPECTED_NORMALIZED_SOURCE_SHA256,
        "protocol_and_manifests_frozen": failed_gates["protocol_and_manifests_frozen"],
        "execution_freeze_exact": evidence["freeze_validation"]["exact"],
        "execution_record_exact": execution["validation"]["valid"],
        "failed_tree_exact": failed_gates["failed_tree_exact"],
        "original_source_tree_exact": failed_gates["original_source_tree_exact"],
        "startup_failure_tree_exact": failed_gates["startup_failure_tree_exact"],
        "published_tree_exact": publication["published_tree_sha256"]
        == PUBLISHED_TREE_SHA256,
        "byte_identity_exact": publication["all_bytes_identical"],
        "hardlinks_absent": publication["all_inodes_distinct"]
        and publication["all_link_counts_one"],
        "artifact_hash_graph_exact": failed_gates["packet_hash_graph_exact"]
        and evidence["target_graph"]["artifact_count"] == 29,
        "checkpoint_lineage_exact": failed_gates["checkpoint_lineage_exact"],
        "task_and_input_provenance_exact": failed_gates[
            "task_and_input_provenance_exact"
        ],
        "model_nonmutation_exact": failed_gates["model_nonmutation_exact"],
        "geometry_closure": all(
            failed_gates[name]
            for name in (
                "geometry_shapes_dtypes_finite",
                "matrix_from_hessian_closure",
                "eigensystem_closure",
                "low_basis_projector_closure",
            )
        ),
        "metric_closure": failed_gates["metric_closure"],
        "spectrum_bitwise_closure": failed_gates["spectrum_bitwise_closure"],
        "csv_checkpoint_closure": failed_gates["csv_checkpoint_closure"],
        "scientific_outcome_unchanged": failed_gates["scientific_outcome_unchanged"],
        "publisher_static_policy": evidence["publisher_static"]["policy_pass"],
        "reviewer_static_policy": evidence["reviewer_static"]["policy_pass"],
        "protected_trees_read_only": evidence["protected_before"]
        == evidence["protected_after"],
        "cuda_uninitialized": accelerator_uninitialized(),
        "scientific_execution_absent": execution["record"]["scientific_execution"]
        == SCIENTIFIC_EXECUTION_ZERO
        and execution["validation"]["scientific_execution_absent"],
    }
    require(
        set(values) == REVIEW_GATES
        and len(values) == 24
        and all(value is True for value in values.values()),
        "independent review gate map changed or failed",
    )
    return {name: values[name] for name in sorted(REVIEW_GATES)}


def _protected_tree_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    before = evidence["protected_before"]
    after = evidence["protected_after"]
    require(before == after, "protected trees changed before record construction")
    return {
        "failed_tree_sha256": before["failed"]["tree_sha256"],
        "original_source_tree_sha256": before["original_source"]["tree_sha256"],
        "startup_failure_tree_sha256": before["startup_failure"]["tree_sha256"],
        "published_tree_sha256": before["published"]["tree_sha256"],
        "source_and_publication_unchanged": True,
    }


def build_review_record(
    evidence: Mapping[str, Any], *, created_at_utc: str
) -> dict[str, Any]:
    validate_timestamp(created_at_utc, field="review created_at_utc")
    freeze = evidence["freeze"]
    record = {
        "schema_version": 1,
        "review_protocol_id": REVIEW_PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "valid_independent_salvage_review",
        "created_at_utc": created_at_utc,
        "repository_root": str(ROOT),
        "protocol_sha256": PROTOCOL_SHA256,
        "failed_manifest_sha256": FAILED_MANIFEST_SHA256,
        "derivation_sha256": DERIVATION_SHA256,
        "execution_freeze_sha256": evidence["freeze_sha256"],
        "publisher_source_sha256": freeze["publisher"]["raw_sha256"],
        "publisher_normalized_source_sha256": freeze["publisher"]["normalized_sha256"],
        "publisher_snapshot_sha256": evidence["execution"]["record"][
            "publisher_snapshot_sha256"
        ],
        "reviewer_source_sha256": freeze["reviewer"]["raw_sha256"],
        "reviewer_normalized_source_sha256": freeze["reviewer"]["normalized_sha256"],
        "publisher_test_sha256": freeze["publisher_test"]["sha256"],
        "reviewer_test_sha256": freeze["reviewer_test"]["sha256"],
        "execution_record_sha256": evidence["execution"]["sha256"],
        "failed_tree_sha256": FAILED_TREE_SHA256,
        "published_tree_sha256": PUBLISHED_TREE_SHA256,
        "reviewed_output": str(TARGET),
        "source_staging": str(FAILED_SOURCE),
        "valid": True,
        "recovery_valid": True,
        "scientific_success": False,
        "failed_scientific_success_gates": FAILED_SCIENTIFIC_GATES,
        "read_only": True,
        "acceptance_gates": _review_gate_map(evidence),
        "numeric_audit": dict(evidence["failed_audit"]["numeric_audit"]),
        "byte_copy_audit": dict(evidence["publication"]),
        "protected_tree_audits": _protected_tree_summary(evidence),
        "errors": [],
        "limitations": [
            "No new scientific replay, model construction, task loading, "
            "automatic differentiation, or optimization was performed.",
            "The review validates byte-identical publication of the existing "
            "one-evaluation replay; it is not a replicate.",
            "The preregistered scientific success criterion remains false.",
        ],
    }
    validate_review_record(record)
    return record


def _records_match_expected(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    validate_review_record(observed)
    validate_review_record(expected)
    require(
        deep_exact(observed, expected),
        "existing review record differs from current independent evidence",
    )


def revalidate_production_evidence_before_write(
    evidence: Mapping[str, Any], *, root: Path = ROOT
) -> None:
    root = absolute(root)
    require(root == ROOT, "prewrite root differs from frozen root")
    require(not os.path.lexists(STAGE), "salvage stage appeared before review write")
    freeze = load_json(EXECUTION_FREEZE, root=root)
    freeze_validation = validate_execution_freeze(freeze, root=root)
    require(
        freeze_validation["sha256"] == evidence["freeze_sha256"]
        and deep_exact(freeze, evidence["freeze"]),
        "execution freeze changed before review write",
    )
    require(
        normalized_source_sha256(REVIEWER_SOURCE, root=root)
        == EXPECTED_NORMALIZED_SOURCE_SHA256
        == freeze["reviewer"]["normalized_sha256"]
        and sha256_file(REVIEWER_SOURCE, root=root) == freeze["reviewer"]["raw_sha256"],
        "reviewer source changed before review write",
    )
    policy = freeze["import_policy"]
    audit_static_source(
        PUBLISHER_SOURCE,
        policy,
        expected_raw=freeze["publisher"]["raw_sha256"],
        expected_normalized=freeze["publisher"]["normalized_sha256"],
        root=root,
        reviewer=True,
    )
    audit_static_source(
        REVIEWER_SOURCE,
        policy,
        expected_raw=freeze["reviewer"]["raw_sha256"],
        expected_normalized=freeze["reviewer"]["normalized_sha256"],
        root=root,
        reviewer=True,
    )
    require(
        sha256_file(EXECUTION_RECORD, root=root) == evidence["execution"]["sha256"],
        "execution record changed before review write",
    )
    protected = _audit_protected_and_target(evidence["documents"], root=root)
    require(
        protected == evidence["protected_before"],
        "protected or published tree changed before review write",
    )
    require(accelerator_uninitialized(), "accelerator initialized before review write")


def review_salvage(
    *,
    evidence_collector: Any = collect_production_review_evidence,
    prewrite_validator: Any = revalidate_production_evidence_before_write,
    pending: Path = REVIEW_RECORD_TEMP,
    final: Path = REVIEW_RECORD,
    filesystem_root: Path = ROOT,
    record_publisher: Any = publish_review_record_atomic,
) -> dict[str, Any]:
    started = time.perf_counter()
    filesystem_root = absolute(filesystem_root)
    pending = check_path(pending, root=filesystem_root, allow_missing_leaf=True)
    final = check_path(final, root=filesystem_root, allow_missing_leaf=True)
    require(pending.parent == final.parent, "review record parents differ")
    pending_exists = os.path.lexists(pending)
    final_exists = os.path.lexists(final)
    require(
        not (pending_exists and final_exists),
        "manual quarantine required: both review record paths exist",
    )
    state = (
        "existing_final"
        if final_exists
        else "pending_temp"
        if pending_exists
        else "fresh"
    )
    _log(
        "startup",
        f"protocol={REVIEW_PROTOCOL_ID} state={state} device=cpu "
        "dtype=torch.float64 seed=none cache_mode=read-only-byte-review",
    )
    _log(
        "config",
        f"source={FAILED_SOURCE} target={TARGET} execution_record={EXECUTION_RECORD} "
        f"review_temp={pending} review_final={final}",
    )
    existing: dict[str, Any] | None = None
    if final_exists:
        existing = _load_and_validate_review(final, root=filesystem_root)
    elif pending_exists:
        existing = _load_and_validate_review(pending, root=filesystem_root)

    evidence = evidence_collector()
    created_at = existing["created_at_utc"] if existing is not None else utc_timestamp()
    expected = build_review_record(evidence, created_at_utc=created_at)
    if existing is not None:
        _records_match_expected(existing, expected)

    _log("prewrite-revalidation", f"state={state}")
    prewrite_validator(evidence)
    outcome = record_publisher(
        pending,
        final,
        expected,
        filesystem_root=filesystem_root,
    )
    observed = _load_and_validate_review(final, root=filesystem_root)
    _records_match_expected(observed, expected)
    prewrite_validator(evidence)
    require(not os.path.lexists(pending), "review temp remains after completion")
    elapsed = time.perf_counter() - started
    _log(
        "complete",
        f"status={outcome} elapsed_seconds={elapsed:.3f} review={final} "
        f"target={TARGET} execution_record={EXECUTION_RECORD} "
        "valid=1 recovery_valid=1 scientific_success=0 gates=24/24",
    )
    return {
        "status": outcome,
        "review_record": str(final),
        "execution_record_sha256": evidence["execution"]["sha256"],
        "published_tree_sha256": PUBLISHED_TREE_SHA256,
        "gate_count": 24,
        "scientific_success": False,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    require(len(sys.argv) == 1, "production reviewer accepts no CLI arguments")
    require(
        absolute(Path(__file__)) == absolute(REVIEWER_SOURCE),
        "executed reviewer path differs from frozen path",
    )
    require(ROOT == Path("/home/coder/project"), "production root pin changed")
    review_salvage()


if __name__ == "__main__":
    main()
