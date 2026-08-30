from __future__ import annotations

import os
import sys

# Direct path execution otherwise lets scripts/inspect shadow the stdlib module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _SCRIPT_DIR:
    sys.path.pop(0)
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import struct
import traceback
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_a_low_common_i5_v1"
ITERATION3_PROTOCOL_ID = "one_state_exact_a_p32_unit_common_i3_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration5_exact_a_low_common_production"
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_5_exact_a_low_common/protocol.md"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_5_frozen_dependency_manifest.json"
PRODUCER_SOURCE = ROOT / "scripts/audit_one_state_exact_a_low_common.py"

ITERATION3 = OUTPUT_ROOT / "iteration3_p32_unit_common_production"
ITERATION3_CHECKPOINT = ITERATION3 / "final_checkpoint.pt"
ITERATION3_STATES = ITERATION3 / "state_metrics.csv"
ITERATION3_SPECTRA = ITERATION3 / "state_spectra.csv"
ITERATION3_DECISION = ITERATION3 / "decision.json"
ITERATION3_REVIEW = ITERATION3 / "independent_review.json"
ACTIVE_PARAMETERS = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
)
STATE_BANK = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_fixed_state_probe_variance_h2048/state_bank.csv"
)

DIMENSION = 512
PROJECTED_DIMENSION = 480
ACTIVE_PARAMETER_COUNT = 11_685_120
SOURCE_WEIGHT_INDEX = 378
STATE_POSITION = 2
EPSILON = 1e-4
LOW_THRESHOLD = 0.1
DEAD_THRESHOLD = 1e-4
EXPECTED_LOW_COUNT = 480
EXPECTED_DEAD_COUNT = 309
TARGET_NORM = 0.04892722657548397
OLD_BETA = 22.536727828943093
LOCAL_RADII = (1.0 / 1024.0, 1.0 / 512.0)
ALPHAS = (
    -1.0 / 128.0,
    -1.0 / 256.0,
    -1.0 / 512.0,
    -1.0 / 1024.0,
    0.0,
    1.0 / 1024.0,
    1.0 / 512.0,
    1.0 / 256.0,
    1.0 / 128.0,
    1.0 / 32.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
)
CANDIDATE_ALPHAS = (1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0)
CANCELLATION_NORM_MIN = 0.5
CHAIN_RELATIVE_ERROR_MAX = 0.05
CHAIN_CAUCHY_RELATIVE_ERROR_MAX = 0.01
EFFECTIVE_COSINE_MIN = 0.99
EFFECTIVE_NORM_RATIO_MIN = 0.95
EFFECTIVE_NORM_RATIO_MAX = 1.05
MIDPOINT_RELATIVE_MAX = 0.05
ORTHOGONALITY_MAX = 1e-10
EIGEN_RESIDUAL_MAX = 1e-10
EIGEN_ORDER_TOL = 1e-12
RETAINED_MEMORY_RANGE_MAX = 64 * 1024 * 1024
BLOCK_PEAK_RANGE_MAX = 128 * 1024 * 1024
METRIC_FLOORS = {
    "exact_a_per_dim": 1e-9,
    "damped_full_burg_per_dim": 1e-9,
    "frozen_low_energy": 1e-10,
    "m_max": 1e-8,
    "m_p50": 1e-10,
    "effective_rank": 1e-8,
    "projected_low_p50": 1e-12,
    "projected_low_participation_rank": 1e-8,
    "projected_low_entropy_rank": 1e-8,
}

EXPECTED_CHECKPOINT_FILE_SHA256 = (
    "ccafbad0e0fcf25aaabffe0a406558937d1b448ac06c80b7c40a6637209dc5ad"
)
EXPECTED_ACTIVE_SHA256 = (
    "31abec6439564c392a99b6c82dd1386aed714cb1b2a1ce1c4709088686a0caff"
)
EXPECTED_ACCEPTED_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_PRODUCER_SHA256 = (
    "493eb3e6a29e2bb7250ace71f71f3d2f4541a1482ee4f679c215439f23b5a289"
)
EXPECTED_NORMALIZED_PRODUCER_SHA256 = (
    "8355b545f387e44f69d060b917077d538e51cddd2b643dcdf1770912bfecdd2b"
)
EXPECTED_DEPENDENCY_MANIFEST_SHA256 = (
    "d088533055704dad8d3cd0ec791a709ecb8676ec3d1ad7f9aed8c3949db51cca"
)
EXPECTED_PROTOCOL_SHA256 = (
    "0d94159d451dfa967f7ad2eace954b8d0cf94cf37414d0f5ade15eabdd67d215"
)

EXPECTED_MANIFEST_ARTIFACTS = {
    "a_low_common_line.png",
    "a_low_common_spectra.png",
    "decision.json",
    "endpoint_metrics.csv",
    "executed_source_snapshot.py",
    "finite_candidates.csv",
    "frozen_dependency_manifest_snapshot.json",
    "global_spectra.csv",
    "gradient_diagnostics.json",
    "projected_low_spectra.csv",
    "protocol_snapshot.md",
    "realized_chain.csv",
    "resolved_config.json",
}
EXPECTED_PRODUCER_VALIDITY_GATES = {
    "accepted_checkpoint_matches",
    "active_parameter_count_matches",
    "base_metrics_replay",
    "base_parameter_hash_matches",
    "base_spectrum_replay",
    "dead_count_matches",
    "dependencies_unchanged",
    "direction_norm_matches",
    "eigen_order_passes",
    "eigen_residual_passes",
    "finite_line_valid",
    "gradient_valid",
    "iteration3_checkpoint_matches",
    "live_source_unchanged",
    "low_count_matches",
    "orthogonality_passes",
    "parameters_restored",
    "source_snapshot_matches",
}

# The review was performed against these exact immutable images.
MANUALLY_INSPECTED_PLOTS = {
    "a_low_common_line.png": {
        "sha256": "a2683a5b336c0fa25c6b104c62e9955e98890df5eb1368fa7fc0af4326e887b8",
        "dimensions": [3060, 1800],
        "assessment": (
            "Readable six-panel line plot and consistent with endpoint_metrics.csv. "
            "Only nonnegative alphas are plotted; large-alpha tails compress the small-radius "
            "changes in exact A and m_max, so candidate decisions must come from the table."
        ),
    },
    "a_low_common_spectra.png": {
        "sha256": "d6505144c8d4c99382fe42f4f8b285bb7eaaac9c33aeb0cdc9e9e504244b0a08",
        "dimensions": [2520, 900],
        "assessment": (
            "Readable log-scale global and frozen-low projected spectra at alpha 0, 1/8, "
            "and 1; ordering and endpoint selection agree with the stored spectrum tables."
        ),
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _close(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> bool:
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
    atol: float = 1e-10,
    rtol: float = 1e-9,
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
        if not (left_sequence and right_sequence and len(left) == len(right)):
            return False
        return all(
            _equivalent(l_value, r_value, atol=atol, rtol=rtol)
            for l_value, r_value in zip(left, right)
        )
    if isinstance(left, Real) and isinstance(right, Real):
        return _close(left, right, atol=atol, rtol=rtol)
    return left == right


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


def _normalized_runner_sha256(path: Path) -> str:
    masked = (
        "EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    normalized: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        prefix = next((candidate for candidate in masked if line.startswith(candidate)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} root must be an object")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{path.name} has an empty or duplicate header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path.name} has rows wider than its header")
    return fields, rows


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _bool_text(value: Any) -> bool:
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _png_info(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 10_000:
        raise ValueError(f"missing or implausibly small PNG: {path}")
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG signature: {path}")
    offset = 8
    chunks: list[bytes] = []
    width = height = None
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError(f"truncated PNG chunk: {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ValueError(f"PNG chunk exceeds file: {path}")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        observed_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise ValueError(f"PNG CRC mismatch in {chunk_type!r}: {path}")
        chunks.append(chunk_type)
        if chunk_type == b"IHDR":
            if length != 13 or width is not None:
                raise ValueError(f"invalid PNG IHDR: {path}")
            width, height = struct.unpack(">II", payload[:8])
        if chunk_type == b"IEND":
            if length != 0:
                raise ValueError(f"invalid PNG IEND: {path}")
            saw_iend = True
            offset = chunk_end
            break
        offset = chunk_end
    if offset != len(data) or not saw_iend or width is None or b"IDAT" not in chunks:
        raise ValueError(f"incomplete PNG structure: {path}")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "bytes": len(data),
        "chunk_count": len(chunks),
    }


def _global_spectrum_metrics(eigenvalues: np.ndarray) -> dict[str, float]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.shape != (DIMENSION,) or not np.isfinite(eigenvalues).all():
        raise ValueError("global spectrum must be finite and have 512 entries")
    contribution = np.square(eigenvalues - 1.0)
    contribution_total = max(float(contribution.sum()), 1e-30)
    trace = float(eigenvalues.sum())
    square_sum = max(float(np.square(eigenvalues).sum()), 1e-30)
    r_eigenvalues = (eigenvalues + EPSILON) / (1.0 + EPSILON)
    gradient_eigenvalues = (1.0 - np.reciprocal(r_eigenvalues)) / (
        float(DIMENSION) * (1.0 + EPSILON)
    )
    quantiles = np.quantile(
        eigenvalues,
        [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
        method="linear",
    )
    trace_r = float(r_eigenvalues.mean())
    neg_logdet = float(-np.log(r_eigenvalues).mean())
    exact_a = float(contribution.mean())
    full_b = trace_r + neg_logdet - 1.0
    sorted_gradient = np.sort(gradient_eigenvalues)
    return {
        "exact_a_per_dim": exact_a,
        "a_constant_term": 1.0,
        "a_linear_trace_term": float(-2.0 * eigenvalues.mean()),
        "a_quartic_term": float(np.square(eigenvalues).mean()),
        "trace_m_per_dim": float(eigenvalues.mean()),
        "damped_full_burg_per_dim": full_b,
        "burg_trace_r_term": trace_r,
        "burg_neg_logdet_r_term": neg_logdet,
        "logdet_r_per_dim": float(np.log(r_eigenvalues).mean()),
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
        "m_lt_1e_4_fraction": float(np.mean(eigenvalues < 1e-4)),
        "m_lt_0p01_fraction": float(np.mean(eigenvalues < 0.01)),
        "m_lt_0p1_fraction": float(np.mean(eigenvalues < 0.1)),
        "m_lt_0p5_fraction": float(np.mean(eigenvalues < 0.5)),
        "m_near_1_10pct_fraction": float(np.mean(np.abs(eigenvalues - 1.0) <= 0.1)),
        "m_gt_1_fraction": float(np.mean(eigenvalues > 1.0)),
        "m_gt_2_fraction": float(np.mean(eigenvalues > 2.0)),
        "a_from_m_lt_0p1_share": float(
            contribution[eigenvalues < 0.1].sum() / contribution_total
        ),
        "a_from_m_gt_1_share": float(
            contribution[eigenvalues > 1.0].sum() / contribution_total
        ),
        "burg_matrix_gradient_norm": float(np.linalg.norm(gradient_eigenvalues)),
        "burg_matrix_gradient_eig_min": float(gradient_eigenvalues.min()),
        "burg_matrix_gradient_eig_p50": float(
            sorted_gradient[(DIMENSION - 1) // 2]
        ),
        "burg_matrix_gradient_eig_max": float(gradient_eigenvalues.max()),
        "true_objective": exact_a + OLD_BETA * full_b,
        "a_direct_matrix": exact_a,
        "a_trace_closure": float(
            1.0 - 2.0 * eigenvalues.mean() + np.square(eigenvalues).mean()
        ),
        "a_direct_abs_error": 0.0,
        "a_trace_abs_error": 0.0,
        "m_raw_eig_min": float(eigenvalues.min()),
        "effective_rank": trace * trace / square_sum,
        "effective_rank_fraction": trace * trace / (square_sum * float(DIMENSION)),
        "li_gap_per_dim": float(np.square(np.sqrt(eigenvalues) - 1.0).mean()),
        "a_low90_abs_per_dim": float(
            contribution[: int(math.floor(0.9 * DIMENSION))].sum() / float(DIMENSION)
        ),
        "a_low_lt_0p1_abs_per_dim": float(
            contribution[eigenvalues < 0.1].sum() / float(DIMENSION)
        ),
        "a_high_gt_1_abs_per_dim": float(
            contribution[eigenvalues > 1.0].sum() / float(DIMENSION)
        ),
        "a_top1_share": float(contribution[-1] / contribution_total),
        "a_top10_share": float(contribution[-10:].sum() / contribution_total),
        "top1_trace_share": float(eigenvalues[-1] / max(trace, 1e-30)),
        "top10_trace_share": float(eigenvalues[-10:].sum() / max(trace, 1e-30)),
    }


def _projected_spectrum_metrics(eigenvalues: np.ndarray) -> dict[str, float]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.shape != (PROJECTED_DIMENSION,) or not np.isfinite(eigenvalues).all():
        raise ValueError("projected spectrum must be finite and have 480 entries")
    trace = float(eigenvalues.sum())
    squared = max(float(np.square(eigenvalues).sum()), 1e-30)
    probabilities = eigenvalues / max(trace, 1e-30)
    entropy = float(
        -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum()
    )
    quantiles = np.quantile(
        eigenvalues,
        [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        method="linear",
    )
    return {
        "projected_low_raw_min": float(eigenvalues.min()),
        "projected_low_mean": float(eigenvalues.mean()),
        "projected_low_max": float(eigenvalues.max()),
        "projected_low_p01": float(quantiles[0]),
        "projected_low_p10": float(quantiles[1]),
        "projected_low_p25": float(quantiles[2]),
        "projected_low_p50": float(quantiles[3]),
        "projected_low_p75": float(quantiles[4]),
        "projected_low_p90": float(quantiles[5]),
        "projected_low_p95": float(quantiles[6]),
        "projected_low_p99": float(quantiles[7]),
        "projected_low_participation_rank": trace * trace / squared,
        "projected_low_entropy_rank": math.exp(entropy),
    }


@dataclass(frozen=True)
class _StorageType:
    name: str
    numpy_dtype: str


@dataclass(frozen=True)
class _StorageRef:
    storage_type: _StorageType
    key: str
    location: str
    element_count: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


_STORAGE_TYPES = {
    "FloatStorage": "f4",
    "DoubleStorage": "f8",
    "HalfStorage": "f2",
    "LongStorage": "i8",
    "IntStorage": "i4",
    "ShortStorage": "i2",
    "CharStorage": "i1",
    "ByteStorage": "u1",
    "BoolStorage": "?",
}


def _rebuild_tensor(
    storage: _StorageRef,
    storage_offset: int,
    size: Sequence[int],
    stride: Sequence[int],
    *_unused: Any,
) -> _TensorRef:
    return _TensorRef(
        storage=storage,
        storage_offset=int(storage_offset),
        shape=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
    )


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return _rebuild_tensor
        if module == "torch" and name in _STORAGE_TYPES:
            return _StorageType(name=name, numpy_dtype=_STORAGE_TYPES[name])
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"unsupported checkpoint global: {module}.{name}")

    def persistent_load(self, persistent_id: Any) -> _StorageRef:
        if not (
            isinstance(persistent_id, tuple)
            and len(persistent_id) == 5
            and persistent_id[0] == "storage"
            and isinstance(persistent_id[1], _StorageType)
        ):
            raise pickle.UnpicklingError(f"unsupported persistent id: {persistent_id!r}")
        _, storage_type, key, location, element_count = persistent_id
        return _StorageRef(
            storage_type=storage_type,
            key=str(key),
            location=str(location),
            element_count=int(element_count),
        )


def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    expected = 1
    for size, actual_stride in zip(reversed(shape), reversed(stride)):
        if size > 1 and actual_stride != expected:
            return False
        expected *= max(size, 1)
    return True


def _tensor_bytes(
    archive: zipfile.ZipFile,
    prefix: str,
    tensor: _TensorRef,
    byteorder: str,
) -> bytes:
    raw = archive.read(f"{prefix}/data/{tensor.storage.key}")
    endian = "<" if byteorder == "little" else ">"
    dtype = np.dtype(endian + tensor.storage.storage_type.numpy_dtype)
    expected_storage_bytes = tensor.storage.element_count * dtype.itemsize
    if len(raw) != expected_storage_bytes:
        raise ValueError(
            f"storage {tensor.storage.key} has {len(raw)} bytes, expected {expected_storage_bytes}"
        )
    element_count = math.prod(tensor.shape)
    if _is_contiguous(tensor.shape, tensor.stride):
        start = tensor.storage_offset * dtype.itemsize
        stop = start + element_count * dtype.itemsize
        logical = raw[start:stop]
        if len(logical) != element_count * dtype.itemsize:
            raise ValueError(f"tensor exceeds storage {tensor.storage.key}")
        if byteorder == sys.byteorder or dtype.itemsize == 1:
            return logical
        return (
            np.frombuffer(logical, dtype=dtype)
            .astype(dtype.newbyteorder("="), copy=True)
            .tobytes()
        )
    view = np.ndarray(
        shape=tensor.shape,
        dtype=dtype,
        buffer=raw,
        offset=tensor.storage_offset * dtype.itemsize,
        strides=tuple(value * dtype.itemsize for value in tensor.stride),
    )
    return np.ascontiguousarray(view, dtype=dtype.newbyteorder("=")).tobytes()


def _checkpoint_metadata_and_hash(path: Path) -> tuple[dict[str, Any], str, int]:
    with zipfile.ZipFile(path) as archive:
        data_entries = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(data_entries) != 1:
            raise ValueError(f"checkpoint has {len(data_entries)} data.pkl entries")
        data_entry = data_entries[0]
        prefix = data_entry.rsplit("/", 1)[0]
        checkpoint = _CheckpointUnpickler(io.BytesIO(archive.read(data_entry))).load()
        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint root must be a dictionary")
        byteorder_entry = f"{prefix}/byteorder"
        byteorder = (
            archive.read(byteorder_entry).decode("ascii").strip()
            if byteorder_entry in archive.namelist()
            else "little"
        )
        if byteorder not in {"little", "big"}:
            raise ValueError(f"invalid checkpoint byteorder {byteorder!r}")
        active_names = checkpoint.get("active_names")
        state = checkpoint.get("active_model_state")
        if active_names is None and isinstance(state, Mapping):
            active_names = sorted(state)
        if not isinstance(active_names, list) or not isinstance(state, Mapping):
            raise TypeError("checkpoint lacks active_names or active_model_state")
        if any(not isinstance(name, str) for name in active_names):
            raise TypeError("checkpoint active_names must contain strings")
        if set(active_names) != set(state):
            raise ValueError("checkpoint active_names/state keys differ")
        digest = hashlib.sha256()
        parameter_count = 0
        for name in active_names:
            tensor = state[name]
            if not isinstance(tensor, _TensorRef):
                raise TypeError(f"checkpoint value {name} is not a supported dense tensor")
            parameter_count += math.prod(tensor.shape)
            digest.update(name.encode("utf-8"))
            digest.update(_tensor_bytes(archive, prefix, tensor, byteorder))
    metadata = {key: value for key, value in checkpoint.items() if key != "active_model_state"}
    return metadata, digest.hexdigest(), parameter_count


class Review:
    def __init__(self) -> None:
        self.gates: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.limitations: list[str] = []

    def gate(self, name: str, value: Any) -> bool:
        if name in self.gates:
            raise KeyError(f"duplicate gate {name}")
        result = bool(value)
        self.gates[name] = result
        return result

    def phase(self, name: str, function: Callable[[], None]) -> bool:
        try:
            function()
        except Exception as error:
            self.gates[f"{name}_completed"] = False
            self.errors.append(f"{name}: {type(error).__name__}: {error}")
            self.details[f"{name}_traceback"] = traceback.format_exc(limit=10)
            return False
        self.gates[f"{name}_completed"] = True
        return True


def _write_report(output: Path, review: Review, recomputed_outcome: str | None) -> bool:
    valid = bool(review.gates and all(review.gates.values()) and not review.errors)
    report = {
        "protocol_id": PROTOCOL_ID,
        "review_scope": "independent CPU-only post-run validation from stored artifacts",
        "valid": valid,
        "outcome": recomputed_outcome if valid else None,
        "recomputed_outcome": recomputed_outcome,
        "gates": review.gates,
        "failed_gates": sorted(name for name, passed in review.gates.items() if not passed),
        "recomputed": review.details,
        "warnings": review.warnings,
        "limitations": review.limitations,
        "errors": review.errors,
    }
    report_path = output / "independent_review.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[a-low-i5-review] report={report_path} valid={valid} outcome={report['outcome']}",
        flush=True,
    )
    if not valid:
        print(f"[a-low-i5-review] failed_gates={report['failed_gates']}", flush=True)
        for error in review.errors:
            print(f"[a-low-i5-review] error={error}", flush=True)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    staging = Path(str(output) + ".incomplete")
    if (
        not output.is_dir()
        or not (output / "FINALIZED.json").is_file()
        or (output / "INCOMPLETE").exists()
        or staging.exists()
    ):
        print(
            f"[a-low-i5-review] finalized production unavailable: output={output} "
            f"staging_exists={staging.exists()}",
            flush=True,
        )
        return 1
    print(f"[a-low-i5-review] start output={output} device=cpu", flush=True)

    review = Review()
    context: dict[str, Any] = {}

    def metadata_phase() -> None:
        finalized = _load_json(output / "FINALIZED.json")
        resolved = _load_json(output / "resolved_config.json")
        decision = _load_json(output / "decision.json")
        manifest = _load_json(output / "artifact_manifest.json")
        gradient = _load_json(output / "gradient_diagnostics.json")
        context.update(
            finalized=finalized,
            resolved=resolved,
            decision=decision,
            manifest=manifest,
            gradient=gradient,
        )

        review.gate(
            "protocol_ids_match",
            {
                finalized.get("protocol_id"),
                resolved.get("protocol_id"),
                decision.get("protocol_id"),
                manifest.get("protocol_id"),
            }
            == {PROTOCOL_ID},
        )
        review.gate(
            "finalization_status_complete",
            finalized.get("status") == "complete_awaiting_independent_review"
            and not (output / "INCOMPLETE").exists(),
        )
        review.gate(
            "decision_hash_matches_finalized",
            finalized.get("decision_sha256") == _sha256_file(output / "decision.json"),
        )
        review.gate(
            "manifest_hash_matches_finalized",
            finalized.get("artifact_manifest_sha256")
            == _sha256_file(output / "artifact_manifest.json"),
        )
        artifacts = manifest.get("artifacts")
        review.gate(
            "manifest_has_exact_artifact_set",
            isinstance(artifacts, dict)
            and set(artifacts) == EXPECTED_MANIFEST_ARTIFACTS
            and all(_is_sha256(value) for value in artifacts.values()),
        )
        artifact_results = {
            name: (
                isinstance(artifacts, dict)
                and name in artifacts
                and (output / name).is_file()
                and _sha256_file(output / name) == artifacts[name]
            )
            for name in EXPECTED_MANIFEST_ARTIFACTS
        }
        review.gate("all_manifest_artifact_hashes_match", all(artifact_results.values()))
        allowed_files = EXPECTED_MANIFEST_ARTIFACTS | {
            "artifact_manifest.json",
            "FINALIZED.json",
            "run.log",
            "independent_review.json",
        }
        actual_files = {path.name for path in output.iterdir() if path.is_file()}
        review.gate(
            "production_file_set_has_no_unaccounted_artifacts",
            actual_files.issubset(allowed_files)
            and (allowed_files - {"independent_review.json"}).issubset(actual_files),
        )

        source_snapshot = output / "executed_source_snapshot.py"
        source_hash = _sha256_file(source_snapshot)
        normalized_source_hash = _normalized_runner_sha256(source_snapshot)
        review.gate(
            "executed_source_hashes_match",
            source_hash == EXPECTED_PRODUCER_SHA256
            and source_hash == resolved.get("source_sha256")
            and source_hash == manifest.get("executed_source_sha256")
            and normalized_source_hash == EXPECTED_NORMALIZED_PRODUCER_SHA256
            and normalized_source_hash == resolved.get("normalized_source_sha256")
            and normalized_source_hash
            == manifest.get("executed_normalized_source_sha256"),
        )
        review.gate(
            "live_producer_matches_executed_snapshot",
            PRODUCER_SOURCE.is_file()
            and _sha256_file(PRODUCER_SOURCE) == source_hash
            and _normalized_runner_sha256(PRODUCER_SOURCE) == normalized_source_hash,
        )

        protocol_snapshot = output / "protocol_snapshot.md"
        protocol_hash = _sha256_file(protocol_snapshot)
        review.gate(
            "protocol_snapshot_and_current_hash_match",
            protocol_hash == EXPECTED_PROTOCOL_SHA256
            and protocol_hash == resolved.get("protocol_sha256")
            and PROTOCOL_PATH.is_file()
            and _sha256_file(PROTOCOL_PATH) == protocol_hash,
        )

        dependency_snapshot = output / "frozen_dependency_manifest_snapshot.json"
        dependency_hash = _sha256_file(dependency_snapshot)
        frozen_dependencies = _load_json(dependency_snapshot)
        dependency_results = {
            relative: (
                isinstance(relative, str)
                and _is_sha256(digest)
                and (ROOT / relative).is_file()
                and _sha256_file(ROOT / relative) == digest
            )
            for relative, digest in frozen_dependencies.items()
        }
        review.gate(
            "frozen_dependency_snapshot_hash_matches",
            dependency_hash == EXPECTED_DEPENDENCY_MANIFEST_SHA256
            and dependency_hash == resolved.get("dependency_manifest_sha256")
            and FROZEN_DEPENDENCY_MANIFEST.is_file()
            and _sha256_file(FROZEN_DEPENDENCY_MANIFEST) == dependency_hash,
        )
        review.gate(
            "all_frozen_dependencies_still_match",
            bool(dependency_results)
            and all(dependency_results.values())
            and _equivalent(resolved.get("dependency_matches"), dependency_results),
        )
        accepted_relative = (
            "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing/"
            "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0/"
            "vae_checkpoint.pt"
        )
        iteration3_relative = (
            "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
            "one_state_exact_a_h2048/iteration3_p32_unit_common_production/final_checkpoint.pt"
        )
        review.gate(
            "checkpoint_dependency_hashes_are_frozen",
            frozen_dependencies.get(accepted_relative) == EXPECTED_ACCEPTED_CHECKPOINT_SHA256
            and frozen_dependencies.get(iteration3_relative)
            == EXPECTED_CHECKPOINT_FILE_SHA256
            and dependency_results.get(accepted_relative) is True
            and dependency_results.get(iteration3_relative) is True,
        )
        context["frozen_dependencies"] = frozen_dependencies

        review.gate(
            "resolved_protocol_constants_match",
            resolved.get("device") == "cuda:0"
            and resolved.get("dtype")
            == "float32 model/HVP; FP64 gradient accumulation and diagnostics"
            and resolved.get("seed") == "none; exact full-basis pullback"
            and resolved.get("source_weight_index") == SOURCE_WEIGHT_INDEX
            and resolved.get("state_position") == STATE_POSITION
            and resolved.get("hessian_chunk_size") == 64
            and resolved.get("block_size") == 64
            and resolved.get("expected_low_count") == EXPECTED_LOW_COUNT
            and resolved.get("expected_dead_count") == EXPECTED_DEAD_COUNT
            and _close(resolved.get("epsilon"), EPSILON, atol=0.0, rtol=0.0)
            and _close(resolved.get("low_threshold"), LOW_THRESHOLD, atol=0.0, rtol=0.0)
            and _close(resolved.get("dead_threshold"), DEAD_THRESHOLD, atol=0.0, rtol=0.0)
            and _close(resolved.get("target_norm"), TARGET_NORM, atol=1e-15, rtol=0.0)
            and _equivalent(resolved.get("local_radii"), list(LOCAL_RADII), atol=1e-15, rtol=0.0)
            and _equivalent(resolved.get("alphas"), list(ALPHAS), atol=1e-15, rtol=0.0)
            and _close(
                resolved.get("cancellation_norm_min"),
                CANCELLATION_NORM_MIN,
                atol=0.0,
                rtol=0.0,
            )
            and resolved.get("retained_memory_range_max_bytes")
            == RETAINED_MEMORY_RANGE_MAX
            and resolved.get("block_peak_range_max_bytes") == BLOCK_PEAK_RANGE_MAX
            and _equivalent(resolved.get("metric_floors"), METRIC_FLOORS)
            and Path(str(resolved.get("output_dir"))).resolve() == DEFAULT_OUTPUT.resolve(),
        )
        producer_gates = decision.get("validity_gates")
        review.gate(
            "producer_declared_valid",
            finalized.get("valid") is True
            and decision.get("valid") is True
            and isinstance(producer_gates, dict)
            and set(producer_gates) == EXPECTED_PRODUCER_VALIDITY_GATES
            and all(value is True for value in producer_gates.values()),
        )
        review.gate(
            "producer_outcomes_are_consistent",
            finalized.get("outcome") == decision.get("outcome")
            and finalized.get("outcome") == "no_local_chain_claim",
        )
        review.gate(
            "gradient_file_matches_decision_payload",
            _equivalent(gradient, decision.get("gradient_diagnostics")),
        )

        run_log_path = output / "run.log"
        run_log = run_log_path.read_text(encoding="utf-8", errors="replace")
        context["run_log"] = run_log
        review.gate("run_log_exists_and_nonempty", run_log_path.stat().st_size > 1_000)
        review.gate(
            "run_log_contains_complete_stage_sequence",
            all(
                marker in run_log
                for marker in (
                    "[a-low-i5] stage=load-fresh-checkpoint",
                    "[a-low-i5] stage=dense-base-and-frozen-subspace",
                    "[a-low-i5] stage=exact-two-gradient-pullback",
                    "[a-low-i5] stage=frozen-exact-line",
                    "[a-low-i5] stage=realized-chain-audit",
                    "[a-low-i5] complete valid=True outcome=no_local_chain_claim",
                    "[a-low-i5] published output=",
                )
            )
            and "[a-low-i5] FAILED" not in run_log,
        )

        plot_details: dict[str, Any] = {}
        plots_match = True
        for name, inspected in MANUALLY_INSPECTED_PLOTS.items():
            info = _png_info(output / name)
            match = (
                info["sha256"] == inspected["sha256"]
                and [info["width"], info["height"]] == inspected["dimensions"]
            )
            plots_match &= match
            plot_details[name] = {
                **info,
                "matches_manually_inspected_file": match,
                "manual_assessment": inspected["assessment"],
            }
        review.gate("plots_match_manually_inspected_files", plots_match)
        review.details["plot_inspection"] = plot_details
        review.warnings.append(
            "a_low_common_line.png omits negative-alpha endpoint traces and visually compresses "
            "small-radius exact-A/m_max changes; endpoint_metrics.csv is the decision evidence."
        )
        review.details["hash_review"] = {
            "artifact_hashes_match": artifact_results,
            "frozen_dependency_matches": dependency_results,
            "executed_source_sha256": source_hash,
            "executed_normalized_source_sha256": normalized_source_hash,
            "protocol_sha256": protocol_hash,
            "dependency_manifest_sha256": dependency_hash,
            "run_log_sha256_unmanifested": _sha256_file(run_log_path),
        }

    if not review.phase("metadata", metadata_phase):
        return 0 if _write_report(output, review, None) else 1

    def tables_phase() -> None:
        endpoint_fields, endpoint_rows = _read_csv(output / "endpoint_metrics.csv")
        global_fields, global_rows = _read_csv(output / "global_spectra.csv")
        projected_fields, projected_rows = _read_csv(output / "projected_low_spectra.csv")
        chain_fields, chain_rows = _read_csv(output / "realized_chain.csv")
        candidate_fields, candidate_rows = _read_csv(output / "finite_candidates.csv")
        context.update(
            endpoint_fields=endpoint_fields,
            endpoint_rows=endpoint_rows,
            global_rows=global_rows,
            projected_rows=projected_rows,
            chain_fields=chain_fields,
            chain_rows=chain_rows,
            candidate_fields=candidate_fields,
            candidate_rows=candidate_rows,
        )

        global_metric_fields = set(
            _global_spectrum_metrics(np.linspace(1e-8, 2.0, DIMENSION))
        )
        projected_metric_fields = set(
            _projected_spectrum_metrics(
                np.linspace(1e-8, 0.09, PROJECTED_DIMENSION)
            )
        )
        primary_metric_fields = global_metric_fields | projected_metric_fields | {
            "task_loss",
            "hessian_sec",
            "hessian_symmetry_rel",
            "frozen_low_energy",
            "frozen_dead_energy",
            "count_lt_1e_4",
            "count_lt_1e_2",
            "count_lt_0p1",
        }
        repeat_metric_fields = primary_metric_fields - {"hessian_sec"}
        expected_repeat_fields = {
            f"repeat_{field}_abs_error" for field in repeat_metric_fields
        } | {
            "repeat_global_spectrum_max_abs_error",
            "repeat_projected_spectrum_max_abs_error",
        }
        expected_endpoint_fields = {
            "alpha",
            "endpoint_parameter_hash",
            "repeat_parameter_hash_unchanged",
            *primary_metric_fields,
            *expected_repeat_fields,
        }
        review.gate(
            "endpoint_schema_exact",
            set(endpoint_fields) == expected_endpoint_fields
            and len(endpoint_fields) == len(expected_endpoint_fields),
        )
        review.gate(
            "spectrum_schemas_exact",
            global_fields == ["alpha", "rank", "eigenvalue"]
            and projected_fields == ["alpha", "rank", "eigenvalue"],
        )
        expected_chain_fields = [
            "radius",
            "objective",
            "nominal",
            "s_theta",
            "s_h",
            "direct",
            "relative_n_to_theta",
            "relative_theta_to_h",
            "relative_h_to_direct",
            "relative_n_to_direct",
            "effective_direction_norm",
            "desired_direction_norm",
            "effective_direction_norm_ratio",
            "effective_direction_cosine",
            "midpoint_drift_norm",
            "midpoint_drift_relative",
            "s_theta_a",
            "s_theta_low",
            "expected_signs",
            "transitions_pass",
            "realization_pass",
            "row_pass",
        ]
        review.gate("realized_chain_schema_exact", chain_fields == expected_chain_fields)
        review.gate(
            "finite_candidates_schema_exact",
            candidate_fields == [*endpoint_fields, "candidate_pass"],
        )

        endpoint_numeric_fields = [
            field for field in endpoint_fields if field != "endpoint_parameter_hash"
        ]
        tables_finite = all(
            math.isfinite(float(row[field]))
            for row in endpoint_rows
            for field in endpoint_numeric_fields
        )
        tables_finite &= all(
            math.isfinite(float(row[field]))
            for row in [*global_rows, *projected_rows]
            for field in ("alpha", "rank", "eigenvalue")
        )
        tables_finite &= all(
            math.isfinite(float(row[field]))
            for row in chain_rows
            for field in chain_fields
            if field
            not in {
                "objective",
                "expected_signs",
                "transitions_pass",
                "realization_pass",
                "row_pass",
            }
        )
        tables_finite &= all(
            math.isfinite(float(row[field]))
            for row in candidate_rows
            for field in endpoint_numeric_fields
        )
        review.gate("all_table_numeric_values_finite", tables_finite)

        endpoint_alphas = [_float(row["alpha"]) for row in endpoint_rows]
        review.gate(
            "endpoint_alpha_grid_exact",
            len(endpoint_rows) == len(ALPHAS)
            and np.array_equal(np.asarray(endpoint_alphas), np.asarray(ALPHAS)),
        )
        endpoint_by_alpha = {float(row["alpha"]): row for row in endpoint_rows}
        review.gate(
            "endpoint_alpha_keys_unique",
            len(endpoint_by_alpha) == len(endpoint_rows) == len(ALPHAS),
        )
        context["endpoint_by_alpha"] = endpoint_by_alpha

        parameter_hashes = [row["endpoint_parameter_hash"] for row in endpoint_rows]
        review.gate(
            "endpoint_parameter_hashes_well_formed_and_distinct",
            all(_is_sha256(value) for value in parameter_hashes)
            and len(set(parameter_hashes)) == len(ALPHAS),
        )
        review.gate(
            "endpoint_repeat_parameter_hashes_unchanged",
            all(int(float(row["repeat_parameter_hash_unchanged"])) == 1 for row in endpoint_rows),
        )

        discrete_repeat_fields = {
            "repeat_count_lt_1e_4_abs_error",
            "repeat_count_lt_1e_2_abs_error",
            "repeat_count_lt_0p1_abs_error",
        }
        continuous_repeat_fields = expected_repeat_fields - discrete_repeat_fields
        max_continuous_repeat = max(
            float(row[field])
            for row in endpoint_rows
            for field in continuous_repeat_fields
        )
        review.gate(
            "endpoint_continuous_and_spectrum_repeats_match",
            max_continuous_repeat <= 1e-10,
        )
        review.gate(
            "endpoint_discrete_repeats_match_exactly",
            all(
                float(row[field]) == 0.0
                for row in endpoint_rows
                for field in discrete_repeat_fields
            ),
        )

        def grouped_spectra(
            rows: list[dict[str, str]], dimension: int
        ) -> tuple[dict[float, np.ndarray], bool, bool]:
            grouped: dict[float, list[dict[str, str]]] = {alpha: [] for alpha in ALPHAS}
            unknown = False
            for row in rows:
                alpha = float(row["alpha"])
                if alpha not in grouped:
                    unknown = True
                else:
                    grouped[alpha].append(row)
            spectra: dict[float, np.ndarray] = {}
            grid_exact = not unknown and len(rows) == len(ALPHAS) * dimension
            sorted_nonnegative = True
            for alpha in ALPHAS:
                alpha_rows = grouped[alpha]
                ranks = [int(float(row["rank"])) for row in alpha_rows]
                grid_exact &= ranks == list(range(dimension))
                values = np.asarray(
                    [float(row["eigenvalue"]) for row in alpha_rows], dtype=np.float64
                )
                if values.shape != (dimension,):
                    grid_exact = False
                    continue
                spectra[alpha] = values
                sorted_nonnegative &= bool(
                    np.isfinite(values).all()
                    and np.all(np.diff(values) >= 0.0)
                    and np.all(values >= 0.0)
                )
            return spectra, grid_exact, sorted_nonnegative

        global_by_alpha, global_grid, global_sorted = grouped_spectra(
            global_rows, DIMENSION
        )
        projected_by_alpha, projected_grid, projected_sorted = grouped_spectra(
            projected_rows, PROJECTED_DIMENSION
        )
        context["global_by_alpha"] = global_by_alpha
        context["projected_by_alpha"] = projected_by_alpha
        review.gate("global_14x512_spectrum_grid_exact", global_grid)
        review.gate("projected_14x480_spectrum_grid_exact", projected_grid)
        review.gate(
            "global_and_projected_spectra_sorted_nonnegative",
            global_sorted and projected_sorted,
        )

        global_metrics_match = set(global_by_alpha) == set(ALPHAS)
        projected_metrics_match = set(projected_by_alpha) == set(ALPHAS)
        closure_passes = True
        counts_match = True
        frozen_energy_matches = True
        max_global_metric_error = 0.0
        max_projected_metric_error = 0.0
        recomputed_global: dict[float, dict[str, float]] = {}
        recomputed_projected: dict[float, dict[str, float]] = {}
        if global_metrics_match and projected_metrics_match:
            for alpha in ALPHAS:
                row = endpoint_by_alpha[alpha]
                global_metrics = _global_spectrum_metrics(global_by_alpha[alpha])
                projected_metrics = _projected_spectrum_metrics(projected_by_alpha[alpha])
                recomputed_global[alpha] = global_metrics
                recomputed_projected[alpha] = projected_metrics
                for field, expected in global_metrics.items():
                    observed = float(row[field])
                    max_global_metric_error = max(
                        max_global_metric_error, abs(observed - expected)
                    )
                    if field in {"a_direct_abs_error", "a_trace_abs_error"}:
                        continue
                    global_metrics_match &= _close(observed, expected)
                for field, expected in projected_metrics.items():
                    observed = float(row[field])
                    max_projected_metric_error = max(
                        max_projected_metric_error, abs(observed - expected)
                    )
                    projected_metrics_match &= _close(observed, expected)
                exact_a = float(row["exact_a_per_dim"])
                direct = float(row["a_direct_matrix"])
                trace_closure = float(row["a_trace_closure"])
                closure_passes &= (
                    _close(
                        exact_a,
                        float(row["a_constant_term"])
                        + float(row["a_linear_trace_term"])
                        + float(row["a_quartic_term"]),
                    )
                    and _close(direct, exact_a)
                    and _close(trace_closure, exact_a)
                    and _close(
                        float(row["a_direct_abs_error"]), abs(direct - exact_a), atol=1e-12
                    )
                    and _close(
                        float(row["a_trace_abs_error"]),
                        abs(trace_closure - exact_a),
                        atol=1e-12,
                    )
                    and abs(direct - exact_a) <= 1e-9
                    and abs(trace_closure - exact_a) <= 1e-9
                    and _close(
                        float(row["damped_full_burg_per_dim"]),
                        float(row["burg_trace_r_term"])
                        + float(row["burg_neg_logdet_r_term"])
                        - 1.0,
                    )
                    and _close(
                        float(row["burg_neg_logdet_r_term"]),
                        -float(row["logdet_r_per_dim"]),
                    )
                )
                expected_counts = {
                    "count_lt_1e_4": int(np.sum(global_by_alpha[alpha] < 1e-4)),
                    "count_lt_1e_2": int(np.sum(global_by_alpha[alpha] < 1e-2)),
                    "count_lt_0p1": int(np.sum(global_by_alpha[alpha] < 0.1)),
                }
                counts_match &= all(
                    float(row[field]) == value for field, value in expected_counts.items()
                )
                frozen_energy_matches &= _close(
                    row["frozen_low_energy"], projected_metrics["projected_low_mean"]
                )
        review.gate("global_spectrum_metrics_recompute", global_metrics_match)
        review.gate("projected_spectrum_metrics_recompute", projected_metrics_match)
        review.gate("exact_a_and_b_algebra_closes", closure_passes)
        review.gate("global_threshold_counts_recompute", counts_match)
        review.gate("frozen_low_energy_recomputes_from_projected_spectrum", frozen_energy_matches)
        context["recomputed_global"] = recomputed_global
        context["recomputed_projected"] = recomputed_projected

        base_global = global_by_alpha[0.0]
        base_projected = projected_by_alpha[0.0]
        base_low = base_global[base_global < LOW_THRESHOLD]
        base_dead = base_global[base_global < DEAD_THRESHOLD]
        base_row = endpoint_by_alpha[0.0]
        base_projected_error = float(np.max(np.abs(base_low - base_projected)))
        review.gate(
            "base_frozen_subspaces_recompute",
            len(base_low) == EXPECTED_LOW_COUNT
            and len(base_dead) == EXPECTED_DEAD_COUNT
            and base_projected_error <= 1e-9
            and _close(base_row["frozen_low_energy"], float(base_low.mean()))
            and _close(base_row["frozen_dead_energy"], float(base_dead.mean())),
        )
        review.details["spectrum_review"] = {
            "endpoint_rows": len(endpoint_rows),
            "global_spectrum_rows": len(global_rows),
            "projected_spectrum_rows": len(projected_rows),
            "repeat_metric_columns": len(expected_repeat_fields),
            "max_continuous_repeat_error": max_continuous_repeat,
            "max_global_metric_absolute_error": max_global_metric_error,
            "max_projected_metric_absolute_error": max_projected_metric_error,
            "base_low_count": len(base_low),
            "base_dead_count": len(base_dead),
            "base_projected_vs_global_low_max_absolute_error": base_projected_error,
            "base_frozen_low_energy_from_global_spectrum": float(base_low.mean()),
            "base_frozen_dead_energy_from_global_spectrum": float(base_dead.mean()),
        }

    if not review.phase("tables", tables_phase):
        return 0 if _write_report(output, review, None) else 1

    def checkpoint_replay_phase() -> None:
        decision = context["decision"]
        frozen_dependencies = context["frozen_dependencies"]
        endpoint_by_alpha = context["endpoint_by_alpha"]
        global_by_alpha = context["global_by_alpha"]

        checkpoint_file_hash = _sha256_file(ITERATION3_CHECKPOINT)
        metadata, active_hash, parameter_count = _checkpoint_metadata_and_hash(
            ITERATION3_CHECKPOINT
        )
        active_fields, active_rows = _read_csv(ACTIVE_PARAMETERS)
        if "parameter" not in active_fields:
            raise ValueError("active_parameters.csv lacks parameter")
        active_names = sorted(row["parameter"] for row in active_rows)
        checkpoint_names = metadata.get("active_names")
        review.gate(
            "iteration3_checkpoint_file_hash_matches",
            checkpoint_file_hash == EXPECTED_CHECKPOINT_FILE_SHA256
            and frozen_dependencies.get(
                "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
                "one_state_exact_a_h2048/iteration3_p32_unit_common_production/final_checkpoint.pt"
            )
            == checkpoint_file_hash,
        )
        review.gate(
            "checkpoint_active_key_coverage_exact",
            isinstance(checkpoint_names, list)
            and checkpoint_names == active_names
            and len(checkpoint_names) == len(set(checkpoint_names)),
        )
        review.gate(
            "checkpoint_active_hash_and_count_recompute",
            active_hash == EXPECTED_ACTIVE_SHA256
            and metadata.get("final_parameter_hash") == active_hash
            and parameter_count == ACTIVE_PARAMETER_COUNT,
        )
        review.gate(
            "checkpoint_identity_metadata_matches",
            metadata.get("protocol_id") == ITERATION3_PROTOCOL_ID
            and metadata.get("source_weight_index") == SOURCE_WEIGHT_INDEX
            and metadata.get("z_sha256") == EXPECTED_Z_SHA256,
        )
        review.gate(
            "iteration5_base_hash_matches_checkpoint",
            decision.get("base_parameter_hash") == active_hash
            and endpoint_by_alpha[0.0]["endpoint_parameter_hash"] == active_hash,
        )

        state_bank_fields, state_bank_rows = _read_csv(STATE_BANK)
        if not {"state_position", "source_weight_index"}.issubset(state_bank_fields):
            raise ValueError("state_bank.csv lacks identity columns")
        selected_state = [
            row
            for row in state_bank_rows
            if int(float(row["state_position"])) == STATE_POSITION
        ]
        review.gate(
            "state_source_identity_matches",
            len(selected_state) == 1
            and int(float(selected_state[0]["source_weight_index"]))
            == SOURCE_WEIGHT_INDEX,
        )

        state_fields, state_rows = _read_csv(ITERATION3_STATES)
        if "proposal" not in state_fields:
            raise ValueError("Iteration-3 state_metrics.csv lacks proposal")
        state100_rows = [row for row in state_rows if int(float(row["proposal"])) == 100]
        if len(state100_rows) != 1:
            raise ValueError(f"Iteration-3 state 100 row count is {len(state100_rows)}")
        state100 = state100_rows[0]
        spectrum_fields, spectrum_rows = _read_csv(ITERATION3_SPECTRA)
        if not {"proposal", "rank", "m_eigenvalue", "a_contribution"}.issubset(
            spectrum_fields
        ):
            raise ValueError("Iteration-3 state_spectra.csv lacks required columns")
        state100_spectrum_rows = [
            row for row in spectrum_rows if int(float(row["proposal"])) == 100
        ]
        ranks = [int(float(row["rank"])) for row in state100_spectrum_rows]
        state100_spectrum = np.asarray(
            [float(row["m_eigenvalue"]) for row in state100_spectrum_rows],
            dtype=np.float64,
        )
        review.gate(
            "iteration3_state100_spectrum_grid_exact",
            ranks == list(range(DIMENSION))
            and state100_spectrum.shape == (DIMENSION,)
            and np.isfinite(state100_spectrum).all()
            and bool(np.all(np.diff(state100_spectrum) >= 0.0)),
        )
        state100_contribution = np.asarray(
            [float(row["a_contribution"]) for row in state100_spectrum_rows],
            dtype=np.float64,
        )
        review.gate(
            "iteration3_state100_contribution_recomputes",
            bool(
                np.allclose(
                    state100_contribution,
                    np.square(state100_spectrum - 1.0),
                    atol=1e-12,
                    rtol=1e-10,
                )
            ),
        )
        state100_recomputed = _global_spectrum_metrics(state100_spectrum)
        common_state_fields = set(state100_recomputed).intersection(state_fields)
        review.gate(
            "iteration3_state100_metrics_recompute_from_spectrum",
            all(_close(state100[field], state100_recomputed[field]) for field in common_state_fields),
        )

        base_spectrum_error = float(
            np.max(np.abs(global_by_alpha[0.0] - state100_spectrum))
        )
        review.gate("state100_full_spectrum_replays_at_base", base_spectrum_error <= 1e-9)
        replay_fields = (
            "exact_a_per_dim",
            "damped_full_burg_per_dim",
            "m_max",
            "m_p50",
            "m_lt_1e_4_fraction",
            "m_lt_0p01_fraction",
            "m_lt_0p1_fraction",
        )
        base_metrics = decision.get("base_metrics")
        if not isinstance(base_metrics, dict):
            raise TypeError("decision base_metrics must be an object")
        replay_errors = {
            field: abs(float(base_metrics[field]) - float(state100[field]))
            for field in replay_fields
        }
        endpoint_errors = {
            field: abs(float(endpoint_by_alpha[0.0][field]) - float(state100[field]))
            for field in replay_fields
        }
        review.gate(
            "state100_metrics_replay_at_base",
            max([*replay_errors.values(), *endpoint_errors.values()]) <= 1e-9,
        )
        review.gate(
            "producer_base_replay_records_recompute",
            _equivalent(decision.get("replay_errors"), replay_errors)
            and _close(
                decision.get("replay_spectrum_error"),
                base_spectrum_error,
                atol=1e-15,
                rtol=1e-9,
            ),
        )
        review.gate(
            "decision_base_metrics_recompute_from_spectrum",
            all(_close(base_metrics[field], value) for field, value in state100_recomputed.items()),
        )
        iteration3_decision = _load_json(ITERATION3_DECISION)
        iteration3_review = _load_json(ITERATION3_REVIEW)
        review.gate(
            "iteration3_inputs_were_independently_validated",
            iteration3_decision.get("valid") is True
            and iteration3_review.get("valid") is True,
        )
        review.details["checkpoint_state100_review"] = {
            "checkpoint_file_sha256": checkpoint_file_hash,
            "active_parameter_sha256": active_hash,
            "active_parameter_count": parameter_count,
            "active_tensor_count": len(checkpoint_names),
            "checkpoint_metadata": {
                key: value for key, value in metadata.items() if key != "active_names"
            },
            "state100_metric_absolute_errors": replay_errors,
            "state100_endpoint_metric_absolute_errors": endpoint_errors,
            "state100_spectrum_max_absolute_error": base_spectrum_error,
        }

    if not review.phase("checkpoint_replay", checkpoint_replay_phase):
        return 0 if _write_report(output, review, None) else 1

    def gradient_phase() -> None:
        gradient = context["gradient"]
        decision = context["decision"]
        resolved = context["resolved"]
        base_spectrum = context["global_by_alpha"][0.0]
        base_low = base_spectrum[base_spectrum < LOW_THRESHOLD]

        retained = [int(value) for value in gradient["retained_memory_by_block_bytes"]]
        peaks = [int(value) for value in gradient["peak_memory_by_block_bytes"]]
        block_count = int(gradient["block_count"])
        early = float(np.median(retained[:2]))
        late = float(np.median(retained[-2:]))
        retained_range = float(max(retained) - min(retained))
        peak_range = float(max(peaks) - min(peaks))
        memory_gate = (
            late - early <= RETAINED_MEMORY_RANGE_MAX
            and retained_range <= RETAINED_MEMORY_RANGE_MAX
            and peak_range <= BLOCK_PEAK_RANGE_MAX
        )
        review.gate(
            "blocked_gradient_coverage_dtype_and_memory_recompute",
            gradient.get("basis_count") == DIMENSION
            and gradient.get("block_size") == 64
            and block_count == 8
            and len(retained) == block_count
            and len(peaks) == block_count
            and gradient.get("a_unused_parameter_tensors_all_blocks") == 0
            and gradient.get("low_unused_parameter_tensors_all_blocks") == 0
            and gradient.get("accumulation_dtype") == "float64"
            and gradient.get("all_a_accumulators_float64") is True
            and gradient.get("all_low_accumulators_float64") is True
            and _close(gradient.get("memory_early_median_bytes"), early, atol=0.0)
            and _close(gradient.get("memory_late_median_bytes"), late, atol=0.0)
            and _close(
                gradient.get("memory_growth_bytes"), max(0.0, late - early), atol=0.0
            )
            and _close(gradient.get("retained_memory_range_bytes"), retained_range, atol=0.0)
            and _close(gradient.get("block_peak_min_bytes"), min(peaks), atol=0.0)
            and _close(gradient.get("block_peak_max_bytes"), max(peaks), atol=0.0)
            and _close(gradient.get("block_peak_range_bytes"), peak_range, atol=0.0)
            and gradient.get("memory_growth_gate_pass") is memory_gate,
        )

        projector_gates = {
            "low_count_matches": len(base_low) == EXPECTED_LOW_COUNT,
            "dead_count_matches": int(np.sum(base_spectrum < DEAD_THRESHOLD))
            == EXPECTED_DEAD_COUNT,
            "orthogonality_passes": float(
                gradient["projector_orthogonality_max_abs"]
            )
            <= ORTHOGONALITY_MAX,
            "eigen_residual_passes": float(
                gradient["projector_eigen_residual_relative"]
            )
            <= EIGEN_RESIDUAL_MAX,
            "eigen_order_passes": float(gradient["eigen_order_min_diff"])
            >= -EIGEN_ORDER_TOL,
        }
        review.gate(
            "projector_counts_and_declared_numerical_gates_recompute",
            _is_sha256(gradient.get("projector_hash"))
            and gradient.get("projector_hash") == decision.get("projector_hash")
            and decision.get("low_count") == EXPECTED_LOW_COUNT
            and decision.get("dead_count") == EXPECTED_DEAD_COUNT
            and _equivalent(decision.get("projector_gates"), projector_gates)
            and all(projector_gates.values()),
        )

        expected_k_a_norm = math.sqrt(
            16.0
            / float(DIMENSION * DIMENSION)
            * float(np.sum(base_spectrum * np.square(base_spectrum - 1.0)))
        )
        expected_k_low_norm = math.sqrt(
            4.0
            / float(EXPECTED_LOW_COUNT * EXPECTED_LOW_COUNT)
            * float(base_low.sum())
        )
        review.gate(
            "h_space_cotangent_norm_identities_recompute",
            _close(gradient.get("k_a_norm"), expected_k_a_norm)
            and _close(gradient.get("k_low_norm"), expected_k_low_norm),
        )

        norm_a = _float(gradient["gradient_a_norm"])
        norm_low = _float(gradient["gradient_low_norm"])
        cosine = _float(gradient["gradient_cosine"])
        common_norm = _float(gradient["unit_common_source_norm"])
        common_from_cosine = math.sqrt(max(0.0, 2.0 + 2.0 * cosine))
        direction_norm = _float(decision["direction_norm"])
        nominal_a = -direction_norm * norm_a * (1.0 + cosine) / common_norm
        nominal_low = -direction_norm * norm_low * (1.0 + cosine) / common_norm
        review.gate(
            "gradient_cosine_common_norm_identity_recomputes",
            norm_a > 0.0
            and norm_low > 0.0
            and -1.0 <= cosine <= 1.0
            and _close(common_norm, common_from_cosine, atol=1e-8, rtol=1e-8),
        )
        review.gate(
            "nominal_equal_unit_derivatives_recompute",
            _close(
                decision.get("nominal_a_derivative"),
                nominal_a,
                atol=1e-6,
                rtol=1e-6,
            )
            and _close(
                decision.get("nominal_low_loss_derivative"),
                nominal_low,
                atol=1e-8,
                rtol=1e-6,
            )
            and float(decision["nominal_a_derivative"]) < 0.0
            and float(decision["nominal_low_loss_derivative"]) < 0.0,
        )
        review.gate(
            "common_direction_branch_and_radius_recompute",
            common_norm >= CANCELLATION_NORM_MIN
            and abs(direction_norm - TARGET_NORM) <= 1e-6
            and _close(resolved.get("target_norm"), TARGET_NORM, atol=1e-15, rtol=0.0),
        )
        review.details["gradient_review"] = {
            "gradient_a_norm": norm_a,
            "gradient_low_norm": norm_low,
            "gradient_cosine": cosine,
            "common_norm_stored": common_norm,
            "common_norm_from_cosine": common_from_cosine,
            "direction_norm": direction_norm,
            "nominal_a_stored": float(decision["nominal_a_derivative"]),
            "nominal_a_from_scalar_identity": nominal_a,
            "nominal_low_stored": float(decision["nominal_low_loss_derivative"]),
            "nominal_low_from_scalar_identity": nominal_low,
            "k_a_norm_stored": float(gradient["k_a_norm"]),
            "k_a_norm_from_base_spectrum": expected_k_a_norm,
            "k_low_norm_stored": float(gradient["k_low_norm"]),
            "k_low_norm_from_base_spectrum": expected_k_low_norm,
            "retained_memory_range_bytes": retained_range,
            "block_peak_range_bytes": peak_range,
        }
        review.limitations.append(
            "The full H, Q_low, and pulled-back gradient vectors were not stored. Their exact "
            "entries and projector hash cannot be regenerated post-run; this review instead "
            "checks the frozen source, base-spectrum cotangent norm identities, gradient scalar "
            "identities, projector diagnostics, and realized-chain records."
        )

    if not review.phase("gradient", gradient_phase):
        return 0 if _write_report(output, review, None) else 1

    recomputed_outcome: str | None = None

    def chain_candidates_outcome_phase() -> None:
        nonlocal recomputed_outcome
        decision = context["decision"]
        finalized = context["finalized"]
        resolved = context["resolved"]
        endpoint_fields = context["endpoint_fields"]
        endpoint_rows = context["endpoint_rows"]
        endpoint_by_alpha = context["endpoint_by_alpha"]
        global_by_alpha = context["global_by_alpha"]
        projected_by_alpha = context["projected_by_alpha"]
        chain_rows = context["chain_rows"]
        candidate_rows = context["candidate_rows"]
        run_log = context["run_log"]

        chain_by_key = {
            (row["objective"], float(row["radius"])): row for row in chain_rows
        }
        expected_chain_keys = {
            (objective, radius)
            for objective in ("A", "L_low")
            for radius in LOCAL_RADII
        }
        review.gate(
            "realized_chain_key_grid_exact",
            len(chain_rows) == len(expected_chain_keys)
            and set(chain_by_key) == expected_chain_keys,
        )
        chain_arithmetic = True
        chain_boolean_records = True
        shared_realization_records = True
        recomputed_rows: list[dict[str, Any]] = []
        for radius in LOCAL_RADII:
            plus = endpoint_by_alpha[radius]
            minus = endpoint_by_alpha[-radius]
            expected_direct = {
                "A": (
                    float(plus["exact_a_per_dim"])
                    - float(minus["exact_a_per_dim"])
                )
                / (2.0 * radius),
                "L_low": -(
                    float(plus["frozen_low_energy"])
                    - float(minus["frozen_low_energy"])
                )
                / (2.0 * radius),
            }
            a_row = chain_by_key[("A", radius)]
            low_row = chain_by_key[("L_low", radius)]
            shared_fields = (
                "effective_direction_norm",
                "desired_direction_norm",
                "effective_direction_norm_ratio",
                "effective_direction_cosine",
                "midpoint_drift_norm",
                "midpoint_drift_relative",
                "s_theta_a",
                "s_theta_low",
            )
            shared_realization_records &= all(
                _close(a_row[field], low_row[field], atol=1e-15, rtol=1e-12)
                for field in shared_fields
            )
            for objective in ("A", "L_low"):
                row = chain_by_key[(objective, radius)]
                nominal = float(row["nominal"])
                s_theta = float(row["s_theta"])
                s_h = float(row["s_h"])
                direct = float(row["direct"])
                errors = {
                    "relative_n_to_theta": _relative_error(nominal, s_theta),
                    "relative_theta_to_h": _relative_error(s_theta, s_h),
                    "relative_h_to_direct": _relative_error(s_h, direct),
                    "relative_n_to_direct": _relative_error(nominal, direct),
                }
                expected_nominal = float(
                    decision[
                        "nominal_a_derivative"
                        if objective == "A"
                        else "nominal_low_loss_derivative"
                    ]
                )
                expected_s_theta = float(
                    row["s_theta_a"] if objective == "A" else row["s_theta_low"]
                )
                expected_signs = all(
                    value < 0.0 for value in (nominal, s_theta, s_h, direct)
                )
                transitions_pass = all(
                    errors[field] <= CHAIN_RELATIVE_ERROR_MAX
                    for field in (
                        "relative_n_to_theta",
                        "relative_theta_to_h",
                        "relative_h_to_direct",
                    )
                )
                realization_pass = (
                    float(row["effective_direction_cosine"])
                    >= EFFECTIVE_COSINE_MIN
                    and EFFECTIVE_NORM_RATIO_MIN
                    <= float(row["effective_direction_norm_ratio"])
                    <= EFFECTIVE_NORM_RATIO_MAX
                    and float(row["midpoint_drift_relative"])
                    <= MIDPOINT_RELATIVE_MAX
                )
                row_pass = expected_signs and transitions_pass and realization_pass
                chain_arithmetic &= (
                    _close(nominal, expected_nominal, atol=1e-12)
                    and _close(s_theta, expected_s_theta, atol=1e-12)
                    and _close(direct, expected_direct[objective], atol=1e-12)
                    and all(_close(row[field], value, atol=1e-12) for field, value in errors.items())
                    and _close(
                        row["desired_direction_norm"],
                        decision["direction_norm"],
                        atol=1e-12,
                    )
                )
                chain_boolean_records &= (
                    _bool_text(row["expected_signs"]) is expected_signs
                    and _bool_text(row["transitions_pass"]) is transitions_pass
                    and _bool_text(row["realization_pass"]) is realization_pass
                    and _bool_text(row["row_pass"]) is row_pass
                )
                recomputed_rows.append(
                    {
                        "objective": objective,
                        "radius": radius,
                        "direct_from_endpoints": expected_direct[objective],
                        "relative_errors": errors,
                        "expected_signs": expected_signs,
                        "transitions_pass": transitions_pass,
                        "realization_pass": realization_pass,
                        "row_pass": row_pass,
                    }
                )
        review.gate("realized_chain_arithmetic_recomputes", chain_arithmetic)
        review.gate("realized_chain_boolean_outcomes_recompute", chain_boolean_records)
        review.gate("realized_chain_shared_secants_are_consistent", shared_realization_records)

        cauchy_errors = {
            objective: _relative_error(
                float(chain_by_key[(objective, LOCAL_RADII[0])]["direct"]),
                float(chain_by_key[(objective, LOCAL_RADII[1])]["direct"]),
            )
            for objective in ("A", "L_low")
        }
        row_passes = all(item["row_pass"] for item in recomputed_rows)
        local_chain_pass = row_passes and all(
            value <= CHAIN_CAUCHY_RELATIVE_ERROR_MAX for value in cauchy_errors.values()
        )
        review.gate(
            "local_chain_and_cauchy_decision_recomputes",
            _equivalent(decision.get("cauchy_relative_errors"), cauchy_errors)
            and decision.get("local_chain_pass") is local_chain_pass
            and decision.get("local_chain_decision")
            == ("joint_local_chain_valid" if local_chain_pass else "no_local_chain_claim"),
        )

        tolerances = {
            metric: max(
                5.0
                * max(
                    float(row[f"repeat_{metric}_abs_error"])
                    for row in endpoint_rows
                ),
                floor,
            )
            for metric, floor in METRIC_FLOORS.items()
        }
        review.gate(
            "repeat_derived_candidate_tolerances_recompute",
            _equivalent(decision.get("metric_tolerances"), tolerances)
            and _equivalent(resolved.get("metric_floors"), METRIC_FLOORS),
        )

        candidate_by_alpha = {float(row["alpha"]): row for row in candidate_rows}
        review.gate(
            "finite_candidate_grid_exact",
            len(candidate_rows) == len(CANDIDATE_ALPHAS)
            and list(candidate_by_alpha) == list(CANDIDATE_ALPHAS),
        )
        candidate_copies = all(
            candidate_by_alpha[alpha][field] == endpoint_by_alpha[alpha][field]
            for alpha in CANDIDATE_ALPHAS
            for field in endpoint_fields
        )
        review.gate("finite_candidates_are_exact_endpoint_rows", candidate_copies)

        base = endpoint_by_alpha[0.0]
        candidate_gate_details: dict[str, dict[str, bool]] = {}
        passing_alphas: list[float] = []
        candidate_records_match = True
        spectra_gate = True
        for alpha in CANDIDATE_ALPHAS:
            row = candidate_by_alpha[alpha]
            gates = {
                "exact_a_lower": float(row["exact_a_per_dim"])
                < float(base["exact_a_per_dim"]) - tolerances["exact_a_per_dim"],
                "damped_b_lower": float(row["damped_full_burg_per_dim"])
                < float(base["damped_full_burg_per_dim"])
                - tolerances["damped_full_burg_per_dim"],
                "frozen_low_energy_higher": float(row["frozen_low_energy"])
                > float(base["frozen_low_energy"])
                + tolerances["frozen_low_energy"],
                "global_median_higher": float(row["m_p50"])
                > float(base["m_p50"]) + tolerances["m_p50"],
                "projected_median_higher": float(row["projected_low_p50"])
                > float(base["projected_low_p50"])
                + tolerances["projected_low_p50"],
                "global_max_nonincreasing": float(row["m_max"])
                <= float(base["m_max"]) + tolerances["m_max"],
                "global_effective_rank_nondecreasing": float(row["effective_rank"])
                >= float(base["effective_rank"]) - tolerances["effective_rank"],
                "projected_participation_rank_nondecreasing": float(
                    row["projected_low_participation_rank"]
                )
                >= float(base["projected_low_participation_rank"])
                - tolerances["projected_low_participation_rank"],
                "projected_entropy_rank_nondecreasing": float(
                    row["projected_low_entropy_rank"]
                )
                >= float(base["projected_low_entropy_rank"])
                - tolerances["projected_low_entropy_rank"],
                "count_lt_1e_4_nonincreasing": float(row["count_lt_1e_4"])
                <= float(base["count_lt_1e_4"]),
                "count_lt_1e_2_nonincreasing": float(row["count_lt_1e_2"])
                <= float(base["count_lt_1e_2"]),
                "global_raw_min_within_tolerance": float(row["m_raw_eig_min"])
                >= -1e-8,
                "projected_raw_min_within_tolerance": float(
                    row["projected_low_raw_min"]
                )
                >= -1e-8,
            }
            candidate_pass = all(gates.values())
            candidate_records_match &= _bool_text(row["candidate_pass"]) is candidate_pass
            spectra_gate &= (
                np.isfinite(global_by_alpha[alpha]).all()
                and np.isfinite(projected_by_alpha[alpha]).all()
                and bool(np.all(global_by_alpha[alpha] >= 0.0))
                and bool(np.all(projected_by_alpha[alpha] >= 0.0))
            )
            if candidate_pass:
                passing_alphas.append(alpha)
            candidate_gate_details[str(alpha)] = gates
        review.gate("finite_candidate_gate_records_recompute", candidate_records_match)
        review.gate("finite_candidate_spectra_are_finite_nonnegative", spectra_gate)
        selected_alpha = max(passing_alphas) if passing_alphas else None
        review.gate(
            "finite_candidate_selection_recomputes",
            decision.get("finite_candidate_count") == len(passing_alphas)
            and _equivalent(decision.get("selected_largest_passing_alpha"), selected_alpha)
            and decision.get("finite_line_decision")
            == (
                "finite_joint_repair"
                if selected_alpha is not None
                else "no_guarded_finite_endpoint"
            ),
        )

        all_repeat_fields = [
            field
            for field in endpoint_fields
            if field.startswith("repeat_") and field.endswith("error")
        ]
        max_repeat_error = max(
            float(row[field]) for row in endpoint_rows for field in all_repeat_fields
        )
        finite_line_valid = (
            len(endpoint_rows) == len(ALPHAS)
            and len(context["global_rows"]) == len(ALPHAS) * DIMENSION
            and len(context["projected_rows"])
            == len(ALPHAS) * PROJECTED_DIMENSION
            and all(
                int(float(row["repeat_parameter_hash_unchanged"])) == 1
                for row in endpoint_rows
            )
            and max_repeat_error <= 1e-10
            and all(float(row["a_direct_abs_error"]) <= 1e-9 for row in endpoint_rows)
            and all(float(row["a_trace_abs_error"]) <= 1e-9 for row in endpoint_rows)
        )
        review.gate(
            "finite_line_validity_recomputes",
            finite_line_valid
            and decision.get("validity_gates", {}).get("finite_line_valid") is True,
        )

        if not local_chain_pass:
            recomputed_outcome = "no_local_chain_claim"
        elif selected_alpha is not None:
            recomputed_outcome = "finite_joint_repair"
        else:
            recomputed_outcome = "local_only_finite_coupling"
        review.gate(
            "protocol_outcome_recomputes",
            decision.get("outcome") == recomputed_outcome
            and finalized.get("outcome") == recomputed_outcome,
        )

        endpoint_hash_evidence = all(
            int(float(row["repeat_parameter_hash_unchanged"])) == 1
            for row in endpoint_rows
        )
        restoration_evidence = (
            endpoint_by_alpha[0.0]["endpoint_parameter_hash"] == EXPECTED_ACTIVE_SHA256
            and decision.get("base_parameter_hash") == EXPECTED_ACTIVE_SHA256
            and endpoint_hash_evidence
            and decision.get("validity_gates", {}).get("parameters_restored") is True
            and "base restoration failed" not in run_log
            and run_log.count("[a-low-i5] endpoint=") == len(ALPHAS)
        )
        review.gate("stored_restoration_evidence_is_consistent", restoration_evidence)
        review.details["chain_review"] = {
            "rows": recomputed_rows,
            "cauchy_relative_errors": cauchy_errors,
            "all_rows_pass": row_passes,
            "local_chain_pass": local_chain_pass,
            "producer_local_chain_pass": decision.get("local_chain_pass"),
        }
        review.details["finite_candidate_review"] = {
            "metric_tolerances": tolerances,
            "candidate_gate_results": candidate_gate_details,
            "passing_alphas": passing_alphas,
            "selected_largest_passing_alpha": selected_alpha,
        }
        review.details["outcome_review"] = {
            "recomputed_outcome": recomputed_outcome,
            "producer_outcome": decision.get("outcome"),
            "local_chain_pass": local_chain_pass,
            "finite_candidate_count": len(passing_alphas),
        }
        review.details["restoration_review"] = {
            "base_checkpoint_active_sha256": EXPECTED_ACTIVE_SHA256,
            "alpha_zero_endpoint_sha256": endpoint_by_alpha[0.0][
                "endpoint_parameter_hash"
            ],
            "all_repeat_hashes_unchanged": endpoint_hash_evidence,
            "producer_final_restoration_gate": decision.get("validity_gates", {}).get(
                "parameters_restored"
            ),
            "endpoint_log_records": run_log.count("[a-low-i5] endpoint="),
            "independent_post_restore_hash_available": False,
        }
        review.limitations.append(
            "Repeat evaluations are represented by manifest-bound per-metric and full-spectrum "
            "maximum errors plus unchanged parameter hashes; the second spectra were not stored "
            "as separate rows and therefore cannot be reconstructed independently."
        )
        review.limitations.append(
            "Final restoration is supported by the checkpoint/base/alpha-zero hash chain, all "
            "endpoint repeat no-mutation records, the complete endpoint log, the frozen producer "
            "restoration checks, and the producer gate. No separate post-restore checkpoint was "
            "stored, so the final ephemeral in-memory state cannot be independently rehashed."
        )

    if not review.phase("chain_candidates_outcome", chain_candidates_outcome_phase):
        return 0 if _write_report(output, review, recomputed_outcome) else 1

    valid = _write_report(output, review, recomputed_outcome)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
