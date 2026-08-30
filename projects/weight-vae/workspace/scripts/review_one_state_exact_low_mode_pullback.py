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
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_low_mode_pullback_i4_v1"
ITERATION3_PROTOCOL_ID = "one_state_exact_a_p32_unit_common_i3_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration4_low_mode_pullback_production"
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_4_low_mode_pullback/protocol.md"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_4_frozen_dependency_manifest.json"
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
EPSILON = 1e-4
LOW_THRESHOLD = 0.1
DEAD_THRESHOLD = 1e-4
EXPECTED_LOW_COUNT = 480
EXPECTED_DEAD_COUNT = 309
TARGET_NORM = 0.04892722657548397
ALPHAS = (
    -1.0 / 256.0,
    0.0,
    1.0 / 256.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
)
H_FD_STEPS = (1e-4, 5e-5)
SOURCE_WEIGHT_INDEX = 378
STATE_POSITION = 2
ACTIVE_PARAMETER_COUNT = 11_685_120
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_CHECKPOINT_FILE_SHA256 = (
    "ccafbad0e0fcf25aaabffe0a406558937d1b448ac06c80b7c40a6637209dc5ad"
)
EXPECTED_ACTIVE_SHA256 = "31abec6439564c392a99b6c82dd1386aed714cb1b2a1ce1c4709088686a0caff"
EXPECTED_ACCEPTED_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)

EXPECTED_MANIFEST_ARTIFACTS = {
    "decision.json",
    "endpoint_metrics.csv",
    "endpoint_spectra.csv",
    "executed_source_snapshot.py",
    "frozen_dependency_manifest_snapshot.json",
    "gradient_diagnostics.json",
    "h_space_fd.csv",
    "low_mode_line.png",
    "low_mode_spectra.png",
    "protocol_snapshot.md",
    "resolved_config.json",
}

# These hashes bind the human visual inspection in this review to immutable files.
MANUALLY_INSPECTED_PLOTS = {
    "low_mode_line.png": {
        "sha256": "6cb8951ec27b2654df7007c139e5bdf7e66ddd613ac5b47c77dd2e7faa010697",
        "expected_dimensions": [2700, 1800],
        "assessment": (
            "Readable four-panel line plot; plotted nonnegative endpoints agree with the "
            "tables. The negative FD endpoint is intentionally absent from this producer plot."
        ),
    },
    "low_mode_spectra.png": {
        "sha256": "f296ffc67f08c097187b17185ce79a2646bb83f3fc012d17903055fe32132368",
        "expected_dimensions": [2520, 900],
        "assessment": (
            "Ordered spectra and count fractions are readable and agree with the tables. "
            "The median trace is compressed near zero by the shared linear y-axis, so its "
            "trend must be read from endpoint_metrics.csv."
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
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR" or not data.endswith(
        b"IEND\xaeB`\x82"
    ):
        raise ValueError(f"invalid PNG structure: {path}")
    width, height = struct.unpack(">II", data[16:24])
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "bytes": len(data),
    }


def _spectrum_metrics(eigenvalues: np.ndarray) -> dict[str, float]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    contribution = np.square(eigenvalues - 1.0)
    total_a = max(float(contribution.sum()), 1e-30)
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
    sorted_gradient = np.sort(gradient_eigenvalues)
    return {
        "exact_a_per_dim": float(contribution.mean()),
        "a_constant_term": 1.0,
        "a_trace_closure": float(
            1.0 - 2.0 * eigenvalues.mean() + np.square(eigenvalues).mean()
        ),
        "a_linear_trace_term": float(-2.0 * eigenvalues.mean()),
        "a_quartic_term": float(np.square(eigenvalues).mean()),
        "trace_m_per_dim": float(eigenvalues.mean()),
        "damped_full_burg_per_dim": trace_r + neg_logdet - 1.0,
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
        "m_raw_eig_min": float(quantiles[0]),
        "m_lt_1e_4_fraction": float(np.mean(eigenvalues < 1e-4)),
        "m_lt_0p01_fraction": float(np.mean(eigenvalues < 0.01)),
        "m_lt_0p1_fraction": float(np.mean(eigenvalues < 0.1)),
        "m_lt_0p5_fraction": float(np.mean(eigenvalues < 0.5)),
        "m_near_1_10pct_fraction": float(np.mean(np.abs(eigenvalues - 1.0) <= 0.1)),
        "m_gt_1_fraction": float(np.mean(eigenvalues > 1.0)),
        "m_gt_2_fraction": float(np.mean(eigenvalues > 2.0)),
        "a_from_m_lt_0p1_share": float(contribution[eigenvalues < 0.1].sum() / total_a),
        "a_from_m_gt_1_share": float(contribution[eigenvalues > 1.0].sum() / total_a),
        "burg_matrix_gradient_norm": float(np.linalg.norm(gradient_eigenvalues)),
        "burg_matrix_gradient_eig_min": float(gradient_eigenvalues.min()),
        "burg_matrix_gradient_eig_p50": float(sorted_gradient[(DIMENSION - 1) // 2]),
        "burg_matrix_gradient_eig_max": float(gradient_eigenvalues.max()),
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
        "a_top1_share": float(contribution[-1] / total_a),
        "a_top10_share": float(contribution[-10:].sum() / total_a),
        "top1_trace_share": float(eigenvalues[-1] / max(trace, 1e-30)),
        "top10_trace_share": float(eigenvalues[-10:].sum() / max(trace, 1e-30)),
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
        f"[low-mode-i4-review] report={report_path} valid={valid} "
        f"outcome={report['outcome']}",
        flush=True,
    )
    if not valid:
        print(f"[low-mode-i4-review] failed_gates={report['failed_gates']}", flush=True)
        for error in review.errors:
            print(f"[low-mode-i4-review] error={error}", flush=True)
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
            f"[low-mode-i4-review] finalized production is unavailable: output={output} "
            f"staging_exists={staging.exists()}",
            flush=True,
        )
        return 1
    print(f"[low-mode-i4-review] start output={output} device=cpu", flush=True)

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
            finalized.get("status") == "complete_awaiting_independent_review",
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

        source_snapshot = output / "executed_source_snapshot.py"
        source_hash = _sha256_file(source_snapshot)
        normalized_source_hash = _normalized_runner_sha256(source_snapshot)
        producer_source = ROOT / "scripts/audit_one_state_exact_low_mode_pullback.py"
        review.gate(
            "executed_source_hashes_match",
            source_hash == resolved.get("source_sha256")
            and source_hash == manifest.get("executed_source_sha256")
            and normalized_source_hash == resolved.get("normalized_source_sha256")
            and normalized_source_hash == manifest.get("executed_normalized_source_sha256"),
        )
        review.gate(
            "live_producer_matches_executed_snapshot",
            producer_source.is_file()
            and _sha256_file(producer_source) == source_hash
            and _normalized_runner_sha256(producer_source) == normalized_source_hash,
        )

        protocol_snapshot = output / "protocol_snapshot.md"
        protocol_hash = _sha256_file(protocol_snapshot)
        review.gate(
            "protocol_snapshot_and_current_hash_match",
            protocol_hash == resolved.get("protocol_sha256")
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
            dependency_hash == resolved.get("frozen_dependency_manifest_sha256")
            and FROZEN_DEPENDENCY_MANIFEST.is_file()
            and _sha256_file(FROZEN_DEPENDENCY_MANIFEST) == dependency_hash,
        )
        review.gate(
            "all_frozen_dependencies_still_match",
            bool(dependency_results)
            and all(dependency_results.values())
            and _equivalent(resolved.get("frozen_dependency_matches"), dependency_results),
        )
        accepted_dependencies = [
            (relative, digest)
            for relative, digest in frozen_dependencies.items()
            if relative.endswith("/vae_checkpoint.pt")
        ]
        review.gate(
            "accepted_checkpoint_dependency_hash_matches",
            len(accepted_dependencies) == 1
            and accepted_dependencies[0][1] == EXPECTED_ACCEPTED_CHECKPOINT_SHA256
            and dependency_results.get(accepted_dependencies[0][0]) is True,
        )
        review.details["hash_review"] = {
            "artifact_hashes_match": artifact_results,
            "frozen_dependency_matches": dependency_results,
            "executed_source_sha256": source_hash,
            "executed_normalized_source_sha256": normalized_source_hash,
            "protocol_sha256": protocol_hash,
            "dependency_manifest_sha256": dependency_hash,
        }
        context["frozen_dependencies"] = frozen_dependencies

        review.gate(
            "resolved_protocol_constants_match",
            resolved.get("device") == "cuda:0"
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
            and _equivalent(resolved.get("alphas"), list(ALPHAS), atol=1e-15, rtol=0.0)
            and _equivalent(
                resolved.get("h_fd_steps"), list(H_FD_STEPS), atol=1e-15, rtol=0.0
            ),
        )
        producer_gates = decision.get("validity_gates")
        review.gate(
            "producer_declared_valid",
            finalized.get("valid") is True
            and decision.get("valid") is True
            and isinstance(producer_gates, dict)
            and bool(producer_gates)
            and all(value is True for value in producer_gates.values()),
        )
        review.gate(
            "producer_outcomes_are_consistent",
            finalized.get("outcome") == decision.get("outcome"),
        )

        run_log_path = output / "run.log"
        run_log = run_log_path.read_text(encoding="utf-8", errors="replace")
        context["run_log"] = run_log
        review.gate("run_log_exists_and_nonempty", run_log_path.stat().st_size > 1_000)
        review.gate(
            "run_log_contains_successful_completion",
            "[low-mode-i4] complete valid=True" in run_log
            and "[low-mode-i4] published output=" in run_log
            and "[low-mode-i4] FAILED" not in run_log,
        )

        plot_details: dict[str, Any] = {}
        plots_match_inspection = True
        for name, inspected in MANUALLY_INSPECTED_PLOTS.items():
            info = _png_info(output / name)
            match = (
                info["sha256"] == inspected["sha256"]
                and [info["width"], info["height"]] == inspected["expected_dimensions"]
            )
            plots_match_inspection &= match
            plot_details[name] = {
                **info,
                "matches_manually_inspected_file": match,
                "manual_assessment": inspected["assessment"],
            }
        review.gate("plots_match_manually_inspected_files", plots_match_inspection)
        review.details["plot_inspection"] = plot_details
        review.warnings.append(
            "low_mode_spectra.png uses a shared linear bulk-response axis; the median curve "
            "is visually compressed, although its stored values are independently checked."
        )

    if not review.phase("metadata", metadata_phase):
        return 0 if _write_report(output, review, None) else 1

    def tables_phase() -> None:
        endpoint_fields, endpoint_rows = _read_csv(output / "endpoint_metrics.csv")
        spectrum_fields, spectrum_rows = _read_csv(output / "endpoint_spectra.csv")
        h_fd_fields, h_fd_rows = _read_csv(output / "h_space_fd.csv")
        context.update(
            endpoint_fields=endpoint_fields,
            endpoint_rows=endpoint_rows,
            spectrum_fields=spectrum_fields,
            spectrum_rows=spectrum_rows,
            h_fd_fields=h_fd_fields,
            h_fd_rows=h_fd_rows,
        )

        required_endpoint_fields = {
            "alpha",
            "endpoint_parameter_hash",
            "repeat_parameter_hash_unchanged",
            "exact_a_per_dim",
            "a_constant_term",
            "a_linear_trace_term",
            "a_quartic_term",
            "a_direct_matrix",
            "a_trace_closure",
            "a_direct_abs_error",
            "a_trace_abs_error",
            "trace_m_per_dim",
            "damped_full_burg_per_dim",
            "burg_trace_r_term",
            "burg_neg_logdet_r_term",
            "logdet_r_per_dim",
            "frozen_low_energy",
            "frozen_dead_energy",
            "m_max",
            "m_p50",
            "count_lt_1e_4",
            "count_lt_1e_2",
            "count_lt_0p1",
            "a_low90_abs_per_dim",
            "repeat_spectrum_max_abs_error",
            "repeat_count_lt_1e_4_abs_error",
            "repeat_count_lt_1e_2_abs_error",
            "repeat_count_lt_0p1_abs_error",
        }
        review.gate(
            "endpoint_required_columns_present",
            required_endpoint_fields.issubset(endpoint_fields),
        )
        review.gate(
            "spectrum_columns_exact",
            spectrum_fields == ["alpha", "rank", "m_eigenvalue", "a_contribution"],
        )
        review.gate(
            "h_fd_columns_exact",
            h_fd_fields
            == ["step", "analytic_derivative", "fd_derivative", "relative_error", "sign_agrees"],
        )

        numeric_endpoint_fields = [
            field for field in endpoint_fields if field != "endpoint_parameter_hash"
        ]
        endpoints_numeric_finite = all(
            math.isfinite(float(row[field]))
            for row in endpoint_rows
            for field in numeric_endpoint_fields
        )
        spectra_numeric_finite = all(
            math.isfinite(float(row[field])) for row in spectrum_rows for field in spectrum_fields
        )
        h_fd_numeric_finite = all(
            math.isfinite(float(row[field]))
            for row in h_fd_rows
            for field in h_fd_fields
            if field != "sign_agrees"
        )
        review.gate(
            "tables_numeric_finite",
            endpoints_numeric_finite and spectra_numeric_finite and h_fd_numeric_finite,
        )

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

        parameter_hashes_valid = all(
            _is_sha256(row["endpoint_parameter_hash"]) for row in endpoint_rows
        )
        parameter_hashes_unique = len(
            {row["endpoint_parameter_hash"] for row in endpoint_rows}
        ) == len(endpoint_rows)
        review.gate(
            "endpoint_parameter_hashes_well_formed_and_distinct",
            parameter_hashes_valid and parameter_hashes_unique,
        )
        review.gate(
            "endpoint_repeat_parameter_hashes_unchanged",
            all(int(float(row["repeat_parameter_hash_unchanged"])) == 1 for row in endpoint_rows),
        )

        repeat_fields = [
            field
            for field in endpoint_fields
            if field.startswith("repeat_") and field.endswith("error")
        ]
        continuous_repeat_fields = [
            field
            for field in repeat_fields
            if field
            not in {
                "repeat_count_lt_1e_4_abs_error",
                "repeat_count_lt_1e_2_abs_error",
                "repeat_count_lt_0p1_abs_error",
            }
        ]
        discrete_repeat_fields = sorted(set(repeat_fields) - set(continuous_repeat_fields))
        max_repeat_error = max(
            float(row[field]) for row in endpoint_rows for field in repeat_fields
        )
        review.gate(
            "endpoint_continuous_repeats_match",
            bool(continuous_repeat_fields)
            and max(
                float(row[field])
                for row in endpoint_rows
                for field in continuous_repeat_fields
            )
            <= 1e-10,
        )
        review.gate(
            "endpoint_discrete_repeats_match_exactly",
            bool(discrete_repeat_fields)
            and all(float(row[field]) == 0.0 for row in endpoint_rows for field in discrete_repeat_fields),
        )

        grouped_spectra: dict[float, list[dict[str, str]]] = {alpha: [] for alpha in ALPHAS}
        unknown_spectrum_alpha = False
        for row in spectrum_rows:
            alpha = float(row["alpha"])
            if alpha not in grouped_spectra:
                unknown_spectrum_alpha = True
                continue
            grouped_spectra[alpha].append(row)
        spectrum_grid_exact = not unknown_spectrum_alpha and len(spectrum_rows) == len(ALPHAS) * DIMENSION
        spectrum_sorted = True
        contribution_matches = True
        nonnegative = True
        spectrum_by_alpha: dict[float, np.ndarray] = {}
        max_contribution_error = 0.0
        for alpha in ALPHAS:
            rows = grouped_spectra[alpha]
            ranks = [int(float(row["rank"])) for row in rows]
            spectrum_grid_exact &= ranks == list(range(DIMENSION))
            eigenvalues = np.asarray([float(row["m_eigenvalue"]) for row in rows], dtype=np.float64)
            if len(eigenvalues) != DIMENSION:
                spectrum_grid_exact = False
                continue
            spectrum_by_alpha[alpha] = eigenvalues
            spectrum_sorted &= bool(np.all(np.diff(eigenvalues) >= 0.0))
            nonnegative &= bool(np.all(eigenvalues >= 0.0))
            stored_contribution = np.asarray(
                [float(row["a_contribution"]) for row in rows], dtype=np.float64
            )
            expected_contribution = np.square(eigenvalues - 1.0)
            error = float(np.max(np.abs(stored_contribution - expected_contribution)))
            max_contribution_error = max(max_contribution_error, error)
            contribution_matches &= bool(
                np.allclose(stored_contribution, expected_contribution, atol=1e-12, rtol=1e-10)
            )
        review.gate("full_512_spectrum_grid_exact", spectrum_grid_exact)
        review.gate("spectra_sorted_nonnegative", spectrum_sorted and nonnegative)
        review.gate("spectrum_a_contributions_recompute", contribution_matches)
        context["spectrum_by_alpha"] = spectrum_by_alpha

        max_metric_error = 0.0
        recomputed_by_alpha: dict[float, dict[str, float]] = {}
        all_spectrum_metrics_match = True
        dense_closure = True
        count_metrics_match = True
        if set(spectrum_by_alpha) != set(ALPHAS):
            all_spectrum_metrics_match = False
            dense_closure = False
            count_metrics_match = False
        else:
            for alpha in ALPHAS:
                row = endpoint_by_alpha[alpha]
                eigenvalues = spectrum_by_alpha[alpha]
                recomputed = _spectrum_metrics(eigenvalues)
                recomputed_by_alpha[alpha] = recomputed
                for field, expected in recomputed.items():
                    observed = float(row[field])
                    max_metric_error = max(max_metric_error, abs(observed - expected))
                    all_spectrum_metrics_match &= _close(observed, expected)
                expected_counts = {
                    "count_lt_1e_4": float(np.sum(eigenvalues < 1e-4)),
                    "count_lt_1e_2": float(np.sum(eigenvalues < 1e-2)),
                    "count_lt_0p1": float(np.sum(eigenvalues < 0.1)),
                }
                count_metrics_match &= all(
                    float(row[field]) == expected for field, expected in expected_counts.items()
                )
                exact_a = float(row["exact_a_per_dim"])
                trace_closure = float(row["a_trace_closure"])
                direct = float(row["a_direct_matrix"])
                direct_error = abs(direct - exact_a)
                trace_error = abs(trace_closure - exact_a)
                a_terms = (
                    float(row["a_constant_term"])
                    + float(row["a_linear_trace_term"])
                    + float(row["a_quartic_term"])
                )
                burg_terms = (
                    float(row["burg_trace_r_term"])
                    + float(row["burg_neg_logdet_r_term"])
                    - 1.0
                )
                dense_closure &= (
                    _close(exact_a, a_terms)
                    and _close(float(row["damped_full_burg_per_dim"]), burg_terms)
                    and _close(
                        float(row["burg_neg_logdet_r_term"]),
                        -float(row["logdet_r_per_dim"]),
                    )
                    and _close(float(row["a_direct_abs_error"]), direct_error, atol=1e-12)
                    and _close(float(row["a_trace_abs_error"]), trace_error, atol=1e-12)
                    and direct_error <= 1e-9
                    and trace_error <= 1e-9
                )
        review.gate("spectrum_derived_endpoint_metrics_match", all_spectrum_metrics_match)
        review.gate("spectrum_derived_counts_match", count_metrics_match)
        review.gate("exact_a_and_b_closure_passes", dense_closure)
        context["recomputed_by_alpha"] = recomputed_by_alpha

        base = endpoint_by_alpha[0.0]
        base_eigenvalues = spectrum_by_alpha[0.0]
        base_low = base_eigenvalues[base_eigenvalues < LOW_THRESHOLD]
        base_dead = base_eigenvalues[base_eigenvalues < DEAD_THRESHOLD]
        base_subspace_matches = (
            len(base_low) == EXPECTED_LOW_COUNT
            and len(base_dead) == EXPECTED_DEAD_COUNT
            and _close(float(base["frozen_low_energy"]), float(base_low.mean()))
            and _close(float(base["frozen_dead_energy"]), float(base_dead.mean()))
        )
        review.gate("base_frozen_low_and_dead_energies_recompute", base_subspace_matches)
        review.details["endpoint_spectrum_review"] = {
            "endpoint_rows": len(endpoint_rows),
            "spectrum_rows": len(spectrum_rows),
            "repeat_error_columns": len(repeat_fields),
            "max_repeat_error": max_repeat_error,
            "max_spectrum_contribution_absolute_error": max_contribution_error,
            "max_spectrum_metric_absolute_error": max_metric_error,
            "base_low_count": len(base_low),
            "base_dead_count": len(base_dead),
            "base_frozen_low_energy_from_spectrum": float(base_low.mean()),
            "base_frozen_dead_energy_from_spectrum": float(base_dead.mean()),
        }

    if not review.phase("tables", tables_phase):
        return 0 if _write_report(output, review, None) else 1

    def checkpoint_replay_phase() -> None:
        decision = context["decision"]
        frozen_dependencies = context["frozen_dependencies"]
        endpoint_by_alpha = context["endpoint_by_alpha"]
        spectrum_by_alpha = context["spectrum_by_alpha"]

        checkpoint_file_hash = _sha256_file(ITERATION3_CHECKPOINT)
        metadata, active_hash, parameter_count = _checkpoint_metadata_and_hash(
            ITERATION3_CHECKPOINT
        )
        active_fields, active_rows = _read_csv(ACTIVE_PARAMETERS)
        if "parameter" not in active_fields:
            raise ValueError("active_parameters.csv lacks parameter column")
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
            "iteration4_base_hash_matches_checkpoint",
            decision.get("base_parameter_hash") == active_hash
            and endpoint_by_alpha[0.0]["endpoint_parameter_hash"] == active_hash,
        )

        state_bank_fields, state_bank_rows = _read_csv(STATE_BANK)
        if not {"state_position", "source_weight_index"}.issubset(state_bank_fields):
            raise ValueError("state_bank.csv lacks identity columns")
        selected_state = [
            row for row in state_bank_rows if int(float(row["state_position"])) == STATE_POSITION
        ]
        review.gate(
            "state100_source_identity_matches",
            len(selected_state) == 1
            and int(float(selected_state[0]["source_weight_index"])) == SOURCE_WEIGHT_INDEX,
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
        state100_ranks = [int(float(row["rank"])) for row in state100_spectrum_rows]
        state100_spectrum = np.asarray(
            [float(row["m_eigenvalue"]) for row in state100_spectrum_rows], dtype=np.float64
        )
        review.gate(
            "iteration3_state100_spectrum_grid_exact",
            state100_ranks == list(range(DIMENSION))
            and len(state100_spectrum) == DIMENSION
            and bool(np.all(np.diff(state100_spectrum) >= 0.0)),
        )
        state100_contribution = np.asarray(
            [float(row["a_contribution"]) for row in state100_spectrum_rows], dtype=np.float64
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
        base_spectrum_error = float(
            np.max(np.abs(spectrum_by_alpha[0.0] - state100_spectrum))
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
        metric_errors = {
            field: abs(float(base_metrics[field]) - float(state100[field]))
            for field in replay_fields
        }
        endpoint_metric_errors = {
            field: abs(float(endpoint_by_alpha[0.0][field]) - float(state100[field]))
            for field in replay_fields
        }
        review.gate(
            "state100_metrics_replay_at_base",
            max([*metric_errors.values(), *endpoint_metric_errors.values()]) <= 1e-9,
        )
        review.gate(
            "producer_replay_error_records_recompute",
            _equivalent(decision.get("base_metric_replay_errors"), metric_errors)
            and _close(
                decision.get("base_spectrum_replay_error"),
                base_spectrum_error,
                atol=1e-15,
                rtol=1e-9,
            ),
        )
        state100_recomputed = _spectrum_metrics(state100_spectrum)
        review.gate(
            "state100_metrics_recompute_from_spectrum",
            all(_close(state100[field], value) for field, value in state100_recomputed.items()),
        )

        iteration3_decision = _load_json(ITERATION3_DECISION)
        iteration3_review = _load_json(ITERATION3_REVIEW)
        review.gate(
            "iteration3_inputs_were_validated",
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
            "state100_metric_absolute_errors": metric_errors,
            "state100_endpoint_metric_absolute_errors": endpoint_metric_errors,
            "state100_spectrum_max_absolute_error": base_spectrum_error,
        }

    if not review.phase("checkpoint_replay", checkpoint_replay_phase):
        return 0 if _write_report(output, review, None) else 1

    def pullback_fd_phase() -> None:
        decision = context["decision"]
        gradient = context["gradient"]
        endpoint_by_alpha = context["endpoint_by_alpha"]
        spectrum_by_alpha = context["spectrum_by_alpha"]
        h_fd_rows = context["h_fd_rows"]

        base_spectrum = spectrum_by_alpha[0.0]
        low_eigenvalues = base_spectrum[base_spectrum < LOW_THRESHOLD]
        dead_eigenvalues = base_spectrum[base_spectrum < DEAD_THRESHOLD]
        low_count = len(low_eigenvalues)
        k_norm_from_spectrum = 2.0 * math.sqrt(float(low_eigenvalues.sum())) / float(low_count)
        low_energy_from_spectrum = float(low_eigenvalues.mean())
        dead_energy_from_spectrum = float(dead_eigenvalues.mean())
        review.gate(
            "low_and_dead_counts_recompute",
            low_count == EXPECTED_LOW_COUNT
            and len(dead_eigenvalues) == EXPECTED_DEAD_COUNT
            and decision.get("low_count") == low_count
            and decision.get("dead_count") == len(dead_eigenvalues),
        )
        review.gate(
            "k_low_norm_recomputes_from_exact_formula",
            _close(k_norm_from_spectrum, gradient.get("h_cotangent_norm"), atol=1e-12)
            and _close(k_norm_from_spectrum, decision.get("h_cotangent_norm"), atol=1e-12),
        )
        review.gate(
            "base_frozen_energies_match_k_formula_inputs",
            _close(endpoint_by_alpha[0.0]["frozen_low_energy"], low_energy_from_spectrum)
            and _close(endpoint_by_alpha[0.0]["frozen_dead_energy"], dead_energy_from_spectrum),
        )

        fd_steps = [float(row["step"]) for row in h_fd_rows]
        h_fd_recomputed: list[dict[str, Any]] = []
        h_fd_passes = len(h_fd_rows) == len(H_FD_STEPS) and np.array_equal(
            np.asarray(fd_steps), np.asarray(H_FD_STEPS)
        )
        for row in h_fd_rows:
            analytic = float(row["analytic_derivative"])
            observed = float(row["fd_derivative"])
            relative = abs(observed - analytic) / max(abs(analytic), abs(observed), 1e-30)
            sign_agrees = analytic * observed > 0.0
            h_fd_passes &= (
                _close(analytic, k_norm_from_spectrum, atol=1e-12)
                and _close(float(row["relative_error"]), relative, atol=1e-15)
                and _bool_text(row["sign_agrees"]) == sign_agrees
                and sign_agrees
                and relative <= 1e-5
            )
            h_fd_recomputed.append(
                {
                    "step": float(row["step"]),
                    "analytic_derivative": analytic,
                    "fd_derivative": observed,
                    "relative_error": relative,
                    "sign_agrees": sign_agrees,
                }
            )
        review.gate("h_space_central_fd_recomputes_and_passes", h_fd_passes)

        gradient_norm = float(gradient["gradient_norm"])
        direction_norm = float(gradient["direction_norm"])
        analytic_parameter_derivative = float(gradient["analytic_low_energy_derivative"])
        transmission = gradient_norm / max(k_norm_from_spectrum, 1e-30)
        ideal_directional_derivative = TARGET_NORM * gradient_norm
        review.gate(
            "blocked_pullback_coverage_and_memory_pass",
            int(gradient.get("basis_count", -1)) == DIMENSION
            and int(gradient.get("block_size", -1)) == 64
            and int(gradient.get("block_count", -1)) == 8
            and int(gradient.get("unused_parameter_tensors_all_blocks", -1)) == 0
            and float(gradient.get("memory_growth_bytes", math.inf)) <= 64.0 * 1024.0 * 1024.0
            and gradient.get("memory_growth_gate_pass") is True,
        )
        review.gate(
            "pullback_gradient_and_direction_are_valid",
            math.isfinite(gradient_norm)
            and gradient_norm > 0.0
            and abs(direction_norm - TARGET_NORM) <= 1e-6
            and _close(direction_norm, decision.get("direction_norm"))
            and _close(gradient_norm, decision.get("gradient_norm")),
        )
        review.gate(
            "normalized_transmission_recomputes",
            _close(transmission, gradient.get("normalized_pullback_transmission"), atol=1e-10)
            and _close(transmission, decision.get("normalized_pullback_transmission"), atol=1e-10),
        )
        review.gate(
            "analytic_parameter_derivative_matches_normalized_descent",
            _close(
                analytic_parameter_derivative,
                ideal_directional_derivative,
                atol=1e-9,
                rtol=1e-6,
            )
            and _close(
                analytic_parameter_derivative,
                decision.get("analytic_low_energy_derivative"),
                atol=1e-12,
            ),
        )
        review.details["k_low_and_pullback_review"] = {
            "formula": "K_low = -(2/n_L) Q_L Q_L^T H",
            "recomputed_norm_identity": "||K_low||_F = (2/n_L) sqrt(sum(lambda_i, lambda_i<0.1))",
            "low_count": low_count,
            "dead_count": len(dead_eigenvalues),
            "low_eigenvalue_sum": float(low_eigenvalues.sum()),
            "low_energy_from_spectrum": low_energy_from_spectrum,
            "dead_energy_from_spectrum": dead_energy_from_spectrum,
            "k_low_norm_from_spectrum": k_norm_from_spectrum,
            "h_space_fd": h_fd_recomputed,
            "gradient_norm": gradient_norm,
            "target_direction_norm": TARGET_NORM,
            "observed_direction_norm": direction_norm,
            "normalized_pullback_transmission": transmission,
            "ideal_normalized_descent_derivative": ideal_directional_derivative,
            "stored_analytic_parameter_derivative": analytic_parameter_derivative,
            "full_matrix_formula_replayable": False,
        }
        review.limitations.append(
            "The exact entries/orientation of K_low cannot be reconstructed post-run because "
            "H and Q_L were not stored. The formula-implied norm, low/dead energies, signs, "
            "H-space FD records, and pullback scalar identities are independently recomputed."
        )

    if not review.phase("pullback_fd", pullback_fd_phase):
        return 0 if _write_report(output, review, None) else 1

    recomputed_outcome: str | None = None

    def outcome_phase() -> None:
        nonlocal recomputed_outcome
        decision = context["decision"]
        finalized = context["finalized"]
        endpoint_fields = context["endpoint_fields"]
        endpoint_rows = context["endpoint_rows"]
        endpoint_by_alpha = context["endpoint_by_alpha"]
        gradient = context["gradient"]
        run_log = context["run_log"]

        base = endpoint_by_alpha[0.0]
        positive_rows = [endpoint_by_alpha[alpha] for alpha in ALPHAS if alpha > 0.0]
        repeat_fields = [
            field
            for field in endpoint_fields
            if field.startswith("repeat_") and field.endswith("error")
        ]
        repeat_noise = max(float(base[field]) for field in repeat_fields)
        response_threshold = max(5.0 * repeat_noise, 1e-8)

        def positive_response(row: Mapping[str, str]) -> bool:
            return (
                float(row["frozen_low_energy"])
                > float(base["frozen_low_energy"]) + response_threshold
            )

        def joint_repair(row: Mapping[str, str]) -> bool:
            return (
                positive_response(row)
                and float(row["exact_a_per_dim"]) < float(base["exact_a_per_dim"]) - 1e-8
                and float(row["damped_full_burg_per_dim"])
                < float(base["damped_full_burg_per_dim"]) - 1e-8
                and float(row["m_max"]) <= float(base["m_max"]) + 1e-8
                and float(row["m_p50"]) >= float(base["m_p50"]) - 1e-10
                and float(row["count_lt_1e_4"]) <= float(base["count_lt_1e_4"])
                and float(row["count_lt_1e_2"]) <= float(base["count_lt_1e_2"])
            )

        positive_alphas = [float(row["alpha"]) for row in positive_rows if positive_response(row)]
        joint_alphas = [float(row["alpha"]) for row in positive_rows if joint_repair(row)]
        smallest = endpoint_by_alpha[1.0 / 256.0]
        smallest_conflict = (
            float(smallest["exact_a_per_dim"]) >= float(base["exact_a_per_dim"]) - 1e-8
            or float(smallest["damped_full_burg_per_dim"])
            >= float(base["damped_full_burg_per_dim"]) - 1e-8
            or float(smallest["m_max"]) > float(base["m_max"]) + 1e-8
            or float(smallest["m_p50"]) < float(base["m_p50"]) - 1e-10
            or float(smallest["count_lt_1e_4"]) > float(base["count_lt_1e_4"])
            or float(smallest["count_lt_1e_2"]) > float(base["count_lt_1e_2"])
        )

        plus = endpoint_by_alpha[1.0 / 256.0]
        minus = endpoint_by_alpha[-1.0 / 256.0]
        derivative_fields = (
            "frozen_low_energy",
            "exact_a_per_dim",
            "damped_full_burg_per_dim",
            "m_max",
            "m_p50",
        )
        central_derivatives = {
            field: (float(plus[field]) - float(minus[field])) / (2.0 / 256.0)
            for field in derivative_fields
        }
        analytic_derivative = float(gradient["analytic_low_energy_derivative"])
        fd_derivative = central_derivatives["frozen_low_energy"]
        fd_relative_error = abs(fd_derivative - analytic_derivative) / max(
            abs(fd_derivative), abs(analytic_derivative), 1e-30
        )
        parameter_fd_reliable = (
            fd_derivative > 0.0
            and analytic_derivative > 0.0
            and fd_relative_error <= 0.10
        )
        central_conflicts = [
            name
            for name, conflict in (
                ("exact_a", central_derivatives["exact_a_per_dim"] >= 0.0),
                ("full_b", central_derivatives["damped_full_burg_per_dim"] >= 0.0),
                ("m_max", central_derivatives["m_max"] > 0.0),
                ("m_p50", central_derivatives["m_p50"] < 0.0),
            )
            if conflict
        ]
        if not parameter_fd_reliable:
            recomputed_outcome = "ambiguous_nonlocal_parameter_fd"
        elif joint_alphas:
            recomputed_outcome = "accessible_joint_direction_supported"
        elif positive_alphas and positive_response(smallest) and central_conflicts:
            recomputed_outcome = "directional_tradeoff_observed"
        elif not positive_alphas:
            recomputed_outcome = "no_finite_response_unresolved"
        else:
            recomputed_outcome = "ambiguous"

        review.gate(
            "central_fd_derivatives_recompute",
            _equivalent(decision.get("central_fd_derivatives"), central_derivatives),
        )
        review.gate(
            "central_fd_reliability_recomputes",
            _close(decision.get("parameter_fd_low_energy_derivative"), fd_derivative)
            and _close(decision.get("parameter_fd_relative_error"), fd_relative_error)
            and decision.get("parameter_fd_reliable") is parameter_fd_reliable,
        )
        review.gate(
            "central_fd_conflicts_recompute",
            decision.get("central_fd_conflicts") == central_conflicts,
        )
        review.gate(
            "response_and_joint_counts_recompute",
            _close(decision.get("response_threshold"), response_threshold, atol=1e-15)
            and decision.get("positive_low_response_count") == len(positive_alphas)
            and decision.get("joint_repair_count") == len(joint_alphas)
            and decision.get("smallest_positive_alpha_conflict") is smallest_conflict,
        )
        review.gate(
            "corrected_protocol_outcome_recomputes",
            decision.get("outcome") == recomputed_outcome
            and finalized.get("outcome") == recomputed_outcome,
        )

        endpoint_hash_evidence = all(
            int(float(row["repeat_parameter_hash_unchanged"])) == 1 for row in endpoint_rows
        )
        restoration_evidence = (
            endpoint_by_alpha[0.0]["endpoint_parameter_hash"] == EXPECTED_ACTIVE_SHA256
            and endpoint_hash_evidence
            and decision.get("base_parameter_hash") == EXPECTED_ACTIVE_SHA256
            and decision.get("validity_gates", {}).get("parameters_restored") is True
            and "base restoration failed" not in run_log
            and run_log.count("[low-mode-i4] endpoint=") == len(ALPHAS)
        )
        review.gate("stored_restoration_evidence_is_consistent", restoration_evidence)
        review.details["outcome_review"] = {
            "base_repeat_noise": repeat_noise,
            "response_threshold": response_threshold,
            "positive_low_response_alphas": positive_alphas,
            "joint_repair_alphas": joint_alphas,
            "smallest_positive_alpha_conflict": smallest_conflict,
            "central_fd_derivatives": central_derivatives,
            "central_fd_conflicts": central_conflicts,
            "parameter_fd_relative_error": fd_relative_error,
            "parameter_fd_reliable": parameter_fd_reliable,
            "recomputed_outcome": recomputed_outcome,
            "producer_outcome": decision.get("outcome"),
        }
        review.details["restoration_review"] = {
            "base_checkpoint_active_sha256": EXPECTED_ACTIVE_SHA256,
            "alpha_zero_endpoint_sha256": endpoint_by_alpha[0.0]["endpoint_parameter_hash"],
            "all_repeat_hashes_unchanged": endpoint_hash_evidence,
            "producer_final_restoration_gate": decision.get("validity_gates", {}).get(
                "parameters_restored"
            ),
            "endpoint_log_records": run_log.count("[low-mode-i4] endpoint="),
            "independent_post_restore_hash_available": False,
        }
        review.limitations.append(
            "Final restoration is supported by the checkpoint/base/alpha-zero hash chain, "
            "all endpoint repeat hash no-op records, the complete endpoint log, and the producer "
            "restoration gate. A separate post-restore serialized hash was not stored, so the "
            "ephemeral final in-memory state cannot be independently rehashed post-run."
        )

    if not review.phase("outcome", outcome_phase):
        return 0 if _write_report(output, review, recomputed_outcome) else 1

    valid = _write_report(output, review, recomputed_outcome)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
