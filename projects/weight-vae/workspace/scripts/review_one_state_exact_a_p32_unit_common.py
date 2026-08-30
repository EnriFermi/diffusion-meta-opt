from __future__ import annotations

import os
import sys

# Direct path execution otherwise lets scripts/inspect shadow the stdlib module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _SCRIPT_DIR:
    sys.path.pop(0)
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import argparse
import hashlib
import io
import json
import math
import pickle
import struct
import time
import traceback
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_a_p32_unit_common_i3_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration3_p32_unit_common_production"
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_3_p32_unit_common/protocol.md"
PRE_RUN_REVIEW_PATH = OUTPUT_ROOT / "iteration_3_p32_unit_common/pre_run_review.md"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_3_frozen_dependency_manifest.json"
ACTIVE_PARAMETERS = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
)

PROPOSALS = 100
POOL_DRAWS = 8
PAIRS_PER_DRAW = 4
DIMENSION = 512
TARGET_NORM = 0.04892722657548397
LINE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
EPSILON = 1e-4
TAIL_START = 81
TAIL_REQUIRED = 16
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "1cf3bdeb383c64b36b3ca69a956457833d0aa0d8dd1f6696e3964033518053d4"
)
EXPECTED_ACCEPTED_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
P32_PROTOCOL_ID = "one_state_a_full_burg_p32_training_iteration5_v1"
A_ONLY_PROTOCOL_ID = "one_state_a_optimization_smoke_h2048_v1"

EXPECTED_MANIFEST_ARTIFACTS = {
    "acceptance_and_lower90.png",
    "decision.json",
    "exact_a_b_and_spectrum.png",
    "executed_source_snapshot.py",
    "final_checkpoint.pt",
    "final_replay.json",
    "frozen_dependency_manifest_snapshot.json",
    "line_endpoints.csv",
    "p32_draw_diagnostics.csv",
    "p32_pair_scalars.csv",
    "p32_seed_schedule_preflight.json",
    "pre_run_review_snapshot.md",
    "proposal_diagnostics.csv",
    "protocol_snapshot.md",
    "resolved_config.json",
    "state0_repeatability.csv",
    "state_metrics.csv",
    "state_spectra.csv",
    "tail_noncollapse_audit.csv",
}
ACCEPTANCE_COLUMNS = {
    "a": "exact_a_per_dim",
    "b": "damped_full_burg_per_dim",
    "m_max": "m_max",
    "m_p50": "m_p50",
    "m_lt_0p1_fraction": "m_lt_0p1_fraction",
}
REPEAT_ERROR_COLUMNS = (
    "repeat_a_abs_error",
    "repeat_b_abs_error",
    "repeat_m_max_abs_error",
    "repeat_m_p50_abs_error",
    "repeat_spectrum_max_abs_error",
)
ACCEPTANCE_FAILURES = {
    "exact_a_not_lower",
    "full_b_not_lower",
    "m_max_increased",
    "m_p50_decreased",
    "low_fraction_increased",
    "endpoint_repeat_mismatch",
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


def _equivalent(left: Any, right: Any, *, atol: float = 1e-10, rtol: float = 1e-9) -> bool:
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
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
    ) or (
        isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        if not (
            isinstance(left, Sequence)
            and not isinstance(left, (str, bytes))
            and isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
            and len(left) == len(right)
        ):
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


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _integer_grid(series: pd.Series, start: int, stop: int) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    expected = np.arange(start, stop + 1, dtype=np.float64)
    return bool(
        len(values) == len(expected)
        and np.isfinite(values).all()
        and np.array_equal(np.sort(values), expected)
    )


def _all_numeric_finite(frame: pd.DataFrame, *, excluded: set[str] | None = None) -> bool:
    selected = frame.drop(columns=list(excluded or set()), errors="ignore")
    numeric = selected.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def _binary_column(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return bool(values.notna().all() and values.isin([0, 1]).all())


def _failure_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    return [] if not text else text.split("|")


def _png_is_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 10_000:
        return False
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return False
    width, height = struct.unpack(">II", header[16:24])
    return width >= 640 and height >= 400


def _stable_uint63(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & (
        (1 << 63) - 1
    )


def _branch_seed(proposal: int, draw: int, pair: int, branch: int) -> int:
    if draw == 0:
        return _stable_uint63(A_ONLY_PROTOCOL_ID, "train", proposal, pair, branch)
    return _stable_uint63(P32_PROTOCOL_ID, "p32_extra", proposal, draw, pair, branch)


def _seed_schedule() -> list[dict[str, int]]:
    return [
        {
            "proposal": proposal,
            "draw": draw,
            "pair": pair,
            "seed_1": _branch_seed(proposal, draw, pair, 0),
            "seed_2": _branch_seed(proposal, draw, pair, 1),
        }
        for proposal in range(1, PROPOSALS + 1)
        for draw in range(POOL_DRAWS)
        for pair in range(PAIRS_PER_DRAW)
    ]


def _seed_schedule_hash(rows: list[dict[str, int]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _acceptance_tolerances(noise: Mapping[str, float]) -> dict[str, float]:
    return {
        "a": max(5.0 * float(noise["a"]), 1e-8),
        "b": max(5.0 * float(noise["b"]), 1e-8),
        "m_max": max(5.0 * float(noise["m_max"]), 1e-8),
        "m_p50": max(5.0 * float(noise["m_p50"]), 1e-10),
        "m_lt_0p1_fraction": float(noise["m_lt_0p1_fraction"]),
    }


def _acceptance_failures(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tolerances: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    if not (
        float(candidate["exact_a_per_dim"])
        < float(current["exact_a_per_dim"]) - float(tolerances["a"])
    ):
        failures.append("exact_a_not_lower")
    if not (
        float(candidate["damped_full_burg_per_dim"])
        < float(current["damped_full_burg_per_dim"]) - float(tolerances["b"])
    ):
        failures.append("full_b_not_lower")
    if not (
        float(candidate["m_max"])
        <= float(current["m_max"]) + float(tolerances["m_max"])
    ):
        failures.append("m_max_increased")
    if not (
        float(candidate["m_p50"])
        >= float(current["m_p50"]) - float(tolerances["m_p50"])
    ):
        failures.append("m_p50_decreased")
    if not (
        float(candidate["m_lt_0p1_fraction"])
        <= float(current["m_lt_0p1_fraction"])
        + float(tolerances["m_lt_0p1_fraction"])
    ):
        failures.append("low_fraction_increased")
    return failures


def _dense_closure(frame: pd.DataFrame) -> bool:
    required = {
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
    }
    _require_columns(frame, required, "dense metric frame")

    def values(column: str) -> np.ndarray:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)

    exact_a = values("exact_a_per_dim")
    a_terms = values("a_constant_term") + values("a_linear_trace_term") + values("a_quartic_term")
    burg = values("burg_trace_r_term") + values("burg_neg_logdet_r_term") - 1.0
    trace_r = (values("trace_m_per_dim") + EPSILON) / (1.0 + EPSILON)
    return bool(
        np.isfinite(frame[list(required)].to_numpy(dtype=np.float64)).all()
        and np.isclose(exact_a, a_terms, rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("damped_full_burg_per_dim"), burg, rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("burg_trace_r_term"), trace_r, rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("burg_neg_logdet_r_term"), -values("logdet_r_per_dim"), rtol=1e-9, atol=1e-10).all()
        and np.isclose(values("a_direct_abs_error"), np.abs(values("a_direct_matrix") - exact_a), rtol=1e-7, atol=1e-12).all()
        and np.isclose(values("a_trace_abs_error"), np.abs(values("a_trace_closure") - exact_a), rtol=1e-7, atol=1e-12).all()
        and (values("a_direct_abs_error") <= 1e-9).all()
        and (values("a_trace_abs_error") <= 1e-9).all()
    )


def _spectrum_metrics(eigenvalues: np.ndarray) -> dict[str, float]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    contribution = np.square(eigenvalues - 1.0)
    total_a = max(float(contribution.sum()), 1e-30)
    trace = float(eigenvalues.sum())
    square_sum = max(float(np.square(eigenvalues).sum()), 1e-30)
    r_eigenvalues = (eigenvalues + EPSILON) / (1.0 + EPSILON)
    g_eigenvalues = (1.0 - np.reciprocal(r_eigenvalues)) / (
        float(DIMENSION) * (1.0 + EPSILON)
    )
    quantiles = np.quantile(
        eigenvalues,
        [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
        method="linear",
    )
    trace_r = float(r_eigenvalues.mean())
    neg_logdet = float(-np.log(r_eigenvalues).mean())
    sorted_g = np.sort(g_eigenvalues)
    return {
        "exact_a_per_dim": float(contribution.mean()),
        "a_trace_closure": float(1.0 - 2.0 * eigenvalues.mean() + np.square(eigenvalues).mean()),
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
        "m_lt_1e_4_fraction": float(np.mean(eigenvalues < 1e-4)),
        "m_lt_0p01_fraction": float(np.mean(eigenvalues < 0.01)),
        "m_lt_0p1_fraction": float(np.mean(eigenvalues < 0.1)),
        "m_lt_0p5_fraction": float(np.mean(eigenvalues < 0.5)),
        "m_near_1_10pct_fraction": float(np.mean(np.abs(eigenvalues - 1.0) <= 0.1)),
        "m_gt_1_fraction": float(np.mean(eigenvalues > 1.0)),
        "m_gt_2_fraction": float(np.mean(eigenvalues > 2.0)),
        "a_from_m_lt_0p1_share": float(contribution[eigenvalues < 0.1].sum() / total_a),
        "a_from_m_gt_1_share": float(contribution[eigenvalues > 1.0].sum() / total_a),
        "burg_matrix_gradient_norm": float(np.linalg.norm(g_eigenvalues)),
        "burg_matrix_gradient_eig_min": float(g_eigenvalues.min()),
        "burg_matrix_gradient_eig_p50": float(sorted_g[(DIMENSION - 1) // 2]),
        "burg_matrix_gradient_eig_max": float(g_eigenvalues.max()),
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
        return np.frombuffer(logical, dtype=dtype).astype(dtype.newbyteorder("="), copy=True).tobytes()
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
            raise TypeError("checkpoint lacks active names or active_model_state")
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
            self.details[f"{name}_traceback"] = traceback.format_exc(limit=8)
            return False
        self.gates[f"{name}_completed"] = True
        return True


def _write_report(output: Path, review: Review, scientific_success: bool | None) -> bool:
    valid = bool(review.gates and all(review.gates.values()) and not review.errors)
    report = {
        "protocol_id": PROTOCOL_ID,
        "valid": valid,
        "scientific_success": scientific_success if valid else None,
        "gates": review.gates,
        "scientific_success_gates": review.details.pop("scientific_success_gates", {}),
        "recomputed": review.details,
        "errors": review.errors,
    }
    report_path = output / "independent_review.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(
        f"[exact-a-i3-review] report={report_path} valid={valid} "
        f"scientific_success={report['scientific_success']}",
        flush=True,
    )
    return valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
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
            f"[exact-a-i3-review] finalized production is unavailable: output={output} "
            f"staging_exists={staging.exists()}",
            flush=True,
        )
        return 1
    print(f"[exact-a-i3-review] start output={output}", flush=True)

    review = Review()
    context: dict[str, Any] = {}

    def metadata_phase() -> None:
        finalized = json.loads((output / "FINALIZED.json").read_text(encoding="utf-8"))
        resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
        if not all(isinstance(value, dict) for value in (finalized, resolved, decision, manifest)):
            raise TypeError("metadata JSON roots must be objects")
        context.update(
            finalized=finalized,
            resolved=resolved,
            decision=decision,
            manifest=manifest,
        )
        review.gate(
            "protocol_ids_match",
            {finalized.get("protocol_id"), resolved.get("protocol_id"), decision.get("protocol_id"), manifest.get("protocol_id")}
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
        review.gate(
            "all_manifest_artifact_hashes_match",
            isinstance(artifacts, dict)
            and set(artifacts) == EXPECTED_MANIFEST_ARTIFACTS
            and all(
                (output / name).is_file() and _sha256_file(output / name) == digest
                for name, digest in artifacts.items()
            ),
        )
        source_snapshot = output / "executed_source_snapshot.py"
        source_hash = _sha256_file(source_snapshot)
        normalized_source_hash = _normalized_runner_sha256(source_snapshot)
        review.gate(
            "executed_source_hashes_match",
            source_hash == resolved.get("source_sha256")
            and source_hash == manifest.get("source_sha256")
            and normalized_source_hash == resolved.get("normalized_source_sha256")
            and normalized_source_hash == manifest.get("normalized_source_sha256"),
        )
        protocol_snapshot = output / "protocol_snapshot.md"
        protocol_hash = _sha256_file(protocol_snapshot)
        review.gate(
            "protocol_snapshot_and_current_hash_match",
            protocol_hash == resolved.get("protocol_sha256")
            and PROTOCOL_PATH.is_file()
            and _sha256_file(PROTOCOL_PATH) == protocol_hash,
        )
        review.gate(
            "pre_run_review_snapshot_matches_current",
            PRE_RUN_REVIEW_PATH.is_file()
            and _sha256_file(output / "pre_run_review_snapshot.md")
            == _sha256_file(PRE_RUN_REVIEW_PATH),
        )
        dependency_snapshot = output / "frozen_dependency_manifest_snapshot.json"
        dependency_hash = _sha256_file(dependency_snapshot)
        frozen_dependencies = json.loads(dependency_snapshot.read_text(encoding="utf-8"))
        if not isinstance(frozen_dependencies, dict):
            raise TypeError("frozen dependency manifest must be an object")
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
        review.details["frozen_dependency_matches"] = dependency_results
        checkpoint_dependencies = [
            (relative, digest)
            for relative, digest in frozen_dependencies.items()
            if str(relative).endswith("/vae_checkpoint.pt")
        ]
        review.gate(
            "accepted_input_checkpoint_frozen_hash_matches",
            len(checkpoint_dependencies) == 1
            and checkpoint_dependencies[0][1] == EXPECTED_ACCEPTED_CHECKPOINT_SHA256
            and dependency_results[checkpoint_dependencies[0][0]],
        )
        review.gate(
            "resolved_protocol_constants_match",
            resolved.get("device") == "cuda:0"
            and resolved.get("source_weight_index") == 378
            and resolved.get("state_position") == 2
            and resolved.get("proposals") == PROPOSALS
            and resolved.get("pool_draws") == POOL_DRAWS
            and resolved.get("pairs_per_draw") == PAIRS_PER_DRAW
            and _close(resolved.get("target_norm"), TARGET_NORM, atol=1e-15, rtol=0.0)
            and _equivalent(resolved.get("line_alphas"), list(LINE_ALPHAS), atol=1e-15, rtol=0.0)
            and _close(resolved.get("epsilon"), EPSILON, atol=1e-15, rtol=0.0)
            and resolved.get("hessian_chunk_size") == 64,
        )
        review.gate(
            "run_log_exists_and_nonempty",
            (output / "run.log").is_file() and (output / "run.log").stat().st_size > 1_000,
        )
        run_log = (output / "run.log").read_text(encoding="utf-8", errors="replace")
        review.gate(
            "run_log_contains_completion",
            "[exact-a-i3] complete" in run_log
            and "[exact-a-i3] published output=" in run_log
            and "[exact-a-i3] FAILED" not in run_log,
        )
        review.gate(
            "required_plots_exist_and_are_readable",
            _png_is_readable(output / "exact_a_b_and_spectrum.png")
            and _png_is_readable(output / "acceptance_and_lower90.png"),
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

    if not review.phase("metadata", metadata_phase):
        return 0 if _write_report(output, review, None) else 1

    def table_phase() -> None:
        frames = {
            "state0": pd.read_csv(output / "state0_repeatability.csv"),
            "states": pd.read_csv(output / "state_metrics.csv"),
            "proposals": pd.read_csv(output / "proposal_diagnostics.csv"),
            "line": pd.read_csv(output / "line_endpoints.csv"),
            "pairs": pd.read_csv(output / "p32_pair_scalars.csv"),
            "draws": pd.read_csv(output / "p32_draw_diagnostics.csv"),
            "spectra": pd.read_csv(output / "state_spectra.csv"),
            "tail": pd.read_csv(output / "tail_noncollapse_audit.csv"),
        }
        context["frames"] = frames
        states = frames["states"]
        proposals = frames["proposals"]
        line = frames["line"]
        pairs = frames["pairs"]
        draws = frames["draws"]
        spectra = frames["spectra"]
        tail = frames["tail"]
        state0 = frames["state0"]

        _require_columns(state0, {"evaluation", *ACCEPTANCE_COLUMNS.values()}, "state0_repeatability.csv")
        _require_columns(
            states,
            {
                "proposal",
                "accepted",
                "accepted_alpha",
                "accepted_radius",
                "rho_lower90",
                *ACCEPTANCE_COLUMNS.values(),
            },
            "state_metrics.csv",
        )
        _require_columns(
            proposals,
            {
                "proposal",
                "accepted",
                "selected_alpha",
                "accepted_radius",
                "attempted_endpoint_count",
                "gradient_a_norm",
                "gradient_b_norm",
                "gradient_ab_dot",
                "gradient_ab_cosine",
                "unit_common_source_norm",
                "source_valid",
                "cancellation_gate_pass",
                "base_parameter_hash",
                "final_parameter_hash",
                "failure_counts",
                "cumulative_accepted_count",
                "cumulative_path_radius",
            },
            "proposal_diagnostics.csv",
        )
        _require_columns(
            line,
            {
                "proposal",
                "alpha",
                "passes",
                "failures",
                "repeat_checked",
                "delta_a",
                "delta_b",
                "delta_m_max",
                "delta_m_p50",
                "delta_m_lt_0p1_fraction",
                *REPEAT_ERROR_COLUMNS,
                *ACCEPTANCE_COLUMNS.values(),
            },
            "line_endpoints.csv",
        )
        _require_columns(
            pairs,
            {
                "proposal",
                "draw",
                "pair",
                "seed_1",
                "seed_2",
                "a_loss",
                "b_pseudo_loss",
            },
            "p32_pair_scalars.csv",
        )
        _require_columns(
            draws,
            {"proposal", "draw", "mean_a", "mean_b", "unused_parameter_tensors"},
            "p32_draw_diagnostics.csv",
        )
        _require_columns(spectra, {"proposal", "rank", "m_eigenvalue", "a_contribution"}, "state_spectra.csv")
        _require_columns(tail, {"proposal", "accepted", "rho_lower90", "core_noncollapse_pass"}, "tail_noncollapse_audit.csv")

        review.gate(
            "state0_rows_complete_unique",
            len(state0) == 2
            and state0["evaluation"].is_unique
            and set(state0["evaluation"].astype(str)) == {"primary", "repeat"},
        )
        review.gate(
            "state_keys_0_through_100_complete_unique",
            len(states) == 101
            and states["proposal"].is_unique
            and _integer_grid(states["proposal"], 0, 100),
        )
        review.gate(
            "proposal_keys_1_through_100_complete_unique",
            len(proposals) == 100
            and proposals["proposal"].is_unique
            and _integer_grid(proposals["proposal"], 1, 100),
        )
        review.gate(
            "pair_key_grid_100x8x4_complete_unique",
            len(pairs) == 3200
            and not pairs.duplicated(["proposal", "draw", "pair"]).any()
            and set(map(tuple, pairs[["proposal", "draw", "pair"]].astype(int).to_numpy()))
            == {
                (proposal, draw, pair)
                for proposal in range(1, 101)
                for draw in range(8)
                for pair in range(4)
            },
        )
        review.gate(
            "draw_key_grid_100x8_complete_unique",
            len(draws) == 800
            and not draws.duplicated(["proposal", "draw"]).any()
            and set(map(tuple, draws[["proposal", "draw"]].astype(int).to_numpy()))
            == {(proposal, draw) for proposal in range(1, 101) for draw in range(8)},
        )
        pooled_mean_checks: list[bool] = []
        indexed_draws = draws.set_index(["proposal", "draw"])
        for (proposal, draw), group in pairs.groupby(["proposal", "draw"]):
            row = indexed_draws.loc[(proposal, draw)]
            a_values = group["a_loss"].to_numpy(dtype=np.float64)
            b_values = group["b_pseudo_loss"].to_numpy(dtype=np.float64)
            a_roundoff = 8.0 * np.finfo(np.float32).eps * max(1.0, float(np.abs(a_values).max()))
            b_roundoff = 8.0 * np.finfo(np.float32).eps * max(1.0, float(np.abs(b_values).max()))
            pooled_mean_checks.extend(
                [
                    _close(
                        row["mean_a"],
                        a_values.mean(),
                        atol=a_roundoff,
                        rtol=0.0,
                    ),
                    _close(
                        row["mean_b"],
                        b_values.mean(),
                        atol=b_roundoff,
                        rtol=0.0,
                    ),
                ]
            )
        review.gate("p32_draw_means_recompute_from_pair_rows", all(pooled_mean_checks))
        review.gate(
            "spectrum_key_grid_101x512_complete_unique",
            len(spectra) == 101 * DIMENSION
            and not spectra.duplicated(["proposal", "rank"]).any()
            and all(
                len(group) == DIMENSION
                and _integer_grid(group["rank"], 0, DIMENSION - 1)
                for _, group in spectra.groupby("proposal")
            )
            and set(spectra["proposal"].astype(int)) == set(range(101)),
        )
        review.gate(
            "tail_keys_81_through_100_complete_unique",
            len(tail) == 20
            and tail["proposal"].is_unique
            and _integer_grid(tail["proposal"], TAIL_START, PROPOSALS),
        )
        review.gate(
            "binary_flag_columns_valid",
            _binary_column(states["accepted"])
            and _binary_column(proposals["accepted"])
            and _binary_column(proposals["source_valid"])
            and _binary_column(proposals["cancellation_gate_pass"])
            and _binary_column(line["passes"])
            and _binary_column(line["repeat_checked"])
            and _binary_column(tail["accepted"])
            and _binary_column(tail["core_noncollapse_pass"]),
        )
        review.gate(
            "line_keys_are_unique_and_belong_to_100_proposals",
            not line.duplicated(["proposal", "alpha"]).any()
            and bool(line["proposal"].between(1, 100).all()),
        )
        review.gate(
            "all_required_table_values_finite",
            _all_numeric_finite(state0)
            and _all_numeric_finite(states, excluded={"rho_lower90"})
            and _all_numeric_finite(proposals)
            and _all_numeric_finite(line)
            and _all_numeric_finite(pairs)
            and _all_numeric_finite(draws)
            and _all_numeric_finite(spectra)
            and _all_numeric_finite(tail),
        )

    if not review.phase("tables", table_phase):
        return 0 if _write_report(output, review, None) else 1

    frames: dict[str, pd.DataFrame] = context["frames"]
    decision: dict[str, Any] = context["decision"]
    finalized: dict[str, Any] = context["finalized"]
    recomputed_validity: dict[str, bool] = {}

    def seed_phase() -> None:
        pairs = frames["pairs"].sort_values(["proposal", "draw", "pair"])
        expected = _seed_schedule()
        actual = [
            {
                "proposal": int(row.proposal),
                "draw": int(row.draw),
                "pair": int(row.pair),
                "seed_1": int(row.seed_1),
                "seed_2": int(row.seed_2),
            }
            for row in pairs.itertuples(index=False)
        ]
        seeds = [row[key] for row in actual for key in ("seed_1", "seed_2")]
        schedule_hash = _seed_schedule_hash(expected)
        preflight = json.loads(
            (output / "p32_seed_schedule_preflight.json").read_text(encoding="utf-8")
        )
        seed_pass = bool(
            actual == expected
            and len(seeds) == 6400
            and len(set(seeds)) == 6400
            and preflight.get("row_count") == 3200
            and preflight.get("branch_seed_count") == 6400
            and preflight.get("unique_branch_seed_count") == 6400
            and preflight.get("draw0_matches_canonical_generators") is True
            and preflight.get("schedule_sha256") == schedule_hash
            and preflight.get("valid") is True
        )
        review.gate("seed_schedule_and_6400_branch_seeds_recompute", seed_pass)
        recomputed_validity["seed_schedule_valid"] = seed_pass
        recomputed_validity["actual_pair_key_coverage"] = review.gates[
            "pair_key_grid_100x8x4_complete_unique"
        ]
        recomputed_validity["actual_branch_seeds_unique"] = len(set(seeds)) == 6400
        review.details["seed_schedule_sha256"] = schedule_hash

    review.phase("seed_schedule", seed_phase)

    def repeatability_and_closure_phase() -> None:
        state0 = frames["state0"]
        states = frames["states"].sort_values("proposal").reset_index(drop=True)
        line = frames["line"]
        primary = state0.loc[state0["evaluation"].eq("primary")].iloc[0]
        repeat = state0.loc[state0["evaluation"].eq("repeat")].iloc[0]
        base_noise = {
            key: abs(float(primary[column]) - float(repeat[column]))
            for key, column in ACCEPTANCE_COLUMNS.items()
        }
        base_noise["spectrum_max_abs"] = float(decision["base_noise"]["spectrum_max_abs"])
        tolerances = _acceptance_tolerances(base_noise)
        context["tolerances"] = tolerances
        review.gate(
            "state0_primary_matches_state_zero",
            all(
                column in primary.index
                and _close(primary[column], states.iloc[0][column], atol=1e-10, rtol=1e-9)
                for column in states.columns
                if column in primary.index and column not in {"proposal", "accepted", "accepted_alpha", "accepted_radius", "rho_lower90"}
            ),
        )
        review.gate(
            "base_noise_and_tolerances_recompute",
            _equivalent(base_noise, decision["base_noise"], atol=1e-12, rtol=1e-9)
            and _equivalent(tolerances, decision["acceptance_tolerances"], atol=1e-12, rtol=1e-9),
        )
        state0_repeatable = max(base_noise.values()) <= 1e-10
        review.gate("state0_acceptance_metrics_repeatable", state0_repeatable)
        recomputed_validity["state0_repeatable"] = state0_repeatable
        direct_closure = bool(
            states["a_direct_abs_error"].le(1e-9).all()
            and line["a_direct_abs_error"].le(1e-9).all()
            and state0["a_direct_abs_error"].le(1e-9).all()
        )
        trace_closure = bool(
            states["a_trace_abs_error"].le(1e-9).all()
            and line["a_trace_abs_error"].le(1e-9).all()
            and state0["a_trace_abs_error"].le(1e-9).all()
        )
        review.gate(
            "exact_a_and_b_algebra_closes",
            direct_closure
            and trace_closure
            and _dense_closure(states)
            and _dense_closure(line)
            and _dense_closure(state0),
        )
        recomputed_validity["exact_a_direct_closure"] = direct_closure
        recomputed_validity["exact_a_trace_closure"] = trace_closure
        review.details["base_noise"] = base_noise
        review.details["acceptance_tolerances"] = tolerances

    review.phase("repeatability_and_closure", repeatability_and_closure_phase)

    def spectra_phase() -> None:
        states = frames["states"].sort_values("proposal").reset_index(drop=True)
        spectra = frames["spectra"]
        spectrum_checks: list[bool] = []
        rho_values: dict[int, float | None] = {}
        low90_values: dict[int, float] = {}
        state_eigenvalues: dict[int, np.ndarray] = {}
        max_metric_error = 0.0
        initial_a: float | None = None
        initial_low90: float | None = None
        started = time.perf_counter()
        for proposal in range(101):
            group = spectra.loc[spectra["proposal"].eq(proposal)].sort_values("rank")
            eigenvalues = group["m_eigenvalue"].to_numpy(dtype=np.float64)
            contributions = group["a_contribution"].to_numpy(dtype=np.float64)
            state_eigenvalues[proposal] = eigenvalues
            spectrum_checks.extend(
                [
                    len(eigenvalues) == DIMENSION,
                    np.isfinite(eigenvalues).all(),
                    (np.diff(eigenvalues) >= -1e-12).all(),
                    (eigenvalues >= -1e-12).all(),
                    np.isclose(contributions, np.square(eigenvalues - 1.0), rtol=1e-10, atol=1e-12).all(),
                ]
            )
            recomputed = _spectrum_metrics(eigenvalues)
            state = states.iloc[proposal]
            for field, expected in recomputed.items():
                if field not in state.index:
                    spectrum_checks.append(False)
                    continue
                actual = float(state[field])
                max_metric_error = max(max_metric_error, abs(actual - expected))
                spectrum_checks.append(_close(actual, expected, atol=2e-9, rtol=2e-8))
            low90 = recomputed["a_low90_abs_per_dim"]
            low90_values[proposal] = low90
            if proposal == 0:
                initial_a = recomputed["exact_a_per_dim"]
                initial_low90 = low90
            assert initial_a is not None and initial_low90 is not None
            denominator = initial_a - recomputed["exact_a_per_dim"]
            rho = None if denominator <= 0.0 else (initial_low90 - low90) / denominator
            rho_values[proposal] = rho
            stored_rho = float(state["rho_lower90"])
            spectrum_checks.append(
                math.isnan(stored_rho) if rho is None else _close(stored_rho, rho, atol=2e-9, rtol=2e-8)
            )
            if not args.quiet and proposal % 10 == 0:
                print(
                    f"[exact-a-i3-review] stage=spectra state={proposal}/100 "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        context["state_eigenvalues"] = state_eigenvalues
        context["rho_values"] = rho_values
        context["low90_values"] = low90_values
        review.gate("all_101x512_spectra_recompute_state_metrics", all(spectrum_checks))
        recomputed_validity["spectrum_rows_51712"] = review.gates[
            "spectrum_key_grid_101x512_complete_unique"
        ]
        review.details["spectrum_review"] = {
            "states": 101,
            "rows": 101 * DIMENSION,
            "max_metric_absolute_error": max_metric_error,
            "elapsed_sec": time.perf_counter() - started,
        }

    review.phase("spectra", spectra_phase)

    def line_and_transition_phase() -> None:
        states = frames["states"].sort_values("proposal").reset_index(drop=True)
        proposals = frames["proposals"].sort_values("proposal").reset_index(drop=True)
        line = frames["line"]
        eigenvalues: dict[int, np.ndarray] = context["state_eigenvalues"]
        tolerances: dict[str, float] = context["tolerances"]
        line_grid_checks: list[bool] = []
        line_semantic_checks: list[bool] = []
        selection_checks: list[bool] = []
        transition_checks: list[bool] = []
        repeat_checks: list[bool] = []
        hash_chain_checks: list[bool] = []
        selected_alphas: dict[str, float] = {}
        accepted_count = 0
        cumulative_radius = 0.0
        failure_count_checks: list[bool] = []
        allowed_alphas = np.asarray(LINE_ALPHAS, dtype=np.float64)
        state0 = states.iloc[0]
        transition_checks.extend(
            [
                int(state0["accepted"]) == 1,
                _close(state0["accepted_alpha"], 0.0),
                _close(state0["accepted_radius"], 0.0),
            ]
        )
        for proposal_number in range(1, 101):
            previous = states.iloc[proposal_number - 1]
            current = states.iloc[proposal_number]
            diagnostic = proposals.iloc[proposal_number - 1]
            group = line.loc[line["proposal"].eq(proposal_number)]
            attempted = int(diagnostic["attempted_endpoint_count"])
            expected_alphas = LINE_ALPHAS[:attempted]
            actual_alphas = group["alpha"].to_numpy(dtype=np.float64)
            line_grid_checks.extend(
                [
                    0 <= attempted <= len(LINE_ALPHAS),
                    len(group) == attempted,
                    not group.duplicated(["alpha"]).any(),
                    len(actual_alphas) == len(expected_alphas)
                    and np.allclose(actual_alphas, np.asarray(expected_alphas), rtol=0.0, atol=1e-12),
                    np.isclose(actual_alphas[:, None], allowed_alphas[None, :], rtol=0.0, atol=1e-12).any(axis=1).all()
                    if len(actual_alphas)
                    else True,
                ]
            )
            failure_counts: dict[str, int] = {}
            independently_passing: list[float] = []
            for _, endpoint in group.iterrows():
                base_failures = _acceptance_failures(previous, endpoint, tolerances)
                recorded_failures = _failure_list(endpoint["failures"])
                for failure in recorded_failures:
                    failure_counts[failure] = failure_counts.get(failure, 0) + 1
                repeat_checked = int(endpoint["repeat_checked"])
                repeat_errors = [float(endpoint[column]) for column in REPEAT_ERROR_COLUMNS]
                repeat_fidelity = max(repeat_errors) <= 1e-10
                recorded_pass = int(endpoint["passes"]) == 1
                line_semantic_checks.extend(
                    [
                        set(recorded_failures).issubset(ACCEPTANCE_FAILURES),
                        recorded_pass == (len(recorded_failures) == 0),
                        _close(endpoint["delta_a"], float(endpoint["exact_a_per_dim"]) - float(previous["exact_a_per_dim"]), atol=1e-10),
                        _close(endpoint["delta_b"], float(endpoint["damped_full_burg_per_dim"]) - float(previous["damped_full_burg_per_dim"]), atol=1e-10),
                        _close(endpoint["delta_m_max"], float(endpoint["m_max"]) - float(previous["m_max"]), atol=1e-10),
                        _close(endpoint["delta_m_p50"], float(endpoint["m_p50"]) - float(previous["m_p50"]), atol=1e-10),
                        _close(endpoint["delta_m_lt_0p1_fraction"], float(endpoint["m_lt_0p1_fraction"]) - float(previous["m_lt_0p1_fraction"]), atol=1e-12),
                    ]
                )
                if base_failures:
                    line_semantic_checks.extend(
                        [
                            repeat_checked == 0,
                            max(repeat_errors) == 0.0,
                            recorded_failures == base_failures,
                            not recorded_pass,
                        ]
                    )
                else:
                    line_semantic_checks.append(repeat_checked == 1)
                    if recorded_pass:
                        line_semantic_checks.append(repeat_fidelity)
                    elif not repeat_fidelity:
                        line_semantic_checks.append("endpoint_repeat_mismatch" in recorded_failures)
                    else:
                        line_semantic_checks.append(bool(recorded_failures))
                independent_pass = bool(
                    not base_failures
                    and repeat_checked == 1
                    and repeat_fidelity
                    and not recorded_failures
                )
                line_semantic_checks.append(recorded_pass == independent_pass)
                if independent_pass:
                    independently_passing.append(float(endpoint["alpha"]))

            try:
                reported_failure_counts = json.loads(str(diagnostic["failure_counts"]))
            except json.JSONDecodeError:
                reported_failure_counts = None
            failure_count_checks.append(reported_failure_counts == failure_counts)
            accepted = int(diagnostic["accepted"]) == 1
            accepted_count += int(accepted)
            selected_alpha = max(independently_passing) if independently_passing else 0.0
            selected_alphas[str(proposal_number)] = selected_alpha
            cumulative_radius += selected_alpha * TARGET_NORM if accepted else 0.0
            selection_checks.extend(
                [
                    accepted == bool(independently_passing),
                    int(current["accepted"]) == int(accepted),
                    _close(diagnostic["selected_alpha"], selected_alpha, atol=1e-12, rtol=0.0),
                    _close(current["accepted_alpha"], selected_alpha, atol=1e-12, rtol=0.0),
                    _close(diagnostic["accepted_radius"], selected_alpha * TARGET_NORM if accepted else 0.0, atol=1e-12),
                    _close(current["accepted_radius"], selected_alpha * TARGET_NORM if accepted else 0.0, atol=1e-12),
                    int(diagnostic["cumulative_accepted_count"]) == accepted_count,
                    _close(diagnostic["cumulative_path_radius"], cumulative_radius, atol=1e-10),
                    len(independently_passing) <= 1,
                    not independently_passing
                    or (
                        len(group) > 0
                        and _close(group.iloc[-1]["alpha"], selected_alpha, atol=1e-12, rtol=0.0)
                    ),
                ]
            )
            if accepted:
                selected = group.loc[group["passes"].eq(1)].iloc[0]
                repeat_checks.extend(
                    [
                        int(selected["repeat_checked"]) == 1,
                        max(float(selected[column]) for column in REPEAT_ERROR_COLUMNS) <= 1e-10,
                        not _failure_list(selected["failures"]),
                        abs(float(current["exact_a_per_dim"]) - float(selected["exact_a_per_dim"]))
                        <= float(selected["repeat_a_abs_error"]) + 1e-12,
                        abs(float(current["damped_full_burg_per_dim"]) - float(selected["damped_full_burg_per_dim"]))
                        <= float(selected["repeat_b_abs_error"]) + 1e-12,
                        abs(float(current["m_max"]) - float(selected["m_max"]))
                        <= float(selected["repeat_m_max_abs_error"]) + 1e-12,
                        abs(float(current["m_p50"]) - float(selected["m_p50"]))
                        <= float(selected["repeat_m_p50_abs_error"]) + 1e-12,
                    ]
                )
                transition_checks.extend(
                    [
                        float(current["exact_a_per_dim"])
                        < float(previous["exact_a_per_dim"]) - tolerances["a"],
                        float(current["damped_full_burg_per_dim"])
                        < float(previous["damped_full_burg_per_dim"]) - tolerances["b"],
                        float(current["m_max"])
                        <= float(previous["m_max"]) + tolerances["m_max"],
                        float(current["m_p50"])
                        >= float(previous["m_p50"]) - tolerances["m_p50"],
                        float(current["m_lt_0p1_fraction"])
                        <= float(previous["m_lt_0p1_fraction"])
                        + tolerances["m_lt_0p1_fraction"],
                        str(diagnostic["base_parameter_hash"])
                        != str(diagnostic["final_parameter_hash"]),
                    ]
                )
            else:
                repeat_checks.append(not independently_passing)
                state_metric_columns = [
                    column
                    for column in states.columns
                    if column
                    not in {
                        "proposal",
                        "accepted",
                        "accepted_alpha",
                        "accepted_radius",
                        "rho_lower90",
                    }
                ]
                transition_checks.extend(
                    [
                        all(
                            _close(current[column], previous[column], atol=0.0, rtol=0.0)
                            for column in state_metric_columns
                        ),
                        np.array_equal(eigenvalues[proposal_number], eigenvalues[proposal_number - 1]),
                    ]
                )
            hash_chain_checks.extend(
                [
                    _is_sha256(str(diagnostic["base_parameter_hash"])),
                    _is_sha256(str(diagnostic["final_parameter_hash"])),
                    str(diagnostic["base_parameter_hash"])
                    == (
                        EXPECTED_INITIAL_PARAMETER_SHA256
                        if proposal_number == 1
                        else str(proposals.iloc[proposal_number - 2]["final_parameter_hash"])
                    ),
                    accepted
                    or str(diagnostic["base_parameter_hash"])
                    == str(diagnostic["final_parameter_hash"]),
                ]
            )
        review.gate("line_endpoint_keys_are_descending_evaluated_prefixes", all(line_grid_checks))
        review.gate("line_acceptance_guards_and_failures_recompute", all(line_semantic_checks))
        review.gate("largest_actually_evaluated_passing_alpha_recomputes", all(selection_checks))
        review.gate("accepted_endpoint_repeats_are_faithful", all(repeat_checks))
        review.gate("state_transitions_are_monotonic_or_exact_noops", all(transition_checks))
        review.gate("proposal_failure_counts_recompute", all(failure_count_checks))
        review.gate("parameter_hash_chain_and_rejected_noops_recompute", all(hash_chain_checks))
        source_norm_expected = np.sqrt(
            np.maximum(
                0.0,
                2.0 + 2.0 * proposals["gradient_ab_cosine"].to_numpy(dtype=np.float64),
            )
        )
        cancellation_expected = (
            proposals["unit_common_source_norm"].to_numpy(dtype=np.float64) >= 0.5
        )
        source_pass = bool(
            proposals["source_valid"].eq(1).all()
            and proposals["gradient_a_norm"].gt(0.0).all()
            and proposals["gradient_b_norm"].gt(0.0).all()
            and proposals["gradient_ab_cosine"].between(-1.0, 1.0).all()
            and np.isclose(
                proposals["unit_common_source_norm"].to_numpy(dtype=np.float64),
                source_norm_expected,
                atol=5e-7,
                rtol=5e-6,
            ).all()
            and np.array_equal(
                proposals["cancellation_gate_pass"].to_numpy(dtype=np.int64),
                cancellation_expected.astype(np.int64),
            )
        )
        cancellation_noops = bool(
            (
                proposals.loc[
                    proposals["cancellation_gate_pass"].eq(0),
                    ["accepted", "attempted_endpoint_count"],
                ]
                .eq(0)
                .all()
            ).all()
        )
        review.gate("p32_sources_and_cancellation_decisions_recompute", source_pass)
        review.gate("cancellation_rejections_are_counted_noops", cancellation_noops)
        recomputed_validity["all_component_sources_valid"] = bool(proposals["source_valid"].eq(1).all())
        recomputed_validity["cancellation_rejections_are_counted_noops"] = cancellation_noops
        recomputed_validity["rejections_are_exact_noops"] = all(hash_chain_checks) and bool(
            (
                proposals.loc[proposals["accepted"].eq(0), "base_parameter_hash"]
                == proposals.loc[proposals["accepted"].eq(0), "final_parameter_hash"]
            ).all()
        )
        recomputed_validity["accepted_endpoints_pass_all_guards"] = bool(
            all(repeat_checks)
            and int(line["passes"].sum()) == accepted_count
            and len(line.loc[line["passes"].eq(1)]) == accepted_count
        )
        context["accepted_count"] = accepted_count
        context["cumulative_radius"] = cumulative_radius
        review.details["selected_alphas_by_proposal"] = selected_alphas

    review.phase("line_and_transitions", line_and_transition_phase)

    def checkpoint_phase() -> None:
        proposals = frames["proposals"].sort_values("proposal").reset_index(drop=True)
        checkpoint_path = output / "final_checkpoint.pt"
        metadata, parameter_hash, parameter_count = _checkpoint_metadata_and_hash(checkpoint_path)
        active_frame = pd.read_csv(ACTIVE_PARAMETERS)
        _require_columns(active_frame, {"parameter"}, str(ACTIVE_PARAMETERS))
        expected_active_names = sorted(set(active_frame["parameter"].astype(str)))
        checkpoint_names = metadata.get("active_names")
        checkpoint_pass = bool(
            metadata.get("protocol_id") == PROTOCOL_ID
            and metadata.get("source_weight_index") == 378
            and metadata.get("z_sha256") == EXPECTED_Z_SHA256
            and checkpoint_names == expected_active_names
            and parameter_count == 11_685_120
            and _is_sha256(metadata.get("final_parameter_hash"))
            and metadata.get("final_parameter_hash") == parameter_hash
            and parameter_hash == str(proposals.iloc[-1]["final_parameter_hash"])
        )
        review.gate("final_checkpoint_named_tensor_hash_recomputes", checkpoint_pass)
        context["checkpoint_z_matches"] = metadata.get("z_sha256") == EXPECTED_Z_SHA256
        context["checkpoint_active_parameter_count_matches"] = parameter_count == 11_685_120
        final_replay = json.loads((output / "final_replay.json").read_text(encoding="utf-8"))
        replay_pass = bool(
            isinstance(final_replay, dict)
            and set(final_replay) == {"a", "b", "m_max", "m_p50", "spectrum_max_abs"}
            and all(isinstance(value, Real) and math.isfinite(float(value)) for value in final_replay.values())
            and max(float(value) for value in final_replay.values()) <= 1e-10
        )
        review.gate("final_dense_replay_errors_within_tolerance", replay_pass)
        recomputed_validity["final_replay_matches"] = replay_pass
        review.details["final_checkpoint"] = {
            "file_sha256": _sha256_file(checkpoint_path),
            "parameter_sha256": parameter_hash,
            "parameter_count": parameter_count,
            "active_tensor_count": len(checkpoint_names) if isinstance(checkpoint_names, list) else None,
        }

    review.phase("final_checkpoint", checkpoint_phase)

    scientific_success: bool | None = None

    def tail_and_decision_phase() -> None:
        nonlocal scientific_success
        states = frames["states"].sort_values("proposal").reset_index(drop=True)
        proposals = frames["proposals"].sort_values("proposal").reset_index(drop=True)
        tail_file = frames["tail"].sort_values("proposal").reset_index(drop=True)
        tolerances: dict[str, float] = context["tolerances"]
        eigenvalues: dict[int, np.ndarray] = context["state_eigenvalues"]
        rho_values: dict[int, float | None] = context["rho_values"]
        tail_passes: list[bool] = []
        tail_rows_match: list[bool] = []
        for proposal in range(TAIL_START, PROPOSALS + 1):
            state = states.iloc[proposal]
            stored_tail = tail_file.iloc[proposal - TAIL_START]
            spectrum = eigenvalues[proposal]
            rho = rho_values[proposal]
            core_pass = bool(
                float(spectrum.mean()) >= 0.0456087
                and int(np.sum(spectrum < 1e-4)) <= 51
                and int(np.sum(spectrum < 0.01)) <= 256
                and rho is not None
                and rho >= 0.25
            )
            tail_passes.append(core_pass)
            tail_rows_match.extend(
                [
                    int(stored_tail["proposal"]) == proposal,
                    int(stored_tail["accepted"]) == int(state["accepted"]),
                    bool(stored_tail["core_noncollapse_pass"]) == core_pass,
                    rho is not None
                    and _close(stored_tail["rho_lower90"], rho, atol=2e-9, rtol=2e-8),
                    all(
                        _close(stored_tail[column], state[column], atol=1e-12, rtol=1e-10)
                        for column in tail_file.columns
                        if column in state.index
                        and column not in {"proposal", "accepted", "rho_lower90"}
                        and isinstance(stored_tail[column], Real)
                    ),
                ]
            )
        review.gate("tail_lower90_rho_and_core_gates_recompute_from_spectra", all(tail_rows_match))
        accepted_count = int(states.loc[states["proposal"].gt(0), "accepted"].sum())
        tail_accepts = int(states.loc[states["proposal"].between(81, 100), "accepted"].sum())
        cumulative_radius = float(states.loc[states["proposal"].gt(0), "accepted_radius"].sum())
        strict_decreases = int(
            (states["exact_a_per_dim"].diff().iloc[1:] < -tolerances["a"]).sum()
        )
        accepted_states = states.loc[states["proposal"].gt(0) & states["accepted"].eq(1)]
        spectral_guards = bool(
            all(
                float(row["m_max"])
                <= float(states.iloc[int(row["proposal"]) - 1]["m_max"]) + tolerances["m_max"]
                and float(row["m_p50"])
                >= float(states.iloc[int(row["proposal"]) - 1]["m_p50"]) - tolerances["m_p50"]
                and float(row["m_lt_0p1_fraction"])
                <= float(states.iloc[int(row["proposal"]) - 1]["m_lt_0p1_fraction"]
                )
                + tolerances["m_lt_0p1_fraction"]
                for _, row in accepted_states.iterrows()
            )
        )
        success_gates = {
            "final_exact_a_at_most_0p90": float(states.iloc[-1]["exact_a_per_dim"]) <= 0.90,
            "all_tail_exact_a_at_most_1": bool(
                states.loc[states["proposal"].between(81, 100), "exact_a_per_dim"].le(1.0).all()
            ),
            "accepted_proposals_at_least_20": accepted_count >= 20,
            "tail_accepted_proposals_at_least_4": tail_accepts >= 4,
            "cumulative_path_radius_at_least_0p10": cumulative_radius >= 0.10,
            "final_full_b_below_initial": float(states.iloc[-1]["damped_full_burg_per_dim"])
            < float(states.iloc[0]["damped_full_burg_per_dim"]),
            "strict_a_decreases_at_least_30": strict_decreases >= 30,
            "tail_core_noncollapse_at_least_16_of_20": sum(tail_passes) >= TAIL_REQUIRED,
            "accepted_steps_respect_spectral_guards": spectral_guards,
        }
        scientific_success = bool(all(success_gates.values()))
        review.details["scientific_success_gates"] = success_gates
        review.details["tail_summary"] = {
            "accepted_count": accepted_count,
            "tail_accepted_count": tail_accepts,
            "strict_a_decreases": strict_decreases,
            "cumulative_path_radius": cumulative_radius,
            "tail_core_noncollapse_pass_count": sum(tail_passes),
        }
        review.gate(
            "reported_counts_and_path_radius_recompute",
            decision.get("accepted_count") == accepted_count
            and decision.get("strict_a_decreases") == strict_decreases
            and _close(decision.get("cumulative_path_radius"), cumulative_radius, atol=1e-10)
            and decision.get("tail_core_noncollapse_pass_count") == sum(tail_passes),
        )
        review.gate(
            "reported_scientific_success_gates_recompute",
            _equivalent(decision.get("success_gates"), success_gates)
            and decision.get("success") is scientific_success
            and finalized.get("success") is scientific_success,
        )
        expected_hypothesis = (
            "supported_under_exact_feasibility_controller"
            if scientific_success
            else "not_supported_at_frozen_success_thresholds"
        )
        review.gate(
            "hypothesis_result_matches_scientific_success",
            decision.get("hypothesis_result") == expected_hypothesis,
        )
        review.gate(
            "initial_and_final_metrics_match_state_table",
            all(
                key in states.iloc[0].index and _close(value, states.iloc[0][key], atol=1e-10, rtol=1e-9)
                for key, value in decision["initial_metrics"].items()
                if isinstance(value, Real) and not isinstance(value, bool)
            )
            and all(
                key in states.iloc[-1].index and _close(value, states.iloc[-1][key], atol=1e-10, rtol=1e-9)
                for key, value in decision["final_metrics"].items()
                if isinstance(value, Real) and not isinstance(value, bool)
            ),
        )

        recomputed_validity.update(
            {
                "checkpoint_matches": review.gates[
                    "accepted_input_checkpoint_frozen_hash_matches"
                ],
                "z_matches": bool(context.get("checkpoint_z_matches")),
                "initial_parameters_match": str(proposals.iloc[0]["base_parameter_hash"])
                == EXPECTED_INITIAL_PARAMETER_SHA256,
                "active_parameter_count_matches": bool(
                    context.get("checkpoint_active_parameter_count_matches")
                ),
                "frozen_dependencies_match": review.gates[
                    "all_frozen_dependencies_still_match"
                ],
                "state_rows_101": len(states) == 101,
                "proposal_rows_100": len(proposals) == 100,
                "pair_rows_3200": len(frames["pairs"]) == 3200,
                "draw_rows_800": len(frames["draws"]) == 800,
                "tables_numeric_finite": review.gates["all_required_table_values_finite"],
                "line_exact_a_direct_closure": bool(
                    frames["line"]["a_direct_abs_error"].le(1e-9).all()
                ),
                "line_exact_a_trace_closure": bool(
                    frames["line"]["a_trace_abs_error"].le(1e-9).all()
                ),
            }
        )
        producer_gates = decision["validity_gates"]
        review.details["recomputed_validity_gates"] = recomputed_validity
        review.gate(
            "recomputed_validity_gates_match_decision",
            all(
                key in producer_gates and bool(producer_gates[key]) == bool(value)
                for key, value in recomputed_validity.items()
            ),
        )
        producer_only = {
            "full_ce",
            "final_parameter_hash_unchanged_by_replay",
            "cuda_rng_unchanged",
        }
        review.gate(
            "producer_only_validity_gates_present_and_true",
            producer_only.issubset(producer_gates)
            and all(producer_gates[key] is True for key in producer_only),
        )
        review.gate(
            "decision_and_finalized_validity_agree",
            decision.get("valid") is True and finalized.get("valid") is True,
        )

    review.phase("tail_and_decision", tail_and_decision_phase)
    return 0 if _write_report(output, review, scientific_success) else 1


if __name__ == "__main__":
    raise SystemExit(main())
