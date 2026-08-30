#!/usr/bin/env python3
"""Independent CPU analyzer for the frozen prospective source replication.

This program consumes only sealed runner outputs and persisted latent tensors.
It never imports or executes the Weight-AE.  Formal analysis is fail-closed:
the frozen design hash, runner metadata, complete sufficient-statistic grids,
and source cache hashes must all match before a scientific decision is made.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
FORMAL_ARTIFACT_ROOT = (PROJECT / "artifacts/crossmodal_united_structure").resolve()
DESIGN_PATH = PROJECT / "docs/notes/prospective_geometry_matched_source_replication_design_20260816.md"
FROZEN_DESIGN_SHA256 = "fc23d52d07d0004afd681116578abdba562dff1a415b059fd72ec261b1e47d70"
SOURCE_GAIN_PATH = (
    PROJECT
    / "artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/source_fit_role_gains.csv"
)
SOURCE_GAIN_SHA256 = "e05f2339ba33eb65feee00df2f095655c9319ecf118cd81d7159f6e14f70c4f3"
SOURCE_CACHE_DIR = PROJECT / "artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816"
DEFAULT_OUTPUT_DIR = (
    PROJECT
    / "artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_analysis_20260816"
)
SOURCE_CACHES = {
    26_081_601: (
        SOURCE_CACHE_DIR / "factorial_code_cache_seed_26081601.pt",
        "a585c2ecfe4d1e560c79dca19518543dbdfb5f897bfa413b3ef5148f01995c8f",
    ),
    26_081_602: (
        SOURCE_CACHE_DIR / "factorial_code_cache_seed_26081602.pt",
        "c56f1ed3ccd0038f3f78ec767facd32cc0bba490f2e31229ec5d7b18a211bfd6",
    ),
}

PANELS = ("beans", "trocr_sroie")
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
SCORE_SPLITS = ("A", "B")
TILING_SEEDS = (26_081_801, 26_081_802)
DEPTHS = tuple(range(12))
AGGREGATIONS = ("macro", "micro")
ROLE_AND_GLOBAL_AGGREGATIONS = (*ROLES, *AGGREGATIONS)
CALIBRATIONS = ("raw", "source_role_gain", "source_median_common")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEEDS = {"beans": 26_081_903, "trocr_sroie": 26_081_902}
GAIN_TIE_ATOL = 1e-12
COMMON_GAIN = 0.46301170700708616
GAINS = {
    "attn_query": 0.5270182885626962,
    "attn_key": 0.5562907139098529,
    "attn_value": 0.3990051254514762,
    "attn_output": 0.3564476277035192,
    "ffn_up": 1.0280206811266708,
    "ffn_down": 0.3048084709039325,
}
RADIAL_ROLES = ("attn_value", "attn_output", "ffn_down")

WEIGHT_COLUMNS = (
    "panel_id",
    "tiling_seed",
    "depth",
    "role",
    "matrix_key",
    "arm",
    "prediction_sha256",
    "T",
    "P",
    "D",
)
OPERATOR_COLUMNS = (
    "panel_id",
    "tiling_seed",
    "depth",
    "role",
    "matrix_key",
    "arm",
    "prediction_sha256",
    "score_split",
    "activation_sha256",
    "T",
    "P",
    "D",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")

BOUND_HASH_FIELDS = (
    "design_sha256",
    "runner_sha256",
    "builder_sha256",
    "analyzer_sha256",
    "panel_manifest_sha256",
    "checkpoint_sha256",
)
METADATA_PANEL_KEYS = frozenset(
    {
        "panel_id",
        "path",
        "sha256",
        "checkpoint_sha256",
        "provenance_pass",
        "resource_hash_pass",
        "quality_pass",
        "validity_pass",
    }
)
BINDING_KEYS = frozenset(
    {
        "schema_version",
        "external_contract_path",
        "external_contract_sha256",
        "copied_contract_path",
        "copied_contract_sha256",
        *BOUND_HASH_FIELDS,
        "scripts",
        "model_dependencies",
        "resources",
        "panel_cache",
    }
)
RAW_RUN_MANIFEST_KEYS = frozenset(
    {"schema_version", "manifest_self_excluded", "artifacts", "count"}
)
RAW_RUN_MANIFEST_ROW_KEYS = frozenset({"path", "sha256", "bytes"})

EXPECTED_VERSIONS = {
    "torch": "2.10.0+cu128",
    "transformers": "5.1.0",
    "safetensors": "0.7.0",
    "Pillow": "12.0.0",
    "datasets": "3.6.0",
    "numpy": "2.3.5",
    "scikit-learn": "1.8.0",
    "pandas": "3.0.0",
    "pyarrow": "23.0.0",
}

PANEL_GEOMETRY_META = {
    "source_vit_b_flickr": {
        "panel_domain": "source / vision retrieval",
        "dataset": "Flickr",
        "model": "held-out ViT-B/Flickr",
    },
    "beans": {
        "panel_domain": "P1 / vision classification",
        "dataset": "Beans",
        "model": "nateraw/vit-base-beans",
    },
    "trocr_sroie": {
        "panel_domain": "P2 / vision OCR",
        "dataset": "SROIE",
        "model": "microsoft/trocr-base-printed encoder",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Fresh completed runner output directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Absent analyzer output directory",
    )
    parser.add_argument("--design-path", type=Path, default=DESIGN_PATH)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic-only tests; never reads formal runner outputs",
    )
    parser.add_argument(
        "--self-test-output-dir",
        type=Path,
        help="Optional absent directory in the formal artifact root for persistent synthetic-test evidence",
    )
    args = parser.parse_args(argv)
    if not args.self_test and args.input_dir is None:
        parser.error("formal analysis requires --input-dir")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def scalar_json(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prospective_geometry_matched_analyzer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def flush_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def contains_forbidden_target_marker(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    if "data2vec" in normalized:
        return True
    components: list[str] = []
    for component in normalized.split("/"):
        components.extend(component.split("="))
    return any(
        component in {"target", "targets"}
        or component.startswith("target_")
        or component.startswith("target-")
        for component in components
    )


def reject_forbidden_argv(argv: Sequence[str]) -> None:
    for argument in argv:
        if contains_forbidden_target_marker(str(argument)):
            raise RuntimeError("source-only analyzer rejected a target-related command-line argument")


def reject_forbidden_path(path: Path, label: str) -> Path:
    original = str(path).lower()
    resolved = path.resolve(strict=False)
    if contains_forbidden_target_marker(original) or contains_forbidden_target_marker(str(resolved)):
        raise RuntimeError(f"forbidden target marker in {label} path")
    return resolved


def require_within(path: Path, root: Path, label: str, strict: bool = True) -> Path:
    resolved = path.resolve(strict=strict)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(f"{label} path is outside allowed root {root}: {resolved}") from error
    reject_forbidden_path(resolved, label)
    return resolved


def install_source_only_seal() -> dict[str, Any]:
    state: dict[str, Any] = {
        "installed_before_input_read": True,
        "target_access_events": 0,
        "network_connections": 0,
        "subprocess_launches": 0,
        "filesystem_audit_hook_installed": False,
    }
    audit_local = threading.local()

    filesystem_events = {
        "open",
        "os.open",
        "os.chdir",
        "os.listdir",
        "os.scandir",
        "os.stat",
        "os.lstat",
        "os.readlink",
        "pathlib.Path.glob",
        "pathlib.Path.rglob",
    }

    def audit_path_candidate(candidate: Any) -> None:
        if isinstance(candidate, int):
            return
        try:
            raw = os.fsdecode(os.fspath(candidate))
        except (TypeError, ValueError):
            return
        if contains_forbidden_target_marker(raw):
            state["target_access_events"] += 1
            raise RuntimeError("source-only analyzer rejected a target-marked filesystem path")
        if getattr(audit_local, "resolving", False):
            return
        audit_local.resolving = True
        try:
            resolved = os.path.realpath(raw)
        finally:
            audit_local.resolving = False
        if contains_forbidden_target_marker(resolved):
            state["target_access_events"] += 1
            raise RuntimeError("source-only analyzer rejected a symlink-resolved target path")

    def filesystem_audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in filesystem_events or getattr(audit_local, "resolving", False):
            return
        if event in {"pathlib.Path.glob", "pathlib.Path.rglob"}:
            candidates = args[:1]
        else:
            candidates = args[:1]
        for candidate in candidates:
            audit_path_candidate(candidate)

    sys.addaudithook(filesystem_audit_hook)
    state["filesystem_audit_hook_installed"] = True

    def block_network(*_args: Any, **_kwargs: Any) -> Any:
        state["network_connections"] += 1
        raise RuntimeError("network access is prohibited in the CPU analyzer")

    def block_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        state["subprocess_launches"] += 1
        raise RuntimeError("subprocess execution is prohibited in the CPU analyzer")

    socket.socket.connect = block_network  # type: ignore[method-assign]
    socket.socket.connect_ex = block_network  # type: ignore[method-assign]
    socket.create_connection = block_network  # type: ignore[assignment]
    subprocess.Popen = block_subprocess  # type: ignore[assignment]
    subprocess.run = block_subprocess  # type: ignore[assignment]
    subprocess.call = block_subprocess  # type: ignore[assignment]
    subprocess.check_call = block_subprocess  # type: ignore[assignment]
    subprocess.check_output = block_subprocess  # type: ignore[assignment]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return state


def assert_no_forbidden_modules() -> dict[str, Any]:
    target_modules = sorted(name for name in sys.modules if "data2vec" in name.lower())
    wae_markers = ("weight_vae", "weight_ae", "big_vae")
    wae_modules = sorted(name for name in sys.modules if any(marker in name.lower() for marker in wae_markers))
    if target_modules or wae_modules:
        raise RuntimeError(f"forbidden modules imported: target={target_modules}, weight_ae={wae_modules}")
    return {"target_modules": target_modules, "weight_ae_modules": wae_modules, "pass": True}


def validate_runtime_stack() -> dict[str, Any]:
    versions: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for distribution, expected in EXPECTED_VERSIONS.items():
        if distribution == "torch":
            import torch

            actual = str(torch.__version__)
        else:
            actual = importlib.metadata.version(distribution)
        versions[distribution] = actual
        if actual != expected:
            mismatches[distribution] = {"actual": actual, "expected": expected}
    python_ok = sys.version_info[:2] == (3, 12)
    if mismatches or not python_ok:
        raise RuntimeError(
            f"frozen runtime mismatch: python={platform.python_version()} mismatches={mismatches}"
        )
    return {
        "python": platform.python_version(),
        "packages": versions,
        "cpu_only_analyzer": True,
        "pass": True,
    }


def validate_design_and_gains(design_path: Path) -> dict[str, Any]:
    resolved_design = reject_forbidden_path(design_path, "design").resolve(strict=True)
    if resolved_design != DESIGN_PATH.resolve(strict=True):
        raise RuntimeError(f"design path differs from canonical frozen path: {resolved_design}")
    design_sha = sha256_file(resolved_design)
    if design_sha != FROZEN_DESIGN_SHA256:
        raise RuntimeError(f"frozen design hash mismatch: {design_sha} != {FROZEN_DESIGN_SHA256}")
    gain_path = SOURCE_GAIN_PATH.resolve(strict=True)
    gain_sha = sha256_file(gain_path)
    if gain_sha != SOURCE_GAIN_SHA256:
        raise RuntimeError(f"source gain hash mismatch: {gain_sha} != {SOURCE_GAIN_SHA256}")
    frame = pd.read_csv(gain_path)
    selected = frame[frame["method"] == "ae_cell_mean_c0"]
    if len(selected) != 6 or set(selected["role"].astype(str)) != set(ROLES):
        raise RuntimeError("source gain CSV does not contain the exact six frozen role rows")
    observed = {str(row.role): float(row.gain) for row in selected.itertuples(index=False)}
    for role in ROLES:
        if not math.isclose(observed[role], GAINS[role], rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"frozen gain mismatch for {role}: {observed[role]} != {GAINS[role]}")
    median = float(np.median(np.asarray([observed[role] for role in ROLES], dtype=np.float64)))
    if not math.isclose(median, COMMON_GAIN, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"source median gain mismatch: {median} != {COMMON_GAIN}")
    return {
        "design": {"path": str(resolved_design), "sha256": design_sha, "bytes": resolved_design.stat().st_size},
        "source_gains": {"path": str(gain_path), "sha256": gain_sha, "bytes": gain_path.stat().st_size},
        "gains": observed,
        "source_median_common_gain": median,
        "pass": True,
    }


def _require_exact_mapping_keys(
    value: Any,
    expected: Iterable[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not a mapping")
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise RuntimeError(
            f"{label} exact key set mismatch: missing={sorted(expected_set - observed)} "
            f"extra={sorted(observed - expected_set)}"
        )
    return value


def _validate_panel_identity_record(
    record: Any,
    panel_id: str,
    label: str,
    *,
    exact_metadata_keys: bool,
) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} is not a mapping")
    identity_fields = {"panel_id", "path", "sha256", "checkpoint_sha256"}
    if exact_metadata_keys:
        _require_exact_mapping_keys(record, METADATA_PANEL_KEYS, label)
    elif not identity_fields.issubset(record):
        raise RuntimeError(
            f"{label} lacks panel identity fields: {sorted(identity_fields - set(record))}"
        )
    if record.get("panel_id") != panel_id:
        raise RuntimeError(f"{label} panel_id mismatch: {record.get('panel_id')} != {panel_id}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{label} path must be a nonempty string")
    if contains_forbidden_target_marker(raw_path):
        raise RuntimeError(f"{label} path contains a forbidden target marker")
    panel_path = Path(raw_path)
    if not panel_path.is_absolute():
        raise RuntimeError(f"{label} path is not absolute: {raw_path}")
    resolved_path = panel_path.resolve(strict=True)
    if raw_path != str(resolved_path):
        raise RuntimeError(f"{label} path is not canonical: {raw_path} -> {resolved_path}")
    result = {"panel_id": panel_id, "path": raw_path}
    for field in ("sha256", "checkpoint_sha256"):
        value = str(record.get(field, ""))
        if not HEX64.fullmatch(value):
            raise RuntimeError(f"{label} has malformed {field}")
        result[field] = value
    return result


def _contract_panel_identity_grid(contract: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    panel_cache = contract.get("panel_cache")
    if not isinstance(panel_cache, Mapping):
        raise RuntimeError("preexecution contract panel_cache is not a mapping")
    panels = panel_cache.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != set(PANELS):
        observed = set(panels) if isinstance(panels, Mapping) else type(panels).__name__
        raise RuntimeError(f"preexecution contract panel grid mismatch: {observed} != {set(PANELS)}")
    return {
        panel_id: _validate_panel_identity_record(
            panels[panel_id],
            panel_id,
            f"preexecution contract panel_cache.panels.{panel_id}",
            exact_metadata_keys=False,
        )
        for panel_id in PANELS
    }


def _metadata_panel_identity_grid(metadata: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    panels = metadata.get("panels")
    if not isinstance(panels, list) or len(panels) != len(PANELS):
        raise RuntimeError(f"runner metadata must contain exactly {len(PANELS)} panel records")
    by_panel: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(panels):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"runner metadata panels[{index}] is not a mapping")
        panel_id = record.get("panel_id")
        if not isinstance(panel_id, str) or panel_id in by_panel:
            raise RuntimeError(f"runner metadata panel ID is invalid or duplicated: {panel_id}")
        by_panel[panel_id] = record
    if set(by_panel) != set(PANELS):
        raise RuntimeError(f"runner metadata panel grid mismatch: {set(by_panel)} != {set(PANELS)}")
    return {
        panel_id: _validate_panel_identity_record(
            by_panel[panel_id],
            panel_id,
            f"runner metadata panels.{panel_id}",
            exact_metadata_keys=True,
        )
        for panel_id in PANELS
    }


def _validate_contract_metadata_identity(
    contract: Mapping[str, Any],
    runner_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    for field in BOUND_HASH_FIELDS:
        contract_value = str(contract.get(field, ""))
        metadata_value = str(runner_metadata.get(field, ""))
        if not HEX64.fullmatch(contract_value) or not HEX64.fullmatch(metadata_value):
            raise RuntimeError(f"contract/metadata identity has malformed {field}")
        if metadata_value != contract_value:
            raise RuntimeError(
                f"runner metadata/preexecution contract mismatch for {field}: "
                f"{metadata_value} != {contract_value}"
            )
    contract_panels = _contract_panel_identity_grid(contract)
    metadata_panels = _metadata_panel_identity_grid(runner_metadata)
    if metadata_panels != contract_panels:
        raise RuntimeError(
            "runner metadata/preexecution contract exact panel identity grid mismatch: "
            f"metadata={metadata_panels} contract={contract_panels}"
        )
    return {
        "hash_fields": list(BOUND_HASH_FIELDS),
        "panel_fields": ["panel_id", "path", "sha256", "checkpoint_sha256"],
        "panel_ids": list(PANELS),
        "pass": True,
    }


def _validate_three_way_binding_identity(
    contract: Mapping[str, Any],
    runner_metadata: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_mapping_keys(binding, BINDING_KEYS, "preexecution binding")
    contract_metadata = _validate_contract_metadata_identity(contract, runner_metadata)
    for field in BOUND_HASH_FIELDS:
        if binding.get(field) != contract.get(field) or binding.get(field) != runner_metadata.get(field):
            raise RuntimeError(
                f"runner metadata/preexecution contract/binding mismatch for {field}"
            )
    for field in ("scripts", "model_dependencies", "resources", "panel_cache"):
        if binding.get(field) != contract.get(field):
            raise RuntimeError(f"preexecution binding/contract structural mismatch for {field}")
    binding_panels = _contract_panel_identity_grid({"panel_cache": binding["panel_cache"]})
    metadata_panels = _metadata_panel_identity_grid(runner_metadata)
    contract_panels = _contract_panel_identity_grid(contract)
    if not (binding_panels == metadata_panels == contract_panels):
        raise RuntimeError("metadata/contract/binding exact panel identity grid mismatch")
    return {
        **contract_metadata,
        "three_way_hash_binding": True,
        "three_way_panel_binding": True,
        "structural_fields": ["scripts", "model_dependencies", "resources", "panel_cache"],
    }


def _validate_contract_digest_binding(
    contract_sha256: str,
    runner_metadata: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    values = {
        "actual_contract_sha256": contract_sha256,
        "metadata.contract_sha256": runner_metadata.get("contract_sha256"),
        "binding.external_contract_sha256": binding.get("external_contract_sha256"),
        "binding.copied_contract_sha256": binding.get("copied_contract_sha256"),
    }
    if any(not HEX64.fullmatch(str(value)) for value in values.values()):
        raise RuntimeError(f"malformed preexecution contract digest binding: {values}")
    if len(set(values.values())) != 1:
        raise RuntimeError(f"preexecution contract digest binding mismatch: {values}")


def validate_runner_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "prospective_geometry_matched_replication_runner_v1":
        raise RuntimeError("runner metadata schema_version mismatch")
    for field in (*BOUND_HASH_FIELDS, "contract_sha256"):
        if not HEX64.fullmatch(str(payload.get(field, ""))):
            raise RuntimeError(f"runner metadata has malformed {field}")
    if payload.get("design_sha256") != FROZEN_DESIGN_SHA256:
        raise RuntimeError("runner metadata design hash mismatch")
    if payload.get("execution_status") not in {"COMPLETE", "INVALID_PANEL_PRE_WAE"}:
        raise RuntimeError(f"runner execution status is not terminal: {payload.get('execution_status')}")
    seal = payload.get("target_seal")
    if not isinstance(seal, dict):
        raise RuntimeError("runner metadata lacks target_seal mapping")
    expected_seal = {
        "installed_before_input_read": True,
        "target_access_events": 0,
        "network_connections": 0,
        "subprocess_launches": 0,
    }
    for field, expected in expected_seal.items():
        if seal.get(field) != expected:
            raise RuntimeError(f"runner target seal mismatch for {field}: {seal.get(field)} != {expected}")
    _metadata_panel_identity_grid(payload)
    panels = payload["panels"]
    by_panel: dict[str, Mapping[str, Any]] = {}
    invalid: list[str] = []
    for panel in panels:
        panel_id = str(panel["panel_id"])
        by_panel[panel_id] = panel
        for field in ("provenance_pass", "resource_hash_pass", "quality_pass", "validity_pass"):
            if not isinstance(panel.get(field), bool):
                raise RuntimeError(f"panel {panel_id} lacks boolean {field}")
        if not all(panel[field] for field in ("provenance_pass", "resource_hash_pass", "quality_pass", "validity_pass")):
            invalid.append(panel_id)
    if set(by_panel) != set(PANELS):
        raise RuntimeError(f"runner panel IDs mismatch: {set(by_panel)} != {set(PANELS)}")
    if invalid and payload.get("execution_status") != "INVALID_PANEL_PRE_WAE":
        raise RuntimeError("invalid panel metadata requires INVALID_PANEL_PRE_WAE execution status")
    if not invalid and payload.get("execution_status") != "COMPLETE":
        raise RuntimeError("two valid panels require COMPLETE execution status")
    if not invalid:
        expected_hard_checks = {
            "transpose_role_map_parity_pass",
            "raw_stat_algebra_pass",
            "analytic_scale_parity_pass",
            "decoder_entry_parity_pass",
            "A_B_prediction_sha_pass",
            "derangement_pass",
            "tiling_coverage_pass",
            "finite_nondegenerate_pass",
            "source_only_dataflow_pass",
        }
        hard_checks = payload.get("hard_checks")
        if not isinstance(hard_checks, dict) or set(hard_checks) != expected_hard_checks:
            raise RuntimeError("runner metadata hard_checks exact key set mismatch")
        if not all(value is True for value in hard_checks.values()):
            raise RuntimeError(f"runner metadata contains a failed hard check: {hard_checks}")
    return payload, sorted(invalid)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise RuntimeError(f"{label} contains nulls in required columns")


def _validate_aliases(frame: pd.DataFrame, label: str) -> None:
    aliases = {"target_energy": "T", "pred_energy": "P", "target_pred_dot": "D"}
    for alias, canonical in aliases.items():
        if alias in frame.columns:
            left = frame[alias].to_numpy(dtype=np.float64)
            right = frame[canonical].to_numpy(dtype=np.float64)
            if not np.array_equal(left, right):
                raise RuntimeError(f"{label} descriptive alias {alias} differs from {canonical}")


def _validate_numeric_stats(frame: pd.DataFrame, label: str) -> dict[str, float]:
    values = frame[["T", "P", "D"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains nonfinite sufficient statistics")
    if (frame[["T", "P"]].to_numpy(dtype=np.float64) <= 0.0).any():
        raise RuntimeError(f"{label} contains nonpositive target or decoded prediction energy")
    target = frame["T"].to_numpy(dtype=np.float64)
    pred = frame["P"].to_numpy(dtype=np.float64)
    dot = frame["D"].to_numpy(dtype=np.float64)
    root = np.sqrt(target * pred)
    cauchy_excess = np.maximum(np.abs(dot) - root, 0.0)
    if np.any(cauchy_excess > 1e-10 * np.maximum(root, 1.0)):
        raise RuntimeError(f"{label} violates Cauchy-Schwarz beyond FP64 tolerance")
    raw_sse = target - 2.0 * dot + pred
    negative = raw_sse < -1e-10 * np.maximum.reduce([target, pred, np.ones_like(target)])
    if negative.any():
        raise RuntimeError(f"{label} has materially negative raw SSE")
    return {
        "min_T": float(target.min()),
        "min_P": float(pred.min()),
        "max_cauchy_relative_excess": float(np.max(cauchy_excess / np.maximum(root, 1.0))),
        "min_raw_sse": float(raw_sse.min()),
    }


def _require_exact_set(frame: pd.DataFrame, column: str, expected: set[Any], label: str) -> None:
    observed = set(frame[column].tolist())
    if observed != expected:
        raise RuntimeError(f"{label} exact set mismatch for {column}: {observed} != {expected}")


def _require_constant_within(frame: pd.DataFrame, groups: list[str], columns: list[str], label: str) -> None:
    counts = frame.groupby(groups, sort=False, dropna=False)[columns].nunique(dropna=False)
    if (counts != 1).any().any():
        bad = counts[(counts != 1).any(axis=1)].head(5).to_dict("index")
        raise RuntimeError(f"{label} fields are not constant within groups: {bad}")


def load_and_validate_sufficient_statistics(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    weight_path = input_dir / "weight_sufficient_stats.csv"
    operator_path = input_dir / "operator_sufficient_stats.csv"
    weight = pd.read_csv(weight_path, dtype={"panel_id": str, "role": str, "matrix_key": str, "arm": str, "prediction_sha256": str})
    operator = pd.read_csv(
        operator_path,
        dtype={
            "panel_id": str,
            "role": str,
            "matrix_key": str,
            "arm": str,
            "prediction_sha256": str,
            "score_split": str,
            "activation_sha256": str,
        },
    )
    _require_columns(weight, WEIGHT_COLUMNS, "weight stats")
    _require_columns(operator, OPERATOR_COLUMNS, "operator stats")
    if len(weight) != 864:
        raise RuntimeError(f"weight sufficient-statistic row count {len(weight)} != 864")
    if len(operator) != 1_728:
        raise RuntimeError(f"operator sufficient-statistic row count {len(operator)} != 1728")
    for frame in (weight, operator):
        frame["tiling_seed"] = frame["tiling_seed"].astype(np.int64)
        frame["depth"] = frame["depth"].astype(np.int64)
        for column in ("T", "P", "D"):
            frame[column] = frame[column].astype(np.float64)
    _validate_aliases(weight, "weight stats")
    _validate_aliases(operator, "operator stats")
    weight_numeric = _validate_numeric_stats(weight, "weight stats")
    operator_numeric = _validate_numeric_stats(operator, "operator stats")

    for label, frame in (("weight stats", weight), ("operator stats", operator)):
        _require_exact_set(frame, "panel_id", set(PANELS), label)
        _require_exact_set(frame, "tiling_seed", set(TILING_SEEDS), label)
        _require_exact_set(frame, "depth", set(DEPTHS), label)
        _require_exact_set(frame, "role", set(ROLES), label)
        _require_exact_set(frame, "arm", set(ARMS), label)
        for column in ("prediction_sha256",):
            if not frame[column].map(lambda value: bool(HEX64.fullmatch(str(value)))).all():
                raise RuntimeError(f"{label} contains malformed {column}")
        if (frame["matrix_key"].str.len() == 0).any():
            raise RuntimeError(f"{label} contains empty matrix_key")
    _require_exact_set(operator, "score_split", set(SCORE_SPLITS), "operator stats")
    if not operator["activation_sha256"].map(lambda value: bool(HEX64.fullmatch(str(value)))).all():
        raise RuntimeError("operator stats contain malformed activation_sha256")

    weight_key = ["panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm"]
    operator_key = [*weight_key, "score_split"]
    if weight.duplicated(weight_key).any():
        raise RuntimeError("weight stats contain duplicate formal cells")
    if operator.duplicated(operator_key).any():
        raise RuntimeError("operator stats contain duplicate formal cells")

    weight_arm_sizes = weight.groupby(["panel_id", "tiling_seed", "depth", "role"], sort=False).size()
    operator_arm_sizes = operator.groupby(
        ["panel_id", "score_split", "tiling_seed", "depth", "role"], sort=False
    ).size()
    if len(weight_arm_sizes) != 288 or set(weight_arm_sizes.tolist()) != {3}:
        raise RuntimeError("weight stats do not form 288 complete three-arm cells")
    if len(operator_arm_sizes) != 576 or set(operator_arm_sizes.tolist()) != {3}:
        raise RuntimeError("operator stats do not form 576 complete three-arm cells")

    _require_constant_within(
        weight,
        ["panel_id", "tiling_seed", "depth", "role"],
        ["matrix_key", "T"],
        "weight arm",
    )
    _require_constant_within(
        weight,
        ["panel_id", "depth", "role"],
        ["matrix_key", "T"],
        "weight tiling",
    )
    _require_constant_within(
        operator,
        ["panel_id", "score_split", "tiling_seed", "depth", "role"],
        ["matrix_key", "activation_sha256", "T"],
        "operator arm",
    )
    _require_constant_within(
        operator,
        ["panel_id", "score_split", "depth", "role"],
        ["matrix_key", "activation_sha256", "T"],
        "operator tiling",
    )
    _require_constant_within(
        operator,
        ["panel_id", "tiling_seed", "depth", "role", "arm"],
        ["matrix_key", "prediction_sha256"],
        "A/B prediction reuse",
    )

    for label, frame, groups in (
        ("weight", weight, ["panel_id", "tiling_seed", "depth", "role"]),
        ("operator", operator, ["panel_id", "score_split", "tiling_seed", "depth", "role"]),
    ):
        unique_hashes = frame.groupby(groups, sort=False)["prediction_sha256"].nunique()
        if set(unique_hashes.tolist()) != {3}:
            raise RuntimeError(f"{label} correct/control prediction hashes collide")

    activation_split_counts = operator.groupby(["panel_id", "depth", "role"], sort=False)[
        "activation_sha256"
    ].nunique()
    if set(activation_split_counts.tolist()) != {2}:
        raise RuntimeError("operator A/B activation hashes are not distinct for every matrix")

    weight_lookup = weight[[
        "panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm", "prediction_sha256"
    ]]
    merged = operator.merge(
        weight_lookup,
        on=["panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm"],
        suffixes=("_operator", "_weight"),
        validate="many_to_one",
    )
    if len(merged) != 1_728 or not (
        merged["prediction_sha256_operator"] == merged["prediction_sha256_weight"]
    ).all():
        raise RuntimeError("operator and weight prediction hashes do not agree exactly")

    matrix_map = pd.concat(
        [
            weight[["panel_id", "depth", "role", "matrix_key"]],
            operator[["panel_id", "depth", "role", "matrix_key"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    if len(matrix_map) != 144 or matrix_map.duplicated(["panel_id", "depth", "role"]).any():
        raise RuntimeError("matrix_key mapping is not a one-to-one 144-matrix grid")

    return weight, operator, {
        "pass": True,
        "weight_rows": len(weight),
        "operator_rows": len(operator),
        "weight_unique_cells": int(weight[weight_key].drop_duplicates().shape[0]),
        "operator_unique_cells": int(operator[operator_key].drop_duplicates().shape[0]),
        "weight_numeric": weight_numeric,
        "operator_numeric": operator_numeric,
        "prediction_hashes_reused_exactly_across_A_B": True,
        "prediction_hashes_match_weight_and_operator": True,
        "activation_hashes_distinct_across_A_B": True,
        "matrix_grid_rows": len(matrix_map),
        "input_files": {
            "weight": {"path": str(weight_path.resolve()), "sha256": sha256_file(weight_path), "bytes": weight_path.stat().st_size},
            "operator": {"path": str(operator_path.resolve()), "sha256": sha256_file(operator_path), "bytes": operator_path.stat().st_size},
        },
    }


def _int64_index_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_permutation_manifest(
    input_dir: Path,
    weight: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = input_dir / "permutation_manifest.csv"
    frame = pd.read_csv(path)
    columns = [
        "depth",
        "namespace",
        "panel",
        "role",
        "row_group",
        "tiling_seed",
        "n_col_groups",
        "offset",
        "fixed_points",
        "bijection_pass",
        "mapping_sha256",
        "matrix_key",
        "correct_code_block_sha256",
        "permuted_code_block_sha256",
        "same_row_code_multiset_bit_exact",
    ]
    if list(frame.columns) != columns:
        raise RuntimeError(f"permutation manifest columns/order mismatch: {list(frame.columns)}")
    if len(frame) != 5_184:
        raise RuntimeError(f"permutation manifest row count {len(frame)} != 5184")
    if frame.isna().any().any():
        raise RuntimeError("permutation manifest contains nulls")
    for column in ("depth", "row_group", "tiling_seed", "n_col_groups", "offset", "fixed_points"):
        frame[column] = frame[column].astype(np.int64)
    _require_exact_set(frame, "panel", set(PANELS), "permutation manifest")
    _require_exact_set(frame, "tiling_seed", set(TILING_SEEDS), "permutation manifest")
    _require_exact_set(frame, "depth", set(DEPTHS), "permutation manifest")
    _require_exact_set(frame, "role", set(ROLES), "permutation manifest")
    _require_exact_set(
        frame,
        "namespace",
        {"within_row_code_derangement_v1"},
        "permutation manifest",
    )
    key_columns = ["panel", "tiling_seed", "depth", "role", "row_group"]
    if frame.duplicated(key_columns).any():
        raise RuntimeError("permutation manifest contains duplicate row-group cells")
    matrix_map = {
        (str(row.panel_id), int(row.depth), str(row.role)): str(row.matrix_key)
        for row in weight[["panel_id", "depth", "role", "matrix_key"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    recomputed_rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        panel_id = str(row["panel"])
        depth = int(row["depth"])
        role = str(row["role"])
        tiling_seed = int(row["tiling_seed"])
        row_group = int(row["row_group"])
        d_in, d_out = expected_matrix_shape(role)
        expected_row_groups = d_in // 64
        expected_col_groups = d_out // 64
        if not (0 <= row_group < expected_row_groups):
            raise RuntimeError(f"permutation row_group out of range: {row}")
        if int(row["n_col_groups"]) != expected_col_groups:
            raise RuntimeError(f"permutation n_col_groups mismatch: {row}")
        expected_matrix_key = matrix_map[(panel_id, depth, role)]
        if str(row["matrix_key"]) != expected_matrix_key:
            raise RuntimeError(f"permutation matrix_key mismatch: {row}")
        canonical = {
            "depth": depth,
            "namespace": "within_row_code_derangement_v1",
            "panel": panel_id,
            "role": role,
            "row_group": row_group,
            "tiling_seed": tiling_seed,
        }
        canonical_bytes = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        expected_offset = 1 + value % (expected_col_groups - 1)
        if int(row["offset"]) != expected_offset:
            raise RuntimeError(f"permutation offset mismatch: {row}")
        mapping = np.asarray(
            [(column + expected_offset) % expected_col_groups for column in range(expected_col_groups)],
            dtype=np.int64,
        )
        fixed_points = int(np.sum(mapping == np.arange(expected_col_groups, dtype=np.int64)))
        bijection = bool(np.array_equal(np.sort(mapping), np.arange(expected_col_groups, dtype=np.int64)))
        mapping_sha = _int64_index_sha256(mapping)
        if int(row["fixed_points"]) != fixed_points or fixed_points != 0:
            raise RuntimeError(f"permutation fixed-point check failed: {row}")
        if row["bijection_pass"] is not True or not bijection:
            raise RuntimeError(f"permutation bijection check failed: {row}")
        if str(row["mapping_sha256"]) != mapping_sha:
            raise RuntimeError(f"permutation mapping hash mismatch: {row}")
        if row["same_row_code_multiset_bit_exact"] is not True:
            raise RuntimeError(f"permutation code multiset runtime parity failed: {row}")
        for field in (
            "mapping_sha256",
            "correct_code_block_sha256",
            "permuted_code_block_sha256",
        ):
            if not HEX64.fullmatch(str(row[field])):
                raise RuntimeError(f"permutation manifest malformed {field}: {row}")
        if row["correct_code_block_sha256"] == row["permuted_code_block_sha256"]:
            raise RuntimeError(f"permutation correct/control code block hashes collide: {row}")
        recomputed_rows.append(
            {
                "panel": panel_id,
                "tiling_seed": tiling_seed,
                "depth": depth,
                "role": role,
                "row_group": row_group,
                "offset": expected_offset,
                "mapping_sha256": mapping_sha,
            }
        )
    group_sizes = frame.groupby(["panel", "tiling_seed", "depth", "role"], sort=False).size()
    expected_sizes = {
        role: expected_matrix_shape(role)[0] // 64 for role in ROLES
    }
    for (panel_id, tiling_seed, depth, role), count in group_sizes.items():
        if int(count) != expected_sizes[str(role)]:
            raise RuntimeError(
                f"permutation row-group count mismatch {panel_id}/{tiling_seed}/{depth}/{role}: {count}"
            )
    if len(group_sizes) != 288:
        raise RuntimeError(f"permutation matrix group count {len(group_sizes)} != 288")
    return frame, {
        "pass": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "matrix_groups": len(group_sizes),
        "canonical_offsets_recomputed": True,
        "mapping_hashes_recomputed": True,
        "zero_fixed_points": True,
        "bijections": True,
        "runtime_code_multiset_checks_all_true": True,
        "recomputed_row_sha256": hashlib.sha256(
            json.dumps(recomputed_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def gains_for_roles(calibration: str, roles: Sequence[str]) -> np.ndarray:
    if calibration == "raw":
        return np.ones(len(roles), dtype=np.float64)
    if calibration == "source_role_gain":
        return np.asarray([GAINS[role] for role in roles], dtype=np.float64)
    if calibration == "source_median_common":
        return np.full(len(roles), COMMON_GAIN, dtype=np.float64)
    raise KeyError(f"unknown calibration: {calibration}")


def scaled_metrics(
    target: np.ndarray | float,
    pred: np.ndarray | float,
    dot: np.ndarray | float,
    gain: np.ndarray | float,
) -> dict[str, np.ndarray]:
    target_value, pred_value, dot_value, gain_value = np.broadcast_arrays(
        np.asarray(target, dtype=np.float64),
        np.asarray(pred, dtype=np.float64),
        np.asarray(dot, dtype=np.float64),
        np.asarray(gain, dtype=np.float64),
    )
    if np.any(target_value <= 0.0) or np.any(pred_value <= 0.0) or np.any(gain_value <= 0.0):
        raise RuntimeError("scaled metrics require positive T, P, and gain")
    pred_scaled = gain_value * gain_value * pred_value
    dot_scaled = gain_value * dot_value
    error = target_value - 2.0 * dot_scaled + pred_scaled
    tolerance = 1e-10 * np.maximum.reduce([target_value, pred_scaled, np.ones_like(target_value)])
    if np.any(error < -tolerance):
        raise RuntimeError("analytic gain produced materially negative SSE")
    error = np.maximum(error, 0.0)
    root = np.sqrt(target_value * pred_scaled)
    cosine_raw = dot_scaled / root
    if np.any(np.abs(cosine_raw) > 1.0 + 1e-10):
        raise RuntimeError("analytic gain produced an invalid cosine")
    cosine = np.clip(cosine_raw, -1.0, 1.0)
    norm_ratio = np.sqrt(pred_scaled / target_value)
    angular_floor = 1.0 - cosine * cosine
    radial_penalty = (norm_ratio - cosine) ** 2
    relative_error = error / target_value
    decomposition = angular_floor + radial_penalty
    if not np.allclose(relative_error, decomposition, rtol=1e-10, atol=1e-12):
        maximum = float(np.max(np.abs(relative_error - decomposition)))
        raise RuntimeError(f"radial/angular decomposition failed: max_abs={maximum}")
    direct = (
        target_value - 2.0 * gain_value * dot_value + gain_value * gain_value * pred_value
    ) / target_value
    if not np.allclose(relative_error, direct, rtol=1e-12, atol=1e-12):
        raise RuntimeError("direct and scaled sufficient-statistic gain formulas disagree")
    return {
        "T": target_value,
        "P_scaled": pred_scaled,
        "D_scaled": dot_scaled,
        "SSE": error,
        "E": relative_error,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "angular_floor": angular_floor,
        "radial_penalty": radial_penalty,
    }


def verify_analytic_gain_algebra(weight: pd.DataFrame, operator: pd.DataFrame) -> dict[str, Any]:
    maximum = 0.0
    checked = 0
    for label, frame in (("weight", weight), ("operator", operator)):
        roles = frame["role"].astype(str).tolist()
        target = frame["T"].to_numpy(dtype=np.float64)
        pred = frame["P"].to_numpy(dtype=np.float64)
        dot = frame["D"].to_numpy(dtype=np.float64)
        for calibration in CALIBRATIONS:
            gains = gains_for_roles(calibration, roles)
            metric = scaled_metrics(target, pred, dot, gains)
            scaled_again = scaled_metrics(
                target,
                metric["P_scaled"],
                metric["D_scaled"],
                np.ones(len(frame), dtype=np.float64),
            )
            difference = np.abs(metric["E"] - scaled_again["E"])
            maximum = max(maximum, float(np.max(difference)))
            checked += len(frame)
        if not np.isfinite(maximum):
            raise RuntimeError(f"nonfinite analytic parity result for {label}")
    return {
        "pass": True,
        "rows_times_calibrations_checked": checked,
        "max_abs_direct_vs_scaled_E": maximum,
        "calibrations": list(CALIBRATIONS),
        "role_gains": GAINS,
        "common_gain": COMMON_GAIN,
    }


def _metric_record(metric: Mapping[str, np.ndarray], index: int | None = None) -> dict[str, float]:
    def extract(name: str) -> float:
        value = np.asarray(metric[name])
        if index is None:
            if value.size != 1:
                raise RuntimeError(f"expected scalar metric {name}, got shape {value.shape}")
            return float(value.reshape(-1)[0])
        return float(value[index])

    return {
        "target_energy": extract("T"),
        "scaled_prediction_energy": extract("P_scaled"),
        "scaled_target_prediction_dot": extract("D_scaled"),
        "sse": extract("SSE"),
        "E": extract("E"),
        "cosine": extract("cosine"),
        "norm_ratio": extract("norm_ratio"),
        "angular_floor": extract("angular_floor"),
        "radial_penalty": extract("radial_penalty"),
    }


def build_aggregates(frame: pd.DataFrame, space: str) -> pd.DataFrame:
    if space not in {"weight", "operator"}:
        raise ValueError(space)
    group_columns = ["panel_id"]
    if space == "operator":
        group_columns.append("score_split")
    group_columns.extend(["tiling_seed", "arm"])
    rows: list[dict[str, Any]] = []
    for group_key, selected in frame.groupby(group_columns, sort=True):
        if len(selected) != 72:
            raise RuntimeError(f"incomplete aggregate input cell {space}/{group_key}: {len(selected)}")
        base = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
        role_sums = (
            selected.groupby("role", sort=False)[["T", "P", "D"]]
            .sum()
            .loc[list(ROLES)]
        )
        for calibration in CALIBRATIONS:
            gains = gains_for_roles(calibration, ROLES)
            role_metric = scaled_metrics(
                role_sums["T"].to_numpy(dtype=np.float64),
                role_sums["P"].to_numpy(dtype=np.float64),
                role_sums["D"].to_numpy(dtype=np.float64),
                gains,
            )
            role_records: list[dict[str, Any]] = []
            for role_index, role in enumerate(ROLES):
                record = {
                    **base,
                    "space": space,
                    "calibration": calibration,
                    "aggregation": role,
                    "matrices": 12,
                    "gain_min": float(gains[role_index]),
                    "gain_max": float(gains[role_index]),
                    **_metric_record(role_metric, role_index),
                }
                rows.append(record)
                role_records.append(record)
            macro = {
                **base,
                "space": space,
                "calibration": calibration,
                "aggregation": "macro",
                "matrices": 72,
                "gain_min": float(gains.min()),
                "gain_max": float(gains.max()),
                "target_energy": float(role_metric["T"].sum()),
                "scaled_prediction_energy": float(role_metric["P_scaled"].sum()),
                "scaled_target_prediction_dot": float(role_metric["D_scaled"].sum()),
                "sse": float(role_metric["SSE"].sum()),
            }
            for field in ("E", "cosine", "norm_ratio", "angular_floor", "radial_penalty"):
                macro[field] = float(np.mean(role_metric[field]))
            rows.append(macro)

            micro_metric = scaled_metrics(
                float(role_metric["T"].sum()),
                float(role_metric["P_scaled"].sum()),
                float(role_metric["D_scaled"].sum()),
                1.0,
            )
            micro = {
                **base,
                "space": space,
                "calibration": calibration,
                "aggregation": "micro",
                "matrices": 72,
                "gain_min": float(gains.min()),
                "gain_max": float(gains.max()),
                **_metric_record(micro_metric),
            }
            if not math.isclose(
                micro["sse"],
                float(role_metric["SSE"].sum()),
                rel_tol=1e-12,
                abs_tol=1e-8,
            ):
                raise RuntimeError(f"{space} role-dependent micro SSE algebra mismatch")
            rows.append(micro)
    result = pd.DataFrame(rows)
    expected = 288 if space == "weight" else 576
    if len(result) != expected:
        raise RuntimeError(f"{space} aggregate row count {len(result)} != {expected}")
    numeric = result.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{space} aggregate table contains nonfinite values")
    return result


def _cell_stat_arrays(
    operator: pd.DataFrame,
    panel_id: str,
    score_split: str,
    tiling_seed: int,
    arm: str,
) -> dict[str, np.ndarray]:
    selected = operator[
        (operator["panel_id"] == panel_id)
        & (operator["score_split"] == score_split)
        & (operator["tiling_seed"] == tiling_seed)
        & (operator["arm"] == arm)
    ]
    if len(selected) != 72:
        raise RuntimeError(f"incomplete bootstrap cell {panel_id}/{score_split}/{tiling_seed}/{arm}")
    indexed = selected.set_index(["depth", "role"])
    output: dict[str, np.ndarray] = {}
    for field in ("T", "P", "D"):
        output[field] = np.asarray(
            [[float(indexed.loc[(depth, role), field]) for role in ROLES] for depth in DEPTHS],
            dtype=np.float64,
        )
    return output


def _scope_draw_metrics(
    arrays: Mapping[str, np.ndarray],
    sampled_depths: np.ndarray,
    calibration: str,
) -> dict[str, dict[str, np.ndarray]]:
    if sampled_depths.ndim != 2 or sampled_depths.shape[1] != 12:
        raise RuntimeError(f"bad paired bootstrap index shape: {sampled_depths.shape}")
    target = arrays["T"][sampled_depths, :].sum(axis=1)
    pred = arrays["P"][sampled_depths, :].sum(axis=1)
    dot = arrays["D"][sampled_depths, :].sum(axis=1)
    gains = gains_for_roles(calibration, ROLES).reshape(1, len(ROLES))
    role_metric = scaled_metrics(target, pred, dot, gains)
    output: dict[str, dict[str, np.ndarray]] = {}
    for field in ("E", "cosine", "norm_ratio", "angular_floor", "radial_penalty"):
        output[field] = {"macro": np.mean(role_metric[field], axis=1)}
    micro_metric = scaled_metrics(
        role_metric["T"].sum(axis=1),
        role_metric["P_scaled"].sum(axis=1),
        role_metric["D_scaled"].sum(axis=1),
        1.0,
    )
    for field in output:
        output[field]["micro"] = micro_metric[field]
    return output


def build_bootstrap_tables(
    operator: pd.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    error_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    did_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    absolute_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "draws": draws,
        "bootstrap_unit": "12_transformer_blocks_with_all_roles_arms_calibrations_carried_together",
        "quantile_method": "numpy_linear",
        "panels": {},
    }
    exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
    for panel_id in PANELS:
        bootstrap_seed = BOOTSTRAP_SEEDS[panel_id]
        rng = np.random.default_rng(bootstrap_seed)
        sampled = rng.integers(0, 12, size=(draws, 12), dtype=np.int64)
        sampled_le = np.ascontiguousarray(sampled, dtype="<i8")
        manifest["panels"][panel_id] = {
            "seed": bootstrap_seed,
            "index_shape": list(sampled.shape),
            "index_sha256": hashlib.sha256(sampled_le.tobytes(order="C")).hexdigest(),
        }
        for score_split in SCORE_SPLITS:
            for tiling_seed in TILING_SEEDS:
                arrays = {
                    arm: _cell_stat_arrays(operator, panel_id, score_split, tiling_seed, arm)
                    for arm in ARMS
                }
                draw_metrics: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
                point_metrics: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
                for calibration in CALIBRATIONS:
                    for arm in ARMS:
                        draw_metrics[(calibration, arm)] = _scope_draw_metrics(
                            arrays[arm], sampled, calibration
                        )
                        point_metrics[(calibration, arm)] = _scope_draw_metrics(
                            arrays[arm], exact_depths, calibration
                        )
                base = {
                    "panel_id": panel_id,
                    "score_split": score_split,
                    "tiling_seed": tiling_seed,
                    "draws": draws,
                    "bootstrap_seed": bootstrap_seed,
                }
                for comparator in COMPARATORS:
                    for aggregation in AGGREGATIONS:
                        correct_e = draw_metrics[("source_role_gain", "correct")]["E"][aggregation]
                        control_e = draw_metrics[("source_role_gain", comparator)]["E"][aggregation]
                        point_correct = float(
                            point_metrics[("source_role_gain", "correct")]["E"][aggregation][0]
                        )
                        point_control = float(
                            point_metrics[("source_role_gain", comparator)]["E"][aggregation][0]
                        )
                        if point_correct <= 0.0 or np.any(correct_e <= 0.0):
                            raise RuntimeError("comparator/correct ratio is undefined for nonpositive correct E_X")
                        ratios = control_e / correct_e
                        ratio_l05 = quantile(ratios, 0.05)
                        error_rows.append(
                            {
                                **base,
                                "comparator": comparator,
                                "aggregation": aggregation,
                                "calibration": "source_role_gain",
                                "point_correct_E_X": point_correct,
                                "point_comparator_E_X": point_control,
                                "point_ratio_comparator_over_correct": point_control / point_correct,
                                "ratio_l05": ratio_l05,
                                "ratio_u95": quantile(ratios, 0.95),
                                "passes": bool(point_control / point_correct >= 1.05 and ratio_l05 > 1.0),
                            }
                        )
                        correct_cos = draw_metrics[("source_role_gain", "correct")]["cosine"][aggregation]
                        control_cos = draw_metrics[("source_role_gain", comparator)]["cosine"][aggregation]
                        delta_cos = correct_cos - control_cos
                        point_delta_cos = float(
                            point_metrics[("source_role_gain", "correct")]["cosine"][aggregation][0]
                            - point_metrics[("source_role_gain", comparator)]["cosine"][aggregation][0]
                        )
                        cosine_l05 = quantile(delta_cos, 0.05)
                        cosine_rows.append(
                            {
                                **base,
                                "comparator": comparator,
                                "aggregation": aggregation,
                                "calibration": "source_role_gain",
                                "point_delta_correct_minus_comparator": point_delta_cos,
                                "delta_l05": cosine_l05,
                                "delta_u95": quantile(delta_cos, 0.95),
                                "passes": bool(cosine_l05 > 0.0),
                            }
                        )

                for aggregation in AGGREGATIONS:
                    gain_difference = (
                        draw_metrics[("source_role_gain", "correct")]["E"][aggregation]
                        - draw_metrics[("source_role_gain", "zero_code")]["E"][aggregation]
                    )
                    raw_difference = (
                        draw_metrics[("raw", "correct")]["E"][aggregation]
                        - draw_metrics[("raw", "zero_code")]["E"][aggregation]
                    )
                    did = gain_difference - raw_difference
                    point_did = float(
                        (
                            point_metrics[("source_role_gain", "correct")]["E"][aggregation][0]
                            - point_metrics[("source_role_gain", "zero_code")]["E"][aggregation][0]
                        )
                        - (
                            point_metrics[("raw", "correct")]["E"][aggregation][0]
                            - point_metrics[("raw", "zero_code")]["E"][aggregation][0]
                        )
                    )
                    did_u95 = quantile(did, 0.95)
                    did_rows.append(
                        {
                            **base,
                            "aggregation": aggregation,
                            "point_difference_in_differences": point_did,
                            "did_l05": quantile(did, 0.05),
                            "did_u95": did_u95,
                            "passes": bool(did_u95 < 0.0),
                        }
                    )

                    role_e = draw_metrics[("source_role_gain", "correct")]["E"][aggregation]
                    common_e = draw_metrics[("source_median_common", "correct")]["E"][aggregation]
                    role_minus_common = role_e - common_e
                    point_role = float(
                        point_metrics[("source_role_gain", "correct")]["E"][aggregation][0]
                    )
                    point_common = float(
                        point_metrics[("source_median_common", "correct")]["E"][aggregation][0]
                    )
                    common_u95 = quantile(role_minus_common, 0.95)
                    common_rows.append(
                        {
                            **base,
                            "aggregation": aggregation,
                            "point_role_E_X": point_role,
                            "point_common_E_X": point_common,
                            "point_role_minus_common": point_role - point_common,
                            "delta_l05": quantile(role_minus_common, 0.05),
                            "delta_u95": common_u95,
                            "passes": bool(point_role - point_common < 0.0 and common_u95 < 0.0),
                        }
                    )

                    absolute = role_e
                    point_absolute = point_role
                    absolute_u95 = quantile(absolute, 0.95)
                    absolute_rows.append(
                        {
                            **base,
                            "aggregation": aggregation,
                            "point_E_X": point_absolute,
                            "E_X_l05": quantile(absolute, 0.05),
                            "E_X_u95": absolute_u95,
                            "literal_zero_E_X": 1.0,
                            "passes": bool(point_absolute < 1.0 and absolute_u95 < 1.0),
                        }
                    )
    outputs = tuple(
        pd.DataFrame(rows)
        for rows in (error_rows, cosine_rows, did_rows, common_rows, absolute_rows)
    )
    expected_counts = (32, 32, 16, 16, 16)
    actual_counts = tuple(len(frame) for frame in outputs)
    if actual_counts != expected_counts:
        raise RuntimeError(f"bootstrap table counts {actual_counts} != {expected_counts}")
    return (*outputs, manifest)


def build_role_gain_mappings(operator: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_gain_values = np.asarray([GAINS[role] for role in ROLES], dtype=np.float64)
    permutations = list(itertools.permutations(range(len(ROLES))))
    identity_tuple = tuple(range(len(ROLES)))
    rows: list[dict[str, Any]] = []
    for panel_id in PANELS:
        for score_split in SCORE_SPLITS:
            for tiling_seed in TILING_SEEDS:
                selected = operator[
                    (operator["panel_id"] == panel_id)
                    & (operator["score_split"] == score_split)
                    & (operator["tiling_seed"] == tiling_seed)
                    & (operator["arm"] == "correct")
                ]
                role_sums = selected.groupby("role", sort=False)[["T", "P", "D"]].sum().loc[list(ROLES)]
                target = role_sums["T"].to_numpy(dtype=np.float64)
                pred = role_sums["P"].to_numpy(dtype=np.float64)
                dot = role_sums["D"].to_numpy(dtype=np.float64)
                for mapping_id, permutation in enumerate(permutations):
                    assigned = source_gain_values[np.asarray(permutation, dtype=np.int64)]
                    metric = scaled_metrics(target, pred, dot, assigned)
                    macro = float(np.mean(metric["E"]))
                    micro = float(np.sum(metric["SSE"]) / np.sum(metric["T"]))
                    mapping = ";".join(
                        f"{target_role}<-{ROLES[source_index]}"
                        for target_role, source_index in zip(ROLES, permutation)
                    )
                    for aggregation, value in (("macro", macro), ("micro", micro)):
                        rows.append(
                            {
                                "panel_id": panel_id,
                                "score_split": score_split,
                                "tiling_seed": tiling_seed,
                                "aggregation": aggregation,
                                "mapping_id": mapping_id,
                                "mapping": mapping,
                                "is_identity": permutation == identity_tuple,
                                "E_X": value,
                            }
                        )
    frame = pd.DataFrame(rows)
    if len(frame) != 11_520:
        raise RuntimeError(f"role gain mapping row count {len(frame)} != 11520")
    summaries: list[dict[str, Any]] = []
    group_columns = ["panel_id", "score_split", "tiling_seed", "aggregation"]
    for key, selected in frame.groupby(group_columns, sort=True):
        if len(selected) != 720 or int(selected["is_identity"].sum()) != 1:
            raise RuntimeError(f"malformed 720-mapping group: {key}")
        identity = float(selected.loc[selected["is_identity"], "E_X"].iloc[0])
        values = selected["E_X"].to_numpy(dtype=np.float64)
        conservative_worst_rank, strictly_better, tied = conservative_rank(values, identity)
        summaries.append(
            {
                **dict(zip(group_columns, key)),
                "identity_E_X": identity,
                "strictly_better_count": strictly_better,
                "identity_tie_count_atol_1e_12": tied,
                "identity_conservative_worst_rank": conservative_worst_rank,
                "identity_top_5pct": bool(conservative_worst_rank <= 36),
                "best_E_X": float(values.min()),
                "median_E_X": float(np.median(values)),
                "worst_E_X": float(values.max()),
                "permutations": 720,
                "tie_atol": GAIN_TIE_ATOL,
            }
        )
    summary = pd.DataFrame(summaries)
    if len(summary) != 16:
        raise RuntimeError(f"role gain mapping summary row count {len(summary)} != 16")
    return frame, summary


def conservative_rank(values: np.ndarray, identity: float) -> tuple[int, int, int]:
    array = np.asarray(values, dtype=np.float64)
    strictly_better = int(np.sum(array < identity - GAIN_TIE_ATOL))
    tied = int(np.sum(np.abs(array - identity) <= GAIN_TIE_ATOL))
    if tied < 1:
        raise RuntimeError("identity value is absent from its own mapping distribution")
    return strictly_better + tied, strictly_better, tied


def choose_outcome(validity: bool, full: bool, directional: bool) -> str:
    if not validity:
        return "INVALID_PANEL"
    if full:
        return "FULL_REPLICATION_PASS"
    if directional:
        return "DIRECTIONAL_ONLY_OR_MIXED"
    return "MECHANISM_FAIL"


def _one_row(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def build_criteria_and_decisions(
    operator_aggregates: pd.DataFrame,
    error_effects: pd.DataFrame,
    cosine_effects: pd.DataFrame,
    did: pd.DataFrame,
    common: pd.DataFrame,
    absolute: pd.DataFrame,
    mapping_summary: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in error_effects.to_dict("records"):
        rows.append({"criterion": 1, "cell_type": "calibrated_error_ratio", **record})
    for record in cosine_effects.to_dict("records"):
        rows.append({"criterion": 2, "cell_type": "operator_cosine_effect", **record})

    for panel_id in PANELS:
        for score_split in SCORE_SPLITS:
            for tiling_seed in TILING_SEEDS:
                for role in ROLES:
                    correct = _one_row(
                        operator_aggregates,
                        panel_id=panel_id,
                        score_split=score_split,
                        tiling_seed=tiling_seed,
                        arm="correct",
                        calibration="source_role_gain",
                        aggregation=role,
                    )
                    zero_code = _one_row(
                        operator_aggregates,
                        panel_id=panel_id,
                        score_split=score_split,
                        tiling_seed=tiling_seed,
                        arm="zero_code",
                        calibration="source_role_gain",
                        aggregation=role,
                    )
                    passed = bool(float(correct.E) < float(zero_code.E) and float(correct.E) < 1.0)
                    rows.append(
                        {
                            "criterion": 3,
                            "cell_type": "role_correct_below_zero_code_and_literal_zero",
                            "panel_id": panel_id,
                            "score_split": score_split,
                            "tiling_seed": tiling_seed,
                            "aggregation": role,
                            "point_correct_E_X": float(correct.E),
                            "point_zero_code_E_X": float(zero_code.E),
                            "literal_zero_E_X": 1.0,
                            "passes": passed,
                        }
                    )
    for record in did.to_dict("records"):
        rows.append({"criterion": 4, "cell_type": "calibration_difference_in_differences", **record})

    for panel_id in PANELS:
        for score_split in SCORE_SPLITS:
            for tiling_seed in TILING_SEEDS:
                for role in RADIAL_ROLES:
                    calibrated = _one_row(
                        operator_aggregates,
                        panel_id=panel_id,
                        score_split=score_split,
                        tiling_seed=tiling_seed,
                        arm="correct",
                        calibration="source_role_gain",
                        aggregation=role,
                    )
                    raw = _one_row(
                        operator_aggregates,
                        panel_id=panel_id,
                        score_split=score_split,
                        tiling_seed=tiling_seed,
                        arm="correct",
                        calibration="raw",
                        aggregation=role,
                    )
                    rows.append(
                        {
                            "criterion": 5,
                            "cell_type": "role_radial_penalty_reduction",
                            "panel_id": panel_id,
                            "score_split": score_split,
                            "tiling_seed": tiling_seed,
                            "aggregation": role,
                            "raw_radial_penalty": float(raw.radial_penalty),
                            "calibrated_radial_penalty": float(calibrated.radial_penalty),
                            "point_delta_calibrated_minus_raw": float(
                                calibrated.radial_penalty - raw.radial_penalty
                            ),
                            "passes": bool(calibrated.radial_penalty < raw.radial_penalty),
                        }
                    )
    for record in common.to_dict("records"):
        rows.append({"criterion": 6, "cell_type": "source_role_gain_vs_common", **record})
    for record in mapping_summary.to_dict("records"):
        rows.append(
            {
                "criterion": 6,
                "cell_type": "identity_role_gain_mapping_rank",
                **record,
                "passes": bool(record["identity_top_5pct"]),
            }
        )
    for record in absolute.to_dict("records"):
        rows.append({"criterion": 7, "cell_type": "absolute_operator_quality", **record})

    criteria = pd.DataFrame(rows)
    expected_by_type = {
        "calibrated_error_ratio": 32,
        "operator_cosine_effect": 32,
        "role_correct_below_zero_code_and_literal_zero": 48,
        "calibration_difference_in_differences": 16,
        "role_radial_penalty_reduction": 24,
        "source_role_gain_vs_common": 16,
        "identity_role_gain_mapping_rank": 16,
        "absolute_operator_quality": 16,
    }
    observed_by_type = criteria["cell_type"].value_counts().to_dict()
    if observed_by_type != expected_by_type or len(criteria) != 200:
        raise RuntimeError(f"criterion count mismatch: {observed_by_type} total={len(criteria)}")
    if criteria["passes"].isna().any():
        raise RuntimeError("criterion table contains missing pass values")
    criteria["passes"] = criteria["passes"].astype(bool)

    panel_decisions: dict[str, Any] = {}
    panel_metadata = {str(row["panel_id"]): row for row in metadata["panels"]}
    for panel_id in PANELS:
        panel_rows = criteria[criteria["panel_id"] == panel_id]
        condition_passes = {
            str(condition): bool(panel_rows[panel_rows["criterion"] == condition]["passes"].all())
            for condition in range(1, 8)
        }
        condition_counts = {
            str(condition): {
                "passed": int(panel_rows[panel_rows["criterion"] == condition]["passes"].sum()),
                "total": int(len(panel_rows[panel_rows["criterion"] == condition])),
            }
            for condition in range(1, 8)
        }
        validity = all(
            bool(panel_metadata[panel_id][field])
            for field in ("provenance_pass", "resource_hash_pass", "quality_pass", "validity_pass")
        )
        full = validity and all(condition_passes.values())
        directional_error = error_effects[
            (error_effects["panel_id"] == panel_id)
            & (error_effects["comparator"] == "permuted_within_row")
        ]
        directional_cosine = cosine_effects[
            (cosine_effects["panel_id"] == panel_id)
            & (cosine_effects["comparator"] == "permuted_within_row")
        ]
        if len(directional_error) != 8 or len(directional_cosine) != 8:
            raise RuntimeError(f"directional cell count mismatch for {panel_id}")
        directional = bool(
            (directional_error["point_correct_E_X"] < directional_error["point_comparator_E_X"]).all()
            and (directional_cosine["delta_l05"] > 0.0).all()
        )
        label = choose_outcome(validity, full, directional)
        panel_decisions[panel_id] = {
            "outcome": label,
            "validity_pass": validity,
            "conditions": condition_passes,
            "condition_cell_counts": condition_counts,
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
        "panel_outcomes": {panel: panel_decisions[panel]["outcome"] for panel in PANELS},
        "both_panels_full_replication_pass": all(
            panel_decisions[panel]["outcome"] == "FULL_REPLICATION_PASS" for panel in PANELS
        ),
        "later_confirmatory_target_protocol_may_be_frozen": all(
            panel_decisions[panel]["outcome"] == "FULL_REPLICATION_PASS" for panel in PANELS
        ),
        "target_data_unsealed": False,
        "scope": "two unseen ViT-style vision encoders; not cross-domain evidence",
    }
    return criteria, panel_decisions, experiment


def expected_tiles_for_role(role: str) -> int:
    if role in {"attn_query", "attn_key", "attn_value", "attn_output"}:
        return 144
    if role in {"ffn_up", "ffn_down"}:
        return 576
    raise KeyError(role)


def expected_matrix_shape(role: str) -> tuple[int, int]:
    if role in {"attn_query", "attn_key", "attn_value", "attn_output"}:
        return 768, 768
    if role == "ffn_up":
        return 768, 3_072
    if role == "ffn_down":
        return 3_072, 768
    raise KeyError(role)


def _load_torch_cpu(path: Path) -> Any:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CPU analyzer unexpectedly exposes a CUDA device")
    return payload


def _validate_code_tensor(value: Any, expected_rows: int, require_float32: bool, label: str) -> np.ndarray:
    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{label} code is not a tensor")
    if value.device.type != "cpu":
        raise RuntimeError(f"{label} code is not on CPU")
    if tuple(value.shape) != (expected_rows, 512):
        raise RuntimeError(f"{label} code shape {tuple(value.shape)} != {(expected_rows, 512)}")
    if require_float32 and value.dtype != torch.float32:
        raise RuntimeError(f"{label} code dtype {value.dtype} != torch.float32")
    converted = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(converted).all()):
        raise RuntimeError(f"{label} code contains nonfinite values")
    return np.ascontiguousarray(converted.numpy(), dtype=np.float32)


def _validate_index_groups(
    groups: Any,
    dimension: int,
    label: str,
) -> tuple[str, int]:
    import torch

    expected_group_count = dimension // 64
    if not isinstance(groups, (list, tuple)) or len(groups) != expected_group_count:
        raise RuntimeError(
            f"{label} index groups count {len(groups) if isinstance(groups, (list, tuple)) else -1} "
            f"!= {expected_group_count}"
        )
    tensors: list[Any] = []
    for index, group in enumerate(groups):
        if not isinstance(group, torch.Tensor):
            raise RuntimeError(f"{label} group {index} is not a tensor")
        if group.device.type != "cpu" or group.dtype != torch.int64 or tuple(group.shape) != (64,):
            raise RuntimeError(
                f"{label} group {index} contract mismatch: device={group.device} dtype={group.dtype} shape={tuple(group.shape)}"
            )
        tensors.append(group.detach().cpu().contiguous())
    stacked = torch.stack(tensors).contiguous()
    array = np.ascontiguousarray(stacked.numpy(), dtype="<i8")
    flattened = array.reshape(-1)
    if not np.array_equal(np.sort(flattened), np.arange(dimension, dtype=np.int64)):
        raise RuntimeError(f"{label} index partition does not cover 0..{dimension - 1} exactly once")
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest(), expected_group_count


def load_and_validate_tiling_indices(
    input_dir: Path,
    matrix_map: Mapping[tuple[str, int, str], str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = input_dir / "tiling_indices.pt"
    payload = _load_torch_cpu(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != "prospective_tiling_indices_v1":
        raise RuntimeError("tiling index cache schema mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, dict) or len(entries) != 288:
        raise RuntimeError("tiling index cache must contain exactly 288 entries")
    expected_keys: set[str] = set()
    total_tiles = {panel: {seed: 0 for seed in TILING_SEEDS} for panel in PANELS}
    audited: dict[str, dict[str, Any]] = {}
    for panel_id in PANELS:
        for tiling_seed in TILING_SEEDS:
            for depth in DEPTHS:
                for role in ROLES:
                    key = f"{panel_id}|seed={tiling_seed}|depth={depth:02d}|role={role}"
                    expected_keys.add(key)
                    entry = entries.get(key)
                    if not isinstance(entry, dict):
                        raise RuntimeError(f"tiling index cache lacks {key}")
                    expected_fields = {
                        "panel_id": panel_id,
                        "tiling_seed": tiling_seed,
                        "depth": depth,
                        "role": role,
                        "matrix_key": matrix_map[(panel_id, depth, role)],
                    }
                    for field, expected in expected_fields.items():
                        if entry.get(field) != expected:
                            raise RuntimeError(f"tiling entry {key} field {field} mismatch")
                    d_in, d_out = expected_matrix_shape(role)
                    row_sha, row_groups = _validate_index_groups(entry.get("rows"), d_in, f"{key}/rows")
                    col_sha, col_groups = _validate_index_groups(entry.get("cols"), d_out, f"{key}/cols")
                    if entry.get("row_index_sha256") != row_sha:
                        raise RuntimeError(f"tiling row hash mismatch for {key}")
                    if entry.get("column_index_sha256") != col_sha:
                        raise RuntimeError(f"tiling column hash mismatch for {key}")
                    partition_sha = str(entry.get("partition_sha256", ""))
                    recomputed_partition_sha = hashlib.sha256(
                        f"{row_sha}|{col_sha}".encode("ascii")
                    ).hexdigest()
                    if partition_sha != recomputed_partition_sha:
                        raise RuntimeError(
                            f"tiling partition hash mismatch for {key}: {partition_sha} != {recomputed_partition_sha}"
                        )
                    tiles = row_groups * col_groups
                    if tiles != expected_tiles_for_role(role):
                        raise RuntimeError(f"tiling tile count mismatch for {key}: {tiles}")
                    total_tiles[panel_id][tiling_seed] += tiles
                    audited[key] = {
                        **expected_fields,
                        "row_index_sha256": row_sha,
                        "column_index_sha256": col_sha,
                        "partition_sha256": partition_sha,
                        "row_groups": row_groups,
                        "column_groups": col_groups,
                        "tiles": tiles,
                    }
    if set(entries) != expected_keys:
        raise RuntimeError("tiling index cache exact key grid mismatch")
    if any(total_tiles[panel][seed] != 20_736 for panel in PANELS for seed in TILING_SEEDS):
        raise RuntimeError(f"tiling index total tile counts mismatch: {total_tiles}")
    return audited, {
        "pass": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "entries": len(entries),
        "total_tiles": total_tiles,
        "row_and_column_hashes_recomputed": True,
        "coverage_exactly_once": True,
    }


def summarize_code_features(codes: np.ndarray) -> np.ndarray:
    value = np.asarray(codes, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 512 or value.shape[0] <= 0:
        raise RuntimeError(f"bad code array shape for geometry: {value.shape}")
    if not np.isfinite(value).all():
        raise RuntimeError("nonfinite code array in geometry")
    mean = value.mean(axis=0)
    std = value.std(axis=0, ddof=0)
    quantiles = np.quantile(value, (0.10, 0.50, 0.90), axis=0, method="linear")
    feature = np.concatenate([mean, std, quantiles[0], quantiles[1], quantiles[2]])
    if feature.shape != (2_560,) or not np.isfinite(feature).all():
        raise RuntimeError("geometry feature summary is malformed")
    return np.ascontiguousarray(feature, dtype=np.float64)


def _geometry_meta_record(
    panel_id: str,
    tiling_seed: int,
    tiling_index: int,
    depth: int,
    role: str,
    matrix_key: str,
    code_sha256: str,
) -> dict[str, Any]:
    panel_meta = PANEL_GEOMETRY_META[panel_id]
    depth_bin_index = min(depth // 3, 3)
    depth_bins = ("0-2", "3-5", "6-8", "9-11")
    return {
        "panel_id": panel_id,
        "panel_domain": panel_meta["panel_domain"],
        "dataset": panel_meta["dataset"],
        "model": panel_meta["model"],
        "tiling_seed": tiling_seed,
        "tiling_index": tiling_index,
        "depth": depth,
        "normalized_depth": depth / 11.0,
        "depth_bin": depth_bins[depth_bin_index],
        "role": role,
        "matrix_key": matrix_key,
        "code_sha256": code_sha256,
    }


def load_geometry_inputs(
    input_dir: Path,
    weight: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    new_cache_path = input_dir / "correct_latent_codes.pt"
    new_manifest_path = input_dir / "correct_latent_code_manifest.csv"
    new_payload = _load_torch_cpu(new_cache_path)
    if not isinstance(new_payload, dict):
        raise RuntimeError("new latent cache is not a mapping")
    if new_payload.get("schema_version") != "prospective_correct_latent_codes_v1":
        raise RuntimeError("new latent cache schema mismatch")
    entries = new_payload.get("entries")
    if not isinstance(entries, dict) or len(entries) != 288:
        raise RuntimeError(f"new latent cache must contain exactly 288 entries, got {type(entries)}/{len(entries) if isinstance(entries, dict) else -1}")

    matrix_map_frame = weight[["panel_id", "depth", "role", "matrix_key"]].drop_duplicates()
    matrix_map = {
        (str(row.panel_id), int(row.depth), str(row.role)): str(row.matrix_key)
        for row in matrix_map_frame.itertuples(index=False)
    }
    tiling_entries, tiling_audit = load_and_validate_tiling_indices(input_dir, matrix_map)
    features: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    cache_entry_audit: list[dict[str, Any]] = []
    expected_new_keys: set[str] = set()
    for panel_id in PANELS:
        for tiling_index, tiling_seed in enumerate(TILING_SEEDS, start=1):
            tile_total = 0
            for depth in DEPTHS:
                for role in ROLES:
                    cache_key = f"{panel_id}|seed={tiling_seed}|depth={depth:02d}|role={role}"
                    expected_new_keys.add(cache_key)
                    entry = entries.get(cache_key)
                    if not isinstance(entry, dict):
                        raise RuntimeError(f"new latent cache lacks entry {cache_key}")
                    expected_matrix_key = matrix_map[(panel_id, depth, role)]
                    exact_fields = {
                        "panel_id": panel_id,
                        "tiling_seed": tiling_seed,
                        "depth": depth,
                        "role": role,
                        "matrix_key": expected_matrix_key,
                    }
                    for field, expected in exact_fields.items():
                        if entry.get(field) != expected:
                            raise RuntimeError(
                                f"new latent entry {cache_key} field {field}: {entry.get(field)} != {expected}"
                            )
                    for field in ("tensor_sha256", "row_index_sha256", "column_index_sha256"):
                        if not HEX64.fullmatch(str(entry.get(field, ""))):
                            raise RuntimeError(f"new latent entry {cache_key} has malformed {field}")
                    tiling_entry = tiling_entries[cache_key]
                    if entry["row_index_sha256"] != tiling_entry["row_index_sha256"]:
                        raise RuntimeError(f"new latent row-index hash differs from tiling cache: {cache_key}")
                    if entry["column_index_sha256"] != tiling_entry["column_index_sha256"]:
                        raise RuntimeError(f"new latent column-index hash differs from tiling cache: {cache_key}")
                    expected_rows = expected_tiles_for_role(role)
                    codes = _validate_code_tensor(
                        entry.get("codes"), expected_rows, True, f"new/{cache_key}"
                    )
                    actual_sha = sha256_array(codes)
                    if actual_sha != entry["tensor_sha256"]:
                        raise RuntimeError(
                            f"new latent entry tensor hash mismatch {cache_key}: {actual_sha} != {entry['tensor_sha256']}"
                        )
                    if "codes_shape" in entry and list(entry["codes_shape"]) != [expected_rows, 512]:
                        raise RuntimeError(f"new latent entry codes_shape mismatch: {cache_key}")
                    if "codes_dtype" in entry and str(entry["codes_dtype"]) not in {"torch.float32", "float32"}:
                        raise RuntimeError(f"new latent entry codes_dtype mismatch: {cache_key}")
                    tile_total += expected_rows
                    features.append(summarize_code_features(codes))
                    metadata_rows.append(
                        _geometry_meta_record(
                            panel_id,
                            tiling_seed,
                            tiling_index,
                            depth,
                            role,
                            expected_matrix_key,
                            actual_sha,
                        )
                    )
                    cache_entry_audit.append(
                        {
                            "cache_key": cache_key,
                            **exact_fields,
                            "tensor_sha256": actual_sha,
                            "row_index_sha256": entry["row_index_sha256"],
                            "column_index_sha256": entry["column_index_sha256"],
                            "tiles": expected_rows,
                            "code_dim": 512,
                            "codes_dtype": "float32",
                        }
                    )
            if tile_total != 20_736:
                raise RuntimeError(f"new latent tile count {panel_id}/{tiling_seed}: {tile_total} != 20736")
    if set(entries) != expected_new_keys:
        extra = sorted(set(entries) - expected_new_keys)[:5]
        missing = sorted(expected_new_keys - set(entries))[:5]
        raise RuntimeError(f"new latent cache key grid mismatch: extra={extra} missing={missing}")

    manifest = pd.read_csv(new_manifest_path)
    manifest_columns = (
        "panel_id",
        "tiling_seed",
        "depth",
        "role",
        "matrix_key",
        "cache_key",
        "tensor_sha256",
        "row_index_sha256",
        "column_index_sha256",
        "tiles",
        "code_dim",
        "codes_dtype",
    )
    _require_columns(manifest, manifest_columns, "correct latent code manifest")
    if len(manifest) != 288 or manifest.duplicated(["cache_key"]).any():
        raise RuntimeError("correct latent code manifest is not a unique 288-entry grid")
    audit_frame = pd.DataFrame(cache_entry_audit)
    manifest_compare = manifest[list(manifest_columns)].copy()
    audit_compare = audit_frame[list(manifest_columns)].copy()
    for frame in (manifest_compare, audit_compare):
        frame["tiling_seed"] = frame["tiling_seed"].astype(np.int64)
        frame["depth"] = frame["depth"].astype(np.int64)
        frame["tiles"] = frame["tiles"].astype(np.int64)
        frame["code_dim"] = frame["code_dim"].astype(np.int64)
        for column in ("panel_id", "role", "matrix_key", "cache_key", "tensor_sha256", "row_index_sha256", "column_index_sha256", "codes_dtype"):
            frame[column] = frame[column].astype(str)
        frame["codes_dtype"] = frame["codes_dtype"].replace({"torch.float32": "float32"})
    manifest_compare = manifest_compare.sort_values("cache_key").reset_index(drop=True)
    audit_compare = audit_compare.sort_values("cache_key").reset_index(drop=True)
    if not manifest_compare.equals(audit_compare):
        raise RuntimeError("correct latent code manifest differs from independently audited PT entries")

    source_cache_audit: dict[str, Any] = {}
    for tiling_index, (tiling_seed, (cache_path, expected_sha)) in enumerate(SOURCE_CACHES.items(), start=1):
        actual_file_sha = sha256_file(cache_path.resolve(strict=True))
        if actual_file_sha != expected_sha:
            raise RuntimeError(f"source latent cache hash mismatch for {tiling_seed}")
        payload = _load_torch_cpu(cache_path)
        if not isinstance(payload, dict) or int(payload.get("tiling_seed", -1)) != tiling_seed:
            raise RuntimeError(f"source latent cache metadata mismatch for {tiling_seed}")
        codes_map = payload.get("codes")
        if not isinstance(codes_map, dict):
            raise RuntimeError(f"source latent cache lacks codes mapping for {tiling_seed}")
        selected_keys = {
            key for key in codes_map if str(key).endswith("|enc=cell|code=correct")
        }
        expected_source_keys: set[str] = set()
        tile_total = 0
        for depth in DEPTHS:
            for role in ROLES:
                key = f"seed={tiling_seed}|depth={depth}|role={role}|enc=cell|code=correct"
                expected_source_keys.add(key)
                expected_rows = expected_tiles_for_role(role)
                codes = _validate_code_tensor(
                    codes_map.get(key), expected_rows, False, f"source/{key}"
                )
                code_sha = sha256_array(codes)
                tile_total += expected_rows
                features.append(summarize_code_features(codes))
                metadata_rows.append(
                    _geometry_meta_record(
                        "source_vit_b_flickr",
                        tiling_seed,
                        tiling_index,
                        depth,
                        role,
                        f"source_vit_b_flickr|depth={depth:02d}|role={role}",
                        code_sha,
                    )
                )
        if selected_keys != expected_source_keys or tile_total != 20_736:
            raise RuntimeError(
                f"source cache exact correct-code grid mismatch seed={tiling_seed} keys={len(selected_keys)} tiles={tile_total}"
            )
        source_cache_audit[str(tiling_seed)] = {
            "path": str(cache_path.resolve()),
            "sha256": actual_file_sha,
            "bytes": cache_path.stat().st_size,
            "correct_entries": len(selected_keys),
            "correct_tiles": tile_total,
        }
        del payload

    metadata = pd.DataFrame(metadata_rows)
    feature_array = np.stack(features, axis=0)
    panel_counts = metadata["panel_id"].value_counts().to_dict()
    if len(metadata) != 432 or feature_array.shape != (432, 2_560):
        raise RuntimeError(f"geometry population shape mismatch: metadata={len(metadata)} features={feature_array.shape}")
    if panel_counts != {"beans": 144, "trocr_sroie": 144, "source_vit_b_flickr": 144}:
        raise RuntimeError(f"geometry panel row counts mismatch: {panel_counts}")
    audit = {
        "pass": True,
        "rows": len(metadata),
        "features": feature_array.shape[1],
        "panel_counts": panel_counts,
        "new_cache": {
            "path": str(new_cache_path.resolve()),
            "sha256": sha256_file(new_cache_path),
            "bytes": new_cache_path.stat().st_size,
            "entries": len(entries),
        },
        "new_manifest": {
            "path": str(new_manifest_path.resolve()),
            "sha256": sha256_file(new_manifest_path),
            "bytes": new_manifest_path.stat().st_size,
            "rows": len(manifest),
        },
        "tiling_indices": tiling_audit,
        "source_caches": source_cache_audit,
    }
    return metadata, feature_array, audit


def fit_fixed_geometry(
    metadata: pd.DataFrame,
    raw_features: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.decomposition import PCA

    features = np.asarray(raw_features, dtype=np.float64)
    if features.shape != (432, 2_560):
        raise RuntimeError(f"fixed PCA input shape {features.shape} != (432, 2560)")
    ordered_metadata = metadata.copy().reset_index(drop=True)
    grid_columns = ["panel_id", "tiling_index", "depth", "role"]
    if ordered_metadata.duplicated(grid_columns).any():
        raise RuntimeError("geometry metadata contains duplicate fixed-population rows")
    expected_grid = {
        (panel_id, tiling_index, depth, role)
        for panel_id in ("source_vit_b_flickr", "beans", "trocr_sroie")
        for tiling_index in (1, 2)
        for depth in DEPTHS
        for role in ROLES
    }
    observed_grid = set(ordered_metadata[grid_columns].itertuples(index=False, name=None))
    if observed_grid != expected_grid:
        raise RuntimeError("geometry metadata does not equal the canonical 432-row grid")
    ordered_metadata["_original_row"] = np.arange(432, dtype=np.int64)
    ordered_metadata["_panel_order"] = ordered_metadata["panel_id"].map(
        {"source_vit_b_flickr": 0, "beans": 1, "trocr_sroie": 2}
    )
    ordered_metadata["_role_order"] = ordered_metadata["role"].map(
        {role: index for index, role in enumerate(ROLES)}
    )
    ordered_metadata = ordered_metadata.sort_values(
        ["_panel_order", "tiling_index", "depth", "_role_order"], kind="mergesort"
    ).reset_index(drop=True)
    order = ordered_metadata["_original_row"].to_numpy(dtype=np.int64)
    features = features[order]
    ordered_metadata = ordered_metadata.drop(
        columns=["_original_row", "_panel_order", "_role_order"]
    )
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0, ddof=0)
    kept = feature_std >= 1e-12
    if int(kept.sum()) < 10:
        raise RuntimeError(f"only {int(kept.sum())} nondegenerate geometry features remain")
    standardized = (features[:, kept] - feature_mean[kept]) / feature_std[kept]
    if not np.isfinite(standardized).all():
        raise RuntimeError("standardized geometry features contain nonfinite values")
    pca = PCA(n_components=10, svd_solver="full")
    scores = pca.fit_transform(standardized)
    if scores.shape != (432, 10) or pca.components_.shape != (10, int(kept.sum())):
        raise RuntimeError("PCA output shape mismatch")
    score_frame = ordered_metadata
    for component in range(10):
        score_frame[f"PC{component + 1}"] = scores[:, component]
    write_csv(output_dir / "geometry_pca_scores.csv", score_frame)

    scaler = pd.DataFrame(
        {
            "feature_index": np.arange(2_560, dtype=np.int64),
            "feature_block": np.repeat(("mean", "std", "q10", "q50", "q90"), 512),
            "latent_coordinate": np.tile(np.arange(512, dtype=np.int64), 5),
            "population_mean": feature_mean,
            "population_std": feature_std,
            "kept": kept,
        }
    )
    write_csv(output_dir / "geometry_feature_scaler.csv", scaler)
    kept_indices = np.flatnonzero(kept)
    loading_rows = [
        {
            "component": component + 1,
            "feature_index": int(feature_index),
            "loading": float(pca.components_[component, local_index]),
        }
        for component in range(10)
        for local_index, feature_index in enumerate(kept_indices)
    ]
    write_csv(output_dir / "geometry_pca_loadings.csv", pd.DataFrame(loading_rows))
    explained = pd.DataFrame(
        {
            "component": np.arange(1, 11, dtype=np.int64),
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "singular_value": pca.singular_values_,
        }
    )
    write_csv(output_dir / "geometry_pca_explained_variance.csv", explained)
    np.savez_compressed(
        output_dir / "geometry_pca_inputs_and_model.npz",
        raw_features=features,
        feature_mean=feature_mean,
        feature_std=feature_std,
        kept_feature_mask=kept,
        standardized_features=standardized,
        pca_components=pca.components_,
        pca_scores=scores,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
    )
    audit = {
        "pass": True,
        "population_rows": len(score_frame),
        "raw_features": 2_560,
        "kept_features": int(kept.sum()),
        "dropped_features": int((~kept).sum()),
        "standardization_variance": "population_ddof_0",
        "feature_drop_threshold": 1e-12,
        "quantile_method": "FP64_numpy_linear",
        "canonical_row_order": "source_vit_b_flickr_then_beans_then_trocr_sroie;tiling_index;depth;frozen_role_order",
        "pca_components": 10,
        "svd_solver": "full",
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "explained_variance_ratio_sum_10": float(pca.explained_variance_ratio_.sum()),
    }
    return score_frame, audit


def _plot_categorical(scores: pd.DataFrame, column: str, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    categories = list(dict.fromkeys(scores[column].astype(str).tolist()))
    colors = {value: plt.get_cmap("tab10")(index % 10) for index, value in enumerate(categories)}
    markers = {1: "o", 2: "^"}
    fig, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    for category in categories:
        for tiling_index in (1, 2):
            selected = scores[
                (scores[column].astype(str) == category) & (scores["tiling_index"] == tiling_index)
            ]
            axis.scatter(
                selected["PC1"],
                selected["PC2"],
                s=38,
                alpha=0.78,
                color=colors[category],
                marker=markers[tiling_index],
                edgecolors="white",
                linewidths=0.35,
            )
    category_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[value], label=value, markersize=8)
        for value in categories
    ]
    marker_handles = [
        Line2D([0], [0], marker=markers[index], linestyle="", color="black", label=f"tiling {index}", markersize=8)
        for index in (1, 2)
    ]
    axis.legend(handles=[*category_handles, *marker_handles], loc="best", fontsize=9, frameon=True)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(title)
    axis.axhline(0.0, color="0.88", linewidth=0.8, zorder=0)
    axis.axvline(0.0, color="0.88", linewidth=0.8, zorder=0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_depth(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {1: "o", 2: "^"}
    fig, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    scatter = None
    for tiling_index in (1, 2):
        selected = scores[scores["tiling_index"] == tiling_index]
        scatter = axis.scatter(
            selected["PC1"],
            selected["PC2"],
            c=selected["normalized_depth"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=40,
            alpha=0.78,
            marker=markers[tiling_index],
            edgecolors="white",
            linewidths=0.3,
            label=f"tiling {tiling_index}",
        )
    if scatter is None:
        raise RuntimeError("empty depth geometry plot")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("normalized depth (depth / 11)")
    axis.legend(loc="best", fontsize=9)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("Fixed PCA coordinates coloured by normalized depth")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_role_facets(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = list(PANEL_GEOMETRY_META)
    colors = {panel: plt.get_cmap("tab10")(index) for index, panel in enumerate(panels)}
    markers = {1: "o", 2: "^"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    for axis, role in zip(axes.flat, ROLES):
        subset = scores[scores["role"] == role]
        for panel in panels:
            for tiling_index in (1, 2):
                selected = subset[
                    (subset["panel_id"] == panel) & (subset["tiling_index"] == tiling_index)
                ]
                axis.scatter(
                    selected["PC1"],
                    selected["PC2"],
                    s=40,
                    alpha=0.8,
                    color=colors[panel],
                    marker=markers[tiling_index],
                    edgecolors="white",
                    linewidths=0.3,
                )
        axis.set_title(role)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[panel], label=panel, markersize=8)
        for panel in panels
    ] + [
        Line2D([0], [0], marker=markers[index], linestyle="", color="black", label=f"tiling {index}", markersize=8)
        for index in (1, 2)
    ]
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.06, right=0.985, wspace=0.12, hspace=0.24)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=9)
    fig.suptitle("Fixed PC1/PC2 by six-role layer type, coloured by panel", fontsize=14, y=0.985)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_depth_bin_facets(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    depth_bins = ("0-2", "3-5", "6-8", "9-11")
    panels = list(PANEL_GEOMETRY_META)
    colors = {panel: plt.get_cmap("tab10")(index) for index, panel in enumerate(panels)}
    markers = {1: "o", 2: "^"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    for axis, depth_bin in zip(axes.flat, depth_bins):
        subset = scores[scores["depth_bin"] == depth_bin]
        for panel in panels:
            for tiling_index in (1, 2):
                selected = subset[
                    (subset["panel_id"] == panel) & (subset["tiling_index"] == tiling_index)
                ]
                axis.scatter(
                    selected["PC1"],
                    selected["PC2"],
                    s=40,
                    alpha=0.8,
                    color=colors[panel],
                    marker=markers[tiling_index],
                    edgecolors="white",
                    linewidths=0.3,
                )
        axis.set_title(f"depth {depth_bin}")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[panel], label=panel, markersize=8)
        for panel in panels
    ] + [
        Line2D([0], [0], marker=markers[index], linestyle="", color="black", label=f"tiling {index}", markersize=8)
        for index in (1, 2)
    ]
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.07, right=0.985, wspace=0.12, hspace=0.24)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=9)
    fig.suptitle("Fixed PC1/PC2 by four predeclared depth bins", fontsize=14, y=0.985)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def inspect_plot(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        width, height = image.size
    standard_deviation = float(rgb.astype(np.float64).std())
    if path.stat().st_size < 10_000 or width < 1_200 or height < 900 or standard_deviation < 2.0:
        raise RuntimeError(
            f"plot readability audit failed {path.name}: bytes={path.stat().st_size} size={width}x{height} std={standard_deviation}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "rgb_std": standard_deviation,
        "pass": True,
    }


def render_geometry_plots(scores: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    plot_paths = {
        "panel_domain": output_dir / "geometry_pc1_pc2_by_panel_domain.png",
        "dataset": output_dir / "geometry_pc1_pc2_by_dataset.png",
        "model": output_dir / "geometry_pc1_pc2_by_model.png",
        "role": output_dir / "geometry_pc1_pc2_by_role.png",
        "depth": output_dir / "geometry_pc1_pc2_by_depth.png",
        "role_facets": output_dir / "geometry_pc1_pc2_role_facets_by_panel.png",
        "depth_bin_facets": output_dir / "geometry_pc1_pc2_depth_bin_facets_by_panel.png",
    }
    _plot_categorical(
        scores,
        "panel_domain",
        "Fixed PCA coordinates coloured by panel/domain",
        plot_paths["panel_domain"],
    )
    _plot_categorical(scores, "dataset", "Fixed PCA coordinates coloured by dataset", plot_paths["dataset"])
    _plot_categorical(scores, "model", "Fixed PCA coordinates coloured by checkpoint/model", plot_paths["model"])
    _plot_categorical(scores, "role", "Fixed PCA coordinates coloured by layer role", plot_paths["role"])
    _plot_depth(scores, plot_paths["depth"])
    _plot_role_facets(scores, plot_paths["role_facets"])
    _plot_depth_bin_facets(scores, plot_paths["depth_bin_facets"])
    audits = {name: inspect_plot(path) for name, path in plot_paths.items()}
    return {"pass": all(row["pass"] for row in audits.values()), "plots": audits}


def _iter_local_file_records(value: Any, prefix: str) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield prefix, value
            return
        for key, child in value.items():
            yield from _iter_local_file_records(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_local_file_records(child, f"{prefix}[{index}]")


def _audit_contract_file_record(label: str, record: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(record.get("path", "")))
    if not raw_path.is_absolute():
        raw_path = PROJECT / raw_path
    path = reject_forbidden_path(raw_path, label).resolve(strict=True)
    allowed_roots = (PROJECT.resolve(strict=True), Path(sys.prefix).resolve(strict=True))
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise RuntimeError(f"contract local file {label} lies outside project/runtime roots: {path}")
    expected_sha = str(record.get("sha256", ""))
    if not HEX64.fullmatch(expected_sha):
        raise RuntimeError(f"contract local file {label} has malformed SHA-256")
    actual_sha = sha256_file(path)
    actual_bytes = path.stat().st_size
    if actual_sha != expected_sha:
        raise RuntimeError(f"contract local file hash mismatch {label}: {actual_sha} != {expected_sha}")
    if "bytes" in record and int(record["bytes"]) != actual_bytes:
        raise RuntimeError(f"contract local file size mismatch {label}")
    return {"label": label, "path": str(path), "sha256": actual_sha, "bytes": actual_bytes}


def validate_preexecution_contract(
    input_dir: Path,
    analyzer_sha256: str,
    runner_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = input_dir / "preexecution_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "prospective_geometry_matched_replication_preexecution_v1":
        raise RuntimeError("preexecution contract schema_version mismatch")
    required_hashes = (
        "design_sha256",
        "runner_sha256",
        "builder_sha256",
        "analyzer_sha256",
        "panel_manifest_sha256",
        "checkpoint_sha256",
    )
    for field in required_hashes:
        if not HEX64.fullmatch(str(payload.get(field, ""))):
            raise RuntimeError(f"preexecution contract has malformed {field}")
    if payload["design_sha256"] != FROZEN_DESIGN_SHA256:
        raise RuntimeError("preexecution contract design SHA mismatch")
    if payload["analyzer_sha256"] != analyzer_sha256:
        raise RuntimeError(
            f"preexecution contract analyzer SHA mismatch: {payload.get('analyzer_sha256')} != {analyzer_sha256}"
        )
    contract_sha = sha256_file(path)
    metadata_cross_checks = {field: payload[field] for field in BOUND_HASH_FIELDS}
    metadata_cross_checks["contract_sha256"] = contract_sha
    for field, expected in metadata_cross_checks.items():
        if runner_metadata.get(field) != expected:
            raise RuntimeError(
                f"runner metadata/preexecution cross-check failed for {field}: "
                f"{runner_metadata.get(field)} != {expected}"
            )
    required_objects = (
        "runtime",
        "execution_config",
        "resources",
        "model_dependencies",
        "scripts",
        "panel_cache",
        "expected_counts",
        "dataflow_contract",
        "target_seal",
    )
    for field in required_objects:
        if not isinstance(payload.get(field), dict) or not payload[field]:
            raise RuntimeError(f"preexecution contract lacks nonempty mapping {field}")
    identity_binding = _validate_contract_metadata_identity(payload, runner_metadata)

    scripts = payload["scripts"]
    if set(scripts) != {"runner", "builder", "analyzer"}:
        raise RuntimeError(f"preexecution scripts grid mismatch: {set(scripts)}")
    expected_script_hashes = {
        "runner": payload["runner_sha256"],
        "builder": payload["builder_sha256"],
        "analyzer": payload["analyzer_sha256"],
    }
    audited_records: list[dict[str, Any]] = []
    for script_name, expected_sha in expected_script_hashes.items():
        record = scripts[script_name]
        if not isinstance(record, dict) or record.get("sha256") != expected_sha:
            raise RuntimeError(f"preexecution script hash cross-check failed for {script_name}")
        audited = _audit_contract_file_record(f"scripts.{script_name}", record)
        audited_records.append(audited)
        if script_name == "analyzer" and Path(audited["path"]) != Path(__file__).resolve(strict=True):
            raise RuntimeError("preexecution analyzer path differs from the running analyzer")

    resource_records = list(_iter_local_file_records(payload["resources"], "resources"))
    dependency_records = list(
        _iter_local_file_records(payload["model_dependencies"], "model_dependencies")
    )
    panel_records = list(_iter_local_file_records(payload["panel_cache"], "panel_cache"))
    if not resource_records or not dependency_records or not panel_records:
        raise RuntimeError(
            "preexecution contract must expose local resource, dependency, and panel-manifest file records"
        )
    for label, record in (*resource_records, *dependency_records, *panel_records):
        if "bytes" not in record:
            raise RuntimeError(f"preexecution local resource lacks byte count: {label}")
        audited_records.append(_audit_contract_file_record(label, record))
    audited_hashes = {record["sha256"] for record in audited_records}
    if payload["panel_manifest_sha256"] not in audited_hashes:
        raise RuntimeError("panel_manifest_sha256 is not bound to a rehashed local file")
    if payload["checkpoint_sha256"] not in audited_hashes:
        raise RuntimeError("checkpoint_sha256 is not bound to a rehashed local file")
    panel_cache = payload["panel_cache"]
    if panel_cache.get("manifest_sha256") != payload["panel_manifest_sha256"]:
        raise RuntimeError("top-level panel_manifest_sha256 differs from panel_cache.manifest_sha256")
    resources = payload["resources"]
    ae_checkpoint = resources.get("ae_checkpoint")
    if not isinstance(ae_checkpoint, Mapping) or ae_checkpoint.get("sha256") != payload["checkpoint_sha256"]:
        raise RuntimeError("top-level checkpoint_sha256 differs from resources.ae_checkpoint.sha256")

    expected_counts = payload["expected_counts"]
    required_counts = {
        "panels": 2,
        "depths": 12,
        "roles": 6,
        "matrices_per_panel": 72,
        "tilings": 2,
        "score_splits": 2,
        "arms": 3,
        "tiles_per_panel_tiling": 20_736,
        "decoded_full_predictions_per_panel": 432,
        "decoded_full_predictions_total": 864,
        "weight_sufficient_stat_rows": 864,
        "operator_sufficient_stat_rows": 1_728,
        "correct_latent_cache_entries": 288,
        "tiling_manifest_rows": 288,
        "permutation_manifest_rows": 5_184,
        "primary_calibrated_error_rows": 32,
        "directional_criterion_rows": 32,
        "operator_aggregate_rows": 576,
        "weight_aggregate_rows": 288,
        "criterion1_error_rows": 32,
        "criterion2_cosine_rows": 32,
        "criterion3_role_rows": 48,
        "criterion4_did_rows": 16,
        "criterion5_radial_rows": 24,
        "criterion6_common_rows": 16,
        "criterion6_mapping_summary_rows": 16,
        "criterion6_mapping_full_rows": 11_520,
        "criterion7_absolute_rows": 16,
        "criterion_cells_rows": 200,
        "geometry_rows": 432,
        "geometry_plots": 7,
    }
    for field, expected in required_counts.items():
        if int(expected_counts.get(field, -1)) != expected:
            raise RuntimeError(
                f"preexecution expected count mismatch {field}: {expected_counts.get(field)} != {expected}"
            )
    expected_role_tiles = {role: expected_tiles_for_role(role) for role in ROLES}
    if expected_counts.get("tiles_per_role_matrix") != expected_role_tiles:
        raise RuntimeError(
            f"preexecution tiles_per_role_matrix mismatch: {expected_counts.get('tiles_per_role_matrix')} "
            f"!= {expected_role_tiles}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": contract_sha,
        "bytes": path.stat().st_size,
        "design_sha256": payload["design_sha256"],
        "analyzer_sha256": payload["analyzer_sha256"],
        "runner_sha256": payload["runner_sha256"],
        "builder_sha256": payload["builder_sha256"],
        "panel_manifest_sha256": payload["panel_manifest_sha256"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "metadata_contract_identity_binding": identity_binding,
        "local_files_rehashed": audited_records,
        "expected_counts_checked": required_counts,
        "tiles_per_role_matrix_checked": expected_role_tiles,
        "pass": True,
    }


def validate_preexecution_binding(
    input_dir: Path,
    contract_audit: Mapping[str, Any],
    runner_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    binding_path = input_dir / "preexecution_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("schema_version") != "prospective_geometry_matched_replication_binding_v1":
        raise RuntimeError("preexecution binding schema_version mismatch")
    _require_exact_mapping_keys(binding, BINDING_KEYS, "preexecution binding")
    contract_path = (input_dir / "preexecution_contract.json").resolve(strict=True)

    def resolve_bound_path(raw: Any, label: str) -> Path:
        value = Path(str(raw))
        if not value.is_absolute():
            value = PROJECT / value
        return reject_forbidden_path(value, label).resolve(strict=True)

    external_path = resolve_bound_path(binding["external_contract_path"], "external contract")
    copied_path = resolve_bound_path(binding["copied_contract_path"], "copied contract")
    require_within(external_path, PROJECT, "external contract", strict=True)
    require_within(copied_path, FORMAL_ARTIFACT_ROOT, "copied contract", strict=True)
    if copied_path != contract_path:
        raise RuntimeError(f"binding copied contract path mismatch: {copied_path} != {contract_path}")
    external_sha = sha256_file(external_path)
    copied_sha = sha256_file(copied_path)
    contract_sha = str(contract_audit["sha256"])
    if not (external_sha == copied_sha == contract_sha):
        raise RuntimeError("external/copied/metadata preexecution contract SHA binding mismatch")
    _validate_contract_digest_binding(contract_sha, runner_metadata, binding)
    if external_path.read_bytes() != copied_path.read_bytes():
        raise RuntimeError("external and copied preexecution contracts are not byte-identical")

    contract_payload = json.loads(copied_path.read_text(encoding="utf-8"))
    three_way_identity = _validate_three_way_binding_identity(
        contract_payload,
        runner_metadata,
        binding,
    )
    return {
        "path": str(binding_path.resolve()),
        "sha256": sha256_file(binding_path),
        "bytes": binding_path.stat().st_size,
        "external_contract_path": str(external_path),
        "external_contract_sha256": external_sha,
        "copied_contract_path": str(copied_path),
        "copied_contract_sha256": copied_sha,
        "contracts_byte_identical": True,
        "scalar_and_resource_bindings_match": True,
        "metadata_contract_binding_identity": three_way_identity,
        "pass": True,
    }


def validate_sufficient_statistics_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "prospective_sufficient_statistics_v1":
        raise RuntimeError("sufficient-statistic schema_version mismatch")
    expected_files = {
        "weight": {
            "filename": "weight_sufficient_stats.csv",
            "row_count": 864,
            "key_columns": ["panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm"],
            "space": "target=W; prediction=W_hat",
        },
        "operator": {
            "filename": "operator_sufficient_stats.csv",
            "row_count": 1_728,
            "key_columns": [
                "panel_id",
                "score_split",
                "tiling_seed",
                "depth",
                "role",
                "matrix_key",
                "arm",
            ],
            "space": "target=X@W; prediction=X@W_hat",
        },
    }
    if payload.get("files") != expected_files:
        raise RuntimeError(f"sufficient-statistic file schema mismatch: {payload.get('files')}")
    expected_statistics = {
        "T": {"dtype": "float64", "semantic": "squared_target_norm"},
        "P": {"dtype": "float64", "semantic": "squared_prediction_norm"},
        "D": {"dtype": "float64", "semantic": "target_prediction_inner_product"},
    }
    if payload.get("canonical_statistics") != expected_statistics:
        raise RuntimeError("canonical T/P/D statistic semantics mismatch")
    expected_aliases = {
        "target_energy": "T",
        "pred_energy": "P",
        "target_pred_dot": "D",
    }
    if payload.get("checked_aliases") != expected_aliases:
        raise RuntimeError("sufficient-statistic alias contract mismatch")
    expected_numeric = {
        "decision_columns": ["T", "P", "D"],
        "aliases_are_not_independent_values": True,
        "accumulation_dtype": "float64",
        "float_serialization": "Python float repr, IEEE-754 binary64 decimal round-trip",
        "finite": True,
        "T_positive": True,
        "P_positive": True,
    }
    if payload.get("numeric_contract") != expected_numeric:
        raise RuntimeError("sufficient-statistic numeric contract mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "schema_version": payload["schema_version"],
        "files": expected_files,
        "canonical_statistics": expected_statistics,
        "checked_aliases": expected_aliases,
        "numeric_contract": expected_numeric,
        "pass": True,
    }


def validate_raw_run_manifest(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "artifact_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("raw runner artifact manifest itself must not be a symlink")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_exact_mapping_keys(payload, RAW_RUN_MANIFEST_KEYS, "raw runner artifact manifest")
    if payload["schema_version"] != "prospective_geometry_matched_replication_artifact_manifest_v1":
        raise RuntimeError("raw runner artifact manifest schema_version mismatch")
    rows = payload.get("artifacts")
    if not isinstance(rows, list):
        raise RuntimeError("raw runner artifact manifest lacks an artifacts list")
    if isinstance(payload["count"], bool) or not isinstance(payload["count"], int):
        raise RuntimeError("raw runner artifact manifest count must be an integer")
    if payload["count"] != len(rows):
        raise RuntimeError("raw runner artifact manifest count field mismatch")
    if payload["manifest_self_excluded"] is not True:
        raise RuntimeError("raw runner artifact manifest self-exclusion flag is false")
    declared: set[str] = set()
    for row in rows:
        _require_exact_mapping_keys(row, RAW_RUN_MANIFEST_ROW_KEYS, "raw runner artifact manifest row")
        relative_value = row["path"]
        if not isinstance(relative_value, str):
            raise RuntimeError("raw runner artifact manifest row path is not a string")
        relative = relative_value
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or contains_forbidden_target_marker(relative)
        ):
            raise RuntimeError(f"unsafe raw runner manifest path: {relative}")
        if relative == "artifact_manifest.json":
            raise RuntimeError("raw runner manifest must exclude itself")
        if relative in declared:
            raise RuntimeError(f"duplicate raw runner manifest path: {relative}")
        declared.add(relative)
        unresolved_path = input_dir / relative_path
        cursor = input_dir
        for component in relative_path.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise RuntimeError(f"raw runner manifest must not follow symlinks: {relative}")
        path = unresolved_path.resolve(strict=True)
        try:
            path.relative_to(input_dir)
        except ValueError as error:
            raise RuntimeError(f"raw runner manifest path escapes input directory: {relative}") from error
        if not path.is_file():
            raise RuntimeError(f"raw runner manifest entry is not a regular file: {relative}")
        if not HEX64.fullmatch(str(row["sha256"])):
            raise RuntimeError(f"raw runner artifact manifest row has malformed SHA-256: {relative}")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise RuntimeError(f"raw runner artifact manifest row has invalid bytes: {relative}")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        if row["sha256"] != actual_sha or row["bytes"] != actual_bytes:
            raise RuntimeError(
                f"raw runner artifact mismatch {relative}: sha={actual_sha} bytes={actual_bytes}"
            )
    tree_entries = list(input_dir.rglob("*"))
    symlinks = [path.relative_to(input_dir).as_posix() for path in tree_entries if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"raw runner directory contains prohibited symlinks: {symlinks[:5]}")
    actual_files = {
        path.relative_to(input_dir).as_posix()
        for path in tree_entries
        if path.is_file() and path != manifest_path
    }
    if declared != actual_files:
        raise RuntimeError(
            f"raw runner manifest completeness mismatch: missing={sorted(actual_files - declared)[:5]} "
            f"stale={sorted(declared - actual_files)[:5]}"
        )
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "declared_artifacts": len(rows),
        "complete": True,
        "manifest_self_excluded": True,
        "pass": True,
    }


def artifact_manifest(output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative == "artifact_manifest.json":
            continue
        rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "schema_version": "prospective_geometry_matched_analyzer_manifest_v1",
        "manifest_self_excluded": True,
        "artifacts": rows,
        "count": len(rows),
    }


def build_readme(
    output_dir: Path,
    panel_decisions: Mapping[str, Any],
    experiment: Mapping[str, Any],
    criteria: pd.DataFrame,
    plot_audit: Mapping[str, Any],
) -> None:
    lines = [
        "# Prospective geometry-matched source replication: independent analysis",
        "",
        f"Frozen design SHA-256: {FROZEN_DESIGN_SHA256}.",
        "",
        "This report was reconstructed on CPU from sealed FP64 sufficient statistics. The analyzer did not import or execute the Weight-AE.",
        "",
        "## Outcome",
        "",
    ]
    for panel_id in PANELS:
        decision = panel_decisions[panel_id]
        lines.append(f"- {panel_id}: {decision['outcome']}")
        for condition in range(1, 8):
            count = decision["condition_cell_counts"][str(condition)]
            lines.append(
                f"  - condition {condition}: {count['passed']}/{count['total']} cells pass"
            )
    lines.extend(
        [
            "",
            f"Both panels full pass: {experiment['both_panels_full_replication_pass']}.",
            f"Later target protocol may be frozen: {experiment['later_confirmatory_target_protocol_may_be_frozen']}.",
            "Target data remain sealed.",
            "",
            "## Evidence",
            "",
            f"- criterion_cells.csv contains {len(criteria)} exact decision cells.",
            "- operator_aggregates.csv and weight_aggregates.csv contain role, macro, and role-aware micro metrics.",
            "- criterion1_error_bootstrap.csv and criterion2_cosine_bootstrap.csv contain the primary paired intervals.",
            "- role_gain_mappings.csv contains every one of the 720 mappings in every formal cell.",
            f"- Seven fixed PCA-only exploratory views passed the mechanical readability audit: {plot_audit['pass']}.",
            "",
            "## Scope",
            "",
            "Intervals are conditional on the fixed checkpoints, score images/tokens, and source gain fit. A/B and tilings are robustness repeats, not independent sample size. Geometry is exploratory and non-gating. This experiment concerns two unseen ViT-style vision encoders and is not cross-domain evidence.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _synthetic_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _expect_runtime_rejection(label: str, callback: Any) -> str:
    try:
        callback()
    except RuntimeError:
        return label
    raise AssertionError(f"synthetic corruption was not rejected: {label}")


def _synthetic_identity_binding_tests(temp: Path) -> dict[str, Any]:
    panel_records: dict[str, dict[str, Any]] = {}
    for panel_id in PANELS:
        panel_path = (temp / f"identity_{panel_id}.bin").resolve()
        panel_path.write_bytes(f"synthetic panel {panel_id}\n".encode("utf-8"))
        panel_records[panel_id] = {
            "panel_id": panel_id,
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "checkpoint_sha256": _synthetic_hash("panel checkpoint", panel_id),
        }
    hashes = {field: _synthetic_hash("top-level", field) for field in BOUND_HASH_FIELDS}
    contract: dict[str, Any] = {
        **hashes,
        "scripts": {"fixture": "scripts"},
        "model_dependencies": {"fixture": "dependencies"},
        "resources": {"fixture": "resources"},
        "panel_cache": {
            "manifest_sha256": hashes["panel_manifest_sha256"],
            "panels": panel_records,
        },
    }
    contract_digest = _synthetic_hash("copied preexecution contract")
    metadata: dict[str, Any] = {
        **hashes,
        "contract_sha256": contract_digest,
        "panels": [
            {
                **panel_records[panel_id],
                "provenance_pass": True,
                "resource_hash_pass": True,
                "quality_pass": True,
                "validity_pass": True,
            }
            for panel_id in PANELS
        ],
    }
    binding: dict[str, Any] = {
        "schema_version": "prospective_geometry_matched_replication_binding_v1",
        "external_contract_path": str((temp / "external_contract.json").resolve()),
        "external_contract_sha256": contract_digest,
        "copied_contract_path": str((temp / "copied_contract.json").resolve()),
        "copied_contract_sha256": contract_digest,
        **hashes,
        "scripts": contract["scripts"],
        "model_dependencies": contract["model_dependencies"],
        "resources": contract["resources"],
        "panel_cache": contract["panel_cache"],
    }
    contract_metadata_audit = _validate_contract_metadata_identity(contract, metadata)
    three_way_audit = _validate_three_way_binding_identity(contract, metadata, binding)
    _validate_contract_digest_binding(contract_digest, metadata, binding)

    metadata_hash_rejections: list[str] = []
    for field in BOUND_HASH_FIELDS:
        corrupted = json.loads(json.dumps(metadata))
        corrupted[field] = _synthetic_hash("valid-hex metadata corruption", field)
        metadata_hash_rejections.append(
            _expect_runtime_rejection(
                field,
                lambda corrupted=corrupted: _validate_contract_metadata_identity(
                    contract, corrupted
                ),
            )
        )
    corrupted = json.loads(json.dumps(metadata))
    corrupted["contract_sha256"] = _synthetic_hash("valid-hex metadata contract digest corruption")
    metadata_hash_rejections.append(
        _expect_runtime_rejection(
            "contract_sha256",
            lambda: _validate_contract_digest_binding(contract_digest, corrupted, binding),
        )
    )

    metadata_panel_rejections: list[str] = []
    corrupted = json.loads(json.dumps(metadata))
    corrupted["panels"][0]["panel_id"] = "unexpected_panel"
    metadata_panel_rejections.append(
        _expect_runtime_rejection(
            "panel_grid",
            lambda: _validate_contract_metadata_identity(contract, corrupted),
        )
    )
    corrupted = json.loads(json.dumps(metadata))
    corrupted["panels"][0]["unexpected_field"] = True
    metadata_panel_rejections.append(
        _expect_runtime_rejection(
            "panel_exact_keys",
            lambda: _validate_contract_metadata_identity(contract, corrupted),
        )
    )
    for panel_index, panel_id in enumerate(PANELS):
        other_panel = PANELS[1 - panel_index]
        for field in ("path", "sha256", "checkpoint_sha256"):
            corrupted = json.loads(json.dumps(metadata))
            if field == "path":
                corrupted["panels"][panel_index][field] = panel_records[other_panel][field]
            else:
                corrupted["panels"][panel_index][field] = _synthetic_hash(
                    "valid-hex metadata panel corruption", panel_id, field
                )
            label = f"{panel_id}.{field}"
            metadata_panel_rejections.append(
                _expect_runtime_rejection(
                    label,
                    lambda corrupted=corrupted: _validate_contract_metadata_identity(
                        contract, corrupted
                    ),
                )
            )

    binding_hash_rejections: list[str] = []
    for field in BOUND_HASH_FIELDS:
        corrupted = json.loads(json.dumps(binding))
        corrupted[field] = _synthetic_hash("valid-hex binding corruption", field)
        binding_hash_rejections.append(
            _expect_runtime_rejection(
                field,
                lambda corrupted=corrupted: _validate_three_way_binding_identity(
                    contract, metadata, corrupted
                ),
            )
        )
    for field in ("external_contract_sha256", "copied_contract_sha256"):
        corrupted = json.loads(json.dumps(binding))
        corrupted[field] = _synthetic_hash("valid-hex binding contract digest corruption", field)
        binding_hash_rejections.append(
            _expect_runtime_rejection(
                field,
                lambda corrupted=corrupted: _validate_contract_digest_binding(
                    contract_digest, metadata, corrupted
                ),
            )
        )

    binding_panel_rejections: list[str] = []
    for panel_index, panel_id in enumerate(PANELS):
        other_panel = PANELS[1 - panel_index]
        for field in ("path", "sha256", "checkpoint_sha256"):
            corrupted = json.loads(json.dumps(binding))
            if field == "path":
                corrupted["panel_cache"]["panels"][panel_id][field] = panel_records[other_panel][field]
            else:
                corrupted["panel_cache"]["panels"][panel_id][field] = _synthetic_hash(
                    "valid-hex binding panel corruption", panel_id, field
                )
            label = f"{panel_id}.{field}"
            binding_panel_rejections.append(
                _expect_runtime_rejection(
                    label,
                    lambda corrupted=corrupted: _validate_three_way_binding_identity(
                        contract, metadata, corrupted
                    ),
                )
            )
    corrupted = json.loads(json.dumps(binding))
    corrupted["unexpected_field"] = True
    binding_header_rejection = _expect_runtime_rejection(
        "binding_exact_keys",
        lambda: _validate_three_way_binding_identity(contract, metadata, corrupted),
    )

    return {
        "contract_metadata_baseline": contract_metadata_audit,
        "three_way_baseline": three_way_audit,
        "metadata_hash_corruptions_rejected": metadata_hash_rejections,
        "metadata_panel_corruptions_rejected": metadata_panel_rejections,
        "binding_hash_corruptions_rejected": binding_hash_rejections,
        "binding_panel_corruptions_rejected": binding_panel_rejections,
        "binding_header_corruption_rejected": binding_header_rejection,
        "pass": True,
    }


def _synthetic_raw_manifest_tests(temp: Path) -> dict[str, Any]:
    def initialize_case(label: str) -> tuple[Path, dict[str, Any]]:
        case_dir = temp / f"raw_manifest_{label}"
        case_dir.mkdir()
        payload_path = case_dir / "payload.bin"
        payload_path.write_bytes(f"synthetic raw manifest payload {label}\n".encode("utf-8"))
        manifest = {
            "schema_version": "prospective_geometry_matched_replication_artifact_manifest_v1",
            "manifest_self_excluded": True,
            "artifacts": [
                {
                    "path": "payload.bin",
                    "sha256": sha256_file(payload_path),
                    "bytes": payload_path.stat().st_size,
                }
            ],
            "count": 1,
        }
        return case_dir, manifest

    valid_dir, valid_manifest = initialize_case("valid")
    write_json(valid_dir / "artifact_manifest.json", valid_manifest)
    valid_audit = validate_raw_run_manifest(valid_dir)

    rejected: list[str] = []

    schema_dir, manifest = initialize_case("schema")
    manifest["schema_version"] = "wrong_schema"
    write_json(schema_dir / "artifact_manifest.json", manifest)
    rejected.append(
        _expect_runtime_rejection("schema_version", lambda: validate_raw_run_manifest(schema_dir))
    )

    header_dir, manifest = initialize_case("header")
    manifest["unexpected_field"] = True
    write_json(header_dir / "artifact_manifest.json", manifest)
    rejected.append(
        _expect_runtime_rejection("exact_top_level_keys", lambda: validate_raw_run_manifest(header_dir))
    )

    row_dir, manifest = initialize_case("row")
    manifest["artifacts"][0]["unexpected_field"] = True
    write_json(row_dir / "artifact_manifest.json", manifest)
    rejected.append(
        _expect_runtime_rejection("exact_row_keys", lambda: validate_raw_run_manifest(row_dir))
    )

    forbidden_dir, manifest = initialize_case("forbidden")
    manifest["artifacts"][0]["path"] = "target-probe.bin"
    write_json(forbidden_dir / "artifact_manifest.json", manifest)
    rejected.append(
        _expect_runtime_rejection("forbidden_path", lambda: validate_raw_run_manifest(forbidden_dir))
    )

    symlink_dir, manifest = initialize_case("symlink")
    symlink_path = symlink_dir / "alias.bin"
    symlink_path.symlink_to("payload.bin")
    manifest["artifacts"].append(
        {
            "path": "alias.bin",
            "sha256": sha256_file(symlink_dir / "payload.bin"),
            "bytes": (symlink_dir / "payload.bin").stat().st_size,
        }
    )
    manifest["count"] = 2
    write_json(symlink_dir / "artifact_manifest.json", manifest)
    rejected.append(
        _expect_runtime_rejection("declared_symlink", lambda: validate_raw_run_manifest(symlink_dir))
    )

    manifest_symlink_dir, manifest = initialize_case("manifest_link")
    real_manifest = manifest_symlink_dir / "real_manifest.json"
    write_json(real_manifest, manifest)
    (manifest_symlink_dir / "artifact_manifest.json").symlink_to("real_manifest.json")
    rejected.append(
        _expect_runtime_rejection(
            "manifest_symlink", lambda: validate_raw_run_manifest(manifest_symlink_dir)
        )
    )

    completeness_dir, manifest = initialize_case("completeness")
    (completeness_dir / "undeclared.bin").write_bytes(b"undeclared\n")
    write_json(completeness_dir / "artifact_manifest.json", manifest)
    rejected.append(
        _expect_runtime_rejection(
            "self_excluding_completeness", lambda: validate_raw_run_manifest(completeness_dir)
        )
    )

    return {
        "valid_manifest_audit": valid_audit,
        "corruptions_rejected": rejected,
        "expected_rejection_count": 7,
        "pass": len(rejected) == 7,
    }


def build_synthetic_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    cosine_by_arm = {
        "correct": 0.94,
        "permuted_within_row": 0.40,
        "zero_code": 0.10,
    }
    weight_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    for panel_index, panel_id in enumerate(PANELS):
        for tiling_seed in TILING_SEEDS:
            for depth in DEPTHS:
                for role_index, role in enumerate(ROLES):
                    matrix_key = f"{panel_id}|depth={depth:02d}|role={role}"
                    gain = GAINS[role]
                    weight_target = 1_000.0 + 13.0 * panel_index + 7.0 * role_index + depth
                    for arm in ARMS:
                        cosine = cosine_by_arm[arm]
                        radial_ratio = cosine / gain
                        weight_pred = weight_target * radial_ratio * radial_ratio
                        weight_dot = cosine * math.sqrt(weight_target * weight_pred)
                        prediction_sha = _synthetic_hash(
                            "prediction", panel_id, tiling_seed, depth, role, arm
                        )
                        weight_rows.append(
                            {
                                "panel_id": panel_id,
                                "tiling_seed": tiling_seed,
                                "depth": depth,
                                "role": role,
                                "matrix_key": matrix_key,
                                "arm": arm,
                                "prediction_sha256": prediction_sha,
                                "T": weight_target,
                                "P": weight_pred,
                                "D": weight_dot,
                            }
                        )
                        for split_index, score_split in enumerate(SCORE_SPLITS):
                            operator_target = weight_target * (1.5 + 0.2 * split_index)
                            operator_pred = operator_target * radial_ratio * radial_ratio
                            operator_dot = cosine * math.sqrt(operator_target * operator_pred)
                            operator_rows.append(
                                {
                                    "panel_id": panel_id,
                                    "tiling_seed": tiling_seed,
                                    "depth": depth,
                                    "role": role,
                                    "matrix_key": matrix_key,
                                    "arm": arm,
                                    "prediction_sha256": prediction_sha,
                                    "score_split": score_split,
                                    "activation_sha256": _synthetic_hash(
                                        "activation", panel_id, score_split, depth, role
                                    ),
                                    "T": operator_target,
                                    "P": operator_pred,
                                    "D": operator_dot,
                                }
                            )
    return pd.DataFrame(weight_rows), pd.DataFrame(operator_rows)


def build_synthetic_permutation_manifest(weight: pd.DataFrame) -> pd.DataFrame:
    matrix_map = {
        (str(row.panel_id), int(row.depth), str(row.role)): str(row.matrix_key)
        for row in weight[["panel_id", "depth", "role", "matrix_key"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    namespace = "within_row_code_derangement_v1"
    for panel_id in PANELS:
        for tiling_seed in TILING_SEEDS:
            for depth in DEPTHS:
                for role in ROLES:
                    d_in, d_out = expected_matrix_shape(role)
                    n_col_groups = d_out // 64
                    for row_group in range(d_in // 64):
                        canonical = {
                            "depth": depth,
                            "namespace": namespace,
                            "panel": panel_id,
                            "role": role,
                            "row_group": row_group,
                            "tiling_seed": tiling_seed,
                        }
                        digest = hashlib.sha256(
                            json.dumps(
                                canonical,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            ).encode("utf-8")
                        ).digest()
                        offset = 1 + int.from_bytes(digest[:8], "big") % (n_col_groups - 1)
                        mapping = np.asarray(
                            [(column + offset) % n_col_groups for column in range(n_col_groups)],
                            dtype=np.int64,
                        )
                        rows.append(
                            {
                                **canonical,
                                "n_col_groups": n_col_groups,
                                "offset": offset,
                                "fixed_points": 0,
                                "bijection_pass": True,
                                "mapping_sha256": _int64_index_sha256(mapping),
                                "matrix_key": matrix_map[(panel_id, depth, role)],
                                "correct_code_block_sha256": _synthetic_hash(
                                    "correct_block", panel_id, tiling_seed, depth, role, row_group
                                ),
                                "permuted_code_block_sha256": _synthetic_hash(
                                    "permuted_block", panel_id, tiling_seed, depth, role, row_group
                                ),
                                "same_row_code_multiset_bit_exact": True,
                            }
                        )
    columns = [
        "depth",
        "namespace",
        "panel",
        "role",
        "row_group",
        "tiling_seed",
        "n_col_groups",
        "offset",
        "fixed_points",
        "bijection_pass",
        "mapping_sha256",
        "matrix_key",
        "correct_code_block_sha256",
        "permuted_code_block_sha256",
        "same_row_code_multiset_bit_exact",
    ]
    return pd.DataFrame(rows)[columns]


def build_synthetic_geometry() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(26_081_999)
    metadata_rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    panel_ids = ("source_vit_b_flickr", "beans", "trocr_sroie")
    source_seeds = (26_081_601, 26_081_602)
    for panel_index, panel_id in enumerate(panel_ids):
        seeds = source_seeds if panel_id == "source_vit_b_flickr" else TILING_SEEDS
        for tiling_index, tiling_seed in enumerate(seeds, start=1):
            for depth in DEPTHS:
                for role_index, role in enumerate(ROLES):
                    feature = rng.normal(0.0, 0.2, size=2_560)
                    feature[:32] += panel_index * 0.8
                    feature[32:64] += role_index * 0.25
                    feature[64:96] += depth / 11.0
                    feature[96:112] += tiling_index * 0.1
                    features.append(feature)
                    metadata_rows.append(
                        _geometry_meta_record(
                            panel_id,
                            tiling_seed,
                            tiling_index,
                            depth,
                            role,
                            f"synthetic|{panel_id}|{depth}|{role}",
                            _synthetic_hash("code", panel_id, tiling_seed, depth, role),
                        )
                    )
    feature_array = np.stack(features).astype(np.float64)
    feature_array[:, -1] = 7.0
    return pd.DataFrame(metadata_rows), feature_array


def synthetic_self_test(
    seal: dict[str, Any],
    persistent_output: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    weight, operator = build_synthetic_statistics()
    if persistent_output is None:
        context: Any = tempfile.TemporaryDirectory(
            prefix="prospective_replication_analyzer_selftest_"
        )
    else:
        if persistent_output.exists():
            raise RuntimeError(f"synthetic self-test output must be absent: {persistent_output}")
        persistent_output.mkdir(parents=True, exist_ok=False)
        context = contextlib.nullcontext(str(persistent_output))
    with context as temp_raw:
        temp = Path(temp_raw)
        target_events_before = int(seal["target_access_events"])
        rejected_probe_names: list[str] = []
        for probe_name in ("data2vec_forbidden_probe", "target-forbidden-probe"):
            try:
                (temp / probe_name).open("rb")
            except RuntimeError:
                rejected_probe_names.append(probe_name)
        if len(rejected_probe_names) != 2 or int(seal["target_access_events"]) != target_events_before + 2:
            raise AssertionError("filesystem audit hook did not reject/count both forbidden-path probes")
        with tempfile.TemporaryDirectory(prefix="prospective_binding_selfcheck_") as binding_temp:
            binding_test_root = Path(binding_temp)
            identity_binding_selftest = _synthetic_identity_binding_tests(binding_test_root)
            raw_manifest_selftest = _synthetic_raw_manifest_tests(binding_test_root)
        stats_dir = temp / "stats"
        stats_dir.mkdir()
        write_csv(stats_dir / "weight_sufficient_stats.csv", weight)
        write_csv(stats_dir / "operator_sufficient_stats.csv", operator)
        loaded_weight, loaded_operator, grid_audit = load_and_validate_sufficient_statistics(stats_dir)
        write_csv(
            stats_dir / "permutation_manifest.csv",
            build_synthetic_permutation_manifest(loaded_weight),
        )
        _, permutation_audit = validate_permutation_manifest(stats_dir, loaded_weight)
        algebra = verify_analytic_gain_algebra(loaded_weight, loaded_operator)
        weight_aggregates = build_aggregates(loaded_weight, "weight")
        operator_aggregates = build_aggregates(loaded_operator, "operator")
        error, cosine, did, common, absolute, bootstrap_manifest = build_bootstrap_tables(
            loaded_operator, BOOTSTRAP_DRAWS
        )
        mappings, mapping_summary = build_role_gain_mappings(loaded_operator)
        metadata = {
            "panels": [
                {
                    "panel_id": panel_id,
                    "provenance_pass": True,
                    "resource_hash_pass": True,
                    "quality_pass": True,
                    "validity_pass": True,
                }
                for panel_id in PANELS
            ]
        }
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
        if not experiment["both_panels_full_replication_pass"]:
            raise AssertionError(f"synthetic passing construction did not pass: {decisions}")

        selected = loaded_operator[
            (loaded_operator["panel_id"] == "beans")
            & (loaded_operator["score_split"] == "A")
            & (loaded_operator["tiling_seed"] == TILING_SEEDS[0])
            & (loaded_operator["arm"] == "correct")
        ]
        role_sums = selected.groupby("role", sort=False)[["T", "P", "D"]].sum().loc[list(ROLES)]
        gain_vector = gains_for_roles("source_role_gain", ROLES)
        manual_target = float(role_sums["T"].sum())
        manual_pred = float(np.sum(gain_vector * gain_vector * role_sums["P"].to_numpy()))
        manual_dot = float(np.sum(gain_vector * role_sums["D"].to_numpy()))
        manual_micro = float(scaled_metrics(manual_target, manual_pred, manual_dot, 1.0)["E"])
        stored_micro = float(
            _one_row(
                operator_aggregates,
                panel_id="beans",
                score_split="A",
                tiling_seed=TILING_SEEDS[0],
                arm="correct",
                calibration="source_role_gain",
                aggregation="micro",
            ).E
        )
        if not math.isclose(manual_micro, stored_micro, rel_tol=0.0, abs_tol=1e-14):
            raise AssertionError("synthetic role-aware micro formula does not match manual formula")

        rank_test = conservative_rank(np.asarray([0.5, 1.0 - 5e-13, 1.0, 1.0 + 5e-13, 1.1]), 1.0)
        if rank_test != (4, 1, 3):
            raise AssertionError(f"conservative tie rank test failed: {rank_test}")
        precedence = [
            choose_outcome(False, True, True),
            choose_outcome(True, True, False),
            choose_outcome(True, False, True),
            choose_outcome(True, False, False),
        ]
        expected_precedence = [
            "INVALID_PANEL",
            "FULL_REPLICATION_PASS",
            "DIRECTIONAL_ONLY_OR_MIXED",
            "MECHANISM_FAIL",
        ]
        if precedence != expected_precedence:
            raise AssertionError(f"outcome precedence test failed: {precedence}")

        invalid_metadata = json.loads(json.dumps(metadata))
        invalid_metadata["panels"][0]["quality_pass"] = False
        _, invalid_decisions, _ = build_criteria_and_decisions(
            operator_aggregates,
            error,
            cosine,
            did,
            common,
            absolute,
            mapping_summary,
            invalid_metadata,
        )
        if invalid_decisions["beans"]["outcome"] != "INVALID_PANEL":
            raise AssertionError("INVALID_PANEL did not override passing scientific criteria")

        corrupt_dir = temp / "corrupt_stats"
        corrupt_dir.mkdir()
        write_csv(corrupt_dir / "weight_sufficient_stats.csv", loaded_weight)
        corrupted = loaded_operator.copy()
        base_mask = (
            (corrupted["panel_id"] == "beans")
            & (corrupted["score_split"] == "A")
            & (corrupted["tiling_seed"] == TILING_SEEDS[0])
            & (corrupted["depth"] == 0)
            & (corrupted["role"] == "attn_query")
        )
        correct_sha = str(corrupted.loc[base_mask & (corrupted["arm"] == "correct"), "prediction_sha256"].iloc[0])
        corrupted.loc[base_mask & (corrupted["arm"] == "permuted_within_row"), "prediction_sha256"] = correct_sha
        write_csv(corrupt_dir / "operator_sufficient_stats.csv", corrupted)
        corruption_rejected = False
        try:
            load_and_validate_sufficient_statistics(corrupt_dir)
        except RuntimeError:
            corruption_rejected = True
        if not corruption_rejected:
            raise AssertionError("synthetic prediction-hash collision was not rejected")

        import torch

        synthetic_row_groups = [
            torch.arange(begin, begin + 64, dtype=torch.int64)
            for begin in range(0, 768, 64)
        ]
        synthetic_col_groups = [
            torch.arange(begin, begin + 64, dtype=torch.int64)
            for begin in range(0, 3_072, 64)
        ]
        row_sha, row_group_count = _validate_index_groups(
            synthetic_row_groups, 768, "synthetic/rows"
        )
        col_sha, col_group_count = _validate_index_groups(
            synthetic_col_groups, 3_072, "synthetic/cols"
        )
        partition_sha = hashlib.sha256(f"{row_sha}|{col_sha}".encode("ascii")).hexdigest()
        if row_group_count * col_group_count != 576 or not HEX64.fullmatch(partition_sha):
            raise AssertionError("synthetic tiling hash/count contract failed")
        corrupted_groups = [group.clone() for group in synthetic_row_groups]
        corrupted_groups[-1][-1] = corrupted_groups[-1][-2]
        tiling_corruption_rejected = False
        try:
            _validate_index_groups(corrupted_groups, 768, "synthetic/corrupt_rows")
        except RuntimeError:
            tiling_corruption_rejected = True
        if not tiling_corruption_rejected:
            raise AssertionError("synthetic tiling coverage corruption was not rejected")

        schema_payload = {
            "schema_version": "prospective_sufficient_statistics_v1",
            "files": {
                "weight": {
                    "filename": "weight_sufficient_stats.csv",
                    "row_count": 864,
                    "key_columns": ["panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm"],
                    "space": "target=W; prediction=W_hat",
                },
                "operator": {
                    "filename": "operator_sufficient_stats.csv",
                    "row_count": 1_728,
                    "key_columns": [
                        "panel_id",
                        "score_split",
                        "tiling_seed",
                        "depth",
                        "role",
                        "matrix_key",
                        "arm",
                    ],
                    "space": "target=X@W; prediction=X@W_hat",
                },
            },
            "canonical_statistics": {
                "T": {"dtype": "float64", "semantic": "squared_target_norm"},
                "P": {"dtype": "float64", "semantic": "squared_prediction_norm"},
                "D": {"dtype": "float64", "semantic": "target_prediction_inner_product"},
            },
            "checked_aliases": {
                "target_energy": "T",
                "pred_energy": "P",
                "target_pred_dot": "D",
            },
            "numeric_contract": {
                "decision_columns": ["T", "P", "D"],
                "aliases_are_not_independent_values": True,
                "accumulation_dtype": "float64",
                "float_serialization": "Python float repr, IEEE-754 binary64 decimal round-trip",
                "finite": True,
                "T_positive": True,
                "P_positive": True,
            },
        }
        schema_test_path = temp / "sufficient_statistics_schema.json"
        write_json(schema_test_path, schema_payload)
        schema_selftest = validate_sufficient_statistics_schema(schema_test_path)

        geometry_dir = temp / "geometry"
        geometry_dir.mkdir()
        geometry_meta, geometry_features = build_synthetic_geometry()
        geometry_scores, pca_audit = fit_fixed_geometry(
            geometry_meta, geometry_features, geometry_dir
        )
        plot_audit = render_geometry_plots(geometry_scores, geometry_dir)
        if not plot_audit["pass"] or len(plot_audit["plots"]) != 7:
            raise AssertionError("synthetic geometry plot audit failed")

        return {
            "status": "PASS",
            "scope": "synthetic_only_no_formal_runner_outputs_no_weight_ae",
            "elapsed_seconds": time.monotonic() - started,
            "grid": grid_audit,
            "analytic_gain_algebra": algebra,
            "counts": {
                "weight_aggregates": len(weight_aggregates),
                "operator_aggregates": len(operator_aggregates),
                "criterion1_error": len(error),
                "criterion2_cosine": len(cosine),
                "criterion3_role": int((criteria["criterion"] == 3).sum()),
                "criterion4_did": len(did),
                "criterion5_radial": int((criteria["criterion"] == 5).sum()),
                "criterion6_common": len(common),
                "criterion6_mapping_summary": len(mapping_summary),
                "criterion6_mapping_full": len(mappings),
                "criterion7_absolute": len(absolute),
                "criterion_cells": len(criteria),
            },
            "bootstrap_index_hashes": bootstrap_manifest["panels"],
            "synthetic_panel_outcomes": {
                panel: decisions[panel]["outcome"] for panel in PANELS
            },
            "invalid_precedence_test": invalid_decisions["beans"]["outcome"],
            "corruption_rejected": corruption_rejected,
            "tiling_coverage_corruption_rejected": tiling_corruption_rejected,
            "synthetic_tiling_partition_sha256": partition_sha,
            "structural_schema_validation_pass": schema_selftest["pass"],
            "identity_binding_corruption_tests": identity_binding_selftest,
            "raw_manifest_corruption_tests": raw_manifest_selftest,
            "permutation_manifest_validation": permutation_audit,
            "tie_rank_test": {
                "worst_rank": rank_test[0],
                "strictly_better": rank_test[1],
                "tied": rank_test[2],
            },
            "pca": pca_audit,
            "plot_count": len(plot_audit["plots"]),
            "plot_readability_pass": plot_audit["pass"],
            "filesystem_forbidden_probes_rejected": rejected_probe_names,
        }


def _write_invalid_pre_wae_outputs(
    output_dir: Path,
    metadata: Mapping[str, Any],
    invalid_panels: Sequence[str],
    logger: logging.Logger,
    seal: Mapping[str, Any],
) -> None:
    decisions = {
        panel_id: {
            "outcome": "INVALID_PANEL" if panel_id in invalid_panels else "NOT_ANALYZED_PROTOCOL_STOP",
            "scientific_metrics_computed": False,
        }
        for panel_id in PANELS
    }
    experiment = {
        "execution_status": "INVALID_PANEL_PRE_WAE",
        "invalid_panels": list(invalid_panels),
        "panel_outcomes": decisions,
        "later_confirmatory_target_protocol_may_be_frozen": False,
        "target_data_unsealed": False,
    }
    write_json(output_dir / "panel_decisions.json", decisions)
    write_json(output_dir / "experiment_decision.json", experiment)
    write_json(output_dir / "runner_metadata_snapshot.json", metadata)
    write_json(output_dir / "analyzer_source_only_seal.json", dict(seal))
    (output_dir / "README.md").write_text(
        "# Prospective replication analysis\n\n"
        "The protocol stopped at INVALID_PANEL_PRE_WAE. No Weight-AE scientific outcome was analyzed.\n",
        encoding="utf-8",
    )
    logger.info("terminal_status=INVALID_PANEL_PRE_WAE invalid_panels=%s", list(invalid_panels))
    logger.info("stage=artifact_manifest")
    flush_logger(logger)
    write_json(output_dir / "artifact_manifest.json", artifact_manifest(output_dir))


def run_formal_analysis(args: argparse.Namespace, seal: dict[str, Any]) -> None:
    input_dir = require_within(args.input_dir, FORMAL_ARTIFACT_ROOT, "input", strict=True)
    output_dir = require_within(args.output_dir, FORMAL_ARTIFACT_ROOT, "output", strict=False)
    if output_dir.exists():
        raise RuntimeError(f"formal analyzer output path must be absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logging(output_dir)
    started = time.monotonic()
    script_path = Path(__file__).resolve(strict=True)
    analyzer_sha = sha256_file(script_path)
    resolved_config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "design_path": str(args.design_path.resolve(strict=False)),
        "frozen_design_sha256": FROZEN_DESIGN_SHA256,
        "analyzer_path": str(script_path),
        "analyzer_sha256": analyzer_sha,
        "device": "cpu",
        "sufficient_statistic_dtype": "FP64",
        "persisted_new_code_dtype": "FP32",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "cache_mode": "sealed_runner_outputs_and_exact_source_caches",
        "geometry_protocol": "fixed_PCA10_full_PC1_PC2_only_non_gating",
    }
    write_json(output_dir / "resolved_config.json", resolved_config)
    logger.info(
        "resolved_config input=%s output=%s device=cpu stats_dtype=FP64 code_dtype=FP32 "
        "bootstrap_draws=%s bootstrap_seeds=%s cache_mode=sealed verbose=true",
        input_dir,
        output_dir,
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEEDS,
    )

    logger.info("stage=runtime_and_frozen_input_audit")
    runtime = validate_runtime_stack()
    frozen_inputs = validate_design_and_gains(args.design_path)
    module_audit = assert_no_forbidden_modules()
    metadata_path = input_dir / "runner_metadata.json"
    metadata, invalid_panels = validate_runner_metadata(metadata_path)
    preexecution = validate_preexecution_contract(input_dir, analyzer_sha, metadata)
    preexecution_binding = validate_preexecution_binding(input_dir, preexecution, metadata)
    raw_run_manifest = validate_raw_run_manifest(input_dir)
    schema_path = input_dir / "sufficient_statistics_schema.json"
    schema_audit = validate_sufficient_statistics_schema(schema_path)
    input_audit: dict[str, Any] = {
        "runtime": runtime,
        "frozen_inputs": frozen_inputs,
        "module_audit": module_audit,
        "runner_metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256_file(metadata_path),
            "bytes": metadata_path.stat().st_size,
        },
        "preexecution_contract": preexecution,
        "preexecution_binding": preexecution_binding,
        "raw_run_manifest": raw_run_manifest,
        "sufficient_statistics_schema": schema_audit,
    }
    write_json(output_dir / "input_audit.json", input_audit)
    if invalid_panels:
        _write_invalid_pre_wae_outputs(output_dir, metadata, invalid_panels, logger, seal)
        return

    logger.info("stage=sufficient_statistic_grid_validation expected_weight=864 expected_operator=1728")
    weight, operator, grid_audit = load_and_validate_sufficient_statistics(input_dir)
    write_json(output_dir / "sufficient_statistic_grid_audit.json", grid_audit)
    logger.info(
        "grid_pass weight_rows=%s operator_rows=%s prediction_A_B_reuse=true",
        len(weight),
        len(operator),
    )
    logger.info("stage=within_row_derangement_manifest rows=5184")
    permutation_manifest, permutation_audit = validate_permutation_manifest(input_dir, weight)
    write_json(output_dir / "permutation_manifest_audit.json", permutation_audit)

    logger.info("stage=analytic_gain_algebra")
    gain_audit = verify_analytic_gain_algebra(weight, operator)
    write_json(output_dir / "analytic_gain_algebra_audit.json", gain_audit)

    logger.info("stage=aggregate_metrics role_macro_role_aware_micro")
    weight_aggregates = build_aggregates(weight, "weight")
    operator_aggregates = build_aggregates(operator, "operator")
    write_csv(output_dir / "weight_aggregates.csv", weight_aggregates)
    write_csv(output_dir / "operator_aggregates.csv", operator_aggregates)

    logger.info("stage=paired_block_bootstrap draws=%s panels=%s", BOOTSTRAP_DRAWS, PANELS)
    error, cosine, did, common, absolute, bootstrap_manifest = build_bootstrap_tables(
        operator, BOOTSTRAP_DRAWS
    )
    write_csv(output_dir / "criterion1_error_bootstrap.csv", error)
    write_csv(output_dir / "criterion2_cosine_bootstrap.csv", cosine)
    write_csv(output_dir / "criterion4_did_bootstrap.csv", did)
    write_csv(output_dir / "criterion6_common_bootstrap.csv", common)
    write_csv(output_dir / "criterion7_absolute_bootstrap.csv", absolute)
    write_json(output_dir / "bootstrap_manifest.json", bootstrap_manifest)

    logger.info("stage=all_role_gain_mappings mappings_per_cell=720")
    mappings, mapping_summary = build_role_gain_mappings(operator)
    write_csv(output_dir / "role_gain_mappings.csv", mappings)
    write_csv(output_dir / "role_gain_mapping_summary.csv", mapping_summary)

    logger.info("stage=criteria_and_exact_outcome_precedence")
    criteria, panel_decisions, experiment = build_criteria_and_decisions(
        operator_aggregates,
        error,
        cosine,
        did,
        common,
        absolute,
        mapping_summary,
        metadata,
    )
    write_csv(output_dir / "criterion_cells.csv", criteria)
    write_json(output_dir / "panel_decisions.json", panel_decisions)
    write_json(output_dir / "experiment_decision.json", experiment)

    logger.info("stage=geometry_cache_audit population=432 features=2560")
    geometry_metadata, geometry_features, geometry_input_audit = load_geometry_inputs(
        input_dir, weight
    )
    write_csv(output_dir / "geometry_input_rows.csv", geometry_metadata)
    write_json(output_dir / "geometry_input_audit.json", geometry_input_audit)
    logger.info("stage=fixed_geometry_pca n_components=10 solver=full")
    geometry_scores, pca_audit = fit_fixed_geometry(
        geometry_metadata, geometry_features, output_dir
    )
    write_json(output_dir / "geometry_pca_audit.json", pca_audit)
    logger.info("stage=fixed_geometry_plots views=7 axes=PC1_PC2")
    plot_audit = render_geometry_plots(geometry_scores, output_dir)
    write_json(output_dir / "geometry_plot_readability_audit.json", plot_audit)

    logger.info("stage=raw_runner_immutability_recheck")
    raw_run_manifest_end = validate_raw_run_manifest(input_dir)
    if raw_run_manifest_end != raw_run_manifest:
        raise RuntimeError("raw runner manifest or its covered files changed during CPU analysis")
    write_json(
        output_dir / "raw_runner_immutability_recheck.json",
        {"pass": True, "initial": raw_run_manifest, "final": raw_run_manifest_end},
    )

    expected_counts = {
        "weight_sufficient_stats": 864,
        "operator_sufficient_stats": 1_728,
        "permutation_manifest_rows": 5_184,
        "weight_aggregates": 288,
        "operator_aggregates": 576,
        "criterion1_error": 32,
        "criterion2_cosine": 32,
        "criterion3_role": 48,
        "criterion4_did": 16,
        "criterion5_radial": 24,
        "criterion6_common": 16,
        "criterion6_mapping_summary": 16,
        "criterion6_mapping_full": 11_520,
        "criterion7_absolute": 16,
        "criterion_cells": 200,
        "geometry_rows": 432,
        "geometry_plots": 7,
    }
    actual_counts = {
        "weight_sufficient_stats": len(weight),
        "operator_sufficient_stats": len(operator),
        "permutation_manifest_rows": len(permutation_manifest),
        "weight_aggregates": len(weight_aggregates),
        "operator_aggregates": len(operator_aggregates),
        "criterion1_error": len(error),
        "criterion2_cosine": len(cosine),
        "criterion3_role": int((criteria["criterion"] == 3).sum()),
        "criterion4_did": len(did),
        "criterion5_radial": int((criteria["criterion"] == 5).sum()),
        "criterion6_common": len(common),
        "criterion6_mapping_summary": len(mapping_summary),
        "criterion6_mapping_full": len(mappings),
        "criterion7_absolute": len(absolute),
        "criterion_cells": len(criteria),
        "geometry_rows": len(geometry_scores),
        "geometry_plots": len(plot_audit["plots"]),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"final artifact row-count mismatch: {actual_counts} != {expected_counts}")
    write_json(
        output_dir / "artifact_count_audit.json",
        {"pass": True, "expected": expected_counts, "actual": actual_counts},
    )
    build_readme(output_dir, panel_decisions, experiment, criteria, plot_audit)
    seal.update(assert_no_forbidden_modules())
    if any(int(seal[field]) != 0 for field in ("target_access_events", "network_connections", "subprocess_launches")):
        raise RuntimeError(f"analyzer source-only seal recorded forbidden access: {seal}")
    write_json(output_dir / "analyzer_source_only_seal.json", seal)
    write_json(
        output_dir / "run_summary.json",
        {
            "status": "COMPLETE",
            "elapsed_seconds": time.monotonic() - started,
            "panel_outcomes": experiment["panel_outcomes"],
            "later_confirmatory_target_protocol_may_be_frozen": experiment[
                "later_confirmatory_target_protocol_may_be_frozen"
            ],
            "target_data_unsealed": False,
            "artifact_counts": actual_counts,
        },
    )
    logger.info(
        "complete elapsed=%.1fs outcomes=%s output=%s",
        time.monotonic() - started,
        experiment["panel_outcomes"],
        output_dir,
    )
    logger.info("stage=artifact_manifest self_excluded=true")
    flush_logger(logger)
    write_json(output_dir / "artifact_manifest.json", artifact_manifest(output_dir))


def main(argv: Sequence[str] | None = None) -> None:
    seal = install_source_only_seal()
    reject_forbidden_argv(list(argv) if argv is not None else sys.argv[1:])
    args = parse_args(argv)
    if args.self_test:
        persistent_output = None
        if args.self_test_output_dir is not None:
            persistent_output = require_within(
                args.self_test_output_dir,
                FORMAL_ARTIFACT_ROOT,
                "synthetic self-test output",
                strict=False,
            )
        result = synthetic_self_test(seal, persistent_output)
        result["runtime"] = validate_runtime_stack()
        result["analyzer_sha256"] = sha256_file(Path(__file__).resolve(strict=True))
        result["frozen_design_sha256"] = FROZEN_DESIGN_SHA256
        result["source_only_seal"] = seal
        if persistent_output is not None:
            write_json(persistent_output / "self_test_result.json", result)
            (persistent_output / "self_test.log").write_text(
                "status=PASS\n"
                "scope=synthetic_only_no_formal_runner_outputs_no_weight_ae\n"
                f"analyzer_sha256={result['analyzer_sha256']}\n"
                f"elapsed_seconds={result['elapsed_seconds']}\n",
                encoding="utf-8",
            )
            write_json(
                persistent_output / "artifact_manifest.json",
                artifact_manifest(persistent_output),
            )
            result["persistent_output"] = str(persistent_output)
            result["persistent_manifest_sha256"] = sha256_file(
                persistent_output / "artifact_manifest.json"
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    run_formal_analysis(args, seal)


if __name__ == "__main__":
    main()
