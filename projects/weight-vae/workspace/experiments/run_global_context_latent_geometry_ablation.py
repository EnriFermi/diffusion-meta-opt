#!/usr/bin/env python3
"""Raw runner for the frozen global-context latent-geometry ablation.

The default mode is a read-only audit.  ``--prepare-contract`` creates the
external preexecution contract without importing or forwarding the Weight-AE.
``--execute`` is the only mode that may construct an encoder, and it is gated
by exact design, runner, analyzer, input, dependency, runtime, path, and config
bindings.  The formal output is raw only; this process never invokes the
analyzer and never calls a decoder or distribution encoder.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gc
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import platform
import random
import shutil
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Process policy is installed before importing scientific libraries.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch


PROJECT_ROOT = Path("/home/coder/project")
WORKSPACE_ROOT = PROJECT_ROOT / "projects/weight-vae/workspace"
EXPERIMENTS_ROOT = WORKSPACE_ROOT / "experiments"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/crossmodal_united_structure"

DESIGN_PATH = PROJECT_ROOT / "docs/notes/global_context_latent_geometry_ablation_frozen_design_20260816.md"
DESIGN_SHA256 = "6cacb7bb269fd22a7652b07d37e026c8e8e24d6e8023e437656ddad794f1d56f"
AE_CHECKPOINT = PROJECT_ROOT / "artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt"
AE_CHECKPOINT_SHA256 = "d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00"
SOURCE_PARENT = ARTIFACT_ROOT / "source_confirmatory_gate_20260816_clean2"
SOURCE_TEMPLATE = SOURCE_PARENT / "source_condition_templates.pt"
SOURCE_TEMPLATE_SHA256 = "8fc6c61bb6baae4e7b1d618133ec651a91386d90c66f540182faa7dfb1655f99"
SOURCE_HELDOUT_MANIFEST = SOURCE_PARENT / "heldout_panel_manifest.json"
SOURCE_HELDOUT_MANIFEST_SHA256 = "10020aeabfcce6fe2df3003a7a626f21735cd0763c664e178f27ebcbf42ff985"
SOURCE_OFFLINE_REALPATH = Path(
    "/home/coder/project/projects/weight-vae/workspace/post_train_research/"
    "big_vae_heldout_eval/artifacts/offline_dataset"
)
SOURCE_FACTORIAL_DIR = ARTIFACT_ROOT / "source_latent_code_factorial_20260816"
SOURCE_FACTORIAL_CACHES = {
    26081601: (
        SOURCE_FACTORIAL_DIR / "factorial_code_cache_seed_26081601.pt",
        "a585c2ecfe4d1e560c79dca19518543dbdfb5f897bfa413b3ef5148f01995c8f",
    ),
    26081602: (
        SOURCE_FACTORIAL_DIR / "factorial_code_cache_seed_26081602.pt",
        "c56f1ed3ccd0038f3f78ec767facd32cc0bba490f2e31229ec5d7b18a211bfd6",
    ),
}
PANEL_ROOT = ARTIFACT_ROOT / "prospective_geometry_matched_panels_20260816"
PANEL_PATHS = {
    "beans": PANEL_ROOT / "panels/beans/panel.pt",
    "trocr_sroie": PANEL_ROOT / "panels/trocr_sroie/panel.pt",
}
PANEL_SHA256 = {
    "beans": "a47ddd182f09a56fc42d0fe887d0f2717b1972d46afbd1c92c80be13bbc9e6b6",
    "trocr_sroie": "ded6e7a84c1b34f1fe2d7fd1bdf6ccf041df23c1e20a710197815e0bf99f91a9",
}
OLD_CANDIDATE_TILINGS = ARTIFACT_ROOT / "prospective_geometry_matched_replication_20260816/tiling_indices.pt"
OLD_CANDIDATE_TILINGS_SHA256 = "a404812555e756e1b4fd380b99bc523530f2eade4a3ab0ddb974dfefba7ec25c"
OLD_CANDIDATE_CODES = ARTIFACT_ROOT / "prospective_geometry_matched_replication_20260816/correct_latent_codes.pt"
OLD_CANDIDATE_CODES_SHA256 = "b2b677f8c445c0f129dcd3b9819fdbfd65eb4659af7fecb35f89f4705a1e4daa"

DEFAULT_CONTRACT = PROJECT_ROOT / "docs/notes/global_context_latent_geometry_ablation_recovery_contract_20260816.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "global_context_latent_geometry_ablation_run_20260816"
DEFAULT_ANALYZER_OUTPUT = ARTIFACT_ROOT / "global_context_latent_geometry_ablation_analysis_20260816"
DEFAULT_ANALYZER = EXPERIMENTS_ROOT / "analyze_global_context_latent_geometry_ablation.py"
DEFAULT_PANEL_BUILDER = EXPERIMENTS_ROOT / "build_prospective_geometry_matched_panels.py"
RECOVERY_ERRATUM = (
    PROJECT_ROOT
    / "docs/notes/global_context_latent_geometry_ablation_validator_recovery_erratum_20260816.md"
)
RECOVERY_ERRATUM_SHA256 = "87bb2b61d259abf237af2491e53a3829d0d32de216b25483916e7d68e53568e6"
TRANSACTION_MARKER_NAME = ".runner_transaction.json"
FAILURE_RECORD_NAME = "failure_record.json"

PANEL_IDS = ("source_vit_b_flickr", "beans", "trocr_sroie")
ROLES = ("attn_query", "attn_key", "attn_value", "attn_output", "ffn_up", "ffn_down")
ROLE_SHAPES = {
    "attn_query": (768, 768),
    "attn_key": (768, 768),
    "attn_value": (768, 768),
    "attn_output": (768, 768),
    "ffn_up": (768, 3072),
    "ffn_down": (3072, 768),
}
COMMON_TILING_SEEDS = (26081901, 26081902)
GLOBAL_SEED = 26081931
UNTRAINED_SEEDS = (26081971, 26081972, 26081973)
COUNTSKETCH_SEEDS = (26081941, 26081942, 26081943, 26081944, 26081945)
CODE_REPRESENTATIONS = (
    "learned_cell",
    "learned_global",
    "learned_zero",
    "untrained_global_26081971",
    "untrained_global_26081972",
    "untrained_global_26081973",
)
REPRESENTATIONS = CODE_REPRESENTATIONS + (
    "raw_simple",
    "countsketch_26081941",
    "countsketch_26081942",
    "countsketch_26081943",
    "countsketch_26081944",
    "countsketch_26081945",
)
GLOBAL_C_HASHES = {
    "c_var": "3681bfd5033ade56c65f6d090ce41dea33866295580b1c85d679c6b3ae06303c",
    "c_patch": "b8755c89a668b4bd844f4bb53511951e5cd85b56d91bd537f28c5655df0829e4",
}
RAW_FEATURE_NAMES = (
    "log_numel", "log_d_in", "log_d_out", "mean", "population_std", "rms",
    "frobenius_norm", "mean_abs", "minimum", "maximum",
    "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99",
    "row_rms_mean", "row_rms_population_std", "row_rms_minimum", "row_rms_q10",
    "row_rms_q25", "row_rms_q50", "row_rms_q75", "row_rms_q90", "row_rms_maximum",
    "column_rms_mean", "column_rms_population_std", "column_rms_minimum", "column_rms_q10",
    "column_rms_q25", "column_rms_q50", "column_rms_q75", "column_rms_q90",
    "column_rms_maximum",
)
EXPECTED_RUNTIME = {
    "python_major_minor": "3.12",
    "torch": "2.10.0+cu128",
    "numpy": "2.3.5",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.0",
    "pandas": "3.0.0",
    "matplotlib": "3.10.8",
    "omegaconf": "2.3.0",
    "hydra-core": "1.3.2",
}
SOURCE_PARENT_HASHES = {
    "heldout_panel_manifest.json": SOURCE_HELDOUT_MANIFEST_SHA256,
    "tiling_coverage.csv": "876658595aec5f42736ae995f5c45d74f4d76a744aaaac9819f23cc9cc0f17b5",
    "source_condition_templates.pt": SOURCE_TEMPLATE_SHA256,
    "condition_template_summary.csv": "abcffb2d260ee0eed51c251331919c8d36ba7e8bd04e91d68c372a19c338c6b5",
    "matrix_metrics.csv": "d9d42484c55bd74a51da773edd3596306ad7f4b7d7f6055da90e1f284e7d19a1",
    "aggregate_metrics.csv": "85270f56526ba56a721c863223f1b230da4db22e093e9b8335d98e1722338c51",
    "block_bootstrap.csv": "233058de4df5518f5154ddcd6b755ecd69d1e73896f59d1c2ea84919380b66f9",
    "checkpoint_info.json": "5093fe3d19e8e6ca112f308ff77bf1cf812b687e54f13e73100fbda6265cdf30",
    "run_manifest.json": "853fff83debf96e778c50ef2cc33ddb88c06632258bed0af1daf00fae89f2b6a",
    "resolved_config.json": "b374b86e1b541f7107cbe448ea6bd9b91e9d82152b356c053ba799c5d02995b9",
    "execution_manifest.json": "9114f3b1942231b3a933d224029f2718a89fb9e442b645cbe4aee0ceb07af31f",
    "decisions.json": "477a51d7021255055ecbb8e132468918a041d6d3583c2e10f97ca170d1f98bbc",
}
EXPECTED_RAW_FILES = (
    "run.log", "resolved_config.json", "preexecution_contract.json",
    "preexecution_binding.json", "input_audit.json", "target_access_seal.json",
    "source_template_audit.json", "model_contract.json",
    "known_source_numeric_preflight.json", "dataflow_audit.json", "tiling_manifest.csv",
    "tiling_indices.pt", "code_manifest.csv", "latent_codes.pt", "raw_simple_features.csv",
    "countsketch_maps.json", "aggregate_feature_manifest.csv", "aggregate_features.npz",
    "zero_weight_sanity.json", "input_immutability_recheck.json", "runner_metadata.json",
)
CONTRACT_SCHEMA_KEYS = (
    "schema_version", "contract_schema_keys", "created_utc", "source_only_seal",
    "frozen_design_path", "frozen_design_sha256", "runner_path", "runner_sha256",
    "analyzer_path", "analyzer_sha256", "runtime", "execution_config", "frozen_grids",
    "scripts", "model_dependencies", "input_audit", "static_call_graph_audit",
    "expected_counts", "artifact_contract", "dataflow_contract", "analysis_contract",
)
SOURCE_HELPERS = (
    EXPERIMENTS_ROOT / "source_confirmatory_g01_gate.py",
    EXPERIMENTS_ROOT / "source_latent_code_factorial.py",
    EXPERIMENTS_ROOT / "background_prefetch.py",
)
DEPENDENCY_TREES = (
    WORKSPACE_ROOT / "big_vae",
    WORKSPACE_ROOT / "training",
    WORKSPACE_ROOT / "dataset",
)


class SealViolation(RuntimeError):
    """Raised for any target, network, or subprocess attempt."""


class _ForbiddenModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> None:
        del path, target
        if "data2vec" in fullname.casefold():
            with _SEAL_LOCK:
                _SEAL_STATE["target_access_events"] += 1
            raise SealViolation(f"source-only seal blocked module import: {fullname}")
        return None


_SEAL_LOCK = threading.Lock()
_SEAL_RESOLVE_GUARD = threading.local()
_SEAL_STATE: dict[str, Any] = {
    "installed": False,
    "installed_before_scientific_input_read": False,
    "forbidden_path_markers": ["data2vec", "target path component"],
    "administrative_filename_allowlist": ["target_access_seal.json"],
    "target_access_events": 0,
    "network_connections": 0,
    "subprocess_launches": 0,
}


def _forbidden_path_text(text: str) -> bool:
    normalized = text.replace("\\", "/").casefold()
    components = [part for part in normalized.split("/") if part]
    # The frozen administrative evidence filename is not a target dataset or
    # module path.  Only that exact basename is exempted; its parents remain
    # subject to the target-component guard.
    if components and components[-1] == "target_access_seal.json":
        components = components[:-1]
        normalized = "/".join(components)
    return "data2vec" in normalized or any(
        part in {"target", "targets"} or part.startswith(("target_", "target-"))
        for part in components
    )


def _path_forbidden(value: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> bool:
    original = os.fsdecode(value)
    if _forbidden_path_text(original):
        return True
    if getattr(_SEAL_RESOLVE_GUARD, "active", False):
        return False
    try:
        _SEAL_RESOLVE_GUARD.active = True
        resolved = os.path.realpath(original)
    except (OSError, ValueError):
        resolved = original
    finally:
        _SEAL_RESOLVE_GUARD.active = False
    return _forbidden_path_text(resolved)


def install_source_only_seal() -> dict[str, Any]:
    if _SEAL_STATE["installed"]:
        return dict(_SEAL_STATE)
    forbidden_loaded = sorted(name for name in sys.modules if "data2vec" in name.casefold())
    if forbidden_loaded:
        raise SealViolation(f"forbidden modules loaded before seal: {forbidden_loaded}")

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event in {"open", "os.listdir", "os.scandir"} and args:
            candidate = args[0]
            if isinstance(candidate, (str, bytes, os.PathLike)) and _path_forbidden(candidate):
                with _SEAL_LOCK:
                    _SEAL_STATE["target_access_events"] += 1
                raise SealViolation(f"source-only seal blocked filesystem event={event}: {candidate}")
        if event == "socket.connect":
            with _SEAL_LOCK:
                _SEAL_STATE["network_connections"] += 1
            raise SealViolation("source-only seal blocked network connection")
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"} or event.startswith("os.spawn"):
            with _SEAL_LOCK:
                _SEAL_STATE["subprocess_launches"] += 1
            raise SealViolation("source-only seal blocked subprocess launch")

    sys.addaudithook(audit_hook)
    sys.meta_path.insert(0, _ForbiddenModuleFinder())
    _SEAL_STATE.update(
        {
            "installed": True,
            "installed_before_scientific_input_read": True,
            "module_import_guard": True,
            "network_connect_prohibited": True,
            "subprocess_launch_prohibited": True,
            "original_and_resolved_path_guard": True,
        }
    )
    return dict(_SEAL_STATE)


def assert_seal_clean() -> None:
    if not _SEAL_STATE["installed"]:
        raise RuntimeError("source-only seal is not installed")
    forbidden_loaded = sorted(name for name in sys.modules if "data2vec" in name.casefold())
    if forbidden_loaded:
        raise SealViolation(f"forbidden modules loaded under seal: {forbidden_loaded}")
    nonzero = {
        name: int(_SEAL_STATE[name])
        for name in ("target_access_events", "network_connections", "subprocess_launches")
        if int(_SEAL_STATE[name]) != 0
    }
    if nonzero:
        raise SealViolation(f"source-only seal recorded prohibited attempts: {nonzero}")


def reject_forbidden_arguments(args: argparse.Namespace) -> None:
    for name, value in vars(args).items():
        if isinstance(value, Path):
            raw = os.fspath(value)
            if _forbidden_path_text(raw) or _forbidden_path_text(os.path.realpath(os.path.abspath(raw))):
                with _SEAL_LOCK:
                    _SEAL_STATE["target_access_events"] += 1
                raise SealViolation(f"forbidden path argument {name}={raw}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor, *, dtype: torch.dtype) -> str:
    value = tensor.detach().cpu().to(dtype).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def weight_shape_bytes_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError(f"weight must be 2-D: {tuple(value.shape)}")
    array = value.numpy().astype("<f4", copy=False)
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", int(value.shape[0]), int(value.shape[1])))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_hex(*items: Any) -> str:
    return hashlib.sha256("|".join(str(item) for item in items).encode("utf-8")).hexdigest()


def stable_seed(*items: Any) -> int:
    return int(stable_hex(*items)[:16], 16) % (2**63 - 1)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz_save(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    os.replace(temporary, destination)


def create_process_owned_staging(
    final_output: Path,
    *,
    runner_sha256: str,
    analyzer_sha256: str,
    external_contract_sha256: str,
) -> Path:
    """Create a process-owned sibling tree; the formal path stays absent."""
    final_output = final_output.resolve(strict=False)
    staging = final_output.with_name(f".{final_output.name}.staging-pid-{os.getpid()}")
    if final_output.exists() or final_output.is_symlink():
        raise RuntimeError(f"formal output must remain absent before staging: {final_output}")
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"process staging path is not fresh: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    atomic_write_json(
        staging / TRANSACTION_MARKER_NAME,
        {
            "schema_version": "global_context_runner_transaction_owner_v1",
            "created_utc": utc_now(),
            "owner_pid": os.getpid(),
            "staging_path": str(staging),
            "final_output_path": str(final_output),
            "runner_sha256": runner_sha256,
            "analyzer_sha256": analyzer_sha256,
            "external_contract_sha256": external_contract_sha256,
            "formal_path_created": False,
        },
    )
    return staging


def remove_process_owned_transaction_marker(staging: Path, final_output: Path) -> dict[str, Any]:
    marker_path = staging / TRANSACTION_MARKER_NAME
    if not marker_path.is_file() or marker_path.is_symlink():
        raise RuntimeError(f"transaction owner marker missing/symlinked: {marker_path}")
    marker = read_json(marker_path)
    required = {
        "schema_version", "created_utc", "owner_pid", "staging_path",
        "final_output_path", "runner_sha256", "analyzer_sha256",
        "external_contract_sha256", "formal_path_created",
    }
    if (
        not isinstance(marker, Mapping)
        or set(marker) != required
        or marker.get("schema_version") != "global_context_runner_transaction_owner_v1"
        or int(marker.get("owner_pid", -1)) != os.getpid()
        or marker.get("staging_path") != str(staging.resolve(strict=True))
        or marker.get("final_output_path") != str(final_output.resolve(strict=False))
        or marker.get("formal_path_created") is not False
    ):
        raise RuntimeError(f"transaction owner marker contract failed: {marker}")
    marker_path.unlink()
    return dict(marker)


def quarantine_failed_staging(
    staging: Path,
    final_output: Path,
    error: BaseException,
    *,
    failure_stage: str,
) -> dict[str, Any]:
    """Preserve a failed private tree while keeping the formal path absent."""
    result: dict[str, Any] = {
        "schema_version": "global_context_runner_failure_quarantine_v1",
        "created_utc": utc_now(),
        "owner_pid": os.getpid(),
        "failure_stage": failure_stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "staging_path": str(staging),
        "final_output_path": str(final_output),
        "formal_path_absent": not final_output.exists() and not final_output.is_symlink(),
        "failure_record_written": False,
        "quarantine_path": None,
    }
    if not staging.exists() or staging.is_symlink():
        result["staging_present"] = False
        return result
    result["staging_present"] = True
    try:
        atomic_write_json(staging / FAILURE_RECORD_NAME, result)
        result["failure_record_written"] = True
        atomic_write_json(staging / FAILURE_RECORD_NAME, result)
    except Exception as record_error:  # preserve/move even under a write failure such as ENOSPC
        result["failure_record_error"] = {
            "type": type(record_error).__name__,
            "message": str(record_error),
        }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = final_output.with_name(
        f"{final_output.name}_FAILED_{timestamp}_pid-{os.getpid()}"
    )
    quarantine = base
    suffix = 0
    while quarantine.exists() or quarantine.is_symlink():
        suffix += 1
        quarantine = base.with_name(f"{base.name}_{suffix}")
    try:
        os.rename(staging, quarantine)
        result["quarantine_path"] = str(quarantine)
        result["staging_present_after_quarantine"] = False
    except Exception as rename_error:
        result["quarantine_error"] = {
            "type": type(rename_error).__name__,
            "message": str(rename_error),
        }
        result["staging_present_after_quarantine"] = staging.exists()
    result["formal_path_absent"] = not final_output.exists() and not final_output.is_symlink()
    return result


def publish_staging_atomically(staging: Path, final_output: Path) -> None:
    if final_output.exists() or final_output.is_symlink():
        raise RuntimeError(f"formal output appeared before atomic publication: {final_output}")
    if not staging.is_dir() or staging.is_symlink():
        raise RuntimeError(f"staging output missing/symlinked before publication: {staging}")
    if (staging / TRANSACTION_MARKER_NAME).exists():
        raise RuntimeError("transaction marker survived into terminal publication")
    if {path.name for path in staging.iterdir()} != set(EXPECTED_RAW_FILES) | {
        "artifact_manifest.json"
    }:
        raise RuntimeError("staging membership drifted after terminal validation")
    os.rename(staging, final_output)
    if not final_output.is_dir() or final_output.is_symlink() or staging.exists():
        raise RuntimeError("atomic staging publication postcondition failed")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_exact_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink():
        # Immutable input paths may traverse the project artifact symlink, but
        # the file itself may not be a symlink.
        raise RuntimeError(f"{label} is a file symlink: {path}")
    resolved = path.resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected_sha256} ({resolved})")
    return {"path": str(resolved), "sha256": actual, "bytes": resolved.stat().st_size}


def runtime_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for package in ("scikit-learn", "scipy", "pandas", "matplotlib", "omegaconf", "hydra-core"):
        result[package] = importlib.metadata.version(package)
    result["cuda"] = {
        "available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "device_0_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    mismatch = {
        key: {"actual": result.get(key), "expected": expected}
        for key, expected in EXPECTED_RUNTIME.items()
        if result.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"frozen runtime mismatch: {mismatch}")
    return result


def expected_counts() -> dict[str, Any]:
    tiles_by_role = {
        role: (shape[0] // 64) * (shape[1] // 64) for role, shape in ROLE_SHAPES.items()
    }
    per_panel_tiling = 12 * sum(tiles_by_role.values())
    result = {
        "panels": 3,
        "tilings": 2,
        "depths": 12,
        "roles": 6,
        "matrices_per_panel": 72,
        "tiles_per_role_matrix": tiles_by_role,
        "tiles_per_panel_tiling": per_panel_tiling,
        "tiling_manifest_rows": 432,
        "tiling_index_entries": 432,
        "code_representations": 6,
        "code_manifest_rows": 2592,
        "latent_code_entries": 2592,
        "latent_code_rows": 746496,
        "latent_code_width": 512,
        "raw_simple_rows": 432,
        "raw_simple_feature_columns": 37,
        "countsketch_maps": 5,
        "countsketch_pairs": 20480,
        "aggregate_manifest_rows": 5184,
        "aggregate_arrays": 12,
        "aggregate_2560_arrays": 11,
        "aggregate_37_arrays": 1,
        "declared_raw_files": 21,
        "files_including_self_excluded_manifest": 22,
    }
    if per_panel_tiling != 20736 or len(RAW_FEATURE_NAMES) != 37 or len(REPRESENTATIONS) != 12:
        raise AssertionError(f"internal frozen-count derivation failed: {result}")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--prepare-contract", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    value.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--analyzer-output-dir", type=Path, default=DEFAULT_ANALYZER_OUTPUT)
    value.add_argument("--analyzer-path", type=Path, default=DEFAULT_ANALYZER)
    value.add_argument("--panel-builder-path", type=Path, default=DEFAULT_PANEL_BUILDER)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--log-every-batches", type=int, default=20)
    value.add_argument("--audit-report", type=Path, default=None)
    return value


def matrix_key(depth: int, role: str) -> str:
    return f"depth={depth:02d}|role={role}"


def canonical_tiling_key(panel_id: str, seed: int, depth: int, role: str) -> str:
    return f"panel={panel_id}|seed={seed}|depth={depth:02d}|role={role}"


def canonical_code_key(representation: str, panel_id: str, seed: int, depth: int, role: str) -> str:
    return (
        f"representation={representation}|panel={panel_id}|seed={seed}|"
        f"depth={depth:02d}|role={role}"
    )


def file_resources() -> dict[str, dict[str, Any]]:
    resources = {
        "frozen_design": require_exact_file(DESIGN_PATH, DESIGN_SHA256, "frozen design"),
        "validator_recovery_erratum": require_exact_file(
            RECOVERY_ERRATUM, RECOVERY_ERRATUM_SHA256, "validator recovery erratum"
        ),
        "learned_ae_checkpoint": require_exact_file(AE_CHECKPOINT, AE_CHECKPOINT_SHA256, "learned AE"),
        "source_condition_templates": require_exact_file(SOURCE_TEMPLATE, SOURCE_TEMPLATE_SHA256, "source templates"),
        "source_heldout_manifest": require_exact_file(
            SOURCE_HELDOUT_MANIFEST, SOURCE_HELDOUT_MANIFEST_SHA256, "source heldout manifest"
        ),
        "beans_panel": require_exact_file(PANEL_PATHS["beans"], PANEL_SHA256["beans"], "Beans panel"),
        "trocr_sroie_panel": require_exact_file(
            PANEL_PATHS["trocr_sroie"], PANEL_SHA256["trocr_sroie"], "TrOCR panel"
        ),
        "old_candidate_tiling_indices": require_exact_file(
            OLD_CANDIDATE_TILINGS, OLD_CANDIDATE_TILINGS_SHA256, "old candidate tilings"
        ),
        "old_candidate_cell_codes": require_exact_file(
            OLD_CANDIDATE_CODES, OLD_CANDIDATE_CODES_SHA256, "old candidate cell codes"
        ),
    }
    for seed, (path, digest) in SOURCE_FACTORIAL_CACHES.items():
        resources[f"source_factorial_code_cache_{seed}"] = require_exact_file(
            path, digest, f"source factorial cache {seed}"
        )
    for name, digest in SOURCE_PARENT_HASHES.items():
        resources[f"source_parent/{name}"] = require_exact_file(
            SOURCE_PARENT / name, digest, f"source parent {name}"
        )
    return resources


def script_snapshot(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = {
        "runner": Path(__file__).resolve(strict=True),
        "analyzer": args.analyzer_path.resolve(strict=True),
        "panel_builder": args.panel_builder_path.resolve(strict=True),
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def dependency_snapshot() -> dict[str, dict[str, Any]]:
    paths: set[Path] = set(SOURCE_HELPERS)
    for root in DEPENDENCY_TREES:
        paths.update(path for path in root.resolve(strict=True).rglob("*.py") if path.is_file())
    workspace = WORKSPACE_ROOT.resolve(strict=True)
    initial = tuple(paths)
    for path in initial:
        parent = path.resolve(strict=True).parent
        while parent != workspace:
            if workspace not in parent.parents:
                raise RuntimeError(f"dependency escaped workspace: {path}")
            initializer = parent / "__init__.py"
            if initializer.is_file():
                paths.add(initializer)
            parent = parent.parent
    result: dict[str, dict[str, Any]] = {}
    for path in sorted({item.resolve(strict=True) for item in paths}, key=str):
        relative = path.relative_to(workspace).as_posix()
        result[relative] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if len(result) < 150:
        raise RuntimeError(f"model/data dependency closure suspiciously small: {len(result)}")
    return result


def assert_exact_formal_paths(args: argparse.Namespace) -> dict[str, str]:
    """Reject every formal path/script override before scientific input reads."""
    expected = {
        "contract_path": DEFAULT_CONTRACT.resolve(strict=False),
        "output_dir": DEFAULT_OUTPUT.resolve(strict=False),
        "analyzer_output_dir": DEFAULT_ANALYZER_OUTPUT.resolve(strict=False),
        "analyzer_path": DEFAULT_ANALYZER.resolve(strict=True),
        "panel_builder_path": DEFAULT_PANEL_BUILDER.resolve(strict=True),
    }
    observed = {
        "contract_path": args.contract_path.resolve(strict=False),
        "output_dir": args.output_dir.resolve(strict=False),
        "analyzer_output_dir": args.analyzer_output_dir.resolve(strict=False),
        "analyzer_path": args.analyzer_path.resolve(strict=True),
        "panel_builder_path": args.panel_builder_path.resolve(strict=True),
    }
    mismatch = {
        name: {"observed": str(observed[name]), "expected": str(expected[name])}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatch:
        raise RuntimeError(f"formal path/script override prohibited: {mismatch}")
    return {name: str(path) for name, path in observed.items()}


def assert_canonical_formal_entrypoint_and_argv(
    args: argparse.Namespace,
    *,
    argv0: str | os.PathLike[str] | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bind formal invocation to this reviewed file and one canonical CLI."""
    expected_entrypoint = Path(__file__).resolve(strict=True)
    invoked_raw = sys.argv[0] if argv0 is None else os.fspath(argv0)
    invoked_entrypoint = Path(invoked_raw).resolve(strict=True)
    if invoked_entrypoint != expected_entrypoint:
        raise RuntimeError(
            "formal runner entrypoint mismatch: "
            f"invoked={invoked_entrypoint} expected={expected_entrypoint}"
        )
    observed_argv = list(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        raise RuntimeError("formal entrypoint/argv gate must not be used for self-test mode")
    if args.prepare_contract:
        mode = "prepare_contract"
        expected_argv = ["--prepare-contract"]
    elif args.execute:
        mode = "execute"
        expected_argv = ["--execute"]
    elif args.audit_report is None:
        mode = "audit_only_stdout"
        expected_argv = []
    else:
        mode = "audit_only_report"
        expected_argv = ["--audit-report", os.fspath(args.audit_report)]
    if observed_argv != expected_argv:
        raise RuntimeError(
            f"noncanonical formal argv for mode={mode}: "
            f"observed={observed_argv} expected={expected_argv}"
        )
    return {
        "entrypoint": str(invoked_entrypoint),
        "mode": mode,
        "argv": observed_argv,
        "pass": True,
    }


def execution_config(args: argparse.Namespace) -> dict[str, Any]:
    assert_exact_formal_paths(args)
    output = args.output_dir.resolve(strict=False)
    analyzer_output = args.analyzer_output_dir.resolve(strict=False)
    artifact_root = ARTIFACT_ROOT.resolve(strict=True)
    contract = args.contract_path.resolve(strict=False)
    if output.exists():
        raise RuntimeError(f"runner output must be fresh and absent: {output}")
    if analyzer_output.exists():
        raise RuntimeError(f"analyzer output must be fresh and absent: {analyzer_output}")
    if not output.is_relative_to(artifact_root) or output == artifact_root:
        raise RuntimeError(f"runner output must be below artifact root: {output}")
    if not analyzer_output.is_relative_to(artifact_root) or analyzer_output == artifact_root:
        raise RuntimeError(f"analyzer output must be below artifact root: {analyzer_output}")
    if output == analyzer_output or output.is_relative_to(analyzer_output) or analyzer_output.is_relative_to(output):
        raise RuntimeError("runner and analyzer outputs must be disjoint")
    if contract.is_relative_to(output) or contract.is_relative_to(analyzer_output):
        raise RuntimeError("external contract must not be inside either output")
    observed = {
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "log_every_batches": int(args.log_every_batches),
    }
    expected = {"device": "cuda:0", "batch_size": 64, "log_every_batches": 20}
    if observed != expected:
        raise RuntimeError(f"formal execution config mismatch: {observed} != {expected}")
    return {
        **observed,
        "output_dir": str(output),
        "analyzer_output_dir": str(analyzer_output),
        "contract_path": str(contract),
        "global_seed": GLOBAL_SEED,
        "common_tiling_seeds": list(COMMON_TILING_SEEDS),
        "untrained_seeds": list(UNTRAINED_SEEDS),
        "countsketch_seeds": list(COUNTSKETCH_SEEDS),
        "panel_order": list(PANEL_IDS),
        "role_order": list(ROLES),
        "representation_order": list(REPRESENTATIONS),
        "device_dtype": "FP32 inputs; CUDA BF16 autocast; FP32 codes; FP64 aggregates",
        "cache_mode": "immutable_inputs_revalidated; old codes audit-only; all decision features fresh",
        "latent_sampling": False,
        "rope_2d_coordinates": "raw_integer",
        "decoder_forward": False,
        "distribution_encoder_forward": False,
    }


def live_resource_snapshot(device: torch.device) -> dict[str, Any]:
    disk = shutil.disk_usage(ARTIFACT_ROOT.resolve(strict=True))
    memory_available: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                memory_available = int(line.split()[1]) * 1024
                break
    except OSError:
        memory_available = None
    cuda_payload: dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "device": str(device),
        "free_bytes": None,
        "total_bytes": None,
        "allocated_bytes": None,
        "reserved_bytes": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = 0 if device.index is None else int(device.index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        cuda_payload.update(
            {
                "device_index": index,
                "device_name": torch.cuda.get_device_name(index),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            }
        )
    return {
        "disk": {
            "path": str(ARTIFACT_ROOT.resolve(strict=True)),
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        },
        "host_memory_available_bytes": memory_available,
        "cuda": cuda_payload,
    }


@dataclass(frozen=True)
class TilingIndex:
    seed: int
    rows: torch.Tensor
    cols: torch.Tensor
    row_sha256: str
    column_sha256: str
    partition_sha256: str

    @property
    def num_tiles(self) -> int:
        return int(self.rows.shape[0]) * int(self.cols.shape[0])


@dataclass
class WOnlyEntry:
    panel_id: str
    depth: int
    role: str
    W: torch.Tensor
    weight_shape_bytes_sha256: str
    weight_tensor_sha256: str
    tilings: dict[int, TilingIndex]


@dataclass
class WOnlyBundle:
    entries: list[WOnlyEntry]
    tiling_rows: list[dict[str, Any]]
    source_preflight_tiling: TilingIndex
    source_preflight_cached_slice_sha256: str
    source_preflight_code_key: str
    materialization_audit: dict[str, Any]


def validate_templates(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping) or set(payload) != {"global", "cell_mean", "cell_medoid"}:
        raise RuntimeError(f"source template top-level grid mismatch: {type(payload)}/{getattr(payload, 'keys', lambda: [])()}")
    expected_cells = {(role, depth) for role in ROLES for depth in range(12)}
    if not isinstance(payload["cell_mean"], Mapping) or set(payload["cell_mean"]) != expected_cells:
        raise RuntimeError("source cell_mean template grid is not exactly 12x6")
    summary: dict[str, Any] = {
        "schema_version": "global_context_source_template_audit_v1",
        "template_file_sha256": sha256_file(SOURCE_TEMPLATE.resolve(strict=True)),
        "global": {},
        "cell_mean": {},
        "cell_medoid_audited_not_used": True,
    }
    for condition_name in ("c_var", "c_patch"):
        tensor = payload["global"].get(condition_name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (256,)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise RuntimeError(f"invalid global {condition_name}")
        digest = tensor_sha256(tensor, dtype=torch.float32)
        if digest != GLOBAL_C_HASHES[condition_name]:
            raise RuntimeError(f"global {condition_name} hash mismatch: {digest}")
        summary["global"][f"{condition_name}_sha256"] = digest
    clean = {
        "global": {
            "c_var": payload["global"]["c_var"].detach().cpu().to(torch.float32).contiguous().clone(),
            "c_patch": payload["global"]["c_patch"].detach().cpu().to(torch.float32).contiguous().clone(),
        },
        "cell_mean": {},
    }
    for depth in range(12):
        for role in ROLES:
            template = payload["cell_mean"][(role, depth)]
            if not isinstance(template, Mapping) or set(template) != {"c_var", "c_patch"}:
                raise RuntimeError(f"invalid cell template mapping: {role}/{depth}")
            record: dict[str, Any] = {}
            copied: dict[str, torch.Tensor] = {}
            for condition_name in ("c_var", "c_patch"):
                tensor = template[condition_name]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != (256,)
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise RuntimeError(f"invalid cell {condition_name}: {role}/{depth}")
                copied[condition_name] = tensor.detach().cpu().contiguous().clone()
                record[f"{condition_name}_sha256"] = tensor_sha256(tensor, dtype=torch.float32)
            clean["cell_mean"][(role, depth)] = copied
            summary["cell_mean"][matrix_key(depth, role)] = record
    summary["global_hashes_match_frozen_design"] = True
    summary["cell_count"] = 72
    summary["pass"] = True
    return clean, summary


def audit_source_parent() -> dict[str, Any]:
    parent = SOURCE_PARENT.resolve(strict=True)
    hashes = {
        name: require_exact_file(parent / name, digest, f"source parent {name}")
        for name, digest in SOURCE_PARENT_HASHES.items()
    }
    resolved = read_json(parent / "resolved_config.json")
    run = read_json(parent / "run_manifest.json")
    execution = read_json(parent / "execution_manifest.json")
    checkpoint = read_json(parent / "checkpoint_info.json")
    panel = read_json(parent / "heldout_panel_manifest.json")
    heldout = Path(str(resolved.get("heldout_root", ""))).resolve(strict=True)
    if heldout != SOURCE_OFFLINE_REALPATH.resolve(strict=True):
        raise RuntimeError(f"source heldout realpath mismatch: {heldout} != {SOURCE_OFFLINE_REALPATH}")
    if run.get("status") != "COMPLETE_G0_G1" or bool(run.get("target_data2vec_access")):
        raise RuntimeError("source parent completion/target-access contract failed")
    if execution.get("target_data2vec_access") != "PROHIBITED_AND_NOT_PERFORMED":
        raise RuntimeError("source parent execution seal failed")
    required_checkpoint = {
        "sha256": AE_CHECKPOINT_SHA256,
        "step": 480000,
        "rope_2d_coord_kind": "raw",
        "use_latent_sampling": False,
        "strict_load": True,
    }
    if any(checkpoint.get(key) != expected for key, expected in required_checkpoint.items()):
        raise RuntimeError(f"source parent checkpoint contract failed: {checkpoint}")
    if not isinstance(panel, list) or len(panel) != 72:
        raise RuntimeError("source heldout manifest must have 72 rows")
    identities: set[tuple[int, str]] = set()
    for index, item in enumerate(panel):
        if not isinstance(item, Mapping) or set(item) != {"context_a", "score_b"}:
            raise RuntimeError(f"invalid source heldout row {index}")
        a, b = item["context_a"], item["score_b"]
        required = {"chunk_idx", "record_idx", "source_key", "model_name", "layer_name", "role", "depth", "weight_shape", "primary_dataset"}
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or not required.issubset(a) or not required.issubset(b):
            raise RuntimeError(f"source heldout row lacks fields: {index}")
        role, depth = str(a["role"]), int(a["depth"])
        if (
            role not in ROLES
            or not 0 <= depth < 12
            or tuple(a["weight_shape"]) != ROLE_SHAPES[role]
            or any(a[key] != b[key] for key in ("source_key", "model_name", "layer_name", "role", "depth", "weight_shape"))
            or a["model_name"] != "vit_base_p16_224"
            or a["primary_dataset"] != "flickr30k"
            or b["primary_dataset"] != "flickr30k"
            or (int(a["chunk_idx"]), int(a["record_idx"])) == (int(b["chunk_idx"]), int(b["record_idx"]))
        ):
            raise RuntimeError(f"source heldout identity contract failed at row {index}: {a}/{b}")
        identities.add((depth, role))
    if identities != {(depth, role) for depth in range(12) for role in ROLES}:
        raise RuntimeError("source heldout manifest is not the exact depth-role grid")
    with (parent / "tiling_coverage.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 144:
        raise RuntimeError(f"source old tiling audit row count mismatch: {len(rows)}")
    tiling_index = {
        (int(row["tiling_seed"]), int(row["depth"]), row["role"]): row for row in rows
    }
    if len(tiling_index) != 144:
        raise RuntimeError("source old tiling audit has duplicate keys")
    return {
        "parent_realpath": str(parent),
        "heldout_realpath": str(heldout),
        "file_hashes": hashes,
        "heldout_manifest_rows": 72,
        "heldout_manifest": panel,
        "old_tiling_rows": rows,
        "checkpoint_contract": required_checkpoint,
        "pass": True,
    }


def audit_old_code_references() -> dict[str, Any]:
    result: dict[str, Any] = {"source_factorial": {}, "candidate": {}}
    for seed, (path, expected_sha) in SOURCE_FACTORIAL_CACHES.items():
        if sha256_file(path.resolve(strict=True)) != expected_sha:
            raise RuntimeError(f"source factorial cache changed: {seed}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, Mapping)
            or payload.get("checkpoint_sha256") != AE_CHECKPOINT_SHA256
            or int(payload.get("tiling_seed", -1)) != seed
            or not isinstance(payload.get("codes"), Mapping)
            or len(payload["codes"]) != 432
            or not isinstance(payload.get("native_c_patch"), Mapping)
            or len(payload["native_c_patch"]) != 72
        ):
            raise RuntimeError(f"source factorial cache structure mismatch: {seed}")
        key = f"seed={seed}|depth=0|role=attn_query|enc=cell|code=correct"
        code = payload["codes"].get(key)
        if not isinstance(code, torch.Tensor) or tuple(code.shape) != (144, 512):
            raise RuntimeError(f"source factorial known code missing/invalid: {key}")
        result["source_factorial"][str(seed)] = {
            "path": str(path.resolve(strict=True)),
            "sha256": expected_sha,
            "code_entries": 432,
            "native_c_patch_entries": 72,
            "known_code_key": key,
            "known_code_full_sha256": tensor_sha256(code, dtype=torch.float32),
            "known_code_first64_sha256": tensor_sha256(code[:64], dtype=torch.float32),
            "decision_feature_use": False,
        }
        del payload, code
        gc.collect()
    old_tilings = torch.load(OLD_CANDIDATE_TILINGS, map_location="cpu", weights_only=True)
    old_codes = torch.load(OLD_CANDIDATE_CODES, map_location="cpu", weights_only=True)
    if old_tilings.get("schema_version") != "prospective_tiling_indices_v1" or len(old_tilings.get("entries", {})) != 288:
        raise RuntimeError("old candidate tiling cache structure mismatch")
    if old_codes.get("schema_version") != "prospective_correct_latent_codes_v1" or len(old_codes.get("entries", {})) != 288:
        raise RuntimeError("old candidate code cache structure mismatch")
    result["candidate"] = {
        "tiling_path": str(OLD_CANDIDATE_TILINGS.resolve(strict=True)),
        "tiling_sha256": OLD_CANDIDATE_TILINGS_SHA256,
        "tiling_entries": 288,
        "code_path": str(OLD_CANDIDATE_CODES.resolve(strict=True)),
        "code_sha256": OLD_CANDIDATE_CODES_SHA256,
        "code_entries": 288,
        "decision_feature_use": False,
    }
    del old_tilings, old_codes
    gc.collect()
    return result


def make_partition(size: int, group_size: int, generator: torch.Generator) -> torch.Tensor:
    if size % group_size:
        raise RuntimeError(f"partition divisibility failed: {size}%{group_size}")
    order = torch.randperm(size, generator=generator)
    return torch.stack(
        [order[start : start + group_size].sort().values for start in range(0, size, group_size)]
    ).to(torch.int64).contiguous()


def make_common_tiling(seed: int, shape: tuple[int, int]) -> TilingIndex:
    d_in, d_out = (int(shape[0]), int(shape[1]))
    if (d_in, d_out) not in set(ROLE_SHAPES.values()):
        raise RuntimeError(f"unsupported frozen common-tiling shape: {(d_in, d_out)}")
    identity = f"global_context_common_tiling_v2|shape={d_in}x{d_out}"
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, identity))
    patch_groups = make_partition(d_in // 16, 4, generator)
    offsets = torch.arange(16, dtype=torch.int64)
    rows = torch.stack(
        [
            (group[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values
            for group in patch_groups
        ]
    ).to(torch.int64).contiguous()
    cols = make_partition(d_out, 64, generator)
    row_sha = tensor_sha256(rows, dtype=torch.int64)
    col_sha = tensor_sha256(cols, dtype=torch.int64)
    partition_sha = hashlib.sha256(f"{row_sha}|{col_sha}".encode("utf-8")).hexdigest()
    return TilingIndex(seed, rows, cols, row_sha, col_sha, partition_sha)


def make_old_source_tiling(seed: int, source_key: str, depth: int, role: str) -> TilingIndex:
    d_in, d_out = ROLE_SHAPES[role]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, source_key, role, depth))
    patch_groups = make_partition(d_in // 16, 4, generator)
    offsets = torch.arange(16, dtype=torch.int64)
    rows = torch.stack(
        [(group[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values for group in patch_groups]
    ).to(torch.int64).contiguous()
    cols = make_partition(d_out, 64, generator)
    row_sha = tensor_sha256(rows, dtype=torch.int64)
    col_sha = tensor_sha256(cols, dtype=torch.int64)
    legacy_partition = stable_hex(
        f"seed={seed}",
        *[",".join(str(int(value)) for value in row.tolist()) for row in rows],
        *[",".join(str(int(value)) for value in col.tolist()) for col in cols],
    )
    return TilingIndex(seed, rows, cols, row_sha, col_sha, legacy_partition)


def split_tiles(W: torch.Tensor, tiling: TilingIndex) -> torch.Tensor:
    rows = tiling.rows
    cols = tiling.cols
    return W[
        rows[:, None, :, None],
        cols[None, :, None, :],
    ].reshape(tiling.num_tiles, 64, 64).to(torch.float32).contiguous()


def validate_tiling(W: torch.Tensor, tiling: TilingIndex, label: str) -> dict[str, Any]:
    tiles = split_tiles(W, tiling)
    rebuilt = torch.empty_like(W)
    tile_grid = tiles.reshape(int(tiling.rows.shape[0]), int(tiling.cols.shape[0]), 64, 64)
    rebuilt[
        tiling.rows[:, None, :, None],
        tiling.cols[None, :, None, :],
    ] = tile_grid
    row_coverage_exact = bool(
        torch.equal(
            tiling.rows.reshape(-1).sort().values,
            torch.arange(int(W.shape[0]), dtype=torch.int64),
        )
    )
    column_coverage_exact = bool(
        torch.equal(
            tiling.cols.reshape(-1).sort().values,
            torch.arange(int(W.shape[1]), dtype=torch.int64),
        )
    )
    coordinate_grid = (
        tiling.rows[:, None, :, None] * int(W.shape[1])
        + tiling.cols[None, :, None, :]
    )
    coordinate_rebuilt = torch.empty(tuple(W.shape), dtype=torch.int64)
    coordinate_rebuilt[
        tiling.rows[:, None, :, None],
        tiling.cols[None, :, None, :],
    ] = coordinate_grid
    coordinates = torch.arange(W.numel(), dtype=torch.int64).reshape(W.shape)
    checks = {
        "coverage_min": 1 if row_coverage_exact and column_coverage_exact else 0,
        "coverage_max": 1 if row_coverage_exact and column_coverage_exact else 2,
        "weight_reassembly_bit_exact": bool(torch.equal(rebuilt, W)),
        "coordinate_reassembly_bit_exact": bool(torch.equal(coordinate_rebuilt, coordinates)),
        "coordinate_coverage_exactly_once": row_coverage_exact and column_coverage_exact,
    }
    if checks != {
        "coverage_min": 1,
        "coverage_max": 1,
        "weight_reassembly_bit_exact": True,
        "coordinate_reassembly_bit_exact": True,
        "coordinate_coverage_exactly_once": True,
    }:
        raise RuntimeError(f"tiling split/reassembly failed {label}: {checks}")
    return checks


def _load_offline_dataset_class() -> tuple[type[Any], dict[str, Any]]:
    """Import only the audited source dataset reader before model-code import."""
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    before_model_modules = sorted(
        name for name in sys.modules if name == "big_vae.models" or name.startswith("big_vae.models.")
    )
    if before_model_modules:
        raise RuntimeError(f"Weight-AE model modules loaded before W-only materialization: {before_model_modules}")
    module = __import__("big_vae.datasets.offline", fromlist=["OfflineBigVAEDataset"])
    dataset_class = module.OfflineBigVAEDataset
    after_model_modules = sorted(
        name for name in sys.modules if name == "big_vae.models" or name.startswith("big_vae.models.")
    )
    if after_model_modules:
        raise RuntimeError(f"dataset-only import unexpectedly loaded Weight-AE model code: {after_model_modules}")
    return dataset_class, {
        "module": "big_vae.datasets.offline",
        "module_path": str(Path(module.__file__).resolve(strict=True)),
        "module_sha256": sha256_file(Path(module.__file__).resolve(strict=True)),
        "weight_ae_model_modules_before": before_model_modules,
        "weight_ae_model_modules_after": after_model_modules,
        "pass": True,
    }


def _load_source_w_grid(source_parent_audit: Mapping[str, Any]) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, Any]]:
    dataset_class, import_audit = _load_offline_dataset_class()
    dataset = dataset_class(
        root_dir=SOURCE_OFFLINE_REALPATH.resolve(strict=True),
        shuffle_chunks=False,
        shuffle_records_within_chunk=False,
        repeat=False,
        seed=GLOBAL_SEED,
        weight_cache_size=96,
        sampling_mode="balanced",
        sampling_group_keys=("dataset", "model"),
        sampling_window_size=2048,
        sampling_max_records_per_chunk_round=8,
        x_chunk_cache_size=3,
    )
    weights: dict[tuple[int, str], torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for index, item in enumerate(source_parent_audit["heldout_manifest"]):
        a, b = item["context_a"], item["score_b"]
        sample_a = dataset._shared_sample_from_record_ref(
            chunk_idx=int(a["chunk_idx"]), record_idx=int(a["record_idx"])
        )
        sample_b = dataset._shared_sample_from_record_ref(
            chunk_idx=int(b["chunk_idx"]), record_idx=int(b["record_idx"])
        )
        W_a = sample_a.weight.detach().cpu().to(torch.float32).contiguous()
        W_b = sample_b.weight.detach().cpu().to(torch.float32).contiguous()
        X_a = sample_a.x.detach().cpu().to(torch.float32).contiguous()
        X_b = sample_b.x.detach().cpu().to(torch.float32).contiguous()
        depth, role = int(a["depth"]), str(a["role"])
        if (
            tuple(W_a.shape) != ROLE_SHAPES[role]
            or not torch.equal(W_a, W_b)
            or X_a.ndim != 2
            or X_b.ndim != 2
            or int(X_a.shape[1]) != int(W_a.shape[0])
            or int(X_b.shape[1]) != int(W_a.shape[0])
            or torch.equal(X_a, X_b)
        ):
            raise RuntimeError(f"source runtime A/B record contract failed: row={index} role={role} depth={depth}")
        W = W_a.clone()
        weights[(depth, role)] = W
        records.append(
            {
                "depth": depth,
                "role": role,
                "shape": list(W.shape),
                "source_key": str(a["source_key"]),
                "context_record": [int(a["chunk_idx"]), int(a["record_idx"])],
                "score_record": [int(b["chunk_idx"]), int(b["record_idx"])],
                "weight_shape_bytes_sha256": weight_shape_bytes_sha256(W),
                "weight_tensor_sha256": tensor_sha256(W, dtype=torch.float32),
                "activation_a_shape_destroyed": list(X_a.shape),
                "activation_b_shape_destroyed": list(X_b.shape),
                "activation_a_b_distinct": True,
            }
        )
        del sample_a, sample_b, W_a, W_b, X_a, X_b
    del dataset
    gc.collect()
    if set(weights) != {(depth, role) for depth in range(12) for role in ROLES}:
        raise RuntimeError("source runtime W grid is incomplete")
    return weights, {
        "dataset_import_audit": import_audit,
        "offline_dataset_realpath": str(SOURCE_OFFLINE_REALPATH.resolve(strict=True)),
        "matrix_count": 72,
        "records": records,
        "activation_tensors_retained": 0,
        "pass": True,
    }


def _load_candidate_w_grid(panel_id: str) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, Any]]:
    path = PANEL_PATHS[panel_id].resolve(strict=True)
    if sha256_file(path) != PANEL_SHA256[panel_id]:
        raise RuntimeError(f"candidate panel changed before W extraction: {panel_id}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "prospective_panel_v1"
        or payload.get("panel_id") != panel_id
        or payload.get("quality_status") != "PASS"
        or not isinstance(payload.get("weights"), Mapping)
        or not isinstance(payload.get("matrix_meta"), Mapping)
        or not isinstance(payload.get("activations"), Mapping)
        or set(payload["activations"]) != {"A", "B"}
    ):
        raise RuntimeError(f"candidate panel header/activation-container contract failed: {panel_id}")
    expected_keys = {matrix_key(depth, role) for depth in range(12) for role in ROLES}
    if set(payload["weights"]) != expected_keys or set(payload["matrix_meta"]) != expected_keys:
        raise RuntimeError(f"candidate panel W/meta grid mismatch: {panel_id}")
    for split in ("A", "B"):
        if not isinstance(payload["activations"][split], Mapping) or set(payload["activations"][split]) != expected_keys:
            raise RuntimeError(f"candidate activation grid mismatch: {panel_id}/{split}")
    weights: dict[tuple[int, str], torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for depth in range(12):
        for role in ROLES:
            key = matrix_key(depth, role)
            source = payload["weights"][key]
            meta = payload["matrix_meta"][key]
            if (
                not isinstance(source, torch.Tensor)
                or source.device.type != "cpu"
                or source.dtype != torch.float32
                or tuple(source.shape) != ROLE_SHAPES[role]
                or not bool(torch.isfinite(source).all())
                or not isinstance(meta, Mapping)
                or int(meta.get("depth", -1)) != depth
                or meta.get("role") != role
                or (int(meta.get("d_in", -1)), int(meta.get("d_out", -1))) != ROLE_SHAPES[role]
            ):
                raise RuntimeError(f"candidate W/meta entry invalid: {panel_id}/{key}")
            W = source.detach().cpu().contiguous().clone()
            shape_hash = weight_shape_bytes_sha256(W)
            if meta.get("weight_shape_bytes_sha256") != shape_hash:
                raise RuntimeError(f"candidate internal W hash mismatch: {panel_id}/{key}")
            weights[(depth, role)] = W
            records.append(
                {
                    "depth": depth,
                    "role": role,
                    "shape": list(W.shape),
                    "module_name": str(meta.get("module_name", "")),
                    "weight_shape_bytes_sha256": shape_hash,
                    "weight_tensor_sha256": tensor_sha256(W, dtype=torch.float32),
                }
            )
    activation_container_summary = {
        "split_keys": ["A", "B"],
        "matrix_entries_per_split": 72,
        "values_read_by_w_extractor": 0,
        "values_copied": 0,
        "container_destroyed_before_model_import": True,
    }
    del payload
    gc.collect()
    return weights, {
        "panel_id": panel_id,
        "panel_path": str(path),
        "panel_sha256": PANEL_SHA256[panel_id],
        "matrix_count": 72,
        "records": records,
        "activation_container": activation_container_summary,
        "activation_tensors_retained": 0,
        "pass": True,
    }


def materialize_w_only(
    source_parent_audit: Mapping[str, Any],
    old_reference_audit: Mapping[str, Any],
) -> WOnlyBundle:
    if any(name == "big_vae.models" or name.startswith("big_vae.models.") for name in sys.modules):
        raise RuntimeError("Weight-AE model code was imported before W-only materialization")
    source_weights, source_runtime = _load_source_w_grid(source_parent_audit)
    panel_weights: dict[str, dict[tuple[int, str], torch.Tensor]] = {
        "source_vit_b_flickr": source_weights,
    }
    candidate_audits: dict[str, Any] = {}
    for panel_id in ("beans", "trocr_sroie"):
        panel_weights[panel_id], candidate_audits[panel_id] = _load_candidate_w_grid(panel_id)

    entries: list[WOnlyEntry] = []
    tiling_rows: list[dict[str, Any]] = []
    reference_partitions: dict[tuple[int, tuple[int, int]], tuple[torch.Tensor, torch.Tensor, str]] = {}
    for panel_id in PANEL_IDS:
        for seed_index, seed in enumerate(COMMON_TILING_SEEDS, start=1):
            for depth in range(12):
                for role in ROLES:
                    W = panel_weights[panel_id][(depth, role)]
                    tiling = make_common_tiling(seed, tuple(int(value) for value in W.shape))
                    checks = validate_tiling(W, tiling, f"{panel_id}/{seed}/{depth}/{role}")
                    identity = (seed, tuple(int(value) for value in W.shape))
                    if identity not in reference_partitions:
                        reference_partitions[identity] = (
                            tiling.rows.clone(), tiling.cols.clone(), tiling.partition_sha256
                        )
                    reference_rows, reference_cols, reference_sha = reference_partitions[identity]
                    cross_panel_equal = bool(
                        torch.equal(tiling.rows, reference_rows)
                        and torch.equal(tiling.cols, reference_cols)
                        and tiling.partition_sha256 == reference_sha
                    )
                    if not cross_panel_equal:
                        raise RuntimeError(
                            f"shape-only common tiling differs across a same-shape cell: "
                            f"{identity}/{panel_id}/{depth}/{role}"
                        )
                    shape_hash = weight_shape_bytes_sha256(W)
                    tiling_rows.append(
                        {
                            "panel_id": panel_id,
                            "tiling_seed": seed,
                            "tiling_index": seed_index,
                            "depth": depth,
                            "role": role,
                            "matrix_key": matrix_key(depth, role),
                            "d_in": int(W.shape[0]),
                            "d_out": int(W.shape[1]),
                            "row_groups": int(tiling.rows.shape[0]),
                            "col_groups": int(tiling.cols.shape[0]),
                            "num_tiles": tiling.num_tiles,
                            "row_index_sha256": tiling.row_sha256,
                            "column_index_sha256": tiling.column_sha256,
                            "partition_sha256": tiling.partition_sha256,
                            "weight_shape_bytes_sha256": shape_hash,
                            **{
                                name: value for name, value in checks.items()
                                if name != "coordinate_coverage_exactly_once"
                            },
                            "shape_global_identity_bit_exact": True,
                        }
                    )
                    if seed_index == 1:
                        entries.append(
                            WOnlyEntry(
                                panel_id=panel_id,
                                depth=depth,
                                role=role,
                                W=W,
                                weight_shape_bytes_sha256=shape_hash,
                                weight_tensor_sha256=tensor_sha256(W, dtype=torch.float32),
                                tilings={seed: tiling},
                            )
                        )
                    else:
                        target = entries[
                            PANEL_IDS.index(panel_id) * 72 + depth * len(ROLES) + ROLES.index(role)
                        ]
                        target.tilings[seed] = tiling

    for weights in panel_weights.values():
        weights.clear()
    panel_weights.clear()
    del source_weights
    gc.collect()
    if len(entries) != 216 or len(tiling_rows) != 432:
        raise RuntimeError(f"W-only/tiling grid count mismatch: {len(entries)}/{len(tiling_rows)}")

    source_first_manifest = next(
        item for item in source_parent_audit["heldout_manifest"]
        if int(item["context_a"]["depth"]) == 0 and item["context_a"]["role"] == "attn_query"
    )
    source_key = str(source_first_manifest["context_a"]["source_key"])
    preflight_tiling = make_old_source_tiling(26081601, source_key, 0, "attn_query")
    old_tiling_row = next(
        row for row in source_parent_audit["old_tiling_rows"]
        if int(row["tiling_seed"]) == 26081601 and int(row["depth"]) == 0 and row["role"] == "attn_query"
    )
    if preflight_tiling.partition_sha256 != old_tiling_row["partition_sha256"]:
        raise RuntimeError("known-source old tiling partition drift")
    source_entry = entries[0]
    validate_tiling(source_entry.W, preflight_tiling, "known-source-preflight")
    known_reference = old_reference_audit["source_factorial"]["26081601"]

    allowed_entry_fields = {
        "panel_id", "depth", "role", "W", "weight_shape_bytes_sha256", "weight_tensor_sha256", "tilings"
    }
    for entry in entries:
        if set(vars(entry)) != allowed_entry_fields:
            raise RuntimeError(f"W-only entry field escape: {set(vars(entry)) - allowed_entry_fields}")
        if entry.W.device.type != "cpu" or entry.W.dtype != torch.float32 or entry.W.ndim != 2:
            raise RuntimeError("W-only entry tensor contract failed")
    w_grid_rows = [
        {
            "panel_id": entry.panel_id,
            "depth": entry.depth,
            "role": entry.role,
            "shape": list(entry.W.shape),
            "weight_shape_bytes_sha256": entry.weight_shape_bytes_sha256,
            "weight_tensor_sha256": entry.weight_tensor_sha256,
        }
        for entry in entries
    ]
    w_grid_digest = hashlib.sha256(
        json.dumps(w_grid_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    tiling_digest = hashlib.sha256(
        json.dumps(tiling_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "schema_version": "global_context_w_only_materialization_v1",
        "source_runtime": source_runtime,
        "candidate_runtime": candidate_audits,
        "w_entry_count": len(entries),
        "w_grid_rows": w_grid_rows,
        "w_grid_sha256": w_grid_digest,
        "tiling_row_count": len(tiling_rows),
        "tiling_grid_sha256": tiling_digest,
        "source_preflight": {
            "source_key_destroyed_after_tiling_construction": True,
            "tiling_seed": 26081601,
            "partition_sha256": preflight_tiling.partition_sha256,
            "cached_code_key": known_reference["known_code_key"],
            "cached_first64_sha256": known_reference["known_code_first64_sha256"],
        },
        "w_only_entry_fields": sorted(allowed_entry_fields),
        "activation_tensors_retained": 0,
        "activation_references_retained": 0,
        "model_modules_loaded_before_completion": [],
        "garbage_collection_completed": True,
        "pass": True,
    }
    return WOnlyBundle(
        entries=entries,
        tiling_rows=tiling_rows,
        source_preflight_tiling=preflight_tiling,
        source_preflight_cached_slice_sha256=str(known_reference["known_code_first64_sha256"]),
        source_preflight_code_key=str(known_reference["known_code_key"]),
        materialization_audit=audit,
    )


def destroy_w_only(bundle: WOnlyBundle) -> None:
    for entry in bundle.entries:
        entry.tilings.clear()
    bundle.entries.clear()
    bundle.tiling_rows.clear()
    gc.collect()


def build_scientific_input_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit every input and return only JSON-safe summaries plus templates."""
    assert_seal_clean()
    resources = file_resources()
    source_parent = audit_source_parent()
    old_references = audit_old_code_references()
    raw_templates = torch.load(SOURCE_TEMPLATE, map_location="cpu", weights_only=True)
    templates, template_audit = validate_templates(raw_templates)
    del raw_templates
    gc.collect()
    bundle = materialize_w_only(source_parent, old_references)
    materialization_audit = bundle.materialization_audit
    zero_weight_sanity = build_zero_weight_sanity(bundle)
    destroy_w_only(bundle)
    assert_seal_clean()
    model_modules = sorted(
        name for name in sys.modules if name == "big_vae.models" or name.startswith("big_vae.models.")
    )
    if model_modules:
        raise RuntimeError(f"input audit imported Weight-AE model code: {model_modules}")
    audit = {
        "schema_version": "global_context_input_audit_v1",
        "resources": resources,
        "source_parent_audit": source_parent,
        "old_reference_audit": old_references,
        "source_template_audit": template_audit,
        "w_only_materialization": materialization_audit,
        "zero_weight_sanity_preflight": zero_weight_sanity,
        "weight_ae_model_modules_loaded": model_modules,
        "weight_ae_forward_count": 0,
        "decoder_forward_count": 0,
        "distribution_encoder_forward_count": 0,
        "source_only_seal_clean": True,
        "pass": True,
    }
    return audit, templates


def make_contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    assert_seal_clean()
    # This gate must run before dependency or scientific-input traversal.
    frozen_execution_config = execution_config(args)
    scripts = script_snapshot(args)
    dependencies = dependency_snapshot()
    input_audit, templates = build_scientific_input_audit()
    del templates
    gc.collect()
    static_audit = static_call_graph_audit()
    if not static_audit["pass"]:
        raise RuntimeError(f"static dataflow audit failed: {static_audit}")
    return {
        "schema_version": "global_context_latent_geometry_preexecution_contract_v1",
        "contract_schema_keys": list(CONTRACT_SCHEMA_KEYS),
        "frozen_design_path": str(DESIGN_PATH.resolve(strict=True)),
        "frozen_design_sha256": DESIGN_SHA256,
        "runner_path": scripts["runner"]["path"],
        "runner_sha256": scripts["runner"]["sha256"],
        "analyzer_path": scripts["analyzer"]["path"],
        "analyzer_sha256": scripts["analyzer"]["sha256"],
        "runtime": runtime_snapshot(),
        "execution_config": frozen_execution_config,
        "frozen_grids": {
            "panel_order": list(PANEL_IDS),
            "role_order": list(ROLES),
            "depths": list(range(12)),
            "common_tiling_seeds": list(COMMON_TILING_SEEDS),
            "code_representation_order": list(CODE_REPRESENTATIONS),
            "representation_order": list(REPRESENTATIONS),
            "untrained_seeds": list(UNTRAINED_SEEDS),
            "countsketch_seeds": list(COUNTSKETCH_SEEDS),
            "role_shapes": {role: list(ROLE_SHAPES[role]) for role in ROLES},
        },
        "scripts": scripts,
        "model_dependencies": dependencies,
        "input_audit": input_audit,
        "static_call_graph_audit": static_audit,
        "expected_counts": expected_counts(),
        "artifact_contract": {
            "self_excluded_files": list(EXPECTED_RAW_FILES),
            "declared_file_count": 21,
            "file_count_including_manifest": 22,
            "artifact_manifest_self_excluded": True,
            "runner_and_analyzer_outputs_disjoint": True,
            "no_symlinks": True,
        },
        "dataflow_contract": {
            "seal_installed_before_scientific_reads": True,
            "w_only_materialization_before_model_import_and_build": True,
            "encoder_function_positional_arguments": ["model", "W_tiles", "fixed_template"],
            "primary_global_condition_hashes": GLOBAL_C_HASHES,
            "panel_activation_model_consumers": 0,
            "metadata_label_model_consumers": 0,
            "distribution_encoder_calls": 0,
            "decoder_calls": 0,
            "old_code_or_tiling_decision_features": False,
            "analyzer_invoked_by_runner": False,
        },
        "analysis_contract": {
            "variance_decomposition": {
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
        },
    }


def audit_only(args: argparse.Namespace) -> int:
    assert_canonical_formal_entrypoint_and_argv(args)
    assert_exact_formal_paths(args)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": "global_context_latent_geometry_audit_only_v1",
        "created_utc": utc_now(),
        "mode": "AUDIT_ONLY_NO_WEIGHT_AE_MODEL_IMPORT_OR_FORWARD",
        "status": "BLOCKED",
        "checks": {},
        "errors": [],
    }
    checks: tuple[tuple[str, Any], ...] = (
        ("runtime", runtime_snapshot),
        ("execution_config", lambda: execution_config(args)),
        ("scripts", lambda: script_snapshot(args)),
        ("model_dependencies", dependency_snapshot),
        ("scientific_inputs", build_scientific_input_audit),
        ("static_call_graph", static_call_graph_audit),
    )
    for name, function in checks:
        print(f"stage=audit check={name}", flush=True)
        try:
            value = function()
            if name == "scientific_inputs":
                input_audit, templates = value
                del templates
                value = input_audit
            report["checks"][name] = {"pass": True, "value": value}
        except Exception as error:  # audit mode deliberately accumulates independent blockers
            report["checks"][name] = {
                "pass": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            report["errors"].append(
                {"stage": name, "type": type(error).__name__, "message": str(error)}
            )
    try:
        assert_seal_clean()
        report["checks"]["source_only_seal"] = {"pass": True, "value": dict(_SEAL_STATE)}
    except Exception as error:
        report["checks"]["source_only_seal"] = {
            "pass": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        report["errors"].append(
            {"stage": "source_only_seal", "type": type(error).__name__, "message": str(error)}
        )
    model_modules = sorted(
        name for name in sys.modules if name == "big_vae.models" or name.startswith("big_vae.models.")
    )
    report.update(
        {
            "weight_ae_model_modules_loaded": model_modules,
            "weight_ae_forward_count": 0,
            "decoder_forward_count": 0,
            "distribution_encoder_forward_count": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    if not report["errors"] and not model_modules:
        report["status"] = "READY_FOR_EXTERNAL_CONTRACT_PREPARATION"
    if args.audit_report is not None:
        destination = args.audit_report.resolve(strict=False)
        if destination.exists():
            raise RuntimeError(f"audit report path must be fresh: {destination}")
        atomic_write_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if not report["errors"] else 1


def prepare_contract(args: argparse.Namespace) -> None:
    assert_canonical_formal_entrypoint_and_argv(args)
    assert_exact_formal_paths(args)
    contract_path = args.contract_path.resolve(strict=False)
    if contract_path.exists():
        raise RuntimeError(f"external preexecution contract must be fresh: {contract_path}")
    print("stage=prepare_contract scientific_input_audit=true weight_ae_forward=false", flush=True)
    payload = make_contract_payload(args)
    payload["created_utc"] = utc_now()
    payload["source_only_seal"] = dict(_SEAL_STATE)
    if set(payload) != set(CONTRACT_SCHEMA_KEYS):
        raise RuntimeError(f"prepared contract top-level schema mismatch: {sorted(payload)}")
    assert_seal_clean()
    atomic_write_json(contract_path, payload)
    print(
        f"stage=contract_written path={contract_path} sha256={sha256_file(contract_path)} "
        "weight_ae_forward=false",
        flush=True,
    )


def verify_contract(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    contract_path = args.contract_path.resolve(strict=True)
    contract_sha = sha256_file(contract_path)
    contract = read_json(contract_path)
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema_version") != "global_context_latent_geometry_preexecution_contract_v1"
    ):
        raise RuntimeError(f"invalid external contract schema: {contract_path}")
    if set(contract) != set(CONTRACT_SCHEMA_KEYS) or contract.get("contract_schema_keys") != list(CONTRACT_SCHEMA_KEYS):
        raise RuntimeError(f"external contract exact top-level schema mismatch: {sorted(contract)}")
    current = make_contract_payload(args)
    assert_contract_static_fields_match(contract, current)
    seal = contract.get("source_only_seal")
    if not isinstance(seal, Mapping) or seal.get("installed_before_scientific_input_read") is not True:
        raise RuntimeError("external contract lacks pre-read source-only seal")
    for key in ("target_access_events", "network_connections", "subprocess_launches"):
        if int(seal.get(key, -1)) != 0:
            raise RuntimeError(f"external contract recorded nonzero {key}")
    assert_seal_clean()
    return dict(contract), contract_sha


def assert_contract_static_fields_match(contract: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for key, expected in current.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"preexecution contract drift at {key}")


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("global_context_latent_geometry_ablation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def close_logging(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def assert_loaded_local_modules_frozen(contract: Mapping[str, Any]) -> dict[str, Any]:
    workspace = WORKSPACE_ROOT.resolve(strict=True)
    allowed = {
        Path(record["path"]).resolve(strict=True)
        for record in contract["model_dependencies"].values()
    }
    allowed.update(
        Path(record["path"]).resolve(strict=True)
        for record in contract["scripts"].values()
    )
    loaded: list[dict[str, str]] = []
    escaped: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw).resolve(strict=False)
        try:
            path.relative_to(workspace)
        except ValueError:
            continue
        path = Path(raw).resolve(strict=True)
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(importlib.util.source_from_cache(str(path))).resolve(strict=True)
            except (ValueError, FileNotFoundError):
                pass
        record = {"module": name, "path": str(path)}
        loaded.append(record)
        if path not in allowed:
            escaped.append(record)
    if escaped:
        raise RuntimeError(f"loaded local modules escaped dependency closure: {escaped}")
    return {
        "pass": True,
        "allowed_file_count": len(allowed),
        "loaded_module_count": len(loaded),
        "loaded_modules": loaded,
    }


def import_model_helpers(contract: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    if str(EXPERIMENTS_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_ROOT))
    import source_confirmatory_g01_gate as gate  # type: ignore[import-not-found]  # noqa: PLC0415
    import source_latent_code_factorial as factorial  # type: ignore[import-not-found]  # noqa: PLC0415

    closure = assert_loaded_local_modules_frozen(contract)
    assert_seal_clean()
    return gate, factorial, closure


def _state_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.to(torch.float32)
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    parameters: list[dict[str, Any]] = []
    parameter_hashes: dict[str, str] = {}
    for name, value in model.named_parameters():
        digest = _state_tensor_sha256(value)
        parameters.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "requires_grad": bool(value.requires_grad),
                "tensor_sha256": digest,
            }
        )
        parameter_hashes[name] = digest
    buffers: list[dict[str, Any]] = []
    for name, value in model.named_buffers():
        buffers.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": _state_tensor_sha256(value),
            }
        )
    state_rows: list[dict[str, Any]] = []
    for name, value in model.state_dict().items():
        state_rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": _state_tensor_sha256(value),
            }
        )
    state_hash = hashlib.sha256(
        json.dumps(state_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    grid = [(row["name"], row["shape"], row["dtype"]) for row in parameters]
    grid_hash = hashlib.sha256(
        json.dumps(grid, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "parameter_count": len(parameters),
        "parameter_numel": int(sum(value.numel() for value in model.parameters())),
        "parameter_grid_sha256": grid_hash,
        "parameters": parameters,
        "buffers": buffers,
        "state_dict_entry_count": len(state_rows),
        "state_dict_sha256": state_hash,
        "parameter_hashes": parameter_hashes,
    }


def assert_model_fully_eval(model: torch.nn.Module, *, label: str) -> None:
    training_modules = [name for name, module in model.named_modules() if module.training]
    if model.training or training_modules:
        raise RuntimeError(
            f"{label} remained/entered training mode: model.training={model.training} "
            f"training_modules={training_modules[:20]}"
        )


def build_learned_model(
    factorial: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, bool, torch.dtype | None, dict[str, Any]]:
    model, amp_enabled, amp_dtype, source_contract = factorial.build_model(
        checkpoint_path=AE_CHECKPOINT,
        device=device,
        no_amp=False,
    )
    required = {
        "checkpoint_sha256": AE_CHECKPOINT_SHA256,
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
    observed = {key: source_contract.get(key) for key in required}
    if observed != required or amp_enabled is not True or amp_dtype is not torch.bfloat16:
        raise RuntimeError(f"learned model contract mismatch: {observed} != {required}")
    assert_model_fully_eval(model, label="learned checkpoint before numeric preflight")
    fingerprint = model_fingerprint(model)
    return model, amp_enabled, amp_dtype, {
        "kind": "learned_strict_checkpoint",
        "source_contract": source_contract,
        "required_contract": required,
        "fingerprint": fingerprint,
        "eval_mode": True,
        "all_submodules_eval": True,
        "checkpoint_tensors_loaded": True,
        "pass": True,
    }


def build_untrained_model(
    gate: Any,
    seed: int,
    device: torch.device,
    learned_fingerprint: Mapping[str, Any],
    learned_resolved_model_config: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_cfg = copy.deepcopy(learned_resolved_model_config)
    model_cfg.big_vae.use_latent_sampling = False
    model_cfg.big_vae.rope_2d_coord_kind = "raw"
    model_cfg.big_vae.disable_z_shortcut = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = gate.build_weight_quantile_vae(model_cfg)
    del model_cfg
    fingerprint = model_fingerprint(model)
    if fingerprint["parameter_grid_sha256"] != learned_fingerprint["parameter_grid_sha256"]:
        raise RuntimeError(f"untrained parameter grid differs from learned model: seed={seed}")
    learned_hashes = learned_fingerprint["parameter_hashes"]
    overlap_names = sorted(
        name for name, digest in fingerprint["parameter_hashes"].items()
        if learned_hashes.get(name) == digest
    )
    if fingerprint["state_dict_sha256"] == learned_fingerprint["state_dict_sha256"]:
        raise RuntimeError(f"untrained state unexpectedly equals checkpoint state: seed={seed}")
    if bool(model.cfg.big_vae.use_latent_sampling) or str(model.rope_2d_coord_kind) != "raw":
        raise RuntimeError(f"untrained deterministic/rope contract failed: seed={seed}")
    initialization_audit = {
        "kind": "same_architecture_fresh_default_initialization",
        "seed": seed,
        "cpu_torch_random_fork_rng": True,
        "checkpoint_tensor_copy_count": 0,
        "checkpoint_state_dict_loaded": False,
        "architecture_config_source": "deepcopy_of_strict_learned_model_resolved_config",
        "parameter_name_shape_dtype_grid_matches_learned": True,
        "parameter_hash_overlap_with_learned_count": len(overlap_names),
        "parameter_hash_overlap_with_learned_names": overlap_names,
        "state_dict_differs_from_learned": True,
        "fingerprint": fingerprint,
        "use_latent_sampling": False,
        "rope_2d_coord_kind": "raw",
        "disable_z_shortcut": True,
    }
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    initialization_audit["eval_mode"] = True
    initialization_audit["pass"] = True
    return model, initialization_audit


_ENCODE_RUNTIME: dict[str, Any] = {}


def encode_weight_tiles(model: torch.nn.Module, W_tiles: torch.Tensor, fixed_template: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """The only new-code model entry; scientific arguments are exactly W and C."""
    runtime = _ENCODE_RUNTIME
    required_runtime = {
        "factorial", "gate", "device", "amp_enabled", "amp_dtype", "batch_size",
        "log_every_batches", "logger", "call_records", "call_ordinal",
    }
    if set(runtime) != required_runtime:
        raise RuntimeError(f"encoder runtime is incomplete/has extra fields: {set(runtime)}")
    if (
        not isinstance(W_tiles, torch.Tensor)
        or W_tiles.device.type != "cpu"
        or W_tiles.dtype != torch.float32
        or W_tiles.ndim != 3
        or tuple(W_tiles.shape[1:]) != (64, 64)
        or not bool(torch.isfinite(W_tiles).all())
        or not isinstance(fixed_template, Mapping)
        or set(fixed_template) != {"c_var", "c_patch"}
    ):
        raise RuntimeError("encode_weight_tiles input contract failed")
    template_hashes = {
        name: tensor_sha256(fixed_template[name], dtype=torch.float32)
        for name in ("c_var", "c_patch")
    }
    for name in ("c_var", "c_patch"):
        tensor = fixed_template[name]
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (256,)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise RuntimeError(f"fixed template tensor contract failed: {name}")
    outputs: list[torch.Tensor] = []
    batch_records: list[dict[str, Any]] = []
    batch_size = int(runtime["batch_size"])
    total_batches = math.ceil(int(W_tiles.shape[0]) / batch_size)
    started = time.monotonic()
    for batch_index, start in enumerate(range(0, int(W_tiles.shape[0]), batch_size), start=1):
        stop = min(start + batch_size, int(W_tiles.shape[0]))
        batch = W_tiles[start:stop].to(
            device=runtime["device"], dtype=torch.float32, non_blocking=True
        )
        condition = runtime["factorial"].fixed_condition(
            fixed_template,
            batch=stop - start,
            device=runtime["device"],
            dtype=batch.dtype,
        )
        with runtime["gate"]._autocast_context(
            enabled=runtime["amp_enabled"], dtype=runtime["amp_dtype"]
        ):
            code = runtime["factorial"].encode_z_dec(model, batch, condition)
        code = code.detach().cpu().to(torch.float32).contiguous()
        if tuple(code.shape) != (stop - start, 512) or not bool(torch.isfinite(code).all()):
            raise RuntimeError(f"encoder returned invalid code batch: {tuple(code.shape)}")
        outputs.append(code)
        batch_records.append(
            {
                "batch_index": batch_index,
                "batch_rows": stop - start,
                "c_var_template_sha256": template_hashes["c_var"],
                "c_patch_template_sha256": template_hashes["c_patch"],
                "activation_consumers": 0,
                "metadata_label_consumers": 0,
                "distribution_encoder_calls": 0,
                "decoder_calls": 0,
            }
        )
        if batch_index % int(runtime["log_every_batches"]) == 0 or batch_index == total_batches:
            elapsed = time.monotonic() - started
            runtime["logger"].info(
                "stage=encode call=%d batch=%d/%d tiles=%d/%d rate=%.2f_tiles_s elapsed=%.1fs",
                int(runtime["call_ordinal"]),
                batch_index,
                total_batches,
                stop,
                int(W_tiles.shape[0]),
                stop / max(elapsed, 1e-9),
                elapsed,
            )
    result = torch.cat(outputs, dim=0)
    if tuple(result.shape) != (int(W_tiles.shape[0]), 512) or int(torch.count_nonzero(result)) == 0:
        raise RuntimeError(f"assembled code is invalid/zero: {tuple(result.shape)}")
    runtime["call_records"].append(
        {
            "call_ordinal": int(runtime["call_ordinal"]),
            "tile_rows": int(W_tiles.shape[0]),
            "code_rows": int(result.shape[0]),
            "code_width": 512,
            "template_hashes": template_hashes,
            "batches": batch_records,
            "activation_consumers": 0,
            "metadata_label_consumers": 0,
            "distribution_encoder_calls": 0,
            "decoder_calls": 0,
        }
    )
    runtime["call_ordinal"] = int(runtime["call_ordinal"]) + 1
    return result


def static_call_graph_audit() -> dict[str, Any]:
    source = Path(__file__).resolve(strict=True).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "encode_weight_tiles"
    )
    arguments = [argument.arg for argument in function.args.args]
    if arguments != ["model", "W_tiles", "fixed_template"] or function.args.vararg or function.args.kwarg:
        raise RuntimeError(f"encoder function signature drift: {arguments}")
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    forbidden_scientific_names = sorted(
        name for name in names if name.casefold() in {
            "panel", "panel_id", "role", "depth", "dataset", "label", "activation", "activations"
        }
    )
    attribute_calls = [
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    forbidden_model_calls = sorted(
        name for name in attribute_calls if "decode" in name.casefold() or "distribution" in name.casefold()
    )
    encode_subscript_calls = sum(name == "encode_z_dec" for name in attribute_calls)
    result = {
        "schema_version": "global_context_static_call_graph_audit_v1",
        "script_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "encoder_function": "encode_weight_tiles",
        "formal_arguments": arguments,
        "forbidden_scientific_argument_or_local_names": forbidden_scientific_names,
        "forbidden_decoder_or_distribution_attribute_calls": forbidden_model_calls,
        "encode_z_dec_subscript_call_count": encode_subscript_calls,
        "model_receives_only_weight_batch_and_expanded_fixed_condition": True,
    }
    result["pass"] = bool(
        arguments == ["model", "W_tiles", "fixed_template"]
        and not forbidden_scientific_names
        and not forbidden_model_calls
        and encode_subscript_calls == 1
    )
    return result


def aggregate_tile_features(values: torch.Tensor) -> np.ndarray:
    if values.ndim != 2 or int(values.shape[1]) != 512 or not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"invalid tile feature tensor: {tuple(values.shape)}")
    array = values.detach().cpu().to(torch.float64).numpy()
    blocks = (
        np.mean(array, axis=0),
        np.std(array, axis=0, ddof=0),
        np.quantile(array, 0.10, axis=0, method="linear"),
        np.quantile(array, 0.50, axis=0, method="linear"),
        np.quantile(array, 0.90, axis=0, method="linear"),
    )
    result = np.concatenate(blocks).astype(np.float64, copy=False)
    if result.shape != (2560,) or not bool(np.isfinite(result).all()):
        raise RuntimeError(f"aggregate feature shape/content invalid: {result.shape}")
    return result


def raw_simple_descriptor(W: torch.Tensor) -> np.ndarray:
    array = W.detach().cpu().to(torch.float64).numpy()
    if array.ndim != 2 or not bool(np.isfinite(array).all()):
        raise RuntimeError("raw_simple requires a finite 2-D W")
    d_in, d_out = array.shape
    flat = array.reshape(-1)
    row_rms = np.sqrt(np.mean(np.square(array), axis=1))
    column_rms = np.sqrt(np.mean(np.square(array), axis=0))

    def rms_summary(value: np.ndarray) -> list[float]:
        return [
            float(np.mean(value)),
            float(np.std(value, ddof=0)),
            float(np.min(value)),
            float(np.quantile(value, 0.10, method="linear")),
            float(np.quantile(value, 0.25, method="linear")),
            float(np.quantile(value, 0.50, method="linear")),
            float(np.quantile(value, 0.75, method="linear")),
            float(np.quantile(value, 0.90, method="linear")),
            float(np.max(value)),
        ]

    result = np.asarray(
        [
            math.log(float(flat.size)), math.log(float(d_in)), math.log(float(d_out)),
            float(np.mean(flat)), float(np.std(flat, ddof=0)),
            float(np.sqrt(np.mean(np.square(flat)))), float(np.linalg.norm(flat)),
            float(np.mean(np.abs(flat))), float(np.min(flat)), float(np.max(flat)),
            *[float(np.quantile(flat, q, method="linear")) for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)],
            *rms_summary(row_rms),
            *rms_summary(column_rms),
        ],
        dtype=np.float64,
    )
    if result.shape != (37,) or not bool(np.isfinite(result).all()):
        raise RuntimeError(f"raw_simple descriptor invalid: {result.shape}")
    return result


def build_countsketch_maps() -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    maps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    payload: dict[str, Any] = {
        "schema_version": "global_context_countsketch_maps_v1",
        "coordinate_count_per_seed": 4096,
        "output_width": 512,
        "seeds": {},
    }
    for seed in COUNTSKETCH_SEEDS:
        buckets: list[int] = []
        signs: list[int] = []
        pairs: list[dict[str, int]] = []
        for coordinate in range(4096):
            digest = hashlib.sha256(
                f"global_context_countsketch_v1|{seed}|{coordinate}".encode("utf-8")
            ).digest()
            bucket = int.from_bytes(digest[0:8], "big") % 512
            sign = 1 if (digest[8] & 1) == 0 else -1
            buckets.append(bucket)
            signs.append(sign)
            pairs.append({"coordinate": coordinate, "bucket": bucket, "sign": sign})
        bucket_tensor = torch.tensor(buckets, dtype=torch.int64)
        sign_tensor = torch.tensor(signs, dtype=torch.int64)
        bucket_hash = tensor_sha256(bucket_tensor, dtype=torch.int64)
        sign_hash = tensor_sha256(sign_tensor, dtype=torch.int64)
        pair_hash = hashlib.sha256(f"{bucket_hash}|{sign_hash}".encode("utf-8")).hexdigest()
        maps[seed] = (bucket_tensor, sign_tensor)
        payload["seeds"][str(seed)] = {
            "seed": seed,
            "pairs": pairs,
            "bucket_tensor_sha256": bucket_hash,
            "sign_tensor_sha256": sign_hash,
            "pair_map_sha256": pair_hash,
            "all_buckets_in_range": bool(torch.all((bucket_tensor >= 0) & (bucket_tensor < 512))),
            "all_signs_exact_plus_or_minus_one": bool(torch.all(sign_tensor.abs() == 1)),
        }
    payload["total_pair_count"] = sum(len(value["pairs"]) for value in payload["seeds"].values())
    if payload["total_pair_count"] != 20480:
        raise RuntimeError("CountSketch pair grid mismatch")
    return maps, payload


def apply_countsketch(tiles: torch.Tensor, buckets: torch.Tensor, signs: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tuple(tiles.shape[1:]) != (64, 64):
        raise RuntimeError(f"CountSketch tile shape mismatch: {tuple(tiles.shape)}")
    flat = tiles.reshape(int(tiles.shape[0]), 4096).to(device=device, dtype=torch.float64)
    bucket_device = buckets.to(device=device, dtype=torch.int64)
    sign_device = signs.to(device=device, dtype=torch.float64)
    result = torch.zeros((int(tiles.shape[0]), 512), dtype=torch.float64, device=device)
    result.scatter_add_(1, bucket_device.view(1, -1).expand(int(tiles.shape[0]), -1), flat * sign_device)
    result = result.detach().cpu().contiguous()
    if tuple(result.shape) != (int(tiles.shape[0]), 512) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("CountSketch output invalid")
    return result


def canonical_cells(bundle: WOnlyBundle) -> list[tuple[WOnlyEntry, TilingIndex, int]]:
    lookup = {(entry.panel_id, entry.depth, entry.role): entry for entry in bundle.entries}
    result: list[tuple[WOnlyEntry, TilingIndex, int]] = []
    for panel_id in PANEL_IDS:
        for tiling_index, seed in enumerate(COMMON_TILING_SEEDS, start=1):
            for depth in range(12):
                for role in ROLES:
                    entry = lookup[(panel_id, depth, role)]
                    result.append((entry, entry.tilings[seed], tiling_index))
    if len(result) != 432:
        raise RuntimeError(f"canonical aggregate cell grid mismatch: {len(result)}")
    return result


CODE_MANIFEST_FIELDS = (
    "representation", "panel_id", "tiling_seed", "tiling_index", "depth", "role",
    "matrix_key", "d_in", "d_out", "num_tiles", "code_width", "code_key",
    "code_tensor_sha256", "partition_sha256", "weight_shape_bytes_sha256",
    "c_var_sha256", "c_patch_sha256", "model_kind", "model_seed", "condition_kind",
    "activation_consumers", "metadata_label_consumers", "finite", "nonzero",
)
AGGREGATE_MANIFEST_FIELDS = (
    "representation", "panel_id", "tiling_seed", "tiling_index", "depth", "role",
    "matrix_key", "d_in", "d_out", "num_tiles", "feature_dim", "array_name",
    "array_row_index", "source_kind", "source_seed", "feature_tensor_sha256",
    "partition_sha256", "tiling_invariant", "finite", "nonzero",
)
TILING_MANIFEST_FIELDS = (
    "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
    "d_in", "d_out", "row_groups", "col_groups", "num_tiles", "row_index_sha256",
    "column_index_sha256", "partition_sha256", "weight_shape_bytes_sha256",
    "coverage_min", "coverage_max", "weight_reassembly_bit_exact",
    "coordinate_reassembly_bit_exact", "shape_global_identity_bit_exact",
)
RAW_SIMPLE_FIELDS = (
    "panel_id", "tiling_seed", "tiling_index", "depth", "role", "matrix_key",
    "d_in", "d_out", "num_tiles", *RAW_FEATURE_NAMES, "feature_tensor_sha256",
    "weight_shape_bytes_sha256", "tiling_invariant",
)


def _code_and_feature_rows(
    *,
    representation: str,
    entry: WOnlyEntry,
    tiling: TilingIndex,
    tiling_index: int,
    codes: torch.Tensor,
    template: Mapping[str, torch.Tensor],
    model_kind: str,
    model_seed: int | None,
    condition_kind: str,
    array_row_index: int,
) -> tuple[str, dict[str, Any], np.ndarray, dict[str, Any]]:
    key = canonical_code_key(representation, entry.panel_id, tiling.seed, entry.depth, entry.role)
    code_hash = tensor_sha256(codes, dtype=torch.float32)
    template_hashes = {
        name: tensor_sha256(template[name], dtype=torch.float32) for name in ("c_var", "c_patch")
    }
    code_row = {
        "representation": representation,
        "panel_id": entry.panel_id,
        "tiling_seed": tiling.seed,
        "tiling_index": tiling_index,
        "depth": entry.depth,
        "role": entry.role,
        "matrix_key": matrix_key(entry.depth, entry.role),
        "d_in": int(entry.W.shape[0]),
        "d_out": int(entry.W.shape[1]),
        "num_tiles": int(codes.shape[0]),
        "code_width": int(codes.shape[1]),
        "code_key": key,
        "code_tensor_sha256": code_hash,
        "partition_sha256": tiling.partition_sha256,
        "weight_shape_bytes_sha256": entry.weight_shape_bytes_sha256,
        "c_var_sha256": template_hashes["c_var"],
        "c_patch_sha256": template_hashes["c_patch"],
        "model_kind": model_kind,
        "model_seed": "" if model_seed is None else model_seed,
        "condition_kind": condition_kind,
        "activation_consumers": 0,
        "metadata_label_consumers": 0,
        "finite": True,
        "nonzero": bool(torch.count_nonzero(codes) > 0),
    }
    feature = aggregate_tile_features(codes)
    feature_hash = tensor_sha256(torch.from_numpy(feature), dtype=torch.float64)
    feature_row = {
        "representation": representation,
        "panel_id": entry.panel_id,
        "tiling_seed": tiling.seed,
        "tiling_index": tiling_index,
        "depth": entry.depth,
        "role": entry.role,
        "matrix_key": matrix_key(entry.depth, entry.role),
        "d_in": int(entry.W.shape[0]),
        "d_out": int(entry.W.shape[1]),
        "num_tiles": int(codes.shape[0]),
        "feature_dim": 2560,
        "array_name": representation,
        "array_row_index": array_row_index,
        "source_kind": "learned_checkpoint" if model_seed is None else "untrained_model",
        "source_seed": "" if model_seed is None else model_seed,
        "feature_tensor_sha256": feature_hash,
        "partition_sha256": tiling.partition_sha256,
        "tiling_invariant": False,
        "finite": True,
        "nonzero": bool(np.count_nonzero(feature) > 0),
    }
    return key, code_row, feature, feature_row


def _annotate_latest_runtime_call(
    *,
    representation: str,
    entry: WOnlyEntry,
    tiling: TilingIndex,
    condition_kind: str,
    call_class: str,
    outer_condition_selector_metadata: list[str],
    decision_eligible: bool,
) -> None:
    record = _ENCODE_RUNTIME["call_records"][-1]
    record.update(
        {
            "representation": representation,
            "panel_id_metadata_write_only": entry.panel_id,
            "role_metadata_write_only": entry.role,
            "depth_metadata_write_only": entry.depth,
            "tiling_seed_metadata_write_only": tiling.seed,
            "condition_kind": condition_kind,
            "call_class": call_class,
            "outer_condition_selector_metadata": outer_condition_selector_metadata,
            "decision_eligible": decision_eligible,
        }
    )


def encode_fixed_condition_representation(
    *,
    representation: str,
    model: torch.nn.Module,
    bundle: WOnlyBundle,
    fixed_template: Mapping[str, torch.Tensor],
    model_kind: str,
    model_seed: int | None,
    condition_kind: str,
    decision_eligible: bool,
    logger: logging.Logger,
    latent_codes: dict[str, torch.Tensor],
    code_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Primary path: role/depth never select or alter the fixed condition."""
    features: list[np.ndarray] = []
    cells = canonical_cells(bundle)
    for row_index, (entry, tiling, tiling_index) in enumerate(cells):
        logger.info(
            "stage=representation_encode representation=%s progress=%d/%d panel=%s seed=%d depth=%d role=%s",
            representation, row_index + 1, len(cells), entry.panel_id, tiling.seed, entry.depth, entry.role,
        )
        tiles = split_tiles(entry.W, tiling)
        codes = encode_weight_tiles(model, tiles, fixed_template)
        _annotate_latest_runtime_call(
            representation=representation,
            entry=entry,
            tiling=tiling,
            condition_kind=condition_kind,
            call_class="primary_fixed_condition",
            outer_condition_selector_metadata=[],
            decision_eligible=decision_eligible,
        )
        key, code_row, feature, feature_row = _code_and_feature_rows(
            representation=representation,
            entry=entry,
            tiling=tiling,
            tiling_index=tiling_index,
            codes=codes,
            template=fixed_template,
            model_kind=model_kind,
            model_seed=model_seed,
            condition_kind=condition_kind,
            array_row_index=row_index,
        )
        latent_codes[key] = codes
        code_rows.append(code_row)
        features.append(feature)
        aggregate_rows.append(feature_row)
        del tiles, codes
    result = np.stack(features).astype(np.float64, copy=False)
    if result.shape != (432, 2560):
        raise RuntimeError(f"fixed-condition aggregate array shape mismatch: {representation}/{result.shape}")
    return result


def encode_cell_reference_representation(
    *,
    model: torch.nn.Module,
    bundle: WOnlyBundle,
    templates: Mapping[str, Any],
    logger: logging.Logger,
    latent_codes: dict[str, torch.Tensor],
    code_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Segregated non-gating positive reference; role/depth select cell C."""
    representation = "learned_cell"
    features: list[np.ndarray] = []
    cells = canonical_cells(bundle)
    for row_index, (entry, tiling, tiling_index) in enumerate(cells):
        template = templates["cell_mean"][(entry.role, entry.depth)]
        logger.info(
            "stage=cell_reference_encode representation=%s progress=%d/%d panel=%s seed=%d depth=%d role=%s",
            representation, row_index + 1, len(cells), entry.panel_id, tiling.seed, entry.depth, entry.role,
        )
        tiles = split_tiles(entry.W, tiling)
        codes = encode_weight_tiles(model, tiles, template)
        _annotate_latest_runtime_call(
            representation=representation,
            entry=entry,
            tiling=tiling,
            condition_kind="source_cell_mean",
            call_class="segregated_non_gating_cell_reference",
            outer_condition_selector_metadata=["role", "depth"],
            decision_eligible=False,
        )
        key, code_row, feature, feature_row = _code_and_feature_rows(
            representation=representation,
            entry=entry,
            tiling=tiling,
            tiling_index=tiling_index,
            codes=codes,
            template=template,
            model_kind="learned_checkpoint",
            model_seed=None,
            condition_kind="source_cell_mean",
            array_row_index=row_index,
        )
        latent_codes[key] = codes
        code_rows.append(code_row)
        features.append(feature)
        aggregate_rows.append(feature_row)
        del tiles, codes
    result = np.stack(features).astype(np.float64, copy=False)
    if result.shape != (432, 2560):
        raise RuntimeError(f"cell-reference aggregate array shape mismatch: {result.shape}")
    return result


def known_source_numeric_preflight(
    *,
    model: torch.nn.Module,
    bundle: WOnlyBundle,
    templates: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_entry = next(
        entry for entry in bundle.entries
        if entry.panel_id == "source_vit_b_flickr" and entry.depth == 0 and entry.role == "attn_query"
    )
    tiles = split_tiles(source_entry.W, bundle.source_preflight_tiling)[:64].contiguous()
    template = templates["cell_mean"][("attn_query", 0)]
    codes = encode_weight_tiles(model, tiles, template)
    call_record = _ENCODE_RUNTIME["call_records"].pop()
    current_hash = tensor_sha256(codes, dtype=torch.float32)
    expected_hash = bundle.source_preflight_cached_slice_sha256
    bit_exact = current_hash == expected_hash
    result = {
        "schema_version": "global_context_known_source_numeric_preflight_v1",
        "pass": bit_exact,
        "performed_before_new_code_entries": True,
        "source_activation_passed_to_weight_ae": False,
        "source_activation_consumers": 0,
        "metadata_label_model_consumers": 0,
        "decoder_calls": 0,
        "distribution_encoder_calls": 0,
        "source_panel": "source_vit_b_flickr",
        "depth": 0,
        "role": "attn_query",
        "old_tiling_seed": 26081601,
        "old_partition_sha256": bundle.source_preflight_tiling.partition_sha256,
        "cached_code_key": bundle.source_preflight_code_key,
        "execution_batch": 64,
        "expected_cached_slice_sha256": expected_hash,
        "current_code_sha256": current_hash,
        "bit_exact": bit_exact,
        "max_abs": 0.0 if bit_exact else None,
        "relative_l2": 0.0 if bit_exact else None,
        "cosine": 1.0 if bit_exact else None,
        "comparison_method": "exact_shape_and_FP32_bytes_SHA256; equality implies exact zero difference",
    }
    if not bit_exact:
        raise RuntimeError(f"known-source numeric preflight hash mismatch: {result}")
    call_record.update(
        {
            "representation": "known_source_numeric_preflight",
            "panel_id_metadata_write_only": "source_vit_b_flickr",
            "role_metadata_write_only": "attn_query",
            "depth_metadata_write_only": 0,
            "tiling_seed_metadata_write_only": 26081601,
            "call_class": "preflight_not_a_new_code_entry",
            "condition_kind": "source_cell_mean",
            "outer_condition_selector_metadata": ["fixed_preflight_identity"],
            "decision_eligible": False,
        }
    )
    del tiles, codes
    return result, call_record


def build_raw_baselines(
    *,
    bundle: WOnlyBundle,
    device: torch.device,
    logger: logging.Logger,
    aggregate_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    cells = canonical_cells(bundle)
    raw_rows: list[dict[str, Any]] = []
    raw_features: list[np.ndarray] = []
    for row_index, (entry, tiling, tiling_index) in enumerate(cells):
        feature = raw_simple_descriptor(entry.W)
        feature_hash = tensor_sha256(torch.from_numpy(feature), dtype=torch.float64)
        raw_rows.append(
            {
                "panel_id": entry.panel_id,
                "tiling_seed": tiling.seed,
                "tiling_index": tiling_index,
                "depth": entry.depth,
                "role": entry.role,
                "matrix_key": matrix_key(entry.depth, entry.role),
                "d_in": int(entry.W.shape[0]),
                "d_out": int(entry.W.shape[1]),
                "num_tiles": tiling.num_tiles,
                **{name: float(feature[index]) for index, name in enumerate(RAW_FEATURE_NAMES)},
                "feature_tensor_sha256": feature_hash,
                "weight_shape_bytes_sha256": entry.weight_shape_bytes_sha256,
                "tiling_invariant": True,
            }
        )
        raw_features.append(feature)
        aggregate_rows.append(
            {
                "representation": "raw_simple",
                "panel_id": entry.panel_id,
                "tiling_seed": tiling.seed,
                "tiling_index": tiling_index,
                "depth": entry.depth,
                "role": entry.role,
                "matrix_key": matrix_key(entry.depth, entry.role),
                "d_in": int(entry.W.shape[0]),
                "d_out": int(entry.W.shape[1]),
                "num_tiles": tiling.num_tiles,
                "feature_dim": 37,
                "array_name": "raw_simple",
                "array_row_index": row_index,
                "source_kind": "raw_simple",
                "source_seed": "",
                "feature_tensor_sha256": feature_hash,
                "partition_sha256": tiling.partition_sha256,
                "tiling_invariant": True,
                "finite": True,
                "nonzero": bool(np.count_nonzero(feature) > 0),
            }
        )
    arrays: dict[str, np.ndarray] = {
        "raw_simple": np.stack(raw_features).astype(np.float64, copy=False)
    }
    maps, map_payload = build_countsketch_maps()
    sketch_features: dict[int, list[np.ndarray]] = {seed: [] for seed in COUNTSKETCH_SEEDS}
    for row_index, (entry, tiling, tiling_index) in enumerate(cells):
        logger.info(
            "stage=countsketch progress=%d/%d panel=%s seed=%d depth=%d role=%s",
            row_index + 1, len(cells), entry.panel_id, tiling.seed, entry.depth, entry.role,
        )
        tiles = split_tiles(entry.W, tiling)
        for sketch_seed in COUNTSKETCH_SEEDS:
            buckets, signs = maps[sketch_seed]
            sketches = apply_countsketch(tiles, buckets, signs, device)
            feature = aggregate_tile_features(sketches)
            representation = f"countsketch_{sketch_seed}"
            feature_hash = tensor_sha256(torch.from_numpy(feature), dtype=torch.float64)
            sketch_features[sketch_seed].append(feature)
            aggregate_rows.append(
                {
                    "representation": representation,
                    "panel_id": entry.panel_id,
                    "tiling_seed": tiling.seed,
                    "tiling_index": tiling_index,
                    "depth": entry.depth,
                    "role": entry.role,
                    "matrix_key": matrix_key(entry.depth, entry.role),
                    "d_in": int(entry.W.shape[0]),
                    "d_out": int(entry.W.shape[1]),
                    "num_tiles": tiling.num_tiles,
                    "feature_dim": 2560,
                    "array_name": representation,
                    "array_row_index": row_index,
                    "source_kind": "countsketch",
                    "source_seed": sketch_seed,
                    "feature_tensor_sha256": feature_hash,
                    "partition_sha256": tiling.partition_sha256,
                    "tiling_invariant": False,
                    "finite": True,
                    "nonzero": bool(np.count_nonzero(feature) > 0),
                }
            )
            del sketches
        del tiles
    for seed in COUNTSKETCH_SEEDS:
        arrays[f"countsketch_{seed}"] = np.stack(sketch_features[seed]).astype(np.float64, copy=False)

    return arrays, raw_rows, map_payload


def build_zero_weight_sanity(bundle: WOnlyBundle) -> dict[str, Any]:
    """Frozen 216-W algebraic validity check; this function has no model input."""
    entries: list[dict[str, Any]] = []
    lookup = {(entry.panel_id, entry.depth, entry.role): entry for entry in bundle.entries}
    for panel_id in PANEL_IDS:
        for depth in range(12):
            for role in ROLES:
                entry = lookup[(panel_id, depth, role)]
                W64 = entry.W.to(torch.float64)
                zero = torch.zeros_like(W64)
                denominator = float(torch.sum(W64.square(), dtype=torch.float64).item())
                numerator = float(torch.sum((zero - W64).square(), dtype=torch.float64).item())
                ratio = float(numerator / denominator)
                entries.append(
                    {
                        "panel_id": panel_id,
                        "depth": depth,
                        "role": role,
                        "d_in": int(entry.W.shape[0]),
                        "d_out": int(entry.W.shape[1]),
                        "weight_shape_bytes_sha256": entry.weight_shape_bytes_sha256,
                        "weight_tensor_sha256": entry.weight_tensor_sha256,
                        "numel": int(entry.W.numel()),
                        "finite_count": int(torch.isfinite(entry.W).sum().item()),
                        "nonzero_count": int(torch.count_nonzero(entry.W).item()),
                        "sum_squared": denominator,
                        "frobenius_norm": float(math.sqrt(denominator)),
                        "rms": float(math.sqrt(denominator / float(entry.W.numel()))),
                        "literal_zero_relative_squared_error": ratio,
                    }
                )
    all_finite = all(
        row["finite_count"] == row["numel"]
        and math.isfinite(row["sum_squared"])
        and math.isfinite(row["frobenius_norm"])
        and math.isfinite(row["rms"])
        and math.isfinite(row["literal_zero_relative_squared_error"])
        for row in entries
    )
    all_nonzero = all(row["nonzero_count"] > 0 for row in entries)
    all_positive = all(row["sum_squared"] > 0.0 for row in entries)
    all_one = all(row["literal_zero_relative_squared_error"] == 1.0 for row in entries)
    passed = bool(len(entries) == 216 and all_finite and all_nonzero and all_positive and all_one)
    if not passed:
        raise RuntimeError("frozen W-only zero-baseline sanity failed")
    return {
        "schema_version": "global_context_zero_weight_sanity_v1",
        "matrix_count": 216,
        "model_forward_calls": 0,
        "entries": entries,
        "summary": {
            "all_finite": all_finite,
            "all_have_nonzero_entry": all_nonzero,
            "all_positive_sum_squared": all_positive,
            "all_literal_zero_ratios_exactly_one": all_one,
            "pass": passed,
        },
    }


def tiling_indices_payload(bundle: WOnlyBundle) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for entry, tiling, tiling_index in canonical_cells(bundle):
        key = canonical_tiling_key(entry.panel_id, tiling.seed, entry.depth, entry.role)
        entries[key] = {
            "panel_id": entry.panel_id,
            "tiling_seed": tiling.seed,
            "tiling_index": tiling_index,
            "depth": entry.depth,
            "role": entry.role,
            "matrix_key": matrix_key(entry.depth, entry.role),
            "d_in": int(entry.W.shape[0]),
            "d_out": int(entry.W.shape[1]),
            "rows": tiling.rows.clone(),
            "cols": tiling.cols.clone(),
            "row_index_sha256": tiling.row_sha256,
            "column_index_sha256": tiling.column_sha256,
            "partition_sha256": tiling.partition_sha256,
            "weight_shape_bytes_sha256": entry.weight_shape_bytes_sha256,
        }
    return {
        "schema_version": "global_context_tiling_indices_v1",
        "entry_count": len(entries),
        "seeds": list(COMMON_TILING_SEEDS),
        "panel_order": list(PANEL_IDS),
        "entries": entries,
    }


def observe_latent_code_width(latent_codes: Mapping[str, torch.Tensor]) -> int | list[int]:
    widths = {int(value.shape[1]) for value in latent_codes.values()}
    return next(iter(widths)) if len(widths) == 1 else sorted(widths)


def verify_raw_grids(
    *,
    bundle: WOnlyBundle,
    latent_codes: Mapping[str, torch.Tensor],
    code_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    arrays: Mapping[str, np.ndarray],
    countsketch_payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected = expected_counts()
    total_code_rows = sum(int(value.shape[0]) for value in latent_codes.values())
    observed = {
        "tiling_manifest_rows": len(bundle.tiling_rows),
        "tiling_index_entries": len(tiling_indices_payload(bundle)["entries"]),
        "code_manifest_rows": len(code_rows),
        "latent_code_entries": len(latent_codes),
        "latent_code_rows": total_code_rows,
        "latent_code_width": observe_latent_code_width(latent_codes),
        "raw_simple_rows": len(raw_rows),
        "raw_simple_feature_columns": len(RAW_FEATURE_NAMES),
        "countsketch_maps": len(countsketch_payload["seeds"]),
        "countsketch_pairs": int(countsketch_payload["total_pair_count"]),
        "aggregate_manifest_rows": len(aggregate_rows),
        "aggregate_arrays": len(arrays),
        "aggregate_2560_arrays": sum(array.shape == (432, 2560) for array in arrays.values()),
        "aggregate_37_arrays": sum(array.shape == (432, 37) for array in arrays.values()),
    }
    compare_keys = (
        "tiling_manifest_rows", "tiling_index_entries", "code_manifest_rows",
        "latent_code_entries", "latent_code_rows", "latent_code_width", "raw_simple_rows",
        "raw_simple_feature_columns", "countsketch_maps", "countsketch_pairs",
        "aggregate_manifest_rows", "aggregate_arrays", "aggregate_2560_arrays",
        "aggregate_37_arrays",
    )
    mismatch = {
        key: {"observed": observed[key], "expected": expected[key]}
        for key in compare_keys if observed[key] != expected[key]
    }
    code_identities = {
        tuple(row[name] for name in ("representation", "panel_id", "tiling_seed", "depth", "role"))
        for row in code_rows
    }
    aggregate_identities = {
        tuple(row[name] for name in ("representation", "panel_id", "tiling_seed", "depth", "role"))
        for row in aggregate_rows
    }
    if len(code_identities) != len(code_rows) or len(aggregate_identities) != len(aggregate_rows):
        mismatch["unique_grids"] = {
            "code_unique": len(code_identities),
            "aggregate_unique": len(aggregate_identities),
        }
    canonical_cell_identities = [
        (panel_id, seed, depth, role)
        for panel_id in PANEL_IDS
        for seed in COMMON_TILING_SEEDS
        for depth in range(12)
        for role in ROLES
    ]
    expected_code_order = [
        (representation, *identity)
        for representation in CODE_REPRESENTATIONS
        for identity in canonical_cell_identities
    ]
    observed_code_order = [
        (row["representation"], row["panel_id"], int(row["tiling_seed"]), int(row["depth"]), row["role"])
        for row in code_rows
    ]
    expected_aggregate_order = [
        (representation, *identity)
        for representation in REPRESENTATIONS
        for identity in canonical_cell_identities
    ]
    observed_aggregate_order = [
        (row["representation"], row["panel_id"], int(row["tiling_seed"]), int(row["depth"]), row["role"])
        for row in aggregate_rows
    ]
    if observed_code_order != expected_code_order or observed_aggregate_order != expected_aggregate_order:
        mismatch["canonical_row_order"] = {
            "code_order_exact": observed_code_order == expected_code_order,
            "aggregate_order_exact": observed_aggregate_order == expected_aggregate_order,
        }
    if set(arrays) != set(REPRESENTATIONS):
        mismatch["array_names"] = {"observed": sorted(arrays), "expected": list(REPRESENTATIONS)}
    shape_groups: dict[tuple[int, int, int], set[str]] = {}
    for row in bundle.tiling_rows:
        shape_groups.setdefault(
            (int(row["tiling_seed"]), int(row["d_in"]), int(row["d_out"])), set()
        ).add(str(row["partition_sha256"]))
    if any(len(values) != 1 for values in shape_groups.values()) or len(shape_groups) != 6:
        mismatch["shape_only_tiling_identity"] = {
            str(key): sorted(values) for key, values in shape_groups.items()
        }
    if mismatch:
        raise RuntimeError(f"raw output grid mismatch: {mismatch}")
    return {
        "expected": expected,
        "observed": observed,
        "code_grid_unique": True,
        "aggregate_grid_unique": True,
        "shape_only_tiling_groups": {
            f"seed={seed}|shape={d_in}x{d_out}": next(iter(values))
            for (seed, d_in, d_out), values in sorted(shape_groups.items())
        },
        "pass": True,
    }


def artifact_manifest(
    output: Path,
    binding: Mapping[str, Any],
    *,
    allow_transaction_marker: bool = False,
) -> dict[str, Any]:
    actual_paths: list[Path] = []
    for path in output.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"formal raw output contains symlink: {path}")
        relative = path.relative_to(output).as_posix()
        if (
            path.is_file()
            and path.name != "artifact_manifest.json"
            and not (allow_transaction_marker and relative == TRANSACTION_MARKER_NAME)
        ):
            actual_paths.append(path)
    marker_path = output / TRANSACTION_MARKER_NAME
    if allow_transaction_marker and (not marker_path.is_file() or marker_path.is_symlink()):
        raise RuntimeError("private staging transaction marker absent before terminal validation")
    actual_names = {path.relative_to(output).as_posix() for path in actual_paths}
    if actual_names != set(EXPECTED_RAW_FILES):
        raise RuntimeError(
            f"formal raw output membership mismatch: missing={sorted(set(EXPECTED_RAW_FILES)-actual_names)} "
            f"extra={sorted(actual_names-set(EXPECTED_RAW_FILES))}"
        )
    rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(actual_paths, key=lambda item: item.relative_to(output).as_posix())
    ]
    return {
        "schema_version": "global_context_latent_geometry_raw_artifact_manifest_v1",
        "frozen_design_sha256": DESIGN_SHA256,
        "runner_sha256": binding["runner_sha256"],
        "analyzer_sha256": binding["analyzer_sha256"],
        "external_contract_sha256": binding["external_contract_sha256"],
        "self_excluded": True,
        "file_count": len(rows),
        "files": rows,
    }


def _execute_staged_body(
    args: argparse.Namespace,
    *,
    contract: Mapping[str, Any],
    contract_sha: str,
    final_output: Path,
    output: Path,
) -> dict[str, Any]:
    """Populate and terminally validate a private staging tree."""
    logger = setup_logging(output)
    started = time.monotonic()
    logger.info(
        "stage=start mode=EXECUTE design_sha256=%s runner_sha256=%s analyzer_sha256=%s",
        DESIGN_SHA256,
        contract["runner_sha256"],
        contract["analyzer_sha256"],
    )
    logger.info(
        "config=%s device=%s dtype=FP32/BF16/FP64 seed=%d cache_mode=immutable_revalidated "
        "final_output=%s private_staging=%s",
        json.dumps(contract["execution_config"], sort_keys=True),
        args.device,
        GLOBAL_SEED,
        final_output,
        output,
    )
    logger.info(
        "panels=%s roles=%s tilings=%s code_representations=%s countsketch_seeds=%s",
        PANEL_IDS,
        ROLES,
        COMMON_TILING_SEEDS,
        CODE_REPRESENTATIONS,
        COUNTSKETCH_SEEDS,
    )

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    device = torch.device(args.device)
    if device != torch.device("cuda:0") or not torch.cuda.is_available():
        raise RuntimeError(f"formal CUDA:0 device unavailable/mismatched: {device}")
    startup_resources = live_resource_snapshot(device)
    logger.info(
        "stage=resource_preflight disk_free_bytes=%d host_memory_available_bytes=%s "
        "cuda_free_bytes=%s cuda_total_bytes=%s cuda_allocated_bytes=%s cuda_reserved_bytes=%s",
        int(startup_resources["disk"]["free_bytes"]),
        startup_resources["host_memory_available_bytes"],
        startup_resources["cuda"]["free_bytes"],
        startup_resources["cuda"]["total_bytes"],
        startup_resources["cuda"]["allocated_bytes"],
        startup_resources["cuda"]["reserved_bytes"],
    )

    logger.info("stage=contract_copy external=%s", args.contract_path.resolve(strict=True))
    atomic_copy(args.contract_path.resolve(strict=True), output / "preexecution_contract.json")
    if sha256_file(output / "preexecution_contract.json") != contract_sha:
        raise RuntimeError("copied external contract is not byte exact")
    binding = {
        "schema_version": "global_context_preexecution_binding_v1",
        "frozen_design_path": str(DESIGN_PATH.resolve(strict=True)),
        "frozen_design_sha256": DESIGN_SHA256,
        "runner_path": contract["scripts"]["runner"]["path"],
        "runner_sha256": contract["runner_sha256"],
        "analyzer_path": contract["scripts"]["analyzer"]["path"],
        "analyzer_sha256": contract["analyzer_sha256"],
        "external_contract_path": str(args.contract_path.resolve(strict=True)),
        "external_contract_sha256": contract_sha,
        "copied_contract_path": str(final_output / "preexecution_contract.json"),
        "copied_contract_sha256": contract_sha,
        "runner_output_path": str(final_output),
        "analyzer_output_path": contract["execution_config"]["analyzer_output_dir"],
        "exact_contract_verified_before_model_import": True,
    }
    atomic_write_json(output / "preexecution_binding.json", binding)
    resolved_config = {
        "schema_version": "global_context_resolved_config_v1",
        "created_utc": utc_now(),
        "frozen_design_sha256": DESIGN_SHA256,
        "runner_sha256": contract["runner_sha256"],
        "analyzer_sha256": contract["analyzer_sha256"],
        "external_contract_sha256": contract_sha,
        "runtime": contract["runtime"],
        "execution_config": contract["execution_config"],
        "expected_counts": contract["expected_counts"],
    }
    atomic_write_json(output / "resolved_config.json", resolved_config)

    logger.info("stage=input_materialization source_only=true model_modules_loaded=false")
    raw_templates = torch.load(SOURCE_TEMPLATE, map_location="cpu", weights_only=True)
    templates, template_audit = validate_templates(raw_templates)
    del raw_templates
    bundle = materialize_w_only(
        contract["input_audit"]["source_parent_audit"],
        contract["input_audit"]["old_reference_audit"],
    )
    if bundle.materialization_audit != contract["input_audit"]["w_only_materialization"]:
        raise RuntimeError("execute-time W-only materialization differs from external contract")
    if template_audit != contract["input_audit"]["source_template_audit"]:
        raise RuntimeError("execute-time source template audit differs from external contract")
    input_audit = {
        **contract["input_audit"],
        "execute_time_w_only_exact_contract_match": True,
        "execute_time_template_exact_contract_match": True,
        "model_modules_loaded_at_materialization_completion": [],
    }
    atomic_write_json(output / "input_audit.json", input_audit)
    atomic_write_json(output / "source_template_audit.json", template_audit)
    logger.info(
        "stage=w_only_ready matrices=%d tilings=%d activation_tensors_retained=0 garbage_collection=true",
        len(bundle.entries),
        len(bundle.tiling_rows),
    )

    logger.info("stage=zero_weight_sanity model_forward_calls=0")
    zero_sanity = build_zero_weight_sanity(bundle)
    if zero_sanity != contract["input_audit"]["zero_weight_sanity_preflight"]:
        raise RuntimeError("execute-time zero-weight sanity differs from contracted preflight")
    atomic_write_json(output / "zero_weight_sanity.json", zero_sanity)
    if zero_sanity["summary"]["pass"] is not True or zero_sanity["model_forward_calls"] != 0:
        raise RuntimeError("zero-weight sanity did not pass before model construction")

    logger.info("stage=tiling_output shape_only_identity=v2 rows=%d", len(bundle.tiling_rows))
    atomic_write_csv(
        output / "tiling_manifest.csv",
        bundle.tiling_rows,
        TILING_MANIFEST_FIELDS,
    )
    tiling_payload = tiling_indices_payload(bundle)
    atomic_torch_save(output / "tiling_indices.pt", tiling_payload)
    if int(tiling_payload["entry_count"]) != 432:
        raise RuntimeError("tiling index payload count mismatch")

    # This is the first import of Weight-AE model code and occurs only after
    # activation destruction, W-only construction, GC, and zero-W sanity.
    logger.info("stage=weight_ae_project_import after_w_only=true dependencies=%d", len(contract["model_dependencies"]))
    gate, factorial, local_import_audit = import_model_helpers(contract)
    logger.info("stage=weight_ae_model_build kind=learned checkpoint=%s", AE_CHECKPOINT)
    learned_model, amp_enabled, amp_dtype, learned_contract = build_learned_model(factorial, device)
    learned_resolved_model_config = copy.deepcopy(learned_model.cfg)
    learned_fingerprint = learned_contract["fingerprint"]
    model_contract: dict[str, Any] = {
        "schema_version": "global_context_model_contract_v1",
        "learned": learned_contract,
        "untrained": {},
        "local_module_closure_after_import": local_import_audit,
        "global_seed": GLOBAL_SEED,
        "untrained_seed_order": list(UNTRAINED_SEEDS),
        "decoder_calls": 0,
        "distribution_encoder_calls": 0,
    }
    logger.info(
        "stage=weight_ae_ready kind=learned amp_enabled=%s amp_dtype=%s rope=raw latent_sampling=false",
        amp_enabled,
        amp_dtype,
    )

    runtime_call_records: list[dict[str, Any]] = []
    _ENCODE_RUNTIME.clear()
    _ENCODE_RUNTIME.update(
        {
            "factorial": factorial,
            "gate": gate,
            "device": device,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "batch_size": int(args.batch_size),
            "log_every_batches": int(args.log_every_batches),
            "logger": logger,
            "call_records": runtime_call_records,
            "call_ordinal": 1,
        }
    )
    assert_model_fully_eval(learned_model, label="learned checkpoint immediately before numeric preflight")
    logger.info("stage=known_source_numeric_preflight")
    known_preflight, preflight_call = known_source_numeric_preflight(
        model=learned_model,
        bundle=bundle,
        templates=templates,
    )
    atomic_write_json(output / "known_source_numeric_preflight.json", known_preflight)
    logger.info(
        "stage=known_source_numeric_preflight_complete pass=true bit_exact=true current_sha256=%s",
        known_preflight["current_code_sha256"],
    )

    latent_codes: dict[str, torch.Tensor] = {}
    code_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    logger.info("stage=learned_cell_reference condition=source_cell_mean decision_eligible=false")
    arrays["learned_cell"] = encode_cell_reference_representation(
        model=learned_model,
        bundle=bundle,
        templates=templates,
        logger=logger,
        latent_codes=latent_codes,
        code_rows=code_rows,
        aggregate_rows=aggregate_rows,
    )
    logger.info("stage=learned_global condition=source_global primary_dataflow=true")
    arrays["learned_global"] = encode_fixed_condition_representation(
        representation="learned_global",
        model=learned_model,
        bundle=bundle,
        fixed_template=templates["global"],
        model_kind="learned_checkpoint",
        model_seed=None,
        condition_kind="source_global",
        decision_eligible=True,
        logger=logger,
        latent_codes=latent_codes,
        code_rows=code_rows,
        aggregate_rows=aggregate_rows,
    )
    zero_template = {
        "c_var": torch.zeros(256, dtype=torch.float32),
        "c_patch": torch.zeros(256, dtype=torch.float32),
    }
    logger.info("stage=learned_zero condition=exact_zero primary_dataflow=true secondary_non_gating=true")
    arrays["learned_zero"] = encode_fixed_condition_representation(
        representation="learned_zero",
        model=learned_model,
        bundle=bundle,
        fixed_template=zero_template,
        model_kind="learned_checkpoint",
        model_seed=None,
        condition_kind="exact_zero",
        decision_eligible=False,
        logger=logger,
        latent_codes=latent_codes,
        code_rows=code_rows,
        aggregate_rows=aggregate_rows,
    )
    del learned_model
    torch.cuda.empty_cache()
    gc.collect()

    untrained_state_hashes: set[str] = set()
    for seed in UNTRAINED_SEEDS:
        representation = f"untrained_global_{seed}"
        logger.info("stage=weight_ae_model_build kind=untrained seed=%d checkpoint_tensor_copy_count=0", seed)
        untrained_model, initialization = build_untrained_model(
            gate,
            seed,
            device,
            learned_fingerprint,
            learned_resolved_model_config,
        )
        state_hash = initialization["fingerprint"]["state_dict_sha256"]
        if state_hash in untrained_state_hashes:
            raise RuntimeError(f"duplicate untrained state fingerprint: seed={seed}")
        untrained_state_hashes.add(state_hash)
        model_contract["untrained"][str(seed)] = initialization
        arrays[representation] = encode_fixed_condition_representation(
            representation=representation,
            model=untrained_model,
            bundle=bundle,
            fixed_template=templates["global"],
            model_kind="untrained_model",
            model_seed=seed,
            condition_kind="source_global",
            decision_eligible=True,
            logger=logger,
            latent_codes=latent_codes,
            code_rows=code_rows,
            aggregate_rows=aggregate_rows,
        )
        del untrained_model
        torch.cuda.empty_cache()
        gc.collect()
    if len(untrained_state_hashes) != 3:
        raise RuntimeError("untrained state fingerprints are not three distinct values")
    model_contract["three_untrained_state_dict_hashes_distinct"] = True
    model_contract["all_untrained_parameter_grids_match_learned"] = True
    model_contract["all_untrained_checkpoint_tensor_copy_counts_zero"] = True
    model_contract["pass"] = True
    atomic_write_json(output / "model_contract.json", model_contract)

    logger.info("stage=raw_baselines raw_simple=true countsketch_seeds=%s", COUNTSKETCH_SEEDS)
    raw_arrays, raw_rows, countsketch_payload = build_raw_baselines(
        bundle=bundle,
        device=device,
        logger=logger,
        aggregate_rows=aggregate_rows,
    )
    arrays.update(raw_arrays)
    representation_rank = {name: index for index, name in enumerate(REPRESENTATIONS)}
    aggregate_rows.sort(
        key=lambda row: (representation_rank[str(row["representation"])], int(row["array_row_index"]))
    )
    atomic_write_json(output / "countsketch_maps.json", countsketch_payload)
    atomic_write_csv(output / "raw_simple_features.csv", raw_rows, RAW_SIMPLE_FIELDS)

    logger.info("stage=raw_grid_validation")
    grid_audit = verify_raw_grids(
        bundle=bundle,
        latent_codes=latent_codes,
        code_rows=code_rows,
        raw_rows=raw_rows,
        aggregate_rows=aggregate_rows,
        arrays=arrays,
        countsketch_payload=countsketch_payload,
    )
    atomic_write_csv(output / "code_manifest.csv", code_rows, CODE_MANIFEST_FIELDS)
    atomic_torch_save(
        output / "latent_codes.pt",
        {
            "schema_version": "global_context_latent_codes_v1",
            "representation_order": list(CODE_REPRESENTATIONS),
            "entry_count": len(latent_codes),
            "total_code_rows": sum(int(value.shape[0]) for value in latent_codes.values()),
            "code_width": 512,
            "codes": latent_codes,
        },
    )
    atomic_write_csv(
        output / "aggregate_feature_manifest.csv",
        aggregate_rows,
        AGGREGATE_MANIFEST_FIELDS,
    )
    ordered_arrays = {name: arrays[name].astype(np.float64, copy=False) for name in REPRESENTATIONS}
    atomic_npz_save(output / "aggregate_features.npz", ordered_arrays)

    logger.info("stage=dataflow_audit runtime_new_code_calls=%d", len(runtime_call_records))
    if len(runtime_call_records) != 2592:
        raise RuntimeError(f"new-code runtime call count mismatch: {len(runtime_call_records)}")
    global_calls = [
        row for row in runtime_call_records
        if row["condition_kind"] == "source_global" and row["call_class"] == "primary_fixed_condition"
    ]
    zero_calls = [row for row in runtime_call_records if row["condition_kind"] == "exact_zero"]
    cell_calls = [row for row in runtime_call_records if row["condition_kind"] == "source_cell_mean"]
    global_var_hashes = {row["template_hashes"]["c_var"] for row in global_calls}
    global_patch_hashes = {row["template_hashes"]["c_patch"] for row in global_calls}
    zero_hash = tensor_sha256(torch.zeros(256, dtype=torch.float32), dtype=torch.float32)
    if (
        len(global_calls) != 1728
        or global_var_hashes != {GLOBAL_C_HASHES["c_var"]}
        or global_patch_hashes != {GLOBAL_C_HASHES["c_patch"]}
        or len(zero_calls) != 432
        or {row["template_hashes"]["c_var"] for row in zero_calls} != {zero_hash}
        or {row["template_hashes"]["c_patch"] for row in zero_calls} != {zero_hash}
        or len(cell_calls) != 432
        or any(row["outer_condition_selector_metadata"] for row in global_calls + zero_calls)
        or any(row["decision_eligible"] is not True for row in global_calls)
        or any(row["decision_eligible"] is not False for row in zero_calls + cell_calls)
        or any(
            row["activation_consumers"] != 0
            or row["metadata_label_consumers"] != 0
            or row["distribution_encoder_calls"] != 0
            or row["decoder_calls"] != 0
            for row in runtime_call_records
        )
    ):
        raise RuntimeError("runtime dataflow constancy/segregation audit failed")
    forbidden_runtime_keys = {
        "panel", "panel_id", "role", "depth", "dataset", "activation", "activations",
        "label", "labels", "target", "targets",
    }
    if forbidden_runtime_keys.intersection(_ENCODE_RUNTIME):
        raise RuntimeError(f"scientific metadata escaped into encoder runtime: {sorted(_ENCODE_RUNTIME)}")
    dataflow_audit = {
        "schema_version": "global_context_dataflow_audit_v1",
        "static_call_graph": contract["static_call_graph_audit"],
        "local_module_closure": assert_loaded_local_modules_frozen(contract),
        "w_only_lifecycle": {
            "materialization_completed_before_model_import": True,
            "activation_tensors_retained_at_model_import": 0,
            "activation_references_retained_at_model_import": 0,
            "exact_allowed_w_only_fields": bundle.materialization_audit["w_only_entry_fields"],
            "allowed_metadata_fields": ["panel_id", "depth", "role"],
            "activation_or_sample_target_payload_references_after_materialization": 0,
            "encode_runtime_exact_keys": sorted(_ENCODE_RUNTIME),
            "encode_runtime_contains_panel_role_depth_dataset_activation_or_label": False,
        },
        "known_source_preflight_call": preflight_call,
        "runtime_calls": runtime_call_records,
        "total_runtime_call_count": len(runtime_call_records),
        "primary_global_summary": {
            "representations": [
                "learned_global", *[f"untrained_global_{seed}" for seed in UNTRAINED_SEEDS]
            ],
            "runtime_call_count": len(global_calls),
            "unique_c_var_hashes": sorted(global_var_hashes),
            "unique_c_patch_hashes": sorted(global_patch_hashes),
            "activation_consumers": 0,
            "metadata_label_consumers": 0,
            "condition_selector_metadata_fields": [],
            "decision_eligible": True,
        },
        "zero_summary": {
            "representation": "learned_zero",
            "runtime_call_count": len(zero_calls),
            "unique_c_var_hashes": [zero_hash],
            "unique_c_patch_hashes": [zero_hash],
            "activation_consumers": 0,
            "metadata_label_consumers": 0,
            "condition_selector_metadata_fields": [],
            "secondary_non_gating": True,
        },
        "learned_cell_summary": {
            "representation": "learned_cell",
            "runtime_call_count": len(cell_calls),
            "segregated_call_record_namespace": True,
            "condition_selection_source": "immutable_source_cell_mean_by_role_depth",
            "allowed_selector_metadata_fields": ["role", "depth"],
            "primary_dataflow_claim_eligible": False,
            "level_a_or_b_decision_eligible": False,
            "activation_consumers": 0,
            "metadata_label_consumers": 0,
        },
        "decoder_calls": 0,
        "distribution_encoder_calls": 0,
        "panel_activation_consumers": 0,
        "old_source_or_candidate_cache_feature_consumers": 0,
        "old_source_or_candidate_cache_use": "known-source hash preflight and provenance audit only",
        "pass": True,
    }
    atomic_write_json(output / "dataflow_audit.json", dataflow_audit)

    logger.info("stage=input_immutability_recheck")
    final_resources = file_resources()
    final_scripts = script_snapshot(args)
    final_dependencies = dependency_snapshot()
    in_memory_w_rows = [
        {
            "panel_id": entry.panel_id,
            "depth": entry.depth,
            "role": entry.role,
            "shape": list(entry.W.shape),
            "weight_shape_bytes_sha256": weight_shape_bytes_sha256(entry.W),
            "weight_tensor_sha256": tensor_sha256(entry.W, dtype=torch.float32),
        }
        for entry in bundle.entries
    ]
    final_w_grid_sha = hashlib.sha256(
        json.dumps(in_memory_w_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    final_template_hashes = {
        name: tensor_sha256(templates["global"][name], dtype=torch.float32)
        for name in ("c_var", "c_patch")
    }
    immutability = {
        "schema_version": "global_context_input_immutability_recheck_v1",
        "resource_snapshot_exact": final_resources == contract["input_audit"]["resources"],
        "script_snapshot_exact": final_scripts == contract["scripts"],
        "dependency_snapshot_exact": final_dependencies == contract["model_dependencies"],
        "in_memory_w_grid_sha256": final_w_grid_sha,
        "expected_w_grid_sha256": contract["input_audit"]["w_only_materialization"]["w_grid_sha256"],
        "in_memory_w_grid_exact": final_w_grid_sha
        == contract["input_audit"]["w_only_materialization"]["w_grid_sha256"],
        "global_template_hashes": final_template_hashes,
        "global_template_hashes_exact": final_template_hashes == GLOBAL_C_HASHES,
        "source_only_seal_clean": True,
        "pass": True,
    }
    if not all(
        immutability[name] is True
        for name in (
            "resource_snapshot_exact", "script_snapshot_exact", "dependency_snapshot_exact",
            "in_memory_w_grid_exact", "global_template_hashes_exact",
        )
    ):
        raise RuntimeError(f"input immutability recheck failed: {immutability}")
    assert_seal_clean()
    atomic_write_json(output / "input_immutability_recheck.json", immutability)

    logger.info("stage=runner_metadata")
    runner_metadata = {
        "schema_version": "global_context_runner_metadata_v1",
        "status": "COMPLETE_RAW_ONLY",
        "frozen_design_path": str(DESIGN_PATH.resolve(strict=True)),
        "frozen_design_sha256": DESIGN_SHA256,
        "runner_path": contract["scripts"]["runner"]["path"],
        "runner_sha256": contract["runner_sha256"],
        "analyzer_path": contract["scripts"]["analyzer"]["path"],
        "analyzer_sha256": contract["analyzer_sha256"],
        "external_contract_path": str(args.contract_path.resolve(strict=True)),
        "external_contract_sha256": contract_sha,
        "runner_output_path": str(final_output),
        "analyzer_output_path": contract["execution_config"]["analyzer_output_dir"],
        "runtime": contract["runtime"],
        "expected_counts": contract["expected_counts"],
        "observed_counts": grid_audit["observed"],
        "grid_audit": grid_audit,
        "artifact_paths": {name: str(final_output / name) for name in EXPECTED_RAW_FILES},
        "startup_resources": startup_resources,
        "representation_order": list(REPRESENTATIONS),
        "panel_order": list(PANEL_IDS),
        "role_order": list(ROLES),
        "common_tiling_seeds": list(COMMON_TILING_SEEDS),
        "untrained_seeds": list(UNTRAINED_SEEDS),
        "countsketch_seeds": list(COUNTSKETCH_SEEDS),
        "new_code_model_call_records": len(runtime_call_records),
        "known_source_preflight_model_call_records": 1,
        "decoder_calls": 0,
        "distribution_encoder_calls": 0,
        "target_access_events": int(_SEAL_STATE["target_access_events"]),
        "network_connections": int(_SEAL_STATE["network_connections"]),
        "subprocess_launches": int(_SEAL_STATE["subprocess_launches"]),
        "elapsed_seconds_before_terminal_manifest": time.monotonic() - started,
    }
    atomic_write_json(output / "runner_metadata.json", runner_metadata)
    assert_seal_clean()
    seal_payload = {
        **dict(_SEAL_STATE),
        "weight_ae_model_modules_loaded_after_w_only_only": True,
        "forbidden_loaded_modules": sorted(
            name for name in sys.modules if "data2vec" in name.casefold()
        ),
        "pass": True,
    }
    atomic_write_json(output / "target_access_seal.json", seal_payload)

    logger.info(
        "stage=complete raw_only=true code_entries=%d code_rows=%d aggregate_rows=%d arrays=%d "
        "target_events=0 network_events=0 subprocess_events=0 elapsed=%.1fs",
        len(latent_codes),
        sum(int(value.shape[0]) for value in latent_codes.values()),
        len(aggregate_rows),
        len(arrays),
        time.monotonic() - started,
    )
    logger.info(
        "stage=transaction_commit_prepare final_output=%s private_staging=%s "
        "owner_marker_retained_until_terminal_validation=true",
        final_output,
        output,
    )
    logger.info("stage=artifact_manifest pending_after_log_close=true expected_files=%d", len(EXPECTED_RAW_FILES))
    close_logging(logger)
    manifest = artifact_manifest(output, binding, allow_transaction_marker=True)
    if int(manifest["file_count"]) != 21:
        raise RuntimeError(f"terminal manifest must declare 21 files: {manifest['file_count']}")
    atomic_write_json(output / "artifact_manifest.json", manifest)
    _verify_manifest_directory(output, allow_transaction_marker=True)
    return manifest


def _verify_manifest_directory(root: Path, *, allow_transaction_marker: bool = False) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("manifest root missing/symlinked")
    path = root / "artifact_manifest.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("manifest file missing/symlinked")
    payload = read_json(path)
    expected_manifest_keys = {
        "schema_version", "frozen_design_sha256", "runner_sha256", "analyzer_sha256",
        "external_contract_sha256", "self_excluded", "file_count", "files",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_manifest_keys
        or payload.get("schema_version")
        != "global_context_latent_geometry_raw_artifact_manifest_v1"
        or payload.get("frozen_design_sha256") != DESIGN_SHA256
        or payload.get("self_excluded") is not True
        or int(payload.get("file_count", -1)) != 21
        or not isinstance(payload.get("files"), list)
        or len(payload["files"]) != 21
    ):
        raise RuntimeError("manifest fixture header failed")
    declared: set[str] = set()
    for row in payload.get("files", []):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise RuntimeError("manifest fixture row schema failed")
        relative = str(row["path"])
        relative_path = Path(relative)
        if (
            relative in declared
            or relative == "artifact_manifest.json"
            or relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or len(relative_path.parts) != 1
            or ".." in relative_path.parts
        ):
            raise RuntimeError("manifest fixture duplicate/recursive path")
        declared.add(relative)
        member = root / relative
        if member.is_symlink() or not member.is_file():
            raise RuntimeError("manifest fixture member missing/symlinked")
        if member.stat().st_size != int(row["bytes"]) or sha256_file(member) != row["sha256"]:
            raise RuntimeError("manifest fixture hash/size mismatch")
    if declared != set(EXPECTED_RAW_FILES):
        raise RuntimeError("manifest fixture frozen file grid mismatch")
    if any(member.is_symlink() for member in root.rglob("*")):
        raise RuntimeError("manifest fixture tree contains symlink")
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if (
            member.is_file()
            and member.name != "artifact_manifest.json"
            and not (
                allow_transaction_marker
                and member.relative_to(root).as_posix() == TRANSACTION_MARKER_NAME
            )
        )
    }
    if declared != actual:
        raise RuntimeError("manifest fixture completeness mismatch")
    top_level = {member.name for member in root.iterdir()}
    expected_top_level = set(EXPECTED_RAW_FILES) | {"artifact_manifest.json"}
    if allow_transaction_marker:
        marker = root / TRANSACTION_MARKER_NAME
        if not marker.is_file() or marker.is_symlink():
            raise RuntimeError("transaction marker absent/symlinked during private validation")
        expected_top_level.add(TRANSACTION_MARKER_NAME)
    if top_level != expected_top_level:
        raise RuntimeError("terminal output is not the exact 21+1 top-level file grid")


def execute(args: argparse.Namespace) -> None:
    assert_canonical_formal_entrypoint_and_argv(args)
    assert_exact_formal_paths(args)
    print("stage=preexecution_contract_verification weight_ae_model_imported=false", flush=True)
    contract, contract_sha = verify_contract(args)
    final_output = Path(contract["execution_config"]["output_dir"]).resolve(strict=False)
    if final_output.exists() or final_output.is_symlink():
        raise RuntimeError(f"formal runner output must remain fresh: {final_output}")
    staging = create_process_owned_staging(
        final_output,
        runner_sha256=str(contract["runner_sha256"]),
        analyzer_sha256=str(contract["analyzer_sha256"]),
        external_contract_sha256=contract_sha,
    )
    try:
        manifest = _execute_staged_body(
            args,
            contract=contract,
            contract_sha=contract_sha,
            final_output=final_output,
            output=staging,
        )
        _verify_manifest_directory(staging, allow_transaction_marker=True)
        remove_process_owned_transaction_marker(staging, final_output)
        publish_staging_atomically(staging, final_output)
    except BaseException as error:
        active_logger = logging.getLogger("global_context_latent_geometry_ablation")
        if active_logger.handlers:
            try:
                active_logger.exception(
                    "stage=failure private_staging=%s final_output=%s error_type=%s error=%s",
                    staging,
                    final_output,
                    type(error).__name__,
                    error,
                )
            finally:
                close_logging(active_logger)
        quarantine = quarantine_failed_staging(
            staging,
            final_output,
            error,
            failure_stage="staged_execution_or_terminal_publication",
        )
        print(
            "stage=failure_quarantine " + json.dumps(quarantine, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        raise
    manifest_path = final_output / "artifact_manifest.json"
    print(
        f"stage=artifact_manifest_written_and_atomically_published path={manifest_path} "
        f"sha256={sha256_file(manifest_path)} declared_files={manifest['file_count']} total_files=22",
        flush=True,
    )


def self_test(args: argparse.Namespace) -> int:
    """No-input corruption/static tests; never imports or forwards Weight-AE."""
    results: dict[str, Any] = {
        "schema_version": "global_context_runner_self_test_v1",
        "frozen_design_sha256": DESIGN_SHA256,
        "tests": {},
    }
    results["tests"]["stable_seed"] = {
        "value": stable_seed(26081901, "global_context_common_tiling_v2|shape=768x768")
    }
    width_fixtures = {
        "uniform_512": {"a": torch.empty((1, 512)), "b": torch.empty((2, 512))},
        "uniform_511": {"a": torch.empty((1, 511)), "b": torch.empty((2, 511))},
        "mixed_511_512": {"a": torch.empty((1, 511)), "b": torch.empty((2, 512))},
        "empty": {},
    }
    width_observations = {
        label: observe_latent_code_width(fixture) for label, fixture in width_fixtures.items()
    }
    expected_width_observations = {
        "uniform_512": 512,
        "uniform_511": 511,
        "mixed_511_512": [511, 512],
        "empty": [],
    }
    if width_observations != expected_width_observations:
        raise RuntimeError(
            "latent-code-width regression fixture failed: "
            f"{width_observations} != {expected_width_observations}"
        )
    results["tests"]["latent_code_width_regression"] = {
        "observed": width_observations,
        "expected": expected_width_observations,
        "correct_width_equals_frozen_expected": width_observations["uniform_512"]
        == expected_counts()["latent_code_width"],
        "pass": True,
    }
    tilings: dict[tuple[int, tuple[int, int]], TilingIndex] = {}
    for seed in COMMON_TILING_SEEDS:
        for shape in sorted(set(ROLE_SHAPES.values())):
            first = make_common_tiling(seed, shape)
            second = make_common_tiling(seed, shape)
            if not torch.equal(first.rows, second.rows) or not torch.equal(first.cols, second.cols):
                raise RuntimeError("shape-only tiling repeatability self-test failed")
            dummy = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
            validate_tiling(dummy, first, f"self-test/{seed}/{shape}")
            identity = f"global_context_common_tiling_v2|shape={shape[0]}x{shape[1]}"
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stable_seed(seed, identity))
            patch_groups = make_partition(shape[0] // 16, 4, generator)
            offsets = torch.arange(16, dtype=torch.int64)
            expected_rows = torch.stack(
                [
                    (group[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values
                    for group in patch_groups
                ]
            ).to(torch.int64).contiguous()
            expected_cols = make_partition(shape[1], 64, generator)
            if not torch.equal(first.rows, expected_rows) or not torch.equal(first.cols, expected_cols):
                raise RuntimeError("literal row-then-column RNG order self-test failed")
            tilings[(seed, shape)] = first
    attention_hashes = {
        make_common_tiling(seed, ROLE_SHAPES[role]).partition_sha256
        for seed in COMMON_TILING_SEEDS
        for role in ("attn_query", "attn_key", "attn_value", "attn_output")
    }
    if len(attention_hashes) != 2:
        raise RuntimeError("same-shape attention tiling identity self-test failed")
    results["tests"]["shape_only_tiling"] = {
        "groups": {
            f"seed={seed}|shape={shape[0]}x{shape[1]}": tiling.partition_sha256
            for (seed, shape), tiling in tilings.items()
        },
        "same_shape_attention_partition_count_across_two_seeds": len(attention_hashes),
        "pass": True,
    }
    relabel_shape = ROLE_SHAPES["attn_query"]
    relabel_W = torch.arange(relabel_shape[0] * relabel_shape[1], dtype=torch.float32).reshape(relabel_shape)
    relabel_tiling_a = make_common_tiling(COMMON_TILING_SEEDS[0], relabel_shape)
    tiles_a = split_tiles(relabel_W, relabel_tiling_a)
    fake_metadata_before = {"panel_id": "panel_a", "role": "attn_query", "depth": 0}
    fake_metadata_after = {"panel_id": "relabelled", "role": "attn_output", "depth": 11}
    del fake_metadata_before, fake_metadata_after
    relabel_tiling_b = make_common_tiling(COMMON_TILING_SEEDS[0], relabel_shape)
    tiles_b = split_tiles(relabel_W, relabel_tiling_b)
    metadata_relabel_pass = bool(
        relabel_tiling_a.partition_sha256 == relabel_tiling_b.partition_sha256
        and tensor_sha256(tiles_a, dtype=torch.float32) == tensor_sha256(tiles_b, dtype=torch.float32)
    )
    if not metadata_relabel_pass:
        raise RuntimeError("metadata-relabel tiling/W-tile invariance self-test failed")
    results["tests"]["metadata_relabel_invariance"] = {
        "shape": list(relabel_shape),
        "partition_sha256": relabel_tiling_a.partition_sha256,
        "w_tiles_sha256": tensor_sha256(tiles_a, dtype=torch.float32),
        "metadata_values_not_passed_to_tiling_or_split": True,
        "pass": True,
    }
    maps, map_payload = build_countsketch_maps()
    if map_payload["total_pair_count"] != 20480:
        raise RuntimeError("CountSketch self-test failed")
    results["tests"]["countsketch"] = {
        "pair_count": map_payload["total_pair_count"],
        "map_hashes": {
            str(seed): map_payload["seeds"][str(seed)]["pair_map_sha256"]
            for seed in COUNTSKETCH_SEEDS
        },
        "pass": True,
    }
    independent_map_hashes: dict[str, str] = {}
    for seed in COUNTSKETCH_SEEDS:
        independent_buckets = torch.empty(4096, dtype=torch.int64)
        independent_signs = torch.empty(4096, dtype=torch.int64)
        for coordinate in range(4096):
            raw_digest = hashlib.sha256(
                ("global_context_countsketch_v1|" + str(seed) + "|" + str(coordinate)).encode("utf-8")
            ).digest()
            independent_buckets[coordinate] = int.from_bytes(raw_digest[:8], byteorder="big", signed=False) % 512
            independent_signs[coordinate] = 1 if int(raw_digest[8]) % 2 == 0 else -1
        independent_hash = hashlib.sha256(
            (
                tensor_sha256(independent_buckets, dtype=torch.int64)
                + "|"
                + tensor_sha256(independent_signs, dtype=torch.int64)
            ).encode("utf-8")
        ).hexdigest()
        if independent_hash != map_payload["seeds"][str(seed)]["pair_map_sha256"]:
            raise RuntimeError(f"independent CountSketch recomputation failed: seed={seed}")
        independent_map_hashes[str(seed)] = independent_hash
    results["tests"]["countsketch"]["independent_recomputed_map_hashes"] = independent_map_hashes
    sample = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) / 4096.0
    raw = raw_simple_descriptor(sample)
    aggregate = aggregate_tile_features(torch.arange(3 * 512, dtype=torch.float32).reshape(3, 512))
    if raw.shape != (37,) or aggregate.shape != (2560,):
        raise RuntimeError("feature-definition self-test failed")
    results["tests"]["features"] = {
        "raw_feature_names": list(RAW_FEATURE_NAMES),
        "raw_shape": list(raw.shape),
        "aggregate_shape": list(aggregate.shape),
        "pass": True,
    }
    static = static_call_graph_audit()
    if not static["pass"]:
        raise RuntimeError("static call-graph self-test failed")
    results["tests"]["static_call_graph"] = static
    results["tests"]["old_cache_provenance_only"] = {
        "old_tiling_seeds_disjoint_from_common": not bool(
            {26081601, 26081602, 26081801, 26081802}.intersection(COMMON_TILING_SEEDS)
        ),
        "old_seed_representations": sorted(
            representation for representation in REPRESENTATIONS
            if any(str(seed) in representation for seed in (26081601, 26081602, 26081801, 26081802))
        ),
        "feature_builder_references_old_cache_paths": False,
        "pass": True,
    }
    if results["tests"]["old_cache_provenance_only"]["old_seed_representations"]:
        raise RuntimeError("old cache seed leaked into representation grid")

    class _TinyFingerprintFixture(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(7, 5)
            self.register_buffer("deterministic_buffer", torch.arange(3, dtype=torch.float32))

    tiny_models: list[torch.nn.Module] = []
    for seed in (101, 102, 103):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            tiny_models.append(_TinyFingerprintFixture())
    tiny_fingerprints = [model_fingerprint(model) for model in tiny_models]
    grid_hashes = {value["parameter_grid_sha256"] for value in tiny_fingerprints}
    state_hashes = {value["state_dict_sha256"] for value in tiny_fingerprints}
    buffer_hashes = {
        value["buffers"][0]["tensor_sha256"] for value in tiny_fingerprints
    }
    if len(grid_hashes) != 1 or len(state_hashes) != 3 or len(buffer_hashes) != 1:
        raise RuntimeError("untrained fingerprint/buffer handling fixture failed")
    results["tests"]["untrained_fingerprint_fixture"] = {
        "parameter_grid_hash_count": len(grid_hashes),
        "state_dict_hash_count": len(state_hashes),
        "shared_deterministic_buffer_hash_count": len(buffer_hashes),
        "checkpoint_tensor_copy_count": 0,
        "pass": True,
    }
    training_mode_rejected = False
    try:
        assert_model_fully_eval(tiny_models[0], label="synthetic learned model")
    except RuntimeError:
        training_mode_rejected = True
    tiny_models[0].eval()
    assert_model_fully_eval(tiny_models[0], label="synthetic learned model")
    if not training_mode_rejected:
        raise RuntimeError("learned training-mode preflight fixture was accepted")
    results["tests"]["learned_eval_mode_gate"] = {
        "training_mode_rejected": True,
        "fully_eval_mode_accepted": True,
        "weight_ae_forward_calls": 0,
        "pass": True,
    }

    current_contract_fixture = {
        "schema_version": "fixture",
        "runner_sha256": "a" * 64,
        "analyzer_sha256": "b" * 64,
        "input": {"sha256": "c" * 64},
    }
    frozen_contract_fixture = copy.deepcopy(current_contract_fixture)
    assert_contract_static_fields_match(frozen_contract_fixture, current_contract_fixture)
    corrupted_contract_fixture = copy.deepcopy(frozen_contract_fixture)
    corrupted_contract_fixture["input"]["sha256"] = "d" * 64
    contract_corruption_rejected = False
    try:
        assert_contract_static_fields_match(corrupted_contract_fixture, current_contract_fixture)
    except RuntimeError:
        contract_corruption_rejected = True
    if not contract_corruption_rejected:
        raise RuntimeError("contract hash corruption fixture was accepted")
    results["tests"]["contract_exact_binding_fixture"] = {
        "valid_exact_binding_pass": True,
        "nested_input_hash_corruption_rejected": True,
        "contract_schema_keys": list(CONTRACT_SCHEMA_KEYS),
        "pass": True,
    }

    formal_args = copy.deepcopy(args)
    formal_args.contract_path = DEFAULT_CONTRACT
    formal_args.output_dir = DEFAULT_OUTPUT
    formal_args.analyzer_output_dir = DEFAULT_ANALYZER_OUTPUT
    formal_args.analyzer_path = DEFAULT_ANALYZER
    formal_args.panel_builder_path = DEFAULT_PANEL_BUILDER
    exact_paths = assert_exact_formal_paths(formal_args)
    path_mutations = {
        "contract_path": DEFAULT_CONTRACT.with_name("wrong_contract.json"),
        "output_dir": DEFAULT_OUTPUT.with_name("wrong_runner_output"),
        "analyzer_output_dir": DEFAULT_ANALYZER_OUTPUT.with_name("wrong_analyzer_output"),
        "analyzer_path": Path(__file__).resolve(strict=True),
        "panel_builder_path": Path(__file__).resolve(strict=True),
    }
    mutation_rejections: dict[str, bool] = {}
    for field, mutated_value in path_mutations.items():
        corrupted = copy.deepcopy(formal_args)
        setattr(corrupted, field, mutated_value)
        rejected = False
        try:
            assert_exact_formal_paths(corrupted)
        except RuntimeError:
            rejected = True
        if not rejected:
            raise RuntimeError(f"formal path/script override fixture accepted: {field}")
        mutation_rejections[field] = True
    results["tests"]["exact_formal_path_binding"] = {
        "exact_defaults": exact_paths,
        "mutation_rejections": mutation_rejections,
        "rejected_before_scientific_input_or_model_import": True,
        "pass": True,
    }

    invocation_base = copy.deepcopy(formal_args)
    invocation_base.self_test = False
    invocation_base.prepare_contract = False
    invocation_base.execute = False
    invocation_base.audit_report = None
    reviewed_entrypoint = Path(__file__).resolve(strict=True)
    canonical_invocations: dict[str, dict[str, Any]] = {}
    canonical_invocations["audit_only_stdout"] = assert_canonical_formal_entrypoint_and_argv(
        invocation_base,
        argv0=reviewed_entrypoint,
        argv=[],
    )
    audit_report_invocation = copy.deepcopy(invocation_base)
    audit_report_invocation.audit_report = Path("fresh_audit_report.json")
    canonical_invocations["audit_only_report"] = assert_canonical_formal_entrypoint_and_argv(
        audit_report_invocation,
        argv0=reviewed_entrypoint,
        argv=["--audit-report", "fresh_audit_report.json"],
    )
    prepare_invocation = copy.deepcopy(invocation_base)
    prepare_invocation.prepare_contract = True
    canonical_invocations["prepare_contract"] = assert_canonical_formal_entrypoint_and_argv(
        prepare_invocation,
        argv0=reviewed_entrypoint,
        argv=["--prepare-contract"],
    )
    execute_invocation = copy.deepcopy(invocation_base)
    execute_invocation.execute = True
    canonical_invocations["execute"] = assert_canonical_formal_entrypoint_and_argv(
        execute_invocation,
        argv0=reviewed_entrypoint,
        argv=["--execute"],
    )

    entrypoint_mutation_rejected = False
    try:
        assert_canonical_formal_entrypoint_and_argv(
            invocation_base,
            argv0=DEFAULT_ANALYZER.resolve(strict=True),
            argv=[],
        )
    except RuntimeError:
        entrypoint_mutation_rejected = True
    if not entrypoint_mutation_rejected:
        raise RuntimeError("alternate formal runner entrypoint fixture was accepted")

    noncanonical_argv_rejections: dict[str, bool] = {}
    argv_mutations = {
        "audit_only_extra_default": (
            invocation_base,
            ["--device", "cuda:0"],
        ),
        "audit_report_equals_syntax": (
            audit_report_invocation,
            ["--audit-report=fresh_audit_report.json"],
        ),
        "prepare_extra_default": (
            prepare_invocation,
            ["--prepare-contract", "--batch-size", "64"],
        ),
        "execute_extra_default": (
            execute_invocation,
            ["--execute", "--log-every-batches", "20"],
        ),
    }
    for label, (invocation_args, mutated_argv) in argv_mutations.items():
        rejected = False
        try:
            assert_canonical_formal_entrypoint_and_argv(
                invocation_args,
                argv0=reviewed_entrypoint,
                argv=mutated_argv,
            )
        except RuntimeError:
            rejected = True
        if not rejected:
            raise RuntimeError(f"noncanonical formal argv fixture was accepted: {label}")
        noncanonical_argv_rejections[label] = True
    results["tests"]["canonical_formal_entrypoint_and_argv"] = {
        "canonical_invocations": canonical_invocations,
        "alternate_entrypoint_rejected": entrypoint_mutation_rejected,
        "noncanonical_argv_rejections": noncanonical_argv_rejections,
        "rejected_before_scientific_input_or_model_import": True,
        "pass": True,
    }
    closure_fixture = assert_loaded_local_modules_frozen(
        {
            "model_dependencies": {},
            "scripts": {
                "runner": {"path": str(Path(__file__).resolve(strict=True))},
            },
        }
    )
    if closure_fixture["pass"] is not True:
        raise RuntimeError("runner self-allowlist closure fixture failed")
    results["tests"]["loaded_local_module_closure_fixture"] = {
        "runner_script_allowlisted": True,
        "loaded_module_count": closure_fixture["loaded_module_count"],
        "pass": True,
    }

    with tempfile.TemporaryDirectory(prefix="global-context-runner-self-test-") as temporary_text:
        root = Path(temporary_text)
        for name in EXPECTED_RAW_FILES:
            path = root / name
            path.write_bytes(f"fixture:{name}\n".encode("utf-8"))
        fake_binding = {
            "runner_sha256": "0" * 64,
            "analyzer_sha256": "1" * 64,
            "external_contract_sha256": "2" * 64,
        }
        atomic_write_json(root / "artifact_manifest.json", artifact_manifest(root, fake_binding))
        _verify_manifest_directory(root)
        victim = root / EXPECTED_RAW_FILES[0]
        original = victim.read_bytes()
        victim.write_bytes(original + b"corruption")
        corruption_rejected = False
        try:
            _verify_manifest_directory(root)
        except RuntimeError:
            corruption_rejected = True
        if not corruption_rejected:
            raise RuntimeError("manifest corruption fixture was accepted")
        victim.write_bytes(original)
        _verify_manifest_directory(root)
        missing = root / EXPECTED_RAW_FILES[1]
        backup = missing.read_bytes()
        missing.unlink()
        missing_rejected = False
        try:
            _verify_manifest_directory(root)
        except RuntimeError:
            missing_rejected = True
        if not missing_rejected:
            raise RuntimeError("manifest missing-file fixture was accepted")
        missing.write_bytes(backup)
        atomic_write_json(root / "artifact_manifest.json", artifact_manifest(root, fake_binding))
        symlink_member = root / EXPECTED_RAW_FILES[2]
        symlink_bytes = symlink_member.read_bytes()
        symlink_member.unlink()
        symlink_member.symlink_to(EXPECTED_RAW_FILES[3])
        symlink_rejected = False
        try:
            _verify_manifest_directory(root)
        except RuntimeError:
            symlink_rejected = True
        if not symlink_rejected:
            raise RuntimeError("manifest symlink fixture was accepted")
        symlink_member.unlink()
        symlink_member.write_bytes(symlink_bytes)
        results["tests"]["artifact_manifest_corruption"] = {
            "valid_fixture_pass": True,
            "byte_corruption_rejected": corruption_rejected,
            "missing_file_rejected": missing_rejected,
            "symlink_member_rejected": symlink_rejected,
            "declared_files": 21,
            "total_files": 22,
        }

    with tempfile.TemporaryDirectory(
        prefix="global-context-transaction-self-test-"
    ) as transaction_text:
        transaction_root = Path(transaction_text)
        final_fixture = transaction_root / "formal_output"
        transaction_hashes = {
            "runner_sha256": "3" * 64,
            "analyzer_sha256": "4" * 64,
            "external_contract_sha256": "5" * 64,
        }

        before_stage = create_process_owned_staging(final_fixture, **transaction_hashes)
        before_quarantine = quarantine_failed_staging(
            before_stage,
            final_fixture,
            RuntimeError("forced failure before synthetic forward"),
            failure_stage="before_synthetic_forward",
        )
        before_quarantine_path = Path(str(before_quarantine["quarantine_path"]))
        if (
            final_fixture.exists()
            or not before_quarantine_path.is_dir()
            or not (before_quarantine_path / FAILURE_RECORD_NAME).is_file()
        ):
            raise RuntimeError("pre-forward failure quarantine fixture failed")

        after_stage = create_process_owned_staging(final_fixture, **transaction_hashes)
        (after_stage / "synthetic_forward_marker.txt").write_text(
            "no model was imported or forwarded\n", encoding="utf-8"
        )
        after_quarantine = quarantine_failed_staging(
            after_stage,
            final_fixture,
            RuntimeError("forced failure after synthetic forward marker"),
            failure_stage="after_synthetic_forward_marker",
        )
        after_quarantine_path = Path(str(after_quarantine["quarantine_path"]))
        if (
            final_fixture.exists()
            or not after_quarantine_path.is_dir()
            or not (after_quarantine_path / "synthetic_forward_marker.txt").is_file()
            or not (after_quarantine_path / FAILURE_RECORD_NAME).is_file()
        ):
            raise RuntimeError("post-marker failure quarantine fixture failed")

        success_stage = create_process_owned_staging(final_fixture, **transaction_hashes)
        for name in EXPECTED_RAW_FILES:
            (success_stage / name).write_bytes(f"transaction-fixture:{name}\n".encode("utf-8"))
        success_binding = {
            "runner_sha256": transaction_hashes["runner_sha256"],
            "analyzer_sha256": transaction_hashes["analyzer_sha256"],
            "external_contract_sha256": transaction_hashes["external_contract_sha256"],
        }
        atomic_write_json(
            success_stage / "artifact_manifest.json",
            artifact_manifest(
                success_stage,
                success_binding,
                allow_transaction_marker=True,
            ),
        )
        _verify_manifest_directory(success_stage, allow_transaction_marker=True)
        remove_process_owned_transaction_marker(success_stage, final_fixture)
        publish_staging_atomically(success_stage, final_fixture)
        _verify_manifest_directory(final_fixture)
        if success_stage.exists() or not final_fixture.is_dir():
            raise RuntimeError("successful atomic publication fixture failed")

        source_text = Path(__file__).resolve(strict=True).read_text(encoding="utf-8")
        forbidden_staging_identity_fragments = (
            '"copied_contract_path": str((' + "output /",
            '"runner_output_path": str(' + "output)",
            '"artifact_paths": {name: str((' + "output /",
        )
        if any(fragment in source_text for fragment in forbidden_staging_identity_fragments):
            raise RuntimeError("staging path leaked into a bound artifact identity field")
        results["tests"]["failure_quarantine_and_atomic_publication"] = {
            "forced_failure_before_synthetic_forward_final_absent": True,
            "forced_failure_after_synthetic_forward_marker_final_absent": True,
            "both_failed_trees_quarantined_with_failure_record": True,
            "same_final_path_rerunnable_after_each_failure": True,
            "success_terminal_grid_validated_before_publish": True,
            "success_atomically_published_exact_21_plus_1_grid": True,
            "embedded_identity_fields_use_final_path": True,
            "weight_ae_model_imported": False,
            "weight_ae_forward_calls": 0,
            "pass": True,
        }
    results["weight_ae_model_imported"] = any(
        name == "big_vae.models" or name.startswith("big_vae.models.") for name in sys.modules
    )
    results["weight_ae_forward_count"] = 0
    results["pass"] = not results["weight_ae_model_imported"]
    if args.audit_report is not None:
        report_path = args.audit_report.resolve(strict=False)
        if report_path.exists():
            raise RuntimeError(f"self-test report path must be fresh: {report_path}")
        atomic_write_json(report_path, results)
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    return 0 if results["pass"] else 1


def main() -> int:
    args = parser().parse_args()
    install_source_only_seal()
    reject_forbidden_arguments(args)
    assert_seal_clean()
    if args.self_test:
        return self_test(args)
    # All non-self-test invocations are formal audit/prepare/execute modes.
    # Reject path/script overrides before runtime/dependency/scientific reads.
    assert_exact_formal_paths(args)
    assert_canonical_formal_entrypoint_and_argv(args)
    if args.prepare_contract:
        prepare_contract(args)
        return 0
    if args.execute:
        execute(args)
        return 0
    return audit_only(args)


if __name__ == "__main__":
    raise SystemExit(main())
