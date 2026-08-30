#!/usr/bin/env python3
"""Formal Weight-AE runner for the frozen geometry-matched replication.

The default mode is audit-only.  It installs the source-only target seal before
reading any scientific input and never imports or executes the Weight-AE.
``--prepare-contract`` performs the same no-Weight-AE audit and writes an exact
preexecution contract.  Only ``--execute`` may import the audited source model
code and run Weight-AE forwards, and it first verifies that contract byte for
byte against the current inputs and implementation.

This program deliberately does not run either prospective panel's task model.
Those forwards belong exclusively to the independently sealed panel builder.
It also does not invoke the CPU analyzer; subprocess launches are prohibited in
the formal runner process.
"""

from __future__ import annotations

import argparse
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
import struct
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Offline flags are process policy, not a best-effort model-loading option.
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

DESIGN_PATH = PROJECT_ROOT / "docs/notes/prospective_geometry_matched_source_replication_design_20260816.md"
DESIGN_SHA256 = "fc23d52d07d0004afd681116578abdba562dff1a415b059fd72ec261b1e47d70"
AE_CHECKPOINT = PROJECT_ROOT / "artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt"
AE_CHECKPOINT_SHA256 = "d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00"
SOURCE_TEMPLATE = ARTIFACT_ROOT / "source_confirmatory_gate_20260816_clean2/source_condition_templates.pt"
SOURCE_TEMPLATE_SHA256 = "8fc6c61bb6baae4e7b1d618133ec651a91386d90c66f540182faa7dfb1655f99"
SOURCE_GAINS = ARTIFACT_ROOT / "source_confirmatory_gate_20260816_clean2/source_fit_role_gains.csv"
SOURCE_GAINS_SHA256 = "e05f2339ba33eb65feee00df2f095655c9319ecf118cd81d7159f6e14f70c4f3"
SOURCE_SCOUT_MANIFEST = ARTIFACT_ROOT / "prospective_source_panel_scout_20260816/artifact_manifest.json"
SOURCE_SCOUT_MANIFEST_SHA256 = "623cbf54611bdba49a0007840b52457cbfa382935507cb13020ffef5e44e5254"
SOURCE_PARENT = ARTIFACT_ROOT / "source_confirmatory_gate_20260816_clean2"
SOURCE_FACTORIAL_CACHE = (
    ARTIFACT_ROOT / "source_latent_code_factorial_20260816/factorial_code_cache_seed_26081601.pt"
)
SOURCE_FACTORIAL_CACHE_SHA256 = "a585c2ecfe4d1e560c79dca19518543dbdfb5f897bfa413b3ef5148f01995c8f"
SOURCE_FACTORIAL_CACHE_SECONDARY = (
    ARTIFACT_ROOT / "source_latent_code_factorial_20260816/factorial_code_cache_seed_26081602.pt"
)
SOURCE_FACTORIAL_CACHE_SECONDARY_SHA256 = "c56f1ed3ccd0038f3f78ec767facd32cc0bba490f2e31229ec5d7b18a211bfd6"
SOURCE_PARENT_HASHES = {
    "heldout_panel_manifest.json": "10020aeabfcce6fe2df3003a7a626f21735cd0763c664e178f27ebcbf42ff985",
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

DEFAULT_PANEL_CACHE = ARTIFACT_ROOT / "prospective_geometry_matched_panels_20260816"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "prospective_geometry_matched_replication_20260816"
DEFAULT_CONTRACT = ARTIFACT_ROOT / "prospective_geometry_matched_replication_preexecution_contract_20260816.json"
DEFAULT_BUILDER = EXPERIMENTS_ROOT / "build_prospective_geometry_matched_panels.py"
DEFAULT_ANALYZER = EXPERIMENTS_ROOT / "analyze_prospective_geometry_matched_replication.py"

PANEL_IDS = ("beans", "trocr_sroie")
ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
ROLE_SHAPES = {
    "attn_query": (768, 768),
    "attn_key": (768, 768),
    "attn_value": (768, 768),
    "attn_output": (768, 768),
    "ffn_up": (768, 3072),
    "ffn_down": (3072, 768),
}
PANEL_CHECKPOINT_SHA256 = {
    "beans": "fc443b145fcc3a09cf07eb28b94e9a989ec3f0e8f7255e1fa08c0953ed4bae91",
    "trocr_sroie": "1cf4a6eedab26afaaf505f1c7f73d9634944924dbd1ed049d569db98039cd596",
}
BEANS_MODEL_ROOT = PROJECT_ROOT / "projects/shared/storage/data/models/vit_base_beans_nateraw_41f85ace"
BEANS_DATA_ROOT = PROJECT_ROOT / "projects/shared/storage/data/datasets/beans_27aa014c"
TROCR_MODEL_ROOT = (
    PROJECT_ROOT
    / "projects/shared/storage/data/models/trocr_base_printed/models--microsoft--trocr-base-printed/"
    "snapshots/93450be3f1ed40a930690d951ef3932687cc1892"
)
SROIE_DATA_ROOT = PROJECT_ROOT / "projects/shared/storage/data/datasets/sroie_text_recognition_04f6537e"
PANEL_RESOURCE_SPECS: dict[str, dict[str, tuple[Path, str]]] = {
    "beans": {
        "beans_model_weights": (
            BEANS_MODEL_ROOT / "pytorch_model.bin",
            PANEL_CHECKPOINT_SHA256["beans"],
        ),
        "beans_model_config": (
            BEANS_MODEL_ROOT / "config.json",
            "366d2ad9bf48e94932bb83e0d2367e6f95ed18befb1a4ef6903934b7af737237",
        ),
        "beans_preprocessor_config": (
            BEANS_MODEL_ROOT / "preprocessor_config.json",
            "af4eb4d79cf61b47010fc0bc9352ee967579c417423b4917188d809b7e048948",
        ),
        "beans_train_parquet": (
            BEANS_DATA_ROOT / "data/train-00000-of-00001.parquet",
            "7f905a7323966a58e89b8e839ed656bb869fc82d16a3fadc7dce40972a5f8b19",
        ),
        "beans_validation_parquet": (
            BEANS_DATA_ROOT / "data/validation-00000-of-00001.parquet",
            "33f774593d8b31585457b70c224744e9409ffdee4e91a11822b1ebfe8242928f",
        ),
        "beans_test_parquet": (
            BEANS_DATA_ROOT / "data/test-00000-of-00001.parquet",
            "534a6b0648f585d69b7ec0ad7a7540720d60c8db8106dc6d0508296316f6cb27",
        ),
    },
    "trocr_sroie": {
        "trocr_model_weights": (
            TROCR_MODEL_ROOT / "model.safetensors",
            PANEL_CHECKPOINT_SHA256["trocr_sroie"],
        ),
        "trocr_model_config": (
            TROCR_MODEL_ROOT / "config.json",
            "5bda1deab455661feb3d91906656e5600e2ca520d5c00a2a03836614b850c93e",
        ),
        "trocr_preprocessor_config": (
            TROCR_MODEL_ROOT / "preprocessor_config.json",
            "2fcc0da9466ee00be0403b26027373039e032820ebac409e207b32e52e52119d",
        ),
        "trocr_generation_config": (
            TROCR_MODEL_ROOT / "generation_config.json",
            "41149cdcffec4d657f32dfcddd9b208037f01286c9e07945c724908c58ed0193",
        ),
        "trocr_tokenizer_config": (
            TROCR_MODEL_ROOT / "tokenizer_config.json",
            "5a1356884c6ae736a621841535264ba7c5bebd52f169258add2c48fcbb32d50a",
        ),
        "trocr_special_tokens_map": (
            TROCR_MODEL_ROOT / "special_tokens_map.json",
            "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
        ),
        "trocr_vocab": (
            TROCR_MODEL_ROOT / "vocab.json",
            "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
        ),
        "trocr_merges": (
            TROCR_MODEL_ROOT / "merges.txt",
            "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        ),
        "sroie_test_zip": (
            SROIE_DATA_ROOT / "test.zip",
            "533dba4d017a70617943f857fe01a986d34da5095255112f88af03df4325484b",
        ),
    },
}
ACTIVATION_ROWS = {"beans": 1008, "trocr_sroie": 1024}
SCORE_SPLITS = ("A", "B")
TILING_SEEDS = (26081801, 26081802)
ARMS = ("correct", "permuted_within_row", "zero_code")
GLOBAL_SEED = 26081800
SOURCE_ROLE_GAINS = {
    "attn_query": 0.5270182885626962,
    "attn_key": 0.5562907139098529,
    "attn_value": 0.3990051254514762,
    "attn_output": 0.3564476277035192,
    "ffn_up": 1.0280206811266708,
    "ffn_down": 0.3048084709039325,
}
SOURCE_MEDIAN_COMMON_GAIN = 0.46301170700708616

EXPECTED_RUNTIME = {
    "python_major_minor": "3.12",
    "torch": "2.10.0+cu128",
    "transformers": "5.1.0",
    "safetensors": "0.7.0",
    "Pillow": "12.0.0",
    "datasets": "3.6.0",
    "numpy": "2.3.5",
    "scikit-learn": "1.8.0",
    "pandas": "3.0.0",
    "pyarrow": "23.0.0",
    "omegaconf": "2.3.0",
    "hydra-core": "1.3.2",
    "matplotlib": "3.10.8",
    "scipy": "1.17.0",
}

# The exact source helpers plus recursively frozen local model, training, and
# dataset trees.  The helper imports execute code from all three trees (and the
# background prefetch helper), so a narrower model-only snapshot is not a
# closed import provenance boundary.  Expansion occurs only after the
# source-only seal is installed.
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
REQUIRED_DEPENDENCY_FILES = (
    WORKSPACE_ROOT / "training/__init__.py",
    WORKSPACE_ROOT / "training/forensics.py",
    WORKSPACE_ROOT / "training/optim.py",
    WORKSPACE_ROOT / "training/runtime.py",
    EXPERIMENTS_ROOT / "background_prefetch.py",
)

REQUIRED_PANEL_KEYS = {
    "schema_version",
    "panel_id",
    "checkpoint_sha256",
    "quality_status",
    "weights",
    "activations",
    "matrix_meta",
    "resource_hashes",
    "build_manifest",
}
REQUIRED_BUILD_MANIFEST_KEYS = {
    "panel_id",
    "panel_valid",
    "seal_pass",
    "provenance_pass",
    "geometry_pass",
    "quality_gate_pass",
    "unseen_weight_pass",
    "activation_gate_pass",
    "matrix_count",
    "exact_overlap_count",
    "training_bank_compatible_count",
    "score_splits",
    "evidence_files",
}
REQUIRED_PANEL_EVIDENCE_LABELS = {
    "resource_hashes",
    "unseen_weight_summary",
    "unseen_weight_audit",
    "training_bank_compatible_weight_hashes",
    "resolved_config",
    "sample_manifest",
    "token_manifest",
    "model_loading",
    "runtime_matrix_parity",
    "quality_predictions",
    "quality_metrics",
    "activation_gate",
    "panel_tensor_manifest",
}


class SealViolation(RuntimeError):
    """Raised when a sealed target, network, or subprocess access is attempted."""


class _ForbiddenModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> None:
        del path, target
        if "data2vec" in fullname.lower():
            with _SEAL_LOCK:
                _SEAL_STATE["target_access_events"] += 1
            raise SealViolation(f"source-only seal blocked module import: {fullname}")
        return None


_SEAL_LOCK = threading.Lock()
_SEAL_STATE: dict[str, Any] = {
    "installed": False,
    "installed_before_input_read": False,
    "forbidden_path_markers": ["data2vec", "target path component"],
    "target_access_events": 0,
    "network_connections": 0,
    "subprocess_launches": 0,
}
_SEAL_RESOLVE_GUARD = threading.local()


def _forbidden_path_text(text: str) -> bool:
    normalized = text.replace("\\", "/").casefold()
    components = [component for component in normalized.split("/") if component]
    return "data2vec" in normalized or any(
        component in {"target", "targets"} or component.startswith(("target_", "target-"))
        for component in components
    )


def _path_contains_forbidden_marker(value: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> bool:
    original = os.fsdecode(value)
    if _forbidden_path_text(original):
        return True
    if getattr(_SEAL_RESOLVE_GUARD, "active", False):
        return False
    try:
        _SEAL_RESOLVE_GUARD.active = True
        resolved = os.path.realpath(os.fsdecode(value))
    except (OSError, ValueError):
        resolved = original
    finally:
        _SEAL_RESOLVE_GUARD.active = False
    return _forbidden_path_text(resolved)


def install_source_only_seal() -> dict[str, Any]:
    """Install irreversible process-wide target/network/subprocess guards."""
    if _SEAL_STATE["installed"]:
        return dict(_SEAL_STATE)

    forbidden_loaded = sorted(name for name in sys.modules if "data2vec" in name.lower())
    if forbidden_loaded:
        raise SealViolation(f"data2vec modules were loaded before seal installation: {forbidden_loaded}")

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event in {"open", "os.listdir", "os.scandir"} and args:
            candidate = args[0]
            if isinstance(candidate, (str, bytes, os.PathLike)) and _path_contains_forbidden_marker(candidate):
                with _SEAL_LOCK:
                    _SEAL_STATE["target_access_events"] += 1
                raise SealViolation(f"source-only seal blocked filesystem event={event}: {candidate}")
        if event == "socket.connect":
            with _SEAL_LOCK:
                _SEAL_STATE["network_connections"] += 1
            raise SealViolation("source-only seal blocked a network connection")
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"} or event.startswith(
            "os.spawn"
        ):
            with _SEAL_LOCK:
                _SEAL_STATE["subprocess_launches"] += 1
            raise SealViolation("source-only seal blocked a subprocess launch")

    sys.addaudithook(audit_hook)
    sys.meta_path.insert(0, _ForbiddenModuleFinder())
    _SEAL_STATE.update(
        {
            "installed": True,
            "installed_before_input_read": True,
            "audit_events": [
                "open",
                "os.listdir",
                "os.scandir",
                "socket.connect",
                "subprocess.Popen",
                "os.system",
                "os.posix_spawn",
                "os.spawn*",
                "os.exec",
            ],
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
    loaded = sorted(name for name in sys.modules if "data2vec" in name.lower())
    if loaded:
        raise SealViolation(f"forbidden modules are present after seal installation: {loaded}")
    nonzero = {
        key: int(_SEAL_STATE[key])
        for key in ("target_access_events", "network_connections", "subprocess_launches")
        if int(_SEAL_STATE[key]) != 0
    }
    if nonzero:
        raise SealViolation(f"source-only seal recorded prohibited attempts: {nonzero}")


def reject_forbidden_argv(argv: Sequence[str]) -> None:
    bad = [value for value in argv if _forbidden_path_text(value)]
    if bad:
        with _SEAL_LOCK:
            _SEAL_STATE["target_access_events"] += len(bad)
        raise SealViolation(f"command line contains forbidden target marker: {bad}")


def reject_forbidden_path_arguments(args: argparse.Namespace) -> None:
    for name in (
        "panel_cache_dir",
        "builder_path",
        "analyzer_path",
        "contract_path",
        "output_dir",
        "audit_report",
    ):
        value = getattr(args, name, None)
        if value is None:
            continue
        raw = os.fspath(value)
        # Check the original spelling before resolving a symlink, then check
        # the resolved spelling without opening the target.
        if _forbidden_path_text(raw) or _forbidden_path_text(os.path.realpath(os.path.abspath(raw))):
            with _SEAL_LOCK:
                _SEAL_STATE["target_access_events"] += 1
            raise SealViolation(f"forbidden target-related path argument {name}={raw}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare-contract",
        action="store_true",
        help="Audit all frozen inputs and atomically create the preexecution contract; no Weight-AE import/forward.",
    )
    modes.add_argument(
        "--execute",
        action="store_true",
        help="Verify the exact preexecution contract, then execute the formal Weight-AE runner.",
    )
    value.add_argument("--panel-cache-dir", type=Path, default=DEFAULT_PANEL_CACHE)
    value.add_argument("--builder-path", type=Path, default=DEFAULT_BUILDER)
    value.add_argument("--analyzer-path", type=Path, default=DEFAULT_ANALYZER)
    value.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--no-amp", action="store_true")
    value.add_argument("--log-every-batches", type=int, default=1)
    value.add_argument(
        "--audit-report",
        type=Path,
        default=None,
        help="Optional fresh JSON path for the default audit-only report.",
    )
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Legacy source-compatible tensor hash used by code/prediction manifests."""
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def weight_shape_bytes_sha256(tensor: torch.Tensor) -> str:
    """Frozen unseen-weight hash: little-endian shape prefix, NUL, FP32 bytes."""
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {tuple(value.shape)}")
    array = value.numpy().astype("<f4", copy=False)
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", int(value.shape[0]), int(value.shape[1])))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def index_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.int64).contiguous()
    array = value.numpy().astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def matrix_key(depth: int, role: str) -> str:
    return f"depth={depth:02d}|role={role}"


def expected_matrix_keys() -> tuple[str, ...]:
    return tuple(matrix_key(depth, role) for depth in range(12) for role in ROLES)


def require_keys(mapping: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise RuntimeError(f"{label} is missing required keys: {missing}")


def require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{label} is not a lowercase SHA-256: {value!r}")
    return text


def require_exact_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected_sha256} ({resolved})")
    return {"path": str(resolved), "sha256": actual, "bytes": resolved.stat().st_size}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    columns = list(fieldnames) if fieldnames is not None else list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
    os.replace(temporary, destination)


def runtime_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in (
        "transformers",
        "safetensors",
        "Pillow",
        "datasets",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "omegaconf",
        "hydra-core",
        "matplotlib",
        "scipy",
    ):
        packages[distribution] = importlib.metadata.version(distribution)
    uname = os.uname()
    result = {
        "python": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "platform": {
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
        **packages,
    }
    mismatches = {
        key: {"actual": result.get(key), "expected": expected}
        for key, expected in EXPECTED_RUNTIME.items()
        if result.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen runtime mismatch: {mismatches}")
    return result


def immutable_resource_snapshot() -> dict[str, dict[str, Any]]:
    resources: dict[str, dict[str, Any]] = {
        "design": require_exact_file(DESIGN_PATH, DESIGN_SHA256, "frozen design"),
        "ae_checkpoint": require_exact_file(AE_CHECKPOINT, AE_CHECKPOINT_SHA256, "canonical deterministic AE"),
        "source_condition_templates": require_exact_file(
            SOURCE_TEMPLATE, SOURCE_TEMPLATE_SHA256, "source condition templates"
        ),
        "source_role_gains": require_exact_file(SOURCE_GAINS, SOURCE_GAINS_SHA256, "source role gains"),
        "source_scout_manifest": require_exact_file(
            SOURCE_SCOUT_MANIFEST, SOURCE_SCOUT_MANIFEST_SHA256, "source unseen-weight scout manifest"
        ),
        "source_factorial_code_cache_seed_26081601": require_exact_file(
            SOURCE_FACTORIAL_CACHE,
            SOURCE_FACTORIAL_CACHE_SHA256,
            "sealed source factorial code cache seed 26081601",
        ),
        "source_factorial_code_cache_seed_26081602": require_exact_file(
            SOURCE_FACTORIAL_CACHE_SECONDARY,
            SOURCE_FACTORIAL_CACHE_SECONDARY_SHA256,
            "sealed source factorial code cache seed 26081602",
        ),
    }
    for name, expected_sha in SOURCE_PARENT_HASHES.items():
        resources[f"source_parent/{name}"] = require_exact_file(
            SOURCE_PARENT / name,
            expected_sha,
            f"sealed source parent {name}",
        )
    return resources


def script_snapshot(builder_path: Path, analyzer_path: Path) -> dict[str, dict[str, Any]]:
    runner = Path(__file__).resolve(strict=True)
    builder = builder_path.resolve(strict=True)
    analyzer = analyzer_path.resolve(strict=True)
    return {
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "builder": {"path": str(builder), "sha256": sha256_file(builder)},
        "analyzer": {"path": str(analyzer), "sha256": sha256_file(analyzer)},
    }


def dependency_snapshot() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    paths = set(SOURCE_HELPERS)
    for root in DEPENDENCY_TREES:
        resolved_root = root.resolve(strict=True)
        paths.update(path for path in resolved_root.rglob("*.py") if path.is_file())

    # Importing a child module executes every package-parent ``__init__.py``.
    # Include those initializers even for the exact (non-recursive) helpers,
    # then fail closed if the expansion was not complete.
    workspace_root = WORKSPACE_ROOT.resolve(strict=True)
    seeds = tuple(paths)
    required_initializers: set[Path] = set()
    for seed in seeds:
        parent = seed.resolve(strict=True).parent
        while parent != workspace_root:
            if workspace_root not in parent.parents:
                raise RuntimeError(f"dependency escaped workspace root: {seed}")
            initializer = parent / "__init__.py"
            if initializer.is_file():
                required_initializers.add(initializer.resolve(strict=True))
            parent = parent.parent
    paths.update(required_initializers)

    resolved_paths = {path.resolve(strict=True) for path in paths}
    required_exact = {
        path.resolve(strict=True) for path in (*SOURCE_HELPERS, *REQUIRED_DEPENDENCY_FILES)
    }
    missing_exact = sorted(str(path) for path in required_exact - resolved_paths)
    if missing_exact:
        raise RuntimeError(f"required local dependencies are absent from closure: {missing_exact}")
    missing_initializers = sorted(str(path) for path in required_initializers - resolved_paths)
    if missing_initializers:
        raise RuntimeError(
            f"package-parent initializers are absent from dependency closure: {missing_initializers}"
        )
    if len(resolved_paths) < 150:
        raise RuntimeError(f"recursive model dependency closure is suspiciously small: {len(paths)}")
    for resolved in sorted(resolved_paths, key=lambda item: str(item)):
        relative = str(resolved.relative_to(workspace_root))
        result[relative] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    return result


def assert_loaded_local_modules_frozen(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fail if project imports execute a local file outside the frozen contract."""

    workspace_root = WORKSPACE_ROOT.resolve(strict=True)
    allowed: set[Path] = set()
    for record in contract["model_dependencies"].values():
        allowed.add(Path(record["path"]).resolve(strict=True))
    for record in contract["scripts"].values():
        allowed.add(Path(record["path"]).resolve(strict=True))

    loaded: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        unresolved_path = Path(raw_path)
        module_path = unresolved_path.resolve(strict=False)
        try:
            module_path.relative_to(workspace_root)
        except ValueError:
            continue
        # A loaded module claiming to be local must resolve to a real file.
        module_path = unresolved_path.resolve(strict=True)
        if module_path.suffix in {".pyc", ".pyo"}:
            try:
                source_path = Path(importlib.util.source_from_cache(str(module_path))).resolve(strict=True)
            except (ValueError, FileNotFoundError):
                source_path = module_path
            module_path = source_path
        try:
            relative = module_path.relative_to(workspace_root)
        except ValueError:
            continue
        record = {
            "module": module_name,
            "path": str(module_path),
            "relative_path": str(relative),
        }
        loaded.append(record)
        if module_path not in allowed:
            missing.append(record)
    if missing:
        raise RuntimeError(
            "loaded local modules escaped the frozen dependency closure: "
            + json.dumps(missing, sort_keys=True)
        )
    return {
        "pass": True,
        "allowed_file_count": len(allowed),
        "loaded_module_count": len(loaded),
        "loaded_modules": loaded,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_embedded_file_records(value: Any) -> Any:
    """Keep declared paths for audit without creating ambiguous file records.

    Contract-local file records always have canonical ``path,sha256,bytes``.
    Embedded source manifests often have only ``path,sha256``; renaming their
    path field avoids having the analyzer mistake a declaration for a verified
    local-file record while preserving the complete declaration.
    """
    if isinstance(value, Mapping):
        result = {str(key): sanitize_embedded_file_records(child) for key, child in value.items()}
        if "path" in result and "sha256" in result and "bytes" not in result:
            result["declared_path"] = result.pop("path")
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_embedded_file_records(child) for child in value]
    return value


def validate_builder_artifact_manifest(
    cache_root: Path,
    *,
    builder_sha256: str,
) -> dict[str, Any]:
    manifest_path = (cache_root / "artifact_manifest.json").resolve(strict=True)
    payload = load_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise RuntimeError("builder artifact_manifest.json is not a mapping")
    expected_header = {
        "schema_version": "prospective_panel_artifact_manifest_v1",
        "design_sha256": DESIGN_SHA256,
        "builder_sha256": builder_sha256,
    }
    for field, expected in expected_header.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"builder artifact manifest {field} mismatch: {payload.get(field)!r} != {expected!r}"
            )
    rows = payload.get("files")
    if not isinstance(rows, list) or int(payload.get("file_count", -1)) != len(rows):
        raise RuntimeError("builder artifact manifest files/file_count mismatch")
    declared: set[str] = set()
    audited_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise RuntimeError(f"invalid builder artifact manifest row {index}: {row}")
        relative_text = str(row["path"])
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or relative_text != relative.as_posix()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or any("staging" in part.casefold() or part.casefold().startswith(".tmp") for part in relative.parts)
            or _forbidden_path_text(relative_text)
        ):
            raise RuntimeError(f"unsafe/noncanonical builder artifact path: {relative_text!r}")
        if relative_text == "artifact_manifest.json" or relative_text in declared:
            raise RuntimeError(f"recursive/duplicate builder artifact path: {relative_text}")
        declared.add(relative_text)
        declared_path = cache_root / relative
        if declared_path.is_symlink():
            raise RuntimeError(f"builder artifact manifest points to a symlink: {declared_path}")
        path = declared_path.resolve(strict=True)
        if not path.is_relative_to(cache_root) or not path.is_file():
            raise RuntimeError(f"builder artifact escapes/is not a regular final-root file: {path}")
        expected_sha = require_sha256(row["sha256"], f"builder artifact {relative_text}.sha256")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual_sha != expected_sha or actual_bytes != int(row["bytes"]):
            raise RuntimeError(
                f"builder artifact mismatch {relative_text}: sha={actual_sha} bytes={actual_bytes}"
            )
        audited_rows.append(
            {"declared_path": relative_text, "sha256": actual_sha, "bytes": actual_bytes}
        )
    actual: set[str] = set()
    for path in cache_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink prohibited inside committed panel cache: {path}")
        if path.is_file() and path != manifest_path:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(cache_root):
                raise RuntimeError(f"committed panel cache member escapes root: {path} -> {resolved}")
            relative = path.relative_to(cache_root).as_posix()
            if _forbidden_path_text(relative) or "staging" in relative.casefold():
                raise RuntimeError(f"forbidden/staging member in committed panel cache: {relative}")
            actual.add(relative)
    if actual != declared:
        raise RuntimeError(
            f"builder artifact manifest completeness mismatch: "
            f"missing={sorted(actual - declared)[:10]} stale={sorted(declared - actual)[:10]}"
        )
    return {
        "manifest_file": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "schema_version": payload["schema_version"],
        "file_count": len(rows),
        "complete_exact_root_membership": True,
        "no_symlinks_or_staging_members": True,
        "audited_rows": audited_rows,
    }


def _validate_evidence_files(
    evidence: Any,
    *,
    cache_root: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(evidence, Mapping) or not evidence:
        raise RuntimeError(f"{label} must be a nonempty mapping")
    checked: dict[str, dict[str, Any]] = {}
    for name, entry in sorted(evidence.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise RuntimeError(f"{label} has invalid entry: {name!r}")
        require_keys(entry, {"path", "sha256"}, f"{label}[{name}]")
        raw_path = str(entry["path"])
        path = Path(raw_path)
        if _forbidden_path_text(raw_path):
            raise RuntimeError(f"{label}[{name}] contains a forbidden target path: {raw_path}")
        if any("staging" in part.lower() or part.lower().startswith(".tmp") for part in path.parts):
            raise RuntimeError(f"{label}[{name}] contains a staging/temp path remnant: {raw_path}")
        if path.is_absolute():
            resolved = path.resolve(strict=True)
            if str(resolved) != raw_path:
                raise RuntimeError(
                    f"{label}[{name}] absolute path is not its canonical final realpath: {raw_path} -> {resolved}"
                )
            path_kind = "canonical_absolute"
        else:
            if (
                not raw_path
                or raw_path != path.as_posix()
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise RuntimeError(f"{label}[{name}] path is not a normalized relative path: {raw_path!r}")
            resolved = (cache_root / path).resolve(strict=True)
            path_kind = "normalized_relative"
        if not resolved.is_relative_to(cache_root):
            raise RuntimeError(f"{label}[{name}] escapes panel cache: {resolved}")
        expected = require_sha256(entry["sha256"], f"{label}[{name}].sha256")
        actual = sha256_file(resolved)
        if actual != expected:
            raise RuntimeError(f"{label}[{name}] hash mismatch: {actual} != {expected}")
        checked[name] = {
            "declared_path": raw_path,
            "path_kind": path_kind,
            "path": str(resolved),
            "sha256": actual,
            "bytes": resolved.stat().st_size,
        }
    return checked


def _validate_panel_payload(
    payload: Any,
    *,
    panel_id: str,
    panel_path: Path,
    panel_sha256: str,
    cache_root: Path,
    builder_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{panel_id} panel.pt is not a mapping")
    require_keys(payload, REQUIRED_PANEL_KEYS, f"{panel_id} panel")
    if payload["schema_version"] != "prospective_panel_v1":
        raise RuntimeError(f"{panel_id} panel schema mismatch: {payload['schema_version']!r}")
    if payload["panel_id"] != panel_id:
        raise RuntimeError(f"{panel_id} payload panel_id mismatch: {payload['panel_id']!r}")
    if payload["checkpoint_sha256"] != PANEL_CHECKPOINT_SHA256[panel_id]:
        raise RuntimeError(f"{panel_id} checkpoint SHA mismatch in panel payload")
    if payload["quality_status"] != "PASS":
        raise RuntimeError(f"{panel_id} quality status is not PASS: {payload['quality_status']!r}")

    weights = payload["weights"]
    activations = payload["activations"]
    metadata = payload["matrix_meta"]
    if not isinstance(weights, Mapping) or not isinstance(activations, Mapping) or not isinstance(metadata, Mapping):
        raise RuntimeError(f"{panel_id} weights/activations/matrix_meta must be mappings")
    expected_keys = set(expected_matrix_keys())
    if set(weights) != expected_keys or set(metadata) != expected_keys:
        raise RuntimeError(
            f"{panel_id} matrix grid mismatch: weights={len(weights)} meta={len(metadata)} expected=72"
        )
    if set(activations) != set(SCORE_SPLITS):
        raise RuntimeError(f"{panel_id} activation split keys mismatch: {sorted(activations)}")
    for split in SCORE_SPLITS:
        if not isinstance(activations[split], Mapping) or set(activations[split]) != expected_keys:
            raise RuntimeError(f"{panel_id}/{split} activation matrix grid mismatch")

    matrix_audit: dict[str, dict[str, Any]] = {}
    for depth in range(12):
        for role in ROLES:
            key = matrix_key(depth, role)
            W = weights[key]
            if not isinstance(W, torch.Tensor):
                raise RuntimeError(f"{panel_id}/{key} weight is not a tensor")
            expected_shape = ROLE_SHAPES[role]
            if W.device.type != "cpu" or W.dtype != torch.float32 or W.ndim != 2 or tuple(W.shape) != expected_shape:
                raise RuntimeError(
                    f"{panel_id}/{key} invalid W: device={W.device} dtype={W.dtype} shape={tuple(W.shape)}"
                )
            if not bool(torch.isfinite(W).all()):
                raise RuntimeError(f"{panel_id}/{key} W is non-finite")
            meta = metadata[key]
            if not isinstance(meta, Mapping):
                raise RuntimeError(f"{panel_id}/{key} metadata is not a mapping")
            require_keys(
                meta,
                {"depth", "role", "d_in", "d_out", "module_name", "weight_shape_bytes_sha256"},
                f"{panel_id}/{key} metadata",
            )
            expected_meta = {
                "depth": depth,
                "role": role,
                "d_in": expected_shape[0],
                "d_out": expected_shape[1],
            }
            for name, expected_value in expected_meta.items():
                if meta[name] != expected_value:
                    raise RuntimeError(
                        f"{panel_id}/{key} metadata {name} mismatch: {meta[name]!r} != {expected_value!r}"
                    )
            if not isinstance(meta["module_name"], str) or not meta["module_name"]:
                raise RuntimeError(f"{panel_id}/{key} module_name is empty")
            shape_hash = weight_shape_bytes_sha256(W)
            if meta["weight_shape_bytes_sha256"] != shape_hash:
                raise RuntimeError(
                    f"{panel_id}/{key} shape+bytes hash mismatch: "
                    f"{meta['weight_shape_bytes_sha256']} != {shape_hash}"
                )
            split_hashes: dict[str, str] = {}
            split_nonzero: dict[str, int] = {}
            split_std: dict[str, float] = {}
            for split in SCORE_SPLITS:
                X = activations[split][key]
                expected_x_shape = (ACTIVATION_ROWS[panel_id], expected_shape[0])
                if (
                    not isinstance(X, torch.Tensor)
                    or X.device.type != "cpu"
                    or X.dtype != torch.float32
                    or X.ndim != 2
                    or tuple(X.shape) != expected_x_shape
                ):
                    raise RuntimeError(
                        f"{panel_id}/{split}/{key} invalid X: "
                        f"type={type(X)} device={getattr(X, 'device', None)} "
                        f"dtype={getattr(X, 'dtype', None)} shape={getattr(X, 'shape', None)}"
                    )
                if not bool(torch.isfinite(X).all()):
                    raise RuntimeError(f"{panel_id}/{split}/{key} X is non-finite")
                nonzero = int(torch.count_nonzero(X).item())
                standard_deviation = float(X.std(unbiased=False).item())
                if nonzero <= 0 or not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
                    raise RuntimeError(
                        f"{panel_id}/{split}/{key} X is zero/degenerate: nonzero={nonzero} std={standard_deviation}"
                    )
                split_hashes[split] = tensor_sha256(X)
                split_nonzero[split] = nonzero
                split_std[split] = standard_deviation
            if split_hashes["A"] == split_hashes["B"] or torch.equal(
                activations["A"][key], activations["B"][key]
            ):
                raise RuntimeError(f"{panel_id}/{key} A/B activation tensors are not distinct")
            matrix_audit[key] = {
                "shape": list(expected_shape),
                "module_name": meta["module_name"],
                "weight_shape_bytes_sha256": shape_hash,
                "weight_tensor_sha256": tensor_sha256(W),
                "activation_tensor_sha256": split_hashes,
                "activation_nonzero_count": split_nonzero,
                "activation_population_std": split_std,
                "extra_matrix_meta_keys": sorted(set(meta) - {
                    "depth", "role", "d_in", "d_out", "module_name", "weight_shape_bytes_sha256"
                }),
            }

    qkv_equality: dict[str, dict[str, bool]] = {}
    for depth in range(12):
        depth_checks: dict[str, bool] = {}
        for split in SCORE_SPLITS:
            query = activations[split][matrix_key(depth, "attn_query")]
            key_tensor = activations[split][matrix_key(depth, "attn_key")]
            value = activations[split][matrix_key(depth, "attn_value")]
            equal = bool(torch.equal(query, key_tensor) and torch.equal(query, value))
            if not equal:
                raise RuntimeError(f"{panel_id}/{split}/depth={depth:02d} Q/K/V input activations differ")
            depth_checks[split] = equal
        qkv_equality[f"depth={depth:02d}"] = depth_checks

    resources = payload["resource_hashes"]
    if not isinstance(resources, Mapping) or not resources:
        raise RuntimeError(f"{panel_id} resource_hashes must be a nonempty mapping")
    normalized_resources: dict[str, str] = {}
    for name, digest in sorted(resources.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str):
            raise RuntimeError(f"{panel_id} resource_hash key is not a string")
        normalized_resources[name] = require_sha256(digest, f"{panel_id}.resource_hashes[{name}]")
    expected_resources = PANEL_RESOURCE_SPECS[panel_id]
    missing_resources = sorted(set(expected_resources) - set(normalized_resources))
    if missing_resources:
        raise RuntimeError(f"{panel_id} resource_hashes lacks frozen resources: {missing_resources}")
    verified_resource_files: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha) in expected_resources.items():
        if normalized_resources[name] != expected_sha:
            raise RuntimeError(
                f"{panel_id} declared resource hash mismatch for {name}: "
                f"{normalized_resources[name]} != {expected_sha}"
            )
        verified_resource_files[name] = require_exact_file(
            path,
            expected_sha,
            f"{panel_id} frozen resource {name}",
        )

    build = payload["build_manifest"]
    if not isinstance(build, Mapping):
        raise RuntimeError(f"{panel_id} build_manifest is not a mapping")
    require_keys(build, REQUIRED_BUILD_MANIFEST_KEYS, f"{panel_id} build_manifest")
    require_keys(
        build,
        {"schema_version", "design_sha256", "builder_sha256", "checkpoint_sha256", "seal"},
        f"{panel_id} build_manifest",
    )
    if build["schema_version"] != "prospective_panel_build_manifest_v1":
        raise RuntimeError(f"{panel_id} build_manifest schema mismatch: {build['schema_version']!r}")
    direct_hash_contract = {
        "design_sha256": DESIGN_SHA256,
        "builder_sha256": builder_sha256,
        "checkpoint_sha256": PANEL_CHECKPOINT_SHA256[panel_id],
    }
    for name, expected_value in direct_hash_contract.items():
        if build[name] != expected_value:
            raise RuntimeError(
                f"{panel_id} build_manifest.{name} mismatch: {build[name]!r} != {expected_value!r}"
            )
    builder_seal = build["seal"]
    if not isinstance(builder_seal, Mapping):
        raise RuntimeError(f"{panel_id} build_manifest.seal is not a mapping")
    required_builder_seal = {
        "installed": True,
        "forbidden_path_markers": ["data2vec", "target path component"],
        "blocked_path_events": [],
        "blocked_import_events": [],
        "forbidden_loaded_modules": [],
        "network_connection_events": [],
        "subprocess_events": [],
        "target_access_event_count": 0,
        "network_connection_count": 0,
        "subprocess_count": 0,
        "pass": True,
    }
    if any(builder_seal.get(key) != expected for key, expected in required_builder_seal.items()):
        raise RuntimeError(f"{panel_id} builder seal is not an exact clean source-only seal: {builder_seal}")
    if not isinstance(builder_seal.get("hidden_directory_entries"), list):
        raise RuntimeError(f"{panel_id} builder seal lacks hidden_directory_entries audit list")
    exact_values = {
        "panel_id": panel_id,
        "panel_valid": True,
        "seal_pass": True,
        "provenance_pass": True,
        "geometry_pass": True,
        "quality_gate_pass": True,
        "unseen_weight_pass": True,
        "activation_gate_pass": True,
        "matrix_count": 72,
        "exact_overlap_count": 0,
        "training_bank_compatible_count": 471,
        "score_splits": ["A", "B"],
    }
    for name, expected_value in exact_values.items():
        if build[name] != expected_value:
            raise RuntimeError(
                f"{panel_id} build_manifest.{name} mismatch: {build[name]!r} != {expected_value!r}"
            )
    evidence = _validate_evidence_files(
        build["evidence_files"], cache_root=cache_root, label=f"{panel_id}.build_manifest.evidence_files"
    )
    missing_evidence = sorted(REQUIRED_PANEL_EVIDENCE_LABELS - set(evidence))
    if missing_evidence:
        raise RuntimeError(f"{panel_id} evidence_files lacks required labels: {missing_evidence}")
    bank_hash_path = Path(evidence["training_bank_compatible_weight_hashes"]["path"])
    with bank_hash_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        bank_hash_rows = list(reader)
        observed_columns = list(reader.fieldnames or [])
    expected_bank_columns = [
        "source_key",
        "model_name",
        "layer_name",
        "d_in",
        "d_out",
        "weight_path",
        "weight_file_bytes",
        "declared_weight_size_bytes",
        "weight_shape_bytes_sha256",
    ]
    if observed_columns != expected_bank_columns or len(bank_hash_rows) != 471:
        raise RuntimeError(
            f"{panel_id} training-bank hash manifest structure mismatch: "
            f"columns={observed_columns} rows={len(bank_hash_rows)}"
        )
    if len({row["source_key"] for row in bank_hash_rows}) != 471:
        raise RuntimeError(f"{panel_id} training-bank hash manifest has duplicate source keys")
    for row in bank_hash_rows:
        shape = (int(row["d_in"]), int(row["d_out"]))
        relative_weight = Path(row["weight_path"])
        if (
            shape not in set(ROLE_SHAPES.values())
            or relative_weight.is_absolute()
            or ".." in relative_weight.parts
            or _forbidden_path_text(row["weight_path"])
            or int(row["weight_file_bytes"]) <= 0
            or int(row["weight_file_bytes"]) != int(row["declared_weight_size_bytes"])
        ):
            raise RuntimeError(f"{panel_id} malformed training-bank hash row: {row}")
        require_sha256(
            row["weight_shape_bytes_sha256"],
            f"{panel_id} training-bank row {row['source_key']} weight_shape_bytes_sha256",
        )
    resource_evidence = load_json(Path(evidence["resource_hashes"]["path"]))
    if not isinstance(resource_evidence, Mapping):
        raise RuntimeError(f"{panel_id} resource_hashes evidence is not a mapping")
    evidence_resource_files: dict[str, dict[str, Any]] = {}
    for name, declared_sha in normalized_resources.items():
        record = resource_evidence.get(name)
        if not isinstance(record, Mapping) or record.get("sha256") != declared_sha:
            raise RuntimeError(
                f"{panel_id} resource evidence/declaration mismatch for {name}: {record} / {declared_sha}"
            )
        require_keys(
            record,
            {"realpath", "bytes", "sha256", "expected_sha256", "pass"},
            f"{panel_id} resource evidence {name}",
        )
        if record["expected_sha256"] != declared_sha or record["pass"] is not True:
            raise RuntimeError(f"{panel_id} resource evidence failed expected/pass fields for {name}")
        resource_path = Path(str(record["realpath"]))
        if not resource_path.is_absolute():
            raise RuntimeError(f"{panel_id} resource evidence realpath is not absolute for {name}")
        resolved_resource = resource_path.resolve(strict=True)
        if str(resolved_resource) != str(resource_path):
            raise RuntimeError(f"{panel_id} resource evidence realpath is not canonical for {name}")
        actual_resource = require_exact_file(
            resolved_resource,
            declared_sha,
            f"{panel_id} evidence-bound resource {name}",
        )
        if actual_resource["bytes"] != int(record["bytes"]):
            raise RuntimeError(f"{panel_id} resource evidence byte count mismatch for {name}")
        evidence_resource_files[name] = actual_resource
    return {
        "panel_id": panel_id,
        "path": str(panel_path),
        "sha256": panel_sha256,
        "bytes": panel_path.stat().st_size,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "quality_status": payload["quality_status"],
        "schema_version": payload["schema_version"],
        "matrix_count": len(weights),
        "activation_rows": ACTIVATION_ROWS[panel_id],
        "resource_hashes": normalized_resources,
        "verified_resource_files": verified_resource_files,
        "evidence_resource_files": evidence_resource_files,
        "build_manifest": sanitize_embedded_file_records(build),
        "evidence_files_verified": evidence,
        "training_bank_hash_manifest_audit": {
            "path": str(bank_hash_path.resolve(strict=True)),
            "sha256": sha256_file(bank_hash_path),
            "bytes": bank_hash_path.stat().st_size,
            "row_count": 471,
            "unique_source_keys": 471,
            "columns": expected_bank_columns,
            "pass": True,
        },
        "matrix_audit": matrix_audit,
        "qkv_activation_input_bit_exact": qkv_equality,
        "extra_top_level_keys": sorted(set(payload) - REQUIRED_PANEL_KEYS),
        "provenance_pass": True,
        "resource_hash_pass": True,
        "quality_pass": True,
        "validity_pass": True,
    }


def validate_panel_cache(panel_cache_dir: Path, builder_path: Path) -> dict[str, Any]:
    cache_root = panel_cache_dir.resolve(strict=True)
    if not cache_root.is_relative_to(ARTIFACT_ROOT.resolve(strict=True)):
        raise RuntimeError(f"panel cache must be under {ARTIFACT_ROOT}: {cache_root}")
    manifest_path = (cache_root / "panel_manifest.json").resolve(strict=True)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("panel_manifest.json is not a mapping")
    require_keys(
        manifest,
        {"schema_version", "design_sha256", "builder_sha256", "created_utc", "panels"},
        "panel manifest",
    )
    if manifest["schema_version"] != "prospective_panel_manifest_v1":
        raise RuntimeError(f"panel root schema mismatch: {manifest['schema_version']!r}")
    if manifest["design_sha256"] != DESIGN_SHA256:
        raise RuntimeError("panel root design SHA does not match frozen design")
    builder = builder_path.resolve(strict=True)
    builder_sha = sha256_file(builder)
    if manifest["builder_sha256"] != builder_sha:
        raise RuntimeError(
            f"panel root builder SHA mismatch: {manifest['builder_sha256']} != {builder_sha}"
        )
    artifact_audit = validate_builder_artifact_manifest(
        cache_root,
        builder_sha256=builder_sha,
    )
    panel_entries = manifest["panels"]
    if not isinstance(panel_entries, Mapping) or set(panel_entries) != set(PANEL_IDS):
        raise RuntimeError(f"panel root must contain exactly {PANEL_IDS}: {list(panel_entries) if isinstance(panel_entries, Mapping) else type(panel_entries)}")

    summaries: dict[str, dict[str, Any]] = {}
    for panel_id in PANEL_IDS:
        entry = panel_entries[panel_id]
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"panel root entry {panel_id} is not a mapping")
        require_keys(
            entry,
            {"panel_id", "path", "sha256", "quality_status", "checkpoint_sha256"},
            f"panel root entry {panel_id}",
        )
        if entry["panel_id"] != panel_id or entry["quality_status"] != "PASS":
            raise RuntimeError(f"invalid root identity/quality for {panel_id}: {entry}")
        if entry["checkpoint_sha256"] != PANEL_CHECKPOINT_SHA256[panel_id]:
            raise RuntimeError(f"root checkpoint SHA mismatch for {panel_id}")
        raw_path = str(entry["path"])
        panel_path = Path(raw_path)
        if _forbidden_path_text(raw_path):
            raise RuntimeError(f"root panel path contains a forbidden target component: {raw_path}")
        if not panel_path.is_absolute():
            raise RuntimeError(f"root panel path must be absolute for {panel_id}: {raw_path}")
        resolved = panel_path.resolve(strict=True)
        if raw_path != str(resolved):
            raise RuntimeError(f"root panel path is not canonical for {panel_id}: {raw_path} -> {resolved}")
        if not resolved.is_relative_to(cache_root):
            raise RuntimeError(f"root panel path escapes cache for {panel_id}: {resolved}")
        expected_sha = require_sha256(entry["sha256"], f"panel root {panel_id}.sha256")
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            raise RuntimeError(f"panel file SHA mismatch for {panel_id}: {actual_sha} != {expected_sha}")
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
        summaries[panel_id] = _validate_panel_payload(
            payload,
            panel_id=panel_id,
            panel_path=resolved,
            panel_sha256=actual_sha,
            cache_root=cache_root,
            builder_sha256=builder_sha,
        )
        del payload
        gc.collect()
    manifest_snapshot = sanitize_embedded_file_records(manifest)
    return {
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_file": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "manifest": manifest_snapshot,
        "artifact_manifest_audit": artifact_audit,
        "root_extra_keys": sorted(
            set(manifest) - {"schema_version", "design_sha256", "builder_sha256", "created_utc", "panels"}
        ),
        "panels": summaries,
    }


def expected_counts() -> dict[str, Any]:
    tiles_per_role = {
        role: (ROLE_SHAPES[role][0] // 64) * (ROLE_SHAPES[role][1] // 64)
        for role in ROLES
    }
    tiles_per_panel_tiling = 12 * sum(tiles_per_role.values())
    result = {
        "panels": 2,
        "depths": 12,
        "roles": 6,
        "matrices_per_panel": 72,
        "tilings": 2,
        "score_splits": 2,
        "arms": 3,
        "tiles_per_role_matrix": tiles_per_role,
        "tiles_per_panel_tiling": tiles_per_panel_tiling,
        "decoded_full_predictions_per_panel": 72 * 2 * 3,
        "decoded_full_predictions_total": 2 * 72 * 2 * 3,
        "weight_sufficient_stat_rows": 2 * 72 * 2 * 3,
        "operator_sufficient_stat_rows": 2 * 2 * 72 * 2 * 3,
        "correct_latent_cache_entries": 2 * 72 * 2,
        "tiling_manifest_rows": 2 * 72 * 2,
        "permutation_manifest_rows": 2 * 2 * 12 * sum(ROLE_SHAPES[role][0] // 64 for role in ROLES),
        "primary_calibrated_error_rows": 2 * 2 * 2 * 2 * 2,
        "directional_criterion_rows": 2 * 2 * 2 * 2 * 2,
        "operator_aggregate_rows": 576,
        "weight_aggregate_rows": 288,
        "criterion1_error_rows": 32,
        "criterion2_cosine_rows": 32,
        "criterion3_role_rows": 48,
        "criterion4_did_rows": 16,
        "criterion5_radial_rows": 24,
        "criterion6_common_rows": 16,
        "criterion6_mapping_summary_rows": 16,
        "criterion6_mapping_full_rows": 11520,
        "criterion7_absolute_rows": 16,
        "criterion_cells_rows": 200,
        "geometry_rows": 432,
        "geometry_plots": 7,
    }
    if tiles_per_panel_tiling != 20736:
        raise AssertionError(f"internal tile-count derivation failed: {result}")
    return result


def execution_config(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve(strict=False)
    artifact_root = ARTIFACT_ROOT.resolve(strict=True)
    panel_cache = args.panel_cache_dir.resolve(strict=False)
    contract_path = args.contract_path.resolve(strict=False)
    if not output.is_relative_to(artifact_root) or output == artifact_root:
        raise RuntimeError(f"formal output must be a child of {ARTIFACT_ROOT}: {output}")
    if output.exists():
        raise RuntimeError(f"formal output directory must be fresh and absent: {output}")
    if output.is_relative_to(panel_cache) or panel_cache.is_relative_to(output):
        raise RuntimeError(f"formal output and panel cache must be disjoint: output={output} cache={panel_cache}")
    if contract_path.is_relative_to(output):
        raise RuntimeError(f"external preexecution contract cannot be inside the future formal output: {contract_path}")
    if args.batch_size <= 0 or args.log_every_batches <= 0:
        raise RuntimeError("batch-size and log-every-batches must be positive")
    formal_forward_contract = {
        "device": "cuda:0",
        "batch_size": 64,
        "no_amp": False,
        "log_every_batches": 1,
    }
    observed_forward_contract = {
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "no_amp": bool(args.no_amp),
        "log_every_batches": int(args.log_every_batches),
    }
    if observed_forward_contract != formal_forward_contract:
        raise RuntimeError(
            f"formal forward contract mismatch: observed={observed_forward_contract} "
            f"expected={formal_forward_contract}"
        )
    return {
        "output_dir": str(output),
        "panel_cache_dir": str(args.panel_cache_dir.resolve(strict=False)),
        "builder_path": str(args.builder_path.resolve(strict=False)),
        "analyzer_path": str(args.analyzer_path.resolve(strict=False)),
        "device": args.device,
        "batch_size": int(args.batch_size),
        "no_amp": bool(args.no_amp),
        "log_every_batches": int(args.log_every_batches),
        "global_seed": GLOBAL_SEED,
        "tiling_seeds": list(TILING_SEEDS),
        "score_splits": list(SCORE_SPLITS),
        "arms": list(ARMS),
        "context": "source_frozen_cell_mean_by_role_depth_for_both_encoder_and_decoder",
        "panel_activations_to_weight_ae": False,
        "latent_sampling": False,
        "rope_2d_coordinates": "raw_integer",
        "formal_forward_contract": formal_forward_contract,
    }


def make_contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    assert_seal_clean()
    resources = immutable_resource_snapshot()
    scripts = script_snapshot(args.builder_path, args.analyzer_path)
    dependencies = dependency_snapshot()
    panels = validate_panel_cache(args.panel_cache_dir, args.builder_path)
    if panels["manifest"]["builder_sha256"] != scripts["builder"]["sha256"]:
        raise RuntimeError("panel manifest and script snapshot disagree on builder SHA")
    assert_seal_clean()
    return {
        "schema_version": "prospective_geometry_matched_replication_preexecution_v1",
        "design_sha256": DESIGN_SHA256,
        "runner_sha256": scripts["runner"]["sha256"],
        "builder_sha256": scripts["builder"]["sha256"],
        "analyzer_sha256": scripts["analyzer"]["sha256"],
        "panel_manifest_sha256": panels["manifest_sha256"],
        "checkpoint_sha256": resources["ae_checkpoint"]["sha256"],
        "runtime": runtime_snapshot(),
        "execution_config": execution_config(args),
        "resources": resources,
        "scripts": scripts,
        "model_dependencies": dependencies,
        "panel_cache": panels,
        "expected_counts": expected_counts(),
        "dataflow_contract": {
            "panel_builder_can_import_weight_ae": False,
            "runner_executes_panel_quality_models": False,
            "panel_activations_consumers": ["operator_sufficient_statistics_only"],
            "weight_ae_inputs": ["weight_tiles", "source_frozen_cell_mean_templates"],
            "source_gains_applied_by_runner": False,
            "source_gains_applied_by_cpu_analyzer": True,
            "full_predictions_persisted": False,
            "prediction_reused_for_score_splits": True,
            "analyzer_invoked_by_runner": False,
        },
    }


def audit_only(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": "prospective_geometry_matched_replication_audit_v1",
        "created_utc": utc_now(),
        "mode": "AUDIT_ONLY_NO_WEIGHT_AE_IMPORT_OR_FORWARD",
        "status": "BLOCKED",
        "errors": [],
        "checks": {},
        "target_seal": dict(_SEAL_STATE),
    }
    checks: tuple[tuple[str, Any], ...] = (
        ("immutable_resources", immutable_resource_snapshot),
        ("scripts", lambda: script_snapshot(args.builder_path, args.analyzer_path)),
        ("model_dependencies", dependency_snapshot),
        ("runtime", runtime_snapshot),
        ("execution_config", lambda: execution_config(args)),
        ("panel_cache", lambda: validate_panel_cache(args.panel_cache_dir, args.builder_path)),
    )
    for name, function in checks:
        try:
            report["checks"][name] = {"pass": True, "value": function()}
        except Exception as error:  # preserve all independent no-forward blockers
            report["checks"][name] = {
                "pass": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            report["errors"].append(
                {"stage": name, "type": type(error).__name__, "message": str(error)}
            )
    try:
        assert_seal_clean()
        report["checks"]["target_seal_clean"] = {"pass": True}
    except Exception as error:
        report["checks"]["target_seal_clean"] = {
            "pass": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        report["errors"].append(
            {"stage": "target_seal_clean", "type": type(error).__name__, "message": str(error)}
        )
    if not report["errors"]:
        report["status"] = "READY_FOR_CONTRACT_PREPARATION"
    weight_ae_modules = sorted(
        name
        for name in sys.modules
        if name == "big_vae"
        or name.startswith("big_vae.")
        or name.startswith("source_latent_code_factorial")
        or name.startswith("source_confirmatory_g01_gate")
    )
    report["weight_ae_modules"] = weight_ae_modules
    report["weight_ae_imported"] = bool(weight_ae_modules)
    report["weight_ae_forward"] = False
    report["target_seal"] = dict(_SEAL_STATE)
    report["elapsed_seconds"] = time.monotonic() - started
    if args.audit_report is not None:
        destination = args.audit_report.resolve(strict=False)
        if destination.exists():
            raise RuntimeError(f"audit report path must be fresh: {destination}")
        atomic_write_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0, report


def prepare_contract(args: argparse.Namespace) -> None:
    contract_path = args.contract_path.resolve(strict=False)
    if contract_path.exists():
        raise RuntimeError(f"preexecution contract path must be fresh: {contract_path}")
    print("stage=immutable_input_and_panel_audit mode=PREPARE_CONTRACT no_weight_ae_import=true", flush=True)
    payload = make_contract_payload(args)
    payload["created_utc"] = utc_now()
    payload["target_seal"] = dict(_SEAL_STATE)
    assert_seal_clean()
    atomic_write_json(contract_path, payload)
    print(
        f"stage=contract_written path={contract_path} sha256={sha256_file(contract_path)} "
        "weight_ae_imported=false weight_ae_forward=false",
        flush=True,
    )


def verify_contract(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    contract_path = args.contract_path.resolve(strict=True)
    contract_sha = sha256_file(contract_path)
    contract = load_json(contract_path)
    if not isinstance(contract, Mapping) or contract.get("schema_version") != "prospective_geometry_matched_replication_preexecution_v1":
        raise RuntimeError(f"invalid preexecution contract schema: {contract_path}")
    current = make_contract_payload(args)
    for key in (
        "schema_version",
        "design_sha256",
        "runner_sha256",
        "builder_sha256",
        "analyzer_sha256",
        "panel_manifest_sha256",
        "checkpoint_sha256",
        "runtime",
        "execution_config",
        "resources",
        "scripts",
        "model_dependencies",
        "panel_cache",
        "expected_counts",
        "dataflow_contract",
    ):
        if contract.get(key) != current[key]:
            raise RuntimeError(f"preexecution contract drift at {key}")
    sealed = contract.get("target_seal")
    if not isinstance(sealed, Mapping) or not bool(sealed.get("installed_before_input_read")):
        raise RuntimeError("preexecution contract lacks the source-only seal")
    for key in ("target_access_events", "network_connections", "subprocess_launches"):
        if int(sealed.get(key, -1)) != 0:
            raise RuntimeError(f"preexecution contract recorded nonzero {key}")
    assert_seal_clean()
    return dict(contract), contract_sha


@dataclass(frozen=True)
class PanelHandle:
    panel_id: str
    path: Path
    sha256: str
    checkpoint_sha256: str


def panel_handles(contract: Mapping[str, Any]) -> list[PanelHandle]:
    values = contract["panel_cache"]["panels"]
    return [
        PanelHandle(
            panel_id=panel_id,
            path=Path(values[panel_id]["path"]),
            sha256=values[panel_id]["sha256"],
            checkpoint_sha256=values[panel_id]["checkpoint_sha256"],
        )
        for panel_id in PANEL_IDS
    ]


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("prospective_geometry_matched_replication")
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


def validate_templates(templates: Any) -> dict[str, Any]:
    if not isinstance(templates, Mapping) or "cell_mean" not in templates:
        raise RuntimeError("source template payload lacks cell_mean")
    cells = templates["cell_mean"]
    expected = {(role, depth) for role in ROLES for depth in range(12)}
    if not isinstance(cells, Mapping) or set(cells) != expected:
        raise RuntimeError(f"source cell_mean grid mismatch: observed={len(cells) if isinstance(cells, Mapping) else type(cells)}")
    summary: dict[str, Any] = {}
    for role in ROLES:
        for depth in range(12):
            key = (role, depth)
            template = cells[key]
            if not isinstance(template, Mapping) or set(template) < {"c_var", "c_patch"}:
                raise RuntimeError(f"invalid source template for {key}")
            entry: dict[str, Any] = {}
            for name in ("c_var", "c_patch"):
                tensor = template[name]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != (256,)
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise RuntimeError(f"invalid {name} for source template {key}")
                entry[f"{name}_sha256"] = tensor_sha256(tensor)
            summary[matrix_key(depth, role)] = entry
    return summary


def make_tiling(gate: Any, W: torch.Tensor, *, panel: PanelHandle, depth: int, role: str, tiling_seed: int) -> Any:
    identity = f"{panel.panel_id}|{panel.checkpoint_sha256}|depth={depth:02d}|role={role}"
    generator = torch.Generator(device="cpu")
    generator.manual_seed(gate.stable_seed(tiling_seed, identity, role, depth))
    patch_groups = gate.make_partition(int(W.shape[0]) // 16, 4, generator)
    offsets = torch.arange(16, dtype=torch.int64)
    rows = tuple(
        (group.to(torch.int64)[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values
        for group in patch_groups
    )
    cols = tuple(group.to(torch.int64) for group in gate.make_partition(int(W.shape[1]), 64, generator))
    return gate.Tiling(seed=tiling_seed, rows=rows, cols=cols)


def validate_tiling(gate: Any, W: torch.Tensor, tiling: Any, *, label: str) -> dict[str, Any]:
    tiles = gate.split_tiles(W, tiling)
    rebuilt, coverage = gate.reassemble(tiles, tiling, tuple(W.shape))
    coordinates = torch.arange(W.numel(), dtype=torch.int64).reshape(W.shape)
    coordinate_rebuilt, coordinate_coverage = gate.reassemble(
        gate.split_tiles(coordinates, tiling), tiling, tuple(W.shape)
    )
    if not torch.equal(rebuilt, W):
        raise RuntimeError(f"W tiling reassembly is not bit exact: {label}")
    if not torch.equal(coordinate_rebuilt, coordinates):
        raise RuntimeError(f"coordinate tiling reassembly is not bit exact: {label}")
    if not bool(torch.all(coverage == 1)) or not bool(torch.all(coordinate_coverage == 1)):
        raise RuntimeError(f"tiling coverage is not exactly one: {label}")
    row_tensor = torch.stack(tiling.rows).to(torch.int64).contiguous()
    col_tensor = torch.stack(tiling.cols).to(torch.int64).contiguous()
    row_sha = index_tensor_sha256(row_tensor)
    col_sha = index_tensor_sha256(col_tensor)
    partition_sha = hashlib.sha256(f"{row_sha}|{col_sha}".encode("utf-8")).hexdigest()
    return {
        "row_tensor": row_tensor,
        "col_tensor": col_tensor,
        "row_index_sha256": row_sha,
        "column_index_sha256": col_sha,
        "partition_sha256": partition_sha,
        "num_tiles": int(tiling.num_tiles),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "weight_reassembly_bit_exact": True,
        "coordinate_reassembly_bit_exact": True,
    }


def derangement_for_row(
    *,
    panel_id: str,
    tiling_seed: int,
    depth: int,
    role: str,
    row_group: int,
    n_col_groups: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if n_col_groups <= 1:
        raise RuntimeError("within-row derangement needs at least two column groups")
    payload = {
        "depth": int(depth),
        "namespace": "within_row_code_derangement_v1",
        "panel": panel_id,
        "role": role,
        "row_group": int(row_group),
        "tiling_seed": int(tiling_seed),
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    offset = 1 + value % (n_col_groups - 1)
    mapping = (torch.arange(n_col_groups, dtype=torch.int64) + offset) % n_col_groups
    fixed_points = int(torch.count_nonzero(mapping == torch.arange(n_col_groups)).item())
    if fixed_points != 0 or not torch.equal(mapping.sort().values, torch.arange(n_col_groups)):
        raise RuntimeError(f"invalid within-row derangement: {payload}")
    return mapping, {
        **payload,
        "n_col_groups": n_col_groups,
        "offset": int(offset),
        "fixed_points": fixed_points,
        "bijection_pass": True,
        "mapping_sha256": index_tensor_sha256(mapping),
    }


def permute_codes_within_rows(
    correct: torch.Tensor,
    *,
    panel_id: str,
    tiling_seed: int,
    depth: int,
    role: str,
    n_row_groups: int,
    n_col_groups: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if tuple(correct.shape[1:]) != (512,) or int(correct.shape[0]) != n_row_groups * n_col_groups:
        raise RuntimeError(f"correct code grid mismatch for {panel_id}/{depth}/{role}: {tuple(correct.shape)}")
    result = torch.empty_like(correct)
    rows: list[dict[str, Any]] = []
    for row_group in range(n_row_groups):
        start = row_group * n_col_groups
        end = start + n_col_groups
        source, manifest = derangement_for_row(
            panel_id=panel_id,
            tiling_seed=tiling_seed,
            depth=depth,
            role=role,
            row_group=row_group,
            n_col_groups=n_col_groups,
        )
        block = correct[start:end]
        permuted = block[source]
        if not torch.equal(permuted[source.argsort()], block):
            raise RuntimeError(f"code multiset was not preserved for {panel_id}/{depth}/{role}/row={row_group}")
        result[start:end] = permuted
        rows.append(
            {
                **manifest,
                "matrix_key": matrix_key(depth, role),
                "correct_code_block_sha256": tensor_sha256(block),
                "permuted_code_block_sha256": tensor_sha256(permuted),
                "same_row_code_multiset_bit_exact": True,
            }
        )
    return result, rows


def encode_codes(
    *,
    factorial: Any,
    gate: Any,
    model: torch.nn.Module,
    tiles: torch.Tensor,
    template: Mapping[str, torch.Tensor],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    logger: logging.Logger,
    label: str,
    log_every: int,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    started = time.monotonic()
    total_batches = math.ceil(int(tiles.shape[0]) / batch_size)
    for batch_index, start in enumerate(range(0, int(tiles.shape[0]), batch_size), start=1):
        stop = min(start + batch_size, int(tiles.shape[0]))
        W_batch = tiles[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
        condition = factorial.fixed_condition(
            template,
            batch=stop - start,
            device=device,
            dtype=W_batch.dtype,
        )
        with gate._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
            codes = factorial.encode_z_dec(model, W_batch, condition)
        codes = codes.detach().cpu().to(torch.float32).contiguous()
        if tuple(codes.shape) != (stop - start, 512) or not bool(torch.isfinite(codes).all()):
            raise RuntimeError(f"invalid z_dec batch for {label}: {tuple(codes.shape)}")
        batches.append(codes)
        if batch_index % log_every == 0 or batch_index == total_batches:
            elapsed = time.monotonic() - started
            logger.info(
                "stage=encode label=%s batch=%d/%d tiles=%d/%d rate=%.2f_tiles_s elapsed=%.1fs",
                label,
                batch_index,
                total_batches,
                stop,
                int(tiles.shape[0]),
                stop / max(elapsed, 1e-9),
                elapsed,
            )
    result = torch.cat(batches, dim=0)
    if tuple(result.shape) != (int(tiles.shape[0]), 512):
        raise RuntimeError(f"assembled z_dec shape mismatch for {label}: {tuple(result.shape)}")
    return result


def decode_codes(
    *,
    factorial: Any,
    gate: Any,
    model: torch.nn.Module,
    codes: torch.Tensor,
    template: Mapping[str, torch.Tensor],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    logger: logging.Logger,
    label: str,
    log_every: int,
) -> torch.Tensor:
    decoded: list[torch.Tensor] = []
    started = time.monotonic()
    total_batches = math.ceil(int(codes.shape[0]) / batch_size)
    for batch_index, start in enumerate(range(0, int(codes.shape[0]), batch_size), start=1):
        stop = min(start + batch_size, int(codes.shape[0]))
        z_batch = codes[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
        _c_var, c_patch, _c_pooled = factorial.fixed_condition(
            template,
            batch=stop - start,
            device=device,
            dtype=z_batch.dtype,
        )
        with gate._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
            prediction = factorial.decode_z_dec(model, z_batch, c_patch)
        prediction = prediction.detach().cpu().to(torch.float32).contiguous()
        if tuple(prediction.shape) != (stop - start, 64, 64) or not bool(torch.isfinite(prediction).all()):
            raise RuntimeError(f"invalid decoded tile batch for {label}: {tuple(prediction.shape)}")
        decoded.append(prediction)
        if batch_index % log_every == 0 or batch_index == total_batches:
            elapsed = time.monotonic() - started
            logger.info(
                "stage=decode label=%s batch=%d/%d tiles=%d/%d rate=%.2f_tiles_s elapsed=%.1fs",
                label,
                batch_index,
                total_batches,
                stop,
                int(codes.shape[0]),
                stop / max(elapsed, 1e-9),
                elapsed,
            )
    return torch.cat(decoded, dim=0)


def decoder_entry_parity(
    *,
    factorial: Any,
    gate: Any,
    model: torch.nn.Module,
    W: torch.Tensor,
    tiling: Any,
    template: Mapping[str, torch.Tensor],
    role: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> list[dict[str, Any]]:
    tiles = gate.split_tiles(W, tiling)
    rows: list[dict[str, Any]] = []
    for execution_batch in (1, 2):
        batch = tiles[:execution_batch].to(device=device, dtype=torch.float32)
        with gate._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
            condition = factorial.fixed_condition(
                template, batch=execution_batch, device=device, dtype=batch.dtype
            )
            z_dec, latents, encoder_tokens = factorial.encode_z_dec(
                model, batch, condition, return_encoder_tokens=True
            )
            c_patch = condition[1]
            d_in_mask, d_out_mask, patch_mask, T, d_in_pad = factorial.masks(execution_batch, device)
            common = {
                "dist_patch_by_patch": c_patch,
                "patch_mask": patch_mask,
                "d_in_mask": d_in_mask,
                "d_out_mask": d_out_mask,
                "d_in": 64,
                "d_out": 64,
                "d_in_pad": d_in_pad,
                "T": T,
            }
            historical = model._decode_from_latent_slots(
                latents, **common, encoder_patch_tokens=encoder_tokens
            )[0]
            explicit = factorial.decode_z_dec(model, z_dec, c_patch)
            none_tokens = model._decode_from_latent_slots(
                latents, **common, encoder_patch_tokens=None
            )[0]
            zero_tokens = model._decode_from_latent_slots(
                latents, **common, encoder_patch_tokens=torch.zeros_like(encoder_tokens)
            )[0]
            forced = model._decode_from_decoder_latent(z_dec, **common, disable_z_shortcut=True)[0]
        row = {
            "role": role,
            "execution_batch": execution_batch,
            "historical_vs_explicit_z_bit_exact": bool(torch.equal(historical, explicit)),
            "historical_vs_explicit_z_max_abs": float((historical.float() - explicit.float()).abs().max()),
            "encoder_tokens_actual_vs_none_bit_exact": bool(torch.equal(historical, none_tokens)),
            "encoder_tokens_actual_vs_zero_bit_exact": bool(torch.equal(historical, zero_tokens)),
            "z_shortcut_default_vs_forced_off_bit_exact": bool(torch.equal(explicit, forced)),
            "z_dec_shape": list(z_dec.shape),
            "z_dec_finite": bool(torch.isfinite(z_dec).all()),
            "z_dec_nonzero": bool(torch.count_nonzero(z_dec) > 0),
            "patch_size": int(model.cfg.patch_size),
            "T": int(T),
            "d_in_pad": int(d_in_pad),
            "all_patch_mask_valid": bool(torch.all(patch_mask)),
        }
        required = (
            row["historical_vs_explicit_z_bit_exact"]
            and row["encoder_tokens_actual_vs_none_bit_exact"]
            and row["encoder_tokens_actual_vs_zero_bit_exact"]
            and row["z_shortcut_default_vs_forced_off_bit_exact"]
            and row["z_dec_finite"]
            and row["z_dec_nonzero"]
            and row["historical_vs_explicit_z_max_abs"] <= 1e-6
            and row["z_dec_shape"] == [execution_batch, 512]
            and row["patch_size"] == 16
            and row["T"] == 4
            and row["d_in_pad"] == 64
            and row["all_patch_mask_valid"]
        )
        row["pass"] = bool(required)
        if not required:
            raise RuntimeError(f"decoder-entry parity failed: {row}")
        rows.append(row)
    return rows


def known_source_numeric_preflight(
    *,
    factorial: Any,
    gate: Any,
    model: torch.nn.Module,
    templates: Mapping[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Recreate a sealed source code before touching any candidate panel.

    This is a numeric path lock, not merely a configuration assertion.  The
    exact source W tile partition and fixed cell template are encoded with the
    current model, compared with the historical source factorial z_dec cache,
    and passed through both the historical latent-slots entry and the explicit
    z_dec entry.  No source or candidate activation is passed to the Weight-AE.
    """
    logger.info("stage=known_source_numeric_preflight source_parent=%s", SOURCE_PARENT)
    parent_audit = factorial.audit_parent(SOURCE_PARENT.resolve(strict=True))
    source_item = next(
        item
        for item in parent_audit["heldout_panel"]
        if int(item["context_a"]["depth"]) == 0 and item["context_a"]["role"] == "attn_query"
    )
    context = factorial.ref_from_dict(source_item["context_a"])
    score = factorial.ref_from_dict(source_item["score_b"])
    dataset = gate.build_dataset(Path(parent_audit["heldout_realpath"]), 26081677)
    W, X_source = gate.load_sample(dataset, context)
    # X_source is intentionally destroyed without entering any model method.
    source_activation_shape = list(X_source.shape)
    source_activation_sha256 = tensor_sha256(X_source)
    del X_source
    record = gate.MatrixRecord(
        model_name=context.model_name,
        depth=context.depth,
        canonical_depth=context.depth,
        role=context.role,
        layer_name=context.layer_name,
        source_key=context.source_key,
        context_ref=context,
        score_ref=score,
        W=W,
        X_context=torch.empty((0, int(W.shape[0])), dtype=torch.float32),
        X_score=torch.empty((0, int(W.shape[0])), dtype=torch.float32),
    )
    seed = 26081601
    tiling = gate.make_tiling(record, seed)
    tiling_audit = gate.validate_tiling(record, tiling)
    parent_tiling = next(
        row
        for row in parent_audit["tiling_rows"]
        if int(row["tiling_seed"]) == seed
        and int(row["depth"]) == 0
        and row["role"] == "attn_query"
    )
    if tiling_audit["partition_sha256"] != parent_tiling["partition_sha256"]:
        raise RuntimeError(
            "known-source tiling changed: "
            f"{tiling_audit['partition_sha256']} != {parent_tiling['partition_sha256']}"
        )
    tiles = gate.split_tiles(W, tiling)
    execution_batch = 64
    W_batch = tiles[:execution_batch].to(device=device, dtype=torch.float32)
    template = templates["cell_mean"][("attn_query", 0)]

    cache = torch.load(SOURCE_FACTORIAL_CACHE, map_location="cpu", weights_only=True)
    if cache.get("checkpoint_sha256") != AE_CHECKPOINT_SHA256 or int(cache.get("tiling_seed", -1)) != seed:
        raise RuntimeError("sealed source factorial cache header mismatch")
    code_key = "seed=26081601|depth=0|role=attn_query|enc=cell|code=correct"
    if code_key not in cache.get("codes", {}):
        raise RuntimeError(f"sealed source factorial cache lacks {code_key}")
    cached_full = cache["codes"][code_key].detach().cpu().to(torch.float32).contiguous()
    if tuple(cached_full.shape) != (144, 512) or not bool(torch.isfinite(cached_full).all()):
        raise RuntimeError(f"sealed source code has invalid shape/content: {tuple(cached_full.shape)}")
    cached = cached_full[:execution_batch]

    with gate._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
        condition = factorial.fixed_condition(
            template,
            batch=execution_batch,
            device=device,
            dtype=W_batch.dtype,
        )
        current_z, latents, encoder_tokens = factorial.encode_z_dec(
            model,
            W_batch,
            condition,
            return_encoder_tokens=True,
        )
        c_patch = condition[1]
        d_in_mask, d_out_mask, patch_mask, T, d_in_pad = factorial.masks(execution_batch, device)
        common = {
            "dist_patch_by_patch": c_patch,
            "patch_mask": patch_mask,
            "d_in_mask": d_in_mask,
            "d_out_mask": d_out_mask,
            "d_in": 64,
            "d_out": 64,
            "d_in_pad": d_in_pad,
            "T": T,
        }
        historical = model._decode_from_latent_slots(
            latents,
            **common,
            encoder_patch_tokens=encoder_tokens,
        )[0]
        explicit = factorial.decode_z_dec(model, current_z, c_patch)
        no_tokens = model._decode_from_latent_slots(
            latents,
            **common,
            encoder_patch_tokens=None,
        )[0]
        zero_tokens = model._decode_from_latent_slots(
            latents,
            **common,
            encoder_patch_tokens=torch.zeros_like(encoder_tokens),
        )[0]
        forced_no_shortcut = model._decode_from_decoder_latent(
            current_z,
            **common,
            disable_z_shortcut=True,
        )[0]

    current = current_z.detach().cpu().to(torch.float32).contiguous()
    difference = (current.to(torch.float64) - cached.to(torch.float64)).reshape(-1)
    cached_flat = cached.to(torch.float64).reshape(-1)
    current_flat = current.to(torch.float64).reshape(-1)
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    reference_norm = float(torch.linalg.vector_norm(cached_flat).item())
    denominator = float(torch.linalg.vector_norm(cached_flat).item() * torch.linalg.vector_norm(current_flat).item())
    cosine = float(torch.dot(cached_flat, current_flat).item() / max(denominator, 1e-30))
    comparison = {
        "bit_exact": bool(torch.equal(current, cached)),
        "current_tensor_sha256": tensor_sha256(current),
        "cached_slice_tensor_sha256": tensor_sha256(cached),
        "cached_full_tensor_sha256": tensor_sha256(cached_full),
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "relative_l2": difference_norm / max(reference_norm, 1e-30),
        "cosine": cosine,
        "fallback_thresholds": {
            "max_abs_lte": 0.0078125,
            "relative_l2_lte": 0.001,
            "cosine_gte": 0.999999,
        },
    }
    comparison["tight_bfloat16_aware_pass"] = bool(
        comparison["bit_exact"]
        or (
            comparison["max_abs"] <= comparison["fallback_thresholds"]["max_abs_lte"]
            and comparison["relative_l2"] <= comparison["fallback_thresholds"]["relative_l2_lte"]
            and comparison["cosine"] >= comparison["fallback_thresholds"]["cosine_gte"]
        )
    )
    decoder_checks = {
        "historical_vs_explicit_z_bit_exact": bool(torch.equal(historical, explicit)),
        "historical_vs_explicit_z_max_abs": float((historical.float() - explicit.float()).abs().max()),
        "encoder_tokens_actual_vs_none_bit_exact": bool(torch.equal(historical, no_tokens)),
        "encoder_tokens_actual_vs_zero_bit_exact": bool(torch.equal(historical, zero_tokens)),
        "z_shortcut_default_vs_forced_off_bit_exact": bool(torch.equal(explicit, forced_no_shortcut)),
    }
    decoder_checks["pass"] = bool(
        decoder_checks["historical_vs_explicit_z_bit_exact"]
        and decoder_checks["historical_vs_explicit_z_max_abs"] <= 1e-6
        and decoder_checks["encoder_tokens_actual_vs_none_bit_exact"]
        and decoder_checks["encoder_tokens_actual_vs_zero_bit_exact"]
        and decoder_checks["z_shortcut_default_vs_forced_off_bit_exact"]
    )
    passed = bool(comparison["tight_bfloat16_aware_pass"] and decoder_checks["pass"])
    result = {
        "schema_version": "prospective_known_source_numeric_preflight_v1",
        "pass": passed,
        "performed_before_candidate_decode": True,
        "source_activation_passed_to_weight_ae": False,
        "source_activation_shape": source_activation_shape,
        "source_activation_sha256": source_activation_sha256,
        "source_key": context.source_key,
        "depth": context.depth,
        "role": context.role,
        "tiling_seed": seed,
        "partition_sha256": tiling_audit["partition_sha256"],
        "code_key": code_key,
        "execution_batch": execution_batch,
        "cache_path": str(SOURCE_FACTORIAL_CACHE.resolve(strict=True)),
        "cache_file_sha256": sha256_file(SOURCE_FACTORIAL_CACHE.resolve(strict=True)),
        "code_comparison": comparison,
        "decoder_entry_checks": decoder_checks,
    }
    if not passed:
        raise RuntimeError(f"known-source numeric preflight failed: {result}")
    logger.info(
        "stage=known_source_numeric_preflight_complete pass=true code_bit_exact=%s max_abs=%.6g rel_l2=%.6g cosine=%.9f",
        comparison["bit_exact"],
        comparison["max_abs"],
        comparison["relative_l2"],
        comparison["cosine"],
    )
    del dataset, cache, cached_full, cached, current, current_z, latents, encoder_tokens
    return result


def assert_raw_and_scaled_stat_algebra(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    T: float,
    P: float,
    D: float,
    role: str,
    label: str,
) -> None:
    for gain_name, gain in (
        ("raw", 1.0),
        ("source_role_gain", SOURCE_ROLE_GAINS[role]),
        ("source_median_common", SOURCE_MEDIAN_COMMON_GAIN),
    ):
        direct = float(torch.sum((target - gain * prediction).square(), dtype=torch.float64).item())
        analytic = float(T - 2.0 * gain * D + gain * gain * P)
        scale = max(abs(T) + 2.0 * abs(gain * D) + gain * gain * abs(P), 1.0)
        if (
            not math.isfinite(direct)
            or not math.isfinite(analytic)
            or abs(direct - analytic) > 1e-10 * scale
        ):
            raise RuntimeError(
                f"raw/analytic sufficient-stat algebra mismatch {label}/{gain_name}: "
                f"direct={direct} analytic={analytic} scale={scale}"
            )


def raw_weight_stats(
    W: torch.Tensor,
    prediction: torch.Tensor,
    device: torch.device,
    *,
    role: str,
) -> tuple[float, float, float]:
    W64 = W.to(device=device, dtype=torch.float64)
    P64 = prediction.to(device=device, dtype=torch.float64)
    T = float(torch.sum(W64.square(), dtype=torch.float64).item())
    P = float(torch.sum(P64.square(), dtype=torch.float64).item())
    D = float(torch.sum(W64 * P64, dtype=torch.float64).item())
    if not all(math.isfinite(value) for value in (T, P, D)) or T <= 0 or P <= 0:
        raise RuntimeError(f"invalid weight sufficient stats: T={T} P={P} D={D}")
    assert_raw_and_scaled_stat_algebra(
        W64,
        P64,
        T=T,
        P=P,
        D=D,
        role=role,
        label="weight",
    )
    return T, P, D


def prepare_operator_targets(
    W: torch.Tensor,
    activations: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, float, str]]:
    W64 = W.to(device=device, dtype=torch.float64)
    result: dict[str, tuple[torch.Tensor, torch.Tensor, float, str]] = {}
    for split in SCORE_SPLITS:
        X_cpu = activations[split]
        X64 = X_cpu.to(device=device, dtype=torch.float64)
        target = X64 @ W64
        T = float(torch.sum(target.square(), dtype=torch.float64).item())
        if not math.isfinite(T) or T <= 0:
            raise RuntimeError(f"invalid operator target energy for split={split}: {T}")
        result[split] = (X64, target, T, tensor_sha256(X_cpu))
    return result


def raw_operator_stats(
    prediction: torch.Tensor,
    prepared: tuple[torch.Tensor, torch.Tensor, float, str],
    device: torch.device,
    *,
    role: str,
) -> tuple[float, float, float, str]:
    X64, target, T, activation_sha = prepared
    pred = X64 @ prediction.to(device=device, dtype=torch.float64)
    P = float(torch.sum(pred.square(), dtype=torch.float64).item())
    D = float(torch.sum(target * pred, dtype=torch.float64).item())
    if not all(math.isfinite(value) for value in (T, P, D)) or T <= 0 or P <= 0:
        raise RuntimeError(f"invalid operator sufficient stats: T={T} P={P} D={D}")
    assert_raw_and_scaled_stat_algebra(
        target,
        pred,
        T=T,
        P=P,
        D=D,
        role=role,
        label="operator",
    )
    return T, P, D, activation_sha


def stat_aliases(T: float, P: float, D: float) -> dict[str, float]:
    values = {
        "T": float(T),
        "P": float(P),
        "D": float(D),
        "target_energy": float(T),
        "pred_energy": float(P),
        "target_pred_dot": float(D),
    }
    if values["T"] != values["target_energy"] or values["P"] != values["pred_energy"] or values["D"] != values["target_pred_dot"]:
        raise AssertionError("sufficient-stat aliases diverged")
    return values


def assert_output_grids(
    weight_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    tiling_rows: list[dict[str, Any]],
    permutation_rows: list[dict[str, Any]],
    code_manifest_rows: list[dict[str, Any]],
) -> None:
    counts = expected_counts()
    observed = {
        "weight_sufficient_stat_rows": len(weight_rows),
        "operator_sufficient_stat_rows": len(operator_rows),
        "decoded_full_predictions_total": len(prediction_rows),
        "tiling_manifest_rows": len(tiling_rows),
        "permutation_manifest_rows": len(permutation_rows),
        "correct_latent_cache_entries": len(code_manifest_rows),
    }
    for name, count in observed.items():
        if count != counts[name]:
            raise RuntimeError(f"output row-count mismatch at {name}: {count} != {counts[name]}")
    key_specs = (
        (weight_rows, ("panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm")),
        (
            operator_rows,
            ("panel_id", "score_split", "tiling_seed", "depth", "role", "matrix_key", "arm"),
        ),
        (prediction_rows, ("panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm")),
        (tiling_rows, ("panel_id", "tiling_seed", "depth", "role", "matrix_key")),
        (permutation_rows, ("panel", "tiling_seed", "depth", "role", "matrix_key", "row_group")),
        (code_manifest_rows, ("panel_id", "tiling_seed", "depth", "role", "matrix_key")),
    )
    for rows, keys in key_specs:
        identities = {tuple(row[key] for key in keys) for row in rows}
        if len(identities) != len(rows):
            raise RuntimeError(f"duplicate output grid for keys={keys}: rows={len(rows)} unique={len(identities)}")
    for row in weight_rows + operator_rows:
        if not (
            float(row["T"]) == float(row["target_energy"])
            and float(row["P"]) == float(row["pred_energy"])
            and float(row["D"]) == float(row["target_pred_dot"])
        ):
            raise RuntimeError("T/P/D descriptive-alias parity failed")


def execute(args: argparse.Namespace) -> None:
    print("stage=preexecution_contract_verification weight_ae_imported=false", flush=True)
    contract, contract_sha = verify_contract(args)
    output = Path(contract["execution_config"]["output_dir"])
    if output.exists():
        raise RuntimeError(f"formal output directory must be absent: {output}")
    output.mkdir(parents=True, exist_ok=False)
    logger = setup_logging(output)
    started = time.monotonic()
    logger.info("stage=start mode=EXECUTE design_sha256=%s", DESIGN_SHA256)
    logger.info(
        "config=%s device=%s dtype=float32 score_dtype=float64 seed=%d cache_mode=verified_panel_v1 output=%s",
        json.dumps(contract["execution_config"], sort_keys=True),
        args.device,
        GLOBAL_SEED,
        output,
    )
    logger.info("tiling_seeds=%s arms=%s score_splits=%s", TILING_SEEDS, ARMS, SCORE_SPLITS)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")

    logger.info("stage=source_template_load path=%s", SOURCE_TEMPLATE)
    templates = torch.load(SOURCE_TEMPLATE, map_location="cpu", weights_only=True)
    template_summary = validate_templates(templates)
    atomic_write_json(output / "source_template_audit.json", template_summary)

    # This is the first project-code import and occurs only after the seal and
    # exact contract verification.  Neither audit nor prepare mode reaches it.
    logger.info(
        "stage=weight_ae_project_import dependencies=%d",
        len(contract["model_dependencies"]),
    )
    if str(EXPERIMENTS_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_ROOT))
    import source_confirmatory_g01_gate as gate  # type: ignore[import-not-found]  # noqa: PLC0415
    import source_latent_code_factorial as factorial  # type: ignore[import-not-found]  # noqa: PLC0415

    local_import_audit = {
        "after_helper_imports": assert_loaded_local_modules_frozen(contract),
    }
    assert_seal_clean()
    logger.info("stage=weight_ae_build checkpoint=%s", AE_CHECKPOINT)
    model, amp_enabled, amp_dtype, model_contract = factorial.build_model(
        checkpoint_path=AE_CHECKPOINT,
        device=device,
        no_amp=bool(args.no_amp),
    )
    required_model_contract = {
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
    observed_model_contract = {key: model_contract.get(key) for key in required_model_contract}
    if observed_model_contract != required_model_contract:
        raise RuntimeError(
            f"canonical deterministic AE contract failed closed: "
            f"observed={observed_model_contract} expected={required_model_contract}"
        )
    if amp_enabled is not True or amp_dtype is not torch.bfloat16:
        raise RuntimeError(f"formal AMP contract failed: enabled={amp_enabled} dtype={amp_dtype}")
    local_import_audit["after_model_build"] = assert_loaded_local_modules_frozen(contract)
    atomic_write_json(output / "loaded_local_module_audit.json", local_import_audit)
    logger.info(
        "stage=weight_ae_ready amp_enabled=%s amp_dtype=%s latent_sampling=%s rope=%s",
        amp_enabled,
        amp_dtype,
        model_contract["use_latent_sampling"],
        model_contract["rope_2d_coord_kind"],
    )
    known_source_preflight = known_source_numeric_preflight(
        factorial=factorial,
        gate=gate,
        model=model,
        templates=templates,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        logger=logger,
    )
    atomic_write_json(output / "known_source_numeric_preflight.json", known_source_preflight)
    assert_seal_clean()

    weight_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    tiling_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    code_manifest_rows: list[dict[str, Any]] = []
    correct_cache: dict[str, dict[str, Any]] = {}
    tiling_cache: dict[str, dict[str, Any]] = {}
    parity_rows: list[dict[str, Any]] = []
    prediction_counter = 0

    for panel_index, panel in enumerate(panel_handles(contract), start=1):
        logger.info(
            "stage=panel_load panel=%s panel_index=%d/%d path=%s cache_hit=true sha256=%s",
            panel.panel_id,
            panel_index,
            len(PANEL_IDS),
            panel.path,
            panel.sha256,
        )
        if sha256_file(panel.path) != panel.sha256:
            raise RuntimeError(f"panel changed after contract verification: {panel.panel_id}")
        payload = torch.load(panel.path, map_location="cpu", weights_only=True)
        # Revalidate tensors immediately before any decode, independent of the
        # earlier contract preparation and current-process contract audit.
        current_panel_audit = _validate_panel_payload(
            payload,
            panel_id=panel.panel_id,
            panel_path=panel.path,
            panel_sha256=panel.sha256,
            cache_root=Path(contract["panel_cache"]["cache_root"]),
            builder_sha256=contract["builder_sha256"],
        )
        if current_panel_audit != contract["panel_cache"]["panels"][panel.panel_id]:
            raise RuntimeError(f"panel internal audit drift before decode: {panel.panel_id}")

        weights = payload["weights"]
        activations_by_split = payload["activations"]
        for depth in range(12):
            for role in ROLES:
                key = matrix_key(depth, role)
                W = weights[key].contiguous()
                template = templates["cell_mean"][(role, depth)]
                prepared_operator = prepare_operator_targets(
                    W,
                    {split: activations_by_split[split][key] for split in SCORE_SPLITS},
                    device,
                )
                for tiling_seed in TILING_SEEDS:
                    label_base = (
                        f"panel={panel.panel_id}|seed={tiling_seed}|depth={depth:02d}|role={role}"
                    )
                    logger.info("stage=tiling %s", label_base)
                    tiling = make_tiling(
                        gate,
                        W,
                        panel=panel,
                        depth=depth,
                        role=role,
                        tiling_seed=tiling_seed,
                    )
                    tiling_audit = validate_tiling(gate, W, tiling, label=label_base)
                    expected_tiles = (ROLE_SHAPES[role][0] // 64) * (ROLE_SHAPES[role][1] // 64)
                    if tiling_audit["num_tiles"] != expected_tiles:
                        raise RuntimeError(
                            f"tile count mismatch for {label_base}: {tiling_audit['num_tiles']} != {expected_tiles}"
                        )
                    cache_key = f"{panel.panel_id}|seed={tiling_seed}|depth={depth:02d}|role={role}"
                    tiling_cache[cache_key] = {
                        "panel_id": panel.panel_id,
                        "tiling_seed": tiling_seed,
                        "depth": depth,
                        "role": role,
                        "matrix_key": key,
                        "rows": [row.cpu().to(torch.int64).contiguous() for row in tiling.rows],
                        "cols": [col.cpu().to(torch.int64).contiguous() for col in tiling.cols],
                        "row_index_sha256": tiling_audit["row_index_sha256"],
                        "column_index_sha256": tiling_audit["column_index_sha256"],
                        "partition_sha256": tiling_audit["partition_sha256"],
                    }
                    tiling_rows.append(
                        {
                            "panel_id": panel.panel_id,
                            "tiling_seed": tiling_seed,
                            "depth": depth,
                            "role": role,
                            "matrix_key": key,
                            "shape": f"{W.shape[0]}x{W.shape[1]}",
                            "row_groups": len(tiling.rows),
                            "column_groups": len(tiling.cols),
                            "num_tiles": tiling_audit["num_tiles"],
                            "coverage_min": tiling_audit["coverage_min"],
                            "coverage_max": tiling_audit["coverage_max"],
                            "weight_reassembly_bit_exact": tiling_audit["weight_reassembly_bit_exact"],
                            "coordinate_reassembly_bit_exact": tiling_audit["coordinate_reassembly_bit_exact"],
                            "row_index_sha256": tiling_audit["row_index_sha256"],
                            "column_index_sha256": tiling_audit["column_index_sha256"],
                            "partition_sha256": tiling_audit["partition_sha256"],
                        }
                    )
                    if panel.panel_id == PANEL_IDS[0] and depth == 0 and role in {"attn_query", "ffn_up"} and tiling_seed == TILING_SEEDS[0]:
                        logger.info("stage=decoder_entry_parity %s", label_base)
                        parity_rows.extend(
                            decoder_entry_parity(
                                factorial=factorial,
                                gate=gate,
                                model=model,
                                W=W,
                                tiling=tiling,
                                template=template,
                                role=role,
                                device=device,
                                amp_enabled=amp_enabled,
                                amp_dtype=amp_dtype,
                            )
                        )

                    tiles = gate.split_tiles(W, tiling).to(torch.float32).contiguous()
                    correct_codes = encode_codes(
                        factorial=factorial,
                        gate=gate,
                        model=model,
                        tiles=tiles,
                        template=template,
                        device=device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        batch_size=args.batch_size,
                        logger=logger,
                        label=label_base,
                        log_every=args.log_every_batches,
                    )
                    code_sha = tensor_sha256(correct_codes)
                    correct_cache[cache_key] = {
                        "panel_id": panel.panel_id,
                        "tiling_seed": tiling_seed,
                        "depth": depth,
                        "role": role,
                        "matrix_key": key,
                        "codes": correct_codes,
                        "codes_shape": list(correct_codes.shape),
                        "codes_dtype": str(correct_codes.dtype),
                        "tensor_sha256": code_sha,
                        "row_index_sha256": tiling_audit["row_index_sha256"],
                        "column_index_sha256": tiling_audit["column_index_sha256"],
                    }
                    code_manifest_rows.append(
                        {
                            "panel_id": panel.panel_id,
                            "tiling_seed": tiling_seed,
                            "depth": depth,
                            "role": role,
                            "matrix_key": key,
                            "cache_key": cache_key,
                            "tensor_sha256": code_sha,
                            "row_index_sha256": tiling_audit["row_index_sha256"],
                            "column_index_sha256": tiling_audit["column_index_sha256"],
                            "tiles": int(correct_codes.shape[0]),
                            "code_dim": int(correct_codes.shape[1]),
                            "codes_dtype": "float32",
                        }
                    )
                    permuted_codes, permutation_block = permute_codes_within_rows(
                        correct_codes,
                        panel_id=panel.panel_id,
                        tiling_seed=tiling_seed,
                        depth=depth,
                        role=role,
                        n_row_groups=len(tiling.rows),
                        n_col_groups=len(tiling.cols),
                    )
                    permutation_rows.extend(permutation_block)
                    zero_codes = torch.zeros_like(correct_codes)
                    if torch.count_nonzero(zero_codes).item() != 0:
                        raise AssertionError("zero_code construction is not exactly zero")
                    arm_codes = {
                        "correct": correct_codes,
                        "permuted_within_row": permuted_codes,
                        "zero_code": zero_codes,
                    }
                    hashes_this_cell: dict[str, str] = {}
                    for arm in ARMS:
                        label = f"{label_base}|arm={arm}"
                        decoded_tiles = decode_codes(
                            factorial=factorial,
                            gate=gate,
                            model=model,
                            codes=arm_codes[arm],
                            template=template,
                            device=device,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                            batch_size=args.batch_size,
                            logger=logger,
                            label=label,
                            log_every=args.log_every_batches,
                        )
                        prediction, coverage = gate.reassemble(decoded_tiles, tiling, tuple(W.shape))
                        prediction = prediction.to(torch.float32).contiguous()
                        if not bool(torch.all(coverage == 1)) or not bool(torch.isfinite(prediction).all()):
                            raise RuntimeError(f"invalid full prediction for {label}")
                        prediction_sha = tensor_sha256(prediction)
                        hashes_this_cell[arm] = prediction_sha
                        prediction_counter += 1
                        prediction_rows.append(
                            {
                                "panel_id": panel.panel_id,
                                "tiling_seed": tiling_seed,
                                "depth": depth,
                                "role": role,
                                "matrix_key": key,
                                "arm": arm,
                                "prediction_sha256": prediction_sha,
                                "prediction_shape": f"{prediction.shape[0]}x{prediction.shape[1]}",
                                "prediction_persisted": False,
                                "score_split_reuse_count": 2,
                            }
                        )
                        T_w, P_w, D_w = raw_weight_stats(
                            W,
                            prediction,
                            device,
                            role=role,
                        )
                        weight_rows.append(
                            {
                                "panel_id": panel.panel_id,
                                "tiling_seed": tiling_seed,
                                "depth": depth,
                                "role": role,
                                "matrix_key": key,
                                "arm": arm,
                                "prediction_sha256": prediction_sha,
                                **stat_aliases(T_w, P_w, D_w),
                            }
                        )
                        operator_hashes: list[str] = []
                        for split in SCORE_SPLITS:
                            T_x, P_x, D_x, activation_sha = raw_operator_stats(
                                prediction,
                                prepared_operator[split],
                                device,
                                role=role,
                            )
                            operator_rows.append(
                                {
                                    "panel_id": panel.panel_id,
                                    "score_split": split,
                                    "tiling_seed": tiling_seed,
                                    "depth": depth,
                                    "role": role,
                                    "matrix_key": key,
                                    "arm": arm,
                                    "prediction_sha256": prediction_sha,
                                    "activation_sha256": activation_sha,
                                    **stat_aliases(T_x, P_x, D_x),
                                }
                            )
                            operator_hashes.append(prediction_sha)
                        if len(set(operator_hashes)) != 1:
                            raise RuntimeError(f"A/B prediction SHA mismatch for {label}")
                        logger.info(
                            "stage=prediction_complete prediction=%d/%d %s prediction_sha256=%s weight_E_raw=%.6g elapsed=%.1fs",
                            prediction_counter,
                            expected_counts()["decoded_full_predictions_total"],
                            label,
                            prediction_sha,
                            (T_w - 2.0 * D_w + P_w) / T_w,
                            time.monotonic() - started,
                        )
                        del decoded_tiles, prediction
                    if len(set(hashes_this_cell.values())) != len(ARMS):
                        raise RuntimeError(f"prediction hash collision across intervention arms: {label_base} {hashes_this_cell}")
                    del tiles, permuted_codes, zero_codes, arm_codes
                del prepared_operator
        del payload, weights, activations_by_split
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("stage=panel_complete panel=%s elapsed=%.1fs", panel.panel_id, time.monotonic() - started)

    if len(parity_rows) != 4 or not all(bool(row["pass"]) for row in parity_rows):
        raise RuntimeError(f"decoder-entry parity grid incomplete/failed: {parity_rows}")
    if len(correct_cache) != expected_counts()["correct_latent_cache_entries"]:
        raise RuntimeError(f"correct latent cache entry mismatch: {len(correct_cache)}")
    if len(tiling_cache) != expected_counts()["tiling_manifest_rows"]:
        raise RuntimeError(f"tiling cache entry mismatch: {len(tiling_cache)}")
    assert_output_grids(
        weight_rows,
        operator_rows,
        prediction_rows,
        tiling_rows,
        permutation_rows,
        code_manifest_rows,
    )
    assert_seal_clean()

    logger.info("stage=artifact_write")
    atomic_write_csv(output / "weight_sufficient_stats.csv", weight_rows)
    atomic_write_csv(output / "operator_sufficient_stats.csv", operator_rows)
    atomic_write_csv(output / "prediction_manifest.csv", prediction_rows)
    atomic_write_csv(output / "tiling_manifest.csv", tiling_rows)
    atomic_write_csv(output / "permutation_manifest.csv", permutation_rows)
    atomic_write_csv(output / "correct_latent_code_manifest.csv", code_manifest_rows)
    atomic_torch_save(
        output / "correct_latent_codes.pt",
        {
            "schema_version": "prospective_correct_latent_codes_v1",
            "entries": correct_cache,
        },
    )
    atomic_torch_save(
        output / "tiling_indices.pt",
        {
            "schema_version": "prospective_tiling_indices_v1",
            "entries": tiling_cache,
        },
    )
    atomic_write_json(output / "decoder_entry_parity.json", {"pass": True, "probes": parity_rows})
    atomic_write_json(
        output / "sufficient_statistics_schema.json",
        {
            "schema_version": "prospective_sufficient_statistics_v1",
            "files": {
                "weight": {
                    "filename": "weight_sufficient_stats.csv",
                    "row_count": 864,
                    "key_columns": [
                        "panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm"
                    ],
                    "space": "target=W; prediction=W_hat",
                },
                "operator": {
                    "filename": "operator_sufficient_stats.csv",
                    "row_count": 1728,
                    "key_columns": [
                        "panel_id", "score_split", "tiling_seed", "depth", "role", "matrix_key", "arm"
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
        },
    )
    atomic_write_json(
        output / "schema.json",
        {
            "schema_version": "prospective_geometry_matched_replication_output_v1",
            "expected_counts": expected_counts(),
            "tiling_index_hash_contract": {
                "storage": "tiling_indices.pt entries contain ordered lists of CPU int64 length-64 tensors",
                "row_materialization": "torch.stack(entry['rows']).to(torch.int64).contiguous()",
                "column_materialization": "torch.stack(entry['cols']).to(torch.int64).contiguous()",
                "hash": "SHA256(str(tuple(tensor.shape)).encode('utf-8') + contiguous little-endian int64 C bytes)",
                "partition_sha256": "SHA256((row_index_sha256+'|'+column_index_sha256).encode('utf-8'))",
                "coverage_requirement": "each matrix coordinate exactly once; value and unique-coordinate reassembly bit exact",
                "entry_count": 288,
            },
            "full_prediction_matrices_persisted": False,
            "operator_split_predictions_duplicated": False,
            "source_gains_applied": False,
            "analyzer_path": contract["scripts"]["analyzer"]["path"],
            "analyzer_sha256": contract["scripts"]["analyzer"]["sha256"],
        },
    )
    atomic_write_json(output / "resolved_config.json", contract["execution_config"])
    atomic_write_json(output / "model_contract.json", model_contract)
    atomic_copy_file(args.contract_path.resolve(strict=True), output / "preexecution_contract.json")
    if sha256_file(output / "preexecution_contract.json") != contract_sha:
        raise RuntimeError("output preexecution contract is not byte-identical to the external frozen contract")
    atomic_write_json(
        output / "preexecution_binding.json",
        {
            "schema_version": "prospective_geometry_matched_replication_binding_v1",
            "external_contract_path": str(args.contract_path.resolve(strict=True)),
            "external_contract_sha256": contract_sha,
            "copied_contract_path": str((output / "preexecution_contract.json").resolve(strict=True)),
            "copied_contract_sha256": sha256_file(output / "preexecution_contract.json"),
            "design_sha256": DESIGN_SHA256,
            "runner_sha256": contract["runner_sha256"],
            "builder_sha256": contract["builder_sha256"],
            "analyzer_sha256": contract["analyzer_sha256"],
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "scripts": contract["scripts"],
            "model_dependencies": contract["model_dependencies"],
            "resources": contract["resources"],
            "panel_cache": contract["panel_cache"],
            "panel_manifest_sha256": contract["panel_cache"]["manifest_sha256"],
        },
    )
    local_import_audit["after_all_weight_ae_forwards"] = assert_loaded_local_modules_frozen(contract)
    atomic_write_json(output / "loaded_local_module_audit.json", local_import_audit)
    metadata = {
        "schema_version": "prospective_geometry_matched_replication_runner_v1",
        "design_sha256": DESIGN_SHA256,
        "runner_sha256": contract["runner_sha256"],
        "builder_sha256": contract["builder_sha256"],
        "analyzer_sha256": contract["analyzer_sha256"],
        "contract_sha256": contract_sha,
        "panel_manifest_sha256": contract["panel_manifest_sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "execution_status": "COMPLETE",
        "target_seal": {
            "installed_before_input_read": bool(_SEAL_STATE["installed_before_input_read"]),
            "target_access_events": int(_SEAL_STATE["target_access_events"]),
            "network_connections": int(_SEAL_STATE["network_connections"]),
            "subprocess_launches": int(_SEAL_STATE["subprocess_launches"]),
        },
        "panels": [
            {
                "panel_id": panel_id,
                "path": contract["panel_cache"]["panels"][panel_id]["path"],
                "sha256": contract["panel_cache"]["panels"][panel_id]["sha256"],
                "checkpoint_sha256": contract["panel_cache"]["panels"][panel_id]["checkpoint_sha256"],
                "provenance_pass": True,
                "resource_hash_pass": True,
                "quality_pass": True,
                "validity_pass": True,
            }
            for panel_id in PANEL_IDS
        ],
        "counts": expected_counts(),
        "decoder_entry_parity_pass": True,
        "known_source_numeric_preflight_pass": bool(known_source_preflight["pass"]),
        "hard_checks": {
            "transpose_role_map_parity_pass": True,
            "raw_stat_algebra_pass": True,
            "analytic_scale_parity_pass": True,
            "decoder_entry_parity_pass": True,
            "A_B_prediction_sha_pass": True,
            "derangement_pass": True,
            "tiling_coverage_pass": True,
            "finite_nondegenerate_pass": True,
            "source_only_dataflow_pass": True,
            "loaded_local_module_closure_pass": True,
        },
        "panel_activations_to_weight_ae": False,
        "full_prediction_matrices_persisted": False,
        "source_gains_applied_by_runner": False,
        "analyzer_invoked": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(output / "runner_metadata.json", metadata)
    logger.info(
        "stage=complete predictions=%d weight_rows=%d operator_rows=%d z_cache_entries=%d elapsed=%.1fs output=%s",
        len(prediction_rows),
        len(weight_rows),
        len(operator_rows),
        len(correct_cache),
        time.monotonic() - started,
        output,
    )
    logger.info(
        "artifacts=weight_sufficient_stats.csv,operator_sufficient_stats.csv,correct_latent_codes.pt,tiling_indices.pt"
    )
    close_logging(logger)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    assert_seal_clean()

    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    artifact_manifest = {
        "schema_version": "prospective_geometry_matched_replication_artifact_manifest_v1",
        "manifest_self_excluded": True,
        "artifacts": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
        "count": len(files),
    }
    atomic_write_json(output / "artifact_manifest.json", artifact_manifest)
    print(
        f"stage=artifact_manifest_written path={output / 'artifact_manifest.json'} "
        f"sha256={sha256_file(output / 'artifact_manifest.json')} files={len(files)}",
        flush=True,
    )


def main() -> None:
    args = parser().parse_args()
    # The seal is deliberately the first action after argument parsing.  No
    # builder, panel, source resource, output, or project module has been read.
    install_source_only_seal()
    reject_forbidden_argv(sys.argv)
    reject_forbidden_path_arguments(args)
    if args.execute:
        execute(args)
    elif args.prepare_contract:
        prepare_contract(args)
    else:
        audit_only(args)


if __name__ == "__main__":
    main()
