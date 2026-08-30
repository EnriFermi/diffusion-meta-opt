#!/usr/bin/env python3
"""Build the frozen Beans and TrOCR/SROIE geometry-matched source panels.

The default mode is a read-only audit.  ``--execute`` is required to load
models, evaluate quality, collect activations, or write a panel cache.  This
program intentionally has no imports from the Weight-AE implementation and
never computes a Weight-AE prediction or outcome.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import logging
import math
import os
import platform
import re
import socket
import statistics
import struct
import subprocess
import sys
import time
import traceback
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


# These are set before any optional ML/data dependency is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


PROJECT = Path("/home/coder/project")
DESIGN_PATH = (
    PROJECT
    / "projects/weight-vae/workspace/docs/notes/"
    "prospective_geometry_matched_source_replication_design_20260816.md"
)
DESIGN_SHA256 = "fc23d52d07d0004afd681116578abdba562dff1a415b059fd72ec261b1e47d70"
DEFAULT_OUTPUT = (
    PROJECT
    / "artifacts/crossmodal_united_structure/prospective_geometry_matched_panels_20260816"
)

BEANS_MODEL_ID = "nateraw/vit-base-beans"
BEANS_MODEL_REVISION = "41f85ace09a4613c2c65495b3b8465c4ceee1d00"
BEANS_MODEL = PROJECT / "projects/shared/storage/data/models/vit_base_beans_nateraw_41f85ace"
BEANS_DATA_REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
BEANS_DATA = PROJECT / "projects/shared/storage/data/datasets/beans_27aa014c"

TROCR_MODEL_ID = "microsoft/trocr-base-printed"
TROCR_MODEL_REVISION = "93450be3f1ed40a930690d951ef3932687cc1892"
TROCR_MODEL = (
    PROJECT
    / "projects/shared/storage/data/models/trocr_base_printed/"
    "models--microsoft--trocr-base-printed/snapshots/"
    "93450be3f1ed40a930690d951ef3932687cc1892"
)
SROIE_DATA_REVISION = "04f6537e418eeb88863d617eb27817cc496522d7"
SROIE_DATA = PROJECT / "projects/shared/storage/data/datasets/sroie_text_recognition_04f6537e"

AE_BANK = (
    PROJECT
    / "projects/shared/storage/artifacts/training/checkpoints/"
    "weight_quantile_vae/stage_1/offline_dataset"
)
SCOUT_MANIFEST = (
    PROJECT
    / "projects/shared/storage/artifacts/crossmodal_united_structure/"
    "prospective_source_panel_scout_20260816/artifact_manifest.json"
)

PANEL_IDS = ("beans", "trocr_sroie")
SCORE_SPLITS = ("A", "B")
PANEL_RESOURCE_LABELS = {
    "beans": (
        "beans_model_weights",
        "beans_model_config",
        "beans_preprocessor_config",
        "beans_train_parquet",
        "beans_validation_parquet",
        "beans_test_parquet",
    ),
    "trocr_sroie": (
        "trocr_model_weights",
        "trocr_model_config",
        "trocr_preprocessor_config",
        "trocr_generation_config",
        "trocr_tokenizer_config",
        "trocr_special_tokens_map",
        "trocr_vocab",
        "trocr_merges",
        "sroie_test_zip",
    ),
}
ROLES: tuple[tuple[str, str], ...] = (
    ("attn_query", "attention.attention.query"),
    ("attn_key", "attention.attention.key"),
    ("attn_value", "attention.attention.value"),
    ("attn_output", "attention.output.dense"),
    ("ffn_up", "intermediate.dense"),
    ("ffn_down", "output.dense"),
)
EXPECTED_SHAPE_BY_ROLE = {
    "attn_query": (768, 768),
    "attn_key": (768, 768),
    "attn_value": (768, 768),
    "attn_output": (768, 768),
    "ffn_up": (768, 3072),
    "ffn_down": (3072, 768),
}
COMPATIBLE_SHAPES = frozenset(EXPECTED_SHAPE_BY_ROLE.values())

BEANS_SELECTION_SEED = 26_081_831
SROIE_PARTITION_SEED = 26_081_821
BEANS_QUALITY_GATE = 0.85
SROIE_NONEMPTY_GATE = 0.99
SROIE_EXACT_GATE = 0.50
SROIE_CORPUS_CER_GATE = 0.25
SROIE_MEDIAN_CER_GATE = 0.20

EXPECTED_RUNTIME = {
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


@dataclass(frozen=True)
class ResourceSpec:
    label: str
    path: Path
    sha256: str


RESOURCE_SPECS: tuple[ResourceSpec, ...] = (
    ResourceSpec("frozen_design", DESIGN_PATH, DESIGN_SHA256),
    ResourceSpec(
        "unseen_weight_scout_manifest",
        SCOUT_MANIFEST,
        "623cbf54611bdba49a0007840b52457cbfa382935507cb13020ffef5e44e5254",
    ),
    ResourceSpec(
        "ae_bank_manifest",
        AE_BANK / "manifest.json",
        "b49a4b2045680703ce54276676e3083b70f179317a960f4229b2829b3af41721",
    ),
    ResourceSpec(
        "ae_bank_sources",
        AE_BANK / "sources.json",
        "e353bba6d6924dd55be0e4cb1e78e1b34cd656020d203db89f2b60874a86b45d",
    ),
    ResourceSpec(
        "beans_model_weights",
        BEANS_MODEL / "pytorch_model.bin",
        "fc443b145fcc3a09cf07eb28b94e9a989ec3f0e8f7255e1fa08c0953ed4bae91",
    ),
    ResourceSpec(
        "beans_model_config",
        BEANS_MODEL / "config.json",
        "366d2ad9bf48e94932bb83e0d2367e6f95ed18befb1a4ef6903934b7af737237",
    ),
    ResourceSpec(
        "beans_preprocessor_config",
        BEANS_MODEL / "preprocessor_config.json",
        "af4eb4d79cf61b47010fc0bc9352ee967579c417423b4917188d809b7e048948",
    ),
    ResourceSpec(
        "beans_train_parquet",
        BEANS_DATA / "data/train-00000-of-00001.parquet",
        "7f905a7323966a58e89b8e839ed656bb869fc82d16a3fadc7dce40972a5f8b19",
    ),
    ResourceSpec(
        "beans_validation_parquet",
        BEANS_DATA / "data/validation-00000-of-00001.parquet",
        "33f774593d8b31585457b70c224744e9409ffdee4e91a11822b1ebfe8242928f",
    ),
    ResourceSpec(
        "beans_test_parquet",
        BEANS_DATA / "data/test-00000-of-00001.parquet",
        "534a6b0648f585d69b7ec0ad7a7540720d60c8db8106dc6d0508296316f6cb27",
    ),
    ResourceSpec(
        "trocr_model_weights",
        TROCR_MODEL / "model.safetensors",
        "1cf4a6eedab26afaaf505f1c7f73d9634944924dbd1ed049d569db98039cd596",
    ),
    ResourceSpec(
        "trocr_model_config",
        TROCR_MODEL / "config.json",
        "5bda1deab455661feb3d91906656e5600e2ca520d5c00a2a03836614b850c93e",
    ),
    ResourceSpec(
        "trocr_preprocessor_config",
        TROCR_MODEL / "preprocessor_config.json",
        "2fcc0da9466ee00be0403b26027373039e032820ebac409e207b32e52e52119d",
    ),
    ResourceSpec(
        "trocr_generation_config",
        TROCR_MODEL / "generation_config.json",
        "41149cdcffec4d657f32dfcddd9b208037f01286c9e07945c724908c58ed0193",
    ),
    ResourceSpec(
        "trocr_tokenizer_config",
        TROCR_MODEL / "tokenizer_config.json",
        "5a1356884c6ae736a621841535264ba7c5bebd52f169258add2c48fcbb32d50a",
    ),
    ResourceSpec(
        "trocr_special_tokens_map",
        TROCR_MODEL / "special_tokens_map.json",
        "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
    ),
    ResourceSpec(
        "trocr_vocab",
        TROCR_MODEL / "vocab.json",
        "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
    ),
    ResourceSpec(
        "trocr_merges",
        TROCR_MODEL / "merges.txt",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
    ResourceSpec(
        "sroie_test_zip",
        SROIE_DATA / "test.zip",
        "533dba4d017a70617943f857fe01a986d34da5095255112f88af03df4325484b",
    ),
)


class _ForbiddenModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, seal: "ResearchSeal") -> None:
        self.seal = seal

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any | None = None,
    ) -> Any | None:
        del path, target
        if "data2vec" in fullname.casefold():
            self.seal.blocked_import_events.append(fullname)
            raise RuntimeError(f"research seal blocked forbidden module import: {fullname}")
        return None


class ResearchSeal:
    """Process-wide target/network/subprocess guard installed before research reads."""

    def __init__(self, project_root: Path = PROJECT) -> None:
        self.project_root = os.path.realpath(os.fspath(project_root))
        self.installed = False
        self.blocked_path_events: list[dict[str, str]] = []
        self.network_connection_events: list[str] = []
        self.subprocess_events: list[str] = []
        self.blocked_import_events: list[str] = []
        self.hidden_directory_entries: list[dict[str, str]] = []
        self._originals: dict[str, Any] = {}

    @staticmethod
    def _forbidden_path_text(text: str) -> bool:
        normalized = text.replace("\\", "/").casefold()
        components = [value for value in normalized.split("/") if value]
        return "data2vec" in normalized or any(
            value in {"target", "targets"} or value.startswith(("target_", "target-"))
            for value in components
        )

    def guard_path(self, raw_path: Any, *, operation: str) -> None:
        if isinstance(raw_path, int) or raw_path is None:
            return
        try:
            raw = os.fsdecode(raw_path)
        except TypeError:
            return
        raw_absolute = os.path.abspath(raw)
        resolved = os.path.realpath(raw_absolute)
        if self._forbidden_path_text(raw) or self._forbidden_path_text(resolved):
            event = {"operation": operation, "raw": raw, "resolved": resolved}
            self.blocked_path_events.append(event)
            raise RuntimeError(f"research seal blocked forbidden path: {event}")

    def _audit_hook(self, event: str, args: tuple[Any, ...]) -> None:
        if event in {"open", "os.listdir", "os.scandir"} and args:
            self.guard_path(args[0], operation=f"audit:{event}")
        if event == "socket.connect":
            self.network_connection_events.append(repr(args))
            raise RuntimeError("research seal blocked socket.connect")
        if event == "subprocess.Popen" or event == "os.system" or event.startswith(("os.exec", "os.spawn")):
            self.subprocess_events.append(f"{event}:{args!r}")
            raise RuntimeError(f"research seal blocked subprocess launch: {event}")

    def install(self) -> None:
        if self.installed:
            return
        forbidden_loaded = sorted(
            name for name in sys.modules if "data2vec" in name.casefold()
        )
        if forbidden_loaded:
            self.blocked_import_events.extend(forbidden_loaded)
            raise RuntimeError(
                f"research seal found forbidden modules loaded before installation: {forbidden_loaded}"
            )
        sys.addaudithook(self._audit_hook)
        sys.meta_path.insert(0, _ForbiddenModuleFinder(self))

        self._originals["builtins.open"] = builtins.open
        self._originals["io.open"] = io.open
        self._originals["os.open"] = os.open
        self._originals["os.listdir"] = os.listdir
        self._originals["os.scandir"] = os.scandir
        self._originals["socket.connect"] = socket.socket.connect
        self._originals["socket.connect_ex"] = socket.socket.connect_ex
        self._originals["socket.create_connection"] = socket.create_connection
        self._originals["subprocess.Popen"] = subprocess.Popen
        self._originals["subprocess.run"] = subprocess.run
        self._originals["subprocess.call"] = subprocess.call
        self._originals["subprocess.check_call"] = subprocess.check_call
        self._originals["subprocess.check_output"] = subprocess.check_output
        self._originals["os.system"] = os.system
        self._originals["os.popen"] = os.popen

        def guarded_builtin_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            self.guard_path(file, operation="builtins.open")
            return self._originals["builtins.open"](file, *args, **kwargs)

        def guarded_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            self.guard_path(file, operation="io.open")
            return self._originals["io.open"](file, *args, **kwargs)

        def guarded_os_open(path: Any, *args: Any, **kwargs: Any) -> Any:
            self.guard_path(path, operation="os.open")
            return self._originals["os.open"](path, *args, **kwargs)

        def guarded_listdir(path: Any = ".") -> Any:
            self.guard_path(path, operation="os.listdir")
            entries = self._originals["os.listdir"](path)
            visible = []
            for entry in entries:
                raw_child = (
                    os.fsdecode(entry)
                    if isinstance(path, int)
                    else os.path.join(os.fsdecode(path), os.fsdecode(entry))
                )
                resolved_child = os.path.realpath(os.path.abspath(raw_child))
                if self._forbidden_path_text(raw_child) or self._forbidden_path_text(
                    resolved_child
                ):
                    self.hidden_directory_entries.append(
                        {
                            "operation": "os.listdir:filter_child_before_access",
                            "raw": raw_child,
                            "resolved": resolved_child,
                        }
                    )
                    continue
                visible.append(entry)
            return visible

        def guarded_scandir(path: Any = ".") -> Any:
            self.guard_path(path, operation="os.scandir")
            original_iterator = self._originals["os.scandir"](path)
            seal = self

            class FilteredScandirIterator:
                def __iter__(self) -> "FilteredScandirIterator":
                    return self

                def __next__(self) -> Any:
                    while True:
                        entry = next(original_iterator)
                        raw_child = os.fspath(entry.path)
                        resolved_child = os.path.realpath(os.path.abspath(raw_child))
                        if seal._forbidden_path_text(raw_child) or seal._forbidden_path_text(
                            resolved_child
                        ):
                            seal.hidden_directory_entries.append(
                                {
                                    "operation": "os.scandir:filter_child_before_access",
                                    "raw": raw_child,
                                    "resolved": resolved_child,
                                }
                            )
                            continue
                        return entry

                def __enter__(self) -> "FilteredScandirIterator":
                    original_iterator.__enter__()
                    return self

                def __exit__(self, *args: Any) -> Any:
                    return original_iterator.__exit__(*args)

                def close(self) -> None:
                    original_iterator.close()

            return FilteredScandirIterator()

        def block_network(*args: Any, **kwargs: Any) -> Any:
            self.network_connection_events.append(f"args={args!r},kwargs={kwargs!r}")
            raise RuntimeError("research seal blocked a network connection")

        def block_subprocess(*args: Any, **kwargs: Any) -> Any:
            self.subprocess_events.append(f"args={args!r},kwargs={kwargs!r}")
            raise RuntimeError("research seal blocked a subprocess launch")

        builtins.open = guarded_builtin_open
        io.open = guarded_io_open
        os.open = guarded_os_open
        os.listdir = guarded_listdir
        os.scandir = guarded_scandir
        socket.socket.connect = block_network
        socket.socket.connect_ex = block_network
        socket.create_connection = block_network
        subprocess.Popen = block_subprocess
        subprocess.run = block_subprocess
        subprocess.call = block_subprocess
        subprocess.check_call = block_subprocess
        subprocess.check_output = block_subprocess
        os.system = block_subprocess
        os.popen = block_subprocess
        self.installed = True

    def snapshot(self) -> dict[str, Any]:
        forbidden_loaded = sorted(name for name in sys.modules if "data2vec" in name.casefold())
        return {
            "installed": self.installed,
            "forbidden_path_markers": ["data2vec", "target path component"],
            "blocked_path_events": list(self.blocked_path_events),
            "blocked_import_events": list(self.blocked_import_events),
            "hidden_directory_entries": list(self.hidden_directory_entries),
            "forbidden_loaded_modules": forbidden_loaded,
            "network_connection_events": list(self.network_connection_events),
            "subprocess_events": list(self.subprocess_events),
            "target_access_event_count": len(self.blocked_path_events) + len(self.blocked_import_events),
            "network_connection_count": len(self.network_connection_events),
            "subprocess_count": len(self.subprocess_events),
            "pass": bool(
                self.installed
                and not self.blocked_path_events
                and not self.blocked_import_events
                and not forbidden_loaded
                and not self.network_connection_events
                and not self.subprocess_events
            ),
        }


@dataclass(frozen=True)
class Runtime:
    torch: Any
    np: Any
    pq: Any
    Image: Any
    safe_open: Any
    ViTImageProcessor: Any
    ViTForImageClassification: Any
    TrOCRProcessor: Any
    VisionEncoderDecoderModel: Any


@dataclass
class SampleRecord:
    panel_id: str
    partition: str
    ordinal: int
    stable_id: str
    dataset_index: int | None
    class_id: int | None
    class_name: str
    file_name: str
    reference_text: str
    canonical_rank_json: str
    rank_sha256: str
    raw_bytes: bytes
    raw_sha256: str
    decoded_rgb_sha256: str
    width: int
    height: int
    source_container: str
    source_member: str

    def manifest_row(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "partition": self.partition,
            "ordinal": self.ordinal,
            "stable_id": self.stable_id,
            "dataset_index": "" if self.dataset_index is None else self.dataset_index,
            "class_id": "" if self.class_id is None else self.class_id,
            "class_name": self.class_name,
            "file_name": self.file_name,
            "reference_text": self.reference_text,
            "canonical_rank_json": self.canonical_rank_json,
            "rank_sha256": self.rank_sha256,
            "raw_sha256": self.raw_sha256,
            "decoded_rgb_sha256": self.decoded_rgb_sha256,
            "width": self.width,
            "height": self.height,
            "source_container": self.source_container,
            "source_member": self.source_member,
        }


@dataclass(frozen=True)
class MatrixRecord:
    panel_id: str
    matrix_key: str
    depth: int
    role: str
    state_key: str
    module_name: str
    weight: Any
    weight_shape_bytes_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run quality and build panels.")
    mode.add_argument("--audit-only", dest="execute", action="store_false", help="Read-only preflight (default).")
    parser.set_defaults(execute=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beans-quality-batch-size", type=int, default=32)
    parser.add_argument("--beans-activation-batch-size", type=int, default=16)
    parser.add_argument("--trocr-quality-batch-size", type=int, default=8)
    parser.add_argument("--trocr-activation-batch-size", type=int, default=8)
    parser.add_argument("--proximity-ae-chunk-size", type=int, default=8)
    args = parser.parse_args(argv)
    for name in (
        "beans_quality_batch_size",
        "beans_activation_batch_size",
        "trocr_quality_batch_size",
        "trocr_activation_batch_size",
        "proximity_ae_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def load_runtime() -> Runtime:
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers.models.trocr.processing_trocr import TrOCRProcessor
    from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import (
        VisionEncoderDecoderModel,
    )
    from transformers.models.vit.image_processing_vit import ViTImageProcessor
    from transformers.models.vit.modeling_vit import ViTForImageClassification

    return Runtime(
        torch=torch,
        np=np,
        pq=pq,
        Image=Image,
        safe_open=safe_open,
        ViTImageProcessor=ViTImageProcessor,
        ViTForImageClassification=ViTForImageClassification,
        TrOCRProcessor=TrOCRProcessor,
        VisionEncoderDecoderModel=VisionEncoderDecoderModel,
    )


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in EXPECTED_RUNTIME},
    }


def verify_runtime_versions() -> dict[str, Any]:
    found = runtime_versions()
    expected_distribution = dict(EXPECTED_RUNTIME)
    # PyTorch's wheel metadata omits the local CUDA tag while torch.__version__
    # includes it.  Audit mode stays ML-import-free; execute mode checks both.
    expected_distribution["torch"] = EXPECTED_RUNTIME["torch"].split("+", 1)[0]
    mismatches = {
        name: {"expected": expected, "found": found[name]}
        for name, expected in expected_distribution.items()
        if found[name] != expected
    }
    if sys.version_info[:2] != (3, 12):
        mismatches["python"] = {"expected": "3.12.x", "found": found["python"]}
    if mismatches:
        raise RuntimeError(f"runtime stack mismatch: {mismatches}")
    return {
        "versions": found,
        "expected_distribution_versions": {
            "python": "3.12.x",
            **expected_distribution,
        },
        "execute_runtime_checks": {
            "torch.__version__": EXPECTED_RUNTIME["torch"],
            "torch.version.cuda": "12.8",
        },
        "execute_runtime_checks_deferred": True,
        "pass": True,
    }


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_rank(value: Mapping[str, Any]) -> tuple[str, str]:
    payload = canonical_json(value)
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path, seal: ResearchSeal | None = None) -> str:
    if seal is not None:
        seal.guard_path(path, operation="sha256_file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def shape_bytes_weight_sha256(weight: Any, runtime: Runtime) -> str:
    torch = runtime.torch
    value = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError(f"formal weight hash requires a matrix, got {tuple(value.shape)}")
    array = value.numpy().astype("<f4", copy=False)
    payload = struct.pack("<QQ", int(value.shape[0]), int(value.shape[1])) + b"\0"
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(memoryview(array))
    return digest.hexdigest()


def tensor_manifest_sha256(tensor: Any, runtime: Runtime) -> str:
    torch = runtime.torch
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(struct.pack("<Q", value.ndim))
    digest.update(struct.pack("<" + "Q" * value.ndim, *[int(item) for item in value.shape]))
    digest.update(memoryview(value.numpy().astype("<f4", copy=False)))
    return digest.hexdigest()


def decoded_rgb_hash(raw_bytes: bytes, runtime: Runtime) -> tuple[str, int, int]:
    with runtime.Image.open(io.BytesIO(raw_bytes)) as raw_image:
        image = raw_image.convert("RGB")
        width, height = image.size
        pixels = image.tobytes()
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", int(height), int(width)))
    digest.update(pixels)
    return digest.hexdigest(), int(width), int(height)


def decode_rgb(raw_bytes: bytes, runtime: Runtime) -> Any:
    with runtime.Image.open(io.BytesIO(raw_bytes)) as raw_image:
        return raw_image.convert("RGB")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    values = [dict(row) for row in rows]
    if not values and fieldnames is None:
        raise ValueError(f"cannot infer empty CSV schema for {path}")
    columns = list(fieldnames or values[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def verify_resources(seal: ResearchSeal, logger: logging.Logger | None = None) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(RESOURCE_SPECS, start=1):
        seal.guard_path(spec.path, operation=f"resource:{spec.label}:raw")
        resolved = Path(os.path.realpath(os.fspath(spec.path)))
        seal.guard_path(resolved, operation=f"resource:{spec.label}:resolved")
        if not resolved.is_file():
            raise FileNotFoundError(f"missing frozen resource {spec.label}: {resolved}")
        started = time.monotonic()
        actual = sha256_file(resolved, seal)
        if actual != spec.sha256:
            raise RuntimeError(
                f"resource hash mismatch {spec.label}: expected={spec.sha256} actual={actual} path={resolved}"
            )
        result = {
            "path": str(spec.path),
            "realpath": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": actual,
            "expected_sha256": spec.sha256,
            "pass": True,
        }
        results[spec.label] = result
        if logger is not None:
            logger.info(
                "stage=resource_hash item=%d/%d label=%s bytes=%d sha256=%s elapsed_s=%.2f",
                index,
                len(RESOURCE_SPECS),
                spec.label,
                result["bytes"],
                actual,
                time.monotonic() - started,
            )
    return results


def matrix_key(depth: int, role: str) -> str:
    return f"depth={depth:02d}|role={role}"


def expected_matrix_keys() -> set[str]:
    return {matrix_key(depth, role) for depth in range(12) for role, _suffix in ROLES}


def validate_hash_collisions(rows: Sequence[SampleRecord], *, context: str) -> None:
    stable_owners: dict[str, SampleRecord] = {}
    for row in rows:
        previous = stable_owners.get(row.stable_id)
        if previous is not None:
            raise RuntimeError(
                f"{context}: duplicate stable_id={row.stable_id} "
                f"partitions={previous.partition}/{row.partition}"
            )
        stable_owners[row.stable_id] = row

    for field in ("raw_sha256", "decoded_rgb_sha256"):
        owners: dict[str, set[str]] = {}
        for row in rows:
            value = str(getattr(row, field))
            previous_partitions = owners.setdefault(value, set())
            if previous_partitions and row.partition not in previous_partitions:
                raise RuntimeError(
                    f"{context}: cross-partition collision field={field} value={value} "
                    f"partitions={sorted(previous_partitions)}/{row.partition}"
                )
            previous_partitions.add(row.partition)

    canonical_rows: dict[str, SampleRecord] = {}
    for row in rows:
        value = canonical_json(row.manifest_row())
        previous = canonical_rows.get(value)
        if previous is not None:
            raise RuntimeError(
                f"{context}: duplicate canonical manifest row "
                f"samples={previous.stable_id}/{row.stable_id}"
            )
        canonical_rows[value] = row


def _image_payload_bytes(value: Mapping[str, Any]) -> bytes:
    payload = value.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected embedded image bytes, got {type(payload)}")
    return bytes(payload)


def prepare_beans_samples(runtime: Runtime, seal: ResearchSeal, logger: logging.Logger) -> tuple[list[SampleRecord], list[str]]:
    train_path = BEANS_DATA / "data/train-00000-of-00001.parquet"
    test_path = BEANS_DATA / "data/test-00000-of-00001.parquet"
    for path in (train_path, test_path):
        seal.guard_path(path, operation="beans_parquet")
    logger.info("stage=beans_selection_load train=%s test=%s", train_path, test_path)
    train_file = runtime.pq.ParquetFile(train_path)
    test_file = runtime.pq.ParquetFile(test_path)
    if train_file.metadata.num_rows != 1034 or test_file.metadata.num_rows != 128:
        raise RuntimeError(
            f"Beans split-size drift train={train_file.metadata.num_rows} test={test_file.metadata.num_rows}"
        )
    metadata = train_file.schema_arrow.metadata or {}
    if b"huggingface" not in metadata:
        raise RuntimeError("Beans Parquet lacks Hugging Face feature metadata")
    features = json.loads(metadata[b"huggingface"].decode("utf-8"))["info"]["features"]
    class_names = list(features["labels"]["names"])
    if class_names != ["angular_leaf_spot", "bean_rust", "healthy"]:
        raise RuntimeError(f"Beans class-name drift: {class_names}")
    train_rows = train_file.read(columns=["image_file_path", "image", "labels"]).to_pylist()
    test_rows = test_file.read(columns=["image_file_path", "image", "labels"]).to_pylist()

    result: list[SampleRecord] = []
    for dataset_index, row in enumerate(test_rows):
        raw = _image_payload_bytes(row["image"])
        rgb_hash, width, height = decoded_rgb_hash(raw, runtime)
        class_id = int(row["labels"])
        result.append(
            SampleRecord(
                panel_id="beans",
                partition="quality",
                ordinal=dataset_index,
                stable_id=f"beans:test:{dataset_index}",
                dataset_index=dataset_index,
                class_id=class_id,
                class_name=class_names[class_id],
                file_name=str(row["image_file_path"]),
                reference_text="",
                canonical_rank_json="",
                rank_sha256="",
                raw_bytes=raw,
                raw_sha256=sha256_bytes(raw),
                decoded_rgb_sha256=rgb_hash,
                width=width,
                height=height,
                source_container=str(test_path),
                source_member=f"row={dataset_index}",
            )
        )

    ranked_by_class: dict[int, list[tuple[str, str, int, Mapping[str, Any]]]] = {0: [], 1: [], 2: []}
    all_rank_hashes: set[str] = set()
    for dataset_index, row in enumerate(train_rows):
        class_id = int(row["labels"])
        payload = {
            "class_id": class_id,
            "dataset_index": dataset_index,
            "namespace": "beans_activation_rank_v1",
            "seed": BEANS_SELECTION_SEED,
        }
        canonical, digest = canonical_rank(payload)
        if digest in all_rank_hashes:
            raise RuntimeError(f"Beans full SHA-256 rank collision: {digest}")
        all_rank_hashes.add(digest)
        ranked_by_class[class_id].append((digest, canonical, dataset_index, row))

    partition_ordinals = {"A": 0, "B": 0}
    for class_id in range(3):
        ranked = sorted(ranked_by_class[class_id], key=lambda value: value[0])
        if len(ranked) < 84:
            raise RuntimeError(f"Beans class {class_id} has only {len(ranked)} train rows")
        for class_rank, (digest, canonical, dataset_index, row) in enumerate(ranked[:84]):
            partition = "A" if class_rank <= 41 else "B"
            raw = _image_payload_bytes(row["image"])
            rgb_hash, width, height = decoded_rgb_hash(raw, runtime)
            result.append(
                SampleRecord(
                    panel_id="beans",
                    partition=partition,
                    ordinal=partition_ordinals[partition],
                    stable_id=f"beans:train:{dataset_index}",
                    dataset_index=dataset_index,
                    class_id=class_id,
                    class_name=class_names[class_id],
                    file_name=str(row["image_file_path"]),
                    reference_text="",
                    canonical_rank_json=canonical,
                    rank_sha256=digest,
                    raw_bytes=raw,
                    raw_sha256=sha256_bytes(raw),
                    decoded_rgb_sha256=rgb_hash,
                    width=width,
                    height=height,
                    source_container=str(train_path),
                    source_member=f"row={dataset_index}",
                )
            )
            partition_ordinals[partition] += 1
    counts = {partition: sum(row.partition == partition for row in result) for partition in ("quality", "A", "B")}
    if counts != {"quality": 128, "A": 126, "B": 126}:
        raise RuntimeError(f"Beans partition count failure: {counts}")
    validate_hash_collisions(result, context="Beans quality/A/B")
    logger.info("stage=beans_selection_complete counts=%s classes=%s", counts, class_names)
    return result, class_names


def prepare_sroie_samples(runtime: Runtime, seal: ResearchSeal, logger: logging.Logger) -> list[SampleRecord]:
    test_zip = SROIE_DATA / "test.zip"
    seal.guard_path(test_zip, operation="sroie_test_zip")
    logger.info("stage=sroie_selection_load archive=%s", test_zip)
    with zipfile.ZipFile(test_zip, "r") as archive:
        metadata_bytes = archive.read("test/metadata.jsonl")
        metadata_rows = [json.loads(line) for line in metadata_bytes.decode("utf-8").splitlines() if line]
        if len(metadata_rows) != 18_704:
            raise RuntimeError(f"SROIE test metadata row drift: {len(metadata_rows)}")
        filenames = [str(row["file_name"]) for row in metadata_rows]
        if len(set(filenames)) != len(filenames):
            raise RuntimeError("SROIE metadata has duplicate file names")
        ranked: list[tuple[str, str, str, str]] = []
        rank_hashes: set[str] = set()
        for row in metadata_rows:
            file_name = str(row["file_name"])
            reference = str(row["text"])
            payload = {
                "file_name": file_name,
                "namespace": "sroie_partition_v1",
                "reference_text": reference,
                "seed": SROIE_PARTITION_SEED,
            }
            canonical, digest = canonical_rank(payload)
            if digest in rank_hashes:
                raise RuntimeError(f"SROIE full SHA-256 partition collision: {digest}")
            rank_hashes.add(digest)
            ranked.append((digest, canonical, file_name, reference))
        ranked.sort(key=lambda value: value[0])
        selected = ranked[:1280]
        result: list[SampleRecord] = []
        partition_ordinals = {"quality": 0, "A": 0, "B": 0}
        for global_rank, (digest, canonical, file_name, reference) in enumerate(selected):
            if global_rank <= 1023:
                partition = "quality"
            elif global_rank <= 1151:
                partition = "A"
            else:
                partition = "B"
            member = f"test/{file_name}"
            raw = archive.read(member)
            rgb_hash, width, height = decoded_rgb_hash(raw, runtime)
            result.append(
                SampleRecord(
                    panel_id="trocr_sroie",
                    partition=partition,
                    ordinal=partition_ordinals[partition],
                    stable_id=f"sroie:test:{file_name}",
                    dataset_index=None,
                    class_id=None,
                    class_name="",
                    file_name=file_name,
                    reference_text=reference,
                    canonical_rank_json=canonical,
                    rank_sha256=digest,
                    raw_bytes=raw,
                    raw_sha256=sha256_bytes(raw),
                    decoded_rgb_sha256=rgb_hash,
                    width=width,
                    height=height,
                    source_container=str(test_zip),
                    source_member=member,
                )
            )
            partition_ordinals[partition] += 1
    counts = {partition: sum(row.partition == partition for row in result) for partition in ("quality", "A", "B")}
    if counts != {"quality": 1024, "A": 128, "B": 128}:
        raise RuntimeError(f"SROIE partition count failure: {counts}")
    validate_hash_collisions(result, context="SROIE quality/A/B")
    logger.info("stage=sroie_selection_complete counts=%s", counts)
    return result


def token_manifest_rows(
    records: Sequence[SampleRecord],
    *,
    panel_id: str,
    patch_token_max: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[int]]]:
    rows: list[dict[str, Any]] = []
    selected_map: dict[tuple[str, str], list[int]] = {}
    for sample in records:
        if sample.partition not in SCORE_SPLITS:
            continue
        candidates: list[tuple[str, str, int]] = []
        digests: set[str] = set()
        for token_index in range(1, patch_token_max + 1):
            payload = {
                "image_id": sample.stable_id,
                "namespace": "activation_token_rank_v1",
                "panel": panel_id,
                "score_split": sample.partition,
                "token_index": token_index,
            }
            canonical, digest = canonical_rank(payload)
            if digest in digests:
                raise RuntimeError(
                    f"token-rank SHA-256 collision panel={panel_id} image={sample.stable_id} digest={digest}"
                )
            digests.add(digest)
            candidates.append((digest, canonical, token_index))
        selected = sorted(candidates, key=lambda value: value[0])[:7]
        indices = [0] + [token_index for _digest, _canonical, token_index in selected]
        if len(indices) != 8 or len(set(indices)) != 8:
            raise RuntimeError(f"invalid selected tokens for {sample.stable_id}: {indices}")
        selected_map[(sample.partition, sample.stable_id)] = indices
        rows.append(
            {
                "panel_id": panel_id,
                "score_split": sample.partition,
                "sample_ordinal": sample.ordinal,
                "stable_id": sample.stable_id,
                "token_slot": 0,
                "token_index": 0,
                "is_cls": True,
                "canonical_rank_json": "",
                "rank_sha256": "",
            }
        )
        for slot, (digest, canonical, token_index) in enumerate(selected, start=1):
            rows.append(
                {
                    "panel_id": panel_id,
                    "score_split": sample.partition,
                    "sample_ordinal": sample.ordinal,
                    "stable_id": sample.stable_id,
                    "token_slot": slot,
                    "token_index": token_index,
                    "is_cls": False,
                    "canonical_rank_json": canonical,
                    "rank_sha256": digest,
                }
            )
    expected_images = 252 if panel_id == "beans" else 256
    if len(selected_map) != expected_images or len(rows) != expected_images * 8:
        raise RuntimeError(
            f"token manifest count failure panel={panel_id}: images={len(selected_map)} rows={len(rows)}"
        )
    return rows, selected_map


def extract_candidate_matrices(runtime: Runtime, logger: logging.Logger) -> dict[str, dict[str, MatrixRecord]]:
    torch = runtime.torch
    result: dict[str, dict[str, MatrixRecord]] = {panel_id: {} for panel_id in PANEL_IDS}

    logger.info("stage=matrix_extract panel=beans checkpoint=%s", BEANS_MODEL / "pytorch_model.bin")
    raw_payload = torch.load(
        BEANS_MODEL / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    beans_state = raw_payload.get("state_dict", raw_payload) if isinstance(raw_payload, dict) else raw_payload
    if not isinstance(beans_state, Mapping):
        raise TypeError(f"unsupported Beans checkpoint payload: {type(beans_state)}")
    for depth in range(12):
        for role, suffix in ROLES:
            key = matrix_key(depth, role)
            state_key = f"vit.encoder.layer.{depth}.{suffix}.weight"
            if state_key not in beans_state or not torch.is_tensor(beans_state[state_key]):
                raise KeyError(f"missing Beans core tensor: {state_key}")
            weight = beans_state[state_key].detach().to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
            expected = EXPECTED_SHAPE_BY_ROLE[role]
            if tuple(weight.shape) != expected:
                raise RuntimeError(f"Beans shape mismatch {key}: {tuple(weight.shape)} != {expected}")
            if not torch.isfinite(weight).all() or torch.count_nonzero(weight).item() == 0:
                raise RuntimeError(f"Beans invalid core tensor: {key}")
            result["beans"][key] = MatrixRecord(
                panel_id="beans",
                matrix_key=key,
                depth=depth,
                role=role,
                state_key=state_key,
                module_name=f"vit.encoder.layer.{depth}.{suffix}",
                weight=weight,
                weight_shape_bytes_sha256=shape_bytes_weight_sha256(weight, runtime),
            )
    del beans_state, raw_payload

    logger.info("stage=matrix_extract panel=trocr_sroie checkpoint=%s", TROCR_MODEL / "model.safetensors")
    with runtime.safe_open(str(TROCR_MODEL / "model.safetensors"), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for depth in range(12):
            for role, suffix in ROLES:
                key = matrix_key(depth, role)
                state_key = f"encoder.encoder.layer.{depth}.{suffix}.weight"
                if state_key not in available:
                    raise KeyError(f"missing TrOCR core tensor: {state_key}")
                raw = handle.get_tensor(state_key)
                weight = raw.detach().to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
                expected = EXPECTED_SHAPE_BY_ROLE[role]
                if tuple(weight.shape) != expected:
                    raise RuntimeError(f"TrOCR shape mismatch {key}: {tuple(weight.shape)} != {expected}")
                if not torch.isfinite(weight).all() or torch.count_nonzero(weight).item() == 0:
                    raise RuntimeError(f"TrOCR invalid core tensor: {key}")
                result["trocr_sroie"][key] = MatrixRecord(
                    panel_id="trocr_sroie",
                    matrix_key=key,
                    depth=depth,
                    role=role,
                    state_key=state_key,
                    module_name=f"encoder.layer.{depth}.{suffix}",
                    weight=weight,
                    weight_shape_bytes_sha256=shape_bytes_weight_sha256(weight, runtime),
                )

    required = expected_matrix_keys()
    for panel_id, records in result.items():
        if set(records) != required or len(records) != 72:
            raise RuntimeError(
                f"candidate geometry grid failure panel={panel_id} "
                f"missing={sorted(required - set(records))} extra={sorted(set(records) - required)}"
            )
    logger.info("stage=matrix_extract_complete panels=%d matrices_per_panel=72 dtype=float32", len(result))
    return result


def _load_ae_weight(runtime: Runtime, source: Mapping[str, Any], seal: ResearchSeal) -> Any:
    path = AE_BANK / str(source["weight_path"])
    seal.guard_path(path, operation="ae_bank_weight")
    payload = runtime.torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    weight = payload.get("weight") if isinstance(payload, Mapping) else None
    if not runtime.torch.is_tensor(weight):
        raise TypeError(f"invalid AE-bank weight payload: {path}")
    value = weight.detach().to(device="cpu", dtype=runtime.torch.float32).contiguous()
    if tuple(value.shape) != tuple(int(item) for item in source["weight_shape"]):
        raise RuntimeError(
            f"AE-bank weight shape drift source={source['source_key']}: "
            f"{tuple(value.shape)} != {tuple(source['weight_shape'])}"
        )
    return value


def audit_unseen_weights(
    *,
    runtime: Runtime,
    seal: ResearchSeal,
    candidates: Mapping[str, Mapping[str, MatrixRecord]],
    device: Any,
    ae_chunk_size: int,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    sources_payload = json.loads((AE_BANK / "sources.json").read_text(encoding="utf-8"))
    sources = list(sources_payload["sources"])
    compatible = [source for source in sources if tuple(source["weight_shape"]) in COMPATIBLE_SHAPES]
    if len(sources) != 1114 or len(compatible) != 471:
        raise RuntimeError(f"AE-bank source-count drift total={len(sources)} compatible={len(compatible)}")
    logger.info(
        "stage=unseen_exact_hash_start training_sources=%d compatible_sources=%d candidate_matrices=%d",
        len(sources),
        len(compatible),
        sum(len(values) for values in candidates.values()),
    )

    source_hashes: dict[str, list[Mapping[str, Any]]] = {}
    source_formal_hash: dict[str, str] = {}
    training_bank_hash_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, source in enumerate(compatible, start=1):
        weight = _load_ae_weight(runtime, source, seal)
        digest = shape_bytes_weight_sha256(weight, runtime)
        source_hashes.setdefault(digest, []).append(source)
        source_formal_hash[str(source["source_key"])] = digest
        weight_path = AE_BANK / str(source["weight_path"])
        actual_weight_file_bytes = weight_path.stat().st_size
        declared_weight_file_bytes = int(source["weight_size_bytes"])
        if actual_weight_file_bytes != declared_weight_file_bytes:
            raise RuntimeError(
                f"AE-bank weight file-size drift source={source['source_key']}: "
                f"{actual_weight_file_bytes} != {declared_weight_file_bytes}"
            )
        training_bank_hash_rows.append(
            {
                "source_key": str(source["source_key"]),
                "model_name": str(source["model_name"]),
                "layer_name": str(source["layer_name"]),
                "d_in": int(weight.shape[0]),
                "d_out": int(weight.shape[1]),
                "weight_path": str(source["weight_path"]),
                "weight_file_bytes": actual_weight_file_bytes,
                "declared_weight_size_bytes": declared_weight_file_bytes,
                "weight_shape_bytes_sha256": digest,
            }
        )
        if index % 50 == 0 or index == len(compatible):
            elapsed = time.monotonic() - started
            logger.info(
                "stage=unseen_exact_hash progress=%d/%d rate=%.1f_weights_s elapsed_s=%.1f",
                index,
                len(compatible),
                index / max(elapsed, 1e-9),
                elapsed,
            )

    candidate_list = [record for panel in PANEL_IDS for record in candidates[panel].values()]
    exact_matches: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in candidate_list:
        exact_matches[(record.panel_id, record.matrix_key)] = list(
            source_hashes.get(record.weight_shape_bytes_sha256, [])
        )
    exact_overlap_count = sum(len(values) for values in exact_matches.values())
    if exact_overlap_count != 0:
        raise RuntimeError(f"candidate weights overlap AE bank exactly: matches={exact_overlap_count}")

    logger.info(
        "stage=proximity_start contract=FP64_cosine_and_relative_frobenius denominator=candidate_norm "
        "device=%s chunk=%d",
        device,
        ae_chunk_size,
    )
    proximity: dict[tuple[str, str], dict[str, Any]] = {}
    for shape in sorted(COMPATIBLE_SHAPES):
        candidate_group = [record for record in candidate_list if tuple(record.weight.shape) == shape]
        source_group = [source for source in compatible if tuple(source["weight_shape"]) == shape]
        if not candidate_group or not source_group:
            raise RuntimeError(f"empty proximity group shape={shape}")
        candidate_stack = runtime.torch.stack(
            [record.weight.reshape(-1) for record in candidate_group], dim=0
        ).to(device=device, dtype=runtime.torch.float64)
        candidate_norm_sq = (candidate_stack * candidate_stack).sum(dim=1)
        candidate_norm = candidate_norm_sq.sqrt()
        if not runtime.torch.isfinite(candidate_norm).all() or (candidate_norm <= 0).any():
            raise RuntimeError(f"invalid candidate norms for shape={shape}")
        best_cosine = runtime.torch.full(
            (len(candidate_group),), -runtime.torch.inf, dtype=runtime.torch.float64, device=device
        )
        best_cosine_index = runtime.torch.full(
            (len(candidate_group),), -1, dtype=runtime.torch.long, device=device
        )
        best_relative = runtime.torch.full(
            (len(candidate_group),), runtime.torch.inf, dtype=runtime.torch.float64, device=device
        )
        best_relative_index = runtime.torch.full(
            (len(candidate_group),), -1, dtype=runtime.torch.long, device=device
        )
        group_started = time.monotonic()
        for begin in range(0, len(source_group), ae_chunk_size):
            end = min(begin + ae_chunk_size, len(source_group))
            bank_cpu = runtime.torch.stack(
                [_load_ae_weight(runtime, source, seal).reshape(-1) for source in source_group[begin:end]],
                dim=0,
            )
            bank = bank_cpu.to(device=device, dtype=runtime.torch.float64)
            bank_norm_sq = (bank * bank).sum(dim=1)
            bank_norm = bank_norm_sq.sqrt()
            if not runtime.torch.isfinite(bank_norm).all() or (bank_norm <= 0).any():
                raise RuntimeError(f"invalid AE-bank norm shape={shape} chunk={begin}:{end}")
            dots = candidate_stack @ bank.transpose(0, 1)
            cosines = dots / (candidate_norm[:, None] * bank_norm[None, :])
            distance_sq = (
                candidate_norm_sq[:, None] + bank_norm_sq[None, :] - 2.0 * dots
            ).clamp_min(0.0)
            relative = distance_sq.sqrt() / candidate_norm[:, None]
            chunk_cosine, chunk_cosine_local = cosines.max(dim=1)
            update_cosine = chunk_cosine > best_cosine
            best_cosine = runtime.torch.where(update_cosine, chunk_cosine, best_cosine)
            best_cosine_index = runtime.torch.where(
                update_cosine, chunk_cosine_local + begin, best_cosine_index
            )
            chunk_relative, chunk_relative_local = relative.min(dim=1)
            update_relative = chunk_relative < best_relative
            best_relative = runtime.torch.where(update_relative, chunk_relative, best_relative)
            best_relative_index = runtime.torch.where(
                update_relative, chunk_relative_local + begin, best_relative_index
            )
            del bank_cpu, bank, bank_norm_sq, bank_norm, dots, cosines, distance_sq, relative
            if begin == 0 or end == len(source_group) or (begin // ae_chunk_size + 1) % 10 == 0:
                elapsed = time.monotonic() - group_started
                logger.info(
                    "stage=proximity shape=%sx%s bank=%d/%d candidate=%d rate=%.1f_pairs_s elapsed_s=%.1f",
                    shape[0],
                    shape[1],
                    end,
                    len(source_group),
                    len(candidate_group),
                    (end * len(candidate_group)) / max(elapsed, 1e-9),
                    elapsed,
                )
        cosine_values = best_cosine.detach().cpu().tolist()
        cosine_indices = best_cosine_index.detach().cpu().tolist()
        relative_values = best_relative.detach().cpu().tolist()
        relative_indices = best_relative_index.detach().cpu().tolist()
        for index, record in enumerate(candidate_group):
            cosine_source = source_group[int(cosine_indices[index])]
            relative_source = source_group[int(relative_indices[index])]
            proximity[(record.panel_id, record.matrix_key)] = {
                "nearest_cosine": float(cosine_values[index]),
                "nearest_cosine_source_key": str(cosine_source["source_key"]),
                "nearest_cosine_model_name": str(cosine_source["model_name"]),
                "nearest_cosine_layer_name": str(cosine_source["layer_name"]),
                "nearest_relative_frobenius": float(relative_values[index]),
                "nearest_relative_frobenius_source_key": str(relative_source["source_key"]),
                "nearest_relative_frobenius_model_name": str(relative_source["model_name"]),
                "nearest_relative_frobenius_layer_name": str(relative_source["layer_name"]),
            }
        del candidate_stack, candidate_norm_sq, candidate_norm
        if device.type == "cuda":
            runtime.torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for record in sorted(candidate_list, key=lambda item: (item.panel_id, item.depth, item.role)):
        matches = exact_matches[(record.panel_id, record.matrix_key)]
        values = proximity[(record.panel_id, record.matrix_key)]
        rows.append(
            {
                "panel_id": record.panel_id,
                "matrix_key": record.matrix_key,
                "depth": record.depth,
                "role": record.role,
                "state_key": record.state_key,
                "d_in": int(record.weight.shape[0]),
                "d_out": int(record.weight.shape[1]),
                "weight_shape_bytes_sha256": record.weight_shape_bytes_sha256,
                "exact_overlap_count": len(matches),
                "exact_overlap_source_keys": ";".join(str(value["source_key"]) for value in matches),
                **values,
            }
        )
    summary = {
        "contract": {
            "exact_hash": "SHA256(struct.pack('<QQ',d_in,d_out)+b'\\0'+contiguous little-endian FP32 bytes)",
            "cosine": "FP64 dot(W_candidate,W_bank)/(norm_candidate*norm_bank)",
            "relative_frobenius": "FP64 ||W_candidate-W_bank||_F / ||W_candidate||_F",
            "proximity_is_non_gating": True,
        },
        "training_bank_total_count": len(sources),
        "training_bank_compatible_count": len(compatible),
        "training_bank_unique_formal_hash_count": len(source_hashes),
        "training_bank_hash_manifest_row_count": len(training_bank_hash_rows),
        "candidate_matrix_count": len(candidate_list),
        "exact_overlap_count": exact_overlap_count,
        "panels": {
            panel_id: {
                "matrix_count": len(candidates[panel_id]),
                "exact_overlap_count": sum(
                    len(exact_matches[(panel_id, key)]) for key in candidates[panel_id]
                ),
                "unseen_weight_pass": all(
                    not exact_matches[(panel_id, key)] for key in candidates[panel_id]
                ),
            }
            for panel_id in PANEL_IDS
        },
    }
    logger.info(
        "stage=unseen_complete candidate_matrices=%d compatible_bank=%d exact_overlap=%d",
        len(candidate_list),
        len(compatible),
        exact_overlap_count,
    )
    if len(training_bank_hash_rows) != 471 or len(source_formal_hash) != 471:
        raise RuntimeError(
            f"training-bank formal-hash manifest incomplete: "
            f"rows={len(training_bank_hash_rows)} unique_sources={len(source_formal_hash)}"
        )
    return rows, summary, training_bank_hash_rows


def _jsonable(value: Any) -> Any:
    """Convert model-loading diagnostics to a stable JSON-compatible tree."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def validate_loading_info(panel_id: str, loading_info: Mapping[str, Any]) -> dict[str, Any]:
    """Require strict checkpoint loading, allowing only TrOCR's unused ViT pooler."""
    missing = sorted(str(value) for value in loading_info.get("missing_keys", []))
    unexpected = sorted(str(value) for value in loading_info.get("unexpected_keys", []))
    mismatched_raw = loading_info.get("mismatched_keys", [])
    mismatched = list(mismatched_raw or [])
    errors = [str(value) for value in loading_info.get("error_msgs", [])]
    allowed_missing: list[set[str]] = [set()]
    if panel_id == "trocr_sroie":
        allowed_missing.extend(
            [
                {"encoder.pooler.dense.bias", "encoder.pooler.dense.weight"},
                {"encoder.encoder.pooler.dense.bias", "encoder.encoder.pooler.dense.weight"},
            ]
        )
    if set(missing) not in allowed_missing or unexpected or mismatched or errors:
        raise RuntimeError(
            f"non-strict model load panel={panel_id}: missing={missing} "
            f"unexpected={unexpected} mismatched={mismatched} errors={errors}"
        )
    return {
        "panel_id": panel_id,
        "missing_keys": missing,
        "allowed_missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": _jsonable(mismatched),
        "error_msgs": errors,
        "strict_load_pass": True,
        "raw_loading_info": _jsonable(loading_info),
    }


def processor_snapshot(processor: Any) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", processor)
    fields = (
        "do_resize",
        "size",
        "resample",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "image_mean",
        "image_std",
        "do_convert_rgb",
    )
    return {
        "class": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "image_processor_class": (
            f"{image_processor.__class__.__module__}.{image_processor.__class__.__name__}"
        ),
        "settings": {
            field: _jsonable(getattr(image_processor, field))
            for field in fields
            if hasattr(image_processor, field)
        },
    }


def _chunks(values: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for begin in range(0, len(values), batch_size):
        yield values[begin : begin + batch_size]


def _to_device_batch(batch: Mapping[str, Any], device: Any, runtime: Runtime) -> dict[str, Any]:
    return {
        key: value.to(device=device) if runtime.torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _resolve_module(root: Any, dotted_name: str) -> Any:
    current = root
    for component in dotted_name.split("."):
        if component.isdigit():
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


def verify_runtime_matrix_parity(
    *,
    panel_id: str,
    model_root: Any,
    matrices: Mapping[str, MatrixRecord],
    runtime: Runtime,
) -> dict[str, Any]:
    """Bind checkpoint extraction to the exact Linear objects used for hooks."""
    checked: list[dict[str, Any]] = []
    for key in sorted(matrices):
        record = matrices[key]
        module = _resolve_module(model_root, record.module_name)
        if not isinstance(module, runtime.torch.nn.Linear):
            raise TypeError(
                f"runtime module is not Linear panel={panel_id} key={key}: {type(module)}"
            )
        runtime_weight = (
            module.weight.detach()
            .to(device="cpu", dtype=runtime.torch.float32)
            .transpose(0, 1)
            .contiguous()
        )
        digest = shape_bytes_weight_sha256(runtime_weight, runtime)
        if digest != record.weight_shape_bytes_sha256 or not runtime.torch.equal(
            runtime_weight, record.weight
        ):
            raise RuntimeError(
                f"runtime/checkpoint matrix parity failed panel={panel_id} key={key} "
                f"expected={record.weight_shape_bytes_sha256} actual={digest}"
            )
        checked.append(
            {
                "matrix_key": key,
                "module_name": record.module_name,
                "weight_shape_bytes_sha256": digest,
                "shape": list(runtime_weight.shape),
            }
        )
    if len(checked) != 72:
        raise RuntimeError(f"runtime parity count failure panel={panel_id}: {len(checked)}")
    return {"panel_id": panel_id, "matrix_count": len(checked), "pass": True, "matrices": checked}


def evaluate_beans_quality(
    *,
    model: Any,
    processor: Any,
    records: Sequence[SampleRecord],
    class_names: Sequence[str],
    batch_size: int,
    device: Any,
    runtime: Runtime,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality = sorted((row for row in records if row.partition == "quality"), key=lambda row: row.ordinal)
    if len(quality) != 128:
        raise RuntimeError(f"Beans quality count failure: {len(quality)}")
    id2label = getattr(model.config, "id2label", {})
    model_names = [str(id2label.get(index, id2label.get(str(index), ""))) for index in range(3)]
    label_agreement = list(class_names) == model_names
    prediction_rows: list[dict[str, Any]] = []
    correct = 0
    predicted_labels: set[int] = set()
    reference_labels: set[int] = set()
    started = time.monotonic()
    with runtime.torch.inference_mode():
        for batch_index, batch_records in enumerate(_chunks(quality, batch_size), start=1):
            images = [decode_rgb(row.raw_bytes, runtime) for row in batch_records]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device)
            logits = model(pixel_values=pixel_values).logits
            if tuple(logits.shape) != (len(batch_records), 3) or not runtime.torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Beans non-finite/wrong logits batch={batch_index}: shape={tuple(logits.shape)}"
                )
            logits_cpu = logits.detach().to(device="cpu", dtype=runtime.torch.float32)
            predictions = logits_cpu.argmax(dim=1).tolist()
            for row, predicted, row_logits in zip(batch_records, predictions, logits_cpu.tolist(), strict=True):
                reference = int(row.class_id)
                predicted = int(predicted)
                correct += int(predicted == reference)
                reference_labels.add(reference)
                predicted_labels.add(predicted)
                prediction_rows.append(
                    {
                        "ordinal": row.ordinal,
                        "stable_id": row.stable_id,
                        "dataset_index": row.dataset_index,
                        "file_name": row.file_name,
                        "raw_sha256": row.raw_sha256,
                        "decoded_rgb_sha256": row.decoded_rgb_sha256,
                        "reference_class_id": reference,
                        "reference_class_name": class_names[reference],
                        "predicted_class_id": predicted,
                        "predicted_class_name": model_names[predicted],
                        "correct": bool(predicted == reference),
                        "logit_0": float(row_logits[0]),
                        "logit_1": float(row_logits[1]),
                        "logit_2": float(row_logits[2]),
                    }
                )
            elapsed = time.monotonic() - started
            logger.info(
                "stage=quality panel=beans batch=%d/%d examples=%d/%d top1=%.6f "
                "rate=%.1f_examples_s elapsed_s=%.1f",
                batch_index,
                math.ceil(len(quality) / batch_size),
                len(prediction_rows),
                len(quality),
                correct / len(prediction_rows),
                len(prediction_rows) / max(elapsed, 1e-9),
                elapsed,
            )
    top1 = correct / len(quality)
    metrics = {
        "panel_id": "beans",
        "quality_split": "test_all_128",
        "sample_count": len(quality),
        "correct_count": correct,
        "top1_accuracy": top1,
        "top1_gate": BEANS_QUALITY_GATE,
        "dataset_class_names": list(class_names),
        "model_id2label": model_names,
        "class_label_agreement": label_agreement,
        "reference_class_ids": sorted(reference_labels),
        "predicted_class_ids": sorted(predicted_labels),
        "reference_class_coverage": len(reference_labels),
        "predicted_class_coverage": len(predicted_labels),
        "finite_logits": True,
    }
    metrics["quality_gate_pass"] = bool(
        top1 >= BEANS_QUALITY_GATE
        and label_agreement
        and reference_labels == {0, 1, 2}
        and predicted_labels == {0, 1, 2}
    )
    logger.info(
        "stage=quality_complete panel=beans pass=%s top1=%.6f correct=%d/%d "
        "reference_coverage=%d/3 predicted_coverage=%d/3 label_agreement=%s",
        metrics["quality_gate_pass"],
        top1,
        correct,
        len(quality),
        len(reference_labels),
        len(predicted_labels),
        label_agreement,
    )
    return prediction_rows, metrics


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_character in enumerate(right, start=1):
        current = [row_index]
        for column_index, left_character in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _effective_generated_ids(ids: Sequence[int], *, eos_id: int, pad_id: int) -> list[int]:
    result: list[int] = []
    for index, raw_value in enumerate(ids):
        value = int(raw_value)
        result.append(value)
        if index > 0 and value == eos_id:
            break
    while result and result[-1] == pad_id:
        result.pop()
    return result


def evaluate_trocr_quality(
    *,
    model: Any,
    processor: Any,
    records: Sequence[SampleRecord],
    batch_size: int,
    device: Any,
    runtime: Runtime,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality = sorted((row for row in records if row.partition == "quality"), key=lambda row: row.ordinal)
    if len(quality) != 1024:
        raise RuntimeError(f"SROIE quality count failure: {len(quality)}")
    frozen_ids = {
        "decoder_start_token_id": 2,
        "eos_token_id": 2,
        "pad_token_id": 1,
    }
    found_ids = {
        "decoder_start_token_id": int(model.generation_config.decoder_start_token_id),
        "eos_token_id": int(model.generation_config.eos_token_id),
        "pad_token_id": int(model.generation_config.pad_token_id),
    }
    if found_ids != frozen_ids:
        raise RuntimeError(f"TrOCR generation-ID drift: expected={frozen_ids} found={found_ids}")

    prediction_rows: list[dict[str, Any]] = []
    total_edits = 0
    total_reference_characters = 0
    exact_count = 0
    nonempty_count = 0
    example_cers: list[float] = []
    started = time.monotonic()
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 96,
        "use_cache": False,
        "return_dict_in_generate": False,
        "output_scores": False,
    }
    with runtime.torch.inference_mode():
        for batch_index, batch_records in enumerate(_chunks(quality, batch_size), start=1):
            images = [decode_rgb(row.raw_bytes, runtime) for row in batch_records]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device)
            generated = model.generate(pixel_values=pixel_values, **generation_kwargs)
            if not runtime.torch.is_tensor(generated) or generated.ndim != 2:
                raise RuntimeError(f"TrOCR generate returned invalid value: {type(generated)}")
            if generated.shape[0] != len(batch_records) or generated.shape[1] > 97:
                raise RuntimeError(
                    f"TrOCR generation shape failure batch={batch_index}: {tuple(generated.shape)}"
                )
            generated_cpu = generated.detach().to(device="cpu", dtype=runtime.torch.long)
            predictions = processor.batch_decode(
                generated_cpu,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row, prediction, ids_tensor in zip(
                batch_records, predictions, generated_cpu, strict=True
            ):
                ids = [int(value) for value in ids_tensor.tolist()]
                effective_ids = _effective_generated_ids(ids, eos_id=2, pad_id=1)
                normalized_reference = normalize_ocr_text(row.reference_text)
                normalized_prediction = normalize_ocr_text(str(prediction))
                if not normalized_reference:
                    raise RuntimeError(f"empty normalized SROIE reference: {row.file_name}")
                edits = levenshtein_distance(normalized_reference, normalized_prediction)
                denominator = len(normalized_reference)
                per_example_cer = edits / denominator
                total_edits += edits
                total_reference_characters += denominator
                exact = normalized_reference == normalized_prediction
                nonempty = bool(normalized_prediction)
                exact_count += int(exact)
                nonempty_count += int(nonempty)
                example_cers.append(per_example_cer)
                prediction_rows.append(
                    {
                        "ordinal": row.ordinal,
                        "stable_id": row.stable_id,
                        "file_name": row.file_name,
                        "raw_sha256": row.raw_sha256,
                        "decoded_rgb_sha256": row.decoded_rgb_sha256,
                        "reference_text": row.reference_text,
                        "prediction_text": str(prediction),
                        "normalized_reference": normalized_reference,
                        "normalized_prediction": normalized_prediction,
                        "generated_token_ids_json": json.dumps(ids, separators=(",", ":")),
                        "effective_generated_token_ids_json": json.dumps(
                            effective_ids, separators=(",", ":")
                        ),
                        "generated_length": len(ids),
                        "effective_generated_length": len(effective_ids),
                        "edit_distance": edits,
                        "reference_denominator": denominator,
                        "per_example_cer": per_example_cer,
                        "exact_match": exact,
                        "nonempty_prediction": nonempty,
                    }
                )
            elapsed = time.monotonic() - started
            logger.info(
                "stage=quality panel=trocr_sroie batch=%d/%d examples=%d/%d exact=%.6f "
                "corpus_cer=%.6f nonempty=%.6f rate=%.1f_examples_s elapsed_s=%.1f",
                batch_index,
                math.ceil(len(quality) / batch_size),
                len(prediction_rows),
                len(quality),
                exact_count / len(prediction_rows),
                total_edits / total_reference_characters,
                nonempty_count / len(prediction_rows),
                len(prediction_rows) / max(elapsed, 1e-9),
                elapsed,
            )
    nonempty_fraction = nonempty_count / len(quality)
    exact_fraction = exact_count / len(quality)
    corpus_cer = total_edits / total_reference_characters
    median_cer = float(statistics.median(example_cers))
    metrics = {
        "panel_id": "trocr_sroie",
        "quality_split": "canonical_sroie_rank_0_1023",
        "sample_count": len(quality),
        "generation_kwargs": generation_kwargs,
        "generation_ids": found_ids,
        "nonempty_prediction_count": nonempty_count,
        "nonempty_prediction_fraction": nonempty_fraction,
        "nonempty_prediction_gate": SROIE_NONEMPTY_GATE,
        "exact_match_count": exact_count,
        "exact_match_fraction": exact_fraction,
        "exact_match_gate": SROIE_EXACT_GATE,
        "total_edit_distance": total_edits,
        "total_reference_characters": total_reference_characters,
        "corpus_cer": corpus_cer,
        "corpus_cer_gate": SROIE_CORPUS_CER_GATE,
        "median_per_example_cer": median_cer,
        "median_per_example_cer_gate": SROIE_MEDIAN_CER_GATE,
        "all_normalized_references_nonempty": True,
    }
    metrics["quality_gate_pass"] = bool(
        nonempty_fraction >= SROIE_NONEMPTY_GATE
        and exact_fraction >= SROIE_EXACT_GATE
        and corpus_cer <= SROIE_CORPUS_CER_GATE
        and median_cer <= SROIE_MEDIAN_CER_GATE
    )
    logger.info(
        "stage=quality_complete panel=trocr_sroie pass=%s exact=%.6f corpus_cer=%.6f "
        "median_cer=%.6f nonempty=%.6f",
        metrics["quality_gate_pass"],
        exact_fraction,
        corpus_cer,
        median_cer,
        nonempty_fraction,
    )
    return prediction_rows, metrics


def collect_panel_activations(
    *,
    panel_id: str,
    model_root: Any,
    processor: Any,
    records: Sequence[SampleRecord],
    selected_tokens: Mapping[tuple[str, str], Sequence[int]],
    matrices: Mapping[str, MatrixRecord],
    batch_size: int,
    expected_sequence_length: int,
    device: Any,
    runtime: Runtime,
    forward_encoder: Callable[[Any], Any],
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Collect exact pre-Linear rows; Q/K/V aliases share final tensor storage."""
    torch = runtime.torch
    activations: dict[str, dict[str, Any]] = {split: {} for split in SCORE_SPLITS}
    audit_rows: list[dict[str, Any]] = []
    expected_rows = 1008 if panel_id == "beans" else 1024

    for split in SCORE_SPLITS:
        split_records = sorted(
            (row for row in records if row.partition == split), key=lambda row: row.ordinal
        )
        expected_images = 126 if panel_id == "beans" else 128
        if len(split_records) != expected_images:
            raise RuntimeError(
                f"activation image count failure panel={panel_id} split={split}: "
                f"{len(split_records)} != {expected_images}"
            )
        captured: dict[str, list[Any]] = {key: [] for key in matrices}
        invocation_counts: dict[str, int] = {key: 0 for key in matrices}
        batch_context: dict[str, Any] = {"indices": None, "batch_size": 0, "batch_index": 0}
        handles: list[Any] = []

        def make_hook(key: str, expected_width: int) -> Callable[[Any, tuple[Any, ...]], None]:
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise RuntimeError(
                        f"activation hook missing tensor input panel={panel_id} split={split} key={key}"
                    )
                hidden = inputs[0]
                indices = batch_context["indices"]
                if hidden.ndim != 3 or hidden.shape[0] != batch_context["batch_size"]:
                    raise RuntimeError(
                        f"activation hook batch alignment failure panel={panel_id} split={split} "
                        f"key={key} shape={tuple(hidden.shape)} expected_batch={batch_context['batch_size']}"
                    )
                if hidden.shape[1] != expected_sequence_length or hidden.shape[2] != expected_width:
                    raise RuntimeError(
                        f"activation hook geometry failure panel={panel_id} split={split} key={key} "
                        f"shape={tuple(hidden.shape)} expected=(*,{expected_sequence_length},{expected_width})"
                    )
                if indices is None or tuple(indices.shape) != (hidden.shape[0], 8):
                    raise RuntimeError(
                        f"activation token-index alignment failure panel={panel_id} split={split} key={key}"
                    )
                gather_index = indices[:, :, None].expand(-1, -1, expected_width)
                selected = hidden.gather(dim=1, index=gather_index).reshape(-1, expected_width)
                selected_cpu = selected.detach().to(device="cpu", dtype=torch.float32).contiguous()
                if not torch.isfinite(selected_cpu).all() or torch.count_nonzero(selected_cpu).item() == 0:
                    raise RuntimeError(
                        f"non-finite/zero activation panel={panel_id} split={split} key={key} "
                        f"batch={batch_context['batch_index']}"
                    )
                captured[key].append(selected_cpu)
                invocation_counts[key] += 1

            return hook

        for key, record in matrices.items():
            module = _resolve_module(model_root, record.module_name)
            handles.append(
                module.register_forward_pre_hook(make_hook(key, int(record.weight.shape[0])))
            )

        started = time.monotonic()
        processed = 0
        try:
            with torch.inference_mode():
                for batch_index, batch_records in enumerate(
                    _chunks(split_records, batch_size), start=1
                ):
                    images = [decode_rgb(row.raw_bytes, runtime) for row in batch_records]
                    inputs = processor(images=images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device=device)
                    index_rows = [
                        list(selected_tokens[(split, row.stable_id)]) for row in batch_records
                    ]
                    indices = torch.tensor(index_rows, dtype=torch.long, device=device)
                    if int(indices.min().item()) < 0 or int(indices.max().item()) >= expected_sequence_length:
                        raise RuntimeError(
                            f"activation token out of bounds panel={panel_id} split={split} "
                            f"range={int(indices.min().item())}:{int(indices.max().item())}"
                        )
                    before_counts = dict(invocation_counts)
                    batch_context.update(
                        {"indices": indices, "batch_size": len(batch_records), "batch_index": batch_index}
                    )
                    forward_encoder(pixel_values)
                    bad_invocations = {
                        key: invocation_counts[key] - before_counts[key]
                        for key in matrices
                        if invocation_counts[key] - before_counts[key] != 1
                    }
                    if bad_invocations:
                        raise RuntimeError(
                            f"activation hook invocation failure panel={panel_id} split={split} "
                            f"batch={batch_index}: {bad_invocations}"
                        )
                    processed += len(batch_records)
                    elapsed = time.monotonic() - started
                    logger.info(
                        "stage=activation panel=%s split=%s batch=%d/%d images=%d/%d "
                        "rows_per_matrix=%d rate=%.1f_images_s elapsed_s=%.1f",
                        panel_id,
                        split,
                        batch_index,
                        math.ceil(len(split_records) / batch_size),
                        processed,
                        len(split_records),
                        processed * 8,
                        processed / max(elapsed, 1e-9),
                        elapsed,
                    )
        finally:
            for handle in handles:
                handle.remove()

        split_tensors = {key: torch.cat(chunks, dim=0) for key, chunks in captured.items()}
        for key, tensor in split_tensors.items():
            expected_width = int(matrices[key].weight.shape[0])
            if tuple(tensor.shape) != (expected_rows, expected_width):
                raise RuntimeError(
                    f"activation tensor shape failure panel={panel_id} split={split} key={key}: "
                    f"{tuple(tensor.shape)} != {(expected_rows, expected_width)}"
                )
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise RuntimeError(f"activation tensor dtype/device failure: {panel_id}/{split}/{key}")

        # The architecture feeds exactly one hidden-state tensor to Q, K and V.
        # Verify bit identity before retaining a single aliased basis for all three roles.
        for depth in range(12):
            query_key = matrix_key(depth, "attn_query")
            key_key = matrix_key(depth, "attn_key")
            value_key = matrix_key(depth, "attn_value")
            query = split_tensors[query_key]
            key_tensor = split_tensors[key_key]
            value = split_tensors[value_key]
            if not torch.equal(query, key_tensor) or not torch.equal(query, value):
                raise RuntimeError(
                    f"Q/K/V activation input mismatch panel={panel_id} split={split} depth={depth}"
                )
            split_tensors[key_key] = query
            split_tensors[value_key] = query

        activations[split] = split_tensors
        for key in sorted(split_tensors):
            tensor = split_tensors[key]
            audit_rows.append(
                {
                    "panel_id": panel_id,
                    "score_split": split,
                    "matrix_key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "device": tensor.device.type,
                    "finite": bool(torch.isfinite(tensor).all()),
                    "nonzero_count": int(torch.count_nonzero(tensor).item()),
                    "tensor_sha256": tensor_manifest_sha256(tensor, runtime),
                    "qkv_shared_storage": bool(
                        matrices[key].role not in {"attn_key", "attn_value"}
                        or tensor.data_ptr()
                        == split_tensors[matrix_key(matrices[key].depth, "attn_query")].data_ptr()
                    ),
                }
            )

    if set(activations) != set(SCORE_SPLITS):
        raise RuntimeError(f"activation split grid failure panel={panel_id}")
    for split in SCORE_SPLITS:
        if set(activations[split]) != expected_matrix_keys():
            raise RuntimeError(f"activation matrix grid failure panel={panel_id} split={split}")
    gate = {
        "panel_id": panel_id,
        "score_splits": list(SCORE_SPLITS),
        "expected_rows_per_matrix_per_split": expected_rows,
        "matrix_count_per_split": 72,
        "qkv_bit_exact_and_storage_shared": True,
        "finite_nonzero": True,
        "activation_gate_pass": True,
        "tensors": audit_rows,
    }
    logger.info(
        "stage=activation_complete panel=%s splits=A,B matrices_per_split=72 rows_per_matrix=%d pass=true",
        panel_id,
        expected_rows,
    )
    return activations, gate


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def evidence_entry(path: Path, *, output_root: Path, seal: ResearchSeal) -> dict[str, str]:
    relative = path.relative_to(output_root).as_posix()
    return {"path": relative, "sha256": sha256_file(path, seal)}


def checkpoint_sha256(panel_id: str) -> str:
    if panel_id == "beans":
        return next(spec.sha256 for spec in RESOURCE_SPECS if spec.label == "beans_model_weights")
    if panel_id == "trocr_sroie":
        return next(spec.sha256 for spec in RESOURCE_SPECS if spec.label == "trocr_model_weights")
    raise KeyError(panel_id)


def matrix_metadata(matrices: Mapping[str, MatrixRecord]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "depth": record.depth,
            "role": record.role,
            "d_in": int(record.weight.shape[0]),
            "d_out": int(record.weight.shape[1]),
            "module_name": record.module_name,
            "layer_name": record.state_key,
            "state_key": record.state_key,
            "shape": [int(value) for value in record.weight.shape],
            "weight_shape_bytes_sha256": record.weight_shape_bytes_sha256,
            "weight_sha": record.weight_shape_bytes_sha256,
        }
        for key, record in sorted(matrices.items())
    }


def build_tensor_manifest(
    *,
    panel_id: str,
    matrices: Mapping[str, MatrixRecord],
    activations: Mapping[str, Mapping[str, Any]],
    runtime: Runtime,
) -> dict[str, Any]:
    return {
        "schema_version": "prospective_panel_tensor_manifest_v1",
        "panel_id": panel_id,
        "weights": {
            key: {
                "shape": list(record.weight.shape),
                "dtype": "float32",
                "device": "cpu",
                "weight_shape_bytes_sha256": record.weight_shape_bytes_sha256,
                "tensor_manifest_sha256": tensor_manifest_sha256(record.weight, runtime),
            }
            for key, record in sorted(matrices.items())
        },
        "activations": {
            split: {
                key: {
                    "shape": list(tensor.shape),
                    "dtype": "float32",
                    "device": "cpu",
                    "tensor_manifest_sha256": tensor_manifest_sha256(tensor, runtime),
                    "storage_alias": (
                        matrix_key(matrices[key].depth, "attn_query")
                        if matrices[key].role in {"attn_key", "attn_value"}
                        else key
                    ),
                }
                for key, tensor in sorted(split_values.items())
            }
            for split, split_values in sorted(activations.items())
        },
    }


def validate_panel_payload(
    payload: Mapping[str, Any],
    *,
    runtime: Runtime,
    evidence_root: Path,
    verify_evidence: bool = True,
) -> dict[str, Any]:
    torch = runtime.torch
    required_root = {
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
    missing_root = required_root - set(payload)
    if missing_root:
        raise RuntimeError(f"panel payload missing root keys: {sorted(missing_root)}")
    if payload["schema_version"] != "prospective_panel_v1":
        raise RuntimeError(f"panel schema drift: {payload['schema_version']}")
    panel_id = str(payload["panel_id"])
    if panel_id not in PANEL_IDS:
        raise RuntimeError(f"unknown panel ID: {panel_id}")
    if payload["checkpoint_sha256"] != checkpoint_sha256(panel_id):
        raise RuntimeError(f"checkpoint SHA mismatch in panel payload: {panel_id}")
    if payload["quality_status"] != "PASS":
        raise RuntimeError(f"panel quality status is not PASS: {panel_id}")

    weights = payload["weights"]
    activations = payload["activations"]
    metadata = payload["matrix_meta"]
    expected_keys = expected_matrix_keys()
    if set(weights) != expected_keys or set(metadata) != expected_keys:
        raise RuntimeError(f"weight/metadata grid mismatch panel={panel_id}")
    if set(activations) != set(SCORE_SPLITS):
        raise RuntimeError(f"activation split grid mismatch panel={panel_id}")
    expected_rows = 1008 if panel_id == "beans" else 1024
    for key in sorted(expected_keys):
        meta = metadata[key]
        expected_meta_fields = {
            "depth",
            "role",
            "d_in",
            "d_out",
            "module_name",
            "weight_shape_bytes_sha256",
        }
        if expected_meta_fields - set(meta):
            raise RuntimeError(
                f"matrix metadata fields missing panel={panel_id} key={key}: "
                f"{sorted(expected_meta_fields - set(meta))}"
            )
        depth = int(meta["depth"])
        role = str(meta["role"])
        if key != matrix_key(depth, role) or role not in dict(ROLES):
            raise RuntimeError(f"matrix metadata identity mismatch panel={panel_id} key={key}")
        expected_shape = EXPECTED_SHAPE_BY_ROLE[role]
        if (int(meta["d_in"]), int(meta["d_out"])) != expected_shape:
            raise RuntimeError(f"matrix metadata shape mismatch panel={panel_id} key={key}")
        weight = weights[key]
        if (
            not torch.is_tensor(weight)
            or weight.device.type != "cpu"
            or weight.dtype != torch.float32
            or tuple(weight.shape) != expected_shape
            or not torch.isfinite(weight).all()
            or torch.count_nonzero(weight).item() == 0
        ):
            raise RuntimeError(f"invalid panel weight tensor panel={panel_id} key={key}")
        actual_weight_sha = shape_bytes_weight_sha256(weight, runtime)
        if actual_weight_sha != meta["weight_shape_bytes_sha256"]:
            raise RuntimeError(f"panel weight internal SHA mismatch panel={panel_id} key={key}")
        for split in SCORE_SPLITS:
            if set(activations[split]) != expected_keys:
                raise RuntimeError(f"activation key grid mismatch panel={panel_id} split={split}")
            tensor = activations[split][key]
            if (
                not torch.is_tensor(tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != (expected_rows, expected_shape[0])
                or not torch.isfinite(tensor).all()
                or torch.count_nonzero(tensor).item() == 0
            ):
                raise RuntimeError(
                    f"invalid activation tensor panel={panel_id} split={split} key={key}"
                )
    for split in SCORE_SPLITS:
        for depth in range(12):
            query = activations[split][matrix_key(depth, "attn_query")]
            key_tensor = activations[split][matrix_key(depth, "attn_key")]
            value = activations[split][matrix_key(depth, "attn_value")]
            if not torch.equal(query, key_tensor) or not torch.equal(query, value):
                raise RuntimeError(
                    f"panel Q/K/V bit parity failure panel={panel_id} split={split} depth={depth}"
                )
            if query.data_ptr() != key_tensor.data_ptr() or query.data_ptr() != value.data_ptr():
                raise RuntimeError(
                    f"panel Q/K/V storage alias failure panel={panel_id} split={split} depth={depth}"
                )

    build = payload["build_manifest"]
    required_build = {
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
    for field, expected in required_build.items():
        if build.get(field) != expected:
            raise RuntimeError(
                f"build manifest failure panel={panel_id} field={field} "
                f"expected={expected!r} found={build.get(field)!r}"
            )
    evidence_files = build.get("evidence_files")
    if not isinstance(evidence_files, Mapping) or not evidence_files:
        raise RuntimeError(f"empty build evidence map panel={panel_id}")
    if verify_evidence:
        for label, entry in evidence_files.items():
            if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
                raise RuntimeError(f"bad evidence entry panel={panel_id} label={label}: {entry}")
            relative = Path(str(entry["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe evidence path panel={panel_id} label={label}: {relative}")
            evidence_path = evidence_root / relative
            if not evidence_path.is_file() or sha256_file(evidence_path) != entry["sha256"]:
                raise RuntimeError(
                    f"evidence missing/hash mismatch panel={panel_id} label={label}: {evidence_path}"
                )
    if not isinstance(payload["resource_hashes"], Mapping) or not payload["resource_hashes"]:
        raise RuntimeError(f"empty resource hash map panel={panel_id}")
    return {
        "panel_id": panel_id,
        "matrix_count": len(weights),
        "activation_splits": list(SCORE_SPLITS),
        "rows_per_matrix_per_split": expected_rows,
        "pass": True,
    }


def _load_model_with_info(model_class: Any, model_path: Path, runtime: Runtime) -> tuple[Any, dict[str, Any]]:
    loaded = model_class.from_pretrained(
        str(model_path),
        local_files_only=True,
        output_loading_info=True,
        dtype=runtime.torch.float32,
        low_cpu_mem_usage=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError(f"from_pretrained did not return (model, loading_info): {type(loaded)}")
    model, loading_info = loaded
    if not isinstance(loading_info, Mapping):
        raise TypeError(f"invalid loading info: {type(loading_info)}")
    return model, dict(loading_info)


def build_one_panel(
    *,
    panel_id: str,
    records: Sequence[SampleRecord],
    class_names: Sequence[str] | None,
    selected_tokens: Mapping[tuple[str, str], Sequence[int]],
    matrices: Mapping[str, MatrixRecord],
    resource_results: Mapping[str, Mapping[str, Any]],
    unseen_summary: Mapping[str, Any],
    stage_root: Path,
    final_root: Path,
    builder_sha256: str,
    args: argparse.Namespace,
    device: Any,
    runtime: Runtime,
    seal: ResearchSeal,
    logger: logging.Logger,
) -> dict[str, Any]:
    torch = runtime.torch
    panel_dir = stage_root / "panels" / panel_id
    panel_dir.mkdir(parents=True, exist_ok=False)
    logger.info(
        "stage=panel_build_start panel=%s samples=%d matrices=%d output=%s",
        panel_id,
        len(records),
        len(matrices),
        panel_dir,
    )

    sample_rows = [row.manifest_row() for row in sorted(records, key=lambda item: (item.partition, item.ordinal))]
    token_rows, regenerated_token_map = token_manifest_rows(
        records,
        panel_id=panel_id,
        patch_token_max=196 if panel_id == "beans" else 576,
    )
    if {key: list(value) for key, value in regenerated_token_map.items()} != {
        key: list(value) for key, value in selected_tokens.items()
    }:
        raise RuntimeError(f"token-selection regeneration mismatch panel={panel_id}")
    sample_manifest_path = panel_dir / "sample_manifest.jsonl"
    token_manifest_path = panel_dir / "token_manifest.jsonl"
    write_jsonl(sample_manifest_path, sample_rows)
    write_jsonl(token_manifest_path, token_rows)
    sample_manifest_sha = sha256_file(sample_manifest_path, seal)
    token_manifest_sha = sha256_file(token_manifest_path, seal)
    logger.info(
        "stage=selection_manifest panel=%s sample_rows=%d token_rows=%d sample_sha256=%s "
        "token_sha256=%s",
        panel_id,
        len(sample_rows),
        len(token_rows),
        sample_manifest_sha,
        token_manifest_sha,
    )

    if panel_id == "beans":
        processor = runtime.ViTImageProcessor.from_pretrained(
            str(BEANS_MODEL), local_files_only=True
        )
        model, raw_loading_info = _load_model_with_info(
            runtime.ViTForImageClassification, BEANS_MODEL, runtime
        )
        activation_root = model
    else:
        processor = runtime.TrOCRProcessor.from_pretrained(
            str(TROCR_MODEL), local_files_only=True
        )
        model, raw_loading_info = _load_model_with_info(
            runtime.VisionEncoderDecoderModel, TROCR_MODEL, runtime
        )
        activation_root = model.encoder
    loading_audit = validate_loading_info(panel_id, raw_loading_info)
    loading_audit["processor"] = processor_snapshot(processor)
    loading_audit["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
    loading_audit["model_id"] = BEANS_MODEL_ID if panel_id == "beans" else TROCR_MODEL_ID
    loading_audit["revision"] = (
        BEANS_MODEL_REVISION if panel_id == "beans" else TROCR_MODEL_REVISION
    )
    loading_path = panel_dir / "model_loading.json"
    write_json(loading_path, loading_audit)

    parity = verify_runtime_matrix_parity(
        panel_id=panel_id,
        model_root=activation_root,
        matrices=matrices,
        runtime=runtime,
    )
    parity_path = panel_dir / "runtime_matrix_parity.json"
    write_json(parity_path, parity)
    model.eval()
    model.to(device=device, dtype=torch.float32)

    if panel_id == "beans":
        assert class_names is not None
        quality_rows, quality_metrics = evaluate_beans_quality(
            model=model,
            processor=processor,
            records=records,
            class_names=class_names,
            batch_size=args.beans_quality_batch_size,
            device=device,
            runtime=runtime,
            logger=logger,
        )
    else:
        quality_rows, quality_metrics = evaluate_trocr_quality(
            model=model,
            processor=processor,
            records=records,
            batch_size=args.trocr_quality_batch_size,
            device=device,
            runtime=runtime,
            logger=logger,
        )
    quality_predictions_path = panel_dir / "quality_predictions.csv"
    quality_metrics_path = panel_dir / "quality_metrics.json"
    write_csv(quality_predictions_path, quality_rows)
    write_json(quality_metrics_path, quality_metrics)
    if not quality_metrics["quality_gate_pass"]:
        logger.error("stage=panel_invalid panel=%s reason=quality_gate", panel_id)
        if device.type == "cuda":
            model.to(device="cpu")
            torch.cuda.empty_cache()
        return {
            "panel_id": panel_id,
            "panel_valid": False,
            "quality_status": "FAIL",
            "quality_metrics": quality_metrics,
            "panel_dir": str(panel_dir.relative_to(stage_root)),
        }

    def forward_encoder(pixel_values: Any) -> Any:
        if panel_id == "beans":
            return model(pixel_values=pixel_values)
        return model.encoder(pixel_values=pixel_values)

    if panel_id == "beans":
        activation_batch_size = args.beans_activation_batch_size
        expected_sequence_length = 197
    else:
        activation_batch_size = args.trocr_activation_batch_size
        expected_sequence_length = 577
    activations, activation_gate = collect_panel_activations(
        panel_id=panel_id,
        model_root=activation_root,
        processor=processor,
        records=records,
        selected_tokens=selected_tokens,
        matrices=matrices,
        batch_size=activation_batch_size,
        expected_sequence_length=expected_sequence_length,
        device=device,
        runtime=runtime,
        forward_encoder=forward_encoder,
        logger=logger,
    )
    activation_gate_path = panel_dir / "activation_gate.json"
    write_json(activation_gate_path, activation_gate)
    tensor_manifest = build_tensor_manifest(
        panel_id=panel_id,
        matrices=matrices,
        activations=activations,
        runtime=runtime,
    )
    tensor_manifest_path = panel_dir / "panel_tensor_manifest.json"
    write_json(tensor_manifest_path, tensor_manifest)

    global_paths = {
        "resource_hashes": stage_root / "resource_hashes.json",
        "unseen_weight_summary": stage_root / "unseen_weight_summary.json",
        "unseen_weight_audit": stage_root / "unseen_weight_audit.csv",
        "training_bank_compatible_weight_hashes": (
            stage_root / "training_bank_compatible_weight_hashes.csv"
        ),
        "resolved_config": stage_root / "resolved_config.json",
    }
    local_paths = {
        "sample_manifest": sample_manifest_path,
        "token_manifest": token_manifest_path,
        "model_loading": loading_path,
        "runtime_matrix_parity": parity_path,
        "quality_predictions": quality_predictions_path,
        "quality_metrics": quality_metrics_path,
        "activation_gate": activation_gate_path,
        "panel_tensor_manifest": tensor_manifest_path,
    }
    evidence_files = {
        label: evidence_entry(path, output_root=stage_root, seal=seal)
        for label, path in {**global_paths, **local_paths}.items()
    }
    panel_unseen = unseen_summary["panels"][panel_id]
    seal_snapshot = seal.snapshot()
    build_manifest = {
        "schema_version": "prospective_panel_build_manifest_v1",
        "panel_id": panel_id,
        "panel_valid": True,
        "seal_pass": bool(seal_snapshot["pass"]),
        "provenance_pass": bool(loading_audit["strict_load_pass"] and parity["pass"]),
        "geometry_pass": set(matrices) == expected_matrix_keys() and len(matrices) == 72,
        "quality_gate_pass": bool(quality_metrics["quality_gate_pass"]),
        "unseen_weight_pass": bool(panel_unseen["unseen_weight_pass"]),
        "activation_gate_pass": bool(activation_gate["activation_gate_pass"]),
        "matrix_count": len(matrices),
        "exact_overlap_count": int(panel_unseen["exact_overlap_count"]),
        "training_bank_compatible_count": int(unseen_summary["training_bank_compatible_count"]),
        "score_splits": list(SCORE_SPLITS),
        "design_sha256": DESIGN_SHA256,
        "builder_sha256": builder_sha256,
        "checkpoint_sha256": checkpoint_sha256(panel_id),
        "sample_manifest_sha256": sample_manifest_sha,
        "selection_manifest_sha256": sample_manifest_sha,
        "token_manifest_sha256": token_manifest_sha,
        "quality_metrics": quality_metrics,
        "seal": seal_snapshot,
        "evidence_files": evidence_files,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not all(
        build_manifest[field]
        for field in (
            "panel_valid",
            "seal_pass",
            "provenance_pass",
            "geometry_pass",
            "quality_gate_pass",
            "unseen_weight_pass",
            "activation_gate_pass",
        )
    ):
        raise RuntimeError(f"panel validity conjunction failed panel={panel_id}: {build_manifest}")

    payload = {
        "schema_version": "prospective_panel_v1",
        "panel_id": panel_id,
        "checkpoint_sha256": checkpoint_sha256(panel_id),
        "quality_status": "PASS",
        "weights": {key: record.weight for key, record in sorted(matrices.items())},
        "activations": activations,
        "matrix_meta": matrix_metadata(matrices),
        "resource_hashes": {
            label: str(resource_results[label]["sha256"])
            for label in PANEL_RESOURCE_LABELS[panel_id]
        },
        "build_manifest": build_manifest,
        "selection_manifest_sha256": sample_manifest_sha,
        "token_manifest_sha256": token_manifest_sha,
    }
    validate_panel_payload(payload, runtime=runtime, evidence_root=stage_root)
    candidate_path = panel_dir / "panel_candidate.pt"
    torch.save(payload, candidate_path)
    reloaded = torch.load(candidate_path, map_location="cpu", weights_only=True, mmap=True)
    validate_panel_payload(reloaded, runtime=runtime, evidence_root=stage_root)
    del reloaded, payload, activations
    if device.type == "cuda":
        model.to(device="cpu")
        torch.cuda.empty_cache()
    logger.info(
        "stage=panel_candidate_complete panel=%s path=%s bytes=%d quality=PASS",
        panel_id,
        candidate_path,
        candidate_path.stat().st_size,
    )
    return {
        "panel_id": panel_id,
        "panel_valid": True,
        "quality_status": "PASS",
        "quality_metrics": quality_metrics,
        "candidate_path": candidate_path,
        "final_panel_path": Path(os.path.realpath(os.fspath(final_root / "panels" / panel_id / "panel.pt"))),
        "checkpoint_sha256": checkpoint_sha256(panel_id),
    }


def configure_logger(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("prospective_panel_builder")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        fmt="%(asctime)sZ level=%(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def close_file_log(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            logger.removeHandler(handler)
            handler.close()


def build_artifact_manifest(
    root: Path,
    *,
    builder_sha256: str,
    seal: ResearchSeal,
) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    files: list[dict[str, Any]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path, seal),
            }
        )
    manifest = {
        "schema_version": "prospective_panel_artifact_manifest_v1",
        "design_sha256": DESIGN_SHA256,
        "builder_sha256": builder_sha256,
        "file_count": len(files),
        "files": files,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_artifact_manifest(root: Path, seal: ResearchSeal) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed_paths = {str(row["path"]) for row in manifest["files"]}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if listed_paths != actual_paths:
        raise RuntimeError(
            f"artifact manifest completeness failure missing={sorted(actual_paths - listed_paths)} "
            f"extra={sorted(listed_paths - actual_paths)}"
        )
    for row in manifest["files"]:
        path = root / str(row["path"])
        actual_sha = sha256_file(path, seal)
        if path.stat().st_size != int(row["bytes"]) or actual_sha != row["sha256"]:
            raise RuntimeError(f"artifact manifest mismatch: {path}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path, seal),
        "file_count": len(manifest["files"]),
        "pass": True,
    }


def quarantine_panel_files(stage_root: Path) -> None:
    for path in stage_root.glob("panels/*/panel.pt"):
        destination = path.with_name("panel_candidate_NOT_FORMAL.pt")
        if destination.exists():
            destination = path.with_name(
                f"panel_candidate_NOT_FORMAL_{int(time.time_ns())}.pt"
            )
        os.replace(path, destination)
    root_manifest = stage_root / "panel_manifest.json"
    if root_manifest.exists():
        os.replace(root_manifest, stage_root / "panel_manifest_NOT_FORMAL.json")


def failure_destination(final_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return final_root.with_name(f"{final_root.name}.FAILED-{timestamp}-{os.getpid()}")


def execute_build(
    *,
    args: argparse.Namespace,
    seal: ResearchSeal,
    initial_logger: logging.Logger,
    builder_sha256: str,
    version_audit: Mapping[str, Any],
    runtime: Runtime,
) -> int:
    final_root = Path(os.path.realpath(os.fspath(args.output_dir)))
    seal.guard_path(args.output_dir, operation="output_dir:raw")
    seal.guard_path(final_root, operation="output_dir:resolved")
    if final_root.exists():
        raise FileExistsError(
            f"formal panel output must be a fresh absent path; refusing reuse: {final_root}"
        )
    final_root.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_root = final_root.with_name(
        f".{final_root.name}.staging-{timestamp}-{os.getpid()}"
    )
    if stage_root.exists():
        raise FileExistsError(f"fresh staging path collision: {stage_root}")
    stage_root.mkdir(parents=False, exist_ok=False)
    logger = configure_logger(stage_root / "runtime.log")
    del initial_logger

    try:
        logger.info(
            "stage=startup experiment=prospective_geometry_matched_panels execute=true "
            "device=%s dtype=float32 cache_mode=fresh_absent_no_resume output=%s staging=%s",
            args.device,
            final_root,
            stage_root,
        )
        logger.info(
            "stage=startup seeds beans_selection=%d sroie_partition=%d token_rank=canonical_sha256 "
            "design_sha256=%s builder_sha256=%s",
            BEANS_SELECTION_SEED,
            SROIE_PARTITION_SEED,
            DESIGN_SHA256,
            builder_sha256,
        )
        resource_results = verify_resources(seal, logger)
        write_json(stage_root / "resource_hashes.json", resource_results)

        logger.info("stage=runtime_bind stack=%s", version_audit["versions"])
        torch = runtime.torch
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"formal builder requires CUDA/H100-class execution, got {device}")
        torch.cuda.set_device(device)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        resolved_config = {
            "schema_version": "prospective_panel_builder_config_v1",
            "execute": True,
            "output_dir": str(final_root),
            "atomic_build_mode": "fresh_sibling_staging_then_single_rename",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "dtype": "float32",
            "seed": 0,
            "selection_seeds": {
                "beans": BEANS_SELECTION_SEED,
                "sroie": SROIE_PARTITION_SEED,
            },
            "cache_mode": "fresh_absent_no_resume",
            "batch_sizes": {
                "beans_quality": args.beans_quality_batch_size,
                "beans_activation": args.beans_activation_batch_size,
                "trocr_quality": args.trocr_quality_batch_size,
                "trocr_activation": args.trocr_activation_batch_size,
                "proximity_ae_chunk": args.proximity_ae_chunk_size,
            },
            "models": {
                "beans": {"id": BEANS_MODEL_ID, "revision": BEANS_MODEL_REVISION},
                "trocr_sroie": {"id": TROCR_MODEL_ID, "revision": TROCR_MODEL_REVISION},
            },
            "datasets": {
                "beans_revision": BEANS_DATA_REVISION,
                "sroie_revision": SROIE_DATA_REVISION,
            },
            "runtime": version_audit,
            "design_sha256": DESIGN_SHA256,
            "builder_sha256": builder_sha256,
            "seal_at_start": seal.snapshot(),
        }
        write_json(stage_root / "resolved_config.json", resolved_config)
        logger.info(
            "stage=runtime_ready device=%s device_name=%s dtype=float32 tf32=false seed=0",
            device,
            resolved_config["device_name"],
        )

        # Frozen order: matrix geometry and unseen-weight audit precede image decoding.
        candidates = extract_candidate_matrices(runtime, logger)
        unseen_rows, unseen_summary, training_bank_hash_rows = audit_unseen_weights(
            runtime=runtime,
            seal=seal,
            candidates=candidates,
            device=device,
            ae_chunk_size=args.proximity_ae_chunk_size,
            logger=logger,
        )
        write_csv(stage_root / "unseen_weight_audit.csv", unseen_rows)
        write_csv(
            stage_root / "training_bank_compatible_weight_hashes.csv",
            training_bank_hash_rows,
        )
        write_json(stage_root / "unseen_weight_summary.json", unseen_summary)

        beans_records, class_names = prepare_beans_samples(runtime, seal, logger)
        sroie_records = prepare_sroie_samples(runtime, seal, logger)
        _, beans_tokens = token_manifest_rows(
            beans_records, panel_id="beans", patch_token_max=196
        )
        _, sroie_tokens = token_manifest_rows(
            sroie_records, panel_id="trocr_sroie", patch_token_max=576
        )

        results: dict[str, dict[str, Any]] = {}
        results["beans"] = build_one_panel(
            panel_id="beans",
            records=beans_records,
            class_names=class_names,
            selected_tokens=beans_tokens,
            matrices=candidates["beans"],
            resource_results=resource_results,
            unseen_summary=unseen_summary,
            stage_root=stage_root,
            final_root=final_root,
            builder_sha256=builder_sha256,
            args=args,
            device=device,
            runtime=runtime,
            seal=seal,
            logger=logger,
        )
        results["trocr_sroie"] = build_one_panel(
            panel_id="trocr_sroie",
            records=sroie_records,
            class_names=None,
            selected_tokens=sroie_tokens,
            matrices=candidates["trocr_sroie"],
            resource_results=resource_results,
            unseen_summary=unseen_summary,
            stage_root=stage_root,
            final_root=final_root,
            builder_sha256=builder_sha256,
            args=args,
            device=device,
            runtime=runtime,
            seal=seal,
            logger=logger,
        )

        if not all(result["panel_valid"] for result in results.values()):
            failure = {
                "schema_version": "prospective_panel_build_failure_v1",
                "status": "INVALID_PANEL",
                "reason": "one_or_more_quality_or_panel_gates_failed",
                "design_sha256": DESIGN_SHA256,
                "builder_sha256": builder_sha256,
                "panels": _jsonable(results),
                "seal": seal.snapshot(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(stage_root / "failure_manifest.json", failure)
            quarantine_panel_files(stage_root)
            failed_root = failure_destination(final_root)
            logger.error(
                "stage=end_summary status=INVALID_PANEL output_committed=false "
                "combined_manifest_emitted=false panel_pt_emitted=false "
                "failure_artifacts_destination=%s panels=%s",
                failed_root,
                _jsonable(results),
            )
            close_file_log(logger)
            build_artifact_manifest(stage_root, builder_sha256=builder_sha256, seal=seal)
            os.replace(stage_root, failed_root)
            logger.error(
                "stage=failure_artifacts_preserved path=%s combined_manifest_emitted=false panel_pt_emitted=false",
                failed_root,
            )
            return 2

        panel_entries: dict[str, dict[str, Any]] = {}
        for panel_id in PANEL_IDS:
            candidate_path = Path(results[panel_id]["candidate_path"])
            panel_path = candidate_path.with_name("panel.pt")
            os.replace(candidate_path, panel_path)
            panel_sha = sha256_file(panel_path, seal)
            panel_entries[panel_id] = {
                "panel_id": panel_id,
                "path": str(results[panel_id]["final_panel_path"]),
                "sha256": panel_sha,
                "quality_status": "PASS",
                "checkpoint_sha256": checkpoint_sha256(panel_id),
            }
        root_manifest = {
            "schema_version": "prospective_panel_manifest_v1",
            "design_sha256": DESIGN_SHA256,
            "builder_sha256": builder_sha256,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "panels": panel_entries,
        }
        write_json(stage_root / "panel_manifest.json", root_manifest)
        logger.info(
            "stage=combined_manifest_ready panels=beans,trocr_sroie quality=PASS exact_overlap=0"
        )
        if not seal.snapshot()["pass"]:
            raise RuntimeError(f"research seal failed before commit: {seal.snapshot()}")

        # Freeze runtime.log before hashing every output file.
        logger.info(
            "stage=end_summary status=PASS output_destination=%s combined_manifest=%s "
            "beans_panel_sha256=%s beans_top1=%.6f trocr_panel_sha256=%s "
            "trocr_exact=%.6f trocr_corpus_cer=%.6f matrices=144 exact_overlap=0",
            final_root,
            final_root / "panel_manifest.json",
            panel_entries["beans"]["sha256"],
            results["beans"]["quality_metrics"]["top1_accuracy"],
            panel_entries["trocr_sroie"]["sha256"],
            results["trocr_sroie"]["quality_metrics"]["exact_match_fraction"],
            results["trocr_sroie"]["quality_metrics"]["corpus_cer"],
        )
        close_file_log(logger)
        artifact_manifest = build_artifact_manifest(
            stage_root, builder_sha256=builder_sha256, seal=seal
        )
        os.replace(stage_root, final_root)
        verification = verify_artifact_manifest(final_root, seal)
        committed_root = json.loads((final_root / "panel_manifest.json").read_text(encoding="utf-8"))
        for panel_id, entry in committed_root["panels"].items():
            path = Path(entry["path"])
            if Path(os.path.realpath(os.fspath(path))) != path or not path.is_file():
                raise RuntimeError(f"committed canonical panel path failure: {panel_id}/{path}")
            if sha256_file(path, seal) != entry["sha256"]:
                raise RuntimeError(f"committed panel SHA failure: {panel_id}/{path}")
        logger.info(
            "stage=commit_complete output=%s panel_manifest=%s artifact_manifest_sha256=%s "
            "files=%d matrices=144 quality=PASS",
            final_root,
            final_root / "panel_manifest.json",
            verification["sha256"],
            artifact_manifest["file_count"],
        )
        return 0
    except Exception as error:
        logger.exception("stage=build_exception error=%s", error)
        try:
            if stage_root.exists():
                quarantine_panel_files(stage_root)
                write_json(
                    stage_root / "failure_manifest.json",
                    {
                        "schema_version": "prospective_panel_build_failure_v1",
                        "status": "BUILD_EXCEPTION",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "design_sha256": DESIGN_SHA256,
                        "builder_sha256": builder_sha256,
                        "seal": seal.snapshot(),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                failed_root = failure_destination(final_root)
                logger.error(
                    "stage=end_summary status=BUILD_EXCEPTION output_committed=false "
                    "combined_manifest_emitted=false panel_pt_emitted=false "
                    "failure_artifacts_destination=%s error_type=%s error=%s",
                    failed_root,
                    type(error).__name__,
                    error,
                )
                close_file_log(logger)
                build_artifact_manifest(stage_root, builder_sha256=builder_sha256, seal=seal)
                os.replace(stage_root, failed_root)
                logger.error(
                    "stage=failure_artifacts_preserved path=%s combined_manifest_emitted=false "
                    "panel_pt_emitted=false",
                    failed_root,
                )
            elif final_root.exists():
                # A post-rename verification failure must not leave an apparently
                # valid cache at the configured formal path.
                quarantine_panel_files(final_root)
                write_json(
                    final_root / "failure_manifest.json",
                    {
                        "schema_version": "prospective_panel_build_failure_v1",
                        "status": "POST_COMMIT_VERIFICATION_EXCEPTION",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "design_sha256": DESIGN_SHA256,
                        "builder_sha256": builder_sha256,
                        "seal": seal.snapshot(),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                build_artifact_manifest(final_root, builder_sha256=builder_sha256, seal=seal)
                failed_root = failure_destination(final_root)
                os.replace(final_root, failed_root)
                logger.error(
                    "stage=post_commit_quarantine path=%s formal_path_removed=true",
                    failed_root,
                )
        except Exception as preserve_error:
            logger.error(
                "stage=failure_preservation_error staging=%s error=%s",
                stage_root,
                preserve_error,
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    version_audit = verify_runtime_versions()
    runtime: Runtime | None = None
    if args.execute:
        # Mirror the formal Weight-AE runner's bootstrap: import audited runtime
        # dependencies before installing the path audit hook. Transformers 5.1
        # enumerates every installed model package (including its own data2vec
        # namespace) at import time. No research/model/data/output input is read
        # here; the irreversible seal is installed before all such reads below.
        runtime = load_runtime()
        imported_runtime = {
            "torch.__version__": str(runtime.torch.__version__),
            "torch.version.cuda": str(runtime.torch.version.cuda),
        }
        expected_imported_runtime = version_audit["execute_runtime_checks"]
        if imported_runtime != expected_imported_runtime:
            raise RuntimeError(
                f"imported PyTorch runtime mismatch: expected={expected_imported_runtime} "
                f"found={imported_runtime}"
            )
        version_audit = {
            **version_audit,
            "execute_runtime_checks_deferred": False,
            "execute_runtime_found": imported_runtime,
        }
    seal = ResearchSeal()
    seal.install()
    logger = configure_logger()
    logger.info(
        "stage=seal_install installed=true offline=true network_blocked=true subprocess_blocked=true "
        "target_path_blocked=true forbidden_module=data2vec"
    )
    builder_path = Path(os.path.realpath(__file__))
    seal.guard_path(builder_path, operation="builder_source")
    builder_sha256 = sha256_file(builder_path, seal)

    if not args.execute:
        logger.info(
            "stage=audit_start mode=read_only_no_model_run design_sha256=%s builder_sha256=%s",
            DESIGN_SHA256,
            builder_sha256,
        )
        resources = verify_resources(seal, logger)
        seal_snapshot = seal.snapshot()
        if not seal_snapshot["pass"]:
            raise RuntimeError(f"research seal audit failed: {seal_snapshot}")
        seal.guard_path(args.output_dir, operation="audit_output_dir:raw")
        requested_output_realpath = Path(os.path.realpath(os.fspath(args.output_dir)))
        seal.guard_path(requested_output_realpath, operation="audit_output_dir:resolved")
        result = {
            "schema_version": "prospective_panel_builder_audit_v1",
            "mode": "audit_only",
            "model_run": False,
            "output_written": False,
            "execute_required_for_build": True,
            "design_path": str(DESIGN_PATH),
            "design_sha256": DESIGN_SHA256,
            "builder_path": str(builder_path),
            "builder_sha256": builder_sha256,
            "runtime": version_audit,
            "resource_count": len(resources),
            "resources": resources,
            "seal": seal_snapshot,
            "requested_output": str(args.output_dir),
            "requested_output_exists": requested_output_realpath.exists(),
        }
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        logger.info(
            "stage=audit_complete pass=true resources=%d model_run=false output_written=false",
            len(resources),
        )
        return 0

    assert runtime is not None
    return execute_build(
        args=args,
        seal=seal,
        initial_logger=logger,
        builder_sha256=builder_sha256,
        version_audit=version_audit,
        runtime=runtime,
    )


if __name__ == "__main__":
    raise SystemExit(main())
