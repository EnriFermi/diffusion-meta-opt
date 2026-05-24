from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from dataset.shared.types import SharedSample

from dataset.big_vae_offline_parts.metadata import *

def _weight_path(weights_dir: Path, source_key: str) -> Path:
    return weights_dir / source_key[:2] / f"{source_key}.pt"


def _chunk_index_path(chunk_index_dir: Path, chunk_id: str) -> Path:
    return chunk_index_dir / f"{str(chunk_id).strip()}.json"


def _normalize_sampling_group_keys(value: Any) -> tuple[str, ...]:
    if value is None:
        raw_items: list[str] = []
    elif isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        raw_items = [str(item).strip() for item in value]
    else:
        raise TypeError("offline_dataset.sampling.group_keys must be a string or sequence of strings")

    normalized: list[str] = []
    for item in raw_items:
        if not item:
            continue
        key = _BALANCED_SAMPLING_GROUP_KEY_ALIASES.get(item.strip().lower())
        if key is None:
            valid = ", ".join(sorted(_BALANCED_SAMPLING_GROUP_KEY_ALIASES))
            raise ValueError(f"Unsupported offline_dataset.sampling.group_keys item {item!r}; valid keys: {valid}")
        if key not in normalized:
            normalized.append(key)

    if not normalized:
        return ("dataset", "model", "layer_type", "depth")
    return tuple(normalized)


def _normalize_offline_sampling_mode(value: Any) -> str:
    normalized = str(value or "random").strip().lower()
    aliases = {
        "random": "random",
        "shuffle": "random",
        "shuffled": "random",
        "balanced": "balanced",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {"random", "balanced"}:
        raise ValueError(f"Unsupported offline_dataset.sampling.mode: {value!r}")
    return resolved


def _chunk_record_index_entry(
    record: dict[str, Any],
    *,
    record_idx: int,
    source_weight_shape: Sequence[int] | None,
) -> dict[str, Any]:
    source_key = str(record.get("source_key", "") or "")
    model_name = str(record.get("model_name", "") or "").strip() or "<unknown_model>"
    layer_name = str(record.get("layer_name", "") or "").strip() or "<unknown_layer>"
    meta = dict(record.get("meta", {}) or {})
    dataset_names = _sample_dataset_names(meta)
    primary_dataset = dataset_names[0] if dataset_names else "<unknown_dataset>"

    d_in = int(source_weight_shape[0]) if source_weight_shape is not None and len(source_weight_shape) >= 1 else 0
    d_out = int(source_weight_shape[1]) if source_weight_shape is not None and len(source_weight_shape) >= 2 else 0
    depth = infer_layer_depth(layer_name)
    return {
        "record_idx": int(record_idx),
        "source_key": source_key,
        "model_name": model_name,
        "layer_name": layer_name,
        "primary_dataset": primary_dataset,
        "dataset_names": dataset_names,
        "layer_type": infer_layer_type(layer_name),
        "layer_depth": int(depth) if depth is not None else None,
        "depth_label": str(depth) if depth is not None else "unknown",
        "weight_shape": [d_in, d_out],
        "shape": _shape_key(d_in=d_in, d_out=d_out),
        "d_in_bucket": _pow2_bucket(d_in),
        "d_out_bucket": _pow2_bucket(d_out),
        "num_params_bucket": _pow2_bucket(d_in * d_out),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _load_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def _directory_size_bytes(root_dir: Path) -> int:
    total = 0
    for path in root_dir.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return int(total)


class BigVAEOfflineDatasetWriter:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        target_size_bytes: int,
        patch_size: int,
        max_T_patches: int,
        max_d_out: int,
        max_x_rows: int,
        x_chunk_size_records: int,
        max_samples_per_source: int,
        enforce_stage_compatibility: bool,
        overwrite_existing: bool,
        seed: int,
        logger: logging.Logger | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        root_dir_str = str(root_dir).strip()
        if not root_dir_str:
            raise ValueError("Offline BigVAE dataset root_dir must be non-empty")
        self.root_dir = Path(root_dir_str)
        self.target_size_bytes = max(1, int(target_size_bytes))
        self.patch_size = max(1, int(patch_size))
        self.max_T_patches = max(1, int(max_T_patches))
        self.max_d_out = max(1, int(max_d_out))
        self.max_x_rows = max(0, int(max_x_rows))
        self.enforce_stage_compatibility = bool(enforce_stage_compatibility)
        self.min_d_in = self.patch_size * self.max_T_patches if self.enforce_stage_compatibility else 0
        self.min_d_out = self.max_d_out if self.enforce_stage_compatibility else 0
        self.x_chunk_size_records = max(1, int(x_chunk_size_records))
        self.max_samples_per_source = max(0, int(max_samples_per_source))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config_snapshot = dict(config_snapshot or {})

        self.weights_dir = self.root_dir / "weights"
        self.x_chunks_dir = self.root_dir / "x_chunks"
        self.chunk_index_dir = self.root_dir / "chunk_index"
        self.manifest_path = self.root_dir / "manifest.json"
        self.sources_index_path = self.root_dir / "sources.json"
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

        self._seen_samples = 0
        self._accepted_records = 0
        self._skipped_invalid = 0
        self._skipped_incompatible = 0
        self._skipped_source_cap = 0
        self._total_file_bytes = 0
        self._pending_x_bytes_estimate = 0
        self._chunk_index = 0
        self._closed = False

        self._model_counts: Counter[str] = Counter()
        self._layer_counts: Counter[str] = Counter()
        self._dataset_counts: Counter[str] = Counter()
        self._source_counts: Counter[str] = Counter()
        self._chunk_buffer: list[dict[str, Any]] = []
        self._source_index: dict[str, dict[str, Any]] = {}

        self._prepare_root(overwrite_existing=bool(overwrite_existing))

    @property
    def reached_target_size(self) -> bool:
        return int(self._total_file_bytes) >= int(self.target_size_bytes)

    def ingest(self, sample: Any) -> str:
        if self._closed:
            raise RuntimeError("Writer is already closed")

        self._seen_samples += 1
        x = getattr(sample, "x", None)
        W = getattr(sample, "weight", None)
        if not (
            torch.is_tensor(x)
            and torch.is_tensor(W)
            and x.ndim == 2
            and W.ndim == 2
            and int(x.shape[1]) == int(W.shape[0])
            and int(x.shape[0]) > 0
            and int(W.shape[0]) > 0
            and int(W.shape[1]) > 0
        ):
            self._skipped_invalid += 1
            return "invalid"

        x_cpu = _prepare_cpu_sample_tensor(x)
        W_cpu = _prepare_cpu_sample_tensor(W)
        meta = dict(getattr(sample, "meta", {}) or {})
        model_name = str(getattr(sample, "model_name", "") or "").strip()
        layer_name = str(getattr(sample, "layer_name", "") or "").strip()

        if self.max_x_rows > 0 and int(x_cpu.shape[0]) > int(self.max_x_rows):
            keep = torch.randperm(int(x_cpu.shape[0]), generator=self._rng)[: int(self.max_x_rows)]
            keep = keep.sort().values
            x_cpu = x_cpu[keep]
            meta["offline_builder_max_x_rows_applied"] = True
            meta["offline_builder_selected_row_count"] = int(keep.numel())
            meta["offline_builder_selected_row_indices_preview"] = [int(idx) for idx in keep[:32].tolist()]

        if self.enforce_stage_compatibility and (
            int(W_cpu.shape[0]) < int(self.min_d_in) or int(W_cpu.shape[1]) < int(self.min_d_out)
        ):
            self._skipped_incompatible += 1
            return "incompatible"

        source_key = _source_key(model_name=model_name, layer_name=layer_name)
        if self.max_samples_per_source > 0 and int(self._source_counts[source_key]) >= int(self.max_samples_per_source):
            self._skipped_source_cap += 1
            return "source_cap"

        self._write_weight_if_needed(
            source_key=source_key,
            model_name=model_name,
            layer_name=layer_name,
            weight=W_cpu,
            meta=meta,
        )

        record = {
            "source_key": source_key,
            "model_name": model_name,
            "layer_name": layer_name,
            "x": x_cpu,
            "meta": meta,
        }
        self._chunk_buffer.append(record)
        self._pending_x_bytes_estimate += int(x_cpu.numel()) * int(x_cpu.element_size())

        self._accepted_records += 1
        self._source_counts[source_key] += 1
        self._model_counts[model_name or "<unknown_model>"] += 1
        self._layer_counts[layer_name or "<unknown_layer>"] += 1

        source_entry = self._source_index[source_key]
        source_entry["num_records"] = int(source_entry.get("num_records", 0)) + 1
        for dataset_name in _sample_dataset_names(meta):
            self._dataset_counts[dataset_name] += 1
            source_entry.setdefault("dataset_names", set()).add(dataset_name)

        if len(self._chunk_buffer) >= self.x_chunk_size_records:
            self._flush_x_chunk(force_partial=False)
        elif int(self._total_file_bytes + self._pending_x_bytes_estimate) >= int(self.target_size_bytes):
            self._flush_x_chunk(force_partial=True)

        return "accepted"

    def stats(self) -> dict[str, Any]:
        top_source_key = None
        top_source_records = 0
        if self._source_counts:
            top_source_key, top_source_records = self._source_counts.most_common(1)[0]
        return {
            "seen_samples": int(self._seen_samples),
            "accepted_records": int(self._accepted_records),
            "skipped_invalid": int(self._skipped_invalid),
            "skipped_incompatible": int(self._skipped_incompatible),
            "skipped_source_cap": int(self._skipped_source_cap),
            "unique_sources": int(len(self._source_index)),
            "unique_models": int(sum(1 for key in self._model_counts if key != "<unknown_model>")),
            "unique_layers": int(sum(1 for key in self._layer_counts if key != "<unknown_layer>")),
            "target_size_bytes": int(self.target_size_bytes),
            "written_size_bytes": int(self._total_file_bytes),
            "written_size_gb": float(self._total_file_bytes) / (1024.0 ** 3),
            "enforce_stage_compatibility": bool(self.enforce_stage_compatibility),
            "pending_records": int(len(self._chunk_buffer)),
            "pending_x_bytes_estimate": int(self._pending_x_bytes_estimate),
            "top_source_key": top_source_key,
            "top_source_records": int(top_source_records),
        }

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._build_manifest()
        self._flush_x_chunk(force_partial=True)
        self._write_sidecars()
        self._closed = True
        return self._build_manifest()

    def _prepare_root(self, *, overwrite_existing: bool) -> None:
        if self.root_dir.exists():
            has_payload = any(self.root_dir.iterdir())
            if has_payload and not overwrite_existing:
                raise FileExistsError(
                    f"Offline BigVAE dataset root already exists and is not empty: {self.root_dir}. "
                    "Set train.offline_dataset.builder.overwrite_existing=true to replace it."
                )
            if has_payload:
                shutil.rmtree(self.root_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.x_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_index_dir.mkdir(parents=True, exist_ok=True)

    def _write_weight_if_needed(
        self,
        *,
        source_key: str,
        model_name: str,
        layer_name: str,
        weight: torch.Tensor,
        meta: dict[str, Any],
    ) -> None:
        if source_key in self._source_index:
            return

        file_path = _weight_path(self.weights_dir, source_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": OFFLINE_BIG_VAE_FORMAT_VERSION,
            "source_key": source_key,
            "model_name": model_name,
            "layer_name": layer_name,
            "weight": weight,
            "weight_shape": [int(dim) for dim in weight.shape],
            "created_at": float(time.time()),
        }
        torch.save(payload, file_path)
        file_size = int(file_path.stat().st_size)
        self._total_file_bytes += file_size

        self._source_index[source_key] = {
            "source_key": source_key,
            "model_name": model_name,
            "layer_name": layer_name,
            "weight_path": str(file_path.relative_to(self.root_dir)),
            "weight_shape": [int(dim) for dim in weight.shape],
            "weight_size_bytes": int(file_size),
            "num_records": 0,
            "dataset_names": set(_sample_dataset_names(meta)),
            "estimated_stage_slices_per_record": int(
                estimate_stage_slice_capacity(
                    d_in=int(weight.shape[0]),
                    d_out=int(weight.shape[1]),
                    patch_size=self.patch_size,
                    max_T_patches=self.max_T_patches,
                    max_d_out=self.max_d_out,
                )
            ),
        }

    def _flush_x_chunk(self, *, force_partial: bool) -> None:
        if not self._chunk_buffer:
            return
        if not force_partial and len(self._chunk_buffer) < self.x_chunk_size_records:
            return

        records = self._chunk_buffer
        self._chunk_buffer = []
        self._pending_x_bytes_estimate = 0

        chunk_id = f"x_chunk_{self._chunk_index:08d}"
        self._chunk_index += 1
        chunk_path = self.x_chunks_dir / f"{chunk_id}.pt"
        chunk_index_path = _chunk_index_path(self.chunk_index_dir, chunk_id)
        chunk_meta = {
            "chunk_id": chunk_id,
            "created_at": float(time.time()),
            "num_records": int(len(records)),
            "unique_sources": int(len({str(item['source_key']) for item in records})),
            "top_models": [[name, int(count)] for name, count in Counter(str(item["model_name"]) for item in records).most_common(5)],
        }
        torch.save(
            {
                "format_version": OFFLINE_BIG_VAE_FORMAT_VERSION,
                "chunk_id": chunk_id,
                "records": records,
                "meta": chunk_meta,
            },
            chunk_path,
        )
        self._total_file_bytes += int(chunk_path.stat().st_size)
        _atomic_write_json(
            chunk_index_path,
            self._build_chunk_index_payload(
                chunk_id=chunk_id,
                chunk_path=chunk_path,
                records=records,
            ),
        )
        self._write_sidecars()

    def _build_chunk_index_payload(
        self,
        *,
        chunk_id: str,
        chunk_path: Path,
        records: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        index_records = []
        for record_idx, record in enumerate(records):
            source_key = str(record.get("source_key", "") or "")
            source_meta = self._source_index.get(source_key, {})
            weight_shape = source_meta.get("weight_shape", [0, 0]) if isinstance(source_meta, dict) else [0, 0]
            index_records.append(
                _chunk_record_index_entry(
                    record,
                    record_idx=record_idx,
                    source_weight_shape=weight_shape,
                )
            )
        return {
            "format_version": OFFLINE_BIG_VAE_FORMAT_VERSION,
            "chunk_id": chunk_id,
            "chunk_path": str(chunk_path.relative_to(self.root_dir)),
            "num_records": int(len(index_records)),
            "records": index_records,
        }

    def _build_sources_payload(self) -> dict[str, Any]:
        sources = []
        for source_key, payload in sorted(self._source_index.items(), key=lambda item: item[0]):
            source_payload = dict(payload)
            dataset_names = source_payload.get("dataset_names", set())
            if isinstance(dataset_names, set):
                source_payload["dataset_names"] = sorted(str(item) for item in dataset_names)
            sources.append(source_payload)
        return {
            "format_version": OFFLINE_BIG_VAE_FORMAT_VERSION,
            "num_sources": int(len(sources)),
            "sources": sources,
        }

    def _build_manifest(self) -> dict[str, Any]:
        actual_size_bytes = _directory_size_bytes(self.root_dir)
        return {
            "format_version": OFFLINE_BIG_VAE_FORMAT_VERSION,
            "created_at": float(time.time()),
            "root_dir": str(self.root_dir),
            "target_size_bytes": int(self.target_size_bytes),
            "actual_size_bytes": int(actual_size_bytes),
            "actual_size_gb": float(actual_size_bytes) / (1024.0 ** 3),
            "accepted_records": int(self._accepted_records),
            "seen_samples": int(self._seen_samples),
            "skipped_invalid": int(self._skipped_invalid),
            "skipped_incompatible": int(self._skipped_incompatible),
            "skipped_source_cap": int(self._skipped_source_cap),
            "unique_sources": int(len(self._source_index)),
            "unique_models": int(sum(1 for key in self._model_counts if key != "<unknown_model>")),
            "unique_layers": int(sum(1 for key in self._layer_counts if key != "<unknown_layer>")),
            "top_models": [[name, int(count)] for name, count in self._model_counts.most_common(16)],
            "top_layers": [[name, int(count)] for name, count in self._layer_counts.most_common(16)],
            "top_datasets": [[name, int(count)] for name, count in self._dataset_counts.most_common(16)],
            "top_sources": [[name, int(count)] for name, count in self._source_counts.most_common(16)],
            "stage_targets": {
                "patch_size": int(self.patch_size),
                "max_T_patches": int(self.max_T_patches),
                "max_d_out": int(self.max_d_out),
                "max_x_rows": int(self.max_x_rows),
                "enforce_stage_compatibility": bool(self.enforce_stage_compatibility),
                "min_d_in": int(self.min_d_in),
                "min_d_out": int(self.min_d_out),
            },
            "layout": {
                "weights_dir": str(self.weights_dir.relative_to(self.root_dir)),
                "x_chunks_dir": str(self.x_chunks_dir.relative_to(self.root_dir)),
                "chunk_index_dir": str(self.chunk_index_dir.relative_to(self.root_dir)),
                "sources_index": str(self.sources_index_path.relative_to(self.root_dir)),
            },
            "config_snapshot": self.config_snapshot,
        }

    def _write_sidecars(self) -> None:
        _atomic_write_json(self.sources_index_path, self._build_sources_payload())
        _atomic_write_json(self.manifest_path, self._build_manifest())

__all__ = [
    '_weight_path',
    '_chunk_index_path',
    '_normalize_sampling_group_keys',
    '_normalize_offline_sampling_mode',
    '_chunk_record_index_entry',
    '_atomic_write_json',
    '_load_json_payload',
    '_directory_size_bytes',
    'BigVAEOfflineDatasetWriter',
]
