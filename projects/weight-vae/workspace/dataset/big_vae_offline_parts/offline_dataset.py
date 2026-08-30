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
from dataset.big_vae_offline_parts.chunk_io import *

class OfflineBigVAEDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        root_dir: str | Path,
        shuffle_chunks: bool = True,
        shuffle_records_within_chunk: bool = True,
        repeat: bool = True,
        seed: int = 42,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        shard_by_chunk: bool = True,
        weight_cache_size: int = 64,
        sampling_mode: str = "random",
        sampling_group_keys: Sequence[str] | str | None = None,
        sampling_window_size: int = 2048,
        sampling_max_records_per_chunk_round: int = 8,
        x_chunk_cache_size: int = 4,
        runtime_enforce_stage_compatibility: bool = False,
        runtime_min_d_in: int = 0,
        runtime_min_d_out: int = 0,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.sources_index_path = self.root_dir / "sources.json"
        self.weights_dir = self.root_dir / "weights"
        self.x_chunks_dir = self.root_dir / "x_chunks"
        self.chunk_index_dir = self.root_dir / "chunk_index"
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_records_within_chunk = bool(shuffle_records_within_chunk)
        self.repeat = bool(repeat)
        self.seed = int(seed)
        self.shard_rank = max(0, int(shard_rank))
        self.shard_world_size = max(1, int(shard_world_size))
        self.shard_by_chunk = bool(shard_by_chunk)
        self.weight_cache_size = max(1, int(weight_cache_size))
        self.sampling_mode = _normalize_offline_sampling_mode(sampling_mode)
        self.sampling_group_keys = _normalize_sampling_group_keys(sampling_group_keys)
        self.sampling_window_size = max(1, int(sampling_window_size))
        self.sampling_max_records_per_chunk_round = max(1, int(sampling_max_records_per_chunk_round))
        self.x_chunk_cache_size = max(1, int(x_chunk_cache_size))
        self.runtime_enforce_stage_compatibility = bool(runtime_enforce_stage_compatibility)
        self.runtime_min_d_in = max(0, int(runtime_min_d_in))
        self.runtime_min_d_out = max(0, int(runtime_min_d_out))
        self.logger = logging.getLogger(self.__class__.__name__)

        self._weight_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._x_chunk_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._active_iter: Iterator[SharedSample] | None = None
        self._manifest = self._load_manifest()
        self._sources_index = self._load_sources_index()
        self._chunk_paths = sorted(self.x_chunks_dir.glob("*.pt"))
        if not self._chunk_paths:
            raise FileNotFoundError(f"No offline BigVAE x chunks found under {self.x_chunks_dir}")
        self._effective_chunk_indices = self._resolve_effective_chunk_indices()
        self.logger.info(
            "Opening Offline BigVAE dataset: root=%s chunks=%s effective_chunks=%s shard_rank=%s "
            "shard_world_size=%s sampling_mode=%s runtime_stage_filter=%s",
            self.root_dir,
            len(self._chunk_paths),
            len(self._effective_chunk_indices),
            self.shard_rank,
            self.shard_world_size,
            self.sampling_mode,
            self.runtime_enforce_stage_compatibility,
        )
        self._chunk_index_payloads: list[dict[str, Any]] | None = None
        self._balanced_group_to_refs: dict[tuple[str, ...], list[tuple[int, int]]] | None = None
        self._runtime_filtered_record_count = 0
        self._runtime_compatible_record_count = 0
        self._runtime_compatible_record_refs_by_chunk: dict[int, tuple[int, ...]] | None = None
        if self.runtime_enforce_stage_compatibility:
            self.logger.info(
                "Loading Offline BigVAE chunk indexes for runtime stage filter: chunks=%s min_d_in=%s min_d_out=%s",
                len(self._effective_chunk_indices),
                self.runtime_min_d_in,
                self.runtime_min_d_out,
            )
            self._chunk_index_payloads = self._load_chunk_index_payloads()
            (
                self._runtime_compatible_record_refs_by_chunk,
                self._runtime_compatible_record_count,
                self._runtime_filtered_record_count,
            ) = self._build_runtime_compatible_record_refs_by_chunk(self._chunk_index_payloads)
            self.logger.info(
                "Offline BigVAE runtime stage filter ready: compatible_records=%s filtered_records=%s",
                self._runtime_compatible_record_count,
                self._runtime_filtered_record_count,
            )
        if self.sampling_mode == "balanced":
            if self._chunk_index_payloads is None:
                self.logger.info(
                    "Loading Offline BigVAE chunk indexes for balanced sampling: chunks=%s group_keys=%s",
                    len(self._effective_chunk_indices),
                    list(self.sampling_group_keys),
                )
                self._chunk_index_payloads = self._load_chunk_index_payloads()
            self._balanced_group_to_refs = self._build_balanced_group_index(self._chunk_index_payloads)
            self.logger.info(
                "Offline BigVAE balanced sampling index ready: groups=%s",
                len(self._balanced_group_to_refs),
            )

    def __iter__(self) -> Iterator[SharedSample]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = int(worker_info.id) if worker_info is not None else 0
        worker_count = int(worker_info.num_workers) if worker_info is not None else 1
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            if self.sampling_mode == "balanced":
                yield from self._iter_balanced_epoch(rng, worker_id=worker_id, worker_count=worker_count)
            else:
                yield from self._iter_random_epoch(rng, worker_id=worker_id, worker_count=worker_count)

            if not self.repeat:
                return
            epoch += 1

    def maybe_collect(self, step_idx: int) -> None:
        del step_idx

    def try_next_sample(self) -> SharedSample | None:
        if self._active_iter is None:
            self._active_iter = iter(self)
        try:
            return next(self._active_iter)
        except StopIteration:
            self._active_iter = None
            return None

    def cache_size(self) -> int:
        return int(self._manifest.get("accepted_records", len(self._chunk_paths)))

    def debug_snapshot(self, preview: int = 5) -> dict[str, Any]:
        preview_int = max(1, int(preview))
        return {
            "root_dir": str(self.root_dir),
            "accepted_records": int(self._manifest.get("accepted_records", 0)),
            "unique_sources": int(self._manifest.get("unique_sources", 0)),
            "num_chunks": int(len(self._chunk_paths)),
            "chunk_head": [path.name for path in self._chunk_paths[:preview_int]],
            "weight_cache_size": int(len(self._weight_cache)),
            "x_chunk_cache_size": int(len(self._x_chunk_cache)),
            "shard_rank": int(self.shard_rank),
            "shard_world_size": int(self.shard_world_size),
            "shard_by_chunk": bool(self.shard_by_chunk),
            "sampling_mode": self.sampling_mode,
            "sampling_group_keys": list(self.sampling_group_keys),
            "runtime_enforce_stage_compatibility": bool(self.runtime_enforce_stage_compatibility),
            "runtime_min_d_in": int(self.runtime_min_d_in),
            "runtime_min_d_out": int(self.runtime_min_d_out),
            "runtime_compatible_records": int(self._runtime_compatible_record_count),
            "runtime_filtered_records": int(self._runtime_filtered_record_count),
        }

    def summary(self) -> dict[str, Any]:
        payload = dict(self._manifest)
        payload["sampling_mode"] = self.sampling_mode
        payload["sampling_group_keys"] = list(self.sampling_group_keys)
        payload["runtime_enforce_stage_compatibility"] = bool(self.runtime_enforce_stage_compatibility)
        payload["runtime_min_d_in"] = int(self.runtime_min_d_in)
        payload["runtime_min_d_out"] = int(self.runtime_min_d_out)
        payload["runtime_compatible_records"] = int(self._runtime_compatible_record_count)
        payload["runtime_filtered_records"] = int(self._runtime_filtered_record_count)
        return payload

    def close(self) -> None:
        self._weight_cache.clear()
        self._x_chunk_cache.clear()
        self._active_iter = None

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Offline BigVAE manifest not found: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Offline BigVAE manifest must be a dict, got {type(payload)}")
        return payload

    def _load_sources_index(self) -> dict[str, dict[str, Any]]:
        if not self.sources_index_path.exists():
            return {}
        payload = json.loads(self.sources_index_path.read_text(encoding="utf-8"))
        sources = payload.get("sources", [])
        if not isinstance(sources, list):
            return {}
        result: dict[str, dict[str, Any]] = {}
        for item in sources:
            if not isinstance(item, dict):
                continue
            source_key = str(item.get("source_key", ""))
            if not source_key:
                continue
            result[source_key] = dict(item)
        return result

    def _resolve_effective_chunk_indices(self) -> list[int]:
        if self.shard_by_chunk and self.shard_world_size > 1:
            chunk_indices = list(range(len(self._chunk_paths)))[self.shard_rank :: self.shard_world_size]
            if not chunk_indices:
                raise RuntimeError(
                    "Offline BigVAE dataset shard is empty. "
                    f"rank={self.shard_rank} world_size={self.shard_world_size} chunk_count={len(self._chunk_paths)}"
                )
            return chunk_indices
        return list(range(len(self._chunk_paths)))

    def _iter_random_epoch(
        self,
        rng: random.Random,
        *,
        worker_id: int = 0,
        worker_count: int = 1,
    ) -> Iterator[SharedSample]:
        if self._runtime_compatible_record_refs_by_chunk is not None:
            chunk_indices = [
                int(chunk_idx)
                for chunk_idx in self._effective_chunk_indices
                if self._runtime_compatible_record_refs_by_chunk.get(int(chunk_idx))
            ]
        else:
            chunk_indices = list(self._effective_chunk_indices)
        if self.shuffle_chunks and len(chunk_indices) > 1:
            rng.shuffle(chunk_indices)
        chunk_indices = self._worker_shard_chunk_indices(chunk_indices, worker_id=worker_id, worker_count=worker_count)

        for chunk_idx in chunk_indices:
            if self._runtime_compatible_record_refs_by_chunk is not None:
                order = list(self._runtime_compatible_record_refs_by_chunk.get(int(chunk_idx), ()))
            else:
                payload = self._load_x_chunk(chunk_idx)
                records = payload.get("records", [])
                if not isinstance(records, list):
                    raise TypeError(f"Offline BigVAE chunk {self._chunk_paths[chunk_idx]} has invalid records payload")
                order = list(range(len(records)))
            if self.shuffle_records_within_chunk and len(order) > 1:
                rng.shuffle(order)
            for record_idx in order:
                yield self._shared_sample_from_record_ref(chunk_idx=chunk_idx, record_idx=record_idx)

    def _iter_balanced_epoch(
        self,
        rng: random.Random,
        *,
        worker_id: int = 0,
        worker_count: int = 1,
    ) -> Iterator[SharedSample]:
        if self._balanced_group_to_refs is None:
            raise RuntimeError("balanced sampling requested but group index is not initialized")
        worker_chunk_set = set(
            self._worker_shard_chunk_indices(
                list(self._effective_chunk_indices),
                worker_id=worker_id,
                worker_count=worker_count,
            )
        )
        for chunk_idx, record_idx in self._build_balanced_epoch_record_refs(rng, worker_chunk_set=worker_chunk_set):
            yield self._shared_sample_from_record_ref(chunk_idx=chunk_idx, record_idx=record_idx)

    def _build_balanced_epoch_record_refs(
        self,
        rng: random.Random,
        *,
        worker_chunk_set: set[int] | None = None,
    ) -> Iterator[tuple[int, int]]:
        if self._balanced_group_to_refs is None:
            return

        group_to_pending: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        for group_key, refs in self._balanced_group_to_refs.items():
            if worker_chunk_set is None:
                pending = list(refs)
            else:
                pending = [ref for ref in refs if int(ref[0]) in worker_chunk_set]
            rng.shuffle(pending)
            if pending:
                group_to_pending[group_key] = pending

        active_groups = list(group_to_pending.keys())
        rng.shuffle(active_groups)
        cursor = 0
        while active_groups:
            active_group_count = len(active_groups)
            window: list[tuple[int, int]] = []
            while active_groups and len(window) < self.sampling_window_size:
                if cursor >= len(active_groups):
                    cursor = 0
                    if len(active_groups) > 1:
                        rng.shuffle(active_groups)
                group_key = active_groups[cursor]
                pending = group_to_pending[group_key]
                window.append(pending.pop())
                if pending:
                    cursor = (cursor + 1) % len(active_groups)
                else:
                    active_groups.pop(cursor)
                    del group_to_pending[group_key]
                    if active_groups:
                        cursor %= len(active_groups)
                    else:
                        cursor = 0
            preserve_prefix = min(active_group_count, len(window))
            for ref in window[:preserve_prefix]:
                yield ref
            yield from self._reorder_record_window_for_chunk_locality(window[preserve_prefix:])

    def _reorder_record_window_for_chunk_locality(self, window: Sequence[tuple[int, int]]) -> Iterator[tuple[int, int]]:
        if not window:
            return
        per_chunk: OrderedDict[int, deque[tuple[int, int]]] = OrderedDict()
        for ref in window:
            per_chunk.setdefault(int(ref[0]), deque()).append(ref)

        active_chunk_indices = list(per_chunk.keys())
        while active_chunk_indices:
            next_active_chunk_indices: list[int] = []
            for chunk_idx in active_chunk_indices:
                queue = per_chunk[chunk_idx]
                take_n = min(self.sampling_max_records_per_chunk_round, len(queue))
                for _ in range(take_n):
                    yield queue.popleft()
                if queue:
                    next_active_chunk_indices.append(chunk_idx)
            active_chunk_indices = next_active_chunk_indices

    def _build_balanced_group_index(
        self,
        chunk_index_payloads: Sequence[dict[str, Any]],
    ) -> dict[tuple[str, ...], list[tuple[int, int]]]:
        group_to_refs: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        for chunk_idx, payload in enumerate(chunk_index_payloads):
            records = payload.get("records", [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                if not self._record_index_is_runtime_compatible(record):
                    continue
                record_idx = int(record.get("record_idx", 0))
                group_key = self._balanced_group_key_for_record(record)
                group_to_refs.setdefault(group_key, []).append((int(self._effective_chunk_indices[chunk_idx]), record_idx))
        return group_to_refs

    def _record_index_is_runtime_compatible(self, record: dict[str, Any]) -> bool:
        if not self.runtime_enforce_stage_compatibility:
            return True
        weight_shape = record.get("weight_shape", [])
        if not isinstance(weight_shape, Sequence) or len(weight_shape) < 2:
            return False
        d_in = int(weight_shape[0])
        d_out = int(weight_shape[1])
        if self.runtime_min_d_in > 0 and d_in < self.runtime_min_d_in:
            return False
        if self.runtime_min_d_out > 0 and d_out < self.runtime_min_d_out:
            return False
        return True

    def _build_runtime_compatible_record_refs_by_chunk(
        self,
        chunk_index_payloads: Sequence[dict[str, Any]],
    ) -> tuple[dict[int, tuple[int, ...]], int, int]:
        compatible_refs_by_chunk: dict[int, tuple[int, ...]] = {}
        compatible_count = 0
        filtered_count = 0
        for payload_idx, payload in enumerate(chunk_index_payloads):
            records = payload.get("records", [])
            if not isinstance(records, list):
                continue
            chunk_refs: list[int] = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                if self._record_index_is_runtime_compatible(record):
                    chunk_refs.append(int(record.get("record_idx", 0)))
                    compatible_count += 1
                else:
                    filtered_count += 1
            if chunk_refs:
                compatible_refs_by_chunk[int(self._effective_chunk_indices[payload_idx])] = tuple(chunk_refs)
        return compatible_refs_by_chunk, compatible_count, filtered_count

    @staticmethod
    def _worker_shard_chunk_indices(
        chunk_indices: Sequence[int],
        *,
        worker_id: int,
        worker_count: int,
    ) -> list[int]:
        if worker_count <= 1:
            return [int(chunk_idx) for chunk_idx in chunk_indices]
        return [int(chunk_idx) for idx, chunk_idx in enumerate(chunk_indices) if idx % worker_count == worker_id]

    def _balanced_group_key_for_record(self, record: dict[str, Any]) -> tuple[str, ...]:
        components: list[str] = []
        for key in self.sampling_group_keys:
            if key == "dataset":
                components.append(str(record.get("primary_dataset", "<unknown_dataset>")))
            elif key == "model":
                components.append(str(record.get("model_name", "<unknown_model>")))
            elif key == "layer_type":
                components.append(str(record.get("layer_type", "other_linear")))
            elif key == "depth":
                components.append(str(record.get("depth_label", "unknown")))
            elif key == "shape":
                components.append(str(record.get("shape", "0x0")))
            elif key == "source":
                components.append(str(record.get("source_key", "")))
            elif key == "layer":
                components.append(str(record.get("layer_name", "<unknown_layer>")))
            elif key == "d_in_bucket":
                components.append(str(record.get("d_in_bucket", "0")))
            elif key == "d_out_bucket":
                components.append(str(record.get("d_out_bucket", "0")))
            elif key == "num_params_bucket":
                components.append(str(record.get("num_params_bucket", "0")))
            else:
                raise ValueError(f"Unsupported balanced group key: {key!r}")
        return tuple(components)

    def _load_chunk_index_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        total = len(self._effective_chunk_indices)
        t0 = time.perf_counter()
        last_log_t = t0
        for offset, chunk_idx in enumerate(self._effective_chunk_indices, start=1):
            payloads.append(self._load_chunk_index_payload(chunk_idx))
            now = time.perf_counter()
            if offset == total or offset == 1 or offset % 100 == 0 or now - last_log_t >= 10.0:
                elapsed = max(1e-6, now - t0)
                self.logger.info(
                    "Offline BigVAE chunk index load progress: chunks=%s/%s rate_chunks_per_s=%.2f elapsed_s=%.1f",
                    offset,
                    total,
                    float(offset) / elapsed,
                    elapsed,
                )
                last_log_t = now
        return payloads

    def _load_chunk_index_payload(self, chunk_idx: int) -> dict[str, Any]:
        chunk_path = self._chunk_paths[chunk_idx]
        chunk_id = chunk_path.stem
        index_path = _chunk_index_path(self.chunk_index_dir, chunk_id)
        if index_path.exists():
            payload = _load_json_payload(index_path)
            records = payload.get("records", [])
            if isinstance(records, list):
                return payload
            raise TypeError(f"Offline BigVAE chunk index has invalid records payload: {index_path}")

        self.logger.warning(
            "Offline BigVAE chunk index missing for %s; generating it by scanning the chunk once",
            chunk_path,
        )
        chunk_payload = self._load_x_chunk(chunk_idx)
        records = chunk_payload.get("records", [])
        if not isinstance(records, list):
            raise TypeError(f"Offline BigVAE chunk {chunk_path} has invalid records payload")
        payload = self._build_chunk_index_payload_from_loaded_chunk(
            chunk_id=chunk_id,
            chunk_path=chunk_path,
            records=records,
        )
        _atomic_write_json(index_path, payload)
        return payload

    def _build_chunk_index_payload_from_loaded_chunk(
        self,
        *,
        chunk_id: str,
        chunk_path: Path,
        records: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        index_records = []
        for record_idx, record in enumerate(records):
            source_key = str(record.get("source_key", "") or "")
            source_meta = self._sources_index.get(source_key, {})
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

    def _load_x_chunk(self, chunk_idx: int) -> dict[str, Any]:
        if chunk_idx in self._x_chunk_cache:
            payload = self._x_chunk_cache.pop(chunk_idx)
            self._x_chunk_cache[chunk_idx] = payload
            return payload

        chunk_path = self._chunk_paths[chunk_idx]
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Offline BigVAE chunk payload must be a dict, got {type(payload)!r}: {chunk_path}")
        self._x_chunk_cache[chunk_idx] = payload
        while len(self._x_chunk_cache) > self.x_chunk_cache_size:
            self._x_chunk_cache.popitem(last=False)
        return payload

    def _shared_sample_from_record_ref(self, *, chunk_idx: int, record_idx: int) -> SharedSample:
        payload = self._load_x_chunk(chunk_idx)
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise TypeError(f"Offline BigVAE chunk {self._chunk_paths[chunk_idx]} has invalid records payload")
        record = records[record_idx]
        source_key = str(record.get("source_key", ""))
        x = record.get("x")
        if not torch.is_tensor(x):
            raise TypeError(f"Offline BigVAE record in {self._chunk_paths[chunk_idx]} is missing tensor 'x'")
        x_cpu = _prepare_cpu_sample_tensor(x)
        weight = self._load_weight(source_key)
        return SharedSample(
            model_name=str(record.get("model_name", "")),
            layer_name=str(record.get("layer_name", "")),
            weight=weight,
            x=x_cpu,
            y=torch.empty((int(x_cpu.shape[0]), 0), dtype=torch.float32),
            meta=dict(record.get("meta", {}) or {}),
        )

    def _load_weight(self, source_key: str) -> torch.Tensor:
        if source_key in self._weight_cache:
            weight = self._weight_cache.pop(source_key)
            self._weight_cache[source_key] = weight
            return weight

        weight_path = _weight_path(self.weights_dir, source_key)
        if not weight_path.exists():
            raise FileNotFoundError(f"Offline BigVAE weight not found for source_key={source_key}: {weight_path}")
        payload = torch.load(weight_path, map_location="cpu", weights_only=False)
        weight = payload.get("weight")
        if not torch.is_tensor(weight):
            raise TypeError(f"Offline BigVAE weight payload is invalid for source_key={source_key}")
        weight_cpu = _prepare_cpu_sample_tensor(weight)
        self._weight_cache[source_key] = weight_cpu
        while len(self._weight_cache) > self.weight_cache_size:
            self._weight_cache.popitem(last=False)
        return weight_cpu

__all__ = [
    'OfflineBigVAEDataset',
]
