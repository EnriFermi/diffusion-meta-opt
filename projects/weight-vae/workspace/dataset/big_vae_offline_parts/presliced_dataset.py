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
from dataset.big_vae_offline_parts.offline_dataset import *
from dataset.big_vae_offline_parts.presliced_utils import *

class PreslicedBigVAEWriter:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        spec: dict[str, Any],
        chunk_size_slices: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.spec = dict(spec)
        self.chunk_size_slices = max(1, int(chunk_size_slices))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.chunks_dir = self.root_dir / "chunks"
        self.manifest_path = self.root_dir / "manifest.json"
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.spec.get("seed", 42)))
        self._chunk_buffer: list[dict[str, Any]] = []
        self._chunk_index = 0
        self._num_slices = 0
        self._seen_sources = 0
        self._skipped_invalid = 0
        self._model_counts: Counter[str] = Counter()
        self._layer_counts: Counter[str] = Counter()
        self._dataset_counts: Counter[str] = Counter()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    @property
    def num_slices(self) -> int:
        return int(self._num_slices)

    @property
    def seen_sources(self) -> int:
        return int(self._seen_sources)

    def ingest_source_sample(self, sample: SharedSample, *, remaining_slices: int) -> int:
        if remaining_slices <= 0:
            return 0
        self._seen_sources += 1
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
            return 0

        x_cpu = _prepare_cpu_sample_tensor(x)
        W_cpu = _prepare_cpu_sample_tensor(W)
        patch_size = int(self.spec["patch_size"])
        max_T_patches = int(self.spec["max_T_patches"])
        max_d_out = int(self.spec["max_d_out"])
        target_x_rows = int(self.spec["target_x_rows"])
        target_d_in = int(self.spec["target_d_in"])
        target_d_out = int(self.spec["target_d_out"])

        T_total = int(W_cpu.shape[0]) // int(patch_size)
        if T_total <= 0:
            self._skipped_invalid += 1
            return 0
        T_use = min(int(max_T_patches), T_total)
        d_out_use = min(int(max_d_out), int(W_cpu.shape[1]))
        if T_use <= 0 or d_out_use <= 0:
            self._skipped_invalid += 1
            return 0

        row_groups = _presliced_index_groups(num_items=T_total, group_size=T_use, rng=self._rng)
        col_groups = _presliced_index_groups(num_items=int(W_cpu.shape[1]), group_size=d_out_use, rng=self._rng)
        capacities: list[int] = []
        if row_groups is not None:
            capacities.append(len(row_groups))
        if col_groups is not None:
            capacities.append(len(col_groups))
        source_capacity = max(1, min(capacities) if capacities else 1)
        source_capacity = min(int(source_capacity), int(remaining_slices))

        x_base, x_mask = _presliced_pad_x(x_cpu, target_x_rows, self._rng)
        meta = dict(getattr(sample, "meta", {}) or {})
        model_name = str(getattr(sample, "model_name", "") or "").strip()
        layer_name = str(getattr(sample, "layer_name", "") or "").strip()
        source_key = _source_key(model_name=model_name, layer_name=layer_name)
        written = 0
        for slice_idx in range(source_capacity):
            W_i = W_cpu
            x_i = x_base
            if row_groups is not None:
                patch_idx = row_groups[slice_idx]
                offsets = torch.arange(int(patch_size), dtype=torch.long)
                row_idx = (patch_idx.unsqueeze(1) * int(patch_size) + offsets.unsqueeze(0)).flatten()
                row_idx = row_idx.clamp(max=int(W_cpu.shape[0]) - 1)
                W_i = W_i[row_idx, :]
                x_i = x_i[:, row_idx]
            if col_groups is not None:
                col_idx = col_groups[slice_idx]
                W_i = W_i[:, col_idx]

            try:
                W_pad, x_pad, d_in_mask, d_out_mask = _presliced_pad_matrix(
                    W_i,
                    x_i,
                    target_d_in=target_d_in,
                    target_d_out=target_d_out,
                )
            except ValueError:
                self._skipped_invalid += 1
                continue

            record_meta = dict(meta)
            record_meta.update(
                {
                    "presliced": True,
                    "presliced_source_key": source_key,
                    "presliced_source_slice_idx": int(slice_idx),
                    "presliced_original_weight_shape": [int(dim) for dim in W_cpu.shape],
                    "presliced_original_x_shape": [int(dim) for dim in x_cpu.shape],
                }
            )
            record = {
                "model_name": model_name,
                "layer_name": layer_name,
                "source_key": source_key,
                "W": W_pad,
                "x": x_pad,
                "x_mask": x_mask.clone(),
                "d_in_mask": d_in_mask,
                "d_out_mask": d_out_mask,
                "meta": record_meta,
            }
            self._chunk_buffer.append(record)
            self._num_slices += 1
            written += 1
            self._model_counts[model_name or "<unknown_model>"] += 1
            self._layer_counts[layer_name or "<unknown_layer>"] += 1
            for dataset_name in _sample_dataset_names(record_meta):
                self._dataset_counts[dataset_name] += 1
            if len(self._chunk_buffer) >= self.chunk_size_slices:
                self._flush_chunk()
        return written

    def close(self) -> dict[str, Any]:
        self._flush_chunk()
        manifest = self._build_manifest()
        _atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _flush_chunk(self) -> None:
        if not self._chunk_buffer:
            return
        chunk_id = f"presliced_chunk_{self._chunk_index:08d}"
        self._chunk_index += 1
        chunk_path = self.chunks_dir / f"{chunk_id}.pt"
        torch.save(
            {
                "format_version": PRESLICED_BIG_VAE_FORMAT_VERSION,
                "chunk_id": chunk_id,
                "records": self._chunk_buffer,
                "spec": self.spec,
            },
            chunk_path,
        )
        self._chunk_buffer = []

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "format_version": PRESLICED_BIG_VAE_FORMAT_VERSION,
            "created_at": float(time.time()),
            "root_dir": str(self.root_dir),
            "fingerprint": str(self.spec.get("fingerprint", "")),
            "spec": self.spec,
            "num_slices": int(self._num_slices),
            "seen_sources": int(self._seen_sources),
            "skipped_invalid": int(self._skipped_invalid),
            "chunk_size_slices": int(self.chunk_size_slices),
            "num_chunks": int(self._chunk_index),
            "actual_size_bytes": int(_directory_size_bytes(self.root_dir)),
            "top_models": [[name, int(count)] for name, count in self._model_counts.most_common(16)],
            "top_layers": [[name, int(count)] for name, count in self._layer_counts.most_common(16)],
            "top_datasets": [[name, int(count)] for name, count in self._dataset_counts.most_common(16)],
        }


class PreslicedBigVAEDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        root_dir: str | Path,
        shuffle_chunks: bool = True,
        shuffle_slices_within_chunk: bool = True,
        repeat: bool = True,
        seed: int = 42,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        shard_by_chunk: bool = True,
        chunk_cache_size: int = 4,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.chunks_dir = self.root_dir / "chunks"
        self.manifest_path = self.root_dir / "manifest.json"
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_slices_within_chunk = bool(shuffle_slices_within_chunk)
        self.repeat = bool(repeat)
        self.seed = int(seed)
        self.shard_rank = max(0, int(shard_rank))
        self.shard_world_size = max(1, int(shard_world_size))
        self.shard_by_chunk = bool(shard_by_chunk)
        self.chunk_cache_size = max(1, int(chunk_cache_size))
        self._chunk_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._manifest = self._load_manifest()
        self._chunk_paths = sorted(self.chunks_dir.glob("*.pt"))
        if not self._chunk_paths:
            raise FileNotFoundError(f"No presliced BigVAE chunks found under {self.chunks_dir}")
        self._effective_chunk_indices = self._resolve_effective_chunk_indices()

    def __iter__(self) -> Iterator[SharedSample]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = int(worker_info.id) if worker_info is not None else 0
        worker_count = int(worker_info.num_workers) if worker_info is not None else 1
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            chunk_indices = list(self._effective_chunk_indices)
            if self.shuffle_chunks and len(chunk_indices) > 1:
                rng.shuffle(chunk_indices)
            chunk_indices = self._worker_shard_chunk_indices(chunk_indices, worker_id=worker_id, worker_count=worker_count)
            for chunk_idx in chunk_indices:
                payload = self._load_chunk(chunk_idx)
                records = payload.get("records", [])
                if not isinstance(records, list):
                    raise TypeError(f"Presliced BigVAE chunk {self._chunk_paths[chunk_idx]} has invalid records payload")
                order = list(range(len(records)))
                if self.shuffle_slices_within_chunk and len(order) > 1:
                    rng.shuffle(order)
                for record_idx in order:
                    yield self._shared_sample_from_record(record=records[record_idx])
            if not self.repeat:
                return
            epoch += 1

    def maybe_collect(self, step_idx: int) -> None:
        del step_idx

    def cache_size(self) -> int:
        return int(self._manifest.get("num_slices", 0))

    def summary(self) -> dict[str, Any]:
        return dict(self._manifest)

    def close(self) -> None:
        self._chunk_cache.clear()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Presliced BigVAE manifest not found: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Presliced BigVAE manifest must be a dict, got {type(payload)}")
        return payload

    def _resolve_effective_chunk_indices(self) -> list[int]:
        if self.shard_by_chunk and self.shard_world_size > 1:
            chunk_indices = list(range(len(self._chunk_paths)))[self.shard_rank :: self.shard_world_size]
            if not chunk_indices:
                raise RuntimeError(
                    "Presliced BigVAE dataset shard is empty. "
                    f"rank={self.shard_rank} world_size={self.shard_world_size} chunk_count={len(self._chunk_paths)}"
                )
            return chunk_indices
        return list(range(len(self._chunk_paths)))

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

    def _load_chunk(self, chunk_idx: int) -> dict[str, Any]:
        if chunk_idx in self._chunk_cache:
            payload = self._chunk_cache.pop(chunk_idx)
            self._chunk_cache[chunk_idx] = payload
            return payload

        chunk_path = self._chunk_paths[chunk_idx]
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Presliced BigVAE chunk payload must be a dict, got {type(payload)!r}: {chunk_path}")
        self._chunk_cache[chunk_idx] = payload
        while len(self._chunk_cache) > self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return payload

    @staticmethod
    def _shared_sample_from_record(*, record: dict[str, Any]) -> SharedSample:
        W = record.get("W")
        x = record.get("x")
        x_mask = record.get("x_mask")
        d_in_mask = record.get("d_in_mask")
        d_out_mask = record.get("d_out_mask")
        if not torch.is_tensor(W) or not torch.is_tensor(x):
            raise TypeError("Presliced BigVAE record is missing tensor W/x")
        if not torch.is_tensor(x_mask) or not torch.is_tensor(d_in_mask) or not torch.is_tensor(d_out_mask):
            raise TypeError("Presliced BigVAE record is missing mask tensors")
        meta = dict(record.get("meta", {}) or {})
        meta["x_mask"] = x_mask
        meta["d_in_mask"] = d_in_mask
        meta["d_out_mask"] = d_out_mask
        meta["source_key"] = str(record.get("source_key", ""))
        return SharedSample(
            model_name=str(record.get("model_name", "")),
            layer_name=str(record.get("layer_name", "")),
            weight=_prepare_cpu_sample_tensor(W),
            x=_prepare_cpu_sample_tensor(x),
            y=torch.empty((int(x.shape[0]), 0), dtype=torch.float32),
            meta=meta,
        )


def ensure_presliced_big_vae_dataset(
    cfg: DictConfig,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger_local = logger or logging.getLogger("dataset.big_vae_offline")
    preslicing_cfg = _preslicing_cfg(cfg)
    logger_local.info(
        "Presliced BigVAE dataset check starting: root=%s source_root=%s num_slices=%s",
        _preslicing_root(cfg),
        _preslicing_source_root(cfg),
        int(preslicing_cfg.get("num_slices", 0)),
    )
    spec = resolve_presliced_big_vae_spec(cfg, logger=logger_local)
    root_dir = Path(str(spec["root_dir"]))
    ok, manifest, reason = _presliced_manifest_matches(root_dir, spec)
    if ok and manifest is not None:
        logger_local.info(
            "Presliced BigVAE dataset cache hit: root=%s num_slices=%s fingerprint=%s",
            root_dir,
            int(manifest.get("num_slices", 0)),
            str(spec.get("fingerprint", "")),
        )
        return manifest

    logger_local.info(
        "Presliced BigVAE dataset cache miss: root=%s reason=%s; building num_slices=%s "
        "target_x_rows=%s target_d_in=%s target_d_out=%s source_root=%s fingerprint=%s",
        root_dir,
        reason,
        int(spec["num_slices"]),
        int(spec["target_x_rows"]),
        int(spec["target_d_in"]),
        int(spec["target_d_out"]),
        str(spec["source_root_dir"]),
        str(spec["fingerprint"]),
    )
    if manifest is not None:
        logger_local.info(
            "Existing presliced manifest summary: num_slices=%s fingerprint=%s format_version=%s",
            int(manifest.get("num_slices", 0)),
            str(manifest.get("fingerprint", "")),
            int(manifest.get("format_version", 0)),
        )

    tmp_root = root_dir.parent / f".{root_dir.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    chunk_size_slices = int(preslicing_cfg.get("chunk_size_slices", 1024))
    log_every_slices = max(1, int(preslicing_cfg.get("log_every_slices", 10000)))
    writer = PreslicedBigVAEWriter(
        root_dir=tmp_root,
        spec=spec,
        chunk_size_slices=chunk_size_slices,
        logger=logger_local,
    )
    t0 = time.perf_counter()
    last_log_slices = 0
    source_dataset: OfflineBigVAEDataset | None = None
    try:
        with offline_big_vae_data_pipeline(cfg, logger=logger_local, rank=0, world_size=1) as (source_dataset, _collector):
            source_iter = iter(source_dataset)
            while writer.num_slices < int(spec["num_slices"]):
                remaining = int(spec["num_slices"]) - int(writer.num_slices)
                sample = next(source_iter)
                written = writer.ingest_source_sample(sample, remaining_slices=remaining)
                if written <= 0:
                    continue
                if writer.num_slices - last_log_slices >= log_every_slices or writer.num_slices >= int(spec["num_slices"]):
                    elapsed = max(1e-6, time.perf_counter() - t0)
                    logger_local.info(
                        "Presliced BigVAE build progress: slices=%s/%s seen_sources=%s skipped_invalid=%s "
                        "rate_slices_per_s=%.2f elapsed_s=%.1f",
                        int(writer.num_slices),
                        int(spec["num_slices"]),
                        int(writer.seen_sources),
                        int(writer._skipped_invalid),
                        float(writer.num_slices) / elapsed,
                        elapsed,
                    )
                    last_log_slices = int(writer.num_slices)
        built_manifest = writer.close()
        built_manifest["root_dir"] = str(root_dir)
        built_manifest["actual_size_bytes"] = int(_directory_size_bytes(tmp_root))
        _atomic_write_json(tmp_root / "manifest.json", built_manifest)
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_root, root_dir)
        final_manifest = _load_json_payload(root_dir / "manifest.json")
        elapsed = max(1e-6, time.perf_counter() - t0)
        logger_local.info(
            "Presliced BigVAE dataset build complete: root=%s num_slices=%s chunks=%s size_gb=%.2f "
            "seen_sources=%s skipped_invalid=%s elapsed_s=%.1f rate_slices_per_s=%.2f",
            root_dir,
            int(final_manifest.get("num_slices", 0)),
            int(final_manifest.get("num_chunks", 0)),
            float(final_manifest.get("actual_size_bytes", 0)) / (1024.0 ** 3),
            int(final_manifest.get("seen_sources", 0)),
            int(final_manifest.get("skipped_invalid", 0)),
            elapsed,
            float(final_manifest.get("num_slices", 0)) / elapsed,
        )
        return final_manifest
    except Exception:
        try:
            shutil.rmtree(tmp_root)
        except FileNotFoundError:
            pass
        raise
    finally:
        if source_dataset is not None:
            source_dataset.close()


@contextlib.contextmanager
def presliced_big_vae_data_pipeline(
    cfg: DictConfig,
    *,
    logger: logging.Logger | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[tuple[PreslicedBigVAEDataset, None]]:
    preslicing_cfg = _preslicing_cfg(cfg)
    spec = resolve_presliced_big_vae_spec(cfg)
    dataset = PreslicedBigVAEDataset(
        root_dir=str(spec["root_dir"]),
        shuffle_chunks=bool(preslicing_cfg.get("shuffle_chunks", True)),
        shuffle_slices_within_chunk=bool(preslicing_cfg.get("shuffle_slices_within_chunk", True)),
        repeat=bool(preslicing_cfg.get("repeat", True)),
        seed=int(preslicing_cfg.get("seed", cfg.data.get("seed", 42))),
        shard_rank=int(rank),
        shard_world_size=max(1, int(world_size)),
        shard_by_chunk=bool(preslicing_cfg.get("shard_by_rank", True)),
        chunk_cache_size=int(preslicing_cfg.get("chunk_cache_size", 4)),
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_offline")
    summary = dataset.summary()
    logger_local.info(
        "Presliced BigVAE dataset ready: root=%s num_slices=%s chunks=%s size_gb=%.2f "
        "target_x_rows=%s target_d_in=%s target_d_out=%s shard_rank=%s shard_world_size=%s",
        str(spec["root_dir"]),
        int(summary.get("num_slices", 0)),
        int(summary.get("num_chunks", 0)),
        float(summary.get("actual_size_bytes", 0)) / (1024.0 ** 3),
        int(spec["target_x_rows"]),
        int(spec["target_d_in"]),
        int(spec["target_d_out"]),
        int(rank),
        int(world_size),
    )
    try:
        yield dataset, None
    finally:
        dataset.close()


@contextlib.contextmanager
def offline_big_vae_data_pipeline(
    cfg: DictConfig,
    *,
    logger: logging.Logger | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[tuple[OfflineBigVAEDataset, None]]:
    offline_cfg = cfg.train.get("offline_dataset", {})
    if offline_cfg is None:
        offline_cfg = {}
    if not isinstance(offline_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset must be a mapping")
    sampling_cfg = offline_cfg.get("sampling", {})
    if sampling_cfg is None:
        sampling_cfg = {}
    if not isinstance(sampling_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset.sampling must be a mapping")

    patch_size, max_T_patches, max_d_out, _max_x_rows = resolve_big_vae_curriculum_targets(cfg)
    root_dir = str(offline_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("train.offline_dataset.root_dir must be set when using offline_big_vae_data_pipeline")

    dataset = OfflineBigVAEDataset(
        root_dir=root_dir,
        shuffle_chunks=bool(offline_cfg.get("shuffle_chunks", True)),
        shuffle_records_within_chunk=bool(offline_cfg.get("shuffle_records_within_chunk", True)),
        repeat=bool(offline_cfg.get("repeat", True)),
        seed=int(cfg.data.get("seed", 42)),
        shard_rank=int(rank),
        shard_world_size=max(1, int(world_size)),
        shard_by_chunk=bool(offline_cfg.get("shard_by_rank", True)),
        weight_cache_size=int(offline_cfg.get("weight_cache_size", 64)),
        sampling_mode=str(sampling_cfg.get("mode", "random")),
        sampling_group_keys=sampling_cfg.get("group_keys", ("dataset", "model", "layer_type", "depth")),
        sampling_window_size=int(sampling_cfg.get("window_size_records", 2048)),
        sampling_max_records_per_chunk_round=int(sampling_cfg.get("max_records_per_chunk_round", 8)),
        x_chunk_cache_size=int(sampling_cfg.get("x_chunk_cache_size", 4)),
        runtime_enforce_stage_compatibility=bool(offline_cfg.get("runtime_enforce_stage_compatibility", False)),
        runtime_min_d_in=int(patch_size) * int(max_T_patches),
        runtime_min_d_out=int(max_d_out),
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_offline")
    summary = dataset.summary()
    logger_local.info(
        "Offline BigVAE dataset ready: root=%s accepted_records=%s unique_sources=%s actual_size_gb=%.2f "
        "sampling_mode=%s sampling_group_keys=%s runtime_stage_filter=%s compatible_records=%s filtered_records=%s",
        summary.get("root_dir", str(dataset.root_dir)),
        int(summary.get("accepted_records", 0)),
        int(summary.get("unique_sources", 0)),
        float(summary.get("actual_size_gb", 0.0)),
        str(summary.get("sampling_mode", "random")),
        list(summary.get("sampling_group_keys", [])),
        bool(summary.get("runtime_enforce_stage_compatibility", False)),
        int(summary.get("runtime_compatible_records", 0)),
        int(summary.get("runtime_filtered_records", 0)),
    )
    try:
        yield dataset, None
    finally:
        dataset.close()

__all__ = [
    'PreslicedBigVAEWriter',
    'PreslicedBigVAEDataset',
    'ensure_presliced_big_vae_dataset',
    'presliced_big_vae_data_pipeline',
    'offline_big_vae_data_pipeline',
]
