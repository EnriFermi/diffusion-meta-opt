from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import shutil
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from dataset.shared.types import SharedSample


OFFLINE_BIG_VAE_FORMAT_VERSION = 1


def resolve_big_vae_curriculum_targets(cfg: DictConfig) -> tuple[int, int, int, int]:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    stage = max(1, int(train_cfg.get("stage", 1)))
    base_T = int(train_cfg.get("stage_base_T_patches", 4))
    base_d_out = int(train_cfg.get("stage_base_d_out", 16))
    scale = int(train_cfg.get("stage_scale_factor", 2))
    patch_size = max(1, int(model_cfg.get("patch_size", 16)))
    max_T_patches = base_T * (scale ** (stage - 1))
    max_d_out = base_d_out * (scale ** (stage - 1))
    max_x_rows = max(0, int(train_cfg.get("max_x_rows", 0)))
    return patch_size, max_T_patches, max_d_out, max_x_rows


def resolve_offline_target_size_bytes(offline_cfg: dict[str, Any] | DictConfig | None) -> int:
    cfg = dict(offline_cfg or {})
    raw_bytes = cfg.get("target_size_bytes", 0)
    if raw_bytes not in (None, ""):
        try:
            target_size_bytes = int(raw_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("train.offline_dataset.builder.target_size_bytes must be an integer") from exc
        if target_size_bytes > 0:
            return target_size_bytes

    raw_gb = cfg.get("target_size_gb", 0)
    if raw_gb not in (None, ""):
        try:
            target_size_gb = float(raw_gb)
        except (TypeError, ValueError) as exc:
            raise ValueError("train.offline_dataset.builder.target_size_gb must be a number") from exc
        if target_size_gb > 0.0:
            return int(target_size_gb * (1024.0 ** 3))

    raise ValueError(
        "Offline BigVAE dataset build requires a positive target size. "
        "Set train.offline_dataset.builder.target_size_bytes or target_size_gb."
    )


def estimate_stage_slice_capacity(
    *,
    d_in: int,
    d_out: int,
    patch_size: int,
    max_T_patches: int,
    max_d_out: int,
) -> int:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    total_patches = int(d_in) // int(patch_size)
    if total_patches <= 0 or int(d_out) <= 0:
        return 0

    used_patches = min(int(max_T_patches), total_patches)
    used_d_out = min(int(max_d_out), int(d_out))
    if used_patches <= 0 or used_d_out <= 0:
        return 0

    row_groups = 1 if total_patches <= used_patches else total_patches // used_patches
    col_groups = 1 if int(d_out) <= used_d_out else int(d_out) // used_d_out

    if total_patches <= used_patches and int(d_out) <= used_d_out:
        return 1
    return max(1, min(int(row_groups), int(col_groups)))


def _prepare_cpu_sample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        return tensor.to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _sample_dataset_names(meta: dict[str, Any] | None) -> list[str]:
    if not isinstance(meta, dict):
        return []
    image_meta = meta.get("image_meta", [])
    if not isinstance(image_meta, list):
        return []
    dataset_names = {
        str(item.get("dataset_name")).strip()
        for item in image_meta
        if isinstance(item, dict) and str(item.get("dataset_name", "")).strip()
    }
    return sorted(dataset_names)


def _source_key(model_name: str, layer_name: str) -> str:
    payload = f"{str(model_name).strip()}\n{str(layer_name).strip()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _weight_path(weights_dir: Path, source_key: str) -> Path:
    return weights_dir / source_key[:2] / f"{source_key}.pt"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


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
        self.min_d_in = self.patch_size * self.max_T_patches
        self.min_d_out = self.max_d_out
        self.x_chunk_size_records = max(1, int(x_chunk_size_records))
        self.max_samples_per_source = max(0, int(max_samples_per_source))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config_snapshot = dict(config_snapshot or {})

        self.weights_dir = self.root_dir / "weights"
        self.x_chunks_dir = self.root_dir / "x_chunks"
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

        if int(W_cpu.shape[0]) < int(self.min_d_in) or int(W_cpu.shape[1]) < int(self.min_d_out):
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
        self._write_sidecars()

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
                "min_d_in": int(self.min_d_in),
                "min_d_out": int(self.min_d_out),
            },
            "layout": {
                "weights_dir": str(self.weights_dir.relative_to(self.root_dir)),
                "x_chunks_dir": str(self.x_chunks_dir.relative_to(self.root_dir)),
                "sources_index": str(self.sources_index_path.relative_to(self.root_dir)),
            },
            "config_snapshot": self.config_snapshot,
        }

    def _write_sidecars(self) -> None:
        _atomic_write_json(self.sources_index_path, self._build_sources_payload())
        _atomic_write_json(self.manifest_path, self._build_manifest())


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
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.sources_index_path = self.root_dir / "sources.json"
        self.weights_dir = self.root_dir / "weights"
        self.x_chunks_dir = self.root_dir / "x_chunks"
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_records_within_chunk = bool(shuffle_records_within_chunk)
        self.repeat = bool(repeat)
        self.seed = int(seed)
        self.shard_rank = max(0, int(shard_rank))
        self.shard_world_size = max(1, int(shard_world_size))
        self.shard_by_chunk = bool(shard_by_chunk)
        self.weight_cache_size = max(1, int(weight_cache_size))
        self.logger = logging.getLogger(self.__class__.__name__)

        self._weight_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._active_iter: Iterator[SharedSample] | None = None
        self._manifest = self._load_manifest()
        self._sources_index = self._load_sources_index()
        self._chunk_paths = sorted(self.x_chunks_dir.glob("*.pt"))
        if not self._chunk_paths:
            raise FileNotFoundError(f"No offline BigVAE x chunks found under {self.x_chunks_dir}")

    def __iter__(self) -> Iterator[SharedSample]:
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            chunk_paths = list(self._chunk_paths)
            if self.shuffle_chunks and len(chunk_paths) > 1:
                rng.shuffle(chunk_paths)
            if self.shard_by_chunk and self.shard_world_size > 1:
                chunk_paths = chunk_paths[self.shard_rank :: self.shard_world_size]
                if not chunk_paths:
                    raise RuntimeError(
                        "Offline BigVAE dataset shard is empty. "
                        f"rank={self.shard_rank} world_size={self.shard_world_size} chunk_count={len(self._chunk_paths)}"
                    )

            for chunk_path in chunk_paths:
                payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
                records = payload.get("records", [])
                if not isinstance(records, list):
                    raise TypeError(f"Offline BigVAE chunk {chunk_path} has invalid records payload")
                order = list(range(len(records)))
                if self.shuffle_records_within_chunk and len(order) > 1:
                    rng.shuffle(order)
                for idx in order:
                    record = records[idx]
                    source_key = str(record.get("source_key", ""))
                    x = record.get("x")
                    if not torch.is_tensor(x):
                        raise TypeError(f"Offline BigVAE record in {chunk_path} is missing tensor 'x'")
                    x_cpu = _prepare_cpu_sample_tensor(x)
                    weight = self._load_weight(source_key)
                    yield SharedSample(
                        model_name=str(record.get("model_name", "")),
                        layer_name=str(record.get("layer_name", "")),
                        weight=weight,
                        x=x_cpu,
                        y=torch.empty((int(x_cpu.shape[0]), 0), dtype=torch.float32),
                        meta=dict(record.get("meta", {}) or {}),
                    )

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
            "shard_rank": int(self.shard_rank),
            "shard_world_size": int(self.shard_world_size),
            "shard_by_chunk": bool(self.shard_by_chunk),
        }

    def summary(self) -> dict[str, Any]:
        return dict(self._manifest)

    def close(self) -> None:
        self._weight_cache.clear()
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
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_offline")
    summary = dataset.summary()
    logger_local.info(
        "Offline BigVAE dataset ready: root=%s accepted_records=%s unique_sources=%s actual_size_gb=%.2f",
        summary.get("root_dir", str(dataset.root_dir)),
        int(summary.get("accepted_records", 0)),
        int(summary.get("unique_sources", 0)),
        float(summary.get("actual_size_gb", 0.0)),
    )
    try:
        yield dataset, None
    finally:
        dataset.close()


def build_big_vae_offline_dataset(
    cfg: DictConfig,
    *,
    dataset_iter: Iterator[Any],
    logger: logging.Logger | None = None,
    max_seen_samples: int = 0,
) -> dict[str, Any]:
    offline_cfg = cfg.train.get("offline_dataset", {})
    if offline_cfg is None:
        offline_cfg = {}
    if not isinstance(offline_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset must be a mapping")
    builder_cfg = offline_cfg.get("builder", {})
    if builder_cfg is None:
        builder_cfg = {}
    if not isinstance(builder_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset.builder must be a mapping")

    patch_size, max_T_patches, max_d_out, max_x_rows = resolve_big_vae_curriculum_targets(cfg)
    target_size_bytes = resolve_offline_target_size_bytes(builder_cfg)
    logger_local = logger or logging.getLogger("dataset.big_vae_offline")
    cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    root_dir = str(offline_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("train.offline_dataset.root_dir must be set when building an offline BigVAE dataset")
    writer = BigVAEOfflineDatasetWriter(
        root_dir=root_dir,
        target_size_bytes=target_size_bytes,
        patch_size=patch_size,
        max_T_patches=max_T_patches,
        max_d_out=max_d_out,
        max_x_rows=max_x_rows,
        x_chunk_size_records=int(builder_cfg.get("x_chunk_size_records", 128)),
        max_samples_per_source=int(builder_cfg.get("max_samples_per_source", 0)),
        overwrite_existing=bool(builder_cfg.get("overwrite_existing", False)),
        seed=int(cfg.data.get("seed", 42)),
        logger=logger_local,
        config_snapshot=cfg_snapshot if isinstance(cfg_snapshot, dict) else {},
    )

    log_every_seen = max(1, int(builder_cfg.get("log_every_seen_samples", 1000)))
    seen_limit = max(0, int(max_seen_samples or builder_cfg.get("max_seen_samples", 0)))

    try:
        while not writer.reached_target_size:
            if seen_limit > 0 and int(writer.stats()["seen_samples"]) >= seen_limit:
                break
            try:
                sample = next(dataset_iter)
            except StopIteration:
                logger_local.warning(
                    "Offline BigVAE build stopped because dataset iterator was exhausted before the target size was reached"
                )
                break
            writer.ingest(sample)
            stats = writer.stats()
            if int(stats["seen_samples"]) % log_every_seen == 0:
                logger_local.info(
                    "Offline BigVAE build progress: seen=%s accepted=%s size_gb=%.2f/%s unique_sources=%s "
                    "skipped_invalid=%s skipped_incompatible=%s skipped_source_cap=%s",
                    int(stats["seen_samples"]),
                    int(stats["accepted_records"]),
                    float(stats["written_size_gb"]),
                    f"{float(target_size_bytes) / (1024.0 ** 3):.2f}",
                    int(stats["unique_sources"]),
                    int(stats["skipped_invalid"]),
                    int(stats["skipped_incompatible"]),
                    int(stats["skipped_source_cap"]),
                )
    finally:
        summary = writer.close()

    if not writer.reached_target_size and seen_limit > 0:
        logger_local.warning(
            "Offline BigVAE build stopped before target size because max_seen_samples=%s was reached. "
            "actual_size_gb=%.2f target_size_gb=%.2f",
            seen_limit,
            float(summary.get("actual_size_gb", 0.0)),
            float(summary.get("target_size_bytes", 0.0)) / (1024.0 ** 3),
        )
    return summary
