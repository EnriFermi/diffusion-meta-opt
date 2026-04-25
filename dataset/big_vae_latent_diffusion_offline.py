from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from dataset.big_vae_offline import OfflineBigVAEDataset, infer_layer_depth, infer_layer_type
from dataset.shared.types import SharedSample
from training.big_vae_latent_diffusion import encode_big_vae_layer_batch


OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION = 1


@dataclass(slots=True)
class _ExplicitSliceTarget:
    patch_size: int
    target_T_patches: int
    target_d_out: int

    @property
    def target_d_in(self) -> int:
        return int(self.patch_size) * int(self.target_T_patches)


def _source_key(model_name: str, layer_name: str) -> str:
    payload = f"{str(model_name).strip()}\n{str(layer_name).strip()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _prepare_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        return tensor.to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _directory_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return total


def _resolve_explicit_slice_target(builder_cfg: Mapping[str, Any], *, patch_size: int) -> _ExplicitSliceTarget:
    target_T_patches = int(builder_cfg.get("target_T_patches", 0))
    target_d_out = int(builder_cfg.get("target_d_out", 0))
    if target_T_patches <= 0:
        raise ValueError("latent_diffusion_dataset.builder.target_T_patches must be > 0")
    if target_d_out <= 0:
        raise ValueError("latent_diffusion_dataset.builder.target_d_out must be > 0")
    return _ExplicitSliceTarget(
        patch_size=int(patch_size),
        target_T_patches=target_T_patches,
        target_d_out=target_d_out,
    )


def _slice_shared_sample_to_explicit_target(
    sample: SharedSample,
    *,
    target: _ExplicitSliceTarget,
    rng: torch.Generator,
) -> SharedSample | None:
    x_cpu = _prepare_cpu_tensor(sample.x)
    weight_cpu = _prepare_cpu_tensor(sample.weight)
    d_in, d_out = int(weight_cpu.shape[0]), int(weight_cpu.shape[1])
    available_full_patches = d_in // int(target.patch_size)
    if available_full_patches < int(target.target_T_patches) or d_out < int(target.target_d_out):
        return None

    if available_full_patches == int(target.target_T_patches):
        patch_idx = torch.arange(available_full_patches, dtype=torch.long)
    else:
        patch_idx = torch.randperm(available_full_patches, generator=rng)[: int(target.target_T_patches)].sort().values
    row_offsets = torch.arange(int(target.patch_size), dtype=torch.long)
    row_idx = (patch_idx.unsqueeze(1) * int(target.patch_size) + row_offsets.unsqueeze(0)).flatten()

    if d_out == int(target.target_d_out):
        col_idx = torch.arange(d_out, dtype=torch.long)
    else:
        col_idx = torch.randperm(d_out, generator=rng)[: int(target.target_d_out)].sort().values

    sliced_weight = weight_cpu.index_select(dim=0, index=row_idx).index_select(dim=1, index=col_idx)
    sliced_x = x_cpu.index_select(dim=1, index=row_idx)
    meta = dict(sample.meta or {})
    meta["latent_diffusion_builder_explicit_slice"] = True
    meta["latent_diffusion_target_T_patches"] = int(target.target_T_patches)
    meta["latent_diffusion_target_d_in"] = int(target.target_d_in)
    meta["latent_diffusion_target_d_out"] = int(target.target_d_out)
    meta["latent_diffusion_source_d_in"] = d_in
    meta["latent_diffusion_source_d_out"] = d_out
    meta["latent_diffusion_available_full_patches"] = int(available_full_patches)
    meta["latent_diffusion_row_patch_idx_preview"] = [int(idx) for idx in patch_idx[:32].tolist()]
    meta["latent_diffusion_col_idx_preview"] = [int(idx) for idx in col_idx[:32].tolist()]

    return SharedSample(
        model_name=str(sample.model_name),
        layer_name=str(sample.layer_name),
        weight=sliced_weight,
        x=sliced_x,
        y=sample.y,
        meta=meta,
    )


class BigVAELatentDiffusionOfflineWriter:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        overwrite_existing: bool,
        chunk_size_records: int,
        patch_size: int,
        target_T_patches: int,
        target_d_out: int,
        logger: logging.Logger | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.root_dir = Path(str(root_dir))
        if not str(self.root_dir).strip():
            raise ValueError("latent diffusion dataset root_dir must be non-empty")
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.chunk_size_records = max(1, int(chunk_size_records))
        self.patch_size = max(1, int(patch_size))
        self.target_T_patches = max(1, int(target_T_patches))
        self.target_d_out = max(1, int(target_d_out))
        self.config_snapshot = dict(config_snapshot or {})
        self.chunks_dir = self.root_dir / "chunks"
        self.manifest_path = self.root_dir / "manifest.json"
        self.stats_path = self.root_dir / "latent_stats.pt"
        self._chunk_buffer: list[dict[str, Any]] = []
        self._chunk_index = 0
        self._accepted_records = 0
        self._closed = False
        self._z_dim: int | None = None
        self._cond_dim: int | None = None
        self._latent_sum: torch.Tensor | None = None
        self._latent_sumsq: torch.Tensor | None = None
        self._prepare_root(overwrite_existing=bool(overwrite_existing))

    def ingest(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("writer is already closed")
        latent_mu = record.get("latent_mu")
        cond_patch = record.get("cond_patch")
        patch_mask = record.get("patch_mask")
        if not torch.is_tensor(latent_mu) or latent_mu.ndim != 1:
            raise TypeError(f"record.latent_mu must be rank-1 tensor, got {type(latent_mu)!r}")
        if not torch.is_tensor(cond_patch) or cond_patch.ndim != 2:
            raise TypeError(f"record.cond_patch must be rank-2 tensor, got {type(cond_patch)!r}")
        if not torch.is_tensor(patch_mask) or patch_mask.ndim != 1:
            raise TypeError(f"record.patch_mask must be rank-1 tensor, got {type(patch_mask)!r}")
        if int(cond_patch.shape[0]) != int(patch_mask.shape[0]):
            raise ValueError(
                f"record.cond_patch and record.patch_mask must align on T, got {tuple(cond_patch.shape)} and {tuple(patch_mask.shape)}"
            )

        latent_mu_cpu = _prepare_cpu_tensor(latent_mu)
        cond_patch_cpu = _prepare_cpu_tensor(cond_patch)
        patch_mask_cpu = patch_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()

        z_dim = int(latent_mu_cpu.numel())
        cond_dim = int(cond_patch_cpu.shape[1])
        if self._z_dim is None:
            self._z_dim = z_dim
            self._latent_sum = torch.zeros(z_dim, dtype=torch.float64)
            self._latent_sumsq = torch.zeros(z_dim, dtype=torch.float64)
        if self._cond_dim is None:
            self._cond_dim = cond_dim
        if z_dim != int(self._z_dim):
            raise ValueError(f"inconsistent z_dim: expected {self._z_dim}, got {z_dim}")
        if cond_dim != int(self._cond_dim):
            raise ValueError(f"inconsistent cond_dim: expected {self._cond_dim}, got {cond_dim}")

        assert self._latent_sum is not None
        assert self._latent_sumsq is not None
        latent_mu_f64 = latent_mu_cpu.to(dtype=torch.float64)
        self._latent_sum += latent_mu_f64
        self._latent_sumsq += latent_mu_f64.pow(2)

        payload = dict(record)
        payload["latent_mu"] = latent_mu_cpu
        payload["cond_patch"] = cond_patch_cpu
        payload["patch_mask"] = patch_mask_cpu
        latent_logvar = payload.get("latent_logvar")
        if torch.is_tensor(latent_logvar):
            payload["latent_logvar"] = _prepare_cpu_tensor(latent_logvar)
        self._chunk_buffer.append(payload)
        self._accepted_records += 1
        if len(self._chunk_buffer) >= self.chunk_size_records:
            self._flush_chunk()

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._build_manifest()
        self._flush_chunk()
        self._write_stats()
        _atomic_write_json(self.manifest_path, self._build_manifest())
        self._closed = True
        return self._build_manifest()

    def _prepare_root(self, *, overwrite_existing: bool) -> None:
        if self.root_dir.exists():
            has_payload = any(self.root_dir.iterdir())
            if has_payload and not overwrite_existing:
                raise FileExistsError(
                    f"latent diffusion dataset root already exists and is not empty: {self.root_dir}. "
                    "Set latent_diffusion_dataset.builder.overwrite_existing=true to replace it."
                )
            if has_payload:
                shutil.rmtree(self.root_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def _flush_chunk(self) -> None:
        if not self._chunk_buffer:
            return
        chunk_id = f"latent_chunk_{self._chunk_index:08d}"
        self._chunk_index += 1
        chunk_path = self.chunks_dir / f"{chunk_id}.pt"
        payload = {
            "format_version": OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
            "chunk_id": chunk_id,
            "records": self._chunk_buffer,
            "created_at": float(time.time()),
        }
        torch.save(payload, chunk_path)
        self._chunk_buffer = []

    def _write_stats(self) -> None:
        if self._z_dim is None or self._latent_sum is None or self._latent_sumsq is None or self._accepted_records <= 0:
            raise RuntimeError("cannot finalize latent diffusion dataset stats without accepted records")
        count = max(1, int(self._accepted_records))
        latent_mean = self._latent_sum / float(count)
        latent_var = (self._latent_sumsq / float(count)) - latent_mean.pow(2)
        latent_std = torch.sqrt(latent_var.clamp_min(1e-6))
        torch.save(
            {
                "latent_mean": latent_mean.to(dtype=torch.float32),
                "latent_std": latent_std.to(dtype=torch.float32),
                "count": int(count),
                "z_dim": int(self._z_dim),
                "cond_dim": int(self._cond_dim or 0),
            },
            self.stats_path,
        )

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "format_version": OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
            "created_at": float(time.time()),
            "root_dir": str(self.root_dir),
            "accepted_records": int(self._accepted_records),
            "num_chunks": int(self._chunk_index),
            "z_dim": int(self._z_dim or 0),
            "cond_dim": int(self._cond_dim or 0),
            "actual_size_bytes": int(_directory_size_bytes(self.root_dir)),
            "actual_size_gb": float(_directory_size_bytes(self.root_dir)) / (1024.0 ** 3),
            "slice_shape": {
                "patch_size": int(self.patch_size),
                "target_T_patches": int(self.target_T_patches),
                "target_d_in": int(self.patch_size) * int(self.target_T_patches),
                "target_d_out": int(self.target_d_out),
            },
            "layout": {
                "chunks_dir": str(self.chunks_dir.relative_to(self.root_dir)),
                "stats_path": str(self.stats_path.relative_to(self.root_dir)),
            },
            "config_snapshot": self.config_snapshot,
        }


class OfflineBigVAELatentDiffusionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        root_dir: str | Path,
        shuffle_chunks: bool = True,
        shuffle_records_within_chunk: bool = True,
        repeat: bool = True,
        seed: int = 42,
        chunk_cache_size: int = 4,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.stats_path = self.root_dir / "latent_stats.pt"
        self.chunks_dir = self.root_dir / "chunks"
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_records_within_chunk = bool(shuffle_records_within_chunk)
        self.repeat = bool(repeat)
        self.seed = int(seed)
        self.chunk_cache_size = max(1, int(chunk_cache_size))
        self.logger = logging.getLogger(self.__class__.__name__)
        self._chunk_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._manifest = self._load_manifest()
        self._stats = self._load_stats()
        self._chunk_paths = sorted(self.chunks_dir.glob("*.pt"))
        if not self._chunk_paths:
            raise FileNotFoundError(f"no latent diffusion chunks found under {self.chunks_dir}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            chunk_indices = list(range(len(self._chunk_paths)))
            if self.shuffle_chunks and len(chunk_indices) > 1:
                rng.shuffle(chunk_indices)
            for chunk_idx in chunk_indices:
                payload = self._load_chunk(chunk_idx)
                records = payload.get("records", [])
                if not isinstance(records, list):
                    raise TypeError(f"chunk {self._chunk_paths[chunk_idx]} has invalid records payload")
                order = list(range(len(records)))
                if self.shuffle_records_within_chunk and len(order) > 1:
                    rng.shuffle(order)
                for record_idx in order:
                    record = records[record_idx]
                    if not isinstance(record, dict):
                        raise TypeError(f"latent diffusion record must be a dict, got {type(record)!r}")
                    yield dict(record)
            if not self.repeat:
                return
            epoch += 1

    def summary(self) -> dict[str, Any]:
        payload = dict(self._manifest)
        payload["latent_stats_count"] = int(self._stats.get("count", 0))
        return payload

    def latent_stats(self) -> dict[str, torch.Tensor | int]:
        return {
            "latent_mean": self._stats["latent_mean"].clone(),
            "latent_std": self._stats["latent_std"].clone(),
            "count": int(self._stats.get("count", 0)),
        }

    def close(self) -> None:
        self._chunk_cache.clear()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"latent diffusion manifest not found: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"latent diffusion manifest must be a dict, got {type(payload)!r}")
        return payload

    def _load_stats(self) -> dict[str, Any]:
        if not self.stats_path.exists():
            raise FileNotFoundError(f"latent diffusion stats not found: {self.stats_path}")
        payload = torch.load(self.stats_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"latent diffusion stats payload must be a dict, got {type(payload)!r}")
        latent_mean = payload.get("latent_mean")
        latent_std = payload.get("latent_std")
        if not torch.is_tensor(latent_mean) or not torch.is_tensor(latent_std):
            raise TypeError("latent diffusion stats must contain tensors 'latent_mean' and 'latent_std'")
        payload["latent_mean"] = _prepare_cpu_tensor(latent_mean)
        payload["latent_std"] = _prepare_cpu_tensor(latent_std)
        return payload

    def _load_chunk(self, chunk_idx: int) -> dict[str, Any]:
        if chunk_idx in self._chunk_cache:
            payload = self._chunk_cache.pop(chunk_idx)
            self._chunk_cache[chunk_idx] = payload
            return payload
        chunk_path = self._chunk_paths[chunk_idx]
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"latent diffusion chunk payload must be a dict, got {type(payload)!r}: {chunk_path}")
        self._chunk_cache[chunk_idx] = payload
        while len(self._chunk_cache) > self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return payload


def collate_big_vae_latent_diffusion_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("items must be non-empty")
    batch = len(items)
    first_latent = items[0].get("latent_mu")
    first_cond_patch = items[0].get("cond_patch")
    if not torch.is_tensor(first_latent) or first_latent.ndim != 1:
        raise TypeError("each item must contain rank-1 tensor 'latent_mu'")
    if not torch.is_tensor(first_cond_patch) or first_cond_patch.ndim != 2:
        raise TypeError("each item must contain rank-2 tensor 'cond_patch'")
    z_dim = int(first_latent.numel())
    cond_dim = int(first_cond_patch.shape[1])
    max_t = max(int(item["cond_patch"].shape[0]) for item in items if torch.is_tensor(item.get("cond_patch")))

    latent_mu = torch.zeros(batch, z_dim, dtype=torch.float32)
    latent_logvar = torch.zeros(batch, z_dim, dtype=torch.float32)
    cond_patch = torch.zeros(batch, max_t, cond_dim, dtype=torch.float32)
    patch_mask = torch.zeros(batch, max_t, dtype=torch.bool)
    model_names: list[str] = []
    layer_names: list[str] = []
    source_keys: list[str] = []
    metas: list[dict[str, Any]] = []
    d_in_list: list[int] = []
    d_out_list: list[int] = []

    for idx, item in enumerate(items):
        item_latent = item.get("latent_mu")
        item_cond_patch = item.get("cond_patch")
        if not torch.is_tensor(item_latent) or not torch.is_tensor(item_cond_patch):
            raise TypeError("each item must contain tensor keys 'latent_mu' and 'cond_patch'")
        if item_latent.ndim != 1 or int(item_latent.numel()) != z_dim:
            raise ValueError(f"latent_mu shape mismatch at item {idx}: expected {(z_dim,)}, got {tuple(item_latent.shape)}")
        if item_cond_patch.ndim != 2 or int(item_cond_patch.shape[1]) != cond_dim:
            raise ValueError(
                f"cond_patch shape mismatch at item {idx}: expected (*,{cond_dim}), got {tuple(item_cond_patch.shape)}"
            )
        current_t = int(item_cond_patch.shape[0])
        latent_mu[idx] = _prepare_cpu_tensor(item_latent)
        item_logvar = item.get("latent_logvar")
        if torch.is_tensor(item_logvar):
            latent_logvar[idx] = _prepare_cpu_tensor(item_logvar)
        cond_patch[idx, :current_t] = _prepare_cpu_tensor(item_cond_patch)
        patch_mask[idx, :current_t] = True
        model_names.append(str(item.get("model_name", "")))
        layer_names.append(str(item.get("layer_name", "")))
        source_keys.append(str(item.get("source_key", "")))
        metas.append(dict(item.get("meta", {}) or {}))
        d_in_list.append(int(item.get("d_in", 0)))
        d_out_list.append(int(item.get("d_out", 0)))

    return {
        "latent_mu": latent_mu,
        "latent_logvar": latent_logvar,
        "cond_patch": cond_patch,
        "patch_mask": patch_mask,
        "model_names": model_names,
        "layer_names": layer_names,
        "source_keys": source_keys,
        "meta": metas,
        "d_in": torch.tensor(d_in_list, dtype=torch.long),
        "d_out": torch.tensor(d_out_list, dtype=torch.long),
    }


def _pad_source_samples(samples: Sequence[Any]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")
    max_rows = max(int(sample.x.shape[0]) for sample in samples)
    max_d_in = max(int(sample.weight.shape[0]) for sample in samples)
    max_d_out = max(int(sample.weight.shape[1]) for sample in samples)
    batch = len(samples)

    X = torch.zeros(batch, max_rows, max_d_in, dtype=torch.float32)
    W = torch.zeros(batch, max_d_in, max_d_out, dtype=torch.float32)
    x_mask = torch.zeros(batch, max_rows, dtype=torch.bool)
    d_in_mask = torch.zeros(batch, max_d_in, dtype=torch.bool)
    d_out_mask = torch.zeros(batch, max_d_out, dtype=torch.bool)
    model_names: list[str] = []
    layer_names: list[str] = []
    metas: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        x = _prepare_cpu_tensor(sample.x)
        weight = _prepare_cpu_tensor(sample.weight)
        rows, d_in = x.shape
        _, d_out = weight.shape
        X[idx, :rows, :d_in] = x
        W[idx, :d_in, :d_out] = weight
        x_mask[idx, :rows] = True
        d_in_mask[idx, :d_in] = True
        d_out_mask[idx, :d_out] = True
        model_names.append(str(getattr(sample, "model_name", "")))
        layer_names.append(str(getattr(sample, "layer_name", "")))
        metas.append(dict(getattr(sample, "meta", {}) or {}))

    return {
        "X": X,
        "W": W,
        "x_mask": x_mask,
        "d_in_mask": d_in_mask,
        "d_out_mask": d_out_mask,
        "model_names": model_names,
        "layer_names": layer_names,
        "meta": metas,
    }


def build_big_vae_latent_diffusion_offline_dataset(
    cfg: DictConfig,
    *,
    big_vae: Any,
    dataset_iter: Iterator[Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("latent_diffusion_dataset", {})
    if not isinstance(dataset_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset must be a mapping")
    builder_cfg = dataset_cfg.get("builder", {})
    if builder_cfg is None:
        builder_cfg = {}
    if not isinstance(builder_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset.builder must be a mapping")

    root_dir = str(dataset_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("latent_diffusion_dataset.root_dir must be set")
    if not bool(getattr(big_vae, "use_distribution_encoder", False)):
        raise ValueError("latent diffusion dataset build requires frozen BigVAE with distribution encoder enabled")
    target = _resolve_explicit_slice_target(builder_cfg, patch_size=int(big_vae.cfg.patch_size))

    cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    writer = BigVAELatentDiffusionOfflineWriter(
        root_dir=root_dir,
        overwrite_existing=bool(builder_cfg.get("overwrite_existing", False)),
        chunk_size_records=int(builder_cfg.get("chunk_size_records", 128)),
        patch_size=int(target.patch_size),
        target_T_patches=int(target.target_T_patches),
        target_d_out=int(target.target_d_out),
        logger=logger,
        config_snapshot=cfg_snapshot if isinstance(cfg_snapshot, dict) else {},
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_latent_diffusion_offline")
    batch_size = max(1, int(builder_cfg.get("batch_size", 8)))
    encode_batch_size = max(1, int(builder_cfg.get("encode_batch_size", 1)))
    log_every_batches = max(1, int(builder_cfg.get("log_every_batches", 100)))
    max_records = max(0, int(builder_cfg.get("max_records", 0)))
    seed = int(builder_cfg.get("seed", dataset_cfg.get("source", {}).get("seed", 42)))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    pending: list[Any] = []
    total_seen = 0
    total_shape_incompatible = 0

    def _flush_pending() -> None:
        nonlocal pending, total_shape_incompatible
        if not pending:
            return
        sliced_pending: list[SharedSample] = []
        for raw_sample in pending:
            sliced = _slice_shared_sample_to_explicit_target(raw_sample, target=target, rng=rng)
            if sliced is None:
                total_shape_incompatible += 1
                continue
            sliced_pending.append(sliced)
        if not sliced_pending:
            pending = []
            return
        model_device = next(big_vae.parameters()).device
        for start in range(0, len(sliced_pending), encode_batch_size):
            samples_group = sliced_pending[start : start + encode_batch_size]
            batch_payload = _pad_source_samples(samples_group)
            with torch.no_grad():
                encoded = encode_big_vae_layer_batch(
                    big_vae,
                    W=batch_payload["W"].to(device=model_device),
                    X=batch_payload["X"].to(device=model_device),
                    x_mask=batch_payload["x_mask"].to(device=model_device),
                    d_in_mask=batch_payload["d_in_mask"].to(device=model_device),
                    d_out_mask=batch_payload["d_out_mask"].to(device=model_device),
                )
            cond_patch = encoded["cond_patch"]
            patch_mask = encoded["patch_mask"]
            latent_mu = encoded["latent_mu"]
            latent_logvar = encoded["latent_logvar"]
            if not torch.is_tensor(cond_patch) or not torch.is_tensor(patch_mask):
                raise RuntimeError("BigVAE latent diffusion build expected tensor cond_patch and patch_mask")
            if not torch.is_tensor(latent_mu) or not torch.is_tensor(latent_logvar):
                raise RuntimeError("BigVAE latent diffusion build expected tensor latent_mu and latent_logvar")

            batch = len(samples_group)
            for idx in range(batch):
                valid_t = int(patch_mask[idx].to(dtype=torch.long).sum().item())
                writer.ingest(
                    {
                        "source_key": _source_key(batch_payload["model_names"][idx], batch_payload["layer_names"][idx]),
                        "model_name": batch_payload["model_names"][idx],
                        "layer_name": batch_payload["layer_names"][idx],
                        "layer_type": infer_layer_type(batch_payload["layer_names"][idx]),
                        "layer_depth": infer_layer_depth(batch_payload["layer_names"][idx]),
                        "d_in": int(batch_payload["d_in_mask"][idx].to(dtype=torch.long).sum().item()),
                        "d_out": int(batch_payload["d_out_mask"][idx].to(dtype=torch.long).sum().item()),
                        "cond_patch": cond_patch[idx, :valid_t],
                        "patch_mask": torch.ones(valid_t, dtype=torch.bool),
                        "latent_mu": latent_mu[idx],
                        "latent_logvar": latent_logvar[idx],
                        "meta": batch_payload["meta"][idx],
                        "target_T_patches": int(target.target_T_patches),
                        "target_d_out": int(target.target_d_out),
                    }
                )
        pending = []

    for sample in dataset_iter:
        pending.append(sample)
        total_seen += 1
        if len(pending) >= batch_size:
            _flush_pending()
            processed_batches = total_seen // batch_size
            if processed_batches % log_every_batches == 0:
                logger_local.info(
                    "Latent diffusion dataset build progress: seen=%s accepted=%s skipped_shape_incompatible=%s "
                    "target_T_patches=%s target_d_out=%s root=%s",
                    total_seen,
                    int(writer._accepted_records),
                    int(total_shape_incompatible),
                    int(target.target_T_patches),
                    int(target.target_d_out),
                    root_dir,
                )
        if max_records > 0 and total_seen >= max_records:
            break

    _flush_pending()
    summary = writer.close()
    summary["skipped_shape_incompatible"] = int(total_shape_incompatible)
    logger_local.info(
        "Latent diffusion dataset build complete: root=%s accepted_records=%s skipped_shape_incompatible=%s "
        "target_T_patches=%s target_d_out=%s actual_size_gb=%.2f",
        root_dir,
        int(summary.get("accepted_records", 0)),
        int(total_shape_incompatible),
        int(target.target_T_patches),
        int(target.target_d_out),
        float(summary.get("actual_size_gb", 0.0)),
    )
    return summary


@contextlib.contextmanager
def offline_big_vae_latent_diffusion_data_pipeline(
    cfg: DictConfig,
    *,
    logger: logging.Logger | None = None,
) -> Iterator[OfflineBigVAELatentDiffusionDataset]:
    train_dataset_cfg = cfg.get("train", {}).get("dataset", {})
    if train_dataset_cfg is None:
        train_dataset_cfg = {}
    if not isinstance(train_dataset_cfg, (dict, DictConfig)):
        raise TypeError("train.dataset must be a mapping")
    root_dir = str(train_dataset_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("train.dataset.root_dir must be set")
    dataset = OfflineBigVAELatentDiffusionDataset(
        root_dir=root_dir,
        shuffle_chunks=bool(train_dataset_cfg.get("shuffle_chunks", True)),
        shuffle_records_within_chunk=bool(train_dataset_cfg.get("shuffle_records_within_chunk", True)),
        repeat=bool(train_dataset_cfg.get("repeat", True)),
        seed=int(train_dataset_cfg.get("seed", cfg.get("data", {}).get("seed", 42))),
        chunk_cache_size=int(train_dataset_cfg.get("chunk_cache_size", 4)),
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_latent_diffusion_offline")
    summary = dataset.summary()
    logger_local.info(
        "Offline latent diffusion dataset ready: root=%s accepted_records=%s z_dim=%s cond_dim=%s",
        summary.get("root_dir", root_dir),
        int(summary.get("accepted_records", 0)),
        int(summary.get("z_dim", 0)),
        int(summary.get("cond_dim", 0)),
    )
    try:
        yield dataset
    finally:
        dataset.close()


__all__ = [
    "BigVAELatentDiffusionOfflineWriter",
    "OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION",
    "OfflineBigVAELatentDiffusionDataset",
    "build_big_vae_latent_diffusion_offline_dataset",
    "collate_big_vae_latent_diffusion_batch",
    "offline_big_vae_latent_diffusion_data_pipeline",
]
