from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from big_vae.datasets.offline import infer_layer_depth, infer_layer_type
from training.big_vae_latent_diffusion import latent_diffusion_layer_type_to_id

from .source_slicing import _prepare_cpu_tensor

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
        self._record_schema_cache: dict[str, int | bool] | None = None
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
        record_schema = self._infer_record_schema()
        payload["z_dim"] = max(
            int(payload.get("z_dim", 0)),
            int(self._stats.get("z_dim", 0)),
            int(record_schema.get("z_dim", 0)),
        )
        payload["cond_dim"] = max(
            int(payload.get("cond_dim", 0)),
            int(record_schema.get("cond_dim", 0)),
        )
        payload["cond_global_dim"] = max(
            int(payload.get("cond_global_dim", 0)),
            int(self._stats.get("cond_global_dim", 0)),
            int(record_schema.get("cond_global_dim", 0)),
        )
        payload["has_decoder_aux_tensors"] = bool(
            payload.get("has_decoder_aux_tensors", False) or bool(record_schema.get("has_decoder_aux_tensors", False))
        )
        payload["latent_stats_count"] = int(self._stats.get("count", 0))
        return payload

    def latent_stats(self) -> dict[str, torch.Tensor | int]:
        return {
            "latent_mean": self._stats["latent_mean"].clone(),
            "latent_std": self._stats["latent_std"].clone(),
            "count": int(self._stats.get("count", 0)),
            "cond_global_dim": int(self._stats.get("cond_global_dim", 0)),
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

    def _infer_record_schema(self) -> dict[str, int | bool]:
        if self._record_schema_cache is not None:
            return dict(self._record_schema_cache)

        inferred = {
            "z_dim": 0,
            "cond_dim": 0,
            "cond_global_dim": 0,
            "has_decoder_aux_tensors": False,
        }
        for chunk_idx in range(len(self._chunk_paths)):
            payload = self._load_chunk(chunk_idx)
            records = payload.get("records", [])
            if not isinstance(records, list) or not records:
                continue
            record = records[0]
            if not isinstance(record, dict):
                continue
            latent_mu = record.get("latent_mu")
            cond_patch = record.get("cond_patch")
            cond_global = record.get("cond_global")
            inferred["z_dim"] = int(latent_mu.numel()) if torch.is_tensor(latent_mu) else 0
            inferred["cond_dim"] = int(cond_patch.shape[1]) if torch.is_tensor(cond_patch) and cond_patch.ndim == 2 else 0
            inferred["cond_global_dim"] = (
                int(cond_global.numel()) if torch.is_tensor(cond_global) and cond_global.ndim == 1 else 0
            )
            inferred["has_decoder_aux_tensors"] = bool(
                torch.is_tensor(record.get("X"))
                and torch.is_tensor(record.get("W"))
                and torch.is_tensor(record.get("x_mask"))
                and torch.is_tensor(record.get("d_in_mask"))
                and torch.is_tensor(record.get("d_out_mask"))
            )
            break
        self._record_schema_cache = dict(inferred)
        return dict(inferred)


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
    has_cond_global = any(torch.is_tensor(item.get("cond_global")) for item in items)
    cond_global: torch.Tensor | None = None
    if has_cond_global:
        if not all(torch.is_tensor(item.get("cond_global")) for item in items):
            raise ValueError("cond_global must be present for every item in the batch or none")
        first_cond_global = items[0].get("cond_global")
        assert torch.is_tensor(first_cond_global)
        cond_global_dim = int(first_cond_global.numel())
        cond_global = torch.zeros(batch, cond_global_dim, dtype=torch.float32)
    model_names: list[str] = []
    layer_names: list[str] = []
    source_keys: list[str] = []
    metas: list[dict[str, Any]] = []
    d_in_list: list[int] = []
    d_out_list: list[int] = []
    layer_type_ids: list[int] = []
    layer_depths: list[float] = []
    has_decoder_aux_tensors = any(torch.is_tensor(item.get("W")) or torch.is_tensor(item.get("X")) for item in items)
    X: torch.Tensor | None = None
    W: torch.Tensor | None = None
    x_mask: torch.Tensor | None = None
    d_in_mask: torch.Tensor | None = None
    d_out_mask: torch.Tensor | None = None
    if has_decoder_aux_tensors:
        if not all(
            torch.is_tensor(item.get("W"))
            and torch.is_tensor(item.get("X"))
            and torch.is_tensor(item.get("x_mask"))
            and torch.is_tensor(item.get("d_in_mask"))
            and torch.is_tensor(item.get("d_out_mask"))
            for item in items
        ):
            raise ValueError("decoder auxiliary tensors must be present for every item in the batch or none")
        max_rows = max(int(item["X"].shape[0]) for item in items)
        max_d_in = max(int(item["W"].shape[0]) for item in items)
        max_d_out = max(int(item["W"].shape[1]) for item in items)
        X = torch.zeros(batch, max_rows, max_d_in, dtype=torch.float32)
        W = torch.zeros(batch, max_d_in, max_d_out, dtype=torch.float32)
        x_mask = torch.zeros(batch, max_rows, dtype=torch.bool)
        d_in_mask = torch.zeros(batch, max_d_in, dtype=torch.bool)
        d_out_mask = torch.zeros(batch, max_d_out, dtype=torch.bool)

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
        if has_cond_global:
            item_cond_global = item.get("cond_global")
            assert cond_global is not None
            if not torch.is_tensor(item_cond_global) or item_cond_global.ndim != 1 or int(item_cond_global.numel()) != int(cond_global.shape[1]):
                raise ValueError(
                    f"cond_global shape mismatch at item {idx}: expected {(int(cond_global.shape[1]),)}, "
                    f"got {tuple(item_cond_global.shape) if torch.is_tensor(item_cond_global) else type(item_cond_global)!r}"
                )
            cond_global[idx] = _prepare_cpu_tensor(item_cond_global)
        model_names.append(str(item.get("model_name", "")))
        layer_name = str(item.get("layer_name", ""))
        layer_names.append(layer_name)
        source_keys.append(str(item.get("source_key", "")))
        metas.append(dict(item.get("meta", {}) or {}))
        d_in_list.append(int(item.get("d_in", 0)))
        d_out_list.append(int(item.get("d_out", 0)))
        layer_type_name = str(item.get("layer_type", "")).strip() or infer_layer_type(layer_name)
        layer_depth_value = item.get("layer_depth", infer_layer_depth(layer_name))
        layer_type_ids.append(latent_diffusion_layer_type_to_id(layer_type_name))
        layer_depths.append(float(layer_depth_value) if layer_depth_value is not None else -1.0)
        if has_decoder_aux_tensors:
            item_X = _prepare_cpu_tensor(item["X"])
            item_W = _prepare_cpu_tensor(item["W"])
            item_x_mask = item["x_mask"].detach().to(device="cpu", dtype=torch.bool).contiguous()
            item_d_in_mask = item["d_in_mask"].detach().to(device="cpu", dtype=torch.bool).contiguous()
            item_d_out_mask = item["d_out_mask"].detach().to(device="cpu", dtype=torch.bool).contiguous()
            rows = int(item_X.shape[0])
            d_in = int(item_W.shape[0])
            d_out = int(item_W.shape[1])
            assert X is not None and W is not None and x_mask is not None and d_in_mask is not None and d_out_mask is not None
            X[idx, :rows, :d_in] = item_X
            W[idx, :d_in, :d_out] = item_W
            x_mask[idx, :rows] = item_x_mask
            d_in_mask[idx, :d_in] = item_d_in_mask
            d_out_mask[idx, :d_out] = item_d_out_mask

    batch_payload = {
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
        "layer_type_ids": torch.tensor(layer_type_ids, dtype=torch.long),
        "layer_depths": torch.tensor(layer_depths, dtype=torch.float32),
    }
    if cond_global is not None:
        batch_payload["cond_global"] = cond_global
    if has_decoder_aux_tensors:
        assert X is not None and W is not None and x_mask is not None and d_in_mask is not None and d_out_mask is not None
        batch_payload["X"] = X
        batch_payload["W"] = W
        batch_payload["x_mask"] = x_mask
        batch_payload["d_in_mask"] = d_in_mask
        batch_payload["d_out_mask"] = d_out_mask
    return batch_payload
