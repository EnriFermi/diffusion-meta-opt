from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .source_slicing import (
    OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
    _atomic_write_json,
    _directory_size_bytes,
    _prepare_cpu_tensor,
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
        store_decoder_aux_tensors: bool = False,
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
        self.store_decoder_aux_tensors = bool(store_decoder_aux_tensors)
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
        self._cond_global_dim: int | None = None
        self._latent_sum: torch.Tensor | None = None
        self._latent_sumsq: torch.Tensor | None = None
        self._prepare_root(overwrite_existing=bool(overwrite_existing))

    def ingest(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("writer is already closed")
        latent_mu = record.get("latent_mu")
        cond_patch = record.get("cond_patch")
        cond_global = record.get("cond_global")
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
        cond_global_cpu: torch.Tensor | None = None
        if cond_global is not None:
            if not torch.is_tensor(cond_global) or cond_global.ndim != 1:
                raise TypeError(f"record.cond_global must be rank-1 tensor when provided, got {type(cond_global)!r}")
            cond_global_cpu = _prepare_cpu_tensor(cond_global)
        patch_mask_cpu = patch_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
        target_X = record.get("X")
        target_W = record.get("W")
        target_x_mask = record.get("x_mask")
        target_d_in_mask = record.get("d_in_mask")
        target_d_out_mask = record.get("d_out_mask")

        z_dim = int(latent_mu_cpu.numel())
        cond_dim = int(cond_patch_cpu.shape[1])
        cond_global_dim = int(cond_global_cpu.numel()) if cond_global_cpu is not None else 0
        if self._z_dim is None:
            self._z_dim = z_dim
            self._latent_sum = torch.zeros(z_dim, dtype=torch.float64)
            self._latent_sumsq = torch.zeros(z_dim, dtype=torch.float64)
        if self._cond_dim is None:
            self._cond_dim = cond_dim
        if self._cond_global_dim is None:
            self._cond_global_dim = cond_global_dim
        if z_dim != int(self._z_dim):
            raise ValueError(f"inconsistent z_dim: expected {self._z_dim}, got {z_dim}")
        if cond_dim != int(self._cond_dim):
            raise ValueError(f"inconsistent cond_dim: expected {self._cond_dim}, got {cond_dim}")
        if cond_global_dim != int(self._cond_global_dim):
            raise ValueError(f"inconsistent cond_global_dim: expected {self._cond_global_dim}, got {cond_global_dim}")

        assert self._latent_sum is not None
        assert self._latent_sumsq is not None
        latent_mu_f64 = latent_mu_cpu.to(dtype=torch.float64)
        self._latent_sum += latent_mu_f64
        self._latent_sumsq += latent_mu_f64.pow(2)

        payload = dict(record)
        payload["latent_mu"] = latent_mu_cpu
        payload["cond_patch"] = cond_patch_cpu
        if cond_global_cpu is not None:
            payload["cond_global"] = cond_global_cpu
        payload["patch_mask"] = patch_mask_cpu
        latent_logvar = payload.get("latent_logvar")
        if torch.is_tensor(latent_logvar):
            payload["latent_logvar"] = _prepare_cpu_tensor(latent_logvar)
        if self.store_decoder_aux_tensors:
            if not torch.is_tensor(target_X) or target_X.ndim != 2:
                raise TypeError("record.X must be rank-2 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_W) or target_W.ndim != 2:
                raise TypeError("record.W must be rank-2 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_x_mask) or target_x_mask.ndim != 1:
                raise TypeError("record.x_mask must be rank-1 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_d_in_mask) or target_d_in_mask.ndim != 1:
                raise TypeError("record.d_in_mask must be rank-1 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_d_out_mask) or target_d_out_mask.ndim != 1:
                raise TypeError("record.d_out_mask must be rank-1 tensor when store_decoder_aux_tensors=true")
            payload["X"] = _prepare_cpu_tensor(target_X)
            payload["W"] = _prepare_cpu_tensor(target_W)
            payload["x_mask"] = target_x_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
            payload["d_in_mask"] = target_d_in_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
            payload["d_out_mask"] = target_d_out_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
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
                "cond_global_dim": int(self._cond_global_dim or 0),
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
            "cond_global_dim": int(self._cond_global_dim or 0),
            "actual_size_bytes": int(_directory_size_bytes(self.root_dir)),
            "actual_size_gb": float(_directory_size_bytes(self.root_dir)) / (1024.0 ** 3),
            "has_decoder_aux_tensors": bool(self.store_decoder_aux_tensors),
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

