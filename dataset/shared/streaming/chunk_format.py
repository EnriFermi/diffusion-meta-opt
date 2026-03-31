from __future__ import annotations

import gzip
import time
from pathlib import Path
from typing import Any

import torch

from dataset.shared.types import SharedSample


def chunk_suffix(compression: str) -> str:
    normalized = str(compression or "none").lower()
    if normalized == "none":
        return ".pt"
    if normalized == "gzip":
        return ".pt.gz"
    raise ValueError(f"Unsupported chunk compression='{compression}'")


def save_chunk(path: str | Path, samples: list[SharedSample], meta: dict[str, Any] | None, compression: str = "none") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(meta or {})
    chunk_id = str(metadata.get("chunk_id") or _chunk_id_from_path(target))
    created_at = float(metadata.get("created_at", time.time()))

    payload = {
        "chunk_id": chunk_id,
        "created_at": created_at,
        "num_samples": len(samples),
        "samples": samples,
        "meta": metadata,
    }

    normalized = str(compression or "none").lower()
    if normalized == "none":
        with target.open("wb") as handle:
            torch.save(payload, handle)
        return

    if normalized == "gzip":
        with gzip.open(target, "wb") as handle:
            torch.save(payload, handle)
        return

    raise ValueError(f"Unsupported chunk compression='{compression}'")


def load_chunk(path: str | Path) -> tuple[list[SharedSample], dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
    else:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Chunk payload must be dict, got {type(payload)}")

    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise TypeError("Chunk payload field 'samples' must be a list")

    meta = {
        "chunk_id": payload.get("chunk_id"),
        "created_at": payload.get("created_at"),
        "num_samples": payload.get("num_samples"),
        "meta": payload.get("meta", {}),
    }

    return samples, meta


def _chunk_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".pt.gz"):
        return name[: -len(".pt.gz")]
    if name.endswith(".pt"):
        return name[: -len(".pt")]
    return path.stem
