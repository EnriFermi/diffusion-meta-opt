from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset.shared.streaming.backends.base import ChunkStore
from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
from dataset.shared.streaming.backends.s3 import S3ChunkStore
from dataset.shared.streaming.chunk_reader import ChunkReader
from dataset.shared.streaming.chunk_writer import ChunkWriter
from dataset.shared.streaming.config import normalize_streaming_mode


def resolve_streaming_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(cfg or {})
    payload["mode"] = normalize_streaming_mode(payload.get("mode", "none"))
    payload.setdefault("chunk_size_samples", 256)
    payload.setdefault("chunk_format", {"compression": "none"})
    payload.setdefault("producer", {})
    payload.setdefault("consumer", {})
    payload.setdefault("local_disk", {})
    payload.setdefault("s3", {})
    payload.setdefault("distributed", {})
    return payload


def build_chunk_store(streaming_cfg: dict[str, Any]) -> ChunkStore | None:
    mode = normalize_streaming_mode(streaming_cfg.get("mode"))
    if mode == "none":
        return None

    if mode == "local_disk":
        local_cfg = dict(streaming_cfg.get("local_disk", {}))
        root_dir = local_cfg.get("root_dir")
        if not root_dir:
            raise ValueError("streaming.local_disk.root_dir must be set for mode=local_disk")
        return LocalDiskChunkStore(
            root_dir=root_dir,
            max_ready_chunks=int(local_cfg.get("max_ready_chunks", 200)),
            low_watermark_chunks=int(local_cfg.get("low_watermark_chunks", 100)),
        )

    if mode == "s3_bridge":
        s3_cfg = dict(streaming_cfg.get("s3", {}))
        bucket = str(s3_cfg.get("bucket", "")).strip()
        if not bucket:
            raise ValueError("streaming.s3.bucket must be set for mode=s3_bridge")
        return S3ChunkStore(
            bucket=bucket,
            prefix=str(s3_cfg.get("prefix", "")).strip(),
            region=s3_cfg.get("region"),
            endpoint_url=s3_cfg.get("endpoint_url"),
            max_remote_chunks=int(s3_cfg.get("max_remote_chunks", 1000)),
            staging_prefix=str(s3_cfg.get("staging_prefix", "staging")),
            ready_prefix=str(s3_cfg.get("ready_prefix", "ready")),
        )

    raise ValueError(f"Unsupported streaming.mode='{mode}'")


def build_chunk_writer(streaming_cfg: dict[str, Any], store: ChunkStore) -> ChunkWriter:
    producer_cfg = dict(streaming_cfg.get("producer", {}))
    chunk_format_cfg = dict(streaming_cfg.get("chunk_format", {}))
    spool_dir = producer_cfg.get("local_spool_dir")
    if not spool_dir:
        raise ValueError("streaming.producer.local_spool_dir must be set")
    return ChunkWriter(
        store=store,
        spool_dir=Path(spool_dir),
        chunk_size_samples=int(streaming_cfg.get("chunk_size_samples", 256)),
        compression=str(chunk_format_cfg.get("compression", "none")),
        local_max_chunks=int(producer_cfg.get("local_max_chunks", 64)),
    )


def build_chunk_reader(streaming_cfg: dict[str, Any], store: ChunkStore) -> ChunkReader:
    consumer_cfg = dict(streaming_cfg.get("consumer", {}))
    local_cache_dir = consumer_cfg.get("local_cache_dir")
    if not local_cache_dir:
        raise ValueError("streaming.consumer.local_cache_dir must be set")

    return ChunkReader(
        store=store,
        local_cache_dir=local_cache_dir,
        local_max_chunks=int(consumer_cfg.get("local_max_chunks", 16)),
        delete_remote_after=str(consumer_cfg.get("delete_remote_after", "consume")),
        distributed_cfg=dict(streaming_cfg.get("distributed", {})),
    )
