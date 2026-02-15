from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset.shared.streaming.backends.base import ChunkStore
from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
from dataset.shared.streaming.backends.s3 import S3ChunkStore
from dataset.shared.streaming.chunk_reader import ChunkReader
from dataset.shared.streaming.chunk_writer import ChunkWriter
from dataset.shared.streaming.config import normalize_streaming_mode, resolve_refill_after_consumed_chunks


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

    producer_cfg = dict(payload.get("producer", {}))
    consumer_cfg = dict(payload.get("consumer", {}))
    local_cfg = dict(payload.get("local_disk", {}))
    s3_cfg = dict(payload.get("s3", {}))

    # Canonical producer keys.
    producer_cfg.setdefault("spool_dir", None)
    producer_cfg.setdefault("max_pending_spool_chunks", 64)
    producer_cfg.setdefault(
        "ready_store_dir",
        local_cfg.get("root_dir"),
    )
    producer_cfg.setdefault(
        "ready_store_max_chunks",
        int(local_cfg.get("max_ready_chunks", 200)),
    )

    refill_after = resolve_refill_after_consumed_chunks(payload["mode"], producer_cfg, local_cfg, s3_cfg)
    producer_cfg.setdefault("refill_after_consumed_chunks", refill_after)
    producer_cfg.setdefault("stripe_window_chunks", None)
    if producer_cfg.get("stripe_window_chunks") is None:
        producer_cfg["stripe_window_chunks"] = None

    # Canonical consumer keys.
    consumer_cfg.setdefault("cache_dir", None)
    consumer_cfg.setdefault("prefetch_max_chunks", 16)
    consumer_cfg.setdefault("randomize_within_chunk", True)
    consumer_cfg.setdefault("random_seed", None)

    payload["producer"] = producer_cfg
    payload["consumer"] = consumer_cfg
    return payload


def build_chunk_store(streaming_cfg: dict[str, Any]) -> ChunkStore | None:
    mode = normalize_streaming_mode(streaming_cfg.get("mode"))
    if mode == "none":
        return None

    if mode == "local_disk":
        producer_cfg = dict(streaming_cfg.get("producer", {}))
        local_cfg = dict(streaming_cfg.get("local_disk", {}))
        root_dir = producer_cfg.get("ready_store_dir") or local_cfg.get("root_dir")
        if not root_dir:
            spool_dir = producer_cfg.get("spool_dir")
            if spool_dir:
                root_dir = str(Path(spool_dir).parent / "ready_store")
            else:
                raise ValueError(
                    "For mode=local_disk set streaming.producer.ready_store_dir"
                )
        max_ready_chunks = int(
            producer_cfg.get("ready_store_max_chunks", local_cfg.get("max_ready_chunks", 200))
        )
        refill_after = int(
            producer_cfg.get("refill_after_consumed_chunks", max(1, max_ready_chunks // 2))
        )
        low_watermark_chunks = max(0, max_ready_chunks - max(1, refill_after))
        return LocalDiskChunkStore(
            root_dir=root_dir,
            max_ready_chunks=max_ready_chunks,
            low_watermark_chunks=low_watermark_chunks,
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
    spool_dir = producer_cfg.get("spool_dir")
    if not spool_dir:
        raise ValueError("streaming.producer.spool_dir must be set")
    return ChunkWriter(
        store=store,
        spool_dir=Path(spool_dir),
        chunk_size_samples=int(streaming_cfg.get("chunk_size_samples", 256)),
        compression=str(chunk_format_cfg.get("compression", "none")),
        spool_max_pending_chunks=int(producer_cfg.get("max_pending_spool_chunks", 64)),
    )


def build_chunk_reader(streaming_cfg: dict[str, Any], store: ChunkStore) -> ChunkReader:
    consumer_cfg = dict(streaming_cfg.get("consumer", {}))
    cache_dir = consumer_cfg.get("cache_dir")
    if not cache_dir:
        raise ValueError("streaming.consumer.cache_dir must be set")

    return ChunkReader(
        store=store,
        cache_dir=cache_dir,
        prefetch_max_chunks=int(consumer_cfg.get("prefetch_max_chunks", 16)),
        delete_remote_after=str(consumer_cfg.get("delete_remote_after", "consume")),
        distributed_cfg=dict(streaming_cfg.get("distributed", {})),
        randomize_within_chunk=bool(consumer_cfg.get("randomize_within_chunk", True)),
        random_seed=int(consumer_cfg["random_seed"]) if consumer_cfg.get("random_seed") is not None else None,
    )
