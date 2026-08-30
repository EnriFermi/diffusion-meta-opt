from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dataset.shared.streaming.backends.base import ChunkRef, ChunkStore


class S3ChunkStore(ChunkStore):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        region: str | None,
        endpoint_url: str | None,
        max_remote_chunks: int,
        staging_prefix: str,
        ready_prefix: str,
    ) -> None:
        try:
            import boto3
        except Exception as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("boto3 is required for streaming.mode=s3_bridge") from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.staging_prefix = staging_prefix.strip("/")
        self.ready_prefix = ready_prefix.strip("/")
        self.max_remote_chunks = max(1, int(max_remote_chunks))

        self._client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)

    def put_ready(self, chunk_id: str, local_file: str | Path, meta: dict[str, Any]) -> ChunkRef:
        source = Path(local_file)
        suffix = "".join(source.suffixes)
        filename = f"{chunk_id}{suffix}"

        staging_key = _join_key(self.prefix, self.staging_prefix, filename)
        ready_key = _join_key(self.prefix, self.ready_prefix, filename)

        self._client.upload_file(str(source), self.bucket, staging_key)
        self._client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": staging_key},
            Key=ready_key,
        )
        self._client.delete_object(Bucket=self.bucket, Key=staging_key)

        created_at = float(meta.get("created_at", time.time()))
        size_bytes = source.stat().st_size
        return ChunkRef(
            chunk_id=chunk_id,
            uri=f"s3://{self.bucket}/{ready_key}",
            size_bytes=size_bytes,
            created_at=created_at,
            backend_key=ready_key,
        )

    def list_ready(self, limit: int | None = None) -> list[ChunkRef]:
        prefix = _join_key(self.prefix, self.ready_prefix)
        results: list[ChunkRef] = []

        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": prefix,
        }

        while True:
            response = self._client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = str(item.get("Key"))
                if key.endswith("/"):
                    continue
                name = key.rsplit("/", 1)[-1]
                results.append(
                    ChunkRef(
                        chunk_id=_chunk_id_from_filename(name),
                        uri=f"s3://{self.bucket}/{key}",
                        size_bytes=int(item.get("Size", 0)),
                        created_at=float(item.get("LastModified").timestamp()) if item.get("LastModified") else 0.0,
                        backend_key=key,
                    )
                )
                if limit is not None and len(results) >= int(limit):
                    results.sort(key=lambda ref: (ref.created_at, ref.chunk_id))
                    return results[: int(limit)]

            if not response.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = response.get("NextContinuationToken")

        results.sort(key=lambda ref: (ref.created_at, ref.chunk_id))
        if limit is None:
            return results
        return results[: int(limit)]

    def fetch_to_local(self, chunk_ref: ChunkRef, target_path: str | Path) -> Path:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = target.with_suffix(target.suffix + ".tmp")
        self._client.download_file(self.bucket, chunk_ref.backend_key, str(tmp_path))
        tmp_path.replace(target)
        return target

    def delete_ready(self, chunk_ref: ChunkRef) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=chunk_ref.backend_key)

    def count_ready(self) -> int:
        return len(self.list_ready(limit=None))

    def capacity_state(self) -> dict[str, Any]:
        ready = self.count_ready()
        return {
            "backend": "s3_bridge",
            "ready_chunks": ready,
            "max_remote_chunks": self.max_remote_chunks,
            "can_accept": ready < self.max_remote_chunks,
            "needs_fill": ready < self.max_remote_chunks,
        }


def _join_key(*parts: str) -> str:
    normalized = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(normalized)


def _chunk_id_from_filename(name: str) -> str:
    if name.endswith(".pt.gz"):
        return name[: -len(".pt.gz")]
    if name.endswith(".pt"):
        return name[: -len(".pt")]
    return Path(name).stem
