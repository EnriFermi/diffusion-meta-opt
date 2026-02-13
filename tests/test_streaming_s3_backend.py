from __future__ import annotations

import datetime as dt
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset.shared.streaming.backends.s3 import S3ChunkStore


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def copy_object(self, Bucket: str, CopySource: dict, Key: str) -> None:
        src = (CopySource["Bucket"], CopySource["Key"])
        self.objects[(Bucket, Key)] = self.objects[src]

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.objects.pop((Bucket, Key), None)

    def list_objects_v2(self, Bucket: str, Prefix: str, **kwargs):
        contents = []
        for (bucket, key), value in self.objects.items():
            if bucket != Bucket:
                continue
            if not key.startswith(Prefix):
                continue
            contents.append(
                {
                    "Key": key,
                    "Size": len(value),
                    "LastModified": dt.datetime.now(tz=dt.timezone.utc),
                }
            )
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[(bucket, key)])


class _FakeBoto3(types.SimpleNamespace):
    def __init__(self, client_obj: _FakeS3Client) -> None:
        super().__init__(client=lambda *args, **kwargs: client_obj)


class TestStreamingS3Backend(unittest.TestCase):
    def test_put_list_fetch_delete(self) -> None:
        fake_client = _FakeS3Client()
        fake_boto3 = _FakeBoto3(fake_client)

        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            store = S3ChunkStore(
                bucket="test-bucket",
                prefix="prefix",
                region=None,
                endpoint_url=None,
                max_remote_chunks=10,
                staging_prefix="staging",
                ready_prefix="ready",
            )

            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "chunk.pt"
                source.write_bytes(b"abc123")

                ref = store.put_ready("chunk_1", local_file=source, meta={"created_at": 1.0})
                self.assertEqual(ref.chunk_id, "chunk_1")

                ready = store.list_ready(limit=None)
                self.assertEqual(len(ready), 1)
                self.assertEqual(ready[0].chunk_id, "chunk_1")

                target = Path(tmp) / "downloaded.pt"
                store.fetch_to_local(ready[0], target)
                self.assertEqual(target.read_bytes(), b"abc123")

                store.delete_ready(ready[0])
                self.assertEqual(store.count_ready(), 0)


if __name__ == "__main__":
    unittest.main()
