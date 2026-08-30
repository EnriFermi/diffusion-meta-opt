from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def slugify(value: str, max_len: int = 64) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    clean = clean.strip("._")
    if not clean:
        clean = "sample"
    return clean[:max_len]


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=target.parent, encoding="utf-8") as temp_file:
        json.dump(payload, temp_file, indent=2, ensure_ascii=False)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        tmp_name = temp_file.name

    os.replace(tmp_name, target)
