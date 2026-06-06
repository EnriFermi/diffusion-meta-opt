from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ARTIFACT_CONTRACT_VERSION = "big_vae_artifacts_v1"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json_file(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml_file(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(OmegaConf.to_yaml(_jsonable(payload), resolve=False), encoding="utf-8")


def append_csv_row(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists()
    payload = _jsonable(dict(row))
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(payload)


def write_artifact_layout(
    run_dir: str | Path,
    *,
    kind: str,
    run_id: str,
    files: Mapping[str, str | Path] | None = None,
    dirs: Mapping[str, str | Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    target_dir = Path(run_dir)
    payload = {
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "kind": str(kind),
        "run_id": str(run_id),
        "run_dir": str(target_dir),
        "files": {str(key): str(value) for key, value in dict(files or {}).items()},
        "dirs": {str(key): str(value) for key, value in dict(dirs or {}).items()},
        "metadata": _jsonable(dict(metadata or {})),
    }
    path = target_dir / "artifact_layout.json"
    write_json_file(path, payload)
    return path
