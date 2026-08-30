from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Mapping


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip().lower()).strip("-")
    return text or "run"


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def reserve_run_dir(root: Path, run_label: str, *, timestamp: str | None = None) -> tuple[str, Path]:
    runs_root = root / "runs"
    stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    base_run_id = f"{stamp}__{slugify(run_label)}"
    for suffix in range(100):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not reserve unique Celo benchmark run dir under {runs_root}: {base_run_id}")


def write_json_file(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def write_yaml_file(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except Exception:
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True)


def write_artifact_layout(
    target_dir: str | Path,
    *,
    kind: str,
    run_id: str,
    files: Mapping[str, str | Path],
    dirs: Mapping[str, str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(target_dir)
    layout_path = target / "artifact_layout.json"
    payload = {
        "schema_version": 1,
        "kind": kind,
        "run_id": run_id,
        "root": str(target),
        "files": {key: str(value) for key, value in files.items()},
        "dirs": {key: str(value) for key, value in dirs.items()},
        "metadata": dict(metadata or {}),
    }
    write_json_file(layout_path, payload)
    return layout_path

