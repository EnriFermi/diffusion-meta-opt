"""Lineage-safe manifests, checksums, and immutable mmap shard primitives."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from big_vae.weightclip_benchmark.metadata import canonical_json_bytes, redact_secrets


OPERATOR_BANK_BUILDER_SOURCE_PATHS = (
    "training/weightclip_benchmark/build_operator_dataset.py",
)


def _module_name_for_source(path: Path, *, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _local_module_sources(module_name: str, *, root: Path) -> list[Path]:
    """Resolve a module plus package initializers exactly as Python imports them."""

    if not module_name:
        return []
    parts = module_name.split(".")
    module_path = root.joinpath(*parts).with_suffix(".py")
    package_path = root.joinpath(*parts, "__init__.py")
    if not module_path.is_file() and not package_path.is_file():
        return []
    sources: list[Path] = []
    for end in range(1, len(parts)):
        initializer = root.joinpath(*parts[:end], "__init__.py")
        if initializer.is_file():
            sources.append(initializer.resolve())
    if module_path.is_file():
        sources.append(module_path.resolve())
    if package_path.is_file():
        sources.append(package_path.resolve())
    return sources


def _package_initializers_for_source(path: Path, *, root: Path) -> list[Path]:
    """Return every existing initializer executed before a root source module."""

    relative = path.relative_to(root)
    sources: list[Path] = []
    for end in range(1, len(relative.parts)):
        initializer = root.joinpath(*relative.parts[:end], "__init__.py")
        if initializer.is_file():
            sources.append(initializer.resolve())
    return sources


def _resolve_import_from(
    node: ast.ImportFrom,
    *,
    current_source: Path,
    root: Path,
) -> str:
    if node.level == 0:
        return node.module or ""
    current_module = _module_name_for_source(current_source, root=root)
    package = current_module if current_source.name == "__init__.py" else current_module.rpartition(".")[0]
    package_parts = package.split(".") if package else []
    keep = len(package_parts) - node.level + 1
    if keep < 0:
        raise RuntimeError(
            "operator-bank builder source has an invalid relative import: "
            f"path={current_source.relative_to(root)} level={node.level}"
        )
    prefix = ".".join(package_parts[:keep])
    return ".".join(part for part in (prefix, node.module or "") if part)


def _operator_bank_builder_import_closure(*, root: Path) -> list[Path]:
    """Return the deterministic transitive local-Python closure of the builder.

    The pair manifest binds only code that can execute while constructing the
    operator bank.  AE/profile/evaluation modules living in the same package
    are intentionally excluded unless a root reaches them through a local
    import.  Non-constant dynamic imports fail closed because their closure
    cannot be proven statically.
    """

    root = root.resolve()
    local_top_levels = {
        path.name if path.is_dir() else path.stem
        for path in root.iterdir()
        if path.is_dir() or (path.is_file() and path.suffix == ".py")
    }
    roots = [(root / relative).resolve() for relative in OPERATOR_BANK_BUILDER_SOURCE_PATHS]
    pending = list(roots)
    for source_root in roots:
        pending.extend(_package_initializers_for_source(source_root, root=root))
    visited: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        if not source.is_relative_to(root) or not source.is_file():
            raise RuntimeError(f"operator-bank builder source root is missing: {source}")
        visited.add(source)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported_modules: list[tuple[str, int, bool]] = []
        importlib_module_aliases = {"importlib"}
        dynamic_import_aliases = {"__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib":
                        importlib_module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module == "importlib":
                    for alias in node.names:
                        if alias.name == "import_module":
                            dynamic_import_aliases.add(alias.asname or alias.name)
                if node.level == 0 and node.module == "builtins":
                    for alias in node.names:
                        if alias.name == "__import__":
                            dynamic_import_aliases.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend((alias.name, node.lineno, True) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_import_from(node, current_source=source, root=root)
                imported_modules.append((base, node.lineno, True))
                imported_modules.extend(
                    (".".join(part for part in (base, alias.name) if part), node.lineno, False)
                    for alias in node.names
                    if alias.name != "*"
                )
            elif isinstance(node, ast.Call):
                is_dynamic_import = (
                    isinstance(node.func, ast.Name) and node.func.id in dynamic_import_aliases
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in importlib_module_aliases
                )
                if is_dynamic_import:
                    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                        raise RuntimeError(
                            "operator-bank builder source uses a non-constant dynamic import; "
                            f"closure is not provable: path={source.relative_to(root)} line={node.lineno}"
                        )
                    dynamic_module = node.args[0].value
                    if dynamic_module.startswith("."):
                        raise RuntimeError(
                            "operator-bank builder source uses an unsupported relative dynamic import; "
                            f"closure is not provable: path={source.relative_to(root)} line={node.lineno}"
                        )
                    imported_modules.append((dynamic_module, node.lineno, True))
                if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                    raise RuntimeError(
                        "operator-bank builder source uses eval/exec; closure is not provable: "
                        f"path={source.relative_to(root)} line={node.lineno}"
                    )
        for module_name, line, require_resolution in imported_modules:
            resolved = _local_module_sources(module_name, root=root)
            if resolved:
                pending.extend(path for path in resolved if path not in visited)
                continue
            top_level = module_name.partition(".")[0]
            if require_resolution and top_level in local_top_levels:
                raise RuntimeError(
                    "operator-bank builder local import cannot be resolved: "
                    f"path={source.relative_to(root)} line={line} module={module_name}"
                )
    return sorted(visited)


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat_identity(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def sha256_file_stable(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Hash one file only if its physical identity stays fixed across the read."""

    resolved = Path(path).resolve()
    before = file_stat_identity(resolved)
    digest = sha256_file(resolved)
    middle = file_stat_identity(resolved)
    confirm_digest = sha256_file(resolved)
    after = file_stat_identity(resolved)
    if before != middle or middle != after or digest != confirm_digest:
        raise RuntimeError(f"file changed while it was being hashed: {resolved}")
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise RuntimeError(
            f"file hash mismatch: path={resolved} expected={expected_sha256} actual={digest}"
        )
    return {"path": str(resolved), "sha256": digest, "stat": before}


def assert_file_snapshot(snapshot: Mapping[str, Any], *, context: str) -> None:
    path = Path(str(snapshot.get("path", ""))).resolve()
    expected_stat = snapshot.get("stat")
    if not isinstance(expected_stat, Mapping) or file_stat_identity(path) != dict(expected_stat):
        raise RuntimeError(f"{context} changed after its stable hash snapshot: {path}")
    expected_sha256 = str(snapshot.get("sha256", ""))
    if len(expected_sha256) != 64:
        raise RuntimeError(f"{context} snapshot has no valid SHA-256: {path}")
    try:
        sha256_file_stable(path, expected_sha256=expected_sha256)
    except RuntimeError as exc:
        raise RuntimeError(f"{context} changed after its stable hash snapshot: {path}") from exc


def operator_bank_builder_source_seal(*, workspace_root: Path | None = None) -> dict[str, Any]:
    """Hash the exact transitive local import closure that defines operator-bank bytes."""

    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    files: list[dict[str, Any]] = []
    for path in _operator_bank_builder_import_closure(root=root):
        relative = str(path.relative_to(root))
        stable = sha256_file_stable(path)
        files.append({"path": relative, "sha256": stable["sha256"], "bytes": stable["stat"]["bytes"]})
    payload = {
        "schema_version": 2,
        "kind": "weightclip_operator_bank_builder_source",
        "closure": {
            "algorithm": "transitive_local_python_imports_v1",
            "roots": list(OPERATOR_BANK_BUILDER_SOURCE_PATHS),
            "nonconstant_dynamic_imports": "forbidden",
        },
        "files": files,
    }
    payload["source_implementation_seal_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def validate_operator_bank_builder_source_seal(
    seal: Mapping[str, Any], *, workspace_root: Path | None = None
) -> str:
    actual = operator_bank_builder_source_seal(workspace_root=workspace_root)
    if dict(seal) != actual:
        raise RuntimeError("operator-bank builder source differs from the pair-manifest source seal")
    return str(actual["source_implementation_seal_sha256"])


def stable_lineage_split(dataset: str, seeds: Iterable[int], counts: tuple[int, int, int], namespace: str) -> dict[int, str]:
    seeds = list(seeds)
    if sum(counts) != len(seeds):
        raise ValueError("split counts must equal number of lineage seeds")

    def rank(seed: int) -> bytes:
        return hashlib.sha256(f"{namespace}\0{dataset}\0{seed}".encode()).digest()

    ordered = sorted(seeds, key=rank)
    train_n, validation_n, _ = counts
    return {
        seed: "train" if index < train_n else "validation" if index < train_n + validation_n else "internal_test"
        for index, seed in enumerate(ordered)
    }


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    dataset: str
    lineage_id: str
    seed: int
    split: str
    epoch_one_based: int
    checkpoint_index_zero_based: int
    is_primary: bool
    checkpoint_path: str
    checkpoint_sha256: str
    bytes: int
    train_loss: float | None = None
    train_accuracy: float | None = None
    validation_loss: float | None = None
    validation_accuracy: float | None = None
    test_loss: float | None = None
    test_accuracy: float | None = None
    chosen_lr: float | None = None
    status: str = "complete"


@dataclass(frozen=True, slots=True)
class LineageRecord:
    dataset: str
    lineage_id: str
    seed: int
    split: str
    chosen_lr: float
    dataset_sha256: str
    config_sha256: str
    status: str
    checkpoints: int
    rolling_resume_path: str | None
    rolling_resume_sha256: str | None


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    safe = redact_secrets(payload)
    _atomic_bytes(Path(path), json.dumps(safe, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def write_json_immutable(path: str | Path, payload: Any) -> str:
    """Write once; an identical retry is a cache hit and differing content fails."""

    path = Path(path)
    safe = redact_secrets(payload)
    body = json.dumps(safe, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != body:
            raise FileExistsError(f"immutable manifest already exists with different bytes: {path}")
        return sha256_file(path)
    _atomic_bytes(path, body)
    path.chmod(0o444)
    return sha256_file(path)


def _records_as_dicts(records: Iterable[Any]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        value = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record)
        output.append(redact_secrets(value))
    return output


def write_records_immutable(base_path: str | Path, records: Iterable[Any]) -> dict[str, str]:
    """Write canonical JSONL plus Parquet when pyarrow is available."""

    base = Path(base_path)
    rows = _records_as_dicts(records)
    jsonl = base.with_suffix(".jsonl")
    body = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if jsonl.exists() and jsonl.read_bytes() != body:
        raise FileExistsError(f"immutable records differ: {jsonl}")
    if not jsonl.exists():
        _atomic_bytes(jsonl, body)
        jsonl.chmod(0o444)
    result = {str(jsonl): sha256_file(jsonl)}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return result
    parquet = base.with_suffix(".parquet")
    if not parquet.exists():
        parquet.parent.mkdir(parents=True, exist_ok=True)
        temp = parquet.with_name(f".{parquet.name}.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temp, compression="zstd")
        os.replace(temp, parquet)
        parquet.chmod(0o444)
    result[str(parquet)] = sha256_file(parquet)
    return result


def validate_lineage_records(
    lineages: Iterable[LineageRecord],
    checkpoints: Iterable[CheckpointRecord],
    *,
    datasets: int,
    lineages_per_dataset: int,
    expected_primary_per_lineage: int = 2,
    expected_primary_indices: tuple[int, ...] = (43, 44),
) -> dict[str, int]:
    lineage_rows = list(lineages)
    checkpoint_rows = list(checkpoints)
    expected_lineages = datasets * lineages_per_dataset
    if len(lineage_rows) != expected_lineages:
        raise ValueError(f"expected {expected_lineages} lineages, got {len(lineage_rows)}")
    seen_ids: dict[str, str] = {}
    for row in lineage_rows:
        previous = seen_ids.setdefault(row.lineage_id, row.split)
        if previous != row.split:
            raise ValueError(f"lineage crosses splits: {row.lineage_id}")
    primary = [row for row in checkpoint_rows if row.is_primary]
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for row in primary:
        if row.checkpoint_index_zero_based not in expected_primary_indices:
            raise ValueError(f"unexpected primary checkpoint index: {row.checkpoint_index_zero_based}")
        counts[row.lineage_id] = counts.get(row.lineage_id, 0) + 1
        previous_split = seen_ids.get(row.lineage_id)
        if previous_split != row.split:
            raise ValueError(f"checkpoint split mismatch: {row.lineage_id}")
        old_lineage = hashes.setdefault(row.checkpoint_sha256, row.lineage_id)
        if old_lineage != row.lineage_id:
            raise ValueError(f"checkpoint hash duplicated across lineages: {row.checkpoint_sha256}")
    for row in checkpoint_rows:
        expected_zero = -1 if row.epoch_one_based == 0 else row.epoch_one_based - 1
        if row.checkpoint_index_zero_based != expected_zero:
            raise ValueError(
                f"epoch/index mismatch for {row.lineage_id}: "
                f"epoch_one_based={row.epoch_one_based}, index_zero_based={row.checkpoint_index_zero_based}"
            )
    wrong = {key: value for key, value in counts.items() if value != expected_primary_per_lineage}
    if wrong or len(counts) != expected_lineages:
        raise ValueError(f"wrong primary checkpoint counts: {wrong}; covered={len(counts)}")
    return {"lineages": len(lineage_rows), "primary_checkpoints": len(primary), "all_checkpoints": len(checkpoint_rows)}


def validate_operator_checkpoint_selection(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_datasets: Iterable[str],
    expected_lineages_per_dataset: int = 35,
    expected_indices: tuple[int, ...] = (43, 44),
    expected_split: str = "train",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the exact production checkpoint population before bank writes.

    Returns canonicalized row copies and their already hash-verified file
    inventory so callers do not need to read every checkpoint twice.
    """

    selected = [dict(row) for row in rows]
    datasets = tuple(sorted(str(value) for value in expected_datasets))
    if len(set(datasets)) != len(datasets):
        raise ValueError("expected dataset inventory contains duplicates")
    expected_total = len(datasets) * int(expected_lineages_per_dataset) * len(expected_indices)
    if len(selected) != expected_total:
        raise ValueError(f"expected exact {expected_total}-checkpoint operator population, got {len(selected)}")
    actual_datasets = {str(row.get("dataset")) for row in selected}
    if actual_datasets != set(datasets):
        raise ValueError(
            "operator checkpoint dataset inventory mismatch: "
            f"expected={datasets} actual={tuple(sorted(actual_datasets))}"
        )

    logical_keys: set[tuple[str, str, int]] = set()
    paths: set[str] = set()
    by_lineage: dict[tuple[str, str], set[int]] = {}
    file_rows: list[tuple[dict[str, Any], tuple[str, str, int], Path, str]] = []
    for row in selected:
        dataset = str(row.get("dataset"))
        lineage = str(row.get("lineage_id"))
        split = str(row.get("split"))
        if split != expected_split or row.get("is_primary") is not True:
            raise ValueError(
                f"operator checkpoint must be primary/{expected_split}: "
                f"dataset={dataset} lineage={lineage} split={split!r} is_primary={row.get('is_primary')!r}"
            )
        try:
            index = int(row["checkpoint_index_zero_based"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint index: dataset={dataset} lineage={lineage}") from exc
        if index not in expected_indices:
            raise ValueError(
                f"unexpected operator checkpoint index: dataset={dataset} lineage={lineage} index={index}"
            )
        logical = (dataset, lineage, index)
        if logical in logical_keys:
            raise ValueError(f"duplicate operator checkpoint logical identity: {logical}")
        logical_keys.add(logical)
        by_lineage.setdefault((dataset, lineage), set()).add(index)

        raw_path = row.get("checkpoint_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"operator checkpoint path is missing: {logical}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"operator checkpoint is missing: {path}")
        canonical = str(path)
        if canonical in paths:
            raise ValueError(f"duplicate operator checkpoint canonical path: {canonical}")
        paths.add(canonical)
        expected_sha = row.get("checkpoint_sha256")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise ValueError(f"invalid checkpoint SHA-256 in manifest: {logical}")
        row["checkpoint_path"] = canonical
        file_rows.append((row, logical, path, expected_sha))

    for dataset in datasets:
        lineages = {
            lineage: indices
            for (row_dataset, lineage), indices in by_lineage.items()
            if row_dataset == dataset
        }
        if len(lineages) != expected_lineages_per_dataset:
            raise ValueError(
                f"dataset {dataset} must have exactly {expected_lineages_per_dataset} train lineages, "
                f"got {len(lineages)}"
            )
        wrong = {lineage: sorted(indices) for lineage, indices in lineages.items() if indices != set(expected_indices)}
        if wrong:
            raise ValueError(f"lineages do not have exact checkpoint indices {expected_indices}: {wrong}")
    if len(paths) != expected_total or len(logical_keys) != expected_total:
        raise ValueError("operator checkpoint population is not one-to-one with canonical files")
    # Hash only after every cheap population/identity/split/index/path gate has
    # passed. A malformed 700-row manifest must not trigger an expensive full
    # checkpoint read before it is rejected.
    inventory: list[dict[str, Any]] = []
    for row, (dataset, lineage, index), path, expected_sha in file_rows:
        try:
            stable = sha256_file_stable(path, expected_sha256=expected_sha)
        except RuntimeError as exc:
            if "file hash mismatch" not in str(exc):
                raise
            raise RuntimeError(
                f"checkpoint hash mismatch: path={path} expected={expected_sha}"
            ) from exc
        actual_sha = str(stable["sha256"])
        row["checkpoint_sha256"] = actual_sha
        inventory.append(
            {
                "dataset": dataset,
                "lineage_id": lineage,
                "checkpoint_index_zero_based": index,
                "path": str(path),
                "sha256": actual_sha,
                "stat": stable["stat"],
            }
        )
    return selected, inventory


class ImmutableArrayShardWriter:
    """Write fixed-shape record arrays as checksummed mmap-compatible NPY shards."""

    def __init__(self, root: str | Path, bank_name: str, records_per_shard: int) -> None:
        self.root = Path(root)
        self.bank_name = str(bank_name)
        self.records_per_shard = int(records_per_shard)
        self.staging = self.root / f".{self.bank_name}.building"
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True)
        self._buffer: list[dict[str, np.ndarray]] = []
        self._metadata: list[dict[str, Any]] = []
        self._shards: list[dict[str, Any]] = []
        self._schema: dict[str, tuple[tuple[int, ...], str]] | None = None

    def add(self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> tuple[int, int]:
        normalized = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
        schema = {key: (value.shape, str(value.dtype)) for key, value in normalized.items()}
        if self._schema is None:
            self._schema = schema
        elif schema != self._schema:
            raise ValueError(f"record schema mismatch: {schema} != {self._schema}")
        shard_index = len(self._shards)
        offset = len(self._buffer)
        self._buffer.append(normalized)
        self._metadata.append(redact_secrets(dict(metadata)))
        if len(self._buffer) >= self.records_per_shard:
            self.flush()
        return shard_index, offset

    def flush(self) -> None:
        if not self._buffer:
            return
        index = len(self._shards)
        shard_dir = self.staging / f"shard-{index:06d}"
        shard_dir.mkdir()
        files: dict[str, dict[str, Any]] = {}
        for key in sorted(self._buffer[0]):
            path = shard_dir / f"{key}.npy"
            np.save(path, np.stack([record[key] for record in self._buffer], axis=0), allow_pickle=False)
            files[key] = {"path": str(path.relative_to(self.staging)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        meta_path = shard_dir / "records.jsonl"
        meta_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in self._metadata))
        files["records"] = {"path": str(meta_path.relative_to(self.staging)), "sha256": sha256_file(meta_path), "bytes": meta_path.stat().st_size}
        shard_payload = {"index": index, "records": len(self._buffer), "files": files}
        shard_payload["content_sha256"] = hashlib.sha256(canonical_json_bytes(shard_payload)).hexdigest()
        self._shards.append(shard_payload)
        self._buffer.clear()
        self._metadata.clear()

    def finalize(self, contract: Mapping[str, Any]) -> Path:
        self.flush()
        manifest = {
            "format_version": 1,
            "bank": self.bank_name,
            "schema": self._schema,
            "records_per_shard": self.records_per_shard,
            "records": sum(int(shard["records"]) for shard in self._shards),
            "contract": redact_secrets(dict(contract)),
            "shards": self._shards,
        }
        manifest["content_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        write_json_immutable(self.staging / "manifest.json", manifest)
        final = self.root / f"{self.bank_name}-{manifest['content_sha256'][:16]}"
        if final.exists():
            existing = json.loads((final / "manifest.json").read_text())
            if existing["content_sha256"] != manifest["content_sha256"]:
                raise FileExistsError(f"content-address collision: {final}")
            shutil.rmtree(self.staging)
            return final
        os.replace(self.staging, final)
        return final
