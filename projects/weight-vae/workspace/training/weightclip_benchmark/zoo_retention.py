"""Fail-closed, manifest-driven retention planning for the WeightCLIP zoo.

This CLI is deliberately plan-first. ``plan`` and ``verify`` only read the zoo
and write content-addressed audit artifacts outside it. ``apply`` is disabled
in this implementation even when an approval file is supplied; a later,
separately reviewed implementation must add the quarantine/recovery workflow.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Iterable, Mapping

import yaml

from big_vae.weightclip_benchmark.manifests import (
    sha256_file,
    stable_lineage_split,
    write_json_immutable,
)
from big_vae.weightclip_benchmark.metadata import SOURCE_TASKS, canonical_json_bytes, frozen_contract


SCHEMA_VERSION = 1
PRODUCTION_DATASETS = tuple(sorted(task.key for task in SOURCE_TASKS))
PRODUCTION_KEEP_EPOCHS = (1, 4, 8, 16, 24, 32, 44, 45)
PRODUCTION_PRIMARY_INDICES = (43, 44)
RETENTION_SOURCE_PATHS = (
    "big_vae/weightclip_benchmark/manifests.py",
    "big_vae/weightclip_benchmark/metadata.py",
    "training/weightclip_benchmark/build_zoo.py",
    "training/weightclip_benchmark/zoo_retention.py",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LINEAGE_RE = re.compile(r"^checkpoints/([^/]+)/lineage-(\d{4})/(.+)$")
EPOCH_RE = re.compile(r"^epochs/epoch-(\d{3})\.pt$")


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _safe_root(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve()}
    if resolved in forbidden or len(resolved.parts) < 5:
        raise ValueError(f"unsafe {label}: {resolved}")
    return resolved


def _assert_sha(value: str, *, label: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise RuntimeError(f"retention inventory requires a regular non-symlink file: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _hash_with_stable_stat(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, dict[str, int]]:
    before = _stat_identity(path)
    if expected_bytes is not None and before["bytes"] != expected_bytes:
        raise RuntimeError(f"file size drift before hashing: {path}")
    digest = sha256_file(path)
    after = _stat_identity(path)
    if after != before:
        raise RuntimeError(f"file stat identity changed during hashing (concurrent writer): {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"file content hash drift: {path}")
    return digest, before


def retention_source_inventory(*, workspace_root: Path | None = None) -> dict[str, Any]:
    root = (
        workspace_root.expanduser().resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    files: list[dict[str, Any]] = []
    for relative in RETENTION_SOURCE_PATHS:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"retention source dependency is missing/unsafe: {path}")
        digest, snapshot = _hash_with_stable_stat(path)
        files.append({"path": relative, "bytes": snapshot["bytes"], "sha256": digest, "stat": snapshot})
    contract = {
        "schema_version": 1,
        "kind": "weightclip_zoo_retention_source_inventory",
        "workspace_root": str(root),
        "files": files,
    }
    return {**contract, "source_inventory_sha256": _fingerprint(contract)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"expected JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def _referenced_jsonl(manifest: Mapping[str, Any], field: str) -> tuple[Path, str]:
    refs = manifest.get(field)
    if not isinstance(refs, dict):
        raise ValueError(f"final manifest lacks {field}")
    candidates = [(Path(str(path)).resolve(), str(digest)) for path, digest in refs.items() if str(path).endswith(".jsonl")]
    if len(candidates) != 1:
        raise ValueError(f"{field} must reference exactly one JSONL file")
    path, digest = candidates[0]
    _assert_sha(digest, label=f"{field} JSONL SHA")
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"{field} JSONL is missing or hash-drifted: {path}")
    return path, digest


def live_zoo_builder_processes(*, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Return live build_zoo processes without trusting a mutable PID file."""

    found: list[dict[str, Any]] = []
    if not proc_root.is_dir():
        raise RuntimeError(f"process inventory is unavailable: {proc_root}")
    for entry in sorted(proc_root.iterdir(), key=lambda path: path.name):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError(f"cannot inspect live process inventory: {cmdline_path}") from exc
        args = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        joined = " ".join(args)
        if "training.weightclip_benchmark.build_zoo" in joined or re.search(r"(?:^|/)build_zoo\.py(?:\s|$)", joined):
            found.append({"pid": int(entry.name), "argv": args})
    return found


def _validate_production_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("retention config schema_version mismatch")
    policy = config.get("policy")
    safety = config.get("safety")
    paths = config.get("paths")
    if not isinstance(policy, dict) or not isinstance(safety, dict) or not isinstance(paths, dict):
        raise ValueError("retention config requires paths/policy/safety mappings")
    expected = {
        "name": "raw_fp32_grid9_v1",
        "keep_initialization": True,
        "keep_epochs_one_based": list(PRODUCTION_KEEP_EPOCHS),
        "expected_datasets": 10,
        "expected_lineages_per_dataset": 50,
        "expected_archived_states_per_lineage": 46,
        "expected_all_checkpoint_states": 23_000,
        "expected_primary_checkpoint_states": 1_000,
        "primary_checkpoint_indices_zero_based": list(PRODUCTION_PRIMARY_INDICES),
        "require_forward_audit": True,
        "preserve_all_non_candidate_files": True,
        "preserve_rolling_resumes": True,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"production retention policy requires {key}={value!r}")
    if safety.get("fail_if_zoo_builder_live") is not True:
        raise ValueError("retention must fail while a zoo builder is live")
    if safety.get("require_exact_final_manifest_sha256") is not True:
        raise ValueError("retention requires an explicit exact final manifest SHA")
    if safety.get("apply_implementation_enabled") is not False:
        raise ValueError("this reviewed retention build requires apply_implementation_enabled=false")
    if safety.get("require_recovery_drill_before_reclaim") is not True:
        raise ValueError("retention must require a recovery drill before reclaim")
    return {"paths": paths, "policy": policy, "safety": safety}


def _validate_final_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    zoo_root: Path,
    expected_datasets: tuple[str, ...] = PRODUCTION_DATASETS,
    expected_lineages_per_dataset: int = 50,
    expected_states_per_lineage: int = 46,
    expected_primary_indices: tuple[int, ...] = PRODUCTION_PRIMARY_INDICES,
    expected_lineage_split: tuple[int, int, int] = (35, 7, 8),
    seed_namespace: str = "weightclip-resnet18slim-zoo-v1",
    require_forward_audit: bool = True,
) -> dict[str, Any]:
    expected_sha256 = _assert_sha(expected_sha256, label="expected final zoo manifest SHA")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"final zoo manifest is missing: {manifest_path}")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_sha256:
        raise RuntimeError(
            f"final zoo manifest SHA mismatch: expected={expected_sha256} actual={actual_manifest_sha}"
        )
    manifest = _load_json(manifest_path)
    if manifest.get("format_version") != 1:
        raise ValueError("final zoo manifest format_version mismatch")
    if not isinstance(manifest.get("contract"), dict) or _fingerprint(manifest["contract"]) != _fingerprint(
        frozen_contract()
    ):
        raise ValueError("final zoo manifest frozen protocol contract mismatch")
    expected_lineages = len(expected_datasets) * expected_lineages_per_dataset
    expected_all = expected_lineages * expected_states_per_lineage
    expected_primary = expected_lineages * len(expected_primary_indices)
    counts = manifest.get("counts")
    if counts != {
        "lineages": expected_lineages,
        "primary_checkpoints": expected_primary,
        "all_checkpoints": expected_all,
    }:
        raise ValueError(f"final zoo manifest is incomplete: counts={counts!r}")
    if require_forward_audit and manifest.get("audit_forward_every_checkpoint") is not True:
        raise ValueError("final zoo manifest lacks the required every-checkpoint forward audit")
    checkpoint_path, checkpoint_manifest_sha = _referenced_jsonl(manifest, "checkpoint_files")
    lineage_path, lineage_manifest_sha = _referenced_jsonl(manifest, "lineage_files")
    manifest_root = manifest_path.parent.resolve()
    if checkpoint_path.parent != manifest_root or lineage_path.parent != manifest_root:
        raise ValueError("referenced final inventories must reside beside the final manifest")
    checkpoint_rows = _read_jsonl(checkpoint_path)
    lineage_rows = _read_jsonl(lineage_path)
    if len(checkpoint_rows) != expected_all or len(lineage_rows) != expected_lineages:
        raise ValueError("referenced lineage/checkpoint inventories are incomplete")

    expected_epoch_one = set(range(expected_states_per_lineage))
    expected_zero = {-1, *range(expected_states_per_lineage - 1)}
    dataset_counts: Counter[str] = Counter()
    dataset_shas: dict[str, str] = {}
    lineage_ids: set[tuple[str, str]] = set()
    expected_config_sha = hashlib.sha256(
        json.dumps(frozen_contract()["zoo"], sort_keys=True).encode()
    ).hexdigest()
    expected_splits = {
        dataset: stable_lineage_split(
            dataset,
            range(1, expected_lineages_per_dataset + 1),
            expected_lineage_split,
            seed_namespace,
        )
        for dataset in expected_datasets
    }
    for row in lineage_rows:
        dataset = str(row.get("dataset"))
        lineage = str(row.get("lineage_id"))
        if dataset not in expected_datasets or row.get("status") != "complete":
            raise ValueError(f"invalid final lineage row: {dataset}/{lineage}")
        seed = int(row.get("seed"))
        if lineage != f"{dataset}:seed={seed}" or seed not in expected_splits[dataset]:
            raise ValueError(f"lineage identity/seed mismatch: {dataset}/{lineage}/{seed}")
        if row.get("split") != expected_splits[dataset][seed]:
            raise ValueError(f"lineage split mismatch: {lineage}")
        if int(row.get("checkpoints")) != expected_states_per_lineage:
            raise ValueError(f"lineage checkpoint count mismatch: {lineage}")
        dataset_sha = _assert_sha(str(row.get("dataset_sha256")), label=f"dataset SHA {lineage}")
        previous_dataset_sha = dataset_shas.setdefault(dataset, dataset_sha)
        if previous_dataset_sha != dataset_sha:
            raise ValueError(f"inconsistent dataset SHA across lineages: {dataset}")
        config_sha = _assert_sha(str(row.get("config_sha256")), label=f"config SHA {lineage}")
        if config_sha != expected_config_sha:
            raise ValueError(f"lineage frozen protocol config SHA mismatch: {lineage}")
        resume_path_raw = row.get("rolling_resume_path")
        resume_sha_raw = row.get("rolling_resume_sha256")
        if (resume_path_raw is None) != (resume_sha_raw is None):
            raise ValueError(f"rolling resume path/SHA presence mismatch: {lineage}")
        if resume_path_raw is not None:
            resume_path = Path(str(resume_path_raw)).resolve()
            expected_resume = zoo_root / "checkpoints" / dataset / f"lineage-{seed:04d}" / "resume.pt"
            if resume_path != expected_resume or not resume_path.is_file():
                raise ValueError(f"rolling resume path/layout mismatch: {lineage}")
            resume_sha = _assert_sha(str(resume_sha_raw), label=f"rolling resume SHA {lineage}")
            if sha256_file(resume_path) != resume_sha:
                raise RuntimeError(f"rolling resume hash drift: {resume_path}")
        logical = (dataset, lineage)
        if logical in lineage_ids:
            raise ValueError(f"duplicate lineage identity: {logical}")
        lineage_ids.add(logical)
        dataset_counts[dataset] += 1
    if dataset_counts != Counter({dataset: expected_lineages_per_dataset for dataset in expected_datasets}):
        raise ValueError(f"final lineage dataset coverage mismatch: {dict(dataset_counts)}")
    for dataset, declared_sha in sorted(dataset_shas.items()):
        dataset_path = zoo_root / "dataset_pts" / dataset / "dataset.pt"
        if not dataset_path.is_file() or sha256_file(dataset_path) != declared_sha:
            raise RuntimeError(f"source dataset artifact hash drift: {dataset_path}")

    by_lineage: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    state_paths: set[str] = set()
    for row in checkpoint_rows:
        dataset = str(row.get("dataset"))
        lineage = str(row.get("lineage_id"))
        logical = (dataset, lineage)
        if logical not in lineage_ids:
            raise ValueError(f"checkpoint references unknown lineage: {logical}")
        if row.get("status") != "complete" or row.get("split") != expected_splits[dataset][int(row.get("seed"))]:
            raise ValueError(f"checkpoint status/split mismatch: {logical}")
        epoch_one = int(row.get("epoch_one_based"))
        epoch_zero = int(row.get("checkpoint_index_zero_based"))
        expected_index = -1 if epoch_one == 0 else epoch_one - 1
        if epoch_one not in expected_epoch_one or epoch_zero != expected_index:
            raise ValueError(f"invalid epoch mapping: {logical} epoch={epoch_one} index={epoch_zero}")
        if bool(row.get("is_primary")) != (epoch_zero in expected_primary_indices):
            raise ValueError(f"primary flag mismatch: {logical} index={epoch_zero}")
        raw_path = Path(str(row.get("checkpoint_path"))).resolve()
        if not raw_path.is_relative_to(zoo_root):
            raise ValueError(f"checkpoint escapes zoo root: {raw_path}")
        relative = raw_path.relative_to(zoo_root).as_posix()
        expected_relative = (
            f"checkpoints/{dataset}/lineage-{int(row.get('seed')):04d}/initialization.pt"
            if epoch_one == 0
            else f"checkpoints/{dataset}/lineage-{int(row.get('seed')):04d}/epochs/epoch-{epoch_one:03d}.pt"
        )
        if relative != expected_relative:
            raise ValueError(f"checkpoint path/layout mismatch: {relative} != {expected_relative}")
        if relative in state_paths:
            raise ValueError(f"duplicate checkpoint path: {relative}")
        state_paths.add(relative)
        declared_sha = _assert_sha(str(row.get("checkpoint_sha256")), label=f"checkpoint SHA {relative}")
        declared_bytes = int(row.get("bytes"))
        by_lineage[logical].append(
            {
                "relative_path": relative,
                "path": raw_path,
                "sha256": declared_sha,
                "bytes": declared_bytes,
                "epoch_one_based": epoch_one,
                "checkpoint_index_zero_based": epoch_zero,
                "is_primary": bool(row.get("is_primary")),
                "dataset": dataset,
                "lineage_id": lineage,
                "split": str(row.get("split")),
            }
        )
    if set(by_lineage) != lineage_ids:
        raise ValueError("checkpoint inventory does not cover every lineage")
    for logical, rows in by_lineage.items():
        if {row["epoch_one_based"] for row in rows} != expected_epoch_one:
            raise ValueError(f"lineage lacks exact 46-state archive: {logical}")
        if {row["checkpoint_index_zero_based"] for row in rows} != expected_zero:
            raise ValueError(f"lineage checkpoint index set mismatch: {logical}")

    # Structural gates above are intentionally complete before this expensive pass.
    verified_rows: list[dict[str, Any]] = []
    total_checkpoint_rows = sum(len(rows) for rows in by_lineage.values())
    checkpoint_verify_started = time.monotonic()
    print(
        f"[zoo-retention] stage=verify-final-checkpoints total={total_checkpoint_rows} "
        f"manifest={manifest_path}",
        flush=True,
    )
    for logical in sorted(by_lineage):
        for row in sorted(by_lineage[logical], key=lambda value: value["epoch_one_based"]):
            path = row.pop("path")
            actual_sha, snapshot = _hash_with_stable_stat(
                path,
                expected_bytes=int(row["bytes"]),
                expected_sha256=str(row["sha256"]),
            )
            row["sha256"] = actual_sha
            row["stat"] = snapshot
            verified_rows.append(row)
            if len(verified_rows) % 250 == 0 or len(verified_rows) == total_checkpoint_rows:
                elapsed = max(time.monotonic() - checkpoint_verify_started, 1e-9)
                print(
                    f"[zoo-retention] stage=verify-final-checkpoints "
                    f"progress={len(verified_rows)}/{total_checkpoint_rows} "
                    f"elapsed_s={elapsed:.1f} rate_files_s={len(verified_rows) / elapsed:.2f}",
                    flush=True,
                )
    return {
        "manifest": manifest,
        "manifest_sha256": actual_manifest_sha,
        "checkpoint_manifest_path": str(checkpoint_path),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "lineage_manifest_path": str(lineage_path),
        "lineage_manifest_sha256": lineage_manifest_sha,
        "checkpoint_rows": verified_rows,
    }


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []

    def fail_walk(error: OSError) -> None:
        raise RuntimeError(f"cannot inventory zoo path: {error.filename}") from error

    for directory, names, filenames in os.walk(root, followlinks=False, onerror=fail_walk):
        names.sort()
        filenames.sort()
        base = Path(directory)
        for name in names:
            if (base / name).is_symlink():
                raise ValueError(f"symlink is forbidden in zoo retention inventory: {base / name}")
        for name in filenames:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"symlink is forbidden in zoo retention inventory: {path}")
            files.append(path)
    return files


def _classify_file(relative_path: str, *, keep_epochs: set[int]) -> tuple[str, str]:
    match = LINEAGE_RE.fullmatch(relative_path)
    if match:
        tail = match.group(3)
        if tail == "initialization.pt":
            return "keep", "initialization"
        epoch_match = EPOCH_RE.fullmatch(tail)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            if epoch in keep_epochs:
                return "keep", "raw_fp32_grid9"
            return "quarantine_candidate", "outside_raw_fp32_grid9"
        if tail == "resume.pt":
            return "keep", "rolling_resume_preserved_pending_separate_decision"
        if tail in {"resolved_config.json", "metrics.jsonl", "complete.json"}:
            return "keep", "lineage_provenance"
    if relative_path.startswith("manifests/"):
        return "keep", "final_manifest_and_hashes"
    if relative_path.startswith("dataset_pts/") or relative_path.startswith("ood_dataset_pts/"):
        return "keep", "dataset_artifact"
    if relative_path.endswith("/lr_sweep.json") and relative_path.startswith("checkpoints/"):
        return "keep", "lr_sweep_evidence"
    return "keep", "preserve_unclassified_non_candidate"


def _inventory_zoo(
    zoo_root: Path,
    *,
    checkpoint_rows: Iterable[Mapping[str, Any]],
    keep_epochs: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_by_path = {str(row["relative_path"]): dict(row) for row in checkpoint_rows}
    files = _walk_files(zoo_root)
    print(
        f"[zoo-retention] stage=inventory-zoo files={len(files)} root={zoo_root}",
        flush=True,
    )
    actual_relatives = {path.relative_to(zoo_root).as_posix() for path in files}
    if not set(checkpoint_by_path).issubset(actual_relatives):
        raise RuntimeError("checkpoint manifest paths disappeared during retention inventory")
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    byte_totals: Counter[str] = Counter()
    reason_totals: Counter[str] = Counter()
    reason_byte_totals: Counter[str] = Counter()
    inventory_started = time.monotonic()
    for file_index, path in enumerate(files, start=1):
        relative = path.relative_to(zoo_root).as_posix()
        action, reason = _classify_file(relative, keep_epochs=keep_epochs)
        checkpoint = checkpoint_by_path.get(relative)
        if checkpoint is not None:
            expected_action = "keep" if checkpoint["epoch_one_based"] == 0 or checkpoint["epoch_one_based"] in keep_epochs else "quarantine_candidate"
            if action != expected_action:
                raise RuntimeError(f"retention classification drift: {relative}")
            digest = str(checkpoint["sha256"])
            size = int(checkpoint["bytes"])
            snapshot = dict(checkpoint["stat"])
            if _stat_identity(path) != snapshot:
                raise RuntimeError(f"checkpoint stat identity drifted before inventory commit: {path}")
        else:
            if action == "quarantine_candidate":
                raise RuntimeError(f"unmanifested file cannot become a retention candidate: {relative}")
            digest, snapshot = _hash_with_stable_stat(path)
            size = snapshot["bytes"]
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": digest,
                "stat": snapshot,
                "action": action,
                "reason": reason,
            }
        )
        totals[action] += 1
        byte_totals[action] += size
        reason_totals[reason] += 1
        reason_byte_totals[reason] += size
        if file_index % 1_000 == 0 or file_index == len(files):
            elapsed = max(time.monotonic() - inventory_started, 1e-9)
            print(
                f"[zoo-retention] stage=inventory-zoo progress={file_index}/{len(files)} "
                f"elapsed_s={elapsed:.1f} rate_files_s={file_index / elapsed:.2f}",
                flush=True,
            )
    summary = {
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "action_file_counts": dict(sorted(totals.items())),
        "action_bytes": dict(sorted(byte_totals.items())),
        "reason_file_counts": dict(sorted(reason_totals.items())),
        "reason_bytes": dict(sorted(reason_byte_totals.items())),
        "retained_state_files": sum(
            1
            for row in rows
            if row["action"] == "keep"
            and (row["reason"] == "initialization" or row["reason"] == "raw_fp32_grid9")
        ),
        "primary_state_files": sum(
            1
            for row in checkpoint_rows
            if bool(row["is_primary"])
            and (int(row["epoch_one_based"]) == 0 or int(row["epoch_one_based"]) in keep_epochs)
        ),
    }
    return rows, summary


def _assert_post_scan_stability(
    *,
    zoo_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    expected_snapshots: Mapping[str, Mapping[str, int]],
    proc_root: Path,
    checkpoint_manifest_path: Path | None = None,
    checkpoint_manifest_sha256: str | None = None,
    lineage_manifest_path: Path | None = None,
    lineage_manifest_sha256: str | None = None,
) -> None:
    if live := live_zoo_builder_processes(proc_root=proc_root):
        raise RuntimeError(f"zoo builder appeared during retention scan: {live}")
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha256:
        raise RuntimeError("final zoo manifest drifted during retention scan")
    current_relative_paths = {
        path.relative_to(zoo_root).as_posix() for path in _walk_files(zoo_root)
    }
    expected_relative_paths = set(expected_snapshots)
    if current_relative_paths != expected_relative_paths:
        missing = sorted(expected_relative_paths - current_relative_paths)[:5]
        added = sorted(current_relative_paths - expected_relative_paths)[:5]
        raise RuntimeError(f"zoo inventory paths drifted during retention scan: missing={missing} added={added}")
    for relative in sorted(expected_relative_paths):
        if _stat_identity(zoo_root / relative) != dict(expected_snapshots[relative]):
            raise RuntimeError(f"zoo file stat identity drifted during retention scan: {relative}")
    for label, path, expected_sha in (
        ("checkpoint manifest", checkpoint_manifest_path, checkpoint_manifest_sha256),
        ("lineage manifest", lineage_manifest_path, lineage_manifest_sha256),
    ):
        if path is not None and (not path.is_file() or sha256_file(path) != expected_sha):
            raise RuntimeError(f"{label} drifted during retention scan: {path}")


def build_plan(
    config_path: Path,
    *,
    expected_final_manifest_sha256: str,
    proc_root: Path = Path("/proc"),
    production: bool = True,
    source_workspace_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    config = _load_mapping(config_path)
    resolved = _validate_production_config(config) if production else config
    paths = resolved["paths"]
    policy = resolved["policy"]
    safety = resolved["safety"]
    zoo_root = _safe_root(Path(str(paths["zoo_root"])), label="zoo root")
    manifest_path = Path(str(paths["final_zoo_manifest"])).resolve()
    output_root = _safe_root(Path(str(paths["plan_output_root"])), label="plan output root")
    quarantine_root = _safe_root(Path(str(paths["quarantine_root"])), label="quarantine root")
    if output_root.is_relative_to(zoo_root) or quarantine_root.is_relative_to(zoo_root):
        raise ValueError("plan/quarantine roots must be outside the live zoo root")
    print(
        "[zoo-retention] stage=resolved-plan-config "
        f"device=cpu dtype=raw-file-bytes seed=not-applicable cache=immutable "
        f"policy={policy['name']} keep_initialization={policy['keep_initialization']} "
        f"keep_epochs_one_based={policy['keep_epochs_one_based']} zoo_root={zoo_root} "
        f"manifest={manifest_path} output_root={output_root} quarantine_root={quarantine_root}",
        flush=True,
    )
    if safety.get("fail_if_zoo_builder_live") and (live := live_zoo_builder_processes(proc_root=proc_root)):
        raise RuntimeError(f"zoo builder is still live; retention planning is forbidden: {live}")
    validated = _validate_final_manifest(
        manifest_path,
        expected_sha256=expected_final_manifest_sha256,
        zoo_root=zoo_root,
        expected_datasets=PRODUCTION_DATASETS if production else tuple(policy["dataset_names"]),
        expected_lineages_per_dataset=int(policy["expected_lineages_per_dataset"]),
        expected_states_per_lineage=int(policy["expected_archived_states_per_lineage"]),
        expected_primary_indices=tuple(int(v) for v in policy["primary_checkpoint_indices_zero_based"]),
        expected_lineage_split=(35, 7, 8) if production else tuple(policy["lineage_split"]),
        seed_namespace=(
            "weightclip-resnet18slim-zoo-v1" if production else str(policy["seed_namespace"])
        ),
        require_forward_audit=bool(policy["require_forward_audit"]),
    )
    keep_epochs = {int(value) for value in policy["keep_epochs_one_based"]}
    rows, summary = _inventory_zoo(
        zoo_root,
        checkpoint_rows=validated["checkpoint_rows"],
        keep_epochs=keep_epochs,
    )
    expected_snapshots = {
        str(row["relative_path"]): dict(row["stat"])
        for row in rows
    }
    _assert_post_scan_stability(
        zoo_root=zoo_root,
        manifest_path=manifest_path,
        manifest_sha256=validated["manifest_sha256"],
        expected_snapshots=expected_snapshots,
        proc_root=proc_root,
        checkpoint_manifest_path=Path(validated["checkpoint_manifest_path"]),
        checkpoint_manifest_sha256=validated["checkpoint_manifest_sha256"],
        lineage_manifest_path=Path(validated["lineage_manifest_path"]),
        lineage_manifest_sha256=validated["lineage_manifest_sha256"],
    )
    expected_retained_states = len(PRODUCTION_DATASETS if production else policy["dataset_names"]) * int(
        policy["expected_lineages_per_dataset"]
    ) * (1 + len(keep_epochs))
    if summary["retained_state_files"] != expected_retained_states:
        raise RuntimeError(
            f"grid9 retained-state count mismatch: {summary['retained_state_files']} != {expected_retained_states}"
        )
    if summary["primary_state_files"] != int(policy["expected_primary_checkpoint_states"]):
        raise RuntimeError("retention plan does not preserve every primary checkpoint")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": "weightclip_zoo_retention_plan",
        "status": "plan_only_unapproved",
        "destructive_apply_enabled": False,
        "policy": policy,
        "safety": safety,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_implementation": retention_source_inventory(
            workspace_root=source_workspace_root
        ),
        "zoo_root": str(zoo_root),
        "quarantine_root": str(quarantine_root),
        "final_zoo_manifest_path": str(manifest_path),
        "final_zoo_manifest_sha256": validated["manifest_sha256"],
        "checkpoint_manifest_path": validated["checkpoint_manifest_path"],
        "checkpoint_manifest_sha256": validated["checkpoint_manifest_sha256"],
        "lineage_manifest_path": validated["lineage_manifest_path"],
        "lineage_manifest_sha256": validated["lineage_manifest_sha256"],
        "inventory": rows,
        "summary": summary,
        "recovery_contract": {
            "workflow": "atomic_quarantine_then_verify_then_recovery_drill_then_separate_reclaim_approval",
            "recovery_drill_required_before_reclaim": True,
            "reclaim_implemented": False,
        },
    }
    plan_sha = _fingerprint(contract)
    plan = {**contract, "plan_sha256": plan_sha}
    output = output_root / f"retention-plan-{plan_sha[:16]}.json"
    write_json_immutable(output, plan)
    return plan, output


def verify_plan(
    plan_path: Path,
    *,
    proc_root: Path = Path("/proc"),
    source_workspace_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    plan_path = plan_path.resolve()
    plan, expected_plan_sha = _validated_plan(
        plan_path, source_workspace_root=source_workspace_root
    )
    if live := live_zoo_builder_processes(proc_root=proc_root):
        raise RuntimeError(f"zoo builder is still live; retention verification is forbidden: {live}")
    zoo_root = _safe_root(Path(str(plan["zoo_root"])), label="zoo root")
    manifest_path = Path(str(plan["final_zoo_manifest_path"])).resolve()
    if sha256_file(manifest_path) != plan["final_zoo_manifest_sha256"]:
        raise RuntimeError("final zoo manifest drifted after retention planning")
    current_paths = {path.relative_to(zoo_root).as_posix(): path for path in _walk_files(zoo_root)}
    planned_rows = plan.get("inventory")
    if not isinstance(planned_rows, list):
        raise ValueError("retention plan inventory is missing")
    planned_paths = {str(row["relative_path"]) for row in planned_rows}
    if set(current_paths) != planned_paths:
        missing = sorted(planned_paths - set(current_paths))[:5]
        added = sorted(set(current_paths) - planned_paths)[:5]
        raise RuntimeError(f"zoo inventory drift: missing={missing} added={added}")
    checked_bytes = 0
    verified_snapshots: dict[str, dict[str, int]] = {}
    verify_started = time.monotonic()
    print(
        f"[zoo-retention] stage=verify-plan-inventory total={len(planned_rows)} root={zoo_root}",
        flush=True,
    )
    for file_index, row in enumerate(planned_rows, start=1):
        relative = str(row["relative_path"])
        path = current_paths[relative]
        size = int(row["bytes"])
        digest = _assert_sha(str(row["sha256"]), label=f"planned file SHA {relative}")
        planned_snapshot = row.get("stat")
        if not isinstance(planned_snapshot, dict):
            raise ValueError(f"retention plan lacks file stat identity: {relative}")
        actual_digest, actual_snapshot = _hash_with_stable_stat(
            path,
            expected_bytes=size,
            expected_sha256=digest,
        )
        if actual_digest != digest or actual_snapshot != planned_snapshot:
            raise RuntimeError(f"zoo file content/stat drift after retention planning: {path}")
        verified_snapshots[relative] = actual_snapshot
        checked_bytes += size
        if file_index % 250 == 0 or file_index == len(planned_rows):
            elapsed = max(time.monotonic() - verify_started, 1e-9)
            print(
                f"[zoo-retention] stage=verify-plan-inventory "
                f"progress={file_index}/{len(planned_rows)} bytes={checked_bytes} "
                f"elapsed_s={elapsed:.1f} rate_files_s={file_index / elapsed:.2f}",
                flush=True,
            )
    _assert_post_scan_stability(
        zoo_root=zoo_root,
        manifest_path=manifest_path,
        manifest_sha256=str(plan["final_zoo_manifest_sha256"]),
        expected_snapshots=verified_snapshots,
        proc_root=proc_root,
        checkpoint_manifest_path=Path(str(plan["checkpoint_manifest_path"])),
        checkpoint_manifest_sha256=str(plan["checkpoint_manifest_sha256"]),
        lineage_manifest_path=Path(str(plan["lineage_manifest_path"])),
        lineage_manifest_sha256=str(plan["lineage_manifest_sha256"]),
    )
    report_contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": "weightclip_zoo_retention_verification",
        "status": "verified_read_only_no_apply",
        "plan_path": str(plan_path),
        "plan_artifact_sha256": sha256_file(plan_path),
        "plan_sha256": expected_plan_sha,
        "final_zoo_manifest_sha256": plan["final_zoo_manifest_sha256"],
        "verified_files": len(planned_rows),
        "verified_bytes": checked_bytes,
        "inventory_exact": True,
        "hashes_exact": True,
        "recovery_drill_status": "not_run_no_quarantine_exists",
        "destructive_apply_enabled": False,
    }
    report_sha = _fingerprint(report_contract)
    report = {**report_contract, "verification_sha256": report_sha}
    output = plan_path.parent / f"retention-verification-{expected_plan_sha[:16]}-{report_sha[:16]}.json"
    write_json_immutable(output, report)
    return report, output


def _validate_approval(approval_path: Path, plan: Mapping[str, Any]) -> None:
    approval = _load_json(approval_path)
    required = {
        "status": "approved",
        "approved_by_user": True,
        "plan_sha256": plan["plan_sha256"],
        "final_zoo_manifest_sha256": plan["final_zoo_manifest_sha256"],
    }
    for key, value in required.items():
        if approval.get(key) != value:
            raise RuntimeError(f"retention approval mismatch: {key}")
    _assert_sha(str(approval.get("recovery_drill_report_sha256")), label="recovery drill report SHA")


def _validated_plan(
    plan_path: Path, *, source_workspace_root: Path | None = None
) -> tuple[dict[str, Any], str]:
    plan = _load_json(plan_path)
    expected_plan_sha = _assert_sha(str(plan.get("plan_sha256")), label="plan SHA")
    contract = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _fingerprint(contract) != expected_plan_sha:
        raise RuntimeError("retention plan content hash mismatch")
    if plan.get("status") != "plan_only_unapproved" or plan.get("destructive_apply_enabled") is not False:
        raise RuntimeError("retention plan is not an unapproved read-only plan")
    source = plan.get("source_implementation")
    if not isinstance(source, dict):
        raise RuntimeError("retention plan lacks its source implementation inventory")
    current_source = retention_source_inventory(workspace_root=source_workspace_root)
    if current_source != source:
        raise RuntimeError("retention source implementation drifted after planning")
    config_path = Path(str(plan.get("config_path"))).resolve()
    if not config_path.is_file() or sha256_file(config_path) != plan.get("config_sha256"):
        raise RuntimeError("retention config drifted after planning")
    return plan, expected_plan_sha


def refuse_apply(
    plan_path: Path,
    *,
    approval_path: Path | None,
    proc_root: Path = Path("/proc"),
) -> None:
    plan, _plan_sha = _validated_plan(plan_path.resolve())
    if live := live_zoo_builder_processes(proc_root=proc_root):
        raise RuntimeError(f"zoo builder is still live; retention apply is forbidden: {live}")
    manifest_path = Path(str(plan["final_zoo_manifest_path"])).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != plan["final_zoo_manifest_sha256"]:
        raise RuntimeError("final zoo manifest drifted after retention planning")
    if approval_path is None:
        raise RuntimeError("retention apply requires an explicit user-approved exact plan JSON")
    _validate_approval(approval_path.resolve(), plan)
    raise RuntimeError(
        "destructive retention apply is intentionally disabled in this reviewed build; "
        "implement and independently review atomic quarantine/recovery/reclaim before enabling it"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan/verify WeightCLIP zoo retention without deleting data")
    parser.add_argument("stage", nargs="?", choices=("plan", "verify", "apply"), default="plan")
    parser.add_argument("--config", type=Path, default=Path("conf/weightclip_benchmark/zoo_retention.yaml"))
    parser.add_argument("--final-zoo-manifest-sha256", type=str)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approval", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        f"[zoo-retention] stage={args.stage} config={args.config.resolve()} "
        "mode=read-only-unless-separately-approved cache=immutable",
        flush=True,
    )
    if args.stage == "plan":
        if args.final_zoo_manifest_sha256 is None:
            raise ValueError("plan requires --final-zoo-manifest-sha256")
        plan, output = build_plan(
            args.config,
            expected_final_manifest_sha256=args.final_zoo_manifest_sha256,
        )
        print(
            f"[zoo-retention] status=PLAN_ONLY_UNAPPROVED plan_sha256={plan['plan_sha256']} "
            f"files={plan['summary']['files']} keep_bytes={plan['summary']['action_bytes'].get('keep', 0)} "
            f"quarantine_candidate_bytes={plan['summary']['action_bytes'].get('quarantine_candidate', 0)} "
            f"artifact={output}",
            flush=True,
        )
        return
    if args.plan is None:
        raise ValueError(f"{args.stage} requires --plan")
    if args.stage == "verify":
        report, output = verify_plan(args.plan)
        print(
            f"[zoo-retention] status={report['status']} files={report['verified_files']} "
            f"bytes={report['verified_bytes']} artifact={output}",
            flush=True,
        )
        return
    refuse_apply(args.plan, approval_path=args.approval)


if __name__ == "__main__":
    main()
