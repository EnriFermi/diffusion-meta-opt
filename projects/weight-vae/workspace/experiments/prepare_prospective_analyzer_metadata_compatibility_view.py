#!/usr/bin/env python3
"""Build the frozen-analyzer metadata compatibility view without reading outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_20260816"
DEFAULT_OUTPUT = PROJECT / "artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_analyzer_compat_input_20260816"
DEFAULT_REPORT = PROJECT / "artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_analyzer_compat_amendment_20260816"
ANALYZER = PROJECT / "experiments/analyze_prospective_geometry_matched_replication.py"
AMENDMENT_NOTE = PROJECT / "docs/notes/prospective_geometry_matched_analyzer_metadata_compatibility_amendment_20260816.md"

EXPECTED_SOURCE_MANIFEST_SHA256 = "b30b8064ccd98e6d83b0e557e1267be5cd4abfc0d6e9b5430b2f4c1f5f95b80e"
EXPECTED_ANALYZER_SHA256 = "94cfc4012f1e64e272d43ff1045a86b2b8d5870777f9c4f14bcdd0e14c25cb8d"
EXPECTED_AMENDMENT_NOTE_SHA256 = "bd5226a7c5a5c44d2baec69b440dbf9a03e6395543607e0e24f1b07e8bf4c0c2"
EXPECTED_PAYLOAD_COUNT = 20
METADATA_DELETION = "loaded_local_module_closure_pass"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == "artifact_manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"compatibility payload must be a regular non-symlink file: {path}")
        rows.append(
            {
                "bytes": path.stat().st_size,
                "path": path.name,
                "sha256": sha256_file(path),
            }
        )
    return rows


def verify_manifest(root: Path, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"missing regular artifact manifest: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise RuntimeError(
            f"source artifact manifest SHA mismatch: {manifest_sha} != {expected_manifest_sha256}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "prospective_geometry_matched_replication_artifact_manifest_v1":
        raise RuntimeError("artifact manifest schema mismatch")
    if payload.get("manifest_self_excluded") is not True:
        raise RuntimeError("artifact manifest is not self-excluding")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or payload.get("count") != len(rows):
        raise RuntimeError("artifact manifest list/count mismatch")
    declared = {str(row.get("path")) for row in rows}
    actual = {path.name for path in root.iterdir() if path.name != "artifact_manifest.json"}
    if declared != actual or len(declared) != len(rows):
        raise RuntimeError(f"artifact manifest membership mismatch: declared={declared} actual={actual}")
    for row in rows:
        if set(row) != {"bytes", "path", "sha256"}:
            raise RuntimeError(f"artifact manifest row schema mismatch: {row}")
        path = root / str(row["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"manifest payload is not a regular non-symlink file: {path}")
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"artifact manifest hash/size mismatch: {path}")
    return {
        "manifest_sha256": manifest_sha,
        "payload_count": len(rows),
        "payloads": {str(row["path"]): {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])} for row in rows},
    }


def semantic_diff(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                rows.append({"path": path, "kind": "added", "before": None, "after": right[key]})
            elif key not in right:
                rows.append({"path": path, "kind": "deleted", "before": left[key], "after": None})
            else:
                rows.extend(semantic_diff(left[key], right[key], path))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": prefix, "kind": "changed", "before": left, "after": right}]
        rows = []
        for index, (before, after) in enumerate(zip(left, right, strict=True)):
            rows.extend(semantic_diff(before, after, f"{prefix}[{index}]"))
        return rows
    if left != right or type(left) is not type(right):
        return [{"path": prefix, "kind": "changed", "before": left, "after": right}]
    return []


def report_manifest(root: Path) -> dict[str, Any]:
    rows = manifest_rows(root)
    return {
        "artifacts": rows,
        "count": len(rows),
        "manifest_self_excluded": True,
        "schema_version": "prospective_analyzer_metadata_compatibility_amendment_manifest_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input_dir.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    report_dir = args.report_dir.resolve(strict=False)
    started = time.monotonic()
    print(
        "resolved_config "
        f"input={source} output={output} report={report_dir} "
        "mode=metadata_schema_projection copy_mode=independent verbose=true"
    )
    if output.exists() or report_dir.exists():
        raise FileExistsError(f"fresh output/report required: output={output} report={report_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir.parent.mkdir(parents=True, exist_ok=True)

    print("stage=frozen_authorization_audit")
    analyzer_sha = sha256_file(ANALYZER)
    note_sha = sha256_file(AMENDMENT_NOTE)
    if analyzer_sha != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError(f"frozen analyzer SHA mismatch: {analyzer_sha}")
    if note_sha != EXPECTED_AMENDMENT_NOTE_SHA256:
        raise RuntimeError(f"amendment-note SHA mismatch: {note_sha}")
    source_before = verify_manifest(source, EXPECTED_SOURCE_MANIFEST_SHA256)
    if source_before["payload_count"] != EXPECTED_PAYLOAD_COUNT:
        raise RuntimeError(f"source payload count mismatch: {source_before['payload_count']}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        print(f"stage=independent_copy files={EXPECTED_PAYLOAD_COUNT}")
        for index, name in enumerate(sorted(source_before["payloads"]), start=1):
            source_path = source / name
            destination_path = temporary / name
            shutil.copy2(source_path, destination_path, follow_symlinks=False)
            if source_path.stat().st_dev == destination_path.stat().st_dev and source_path.stat().st_ino == destination_path.stat().st_ino:
                raise RuntimeError(f"copy unexpectedly shares an inode with source: {name}")
            if sha256_file(destination_path) != source_before["payloads"][name]["sha256"]:
                raise RuntimeError(f"copy hash mismatch: {name}")
            print(f"stage=copy file={index}/{EXPECTED_PAYLOAD_COUNT} name={name}")

        print("stage=runner_metadata_projection")
        metadata_path = temporary / "runner_metadata.json"
        metadata_before = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_after = copy.deepcopy(metadata_before)
        hard_checks = metadata_after.get("hard_checks")
        if not isinstance(hard_checks, dict) or hard_checks.get(METADATA_DELETION) is not True:
            raise RuntimeError(f"required true metadata key is absent: {METADATA_DELETION}")
        del hard_checks[METADATA_DELETION]
        metadata_diff = semantic_diff(metadata_before, metadata_after)
        expected_metadata_diff = [
            {
                "path": f"hard_checks.{METADATA_DELETION}",
                "kind": "deleted",
                "before": True,
                "after": None,
            }
        ]
        if metadata_diff != expected_metadata_diff:
            raise RuntimeError(f"runner metadata diff is not the authorized one-key deletion: {metadata_diff}")
        metadata_old = {"bytes": metadata_path.stat().st_size, "sha256": sha256_file(metadata_path)}
        write_json(metadata_path, metadata_after)
        metadata_new = {"bytes": metadata_path.stat().st_size, "sha256": sha256_file(metadata_path)}

        print("stage=binding_path_projection")
        binding_path = temporary / "preexecution_binding.json"
        binding_before = json.loads(binding_path.read_text(encoding="utf-8"))
        binding_after = copy.deepcopy(binding_before)
        expected_old_contract = str((source / "preexecution_contract.json").resolve(strict=True))
        old_contract = str(Path(str(binding_after.get("copied_contract_path", ""))).resolve(strict=True))
        if old_contract != expected_old_contract:
            raise RuntimeError(f"original copied_contract_path mismatch: {old_contract} != {expected_old_contract}")
        final_contract = str((output / "preexecution_contract.json").resolve(strict=False))
        binding_after["copied_contract_path"] = final_contract
        binding_diff = semantic_diff(binding_before, binding_after)
        expected_binding_diff = [
            {
                "path": "copied_contract_path",
                "kind": "changed",
                "before": binding_before["copied_contract_path"],
                "after": final_contract,
            }
        ]
        if binding_diff != expected_binding_diff:
            raise RuntimeError(f"preexecution binding diff is not the authorized path change: {binding_diff}")
        binding_old = {"bytes": binding_path.stat().st_size, "sha256": sha256_file(binding_path)}
        write_json(binding_path, binding_after)
        binding_new = {"bytes": binding_path.stat().st_size, "sha256": sha256_file(binding_path)}

        print("stage=compatibility_manifest")
        rows = manifest_rows(temporary)
        if len(rows) != EXPECTED_PAYLOAD_COUNT:
            raise RuntimeError(f"compatibility payload count mismatch: {len(rows)}")
        write_json(
            temporary / "artifact_manifest.json",
            {
                "artifacts": rows,
                "count": len(rows),
                "manifest_self_excluded": True,
                "schema_version": "prospective_geometry_matched_replication_artifact_manifest_v1",
            },
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    compat = verify_manifest(output)
    changed = {"runner_metadata.json", "preexecution_binding.json"}
    unchanged: dict[str, Any] = {}
    for name, source_record in source_before["payloads"].items():
        destination = output / name
        if destination.stat().st_dev == (source / name).stat().st_dev and destination.stat().st_ino == (source / name).stat().st_ino:
            raise RuntimeError(f"compatibility payload shares source inode: {name}")
        if name not in changed:
            if sha256_file(destination) != source_record["sha256"] or destination.stat().st_size != source_record["bytes"]:
                raise RuntimeError(f"unauthorized compatibility payload change: {name}")
            unchanged[name] = source_record
    if len(unchanged) != EXPECTED_PAYLOAD_COUNT - len(changed):
        raise RuntimeError(f"unchanged payload count mismatch: {len(unchanged)}")

    print("stage=source_immutability_recheck")
    source_after = verify_manifest(source, EXPECTED_SOURCE_MANIFEST_SHA256)
    if source_after != source_before:
        raise RuntimeError("source manifest verification changed during compatibility preparation")

    report_dir.mkdir(parents=False, exist_ok=False)
    report = {
        "schema_version": "prospective_analyzer_metadata_compatibility_amendment_v1",
        "status": "COMPLETE",
        "scope": "metadata schema projection only; no scientific outcome was computed",
        "source_dir": str(source),
        "source_manifest_sha256_before": source_before["manifest_sha256"],
        "source_manifest_sha256_after": source_after["manifest_sha256"],
        "compatibility_dir": str(output),
        "compatibility_manifest_sha256": compat["manifest_sha256"],
        "payload_count": compat["payload_count"],
        "analyzer_path": str(ANALYZER.resolve(strict=True)),
        "analyzer_sha256": analyzer_sha,
        "amendment_note_path": str(AMENDMENT_NOTE.resolve(strict=True)),
        "amendment_note_sha256": note_sha,
        "metadata_transformation": {
            "semantic_diff": metadata_diff,
            "before": metadata_old,
            "after": metadata_new,
        },
        "binding_transformation": {
            "semantic_diff": binding_diff,
            "before": binding_old,
            "after": binding_new,
        },
        "byte_identical_payloads": unchanged,
        "byte_identical_payload_count": len(unchanged),
        "changed_payloads": sorted(changed),
        "independent_copy_no_shared_inodes": True,
        "source_immutability_pass": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(report_dir / "compatibility_report.json", report)
    (report_dir / "README.md").write_text(
        "# Analyzer metadata compatibility amendment\n\n"
        "Verdict: `COMPLETE_METADATA_ONLY`.\n\n"
        f"- Source manifest: `{source_before['manifest_sha256']}` (unchanged before/after).\n"
        f"- Compatibility manifest: `{compat['manifest_sha256']}`.\n"
        f"- Frozen analyzer: `{analyzer_sha}`.\n"
        "- Authorized changes: delete one redundant true hard-check key; rewrite one copied-contract path.\n"
        f"- Byte-identical payloads: `{len(unchanged)}/{EXPECTED_PAYLOAD_COUNT}`; changed metadata payloads: `2/{EXPECTED_PAYLOAD_COUNT}`.\n"
        "- No scientific outcome was computed by this preparer.\n",
        encoding="utf-8",
    )
    write_json(report_dir / "artifact_manifest.json", report_manifest(report_dir))
    print(
        "stage=complete "
        f"elapsed={time.monotonic() - started:.2f}s "
        f"compatibility_manifest={compat['manifest_sha256']} "
        f"report={report_dir / 'compatibility_report.json'}"
    )


if __name__ == "__main__":
    main()
