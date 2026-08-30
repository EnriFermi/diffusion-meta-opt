#!/usr/bin/env python3
"""Rebase the exact frozen analyzer path so its existing root allowlist is valid."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts/crossmodal_united_structure"
).resolve(strict=True)
DEFAULT_INPUT = (
    ARTIFACT_ROOT
    / "prospective_geometry_matched_replication_analyzer_compat_input_20260816"
)
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "prospective_geometry_matched_replication_analyzer_project_root_compat_input_20260816"
)
DEFAULT_REPORT = (
    ARTIFACT_ROOT
    / "prospective_geometry_matched_replication_analyzer_project_root_compat_amendment_20260816"
)
ORIGINAL_RAW = ARTIFACT_ROOT / "prospective_geometry_matched_replication_20260816"
FROZEN_ANALYZER = WORKSPACE / "experiments/analyze_prospective_geometry_matched_replication.py"
DEFAULT_ANALYZER_ROOT = PROJECT_ROOT / "_analyzer_compat_20260816"
ANALYZER_FILENAME = "analyze_prospective_geometry_matched_replication.py"
EXTERNAL_CONTRACT_FILENAME = "preexecution_contract_project_root_rebased_20260816.json"
AMENDMENT_NOTE = (
    WORKSPACE
    / "docs/notes/prospective_geometry_matched_analyzer_project_root_rebase_amendment_20260816.md"
)

EXPECTED_ORIGINAL_MANIFEST_SHA256 = "b30b8064ccd98e6d83b0e557e1267be5cd4abfc0d6e9b5430b2f4c1f5f95b80e"
EXPECTED_INPUT_MANIFEST_SHA256 = "b265015ba57592fe2125f4cbe2d999e3d7de6270a2a01091fb97964d9a19a9b7"
EXPECTED_ANALYZER_SHA256 = "94cfc4012f1e64e272d43ff1045a86b2b8d5870777f9c4f14bcdd0e14c25cb8d"
EXPECTED_AMENDMENT_NOTE_SHA256 = "1841e987382ccdc78159f0b1298e83471019ab63a6f5e20e0ed8e42994362d4d"
EXPECTED_PAYLOAD_COUNT = 20
CHANGED_PAYLOADS = {
    "preexecution_binding.json",
    "preexecution_contract.json",
    "runner_metadata.json",
}


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
    if type(left) is not type(right) or left != right:
        return [{"path": prefix, "kind": "changed", "before": left, "after": right}]
    return []


def artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == "artifact_manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"payload must be a regular non-symlink file: {path}")
        rows.append(
            {"bytes": path.stat().st_size, "path": path.name, "sha256": sha256_file(path)}
        )
    return rows


def write_runner_manifest(root: Path) -> str:
    rows = artifact_rows(root)
    if len(rows) != EXPECTED_PAYLOAD_COUNT:
        raise RuntimeError(f"runner payload count mismatch: {len(rows)}")
    write_json(
        root / "artifact_manifest.json",
        {
            "artifacts": rows,
            "count": len(rows),
            "manifest_self_excluded": True,
            "schema_version": "prospective_geometry_matched_replication_artifact_manifest_v1",
        },
    )
    return sha256_file(root / "artifact_manifest.json")


def verify_runner_manifest(root: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"missing regular runner manifest: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if expected_sha256 is not None and manifest_sha != expected_sha256:
        raise RuntimeError(f"runner manifest SHA mismatch: {manifest_sha} != {expected_sha256}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {"artifacts", "count", "manifest_self_excluded", "schema_version"}:
        raise RuntimeError("runner manifest exact top-level schema mismatch")
    if payload["schema_version"] != "prospective_geometry_matched_replication_artifact_manifest_v1":
        raise RuntimeError("runner manifest schema version mismatch")
    if payload["manifest_self_excluded"] is not True:
        raise RuntimeError("runner manifest self-exclusion mismatch")
    rows = payload["artifacts"]
    if not isinstance(rows, list) or payload["count"] != len(rows):
        raise RuntimeError("runner manifest list/count mismatch")
    declared = {str(row.get("path")) for row in rows}
    actual = {path.name for path in root.iterdir() if path.name != "artifact_manifest.json"}
    if declared != actual or len(declared) != len(rows):
        raise RuntimeError(f"runner manifest membership mismatch: {declared} != {actual}")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != {"bytes", "path", "sha256"}:
            raise RuntimeError(f"runner manifest row schema mismatch: {row}")
        name = str(row["path"])
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"runner payload is not a regular non-symlink file: {path}")
        record = {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"runner payload hash/size mismatch: {path}")
        records[name] = record
    return {"manifest_sha256": manifest_sha, "payload_count": len(rows), "payloads": records}


def independent_copy(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination, follow_symlinks=False)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (destination_stat.st_dev, destination_stat.st_ino):
        raise RuntimeError(f"copy unexpectedly shares source inode: {source}")
    if source_stat.st_size != destination_stat.st_size or sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"independent copy mismatch: {source} -> {destination}")


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_disjoint(paths: Mapping[str, Path]) -> None:
    names = sorted(paths)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = paths[left_name]
            right = paths[right_name]
            if is_within(left, right) or is_within(right, left):
                raise RuntimeError(
                    f"compatibility paths must be disjoint: {left_name}={left} {right_name}={right}"
                )


def iter_file_records(value: Any, prefix: str = "") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield prefix, value
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_file_records(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_file_records(child, f"{prefix}[{index}]")


def verify_rebased_contract_paths(
    contract: Mapping[str, Any],
    analyzer_path: Path,
    record_path_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    expected_project = PROJECT_ROOT.resolve(strict=True)
    computed_project = analyzer_path.resolve(strict=True).parents[1]
    if computed_project != expected_project:
        raise RuntimeError(f"rebased analyzer PROJECT mismatch: {computed_project} != {expected_project}")
    allowed_roots = (expected_project, Path(sys.prefix).resolve(strict=True))
    audits: list[dict[str, Any]] = []
    for label, record in iter_file_records(contract):
        raw = Path(str(record["path"]))
        if not raw.is_absolute():
            raw = expected_project / raw
        path = raw.resolve(strict=False)
        if not any(is_within(path, root) for root in allowed_roots):
            raise RuntimeError(f"rebased contract path remains outside allowed roots: {label}={path}")
        audit_path = (
            record_path_overrides[label].resolve(strict=True)
            if record_path_overrides is not None and label in record_path_overrides
            else raw.resolve(strict=True)
        )
        expected_sha = str(record["sha256"])
        if sha256_file(audit_path) != expected_sha:
            raise RuntimeError(f"rebased contract file SHA mismatch: {label}={path}")
        if "bytes" in record and audit_path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"rebased contract file size mismatch: {label}={path}")
        audits.append(
            {
                "label": label,
                "path": str(path),
                "audited_path": str(audit_path),
                "sha256": expected_sha,
            }
        )
    return {
        "computed_project": str(computed_project),
        "allowed_roots": [str(root) for root in allowed_roots],
        "file_records_rehashed": len(audits),
        "records": audits,
        "pass": True,
    }


def report_manifest(root: Path) -> dict[str, Any]:
    rows = artifact_rows(root)
    return {
        "artifacts": rows,
        "count": len(rows),
        "manifest_self_excluded": True,
        "schema_version": "prospective_analyzer_project_root_rebase_amendment_manifest_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--analyzer-root", type=Path, default=DEFAULT_ANALYZER_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input_dir.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    report = args.report_dir.resolve(strict=False)
    analyzer_root = args.analyzer_root.resolve(strict=False)
    original = ORIGINAL_RAW.resolve(strict=True)
    started = time.monotonic()
    print(
        "resolved_config "
        f"input={source} output={output} report={report} analyzer_root={analyzer_root} "
        "device=cpu mode=exact_analyzer_path_rebase copy_mode=independent verbose=true"
    )
    require_disjoint(
        {"source": source, "output": output, "report": report, "analyzer_root": analyzer_root}
    )
    for path in (output, report, analyzer_root):
        if path.exists():
            raise FileExistsError(f"fresh compatibility target required: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    print("stage=frozen_authorization_audit")
    analyzer_sha = sha256_file(FROZEN_ANALYZER)
    note_sha = sha256_file(AMENDMENT_NOTE)
    if analyzer_sha != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError(f"frozen analyzer SHA mismatch: {analyzer_sha}")
    if note_sha != EXPECTED_AMENDMENT_NOTE_SHA256:
        raise RuntimeError(f"project-root amendment note SHA mismatch: {note_sha}")
    original_before = verify_runner_manifest(original, EXPECTED_ORIGINAL_MANIFEST_SHA256)
    source_before = verify_runner_manifest(source, EXPECTED_INPUT_MANIFEST_SHA256)
    if source_before["payload_count"] != EXPECTED_PAYLOAD_COUNT:
        raise RuntimeError(f"input payload count mismatch: {source_before['payload_count']}")

    output_staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    analyzer_staging = Path(
        tempfile.mkdtemp(prefix=f".{analyzer_root.name}.tmp-", dir=analyzer_root.parent)
    )
    try:
        print(f"stage=independent_payload_copy files={EXPECTED_PAYLOAD_COUNT}")
        for index, name in enumerate(sorted(source_before["payloads"]), start=1):
            independent_copy(source / name, output_staging / name)
            print(f"stage=copy file={index}/{EXPECTED_PAYLOAD_COUNT} name={name}")

        print("stage=exact_analyzer_copy")
        staged_analyzer = analyzer_staging / ANALYZER_FILENAME
        independent_copy(FROZEN_ANALYZER, staged_analyzer)
        final_analyzer = analyzer_root / ANALYZER_FILENAME
        if sha256_file(staged_analyzer) != EXPECTED_ANALYZER_SHA256:
            raise RuntimeError("staged analyzer is not byte-identical to the frozen analyzer")

        print("stage=contract_analyzer_path_rebase")
        contract_path = output_staging / "preexecution_contract.json"
        contract_before = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_after = copy.deepcopy(contract_before)
        analyzer_record = contract_after.get("scripts", {}).get("analyzer")
        if not isinstance(analyzer_record, dict):
            raise RuntimeError("contract scripts.analyzer is not a mapping")
        if analyzer_record.get("sha256") != EXPECTED_ANALYZER_SHA256:
            raise RuntimeError("contract analyzer SHA is not the frozen analyzer SHA")
        analyzer_record["path"] = str(final_analyzer)
        contract_diff = semantic_diff(contract_before, contract_after)
        expected_contract_diff = [
            {
                "path": "scripts.analyzer.path",
                "kind": "changed",
                "before": contract_before["scripts"]["analyzer"]["path"],
                "after": str(final_analyzer),
            }
        ]
        if contract_diff != expected_contract_diff:
            raise RuntimeError(f"contract diff exceeds the authorized analyzer path rebase: {contract_diff}")
        contract_old = {"bytes": contract_path.stat().st_size, "sha256": sha256_file(contract_path)}
        write_json(contract_path, contract_after)
        contract_new = {"bytes": contract_path.stat().st_size, "sha256": sha256_file(contract_path)}
        contract_sha = contract_new["sha256"]

        print("stage=runner_metadata_contract_digest_rebind")
        metadata_path = output_staging / "runner_metadata.json"
        metadata_before = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_after = copy.deepcopy(metadata_before)
        metadata_after["contract_sha256"] = contract_sha
        metadata_diff = semantic_diff(metadata_before, metadata_after)
        expected_metadata_diff = [
            {
                "path": "contract_sha256",
                "kind": "changed",
                "before": metadata_before["contract_sha256"],
                "after": contract_sha,
            }
        ]
        if metadata_diff != expected_metadata_diff:
            raise RuntimeError(f"metadata diff exceeds the authorized digest rebind: {metadata_diff}")
        metadata_old = {"bytes": metadata_path.stat().st_size, "sha256": sha256_file(metadata_path)}
        write_json(metadata_path, metadata_after)
        metadata_new = {"bytes": metadata_path.stat().st_size, "sha256": sha256_file(metadata_path)}

        print("stage=preexecution_binding_rebase")
        binding_path = output_staging / "preexecution_binding.json"
        binding_before = json.loads(binding_path.read_text(encoding="utf-8"))
        binding_after = copy.deepcopy(binding_before)
        final_external_contract = analyzer_root / EXTERNAL_CONTRACT_FILENAME
        final_copied_contract = output / "preexecution_contract.json"
        binding_after["scripts"]["analyzer"]["path"] = str(final_analyzer)
        binding_after["external_contract_path"] = str(final_external_contract)
        binding_after["external_contract_sha256"] = contract_sha
        binding_after["copied_contract_path"] = str(final_copied_contract)
        binding_after["copied_contract_sha256"] = contract_sha
        binding_diff = semantic_diff(binding_before, binding_after)
        expected_binding_paths = {
            "copied_contract_path",
            "copied_contract_sha256",
            "external_contract_path",
            "external_contract_sha256",
            "scripts.analyzer.path",
        }
        if {row["path"] for row in binding_diff} != expected_binding_paths or any(
            row["kind"] != "changed" for row in binding_diff
        ):
            raise RuntimeError(f"binding diff exceeds the authorized five-field rebase: {binding_diff}")
        expected_binding_after = {
            "copied_contract_path": str(final_copied_contract),
            "copied_contract_sha256": contract_sha,
            "external_contract_path": str(final_external_contract),
            "external_contract_sha256": contract_sha,
            "scripts.analyzer.path": str(final_analyzer),
        }
        for row in binding_diff:
            if row["after"] != expected_binding_after[row["path"]]:
                raise RuntimeError(f"binding rebase value mismatch: {row}")
        if binding_after["scripts"] != contract_after["scripts"]:
            raise RuntimeError("binding and amended contract scripts differ")
        for field in ("resources", "model_dependencies", "panel_cache"):
            if binding_after[field] != contract_after[field]:
                raise RuntimeError(f"binding and amended contract differ for {field}")
        binding_old = {"bytes": binding_path.stat().st_size, "sha256": sha256_file(binding_path)}
        write_json(binding_path, binding_after)
        binding_new = {"bytes": binding_path.stat().st_size, "sha256": sha256_file(binding_path)}

        print("stage=external_contract_copy")
        staged_external_contract = analyzer_staging / EXTERNAL_CONTRACT_FILENAME
        independent_copy(contract_path, staged_external_contract)
        if staged_external_contract.read_bytes() != contract_path.read_bytes():
            raise RuntimeError("external and copied amended contracts are not byte-identical")

        print("stage=project_root_path_simulation")
        path_audit = verify_rebased_contract_paths(
            contract_after,
            staged_analyzer,
            {"scripts.analyzer": staged_analyzer},
        )
        # The staged filename is under a temporary sibling directory, so its own
        # parents[1] is still PROJECT_ROOT; the final filename is checked again after commit.
        if path_audit["computed_project"] != str(PROJECT_ROOT.resolve(strict=True)):
            raise RuntimeError("staged analyzer did not compute the intended project root")

        print("stage=compatibility_manifest")
        compatibility_manifest_sha = write_runner_manifest(output_staging)
        os.replace(analyzer_staging, analyzer_root)
        os.replace(output_staging, output)
    except BaseException:
        if output_staging.exists():
            shutil.rmtree(output_staging)
        if analyzer_staging.exists():
            shutil.rmtree(analyzer_staging)
        raise

    print("stage=post_commit_exact_audit")
    final_analyzer = analyzer_root / ANALYZER_FILENAME
    final_external_contract = analyzer_root / EXTERNAL_CONTRACT_FILENAME
    if final_analyzer.is_symlink() or final_external_contract.is_symlink():
        raise RuntimeError("rebased analyzer root contains a symlink")
    if sha256_file(final_analyzer) != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError("committed analyzer copy SHA mismatch")
    if final_analyzer.resolve(strict=True).parents[1] != PROJECT_ROOT.resolve(strict=True):
        raise RuntimeError("committed analyzer computes the wrong project root")
    if final_external_contract.read_bytes() != (output / "preexecution_contract.json").read_bytes():
        raise RuntimeError("committed external/copied amended contracts differ")
    if sha256_file(final_external_contract) != contract_sha:
        raise RuntimeError("committed external contract digest mismatch")
    compatibility = verify_runner_manifest(output, compatibility_manifest_sha)
    changed: list[str] = []
    unchanged: dict[str, Any] = {}
    shared_inodes: list[str] = []
    for name, source_record in source_before["payloads"].items():
        source_path = source / name
        destination_path = output / name
        if (source_path.stat().st_dev, source_path.stat().st_ino) == (
            destination_path.stat().st_dev,
            destination_path.stat().st_ino,
        ):
            shared_inodes.append(name)
        if sha256_file(source_path) != sha256_file(destination_path):
            changed.append(name)
        else:
            unchanged[name] = source_record
    if set(changed) != CHANGED_PAYLOADS or len(unchanged) != 17 or shared_inodes:
        raise RuntimeError(
            f"post-commit payload audit failed: changed={changed} unchanged={len(unchanged)} "
            f"shared={shared_inodes}"
        )
    committed_contract = json.loads((output / "preexecution_contract.json").read_text())
    path_audit = verify_rebased_contract_paths(committed_contract, final_analyzer)

    print("stage=ancestor_input_immutability_recheck")
    source_after = verify_runner_manifest(source, EXPECTED_INPUT_MANIFEST_SHA256)
    original_after = verify_runner_manifest(original, EXPECTED_ORIGINAL_MANIFEST_SHA256)
    if source_after != source_before or original_after != original_before:
        raise RuntimeError("an ancestor formal input changed during project-root rebase")

    report_staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.tmp-", dir=report.parent))
    report_payload = {
        "schema_version": "prospective_analyzer_project_root_rebase_amendment_v1",
        "status": "COMPLETE",
        "scope": "administrative path-layout rebase; no scientific statistic parsed or computed",
        "input_dir": str(source),
        "input_manifest_sha256_before": source_before["manifest_sha256"],
        "input_manifest_sha256_after": source_after["manifest_sha256"],
        "original_raw_dir": str(original),
        "original_raw_manifest_sha256_before": original_before["manifest_sha256"],
        "original_raw_manifest_sha256_after": original_after["manifest_sha256"],
        "output_dir": str(output),
        "output_manifest_sha256": compatibility["manifest_sha256"],
        "analyzer_source_path": str(FROZEN_ANALYZER.resolve(strict=True)),
        "analyzer_copy_path": str(final_analyzer.resolve(strict=True)),
        "analyzer_sha256": EXPECTED_ANALYZER_SHA256,
        "analyzer_copy_independent_inode": (
            (FROZEN_ANALYZER.stat().st_dev, FROZEN_ANALYZER.stat().st_ino)
            != (final_analyzer.stat().st_dev, final_analyzer.stat().st_ino)
        ),
        "analyzer_computed_project_root": str(final_analyzer.resolve(strict=True).parents[1]),
        "amendment_note_path": str(AMENDMENT_NOTE.resolve(strict=True)),
        "amendment_note_sha256": note_sha,
        "amended_contract_sha256": contract_sha,
        "external_contract_path": str(final_external_contract.resolve(strict=True)),
        "external_and_copied_contracts_byte_identical": True,
        "contract_transformation": {
            "semantic_diff": contract_diff,
            "before": contract_old,
            "after": contract_new,
        },
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
        "changed_payloads": sorted(changed),
        "byte_identical_payloads": unchanged,
        "byte_identical_payload_count": len(unchanged),
        "shared_source_inodes": shared_inodes,
        "project_root_path_audit": path_audit,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(report_staging / "project_root_rebase_report.json", report_payload)
    (report_staging / "README.md").write_text(
        "# Exact-analyzer project-root rebase\n\n"
        "Verdict: `COMPLETE_ADMIN_PATH_REBASE`.\n\n"
        f"- Input manifest unchanged: `{source_before['manifest_sha256']}`.\n"
        f"- Output compatibility manifest: `{compatibility['manifest_sha256']}`.\n"
        f"- Exact analyzer SHA: `{EXPECTED_ANALYZER_SHA256}`.\n"
        f"- Amended contract SHA: `{contract_sha}`.\n"
        f"- Analyzer-computed PROJECT: `{final_analyzer.resolve(strict=True).parents[1]}`.\n"
        "- Changed payloads: three administrative JSON files; 17/20 payloads byte-identical.\n"
        "- No scientific statistic was parsed or computed by this preparer.\n",
        encoding="utf-8",
    )
    write_json(report_staging / "artifact_manifest.json", report_manifest(report_staging))
    os.replace(report_staging, report)
    print(
        "stage=complete "
        f"elapsed={time.monotonic() - started:.2f}s "
        f"output_manifest={compatibility['manifest_sha256']} contract_sha={contract_sha} "
        f"report={report / 'project_root_rebase_report.json'}"
    )


if __name__ == "__main__":
    main()
