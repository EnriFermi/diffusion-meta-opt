#!/usr/bin/env python3
"""Download and relocate the private WeightCLIP production operator dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    WORKSPACE_ROOT
    / "conf"
    / "weightclip_benchmark"
    / "operator_dataset_remote_hf.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "weightclip_operator_dataset_remote_v1":
        raise RuntimeError(f"unsupported remote contract: {path}")
    return contract


def _resolve_pair_manifest(snapshot: Path, contract: dict[str, Any]) -> tuple[Path, str]:
    source_contract = contract["source_contract"]
    source_path = snapshot / source_contract["pair_manifest"]
    pair = json.loads(source_path.read_text(encoding="utf-8"))

    weight_path = (snapshot / contract["banks"]["weight"]["path"]).resolve()
    context_path = (snapshot / contract["banks"]["context"]["path"]).resolve()
    coverage_path = (snapshot / source_contract["coverage"]).resolve()
    permutation_jsonl = (snapshot / source_contract["permutation_jsonl"]).resolve()
    permutation_parquet = (snapshot / source_contract["permutation_parquet"]).resolve()

    pair["weight_tile_bank"] = str(weight_path)
    pair["context_bank"] = str(context_path)
    pair["coverage_path"] = str(coverage_path)
    original_permutation_hashes = pair["permutation_files"]
    pair["permutation_files"] = {
        str(permutation_jsonl): original_permutation_hashes[
            next(key for key in original_permutation_hashes if key.endswith(".jsonl"))
        ],
        str(permutation_parquet): original_permutation_hashes[
            next(key for key in original_permutation_hashes if key.endswith(".parquet"))
        ],
    }

    resolved_path = snapshot / "operator_dataset-resolved-c478efe716506765.json"
    payload = json.dumps(pair, indent=2, sort_keys=True) + "\n"
    resolved_path.write_text(payload, encoding="utf-8")
    return resolved_path, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    contract = _load_contract(args.contract.resolve())
    repo_id = args.repo_id or contract["repo_id"]
    revision = args.revision or contract["revision"]
    token = args.token or os.environ.get("HF_TOKEN") or True
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("[weightclip-download] stage=resolve")
    print(f"[weightclip-download] repo={repo_id} revision={revision}")
    print(f"[weightclip-download] output_root={output_root}")
    print(f"[weightclip-download] expected_bytes={contract['active_payload']['bytes']}")
    print("[weightclip-download] stage=download")
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=contract["repo_type"],
            revision=revision,
            local_dir=output_root,
            token=token,
            max_workers=args.max_workers,
        )
    ).resolve()

    print("[weightclip-download] stage=validate")
    source_contract = contract["source_contract"]
    small_files = {
        source_contract["pair_manifest"]: source_contract["pair_manifest_sha256"],
        source_contract["coverage"]: source_contract["coverage_sha256"],
        source_contract["permutation_jsonl"]: source_contract["permutation_jsonl_sha256"],
        source_contract["permutation_parquet"]: source_contract[
            "permutation_parquet_sha256"
        ],
        f"{contract['banks']['weight']['path']}/manifest.json": contract["banks"][
            "weight"
        ]["manifest_sha256"],
        f"{contract['banks']['context']['path']}/manifest.json": contract["banks"][
            "context"
        ]["manifest_sha256"],
    }
    for relative, expected_hash in small_files.items():
        path = snapshot_path / relative
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: {actual_hash} != {expected_hash}"
            )

    for bank_name in ("weight", "context"):
        bank = contract["banks"][bank_name]
        count, byte_count = _tree_stats(snapshot_path / bank["path"])
        if (count, byte_count) != (bank["files"], bank["bytes"]):
            raise RuntimeError(
                f"{bank_name} bank inventory mismatch: "
                f"{count} files/{byte_count} bytes != "
                f"{bank['files']} files/{bank['bytes']} bytes"
            )
        print(
            f"[weightclip-download] bank={bank_name} files={count} bytes={byte_count}"
        )

    resolved_path, resolved_sha256 = _resolve_pair_manifest(snapshot_path, contract)
    print("[weightclip-download] stage=complete")
    print(f"[weightclip-download] snapshot={snapshot_path}")
    print(f"[weightclip-download] pair_manifest={resolved_path}")
    print(f"[weightclip-download] pair_manifest_sha256={resolved_sha256}")


if __name__ == "__main__":
    main()
