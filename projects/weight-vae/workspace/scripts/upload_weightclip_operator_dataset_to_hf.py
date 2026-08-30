#!/usr/bin/env python3
"""Upload the exact active WeightCLIP operator banks to a private HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    WORKSPACE_ROOT
    / "conf"
    / "weightclip_benchmark"
    / "operator_dataset_remote_hf.json"
)
DEFAULT_CARD = WORKSPACE_ROOT / "docs" / "weightclip_operator_dataset_hf.md"
DEFAULT_SOURCE_ROOT = Path(
    "/home/coder/project/projects/shared/storage/data/"
    "weightclip_resnet18slim_zoo/operator_dataset"
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


def _validate_source(root: Path, contract: dict[str, Any]) -> None:
    source = contract["source_contract"]
    small_files = {
        source["pair_manifest"]: source["pair_manifest_sha256"],
        source["coverage"]: source["coverage_sha256"],
        source["permutation_jsonl"]: source["permutation_jsonl_sha256"],
        source["permutation_parquet"]: source["permutation_parquet_sha256"],
        f"{contract['banks']['weight']['path']}/manifest.json": contract["banks"][
            "weight"
        ]["manifest_sha256"],
        f"{contract['banks']['context']['path']}/manifest.json": contract["banks"][
            "context"
        ]["manifest_sha256"],
    }
    for relative, expected_hash in small_files.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing source file: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: {actual_hash} != {expected_hash}"
            )

    for bank_name in ("weight", "context"):
        bank = contract["banks"][bank_name]
        count, byte_count = _tree_stats(root / bank["path"])
        if (count, byte_count) != (bank["files"], bank["bytes"]):
            raise RuntimeError(
                f"{bank_name} bank inventory mismatch: "
                f"{count} files/{byte_count} bytes != "
                f"{bank['files']} files/{bank['bytes']} bytes"
            )


def _expected_payload(root: Path, contract: dict[str, Any]) -> dict[str, int]:
    paths: list[Path] = []
    for bank_name in ("weight", "context"):
        paths.extend(
            item
            for item in (root / contract["banks"][bank_name]["path"]).rglob("*")
            if item.is_file()
        )
    source = contract["source_contract"]
    paths.extend(
        root / relative
        for relative in (
            source["pair_manifest"],
            source["coverage"],
            source["permutation_jsonl"],
            source["permutation_parquet"],
        )
    )
    return {path.relative_to(root).as_posix(): path.stat().st_size for path in paths}


def _remote_file_sizes(api: HfApi, repo_id: str, revision: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
    ):
        size = getattr(item, "size", None)
        path = getattr(item, "path", None)
        if path is not None and size is not None:
            result[str(path)] = int(size)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--token", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    card_path = args.card.expanduser().resolve()
    contract = _load_contract(contract_path)
    repo_id = args.repo_id or contract["repo_id"]
    token = args.token or os.environ.get("HF_TOKEN") or True
    xet_cache = os.environ.get("HF_XET_CACHE")
    if not xet_cache:
        raise RuntimeError(
            "HF_XET_CACHE must point to a filesystem with enough free space; "
            "for this machine use /var/tmp/weightclip-hf-xet"
        )
    api = HfApi(token=token)

    print("[weightclip-upload] stage=validate-source")
    print(f"[weightclip-upload] repo={repo_id} private=true")
    print(f"[weightclip-upload] source_root={source_root}")
    print(f"[weightclip-upload] contract={contract_path}")
    print(f"[weightclip-upload] workers={args.num_workers}")
    print(f"[weightclip-upload] xet_cache={Path(xet_cache).expanduser().resolve()}")
    _validate_source(source_root, contract)
    expected = _expected_payload(source_root, contract)
    expected_bytes = sum(expected.values())
    if (len(expected), expected_bytes) != (
        contract["active_payload"]["files"],
        contract["active_payload"]["bytes"],
    ):
        raise RuntimeError("active payload total does not match the frozen contract")
    print(
        f"[weightclip-upload] payload_files={len(expected)} "
        f"payload_bytes={expected_bytes}"
    )
    if args.dry_run:
        print("[weightclip-upload] stage=dry-run-complete")
        return

    if not args.validate_only:
        print("[weightclip-upload] stage=create-private-repo")
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        if not info.private:
            raise RuntimeError(f"refusing to upload: {repo_id} is not private")

        print("[weightclip-upload] stage=upload-card-and-contract")
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add WeightCLIP operator dataset card",
        )
        api.upload_file(
            path_or_fileobj=contract_path,
            path_in_repo="dataset_contract.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add immutable dataset contract",
        )

        source = contract["source_contract"]
        allow_patterns = [
            f"{contract['banks']['weight']['path']}/**",
            f"{contract['banks']['context']['path']}/**",
            source["pair_manifest"],
            source["coverage"],
            source["permutation_jsonl"],
            source["permutation_parquet"],
        ]
        print("[weightclip-upload] stage=upload-active-payload")
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=source_root,
            repo_type="dataset",
            revision="main",
            private=True,
            allow_patterns=allow_patterns,
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=30,
        )

        payload_revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
        remote_contract = dict(contract)
        remote_contract["payload_revision"] = payload_revision
        remote_contract["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            json.dump(remote_contract, handle, indent=2, sort_keys=True)
            handle.write("\n")
            remote_contract_path = Path(handle.name)
        try:
            api.upload_file(
                path_or_fileobj=remote_contract_path,
                path_in_repo="dataset_contract.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Record completed payload revision",
            )
        finally:
            remote_contract_path.unlink(missing_ok=True)

    print("[weightclip-upload] stage=validate-remote")
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not info.private:
        raise RuntimeError(f"remote repository unexpectedly public: {repo_id}")
    remote = _remote_file_sizes(api, repo_id, info.sha)
    missing = sorted(path for path in expected if path not in remote)
    wrong_size = sorted(
        (path, expected[path], remote.get(path))
        for path in expected
        if path in remote and remote[path] != expected[path]
    )
    if missing or wrong_size:
        raise RuntimeError(
            f"remote payload mismatch: missing={missing[:10]} "
            f"wrong_size={wrong_size[:10]}"
        )

    source = contract["source_contract"]
    remote_small_files = {
        source["pair_manifest"]: source["pair_manifest_sha256"],
        source["coverage"]: source["coverage_sha256"],
        source["permutation_jsonl"]: source["permutation_jsonl_sha256"],
        source["permutation_parquet"]: source["permutation_parquet_sha256"],
        f"{contract['banks']['weight']['path']}/manifest.json": contract["banks"][
            "weight"
        ]["manifest_sha256"],
        f"{contract['banks']['context']['path']}/manifest.json": contract["banks"][
            "context"
        ]["manifest_sha256"],
    }
    for relative, expected_hash in remote_small_files.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=info.sha,
                filename=relative,
                token=token,
            )
        )
        actual_hash = _sha256(downloaded)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"remote SHA-256 mismatch for {relative}: "
                f"{actual_hash} != {expected_hash}"
            )

    remote_contract_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=info.sha,
            filename="dataset_contract.json",
            token=token,
        )
    )
    remote_contract = _load_contract(remote_contract_path)
    if remote_contract.get("payload_revision") is None:
        raise RuntimeError("remote contract does not record a payload revision")

    print("[weightclip-upload] stage=complete")
    print(f"[weightclip-upload] url=https://huggingface.co/datasets/{repo_id}")
    print(f"[weightclip-upload] revision={info.sha}")
    print(f"[weightclip-upload] payload_revision={remote_contract['payload_revision']}")
    print(f"[weightclip-upload] validated_files={len(expected)}")
    print(f"[weightclip-upload] validated_bytes={expected_bytes}")


if __name__ == "__main__":
    main()
