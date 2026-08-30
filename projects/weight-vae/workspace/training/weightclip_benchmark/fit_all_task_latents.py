#!/usr/bin/env python3
"""Execute every immutable source bundle through its matched codec fit CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from big_vae.flow_matching.dataset import GroupedLatentRecord
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef, load_task_fit_bundle
from big_vae.weightclip_benchmark.manifests import write_json_immutable


def validate_cached_task_fit(
    *,
    record_path: Path,
    bundle: dict[str, object],
    codec: str,
    parity_ref: ArtifactRef,
) -> bool:
    metrics_path = record_path.with_suffix(".metrics.json")
    if not record_path.exists() and not metrics_path.exists():
        return False
    if not record_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"partial task-fit cache exists for {record_path}")
    record = GroupedLatentRecord.load(record_path)
    identity = bundle["identity"]
    for field in ("group_id", "dataset_id", "lineage_id", "checkpoint_id", "split", "codec"):
        if str(getattr(record, field)) != str(identity[field]):
            raise ValueError(f"cached task-fit {field} drift for {record_path}")
    if record.codec != codec or record.codec_fingerprint != bundle["provenance"]["codec_fingerprint"]:
        raise ValueError(f"cached task-fit codec binding drift for {record_path}")
    bound_parity = record.provenance.get("fit_protocol_provenance", {}).get("parity_seal", {})
    if bound_parity.get("sha256") != parity_ref.sha256:
        raise ValueError(f"cached task-fit parity drift for {record_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("fit_status") != "success" or metrics.get("improved") is not True:
        raise ValueError(f"cached task-fit metrics are incomplete for {record_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-index", type=Path, required=True)
    parser.add_argument("--fit-config", type=Path, required=True)
    parser.add_argument("--parity-seal", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.bundle_index.read_text(encoding="utf-8"))
    codec = str(index["codec"])
    fit_config = yaml.safe_load(args.fit_config.read_text(encoding="utf-8"))
    output_dir = Path(fit_config["fit"]["output_dir"]).resolve()
    parity_ref = ArtifactRef.create(args.parity_seal)
    module = (
        "training.weightclip_benchmark.fit_task_latents"
        if codec == "ours"
        else "training.weightclip_benchmark.fit_weightclip_task_latents"
    )
    bundle_refs: list[ArtifactRef] = []
    group_ids: list[str] = []
    for position, row in enumerate(index["bundles"], 1):
        bundle = ArtifactRef.from_mapping(row["bundle"])
        bundle.verify(f"{codec} task bundle")
        loaded_bundle = load_task_fit_bundle(bundle.path)
        if loaded_bundle["identity"]["codec"] != codec:
            raise ValueError("task bundle codec disagrees with bundle index")
        bundle_refs.append(bundle)
        group_id = str(loaded_bundle["identity"]["group_id"])
        group_ids.append(group_id)
        record_path = output_dir / f"{group_id}.pt"
        if validate_cached_task_fit(
            record_path=record_path,
            bundle=loaded_bundle,
            codec=codec,
            parity_ref=parity_ref,
        ):
            print(f"[fit-all:cache] codec={codec} item={position}/{len(index['bundles'])} hit={record_path}")
            continue
        command = [
            sys.executable, "-m", module,
            "--config", str(args.fit_config.resolve()),
            "--bundle", bundle.path,
            "--parity-seal", str(args.parity_seal.resolve()),
        ]
        print(f"[fit-all] codec={codec} item={position}/{len(index['bundles'])} bundle={bundle.path}")
        subprocess.run(command, check=True)
    record_refs: list[dict[str, object]] = []
    for group_id in group_ids:
        record_path = output_dir / f"{group_id}.pt"
        record = GroupedLatentRecord.load(record_path)
        if record.group_id != group_id or record.codec != codec:
            raise ValueError(f"task-fit record identity mismatch: {record_path}")
        bound_parity = record.provenance.get("fit_protocol_provenance", {}).get("parity_seal", {})
        if bound_parity.get("sha256") != parity_ref.sha256:
            raise ValueError(f"task-fit record is not bound to parity seal: {record_path}")
        reference = ArtifactRef.create(record_path)
        record_refs.append({"path": reference.path, "sha256": reference.sha256, "bytes": reference.bytes})
    bundle_index_ref = ArtifactRef.create(args.bundle_index)
    fit_config_ref = ArtifactRef.create(args.fit_config)
    write_json_immutable(
        args.completion_output,
        {
            "schema_version": 1,
            "kind": "task_fit_completion",
            "codec": codec,
            "bundle_index": {
                "path": bundle_index_ref.path,
                "sha256": bundle_index_ref.sha256,
                "bytes": bundle_index_ref.bytes,
            },
            "fit_config": {
                "path": fit_config_ref.path,
                "sha256": fit_config_ref.sha256,
                "bytes": fit_config_ref.bytes,
            },
            "parity_seal": {
                "path": parity_ref.path,
                "sha256": parity_ref.sha256,
                "bytes": parity_ref.bytes,
            },
            "group_ids": group_ids,
            "bundles": [
                {"path": ref.path, "sha256": ref.sha256, "bytes": ref.bytes} for ref in bundle_refs
            ],
            "records": record_refs,
        },
    )
    print(
        f"[fit-all:complete] codec={codec} records={len(record_refs)} "
        f"output={args.completion_output.resolve()}"
    )


if __name__ == "__main__":
    main()
