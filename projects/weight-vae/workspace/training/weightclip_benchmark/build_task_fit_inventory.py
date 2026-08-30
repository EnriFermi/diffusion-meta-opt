#!/usr/bin/env python3
"""Seal exact expected/observed grouped z_task record inventory for both codecs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
from pathlib import Path

from big_vae.flow_matching.dataset import GroupedLatentRecord
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef, load_task_fit_bundle
from big_vae.weightclip_benchmark.manifests import write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-bundle-index", type=Path, required=True)
    parser.add_argument("--weightclip-bundle-index", type=Path, required=True)
    parser.add_argument("--ours-record-glob", required=True)
    parser.add_argument("--weightclip-record-glob", required=True)
    parser.add_argument("--parity-seal", type=Path, required=True)
    parser.add_argument("--ours-completion", type=Path, required=True)
    parser.add_argument("--weightclip-completion", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parity_ref = ArtifactRef.create(args.parity_seal)
    expected: dict[str, list[str]] = {}
    observed_refs: dict[str, list[dict[str, object]]] = {}
    validation_refs: dict[str, list[dict[str, object]]] = {}
    validation_bundle_refs: dict[str, list[dict[str, object]]] = {}
    validation_ids: dict[str, list[str]] = {}
    for codec, index_path, pattern, completion_path in (
        ("ours", args.ours_bundle_index, args.ours_record_glob, args.ours_completion),
        ("weightclip", args.weightclip_bundle_index, args.weightclip_record_glob, args.weightclip_completion),
    ):
        completion_ref = ArtifactRef.create(completion_path)
        completion = json.loads(Path(completion_ref.path).read_text(encoding="utf-8"))
        if (
            completion.get("schema_version") != 1
            or completion.get("kind") != "task_fit_completion"
            or completion.get("codec") != codec
        ):
            raise ValueError(f"invalid {codec} task-fit completion artifact")
        if ArtifactRef.from_mapping(completion["bundle_index"]) != ArtifactRef.create(index_path):
            raise ValueError(f"{codec} completion bundle index drift")
        if ArtifactRef.from_mapping(completion["parity_seal"]) != parity_ref:
            raise ValueError(f"{codec} completion parity seal drift")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        bundles = [(row["bundle"], load_task_fit_bundle(row["bundle"]["path"])) for row in index["bundles"]]
        expected[codec] = sorted(bundle["identity"]["group_id"] for _, bundle in bundles)
        record_paths = sorted(glob.glob(pattern))
        completion_records = [ArtifactRef.from_mapping(row) for row in completion["records"]]
        completed_paths = sorted(
            str(Path(ref.verify(f"{codec} completed record")).resolve()) for ref in completion_records
        )
        if completed_paths != [str(Path(path).resolve()) for path in record_paths]:
            raise ValueError(f"{codec} completion record inventory disagrees with record glob")
        records = [GroupedLatentRecord.load(path) for path in record_paths]
        actual = sorted(record.group_id for record in records)
        if actual != expected[codec]:
            raise ValueError(f"{codec} task-fit inventory mismatch: expected={len(expected[codec])} actual={len(actual)}")
        for record in records:
            if record.provenance.get("fit_protocol_provenance", {}).get("parity_seal", {}).get("sha256") != parity_ref.sha256:
                raise ValueError(f"{codec} record {record.group_id} is not bound to parity seal")
        observed_refs[codec] = [asdict(ArtifactRef.create(path)) for path in record_paths]
        validation_ids[codec] = sorted(record.group_id for record in records if record.split == "validation")
        validation_refs[codec] = [
            asdict(ArtifactRef.create(path))
            for path, record in zip(record_paths, records, strict=True)
            if record.split == "validation"
        ]
        validation_bundle_refs[codec] = [
            dict(reference)
            for reference, bundle in bundles
            if str(bundle["identity"]["group_id"]) in set(validation_ids[codec])
        ]
    payload = {
        "schema_version": 1,
        "kind": "matched_task_fit_inventory",
        "parity_seal": asdict(parity_ref),
        "group_ids": expected["ours"],
        "codec_group_ids": expected,
        "records": observed_refs,
    }
    if expected["ours"] != expected["weightclip"]:
        raise ValueError("ours/WeightCLIP expected task-fit group ids differ")
    if validation_ids["ours"] != validation_ids["weightclip"]:
        raise ValueError("ours/WeightCLIP validation task-fit group ids differ")
    by_dataset: dict[str, int] = {}
    for group_id in validation_ids["ours"]:
        dataset = group_id.split(":", 1)[0]
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
    if len(by_dataset) != 10 or set(by_dataset.values()) != {7}:
        raise ValueError(f"validation inventory must be exactly 10 datasets x 7 lineages, got {by_dataset}")
    validation_payload = {
        "schema_version": 1,
        "kind": "matched_e4_validation_inventory",
        "parity_seal": asdict(parity_ref),
        "group_ids": validation_ids["ours"],
        "expected_dataset_count": 10,
        "expected_lineages_per_dataset": 7,
        "records": validation_refs,
        "task_bundles": validation_bundle_refs,
    }
    write_json_immutable(args.validation_output, validation_payload)
    payload["validation_inventory"] = asdict(ArtifactRef.create(args.validation_output))
    write_json_immutable(args.output, payload)
    print(f"[task-fit-inventory:done] groups={len(expected['ours'])} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
