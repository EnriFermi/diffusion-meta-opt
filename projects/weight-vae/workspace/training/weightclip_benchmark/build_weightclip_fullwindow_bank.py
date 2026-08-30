#!/usr/bin/env python3
"""Materialize hash-bound common-zoo X/Y for the 25-window WeightCLIP extension."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import yaml

from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef, load_task_fit_bundle
from training.weightclip_benchmark.build_weightclip_multiwindow_codes import _commit_torch_artifact


def build_fullwindow_bank(raw: dict[str, object]) -> dict[str, object]:
    if "source_bundle_index" in raw:
        index_ref = ArtifactRef.create(raw["source_bundle_index"])
        index = json.loads(Path(index_ref.path).read_text(encoding="utf-8"))
        references = [ArtifactRef.from_mapping(row["bundle"]) for row in index["bundles"]]
    else:
        index_ref = None
        references = [ArtifactRef.create(path) for path in raw["source_task_bundles"]]
    pairs = [
        (reference, load_task_fit_bundle(reference.path, expected_kind="weightclip_task_fit"))
        for reference in references
    ]
    observed_splits = {str(bundle["identity"]["split"]) for _, bundle in pairs}
    if observed_splits - {"train", "validation", "internal_test"}:
        raise ValueError(f"full-window bank bundle index has unknown splits: {sorted(observed_splits)}")
    train_pairs = [(reference, bundle) for reference, bundle in pairs if bundle["identity"]["split"] == "train"]
    if not train_pairs:
        raise ValueError("full-window bank requires source task bundles")
    references = [reference for reference, _ in train_pairs]
    bundles = [bundle for _, bundle in train_pairs]
    train_by_dataset: dict[str, int] = {}
    for bundle in bundles:
        dataset_id = str(bundle["identity"]["dataset_id"])
        train_by_dataset[dataset_id] = train_by_dataset.get(dataset_id, 0) + 1
    if len(train_by_dataset) != 10 or set(train_by_dataset.values()) != {35}:
        raise ValueError(f"full-window bank requires frozen 10 datasets x 35 train lineages, got {train_by_dataset}")
    fingerprints = {bundle["provenance"]["codec_fingerprint"] for bundle in bundles}
    if len(fingerprints) != 1:
        raise ValueError("source task bundles mix WeightCLIP codec fingerprints")
    shapes = {tuple(bundle["z_enc"].shape) for bundle in bundles}
    if len(shapes) != 1:
        raise ValueError(f"source full-window code geometry is not common: {sorted(shapes)}")
    shape = next(iter(shapes))
    if shape[1] != 512:
        raise ValueError(f"official native window size must remain 512, got {shape}")
    X = torch.stack([bundle["dataset_embedding"] for bundle in bundles]).float()
    grouped = torch.stack([bundle["z_enc"] for bundle in bundles]).float()
    Y = grouped.reshape(len(grouped), grouped.shape[1] * grouped.shape[2], grouped.shape[3])
    output = Path(raw["output_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "weightclip_commonzoo_fullwindow_code_bank",
        "codec": "weightclip",
        "codec_fingerprint": next(iter(fingerprints)),
        "X_dataset_embeddings": X,
        "Y_fullwindow_codes": Y,
        "window_count": int(grouped.shape[1]),
        "window_size": int(grouped.shape[2]),
        "latent_dim": int(grouped.shape[3]),
        "source_bundle_refs": [asdict(reference) for reference in references],
        "source_bundle_index": None if index_ref is None else asdict(index_ref),
        "identities": [bundle["identity"] for bundle in bundles],
        "prompt_manifests": [bundle["dataset_prompt_provenance"] for bundle in bundles],
        "provenance": {
            "substrate_label": "WeightCLIP codec + common-zoo full-window code bank",
            "released_mapper_or_bank": False,
            "all_windows_concatenated_before_prior_fit": True,
        },
    }
    _commit_torch_artifact(
        output,
        payload,
        {"kind": payload["kind"], "ledger": payload["provenance"]},
    )
    print(f"[weightclip-fullwindow-bank] models={len(bundles)} shape_X={tuple(X.shape)} shape_Y={tuple(Y.shape)} output={output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build_fullwindow_bank(yaml.safe_load(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
