#!/usr/bin/env python3
"""Resolve executable Stage-D/G configs from immutable zoo dataset manifests."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import yaml

from big_vae.weightclip_benchmark.contract import DEFAULT_CONTRACT
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef
from big_vae.weightclip_benchmark.manifests import write_json_immutable


def _write_yaml_immutable(path: Path, payload: Mapping[str, Any]) -> ArtifactRef:
    encoded = yaml.safe_dump(dict(payload), sort_keys=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable resolved config differs: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
    return ArtifactRef.create(path)


def _indices(dataset: str, size: int, count: int, namespace: str) -> list[int]:
    if size < count:
        raise ValueError(f"{dataset}: train split has {size} examples, needs {count}")
    seed = int.from_bytes(hashlib.sha256(f"{namespace}:{dataset}".encode()).digest()[:8], "big")
    return random.Random(seed).sample(range(size), count)


def resolve_configs(*, zoo_config: Path, evaluation_config: Path, output_dir: Path) -> dict[str, Any]:
    zoo = yaml.safe_load(zoo_config.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(evaluation_config.read_text(encoding="utf-8"))
    source_manifest_path = Path(zoo["paths"]["dataset_pt_root"]) / "manifest.json"
    ood_manifest_path = Path(zoo["paths"]["ood_dataset_pt_root"]) / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    ood_manifest = json.loads(ood_manifest_path.read_text(encoding="utf-8"))
    if set(ood_manifest["datasets"]) != set(DEFAULT_CONTRACT.ood_datasets):
        raise ValueError("OOD manifest ids disagree with the frozen hyphenated contract")
    source_registry: dict[str, Any] = {}
    for dataset, row in sorted(source_manifest["datasets"].items()):
        reference = ArtifactRef.create(row["dataset_pt"])
        if reference.sha256 != row["dataset_pt_sha256"]:
            raise ValueError(f"source dataset manifest hash mismatch: {dataset}")
        source_registry[dataset] = {"dataset_pt": asdict(reference), "num_classes": int(row["classes"])}
    ood_registry: dict[str, Any] = {}
    ood_bundle_registry: dict[str, Any] = {}
    for dataset, row in sorted(ood_manifest["datasets"].items()):
        reference = ArtifactRef.create(row["dataset_pt"])
        if reference.sha256 != row["dataset_pt_sha256"]:
            raise ValueError(f"OOD dataset manifest hash mismatch: {dataset}")
        train_size = int(row["split_sizes"]["trainset"])
        prompt_sets = [_indices(dataset, train_size, 10, f"weightclip-prompt-{index:03d}") for index in range(100)]
        if len({tuple(item) for item in prompt_sets}) != 100:
            raise AssertionError("prompt-set derivation unexpectedly collided")
        ood_registry[dataset] = {"dataset_pt": asdict(reference), "num_classes": int(row["classes"])}
        ood_bundle_registry[dataset] = {
            "dataset_pt": asdict(reference),
            "num_classes": int(row["classes"]),
            "context_indices": _indices(dataset, train_size, min(512, train_size), "activation-context-v1"),
            "prompt_indices": prompt_sets[0],
            "prompt_candidate_indices": prompt_sets,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation["datasets"] = ood_registry
    evaluation_path = output_dir / "evaluation.final_three_seed.resolved.yaml"
    evaluation_ref = _write_yaml_immutable(evaluation_path, evaluation)
    official = evaluation["official_weightclip"]
    ood_config = {
        "device": "cuda",
        "width_mult": 0.5,
        "template_seed": 0,
        "output_dir": str((output_dir.parent / "ood_conditioning_bundles").resolve()),
        "official_weightclip": {
            "repo_path": official["repo_path"], "cache_dir": official["cache_dir"],
            "checkpoint_path": official["checkpoint_path"],
            "dataset_encoder_path": official["dataset_encoder_path"],
        },
        "datasets": ood_bundle_registry,
    }
    ood_path = output_dir / "ood_conditioning.resolved.yaml"
    ood_ref = _write_yaml_immutable(ood_path, ood_config)
    anchor_config = {
        "device": "cuda:0",
        "output_root": str((output_dir.parent / "ood_anchors").resolve()),
        "evaluation_seeds": [int(seed) for seed in evaluation["evaluation"]["evaluation_seeds"]],
        "seed_offset": 10000,
        "precommitted_lr": 0.1,
        "protocol": dict(zoo["protocol"]),
        "datasets": ood_registry,
    }
    anchor_path = output_dir / "ood_anchors.final_three_seed.resolved.yaml"
    anchor_ref = _write_yaml_immutable(anchor_path, anchor_config)
    index = {
        "schema_version": 1,
        "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
        "source_manifest": asdict(ArtifactRef.create(source_manifest_path)),
        "ood_manifest": asdict(ArtifactRef.create(ood_manifest_path)),
        "evaluation_config": asdict(evaluation_ref),
        "ood_conditioning_config": asdict(ood_ref),
        "ood_anchor_config": asdict(anchor_ref),
        "source_datasets": source_registry,
        "launch_order": [
            "seal approved AE", "build source task-fit bundles (ours and WeightCLIP)",
            "fit z_task records", "train/seal four flows", "build OOD conditioning bundles",
            "build common-zoo full-window bank", "produce ridge/memory/nearest-code artifacts",
            "train target anchors", "build candidate manifest", "evaluate",
        ],
    }
    index_path = output_dir / "stage_dg.final_three_seed.index.json"
    write_json_immutable(index_path, index)
    print(f"[stage-dg-config:done] evaluation={evaluation_path} ood={ood_path} index={index_path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo-config", type=Path, default=Path("conf/weightclip_benchmark/zoo.yaml"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("conf/weightclip_benchmark/evaluation.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    resolve_configs(
        zoo_config=args.zoo_config.resolve(), evaluation_config=args.evaluation_config.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
