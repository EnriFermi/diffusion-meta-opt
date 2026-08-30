#!/usr/bin/env python3
"""Train target-data ResNet18Slim anchors for explicitly oracle-labeled arms."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import yaml

from big_vae.weightclip_benchmark.contract import DEFAULT_CONTRACT
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef, load_train_dataset_pt
from big_vae.weightclip_benchmark.manifests import write_json_atomic, write_records_immutable
from big_vae.weightclip_benchmark.metadata import ZooProtocol
from big_vae.weightclip_benchmark.resnet18slim import ResNet18Slim
from training.weightclip_benchmark.build_zoo import atomic_torch_save, train_epoch


def validate_cached_anchor(
    *,
    complete_path: Path,
    checkpoint_path: Path,
    expected_contract_sha256: str,
) -> ArtifactRef:
    """Fail closed on any stale anchor config, dataset, protocol, or bytes."""

    cached = json.loads(complete_path.read_text(encoding="utf-8"))
    if cached.get("anchor_contract_sha256") != expected_contract_sha256:
        raise ValueError(f"stale OOD anchor completion contract: {complete_path}")
    checkpoint_ref = ArtifactRef.create(checkpoint_path)
    if cached.get("checkpoint") != asdict(checkpoint_ref):
        raise ValueError(f"OOD anchor checkpoint hash disagrees with completion: {checkpoint_path}")
    if cached.get("ood_validation_access") is not False or cached.get("ood_test_access") is not False:
        raise ValueError("cached OOD anchor does not prove train-only access")
    return checkpoint_ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(raw["output_root"]).resolve()
    protocol_values = dict(raw["protocol"])
    for name in ("primary_checkpoint_indices", "lineage_split", "lr_candidates"):
        protocol_values[name] = tuple(protocol_values[name])
    protocol = ZooProtocol(**protocol_values)
    protocol.validate()
    if protocol.epochs != 45 or protocol.scheduler_epochs != 50:
        raise ValueError("OOD anchors must use the frozen 45/50 WeightCLIP training schedule")
    rows: list[dict[str, Any]] = []
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for dataset, dataset_row in sorted(raw["datasets"].items()):
        if dataset not in DEFAULT_CONTRACT.ood_datasets:
            raise ValueError(f"non-contract OOD dataset {dataset}")
        dataset_ref = ArtifactRef.from_mapping(dataset_row["dataset_pt"])
        dataset_ref.verify(f"OOD anchor dataset {dataset}")
        chosen_lr = float(dataset_row.get("precommitted_lr", raw["precommitted_lr"]))
        if chosen_lr not in protocol.lr_candidates:
            raise ValueError(f"OOD anchor LR {chosen_lr} is outside the frozen source LR grid")
        for evaluation_seed in raw["evaluation_seeds"]:
            seed = int(evaluation_seed) + int(raw.get("seed_offset", 10000))
            lineage_dir = output_root / "anchors" / dataset / f"lineage-{seed:04d}"
            complete = lineage_dir / "complete.json"
            checkpoint_path = lineage_dir / "epochs" / "epoch-045.pt"
            anchor_contract = {
                "dataset": dataset,
                "dataset_sha256": dataset_ref.sha256,
                "seed": seed,
                "evaluation_seed": int(evaluation_seed),
                "lr": chosen_lr,
                "protocol_sha256": protocol_sha256,
                "lr_policy": "precommitted_before_ood_unseal",
            }
            anchor_contract_sha256 = hashlib.sha256(
                json.dumps(anchor_contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if complete.exists():
                checkpoint_ref = validate_cached_anchor(
                    complete_path=complete,
                    checkpoint_path=checkpoint_path,
                    expected_contract_sha256=anchor_contract_sha256,
                )
            if not complete.exists():
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                trainset = load_train_dataset_pt(dataset_ref)["trainset"]
                # OOD seal: architecture/classes/LR/training see trainset only.
                classes = int(torch.unique(trainset.targets).numel())
                device = str(raw.get("device", "cuda:0"))
                model = ResNet18Slim(
                    3, classes, "relu", protocol.dropout, protocol.init_type, protocol.width_mult
                ).to(device)
                optimizer = SGD(
                    model.parameters(), lr=chosen_lr, momentum=protocol.momentum,
                    weight_decay=protocol.weight_decay,
                )
                scheduler = OneCycleLR(
                    optimizer, max_lr=chosen_lr, epochs=protocol.scheduler_epochs,
                    steps_per_epoch=math.ceil(len(trainset) / protocol.batch_size),
                )
                lineage_dir.mkdir(parents=True, exist_ok=True)
                last_loss = last_accuracy = 0.0
                for _epoch in range(protocol.epochs):
                    last_loss, last_accuracy = train_epoch(
                        model, trainset, optimizer, scheduler, protocol.batch_size, device
                    )
                atomic_torch_save(
                    {key: value.detach().cpu().float() for key, value in model.state_dict().items()}, checkpoint_path
                )
                checkpoint_ref = ArtifactRef.create(checkpoint_path)
                write_json_atomic(
                    complete,
                    {
                        **anchor_contract,
                        "anchor_contract_sha256": anchor_contract_sha256,
                        "checkpoint": asdict(checkpoint_ref),
                        "epoch_one_based": protocol.epochs,
                        "train_loss": last_loss, "train_accuracy": last_accuracy,
                        "ood_validation_access": False, "ood_test_access": False,
                        "lr_policy": "precommitted_before_ood_unseal",
                    },
                )
            checkpoint_ref = ArtifactRef.create(checkpoint_path)
            rows.append(
                {
                    "schema_version": 1,
                    "dataset": dataset,
                    "evaluation_seed": int(evaluation_seed),
                    "training_seed": seed,
                    "checkpoint": asdict(checkpoint_ref),
                    "dataset_pt": asdict(dataset_ref),
                    "chosen_lr": chosen_lr,
                    "checkpoint_epoch_one_based": 45,
                    "head_policy": "preserve_source",
                    "batchnorm_policy": "preserve_source",
                    "evaluation_method": "anchor_untouched",
                    "target_weight_access": "oracle_anchor_only",
                    "lr_policy": "precommitted_before_ood_unseal",
                    "anchor_contract_sha256": anchor_contract_sha256,
                    "protocol_sha256": protocol_sha256,
                    "ood_validation_access": False,
                    "ood_test_access": False,
                    "complete_record": json.loads(complete.read_text(encoding="utf-8")),
                }
            )
    paths = write_records_immutable(output_root / "anchor_manifest", rows)
    print(f"[ood-anchors:done] anchors={len(rows)} manifests={paths}")


if __name__ == "__main__":
    main()
