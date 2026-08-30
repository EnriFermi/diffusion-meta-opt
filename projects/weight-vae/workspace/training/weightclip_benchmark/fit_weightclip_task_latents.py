#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    load_checkpoint_state,
    load_dataset_pt,
    load_task_fit_bundle,
    assert_decoder_provenance_matches_bundle,
)
from big_vae.weightclip_benchmark.weightclip_full import WeightCLIPTaskFitConfig, fit_weightclip_multiwindow_task_latent
from big_vae.weightclip_benchmark.paired_sampling import ResettableEpochSampler


def _batches(images: torch.Tensor, labels: torch.Tensor, size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(images[start : start + size], labels[start : start + size]) for start in range(0, len(images), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a grouped official WeightCLIP multi-window z_task")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--parity-seal", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    parity_ref = ArtifactRef.create(args.parity_seal)
    parity = json.loads(Path(parity_ref.path).read_text(encoding="utf-8"))
    if ArtifactRef.from_mapping(parity["weightclip_config"]).sha256 != ArtifactRef.create(args.config).sha256:
        raise ValueError("WeightCLIP fit config disagrees with task-fit parity seal")
    bundle = load_task_fit_bundle(args.bundle, expected_kind="weightclip_task_fit")
    dataset = load_dataset_pt(ArtifactRef.from_mapping(bundle["dataset_pt"]))
    source_ref = ArtifactRef.from_mapping(bundle["source_checkpoint"])
    source_state = load_checkpoint_state(source_ref)
    decoder_cfg = raw["decoder"]
    module_name, attribute = decoder_cfg["factory"].rsplit(".", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    kwargs = dict(decoder_cfg.get("kwargs", {}))
    kwargs.update(
        checkpoint_path=decoder_cfg["checkpoint"],
        anchor_state=source_state,
        anchor_state_sha256=source_ref.sha256,
    )
    decoder = factory(**kwargs)
    bundled_decoder = bundle["provenance"]["decoder"]
    assert_decoder_provenance_matches_bundle(decoder.provenance, bundled_decoder, codec="weightclip")
    configured_checkpoint = ArtifactRef.create(decoder_cfg["checkpoint"])
    bundled_checkpoint_sha = bundled_decoder.get("checkpoint_sha256") or bundled_decoder.get("official", {}).get("checkpoint", {}).get("sha256")
    if configured_checkpoint.sha256 != bundled_checkpoint_sha or decoder.provenance.get("checkpoint_sha256") != bundled_checkpoint_sha:
        raise ValueError("fit config/loaded WeightCLIP decoder disagrees with task bundle checkpoint hash")
    configured_encoder = ArtifactRef.create(decoder_cfg["kwargs"]["dataset_encoder_path"])
    bundled_encoder_sha = bundled_decoder.get("official", {}).get("dataset_encoder", {}).get("sha256")
    if bundled_encoder_sha is not None and configured_encoder.sha256 != bundled_encoder_sha:
        raise ValueError("fit config WeightCLIP dataset encoder disagrees with task bundle hash")
    batch_size = int(raw.get("task_batch_size", 128))
    fit_raw = dict(raw["fit"])
    configured_fit_batch = fit_raw.get("task_batch_size", batch_size)
    if int(configured_fit_batch) != batch_size:
        raise ValueError("top-level and fit task_batch_size disagree")
    fit_raw["task_batch_size"] = batch_size
    fit_config = WeightCLIPTaskFitConfig(**fit_raw)
    if float(raw["resnet"]["dropout"]) != fit_config.model_dropout:
        raise ValueError("WeightCLIP resnet/fit dropout configs disagree")
    train_dataset = TensorDataset(dataset["trainset"].data, dataset["trainset"].targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ResettableEpochSampler(len(train_dataset), fit_config.seed),
    )
    print(json.dumps({"stage": "weightclip_multiwindow_bundle", "path": str(args.bundle.resolve()), "cache": "hit"}), flush=True)
    fit_weightclip_multiwindow_task_latent(
        initial_code=bundle["z_enc"],
        decoder=decoder,
        source_state=source_state,
        train_batches=train_loader,
        validation_batches=_batches(dataset["valset"].data, dataset["valset"].targets, batch_size),
        context_provenance=bundle["context_provenance"],
        identity=bundle["identity"],
        dataset_embedding=bundle["dataset_embedding"],
        dataset_embedding_bank=bundle["dataset_embedding_bank"],
        dataset_embedding_bank_provenance=bundle["dataset_embedding_bank_provenance"],
        architecture_features=bundle["architecture_features"],
        config=fit_config,
        fit_protocol_provenance={"parity_seal": asdict(parity_ref), "matched_protocol": parity["matched_protocol"]},
    )


if __name__ == "__main__":
    main()
