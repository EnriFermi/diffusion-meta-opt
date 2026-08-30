#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, TensorDataset

from big_vae.flow_matching.dataset import TileIdentity
from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    load_checkpoint_state,
    load_dataset_pt,
    load_sealed_ours_decoder,
    load_task_fit_bundle,
    assert_decoder_provenance_matches_bundle,
)
from big_vae.weightclip_benchmark.resnet_functional import FunctionalResNetConfig
from big_vae.weightclip_benchmark.paired_sampling import ResettableEpochSampler
from big_vae.weightclip_benchmark.task_latent_fit import LayerTileLayout, TaskLatentFitConfig, fit_task_latents


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one grouped checkpoint's z_task codes through a frozen decoder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True, help="Prepared tensors/layout/source-state bundle")
    parser.add_argument("--parity-seal", type=Path, required=True)
    args = parser.parse_args()
    raw = _load_config(args.config)
    parity_ref = ArtifactRef.create(args.parity_seal)
    parity = json.loads(Path(parity_ref.path).read_text(encoding="utf-8"))
    if ArtifactRef.from_mapping(parity["ours_config"]).sha256 != ArtifactRef.create(args.config).sha256:
        raise ValueError("ours fit config disagrees with task-fit parity seal")
    bundle = load_task_fit_bundle(args.bundle, expected_kind="ours_task_fit")
    print(json.dumps({"stage": "load_bundle", "path": str(args.bundle.resolve()), "cache": "hit"}), flush=True)
    decoder_cfg = raw["decoder"]
    seal_ref = ArtifactRef.from_mapping(bundle["provenance"]["codec_seal"])
    decoder = load_sealed_ours_decoder(seal_ref, device=str(raw["fit"]["device"]))
    assert_decoder_provenance_matches_bundle(
        decoder.provenance,
        bundle["provenance"]["decoder"],
        codec="ours",
    )
    configured_checkpoint = ArtifactRef.create(decoder_cfg["checkpoint"])
    configured_model_config = ArtifactRef.create(decoder_cfg["kwargs"]["model_config_path"])
    if configured_checkpoint.sha256 != decoder.provenance["checkpoint_sha256"]:
        raise ValueError("fit config decoder checkpoint disagrees with task bundle codec seal")
    if configured_model_config.sha256 != decoder.provenance["model_config_sha256"]:
        raise ValueError("fit config decoder model config disagrees with task bundle codec seal")
    batch_size = int(raw.get("task_batch_size", 128))
    dataset = load_dataset_pt(ArtifactRef.from_mapping(bundle["dataset_pt"]))
    source_state = load_checkpoint_state(ArtifactRef.from_mapping(bundle["source_checkpoint"]))
    train_dataset = TensorDataset(dataset["trainset"].data, dataset["trainset"].targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ResettableEpochSampler(len(train_dataset), int(raw["fit"]["seed"])),
    )
    val_loader = DataLoader(
        TensorDataset(dataset["valset"].data, dataset["valset"].targets), batch_size=batch_size, shuffle=False
    )
    layouts = [LayerTileLayout(**item) for item in bundle["layouts"]]
    identities = [TileIdentity(**item) for item in bundle["tile_identities"]]
    fit_raw = dict(raw["fit"])
    configured_fit_batch = fit_raw.get("task_batch_size", batch_size)
    if int(configured_fit_batch) != batch_size:
        raise ValueError("top-level and fit task_batch_size disagree")
    fit_raw["task_batch_size"] = batch_size
    fit_config = TaskLatentFitConfig(**fit_raw)
    resnet_raw = dict(raw["resnet"])
    inferred_classes = int(source_state["fc.weight"].shape[0])
    inferred_channels_in = int(source_state["conv1.weight"].shape[1])
    inferred_width = float(source_state["conv1.weight"].shape[0]) / 64.0
    for key, inferred in (
        ("num_classes", inferred_classes),
        ("channels_in", inferred_channels_in),
        ("width_mult", inferred_width),
    ):
        configured = resnet_raw.get(key)
        if configured is not None and float(configured) != float(inferred):
            raise ValueError(f"configured resnet {key}={configured} disagrees with source checkpoint {inferred}")
        resnet_raw[key] = inferred
    model_config = FunctionalResNetConfig(**resnet_raw)
    _, result = fit_task_latents(
        initial_codes=bundle["z_enc"],
        decoder=decoder,
        layouts=layouts,
        architecture_features=bundle["architecture_features"],
        tile_mask=bundle.get("tile_mask"),
        source_state=source_state,
        model_config=model_config,
        train_loader=train_loader,
        validation_loader=val_loader,
        config=fit_config,
        identity=bundle["identity"],
        dataset_embedding=bundle["dataset_embedding"],
        dataset_embedding_bank=bundle["dataset_embedding_bank"],
        dataset_embedding_bank_provenance=bundle["dataset_embedding_bank_provenance"],
        tile_identities=identities,
        context_images=bundle["context_images"],
        context_provenance=bundle["context_provenance"],
        fit_protocol_provenance={"parity_seal": asdict(parity_ref), "matched_protocol": parity["matched_protocol"]},
    )
    if not result.improved:
        raise RuntimeError(
            f"task-fit gate failed: best validation loss {result.best_validation_loss:.6g} "
            f"did not beat z_enc {result.initial_validation_loss:.6g}"
        )


if __name__ == "__main__":
    main()
