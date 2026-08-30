#!/usr/bin/env python3
"""Build immutable Stage-D bundles from zoo manifests and pinned codecs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import yaml
import torch

from big_vae.weightclip_benchmark.decoder_adapters import OfficialWeightCLIPMultiWindowDecoderAdapter
from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    build_ours_task_fit_bundle,
    build_weightclip_task_fit_bundle,
    create_ood_template_checkpoint,
    load_checkpoint_state,
    load_sealed_ours_decoder,
)
from big_vae.weightclip_benchmark.manifests import write_json_immutable
from big_vae.weightclip_benchmark.official_bridge import OfficialWeightCLIPBridge, redact_secrets


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("checkpoint manifest is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    codec = str(raw["codec"])
    device = str(raw.get("device", "cuda"))
    manifest_ref = ArtifactRef.create(raw["checkpoint_manifest"])
    rows = _load_rows(Path(manifest_ref.path))
    selected_index = int(raw.get("checkpoint_index_zero_based", 44))
    selected = [row for row in rows if int(row["checkpoint_index_zero_based"]) == selected_index]
    if not selected:
        raise ValueError(f"no checkpoint rows at index {selected_index}")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("config.datasets must map dataset ids to dataset.pt artifact refs")
    official = dict(raw["official_weightclip"])
    bridge = OfficialWeightCLIPBridge(official["repo_path"], official["cache_dir"], device=device, verbose=True)
    encoder, encoder_provenance = bridge.load_dataset_encoder_only(
        dataset_encoder_path=official["dataset_encoder_path"], allow_download=False
    )
    output = Path(raw["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    codec_seal_ref = ArtifactRef.create(raw["codec_seal"]) if codec == "ours" else None
    pair_manifest_ref = ArtifactRef.create(raw["pair_manifest"]) if codec == "ours" else None
    ours_decoder = load_sealed_ours_decoder(codec_seal_ref, device=device) if codec_seal_ref is not None else None
    official_bundle = None
    written: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        dataset_row = datasets.get(row["dataset"])
        if not isinstance(dataset_row, dict):
            raise ValueError(f"no sealed dataset.pt ref for {row['dataset']}")
        dataset_ref = ArtifactRef.from_mapping(dataset_row["dataset_pt"])
        dataset_ref.verify(f"dataset {row['dataset']}")
        target = output / str(row["dataset"]) / f"{row['lineage_id']}.pt"
        print(f"[task-bundle] codec={codec} item={index}/{len(selected)} dataset={row['dataset']} lineage={row['lineage_id']}")
        if codec == "ours":
            assert ours_decoder is not None and codec_seal_ref is not None and pair_manifest_ref is not None
            result = build_ours_task_fit_bundle(
                checkpoint_row=row,
                dataset_ref=dataset_ref,
                pair_manifest_ref=pair_manifest_ref,
                codec_seal_ref=codec_seal_ref,
                decoder=ours_decoder,
                dataset_encoder=encoder,
                dataset_encoder_provenance=encoder_provenance,
                output_path=target,
                context_indices=raw["context_indices"],
                prompt_indices=raw["prompt_indices"],
                prompt_candidate_count=int(raw.get("prompt_candidate_count", 100)),
                prompt_candidate_seed=int(raw.get("prompt_candidate_seed", 0)),
                encode_batch_size=int(raw.get("encode_batch_size", 8)),
            )
        elif codec == "weightclip":
            source = load_checkpoint_state(ArtifactRef.create(row["checkpoint_path"]))
            if official_bundle is None:
                official_bundle = bridge.load_codec_tokenizer_only(
                    reference_state=source,
                    reference_state_sha256=str(row["checkpoint_sha256"]),
                    checkpoint_path=official["checkpoint_path"],
                    dataset_encoder_path=official["dataset_encoder_path"],
                    allow_download=False,
                )
            else:
                official_bundle = bridge.retarget_tokenizer(
                    official_bundle,
                    reference_state=source,
                    reference_state_sha256=str(row["checkpoint_sha256"]),
                )
            decoder = OfficialWeightCLIPMultiWindowDecoderAdapter(
                official_bundle.weight_model,
                official_bundle.tokenizer,
                anchor_state=source,
                window_size=512,
                provenance={
                    "codec": "weightclip",
                    "checkpoint_sha256": official_bundle.provenance["checkpoint"]["sha256"],
                    "official": official_bundle.provenance,
                },
            )
            template_ref = create_ood_template_checkpoint(
                output / "architecture_default_templates" / f"resnet18slim_c{int(source['fc.weight'].shape[0])}.pt",
                num_classes=int(source["fc.weight"].shape[0]),
                width_mult=float(source["conv1.weight"].shape[0]) / 64.0,
                seed=0,
            )
            template_state = load_checkpoint_state(template_ref)
            template_bundle = bridge.retarget_tokenizer(
                official_bundle,
                reference_state=template_state,
                reference_state_sha256=template_ref.sha256,
            )
            template_decoder = OfficialWeightCLIPMultiWindowDecoderAdapter(
                template_bundle.weight_model,
                template_bundle.tokenizer,
                anchor_state=template_state,
                window_size=512,
                provenance={
                    "codec": "weightclip",
                    "checkpoint_sha256": template_bundle.provenance["checkpoint"]["sha256"],
                    "official": template_bundle.provenance,
                    "reference_is_architecture_default_template": True,
                },
            )
            with torch.inference_mode():
                template_z_enc = template_decoder.encode_anchor().detach().cpu().float()
            result = build_weightclip_task_fit_bundle(
                checkpoint_row=row,
                dataset_ref=dataset_ref,
                decoder=decoder,
                dataset_encoder=encoder,
                dataset_encoder_provenance=encoder_provenance,
                output_path=target,
                context_indices=raw["context_indices"],
                prompt_indices=raw["prompt_indices"],
                prompt_candidate_count=int(raw.get("prompt_candidate_count", 100)),
                prompt_candidate_seed=int(raw.get("prompt_candidate_seed", 0)),
                template_checkpoint_ref=template_ref,
                template_z_enc=template_z_enc,
                template_token_mask=template_decoder.window_token_mask,
            )
        else:
            raise ValueError(f"unsupported codec {codec!r}")
        written.append(result)
    summary = {
        "schema_version": 1,
        "codec": codec,
        "checkpoint_manifest": asdict(manifest_ref),
        "checkpoint_index_zero_based": selected_index,
        "bundles": written,
        "resolved_config": redact_secrets(raw),
    }
    write_json_immutable(output / "bundle_index.json", summary)
    print(f"[task-bundle:done] count={len(written)} index={output / 'bundle_index.json'}")


if __name__ == "__main__":
    main()
