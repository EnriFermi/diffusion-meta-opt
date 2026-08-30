#!/usr/bin/env python3
"""Build target-weight-free conditioning bundles for anchor-free OOD arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from big_vae.weightclip_benchmark.decoder_adapters import OfficialWeightCLIPMultiWindowDecoderAdapter
from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    build_ood_conditioning_bundle,
    create_ood_template_checkpoint,
    load_checkpoint_state,
)
from big_vae.weightclip_benchmark.official_bridge import OfficialWeightCLIPBridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(raw["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = str(raw.get("device", "cuda"))
    official = raw["official_weightclip"]
    bridge = OfficialWeightCLIPBridge(official["repo_path"], official["cache_dir"], device=device, verbose=True)
    encoder, encoder_provenance = bridge.load_dataset_encoder_only(
        dataset_encoder_path=official["dataset_encoder_path"], allow_download=False
    )
    loaded = None
    for dataset_id, row in raw["datasets"].items():
        dataset_ref = ArtifactRef.from_mapping(row["dataset_pt"])
        dataset_ref.verify(f"OOD dataset {dataset_id}")
        template_ref = create_ood_template_checkpoint(
            output / "templates" / f"resnet18slim_c{int(row['num_classes'])}.pt",
            num_classes=int(row["num_classes"]),
            width_mult=float(raw.get("width_mult", 0.5)),
            seed=int(raw.get("template_seed", 0)),
        )
        state = load_checkpoint_state(template_ref)
        if loaded is None:
            loaded = bridge.load_codec_tokenizer_only(
                reference_state=state,
                reference_state_sha256=template_ref.sha256,
                checkpoint_path=official["checkpoint_path"],
                dataset_encoder_path=official["dataset_encoder_path"],
            )
        else:
            loaded = bridge.retarget_tokenizer(
                loaded, reference_state=state, reference_state_sha256=template_ref.sha256
            )
        decoder = OfficialWeightCLIPMultiWindowDecoderAdapter(
            loaded.weight_model,
            loaded.tokenizer,
            anchor_state=state,
            window_size=512,
            provenance={
                "codec": "weightclip",
                "checkpoint_sha256": loaded.provenance["checkpoint"]["sha256"],
                "official": loaded.provenance,
                "reference_is_architecture_default_template": True,
            },
        )
        result = build_ood_conditioning_bundle(
            dataset_id=str(dataset_id),
            dataset_ref=dataset_ref,
            template_ref=template_ref,
            dataset_encoder=encoder,
            dataset_encoder_provenance=encoder_provenance,
            output_path=output / f"{dataset_id}.pt",
            context_indices=row["context_indices"],
            prompt_indices=row["prompt_indices"],
            prompt_candidate_indices=row.get("prompt_candidate_indices"),
            weightclip_decoder=decoder,
        )
        print(f"[ood-conditioning] dataset={dataset_id} bundle={result['bundle']['path']}")


if __name__ == "__main__":
    main()
