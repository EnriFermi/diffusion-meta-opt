#!/usr/bin/env python3
"""Encode sealed train-only OOD anchors into both codec-specific z_enc bundles."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from big_vae.weightclip_benchmark.decoder_adapters import OfficialWeightCLIPMultiWindowDecoderAdapter
from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    build_ours_task_fit_bundle,
    build_weightclip_task_fit_bundle,
    capture_train_only_operator_samples,
    load_checkpoint_state,
    load_sealed_ours_decoder,
)
from big_vae.weightclip_benchmark.manifests import write_json_immutable
from big_vae.weightclip_benchmark.official_bridge import OfficialWeightCLIPBridge, redact_secrets


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("OOD anchor manifest is empty")
    return rows


def build_anchor_codec_bundles(config: Mapping[str, Any]) -> dict[str, Any]:
    device = str(config.get("device", "cuda"))
    anchor_manifest = ArtifactRef.create(config["anchor_manifest"])
    rows = _rows(Path(anchor_manifest.path))
    datasets = config["datasets"]
    seal_ref = ArtifactRef.create(config["ours_codec_seal"])
    ours_decoder = load_sealed_ours_decoder(seal_ref, device=device)
    official = config["official_weightclip"]
    bridge = OfficialWeightCLIPBridge(official["repo_path"], official["cache_dir"], device=device, verbose=True)
    dataset_encoder, encoder_provenance = bridge.load_dataset_encoder_only(
        dataset_encoder_path=official["dataset_encoder_path"], allow_download=False
    )
    output = Path(config["output_dir"]).resolve()
    written: list[dict[str, Any]] = []
    official_loaded = None
    for row in rows:
        if row.get("ood_validation_access") is not False or row.get("ood_test_access") is not False:
            raise ValueError("OOD anchor manifest is not sealed train-only")
        checkpoint_ref = ArtifactRef.from_mapping(row["checkpoint"])
        checkpoint_ref.verify("OOD anchor checkpoint")
        complete = row.get("complete_record", {})
        if complete.get("checkpoint") != asdict(checkpoint_ref):
            raise ValueError("OOD anchor completion does not bind its checkpoint")
        if complete.get("anchor_contract_sha256") != row.get("anchor_contract_sha256"):
            raise ValueError("OOD anchor manifest and completion bind different contracts")
        if complete.get("ood_validation_access") is not False or complete.get("ood_test_access") is not False:
            raise ValueError("OOD anchor completion is not train-only")
        dataset = str(row["dataset"])
        dataset_ref = ArtifactRef.from_mapping(datasets[dataset]["dataset_pt"])
        if dataset_ref.sha256 != ArtifactRef.from_mapping(row["dataset_pt"]).sha256:
            raise ValueError("OOD anchor and codec-bundle config disagree on dataset")
        checkpoint_row = {
            "dataset": dataset,
            "lineage_id": f"ood-anchor:seed={int(row['training_seed'])}",
            "split": "ood_oracle_anchor_train_only",
            "checkpoint_path": checkpoint_ref.path,
            "checkpoint_sha256": checkpoint_ref.sha256,
        }
        context_indices = datasets[dataset]["context_indices"]
        prompt_indices = datasets[dataset]["prompt_indices"]
        ours = build_ours_task_fit_bundle(
            checkpoint_row=checkpoint_row,
            dataset_ref=dataset_ref,
            pair_manifest_ref=None,
            codec_seal_ref=seal_ref,
            decoder=ours_decoder,
            dataset_encoder=dataset_encoder,
            dataset_encoder_provenance=encoder_provenance,
            output_path=output / "ours" / dataset / f"seed-{int(row['evaluation_seed']):04d}.pt",
            context_indices=context_indices,
            prompt_indices=prompt_indices,
            encode_batch_size=int(config.get("encode_batch_size", 8)),
            operator_samples=capture_train_only_operator_samples(
                checkpoint_ref=checkpoint_ref,
                dataset_ref=dataset_ref,
                context_indices=context_indices,
                device=device,
                batch_size=int(config.get("capture_batch_size", 32)),
            ),
            train_only_dataset=True,
        )
        state = load_checkpoint_state(checkpoint_ref)
        if official_loaded is None:
            official_loaded = bridge.load_codec_tokenizer_only(
                reference_state=state,
                reference_state_sha256=checkpoint_ref.sha256,
                checkpoint_path=official["checkpoint_path"],
                dataset_encoder_path=official["dataset_encoder_path"],
                allow_download=False,
            )
        else:
            official_loaded = bridge.retarget_tokenizer(
                official_loaded, reference_state=state, reference_state_sha256=checkpoint_ref.sha256
            )
        wc_decoder = OfficialWeightCLIPMultiWindowDecoderAdapter(
            official_loaded.weight_model,
            official_loaded.tokenizer,
            anchor_state=state,
            window_size=512,
            provenance={
                "codec": "weightclip",
                "checkpoint_sha256": official_loaded.provenance["checkpoint"]["sha256"],
                "official": official_loaded.provenance,
            },
        )
        weightclip = build_weightclip_task_fit_bundle(
            checkpoint_row=checkpoint_row,
            dataset_ref=dataset_ref,
            decoder=wc_decoder,
            dataset_encoder=dataset_encoder,
            dataset_encoder_provenance=encoder_provenance,
            output_path=output / "weightclip" / dataset / f"seed-{int(row['evaluation_seed']):04d}.pt",
            context_indices=context_indices,
            prompt_indices=prompt_indices,
            train_only_dataset=True,
        )
        written.append(
            {
                "dataset": dataset,
                "evaluation_seed": int(row["evaluation_seed"]),
                "anchor_contract_sha256": row["anchor_contract_sha256"],
                "ours": ours,
                "weightclip": weightclip,
            }
        )
    index = {
        "schema_version": 1,
        "kind": "ood_anchor_codec_zenc_bundle_index",
        "anchor_manifest": asdict(anchor_manifest),
        "ours_codec_seal": asdict(seal_ref),
        "bundles": written,
        "target_validation_access": False,
        "target_test_access": False,
        "resolved_config": redact_secrets(dict(config)),
    }
    write_json_immutable(output / "bundle_index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = build_anchor_codec_bundles(config)
    print(f"[ood-anchor-codecs:done] bundles={len(result['bundles'])} output={config['output_dir']}")


if __name__ == "__main__":
    main()
