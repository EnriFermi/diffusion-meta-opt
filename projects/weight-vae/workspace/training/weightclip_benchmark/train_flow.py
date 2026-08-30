#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from big_vae.flow_matching.dataset import (
    DeterministicStepBatchSampler,
    GroupedLatentDataset,
    GroupedLatentRecord,
    LatentNormalizer,
    collate_latent_tiles,
    records_manifest,
    validate_record_codec_fingerprints,
    validate_cross_codec_prompt_banks,
    validate_successful_task_fit_records,
)
from big_vae.flow_matching.model import ConditionalVelocityTransformer, FlowModelConfig
from big_vae.flow_matching.train import FlowTrainConfig, assert_matched_flow_budgets, train_flow


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def _load_records(patterns: list[str]) -> tuple[list[GroupedLatentRecord], list[str]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no grouped latent records matched {patterns}")
    records = [GroupedLatentRecord.load(path) for path in paths]
    return records, paths


def _normalize_records(records: list[GroupedLatentRecord], normalizer: LatentNormalizer) -> None:
    for record in records:
        record.z_task = normalizer.normalize(record.z_task)
        if record.z_enc is not None:
            record.z_enc = normalizer.normalize(record.z_enc)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_group_ids(data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    if "expected_group_ids" in data:
        values = [str(value) for value in data["expected_group_ids"]]
        payload = {"source": "inline", "group_ids": values}
        payload["sha256"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values, payload
    marker = str(data.get("expected_inventory_path", ""))
    if not marker or marker.startswith("REQUIRED_"):
        raise ValueError("data.expected_inventory_path is required before flow training")
    path = Path(marker).resolve()
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        values = [str(value) for value in raw]
    elif isinstance(raw, dict) and isinstance(raw.get("group_ids"), list):
        values = [str(value) for value in raw["group_ids"]]
    elif isinstance(raw, dict) and isinstance(raw.get("records"), list):
        values = [str(row["group_id"]) for row in raw["records"]]
    else:
        raise ValueError("expected inventory must be a group-id list or contain group_ids/records")
    return values, {"source": str(path), "sha256": _sha256_file(path), "group_ids": values}


def _normalizers_equal(first: LatentNormalizer, second: LatentNormalizer) -> bool:
    return (
        first.fit_split == second.fit_split
        and first.codec_fingerprint == second.codec_fingerprint
        and first.eps == second.eps
        and torch.equal(first.center, second.center)
        and torch.equal(first.scale, second.scale)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one matched conditional flow in a codec latent space")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = _load_config(args.config)
    records, paths = _load_records(list(raw["data"]["record_globs"]))
    expected_codec = str(raw["data"]["codec"])
    codec_fingerprint = validate_record_codec_fingerprints(records, expected_codec=expected_codec)
    expected_group_ids, inventory = _expected_group_ids(raw["data"])
    validate_successful_task_fit_records(records, expected_group_ids=expected_group_ids)
    train_records = [record for record in records if record.split == "train"]
    if not train_records:
        raise ValueError("normalizer requires train records")
    train_codes = torch.cat([record.z_task for record in train_records], dim=0)
    train_masks = torch.cat(
        [
            record.tile_mask
            if expected_codec == "ours"
            else record.tile_mask & record.architecture_features[..., 1].to(dtype=torch.bool)
            for record in train_records
        ],
        dim=0,
    )
    if expected_codec == "weightclip" and not train_masks.any():
        raise ValueError("WeightCLIP controlled flow normalizer has zero body tokens")
    fitted_normalizer = LatentNormalizer.fit(
        train_codes,
        masks=train_masks,
        split="train",
        codec_fingerprint=codec_fingerprint,
    )
    output_dir = Path(raw["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer_path = output_dir / "latent_normalizer.pt"
    checkpoint_path = output_dir / "latest.pt"
    if checkpoint_path.exists() and not normalizer_path.exists():
        raise ValueError("resume checkpoint exists without its frozen latent normalizer")
    if normalizer_path.exists():
        normalizer = LatentNormalizer.from_state_dict(
            torch.load(normalizer_path, map_location="cpu", weights_only=False)
        )
        if not _normalizers_equal(normalizer, fitted_normalizer):
            raise ValueError("existing latent normalizer differs from current train-record contents")
    else:
        normalizer = fitted_normalizer
        torch.save(normalizer.state_dict(), normalizer_path)
    _normalize_records(records, normalizer)
    path_kind = str(raw["train"]["path_kind"])
    require_anchor = path_kind == "paired_anchor"
    train_dataset = GroupedLatentDataset(records, split="train", require_anchor=require_anchor)
    validation_dataset = GroupedLatentDataset(records, split="validation", require_anchor=require_anchor)
    train_cfg = FlowTrainConfig(**raw["train"])
    if train_cfg.microbatch_size * train_cfg.grad_accum_steps != train_cfg.effective_batch_size:
        raise ValueError("flow effective/microbatch/accumulation budget is inconsistent")
    microbatch_size = train_cfg.microbatch_size
    loader_args = {
        "num_workers": int(raw["data"].get("num_workers", 4)),
        "pin_memory": True,
        "persistent_workers": int(raw["data"].get("num_workers", 4)) > 0,
        "collate_fn": collate_latent_tiles,
    }
    train_sampler = DeterministicStepBatchSampler(
        len(train_dataset),
        microbatch_size,
        total_steps=train_cfg.steps * train_cfg.grad_accum_steps,
        seed=train_cfg.seed,
        prompt_bank_size=train_dataset.prompt_bank_size,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_args)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )
    model_cfg = dict(raw["model"])
    first = records[0]
    model_cfg.setdefault("latent_dim", int(first.z_task.shape[-1]))
    model_cfg.setdefault("dataset_embedding_dim", int(first.dataset_embedding.shape[-1]))
    model_cfg.setdefault("architecture_feature_dim", int(first.architecture_features.shape[-1]))
    model_cfg.setdefault("max_latent_tokens", int(first.z_task.shape[1]))
    model = ConditionalVelocityTransformer(FlowModelConfig(**model_cfg))
    # Construct the peer codec model from its records and compare actual core budgets
    # before any optimizer state is created. This also enforces one shared condition width.
    matched_globs = raw["data"].get("matched_record_globs")
    if not isinstance(matched_globs, dict) or set(matched_globs) != {"ours", "weightclip"}:
        raise ValueError("data.matched_record_globs must name exact ours and weightclip record sets")
    matched_models = {expected_codec: model}
    peer_codec = "weightclip" if expected_codec == "ours" else "ours"
    peer_records, _ = _load_records(list(matched_globs[peer_codec]))
    validate_record_codec_fingerprints(peer_records, expected_codec=peer_codec)
    if expected_codec == "ours":
        validate_cross_codec_prompt_banks(records, peer_records)
    else:
        validate_cross_codec_prompt_banks(peer_records, records)
    peer = peer_records[0]
    if int(peer.dataset_embedding.shape[-1]) != model.config.dataset_embedding_dim:
        raise ValueError("ours/WeightCLIP dataset conditioning widths differ")
    if int(peer.architecture_features.shape[-1]) != model.config.architecture_feature_dim:
        raise ValueError("ours/WeightCLIP architecture condition schemas differ")
    peer_cfg = dict(raw["model"])
    peer_cfg.update(
        latent_dim=int(peer.z_task.shape[-1]),
        dataset_embedding_dim=int(peer.dataset_embedding.shape[-1]),
        architecture_feature_dim=int(peer.architecture_features.shape[-1]),
        max_latent_tokens=int(peer.z_task.shape[1]),
    )
    matched_models[peer_codec] = ConditionalVelocityTransformer(FlowModelConfig(**peer_cfg))
    parameter_ledgers = assert_matched_flow_budgets(matched_models)
    record_artifacts = [
        {"path": str(Path(path).resolve()), "sha256": _sha256_file(path), "bytes": Path(path).stat().st_size}
        for path in paths
    ]
    record_manifest = records_manifest(records)
    artifact_manifest_hash = hashlib.sha256(
        json.dumps(record_artifacts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = train_flow(
        model,
        train_loader,
        validation_loader,
        train_cfg,
        provenance={
            "config_path": str(args.config.resolve()),
            "record_count": len(records),
            "record_manifest": record_manifest,
            "record_artifacts": record_artifacts,
            "record_artifacts_sha256": artifact_manifest_hash,
            "expected_inventory": inventory,
            "config_sha256": _sha256_file(args.config),
            "codec": expected_codec,
            "codec_fingerprint": codec_fingerprint,
            "normalizer_path": str(normalizer_path.resolve()),
            "normalizer_fit_split": "train",
            "parameter_ledger": model.parameter_ledger(),
            "matched_parameter_ledgers": parameter_ledgers,
            "validation_solver": train_cfg.validation_solver,
            "validation_nfe_steps": train_cfg.validation_nfe_steps,
            "activation_conditioning": raw["data"].get("activation_conditioning"),
            "objective_token_policy": (
                "official_valid_and_controlled_body" if expected_codec == "weightclip" else "codec_valid_tiles"
            ),
        },
    )
    print(
        json.dumps(
            {
                "stage": "flow_unsealed",
                "reason": "run frozen E4 decoded validation then seal_flow",
                "checkpoint": summary["checkpoint"],
                "normalizer": str(normalizer_path.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
