#!/usr/bin/env python3
"""Seal a user-approved completed deterministic AE for Stage-D/G consumers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from big_vae.weightclip_benchmark.ae_candidate_config import (
    ResolvedAECandidate,
    assert_exact_candidate_model,
    normalize_launcher_runtime_config,
    production_scientific_config_sha256,
    resolve_ae_candidate,
)
from big_vae.weightclip_benchmark.ae_scaling import (
    ApprovalRequirements,
    ScalingAxes,
    apply_scaling_axes,
    canonical_fingerprint,
    require_production_approval,
)
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef
from big_vae.weightclip_benchmark.manifests import sha256_file, write_json_immutable
from big_vae.models.big_weight_vae import BigWeightVAE
from training.big_vae.model_config import build_big_vae_model_config


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    # PyYAML 1.1 parses JSON scientific-notation numbers such as ``5e-05`` as
    # strings.  Resolved launch/approval artifacts are JSON and must retain the
    # exact numeric types produced by Hydra.
    payload = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping in {path}")
    return dict(payload)


def _expected_model_state_schema(resolved_model_config: Mapping[str, Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    with torch.device("meta"):
        model = BigWeightVAE(build_big_vae_model_config(resolved_model_config))
    return {
        str(key): (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        for key, tensor in model.state_dict().items()
    }


def _validate_checkpoint_provenance_and_state(
    checkpoint: Mapping[str, Any],
    *,
    resolved_launch: Mapping[str, Any],
    canonical_candidate: ResolvedAECandidate,
    requirements: ApprovalRequirements,
) -> None:
    checkpoint_config = checkpoint.get("config")
    if (
        not isinstance(checkpoint_config, Mapping)
        or canonical_fingerprint(normalize_launcher_runtime_config(checkpoint_config))
        != canonical_fingerprint(normalize_launcher_runtime_config(resolved_launch))
    ):
        raise ValueError("AE checkpoint embedded config differs from the approved resolved launch")
    assert_exact_candidate_model(
        canonical_candidate,
        checkpoint_config,
        context="AE checkpoint embedded model",
    )
    checkpoint_preflight = checkpoint_config.get("weightclip_launch_preflight", {})
    required_preflight = {
        "candidate": requirements.candidate_name,
        "model_fingerprint_sha256": requirements.model_fingerprint_sha256,
        "production_scientific_config_sha256": requirements.production_scientific_config_sha256,
        "source_implementation_seal_sha256": requirements.source_implementation_seal_sha256,
        "artifact_set_fingerprint_sha256": requirements.candidate_artifact_set_sha256,
        "candidate_report_sha256": requirements.candidate_report_sha256,
        "candidate_index_sha256": requirements.candidate_index_sha256,
        "runtime_profile_summary_sha256": requirements.runtime_profile_summary_sha256,
        "producer_stress_profile_summary_sha256": requirements.producer_stress_profile_summary_sha256,
        "pair_manifest_sha256": requirements.pair_manifest_sha256,
        "trainable_parameters": requirements.trainable_parameters,
        "approval_request_sha256": requirements.approval_request_sha256,
        "approval_request_fingerprint_sha256": (
            requirements.approval_request_fingerprint_sha256
        ),
        "canonical_config_sha256": requirements.canonical_config_sha256,
        "approved_config_sha256": requirements.approved_config_sha256,
        "approval_file_sha256": resolved_launch.get("weightclip_launch_preflight", {}).get(
            "approval_file_sha256"
        ),
        "disk_preflight_report_sha256": resolved_launch.get(
            "weightclip_launch_preflight", {}
        ).get("disk_preflight_report_sha256"),
    }
    mismatches = {
        key: {"expected": expected, "actual": checkpoint_preflight.get(key)}
        for key, expected in required_preflight.items()
        if checkpoint_preflight.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"AE checkpoint preflight provenance mismatch: {mismatches}")
    if (
        production_scientific_config_sha256(checkpoint_config)
        != requirements.production_scientific_config_sha256
    ):
        raise ValueError("AE checkpoint scientific config differs from the approved production config")
    raw_state = next(
        (checkpoint[key] for key in ("model_state_dict", "model_state", "model") if key in checkpoint),
        None,
    )
    if not isinstance(raw_state, Mapping):
        raise ValueError("AE checkpoint lacks a model state")
    actual_schema = {
        str(key): (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        for key, tensor in raw_state.items()
        if isinstance(tensor, torch.Tensor)
    }
    if len(actual_schema) != len(raw_state):
        raise ValueError("AE checkpoint model state contains non-tensor entries")
    expected_schema = _expected_model_state_schema(canonical_candidate.resolved_model_config)
    if actual_schema != expected_schema:
        missing = sorted(set(expected_schema) - set(actual_schema))[:8]
        unexpected = sorted(set(actual_schema) - set(expected_schema))[:8]
        wrong = [
            key for key in sorted(set(actual_schema) & set(expected_schema))
            if actual_schema[key] != expected_schema[key]
        ][:8]
        raise ValueError(
            "AE checkpoint state_dict key/shape/dtype schema mismatch: "
            f"missing={missing} unexpected={unexpected} wrong={wrong}"
        )


def build_seal(
    *, config_path: Path, candidate_name: str, approval_request_path: Path,
    approval_path: Path,
    checkpoint_path: Path, resolved_model_config_path: Path, output_path: Path,
) -> dict[str, Any]:
    from training.weightclip_benchmark.approve_ae import validate_approved_transition

    transition = validate_approved_transition(
        approval_request_path=approval_request_path,
        approved_config_path=config_path,
        approval_path=approval_path,
    )
    config = _load_mapping(config_path)
    candidate = next((row for row in config["candidates"] if row["name"] == candidate_name), None)
    if candidate is None:
        raise ValueError(f"unknown AE candidate {candidate_name!r}")
    base_path = (config_path.parent / str(config["base_model_config"])).resolve()
    canonical_candidate = resolve_ae_candidate(
        apply_scaling_axes(_load_mapping(base_path), ScalingAxes.from_mapping(candidate))
    )
    fingerprint = canonical_candidate.model_fingerprint_sha256
    resolved = _load_mapping(resolved_model_config_path)
    assert_exact_candidate_model(canonical_candidate, resolved, context="AE seal resolved launch")
    launch = resolved.get("weightclip_launch_preflight", {})
    if launch.get("candidate") != candidate_name or launch.get("model_fingerprint_sha256") != fingerprint:
        raise ValueError("resolved launch config is not bound to the approved AE candidate")
    requirements = ApprovalRequirements(
        candidate_name=candidate_name,
        model_fingerprint_sha256=fingerprint,
        tile_rows=int(config["tile_shape"][0]), tile_cols=int(config["tile_shape"][1]),
        training_steps=int(config["training_steps"]),
        production_scientific_config_sha256=launch.get("production_scientific_config_sha256"),
        candidate_artifact_set_sha256=launch.get("artifact_set_fingerprint_sha256"),
        candidate_report_sha256=launch.get("candidate_report_sha256"),
        candidate_index_sha256=launch.get("candidate_index_sha256"),
        trainable_parameters=launch.get("trainable_parameters"),
        runtime_profile_summary_sha256=launch.get("runtime_profile_summary_sha256"),
        producer_stress_profile_summary_sha256=launch.get(
            "producer_stress_profile_summary_sha256"
        ),
        pair_manifest_sha256=launch.get("pair_manifest_sha256"),
        source_implementation_seal_sha256=launch.get("source_implementation_seal_sha256"),
        approval_request_sha256=launch.get("approval_request_sha256"),
        approval_request_fingerprint_sha256=launch.get(
            "approval_request_fingerprint_sha256"
        ),
        canonical_config_sha256=launch.get("canonical_config_sha256"),
        approved_config_sha256=launch.get("approved_config_sha256"),
    )
    expected_chain = {
        "approval_request_sha256": sha256_file(approval_request_path),
        "approval_request_fingerprint_sha256": transition["request"][
            "approval_request_fingerprint_sha256"
        ],
        "canonical_config_sha256": transition["canonical_config_sha256"],
        "approved_config_sha256": transition["approved_config_sha256"],
        "approval_file_sha256": transition["approval_sha256"],
    }
    mismatched_chain = {
        key: {"expected": value, "actual": launch.get(key)}
        for key, value in expected_chain.items()
        if launch.get(key) != value
    }
    if mismatched_chain:
        raise ValueError(f"resolved launch approval-chain mismatch: {mismatched_chain}")
    disk_preflight_path = Path(str(launch.get("disk_preflight_report_path", ""))).resolve()
    if (
        not disk_preflight_path.is_file()
        or sha256_file(disk_preflight_path) != launch.get("disk_preflight_report_sha256")
    ):
        raise ValueError("resolved launch disk-preflight report is missing or tampered")
    approval = require_production_approval(approval_path, requirements)
    checkpoint_ref = ArtifactRef.create(checkpoint_path)
    checkpoint = torch.load(checkpoint_ref.path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("AE checkpoint is not a mapping")
    _validate_checkpoint_provenance_and_state(
        checkpoint,
        resolved_launch=resolved,
        canonical_candidate=canonical_candidate,
        requirements=requirements,
    )
    checkpoint_step = checkpoint.get("global_step", checkpoint.get("step"))
    approved_steps = int(config["training_steps"])
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise ValueError("AE checkpoint step must be an explicit integer")
    if checkpoint_step != approved_steps:
        raise ValueError(
            "AE checkpoint step differs from the exact approved training budget: "
            f"checkpoint={checkpoint_step} approved={approved_steps}"
        )
    model_config_ref = ArtifactRef.create(resolved_model_config_path)
    approval_ref = ArtifactRef.create(approval_path)
    approval_request_ref = ArtifactRef.create(approval_request_path)
    approved_config_ref = ArtifactRef.create(config_path)
    canonical_config_ref = ArtifactRef.create(transition["canonical_config_path"])
    disk_preflight_ref = ArtifactRef.create(disk_preflight_path)
    seal = {
        "schema_version": 1,
        "codec": "ours",
        "deterministic_ae": True,
        "tile_shape": [128, 128],
        "candidate_name": candidate_name,
        "model_fingerprint_sha256": fingerprint,
        "production_scientific_config_sha256": requirements.production_scientific_config_sha256,
        "source_implementation_seal_sha256": requirements.source_implementation_seal_sha256,
        "candidate_artifact_set_sha256": requirements.candidate_artifact_set_sha256,
        "candidate_report_sha256": requirements.candidate_report_sha256,
        "candidate_index_sha256": requirements.candidate_index_sha256,
        "runtime_profile_summary_sha256": requirements.runtime_profile_summary_sha256,
        "producer_stress_profile_summary_sha256": requirements.producer_stress_profile_summary_sha256,
        "pair_manifest_sha256": requirements.pair_manifest_sha256,
        "approval_request_sha256": requirements.approval_request_sha256,
        "approval_request_fingerprint_sha256": (
            requirements.approval_request_fingerprint_sha256
        ),
        "canonical_config_sha256": requirements.canonical_config_sha256,
        "approved_config_sha256": requirements.approved_config_sha256,
        "approval_file_sha256": transition["approval_sha256"],
        "disk_preflight": asdict(disk_preflight_ref),
        "trainable_parameters": requirements.trainable_parameters,
        "training_steps": int(config["training_steps"]),
        "checkpoint": asdict(checkpoint_ref),
        "model_config": asdict(model_config_ref),
        "approval": asdict(approval_ref),
        "approval_request": asdict(approval_request_ref),
        "approved_launch_config": asdict(approved_config_ref),
        "canonical_pending_config": asdict(canonical_config_ref),
        "approval_record": approval,
    }
    digest = write_json_immutable(output_path, seal)
    print(f"[seal-ae:done] seal={output_path.resolve()} sha256={digest} checkpoint={checkpoint_ref.sha256}")
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--approval-request", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_seal(
        config_path=args.config.resolve(), candidate_name=args.candidate,
        approval_request_path=args.approval_request.resolve(),
        approval_path=args.approval_file.resolve(), checkpoint_path=args.checkpoint.resolve(),
        resolved_model_config_path=args.resolved_model_config.resolve(), output_path=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
