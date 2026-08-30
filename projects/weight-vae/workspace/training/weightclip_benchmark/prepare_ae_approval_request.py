from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from big_vae.weightclip_benchmark.ae_candidate_config import (
    assert_exact_candidate_model,
    compose_resolved_train_config,
    production_scientific_config_sha256,
    resolve_ae_candidate,
)
from big_vae.weightclip_benchmark.ae_scaling import (
    ApprovalRequirements,
    apply_scaling_axes,
    approval_template,
    canonical_fingerprint,
    exact_parameter_ledger,
)
from big_vae.weightclip_benchmark.manifests import sha256_file, write_json_immutable
from training.weightclip_benchmark.compare_ae_runtime_profiles import (
    validate_comparison_report,
)
from training.weightclip_benchmark.train_ae import (
    _assert_resolved_contract,
    _candidate,
    _load_yaml,
    _operator_bank_overrides,
    _resolve_path,
    _validate_candidate_artifact_index,
    _validate_runtime_profile_summary,
)


REQUEST_SCHEMA = "weightclip_ae_production_approval_request_v1"


def _build_request_contract(
    *,
    config_path: Path,
    candidate_name: str,
    candidate_artifact_index: Path,
    pair_manifest: Path,
    production_exact_summary: Path,
    producer_stress_summary: Path,
    historical_comparison: Path,
 ) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    axes = _candidate(config, candidate_name)
    base_path = _resolve_path(str(config["base_model_config"]), relative_to=config_path.parent)
    candidate = resolve_ae_candidate(apply_scaling_axes(_load_yaml(base_path), axes))
    parameter_ledger = exact_parameter_ledger(candidate.resolved_model_config)
    trainable_parameters = int(parameter_ledger["trainable_parameters"])
    bank_overrides, pair_path = _operator_bank_overrides(
        config, config_path, pair_manifest=pair_manifest
    )
    production_resolved = compose_resolved_train_config(
        [
            *candidate.hydra_model_overrides,
            f"train.max_steps={int(config['training_steps'])}",
            "train.kl_beta=0.0",
            "train.kl_schedule.enabled=false",
            *bank_overrides,
        ]
    )
    assert_exact_candidate_model(candidate, production_resolved, context="approval request")
    _assert_resolved_contract(production_resolved, pair_path)
    scientific_sha = production_scientific_config_sha256(production_resolved)
    candidate_artifacts = _validate_candidate_artifact_index(
        candidate_artifact_index.resolve(),
        candidate_name=candidate_name,
        model_fingerprint_sha256=candidate.model_fingerprint_sha256,
        trainable_parameters=trainable_parameters,
    )
    common = {
        "candidate_name": candidate_name,
        "model_fingerprint_sha256": candidate.model_fingerprint_sha256,
        "trainable_parameters": trainable_parameters,
        "pair_manifest_sha256": sha256_file(pair_path),
        "production_scientific_sha256": scientific_sha,
        "source_implementation_seal_sha256": str(
            candidate_artifacts["source_implementation_seal_sha256"]
        ),
        "candidate_artifact_set_sha256": str(
            candidate_artifacts["artifact_set_fingerprint_sha256"]
        ),
        "candidate_report_sha256": str(candidate_artifacts["candidate_report_sha256"]),
        "candidate_index_sha256": str(candidate_artifacts["candidate_index_sha256"]),
        "production_resolved": production_resolved,
    }
    exact = _validate_runtime_profile_summary(
        production_exact_summary.resolve(),
        **common,
        expected_profile_mode="production_exact",
    )
    stress = _validate_runtime_profile_summary(
        producer_stress_summary.resolve(),
        **common,
        expected_profile_mode="producer_stress",
    )
    comparison = validate_comparison_report(historical_comparison.resolve())
    if comparison["decision"]["recommended_candidate"] != candidate_name:
        raise RuntimeError("requested candidate differs from the historical recommendation")
    requirements = ApprovalRequirements(
        candidate_name=candidate_name,
        model_fingerprint_sha256=candidate.model_fingerprint_sha256,
        tile_rows=int(config["tile_shape"][0]),
        tile_cols=int(config["tile_shape"][1]),
        training_steps=int(config["training_steps"]),
        production_scientific_config_sha256=scientific_sha,
        candidate_artifact_set_sha256=str(
            candidate_artifacts["artifact_set_fingerprint_sha256"]
        ),
        candidate_report_sha256=str(candidate_artifacts["candidate_report_sha256"]),
        candidate_index_sha256=str(candidate_artifacts["candidate_index_sha256"]),
        trainable_parameters=trainable_parameters,
        runtime_profile_summary_sha256=str(exact["runtime_profile_summary_sha256"]),
        producer_stress_profile_summary_sha256=str(
            stress["runtime_profile_summary_sha256"]
        ),
        pair_manifest_sha256=sha256_file(pair_path),
        source_implementation_seal_sha256=str(
            candidate_artifacts["source_implementation_seal_sha256"]
        ),
    )
    tool_path = Path(__file__).resolve()
    return {
        "schema": REQUEST_SCHEMA,
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "base_model_config": {"path": str(base_path), "sha256": sha256_file(base_path)},
            "candidate_artifact_index": {
                "path": str(candidate_artifact_index.resolve()),
                "sha256": str(candidate_artifacts["candidate_index_sha256"]),
            },
            "pair_manifest": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
            "production_exact_summary": {
                "path": str(production_exact_summary.resolve()),
                "sha256": str(exact["runtime_profile_summary_sha256"]),
            },
            "producer_stress_summary": {
                "path": str(producer_stress_summary.resolve()),
                "sha256": str(stress["runtime_profile_summary_sha256"]),
            },
            "historical_comparison": {
                "path": str(historical_comparison.resolve()),
                "sha256": sha256_file(historical_comparison),
            },
            "request_tool_source": {"path": str(tool_path), "sha256": sha256_file(tool_path)},
        },
        "approval": approval_template(requirements),
        "candidate_index": {
            "path": str(candidate_artifact_index.resolve()),
            "sha256": str(candidate_artifacts["candidate_index_sha256"]),
        },
        "production_exact_evidence": exact,
        "producer_stress_evidence": stress,
        "historical_comparison": {
            "path": str(historical_comparison.resolve()),
            "sha256": sha256_file(historical_comparison),
            "comparison_fingerprint_sha256": comparison[
                "comparison_fingerprint_sha256"
            ],
        },
        "decision": {
            "status": "pending_explicit_user_approval",
            "approved_by_user": False,
            "production_launch_authorized": False,
        },
    }


def materialize_pending_request(
    *,
    config_path: Path,
    candidate_name: str,
    candidate_artifact_index: Path,
    pair_manifest: Path,
    production_exact_summary: Path,
    producer_stress_summary: Path,
    historical_comparison: Path,
    output_root: Path,
) -> Path:
    contract = _build_request_contract(
        config_path=config_path,
        candidate_name=candidate_name,
        candidate_artifact_index=candidate_artifact_index,
        pair_manifest=pair_manifest,
        production_exact_summary=production_exact_summary,
        producer_stress_summary=producer_stress_summary,
        historical_comparison=historical_comparison,
    )
    fingerprint = canonical_fingerprint(contract)
    request = {**contract, "approval_request_fingerprint_sha256": fingerprint}
    path = output_root.resolve() / f"approval-request-{fingerprint[:16]}" / "request.json"
    write_json_immutable(path, request)
    validate_pending_request(path)
    return path


def validate_pending_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = {
        key: value
        for key, value in payload.items()
        if key != "approval_request_fingerprint_sha256"
    }
    fingerprint = str(payload.get("approval_request_fingerprint_sha256", ""))
    if (
        payload.get("schema") != REQUEST_SCHEMA
        or canonical_fingerprint(contract) != fingerprint
        or path.parent.name != f"approval-request-{fingerprint[:16]}"
        or payload.get("approval", {}).get("status") != "pending_user_approval"
        or payload.get("approval", {}).get("approved_by_user") is not False
        or payload.get("decision")
        != {
            "status": "pending_explicit_user_approval",
            "approved_by_user": False,
            "production_launch_authorized": False,
        }
    ):
        raise RuntimeError("pending approval request contract is invalid")
    inputs = payload.get("inputs", {})
    expected_input_keys = {
        "config",
        "base_model_config",
        "candidate_artifact_index",
        "pair_manifest",
        "production_exact_summary",
        "producer_stress_summary",
        "historical_comparison",
        "request_tool_source",
    }
    if set(inputs) != expected_input_keys:
        raise RuntimeError("pending approval request input inventory is incomplete or has extras")
    for name, ref in inputs.items():
        source = Path(str(ref.get("path", ""))).resolve()
        if not source.is_file() or sha256_file(source) != ref.get("sha256"):
            raise RuntimeError(f"pending approval request input is missing/tampered: {name}")
    approval = payload["approval"]
    expected = _build_request_contract(
        config_path=Path(inputs["config"]["path"]),
        candidate_name=str(approval["candidate_name"]),
        candidate_artifact_index=Path(inputs["candidate_artifact_index"]["path"]),
        pair_manifest=Path(inputs["pair_manifest"]["path"]),
        production_exact_summary=Path(inputs["production_exact_summary"]["path"]),
        producer_stress_summary=Path(inputs["producer_stress_summary"]["path"]),
        historical_comparison=Path(inputs["historical_comparison"]["path"]),
    )
    if expected != contract:
        raise RuntimeError("pending approval request does not recompute from its bound inputs")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a non-authorizing AE approval request")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-artifact-index", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--production-exact-summary", type=Path, required=True)
    parser.add_argument("--producer-stress-summary", type=Path, required=True)
    parser.add_argument("--historical-comparison", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = materialize_pending_request(
        config_path=args.config,
        candidate_name=args.candidate,
        candidate_artifact_index=args.candidate_artifact_index,
        pair_manifest=args.pair_manifest,
        production_exact_summary=args.production_exact_summary,
        producer_stress_summary=args.producer_stress_summary,
        historical_comparison=args.historical_comparison,
        output_root=args.output_root,
    )
    print(f"[weightclip-ae-approval-request] pending={path} sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
