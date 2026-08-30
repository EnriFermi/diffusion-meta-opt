#!/usr/bin/env python3
"""Materialize and validate an explicit, immutable WeightCLIP AE approval transition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from big_vae.weightclip_benchmark.ae_scaling import (
    ApprovalRequirements,
    PRODUCTION_APPROVAL_RECORD_SCHEMA,
    PRODUCTION_APPROVAL_SCOPE,
    approval_template,
)
from big_vae.weightclip_benchmark.manifests import sha256_file, write_json_immutable
from training.weightclip_benchmark.prepare_ae_approval_request import (
    validate_pending_request,
)
from training.weightclip_benchmark.train_ae import _load_yaml, _resolve_path


APPROVED_CONFIG_STATUS = "approved_user_architecture_scaling"
APPROVED_CONFIG_MUTABLE_FIELDS = frozenset(
    {
        "status",
        "production_launch_enabled_after_approval",
        "selected_candidate",
        "approval_file",
    }
)


def _requirements_from_request(
    request: Mapping[str, Any], *, approved_config_sha256: str
) -> ApprovalRequirements:
    approval = request["approval"]
    return ApprovalRequirements(
        candidate_name=str(approval["candidate_name"]),
        model_fingerprint_sha256=str(approval["model_fingerprint_sha256"]),
        tile_rows=int(approval["tile_rows"]),
        tile_cols=int(approval["tile_cols"]),
        training_steps=int(approval["training_steps"]),
        production_scientific_config_sha256=str(
            approval["production_scientific_config_sha256"]
        ),
        candidate_artifact_set_sha256=str(approval["candidate_artifact_set_sha256"]),
        candidate_report_sha256=str(approval["candidate_report_sha256"]),
        candidate_index_sha256=str(request["candidate_index"]["sha256"]),
        trainable_parameters=int(approval["trainable_parameters"]),
        runtime_profile_summary_sha256=str(approval["runtime_profile_summary_sha256"]),
        producer_stress_profile_summary_sha256=str(
            approval["producer_stress_profile_summary_sha256"]
        ),
        pair_manifest_sha256=str(approval["pair_manifest_sha256"]),
        source_implementation_seal_sha256=str(
            approval["source_implementation_seal_sha256"]
        ),
        approval_request_sha256=sha256_file(Path(request["_validated_path"])),
        approval_request_fingerprint_sha256=str(
            request["approval_request_fingerprint_sha256"]
        ),
        canonical_config_sha256=str(request["inputs"]["config"]["sha256"]),
        approved_config_sha256=approved_config_sha256,
    )


def _assert_approved_config_diff(
    canonical: Mapping[str, Any],
    approved: Mapping[str, Any],
    *,
    candidate_name: str,
    approval_path: Path,
    approved_config_path: Path,
) -> None:
    changed = {key for key in set(canonical) | set(approved) if canonical.get(key) != approved.get(key)}
    if changed != APPROVED_CONFIG_MUTABLE_FIELDS:
        raise RuntimeError(
            "approved config differs from canonical config outside the exact approval allowlist: "
            f"changed={sorted(changed)} expected={sorted(APPROVED_CONFIG_MUTABLE_FIELDS)}"
        )
    if approved.get("status") != APPROVED_CONFIG_STATUS:
        raise RuntimeError("approved config has the wrong approval status")
    if approved.get("production_launch_enabled_after_approval") is not True:
        raise RuntimeError("approved config does not enable the approval-gated launcher")
    if approved.get("selected_candidate") != candidate_name:
        raise RuntimeError("approved config selected_candidate differs from the approval request")
    configured_approval = _resolve_path(
        str(approved.get("approval_file", "")), relative_to=approved_config_path.parent
    )
    if configured_approval != approval_path.resolve():
        raise RuntimeError("approved config approval_file does not resolve to the approval record")


def _expected_user_statement(*, candidate_name: str, request_fingerprint: str) -> str:
    return (
        "I explicitly approve WeightCLIP AE production candidate "
        f"{candidate_name} for approval request {request_fingerprint}."
    )


def _validate_user_approval(
    statement: str,
    approved_at_utc: str,
    *,
    candidate_name: str,
    request_fingerprint: str,
) -> None:
    expected_statement = _expected_user_statement(
        candidate_name=candidate_name, request_fingerprint=request_fingerprint
    )
    if statement != expected_statement:
        raise RuntimeError(
            "explicit user approval statement must exactly affirm the bound candidate and full request fingerprint"
        )
    try:
        parsed = datetime.fromisoformat(approved_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("approved_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("approved_at_utc must include an explicit timezone")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError("approved_at_utc must use UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if approved_at_utc != canonical:
        raise RuntimeError(f"approved_at_utc must use canonical UTC form: {canonical}")


def _expected_output_paths(
    *, canonical_config_path: Path, request_fingerprint: str
) -> tuple[Path, Path]:
    stem = canonical_config_path.stem
    prefix = request_fingerprint[:16]
    return (
        canonical_config_path.parent / f"{stem}.approved-{prefix}.json",
        canonical_config_path.parent / f"{stem}.user-approval-{prefix}.json",
    )


def _approval_record(
    *,
    request: Mapping[str, Any],
    request_path: Path,
    canonical_config_path: Path,
    approved_config_path: Path,
    approval_path: Path,
    user_statement: str,
    approved_at_utc: str,
) -> dict[str, Any]:
    requirements = _requirements_from_request(
        {**request, "_validated_path": str(request_path.resolve())},
        approved_config_sha256=sha256_file(approved_config_path),
    )
    record = approval_template(requirements)
    record.update(
        {
            "schema": PRODUCTION_APPROVAL_RECORD_SCHEMA,
            "status": "approved",
            "approval_scope": PRODUCTION_APPROVAL_SCOPE,
            "approved_by_user": True,
            "production_launch_authorized": True,
            "approval_record_path": str(approval_path.resolve()),
            "pending_request": {
                "path": str(request_path.resolve()),
                "sha256": requirements.approval_request_sha256,
                "fingerprint_sha256": requirements.approval_request_fingerprint_sha256,
            },
            "canonical_config": {
                "path": str(canonical_config_path.resolve()),
                "sha256": requirements.canonical_config_sha256,
            },
            "approved_config": {
                "path": str(approved_config_path.resolve()),
                "sha256": requirements.approved_config_sha256,
            },
            "historical_comparison": dict(request["historical_comparison"]),
            "user_approval": {
                "statement": user_statement,
                "approved_at_utc": approved_at_utc,
                "supplied_explicitly_by_caller": True,
            },
            "note": "Explicit user approval of this exact recomputed artifact chain.",
        }
    )
    return record


def materialize_approved_transition(
    *,
    approval_request_path: Path,
    canonical_config_path: Path,
    approved_config_path: Path,
    approval_path: Path,
    user_statement: str,
    approved_at_utc: str | None = None,
    explicit_user_approval: bool = False,
) -> tuple[Path, Path]:
    if not explicit_user_approval:
        raise RuntimeError("approval transition requires an explicit user-approval confirmation flag")
    if approved_at_utc is None:
        raise RuntimeError("approved_at_utc must be supplied explicitly for an idempotent approval")
    request_path = approval_request_path.resolve()
    canonical_path = canonical_config_path.resolve()
    approved_path = approved_config_path.resolve()
    record_path = approval_path.resolve()
    request = validate_pending_request(request_path)
    request_fingerprint = str(request["approval_request_fingerprint_sha256"])
    candidate_name = str(request["approval"]["candidate_name"])
    _validate_user_approval(
        user_statement,
        approved_at_utc,
        candidate_name=candidate_name,
        request_fingerprint=request_fingerprint,
    )
    bound_canonical = Path(request["inputs"]["config"]["path"]).resolve()
    if canonical_path != bound_canonical:
        raise RuntimeError("canonical config path differs from the pending approval request")
    canonical_sha = sha256_file(canonical_path)
    if canonical_sha != request["inputs"]["config"]["sha256"]:
        raise RuntimeError("canonical config changed after the pending approval request")
    if approved_path.parent != canonical_path.parent or record_path.parent != canonical_path.parent:
        raise RuntimeError(
            "approved config and approval record must be materialized beside the canonical config "
            "to preserve every relative-path meaning"
        )
    expected_approved, expected_record = _expected_output_paths(
        canonical_config_path=canonical_path, request_fingerprint=request_fingerprint
    )
    if approved_path != expected_approved or record_path != expected_record:
        raise RuntimeError(
            "approval transition outputs must use the exact request-derived paths: "
            f"approved={expected_approved} approval={expected_record}"
        )
    if approved_path in {canonical_path, record_path} or record_path == canonical_path:
        raise RuntimeError("canonical, approved-config, and approval-record paths must be distinct")
    canonical = _load_yaml(canonical_path)
    approved = dict(canonical)
    approved.update(
        {
            "status": APPROVED_CONFIG_STATUS,
            "production_launch_enabled_after_approval": True,
            "selected_candidate": candidate_name,
            "approval_file": record_path.name,
        }
    )
    _assert_approved_config_diff(
        canonical,
        approved,
        candidate_name=candidate_name,
        approval_path=record_path,
        approved_config_path=approved_path,
    )
    write_json_immutable(approved_path, approved)
    record = _approval_record(
        request=request,
        request_path=request_path,
        canonical_config_path=canonical_path,
        approved_config_path=approved_path,
        approval_path=record_path,
        user_statement=user_statement,
        approved_at_utc=approved_at_utc,
    )
    write_json_immutable(record_path, record)
    if sha256_file(canonical_path) != canonical_sha:
        raise RuntimeError("canonical config changed while materializing the approval transition")
    validate_approved_transition(
        approval_request_path=request_path,
        approved_config_path=approved_path,
        approval_path=record_path,
    )
    return approved_path, record_path


def validate_approved_transition(
    *,
    approval_request_path: Path,
    approved_config_path: Path,
    approval_path: Path,
) -> dict[str, Any]:
    request_path = approval_request_path.resolve()
    approved_path = approved_config_path.resolve()
    record_path = approval_path.resolve()
    request = validate_pending_request(request_path)
    canonical_path = Path(request["inputs"]["config"]["path"]).resolve()
    if approved_path.parent != canonical_path.parent or record_path.parent != canonical_path.parent:
        raise RuntimeError(
            "approved config and approval record are no longer beside the canonical config"
        )
    expected_approved, expected_record = _expected_output_paths(
        canonical_config_path=canonical_path,
        request_fingerprint=str(request["approval_request_fingerprint_sha256"]),
    )
    if approved_path != expected_approved or record_path != expected_record:
        raise RuntimeError("approval transition artifact paths are not request-derived")
    if sha256_file(canonical_path) != request["inputs"]["config"]["sha256"]:
        raise RuntimeError("canonical config changed after approval request materialization")
    canonical = _load_yaml(canonical_path)
    approved = _load_yaml(approved_path)
    writable_outputs = [
        str(path)
        for path in (approved_path, record_path)
        if path.stat().st_mode & 0o222
    ]
    if writable_outputs:
        raise RuntimeError(f"approval transition artifacts are not immutable: {writable_outputs}")
    candidate_name = str(request["approval"]["candidate_name"])
    _assert_approved_config_diff(
        canonical,
        approved,
        candidate_name=candidate_name,
        approval_path=record_path,
        approved_config_path=approved_path,
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise RuntimeError("approval record is not a mapping")
    user_approval = record.get("user_approval", {})
    _validate_user_approval(
        str(user_approval.get("statement", "")),
        str(user_approval.get("approved_at_utc", "")),
        candidate_name=str(request["approval"]["candidate_name"]),
        request_fingerprint=str(request["approval_request_fingerprint_sha256"]),
    )
    if user_approval.get("supplied_explicitly_by_caller") is not True:
        raise RuntimeError("approval record lacks explicit caller confirmation")
    expected = _approval_record(
        request=request,
        request_path=request_path,
        canonical_config_path=canonical_path,
        approved_config_path=approved_path,
        approval_path=record_path,
        user_statement=str(user_approval["statement"]),
        approved_at_utc=str(user_approval["approved_at_utc"]),
    )
    if dict(record) != expected:
        raise RuntimeError("approval record does not recompute from its exact request/config chain")
    return {
        "request": request,
        "canonical_config_path": canonical_path,
        "canonical_config_sha256": sha256_file(canonical_path),
        "approved_config": approved,
        "approved_config_path": approved_path,
        "approved_config_sha256": sha256_file(approved_path),
        "approval": dict(record),
        "approval_path": record_path,
        "approval_sha256": sha256_file(record_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-request", type=Path, required=True)
    parser.add_argument("--canonical-config", type=Path, required=True)
    parser.add_argument("--approved-config", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--user-statement", required=True)
    parser.add_argument("--approved-at-utc", required=True)
    parser.add_argument("--confirm-explicit-user-approval", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    approved_path, record_path = materialize_approved_transition(
        approval_request_path=args.approval_request,
        canonical_config_path=args.canonical_config,
        approved_config_path=args.approved_config,
        approval_path=args.approval_file,
        user_statement=args.user_statement,
        approved_at_utc=args.approved_at_utc,
        explicit_user_approval=args.confirm_explicit_user_approval,
    )
    print(
        "[weightclip-ae-approval] status=approved "
        f"config={approved_path} config_sha256={sha256_file(approved_path)} "
        f"approval={record_path} approval_sha256={sha256_file(record_path)}"
    )


if __name__ == "__main__":
    main()
