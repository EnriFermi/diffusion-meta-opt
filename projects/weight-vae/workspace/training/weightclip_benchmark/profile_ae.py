from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from big_vae.weightclip_benchmark.ae_candidate_config import (
    HYDRA_TRAIN_CONFIG_NAME,
    ae_source_implementation_seal,
    resolve_ae_candidate,
)
from big_vae.weightclip_benchmark.ae_scaling import (
    ApprovalRequirements,
    ScalingAxes,
    apply_scaling_axes,
    approval_template,
    build_candidate_profile,
    canonical_fingerprint,
)
from big_vae.weightclip_benchmark.manifests import sha256_file, write_json_immutable


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (relative_to / path).resolve()


def _write_text_immutable(path: Path, body: str) -> str:
    payload = body.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable text artifact differs: {path}")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _legacy_supersession(
    output_dir: Path,
    profiles: list[dict[str, Any]],
    *,
    current_artifact_dir: Path,
    current_source_implementation_seal_sha256: str,
) -> dict[str, Any]:
    legacy_report = output_dir / "ae_candidate_profiles.json"
    legacy_profiles: list[dict[str, Any]] = []
    if legacy_report.is_file():
        payload = json.loads(legacy_report.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("profiles"), list):
            legacy_profiles = list(payload["profiles"])
    new_by_name = {
        str(profile["candidate"]["name"]): str(profile["model_fingerprint_sha256"])
        for profile in profiles
    }
    legacy_artifacts = []
    for path in sorted(output_dir.glob("ae_candidate_*")) + sorted(
        output_dir.glob("approval_PENDING_*.json")
    ):
        if path.is_file():
            legacy_artifacts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "supersession_reasons": ["legacy_unsealed_root_artifact"],
                }
            )
    for directory in sorted(output_dir.glob("candidate_profiles-*")):
        index = directory / "artifact_index.json"
        if directory.resolve() != current_artifact_dir.resolve() and index.is_file():
            reasons: list[str] = []
            try:
                prior_index = json.loads(index.read_text(encoding="utf-8"))
                report_ref = prior_index.get("report", {})
                report_path = Path(str(report_ref.get("path", ""))).resolve()
                prior_report = json.loads(report_path.read_text(encoding="utf-8"))
                if prior_report.get("fingerprint_source") != "exact_hydra_composed_worker_model":
                    reasons.append("noncanonical_model_resolution")
                if (
                    prior_report.get("source_implementation_seal_sha256")
                    != current_source_implementation_seal_sha256
                ):
                    reasons.append("source_implementation_seal_changed")
            except (json.JSONDecodeError, OSError, TypeError):
                reasons.append("unverifiable_prior_bundle")
            if not reasons:
                reasons.append("canonical_report_contract_changed")
            legacy_artifacts.append(
                {
                    "path": str(index.resolve()),
                    "sha256": sha256_file(index),
                    "supersession_reasons": reasons,
                }
            )
    reason_codes = sorted(
        {
            reason
            for artifact in legacy_artifacts
            for reason in artifact["supersession_reasons"]
        }
    )
    return {
        "status": "supersedes_prior_candidate_artifacts",
        "reason": (
            "Prior artifacts are superseded because their canonical report contract is not current. "
            "Per-artifact reason codes distinguish source-implementation drift from legacy "
            "model-resolution or unsealed-artifact causes."
        ),
        "reason_codes": reason_codes,
        "current_source_implementation_seal_sha256": current_source_implementation_seal_sha256,
        "legacy_artifacts": legacy_artifacts,
        "fingerprint_transitions": [
            {
                "candidate_name": str(row.get("candidate", {}).get("name", "")),
                "superseded_fingerprint_sha256": str(row.get("model_fingerprint_sha256", "")),
                "replacement_fingerprint_sha256": new_by_name.get(
                    str(row.get("candidate", {}).get("name", ""))
                ),
            }
            for row in legacy_profiles
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact meta-device profiler for WeightCLIP AE scaling candidates")
    parser.add_argument("--config", type=Path, default=Path("conf/weightclip_benchmark/ae_700m.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    cfg = _load_yaml(config_path)
    base_path = _resolve_path(str(cfg["base_model_config"]), relative_to=config_path.parent)
    base = _load_yaml(base_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _resolve_path(str(cfg["profile_output_dir"]), relative_to=config_path.parent)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_rows, tile_cols = map(int, cfg["tile_shape"])
    target = int(cfg["target_trainable_parameters"])
    tolerance = float(cfg["target_relative_tolerance"])
    steps = int(cfg["training_steps"])
    rate_contract = dict(cfg["representation_rate_contract"])
    mean_valid_weight_scalars = float(
        rate_contract["operator_mean_valid_weight_scalars_per_tile"]
    )

    print("[weightclip-ae-profile] stage=load-config", flush=True)
    print(
        "[weightclip-ae-profile] "
        f"config={config_path} base={base_path} device=meta dtype=parameter-native "
        f"seed=not-applicable cache_mode=none output_dir={output_dir}",
        flush=True,
    )
    started = time.monotonic()
    profiles: list[dict[str, Any]] = []
    source_implementation = ae_source_implementation_seal()
    for index, raw_candidate in enumerate(cfg["candidates"], start=1):
        axes = ScalingAxes.from_mapping(raw_candidate)
        print(
            f"[weightclip-ae-profile] stage=instantiate candidate={axes.name} "
            f"progress={index}/{len(cfg['candidates'])}",
            flush=True,
        )
        # Exact sequence is part of the approval contract: apply only explicit
        # axes, compose the production Hydra tree, then profile that resolved
        # model.  Never profile the PyYAML seed directly.
        candidate = resolve_ae_candidate(apply_scaling_axes(base, axes))
        profile = build_candidate_profile(
            candidate.resolved_model_config,
            axes,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            target_trainable_parameters=target,
            mean_valid_weight_scalars=mean_valid_weight_scalars,
        )
        if axes.num_latents != int(rate_contract["selected_num_latents"]) or axes.d_lat != int(
            rate_contract["selected_d_lat"]
        ):
            raise RuntimeError(
                f"candidate {axes.name} violates the shared WeightCLIP-matched bottleneck"
            )
        if profile["model_fingerprint_sha256"] != candidate.model_fingerprint_sha256:
            raise RuntimeError(f"candidate profile fingerprint drift for {axes.name}")
        if abs(float(profile["relative_trainable_parameter_error"])) > tolerance:
            raise RuntimeError(
                f"candidate {axes.name} violates target tolerance {tolerance}: "
                f"error={profile['relative_trainable_parameter_error']}"
            )
        profile["hydra_resolution"] = {
            "config_name": HYDRA_TRAIN_CONFIG_NAME,
            "model_overrides": list(candidate.hydra_model_overrides),
            "exact_worker_model_equality_required": True,
        }
        profile["input_hashes"] = {
            "candidate_config_sha256": sha256_file(config_path),
            "base_model_config_sha256": sha256_file(base_path),
            "resolved_model_config_sha256": candidate.model_fingerprint_sha256,
        }
        profiles.append(profile)
        ledger = profile["parameter_ledger"]
        print(
            f"[weightclip-ae-profile] candidate={axes.name} "
            f"trainable={ledger['trainable_parameters']:,} total={ledger['total_parameters']:,} "
            f"target_error={profile['relative_trainable_parameter_error']:+.3%} "
            f"fingerprint={profile['model_fingerprint_sha256']}",
            flush=True,
        )

    report_contract = {
        "status": "pending_user_architecture_scaling_approval",
        "schema_version": 2,
        "fingerprint_source": "exact_hydra_composed_worker_model",
        "approval_contract_schema_version": 2,
        "source_implementation_seal_sha256": source_implementation[
            "source_implementation_seal_sha256"
        ],
        "candidate_config_sha256": sha256_file(config_path),
        "base_model_config_sha256": sha256_file(base_path),
        "target_trainable_parameters": target,
        "target_relative_tolerance": tolerance,
        "tile_shape": [tile_rows, tile_cols],
        "training_steps": steps,
        "representation_rate_contract": rate_contract,
        "profiles": profiles,
    }
    artifact_set_fingerprint = canonical_fingerprint(report_contract)
    artifact_dir = output_dir / f"candidate_profiles-{artifact_set_fingerprint[:16]}"
    report = {
        **report_contract,
        "artifact_set_fingerprint_sha256": artifact_set_fingerprint,
        "config_path": str(config_path),
        "base_model_config_path": str(base_path),
    }
    report_path = artifact_dir / "ae_candidate_profiles.json"
    report_sha256 = write_json_immutable(report_path, report)

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "candidate_name",
            "trainable_parameters",
            "total_parameters",
            "frozen_parameters",
            "relative_target_error",
            "latent_scalars",
            "valid_scalars_per_latent_scalar",
            "model_fingerprint_sha256",
        ],
    )
    writer.writeheader()
    for profile in profiles:
        ledger = profile["parameter_ledger"]
        rate = profile["representation_rate"]
        writer.writerow(
            {
                "candidate_name": profile["candidate"]["name"],
                "trainable_parameters": ledger["trainable_parameters"],
                "total_parameters": ledger["total_parameters"],
                "frozen_parameters": ledger["frozen_parameters"],
                "relative_target_error": profile["relative_trainable_parameter_error"],
                "latent_scalars": rate["latent_scalars"],
                "valid_scalars_per_latent_scalar": rate["valid_scalars_per_latent_scalar"],
                "model_fingerprint_sha256": profile["model_fingerprint_sha256"],
            }
        )
    csv_path = artifact_dir / "ae_candidate_summary.csv"
    csv_sha256 = _write_text_immutable(csv_path, csv_buffer.getvalue())

    approval_refs = []
    for profile in profiles:
        axes_name = str(profile["candidate"]["name"])
        template = approval_template(
            ApprovalRequirements(
                candidate_name=axes_name,
                model_fingerprint_sha256=str(profile["model_fingerprint_sha256"]),
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                training_steps=steps,
                candidate_artifact_set_sha256=artifact_set_fingerprint,
                candidate_report_sha256=report_sha256,
                trainable_parameters=int(profile["parameter_ledger"]["trainable_parameters"]),
                source_implementation_seal_sha256=str(
                    source_implementation["source_implementation_seal_sha256"]
                ),
            )
        )
        approval_path = artifact_dir / f"approval_PENDING_{axes_name}.json"
        approval_refs.append(
            {"path": str(approval_path.resolve()), "sha256": write_json_immutable(approval_path, template)}
        )
    supersession_path = artifact_dir / "supersession.json"
    supersession_sha256 = write_json_immutable(
        supersession_path,
        _legacy_supersession(
            output_dir,
            profiles,
            current_artifact_dir=artifact_dir,
            current_source_implementation_seal_sha256=str(
                source_implementation["source_implementation_seal_sha256"]
            ),
        ),
    )
    source_seal_path = artifact_dir / "source_implementation_seal.json"
    source_seal_artifact_sha256 = write_json_immutable(source_seal_path, source_implementation)
    index = {
        "schema_version": 1,
        "artifact_set_fingerprint_sha256": artifact_set_fingerprint,
        "report": {"path": str(report_path.resolve()), "sha256": report_sha256},
        "summary": {"path": str(csv_path.resolve()), "sha256": csv_sha256},
        "pending_approvals": approval_refs,
        "supersession": {"path": str(supersession_path.resolve()), "sha256": supersession_sha256},
        "source_implementation": {
            "path": str(source_seal_path.resolve()),
            "sha256": source_seal_artifact_sha256,
            "source_implementation_seal_sha256": source_implementation[
                "source_implementation_seal_sha256"
            ],
        },
    }
    index_path = artifact_dir / "artifact_index.json"
    index_sha256 = write_json_immutable(index_path, index)
    print(
        f"[weightclip-ae-profile] stage=complete elapsed_s={time.monotonic() - started:.2f} "
        f"report={report_path} report_sha256={report_sha256} summary={csv_path} "
        f"index={index_path} index_sha256={index_sha256} status=pending-user-approval",
        flush=True,
    )


if __name__ == "__main__":
    main()
