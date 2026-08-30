#!/usr/bin/env python3
"""Expand a sealed arm specification into an immutable evaluation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
import torch

from big_vae.weightclip_benchmark.contract import CandidateProtocol, DEFAULT_CONTRACT
from big_vae.weightclip_benchmark.coverage import validate_candidate_cardinality, validate_exact_grid
from big_vae.weightclip_benchmark.evaluation import method_policies
from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    load_ood_conditioning_bundle,
    load_task_fit_bundle,
)
from big_vae.weightclip_benchmark.manifests import write_json_immutable, write_records_immutable
from big_vae.weightclip_benchmark.official_bridge import redact_secrets


def _candidate_count(protocol: CandidateProtocol) -> int:
    if protocol == CandidateProtocol.CONTROLLED_SINGLE:
        return 1
    if protocol == CandidateProtocol.CONTROLLED_VALIDATION_BEST_K:
        return DEFAULT_CONTRACT.evaluation.controlled_candidate_count
    if protocol == CandidateProtocol.NATIVE_TEST_TOP5_ORACLE:
        return DEFAULT_CONTRACT.evaluation.native_candidate_count
    raise AssertionError(protocol)


def _resolve(value: Any, *, dataset: str) -> Any:
    if isinstance(value, Mapping) and set(value) == {"artifact_path"}:
        path = str(value["artifact_path"]).format(dataset=dataset)
        reference = ArtifactRef.create(path)
        return {"path": reference.path, "sha256": reference.sha256, "bytes": reference.bytes}
    if isinstance(value, Mapping):
        return {str(key): _resolve(item, dataset=dataset) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, dataset=dataset) for item in value]
    if isinstance(value, str):
        return value.format(dataset=dataset)
    return value


def build_candidate_rows(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("candidate spec requires a nonempty arms list")
    for arm in arms:
        method = str(arm["method"])
        protocol = CandidateProtocol(str(arm["protocol"]))
        head, bn = method_policies(method)
        required = bool(arm.get("required", True))
        count = _candidate_count(protocol)
        configured_count = int(arm.get("candidate_count", count))
        if configured_count != count:
            raise ValueError(f"{method}/{protocol.value} requires exactly {count} candidates, got {configured_count}")
        for dataset in arm["datasets"]:
            try:
                base_payload = _resolve(arm.get("payload", {}), dataset=str(dataset))
                override = _resolve(arm.get("dataset_payload", {}).get(str(dataset), {}), dataset=str(dataset))
            except FileNotFoundError as exc:
                if required:
                    raise
                skipped.append({"method": method, "dataset": str(dataset), "reason": str(exc)})
                continue
            payload = dict(base_payload) | dict(override)
            if "method_kind" in payload and payload["method_kind"] != method:
                raise ValueError(f"payload method_kind disagrees with arm method {method}")
            payload["method_kind"] = method
            payload["dataset"] = str(dataset)
            code_candidates = payload.get("code_candidates")
            needs_indexed_codes = count > 1 and method.startswith("weightclip_commonzoo_fullwindow_")
            if needs_indexed_codes:
                if not isinstance(code_candidates, list) or len(code_candidates) != count:
                    raise ValueError(f"{method} requires exactly {count} executed prompt/code artifacts")
                hashes = [str(item.get("sha256")) for item in code_candidates]
                if len(set(hashes)) != count:
                    raise ValueError(f"{method} candidate artifact references are duplicated")
                prompt_hashes: list[str] = []
                tensor_hashes: list[str] = []
                for item in code_candidates:
                    code_payload = torch.load(
                        ArtifactRef.from_mapping(item).verify(f"{method} native code candidate"),
                        map_location="cpu",
                        weights_only=False,
                    )
                    producer = code_payload.get("producer", {}) if isinstance(code_payload, Mapping) else {}
                    prompt_hashes.append(str(producer.get("prompt_set_sha256", "")))
                    tensor_hashes.append(str(producer.get("code_tensor_sha256", "")))
                if "" in prompt_hashes or len(set(prompt_hashes)) != count:
                    raise ValueError(f"{method} requires {count} distinct precommitted prompt-index sets")
                if "" in tensor_hashes:
                    raise ValueError(f"{method} candidate artifacts omit code tensor hashes")
                native_unique_code_count = len(set(tensor_hashes))
                native_duplicate_rate = 1.0 - len(set(tensor_hashes)) / len(tensor_hashes)
            else:
                _validate_arm_payload(method, payload)
            candidate_prompt_sets = None
            if "conditioning_bundle" in payload:
                if count == 1 and method.startswith("weightclip_") and method != "weightclip_flow" and "code" in payload:
                    code_payload = torch.load(
                        ArtifactRef.from_mapping(payload["code"]).verify(f"{method} produced code"),
                        map_location="cpu",
                        weights_only=False,
                    )
                    candidate_prompt_sets = [code_payload.get("producer", {}).get("prompt_indices")]
                else:
                    conditioning = load_ood_conditioning_bundle(
                        ArtifactRef.from_mapping(payload["conditioning_bundle"]).path
                    )
                    prompt_provenance = conditioning.get("dataset_prompt_provenance", {})
                    candidate_prompt_sets = prompt_provenance.get("candidate_image_indices")
                    if candidate_prompt_sets is None and count == 1:
                        candidate_prompt_sets = [prompt_provenance.get("image_indices")]
                if not isinstance(candidate_prompt_sets, list) or len(candidate_prompt_sets) < count:
                    raise ValueError(f"{method} requires at least {count} precommitted prompt subsets")
            elif "task_bundle" in payload:
                task_bundle = load_task_fit_bundle(ArtifactRef.from_mapping(payload["task_bundle"]).path)
                prompt_provenance = task_bundle.get("dataset_embedding_bank_provenance", {})
                candidate_prompt_sets = prompt_provenance.get("candidate_indices")
                if not isinstance(candidate_prompt_sets, list) or len(candidate_prompt_sets) < count:
                    raise ValueError(f"{method} paired flow requires at least {count} precommitted prompt subsets")
            for evaluation_seed in arm["evaluation_seeds"]:
                for candidate_index in range(count):
                    candidate_seed = int(evaluation_seed) * 1_000_003 + candidate_index
                    candidate_payload = dict(payload) | {"seed": candidate_seed}
                    if candidate_prompt_sets is not None:
                        prompt_indices = candidate_prompt_sets[candidate_index]
                        candidate_payload.update(
                            {
                                "prompt_candidate_index": candidate_index,
                                "prompt_indices": prompt_indices,
                                "prompt_set_sha256": hashlib.sha256(
                                    json.dumps(prompt_indices, separators=(",", ":")).encode()
                                ).hexdigest(),
                            }
                        )
                    if needs_indexed_codes:
                        candidate_payload.pop("code_candidates", None)
                        candidate_payload["code"] = code_candidates[candidate_index]
                        _validate_arm_payload(method, candidate_payload)
                    rows.append(
                        {
                            "method": method,
                            "dataset": str(dataset),
                            "evaluation_seed": int(evaluation_seed),
                            "protocol": protocol.value,
                            "candidate_id": (
                                f"{method}:{dataset}:e{evaluation_seed}:"
                                f"p{protocol.value}:c{candidate_index:03d}"
                            ),
                            "payload": candidate_payload,
                            "head_policy": head.value,
                            "batchnorm_policy": bn.value,
                            "head_module_path": "fc",
                            "generation_seconds": 0.0,
                            "generation_nfe": int(payload.get("nfe_steps", 0)) * (2 if payload.get("solver", "heun") == "heun" else 1),
                            "metadata": {
                                "candidate_index": candidate_index,
                                "manifest_generated": True,
                                **(
                                    {
                                        "effective_unique_code_count": native_unique_code_count,
                                        "code_duplicate_rate": native_duplicate_rate,
                                    }
                                    if needs_indexed_codes
                                    else {}
                                ),
                            },
                        }
                    )
    if not rows:
        raise ValueError("candidate spec produced no runnable rows")
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate spec produced globally duplicate candidate_id values")
    return rows, skipped


def _validate_arm_payload(method: str, payload: Mapping[str, Any]) -> None:
    if method in {"ours_flow", "weightclip_flow", "ours_flow_oracle_anchor", "weightclip_flow_oracle_anchor"}:
        path_kind = str(payload.get("path_kind", ""))
        if path_kind == "gaussian":
            if "conditioning_bundle" not in payload or "task_bundle" in payload or "anchors" in payload:
                raise ValueError(f"anchor-free {method} requires only a target-weight-free conditioning_bundle")
        elif path_kind == "paired_anchor":
            if "task_bundle" not in payload or "conditioning_bundle" in payload:
                raise ValueError(f"paired {method} requires only a target task_bundle")
        else:
            raise ValueError(f"{method} has unsupported path_kind {path_kind!r}")
    elif method == "ours_reconstruction":
        if "task_bundle" not in payload or "conditioning_bundle" in payload:
            raise ValueError("ours reconstruction requires a task_bundle")
    elif method.startswith("weightclip_"):
        if "conditioning_bundle" not in payload or "task_bundle" in payload:
            raise ValueError(f"controlled OOD method {method} requires a conditioning_bundle and forbids task_bundle")
        if method not in {"weightclip_flow"}:
            code_ref = ArtifactRef.from_mapping(payload.get("code", {}))
            code_payload = torch.load(code_ref.verify(f"{method} produced code"), map_location="cpu", weights_only=False)
            producer = code_payload.get("producer", {}) if isinstance(code_payload, Mapping) else {}
            expected_mode = {
                "weightclip_commonzoo_fullwindow_ridge": "ridge",
                "weightclip_commonzoo_fullwindow_memory": "memory",
                "weightclip_commonzoo_fullwindow_nearest_code": "nearest_code",
                "weightclip_commonzoo_fullwindow_ridge_native_oracle": "ridge_native_oracle",
                "weightclip_commonzoo_fullwindow_memory_native_oracle": "memory_native_oracle",
                "weightclip_commonzoo_fullwindow_nearest_code_native_oracle": "nearest_code_native_oracle",
            }.get(method)
            if (
                producer.get("execution_status") != "complete"
                or producer.get("substrate_label")
                != "WeightCLIP codec + common-zoo full-window prior"
                or producer.get("mode") != expected_mode
                or str(code_payload.get("dataset_id")) != str(payload["dataset"])
            ):
                raise ValueError(f"{method} code lacks an executed, dataset-bound multi-window producer")
    elif method in {"anchor", "anchor_untouched"}:
        if "checkpoint" not in payload:
            raise ValueError("anchor method requires a checkpoint artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows, skipped = build_candidate_rows(spec)
    validate_candidate_cardinality(rows)
    expected_grid_ref = ArtifactRef.create(spec["expected_grid"])
    expected_grid_payload = __import__("json").loads(Path(expected_grid_ref.path).read_text(encoding="utf-8"))
    expected_cells = expected_grid_payload.get("cells", expected_grid_payload)
    if not isinstance(expected_cells, list):
        raise TypeError("expected grid artifact must be a list or contain cells")
    validate_exact_grid(rows, expected_cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = write_records_immutable(args.output.with_suffix(""), rows)
    index = {
        "schema_version": 1,
        "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
        "rows": len(rows),
        "files": paths,
        "skipped_optional_arms": skipped,
        "expected_grid": {
            "path": expected_grid_ref.path,
            "sha256": expected_grid_ref.sha256,
            "bytes": expected_grid_ref.bytes,
        },
        "resolved_spec": redact_secrets(spec),
    }
    write_json_immutable(args.output.with_suffix(".index.json"), index)
    print(f"[candidate-manifest:done] rows={len(rows)} skipped={len(skipped)} files={paths}")


if __name__ == "__main__":
    main()
