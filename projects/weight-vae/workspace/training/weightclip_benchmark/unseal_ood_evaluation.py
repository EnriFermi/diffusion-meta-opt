#!/usr/bin/env python3
"""Create the explicit user-authorized OOD test-unseal after all priors freeze."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from big_vae.weightclip_benchmark.contract import DEFAULT_CONTRACT
from big_vae.weightclip_benchmark.coverage import (
    canonical_final_grid_cells,
    validate_candidate_cardinality,
    validate_exact_grid,
)
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef
from big_vae.weightclip_benchmark.manifests import write_json_immutable


_FLOW_METHODS = {
    "ours_flow": ("ours", "gaussian"),
    "ours_flow_oracle_anchor": ("ours", "paired_anchor"),
    "weightclip_flow": ("weightclip", "gaussian"),
    "weightclip_flow_oracle_anchor": ("weightclip", "paired_anchor"),
}


def validate_final_grid_payload(payload: dict[str, object]) -> None:
    if (
        payload.get("kind") != "frozen_expected_evaluation_grid"
        or payload.get("benchmark_tier") != "final"
        or payload.get("contract_fingerprint") != DEFAULT_CONTRACT.fingerprint()
    ):
        raise ValueError("OOD test unseal accepts only the final paper benchmark expected grid")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError("final paper benchmark expected grid has no canonical cells")
    validate_exact_grid(cells, canonical_final_grid_cells())


def validate_unseal_bindings(
    *,
    flow_seal_paths: list[Path],
    candidate_rows: list[dict[str, object]],
    ae_seal_ref: ArtifactRef,
) -> list[ArtifactRef]:
    if len(flow_seal_paths) != 4:
        raise ValueError("OOD unseal requires exactly four frozen flow seals")
    references = [ArtifactRef.create(path) for path in flow_seal_paths]
    if len({reference.sha256 for reference in references}) != 4:
        raise ValueError("OOD unseal flow seals must be four distinct artifacts")
    seals: dict[tuple[str, str], tuple[ArtifactRef, dict[str, object]]] = {}
    for reference in references:
        payload = json.loads(Path(reference.path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != 4:
            raise ValueError("OOD unseal requires flow seal schema v4")
        key = (str(payload.get("codec")), str(payload.get("path_kind")))
        if key in seals:
            raise ValueError(f"duplicate flow seal arm {key}")
        seals[key] = (reference, payload)
    expected = set(_FLOW_METHODS.values())
    if set(seals) != expected:
        raise ValueError(f"flow seals must cover exact four arms: got={sorted(seals)}")
    solvers = {(str(payload["solver"]), int(payload["nfe_steps"])) for _, payload in seals.values()}
    global_selections = {
        str(payload.get("global_e4_selection_artifact", {}).get("sha256"))
        for _, payload in seals.values()
    }
    if len(solvers) != 1 or len(global_selections) != 1 or "None" in global_selections:
        raise ValueError("four flow seals do not share one frozen global solver/NFE selection")
    observed_flow_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        method = str(row.get("method"))
        if method not in _FLOW_METHODS:
            continue
        key = _FLOW_METHODS[method]
        observed_flow_keys.add(key)
        reference, seal = seals[key]
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("flow candidate row has no payload mapping")
        if ArtifactRef.from_mapping(payload.get("flow_seal", {})) != reference:
            raise ValueError(f"candidate {method} points to an unsealed flow seal")
        expected_checkpoint = ArtifactRef.create(Path(reference.path).parent / str(seal["checkpoint_name"]))
        expected_normalizer = ArtifactRef.create(Path(reference.path).parent / str(seal["normalizer_name"]))
        if ArtifactRef.from_mapping(payload.get("flow_checkpoint", {})) != expected_checkpoint:
            raise ValueError(f"candidate {method} flow checkpoint disagrees with seal")
        if ArtifactRef.from_mapping(payload.get("normalizer", {})) != expected_normalizer:
            raise ValueError(f"candidate {method} normalizer disagrees with seal")
        if payload.get("solver") != seal["solver"] or int(payload.get("nfe_steps", -1)) != int(seal["nfe_steps"]):
            raise ValueError(f"candidate {method} solver/NFE disagrees with seal")
        if key[0] == "ours" and ArtifactRef.from_mapping(payload.get("codec_seal", {})) != ae_seal_ref:
            raise ValueError("ours flow candidate points to a different AE codec seal")
    if observed_flow_keys != expected:
        raise ValueError(f"candidate manifest omits sealed flow arms: {sorted(expected - observed_flow_keys)}")
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-grid", type=Path, required=True)
    parser.add_argument("--ae-seal", type=Path, required=True)
    parser.add_argument("--flow-seal", type=Path, action="append", required=True)
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    approval = json.loads(args.approval_file.read_text(encoding="utf-8"))
    expected = {
        "status": "approved",
        "approval_scope": "weightclip_ood_test_unseal",
        "approved_by_user": True,
        "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("OOD unseal approval does not match the frozen contract")
    candidate_rows = [
        json.loads(line) for line in args.candidate_manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    validate_candidate_cardinality(candidate_rows)
    ae_seal_ref = ArtifactRef.create(args.ae_seal)
    flow_seal_refs = validate_unseal_bindings(
        flow_seal_paths=[path.resolve() for path in args.flow_seal],
        candidate_rows=candidate_rows,
        ae_seal_ref=ae_seal_ref,
    )
    grid_payload = json.loads(args.expected_grid.read_text(encoding="utf-8"))
    validate_final_grid_payload(grid_payload)
    validate_exact_grid(candidate_rows, grid_payload.get("cells", grid_payload))
    payload = {
        "schema_version": 1,
        "status": "OOD_TEST_UNLOCKED",
        "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
        "evaluation_config": asdict(ArtifactRef.create(args.evaluation_config)),
        "candidate_manifest": asdict(ArtifactRef.create(args.candidate_manifest)),
        "expected_grid": asdict(ArtifactRef.create(args.expected_grid)),
        "ae_seal": asdict(ae_seal_ref),
        "flow_seals": [asdict(reference) for reference in flow_seal_refs],
        "approval": asdict(ArtifactRef.create(args.approval_file)),
        "candidate_protocols_frozen": [
            "controlled_single", "controlled_validation_best_k", "native_test_top5_oracle"
        ],
    }
    digest = write_json_immutable(args.output, payload)
    print(f"[ood-unseal:done] path={args.output.resolve()} sha256={digest}")


if __name__ == "__main__":
    main()
