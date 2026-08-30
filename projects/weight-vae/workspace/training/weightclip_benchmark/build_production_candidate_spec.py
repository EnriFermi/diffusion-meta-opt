#!/usr/bin/env python3
"""Resolve the complete Stage-G arm table from sealed upstream artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from big_vae.weightclip_benchmark.contract import DEFAULT_CONTRACT
from big_vae.weightclip_benchmark.coverage import (
    FINAL_EVALUATION_SEEDS,
    canonical_final_grid_cells,
    canonical_grid_cells,
)
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef
from big_vae.weightclip_benchmark.manifests import write_json_immutable
from training.weightclip_benchmark.prepare_stage_dg_configs import _write_yaml_immutable


EXPLORATORY_EVALUATION_SEEDS = (0,)


def _ref(path: str | Path) -> dict[str, Any]:
    return asdict(ArtifactRef.create(path))


def _flow_payload(seal_path: Path) -> dict[str, Any]:
    seal_ref = ArtifactRef.create(seal_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    checkpoint = seal_path.parent / str(seal["checkpoint_name"])
    normalizer = seal_path.parent / str(seal["normalizer_name"])
    return {
        "flow_seal": asdict(seal_ref),
        "flow_checkpoint": _ref(checkpoint),
        "normalizer": _ref(normalizer),
        "solver": str(seal["solver"]),
        "nfe_steps": int(seal["nfe_steps"]),
        "path_kind": str(seal["path_kind"]),
    }


def _code_refs(root: Path, mode: str, dataset: str, count: int) -> list[dict[str, Any]]:
    paths = [root / mode / dataset / f"candidate-{index:03d}.pt" for index in range(count)]
    return [_ref(path) for path in paths]


def _anchor_rows(path: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    index = {(str(row["dataset"]), int(row["evaluation_seed"])): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("anchor manifest contains duplicate dataset/evaluation-seed rows")
    return index


def _codec_bundle_rows(path: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("bundles", [])
    index = {(str(row["dataset"]), int(row["evaluation_seed"])): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("OOD codec-bundle index contains duplicate dataset/evaluation-seed rows")
    return index


def build_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    datasets = [str(item) for item in config["datasets"]]
    evaluation_seeds = [int(item) for item in config["evaluation_seeds"]]
    benchmark_tier = str(config.get("benchmark_tier", "final"))
    if set(datasets) != set(DEFAULT_CONTRACT.ood_datasets):
        raise ValueError("candidate-spec datasets disagree with frozen OOD contract")
    if benchmark_tier == "final":
        if tuple(evaluation_seeds) != FINAL_EVALUATION_SEEDS:
            raise ValueError(f"final benchmark requires paired evaluation seeds {FINAL_EVALUATION_SEEDS}")
        generative_protocols = (
            "controlled_single", "controlled_validation_best_k", "native_test_top5_oracle",
        )
    elif benchmark_tier == "exploratory_single_seed0":
        if tuple(evaluation_seeds) != EXPLORATORY_EVALUATION_SEEDS:
            raise ValueError("exploratory_single_seed0 requires exactly evaluation seed [0]")
        generative_protocols = ("controlled_single",)
    else:
        raise ValueError(f"unknown benchmark_tier {benchmark_tier!r}")
    ae_seal = _ref(config["ae_seal"])
    official_checkpoint = _ref(config["official_weightclip"]["checkpoint_path"])
    official_encoder = _ref(config["official_weightclip"]["dataset_encoder_path"])
    flow = {
        name: _flow_payload(Path(path).resolve())
        for name, path in config["flow_seals"].items()
    }
    expected_flow_keys = {
        "ours_anchor_free", "ours_oracle_anchor",
        "weightclip_anchor_free", "weightclip_oracle_anchor",
    }
    if set(flow) != expected_flow_keys:
        raise ValueError(f"candidate-spec flow seals must be exactly {sorted(expected_flow_keys)}")
    conditioning_root = Path(config["conditioning_bundle_root"]).resolve()
    code_root = Path(config["weightclip_code_root"]).resolve()
    anchor_rows = _anchor_rows(Path(config["anchor_manifest"]).resolve())
    codec_rows = _codec_bundle_rows(Path(config["anchor_codec_bundle_index"]).resolve())
    arms: list[dict[str, Any]] = []

    def add(
        method: str,
        protocol: str,
        dataset: str,
        seed: int,
        payload: Mapping[str, Any],
    ) -> None:
        arms.append(
            {
                "method": method,
                "protocol": protocol,
                "candidate_count": (
                    1
                    if protocol == "controlled_single"
                    else DEFAULT_CONTRACT.evaluation.native_candidate_count
                    if protocol == "native_test_top5_oracle"
                    else DEFAULT_CONTRACT.evaluation.controlled_candidate_count
                ),
                "datasets": [dataset],
                "evaluation_seeds": [seed],
                "required": True,
                "payload": dict(payload),
            }
        )

    for dataset in datasets:
        conditioning = _ref(conditioning_root / f"{dataset}.pt")
        for seed in evaluation_seeds:
            anchor_key = (dataset, seed)
            if anchor_key not in anchor_rows or anchor_key not in codec_rows:
                raise ValueError(f"missing target anchor/codec bundle for {anchor_key}")
            anchor = anchor_rows[anchor_key]
            codec = codec_rows[anchor_key]
            common = {"num_classes": int(config["num_classes"][dataset]), "width_mult": 0.5}
            add("scratch", "controlled_single", dataset, seed, common)
            add("anchor_untouched", "controlled_single", dataset, seed, {"checkpoint": anchor["checkpoint"]})
            add("anchor", "controlled_single", dataset, seed, {"checkpoint": anchor["checkpoint"]})
            ours_common = {
                **flow["ours_anchor_free"],
                "conditioning_bundle": conditioning,
                "codec_seal": ae_seal,
                "activation_conditioning": True,
            }
            wc_common = {
                **flow["weightclip_anchor_free"],
                "conditioning_bundle": conditioning,
                "official_checkpoint": official_checkpoint,
                "official_dataset_encoder": official_encoder,
            }
            for protocol in generative_protocols:
                add("ours_flow", protocol, dataset, seed, ours_common)
                add("weightclip_flow", protocol, dataset, seed, wc_common)
            for protocol in generative_protocols:
                add(
                    "ours_flow_oracle_anchor",
                    protocol,
                    dataset,
                    seed,
                    {
                        **flow["ours_oracle_anchor"], "task_bundle": codec["ours"]["bundle"],
                        "codec_seal": ae_seal, "activation_conditioning": True,
                    },
                )
                add(
                    "weightclip_flow_oracle_anchor",
                    protocol,
                    dataset,
                    seed,
                    {
                        **flow["weightclip_oracle_anchor"], "task_bundle": codec["weightclip"]["bundle"],
                        "official_checkpoint": official_checkpoint,
                        "official_dataset_encoder": official_encoder,
                    },
                )
            for stem in ("ridge", "memory", "nearest_code"):
                method = f"weightclip_commonzoo_fullwindow_{stem}"
                controlled_code_count = (
                    DEFAULT_CONTRACT.evaluation.controlled_candidate_count
                    if benchmark_tier == "final"
                    else 1
                )
                codes = _code_refs(code_root, stem, dataset, controlled_code_count)
                base = {
                    "conditioning_bundle": conditioning,
                    "official_checkpoint": official_checkpoint,
                    "official_dataset_encoder": official_encoder,
                }
                add(method, "controlled_single", dataset, seed, {**base, "code": codes[0]})
                if benchmark_tier == "final":
                    add(method, "controlled_validation_best_k", dataset, seed, {**base, "code_candidates": codes})
                    native_mode = f"{stem}_native_oracle"
                    native_method = f"weightclip_commonzoo_fullwindow_{native_mode}"
                    native_codes = _code_refs(
                        code_root, native_mode, dataset, DEFAULT_CONTRACT.evaluation.native_candidate_count
                    )
                    add(
                        native_method,
                        "native_test_top5_oracle",
                        dataset,
                        seed,
                        {**base, "code_candidates": native_codes},
                    )
    cells = canonical_grid_cells(
        {
            "method": arm["method"],
            "dataset": dataset,
            "evaluation_seed": seed,
            "protocol": arm["protocol"],
        }
        for arm in arms
        for dataset in arm["datasets"]
        for seed in arm["evaluation_seeds"]
    )
    if benchmark_tier == "final" and cells != canonical_final_grid_cells():
        raise ValueError("resolved final candidate arms disagree with the independently frozen canonical grid")
    grid_payload = {
        "schema_version": 1,
        "kind": (
            "frozen_expected_evaluation_grid"
            if benchmark_tier == "final"
            else "exploratory_single_seed0_expected_evaluation_grid"
        ),
        "benchmark_tier": benchmark_tier,
        "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
        "cells": cells,
    }
    grid_path = Path(config["expected_grid_output"]).resolve()
    write_json_immutable(grid_path, grid_payload)
    spec = {
        "schema_version": 1,
        "benchmark_tier": benchmark_tier,
        "scientific_status": (
            "final_paper_benchmark"
            if benchmark_tier == "final"
            else "exploratory_only_not_valid_for_final_unseal_or_reporting"
        ),
        "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
        "expected_grid": str(grid_path),
        "arms": arms,
    }
    _write_yaml_immutable(Path(config["candidate_spec_output"]).resolve(), spec)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build_spec(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    print(f"[candidate-spec:done] arms={len(result['arms'])} output={args.config}")


if __name__ == "__main__":
    main()
