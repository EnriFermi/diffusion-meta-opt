#!/usr/bin/env python3
"""Seal exact ours/WeightCLIP z_task optimization parity before any fits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef
from big_vae.weightclip_benchmark.manifests import write_json_immutable
from big_vae.weightclip_benchmark.task_latent_fit import (
    TaskLatentFitConfig,
    assert_task_fit_protocol_parity,
    task_fit_protocol,
)
from big_vae.weightclip_benchmark.weightclip_full import WeightCLIPTaskFitConfig


def seal_parity(ours_path: Path, weightclip_path: Path, output: Path) -> dict[str, object]:
    ours_raw = yaml.safe_load(ours_path.read_text(encoding="utf-8"))
    weightclip_raw = yaml.safe_load(weightclip_path.read_text(encoding="utf-8"))
    ours_fit = dict(ours_raw["fit"])
    weightclip_fit = dict(weightclip_raw["fit"])
    ours_fit["task_batch_size"] = int(ours_raw["task_batch_size"])
    weightclip_fit["task_batch_size"] = int(weightclip_raw["task_batch_size"])
    ours = TaskLatentFitConfig(**ours_fit)
    weightclip = WeightCLIPTaskFitConfig(**weightclip_fit)
    assert_task_fit_protocol_parity(ours, weightclip)
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "task_fit_protocol_parity_seal",
        "ours_config": asdict(ArtifactRef.create(ours_path)),
        "weightclip_config": asdict(ArtifactRef.create(weightclip_path)),
        "matched_protocol": task_fit_protocol(ours),
    }
    write_json_immutable(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-config", type=Path, required=True)
    parser.add_argument("--weightclip-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal_parity(args.ours_config.resolve(), args.weightclip_config.resolve(), args.output.resolve())
    print(f"[task-fit-parity:done] output={args.output.resolve()}")


if __name__ == "__main__":
    main()
