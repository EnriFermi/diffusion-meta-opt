from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
import torch

from big_vae.datasets.operator_bank import OperatorBundleRequest
from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker
from training.big_vae.two_operator_overfit import TwoOperatorTrainingDataset
from training.weightclip_benchmark.run_ae_v4_perirms_50k import (
    _build_worker_config as _build_v5_worker_config,
)
from training.weightclip_benchmark.run_ae_v4_perirms_50k import _load_spec as _load_v5_spec


_SCHEMA_TO_CONTRACT = {
    "weightclip_ae_two_full_operator_overfit_v1": {
        "architecture": "latent_mandatory_bridge_prenorm_v5",
        "steps": 5_000,
        "bounded_horizon": False,
    },
    "weightclip_ae_v6_two_full_operator_overfit_v1": {
        "architecture": "latent_mandatory_bridge_posfilm_v6",
        "steps": 2_000,
        "bounded_horizon": True,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded V5 overfit smoke on every tile of exactly two full operators."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v5_two_operator_overfit_5000step.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved contract only.")
    return parser.parse_args()


def _load_overfit_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") not in _SCHEMA_TO_CONTRACT:
        raise ValueError(
            "two-operator smoke config has an unsupported schema: "
            f"{payload.get('schema') if isinstance(payload, dict) else type(payload)!r}"
        )
    contract = _SCHEMA_TO_CONTRACT[str(payload["schema"])]
    if int(payload.get("steps", -1)) != int(contract["steps"]):
        raise ValueError(
            f"{payload['schema']} is hard-capped at exactly {int(contract['steps'])} steps"
        )
    allowed = {
        "schema",
        "base_v5_run_config",
        "output_root",
        "run_id",
        "steps",
        "checkpoint_every",
        "log_every",
        "comet_enabled",
        "eval_every_steps",
        "free_weight_control_steps",
        "free_weight_control_lr",
        "identity_diagnostics",
        "early_success",
        "selected_operators",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown two-operator smoke config keys: {unknown}")
    required = allowed - {
        "comet_enabled",
        "free_weight_control_steps",
        "free_weight_control_lr",
        "identity_diagnostics",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"missing two-operator smoke config keys: {missing}")
    early = payload.get("early_success")
    expected_early = {
        "enabled",
        "min_step",
        "consecutive_evals",
        "struct_dir_max",
        "struct_scale_max",
        "nrmse_max",
    }
    if str(payload["schema"]) == "weightclip_ae_v6_two_full_operator_overfit_v1":
        expected_early.update({"matched_mean_dir_max", "swap_dir_delta_min"})
    else:
        expected_early.add("swap_total_delta_min")
    if not isinstance(early, dict) or set(early) != expected_early:
        raise ValueError(
            "early_success must contain exactly "
            f"{sorted(expected_early)}, got {sorted(early) if isinstance(early, dict) else type(early)}"
        )
    if int(payload["eval_every_steps"]) < 1 or int(payload["checkpoint_every"]) < 1:
        raise ValueError("evaluation and checkpoint intervals must be positive")
    selected = payload.get("selected_operators")
    if not isinstance(selected, list) or len(selected) != 2:
        raise ValueError("selected_operators must contain exactly two full-operator identities")
    identities: list[tuple[str, str]] = []
    for row in selected:
        if not isinstance(row, Mapping):
            raise TypeError("selected_operators entries must be mappings")
        identity = (str(row.get("checkpoint_sha256", "")), str(row.get("layer_key", "")))
        if len(identity[0]) != 64 or any(char not in "0123456789abcdef" for char in identity[0]):
            raise ValueError(f"invalid lowercase checkpoint SHA256: {identity[0]!r}")
        if not identity[1]:
            raise ValueError("selected operator layer_key must be non-empty")
        identities.append(identity)
    if len(set(identities)) != 2:
        raise ValueError("the two selected full operators must have distinct identities")
    return payload


def _operator_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    operator = cfg["train"]["operator_bank"]
    selected = operator["two_operator_overfit"]["selected_operators"]
    source = TwoOperatorTrainingDataset(
        operator["pair_manifest"],
        selected_operators=selected,
        seed=int(cfg["data"]["seed"]),
        hot_shards=int(operator["hot_shards"]),
        expected_pair_manifest_sha256=str(operator["pair_manifest_sha256"]),
    )
    rows: list[dict[str, Any]] = []
    matrix_hashes: list[str] = []
    matrices: list[torch.Tensor] = []
    for key in source._keys:
        locations = source.operator_groups[key]
        first = locations[0].metadata
        matrix_rows, matrix_cols = map(int, first["operator"]["matrix_shape"])
        expected_starts = {
            (row, col)
            for row in range(0, matrix_rows, 128)
            for col in range(0, matrix_cols, 128)
        }
        actual_starts = {
            (int(location.metadata["tile"]["row_start"]), int(location.metadata["tile"]["col_start"]))
            for location in locations
        }
        if actual_starts != expected_starts or len(actual_starts) != len(locations):
            raise RuntimeError(f"selected operator {key} does not have exact complete tile coverage")
        bundle = source._materialize_bundle(OperatorBundleRequest(cycle=0, key=key), None)
        reconstructed = torch.zeros_like(bundle.matrix)
        for local_index in range(bundle.tile_count):
            sample = source._sample_from_bundle(bundle, local_index)
            tile_meta = sample.meta
            row_start = int(tile_meta["tile_row_start"])
            col_start = int(tile_meta["tile_col_start"])
            valid_rows = int(sample.meta["d_in_mask"].sum().item())
            valid_cols = int(sample.meta["d_out_mask"].sum().item())
            reconstructed[row_start : row_start + valid_rows, col_start : col_start + valid_cols] = sample.weight[
                :valid_rows, :valid_cols
            ]
        if not torch.equal(reconstructed, bundle.matrix):
            raise RuntimeError(f"tile stitch is not bitwise equal to selected full operator {key}")
        matrix_sha256 = hashlib.sha256(bundle.matrix.contiguous().numpy().tobytes()).hexdigest()
        matrix_hashes.append(matrix_sha256)
        matrices.append(bundle.matrix)
        rows.append(
            {
                "checkpoint_sha256": key[0],
                "layer_key": key[1],
                "dataset": str(first["dataset"]),
                "lineage_id": str(first["lineage_id"]),
                "checkpoint_index_zero_based": int(first["checkpoint_index_zero_based"]),
                "matrix_shape": [matrix_rows, matrix_cols],
                "tile_count": len(locations),
                "canonical_only": True,
                "complete_full_matrix_coverage": True,
                "tile_stitch_bitwise_equal": True,
                "matrix_float32_sha256": matrix_sha256,
            }
        )
    cycle, views = source._cycle_plan(0)
    next_cycle, next_views = source._cycle_plan(1)
    if cycle != next_cycle or views != next_views:
        raise RuntimeError("two-operator data cycle is not fixed across repetitions")
    if len(set(matrix_hashes)) != 2:
        raise RuntimeError("the two selected operator matrices must have distinct float32 contents")
    matrix_cosine = float(
        torch.nn.functional.cosine_similarity(matrices[0].flatten(), matrices[1].flatten(), dim=0).item()
    )
    return {
        "schema": "two_full_operator_fixed_cycle_manifest_v1",
        "pair_manifest": str(source.pair_manifest_path),
        "pair_manifest_sha256": str(operator["pair_manifest_sha256"]),
        "operators": rows,
        "operator_count": len(rows),
        "tiles_per_cycle": len(cycle),
        "cycle_unique_tile_count": len(set(cycle)),
        "cycle_repeats_bitwise_same_logical_order": True,
        "gauge": "canonical",
        "cross_operator_weight_cosine": matrix_cosine,
    }


def _build_worker_config(spec: dict[str, Any]) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    contract = _SCHEMA_TO_CONTRACT[str(spec["schema"])]
    base_spec_path = Path(str(spec["base_v5_run_config"])).expanduser().resolve()
    base_spec = _load_v5_spec(base_spec_path)
    base_spec["output_root"] = str(Path(str(spec["output_root"])).expanduser().resolve())
    base_spec["run_id"] = str(spec["run_id"])
    base_spec["gradient_noise_monitor"] = {"enabled": False}
    base_spec["comet_tags"] = [
        "bounded-smoke",
        "two-full-operators",
        str(contract["architecture"]),
        "overfit",
    ]
    cfg, output_root = _build_v5_worker_config(base_spec)
    actual_architecture = str(cfg["model"]["big_vae"]["architecture_version"])
    if actual_architecture != str(contract["architecture"]):
        raise ValueError(
            "two-operator base architecture mismatch: "
            f"expected={contract['architecture']!r} actual={actual_architecture!r}"
        )

    train = cfg["train"]
    steps = int(contract["steps"])
    train.update(
        {
            "stop_after_step": steps,
            "slice_batch_size": 18,
            "grad_accum_steps": 1,
            "checkpoint_every": int(spec["checkpoint_every"]),
            "log_every": int(spec["log_every"]),
        }
    )
    if bool(contract["bounded_horizon"]):
        train["max_steps"] = steps
    train["resume_state"]["save_every"] = int(spec["checkpoint_every"])
    train["fixed_training_batch"]["enabled"] = False
    train["batch_source_mixing"].update({"enabled": False, "max_source_samples": 1, "uniqueness": "none"})
    train["offline_batch_prefetch"].update({"queue_size": 2})
    operator = train["operator_bank"]
    operator.update(
        {
            "approved_weightclip_contract": False,
            "permutation_views": False,
            "canonical_probability": 1.0,
            "loader_workers": 0,
            "loader_batch_size": 1,
            "loader_persistent_workers": False,
            "two_operator_overfit": {
                "enabled": True,
                "selected_operators": list(spec["selected_operators"]),
                "early_success": dict(spec["early_success"]),
                "eval_every_steps": int(spec["eval_every_steps"]),
                "metrics_path": str(
                    Path(cfg["train"]["checkpoint_dir"]) / "stage_1" / "two_operator_metrics.jsonl"
                ),
                "success_path": str(
                    Path(cfg["train"]["checkpoint_dir"]) / "stage_1" / "two_operator_success.json"
                ),
                "free_weight_control_steps": int(spec.get("free_weight_control_steps", 50)),
                "free_weight_control_lr": float(spec.get("free_weight_control_lr", 1.0e-2)),
                "include_identity_diagnostics": bool(spec.get("identity_diagnostics", False)),
            },
        }
    )
    comet_enabled = bool(spec.get("comet_enabled", False))
    train["telemetry"]["comet"]["enabled"] = comet_enabled
    train["telemetry"]["wandb"]["enabled"] = False
    manifest = _operator_manifest(cfg)
    if manifest["operator_count"] != 2 or manifest["cycle_unique_tile_count"] != manifest["tiles_per_cycle"]:
        raise AssertionError("two-operator smoke must use every tile of exactly two full matrices once per cycle")
    if int(train["slice_batch_size"]) * int(train["grad_accum_steps"]) != manifest["tiles_per_cycle"]:
        raise AssertionError("each optimizer step must consume one exact complete two-operator tile cycle")
    return cfg, output_root, manifest


def main() -> None:
    args = _parse_args()
    spec = _load_overfit_spec(args.config.expanduser().resolve())
    cfg, output_root, operator_manifest = _build_worker_config(spec)
    summary = {
        "architecture": cfg["model"]["big_vae"]["architecture_version"],
        "device": cfg["train"]["device"],
        "dtype": cfg["train"].get("amp_dtype", "bf16"),
        "seed": cfg["data"]["seed"],
        "steps": cfg["train"]["stop_after_step"],
        "scheduler_horizon": cfg["train"]["max_steps"],
        "physical_batch": cfg["train"]["slice_batch_size"],
        "grad_accum_steps": cfg["train"]["grad_accum_steps"],
        "learning_rate": cfg["train"]["lr"],
        "loss": {
            "behavioral_coef": cfg["train"]["behavioral_coef"],
            "structural_coef": cfg["train"]["structural_coef"],
            "structural": dict(cfg["train"]["struct_loss"]),
        },
        "output_root": str(output_root),
        "operator_manifest": operator_manifest,
    }
    print("[two-operator-overfit] stage=preflight", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[two-operator-overfit] stage=dry-run-complete no_files_written=true", flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=False)
    run_root = Path(cfg["training_artifacts"]["run_root_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_run_config.json"
    manifest_path = output_root / "two_full_operator_manifest.json"
    resolved_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(operator_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(f"[two-operator-overfit] stage=train resolved_config={resolved_path} manifest={manifest_path}", flush=True)
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=cfg,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )
    checkpoint_root = Path(cfg["train"]["checkpoint_dir"]) / "stage_1"
    checkpoints = sorted(checkpoint_root.glob("step_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"two-operator smoke finished without a checkpoint in {checkpoint_root}")
    latest = checkpoints[-1]
    success_path = Path(cfg["train"]["operator_bank"]["two_operator_overfit"]["success_path"])
    print(
        f"[two-operator-overfit] stage=complete checkpoint={latest} manifest={manifest_path} "
        f"early_success={success_path.is_file()} success_artifact={success_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
