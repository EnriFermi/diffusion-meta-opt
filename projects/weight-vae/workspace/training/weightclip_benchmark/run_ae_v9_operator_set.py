from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from big_vae.datasets.operator_bank import OperatorBankTrainingDataset
from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.operator_set_overfit import CanonicalOperatorSetTrainingDataset
from training.big_vae.worker import _run_worker
from training.weightclip_benchmark.run_ae_v4_perirms_50k import (
    _build_worker_config as _build_base_worker_config,
)
from training.weightclip_benchmark.run_ae_v4_perirms_50k import _load_spec as _load_base_spec


SCHEMA = "weightclip_ae_v9_exact64_nonlinear_v1"
V9A_SCHEMA = "weightclip_ae_v9a_exact64_sqrt_depth_v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fresh exact64 V9 nonlinear-content diagnostic (1984 updates)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v9_exact64_nonlinear_1984step.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "base_run_config",
        "output_root",
        "run_id",
        "seed",
        "layer_key",
        "operator_count",
        "steps",
        "physical_batch",
        "grad_accum_steps",
        "eval_every_steps",
        "eval_batch_size",
        "log_every",
        "comet_enabled",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(
            f"V9 exact64 config keys mismatch: expected={sorted(expected)} "
            f"actual={sorted(payload) if isinstance(payload, dict) else type(payload)}"
        )
    if payload["schema"] not in {SCHEMA, V9A_SCHEMA}:
        raise ValueError(f"V9 exact64 schema must be one of {sorted((SCHEMA, V9A_SCHEMA))!r}")
    frozen = {
        "operator_count": 64,
        "steps": 1984,
        "physical_batch": 6,
        "grad_accum_steps": 3,
        "eval_every_steps": 256,
        "eval_batch_size": 8,
        "seed": 42,
    }
    for key, value in frozen.items():
        if int(payload[key]) != value:
            raise ValueError(f"V9 exact64 requires {key}={value}")
    if str(payload["layer_key"]) != "layer3.0.conv2.weight":
        raise ValueError("V9 exact64 fixes layer3.0.conv2.weight (1152x128, nine tiles)")
    return payload


def _hash_order(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).hexdigest()


def _balanced_operator_selection(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in candidates:
        dataset = str(row["dataset"])
        lineage = str(row["lineage_id"])
        epoch = int(row["checkpoint_index_zero_based"])
        if epoch not in {43, 44}:
            continue
        if epoch in by_dataset[dataset][lineage]:
            raise ValueError(f"duplicate dataset/lineage/epoch candidate: {dataset}/{lineage}/{epoch}")
        by_dataset[dataset][lineage][epoch] = row
    if len(by_dataset) != 10:
        raise ValueError(f"V9 exact64 requires exactly ten datasets, got {sorted(by_dataset)}")
    datasets = sorted(by_dataset, key=lambda name: _hash_order(seed, "dataset", name))
    extra_datasets = set(datasets[:4])
    selected: list[dict[str, Any]] = []
    epoch_counts: Counter[int] = Counter()
    for dataset in datasets:
        quota = 7 if dataset in extra_datasets else 6
        complete_lineages = [
            lineage for lineage, epochs in by_dataset[dataset].items() if set(epochs) == {43, 44}
        ]
        complete_lineages.sort(key=lambda lineage: _hash_order(seed, dataset, lineage))
        if len(complete_lineages) < quota:
            raise ValueError(f"dataset {dataset!r} lacks {quota} complete unique lineages")
        for lineage in complete_lineages[:quota]:
            epoch = 43 if len(selected) % 2 == 0 else 44
            row = by_dataset[dataset][lineage][epoch]
            selected.append(
                {
                    "checkpoint_sha256": str(row["checkpoint_sha256"]),
                    "layer_key": str(row["layer_key"]),
                    "dataset": dataset,
                    "lineage_id": lineage,
                    "checkpoint_index_zero_based": epoch,
                    "matrix_shape": list(row["matrix_shape"]),
                    "tile_count": int(row["tile_count"]),
                }
            )
            epoch_counts[epoch] += 1
    if len(selected) != 64 or len(
        {(row["dataset"], row["lineage_id"]) for row in selected}
    ) != 64:
        raise RuntimeError("V9 exact64 selection failed count/unique-lineage contract")
    if epoch_counts != Counter({43: 32, 44: 32}):
        raise RuntimeError(f"V9 exact64 epoch balance mismatch: {dict(epoch_counts)}")
    dataset_counts = Counter(str(row["dataset"]) for row in selected)
    if sorted(dataset_counts.values()) != [6] * 6 + [7] * 4:
        raise RuntimeError(f"V9 exact64 dataset quotas mismatch: {dict(dataset_counts)}")
    return selected


def _select_from_bank(cfg: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    operator = cfg["train"]["operator_bank"]
    source = OperatorBankTrainingDataset(
        operator["pair_manifest"],
        seed=int(spec["seed"]),
        repeat=True,
        permutation_views=False,
        canonical_probability=1.0,
        hot_shards=1,
        expected_pair_manifest_sha256=str(operator["pair_manifest_sha256"]),
        rank=0,
        world_size=1,
    )
    layer_key = str(spec["layer_key"])
    candidates: list[dict[str, Any]] = []
    for key, locations in source.operator_groups.items():
        if key[1] != layer_key:
            continue
        metadata = locations[0].metadata
        operator_meta = metadata["operator"]
        candidates.append(
            {
                "checkpoint_sha256": key[0],
                "layer_key": key[1],
                "dataset": str(metadata["dataset"]),
                "lineage_id": str(metadata["lineage_id"]),
                "checkpoint_index_zero_based": int(metadata["checkpoint_index_zero_based"]),
                "matrix_shape": list(operator_meta["matrix_shape"]),
                "tile_count": len(locations),
            }
        )
    selected = _balanced_operator_selection(candidates, seed=int(spec["seed"]))
    if any(row["matrix_shape"] != [1152, 128] or row["tile_count"] != 9 for row in selected):
        raise RuntimeError("V9 exact64 selection must contain complete 1152x128 nine-tile matrices")
    keys = [(row["checkpoint_sha256"], row["layer_key"]) for row in selected]
    rounds = CanonicalOperatorSetTrainingDataset._round_robin_rounds(
        keys,
        seed=int(spec["seed"]),
    )
    selection_sha256 = CanonicalOperatorSetTrainingDataset._selection_sha256(keys)
    schedule_sha256 = CanonicalOperatorSetTrainingDataset._schedule_sha256(rounds)
    manifest = {
        "schema": "weightclip_ae_v9_exact64_selection_v1",
        "operators": selected,
        "operator_count": 64,
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in selected).items())),
        "checkpoint_index_counts": dict(
            sorted(Counter(row["checkpoint_index_zero_based"] for row in selected).items())
        ),
        "unique_dataset_lineage_count": len(
            {(row["dataset"], row["lineage_id"]) for row in selected}
        ),
        "selection_sha256": selection_sha256,
        "schedule_sha256": schedule_sha256,
        "training_round_count": 62,
        "training_match_count": 1984,
        "heldout_derangement_round_index": 62,
        "heldout_derangement_pairs": rounds[62],
        "tiles_per_operator": 9,
        "tiles_per_update": 18,
        "total_training_tiles": 1984 * 18,
    }
    return selected, manifest


def _build_worker_config(spec: Mapping[str, Any]) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    base_spec = _load_base_spec(Path(str(spec["base_run_config"])).expanduser().resolve())
    base_spec["output_root"] = str(Path(str(spec["output_root"])).expanduser().resolve())
    base_spec["run_id"] = str(spec["run_id"])
    base_spec["gradient_noise_monitor"] = {"enabled": False}
    base_spec["comet_tags"] = ["v9", "exact64", "nonlinear-content", "structural-only"]
    cfg, output_root = _build_base_worker_config(base_spec)
    cfg["data"]["seed"] = int(spec["seed"])
    big = cfg["model"]["big_vae"]
    is_v9a = str(spec["schema"]) == V9A_SCHEMA
    big.update(
        {
            "architecture_version": (
                "carrier_mean_content_posfilm_v9a"
                if is_v9a
                else "hybrid_nonlinear_content_posfilm_v9"
            ),
            "normalize_latent_slots_before_mu": False,
            "v9_refinement_blocks": 40,
            "v9_router_width": 384,
            "v9_router_ffn_width": 1536,
            "v9_score_cap": 0.1,
            "v9_residual_scale": 1.0 / (80.0**0.5),
            "v9_anchor_sigma_keys": 0.6,
            "v9_bypass_refinement": False,
            "v9a_carrier_mix": 0.1,
            "v9a_protected_anchor_floor": 0.25,
            "v9a_aggregation": "sqrt_depth_sum",
        }
    )
    train = cfg["train"]
    train.update(
        {
            "max_steps": 1984,
            "stop_after_step": 1984,
            "slice_batch_size": 6,
            "grad_accum_steps": 3,
            "checkpoint_every": 1984,
            "checkpoint_latest_copy": False,
            "log_every": int(spec["log_every"]),
            "compile": False,
            "resume_checkpoint": "",
        }
    )
    train["resume_state"].update(
        {"enabled": False, "auto_resume": False, "explicit_checkpoint": "", "save_every": 1984}
    )
    train["fixed_training_batch"]["enabled"] = False
    train["batch_source_mixing"].update({"enabled": False, "uniqueness": "none"})
    train["offline_batch_prefetch"].update({"queue_size": 2})
    selected, manifest = _select_from_bank(cfg, spec)
    operator = train["operator_bank"]
    operator.update(
        {
            "approved_weightclip_contract": False,
            "permutation_views": False,
            "canonical_probability": 1.0,
            "loader_workers": 0,
            "loader_batch_size": 1,
            "loader_persistent_workers": False,
            "two_operator_overfit": {"enabled": False},
            "operator_set_overfit": {
                "enabled": True,
                "selected_operators": [
                    dict(row)
                    for row in selected
                ],
                "selection_sha256": manifest["selection_sha256"],
                "schedule_sha256": manifest["schedule_sha256"],
                "expected_total_parameters": 718_513_281,
                "expected_trainable_parameters": 239_113_361,
                "expected_active_encoder_parameters": 68_803_968,
                "eval_every_steps": 256,
                "eval_batch_size": 8,
                "metrics_path": str(output_root / "operator_set_metrics.jsonl"),
            },
        }
    )
    train["telemetry"]["comet"]["enabled"] = bool(spec["comet_enabled"])
    train["telemetry"]["wandb"]["enabled"] = False
    train["telemetry"]["grad_layer_monitor"].update(
        {
            "enabled": True,
            "every_steps": 256,
            "weights_only": True,
            "save_csv": True,
            "reset_csv_on_start": True,
            "save_plot": False,
            "plot_every_steps": 0,
            "save_heatmap": False,
            "include_prefixes": [
                "distribution_encoder",
                "clean_content_readout_v8",
                "hybrid_content_readout_v9",
                "mandatory_latent_bridge",
                "direction_head",
            ],
        }
    )
    structural = train["struct_loss"]
    if (
        float(train["behavioral_coef"]) != 0.0
        or float(train["structural_coef"]) != 1.0
        or float(structural["lambda_dir"]) != 1.0
        or float(structural["lambda_scale"]) != 0.1
        or float(structural["lambda_rec"]) != 0.0
        or float(structural["lambda_rel"]) != 0.0
    ):
        raise AssertionError("V9 exact64 must preserve the frozen structural-only objective")
    return cfg, output_root, manifest


def main() -> None:
    args = _parse_args()
    spec = _load_spec(args.config.expanduser().resolve())
    cfg, output_root, manifest = _build_worker_config(spec)
    summary = {
        "schema": spec["schema"],
        "architecture": cfg["model"]["big_vae"]["architecture_version"],
        "device": cfg["train"]["device"],
        "dtype": cfg["train"].get("amp_dtype", "bf16"),
        "seed": cfg["data"]["seed"],
        "output_root": str(output_root),
        "steps": 1984,
        "physical_batch": 6,
        "grad_accum_steps": 3,
        "checkpoint_policy": "one final step_0001984.pt; no latest; no resume-state",
        "parameter_contract": {
            "total": 718_513_281,
            "trainable": 239_113_361,
            "active_encoder": 68_803_968,
            "v9_refiner": 63_175_680,
            "per_v9_block": 1_579_392,
        },
        "selection": manifest,
    }
    print("[v9-exact64] stage=preflight", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[v9-exact64] stage=dry-run-complete no_files_written=true", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=False)
    Path(cfg["training_artifacts"]["run_root_dir"]).mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_run_config.json"
    selection_path = output_root / "operator_set_selection.json"
    resolved_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(
        f"[v9-exact64] stage=train resolved={resolved_path} selection={selection_path}",
        flush=True,
    )
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=cfg,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )
    checkpoint_root = Path(cfg["train"]["checkpoint_dir"]) / "stage_1"
    expected = checkpoint_root / "step_0001984.pt"
    if not expected.is_file() or (checkpoint_root / "latest.pt").exists():
        raise RuntimeError("V9 exact64 final-only checkpoint contract failed")
    print(f"[v9-exact64] stage=complete checkpoint={expected}", flush=True)


if __name__ == "__main__":
    main()
