from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker
from training.weightclip_benchmark.run_ae_v4_perirms_50k import (
    _build_worker_config as _build_base_worker_config,
)
from training.weightclip_benchmark.run_ae_v4_perirms_50k import (
    _load_spec as _load_base_spec,
)
from training.weightclip_benchmark.run_ae_v9_operator_set import _select_from_bank


SCHEMA = "weightclip_ae_v4_normalized_exact64_v1"
EXPECTED_TOTAL_PARAMETERS = 725_333_585
EXPECTED_TRAINABLE_PARAMETERS = 706_305_089


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original 725M V4 AE on exact64 with normalized weight input."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "conf/weightclip_benchmark/ae_v4_normalized_exact64_5000step.yaml"
        ),
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
        "weight_input_normalization_kind",
        "weight_input_scale_qmax",
        "weight_input_log2_scale_mean",
        "weight_input_log2_scale_std",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("V4 normalized exact64 config keys do not match the schema")
    if payload["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    frozen = {
        "operator_count": 64,
        "steps": 5000,
        "physical_batch": 6,
        "grad_accum_steps": 3,
        "eval_every_steps": 256,
        "eval_batch_size": 8,
        "seed": 42,
    }
    for key, value in frozen.items():
        if int(payload[key]) != value:
            raise ValueError(f"V4 normalized exact64 requires {key}={value}")
    if str(payload["layer_key"]) != "layer3.0.conv2.weight":
        raise ValueError("V4 normalized exact64 fixes layer3.0.conv2.weight")
    if str(payload["weight_input_normalization_kind"]) != "per_output_maxabs_q7":
        raise ValueError("the intervention must remain per_output_maxabs_q7")
    return payload


def _build_worker_config(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    base_spec = _load_base_spec(Path(str(spec["base_run_config"])).resolve())
    base_spec["output_root"] = str(Path(str(spec["output_root"])).resolve())
    base_spec["run_id"] = str(spec["run_id"])
    base_spec["gradient_noise_monitor"] = {"enabled": False}
    base_spec["comet_tags"] = [
        "v4",
        "exact64",
        "normalized-weight-input",
        "structural-only",
    ]
    cfg, output_root = _build_base_worker_config(base_spec)
    cfg["data"]["seed"] = int(spec["seed"])
    cfg["model"]["big_vae"].update(
        {
            "architecture_version": "latent_feedback_perirms_qknorm_v4",
            "weight_input_normalization_kind": str(
                spec["weight_input_normalization_kind"]
            ),
            "weight_input_scale_qmax": float(spec["weight_input_scale_qmax"]),
            "weight_input_log2_scale_mean": float(
                spec["weight_input_log2_scale_mean"]
            ),
            "weight_input_log2_scale_std": float(
                spec["weight_input_log2_scale_std"]
            ),
        }
    )
    train = cfg["train"]
    train.update(
        {
            "max_steps": int(spec["steps"]),
            "stop_after_step": int(spec["steps"]),
            "slice_batch_size": 6,
            "grad_accum_steps": 3,
            "checkpoint_every": int(spec["steps"]),
            "checkpoint_latest_copy": False,
            "log_every": int(spec["log_every"]),
            "compile": False,
            "tf32": False,
            "resume_checkpoint": "",
        }
    )
    train["resume_state"].update(
        {
            "enabled": False,
            "auto_resume": False,
            "explicit_checkpoint": "",
            "save_every": int(spec["steps"]),
        }
    )
    train["fixed_training_batch"]["enabled"] = False
    train["batch_source_mixing"].update({"enabled": False, "uniqueness": "none"})
    train["offline_batch_prefetch"].update({"queue_size": 2})
    selected, manifest = _select_from_bank(cfg, spec)
    base_cycle_matches = int(manifest["training_match_count"])
    requested_matches = int(spec["steps"])
    manifest["schema"] = "weightclip_ae_v4_normalized_exact64_selection_v1"
    manifest["schedule_scope"] = "base_cycle_repeated_to_training_horizon"
    manifest["base_cycle_training_round_count"] = int(
        manifest.pop("training_round_count")
    )
    manifest["base_cycle_training_match_count"] = base_cycle_matches
    manifest["training_match_count"] = requested_matches
    manifest["training_cycle_repetitions_full"] = requested_matches // base_cycle_matches
    manifest["training_cycle_tail_matches"] = requested_matches % base_cycle_matches
    manifest["total_training_tiles"] = requested_matches * int(
        manifest["tiles_per_update"]
    )
    train["operator_bank"].update(
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
                "selected_operators": [dict(row) for row in selected],
                "selection_sha256": manifest["selection_sha256"],
                "schedule_sha256": manifest["schedule_sha256"],
                "expected_total_parameters": EXPECTED_TOTAL_PARAMETERS,
                "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
                "expected_active_encoder_parameters": -1,
                "eval_every_steps": int(spec["eval_every_steps"]),
                "eval_batch_size": int(spec["eval_batch_size"]),
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
                "patch_tokenizer",
                "weight_input_scale_mlp",
                "patch_token_proj",
                "encoder_layers",
                "latent_to_weight_feedback",
                "decoder_layers",
                "output_head",
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
        raise AssertionError("V4 normalized exact64 must preserve the V4 structural objective")
    return cfg, output_root, manifest


def main() -> None:
    args = _parse_args()
    spec = _load_spec(args.config.resolve())
    cfg, output_root, manifest = _build_worker_config(spec)
    summary = {
        "schema": SCHEMA,
        "architecture": "latent_feedback_perirms_qknorm_v4",
        "intervention": "normalized encoder input only; decoder/loss target raw W",
        "normalization": {
            "kind": spec["weight_input_normalization_kind"],
            "qmax": spec["weight_input_scale_qmax"],
            "log2_scale_mean": spec["weight_input_log2_scale_mean"],
            "log2_scale_std": spec["weight_input_log2_scale_std"],
        },
        "device": cfg["train"]["device"],
        "dtype": cfg["train"].get("amp_dtype", "bf16"),
        "seed": cfg["data"]["seed"],
        "output_root": str(output_root),
        "steps": int(spec["steps"]),
        "physical_batch": int(spec["physical_batch"]),
        "grad_accum_steps": int(spec["grad_accum_steps"]),
        "parameter_contract": {
            "total": EXPECTED_TOTAL_PARAMETERS,
            "trainable": EXPECTED_TRAINABLE_PARAMETERS,
        },
        "selection": manifest,
    }
    print("[v4-normalized-exact64] stage=preflight", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print(
            "[v4-normalized-exact64] stage=dry-run-complete no_files_written=true",
            flush=True,
        )
        return
    output_root.mkdir(parents=True, exist_ok=False)
    Path(cfg["training_artifacts"]["run_root_dir"]).mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_run_config.json"
    selection_path = output_root / "operator_set_selection.json"
    resolved_path.write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(
        f"[v4-normalized-exact64] stage=train resolved={resolved_path} "
        f"selection={selection_path}",
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
    expected = checkpoint_root / f"step_{int(spec['steps']):07d}.pt"
    if not expected.is_file() or (checkpoint_root / "latest.pt").exists():
        raise RuntimeError("V4 normalized exact64 final-only checkpoint contract failed")
    print(f"[v4-normalized-exact64] stage=complete checkpoint={expected}", flush=True)


if __name__ == "__main__":
    main()
