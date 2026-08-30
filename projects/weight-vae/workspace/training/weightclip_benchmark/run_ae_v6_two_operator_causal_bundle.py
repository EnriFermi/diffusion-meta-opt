from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.v6_causal_bundle import ARMS, SCHEMA
from training.big_vae.worker import _run_worker
from training.weightclip_benchmark.run_ae_two_operator_overfit import (
    _build_worker_config as _build_two_operator_config,
    _load_overfit_spec,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded fresh V6 two-operator causal arm "
            "(never resumes; 500 steps, or exact 504-step singleton cycle)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v6_two_operator_causal_bundle_500step.yaml"),
    )
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("schema", "")) != SCHEMA:
        raise ValueError(f"causal-bundle config schema must be {SCHEMA!r}")
    allowed = {
        "schema",
        "base_two_operator_config",
        "steps",
        "eval_every_steps",
        "checkpoint_every",
        "log_every",
        "seed",
        "existing_v6_baseline_step500",
        "arms",
    }
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"causal-bundle config keys mismatch: missing={missing} unknown={unknown}")
    if int(payload["steps"]) != 500:
        raise ValueError("causal-bundle default steps must equal 500")
    if int(payload["eval_every_steps"]) != 100:
        raise ValueError("causal-bundle eval_every_steps must equal 100")
    if int(payload["checkpoint_every"]) != 500:
        raise ValueError("causal-bundle checkpoint_every must equal 500")
    arms = payload["arms"]
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError(f"causal-bundle arms must be exactly {ARMS}")
    allowed_arm = {
        "output_root",
        "run_id",
        "steps",
        "zero_all_return_biases",
        "freeze_zeroed_biases",
        "homogeneous_first_operator",
        "cross_refresh_rms",
        "fixed_x_w_swap_dir_delta_min",
    }
    for arm_name, arm in arms.items():
        if not isinstance(arm, Mapping):
            raise TypeError(f"causal-bundle arm {arm_name!r} must be a mapping")
        arm_unknown = sorted(set(arm) - allowed_arm)
        arm_missing = sorted({"output_root", "run_id"} - set(arm))
        if arm_unknown or arm_missing:
            raise ValueError(
                f"causal-bundle arm {arm_name!r} keys mismatch: "
                f"missing={arm_missing} unknown={arm_unknown}"
            )
        if bool(arm.get("zero_all_return_biases", False)) and arm_name != "zero_common_bias":
            raise ValueError("zero_all_return_biases is valid only for zero_common_bias")
        if bool(arm.get("freeze_zeroed_biases", False)) and arm_name != "zero_common_bias":
            raise ValueError("freeze_zeroed_biases is valid only for zero_common_bias")
        if arm_name == "mandatory_cross_refresh" and float(arm.get("cross_refresh_rms", -1.0)) != 1.0:
            raise ValueError("mandatory_cross_refresh must declare cross_refresh_rms: 1.0")
        if "fixed_x_w_swap_dir_delta_min" in arm and arm_name != "clean_content_readout":
            raise ValueError("fixed_x_w_swap_dir_delta_min is valid only for clean_content_readout")
        if arm_name == "clean_content_readout" and float(
            arm.get("fixed_x_w_swap_dir_delta_min", -1.0)
        ) != 0.03:
            raise ValueError("clean_content_readout fixes fixed-X W-swap direction delta at 0.03")
        expected_steps = 504 if arm_name == "cyclic_singleton" else 500
        if int(arm.get("steps", expected_steps)) != expected_steps:
            raise ValueError(f"causal-bundle arm {arm_name!r} must use exactly {expected_steps} steps")
    return payload


def _build_worker_config(
    bundle_spec: Mapping[str, Any],
    arm_name: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if arm_name not in ARMS:
        raise ValueError(f"unsupported causal arm {arm_name!r}")
    base_path = Path(str(bundle_spec["base_two_operator_config"])).expanduser().resolve()
    overfit_spec = copy.deepcopy(_load_overfit_spec(base_path))
    arm = dict(bundle_spec["arms"][arm_name])
    steps = 504 if arm_name == "cyclic_singleton" else 500
    overfit_spec["output_root"] = str(Path(str(arm["output_root"])).expanduser().resolve())
    overfit_spec["run_id"] = str(arm["run_id"])
    overfit_spec["checkpoint_every"] = steps
    overfit_spec["log_every"] = int(bundle_spec["log_every"])
    overfit_spec["eval_every_steps"] = int(bundle_spec["eval_every_steps"])
    overfit_spec["comet_enabled"] = False
    overfit_spec["early_success"]["enabled"] = False
    cfg, output_root, manifest = _build_two_operator_config(overfit_spec)

    if arm_name == "mandatory_cross_refresh":
        cfg["model"]["big_vae"]["architecture_version"] = "latent_mandatory_cross_refresh_posfilm_v7"
    elif arm_name == "clean_content_readout":
        cfg["model"]["big_vae"]["architecture_version"] = "clean_content_readout_posfilm_v8"
        cfg["model"]["big_vae"]["normalize_latent_slots_before_mu"] = False

    train = cfg["train"]
    train.update(
        {
            "max_steps": steps,
            "stop_after_step": steps,
            "checkpoint_every": steps,
            "log_every": int(bundle_spec["log_every"]),
            "compile": False,
        }
    )
    train["resume_checkpoint"] = ""
    train["resume_state"].update(
        {
            "enabled": False,
            "auto_resume": False,
            "explicit_checkpoint": "",
            "save_every": steps,
        }
    )
    two_operator = train["operator_bank"]["two_operator_overfit"]
    two_operator["eval_every_steps"] = 100
    two_operator["early_success"]["enabled"] = False
    if arm_name == "clean_content_readout":
        two_operator["early_success"]["fixed_x_w_swap_dir_delta_min"] = float(
            arm["fixed_x_w_swap_dir_delta_min"]
        )
    train["v6_causal_bundle"] = {
        "enabled": True,
        "schema": SCHEMA,
        "arm": arm_name,
        "ledger_path": str(output_root / "v6_causal_intervention_ledger.json"),
        "zero_all_return_biases": bool(arm.get("zero_all_return_biases", False)),
        "freeze_zeroed_biases": bool(arm.get("freeze_zeroed_biases", False)),
        "homogeneous_first_operator": int(arm.get("homogeneous_first_operator", 0)),
        "centered_value_gain_max": 1.0e4,
        "cross_refresh_rms": float(arm.get("cross_refresh_rms", 1.0)),
    }
    if arm_name in {"mandatory_cross_refresh", "clean_content_readout"}:
        grad_monitor = train["telemetry"]["grad_layer_monitor"]
        include_prefixes = (
            [
                "distribution_encoder",
                "clean_content_readout_v8",
                "mandatory_latent_bridge",
                "direction_head",
            ]
            if arm_name == "clean_content_readout"
            else [
                "patch_tokenizer",
                "patch_token_proj",
                "encoder_layers.0.perceiver_block",
                "encoder_layers.9.perceiver_block",
                "mandatory_latent_bridge",
                "direction_head",
            ]
        )
        grad_monitor.update(
            {
                "enabled": True,
                "every_steps": 100,
                "weights_only": True,
                "save_csv": True,
                "reset_csv_on_start": True,
                "save_plot": False,
                "plot_every_steps": 0,
                "save_heatmap": False,
                "include_prefixes": include_prefixes,
            }
        )
    cfg["data"]["seed"] = int(bundle_spec["seed"])
    if int(train["slice_batch_size"]) != 18 or int(train["grad_accum_steps"]) != 1:
        raise AssertionError("causal bundle must preserve one exact mixed B18 cycle per optimizer step")
    if bool(train["resume_state"]["enabled"]) or bool(train["resume_state"]["auto_resume"]):
        raise AssertionError("causal bundle must be fresh and resume-free")
    structural = train["struct_loss"]
    if (
        float(train["behavioral_coef"]) != 0.0
        or float(train["structural_coef"]) != 1.0
        or float(structural["lambda_dir"]) != 1.0
        or float(structural["lambda_scale"]) != 0.1
        or float(structural["lambda_rec"]) != 0.0
        or float(structural["lambda_rel"]) != 0.0
        or float(structural["gamma"]) != 0.5
        or float(structural["huber_delta"]) != 0.1
    ):
        raise AssertionError("causal bundle must preserve the exact V6 two-operator structural objective")
    if arm_name in {"mandatory_cross_refresh", "clean_content_readout"}:
        expected_grad_prefixes = (
            [
                "distribution_encoder",
                "clean_content_readout_v8",
                "mandatory_latent_bridge",
                "direction_head",
            ]
            if arm_name == "clean_content_readout"
            else [
                "patch_tokenizer",
                "patch_token_proj",
                "encoder_layers.0.perceiver_block",
                "encoder_layers.9.perceiver_block",
                "mandatory_latent_bridge",
                "direction_head",
            ]
        )
        grad_monitor = train["telemetry"]["grad_layer_monitor"]
        if (
            not bool(grad_monitor["enabled"])
            or int(grad_monitor["every_steps"]) != 100
            or not bool(grad_monitor["save_csv"])
            or bool(grad_monitor["save_plot"])
            or bool(grad_monitor["save_heatmap"])
            or list(grad_monitor["include_prefixes"]) != expected_grad_prefixes
        ):
            raise AssertionError(f"{arm_name} requires the exact local CSV gradient monitor contract")
    if arm_name == "clean_content_readout" and bool(
        cfg["model"]["big_vae"]["normalize_latent_slots_before_mu"]
    ):
        raise AssertionError("clean_content_readout requires raw, unnormalized latent slots")
    return cfg, output_root, manifest


def main() -> None:
    args = _parse_args()
    bundle_spec = _load_spec(args.config.expanduser().resolve())
    cfg, output_root, manifest = _build_worker_config(bundle_spec, args.arm)
    summary = {
        "schema": SCHEMA,
        "arm": args.arm,
        "architecture": cfg["model"]["big_vae"]["architecture_version"],
        "seed": cfg["data"]["seed"],
        "steps": cfg["train"]["max_steps"],
        "eval_every_steps": cfg["train"]["operator_bank"]["two_operator_overfit"]["eval_every_steps"],
        "physical_batch": cfg["train"]["slice_batch_size"],
        "grad_accum_steps": cfg["train"]["grad_accum_steps"],
        "compile": cfg["train"]["compile"],
        "resume_enabled": cfg["train"]["resume_state"]["enabled"],
        "output_root": str(output_root),
        "operator_manifest": manifest,
        "intervention": dict(cfg["train"]["v6_causal_bundle"]),
        "grad_layer_monitor": dict(cfg["train"]["telemetry"]["grad_layer_monitor"]),
    }
    print("[v6-causal-bundle] stage=preflight", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[v6-causal-bundle] stage=dry-run-complete no_files_written=true", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=False)
    run_root = Path(cfg["training_artifacts"]["run_root_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    resolved = output_root / "resolved_run_config.json"
    operator_manifest_path = output_root / "two_full_operator_manifest.json"
    resolved.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    operator_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(
        f"[v6-causal-bundle] stage=train arm={args.arm} resolved={resolved} "
        f"manifest={operator_manifest_path}",
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
    expected_step = int(cfg["train"]["stop_after_step"])
    expected = checkpoint_root / f"step_{expected_step:07d}.pt"
    if not expected.is_file():
        raise RuntimeError(f"causal arm finished without exact final checkpoint {expected}")
    print(
        f"[v6-causal-bundle] stage=complete arm={args.arm} checkpoint={expected} "
        f"ledger={cfg['train']['v6_causal_bundle']['ledger_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
