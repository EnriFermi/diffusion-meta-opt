from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker


_SCHEMA_TO_ARCHITECTURE = {
    "weightclip_ae_v4_perirms_50k_v1": "latent_feedback_perirms_qknorm_v4",
    "weightclip_ae_v5_mandatory_bridge_50k_v1": "latent_mandatory_bridge_prenorm_v5",
    "weightclip_ae_v6_mandatory_bridge_posfilm_50k_v1": "latent_mandatory_bridge_posfilm_v6",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fresh deterministic Weight-AE architecture for 50k steps")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v4_perirms_50k.yaml"),
    )
    return parser.parse_args()


def _replace_artifact_root(value: Any, *, old_root: str, new_root: str) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _replace_artifact_root(child, old_root=old_root, new_root=new_root)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_artifact_root(child, old_root=old_root, new_root=new_root) for child in value]
    if isinstance(value, str) and value.startswith(old_root):
        return new_root + value[len(old_root) :]
    return value


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("v4 run config must be a mapping")
    schema = str(payload.get("schema", ""))
    if schema not in _SCHEMA_TO_ARCHITECTURE:
        raise ValueError(f"unexpected v4 config schema: {payload.get('schema')!r}")
    expected_architecture = _SCHEMA_TO_ARCHITECTURE[schema]
    if payload.get("architecture_version") != expected_architecture:
        raise ValueError(
            "run config has the wrong architecture_version: "
            f"expected {expected_architecture!r}, got {payload.get('architecture_version')!r}"
        )
    if int(payload.get("training_horizon_steps", 0)) != 500_000:
        raise ValueError("v4 scientific training horizon is frozen to exactly 500,000 steps")
    if int(payload.get("stop_after_step", 0)) != 50_000:
        raise ValueError("v4 first decision checkpoint is frozen to exactly 50,000 steps")
    return payload


def _assert_exact_production_contract(base: dict[str, Any], expected: dict[str, Any]) -> None:
    base_architecture = str(base["model"]["big_vae"].get("architecture_version", "legacy_v1"))
    if base_architecture != "latent_feedback_postnorm_v2":
        raise ValueError(
            "v4 must start from the frozen production scientific config, not a migrated model state; "
            f"got base architecture {base_architecture!r}"
        )
    train = base["train"]
    behavioral = train["behavioral_loss"]
    structural = train["struct_loss"]
    actual = {
        "slice_batch_size": int(train["slice_batch_size"]),
        "grad_accum_steps": int(train["grad_accum_steps"]),
        "learning_rate": float(train["lr"]),
        "scheduler_name": str(train["scheduler_name"]),
        "behavioral_coef": float(train["behavioral_coef"]),
        "behavioral_lambda_operator": float(behavioral["lambda_operator"]),
        "behavioral_lambda_dir": float(behavioral["lambda_dir"]),
        "behavioral_lambda_scale": float(behavioral["lambda_scale"]),
        "structural_coef": float(train["structural_coef"]),
        "structural_lambda_dir": float(structural["lambda_dir"]),
        "structural_lambda_scale": float(structural["lambda_scale"]),
        "structural_lambda_rec": float(structural["lambda_rec"]),
        "structural_lambda_rel": float(structural["lambda_rel"]),
    }
    normalized_expected = {
        key: (float(value) if isinstance(actual[key], float) else value)
        for key, value in expected.items()
    }
    if actual != normalized_expected:
        raise ValueError(f"base production scientific contract drifted: expected={normalized_expected} actual={actual}")
    if not bool(train["operator_bank"]["enabled"]):
        raise ValueError("v4 run requires the production operator bank")
    if not str(train["operator_bank"].get("pair_manifest", "")).strip():
        raise ValueError("v4 run requires the production operator-bank pair manifest")


def _build_worker_config(spec: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    base_path = Path(str(spec["base_resolved_config"])).expanduser().resolve()
    output_root = Path(str(spec["output_root"])).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh v4 output already exists: {output_root}")
    base = json.loads(base_path.read_text())
    if not isinstance(base, dict):
        raise TypeError("base resolved config must be a mapping")
    expected = spec.get("expected_production_contract")
    if not isinstance(expected, dict):
        raise TypeError("expected_production_contract must be a mapping")
    _assert_exact_production_contract(base, expected)

    old_root = str(base["training_artifacts"]["base_root_dir"])
    cfg = _replace_artifact_root(base, old_root=old_root, new_root=str(output_root))
    architecture = str(spec["architecture_version"])
    run_id = str(spec.get("run_id", "weightclip_ae_725m_lat384_perirms_qknorm_v4_50k"))
    if not run_id.strip():
        raise ValueError("v4 run_id must be non-empty")
    run_root = output_root / "runs" / run_id
    log_dir = run_root / "logs"
    checkpoint_dir = output_root / "checkpoints" / "train" / run_id
    stage_dir = checkpoint_dir / "stage_1"

    big_vae = cfg["model"]["big_vae"]
    big_vae.update(
        {
            "architecture_version": architecture,
            "use_latent_sampling": False,
            "use_encoder_mu_head": False,
            "normalize_latent_slots_before_mu": False,
        }
    )
    if architecture in {
        "latent_mandatory_bridge_prenorm_v5",
        "latent_mandatory_bridge_posfilm_v6",
    }:
        big_vae.update(
            {
                "decoder_bridge_attn_dim": int(spec.get("decoder_bridge_attn_dim", 512)),
                "decoder_bridge_n_heads": int(spec.get("decoder_bridge_n_heads", 8)),
            }
        )
    train = cfg["train"]
    train.update(
        {
            "max_steps": int(spec["training_horizon_steps"]),
            "stop_after_step": int(spec["stop_after_step"]),
            "grad_accum_steps": int(spec.get("grad_accum_steps", train["grad_accum_steps"])),
            "kl_beta": 0.0,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_every": 10_000,
            "resume_checkpoint": "",
        }
    )
    loss = spec.get("loss")
    if loss is not None:
        if not isinstance(loss, dict):
            raise TypeError("v4 loss override must be a mapping")
        train["behavioral_coef"] = float(loss["behavioral_coef"])
        train["structural_coef"] = float(loss["structural_coef"])
        train["struct_loss"].update(
            {
                "lambda_dir": float(loss["structural_lambda_dir"]),
                "lambda_scale": float(loss["structural_lambda_scale"]),
                "lambda_rec": float(loss["structural_lambda_rec"]),
                "lambda_rel": float(loss["structural_lambda_rel"]),
            }
        )
    train["kl_schedule"]["enabled"] = False
    train["latent_sampling_gate"]["enabled"] = False
    train["bounded_runtime_profile"]["enabled"] = False
    train["resume_state"].update(
        {
            "enabled": True,
            "auto_resume": False,
            "explicit_checkpoint": "",
            "dir": str(stage_dir / "resume_state"),
            "save_every": 1_000,
        }
    )
    train["telemetry"]["comet"].update(
        {
            "experiment_name": run_id,
            "tags": list(
                spec.get(
                    "comet_tags",
                    ["fresh", "deterministic-ae", "peri-rms", "qk-norm", "50k"],
                )
            ),
        }
    )
    cfg["data"]["seed"] = int(spec["seed"])
    cfg["run_config"]["name"] = run_id
    cfg["weightclip_runtime_profile_preflight"] = {}
    cfg["logging"].update(
        {
            "dir": str(log_dir),
            "file_path": str(log_dir / "train_rank0.log"),
            "file_name": "train_rank0.log",
        }
    )
    cfg["training_artifacts"].update(
        {
            "base_root_dir": str(output_root),
            "root_dir": str(output_root),
            "run_id": run_id,
            "run_root_dir": str(run_root),
            "runs_dir": str(output_root / "runs"),
            "logs_dir": str(log_dir),
            "reports_dir": str(run_root / "reports"),
            "crashes_dir": str(run_root / "crashes"),
            "checkpoints_dir": str(output_root / "checkpoints"),
            "big_vae_checkpoint_dir": str(checkpoint_dir),
            "tmp_dir": str(output_root / "tmp"),
        }
    )
    cfg["collector"]["diagnostics"].update(
        {
            "crash_report_path": str(run_root / "crashes" / "collector_crash_report.json"),
            "worker_status_dir": str(run_root / "reports" / "dataset_workers"),
        }
    )
    cfg["train"]["fixed_training_batch"]["dump_path"] = str(stage_dir / "fixed_training_batch.pt")
    grad = cfg["train"]["telemetry"]["grad_layer_monitor"]
    grad.update(
        {
            "csv_path": str(stage_dir / "grad_layer_rms.csv"),
            "plot_path": str(stage_dir / "grad_layer_rms.png"),
            "heatmap_path": str(stage_dir / "grad_layer_rms_heatmap.png"),
            "reset_csv_on_start": True,
        }
    )
    noise = spec.get("gradient_noise_monitor", {})
    if not isinstance(noise, dict):
        raise TypeError("gradient_noise_monitor spec must be a mapping")
    train["telemetry"]["gradient_noise_monitor"] = {
        "enabled": bool(noise.get("enabled", False)),
        "run_at_start": bool(noise.get("run_at_start", False)),
        "every_steps": int(noise.get("every_steps", 1_000)),
        "start_logical_index": int(noise.get("start_logical_index", 7_000_000)),
        "panel_size": int(noise.get("panel_size", 65_536)),
        "csv_path": str(stage_dir / "gradient_noise_monitor.csv"),
        "sample_manifest_path": str(stage_dir / "gradient_noise_samples.json"),
    }

    if big_vae["use_latent_sampling"] or big_vae["use_encoder_mu_head"]:
        raise AssertionError("v4 run must remain deterministic")
    if big_vae["normalize_latent_slots_before_mu"]:
        raise AssertionError("v4 run must keep the raw latent bottleneck (no flattened latent normalization)")
    if train["kl_beta"] != 0.0 or train["kl_schedule"]["enabled"] or train["latent_sampling_gate"]["enabled"]:
        raise AssertionError("v4 run must keep KL and latent sampling schedules disabled")
    if train["max_steps"] != 500_000 or train["stop_after_step"] != 50_000:
        raise AssertionError("v4 run must preserve the 500k horizon and stop first at the 50k decision checkpoint")
    if train["resume_checkpoint"] or train["resume_state"]["explicit_checkpoint"]:
        raise AssertionError("v4 is a fresh run and must not consume v2/v3 resume state")
    if int(train["grad_accum_steps"]) <= 0:
        raise ValueError("v4 grad_accum_steps must be positive")
    return cfg, output_root


def main() -> None:
    args = _parse_args()
    spec = _load_spec(args.config.expanduser().resolve())
    cfg, output_root = _build_worker_config(spec)
    output_root.mkdir(parents=True, exist_ok=False)
    run_root = Path(cfg["training_artifacts"]["run_root_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_run_config.json"
    resolved_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(
        "[weight-ae] stage=startup mode=fresh deterministic=true "
        f"architecture={cfg['model']['big_vae']['architecture_version']} "
        f"device={cfg['train']['device']} seed={cfg['data']['seed']} "
        f"training_horizon={cfg['train']['max_steps']} stop_after={cfg['train']['stop_after_step']} "
        f"physical_batch={cfg['train']['slice_batch_size']} "
        f"grad_accum={cfg['train']['grad_accum_steps']} "
        f"effective_batch={int(cfg['train']['slice_batch_size']) * int(cfg['train']['grad_accum_steps'])} "
        f"behavioral_coef={cfg['train']['behavioral_coef']} structural_coef={cfg['train']['structural_coef']} "
        f"lr={cfg['train']['lr']} output={output_root}",
        flush=True,
    )
    print(f"[weight-ae] resolved_config={resolved_path}", flush=True)
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=cfg,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )
    expected_checkpoint = Path(cfg["train"]["checkpoint_dir"]) / "stage_1" / "step_0050000.pt"
    if not expected_checkpoint.is_file():
        raise RuntimeError(f"v4 run finished without expected checkpoint: {expected_checkpoint}")
    print(f"[weight-ae] stage=complete checkpoint={expected_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
