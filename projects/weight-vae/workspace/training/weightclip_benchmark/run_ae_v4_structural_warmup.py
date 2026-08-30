from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker
from training.weightclip_benchmark.run_ae_v4_perirms_50k import _replace_artifact_root


_SCHEMA = "weightclip_ae_v4_structural_warmup_v1"
_ARCHITECTURE = "latent_feedback_perirms_qknorm_v4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume v4 with behavioral gradients disabled and structural direction/scale only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v4_structural_warmup_from159k.yaml"),
    )
    return parser.parse_args()


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError(f"unexpected structural-warmup config schema: {payload!r}")
    start = int(payload["start_step"])
    stop = int(payload["stop_after_step"])
    horizon = int(payload["training_horizon_steps"])
    if not (0 < start < stop <= horizon == 500_000):
        raise ValueError(f"invalid structural-warmup interval: start={start} stop={stop} horizon={horizon}")
    return payload


def _validate_resume(path: Path, *, expected_step: int) -> None:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    try:
        if int(payload.get("step", -1)) != expected_step:
            raise ValueError(f"resume step mismatch: expected={expected_step} actual={payload.get('step')}")
        required = (
            "model_state",
            "optimizer_state",
            "optimizer_param_names",
            "scheduler_state",
            "scaler_state",
            "rng_state",
            "config",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"resume checkpoint is incomplete: missing={missing}")
    finally:
        del payload


def _build_worker_config(spec: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    source_config = Path(str(spec["source_resolved_config"])).expanduser().resolve()
    resume_path = Path(str(spec["source_resume_checkpoint"])).expanduser().resolve()
    output_root = Path(str(spec["output_root"])).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"structural-warmup output already exists: {output_root}")
    _validate_resume(resume_path, expected_step=int(spec["start_step"]))

    source = json.loads(source_config.read_text())
    old_root = str(source["training_artifacts"]["base_root_dir"])
    cfg = _replace_artifact_root(source, old_root=old_root, new_root=str(output_root))
    run_id = str(spec["run_id"])
    run_root = output_root / "runs" / run_id
    log_dir = run_root / "logs"
    checkpoint_dir = output_root / "checkpoints" / "train" / run_id
    stage_dir = checkpoint_dir / "stage_1"

    big_vae = cfg["model"]["big_vae"]
    if str(big_vae["architecture_version"]) != _ARCHITECTURE:
        raise ValueError("source checkpoint is not the v4 architecture")
    if bool(big_vae["use_latent_sampling"]) or bool(big_vae["use_encoder_mu_head"]):
        raise ValueError("structural warmup must remain a deterministic AE")

    loss = spec["loss"]
    train = cfg["train"]
    train.update(
        {
            "max_steps": int(spec["training_horizon_steps"]),
            "stop_after_step": int(spec["stop_after_step"]),
            "grad_accum_steps": int(spec.get("grad_accum_steps", train.get("grad_accum_steps", 1))),
            "behavioral_coef": float(loss["behavioral_coef"]),
            "structural_coef": float(loss["structural_coef"]),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_every": 10_000,
            "resume_checkpoint": "",
            "kl_beta": 0.0,
        }
    )
    structural = train["struct_loss"]
    structural.update(
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
            "explicit_checkpoint": str(resume_path),
            "dir": str(stage_dir / "resume_state"),
            "save_every": 1_000,
            "load_model_state": True,
            "load_optimizer_state": True,
            "load_scheduler_state": True,
            "load_scaler_state": True,
            "load_rng_state": True,
            "load_step": True,
        }
    )
    target_accum = int(train["grad_accum_steps"])
    source_accum = int(spec.get("source_grad_accum_steps", target_accum))
    if target_accum != source_accum:
        start_logical_index = int(spec["operator_stream_start_logical_index"])
        logical_index_offset = (
            start_logical_index
            - int(spec["start_step"]) * int(train["slice_batch_size"]) * target_accum
        )
        train["operator_bank"]["logical_index_offset"] = logical_index_offset
        train["resume_state"]["operator_stream_transition"] = {
            "enabled": True,
            "source_step": int(spec["start_step"]),
            "start_logical_index": start_logical_index,
            "source_grad_accum_steps": source_accum,
            "target_grad_accum_steps": target_accum,
        }

    comet = train["telemetry"]["comet"]
    comet.update(
        {
            "existing_experiment_key": "",
            "experiment_name": run_id,
            "tags": [
                "v4",
                "structural-only",
                f"resume-{int(spec['start_step'])}",
                f"effective-batch-{int(train['slice_batch_size']) * int(train['grad_accum_steps'])}",
                "gradient-noise-monitor" if bool(spec.get("gradient_noise_monitor", {}).get("enabled", False)) else "causal-warmup",
            ],
        }
    )
    train["telemetry"]["wandb"]["enabled"] = False
    cfg["run_config"]["name"] = run_id
    cfg["logging"].update(
        {"dir": str(log_dir), "file_path": str(log_dir / "train_rank0.log"), "file_name": "train_rank0.log"}
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
    train["fixed_training_batch"]["dump_path"] = str(stage_dir / "fixed_training_batch.pt")
    grad = train["telemetry"]["grad_layer_monitor"]
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

    exact_loss = {
        "behavioral_coef": float(train["behavioral_coef"]),
        "structural_coef": float(train["structural_coef"]),
        "structural_dir": float(structural["lambda_dir"]),
        "structural_scale": float(structural["lambda_scale"]),
        "structural_rec": float(structural["lambda_rec"]),
        "structural_rel": float(structural["lambda_rel"]),
    }
    expected_loss = {
        "behavioral_coef": 0.0,
        "structural_coef": 1.0,
        "structural_dir": 1.0,
        "structural_scale": 10.0,
        "structural_rec": 0.0,
        "structural_rel": 0.0,
    }
    if exact_loss != expected_loss:
        raise ValueError(f"structural-only loss contract mismatch: {exact_loss}")
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
        "[ae-v4-structural] stage=startup mode=resume objective=structural_dir+scale "
        f"start={spec['start_step']} stop={spec['stop_after_step']} horizon={cfg['train']['max_steps']} "
        f"device={cfg['train']['device']} dtype={cfg['train'].get('amp_dtype', 'auto')} "
        f"seed={cfg['data']['seed']} physical_batch={cfg['train']['slice_batch_size']} "
        f"grad_accum={cfg['train']['grad_accum_steps']} "
        f"effective_batch={int(cfg['train']['slice_batch_size']) * int(cfg['train']['grad_accum_steps'])} "
        f"lr={cfg['train']['lr']} "
        f"resume={cfg['train']['resume_state']['explicit_checkpoint']} output={output_root}",
        flush=True,
    )
    print(f"[ae-v4-structural] resolved_config={resolved_path}", flush=True)
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=cfg,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )
    stop = int(spec["stop_after_step"])
    expected_resume = Path(cfg["train"]["resume_state"]["dir"]) / f"step_{stop:07d}.pt"
    if not expected_resume.is_file():
        raise RuntimeError(f"structural warmup ended without exact resume: {expected_resume}")
    print(f"[ae-v4-structural] stage=complete resume={expected_resume}", flush=True)


if __name__ == "__main__":
    main()
