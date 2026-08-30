from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the fresh WeightClip PreNorm-v3 VAE")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v3_vae_50k.yaml"),
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-after-step", type=int, default=None)
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


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("VAE run config must be a mapping")
    if raw.get("schema") != "weightclip_ae_v3_vae_50k_v1":
        raise ValueError("unexpected VAE run config schema")
    if str(raw.get("architecture_version")) != "latent_feedback_prenorm_v3":
        raise ValueError("VAE run requires latent_feedback_prenorm_v3")
    return raw


def _build_worker_config(
    spec: dict[str, Any],
    *,
    resume_checkpoint: Path | None,
    max_steps_override: int | None,
    stop_after_step_override: int | None,
) -> tuple[dict[str, Any], Path, bool]:
    base_path = Path(str(spec["base_resolved_config"])).resolve()
    output_root = Path(str(spec["output_root"])).resolve()
    is_resume = resume_checkpoint is not None
    if not is_resume and output_root.exists():
        raise FileExistsError(f"fresh VAE output already exists: {output_root}")
    if is_resume:
        resume_checkpoint = resume_checkpoint.resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
        if not output_root.is_dir():
            raise FileNotFoundError(f"resume output root does not exist: {output_root}")

    base = json.loads(base_path.read_text())
    if not isinstance(base, dict):
        raise TypeError("base resolved config must be a mapping")
    old_root = str(base["training_artifacts"]["base_root_dir"])
    cfg = _replace_artifact_root(base, old_root=old_root, new_root=str(output_root))

    configured_horizon = int(spec["training_horizon_steps"])
    max_steps = int(max_steps_override) if max_steps_override is not None else configured_horizon
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    stop_after_step = (
        int(stop_after_step_override)
        if stop_after_step_override is not None
        else int(spec["stop_after_step"])
    )
    if stop_after_step < 1 or stop_after_step > max_steps:
        raise ValueError("stop_after_step must be in [1, max_steps]")

    run_id = "weightclip_ae_725m_lat384_prenorm_v3_vae"
    run_root = output_root / "runs" / run_id
    log_dir = run_root / "logs"
    checkpoint_dir = output_root / "checkpoints" / "train" / "weightclip_gpu_1280_lat384_prenorm_v3_vae"
    stage_dir = checkpoint_dir / "stage_1"

    big_vae = cfg["model"]["big_vae"]
    big_vae.update(
        {
            "architecture_version": "latent_feedback_prenorm_v3",
            "use_latent_sampling": True,
            "use_encoder_mu_head": True,
            "normalize_latent_slots_before_mu": False,
            "latent_prior_kind": "gaussian",
            "latent_sampling_min_std": 0.0,
            "latent_sampling_logvar_min": -6.0,
            "latent_sampling_logvar_max": 4.0,
        }
    )

    train = cfg["train"]
    train.update(
        {
            "max_steps": max_steps,
            "stop_after_step": stop_after_step,
            "slice_batch_size": int(spec["slice_batch_size"]),
            "grad_accum_steps": int(spec["grad_accum_steps"]),
            "lr": float(spec["learning_rate"]),
            "weight_decay": float(spec["weight_decay"]),
            "weight_decay_2d_only": bool(spec["weight_decay_2d_only"]),
            "scheduler_name": str(spec["scheduler_name"]),
            "warmup_steps": 0,
            "kl_beta": float(spec["kl_beta"]),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_every": int(spec["checkpoint_every"]),
            "resume_checkpoint": "",
            "log_every": int(spec["log_every"]),
            "patch_tokenizer_alpha_lr": 0.0,
            "patch_tokenizer_alpha_weight_decay": 0.0,
        }
    )
    train["kl_schedule"].update({"enabled": False, "start_beta": float(spec["kl_beta"])})
    train["latent_sampling_gate"].update(
        {
            "enabled": True,
            "start_step": int(spec["sampling_gate_start_step"]),
            "ramp_steps": int(spec["sampling_gate_ramp_steps"]),
            "start_value": 0.0,
            "end_value": 1.0,
        }
    )
    train["bounded_runtime_profile"]["enabled"] = False
    train["resume_state"].update(
        {
            "enabled": True,
            "auto_resume": False,
            "explicit_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
            "dir": str(stage_dir / "resume_state"),
            "save_every": int(spec["resume_state_every"]),
            "load_model_state": True,
            "load_optimizer_state": True,
            "load_scheduler_state": True,
            "load_scaler_state": True,
            "load_rng_state": True,
            "load_step": True,
        }
    )
    train["telemetry"]["comet"].update(
        {
            "enabled": True,
            "experiment_name": run_id,
            "tags": ["vae", "fresh" if not is_resume else "resume", "latent_feedback_prenorm_v3", "50k"],
        }
    )
    train["telemetry"]["wandb"]["enabled"] = False
    train["telemetry"]["grad_layer_monitor"].update(
        {
            "enabled": True,
            "every_steps": int(spec["gradient_monitor_every"]),
            "csv_path": str(stage_dir / "grad_layer_rms.csv"),
            "save_csv": True,
            "save_plot": False,
            "save_heatmap": False,
            "weights_only": False,
            "reset_csv_on_start": not is_resume,
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
    return cfg, output_root, is_resume


def main() -> None:
    args = _parse_args()
    spec = _load_config(args.config.resolve())
    cfg, output_root, is_resume = _build_worker_config(
        spec,
        resume_checkpoint=args.resume_checkpoint,
        max_steps_override=args.max_steps,
        stop_after_step_override=args.stop_after_step,
    )
    output_root.mkdir(parents=True, exist_ok=is_resume)
    run_root = Path(cfg["training_artifacts"]["run_root_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / f"resolved_run_config_to_step_{int(cfg['train']['stop_after_step']):07d}.json"
    resolved_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])

    print("[ae-v3-vae] stage=startup", flush=True)
    print(
        "[ae-v3-vae] "
        f"mode={'resume' if is_resume else 'fresh'} architecture={cfg['model']['big_vae']['architecture_version']} "
        f"device={cfg['train']['device']} seed={cfg['data']['seed']} "
        f"horizon={cfg['train']['max_steps']} stop_after={cfg['train']['stop_after_step']} "
        f"batch={cfg['train']['slice_batch_size']} lr={cfg['train']['lr']} beta={cfg['train']['kl_beta']} "
        f"sampling_ramp={cfg['train']['latent_sampling_gate']['ramp_steps']} output={output_root}",
        flush=True,
    )
    print(f"[ae-v3-vae] resolved_config={resolved_path}", flush=True)
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=cfg,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )
    expected_checkpoint = (
        Path(cfg["train"]["checkpoint_dir"])
        / "stage_1"
        / f"step_{int(cfg['train']['stop_after_step']):07d}.pt"
    )
    if not expected_checkpoint.is_file():
        raise RuntimeError(f"VAE run finished without expected checkpoint: {expected_checkpoint}")
    print(f"[ae-v3-vae] stage=complete checkpoint={expected_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
