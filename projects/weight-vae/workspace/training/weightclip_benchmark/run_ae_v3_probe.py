from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker


PROBE_STEPS = 500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fresh bounded 500-step Weight-AE v3 probe")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v3_500step_probe.yaml"),
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


def _load_probe_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("probe config must be a mapping")
    if raw.get("schema") != "weightclip_ae_v3_500step_probe_v1":
        raise ValueError("unexpected probe config schema")
    if int(raw.get("steps", -1)) != PROBE_STEPS:
        raise ValueError(f"probe is hard-capped at exactly {PROBE_STEPS} optimizer steps")
    if str(raw.get("architecture_version")) != "latent_feedback_prenorm_v3":
        raise ValueError("probe requires latent_feedback_prenorm_v3")
    return raw


def _build_worker_config(probe: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    base_path = Path(str(probe["base_resolved_config"])).resolve()
    output_root = Path(str(probe["output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh probe output already exists: {output_root}")

    base = json.loads(base_path.read_text())
    if not isinstance(base, dict):
        raise TypeError("base resolved config must be a mapping")
    old_root = str(base["training_artifacts"]["base_root_dir"])
    cfg = _replace_artifact_root(base, old_root=old_root, new_root=str(output_root))

    run_id = "weightclip_ae_725m_lat384_prenorm_v3_fresh_500step"
    run_root = output_root / "runs" / run_id
    log_dir = run_root / "logs"
    checkpoint_dir = output_root / "checkpoints" / "train" / "weightclip_gpu_1280_lat384_prenorm_v3"
    stage_dir = checkpoint_dir / "stage_1"

    cfg["model"]["big_vae"]["architecture_version"] = "latent_feedback_prenorm_v3"
    train = cfg["train"]
    train.update(
        {
            "max_steps": PROBE_STEPS,
            "slice_batch_size": int(probe["slice_batch_size"]),
            "grad_accum_steps": int(probe["grad_accum_steps"]),
            "lr": float(probe["learning_rate"]),
            "weight_decay": float(probe["weight_decay"]),
            "weight_decay_2d_only": bool(probe["weight_decay_2d_only"]),
            "scheduler_name": str(probe["scheduler_name"]),
            "warmup_steps": 0,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_every": int(probe["checkpoint_every"]),
            "resume_checkpoint": "",
            "log_every": int(probe["log_every"]),
            "patch_tokenizer_alpha_lr": 0.0,
            "patch_tokenizer_alpha_weight_decay": 0.0,
        }
    )
    train["bounded_runtime_profile"]["enabled"] = False
    train["resume_state"].update(
        {
            "enabled": True,
            "auto_resume": False,
            "explicit_checkpoint": "",
            "dir": str(stage_dir / "resume_state"),
            "save_every": int(probe["resume_state_every"]),
        }
    )
    train["telemetry"]["comet"].update(
        {
            "enabled": False,
            "experiment_name": run_id,
            "tags": ["probe", "fresh", "latent_feedback_prenorm_v3", "500-step"],
        }
    )
    train["telemetry"]["wandb"]["enabled"] = False
    train["telemetry"]["grad_layer_monitor"].update(
        {
            "enabled": True,
            "every_steps": int(probe["gradient_monitor_every"]),
            "csv_path": str(stage_dir / "grad_layer_rms.csv"),
            "save_csv": True,
            "save_plot": False,
            "save_heatmap": False,
            "reset_csv_on_start": True,
        }
    )

    cfg["data"]["seed"] = int(probe["seed"])
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
    return cfg, output_root


def main() -> None:
    args = _parse_args()
    probe = _load_probe_config(args.config.resolve())
    cfg, output_root = _build_worker_config(probe)
    output_root.mkdir(parents=True, exist_ok=False)
    run_root = Path(cfg["training_artifacts"]["run_root_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_probe_config.json"
    resolved_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])

    print("[ae-v3-probe] stage=startup", flush=True)
    print(
        "[ae-v3-probe] "
        f"architecture={cfg['model']['big_vae']['architecture_version']} "
        f"device={cfg['train']['device']} "
        f"amp={cfg['train'].get('amp_dtype', cfg['train'].get('amp', 'auto'))} "
        f"seed={cfg['data']['seed']} steps={cfg['train']['max_steps']} "
        f"batch={cfg['train']['slice_batch_size']} lr={cfg['train']['lr']} "
        f"scheduler={cfg['train']['scheduler_name']} output={output_root}",
        flush=True,
    )
    print(f"[ae-v3-probe] resolved_config={resolved_path}", flush=True)
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=cfg,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )
    expected_checkpoint = Path(cfg["train"]["checkpoint_dir"]) / "stage_1" / "step_0000500.pt"
    if not expected_checkpoint.is_file():
        raise RuntimeError(f"500-step probe finished without expected checkpoint: {expected_checkpoint}")
    print(f"[ae-v3-probe] stage=complete checkpoint={expected_checkpoint}", flush=True)
    print(f"[ae-v3-probe] artifacts={output_root}", flush=True)


if __name__ == "__main__":
    main()
