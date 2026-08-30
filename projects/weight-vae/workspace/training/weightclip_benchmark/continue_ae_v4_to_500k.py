from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker


_ARCHITECTURE = "latent_feedback_perirms_qknorm_v4"
_FINAL_STEP = 500_000
_HANDOFF_STEP = 50_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the bounded v4 run, then resume its exact state from 50k to 500k."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, required=True)
    parser.add_argument("--comet-experiment-key", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def _inherit_tracking_environment(pid: int) -> None:
    environ_path = Path(f"/proc/{pid}/environ")
    if not environ_path.is_file():
        return
    entries = environ_path.read_bytes().split(b"\0")
    for key in ("COMET_API_KEY", "COMET_WORKSPACE"):
        if os.environ.get(key):
            continue
        prefix = f"{key}=".encode()
        for entry in entries:
            if entry.startswith(prefix):
                os.environ[key] = entry[len(prefix) :].decode()
                break


def _wait_for_pid(pid: int, *, poll_seconds: float) -> None:
    started = time.monotonic()
    next_report = started
    while Path(f"/proc/{pid}").exists():
        now = time.monotonic()
        if now >= next_report:
            print(
                f"[ae-v4-continuation] stage=waiting pid={pid} elapsed_s={now - started:.0f}",
                flush=True,
            )
            next_report = now + 600.0
        time.sleep(max(1.0, poll_seconds))


def _resume_path(output_root: Path) -> Path:
    run_id = "weightclip_ae_725m_lat384_perirms_qknorm_v4_50k"
    return (
        output_root
        / "checkpoints"
        / "train"
        / run_id
        / "stage_1"
        / "resume_state"
        / f"step_{_HANDOFF_STEP:07d}.pt"
    )


def _model_checkpoint_path(output_root: Path, step: int) -> Path:
    run_id = "weightclip_ae_725m_lat384_perirms_qknorm_v4_50k"
    return output_root / "checkpoints" / "train" / run_id / "stage_1" / f"step_{step:07d}.pt"


def _build_continuation_config(
    *,
    output_root: Path,
    resume_path: Path,
    comet_experiment_key: str,
) -> dict[str, Any]:
    resolved_path = output_root / "resolved_run_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"missing original resolved config: {resolved_path}")
    if not resume_path.is_file():
        raise FileNotFoundError(f"missing exact 50k resume state: {resume_path}")
    model_checkpoint = _model_checkpoint_path(output_root, _HANDOFF_STEP)
    if not model_checkpoint.is_file():
        raise FileNotFoundError(f"missing exact 50k model checkpoint: {model_checkpoint}")

    cfg = json.loads(resolved_path.read_text())
    if cfg["model"]["big_vae"]["architecture_version"] != _ARCHITECTURE:
        raise ValueError("continuation config is not the approved v4 architecture")
    train = cfg["train"]
    if int(train["max_steps"]) != _FINAL_STEP:
        raise ValueError("continuation must preserve the original 500k scheduler horizon")
    if str(train["scheduler_name"]).strip().lower() not in {"constant", "constant_lr"}:
        raise ValueError("v4 continuation requires the original constant scheduler")

    train["stop_after_step"] = _FINAL_STEP
    # The rolling exact resume remains at 1k. Keeping full inference-model
    # checkpoints every 50k avoids exhausting the current filesystem.
    train["checkpoint_every"] = 50_000
    train["resume_checkpoint"] = ""
    train["resume_state"].update(
        {
            "enabled": True,
            "auto_resume": False,
            "explicit_checkpoint": str(resume_path),
            "load_model_state": True,
            "load_optimizer_state": True,
            "load_scheduler_state": True,
            "load_scaler_state": True,
            "load_rng_state": True,
            "load_step": True,
        }
    )
    train["telemetry"]["grad_layer_monitor"]["reset_csv_on_start"] = False
    comet = train["telemetry"]["comet"]
    comet["existing_experiment_key"] = comet_experiment_key
    comet["tags"] = list(dict.fromkeys([*comet.get("tags", []), "resumed-50k-to-500k"]))
    return cfg


def _validate_resume_step(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    try:
        step = int(payload.get("step", -1))
    finally:
        del payload
    if step != _HANDOFF_STEP:
        raise RuntimeError(f"expected resume step {_HANDOFF_STEP}, got {step}: {path}")


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    pid = int(args.wait_for_pid)
    _inherit_tracking_environment(pid)
    print(
        f"[ae-v4-continuation] stage=armed pid={pid} output={output_root} "
        f"handoff_step={_HANDOFF_STEP} final_step={_FINAL_STEP}",
        flush=True,
    )
    _wait_for_pid(pid, poll_seconds=float(args.poll_seconds))

    resume_path = _resume_path(output_root)
    _validate_resume_step(resume_path)
    cfg = _build_continuation_config(
        output_root=output_root,
        resume_path=resume_path,
        comet_experiment_key=str(args.comet_experiment_key),
    )
    continuation_config = output_root / "resolved_run_config_resume_0050000_to_0500000.json"
    continuation_config.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(
        f"[ae-v4-continuation] stage=resuming resume={resume_path} "
        f"resolved_config={continuation_config}",
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
    final_checkpoint = _model_checkpoint_path(output_root, _FINAL_STEP)
    if not final_checkpoint.is_file():
        raise RuntimeError(f"continuation ended without final checkpoint: {final_checkpoint}")
    print(f"[ae-v4-continuation] stage=complete checkpoint={final_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
