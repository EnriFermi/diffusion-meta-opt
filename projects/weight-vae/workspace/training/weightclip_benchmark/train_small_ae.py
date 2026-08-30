from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from training.big_vae.worker import _run_worker
from training.weightclip_benchmark.replay_then_resume_ae import _load_mapping, _prepare_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fast latent-feedback AE for 500k steps.")
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--model-override", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def _run(cfg: Any) -> None:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("resolved training config must be a mapping")
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=payload,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )


def main() -> None:
    args = _parse_args()
    profile = args.profile_config.expanduser().resolve()
    production = args.production_config.expanduser().resolve()
    override_path = args.model_override.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    resume_checkpoint = (
        args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint is not None else None
    )

    override = OmegaConf.load(override_path)
    cfg = _prepare_cfg(
        base=_load_mapping(profile),
        production=_load_mapping(production),
        artifact_root=artifact_root,
        checkpoint_root=checkpoint_root,
        run_id=("weightclip_ae_small_fast_v2_resume" if resume_checkpoint else "weightclip_ae_small_fast_v2_500k"),
        replay_step=500_000,
        resume_checkpoint=resume_checkpoint,
        architecture_version="latent_feedback_postnorm_v2",
    )
    with open_dict(cfg):
        cfg.model = OmegaConf.merge(cfg.model, override.model)
        cfg.train.max_steps = 500_000
        cfg.train.stop_after_step = 0
        cfg.train.lr = 3e-4
        cfg.train.scheduler_name = "constant"
        cfg.train.checkpoint_every = 10_000
        cfg.train.resume_state.enabled = True
        cfg.train.resume_state.auto_resume = False
        cfg.train.resume_state.save_every = 1_000
        cfg.train.resume_state.explicit_checkpoint = str(resume_checkpoint) if resume_checkpoint else ""
        cfg.train.telemetry.comet.enabled = True
        cfg.train.telemetry.comet.api_key = os.environ.get("COMET_API_KEY", "")
        cfg.train.telemetry.comet.workspace = os.environ.get("COMET_WORKSPACE", "")
        cfg.train.telemetry.comet.project_name = "big_weight_vae"
        cfg.train.telemetry.comet.experiment_name = "weightclip-ae-small-fast-v2-500k"
        cfg.train.telemetry.comet.tags = ["production", "small-fast-v2", "fresh-500k"]
        cfg.train.telemetry.wandb.enabled = False

    resume_dir = checkpoint_root / "stage_1" / "resume_state"
    if resume_checkpoint is None and any(resume_dir.glob("step_*.pt")):
        raise RuntimeError(f"fresh start refused because resume checkpoints already exist: {resume_dir}")

    print(
        "[small-ae] start="
        f"{'resume' if resume_checkpoint else 'fresh-step-0'} max_steps=500000 lr=3e-4 "
        f"model_override={override_path} checkpoint_root={checkpoint_root}",
        flush=True,
    )
    _run(cfg)


if __name__ == "__main__":
    main()

