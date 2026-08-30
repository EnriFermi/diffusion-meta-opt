from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from big_vae.models import build_weight_quantile_vae
from training.big_vae.checkpointing import _save_checkpoint_payload
from training.big_vae.runtime import _build_model_cfg, _build_optimizer
from training.big_vae.runtime import _configure_run_artifacts
from training.big_vae.worker import _run_worker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a lost AE prefix, checkpoint it, then resume with a constant LR."
    )
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--replay-step", type=int, default=0)
    parser.add_argument("--source-resume", type=Path)
    parser.add_argument(
        "--architecture-version",
        choices=("legacy_v1", "latent_feedback_postnorm_v2"),
        default="legacy_v1",
    )
    return parser.parse_args()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping config: {path}")
    return payload


def _prepare_cfg(
    *,
    base: dict[str, Any],
    production: dict[str, Any],
    artifact_root: Path,
    checkpoint_root: Path,
    run_id: str,
    replay_step: int,
    resume_checkpoint: Path | None,
    architecture_version: str = "legacy_v1",
) -> DictConfig:
    cfg = OmegaConf.create(base)
    with open_dict(cfg):
        cfg.hf.token = os.environ.get("HF_TOKEN", "")
        cfg.logging = OmegaConf.create(production.get("logging", {}))
        cfg.model.big_vae.architecture_version = architecture_version
        cfg.model.big_vae.latent_feedback_start_layer = 1
        cfg.model.big_vae.latent_feedback_attn_dim = 512
        cfg.model.big_vae.latent_feedback_n_heads = 16

        cfg.training_artifacts.base_root_dir = str(artifact_root)
        cfg.training_artifacts.root_dir = str(artifact_root)
        cfg.training_artifacts.runs_dir = str(artifact_root / "runs")
        cfg.training_artifacts.run_id = run_id
        cfg.training_artifacts.checkpoints_dir = str(artifact_root / "checkpoints")
        cfg.training_artifacts.datasets_dir = str(artifact_root / "datasets")
        cfg.training_artifacts.eval_dir = str(artifact_root / "eval")
        cfg.training_artifacts.tmp_dir = str(artifact_root / "tmp")
        cfg.training_artifacts.big_vae_checkpoint_dir = str(checkpoint_root)
        cfg.training_artifacts.big_vae_offline_dataset_base_dir = str(
            artifact_root / "datasets" / "offline" / "big_vae"
        )
        cfg.training_artifacts.big_vae_presliced_dataset_base_dir = str(
            artifact_root / "datasets" / "presliced" / "big_vae" / "default"
        )

        cfg.train.bounded_runtime_profile.enabled = False
        cfg.train.max_steps = 500_000
        cfg.train.stop_after_step = replay_step if resume_checkpoint is None else 0
        cfg.train.scheduler_name = "cosine" if resume_checkpoint is None else "constant"
        cfg.train.lr = 5e-5
        cfg.train.checkpoint_every = 10_000
        cfg.train.checkpoint_dir = str(checkpoint_root)
        cfg.train.resume_checkpoint = ""
        cfg.train.resume_state.enabled = True
        cfg.train.resume_state.auto_resume = False
        cfg.train.resume_state.save_every = 1_000
        cfg.train.resume_state.dir = str(checkpoint_root / "stage_1" / "resume_state")
        cfg.train.resume_state.explicit_checkpoint = (
            str(resume_checkpoint) if resume_checkpoint is not None else ""
        )
        cfg.train.resume_state.load_model_state = True
        cfg.train.resume_state.load_optimizer_state = True
        cfg.train.resume_state.load_scheduler_state = True
        cfg.train.resume_state.load_scaler_state = True
        cfg.train.resume_state.load_rng_state = True
        cfg.train.resume_state.load_step = True

        comet = cfg.train.telemetry.comet
        comet.enabled = resume_checkpoint is not None
        comet.api_key = os.environ.get("COMET_API_KEY", "")
        comet.workspace = os.environ.get("COMET_WORKSPACE", "")
        comet.project_name = "big_weight_vae"
        comet.experiment_name = (
            f"weightclip-ae-704m-lat384-{architecture_version}-resume-step{replay_step}"
            if resume_checkpoint is not None
            else ""
        )
        comet.tags = ["production", "constant-lr", architecture_version, f"resumed-step-{replay_step}"]
        cfg.train.telemetry.wandb.enabled = False

    run_artifacts = _configure_run_artifacts(cfg)
    with open_dict(cfg):
        cfg.train.checkpoint_dir = str(checkpoint_root)
        cfg.train.resume_state.dir = str(checkpoint_root / "stage_1" / "resume_state")
        cfg.train.fixed_training_batch.dump_path = str(
            checkpoint_root / "stage_1" / "fixed_training_batch.pt"
        )
        grad = cfg.train.telemetry.grad_layer_monitor
        grad.csv_path = str(checkpoint_root / "stage_1" / "grad_layer_rms.csv")
        grad.plot_path = str(checkpoint_root / "stage_1" / "grad_layer_rms.png")
        grad.heatmap_path = str(checkpoint_root / "stage_1" / "grad_layer_rms_heatmap.png")
        cfg.collector.diagnostics.crash_report_path = str(
            Path(run_artifacts["crashes_dir"]) / "collector_crash_report.json"
        )
        cfg.collector.diagnostics.worker_status_dir = str(
            Path(run_artifacts["reports_dir"]) / "dataset_workers"
        )
    return cfg


def _optimizer_param_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[list[str]]:
    name_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        [name_by_identity[id(parameter)] for parameter in group["params"]]
        for group in optimizer.param_groups
    ]


def _migrate_optimizer_state_by_name(
    *,
    source: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reset_prefixes: tuple[str, ...],
) -> tuple[dict[str, Any], list[list[str]], dict[str, int]]:
    old_state = source.get("optimizer_state")
    old_names = source.get("optimizer_param_names")
    if not isinstance(old_state, dict) or not isinstance(old_names, list):
        raise RuntimeError("source resume checkpoint lacks named optimizer state")
    old_groups = old_state.get("param_groups")
    if not isinstance(old_groups, list) or len(old_groups) != len(old_names):
        raise RuntimeError("source optimizer group/name inventory is malformed")

    old_by_name: dict[str, dict[str, Any]] = {}
    for group, names in zip(old_groups, old_names, strict=True):
        param_ids = group.get("params", [])
        if len(param_ids) != len(names):
            raise RuntimeError("source optimizer parameter/name lengths differ")
        for param_id, name in zip(param_ids, names, strict=True):
            state = old_state.get("state", {}).get(param_id)
            if state is not None:
                old_by_name[str(name)] = state

    new_names = _optimizer_param_names(model, optimizer)
    new_state = optimizer.state_dict()
    parameter_by_name = dict(model.named_parameters())
    copied = 0
    reset = 0
    fresh = 0
    for group, names in zip(new_state["param_groups"], new_names, strict=True):
        for param_id, name in zip(group["params"], names, strict=True):
            if name.startswith(reset_prefixes):
                reset += 1
                continue
            source_param_state = old_by_name.get(name)
            if source_param_state is None:
                fresh += 1
                continue
            parameter = parameter_by_name[name]
            migrated_param_state: dict[str, Any] = {}
            for state_name, value in source_param_state.items():
                if torch.is_tensor(value) and value.ndim > 0 and tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"optimizer tensor shape mismatch for {name}.{state_name}: "
                        f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                    )
                migrated_param_state[state_name] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            new_state["state"][param_id] = migrated_param_state
            copied += 1

    for group in new_state["param_groups"]:
        group["lr"] = 5e-5
        group["initial_lr"] = 5e-5
    return new_state, new_names, {"copied": copied, "reset": reset, "fresh": fresh}


def _migrate_resume_checkpoint(
    *,
    source_path: Path,
    output_path: Path,
    cfg: DictConfig,
) -> dict[str, Any]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if int(source.get("step", 0)) <= 0:
        raise RuntimeError(f"source resume checkpoint has invalid step: {source_path}")

    torch.manual_seed(20260821)
    model = build_weight_quantile_vae(_build_model_cfg(cfg))
    target_state = model.state_dict()
    source_state = source.get("model_state")
    if not isinstance(source_state, dict):
        raise RuntimeError("source resume checkpoint lacks model_state")

    copied_model = 0
    reset_model = 0
    fresh_model = 0
    for name, target_value in target_state.items():
        if name.startswith("scale_head."):
            reset_model += 1
            continue
        source_value = source_state.get(name)
        if source_value is None:
            fresh_model += 1
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"model tensor shape mismatch for {name}: {tuple(source_value.shape)} != {tuple(target_value.shape)}"
            )
        target_state[name] = source_value.clone()
        copied_model += 1
    model.load_state_dict(target_state, strict=True)

    optimizer = _build_optimizer(model=model, cfg=cfg, device=torch.device("cpu"))
    optimizer_state, optimizer_names, optimizer_counts = _migrate_optimizer_state_by_name(
        source=source,
        model=model,
        optimizer=optimizer,
        reset_prefixes=("scale_head.",),
    )
    migrated = dict(source)
    migrated["model_state"] = model.state_dict()
    migrated["optimizer_state"] = optimizer_state
    migrated["optimizer_param_names"] = optimizer_names
    migrated["config"] = OmegaConf.to_container(cfg, resolve=True)
    migrated["architecture_migration"] = {
        "kind": "legacy_v1_to_latent_feedback_postnorm_v2",
        "source_checkpoint": str(source_path),
        "source_step": int(source["step"]),
        "model_tensors_copied": copied_model,
        "model_tensors_reset": reset_model,
        "model_tensors_fresh": fresh_model,
        "optimizer_parameters": optimizer_counts,
        "reset_prefixes": ["scale_head."],
        "migration_seed": 20260821,
    }
    _save_checkpoint_payload(migrated, output_path)
    output_path.chmod(0o444)
    return dict(migrated["architecture_migration"])


def _run(cfg: DictConfig) -> None:
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
    profile_config = args.profile_config.expanduser().resolve()
    production_config = args.production_config.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    replay_step = int(args.replay_step)
    source_resume = args.source_resume.expanduser().resolve() if args.source_resume is not None else None
    if source_resume is None and replay_step < 1:
        raise ValueError("--replay-step must be positive unless --source-resume is supplied")
    base = _load_mapping(profile_config)
    production = _load_mapping(production_config)

    if source_resume is not None:
        source_payload = torch.load(source_resume, map_location="cpu", weights_only=False)
        replay_step = int(source_payload.get("step", 0))
        del source_payload
        if replay_step < 1:
            raise ValueError(f"invalid source resume step: {source_resume}")
        continuation_cfg = _prepare_cfg(
            base=base,
            production=production,
            artifact_root=artifact_root,
            checkpoint_root=checkpoint_root,
            run_id=f"weightclip_ae_704m_lat384_v2_resume_{replay_step}",
            replay_step=replay_step,
            resume_checkpoint=checkpoint_root / "stage_1" / "resume_state" / f"step_{replay_step:07d}.pt",
            architecture_version=args.architecture_version,
        )
        migrated_path = checkpoint_root / "stage_1" / "resume_state" / f"step_{replay_step:07d}.pt"
        if not migrated_path.exists():
            migration = _migrate_resume_checkpoint(
                source_path=source_resume,
                output_path=migrated_path,
                cfg=continuation_cfg,
            )
            print(f"[migration] created {migrated_path}: {migration}", flush=True)
        else:
            print(f"[migration] reusing existing migrated resume {migrated_path}", flush=True)
        _run(continuation_cfg)
        return

    print(
        f"[recovery] replaying steps 1..{replay_step} with cosine horizon=500000; "
        f"checkpoint_root={checkpoint_root}",
        flush=True,
    )
    replay_cfg = _prepare_cfg(
        base=base,
        production=production,
        artifact_root=artifact_root,
        checkpoint_root=checkpoint_root,
        run_id=f"weightclip_ae_704m_lat384_replay_to_{replay_step}",
        replay_step=replay_step,
        resume_checkpoint=None,
        architecture_version=args.architecture_version,
    )
    _run(replay_cfg)

    resume_checkpoint = checkpoint_root / "stage_1" / "resume_state" / f"step_{replay_step:07d}.pt"
    if not resume_checkpoint.is_file():
        raise RuntimeError(f"replay finished without the required resume checkpoint: {resume_checkpoint}")
    print(
        f"[recovery] checkpoint ready at step={replay_step}; resuming with constant lr=5e-5",
        flush=True,
    )
    continuation_cfg = _prepare_cfg(
        base=base,
        production=production,
        artifact_root=artifact_root,
        checkpoint_root=checkpoint_root,
        run_id="weightclip_ae_704m_lat384_constant_resume_500k",
        replay_step=replay_step,
        resume_checkpoint=resume_checkpoint,
        architecture_version=args.architecture_version,
    )
    _run(continuation_cfg)


if __name__ == "__main__":
    main()
