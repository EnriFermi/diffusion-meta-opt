from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from dataset.logging_utils import configure_process_logging
from dataset.big_vae_latent_diffusion_offline import (
    collate_big_vae_latent_diffusion_batch,
    offline_big_vae_latent_diffusion_data_pipeline,
)
from experiments.train_big_vae import (
    _find_latest_resume_state_checkpoint,
    _normalize_model_state_dict_keys,
    _resume_state_load_policy,
    _save_checkpoint_payload,
    _unwrap_model_for_state_io,
    compute_grad_stats,
)
from models.layer_latent_diffusion_prior import (
    LayerLatentDiffusionPrior,
    build_layer_latent_diffusion_prior_config,
    compute_layer_latent_diffusion_loss,
)
from training.optim import build_adamw_optimizer, build_cosine_scheduler
from training.big_vae_latent_diffusion import (
    build_cond_global_from_dist_var_pooled,
    build_layer_metadata_condition_vector,
    encode_big_vae_distribution_context_batch,
    infer_big_vae_cond_global_dim_from_checkpoint,
    latent_diffusion_layer_metadata_cond_dim,
    load_big_vae_with_trainable_distribution_encoder,
    load_frozen_big_vae_from_checkpoint,
)
from training.runtime import (
    autocast_context,
    configure_per_run_artifacts,
    create_grad_scaler,
    maybe_compile_model,
    resolve_amp,
    set_speed_optimizations,
)


LOGGER = logging.getLogger("train_big_vae_latent_diffusion_prior")


@dataclass(slots=True)
class _DistributionEncoderFinetuneConfig:
    enabled: bool = False
    lr: float = 0.0
    weight_decay: float = 0.0


class _JointPriorTrainingModule(nn.Module):
    def __init__(
        self,
        *,
        prior: torch.nn.Module,
        distribution_encoder: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.distribution_encoder = distribution_encoder


def _resolve_distribution_encoder_finetune_cfg(cfg: DictConfig) -> _DistributionEncoderFinetuneConfig:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, (dict, DictConfig)):
        raise TypeError("train must be a mapping")
    raw_cfg = train_cfg.get("distribution_encoder_finetune", {})
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, (dict, DictConfig)):
        raise TypeError("train.distribution_encoder_finetune must be a mapping")
    train_lr = float(train_cfg.get("lr", 1e-4))
    train_weight_decay = float(train_cfg.get("weight_decay", 0.01))
    return _DistributionEncoderFinetuneConfig(
        enabled=bool(raw_cfg.get("enabled", False)),
        lr=float(raw_cfg.get("lr", train_lr)),
        weight_decay=float(raw_cfg.get("weight_decay", train_weight_decay)),
    )


def _flatten_for_tracking(
    payload: Any,
    *,
    prefix: str = "",
    out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = {} if out is None else out
    if isinstance(payload, (dict, DictConfig)):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_for_tracking(value, prefix=next_prefix, out=target)
        return target
    if isinstance(payload, (list, tuple, ListConfig)):
        target[prefix] = list(payload)
        return target
    target[prefix] = payload
    return target


class CometTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        self.logger = logger
        self.experiment: Any | None = None
        self.enabled = False
        telemetry_cfg = cfg.train.get("telemetry", {})
        comet_cfg = telemetry_cfg.get("comet", {}) if isinstance(telemetry_cfg, (dict, DictConfig)) else {}
        if not bool(comet_cfg.get("enabled", False)):
            return
        try:
            from comet_ml import Experiment, OfflineExperiment  # type: ignore
        except Exception as exc:
            self.logger.warning("Comet is enabled but comet_ml is unavailable: %s", exc)
            return

        api_key = str(comet_cfg.get("api_key", "")).strip()
        workspace = str(comet_cfg.get("workspace", "")).strip()
        project_name = str(comet_cfg.get("project_name", "big_vae_latent_diffusion_prior")).strip() or (
            "big_vae_latent_diffusion_prior"
        )
        experiment_name = str(comet_cfg.get("experiment_name", "")).strip()
        offline_dir = str(comet_cfg.get("offline_directory", "")).strip()
        log_code = bool(comet_cfg.get("log_code", False))
        try:
            if api_key:
                exp = Experiment(
                    api_key=api_key,
                    project_name=project_name,
                    workspace=workspace or None,
                    auto_output_logging="simple",
                    log_code=log_code,
                )
            else:
                exp = OfflineExperiment(
                    project_name=project_name,
                    workspace=workspace or None,
                    auto_output_logging="simple",
                    log_code=log_code,
                    offline_directory=offline_dir or None,
                )
            if experiment_name:
                exp.set_name(experiment_name)
            tags = comet_cfg.get("tags", [])
            if isinstance(tags, (list, tuple, ListConfig)):
                for tag in tags:
                    exp.add_tag(str(tag))
            exp.log_parameters(_flatten_for_tracking(OmegaConf.to_container(cfg, resolve=True)))
            self.experiment = exp
            self.enabled = True
        except Exception as exc:
            self.logger.warning("Failed to initialize Comet tracker: %s", exc)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_metrics(metrics, step=int(step))
        except Exception as exc:
            self.logger.warning("Comet metric log failed at step=%s: %s", step, exc)

    def end(self) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.end()
        except Exception:
            pass


class WandbTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        self.logger = logger
        self.run: Any | None = None
        self.enabled = False
        telemetry_cfg = cfg.train.get("telemetry", {})
        wandb_cfg = telemetry_cfg.get("wandb", {}) if isinstance(telemetry_cfg, (dict, DictConfig)) else {}
        if not bool(wandb_cfg.get("enabled", False)):
            return
        try:
            import wandb  # type: ignore
        except Exception as exc:
            self.logger.warning("W&B is enabled but wandb is unavailable: %s", exc)
            return

        project_name = str(wandb_cfg.get("project_name", "big_vae_latent_diffusion_prior")).strip() or (
            "big_vae_latent_diffusion_prior"
        )
        entity = str(wandb_cfg.get("entity", "")).strip()
        run_name = str(wandb_cfg.get("run_name", "")).strip()
        run_mode = str(wandb_cfg.get("mode", "offline")).strip().lower() or "offline"
        run_dir = str(wandb_cfg.get("dir", "")).strip()
        tags = wandb_cfg.get("tags", [])
        tags_list = [str(tag) for tag in tags] if isinstance(tags, (list, tuple, ListConfig)) else None
        try:
            self.run = wandb.init(
                project=project_name,
                entity=entity or None,
                name=run_name or None,
                mode=run_mode,
                dir=run_dir or None,
                config=_flatten_for_tracking(OmegaConf.to_container(cfg, resolve=True)),
                tags=tags_list,
            )
            if self.run is not None:
                self.enabled = True
        except Exception as exc:
            self.logger.warning("Failed to initialize W&B tracker: %s", exc)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=int(step))
        except Exception as exc:
            self.logger.warning("W&B metric log failed at step=%s: %s", step, exc)

    def end(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass


def _promote_run_profile_to_root(cfg: DictConfig) -> None:
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return
    expected_sections = ("train", "model", "training_artifacts", "logging", "hf")
    with open_dict(cfg):
        for section in expected_sections:
            if section not in cfg and section in run_profiles_cfg:
                cfg[section] = run_profiles_cfg[section]


def _resolve_prior_cfg(
    cfg: DictConfig,
    *,
    dataset_summary: dict[str, Any],
    dataset_stats: dict[str, Any] | None = None,
    live_cond_global_dim: int | None = None,
) -> DictConfig:
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, (dict, DictConfig)):
        raise TypeError("model must be a mapping")
    prior_cfg = model_cfg.get("latent_diffusion_prior", {})
    if not isinstance(prior_cfg, (dict, DictConfig)):
        raise TypeError("model.latent_diffusion_prior must be a mapping")
    stats_payload = dict(dataset_stats or {})
    with open_dict(cfg):
        if "latent_diffusion_prior" not in cfg.model:
            cfg.model.latent_diffusion_prior = {}
        if int(prior_cfg.get("z_dim", 0)) <= 0:
            cfg.model.latent_diffusion_prior.z_dim = max(
                int(dataset_summary.get("z_dim", 0)),
                int(stats_payload.get("z_dim", 0)),
            )
        if int(prior_cfg.get("cond_dim", 0)) <= 0:
            cfg.model.latent_diffusion_prior.cond_dim = int(dataset_summary.get("cond_dim", 0))
        metadata_cond_dim = latent_diffusion_layer_metadata_cond_dim(
            use_layer_type_conditioning=bool(prior_cfg.get("use_layer_type_conditioning", False)),
            use_layer_depth_conditioning=bool(prior_cfg.get("use_layer_depth_conditioning", False)),
            depth_fourier_dim=int(prior_cfg.get("layer_depth_fourier_dim", 16)),
        )
        if int(prior_cfg.get("cond_global_dim", 0)) <= 0:
            resolved_cond_global_dim = max(
                int(dataset_summary.get("cond_global_dim", 0)),
                int(stats_payload.get("cond_global_dim", 0)),
                max(0, int(live_cond_global_dim or 0)),
            )
            cfg.model.latent_diffusion_prior.cond_global_dim = int(resolved_cond_global_dim) + int(metadata_cond_dim)
    return cfg.model.latent_diffusion_prior


def _concat_optional_condition_vectors(
    lhs: torch.Tensor | None,
    rhs: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if lhs is None and rhs is None:
        return None
    pieces: list[torch.Tensor] = []
    for item in (lhs, rhs):
        if item is None:
            continue
        if item.ndim != 2 or int(item.shape[0]) != int(batch_size):
            raise ValueError(f"Condition vector must be [B,D] with batch={batch_size}, got {tuple(item.shape)}")
        pieces.append(item.to(device=device, dtype=dtype))
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=-1)


def _resolve_checkpoint_layout(cfg: DictConfig) -> int:
    train_cfg = cfg.train
    stage = max(1, int(train_cfg.get("stage", 1)))
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "./artifacts/training/checkpoints/big_vae_latent_diffusion_prior")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return stage


def _dummy_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


def _build_prior_training_optimizer(
    *,
    prior_model: torch.nn.Module,
    distribution_encoder_model: torch.nn.Module | None,
    distribution_encoder_cfg: _DistributionEncoderFinetuneConfig,
    cfg: DictConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    if distribution_encoder_model is None:
        return build_adamw_optimizer(
            model=prior_model,
            cfg=cfg,
            device=device,
            section="train",
            default_lr=3e-4,
            default_weight_decay=0.01,
        )

    train_cfg = cfg["train"]
    optimizer_name = str(train_cfg.get("optimizer_name", "adamw")).strip().lower()
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    betas_cfg = train_cfg.get("betas", [0.9, 0.95])
    beta1 = float(betas_cfg[0])
    beta2 = float(betas_cfg[1])
    eps = float(train_cfg.get("eps", 1e-8))

    prior_params = [param for param in prior_model.parameters() if param.requires_grad]
    dist_params = [param for param in distribution_encoder_model.parameters() if param.requires_grad]
    param_groups: list[dict[str, Any]] = []
    if prior_params:
        param_groups.append(
            {
                "params": prior_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "group_name": "prior",
            }
        )
    if dist_params:
        param_groups.append(
            {
                "params": dist_params,
                "lr": float(distribution_encoder_cfg.lr),
                "weight_decay": float(distribution_encoder_cfg.weight_decay),
                "group_name": "distribution_encoder",
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters were found for latent diffusion prior optimizer construction")

    kwargs: dict[str, Any] = {
        "betas": (beta1, beta2),
        "eps": eps,
    }
    if optimizer_name == "adamw":
        optimizer_cls = torch.optim.AdamW
    elif optimizer_name == "adam":
        optimizer_cls = torch.optim.Adam
    else:
        raise ValueError(
            f"Unsupported train.optimizer_name={optimizer_name!r}. Expected one of: 'adamw', 'adam'"
        )

    params = inspect.signature(optimizer_cls).parameters
    use_fused = "fused" in params and device.type == "cuda"
    use_foreach = "foreach" in params and not use_fused
    if "fused" in params:
        kwargs["fused"] = use_fused
    if "foreach" in params:
        kwargs["foreach"] = use_foreach
    return optimizer_cls(param_groups, **kwargs)


def _save_latent_diffusion_prior_checkpoint(
    *,
    model: torch.nn.Module,
    distribution_encoder_model: torch.nn.Module | None,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
    stage: int = 1,
) -> None:
    base_dir = Path(str(cfg.train.get("checkpoint_dir", "./checkpoints/big_vae_latent_diffusion_prior")))
    checkpoint_dir = base_dir / f"stage_{stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = _unwrap_model_for_state_io(model)
    payload: dict[str, Any] = {
        "step": step_idx,
        "stage": stage,
        "model_state": model_to_save.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if distribution_encoder_model is not None:
        payload["distribution_encoder_state"] = distribution_encoder_model.state_dict()

    step_path = checkpoint_dir / f"step_{step_idx:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    _save_checkpoint_payload(payload, step_path)
    _save_checkpoint_payload(payload, latest_path)
    logger.info("Model checkpoint saved: %s", step_path)


def _save_latent_diffusion_prior_resume_state_checkpoint(
    *,
    model: torch.nn.Module,
    distribution_encoder_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
    stage: int,
    state_dir: Path,
) -> None:
    model_to_save = _unwrap_model_for_state_io(model)
    payload: dict[str, Any] = {
        "step": step_idx,
        "stage": stage,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if distribution_encoder_model is not None:
        payload["distribution_encoder_state"] = distribution_encoder_model.state_dict()

    state_path = state_dir / f"step_{step_idx:07d}.pt"
    _save_checkpoint_payload(payload, state_path)

    for stale_path in sorted(state_dir.glob("step_*.pt")):
        if stale_path == state_path:
            continue
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not delete stale resume-state checkpoint %s: %s", stale_path, exc)
    logger.info("Resume-state checkpoint saved: %s", state_path)


def _load_latent_diffusion_prior_training_state_from_checkpoint(
    *,
    model: torch.nn.Module,
    distribution_encoder_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    path: Path,
    logger: logging.Logger,
    load_model_state: bool = True,
    load_optimizer_state: bool = True,
    load_scheduler_state: bool = True,
    load_scaler_state: bool = True,
    load_step: bool = True,
) -> int:
    logger.info("Loading training state from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"latent diffusion prior checkpoint must contain a dict payload, got {type(ckpt)!r}")

    if load_model_state:
        state_dict = ckpt.get("model_state")
        if not isinstance(state_dict, dict):
            raise KeyError(f"checkpoint has no dict model_state: {path}")
        missing, unexpected = _unwrap_model_for_state_io(model).load_state_dict(
            _normalize_model_state_dict_keys(state_dict),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Latent diffusion prior checkpoint model_state mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )

        checkpoint_dist_state = ckpt.get("distribution_encoder_state")
        if distribution_encoder_model is not None:
            if checkpoint_dist_state is None:
                if load_optimizer_state and ckpt.get("optimizer_state") is not None:
                    raise RuntimeError(
                        "Checkpoint has optimizer_state but no distribution_encoder_state, so it cannot resume "
                        "with train.distribution_encoder_finetune.enabled=true. Disable optimizer-state resume or "
                        "start from a checkpoint created with distribution encoder fine-tuning enabled."
                    )
                logger.warning(
                    "Checkpoint %s has no distribution_encoder_state; keeping current BigVAE distribution encoder init",
                    path,
                )
            else:
                if not isinstance(checkpoint_dist_state, dict):
                    raise TypeError(
                        f"checkpoint field distribution_encoder_state must be a dict, got {type(checkpoint_dist_state)!r}"
                    )
                missing, unexpected = distribution_encoder_model.load_state_dict(
                    _normalize_model_state_dict_keys(checkpoint_dist_state),
                    strict=False,
                )
                if missing or unexpected:
                    raise RuntimeError(
                        "Latent diffusion prior checkpoint distribution_encoder_state mismatch: "
                        f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
                    )
        elif checkpoint_dist_state is not None and load_optimizer_state and ckpt.get("optimizer_state") is not None:
            raise RuntimeError(
                "Checkpoint contains distribution_encoder_state but current config has "
                "train.distribution_encoder_finetune.enabled=false. Disable optimizer-state resume or use a matching config."
            )

    optimizer_state = ckpt.get("optimizer_state")
    if load_optimizer_state and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = ckpt.get("scheduler_state")
    if load_scheduler_state and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = ckpt.get("scaler_state")
    if load_scaler_state and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    step = int(ckpt.get("step", 0) or 0) if load_step else 0
    stage = ckpt.get("stage", "?")
    logger.info(
        "Training state loaded successfully (path=%s step=%s stage=%s load_model_state=%s "
        "load_optimizer_state=%s load_scheduler_state=%s load_scaler_state=%s load_step=%s)",
        path,
        step,
        stage,
        load_model_state,
        load_optimizer_state,
        load_scheduler_state,
        load_scaler_state,
        load_step,
    )
    return step


def _get_optimizer_group_lr(
    optimizer: torch.optim.Optimizer,
    group_name: str,
    default: float = 0.0,
) -> float:
    for group in optimizer.param_groups:
        if str(group.get("group_name", "")) == str(group_name):
            return float(group.get("lr", default))
    return float(default)


def _next_batch(iterator: Iterator[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    items = [next(iterator) for _ in range(max(1, int(batch_size)))]
    return collate_big_vae_latent_diffusion_batch(items)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "latent_mu",
        "latent_logvar",
        "cond_patch",
        "cond_global",
        "layer_type_ids",
        "layer_depths",
        "patch_mask",
        "d_in",
        "d_out",
        "X",
        "W",
        "x_mask",
        "d_in_mask",
        "d_out_mask",
    ):
        value = moved.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


@hydra.main(version_base=None, config_path="../conf", config_name="config_big_vae_latent_diffusion_prior")
def main(cfg: DictConfig) -> None:
    _promote_run_profile_to_root(cfg)
    artifacts = configure_per_run_artifacts(cfg, run_label="train_big_vae_latent_diffusion_prior")
    with open_dict(cfg):
        cfg.train.checkpoint_dir = str(Path(artifacts["root_dir"]) / "checkpoints" / "big_vae_latent_diffusion_prior")

    log_path = configure_process_logging(cfg=cfg, role="train_big_vae_latent_diffusion_prior", rank=0, force=True)
    logger = LOGGER
    logger.info("Starting latent diffusion prior training")
    logger.info("Run log file: %s", log_path)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device(str(cfg.train.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    set_speed_optimizations(cfg, device, section="train")
    amp_enabled, amp_dtype = resolve_amp(cfg, device, section="train")
    stage = _resolve_checkpoint_layout(cfg)

    comet_tracker: CometTracker | None = None
    wandb_tracker: WandbTracker | None = None
    with offline_big_vae_latent_diffusion_data_pipeline(cfg, logger=logger) as dataset:
        dataset_summary = dataset.summary()
        dataset_stats = dataset.latent_stats()
        distribution_encoder_finetune_cfg = _resolve_distribution_encoder_finetune_cfg(cfg)
        conditioning_checkpoint = str(cfg.get("latent_diffusion_prior", {}).get("big_vae_checkpoint", "")).strip()
        live_cond_global_dim = 0
        if distribution_encoder_finetune_cfg.enabled:
            if not conditioning_checkpoint:
                raise ValueError(
                    "latent_diffusion_prior.big_vae_checkpoint must be set when "
                    "train.distribution_encoder_finetune.enabled=true"
                )
            live_cond_global_dim = infer_big_vae_cond_global_dim_from_checkpoint(conditioning_checkpoint)
        prior_cfg_raw = _resolve_prior_cfg(
            cfg,
            dataset_summary=dataset_summary,
            dataset_stats=dataset_stats,
            live_cond_global_dim=live_cond_global_dim,
        )
        prior_cfg = build_layer_latent_diffusion_prior_config(prior_cfg_raw)

        logger.info(
            "Latent dataset summary: root=%s accepted_records=%s z_dim=%s cond_dim=%s cond_global_dim=%s has_decoder_aux_tensors=%s",
            dataset_summary.get("root_dir", ""),
            int(dataset_summary.get("accepted_records", 0)),
            int(dataset_summary.get("z_dim", 0)),
            int(dataset_summary.get("cond_dim", 0)),
            int(dataset_summary.get("cond_global_dim", 0)),
            bool(dataset_summary.get("has_decoder_aux_tensors", False)),
        )
        logger.info(
            "Model config: z_dim=%s num_latent_tokens=%s cond_dim=%s cond_global_dim=%s "
            "layer_type_conditioning=%s layer_depth_conditioning=%s d_model=%s n_layers=%s n_heads=%s prediction_type=%s",
            prior_cfg.z_dim,
            prior_cfg.num_latent_tokens,
            prior_cfg.cond_dim,
            prior_cfg.cond_global_dim,
            bool(prior_cfg.use_layer_type_conditioning),
            bool(prior_cfg.use_layer_depth_conditioning),
            prior_cfg.d_model,
            prior_cfg.n_layers,
            prior_cfg.n_heads,
            prior_cfg.prediction_type,
        )

        model = LayerLatentDiffusionPrior(prior_cfg).to(device)
        model.set_latent_normalization_stats(
            latent_mean=dataset_stats["latent_mean"],
            latent_std=dataset_stats["latent_std"],
        )
        model = maybe_compile_model(model, cfg, logger, section="train", label="latent_diffusion_prior")
        conditioning_big_vae = None
        trainable_distribution_encoder = None
        if distribution_encoder_finetune_cfg.enabled:
            if not bool(dataset_summary.get("has_decoder_aux_tensors", False)):
                raise ValueError(
                    "Distribution encoder fine-tuning requires offline latent diffusion dataset with stored X/W/masks. "
                    "Rebuild the dataset with latent_diffusion_dataset.builder.store_decoder_aux_tensors=true."
                )
            conditioning_big_vae = load_big_vae_with_trainable_distribution_encoder(conditioning_checkpoint, device=device)
            trainable_distribution_encoder = conditioning_big_vae.distribution_encoder
            logger.info(
                "Loaded BigVAE distribution encoder for joint conditioning fine-tuning: checkpoint=%s "
                "live_cond_global_dim=%s lr=%.3e weight_decay=%.3e",
                conditioning_checkpoint,
                int(live_cond_global_dim),
                float(distribution_encoder_finetune_cfg.lr),
                float(distribution_encoder_finetune_cfg.weight_decay),
            )
        joint_train_module = _JointPriorTrainingModule(
            prior=model,
            distribution_encoder=trainable_distribution_encoder,
        )
        decoder_aux_model = None
        if bool(prior_cfg.use_decoder_aux):
            if not bool(dataset_summary.get("has_decoder_aux_tensors", False)):
                raise ValueError(
                    "decoder auxiliary loss requires offline latent diffusion dataset with stored reconstruction targets. "
                    "Rebuild the dataset with latent_diffusion_dataset.builder.store_decoder_aux_tensors=true."
                )
            aux_checkpoint = str(cfg.get("latent_diffusion_prior", {}).get("big_vae_checkpoint", "")).strip()
            if not aux_checkpoint:
                raise ValueError(
                    "latent_diffusion_prior.big_vae_checkpoint must be set when model.latent_diffusion_prior.use_decoder_aux=true"
                )
            decoder_aux_model = load_frozen_big_vae_from_checkpoint(aux_checkpoint, device=device)
            logger.info("Loaded frozen BigVAE decoder for auxiliary loss: %s", aux_checkpoint)

        optimizer = _build_prior_training_optimizer(
            prior_model=model,
            distribution_encoder_model=trainable_distribution_encoder,
            distribution_encoder_cfg=distribution_encoder_finetune_cfg,
            cfg=cfg,
            device=device,
        )
        scheduler = build_cosine_scheduler(
            optimizer=optimizer,
            cfg=cfg,
            section="train",
            default_max_steps=1000,
            default_warmup_steps=100,
            default_min_lr_ratio=0.1,
        )
        scheduler_is_dummy = scheduler is None
        if scheduler is None:
            scheduler = _dummy_scheduler(optimizer)
        scaler = create_grad_scaler(device=device, enabled=amp_enabled)

        comet_tracker = CometTracker(cfg, logger)
        wandb_tracker = WandbTracker(cfg, logger)

        checkpoint_every = max(1, int(cfg.train.get("checkpoint_every", 1000)))
        log_every = max(1, int(cfg.train.get("log_every", 50)))
        grad_accum_steps = max(1, int(cfg.train.get("grad_accum_steps", 1)))
        batch_size = max(1, int(cfg.train.get("batch_size", 8)))
        max_steps = max(1, int(cfg.train.get("max_steps", 1000)))
        grad_clip_norm = float(cfg.train.get("grad_clip_norm", 0.0))
        resume_state_cfg = cfg.train.get("resume_state", {})
        if resume_state_cfg is None:
            resume_state_cfg = {}
        if not isinstance(resume_state_cfg, (dict, DictConfig)):
            raise TypeError("train.resume_state must be a mapping")
        resume_state_enabled = bool(resume_state_cfg.get("enabled", True))
        resume_state_auto_resume = bool(resume_state_cfg.get("auto_resume", True))
        resume_state_save_every = max(1, int(resume_state_cfg.get("save_every", checkpoint_every)))
        resume_state_load_policy = _resume_state_load_policy(resume_state_cfg)
        default_resume_state_dir = Path(str(cfg.train.get("checkpoint_dir"))) / f"stage_{stage}" / "resume_state"
        resume_state_dir = Path(str(resume_state_cfg.get("dir", default_resume_state_dir)))
        resume_state_dir.mkdir(parents=True, exist_ok=True)

        resumed_step = 0
        if resume_state_enabled and resume_state_auto_resume:
            candidate = _find_latest_resume_state_checkpoint(resume_state_dir)
            if candidate is not None:
                logger.info("Auto-resume candidate found: %s", candidate)
                resumed_step = _load_latent_diffusion_prior_training_state_from_checkpoint(
                    model=model,
                    distribution_encoder_model=trainable_distribution_encoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    path=candidate,
                    logger=logger,
                    load_model_state=resume_state_load_policy["load_model_state"],
                    load_optimizer_state=resume_state_load_policy["load_optimizer_state"],
                    load_scheduler_state=resume_state_load_policy["load_scheduler_state"],
                    load_scaler_state=resume_state_load_policy["load_scaler_state"],
                    load_step=resume_state_load_policy["load_step"],
                )

        iterator = iter(dataset)
        optimizer.zero_grad(set_to_none=True)

        for global_step in range(resumed_step + 1, max_steps + 1):
            step_start = time.perf_counter()
            step_loss_sum = 0.0
            step_patch_tokens = 0.0
            step_patch_tokens_max = 0.0
            step_alpha_sum = 0.0
            step_sigma_sum = 0.0
            step_clean_std = 0.0
            step_pred_std = 0.0
            step_target_std = 0.0
            step_diffusion_loss = 0.0
            step_decoder_aux_loss = 0.0
            step_decoder_aux_behavioral = 0.0
            step_decoder_aux_behavioral_operator = 0.0
            step_decoder_aux_behavioral_dir = 0.0
            step_decoder_aux_behavioral_scale = 0.0
            step_decoder_aux_structural = 0.0
            step_decoder_aux_applied_fraction = 0.0
            step_decoder_aux_weight_mean = 0.0
            step_live_cond_alignment = 0.0

            for micro_idx in range(grad_accum_steps):
                batch = _move_batch_to_device(_next_batch(iterator, batch_size=batch_size), device=device)
                timesteps = model.schedule.sample_timesteps(batch["latent_mu"].shape[0], device=device)
                alpha_t, sigma_t = model.schedule.alpha_sigma(timesteps, x_ndim=2)
                step_alpha_sum += float(alpha_t.mean().detach().item())
                step_sigma_sum += float(sigma_t.mean().detach().item())

                with autocast_context(amp_enabled, amp_dtype):
                    cond_patch = batch["cond_patch"]
                    cond_global = batch.get("cond_global")
                    patch_mask = batch["patch_mask"]
                    offline_cond_patch_alignment_mse = None
                    if conditioning_big_vae is not None:
                        if batch.get("X") is None or batch.get("x_mask") is None or batch.get("d_in_mask") is None:
                            raise ValueError(
                                "Distribution encoder fine-tuning requires batch tensors X, x_mask, and d_in_mask"
                            )
                        cond_payload = encode_big_vae_distribution_context_batch(
                            conditioning_big_vae,
                            X=batch["X"],
                            x_mask=batch.get("x_mask"),
                            d_in_mask=batch.get("d_in_mask"),
                        )
                        cond_patch = cond_payload["cond_patch"]
                        patch_mask = cond_payload["patch_mask"]
                        if not torch.is_tensor(cond_patch) or not torch.is_tensor(patch_mask):
                            raise RuntimeError("Distribution encoder fine-tuning expected tensor cond_patch and patch_mask")
                        cond_global = build_cond_global_from_dist_var_pooled(
                            dist_var_pooled=cond_payload.get("dist_var_pooled"),
                            patch_mask=patch_mask,
                        )
                        valid_mask = patch_mask.unsqueeze(-1).to(dtype=cond_patch.dtype)
                        denom = valid_mask.sum().clamp_min(1.0)
                        offline_cond_patch_alignment_mse = (
                            ((cond_patch - batch["cond_patch"]) * valid_mask).pow(2).sum() / denom
                        )
                        step_patch_tokens += float(patch_mask.to(dtype=torch.long).sum(dim=1).float().mean().item())
                        step_patch_tokens_max = max(
                            step_patch_tokens_max,
                            float(patch_mask.to(dtype=torch.long).sum(dim=1).max().item()),
                        )
                    else:
                        patch_lengths = patch_mask.to(dtype=torch.long).sum(dim=1)
                        step_patch_tokens += float(patch_lengths.float().mean().item())
                        step_patch_tokens_max = max(step_patch_tokens_max, float(patch_lengths.max().item()))
                    metadata_cond = build_layer_metadata_condition_vector(
                        device=cond_patch.device,
                        dtype=cond_patch.dtype,
                        use_layer_type_conditioning=bool(prior_cfg.use_layer_type_conditioning),
                        use_layer_depth_conditioning=bool(prior_cfg.use_layer_depth_conditioning),
                        depth_fourier_dim=int(prior_cfg.layer_depth_fourier_dim),
                        depth_scale=float(prior_cfg.layer_depth_scale),
                        layer_type_ids=batch.get("layer_type_ids"),
                        layer_depths=batch.get("layer_depths"),
                    )
                    cond_global = _concat_optional_condition_vectors(
                        cond_global,
                        metadata_cond,
                        batch_size=int(cond_patch.shape[0]),
                        device=cond_patch.device,
                        dtype=cond_patch.dtype,
                    )
                    loss_payload = compute_layer_latent_diffusion_loss(
                        model,
                        clean_latents=batch["latent_mu"],
                        cond_patch=cond_patch,
                        timesteps=timesteps,
                        patch_mask=patch_mask,
                        cond_global=cond_global,
                        decoder_aux_model=decoder_aux_model,
                        decoder_aux_W=batch.get("W"),
                        decoder_aux_X=batch.get("X"),
                        decoder_aux_x_mask=batch.get("x_mask"),
                        decoder_aux_d_in_mask=batch.get("d_in_mask"),
                        decoder_aux_d_out_mask=batch.get("d_out_mask"),
                    )
                    loss = loss_payload["loss"]
                    step_diffusion_loss += float(loss_payload["diffusion_loss"].detach().item())
                    step_decoder_aux_loss += float(loss_payload["decoder_aux_loss"].detach().item())
                    step_decoder_aux_behavioral += float(loss_payload["decoder_aux_behavioral_loss"].detach().item())
                    step_decoder_aux_behavioral_operator += float(
                        loss_payload["decoder_aux_behavioral_operator_loss"].detach().item()
                    )
                    step_decoder_aux_behavioral_dir += float(
                        loss_payload["decoder_aux_behavioral_dir_loss"].detach().item()
                    )
                    step_decoder_aux_behavioral_scale += float(
                        loss_payload["decoder_aux_behavioral_scale_loss"].detach().item()
                    )
                    step_decoder_aux_structural += float(loss_payload["decoder_aux_structural_loss"].detach().item())
                    step_decoder_aux_applied_fraction += float(loss_payload["decoder_aux_applied_fraction"].detach().item())
                    step_decoder_aux_weight_mean += float(loss_payload["decoder_aux_weight_mean"].detach().item())
                    step_clean_std += float(loss_payload["clean_tokens"].detach().float().std(unbiased=False).item())
                    step_pred_std += float(loss_payload["pred_tokens"].detach().float().std(unbiased=False).item())
                    step_target_std += float(loss_payload["target_tokens"].detach().float().std(unbiased=False).item())
                    if offline_cond_patch_alignment_mse is not None:
                        step_offline_cond_alignment = float(offline_cond_patch_alignment_mse.detach().item())
                    else:
                        step_offline_cond_alignment = 0.0
                step_loss_sum += float(loss.detach().item())
                scaler.scale(loss / float(grad_accum_steps)).backward()
                step_live_cond_alignment += float(step_offline_cond_alignment)

            scaler.unscale_(optimizer)
            grad_stats = compute_grad_stats(joint_train_module)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(joint_train_module.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not scheduler_is_dummy:
                scheduler.step()

            lr = _get_optimizer_group_lr(optimizer, "prior", float(optimizer.param_groups[0]["lr"]))
            lr_distribution_encoder = _get_optimizer_group_lr(optimizer, "distribution_encoder", 0.0)
            step_time = time.perf_counter() - step_start
            metrics = {
                "loss/total": step_loss_sum / float(grad_accum_steps),
                "loss/diffusion": step_diffusion_loss / float(grad_accum_steps),
                "loss/decoder_aux": step_decoder_aux_loss / float(grad_accum_steps),
                "loss/decoder_aux_behavioral": step_decoder_aux_behavioral / float(grad_accum_steps),
                "loss/decoder_aux_behavioral_operator": step_decoder_aux_behavioral_operator / float(grad_accum_steps),
                "loss/decoder_aux_behavioral_dir": step_decoder_aux_behavioral_dir / float(grad_accum_steps),
                "loss/decoder_aux_behavioral_scale": step_decoder_aux_behavioral_scale / float(grad_accum_steps),
                "loss/decoder_aux_structural": step_decoder_aux_structural / float(grad_accum_steps),
                "schedule/alpha_mean": step_alpha_sum / float(grad_accum_steps),
                "schedule/sigma_mean": step_sigma_sum / float(grad_accum_steps),
                "data/patch_tokens_mean": step_patch_tokens / float(grad_accum_steps),
                "data/patch_tokens_max": step_patch_tokens_max,
                "data/decoder_aux_applied_fraction": step_decoder_aux_applied_fraction / float(grad_accum_steps),
                "data/decoder_aux_weight_mean": step_decoder_aux_weight_mean / float(grad_accum_steps),
                "conditioning/live_enabled": 1.0 if conditioning_big_vae is not None else 0.0,
                "conditioning/layer_type_enabled": 1.0 if prior_cfg.use_layer_type_conditioning else 0.0,
                "conditioning/layer_depth_enabled": 1.0 if prior_cfg.use_layer_depth_conditioning else 0.0,
                "conditioning/offline_alignment_mse": step_live_cond_alignment / float(grad_accum_steps),
                "latent/clean_std": step_clean_std / float(grad_accum_steps),
                "latent/pred_std": step_pred_std / float(grad_accum_steps),
                "latent/target_std": step_target_std / float(grad_accum_steps),
                "optim/lr": lr,
                "optim/lr_prior": lr,
                "optim/lr_distribution_encoder": lr_distribution_encoder,
                "time/step_s": step_time,
                "grad/global_norm": float(grad_stats.get("grad/global_norm", 0.0)),
                "grad/rms": float(grad_stats.get("grad/rms", 0.0)),
                "grad/param_rms": float(grad_stats.get("param/rms", 0.0)),
                "grad/grad_to_param_rms_ratio": float(grad_stats.get("grad_to_param_rms_ratio", 0.0)),
            }

            if device.type == "cuda":
                metrics["gpu/memory_allocated_mb"] = float(torch.cuda.memory_allocated(device) / (1024.0 ** 2))
                metrics["gpu/memory_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024.0 ** 2))
                metrics["gpu/max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

            if global_step == 1 or global_step % log_every == 0:
                logger.info(
                    "step=%s loss=%.6f lr=%.3e alpha=%.4f sigma=%.4f patch_tokens=%.2f grad_norm=%.3e step_s=%.3f",
                    global_step,
                    metrics["loss/total"],
                    metrics["optim/lr"],
                    metrics["schedule/alpha_mean"],
                    metrics["schedule/sigma_mean"],
                    metrics["data/patch_tokens_mean"],
                    metrics["grad/global_norm"],
                    metrics["time/step_s"],
                )

            if comet_tracker is not None and comet_tracker.enabled:
                comet_tracker.log_metrics(metrics, step=global_step)
            if wandb_tracker is not None and wandb_tracker.enabled:
                wandb_tracker.log_metrics(metrics, step=global_step)

            if global_step % checkpoint_every == 0 or global_step == max_steps:
                _save_latent_diffusion_prior_checkpoint(
                    model=model,
                    distribution_encoder_model=trainable_distribution_encoder,
                    cfg=cfg,
                    step_idx=global_step,
                    logger=logger,
                    stage=stage,
                )
            if resume_state_enabled and (global_step % resume_state_save_every == 0 or global_step == max_steps):
                _save_latent_diffusion_prior_resume_state_checkpoint(
                    model=model,
                    distribution_encoder_model=trainable_distribution_encoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cfg=cfg,
                    step_idx=global_step,
                    logger=logger,
                    stage=stage,
                    state_dir=resume_state_dir,
                )

    if comet_tracker is not None:
        comet_tracker.end()
    if wandb_tracker is not None:
        wandb_tracker.end()


if __name__ == "__main__":
    main()
