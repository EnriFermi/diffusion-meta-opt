from __future__ import annotations

import inspect
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from big_vae.eval.vit_tiny_latent_optimization import (
    BigVAELatentTensorStore,
    EvalMetrics,
    autocast_context,
    count_trainable_parameters,
)
from post_train_research.vit_latent_scaling.config import RunConfig
from post_train_research.vit_latent_scaling.data import build_loaders
from post_train_research.vit_latent_scaling.init import (
    build_model,
    collect_calibration_images,
    export_named_tensors,
    initialize_diffusion_prior,
    load_big_vae_components,
    maybe_restore_direct_latent_state,
    prepare_initial_tensors,
    resolve_device,
    seed_everything,
)
from post_train_research.vit_latent_scaling.runtime import (
    CometTracker,
    RunPaths,
    append_metrics_row,
    update_run_index,
    write_summary,
)


@dataclass(slots=True)
class RunResult:
    summary: dict[str, Any]


def _sorted_latent_slot_keys(store: BigVAELatentTensorStore) -> list[str]:
    return sorted(str(key) for key in store.latent_slots.keys())


@torch.no_grad()
def flatten_latent_slots(store: BigVAELatentTensorStore) -> torch.Tensor:
    keys = _sorted_latent_slot_keys(store)
    if not keys:
        return torch.zeros(0, device=store.big_vae.latent_base.device, dtype=store.big_vae.latent_base.dtype)
    return torch.cat([store.materialize_latent_slot(key).detach().reshape(-1) for key in keys], dim=0)


@torch.no_grad()
def flatten_decoded_bigvae_weights(store: BigVAELatentTensorStore) -> torch.Tensor:
    decoded = store.decode_all_matrices()
    if not decoded:
        return torch.zeros(0, device=store.big_vae.latent_base.device, dtype=store.big_vae.latent_base.dtype)
    return torch.cat([decoded[name].detach().reshape(-1) for name in sorted(decoded.keys())], dim=0)


@torch.no_grad()
def estimate_decoder_effective_jacobian_norm(
    store: BigVAELatentTensorStore,
    *,
    eps: float,
    num_probes: int,
) -> dict[str, float]:
    keys = _sorted_latent_slot_keys(store)
    if not keys:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    z0 = flatten_latent_slots(store)
    w0 = flatten_decoded_bigvae_weights(store)
    tiny = torch.finfo(z0.dtype).tiny
    materialized = {key: store.materialize_latent_slot(key).detach().clone() for key in keys}
    values: list[float] = []
    for _ in range(int(num_probes)):
        if str(store.latent_parameterization).strip().lower() == "sphere":
            perturbed: dict[str, torch.Tensor] = {}
            for key in keys:
                z_key = materialized[key]
                radius = z_key.norm().clamp_min(tiny)
                z_unit = z_key / radius
                v_key = torch.randn_like(z_key)
                tangent = v_key - torch.sum(v_key * z_unit) * z_unit
                tangent = tangent / tangent.norm().clamp_min(tiny)
                z1_key = z_key + float(eps) * tangent
                z1_key = z1_key * (radius / z1_key.norm().clamp_min(tiny))
                perturbed[key] = z1_key
            store.load_materialized_latent_slots_state_dict(perturbed, strict=False, update_radii=False)
        else:
            v = torch.randn_like(z0)
            v = v / v.norm().clamp_min(tiny)
            _load_flat_latents(store, z0 + float(eps) * v)
        w1 = flatten_decoded_bigvae_weights(store)
        values.append(float((w1 - w0).norm().item() / float(eps)))
    if str(store.latent_parameterization).strip().lower() == "sphere":
        store.load_materialized_latent_slots_state_dict(materialized, strict=False, update_radii=False)
    else:
        _load_flat_latents(store, z0)
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}


@torch.no_grad()
def _load_flat_latents(store: BigVAELatentTensorStore, flat_latents: torch.Tensor) -> None:
    offset = 0
    for key in _sorted_latent_slot_keys(store):
        target = store.latent_slots[key]
        numel = int(target.numel())
        value = flat_latents[offset : offset + numel].view_as(target).to(device=target.device, dtype=target.dtype)
        store.load_materialized_latent_slots_state_dict({key: value}, strict=False, update_radii=False)
        offset += numel


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, *, device: torch.device, amp_enabled: bool) -> EvalMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss.detach().cpu().item())
        correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu().item())
        examples += int(labels.numel())
    return EvalMetrics(loss=loss_sum / max(1, examples), accuracy=correct / max(1, examples), examples=examples)


def resolve_optimizer_class(name: str) -> type[torch.optim.Optimizer]:
    normalized = str(name).strip().casefold()
    for attr_name in dir(torch.optim):
        attr = getattr(torch.optim, attr_name)
        if inspect.isclass(attr) and issubclass(attr, torch.optim.Optimizer) and attr is not torch.optim.Optimizer:
            if attr_name.casefold() == normalized:
                return attr
    raise ValueError(f"Unknown torch optimizer: {name}")


def build_optimizer(parameters: list[torch.nn.Parameter], cfg: RunConfig) -> torch.optim.Optimizer:
    optimizer_cls = resolve_optimizer_class(cfg.train.optimizer_name)
    signature = inspect.signature(optimizer_cls.__init__)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name not in {"self", "params"}
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    kwargs: dict[str, Any] = {}
    if "lr" in accepted:
        kwargs["lr"] = float(cfg.train.lr)
    if "weight_decay" in accepted:
        kwargs["weight_decay"] = float(cfg.train.weight_decay)
    if "betas" in accepted:
        kwargs["betas"] = (float(cfg.train.adam_beta1), float(cfg.train.adam_beta2))
    if "eps" in accepted:
        kwargs["eps"] = float(cfg.train.adam_eps)
    kwargs.update(dict(cfg.train.optimizer_kwargs))
    return optimizer_cls(parameters, **kwargs)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: RunConfig,
    *,
    planned_steps: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    if cfg.setup.kind != "latent" or cfg.train.latent_lr_scheduler == "constant":
        return None, "constant"
    decay_steps = max(1, min(int(planned_steps), int(cfg.train.latent_lr_decay_steps)))
    floor_ratio = float(cfg.train.latent_lr_floor_ratio)

    def _lambda(step_index: int) -> float:
        progress = min(1.0, max(0.0, float(step_index) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(floor_ratio + (1.0 - floor_ratio) * cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)
    description = (
        f"cosine_decay_to_floor(floor_ratio={floor_ratio:g}, "
        f"decay_steps={decay_steps}/{int(planned_steps)})"
    )
    return scheduler, description


def _checkpoint_payload(
    cfg: RunConfig,
    model: nn.Module,
    initial_tensors: dict[str, torch.Tensor],
    *,
    step: int,
    epoch: int,
    best_accuracy: float,
    source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    target = getattr(model, "_orig_mod", model)
    store = getattr(target, "store")
    payload: dict[str, Any] = {
        "step": int(step),
        "epoch": int(epoch),
        "best_accuracy": float(best_accuracy),
        "setup_kind": str(cfg.setup.kind),
        "profile": cfg.profile.name,
        "config": cfg.to_dict(),
        "vit_config": asdict(cfg.vit_cfg),
        "named_tensors": export_named_tensors(model, list(initial_tensors.keys())),
        "source_checkpoint_path": str(source_payload.get("_checkpoint_path", "")) if source_payload else "",
    }
    if isinstance(store, BigVAELatentTensorStore):
        payload["latent_slots"] = store.materialized_latent_slots_state_dict()
        payload["latent_space"] = str(store.latent_space)
        payload["latent_parameterization"] = str(store.latent_parameterization)
    return payload


def save_checkpoint(
    paths: RunPaths,
    cfg: RunConfig,
    model: nn.Module,
    initial_tensors: dict[str, torch.Tensor],
    *,
    name: str,
    step: int,
    epoch: int,
    best_accuracy: float,
    source_payload: dict[str, Any] | None,
) -> Path:
    payload = _checkpoint_payload(
        cfg,
        model,
        initial_tensors,
        step=step,
        epoch=epoch,
        best_accuracy=best_accuracy,
        source_payload=source_payload,
    )
    path = paths.checkpoints_dir / f"{name}.pt"
    shared_path = paths.shared_checkpoints_dir / f"{name}.pt"
    torch.save(payload, path)
    torch.save(payload, shared_path)
    latest_meta = {
        "run_id": paths.run_id,
        "run_label": cfg.run_label,
        "shared_checkpoint_label": cfg.shared_checkpoint_label,
        "checkpoint_name": name,
        "per_run_checkpoint_path": str(path),
        "shared_checkpoint_path": str(shared_path),
        "step": int(step),
        "epoch": int(epoch),
        "best_accuracy": float(best_accuracy),
    }
    (paths.shared_checkpoints_dir / f"{name}.meta.json").write_text(
        json.dumps(latest_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if name in {"best", "latest", "final"}:
        mirror_meta_path = paths.shared_checkpoints_dir / f"{name}.from_run.json"
        mirror_meta_path.write_text(json.dumps(latest_meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_training(
    cfg: RunConfig,
    paths: RunPaths,
    logger: logging.Logger,
    comet: CometTracker,
) -> RunResult:
    device = resolve_device(cfg.train.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.train.tf32)
        torch.backends.cudnn.benchmark = True

    seed_everything(int(cfg.train.seed))
    train_loader, test_loader = build_loaders(cfg.data, cfg.model, device, seed=cfg.train.seed)
    initial_tensors, source_payload = prepare_initial_tensors(cfg, storage_root=paths.root_dir, logger=logger)
    big_vae, prior = load_big_vae_components(cfg, device=device, logger=logger)
    model = build_model(cfg, cfg.vit_cfg, initial_tensors, big_vae=big_vae, prior=prior).to(device)

    if cfg.init.kind == "source":
        maybe_restore_direct_latent_state(
            model,
            source_payload,
            enabled=bool(cfg.init.source_prefer_direct_latent),
            logger=logger,
        )
    if cfg.init.kind == "diffusion_prior":
        calibration_images = collect_calibration_images(train_loader, num_batches=int(cfg.init.calibration_batches))
        logger.info(
            "Collected %s calibration images for diffusion-prior init",
            0 if calibration_images is None else int(calibration_images.shape[0]),
        )
        initialize_diffusion_prior(model, calibration_images)
    if cfg.train.compile:
        model = torch.compile(model)

    target = getattr(model, "_orig_mod", model)
    store = getattr(target, "store")
    trainable_params = count_trainable_parameters(model)
    decoded_params = int(store.decoded_numel())
    latent_params = int(store.latent_numel()) if isinstance(store, BigVAELatentTensorStore) else 0
    decoded_big_vae_params = int(store.big_vae_decoded_numel()) if isinstance(store, BigVAELatentTensorStore) else 0
    tile_count = int(store.decoded_tile_count()) if isinstance(store, BigVAELatentTensorStore) else 0

    logger.info(
        "run_id=%s profile=%s setup=%s init=%s trainable_params=%s decoded_params=%s latent_params=%s tiles=%s shared_ckpt_dir=%s",
        paths.run_id,
        cfg.profile.name,
        cfg.setup.kind,
        cfg.init.kind if cfg.init.kind != "fresh" else f"fresh/{cfg.init.fresh_latent_mode}",
        trainable_params,
        decoded_params,
        latent_params,
        tile_count,
        paths.shared_checkpoints_dir,
    )

    optimizer = build_optimizer([param for param in model.parameters() if param.requires_grad], cfg)
    planned_steps = max(1, int(cfg.train.max_steps))
    lr_scheduler, lr_schedule_description = build_lr_scheduler(optimizer, cfg, planned_steps=planned_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and bool(cfg.train.amp)))

    save_checkpoint(
        paths,
        cfg,
        model,
        initial_tensors,
        name="init",
        step=0,
        epoch=0,
        best_accuracy=0.0,
        source_payload=source_payload,
    )

    global_step = 0
    best_accuracy = 0.0
    best_step = 0
    last_eval = EvalMetrics(loss=float("nan"), accuracy=0.0, examples=0)
    train_loss_window = 0.0
    train_count_window = 0
    start_time = time.time()

    for epoch_idx in range(int(cfg.train.epochs)):
        model.train()
        for images, labels in train_loader:
            global_step += 1
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            should_log = global_step == 1 or global_step % int(cfg.logging.log_every_steps) == 0
            should_eval = global_step == 1 or global_step % int(cfg.logging.eval_every_steps) == 0
            if global_step >= int(cfg.train.max_steps):
                should_eval = True

            latent_debug = bool(
                cfg.setup.kind == "latent"
                and isinstance(store, BigVAELatentTensorStore)
                and cfg.logging.latent_debug_metrics
                and (should_log or should_eval)
            )
            pre_step_latent = flatten_latent_slots(store) if latent_debug else None
            pre_step_decoded = flatten_decoded_bigvae_weights(store) if latent_debug else None

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, bool(cfg.train.amp)):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=float(cfg.train.label_smoothing))
            scaler.scale(loss).backward()
            if float(cfg.train.grad_clip_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            step_lr = float(optimizer.param_groups[0]["lr"])
            if lr_scheduler is not None:
                lr_scheduler.step()

            batch_examples = int(labels.numel())
            train_loss_window += float(loss.detach().cpu().item()) * batch_examples
            train_count_window += batch_examples

            latent_metrics = {
                "latent_z_norm": float("nan"),
                "delta_z_norm": float("nan"),
                "delta_w_norm": float("nan"),
                "delta_w_over_delta_z": float("nan"),
                "delta_z_cosine_with_z": float("nan"),
                "delta_z_parallel_ratio": float("nan"),
                "delta_z_tangent_ratio": float("nan"),
                "decoder_jacobian_fd_mean": float("nan"),
                "decoder_jacobian_fd_std": float("nan"),
                "decoder_jacobian_fd_max": float("nan"),
            }
            if latent_debug:
                assert pre_step_latent is not None and pre_step_decoded is not None
                post_step_latent = flatten_latent_slots(store)
                post_step_decoded = flatten_decoded_bigvae_weights(store)
                delta_z = post_step_latent - pre_step_latent
                delta_z_norm = float(delta_z.norm().item())
                delta_w_norm = float((post_step_decoded - pre_step_decoded).norm().item())
                tiny = float(torch.finfo(post_step_latent.dtype).tiny)
                delta_w_over_delta_z = delta_w_norm / max(delta_z_norm, tiny)
                latent_z_norm = float(post_step_latent.norm().item())
                if delta_z_norm > 0.0 and latent_z_norm > 0.0:
                    z_unit = post_step_latent / post_step_latent.norm().clamp_min(torch.finfo(post_step_latent.dtype).tiny)
                    parallel = torch.dot(delta_z, z_unit) * z_unit
                    tangent = delta_z - parallel
                    latent_metrics["delta_z_cosine_with_z"] = float(
                        torch.dot(delta_z, post_step_latent).item() / max(delta_z_norm * latent_z_norm, tiny)
                    )
                    latent_metrics["delta_z_parallel_ratio"] = float(parallel.norm().item() / max(delta_z_norm, tiny))
                    latent_metrics["delta_z_tangent_ratio"] = float(tangent.norm().item() / max(delta_z_norm, tiny))
                jacobian = estimate_decoder_effective_jacobian_norm(
                    store,
                    eps=float(cfg.logging.latent_jacobian_eps),
                    num_probes=int(cfg.logging.latent_jacobian_probes),
                )
                latent_metrics.update(
                    {
                        "latent_z_norm": float(latent_z_norm),
                        "delta_z_norm": float(delta_z_norm),
                        "delta_w_norm": float(delta_w_norm),
                        "delta_w_over_delta_z": float(delta_w_over_delta_z),
                        "decoder_jacobian_fd_mean": float(jacobian["mean"]),
                        "decoder_jacobian_fd_std": float(jacobian["std"]),
                        "decoder_jacobian_fd_max": float(jacobian["max"]),
                    }
                )

            if should_eval:
                last_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.train.amp))
                if last_eval.accuracy > best_accuracy:
                    best_accuracy = float(last_eval.accuracy)
                    best_step = int(global_step)
                    if cfg.storage.save_best:
                        best_path = save_checkpoint(
                            paths,
                            cfg,
                            model,
                            initial_tensors,
                            name="best",
                            step=global_step,
                            epoch=epoch_idx + 1,
                            best_accuracy=best_accuracy,
                            source_payload=source_payload,
                        )
                        comet.log_asset(best_path)

            if should_log or should_eval:
                avg_train_loss = train_loss_window / max(1, train_count_window)
                elapsed_s = float(time.time() - start_time)
                row = {
        "run_id": paths.run_id,
        "profile": cfg.profile.name,
        "run_label": cfg.run_label,
        "dataset": cfg.data.dataset,
        "setup_kind": cfg.setup.kind,
                    "init_kind": cfg.init.kind,
                    "step": int(global_step),
                    "epoch": int(epoch_idx + 1),
                    "lr": float(step_lr),
                    "train_loss": float(avg_train_loss),
                    "test_loss": float(last_eval.loss),
                    "test_accuracy": float(last_eval.accuracy),
                    "best_accuracy": float(best_accuracy),
                    "elapsed_s": elapsed_s,
                    "trainable_params": int(trainable_params),
                    "decoded_params": int(decoded_params),
                    "latent_params": int(latent_params),
                    **latent_metrics,
                }
                append_metrics_row(paths.metrics_file, row)
                comet.log_metrics(
                    {
                        key: float(value)
                        for key, value in row.items()
                        if isinstance(value, (int, float)) and math.isfinite(float(value))
                    },
                    step=global_step,
                )
                logger.info(
                    "step=%s epoch=%s lr=%.6g train_loss=%.4f test_loss=%.4f test_acc=%.4f best=%.4f",
                    global_step,
                    epoch_idx + 1,
                    step_lr,
                    avg_train_loss,
                    last_eval.loss,
                    last_eval.accuracy,
                    best_accuracy,
                )
                train_loss_window = 0.0
                train_count_window = 0

            if cfg.storage.checkpoint_every_steps > 0 and global_step % int(cfg.storage.checkpoint_every_steps) == 0:
                step_name = f"step_{global_step:06d}"
                step_path = save_checkpoint(
                    paths,
                    cfg,
                    model,
                    initial_tensors,
                    name=step_name,
                    step=global_step,
                    epoch=epoch_idx + 1,
                    best_accuracy=best_accuracy,
                    source_payload=source_payload,
                )
                if cfg.storage.save_latest:
                    latest_path = save_checkpoint(
                        paths,
                        cfg,
                        model,
                        initial_tensors,
                        name="latest",
                        step=global_step,
                        epoch=epoch_idx + 1,
                        best_accuracy=best_accuracy,
                        source_payload=source_payload,
                    )
                    comet.log_asset(latest_path)
                comet.log_asset(step_path)

            if global_step >= int(cfg.train.max_steps):
                break
        if global_step >= int(cfg.train.max_steps):
            break

    final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.train.amp))
    if final_eval.accuracy > best_accuracy:
        best_accuracy = float(final_eval.accuracy)
        best_step = int(global_step)

    if cfg.storage.save_latest:
        save_checkpoint(
            paths,
            cfg,
            model,
            initial_tensors,
            name="latest",
            step=global_step,
            epoch=epoch_idx + 1,
            best_accuracy=best_accuracy,
            source_payload=source_payload,
        )
    final_path = save_checkpoint(
        paths,
        cfg,
        model,
        initial_tensors,
        name="final",
        step=global_step,
        epoch=epoch_idx + 1,
        best_accuracy=best_accuracy,
        source_payload=source_payload,
    )
    comet.log_asset(final_path)

    summary = {
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_label": cfg.run_label,
        "run_label_dir": str(paths.run_label_dir),
        "shared_checkpoint_label": cfg.shared_checkpoint_label,
        "shared_checkpoints_dir": str(paths.shared_checkpoints_dir),
        "profile": cfg.profile.name,
        "dataset": cfg.data.dataset,
        "model_size": cfg.profile.model_size,
        "setup_kind": cfg.setup.kind,
        "init_kind": cfg.init.kind,
        "fresh_latent_mode": cfg.init.fresh_latent_mode,
        "steps": int(global_step),
        "best_step": int(best_step),
        "final_test_loss": float(final_eval.loss),
        "final_test_accuracy": float(final_eval.accuracy),
        "best_test_accuracy": float(best_accuracy),
        "trainable_params": int(trainable_params),
        "decoded_params": int(decoded_params),
        "latent_params": int(latent_params),
        "bigvae_decoded_params": int(decoded_big_vae_params),
        "tile_count": int(tile_count),
        "lr_schedule": str(lr_schedule_description),
        "source_checkpoint_path": str(source_payload.get("_checkpoint_path", "")) if source_payload else "",
    }
    write_summary(paths.summary_file, summary)
    update_run_index(paths, cfg, summary)
    return RunResult(summary=summary)
