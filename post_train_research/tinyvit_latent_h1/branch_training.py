from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.compare_vit_tiny_latent_optimization import BigVAELatentTensorStore, autocast_context
from post_train_research.tinyvit_latent_h1.config import RunConfig
from post_train_research.tinyvit_latent_h1.runtime import CometTracker, RunPaths, append_csv_row
from post_train_research.tinyvit_latent_h1.source import (
    SourceContext,
    StartPoint,
    build_latent_model,
    build_raw_model,
    build_train_schedule,
    conditioning_state_dict,
    evaluate_train_and_test,
    fetch_batch,
    sanitize_float,
)
from post_train_research.vit_latent_scaling.init import export_named_tensors


@dataclass(slots=True)
class BranchResult:
    branch: str
    lr: float
    start_train_loss: float
    best_train_loss: float
    final_train_loss: float
    start_test_loss: float
    best_test_loss: float
    final_test_loss: float
    start_test_acc: float
    best_test_acc: float
    final_test_acc: float
    recovered: bool
    steps_to_recover: int
    gain: float
    curve_rows: list[dict[str, Any]]
    best_state: dict[str, Any]
    final_state: dict[str, Any]


def branch_checkpoint_payload(
    model: nn.Module,
    source: SourceContext,
    *,
    branch: str,
    start_id: str,
    lr: float,
    step: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branch": str(branch),
        "start_id": str(start_id),
        "lr": float(lr),
        "step": int(step),
        "summary": dict(summary),
        "source_checkpoint_path": str(source.checkpoint_path),
        "vit_config": asdict(source.vit_cfg),
        "named_tensors": export_named_tensors(model, source.all_tensor_names),
    }
    store = getattr(model, "store")
    if isinstance(store, BigVAELatentTensorStore):
        payload["latent_slots"] = store.materialized_latent_slots_state_dict()
        payload["tile_cond_patch"] = conditioning_state_dict(store)
        payload["latent_space"] = str(store.latent_space)
        payload["latent_parameterization"] = str(store.latent_parameterization)
    return payload


def select_best_branch_result(results: list[BranchResult]) -> BranchResult:
    return min(results, key=lambda item: (float(item.best_train_loss), -float(item.best_test_acc), float(item.final_train_loss)))


def _save_anchor_checkpoint(
    path: Path,
    model: nn.Module,
    source: SourceContext,
    *,
    lr: float,
    step: int,
    train_loss: float,
    test_loss: float,
    test_accuracy: float,
) -> dict[str, Any]:
    payload = branch_checkpoint_payload(
        model,
        source,
        branch="anchor",
        start_id="z_star",
        lr=lr,
        step=step,
        summary={"train_loss": train_loss, "test_loss": test_loss, "test_accuracy": test_accuracy},
    )
    torch.save(payload, path)
    return payload


def train_anchor_source(
    cfg: RunConfig,
    source: SourceContext,
    paths: RunPaths,
    *,
    train_dataset: Any,
    train_loader: Any,
    test_loader: Any,
    device: torch.device,
    logger: Any,
    comet: CometTracker,
) -> SourceContext:
    model = build_latent_model(
        cfg,
        source,
        device=device,
        latent_state=source.z_star_state,
        conditioning_state=source.z_star_conditioning_state,
        freeze_direct=False,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg.source.anchor_lr),
        weight_decay=float(cfg.source.anchor_weight_decay),
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
        eps=float(cfg.train.adam_eps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and bool(cfg.train.amp)))
    start_train, start_test = evaluate_train_and_test(model, train_loader, test_loader, device=device, amp_enabled=bool(cfg.train.amp))
    best_train = float(start_train.loss)
    best_train_acc = float(start_train.accuracy)
    best_test_loss = float(start_test.loss)
    best_test_acc = float(start_test.accuracy)
    best_payload = _save_anchor_checkpoint(
        paths.anchor_dir / "init.pt",
        model,
        source,
        lr=float(cfg.source.anchor_lr),
        step=0,
        train_loss=float(start_train.loss),
        test_loss=float(start_test.loss),
        test_accuracy=float(start_test.accuracy),
    )
    torch.save(best_payload, paths.anchor_dir / "best.pt")
    torch.save(best_payload, paths.anchor_dir / "latest.pt")
    logger.info(
        "Anchor step=%s batch_loss=nan train_loss=%.6f test_loss=%.6f test_acc=%.4f best_train=%.6f best_test_acc=%.4f",
        0,
        float(start_train.loss),
        float(start_test.loss),
        float(start_test.accuracy),
        float(best_train),
        float(best_test_acc),
    )
    append_csv_row(
        paths.anchor_curve_file,
        {
            "step": 0,
            "batch_loss": None,
            "train_loss": float(start_train.loss),
            "test_loss": float(start_test.loss),
            "test_accuracy": float(start_test.accuracy),
        },
    )
    train_schedule = build_train_schedule(
        len(train_dataset),
        batch_size=int(cfg.data.train_batch_size),
        steps=int(cfg.source.anchor_steps),
        seed=int(cfg.train.seed),
    )
    final_train = float(start_train.loss)
    final_test_loss = float(start_test.loss)
    final_test_acc = float(start_test.accuracy)
    for step_idx, batch_indices in enumerate(train_schedule, start=1):
        model.train()
        images, labels = fetch_batch(train_dataset, batch_indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(cfg.train.amp)):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        if float(cfg.train.grad_clip_norm) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        should_eval = step_idx == 1 or step_idx % int(cfg.train.eval_every_steps) == 0 or step_idx == int(cfg.source.anchor_steps)
        should_log = step_idx == 1 or step_idx % int(cfg.train.log_every_steps) == 0 or should_eval
        train_loss_value = None
        test_loss_value = None
        test_acc_value = None
        if should_eval:
            train_metrics, test_metrics = evaluate_train_and_test(
                model,
                train_loader,
                test_loader,
                device=device,
                amp_enabled=bool(cfg.train.amp),
            )
            final_train = float(train_metrics.loss)
            final_test_loss = float(test_metrics.loss)
            final_test_acc = float(test_metrics.accuracy)
            if (float(train_metrics.loss), -float(test_metrics.accuracy)) < (best_train, -best_test_acc):
                best_train = float(train_metrics.loss)
                best_train_acc = float(train_metrics.accuracy)
                best_test_loss = float(test_metrics.loss)
                best_test_acc = float(test_metrics.accuracy)
                best_payload = _save_anchor_checkpoint(
                    paths.anchor_dir / "best.pt",
                    model,
                    source,
                    lr=float(cfg.source.anchor_lr),
                    step=step_idx,
                    train_loss=best_train,
                    test_loss=best_test_loss,
                    test_accuracy=best_test_acc,
                )
            train_loss_value = float(train_metrics.loss)
            test_loss_value = float(test_metrics.loss)
            test_acc_value = float(test_metrics.accuracy)
        if should_log:
            append_csv_row(
                paths.anchor_curve_file,
                {
                    "step": int(step_idx),
                    "batch_loss": float(loss.detach().cpu().item()),
                    "train_loss": train_loss_value,
                    "test_loss": test_loss_value,
                    "test_accuracy": test_acc_value,
                },
            )
            metrics = {"anchor.batch_loss": float(loss.detach().cpu().item())}
            if train_loss_value is not None:
                metrics["anchor.train_loss"] = float(train_loss_value)
                metrics["anchor.test_loss"] = float(test_loss_value)
                metrics["anchor.test_accuracy"] = float(test_acc_value)
            comet.log_metrics(metrics, step=int(step_idx))
            logger.info(
                "Anchor step=%s batch_loss=%.6f train_loss=%s test_loss=%s test_acc=%s best_train=%.6f best_test_acc=%.4f",
                int(step_idx),
                float(loss.detach().cpu().item()),
                f"{float(train_loss_value):.6f}" if train_loss_value is not None else "na",
                f"{float(test_loss_value):.6f}" if test_loss_value is not None else "na",
                f"{float(test_acc_value):.4f}" if test_acc_value is not None else "na",
                float(best_train),
                float(best_test_acc),
            )
        if step_idx % int(cfg.train.eval_every_steps) == 0 or step_idx == int(cfg.source.anchor_steps):
            _save_anchor_checkpoint(
                paths.anchor_dir / "latest.pt",
                model,
                source,
                lr=float(cfg.source.anchor_lr),
                step=step_idx,
                train_loss=final_train,
                test_loss=final_test_loss,
                test_accuracy=final_test_acc,
            )
    _save_anchor_checkpoint(
        paths.anchor_dir / "final.pt",
        model,
        source,
        lr=float(cfg.source.anchor_lr),
        step=int(cfg.source.anchor_steps),
        train_loss=final_train,
        test_loss=final_test_loss,
        test_accuracy=final_test_acc,
    )
    best_named = best_payload["named_tensors"]
    raw_param_count = int(sum(int(best_named[name].numel()) for name in source.decoded_names))
    logger.info(
        "Anchor training finished steps=%s best_train=%.6f best_test_acc=%.4f final_test_acc=%.4f",
        int(cfg.source.anchor_steps),
        float(best_train),
        float(best_test_acc),
        float(final_test_acc),
    )
    return replace(
        source,
        checkpoint_path=(paths.anchor_dir / "best.pt"),
        payload=best_payload,
        z_star_state=best_payload["latent_slots"],
        z_star_conditioning_state=best_payload.get("tile_cond_patch", {}),
        z_star_named_tensors=best_payload["named_tensors"],
        z_star_train_metrics=type(source.z_star_train_metrics)(
            loss=best_train,
            accuracy=best_train_acc,
            examples=source.z_star_train_metrics.examples,
        ),
        z_star_test_metrics=type(source.z_star_test_metrics)(
            loss=best_test_loss,
            accuracy=best_test_acc,
            examples=source.z_star_test_metrics.examples,
        ),
        raw_param_count=raw_param_count,
    )


def run_branch_for_lr(
    cfg: RunConfig,
    source: SourceContext,
    start: StartPoint,
    *,
    branch: str,
    lr: float,
    train_dataset: Any,
    train_schedule: list[torch.Tensor],
    train_loader: Any,
    test_loader: Any,
    device: torch.device,
    comet: CometTracker,
    logger: Any,
) -> BranchResult:
    if branch == "latent":
        model = build_latent_model(
            cfg,
            source,
            device=device,
            latent_state=start.latent_state,
            conditioning_state=source.z_star_conditioning_state,
        )
        weight_decay = float(cfg.train.latent_weight_decay)
    elif branch == "raw":
        model = build_raw_model(source, device=device, start_named_tensors=start.named_tensors)
        weight_decay = float(cfg.train.raw_weight_decay)
    else:
        raise ValueError(f"Unsupported branch: {branch!r}")
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(lr),
        weight_decay=weight_decay,
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
        eps=float(cfg.train.adam_eps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and bool(cfg.train.amp)))
    start_train, start_test = evaluate_train_and_test(model, train_loader, test_loader, device=device, amp_enabled=bool(cfg.train.amp))
    best_train = float(start_train.loss)
    best_test_loss = float(start_test.loss)
    best_test_acc = float(start_test.accuracy)
    final_train = float(start_train.loss)
    final_test_loss = float(start_test.loss)
    final_test_acc = float(start_test.accuracy)
    steps_to_recover = 0 if float(start_train.loss) <= float(source.z_star_train_metrics.loss + cfg.train.recover_eps) else -1
    curve_rows: list[dict[str, Any]] = [
        {
            "branch": branch,
            "lr": float(lr),
            "step": 0,
            "batch_loss": None,
            "train_loss": float(start_train.loss),
            "test_loss": float(start_test.loss),
            "test_accuracy": float(start_test.accuracy),
        }
    ]
    best_state = branch_checkpoint_payload(
        model,
        source,
        branch=branch,
        start_id=start.start_id,
        lr=lr,
        step=0,
        summary={"train_loss": float(start_train.loss), "test_loss": float(start_test.loss), "test_accuracy": float(start_test.accuracy)},
    )
    label = f"{start.start_id}.{branch}.lr={float(lr):g}"
    logger.info(
        "Branch %s step=%s batch_loss=nan train_loss=%.6f test_loss=%.6f test_acc=%.4f best_train=%.6f best_test_acc=%.4f",
        label,
        0,
        float(start_train.loss),
        float(start_test.loss),
        float(start_test.accuracy),
        float(best_train),
        float(best_test_acc),
    )
    for step_idx, batch_indices in enumerate(train_schedule, start=1):
        model.train()
        images, labels = fetch_batch(train_dataset, batch_indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(cfg.train.amp)):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        if float(cfg.train.grad_clip_norm) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        should_eval = step_idx == 1 or step_idx % int(cfg.train.eval_every_steps) == 0 or step_idx == int(cfg.train.steps)
        should_log = step_idx == 1 or step_idx % int(cfg.train.log_every_steps) == 0 or should_eval
        train_loss_value = None
        test_loss_value = None
        test_acc_value = None
        if should_eval:
            train_metrics, test_metrics = evaluate_train_and_test(model, train_loader, test_loader, device=device, amp_enabled=bool(cfg.train.amp))
            final_train = float(train_metrics.loss)
            final_test_loss = float(test_metrics.loss)
            final_test_acc = float(test_metrics.accuracy)
            if float(train_metrics.loss) < best_train:
                best_train = float(train_metrics.loss)
                best_test_loss = float(test_metrics.loss)
                best_test_acc = float(test_metrics.accuracy)
                best_state = branch_checkpoint_payload(
                    model,
                    source,
                    branch=branch,
                    start_id=start.start_id,
                    lr=lr,
                    step=step_idx,
                    summary={"train_loss": best_train, "test_loss": best_test_loss, "test_accuracy": best_test_acc},
                )
            if steps_to_recover < 0 and float(train_metrics.loss) <= float(source.z_star_train_metrics.loss + cfg.train.recover_eps):
                steps_to_recover = int(step_idx)
            train_loss_value = float(train_metrics.loss)
            test_loss_value = float(test_metrics.loss)
            test_acc_value = float(test_metrics.accuracy)
        if should_log:
            row = {
                "branch": branch,
                "lr": float(lr),
                "step": int(step_idx),
                "batch_loss": float(loss.detach().cpu().item()),
                "train_loss": train_loss_value,
                "test_loss": test_loss_value,
                "test_accuracy": test_acc_value,
            }
            curve_rows.append(row)
            prefix = f"{start.start_id}.{branch}.lr_{sanitize_float(float(lr))}"
            metrics = {f"{prefix}.batch_loss": float(loss.detach().cpu().item())}
            if train_loss_value is not None:
                metrics[f"{prefix}.train_loss"] = float(train_loss_value)
                metrics[f"{prefix}.test_loss"] = float(test_loss_value)
                metrics[f"{prefix}.test_accuracy"] = float(test_acc_value)
            comet.log_metrics(metrics, step=int(step_idx))
            logger.info(
                "Branch %s step=%s batch_loss=%.6f train_loss=%s test_loss=%s test_acc=%s best_train=%.6f best_test_acc=%.4f",
                label,
                int(step_idx),
                float(loss.detach().cpu().item()),
                f"{float(train_loss_value):.6f}" if train_loss_value is not None else "na",
                f"{float(test_loss_value):.6f}" if test_loss_value is not None else "na",
                f"{float(test_acc_value):.4f}" if test_acc_value is not None else "na",
                float(best_train),
                float(best_test_acc),
            )
    final_state = branch_checkpoint_payload(
        model,
        source,
        branch=branch,
        start_id=start.start_id,
        lr=lr,
        step=int(cfg.train.steps),
        summary={"train_loss": final_train, "test_loss": final_test_loss, "test_accuracy": final_test_acc},
    )
    logger.info(
        "Branch %s finished best_train=%.6f final_train=%.6f best_test_acc=%.4f final_test_acc=%.4f recovered=%s steps_to_recover=%s",
        label,
        float(best_train),
        float(final_train),
        float(best_test_acc),
        float(final_test_acc),
        steps_to_recover >= 0,
        max(0, int(steps_to_recover)),
    )
    return BranchResult(
        branch=str(branch),
        lr=float(lr),
        start_train_loss=float(start_train.loss),
        best_train_loss=float(best_train),
        final_train_loss=float(final_train),
        start_test_loss=float(start_test.loss),
        best_test_loss=float(best_test_loss),
        final_test_loss=float(final_test_loss),
        start_test_acc=float(start_test.accuracy),
        best_test_acc=float(best_test_acc),
        final_test_acc=float(final_test_acc),
        recovered=steps_to_recover >= 0,
        steps_to_recover=max(0, int(steps_to_recover)),
        gain=float(start_train.loss - best_train),
        curve_rows=curve_rows,
        best_state=best_state,
        final_state=final_state,
    )
