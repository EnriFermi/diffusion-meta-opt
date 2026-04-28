from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.compare_vit_tiny_latent_optimization import BigVAELatentTensorStore
from post_train_research.tinyvit_latent_h1.config import RunConfig
from post_train_research.tinyvit_latent_h1.runtime import CometTracker, RunPaths, append_csv_row, update_run_index, write_summary
from post_train_research.tinyvit_latent_h1.source import (
    SourceContext,
    StartPoint,
    build_cifar10_datasets,
    build_eval_loader,
    build_latent_model,
    build_raw_model,
    build_train_schedule,
    evaluate_train_and_test,
    fetch_batch,
    prepare_config_from_source_checkpoint,
    resolve_source_context,
    sanitize_float,
    search_start_points,
)
from post_train_research.vit_latent_scaling.init import export_named_tensors, resolve_device, seed_everything


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


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _branch_checkpoint_payload(
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
        payload["latent_space"] = str(store.latent_space)
        payload["latent_parameterization"] = str(store.latent_parameterization)
    return payload


def _select_best_branch_result(results: list[BranchResult]) -> BranchResult:
    return min(results, key=lambda item: (float(item.best_train_loss), -float(item.best_test_acc), float(item.final_train_loss)))


def _plot_paired_curves(path: Path, latent: BranchResult, raw: BranchResult) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for result, label in ((latent, "latent"), (raw, "raw")):
        eval_rows = [row for row in result.curve_rows if row.get("train_loss") is not None]
        steps = [int(row["step"]) for row in eval_rows]
        axes[0].plot(steps, [float(row["train_loss"]) for row in eval_rows], label=f"{label}@{result.lr:g}")
        axes[1].plot(steps, [float(row["test_loss"]) for row in eval_rows], label=f"{label}@{result.lr:g}")
        axes[2].plot(steps, [float(row["test_accuracy"]) for row in eval_rows], label=f"{label}@{result.lr:g}")
    axes[0].set_title("Train loss")
    axes[1].set_title("Test loss")
    axes[2].set_title("Test accuracy")
    for axis in axes:
        axis.set_xlabel("step")
        axis.legend()
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _tensor_dict_metadata(mapping: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in mapping.items():
        if torch.is_tensor(value):
            metadata[str(key)] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            metadata[str(key)] = {"type": type(value).__name__}
    return metadata


def _checkpoint_payload_json_view(payload: dict[str, Any]) -> dict[str, Any]:
    view = {
        key: value
        for key, value in payload.items()
        if key not in {"named_tensors", "latent_slots"}
    }
    named_tensors = payload.get("named_tensors")
    if isinstance(named_tensors, dict):
        view["named_tensor_count"] = len(named_tensors)
        view["named_tensors"] = _tensor_dict_metadata(named_tensors)
    latent_slots = payload.get("latent_slots")
    if isinstance(latent_slots, dict):
        view["latent_slot_count"] = len(latent_slots)
        view["latent_slots"] = _tensor_dict_metadata(latent_slots)
    return view


def _branch_result_json_view(result: BranchResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["best_state"] = _checkpoint_payload_json_view(result.best_state)
    payload["final_state"] = _checkpoint_payload_json_view(result.final_state)
    return payload


def _run_branch_for_lr(
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
) -> BranchResult:
    if branch == "latent":
        model = build_latent_model(cfg, source, device=device, latent_state=start.latent_state)
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
    best_state = _branch_checkpoint_payload(
        model,
        source,
        branch=branch,
        start_id=start.start_id,
        lr=lr,
        step=0,
        summary={"train_loss": float(start_train.loss), "test_loss": float(start_test.loss), "test_accuracy": float(start_test.accuracy)},
    )
    for step_idx, batch_indices in enumerate(train_schedule, start=1):
        model.train()
        images, labels = fetch_batch(train_dataset, batch_indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        from experiments.compare_vit_tiny_latent_optimization import autocast_context

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
                best_state = _branch_checkpoint_payload(
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
    final_state = _branch_checkpoint_payload(
        model,
        source,
        branch=branch,
        start_id=start.start_id,
        lr=lr,
        step=int(cfg.train.steps),
        summary={"train_loss": final_train, "test_loss": final_test_loss, "test_accuracy": final_test_acc},
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


def _write_start_artifacts(start_dir: Path, start: StartPoint, latent: BranchResult, raw: BranchResult) -> None:
    for row in latent.curve_rows + raw.curve_rows:
        append_csv_row(start_dir / "curves.csv", {"start_id": start.start_id, **row})
    (start_dir / "branch_results.json").write_text(
        json.dumps(
            {
                "start": {
                    "start_id": start.start_id,
                    "epsilon": float(start.epsilon),
                    "direction_index": int(start.direction_index),
                    "alpha": float(start.alpha),
                    "train_metrics": asdict(start.train_metrics),
                    "test_metrics": asdict(start.test_metrics),
                },
                "latent": _branch_result_json_view(latent),
                "raw": _branch_result_json_view(raw),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "start_id": start.start_id,
            "start_named_tensors": start.named_tensors,
            "start_latent_slots": start.latent_state,
            "latent_best": latent.best_state,
            "latent_final": latent.final_state,
            "raw_best": raw.best_state,
            "raw_final": raw.final_state,
        },
        start_dir / "states.pt",
    )
    _plot_paired_curves(start_dir / "paired_curves.png", latent, raw)


def _aggregate_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not paired_rows:
        return []
    groups: list[tuple[str, list[dict[str, Any]]]] = [("overall", paired_rows)]
    epsilons = sorted(set(float(row["epsilon"]) for row in paired_rows))
    for epsilon in epsilons:
        groups.append((f"eps_{sanitize_float(epsilon)}", [row for row in paired_rows if float(row["epsilon"]) == epsilon]))
    summaries: list[dict[str, Any]] = []
    for name, rows in groups:
        latent_gains = sorted(float(row["latent_gain"]) for row in rows)
        raw_gains = sorted(float(row["raw_gain"]) for row in rows)
        latent_recovery = [float(bool(row["latent_recovered"])) for row in rows]
        raw_recovery = [float(bool(row["raw_recovered"])) for row in rows]
        latent_median_gain = latent_gains[len(latent_gains) // 2]
        raw_median_gain = raw_gains[len(raw_gains) // 2]
        summaries.append(
            {
                "group": name,
                "count": len(rows),
                "latent_median_gain": float(latent_median_gain),
                "raw_median_gain": float(raw_median_gain),
                "gain_ratio": float(latent_median_gain / raw_median_gain) if raw_median_gain != 0.0 else float("nan"),
                "latent_recovery_rate": float(sum(latent_recovery) / max(1, len(latent_recovery))),
                "raw_recovery_rate": float(sum(raw_recovery) / max(1, len(raw_recovery))),
            }
        )
    return summaries


def run_experiment(
    cfg: RunConfig,
    paths: RunPaths,
    logger: Any,
    comet: CometTracker,
    *,
    source_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(cfg.train.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.train.tf32)
        torch.backends.cudnn.benchmark = True
    seed_everything(int(cfg.train.seed))
    train_dataset, test_dataset = build_cifar10_datasets(cfg)
    train_loader = build_eval_loader(train_dataset, batch_size=int(cfg.data.eval_batch_size), num_workers=int(cfg.data.num_workers), device=device)
    test_loader = build_eval_loader(test_dataset, batch_size=int(cfg.data.eval_batch_size), num_workers=int(cfg.data.num_workers), device=device)
    source = resolve_source_context(cfg, logger, device=device, train_loader=train_loader, test_loader=test_loader, checkpoint_path=source_checkpoint_path)
    anchor_model = build_latent_model(cfg, source, device=device, latent_state=source.z_star_state)
    starts = search_start_points(cfg, source, anchor_model, train_loader, test_loader, device=device, logger=logger)
    if not starts:
        raise RuntimeError("No valid level-set starts were found around z_star")
    train_schedule = build_train_schedule(len(train_dataset), batch_size=int(cfg.data.train_batch_size), steps=int(cfg.train.steps), seed=int(cfg.train.seed))
    paired_rows: list[dict[str, Any]] = []
    starts_payload: dict[str, Any] = {
        "source_checkpoint_path": str(source.checkpoint_path),
        "z_star_train_metrics": asdict(source.z_star_train_metrics),
        "z_star_test_metrics": asdict(source.z_star_test_metrics),
        "starts": {},
    }
    logger.info(
        "Prepared source z_star train_loss=%.6f test_acc=%.4f decoded_names=%s latent_params=%s raw_params=%s starts=%s",
        float(source.z_star_train_metrics.loss),
        float(source.z_star_test_metrics.accuracy),
        len(source.decoded_names),
        int(source.latent_param_count),
        int(source.raw_param_count),
        len(starts),
    )
    for start in starts:
        latent_runs = [
            _run_branch_for_lr(
                cfg,
                source,
                start,
                branch="latent",
                lr=float(lr),
                train_dataset=train_dataset,
                train_schedule=train_schedule,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                comet=comet,
            )
            for lr in cfg.train.latent_lrs
        ]
        raw_runs = [
            _run_branch_for_lr(
                cfg,
                source,
                start,
                branch="raw",
                lr=float(lr),
                train_dataset=train_dataset,
                train_schedule=train_schedule,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                comet=comet,
            )
            for lr in cfg.train.raw_lrs
        ]
        latent_best = _select_best_branch_result(latent_runs)
        raw_best = _select_best_branch_result(raw_runs)
        start_dir = paths.starts_dir / start.start_id
        start_dir.mkdir(parents=True, exist_ok=True)
        _write_start_artifacts(start_dir, start, latent_best, raw_best)
        comet.log_asset(start_dir / "curves.csv")
        comet.log_asset(start_dir / "paired_curves.png")
        paired_row = {
            "start_id": start.start_id,
            "epsilon": float(start.epsilon),
            "direction_index": int(start.direction_index),
            "alpha": float(start.alpha),
            "f_star_train_loss": float(source.z_star_train_metrics.loss),
            "start_train_loss": float(start.train_metrics.loss),
            "start_test_loss": float(start.test_metrics.loss),
            "start_test_acc": float(start.test_metrics.accuracy),
            "latent_lr": float(latent_best.lr),
            "latent_best_train_loss": float(latent_best.best_train_loss),
            "latent_final_train_loss": float(latent_best.final_train_loss),
            "latent_best_test_loss": float(latent_best.best_test_loss),
            "latent_final_test_loss": float(latent_best.final_test_loss),
            "latent_best_test_acc": float(latent_best.best_test_acc),
            "latent_final_test_acc": float(latent_best.final_test_acc),
            "latent_recovered": bool(latent_best.recovered),
            "latent_steps_to_recover": int(latent_best.steps_to_recover),
            "latent_gain": float(latent_best.gain),
            "raw_lr": float(raw_best.lr),
            "raw_best_train_loss": float(raw_best.best_train_loss),
            "raw_final_train_loss": float(raw_best.final_train_loss),
            "raw_best_test_loss": float(raw_best.best_test_loss),
            "raw_final_test_loss": float(raw_best.final_test_loss),
            "raw_best_test_acc": float(raw_best.best_test_acc),
            "raw_final_test_acc": float(raw_best.final_test_acc),
            "raw_recovered": bool(raw_best.recovered),
            "raw_steps_to_recover": int(raw_best.steps_to_recover),
            "raw_gain": float(raw_best.gain),
            "gain_ratio": float(latent_best.gain / raw_best.gain) if float(raw_best.gain) != 0.0 else float("nan"),
        }
        append_csv_row(paths.paired_results_file, paired_row)
        paired_rows.append(paired_row)
        starts_payload["starts"][start.start_id] = {
            "epsilon": float(start.epsilon),
            "direction_index": int(start.direction_index),
            "alpha": float(start.alpha),
            "train_metrics": asdict(start.train_metrics),
            "test_metrics": asdict(start.test_metrics),
            "latent_slots": start.latent_state,
        }
    torch.save(starts_payload, paths.starts_file)
    aggregate_rows = _aggregate_rows(paired_rows)
    for row in aggregate_rows:
        append_csv_row(paths.aggregate_summary_file, row)
    summary = {
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_label": cfg.run_label,
        "source_checkpoint_path": str(source.checkpoint_path),
        "source_run_dir": cfg.source.run_dir,
        "z_star_train_loss": float(source.z_star_train_metrics.loss),
        "z_star_test_loss": float(source.z_star_test_metrics.loss),
        "z_star_test_acc": float(source.z_star_test_metrics.accuracy),
        "decoded_tensor_count": len(source.decoded_names),
        "latent_param_count": int(source.latent_param_count),
        "raw_param_count": int(source.raw_param_count),
        "paired_start_count": len(paired_rows),
        "aggregate": aggregate_rows,
    }
    write_summary(paths.summary_file, summary)
    update_run_index(paths, cfg, summary)
    return summary


__all__ = ["prepare_config_from_source_checkpoint", "run_experiment"]
