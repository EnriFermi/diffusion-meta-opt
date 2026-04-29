from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from post_train_research.tinyvit_latent_h1.bootstrap import (
    prepare_config_from_source_checkpoint,
    resolve_source_context_from_config,
)
from post_train_research.tinyvit_latent_h1.config import RunConfig
from post_train_research.tinyvit_latent_h1.runtime import CometTracker, RunPaths, append_csv_row, update_run_index, write_summary
from post_train_research.tinyvit_latent_h1.source import (
    StartPoint,
    build_cifar10_datasets,
    build_eval_loader,
    build_latent_model,
    build_raw_model,
    build_train_schedule,
    evaluate_train_and_test,
    sanitize_float,
    search_start_points,
)
from post_train_research.tinyvit_latent_h1.branch_training import (
    BranchResult,
    run_branch_for_lr,
    select_best_branch_result,
    train_anchor_source,
)
from post_train_research.vit_latent_scaling.init import export_named_tensors, resolve_device, seed_everything


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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
            metadata[str(key)] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            metadata[str(key)] = {"type": type(value).__name__}
    return metadata


def _checkpoint_payload_json_view(payload: dict[str, Any]) -> dict[str, Any]:
    view = {key: value for key, value in payload.items() if key not in {"named_tensors", "latent_slots", "tile_cond_patch"}}
    for key in ("named_tensors", "latent_slots", "tile_cond_patch"):
        value = payload.get(key)
        if isinstance(value, dict):
            view[f"{key}_count"] = len(value)
            view[key] = _tensor_dict_metadata(value)
    return view


def _branch_result_json_view(result: BranchResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["best_state"] = _checkpoint_payload_json_view(result.best_state)
    payload["final_state"] = _checkpoint_payload_json_view(result.final_state)
    return payload


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
    for epsilon in sorted(set(float(row["epsilon"]) for row in paired_rows)):
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


def _validate_anchor_reconstruction(
    cfg: RunConfig,
    source: Any,
    anchor_model: torch.nn.Module,
    train_loader: Any,
    test_loader: Any,
    *,
    device: torch.device,
) -> None:
    train_metrics, test_metrics = evaluate_train_and_test(
        anchor_model,
        train_loader,
        test_loader,
        device=device,
        amp_enabled=bool(cfg.train.amp),
    )
    train_gap = abs(float(train_metrics.loss) - float(source.z_star_train_metrics.loss))
    test_gap = abs(float(test_metrics.loss) - float(source.z_star_test_metrics.loss))
    acc_gap = abs(float(test_metrics.accuracy) - float(source.z_star_test_metrics.accuracy))
    if train_gap > 1e-2 or test_gap > 1e-2 or acc_gap > 1e-3:
        raise RuntimeError(
            "Anchor reconstruction mismatch before search: "
            f"stored train_loss={float(source.z_star_train_metrics.loss):.6f} rebuilt train_loss={float(train_metrics.loss):.6f}, "
            f"stored test_loss={float(source.z_star_test_metrics.loss):.6f} rebuilt test_loss={float(test_metrics.loss):.6f}, "
            f"stored test_acc={float(source.z_star_test_metrics.accuracy):.4f} rebuilt test_acc={float(test_metrics.accuracy):.4f}"
        )

    rebuilt_named = export_named_tensors(anchor_model, source.all_tensor_names)
    tensor_gap = max(
        float((rebuilt_named[name] - source.z_star_named_tensors[name]).abs().max().item())
        for name in source.all_tensor_names
    )
    if tensor_gap > 1e-5:
        raise RuntimeError(
            "Anchor tensor reconstruction mismatch before search: "
            f"max_abs_diff={tensor_gap:.6e}"
        )


def _validate_start_reconstruction(
    cfg: RunConfig,
    source: Any,
    start: StartPoint,
    *,
    device: torch.device,
) -> None:
    latent_model = build_latent_model(
        cfg,
        source,
        device=device,
        latent_state=start.latent_state,
        conditioning_state=source.z_star_conditioning_state,
    )
    raw_model = build_latent_model(
        cfg,
        source,
        device=device,
        latent_state=start.latent_state,
        conditioning_state=source.z_star_conditioning_state,
    )
    latent_named = export_named_tensors(latent_model, source.all_tensor_names)
    latent_gap = max(
        float((latent_named[name] - start.named_tensors[name]).abs().max().item())
        for name in source.all_tensor_names
    )
    if latent_gap > 1e-5:
        raise RuntimeError(
            f"Latent branch start reconstruction mismatch for {start.start_id}: max_abs_diff={latent_gap:.6e}"
        )
    raw_model = build_raw_model(source, device=device, start_named_tensors=start.named_tensors)
    raw_named = export_named_tensors(raw_model, source.all_tensor_names)
    raw_gap = max(
        float((raw_named[name] - start.named_tensors[name]).abs().max().item())
        for name in source.all_tensor_names
    )
    if raw_gap > 1e-6:
        raise RuntimeError(
            f"Raw branch start reconstruction mismatch for {start.start_id}: max_abs_diff={raw_gap:.6e}"
        )


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

    source = resolve_source_context_from_config(
        cfg,
        logger,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        checkpoint_path=source_checkpoint_path,
    )
    source_init_checkpoint_path = source.checkpoint_path
    if bool(cfg.source.train_anchor):
        source = train_anchor_source(
            cfg,
            source,
            paths,
            train_dataset=train_dataset,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            logger=logger,
            comet=comet,
        )
        comet.log_asset(paths.anchor_curve_file)
        comet.log_asset(paths.anchor_dir / "best.pt")
        comet.log_asset(paths.anchor_dir / "final.pt")

    anchor_model = build_latent_model(
        cfg,
        source,
        device=device,
        latent_state=source.z_star_state,
        conditioning_state=source.z_star_conditioning_state,
    )
    _validate_anchor_reconstruction(
        cfg,
        source,
        anchor_model,
        train_loader,
        test_loader,
        device=device,
    )
    starts = search_start_points(cfg, source, anchor_model, train_loader, test_loader, device=device, logger=logger)
    if not starts:
        raise RuntimeError("No valid level-set starts were found around z_star")

    train_schedule = build_train_schedule(len(train_dataset), batch_size=int(cfg.data.train_batch_size), steps=int(cfg.train.steps), seed=int(cfg.train.seed))
    paired_rows: list[dict[str, Any]] = []
    starts_payload: dict[str, Any] = {
        "source_checkpoint_path": str(source.checkpoint_path),
        "source_init_checkpoint_path": str(source_init_checkpoint_path),
        "z_star_train_metrics": asdict(source.z_star_train_metrics),
        "z_star_test_metrics": asdict(source.z_star_test_metrics),
        "z_star_tile_cond_patch": source.z_star_conditioning_state,
        "starts": {},
    }
    logger.info(
        "Prepared source z_star train_loss=%.6f test_acc=%.4f decoded_names=%s latent_params=%s raw_params=%s starts=%s train_anchor=%s",
        float(source.z_star_train_metrics.loss),
        float(source.z_star_test_metrics.accuracy),
        len(source.decoded_names),
        int(source.latent_param_count),
        int(source.raw_param_count),
        len(starts),
        bool(cfg.source.train_anchor),
    )

    for start in starts:
        _validate_start_reconstruction(
            cfg,
            source,
            start,
            device=device,
        )
        latent_runs = [
            run_branch_for_lr(
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
            run_branch_for_lr(
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
        latent_best = select_best_branch_result(latent_runs)
        raw_best = select_best_branch_result(raw_runs)
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
        "source_init_checkpoint_path": str(source_init_checkpoint_path),
        "source_checkpoint_path": str(source.checkpoint_path),
        "source_run_dir": cfg.source.run_dir,
        "train_anchor": bool(cfg.source.train_anchor),
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


__all__ = ["prepare_config_from_source_checkpoint", "run_experiment", "_checkpoint_payload_json_view"]
