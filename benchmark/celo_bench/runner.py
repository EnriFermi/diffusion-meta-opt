from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .adapters import AdamOptimizerAdapter, LearnedOptimizationTaskAdapter, build_optimizer_adapter
from .artifacts import reserve_run_dir, resolve_path, slugify, write_artifact_layout, write_json_file, write_yaml_file
from .config import CeloBenchConfig, config_hash, config_to_dict
from .interfaces import BenchTables
from .logging import configure_logger
from .metrics import extract_curve, final_loss_from_curve, score_curve_pair, summarize_scores


def run_or_load(cfg: CeloBenchConfig) -> BenchTables:
    run_id, output_dir = _resolve_output_dir(cfg)
    if (
        cfg.benchmark.cache_first
        and not cfg.benchmark.force_rerun
        and _cache_valid(output_dir, cfg)
    ):
        summary = _read_json(output_dir / "summary.json")
        return BenchTables(
            output_dir=output_dir,
            summary=summary,
            score_rows=_read_csv(output_dir / "scores.csv"),
            curve_rows=_read_csv(output_dir / "curves.csv"),
            selected_adam_rows=_read_csv(output_dir / "selected_adam.csv"),
        )
    return run_all(cfg, run_id=run_id, output_dir=output_dir)


def run_all(cfg: CeloBenchConfig, *, run_id: str | None = None, output_dir: Path | None = None) -> BenchTables:
    run_id, output_dir = _resolve_output_dir(cfg) if output_dir is None else (run_id or output_dir.name, output_dir)
    logs_dir = output_dir / "logs"
    raw_dir = output_dir / "raw"
    logger = configure_logger(output_dir, level=cfg.benchmark.log_level)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config_to_dict(cfg)
    cfg_digest = config_hash(cfg)
    logger.info("Celo benchmark starting: run_id=%s output_dir=%s config_hash=%s", run_id, output_dir, cfg_digest)
    logger.info(
        "Resolved runtime: device=%s dtype=%s seed=%s xla_preallocate=%s",
        cfg.runtime.device,
        cfg.runtime.dtype,
        cfg.runtime.seed,
        cfg.runtime.xla_preallocate,
    )
    logger.info(
        "Evaluation protocol: tasks=%s seeds=%s steps=%s eval_every=%s eval_batches=%s last_eval_batches=%s score_split=%s",
        len(cfg.task_names),
        list(cfg.evaluation.seeds),
        cfg.evaluation.steps,
        cfg.evaluation.eval_every,
        cfg.evaluation.eval_batches,
        cfg.evaluation.last_eval_batches,
        cfg.evaluation.score_split,
    )
    write_json_file(output_dir / "config_resolved.json", cfg_dict)
    write_yaml_file(output_dir / "config_resolved.yaml", cfg_dict)
    write_json_file(output_dir / "config_meta.json", {"config_hash": cfg_digest, "config": cfg_dict})

    score_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    selected_adam_rows: list[dict[str, Any]] = []
    final_scores: list[float] = []
    speedup_scores: list[float] = []
    planned_jobs = _planned_jobs(cfg)

    if cfg.benchmark.dry_run:
        logger.info("Dry run enabled; writing planned jobs without executing tasks")
        summary = {
            "run_id": run_id,
            "config_hash": cfg_digest,
            "dry_run": True,
            "planned_jobs": planned_jobs,
            "ok": True,
        }
        _write_outputs(output_dir, summary, score_rows, curve_rows, selected_adam_rows)
        _write_layout(output_dir, run_id, cfg_digest)
        return BenchTables(output_dir, summary, score_rows, curve_rows, selected_adam_rows)

    progress_items = list(_task_seed_pairs(cfg))
    for task_name, seed in _progress(progress_items, enabled=cfg.benchmark.show_progress, desc="celo_bench"):
        logger.info("Stage task run starting: task=%s seed=%s", task_name, seed)
        best_adam_metrics: Mapping[str, Any] | None = None
        best_adam_lr: float | None = None
        best_adam_final = float("inf")
        if cfg.adam_reference.enabled:
            logger.info("Adam reference sweep: task=%s seed=%s lrs=%s", task_name, seed, list(cfg.adam_reference.lrs))
            for lr in cfg.adam_reference.lrs:
                adapter = AdamOptimizerAdapter(float(lr))
                metrics = _run_curve(cfg, task_name, adapter.name, adapter, seed, raw_dir, logger, metadata={"lr": lr})
                curve_rows.extend(_curve_rows(task_name, seed, adapter.name, metrics, cfg.evaluation.score_split, {"lr": lr}))
                final_loss = final_loss_from_curve(metrics[cfg.evaluation.score_split], alpha=cfg.evaluation.ema_alpha)
                if final_loss < best_adam_final:
                    best_adam_final = final_loss
                    best_adam_lr = float(lr)
                    best_adam_metrics = metrics
            selected_adam_rows.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "lr": best_adam_lr,
                    "final_loss": best_adam_final,
                }
            )

        if best_adam_metrics is None:
            logger.warning("No Adam reference available for task=%s seed=%s; method scoring will be skipped", task_name, seed)
            continue

        for method_spec in cfg.methods:
            if not method_spec.enabled:
                continue
            adapter = build_optimizer_adapter(method_spec)
            logger.info("Method eval starting: task=%s seed=%s method=%s", task_name, seed, method_spec.name)
            metrics = _run_curve(cfg, task_name, method_spec.name, adapter, seed, raw_dir, logger)
            curve_rows.extend(_curve_rows(task_name, seed, method_spec.name, metrics, cfg.evaluation.score_split, {}))
            score = score_curve_pair(
                xs=metrics["eval/xs"],
                best_adam_curve=best_adam_metrics[cfg.evaluation.score_split],
                optimizer_curve=metrics[cfg.evaluation.score_split],
                horizon=cfg.evaluation.steps,
                alpha=cfg.evaluation.ema_alpha,
            )
            row = {"task": task_name, "seed": seed, "method": method_spec.name, **score}
            score_rows.append(row)
            final_scores.append(float(score["final_loss_score"]))
            speedup_scores.append(float(score["speedup_score"]))

    final_summary = summarize_scores(final_scores)
    speedup_summary = summarize_scores(speedup_scores)
    summary = {
        "run_id": run_id,
        "config_hash": cfg_digest,
        "dry_run": False,
        "ok": True,
        "output_dir": str(output_dir),
        "tasks": list(cfg.task_names),
        "seeds": list(cfg.evaluation.seeds),
        "steps": cfg.evaluation.steps,
        "score_split": cfg.evaluation.score_split,
        "planned_jobs": planned_jobs,
        "final_loss_score": final_summary,
        "speedup_score": speedup_summary,
        "files": {
            "summary": str(output_dir / "summary.json"),
            "scores": str(output_dir / "scores.csv"),
            "curves": str(output_dir / "curves.csv"),
            "selected_adam": str(output_dir / "selected_adam.csv"),
        },
    }
    logger.info("Aggregation complete: final_loss_iqm=%.6g speedup_iqm=%.6g", final_summary["iqm"], speedup_summary["iqm"])
    _write_outputs(output_dir, summary, score_rows, curve_rows, selected_adam_rows)
    _write_layout(output_dir, run_id, cfg_digest)
    logger.info("Celo benchmark finished: summary=%s", output_dir / "summary.json")
    return BenchTables(output_dir, summary, score_rows, curve_rows, selected_adam_rows)


def evaluate_task_optimizer(
    *,
    task_name: str,
    optimizer_adapter: Any,
    seed: int,
    cfg: CeloBenchConfig,
) -> Mapping[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", cfg.runtime.xla_preallocate)
    if cfg.runtime.tensorflow_hide_gpus:
        _hide_tensorflow_gpus()
    import jax

    from .eval_training import single_task_training_curves

    task = LearnedOptimizationTaskAdapter(task_name).build()
    opt = optimizer_adapter.build(num_steps=cfg.evaluation.steps)
    key = jax.random.PRNGKey(seed)
    return single_task_training_curves(
        task=task,
        opt=opt,
        num_steps=cfg.evaluation.steps,
        key=key,
        eval_every=cfg.evaluation.eval_every,
        eval_batches=cfg.evaluation.eval_batches,
        last_eval_batches=cfg.evaluation.last_eval_batches,
        eval_task=None,
        device=None,
        metrics_every=cfg.evaluation.metrics_every,
        summary_writer=None,
    )


def _hide_tensorflow_gpus() -> None:
    """Keep TensorFlow/TFDS preprocessing off GPU, matching upstream Celo eval."""
    try:
        import tensorflow as tf
    except Exception:
        return
    try:
        tf.config.experimental.set_visible_devices([], "GPU")
    except RuntimeError:
        # TensorFlow was already initialized. The run can still continue if TF
        # does not touch GPU kernels; otherwise the user should set this earlier.
        return


def _run_curve(
    cfg: CeloBenchConfig,
    task_name: str,
    method_name: str,
    adapter: Any,
    seed: int,
    raw_dir: Path,
    logger: Any,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    method_dir = raw_dir / slugify(task_name) / f"seed_{seed}" / slugify(method_name)
    method_dir.mkdir(parents=True, exist_ok=True)
    output_path = method_dir / f"metrics_unroll{cfg.evaluation.steps}.npz"
    metrics = evaluate_task_optimizer(task_name=task_name, optimizer_adapter=adapter, seed=seed, cfg=cfg)
    np.savez(output_path, **metrics)
    manifest = {
        "task": task_name,
        "method": method_name,
        "seed": seed,
        "steps": cfg.evaluation.steps,
        "metrics_path": str(output_path),
        "metadata": dict(metadata or {}),
    }
    write_json_file(method_dir / "manifest.json", manifest)
    logger.info("Curve written: task=%s seed=%s method=%s path=%s", task_name, seed, method_name, output_path)
    return metrics


def _curve_rows(
    task_name: str,
    seed: int,
    method_name: str,
    metrics: Mapping[str, Any],
    score_split: str,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    xs = np.asarray(metrics.get("eval/xs", []), dtype=float).reshape(-1)
    curve = extract_curve(metrics, score_split)
    rows: list[dict[str, Any]] = []
    for idx, (step, loss) in enumerate(zip(xs, curve)):
        rows.append(
            {
                "task": task_name,
                "seed": seed,
                "method": method_name,
                "point_index": idx,
                "step": float(step),
                "loss": float(loss),
                **{str(key): value for key, value in metadata.items()},
            }
        )
    return rows


def _planned_jobs(cfg: CeloBenchConfig) -> dict[str, Any]:
    pairs = len(cfg.task_names) * len(cfg.evaluation.seeds)
    adam_jobs = pairs * len(cfg.adam_reference.lrs) if cfg.adam_reference.enabled else 0
    method_jobs = pairs * len([method for method in cfg.methods if method.enabled])
    return {
        "task_seed_pairs": pairs,
        "adam_reference_jobs": adam_jobs,
        "method_jobs": method_jobs,
        "total_curve_jobs": adam_jobs + method_jobs,
    }


def _task_seed_pairs(cfg: CeloBenchConfig) -> Iterable[tuple[str, int]]:
    for task_name in cfg.task_names:
        for seed in cfg.evaluation.seeds:
            yield task_name, int(seed)


def _progress(items: list[Any], *, enabled: bool, desc: str) -> Iterable[Any]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc)


def _resolve_output_dir(cfg: CeloBenchConfig) -> tuple[str, Path]:
    if cfg.benchmark.run_dir:
        output_dir = resolve_path(cfg.benchmark.run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir.name, output_dir
    root = resolve_path(cfg.benchmark.artifact_root)
    return reserve_run_dir(root, cfg.benchmark.run_label)


def _cache_valid(output_dir: Path, cfg: CeloBenchConfig) -> bool:
    meta_path = output_dir / "config_meta.json"
    summary_path = output_dir / "summary.json"
    if not meta_path.is_file() or not summary_path.is_file():
        return False
    try:
        meta = _read_json(meta_path)
    except Exception:
        return False
    if meta.get("config_hash") != config_hash(cfg):
        return False
    return all((output_dir / name).is_file() for name in ("scores.csv", "curves.csv", "selected_adam.csv"))


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    score_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    selected_adam_rows: list[dict[str, Any]],
) -> None:
    write_json_file(output_dir / "summary.json", summary)
    _write_csv(output_dir / "scores.csv", score_rows)
    _write_csv(output_dir / "curves.csv", curve_rows)
    _write_csv(output_dir / "selected_adam.csv", selected_adam_rows)


def _write_layout(output_dir: Path, run_id: str, cfg_digest: str) -> None:
    write_artifact_layout(
        output_dir,
        kind="benchmark.celo_bench",
        run_id=run_id,
        files={
            "config_json": output_dir / "config_resolved.json",
            "config_yaml": output_dir / "config_resolved.yaml",
            "config_meta": output_dir / "config_meta.json",
            "summary": output_dir / "summary.json",
            "scores": output_dir / "scores.csv",
            "curves": output_dir / "curves.csv",
            "selected_adam": output_dir / "selected_adam.csv",
            "log": output_dir / "run.log",
        },
        dirs={"logs": output_dir / "logs", "raw": output_dir / "raw"},
        metadata={"config_hash": cfg_digest},
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload
