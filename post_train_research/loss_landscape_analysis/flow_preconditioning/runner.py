from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, config_hash, read_config, run_dir, torch_dtype, write_config
from .contexts import ConditionContext, LossFn, StartItem, build_contexts_for_seed, warped_grid_rows
from .flow_experiment import freeze_flow, make_random_flow, train_flow
from .optimization import Curve, curve_metrics, curve_rows, run_direct_curve, run_flow_curve
from .probe_geometry import (
    flow_coordinate_diagnostic_rows,
    geometry_rows_from_jacobians,
    probe_jacobians_for_flow,
    probe_jacobians_for_theta,
)
from .reporting import aggregate_results, build_interpretation_markdown, paired_deltas, save_figures
from .toy_mlp import seed_offset


FAMILIES = ("direct", "trained_flow", "random_flow")
OPTIMIZERS = ("sgd", "adam")
LOGGER = logging.getLogger("flow_preconditioning")


@dataclass(slots=True)
class ExperimentTables:
    results: pd.DataFrame
    curves: pd.DataFrame
    geometry: pd.DataFrame
    selected_lrs: pd.DataFrame
    aggregate: pd.DataFrame
    paired_deltas: pd.DataFrame
    trajectories: pd.DataFrame
    warped_grids: pd.DataFrame
    config: dict[str, Any]
    output_dir: Path
    figure_paths: dict[str, str]
    interpretation_markdown: str

def _method_name(family: str, optimizer: str) -> str:
    return f"{family}_{optimizer}"


def _lr_grid(cfg: ExperimentConfig, optimizer: str) -> tuple[float, ...]:
    if optimizer == "sgd":
        return tuple(float(v) for v in cfg.sgd_lrs)
    if optimizer == "adam":
        return tuple(float(v) for v in cfg.adam_lrs)
    raise ValueError(f"unknown optimizer {optimizer!r}")


def _curve_id(*parts: object) -> str:
    return "__".join(str(part).replace(".", "p") for part in parts)


def _stable_int(*parts: object, modulo: int = 20_000) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(modulo)


def _log_level(cfg: ExperimentConfig) -> int:
    return int(getattr(logging, str(cfg.log_level).strip().upper(), logging.INFO))


def _configure_progress_logging(output_dir: Path, cfg: ExperimentConfig) -> None:
    if not bool(cfg.progress_log_enabled):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(_log_level(cfg))
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        if getattr(handler, "_flow_preconditioning_managed", False):
            LOGGER.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler._flow_preconditioning_managed = True  # type: ignore[attr-defined]
    file_handler = logging.FileHandler(output_dir / "progress.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._flow_preconditioning_managed = True  # type: ignore[attr-defined]
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def _context_label(ctx: ConditionContext) -> str:
    return (
        f"[{ctx.experiment} task={ctx.task} {ctx.condition_name}={ctx.condition_value} "
        f"seed={ctx.seed}]"
    )


def _run_method_curve(
    *,
    family: str,
    optimizer: str,
    lr: float,
    item: StartItem,
    steps: int,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    trained_flow: torch.nn.Module,
    random_flow: torch.nn.Module,
    store_path: bool,
) -> Curve:
    if family == "direct":
        return run_direct_curve(
            train_loss_fn=train_loss_fn,
            test_loss_fn=test_loss_fn,
            theta0=item.start,
            optimizer_name=optimizer,
            lr=float(lr),
            steps=int(steps),
            store_path=bool(store_path),
        )
    flow = trained_flow if family == "trained_flow" else random_flow
    return run_flow_curve(
        train_loss_fn=train_loss_fn,
        test_loss_fn=test_loss_fn,
        flow=flow,
        theta0=item.start,
        optimizer_name=optimizer,
        lr=float(lr),
        steps=int(steps),
        store_path=bool(store_path),
    )


def _train_flows_and_geometry(
    ctx: ConditionContext,
    cfg: ExperimentConfig,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, object]], list[dict[str, object]]]:
    seed_for_flow = seed_offset(ctx.seed, 50_000 + _stable_int(ctx.experiment, ctx.task, ctx.condition_name, ctx.condition_value))
    label = _context_label(ctx)
    LOGGER.info("%s context flow+geometry start", label)
    flow_result = train_flow(
        probe=ctx.probe,
        theta_samples=ctx.flow_pool,
        cfg=cfg,
        steps=int(ctx.flow_steps),
        seed=int(seed_for_flow),
        log_label=label,
    )
    trained_flow = freeze_flow(flow_result.flow)
    LOGGER.info("%s random_flow build start", label)
    random_flow = freeze_flow(
        make_random_flow(
            ctx.dim,
            cfg,
            device=ctx.flow_pool.device,
            dtype=ctx.flow_pool.dtype,
            seed=int(seed_for_flow) + 9001,
        )
    )

    geometry_rows: list[dict[str, object]] = []
    LOGGER.info("%s geometry original start heldout_samples=%d", label, int(ctx.heldout_pool.shape[0]))
    theta_jac = probe_jacobians_for_theta(ctx.probe, ctx.heldout_pool, create_graph=False)
    geometry_rows.extend(
        geometry_rows_from_jacobians(
            theta_jac,
            experiment=ctx.experiment,
            task=ctx.task,
            condition_name=ctx.condition_name,
            condition_value=ctx.condition_value,
            coordinate="original",
            seed=ctx.seed,
            rho=ctx.numeric_condition,
        )
    )
    LOGGER.info("%s geometry trained_flow start", label)
    trained_jac, _ = probe_jacobians_for_flow(ctx.probe, trained_flow, ctx.heldout_pool, create_graph=False)
    geometry_rows.extend(
        geometry_rows_from_jacobians(
            trained_jac,
            experiment=ctx.experiment,
            task=ctx.task,
            condition_name=ctx.condition_name,
            condition_value=ctx.condition_value,
            coordinate="trained_flow",
            seed=ctx.seed,
            rho=ctx.numeric_condition,
        )
    )
    LOGGER.info("%s geometry random_flow start", label)
    random_jac, _ = probe_jacobians_for_flow(ctx.probe, random_flow, ctx.heldout_pool, create_graph=False)
    LOGGER.info("%s coordinate_map trained_flow diagnostics start", label)
    geometry_rows.extend(
        geometry_rows_from_jacobians(
            random_jac,
            experiment=ctx.experiment,
            task=ctx.task,
            condition_name=ctx.condition_name,
            condition_value=ctx.condition_value,
            coordinate="random_flow",
            seed=ctx.seed,
            rho=ctx.numeric_condition,
        )
    )
    LOGGER.info("%s coordinate_map random_flow diagnostics start", label)
    geometry_rows.extend(
        flow_coordinate_diagnostic_rows(
            trained_flow,
            ctx.heldout_pool,
            experiment=ctx.experiment,
            task=ctx.task,
            condition_name=ctx.condition_name,
            condition_value=ctx.condition_value,
            coordinate="trained_flow",
            seed=ctx.seed,
            rho=ctx.numeric_condition,
        )
    )
    geometry_rows.extend(
        flow_coordinate_diagnostic_rows(
            random_flow,
            ctx.heldout_pool,
            experiment=ctx.experiment,
            task=ctx.task,
            condition_name=ctx.condition_name,
            condition_value=ctx.condition_value,
            coordinate="random_flow",
            seed=ctx.seed,
            rho=ctx.numeric_condition,
        )
    )
    flow_history = [
        {
            "experiment": ctx.experiment,
            "task": ctx.task,
            "condition_name": ctx.condition_name,
            "condition_value": ctx.condition_value,
            "seed": int(ctx.seed),
            **row,
        }
        for row in flow_result.history
    ]
    LOGGER.info("%s context flow+geometry done geometry_rows=%d", label, len(geometry_rows))
    return trained_flow, random_flow, geometry_rows, flow_history


def _raw_records_to_tables(
    records: list[dict[str, Any]],
    *,
    cfg: ExperimentConfig,
    success_threshold_by_key: dict[tuple[str, str, str, str, int, int], float] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    results_rows: list[dict[str, object]] = []
    curve_table_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    ref: dict[tuple[str, str, str, str, str, int, int, int], float] = {}
    for record in records:
        budget = int(record["budget"])
        key = (
            str(record["experiment"]),
            str(record["task"]),
            str(record["condition_name"]),
            str(record["condition_value"]),
            str(record["split"]),
            int(record["seed"]),
            int(budget),
            int(record["start_index"]),
        )
        best = float(np.min(record["curve"].train_loss[: budget + 1]))
        ref[key] = min(ref.get(key, best), best)

    for record in records:
        curve: Curve = record["curve"]
        budget = int(record["budget"])
        new_curve_rows = curve_rows(
                curve,
                curve_id=str(record["curve_id"]),
                experiment=str(record["experiment"]),
                split=str(record["split"]),
                seed=int(record["seed"]),
                rho=float(record["numeric_condition"]),
                method=str(record["method"]),
                optimizer=str(record["optimizer"]),
                lr=float(record["lr"]),
                start_index=int(record["start_index"]),
                budget=budget,
            )
        for curve_row in new_curve_rows:
            curve_row["task"] = str(record["task"])
            curve_row["condition_name"] = str(record["condition_name"])
            curve_row["condition_value"] = str(record["condition_value"])
            curve_row["numeric_condition"] = float(record["numeric_condition"])
        curve_table_rows.extend(new_curve_rows)
        if curve.path is not None:
            for step, values in enumerate(curve.path):
                trajectory_rows.append(
                    {
                        "curve_id": str(record["curve_id"]),
                        "experiment": str(record["experiment"]),
                        "task": str(record["task"]),
                        "condition_name": str(record["condition_name"]),
                        "condition_value": str(record["condition_value"]),
                        "seed": int(record["seed"]),
                        "budget": budget,
                        "method": str(record["method"]),
                        "optimizer": str(record["optimizer"]),
                        "start_index": int(record["start_index"]),
                        "step": int(step),
                        "z_json": json.dumps([float(v) for v in values.tolist()]),
                    }
                )
        ref_key = (
            str(record["experiment"]),
            str(record["task"]),
            str(record["condition_name"]),
            str(record["condition_value"]),
            str(record["split"]),
            int(record["seed"]),
            budget,
            int(record["start_index"]),
        )
        threshold = None
        if success_threshold_by_key is not None:
            threshold = success_threshold_by_key.get(
                (
                    str(record["experiment"]),
                    str(record["task"]),
                    str(record["condition_name"]),
                    str(record["condition_value"]),
                    int(record["seed"]),
                    budget,
                )
            )
        metrics = curve_metrics(
            curve,
            budget=budget,
            l_ref=ref[ref_key],
            eps=float(cfg.aulc_eps),
            success_threshold=threshold,
        )
        row = {
            "curve_id": str(record["curve_id"]),
            "experiment": str(record["experiment"]),
            "task": str(record["task"]),
            "condition_name": str(record["condition_name"]),
            "condition_value": str(record["condition_value"]),
            "numeric_condition": float(record["numeric_condition"]),
            "split": str(record["split"]),
            "candidate": bool(record["candidate"]),
            "seed": int(record["seed"]),
            "budget": budget,
            "family": str(record["family"]),
            "method": str(record["method"]),
            "optimizer": str(record["optimizer"]),
            "lr": float(record["lr"]),
            "start_index": int(record["start_index"]),
            "final_theta_json": (
                json.dumps([float(v) for v in curve.final_theta.tolist()])
                if curve.final_theta is not None and str(record["split"]) == "eval"
                else ""
            ),
        }
        row.update(metrics)
        results_rows.append(row)
    return results_rows, curve_table_rows, trajectory_rows


def _select_lrs(tuning_results: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "experiment",
        "task",
        "condition_name",
        "condition_value",
        "seed",
        "budget",
        "family",
        "method",
        "optimizer",
        "lr",
    ]
    medians = tuning_results.groupby(group_cols, dropna=False)["aulc"].median().reset_index(name="median_tune_aulc")
    select_cols = group_cols[:-1]
    rows: list[dict[str, object]] = []
    for key, frame in medians.groupby(select_cols, dropna=False):
        chosen = frame.sort_values(["median_tune_aulc", "lr"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "experiment": key[0],
                "task": key[1],
                "condition_name": key[2],
                "condition_value": key[3],
                "seed": int(key[4]),
                "budget": int(key[5]),
                "family": key[6],
                "method": key[7],
                "optimizer": key[8],
                "selected_lr": float(chosen["lr"]),
                "median_tune_aulc": float(chosen["median_tune_aulc"]),
            }
        )
    return pd.DataFrame(rows)


def _success_thresholds(tuning_results: pd.DataFrame, selected_lrs: pd.DataFrame) -> dict[tuple[str, str, str, str, int, int], float]:
    thresholds: dict[tuple[str, str, str, str, int, int], float] = {}
    for selected in selected_lrs.itertuples(index=False):
        if selected.method != "direct_adam":
            continue
        frame = tuning_results[
            (tuning_results["experiment"] == selected.experiment)
            & (tuning_results["task"] == selected.task)
            & (tuning_results["condition_name"] == selected.condition_name)
            & (tuning_results["condition_value"] == selected.condition_value)
            & (tuning_results["seed"] == int(selected.seed))
            & (tuning_results["budget"] == int(selected.budget))
            & (tuning_results["method"] == "direct_adam")
            & (tuning_results["lr"] == float(selected.selected_lr))
        ]
        if len(frame):
            thresholds[
                (
                    str(selected.experiment),
                    str(selected.task),
                    str(selected.condition_name),
                    str(selected.condition_value),
                    int(selected.seed),
                    int(selected.budget),
                )
            ] = float(np.percentile(frame["final_train_loss"].to_numpy(dtype=np.float64), 10.0))
    return thresholds


def _make_tuning_records(
    ctx: ConditionContext,
    *,
    trained_flow: torch.nn.Module,
    random_flow: torch.nn.Module,
    cfg: ExperimentConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    label = _context_label(ctx)
    LOGGER.info(
        "%s tuning start starts=%d budgets=%s sgd_lrs=%s adam_lrs=%s",
        label,
        len(ctx.tune_items),
        tuple(int(v) for v in cfg.budgets),
        tuple(float(v) for v in cfg.sgd_lrs),
        tuple(float(v) for v in cfg.adam_lrs),
    )
    for budget in cfg.budgets:
        budget_started = time.time()
        budget_curve_count = 0
        for start_index, item in enumerate(ctx.tune_items):
            if start_index == 0 or start_index + 1 == len(ctx.tune_items):
                LOGGER.info("%s tuning budget=%d start=%d/%d", label, int(budget), start_index + 1, len(ctx.tune_items))
            train_loss = ctx.train_loss_factory(item)
            test_loss = ctx.test_loss_factory(item)
            for family in FAMILIES:
                for optimizer in OPTIMIZERS:
                    for lr in _lr_grid(cfg, optimizer):
                        method = _method_name(family, optimizer)
                        curve = _run_method_curve(
                            family=family,
                            optimizer=optimizer,
                            lr=float(lr),
                            item=item,
                            steps=int(budget),
                            train_loss_fn=train_loss,
                            test_loss_fn=test_loss,
                            trained_flow=trained_flow,
                            random_flow=random_flow,
                            store_path=False,
                        )
                        records.append(
                            {
                                "curve_id": _curve_id(ctx.experiment, ctx.task, "tune", ctx.seed, ctx.condition_value, budget, method, lr, start_index),
                                "curve": curve,
                                "experiment": ctx.experiment,
                                "task": ctx.task,
                                "condition_name": ctx.condition_name,
                                "condition_value": str(ctx.condition_value),
                                "numeric_condition": ctx.numeric_condition,
                                "split": "tune",
                                "candidate": True,
                                "seed": int(ctx.seed),
                                "budget": int(budget),
                                "family": family,
                                "method": method,
                                "optimizer": optimizer,
                                "lr": float(lr),
                                "start_index": int(start_index),
                            }
                        )
                        budget_curve_count += 1
        LOGGER.info(
            "%s tuning budget=%d done curves=%d elapsed_s=%.1f",
            label,
            int(budget),
            budget_curve_count,
            time.time() - budget_started,
        )
    LOGGER.info("%s tuning done curves=%d", label, len(records))
    return records


def _make_eval_records(
    ctx: ConditionContext,
    *,
    trained_flow: torch.nn.Module,
    random_flow: torch.nn.Module,
    selected_lrs: pd.DataFrame,
    cfg: ExperimentConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    label = _context_label(ctx)
    selected = selected_lrs[
        (selected_lrs["experiment"] == ctx.experiment)
        & (selected_lrs["task"] == ctx.task)
        & (selected_lrs["condition_name"] == ctx.condition_name)
        & (selected_lrs["condition_value"] == str(ctx.condition_value))
        & (selected_lrs["seed"] == int(ctx.seed))
    ]
    LOGGER.info("%s eval start selected_lr_rows=%d eval_starts=%d", label, len(selected), len(ctx.eval_items))
    for selected_row in selected.itertuples(index=False):
        row_started = time.time()
        LOGGER.info(
            "%s eval method=%s budget=%d lr=%g starts=%d",
            label,
            selected_row.method,
            int(selected_row.budget),
            float(selected_row.selected_lr),
            len(ctx.eval_items),
        )
        for start_index, item in enumerate(ctx.eval_items):
            if start_index == 0 or start_index + 1 == len(ctx.eval_items):
                LOGGER.info(
                    "%s eval method=%s budget=%d start=%d/%d",
                    label,
                    selected_row.method,
                    int(selected_row.budget),
                    start_index + 1,
                    len(ctx.eval_items),
                )
            train_loss = ctx.train_loss_factory(item)
            test_loss = ctx.test_loss_factory(item)
            store_path = bool(ctx.dim == 2 and int(start_index) < 3 and int(selected_row.budget) == int(max(cfg.budgets)))
            curve = _run_method_curve(
                family=str(selected_row.family),
                optimizer=str(selected_row.optimizer),
                lr=float(selected_row.selected_lr),
                item=item,
                steps=int(selected_row.budget),
                train_loss_fn=train_loss,
                test_loss_fn=test_loss,
                trained_flow=trained_flow,
                random_flow=random_flow,
                store_path=store_path,
            )
            records.append(
                {
                    "curve_id": _curve_id(ctx.experiment, ctx.task, "eval", ctx.seed, ctx.condition_value, selected_row.budget, selected_row.method, start_index),
                    "curve": curve,
                    "experiment": ctx.experiment,
                    "task": ctx.task,
                    "condition_name": ctx.condition_name,
                    "condition_value": str(ctx.condition_value),
                    "numeric_condition": ctx.numeric_condition,
                    "split": "eval",
                    "candidate": False,
                    "seed": int(ctx.seed),
                    "budget": int(selected_row.budget),
                    "family": str(selected_row.family),
                    "method": str(selected_row.method),
                    "optimizer": str(selected_row.optimizer),
                    "lr": float(selected_row.selected_lr),
                    "start_index": int(start_index),
                }
            )
        LOGGER.info(
            "%s eval method=%s budget=%d done elapsed_s=%.1f",
            label,
            selected_row.method,
            int(selected_row.budget),
            time.time() - row_started,
        )
    LOGGER.info("%s eval done curves=%d", label, len(records))
    return records


def _run_context(ctx: ConditionContext, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label = _context_label(ctx)
    started = time.time()
    LOGGER.info(
        "%s context start dim=%d flow_samples=%d heldout_samples=%d tune_starts=%d eval_starts=%d",
        label,
        int(ctx.dim),
        int(ctx.flow_pool.shape[0]),
        int(ctx.heldout_pool.shape[0]),
        len(ctx.tune_items),
        len(ctx.eval_items),
    )
    trained_flow, random_flow, geometry_rows, flow_history = _train_flows_and_geometry(ctx, cfg)
    grid_rows = warped_grid_rows(ctx, trained_flow, random_flow, cfg)
    if grid_rows:
        LOGGER.info("%s warped_grid rows=%d", label, len(grid_rows))
    tuning_records = _make_tuning_records(ctx, trained_flow=trained_flow, random_flow=random_flow, cfg=cfg)
    tuning_results, tuning_curves, _ = _raw_records_to_tables(tuning_records, cfg=cfg, success_threshold_by_key=None)
    tuning_df = pd.DataFrame(tuning_results)
    selected_lrs = _select_lrs(tuning_df)
    LOGGER.info("%s selected_lrs rows=%d", label, len(selected_lrs))
    thresholds = _success_thresholds(tuning_df, selected_lrs)
    LOGGER.info("%s success_thresholds rows=%d", label, len(thresholds))
    eval_records = _make_eval_records(ctx, trained_flow=trained_flow, random_flow=random_flow, selected_lrs=selected_lrs, cfg=cfg)
    eval_results, eval_curves, trajectories = _raw_records_to_tables(eval_records, cfg=cfg, success_threshold_by_key=thresholds)
    LOGGER.info(
        "%s context done result_rows=%d curve_rows=%d geometry_rows=%d elapsed_s=%.1f",
        label,
        len(tuning_results) + len(eval_results),
        len(tuning_curves) + len(eval_curves),
        len(geometry_rows),
        time.time() - started,
    )
    return (
        pd.DataFrame(tuning_results + eval_results),
        pd.DataFrame(tuning_curves + eval_curves),
        pd.DataFrame(geometry_rows),
        selected_lrs,
        pd.DataFrame(trajectories),
        pd.DataFrame(grid_rows),
        pd.DataFrame(flow_history),
    )


def _write_tables(output_dir: Path, cfg: ExperimentConfig, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame = frame.copy()
        for text_col in ("condition_value", "task", "condition_name", "experiment"):
            if text_col in frame.columns:
                frame[text_col] = frame[text_col].astype(str)
        if bool(cfg.write_parquet):
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        if bool(cfg.write_csv) and name in {"results", "selected_lrs", "aggregate", "paired_deltas", "flow_training"}:
            frame.to_csv(output_dir / f"{name}.csv", index=False)


def _load_tables(output_dir: Path) -> ExperimentTables:
    config_payload = read_config(output_dir / "config.json") or {}
    results = pd.read_parquet(output_dir / "results.parquet")
    curves = pd.read_parquet(output_dir / "curves.parquet")
    geometry = pd.read_parquet(output_dir / "geometry.parquet")
    selected_lrs = pd.read_parquet(output_dir / "selected_lrs.parquet")
    aggregate = pd.read_parquet(output_dir / "aggregate.parquet")
    deltas = pd.read_parquet(output_dir / "paired_deltas.parquet")
    trajectories = pd.read_parquet(output_dir / "trajectories.parquet") if (output_dir / "trajectories.parquet").is_file() else pd.DataFrame()
    warped_grids = pd.read_parquet(output_dir / "warped_grids.parquet") if (output_dir / "warped_grids.parquet").is_file() else pd.DataFrame()
    figures_path = output_dir / "figures.json"
    figure_paths = json.loads(figures_path.read_text(encoding="utf-8")) if figures_path.is_file() else {}
    interpretation_path = output_dir / "interpretation.md"
    interpretation = interpretation_path.read_text(encoding="utf-8") if interpretation_path.is_file() else ""
    return ExperimentTables(results, curves, geometry, selected_lrs, aggregate, deltas, trajectories, warped_grids, config_payload, output_dir, figure_paths, interpretation)


def _cache_is_valid(output_dir: Path, cfg: ExperimentConfig) -> bool:
    required = [
        "config.json",
        "results.parquet",
        "curves.parquet",
        "geometry.parquet",
        "selected_lrs.parquet",
        "aggregate.parquet",
        "paired_deltas.parquet",
    ]
    if not all((output_dir / name).is_file() for name in required):
        return False
    payload = read_config(output_dir / "config.json")
    return bool(payload and payload.get("config_hash") == config_hash(cfg))


def run_all_experiments(cfg: ExperimentConfig) -> ExperimentTables:
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    output_dir = run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_progress_logging(output_dir, cfg)
    run_started = time.time()
    LOGGER.info(
        "run start output_dir=%s device=%s seeds=%s budgets=%s",
        output_dir,
        cfg.device,
        tuple(int(v) for v in cfg.seeds),
        tuple(int(v) for v in cfg.budgets),
    )

    contexts = [
        ctx
        for seed in cfg.seeds
        for ctx in build_contexts_for_seed(cfg, seed=int(seed), device=device, dtype=dtype)
    ]
    LOGGER.info("planned contexts=%d", len(contexts))
    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for context_index, ctx in enumerate(contexts, start=1):
        context_started = time.time()
        LOGGER.info("context %d/%d begin %s", context_index, len(contexts), _context_label(ctx))
        parts.append(_run_context(ctx, cfg))
        elapsed = time.time() - run_started
        per_context = elapsed / float(context_index)
        remaining = max(0, len(contexts) - context_index) * per_context
        LOGGER.info(
            "context %d/%d done context_elapsed_s=%.1f run_elapsed_s=%.1f avg_context_s=%.1f eta_s=%.1f eta_h=%.2f",
            context_index,
            len(contexts),
            time.time() - context_started,
            elapsed,
            per_context,
            remaining,
            remaining / 3600.0,
        )

    results = pd.concat([part[0] for part in parts], ignore_index=True) if parts else pd.DataFrame()
    curves = pd.concat([part[1] for part in parts], ignore_index=True) if parts else pd.DataFrame()
    geometry = pd.concat([part[2] for part in parts], ignore_index=True) if parts else pd.DataFrame()
    selected_lrs = pd.concat([part[3] for part in parts], ignore_index=True) if parts else pd.DataFrame()
    trajectories = pd.concat([part[4] for part in parts], ignore_index=True) if parts else pd.DataFrame()
    warped_grids = pd.concat([part[5] for part in parts], ignore_index=True) if parts else pd.DataFrame()
    flow_training = pd.concat([part[6] for part in parts], ignore_index=True) if parts else pd.DataFrame()
    aggregate = aggregate_results(results, cfg)
    deltas = paired_deltas(results)
    interpretation = build_interpretation_markdown(results=results, geometry=geometry, cfg=cfg)
    LOGGER.info("saving figures=%s", bool(cfg.save_figures))
    figures = save_figures(
        results=results,
        curves=curves,
        geometry=geometry,
        output_dir=output_dir,
        cfg=cfg,
        trajectories=trajectories,
        warped_grids=warped_grids,
    ) if cfg.save_figures else {}
    payload = write_config(output_dir / "config.json", cfg)
    LOGGER.info("writing tables output_dir=%s", output_dir)
    _write_tables(
        output_dir,
        cfg,
        {
            "results": results,
            "curves": curves,
            "geometry": geometry,
            "selected_lrs": selected_lrs,
            "aggregate": aggregate,
            "paired_deltas": deltas,
            "trajectories": trajectories,
            "warped_grids": warped_grids,
            "flow_training": flow_training,
        },
    )
    (output_dir / "figures.json").write_text(json.dumps(figures, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "interpretation.md").write_text(interpretation, encoding="utf-8")
    LOGGER.info(
        "run done elapsed_s=%.1f elapsed_h=%.2f results=%d curves=%d geometry=%d",
        time.time() - run_started,
        (time.time() - run_started) / 3600.0,
        len(results),
        len(curves),
        len(geometry),
    )
    return ExperimentTables(results, curves, geometry, selected_lrs, aggregate, deltas, trajectories, warped_grids, payload, output_dir, figures, interpretation)


def run_or_load(cfg: ExperimentConfig) -> ExperimentTables:
    output_dir = run_dir(cfg)
    _configure_progress_logging(output_dir, cfg)
    if bool(cfg.cache_first) and not bool(cfg.force_rerun) and _cache_is_valid(output_dir, cfg):
        LOGGER.info("cache hit output_dir=%s", output_dir)
        return _load_tables(output_dir)
    LOGGER.info("cache miss or force rerun output_dir=%s", output_dir)
    return run_all_experiments(cfg)
