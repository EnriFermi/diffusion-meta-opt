from __future__ import annotations

import json
import hashlib
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
    flow_result = train_flow(
        probe=ctx.probe,
        theta_samples=ctx.flow_pool,
        cfg=cfg,
        steps=int(ctx.flow_steps),
        seed=int(seed_for_flow),
    )
    trained_flow = freeze_flow(flow_result.flow)
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
    random_jac, _ = probe_jacobians_for_flow(ctx.probe, random_flow, ctx.heldout_pool, create_graph=False)
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
    for budget in cfg.budgets:
        for start_index, item in enumerate(ctx.tune_items):
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
    selected = selected_lrs[
        (selected_lrs["experiment"] == ctx.experiment)
        & (selected_lrs["task"] == ctx.task)
        & (selected_lrs["condition_name"] == ctx.condition_name)
        & (selected_lrs["condition_value"] == str(ctx.condition_value))
        & (selected_lrs["seed"] == int(ctx.seed))
    ]
    for selected_row in selected.itertuples(index=False):
        for start_index, item in enumerate(ctx.eval_items):
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
    return records


def _run_context(ctx: ConditionContext, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trained_flow, random_flow, geometry_rows, flow_history = _train_flows_and_geometry(ctx, cfg)
    grid_rows = warped_grid_rows(ctx, trained_flow, random_flow, cfg)
    tuning_records = _make_tuning_records(ctx, trained_flow=trained_flow, random_flow=random_flow, cfg=cfg)
    tuning_results, tuning_curves, _ = _raw_records_to_tables(tuning_records, cfg=cfg, success_threshold_by_key=None)
    tuning_df = pd.DataFrame(tuning_results)
    selected_lrs = _select_lrs(tuning_df)
    thresholds = _success_thresholds(tuning_df, selected_lrs)
    eval_records = _make_eval_records(ctx, trained_flow=trained_flow, random_flow=random_flow, selected_lrs=selected_lrs, cfg=cfg)
    eval_results, eval_curves, trajectories = _raw_records_to_tables(eval_records, cfg=cfg, success_threshold_by_key=thresholds)
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

    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for seed in cfg.seeds:
        for ctx in build_contexts_for_seed(cfg, seed=int(seed), device=device, dtype=dtype):
            parts.append(_run_context(ctx, cfg))

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
    return ExperimentTables(results, curves, geometry, selected_lrs, aggregate, deltas, trajectories, warped_grids, payload, output_dir, figures, interpretation)


def run_or_load(cfg: ExperimentConfig) -> ExperimentTables:
    output_dir = run_dir(cfg)
    if bool(cfg.cache_first) and not bool(cfg.force_rerun) and _cache_is_valid(output_dir, cfg):
        return _load_tables(output_dir)
    return run_all_experiments(cfg)
