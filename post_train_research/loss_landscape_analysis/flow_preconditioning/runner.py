from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, config_hash, read_config, run_dir, torch_dtype, write_config
from .flow_experiment import freeze_flow, make_random_flow, train_flow
from .optimization import Curve, curve_metrics, curve_rows, run_direct_curve, run_flow_curve
from .probe_geometry import (
    ResidualOnlyProbe,
    ResidualThetaProbe,
    fit_probe_scales,
    geometry_rows_from_jacobians,
    probe_jacobians_for_flow,
    probe_jacobians_for_theta,
)
from .reporting import aggregate_results, build_interpretation_markdown, paired_deltas, save_figures
from .toy_mlp import (
    MLPRegressionProblem,
    QuadraticProblem,
    collect_mlp_flow_pool,
    collect_mlp_heldout_geometry_pool,
    collect_quadratic_flow_pool,
    collect_quadratic_heldout_geometry_pool,
    seed_offset,
)


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


def _run_method_curve(
    *,
    family: str,
    optimizer: str,
    lr: float,
    theta0: torch.Tensor,
    steps: int,
    train_loss_fn,
    test_loss_fn,
    trained_flow: torch.nn.Module,
    random_flow: torch.nn.Module,
) -> Curve:
    if family == "direct":
        return run_direct_curve(
            train_loss_fn=train_loss_fn,
            test_loss_fn=test_loss_fn,
            theta0=theta0,
            optimizer_name=optimizer,
            lr=float(lr),
            steps=int(steps),
        )
    flow = trained_flow if family == "trained_flow" else random_flow
    return run_flow_curve(
        train_loss_fn=train_loss_fn,
        test_loss_fn=test_loss_fn,
        flow=flow,
        theta0=theta0,
        optimizer_name=optimizer,
        lr=float(lr),
        steps=int(steps),
    )


def _records_to_tables(
    records: list[dict[str, Any]],
    *,
    metric_budgets: Iterable[int],
    cfg: ExperimentConfig,
    success_threshold_by_seed: dict[tuple[str, int], float] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results_rows: list[dict[str, object]] = []
    curve_table_rows: list[dict[str, object]] = []
    ref: dict[tuple[str, str, int, float, int, int], float] = {}
    budgets = tuple(int(v) for v in metric_budgets)
    for record in records:
        record_budgets = tuple(int(v) for v in record.get("metric_budgets", budgets))
        for budget in record_budgets:
            key = (
                str(record["experiment"]),
                str(record["split"]),
                int(record["seed"]),
                float(record["rho"]),
                int(budget),
                int(record["start_index"]),
            )
            train = record["curve"].train_loss[: int(budget) + 1]
            value = float(np.min(train))
            ref[key] = min(ref.get(key, value), value)

    for record in records:
        curve = record["curve"]
        curve_table_rows.extend(
            curve_rows(
                curve,
                curve_id=str(record["curve_id"]),
                experiment=str(record["experiment"]),
                split=str(record["split"]),
                seed=int(record["seed"]),
                rho=float(record["rho"]),
                method=str(record["method"]),
                optimizer=str(record["optimizer"]),
                lr=float(record["lr"]),
                start_index=int(record["start_index"]),
                budget=int(record["curve_budget"]),
            )
        )
        record_budgets = tuple(int(v) for v in record.get("metric_budgets", budgets))
        for budget in record_budgets:
            ref_key = (
                str(record["experiment"]),
                str(record["split"]),
                int(record["seed"]),
                float(record["rho"]),
                int(budget),
                int(record["start_index"]),
            )
            threshold = None
            if success_threshold_by_seed is not None:
                threshold = success_threshold_by_seed.get((str(record["experiment"]), int(record["seed"])))
            metrics = curve_metrics(
                curve,
                budget=int(budget),
                l_ref=ref[ref_key],
                eps=float(cfg.aulc_eps),
                success_threshold=threshold,
            )
            row = {
                "curve_id": str(record["curve_id"]),
                "experiment": str(record["experiment"]),
                "split": str(record["split"]),
                "candidate": bool(record["candidate"]),
                "seed": int(record["seed"]),
                "rho": float(record["rho"]),
                "budget": int(budget),
                "family": str(record["family"]),
                "method": str(record["method"]),
                "optimizer": str(record["optimizer"]),
                "lr": float(record["lr"]),
                "start_index": int(record["start_index"]),
                "final_theta_json": (
                    json.dumps([float(v) for v in curve.final_theta.tolist()])
                    if curve.final_theta is not None
                    and str(record["split"]) == "eval"
                    and int(record["curve_budget"]) == int(budget)
                    else ""
                ),
            }
            row.update(metrics)
            results_rows.append(row)
    return results_rows, curve_table_rows


def _select_lrs(tuning_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["experiment", "seed", "rho", "budget", "family", "method", "optimizer", "lr"]
    medians = tuning_results.groupby(group_cols, dropna=False)["aulc"].median().reset_index(name="median_tune_aulc")
    select_cols = ["experiment", "seed", "rho", "budget", "family", "method", "optimizer"]
    for key, frame in medians.groupby(select_cols, dropna=False):
        chosen = frame.sort_values(["median_tune_aulc", "lr"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "experiment": key[0],
                "seed": int(key[1]),
                "rho": float(key[2]),
                "budget": int(key[3]),
                "family": key[4],
                "method": key[5],
                "optimizer": key[6],
                "selected_lr": float(chosen["lr"]),
                "median_tune_aulc": float(chosen["median_tune_aulc"]),
            }
        )
    return pd.DataFrame(rows)


def _success_thresholds(
    tuning_results: pd.DataFrame,
    selected_lrs: pd.DataFrame,
    cfg: ExperimentConfig,
) -> dict[tuple[str, int], float]:
    thresholds: dict[tuple[str, int], float] = {}
    max_budget = int(max(cfg.budgets))
    for (experiment, seed), frame in tuning_results.groupby(["experiment", "seed"], dropna=False):
        rho = float(cfg.main_rho) if str(experiment) == "mlp" else 0.0
        selected = selected_lrs[
            (selected_lrs["experiment"] == experiment)
            & (selected_lrs["seed"] == int(seed))
            & (selected_lrs["rho"] == rho)
            & (selected_lrs["budget"] == max_budget)
            & (selected_lrs["method"] == "direct_adam")
        ]
        if selected.empty:
            continue
        lr = float(selected.iloc[0]["selected_lr"])
        direct = frame[
            (frame["rho"] == rho)
            & (frame["budget"] == max_budget)
            & (frame["method"] == "direct_adam")
            & (frame["lr"] == lr)
        ]
        if len(direct):
            thresholds[(str(experiment), int(seed))] = float(np.percentile(direct["final_train_loss"].to_numpy(), 10.0))
    return thresholds


def _train_and_evaluate_flows(
    *,
    experiment: str,
    seed: int,
    rho: float,
    probe,
    theta_pool: torch.Tensor,
    heldout_pool: torch.Tensor,
    cfg: ExperimentConfig,
    flow_steps: int,
    random_seed: int,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, object]], list[dict[str, object]]]:
    flow_result = train_flow(
        probe=probe,
        theta_samples=theta_pool,
        cfg=cfg,
        steps=int(flow_steps),
        seed=int(random_seed),
    )
    trained_flow = freeze_flow(flow_result.flow)
    random_flow = freeze_flow(
        make_random_flow(
            int(theta_pool.shape[1]),
            cfg,
            device=theta_pool.device,
            dtype=theta_pool.dtype,
            seed=int(random_seed) + 9001,
        )
    )

    geometry_rows: list[dict[str, object]] = []
    theta_jac = probe_jacobians_for_theta(probe, heldout_pool, create_graph=False)
    geometry_rows.extend(
        geometry_rows_from_jacobians(theta_jac, experiment=experiment, coordinate="theta", seed=seed, rho=rho)
    )
    trained_jac, _ = probe_jacobians_for_flow(probe, trained_flow, heldout_pool, create_graph=False)
    geometry_rows.extend(
        geometry_rows_from_jacobians(
            trained_jac,
            experiment=experiment,
            coordinate="trained_flow",
            seed=seed,
            rho=rho,
        )
    )
    random_jac, _ = probe_jacobians_for_flow(probe, random_flow, heldout_pool, create_graph=False)
    geometry_rows.extend(
        geometry_rows_from_jacobians(random_jac, experiment=experiment, coordinate="random_flow", seed=seed, rho=rho)
    )
    flow_history = [
        {
            "experiment": experiment,
            "seed": int(seed),
            "rho": float(rho),
            **row,
        }
        for row in flow_result.history
    ]
    return trained_flow, random_flow, geometry_rows, flow_history


def _make_tuning_records(
    *,
    experiment: str,
    seed: int,
    rho: float,
    tune_starts: torch.Tensor,
    max_budget: int,
    train_loss_fn,
    test_loss_fn,
    trained_flow: torch.nn.Module,
    random_flow: torch.nn.Module,
    cfg: ExperimentConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start_index, theta0 in enumerate(tune_starts):
        for family in FAMILIES:
            for optimizer in OPTIMIZERS:
                for lr in _lr_grid(cfg, optimizer):
                    method = _method_name(family, optimizer)
                    curve = _run_method_curve(
                        family=family,
                        optimizer=optimizer,
                        lr=float(lr),
                        theta0=theta0,
                        steps=int(max_budget),
                        train_loss_fn=train_loss_fn,
                        test_loss_fn=test_loss_fn,
                        trained_flow=trained_flow,
                        random_flow=random_flow,
                    )
                    records.append(
                        {
                            "curve_id": _curve_id(experiment, "tune", seed, rho, method, lr, start_index),
                            "curve": curve,
                            "experiment": experiment,
                            "split": "tune",
                            "candidate": True,
                            "seed": int(seed),
                            "rho": float(rho),
                            "curve_budget": int(max_budget),
                            "metric_budgets": tuple(int(v) for v in cfg.budgets),
                            "family": family,
                            "method": method,
                            "optimizer": optimizer,
                            "lr": float(lr),
                            "start_index": int(start_index),
                        }
                    )
    return records


def _make_eval_records(
    *,
    experiment: str,
    seed: int,
    rho: float,
    eval_starts: torch.Tensor,
    selected_lrs: pd.DataFrame,
    train_loss_fn,
    test_loss_fn,
    trained_flow: torch.nn.Module,
    random_flow: torch.nn.Module,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = selected_lrs[
        (selected_lrs["experiment"] == experiment)
        & (selected_lrs["seed"] == int(seed))
        & (selected_lrs["rho"] == float(rho))
    ]
    for selected_row in selected.itertuples(index=False):
        for start_index, theta0 in enumerate(eval_starts):
            curve = _run_method_curve(
                family=str(selected_row.family),
                optimizer=str(selected_row.optimizer),
                lr=float(selected_row.selected_lr),
                theta0=theta0,
                steps=int(selected_row.budget),
                train_loss_fn=train_loss_fn,
                test_loss_fn=test_loss_fn,
                trained_flow=trained_flow,
                random_flow=random_flow,
            )
            records.append(
                {
                    "curve_id": _curve_id(
                        experiment,
                        "eval",
                        seed,
                        rho,
                        selected_row.method,
                        selected_row.selected_lr,
                        selected_row.budget,
                        start_index,
                    ),
                    "curve": curve,
                    "experiment": experiment,
                    "split": "eval",
                    "candidate": False,
                    "seed": int(seed),
                    "rho": float(rho),
                    "curve_budget": int(selected_row.budget),
                    "metric_budgets": (int(selected_row.budget),),
                    "family": str(selected_row.family),
                    "method": str(selected_row.method),
                    "optimizer": str(selected_row.optimizer),
                    "lr": float(selected_row.selected_lr),
                    "start_index": int(start_index),
                }
            )
    return records


def _run_mlp_experiment(cfg: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_tuning_results: list[dict[str, object]] = []
    all_tuning_curves: list[dict[str, object]] = []
    all_selected_lrs: list[pd.DataFrame] = []
    all_eval_records: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, object]] = []
    flow_history_rows: list[dict[str, object]] = []
    max_budget = int(max(cfg.budgets))

    for seed in cfg.seeds:
        problem = MLPRegressionProblem(cfg=cfg, seed=int(seed), device=device, dtype=dtype)
        theta_pool = collect_mlp_flow_pool(problem, cfg, seed=int(seed))
        heldout_pool = collect_mlp_heldout_geometry_pool(problem, cfg, seed=int(seed))
        scales = fit_probe_scales(problem.probe_residual, theta_pool)
        tune_starts = problem.sample_starts(int(cfg.k_tune), seed=seed_offset(int(seed), 303))
        eval_starts = problem.sample_starts(int(cfg.k_eval), seed=seed_offset(int(seed), 404))

        seed_tuning_records: list[dict[str, Any]] = []
        seed_flows: dict[float, tuple[torch.nn.Module, torch.nn.Module]] = {}
        for rho in cfg.rho_values:
            probe = ResidualThetaProbe(problem.probe_residual, scales=scales, rho=float(rho), eps=float(cfg.aulc_eps))
            trained_flow, random_flow, geom, flow_history = _train_and_evaluate_flows(
                experiment="mlp",
                seed=int(seed),
                rho=float(rho),
                probe=probe,
                theta_pool=theta_pool,
                heldout_pool=heldout_pool,
                cfg=cfg,
                flow_steps=int(cfg.flow_steps),
                random_seed=seed_offset(int(seed), 501) + int(round(float(rho) * 1_000_000)),
            )
            seed_flows[float(rho)] = (trained_flow, random_flow)
            geometry_rows.extend(geom)
            flow_history_rows.extend(flow_history)
            seed_tuning_records.extend(
                _make_tuning_records(
                    experiment="mlp",
                    seed=int(seed),
                    rho=float(rho),
                    tune_starts=tune_starts,
                    max_budget=max_budget,
                    train_loss_fn=problem.train_loss,
                    test_loss_fn=problem.test_loss,
                    trained_flow=trained_flow,
                    random_flow=random_flow,
                    cfg=cfg,
                )
            )
        tuning_results, tuning_curves = _records_to_tables(
            seed_tuning_records,
            metric_budgets=cfg.budgets,
            cfg=cfg,
            success_threshold_by_seed=None,
        )
        seed_tuning_df = pd.DataFrame(tuning_results)
        selected = _select_lrs(seed_tuning_df)
        all_selected_lrs.append(selected)
        all_tuning_results.extend(tuning_results)
        all_tuning_curves.extend(tuning_curves)

        for rho in cfg.rho_values:
            trained_flow, random_flow = seed_flows[float(rho)]
            all_eval_records.extend(
                _make_eval_records(
                    experiment="mlp",
                    seed=int(seed),
                    rho=float(rho),
                    eval_starts=eval_starts,
                    selected_lrs=selected,
                    train_loss_fn=problem.train_loss,
                    test_loss_fn=problem.test_loss,
                    trained_flow=trained_flow,
                    random_flow=random_flow,
                )
            )

    selected_lrs = pd.concat(all_selected_lrs, ignore_index=True) if all_selected_lrs else pd.DataFrame()
    thresholds = _success_thresholds(pd.DataFrame(all_tuning_results), selected_lrs, cfg)
    eval_results, eval_curves = _records_to_tables(
        all_eval_records,
        metric_budgets=cfg.budgets,
        cfg=cfg,
        success_threshold_by_seed=thresholds,
    )
    results = pd.DataFrame(all_tuning_results + eval_results)
    curves = pd.DataFrame(all_tuning_curves + eval_curves)
    geometry = pd.DataFrame(geometry_rows)
    flow_history = pd.DataFrame(flow_history_rows)
    return results, curves, geometry, selected_lrs, flow_history


def _run_sanity_experiment(cfg: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_tuning_results: list[dict[str, object]] = []
    all_tuning_curves: list[dict[str, object]] = []
    all_selected_lrs: list[pd.DataFrame] = []
    all_eval_records: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, object]] = []
    flow_history_rows: list[dict[str, object]] = []
    max_budget = int(max(cfg.budgets))

    for seed in cfg.seeds:
        problem = QuadraticProblem(
            dim=int(cfg.sanity_dim),
            condition_number=float(cfg.sanity_condition_number),
            seed=int(seed),
            device=device,
            dtype=dtype,
        )
        theta_pool = collect_quadratic_flow_pool(problem, cfg, seed=int(seed))
        heldout_pool = collect_quadratic_heldout_geometry_pool(problem, cfg, seed=int(seed))
        tune_starts = problem.sample_starts(int(cfg.k_tune), seed=seed_offset(int(seed), 803), init_std=float(cfg.init_std))
        eval_starts = problem.sample_starts(int(cfg.k_eval), seed=seed_offset(int(seed), 804), init_std=float(cfg.init_std))
        probe = ResidualOnlyProbe(problem.residual)
        trained_flow, random_flow, geom, flow_history = _train_and_evaluate_flows(
            experiment="sanity",
            seed=int(seed),
            rho=0.0,
            probe=probe,
            theta_pool=theta_pool,
            heldout_pool=heldout_pool,
            cfg=cfg,
            flow_steps=int(cfg.sanity_flow_steps),
            random_seed=seed_offset(int(seed), 805),
        )
        geometry_rows.extend(geom)
        flow_history_rows.extend(flow_history)
        tuning_records = _make_tuning_records(
            experiment="sanity",
            seed=int(seed),
            rho=0.0,
            tune_starts=tune_starts,
            max_budget=max_budget,
            train_loss_fn=problem.train_loss,
            test_loss_fn=problem.test_loss,
            trained_flow=trained_flow,
            random_flow=random_flow,
            cfg=cfg,
        )
        tuning_results, tuning_curves = _records_to_tables(
            tuning_records,
            metric_budgets=cfg.budgets,
            cfg=cfg,
            success_threshold_by_seed=None,
        )
        seed_tuning_df = pd.DataFrame(tuning_results)
        selected = _select_lrs(seed_tuning_df)
        all_selected_lrs.append(selected)
        all_tuning_results.extend(tuning_results)
        all_tuning_curves.extend(tuning_curves)
        all_eval_records.extend(
            _make_eval_records(
                experiment="sanity",
                seed=int(seed),
                rho=0.0,
                eval_starts=eval_starts,
                selected_lrs=selected,
                train_loss_fn=problem.train_loss,
                test_loss_fn=problem.test_loss,
                trained_flow=trained_flow,
                random_flow=random_flow,
            )
        )

    selected_lrs = pd.concat(all_selected_lrs, ignore_index=True) if all_selected_lrs else pd.DataFrame()
    thresholds = _success_thresholds(pd.DataFrame(all_tuning_results), selected_lrs, cfg)
    eval_results, eval_curves = _records_to_tables(
        all_eval_records,
        metric_budgets=cfg.budgets,
        cfg=cfg,
        success_threshold_by_seed=thresholds,
    )
    return (
        pd.DataFrame(all_tuning_results + eval_results),
        pd.DataFrame(all_tuning_curves + eval_curves),
        pd.DataFrame(geometry_rows),
        selected_lrs,
        pd.DataFrame(flow_history_rows),
    )


def _write_tables(output_dir: Path, cfg: ExperimentConfig, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if bool(cfg.write_csv) and name in {"results", "selected_lrs", "aggregate", "paired_deltas", "flow_training"}:
            frame.to_csv(output_dir / f"{name}.csv", index=False)
        if bool(cfg.write_parquet):
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    if bool(cfg.write_csv) and "results" in tables:
        tables["results"].to_csv(output_dir / "results.csv", index=False)
    if bool(cfg.write_csv) and "selected_lrs" in tables:
        tables["selected_lrs"].to_csv(output_dir / "selected_lrs.csv", index=False)


def _load_tables(output_dir: Path) -> ExperimentTables:
    config_payload = read_config(output_dir / "config.json") or {}
    results = pd.read_parquet(output_dir / "results.parquet")
    curves = pd.read_parquet(output_dir / "curves.parquet")
    geometry = pd.read_parquet(output_dir / "geometry.parquet")
    selected_lrs = pd.read_parquet(output_dir / "selected_lrs.parquet")
    aggregate = pd.read_parquet(output_dir / "aggregate.parquet")
    deltas = pd.read_parquet(output_dir / "paired_deltas.parquet")
    figures_path = output_dir / "figures.json"
    figure_paths = json.loads(figures_path.read_text(encoding="utf-8")) if figures_path.is_file() else {}
    interpretation_path = output_dir / "interpretation.md"
    interpretation = interpretation_path.read_text(encoding="utf-8") if interpretation_path.is_file() else ""
    return ExperimentTables(
        results=results,
        curves=curves,
        geometry=geometry,
        selected_lrs=selected_lrs,
        aggregate=aggregate,
        paired_deltas=deltas,
        config=config_payload,
        output_dir=output_dir,
        figure_paths=figure_paths,
        interpretation_markdown=interpretation,
    )


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
    sanity = _run_sanity_experiment(cfg, device=device, dtype=dtype)
    mlp = _run_mlp_experiment(cfg, device=device, dtype=dtype)
    results = pd.concat([sanity[0], mlp[0]], ignore_index=True)
    curves = pd.concat([sanity[1], mlp[1]], ignore_index=True)
    geometry = pd.concat([sanity[2], mlp[2]], ignore_index=True)
    selected_lrs = pd.concat([sanity[3], mlp[3]], ignore_index=True)
    flow_training = pd.concat([sanity[4], mlp[4]], ignore_index=True)
    aggregate = aggregate_results(results, cfg)
    deltas = paired_deltas(results)
    interpretation = build_interpretation_markdown(results=results, geometry=geometry, cfg=cfg)
    figures = save_figures(results=results, curves=curves, geometry=geometry, output_dir=output_dir, cfg=cfg) if cfg.save_figures else {}
    payload = write_config(output_dir / "config.json", cfg)
    tables = {
        "results": results,
        "curves": curves,
        "geometry": geometry,
        "selected_lrs": selected_lrs,
        "aggregate": aggregate,
        "paired_deltas": deltas,
        "flow_training": flow_training,
    }
    _write_tables(output_dir, cfg, tables)
    (output_dir / "figures.json").write_text(json.dumps(figures, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "interpretation.md").write_text(interpretation, encoding="utf-8")
    return ExperimentTables(
        results=results,
        curves=curves,
        geometry=geometry,
        selected_lrs=selected_lrs,
        aggregate=aggregate,
        paired_deltas=deltas,
        config=payload,
        output_dir=output_dir,
        figure_paths=figures,
        interpretation_markdown=interpretation,
    )


def run_or_load(cfg: ExperimentConfig) -> ExperimentTables:
    output_dir = run_dir(cfg)
    if bool(cfg.cache_first) and not bool(cfg.force_rerun) and _cache_is_valid(output_dir, cfg):
        return _load_tables(output_dir)
    return run_all_experiments(cfg)
