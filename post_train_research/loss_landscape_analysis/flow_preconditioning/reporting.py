from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def bootstrap_ci(values: np.ndarray, *, samples: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if int(array.size) == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(int(seed))
    medians = np.empty(int(samples), dtype=np.float64)
    for idx in range(int(samples)):
        draw = generator.choice(array, size=int(array.size), replace=True)
        medians[idx] = float(np.median(draw))
    return (
        float(np.quantile(medians, alpha / 2.0)),
        float(np.quantile(medians, 1.0 - alpha / 2.0)),
    )


def aggregate_results(results: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    eval_rows = results[(results["split"] == "eval") & (results["candidate"] == False)].copy()  # noqa: E712
    rows: list[dict[str, object]] = []
    group_cols = ["experiment", "rho", "budget", "method", "optimizer"]
    for key, frame in eval_rows.groupby(group_cols, dropna=False):
        values = frame["aulc"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(
            values,
            samples=int(cfg.bootstrap_samples),
            seed=int(cfg.bootstrap_seed) + len(rows),
        )
        rows.append(
            {
                "experiment": key[0],
                "rho": float(key[1]),
                "budget": int(key[2]),
                "method": key[3],
                "optimizer": key[4],
                "n": int(len(frame)),
                "median_aulc": float(np.median(values)) if len(values) else float("nan"),
                "median_aulc_ci_low": ci_low,
                "median_aulc_ci_high": ci_high,
                "median_final_train_loss": float(frame["final_train_loss"].median()),
                "median_final_test_loss": float(frame["final_test_loss"].median()),
                "success_rate": float(frame["success"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    eval_rows = results[(results["split"] == "eval") & (results["candidate"] == False)].copy()  # noqa: E712
    rows: list[dict[str, object]] = []
    key_cols = ["experiment", "seed", "rho", "budget", "optimizer", "start_index"]
    for key, frame in eval_rows.groupby(key_cols, dropna=False):
        by_method = {str(row.method): row for row in frame.itertuples(index=False)}
        direct = by_method.get(f"direct_{key[4]}")
        trained = by_method.get(f"trained_flow_{key[4]}")
        random = by_method.get(f"random_flow_{key[4]}")
        if direct is not None and trained is not None:
            rows.append(
                {
                    "experiment": key[0],
                    "seed": int(key[1]),
                    "rho": float(key[2]),
                    "budget": int(key[3]),
                    "optimizer": key[4],
                    "start_index": int(key[5]),
                    "comparison": "trained_minus_direct",
                    "delta_aulc": float(trained.aulc - direct.aulc),
                    "left_aulc": float(trained.aulc),
                    "right_aulc": float(direct.aulc),
                }
            )
        if random is not None and trained is not None:
            rows.append(
                {
                    "experiment": key[0],
                    "seed": int(key[1]),
                    "rho": float(key[2]),
                    "budget": int(key[3]),
                    "optimizer": key[4],
                    "start_index": int(key[5]),
                    "comparison": "trained_minus_random",
                    "delta_aulc": float(trained.aulc - random.aulc),
                    "left_aulc": float(trained.aulc),
                    "right_aulc": float(random.aulc),
                }
            )
    return pd.DataFrame(rows)


def geometry_summary(geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["experiment", "coordinate", "seed", "rho"]
    for key, frame in geometry.groupby(group_cols, dropna=False):
        finite_cond = frame["condition_number"].replace([np.inf, -np.inf], np.nan)
        trace = frame["trace_g"].to_numpy(dtype=np.float64)
        trace_mean = float(np.nanmean(trace)) if len(trace) else float("nan")
        trace_cv = float(np.nanstd(trace) / max(abs(trace_mean), 1e-12)) if len(trace) else float("nan")
        rows.append(
            {
                "experiment": key[0],
                "coordinate": key[1],
                "seed": int(key[2]),
                "rho": float(key[3]),
                "isometry_objective": float(frame["isometry_objective"].iloc[0]),
                "median_condition_number": float(finite_cond.median()),
                "median_log_condition_number": float(frame["log_condition_number"].replace([np.inf, -np.inf], np.nan).median()),
                "trace_g_cv": trace_cv,
            }
        )
    return pd.DataFrame(rows)


def build_interpretation_markdown(
    *,
    results: pd.DataFrame,
    geometry: pd.DataFrame,
    cfg: ExperimentConfig,
) -> str:
    deltas = paired_deltas(results)
    geom = geometry_summary(geometry)
    main_budget = int(max(cfg.budgets))
    main_rho = float(cfg.main_rho)
    main = deltas[
        (deltas["experiment"] == "mlp")
        & (deltas["rho"] == main_rho)
        & (deltas["budget"] == main_budget)
        & (deltas["comparison"] == "trained_minus_direct")
    ]
    random_cmp = deltas[
        (deltas["experiment"] == "mlp")
        & (deltas["rho"] == main_rho)
        & (deltas["budget"] == main_budget)
        & (deltas["comparison"] == "trained_minus_random")
    ]
    geom_main = geom[(geom["experiment"] == "mlp") & (geom["rho"] == main_rho)]
    before = geom_main[geom_main["coordinate"] == "theta"]
    after = geom_main[geom_main["coordinate"] == "trained_flow"]
    before_iso = float(before["isometry_objective"].median()) if len(before) else float("nan")
    after_iso = float(after["isometry_objective"].median()) if len(after) else float("nan")
    geom_better = math.isfinite(before_iso) and math.isfinite(after_iso) and after_iso < before_iso

    sgd_delta = main[main["optimizer"] == "sgd"]["delta_aulc"].median()
    adam_delta = main[main["optimizer"] == "adam"]["delta_aulc"].median()
    trained_better = float(main["delta_aulc"].median()) < 0.0 if len(main) else False
    survives_random = float(random_cmp["delta_aulc"].median()) < 0.0 if len(random_cmp) else False
    rho_groups = deltas[
        (deltas["experiment"] == "mlp")
        & (deltas["budget"] == main_budget)
        & (deltas["comparison"] == "trained_minus_direct")
    ].groupby("rho")["delta_aulc"].median()
    rho_span = float(rho_groups.max() - rho_groups.min()) if len(rho_groups) else float("nan")

    lines = [
        "## Automatic Interpretation",
        "",
        f"1. Geometry improved on held-out samples: **{'yes' if geom_better else 'no'}** "
        f"(median R theta={before_iso:.4g}, trained-flow={after_iso:.4g}).",
        f"2. Trained-flow improved matched-start AULC at rho={main_rho:g}, T={main_budget}: "
        f"**{'yes' if trained_better else 'no'}** (median delta={float(main['delta_aulc'].median()) if len(main) else float('nan'):.4g}; negative is better).",
        f"3. Improvement survived random-flow baseline: **{'yes' if survives_random else 'no'}** "
        f"(median trained-random delta={float(random_cmp['delta_aulc'].median()) if len(random_cmp) else float('nan'):.4g}).",
        f"4. Stronger optimizer: **{'SGD' if sgd_delta < adam_delta else 'Adam'}** "
        f"(SGD delta={float(sgd_delta):.4g}, Adam delta={float(adam_delta):.4g}).",
        f"5. Rho sensitivity: median delta span across rho values is {rho_span:.4g}; inspect rho table before claiming robustness.",
        "6. Mechanism verdict: "
        + (
            "consistent with learned nonlinear preconditioning."
            if geom_better and trained_better and survives_random
            else "not enough paired evidence to rule out step-size or random reparametrization artifacts."
        ),
    ]
    return "\n".join(lines)


def save_figures(
    *,
    results: pd.DataFrame,
    curves: pd.DataFrame,
    geometry: pd.DataFrame,
    output_dir: Path,
    cfg: ExperimentConfig,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    eval_curves = curves[(curves["split"] == "eval") & (curves["budget"] == int(max(cfg.budgets)))]
    if len(eval_curves):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, frame in eval_curves.groupby("method"):
            med = frame.groupby("step")["train_loss"].median()
            ax.plot(med.index, med.values, label=method)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("median train loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        path = figure_dir / "median_train_loss_curves.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["median_train_loss_curves"] = str(path)

    deltas = paired_deltas(results)
    scatter = deltas[
        (deltas["experiment"] == "mlp")
        & (deltas["comparison"] == "trained_minus_direct")
        & (deltas["rho"] == float(cfg.main_rho))
        & (deltas["budget"] == int(max(cfg.budgets)))
    ]
    if len(scatter):
        fig, ax = plt.subplots(figsize=(5, 5))
        for optimizer, frame in scatter.groupby("optimizer"):
            ax.scatter(frame["right_aulc"], frame["left_aulc"], s=18, alpha=0.7, label=optimizer)
        lo = float(np.nanmin([scatter["right_aulc"].min(), scatter["left_aulc"].min()]))
        hi = float(np.nanmax([scatter["right_aulc"].max(), scatter["left_aulc"].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel("direct AULC")
        ax.set_ylabel("trained-flow AULC")
        ax.legend()
        ax.grid(True, alpha=0.25)
        path = figure_dir / "paired_aulc_scatter.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["paired_aulc_scatter"] = str(path)

    if len(geometry):
        fig, ax = plt.subplots(figsize=(8, 5))
        geom = geometry_summary(geometry)
        show = geom[(geom["experiment"] == "mlp") & (geom["rho"] == float(cfg.main_rho))]
        show.boxplot(column="isometry_objective", by="coordinate", ax=ax)
        ax.set_title("Heldout isometry objective")
        fig.suptitle("")
        ax.set_ylabel("R")
        path = figure_dir / "geometry_isometry_boxplot.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["geometry_isometry_boxplot"] = str(path)

    return paths
