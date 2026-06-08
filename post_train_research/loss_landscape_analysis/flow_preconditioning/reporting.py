from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig


GROUP_COLS = ["experiment", "task", "condition_name", "condition_value"]


def _safe_name(*parts: object) -> str:
    raw = "__".join(str(part) for part in parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


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
    group_cols = GROUP_COLS + ["numeric_condition", "budget", "method", "optimizer"]
    for key, frame in eval_rows.groupby(group_cols, dropna=False):
        values = frame["aulc"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(values, samples=int(cfg.bootstrap_samples), seed=int(cfg.bootstrap_seed) + len(rows))
        rows.append(
            {
                "experiment": key[0],
                "task": key[1],
                "condition_name": key[2],
                "condition_value": key[3],
                "numeric_condition": float(key[4]),
                "budget": int(key[5]),
                "method": key[6],
                "optimizer": key[7],
                "n": int(len(frame)),
                "median_aulc": float(np.median(values)) if len(values) else float("nan"),
                "median_aulc_ci_low": ci_low,
                "median_aulc_ci_high": ci_high,
                "median_final_loss": float(frame["final_train_loss"].median()),
                "median_best_seen_loss": float(frame["best_train_loss"].median()),
                "median_final_test_loss": float(frame["final_test_loss"].median()),
                "success_rate": float(frame["success"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    eval_rows = results[(results["split"] == "eval") & (results["candidate"] == False)].copy()  # noqa: E712
    rows: list[dict[str, object]] = []
    key_cols = GROUP_COLS + ["numeric_condition", "seed", "budget", "optimizer", "start_index"]
    for key, frame in eval_rows.groupby(key_cols, dropna=False):
        by_method = {str(row.method): row for row in frame.itertuples(index=False)}
        optimizer = str(key[7])
        direct = by_method.get(f"direct_{optimizer}")
        trained = by_method.get(f"trained_flow_{optimizer}")
        random = by_method.get(f"random_flow_{optimizer}")
        base = {
            "experiment": key[0],
            "task": key[1],
            "condition_name": key[2],
            "condition_value": key[3],
            "numeric_condition": float(key[4]),
            "seed": int(key[5]),
            "budget": int(key[6]),
            "optimizer": optimizer,
            "start_index": int(key[8]),
        }
        if direct is not None and trained is not None:
            rows.append(
                {
                    **base,
                    "comparison": "trained_minus_direct",
                    "delta_aulc": float(trained.aulc - direct.aulc),
                    "left_aulc": float(trained.aulc),
                    "right_aulc": float(direct.aulc),
                }
            )
        if random is not None and trained is not None:
            rows.append(
                {
                    **base,
                    "comparison": "trained_minus_random",
                    "delta_aulc": float(trained.aulc - random.aulc),
                    "left_aulc": float(trained.aulc),
                    "right_aulc": float(random.aulc),
                }
            )
    return pd.DataFrame(rows)


def geometry_summary(geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if geometry.empty:
        return pd.DataFrame()
    group_cols = GROUP_COLS + ["diagnostic", "coordinate", "seed", "numeric_condition"]
    frame = geometry.copy()
    if "numeric_condition" not in frame.columns:
        frame["numeric_condition"] = frame.get("rho", 0.0)
    for key, group in frame.groupby(group_cols, dropna=False):
        row = {
            "experiment": key[0],
            "task": key[1],
            "condition_name": key[2],
            "condition_value": key[3],
            "diagnostic": key[4],
            "coordinate": key[5],
            "seed": int(key[6]),
            "numeric_condition": float(key[7]),
        }
        if key[4] == "pullback_metric":
            finite_cond = group["condition_number"].replace([np.inf, -np.inf], np.nan)
            trace = group["trace_g"].to_numpy(dtype=np.float64)
            trace_mean = float(np.nanmean(trace)) if len(trace) else float("nan")
            row.update(
                {
                    "isometry_objective": float(group["isometry_objective"].iloc[0]),
                    "median_condition_number": float(finite_cond.median()),
                    "median_log_condition_number": float(group["log_condition_number"].replace([np.inf, -np.inf], np.nan).median()),
                    "trace_g_cv": float(np.nanstd(trace) / max(abs(trace_mean), 1e-12)) if len(trace) else float("nan"),
                    "median_flow_log_abs_det_jacobian": float("nan"),
                    "median_flow_condition_number": float("nan"),
                    "median_flow_displacement_norm": float("nan"),
                }
            )
        else:
            row.update(
                {
                    "isometry_objective": float("nan"),
                    "median_condition_number": float("nan"),
                    "median_log_condition_number": float("nan"),
                    "trace_g_cv": float("nan"),
                    "median_flow_log_abs_det_jacobian": float(group["flow_log_abs_det_jacobian"].median()),
                    "median_flow_condition_number": float(group["flow_condition_number"].replace([np.inf, -np.inf], np.nan).median()),
                    "median_flow_displacement_norm": float(group["flow_displacement_norm"].median()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_interpretation_markdown(*, results: pd.DataFrame, geometry: pd.DataFrame, cfg: ExperimentConfig) -> str:
    deltas = paired_deltas(results)
    geom = geometry_summary(geometry)
    max_budget = int(max(cfg.budgets))
    lines = ["## Automatic Interpretation", ""]
    for experiment in ["E0", "E1", "E2", "E3", "E4"]:
        exp_deltas = deltas[
            (deltas["experiment"] == experiment)
            & (deltas["budget"] == max_budget)
            & (deltas["comparison"] == "trained_minus_direct")
        ]
        exp_random = deltas[
            (deltas["experiment"] == experiment)
            & (deltas["budget"] == max_budget)
            & (deltas["comparison"] == "trained_minus_random")
        ]
        exp_geom = geom[(geom["experiment"] == experiment) & (geom["diagnostic"] == "pullback_metric")]
        before = exp_geom[exp_geom["coordinate"] == "original"]["isometry_objective"]
        after = exp_geom[exp_geom["coordinate"] == "trained_flow"]["isometry_objective"]
        before_med = float(before.median()) if len(before) else float("nan")
        after_med = float(after.median()) if len(after) else float("nan")
        geom_better = math.isfinite(before_med) and math.isfinite(after_med) and after_med < before_med
        median_delta = float(exp_deltas["delta_aulc"].median()) if len(exp_deltas) else float("nan")
        trained_better = math.isfinite(median_delta) and median_delta < 0.0
        random_delta = float(exp_random["delta_aulc"].median()) if len(exp_random) else float("nan")
        survives_random = math.isfinite(random_delta) and random_delta < 0.0
        sgd_delta = exp_deltas[exp_deltas["optimizer"] == "sgd"]["delta_aulc"].median()
        adam_delta = exp_deltas[exp_deltas["optimizer"] == "adam"]["delta_aulc"].median()
        stronger = "SGD" if float(sgd_delta) < float(adam_delta) else "Adam"
        by_condition = exp_deltas.groupby(["task", "condition_name", "condition_value"])["delta_aulc"].median()
        if len(by_condition):
            best_key = by_condition.idxmin()
            worst_key = by_condition.idxmax()
            mattered = f"best={best_key} ({float(by_condition.min()):.4g}), worst={worst_key} ({float(by_condition.max()):.4g})"
        else:
            mattered = "not enough paired rows"
        verdict = (
            "consistent with learned nonlinear preconditioning"
            if geom_better and trained_better and survives_random
            else "not enough paired evidence to rule out LR/step-size or random-coordinate artifacts"
        )
        lines.extend(
            [
                f"### {experiment}",
                f"1. Held-out geometry improved: **{'yes' if geom_better else 'no'}** (median R original={before_med:.4g}, trained={after_med:.4g}).",
                f"2. Trained-flow AULC improved over direct: **{'yes' if trained_better else 'no'}** (median delta={median_delta:.4g}; negative is better).",
                f"3. Improvement survived random-flow baseline: **{'yes' if survives_random else 'no'}** (median trained-random delta={random_delta:.4g}).",
                f"4. Stronger optimizer: **{stronger}** (SGD delta={float(sgd_delta):.4g}, Adam delta={float(adam_delta):.4g}).",
                f"5. Conditions that mattered most: {mattered}.",
                f"6. Mechanism verdict: {verdict}.",
                "",
            ]
        )
    return "\n".join(lines)


def save_figures(
    *,
    results: pd.DataFrame,
    curves: pd.DataFrame,
    geometry: pd.DataFrame,
    output_dir: Path,
    cfg: ExperimentConfig,
    trajectories: pd.DataFrame | None = None,
    warped_grids: pd.DataFrame | None = None,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    max_budget = int(max(cfg.budgets))

    eval_curves = curves[(curves["split"] == "eval") & (curves["budget"] == max_budget)]
    for key, frame in eval_curves.groupby(GROUP_COLS, dropna=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, method_frame in frame.groupby("method"):
            grouped = method_frame.groupby("step")["train_loss"]
            med = grouped.median()
            q25 = grouped.quantile(0.25)
            q75 = grouped.quantile(0.75)
            ax.plot(med.index, med.values, label=method)
            ax.fill_between(med.index, q25.values, q75.values, alpha=0.12)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(" ".join(str(v) for v in key))
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)
        name = _safe_name("median_curves", *key)
        path = figure_dir / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path)

    deltas = paired_deltas(results)
    scatter_rows = deltas[(deltas["budget"] == max_budget) & (deltas["comparison"] == "trained_minus_direct")]
    for key, frame in scatter_rows.groupby(GROUP_COLS, dropna=False):
        fig, ax = plt.subplots(figsize=(5, 5))
        for optimizer, opt_frame in frame.groupby("optimizer"):
            ax.scatter(opt_frame["right_aulc"], opt_frame["left_aulc"], s=18, alpha=0.7, label=optimizer)
        lo = float(np.nanmin([frame["right_aulc"].min(), frame["left_aulc"].min()]))
        hi = float(np.nanmax([frame["right_aulc"].max(), frame["left_aulc"].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel("direct AULC")
        ax.set_ylabel("trained-flow AULC")
        ax.set_title(" ".join(str(v) for v in key))
        ax.grid(True, alpha=0.25)
        ax.legend()
        name = _safe_name("paired_scatter", *key)
        path = figure_dir / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path)

    geom = geometry_summary(geometry)
    pullback = geom[geom["diagnostic"] == "pullback_metric"] if len(geom) else pd.DataFrame()
    for key, frame in pullback.groupby(["experiment", "task", "condition_name", "condition_value"], dropna=False):
        fig, ax = plt.subplots(figsize=(7, 4))
        frame.boxplot(column="isometry_objective", by="coordinate", ax=ax)
        ax.set_title(" ".join(str(v) for v in key))
        fig.suptitle("")
        ax.set_ylabel("R")
        name = _safe_name("geometry", *key)
        path = figure_dir / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path)

    if trajectories is not None and len(trajectories):
        for key, frame in trajectories.groupby(GROUP_COLS, dropna=False):
            fig, ax = plt.subplots(figsize=(6, 5))
            for (method, start_index), traj in frame.groupby(["method", "start_index"]):
                points = np.array([json.loads(value) for value in traj.sort_values("step")["z_json"]], dtype=np.float64)
                if points.shape[1] < 2:
                    continue
                ax.plot(points[:, 0], points[:, 1], linewidth=1.2, label=f"{method}/s{start_index}")
                ax.scatter(points[0, 0], points[0, 1], s=12, color="black")
            ax.set_xlabel("z0")
            ax.set_ylabel("z1")
            ax.set_title("trajectories " + " ".join(str(v) for v in key))
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6, ncol=2)
            name = _safe_name("trajectories", *key)
            path = figure_dir / f"{name}.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths[name] = str(path)

    if warped_grids is not None and len(warped_grids):
        for key, frame in warped_grids[warped_grids["flow"] == "trained_flow"].groupby(GROUP_COLS + ["seed"], dropna=False):
            fig, ax = plt.subplots(figsize=(5, 5))
            for (_axis, _line_index), line in frame.groupby(["axis", "line_index"]):
                line = line.sort_values("point_index")
                ax.plot(line["u0"], line["u1"], color="tab:blue", linewidth=0.7, alpha=0.65)
            ax.set_xlabel("u0 = i(z)_0")
            ax.set_ylabel("u1 = i(z)_1")
            ax.set_title("warped grid " + " ".join(str(v) for v in key))
            ax.grid(True, alpha=0.25)
            name = _safe_name("warped_grid", *key)
            path = figure_dir / f"{name}.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths[name] = str(path)

    return paths
