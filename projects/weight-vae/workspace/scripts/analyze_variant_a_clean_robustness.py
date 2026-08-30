from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
ANALYSIS_DIR = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/clean_harness_analysis"
OUT_DIR = ANALYSIS_DIR / "robustness"

RUNS = {
    "control_clean": "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
    "a_cap1_clean": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
}

KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]


def _load_decoder_results() -> pd.DataFrame:
    rows = pd.read_csv(ANALYSIS_DIR / "clean_decoder_latent_per_start_deltas.csv")
    return rows.sort_values(KEY_COLS).reset_index(drop=True)


def _load_decoder_curves() -> pd.DataFrame:
    rows = pd.read_csv(ANALYSIS_DIR / "clean_decoder_latent_curve_deltas.csv")
    return rows.sort_values(KEY_COLS + ["step"]).reset_index(drop=True)


def _run_dir(label: str) -> Path:
    return ARTIFACT_ROOT / RUNS[label]


def _config_dict(label: str) -> dict[str, object]:
    payload = json.loads((_run_dir(label) / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise RuntimeError(f"{label} config.json does not contain a config mapping")
    return raw_cfg


def _expected_curve_steps(label: str) -> list[int]:
    cfg = _config_dict(label)
    total = int(cfg.get("downstream_steps", 0))
    every = max(1, int(cfg.get("downstream_eval_every", 1)))
    steps = list(range(0, total + 1, every))
    if not steps or steps[-1] != total:
        steps.append(total)
    return steps


def _is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _assert_decoder_curves_complete(label: str) -> None:
    method = "decoder_latent"
    results = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    curves = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    result_rows = results[(results["split"].astype(str) == "eval") & (results["method"].astype(str) == method)].copy()
    curve_rows = curves[(curves["split"].astype(str) == "eval") & (curves["method"].astype(str) == method)].copy()
    if result_rows.empty or curve_rows.empty:
        raise RuntimeError(f"{label} has missing eval decoder_latent downstream rows")
    if "diverged" in result_rows and bool(_is_true(result_rows["diverged"]).any()):
        bad = result_rows.loc[_is_true(result_rows["diverged"]), KEY_COLS].to_dict("records")
        raise RuntimeError(f"{label} has diverged eval decoder results: {bad[:5]}")
    if "diverged" in curve_rows and bool(_is_true(curve_rows["diverged"]).any()):
        bad = curve_rows.loc[_is_true(curve_rows["diverged"]), KEY_COLS + ["step"]].to_dict("records")
        raise RuntimeError(f"{label} has diverged eval decoder curves: {bad[:5]}")

    result_keys = result_rows[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    curve_keys = curve_rows[KEY_COLS].drop_duplicates().sort_values(KEY_COLS).reset_index(drop=True)
    if not result_keys.equals(curve_keys):
        raise RuntimeError(f"{label} eval decoder result keys do not match curve keys")

    expected_steps = _expected_curve_steps(label)
    bad_groups: list[dict[str, object]] = []
    for key, group in curve_rows.groupby(KEY_COLS, sort=False):
        steps = pd.to_numeric(group["step"], errors="coerce").dropna().astype(int).tolist()
        if sorted(steps) != expected_steps:
            bad_groups.append({"key": key, "steps": sorted(steps)})
    if bad_groups:
        raise RuntimeError(
            f"{label} has incomplete eval decoder curves; "
            f"expected_steps={expected_steps}, examples={bad_groups[:3]}"
        )


def _exact_sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    obs = abs(float(values.mean()))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        candidate = abs(float((values * np.asarray(signs, dtype=np.float64)).mean()))
        if candidate >= obs - 1e-15:
            count += 1
    return count / max(total, 1)


def _bootstrap_ci(values: np.ndarray, *, seed: int = 0, samples: int = 50_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(int(samples), len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _center_by_task(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = rows.copy()
    for col in columns:
        out[f"{col}_task_centered"] = out[col] - out.groupby("task_name")[col].transform("mean")
    return out


def _curve_auc_by_start(curves: pd.DataFrame) -> pd.DataFrame:
    n_steps = int(curves["step"].nunique())
    grouped = curves.groupby(KEY_COLS, as_index=False).agg(
        test_curve_aulc_delta=("delta_test_loss", "mean"),
        post0_test_curve_aulc_delta=("delta_test_loss", lambda s: float(pd.to_numeric(s.iloc[1:], errors="coerce").mean())),
        step0_test_loss_delta=("delta_test_loss", "first"),
    )
    grouped["step0_test_curve_contribution"] = grouped["step0_test_loss_delta"] / max(n_steps, 1)
    grouped["post0_test_curve_contribution"] = grouped["test_curve_aulc_delta"] - grouped["step0_test_curve_contribution"]
    return grouped


def _window_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    windows = [
        ("step0", 0, 0),
        ("early_0_25", 0, 25),
        ("post0_25_300", 25, 300),
        ("late_100_300", 100, 300),
        ("full_0_300", 0, 300),
    ]
    for name, lo, hi in windows:
        sub = curves[(curves["step"] >= lo) & (curves["step"] <= hi)].copy()
        by_start = sub.groupby(KEY_COLS, as_index=False).agg(delta=("delta_test_loss", "mean"))
        rows.append(
            {
                "window": name,
                "step_lo": int(lo),
                "step_hi": int(hi),
                "n_starts": int(len(by_start)),
                "mean_test_loss_delta": float(by_start["delta"].mean()),
                "median_test_loss_delta": float(by_start["delta"].median()),
                "frac_a_better": float((by_start["delta"] < 0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "curve_window_summary.csv", index=False)
    return out


def _robust_stats(results: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    curve_auc = _curve_auc_by_start(curves)
    merged = results.merge(curve_auc, on=KEY_COLS, how="left", validate="one_to_one")
    train_delta = pd.to_numeric(merged["delta_aulc"], errors="coerce").to_numpy(dtype=np.float64)
    test_delta = pd.to_numeric(merged["test_curve_aulc_delta"], errors="coerce").to_numpy(dtype=np.float64)
    step0 = pd.to_numeric(merged["delta_step0_test_loss"], errors="coerce")

    centered = _center_by_task(merged, ["delta_aulc", "test_curve_aulc_delta", "delta_step0_test_loss"])
    rows = [
        {
            "metric": "train_loss_aulc_delta",
            "n": int(len(train_delta)),
            "mean": float(np.mean(train_delta)),
            "median": float(np.median(train_delta)),
            "sign_flip_p_two_sided": _exact_sign_flip_pvalue(train_delta),
            "bootstrap95_lo": _bootstrap_ci(train_delta)[0],
            "bootstrap95_hi": _bootstrap_ci(train_delta)[1],
            "corr_with_step0_test_loss_delta": float(pd.Series(train_delta).corr(step0)),
            "task_centered_corr_with_step0_test_loss_delta": float(
                centered["delta_aulc_task_centered"].corr(centered["delta_step0_test_loss_task_centered"])
            ),
            "step0_mean_contribution_to_test_curve_auc": float("nan"),
            "post0_mean_contribution_to_test_curve_auc": float("nan"),
        },
        {
            "metric": "test_loss_curve_aulc_delta",
            "n": int(len(test_delta)),
            "mean": float(np.mean(test_delta)),
            "median": float(np.median(test_delta)),
            "sign_flip_p_two_sided": _exact_sign_flip_pvalue(test_delta),
            "bootstrap95_lo": _bootstrap_ci(test_delta)[0],
            "bootstrap95_hi": _bootstrap_ci(test_delta)[1],
            "corr_with_step0_test_loss_delta": float(pd.Series(test_delta).corr(step0)),
            "task_centered_corr_with_step0_test_loss_delta": float(
                centered["test_curve_aulc_delta_task_centered"].corr(centered["delta_step0_test_loss_task_centered"])
            ),
            "step0_mean_contribution_to_test_curve_auc": float(merged["step0_test_curve_contribution"].mean()),
            "post0_mean_contribution_to_test_curve_auc": float(merged["post0_test_curve_contribution"].mean()),
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    merged.to_csv(OUT_DIR / "per_start_with_test_curve_auc.csv", index=False)
    return out


def _task_summary(results: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    curve_auc = _curve_auc_by_start(curves)
    merged = results.merge(curve_auc, on=KEY_COLS, how="left", validate="one_to_one")
    rows = []
    for task, sub in merged.groupby("task_name"):
        rows.append(
            {
                "task_name": task,
                "n_starts": int(len(sub)),
                "mean_train_aulc_delta": float(sub["delta_aulc"].mean()),
                "median_train_aulc_delta": float(sub["delta_aulc"].median()),
                "mean_test_curve_aulc_delta": float(sub["test_curve_aulc_delta"].mean()),
                "mean_step0_test_loss_delta": float(sub["delta_step0_test_loss"].mean()),
                "mean_reconstruction_rel_l2_delta": float(sub["delta_reconstruction_rel_l2"].mean()),
                "corr_step0_train_aulc": float(sub["delta_step0_test_loss"].corr(sub["delta_aulc"])) if len(sub) >= 3 else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "task_summary.csv", index=False)
    return out


def _leave_one_out(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in results.iterrows():
        keep = results.drop(index=idx)
        rows.append(
            {
                "removed_source_weight_index": int(row["source_weight_index"]),
                "removed_task_name": str(row["task_name"]),
                "removed_delta_aulc": float(row["delta_aulc"]),
                "removed_delta_step0_test_loss": float(row["delta_step0_test_loss"]),
                "mean_delta_aulc_without": float(keep["delta_aulc"].mean()),
                "median_delta_aulc_without": float(keep["delta_aulc"].median()),
                "corr_step0_aulc_without": float(keep["delta_step0_test_loss"].corr(keep["delta_aulc"])),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_delta_aulc_without").reset_index(drop=True)
    out.to_csv(OUT_DIR / "leave_one_out_influence.csv", index=False)
    return out


def _diagnostic_join(results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label in ["control_clean", "a_cap1_clean"]:
        diag = pd.read_csv(_run_dir(label) / "preconditioning_diagnostics.csv")
        diag = diag[diag["start_role"].astype(str) == "eval"].copy()
        keep = [
            "weight_index",
            "task_name",
            "tau",
            "li_A_full_per_dim",
            "hvp_probe_trace_m_per_dim",
            "hvp_probe_a_loss_per_dim",
            "hessian_log_abs_var",
            "hessian_log_abs_spread",
            "hessian_negative_fraction",
        ]
        diag = diag[keep].copy()
        diag = diag.rename(columns={"weight_index": "source_weight_index"})
        diag["label"] = label
        frames.append(diag)
    all_diag = pd.concat(frames, ignore_index=True)
    wide = all_diag.pivot_table(
        index=["source_weight_index", "task_name", "tau"],
        columns="label",
        values=[
            "li_A_full_per_dim",
            "hvp_probe_trace_m_per_dim",
            "hvp_probe_a_loss_per_dim",
            "hessian_log_abs_var",
            "hessian_log_abs_spread",
            "hessian_negative_fraction",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{label}" for metric, label in wide.columns]
    wide = wide.reset_index()
    for metric in [
        "li_A_full_per_dim",
        "hvp_probe_trace_m_per_dim",
        "hvp_probe_a_loss_per_dim",
        "hessian_log_abs_var",
        "hessian_log_abs_spread",
        "hessian_negative_fraction",
    ]:
        wide[f"delta_{metric}"] = wide[f"{metric}_a_cap1_clean"] - wide[f"{metric}_control_clean"]
    merged = results.merge(wide, on=["source_weight_index", "task_name", "tau"], how="left", validate="one_to_one")
    merged.to_csv(OUT_DIR / "per_start_diagnostic_join.csv", index=False)

    rows = []
    for metric in [
        "delta_li_A_full_per_dim",
        "delta_hvp_probe_trace_m_per_dim",
        "delta_hvp_probe_a_loss_per_dim",
        "delta_hessian_log_abs_var",
        "delta_hessian_log_abs_spread",
        "delta_hessian_negative_fraction",
    ]:
        rows.append(
            {
                "diagnostic_delta": metric,
                "corr_with_step0_test_loss_delta": float(merged[metric].corr(merged["delta_step0_test_loss"])),
                "corr_with_train_aulc_delta": float(merged[metric].corr(merged["delta_aulc"])),
                "corr_with_reconstruction_rel_l2_delta": float(merged[metric].corr(merged["delta_reconstruction_rel_l2"])),
                "mean_delta": float(merged[metric].mean()),
                "median_delta": float(merged[metric].median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "diagnostic_join_summary.csv", index=False)
    return merged


def _plot_influence(loo: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), constrained_layout=True)
    order = loo.sort_values("removed_delta_aulc")
    labels = order["removed_source_weight_index"].astype(str).tolist()
    axes[0].bar(np.arange(len(order)), order["removed_delta_aulc"], color=np.where(order["removed_delta_aulc"] > 0, "#f58518", "#4c78a8"))
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(np.arange(len(order)), labels, rotation=70)
    axes[0].set_ylabel("removed start train-AULC delta")
    axes[0].set_title("per-start train-AULC deltas")
    axes[0].grid(axis="y", alpha=0.25)

    order2 = loo.sort_values("mean_delta_aulc_without")
    labels2 = order2["removed_source_weight_index"].astype(str).tolist()
    axes[1].bar(np.arange(len(order2)), order2["mean_delta_aulc_without"], color="#72b7b2")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(np.arange(len(order2)), labels2, rotation=70)
    axes[1].set_ylabel("mean train-AULC delta after removal")
    axes[1].set_title("leave-one-out influence")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "robustness_leave_one_out.png", dpi=190)
    plt.close(fig)


def _plot_task_centered(results: pd.DataFrame) -> None:
    centered = _center_by_task(results, ["delta_aulc", "delta_step0_test_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3), constrained_layout=True)
    for ax, centered_flag in zip(axes, [False, True], strict=True):
        x = centered["delta_step0_test_loss_task_centered"] if centered_flag else centered["delta_step0_test_loss"]
        y = centered["delta_aulc_task_centered"] if centered_flag else centered["delta_aulc"]
        for task, sub_idx in centered.groupby("task_name").groups.items():
            ax.scatter(x.loc[sub_idx], y.loc[sub_idx], s=58, alpha=0.85, label=str(task))
            for idx in sub_idx:
                ax.annotate(str(int(centered.loc[idx, "source_weight_index"])), (x.loc[idx], y.loc[idx]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("step0 test-loss delta" + (" task-centered" if centered_flag else ""))
        ax.set_ylabel("train-AULC delta" + (" task-centered" if centered_flag else ""))
        ax.set_title("task-centered" if centered_flag else "raw")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "robustness_task_centered_step0_vs_aulc.png", dpi=190)
    plt.close(fig)


def _plot_curve_decomposition(summary: pd.DataFrame) -> None:
    row = summary[summary["metric"] == "test_loss_curve_aulc_delta"].iloc[0]
    vals = [
        row["mean"],
        row["step0_mean_contribution_to_test_curve_auc"],
        row["post0_mean_contribution_to_test_curve_auc"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.bar(np.arange(len(vals)), vals, color=["#4c78a8", "#f58518", "#72b7b2"])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(vals)), ["full test curve", "step0 contribution", "post-step0 contribution"], rotation=15, ha="right")
    ax.set_ylabel("A - control mean test-loss curve AULC")
    ax.set_title("test-loss curve AULC decomposition")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "robustness_test_curve_decomposition.png", dpi=190)
    plt.close(fig)


def _plot_diagnostic_join(joined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3), constrained_layout=True)
    panels = [
        ("delta_li_A_full_per_dim", "delta exact li_A / dim"),
        ("delta_hvp_probe_trace_m_per_dim", "delta trace(M) / dim"),
    ]
    for ax, (col, xlabel) in zip(axes, panels, strict=True):
        ax.scatter(joined[col], joined["delta_step0_test_loss"], s=58, alpha=0.85)
        for _, row in joined.iterrows():
            ax.annotate(str(int(row["source_weight_index"])), (row[col], row["delta_step0_test_loss"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("step0 test-loss delta")
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "robustness_diagnostic_delta_vs_step0.png", dpi=190)
    plt.close(fig)


def _write_notes(robust: pd.DataFrame, task: pd.DataFrame, windows: pd.DataFrame) -> None:
    train = robust[robust["metric"] == "train_loss_aulc_delta"].iloc[0]
    test = robust[robust["metric"] == "test_loss_curve_aulc_delta"].iloc[0]
    lines = [
        "# Variant A Clean Robustness Checks",
        "",
        "AULC in `downstream_results.csv` is train-loss AULC. Test-loss curve AULC below is recomputed from `downstream_curves.csv`.",
        "",
        "## Robustness",
        "",
        f"- Train-loss AULC mean delta: `{train['mean']:+.6g}`, median `{train['median']:+.6g}`, sign-flip p `{train['sign_flip_p_two_sided']:.6g}`, bootstrap 95% `[{train['bootstrap95_lo']:+.6g}, {train['bootstrap95_hi']:+.6g}]`.",
        f"- Test-loss curve AULC mean delta: `{test['mean']:+.6g}`, median `{test['median']:+.6g}`, sign-flip p `{test['sign_flip_p_two_sided']:.6g}`, bootstrap 95% `[{test['bootstrap95_lo']:+.6g}, {test['bootstrap95_hi']:+.6g}]`.",
        f"- Corr(step0 test-loss delta, train-AULC delta): `{train['corr_with_step0_test_loss_delta']:.6g}`; task-centered `{train['task_centered_corr_with_step0_test_loss_delta']:.6g}`.",
        f"- Test-loss curve AULC mean is decomposed into step0 contribution `{test['step0_mean_contribution_to_test_curve_auc']:+.6g}` and post-step0 contribution `{test['post0_mean_contribution_to_test_curve_auc']:+.6g}`.",
        "",
        "## Task Summary",
        "",
        task.to_markdown(index=False),
        "",
        "## Curve Windows",
        "",
        windows.to_markdown(index=False),
        "",
        "## Plots",
        "",
        "- `robustness_leave_one_out.png`",
        "- `robustness_task_centered_step0_vs_aulc.png`",
        "- `robustness_test_curve_decomposition.png`",
        "- `robustness_diagnostic_delta_vs_step0.png`",
    ]
    (OUT_DIR / "robustness_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in RUNS:
        _assert_decoder_curves_complete(label)
    results = _load_decoder_results()
    curves = _load_decoder_curves()
    robust = _robust_stats(results, curves)
    task = _task_summary(results, curves)
    loo = _leave_one_out(results)
    windows = _window_summary(curves)
    joined = _diagnostic_join(results)
    _plot_influence(loo)
    _plot_task_centered(results)
    _plot_curve_decomposition(robust)
    _plot_diagnostic_join(joined)
    _write_notes(robust, task, windows)
    print(f"[analyze_variant_a_clean_robustness] wrote {OUT_DIR}")
    print(robust.to_string(index=False))


if __name__ == "__main__":
    main()
