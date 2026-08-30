#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("docs/reparam_preconditioning_experiments/variant_A_causal_debug/rank_repair_m2048_h4096")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _finite_array(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return array[np.isfinite(array)]


def _bootstrap_ci(values: pd.Series | np.ndarray, *, seed: int, reps: int = 4000) -> tuple[float, float]:
    array = _finite_array(values)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        draws[idx] = float(np.mean(rng.choice(array, size=array.size, replace=True)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _summary(values: pd.Series | np.ndarray, *, seed: int) -> dict[str, Any]:
    array = _finite_array(values)
    lo, hi = _bootstrap_ci(array, seed=seed)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)) if array.size else float("nan"),
        "median": float(np.median(array)) if array.size else float("nan"),
        "bootstrap95_low": lo,
        "bootstrap95_high": hi,
        "min": float(np.min(array)) if array.size else float("nan"),
        "max": float(np.max(array)) if array.size else float("nan"),
        "positive_count": int(np.sum(array > 0.0)),
        "negative_count": int(np.sum(array < 0.0)),
    }


def _pearson(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = np.isfinite(xs.to_numpy(dtype=np.float64)) & np.isfinite(ys.to_numpy(dtype=np.float64))
    if int(mask.sum()) < 2:
        return float("nan")
    return float(xs[mask].corr(ys[mask], method="pearson"))


def _spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = np.isfinite(xs.to_numpy(dtype=np.float64)) & np.isfinite(ys.to_numpy(dtype=np.float64))
    if int(mask.sum()) < 2:
        return float("nan")
    return float(xs[mask].corr(ys[mask], method="spearman"))


def _ols_residual(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask].astype(np.float64)
    x = x[mask].astype(np.float64)
    if y.size < 3:
        return {
            "n": int(y.size),
            "intercept": float("nan"),
            "slope_step0": float("nan"),
            "r2": float("nan"),
            "residuals": np.full_like(y, np.nan),
        }
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    residuals = y - pred
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    sse = float(np.sum(residuals**2))
    r2 = float(1.0 - sse / sst) if sst > 0.0 else float("nan")
    return {
        "n": int(y.size),
        "intercept": float(beta[0]),
        "slope_step0": float(beta[1]),
        "r2": r2,
        "residuals": residuals,
    }


def _slug_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _make_ceiling(root: Path, output_dir: Path) -> None:
    repair_path = root / "latent_head_clamp/full64_raw_w0_clamp_adam/full64_latent_head_clamp_repair_summary.csv"
    cg_summary_path = root / "clamped_cg_audit/full64_active_lam0p03_0p1_0p3_probes4_cg80/paired_clamped_cg_summary.csv"
    cg_corr_path = root / "clamped_cg_audit/full64_active_lam0p03_0p1_0p3_probes4_cg80/cg_downstream_correlation_summary.csv"
    margin_path = root / "margin_function_audit/full64_current/intervention_summary.csv"

    repair = _read_csv(repair_path)
    cg_summary = _read_csv(cg_summary_path)
    cg_corr = _read_csv(cg_corr_path)
    margin = _read_csv(margin_path)

    rows: list[dict[str, Any]] = []

    keep_methods = {
        "decoder_latent",
        "decoder_latent_raw_classifier_head_clamp",
        "decoder_latent_raw_fc2_weight_clamp",
        "decoder_latent_raw_fc2_bias_clamp",
        "raw",
    }
    keep_metrics = {"test_mean_loss_delta", "step0_test_loss_delta", "post0_test_loss_mean_delta"}
    for _, row in repair[
        repair["method"].astype(str).isin(keep_methods) & repair["metric"].astype(str).isin(keep_metrics)
    ].iterrows():
        rows.append(
            {
                "category": "damage_repair_downstream",
                "method": row["method"],
                "metric": row["metric"],
                "n": int(row["n"]),
                "mean": float(row["mean"]),
                "median": float(row["median"]),
                "bootstrap95_low": float(row["bootstrap95_low"]),
                "bootstrap95_high": float(row["bootstrap95_high"]),
                "rescue_vs_plain_mean_test": row.get("rescue_vs_plain_mean_test", float("nan")),
            }
        )

    keep_cg_methods = {
        "decoder_latent",
        "decoder_latent_raw_classifier_head_clamp",
        "decoder_latent_raw_fc2_weight_clamp",
        "decoder_latent_raw_fc2_bias_clamp",
    }
    keep_cg_metrics = {"c3_forward", "c3_total", "c3_inverse", "trace_jtj_per_dim"}
    cg_filtered = cg_summary[
        (cg_summary["probe_space"].astype(str) == "active")
        & cg_summary["method"].astype(str).isin(keep_cg_methods)
        & cg_summary["metric"].astype(str).isin(keep_cg_metrics)
    ]
    for _, row in cg_filtered.iterrows():
        rows.append(
            {
                "category": "remaining_active_cg",
                "method": row["method"],
                "metric": row["metric"],
                "n": int(row["n"]),
                "mean": float(row["mean_delta"]),
                "median": float(row["median_delta"]),
                "bootstrap95_low": float(row["bootstrap95_delta_low"]),
                "bootstrap95_high": float(row["bootstrap95_delta_high"]),
                "better_count_A_lt_control": int(row["better_count_A_lt_control"]),
                "worse_count_A_gt_control": int(row["worse_count_A_gt_control"]),
            }
        )

    keep_corr_metrics = {"c3_forward_delta", "c3_total_delta", "c3_inverse_delta"}
    keep_downstream = {"test_mean_loss_delta", "step0_test_loss_delta", "post0_test_loss_mean_delta"}
    corr_filtered = cg_corr[
        cg_corr["method"].astype(str).isin(keep_cg_methods)
        & cg_corr["cg_metric"].astype(str).isin(keep_corr_metrics)
        & cg_corr["downstream_metric"].astype(str).isin(keep_downstream)
    ]
    for _, row in corr_filtered.iterrows():
        rows.append(
            {
                "category": "cg_downstream_alignment",
                "method": row["method"],
                "metric": f"{row['cg_metric']}__vs__{row['downstream_metric']}",
                "n": int(row["n_sources"]),
                "mean": float(row["pearson"]),
                "median": float(row["spearman"]),
                "cg_mean": float(row["cg_mean"]),
                "cg_median": float(row["cg_median"]),
                "downstream_mean": float(row["downstream_mean"]),
                "downstream_median": float(row["downstream_median"]),
            }
        )

    keep_candidates = {
        "A_decoded",
        "full_control_classifier_head",
        "full_control_fc2_weight",
        "full_control_fc2_bias",
        "control_row_direction_A_row_scale",
        "A_row_direction_control_row_scale",
    }
    margin_filtered = margin[
        margin["subset"].astype(str).isin({"all", "positive_step0"})
        & margin["candidate"].astype(str).isin(keep_candidates)
    ]
    for _, row in margin_filtered.iterrows():
        rows.append(
            {
                "category": "row_direction_margin_channel",
                "method": f"{row['subset']}:{row['candidate']}",
                "metric": "step0_test_loss_after_intervention",
                "n": int(row["starts"]),
                "mean": float(row["mean_gap_after_candidate"]),
                "median": float(row["median_gap_after_candidate"]),
                "rescue_vs_plain_mean_test": float(row["mean_rescue_fraction"]),
                "median_rescue_fraction": float(row["median_rescue_fraction"]),
                "worse_after_count": int(row["worse_after_count"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "ceiling_summary.csv"
    summary.to_csv(summary_path, index=False)

    def pick_repair(method: str, metric: str) -> pd.Series:
        match = repair[(repair["method"] == method) & (repair["metric"] == metric)]
        if match.empty:
            raise RuntimeError(f"missing repair row method={method} metric={metric}")
        return match.iloc[0]

    def pick_cg(method: str, metric: str) -> pd.Series:
        match = cg_filtered[(cg_filtered["method"] == method) & (cg_filtered["metric"] == metric)]
        if match.empty:
            raise RuntimeError(f"missing cg row method={method} metric={metric}")
        return match.iloc[0]

    def pick_corr(method: str, cg_metric: str, downstream_metric: str) -> pd.Series:
        match = cg_corr[
            (cg_corr["method"] == method)
            & (cg_corr["cg_metric"] == cg_metric)
            & (cg_corr["downstream_metric"] == downstream_metric)
        ]
        if match.empty:
            raise RuntimeError(
                f"missing corr row method={method} cg_metric={cg_metric} downstream_metric={downstream_metric}"
            )
        return match.iloc[0]

    plain_test = pick_repair("decoder_latent", "test_mean_loss_delta")
    plain_step0 = pick_repair("decoder_latent", "step0_test_loss_delta")
    fc2_test = pick_repair("decoder_latent_raw_fc2_weight_clamp", "test_mean_loss_delta")
    fc2_step0 = pick_repair("decoder_latent_raw_fc2_weight_clamp", "step0_test_loss_delta")
    bias_test = pick_repair("decoder_latent_raw_fc2_bias_clamp", "test_mean_loss_delta")
    fc2_c3 = pick_cg("decoder_latent_raw_fc2_weight_clamp", "c3_forward")
    fc2_total = pick_cg("decoder_latent_raw_fc2_weight_clamp", "c3_total")
    fc2_corr = pick_corr("decoder_latent_raw_fc2_weight_clamp", "c3_forward_delta", "test_mean_loss_delta")

    review_lines = [
        "# Pre-Run Ceiling From Existing Clamp/CG Evidence",
        "",
        "Purpose: set falsifiable expectations for the running direction-guard experiment before seeing its downstream result.",
        "",
        "Source artifacts:",
        f"- `{repair_path}`",
        f"- `{cg_summary_path}`",
        f"- `{cg_corr_path}`",
        f"- `{margin_path}`",
        "",
        "Key facts from accepted old `m2048,h4096` artifacts:",
        f"- Plain latent A-control test-mean delta: `{float(plain_test['mean']):.6g}`; step0 delta: `{float(plain_step0['mean']):.6g}`.",
        f"- Raw `fc2.weight` clamp test-mean delta: `{float(fc2_test['mean']):.6g}`; step0 delta: `{float(fc2_step0['mean']):.6g}`; mean rescue vs plain test: `{float(fc2_test['rescue_vs_plain_mean_test']):.6g}`.",
        f"- Raw `fc2.bias` clamp test-mean delta: `{float(bias_test['mean']):.6g}`; it does not rescue the plain harm.",
        f"- After raw `fc2.weight` clamp, active `c3_forward` mean delta is `{float(fc2_c3['mean_delta']):.6g}` with CI `[{float(fc2_c3['bootstrap95_delta_low']):.6g}, {float(fc2_c3['bootstrap95_delta_high']):.6g}]`.",
        f"- After raw `fc2.weight` clamp, active `c3_total` mean delta is `{float(fc2_total['mean_delta']):.6g}` with CI `[{float(fc2_total['bootstrap95_delta_low']):.6g}, {float(fc2_total['bootstrap95_delta_high']):.6g}]`.",
        f"- After raw `fc2.weight` clamp, source-level `c3_forward_delta` vs test-mean delta has Pearson `{float(fc2_corr['pearson']):.6g}` and Spearman `{float(fc2_corr['spearman']):.6g}`.",
        "",
        "Interpretation gate:",
        "- If the direction guard removes row-direction/margin/step0 damage but A only ties the guarded control, that is consistent with the existing ceiling: the damage channel was causal, while residual active exact-`c3` upside was tiny/mixed.",
        "- If the direction guard removes damage and A robustly beats the guarded control, that is stronger than the post-hoc clamp ceiling predicted and needs a follow-up active-CG/trajectory explanation.",
        "- If the direction guard does not remove the row-direction/margin channel, the next step is implementation/target audit, not coefficient sweeping.",
        "",
        f"Machine-readable summary: `{summary_path}`",
    ]
    (output_dir / "ceiling_review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    manifest = {
        "kind": "pre_run_ceiling_from_clamp",
        "inputs": [_file_signature(p) for p in [repair_path, cg_summary_path, cg_corr_path, margin_path]],
        "outputs": [str(summary_path), str(output_dir / "ceiling_review.md")],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _make_ch4_residual(root: Path, output_dir: Path, *, seed: int) -> None:
    curve_path = root / "analysis/rank_repair_decoder_curve_deltas.csv"
    step0_path = root / "step0_mediation/step0_mediation_per_start.csv"
    trajectory_path = root / "trajectory_discriminator/lambda0p1_stress4_steps300_noproject/trajectory_paired_cg_deltas.csv"

    curves = _read_csv(curve_path)
    step0 = _read_csv(step0_path)
    if "source_weight_index" not in curves.columns or "delta_test_loss" not in curves.columns:
        raise RuntimeError(f"{curve_path} does not contain required columns")
    if "source_weight_index" not in step0.columns or "step0_test_loss_delta" not in step0.columns:
        raise RuntimeError(f"{step0_path} does not contain required columns")

    curves = curves[curves["method"].astype(str) == "decoder_latent"].copy()
    curves["step"] = pd.to_numeric(curves["step"], errors="raise").astype(int)
    max_step = int(curves["step"].max())
    post0 = (
        curves[curves["step"] > 0]
        .groupby("source_weight_index", as_index=False)
        .agg(
            post0_curve_mean_delta=("delta_test_loss", "mean"),
            late_ge300_curve_mean_delta=("delta_test_loss", lambda s: float(pd.to_numeric(s, errors="coerce")[curves.loc[s.index, "step"] >= 300].mean())),
        )
    )
    final = curves[curves["step"] == max_step][["source_weight_index", "delta_test_loss"]].rename(
        columns={"delta_test_loss": "final_curve_delta"}
    )
    step0_keep = step0[
        [
            "source_weight_index",
            "task_name",
            "tau",
            "aulc_delta",
            "final_test_loss_delta",
            "step0_test_loss_delta",
        ]
    ].copy()
    joined = step0_keep.merge(post0, on="source_weight_index", how="inner").merge(final, on="source_weight_index", how="inner")

    metric_rows: list[dict[str, Any]] = []
    for metric in [
        "aulc_delta",
        "post0_curve_mean_delta",
        "late_ge300_curve_mean_delta",
        "final_curve_delta",
        "final_test_loss_delta",
    ]:
        y = pd.to_numeric(joined[metric], errors="coerce").to_numpy(dtype=np.float64)
        x = pd.to_numeric(joined["step0_test_loss_delta"], errors="coerce").to_numpy(dtype=np.float64)
        ols = _ols_residual(y, x)
        summ = _summary(joined[metric], seed=seed)
        res_summ = _summary(ols["residuals"], seed=seed + 17)
        metric_rows.append(
            {
                "metric": metric,
                **{f"raw_{k}": v for k, v in summ.items()},
                "pearson_vs_step0": _pearson(joined["step0_test_loss_delta"], joined[metric]),
                "spearman_vs_step0": _spearman(joined["step0_test_loss_delta"], joined[metric]),
                "ols_intercept": ols["intercept"],
                "ols_slope_step0": ols["slope_step0"],
                "ols_r2_step0": ols["r2"],
                **{f"residual_{k}": v for k, v in res_summ.items()},
            }
        )
        joined[f"{metric}_residual_after_step0"] = np.nan
        mask = np.isfinite(y) & np.isfinite(x)
        joined.loc[mask, f"{metric}_residual_after_step0"] = ols["residuals"]

    loo_rows: list[dict[str, Any]] = []
    for source in [None, *joined["source_weight_index"].astype(int).tolist()]:
        if source is None:
            sub = joined.copy()
            label = "none"
        else:
            sub = joined[joined["source_weight_index"].astype(int) != int(source)].copy()
            label = str(int(source))
        for metric in ["aulc_delta", "post0_curve_mean_delta", "late_ge300_curve_mean_delta", "final_curve_delta"]:
            y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=np.float64)
            x = pd.to_numeric(sub["step0_test_loss_delta"], errors="coerce").to_numpy(dtype=np.float64)
            ols = _ols_residual(y, x)
            raw = _summary(sub[metric], seed=seed)
            res = _summary(ols["residuals"], seed=seed + 29)
            loo_rows.append(
                {
                    "left_out_source_weight_index": label,
                    "metric": metric,
                    "n": int(len(sub)),
                    "raw_mean": raw["mean"],
                    "raw_median": raw["median"],
                    "raw_bootstrap95_low": raw["bootstrap95_low"],
                    "raw_bootstrap95_high": raw["bootstrap95_high"],
                    "pearson_vs_step0": _pearson(sub["step0_test_loss_delta"], sub[metric]),
                    "spearman_vs_step0": _spearman(sub["step0_test_loss_delta"], sub[metric]),
                    "ols_r2_step0": ols["r2"],
                    "residual_mean_after_step0": res["mean"],
                    "residual_median_after_step0": res["median"],
                    "residual_bootstrap95_low": res["bootstrap95_low"],
                    "residual_bootstrap95_high": res["bootstrap95_high"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = output_dir / "post0_residual_per_start.csv"
    summary_path = output_dir / "post0_residual_summary.csv"
    loo_path = output_dir / "leave_one_out.csv"
    joined.to_csv(joined_path, index=False)
    pd.DataFrame(metric_rows).to_csv(summary_path, index=False)
    pd.DataFrame(loo_rows).to_csv(loo_path, index=False)

    trajectory_summary_path = output_dir / "trajectory_cg_step_summary.csv"
    trajectory_lines: list[str] = []
    if trajectory_path.is_file():
        traj = _read_csv(trajectory_path)
        traj_summary = (
            traj.groupby("step", as_index=False)
            .agg(
                n=("source_weight_index", "count"),
                c3_forward_rel_delta_mean=("c3_forward_rel_delta", "mean"),
                c3_forward_rel_delta_median=("c3_forward_rel_delta", "median"),
                c3_forward_delta_mean=("c3_forward_delta", "mean"),
                c3_forward_delta_median=("c3_forward_delta", "median"),
                normal_residual_fraction_control_mean=("normal_residual_fraction_control", "mean"),
                normal_residual_fraction_a_mean=("normal_residual_fraction_a", "mean"),
                raw_adam_projection_residual_control_mean=("fresh_raw_adam_projection_residual_fraction_control", "mean"),
                raw_adam_projection_residual_a_mean=("fresh_raw_adam_projection_residual_fraction_a", "mean"),
            )
        )
        traj_summary.to_csv(trajectory_summary_path, index=False)
        trajectory_lines = [
            "",
            "Stress4 trajectory CG summary:",
            f"- `{trajectory_summary_path}`",
        ]
    else:
        pd.DataFrame().to_csv(trajectory_summary_path, index=False)

    summary_df = pd.DataFrame(metric_rows)
    aulc = summary_df[summary_df["metric"] == "aulc_delta"].iloc[0]
    post0_row = summary_df[summary_df["metric"] == "post0_curve_mean_delta"].iloc[0]
    late = summary_df[summary_df["metric"] == "late_ge300_curve_mean_delta"].iloc[0]
    final_row = summary_df[summary_df["metric"] == "final_curve_delta"].iloc[0]
    loo = pd.read_csv(loo_path)
    loo_2295 = loo[(loo["left_out_source_weight_index"].astype(str) == "2295") & (loo["metric"] == "aulc_delta")]
    loo_2295_text = ""
    if not loo_2295.empty:
        row = loo_2295.iloc[0]
        loo_2295_text = (
            f"- Excluding source `2295`, AULC raw mean is `{float(row['raw_mean']):.6g}` "
            f"with residual mean after step0 `{float(row['residual_mean_after_step0']):.6g}`."
        )

    review_lines = [
        "# CH4 Post0 Residual Screen",
        "",
        "Purpose: test whether the accepted old `m2048,h4096` branch already shows a systematic post-step0/offline residual after conditioning on decoded-start damage.",
        "",
        "Source artifacts:",
        f"- `{curve_path}`",
        f"- `{step0_path}`",
        f"- `{trajectory_path}`",
        "",
        "Key results:",
        f"- AULC delta vs step0: Pearson `{float(aulc['pearson_vs_step0']):.6g}`, Spearman `{float(aulc['spearman_vs_step0']):.6g}`, OLS R2 `{float(aulc['ols_r2_step0']):.6g}`.",
        f"- Post0 curve-mean delta raw mean `{float(post0_row['raw_mean']):.6g}` with CI `[{float(post0_row['raw_bootstrap95_low']):.6g}, {float(post0_row['raw_bootstrap95_high']):.6g}]`; residual mean after step0 `{float(post0_row['residual_mean']):.6g}`.",
        f"- Late `step>=300` curve-mean delta raw mean `{float(late['raw_mean']):.6g}` with CI `[{float(late['raw_bootstrap95_low']):.6g}, {float(late['raw_bootstrap95_high']):.6g}]`; residual mean after step0 `{float(late['residual_mean']):.6g}`.",
        f"- Final-step curve delta raw mean `{float(final_row['raw_mean']):.6g}` with CI `[{float(final_row['raw_bootstrap95_low']):.6g}, {float(final_row['raw_bootstrap95_high']):.6g}]`; residual mean after step0 `{float(final_row['residual_mean']):.6g}`.",
    ]
    if loo_2295_text:
        review_lines.append(loo_2295_text)
    review_lines.extend(
        [
            "",
            "Interpretation gate:",
            "- If post0/late means and residual means are near zero while AULC is explained by step0, CH4 is not the main mechanism for the accepted old `m2048,h4096` failure.",
            "- This does not exclude broader online/trajectory mismatch mechanisms for future guarded runs or direct Li/Kron methods.",
            "",
            f"Machine-readable outputs: `{summary_path}`, `{loo_path}`, `{joined_path}`.",
            *trajectory_lines,
        ]
    )
    (output_dir / "ch4_residual_review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    manifest = {
        "kind": "pre_run_ch4_post0_residual_screen",
        "inputs": [_file_signature(p) for p in [curve_path, step0_path] + ([trajectory_path] if trajectory_path.is_file() else [])],
        "outputs": [str(summary_path), str(loo_path), str(joined_path), str(output_dir / "ch4_residual_review.md"), str(trajectory_summary_path)],
        "max_step": max_step,
        "seed": int(seed),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pre-run causal gates from accepted m2048/h4096 artifacts.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ceiling-output-dir", type=Path, default=DEFAULT_ROOT / "direction_guard/pre_run_ceiling_from_clamp")
    parser.add_argument("--ch4-output-dir", type=Path, default=DEFAULT_ROOT / "direction_guard/pre_run_ch4_residual")
    parser.add_argument("--seed", type=int, default=90210)
    args = parser.parse_args()

    root = args.root.resolve()
    ceiling_dir = args.ceiling_output_dir.resolve()
    ch4_dir = args.ch4_output_dir.resolve()
    print(f"[pre_run_gates] root={root}", flush=True)
    print(f"[pre_run_gates] ceiling_output_dir={ceiling_dir}", flush=True)
    print(f"[pre_run_gates] ch4_output_dir={ch4_dir}", flush=True)

    _make_ceiling(root, ceiling_dir)
    _make_ch4_residual(root, ch4_dir, seed=int(args.seed))
    print(f"[pre_run_gates] wrote {ceiling_dir / 'ceiling_review.md'}", flush=True)
    print(f"[pre_run_gates] wrote {ch4_dir / 'ch4_residual_review.md'}", flush=True)


if __name__ == "__main__":
    main()
