"""Per-start optimizer interaction audit for Variant A CH2.

This is a read-only post-hoc audit over existing downstream CSVs.  It tests the
CH2 prediction that replacing latent Adam with latent SGD/momentum should
systematically improve A-vs-control deltas if Adam-in-z is the main mechanism
hiding A's benefit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "rank_repair_m2048_h4096"
)
DEFAULT_MAIN = (
    DEFAULT_BASE
    / "optimizer_panel/selected16_adam_sgd_momentum/paired_downstream_deltas.csv"
)
DEFAULT_BOUNDARY = (
    DEFAULT_BASE
    / "optimizer_panel/selected16_sgd_boundary_lr_0p03_0p3/paired_downstream_deltas.csv"
)
DEFAULT_STEP0 = DEFAULT_BASE / "step0_mediation/step0_mediation_per_start.csv"
DEFAULT_BANK = DEFAULT_BASE / "trajectory_discriminator/selected_16_start_bank.csv"
DEFAULT_OUT = DEFAULT_BASE / "optimizer_panel/ch2_interaction_audit"


KEYS = ["source_weight_index", "task_name", "tau"]
METRICS = [
    "test_mean_loss_delta",
    "test_trapz_loss_delta",
    "post0_test_loss_mean_delta",
    "train_mean_loss_delta",
]
METHOD_ALIASES = {
    "decoder_latent": "adam",
    "decoder_latent_sgd": "sgd_main",
    "decoder_latent_sgd_momentum": "momentum_main",
}
BOUNDARY_ALIASES = {
    "decoder_latent_sgd": "sgd_boundary",
    "decoder_latent_sgd_momentum": "momentum_boundary",
}


def _require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def _wide_by_method(path: Path, aliases: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, KEYS + ["method", "step0_test_loss_delta"] + METRICS, path)
    df = df[df["method"].isin(aliases)].copy()
    df["method_alias"] = df["method"].map(aliases)
    dupes = df.duplicated(KEYS + ["method_alias"], keep=False)
    if dupes.any():
        raise ValueError(f"{path} has duplicate key/method rows")
    value_cols = ["step0_test_loss_delta", "lr"] + METRICS
    wide = df.pivot(index=KEYS, columns="method_alias", values=value_cols)
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    return wide


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    idx = rng.integers(0, values.size, size=(4000, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return (float(lo), float(hi))


def _summarize(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("all", df)] if group_col is None else list(df.groupby(group_col, dropna=False))
    rng = np.random.default_rng(20260709)
    rescue_cols = [
        "rescue_test_mean_sgd_main",
        "rescue_test_mean_momentum_main",
        "rescue_test_mean_sgd_boundary",
        "rescue_test_mean_momentum_boundary",
        "rescue_post0_sgd_main",
        "rescue_post0_momentum_main",
        "rescue_post0_sgd_boundary",
        "rescue_post0_momentum_boundary",
    ]
    delta_cols = [
        "test_mean_loss_delta_adam",
        "test_mean_loss_delta_sgd_main",
        "test_mean_loss_delta_momentum_main",
        "test_mean_loss_delta_sgd_boundary",
        "test_mean_loss_delta_momentum_boundary",
    ]
    for group_name, g in groups:
        row: dict[str, object] = {
            "group_col": group_col or "all",
            "group": str(group_name),
            "n": int(len(g)),
            "step0_mean": float(g["step0_test_loss_delta_adam"].mean()),
            "step0_median": float(g["step0_test_loss_delta_adam"].median()),
        }
        for col in delta_cols + rescue_cols:
            if col not in g.columns:
                continue
            values = g[col].to_numpy(dtype=float)
            lo, hi = _bootstrap_ci(values, rng=rng)
            row[f"{col}_mean"] = float(np.nanmean(values))
            row[f"{col}_median"] = float(np.nanmedian(values))
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
            if col.startswith("rescue_"):
                row[f"{col}_negative_fraction"] = float(np.mean(values < 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _corr_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rescue_cols = [col for col in df.columns if col.startswith("rescue_")]
    targets = ["step0_test_loss_delta_adam", "test_mean_loss_delta_adam"]
    for target in targets:
        for col in rescue_cols:
            sub = df[[target, col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 3:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(sub[target].corr(sub[col], method="pearson"))
                spearman = float(sub[target].corr(sub[col], method="spearman"))
            rows.append(
                {
                    "target": target,
                    "feature": col,
                    "n": int(len(sub)),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def build_audit(
    *,
    main_csv: Path,
    boundary_csv: Path,
    step0_csv: Path,
    start_bank_csv: Path,
    output_dir: Path,
) -> None:
    print("[optimizer_interaction] reading inputs")
    print(f"  main_csv={main_csv}")
    print(f"  boundary_csv={boundary_csv}")
    print(f"  step0_csv={step0_csv}")
    print(f"  start_bank_csv={start_bank_csv}")
    print(f"  output_dir={output_dir}")

    main = _wide_by_method(main_csv, METHOD_ALIASES)
    boundary = _wide_by_method(boundary_csv, BOUNDARY_ALIASES)
    bank = pd.read_csv(start_bank_csv)
    _require_columns(bank, KEYS + ["discriminator_selection", "start_bank_position"], start_bank_csv)
    step0 = pd.read_csv(step0_csv)
    _require_columns(step0, KEYS + ["aulc_delta", "step0_test_loss_delta"], step0_csv)

    merged = main.merge(boundary, on=KEYS, how="inner", validate="one_to_one")
    merged = merged.merge(
        bank[KEYS + ["discriminator_selection", "start_bank_position"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        step0[KEYS + ["aulc_delta", "step0_test_loss_delta"]].rename(
            columns={
                "aulc_delta": "full64_adam_aulc_delta",
                "step0_test_loss_delta": "full64_step0_test_loss_delta",
            }
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    expected_methods = ["adam", "sgd_main", "momentum_main", "sgd_boundary", "momentum_boundary"]
    for method in expected_methods:
        step_col = f"step0_test_loss_delta_{method}"
        if step_col in merged.columns:
            merged[f"step0_minus_adam_{method}"] = (
                merged[step_col] - merged["step0_test_loss_delta_adam"]
            )

    for opt in ["sgd_main", "momentum_main", "sgd_boundary", "momentum_boundary"]:
        merged[f"rescue_test_mean_{opt}"] = (
            merged[f"test_mean_loss_delta_{opt}"] - merged["test_mean_loss_delta_adam"]
        )
        merged[f"rescue_test_trapz_{opt}"] = (
            merged[f"test_trapz_loss_delta_{opt}"] - merged["test_trapz_loss_delta_adam"]
        )
        merged[f"rescue_post0_{opt}"] = (
            merged[f"post0_test_loss_mean_delta_{opt}"]
            - merged["post0_test_loss_mean_delta_adam"]
        )

    merged["step0_sign"] = np.where(
        merged["step0_test_loss_delta_adam"] > 0, "positive_step0", "negative_or_zero_step0"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_start_path = output_dir / "optimizer_interaction_by_start.csv"
    summary_path = output_dir / "optimizer_interaction_summary.csv"
    corr_path = output_dir / "optimizer_interaction_correlations.csv"
    review_path = output_dir / "optimizer_interaction_review.md"

    summaries = [
        _summarize(merged),
        _summarize(merged, "discriminator_selection"),
        _summarize(merged, "step0_sign"),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    corrs = _corr_rows(merged)

    max_step0_mismatch = float(
        np.nanmax(
            np.abs(
                merged[
                    [
                        "step0_minus_adam_sgd_main",
                        "step0_minus_adam_momentum_main",
                        "step0_minus_adam_sgd_boundary",
                        "step0_minus_adam_momentum_boundary",
                    ]
                ].to_numpy(dtype=float)
            )
        )
    )

    merged.to_csv(per_start_path, index=False)
    summary.to_csv(summary_path, index=False)
    corrs.to_csv(corr_path, index=False)

    all_row = summary[(summary["group_col"] == "all") & (summary["group"] == "all")].iloc[0]
    pos_row = summary[
        (summary["group_col"] == "step0_sign") & (summary["group"] == "positive_step0")
    ].iloc[0]
    neg_row = summary[
        (summary["group_col"] == "step0_sign")
        & (summary["group"] == "negative_or_zero_step0")
    ].iloc[0]
    step0_corr = corrs[
        (corrs["target"] == "step0_test_loss_delta_adam")
        & (corrs["feature"] == "rescue_test_mean_sgd_main")
    ].iloc[0]
    review = f"""# CH2 Optimizer Interaction Audit

Scope: read-only per-start audit over the existing current `m=2048,h=4096`
selected-16 optimizer panel. No training was run.

## Inputs

- Main panel: `{main_csv}`
- Boundary LR panel: `{boundary_csv}`
- Selected start bank: `{start_bank_csv}`
- Full64 step0 mediation table: `{step0_csv}`

## Validity Checks

- Joined starts: `{len(merged)}`
- Max step0 mismatch between Adam/SGD/momentum rows: `{max_step0_mismatch:.6g}`
- Output per-start table: `{per_start_path}`
- Output summary: `{summary_path}`
- Output correlations: `{corr_path}`

## CH2 Prediction

If latent Adam is the main mechanism hiding Variant A's useful conditioning
effect, true latent SGD or SGD-momentum should make A-control deltas smaller
than under Adam on the same starts. In this table, `rescue = optimizer_delta -
adam_delta`, so CH2 predicts robustly negative rescue values.

## All-Start Results

- Adam `test_mean_loss_delta` mean:
  `{all_row['test_mean_loss_delta_adam_mean']:.8f}`
- Main-grid SGD rescue mean:
  `{all_row['rescue_test_mean_sgd_main_mean']:.8f}` with negative fraction
  `{all_row['rescue_test_mean_sgd_main_negative_fraction']:.3f}`
- Main-grid momentum rescue mean:
  `{all_row['rescue_test_mean_momentum_main_mean']:.8f}` with negative fraction
  `{all_row['rescue_test_mean_momentum_main_negative_fraction']:.3f}`
- Boundary-grid SGD rescue mean:
  `{all_row['rescue_test_mean_sgd_boundary_mean']:.8f}` with negative fraction
  `{all_row['rescue_test_mean_sgd_boundary_negative_fraction']:.3f}`
- Boundary-grid momentum rescue mean:
  `{all_row['rescue_test_mean_momentum_boundary_mean']:.8f}` with negative
  fraction `{all_row['rescue_test_mean_momentum_boundary_negative_fraction']:.3f}`

## Step0 Stratification

- Positive-step0 starts (`n={int(pos_row['n'])}`): Adam
  `test_mean_loss_delta` mean `{pos_row['test_mean_loss_delta_adam_mean']:.8f}`;
  main-grid SGD rescue `{pos_row['rescue_test_mean_sgd_main_mean']:.8f}`;
  main-grid momentum rescue
  `{pos_row['rescue_test_mean_momentum_main_mean']:.8f}`; boundary-grid SGD
  rescue `{pos_row['rescue_test_mean_sgd_boundary_mean']:.8f}`.
- Negative-or-zero-step0 starts (`n={int(neg_row['n'])}`): Adam
  `test_mean_loss_delta` mean `{neg_row['test_mean_loss_delta_adam_mean']:.8f}`;
  main-grid SGD rescue `{neg_row['rescue_test_mean_sgd_main_mean']:.8f}`;
  main-grid momentum rescue
  `{neg_row['rescue_test_mean_momentum_main_mean']:.8f}`; boundary-grid SGD
  rescue `{neg_row['rescue_test_mean_sgd_boundary_mean']:.8f}`.
- `step0_test_loss_delta_adam` versus main-grid SGD rescue correlation:
  Pearson `{step0_corr['pearson']:.6f}`, Spearman
  `{step0_corr['spearman']:.6f}`.

## Narrow Interpretation

This audit further weakens the strong current-chart CH2 prediction. The starts
where A is worse at step0 are not rescued by latent SGD/momentum; their rescue
values are positive or near zero. The apparent optimizer interaction tracks the
same step0 damage axis, rather than revealing a hidden A benefit under true
latent SGD. This does not close CH2 globally: current `m=2048,h=4096`
natural/projected downstream and PSGD/Kron positive-control runs are still
missing.
"""
    review_path.write_text(review)

    print("[optimizer_interaction] wrote:")
    print(f"  {per_start_path}")
    print(f"  {summary_path}")
    print(f"  {corr_path}")
    print(f"  {review_path}")
    print("[optimizer_interaction] all-start rescue means:")
    for col in [
        "rescue_test_mean_sgd_main_mean",
        "rescue_test_mean_momentum_main_mean",
        "rescue_test_mean_sgd_boundary_mean",
        "rescue_test_mean_momentum_boundary_mean",
    ]:
        print(f"  {col}={all_row[col]:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--boundary-csv", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--step0-csv", type=Path, default=DEFAULT_STEP0)
    parser.add_argument("--start-bank-csv", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_audit(
        main_csv=args.main_csv,
        boundary_csv=args.boundary_csv,
        step0_csv=args.step0_csv,
        start_bank_csv=args.start_bank_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
