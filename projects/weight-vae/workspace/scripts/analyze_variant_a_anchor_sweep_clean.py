from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
OUT_DIR = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/clean_harness_analysis/anchor_sweep"

RUNS = {
    "control_clean": "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
    "a_clean": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
    "control_anchor_1e-4": "sage_cnn_vae_smoothing_celo_meta_control_logit_anchor_c0p0001_clean_harness_v1_seed0",
    "a_anchor_1e-4": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_logit_anchor_c0p0001_clean_harness_v1_seed0",
    "control_anchor_3e-4": "sage_cnn_vae_smoothing_celo_meta_control_logit_anchor_c0p0003_clean_harness_v1_seed0",
    "a_anchor_3e-4": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_logit_anchor_c0p0003_clean_harness_v1_seed0",
}

DISPLAY = {
    "control_clean": "control c=0",
    "a_clean": "A c=0",
    "control_anchor_1e-4": "control c=1e-4",
    "a_anchor_1e-4": "A c=1e-4",
    "control_anchor_3e-4": "control c=3e-4",
    "a_anchor_3e-4": "A c=3e-4",
}

PAIRS = [
    ("c=0", 0.0, "control_clean", "a_clean"),
    ("c=1e-4", 1e-4, "control_anchor_1e-4", "a_anchor_1e-4"),
    ("c=3e-4", 3e-4, "control_anchor_3e-4", "a_anchor_3e-4"),
]

KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]
CURVE_KEY_COLS = KEY_COLS + ["step"]
REQUIRED_FILES = [
    "artifact_manifest.json",
    "config.json",
    "downstream_curves.csv",
    "downstream_results.csv",
    "downstream_start_bank.csv",
    "geometry.csv",
    "preconditioning_diagnostics.csv",
    "selected_lrs.csv",
    "vae_metrics.csv",
]


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


def _assert_downstream_complete(label: str, method: str) -> None:
    results = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    curves = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    result_rows = results[(results["split"].astype(str) == "eval") & (results["method"].astype(str) == method)].copy()
    curve_rows = curves[(curves["split"].astype(str) == "eval") & (curves["method"].astype(str) == method)].copy()
    if result_rows.empty:
        raise RuntimeError(f"{label} method={method} has no eval downstream_results rows")
    if curve_rows.empty:
        raise RuntimeError(f"{label} method={method} has no eval downstream_curves rows")
    if "diverged" in result_rows and bool(_is_true(result_rows["diverged"]).any()):
        bad = result_rows.loc[_is_true(result_rows["diverged"]), KEY_COLS].to_dict("records")
        raise RuntimeError(f"{label} method={method} has diverged eval result rows: {bad[:5]}")
    if "diverged" in curve_rows and bool(_is_true(curve_rows["diverged"]).any()):
        bad = curve_rows.loc[_is_true(curve_rows["diverged"]), CURVE_KEY_COLS].to_dict("records")
        raise RuntimeError(f"{label} method={method} has diverged eval curve rows: {bad[:5]}")

    result_keys = result_rows[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    curve_keys = curve_rows[KEY_COLS].drop_duplicates().sort_values(KEY_COLS).reset_index(drop=True)
    if not result_keys.equals(curve_keys):
        raise RuntimeError(f"{label} method={method} downstream_results keys do not match curve keys")

    expected_steps = _expected_curve_steps(label)
    bad_groups: list[dict[str, object]] = []
    for key, group in curve_rows.groupby(KEY_COLS, sort=False):
        steps = pd.to_numeric(group["step"], errors="coerce").dropna().astype(int).tolist()
        if sorted(steps) != expected_steps:
            bad_groups.append({"key": key, "steps": sorted(steps)})
    if bad_groups:
        raise RuntimeError(
            f"{label} method={method} has incomplete eval curves; "
            f"expected_steps={expected_steps}, examples={bad_groups[:3]}"
        )


def _check_complete(label: str) -> None:
    run_dir = _run_dir(label)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{label} missing required files: {missing}")
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    status = manifest.get("summary", {}).get("status")
    if status != "complete":
        raise RuntimeError(f"{label} manifest status={status!r}, expected complete")
    _assert_downstream_complete(label, "raw")
    _assert_downstream_complete(label, "decoder_latent")


def _load_results(label: str, method: str = "decoder_latent") -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_results.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _load_curves(label: str, method: str = "decoder_latent") -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "downstream_curves.csv")
    rows = rows[(rows["split"].astype(str) == "eval") & (rows["method"].astype(str) == method)].copy()
    rows["label"] = label
    return rows


def _selected_lr(label: str, method: str) -> float:
    rows = pd.read_csv(_run_dir(label) / "selected_lrs.csv")
    selected = pd.to_numeric(rows["selected"], errors="coerce") == 1
    hit = rows[(rows["method"].astype(str) == method) & selected]
    if len(hit) != 1:
        raise RuntimeError(f"{label} method={method} has {len(hit)} selected LR rows, expected exactly one")
    return float(hit["candidate_lr"].iloc[0])


def _train_history(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    rows = rows[record_type != "vae_quality"].copy()
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    return rows[rows["step"].notna()].copy()


def _quality_rows(label: str) -> pd.DataFrame:
    rows = pd.read_csv(_run_dir(label) / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    return rows[record_type == "vae_quality"].copy()


def _numeric_median(rows: pd.DataFrame, col: str) -> float:
    if col not in rows:
        return float("nan")
    vals = pd.to_numeric(rows[col], errors="coerce").dropna()
    return float(vals.median()) if not vals.empty else float("nan")


def _numeric_final(rows: pd.DataFrame, col: str) -> float:
    if col not in rows:
        return float("nan")
    vals = pd.to_numeric(rows[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _run_summary(label: str) -> dict[str, float | int | str]:
    decoder = _load_results(label, "decoder_latent")
    raw = _load_results(label, "raw")
    curves = _load_curves(label, "decoder_latent")
    step0 = curves[curves["step"] == 0]
    train = _train_history(label)
    quality = _quality_rows(label)
    diag = pd.read_csv(_run_dir(label) / "preconditioning_diagnostics.csv")
    geom = pd.read_csv(_run_dir(label) / "geometry.csv").iloc[0]

    row: dict[str, float | int | str] = {
        "label": label,
        "display": DISPLAY[label],
        "run_label": RUNS[label],
        "n_eval_starts": int(len(decoder)),
        "selected_raw_lr": _selected_lr(label, "raw"),
        "selected_decoder_lr": _selected_lr(label, "decoder_latent"),
        "raw_train_aulc_mean": float(pd.to_numeric(raw["aulc"], errors="coerce").mean()),
        "raw_train_aulc_median": float(pd.to_numeric(raw["aulc"], errors="coerce").median()),
        "decoder_train_aulc_mean": float(pd.to_numeric(decoder["aulc"], errors="coerce").mean()),
        "decoder_train_aulc_median": float(pd.to_numeric(decoder["aulc"], errors="coerce").median()),
        "decoder_step0_test_loss_mean": float(pd.to_numeric(step0["test_loss"], errors="coerce").mean()),
        "decoder_step0_test_loss_median": float(pd.to_numeric(step0["test_loss"], errors="coerce").median()),
        "decoder_final_test_loss_mean": float(pd.to_numeric(decoder["final_test_loss"], errors="coerce").mean()),
        "decoder_final_test_loss_median": float(pd.to_numeric(decoder["final_test_loss"], errors="coerce").median()),
        "decoder_reconstruction_rel_l2_mean": float(pd.to_numeric(decoder["reconstruction_rel_l2"], errors="coerce").mean()),
        "decoder_reconstruction_rel_l2_median": float(
            pd.to_numeric(decoder["reconstruction_rel_l2"], errors="coerce").median()
        ),
        "train_recon_mse_final": _numeric_final(train, "train_recon_mse"),
        "val_recon_mse_final": _numeric_final(train, "val_recon_mse"),
        "quality_reconstruction_rel_l2_median": _numeric_median(quality, "reconstruction_rel_l2"),
        "train_function_anchor_loss_median": _numeric_median(train, "train_function_anchor_loss"),
        "train_function_anchor_effective_loss_median": _numeric_median(train, "train_function_anchor_effective_loss"),
        "train_function_anchor_ce_delta_median": _numeric_median(train, "train_function_anchor_ce_delta"),
        "train_function_anchor_acc_delta_median": _numeric_median(train, "train_function_anchor_acc_delta"),
        "train_precond_effective_loss_median": _numeric_median(train, "train_precond_effective_loss"),
        "train_precond_grad_norm_median": _numeric_median(train, "train_precond_grad_norm"),
        "train_precond_base_grad_norm_median": _numeric_median(train, "train_precond_base_grad_norm"),
        "isometry_objective": float(geom["isometry_objective"]),
        "condition_median": float(geom["condition_median"]),
        "log_eig_spread_median": float(geom["log_eig_spread_median"]),
    }
    for col in [
        "li_A_full_per_dim",
        "hvp_probe_a_loss_per_dim",
        "hvp_probe_trace_m_per_dim",
        "hessian_log_abs_var",
        "hessian_log_abs_spread",
        "hessian_negative_fraction",
    ]:
        vals = pd.to_numeric(diag[col], errors="coerce").dropna()
        row[f"{col}_median"] = float(vals.median())
        row[f"{col}_p95"] = float(vals.quantile(0.95))
    return row


def _assert_same_eval_keys(label: str, baseline_label: str, method: str = "decoder_latent") -> None:
    rows = _load_results(label, method)[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    base = _load_results(baseline_label, method)[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    if not rows.equals(base):
        raise RuntimeError(f"{label} eval keys differ from {baseline_label} for method={method}")


def _compare_to(label: str, baseline_label: str, comparison: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _assert_same_eval_keys(label, baseline_label, "decoder_latent")
    rows = _load_results(label, "decoder_latent")
    base = _load_results(baseline_label, "decoder_latent")
    base_key = base[KEY_COLS + ["aulc", "final_test_loss", "final_test_acc", "reconstruction_rel_l2", "lr"]].rename(
        columns={
            "aulc": "baseline_train_aulc",
            "final_test_loss": "baseline_final_test_loss",
            "final_test_acc": "baseline_final_test_acc",
            "reconstruction_rel_l2": "baseline_reconstruction_rel_l2",
            "lr": "baseline_lr",
        }
    )
    merged = rows.merge(base_key, on=KEY_COLS, how="left", validate="one_to_one")
    if int(merged["baseline_train_aulc"].notna().sum()) != int(len(merged)):
        raise RuntimeError(f"{label} downstream_results merge failed vs {baseline_label}")
    merged["comparison"] = comparison
    merged["baseline_label"] = baseline_label
    merged["delta_train_aulc"] = merged["aulc"] - merged["baseline_train_aulc"]
    merged["delta_final_test_loss"] = merged["final_test_loss"] - merged["baseline_final_test_loss"]
    merged["delta_final_test_acc"] = merged["final_test_acc"] - merged["baseline_final_test_acc"]
    merged["delta_reconstruction_rel_l2"] = merged["reconstruction_rel_l2"] - merged["baseline_reconstruction_rel_l2"]

    curves = _load_curves(label, "decoder_latent")
    base_curves = _load_curves(baseline_label, "decoder_latent")
    base_curve_key = base_curves[CURVE_KEY_COLS + ["train_loss", "test_loss", "test_acc"]].rename(
        columns={
            "train_loss": "baseline_train_loss",
            "test_loss": "baseline_test_loss",
            "test_acc": "baseline_test_acc",
        }
    )
    curve_merged = curves.merge(base_curve_key, on=CURVE_KEY_COLS, how="left", validate="one_to_one")
    if int(curve_merged["baseline_test_loss"].notna().sum()) != int(len(curve_merged)):
        raise RuntimeError(f"{label} downstream_curves merge failed vs {baseline_label}")
    curve_merged["comparison"] = comparison
    curve_merged["baseline_label"] = baseline_label
    curve_merged["delta_train_loss"] = curve_merged["train_loss"] - curve_merged["baseline_train_loss"]
    curve_merged["delta_test_loss"] = curve_merged["test_loss"] - curve_merged["baseline_test_loss"]
    curve_merged["delta_test_acc"] = curve_merged["test_acc"] - curve_merged["baseline_test_acc"]

    n_steps = int(curve_merged["step"].nunique())
    curve_auc = curve_merged.groupby(KEY_COLS, as_index=False).agg(
        test_curve_aulc_delta=("delta_test_loss", "mean"),
        train_curve_aulc_delta=("delta_train_loss", "mean"),
        step0_test_loss_delta=("delta_test_loss", "first"),
        step0_train_loss_delta=("delta_train_loss", "first"),
    )
    curve_auc["step0_test_curve_contribution"] = curve_auc["step0_test_loss_delta"] / max(n_steps, 1)
    curve_auc["post0_test_curve_contribution"] = (
        curve_auc["test_curve_aulc_delta"] - curve_auc["step0_test_curve_contribution"]
    )
    curve_auc = curve_auc.rename(
        columns={
            "step0_test_loss_delta": "delta_step0_test_loss",
            "step0_train_loss_delta": "delta_step0_train_loss",
        }
    )
    merged = merged.merge(curve_auc, on=KEY_COLS, how="left", validate="one_to_one")
    curve_merged = curve_merged.merge(
        curve_auc[KEY_COLS + ["delta_step0_test_loss", "delta_step0_train_loss"]],
        on=KEY_COLS,
        how="left",
        validate="many_to_one",
    )
    curve_merged["step0_gap_closed_test_loss"] = curve_merged["delta_step0_test_loss"] - curve_merged["delta_test_loss"]
    return merged, curve_merged


def _exact_sign_flip_pvalue(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(vals) == 0 or len(vals) > 20:
        return float("nan")
    observed = abs(float(vals.mean()))
    total = 0
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(vals)):
        total += 1
        candidate = abs(float((vals * np.asarray(signs, dtype=np.float64)).mean()))
        if candidate >= observed - 1e-15:
            count += 1
    return count / max(total, 1)


def _bootstrap_ci(values: pd.Series, *, seed: int = 0, samples: int = 50_000) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(samples, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _comparison_summary(rows: pd.DataFrame, run_summary: pd.DataFrame) -> dict[str, float | int | str]:
    label = str(rows["label"].iloc[0])
    baseline_label = str(rows["baseline_label"].iloc[0])
    train_aulc = pd.to_numeric(rows["delta_train_aulc"], errors="coerce")
    test_curve = pd.to_numeric(rows["test_curve_aulc_delta"], errors="coerce")
    step0 = pd.to_numeric(rows["delta_step0_test_loss"], errors="coerce")
    lo, hi = _bootstrap_ci(train_aulc)
    test_lo, test_hi = _bootstrap_ci(test_curve)
    run_idx = run_summary.set_index("label")
    return {
        "comparison": str(rows["comparison"].iloc[0]),
        "label": label,
        "display": DISPLAY[label],
        "baseline_label": baseline_label,
        "baseline_display": DISPLAY[baseline_label],
        "n_eval_starts": int(len(rows)),
        "selected_decoder_lr": float(run_idx.loc[label, "selected_decoder_lr"]),
        "baseline_selected_decoder_lr": float(run_idx.loc[baseline_label, "selected_decoder_lr"]),
        "mean_delta_train_aulc": float(train_aulc.mean()),
        "median_delta_train_aulc": float(train_aulc.median()),
        "frac_better_train_aulc": float((train_aulc < 0).mean()),
        "sign_flip_p_train_aulc": _exact_sign_flip_pvalue(train_aulc),
        "bootstrap95_train_aulc_lo": lo,
        "bootstrap95_train_aulc_hi": hi,
        "mean_delta_test_curve_aulc": float(test_curve.mean()),
        "median_delta_test_curve_aulc": float(test_curve.median()),
        "frac_better_test_curve_aulc": float((test_curve < 0).mean()),
        "sign_flip_p_test_curve_aulc": _exact_sign_flip_pvalue(test_curve),
        "bootstrap95_test_curve_aulc_lo": test_lo,
        "bootstrap95_test_curve_aulc_hi": test_hi,
        "mean_delta_step0_test_loss": float(step0.mean()),
        "median_delta_step0_test_loss": float(step0.median()),
        "mean_step0_test_curve_contribution": float(rows["step0_test_curve_contribution"].mean()),
        "mean_post0_test_curve_contribution": float(rows["post0_test_curve_contribution"].mean()),
        "corr_step0_vs_train_aulc_delta": float(step0.corr(train_aulc)) if len(rows) >= 3 else float("nan"),
        "corr_step0_vs_test_curve_aulc_delta": float(step0.corr(test_curve)) if len(rows) >= 3 else float("nan"),
        "mean_delta_final_test_loss": float(pd.to_numeric(rows["delta_final_test_loss"], errors="coerce").mean()),
        "mean_delta_reconstruction_rel_l2": float(
            pd.to_numeric(rows["delta_reconstruction_rel_l2"], errors="coerce").mean()
        ),
        "delta_li_A_full_per_dim_p95_run_level": float(
            run_idx.loc[label, "li_A_full_per_dim_p95"] - run_idx.loc[baseline_label, "li_A_full_per_dim_p95"]
        ),
        "delta_hvp_trace_m_per_dim_median_run_level": float(
            run_idx.loc[label, "hvp_probe_trace_m_per_dim_median"]
            - run_idx.loc[baseline_label, "hvp_probe_trace_m_per_dim_median"]
        ),
        "delta_decoder_reconstruction_rel_l2_median_run_level": float(
            run_idx.loc[label, "decoder_reconstruction_rel_l2_median"]
            - run_idx.loc[baseline_label, "decoder_reconstruction_rel_l2_median"]
        ),
    }


def _technical_checks() -> pd.DataFrame:
    rows: list[dict[str, int | str | bool]] = []
    clean_bank = pd.read_csv(_run_dir("control_clean") / "downstream_start_bank.csv")
    clean_cols = list(clean_bank.columns)
    clean_hash = _train_history("control_clean")[["step", "train_vae_batch_index_hash"]].dropna()
    for label in RUNS:
        bank = pd.read_csv(_run_dir(label) / "downstream_start_bank.csv")
        cols = [col for col in clean_cols if col in bank.columns]
        hash_rows = _train_history(label)[["step", "train_vae_batch_index_hash"]].dropna()
        hash_merged = clean_hash.merge(hash_rows, on="step", suffixes=("_clean_control", "_candidate"))
        mismatch = (
            hash_merged["train_vae_batch_index_hash_clean_control"]
            != hash_merged["train_vae_batch_index_hash_candidate"]
        )
        rows.append(
            {
                "label": label,
                "display": DISPLAY[label],
                "start_bank_rows": int(len(bank)),
                "start_bank_common_cols_equal_to_clean_control": bool(clean_bank[cols].equals(bank[cols])),
                "common_batch_hash_steps_vs_clean_control": int(len(hash_merged)),
                "common_batch_hash_mismatches_vs_clean_control": int(mismatch.sum()),
                "common_batch_hashes_match_clean_control": bool(not mismatch.any()),
            }
        )
    return pd.DataFrame(rows)


def _plot_run_summary(summary: pd.DataFrame) -> None:
    order = list(RUNS)
    idx = summary.set_index("label")
    x = np.arange(len(order))
    colors = ["#4c78a8", "#f58518", "#72b7b2", "#e45756", "#54a24b", "#b279a2"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    panels = [
        ("decoder_train_aulc_mean", "decoder train-AULC mean", False),
        ("decoder_step0_test_loss_mean", "decoded step0 test loss", False),
        ("decoder_reconstruction_rel_l2_median", "recon rel-L2 median", False),
        ("li_A_full_per_dim_p95", "li_A / dim p95", True),
        ("hvp_probe_trace_m_per_dim_median", "trace(M) / dim median", True),
        ("train_function_anchor_ce_delta_median", "anchor CE delta median", False),
    ]
    for ax, (col, title, logy) in zip(axes.ravel(), panels, strict=True):
        vals = [float(idx.loc[label, col]) for label in order]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[label] for label in order], rotation=24, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if logy:
            ax.set_yscale("log")
        for i, value in enumerate(vals):
            if np.isfinite(value):
                ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=7)
    fig.savefig(OUT_DIR / "anchor_sweep_run_summary.png", dpi=190)
    plt.close(fig)


def _plot_pair_deltas(pair_summary: pd.DataFrame) -> None:
    order = [pair[0] for pair in PAIRS]
    rows = pair_summary.set_index("pair")
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    panels = [
        ("mean_delta_train_aulc", "A - matched control train-AULC"),
        ("mean_delta_test_curve_aulc", "A - matched control test-curve AULC"),
        ("mean_delta_step0_test_loss", "A - matched control step0 test loss"),
        ("delta_li_A_full_per_dim_p95_run_level", "A - matched control li_A p95"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels, strict=True):
        vals = [float(rows.loc[pair, col]) for pair in order]
        ax.bar(x, vals, color=["#f58518", "#e45756", "#b279a2"])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(vals):
            va = "bottom" if value >= 0 else "top"
            ax.text(i, value, f"{value:.3g}", ha="center", va=va, fontsize=8)
    fig.savefig(OUT_DIR / "anchor_sweep_pair_deltas.png", dpi=190)
    plt.close(fig)


def _plot_absolute_curves(abs_curves: pd.DataFrame, abs_summary: pd.DataFrame) -> None:
    labels = [label for label in RUNS if label != "control_clean"]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for i, label in enumerate(labels):
        rows = abs_curves[abs_curves["label"] == label]
        grouped = rows.groupby("step", as_index=False).agg(
            mean_delta=("delta_test_loss", "mean"),
            sem_delta=(
                "delta_test_loss",
                lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=1) / np.sqrt(max(1, s.notna().sum()))),
            ),
            mean_closed=("step0_gap_closed_test_loss", "mean"),
        )
        color = cmap(i)
        axes[0].plot(grouped["step"], grouped["mean_delta"], marker="o", markersize=3, color=color, label=DISPLAY[label])
        lo = grouped["mean_delta"] - grouped["sem_delta"]
        hi = grouped["mean_delta"] + grouped["sem_delta"]
        axes[0].fill_between(grouped["step"].to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), alpha=0.12, color=color)
        axes[1].plot(grouped["step"], grouped["mean_closed"], marker="o", markersize=3, color=color, label=DISPLAY[label])
    for ax, ylabel, title in [
        (axes[0], "mean test-loss delta vs control c=0", "absolute test-loss gap"),
        (axes[1], "mean step0 gap closed", "post-step0 catch-up"),
    ]:
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("downstream step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "anchor_sweep_absolute_curves.png", dpi=190)
    plt.close(fig)

    order = labels
    rows = abs_summary.set_index("label")
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    panels = [
        ("mean_delta_step0_test_loss", "step0 test-loss delta vs control c=0"),
        ("mean_delta_test_curve_aulc", "test-curve AULC delta vs control c=0"),
    ]
    for ax, (col, title) in zip(axes, panels, strict=True):
        vals = [float(rows.loc[label, col]) for label in order]
        ax.bar(x, vals, color=[cmap(i) for i in range(len(order))])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[label] for label in order], rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(vals):
            va = "bottom" if value >= 0 else "top"
            ax.text(i, value, f"{value:.3g}", ha="center", va=va, fontsize=8)
    fig.savefig(OUT_DIR / "anchor_sweep_absolute_bars.png", dpi=190)
    plt.close(fig)


def _plot_step0_scatter(pair_rows: pd.DataFrame, abs_rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, (pair, _, _, a_label) in enumerate(PAIRS):
        rows = pair_rows[(pair_rows["pair"] == pair) & (pair_rows["label"] == a_label)]
        axes[0].scatter(rows["delta_step0_test_loss"], rows["test_curve_aulc_delta"], s=58, alpha=0.82, color=cmap(i), label=pair)
        for _, row in rows.nlargest(2, "test_curve_aulc_delta").iterrows():
            axes[0].annotate(
                str(int(row["source_weight_index"])),
                (float(row["delta_step0_test_loss"]), float(row["test_curve_aulc_delta"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    for i, label in enumerate([label for label in RUNS if label != "control_clean"]):
        rows = abs_rows[abs_rows["label"] == label]
        axes[1].scatter(
            rows["delta_step0_test_loss"],
            rows["test_curve_aulc_delta"],
            s=48,
            alpha=0.78,
            color=cmap(i),
            label=DISPLAY[label],
        )
    for ax, title in [(axes[0], "matched controls"), (axes[1], "absolute vs control c=0")]:
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("decoded step0 test-loss delta")
        ax.set_ylabel("decoder test-curve AULC delta")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "anchor_sweep_step0_vs_aulc.png", dpi=190)
    plt.close(fig)


def _plot_tail_curves(abs_rows: pd.DataFrame) -> None:
    a3 = abs_rows[abs_rows["label"] == "a_anchor_3e-4"].copy()
    if a3.empty:
        a3 = abs_rows[abs_rows["label"] == "a_anchor_1e-4"].copy()
    source_indices = a3.nlargest(4, "test_curve_aulc_delta")["source_weight_index"].astype(int).tolist()
    labels = list(RUNS)
    all_curves = pd.concat([_load_curves(label, "decoder_latent") for label in labels], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for ax, source_idx in zip(axes.ravel(), source_indices, strict=False):
        for i, label in enumerate(labels):
            rows = all_curves[
                (all_curves["label"] == label)
                & (pd.to_numeric(all_curves["source_weight_index"], errors="coerce").astype("Int64") == int(source_idx))
            ].sort_values("step")
            if rows.empty:
                continue
            task = str(rows["task_name"].iloc[0])
            ax.plot(rows["step"], rows["test_loss"], marker="o", markersize=2.5, linewidth=1.3, color=cmap(i), label=DISPLAY[label])
            ax.set_title(f"source_weight_index={source_idx}, task={task}")
        ax.set_xlabel("downstream step")
        ax.set_ylabel("decoder test loss")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "anchor_sweep_tail_curves.png", dpi=190)
    plt.close(fig)


def _plot_vae_training() -> None:
    labels = list(RUNS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate(labels):
        rows = _train_history(label).sort_values("step")
        color = cmap(i)
        axes[0].plot(rows["step"], pd.to_numeric(rows["val_recon_mse"], errors="coerce"), color=color, label=DISPLAY[label])
        axes[1].plot(
            rows["step"],
            pd.to_numeric(rows.get("train_precond_effective_loss", np.nan), errors="coerce"),
            color=color,
            label=DISPLAY[label],
        )
        axes[2].plot(
            rows["step"],
            pd.to_numeric(rows.get("train_function_anchor_effective_loss", np.nan), errors="coerce"),
            color=color,
            label=DISPLAY[label],
        )
    axes[0].set_title("validation reconstruction MSE")
    axes[1].set_title("preconditioner effective loss")
    axes[2].set_title("function-anchor effective loss")
    for ax in axes:
        ax.set_xlabel("VAE step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[2].set_yscale("symlog", linthresh=1e-8)
    fig.savefig(OUT_DIR / "anchor_sweep_vae_training.png", dpi=190)
    plt.close(fig)


def _write_notes(
    run_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    abs_summary: pd.DataFrame,
    tech: pd.DataFrame,
) -> None:
    pair_lines = []
    for _, row in pair_summary.iterrows():
        pair_lines.append(
            "- "
            f"{row['pair']}: train-AULC delta A-control `{row['mean_delta_train_aulc']:+.6g}` "
            f"(median `{row['median_delta_train_aulc']:+.6g}`, p `{row['sign_flip_p_train_aulc']:.4g}`), "
            f"test-curve AULC delta `{row['mean_delta_test_curve_aulc']:+.6g}`, "
            f"step0 delta `{row['mean_delta_step0_test_loss']:+.6g}`, "
            f"post0 contribution `{row['mean_post0_test_curve_contribution']:+.6g}`, "
            f"li_A p95 delta `{row['delta_li_A_full_per_dim_p95_run_level']:+.6g}`."
        )
    abs_lines = []
    for _, row in abs_summary.iterrows():
        abs_lines.append(
            "- "
            f"{row['display']} vs control c=0: train-AULC delta `{row['mean_delta_train_aulc']:+.6g}`, "
            f"test-curve AULC delta `{row['mean_delta_test_curve_aulc']:+.6g}`, "
            f"step0 delta `{row['mean_delta_step0_test_loss']:+.6g}`."
        )
    failed = tech[
        (~tech["start_bank_common_cols_equal_to_clean_control"])
        | (tech["common_batch_hash_mismatches_vs_clean_control"].astype(int) != 0)
        | (tech["common_batch_hash_steps_vs_clean_control"].astype(int) == 0)
    ]
    checks_line = "all passed" if failed.empty else f"FAILED labels: {failed['label'].tolist()}"
    text = "\n".join(
        [
            "# Variant A Clean Anchor Sweep Analysis",
            "",
            "This analysis compares clean no-anchor A/control with weak logit-anchor coefficients.",
            "Downstream `aulc` from `downstream_results.csv` is train-loss AULC; test-loss curve AULC is recomputed from `downstream_curves.csv`.",
            "",
            "## Matched Pair Deltas",
            *pair_lines,
            "",
            "## Absolute Deltas Vs Clean Control",
            *abs_lines,
            "",
            "## Technical Checks",
            f"- Start-bank and common logged VAE batch-hash checks: `{checks_line}`.",
            "",
            "## Figures",
            "- `anchor_sweep_run_summary.png`",
            "- `anchor_sweep_pair_deltas.png`",
            "- `anchor_sweep_absolute_curves.png`",
            "- `anchor_sweep_absolute_bars.png`",
            "- `anchor_sweep_step0_vs_aulc.png`",
            "- `anchor_sweep_tail_curves.png`",
            "- `anchor_sweep_vae_training.png`",
        ]
    )
    (OUT_DIR / "anchor_sweep_notes.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in RUNS:
        _check_complete(label)

    run_summary = pd.DataFrame([_run_summary(label) for label in RUNS])
    run_summary.to_csv(OUT_DIR / "anchor_sweep_run_summary.csv", index=False)

    pair_frames = []
    pair_curve_frames = []
    pair_summary_rows = []
    for pair_name, anchor_coeff, control_label, a_label in PAIRS:
        rows, curves = _compare_to(a_label, control_label, comparison=f"A_vs_matched_control_{pair_name}")
        rows["pair"] = pair_name
        rows["anchor_coeff"] = anchor_coeff
        curves["pair"] = pair_name
        curves["anchor_coeff"] = anchor_coeff
        pair_frames.append(rows)
        pair_curve_frames.append(curves)
        summary_row = _comparison_summary(rows, run_summary)
        summary_row["pair"] = pair_name
        summary_row["anchor_coeff"] = anchor_coeff
        summary_row["control_label"] = control_label
        summary_row["a_label"] = a_label
        pair_summary_rows.append(summary_row)
    pair_rows = pd.concat(pair_frames, ignore_index=True)
    pair_curves = pd.concat(pair_curve_frames, ignore_index=True)
    pair_summary = pd.DataFrame(pair_summary_rows)
    pair_rows.to_csv(OUT_DIR / "anchor_sweep_pair_per_start.csv", index=False)
    pair_curves.to_csv(OUT_DIR / "anchor_sweep_pair_curves.csv", index=False)
    pair_summary.to_csv(OUT_DIR / "anchor_sweep_pair_summary.csv", index=False)

    abs_frames = []
    abs_curve_frames = []
    abs_summary_rows = []
    for label in [label for label in RUNS if label != "control_clean"]:
        rows, curves = _compare_to(label, "control_clean", comparison="vs_control_clean")
        abs_frames.append(rows)
        abs_curve_frames.append(curves)
        abs_summary_rows.append(_comparison_summary(rows, run_summary))
    abs_rows = pd.concat(abs_frames, ignore_index=True)
    abs_curves = pd.concat(abs_curve_frames, ignore_index=True)
    abs_summary = pd.DataFrame(abs_summary_rows)
    abs_rows.to_csv(OUT_DIR / "anchor_sweep_vs_control_clean_per_start.csv", index=False)
    abs_curves.to_csv(OUT_DIR / "anchor_sweep_vs_control_clean_curves.csv", index=False)
    abs_summary.to_csv(OUT_DIR / "anchor_sweep_vs_control_clean_summary.csv", index=False)

    tech = _technical_checks()
    tech.to_csv(OUT_DIR / "anchor_sweep_technical_checks.csv", index=False)

    _plot_run_summary(run_summary)
    _plot_pair_deltas(pair_summary)
    _plot_absolute_curves(abs_curves, abs_summary)
    _plot_step0_scatter(pair_rows, abs_rows)
    _plot_tail_curves(abs_rows)
    _plot_vae_training()
    _write_notes(run_summary, pair_summary, abs_summary, tech)

    print(f"[analyze_variant_a_anchor_sweep_clean] wrote {OUT_DIR}")
    print(pair_summary.to_string(index=False))
    print(abs_summary.to_string(index=False))
    print(tech.to_string(index=False))


if __name__ == "__main__":
    main()
