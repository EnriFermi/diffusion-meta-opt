#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
KEY_COLS = ["source_weight_index", "start_bank_position", "task_name", "tau"]
CURVE_KEY_COLS = KEY_COLS + ["step"]
RESULT_REQUIRED_FINITE_COLS = [
    "lr",
    "aulc",
    "final_train_loss",
    "best_train_loss",
    "final_test_loss",
    "final_test_acc",
    "reconstruction_rel_l2",
]
CURVE_REQUIRED_FINITE_COLS = [
    "lr",
    "step",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "reconstruction_rel_l2",
]
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
    "vae_checkpoint.pt",
    "weight_pool.pt",
    "weight_pool_records.csv",
]
ALLOWED_A_DIFF_KEYS = {
    "run_label",
    "vae_precond_loss_kind",
    "vae_precond_coeff",
    "vae_precond_burg_coeff",
    "vae_precond_every",
    "vae_precond_samples",
    "vae_precond_pairs",
    "vae_precond_batch_size",
    "vae_precond_probe_scale",
    "vae_precond_estimator_scope",
    "vae_precond_hvp_mode",
    "vae_precond_loss_clip",
    "vae_precond_grad_damping",
    "vae_precond_warmup_steps",
    "vae_precond_ramp_steps",
    "vae_precond_sketch_dim",
    "vae_precond_sketch_refresh",
    "vae_precond_ema_decay",
    "vae_precond_jitter",
    "vae_precond_max_grad_ratio",
    "vae_precond_grad_clip_norm",
}
DIRECTION_COLS = [
    "train_block_direction_loss",
    "train_block_direction_effective_loss",
    "train_block_direction_coeff",
    "train_block_direction_ramp",
    "train_block_direction_top_fraction",
    "train_block_direction_top_k",
    "train_block_direction_row_cos_mean",
    "train_block_direction_row_cos_min",
    "train_block_direction_row_error_mean",
    "train_block_direction_row_error_topk_mean",
    "train_block_direction_row_error_max",
    "train_block_direction_norm_ratio_mean",
    "train_block_direction_norm_ratio_min",
    "train_block_direction_norm_ratio_max",
    "train_block_direction_grad_ratio",
]


def _run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / value).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cfg(run_dir: Path) -> dict[str, Any]:
    payload = _load_json(run_dir / "config.json")
    return dict(payload["config"])


def _manifest_status(run_dir: Path) -> str:
    if not (run_dir / "artifact_manifest.json").is_file():
        return "missing_manifest"
    return str(_load_json(run_dir / "artifact_manifest.json").get("summary", {}).get("status", ""))


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _unexpected_config_diffs(control_cfg: dict[str, Any], a_cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in sorted(set(control_cfg) | set(a_cfg)):
        if key in ALLOWED_A_DIFF_KEYS:
            continue
        if not _json_equal(control_cfg.get(key), a_cfg.get(key)):
            out.append(key)
    return out


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce")


def _finite(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _summ(values: pd.Series) -> dict[str, float | int]:
    arr = _finite(values)
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p95": math.nan, "min": math.nan, "max": math.nan, "final": math.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "final": float(arr[-1]),
    }


def _bootstrap_ci(values: np.ndarray, *, seed: int, reps: int = 4000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for idx in range(reps):
        draws[idx] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _mean_row(values: pd.Series, *, seed: int) -> dict[str, float | int]:
    arr = _finite(values)
    lo, hi = _bootstrap_ci(arr, seed=seed)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else math.nan,
        "median": float(np.median(arr)) if arr.size else math.nan,
        "ci95_low": lo,
        "ci95_high": hi,
        "gt0": int(np.sum(arr > 0.0)),
        "lt0": int(np.sum(arr < 0.0)),
    }


def _train_rows(run_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(run_dir / "vae_metrics.csv")
    if "record_type" in rows:
        rows = rows[rows["record_type"].fillna("").astype(str) != "vae_quality"].copy()
    if "step" in rows:
        rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
        rows = rows[rows["step"].notna()].copy()
    return rows


def _selected_lr_map(run_dir: Path) -> dict[str, float]:
    path = run_dir / "selected_lrs.csv"
    if not path.is_file():
        return {}
    rows = pd.read_csv(path)
    required = {"method", "candidate_lr", "selected"}
    if not required.issubset(rows.columns):
        return {}
    selected = rows[pd.to_numeric(rows["selected"], errors="coerce").fillna(0).astype(int) == 1].copy()
    out: dict[str, float] = {}
    for _, row in selected.iterrows():
        out[str(row["method"])] = float(row["candidate_lr"])
    return out


def _start_bank_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "downstream_start_bank.csv"
    if not path.is_file():
        return {"exists": False}
    rows = pd.read_csv(path)
    out: dict[str, Any] = {
        "exists": True,
        "rows": int(len(rows)),
        "sha256": _file_sha256(path),
    }
    if "start_role" in rows:
        out["start_role_counts"] = {str(k): int(v) for k, v in rows["start_role"].value_counts(dropna=False).items()}
    if "source_weight_index" in rows:
        out["source_weight_index_unique"] = int(rows["source_weight_index"].nunique(dropna=True))
    return out


def _start_bank_core_equal(control_dir: Path, a_dir: Path) -> bool | None:
    c_path = control_dir / "downstream_start_bank.csv"
    a_path = a_dir / "downstream_start_bank.csv"
    if not c_path.is_file() or not a_path.is_file():
        return None
    c = pd.read_csv(c_path)
    a = pd.read_csv(a_path)
    ignore = {"vae_variant", "vae_geometry_reg_coeff"}
    cols = sorted((set(c.columns) & set(a.columns)) - ignore)
    if not cols:
        return False
    sort_cols = [col for col in ["start_bank_position", "source_weight_index", "start_role"] if col in cols]
    if sort_cols:
        c = c.sort_values(sort_cols).reset_index(drop=True)
        a = a.sort_values(sort_cols).reset_index(drop=True)
    return bool(c[cols].reset_index(drop=True).equals(a[cols].reset_index(drop=True)))


def _numeric_health(frame: pd.DataFrame, required_finite_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        out["numeric_columns"] = 0
        return out
    values = numeric.to_numpy(dtype=np.float64)
    out["numeric_columns"] = int(len(numeric.columns))
    out["numeric_nan_count"] = int(np.isnan(values).sum())
    out["numeric_posinf_count"] = int(np.isposinf(values).sum())
    out["numeric_neginf_count"] = int(np.isneginf(values).sum())
    required_missing = [col for col in required_finite_cols if col not in frame.columns]
    out["required_finite_missing_cols"] = required_missing
    required_nonfinite: dict[str, int] = {}
    for col in required_finite_cols:
        if col not in frame.columns:
            continue
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        count = int((~np.isfinite(arr)).sum())
        if count:
            required_nonfinite[col] = count
    out["required_finite_nonfinite_counts"] = required_nonfinite
    out["required_finite_ok"] = not required_missing and not required_nonfinite
    return out


def _downstream_integrity(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    results_path = run_dir / "downstream_results.csv"
    curves_path = run_dir / "downstream_curves.csv"
    expected_eval_starts = int(cfg.get("eval_starts", -1))
    downstream_steps = int(cfg.get("downstream_steps", -1))
    eval_every = int(cfg.get("downstream_eval_every", 25))
    expected_curve_steps = (downstream_steps // eval_every) + 1 if downstream_steps >= 0 and eval_every > 0 else -1
    out["expected_eval_starts"] = expected_eval_starts
    out["expected_curve_steps_per_method"] = expected_curve_steps

    if results_path.is_file():
        results = pd.read_csv(results_path)
        out["downstream_results_rows"] = int(len(results))
        eval_results = results[results["split"].astype(str) == "eval"].copy() if "split" in results else results.copy()
        out["results_eval_rows_by_method"] = {
            str(method): int(len(group)) for method, group in eval_results.groupby("method", dropna=False)
        } if "method" in eval_results else {}
        result_keys = ["split", "method"] + KEY_COLS
        present_result_keys = [key for key in result_keys if key in results.columns]
        out["results_duplicate_key_rows"] = int(results.duplicated(present_result_keys).sum()) if present_result_keys else -1
        out["results_expected_eval_rows_ok"] = bool(
            expected_eval_starts > 0
            and out["results_eval_rows_by_method"]
            and all(count == expected_eval_starts for count in out["results_eval_rows_by_method"].values())
        )
        out["results_numeric_health"] = _numeric_health(results, RESULT_REQUIRED_FINITE_COLS)

    if curves_path.is_file():
        curves = pd.read_csv(curves_path)
        out["downstream_curves_rows"] = int(len(curves))
        eval_curves = curves[curves["split"].astype(str) == "eval"].copy() if "split" in curves else curves.copy()
        out["curves_eval_rows_by_method"] = {
            str(method): int(len(group)) for method, group in eval_curves.groupby("method", dropna=False)
        } if "method" in eval_curves else {}
        out["curves_eval_step_counts_by_method"] = {
            str(method): int(group["step"].nunique(dropna=True)) for method, group in eval_curves.groupby("method", dropna=False)
        } if "method" in eval_curves and "step" in eval_curves else {}
        out["curves_expected_eval_rows_per_method"] = (
            int(expected_eval_starts * expected_curve_steps)
            if expected_eval_starts > 0 and expected_curve_steps > 0
            else -1
        )
        curve_keys = ["split", "method"] + CURVE_KEY_COLS
        present_curve_keys = [key for key in curve_keys if key in curves.columns]
        out["curves_duplicate_key_rows"] = int(curves.duplicated(present_curve_keys).sum()) if present_curve_keys else -1
        out["curves_expected_step_counts_ok"] = bool(
            expected_curve_steps > 0
            and out["curves_eval_step_counts_by_method"]
            and all(count == expected_curve_steps for count in out["curves_eval_step_counts_by_method"].values())
        )
        out["curves_expected_eval_rows_ok"] = bool(
            out["curves_expected_eval_rows_per_method"] > 0
            and out["curves_eval_rows_by_method"]
            and all(count == out["curves_expected_eval_rows_per_method"] for count in out["curves_eval_rows_by_method"].values())
        )
        out["curves_numeric_health"] = _numeric_health(curves, CURVE_REQUIRED_FINITE_COLS)
    return out


def _file_hashes(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ["weight_pool.pt", "weight_pool_records.csv"]:
        path = run_dir / name
        if path.is_file():
            out[name] = _file_sha256(path)
    return out


def _direction_summary(label: str, run_dir: Path) -> dict[str, Any]:
    cfg = _load_cfg(run_dir)
    out: dict[str, Any] = {
        "label": label,
        "run_dir": str(run_dir),
        "manifest_status": _manifest_status(run_dir),
        "missing_files": ";".join(name for name in REQUIRED_FILES if not (run_dir / name).is_file()),
        "latent_dim": int(cfg.get("latent_dim", -1)),
        "vae_hidden_dim": int(cfg.get("vae_hidden_dim", -1)),
        "vae_precond_loss_kind": str(cfg.get("vae_precond_loss_kind", "")),
        "vae_precond_coeff": float(cfg.get("vae_precond_coeff", 0.0)),
        "vae_precond_max_grad_ratio": float(cfg.get("vae_precond_max_grad_ratio", 0.0)),
        "vae_precond_loss_clip": float(cfg.get("vae_precond_loss_clip", 0.0)),
        "vae_block_direction_coeff": float(cfg.get("vae_block_direction_coeff", 0.0)),
        "vae_block_direction_block": str(cfg.get("vae_block_direction_block", "")),
        "vae_block_direction_space": str(cfg.get("vae_block_direction_space", "")),
        "vae_block_direction_loss_kind": str(cfg.get("vae_block_direction_loss_kind", "")),
        "vae_block_direction_top_fraction": float(cfg.get("vae_block_direction_top_fraction", 1.0)),
        "vae_block_direction_ramp_steps": int(cfg.get("vae_block_direction_ramp_steps", 0)),
        "vae_block_direction_grad_diagnostic": bool(cfg.get("vae_block_direction_grad_diagnostic", False)),
        "vae_init_checkpoint": str(cfg.get("vae_init_checkpoint", "")),
        "weight_pool_source_dir": str(cfg.get("weight_pool_source_dir", "")),
    }
    if not (run_dir / "vae_metrics.csv").is_file():
        return out

    train = _train_rows(run_dir)
    for col in DIRECTION_COLS:
        stats = _summ(_num(train, col))
        for key, value in stats.items():
            out[f"{col}_{key}"] = value

    effective = _num(train, "train_block_direction_effective_loss")
    ramp = _num(train, "train_block_direction_ramp")
    ratio = _num(train, "train_block_direction_grad_ratio")
    out["direction_effective_positive_fraction"] = float(np.mean(_finite(effective) > 0.0)) if len(_finite(effective)) else math.nan
    out["direction_ramp_reaches_one"] = bool(np.nanmax(_finite(ramp)) >= 0.999) if len(_finite(ramp)) else False
    finite_ratio = _finite(ratio)
    out["direction_grad_ratio_positive_fraction"] = float(np.mean(finite_ratio > 0.0)) if finite_ratio.size else math.nan

    err = _num(train, "train_block_direction_row_error_mean")
    err_arr = _finite(err)
    if err_arr.size >= 10:
        window = max(1, err_arr.size // 10)
        out["direction_row_error_first10p_median"] = float(np.median(err_arr[:window]))
        out["direction_row_error_last10p_median"] = float(np.median(err_arr[-window:]))
        out["direction_row_error_first_minus_last"] = out["direction_row_error_first10p_median"] - out["direction_row_error_last10p_median"]
    return out


def _paired_method(control_dir: Path, a_dir: Path, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    c_results = pd.read_csv(control_dir / "downstream_results.csv")
    a_results = pd.read_csv(a_dir / "downstream_results.csv")
    c = c_results[(c_results["split"].astype(str) == "eval") & (c_results["method"].astype(str) == method)].copy()
    a = a_results[(a_results["split"].astype(str) == "eval") & (a_results["method"].astype(str) == method)].copy()
    c_key = c[KEY_COLS + ["aulc", "final_test_loss", "final_test_acc", "reconstruction_rel_l2", "lr"]].rename(
        columns={
            "aulc": "control_aulc",
            "final_test_loss": "control_final_test_loss",
            "final_test_acc": "control_final_test_acc",
            "reconstruction_rel_l2": "control_reconstruction_rel_l2",
            "lr": "control_lr",
        }
    )
    paired = a.merge(c_key, on=KEY_COLS, how="inner", validate="one_to_one")
    paired["method"] = method
    paired["delta_aulc"] = paired["aulc"] - paired["control_aulc"]
    paired["delta_final_test_loss"] = paired["final_test_loss"] - paired["control_final_test_loss"]
    paired["delta_final_test_acc"] = paired["final_test_acc"] - paired["control_final_test_acc"]
    paired["delta_reconstruction_rel_l2"] = paired["reconstruction_rel_l2"] - paired["control_reconstruction_rel_l2"]

    c_curves = pd.read_csv(control_dir / "downstream_curves.csv")
    a_curves = pd.read_csv(a_dir / "downstream_curves.csv")
    cc = c_curves[(c_curves["split"].astype(str) == "eval") & (c_curves["method"].astype(str) == method)].copy()
    ac = a_curves[(a_curves["split"].astype(str) == "eval") & (a_curves["method"].astype(str) == method)].copy()
    cc_key = cc[CURVE_KEY_COLS + ["train_loss", "test_loss", "test_acc"]].rename(
        columns={
            "train_loss": "control_train_loss",
            "test_loss": "control_test_loss",
            "test_acc": "control_test_acc",
        }
    )
    curves = ac.merge(cc_key, on=CURVE_KEY_COLS, how="inner", validate="one_to_one")
    curves["method"] = method
    curves["delta_train_loss"] = curves["train_loss"] - curves["control_train_loss"]
    curves["delta_test_loss"] = curves["test_loss"] - curves["control_test_loss"]
    curves["delta_test_acc"] = curves["test_acc"] - curves["control_test_acc"]
    step0 = curves[curves["step"].astype(int) == 0][KEY_COLS + ["delta_test_loss", "delta_train_loss"]].rename(
        columns={"delta_test_loss": "delta_step0_test_loss", "delta_train_loss": "delta_step0_train_loss"}
    )
    paired = paired.merge(step0, on=KEY_COLS, how="left", validate="one_to_one")
    return paired, curves


def _paired_summaries(control_dir: Path, a_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[pd.DataFrame] = []
    for method_idx, method in enumerate(["raw", "decoder_latent"]):
        paired, curves = _paired_method(control_dir, a_dir, method)
        pair_rows.append(paired)
        for metric_idx, metric in enumerate(
            [
                "delta_aulc",
                "delta_final_test_loss",
                "delta_final_test_acc",
                "delta_reconstruction_rel_l2",
                "delta_step0_test_loss",
                "delta_step0_train_loss",
            ]
        ):
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    **_mean_row(paired[metric], seed=1000 + 100 * method_idx + metric_idx),
                }
            )
        post0 = curves[curves["step"].astype(int) > 0].copy()
        if not post0.empty:
            by_start = post0.groupby(KEY_COLS, as_index=False).agg(
                delta_test_loss_post0_curve_mean=("delta_test_loss", "mean"),
                delta_train_loss_post0_curve_mean=("delta_train_loss", "mean"),
            )
            for metric_idx, metric in enumerate(["delta_test_loss_post0_curve_mean", "delta_train_loss_post0_curve_mean"]):
                summary_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        **_mean_row(by_start[metric], seed=2000 + 100 * method_idx + metric_idx),
                    }
                )
    return pd.DataFrame(summary_rows), pd.concat(pair_rows, ignore_index=True)


def _validity(control_dir: Path, a_dir: Path, *, raw_tol: float) -> dict[str, Any]:
    control_cfg = _load_cfg(control_dir)
    a_cfg = _load_cfg(a_dir)
    direction_keys = [
        "vae_block_direction_coeff",
        "vae_block_direction_block",
        "vae_block_direction_space",
        "vae_block_direction_loss_kind",
        "vae_block_direction_top_fraction",
        "vae_block_direction_ramp_steps",
        "vae_block_direction_grad_diagnostic",
    ]
    out: dict[str, Any] = {
        "control_run_dir": str(control_dir),
        "A_run_dir": str(a_dir),
        "control_manifest_status": _manifest_status(control_dir),
        "A_manifest_status": _manifest_status(a_dir),
        "control_missing_files": [name for name in REQUIRED_FILES if not (control_dir / name).is_file()],
        "A_missing_files": [name for name in REQUIRED_FILES if not (a_dir / name).is_file()],
        "unexpected_control_A_config_diff_keys": _unexpected_config_diffs(control_cfg, a_cfg),
        "direction_fields_equal": all(_json_equal(control_cfg.get(k), a_cfg.get(k)) for k in direction_keys),
        "direction_fields": {key: control_cfg.get(key) for key in direction_keys},
        "latent_dim_equal": int(control_cfg.get("latent_dim", -1)) == int(a_cfg.get("latent_dim", -2)),
        "hidden_dim_equal": int(control_cfg.get("vae_hidden_dim", -1)) == int(a_cfg.get("vae_hidden_dim", -2)),
        "seed_equal": int(control_cfg.get("seed", -1)) == int(a_cfg.get("seed", -2)),
        "init_checkpoint_equal": str(control_cfg.get("vae_init_checkpoint", "")) == str(a_cfg.get("vae_init_checkpoint", "")),
        "weight_pool_source_equal": str(control_cfg.get("weight_pool_source_dir", "")) == str(a_cfg.get("weight_pool_source_dir", "")),
        "control_precond_kind": str(control_cfg.get("vae_precond_loss_kind", "")),
        "A_precond_kind": str(a_cfg.get("vae_precond_loss_kind", "")),
        "raw_delta_tol": float(raw_tol),
    }
    control_lrs = _selected_lr_map(control_dir)
    a_lrs = _selected_lr_map(a_dir)
    if control_lrs or a_lrs:
        common_methods = sorted(set(control_lrs) & set(a_lrs))
        out["control_selected_lrs"] = control_lrs
        out["A_selected_lrs"] = a_lrs
        out["selected_lr_common_methods"] = common_methods
        out["selected_lrs_equal_on_common_methods"] = all(
            math.isclose(control_lrs[method], a_lrs[method], rel_tol=0.0, abs_tol=0.0)
            for method in common_methods
        )
        out["selected_lr_methods_equal"] = sorted(control_lrs) == sorted(a_lrs)

    control_start_bank = _start_bank_info(control_dir)
    a_start_bank = _start_bank_info(a_dir)
    out["control_start_bank"] = control_start_bank
    out["A_start_bank"] = a_start_bank
    if control_start_bank.get("exists") and a_start_bank.get("exists"):
        out["start_bank_sha256_equal"] = control_start_bank.get("sha256") == a_start_bank.get("sha256")
        out["start_bank_core_equal"] = _start_bank_core_equal(control_dir, a_dir)

    control_hashes = _file_hashes(control_dir)
    a_hashes = _file_hashes(a_dir)
    out["control_input_file_hashes"] = control_hashes
    out["A_input_file_hashes"] = a_hashes
    for name in sorted(set(control_hashes) | set(a_hashes)):
        out[f"{name}_sha256_equal"] = control_hashes.get(name) == a_hashes.get(name)

    out["control_downstream_integrity"] = _downstream_integrity(control_dir, control_cfg)
    out["A_downstream_integrity"] = _downstream_integrity(a_dir, a_cfg)

    if (control_dir / "downstream_results.csv").is_file() and (a_dir / "downstream_results.csv").is_file():
        paired_summary, paired_rows = _paired_summaries(control_dir, a_dir)
        out["paired_eval_rows_by_method"] = {
            str(method): int(len(group)) for method, group in paired_rows.groupby("method", dropna=False)
        }
        expected_eval_starts = int(control_cfg.get("eval_starts", -1))
        out["paired_eval_rows_match_expected"] = bool(
            expected_eval_starts > 0
            and out["paired_eval_rows_by_method"]
            and all(count == expected_eval_starts for count in out["paired_eval_rows_by_method"].values())
        )
        raw_rows = paired_rows[paired_rows["method"].astype(str) == "raw"].copy()
        out["raw_delta_aulc_abs_max"] = float(pd.to_numeric(raw_rows["delta_aulc"], errors="coerce").abs().max())
        out["raw_delta_final_test_loss_abs_max"] = float(pd.to_numeric(raw_rows["delta_final_test_loss"], errors="coerce").abs().max())
        out["raw_deltas_within_tol"] = bool(
            out["raw_delta_aulc_abs_max"] <= raw_tol and out["raw_delta_final_test_loss_abs_max"] <= raw_tol
        )
        dec = paired_summary[
            (paired_summary["method"].astype(str) == "decoder_latent")
            & (paired_summary["metric"].astype(str) == "delta_aulc")
        ]
        if not dec.empty:
            out["decoder_delta_aulc_mean"] = float(dec.iloc[0]["mean"])
            out["decoder_delta_aulc_ci95_low"] = float(dec.iloc[0]["ci95_low"])
            out["decoder_delta_aulc_ci95_high"] = float(dec.iloc[0]["ci95_high"])
    return out


def _integrity_ok(integrity: dict[str, Any]) -> bool:
    result_health = integrity.get("results_numeric_health", {})
    curve_health = integrity.get("curves_numeric_health", {})
    return bool(
        integrity.get("results_expected_eval_rows_ok") is True
        and integrity.get("curves_expected_step_counts_ok") is True
        and integrity.get("curves_expected_eval_rows_ok") is True
        and integrity.get("results_duplicate_key_rows") == 0
        and integrity.get("curves_duplicate_key_rows") == 0
        and result_health.get("required_finite_ok") is True
        and curve_health.get("required_finite_ok") is True
    )


def _attach_expected_direction_fields(validity: dict[str, Any], expected: dict[str, Any]) -> None:
    if not expected:
        return
    fields = validity.get("direction_fields", {})
    matches: dict[str, bool] = {}
    for key, expected_value in expected.items():
        actual_value = fields.get(key)
        if isinstance(expected_value, float):
            try:
                matches[key] = math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches[key] = False
        elif isinstance(expected_value, int):
            try:
                matches[key] = int(actual_value) == expected_value
            except (TypeError, ValueError):
                matches[key] = False
        else:
            matches[key] = str(actual_value) == str(expected_value)
    validity["expected_direction_fields"] = expected
    validity["expected_direction_field_matches"] = matches
    validity["expected_direction_fields_match"] = bool(matches) and all(matches.values())


def _direction_telemetry_ok(direction: pd.DataFrame) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if direction.empty:
        return False, ["direction telemetry frame is empty"]
    for _, row in direction.iterrows():
        label = str(row.get("label", ""))
        missing = str(row.get("missing_files", ""))
        if "vae_metrics.csv" in missing:
            failures.append(f"{label}: missing vae_metrics.csv")
        if not bool(row.get("direction_ramp_reaches_one", False)):
            failures.append(f"{label}: direction ramp did not reach one")
        for col in ["train_block_direction_effective_loss_n", "train_block_direction_grad_ratio_n"]:
            value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
            if not np.isfinite(value) or float(value) <= 0.0:
                failures.append(f"{label}: {col} is not positive")
        for col in ["direction_effective_positive_fraction", "direction_grad_ratio_positive_fraction"]:
            value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
            if not np.isfinite(value) or float(value) <= 0.0:
                failures.append(f"{label}: {col} is not positive")
    return not failures, failures


def _finalize_acceptance(validity: dict[str, Any], direction: pd.DataFrame) -> None:
    direction_ok, direction_failures = _direction_telemetry_ok(direction)
    checks = {
        "manifests_complete": validity.get("control_manifest_status") == "complete" and validity.get("A_manifest_status") == "complete",
        "required_files_present": not validity.get("control_missing_files") and not validity.get("A_missing_files"),
        "expected_config_diffs_only": not validity.get("unexpected_control_A_config_diff_keys"),
        "direction_fields_equal": validity.get("direction_fields_equal") is True,
        "expected_direction_fields_match": validity.get("expected_direction_fields_match", True) is True,
        "latent_dim_equal": validity.get("latent_dim_equal") is True,
        "hidden_dim_equal": validity.get("hidden_dim_equal") is True,
        "seed_equal": validity.get("seed_equal") is True,
        "init_checkpoint_equal": validity.get("init_checkpoint_equal") is True,
        "weight_pool_source_equal": validity.get("weight_pool_source_equal") is True,
        "input_weight_pool_equal": validity.get("weight_pool.pt_sha256_equal") is True,
        "input_weight_records_equal": validity.get("weight_pool_records.csv_sha256_equal") is True,
        "selected_lr_methods_equal": validity.get("selected_lr_methods_equal") is True,
        "selected_lrs_equal_on_common_methods": validity.get("selected_lrs_equal_on_common_methods") is True,
        "start_bank_sha256_equal": validity.get("start_bank_sha256_equal") is True,
        "start_bank_core_equal": validity.get("start_bank_core_equal") is True,
        "control_downstream_integrity_ok": _integrity_ok(validity.get("control_downstream_integrity", {})),
        "A_downstream_integrity_ok": _integrity_ok(validity.get("A_downstream_integrity", {})),
        "paired_eval_rows_match_expected": validity.get("paired_eval_rows_match_expected") is True,
        "raw_deltas_within_tol": validity.get("raw_deltas_within_tol") is True,
        "direction_telemetry_ok": direction_ok,
    }
    validity["acceptance_checks"] = checks
    validity["direction_telemetry_failures"] = direction_failures
    validity["accepted"] = bool(all(checks.values()))


def _write_review(path: Path, validity: dict[str, Any], direction: pd.DataFrame, paired_summary: pd.DataFrame | None) -> None:
    lines = [
        "# Variant A direction-guard validation",
        "",
        "This is a post-run validity and telemetry check for the guarded A/control pair. It does not by itself prove that the fc2 row-direction damage channel was repaired; that requires downstream clamp and margin/function audits.",
        "",
        "## Validity",
        "",
    ]
    for key in [
        "accepted",
        "acceptance_checks",
        "direction_telemetry_failures",
        "control_manifest_status",
        "A_manifest_status",
        "control_missing_files",
        "A_missing_files",
        "unexpected_control_A_config_diff_keys",
        "direction_fields_equal",
        "direction_fields",
        "expected_direction_fields",
        "expected_direction_field_matches",
        "expected_direction_fields_match",
        "latent_dim_equal",
        "hidden_dim_equal",
        "seed_equal",
        "init_checkpoint_equal",
        "weight_pool_source_equal",
        "control_selected_lrs",
        "A_selected_lrs",
        "selected_lrs_equal_on_common_methods",
        "selected_lr_methods_equal",
        "control_start_bank",
        "A_start_bank",
        "start_bank_sha256_equal",
        "start_bank_core_equal",
        "control_input_file_hashes",
        "A_input_file_hashes",
        "weight_pool.pt_sha256_equal",
        "weight_pool_records.csv_sha256_equal",
        "control_downstream_integrity",
        "A_downstream_integrity",
        "paired_eval_rows_by_method",
        "paired_eval_rows_match_expected",
        "raw_deltas_within_tol",
        "raw_delta_aulc_abs_max",
        "raw_delta_final_test_loss_abs_max",
        "decoder_delta_aulc_mean",
        "decoder_delta_aulc_ci95_low",
        "decoder_delta_aulc_ci95_high",
    ]:
        if key in validity:
            lines.append(f"- `{key}`: `{validity[key]}`")

    lines += ["", "## Direction Telemetry", ""]
    cols = [
        "label",
        "vae_precond_loss_kind",
        "vae_block_direction_coeff",
        "vae_block_direction_loss_kind",
        "vae_block_direction_top_fraction",
        "train_block_direction_effective_loss_median",
        "train_block_direction_grad_ratio_median",
        "train_block_direction_row_cos_mean_median",
        "train_block_direction_row_error_mean_median",
        "train_block_direction_row_error_topk_mean_median",
        "direction_row_error_first_minus_last",
        "direction_ramp_reaches_one",
    ]
    present = [col for col in cols if col in direction.columns]
    if present:
        lines.append(direction[present].to_markdown(index=False))
    else:
        lines.append("Direction telemetry unavailable.")

    if paired_summary is not None and not paired_summary.empty:
        lines += ["", "## Paired Downstream Summary", ""]
        lines.append(paired_summary.to_markdown(index=False))

    lines += [
        "",
        "## Interpretation Gates",
        "",
        "- If configs or raw deltas fail validity, do not interpret A/control downstream deltas.",
        "- If direction telemetry is zero or ramp never reaches one, audit the guard implementation before changing coefficients.",
        "- If downstream damage remains and margin/function audit shows row-direction damage remains, this intervention did not repair the proposed channel.",
        "- If row-direction damage is repaired but A only ties control, CH8 is a causal damage channel but not a sufficient positive fix; residual active-c3/tangent mechanisms remain.",
        "- If row-direction damage is repaired and A beats control on paired downstream, replicate before claiming the fix.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a guarded Variant A/control pair and summarize direction telemetry.")
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--a-run", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-delta-tol", type=float, default=1e-10)
    parser.add_argument("--expected-block-direction-coeff", type=float, default=None)
    parser.add_argument("--expected-block-direction-block", default=None)
    parser.add_argument("--expected-block-direction-space", default=None)
    parser.add_argument("--expected-block-direction-loss-kind", default=None)
    parser.add_argument("--expected-block-direction-top-fraction", type=float, default=None)
    parser.add_argument("--expected-block-direction-ramp-steps", type=int, default=None)
    parser.add_argument("--require-accepted", action="store_true", help="Exit nonzero if the post-run validity gate is rejected.")
    args = parser.parse_args()

    control_dir = _run_dir(args.control_run)
    a_dir = _run_dir(args.a_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    direction = pd.DataFrame([
        _direction_summary("control", control_dir),
        _direction_summary("A", a_dir),
    ])
    validity = _validity(control_dir, a_dir, raw_tol=float(args.raw_delta_tol))
    expected: dict[str, Any] = {}
    if args.expected_block_direction_coeff is not None:
        expected["vae_block_direction_coeff"] = float(args.expected_block_direction_coeff)
    if args.expected_block_direction_block is not None:
        expected["vae_block_direction_block"] = str(args.expected_block_direction_block)
    if args.expected_block_direction_space is not None:
        expected["vae_block_direction_space"] = str(args.expected_block_direction_space)
    if args.expected_block_direction_loss_kind is not None:
        expected["vae_block_direction_loss_kind"] = str(args.expected_block_direction_loss_kind)
    if args.expected_block_direction_top_fraction is not None:
        expected["vae_block_direction_top_fraction"] = float(args.expected_block_direction_top_fraction)
    if args.expected_block_direction_ramp_steps is not None:
        expected["vae_block_direction_ramp_steps"] = int(args.expected_block_direction_ramp_steps)
    _attach_expected_direction_fields(validity, expected)
    paired_summary: pd.DataFrame | None = None
    paired_rows: pd.DataFrame | None = None
    if not validity.get("control_missing_files") and not validity.get("A_missing_files"):
        paired_summary, paired_rows = _paired_summaries(control_dir, a_dir)
        paired_summary.to_csv(args.output_dir / "guard_paired_downstream_summary.csv", index=False)
        paired_rows.to_csv(args.output_dir / "guard_paired_downstream_rows.csv", index=False)

    _finalize_acceptance(validity, direction)
    direction.to_csv(args.output_dir / "guard_direction_telemetry_summary.csv", index=False)
    (args.output_dir / "guard_validity.json").write_text(json.dumps(validity, indent=2, sort_keys=True), encoding="utf-8")
    _write_review(args.output_dir / "guard_validation_review.md", validity, direction, paired_summary)
    print(f"[direction_guard_validation] wrote {args.output_dir}", flush=True)
    if args.require_accepted and not validity.get("accepted"):
        raise SystemExit("[direction_guard_validation] rejected; see guard_validity.json")


if __name__ == "__main__":
    main()
