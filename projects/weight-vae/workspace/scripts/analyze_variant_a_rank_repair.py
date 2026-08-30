from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
DEFAULT_OUT = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/rank_repair_m1024/analysis"
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
]
KEY_COLS = ["source_weight_index", "start_bank_position", "task_name", "tau"]
CURVE_KEY_COLS = KEY_COLS + ["step"]
REQUIRED_START_BANK_COLS = [
    "start_bank_position",
    "source_weight_index",
    "start_role",
    "selection",
    "task_name",
    "tau",
]
CONTROL_A_ALLOWED_CONFIG_DIFF_KEYS = {
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
PROVENANCE_HASH_FILES = [
    ROOT / "scripts/run_variant_a_rank_repair.py",
    ROOT / "scripts/analyze_variant_a_rank_repair.py",
    ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/pipeline.py",
    ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/downstream.py",
    ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/config.py",
]


def _run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / value).resolve()


def _finite_values(frame: pd.DataFrame, columns: list[str]) -> bool:
    existing = [col for col in columns if col in frame]
    if not existing:
        return False
    values = frame[existing].to_numpy(dtype=np.float64)
    return bool(np.isfinite(values).all())


def _finite_logged_values(frame: pd.DataFrame, columns: list[str]) -> bool:
    existing = [col for col in columns if col in frame]
    if not existing:
        return False
    for column in existing:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            return False
    return True


def _duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int:
    if not all(col in frame for col in columns):
        return int(len(frame))
    return int(frame.duplicated(columns).sum())


def _jsonable_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def _unexpected_config_diff_keys(left: dict[str, Any], right: dict[str, Any], *, allowed: set[str]) -> list[str]:
    keys = sorted(set(left) | set(right))
    return [key for key in keys if key not in allowed and not _jsonable_equal(left.get(key), right.get(key))]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_files(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) if path.is_file() else "" for path in paths}


def _run_hashes(run_dir: Path) -> dict[str, str]:
    return _hash_files([run_dir / "weight_pool.pt", run_dir / "weight_pool_records.csv", run_dir / "downstream_start_bank.csv"])


def _hashes_match(*hash_maps: dict[str, str]) -> bool:
    if not hash_maps:
        return False
    names = {Path(path).name for path in hash_maps[0]}
    for name in names:
        values = []
        for item in hash_maps:
            matched = [value for path, value in item.items() if Path(path).name == name]
            if len(matched) != 1 or not matched[0]:
                return False
            values.append(matched[0])
        if len(set(values)) != 1:
            return False
    return True


def _resolved_path_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def _expected_curve_steps(cfg: dict[str, Any]) -> list[int]:
    end = int(cfg["downstream_steps"])
    every = int(cfg.get("downstream_eval_every", 25))
    steps = list(range(0, end + 1, every))
    if not steps or steps[-1] != end:
        steps.append(end)
    return steps


def _bootstrap_ci(values: np.ndarray, *, seed: int, reps: int = 4000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        draws[idx] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _summary(values: pd.Series, *, seed: int) -> dict[str, Any]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    lo, hi = _bootstrap_ci(array, seed=seed)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)) if array.size else float("nan"),
        "median": float(np.median(array)) if array.size else float("nan"),
        "worse_count_delta_gt0": int(np.sum(array > 0.0)),
        "better_count_delta_lt0": int(np.sum(array < 0.0)),
        "bootstrap95_low": lo,
        "bootstrap95_high": hi,
        "min": float(np.min(array)) if array.size else float("nan"),
        "max": float(np.max(array)) if array.size else float("nan"),
    }


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))


def _load_config_payload(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _load_config(run_dir: Path) -> dict[str, Any]:
    payload = _load_config_payload(run_dir)
    return payload["config"]


def _file_config_hash(run_dir: Path) -> str:
    return str(_load_config_payload(run_dir).get("config_hash", ""))


def _manifest_config_hash(run_dir: Path) -> str:
    return str(_load_manifest(run_dir).get("config_hash", ""))


def _check_complete(label: str, run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    manifest = _load_manifest(run_dir) if (run_dir / "artifact_manifest.json").is_file() else {}
    status = manifest.get("summary", {}).get("status")
    return {
        "label": label,
        "run_dir": str(run_dir),
        "missing_files": missing,
        "manifest_status": status,
        "complete": bool(not missing and status == "complete"),
    }


def _selected_lr(run_dir: Path, method: str) -> float:
    rows = pd.read_csv(run_dir / "selected_lrs.csv")
    selected = pd.to_numeric(rows["selected"], errors="coerce").fillna(0).astype(int)
    sub = rows[(rows["method"].astype(str) == method) & (selected == 1)]
    if len(sub) != 1:
        return float("nan")
    return float(sub.iloc[0]["candidate_lr"])


def _quality_rows(run_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(run_dir / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    return rows[record_type == "vae_quality"].copy()


def _train_rows(run_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(run_dir / "vae_metrics.csv")
    record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    rows = rows[record_type != "vae_quality"].copy()
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    return rows[rows["step"].notna()].copy()


def _median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


def _p95(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.quantile(0.95)) if not values.empty else float("nan")


def _last(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _run_summary(label: str, run_dir: Path) -> dict[str, Any]:
    cfg = _load_config(run_dir)
    quality = _quality_rows(run_dir)
    train = _train_rows(run_dir)
    diagnostics = pd.read_csv(run_dir / "preconditioning_diagnostics.csv")
    geometry = pd.read_csv(run_dir / "geometry.csv")
    results = pd.read_csv(run_dir / "downstream_results.csv")
    eval_rows = results[results["split"].astype(str) == "eval"].copy()
    decoder = eval_rows[eval_rows["method"].astype(str) == "decoder_latent"]
    raw = eval_rows[eval_rows["method"].astype(str) == "raw"]
    return {
        "label": label,
        "run_label": run_dir.name,
        "run_dir": str(run_dir),
        "latent_dim": int(cfg["latent_dim"]),
        "vae_hidden_dim": int(cfg["vae_hidden_dim"]),
        "vae_init_checkpoint": str(cfg.get("vae_init_checkpoint", "")),
        "weight_pool_source_dir": str(cfg.get("weight_pool_source_dir", "")),
        "vae_steps": int(cfg["vae_steps"]),
        "downstream_steps": int(cfg["downstream_steps"]),
        "eval_starts": int(cfg["eval_starts"]),
        "selected_raw_lr": _selected_lr(run_dir, "raw"),
        "selected_decoder_lr": _selected_lr(run_dir, "decoder_latent"),
        "decoder_eval_starts": int(decoder["source_weight_index"].nunique()),
        "decoder_aulc_mean": float(decoder["aulc"].mean()),
        "decoder_aulc_median": float(decoder["aulc"].median()),
        "decoder_final_test_loss_mean": float(decoder["final_test_loss"].mean()),
        "raw_aulc_mean": float(raw["aulc"].mean()),
        "raw_aulc_median": float(raw["aulc"].median()),
        "quality_reconstruction_rel_l2_median": _median(quality, "reconstruction_rel_l2"),
        "quality_decoded_test_loss_median": _median(quality, "decoded_test_loss"),
        "quality_raw_test_loss_median": _median(quality, "raw_test_loss"),
        "quality_decoded_minus_raw_test_loss_median": _median(quality, "decoded_test_loss") - _median(quality, "raw_test_loss"),
        "val_recon_mse_final": _last(train, "val_recon_mse"),
        "train_precond_effective_loss_median": _median(train, "train_precond_effective_loss"),
        "train_precond_grad_scale_median": _median(train, "train_precond_grad_scale"),
        "train_block_recon_rel_l2_median": _median(train, "train_block_recon_rel_l2"),
        "train_function_anchor_margin_drop_median": _median(train, "train_function_anchor_margin_drop"),
        "li_A_full_per_dim_median": _median(diagnostics, "li_A_full_per_dim"),
        "li_A_full_per_dim_p95": _p95(diagnostics, "li_A_full_per_dim"),
        "li_A_full_per_dim_max": float(pd.to_numeric(diagnostics.get("li_A_full_per_dim"), errors="coerce").max())
        if "li_A_full_per_dim" in diagnostics
        else float("nan"),
        "hvp_probe_trace_m_per_dim_median": _median(diagnostics, "hvp_probe_trace_m_per_dim"),
        "hvp_probe_a_loss_per_dim_median": _median(diagnostics, "hvp_probe_a_loss_per_dim"),
        "geometry_isometry_objective": _median(geometry, "isometry_objective"),
        "geometry_condition_median": _median(geometry, "condition_median"),
        "geometry_trace_g_median": _median(geometry, "trace_g_median"),
        "downstream_finite": _finite_values(eval_rows, ["aulc", "final_test_loss", "final_train_loss"]),
        "train_finite": _finite_values(train, ["train_loss", "train_recon_mse"])
        and _finite_logged_values(train, ["val_recon_mse"]),
    }


def _paired_downstream(control_dir: Path, a_dir: Path, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    paired["method_analyzed"] = method
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
    curves["method_analyzed"] = method
    curves["delta_train_loss"] = curves["train_loss"] - curves["control_train_loss"]
    curves["delta_test_loss"] = curves["test_loss"] - curves["control_test_loss"]
    curves["delta_test_acc"] = curves["test_acc"] - curves["control_test_acc"]
    step0 = curves[curves["step"].astype(int) == 0][KEY_COLS + ["delta_train_loss", "delta_test_loss", "delta_test_acc"]].rename(
        columns={
            "delta_train_loss": "delta_step0_train_loss",
            "delta_test_loss": "delta_step0_test_loss",
            "delta_test_acc": "delta_step0_test_acc",
        }
    )
    paired = paired.merge(step0, on=KEY_COLS, how="left", validate="one_to_one")
    return paired, curves


def _paired_summary(paired: pd.DataFrame, curves: pd.DataFrame, method: str, *, seed_offset: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, metric in enumerate(
        [
            "delta_aulc",
            "delta_final_test_loss",
            "delta_final_test_acc",
            "delta_reconstruction_rel_l2",
            "delta_step0_test_loss",
        ]
    ):
        rows.append({"method": method, "metric": metric, **_summary(paired[metric], seed=1000 + seed_offset + idx)})
    final_step = int(curves["step"].max())
    final = curves[curves["step"].astype(int) == final_step].copy()
    rows.append({"method": method, "metric": f"delta_test_loss_step{final_step}", **_summary(final["delta_test_loss"], seed=1100 + seed_offset)})
    rows.append({"method": method, "metric": f"delta_train_loss_step{final_step}", **_summary(final["delta_train_loss"], seed=1200 + seed_offset)})
    post0 = curves[curves["step"].astype(int) > 0].copy()
    if not post0.empty:
        post0_by_start = post0.groupby(KEY_COLS, as_index=False).agg(
            delta_test_loss_post0_curve_mean=("delta_test_loss", "mean"),
            delta_train_loss_post0_curve_mean=("delta_train_loss", "mean"),
        )
        rows.append(
            {
                "method": method,
                "metric": "delta_test_loss_post0_curve_mean",
                **_summary(post0_by_start["delta_test_loss_post0_curve_mean"], seed=1300 + seed_offset),
            }
        )
        rows.append(
            {
                "method": method,
                "metric": "delta_train_loss_post0_curve_mean",
                **_summary(post0_by_start["delta_train_loss_post0_curve_mean"], seed=1400 + seed_offset),
            }
        )
    return pd.DataFrame(rows)


def _validation(
    *,
    baseline_dir: Path,
    control_dir: Path,
    a_dir: Path,
    run_checks: list[dict[str, Any]],
    run_summary: pd.DataFrame,
    paired_decoder: pd.DataFrame,
    curves_decoder: pd.DataFrame,
    paired_raw: pd.DataFrame,
    curves_raw: pd.DataFrame,
    raw_delta_tol: float,
) -> dict[str, Any]:
    baseline_cfg = _load_config(baseline_dir)
    control_cfg = _load_config(control_dir)
    a_cfg = _load_config(a_dir)
    c_bank = pd.read_csv(control_dir / "downstream_start_bank.csv")
    a_bank = pd.read_csv(a_dir / "downstream_start_bank.csv")
    common_cols = [col for col in c_bank.columns if col in a_bank.columns]
    control_expected_start_bank_rows = int(control_cfg.get("tune_starts", 0)) + int(control_cfg["eval_starts"])
    a_expected_start_bank_rows = int(a_cfg.get("tune_starts", 0)) + int(a_cfg["eval_starts"])
    control_bank_role_counts = c_bank["start_role"].astype(str).value_counts().to_dict() if "start_role" in c_bank else {}
    a_bank_role_counts = a_bank["start_role"].astype(str).value_counts().to_dict() if "start_role" in a_bank else {}
    control_required_start_bank_cols_present = all(col in c_bank.columns for col in REQUIRED_START_BANK_COLS)
    a_required_start_bank_cols_present = all(col in a_bank.columns for col in REQUIRED_START_BANK_COLS)
    baseline_manifest = _load_manifest(baseline_dir)
    c_manifest = _load_manifest(control_dir)
    a_manifest = _load_manifest(a_dir)
    expected_baseline_checkpoint = str((baseline_dir / "vae_checkpoint.pt").resolve())
    control_init_checkpoint = _resolved_path_text(str(control_cfg.get("vae_init_checkpoint", "")))
    a_init_checkpoint = _resolved_path_text(str(a_cfg.get("vae_init_checkpoint", "")))
    expected_steps = _expected_curve_steps(control_cfg)
    decoder_steps = sorted(pd.to_numeric(curves_decoder["step"], errors="coerce").dropna().astype(int).unique().tolist())
    raw_steps = sorted(pd.to_numeric(curves_raw["step"], errors="coerce").dropna().astype(int).unique().tolist())
    selected_lr_cols = ["selected_raw_lr", "selected_decoder_lr"]
    selected_lrs_finite = bool(
        np.isfinite(run_summary[selected_lr_cols].to_numpy(dtype=np.float64)).all()
        and (run_summary[selected_lr_cols].to_numpy(dtype=np.float64) > 0.0).all()
    )
    core_metrics_finite = bool(run_summary[["downstream_finite", "train_finite"]].astype(bool).all().all())
    raw_delta_aulc_abs_max = float(pd.to_numeric(paired_raw["delta_aulc"], errors="coerce").abs().max())
    raw_delta_final_test_loss_abs_max = float(pd.to_numeric(paired_raw["delta_final_test_loss"], errors="coerce").abs().max())
    control_hashes = _run_hashes(control_dir)
    a_hashes = _run_hashes(a_dir)
    baseline_hashes = _run_hashes(baseline_dir)
    config_diff_unexpected = _unexpected_config_diff_keys(
        control_cfg,
        a_cfg,
        allowed=CONTROL_A_ALLOWED_CONFIG_DIFF_KEYS,
    )
    control_row = run_summary[run_summary["label"].astype(str) == "control"].iloc[0]
    a_row = run_summary[run_summary["label"].astype(str) == "A_clip20"].iloc[0]
    return {
        "all_runs_complete": bool(all(row["complete"] for row in run_checks)),
        "latent_dims": {
            "baseline": int(baseline_cfg["latent_dim"]),
            "control": int(control_cfg["latent_dim"]),
            "A": int(a_cfg["latent_dim"]),
        },
        "vae_hidden_dims": {
            "baseline": int(baseline_cfg["vae_hidden_dim"]),
            "control": int(control_cfg["vae_hidden_dim"]),
            "A": int(a_cfg["vae_hidden_dim"]),
        },
        "baseline_has_empty_init_checkpoint": str(baseline_cfg.get("vae_init_checkpoint", "")) == "",
        "expected_baseline_checkpoint": expected_baseline_checkpoint,
        "control_init_checkpoint": control_init_checkpoint,
        "A_init_checkpoint": a_init_checkpoint,
        "control_A_same_init_checkpoint": control_init_checkpoint == a_init_checkpoint,
        "control_A_init_is_fresh_baseline_checkpoint": (
            control_init_checkpoint == expected_baseline_checkpoint and a_init_checkpoint == expected_baseline_checkpoint
        ),
        "control_A_same_downstream_steps": int(control_cfg["downstream_steps"]) == int(a_cfg["downstream_steps"]),
        "control_A_same_eval_starts": int(control_cfg["eval_starts"]) == int(a_cfg["eval_starts"]),
        "control_A_same_downstream_eval_every": int(control_cfg.get("downstream_eval_every", 25))
        == int(a_cfg.get("downstream_eval_every", 25)),
        "control_A_same_downstream_batch_size": int(control_cfg.get("downstream_batch_size", -1))
        == int(a_cfg.get("downstream_batch_size", -2)),
        "control_A_same_seed": int(control_cfg.get("seed", -1)) == int(a_cfg.get("seed", -2)),
        "control_A_same_weight_pool_source_dir": _resolved_path_text(str(control_cfg.get("weight_pool_source_dir", "")))
        == _resolved_path_text(str(a_cfg.get("weight_pool_source_dir", ""))),
        "control_A_same_start_bank_common_cols": bool(c_bank[common_cols].equals(a_bank[common_cols])),
        "required_start_bank_cols": REQUIRED_START_BANK_COLS,
        "control_required_start_bank_cols_present": bool(control_required_start_bank_cols_present),
        "A_required_start_bank_cols_present": bool(a_required_start_bank_cols_present),
        "start_bank_common_cols_count": int(len(common_cols)),
        "control_start_bank_rows": int(len(c_bank)),
        "A_start_bank_rows": int(len(a_bank)),
        "expected_control_start_bank_rows": control_expected_start_bank_rows,
        "expected_A_start_bank_rows": a_expected_start_bank_rows,
        "control_start_bank_role_counts": {str(k): int(v) for k, v in control_bank_role_counts.items()},
        "A_start_bank_role_counts": {str(k): int(v) for k, v in a_bank_role_counts.items()},
        "control_start_bank_duplicate_keys": _duplicate_count(c_bank, KEY_COLS),
        "A_start_bank_duplicate_keys": _duplicate_count(a_bank, KEY_COLS),
        "paired_decoder_rows": int(len(paired_decoder)),
        "paired_raw_rows": int(len(paired_raw)),
        "decoder_curve_rows": int(len(curves_decoder)),
        "raw_curve_rows": int(len(curves_raw)),
        "expected_paired_rows": int(control_cfg["eval_starts"]),
        "expected_curve_steps": expected_steps,
        "decoder_curve_steps": decoder_steps,
        "raw_curve_steps": raw_steps,
        "decoder_curve_steps_match_expected": decoder_steps == expected_steps,
        "raw_curve_steps_match_expected": raw_steps == expected_steps,
        "expected_decoder_curve_rows": int(control_cfg["eval_starts"]) * len(expected_steps),
        "expected_raw_curve_rows": int(control_cfg["eval_starts"]) * len(expected_steps),
        "paired_decoder_duplicate_keys": _duplicate_count(paired_decoder, KEY_COLS),
        "paired_raw_duplicate_keys": _duplicate_count(paired_raw, KEY_COLS),
        "decoder_curve_duplicate_keys": _duplicate_count(curves_decoder, CURVE_KEY_COLS),
        "raw_curve_duplicate_keys": _duplicate_count(curves_raw, CURVE_KEY_COLS),
        "baseline_manifest_status": baseline_manifest.get("summary", {}).get("status"),
        "control_manifest_status": c_manifest.get("summary", {}).get("status"),
        "A_manifest_status": a_manifest.get("summary", {}).get("status"),
        "baseline_manifest_config_hash": baseline_manifest.get("config_hash", ""),
        "control_manifest_config_hash": c_manifest.get("config_hash", ""),
        "A_manifest_config_hash": a_manifest.get("config_hash", ""),
        "baseline_file_config_hash": _file_config_hash(baseline_dir),
        "control_file_config_hash": _file_config_hash(control_dir),
        "A_file_config_hash": _file_config_hash(a_dir),
        "manifest_config_hashes_match_config_json": (
            _manifest_config_hash(baseline_dir) == _file_config_hash(baseline_dir)
            and _manifest_config_hash(control_dir) == _file_config_hash(control_dir)
            and _manifest_config_hash(a_dir) == _file_config_hash(a_dir)
        ),
        "script_and_core_hashes": _hash_files(PROVENANCE_HASH_FILES),
        "baseline_artifact_hashes": baseline_hashes,
        "control_artifact_hashes": control_hashes,
        "A_artifact_hashes": a_hashes,
        "baseline_control_A_weight_pool_hashes_match": _hashes_match(
            {k: v for k, v in baseline_hashes.items() if Path(k).name in {"weight_pool.pt", "weight_pool_records.csv"}},
            {k: v for k, v in control_hashes.items() if Path(k).name in {"weight_pool.pt", "weight_pool_records.csv"}},
            {k: v for k, v in a_hashes.items() if Path(k).name in {"weight_pool.pt", "weight_pool_records.csv"}},
        ),
        "control_A_config_diff_unexpected_keys": config_diff_unexpected,
        "control_A_config_diff_only_expected_A_fields": len(config_diff_unexpected) == 0,
        "selected_raw_lrs_match": bool(math.isclose(float(control_row["selected_raw_lr"]), float(a_row["selected_raw_lr"]), rel_tol=0.0, abs_tol=0.0)),
        "selected_decoder_lrs_match": bool(
            math.isclose(float(control_row["selected_decoder_lr"]), float(a_row["selected_decoder_lr"]), rel_tol=0.0, abs_tol=0.0)
        ),
        "selected_lrs_finite_positive": selected_lrs_finite,
        "core_metrics_finite": core_metrics_finite,
        "raw_delta_tol": float(raw_delta_tol),
        "raw_delta_aulc_abs_max": raw_delta_aulc_abs_max,
        "raw_delta_final_test_loss_abs_max": raw_delta_final_test_loss_abs_max,
        "raw_deltas_within_tol": bool(
            raw_delta_aulc_abs_max <= float(raw_delta_tol)
            and raw_delta_final_test_loss_abs_max <= float(raw_delta_tol)
        ),
    }


def _plot_outputs(
    out_dir: Path,
    paired_decoder: pd.DataFrame,
    curves_decoder: pd.DataFrame,
    run_summary: pd.DataFrame,
    *,
    experiment_label: str,
) -> None:
    step_summary = curves_decoder.groupby("step", as_index=False).agg(
        mean_delta_test_loss=("delta_test_loss", "mean"),
        median_delta_test_loss=("delta_test_loss", "median"),
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(step_summary["step"], step_summary["mean_delta_test_loss"], marker="o", label="mean")
    ax.plot(step_summary["step"], step_summary["median_delta_test_loss"], marker="o", label="median")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xlabel("downstream step")
    ax.set_ylabel("A - control test loss")
    ax.set_title(f"{experiment_label}: Decoder-Latent Paired Delta")
    ax.legend()
    fig.savefig(out_dir / "rank_repair_decoder_delta_by_step.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(paired_decoder["delta_step0_test_loss"], paired_decoder["delta_aulc"], alpha=0.85)
    ax.axhline(0.0, color="black", lw=1)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("A - control step0 test loss")
    ax.set_ylabel("A - control AULC")
    ax.set_title(f"{experiment_label}: Step0 vs AULC Delta")
    fig.savefig(out_dir / "rank_repair_step0_vs_aulc_delta.png", dpi=180)
    plt.close(fig)

    cols = ["quality_reconstruction_rel_l2_median", "li_A_full_per_dim_p95", "decoder_aulc_mean"]
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 4), constrained_layout=True)
    for ax, col in zip(axes, cols, strict=True):
        sub = run_summary[run_summary["label"].isin(["control", "A_clip20"])].copy()
        ax.bar(sub["label"], sub[col])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(out_dir / "rank_repair_control_A_summary_bars.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = _run_dir(args.baseline_run)
    control_dir = _run_dir(args.control_run)
    a_dir = _run_dir(args.a_run)

    run_checks = [
        _check_complete("baseline", baseline_dir),
        _check_complete("control", control_dir),
        _check_complete("A_clip20", a_dir),
    ]
    pd.DataFrame(run_checks).to_csv(out_dir / "rank_repair_completion_checks.csv", index=False)
    if not all(row["complete"] for row in run_checks):
        raise RuntimeError(f"rank repair runs are incomplete; see {out_dir / 'rank_repair_completion_checks.csv'}")

    run_summary = pd.DataFrame(
        [
            _run_summary("baseline", baseline_dir),
            _run_summary("control", control_dir),
            _run_summary("A_clip20", a_dir),
        ]
    )
    run_summary.to_csv(out_dir / "rank_repair_run_summary.csv", index=False)

    paired_decoder, curves_decoder = _paired_downstream(control_dir, a_dir, "decoder_latent")
    paired_raw, curves_raw = _paired_downstream(control_dir, a_dir, "raw")
    paired_decoder.to_csv(out_dir / "rank_repair_decoder_paired_deltas.csv", index=False)
    curves_decoder.to_csv(out_dir / "rank_repair_decoder_curve_deltas.csv", index=False)
    paired_raw.to_csv(out_dir / "rank_repair_raw_paired_deltas.csv", index=False)
    curves_raw.to_csv(out_dir / "rank_repair_raw_curve_deltas.csv", index=False)

    paired_summary = pd.concat(
        [
            _paired_summary(paired_decoder, curves_decoder, "decoder_latent", seed_offset=0),
            _paired_summary(paired_raw, curves_raw, "raw", seed_offset=100),
        ],
        ignore_index=True,
    )
    paired_summary.to_csv(out_dir / "rank_repair_paired_summary.csv", index=False)

    validation = _validation(
        baseline_dir=baseline_dir,
        control_dir=control_dir,
        a_dir=a_dir,
        run_checks=run_checks,
        run_summary=run_summary,
        paired_decoder=paired_decoder,
        curves_decoder=curves_decoder,
        paired_raw=paired_raw,
        curves_raw=curves_raw,
        raw_delta_tol=float(args.raw_delta_tol),
    )
    control_cfg = _load_config(control_dir)
    a_cfg = _load_config(a_dir)
    validation["accepted"] = bool(
        validation["all_runs_complete"]
        and validation["latent_dims"] == {"baseline": int(args.latent_dim), "control": int(args.latent_dim), "A": int(args.latent_dim)}
        and validation["vae_hidden_dims"]
        == {"baseline": int(args.hidden_dim), "control": int(args.hidden_dim), "A": int(args.hidden_dim)}
        and validation["baseline_has_empty_init_checkpoint"]
        and validation["control_A_same_init_checkpoint"]
        and validation["control_A_init_is_fresh_baseline_checkpoint"]
        and validation["control_A_same_downstream_steps"]
        and validation["control_A_same_downstream_eval_every"]
        and validation["control_A_same_downstream_batch_size"]
        and validation["control_A_same_seed"]
        and validation["control_A_same_weight_pool_source_dir"]
        and validation["control_A_same_eval_starts"]
        and validation["control_A_same_start_bank_common_cols"]
        and validation["control_required_start_bank_cols_present"]
        and validation["A_required_start_bank_cols_present"]
        and validation["control_start_bank_rows"] == validation["expected_control_start_bank_rows"]
        and validation["A_start_bank_rows"] == validation["expected_A_start_bank_rows"]
        and validation["control_start_bank_role_counts"].get("tune") == int(control_cfg.get("tune_starts", 0))
        and validation["control_start_bank_role_counts"].get("eval") == validation["expected_paired_rows"]
        and validation["A_start_bank_role_counts"].get("tune") == int(a_cfg.get("tune_starts", 0))
        and validation["A_start_bank_role_counts"].get("eval") == validation["expected_paired_rows"]
        and validation["control_start_bank_duplicate_keys"] == 0
        and validation["A_start_bank_duplicate_keys"] == 0
        and validation["paired_decoder_rows"] == validation["expected_paired_rows"]
        and validation["paired_raw_rows"] == validation["expected_paired_rows"]
        and validation["decoder_curve_rows"] == validation["expected_decoder_curve_rows"]
        and validation["raw_curve_rows"] == validation["expected_raw_curve_rows"]
        and validation["decoder_curve_steps_match_expected"]
        and validation["raw_curve_steps_match_expected"]
        and validation["paired_decoder_duplicate_keys"] == 0
        and validation["paired_raw_duplicate_keys"] == 0
        and validation["decoder_curve_duplicate_keys"] == 0
        and validation["raw_curve_duplicate_keys"] == 0
        and validation["manifest_config_hashes_match_config_json"]
        and validation["baseline_control_A_weight_pool_hashes_match"]
        and validation["control_A_config_diff_only_expected_A_fields"]
        and validation["selected_lrs_finite_positive"]
        and validation["core_metrics_finite"]
        and validation["raw_deltas_within_tol"]
    )
    (out_dir / "rank_repair_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    experiment_label = f"m{int(args.latent_dim)} h{int(args.hidden_dim)} Rank Repair"
    _plot_outputs(out_dir, paired_decoder, curves_decoder, run_summary, experiment_label=experiment_label)

    decoder_aulc = paired_summary[
        (paired_summary["method"].astype(str) == "decoder_latent")
        & (paired_summary["metric"].astype(str) == "delta_aulc")
    ].iloc[0]
    decoder_final = paired_summary[
        (paired_summary["method"].astype(str) == "decoder_latent")
        & (paired_summary["metric"].astype(str).str.startswith("delta_test_loss_step"))
    ].iloc[0]
    decoder_post0 = paired_summary[
        (paired_summary["method"].astype(str) == "decoder_latent")
        & (paired_summary["metric"].astype(str) == "delta_test_loss_post0_curve_mean")
    ].iloc[0]
    control_row = run_summary[run_summary["label"].astype(str) == "control"].iloc[0]
    a_row = run_summary[run_summary["label"].astype(str) == "A_clip20"].iloc[0]
    review = [
        f"# {experiment_label} Review",
        "",
        f"Validation accepted: `{validation['accepted']}`.",
        "",
        f"Primary comparison: matched `m={int(args.latent_dim)},h={int(args.hidden_dim)} control` vs "
        f"`m={int(args.latent_dim)},h={int(args.hidden_dim)} A_clip20`, not old m512/m1024 artifacts.",
        "",
        "Key downstream result:",
        (
            f"- decoder_latent A-control AULC delta mean `{decoder_aulc['mean']:.6g}`, "
            f"median `{decoder_aulc['median']:.6g}`, bootstrap95 "
            f"`[{decoder_aulc['bootstrap95_low']:.6g}, {decoder_aulc['bootstrap95_high']:.6g}]`, "
            f"worse_count `{int(decoder_aulc['worse_count_delta_gt0'])}/{int(decoder_aulc['n'])}`."
        ),
        (
            f"- final/checkpoint test-loss delta mean `{decoder_final['mean']:.6g}`, "
            f"median `{decoder_final['median']:.6g}`, bootstrap95 "
            f"`[{decoder_final['bootstrap95_low']:.6g}, {decoder_final['bootstrap95_high']:.6g}]`."
        ),
        (
            f"- post-step curve mean test-loss delta `{decoder_post0['mean']:.6g}`, "
            f"median `{decoder_post0['median']:.6g}`, bootstrap95 "
            f"`[{decoder_post0['bootstrap95_low']:.6g}, {decoder_post0['bootstrap95_high']:.6g}]`."
        ),
        "",
        "Proxy/function summary:",
        f"- control recon rel-L2 median `{control_row['quality_reconstruction_rel_l2_median']:.6g}`; A `{a_row['quality_reconstruction_rel_l2_median']:.6g}`.",
        f"- control decoded-minus-raw test loss median `{control_row['quality_decoded_minus_raw_test_loss_median']:.6g}`; A `{a_row['quality_decoded_minus_raw_test_loss_median']:.6g}`.",
        f"- control li_A p95 `{control_row['li_A_full_per_dim_p95']:.6g}`; A `{a_row['li_A_full_per_dim_p95']:.6g}`.",
        f"- control trace/m median `{control_row['hvp_probe_trace_m_per_dim_median']:.6g}`; A `{a_row['hvp_probe_trace_m_per_dim_median']:.6g}`.",
        "",
        "Validation gates:",
        f"- raw A-control AULC abs max `{validation['raw_delta_aulc_abs_max']:.6g}` with tolerance `{validation['raw_delta_tol']:.6g}`.",
        f"- raw final-test-loss abs max `{validation['raw_delta_final_test_loss_abs_max']:.6g}`.",
        f"- latent dims `{validation['latent_dims']}`; VAE hidden dims `{validation['vae_hidden_dims']}`.",
        f"- manifest/config hashes match: `{validation['manifest_config_hashes_match_config_json']}`.",
        f"- control/A init is fresh baseline checkpoint: `{validation['control_A_init_is_fresh_baseline_checkpoint']}`.",
        f"- start-bank rows control/A `{validation['control_start_bank_rows']}`/`{validation['A_start_bank_rows']}`; expected "
        f"`{validation['expected_control_start_bank_rows']}`/`{validation['expected_A_start_bank_rows']}`.",
        f"- start-bank role counts control `{validation['control_start_bank_role_counts']}`; A `{validation['A_start_bank_role_counts']}`.",
        f"- baseline/control/A weight-pool hashes match: `{validation['baseline_control_A_weight_pool_hashes_match']}`.",
        f"- control/A unexpected config diff keys: `{validation['control_A_config_diff_unexpected_keys']}`.",
        f"- selected raw LR match: `{validation['selected_raw_lrs_match']}`; selected decoder LR match: `{validation['selected_decoder_lrs_match']}`.",
        "",
        "Interpretation rule:",
        "- If validation is false, do not use this run for causal claims.",
        "- If selected decoder LR differs, downstream comparison is still under the per-run tuned-LR protocol; fixed-common-LR sensitivity is needed before attributing an effect solely to geometry.",
        "- If AULC delta CI is below zero and worse-count supports A, this is a candidate rank-repair win.",
        "- If A does not clearly win, the next required artifact is a matched trajectory/CG audit measuring normal residual and fixed-absolute-damped c3 terms.",
        "",
        "Artifacts:",
        f"- `{out_dir / 'rank_repair_run_summary.csv'}`",
        f"- `{out_dir / 'rank_repair_paired_summary.csv'}`",
        f"- `{out_dir / 'rank_repair_decoder_paired_deltas.csv'}`",
        f"- `{out_dir / 'rank_repair_validation.json'}`",
        f"- `{out_dir / 'rank_repair_decoder_delta_by_step.png'}`",
        f"- `{out_dir / 'rank_repair_step0_vs_aulc_delta.png'}`",
    ]
    (out_dir / "rank_repair_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(f"[rank_repair_analysis] validation accepted={validation['accepted']} out_dir={out_dir}", flush=True)
    print(
        "[rank_repair_analysis] decoder_delta_aulc "
        f"mean={decoder_aulc['mean']:.6g} median={decoder_aulc['median']:.6g} "
        f"ci=[{decoder_aulc['bootstrap95_low']:.6g},{decoder_aulc['bootstrap95_high']:.6g}]",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched m1024 Variant A rank-repair runs.")
    parser.add_argument(
        "--baseline-run",
        default="sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_m1024_rank_repair_v1_seed0",
    )
    parser.add_argument(
        "--control-run",
        default="sage_cnn_vae_smoothing_celo_meta_control_m1024_fc2recon_lam0p03_marginhuber_c0p001_rank_repair_v1_seed0",
    )
    parser.add_argument(
        "--a-run",
        default="sage_cnn_vae_smoothing_celo_meta_li_a_hvp_m1024_cap0p25_clip20_fc2recon_lam0p03_marginhuber_c0p001_rank_repair_v1_seed0",
    )
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-delta-tol", type=float, default=1e-8)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
