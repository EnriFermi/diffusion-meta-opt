#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    celo_meta_mlp_spec,
    load_torch_cache,
    tiny_cnn_spec,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.downstream import (
    BLOCK_PRESERVED_LATENT_DOWNSTREAM_METHODS,
    DownstreamContext,
    run_downstream_curve,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.progress import make_progress


ARTIFACT_ROOT = Path("artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing").resolve()


def _log(message: str) -> None:
    print(f"[extended_downstream] {message}", flush=True)


def _run_dir(run_name: str) -> Path:
    path = Path(run_name).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / run_name).resolve()


def _load_cfg(run_dir: Path, *, device: str, eval_starts: int, downstream_steps: int | None) -> ExperimentConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**payload["config"])
    overrides: dict[str, Any] = {
        "device": str(device),
        "show_progress": True,
        "progress_backend": "text",
        "tune_starts": 0,
        "eval_starts": int(eval_starts),
    }
    if downstream_steps is not None:
        overrides["downstream_steps"] = int(downstream_steps)
    return replace(cfg, **overrides)


def _spec_for_cfg(cfg: ExperimentConfig):
    value = str(cfg.weight_distribution).strip().lower()
    if value in {"celo", "celo_meta", "celo_meta_mlp", "paper_meta_mlp"}:
        return celo_meta_mlp_spec(cfg)
    if value in {"tiny", "tiny_cnn", "cnn", "fashion_mnist_tiny_cnn"}:
        return tiny_cnn_spec()
    raise ValueError(f"unsupported weight_distribution={cfg.weight_distribution!r}")


def _selected_lr(run_dir: Path, method: str, fallback: float | None) -> float:
    if fallback is not None and math.isfinite(float(fallback)) and float(fallback) > 0.0:
        return float(fallback)
    selected_path = run_dir / "selected_lrs.csv"
    rows = pd.read_csv(selected_path)
    sub = rows[(rows["method"].astype(str) == str(method)) & (pd.to_numeric(rows["selected"], errors="coerce").fillna(0).astype(int) == 1)]
    if sub.empty:
        raise RuntimeError(f"no selected LR for method={method!r} in {selected_path}")
    return float(sub.iloc[0]["candidate_lr"])


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _expected_cache_meta(
    *,
    label: str,
    run_dir: Path,
    cfg: ExperimentConfig,
    methods: list[str],
    eval_starts: int,
    skip_starts: int,
    raw_lr: float,
    decoder_lr: float,
    start_bank: pd.DataFrame,
    start_bank_csv: Path | None = None,
) -> dict[str, Any]:
    meta = {
        "label": str(label),
        "run_dir": str(run_dir.resolve()),
        "methods": [str(v) for v in methods],
        "eval_starts": int(eval_starts),
        "skip_starts": int(skip_starts),
        "downstream_steps": int(cfg.downstream_steps),
        "raw_lr": float(raw_lr),
        "decoder_lr": float(decoder_lr),
        "start_source_weight_indices": [int(v) for v in start_bank["source_weight_index"].astype("int64").tolist()],
        "config_signature": _file_signature(run_dir / "config.json"),
        "checkpoint_signature": _file_signature(run_dir / "vae_checkpoint.pt"),
        "weight_pool_signature": _file_signature(run_dir / "weight_pool.pt"),
        "selected_lrs_signature": _file_signature(run_dir / "selected_lrs.csv"),
    }
    if start_bank_csv is not None:
        meta["start_bank_source"] = "csv"
        meta["start_bank_csv_signature"] = _file_signature(start_bank_csv)
    return meta


def _cache_meta_matches(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if actual_value is None or not math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-15):
                return False, f"{key}: expected={expected_value!r} actual={actual_value!r}"
        elif actual_value != expected_value:
            return False, f"{key}: expected={expected_value!r} actual={actual_value!r}"
    return True, ""


def _validate_cached_frames(
    *,
    results: pd.DataFrame,
    curves: pd.DataFrame,
    label: str,
    methods: list[str],
    eval_starts: int,
) -> None:
    expected_methods = {str(v) for v in methods}
    for name, frame in [("results", results), ("curves", curves)]:
        if "label" not in frame.columns or set(frame["label"].astype(str).unique().tolist()) != {str(label)}:
            raise RuntimeError(f"cache label mismatch in {name} for label={label!r}")
        if "method" not in frame.columns or set(frame["method"].astype(str).unique().tolist()) != expected_methods:
            raise RuntimeError(f"cache method mismatch in {name} for label={label!r}")
    expected_result_rows = int(eval_starts) * len(expected_methods)
    if len(results) != expected_result_rows:
        raise RuntimeError(f"cache result row mismatch for label={label!r}: got={len(results)} expected={expected_result_rows}")
    grouped = curves.groupby(["method", "source_weight_index"], dropna=False).size()
    if len(grouped) != expected_result_rows:
        raise RuntimeError(f"cache curve start/method row mismatch for label={label!r}: got={len(grouped)} expected={expected_result_rows}")


def _format_lr_grid(values: list[float] | None) -> list[float]:
    if values is None:
        return []
    out = [float(v) for v in values]
    if not out:
        return []
    bad = [v for v in out if not math.isfinite(v) or v <= 0.0]
    if bad:
        raise ValueError(f"decoder SGD LR grid must contain only positive finite values; bad={bad}")
    return sorted(set(out))


def _load_weight_pool(run_dir: Path) -> tuple[torch.Tensor, pd.DataFrame, str]:
    payload = load_torch_cache(run_dir / "weight_pool.pt")
    if payload is None or not isinstance(payload.get("weights"), torch.Tensor):
        raise RuntimeError(f"could not load weight_pool.pt from {run_dir}")
    records_path = run_dir / "weight_pool_records.csv"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    return payload["weights"].detach().cpu(), pd.read_csv(records_path), str(payload.get("cache_key", ""))


def _load_vae(run_dir: Path, cfg: ExperimentConfig, weight_dim: int, *, device: torch.device, dtype: torch.dtype):
    payload = load_torch_cache(run_dir / "vae_checkpoint.pt")
    if payload is None:
        raise RuntimeError(f"could not load vae_checkpoint.pt from {run_dir}")
    normalizer = WeightNormalizer.from_state_dict(payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weight_dim))
    vae.load_state_dict(payload["model_state"])
    vae.to(device=device, dtype=dtype).eval()
    val_indices = payload.get("val_indices")
    if not isinstance(val_indices, torch.Tensor):
        raise RuntimeError(f"checkpoint has no val_indices tensor: {run_dir}")
    return vae, normalizer, val_indices.detach().cpu().long()


def _heldout_final_indices(
    *,
    val_indices: torch.Tensor,
    weight_records: pd.DataFrame,
    weights_count: int,
    required: int,
) -> list[int]:
    if weight_records.empty or "step" not in weight_records.columns:
        return [int(v) for v in val_indices[: min(int(required), int(val_indices.numel()))].tolist()]
    final_step = int(pd.to_numeric(weight_records["step"], errors="coerce").max())
    record_steps = torch.as_tensor(
        pd.to_numeric(weight_records["step"], errors="coerce").fillna(-1).astype("int64").to_numpy(copy=True),
        dtype=torch.long,
    )
    if int(val_indices.numel()) > 0:
        clamped = val_indices.clamp(min=0, max=max(0, int(weights_count) - 1))
        selected = val_indices[record_steps.index_select(0, clamped) == final_step]
        if int(selected.numel()) >= int(required):
            return [int(v) for v in selected[: int(required)].tolist()]
    final_all = weight_records.index[pd.to_numeric(weight_records["step"], errors="coerce") == final_step].to_numpy(copy=True)
    if int(final_all.shape[0]) >= int(required):
        return [int(v) for v in final_all[: int(required)].tolist()]
    if int(val_indices.numel()) > 0:
        return [int(v) for v in val_indices[: min(int(required), int(val_indices.numel()))].tolist()]
    return list(range(min(int(required), int(weights_count))))


def _start_bank(
    *,
    val_indices: torch.Tensor,
    weight_records: pd.DataFrame,
    weights_count: int,
    skip_starts: int,
    eval_starts: int | None,
    start_bank_csv: Path | None = None,
) -> pd.DataFrame:
    if start_bank_csv is not None:
        rows = pd.read_csv(start_bank_csv)
        if "source_weight_index" not in rows.columns:
            raise ValueError(f"{start_bank_csv} has no source_weight_index column")
        if rows.empty:
            raise ValueError(f"{start_bank_csv} has no rows")
        rows = rows.copy().reset_index(drop=True)
        if eval_starts is not None:
            if len(rows) < int(eval_starts):
                raise ValueError(f"{start_bank_csv} has {len(rows)} rows, fewer than requested eval_starts={int(eval_starts)}")
            rows = rows.iloc[: int(eval_starts)].copy().reset_index(drop=True)
        source_values = pd.to_numeric(rows["source_weight_index"], errors="raise")
        non_integral = source_values[source_values.isna() | (source_values != np.floor(source_values))]
        if not non_integral.empty:
            raise ValueError(f"{start_bank_csv} has non-integer source_weight_index values: {non_integral.tolist()}")
        source_indices = source_values.astype("int64")
        duplicated = source_indices[source_indices.duplicated()].astype(int).tolist()
        if duplicated:
            raise ValueError(f"{start_bank_csv} has duplicate source_weight_index values: {duplicated}")
        invalid = [int(v) for v in source_indices.tolist() if int(v) < 0 or int(v) >= int(weights_count)]
        if invalid:
            raise ValueError(f"{start_bank_csv} has out-of-range source_weight_index values for weights_count={int(weights_count)}: {invalid}")
        rows["source_weight_index"] = source_indices.astype(int)
        if "start_bank_position" not in rows.columns:
            rows.insert(0, "start_bank_position", list(range(len(rows))))
        if "start_role" not in rows.columns:
            rows["start_role"] = "eval"
        if "selection" not in rows.columns:
            rows["selection"] = f"csv:{start_bank_csv}"
        return rows.reset_index(drop=True)

    if eval_starts is None:
        raise ValueError("eval_starts must be set when --start-bank-csv is absent")
    required = int(skip_starts) + int(eval_starts)
    positions = _heldout_final_indices(
        val_indices=val_indices,
        weight_records=weight_records,
        weights_count=int(weights_count),
        required=required,
    )
    if len(positions) < required:
        raise RuntimeError(f"not enough heldout starts: got={len(positions)} required={required}")
    eval_positions = positions[int(skip_starts) : int(skip_starts) + int(eval_starts)]
    rows = weight_records.iloc[eval_positions].copy().reset_index(drop=True) if not weight_records.empty else pd.DataFrame()
    rows["source_weight_index"] = [int(v) for v in eval_positions]
    rows.insert(0, "start_bank_position", list(range(len(rows))))
    rows["start_role"] = "eval"
    rows["selection"] = f"extended_heldout_final_after_skip_{int(skip_starts)}"
    return rows


def _bootstrap_ci(values: np.ndarray, *, seed: int = 0, reps: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _test_mean_by_start(curves: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "source_weight_index", "task_name", "tau", "lr"]
    if "label" in curves.columns:
        keys = ["label", *keys]
    return (
        curves.groupby(keys, as_index=False)
        .agg(
            test_mean_loss=("test_loss", "mean"),
            test_trapz_loss=("test_loss", lambda x: float(np.trapezoid(np.asarray(x, dtype=np.float64)) / max(1, len(x) - 1))),
            step0_test_loss=("test_loss", "first"),
            post0_test_loss_mean=("test_loss", lambda x: float(np.mean(np.asarray(x, dtype=np.float64)[1:])) if len(x) > 1 else float("nan")),
            train_mean_loss=("train_loss", "mean"),
        )
        .reset_index(drop=True)
    )


def _paired_summary(combined_results: pd.DataFrame, combined_curves: pd.DataFrame, *, labels: list[str], out_dir: Path) -> pd.DataFrame:
    if len(labels) != 2:
        return pd.DataFrame()
    control_label, a_label = labels
    test_means = _test_mean_by_start(combined_curves)
    metrics: list[dict[str, Any]] = []
    delta_rows: list[pd.DataFrame] = []
    for method in sorted(test_means["method"].astype(str).unique().tolist()):
        c = test_means[(test_means["label"] == control_label) & (test_means["method"] == method)].copy()
        a = test_means[(test_means["label"] == a_label) & (test_means["method"] == method)].copy()
        keys = ["method", "source_weight_index", "task_name", "tau", "lr"]
        merged = c.merge(a, on=keys, suffixes=("_control", "_a"), validate="one_to_one")
        if merged.empty:
            raise RuntimeError(f"empty paired merge for method={method!r}; check paired starts and LR consistency")
        merged["label_a"] = a_label
        for col in ["test_mean_loss", "test_trapz_loss", "step0_test_loss", "post0_test_loss_mean", "train_mean_loss"]:
            merged[f"{col}_delta"] = merged[f"{col}_a"] - merged[f"{col}_control"]
            values = merged[f"{col}_delta"].to_numpy(dtype=np.float64)
            lo, hi = _bootstrap_ci(values, seed=100 + len(metrics))
            metrics.append(
                {
                    "method": method,
                    "metric": f"{col}_delta",
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "worse_count_A_gt_control": int(np.sum(values > 0.0)),
                    "better_count_A_lt_control": int(np.sum(values < 0.0)),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
        delta_rows.append(merged)
    deltas = pd.concat(delta_rows, ignore_index=True, sort=False) if delta_rows else pd.DataFrame()
    deltas.to_csv(out_dir / "paired_downstream_deltas.csv", index=False)
    summary = pd.DataFrame(metrics)
    summary.to_csv(out_dir / "paired_downstream_summary.csv", index=False)
    _plot_paired_deltas(deltas, summary, out_dir / "paired_downstream_deltas.png")
    return summary


def _plot_paired_deltas(deltas: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    dec = deltas[deltas["method"].astype(str) == "decoder_latent"].copy()
    if dec.empty:
        return
    dec = dec.sort_values("test_mean_loss_delta").reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    x = np.arange(len(dec))
    colors = np.where(dec["test_mean_loss_delta"].to_numpy(dtype=np.float64) > 0.0, "#c44e52", "#4c78a8")
    axes[0, 0].bar(x, dec["test_mean_loss_delta"], color=colors)
    axes[0, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 0].set_title("Extended Decoder Test-Mean Delta")
    axes[0, 0].set_xlabel("matched eval start, sorted")
    axes[0, 0].set_ylabel("A - control")

    axes[0, 1].bar(x, dec["train_mean_loss_delta"], color=np.where(dec["train_mean_loss_delta"] > 0.0, "#c44e52", "#4c78a8"))
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_title("Extended Decoder Train-Mean Delta")
    axes[0, 1].set_xlabel("matched eval start, same order")

    axes[1, 0].scatter(dec["step0_test_loss_delta"], dec["post0_test_loss_mean_delta"], c=colors)
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 0].axvline(0.0, color="black", linewidth=1.0)
    axes[1, 0].set_title("Step0 vs Post-Step Delta")
    axes[1, 0].set_xlabel("step0 test loss A - control")
    axes[1, 0].set_ylabel("post0 mean test loss A - control")

    lines = []
    for metric in ["test_mean_loss_delta", "test_trapz_loss_delta", "step0_test_loss_delta", "post0_test_loss_mean_delta"]:
        row = summary[(summary["method"] == "decoder_latent") & (summary["metric"] == metric)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"{metric}\n"
            f"  mean={float(r['mean']):.6g}  med={float(r['median']):.6g}  "
            f"worse={int(r['worse_count_A_gt_control'])}/{int(r['n'])}\n"
            f"  CI=[{float(r['bootstrap95_low']):.6g}, {float(r['bootstrap95_high']):.6g}]"
        )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.0, 1.0, "\n\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _tune_decoder_sgd_lrs(
    *,
    cfg: ExperimentConfig,
    spec: Any,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    task_tensors: dict[str, Any],
    starts: torch.Tensor,
    start_records: pd.DataFrame,
    methods: list[str],
    lr_grid: list[float],
    tune_starts: int,
    tune_label: str,
    out_dir: Path,
) -> dict[str, float]:
    sgd_methods = [str(method) for method in methods if str(method).startswith("decoder_latent_sgd")]
    if not sgd_methods:
        return {}
    if not lr_grid:
        raise ValueError("decoder SGD LR grid is empty")
    tune_count = min(int(tune_starts), int(starts.shape[0]))
    if tune_count <= 0:
        raise ValueError("decoder SGD LR tuning requested but tune_starts resolved to zero")
    ctx = DownstreamContext(cfg=cfg, spec=spec, vae=vae, normalizer=normalizer, task_tensors=task_tensors, stop_path=None)
    rows: list[dict[str, Any]] = []
    total = int(len(sgd_methods) * len(lr_grid) * tune_count)
    progress = make_progress(cfg, total=total, desc=f"tune decoder SGD LR ({tune_label})")
    try:
        for method in sgd_methods:
            for lr in lr_grid:
                metrics_for_lr: list[float] = []
                for start_idx in range(tune_count):
                    metadata = start_records.iloc[start_idx].to_dict()
                    _, metrics = run_downstream_curve(
                        ctx=ctx,
                        w0=starts[start_idx],
                        method=str(method),
                        lr=float(lr),
                        steps=int(cfg.downstream_steps),
                        start_index=int(start_idx),
                        start_metadata=metadata,
                        split="sgd_lr_tune",
                    )
                    aulc = float(metrics["aulc"])
                    metrics_for_lr.append(aulc)
                    rows.append(
                        {
                            "tune_label": str(tune_label),
                            "method": str(method),
                            "candidate_lr": float(lr),
                            "start_index": int(start_idx),
                            "source_weight_index": int(metadata.get("source_weight_index", start_idx)),
                            "aulc": aulc,
                            "final_test_loss": float(metrics.get("final_test_loss", float("nan"))),
                            "diverged": bool(metrics.get("diverged", False)),
                        }
                    )
                    progress.set_postfix(
                        {
                            "method": method,
                            "lr": f"{float(lr):.1e}",
                            "start": start_idx,
                            "aulc": f"{aulc:.4g}",
                        }
                    )
                    progress.update(1)
                median_aulc = float(np.median(np.asarray(metrics_for_lr, dtype=np.float64)))
                rows.append(
                    {
                        "tune_label": str(tune_label),
                        "method": str(method),
                        "candidate_lr": float(lr),
                        "start_index": -1,
                        "source_weight_index": -1,
                        "aulc": median_aulc,
                        "final_test_loss": float("nan"),
                        "diverged": False,
                        "summary_row": True,
                    }
                )
    finally:
        progress.close()

    tune_frame = pd.DataFrame(rows)
    tune_frame["summary_row"] = tune_frame.get("summary_row", False).fillna(False).astype(bool)
    summary = tune_frame[tune_frame["summary_row"]].copy()
    selected: dict[str, float] = {}
    selected_rows: list[dict[str, Any]] = []
    for method in sgd_methods:
        sub = summary[summary["method"].astype(str) == str(method)].copy()
        if sub.empty:
            raise RuntimeError(f"no SGD LR tuning rows for method={method!r}")
        best = sub.sort_values(["aulc", "candidate_lr"], ascending=[True, True]).iloc[0]
        selected[str(method)] = float(best["candidate_lr"])
        selected_rows.append(
            {
                "tune_label": str(tune_label),
                "method": str(method),
                "candidate_lr": float(best["candidate_lr"]),
                "tuning_median_aulc": float(best["aulc"]),
                "selected": 1,
                "tune_starts": int(tune_count),
            }
        )
    tune_frame.to_csv(out_dir / "decoder_sgd_lr_tuning_rows.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out_dir / "decoder_sgd_lr_selected.csv", index=False)
    _log(
        "decoder_sgd_lr_tuning "
        f"label={tune_label} tune_starts={tune_count} grid={lr_grid} "
        f"selected={selected} rows={out_dir / 'decoder_sgd_lr_tuning_rows.csv'}"
    )
    return selected


def _tune_decoder_preserve_lrs(
    *,
    cfg: ExperimentConfig,
    spec: Any,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    task_tensors: dict[str, Any],
    starts: torch.Tensor,
    start_records: pd.DataFrame,
    methods: list[str],
    lr_grid: list[float],
    tune_starts: int,
    tune_label: str,
    out_dir: Path,
) -> dict[str, float]:
    preserve_methods = [str(method) for method in methods if str(method) in BLOCK_PRESERVED_LATENT_DOWNSTREAM_METHODS]
    if not preserve_methods:
        return {}
    if not lr_grid:
        raise ValueError("decoder preserve LR grid is empty")
    tune_count = min(int(tune_starts), int(starts.shape[0]))
    if tune_count <= 0:
        raise ValueError("decoder preserve LR tuning requested but tune_starts resolved to zero")
    ctx = DownstreamContext(cfg=cfg, spec=spec, vae=vae, normalizer=normalizer, task_tensors=task_tensors, stop_path=None)
    rows: list[dict[str, Any]] = []
    total = int(len(preserve_methods) * len(lr_grid) * tune_count)
    progress = make_progress(cfg, total=total, desc=f"tune decoder preserve LR ({tune_label})")
    try:
        for method in preserve_methods:
            for lr in lr_grid:
                metrics_for_lr: list[float] = []
                for start_idx in range(tune_count):
                    metadata = start_records.iloc[start_idx].to_dict()
                    _, metrics = run_downstream_curve(
                        ctx=ctx,
                        w0=starts[start_idx],
                        method=str(method),
                        lr=float(lr),
                        steps=int(cfg.downstream_steps),
                        start_index=int(start_idx),
                        start_metadata=metadata,
                        split="preserve_lr_tune",
                    )
                    aulc = float(metrics["aulc"])
                    metrics_for_lr.append(aulc)
                    rows.append(
                        {
                            "tune_label": str(tune_label),
                            "method": str(method),
                            "candidate_lr": float(lr),
                            "start_index": int(start_idx),
                            "source_weight_index": int(metadata.get("source_weight_index", start_idx)),
                            "aulc": aulc,
                            "final_test_loss": float(metrics.get("final_test_loss", float("nan"))),
                            "post_splice_reconstruction_rel_l2": float(metrics.get("post_splice_reconstruction_rel_l2", float("nan"))),
                            "preserved_block_rel_l2_to_donor": float(metrics.get("preserved_block_rel_l2_to_donor", float("nan"))),
                            "diverged": bool(metrics.get("diverged", False)),
                        }
                    )
                    progress.set_postfix(
                        {
                            "method": method.replace("decoder_latent_raw_", "").replace("_clamp", ""),
                            "lr": f"{float(lr):.1e}",
                            "start": start_idx,
                            "aulc": f"{aulc:.4g}",
                        }
                    )
                    progress.update(1)
                median_aulc = float(np.median(np.asarray(metrics_for_lr, dtype=np.float64)))
                rows.append(
                    {
                        "tune_label": str(tune_label),
                        "method": str(method),
                        "candidate_lr": float(lr),
                        "start_index": -1,
                        "source_weight_index": -1,
                        "aulc": median_aulc,
                        "final_test_loss": float("nan"),
                        "post_splice_reconstruction_rel_l2": float("nan"),
                        "preserved_block_rel_l2_to_donor": float("nan"),
                        "diverged": False,
                        "summary_row": True,
                    }
                )
    finally:
        progress.close()

    tune_frame = pd.DataFrame(rows)
    tune_frame["summary_row"] = tune_frame.get("summary_row", False).fillna(False).astype(bool)
    summary = tune_frame[tune_frame["summary_row"]].copy()
    selected: dict[str, float] = {}
    selected_rows: list[dict[str, Any]] = []
    for method in preserve_methods:
        sub = summary[summary["method"].astype(str) == str(method)].copy()
        if sub.empty:
            raise RuntimeError(f"no decoder preserve LR tuning rows for method={method!r}")
        best = sub.sort_values(["aulc", "candidate_lr"], ascending=[True, True]).iloc[0]
        selected[str(method)] = float(best["candidate_lr"])
        selected_rows.append(
            {
                "tune_label": str(tune_label),
                "method": str(method),
                "candidate_lr": float(best["candidate_lr"]),
                "tuning_median_aulc": float(best["aulc"]),
                "selected": 1,
                "tune_starts": int(tune_count),
            }
        )
    tune_frame.to_csv(out_dir / "decoder_preserve_lr_tuning_rows.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out_dir / "decoder_preserve_lr_selected.csv", index=False)
    _log(
        "decoder_preserve_lr_tuning "
        f"label={tune_label} tune_starts={tune_count} grid={lr_grid} "
        f"selected={selected} rows={out_dir / 'decoder_preserve_lr_tuning_rows.csv'}"
    )
    return selected


def _write_block_preserve_validation(combined_results: pd.DataFrame, out_dir: Path) -> None:
    if "preserved_block_rel_l2_to_donor" not in combined_results.columns:
        return
    preserve_methods = {str(method) for method in BLOCK_PRESERVED_LATENT_DOWNSTREAM_METHODS}
    sub = combined_results[combined_results["method"].astype(str).isin(preserve_methods)].copy()
    if sub.empty:
        return
    rows: list[dict[str, Any]] = []
    group_keys = ["label", "method", "block_preserve_block", "block_preserve_keys", "block_preserve_donor"]
    for key_values, group in sub.groupby(group_keys, dropna=False):
        label, method, block, keys, donor = key_values
        preserved = pd.to_numeric(group["preserved_block_rel_l2_to_donor"], errors="coerce")
        post_splice = pd.to_numeric(group["post_splice_reconstruction_rel_l2"], errors="coerce")
        splice_delta = pd.to_numeric(group["splice_delta_rel_l2_to_decoded"], errors="coerce")
        rows.append(
            {
                "label": str(label),
                "method": str(method),
                "block_preserve_block": str(block),
                "block_preserve_keys": str(keys),
                "block_preserve_donor": str(donor),
                "rows": int(len(group)),
                "max_preserved_block_rel_l2_to_donor": float(preserved.max(skipna=True)),
                "mean_post_splice_reconstruction_rel_l2": float(post_splice.mean(skipna=True)),
                "mean_splice_delta_rel_l2_to_decoded": float(splice_delta.mean(skipna=True)),
                "accepted_preserved_block_exact": bool(float(preserved.max(skipna=True)) <= 1e-7),
            }
        )
    path = out_dir / "block_preserve_validation.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _log(f"block_preserve_validation path={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run larger fixed-LR downstream evaluation on existing Variant A/control VAE checkpoints.")
    parser.add_argument("--run", action="append", required=True, help="Run label or path. Pass exactly two for paired summary.")
    parser.add_argument("--label", action="append", required=True, help="Short label matching each --run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-starts",
        type=int,
        default=None,
        help="Number of eval starts. Defaults to 64, or all rows in --start-bank-csv when a CSV bank is supplied.",
    )
    parser.add_argument("--skip-starts", type=int, default=8)
    parser.add_argument("--start-bank-csv", default="", help="Optional CSV start bank with source_weight_index rows to use instead of skip-start selection.")
    parser.add_argument("--downstream-steps", type=int, default=None)
    parser.add_argument("--raw-lr", type=float, default=None)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument(
        "--decoder-sgd-lr",
        type=float,
        default=None,
        help="Explicit LR for decoder_latent_sgd / decoder_latent_sgd_momentum. Required when those methods are requested.",
    )
    parser.add_argument(
        "--decoder-sgd-lr-grid",
        nargs="+",
        type=float,
        default=None,
        help="Control-only candidate LR grid for decoder_latent_sgd* methods. Used when --decoder-sgd-lr is absent.",
    )
    parser.add_argument(
        "--decoder-sgd-tune-starts",
        type=int,
        default=8,
        help="Number of selected starts from the tune label used for control-only decoder SGD LR selection.",
    )
    parser.add_argument(
        "--decoder-sgd-tune-label",
        default="",
        help="Label used for decoder SGD LR tuning. Defaults to the first --label value.",
    )
    parser.add_argument(
        "--decoder-preserve-lr",
        type=float,
        default=None,
        help="Explicit LR for decoder_latent_raw_*_clamp methods. If absent, use --decoder-preserve-lr-grid control-only tuning.",
    )
    parser.add_argument(
        "--decoder-preserve-lr-grid",
        nargs="+",
        type=float,
        default=None,
        help="Control-only candidate LR grid for decoder_latent_raw_*_clamp methods.",
    )
    parser.add_argument(
        "--decoder-preserve-tune-starts",
        type=int,
        default=8,
        help="Number of selected starts from the tune label used for decoder preserve LR selection.",
    )
    parser.add_argument(
        "--decoder-preserve-tune-label",
        default="",
        help="Label used for decoder preserve LR tuning. Defaults to the first --label value.",
    )
    parser.add_argument("--downstream-sgd-momentum", type=float, default=0.9)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "decoder_latent"],
        choices=[
            "raw",
            "decoder_latent",
            "decoder_latent_sgd",
            "decoder_latent_sgd_momentum",
            "decoder_latent_raw_classifier_head_clamp",
            "decoder_latent_raw_fc2_weight_clamp",
            "decoder_latent_raw_fc2_bias_clamp",
        ],
    )
    parser.add_argument("--force", action="store_true", help="Recompute per-label downstream CSVs even if they already exist.")
    args = parser.parse_args()

    if len(args.run) != len(args.label):
        raise ValueError("--run and --label counts must match")
    if len(set(args.label)) != len(args.label):
        raise ValueError("--label values must be unique")
    needs_sgd_lr = any(str(method).startswith("decoder_latent_sgd") for method in args.methods)
    needs_preserve_lr = any(str(method) in BLOCK_PRESERVED_LATENT_DOWNSTREAM_METHODS for method in args.methods)
    decoder_sgd_lr_grid = _format_lr_grid(args.decoder_sgd_lr_grid)
    decoder_preserve_lr_grid = _format_lr_grid(args.decoder_preserve_lr_grid)
    if needs_sgd_lr and args.decoder_sgd_lr is None and not decoder_sgd_lr_grid:
        raise ValueError(
            "requesting decoder_latent_sgd* requires either --decoder-sgd-lr "
            "or a control-only --decoder-sgd-lr-grid"
        )
    if needs_preserve_lr and args.decoder_preserve_lr is None and not decoder_preserve_lr_grid:
        raise ValueError(
            "requesting decoder_latent_raw_*_clamp requires either --decoder-preserve-lr "
            "or a control-only --decoder-preserve-lr-grid"
        )

    started = time.time()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [_run_dir(name) for name in args.run]
    labels = [str(v) for v in args.label]
    start_bank_csv = Path(args.start_bank_csv).expanduser().resolve() if str(args.start_bank_csv).strip() else None
    if start_bank_csv is not None and not start_bank_csv.is_file():
        raise FileNotFoundError(start_bank_csv)
    requested_eval_starts = int(args.eval_starts) if args.eval_starts is not None else None
    cfg_eval_starts = int(requested_eval_starts) if requested_eval_starts is not None else 64
    start_source = f"csv:{start_bank_csv}" if start_bank_csv is not None else f"heldout_final_after_skip:{int(args.skip_starts)}"
    eval_starts_for_log = str(requested_eval_starts) if requested_eval_starts is not None else ("csv_all_rows" if start_bank_csv is not None else str(cfg_eval_starts))
    skip_starts_for_log = f"{args.skip_starts}(ignored_for_csv)" if start_bank_csv is not None else str(args.skip_starts)
    _log(
        "startup "
        f"device={args.device} eval_starts={eval_starts_for_log} skip_starts={skip_starts_for_log} "
        f"start_bank_source={start_source} "
        f"downstream_steps={args.downstream_steps} labels={labels} output_dir={out_dir}"
    )

    base_cfg = _load_cfg(run_dirs[0], device=str(args.device), eval_starts=cfg_eval_starts, downstream_steps=args.downstream_steps)
    device = torch.device(base_cfg.device)
    dtype = torch_dtype(base_cfg)
    _log(f"stage=load_data dtype={dtype} seed={base_cfg.seed} data_root={base_cfg.data_root}")
    task_tensors = _load_task_tensors_for_pipeline(base_cfg, device=device, dtype=dtype)
    _log(
        "loaded_tasks "
        + ", ".join(
            f"{name}:train={tuple(task.train_images.shape)} test={tuple(task.test_images.shape)}"
            for name, task in task_tensors.items()
        )
    )

    weights_cpu, weight_records, weight_key = _load_weight_pool(run_dirs[0])
    spec = _spec_for_cfg(base_cfg)
    _log(f"stage=weight_pool rows={len(weight_records)} weights_shape={tuple(weights_cpu.shape)} cache_key={weight_key}")
    first_vae, _first_normalizer, first_val_indices = _load_vae(run_dirs[0], base_cfg, int(weights_cpu.shape[1]), device=device, dtype=dtype)
    del first_vae, _first_normalizer
    start_bank = _start_bank(
        val_indices=first_val_indices,
        weight_records=weight_records,
        weights_count=int(weights_cpu.shape[0]),
        skip_starts=int(args.skip_starts),
        eval_starts=requested_eval_starts if start_bank_csv is not None else cfg_eval_starts,
        start_bank_csv=start_bank_csv,
    )
    effective_eval_starts = int(len(start_bank))
    base_cfg = replace(base_cfg, eval_starts=effective_eval_starts)
    start_bank.to_csv(out_dir / "downstream_start_bank.csv", index=False)
    _log(
        "stage=start_bank "
        f"source={'csv' if start_bank_csv is not None else 'heldout_skip'} "
        f"rows={effective_eval_starts} source_indices={start_bank['source_weight_index'].astype(int).head(10).tolist()}... "
        f"csv={start_bank_csv if start_bank_csv is not None else ''} "
        f"path={out_dir / 'downstream_start_bank.csv'}"
    )

    weights_device = weights_cpu.to(device=device, dtype=dtype)
    start_indices = torch.as_tensor(start_bank["source_weight_index"].astype("int64").to_numpy(copy=True), dtype=torch.long, device=device)
    starts = weights_device.index_select(0, start_indices)
    start_records = start_bank.reset_index(drop=True)

    all_results: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "runs": args.run,
        "labels": labels,
        "output_dir": str(out_dir),
        "eval_starts": int(effective_eval_starts),
        "requested_eval_starts": int(requested_eval_starts) if requested_eval_starts is not None else None,
        "skip_starts": int(args.skip_starts),
        "start_bank_source": "csv" if start_bank_csv is not None else "heldout_skip",
        "start_bank_csv": str(start_bank_csv) if start_bank_csv is not None else "",
        "downstream_steps": int(base_cfg.downstream_steps),
        "methods": list(args.methods),
        "decoder_sgd_lr": float(args.decoder_sgd_lr) if args.decoder_sgd_lr is not None else None,
        "decoder_sgd_lr_grid": decoder_sgd_lr_grid,
        "decoder_sgd_tune_starts": int(args.decoder_sgd_tune_starts),
        "decoder_sgd_tune_label": str(args.decoder_sgd_tune_label or labels[0]),
        "decoder_preserve_lr": float(args.decoder_preserve_lr) if args.decoder_preserve_lr is not None else None,
        "decoder_preserve_lr_grid": decoder_preserve_lr_grid,
        "decoder_preserve_tune_starts": int(args.decoder_preserve_tune_starts),
        "decoder_preserve_tune_label": str(args.decoder_preserve_tune_label or labels[0]),
        "decoder_preserve_donor": "raw_w0",
        "decoder_preserve_gradient_policy": "donor_detached_zero_grad_on_preserved_block",
        "downstream_sgd_momentum": float(args.downstream_sgd_momentum),
        "device": str(device),
        "dtype": str(dtype),
        "source_indices": [int(v) for v in start_bank["source_weight_index"].astype(int).tolist()],
    }

    tuned_decoder_sgd_lrs: dict[str, float] = {}
    if needs_sgd_lr and args.decoder_sgd_lr is None:
        tune_label = str(args.decoder_sgd_tune_label or labels[0])
        if tune_label not in labels:
            raise ValueError(f"--decoder-sgd-tune-label {tune_label!r} is not in labels={labels}")
        tune_idx = labels.index(tune_label)
        tune_run_dir = run_dirs[tune_idx]
        tune_cfg = _load_cfg(tune_run_dir, device=str(args.device), eval_starts=effective_eval_starts, downstream_steps=args.downstream_steps)
        tune_cfg = replace(tune_cfg, downstream_sgd_momentum=float(args.downstream_sgd_momentum))
        tune_weights, _records, tune_weight_key = _load_weight_pool(tune_run_dir)
        if tuple(tune_weights.shape) != tuple(weights_cpu.shape) or tune_weight_key != weight_key:
            raise RuntimeError(f"weight pool mismatch for SGD LR tune label={tune_label}")
        tune_vae, tune_normalizer, _tune_val_indices = _load_vae(
            tune_run_dir,
            tune_cfg,
            int(weights_cpu.shape[1]),
            device=device,
            dtype=dtype,
        )
        tuned_decoder_sgd_lrs = _tune_decoder_sgd_lrs(
            cfg=tune_cfg,
            spec=spec,
            vae=tune_vae,
            normalizer=tune_normalizer,
            task_tensors=task_tensors,
            starts=starts,
            start_records=start_records,
            methods=list(args.methods),
            lr_grid=decoder_sgd_lr_grid,
            tune_starts=int(args.decoder_sgd_tune_starts),
            tune_label=tune_label,
            out_dir=out_dir,
        )
        manifest["decoder_sgd_lrs"] = dict(tuned_decoder_sgd_lrs)
    elif args.decoder_sgd_lr is not None:
        tuned_decoder_sgd_lrs = {
            str(method): float(args.decoder_sgd_lr)
            for method in args.methods
            if str(method).startswith("decoder_latent_sgd")
        }
        manifest["decoder_sgd_lrs"] = dict(tuned_decoder_sgd_lrs)

    tuned_decoder_preserve_lrs: dict[str, float] = {}
    if needs_preserve_lr and args.decoder_preserve_lr is None:
        tune_label = str(args.decoder_preserve_tune_label or labels[0])
        if tune_label not in labels:
            raise ValueError(f"--decoder-preserve-tune-label {tune_label!r} is not in labels={labels}")
        tune_idx = labels.index(tune_label)
        tune_run_dir = run_dirs[tune_idx]
        tune_cfg = _load_cfg(tune_run_dir, device=str(args.device), eval_starts=effective_eval_starts, downstream_steps=args.downstream_steps)
        tune_weights, _records, tune_weight_key = _load_weight_pool(tune_run_dir)
        if tuple(tune_weights.shape) != tuple(weights_cpu.shape) or tune_weight_key != weight_key:
            raise RuntimeError(f"weight pool mismatch for preserve LR tune label={tune_label}")
        tune_vae, tune_normalizer, _tune_val_indices = _load_vae(
            tune_run_dir,
            tune_cfg,
            int(weights_cpu.shape[1]),
            device=device,
            dtype=dtype,
        )
        tuned_decoder_preserve_lrs = _tune_decoder_preserve_lrs(
            cfg=tune_cfg,
            spec=spec,
            vae=tune_vae,
            normalizer=tune_normalizer,
            task_tensors=task_tensors,
            starts=starts,
            start_records=start_records,
            methods=list(args.methods),
            lr_grid=decoder_preserve_lr_grid,
            tune_starts=int(args.decoder_preserve_tune_starts),
            tune_label=tune_label,
            out_dir=out_dir,
        )
        manifest["decoder_preserve_lrs"] = dict(tuned_decoder_preserve_lrs)
    elif args.decoder_preserve_lr is not None:
        tuned_decoder_preserve_lrs = {
            str(method): float(args.decoder_preserve_lr)
            for method in args.methods
            if str(method) in BLOCK_PRESERVED_LATENT_DOWNSTREAM_METHODS
        }
        manifest["decoder_preserve_lrs"] = dict(tuned_decoder_preserve_lrs)

    for run_dir, label in zip(run_dirs, labels, strict=True):
        cached_results = out_dir / f"downstream_results_{label}.csv"
        cached_curves = out_dir / f"downstream_curves_{label}.csv"
        cfg = _load_cfg(run_dir, device=str(args.device), eval_starts=effective_eval_starts, downstream_steps=args.downstream_steps)
        cfg = replace(cfg, downstream_sgd_momentum=float(args.downstream_sgd_momentum))
        raw_lr = _selected_lr(run_dir, "raw", args.raw_lr)
        decoder_lr = _selected_lr(run_dir, "decoder_latent", args.decoder_lr)
        selected_rows = [
            {"label": label, "method": "raw", "candidate_lr": raw_lr, "selected": 1},
            {"label": label, "method": "decoder_latent", "candidate_lr": decoder_lr, "selected": 1},
        ]
        for method, lr_value in tuned_decoder_sgd_lrs.items():
            selected_rows.append({"label": label, "method": method, "candidate_lr": float(lr_value), "selected": 1})
        for method, lr_value in tuned_decoder_preserve_lrs.items():
            selected_rows.append({"label": label, "method": method, "candidate_lr": float(lr_value), "selected": 1})
        selected = pd.DataFrame(selected_rows)
        cache_meta_path = out_dir / f"downstream_cache_meta_{label}.json"
        expected_cache_meta = _expected_cache_meta(
            label=label,
            run_dir=run_dir,
            cfg=cfg,
            methods=list(args.methods),
            eval_starts=effective_eval_starts,
            skip_starts=int(args.skip_starts) if start_bank_csv is None else 0,
            raw_lr=raw_lr,
            decoder_lr=decoder_lr,
            start_bank=start_bank,
            start_bank_csv=start_bank_csv,
        )
        expected_cache_meta["decoder_sgd_lr"] = float(args.decoder_sgd_lr) if args.decoder_sgd_lr is not None else None
        expected_cache_meta["decoder_sgd_lrs"] = dict(tuned_decoder_sgd_lrs)
        expected_cache_meta["decoder_sgd_lr_grid"] = decoder_sgd_lr_grid
        expected_cache_meta["decoder_sgd_tune_starts"] = int(args.decoder_sgd_tune_starts)
        expected_cache_meta["decoder_sgd_tune_label"] = str(args.decoder_sgd_tune_label or labels[0])
        expected_cache_meta["decoder_preserve_lr"] = float(args.decoder_preserve_lr) if args.decoder_preserve_lr is not None else None
        expected_cache_meta["decoder_preserve_lrs"] = dict(tuned_decoder_preserve_lrs)
        expected_cache_meta["decoder_preserve_lr_grid"] = decoder_preserve_lr_grid
        expected_cache_meta["decoder_preserve_tune_starts"] = int(args.decoder_preserve_tune_starts)
        expected_cache_meta["decoder_preserve_tune_label"] = str(args.decoder_preserve_tune_label or labels[0])
        expected_cache_meta["decoder_preserve_donor"] = "raw_w0"
        expected_cache_meta["decoder_preserve_gradient_policy"] = "donor_detached_zero_grad_on_preserved_block"
        expected_cache_meta["downstream_sgd_momentum"] = float(args.downstream_sgd_momentum)
        if cached_results.is_file() and cached_curves.is_file() and not bool(args.force):
            if cache_meta_path.is_file():
                actual_cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                cache_ok, cache_reason = _cache_meta_matches(actual_cache_meta, expected_cache_meta)
            else:
                cache_ok, cache_reason = False, "missing cache metadata"
            if cache_ok:
                results = pd.read_csv(cached_results)
                curves = pd.read_csv(cached_curves)
                _validate_cached_frames(results=results, curves=curves, label=label, methods=list(args.methods), eval_starts=effective_eval_starts)
                _log(f"cache_hit label={label} results={cached_results} curves={cached_curves} meta={cache_meta_path}")
                all_results.append(results)
                all_curves.append(curves)
                selected_path = out_dir / f"selected_lrs_{label}.csv"
                if selected_path.is_file():
                    all_selected.append(pd.read_csv(selected_path))
                else:
                    all_selected.append(selected)
                continue
            _log(f"cache_stale label={label} reason={cache_reason}; recomputing")
        run_weights, _records, run_weight_key = _load_weight_pool(run_dir)
        if tuple(run_weights.shape) != tuple(weights_cpu.shape) or run_weight_key != weight_key:
            raise RuntimeError(f"weight pool mismatch for {run_dir}")
        vae, normalizer, val_indices = _load_vae(run_dir, cfg, int(weights_cpu.shape[1]), device=device, dtype=dtype)
        if not torch.equal(val_indices, first_val_indices):
            _log(f"WARNING val_indices differ for label={label}; using first run start bank for pairing")
        all_selected.append(selected)
        selected.to_csv(out_dir / f"selected_lrs_{label}.csv", index=False)
        _log(f"stage=eval label={label} run_dir={run_dir} raw_lr={raw_lr:g} decoder_lr={decoder_lr:g}")
        ctx = DownstreamContext(cfg=cfg, spec=spec, vae=vae, normalizer=normalizer, task_tensors=task_tensors, stop_path=None)
        curve_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        total = len(args.methods) * int(effective_eval_starts)
        progress = make_progress(cfg, total=total, desc=f"extended downstream {label}")
        try:
            for method in args.methods:
                if method == "raw":
                    lr = raw_lr
                elif method == "decoder_latent":
                    lr = decoder_lr
                elif method in BLOCK_PRESERVED_LATENT_DOWNSTREAM_METHODS:
                    lr = float(tuned_decoder_preserve_lrs[str(method)])
                else:
                    lr = float(tuned_decoder_sgd_lrs[str(method)])
                for eval_idx in range(int(effective_eval_starts)):
                    metadata = start_records.iloc[eval_idx].to_dict()
                    rows, metrics = run_downstream_curve(
                        ctx=ctx,
                        w0=starts[eval_idx],
                        method=str(method),
                        lr=float(lr),
                        steps=int(cfg.downstream_steps),
                        start_index=int(eval_idx),
                        start_metadata=metadata,
                        split="eval",
                    )
                    for row in rows:
                        row["label"] = label
                    metrics["label"] = label
                    curve_rows.extend(rows)
                    result_rows.append(metrics)
                    progress.set_postfix(
                        {
                            "method": method,
                            "lr": f"{lr:.1e}",
                            "start": eval_idx,
                            "aulc": f"{float(metrics['aulc']):.4g}",
                        }
                    )
                    progress.update(1)
        finally:
            progress.close()
        results = pd.DataFrame(result_rows)
        curves = pd.DataFrame(curve_rows)
        results.to_csv(out_dir / f"downstream_results_{label}.csv", index=False)
        curves.to_csv(out_dir / f"downstream_curves_{label}.csv", index=False)
        cache_meta_path.write_text(json.dumps(expected_cache_meta, indent=2, sort_keys=True), encoding="utf-8")
        all_results.append(results)
        all_curves.append(curves)
        _log(
            f"done_label={label} results={out_dir / f'downstream_results_{label}.csv'} "
            f"curves={out_dir / f'downstream_curves_{label}.csv'} "
            f"decoder_median_aulc={float(results[results['method'].astype(str) == 'decoder_latent']['aulc'].median()):.6g}"
        )

    combined_results = pd.concat(all_results, ignore_index=True, sort=False)
    combined_curves = pd.concat(all_curves, ignore_index=True, sort=False)
    combined_selected = pd.concat(all_selected, ignore_index=True, sort=False)
    combined_results.to_csv(out_dir / "downstream_results.csv", index=False)
    combined_curves.to_csv(out_dir / "downstream_curves.csv", index=False)
    combined_selected.to_csv(out_dir / "selected_lrs.csv", index=False)
    _write_block_preserve_validation(combined_results, out_dir)
    summary = _paired_summary(combined_results, combined_curves, labels=labels, out_dir=out_dir)
    manifest["summary_rows"] = int(len(summary))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not summary.empty:
        _log("paired_summary\n" + summary.to_string(index=False))
    _log(f"done elapsed_sec={time.time() - started:.2f} output_dir={out_dir}")


if __name__ == "__main__":
    main()
