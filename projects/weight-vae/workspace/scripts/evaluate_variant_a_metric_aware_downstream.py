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
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    decode_weights,
    decoder_jacobians,
    encode_weights,
    load_torch_cache,
    logits_from_flat,
    spec_from_payload,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.progress import make_progress


ARTIFACT_ROOT = Path("artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing").resolve()


def _log(message: str) -> None:
    print(f"[metric_aware_downstream] {message}", flush=True)


def _run_dir(run_name: str) -> Path:
    path = Path(run_name).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / run_name).resolve()


def _load_cfg(run_dir: Path, *, device: str, eval_starts: int, downstream_steps: int | None) -> ExperimentConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    cfg = ExperimentConfig(**raw_cfg)
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


def _selected_lr(run_dir: Path, method: str, fallback: float | None) -> float:
    if fallback is not None and math.isfinite(float(fallback)) and float(fallback) > 0.0:
        return float(fallback)
    selected_path = run_dir / "selected_lrs.csv"
    rows = pd.read_csv(selected_path)
    sub = rows[
        (rows["method"].astype(str) == str(method))
        & (pd.to_numeric(rows["selected"], errors="coerce").fillna(0).astype(int) == 1)
    ]
    if sub.empty:
        raise RuntimeError(f"no selected LR for method={method!r} in {selected_path}")
    return float(sub.iloc[0]["candidate_lr"])


def _load_weight_pool(run_dir: Path) -> tuple[torch.Tensor, pd.DataFrame, str, Any]:
    payload = load_torch_cache(run_dir / "weight_pool.pt")
    if payload is None or not isinstance(payload.get("weights"), torch.Tensor):
        raise RuntimeError(f"could not load weight_pool.pt from {run_dir}")
    records = pd.DataFrame(payload.get("records", []))
    if records.empty:
        records_path = run_dir / "weight_pool_records.csv"
        if not records_path.is_file():
            raise FileNotFoundError(records_path)
        records = pd.read_csv(records_path)
    return payload["weights"].detach().cpu(), records, str(payload.get("cache_key", "")), spec_from_payload(payload["spec"])


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
    clamped = val_indices.clamp(min=0, max=max(0, int(weights_count) - 1))
    selected = val_indices[record_steps.index_select(0, clamped) == final_step]
    if int(selected.numel()) >= int(required):
        return [int(v) for v in selected[: int(required)].tolist()]
    final_all = weight_records.index[pd.to_numeric(weight_records["step"], errors="coerce") == final_step].to_numpy(copy=True)
    if int(final_all.shape[0]) >= int(required):
        return [int(v) for v in final_all[: int(required)].tolist()]
    return [int(v) for v in val_indices[: min(int(required), int(val_indices.numel()))].tolist()]


def _start_bank(
    *,
    val_indices: torch.Tensor,
    weight_records: pd.DataFrame,
    weights_count: int,
    skip_starts: int,
    eval_starts: int,
    source_indices: list[int] | None,
    start_bank_csv: Path | None = None,
) -> pd.DataFrame:
    if start_bank_csv is not None:
        rows = pd.read_csv(start_bank_csv)
        if "source_weight_index" not in rows.columns:
            raise ValueError(f"{start_bank_csv} has no source_weight_index column")
        if source_indices:
            order = {int(v): idx for idx, v in enumerate(source_indices)}
            rows = rows[rows["source_weight_index"].astype(int).isin(order)].copy()
            if len(rows) != len(order):
                got = set(rows["source_weight_index"].astype(int).tolist())
                missing = [int(v) for v in source_indices if int(v) not in got]
                raise ValueError(f"{start_bank_csv} missing source indices: {missing}")
            rows["_order"] = rows["source_weight_index"].astype(int).map(order)
            rows = rows.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        if "start_bank_position" not in rows.columns:
            rows.insert(0, "start_bank_position", list(range(len(rows))))
        rows["start_role"] = rows.get("start_role", "eval_metric_aware")
        rows["selection"] = rows.get("selection", f"csv:{start_bank_csv}")
        return rows.reset_index(drop=True)
    if source_indices:
        positions = [int(v) for v in source_indices]
    else:
        required = int(skip_starts) + int(eval_starts)
        all_positions = _heldout_final_indices(
            val_indices=val_indices,
            weight_records=weight_records,
            weights_count=int(weights_count),
            required=required,
        )
        if len(all_positions) < required:
            raise RuntimeError(f"not enough heldout starts: got={len(all_positions)} required={required}")
        positions = all_positions[int(skip_starts) : int(skip_starts) + int(eval_starts)]
    rows = weight_records.iloc[positions].copy().reset_index(drop=True)
    rows["source_weight_index"] = [int(v) for v in positions]
    rows.insert(0, "start_bank_position", list(range(len(rows))))
    rows["start_role"] = "eval_metric_aware"
    rows["selection"] = "explicit_source_indices" if source_indices else f"extended_heldout_final_after_skip_{int(skip_starts)}"
    return rows


def _task_tensors(task_tensors: dict[str, Any], task_name: str):
    if task_name in task_tensors:
        return task_tensors[task_name]
    if len(task_tensors) == 1:
        return next(iter(task_tensors.values()))
    raise KeyError(f"task {task_name!r} not found")


def _batch_indices(task_set: Any, *, batch_size: int, step: int, start_index: int) -> torch.Tensor | None:
    train_count = int(task_set.train_labels.shape[0])
    if int(batch_size) <= 0 or int(batch_size) >= train_count:
        return None
    offset = (int(start_index) * 1009 + int(step) * int(batch_size)) % train_count
    return (torch.arange(int(batch_size), device=task_set.train_labels.device) + offset).remainder(train_count).long()


def _loss_acc(
    flat: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    split: str,
    tau: float,
    batch_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        images, labels = task_set.train_images, task_set.train_labels
        if batch_indices is not None:
            images = images.index_select(0, batch_indices)
            labels = labels.index_select(0, batch_indices)
    elif split == "test":
        images, labels = task_set.test_images, task_set.test_labels
    else:
        raise ValueError(f"unknown split={split!r}")
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc


def _finite_float(value: torch.Tensor | float, *, penalty: float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    if math.isfinite(float(value)):
        return float(value)
    return float(penalty)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().double().flatten()
    b64 = b.detach().double().flatten()
    denom = a64.norm() * b64.norm()
    if float(denom.detach().cpu().item()) <= 1e-30:
        return float("nan")
    return float((torch.dot(a64, b64) / denom).detach().cpu().item())


def _eval_theta(
    theta: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    tau: float,
    penalty: float,
) -> dict[str, float]:
    with torch.no_grad():
        train_loss, train_acc = _loss_acc(theta.detach(), task_set=task_set, spec=spec, split="train", tau=tau)
        test_loss, test_acc = _loss_acc(theta.detach(), task_set=task_set, spec=spec, split="test", tau=tau)
    return {
        "train_loss": _finite_float(train_loss, penalty=penalty),
        "train_acc": float(train_acc.detach().cpu().item()),
        "test_loss": _finite_float(test_loss, penalty=penalty),
        "test_acc": float(test_acc.detach().cpu().item()),
    }


def _linear_norm(jacobian: torch.Tensor, dz: torch.Tensor) -> float:
    return float((jacobian.detach().double() @ dz.detach().double()).norm().detach().cpu().item())


def _natural_direction(
    *,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    z: torch.Tensor,
    theta: torch.Tensor,
    grad_theta: torch.Tensor,
    damping_rel: float,
    jacobian_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    jacobian = decoder_jacobians(
        vae,
        normalizer,
        None,
        z.reshape(1, -1),
        create_graph=False,
        chunk_size=int(jacobian_chunk_size),
    ).squeeze(0).detach()
    j64 = jacobian.double()
    g64 = grad_theta.detach().double()
    metric = j64.T @ j64
    eig = torch.linalg.eigvalsh(metric.float()).detach()
    trace = float(torch.trace(metric).detach().cpu().item())
    eig_mean = trace / max(float(metric.shape[0]), 1.0)
    damping_abs = float(damping_rel) * max(eig_mean, 1e-30)
    eye = torch.eye(int(metric.shape[0]), device=metric.device, dtype=torch.float64)
    dz64 = torch.linalg.solve(metric + damping_abs * eye, -(j64.T @ g64))
    linear = j64 @ dz64
    grad_norm = float(g64.norm().detach().cpu().item())
    linear_norm = float(linear.norm().detach().cpu().item())
    residual_norm = float(((-g64) - linear).norm().detach().cpu().item())
    row = {
        "damping_abs": damping_abs,
        "metric_condition": float(eig.max().detach().cpu().item() / max(float(eig.min().detach().cpu().item()), 1e-30)),
        "metric_trace_per_dim": eig_mean,
        "natural_linear_norm": linear_norm,
        "grad_theta_norm": grad_norm,
        "natural_projection_cos": _cosine(linear, -grad_theta),
        "normal_residual_fraction": residual_norm / max(grad_norm, 1e-30),
        "natural_dz_norm": float(dz64.norm().detach().cpu().item()),
        "theta_norm": float(theta.detach().float().norm().cpu().item()),
    }
    return dz64.to(device=z.device, dtype=z.dtype), row | {"jacobian": jacobian}


def _run_natural_curve(
    *,
    cfg: ExperimentConfig,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    spec: Any,
    task_tensors: dict[str, Any],
    w0: torch.Tensor,
    start_metadata: dict[str, Any],
    start_index: int,
    steps: int,
    target_rel_w0: float,
    damping_rel: float,
    multipliers: tuple[float, ...],
    jacobian_chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    task_name = str(start_metadata.get("task_name", "tiny_cnn"))
    tau = float(start_metadata.get("tau", 1.0))
    task_set = _task_tensors(task_tensors, task_name)
    batch_size = int(getattr(cfg, "downstream_batch_size", 0))
    eval_every = max(1, int(getattr(cfg, "downstream_eval_every", 1)))
    penalty = float(cfg.finite_penalty)
    w0_norm = float(w0.detach().float().norm().cpu().item())
    with torch.no_grad():
        z = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0).detach()
        theta0 = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0).detach()
    mismatch = float((theta0 - w0).float().norm().detach().cpu().item() / max(w0_norm, 1e-12))
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    threshold_step = -1
    diverged = False
    target_weight_norm = float(target_rel_w0) * max(w0_norm, 1e-12)
    source_weight_index = int(start_metadata.get("source_weight_index", start_index))
    start_bank_position = int(start_metadata.get("start_bank_position", start_index))
    source_run = int(start_metadata.get("run", -1))
    source_step = int(start_metadata.get("step", -1))
    source_lr = float(start_metadata.get("source_lr", float("nan")))
    batch_start_index = int(start_metadata.get("start_bank_position", start_index))
    started = time.perf_counter()
    for step in range(int(steps) + 1):
        with torch.no_grad():
            theta_eval = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0).detach()
        should_record = step == 0 or step % eval_every == 0 or step == int(steps) or diverged
        if should_record:
            evals = _eval_theta(theta_eval, task_set=task_set, spec=spec, tau=tau, penalty=penalty)
            if threshold_step < 0 and evals["train_loss"] <= float(cfg.success_threshold):
                threshold_step = int(step)
            rows.append(
                {
                    "method": "decoder_latent_natural_ls",
                    "start_index": int(start_index),
                    "start_bank_position": int(start_bank_position),
                    "source_weight_index": int(source_weight_index),
                    "source_run": int(source_run),
                    "source_step": int(source_step),
                    "source_lr": source_lr,
                    "task_name": task_name,
                    "tau": tau,
                    "lr": float(target_rel_w0),
                    "step": int(step),
                    "train_loss": evals["train_loss"],
                    "train_acc": evals["train_acc"],
                    "test_loss": evals["test_loss"],
                    "test_acc": evals["test_acc"],
                    "reconstruction_rel_l2": mismatch,
                    "diverged": bool(diverged),
                }
            )
        if step == int(steps) or diverged:
            break
        batch_indices = _batch_indices(task_set, batch_size=batch_size, step=step, start_index=batch_start_index)
        theta_req = theta_eval.detach().clone().requires_grad_(True)
        train_loss, _ = _loss_acc(theta_req, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
        if not bool(torch.isfinite(train_loss).detach().cpu().item()):
            diverged = True
            continue
        grad_theta = torch.autograd.grad(train_loss, theta_req, retain_graph=False, create_graph=False)[0].detach()
        dz_base, natural_info = _natural_direction(
            vae=vae,
            normalizer=normalizer,
            z=z,
            theta=theta_eval,
            grad_theta=grad_theta,
            damping_rel=float(damping_rel),
            jacobian_chunk_size=int(jacobian_chunk_size),
        )
        jacobian = natural_info.pop("jacobian")
        natural_linear_norm = max(_linear_norm(jacobian, dz_base), 1e-30)
        base_scale = float(target_weight_norm) / natural_linear_norm
        candidates: list[tuple[float, float, torch.Tensor, torch.Tensor]] = []
        for multiplier in multipliers:
            scale = base_scale * float(multiplier)
            dz = dz_base * scale
            with torch.no_grad():
                theta_candidate = decode_weights(vae, normalizer, (z + dz).reshape(1, -1)).squeeze(0).detach()
            loss_candidate, _ = _loss_acc(
                theta_candidate,
                task_set=task_set,
                spec=spec,
                split="train",
                tau=tau,
                batch_indices=batch_indices,
            )
            loss_value = _finite_float(loss_candidate, penalty=penalty)
            candidates.append((loss_value, float(multiplier), dz, theta_candidate))
        # Include no-op. It prevents the line search from taking destructive steps.
        candidates.append((_finite_float(train_loss, penalty=penalty), 0.0, torch.zeros_like(z), theta_eval.detach()))
        candidates.sort(key=lambda item: (item[0], abs(item[1])))
        best_loss, best_multiplier, best_dz, best_theta = candidates[0]
        z = (z + best_dz).detach()
        accepted_linear_norm = _linear_norm(jacobian, best_dz) if float(best_multiplier) != 0.0 else 0.0
        diag_rows.append(
            {
                "method": "decoder_latent_natural_ls",
                "start_index": int(start_index),
                "source_weight_index": int(source_weight_index),
                "task_name": task_name,
                "tau": tau,
                "step": int(step),
                "batch_loss_before": _finite_float(train_loss, penalty=penalty),
                "batch_loss_after": float(best_loss),
                "accepted_multiplier": float(best_multiplier),
                "accepted_linear_step_norm": accepted_linear_norm,
                "accepted_linear_step_rel_w0": accepted_linear_norm / max(w0_norm, 1e-12),
                "target_weight_step_rel_w0": float(target_rel_w0),
                "base_scale": float(base_scale),
                "elapsed_sec_so_far": time.perf_counter() - started,
                **natural_info,
            }
        )
    train_losses = np.array([float(row["train_loss"]) for row in rows], dtype=np.float64)
    test_losses = np.array([float(row["test_loss"]) for row in rows], dtype=np.float64)
    metrics = {
        "method": "decoder_latent_natural_ls",
        "start_index": int(start_index),
        "start_bank_position": int(start_bank_position),
        "source_weight_index": int(source_weight_index),
        "source_run": int(source_run),
        "source_step": int(source_step),
        "source_lr": source_lr,
        "task_name": task_name,
        "tau": tau,
        "lr": float(target_rel_w0),
        "aulc": float(np.mean(np.minimum(train_losses, penalty))),
        "test_mean_loss": float(np.mean(test_losses)),
        "test_trapz_loss": float(np.trapezoid(test_losses) / max(1, len(test_losses) - 1)),
        "step0_test_loss": float(test_losses[0]),
        "post0_test_loss_mean": float(np.mean(test_losses[1:])) if len(test_losses) > 1 else float("nan"),
        "final_train_loss": float(train_losses[-1]),
        "best_train_loss": float(np.min(train_losses)),
        "final_test_loss": float(test_losses[-1]),
        "best_test_loss": float(np.min(test_losses)),
        "steps_to_threshold": int(threshold_step),
        "reconstruction_rel_l2": mismatch,
        "diverged": bool(diverged),
        "natural_noop_fraction": float(np.mean([float(r["accepted_multiplier"]) == 0.0 for r in diag_rows])) if diag_rows else float("nan"),
        "natural_median_normal_residual_fraction": float(np.median([float(r["normal_residual_fraction"]) for r in diag_rows])) if diag_rows else float("nan"),
        "natural_median_projection_cos": float(np.median([float(r["natural_projection_cos"]) for r in diag_rows])) if diag_rows else float("nan"),
    }
    return rows, metrics, diag_rows


def _bootstrap_ci(values: np.ndarray, *, seed: int = 0, reps: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        means[idx] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_summary(results: pd.DataFrame, labels: list[str], out_dir: Path) -> pd.DataFrame:
    if len(labels) != 2:
        return pd.DataFrame()
    control_label, a_label = labels
    c = results[results["label"].astype(str) == control_label].copy()
    a = results[results["label"].astype(str) == a_label].copy()
    keys = ["method", "source_weight_index", "task_name", "tau", "lr"]
    merged = c.merge(a, on=keys, suffixes=("_control", "_a"), validate="one_to_one")
    if merged.empty:
        raise RuntimeError("empty paired natural downstream merge")
    rows: list[dict[str, Any]] = []
    for col in ["test_mean_loss", "test_trapz_loss", "step0_test_loss", "post0_test_loss_mean", "final_test_loss", "aulc"]:
        values = (merged[f"{col}_a"] - merged[f"{col}_control"]).to_numpy(dtype=np.float64)
        lo, hi = _bootstrap_ci(values, seed=173 + len(rows))
        rows.append(
            {
                "method": "decoder_latent_natural_ls",
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
    merged.to_csv(out_dir / "paired_metric_aware_deltas.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "paired_metric_aware_summary.csv", index=False)
    _plot_paired_summary(merged, summary, out_dir / "paired_metric_aware_deltas.png")
    return summary


def _plot_paired_summary(merged: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    deltas = merged.copy()
    deltas["test_mean_loss_delta"] = deltas["test_mean_loss_a"] - deltas["test_mean_loss_control"]
    deltas["post0_test_loss_mean_delta"] = deltas["post0_test_loss_mean_a"] - deltas["post0_test_loss_mean_control"]
    deltas["step0_test_loss_delta"] = deltas["step0_test_loss_a"] - deltas["step0_test_loss_control"]
    deltas = deltas.sort_values("test_mean_loss_delta").reset_index(drop=True)
    x = np.arange(len(deltas))
    colors = np.where(deltas["test_mean_loss_delta"].to_numpy(dtype=np.float64) > 0.0, "#c44e52", "#4c78a8")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].bar(x, deltas["test_mean_loss_delta"], color=colors)
    axes[0, 0].axhline(0.0, color="black", lw=1)
    axes[0, 0].set_title("Metric-Aware Natural-LS Test-Mean Delta")
    axes[0, 0].set_xlabel("matched eval start, sorted")
    axes[0, 0].set_ylabel("A - control")
    axes[0, 1].scatter(deltas["step0_test_loss_delta"], deltas["post0_test_loss_mean_delta"], c=colors)
    axes[0, 1].axhline(0.0, color="black", lw=1)
    axes[0, 1].axvline(0.0, color="black", lw=1)
    axes[0, 1].set_title("Step0 vs Post-Step Delta")
    axes[0, 1].set_xlabel("step0 A - control")
    axes[0, 1].set_ylabel("post0 mean A - control")
    axes[1, 0].scatter(deltas["natural_noop_fraction_control"], deltas["natural_noop_fraction_a"], c=colors)
    axes[1, 0].plot([0, 1], [0, 1], color="black", lw=1)
    axes[1, 0].set_title("Line-Search No-Op Fraction")
    axes[1, 0].set_xlabel("control")
    axes[1, 0].set_ylabel("A")
    lines = []
    for metric in ["test_mean_loss_delta", "post0_test_loss_mean_delta", "final_test_loss_delta", "aulc_delta"]:
        row = summary[summary["metric"] == metric]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"{metric}\n"
            f"  mean={float(r['mean']):.6g} med={float(r['median']):.6g} worse={int(r['worse_count_A_gt_control'])}/{int(r['n'])}\n"
            f"  CI=[{float(r['bootstrap95_low']):.6g}, {float(r['bootstrap95_high']):.6g}]"
        )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.0, 1.0, "\n\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full downstream with damped latent natural-gradient line search.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-starts", type=int, default=16)
    parser.add_argument("--skip-starts", type=int, default=8)
    parser.add_argument("--start-bank-csv", default="")
    parser.add_argument("--source-index", action="append", type=int, default=[])
    parser.add_argument("--downstream-steps", type=int, default=300)
    parser.add_argument("--target-rel-w0", type=float, default=0.0015)
    parser.add_argument("--damping-rel", type=float, default=1e-4)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        raise ValueError("--run and --label counts must match")
    if len(set(args.label)) != len(args.label):
        raise ValueError("--label values must be unique")
    started = time.perf_counter()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [_run_dir(v) for v in args.run]
    labels = [str(v) for v in args.label]
    _log(
        "startup "
        f"device={args.device} labels={labels} eval_starts={args.eval_starts} skip_starts={args.skip_starts} "
        f"steps={args.downstream_steps} target_rel_w0={args.target_rel_w0} damping_rel={args.damping_rel} "
        f"multipliers={args.multipliers} jacobian_chunk_size={args.jacobian_chunk_size} output_dir={out_dir}"
    )
    base_cfg = _load_cfg(run_dirs[0], device=str(args.device), eval_starts=int(args.eval_starts), downstream_steps=int(args.downstream_steps))
    device = torch.device(base_cfg.device)
    dtype = torch_dtype(base_cfg)
    _log(f"stage=load_data dtype={dtype} seed={base_cfg.seed} data_root={base_cfg.data_root}")
    task_tensors = _load_task_tensors_for_pipeline(base_cfg, device=device, dtype=dtype)
    _log("loaded_tasks " + ", ".join(f"{k}:train={tuple(v.train_images.shape)} test={tuple(v.test_images.shape)}" for k, v in task_tensors.items()))
    weights_cpu, weight_records, weight_key, spec = _load_weight_pool(run_dirs[0])
    first_vae, _first_normalizer, first_val_indices = _load_vae(run_dirs[0], base_cfg, int(weights_cpu.shape[1]), device=device, dtype=dtype)
    del first_vae, _first_normalizer
    start_bank = _start_bank(
        val_indices=first_val_indices,
        weight_records=weight_records,
        weights_count=int(weights_cpu.shape[0]),
        skip_starts=int(args.skip_starts),
        eval_starts=int(args.eval_starts),
        source_indices=[int(v) for v in args.source_index],
        start_bank_csv=Path(args.start_bank_csv).expanduser().resolve() if str(args.start_bank_csv).strip() else None,
    )
    start_bank.to_csv(out_dir / "metric_aware_start_bank.csv", index=False)
    _log(
        "stage=start_bank "
        f"rows={len(start_bank)} source_indices={start_bank['source_weight_index'].astype(int).tolist()} "
        f"path={out_dir / 'metric_aware_start_bank.csv'}"
    )
    weights_device = weights_cpu.to(device=device, dtype=dtype)
    start_indices = torch.as_tensor(start_bank["source_weight_index"].astype("int64").to_numpy(copy=True), device=device, dtype=torch.long)
    starts = weights_device.index_select(0, start_indices)
    all_results: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    all_diags: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    for run_dir, label in zip(run_dirs, labels, strict=True):
        result_path = out_dir / f"metric_aware_results_{label}.csv"
        curve_path = out_dir / f"metric_aware_curves_{label}.csv"
        diag_path = out_dir / f"metric_aware_step_diagnostics_{label}.csv"
        if result_path.is_file() and curve_path.is_file() and diag_path.is_file() and not bool(args.force):
            _log(f"cache_hit label={label} results={result_path} curves={curve_path} diagnostics={diag_path}")
            all_results.append(pd.read_csv(result_path))
            all_curves.append(pd.read_csv(curve_path))
            all_diags.append(pd.read_csv(diag_path))
            continue
        cfg = _load_cfg(run_dir, device=str(args.device), eval_starts=int(args.eval_starts), downstream_steps=int(args.downstream_steps))
        run_weights, _records, run_weight_key, _spec = _load_weight_pool(run_dir)
        if tuple(run_weights.shape) != tuple(weights_cpu.shape) or run_weight_key != weight_key:
            raise RuntimeError(f"weight pool mismatch for {run_dir}")
        vae, normalizer, val_indices = _load_vae(run_dir, cfg, int(weights_cpu.shape[1]), device=device, dtype=dtype)
        if not torch.equal(val_indices, first_val_indices):
            _log(f"WARNING val_indices differ for label={label}; using first run start bank")
        raw_lr = _selected_lr(run_dir, "raw", None)
        latent_lr = _selected_lr(run_dir, "decoder_latent", None)
        selected_rows.append({"label": label, "raw_lr": raw_lr, "decoder_latent_lr": latent_lr, "target_rel_w0": float(args.target_rel_w0)})
        _log(f"stage=eval label={label} run_dir={run_dir} raw_lr={raw_lr:g} decoder_lr={latent_lr:g}")
        curve_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        diag_rows: list[dict[str, Any]] = []
        progress = make_progress(cfg, total=len(start_bank), desc=f"metric-aware downstream {label}")
        try:
            for eval_idx in range(len(start_bank)):
                metadata = start_bank.iloc[eval_idx].to_dict()
                rows, metrics, diags = _run_natural_curve(
                    cfg=cfg,
                    vae=vae,
                    normalizer=normalizer,
                    spec=spec,
                    task_tensors=task_tensors,
                    w0=starts[eval_idx],
                    start_metadata=metadata,
                    start_index=int(eval_idx),
                    steps=int(args.downstream_steps),
                    target_rel_w0=float(args.target_rel_w0),
                    damping_rel=float(args.damping_rel),
                    multipliers=tuple(float(v) for v in args.multipliers),
                    jacobian_chunk_size=int(args.jacobian_chunk_size),
                )
                for row in rows:
                    row["label"] = label
                for row in diags:
                    row["label"] = label
                metrics["label"] = label
                curve_rows.extend(rows)
                result_rows.append(metrics)
                diag_rows.extend(diags)
                progress.set_postfix(
                    {
                        "source": int(metadata["source_weight_index"]),
                        "test_mean": f"{float(metrics['test_mean_loss']):.4g}",
                        "noop": f"{float(metrics['natural_noop_fraction']):.2f}",
                    }
                )
                progress.update(1)
        finally:
            progress.close()
        results = pd.DataFrame(result_rows)
        curves = pd.DataFrame(curve_rows)
        diagnostics = pd.DataFrame(diag_rows)
        results.to_csv(result_path, index=False)
        curves.to_csv(curve_path, index=False)
        diagnostics.to_csv(diag_path, index=False)
        all_results.append(results)
        all_curves.append(curves)
        all_diags.append(diagnostics)
        _log(
            f"done_label={label} results={result_path} curves={curve_path} diagnostics={diag_path} "
            f"median_test_mean={float(results['test_mean_loss'].median()):.6g} "
            f"median_noop={float(results['natural_noop_fraction'].median()):.4g}"
        )
    results = pd.concat(all_results, ignore_index=True, sort=False)
    curves = pd.concat(all_curves, ignore_index=True, sort=False)
    diagnostics = pd.concat(all_diags, ignore_index=True, sort=False)
    results.to_csv(out_dir / "metric_aware_results.csv", index=False)
    curves.to_csv(out_dir / "metric_aware_curves.csv", index=False)
    diagnostics.to_csv(out_dir / "metric_aware_step_diagnostics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out_dir / "metric_aware_selected_lrs.csv", index=False)
    summary = _paired_summary(results, labels, out_dir)
    manifest = {
        "runs": [str(v) for v in run_dirs],
        "labels": labels,
        "output_dir": str(out_dir),
        "eval_starts": int(len(start_bank)),
        "skip_starts": int(args.skip_starts),
        "start_bank_csv": str(Path(args.start_bank_csv).expanduser().resolve()) if str(args.start_bank_csv).strip() else "",
        "source_indices": [int(v) for v in start_bank["source_weight_index"].astype(int).tolist()],
        "downstream_steps": int(args.downstream_steps),
        "target_rel_w0": float(args.target_rel_w0),
        "damping_rel": float(args.damping_rel),
        "multipliers": [float(v) for v in args.multipliers],
        "jacobian_chunk_size": int(args.jacobian_chunk_size),
        "summary_rows": int(len(summary)),
        "elapsed_sec": time.perf_counter() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not summary.empty:
        _log("paired_summary\n" + summary.to_string(index=False))
    _log(f"done elapsed_sec={time.perf_counter() - started:.2f} output_dir={out_dir}")


if __name__ == "__main__":
    main()
