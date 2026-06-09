from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .core import FlatSpec, WeightNormalizer, WeightVAE, decode_weights, encode_weights, finite_value, tiny_cnn_logits_from_flat
from .progress import make_progress


@dataclass(slots=True)
class DownstreamContext:
    cfg: ExperimentConfig
    spec: FlatSpec
    vae: WeightVAE
    normalizer: WeightNormalizer
    trained_flow: torch.nn.Module
    random_flow: torch.nn.Module
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def _flat_loss_and_acc(flat: torch.Tensor, ctx: DownstreamContext, *, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        images, labels = ctx.train_images, ctx.train_labels
    elif split == "test":
        images, labels = ctx.test_images, ctx.test_labels
    else:
        raise ValueError(f"split must be train or test, got {split!r}")
    logits = tiny_cnn_logits_from_flat(flat, images, ctx.spec)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc


def _theta_for_method(value: torch.Tensor, method: str, ctx: DownstreamContext) -> torch.Tensor:
    if method == "raw":
        return value
    if method == "decoder_latent":
        return decode_weights(ctx.vae, ctx.normalizer, value.unsqueeze(0)).squeeze(0)
    if method == "decoder_trained_nf":
        z = ctx.trained_flow.inverse(value.unsqueeze(0))[0].squeeze(0)
        return decode_weights(ctx.vae, ctx.normalizer, z.unsqueeze(0)).squeeze(0)
    if method == "decoder_random_nf":
        z = ctx.random_flow.inverse(value.unsqueeze(0))[0].squeeze(0)
        return decode_weights(ctx.vae, ctx.normalizer, z.unsqueeze(0)).squeeze(0)
    raise ValueError(f"unknown method {method!r}")


def _start_for_method(w0: torch.Tensor, method: str, ctx: DownstreamContext) -> tuple[torch.Tensor, float]:
    if method == "raw":
        return w0.detach().clone(), 0.0
    z0 = encode_weights(ctx.vae, ctx.normalizer, w0.reshape(1, -1)).squeeze(0)
    recon = decode_weights(ctx.vae, ctx.normalizer, z0.reshape(1, -1)).squeeze(0)
    mismatch = (recon - w0).float().norm() / w0.float().norm().clamp_min(1e-12)
    if method == "decoder_latent":
        return z0.detach().clone(), float(mismatch.detach().cpu().item())
    if method == "decoder_trained_nf":
        u0 = ctx.trained_flow(z0.reshape(1, -1))[0].squeeze(0)
        return u0.detach().clone(), float(mismatch.detach().cpu().item())
    if method == "decoder_random_nf":
        u0 = ctx.random_flow(z0.reshape(1, -1))[0].squeeze(0)
        return u0.detach().clone(), float(mismatch.detach().cpu().item())
    raise ValueError(f"unknown method {method!r}")


def run_downstream_curve(
    *,
    ctx: DownstreamContext,
    w0: torch.Tensor,
    method: str,
    lr: float,
    steps: int,
    start_index: int,
    split: str,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    value, mismatch = _start_for_method(w0, method, ctx)
    value = value.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([value], lr=float(lr))
    rows: list[dict[str, float | int | str]] = []
    diverged = False
    threshold_step = -1
    for step in range(int(steps) + 1):
        theta = _theta_for_method(value, method, ctx)
        train_loss, train_acc = _flat_loss_and_acc(theta, ctx, split="train")
        with torch.no_grad():
            test_loss, test_acc = _flat_loss_and_acc(theta.detach(), ctx, split="test")
        finite = bool(torch.isfinite(train_loss).detach().cpu().item()) and bool(torch.isfinite(test_loss).detach().cpu().item())
        if not finite:
            diverged = True
        train_loss_value = finite_value(float(train_loss.detach().cpu().item()) if finite else float("inf"), penalty=float(ctx.cfg.finite_penalty))
        test_loss_value = finite_value(float(test_loss.detach().cpu().item()) if finite else float("inf"), penalty=float(ctx.cfg.finite_penalty))
        if threshold_step < 0 and train_loss_value <= float(ctx.cfg.success_threshold):
            threshold_step = int(step)
        rows.append(
            {
                "split": str(split),
                "method": str(method),
                "start_index": int(start_index),
                "lr": float(lr),
                "step": int(step),
                "train_loss": train_loss_value,
                "train_acc": float(train_acc.detach().cpu().item()) if finite else 0.0,
                "test_loss": test_loss_value,
                "test_acc": float(test_acc.detach().cpu().item()) if finite else 0.0,
                "reconstruction_rel_l2": float(mismatch),
                "diverged": bool(diverged),
            }
        )
        if step == int(steps) or diverged:
            break
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
    train_losses = np.array([float(row["train_loss"]) for row in rows], dtype=np.float64)
    test_losses = np.array([float(row["test_loss"]) for row in rows], dtype=np.float64)
    aulc = float(np.mean(np.minimum(train_losses, float(ctx.cfg.finite_penalty))))
    final = rows[-1]
    metrics = {
        "split": str(split),
        "method": str(method),
        "start_index": int(start_index),
        "lr": float(lr),
        "aulc": aulc,
        "final_train_loss": float(final["train_loss"]),
        "best_train_loss": float(np.min(train_losses)),
        "final_test_loss": float(final["test_loss"]),
        "final_test_acc": float(final["test_acc"]),
        "steps_to_threshold": int(threshold_step),
        "reconstruction_rel_l2": float(mismatch),
        "diverged": bool(diverged),
    }
    return rows, metrics


def _lr_grid_for_method(cfg: ExperimentConfig, method: str) -> tuple[float, ...]:
    if method == "raw":
        return tuple(float(v) for v in cfg.raw_lrs)
    if method == "decoder_latent":
        return tuple(float(v) for v in cfg.latent_lrs)
    if method in {"decoder_trained_nf", "decoder_random_nf"}:
        return tuple(float(v) for v in cfg.nf_lrs)
    raise ValueError(f"unknown method {method!r}")


def tune_and_evaluate_downstream(
    *,
    cfg: ExperimentConfig,
    ctx: DownstreamContext,
    starts: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = ("raw", "decoder_latent", "decoder_trained_nf", "decoder_random_nf")
    tune_count = min(int(cfg.tune_starts), int(starts.shape[0]))
    eval_start = tune_count
    eval_count = min(int(cfg.eval_starts), max(0, int(starts.shape[0]) - eval_start))
    if eval_count <= 0:
        eval_start = 0
        eval_count = min(int(cfg.eval_starts), int(starts.shape[0]))
    tuning_rows: list[dict[str, float | int | str | bool]] = []
    selected_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int | str | bool]] = []
    result_rows: list[dict[str, float | int | str | bool]] = []
    total_curves = sum(len(_lr_grid_for_method(cfg, method)) * tune_count + eval_count for method in methods)
    progress = make_progress(cfg, total=total_curves, desc="downstream LR/eval")

    try:
        for method in methods:
            for lr in _lr_grid_for_method(cfg, method):
                lr_metrics: list[float] = []
                for start_idx in range(tune_count):
                    rows, metrics = run_downstream_curve(
                        ctx=ctx,
                        w0=starts[start_idx],
                        method=method,
                        lr=float(lr),
                        steps=int(cfg.downstream_steps),
                        start_index=int(start_idx),
                        split="tune",
                    )
                    tuning_rows.append(metrics)
                    lr_metrics.append(float(metrics["aulc"]))
                    progress.set_postfix({"phase": "tune", "method": method, "lr": f"{float(lr):.1e}", "start": start_idx, "aulc": f"{float(metrics['aulc']):.4g}"})
                    progress.update(1)
                median_aulc = float(np.median(np.array(lr_metrics, dtype=np.float64))) if lr_metrics else float(cfg.finite_penalty)
                selected_rows.append({"method": method, "candidate_lr": float(lr), "tuning_median_aulc": median_aulc})
            method_candidates = [row for row in selected_rows if row["method"] == method]
            best = min(method_candidates, key=lambda row: (float(row["tuning_median_aulc"]), float(row["candidate_lr"])))
            best["selected"] = 1
            selected_lr = float(best["candidate_lr"])
            for eval_idx in range(eval_count):
                start_idx = eval_start + eval_idx
                rows, metrics = run_downstream_curve(
                    ctx=ctx,
                    w0=starts[start_idx],
                    method=method,
                    lr=selected_lr,
                    steps=int(cfg.downstream_steps),
                    start_index=int(eval_idx),
                    split="eval",
                )
                curve_rows.extend(rows)
                result_rows.append(metrics)
                progress.set_postfix({"phase": "eval", "method": method, "lr": f"{selected_lr:.1e}", "start": eval_idx, "aulc": f"{float(metrics['aulc']):.4g}"})
                progress.update(1)
    finally:
        progress.close()

    selected = pd.DataFrame(selected_rows)
    if not selected.empty and "selected" not in selected.columns:
        selected["selected"] = 0
    selected["selected"] = selected.get("selected", 0).fillna(0).astype(int)
    return pd.DataFrame(result_rows), pd.DataFrame(curve_rows), selected
