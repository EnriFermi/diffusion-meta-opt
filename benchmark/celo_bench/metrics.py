from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def ema(data: Any, alpha: float, *, ignore_nan: bool = False) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        return values.copy()
    out = np.zeros_like(values, dtype=float)
    out[0] = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if ignore_nan and np.isnan(value):
            out[idx] = out[idx - 1]
        else:
            out[idx] = out[idx - 1] * alpha + (1.0 - alpha) * value
    return out


def mean_curve(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    return np.nanmean(arr.reshape((-1, arr.shape[-1])), axis=0)


def final_loss_from_curve(curve: Any, *, alpha: float = 0.9) -> float:
    smoothed = ema(mean_curve(curve), alpha, ignore_nan=True)
    finite = smoothed[np.isfinite(smoothed)]
    if finite.size == 0:
        return float("inf")
    return float(finite[-1])


def final_loss_score(best_adam_final_loss: float, optimizer_final_loss: float) -> float:
    if not np.isfinite(best_adam_final_loss) or not np.isfinite(optimizer_final_loss):
        return 0.0
    if optimizer_final_loss <= 0.0:
        return 0.0
    return float(best_adam_final_loss / optimizer_final_loss)


def speedup_score(xs: Any, optimizer_curve: Any, *, target_loss: float, horizon: int, alpha: float = 0.9) -> float:
    if not np.isfinite(target_loss):
        return 0.0
    steps = np.asarray(xs, dtype=float).reshape(-1)
    curve = ema(mean_curve(optimizer_curve), alpha, ignore_nan=True)
    if steps.size != curve.size:
        raise ValueError(f"xs and optimizer curve lengths differ: {steps.size} != {curve.size}")
    reached = np.where(np.isfinite(curve) & (curve <= target_loss))[0]
    if reached.size == 0:
        return 0.0
    first_step = float(steps[int(reached[0])])
    if first_step <= 0.0:
        return float(horizon)
    return float(horizon / first_step)


def interquartile_mean(scores: Any) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values.sort()
    trim = int(np.floor(values.size * 0.25))
    if trim > 0 and values.size - 2 * trim > 0:
        values = values[trim:-trim]
    return float(np.mean(values))


def optimality_gap(scores: Any) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return float(1.0 - np.mean(np.clip(values, 0.0, 1.0)))


def summarize_scores(scores: Any) -> dict[str, float]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0.0, "median": 0.0, "iqm": 0.0, "optimality_gap": 1.0}
    return {
        "count": float(values.size),
        "median": float(np.median(values)),
        "iqm": interquartile_mean(values),
        "optimality_gap": optimality_gap(values),
    }


def score_curve_pair(
    *,
    xs: Any,
    best_adam_curve: Any,
    optimizer_curve: Any,
    horizon: int,
    alpha: float,
) -> dict[str, float]:
    target = final_loss_from_curve(best_adam_curve, alpha=alpha)
    opt_final = final_loss_from_curve(optimizer_curve, alpha=alpha)
    return {
        "adam_final_loss": target,
        "optimizer_final_loss": opt_final,
        "final_loss_score": final_loss_score(target, opt_final),
        "speedup_score": speedup_score(xs, optimizer_curve, target_loss=target, horizon=horizon, alpha=alpha),
    }


def extract_curve(metrics: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in metrics:
        raise KeyError(f"Metrics missing score split {key!r}; available keys: {sorted(metrics)}")
    return mean_curve(metrics[key])

