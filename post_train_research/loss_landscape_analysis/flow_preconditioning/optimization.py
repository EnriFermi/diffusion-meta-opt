from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


LossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class Curve:
    train_loss: np.ndarray
    test_loss: np.ndarray
    final_theta: np.ndarray | None = None


def _make_optimizer(name: str, params: list[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
    value = str(name).strip().lower()
    if value == "sgd":
        return torch.optim.SGD(params, lr=float(lr))
    if value == "adam":
        return torch.optim.Adam(params, lr=float(lr))
    raise ValueError(f"optimizer must be one of {{'sgd', 'adam'}}, got {name!r}")


def run_direct_curve(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    theta0: torch.Tensor,
    optimizer_name: str,
    lr: float,
    steps: int,
) -> Curve:
    theta = torch.nn.Parameter(theta0.detach().clone())
    optimizer = _make_optimizer(optimizer_name, [theta], float(lr))
    train_losses = np.empty(int(steps) + 1, dtype=np.float64)
    test_losses = np.empty(int(steps) + 1, dtype=np.float64)
    for step in range(0, int(steps) + 1):
        with torch.no_grad():
            train_losses[step] = float(train_loss_fn(theta).detach().cpu().item())
            test_losses[step] = float(test_loss_fn(theta).detach().cpu().item())
        if step == int(steps):
            break
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss_fn(theta)
        loss.backward()
        optimizer.step()
    return Curve(
        train_loss=train_losses,
        test_loss=test_losses,
        final_theta=theta.detach().cpu().numpy().astype(np.float64),
    )


def run_flow_curve(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    flow: torch.nn.Module,
    theta0: torch.Tensor,
    optimizer_name: str,
    lr: float,
    steps: int,
) -> Curve:
    flow.eval()
    for param in flow.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        u0 = flow(theta0.detach().reshape(1, -1))[0].reshape(-1)
    u = torch.nn.Parameter(u0.detach().clone())
    optimizer = _make_optimizer(optimizer_name, [u], float(lr))
    train_losses = np.empty(int(steps) + 1, dtype=np.float64)
    test_losses = np.empty(int(steps) + 1, dtype=np.float64)

    def theta_from_u() -> torch.Tensor:
        return flow.inverse(u.reshape(1, -1))[0].reshape(-1)

    for step in range(0, int(steps) + 1):
        with torch.no_grad():
            theta_eval = theta_from_u()
            train_losses[step] = float(train_loss_fn(theta_eval).detach().cpu().item())
            test_losses[step] = float(test_loss_fn(theta_eval).detach().cpu().item())
        if step == int(steps):
            break
        optimizer.zero_grad(set_to_none=True)
        theta = theta_from_u()
        loss = train_loss_fn(theta)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_theta = theta_from_u().detach().cpu().numpy().astype(np.float64)
    return Curve(train_loss=train_losses, test_loss=test_losses, final_theta=final_theta)


def aulc(train_losses: np.ndarray, *, budget: int, l_ref: float, eps: float) -> float:
    values = np.asarray(train_losses[: int(budget) + 1], dtype=np.float64) - float(l_ref) + float(eps)
    values = np.maximum(values, float(eps))
    return float(np.log10(values).sum())


def curve_metrics(
    curve: Curve,
    *,
    budget: int,
    l_ref: float,
    eps: float,
    success_threshold: float | None,
) -> dict[str, float]:
    train = np.asarray(curve.train_loss[: int(budget) + 1], dtype=np.float64)
    test = np.asarray(curve.test_loss[: int(budget) + 1], dtype=np.float64)
    best_train = float(np.min(train))
    threshold = float(success_threshold) if success_threshold is not None else float("nan")
    success = bool(math.isfinite(threshold) and best_train <= threshold)
    return {
        "final_train_loss": float(train[-1]),
        "best_train_loss": best_train,
        "final_test_loss": float(test[-1]),
        "best_test_loss": float(np.min(test)),
        "aulc": aulc(train, budget=int(budget), l_ref=float(l_ref), eps=float(eps)),
        "l_ref": float(l_ref),
        "success_threshold": threshold,
        "success": float(success),
    }


def curve_rows(
    curve: Curve,
    *,
    curve_id: str,
    experiment: str,
    split: str,
    seed: int,
    rho: float,
    method: str,
    optimizer: str,
    lr: float,
    start_index: int,
    budget: int,
) -> list[dict[str, object]]:
    rows = []
    for step, (train_loss, test_loss) in enumerate(zip(curve.train_loss, curve.test_loss, strict=True)):
        rows.append(
            {
                "curve_id": curve_id,
                "experiment": experiment,
                "split": split,
                "seed": int(seed),
                "rho": float(rho),
                "method": method,
                "optimizer": optimizer,
                "lr": float(lr),
                "start_index": int(start_index),
                "budget": int(budget),
                "step": int(step),
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
            }
        )
    return rows
