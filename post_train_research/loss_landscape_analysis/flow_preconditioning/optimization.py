from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


LossFn = Callable[[torch.Tensor], torch.Tensor]
BatchLossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class Curve:
    train_loss: np.ndarray
    test_loss: np.ndarray
    final_theta: np.ndarray | None = None
    path: np.ndarray | None = None


def _vmap_scalar_loss(loss_fn: LossFn, values: torch.Tensor) -> torch.Tensor:
    try:
        from torch.func import vmap

        return vmap(loss_fn)(values)
    except Exception:
        return torch.stack([loss_fn(value) for value in values], dim=0)


def _as_batch_losses(losses: torch.Tensor, batch: int) -> torch.Tensor:
    values = losses.reshape(-1)
    if int(values.shape[0]) != int(batch):
        raise ValueError(f"batched loss must return [{batch}], got {tuple(losses.shape)}")
    return values


def _optimizer_name(name: str) -> str:
    value = str(name).strip().lower()
    if value not in {"sgd", "adam"}:
        raise ValueError(f"optimizer must be one of {{'sgd', 'adam'}}, got {name!r}")
    return value


def _manual_optimizer_step(
    value: torch.Tensor,
    grad: torch.Tensor,
    *,
    optimizer_name: str,
    lr: float,
    step: int,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    if optimizer_name == "sgd":
        return value - float(lr) * grad
    if optimizer_name == "adam":
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        if "m" not in state:
            state["m"] = torch.zeros_like(value)
            state["v"] = torch.zeros_like(value)
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * grad.square()
        m_hat = state["m"] / (1.0 - beta1 ** int(step))
        v_hat = state["v"] / (1.0 - beta2 ** int(step))
        return value - float(lr) * m_hat / (v_hat.sqrt() + eps)
    raise ValueError(f"unknown optimizer {optimizer_name!r}")


def run_direct_curve(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    theta0: torch.Tensor,
    optimizer_name: str,
    lr: float,
    steps: int,
    store_path: bool = False,
) -> Curve:
    optimizer_value = _optimizer_name(optimizer_name)
    theta = theta0.detach().clone().requires_grad_(True)
    train_losses_device = torch.empty(int(steps) + 1, device=theta.device, dtype=torch.float64)
    test_losses_device = torch.empty(int(steps) + 1, device=theta.device, dtype=torch.float64)
    path = np.empty((int(steps) + 1, int(theta0.numel())), dtype=np.float64) if store_path else None
    opt_state: dict[str, torch.Tensor] = {}
    for step in range(0, int(steps) + 1):
        loss = train_loss_fn(theta)
        train_losses_device[step] = loss.detach().to(dtype=torch.float64)
        with torch.no_grad():
            test_losses_device[step] = test_loss_fn(theta).detach().to(dtype=torch.float64)
            if path is not None:
                path[step] = theta.detach().cpu().numpy().astype(np.float64)
        if step == int(steps):
            break
        grad = torch.autograd.grad(loss, theta)[0]
        theta = _manual_optimizer_step(
            theta,
            grad,
            optimizer_name=optimizer_value,
            lr=float(lr),
            step=step + 1,
            state=opt_state,
        ).detach().requires_grad_(True)
    train_losses = train_losses_device.detach().cpu().numpy()
    test_losses = test_losses_device.detach().cpu().numpy()
    return Curve(
        train_loss=train_losses,
        test_loss=test_losses,
        final_theta=theta.detach().cpu().numpy().astype(np.float64),
        path=path,
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
    store_path: bool = False,
) -> Curve:
    optimizer_value = _optimizer_name(optimizer_name)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        u0 = flow(theta0.detach().reshape(1, -1))[0].reshape(-1)
    u = u0.detach().clone().requires_grad_(True)
    train_losses_device = torch.empty(int(steps) + 1, device=u.device, dtype=torch.float64)
    test_losses_device = torch.empty(int(steps) + 1, device=u.device, dtype=torch.float64)
    path = np.empty((int(steps) + 1, int(theta0.numel())), dtype=np.float64) if store_path else None
    opt_state: dict[str, torch.Tensor] = {}

    def theta_from_u() -> torch.Tensor:
        return flow.inverse(u.reshape(1, -1))[0].reshape(-1)

    for step in range(0, int(steps) + 1):
        theta = theta_from_u()
        loss = train_loss_fn(theta)
        train_losses_device[step] = loss.detach().to(dtype=torch.float64)
        with torch.no_grad():
            test_losses_device[step] = test_loss_fn(theta).detach().to(dtype=torch.float64)
            if path is not None:
                path[step] = theta.detach().cpu().numpy().astype(np.float64)
        if step == int(steps):
            break
        grad = torch.autograd.grad(loss, u)[0]
        u = _manual_optimizer_step(
            u,
            grad,
            optimizer_name=optimizer_value,
            lr=float(lr),
            step=step + 1,
            state=opt_state,
        ).detach().requires_grad_(True)
    with torch.no_grad():
        final_theta = theta_from_u().detach().cpu().numpy().astype(np.float64)
    train_losses = train_losses_device.detach().cpu().numpy()
    test_losses = test_losses_device.detach().cpu().numpy()
    return Curve(train_loss=train_losses, test_loss=test_losses, final_theta=final_theta, path=path)


def run_direct_curves_batched(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    theta0_batch: torch.Tensor,
    optimizer_name: str,
    lrs: torch.Tensor,
    steps: int,
) -> list[Curve]:
    return run_direct_curves_batched_loss(
        train_loss_batch_fn=lambda values: _vmap_scalar_loss(train_loss_fn, values),
        test_loss_batch_fn=lambda values: _vmap_scalar_loss(test_loss_fn, values),
        theta0_batch=theta0_batch,
        optimizer_name=optimizer_name,
        lrs=lrs,
        steps=steps,
    )


def run_direct_curves_batched_loss(
    *,
    train_loss_batch_fn: BatchLossFn,
    test_loss_batch_fn: BatchLossFn,
    theta0_batch: torch.Tensor,
    optimizer_name: str,
    lrs: torch.Tensor,
    steps: int,
) -> list[Curve]:
    optimizer_value = _optimizer_name(optimizer_name)
    theta = theta0_batch.detach().clone().requires_grad_(True)
    lr_values = lrs.detach().to(device=theta.device, dtype=theta.dtype).reshape(-1, 1)
    batch = int(theta.shape[0])
    train_losses_device = torch.empty(int(steps) + 1, batch, device=theta.device, dtype=torch.float64)
    test_losses_device = torch.empty(int(steps) + 1, batch, device=theta.device, dtype=torch.float64)
    opt_state: dict[str, torch.Tensor] = {}
    for step in range(0, int(steps) + 1):
        losses = _as_batch_losses(train_loss_batch_fn(theta), batch)
        train_losses_device[step] = losses.detach().to(dtype=torch.float64)
        with torch.no_grad():
            test_losses_device[step] = _as_batch_losses(test_loss_batch_fn(theta), batch).detach().to(dtype=torch.float64)
        if step == int(steps):
            break
        grad = torch.autograd.grad(losses.sum(), theta)[0]
        theta = _manual_optimizer_step(
            theta,
            grad,
            optimizer_name=optimizer_value,
            lr=1.0,
            step=step + 1,
            state=opt_state,
        )
        if optimizer_value == "sgd":
            # _manual_optimizer_step with lr=1 applies a unit update; scale SGD explicitly per row.
            theta = (theta + grad - lr_values * grad).detach().requires_grad_(True)
        else:
            # Recompute Adam update with row-wise lr because lr is scalar in the generic helper.
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            m_hat = opt_state["m"] / (1.0 - beta1 ** int(step + 1))
            v_hat = opt_state["v"] / (1.0 - beta2 ** int(step + 1))
            theta = (theta + m_hat / (v_hat.sqrt() + eps) - lr_values * m_hat / (v_hat.sqrt() + eps)).detach().requires_grad_(True)
    train_losses = train_losses_device.detach().cpu().numpy()
    test_losses = test_losses_device.detach().cpu().numpy()
    final_theta = theta.detach().cpu().numpy().astype(np.float64)
    return [
        Curve(train_loss=train_losses[:, idx], test_loss=test_losses[:, idx], final_theta=final_theta[idx])
        for idx in range(batch)
    ]


def run_flow_curves_batched(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    flow: torch.nn.Module,
    theta0_batch: torch.Tensor,
    optimizer_name: str,
    lrs: torch.Tensor,
    steps: int,
) -> list[Curve]:
    return run_flow_curves_batched_loss(
        train_loss_batch_fn=lambda values: _vmap_scalar_loss(train_loss_fn, values),
        test_loss_batch_fn=lambda values: _vmap_scalar_loss(test_loss_fn, values),
        flow=flow,
        theta0_batch=theta0_batch,
        optimizer_name=optimizer_name,
        lrs=lrs,
        steps=steps,
    )


def run_flow_curves_batched_loss(
    *,
    train_loss_batch_fn: BatchLossFn,
    test_loss_batch_fn: BatchLossFn,
    flow: torch.nn.Module,
    theta0_batch: torch.Tensor,
    optimizer_name: str,
    lrs: torch.Tensor,
    steps: int,
) -> list[Curve]:
    optimizer_value = _optimizer_name(optimizer_name)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        u = flow(theta0_batch.detach())[0]
    u = u.detach().clone().requires_grad_(True)
    lr_values = lrs.detach().to(device=u.device, dtype=u.dtype).reshape(-1, 1)
    batch = int(u.shape[0])
    train_losses_device = torch.empty(int(steps) + 1, batch, device=u.device, dtype=torch.float64)
    test_losses_device = torch.empty(int(steps) + 1, batch, device=u.device, dtype=torch.float64)
    opt_state: dict[str, torch.Tensor] = {}
    for step in range(0, int(steps) + 1):
        theta = flow.inverse(u)[0]
        losses = _as_batch_losses(train_loss_batch_fn(theta), batch)
        train_losses_device[step] = losses.detach().to(dtype=torch.float64)
        with torch.no_grad():
            test_losses_device[step] = _as_batch_losses(test_loss_batch_fn(theta), batch).detach().to(dtype=torch.float64)
        if step == int(steps):
            break
        grad = torch.autograd.grad(losses.sum(), u)[0]
        u = _manual_optimizer_step(
            u,
            grad,
            optimizer_name=optimizer_value,
            lr=1.0,
            step=step + 1,
            state=opt_state,
        )
        if optimizer_value == "sgd":
            u = (u + grad - lr_values * grad).detach().requires_grad_(True)
        else:
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            m_hat = opt_state["m"] / (1.0 - beta1 ** int(step + 1))
            v_hat = opt_state["v"] / (1.0 - beta2 ** int(step + 1))
            update = m_hat / (v_hat.sqrt() + eps)
            u = (u + update - lr_values * update).detach().requires_grad_(True)
    with torch.no_grad():
        final_theta = flow.inverse(u)[0].detach().cpu().numpy().astype(np.float64)
    train_losses = train_losses_device.detach().cpu().numpy()
    test_losses = test_losses_device.detach().cpu().numpy()
    return [
        Curve(train_loss=train_losses[:, idx], test_loss=test_losses[:, idx], final_theta=final_theta[idx])
        for idx in range(batch)
    ]


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
