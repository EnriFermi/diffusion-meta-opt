from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from .e4_debug import (
    E4DebugState,
    evaluate_debug_snapshot,
    evaluate_flow_geometry,
    evaluate_original_geometry,
)
from .flow_experiment import assert_flow_architecture_sanity, make_flow
from .optimization import Curve, LossFn
from .probe_geometry import TensorFn, metric_tensors_from_jacobians, probe_jacobians_for_theta
from .toy_mlp import MLP_DIM


@dataclass(slots=True)
class DirectionTargetCache:
    theta: torch.Tensor
    grad: torch.Tensor
    probe_metric: torch.Tensor
    metric_preconditioner: torch.Tensor
    p_metric: torch.Tensor
    damping: torch.Tensor
    split: str
    rho: float
    metric_alpha: float


@dataclass(slots=True)
class DirectionMatchResult:
    flow: torch.nn.Module
    history: pd.DataFrame
    final_geometry: pd.DataFrame
    target_summary: pd.DataFrame
    output_dir: Path
    metric_alpha: float
    scale_beta: float


def _float_token(value: float) -> str:
    return f"{float(value):.0e}".replace("-", "m").replace("+", "").replace(".", "p")


def direction_match_variant_slug(*, rho: float, metric_alpha: float, scale_beta: float) -> str:
    return f"rho{_float_token(rho)}_alpha{_float_token(metric_alpha)}_beta{_float_token(scale_beta)}"


def direction_target_cache_path(output_dir: Path, *, split: str, rho: float, metric_alpha: float) -> Path:
    return output_dir / f"direction_targets_{split}_rho{_float_token(rho)}_alpha{_float_token(metric_alpha)}.pt"


def _train_gradients(train_loss_fn: LossFn, theta_samples: torch.Tensor) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for theta in theta_samples:
        theta_req = theta.detach().clone().requires_grad_(True)
        loss = train_loss_fn(theta_req)
        rows.append(torch.autograd.grad(loss, theta_req)[0].detach())
    return torch.stack(rows, dim=0)


def compute_direction_targets(
    *,
    train_loss_fn: LossFn,
    probe: TensorFn,
    theta_samples: torch.Tensor,
    metric_alpha: float,
    rho: float,
    split: str,
    eps: float = 1e-12,
) -> DirectionTargetCache:
    if theta_samples.ndim != 2 or int(theta_samples.shape[1]) != MLP_DIM:
        raise ValueError(f"theta_samples must be [N,{MLP_DIM}], got {tuple(theta_samples.shape)}")
    theta = theta_samples.detach()
    grad = _train_gradients(train_loss_fn, theta)
    jacobians = probe_jacobians_for_theta(probe, theta, create_graph=False).detach()
    probe_metric, trace_g, _trace_g2 = metric_tensors_from_jacobians(jacobians)
    dim = int(theta.shape[1])
    damping = float(metric_alpha) * trace_g.detach() / float(dim) + float(eps)
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype).expand(int(theta.shape[0]), dim, dim)
    metric_preconditioner = torch.linalg.solve(probe_metric + damping.reshape(-1, 1, 1) * eye, eye)
    p_metric = torch.bmm(metric_preconditioner, grad.reshape(-1, dim, 1)).reshape(-1, dim)
    return DirectionTargetCache(
        theta=theta.detach(),
        grad=grad.detach(),
        probe_metric=probe_metric.detach(),
        metric_preconditioner=metric_preconditioner.detach(),
        p_metric=p_metric.detach(),
        damping=damping.detach(),
        split=str(split),
        rho=float(rho),
        metric_alpha=float(metric_alpha),
    )


def save_direction_targets(path: Path, targets: DirectionTargetCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "theta": targets.theta.detach().cpu(),
            "grad": targets.grad.detach().cpu(),
            "probe_metric": targets.probe_metric.detach().cpu(),
            "metric_preconditioner": targets.metric_preconditioner.detach().cpu(),
            "p_metric": targets.p_metric.detach().cpu(),
            "damping": targets.damping.detach().cpu(),
            "split": targets.split,
            "rho": float(targets.rho),
            "metric_alpha": float(targets.metric_alpha),
        },
        path,
    )


def load_direction_targets(path: Path, *, device: torch.device, dtype: torch.dtype) -> DirectionTargetCache:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return DirectionTargetCache(
        theta=payload["theta"].to(device=device, dtype=dtype),
        grad=payload["grad"].to(device=device, dtype=dtype),
        probe_metric=payload["probe_metric"].to(device=device, dtype=dtype),
        metric_preconditioner=payload["metric_preconditioner"].to(device=device, dtype=dtype),
        p_metric=payload["p_metric"].to(device=device, dtype=dtype),
        damping=payload["damping"].to(device=device, dtype=dtype),
        split=str(payload["split"]),
        rho=float(payload["rho"]),
        metric_alpha=float(payload["metric_alpha"]),
    )


def load_or_compute_direction_targets(
    *,
    state: E4DebugState,
    theta_samples: torch.Tensor,
    metric_alpha: float,
    split: str,
    force_recompute: bool = False,
) -> DirectionTargetCache:
    path = direction_target_cache_path(
        state.output_dir,
        split=str(split),
        rho=float(state.debug_cfg.rho),
        metric_alpha=float(metric_alpha),
    )
    if path.is_file() and not bool(force_recompute):
        return load_direction_targets(path, device=theta_samples.device, dtype=theta_samples.dtype)
    targets = compute_direction_targets(
        train_loss_fn=state.problem.train_loss,
        probe=state.probe,
        theta_samples=theta_samples,
        metric_alpha=float(metric_alpha),
        rho=float(state.debug_cfg.rho),
        split=str(split),
        eps=float(state.cfg.aulc_eps),
    )
    save_direction_targets(path, targets)
    return targets


def direction_target_summary(targets: DirectionTargetCache) -> dict[str, float | str]:
    eig = torch.linalg.eigvalsh(targets.probe_metric.float()).detach()
    eig_min = eig.min(dim=-1).values
    eig_max = eig.max(dim=-1).values
    cond = eig_max / eig_min.clamp_min(1e-12)
    return {
        "split": targets.split,
        "rho": float(targets.rho),
        "metric_alpha": float(targets.metric_alpha),
        "samples": float(targets.theta.shape[0]),
        "damping_median": float(targets.damping.float().median().cpu().item()),
        "damping_min": float(targets.damping.float().min().cpu().item()),
        "damping_max": float(targets.damping.float().max().cpu().item()),
        "grad_norm_median": float(targets.grad.float().norm(dim=-1).median().cpu().item()),
        "p_metric_norm_median": float(targets.p_metric.float().norm(dim=-1).median().cpu().item()),
        "probe_metric_cond_median": float(cond.median().cpu().item()),
    }


def inverse_jacobians_for_flow(flow: torch.nn.Module, u_samples: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    if u_samples.ndim != 2:
        raise ValueError(f"u_samples must be [B,D], got {tuple(u_samples.shape)}")
    rows: list[torch.Tensor] = []
    for row_idx in range(int(u_samples.shape[0])):
        u = u_samples[row_idx]
        if not u.requires_grad:
            u = u.detach().clone().requires_grad_(True)
        theta = flow.inverse(u.unsqueeze(0))[0].squeeze(0)
        jac_rows: list[torch.Tensor] = []
        for out_idx in range(int(theta.numel())):
            grad_u = torch.autograd.grad(
                theta[out_idx],
                u,
                retain_graph=True,
                create_graph=bool(create_graph),
                allow_unused=False,
            )[0]
            jac_rows.append(grad_u.reshape(-1))
        rows.append(torch.stack(jac_rows, dim=0))
    return torch.stack(rows, dim=0)


def nf_preconditioner_direction(
    flow: torch.nn.Module,
    theta_samples: torch.Tensor,
    grad_samples: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_samples = flow(theta_samples)[0]
    inverse_jacobians = inverse_jacobians_for_flow(flow, u_samples, create_graph=bool(create_graph))
    j_t_grad = torch.einsum("bod,bo->bd", inverse_jacobians, grad_samples)
    p_nf = torch.einsum("bod,bd->bo", inverse_jacobians, j_t_grad)
    return p_nf, inverse_jacobians, u_samples


def direction_match_loss(
    flow: torch.nn.Module,
    targets: DirectionTargetCache,
    *,
    scale_beta: float,
    create_graph: bool,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    p_nf, _inverse_jacobians, _u_samples = nf_preconditioner_direction(
        flow,
        targets.theta,
        targets.grad,
        create_graph=bool(create_graph),
    )
    cosine = F.cosine_similarity(p_nf, targets.p_metric, dim=-1, eps=float(eps))
    dir_loss = (1.0 - cosine).mean()
    p_nf_norm = p_nf.float().norm(dim=-1)
    p_metric_norm = targets.p_metric.float().norm(dim=-1)
    scale_loss = torch.log((p_nf_norm + float(eps)) / (p_metric_norm + float(eps))).square().mean()
    loss = dir_loss + float(scale_beta) * scale_loss.to(dtype=dir_loss.dtype)
    norm_ratio = p_metric_norm / p_nf_norm.clamp_min(float(eps))
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "dir_loss": float(dir_loss.detach().cpu().item()),
        "scale_loss": float(scale_loss.detach().cpu().item()),
        "cos_median": float(cosine.detach().float().median().cpu().item()),
        "cos_mean": float(cosine.detach().float().mean().cpu().item()),
        "norm_ratio_metric_to_nf_median": float(norm_ratio.detach().float().median().cpu().item()),
        "p_nf_norm_median": float(p_nf_norm.detach().median().cpu().item()),
        "p_metric_norm_median": float(p_metric_norm.detach().median().cpu().item()),
    }
    return loss, metrics


def subset_direction_targets(targets: DirectionTargetCache, indices: torch.Tensor, *, split: str | None = None) -> DirectionTargetCache:
    return DirectionTargetCache(
        theta=targets.theta.index_select(0, indices),
        grad=targets.grad.index_select(0, indices),
        probe_metric=targets.probe_metric.index_select(0, indices),
        metric_preconditioner=targets.metric_preconditioner.index_select(0, indices),
        p_metric=targets.p_metric.index_select(0, indices),
        damping=targets.damping.index_select(0, indices),
        split=targets.split if split is None else str(split),
        rho=targets.rho,
        metric_alpha=targets.metric_alpha,
    )


def evaluate_direction_match(
    flow: torch.nn.Module,
    targets: DirectionTargetCache,
    *,
    scale_beta: float,
    max_samples: int = 32,
    prefix: str,
) -> dict[str, float]:
    sample_count = min(max(1, int(max_samples)), int(targets.theta.shape[0]))
    indices = torch.arange(sample_count, device=targets.theta.device)
    subset = subset_direction_targets(targets, indices, split=targets.split)
    flow_was_training = flow.training
    flow.eval()
    try:
        _loss, metrics = direction_match_loss(
            flow,
            subset,
            scale_beta=float(scale_beta),
            create_graph=False,
        )
    finally:
        flow.train(flow_was_training)
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _snapshot_direction_match(
    *,
    state: E4DebugState,
    flow: torch.nn.Module,
    train_targets: DirectionTargetCache,
    heldout_targets: DirectionTargetCache,
    scale_beta: float,
    step: int,
    batch_metrics: dict[str, float],
    elapsed_s: float,
) -> dict[str, float]:
    row = evaluate_debug_snapshot(
        state,
        flow,
        step=int(step),
        batch_loss=float(batch_metrics.get("loss", float("nan"))),
        elapsed_s=float(elapsed_s),
    )
    row["metric_alpha"] = float(train_targets.metric_alpha)
    row["scale_beta"] = float(scale_beta)
    row["batch_dir_loss"] = float(batch_metrics.get("dir_loss", float("nan")))
    row["batch_scale_loss"] = float(batch_metrics.get("scale_loss", float("nan")))
    row["batch_cos_median"] = float(batch_metrics.get("cos_median", float("nan")))
    row["batch_norm_ratio_metric_to_nf_median"] = float(batch_metrics.get("norm_ratio_metric_to_nf_median", float("nan")))
    row.update(
        evaluate_direction_match(
            flow,
            train_targets,
            scale_beta=float(scale_beta),
            max_samples=int(state.debug_cfg.train_eval_samples),
            prefix="direction_train",
        )
    )
    row.update(
        evaluate_direction_match(
            flow,
            heldout_targets,
            scale_beta=float(scale_beta),
            max_samples=int(state.debug_cfg.heldout_eval_samples),
            prefix="direction_heldout",
        )
    )
    return row


def train_direction_match_flow(
    *,
    state: E4DebugState,
    train_targets: DirectionTargetCache,
    heldout_targets: DirectionTargetCache,
    scale_beta: float,
    seed_offset: int = 70_000,
) -> DirectionMatchResult:
    if float(train_targets.metric_alpha) != float(heldout_targets.metric_alpha):
        raise ValueError("train and heldout targets must use the same metric_alpha")
    debug_cfg = state.debug_cfg
    variant_slug = direction_match_variant_slug(
        rho=float(debug_cfg.rho),
        metric_alpha=float(train_targets.metric_alpha),
        scale_beta=float(scale_beta),
    )
    output_dir = state.output_dir / "direction_match" / variant_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variant_config.json").write_text(
        json.dumps(
            {
                "debug_config": asdict(debug_cfg),
                "variant": {
                    "rho": float(debug_cfg.rho),
                    "metric_alpha": float(train_targets.metric_alpha),
                    "scale_beta": float(scale_beta),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    torch.manual_seed(int(debug_cfg.seed) + int(seed_offset) + int(round(abs(math.log10(max(float(train_targets.metric_alpha), 1e-12))) * 1000)))
    flow = make_flow(MLP_DIM, state.cfg).to(device=train_targets.theta.device, dtype=train_targets.theta.dtype)
    assert_flow_architecture_sanity(flow, train_targets.theta, state.cfg, log_label=f"[E4 direction {variant_slug}]")
    optimizer = torch.optim.Adam(flow.parameters(), lr=float(debug_cfg.flow_lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(debug_cfg.seed) + int(seed_offset) + 17)
    sample_count = int(train_targets.theta.shape[0])
    batch_size = min(max(1, int(debug_cfg.flow_batch_size)), sample_count)
    eval_every = max(1, int(debug_cfg.eval_every))
    rows: list[dict[str, float]] = []
    started = time.time()

    rows.append(
        _snapshot_direction_match(
            state=state,
            flow=flow,
            train_targets=train_targets,
            heldout_targets=heldout_targets,
            scale_beta=float(scale_beta),
            step=0,
            batch_metrics={},
            elapsed_s=0.0,
        )
    )
    flow.train()
    for step in range(1, int(debug_cfg.flow_steps) + 1):
        indices = torch.randint(0, sample_count, (batch_size,), generator=generator, device="cpu").to(device=train_targets.theta.device)
        batch_targets = subset_direction_targets(train_targets, indices)
        optimizer.zero_grad(set_to_none=True)
        loss, batch_metrics = direction_match_loss(
            flow,
            batch_targets,
            scale_beta=float(scale_beta),
            create_graph=True,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite E4 direction-match loss at step {step}: {float(loss.detach().cpu().item())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), float(debug_cfg.flow_grad_clip_norm))
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == int(debug_cfg.flow_steps):
            batch_metrics["grad_norm"] = float(torch.as_tensor(grad_norm).detach().cpu().item())
            row = _snapshot_direction_match(
                state=state,
                flow=flow,
                train_targets=train_targets,
                heldout_targets=heldout_targets,
                scale_beta=float(scale_beta),
                step=step,
                batch_metrics=batch_metrics,
                elapsed_s=time.time() - started,
            )
            row["grad_norm"] = batch_metrics["grad_norm"]
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / "history.csv", index=False)

    history = pd.DataFrame(rows)
    final_rows = [
        {"split": "train_eval", "coordinate": "original", **evaluate_original_geometry(state.probe, state.train_eval_pool, prefix="metric")},
        {"split": "train_eval", "coordinate": "flow", **evaluate_flow_geometry(state.probe, flow, state.train_eval_pool, prefix="metric")},
        {"split": "heldout_eval", "coordinate": "original", **evaluate_original_geometry(state.probe, state.heldout_eval_pool, prefix="metric")},
        {"split": "heldout_eval", "coordinate": "flow", **evaluate_flow_geometry(state.probe, flow, state.heldout_eval_pool, prefix="metric")},
    ]
    final_geometry = pd.DataFrame(final_rows)
    target_summary = pd.DataFrame([direction_target_summary(train_targets), direction_target_summary(heldout_targets)])
    history.to_csv(output_dir / "history.csv", index=False)
    final_geometry.to_csv(output_dir / "final_geometry.csv", index=False)
    target_summary.to_csv(output_dir / "target_summary.csv", index=False)
    torch.save(flow.state_dict(), output_dir / "flow_state.pt")
    return DirectionMatchResult(
        flow=flow,
        history=history,
        final_geometry=final_geometry,
        target_summary=target_summary,
        output_dir=output_dir,
        metric_alpha=float(train_targets.metric_alpha),
        scale_beta=float(scale_beta),
    )


def run_scaled_probe_metric_curve(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    probe: TensorFn,
    theta0: torch.Tensor,
    lr: float,
    steps: int,
    metric_alpha: float,
    eps: float = 1e-12,
    store_path: bool = False,
) -> Curve:
    theta = theta0.detach().clone().requires_grad_(True)
    dim = int(theta.numel())
    train_losses_device = torch.empty(int(steps) + 1, device=theta.device, dtype=torch.float64)
    test_losses_device = torch.empty(int(steps) + 1, device=theta.device, dtype=torch.float64)
    path = torch.empty(int(steps) + 1, dim, device=theta.device, dtype=torch.float64) if store_path else None
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)

    for step in range(0, int(steps) + 1):
        loss = train_loss_fn(theta)
        train_losses_device[step] = loss.detach().to(dtype=torch.float64)
        with torch.no_grad():
            test_losses_device[step] = test_loss_fn(theta).detach().to(dtype=torch.float64)
            if path is not None:
                path[step] = theta.detach().to(dtype=torch.float64)
        if step == int(steps):
            break

        grad = torch.autograd.grad(loss, theta)[0].detach()
        jacobian = probe_jacobians_for_theta(probe, theta.detach().reshape(1, -1), create_graph=False).squeeze(0)
        metric = jacobian.transpose(0, 1) @ jacobian
        damping = float(metric_alpha) * torch.trace(metric) / float(dim) + float(eps)
        direction = torch.linalg.solve(metric + damping * eye, grad.reshape(-1, 1)).reshape(-1)
        theta = (theta.detach() - float(lr) * direction.detach()).requires_grad_(True)

    return Curve(
        train_loss=train_losses_device.detach().cpu().numpy(),
        test_loss=test_losses_device.detach().cpu().numpy(),
        final_theta=theta.detach().cpu().numpy(),
        path=path.detach().cpu().numpy() if path is not None else None,
    )
