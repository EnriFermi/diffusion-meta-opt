from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

import torch


TensorFn = Callable[[torch.Tensor], torch.Tensor]


def _jacrev_vmap(func: TensorFn, inputs: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    try:
        from torch.func import jacrev, vmap

        return vmap(jacrev(func))(inputs)
    except Exception:
        rows: list[torch.Tensor] = []
        for x in inputs:
            x_req = x.detach().clone().requires_grad_(True)
            jac = torch.autograd.functional.jacobian(func, x_req, create_graph=bool(create_graph), strict=False)
            rows.append(jac)
        return torch.stack(rows, dim=0)


def exact_jacobians(func: TensorFn, inputs: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be [B,D], got {tuple(inputs.shape)}")
    return _jacrev_vmap(func, inputs, create_graph=bool(create_graph))


@dataclass(frozen=True, slots=True)
class ProbeScales:
    residual_rms: float
    theta_rms: float


@dataclass(frozen=True, slots=True)
class ResidualThetaProbe:
    residual_fn: TensorFn
    scales: ProbeScales
    rho: float
    eps: float = 1e-12

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        residual = self.residual_fn(theta) / max(float(self.scales.residual_rms), float(self.eps))
        damped = float(self.rho) * theta / max(float(self.scales.theta_rms), float(self.eps))
        return torch.cat([residual.reshape(-1), damped.reshape(-1)], dim=0)


@dataclass(frozen=True, slots=True)
class ResidualOnlyProbe:
    residual_fn: TensorFn

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return self.residual_fn(theta).reshape(-1)


def fit_probe_scales(residual_fn: TensorFn, theta_samples: torch.Tensor, *, eps: float = 1e-12) -> ProbeScales:
    residual_rows = []
    with torch.no_grad():
        for theta in theta_samples:
            residual_rows.append(residual_fn(theta).detach().reshape(-1))
        residuals = torch.stack(residual_rows, dim=0)
        residual_rms = residuals.pow(2).mean().sqrt().clamp_min(float(eps))
        theta_rms = theta_samples.detach().pow(2).mean().sqrt().clamp_min(float(eps))
    return ProbeScales(residual_rms=float(residual_rms.cpu().item()), theta_rms=float(theta_rms.cpu().item()))


def metric_tensors_from_jacobians(jacobians: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if jacobians.ndim != 3:
        raise ValueError(f"jacobians must be [B,O,D], got {tuple(jacobians.shape)}")
    metric = torch.matmul(jacobians.transpose(-1, -2), jacobians)
    trace_g = jacobians.pow(2).sum(dim=(-1, -2))
    trace_g2 = metric.square().sum(dim=(-1, -2))
    return metric, trace_g, trace_g2


def isometry_objective_from_jacobians(jacobians: torch.Tensor, *, dim: int, eps: float = 1e-12) -> torch.Tensor:
    _metric, trace_g, trace_g2 = metric_tensors_from_jacobians(jacobians)
    return float(dim * dim) * trace_g2.mean() / trace_g.mean().square().clamp_min(float(eps)) - float(dim)


def probe_jacobians_for_theta(probe: TensorFn, theta_samples: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
    return exact_jacobians(lambda theta: probe(theta), theta_samples, create_graph=bool(create_graph))


def probe_jacobians_for_flow(
    probe: TensorFn,
    flow: torch.nn.Module,
    theta_samples: torch.Tensor,
    *,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_samples = flow(theta_samples)[0]

    def composed(u: torch.Tensor) -> torch.Tensor:
        theta = flow.inverse(u.unsqueeze(0))[0].squeeze(0)
        return probe(theta)

    return exact_jacobians(composed, u_samples, create_graph=bool(create_graph)), u_samples


def geometry_rows_from_jacobians(
    jacobians: torch.Tensor,
    *,
    experiment: str,
    coordinate: str,
    seed: int,
    rho: float,
    eps: float = 1e-12,
) -> list[dict[str, object]]:
    metric, trace_g, trace_g2 = metric_tensors_from_jacobians(jacobians.detach())
    eigenvalues = torch.linalg.eigvalsh(metric.float()).detach().cpu()
    dim = int(jacobians.shape[-1])
    objective = float(isometry_objective_from_jacobians(jacobians.detach(), dim=dim, eps=eps).cpu().item())
    rows: list[dict[str, object]] = []
    for idx in range(int(jacobians.shape[0])):
        eig = eigenvalues[idx]
        eig_min = float(eig.min().item())
        eig_max = float(eig.max().item())
        if eig_min <= float(eps) or not math.isfinite(eig_min):
            cond = float("inf")
            log_cond = float("inf")
        else:
            cond = eig_max / eig_min
            log_cond = float(math.log(cond)) if cond > 0.0 and math.isfinite(cond) else float("inf")
        rows.append(
            {
                "experiment": experiment,
                "coordinate": coordinate,
                "seed": int(seed),
                "rho": float(rho),
                "sample_index": int(idx),
                "isometry_objective": objective,
                "trace_g": float(trace_g[idx].detach().cpu().item()),
                "trace_g2": float(trace_g2[idx].detach().cpu().item()),
                "condition_number": float(cond),
                "log_condition_number": float(log_cond),
                "eig_min": float(eig_min),
                "eig_max": float(eig_max),
                "eigenvalues_json": json.dumps([float(v) for v in eig.tolist()]),
            }
        )
    return rows


def trace_cv(trace_values: torch.Tensor, *, eps: float = 1e-12) -> float:
    values = trace_values.detach().float()
    if int(values.numel()) <= 1:
        return 0.0
    return float((values.std(unbiased=False) / values.mean().abs().clamp_min(float(eps))).cpu().item())
