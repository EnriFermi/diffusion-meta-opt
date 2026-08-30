from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class IntegrationResult:
    endpoint: torch.Tensor
    trajectory: dict[float, torch.Tensor]
    nfe: int


@torch.no_grad()
def integrate_flow(
    velocity_fn: VelocityFn,
    z0: torch.Tensor,
    *,
    method: str = "heun",
    steps: int = 16,
    return_trajectory: bool = False,
) -> IntegrationResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")
    x = z0.clone()
    times = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=torch.float32)
    requested = {0.0, 0.25, 0.5, 0.75, 1.0}
    trajectory: dict[float, torch.Tensor] = {0.0: x.detach().cpu()} if return_trajectory else {}
    nfe = 0
    for index in range(steps):
        t0, t1 = times[index], times[index + 1]
        dt = (t1 - t0).to(dtype=x.dtype)
        t_batch = t0.expand(x.shape[0])
        v0 = velocity_fn(x, t_batch)
        nfe += 1
        if method == "euler":
            x = x + dt * v0
        else:
            proposal = x + dt * v0
            v1 = velocity_fn(proposal, t1.expand(x.shape[0]))
            nfe += 1
            x = x + 0.5 * dt * (v0 + v1)
        if return_trajectory:
            fraction = float(index + 1) / float(steps)
            for target in requested:
                if target not in trajectory and abs(fraction - target) <= 0.5 / steps + 1e-12:
                    trajectory[target] = x.detach().cpu()
    if return_trajectory:
        trajectory[1.0] = x.detach().cpu()
    return IntegrationResult(endpoint=x, trajectory=trajectory, nfe=nfe)
