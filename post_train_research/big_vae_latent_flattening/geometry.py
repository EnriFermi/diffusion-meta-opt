from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True, slots=True)
class RelaxedDistortionStats:
    loss: torch.Tensor
    ratio: torch.Tensor
    zero_baseline: torch.Tensor
    trace_g: torch.Tensor
    trace_g2: torch.Tensor


def mixup_latents(z: torch.Tensor, *, eta: float | None) -> torch.Tensor:
    if eta is None:
        return z
    batch_size = int(z.shape[0])
    if batch_size <= 1:
        return z
    perm = torch.randperm(batch_size, device=z.device)
    alpha = torch.empty(batch_size, 1, device=z.device, dtype=z.dtype).uniform_(-float(eta), 1.0 + float(eta))
    return alpha * z + (1.0 - alpha) * z[perm]


def relaxed_distortion_measure(
    func: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    *,
    eta: float | None = 0.2,
    probes: int = 1,
    loss_scale: str = "readme",
    eps: float = 1e-12,
    create_graph: bool = True,
) -> RelaxedDistortionStats:
    """Stochastic relaxed distortion from the IRVAE reference implementation.

    `func` maps flattened latent coordinates to decoder outputs. The estimator
    uses one or more Hutchinson probes:

        Tr(G)   ~= ||Jv||^2
        Tr(G^2) ~= ||J^T J v||^2

    where G = J^T J is the decoder pullback metric.
    """

    if z.ndim != 2:
        raise ValueError(f"z must be rank-2 [B,D], got {tuple(z.shape)}")
    if int(probes) <= 0:
        raise ValueError(f"probes must be positive, got {probes}")

    batch_size = int(z.shape[0])
    latent_dim = int(z.shape[1])
    trace_g_terms: list[torch.Tensor] = []
    trace_g2_terms: list[torch.Tensor] = []
    for _ in range(int(probes)):
        z_aug = mixup_latents(z, eta=eta)
        v = torch.randn_like(z_aug)
        jvp = torch.autograd.functional.jvp(func, z_aug, v=v, create_graph=create_graph, strict=False)[1]
        trace_g_terms.append(jvp.reshape(batch_size, -1).pow(2).sum(dim=1).mean())
        jtj_v = torch.autograd.functional.vjp(func, z_aug, v=jvp, create_graph=create_graph, strict=False)[1]
        trace_g2_terms.append(jtj_v.reshape(batch_size, -1).pow(2).sum(dim=1).mean())

    trace_g = torch.stack(trace_g_terms).mean()
    trace_g2 = torch.stack(trace_g2_terms).mean()
    ratio = trace_g2 / trace_g.square().clamp_min(float(eps))
    zero_baseline = float(latent_dim) * ratio - 1.0

    scale = str(loss_scale).strip().lower()
    if scale in {"readme", "raw", "ratio"}:
        loss = ratio
    elif scale in {"zero_baseline", "zero", "nonnegative"}:
        loss = zero_baseline
    else:
        raise ValueError("loss_scale must be one of {'readme', 'zero_baseline'}")

    return RelaxedDistortionStats(
        loss=loss,
        ratio=ratio.detach(),
        zero_baseline=zero_baseline.detach(),
        trace_g=trace_g.detach(),
        trace_g2=trace_g2.detach(),
    )

