from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .dataset import TileIdentity


@dataclass(slots=True)
class PathSample:
    z_t: torch.Tensor
    target_velocity: torch.Tensor
    t: torch.Tensor
    z0: torch.Tensor
    z1: torch.Tensor


def _expand_time(t: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if t.ndim != 1 or t.shape[0] != value.shape[0]:
        raise ValueError(f"t must be [batch], got {tuple(t.shape)} for {tuple(value.shape)}")
    while t.ndim < value.ndim:
        t = t.unsqueeze(-1)
    return t.to(dtype=value.dtype, device=value.device)


class GaussianToTargetPath:
    """Straight CFM path from N(0,I) to task-fitted codes."""

    @staticmethod
    def sample(z1: torch.Tensor, t: torch.Tensor, *, generator: torch.Generator | None = None) -> PathSample:
        z0 = torch.randn(z1.shape, dtype=z1.dtype, device=z1.device, generator=generator)
        tau = _expand_time(t, z1)
        return PathSample(z_t=(1.0 - tau) * z0 + tau * z1, target_velocity=z1 - z0, t=t, z0=z0, z1=z1)


class PairedAnchorPath:
    """Exact clean z_enc→z_task path. It deliberately exposes no noise option."""

    @staticmethod
    def sample(
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
        *,
        anchor_identities: Sequence[TileIdentity] | None = None,
        target_identities: Sequence[TileIdentity] | None = None,
    ) -> PathSample:
        if z0.shape != z1.shape:
            raise ValueError(f"anchor/target shape mismatch: {z0.shape} vs {z1.shape}")
        if anchor_identities is not None or target_identities is not None:
            if anchor_identities is None or target_identities is None:
                raise ValueError("both anchor and target identities are required")
            if len(anchor_identities) != len(target_identities) or any(
                a.pairing_key != b.pairing_key for a, b in zip(anchor_identities, target_identities, strict=True)
            ):
                raise ValueError("fatal z_enc/z_task pairing mismatch")
        tau = _expand_time(t, z1)
        return PathSample(z_t=(1.0 - tau) * z0 + tau * z1, target_velocity=z1 - z0, t=t, z0=z0, z1=z1)
