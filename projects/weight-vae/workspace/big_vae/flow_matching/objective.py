from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from .paths import GaussianToTargetPath, PairedAnchorPath, PathSample


@dataclass(slots=True)
class FlowLoss:
    loss: torch.Tensor
    velocity_mse: torch.Tensor
    endpoint_distance: torch.Tensor
    sample: PathSample


def controlled_objective_mask(batch: Mapping[str, Any]) -> torch.Tensor | None:
    """Return the scientific objective mask, not merely the codec-valid mask.

    WeightCLIP retains all official window rows as transformer/decoder context,
    but the controlled benchmark transfers only conv/BN-affine body rows. Both
    Gaussian and paired paths therefore optimize exactly those body rows.
    """

    mask = batch.get("token_mask")
    codecs = batch.get("codec")
    if codecs is None:
        return mask
    if isinstance(codecs, str):
        codecs = [codecs]
    unique = {str(codec) for codec in codecs}
    if len(unique) != 1:
        raise ValueError(f"one flow batch cannot mix codecs: {sorted(unique)}")
    if unique == {"weightclip"}:
        body = batch["architecture_features"][..., 1].to(dtype=torch.bool)
        return body if mask is None else mask.to(dtype=torch.bool) & body
    if unique != {"ours"}:
        raise ValueError(f"unsupported flow batch codec {next(iter(unique))!r}")
    return mask


def conditional_flow_matching_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    path_kind: str,
    generator: torch.Generator | None = None,
) -> FlowLoss:
    z1 = batch["z_task"]
    batch_size = int(z1.shape[0])
    t = torch.rand(batch_size, device=z1.device, dtype=torch.float32, generator=generator)
    if path_kind == "gaussian":
        sample = GaussianToTargetPath.sample(z1, t, generator=generator)
    elif path_kind == "paired_anchor":
        if "z_enc" not in batch:
            raise ValueError("paired_anchor objective requires z_enc")
        target_identities = batch.get("identity")
        anchor_identities = batch.get("anchor_identity")
        sample = PairedAnchorPath.sample(
            batch["z_enc"],
            z1,
            t,
            anchor_identities=anchor_identities,
            target_identities=target_identities,
        )
    else:
        raise ValueError(f"unsupported path_kind {path_kind!r}")
    mask = controlled_objective_mask(batch)
    predicted = model(
        sample.z_t,
        sample.t,
        dataset_embedding=batch["dataset_embedding"],
        architecture_features=batch["architecture_features"],
        token_mask=mask,
    )
    if predicted.shape != sample.target_velocity.shape:
        raise ValueError(f"velocity shape mismatch: {predicted.shape} vs {sample.target_velocity.shape}")
    squared = (predicted.float() - sample.target_velocity.float()).square().mean(dim=-1)
    if mask is not None:
        mask = mask.to(device=squared.device, dtype=squared.dtype)
        velocity_mse = (squared * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        velocity_mse = squared.mean()
    endpoint_squared = (sample.z1.float() - sample.z0.float()).square().mean(dim=-1)
    if mask is not None:
        endpoint_mask = mask.to(device=endpoint_squared.device, dtype=endpoint_squared.dtype)
        endpoint_distance = ((endpoint_squared * endpoint_mask).sum() / endpoint_mask.sum().clamp_min(1.0)).sqrt()
    else:
        endpoint_distance = endpoint_squared.mean().sqrt()
    return FlowLoss(loss=velocity_mse, velocity_mse=velocity_mse.detach(), endpoint_distance=endpoint_distance.detach(), sample=sample)
