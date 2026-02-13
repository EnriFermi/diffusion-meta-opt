from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class LayerIORecord:
    """Aggregated Linear-layer forward IO captured from a single model run."""

    model_name: str
    layer_name: str
    weight: torch.Tensor
    inputs: torch.Tensor
    outputs: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)
