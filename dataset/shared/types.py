from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class MixedImageMeta:
    """Metadata for one source image in a mixed collector batch."""

    dataset_name: str
    source_id: str | int


@dataclass(slots=True)
class SharedSample:
    """Atomic cached training sample produced by collector inference."""

    model_name: str
    layer_name: str
    weight: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CollectorJobStats:
    """Summary of one collector inference job."""

    model_name: str
    num_images: int
    num_layers: int
    num_samples_emitted: int
    dataset_mix: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    raw_batch_fetch_s: float = 0.0
    model_infer_s: float = 0.0
    atomize_emit_s: float = 0.0
    memory_rss_before_mb: float | None = None
    memory_rss_after_mb: float | None = None
    memory_rss_delta_mb: float | None = None
    memory_hwm_before_mb: float | None = None
    memory_hwm_after_mb: float | None = None
    memory_hwm_delta_mb: float | None = None
    memory_children_rss_before_mb: float | None = None
    memory_children_rss_after_mb: float | None = None
    memory_children_rss_delta_mb: float | None = None
