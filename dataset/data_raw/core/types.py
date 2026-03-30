from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(slots=True)
class ImageSample:
    """Single image sample returned by a virtualized dataset."""

    image: Image.Image
    dataset_name: str
    sample_id: str | int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImageSampleRef:
    """Lightweight reference to a disk-backed image sample."""

    dataset_name: str
    sample_id: str | int
    meta: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None
    file_name: str | None = None
