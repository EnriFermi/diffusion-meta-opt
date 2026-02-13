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
