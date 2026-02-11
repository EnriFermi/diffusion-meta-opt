from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def decode_to_pil(value: Any) -> Image.Image:
    """Decode a datasets image payload into RGB PIL.Image."""

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")

    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")

    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return Image.fromarray(value).convert("RGB")
    except Exception:
        pass

    if hasattr(value, "convert"):
        try:
            return value.convert("RGB")
        except Exception:
            pass

    raise TypeError(f"Cannot decode image payload of type {type(value)}")


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_tensor_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
