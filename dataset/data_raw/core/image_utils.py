from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

from PIL import Image


def seed_everything(seed: int) -> None:
    import torch

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
            return _open_pil_path(str(value["path"]))

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
    return _open_pil_path(str(path))


def _open_pil_path(path: str) -> Image.Image:
    # Fast path: regular local filesystem path.
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as local_exc:
        # Some HF datasets expose image references as virtual paths
        # like `zip://...::http://...`; these must be opened via fsspec.
        if "://" not in path and "::" not in path:
            raise local_exc

    try:
        import fsspec
    except Exception as exc:
        raise RuntimeError(f"Cannot open non-local image path without fsspec: {path}") from exc

    with fsspec.open(path, mode="rb").open() as handle:
        with Image.open(handle) as image:
            return image.convert("RGB")


def build_tensor_transform(image_size: int) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
