from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import requests
from PIL import Image

from dataset.data_raw.core.chunk_cache import ChunkCache

LOGGER = logging.getLogger(__name__)


def fetch_image_to_cache(
    url: str,
    cache: ChunkCache,
    sample_key: str | int,
    timeout: int = 10,
    retries: int = 2,
    chunk_id: str | None = None,
) -> Path | None:
    """Download URL image, decode as PIL, save into chunk cache."""

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return cache.save_image(sample_id=sample_key, image=image, chunk_id=chunk_id)
        except Exception as exc:
            if attempt >= retries:
                LOGGER.warning("URL fetch failed for %s after %s attempts: %s", url, retries + 1, exc)
                return None
            time.sleep(0.25 * (attempt + 1))

    return None
