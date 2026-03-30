from __future__ import annotations

import io
import logging
import time
from collections import OrderedDict
import contextlib
from pathlib import Path

import requests
from PIL import Image

from dataset.data_raw.core.chunk_cache import ChunkCache

LOGGER = logging.getLogger(__name__)
FAILED_URL_COOLDOWN_S = 600.0
FAILED_URL_MAX_ENTRIES = 20_000
REQUEST_HEADERS = {
    "User-Agent": "diff-meta-opt-dataset-fetcher/1.0 (+https://huggingface.co/)",
    "Accept": "image/*,*/*;q=0.8",
}
_FAILED_URLS: OrderedDict[str, float] = OrderedDict()


def _should_skip_url(url: str) -> bool:
    expiry = _FAILED_URLS.get(url)
    if expiry is None:
        return False

    if time.monotonic() >= expiry:
        with contextlib.suppress(KeyError):
            del _FAILED_URLS[url]
        return False
    return True


def _remember_failed_url(url: str) -> None:
    _FAILED_URLS[url] = time.monotonic() + FAILED_URL_COOLDOWN_S
    _FAILED_URLS.move_to_end(url)
    while len(_FAILED_URLS) > FAILED_URL_MAX_ENTRIES:
        _FAILED_URLS.popitem(last=False)


def fetch_image_to_cache(
    url: str,
    cache: ChunkCache,
    sample_key: str | int,
    timeout: int = 10,
    retries: int = 2,
    chunk_id: str | None = None,
) -> Path | None:
    """Download URL image, decode as PIL, save into chunk cache."""
    if _should_skip_url(url):
        return None

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return cache.save_image(sample_id=sample_key, image=image, chunk_id=chunk_id)
        except Exception as exc:
            if attempt >= retries:
                _remember_failed_url(url)
                LOGGER.warning("URL fetch failed for %s after %s attempts: %s", url, retries + 1, exc)
                return None
            time.sleep(0.25 * (attempt + 1))

    return None


def fetch_image_to_memory(
    url: str,
    timeout: int = 10,
    retries: int = 2,
) -> Image.Image | None:
    """Download URL image and return decoded RGB PIL.Image."""
    if _should_skip_url(url):
        return None

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            if attempt >= retries:
                _remember_failed_url(url)
                LOGGER.warning("URL fetch failed for %s after %s attempts: %s", url, retries + 1, exc)
                return None
            time.sleep(0.25 * (attempt + 1))

    return None
