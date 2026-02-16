from __future__ import annotations

import logging
from typing import Any

import datasets
from datasets import load_dataset

from dataset.data_raw.core.config import to_plain_dict

LOGGER = logging.getLogger(__name__)


def load_hf_dataset(cfg: Any, token: str | None, seed: int):
    """Load Hugging Face dataset with explicit token when possible."""

    plain = to_plain_dict(cfg)
    hf_cfg = plain.get("hf", {})
    if not isinstance(hf_cfg, dict):
        raise TypeError("Dataset config must have 'hf' mapping")

    repo = hf_cfg["repo"]
    subset = hf_cfg.get("subset")
    split = hf_cfg.get("split", "train")
    streaming = bool(hf_cfg.get("streaming", False))
    trust_remote_code = bool(hf_cfg.get("trust_remote_code", False))

    base_kwargs = {
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
    }

    try:
        dataset = _load_dataset_with_best_auth(
            repo=repo,
            subset=subset,
            kwargs={"split": split, **base_kwargs},
            token=token,
        )
    except ValueError as exc:
        # Some datasets expose only a subset of splits (e.g. ["test"]).
        # Fallback to loading split map and selecting the closest available split.
        text = str(exc)
        if "Bad split" not in text:
            raise
        LOGGER.warning("Dataset '%s' split '%s' is unavailable: %s", repo, split, text)
        dataset = _load_dataset_with_best_auth(
            repo=repo,
            subset=subset,
            kwargs=base_kwargs,
            token=token,
        )
        resolved_split = _select_split(dataset=dataset, preferred=str(split))
        LOGGER.warning("Dataset '%s': using fallback split '%s' instead of '%s'", repo, resolved_split, split)
        dataset = dataset[resolved_split]

    if streaming and hasattr(dataset, "shuffle"):
        shuffle_buffer = int(hf_cfg.get("shuffle_buffer", 10_000))
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        if bool(hf_cfg.get("stream_via_datasets_server", False)):
            dataset = _disable_streaming_image_decode(dataset=dataset, cfg=plain)
        dataset = _project_streaming_columns(dataset=dataset, cfg=plain)

    return dataset


def _load_dataset_with_best_auth(
    repo: str,
    subset: str | None,
    kwargs: dict[str, Any],
    token: str | None,
):
    if token:
        try:
            return load_dataset(repo, subset, token=token, **kwargs)
        except TypeError:
            LOGGER.debug("load_dataset(..., token=...) is unsupported; trying use_auth_token")
            try:
                return load_dataset(repo, subset, use_auth_token=token, **kwargs)
            except TypeError:
                LOGGER.debug("load_dataset(..., use_auth_token=...) is unsupported; relying on env token")

    return load_dataset(repo, subset, **kwargs)


def _select_split(dataset: Any, preferred: str) -> str:
    keys = list(getattr(dataset, "keys", lambda: [])())
    if not keys:
        raise ValueError("Cannot resolve split fallback: dataset has no split keys")

    preferred_order = [preferred, "train", "validation", "test"]
    for name in preferred_order:
        if name in keys:
            return name

    return keys[0]


def _disable_streaming_image_decode(dataset: Any, cfg: dict[str, Any]) -> Any:
    schema = cfg.get("schema") or {}
    if str(schema.get("image_mode", "image_field")) != "image_field":
        return dataset

    candidates: list[str] = []
    primary = schema.get("image_field")
    if isinstance(primary, str) and primary:
        candidates.append(primary)

    for value in schema.get("image_field_candidates", []):
        if isinstance(value, str) and value and value not in candidates:
            candidates.append(value)

    for fallback in ("image", "img", "jpg", "png"):
        if fallback not in candidates:
            candidates.append(fallback)

    available: set[str] = set()
    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, list):
        available.update(str(value) for value in column_names)
    features = getattr(dataset, "features", None)
    if isinstance(features, dict):
        available.update(str(value) for value in features.keys())

    for column_name in candidates:
        if available and column_name not in available:
            continue
        try:
            dataset = dataset.cast_column(column_name, datasets.Image(decode=False))
            LOGGER.debug("HF streaming decode disabled for column '%s'", column_name)
        except Exception:
            continue

    return dataset


def _project_streaming_columns(dataset: Any, cfg: dict[str, Any]) -> Any:
    schema = cfg.get("schema") or {}
    requested: list[str] = []

    for key in (
        "image_field",
        "url_field",
        "id_field",
    ):
        value = schema.get(key)
        if isinstance(value, str) and value:
            requested.append(value)

    for key in (
        "image_field_candidates",
        "url_field_candidates",
        "extra_fields",
    ):
        for value in schema.get(key, []):
            if isinstance(value, str) and value:
                requested.append(value)

    requested.extend(["image", "img", "id", "image_id"])

    dedup_requested: list[str] = []
    for value in requested:
        if value not in dedup_requested:
            dedup_requested.append(value)

    available: set[str] = set()
    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, list):
        available.update(str(value) for value in column_names)

    features = getattr(dataset, "features", None)
    if isinstance(features, dict):
        available.update(str(value) for value in features.keys())

    if not available:
        return dataset

    selected = [name for name in dedup_requested if name in available]
    if not selected:
        return dataset

    try:
        return dataset.select_columns(selected)
    except Exception:
        return dataset
