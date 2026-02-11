from __future__ import annotations

import logging
from typing import Any

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
