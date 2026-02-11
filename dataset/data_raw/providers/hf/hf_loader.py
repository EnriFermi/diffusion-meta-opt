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

    kwargs = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
    }

    dataset = _load_dataset_with_best_auth(repo=repo, subset=subset, kwargs=kwargs, token=token)

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
