from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional at import time for lightweight tests
    OmegaConf = None  # type: ignore[assignment]


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert DictConfig/dict-like config into a regular Python dict."""

    if cfg is None:
        return {}

    if isinstance(cfg, dict):
        return deepcopy(cfg)

    if OmegaConf is not None and OmegaConf.is_config(cfg):
        plain = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(plain, dict):
            return plain

    raise TypeError(f"Unsupported config type: {type(cfg)}")


def nested_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for chunk in path.split("."):
        if not isinstance(current, dict):
            return default
        if chunk not in current:
            return default
        current = current[chunk]
    return current
