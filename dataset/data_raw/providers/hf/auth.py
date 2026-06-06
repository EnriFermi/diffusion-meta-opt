from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from dataset.data_raw.core.config import nested_get, to_plain_dict
from dataset.data_raw.core.fs_utils import ensure_dir

HF_TOKEN_MISSING_ERROR = "HF token missing in top-level config: set hf.token"
_TOKEN_PLACEHOLDERS = {
    "YOUR_TOKEN_HERE",
    "YOUR_HF_TOKEN_HERE",
    "<YOUR_TOKEN_HERE>",
    "<YOUR_HF_TOKEN_HERE>",
    "CHANGE_ME",
}

_LOGIN_LOCK = threading.Lock()
_LOGIN_DONE = False
LOGGER = logging.getLogger(__name__)


def get_hf_token(cfg: Any) -> str | None:
    plain = to_plain_dict(cfg)
    token_value = nested_get(plain, "hf.token")
    if token_value is None:
        return None

    token = str(token_value).strip()
    if not token:
        return None
    if token.upper() in {"NULL", "NONE"}:
        return None
    if token in _TOKEN_PLACEHOLDERS:
        return None
    return token


def gated_datasets_missing_token(cfg: Any, active_dataset_cfgs: dict[str, Any]) -> list[str]:
    token = get_hf_token(cfg)
    if token:
        return []

    missing_for: list[str] = []
    for dataset_name, dataset_cfg in active_dataset_cfgs.items():
        plain = to_plain_dict(dataset_cfg)
        enabled = bool(plain.get("enabled", True))
        gated = bool(plain.get("gated", False))
        if enabled and gated:
            missing_for.append(str(dataset_name))
    return sorted(missing_for)


def validate_gated_datasets_token(cfg: Any, active_dataset_cfgs: dict[str, Any]) -> None:
    missing_for = gated_datasets_missing_token(cfg, active_dataset_cfgs)
    if missing_for:
        names = ", ".join(sorted(missing_for))
        raise ValueError(
            f"{HF_TOKEN_MISSING_ERROR}. Set hf.token in conf/big_vae/train/default.yaml. "
            f"Accept the dataset license on Hugging Face dataset page. "
            f"Ensure token has access. Affected datasets: {names}"
        )


def init_hf_auth(cfg: Any, allow_missing_token: bool = False) -> str | None:
    plain = to_plain_dict(cfg)
    hf_cfg = plain.get("hf", {})
    if not isinstance(hf_cfg, dict):
        raise TypeError("cfg.hf must be a mapping")

    token = get_hf_token(plain)
    if not token and not allow_missing_token:
        raise ValueError(
            f"{HF_TOKEN_MISSING_ERROR}. Set hf.token in conf/big_vae/train/default.yaml."
        )

    hf_home = str(hf_cfg.get("hf_home", "./data/hf_home"))
    datasets_cache = str(hf_cfg.get("datasets_cache", "./data/hf_datasets_cache"))
    hub_cache = str(hf_cfg.get("hub_cache", "./data/hf_hub_cache"))

    for folder in (hf_home, datasets_cache, hub_cache):
        ensure_dir(Path(folder))

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    os.environ["HF_HUB_CACHE"] = hub_cache

    if token:
        os.environ["HF_HUB_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        global _LOGIN_DONE
        with _LOGIN_LOCK:
            if not _LOGIN_DONE:
                from huggingface_hub import login

                try:
                    login(token=token, add_to_git_credential=False)
                except Exception as exc:
                    # HF `/whoami-v2` is rate-limited and may fail under many short-lived processes.
                    # Keep going with explicit token in env; hub/datasets calls still receive auth.
                    if _is_whoami_rate_limited(exc) or _is_stored_token_lookup_error(exc):
                        LOGGER.warning(
                            "HF login skipped due non-fatal auth cache issue; continuing with token from env. error=%s",
                            exc,
                        )
                    else:
                        raise
                _LOGIN_DONE = True

    return token


def _is_whoami_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    if "whoami-v2" not in text:
        return False
    return ("429" in text) or ("too many requests" in text) or ("rate limit" in text)


def _is_stored_token_lookup_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "stored_tokens" in text and "not found" in text and "token" in text
