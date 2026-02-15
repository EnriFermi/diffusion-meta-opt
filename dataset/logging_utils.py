from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.core.fs_utils import ensure_dir

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_PATH_ENV = "DATA_PIPELINE_LOG_FILE"


def resolve_log_path(cfg: Any) -> Path:
    """Resolve final log file path from config (with env override for child processes)."""
    env_path = os.environ.get(LOG_PATH_ENV)
    if env_path:
        return Path(env_path)

    plain = to_plain_dict(cfg)
    logging_cfg = plain.get("logging", {})
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}

    file_path = logging_cfg.get("file_path")
    if file_path:
        return Path(str(file_path))

    log_dir = Path(str(logging_cfg.get("dir", "logs")))
    file_name = str(logging_cfg.get("file_name", "run.log"))
    return log_dir / file_name


def configure_root_logging(cfg: Any, rank: int = 0, force: bool = True) -> Path:
    """Configure root logger with dual sink: console + file."""
    plain = to_plain_dict(cfg)
    data_cfg = plain.get("data", {})
    logging_cfg = plain.get("logging", {})

    if not isinstance(data_cfg, dict):
        data_cfg = {}
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}

    level_name = str(data_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    if rank != 0:
        level = max(level, logging.WARNING)

    fmt = str(logging_cfg.get("format", DEFAULT_LOG_FORMAT))
    log_path = resolve_log_path(plain)

    ensure_dir(log_path.parent)
    os.environ[LOG_PATH_ENV] = str(log_path)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(filename=log_path, mode="a", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(level=level, handlers=handlers, force=force)
    return log_path

