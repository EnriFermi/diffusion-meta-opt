from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from post_train_research.big_vae_latent_flattening.config import RunConfig, project_root


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    metrics_csv: Path
    config_json: Path
    raw_config_json: Path
    log_path: Path


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = project_root() / resolved
    return resolved


def prepare_run_paths(cfg: RunConfig) -> RunPaths:
    root = resolve_path(cfg.storage.root_dir)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / f"{run_id}__{cfg.run_label}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        metrics_csv=run_dir / "metrics.csv",
        config_json=run_dir / "config.json",
        raw_config_json=run_dir / "raw_config.json",
        log_path=run_dir / "run.log",
    )


def configure_logger(paths: RunPaths) -> logging.Logger:
    logger = logging.getLogger("big_vae_latent_flattening")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(paths.log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def write_config_snapshots(paths: RunPaths, cfg: RunConfig, raw_cfg: Mapping[str, Any]) -> None:
    paths.config_json.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    paths.raw_config_json.write_text(json.dumps(dict(raw_cfg), indent=2, sort_keys=True), encoding="utf-8")


def append_metrics_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def save_checkpoint(
    path: Path,
    *,
    cfg: RunConfig,
    flow: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Mapping[str, Any],
    big_vae_checkpoint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "config": cfg.to_dict(),
            "flow_state": flow.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": dict(metrics),
            "big_vae_checkpoint": str(big_vae_checkpoint),
        },
        path,
    )

