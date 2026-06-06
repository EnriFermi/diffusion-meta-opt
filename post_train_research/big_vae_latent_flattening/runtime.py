from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from big_vae.runtime.artifacts import append_csv_row, write_artifact_layout, write_json_file, write_yaml_file
from post_train_research.big_vae_latent_flattening.config import RunConfig, project_root


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    metrics_csv: Path
    config_yaml: Path
    config_json: Path
    log_path: Path
    summary_json: Path


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
        config_yaml=run_dir / "config_resolved.yaml",
        config_json=run_dir / "config_resolved.json",
        log_path=run_dir / "run.log",
        summary_json=run_dir / "summary.json",
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
    payload = {"resolved": cfg.to_dict(), "raw": dict(raw_cfg)}
    write_json_file(paths.config_json, payload)
    write_yaml_file(paths.config_yaml, payload)
    write_artifact_layout(
        paths.run_dir,
        kind="post_train.big_vae_latent_flattening",
        run_id=paths.run_id,
        files={
            "config_yaml": paths.config_yaml,
            "config_json": paths.config_json,
            "log": paths.log_path,
            "metrics": paths.metrics_csv,
            "summary": paths.summary_json,
        },
        dirs={"checkpoints": paths.checkpoints_dir},
        metadata={"run_label": cfg.run_label, "big_vae_checkpoint": cfg.big_vae.checkpoint},
    )


def append_metrics_row(path: Path, row: Mapping[str, Any]) -> None:
    append_csv_row(path, row)


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
