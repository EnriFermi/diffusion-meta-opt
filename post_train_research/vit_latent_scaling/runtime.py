from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from big_vae.runtime.artifacts import append_csv_row, write_artifact_layout, write_json_file, write_yaml_file
from post_train_research.vit_latent_scaling.config import CometConfig, RunConfig


@dataclass(frozen=True, slots=True)
class RunPaths:
    root_dir: Path
    runs_dir: Path
    shared_checkpoints_root_dir: Path
    shared_checkpoints_dir: Path
    run_label_dir: Path
    run_dir: Path
    run_id: str
    log_file: Path
    metrics_file: Path
    summary_file: Path
    config_yaml: Path
    config_json: Path
    checkpoints_dir: Path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip().lower()).strip("-")
    return cleaned or "run"


def flatten_for_tracking(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_for_tracking(item, prefix=next_prefix))
        return flat
    if isinstance(value, (list, tuple)):
        flat[prefix] = json.dumps(value)
        return flat
    flat[prefix] = value
    return flat


def prepare_run_paths(cfg: RunConfig) -> RunPaths:
    root_dir = Path(cfg.storage.root_dir).expanduser().resolve()
    runs_dir = root_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    shared_checkpoints_root_dir = (
        Path(cfg.storage.shared_checkpoint_root_dir).expanduser().resolve()
        if cfg.storage.shared_checkpoint_root_dir.strip()
        else (root_dir / "checkpoints").resolve()
    )
    shared_checkpoints_root_dir.mkdir(parents=True, exist_ok=True)
    shared_checkpoints_dir = shared_checkpoints_root_dir / slugify(cfg.shared_checkpoint_label)
    shared_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_label_dir = runs_dir / slugify(cfg.run_label)
    run_label_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = slugify(cfg.run_label)
    run_id = f"{stamp}__{base}"
    run_dir = run_label_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{stamp}__{base}__{suffix:02d}"
        run_dir = run_label_dir / run_id
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root_dir=root_dir,
        runs_dir=runs_dir,
        shared_checkpoints_root_dir=shared_checkpoints_root_dir,
        shared_checkpoints_dir=shared_checkpoints_dir,
        run_label_dir=run_label_dir,
        run_dir=run_dir,
        run_id=run_id,
        log_file=run_dir / "run.log",
        metrics_file=run_dir / "metrics.csv",
        summary_file=run_dir / "summary.json",
        config_yaml=run_dir / "config_resolved.yaml",
        config_json=run_dir / "config_resolved.json",
        checkpoints_dir=checkpoints_dir,
    )


def configure_logger(paths: RunPaths) -> logging.Logger:
    logger = logging.getLogger(f"vit_latent_scaling.{paths.run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(paths.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_config_snapshots(paths: RunPaths, cfg: RunConfig, raw_cfg: dict[str, Any]) -> None:
    payload = {"resolved": cfg.to_dict(), "raw": raw_cfg}
    write_json_file(paths.config_json, payload)
    write_yaml_file(paths.config_yaml, payload)
    write_artifact_layout(
        paths.run_dir,
        kind="post_train.vit_latent_scaling",
        run_id=paths.run_id,
        files={
            "config_yaml": paths.config_yaml,
            "config_json": paths.config_json,
            "log": paths.log_file,
            "metrics": paths.metrics_file,
            "summary": paths.summary_file,
        },
        dirs={
            "checkpoints": paths.checkpoints_dir,
            "shared_checkpoints": paths.shared_checkpoints_dir,
        },
        metadata={"run_label": cfg.run_label, "profile": cfg.profile.name, "setup": cfg.setup.kind},
    )


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    append_csv_row(path, row)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    write_json_file(path, payload)


def update_run_index(paths: RunPaths, cfg: RunConfig, summary: dict[str, Any]) -> None:
    index_path = paths.root_dir / "run_index.jsonl"
    record = {
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_label": cfg.run_label,
        "run_label_dir": str(paths.run_label_dir),
        "shared_checkpoint_label": cfg.shared_checkpoint_label,
        "shared_checkpoints_dir": str(paths.shared_checkpoints_dir),
        "profile": cfg.profile.name,
        "dataset": cfg.data.dataset,
        "model_size": cfg.profile.model_size,
        "setup_kind": cfg.setup.kind,
        "init_kind": cfg.init.kind,
        "fresh_latent_mode": cfg.init.fresh_latent_mode,
        "summary": summary,
    }
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


class CometTracker:
    def __init__(self, cfg: RunConfig, paths: RunPaths, logger: logging.Logger) -> None:
        self.logger = logger
        self.enabled = False
        self.experiment: Any | None = None
        self._init(cfg.comet, cfg, paths)

    def _init(self, comet_cfg: CometConfig, cfg: RunConfig, paths: RunPaths) -> None:
        if not comet_cfg.enabled:
            return
        try:
            from comet_ml import Experiment, OfflineExperiment  # type: ignore
        except Exception as exc:
            self.logger.warning("Comet requested but comet_ml is unavailable: %s", exc)
            return
        try:
            if comet_cfg.api_key:
                experiment = Experiment(
                    api_key=comet_cfg.api_key,
                    project_name=comet_cfg.project_name,
                    workspace=comet_cfg.workspace or None,
                    auto_output_logging="simple",
                    log_code=comet_cfg.log_code,
                )
            else:
                experiment = OfflineExperiment(
                    project_name=comet_cfg.project_name,
                    workspace=comet_cfg.workspace or None,
                    auto_output_logging="simple",
                    log_code=comet_cfg.log_code,
                    offline_directory=comet_cfg.offline_directory or str(paths.run_dir / "comet_offline"),
                )
            experiment_name = comet_cfg.experiment_name or cfg.run_label
            experiment.set_name(experiment_name)
            for tag in list(cfg.storage.tags) + list(comet_cfg.tags):
                if str(tag).strip():
                    experiment.add_tag(str(tag))
            experiment.log_parameters(flatten_for_tracking(cfg.to_dict()))
            experiment.log_parameters({"run_id": paths.run_id, "run_dir": str(paths.run_dir)})
            self.experiment = experiment
            self.enabled = True
        except Exception as exc:
            self.logger.warning("Failed to initialize Comet tracker: %s", exc)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_metrics(metrics, step=int(step))
        except Exception as exc:
            self.logger.warning("Comet metric logging failed at step=%s: %s", step, exc)

    def log_asset(self, path: Path) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_asset(str(path))
        except Exception as exc:
            self.logger.warning("Comet asset logging failed for %s: %s", path, exc)

    def end(self) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.end()
        except Exception:
            pass
