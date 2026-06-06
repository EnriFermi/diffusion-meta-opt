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

from big_vae.runtime.artifacts import append_csv_row as append_standard_csv_row
from big_vae.runtime.artifacts import write_artifact_layout, write_json_file, write_yaml_file
from post_train_research.tinyvit_latent_h1.config import CometConfig, RunConfig


@dataclass(frozen=True, slots=True)
class RunPaths:
    root_dir: Path
    runs_dir: Path
    run_label_dir: Path
    run_dir: Path
    run_id: str
    log_file: Path
    config_yaml: Path
    config_json: Path
    summary_file: Path
    paired_results_file: Path
    aggregate_summary_file: Path
    starts_file: Path
    starts_dir: Path
    plots_dir: Path
    anchor_dir: Path
    anchor_curve_file: Path


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
    starts_dir = run_dir / "starts"
    plots_dir = run_dir / "plots"
    anchor_dir = run_dir / "anchor"
    starts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root_dir=root_dir,
        runs_dir=runs_dir,
        run_label_dir=run_label_dir,
        run_dir=run_dir,
        run_id=run_id,
        log_file=run_dir / "run.log",
        config_yaml=run_dir / "config_resolved.yaml",
        config_json=run_dir / "config_resolved.json",
        summary_file=run_dir / "summary.json",
        paired_results_file=run_dir / "paired_results.csv",
        aggregate_summary_file=run_dir / "aggregate_summary.csv",
        starts_file=run_dir / "starts.pt",
        starts_dir=starts_dir,
        plots_dir=plots_dir,
        anchor_dir=anchor_dir,
        anchor_curve_file=anchor_dir / "curves.csv",
    )


def configure_logger(paths: RunPaths) -> logging.Logger:
    logger = logging.getLogger(f"tinyvit_latent_h1.{paths.run_id}")
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
        kind="post_train.tinyvit_latent_h1",
        run_id=paths.run_id,
        files={
            "config_yaml": paths.config_yaml,
            "config_json": paths.config_json,
            "log": paths.log_file,
            "summary": paths.summary_file,
            "paired_results": paths.paired_results_file,
            "aggregate_summary": paths.aggregate_summary_file,
            "starts": paths.starts_file,
        },
        dirs={
            "starts": paths.starts_dir,
            "plots": paths.plots_dir,
            "anchor": paths.anchor_dir,
        },
        metadata={"run_label": cfg.run_label, "source_run_dir": cfg.source.run_dir},
    )


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    append_standard_csv_row(path, row)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    write_json_file(path, payload)


def update_run_index(paths: RunPaths, cfg: RunConfig, summary: dict[str, Any]) -> None:
    index_path = paths.root_dir / "run_index.jsonl"
    record = {
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_label": cfg.run_label,
        "source_run_dir": cfg.source.run_dir,
        "summary": summary,
    }
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


class CometTracker:
    def __init__(self, cfg: RunConfig, paths: RunPaths, logger: logging.Logger) -> None:
        self.logger = logger
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
            experiment.set_name(comet_cfg.experiment_name or cfg.run_label)
            for tag in list(cfg.experiment.tags) + list(comet_cfg.tags):
                if str(tag).strip():
                    experiment.add_tag(str(tag))
            experiment.log_parameters(flatten_for_tracking(cfg.to_dict()))
            experiment.log_parameters({"run_id": paths.run_id, "run_dir": str(paths.run_dir)})
            self.experiment = experiment
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
