from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class LoadReportWriter:
    """
    Persists dataset/model load status to two files:
    - events JSONL (append-only)
    - summary JSON (latest consolidated snapshot)
    """

    def __init__(self, cfg_dict: dict[str, Any]) -> None:
        self._lock = threading.Lock()

        events_path, summary_path = _resolve_paths(cfg_dict)
        self.events_path = events_path
        self.summary_path = summary_path

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

        self._expected_datasets: set[str] = set()
        self._expected_models: set[str] = set()

        self._loaded_datasets: set[str] = set()
        self._loaded_models: set[str] = set()

        self._failed_datasets: dict[str, dict[str, Any]] = {}
        self._failed_models: dict[str, dict[str, Any]] = {}

    def set_expected(self, datasets: list[str], models: list[str]) -> None:
        with self._lock:
            self._expected_datasets = {str(item) for item in datasets}
            self._expected_models = {str(item) for item in models}
            self._append_event_locked(
                "expected",
                {
                    "datasets": sorted(self._expected_datasets),
                    "models": sorted(self._expected_models),
                },
            )
            self._write_summary_locked()

    def mark_dataset_loaded(self, dataset_name: str, *, phase: str, details: dict[str, Any] | None = None) -> None:
        name = str(dataset_name)
        with self._lock:
            self._loaded_datasets.add(name)
            self._failed_datasets.pop(name, None)
            self._append_event_locked(
                "dataset_loaded",
                {
                    "dataset": name,
                    "phase": str(phase),
                    "details": details or {},
                },
            )
            self._write_summary_locked()

    def mark_dataset_failed(
        self,
        dataset_name: str,
        *,
        phase: str,
        error: str,
        traceback_text: str,
    ) -> None:
        name = str(dataset_name)
        with self._lock:
            self._failed_datasets[name] = {
                "phase": str(phase),
                "error": str(error),
                "traceback": str(traceback_text),
                "timestamp": time.time(),
            }
            self._loaded_datasets.discard(name)
            self._append_event_locked(
                "dataset_failed",
                {
                    "dataset": name,
                    "phase": str(phase),
                    "error": str(error),
                    "traceback": str(traceback_text),
                },
            )
            self._write_summary_locked()

    def mark_model_loaded(self, model_name: str, *, phase: str, details: dict[str, Any] | None = None) -> None:
        name = str(model_name)
        with self._lock:
            self._loaded_models.add(name)
            self._failed_models.pop(name, None)
            self._append_event_locked(
                "model_loaded",
                {
                    "model": name,
                    "phase": str(phase),
                    "details": details or {},
                },
            )
            self._write_summary_locked()

    def mark_model_failed(
        self,
        model_name: str,
        *,
        phase: str,
        error: str,
        traceback_text: str,
    ) -> None:
        name = str(model_name)
        with self._lock:
            self._failed_models[name] = {
                "phase": str(phase),
                "error": str(error),
                "traceback": str(traceback_text),
                "timestamp": time.time(),
            }
            self._loaded_models.discard(name)
            self._append_event_locked(
                "model_failed",
                {
                    "model": name,
                    "phase": str(phase),
                    "error": str(error),
                    "traceback": str(traceback_text),
                },
            )
            self._write_summary_locked()

    def mark_runtime_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._append_event_locked(str(event), payload or {})
            self._write_summary_locked()

    def _append_event_locked(self, event: str, payload: dict[str, Any]) -> None:
        line = {
            "timestamp": time.time(),
            "event": str(event),
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def _write_summary_locked(self) -> None:
        pending_datasets = sorted(self._expected_datasets - self._loaded_datasets - set(self._failed_datasets.keys()))
        pending_models = sorted(self._expected_models - self._loaded_models - set(self._failed_models.keys()))

        payload = {
            "updated_at": time.time(),
            "paths": {
                "events": str(self.events_path),
                "summary": str(self.summary_path),
            },
            "expected": {
                "datasets": sorted(self._expected_datasets),
                "models": sorted(self._expected_models),
            },
            "loaded": {
                "datasets": sorted(self._loaded_datasets),
                "models": sorted(self._loaded_models),
            },
            "failed": {
                "datasets": self._failed_datasets,
                "models": self._failed_models,
            },
            "pending": {
                "datasets": pending_datasets,
                "models": pending_models,
            },
        }

        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _resolve_paths(cfg_dict: dict[str, Any]) -> tuple[Path, Path]:
    logging_cfg = cfg_dict.get("logging", {}) if isinstance(cfg_dict, dict) else {}
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}
    collector_cfg = cfg_dict.get("collector", {}) if isinstance(cfg_dict, dict) else {}
    if not isinstance(collector_cfg, dict):
        collector_cfg = {}
    diagnostics_cfg = collector_cfg.get("diagnostics", {})
    if not isinstance(diagnostics_cfg, dict):
        diagnostics_cfg = {}
    training_artifacts_cfg = cfg_dict.get("training_artifacts", {}) if isinstance(cfg_dict, dict) else {}
    if not isinstance(training_artifacts_cfg, dict):
        training_artifacts_cfg = {}

    file_path = logging_cfg.get("file_path")
    if file_path:
        base_log_path = Path(str(file_path))
        default_directory = base_log_path.parent
        stem = base_log_path.stem
    else:
        default_directory = Path(str(logging_cfg.get("dir", "logs")))
        stem = str(logging_cfg.get("project_name", "run"))

    reports_dir = diagnostics_cfg.get("load_reports_dir")
    if reports_dir:
        directory = Path(str(reports_dir))
    else:
        artifacts_reports_dir = training_artifacts_cfg.get("reports_dir")
        if artifacts_reports_dir:
            directory = Path(str(artifacts_reports_dir))
        else:
            directory = default_directory

    events_path = directory / f"{stem}_load_report.jsonl"
    summary_path = directory / f"{stem}_load_summary.json"
    return events_path, summary_path
