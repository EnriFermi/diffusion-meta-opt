from __future__ import annotations

import logging
import traceback
from collections import OrderedDict
from typing import Any

from dataset.data_raw.core.config import to_plain_dict
from dataset.models.base_virtual_model import BaseVirtualModel
from dataset.models.model_runner import merge_layer_records
from dataset.models.registry import create_model
from dataset.models.types import LayerIORecord
from dataset.shared.load_report import LoadReportWriter


class ModelPool:
    """LRU model pool with runtime device override and local-only loading option."""

    def __init__(
        self,
        global_cfg: Any,
        model_cfgs: dict[str, Any],
        device_override: str | None,
        max_loaded_models: int,
        runtime_local_only: bool,
        release_device_on_unload: bool = True,
        empty_cuda_cache_on_unload: bool = True,
        load_report: LoadReportWriter | None = None,
    ) -> None:
        self.global_cfg = global_cfg
        self.global_cfg_dict = to_plain_dict(global_cfg)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_report = load_report

        self.model_cfgs: dict[str, dict[str, Any]] = {
            str(name): to_plain_dict(cfg) for name, cfg in model_cfgs.items()
        }

        self.device_override = device_override
        self.max_loaded_models = max(1, int(max_loaded_models))
        self.runtime_local_only = bool(runtime_local_only)
        self.release_device_on_unload = bool(release_device_on_unload)
        self.empty_cuda_cache_on_unload = bool(empty_cuda_cache_on_unload)

        self._loaded_models: OrderedDict[str, BaseVirtualModel] = OrderedDict()

    def run(self, model_name: str, pil_batch: list[Any]) -> list[LayerIORecord]:
        if not pil_batch:
            return []

        model = self._ensure_loaded(model_name)

        model_cfg = self.model_cfgs[model_name]
        micro_batch_size = max(1, int(model_cfg.get("batch_size", len(pil_batch))))

        records: list[LayerIORecord] = []
        for start in range(0, len(pil_batch), micro_batch_size):
            chunk = pil_batch[start : start + micro_batch_size]
            records.extend(model.run(chunk))

        self._touch(model_name)
        return merge_layer_records(records)

    def preload_local(self, model_name: str) -> None:
        if model_name not in self.model_cfgs:
            raise KeyError(f"Unknown model '{model_name}'")

        preload_cfg = dict(self.model_cfgs[model_name])
        preload_cfg["device"] = "cpu"
        preload_cfg["local_files_only"] = False

        model = create_model(model_name, cfg=preload_cfg, global_cfg=self.global_cfg)
        try:
            model.load()
        finally:
            model.unload()

    def predownload_models(self, model_names: list[str]) -> None:
        for model_name in model_names:
            self.logger.info("Predownloading model artifacts: %s", model_name)
            self.preload_local(model_name)

    def unload_all(self) -> None:
        while self._loaded_models:
            _, model = self._loaded_models.popitem(last=False)
            try:
                model.unload()
            except Exception:
                continue

    def stats(self) -> dict[str, Any]:
        return {
            "device_override": self.device_override,
            "runtime_local_only": self.runtime_local_only,
            "max_loaded_models": self.max_loaded_models,
            "release_device_on_unload": self.release_device_on_unload,
            "empty_cuda_cache_on_unload": self.empty_cuda_cache_on_unload,
            "loaded_models": list(self._loaded_models.keys()),
        }

    def _ensure_loaded(self, model_name: str) -> BaseVirtualModel:
        if model_name in self._loaded_models:
            self._touch(model_name)
            return self._loaded_models[model_name]

        if model_name not in self.model_cfgs:
            available = ", ".join(sorted(self.model_cfgs.keys()))
            raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

        runtime_cfg = dict(self.model_cfgs[model_name])
        if self.device_override:
            runtime_cfg["device"] = self.device_override
        runtime_cfg["local_files_only"] = self.runtime_local_only
        runtime_cfg["release_device_on_unload"] = self.release_device_on_unload
        runtime_cfg["empty_cuda_cache_on_unload"] = self.empty_cuda_cache_on_unload

        model: BaseVirtualModel | None = None
        try:
            model = create_model(model_name, cfg=runtime_cfg, global_cfg=self.global_cfg)
            model.load()
        except Exception as exc:
            if model is not None:
                try:
                    model.unload()
                except Exception:
                    pass

            if self.load_report is not None:
                self.load_report.mark_model_failed(
                    model_name=model_name,
                    phase="load",
                    error=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            raise

        if self.load_report is not None:
            self.load_report.mark_model_loaded(
                model_name=model_name,
                phase="load",
                details={
                    "device": runtime_cfg.get("device"),
                    "local_files_only": bool(runtime_cfg.get("local_files_only", False)),
                },
            )

        self._loaded_models[model_name] = model
        self._touch(model_name)
        self._evict_if_needed()

        return self._loaded_models[model_name]

    def _evict_if_needed(self) -> None:
        while len(self._loaded_models) > self.max_loaded_models:
            old_name, old_model = self._loaded_models.popitem(last=False)
            self.logger.info("Evicting model from collector pool (LRU): %s", old_name)
            try:
                old_model.unload()
            except Exception as exc:
                self.logger.warning("Failed to unload model '%s': %s", old_name, exc)

    def _touch(self, model_name: str) -> None:
        self._loaded_models.move_to_end(model_name)
