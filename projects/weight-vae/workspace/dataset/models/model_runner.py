from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import torch

from dataset.data_raw.core.config import to_plain_dict
from dataset.models.base_virtual_model import BaseVirtualModel
from dataset.models.registry import create_model
from dataset.models.types import LayerIORecord


class ModelRunner:
    """LRU model runtime with per-model microbatching."""

    def __init__(self, global_cfg: Any, model_cfgs: dict[str, Any]) -> None:
        self.global_cfg = global_cfg
        self.global_cfg_dict = to_plain_dict(global_cfg)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_cfgs: dict[str, dict[str, Any]] = {
            str(name): to_plain_dict(cfg) for name, cfg in model_cfgs.items()
        }

        self.max_loaded_models = self._resolve_max_loaded_models()
        self._loaded_models: OrderedDict[str, BaseVirtualModel] = OrderedDict()
        self._num_runs = 0

    def run_model_on_images(self, model_name: str, pil_images: list[Any]) -> list[LayerIORecord]:
        if not pil_images:
            return []

        model = self._ensure_loaded(model_name)
        model_cfg = self.model_cfgs[model_name]
        micro_batch_size = max(1, int(model_cfg.get("batch_size", len(pil_images))))

        all_records: list[LayerIORecord] = []
        for chunk in _chunked(pil_images, micro_batch_size):
            all_records.extend(model.run(chunk))
            self._num_runs += 1

        self._touch(model_name)
        return merge_layer_records(all_records)

    def shutdown(self) -> None:
        while self._loaded_models:
            _, model = self._loaded_models.popitem(last=False)
            try:
                model.unload()
            except Exception:
                continue

    def stats(self) -> dict[str, Any]:
        return {
            "max_loaded_models": self.max_loaded_models,
            "loaded_models": list(self._loaded_models.keys()),
            "num_runs": self._num_runs,
            "available_models": sorted(self.model_cfgs.keys()),
        }

    def _ensure_loaded(self, model_name: str) -> BaseVirtualModel:
        if model_name in self._loaded_models:
            self._touch(model_name)
            return self._loaded_models[model_name]

        if model_name not in self.model_cfgs:
            available = ", ".join(sorted(self.model_cfgs.keys()))
            raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

        model_cfg = self.model_cfgs[model_name]
        model = create_model(model_name, cfg=model_cfg, global_cfg=self.global_cfg)
        model.load()

        self._loaded_models[model_name] = model
        self._touch(model_name)
        self._evict_if_needed()

        return self._loaded_models[model_name]

    def _evict_if_needed(self) -> None:
        while len(self._loaded_models) > self.max_loaded_models:
            old_name, old_model = self._loaded_models.popitem(last=False)
            self.logger.info("Evicting model from memory (LRU): %s", old_name)
            try:
                old_model.unload()
            except Exception as exc:
                self.logger.warning("Failed to unload model '%s': %s", old_name, exc)

    def _touch(self, model_name: str) -> None:
        self._loaded_models.move_to_end(model_name)

    def _resolve_max_loaded_models(self) -> int:
        data_cfg = self.global_cfg_dict.get("data", {})
        value = data_cfg.get("max_loaded_models")
        if value is None:
            models_cfg = self.global_cfg_dict.get("models", {})
            value = models_cfg.get("max_loaded_models", 1)

        resolved = int(value)
        return max(1, resolved)


def merge_layer_records(records: list[LayerIORecord]) -> list[LayerIORecord]:
    if not records:
        return []

    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for record in records:
        key = (record.model_name, record.layer_name)
        state = grouped.get(key)
        if state is None:
            state = {
                "model_name": record.model_name,
                "layer_name": record.layer_name,
                "weight": record.weight,
                "inputs": [],
                "outputs": [],
                "meta": {
                    "num_calls": 0,
                    "num_microbatches": 0,
                    "input_shape_list": [],
                    "output_shape_list": [],
                },
            }
            grouped[key] = state

        state["inputs"].append(record.inputs)
        state["outputs"].append(record.outputs)
        state["meta"]["num_microbatches"] += 1
        state["meta"]["num_calls"] += int(record.meta.get("num_calls", 0))
        state["meta"]["input_shape_list"].extend(record.meta.get("input_shape_list", []))
        state["meta"]["output_shape_list"].extend(record.meta.get("output_shape_list", []))

    merged: list[LayerIORecord] = []
    for state in grouped.values():
        if state["inputs"]:
            merged_inputs = _merge_row_chunks(state["inputs"])
        else:
            merged_inputs = torch.empty((0, 0), dtype=torch.float32)

        if state["outputs"]:
            merged_outputs = _merge_row_chunks(state["outputs"])
        else:
            merged_outputs = torch.empty((0, 0), dtype=torch.float32)

        state["meta"]["num_input_rows"] = int(merged_inputs.shape[0])
        state["meta"]["num_output_rows"] = int(merged_outputs.shape[0])

        merged.append(
            LayerIORecord(
                model_name=state["model_name"],
                layer_name=state["layer_name"],
                weight=state["weight"],
                inputs=merged_inputs,
                outputs=merged_outputs,
                meta=state["meta"],
            )
        )

    merged.sort(key=lambda item: (item.model_name, item.layer_name))
    return merged


def _chunked(values: list[Any], chunk_size: int) -> list[list[Any]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _merge_row_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return torch.empty((0, 0), dtype=torch.float32)
    if len(chunks) == 1:
        return chunks[0]

    first = chunks[0]
    tail_shape = tuple(first.shape[1:])
    dtype = first.dtype
    device = first.device
    if any(tuple(chunk.shape[1:]) != tail_shape or chunk.dtype != dtype or chunk.device != device for chunk in chunks[1:]):
        return torch.cat(chunks, dim=0)

    total_rows = sum(int(chunk.shape[0]) for chunk in chunks)
    merged = torch.empty((total_rows, *tail_shape), dtype=dtype, device=device)
    cursor = 0
    for chunk in chunks:
        rows = int(chunk.shape[0])
        merged[cursor : cursor + rows].copy_(chunk)
        cursor += rows
    return merged
