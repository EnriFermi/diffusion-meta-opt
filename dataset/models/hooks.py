from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn


def is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def flatten_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor (..., D) into (N, D) while preserving order."""

    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    return tensor.reshape(-1, tensor.shape[-1])


def make_linear_hook(layer_name: str, buffers_dict: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> Any:
    max_records_per_layer = cfg.get("max_records_per_layer")
    max_records_int = int(max_records_per_layer) if max_records_per_layer is not None else None

    def hook(_: nn.Module, inputs: Any, outputs: Any) -> None:
        buffer = buffers_dict[layer_name]
        buffer["num_calls"] += 1

        input_tensor = _extract_first_tensor(inputs)
        if input_tensor is not None:
            buffer["input_shape_list"].append(tuple(input_tensor.shape))
            flat_inputs = flatten_to_2d(input_tensor.detach())
            _append_with_cap(buffer["inputs"], flat_inputs, max_records_int)

        output_tensor = _extract_first_tensor(outputs)
        if output_tensor is not None:
            buffer["output_shape_list"].append(tuple(output_tensor.shape))
            flat_outputs = flatten_to_2d(output_tensor.detach())
            _append_with_cap(buffer["outputs"], flat_outputs, max_records_int)

    return hook


def attach_hooks(model: nn.Module, cfg: dict[str, Any] | None = None) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    hooks_cfg = cfg or {}
    include_regex = hooks_cfg.get("include_regex")
    exclude_regex = hooks_cfg.get("exclude_regex")
    include_pattern = re.compile(str(include_regex)) if include_regex else None
    exclude_pattern = re.compile(str(exclude_regex)) if exclude_regex else None

    max_layers = hooks_cfg.get("max_layers")
    max_layers_int = int(max_layers) if max_layers is not None else None

    handles: list[Any] = []
    buffers: dict[str, dict[str, Any]] = {}

    for layer_name, module in model.named_modules():
        if not is_linear(module):
            continue
        if include_pattern and not include_pattern.search(layer_name):
            continue
        if exclude_pattern and exclude_pattern.search(layer_name):
            continue
        if max_layers_int is not None and len(buffers) >= max_layers_int:
            break

        buffers[layer_name] = {
            "module": module,
            "inputs": [],
            "outputs": [],
            "input_shape_list": [],
            "output_shape_list": [],
            "num_calls": 0,
        }

        hook = make_linear_hook(layer_name, buffers, hooks_cfg)
        handles.append(module.register_forward_hook(hook))

    return handles, buffers


def detach_hooks(handles: list[Any]) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            continue


def _extract_first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
        return None

    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
        return None

    return None


def _append_with_cap(storage: list[torch.Tensor], values: torch.Tensor, max_rows: int | None) -> None:
    if max_rows is None:
        storage.append(values)
        return

    current_rows = sum(chunk.shape[0] for chunk in storage)
    remaining = max_rows - current_rows
    if remaining <= 0:
        return

    if values.shape[0] > remaining:
        values = values[:remaining]

    storage.append(values)
