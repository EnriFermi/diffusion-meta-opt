from __future__ import annotations

import importlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import CELO_PAPER_17_TASKS


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    import_path: str
    source: str = "amoudgl/learned_optimization"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TaskRegistry:
    def __init__(self) -> None:
        self._items: OrderedDict[str, TaskSpec] = OrderedDict()

    def register(self, spec: TaskSpec) -> None:
        if spec.name in self._items:
            raise ValueError(f"Duplicate Celo benchmark task registered: {spec.name}")
        self._items[spec.name] = spec

    def get(self, name: str) -> TaskSpec:
        try:
            return self._items[name]
        except KeyError as exc:
            known = ", ".join(self._items)
            raise KeyError(f"Unknown Celo benchmark task {name!r}. Known tasks: {known}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._items)

    def specs(self) -> tuple[TaskSpec, ...]:
        return tuple(self._items.values())


def import_object(import_path: str) -> Any:
    path = str(import_path).strip()
    if not path:
        raise ValueError("Import path must be non-empty")
    if ":" in path:
        module_name, attr_path = path.split(":", 1)
    else:
        module_name, attr_path = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def build_task(name: str) -> Any:
    spec = TASK_REGISTRY.get(name)
    constructor = import_object(spec.import_path)
    return constructor()


def validate_task_imports(task_names: tuple[str, ...] | None = None) -> dict[str, str]:
    errors: dict[str, str] = {}
    for name in task_names or TASK_REGISTRY.names():
        try:
            obj = import_object(TASK_REGISTRY.get(name).import_path)
            if not callable(obj):
                errors[name] = "resolved object is not callable"
        except Exception as exc:  # pragma: no cover - exercised only with optional envs.
            errors[name] = f"{type(exc).__name__}: {exc}"
    return errors


def _default_registry() -> TaskRegistry:
    registry = TaskRegistry()
    mapping = {
        "ImageMLP_FashionMnist_Relu128x128": "learned_optimization.tasks.fixed.image_mlp.ImageMLP_FashionMnist_Relu128x128",
        "ImageMLP_Cifar10_128x128x128_LayerNorm_Relu": (
            "learned_optimization.tasks.fixed.image_mlp.ImageMLP_Cifar10_128x128x128_LayerNorm_Relu"
        ),
        "ImageMLP_Cifar10_128x128x128_Tanh_bs128": (
            "learned_optimization.tasks.fixed.image_mlp.ImageMLP_Cifar10_128x128x128_Tanh_bs128"
        ),
        "Conv_Cifar10_32x64x64": "learned_optimization.tasks.fixed.conv.Conv_Cifar10_32x64x64",
        "Conv_Cifar10_32x64x64_batchnorm": "learned_optimization.tasks.fixed.conv.Conv_Cifar10_32x64x64_batchnorm",
        "Conv_Cifar10_32x64x64_layernorm": "learned_optimization.tasks.fixed.conv.Conv_Cifar10_32x64x64_layernorm",
        "Conv_Cifar100_32x64x64": "learned_optimization.tasks.fixed.conv.Conv_Cifar100_32x64x64",
        "VIT_Cifar100_wideshallow": "learned_optimization.tasks.fixed.vit.VIT_Cifar100_wideshallow",
        "VIT_Cifar100_skinnydeep": "learned_optimization.tasks.fixed.vit.VIT_Cifar100_skinnydeep",
        "TransformerLM_LM1B_MultiRuntime_0": (
            "learned_optimization.tasks.fixed.transformer_lm.TransformerLM_LM1B_MultiRuntime_0"
        ),
        "TransformerLM_LM1B_MultiRuntime_2": (
            "learned_optimization.tasks.fixed.transformer_lm.TransformerLM_LM1B_MultiRuntime_2"
        ),
        "TransformerLM_LM1B_MultiRuntime_5": (
            "learned_optimization.tasks.fixed.transformer_lm.TransformerLM_LM1B_MultiRuntime_5"
        ),
        "ImageMLPAE_Cifar10_128x32x128_bs256": (
            "learned_optimization.tasks.fixed.image_mlp_ae.ImageMLPAE_Cifar10_128x32x128_bs256"
        ),
        "ImageMLPAE_Mnist_128x32x128_bs128": (
            "learned_optimization.tasks.fixed.image_mlp_ae.ImageMLPAE_Mnist_128x32x128_bs128"
        ),
        "RNNLM_lm1b32k_Patch32_LSTM256_Embed128": (
            "learned_optimization.tasks.fixed.rnn_lm.RNNLM_lm1b32k_Patch32_LSTM256_Embed128"
        ),
        "RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128": (
            "learned_optimization.tasks.fixed.rnn_lm.RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128"
        ),
        "LOpt_AdafacMLPLOpt_FashionMnist_50": "learned_optimization.tasks.fixed.lopt.LOpt_AdafacMLPLOpt_FashionMnist_50",
    }
    for name in CELO_PAPER_17_TASKS:
        registry.register(TaskSpec(name=name, import_path=mapping[name]))
    return registry


TASK_REGISTRY = _default_registry()

