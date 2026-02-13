from __future__ import annotations

from typing import Any, Callable

from dataset.data_raw.core.config import to_plain_dict
from dataset.models.base_virtual_model import BaseVirtualModel
from dataset.models.providers.transformers import (
    HFClipRunner,
    HFDenseRunner,
    HFEncDecRunner,
    HFSiglipRunner,
    HFViTRunner,
)

ModelBuilder = Callable[[Any, Any], BaseVirtualModel]

_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "hf_vit_runner": HFViTRunner,
    "hf_clip_runner": HFClipRunner,
    "hf_siglip_runner": HFSiglipRunner,
    "hf_dense_runner": HFDenseRunner,
    "hf_encdec_runner": HFEncDecRunner,
}


def register_model_builder(name: str, builder: ModelBuilder) -> None:
    key = str(name)
    if key in _MODEL_REGISTRY:
        raise ValueError(f"Model builder '{key}' is already registered")
    _MODEL_REGISTRY[key] = builder


def create_model(name: str, cfg: Any, global_cfg: Any) -> BaseVirtualModel:
    plain_cfg = to_plain_dict(cfg)
    provider = str(plain_cfg.get("provider", "transformers"))
    if provider != "transformers":
        raise KeyError(f"Unsupported model provider '{provider}'")

    runner = str(plain_cfg.get("runner") or _infer_runner(name=name, cfg=plain_cfg))

    if runner not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model runner '{runner}'. Available: {available}")

    builder = _MODEL_REGISTRY[runner]
    model = builder(cfg, global_cfg)

    if not getattr(model, "name", None):
        setattr(model, "name", name)

    return model


def list_model_builders() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


def _infer_runner(name: str, cfg: dict[str, Any]) -> str:
    explicit = cfg.get("model_family")
    if explicit:
        family = str(explicit).lower()
        if family in {"clip"}:
            return "hf_clip_runner"
        if family in {"siglip"}:
            return "hf_siglip_runner"
        if family in {"dense", "mask2former", "grounding_dino"}:
            return "hf_dense_runner"
        if family in {"encdec", "blip", "donut", "trocr"}:
            return "hf_encdec_runner"

    lowered = str(name).lower()
    if "siglip" in lowered:
        return "hf_siglip_runner"
    if "clip" in lowered:
        return "hf_clip_runner"
    if any(token in lowered for token in ["mask2former", "grounding"]):
        return "hf_dense_runner"
    if any(token in lowered for token in ["blip", "donut", "trocr"]):
        return "hf_encdec_runner"

    return "hf_vit_runner"
