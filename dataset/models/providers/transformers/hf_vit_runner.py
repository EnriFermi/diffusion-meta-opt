from __future__ import annotations

from typing import Any

from PIL import Image

from dataset.models.providers.transformers.hf_base_runner import HFBaseRunner


class HFViTRunner(HFBaseRunner):
    """Generic vision-encoder runner for DINOv2/DeiT/BEiT/ViT-MAE/SwinV2-like models."""

    def _load_model(self, token: str | None) -> Any:
        from transformers import AutoModel

        return self._from_pretrained(AutoModel, token=token)

    def _load_processor(self, token: str | None) -> Any:
        from transformers import AutoImageProcessor, AutoProcessor

        try:
            return self._from_pretrained(AutoImageProcessor, token=token)
        except Exception:
            return self._from_pretrained(AutoProcessor, token=token)

    def prepare_inputs(self, batch_pil: list[Image.Image]) -> dict[str, Any]:
        payload = self._prepare_processor_payload(images=batch_pil)
        return self._to_device_inputs(payload)

    def forward_impl(self, model_inputs: dict[str, Any], batch_pil: list[Image.Image]) -> Any:
        del batch_pil
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        if self.run_mode not in {"vision_only", "encoder_only", "full"}:
            raise ValueError(f"Unsupported run_mode='{self.run_mode}' for {self.name}")

        return self._model(**model_inputs)
