from __future__ import annotations

from typing import Any

from PIL import Image

from dataset.models.providers.transformers.hf_base_runner import HFBaseRunner


class HFDenseRunner(HFBaseRunner):
    """Dense prediction runner (Mask2Former / GroundingDINO / DETR / SegFormer)."""

    def __init__(self, cfg: Any, global_cfg: Any) -> None:
        super().__init__(cfg=cfg, global_cfg=global_cfg)
        self.architecture = str(self.cfg_dict.get("architecture", "mask2former")).lower()

    def _load_model(self, token: str | None) -> Any:
        from transformers import (
            DetrForObjectDetection,
            GroundingDinoForObjectDetection,
            Mask2FormerForUniversalSegmentation,
            SegformerForSemanticSegmentation,
        )

        if self.architecture == "grounding_dino":
            return self._from_pretrained(GroundingDinoForObjectDetection, token=token)
        if self.architecture == "mask2former":
            return self._from_pretrained(Mask2FormerForUniversalSegmentation, token=token)
        if self.architecture == "detr":
            return self._from_pretrained(DetrForObjectDetection, token=token)
        if self.architecture == "segformer":
            return self._from_pretrained(SegformerForSemanticSegmentation, token=token)

        raise ValueError(f"Unsupported dense architecture='{self.architecture}' for {self.name}")

    def _load_processor(self, token: str | None) -> Any:
        from transformers import AutoImageProcessor, AutoProcessor

        try:
            return self._from_pretrained(AutoProcessor, token=token)
        except Exception:
            return self._from_pretrained(AutoImageProcessor, token=token)

    def prepare_inputs(self, batch_pil: list[Image.Image]) -> dict[str, Any]:
        if self.architecture == "grounding_dino":
            prompt = self._grounding_prompt()
            prompts = [prompt for _ in range(len(batch_pil))]
            payload = self._prepare_processor_payload(images=batch_pil, text=prompts, padding=True, truncation=True)
        else:
            payload = self._prepare_processor_payload(images=batch_pil)

        return self._to_device_inputs(payload)

    def forward_impl(self, model_inputs: dict[str, Any], batch_pil: list[Image.Image]) -> Any:
        del batch_pil
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        if self.run_mode not in {"vision_only", "encoder_only", "full"}:
            raise ValueError(f"Unsupported run_mode='{self.run_mode}' for {self.name}")

        # GroundingDINO requires text fields in forward; for vision-only we send a default prompt.
        return self._model(**model_inputs)

    def _grounding_prompt(self) -> str:
        if self.run_mode == "full":
            return str(self.cfg_dict.get("text_prompt", "a generic object"))
        return str(self.cfg_dict.get("dummy_text_prompt", "object"))
