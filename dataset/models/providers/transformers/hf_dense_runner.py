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
        if self.architecture == "grounding_dino" and self.run_mode in {"vision_only", "encoder_only"}:
            # Vision-only path does not need tokenizer/text processor.
            try:
                return self._load_auto_image_processor_like(token)
            except Exception:
                return self._load_auto_processor(token)

        try:
            return self._load_auto_processor(token)
        except Exception:
            return self._load_auto_image_processor_like(token)

    def prepare_inputs(self, batch_pil: list[Image.Image]) -> dict[str, Any]:
        if self.architecture == "grounding_dino" and self.run_mode == "full":
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

        if self.architecture == "grounding_dino" and self.run_mode in {"vision_only", "encoder_only"}:
            return self._run_grounding_dino_vision_only(model_inputs)

        return self._model(**model_inputs)

    def _grounding_prompt(self) -> str:
        if self.run_mode == "full":
            return str(self.cfg_dict.get("text_prompt", "a generic object"))
        return str(self.cfg_dict.get("dummy_text_prompt", "object"))

    def _run_grounding_dino_vision_only(self, model_inputs: dict[str, Any]) -> Any:
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        core_model = getattr(self._model, "model", None)
        if core_model is None:
            return self._model(**model_inputs)

        backbone = getattr(core_model, "backbone", None)
        if backbone is None:
            return self._model(**model_inputs)

        pixel_values = model_inputs.get("pixel_values")
        if pixel_values is None:
            raise KeyError(f"Missing 'pixel_values' for {self.name}")

        forward_kwargs: dict[str, Any] = {"pixel_values": pixel_values}
        if "pixel_mask" in model_inputs:
            forward_kwargs["pixel_mask"] = model_inputs["pixel_mask"]

        try:
            return backbone(**forward_kwargs)
        except TypeError:
            forward_kwargs.pop("pixel_mask", None)
            return backbone(**forward_kwargs)
