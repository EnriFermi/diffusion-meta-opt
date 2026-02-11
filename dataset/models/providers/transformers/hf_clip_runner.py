from __future__ import annotations

from typing import Any

from PIL import Image

from dataset.models.providers.transformers.hf_base_runner import HFBaseRunner


class HFClipRunner(HFBaseRunner):
    """CLIP runner with vision-only/full modes."""

    def __init__(self, cfg: Any, global_cfg: Any) -> None:
        super().__init__(cfg=cfg, global_cfg=global_cfg)
        if self.run_mode in {"vision_only", "encoder_only"} and not self.include_regex:
            self.include_regex = r"^vision_model\."

    def _load_model(self, token: str | None) -> Any:
        from transformers import CLIPModel

        return self._from_pretrained(CLIPModel, token=token)

    def _load_processor(self, token: str | None) -> Any:
        from transformers import AutoProcessor

        return self._from_pretrained(AutoProcessor, token=token)

    def prepare_inputs(self, batch_pil: list[Image.Image]) -> dict[str, Any]:
        if self.run_mode == "full":
            prompts = self._build_prompts(len(batch_pil))
            payload = self._prepare_processor_payload(images=batch_pil, text=prompts, padding=True, truncation=True)
        else:
            payload = self._prepare_processor_payload(images=batch_pil)

        return self._to_device_inputs(payload)

    def forward_impl(self, model_inputs: dict[str, Any], batch_pil: list[Image.Image]) -> Any:
        del batch_pil
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        if self.run_mode in {"vision_only", "encoder_only"}:
            return self._model.get_image_features(pixel_values=model_inputs["pixel_values"])

        if self.run_mode == "full":
            return self._model(**model_inputs)

        raise ValueError(f"Unsupported run_mode='{self.run_mode}' for {self.name}")

    def _build_prompts(self, batch_size: int) -> list[str]:
        prompt = str(self.cfg_dict.get("text_prompt", "an image"))
        return [prompt for _ in range(batch_size)]
