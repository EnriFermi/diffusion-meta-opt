from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from dataset.models.providers.transformers.hf_base_runner import HFBaseRunner


class HFEncDecRunner(HFBaseRunner):
    """Encoder-decoder runner for BLIP / Donut / TrOCR families."""

    def __init__(self, cfg: Any, global_cfg: Any) -> None:
        super().__init__(cfg=cfg, global_cfg=global_cfg)
        self.architecture = str(self.cfg_dict.get("architecture", "vision_encoder_decoder")).lower()

        if self.run_mode in {"vision_only", "vision_encoder_only", "encoder_only"} and not self.include_regex:
            self.include_regex = r"^(vision_model|encoder)\."

    def _load_model(self, token: str | None) -> Any:
        from transformers import BlipForConditionalGeneration, VisionEncoderDecoderModel

        if self.architecture == "blip":
            return self._from_pretrained(BlipForConditionalGeneration, token=token)

        if self.architecture in {"vision_encoder_decoder", "donut", "trocr"}:
            return self._from_pretrained(VisionEncoderDecoderModel, token=token)

        raise ValueError(f"Unsupported encoder-decoder architecture='{self.architecture}' for {self.name}")

    def _load_processor(self, token: str | None) -> Any:
        from transformers import AutoProcessor

        return self._from_pretrained(AutoProcessor, token=token)

    def prepare_inputs(self, batch_pil: list[Image.Image]) -> dict[str, Any]:
        payload = self._prepare_processor_payload(images=batch_pil)
        return self._to_device_inputs(payload)

    def forward_impl(self, model_inputs: dict[str, Any], batch_pil: list[Image.Image]) -> Any:
        del batch_pil
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        if self.run_mode in {"vision_only", "vision_encoder_only", "encoder_only"}:
            return self._run_encoder_only(model_inputs)

        if self.run_mode == "full":
            full_inputs = dict(model_inputs)
            if "decoder_input_ids" not in full_inputs:
                full_inputs["decoder_input_ids"] = self._default_decoder_input_ids(
                    batch_size=int(model_inputs["pixel_values"].shape[0])
                )
            return self._model(**full_inputs)

        raise ValueError(f"Unsupported run_mode='{self.run_mode}' for {self.name}")

    def _run_encoder_only(self, model_inputs: dict[str, Any]) -> Any:
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        if self.architecture == "blip":
            vision_model = getattr(self._model, "vision_model", None)
            if vision_model is None and hasattr(self._model, "model"):
                vision_model = getattr(self._model.model, "vision_model", None)
            if vision_model is None:
                return self._model(pixel_values=model_inputs["pixel_values"])
            return vision_model(pixel_values=model_inputs["pixel_values"])

        encoder = None
        get_encoder = getattr(self._model, "get_encoder", None)
        if callable(get_encoder):
            encoder = get_encoder()
        if encoder is None:
            encoder = getattr(self._model, "encoder", None)
        if encoder is None:
            return self._model(pixel_values=model_inputs["pixel_values"])

        encoder_kwargs = {"pixel_values": model_inputs["pixel_values"]}
        if "pixel_mask" in model_inputs:
            encoder_kwargs["pixel_mask"] = model_inputs["pixel_mask"]
        return encoder(**encoder_kwargs)

    def _default_decoder_input_ids(self, batch_size: int) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        start_token_id = self.cfg_dict.get("decoder_start_token_id")
        if start_token_id is None:
            config = getattr(self._model, "config", None)
            if config is not None:
                start_token_id = getattr(config, "decoder_start_token_id", None)
                if start_token_id is None:
                    start_token_id = getattr(config, "bos_token_id", None)
        if start_token_id is None:
            start_token_id = 0

        return torch.full((batch_size, 1), int(start_token_id), dtype=torch.long, device=self.device)
