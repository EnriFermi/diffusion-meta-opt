from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.providers.hf.auth import HF_TOKEN_MISSING_ERROR, get_hf_token, init_hf_auth
from dataset.models.base_virtual_model import BaseVirtualModel
from dataset.models.hooks import attach_hooks, detach_hooks
from dataset.models.types import LayerIORecord


class HFBaseRunner(BaseVirtualModel):
    """Shared implementation for HF model wrappers with Linear hooks."""

    def __init__(self, cfg: Any, global_cfg: Any) -> None:
        super().__init__(cfg=cfg, global_cfg=global_cfg)

        self.cfg_dict = to_plain_dict(cfg)
        self.global_cfg_dict = to_plain_dict(global_cfg)

        self.name = str(self.cfg_dict.get("name", "unknown_model"))
        self.logger = logging.getLogger(f"dataset.models.{self.name}")

        self.hf_repo = str(self.cfg_dict["hf_repo"])
        self.revision = self.cfg_dict.get("revision")
        self.gated = bool(self.cfg_dict.get("gated", False))
        self.trust_remote_code = bool(self.cfg_dict.get("trust_remote_code", False))
        self.local_files_only = bool(self.cfg_dict.get("local_files_only", False))

        self.device = self._resolve_device(str(self.cfg_dict.get("device", "auto")))
        configured_dtype = self._resolve_dtype(str(self.cfg_dict.get("dtype", "float32")))
        self.run_dtype = self._resolve_run_dtype(configured_dtype=configured_dtype, device=self.device)

        self.run_mode = str(self.cfg_dict.get("run_mode", "vision_only")).lower()

        limits_cfg = self.cfg_dict.get("limits", {})
        hook_filter_cfg = self.cfg_dict.get("hook_filter", {})

        self.max_records_per_layer = limits_cfg.get("max_records_per_layer")
        self.max_layers = limits_cfg.get("max_layers")
        self.include_regex = hook_filter_cfg.get("include_regex")
        self.exclude_regex = hook_filter_cfg.get("exclude_regex")

        data_root = self._resolve_data_root()
        cache_subdir = str(self.cfg_dict.get("cache_subdir", f"models/{self.name}"))
        self.cache_dir = Path(data_root) / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model: Any | None = None
        self._processor: Any | None = None
        self._hook_handles: list[Any] = []
        self._hook_buffers: dict[str, dict[str, Any]] = {}

        self._loaded = False
        self._num_runs = 0

    def load(self) -> None:
        if self._loaded:
            return

        token = get_hf_token(self.global_cfg_dict)
        if self.gated and not token:
            raise ValueError(
                f"{HF_TOKEN_MISSING_ERROR}. Set hf.token in conf/config.yaml. "
                f"Accept the dataset license on Hugging Face dataset page. "
                f"Ensure token has access. Affected model: {self.name}"
            )

        token = init_hf_auth(self.global_cfg_dict, allow_missing_token=not self.gated)

        self._processor = self._load_processor(token)
        self._model = self._load_model(token)
        self._model.eval()

        if self.run_dtype != torch.float32:
            self._model = self._model.to(dtype=self.run_dtype)
        self._model = self._model.to(self.device)

        hook_cfg = {
            "include_regex": self.include_regex,
            "exclude_regex": self.exclude_regex,
            "max_records_per_layer": self.max_records_per_layer,
            "max_layers": self.max_layers,
        }
        self._hook_handles, self._hook_buffers = attach_hooks(self._model, cfg=hook_cfg)

        self._loaded = True
        self.logger.info(
            "Loaded model '%s' on %s (%s) with %s Linear hooks",
            self.name,
            self.device,
            self.run_dtype,
            len(self._hook_buffers),
        )

    def unload(self) -> None:
        if not self._loaded:
            return

        detach_hooks(self._hook_handles)
        self._hook_handles = []
        self._hook_buffers = {}

        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass

        self._model = None
        self._processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, batch_pil: list[Image.Image]) -> list[LayerIORecord]:
        if not batch_pil:
            return []

        if not self._loaded:
            self.load()

        if self._model is None or self._processor is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        self._reset_buffers()

        model_inputs = self.prepare_inputs(batch_pil)
        with torch.no_grad():
            self.forward_impl(model_inputs=model_inputs, batch_pil=batch_pil)

        self._num_runs += 1
        return self._build_records()

    def stats(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "hf_repo": self.hf_repo,
            "run_mode": self.run_mode,
            "loaded": self._loaded,
            "device": str(self.device),
            "dtype": str(self.run_dtype),
            "num_linear_layers": len(self._hook_buffers),
            "num_runs": self._num_runs,
            "cache_dir": str(self.cache_dir),
            "local_files_only": self.local_files_only,
        }

    @abstractmethod
    def _load_model(self, token: str | None) -> Any:
        """Load HF model."""

    @abstractmethod
    def _load_processor(self, token: str | None) -> Any:
        """Load HF processor/image processor."""

    @abstractmethod
    def prepare_inputs(self, batch_pil: list[Image.Image]) -> dict[str, Any]:
        """Convert batch of PIL images into model input tensors."""

    @abstractmethod
    def forward_impl(self, model_inputs: dict[str, Any], batch_pil: list[Image.Image]) -> Any:
        """Run forward pass for the configured run mode."""

    def _build_records(self) -> list[LayerIORecord]:
        records: list[LayerIORecord] = []

        for layer_name, state in self._hook_buffers.items():
            if not state["inputs"] and not state["outputs"]:
                continue

            module = state["module"]
            weight = module.weight.detach().to(device="cpu", dtype=torch.float32, copy=True)

            input_chunks = state["inputs"]
            output_chunks = state["outputs"]

            if input_chunks:
                inputs = torch.cat(input_chunks, dim=0)
            else:
                in_features = int(getattr(module, "in_features", 0))
                inputs = torch.empty((0, in_features), dtype=torch.float32)

            if output_chunks:
                outputs = torch.cat(output_chunks, dim=0)
            else:
                out_features = int(getattr(module, "out_features", 0))
                outputs = torch.empty((0, out_features), dtype=torch.float32)

            meta = {
                "input_shape_list": list(state["input_shape_list"]),
                "output_shape_list": list(state["output_shape_list"]),
                "num_calls": int(state["num_calls"]),
                "num_input_rows": int(inputs.shape[0]),
                "num_output_rows": int(outputs.shape[0]),
            }

            records.append(
                LayerIORecord(
                    model_name=self.name,
                    layer_name=layer_name,
                    weight=weight,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )
            )

        return records

    def _reset_buffers(self) -> None:
        for state in self._hook_buffers.values():
            state["inputs"] = []
            state["outputs"] = []
            state["input_shape_list"] = []
            state["output_shape_list"] = []
            state["num_calls"] = 0

    def _from_pretrained(self, loader_cls: Any, token: str | None, **kwargs: Any) -> Any:
        load_kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.revision:
            load_kwargs["revision"] = self.revision
        load_kwargs.update(kwargs)

        if token:
            try:
                return loader_cls.from_pretrained(self.hf_repo, token=token, **load_kwargs)
            except TypeError:
                try:
                    return loader_cls.from_pretrained(self.hf_repo, use_auth_token=token, **load_kwargs)
                except TypeError:
                    return loader_cls.from_pretrained(self.hf_repo, **load_kwargs)

        return loader_cls.from_pretrained(self.hf_repo, **load_kwargs)

    def _prepare_processor_payload(self, **kwargs: Any) -> dict[str, Any]:
        if self._processor is None:
            raise RuntimeError(f"Processor is unavailable for model '{self.name}'")

        payload = self._processor(return_tensors="pt", **kwargs)
        if not hasattr(payload, "items"):
            raise TypeError(f"Processor output is not mapping-like: {type(payload)}")
        return dict(payload.items())

    def _to_device_inputs(self, payload: dict[str, Any]) -> dict[str, Any]:
        converted: dict[str, Any] = {}
        for key, value in payload.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    converted[key] = value.to(device=self.device, dtype=self.run_dtype)
                else:
                    converted[key] = value.to(device=self.device)
            else:
                converted[key] = value
        return converted

    def _resolve_data_root(self) -> str:
        if "data" in self.global_cfg_dict:
            data_cfg = self.global_cfg_dict["data"]
            if isinstance(data_cfg, dict) and data_cfg.get("path"):
                return str(data_cfg["path"])

        return "./data"

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> torch.dtype:
        normalized = str(dtype_name).lower()
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _resolve_run_dtype(configured_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        if device.type == "cuda":
            return configured_dtype
        return torch.float32

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        normalized = str(device_name).lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")
