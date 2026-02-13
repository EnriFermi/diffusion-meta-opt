from __future__ import annotations

import unittest

import torch

from dataset.data_raw.providers.hf.auth import HF_TOKEN_MISSING_ERROR, validate_gated_datasets_token
from dataset.models.providers.transformers.hf_base_runner import HFBaseRunner


class _DummyHFRunner(HFBaseRunner):
    def _load_model(self, token: str | None):
        return torch.nn.Linear(4, 4)

    def _load_processor(self, token: str | None):
        return object()

    def prepare_inputs(self, batch_pil):
        return {}

    def forward_impl(self, model_inputs, batch_pil):
        return None


class TestHFTokenRequirements(unittest.TestCase):
    def test_missing_token_with_enabled_gated_dataset_fails(self) -> None:
        cfg = {
            "hf": {"token": None},
        }
        active_cfgs = {
            "mapillary_vistas_v2": {"enabled": True, "gated": True},
        }

        with self.assertRaisesRegex(ValueError, HF_TOKEN_MISSING_ERROR):
            validate_gated_datasets_token(cfg, active_cfgs)

    def test_missing_token_with_disabled_gated_dataset_passes(self) -> None:
        cfg = {
            "hf": {"token": None},
        }
        active_cfgs = {
            "mapillary_vistas_v2": {"enabled": False, "gated": True},
        }

        validate_gated_datasets_token(cfg, active_cfgs)

    def test_missing_token_with_non_gated_dataset_passes(self) -> None:
        cfg = {
            "hf": {"token": None},
        }
        active_cfgs = {
            "flickr30k": {"enabled": True, "gated": False},
        }

        validate_gated_datasets_token(cfg, active_cfgs)

    def test_token_present_with_enabled_gated_dataset_passes(self) -> None:
        cfg = {
            "hf": {"token": "hf_xxx"},
        }
        active_cfgs = {
            "relaion400m": {"enabled": True, "gated": True},
        }

        validate_gated_datasets_token(cfg, active_cfgs)

    def test_placeholder_token_for_enabled_gated_dataset_fails(self) -> None:
        cfg = {
            "hf": {"token": "YOUR_TOKEN_HERE"},
        }
        active_cfgs = {
            "mapillary_vistas_v2": {"enabled": True, "gated": True},
        }

        with self.assertRaisesRegex(ValueError, HF_TOKEN_MISSING_ERROR):
            validate_gated_datasets_token(cfg, active_cfgs)

    def test_missing_token_with_enabled_gated_dataset_fails_for_streaming_profile(self) -> None:
        cfg = {
            "hf": {"token": None},
            "streaming": {"mode": "local_disk"},
        }
        active_cfgs = {
            "mapillary_vistas_v2": {"enabled": True, "gated": True},
        }

        with self.assertRaisesRegex(ValueError, HF_TOKEN_MISSING_ERROR):
            validate_gated_datasets_token(cfg, active_cfgs)

    def test_missing_token_with_gated_model_fails(self) -> None:
        global_cfg = {
            "hf": {"token": None},
            "data": {"path": "./data"},
            "streaming": {"mode": "s3_bridge"},
        }
        model_cfg = {
            "name": "dummy_gated_model",
            "hf_repo": "org/private-model",
            "gated": True,
            "cache_subdir": "models/dummy_gated_model",
            "device": "cpu",
            "dtype": "float32",
            "run_mode": "vision_only",
            "limits": {},
            "hook_filter": {},
        }

        runner = _DummyHFRunner(cfg=model_cfg, global_cfg=global_cfg)
        with self.assertRaisesRegex(ValueError, HF_TOKEN_MISSING_ERROR):
            runner.load()


if __name__ == "__main__":
    unittest.main()
