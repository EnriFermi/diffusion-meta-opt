from __future__ import annotations

import unittest

from dataset.data_raw.providers.hf.auth import HF_TOKEN_MISSING_ERROR, validate_gated_datasets_token


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


if __name__ == "__main__":
    unittest.main()
