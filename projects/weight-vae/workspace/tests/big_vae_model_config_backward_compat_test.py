from __future__ import annotations

from training.big_vae.model_config import build_big_vae_model_config


def test_missing_rope_coordinate_kind_uses_legacy_raw_coordinates() -> None:
    cfg = build_big_vae_model_config({"model": {"big_vae": {}}})

    assert cfg.big_vae.rope_2d_coord_kind == "raw"


def test_explicit_normalized_rope_coordinate_kind_is_preserved() -> None:
    cfg = build_big_vae_model_config(
        {
            "model": {
                "big_vae": {
                    "rope_2d_coord_kind": "normalized_center",
                }
            }
        }
    )

    assert cfg.big_vae.rope_2d_coord_kind == "normalized_center"
