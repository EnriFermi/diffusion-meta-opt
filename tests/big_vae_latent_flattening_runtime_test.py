from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from post_train_research.big_vae_latent_flattening.config import build_run_config
from post_train_research.big_vae_latent_flattening.debug import (
    _tensor_debug_stats,
    _top_named_tensor_debug_rows,
)
from post_train_research.big_vae_latent_flattening.train import configure_torch_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf" / "big_vae_latent_flattening"


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR.resolve())):
        return compose(config_name="config", overrides=overrides or [])


def test_latent_flattening_forces_math_attention_by_default() -> None:
    cfg = _compose(
        [
            "big_vae.checkpoint=/tmp/big_vae.pt",
            "data.offline_root=/tmp/offline_dataset",
        ]
    )

    run_cfg, _ = build_run_config(cfg)

    assert run_cfg.train.force_math_attention is True
    assert run_cfg.train.skip_nonfinite_updates is True
    assert run_cfg.train.max_consecutive_nonfinite_steps == 20
    assert run_cfg.train.nonfinite_debug_topk == 8


def test_latent_flattening_math_attention_runtime_switches_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose(
        [
            "big_vae.checkpoint=/tmp/big_vae.pt",
            "data.offline_root=/tmp/offline_dataset",
            "train.tf32=false",
            "train.force_math_attention=false",
        ]
    )
    run_cfg, _ = build_run_config(cfg)
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", lambda enabled: calls.append(("flash", enabled)))
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_mem_efficient_sdp",
        lambda enabled: calls.append(("mem_efficient", enabled)),
    )
    monkeypatch.setattr(torch.backends.cuda, "enable_math_sdp", lambda enabled: calls.append(("math", enabled)))

    configure_torch_runtime(run_cfg, torch.device("cuda:0"))

    assert calls == []


def test_latent_flattening_math_attention_runtime_switches_disable_efficient_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose(
        [
            "big_vae.checkpoint=/tmp/big_vae.pt",
            "data.offline_root=/tmp/offline_dataset",
            "train.tf32=false",
        ]
    )
    run_cfg, _ = build_run_config(cfg)
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", lambda enabled: calls.append(("flash", enabled)))
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_mem_efficient_sdp",
        lambda enabled: calls.append(("mem_efficient", enabled)),
    )
    monkeypatch.setattr(torch.backends.cuda, "enable_math_sdp", lambda enabled: calls.append(("math", enabled)))

    configure_torch_runtime(run_cfg, torch.device("cuda:0"))

    assert calls == [("flash", False), ("mem_efficient", False), ("math", True)]


def test_latent_flattening_tensor_debug_stats_count_nonfinite_values() -> None:
    tensor = torch.tensor([0.0, float("nan"), float("inf"), -float("inf"), 3.0])

    stats = _tensor_debug_stats("x", tensor)

    assert stats["numel"] == 5
    assert stats["finite_count"] == 2
    assert stats["nan_count"] == 1
    assert stats["posinf_count"] == 1
    assert stats["neginf_count"] == 1
    assert stats["all_finite"] is False
    assert stats["first_nonfinite_flat_indices"] == [1, 2, 3]


def test_latent_flattening_named_tensor_debug_rows_prioritize_nonfinite() -> None:
    rows = _top_named_tensor_debug_rows(
        iter(
            [
                ("finite_small", torch.tensor([1.0])),
                ("finite_large", torch.tensor([100.0])),
                ("bad", torch.tensor([float("nan")])),
            ]
        ),
        topk=2,
    )

    assert [row["name"] for row in rows] == ["bad", "finite_large"]
    assert rows[0]["bad_count"] == 1
