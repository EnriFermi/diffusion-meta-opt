from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from post_train_research.big_vae_latent_flattening.config import build_run_config
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
