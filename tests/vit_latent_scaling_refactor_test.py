from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch
from hydra import compose, initialize_config_dir

from post_train_research.vit_latent_scaling.config import build_run_config
from post_train_research.vit_latent_scaling.init import adapt_named_tensors, resolve_source_checkpoint_path
from post_train_research.vit_latent_scaling.train import build_lr_scheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf" / "vit_latent_scaling"


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR.resolve())):
        return compose(config_name="config", overrides=overrides or [])


def test_single_config_builds_cifar10_50k_raw_profile() -> None:
    cfg = _compose(
        [
            "experiment.profile=cifar10_50k",
            "setup.kind=raw",
            "init.kind=fresh",
            "data.download=true",
        ]
    )
    run_cfg, _ = build_run_config(cfg)

    assert run_cfg.profile.name == "cifar10_50k"
    assert run_cfg.data.dataset == "cifar10"
    assert run_cfg.model.hidden_dim == 64
    assert run_cfg.model.depth == 1
    assert run_cfg.data.batch_size == 256
    assert run_cfg.train.max_steps == 3000


def test_source_init_requires_source_run_dir() -> None:
    cfg = _compose(
        [
            "setup.kind=latent",
            "setup.big_vae_checkpoint=/tmp/big_vae.pt",
            "init.kind=source",
            "init.source_run_dir=",
        ]
    )
    with pytest.raises(ValueError):
        build_run_config(cfg)


def test_adapt_named_tensors_resizes_patch_embed_and_pos_embed() -> None:
    source = {
        "patch_embed.weight": torch.arange(64 * 3 * 4 * 4, dtype=torch.float32).view(64, 3, 4, 4),
        "pos_embed": torch.randn(1, 17, 32),
    }
    target = {
        "patch_embed.weight": torch.zeros(64, 3, 8, 8, dtype=torch.float32),
        "pos_embed": torch.zeros(1, 65, 64, dtype=torch.float32),
    }
    adapted, report = adapt_named_tensors(source, target)

    assert adapted["patch_embed.weight"].shape == (64, 3, 8, 8)
    assert adapted["pos_embed"].shape == (1, 65, 64)
    assert report["reshaped"] == 2


def test_resolve_source_checkpoint_path_accepts_run_dir_and_checkpoint_alias() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run_dir = root / "runs" / "abc123"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "best.pt").write_bytes(b"x")
        resolved = resolve_source_checkpoint_path(root, source_run_dir="abc123", checkpoint_name="best")
        assert resolved == (checkpoints_dir / "best.pt").resolve()


def test_latent_scheduler_decays_to_floor() -> None:
    cfg = _compose(
        [
            "setup.kind=latent",
            "setup.big_vae_checkpoint=/tmp/big_vae.pt",
            "init.kind=fresh",
            "train.lr=0.01",
            "train.latent_lr_scheduler=cosine_decay_to_floor",
            "train.latent_lr_floor_ratio=0.2",
            "train.latent_lr_decay_steps=5",
        ]
    )
    run_cfg, _ = build_run_config(cfg)
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=run_cfg.train.lr)
    scheduler, description = build_lr_scheduler(optimizer, run_cfg, planned_steps=10)

    assert scheduler is not None
    assert "cosine_decay_to_floor" in description
    observed = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        observed.append(float(optimizer.param_groups[0]["lr"]))
    assert abs(observed[0] - 0.01) < 1e-12
    assert abs(observed[5] - 0.002) < 1e-12
    assert abs(observed[-1] - 0.002) < 1e-12
