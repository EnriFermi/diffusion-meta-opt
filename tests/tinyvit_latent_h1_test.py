from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch
from hydra import compose, initialize_config_dir

from post_train_research.tinyvit_latent_h1.config import build_run_config
from post_train_research.tinyvit_latent_h1.experiment import _checkpoint_payload_json_view
from post_train_research.tinyvit_latent_h1.source import build_train_schedule, resolve_source_checkpoint_for_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf" / "tinyvit_latent_h1"


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR.resolve())):
        return compose(config_name="config", overrides=overrides or [])


def test_config_builds_with_required_source_run_dir() -> None:
    cfg = _compose(
        [
            "experiment.run_label=h1_smoke",
            "source.run_dir=my_source_run",
            "train.raw_lrs=[0.001,0.0003]",
            "train.latent_lrs=[0.01,0.003]",
        ]
    )
    run_cfg, _ = build_run_config(cfg)

    assert run_cfg.run_label == "h1_smoke"
    assert run_cfg.source.run_dir == "my_source_run"
    assert run_cfg.model.depth == 3
    assert run_cfg.search.epsilons == [0.05, 0.1]


def test_missing_source_run_dir_is_rejected() -> None:
    cfg = _compose(["source.run_dir="])
    with pytest.raises(ValueError):
        build_run_config(cfg)


def test_resolve_source_checkpoint_prefers_shared_checkpoint_store() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        shared_dir = root / "checkpoints" / "shared_model"
        shared_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = shared_dir / "best.pt"
        checkpoint_path.write_bytes(b"x")
        cfg = _compose(
            [
                f"source.vit_scaling_artifacts_root={root}",
                "source.run_dir=shared_model",
                "source.checkpoint_name=best",
            ]
        )
        run_cfg, _ = build_run_config(cfg)
        resolved = resolve_source_checkpoint_for_experiment(run_cfg)
        assert resolved == checkpoint_path.resolve()


def test_build_train_schedule_is_deterministic() -> None:
    left = build_train_schedule(10, batch_size=4, steps=5, seed=123)
    right = build_train_schedule(10, batch_size=4, steps=5, seed=123)
    assert len(left) == 5
    assert all(l.equal(r) for l, r in zip(left, right, strict=True))


def test_checkpoint_payload_json_view_strips_tensor_values() -> None:
    payload = {
        "branch": "latent",
        "named_tensors": {"head.weight": torch.zeros(10, 64)},
        "latent_slots": {"p0001_t0000": torch.zeros(32, 256)},
    }
    view = _checkpoint_payload_json_view(payload)

    assert view["named_tensor_count"] == 1
    assert view["latent_slot_count"] == 1
    assert view["named_tensors"]["head.weight"]["shape"] == [10, 64]
    assert view["latent_slots"]["p0001_t0000"]["shape"] == [32, 256]
