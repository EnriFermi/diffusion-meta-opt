from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest
import torch
from hydra import compose, initialize_config_dir

from post_train_research.tinyvit_latent_h1.config import build_run_config
from post_train_research.tinyvit_latent_h1.experiment import _checkpoint_payload_json_view
from post_train_research.tinyvit_latent_h1 import source as source_mod
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


def test_random_bootstrap_allows_empty_source_run_dir() -> None:
    cfg = _compose(
        [
            "source.bootstrap_kind=random",
            "source.run_dir=",
        ]
    )
    run_cfg, _ = build_run_config(cfg)
    assert run_cfg.source.bootstrap_kind == "random"
    assert run_cfg.source.run_dir == ""


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


def test_levelset_search_returns_payload_at_returned_hi(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.value = torch.zeros(1)
            self.latent_slots = {"slot": self.value}

        def materialize_latent_slot(self, key: str) -> torch.Tensor:
            assert key == "slot"
            return self.value

        def materialized_latent_slots_state_dict(self) -> dict[str, torch.Tensor]:
            return {"slot": self.value.detach().clone()}

        def load_materialized_latent_slots_state_dict(
            self,
            state: dict[str, torch.Tensor],
            *,
            strict: bool = True,
            update_radii: bool = True,
        ) -> None:
            del strict, update_radii
            self.value = state["slot"].detach().clone()
            self.latent_slots["slot"] = self.value

    store = FakeStore()
    model = SimpleNamespace(store=store)
    source = SimpleNamespace(all_tensor_names=["slot"], z_star_state={"slot": torch.zeros(1)})

    def fake_evaluate_train_and_test(*args, **kwargs):
        del args, kwargs
        loss = abs(float(store.value.item()))
        metrics = source_mod.EvalMetrics(loss=loss, accuracy=0.0, examples=1)
        return metrics, metrics

    def fake_export_named_tensors(*args, **kwargs):
        del args, kwargs
        return {"slot": store.value.detach().clone()}

    monkeypatch.setattr(source_mod, "BigVAELatentTensorStore", FakeStore)
    monkeypatch.setattr(source_mod, "evaluate_train_and_test", fake_evaluate_train_and_test)
    monkeypatch.setattr(source_mod, "export_named_tensors", fake_export_named_tensors)

    result = source_mod._find_levelset_alpha(
        model,
        source,
        train_loader=None,
        test_loader=None,
        base_flat=torch.zeros(1),
        base_loss=0.0,
        direction=torch.ones(1),
        epsilon=0.8,
        device=torch.device("cpu"),
        amp_enabled=False,
        alpha_min=1.0,
        alpha_max=1.0,
        bracket_multiplier=2.0,
        binary_search_steps=2,
    )

    assert result is not None
    alpha, latent_state, train_metrics, _test_metrics, named_tensors = result
    assert alpha == 1.0
    assert latent_state["slot"].item() == pytest.approx(alpha)
    assert train_metrics.loss == pytest.approx(abs(alpha))
    assert named_tensors["slot"].item() == pytest.approx(alpha)


def test_checkpoint_payload_json_view_strips_tensor_values() -> None:
    payload = {
        "branch": "latent",
        "named_tensors": {"head.weight": torch.zeros(10, 64)},
        "latent_slots": {"p0001_t0000": torch.zeros(32, 256)},
        "tile_cond_patch": {"p0001_t0000": torch.zeros(4, 32)},
    }
    view = _checkpoint_payload_json_view(payload)

    assert view["named_tensor_count"] == 1
    assert view["latent_slot_count"] == 1
    assert view["tile_cond_patch_count"] == 1
    assert view["named_tensors"]["head.weight"]["shape"] == [10, 64]
    assert view["latent_slots"]["p0001_t0000"]["shape"] == [32, 256]
    assert view["tile_cond_patch"]["p0001_t0000"]["shape"] == [4, 32]


def test_anchor_training_requires_positive_steps_and_lr() -> None:
    cfg = _compose(
        [
            "source.bootstrap_kind=random",
            "source.train_anchor=true",
            "source.anchor_steps=0",
            "source.anchor_lr=0.0",
        ]
    )
    with pytest.raises(ValueError):
        build_run_config(cfg)
