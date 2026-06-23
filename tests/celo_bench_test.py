from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.celo_bench.adapters import AdamWOptimizerAdapter, ImportPathOptimizerAdapter, build_optimizer_adapter
from benchmark.celo_bench.artifacts import reserve_run_dir
from benchmark.celo_bench.config import (
    CELO_PAPER_17_TASKS,
    AdamReferenceConfig,
    BenchmarkConfig,
    CeloBenchConfig,
    EvaluationConfig,
    OptimizerSpec,
    config_hash,
)
from benchmark.celo_bench.metrics import (
    ema,
    final_loss_score,
    interquartile_mean,
    optimality_gap,
    speedup_score,
    summarize_scores,
)
from benchmark.celo_bench.registry import TASK_REGISTRY, TaskRegistry, TaskSpec, validate_task_imports
from benchmark.celo_bench import runner


class FakeOptimizer:
    pass


def fake_optimizer_factory(**_: Any) -> FakeOptimizer:
    return FakeOptimizer()


def test_registry_contains_exact_paper_tasks() -> None:
    assert TASK_REGISTRY.names() == CELO_PAPER_17_TASKS


def test_registry_rejects_duplicate_task_names() -> None:
    registry = TaskRegistry()
    registry.register(TaskSpec(name="task", import_path="math.sqrt"))
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(TaskSpec(name="task", import_path="math.sqrt"))


def test_import_path_optimizer_adapter_loads_factory() -> None:
    adapter = ImportPathOptimizerAdapter(
        name="fake",
        import_path="tests.celo_bench_test:fake_optimizer_factory",
    )
    assert isinstance(adapter.build(num_steps=10), FakeOptimizer)


def test_adamw_optimizer_adapter_parses_metadata_without_importing_optional_env() -> None:
    adapter = build_optimizer_adapter(
        OptimizerSpec(
            name="adamw",
            kind="adamw",
            metadata={"lr": 1e-3, "weight_decay": 1e-4, "b1": 0.8, "b2": 0.9, "eps": 1e-7},
        )
    )
    assert isinstance(adapter, AdamWOptimizerAdapter)
    assert adapter.lr == pytest.approx(1e-3)
    assert adapter.weight_decay == pytest.approx(1e-4)
    assert adapter.b1 == pytest.approx(0.8)
    assert adapter.b2 == pytest.approx(0.9)
    assert adapter.eps == pytest.approx(1e-7)


def test_metrics_match_celo_scoring_rules() -> None:
    np.testing.assert_allclose(ema([10.0, 0.0, 0.0], 0.9), [10.0, 9.0, 8.1])
    assert final_loss_score(2.0, 4.0) == 0.5
    assert final_loss_score(float("nan"), 4.0) == 0.0
    assert speedup_score([0, 10, 20], [5.0, 3.0, 1.0], target_loss=2.0, horizon=20, alpha=0.0) == 1.0
    assert interquartile_mean([0.0, 1.0, 2.0, 100.0]) == 1.5
    assert optimality_gap([0.0, 0.5, 1.5]) == pytest.approx(0.5)
    summary = summarize_scores([0.0, 1.0, 2.0, 100.0])
    assert summary["median"] == 1.5
    assert summary["iqm"] == 1.5


def test_config_hash_changes_for_benchmark_relevant_fields(tmp_path: Path) -> None:
    cfg = CeloBenchConfig(benchmark=BenchmarkConfig(run_dir=str(tmp_path / "run")), task_names=("FakeTask",))
    changed = CeloBenchConfig(
        benchmark=BenchmarkConfig(run_dir=str(tmp_path / "other")),
        task_names=("FakeTask",),
        evaluation=EvaluationConfig(steps=123),
    )
    same_results = CeloBenchConfig(benchmark=BenchmarkConfig(run_dir=str(tmp_path / "other")), task_names=("FakeTask",))
    assert config_hash(cfg) == config_hash(same_results)
    assert config_hash(cfg) != config_hash(changed)


def test_runtime_defaults_hide_tensorflow_gpus() -> None:
    assert CeloBenchConfig().runtime.tensorflow_hide_gpus is True


def test_adamw_full_config_is_full_benchmark() -> None:
    import yaml

    path = Path("conf/celo_bench/adamw_full.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = CeloBenchConfig.from_mapping(payload)
    assert cfg.task_names == CELO_PAPER_17_TASKS
    assert cfg.evaluation.steps == 2000
    assert cfg.evaluation.seeds == (0, 1, 2)
    assert cfg.methods[0].kind == "adamw"


def test_reserve_run_dir_collision_gets_suffix(tmp_path: Path) -> None:
    first_id, _ = reserve_run_dir(tmp_path, "smoke", timestamp="20260101_000000")
    second_id, _ = reserve_run_dir(tmp_path, "smoke", timestamp="20260101_000000")
    assert first_id == "20260101_000000__smoke"
    assert second_id == "20260101_000000__smoke_01"


def test_run_or_load_writes_artifacts_and_uses_matching_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_eval(*, task_name: str, optimizer_adapter: Any, seed: int, cfg: CeloBenchConfig) -> dict[str, np.ndarray]:
        calls.append((task_name, optimizer_adapter.name, seed))
        if optimizer_adapter.name == "adam_lr_0.1":
            loss = np.asarray([4.0, 3.0, 2.0])
        elif optimizer_adapter.name == "adam_lr_0.01":
            loss = np.asarray([4.0, 3.5, 3.0])
        else:
            loss = np.asarray([4.0, 2.5, 1.0])
        return {
            "eval/xs": np.asarray([0, 10, 20]),
            "eval/train/loss": loss,
            "eval/outer_valid/loss": loss + 0.1,
            "eval/test/loss": loss + 0.2,
        }

    monkeypatch.setattr(runner, "evaluate_task_optimizer", fake_eval)
    cfg = CeloBenchConfig(
        benchmark=BenchmarkConfig(run_dir=str(tmp_path / "run"), show_progress=False),
        task_names=("FakeTask",),
        evaluation=EvaluationConfig(steps=20, seeds=(0,), eval_every=10, eval_batches=1, last_eval_batches=1),
        adam_reference=AdamReferenceConfig(enabled=True, lrs=(0.1, 0.01)),
        methods=(
            OptimizerSpec(
                name="fake_method",
                kind="import_path",
                import_path="tests.celo_bench_test:fake_optimizer_factory",
            ),
        ),
    )
    tables = runner.run_or_load(cfg)
    assert (tables.output_dir / "summary.json").is_file()
    assert (tables.output_dir / "config_resolved.json").is_file()
    assert (tables.output_dir / "artifact_layout.json").is_file()
    assert tables.summary["final_loss_score"]["count"] == 1.0
    assert len(calls) == 3

    def fail_eval(**_: Any) -> dict[str, np.ndarray]:
        raise AssertionError("cache should avoid rerunning curves")

    monkeypatch.setattr(runner, "evaluate_task_optimizer", fail_eval)
    cached = runner.run_or_load(cfg)
    assert cached.summary["config_hash"] == tables.summary["config_hash"]

    changed = CeloBenchConfig(
        benchmark=BenchmarkConfig(run_dir=str(tmp_path / "run"), show_progress=False),
        task_names=("FakeTask",),
        evaluation=EvaluationConfig(steps=21, seeds=(0,), eval_every=10, eval_batches=1, last_eval_batches=1),
        adam_reference=AdamReferenceConfig(enabled=True, lrs=(0.1, 0.01)),
    )
    with pytest.raises(AssertionError, match="cache should avoid"):
        runner.run_or_load(changed)


def test_real_task_imports_when_optional_celo_env_is_available() -> None:
    errors = validate_task_imports()
    if errors:
        pytest.skip(f"optional Celo benchmark environment is incomplete: {errors}")
    assert not errors
