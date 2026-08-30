"""Celo/VeLOdrome benchmark harness."""

from .config import (
    ADAM_LR_GRID,
    CELO_PAPER_17_TASKS,
    AdamReferenceConfig,
    BenchmarkConfig,
    CeloBenchConfig,
    EvaluationConfig,
    OptimizerSpec,
    RuntimeConfig,
    config_hash,
)
from .interfaces import BenchTables

__all__ = [
    "ADAM_LR_GRID",
    "CELO_PAPER_17_TASKS",
    "AdamReferenceConfig",
    "BenchmarkConfig",
    "BenchTables",
    "CeloBenchConfig",
    "EvaluationConfig",
    "OptimizerSpec",
    "RuntimeConfig",
    "config_hash",
    "run_all",
    "run_or_load",
]


def run_all(*args, **kwargs):
    from .runner import run_all as _run_all

    return _run_all(*args, **kwargs)


def run_or_load(*args, **kwargs):
    from .runner import run_or_load as _run_or_load

    return _run_or_load(*args, **kwargs)
