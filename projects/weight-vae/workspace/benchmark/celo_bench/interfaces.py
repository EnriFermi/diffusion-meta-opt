from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


class TaskAdapter(Protocol):
    name: str

    def build(self) -> Any:
        """Return a learned_optimization-compatible task."""


class OptimizerAdapter(Protocol):
    name: str

    def build(self, *, num_steps: int) -> Any:
        """Return a learned_optimization-compatible optimizer."""


@dataclass(frozen=True, slots=True)
class CurveRunSpec:
    task_name: str
    method_name: str
    seed: int
    steps: int
    output_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CurveResult:
    spec: CurveRunSpec
    metrics: Mapping[str, Any]


@dataclass(slots=True)
class BenchTables:
    output_dir: Path
    summary: dict[str, Any]
    score_rows: list[dict[str, Any]]
    curve_rows: list[dict[str, Any]]
    selected_adam_rows: list[dict[str, Any]]
    skipped_rows: list[dict[str, Any]] = field(default_factory=list)
