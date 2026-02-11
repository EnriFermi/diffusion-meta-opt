from __future__ import annotations

import random
from collections import deque
from typing import Any


class ModelScheduler:
    """Model-first scheduler with burst control."""

    def __init__(
        self,
        model_names: list[str],
        model_weights: dict[str, float] | None,
        policy: str,
        burst_jobs: int,
        seed: int,
    ) -> None:
        if not model_names:
            raise ValueError("No collectable models were provided to ModelScheduler")

        self.model_names = [str(name) for name in model_names]
        self.model_weights = {name: float((model_weights or {}).get(name, 1.0)) for name in self.model_names}
        self.policy = str(policy).lower()
        self.burst_jobs = max(1, int(burst_jobs))

        self._rng = random.Random(seed)
        self._cycle: deque[str] = deque()
        self._current_model: str | None = None
        self._burst_remaining = 0

    def next_model(self) -> str:
        if self._burst_remaining > 0 and self._current_model is not None:
            self._burst_remaining -= 1
            return self._current_model

        if self.policy == "shuffled_cycle":
            model = self._next_shuffled_cycle()
        elif self.policy == "weighted":
            model = self._next_weighted()
        else:
            raise ValueError(f"Unsupported collector.model_policy='{self.policy}'")

        self._current_model = model
        self._burst_remaining = self.burst_jobs - 1
        return model

    def _next_shuffled_cycle(self) -> str:
        if not self._cycle:
            items = self.model_names[:]
            self._rng.shuffle(items)
            self._cycle.extend(items)
        return self._cycle.popleft()

    def _next_weighted(self) -> str:
        candidates = self.model_names[:]
        weights = [self.model_weights.get(name, 1.0) for name in candidates]

        if len(candidates) > 1 and self._current_model in candidates:
            # Avoid long runs of the same model after burst boundary.
            filtered_candidates: list[str] = []
            filtered_weights: list[float] = []
            for name, weight in zip(candidates, weights, strict=False):
                if name == self._current_model:
                    continue
                filtered_candidates.append(name)
                filtered_weights.append(weight)
            if filtered_candidates:
                candidates = filtered_candidates
                weights = filtered_weights

        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def stats(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "burst_jobs": self.burst_jobs,
            "current_model": self._current_model,
            "burst_remaining": self._burst_remaining,
            "num_models": len(self.model_names),
        }
