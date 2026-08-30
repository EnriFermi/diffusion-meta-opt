from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://datasets-server.huggingface.co"


class DatasetServerSampler:
    """Sample rows from HF datasets-server without full split materialization."""

    def __init__(
        self,
        repo: str,
        split: str,
        subset: str | None,
        token: str | None,
        seed: int,
        timeout_s: float = 15.0,
        max_retries: int = 3,
        base_url: str = _DEFAULT_BASE_URL,
        first_rows_cache_ttl_s: float = 300.0,
        rows_retry_cooldown_s: float = 30.0,
        rows_unstable_log_interval_s: float = 300.0,
    ) -> None:
        self.repo = str(repo)
        self.split = str(split)
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.first_rows_cache_ttl_s = max(1.0, float(first_rows_cache_ttl_s))
        self.rows_retry_cooldown_s = max(1.0, float(rows_retry_cooldown_s))
        self.rows_unstable_log_interval_s = max(1.0, float(rows_unstable_log_interval_s))

        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

        self.config = str(subset) if subset else self._resolve_config()
        self.num_examples = self._resolve_num_examples()

        rng = random.Random(seed)
        self._cursor = rng.randrange(self.num_examples) if self.num_examples and self.num_examples > 0 else 0
        self._step = _coprime_step(rng=rng, modulus=self.num_examples)
        self._first_rows_cache: list[tuple[dict[str, Any], str | int]] = []
        self._first_rows_cache_loaded_at = 0.0
        self._first_rows_cursor = 0
        self._prefer_first_rows_until = 0.0
        self._last_rows_unstable_log_at = 0.0
        self._refresh_first_rows_cache(force=True)

    def next_record(self) -> tuple[dict[str, Any], str | int]:
        now = time.monotonic()
        if now < self._prefer_first_rows_until:
            cached = self._next_from_first_rows_cache(refresh_if_stale=True)
            if cached is not None:
                return cached

        last_error: Exception | None = None

        for _ in range(max(1, self.max_retries)):
            offset = self._next_offset()
            try:
                payload = self._request_json(
                    path="/rows",
                    params={
                        "dataset": self.repo,
                        "config": self.config,
                        "split": self.split,
                        "offset": offset,
                        "length": 1,
                    },
                )
                rows = payload.get("rows") or []
                if not rows:
                    continue
                row_payload = rows[0]
                row = row_payload.get("row") or {}
                sample_id = row_payload.get("row_idx", offset)
                if not isinstance(row, dict):
                    continue
                return row, sample_id
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break

        cached = self._next_from_first_rows_cache(force_refresh=True)
        if cached is not None:
            self._prefer_first_rows_until = time.monotonic() + self.rows_retry_cooldown_s
            if (time.monotonic() - self._last_rows_unstable_log_at) >= self.rows_unstable_log_interval_s:
                LOGGER.warning(
                    "datasets-server /rows is unstable for repo=%s (split=%s, config=%s). "
                    "Temporarily falling back to refreshed /first-rows cache for %.1fs.",
                    self.repo,
                    self.split,
                    self.config,
                    self.rows_retry_cooldown_s,
                )
                self._last_rows_unstable_log_at = time.monotonic()
            return cached

        if last_error is not None:
            raise RuntimeError(
                f"datasets-server row fetch failed for repo={self.repo}, split={self.split}, config={self.config}: "
                f"{last_error}"
            ) from last_error
        raise RuntimeError(
            f"datasets-server row fetch returned no rows for repo={self.repo}, split={self.split}, config={self.config}"
        )

    def _resolve_config(self) -> str:
        payload = self._request_json(path="/splits", params={"dataset": self.repo})
        items = payload.get("splits") or []
        if not items:
            raise RuntimeError(f"No splits returned by datasets-server for dataset={self.repo}")

        for item in items:
            if str(item.get("split")) == self.split and item.get("config"):
                return str(item["config"])

        fallback = items[0].get("config")
        if not fallback:
            raise RuntimeError(f"Cannot resolve config for dataset={self.repo}")
        LOGGER.warning(
            "datasets-server: split '%s' was not found for repo=%s; using config '%s'",
            self.split,
            self.repo,
            fallback,
        )
        return str(fallback)

    def _resolve_num_examples(self) -> int | None:
        payload = self._request_json(path="/info", params={"dataset": self.repo, "config": self.config})
        dataset_info = payload.get("dataset_info") or {}
        splits = dataset_info.get("splits") or {}
        split_info = splits.get(self.split) or {}
        raw = split_info.get("num_examples")
        if raw is None:
            return None
        try:
            value = int(raw)
        except Exception:
            return None
        return value if value > 0 else None

    def _next_offset(self) -> int:
        if self.num_examples and self.num_examples > 0:
            current = self._cursor % self.num_examples
            self._cursor = (self._cursor + self._step) % self.num_examples
            return int(current)

        current = self._cursor
        self._cursor += 1
        return int(current)

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f"{self.base_url}{path}"

        for _ in range(max(1, self.max_retries)):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout_s)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                raise RuntimeError(f"datasets-server returned non-dict payload: {type(payload)}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"datasets-server request failed: {url} params={params} error={last_error}") from last_error
        raise RuntimeError(f"datasets-server request failed: {url} params={params}")

    def _load_first_rows_cache(self) -> list[tuple[dict[str, Any], str | int]]:
        try:
            payload = self._request_json(
                path="/first-rows",
                params={
                    "dataset": self.repo,
                    "config": self.config,
                    "split": self.split,
                },
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "datasets-server first-rows preload failed for repo=%s split=%s config=%s: %s",
                self.repo,
                self.split,
                self.config,
                exc,
            )
            return []

        rows = payload.get("rows") or []
        cache: list[tuple[dict[str, Any], str | int]] = []
        for row_payload in rows:
            row = row_payload.get("row")
            if not isinstance(row, dict):
                continue
            sample_id = row_payload.get("row_idx", len(cache))
            cache.append((row, sample_id))
        return cache

    def _refresh_first_rows_cache(self, *, force: bool) -> None:
        now = time.monotonic()
        if (
            (not force)
            and self._first_rows_cache
            and (now - self._first_rows_cache_loaded_at) < self.first_rows_cache_ttl_s
        ):
            return

        cache = self._load_first_rows_cache()
        if cache or not self._first_rows_cache:
            self._first_rows_cache = cache
            self._first_rows_cursor = 0
        self._first_rows_cache_loaded_at = now

    def _next_from_first_rows_cache(
        self,
        *,
        refresh_if_stale: bool = False,
        force_refresh: bool = False,
    ) -> tuple[dict[str, Any], str | int] | None:
        if force_refresh:
            self._refresh_first_rows_cache(force=True)
        elif refresh_if_stale:
            self._refresh_first_rows_cache(force=False)

        if not self._first_rows_cache:
            return None
        item = self._first_rows_cache[self._first_rows_cursor % len(self._first_rows_cache)]
        self._first_rows_cursor += 1
        return item


def _coprime_step(rng: random.Random, modulus: int | None) -> int:
    if modulus is None or modulus <= 1:
        return 1

    for _ in range(32):
        candidate = rng.randint(1, modulus - 1)
        if math.gcd(candidate, modulus) == 1:
            return candidate

    return 1
