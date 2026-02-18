from __future__ import annotations

import logging
import random
import traceback
from collections import Counter
from typing import Any

from omegaconf import DictConfig

from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.providers.hf import register_all_adapters
from dataset.data_raw.providers.hf.auth import init_hf_auth, validate_gated_datasets_token
from dataset.data_raw.registry import create_dataset
from dataset.shared.compatibility_index import CompatibilityIndex
from dataset.shared.load_report import LoadReportWriter
from dataset.shared.types import MixedImageMeta


class RawDatasetPool:
    """Manage raw virtual datasets and provide mixed PIL sampling."""

    def __init__(
        self,
        cfg: DictConfig | dict[str, Any],
        index: CompatibilityIndex,
        load_report: LoadReportWriter | None = None,
    ) -> None:
        self.cfg = cfg
        self.cfg_dict = to_plain_dict(cfg)
        self.index = index
        self.load_report = load_report

        self.logger = logging.getLogger(self.__class__.__name__)

        data_cfg = self.cfg_dict.get("data") or {}
        self.data_root = str(data_cfg.get("path", "./data"))
        self.seed = int(data_cfg.get("seed", 0))
        self._rng = random.Random(self.seed)

        register_all_adapters()

        validate_gated_datasets_token(self.cfg_dict, self.index.dataset_cfgs)
        hf_token = init_hf_auth(self.cfg_dict, allow_missing_token=not self._requires_hf_token())

        hf_cfg = to_plain_dict(self.cfg_dict.get("hf", {}))
        hf_cfg["token"] = hf_token

        self.datasets: dict[str, Any] = {}
        self.dataset_weights: dict[str, float] = {}
        runtime_logging = to_plain_dict(self.cfg_dict.get("logging", {}))

        for dataset_name, ds_cfg in self.index.dataset_cfgs.items():
            if not bool(ds_cfg.get("enabled", True)):
                continue

            ds_cfg_runtime = to_plain_dict(ds_cfg)
            ds_cfg_runtime["runtime_logging"] = runtime_logging

            try:
                dataset = create_dataset(
                    name=dataset_name,
                    cfg=ds_cfg_runtime,
                    global_root=self.data_root,
                    seed=self.seed,
                    hf_cfg=hf_cfg,
                )
            except Exception as exc:
                if self.load_report is not None:
                    self.load_report.mark_dataset_failed(
                        dataset_name=dataset_name,
                        phase="create",
                        error=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                raise

            self.datasets[dataset_name] = dataset
            self.dataset_weights[dataset_name] = float(ds_cfg_runtime.get("sampling_weight", 1.0))
            if self.load_report is not None:
                self.load_report.mark_dataset_loaded(
                    dataset_name=dataset_name,
                    phase="create",
                    details={
                        "sampling_weight": float(ds_cfg_runtime.get("sampling_weight", 1.0)),
                        "collector_device": ds_cfg_runtime.get("collector_device"),
                        "models": list(ds_cfg_runtime.get("models", [])),
                    },
                )

        if not self.datasets:
            raise ValueError("RawDatasetPool has no enabled datasets")

        self._started = False

    def start(self) -> None:
        if self._started:
            return

        for dataset_name, dataset in self.datasets.items():
            self.logger.info("Starting raw dataset worker: %s", dataset_name)
            try:
                dataset.start()
                if self.load_report is not None:
                    self.load_report.mark_dataset_loaded(dataset_name=dataset_name, phase="start")
            except Exception as exc:
                if self.load_report is not None:
                    self.load_report.mark_dataset_failed(
                        dataset_name=dataset_name,
                        phase="start",
                        error=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                raise

        self._started = True

    def shutdown(self) -> None:
        if not self._started:
            return

        for dataset_name, dataset in self.datasets.items():
            self.logger.info("Shutting down raw dataset worker: %s", dataset_name)
            dataset.close()

        self._started = False

    def get_pil_batch(self, dataset_name: str, n: int) -> tuple[list[Any], list[str | int], list[dict[str, Any]]]:
        if n <= 0:
            return [], [], []

        dataset = self.datasets.get(dataset_name)
        if dataset is None:
            raise KeyError(f"Unknown dataset '{dataset_name}'")

        samples = dataset.get_batch(n)
        pil_images = [sample.image for sample in samples]
        source_ids = [sample.sample_id for sample in samples]
        metas = [sample.meta for sample in samples]
        return pil_images, source_ids, metas

    def sample_mixed_batch(
        self,
        model_name: str,
        batch_size: int,
        dataset_sampling_strategy: str,
        max_dataset_fraction_per_batch: float,
    ) -> tuple[list[Any], list[MixedImageMeta]]:
        if batch_size <= 0:
            return [], []

        dataset_names = self.index.get_datasets_for_model(model_name)
        if not dataset_names:
            raise ValueError(f"No supporting datasets for model '{model_name}'")

        dataset_weights = self.index.get_dataset_weights_for_model(model_name)
        effective_max_fraction = _normalize_max_dataset_fraction_per_batch(
            requested=float(max_dataset_fraction_per_batch),
            num_datasets=len(dataset_names),
        )
        if effective_max_fraction > float(max_dataset_fraction_per_batch):
            self.logger.warning(
                "Adjusted collector.max_dataset_fraction_per_batch for model=%s: requested=%.6f, effective=%.6f, "
                "num_datasets=%s (need cap * n_dataset >= 1.0)",
                model_name,
                float(max_dataset_fraction_per_batch),
                effective_max_fraction,
                len(dataset_names),
            )
        counts = _allocate_dataset_counts(
            dataset_names=dataset_names,
            dataset_weights=dataset_weights,
            batch_size=batch_size,
            dataset_sampling_strategy=dataset_sampling_strategy,
            max_dataset_fraction=effective_max_fraction,
            rng=self._rng,
        )

        all_images: list[Any] = []
        all_meta: list[MixedImageMeta] = []

        for dataset_name, count in counts.items():
            if count <= 0:
                continue

            pil_images, source_ids, _ = self.get_pil_batch(dataset_name=dataset_name, n=count)
            for image, source_id in zip(pil_images, source_ids, strict=False):
                all_images.append(image)
                all_meta.append(MixedImageMeta(dataset_name=dataset_name, source_id=source_id))

        if len(all_images) < batch_size:
            self.logger.warning(
                "Mixed batch underfilled for model=%s: requested=%s, got=%s",
                model_name,
                batch_size,
                len(all_images),
            )
            self._fill_batch_fallback(batch_size=batch_size, dataset_names=dataset_names, all_images=all_images, all_meta=all_meta)

        indices = list(range(len(all_images)))
        self._rng.shuffle(indices)

        shuffled_images = [all_images[i] for i in indices[:batch_size]]
        shuffled_meta = [all_meta[i] for i in indices[:batch_size]]

        return shuffled_images, shuffled_meta

    def stats(self) -> dict[str, Any]:
        payload = {}
        for dataset_name, dataset in self.datasets.items():
            payload[dataset_name] = dataset.stats()
        return payload

    def _requires_hf_token(self) -> bool:
        return any(bool(cfg.get("gated", False)) for cfg in self.index.dataset_cfgs.values())

    def _fill_batch_fallback(
        self,
        batch_size: int,
        dataset_names: list[str],
        all_images: list[Any],
        all_meta: list[MixedImageMeta],
    ) -> None:
        names = dataset_names[:]
        while len(all_images) < batch_size and names:
            self._rng.shuffle(names)
            progress = False
            for dataset_name in names:
                pil_images, source_ids, _ = self.get_pil_batch(dataset_name=dataset_name, n=1)
                if not pil_images:
                    continue
                all_images.append(pil_images[0])
                all_meta.append(MixedImageMeta(dataset_name=dataset_name, source_id=source_ids[0]))
                progress = True
                if len(all_images) >= batch_size:
                    break
            if not progress:
                break


def _allocate_dataset_counts(
    dataset_names: list[str],
    dataset_weights: list[float],
    batch_size: int,
    dataset_sampling_strategy: str,
    max_dataset_fraction: float,
    rng: random.Random,
) -> dict[str, int]:
    if len(dataset_names) == 1:
        return {dataset_names[0]: batch_size}

    cap_count = max(1, int(batch_size * max(0.0, min(1.0, float(max_dataset_fraction)))))
    normalized_strategy = str(dataset_sampling_strategy).lower()
    if normalized_strategy in {"weighted_random", "multinomial", "weighted"}:
        weights = _normalize_weights(dataset_weights)
    elif normalized_strategy in {"uniform_random", "uniform"}:
        weights = _normalize_weights([1.0] * len(dataset_names))
    else:
        raise ValueError(
            f"Unsupported collector.dataset_sampling_strategy='{dataset_sampling_strategy}'. "
            "Use one of: uniform_random, weighted_random"
        )

    counts: Counter[str] = Counter()

    for _ in range(batch_size):
        candidates: list[str] = []
        candidate_weights: list[float] = []

        for dataset_name, prob in zip(dataset_names, weights, strict=False):
            if counts[dataset_name] < cap_count:
                candidates.append(dataset_name)
                candidate_weights.append(prob)

        if not candidates:
            candidates = dataset_names
            candidate_weights = weights

        chosen = rng.choices(candidates, weights=candidate_weights, k=1)[0]
        counts[chosen] += 1

    return {name: counts[name] for name in dataset_names}


def _normalize_weights(values: list[float]) -> list[float]:
    safe = [max(0.0, float(item)) for item in values]
    total = sum(safe)
    if total <= 0:
        return [1.0 / len(values) for _ in values]
    return [item / total for item in safe]


def _normalize_max_dataset_fraction_per_batch(requested: float, num_datasets: int, eps: float = 1e-6) -> float:
    n = max(1, int(num_datasets))
    clipped = max(0.0, min(1.0, float(requested)))
    if n <= 1:
        return 1.0
    min_required = (1.0 / float(n)) + float(eps)
    return min(1.0, max(clipped, min_required))
