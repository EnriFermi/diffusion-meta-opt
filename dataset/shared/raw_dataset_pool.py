from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

from omegaconf import DictConfig

from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.providers.hf import register_all_adapters
from dataset.data_raw.providers.hf.auth import init_hf_auth, validate_gated_datasets_token
from dataset.data_raw.registry import create_dataset
from dataset.shared.compatibility_index import CompatibilityIndex
from dataset.shared.types import MixedImageMeta


class RawDatasetPool:
    """Manage raw virtual datasets and provide mixed PIL sampling."""

    def __init__(self, cfg: DictConfig | dict[str, Any], index: CompatibilityIndex) -> None:
        self.cfg = cfg
        self.cfg_dict = to_plain_dict(cfg)
        self.index = index

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

        for dataset_name, ds_cfg in self.index.dataset_cfgs.items():
            if not bool(ds_cfg.get("enabled", True)):
                continue

            dataset = create_dataset(
                name=dataset_name,
                cfg=ds_cfg,
                global_root=self.data_root,
                seed=self.seed,
                hf_cfg=hf_cfg,
            )
            self.datasets[dataset_name] = dataset
            self.dataset_weights[dataset_name] = float(ds_cfg.get("sampling_weight", 1.0))

        if not self.datasets:
            raise ValueError("RawDatasetPool has no enabled datasets")

        self._started = False

    def start(self) -> None:
        if self._started:
            return

        for dataset_name, dataset in self.datasets.items():
            self.logger.info("Starting raw dataset worker: %s", dataset_name)
            dataset.start()

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
        mixing_policy: str,
        mix_cap_per_dataset: float,
    ) -> tuple[list[Any], list[MixedImageMeta]]:
        if batch_size <= 0:
            return [], []

        dataset_names = self.index.get_datasets_for_model(model_name)
        if not dataset_names:
            raise ValueError(f"No supporting datasets for model '{model_name}'")

        dataset_weights = self.index.get_dataset_weights_for_model(model_name)
        counts = _allocate_dataset_counts(
            dataset_names=dataset_names,
            dataset_weights=dataset_weights,
            batch_size=batch_size,
            mixing_policy=mixing_policy,
            cap_fraction=mix_cap_per_dataset,
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
    mixing_policy: str,
    cap_fraction: float,
    rng: random.Random,
) -> dict[str, int]:
    if len(dataset_names) == 1:
        return {dataset_names[0]: batch_size}

    cap_count = max(1, int(batch_size * float(cap_fraction)))
    weights = _normalize_weights(dataset_weights if mixing_policy == "multinomial" else [1.0] * len(dataset_names))

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
