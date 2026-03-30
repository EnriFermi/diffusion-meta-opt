from __future__ import annotations

import logging
import random
import traceback
from collections import Counter
from typing import Any

from omegaconf import DictConfig

from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.core.types import ImageSampleRef
from dataset.data_raw.providers.hf import register_all_adapters
from dataset.data_raw.providers.hf.auth import (
    gated_datasets_missing_token,
    init_hf_auth,
    validate_gated_datasets_token,
)
from dataset.data_raw.registry import create_dataset
from dataset.shared.compatibility_index import CompatibilityIndex
from .load_report import LoadReportWriter
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
        collector_cfg = self.cfg_dict.get("collector") or {}
        dataset_pool_cfg = collector_cfg.get("dataset_pool") or {}
        self.continue_on_create_failure = bool(dataset_pool_cfg.get("continue_on_create_failure", True))
        self.continue_on_start_failure = bool(dataset_pool_cfg.get("continue_on_start_failure", True))
        self.disable_dataset_on_runtime_failure = bool(dataset_pool_cfg.get("disable_dataset_on_runtime_failure", True))
        self.continue_on_missing_hf_token_for_gated_datasets = bool(
            dataset_pool_cfg.get("continue_on_missing_hf_token_for_gated_datasets", True)
        )
        self.max_runtime_failures_before_disable = max(
            1,
            int(dataset_pool_cfg.get("max_runtime_failures_before_disable", 3)),
        )

        self.datasets: dict[str, Any] = {}
        self.dataset_weights: dict[str, float] = {}
        self._disabled_datasets: set[str] = set()
        self._dataset_failure_counts: Counter[str] = Counter()
        self._dataset_last_failure: dict[str, str] = {}

        register_all_adapters()

        missing_token_datasets = gated_datasets_missing_token(self.cfg_dict, self.index.dataset_cfgs)
        if missing_token_datasets:
            if self.continue_on_missing_hf_token_for_gated_datasets:
                reason = (
                    "HF token missing for gated dataset. Configure hf.token and accept HF gated licenses, "
                    "or keep dataset disabled."
                )
                self.logger.error(
                    "HF token missing. Disabling gated datasets and continuing: %s",
                    ", ".join(missing_token_datasets),
                )
                for dataset_name in missing_token_datasets:
                    self._disabled_datasets.add(str(dataset_name))
                    self._dataset_last_failure[str(dataset_name)] = reason
                    if self.load_report is not None:
                        self.load_report.mark_dataset_failed(
                            dataset_name=str(dataset_name),
                            phase="token_validation",
                            error=reason,
                            traceback_text=None,
                        )
            else:
                validate_gated_datasets_token(self.cfg_dict, self.index.dataset_cfgs)

        hf_token = init_hf_auth(self.cfg_dict, allow_missing_token=not self._requires_hf_token())

        hf_cfg = to_plain_dict(self.cfg_dict.get("hf", {}))
        hf_cfg["token"] = hf_token

        runtime_logging = to_plain_dict(self.cfg_dict.get("logging", {}))
        training_artifacts_cfg = to_plain_dict(self.cfg_dict.get("training_artifacts", {}))
        train_cfg = to_plain_dict(self.cfg_dict.get("train", {}))
        if not train_cfg:
            train_cfg = to_plain_dict(self.cfg_dict.get("mini_train", {}))
        collector_cfg = to_plain_dict(self.cfg_dict.get("collector", {}))

        for dataset_name, ds_cfg in self.index.dataset_cfgs.items():
            dataset_key = str(dataset_name)
            if self._is_dataset_disabled(dataset_key):
                self.logger.warning("Skipping disabled dataset during create phase: %s", dataset_key)
                continue
            if not bool(ds_cfg.get("enabled", True)):
                continue

            ds_cfg_runtime = to_plain_dict(ds_cfg)
            ds_cfg_runtime["runtime_logging"] = runtime_logging
            ds_cfg_runtime["runtime_forensics"] = {
                "training_artifacts": training_artifacts_cfg,
                "train_forensics": to_plain_dict(train_cfg.get("forensics", {})),
                "collector_diagnostics": to_plain_dict(collector_cfg.get("diagnostics", {})),
            }

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
                if self.continue_on_create_failure:
                    self.logger.error("Disabling dataset '%s' after create failure: %s", dataset_name, exc)
                    self._disabled_datasets.add(dataset_key)
                    self._dataset_last_failure[dataset_key] = str(exc)
                    continue
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
            raise ValueError("RawDatasetPool has no enabled datasets after filtering failures")

        self._started = False

    def start(self) -> None:
        if self._started:
            return

        for dataset_name, dataset in self.datasets.items():
            if self._is_dataset_disabled(dataset_name):
                self.logger.warning("Skipping raw dataset worker start for disabled dataset: %s", dataset_name)
                continue
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
                if self.continue_on_start_failure:
                    self._disable_dataset(
                        dataset_name,
                        reason=f"start_failure: {exc}",
                    )
                    continue
                raise

        self._ensure_has_active_datasets()

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

        if self._is_dataset_disabled(dataset_name):
            return [], [], []

        dataset = self.datasets.get(dataset_name)
        if dataset is None:
            self._mark_runtime_failure(dataset_name, reason="unknown dataset")
            return [], [], []

        try:
            samples = dataset.get_batch(n)
        except Exception as exc:
            self._mark_runtime_failure(
                dataset_name,
                reason=f"get_batch_error: {exc}",
            )
            return [], [], []
        if not samples:
            try:
                stats = dataset.stats()
            except Exception as exc:
                self._mark_runtime_failure(
                    dataset_name,
                    reason=f"empty_batch_and_stats_error: {exc}",
                )
                return [], [], []
            if bool(stats.get("worker_permanently_stopped", False)):
                reason = str(stats.get("worker_last_error") or "worker_permanently_stopped")
                self._disable_dataset(str(dataset_name), reason=reason)
                self._ensure_has_active_datasets()
            return [], [], []
        pil_images = [sample.image for sample in samples]
        source_ids = [sample.sample_id for sample in samples]
        metas = [sample.meta for sample in samples]
        if pil_images:
            self._dataset_failure_counts.pop(str(dataset_name), None)
            self._dataset_last_failure.pop(str(dataset_name), None)
        return pil_images, source_ids, metas

    def get_image_ref_batch(
        self,
        dataset_name: str,
        n: int,
    ) -> tuple[list[ImageSampleRef], list[str | int], list[dict[str, Any]]]:
        if n <= 0:
            return [], [], []

        if self._is_dataset_disabled(dataset_name):
            return [], [], []

        dataset = self.datasets.get(dataset_name)
        if dataset is None:
            self._mark_runtime_failure(dataset_name, reason="unknown dataset")
            return [], [], []

        try:
            refs = dataset.get_batch_refs(n)
        except NotImplementedError:
            self._mark_runtime_failure(
                dataset_name,
                reason="get_batch_refs_not_supported",
            )
            return [], [], []
        except Exception as exc:
            self._mark_runtime_failure(
                dataset_name,
                reason=f"get_batch_refs_error: {exc}",
            )
            return [], [], []

        if not refs:
            try:
                stats = dataset.stats()
            except Exception as exc:
                self._mark_runtime_failure(
                    dataset_name,
                    reason=f"empty_ref_batch_and_stats_error: {exc}",
                )
                return [], [], []
            if bool(stats.get("worker_permanently_stopped", False)):
                reason = str(stats.get("worker_last_error") or "worker_permanently_stopped")
                self._disable_dataset(str(dataset_name), reason=reason)
                self._ensure_has_active_datasets()
            return [], [], []

        source_ids = [ref.sample_id for ref in refs]
        metas = [ref.meta if isinstance(ref.meta, dict) else {} for ref in refs]
        self._dataset_failure_counts.pop(str(dataset_name), None)
        self._dataset_last_failure.pop(str(dataset_name), None)
        return refs, source_ids, metas

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
            self.logger.warning("No supporting datasets configured for model '%s'", model_name)
            return [], []

        raw_dataset_weights = self.index.get_dataset_weights_for_model(model_name)
        active_pairs = [
            (str(name), float(weight))
            for name, weight in zip(dataset_names, raw_dataset_weights, strict=False)
            if not self._is_dataset_disabled(str(name))
        ]
        if not active_pairs:
            self.logger.warning(
                "No active datasets left for model '%s' (configured=%s, disabled=%s)",
                model_name,
                dataset_names,
                sorted(self._disabled_datasets),
            )
            self._ensure_has_active_datasets()
            return [], []
        dataset_names = [name for name, _ in active_pairs]
        dataset_weights = [weight for _, weight in active_pairs]
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

    def sample_mixed_batch_refs(
        self,
        model_name: str,
        batch_size: int,
        dataset_sampling_strategy: str,
        max_dataset_fraction_per_batch: float,
    ) -> tuple[list[ImageSampleRef], list[MixedImageMeta]]:
        if batch_size <= 0:
            return [], []

        dataset_names = self.index.get_datasets_for_model(model_name)
        if not dataset_names:
            self.logger.warning("No supporting datasets configured for model '%s'", model_name)
            return [], []

        raw_dataset_weights = self.index.get_dataset_weights_for_model(model_name)
        active_pairs = [
            (str(name), float(weight))
            for name, weight in zip(dataset_names, raw_dataset_weights, strict=False)
            if not self._is_dataset_disabled(str(name))
        ]
        if not active_pairs:
            self.logger.warning(
                "No active datasets left for model '%s' (configured=%s, disabled=%s)",
                model_name,
                dataset_names,
                sorted(self._disabled_datasets),
            )
            self._ensure_has_active_datasets()
            return [], []
        dataset_names = [name for name, _ in active_pairs]
        dataset_weights = [weight for _, weight in active_pairs]
        effective_max_fraction = _normalize_max_dataset_fraction_per_batch(
            requested=float(max_dataset_fraction_per_batch),
            num_datasets=len(dataset_names),
        )
        counts = _allocate_dataset_counts(
            dataset_names=dataset_names,
            dataset_weights=dataset_weights,
            batch_size=batch_size,
            dataset_sampling_strategy=dataset_sampling_strategy,
            max_dataset_fraction=effective_max_fraction,
            rng=self._rng,
        )

        all_refs: list[ImageSampleRef] = []
        all_meta: list[MixedImageMeta] = []

        for dataset_name, count in counts.items():
            if count <= 0:
                continue
            image_refs, source_ids, _ = self.get_image_ref_batch(dataset_name=dataset_name, n=count)
            for image_ref, source_id in zip(image_refs, source_ids, strict=False):
                all_refs.append(image_ref)
                all_meta.append(MixedImageMeta(dataset_name=dataset_name, source_id=source_id))

        if len(all_refs) < batch_size:
            self.logger.warning(
                "Mixed ref batch underfilled for model=%s: requested=%s, got=%s",
                model_name,
                batch_size,
                len(all_refs),
            )
            self._fill_batch_fallback_refs(
                batch_size=batch_size,
                dataset_names=dataset_names,
                all_refs=all_refs,
                all_meta=all_meta,
            )

        indices = list(range(len(all_refs)))
        self._rng.shuffle(indices)

        shuffled_refs = [all_refs[i] for i in indices[:batch_size]]
        shuffled_meta = [all_meta[i] for i in indices[:batch_size]]
        return shuffled_refs, shuffled_meta

    def materialize_image_refs(
        self,
        image_refs: list[ImageSampleRef],
        image_meta: list[MixedImageMeta],
    ) -> tuple[list[Any], list[MixedImageMeta]]:
        pil_batch: list[Any] = []
        filtered_meta: list[MixedImageMeta] = []

        for image_ref, meta in zip(image_refs, image_meta, strict=False):
            dataset_name = str(image_ref.dataset_name)
            if self._is_dataset_disabled(dataset_name):
                continue
            dataset = self.datasets.get(dataset_name)
            if dataset is None:
                self._mark_runtime_failure(dataset_name, reason="unknown dataset during ref materialization")
                continue

            try:
                sample = dataset.load_ref(image_ref)
            except Exception as exc:
                self._mark_runtime_failure(dataset_name, reason=f"load_ref_error: {exc}")
                continue
            if sample is None:
                continue
            pil_batch.append(sample.image)
            filtered_meta.append(meta)

        return pil_batch, filtered_meta

    def release_image_refs(self, image_refs: list[ImageSampleRef]) -> None:
        refs_by_dataset: dict[str, list[ImageSampleRef]] = {}
        for ref in image_refs:
            refs_by_dataset.setdefault(str(ref.dataset_name), []).append(ref)

        for dataset_name, refs in refs_by_dataset.items():
            dataset = self.datasets.get(dataset_name)
            if dataset is None:
                continue
            try:
                dataset.release_refs(refs)
            except Exception as exc:
                self.logger.warning("Failed to release image refs for dataset '%s': %s", dataset_name, exc)

    def stats(self) -> dict[str, Any]:
        payload = {}
        for dataset_name, dataset in self.datasets.items():
            if self._is_dataset_disabled(dataset_name):
                payload[dataset_name] = {
                    "dataset": str(dataset_name),
                    "disabled": True,
                    "disable_reason": self._dataset_last_failure.get(str(dataset_name)),
                    "worker_alive": False,
                    "worker_pid": None,
                    "worker_exitcode": None,
                    "worker_restarts": int(self._dataset_failure_counts.get(str(dataset_name), 0)),
                    "worker_permanently_stopped": True,
                    "worker_last_error": self._dataset_last_failure.get(str(dataset_name)),
                    "chunks_on_disk": 0,
                    "chunks_loaded": 0,
                    "samples_served": 0,
                }
                continue
            payload[dataset_name] = dataset.stats()
        return payload

    def worker_processes(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for dataset_name, dataset in self.datasets.items():
            dataset_key = str(dataset_name)
            if self._is_dataset_disabled(dataset_key):
                payload[dataset_key] = {
                    "dataset": dataset_key,
                    "disabled": True,
                    "disable_reason": self._dataset_last_failure.get(dataset_key),
                    "worker_pid": None,
                    "worker_alive": False,
                    "worker_exitcode": None,
                    "worker_restarts": int(self._dataset_failure_counts.get(dataset_key, 0)),
                    "worker_permanently_stopped": True,
                    "worker_last_error": self._dataset_last_failure.get(dataset_key),
                    "chunks_on_disk": 0,
                    "chunks_loaded": 0,
                    "samples_served": 0,
                    "samples_per_second": 0.0,
                    "chunk_ready_rate_per_second": 0.0,
                    "records_materialized": 0,
                    "worker_uptime_s": 0.0,
                }
                continue
            try:
                stats = dataset.stats()
            except Exception as exc:
                self._mark_runtime_failure(dataset_key, reason=f"stats_error: {exc}")
                payload[dataset_key] = {
                    "dataset": dataset_key,
                    "disabled": self._is_dataset_disabled(dataset_key),
                    "disable_reason": self._dataset_last_failure.get(dataset_key),
                    "worker_pid": None,
                    "worker_alive": False,
                    "worker_exitcode": None,
                    "worker_restarts": int(self._dataset_failure_counts.get(dataset_key, 0)),
                    "worker_permanently_stopped": False,
                    "worker_last_error": f"stats_error: {exc}",
                    "samples_per_second": 0.0,
                    "chunk_ready_rate_per_second": 0.0,
                    "records_materialized": 0,
                    "worker_uptime_s": 0.0,
                }
                continue

            payload[dataset_key] = {
                "dataset": dataset_key,
                "disabled": False,
                "disable_reason": None,
                "worker_pid": stats.get("worker_pid"),
                "worker_alive": bool(stats.get("worker_alive", False)),
                "worker_exitcode": stats.get("worker_exitcode"),
                "worker_restarts": int(stats.get("worker_restarts", 0) or 0),
                "worker_permanently_stopped": bool(stats.get("worker_permanently_stopped", False)),
                "worker_last_error": stats.get("worker_last_error"),
                "chunks_on_disk": int(stats.get("chunks_on_disk", 0) or 0),
                "chunks_loaded": int(stats.get("chunks_loaded", 0) or 0),
                "samples_served": int(stats.get("samples_served", 0) or 0),
                "samples_per_second": float(stats.get("samples_per_second", 0.0) or 0.0),
                "chunk_ready_rate_per_second": float(stats.get("chunk_ready_rate_per_second", 0.0) or 0.0),
                "records_materialized": int(stats.get("records_materialized", 0) or 0),
                "worker_uptime_s": float(stats.get("worker_uptime_s", 0.0) or 0.0),
            }
        return payload

    def _requires_hf_token(self) -> bool:
        for dataset_name, cfg in self.index.dataset_cfgs.items():
            name_key = str(dataset_name)
            if name_key in self._disabled_datasets:
                continue
            if not bool(cfg.get("enabled", True)):
                continue
            if bool(cfg.get("gated", False)):
                return True
        return False

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

    def _fill_batch_fallback_refs(
        self,
        batch_size: int,
        dataset_names: list[str],
        all_refs: list[ImageSampleRef],
        all_meta: list[MixedImageMeta],
    ) -> None:
        names = dataset_names[:]
        while len(all_refs) < batch_size and names:
            self._rng.shuffle(names)
            progress = False
            for dataset_name in names:
                image_refs, source_ids, _ = self.get_image_ref_batch(dataset_name=dataset_name, n=1)
                if not image_refs:
                    continue
                all_refs.append(image_refs[0])
                all_meta.append(MixedImageMeta(dataset_name=dataset_name, source_id=source_ids[0]))
                progress = True
                if len(all_refs) >= batch_size:
                    break
            if not progress:
                break

    def _mark_runtime_failure(self, dataset_name: str, *, reason: str) -> None:
        key = str(dataset_name)
        self._dataset_failure_counts[key] += 1
        self._dataset_last_failure[key] = str(reason)
        failures = int(self._dataset_failure_counts[key])
        self.logger.warning(
            "Dataset '%s' runtime failure (%s/%s): %s",
            key,
            failures,
            self.max_runtime_failures_before_disable,
            reason,
        )
        if self.disable_dataset_on_runtime_failure and failures >= self.max_runtime_failures_before_disable:
            self._disable_dataset(key, reason=reason)

    def _disable_dataset(self, dataset_name: str, *, reason: str) -> None:
        key = str(dataset_name)
        if key in self._disabled_datasets:
            return
        self._disabled_datasets.add(key)
        self._dataset_last_failure[key] = str(reason)
        self.logger.error("Disabled dataset '%s': %s", key, reason)
        dataset = self.datasets.get(key)
        if dataset is not None:
            try:
                dataset.close()
            except Exception:
                pass

    def _is_dataset_disabled(self, dataset_name: str) -> bool:
        return str(dataset_name) in self._disabled_datasets

    def _ensure_has_active_datasets(self) -> None:
        active = [name for name in self.datasets if name not in self._disabled_datasets]
        if not active:
            raise RuntimeError(
                "RawDatasetPool has no active datasets left after failures; cannot continue collection"
            )


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
