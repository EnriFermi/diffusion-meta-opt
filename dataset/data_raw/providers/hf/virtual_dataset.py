from __future__ import annotations

import logging
import multiprocessing as mp
import random
import time
from collections import deque
from pathlib import Path
from queue import Empty, Full
from typing import Any

from dataset.data_raw.core.base_virtual_dataset import BaseVirtualDataset
from dataset.data_raw.core.chunk_cache import ChunkCache
from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.core.fs_utils import ensure_dir, read_json, utc_now_iso, write_json_atomic
from dataset.data_raw.core.image_utils import decode_to_pil, load_image
from dataset.data_raw.core.types import ImageSample
from dataset.data_raw.providers.hf.auth import init_hf_auth
from dataset.data_raw.providers.hf.hf_loader import load_hf_dataset
from dataset.data_raw.providers.hf.url_fetch import fetch_image_to_cache


class HFVirtualDataset(BaseVirtualDataset):
    """HF-backed virtual dataset with one prefetch process and chunked disk cache."""

    def __init__(self, cfg: Any, global_data_root: str, seed: int, hf_cfg: Any) -> None:
        super().__init__(cfg=cfg, global_data_root=global_data_root, seed=seed)

        self.cfg_dict = to_plain_dict(cfg)
        self.hf_cfg = to_plain_dict({"hf": hf_cfg})["hf"]

        self.name = str(self.cfg_dict["name"])
        self.logger = logging.getLogger(f"dataset.{self.name}")

        self.dataset_root = ensure_dir(Path(global_data_root) / self.name)
        self.meta_path = self.dataset_root / "meta.json"

        cache_cfg = self.cfg_dict.get("cache", {})
        self.num_chunks_kept = int(cache_cfg.get("num_chunks_kept", 2))
        self.chunk_cache = ChunkCache(self.dataset_root, num_chunks_kept=self.num_chunks_kept)

        worker_cfg = self.cfg_dict.get("worker", {})
        self.get_timeout_s = float(worker_cfg.get("get_timeout_s", 5.0))
        self.startup_get_timeout_s = float(worker_cfg.get("startup_get_timeout_s", max(300.0, self.get_timeout_s * 20)))
        self.idle_sleep_s = float(worker_cfg.get("idle_sleep_s", 0.2))
        self.max_worker_restarts = int(worker_cfg.get("max_worker_restarts", 5))

        self._ctx = mp.get_context("spawn")
        self._events_queue: Any | None = None
        self._stop_event: Any | None = None
        self._worker: mp.Process | None = None
        self._worker_restarts = 0
        self._worker_permanently_stopped = False
        self._last_worker_error: str | None = None

        self._known_chunks: set[str] = set()
        self._chunk_queue: deque[dict[str, Any]] = deque()
        self._served_samples = 0

        self._init_meta_if_missing()
        self._load_existing_chunks()

    def start(self) -> None:
        self._spawn_worker(force=True)

    def get_batch(self, batch_size: int) -> list[ImageSample]:
        if batch_size <= 0:
            return []

        timeout_s = self.get_timeout_s
        # First chunk warmup can be slower (metadata/parquet/network); avoid false empty batches at startup.
        if self._served_samples == 0 and not self._chunk_queue:
            timeout_s = max(timeout_s, self.startup_get_timeout_s)

        deadline = time.monotonic() + timeout_s
        batch: list[ImageSample] = []

        while len(batch) < batch_size:
            self._drain_worker_events()
            sample = self._pop_sample()
            if sample is not None:
                batch.append(sample)
                continue

            self._ensure_worker_alive()
            if self._worker_permanently_stopped and not self._chunk_queue:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(self.idle_sleep_s)

        if len(batch) < batch_size:
            if self._last_worker_error:
                self.logger.warning(
                    "Requested %s samples, returned %s (cache underfilled). Last worker error: %s",
                    batch_size,
                    len(batch),
                    self._last_worker_error,
                )
            else:
                self.logger.warning(
                    "Requested %s samples, returned %s (cache currently underfilled)",
                    batch_size,
                    len(batch),
                )

        return batch

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=2)

        self._worker = None

        if self._events_queue is not None:
            try:
                self._events_queue.close()
            except Exception:
                pass
            self._events_queue = None

    def stats(self) -> dict[str, Any]:
        self._drain_worker_events()
        self._ensure_worker_alive()

        return {
            "dataset": self.name,
            "dataset_root": str(self.dataset_root),
            "chunks_on_disk": len(self.chunk_cache.list_chunks()),
            "chunks_loaded": len(self._chunk_queue),
            "images_on_disk": self.chunk_cache.count_images(),
            "worker_alive": bool(self._worker and self._worker.is_alive()),
            "worker_restarts": self._worker_restarts,
            "samples_served": self._served_samples,
        }

    def _spawn_worker(self, force: bool) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        if self._worker_restarts >= self.max_worker_restarts and not force:
            return

        if self._events_queue is not None:
            try:
                self._events_queue.close()
            except Exception:
                pass

        self._events_queue = self._ctx.Queue(maxsize=512)
        self._stop_event = self._ctx.Event()

        worker_payload = {
            "dataset_cfg": self.cfg_dict,
            "hf_cfg": self.hf_cfg,
            "global_data_root": str(self.global_data_root),
            "seed": int(self.seed),
        }

        self._worker = self._ctx.Process(
            target=hf_dataset_prefetch_worker,
            args=(worker_payload, self._events_queue, self._stop_event),
            daemon=True,
            name=f"{self.name}_prefetch",
        )
        self._worker.start()
        self._worker_permanently_stopped = False
        self.logger.info("Started worker pid=%s", self._worker.pid)

    def _ensure_worker_alive(self) -> None:
        if self._worker is None:
            self._spawn_worker(force=True)
            return

        if self._worker.is_alive():
            return

        if self._worker_restarts >= self.max_worker_restarts:
            if not self._worker_permanently_stopped:
                self.logger.error("Worker exited with code %s", self._worker.exitcode)
                self.logger.error("Max worker restarts reached: %s", self.max_worker_restarts)
            self._worker_permanently_stopped = True
            return

        self.logger.error("Worker exited with code %s", self._worker.exitcode)
        self._worker_restarts += 1
        self._spawn_worker(force=True)

    def _drain_worker_events(self) -> None:
        if self._events_queue is None:
            return

        while True:
            try:
                event = self._events_queue.get_nowait()
            except Empty:
                break

            event_type = event.get("type")
            if event_type == "chunk_ready":
                self._register_chunk(str(event["chunk_id"]))
            elif event_type == "chunk_evicted":
                self._forget_chunk(str(event["chunk_id"]))
            elif event_type == "error":
                message = str(event.get("message", "unknown worker error"))
                self._last_worker_error = message
                self.logger.error("Worker error: %s", message)
            else:
                self.logger.debug("Worker event: %s", event)

    def _register_chunk(self, chunk_id: str) -> None:
        if chunk_id in self._known_chunks:
            return

        try:
            manifest = self.chunk_cache.load_chunk(chunk_id)
        except FileNotFoundError:
            # Stale chunk event: chunk may have been evicted before we processed event.
            self.logger.debug("Skipping stale chunk event for missing chunk: %s", chunk_id)
            return
        except Exception as exc:
            self.logger.warning("Cannot load chunk %s: %s", chunk_id, exc)
            return

        items = manifest.get("items", [])
        if not items:
            return

        self._chunk_queue.append(
            {
                "chunk_id": chunk_id,
                "items": items,
                "cursor": 0,
            }
        )
        self._known_chunks.add(chunk_id)

    def _forget_chunk(self, chunk_id: str) -> None:
        self._known_chunks.discard(chunk_id)
        self._chunk_queue = deque(state for state in self._chunk_queue if state.get("chunk_id") != chunk_id)

    def _load_existing_chunks(self) -> None:
        for chunk_id in self.chunk_cache.list_chunks():
            self._register_chunk(chunk_id)

    def _pop_sample(self) -> ImageSample | None:
        while self._chunk_queue:
            state = self._chunk_queue[0]
            chunk_id = str(state["chunk_id"])
            items = state["items"]
            cursor = int(state["cursor"])

            if cursor >= len(items):
                self._consume_chunk(chunk_id)
                continue

            entry = items[cursor]
            state["cursor"] = cursor + 1

            file_name = entry.get("file")
            if not file_name:
                continue

            image_path = self.chunk_cache.chunk_path(chunk_id) / str(file_name)
            if not image_path.exists():
                continue

            try:
                image = load_image(image_path)
            except Exception as exc:
                self.logger.warning("Failed to load image %s: %s", image_path, exc)
                continue

            if state["cursor"] >= len(items):
                self._consume_chunk(chunk_id)

            sample_id = entry.get("sample_id", "unknown")
            meta = entry.get("meta", {})

            self._served_samples += 1
            return ImageSample(
                image=image,
                dataset_name=self.name,
                sample_id=sample_id,
                meta=meta if isinstance(meta, dict) else {},
            )

        return None

    def _consume_chunk(self, chunk_id: str) -> None:
        self._forget_chunk(chunk_id)
        self.chunk_cache.remove_chunk(chunk_id)

    def _init_meta_if_missing(self) -> None:
        if self.meta_path.exists():
            return

        dataset_size: int | None = None
        try:
            token = str(self.hf_cfg.get("token")) if self.hf_cfg.get("token") else None
            if not bool(self.cfg_dict.get("hf", {}).get("streaming", False)):
                dataset = load_hf_dataset(self.cfg_dict, token=token, seed=int(self.seed))
                dataset_size = len(dataset)  # type: ignore[arg-type]
        except Exception as exc:
            self.logger.warning("Could not infer dataset size for meta.json: %s", exc)

        hf_local = self.cfg_dict.get("hf", {})
        schema = self.cfg_dict.get("schema", {})

        payload = {
            "dataset_name": self.name,
            "hf_repo": hf_local.get("repo"),
            "hf_subset": hf_local.get("subset"),
            "hf_split": hf_local.get("split"),
            "schema": {
                "image_mode": schema.get("image_mode"),
                "image_field": schema.get("image_field"),
                "url_field": schema.get("url_field"),
                "extra_fields": schema.get("extra_fields", []),
            },
            "streaming": bool(hf_local.get("streaming", False)),
            "trust_remote_code": bool(hf_local.get("trust_remote_code", False)),
            "dataset_size": dataset_size,
            "gated": bool(self.cfg_dict.get("gated", False)),
            "licensing_note": self.cfg_dict.get("licensing_note", ""),
            "created_at": utc_now_iso(),
        }
        write_json_atomic(self.meta_path, payload)


def hf_dataset_prefetch_worker(worker_payload: dict[str, Any], events_queue: Any, stop_event: Any) -> None:
    dataset_cfg = to_plain_dict(worker_payload["dataset_cfg"])
    hf_cfg = to_plain_dict({"hf": worker_payload["hf_cfg"]})["hf"]

    dataset_name = str(dataset_cfg["name"])
    logger = logging.getLogger(f"dataset.worker.{dataset_name}")
    _configure_logging(dataset_cfg)

    top_cfg = {"hf": hf_cfg}
    allow_missing_token = not bool(dataset_cfg.get("gated", False))
    token = init_hf_auth(top_cfg, allow_missing_token=allow_missing_token)

    global_data_root = str(worker_payload["global_data_root"])
    seed = int(worker_payload["seed"])

    dataset_root = ensure_dir(Path(global_data_root) / dataset_name)
    cache_cfg = dataset_cfg.get("cache", {})
    chunk_size = int(cache_cfg.get("chunk_size_images", 1024))
    num_chunks_kept = int(cache_cfg.get("num_chunks_kept", 2))

    worker_cfg = dataset_cfg.get("worker", {})
    idle_sleep_s = float(worker_cfg.get("idle_sleep_s", 0.5))
    request_timeout_s = int(worker_cfg.get("request_timeout_s", 10))
    max_retries = int(worker_cfg.get("max_retries", 2))
    max_chunk_build_seconds = float(worker_cfg.get("max_chunk_build_seconds", 20.0))

    cache = ChunkCache(dataset_root=dataset_root, num_chunks_kept=num_chunks_kept)

    rng = random.Random(seed + int(time.time()))

    try:
        dataset = load_hf_dataset(dataset_cfg, token=token, seed=seed)
    except Exception as exc:
        message = _format_dataset_load_error(dataset_cfg=dataset_cfg, exc=exc)
        logger.exception("Dataset load failed for %s", dataset_name)
        _emit_event(events_queue, {"type": "error", "dataset": dataset_name, "message": message})
        return

    streaming = bool(dataset_cfg.get("hf", {}).get("streaming", False))
    dataset_size = _safe_len(dataset) if not streaming else None
    iterator = iter(dataset) if streaming else None
    stream_counter = 0

    logger.info("Worker started for dataset=%s streaming=%s", dataset_name, streaming)

    while not stop_event.is_set():
        try:
            evicted = cache.evict_old_chunks()
            for chunk_id in evicted:
                _emit_event(events_queue, {"type": "chunk_evicted", "chunk_id": chunk_id, "dataset": dataset_name})

            chunks_now = cache.list_chunks()
            if len(chunks_now) >= num_chunks_kept:
                time.sleep(idle_sleep_s)
                continue

            chunk_id = cache.start_chunk(_new_chunk_id(rng))
            records: list[dict[str, Any]] = []
            attempts = 0
            max_attempts = max(chunk_size * 10, chunk_size + 32)
            chunk_started_at = time.monotonic()

            while len(records) < chunk_size and not stop_event.is_set():
                attempts += 1
                if attempts > max_attempts:
                    logger.warning(
                        "Stopping chunk fill early after %s attempts (%s records collected)",
                        attempts,
                        len(records),
                    )
                    break

                if records and (time.monotonic() - chunk_started_at) >= max_chunk_build_seconds:
                    logger.debug(
                        "Finalizing partial chunk after %.1fs with %s records",
                        time.monotonic() - chunk_started_at,
                        len(records),
                    )
                    break

                record, sample_id, iterator, stream_counter = _next_record(
                    dataset=dataset,
                    dataset_cfg=dataset_cfg,
                    rng=rng,
                    dataset_size=dataset_size,
                    iterator=iterator,
                    stream_counter=stream_counter,
                    token=token,
                    seed=seed,
                )
                if record is None:
                    continue

                materialized = _materialize_image(
                    record=record,
                    sample_id=sample_id,
                    dataset_cfg=dataset_cfg,
                    cache=cache,
                    chunk_id=chunk_id,
                    timeout=request_timeout_s,
                    retries=max_retries,
                )
                if materialized is None:
                    continue

                image_path, item_meta = materialized
                records.append(
                    {
                        "sample_id": str(sample_id),
                        "file": Path(image_path).name,
                        "meta": item_meta,
                    }
                )

            if records:
                cache.finalize_chunk(chunk_id, records)
                _emit_event(events_queue, {"type": "chunk_ready", "chunk_id": chunk_id, "dataset": dataset_name})
            else:
                cache.remove_chunk(chunk_id)
                time.sleep(idle_sleep_s)
        except Exception as exc:
            logger.exception("Worker loop failure for dataset=%s", dataset_name)
            _emit_event(events_queue, {"type": "error", "dataset": dataset_name, "message": str(exc)})
            time.sleep(max(1.0, idle_sleep_s))


def _safe_len(dataset: Any) -> int | None:
    try:
        return len(dataset)  # type: ignore[arg-type]
    except Exception:
        return None


def _next_record(
    dataset: Any,
    dataset_cfg: dict[str, Any],
    rng: random.Random,
    dataset_size: int | None,
    iterator: Any,
    stream_counter: int,
    token: str | None,
    seed: int,
) -> tuple[dict[str, Any] | None, str | int, Any, int]:
    hf_cfg = dataset_cfg.get("hf", {})
    schema = dataset_cfg.get("schema", {})
    id_field = schema.get("id_field")

    if bool(hf_cfg.get("streaming", False)):
        if iterator is None:
            iterator = iter(dataset)
        try:
            record = next(iterator)
        except StopIteration:
            dataset = load_hf_dataset(dataset_cfg, token=token, seed=seed)
            iterator = iter(dataset)
            record = next(iterator)

        stream_counter += 1
        sample_id = _extract_sample_id(record, id_field=id_field, fallback=f"stream_{stream_counter}")
        return record, sample_id, iterator, stream_counter

    if not dataset_size:
        return None, "unknown", iterator, stream_counter

    random_index = rng.randrange(dataset_size)
    record = dataset[random_index]
    sample_id = _extract_sample_id(record, id_field=id_field, fallback=random_index)
    return record, sample_id, iterator, stream_counter


def _extract_sample_id(record: dict[str, Any], id_field: str | None, fallback: str | int) -> str | int:
    if id_field and id_field in record:
        return record[id_field]

    for candidate in ("id", "image_id", "sample_id", "key"):
        if candidate in record:
            return record[candidate]

    return fallback


def _materialize_image(
    record: dict[str, Any],
    sample_id: str | int,
    dataset_cfg: dict[str, Any],
    cache: ChunkCache,
    chunk_id: str,
    timeout: int,
    retries: int,
) -> tuple[Path, dict[str, Any]] | None:
    schema = dataset_cfg.get("schema", {})
    mode = str(schema.get("image_mode", "image_field"))

    image_path: Path | None = None

    if mode == "image_field":
        field = _resolve_image_field(record, schema)
        if field is None:
            return None
        image = decode_to_pil(record[field])
        image_path = cache.save_image(sample_id=sample_id, image=image, chunk_id=chunk_id)
    elif mode == "url_field":
        url = _resolve_url(record, schema)
        if not url:
            return None
        image_path = fetch_image_to_cache(
            url=url,
            cache=cache,
            sample_key=sample_id,
            timeout=timeout,
            retries=retries,
            chunk_id=chunk_id,
        )
    else:
        raise ValueError(f"Unsupported schema.image_mode: {mode}")

    if image_path is None:
        return None

    item_meta: dict[str, Any] = {}
    for extra_key in schema.get("extra_fields", []):
        if extra_key in record:
            item_meta[extra_key] = _safe_meta_value(record[extra_key])

    return image_path, item_meta


def _resolve_image_field(record: dict[str, Any], schema: dict[str, Any]) -> str | None:
    primary = schema.get("image_field")
    candidates = [primary, *schema.get("image_field_candidates", []), "image", "img"]

    for candidate in candidates:
        if candidate and candidate in record:
            return str(candidate)

    return None


def _resolve_url(record: dict[str, Any], schema: dict[str, Any]) -> str | None:
    primary = schema.get("url_field")
    candidates = [
        primary,
        *schema.get("url_field_candidates", []),
        "url",
        "image_url",
        "imageURL",
        "jpg",
    ]

    for candidate in candidates:
        if not candidate or candidate not in record:
            continue
        value = record[candidate]
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value

    return None


def _safe_meta_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_meta_value(item) for item in value[:32]]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in list(value.items())[:32]:
            compact[str(key)] = _safe_meta_value(item)
        return compact
    return str(value)


def _new_chunk_id(rng: random.Random) -> str:
    return f"chunk_{int(time.time() * 1000)}_{rng.randint(1000, 9999)}"


def _emit_event(queue_obj: Any, payload: dict[str, Any]) -> None:
    try:
        queue_obj.put_nowait(payload)
    except Full:
        pass


def _format_dataset_load_error(dataset_cfg: dict[str, Any], exc: Exception) -> str:
    if not bool(dataset_cfg.get("gated", False)):
        return str(exc)

    base = (
        f"{exc}. Set hf.token in conf/config.yaml. "
        "Accept the dataset license on Hugging Face dataset page. "
        "Ensure token has access (for fine-grained tokens enable access to public gated repositories)."
    )
    return base


def _configure_logging(dataset_cfg: dict[str, Any]) -> None:
    level_name = str(dataset_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
