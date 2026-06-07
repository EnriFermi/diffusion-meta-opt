from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from big_vae.runtime.artifacts import write_artifact_layout

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_train_research.big_vae_heldout_eval.common import (
    HELDOUT_DATASET_MODELS,
    apply_heldout_collector_defaults_from_env,
    apply_heldout_log_dir,
    allowed_pairs,
    apply_builder_defaults_from_env,
    apply_heldout_data_profile,
    apply_offline_dataset_defaults,
    counter_to_rows,
    default_heldout_root,
    env_bool,
    env_float,
    env_int,
    env_path,
    coverage_report,
    primary_dataset_name,
    promote_run_profile_to_root,
    resolved_cfg_snapshot,
    sanitize_programmatic_hydra_logging,
    sample_pair,
    validate_heldout_preflight,
    write_csv,
    write_json,
)
from dataset import data_pipeline
from big_vae.datasets.offline import (
    BigVAEOfflineDatasetWriter,
    resolve_big_vae_curriculum_targets,
    resolve_offline_target_size_bytes,
)
from dataset.logging_utils import configure_process_logging


LOGGER = logging.getLogger("build_big_vae_heldout_offline_dataset")


def _compact_collector_snapshot(dataset: Any, collector: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if dataset is not None and hasattr(dataset, "debug_snapshot"):
        try:
            payload["dataset"] = dataset.debug_snapshot(preview=3)
        except Exception as exc:
            payload["dataset_error"] = repr(exc)
    if collector is not None and hasattr(collector, "stats"):
        try:
            stats = collector.stats()
            payload["collector"] = {
                "mode": stats.get("mode"),
                "streaming_mode": stats.get("streaming_mode"),
                "async_process_alive": stats.get("async_process_alive"),
                "cache_size": stats.get("cache_size"),
                "jobs_total": stats.get("jobs_total"),
                "items_emitted": stats.get("items_emitted"),
                "jobs_by_model": stats.get("jobs_by_model"),
                "sink": stats.get("sink"),
                "scheduler": stats.get("scheduler"),
            }
        except Exception as exc:
            payload["collector_error"] = repr(exc)
    return payload


def _format_missing_pairs(
    pair_counts: Counter[tuple[str, str]],
    expected_pairs: list[tuple[str, str]],
    *,
    records_per_pair: int,
    limit: int = 12,
) -> list[str]:
    if records_per_pair <= 0:
        return []
    missing = [
        (f"{dataset_name}::{model_name}", int(pair_counts.get((dataset_name, model_name), 0)))
        for dataset_name, model_name in expected_pairs
        if int(pair_counts.get((dataset_name, model_name), 0)) < records_per_pair
    ]
    missing.sort(key=lambda item: (item[1], item[0]))
    return [f"{label}:{count}/{records_per_pair}" for label, count in missing[: max(1, int(limit))]]


def _next_sample_with_collection(
    dataset: Any,
    collector: Any,
    *,
    seen_total: int,
    logger: logging.Logger,
    wait_context: Any,
) -> Any:
    if not hasattr(dataset, "try_next_sample"):
        dataset_iter = iter(dataset)
        if collector is not None and not bool(getattr(collector, "is_async_mode", False)):
            dataset.maybe_collect(seen_total)
        return next(dataset_iter)

    timeout_s = max(0.0, env_float("HELDOUT_SAMPLE_WAIT_TIMEOUT_S", 1800.0))
    status_every_s = max(0.0, env_float("HELDOUT_SAMPLE_WAIT_STATUS_EVERY_S", 60.0))
    poll_s = max(0.1, env_float("HELDOUT_SAMPLE_WAIT_POLL_S", 1.0))
    fail_on_timeout = env_bool("HELDOUT_FAIL_ON_SAMPLE_WAIT_TIMEOUT", True)
    started = time.monotonic()
    last_status = started

    while True:
        if collector is not None and not bool(getattr(collector, "is_async_mode", False)):
            dataset.maybe_collect(seen_total)

        sample = dataset.try_next_sample()
        if sample is not None:
            return sample

        if collector is not None and hasattr(collector, "assert_healthy"):
            collector.assert_healthy()

        now = time.monotonic()
        waited_s = now - started
        if status_every_s > 0.0 and now - last_status >= status_every_s:
            logger.warning(
                "Waiting for held-out sample: waited_s=%.1f context=%s runtime=%s",
                waited_s,
                wait_context() if callable(wait_context) else {},
                _compact_collector_snapshot(dataset, collector),
            )
            last_status = now

        if timeout_s > 0.0 and waited_s >= timeout_s:
            message = (
                "Timed out waiting for held-out sample "
                f"(waited_s={waited_s:.1f}, timeout_s={timeout_s:.1f}, "
                f"context={wait_context() if callable(wait_context) else {}}, "
                f"runtime={_compact_collector_snapshot(dataset, collector)})"
            )
            if fail_on_timeout:
                raise TimeoutError(message)
            logger.warning("%s; stopping build because HELDOUT_FAIL_ON_SAMPLE_WAIT_TIMEOUT=false", message)
            raise StopIteration

        time.sleep(poll_s)


def _should_accept_pair_in_size_mode(
    pair_counts: Counter[tuple[str, str]],
    pair: tuple[str, str],
    *,
    balance_slack: int,
) -> bool:
    if balance_slack < 0:
        return True
    expected_pairs = sorted(allowed_pairs())
    min_count = min(int(pair_counts.get(expected_pair, 0)) for expected_pair in expected_pairs)
    return int(pair_counts.get(pair, 0)) <= min_count + int(balance_slack)


def _active_pairs_for_counts(
    pair_counts: Counter[tuple[str, str]],
    expected_pairs: list[tuple[str, str]],
    *,
    records_per_pair: int,
) -> list[tuple[str, str]]:
    if records_per_pair <= 0:
        return list(expected_pairs)
    return [
        pair
        for pair in expected_pairs
        if int(pair_counts.get(pair, 0)) < int(records_per_pair)
    ]


def _build_balanced_heldout_dataset(
    cfg: DictConfig,
    *,
    dataset: Any,
    collector: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    offline_cfg = cfg.train.get("offline_dataset", {})
    builder_cfg = offline_cfg.get("builder", {})
    patch_size, max_T_patches, max_d_out, max_x_rows = resolve_big_vae_curriculum_targets(cfg)
    root_dir = Path(str(offline_cfg.get("root_dir", ""))).expanduser()
    if not root_dir.is_absolute():
        root_dir = Path.cwd() / root_dir

    target_size_bytes = resolve_offline_target_size_bytes(builder_cfg)
    records_per_pair = max(0, int(builder_cfg.get("heldout_records_per_pair", 0)))
    max_seen_samples = max(0, int(builder_cfg.get("max_seen_samples", 0)))
    log_every_seen = max(1, int(builder_cfg.get("log_every_seen_samples", 1000)))
    stop_on_target_size = env_bool("HELDOUT_STOP_ON_TARGET_SIZE", records_per_pair <= 0)
    balance_slack = env_int("HELDOUT_BALANCE_SLACK_PER_PAIR", 32)
    expected_pairs = sorted(allowed_pairs())

    writer = BigVAEOfflineDatasetWriter(
        root_dir=root_dir,
        target_size_bytes=target_size_bytes,
        patch_size=patch_size,
        max_T_patches=max_T_patches,
        max_d_out=max_d_out,
        max_x_rows=max_x_rows,
        x_chunk_size_records=int(builder_cfg.get("x_chunk_size_records", 128)),
        max_samples_per_source=int(builder_cfg.get("max_samples_per_source", 0)),
        enforce_stage_compatibility=bool(builder_cfg.get("enforce_stage_compatibility", False)),
        overwrite_existing=bool(builder_cfg.get("overwrite_existing", False)),
        seed=int(cfg.data.get("seed", 42)),
        logger=logger,
        config_snapshot=resolved_cfg_snapshot(cfg),
    )

    seen_total = 0
    pair_counts: Counter[tuple[str, str]] = Counter()
    dataset_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    last_active_pairs: tuple[tuple[str, str], ...] | None = None

    def pair_targets_reached() -> bool:
        if records_per_pair <= 0:
            return False
        return all(int(pair_counts.get(pair, 0)) >= records_per_pair for pair in expected_pairs)

    def update_collector_active_pairs() -> None:
        nonlocal last_active_pairs
        if collector is None or not hasattr(collector, "set_active_pairs"):
            return
        active_pairs = tuple(
            _active_pairs_for_counts(
                pair_counts,
                expected_pairs,
                records_per_pair=records_per_pair,
            )
        )
        if active_pairs == last_active_pairs:
            return
        collector.set_active_pairs(list(active_pairs))
        last_active_pairs = active_pairs
        logger.info(
            "Updated held-out active collector pairs: active=%s total=%s",
            len(active_pairs),
            len(expected_pairs),
        )

    def wait_context() -> dict[str, Any]:
        writer_stats = writer.stats()
        return {
            "seen_total": int(seen_total),
            "accepted": int(writer_stats["accepted_records"]),
            "size_gb": round(float(writer_stats["written_size_gb"]), 3),
            "pair_min": min(int(pair_counts.get(pair, 0)) for pair in expected_pairs),
            "pair_max": max(int(pair_counts.get(pair, 0)) for pair in expected_pairs),
            "missing_pairs": _format_missing_pairs(
                pair_counts,
                expected_pairs,
                records_per_pair=records_per_pair,
            ),
            "skipped_pair_cap": int(status_counts["skipped_pair_cap"]),
            "skipped_unexpected_pair": int(status_counts["skipped_unexpected_pair"]),
        }

    try:
        update_collector_active_pairs()
        while True:
            if records_per_pair > 0 and pair_targets_reached():
                logger.info("Held-out build reached records_per_pair=%s for all pairs", records_per_pair)
                break
            if stop_on_target_size and writer.reached_target_size:
                logger.info("Held-out build reached target_size_bytes=%s", target_size_bytes)
                break
            if max_seen_samples > 0 and seen_total >= max_seen_samples:
                logger.warning("Held-out build reached max_seen_samples=%s", max_seen_samples)
                break

            try:
                sample = _next_sample_with_collection(
                    dataset,
                    collector,
                    seen_total=seen_total,
                    logger=logger,
                    wait_context=wait_context,
                )
            except StopIteration:
                logger.warning("Held-out build dataset iterator exhausted")
                break

            seen_total += 1
            pair = sample_pair(sample)
            if pair not in expected_pairs:
                status_counts["skipped_unexpected_pair"] += 1
                continue
            if records_per_pair > 0 and int(pair_counts[pair]) >= records_per_pair:
                status_counts["skipped_pair_cap"] += 1
                continue
            if records_per_pair <= 0 and not _should_accept_pair_in_size_mode(
                pair_counts,
                pair,
                balance_slack=balance_slack,
            ):
                status_counts["skipped_balance_slack"] += 1
                continue

            status = writer.ingest(sample)
            status_counts[str(status)] += 1
            if status == "accepted":
                dataset_name, model_name = pair
                pair_counts[pair] += 1
                dataset_counts[dataset_name] += 1
                model_counts[model_name] += 1
                if records_per_pair > 0 and int(pair_counts[pair]) >= records_per_pair:
                    update_collector_active_pairs()

            if seen_total % log_every_seen == 0:
                writer_stats = writer.stats()
                min_pair = min(int(pair_counts.get(pair, 0)) for pair in expected_pairs)
                max_pair = max(int(pair_counts.get(pair, 0)) for pair in expected_pairs)
                logger.info(
                    "Held-out build progress: seen_total=%s accepted=%s size_gb=%.2f "
                    "pair_min=%s pair_max=%s missing_pairs=%s skipped_pair_cap=%s "
                    "skipped_balance=%s writer_skipped_invalid=%s",
                    seen_total,
                    int(writer_stats["accepted_records"]),
                    float(writer_stats["written_size_gb"]),
                    min_pair,
                    max_pair,
                    _format_missing_pairs(pair_counts, expected_pairs, records_per_pair=records_per_pair),
                    int(status_counts["skipped_pair_cap"]),
                    int(status_counts["skipped_balance_slack"]),
                    int(writer_stats["skipped_invalid"]),
                )
    finally:
        manifest = writer.close()

    summary = {
        "root_dir": str(root_dir),
        "manifest": manifest,
        "heldout_dataset_models": {key: list(value) for key, value in HELDOUT_DATASET_MODELS.items()},
        "records_per_pair": int(records_per_pair),
        "target_size_bytes": int(target_size_bytes),
        "stop_on_target_size": bool(stop_on_target_size),
        "balance_slack_per_pair": int(balance_slack),
        "seen_total": int(seen_total),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "pair_counts": {f"{dataset}::{model}": int(count) for (dataset, model), count in sorted(pair_counts.items())},
        "dataset_counts": {str(key): int(value) for key, value in sorted(dataset_counts.items())},
        "model_counts": {str(key): int(value) for key, value in sorted(model_counts.items())},
    }
    summary["coverage"] = coverage_report(
        observed_datasets=dataset_counts.keys(),
        observed_models=model_counts.keys(),
        observed_pairs=pair_counts.keys(),
        skipped={
            "unexpected_pair": int(status_counts["skipped_unexpected_pair"]),
            "writer_invalid": int(manifest.get("skipped_invalid", 0)) if isinstance(manifest, dict) else 0,
            "writer_incompatible": int(manifest.get("skipped_incompatible", 0)) if isinstance(manifest, dict) else 0,
        },
    )

    analysis_dir = root_dir / "heldout_build_analysis"
    write_json(analysis_dir / "summary.json", summary)
    write_csv(analysis_dir / "pair_counts.csv", counter_to_rows(pair_counts))
    write_csv(analysis_dir / "dataset_counts.csv", counter_to_rows(dataset_counts))
    write_csv(analysis_dir / "model_counts.csv", counter_to_rows(model_counts))
    write_json(root_dir / "heldout_pairs.json", {"dataset_models": summary["heldout_dataset_models"]})
    return summary


def _load_base_config() -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        return compose(config_name="big_vae/train/default")


def main() -> None:
    cfg = _load_base_config()
    promote_run_profile_to_root(cfg)
    sanitize_programmatic_hydra_logging(cfg, role="build_big_vae_heldout_offline_dataset")
    root_dir = env_path(
        "HELDOUT_ROOT",
        default_heldout_root(),
    )
    log_dir = apply_heldout_log_dir(cfg, root_dir=root_dir)
    write_artifact_layout(
        root_dir.parent,
        kind="post_train.heldout_dataset_build",
        run_id=root_dir.name,
        files={"manifest": root_dir / "manifest.json", "summary": root_dir / "heldout_build_analysis" / "summary.json"},
        dirs={"dataset": root_dir, "logs": log_dir, "analysis": root_dir / "heldout_build_analysis"},
        metadata={"heldout_root": root_dir},
    )
    apply_heldout_data_profile(cfg)
    apply_heldout_collector_defaults_from_env(cfg)
    apply_offline_dataset_defaults(cfg, root_dir=root_dir)
    apply_builder_defaults_from_env(cfg, root_dir=root_dir)
    validate_heldout_preflight(cfg)

    log_path = configure_process_logging(cfg=cfg, role="build_big_vae_heldout_offline_dataset", rank=0, force=True)
    logger = LOGGER
    logger.info("Run log file: %s", log_path)
    logger.info("Held-out log dir: %s", log_dir)
    logger.info("Starting held-out BigVAE offline dataset build")
    logger.info("Held-out pairs: %s", {key: list(value) for key, value in HELDOUT_DATASET_MODELS.items()})
    logger.info("Effective offline root: %s", root_dir)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    with data_pipeline(cfg, logger=logger, emit_run_report=True, rank=0) as (dataset, collector):
        summary = _build_balanced_heldout_dataset(
            cfg,
            dataset=dataset,
            collector=collector,
            logger=logger,
        )

    manifest = summary.get("manifest", {})
    logger.info(
        "Held-out offline dataset build complete: root=%s actual_size_gb=%.2f accepted_records=%s unique_sources=%s",
        summary.get("root_dir", str(root_dir)),
        float(manifest.get("actual_size_gb", 0.0)) if isinstance(manifest, dict) else 0.0,
        int(manifest.get("accepted_records", 0)) if isinstance(manifest, dict) else 0,
        int(manifest.get("unique_sources", 0)) if isinstance(manifest, dict) else 0,
    )


if __name__ == "__main__":
    main()
