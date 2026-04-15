from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    HELDOUT_DATASET_MODELS,
    allowed_pairs,
    apply_builder_defaults_from_env,
    apply_heldout_data_profile,
    apply_offline_dataset_defaults,
    counter_to_rows,
    env_bool,
    env_int,
    env_path,
    primary_dataset_name,
    promote_run_profile_to_root,
    resolved_cfg_snapshot,
    sanitize_programmatic_hydra_logging,
    sample_pair,
    write_csv,
    write_json,
)
from dataset import data_pipeline
from dataset.big_vae_offline import (
    BigVAEOfflineDatasetWriter,
    resolve_big_vae_curriculum_targets,
    resolve_offline_target_size_bytes,
)
from dataset.logging_utils import configure_process_logging


LOGGER = logging.getLogger("build_big_vae_heldout_offline_dataset")


def _dataset_iterator_with_collection(dataset: Any, collector: Any) -> Iterator[Any]:
    dataset_iter = iter(dataset)
    seen = 0
    while True:
        if collector is not None and not bool(getattr(collector, "is_async_mode", False)):
            dataset.maybe_collect(seen)
        yield next(dataset_iter)
        seen += 1


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


def _build_balanced_heldout_dataset(
    cfg: DictConfig,
    *,
    dataset_iter: Iterator[Any],
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

    def pair_targets_reached() -> bool:
        if records_per_pair <= 0:
            return False
        return all(int(pair_counts.get(pair, 0)) >= records_per_pair for pair in expected_pairs)

    try:
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
                sample = next(dataset_iter)
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

            if seen_total % log_every_seen == 0:
                writer_stats = writer.stats()
                min_pair = min(int(pair_counts.get(pair, 0)) for pair in expected_pairs)
                max_pair = max(int(pair_counts.get(pair, 0)) for pair in expected_pairs)
                logger.info(
                    "Held-out build progress: seen_total=%s accepted=%s size_gb=%.2f "
                    "pair_min=%s pair_max=%s skipped_pair_cap=%s skipped_balance=%s writer_skipped_invalid=%s",
                    seen_total,
                    int(writer_stats["accepted_records"]),
                    float(writer_stats["written_size_gb"]),
                    min_pair,
                    max_pair,
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

    analysis_dir = root_dir / "heldout_build_analysis"
    write_json(analysis_dir / "summary.json", summary)
    write_csv(analysis_dir / "pair_counts.csv", counter_to_rows(pair_counts))
    write_csv(analysis_dir / "dataset_counts.csv", counter_to_rows(dataset_counts))
    write_csv(analysis_dir / "model_counts.csv", counter_to_rows(model_counts))
    write_json(root_dir / "heldout_pairs.json", {"dataset_models": summary["heldout_dataset_models"]})
    return summary


def _load_base_config() -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        return compose(config_name="config")


def main() -> None:
    cfg = _load_base_config()
    promote_run_profile_to_root(cfg)
    sanitize_programmatic_hydra_logging(cfg, role="build_big_vae_heldout_offline_dataset")
    root_dir = env_path(
        "HELDOUT_ROOT",
        "post_train_research/big_vae_heldout_eval/artifacts/offline_dataset",
    )
    apply_heldout_data_profile(cfg)
    apply_offline_dataset_defaults(cfg, root_dir=root_dir)
    apply_builder_defaults_from_env(cfg, root_dir=root_dir)

    log_path = configure_process_logging(cfg=cfg, role="build_big_vae_heldout_offline_dataset", rank=0, force=True)
    logger = LOGGER
    logger.info("Run log file: %s", log_path)
    logger.info("Starting held-out BigVAE offline dataset build")
    logger.info("Held-out pairs: %s", {key: list(value) for key, value in HELDOUT_DATASET_MODELS.items()})
    logger.info("Effective offline root: %s", root_dir)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    with data_pipeline(cfg, logger=logger, emit_run_report=True, rank=0) as (dataset, collector):
        summary = _build_balanced_heldout_dataset(
            cfg,
            dataset_iter=_dataset_iterator_with_collection(dataset, collector),
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
