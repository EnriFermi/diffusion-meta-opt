from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_train_research.tinyvit_latent_h1.config import build_run_config
from post_train_research.tinyvit_latent_h1.experiment import prepare_config_from_source_checkpoint, run_experiment
from post_train_research.tinyvit_latent_h1.runtime import CometTracker, configure_logger, prepare_run_paths, write_config_snapshots


@hydra.main(version_base=None, config_path="../../conf/tinyvit_latent_h1", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg, raw_cfg = build_run_config(cfg)
    paths = prepare_run_paths(run_cfg)
    logger = configure_logger(paths)
    source_checkpoint_path = prepare_config_from_source_checkpoint(run_cfg, logger)
    write_config_snapshots(paths, run_cfg, raw_cfg)
    comet = CometTracker(run_cfg, paths, logger)
    logger.info("Starting run_id=%s run_label=%s source=%s", paths.run_id, run_cfg.run_label, run_cfg.source.run_dir)
    try:
        summary = run_experiment(run_cfg, paths, logger, comet, source_checkpoint_path=source_checkpoint_path)
        logger.info(
            "Finished run_id=%s paired_starts=%s z_star_test_acc=%.4f",
            paths.run_id,
            summary["paired_start_count"],
            summary["z_star_test_acc"],
        )
    finally:
        comet.end()


if __name__ == "__main__":
    main()
