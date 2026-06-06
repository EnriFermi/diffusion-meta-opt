from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_train_research.vit_latent_scaling.config import build_run_config
from post_train_research.vit_latent_scaling.runtime import CometTracker, configure_logger, prepare_run_paths, write_config_snapshots
from post_train_research.vit_latent_scaling.train import run_training
from training.runtime import patch_argparse_lazy_help_for_hydra_py314


patch_argparse_lazy_help_for_hydra_py314()


@hydra.main(version_base=None, config_path="../../conf/vit_latent_scaling", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg, raw_cfg = build_run_config(cfg)
    paths = prepare_run_paths(run_cfg)
    logger = configure_logger(paths)
    write_config_snapshots(paths, run_cfg, raw_cfg)
    comet = CometTracker(run_cfg, paths, logger)
    logger.info(
        "Starting run_id=%s profile=%s setup=%s init=%s run_dir=%s",
        paths.run_id,
        run_cfg.profile.name,
        run_cfg.setup.kind,
        run_cfg.init.kind if run_cfg.init.kind != "fresh" else f"fresh/{run_cfg.init.fresh_latent_mode}",
        paths.run_dir,
    )
    try:
        result = run_training(run_cfg, paths, logger, comet)
        logger.info("Finished run_id=%s best_acc=%.4f final_acc=%.4f", paths.run_id, result.summary["best_test_accuracy"], result.summary["final_test_accuracy"])
    finally:
        comet.end()


if __name__ == "__main__":
    main()
