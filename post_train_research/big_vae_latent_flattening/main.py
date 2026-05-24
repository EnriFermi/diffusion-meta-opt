from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_train_research.big_vae_latent_flattening.config import build_run_config
from post_train_research.big_vae_latent_flattening.runtime import configure_logger, prepare_run_paths, write_config_snapshots
from post_train_research.big_vae_latent_flattening.train import run_training
from training.runtime import patch_argparse_lazy_help_for_hydra_py314


patch_argparse_lazy_help_for_hydra_py314()


@hydra.main(version_base=None, config_path="../../conf/big_vae_latent_flattening", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg, raw_cfg = build_run_config(cfg)
    paths = prepare_run_paths(run_cfg)
    logger = configure_logger(paths)
    write_config_snapshots(paths, run_cfg, raw_cfg)
    logger.info("Starting run_id=%s run_dir=%s", paths.run_id, paths.run_dir)
    result = run_training(run_cfg, paths, logger)
    logger.info("Finished run_id=%s final_checkpoint=%s", paths.run_id, result["checkpoint"])


if __name__ == "__main__":
    main()
