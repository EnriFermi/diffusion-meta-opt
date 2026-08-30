from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import data_pipeline, setup_logging
from big_vae.datasets.offline import (
    ensure_presliced_big_vae_dataset,
    offline_big_vae_data_pipeline,
    presliced_big_vae_data_pipeline,
)
from dataset.logging_utils import LOG_PATH_ENV, configure_process_logging, resolve_process_log_path
from experiments.background_prefetch import BackgroundPrefetcher
from big_vae.models import (
    BigVAEConfig,
    DistributionConfig,
    EncoderConfig,
    MiniVAEConfig,
    ModelConfig,
    WeightQuantileVAE,
    build_weight_quantile_vae,
)
from training.optim import build_adamw_optimizer, build_cosine_scheduler
from training.forensics import (
    apply_nccl_forensics_env,
    emit_fatal_report,
    maybe_enable_core_dumps,
    maybe_redirect_stdio,
    monitor_send_event,
    start_process_monitor,
    stop_process_monitor,
)
from training.runtime import (
    autocast_context as runtime_autocast_context,
    configure_per_run_artifacts as runtime_configure_per_run_artifacts,
    create_grad_scaler as runtime_create_grad_scaler,
    find_free_port as runtime_find_free_port,
    get_rank_logger,
    maybe_compile_model,
    resolve_amp as runtime_resolve_amp,
    resolve_backend as runtime_resolve_backend,
    resolve_device as runtime_resolve_device,
    resolve_world_size as runtime_resolve_world_size,
    seed_everything as runtime_seed_everything,
    set_speed_optimizations as runtime_set_speed_optimizations,
)
from big_vae.runtime.artifacts import write_artifact_layout, write_json_file, write_yaml_file

import hydra

from training.big_vae.runtime import _configure_run_artifacts, _find_free_port, _promote_run_profile_to_root, _resolve_world_size
from training.big_vae.worker import _run_worker

_HYDRA_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "conf")


def _patch_argparse_lazy_help_for_hydra_py314() -> None:
    """Hydra 1.3 passes a lazy help object; Python 3.14 argparse now validates help as a string."""
    if getattr(argparse.ArgumentParser, "_hydra_lazy_help_py314_patch", False):
        return
    original_check_help = getattr(argparse.ArgumentParser, "_check_help", None)
    if original_check_help is None:
        return

    def patched_check_help(self: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        original_check_help(self, action)

    argparse.ArgumentParser._check_help = patched_check_help  # type: ignore[method-assign]
    argparse.ArgumentParser._hydra_lazy_help_py314_patch = True  # type: ignore[attr-defined]


_patch_argparse_lazy_help_for_hydra_py314()


_SECRET_CONFIG_KEY_MARKERS = ("api_key", "token", "secret", "password")


def _redact_config_secrets(value: Any, *, key: str = "") -> Any:
    key_lower = key.lower()
    if any(marker in key_lower for marker in _SECRET_CONFIG_KEY_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(child_key): _redact_config_secrets(child_value, key=str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_redact_config_secrets(item) for item in value]
    return value


def _write_run_artifact_metadata(cfg: DictConfig, run_artifacts: dict[str, str]) -> None:
    run_root_dir = Path(run_artifacts["run_root_dir"])
    run_root_dir.mkdir(parents=True, exist_ok=True)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    redacted_payload = _redact_config_secrets(config_payload)
    write_yaml_file(run_root_dir / "config_resolved.yaml", redacted_payload)
    write_json_file(run_root_dir / "config_resolved.json", redacted_payload)
    write_artifact_layout(
        run_root_dir,
        kind="train.big_vae",
        run_id=str(run_artifacts["run_id"]),
        files={
            "config_yaml": run_root_dir / "config_resolved.yaml",
            "config_json": run_root_dir / "config_resolved.json",
            "log": run_artifacts["logs_dir"],
        },
        dirs={
            "logs": run_artifacts["logs_dir"],
            "reports": run_artifacts["reports_dir"],
            "crashes": run_artifacts["crashes_dir"],
            "checkpoints": run_artifacts["big_vae_checkpoint_dir"],
            "offline_dataset": run_artifacts["big_vae_offline_dataset_base_dir"],
            "presliced_dataset": run_artifacts["big_vae_presliced_dataset_base_dir"],
        },
        metadata=run_artifacts,
    )



def _spawn_entry(
    rank: int,
    world_size: int,
    cfg_dict: dict[str, Any],
    master_addr: str,
    master_port: int,
    monitor_queue: Any | None = None,
) -> None:
    _run_worker(
        rank=rank,
        world_size=world_size,
        cfg_dict=cfg_dict,
        master_addr=master_addr,
        master_port=master_port,
        monitor_queue=monitor_queue,
    )


@hydra.main(version_base=None, config_path=_HYDRA_CONFIG_PATH, config_name="big_vae/train/default")
def main(cfg: DictConfig) -> None:
    _promote_run_profile_to_root(cfg)
    print("[train_big_vae] Hydra config composed; entering launcher", flush=True)
    run_artifacts = _configure_run_artifacts(cfg)
    _write_run_artifact_metadata(cfg, run_artifacts)
    maybe_redirect_stdio(cfg, role="train_launcher", section="train")
    maybe_enable_core_dumps(cfg, section="train", logger=None)
    launcher_log_path = configure_process_logging(cfg=cfg, role="train_launcher", force=True)
    print(
        f"[train_big_vae] run_id={run_artifacts.get('run_id')} artifacts_root={run_artifacts.get('root_dir')}",
        flush=True,
    )
    train_cfg = cfg.get("train", {})
    print(
        "[train_big_vae] effective train config: "
        f"distributed={train_cfg.get('distributed')} "
        f"num_gpus={train_cfg.get('num_gpus')} "
        f"device={train_cfg.get('device')} "
        f"compile={train_cfg.get('compile')} "
        f"compile_mode={train_cfg.get('compile_mode')} "
        f"compile_dynamic={train_cfg.get('compile_dynamic')}",
        flush=True,
    )
    # Freeze one shared log path before spawning worker processes.
    if not os.environ.get(LOG_PATH_ENV):
        os.environ[LOG_PATH_ENV] = str(resolve_process_log_path(cfg, role="train_launcher"))
    logging.getLogger("train.launcher").info("Run log file: %s", launcher_log_path)

    world_size = _resolve_world_size(cfg)
    print(f"[train_big_vae] resolved world_size={world_size}", flush=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    apply_nccl_forensics_env(cfg_dict, section="train")
    monitor_process, monitor_queue, monitor_stop_event = start_process_monitor(
        cfg_dict,
        section="train",
        label="train_big_vae",
    )
    monitor_send_event(
        monitor_queue,
        {
            "type": "register",
            "pid": int(os.getpid()),
            "role": "train_launcher",
            "metadata": {
                "world_size": int(world_size),
            },
        },
    )

    try:
        if world_size <= 1:
            _run_worker(
                rank=0,
                world_size=1,
                cfg_dict=cfg_dict,
                master_addr="127.0.0.1",
                master_port=0,
                monitor_queue=monitor_queue,
            )
            return

        master_addr = str(cfg.train.get("master_addr", "127.0.0.1"))
        requested_port = int(cfg.train.get("master_port", 0))
        master_port = requested_port if requested_port > 0 else _find_free_port()

        mp.spawn(
            _spawn_entry,
            args=(world_size, cfg_dict, master_addr, master_port, monitor_queue),
            nprocs=world_size,
            join=True,
        )
    except BaseException as exc:
        traceback_text = traceback.format_exc()
        report_path = emit_fatal_report(
            cfg_dict,
            role="train_launcher",
            error=str(exc),
            traceback_text=traceback_text,
            extra={"world_size": int(world_size)},
            section="train",
        )
        monitor_send_event(
            monitor_queue,
            {
                "type": "fatal",
                "pid": int(os.getpid()),
                "role": "train_launcher",
                "error": str(exc),
                "traceback": traceback_text,
                "metadata": {
                    "world_size": int(world_size),
                    "fatal_report_path": report_path,
                },
            },
        )
        raise
    finally:
        monitor_send_event(
            monitor_queue,
            {
                "type": "exit",
                "pid": int(os.getpid()),
                "role": "train_launcher",
                "metadata": {
                    "world_size": int(world_size),
                },
            },
        )
        stop_process_monitor(monitor_process, monitor_queue, monitor_stop_event)
