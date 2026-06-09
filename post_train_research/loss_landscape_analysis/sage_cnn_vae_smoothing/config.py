from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    run_label: str = "sage_cnn_vae_smoothing_fast"
    artifact_root: str = "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    cache_first: bool = True
    force_rerun: bool = False
    device: str = _default_device()
    dtype: str = "float32"
    seed: int = 0

    dataset_name: str = "fashion_mnist"
    data_root: str = "artifacts/loss_landscape_analysis/data"
    download: bool = True
    train_subset: int = 1024
    test_subset: int = 512
    cnn_batch_size: int = 64

    weight_runs: int = 16
    weight_train_steps: int = 80
    weight_snapshot_every: int = 10
    weight_lr: float = 1e-3
    vae_train_fraction: float = 0.8

    latent_dim: int = 16
    vae_hidden_dim: int = 512
    vae_steps: int = 500
    vae_batch_size: int = 64
    vae_lr: float = 1e-3
    beta_kl: float = 1e-4

    flow_steps: int = 300
    flow_batch_size: int = 8
    flow_lr: float = 1e-3
    flow_grad_clip_norm: float = 10.0
    flow_log_every: int = 25
    flow_num_layers: int = 8
    flow_hidden_dim: int = 64
    flow_network_depth: int = 2
    flow_spline_bins: int = 8
    flow_spline_bound: float = 5.0
    flow_eta: float = 0.0
    mixup_alpha_min: float = -0.1
    mixup_alpha_max: float = 1.1
    geometry_eval_samples: int = 16

    tune_starts: int = 4
    eval_starts: int = 8
    downstream_steps: int = 80
    downstream_eval_every: int = 1
    downstream_batch_size: int = 128
    raw_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    latent_lrs: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2)
    nf_lrs: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2)
    success_threshold: float = 0.25
    finite_penalty: float = 1e12

    save_figures: bool = True


def fast_config(**overrides: Any) -> ExperimentConfig:
    values = asdict(ExperimentConfig())
    values.update(overrides)
    return ExperimentConfig(**values)


def paperish_config(**overrides: Any) -> ExperimentConfig:
    values = asdict(ExperimentConfig())
    values.update(
        {
            "run_label": "sage_cnn_vae_smoothing_paperish",
            "train_subset": 4096,
            "test_subset": 1024,
            "weight_runs": 64,
            "weight_train_steps": 200,
            "weight_snapshot_every": 10,
            "vae_steps": 2000,
            "flow_steps": 1000,
            "geometry_eval_samples": 64,
            "tune_starts": 8,
            "eval_starts": 16,
            "downstream_steps": 200,
            "flow_eta": 0.2,
        }
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_hash(cfg: ExperimentConfig) -> str:
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.artifact_root).expanduser().resolve() / str(cfg.run_label)


def torch_dtype(cfg: ExperimentConfig) -> torch.dtype:
    value = str(cfg.dtype).strip().lower()
    if value == "float64":
        return torch.float64
    if value == "float32":
        return torch.float32
    raise ValueError(f"dtype must be one of {{'float32', 'float64'}}, got {cfg.dtype!r}")


def write_config(path: Path, cfg: ExperimentConfig) -> dict[str, Any]:
    payload = {"config_hash": config_hash(cfg), "config": config_to_dict(cfg)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def read_config(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
