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
    run_label: str = "e0_e4_paperish"
    artifact_root: str = "artifacts/loss_landscape_analysis/flow_preconditioning"
    cache_first: bool = True
    force_rerun: bool = False
    device: str = _default_device()
    dtype: str = "float32"

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    k_tune: int = 8
    k_eval: int = 32
    budgets: tuple[int, ...] = (300, 1000)
    e2_gamma_values: tuple[float, ...] = (0.3, 1.0, 3.0)
    e2_dims: tuple[int, ...] = (2, 4)
    e2_objectives: tuple[str, ...] = ("rastrigin_abs", "rosenbrock_abs")
    e3_condition_numbers: tuple[float, ...] = (1.0, 1e2, 1e4)
    e4_rho_values: tuple[float, ...] = (0.0, 1e-3, 1e-2, 5e-2)
    e4_main_rho: float = 1e-2

    init_std: float = 0.5
    e2_init_std: float = 2.0
    e1_domain_z1: tuple[float, float] = (-2.0, 2.0)
    e1_domain_z2: tuple[float, float] = (-4.0, 4.0)
    train_points: int = 128
    probe_points: int = 64
    test_points: int = 512
    train_shift: float = 0.0
    probe_shift: float = 0.37
    test_shift: float = 0.73

    mlp_width: int = 8
    mlp_dim: int = 25
    flow_random_samples: int = 1024
    flow_trajectory_count: int = 32
    flow_trajectory_steps: int = 25
    flow_trajectory_lr: float = 1e-2
    heldout_geometry_samples: int = 256

    flow_steps: int = 3000
    sanity_flow_steps: int = 2000
    flow_batch_size: int = 128
    flow_lr: float = 1e-3
    flow_grad_clip_norm: float = 10.0
    flow_log_every: int = 100
    flow_num_layers: int = 8
    flow_hidden_dim: int = 64
    flow_network_depth: int = 2
    flow_log_scale_clamp: float = 1.5
    flow_dropout: float = 0.0
    random_flow_std: float = 1e-3
    random_flow_near_identity_noise_std: float = 0.0

    sanity_dim: int = 16
    sanity_condition_number: float = 1e3
    sanity_random_samples: int = 512
    sanity_trajectory_count: int = 16
    sanity_trajectory_steps: int = 25
    sanity_heldout_geometry_samples: int = 128
    e1_dim: int = 2
    e1_random_samples: int = 1024
    e1_trajectory_count: int = 32
    e1_trajectory_steps: int = 25
    e1_heldout_geometry_samples: int = 256
    e2_random_samples: int = 1024
    e2_trajectory_count: int = 32
    e2_trajectory_steps: int = 25
    e2_heldout_geometry_samples: int = 256
    e3_dim: int = 16
    e3_output_dim: int = 64
    e3_hidden_dim: int = 128
    e3_skip: float = 0.05
    e3_random_samples: int = 1024
    e3_trajectory_count: int = 32
    e3_trajectory_steps: int = 25
    e3_heldout_geometry_samples: int = 256

    low_dim_grid_points: int = 80
    low_dim_trajectory_budget: int = 300

    sgd_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    adam_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    aulc_eps: float = 1e-12
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 12345

    write_csv: bool = True
    write_parquet: bool = True
    save_figures: bool = True


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
    payload = {
        "config_hash": config_hash(cfg),
        "config": config_to_dict(cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def read_config(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
