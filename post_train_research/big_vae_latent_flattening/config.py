from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    run_label: str
    notes: str
    tags: list[str]


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root_dir: str
    checkpoint_every_steps: int
    save_latest: bool
    save_final: bool


@dataclass(frozen=True, slots=True)
class DataConfig:
    offline_root: str
    shuffle_chunks: bool
    shuffle_records_within_chunk: bool
    repeat: bool
    seed: int
    sampling_mode: str
    sampling_group_keys: list[str]
    sampling_window_size_records: int
    sampling_max_records_per_chunk_round: int
    weight_cache_size: int
    x_chunk_cache_size: int
    runtime_enforce_stage_compatibility: bool
    batch_size: int
    source_pool_size: int
    max_T_patches: int
    max_d_out: int
    max_x_rows: int
    stable_batch_shapes: bool


@dataclass(frozen=True, slots=True)
class BigVAEConfig:
    checkpoint: str


@dataclass(frozen=True, slots=True)
class FlowConfig:
    num_layers: int
    hidden_dim: int
    network_depth: int
    log_scale_clamp: float
    dropout: float


@dataclass(frozen=True, slots=True)
class TrainConfig:
    device: str
    seed: int
    max_steps: int
    lr: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    grad_clip_norm: float
    eta: float
    probes: int
    loss_scale: str
    iso_coef: float
    z_norm_coef: float
    log_every_steps: int
    amp_encode: bool
    tf32: bool
    force_math_attention: bool


@dataclass(frozen=True, slots=True)
class RunConfig:
    experiment: ExperimentConfig
    storage: StorageConfig
    data: DataConfig
    big_vae: BigVAEConfig
    flow: FlowConfig
    train: TrainConfig

    @property
    def run_label(self) -> str:
        label = self.experiment.run_label.strip()
        return label or "big_vae_latent_flattening"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_dict(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved Hydra config must be a mapping")
    return payload


def _nested(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    raise TypeError("tags must be a list")


def build_run_config(cfg: DictConfig) -> tuple[RunConfig, dict[str, Any]]:
    raw = _cfg_dict(cfg)
    experiment = _nested(raw, "experiment")
    storage = _nested(raw, "storage")
    data = _nested(raw, "data")
    big_vae = _nested(raw, "big_vae")
    flow = _nested(raw, "flow")
    train = _nested(raw, "train")

    run_cfg = RunConfig(
        experiment=ExperimentConfig(
            run_label=str(experiment.get("run_label", "")),
            notes=str(experiment.get("notes", "")),
            tags=_tags(experiment.get("tags", [])),
        ),
        storage=StorageConfig(
            root_dir=str(storage.get("root_dir", "./artifacts/big_vae/eval/latent_flattening")),
            checkpoint_every_steps=int(storage.get("checkpoint_every_steps", 250)),
            save_latest=bool(storage.get("save_latest", True)),
            save_final=bool(storage.get("save_final", True)),
        ),
        data=DataConfig(
            offline_root=str(data.get("offline_root", "")),
            shuffle_chunks=bool(data.get("shuffle_chunks", True)),
            shuffle_records_within_chunk=bool(data.get("shuffle_records_within_chunk", True)),
            repeat=bool(data.get("repeat", True)),
            seed=int(data.get("seed", train.get("seed", 42))),
            sampling_mode=str(data.get("sampling_mode", "balanced")),
            sampling_group_keys=[str(item) for item in data.get("sampling_group_keys", ["dataset", "model", "layer_type", "depth"])],
            sampling_window_size_records=int(data.get("sampling_window_size_records", 2048)),
            sampling_max_records_per_chunk_round=int(data.get("sampling_max_records_per_chunk_round", 8)),
            weight_cache_size=int(data.get("weight_cache_size", 64)),
            x_chunk_cache_size=int(data.get("x_chunk_cache_size", 4)),
            runtime_enforce_stage_compatibility=bool(data.get("runtime_enforce_stage_compatibility", False)),
            batch_size=int(data.get("batch_size", 2)),
            source_pool_size=int(data.get("source_pool_size", data.get("batch_size", 2))),
            max_T_patches=int(data.get("max_T_patches", 4)),
            max_d_out=int(data.get("max_d_out", 16)),
            max_x_rows=int(data.get("max_x_rows", 0)),
            stable_batch_shapes=bool(data.get("stable_batch_shapes", True)),
        ),
        big_vae=BigVAEConfig(
            checkpoint=str(big_vae.get("checkpoint", "")),
        ),
        flow=FlowConfig(
            num_layers=int(flow.get("num_layers", 8)),
            hidden_dim=int(flow.get("hidden_dim", 512)),
            network_depth=int(flow.get("network_depth", 2)),
            log_scale_clamp=float(flow.get("log_scale_clamp", 2.0)),
            dropout=float(flow.get("dropout", 0.0)),
        ),
        train=TrainConfig(
            device=str(train.get("device", "auto")),
            seed=int(train.get("seed", 42)),
            max_steps=int(train.get("max_steps", 1000)),
            lr=float(train.get("lr", 1e-4)),
            weight_decay=float(train.get("weight_decay", 1e-6)),
            adam_beta1=float(train.get("adam_beta1", 0.9)),
            adam_beta2=float(train.get("adam_beta2", 0.95)),
            adam_eps=float(train.get("adam_eps", 1e-8)),
            grad_clip_norm=float(train.get("grad_clip_norm", 1.0)),
            eta=float(train.get("eta", 0.2)),
            probes=int(train.get("probes", 1)),
            loss_scale=str(train.get("loss_scale", "readme")),
            iso_coef=float(train.get("iso_coef", 1.0)),
            z_norm_coef=float(train.get("z_norm_coef", 1e-6)),
            log_every_steps=int(train.get("log_every_steps", 10)),
            amp_encode=bool(train.get("amp_encode", True)),
            tf32=bool(train.get("tf32", True)),
            force_math_attention=bool(train.get("force_math_attention", True)),
        ),
    )

    if not run_cfg.big_vae.checkpoint.strip():
        raise ValueError("big_vae.checkpoint must be set")
    if not run_cfg.data.offline_root.strip():
        raise ValueError("data.offline_root must be set")
    if int(run_cfg.data.batch_size) <= 0:
        raise ValueError("data.batch_size must be positive")
    if int(run_cfg.train.max_steps) <= 0:
        raise ValueError("train.max_steps must be positive")

    return run_cfg, raw


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
