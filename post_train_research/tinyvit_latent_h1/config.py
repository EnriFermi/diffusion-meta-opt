from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from experiments.compare_vit_tiny_latent_optimization import ViTTinyConfig


@dataclass(slots=True)
class ExperimentMetaConfig:
    run_label: str
    notes: str
    tags: list[str]


@dataclass(slots=True)
class StorageConfig:
    root_dir: str


@dataclass(slots=True)
class SourceConfig:
    vit_scaling_artifacts_root: str
    run_dir: str
    checkpoint_name: str
    use_checkpoint_vit_config: bool
    use_checkpoint_setup_config: bool
    train_anchor: bool
    anchor_steps: int
    anchor_lr: float
    anchor_weight_decay: float


@dataclass(slots=True)
class DataConfig:
    data_dir: str
    download: bool
    train_subset: int
    test_subset: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int


@dataclass(slots=True)
class ModelConfig:
    image_size: int
    patch_size: int
    in_channels: int
    num_classes: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attention_dropout: float


@dataclass(slots=True)
class SetupConfig:
    big_vae_checkpoint: str
    big_vae_decode: str
    big_vae_tile_T_patches: int
    big_vae_tile_d_out: int
    big_vae_latent_parameterization: str


@dataclass(slots=True)
class SearchConfig:
    epsilons: list[float]
    random_directions: int
    alpha_min: float
    alpha_max: float
    bracket_multiplier: float
    binary_search_steps: int


@dataclass(slots=True)
class TrainConfig:
    device: str
    seed: int
    steps: int
    raw_lrs: list[float]
    latent_lrs: list[float]
    raw_weight_decay: float
    latent_weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    grad_clip_norm: float
    amp: bool
    tf32: bool
    log_every_steps: int
    eval_every_steps: int
    recover_eps: float


@dataclass(slots=True)
class CometConfig:
    enabled: bool
    api_key: str
    workspace: str
    project_name: str
    experiment_name: str
    offline_directory: str
    log_code: bool
    tags: list[str]


@dataclass(slots=True)
class RunConfig:
    experiment: ExperimentMetaConfig
    storage: StorageConfig
    source: SourceConfig
    data: DataConfig
    model: ModelConfig
    setup: SetupConfig
    search: SearchConfig
    train: TrainConfig
    comet: CometConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_label(self) -> str:
        if self.experiment.run_label.strip():
            return self.experiment.run_label.strip()
        return "tinyvit_latent_h1"

    @property
    def vit_cfg(self) -> ViTTinyConfig:
        return ViTTinyConfig(
            image_size=int(self.model.image_size),
            patch_size=int(self.model.patch_size),
            in_channels=int(self.model.in_channels),
            num_classes=int(self.model.num_classes),
            hidden_dim=int(self.model.hidden_dim),
            depth=int(self.model.depth),
            num_heads=int(self.model.num_heads),
            mlp_ratio=float(self.model.mlp_ratio),
            dropout=float(self.model.dropout),
            attention_dropout=float(self.model.attention_dropout),
        )


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


def _as_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    raise TypeError("tags must be a list")


def _as_float_list(value: Any, *, key: str) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    raise TypeError(f"{key} must be a list")


def build_run_config(cfg: DictConfig) -> tuple[RunConfig, dict[str, Any]]:
    raw = _cfg_dict(cfg)
    experiment_raw = _nested(raw, "experiment")
    storage_raw = _nested(raw, "storage")
    source_raw = _nested(raw, "source")
    data_raw = _nested(raw, "data")
    model_raw = _nested(raw, "model")
    setup_raw = _nested(raw, "setup")
    search_raw = _nested(raw, "search")
    train_raw = _nested(raw, "train")
    telemetry_raw = _nested(raw, "telemetry")
    comet_raw = _nested(telemetry_raw, "comet")

    resolved = RunConfig(
        experiment=ExperimentMetaConfig(
            run_label=str(experiment_raw.get("run_label", "")).strip(),
            notes=str(experiment_raw.get("notes", "")).strip(),
            tags=_as_tags(experiment_raw.get("tags", [])),
        ),
        storage=StorageConfig(
            root_dir=str(storage_raw.get("root_dir", "./post_train_research/tinyvit_latent_h1/artifacts")).strip(),
        ),
        source=SourceConfig(
            vit_scaling_artifacts_root=str(
                source_raw.get("vit_scaling_artifacts_root", "./post_train_research/vit_latent_scaling/artifacts")
            ).strip(),
            run_dir=str(source_raw.get("run_dir", "")).strip(),
            checkpoint_name=str(source_raw.get("checkpoint_name", "best")).strip() or "best",
            use_checkpoint_vit_config=bool(source_raw.get("use_checkpoint_vit_config", True)),
            use_checkpoint_setup_config=bool(source_raw.get("use_checkpoint_setup_config", True)),
            train_anchor=bool(source_raw.get("train_anchor", False)),
            anchor_steps=max(0, int(source_raw.get("anchor_steps", 0))),
            anchor_lr=float(source_raw.get("anchor_lr", 0.0)),
            anchor_weight_decay=float(source_raw.get("anchor_weight_decay", 0.0)),
        ),
        data=DataConfig(
            data_dir=str(data_raw.get("data_dir", "./data/cifar10")).strip() or "./data/cifar10",
            download=bool(data_raw.get("download", True)),
            train_subset=max(0, int(data_raw.get("train_subset", 10000))),
            test_subset=max(0, int(data_raw.get("test_subset", 2000))),
            train_batch_size=max(1, int(data_raw.get("train_batch_size", 128))),
            eval_batch_size=max(1, int(data_raw.get("eval_batch_size", 512))),
            num_workers=max(0, int(data_raw.get("num_workers", 4))),
        ),
        model=ModelConfig(
            image_size=max(1, int(model_raw.get("image_size", 32))),
            patch_size=max(1, int(model_raw.get("patch_size", 4))),
            in_channels=max(1, int(model_raw.get("in_channels", 3))),
            num_classes=max(1, int(model_raw.get("num_classes", 10))),
            hidden_dim=max(1, int(model_raw.get("hidden_dim", 128))),
            depth=max(1, int(model_raw.get("depth", 3))),
            num_heads=max(1, int(model_raw.get("num_heads", 4))),
            mlp_ratio=float(model_raw.get("mlp_ratio", 2.0)),
            dropout=float(model_raw.get("dropout", 0.0)),
            attention_dropout=float(model_raw.get("attention_dropout", 0.0)),
        ),
        setup=SetupConfig(
            big_vae_checkpoint=str(setup_raw.get("big_vae_checkpoint", "")).strip(),
            big_vae_decode=str(setup_raw.get("big_vae_decode", "")).strip().lower(),
            big_vae_tile_T_patches=max(0, int(setup_raw.get("big_vae_tile_T_patches", 0))),
            big_vae_tile_d_out=max(0, int(setup_raw.get("big_vae_tile_d_out", 0))),
            big_vae_latent_parameterization=str(setup_raw.get("big_vae_latent_parameterization", "")).strip().lower(),
        ),
        search=SearchConfig(
            epsilons=_as_float_list(search_raw.get("epsilons", [0.05, 0.10]), key="search.epsilons"),
            random_directions=max(1, int(search_raw.get("random_directions", 4))),
            alpha_min=max(1e-12, float(search_raw.get("alpha_min", 1e-5))),
            alpha_max=max(1e-12, float(search_raw.get("alpha_max", 10.0))),
            bracket_multiplier=max(1.01, float(search_raw.get("bracket_multiplier", 1.8))),
            binary_search_steps=max(1, int(search_raw.get("binary_search_steps", 18))),
        ),
        train=TrainConfig(
            device=str(train_raw.get("device", "auto")).strip(),
            seed=int(train_raw.get("seed", 42)),
            steps=max(1, int(train_raw.get("steps", 300))),
            raw_lrs=_as_float_list(train_raw.get("raw_lrs", [1e-3, 3e-4]), key="train.raw_lrs"),
            latent_lrs=_as_float_list(train_raw.get("latent_lrs", [1e-2, 3e-3]), key="train.latent_lrs"),
            raw_weight_decay=float(train_raw.get("raw_weight_decay", 0.0)),
            latent_weight_decay=float(train_raw.get("latent_weight_decay", 0.0)),
            adam_beta1=float(train_raw.get("adam_beta1", 0.9)),
            adam_beta2=float(train_raw.get("adam_beta2", 0.95)),
            adam_eps=float(train_raw.get("adam_eps", 1e-9)),
            grad_clip_norm=float(train_raw.get("grad_clip_norm", 0.0)),
            amp=bool(train_raw.get("amp", True)),
            tf32=bool(train_raw.get("tf32", True)),
            log_every_steps=max(1, int(train_raw.get("log_every_steps", 10))),
            eval_every_steps=max(1, int(train_raw.get("eval_every_steps", 25))),
            recover_eps=max(0.0, float(train_raw.get("recover_eps", 0.01))),
        ),
        comet=CometConfig(
            enabled=bool(comet_raw.get("enabled", False)),
            api_key=str(comet_raw.get("api_key", "")).strip(),
            workspace=str(comet_raw.get("workspace", "")).strip(),
            project_name=str(comet_raw.get("project_name", "tinyvit_latent_h1")).strip() or "tinyvit_latent_h1",
            experiment_name=str(comet_raw.get("experiment_name", "")).strip(),
            offline_directory=str(comet_raw.get("offline_directory", "")).strip(),
            log_code=bool(comet_raw.get("log_code", False)),
            tags=_as_tags(comet_raw.get("tags", [])),
        ),
    )
    validate_run_config(resolved)
    return resolved, raw


def validate_run_config(cfg: RunConfig) -> None:
    if not cfg.source.run_dir:
        raise ValueError("source.run_dir is required")
    if cfg.source.train_anchor:
        if int(cfg.source.anchor_steps) <= 0:
            raise ValueError("source.anchor_steps must be > 0 when source.train_anchor=true")
        if float(cfg.source.anchor_lr) <= 0.0:
            raise ValueError("source.anchor_lr must be > 0 when source.train_anchor=true")
    if not cfg.search.epsilons:
        raise ValueError("search.epsilons must be non-empty")
    if not cfg.train.raw_lrs:
        raise ValueError("train.raw_lrs must be non-empty")
    if not cfg.train.latent_lrs:
        raise ValueError("train.latent_lrs must be non-empty")
    if cfg.model.hidden_dim % cfg.model.num_heads != 0:
        raise ValueError("model.hidden_dim must be divisible by model.num_heads")
    if cfg.model.image_size % cfg.model.patch_size != 0:
        raise ValueError("model.image_size must be divisible by model.patch_size")
