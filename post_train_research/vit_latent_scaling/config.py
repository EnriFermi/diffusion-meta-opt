from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from big_vae.eval.vit_tiny_latent_optimization import ViTTinyConfig


@dataclass(frozen=True, slots=True)
class ProfilePreset:
    name: str
    dataset: str
    model_size: str
    data_env_var: str
    default_data_dir: str
    image_size: int
    patch_size: int
    in_channels: int
    num_classes: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    batch_size: int
    eval_batch_size: int
    max_steps: int
    eval_every_steps: int


PROFILE_PRESETS: dict[str, ProfilePreset] = {
    "mnist_tiny": ProfilePreset("mnist_tiny", "mnist", "tiny", "MNIST_DATA_DIR", "./data/mnist", 28, 4, 1, 10, 64, 4, 4, 4.0, 128, 256, 7500, 375),
    "mnist_small": ProfilePreset("mnist_small", "mnist", "small", "MNIST_DATA_DIR", "./data/mnist", 28, 4, 1, 10, 128, 6, 4, 4.0, 128, 256, 7500, 375),
    "mnist_medium": ProfilePreset("mnist_medium", "mnist", "medium", "MNIST_DATA_DIR", "./data/mnist", 28, 4, 1, 10, 192, 8, 6, 4.0, 128, 256, 7500, 375),
    "cifar10_tiny": ProfilePreset("cifar10_tiny", "cifar10", "tiny", "CIFAR10_DATA_DIR", "./data/cifar10", 32, 4, 3, 10, 192, 6, 3, 4.0, 128, 256, 7500, 375),
    "cifar10_small": ProfilePreset("cifar10_small", "cifar10", "small", "CIFAR10_DATA_DIR", "./data/cifar10", 32, 4, 3, 10, 256, 8, 4, 4.0, 128, 256, 7500, 375),
    "cifar10_medium": ProfilePreset("cifar10_medium", "cifar10", "medium", "CIFAR10_DATA_DIR", "./data/cifar10", 32, 4, 3, 10, 384, 12, 6, 4.0, 128, 256, 7500, 375),
    "cifar10_50k": ProfilePreset("cifar10_50k", "cifar10", "50k", "CIFAR10_DATA_DIR", "./data/cifar10", 32, 8, 3, 10, 64, 1, 4, 2.0, 256, 512, 3000, 150),
    "mnist_mili": ProfilePreset("mnist_mili", "mnist", "mili", "MNIST_DATA_DIR", "./data/mnist", 32, 4, 3, 10, 128, 3, 4, 2.0, 128, 256, 3000, 150),
    "cifar_mili": ProfilePreset("cifar_mili", "cifar10", "mili", "CIFAR10_DATA_DIR", "./data/cifar10", 32, 4, 3, 10, 128, 3, 4, 2.0, 128, 256, 3000, 150),
    "imagenet_tiny": ProfilePreset("imagenet_tiny", "imagenet", "tiny", "IMAGENET_DATA_DIR", "./data/imagenet", 224, 16, 3, 1000, 192, 12, 3, 4.0, 64, 128, 20000, 1000),
    "imagenet_small": ProfilePreset("imagenet_small", "imagenet", "small", "IMAGENET_DATA_DIR", "./data/imagenet", 224, 16, 3, 1000, 384, 12, 6, 4.0, 64, 128, 20000, 1000),
    "imagenet_base": ProfilePreset("imagenet_base", "imagenet", "base", "IMAGENET_DATA_DIR", "./data/imagenet", 224, 16, 3, 1000, 768, 12, 12, 4.0, 32, 64, 20000, 1000),
}


@dataclass(slots=True)
class StorageConfig:
    root_dir: str
    shared_checkpoint_root_dir: str
    shared_checkpoint_label: str
    run_label: str
    notes: str
    tags: list[str]
    checkpoint_every_steps: int
    save_best: bool
    save_latest: bool
    save_final: bool


@dataclass(slots=True)
class DataConfig:
    dataset: str
    data_dir: str
    download: bool
    train_subset: int
    test_subset: int
    batch_size: int
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
    kind: str
    big_vae_checkpoint: str
    big_vae_decode: str
    big_vae_tile_T_patches: int
    big_vae_tile_d_out: int
    big_vae_latent_parameterization: str
    big_vae_latent_noise_std: float
    big_vae_decoder_adapter: str
    big_vae_decoder_adapter_checkpoint: str
    big_vae_decoder_adapter_require_checkpoint_match: bool


@dataclass(slots=True)
class InitConfig:
    kind: str
    fresh_latent_mode: str
    random_init_std: float
    source_run_dir: str
    source_checkpoint: str
    source_prefer_direct_latent: bool
    diffusion_prior_checkpoint: str
    diffusion_prior_steps: int
    diffusion_prior_sampler: str
    diffusion_prior_eta: float
    calibration_batches: int


@dataclass(slots=True)
class TrainConfig:
    device: str
    seed: int
    epochs: int
    max_steps: int
    optimizer_name: str
    optimizer_kwargs: dict[str, Any]
    lr: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    grad_clip_norm: float
    label_smoothing: float
    amp: bool
    tf32: bool
    compile: bool
    latent_lr_scheduler: str
    latent_lr_floor_ratio: float
    latent_lr_decay_steps: int


@dataclass(slots=True)
class LoggingConfig:
    log_every_steps: int
    eval_every_steps: int
    latent_debug_metrics: bool
    latent_jacobian_eps: float
    latent_jacobian_probes: int


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
    profile: ProfilePreset
    storage: StorageConfig
    data: DataConfig
    model: ModelConfig
    setup: SetupConfig
    init: InitConfig
    train: TrainConfig
    logging: LoggingConfig
    comet: CometConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profile"] = asdict(self.profile)
        return payload

    @property
    def run_label(self) -> str:
        if self.storage.run_label.strip():
            return self.storage.run_label.strip()
        init_label = self.init.kind
        if self.setup.kind == "latent" and self.init.kind == "fresh":
            init_label = f"fresh_{self.init.fresh_latent_mode}"
        return f"{self.profile.name}__{self.setup.kind}__{init_label}"

    @property
    def shared_checkpoint_label(self) -> str:
        if self.storage.shared_checkpoint_label.strip():
            return self.storage.shared_checkpoint_label.strip()
        return self.run_label

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


def _resolve_data_dir(profile: ProfilePreset, raw_value: Any) -> str:
    candidate = str(raw_value or "").strip()
    if candidate:
        return candidate
    env_value = os.environ.get(profile.data_env_var, "").strip()
    if env_value:
        return env_value
    return str(profile.default_data_dir)


def _int_override(value: Any, default: int) -> int:
    parsed = int(value or 0)
    return parsed if parsed > 0 else int(default)


def _float_override(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    parsed = float(value)
    return float(default) if parsed == 0.0 else parsed


def _model_value(section: dict[str, Any], key: str, default: int | float) -> int | float:
    value = section.get(key, None)
    if value in {None, 0, 0.0, ""}:
        return default
    return type(default)(value)


def build_run_config(cfg: DictConfig) -> tuple[RunConfig, dict[str, Any]]:
    raw = _cfg_dict(cfg)
    experiment = _nested(raw, "experiment")
    storage_raw = _nested(raw, "storage")
    data_raw = _nested(raw, "data")
    model_raw = _nested(raw, "model")
    setup_raw = _nested(raw, "setup")
    init_raw = _nested(raw, "init")
    train_raw = _nested(raw, "train")
    logging_raw = _nested(raw, "logging")
    telemetry_raw = _nested(raw, "telemetry")
    comet_raw = _nested(telemetry_raw, "comet")

    profile_name = str(experiment.get("profile", "cifar10_50k")).strip()
    if profile_name not in PROFILE_PRESETS:
        raise KeyError(f"Unknown vit_latent_scaling profile: {profile_name}")
    profile = PROFILE_PRESETS[profile_name]

    storage = StorageConfig(
        root_dir=str(storage_raw.get("root_dir", "./artifacts/big_vae/eval/vit_latent_scaling")).strip(),
        shared_checkpoint_root_dir=str(storage_raw.get("shared_checkpoint_root_dir", "")).strip(),
        shared_checkpoint_label=str(storage_raw.get("shared_checkpoint_label", "")).strip(),
        run_label=str(experiment.get("run_label", "")).strip(),
        notes=str(experiment.get("notes", "")).strip(),
        tags=_as_tags(experiment.get("tags", [])),
        checkpoint_every_steps=max(1, int(storage_raw.get("checkpoint_every_steps", 250))),
        save_best=bool(storage_raw.get("save_best", True)),
        save_latest=bool(storage_raw.get("save_latest", True)),
        save_final=bool(storage_raw.get("save_final", True)),
    )

    data = DataConfig(
        dataset=str(data_raw.get("dataset", profile.dataset)).strip() or str(profile.dataset),
        data_dir=_resolve_data_dir(profile, data_raw.get("data_dir", "")),
        download=bool(data_raw.get("download", profile.dataset != "imagenet")),
        train_subset=max(0, int(data_raw.get("train_subset", 0))),
        test_subset=max(0, int(data_raw.get("test_subset", 0))),
        batch_size=_int_override(data_raw.get("batch_size", 0), profile.batch_size),
        eval_batch_size=_int_override(data_raw.get("eval_batch_size", 0), profile.eval_batch_size),
        num_workers=max(0, int(data_raw.get("num_workers", 4))),
    )

    model = ModelConfig(
        image_size=int(_model_value(model_raw, "image_size", profile.image_size)),
        patch_size=int(_model_value(model_raw, "patch_size", profile.patch_size)),
        in_channels=int(_model_value(model_raw, "in_channels", profile.in_channels)),
        num_classes=int(_model_value(model_raw, "num_classes", profile.num_classes)),
        hidden_dim=int(_model_value(model_raw, "hidden_dim", profile.hidden_dim)),
        depth=int(_model_value(model_raw, "depth", profile.depth)),
        num_heads=int(_model_value(model_raw, "num_heads", profile.num_heads)),
        mlp_ratio=float(_model_value(model_raw, "mlp_ratio", profile.mlp_ratio)),
        dropout=float(model_raw.get("dropout", 0.0) or 0.0),
        attention_dropout=float(model_raw.get("attention_dropout", 0.0) or 0.0),
    )

    decoder_adapter = str(setup_raw.get("big_vae_decoder_adapter", "identity")).strip().lower()
    decoder_adapter_checkpoint = str(setup_raw.get("big_vae_decoder_adapter_checkpoint", "") or "").strip()
    if decoder_adapter_checkpoint and decoder_adapter in {"", "identity", "none", "off", "false"}:
        decoder_adapter = "latent_flattening_flow"
    setup = SetupConfig(
        kind=str(setup_raw.get("kind", "latent")).strip().lower(),
        big_vae_checkpoint=str(setup_raw.get("big_vae_checkpoint", "")).strip(),
        big_vae_decode=str(setup_raw.get("big_vae_decode", "weights")).strip().lower(),
        big_vae_tile_T_patches=max(1, int(setup_raw.get("big_vae_tile_T_patches", 4))),
        big_vae_tile_d_out=max(1, int(setup_raw.get("big_vae_tile_d_out", 64))),
        big_vae_latent_parameterization=str(
            setup_raw.get("big_vae_latent_parameterization", "euclidean")
        ).strip().lower(),
        big_vae_latent_noise_std=max(0.0, float(setup_raw.get("big_vae_latent_noise_std", 0.0))),
        big_vae_decoder_adapter=decoder_adapter,
        big_vae_decoder_adapter_checkpoint=decoder_adapter_checkpoint,
        big_vae_decoder_adapter_require_checkpoint_match=bool(
            setup_raw.get("big_vae_decoder_adapter_require_checkpoint_match", False)
        ),
    )

    init = InitConfig(
        kind=str(init_raw.get("kind", "diffusion_prior")).strip().lower(),
        fresh_latent_mode=str(init_raw.get("fresh_latent_mode", "random")).strip().lower(),
        random_init_std=max(0.0, float(init_raw.get("random_init_std", 0.02))),
        source_run_dir=str(init_raw.get("source_run_dir", "")).strip(),
        source_checkpoint=str(init_raw.get("source_checkpoint", "best")).strip() or "best",
        source_prefer_direct_latent=bool(init_raw.get("source_prefer_direct_latent", True)),
        diffusion_prior_checkpoint=str(init_raw.get("diffusion_prior_checkpoint", "")).strip(),
        diffusion_prior_steps=max(1, int(init_raw.get("diffusion_prior_steps", 50))),
        diffusion_prior_sampler=str(init_raw.get("diffusion_prior_sampler", "ddim")).strip().lower(),
        diffusion_prior_eta=float(init_raw.get("diffusion_prior_eta", 0.0)),
        calibration_batches=max(1, int(init_raw.get("calibration_batches", 1))),
    )

    optimizer_kwargs = train_raw.get("optimizer_kwargs", {})
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if not isinstance(optimizer_kwargs, dict):
        raise TypeError("train.optimizer_kwargs must be a mapping")
    train = TrainConfig(
        device=str(train_raw.get("device", "auto")).strip(),
        seed=int(train_raw.get("seed", 42)),
        epochs=max(1, int(train_raw.get("epochs", 30))),
        max_steps=_int_override(train_raw.get("max_steps", 0), profile.max_steps),
        optimizer_name=str(train_raw.get("optimizer_name", "AdamW")).strip(),
        optimizer_kwargs=dict(optimizer_kwargs),
        lr=float(train_raw.get("lr", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 0.0)),
        adam_beta1=float(train_raw.get("adam_beta1", 0.9)),
        adam_beta2=float(train_raw.get("adam_beta2", 0.999)),
        adam_eps=float(train_raw.get("adam_eps", 1e-8)),
        grad_clip_norm=float(train_raw.get("grad_clip_norm", 0.0)),
        label_smoothing=float(train_raw.get("label_smoothing", 0.0)),
        amp=bool(train_raw.get("amp", True)),
        tf32=bool(train_raw.get("tf32", True)),
        compile=bool(train_raw.get("compile", False)),
        latent_lr_scheduler=str(train_raw.get("latent_lr_scheduler", "constant")).strip().lower(),
        latent_lr_floor_ratio=float(train_raw.get("latent_lr_floor_ratio", 0.1)),
        latent_lr_decay_steps=max(0, int(train_raw.get("latent_lr_decay_steps", 0))),
    )

    log_cfg = LoggingConfig(
        log_every_steps=max(1, int(logging_raw.get("log_every_steps", 50))),
        eval_every_steps=_int_override(logging_raw.get("eval_every_steps", 0), profile.eval_every_steps),
        latent_debug_metrics=bool(logging_raw.get("latent_debug_metrics", True)),
        latent_jacobian_eps=float(logging_raw.get("latent_jacobian_eps", 1e-3)),
        latent_jacobian_probes=max(1, int(logging_raw.get("latent_jacobian_probes", 16))),
    )

    comet = CometConfig(
        enabled=bool(comet_raw.get("enabled", False)),
        api_key=str(comet_raw.get("api_key", "")).strip(),
        workspace=str(comet_raw.get("workspace", "")).strip(),
        project_name=str(comet_raw.get("project_name", "vit_latent_scaling")).strip() or "vit_latent_scaling",
        experiment_name=str(comet_raw.get("experiment_name", "")).strip(),
        offline_directory=str(comet_raw.get("offline_directory", "")).strip(),
        log_code=bool(comet_raw.get("log_code", False)),
        tags=_as_tags(comet_raw.get("tags", [])),
    )

    resolved = RunConfig(
        profile=profile,
        storage=storage,
        data=data,
        model=model,
        setup=setup,
        init=init,
        train=train,
        logging=log_cfg,
        comet=comet,
    )
    validate_run_config(resolved)
    return resolved, raw


def validate_run_config(cfg: RunConfig) -> None:
    if cfg.setup.kind not in {"raw", "latent"}:
        raise ValueError("setup.kind must be 'raw' or 'latent'")
    if cfg.setup.kind == "latent" and not cfg.setup.big_vae_checkpoint:
        raise ValueError("setup.big_vae_checkpoint is required for latent setup")
    if cfg.setup.kind == "raw" and cfg.init.kind == "diffusion_prior":
        raise ValueError("init.kind='diffusion_prior' is only valid for latent setup")
    if cfg.setup.big_vae_decode not in {"weights", "all"}:
        raise ValueError("setup.big_vae_decode must be 'weights' or 'all'")
    if cfg.setup.big_vae_latent_parameterization not in {"euclidean", "sphere"}:
        raise ValueError("setup.big_vae_latent_parameterization must be 'euclidean' or 'sphere'")
    if cfg.setup.big_vae_decoder_adapter not in {"", "identity", "none", "off", "false", "latent_flattening_flow", "flow", "ir_smoothing", "latent_smoothing"}:
        raise ValueError("setup.big_vae_decoder_adapter must be identity or latent_flattening_flow")
    if (
        cfg.setup.kind == "latent"
        and cfg.setup.big_vae_decoder_adapter not in {"", "identity", "none", "off", "false"}
        and not cfg.setup.big_vae_decoder_adapter_checkpoint
    ):
        raise ValueError("setup.big_vae_decoder_adapter_checkpoint is required when decoder adapter is enabled")
    if (
        cfg.setup.kind == "latent"
        and cfg.setup.big_vae_decoder_adapter not in {"", "identity", "none", "off", "false"}
        and cfg.init.kind != "diffusion_prior"
    ):
        raise ValueError("setup.big_vae_decoder_adapter currently requires init.kind='diffusion_prior'")
    if cfg.init.kind not in {"fresh", "source", "diffusion_prior"}:
        raise ValueError("init.kind must be one of {'fresh','source','diffusion_prior'}")
    if cfg.init.fresh_latent_mode not in {"base", "random"}:
        raise ValueError("init.fresh_latent_mode must be 'base' or 'random'")
    if cfg.init.kind == "source" and not cfg.init.source_run_dir:
        raise ValueError("init.source_run_dir is required when init.kind='source'")
    if cfg.init.kind == "diffusion_prior" and not cfg.init.diffusion_prior_checkpoint:
        raise ValueError("init.diffusion_prior_checkpoint is required when init.kind='diffusion_prior'")
    if cfg.train.latent_lr_scheduler not in {"constant", "cosine_decay_to_floor"}:
        raise ValueError("train.latent_lr_scheduler must be 'constant' or 'cosine_decay_to_floor'")
    if cfg.train.latent_lr_scheduler == "cosine_decay_to_floor" and cfg.train.latent_lr_decay_steps <= 0:
        raise ValueError("train.latent_lr_decay_steps must be > 0 for cosine_decay_to_floor")
    if cfg.data.dataset == "imagenet" and cfg.data.download:
        raise ValueError("ImageNet download is not supported; set data.download=false")
