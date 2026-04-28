from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compare_vit_tiny_latent_optimization import (
    BigVAELatentTensorStore,
    EvalMetrics,
    FunctionalViTTiny,
    ViTTinyConfig,
    autocast_context,
    count_trainable_parameters,
    load_frozen_big_vae_decoder,
    make_initial_tensors,
    resolve_device,
    seed_everything,
)
from models.weight_quantile_vae import BigWeightVAE
from training.big_vae_latent_diffusion import (
    load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint,
    load_frozen_layer_latent_diffusion_prior,
)


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _repeat_channel_stats(values: tuple[float, ...], *, count: int) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError(f"channel count must be positive, got {count}")
    if len(values) == count:
        return tuple(float(item) for item in values)
    if len(values) == 1:
        return tuple(float(values[0]) for _ in range(int(count)))
    raise ValueError(f"cannot adapt stats of length {len(values)} to channel count {count}")


@dataclass(slots=True)
class ScalingRunConfig:
    output_dir: str
    dataset: str
    model_size: str
    setup: str
    data_dir: str
    download: bool
    device: str
    seed: int
    epochs: int
    max_steps: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    train_subset: int
    test_subset: int
    optimizer_name: str
    optimizer_kwargs: dict[str, Any]
    lr: float
    latent_lr_scheduler: str
    latent_lr_floor_ratio: float
    latent_lr_decay_steps: int
    latent_debug_log_deltas: bool
    latent_debug_jacobian_eps: float
    latent_debug_jacobian_probes: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    label_smoothing: float
    grad_clip_norm: float
    amp: bool
    tf32: bool
    compile: bool
    log_every_steps: int
    eval_every_steps: int
    save_checkpoints: bool
    raw_checkpoint: str
    latent_checkpoint: str
    big_vae_checkpoint: str
    big_vae_latent_init: str
    big_vae_diffusion_prior_checkpoint: str
    big_vae_diffusion_prior_steps: int
    big_vae_diffusion_prior_sampler: str
    big_vae_diffusion_prior_eta: float
    big_vae_decode: str
    big_vae_tile_T_patches: int
    big_vae_tile_d_out: int
    big_vae_encoder_context_rows: int
    big_vae_encoder_context_std: float
    big_vae_encoder_batch_size: int
    big_vae_init_calibration_batches: int
    latent_weight_decay: float
    raw_init_steps: int


def validate_scaling_run_config(cfg: ScalingRunConfig) -> None:
    setup = str(cfg.setup).strip().lower()
    has_latent_checkpoint = bool(str(cfg.latent_checkpoint).strip())
    if setup == "raw" and str(cfg.latent_checkpoint).strip():
        raise ValueError("--latent-checkpoint is only valid for setup=latent")
    if setup == "latent":
        if not str(cfg.big_vae_checkpoint).strip():
            raise ValueError("--big-vae-checkpoint is required for setup=latent")
        if not has_latent_checkpoint:
            latent_init = str(cfg.big_vae_latent_init).strip().lower()
            if latent_init == "encoded" and not str(cfg.raw_checkpoint).strip():
                raise ValueError("--raw-checkpoint is required when --big-vae-latent-init=encoded")
            if latent_init == "diffusion_prior" and not str(
                cfg.big_vae_diffusion_prior_checkpoint
            ).strip():
                raise ValueError(
                    "--big-vae-diffusion-prior-checkpoint is required when --big-vae-latent-init=diffusion_prior"
                )
    latent_lr_scheduler = str(cfg.latent_lr_scheduler).strip().lower()
    if latent_lr_scheduler not in {"constant", "cosine_decay_to_floor"}:
        raise ValueError(
            "--latent-lr-scheduler must be one of {'constant', 'cosine_decay_to_floor'}"
        )
    if not (0.0 < float(cfg.latent_lr_floor_ratio) <= 1.0):
        raise ValueError("--latent-lr-floor-ratio must be in the interval (0, 1]")
    if latent_lr_scheduler == "cosine_decay_to_floor" and int(cfg.latent_lr_decay_steps) <= 0:
        raise ValueError("--latent-lr-decay-steps must be a positive integer")
    if float(cfg.latent_debug_jacobian_eps) <= 0.0:
        raise ValueError("--latent-debug-jacobian-eps must be > 0")
    if int(cfg.latent_debug_jacobian_probes) <= 0:
        raise ValueError("--latent-debug-jacobian-probes must be a positive integer")
    if not str(cfg.optimizer_name).strip():
        raise ValueError("--optimizer-name must be non-empty")
    if not isinstance(cfg.optimizer_kwargs, dict):
        raise ValueError("--optimizer-kwargs must be a mapping")
    if str(cfg.dataset).strip().lower() == "imagenet" and bool(cfg.download):
        cfg.download = False


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> EvalMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss.detach().cpu().item())
        correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu().item())
        examples += int(labels.numel())
    return EvalMetrics(
        loss=loss_sum / max(1, examples),
        accuracy=correct / max(1, examples),
        examples=examples,
    )


def resolve_optimizer_class(name: str) -> type[torch.optim.Optimizer]:
    candidate = str(name).strip()
    if not candidate:
        raise ValueError("Optimizer name must be non-empty")
    normalized = candidate.casefold()
    for attr_name in dir(torch.optim):
        attr = getattr(torch.optim, attr_name)
        if not inspect.isclass(attr):
            continue
        if not issubclass(attr, torch.optim.Optimizer):
            continue
        if attr is torch.optim.Optimizer:
            continue
        if attr_name.casefold() == normalized:
            return attr
    raise ValueError(f"Unknown torch optimizer: {name}")


def _parse_optimizer_kwarg_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none" or lowered == "null":
            return None
        return raw_value


def parse_optimizer_kwargs_cli(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --optimizer-kwarg value {item!r}; expected key=value"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --optimizer-kwarg value {item!r}; empty key")
        parsed[key] = _parse_optimizer_kwarg_value(raw_value.strip())
    return parsed


def build_optimizer(
    parameters: list[torch.nn.Parameter],
    *,
    cfg: ScalingRunConfig,
    setup: str,
) -> torch.optim.Optimizer:
    optimizer_cls = resolve_optimizer_class(cfg.optimizer_name)
    weight_decay = float(cfg.latent_weight_decay if setup == "latent" else cfg.weight_decay)
    signature = inspect.signature(optimizer_cls.__init__)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name not in {"self", "params"}
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    optimizer_kwargs: dict[str, Any] = {}
    if "lr" in accepted:
        optimizer_kwargs["lr"] = float(cfg.lr)
    if "weight_decay" in accepted:
        optimizer_kwargs["weight_decay"] = weight_decay
    if "betas" in accepted:
        optimizer_kwargs["betas"] = (float(cfg.adam_beta1), float(cfg.adam_beta2))
    if "eps" in accepted:
        optimizer_kwargs["eps"] = float(cfg.adam_eps)

    explicit_kwargs = dict(cfg.optimizer_kwargs)
    if not accepts_var_kwargs:
        unsupported = sorted(set(explicit_kwargs) - accepted)
        if unsupported:
            raise ValueError(
                f"Optimizer {optimizer_cls.__name__} does not accept kwargs: {unsupported}"
            )
    optimizer_kwargs.update(explicit_kwargs)
    return optimizer_cls(parameters, **optimizer_kwargs)


def _sorted_latent_slot_keys(store: BigVAELatentTensorStore) -> list[str]:
    return sorted(str(key) for key in store.latent_slots.keys())


@torch.no_grad()
def flatten_latent_slots(store: BigVAELatentTensorStore) -> torch.Tensor:
    keys = _sorted_latent_slot_keys(store)
    if not keys:
        return torch.zeros(0, device=store.big_vae.latent_base.device, dtype=store.big_vae.latent_base.dtype)
    return torch.cat([store.latent_slots[key].detach().reshape(-1) for key in keys], dim=0)


@torch.no_grad()
def load_flat_latent_slots_(store: BigVAELatentTensorStore, flat_latents: torch.Tensor) -> None:
    keys = _sorted_latent_slot_keys(store)
    offset = 0
    for key in keys:
        target = store.latent_slots[key]
        numel = int(target.numel())
        value = flat_latents[offset : offset + numel].view_as(target).to(device=target.device, dtype=target.dtype)
        target.copy_(value)
        offset += numel
    if offset != int(flat_latents.numel()):
        raise ValueError(
            f"flat latent vector length mismatch: consumed {offset}, total {int(flat_latents.numel())}"
        )


@torch.no_grad()
def flatten_decoded_bigvae_weights(store: BigVAELatentTensorStore) -> torch.Tensor:
    decoded = store.decode_all_matrices()
    if not decoded:
        return torch.zeros(0, device=store.big_vae.latent_base.device, dtype=store.big_vae.latent_base.dtype)
    return torch.cat([decoded[name].detach().reshape(-1) for name in sorted(decoded.keys())], dim=0)


@torch.no_grad()
def estimate_decoder_effective_jacobian_norm(
    store: BigVAELatentTensorStore,
    *,
    eps: float,
    num_probes: int,
) -> dict[str, float]:
    z0 = flatten_latent_slots(store)
    if int(z0.numel()) == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    w0 = flatten_decoded_bigvae_weights(store)
    norms: list[float] = []
    tiny = torch.finfo(z0.dtype).tiny
    for _ in range(int(num_probes)):
        v = torch.randn_like(z0)
        v = v / v.norm().clamp_min(tiny)
        load_flat_latent_slots_(store, z0 + float(eps) * v)
        w1 = flatten_decoded_bigvae_weights(store)
        norms.append(float((w1 - w0).norm().item() / float(eps)))
    load_flat_latent_slots_(store, z0)
    values = np.asarray(norms, dtype=np.float64)
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def resolve_planned_train_steps(cfg: ScalingRunConfig, *, steps_per_epoch: int) -> int:
    if int(cfg.max_steps) > 0:
        return max(1, int(cfg.max_steps))
    return max(1, int(cfg.epochs) * max(1, int(steps_per_epoch)))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    cfg: ScalingRunConfig,
    setup: str,
    planned_train_steps: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    scheduler_name = str(cfg.latent_lr_scheduler).strip().lower()
    if str(setup).strip().lower() != "latent" or scheduler_name == "constant":
        return None, "constant"
    if scheduler_name != "cosine_decay_to_floor":
        raise ValueError(f"Unsupported latent lr scheduler: {cfg.latent_lr_scheduler}")

    floor_ratio = float(cfg.latent_lr_floor_ratio)
    configured_decay_steps = int(cfg.latent_lr_decay_steps)
    decay_steps = max(1, min(int(planned_train_steps), configured_decay_steps))

    def _lr_lambda(step_index: int) -> float:
        progress = min(1.0, max(0.0, float(step_index) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(floor_ratio + (1.0 - floor_ratio) * cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    description = (
        "cosine_decay_to_floor("
        f"floor_ratio={floor_ratio:g}, "
        f"configured_decay_steps={configured_decay_steps}, "
        f"decay_steps={decay_steps}/{int(planned_train_steps)})"
    )
    return scheduler, description


def maybe_subset(dataset: Any, limit: int) -> Any:
    if int(limit) <= 0:
        return dataset
    return Subset(dataset, list(range(min(int(limit), len(dataset)))))


def build_mnist_loaders(cfg: ScalingRunConfig, device: torch.device, vit_cfg: ViTTinyConfig) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for MNIST") from exc

    train_steps = []
    eval_steps = []
    if int(vit_cfg.image_size) != 28:
        train_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
        eval_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
    if int(vit_cfg.in_channels) not in {1, 3}:
        raise ValueError(f"MNIST presets support only in_channels in {{1,3}}, got {vit_cfg.in_channels}")
    if int(vit_cfg.in_channels) == 3:
        train_steps.append(transforms.Grayscale(num_output_channels=3))
        eval_steps.append(transforms.Grayscale(num_output_channels=3))
    mean = _repeat_channel_stats(MNIST_MEAN, count=int(vit_cfg.in_channels))
    std = _repeat_channel_stats(MNIST_STD, count=int(vit_cfg.in_channels))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.MNIST(
        root=str(cfg.data_dir),
        train=True,
        download=bool(cfg.download),
        transform=transforms.Compose(train_steps),
    )
    test_set = datasets.MNIST(
        root=str(cfg.data_dir),
        train=False,
        download=bool(cfg.download),
        transform=transforms.Compose(eval_steps),
    )
    return make_loaders(cfg, device, maybe_subset(train_set, cfg.train_subset), maybe_subset(test_set, cfg.test_subset))


def build_cifar10_loaders(
    cfg: ScalingRunConfig,
    device: torch.device,
    vit_cfg: ViTTinyConfig,
) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for CIFAR-10") from exc

    train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    eval_steps = []
    if int(vit_cfg.in_channels) not in {1, 3}:
        raise ValueError(f"CIFAR-10 presets support only in_channels in {{1,3}}, got {vit_cfg.in_channels}")
    if int(vit_cfg.in_channels) == 1:
        train_steps.append(transforms.Grayscale(num_output_channels=1))
        eval_steps.append(transforms.Grayscale(num_output_channels=1))
    if int(vit_cfg.image_size) != 32:
        train_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
        eval_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
    mean = _repeat_channel_stats(CIFAR10_MEAN, count=int(vit_cfg.in_channels))
    std = _repeat_channel_stats(CIFAR10_STD, count=int(vit_cfg.in_channels))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.CIFAR10(
        root=str(cfg.data_dir),
        train=True,
        download=bool(cfg.download),
        transform=transforms.Compose(train_steps),
    )
    test_set = datasets.CIFAR10(
        root=str(cfg.data_dir),
        train=False,
        download=bool(cfg.download),
        transform=transforms.Compose(eval_steps),
    )
    return make_loaders(cfg, device, maybe_subset(train_set, cfg.train_subset), maybe_subset(test_set, cfg.test_subset))


def build_imagenet_loaders(
    cfg: ScalingRunConfig,
    device: torch.device,
    vit_cfg: ViTTinyConfig,
) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for ImageNet") from exc

    data_root = Path(cfg.data_dir).expanduser()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(int(vit_cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_resize = max(int(vit_cfg.image_size), int(round(float(vit_cfg.image_size) * 256.0 / 224.0)))
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(int(vit_cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        train_set = datasets.ImageFolder(str(train_dir), transform=train_transform)
        test_set = datasets.ImageFolder(str(val_dir), transform=eval_transform)
    else:
        train_set = datasets.ImageNet(root=str(data_root), split="train", transform=train_transform)
        test_set = datasets.ImageNet(root=str(data_root), split="val", transform=eval_transform)

    return make_loaders(cfg, device, maybe_subset(train_set, cfg.train_subset), maybe_subset(test_set, cfg.test_subset))


def make_loaders(cfg: ScalingRunConfig, device: torch.device, train_set: Any, test_set: Any) -> tuple[DataLoader, DataLoader]:
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(cfg.seed))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(cfg.num_workers) > 0,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(cfg.num_workers) > 0,
    )
    return train_loader, test_loader


def build_loaders(cfg: ScalingRunConfig, device: torch.device, vit_cfg: ViTTinyConfig) -> tuple[DataLoader, DataLoader]:
    dataset = str(cfg.dataset).strip().lower()
    if dataset == "mnist":
        return build_mnist_loaders(cfg, device, vit_cfg)
    if dataset == "cifar10":
        return build_cifar10_loaders(cfg, device, vit_cfg)
    if dataset == "imagenet":
        return build_imagenet_loaders(cfg, device, vit_cfg)
    raise ValueError(f"unsupported dataset: {cfg.dataset!r}")


def export_named_tensors(model: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    target = getattr(model, "_orig_mod", model)
    return {name: target.w(name).detach().cpu().clone() for name in names}


def load_raw_checkpoint(path: str, vit_cfg: ViTTinyConfig) -> tuple[dict[str, torch.Tensor], int]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"raw checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "named_tensors" not in payload:
        raise KeyError(f"raw checkpoint must contain named_tensors: {checkpoint_path}")
    named_tensors = payload["named_tensors"]
    if not isinstance(named_tensors, dict):
        raise TypeError(f"named_tensors must be a dict in {checkpoint_path}")

    expected = make_initial_tensors(vit_cfg, seed=0)
    loaded: dict[str, torch.Tensor] = {}
    for name, expected_tensor in expected.items():
        if name not in named_tensors:
            raise KeyError(f"raw checkpoint missing tensor {name!r}: {checkpoint_path}")
        tensor = named_tensors[name].detach().cpu().to(dtype=torch.float32)
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"raw checkpoint tensor shape mismatch for {name}: "
                f"got {tuple(tensor.shape)}, expected {tuple(expected_tensor.shape)}"
            )
        loaded[name] = tensor.contiguous()
    return loaded, int(payload.get("step", 0))


def load_latent_checkpoint(path: str) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"latent checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"latent checkpoint payload must be a dict: {checkpoint_path}")
    if str(payload.get("setup", "latent")).strip().lower() != "latent":
        raise ValueError(f"latent checkpoint must have setup='latent': {checkpoint_path}")
    latent_slots = payload.get("latent_slots")
    if not isinstance(latent_slots, dict):
        raise KeyError(f"latent checkpoint must contain latent_slots dict: {checkpoint_path}")
    return payload


def infer_latent_space_from_checkpoint_payload(payload: dict[str, Any]) -> str:
    explicit = str(payload.get("latent_space", "") or "").strip().lower()
    if explicit in {"encoder_slots", "decoder_z"}:
        return explicit
    run_cfg = payload.get("run_config", {})
    if isinstance(run_cfg, dict):
        run_latent_space = str(run_cfg.get("big_vae_latent_space", "") or "").strip().lower()
        if run_latent_space in {"encoder_slots", "decoder_z"}:
            return run_latent_space
        run_latent_init = str(run_cfg.get("big_vae_latent_init", "") or "").strip().lower()
        if run_latent_init == "diffusion_prior":
            return "decoder_z"
    return "encoder_slots"


def load_latent_slots_from_checkpoint(
    latent_slots: nn.ParameterDict,
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    source_state = checkpoint_payload.get("latent_slots", {})
    if not isinstance(source_state, dict):
        raise TypeError("checkpoint_payload['latent_slots'] must be a dict")

    current_state = latent_slots.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    skipped_shape_keys: list[str] = []

    for key, source_value in source_state.items():
        if key not in current_state:
            unexpected_keys.append(str(key))
            continue
        current_value = current_state[key]
        source_tensor = source_value.detach().cpu().to(dtype=current_value.dtype)
        if tuple(source_tensor.shape) != tuple(current_value.shape):
            skipped_shape_keys.append(str(key))
            continue
        compatible_state[str(key)] = source_tensor.contiguous()

    missing_keys = sorted(key for key in current_state.keys() if key not in compatible_state)
    if not compatible_state:
        raise RuntimeError(
            "latent checkpoint did not match any current latent slots; "
            "check model architecture, decode mode, and BigVAE tiling config"
        )

    latent_slots.load_state_dict(compatible_state, strict=False)
    return {
        "loaded_keys": sorted(compatible_state.keys()),
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(unexpected_keys),
        "skipped_shape_keys": sorted(skipped_shape_keys),
        "checkpoint_step": int(checkpoint_payload.get("step", 0)),
        "checkpoint_path": str(checkpoint_payload.get("_checkpoint_path", "")),
    }


def build_model(
    cfg: ScalingRunConfig,
    vit_cfg: ViTTinyConfig,
    initial_tensors: dict[str, torch.Tensor],
    *,
    big_vae_decoder: BigWeightVAE | None,
    big_vae_diffusion_prior: Any | None,
    big_vae_latent_init_override: str | None = None,
    big_vae_latent_space_override: str | None = None,
) -> nn.Module:
    if cfg.setup == "raw":
        return FunctionalViTTiny(vit_cfg, initial_tensors, parameter_mode="direct")
    if cfg.setup != "latent":
        raise ValueError(f"setup must be 'raw' or 'latent', got {cfg.setup!r}")
    if big_vae_decoder is None:
        raise ValueError("--big-vae-checkpoint is required for setup=latent")
    return FunctionalViTTiny(
        vit_cfg,
        initial_tensors,
        parameter_mode="bigvae_latent",
        big_vae=big_vae_decoder,
        big_vae_latent_init=str(
            big_vae_latent_init_override if big_vae_latent_init_override is not None else cfg.big_vae_latent_init
        ),
        big_vae_latent_space=big_vae_latent_space_override,
        big_vae_diffusion_prior=big_vae_diffusion_prior,
        big_vae_diffusion_prior_steps=int(cfg.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(cfg.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(cfg.big_vae_diffusion_prior_eta),
        big_vae_decode=str(cfg.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.big_vae_tile_d_out),
        big_vae_encoder_context_rows=int(cfg.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(cfg.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(cfg.big_vae_encoder_batch_size),
    )


def collect_calibration_images(train_loader: DataLoader, *, num_batches: int) -> torch.Tensor | None:
    batch_limit = max(0, int(num_batches))
    if batch_limit <= 0:
        return None
    images: list[torch.Tensor] = []
    for batch_idx, (batch_images, _batch_labels) in enumerate(train_loader):
        images.append(batch_images.detach().cpu())
        if batch_idx + 1 >= batch_limit:
            break
    if not images:
        return None
    return torch.cat(images, dim=0).contiguous()


def train_once(cfg: ScalingRunConfig, vit_cfg: ViTTinyConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"run": asdict(cfg), "vit": asdict(vit_cfg)})

    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.benchmark = True

    seed_everything(int(cfg.seed))
    train_loader, test_loader = build_loaders(cfg, device, vit_cfg)
    latent_checkpoint_path = str(cfg.latent_checkpoint).strip()
    use_latent_checkpoint_init = cfg.setup == "latent" and bool(latent_checkpoint_path)
    latent_checkpoint_payload: dict[str, Any] | None = None
    calibration_images = None
    if use_latent_checkpoint_init:
        latent_checkpoint_payload = load_latent_checkpoint(latent_checkpoint_path)
        latent_checkpoint_payload["_checkpoint_path"] = latent_checkpoint_path
        print(f"[latent] will initialize latent slots from checkpoint: {latent_checkpoint_path}", flush=True)
    latent_checkpoint_space = (
        infer_latent_space_from_checkpoint_payload(latent_checkpoint_payload)
        if latent_checkpoint_payload is not None
        else None
    )
    use_diffusion_prior_init = (
        cfg.setup == "latent"
        and str(cfg.big_vae_latent_init).strip().lower() == "diffusion_prior"
        and not use_latent_checkpoint_init
    )
    if use_diffusion_prior_init:
        calibration_images = collect_calibration_images(
            train_loader,
            num_batches=int(cfg.big_vae_init_calibration_batches),
        )
        if calibration_images is None:
            raise RuntimeError("failed to collect calibration images for diffusion_prior initialization")
        print(
            f"[latent] collected {int(calibration_images.shape[0])} calibration images for autoregressive diffusion prior init",
            flush=True,
        )

    init_checkpoint_steps = 0
    if cfg.setup == "latent":
        if str(cfg.raw_checkpoint).strip():
            initial_tensors, init_checkpoint_steps = load_raw_checkpoint(cfg.raw_checkpoint, vit_cfg)
            print(f"[latent] loaded raw checkpoint init: {cfg.raw_checkpoint}", flush=True)
        else:
            initial_tensors = make_initial_tensors(vit_cfg, seed=int(cfg.seed))
            print("[latent] no raw checkpoint provided; using fresh ViT initialization from seed", flush=True)
    else:
        initial_tensors = make_initial_tensors(vit_cfg, seed=int(cfg.seed))

    big_vae_decoder = None
    big_vae_diffusion_prior = None
    if cfg.setup == "latent":
        print(f"[latent] loading frozen BigVAE decoder: {cfg.big_vae_checkpoint}", flush=True)
        big_vae_decoder = load_frozen_big_vae_decoder(cfg.big_vae_checkpoint, device=device)
        if use_diffusion_prior_init:
            print(
                f"[latent] loading frozen latent diffusion prior: {cfg.big_vae_diffusion_prior_checkpoint}",
                flush=True,
            )
            big_vae_diffusion_prior = load_frozen_layer_latent_diffusion_prior(
                cfg.big_vae_diffusion_prior_checkpoint,
                device=device,
            )
            if big_vae_decoder is not None:
                loaded_dist_encoder = load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint(
                    cfg.big_vae_diffusion_prior_checkpoint,
                    big_vae=big_vae_decoder,
                )
                if loaded_dist_encoder:
                    print(
                        "[latent] loaded finetuned distribution encoder state from latent diffusion prior checkpoint",
                        flush=True,
                    )

    model = build_model(
        cfg,
        vit_cfg,
        initial_tensors,
        big_vae_decoder=big_vae_decoder,
        big_vae_diffusion_prior=big_vae_diffusion_prior,
        big_vae_latent_init_override=("random" if use_latent_checkpoint_init else None),
        big_vae_latent_space_override=latent_checkpoint_space,
    ).to(device)
    if use_diffusion_prior_init:
        target_model = getattr(model, "_orig_mod", model)
        if not hasattr(target_model, "initialize_bigvae_diffusion_prior"):
            raise TypeError("diffusion_prior init requires FunctionalViTTiny.initialize_bigvae_diffusion_prior")
        target_model.initialize_bigvae_diffusion_prior(calibration_images)
    if bool(cfg.compile):
        model = torch.compile(model)

    target = getattr(model, "_orig_mod", model)
    store = getattr(target, "store")
    init_latent_steps = 0
    init_latent_loaded_keys = 0
    if use_latent_checkpoint_init:
        if not isinstance(store, BigVAELatentTensorStore):
            raise TypeError("latent checkpoint init requires BigVAELatentTensorStore")
        assert latent_checkpoint_payload is not None
        load_report = load_latent_slots_from_checkpoint(store.latent_slots, latent_checkpoint_payload)
        init_latent_steps = int(load_report["checkpoint_step"])
        init_latent_loaded_keys = int(len(load_report["loaded_keys"]))
        print(
            "[latent] loaded latent checkpoint "
            f"keys={init_latent_loaded_keys} "
            f"missing={len(load_report['missing_keys'])} "
            f"unexpected={len(load_report['unexpected_keys'])} "
            f"shape_skipped={len(load_report['skipped_shape_keys'])} "
            f"source_step={init_latent_steps}",
            flush=True,
        )
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = build_optimizer(trainable_parameters, cfg=cfg, setup=str(cfg.setup))
    planned_train_steps = resolve_planned_train_steps(cfg, steps_per_epoch=len(train_loader))
    lr_scheduler, lr_schedule_description = build_lr_scheduler(
        optimizer,
        cfg=cfg,
        setup=str(cfg.setup),
        planned_train_steps=planned_train_steps,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and bool(cfg.amp)))

    trainable_params = count_trainable_parameters(model)
    decoded_params = int(store.decoded_numel())
    latent_params = store.latent_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    decoded_big_vae_params = store.big_vae_decoded_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    tile_count = store.decoded_tile_count() if isinstance(store, BigVAELatentTensorStore) else 0
    print(
        f"[{cfg.setup}] dataset={cfg.dataset} model={cfg.model_size} base_lr={float(cfg.lr):g} "
        f"optimizer={cfg.optimizer_name} "
        f"lr_schedule={lr_schedule_description} "
        f"trainable_params={trainable_params} decoded_params={decoded_params} "
        f"latent_params={latent_params} bigvae_decoded_params={decoded_big_vae_params} tiles={tile_count}",
        flush=True,
    )

    metrics_path = output_dir / "metrics.csv"
    checkpoint_dir = output_dir / "checkpoints"
    global_step = 0
    train_loss_window = 0.0
    train_count_window = 0
    start_time = time.time()
    best_accuracy = 0.0
    best_step = 0
    final_eval = EvalMetrics(loss=float("nan"), accuracy=0.0, examples=0)

    for epoch_idx in range(int(cfg.epochs)):
        model.train()
        for images, labels in train_loader:
            global_step += 1
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            max_steps = int(cfg.max_steps)
            should_log = global_step == 1 or global_step % max(1, int(cfg.log_every_steps)) == 0
            should_eval = global_step == 1 or global_step % max(1, int(cfg.eval_every_steps)) == 0
            if max_steps > 0 and global_step >= max_steps:
                should_eval = True
            latent_debug_this_step = bool(
                cfg.setup == "latent"
                and isinstance(store, BigVAELatentTensorStore)
                and bool(cfg.latent_debug_log_deltas)
                and (should_log or should_eval)
            )
            pre_step_latent = None
            pre_step_decoded = None
            if latent_debug_this_step:
                pre_step_latent = flatten_latent_slots(store)
                pre_step_decoded = flatten_decoded_bigvae_weights(store)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, bool(cfg.amp)):
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=float(cfg.label_smoothing),
                )
            scaler.scale(loss).backward()
            if float(cfg.grad_clip_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            step_lr = float(optimizer.param_groups[0]["lr"])
            if lr_scheduler is not None:
                lr_scheduler.step()

            batch_examples = int(labels.numel())
            train_loss_window += float(loss.detach().cpu().item()) * batch_examples
            train_count_window += batch_examples
            delta_z_norm = float("nan")
            delta_w_norm = float("nan")
            delta_w_over_delta_z = float("nan")
            latent_z_norm = float("nan")
            delta_z_cosine_with_z = float("nan")
            delta_z_parallel_ratio = float("nan")
            delta_z_tangent_ratio = float("nan")
            jacobian_fd_mean = float("nan")
            jacobian_fd_std = float("nan")
            jacobian_fd_max = float("nan")
            if latent_debug_this_step:
                assert pre_step_latent is not None and pre_step_decoded is not None
                post_step_latent = flatten_latent_slots(store)
                post_step_decoded = flatten_decoded_bigvae_weights(store)
                delta_z = post_step_latent - pre_step_latent
                delta_z_norm = float(delta_z.norm().item())
                delta_w_norm = float((post_step_decoded - pre_step_decoded).norm().item())
                delta_w_over_delta_z = float(
                    delta_w_norm / max(delta_z_norm, torch.finfo(post_step_latent.dtype).tiny)
                )
                latent_z_norm = float(post_step_latent.norm().item())
                tiny = torch.finfo(post_step_latent.dtype).tiny
                if delta_z_norm > 0.0 and latent_z_norm > 0.0:
                    z_unit = post_step_latent / post_step_latent.norm().clamp_min(tiny)
                    parallel_component = torch.dot(delta_z, z_unit)
                    delta_z_parallel = parallel_component * z_unit
                    delta_z_tangent = delta_z - delta_z_parallel
                    delta_z_cosine_with_z = float(
                        torch.dot(delta_z, post_step_latent).item()
                        / max(delta_z_norm * latent_z_norm, float(tiny))
                    )
                    delta_z_parallel_ratio = float(
                        delta_z_parallel.norm().item() / max(delta_z_norm, float(tiny))
                    )
                    delta_z_tangent_ratio = float(
                        delta_z_tangent.norm().item() / max(delta_z_norm, float(tiny))
                    )
                jacobian_stats = estimate_decoder_effective_jacobian_norm(
                    store,
                    eps=float(cfg.latent_debug_jacobian_eps),
                    num_probes=int(cfg.latent_debug_jacobian_probes),
                )
                jacobian_fd_mean = float(jacobian_stats["mean"])
                jacobian_fd_std = float(jacobian_stats["std"])
                jacobian_fd_max = float(jacobian_stats["max"])

            if should_eval:
                final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.amp))
                if final_eval.accuracy > best_accuracy:
                    best_accuracy = float(final_eval.accuracy)
                    best_step = int(global_step)
            if should_log or should_eval:
                avg_train_loss = train_loss_window / max(1, train_count_window)
                elapsed_s = time.time() - start_time
                total_steps_with_raw_init = int(global_step + (init_checkpoint_steps if cfg.setup == "latent" else 0))
                row = {
                    "dataset": str(cfg.dataset),
                    "model_size": str(cfg.model_size),
                    "setup": str(cfg.setup),
                    "optimizer_name": str(cfg.optimizer_name),
                    "lr": float(step_lr),
                    "base_lr": float(cfg.lr),
                    "step": int(global_step),
                    "total_steps_with_raw_init": int(total_steps_with_raw_init),
                    "total_steps_with_latent_transfer_init": int(global_step + init_latent_steps),
                    "epoch": int(epoch_idx + 1),
                    "train_loss": float(avg_train_loss),
                    "test_loss": float(final_eval.loss),
                    "test_accuracy": float(final_eval.accuracy),
                    "best_accuracy": float(best_accuracy),
                    "trainable_params": int(trainable_params),
                    "decoded_params": int(decoded_params),
                    "latent_params": int(latent_params),
                    "elapsed_s": float(elapsed_s),
                    "latent_z_norm": float(latent_z_norm),
                    "delta_z_norm": float(delta_z_norm),
                    "delta_w_norm": float(delta_w_norm),
                    "delta_w_over_delta_z": float(delta_w_over_delta_z),
                    "delta_z_cosine_with_z": float(delta_z_cosine_with_z),
                    "delta_z_parallel_ratio": float(delta_z_parallel_ratio),
                    "delta_z_tangent_ratio": float(delta_z_tangent_ratio),
                    "decoder_jacobian_fd_mean": float(jacobian_fd_mean),
                    "decoder_jacobian_fd_std": float(jacobian_fd_std),
                    "decoder_jacobian_fd_max": float(jacobian_fd_max),
                }
                append_csv_row(metrics_path, row)
                message = (
                    f"[{cfg.setup}] step={global_step} total_step={total_steps_with_raw_init} "
                    f"epoch={epoch_idx + 1} lr={step_lr:.6g} train_loss={avg_train_loss:.4f} "
                    f"test_loss={final_eval.loss:.4f} test_acc={final_eval.accuracy:.4f} "
                    f"best={best_accuracy:.4f}"
                )
                if latent_debug_this_step:
                    message += (
                        f" ||z||={latent_z_norm:.6e}"
                        f" ||Δz||={delta_z_norm:.6e}"
                        f" cos(Δz,z)={delta_z_cosine_with_z:.6e}"
                        f" ||Δz_parallel||/||Δz||={delta_z_parallel_ratio:.6e}"
                        f" ||Δz_tangent||/||Δz||={delta_z_tangent_ratio:.6e}"
                        f" ||ΔW||={delta_w_norm:.6e}"
                        f" ||ΔW||/||Δz||={delta_w_over_delta_z:.6e}"
                        f" J_fd_mean={jacobian_fd_mean:.6e}"
                        f" J_fd_std={jacobian_fd_std:.6e}"
                        f" J_fd_max={jacobian_fd_max:.6e}"
                    )
                print(message, flush=True)
                train_loss_window = 0.0
                train_count_window = 0

            if max_steps > 0 and global_step >= max_steps:
                break
        if int(cfg.max_steps) > 0 and global_step >= int(cfg.max_steps):
            break

    final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.amp))
    if final_eval.accuracy > best_accuracy:
        best_accuracy = float(final_eval.accuracy)
        best_step = int(global_step)

    if bool(cfg.save_checkpoints):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if cfg.setup == "raw":
            torch.save(
                {
                    "setup": "raw",
                    "step": int(global_step),
                    "named_tensors": export_named_tensors(model, list(initial_tensors.keys())),
                    "vit_config": asdict(vit_cfg),
                    "run_config": asdict(cfg),
                },
                checkpoint_dir / "raw_final.pt",
            )
        else:
            torch.save(
                {
                    "setup": "latent",
                    "step": int(global_step),
                    "init_raw_checkpoint": str(cfg.raw_checkpoint),
                    "init_raw_steps": int(init_checkpoint_steps),
                    "init_latent_checkpoint": str(cfg.latent_checkpoint),
                    "init_latent_steps": int(init_latent_steps),
                    "latent_space": str(getattr(target.store, "latent_space", "")),
                    "latent_slots": target.store.latent_slots.state_dict(),
                    "vit_config": asdict(vit_cfg),
                    "run_config": asdict(cfg),
                },
                checkpoint_dir / "latent_final.pt",
            )

    summary = {
        "dataset": str(cfg.dataset),
        "model_size": str(cfg.model_size),
        "setup": str(cfg.setup),
        "optimizer_name": str(cfg.optimizer_name),
        "optimizer_kwargs": dict(cfg.optimizer_kwargs),
        "steps": int(global_step),
        "init_raw_steps": int(init_checkpoint_steps if cfg.setup == "latent" else 0),
        "total_steps_with_raw_init": int(global_step + (init_checkpoint_steps if cfg.setup == "latent" else 0)),
        "init_latent_checkpoint": str(cfg.latent_checkpoint) if cfg.setup == "latent" else "",
        "init_latent_steps": int(init_latent_steps if cfg.setup == "latent" else 0),
        "init_latent_loaded_keys": int(init_latent_loaded_keys if cfg.setup == "latent" else 0),
        "total_steps_with_latent_transfer_init": int(global_step + (init_latent_steps if cfg.setup == "latent" else 0)),
        "final_test_loss": float(final_eval.loss),
        "final_test_accuracy": float(final_eval.accuracy),
        "best_test_accuracy": float(best_accuracy),
        "best_step": int(best_step),
        "base_lr": float(cfg.lr),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "latent_lr_scheduler": str(cfg.latent_lr_scheduler),
        "latent_lr_floor_ratio": float(cfg.latent_lr_floor_ratio),
        "latent_lr_decay_steps": int(cfg.latent_lr_decay_steps),
        "latent_debug_log_deltas": bool(cfg.latent_debug_log_deltas),
        "latent_debug_jacobian_eps": float(cfg.latent_debug_jacobian_eps),
        "latent_debug_jacobian_probes": int(cfg.latent_debug_jacobian_probes),
        "lr_schedule": str(lr_schedule_description),
        "weight_decay": float(cfg.latent_weight_decay if cfg.setup == "latent" else cfg.weight_decay),
        "trainable_params": int(trainable_params),
        "decoded_params": int(decoded_params),
        "latent_params": int(latent_params),
        "bigvae_decoded_params": int(decoded_big_vae_params),
        "bigvae_tile_count": int(tile_count),
        "raw_checkpoint": str(cfg.raw_checkpoint) if cfg.setup == "latent" else "",
        "big_vae_checkpoint": str(cfg.big_vae_checkpoint) if cfg.setup == "latent" else "",
        "big_vae_latent_init": str(cfg.big_vae_latent_init) if cfg.setup == "latent" else "",
        "big_vae_latent_space": str(getattr(store, "latent_space", "")) if cfg.setup == "latent" else "",
        "big_vae_diffusion_prior_checkpoint": (
            str(cfg.big_vae_diffusion_prior_checkpoint) if cfg.setup == "latent" else ""
        ),
        "big_vae_diffusion_prior_steps": int(cfg.big_vae_diffusion_prior_steps) if cfg.setup == "latent" else 0,
        "big_vae_diffusion_prior_sampler": (
            str(cfg.big_vae_diffusion_prior_sampler) if cfg.setup == "latent" else ""
        ),
        "big_vae_diffusion_prior_eta": float(cfg.big_vae_diffusion_prior_eta) if cfg.setup == "latent" else 0.0,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> tuple[ScalingRunConfig, ViTTinyConfig]:
    parser = argparse.ArgumentParser(
        description="Scale raw-weight AdamW vs BigVAE latent AdamW optimization for ViT classifiers.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "imagenet"), required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--setup", choices=("raw", "latent"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--optimizer-name", default="AdamW")
    parser.add_argument(
        "--optimizer-kwarg",
        action="append",
        default=[],
        help="Additional optimizer kwarg as key=value. Values are parsed via JSON when possible.",
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--latent-lr-scheduler",
        choices=("constant", "cosine_decay_to_floor"),
        default="cosine_decay_to_floor",
    )
    parser.add_argument("--latent-lr-floor-ratio", type=float, default=0.1)
    parser.add_argument("--latent-lr-decay-steps", type=int, default=5250)
    parser.add_argument("--latent-debug-log-deltas", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latent-debug-jacobian-eps", type=float, default=1e-3)
    parser.add_argument("--latent-debug-jacobian-probes", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--latent-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-checkpoint", default="")
    parser.add_argument("--latent-checkpoint", default="")
    parser.add_argument("--big-vae-checkpoint", default="")
    parser.add_argument(
        "--big-vae-latent-init",
        choices=("base", "random", "encoded", "diffusion_prior"),
        default="encoded",
    )
    parser.add_argument("--big-vae-diffusion-prior-checkpoint", default="")
    parser.add_argument("--big-vae-diffusion-prior-steps", type=int, default=50)
    parser.add_argument("--big-vae-diffusion-prior-sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--big-vae-diffusion-prior-eta", type=float, default=0.0)
    parser.add_argument("--big-vae-decode", choices=("weights", "all"), default="all")
    parser.add_argument("--big-vae-tile-t-patches", "--big-vae-tile-T-patches", dest="big_vae_tile_T_patches", type=int, default=16)
    parser.add_argument("--big-vae-tile-d-out", type=int, default=8)
    parser.add_argument("--big-vae-encoder-context-rows", type=int, default=64)
    parser.add_argument("--big-vae-encoder-context-std", type=float, default=1.0)
    parser.add_argument("--big-vae-encoder-batch-size", type=int, default=16)
    parser.add_argument("--big-vae-init-calibration-batches", type=int, default=1)
    parser.add_argument("--raw-init-steps", type=int, default=0)

    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)

    args = parser.parse_args()
    setup = str(args.setup).strip().lower()

    run_cfg = ScalingRunConfig(
        output_dir=str(args.output_dir),
        dataset=str(args.dataset),
        model_size=str(args.model_size),
        setup=setup,
        data_dir=str(args.data_dir),
        download=bool(args.download),
        device=str(args.device),
        seed=int(args.seed),
        epochs=int(args.epochs),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        train_subset=int(args.train_subset),
        test_subset=int(args.test_subset),
        optimizer_name=str(args.optimizer_name),
        optimizer_kwargs=parse_optimizer_kwargs_cli(list(args.optimizer_kwarg)),
        lr=float(args.lr),
        latent_lr_scheduler=str(args.latent_lr_scheduler),
        latent_lr_floor_ratio=float(args.latent_lr_floor_ratio),
        latent_lr_decay_steps=int(args.latent_lr_decay_steps),
        latent_debug_log_deltas=bool(args.latent_debug_log_deltas),
        latent_debug_jacobian_eps=float(args.latent_debug_jacobian_eps),
        latent_debug_jacobian_probes=int(args.latent_debug_jacobian_probes),
        weight_decay=float(args.weight_decay),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_eps=float(args.adam_eps),
        label_smoothing=float(args.label_smoothing),
        grad_clip_norm=float(args.grad_clip_norm),
        amp=bool(args.amp),
        tf32=bool(args.tf32),
        compile=bool(args.compile),
        log_every_steps=int(args.log_every_steps),
        eval_every_steps=int(args.eval_every_steps),
        save_checkpoints=bool(args.save_checkpoints),
        raw_checkpoint=str(args.raw_checkpoint),
        latent_checkpoint=str(args.latent_checkpoint),
        big_vae_checkpoint=str(args.big_vae_checkpoint),
        big_vae_latent_init=str(args.big_vae_latent_init),
        big_vae_diffusion_prior_checkpoint=str(args.big_vae_diffusion_prior_checkpoint),
        big_vae_diffusion_prior_steps=int(args.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(args.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(args.big_vae_diffusion_prior_eta),
        big_vae_decode=str(args.big_vae_decode),
        big_vae_tile_T_patches=int(args.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(args.big_vae_tile_d_out),
        big_vae_encoder_context_rows=int(args.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(args.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(args.big_vae_encoder_batch_size),
        big_vae_init_calibration_batches=int(args.big_vae_init_calibration_batches),
        latent_weight_decay=float(args.latent_weight_decay),
        raw_init_steps=int(args.raw_init_steps),
    )
    validate_scaling_run_config(run_cfg)
    vit_cfg = ViTTinyConfig(
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        in_channels=int(args.in_channels),
        num_classes=int(args.num_classes),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
    )
    return run_cfg, vit_cfg


def main() -> None:
    cfg, vit_cfg = parse_args()
    random.seed(int(cfg.seed))
    train_once(cfg, vit_cfg)


if __name__ == "__main__":
    main()
