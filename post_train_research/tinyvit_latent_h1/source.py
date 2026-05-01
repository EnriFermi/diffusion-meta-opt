from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments.compare_vit_tiny_latent_optimization import (
    BigVAELatentTensorStore,
    EvalMetrics,
    FunctionalViTTiny,
    autocast_context,
    load_frozen_big_vae_decoder,
    make_initial_tensors,
)
from post_train_research.tinyvit_latent_h1.config import RunConfig
from post_train_research.vit_latent_scaling.init import (
    adapt_named_tensors,
    export_named_tensors,
    load_checkpoint_payload,
    resolve_source_checkpoint_path as resolve_vit_scaling_checkpoint_path,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(slots=True)
class SourceContext:
    checkpoint_path: Path
    payload: dict[str, Any]
    vit_cfg: Any
    initial_tensors: dict[str, torch.Tensor]
    all_tensor_names: list[str]
    decoded_names: list[str]
    latent_space: str
    big_vae: Any
    z_star_state: dict[str, torch.Tensor]
    z_star_conditioning_state: dict[str, torch.Tensor]
    z_star_named_tensors: dict[str, torch.Tensor]
    z_star_train_metrics: EvalMetrics
    z_star_test_metrics: EvalMetrics
    latent_param_count: int
    raw_param_count: int


@dataclass(slots=True)
class StartPoint:
    start_id: str
    epsilon: float
    direction_index: int
    alpha: float
    train_metrics: EvalMetrics
    test_metrics: EvalMetrics
    latent_state: dict[str, torch.Tensor]
    named_tensors: dict[str, torch.Tensor]


def sanitize_float(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def clone_tensor_dict(mapping: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(key): value.detach().cpu().clone() for key, value in mapping.items()}


def clone_named_tensors(mapping: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(name): tensor.detach().cpu().clone() for name, tensor in mapping.items()}


def clone_conditioning_state(mapping: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(key): value.detach().cpu().clone() for key, value in mapping.items()}


def conditioning_state_dict(store: BigVAELatentTensorStore) -> dict[str, torch.Tensor]:
    return clone_conditioning_state(store._tile_cond_patch)


@torch.no_grad()
def load_conditioning_state(
    store: BigVAELatentTensorStore,
    state: dict[str, torch.Tensor] | None,
    *,
    strict: bool = True,
) -> None:
    if not state:
        if strict and store._tile_cond_patch:
            store._tile_cond_patch.clear()
        return
    missing = [str(key) for key in store._tile_specs.keys() if key not in state]
    unexpected = [str(key) for key in state.keys() if key not in store._tile_specs]
    if strict and unexpected:
        raise RuntimeError(f"conditioning state mismatch: unexpected={unexpected[:8]}")
    first_latent = next(iter(store.latent_slots.values()))
    store._tile_cond_patch = {
        str(key): value.detach().to(device=first_latent.device, dtype=first_latent.dtype).contiguous()
        for key, value in state.items()
        if key in store._tile_specs
    }
    if strict and missing and store.use_distribution_encoder:
        raise RuntimeError(f"conditioning state mismatch: missing={missing[:8]}")


def build_cifar10_datasets(cfg: RunConfig):
    from torchvision import datasets, transforms

    steps: list[Any] = []
    if int(cfg.model.in_channels) == 1:
        steps.append(transforms.Grayscale(num_output_channels=1))
    if int(cfg.model.image_size) != 32:
        steps.append(transforms.Resize((int(cfg.model.image_size), int(cfg.model.image_size))))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (CIFAR10_MEAN[0],) * int(cfg.model.in_channels) if int(cfg.model.in_channels) == 1 else CIFAR10_MEAN,
                (CIFAR10_STD[0],) * int(cfg.model.in_channels) if int(cfg.model.in_channels) == 1 else CIFAR10_STD,
            ),
        ]
    )
    root = str(Path(cfg.data.data_dir).expanduser())
    transform = transforms.Compose(steps)
    train_set = datasets.CIFAR10(root=root, train=True, download=bool(cfg.data.download), transform=transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=bool(cfg.data.download), transform=transform)
    if int(cfg.data.train_subset) > 0:
        train_set = Subset(train_set, list(range(min(int(cfg.data.train_subset), len(train_set)))))
    if int(cfg.data.test_subset) > 0:
        test_set = Subset(test_set, list(range(min(int(cfg.data.test_subset), len(test_set)))))
    return train_set, test_set


def build_eval_loader(dataset: Any, *, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(num_workers) > 0,
    )


def build_train_schedule(dataset_len: int, *, batch_size: int, steps: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    schedule: list[torch.Tensor] = []
    while len(schedule) < int(steps):
        permutation = torch.randperm(int(dataset_len), generator=generator)
        for start in range(0, int(dataset_len), int(batch_size)):
            schedule.append(permutation[start : start + int(batch_size)].clone())
            if len(schedule) >= int(steps):
                break
    return schedule


def fetch_batch(dataset: Any, indices: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for raw_idx in indices.tolist():
        image, label = dataset[int(raw_idx)]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images, dim=0).to(device=device, non_blocking=False), torch.tensor(labels, device=device)


def evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device, amp_enabled: bool) -> EvalMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        with autocast_context(device, bool(amp_enabled)):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss.detach().cpu().item())
        correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu().item())
        examples += int(labels.numel())
    return EvalMetrics(loss=loss_sum / max(1, examples), accuracy=correct / max(1, examples), examples=examples)


def sorted_latent_keys(store: BigVAELatentTensorStore) -> list[str]:
    return sorted(str(key) for key in store.latent_slots.keys())


def flatten_materialized_latents(store: BigVAELatentTensorStore) -> torch.Tensor:
    keys = sorted_latent_keys(store)
    return torch.cat([store.materialize_latent_slot(key).detach().reshape(-1) for key in keys], dim=0)


def flat_to_state_dict(flat: torch.Tensor, template: dict[str, torch.Tensor], *, keys: list[str]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    offset = 0
    for key in keys:
        target = template[key]
        numel = int(target.numel())
        state[key] = flat[offset : offset + numel].view_as(target).detach().cpu().clone()
        offset += numel
    return state


def load_materialized_state(store: BigVAELatentTensorStore, state: dict[str, torch.Tensor]) -> None:
    store.load_materialized_latent_slots_state_dict(state, strict=True, update_radii=True)


def resolve_source_checkpoint_for_experiment(cfg: RunConfig) -> Path:
    source_ref = str(cfg.source.run_dir).strip()
    direct = Path(source_ref).expanduser()
    checkpoint_name = str(cfg.source.checkpoint_name).strip() or "best"
    if direct.is_file():
        return direct.resolve()
    if direct.is_dir():
        candidate = direct / f"{checkpoint_name}.pt"
        if candidate.exists():
            return candidate.resolve()
        nested = direct / "checkpoints" / f"{checkpoint_name}.pt"
        if nested.exists():
            return nested.resolve()
    storage_root = Path(cfg.source.vit_scaling_artifacts_root).expanduser().resolve()
    shared_dir = storage_root / "checkpoints" / source_ref
    if shared_dir.is_dir():
        shared_candidate = shared_dir / f"{checkpoint_name}.pt"
        if shared_candidate.exists():
            return shared_candidate.resolve()
    return resolve_vit_scaling_checkpoint_path(
        storage_root,
        source_run_dir=source_ref,
        checkpoint_name=checkpoint_name,
    )


def _apply_source_overrides(cfg: RunConfig, payload: dict[str, Any], logger: logging.Logger) -> None:
    vit_payload = payload.get("vit_config")
    if bool(cfg.source.use_checkpoint_vit_config) and isinstance(vit_payload, dict):
        for key in ("image_size", "patch_size", "in_channels", "num_classes", "hidden_dim", "depth", "num_heads", "mlp_ratio", "dropout", "attention_dropout"):
            if key in vit_payload:
                setattr(cfg.model, key, type(getattr(cfg.model, key))(vit_payload[key]))
        logger.info("Resolved model config from source checkpoint vit_config")
    source_cfg = payload.get("config")
    if bool(cfg.source.use_checkpoint_setup_config) and isinstance(source_cfg, dict):
        setup_payload = source_cfg.get("setup", {})
        if isinstance(setup_payload, dict):
            cfg.setup.big_vae_checkpoint = str(setup_payload.get("big_vae_checkpoint", cfg.setup.big_vae_checkpoint)).strip()
            cfg.setup.big_vae_decode = str(setup_payload.get("big_vae_decode", cfg.setup.big_vae_decode)).strip().lower()
            cfg.setup.big_vae_tile_T_patches = int(setup_payload.get("big_vae_tile_T_patches", cfg.setup.big_vae_tile_T_patches))
            cfg.setup.big_vae_tile_d_out = int(setup_payload.get("big_vae_tile_d_out", cfg.setup.big_vae_tile_d_out))
            cfg.setup.big_vae_latent_parameterization = str(
                setup_payload.get("big_vae_latent_parameterization", cfg.setup.big_vae_latent_parameterization)
            ).strip().lower()
            logger.info("Resolved decoder config from source checkpoint setup")
    if not cfg.setup.big_vae_latent_parameterization and payload.get("latent_parameterization"):
        cfg.setup.big_vae_latent_parameterization = str(payload["latent_parameterization"]).strip().lower()
    if not cfg.setup.big_vae_checkpoint:
        raise ValueError("Could not resolve setup.big_vae_checkpoint from config/source checkpoint")
    if cfg.setup.big_vae_decode not in {"weights", "all"}:
        raise ValueError("setup.big_vae_decode must resolve to 'weights' or 'all'")
    if int(cfg.setup.big_vae_tile_T_patches) <= 0 or int(cfg.setup.big_vae_tile_d_out) <= 0:
        raise ValueError("setup.big_vae_tile_T_patches and setup.big_vae_tile_d_out must resolve to positive values")
    if cfg.setup.big_vae_latent_parameterization not in {"euclidean", "sphere"}:
        raise ValueError("setup.big_vae_latent_parameterization must resolve to 'euclidean' or 'sphere'")


def prepare_config_from_source_checkpoint(cfg: RunConfig, logger: logging.Logger) -> Path:
    checkpoint_path = resolve_source_checkpoint_for_experiment(cfg)
    payload = load_checkpoint_payload(checkpoint_path)
    if not isinstance(payload.get("latent_slots"), dict):
        raise ValueError(f"Source checkpoint must contain latent_slots: {checkpoint_path}")
    _apply_source_overrides(cfg, payload, logger)
    return checkpoint_path


def build_latent_model(
    cfg: RunConfig,
    source: SourceContext,
    *,
    device: torch.device,
    latent_state: dict[str, torch.Tensor],
    conditioning_state: dict[str, torch.Tensor] | None = None,
    freeze_direct: bool = True,
) -> nn.Module:
    model = FunctionalViTTiny(
        source.vit_cfg,
        clone_named_tensors(source.z_star_named_tensors),
        parameter_mode="bigvae_latent",
        big_vae=source.big_vae,
        big_vae_latent_init="base",
        big_vae_latent_space=source.latent_space,
        big_vae_latent_parameterization=str(cfg.setup.big_vae_latent_parameterization),
        big_vae_decode=str(cfg.setup.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.setup.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.setup.big_vae_tile_d_out),
        big_vae_random_init_std=0.0,
        big_vae_latent_noise_std=0.0,
        big_vae_encoder_context_rows=64,
        big_vae_encoder_context_std=1.0,
        big_vae_encoder_batch_size=16,
    ).to(device)
    store = getattr(model, "store")
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("Expected BigVAELatentTensorStore for latent branch")
    if freeze_direct:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in store.latent_slots.parameters():
            param.requires_grad_(True)
    load_materialized_state(store, latent_state)
    load_conditioning_state(store, conditioning_state, strict=False)
    return model


def build_raw_model(source: SourceContext, *, device: torch.device, start_named_tensors: dict[str, torch.Tensor]) -> nn.Module:
    model = FunctionalViTTiny(
        source.vit_cfg,
        clone_named_tensors(start_named_tensors),
        parameter_mode="direct",
    ).to(device)
    store = getattr(model, "store")
    decoded_set = set(source.decoded_names)
    for name, key in store._name_to_key.items():
        module = store.tensors[key]
        module.value.requires_grad_(str(name) in decoded_set)
    return model


def evaluate_train_and_test(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, *, device: torch.device, amp_enabled: bool) -> tuple[EvalMetrics, EvalMetrics]:
    return (
        evaluate(model, train_loader, device=device, amp_enabled=amp_enabled),
        evaluate(model, test_loader, device=device, amp_enabled=amp_enabled),
    )


def _find_levelset_alpha(
    model: nn.Module,
    source: SourceContext,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    base_flat: torch.Tensor,
    base_loss: float,
    direction: torch.Tensor,
    epsilon: float,
    device: torch.device,
    amp_enabled: bool,
    alpha_min: float,
    alpha_max: float,
    bracket_multiplier: float,
    binary_search_steps: int,
) -> tuple[float, dict[str, torch.Tensor], EvalMetrics, EvalMetrics, dict[str, torch.Tensor]] | None:
    store = getattr(model, "store")
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("Level-set search expects latent model")
    keys = sorted_latent_keys(store)
    template = store.materialized_latent_slots_state_dict()
    target_loss = float(base_loss + epsilon)

    def _evaluate_alpha(alpha: float) -> tuple[float, dict[str, torch.Tensor], EvalMetrics, EvalMetrics, dict[str, torch.Tensor]]:
        latent_state = flat_to_state_dict(base_flat + float(alpha) * direction, template, keys=keys)
        load_materialized_state(store, latent_state)
        train_metrics, test_metrics = evaluate_train_and_test(model, train_loader, test_loader, device=device, amp_enabled=amp_enabled)
        named_tensors = export_named_tensors(model, source.all_tensor_names)
        return float(train_metrics.loss), latent_state, train_metrics, test_metrics, named_tensors

    best: tuple[float, dict[str, torch.Tensor], EvalMetrics, EvalMetrics, dict[str, torch.Tensor]] | None = None
    for sign in (1.0, -1.0):
        alpha = float(alpha_min)
        prev_alpha = 0.0
        while alpha <= float(alpha_max):
            payload = _evaluate_alpha(sign * alpha)
            if payload[0] >= target_loss:
                lo = prev_alpha
                hi = alpha
                hi_payload = payload
                for _ in range(int(binary_search_steps)):
                    mid = 0.5 * (lo + hi)
                    mid_payload = _evaluate_alpha(sign * mid)
                    if mid_payload[0] >= target_loss:
                        hi = mid
                        hi_payload = mid_payload
                    else:
                        lo = mid
                candidate = (float(sign * hi),) + hi_payload[1:]
                if best is None or abs(candidate[0]) < abs(best[0]):
                    best = candidate
                break
            prev_alpha = alpha
            alpha *= float(bracket_multiplier)
        load_materialized_state(store, source.z_star_state)
    return best


def resolve_source_context(
    cfg: RunConfig,
    logger: logging.Logger,
    *,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_path: Path | None = None,
) -> SourceContext:
    checkpoint = checkpoint_path or resolve_source_checkpoint_for_experiment(cfg)
    payload = load_checkpoint_payload(checkpoint)
    _apply_source_overrides(cfg, payload, logger)
    if not isinstance(payload.get("latent_slots"), dict):
        raise ValueError(f"Source checkpoint must contain latent_slots: {checkpoint}")
    template = make_initial_tensors(cfg.vit_cfg, seed=int(cfg.train.seed))
    source_named = payload.get("named_tensors")
    if not isinstance(source_named, dict):
        raise ValueError(f"Source checkpoint must contain named_tensors: {checkpoint}")
    initial_tensors, report = adapt_named_tensors(source_named, template)
    logger.info(
        "Resolved source checkpoint=%s (exact=%s reshaped=%s missing=%s)",
        checkpoint,
        report["exact"],
        report["reshaped"],
        report["missing"],
    )
    big_vae = load_frozen_big_vae_decoder(cfg.setup.big_vae_checkpoint, device=device)
    latent_space = str(payload.get("latent_space", "encoder_slots")).strip().lower() or "encoder_slots"
    stub = SourceContext(
        checkpoint_path=checkpoint,
        payload=payload,
        vit_cfg=cfg.vit_cfg,
        initial_tensors=initial_tensors,
        all_tensor_names=list(initial_tensors.keys()),
        decoded_names=[],
        latent_space=latent_space,
        big_vae=big_vae,
        z_star_state={},
        z_star_conditioning_state={},
        z_star_named_tensors=clone_named_tensors(initial_tensors),
        z_star_train_metrics=EvalMetrics(loss=float("nan"), accuracy=0.0, examples=0),
        z_star_test_metrics=EvalMetrics(loss=float("nan"), accuracy=0.0, examples=0),
        latent_param_count=0,
        raw_param_count=0,
    )
    cond_payload = payload.get("tile_cond_patch")
    if cond_payload is None:
        logger.warning(
            "Source checkpoint %s has no tile_cond_patch; latent reconstruction may drift from the original conditioned decoder state",
            checkpoint,
        )
    elif not isinstance(cond_payload, dict):
        raise ValueError(f"Source checkpoint tile_cond_patch must be a dict when present: {checkpoint}")
    anchor_model = build_latent_model(
        cfg,
        stub,
        device=device,
        latent_state=payload["latent_slots"],
        conditioning_state=cond_payload if isinstance(cond_payload, dict) else None,
    )
    store = getattr(anchor_model, "store")
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("Expected BigVAELatentTensorStore for source anchor")
    z_star_state = store.materialized_latent_slots_state_dict()
    z_star_conditioning_state = conditioning_state_dict(store)
    z_star_named = export_named_tensors(anchor_model, list(initial_tensors.keys()))
    train_metrics, test_metrics = evaluate_train_and_test(anchor_model, train_loader, test_loader, device=device, amp_enabled=bool(cfg.train.amp))
    decoded_names = list(store.decoded_tensor_names())
    raw_param_count = int(sum(int(z_star_named[name].numel()) for name in decoded_names))
    return SourceContext(
        checkpoint_path=checkpoint,
        payload=payload,
        vit_cfg=cfg.vit_cfg,
        initial_tensors=initial_tensors,
        all_tensor_names=list(initial_tensors.keys()),
        decoded_names=decoded_names,
        latent_space=latent_space,
        big_vae=big_vae,
        z_star_state=z_star_state,
        z_star_conditioning_state=z_star_conditioning_state,
        z_star_named_tensors=z_star_named,
        z_star_train_metrics=train_metrics,
        z_star_test_metrics=test_metrics,
        latent_param_count=int(store.latent_numel()),
        raw_param_count=raw_param_count,
    )


def search_start_points(
    cfg: RunConfig,
    source: SourceContext,
    anchor_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    logger: logging.Logger,
) -> list[StartPoint]:
    store = getattr(anchor_model, "store")
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("Anchor model must use BigVAELatentTensorStore")
    base_flat = flatten_materialized_latents(store)
    generator = torch.Generator(device=base_flat.device)
    generator.manual_seed(int(cfg.train.seed))
    starts: list[StartPoint] = []
    for direction_index in range(int(cfg.search.random_directions)):
        direction = torch.randn_like(base_flat, generator=generator)
        direction = direction / direction.norm().clamp_min(torch.finfo(direction.dtype).eps)
        for epsilon in cfg.search.epsilons:
            result = _find_levelset_alpha(
                anchor_model,
                source,
                train_loader,
                test_loader,
                base_flat=base_flat,
                base_loss=source.z_star_train_metrics.loss,
                direction=direction,
                epsilon=float(epsilon),
                device=device,
                amp_enabled=bool(cfg.train.amp),
                alpha_min=float(cfg.search.alpha_min),
                alpha_max=float(cfg.search.alpha_max),
                bracket_multiplier=float(cfg.search.bracket_multiplier),
                binary_search_steps=int(cfg.search.binary_search_steps),
            )
            if result is None:
                logger.warning("Failed to find level-set start for eps=%.6f dir=%s", float(epsilon), direction_index)
                continue
            alpha, latent_state, train_metrics, test_metrics, named_tensors = result
            start_id = f"eps_{sanitize_float(float(epsilon))}__dir_{direction_index:02d}"
            starts.append(
                StartPoint(
                    start_id=start_id,
                    epsilon=float(epsilon),
                    direction_index=int(direction_index),
                    alpha=float(alpha),
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    latent_state=clone_tensor_dict(latent_state),
                    named_tensors=clone_named_tensors(named_tensors),
                )
            )
            logger.info(
                "Prepared %s alpha=%.6g train_loss=%.6f test_acc=%.4f",
                start_id,
                float(alpha),
                float(train_metrics.loss),
                float(test_metrics.accuracy),
            )
            load_materialized_state(store, source.z_star_state)
    return starts
