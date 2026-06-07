from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from big_vae.eval.vit_tiny_latent_optimization import (
    BigVAELatentTensorStore,
    FunctionalViTTiny,
    ViTTinyConfig,
    load_frozen_big_vae_decoder,
    make_initial_tensors,
    matrix_to_tensor,
    resolve_device,
    seed_everything,
)
from post_train_research.big_vae_heldout_eval.evaluate_parts.decoder_adapter import (
    latent_flattening_payload_big_vae_checkpoint,
    load_latent_flattening_flow_from_checkpoint,
    paths_match,
)
from training.big_vae_latent_diffusion import (
    load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint,
    load_frozen_layer_latent_diffusion_prior,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


DEFAULT_RADIAL_SCALES = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
]
DEFAULT_THETA_GRID = [
    -0.5 * math.pi,
    -math.pi / 3.0,
    -math.pi / 6.0,
    -math.pi / 12.0,
    0.0,
    math.pi / 12.0,
    math.pi / 6.0,
    math.pi / 3.0,
    0.5 * math.pi,
]
DEFAULT_CHECKPOINT_STEPS = [1, 50, 100, 200, 500]


def _get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting diagnostics") from exc
    return plt


def _higher_order_sdpa_context():
    # Higher-order autograd is not implemented for some efficient SDPA kernels on CUDA.
    # Force the math kernel only for Hessian/HVP-style paths.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        pass
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        )
    except Exception:
        return contextlib.nullcontext()


@dataclass(slots=True)
class LandscapeConfig:
    data_dir: str
    output_dir: str
    device: str
    download: bool
    seeds: list[int]
    train_subset: int
    test_subset: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    diag_batch_size: int
    diagnostic_split: str
    steps: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    log_every: int
    eval_every: int
    checkpoint_steps: list[int]
    big_vae_checkpoint: str
    big_vae_latent_init: str
    big_vae_latent_parameterization: str
    big_vae_diffusion_prior_checkpoint: str
    big_vae_diffusion_prior_steps: int
    big_vae_diffusion_prior_sampler: str
    big_vae_diffusion_prior_eta: float
    big_vae_decode: str
    big_vae_tile_T_patches: int
    big_vae_tile_d_out: int
    big_vae_decoder_adapter: str
    big_vae_decoder_adapter_checkpoint: str
    big_vae_decoder_adapter_require_checkpoint_match: bool
    big_vae_encoder_context_rows: int
    big_vae_encoder_context_std: float
    big_vae_encoder_batch_size: int
    big_vae_init_calibration_batches: int
    image_size: int
    patch_size: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attention_dropout: float
    num_classes: int
    in_channels: int
    radial_scales: list[float]
    theta_grid: list[float]
    angular_random_dirs: int
    jacobian_eps_rel: float
    jacobian_tangent_dirs: int
    compute_accessibility: bool
    compute_hessian: bool
    hessian_subspace_dim: int
    hessian_topk: int
    hessian_trace_probes: int
    compute_2d_slices: bool
    slice_grid_size: int
    slice_scale: float


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


def maybe_unlink(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def load_training_cache_if_compatible(
    cache_path: Path,
    *,
    cfg: LandscapeConfig,
    seed: int,
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    if int(payload.get("seed", -1)) != int(seed):
        return None
    expected_cfg = asdict(cfg)
    if payload.get("config") != expected_cfg:
        return None
    checkpoints = payload.get("checkpoints")
    summary = payload.get("summary")
    if not isinstance(checkpoints, dict) or not isinstance(summary, dict):
        return None
    if "final" not in checkpoints or "init" not in checkpoints or "after_step_1" not in checkpoints:
        return None
    return payload


def build_cifar10_datasets(cfg: LandscapeConfig):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for CIFAR-10 diagnostics") from exc

    data_root = Path(cfg.data_dir).expanduser()
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((int(cfg.image_size), int(cfg.image_size))),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((int(cfg.image_size), int(cfg.image_size))),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_aug = datasets.CIFAR10(root=str(data_root), train=True, download=bool(cfg.download), transform=train_transform)
    train_eval = datasets.CIFAR10(root=str(data_root), train=True, download=bool(cfg.download), transform=eval_transform)
    test_eval = datasets.CIFAR10(root=str(data_root), train=False, download=bool(cfg.download), transform=eval_transform)

    generator = torch.Generator()
    generator.manual_seed(0)
    train_indices = torch.arange(len(train_aug))
    test_indices = torch.arange(len(test_eval))
    if int(cfg.train_subset) > 0:
        train_indices = torch.randperm(len(train_aug), generator=generator)[: int(cfg.train_subset)]
    if int(cfg.test_subset) > 0:
        test_indices = torch.randperm(len(test_eval), generator=generator)[: int(cfg.test_subset)]

    train_aug = Subset(train_aug, train_indices.tolist())
    train_eval = Subset(train_eval, train_indices.tolist())
    test_eval = Subset(test_eval, test_indices.tolist())
    return train_aug, train_eval, test_eval


def build_loaders(cfg: LandscapeConfig, *, device: torch.device):
    train_aug, train_eval, test_eval = build_cifar10_datasets(cfg)
    pin_memory = device.type == "cuda"
    loader_generator = torch.Generator()
    loader_generator.manual_seed(0)

    train_loader = DataLoader(
        train_aug,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(cfg.num_workers) > 0,
        generator=loader_generator,
    )
    train_diag_loader = DataLoader(
        train_eval,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(cfg.num_workers) > 0,
    )
    test_loader = DataLoader(
        test_eval,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(cfg.num_workers) > 0,
    )
    return train_loader, train_diag_loader, test_loader


def collect_fixed_batch(loader: DataLoader, *, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    collected = 0
    for images, labels in loader:
        keep = min(int(batch_size) - collected, int(images.shape[0]))
        images_list.append(images[:keep].to(device=device, non_blocking=False))
        labels_list.append(labels[:keep].to(device=device, non_blocking=False))
        collected += keep
        if collected >= int(batch_size):
            break
    if not images_list:
        raise RuntimeError("failed to collect diagnostic batch")
    images = torch.cat(images_list, dim=0).contiguous()
    labels = torch.cat(labels_list, dim=0).contiguous()
    return images, labels


def sorted_latent_keys(store: BigVAELatentTensorStore) -> list[str]:
    return sorted(str(key) for key in store.latent_slots.keys())


def flatten_materialized_latents(store: BigVAELatentTensorStore) -> torch.Tensor:
    keys = sorted_latent_keys(store)
    if not keys:
        return torch.zeros(0, device=store.big_vae.latent_base.device, dtype=store.big_vae.latent_base.dtype)
    return torch.cat([store.materialize_latent_slot(key).detach().reshape(-1) for key in keys], dim=0)


def latent_layout(store: BigVAELatentTensorStore) -> list[tuple[str, torch.Size, int]]:
    layout: list[tuple[str, torch.Size, int]] = []
    for key in sorted_latent_keys(store):
        shape = store.materialize_latent_slot(key).shape
        layout.append((key, shape, int(math.prod(shape))))
    return layout


def unflatten_latent_state(
    store: BigVAELatentTensorStore,
    flat_z: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    offset = 0
    for key, shape, numel in latent_layout(store):
        state[key] = flat_z[offset : offset + numel].view(shape)
        offset += numel
    if offset != int(flat_z.numel()):
        raise ValueError(
            f"flat latent length mismatch: consumed {offset}, available {int(flat_z.numel())}"
        )
    return state


@torch.no_grad()
def capture_checkpoint_state(
    store: BigVAELatentTensorStore,
    *,
    name: str,
    step: int,
    update_direction: torch.Tensor | None = None,
) -> dict[str, Any]:
    flat_z = flatten_materialized_latents(store).detach().cpu().contiguous()
    return {
        "name": str(name),
        "step": int(step),
        "flat_z": flat_z,
        "latent_norm": float(flat_z.norm().item()),
        "update_direction": (
            None if update_direction is None else update_direction.detach().cpu().contiguous()
        ),
        "latent_parameterization": str(getattr(store, "latent_parameterization", "")),
        "latent_space": str(getattr(store, "latent_space", "")),
    }


def decode_all_matrices_from_latent_state(
    store: BigVAELatentTensorStore,
    latent_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not latent_state:
        return {}

    first_latent = next(iter(latent_state.values()))
    result: dict[str, torch.Tensor] = {}
    for tensor_key, spec in store._specs.items():
        name = store._key_to_name[tensor_key]
        result[name] = torch.zeros(
            spec.matrix_shape,
            device=first_latent.device,
            dtype=first_latent.dtype,
        )

    for (d_in, d_out, T), keys in store._groups.items():
        latents = torch.stack([latent_state[str(key)] for key in keys], dim=0)
        batch = int(latents.shape[0])
        d_in_pad = int(T) * int(store.patch_size)
        device = latents.device
        patch_mask = torch.zeros(batch, int(T), device=device, dtype=torch.bool)
        d_in_mask = torch.zeros(batch, int(d_in), device=device, dtype=torch.bool)
        d_out_mask = torch.zeros(batch, int(d_out), device=device, dtype=torch.bool)
        for item_idx, key in enumerate(keys):
            tile = store._tile_specs[str(key)]
            used_rows = 0
            used_cols = 0
            for segment in tile.segments:
                segment_row_end = int(segment.tile_row_start) + int(segment.row_len)
                d_in_mask[item_idx, int(segment.tile_row_start) : segment_row_end] = True
                used_rows = max(used_rows, segment_row_end)
                used_cols = max(used_cols, int(segment.col_len))
            valid_patches = int(math.ceil(float(used_rows) / float(store.patch_size)))
            patch_mask[item_idx, :valid_patches] = True
            d_out_mask[item_idx, :used_cols] = True

        dist_patch = None
        if store.use_distribution_encoder:
            if all(str(key) in store._tile_cond_patch for key in keys):
                dist_patch = torch.stack(
                    [
                        store._tile_cond_patch[str(key)].to(device=device, dtype=latents.dtype)
                        for key in keys
                    ],
                    dim=0,
                )
            else:
                dist_patch = torch.zeros(batch, int(T), int(store.d_dist), device=device, dtype=latents.dtype)

        if str(store.latent_space).strip().lower() == "decoder_z":
            decoded = store.big_vae._decode_from_decoder_latent(
                latents,
                dist_patch_by_patch=dist_patch,
                patch_mask=patch_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                d_in=int(d_in),
                d_out=int(d_out),
                d_in_pad=int(d_in_pad),
                T=int(T),
            )[0]
        else:
            decoded = store.big_vae._decode_from_latent_slots(
                latents,
                dist_patch_by_patch=dist_patch,
                patch_mask=patch_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                d_in=int(d_in),
                d_out=int(d_out),
                d_in_pad=int(d_in_pad),
                T=int(T),
            )[0]

        for item_idx, key in enumerate(keys):
            tile = store._tile_specs[str(key)]
            for segment in tile.segments:
                tile_row_start = int(segment.tile_row_start)
                tile_row_end = tile_row_start + int(segment.row_len)
                result[segment.tensor_name][
                    int(segment.row_start) : int(segment.row_start) + int(segment.row_len),
                    int(segment.col_start) : int(segment.col_start) + int(segment.col_len),
                ] = decoded[item_idx, tile_row_start:tile_row_end, : int(segment.col_len)]
    return result


def decode_all_tensors_from_latent_state(
    store: BigVAELatentTensorStore,
    latent_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    matrices = decode_all_matrices_from_latent_state(store, latent_state)
    return {
        name: matrix_to_tensor(matrix, store._specs[store._name_to_key[name]])
        for name, matrix in matrices.items()
    }


def flatten_decoded_tensors(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    if not tensors:
        first = torch.tensor([], dtype=torch.float32)
        return first
    return torch.cat([tensors[name].reshape(-1) for name in sorted(tensors.keys())], dim=0)


def forward_from_latent_state(
    model: FunctionalViTTiny,
    latent_state: dict[str, torch.Tensor],
    images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    target = getattr(model, "_orig_mod", model)
    store = target.store
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("expected FunctionalViTTiny backed by BigVAELatentTensorStore")
    decoded_matrices = decode_all_matrices_from_latent_state(store, latent_state)
    decoded_tensors = {
        name: matrix_to_tensor(matrix, store._specs[store._name_to_key[name]])
        for name, matrix in decoded_matrices.items()
    }
    was_training = target.training
    target.eval()
    target._decoded_tensor_cache = decoded_tensors
    try:
        logits = target._forward_with_cached_weights(images)
    finally:
        target._decoded_tensor_cache = None
        if was_training:
            target.train()
    return logits, decoded_matrices, decoded_tensors


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            loss_sum += float(loss.detach().cpu().item())
            correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu().item())
            count += int(labels.numel())
    return loss_sum / max(1, count), correct / max(1, count)


def project_parallel_and_tangent(vector: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tiny = max(float(torch.finfo(vector.dtype).eps), 1e-12)
    ref_norm_sq = torch.dot(reference, reference).clamp_min(tiny)
    parallel = torch.dot(vector, reference) / ref_norm_sq * reference
    tangent = vector - parallel
    return parallel, tangent


def normalized_or_none(vector: torch.Tensor) -> torch.Tensor | None:
    tiny = max(float(torch.finfo(vector.dtype).eps), 1e-12)
    norm = vector.norm()
    if float(norm.item()) <= tiny:
        return None
    return vector / norm


def orthonormal_random_tangent_directions(
    z0: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    directions: list[torch.Tensor] = []
    u = normalized_or_none(z0)
    if u is None:
        return directions
    tiny = max(float(torch.finfo(z0.dtype).eps), 1e-12)
    for _ in range(int(count)):
        v = torch.randn(z0.shape, generator=generator, device=z0.device, dtype=z0.dtype)
        v = v - torch.dot(v, u) * u
        for existing in directions:
            v = v - torch.dot(v, existing) * existing
        norm = v.norm()
        if float(norm.item()) <= tiny:
            continue
        directions.append(v / norm)
    return directions


def loss_acc_from_flat_z(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    flat_z: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    latent_state = unflatten_latent_state(store, flat_z)
    logits, _decoded_mats, _decoded_tensors = forward_from_latent_state(model, latent_state, images)
    loss = F.cross_entropy(logits, labels)
    acc = float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu().item())
    return loss, acc


def gradient_and_optional_weight_gradients(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    flat_z: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    create_graph: bool,
    return_weight_grads: bool,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]:
    latent_state = unflatten_latent_state(store, flat_z)
    autocast_ctx = _higher_order_sdpa_context() if create_graph else contextlib.nullcontext()
    with autocast_ctx:
        logits, _decoded_matrices, decoded_tensors = forward_from_latent_state(model, latent_state, images)
        loss = F.cross_entropy(logits, labels)
        acc = float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu().item())
        weight_grads = None
        if return_weight_grads:
            weight_tensors = [decoded_tensors[name] for name in sorted(decoded_tensors.keys())]
            grads = torch.autograd.grad(
                loss,
                [flat_z, *weight_tensors],
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )
            grad_z = grads[0]
            weight_grads = torch.cat([grad.reshape(-1) for grad in grads[1:]], dim=0)
        else:
            (grad_z,) = torch.autograd.grad(
                loss,
                flat_z,
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )
    return loss, acc, grad_z, weight_grads


def flatten_current_decoded_tensors(store: BigVAELatentTensorStore) -> torch.Tensor:
    decoded = store.decode_all_tensors()
    if not decoded:
        return torch.zeros(0, device=store.big_vae.latent_base.device, dtype=store.big_vae.latent_base.dtype)
    return flatten_decoded_tensors(decoded)


def estimate_decoder_fd_direction(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    direction: torch.Tensor,
    *,
    eps_abs: float,
) -> float:
    direction_unit = normalized_or_none(direction)
    if direction_unit is None or float(eps_abs) <= 0.0:
        return float("nan")
    z1 = z0 + float(eps_abs) * direction_unit
    base_tensors = decode_all_tensors_from_latent_state(store, unflatten_latent_state(store, z0))
    next_tensors = decode_all_tensors_from_latent_state(store, unflatten_latent_state(store, z1))
    w0 = flatten_decoded_tensors(base_tensors)
    w1 = flatten_decoded_tensors(next_tensors)
    return float((w1 - w0).norm().detach().cpu().item() / float(eps_abs))


def hvp_at_flat_z(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    vector: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    z = z0.detach().clone().requires_grad_(True)
    loss, _acc, grad_z, _ = gradient_and_optional_weight_gradients(
        model,
        store,
        z,
        images,
        labels,
        create_graph=True,
        return_weight_grads=False,
    )
    del loss
    dot = torch.dot(grad_z, vector)
    (hvp,) = torch.autograd.grad(dot, z, retain_graph=False, allow_unused=False)
    return hvp.detach()


def _robust_symmetric_eigvals(matrix: torch.Tensor) -> tuple[np.ndarray, str]:
    matrix64 = matrix.detach().to(device="cpu", dtype=torch.float64)
    matrix64 = 0.5 * (matrix64 + matrix64.transpose(0, 1))
    matrix_np = matrix64.numpy()
    if not np.isfinite(matrix_np).all():
        raise RuntimeError("projected Hessian contains non-finite values")

    # Use a cheap, robust scale estimate; spectral-norm SVD can fail on the
    # same ill-conditioned matrices that broke eigvalsh.
    spectral_scale = max(1.0, float(np.max(np.abs(matrix_np))))
    jitter_schedule = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    last_error: Exception | None = None
    identity = np.eye(matrix_np.shape[0], dtype=np.float64)
    for jitter_coeff in jitter_schedule:
        shifted = matrix_np if jitter_coeff == 0.0 else matrix_np + identity * (jitter_coeff * spectral_scale)
        try:
            eigvals = np.linalg.eigvalsh(shifted)
            status = "numpy_eigvalsh" if jitter_coeff == 0.0 else f"numpy_eigvalsh_jitter_{jitter_coeff:g}"
            return eigvals, status
        except np.linalg.LinAlgError as exc:
            last_error = exc
            try:
                eigvals_general = np.linalg.eigvals(shifted)
                imag_abs_max = float(np.max(np.abs(eigvals_general.imag))) if eigvals_general.size else 0.0
                real_abs_max = float(np.max(np.abs(eigvals_general.real))) if eigvals_general.size else 0.0
                if imag_abs_max <= max(1e-9, 1e-9 * max(1.0, real_abs_max)):
                    status = (
                        "numpy_eigvals_real"
                        if jitter_coeff == 0.0
                        else f"numpy_eigvals_real_jitter_{jitter_coeff:g}"
                    )
                    return np.sort(eigvals_general.real), status
            except np.linalg.LinAlgError:
                pass
            continue
    raise RuntimeError(f"failed symmetric eigensolve after jitter retries: {last_error}")


def projected_hessian_diagnostics(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    subspace_dim: int,
    topk: int,
    trace_probes: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    dim = int(z0.numel())
    if dim == 0:
        return {"subspace_dim": 0, "trace_est": 0.0, "eigvals": []}
    m = max(1, min(int(subspace_dim), dim))
    q = torch.randn(dim, m, generator=generator, device=z0.device, dtype=z0.dtype)
    q, _ = torch.linalg.qr(q, mode="reduced")
    hq_cols: list[torch.Tensor] = []
    for idx in range(m):
        hq_cols.append(hvp_at_flat_z(model, store, z0, q[:, idx], images, labels))
    hq = torch.stack(hq_cols, dim=1)
    h_proj = q.transpose(0, 1) @ hq
    h_proj = 0.5 * (h_proj + h_proj.transpose(0, 1))
    eig_solver_status = "ok"
    try:
        eigvals, eig_solver_status = _robust_symmetric_eigvals(h_proj)
    except Exception as exc:
        print(
            f"[latent_landscape] warning: projected Hessian eigensolve failed; "
            f"seeded subspace diagnostics degraded: {exc}",
            flush=True,
        )
        eigvals = np.asarray([], dtype=np.float64)
        eig_solver_status = f"failed:{type(exc).__name__}"

    trace_terms: list[float] = []
    for _ in range(max(1, int(trace_probes))):
        v = torch.randint(0, 2, (dim,), device=z0.device, dtype=torch.int64).to(dtype=z0.dtype)
        v = 2.0 * v - 1.0
        hv = hvp_at_flat_z(model, store, z0, v, images, labels)
        trace_terms.append(float(torch.dot(v, hv).detach().cpu().item()))
    finite_trace_terms = [value for value in trace_terms if math.isfinite(value)]
    trace_est = float(np.mean(finite_trace_terms)) if finite_trace_terms else float("nan")
    sorted_eigs = np.sort(eigvals)
    if len(sorted_eigs) == 0:
        return {
            "subspace_dim": int(m),
            "trace_est": float(trace_est),
            "negative_fraction": float("nan"),
            "eigvals": [],
            "top_eigs": [],
            "bottom_eigs": [],
            "eig_solver_status": str(eig_solver_status),
        }
    topk = max(1, min(int(topk), len(sorted_eigs)))
    return {
        "subspace_dim": int(m),
        "trace_est": float(trace_est),
        "negative_fraction": float(np.mean(sorted_eigs < 0.0)),
        "eigvals": [float(v) for v in sorted_eigs.tolist()],
        "top_eigs": [float(v) for v in sorted_eigs[-topk:].tolist()[::-1]],
        "bottom_eigs": [float(v) for v in sorted_eigs[:topk].tolist()],
        "eig_solver_status": str(eig_solver_status),
    }


def make_vit_cfg(cfg: LandscapeConfig) -> ViTTinyConfig:
    return ViTTinyConfig(
        image_size=int(cfg.image_size),
        patch_size=int(cfg.patch_size),
        in_channels=int(cfg.in_channels),
        num_classes=int(cfg.num_classes),
        hidden_dim=int(cfg.hidden_dim),
        depth=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        mlp_ratio=float(cfg.mlp_ratio),
        dropout=float(cfg.dropout),
        attention_dropout=float(cfg.attention_dropout),
    )


def build_latent_model(
    cfg: LandscapeConfig,
    *,
    device: torch.device,
    seed: int,
    calibration_images: torch.Tensor | None,
) -> FunctionalViTTiny:
    vit_cfg = make_vit_cfg(cfg)
    initial_tensors = make_initial_tensors(vit_cfg, seed=int(seed))
    big_vae = load_frozen_big_vae_decoder(cfg.big_vae_checkpoint, device=device)
    decoder_flow = None
    adapter_kind = str(cfg.big_vae_decoder_adapter).strip().lower()
    if adapter_kind not in {"", "identity", "none", "off", "false"}:
        if adapter_kind not in {"latent_flattening_flow", "flow", "ir_smoothing", "latent_smoothing"}:
            raise ValueError(f"unsupported --big_vae_decoder_adapter={cfg.big_vae_decoder_adapter!r}")
        decoder_flow, flow_cfg, payload = load_latent_flattening_flow_from_checkpoint(
            checkpoint_path=cfg.big_vae_decoder_adapter_checkpoint,
            model=big_vae,
            device=device,
        )
        adapter_big_vae_checkpoint = latent_flattening_payload_big_vae_checkpoint(payload)
        checkpoint_matches = bool(adapter_big_vae_checkpoint) and paths_match(
            adapter_big_vae_checkpoint,
            cfg.big_vae_checkpoint,
        )
        if adapter_big_vae_checkpoint and not checkpoint_matches:
            message = (
                "BigVAE decoder adapter was trained for a different checkpoint: "
                f"adapter_big_vae_checkpoint={adapter_big_vae_checkpoint} "
                f"landscape_big_vae_checkpoint={cfg.big_vae_checkpoint}"
            )
            if bool(cfg.big_vae_decoder_adapter_require_checkpoint_match):
                raise ValueError(message)
            print(f"[latent_landscape] WARNING: {message}", flush=True)
        print(
            "[latent_landscape] BigVAE decoder adapter ready: "
            f"kind={adapter_kind} flow_layers={int(flow_cfg.num_layers)} "
            f"hidden={int(flow_cfg.hidden_dim)} depth={int(flow_cfg.network_depth)} "
            f"checkpoint_matches={bool(checkpoint_matches)}",
            flush=True,
        )
    prior = None
    if str(cfg.big_vae_latent_init).strip().lower() == "diffusion_prior":
        if not str(cfg.big_vae_diffusion_prior_checkpoint).strip():
            raise ValueError("diffusion prior init requires --big-vae-diffusion-prior-checkpoint")
        prior = load_frozen_layer_latent_diffusion_prior(
            cfg.big_vae_diffusion_prior_checkpoint,
            device=device,
        )
        loaded_dist_encoder = load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint(
            cfg.big_vae_diffusion_prior_checkpoint,
            big_vae=big_vae,
        )
        if loaded_dist_encoder:
            print(
                "[latent_landscape] loaded finetuned distribution encoder state from diffusion prior checkpoint",
                flush=True,
            )

    model = FunctionalViTTiny(
        vit_cfg,
        initial_tensors,
        parameter_mode="bigvae_latent",
        big_vae=big_vae,
        big_vae_latent_init=str(cfg.big_vae_latent_init),
        big_vae_latent_parameterization=str(cfg.big_vae_latent_parameterization),
        big_vae_diffusion_prior=prior,
        big_vae_decoder_flow=decoder_flow,
        big_vae_diffusion_prior_steps=int(cfg.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(cfg.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(cfg.big_vae_diffusion_prior_eta),
        big_vae_decode=str(cfg.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.big_vae_tile_d_out),
        big_vae_encoder_context_rows=int(cfg.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(cfg.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(cfg.big_vae_encoder_batch_size),
    ).to(device)
    if calibration_images is not None and str(cfg.big_vae_latent_init).strip().lower() == "diffusion_prior":
        model.initialize_bigvae_diffusion_prior(calibration_images)
    return model


def plot_training_curves(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    plt = _get_pyplot()
    steps = [int(row["step"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    test_loss = [float(row["test_loss"]) for row in rows]
    test_acc = [float(row["test_acc"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, train_loss, label="train_loss")
    axes[0].plot(steps, test_loss, label="test_loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(steps, test_acc, label="test_acc")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_grouped_profiles(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_key: str,
    title: str,
    path: Path,
) -> None:
    if not rows:
        return
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = sorted({str(row[group_key]) for row in rows})
    for group in groups:
        subset = [row for row in rows if str(row[group_key]) == group]
        subset = sorted(subset, key=lambda item: float(item[x_key]))
        ax.plot(
            [float(item[x_key]) for item in subset],
            [float(item[y_key]) for item in subset],
            marker="o",
            label=group,
        )
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_histogram(values: list[float], *, title: str, xlabel: str, path: Path) -> None:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(finite_values, bins=24)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_radial_profile(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    checkpoint_name: str,
    scales: list[float],
) -> list[dict[str, Any]]:
    unit = normalized_or_none(z0)
    if unit is None:
        return []
    base_norm = float(z0.norm().detach().cpu().item())
    if base_norm <= 0.0:
        base_norm = 1.0
    rows: list[dict[str, Any]] = []
    for scale in scales:
        radius = float(base_norm) * float(scale)
        loss, acc = loss_acc_from_flat_z(model, store, unit * radius, images, labels)
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_name": str(checkpoint_name),
                "scale": float(scale),
                "radius": float(radius),
                "loss": float(loss.detach().cpu().item()),
                "acc": float(acc),
            }
        )
    return rows


def evaluate_angular_profiles(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    grad_z: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    checkpoint_name: str,
    theta_grid: list[float],
    random_dirs: int,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    unit = normalized_or_none(z0)
    if unit is None:
        return []
    radius = float(z0.norm().detach().cpu().item())
    directions: list[tuple[str, torch.Tensor]] = []
    grad_parallel, grad_tangent = project_parallel_and_tangent(grad_z, z0)
    del grad_parallel
    grad_tangent_unit = normalized_or_none(grad_tangent)
    if grad_tangent_unit is not None:
        directions.append(("grad_tangent", grad_tangent_unit))
    for idx, direction in enumerate(
        orthonormal_random_tangent_directions(z0, count=int(random_dirs), generator=generator)
    ):
        directions.append((f"rand_tangent_{idx}", direction))
    rows: list[dict[str, Any]] = []
    for direction_id, direction in directions:
        for theta in theta_grid:
            z_theta = radius * (math.cos(float(theta)) * unit + math.sin(float(theta)) * direction)
            loss, acc = loss_acc_from_flat_z(model, store, z_theta, images, labels)
            rows.append(
                {
                    "seed": int(seed),
                    "checkpoint_name": str(checkpoint_name),
                    "direction_id": str(direction_id),
                    "theta": float(theta),
                    "loss": float(loss.detach().cpu().item()),
                    "acc": float(acc),
                }
            )
    return rows


def evaluate_gradient_decomposition(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    checkpoint_name: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    z = z0.detach().clone().requires_grad_(True)
    loss, acc, grad_z, _ = gradient_and_optional_weight_gradients(
        model,
        store,
        z,
        images,
        labels,
        create_graph=False,
        return_weight_grads=False,
    )
    grad_norm = float(grad_z.norm().detach().cpu().item())
    z_norm = float(z0.norm().detach().cpu().item())
    tiny = max(float(torch.finfo(z.dtype).eps), 1e-12)
    g_parallel, g_tangent = project_parallel_and_tangent(grad_z, z0)
    row = {
        "seed": int(seed),
        "checkpoint_name": str(checkpoint_name),
        "loss": float(loss.detach().cpu().item()),
        "acc": float(acc),
        "z_norm": float(z_norm),
        "grad_norm": float(grad_norm),
        "grad_parallel_ratio": float(g_parallel.norm().detach().cpu().item() / max(grad_norm, tiny)),
        "grad_tangent_ratio": float(g_tangent.norm().detach().cpu().item() / max(grad_norm, tiny)),
        "grad_cos_with_z": float(torch.dot(grad_z, z0).detach().cpu().item() / max(grad_norm * z_norm, tiny))
        if grad_norm > 0.0 and z_norm > 0.0
        else float("nan"),
    }
    return row, grad_z.detach()


def evaluate_decoder_jacobian_fd(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    grad_z: torch.Tensor,
    update_direction: torch.Tensor | None,
    *,
    seed: int,
    checkpoint_name: str,
    eps_rel: float,
    tangent_dirs: int,
    generator: torch.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unit = normalized_or_none(z0)
    base_norm = float(z0.norm().detach().cpu().item())
    eps_abs = float(eps_rel) * max(base_norm, 1.0)
    rows: list[dict[str, Any]] = []
    tangent_values: list[float] = []
    summary = {
        "J_radial": float("nan"),
        "J_tangent_mean": float("nan"),
        "J_tangent_std": float("nan"),
        "J_tangent_max": float("nan"),
        "J_grad": float("nan"),
        "J_update": float("nan"),
    }
    if unit is None:
        return rows, summary

    radial_value = estimate_decoder_fd_direction(model, store, z0, unit, eps_abs=eps_abs)
    summary["J_radial"] = radial_value
    rows.append(
        {
            "seed": int(seed),
            "checkpoint_name": str(checkpoint_name),
            "direction_id": "radial",
            "value": float(radial_value),
            "eps_abs": float(eps_abs),
        }
    )

    for idx, tangent in enumerate(
        orthonormal_random_tangent_directions(z0, count=int(tangent_dirs), generator=generator)
    ):
        value = estimate_decoder_fd_direction(model, store, z0, tangent, eps_abs=eps_abs)
        tangent_values.append(value)
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_name": str(checkpoint_name),
                "direction_id": f"tangent_{idx}",
                "value": float(value),
                "eps_abs": float(eps_abs),
            }
        )
    if tangent_values:
        arr = np.asarray(tangent_values, dtype=np.float64)
        summary["J_tangent_mean"] = float(arr.mean())
        summary["J_tangent_std"] = float(arr.std())
        summary["J_tangent_max"] = float(arr.max())

    grad_unit = normalized_or_none(grad_z)
    if grad_unit is not None:
        grad_value = estimate_decoder_fd_direction(model, store, z0, grad_unit, eps_abs=eps_abs)
        summary["J_grad"] = float(grad_value)
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_name": str(checkpoint_name),
                "direction_id": "grad",
                "value": float(grad_value),
                "eps_abs": float(eps_abs),
            }
        )

    update_unit = normalized_or_none(update_direction) if update_direction is not None else None
    if update_unit is not None:
        update_value = estimate_decoder_fd_direction(model, store, z0, update_unit, eps_abs=eps_abs)
        summary["J_update"] = float(update_value)
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_name": str(checkpoint_name),
                "direction_id": "update",
                "value": float(update_value),
                "eps_abs": float(eps_abs),
            }
        )
    return rows, summary


def evaluate_accessibility(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    checkpoint_name: str,
    eps_rel: float,
) -> dict[str, Any]:
    z = z0.detach().clone().requires_grad_(True)
    loss, acc, grad_z, raw_weight_grad = gradient_and_optional_weight_gradients(
        model,
        store,
        z,
        images,
        labels,
        create_graph=False,
        return_weight_grads=True,
    )
    grad_unit = normalized_or_none(grad_z)
    eps_abs = float(eps_rel) * max(float(z0.norm().detach().cpu().item()), 1.0)
    if grad_unit is None or raw_weight_grad is None:
        return {
            "seed": int(seed),
            "checkpoint_name": str(checkpoint_name),
            "loss": float(loss.detach().cpu().item()),
            "acc": float(acc),
            "cos_deltaW_latent_vs_raw": float("nan"),
            "eps_abs": float(eps_abs),
        }

    z1 = z0 - float(eps_abs) * grad_unit.detach()
    w0 = flatten_decoded_tensors(
        decode_all_tensors_from_latent_state(store, unflatten_latent_state(store, z0))
    )
    w1 = flatten_decoded_tensors(
        decode_all_tensors_from_latent_state(store, unflatten_latent_state(store, z1))
    )
    delta_w_latent = w1 - w0
    delta_w_raw = -raw_weight_grad.detach()
    tiny = max(float(torch.finfo(delta_w_latent.dtype).eps), 1e-12)
    cosine = float(
        torch.dot(delta_w_latent, delta_w_raw).detach().cpu().item()
        / max(
            float(delta_w_latent.norm().detach().cpu().item() * delta_w_raw.norm().detach().cpu().item()),
            tiny,
        )
    )
    return {
        "seed": int(seed),
        "checkpoint_name": str(checkpoint_name),
        "loss": float(loss.detach().cpu().item()),
        "acc": float(acc),
        "cos_deltaW_latent_vs_raw": float(cosine),
        "eps_abs": float(eps_abs),
    }


def evaluate_2d_slice(
    model: FunctionalViTTiny,
    store: BigVAELatentTensorStore,
    z0: torch.Tensor,
    grad_z: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    checkpoint_name: str,
    grid_size: int,
    scale: float,
    generator: torch.Generator,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    unit = normalized_or_none(z0)
    if unit is None:
        return [], None
    grad_parallel, grad_tangent = project_parallel_and_tangent(grad_z, z0)
    del grad_parallel
    d1 = normalized_or_none(grad_tangent)
    if d1 is None:
        tangent_dirs = orthonormal_random_tangent_directions(z0, count=1, generator=generator)
        if not tangent_dirs:
            return [], None
        d1 = tangent_dirs[0]
    random_dirs = orthonormal_random_tangent_directions(z0, count=2, generator=generator)
    d2 = None
    for candidate in random_dirs:
        candidate = candidate - torch.dot(candidate, d1) * d1
        d2 = normalized_or_none(candidate)
        if d2 is not None:
            break
    if d2 is None:
        return [], None

    alphas = torch.linspace(-float(scale), float(scale), int(grid_size), device=z0.device, dtype=z0.dtype)
    betas = torch.linspace(-float(scale), float(scale), int(grid_size), device=z0.device, dtype=z0.dtype)
    loss_grid = torch.zeros(len(alphas), len(betas), device=z0.device, dtype=z0.dtype)
    acc_grid = torch.zeros(len(alphas), len(betas), device=z0.device, dtype=z0.dtype)
    rows: list[dict[str, Any]] = []
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            z = z0 + alpha * d1 + beta * d2
            loss, acc = loss_acc_from_flat_z(model, store, z, images, labels)
            loss_value = float(loss.detach().cpu().item())
            acc_value = float(acc)
            loss_grid[i, j] = float(loss_value)
            acc_grid[i, j] = float(acc_value)
            rows.append(
                {
                    "seed": int(seed),
                    "checkpoint_name": str(checkpoint_name),
                    "alpha": float(alpha.detach().cpu().item()),
                    "beta": float(beta.detach().cpu().item()),
                    "loss": float(loss_value),
                    "acc": float(acc_value),
                }
            )
    arrays = {
        "alphas": alphas.detach().cpu().numpy(),
        "betas": betas.detach().cpu().numpy(),
        "loss": loss_grid.detach().cpu().numpy(),
        "acc": acc_grid.detach().cpu().numpy(),
    }
    return rows, arrays


def save_2d_slice_plot(arrays: dict[str, np.ndarray], *, title: str, path: Path) -> None:
    plt = _get_pyplot()
    alphas = arrays["alphas"]
    betas = arrays["betas"]
    loss = arrays["loss"]
    aa, bb = np.meshgrid(betas, alphas)
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(bb, aa, loss, levels=24, cmap="magma")
    fig.colorbar(contour, ax=ax, label="loss")
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_training_for_seed(
    cfg: LandscapeConfig,
    *,
    seed: int,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(int(seed))
    train_loader, train_diag_loader, test_loader = build_loaders(cfg, device=device)
    diagnostic_source = train_diag_loader if str(cfg.diagnostic_split).strip().lower() == "train" else test_loader
    diag_images, diag_labels = collect_fixed_batch(
        diagnostic_source,
        batch_size=int(cfg.diag_batch_size),
        device=device,
    )
    calibration_images = collect_fixed_batch(
        train_diag_loader,
        batch_size=int(cfg.batch_size) * max(1, int(cfg.big_vae_init_calibration_batches)),
        device=device,
    )[0] if str(cfg.big_vae_latent_init).strip().lower() == "diffusion_prior" else None

    model = build_latent_model(
        cfg,
        device=device,
        seed=int(seed),
        calibration_images=calibration_images,
    )
    target = getattr(model, "_orig_mod", model)
    store = target.store
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("diagnostic script expects BigVAELatentTensorStore-backed model")

    decoded_params = int(store.decoded_numel())
    latent_params = int(store.latent_numel())
    tile_count = int(store.decoded_tile_count())
    big_vae_decoded_params = int(store.big_vae_decoded_numel())
    latent_per_tile = (
        int(next(iter(store.latent_slots.values())).numel())
        if len(store.latent_slots) > 0
        else 0
    )
    latent_to_decoded_ratio = float(latent_params / max(1, decoded_params))

    print(
        f"[latent_landscape][seed={seed}] decoded_params={decoded_params} "
        f"latent_params={latent_params} latent/raw={latent_to_decoded_ratio:.6f} "
        f"tiles={tile_count} latent_per_tile={latent_per_tile} "
        f"bigvae_decoded_params={big_vae_decoded_params}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        store.latent_slots.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = run_dir / "diagnostics"
    plots_dir = diagnostics_dir / "plots"
    checkpoints_file = run_dir / "checkpoints.pt"
    train_cache_file = run_dir / "train_cache.pt"
    train_log_csv = run_dir / "train_log.csv"
    write_json(
        run_dir / "config.json",
        {
            "seed": int(seed),
            "config": asdict(cfg),
            "device": str(device),
            "vit_model": asdict(make_vit_cfg(cfg)),
        },
    )

    cached_training = load_training_cache_if_compatible(
        train_cache_file,
        cfg=cfg,
        seed=int(seed),
    )
    if cached_training is not None:
        print(
            f"[latent_landscape][seed={seed}] reusing cached training state from {train_cache_file}",
            flush=True,
        )
        checkpoints = cached_training["checkpoints"]
        summary = dict(cached_training["summary"])
        torch.save({"seed": int(seed), "checkpoints": checkpoints}, checkpoints_file)
    else:
        checkpoints: dict[str, dict[str, Any]] = {}
        checkpoints["init"] = capture_checkpoint_state(store, name="init", step=0, update_direction=None)

        train_rows: list[dict[str, Any]] = []
        best_acc = -float("inf")
        best_step = 0
        plateau_name: str | None = None
        last_scheduled_name: str | None = None
        last_test_loss = float("nan")
        last_test_acc = float("nan")
        max_steps = max(1, int(cfg.steps))
        global_step = 0
        start_time = time.time()
        window_loss_sum = 0.0
        window_examples = 0

        maybe_unlink(train_log_csv)
        for epoch_idx in range(max(1, int(cfg.epochs))):
            model.train()
            for images, labels in train_loader:
                global_step += 1
                images = images.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                pre_step_z = flatten_materialized_latents(store)
                pre_step_w = flatten_current_decoded_tensors(store)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                if float(cfg.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(store.latent_slots.parameters(), float(cfg.grad_clip_norm))
                optimizer.step()

                post_step_z = flatten_materialized_latents(store)
                post_step_w = flatten_current_decoded_tensors(store)
                delta_z = post_step_z - pre_step_z
                delta_w = post_step_w - pre_step_w
                delta_z_norm = float(delta_z.norm().detach().cpu().item())
                delta_w_norm = float(delta_w.norm().detach().cpu().item())
                z_norm = float(post_step_z.norm().detach().cpu().item())
                tiny = max(float(torch.finfo(post_step_z.dtype).eps), 1e-12)
                delta_w_over_delta_z = float(delta_w_norm / max(delta_z_norm, tiny))
                delta_parallel, delta_tangent = project_parallel_and_tangent(delta_z, post_step_z)
                cos_delta_z = (
                    float(torch.dot(delta_z, post_step_z).detach().cpu().item() / max(delta_z_norm * z_norm, tiny))
                    if delta_z_norm > 0.0 and z_norm > 0.0
                    else float("nan")
                )
                delta_parallel_ratio = float(delta_parallel.norm().detach().cpu().item() / max(delta_z_norm, tiny))
                delta_tangent_ratio = float(delta_tangent.norm().detach().cpu().item() / max(delta_z_norm, tiny))

                batch_examples = int(labels.numel())
                window_loss_sum += float(loss.detach().cpu().item()) * batch_examples
                window_examples += batch_examples

                should_log = global_step == 1 or global_step % max(1, int(cfg.log_every)) == 0
                should_eval = global_step == 1 or global_step % max(1, int(cfg.eval_every)) == 0 or global_step >= max_steps

                grad_z_norm = float("nan")
                if should_log or should_eval:
                    z_for_grad = post_step_z.detach().clone().requires_grad_(True)
                    _loss_now, _acc_now, grad_z_now, _ = gradient_and_optional_weight_gradients(
                        model,
                        store,
                        z_for_grad,
                        images,
                        labels,
                        create_graph=False,
                        return_weight_grads=False,
                    )
                    grad_z_norm = float(grad_z_now.norm().detach().cpu().item())

                if should_eval:
                    last_test_loss, last_test_acc = evaluate_model(model, test_loader, device=device)
                    if last_test_acc > best_acc:
                        best_acc = float(last_test_acc)
                        best_step = int(global_step)
                        checkpoints["best_val"] = capture_checkpoint_state(
                            store,
                            name="best_val",
                            step=global_step,
                            update_direction=delta_z,
                        )

                if global_step == 1:
                    checkpoints["after_step_1"] = capture_checkpoint_state(
                        store,
                        name="after_step_1",
                        step=global_step,
                        update_direction=delta_z,
                    )
                if global_step in set(int(step) for step in cfg.checkpoint_steps):
                    checkpoint_name = f"step_{global_step}"
                    checkpoints[checkpoint_name] = capture_checkpoint_state(
                        store,
                        name=checkpoint_name,
                        step=global_step,
                        update_direction=delta_z,
                    )
                    last_scheduled_name = checkpoint_name

                if should_log or should_eval:
                    avg_train_loss = window_loss_sum / max(1, window_examples)
                    row = {
                        "seed": int(seed),
                        "step": int(global_step),
                        "epoch": int(epoch_idx + 1),
                        "train_loss": float(avg_train_loss),
                        "test_loss": float(last_test_loss),
                        "test_acc": float(last_test_acc),
                        "best_acc": float(best_acc if best_acc > -float("inf") else float("nan")),
                        "z_norm": float(z_norm),
                        "grad_z_norm": float(grad_z_norm),
                        "delta_z_norm": float(delta_z_norm),
                        "delta_w_norm": float(delta_w_norm),
                        "delta_w_over_delta_z": float(delta_w_over_delta_z),
                        "cos_delta_z_z": float(cos_delta_z),
                        "delta_z_parallel_ratio": float(delta_parallel_ratio),
                        "delta_z_tangent_ratio": float(delta_tangent_ratio),
                        "elapsed_s": float(time.time() - start_time),
                    }
                    train_rows.append(row)
                    append_csv_row(train_log_csv, row)
                    print(
                        f"[latent_landscape][seed={seed}] step={global_step} "
                        f"train_loss={avg_train_loss:.4f} test_loss={last_test_loss:.4f} test_acc={last_test_acc:.4f} "
                        f"best={best_acc:.4f} ||z||={z_norm:.6e} ||grad_z||={grad_z_norm:.6e} "
                        f"||Δz||={delta_z_norm:.6e} ||ΔW||={delta_w_norm:.6e} "
                        f"||ΔW||/||Δz||={delta_w_over_delta_z:.6e} cos(Δz,z)={cos_delta_z:.6e} "
                        f"||Δz_parallel||/||Δz||={delta_parallel_ratio:.6e} "
                        f"||Δz_tangent||/||Δz||={delta_tangent_ratio:.6e}",
                        flush=True,
                    )
                    window_loss_sum = 0.0
                    window_examples = 0

                if global_step >= max_steps:
                    break
            if global_step >= max_steps:
                break

        final_test_loss, final_test_acc = evaluate_model(model, test_loader, device=device)
        checkpoints["final"] = capture_checkpoint_state(
            store,
            name="final",
            step=global_step,
            update_direction=None,
        )
        if last_scheduled_name is not None and last_scheduled_name in checkpoints:
            checkpoints["plateau_candidate"] = {
                **checkpoints[last_scheduled_name],
                "name": "plateau_candidate",
            }
            plateau_name = last_scheduled_name
        else:
            checkpoints["plateau_candidate"] = {
                **checkpoints["final"],
                "name": "plateau_candidate",
            }
            plateau_name = "final"

        torch.save({"seed": int(seed), "checkpoints": checkpoints}, checkpoints_file)
        plot_training_curves(train_rows, plots_dir / "training_curves.png")
        summary = {
            "seed": int(seed),
            "best_step": int(best_step),
            "best_test_acc": float(best_acc if best_acc > -float("inf") else float("nan")),
            "final_test_loss": float(final_test_loss),
            "final_test_acc": float(final_test_acc),
            "saved_checkpoints": sorted(checkpoints.keys()),
            "plateau_source": str(plateau_name),
        }

    summary.update(
        {
            "decoded_params": int(decoded_params),
            "latent_params": int(latent_params),
            "latent_to_decoded_ratio": float(latent_to_decoded_ratio),
            "tile_count": int(tile_count),
            "latent_per_tile": int(latent_per_tile),
            "big_vae_decoded_params": int(big_vae_decoded_params),
        }
    )
    write_json(run_dir / "summary.json", summary)
    torch.save(
        {
            "seed": int(seed),
            "config": asdict(cfg),
            "summary": summary,
            "checkpoints": checkpoints,
        },
        train_cache_file,
    )

    radial_rows: list[dict[str, Any]] = []
    angular_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    jacobian_rows: list[dict[str, Any]] = []
    accessibility_rows: list[dict[str, Any]] = []
    hessian_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    maybe_unlink(run_dir / "aggregate_summary_seed.csv")
    for filename in [
        "gradient_decomp.csv",
        "radial.csv",
        "angular.csv",
        "decoder_jacobian_fd.csv",
        "accessibility.csv",
        "hessian.csv",
        "slice2d.csv",
    ]:
        maybe_unlink(diagnostics_dir / filename)

    diag_generator = torch.Generator(device=device)
    diag_generator.manual_seed(int(seed) + 1000)

    for checkpoint_name, payload in checkpoints.items():
        z0 = payload["flat_z"].to(device=device, dtype=store.big_vae.latent_base.dtype)
        update_direction = payload.get("update_direction")
        update_direction = (
            None
            if update_direction is None
            else update_direction.to(device=device, dtype=store.big_vae.latent_base.dtype)
        )

        grad_row, grad_z = evaluate_gradient_decomposition(
            model,
            store,
            z0,
            diag_images,
            diag_labels,
            seed=int(seed),
            checkpoint_name=str(checkpoint_name),
        )
        gradient_rows.append(grad_row)
        append_csv_row(diagnostics_dir / "gradient_decomp.csv", grad_row)

        radial = evaluate_radial_profile(
            model,
            store,
            z0,
            diag_images,
            diag_labels,
            seed=int(seed),
            checkpoint_name=str(checkpoint_name),
            scales=list(cfg.radial_scales),
        )
        radial_rows.extend(radial)
        for row in radial:
            append_csv_row(diagnostics_dir / "radial.csv", row)
        plot_grouped_profiles(
            radial,
            x_key="scale",
            y_key="loss",
            group_key="checkpoint_name",
            title=f"Radial profile: {checkpoint_name}",
            path=plots_dir / f"radial_{checkpoint_name}.png",
        )

        angular = evaluate_angular_profiles(
            model,
            store,
            z0,
            grad_z,
            diag_images,
            diag_labels,
            seed=int(seed),
            checkpoint_name=str(checkpoint_name),
            theta_grid=list(cfg.theta_grid),
            random_dirs=int(cfg.angular_random_dirs),
            generator=diag_generator,
        )
        angular_rows.extend(angular)
        for row in angular:
            append_csv_row(diagnostics_dir / "angular.csv", row)
        plot_grouped_profiles(
            angular,
            x_key="theta",
            y_key="loss",
            group_key="direction_id",
            title=f"Angular profile: {checkpoint_name}",
            path=plots_dir / f"angular_{checkpoint_name}.png",
        )

        jac_rows, jac_summary = evaluate_decoder_jacobian_fd(
            model,
            store,
            z0,
            grad_z,
            update_direction,
            seed=int(seed),
            checkpoint_name=str(checkpoint_name),
            eps_rel=float(cfg.jacobian_eps_rel),
            tangent_dirs=int(cfg.jacobian_tangent_dirs),
            generator=diag_generator,
        )
        jacobian_rows.extend(jac_rows)
        for row in jac_rows:
            append_csv_row(diagnostics_dir / "decoder_jacobian_fd.csv", row)
        plot_histogram(
            [float(row["value"]) for row in jac_rows if str(row["direction_id"]).startswith("tangent_")],
            title=f"Decoder J_fd tangent distribution: {checkpoint_name}",
            xlabel="J_fd",
            path=plots_dir / f"jacobian_fd_{checkpoint_name}.png",
        )

        accessibility_row = None
        if bool(cfg.compute_accessibility):
            accessibility_row = evaluate_accessibility(
                model,
                store,
                z0,
                diag_images,
                diag_labels,
                seed=int(seed),
                checkpoint_name=str(checkpoint_name),
                eps_rel=float(cfg.jacobian_eps_rel),
            )
            accessibility_rows.append(accessibility_row)
            append_csv_row(diagnostics_dir / "accessibility.csv", accessibility_row)

        hessian_row = None
        if bool(cfg.compute_hessian):
            hessian_diag = projected_hessian_diagnostics(
                model,
                store,
                z0,
                diag_images,
                diag_labels,
                subspace_dim=int(cfg.hessian_subspace_dim),
                topk=int(cfg.hessian_topk),
                trace_probes=int(cfg.hessian_trace_probes),
                generator=diag_generator,
            )
            hessian_row = {
                "seed": int(seed),
                "checkpoint_name": str(checkpoint_name),
                "subspace_dim": int(hessian_diag["subspace_dim"]),
                "trace_est": float(hessian_diag["trace_est"]),
                "negative_fraction": float(hessian_diag["negative_fraction"]),
                "eig_solver_status": str(hessian_diag.get("eig_solver_status", "unknown")),
                "top_eigs_json": json.dumps(hessian_diag["top_eigs"]),
                "bottom_eigs_json": json.dumps(hessian_diag["bottom_eigs"]),
                "eigvals_json": json.dumps(hessian_diag["eigvals"]),
            }
            hessian_rows.append(hessian_row)
            append_csv_row(diagnostics_dir / "hessian.csv", hessian_row)
            plot_histogram(
                [float(v) for v in hessian_diag["eigvals"]],
                title=f"Projected Hessian eigs: {checkpoint_name}",
                xlabel="eigenvalue",
                path=plots_dir / f"hessian_{checkpoint_name}.png",
            )

        if bool(cfg.compute_2d_slices):
            slice_entries, arrays = evaluate_2d_slice(
                model,
                store,
                z0,
                grad_z,
                diag_images,
                diag_labels,
                seed=int(seed),
                checkpoint_name=str(checkpoint_name),
                grid_size=int(cfg.slice_grid_size),
                scale=float(cfg.slice_scale),
                generator=diag_generator,
            )
            slice_rows.extend(slice_entries)
            for row in slice_entries:
                append_csv_row(diagnostics_dir / "slice2d.csv", row)
            if arrays is not None:
                np.savez(
                    diagnostics_dir / f"slice2d_{checkpoint_name}.npz",
                    **arrays,
                )
                save_2d_slice_plot(
                    arrays,
                    title=f"2D loss slice: {checkpoint_name}",
                    path=plots_dir / f"slice2d_{checkpoint_name}.png",
                )

        center_loss, center_acc = loss_acc_from_flat_z(model, store, z0, diag_images, diag_labels)
        aggregate_rows.append(
            {
                "seed": int(seed),
                "checkpoint_name": str(checkpoint_name),
                "checkpoint_step": int(payload["step"]),
                "diag_loss": float(center_loss.detach().cpu().item()),
                "diag_acc": float(center_acc),
                "decoded_params": int(decoded_params),
                "latent_params": int(latent_params),
                "latent_to_decoded_ratio": float(latent_to_decoded_ratio),
                "tile_count": int(tile_count),
                "latent_per_tile": int(latent_per_tile),
                "big_vae_decoded_params": int(big_vae_decoded_params),
                "latent_norm": float(payload["latent_norm"]),
                "grad_norm": float(grad_row["grad_norm"]),
                "grad_parallel_ratio": float(grad_row["grad_parallel_ratio"]),
                "grad_tangent_ratio": float(grad_row["grad_tangent_ratio"]),
                "J_radial": float(jac_summary["J_radial"]),
                "J_tangent_mean": float(jac_summary["J_tangent_mean"]),
                "J_tangent_std": float(jac_summary["J_tangent_std"]),
                "J_tangent_max": float(jac_summary["J_tangent_max"]),
                "J_grad": float(jac_summary["J_grad"]),
                "J_update": float(jac_summary["J_update"]),
                "accessibility_cos": (
                    float(accessibility_row["cos_deltaW_latent_vs_raw"])
                    if accessibility_row is not None
                    else float("nan")
                ),
                "hessian_trace_est": float(hessian_row["trace_est"]) if hessian_row is not None else float("nan"),
                "hessian_negative_fraction": (
                    float(hessian_row["negative_fraction"]) if hessian_row is not None else float("nan")
                ),
            }
        )

    for row in aggregate_rows:
        append_csv_row(run_dir / "aggregate_summary_seed.csv", row)
    return {
        "summary": summary,
        "aggregate_rows": aggregate_rows,
    }


def parse_args() -> LandscapeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose latent-space loss landscapes for TinyViT on CIFAR-10 using a frozen BigVAE decoder. "
            "The resulting file can be run standalone or pasted into a notebook cell."
        )
    )
    parser.add_argument("--data_dir", default="./data/cifar10")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--train_subset", type=int, default=10000)
    parser.add_argument("--test_subset", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--diag_batch_size", type=int, default=1024)
    parser.add_argument("--diagnostic_split", choices=("train", "test"), default="train")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=100.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--checkpoint_steps", nargs="+", type=int, default=list(DEFAULT_CHECKPOINT_STEPS))

    parser.add_argument("--vae_ckpt", "--big_vae_checkpoint", dest="big_vae_checkpoint", required=True)
    parser.add_argument(
        "--big_vae_latent_init",
        choices=("base", "random", "encoded", "diffusion_prior"),
        default="diffusion_prior",
    )
    parser.add_argument(
        "--big_vae_latent_parameterization",
        choices=("euclidean", "sphere"),
        default="sphere",
    )
    parser.add_argument(
        "--prior_ckpt",
        "--big_vae_diffusion_prior_checkpoint",
        dest="big_vae_diffusion_prior_checkpoint",
        default="",
    )
    parser.add_argument("--big_vae_diffusion_prior_steps", type=int, default=50)
    parser.add_argument("--big_vae_diffusion_prior_sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--big_vae_diffusion_prior_eta", type=float, default=0.0)
    parser.add_argument("--big_vae_decode", choices=("weights", "all"), default="weights")
    parser.add_argument("--big_vae_tile_T_patches", type=int, default=4)
    parser.add_argument("--big_vae_tile_d_out", type=int, default=64)
    parser.add_argument("--big_vae_decoder_adapter", default="identity")
    parser.add_argument("--big_vae_decoder_adapter_checkpoint", default="")
    parser.add_argument("--big_vae_decoder_adapter_require_checkpoint_match", action="store_true")
    parser.add_argument("--big_vae_encoder_context_rows", type=int, default=64)
    parser.add_argument("--big_vae_encoder_context_std", type=float, default=1.0)
    parser.add_argument("--big_vae_encoder_batch_size", type=int, default=16)
    parser.add_argument("--big_vae_init_calibration_batches", type=int, default=1)

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=3)

    parser.add_argument("--radial_scales", nargs="+", type=float, default=list(DEFAULT_RADIAL_SCALES))
    parser.add_argument("--theta_grid", nargs="+", type=float, default=list(DEFAULT_THETA_GRID))
    parser.add_argument("--angular_random_dirs", type=int, default=4)
    parser.add_argument("--jacobian_eps_rel", type=float, default=1e-3)
    parser.add_argument("--jacobian_tangent_dirs", type=int, default=8)
    parser.add_argument("--compute_accessibility", action="store_true")
    parser.add_argument("--compute_hessian", action="store_true")
    parser.add_argument("--hessian_subspace_dim", type=int, default=128)
    parser.add_argument("--hessian_topk", type=int, default=5)
    parser.add_argument("--hessian_trace_probes", type=int, default=16)
    parser.add_argument("--compute_2d_slices", action="store_true")
    parser.add_argument("--slice_grid_size", type=int, default=31)
    parser.add_argument("--slice_scale", type=float, default=1.0)

    args = parser.parse_args()
    if str(args.big_vae_latent_init).strip().lower() == "diffusion_prior" and not str(
        args.big_vae_diffusion_prior_checkpoint
    ).strip():
        raise ValueError(
            "--big_vae_diffusion_prior_checkpoint/--prior_ckpt is required when --big_vae_latent_init=diffusion_prior"
        )
    decoder_adapter = str(args.big_vae_decoder_adapter).strip().lower()
    decoder_adapter_checkpoint = str(args.big_vae_decoder_adapter_checkpoint or "").strip()
    if decoder_adapter_checkpoint and decoder_adapter in {"", "identity", "none", "off", "false"}:
        decoder_adapter = "latent_flattening_flow"
    if decoder_adapter not in {"", "identity", "none", "off", "false"}:
        if not decoder_adapter_checkpoint:
            raise ValueError("--big_vae_decoder_adapter_checkpoint is required when decoder adapter is enabled")
        if str(args.big_vae_latent_init).strip().lower() != "diffusion_prior":
            raise ValueError("--big_vae_decoder_adapter currently requires --big_vae_latent_init=diffusion_prior")

    return LandscapeConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        device=str(args.device),
        download=bool(args.download),
        seeds=[int(seed) for seed in args.seeds],
        train_subset=int(args.train_subset),
        test_subset=int(args.test_subset),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        diag_batch_size=int(args.diag_batch_size),
        diagnostic_split=str(args.diagnostic_split),
        steps=int(args.steps),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.log_every),
        eval_every=int(args.eval_every),
        checkpoint_steps=[int(step) for step in args.checkpoint_steps],
        big_vae_checkpoint=str(args.big_vae_checkpoint),
        big_vae_latent_init=str(args.big_vae_latent_init),
        big_vae_latent_parameterization=str(args.big_vae_latent_parameterization),
        big_vae_diffusion_prior_checkpoint=str(args.big_vae_diffusion_prior_checkpoint),
        big_vae_diffusion_prior_steps=int(args.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(args.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(args.big_vae_diffusion_prior_eta),
        big_vae_decode=str(args.big_vae_decode),
        big_vae_tile_T_patches=int(args.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(args.big_vae_tile_d_out),
        big_vae_decoder_adapter=decoder_adapter,
        big_vae_decoder_adapter_checkpoint=decoder_adapter_checkpoint,
        big_vae_decoder_adapter_require_checkpoint_match=bool(args.big_vae_decoder_adapter_require_checkpoint_match),
        big_vae_encoder_context_rows=int(args.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(args.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(args.big_vae_encoder_batch_size),
        big_vae_init_calibration_batches=int(args.big_vae_init_calibration_batches),
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
        num_classes=int(args.num_classes),
        in_channels=int(args.in_channels),
        radial_scales=[float(v) for v in args.radial_scales],
        theta_grid=[float(v) for v in args.theta_grid],
        angular_random_dirs=int(args.angular_random_dirs),
        jacobian_eps_rel=float(args.jacobian_eps_rel),
        jacobian_tangent_dirs=int(args.jacobian_tangent_dirs),
        compute_accessibility=bool(args.compute_accessibility),
        compute_hessian=bool(args.compute_hessian),
        hessian_subspace_dim=int(args.hessian_subspace_dim),
        hessian_topk=int(args.hessian_topk),
        hessian_trace_probes=int(args.hessian_trace_probes),
        compute_2d_slices=bool(args.compute_2d_slices),
        slice_grid_size=int(args.slice_grid_size),
        slice_scale=float(args.slice_scale),
    )


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    write_json(output_dir / "config.json", {"config": asdict(cfg), "device": str(device)})
    aggregate_summary_path = output_dir / "aggregate_summary.csv"
    maybe_unlink(aggregate_summary_path)
    all_summaries: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        print(
            f"[latent_landscape] starting seed={seed} device={device} lr={cfg.lr:g} "
            f"latent_init={cfg.big_vae_latent_init} latent_parameterization={cfg.big_vae_latent_parameterization}",
            flush=True,
        )
        run_dir = output_dir / "runs" / f"seed_{int(seed)}"
        result = run_training_for_seed(
            cfg,
            seed=int(seed),
            run_dir=run_dir,
            device=device,
        )
        all_summaries.append(result["summary"])
        for row in result["aggregate_rows"]:
            append_csv_row(aggregate_summary_path, row)

    write_json(output_dir / "aggregate_summary.json", {"runs": all_summaries})
    print(f"[latent_landscape] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
