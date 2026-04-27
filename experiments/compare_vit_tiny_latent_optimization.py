from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.weight_quantile_vae import BigWeightVAE
from training.big_vae_latent_diffusion import (
    load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint,
    load_frozen_big_vae_from_checkpoint,
    load_frozen_layer_latent_diffusion_prior,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(slots=True)
class ViTTinyConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    hidden_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0


@dataclass(slots=True)
class ExperimentConfig:
    output_dir: str = "./artifacts/vit_tiny_latent_optimization"
    data_dir: str = "./data/cifar10"
    download: bool = True
    setup: str = "both"
    device: str = "auto"
    seed: int = 42
    epochs: int = 20
    max_steps: int = 0
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    train_subset: int = 0
    test_subset: int = 0
    direct_lr: float = 3e-4
    latent_lr: float = 1e-2
    direct_optimizer: str = "adamw"
    latent_optimizer: str = "adamw"
    direct_weight_decay: float = 0.05
    latent_weight_decay: float = 0.01
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    label_smoothing: float = 0.0
    grad_clip_norm: float = 0.0
    amp: bool = True
    tf32: bool = True
    compile: bool = False
    log_every_steps: int = 50
    eval_every_steps: int = 500
    save_checkpoints: bool = True
    latent_rank: int = 8
    latent_delta_scale: float = 1.0
    latent_factor_init_std: float = 0.02
    big_vae_checkpoint: str = ""
    big_vae_latent_init: str = "random"
    big_vae_diffusion_prior_checkpoint: str = ""
    big_vae_diffusion_prior_steps: int = 50
    big_vae_diffusion_prior_sampler: str = "ddim"
    big_vae_diffusion_prior_eta: float = 0.0
    big_vae_latent_noise_std: float = 0.0
    big_vae_encoder_context_rows: int = 64
    big_vae_encoder_context_std: float = 1.0
    big_vae_encoder_batch_size: int = 16
    big_vae_decode: str = "weights"
    big_vae_tile_T_patches: int = 16
    big_vae_tile_d_out: int = 8
    big_vae_init_fit_steps: int = 0
    big_vae_init_fit_lr: float = 1e-2
    big_vae_init_fit_log_every: int = 10


@dataclass(slots=True)
class EvalMetrics:
    loss: float
    accuracy: float
    examples: int


class DirectTensor(nn.Module):
    def __init__(self, initial: torch.Tensor) -> None:
        super().__init__()
        self.value = nn.Parameter(initial.detach().clone())

    def forward(self) -> torch.Tensor:
        return self.value


class LowRankDecodedTensor(nn.Module):
    """
    Frozen base tensor plus trainable latent factors.

    Matrix-like tensors are decoded as base + scale / sqrt(rank) * (left @ right).
    Vector tensors use an identity decoder base + scale * delta. This keeps the
    experiment focused on optimizer geometry without introducing a pretrained
    decoder dependency.
    """

    def __init__(
        self,
        initial: torch.Tensor,
        *,
        rank: int,
        delta_scale: float,
        factor_init_std: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"latent rank must be > 0, got {rank}")
        self.shape = tuple(int(dim) for dim in initial.shape)
        self.delta_scale = float(delta_scale)
        self.register_buffer("base", initial.detach().clone())

        if initial.ndim >= 2:
            out_dim = int(initial.shape[0])
            in_dim = int(initial.numel() // max(1, out_dim))
            effective_rank = max(1, min(int(rank), out_dim, in_dim))
            self.effective_rank = int(effective_rank)
            self.left = nn.Parameter(torch.empty(out_dim, effective_rank))
            self.right = nn.Parameter(torch.zeros(effective_rank, in_dim))
            nn.init.normal_(self.left, mean=0.0, std=float(factor_init_std))
            self.delta = None
        else:
            self.effective_rank = 0
            self.left = None
            self.right = None
            self.delta = nn.Parameter(torch.zeros_like(initial))

    def forward(self) -> torch.Tensor:
        if self.left is not None and self.right is not None:
            scale = self.delta_scale / math.sqrt(float(self.effective_rank))
            return self.base + scale * (self.left @ self.right).view(self.shape)
        if self.delta is None:
            return self.base
        return self.base + self.delta_scale * self.delta


@dataclass(slots=True)
class TensorMatrixSpec:
    original_shape: tuple[int, ...]
    matrix_shape: tuple[int, int]
    transposed: bool


@dataclass(slots=True)
class BigVAETileSegment:
    tensor_name: str
    tensor_key: str
    tile_row_start: int
    row_start: int
    row_len: int
    col_start: int
    col_len: int


@dataclass(slots=True)
class BigVAEDecodeTileSpec:
    d_in: int
    d_out: int
    T: int
    segments: list[BigVAETileSegment]


def make_tensor_matrix_spec(tensor: torch.Tensor, *, output_first_dim: bool = False) -> TensorMatrixSpec:
    shape = tuple(int(dim) for dim in tensor.shape)
    if tensor.ndim == 0:
        return TensorMatrixSpec(original_shape=shape, matrix_shape=(1, 1), transposed=False)
    if tensor.ndim == 1:
        return TensorMatrixSpec(original_shape=shape, matrix_shape=(int(tensor.numel()), 1), transposed=False)

    rows = int(tensor.shape[0])
    cols = int(tensor.numel() // max(1, rows))
    # Stage-1 records store module weights as [d_in, d_out]. Raw PyTorch module
    # weights are [out, *in], so force that orientation for decoded weights.
    transposed = bool(output_first_dim) or rows < cols
    matrix_shape = (cols, rows) if transposed else (rows, cols)
    return TensorMatrixSpec(original_shape=shape, matrix_shape=matrix_shape, transposed=transposed)


def tensor_to_matrix(tensor: torch.Tensor, spec: TensorMatrixSpec) -> torch.Tensor:
    if len(spec.original_shape) == 0:
        return tensor.reshape(1, 1)
    if len(spec.original_shape) == 1:
        return tensor.reshape(spec.matrix_shape)
    matrix = tensor.reshape(int(spec.original_shape[0]), -1)
    if spec.transposed:
        matrix = matrix.transpose(0, 1)
    return matrix.contiguous()


def matrix_to_tensor(matrix: torch.Tensor, spec: TensorMatrixSpec) -> torch.Tensor:
    if len(spec.original_shape) == 0:
        return matrix.reshape(())
    if len(spec.original_shape) == 1:
        return matrix.reshape(spec.original_shape)
    restored = matrix
    if spec.transposed:
        restored = restored.transpose(0, 1)
    return restored.contiguous().reshape(spec.original_shape)


class TensorStore(nn.Module):
    def __init__(
        self,
        initial_tensors: dict[str, torch.Tensor],
        *,
        mode: str,
        latent_rank: int,
        latent_delta_scale: float,
        latent_factor_init_std: float,
    ) -> None:
        super().__init__()
        if mode not in {"direct", "latent"}:
            raise ValueError(f"mode must be 'direct' or 'latent', got {mode!r}")
        self.mode = str(mode)
        self._name_to_key: dict[str, str] = {}
        modules: dict[str, nn.Module] = {}
        for idx, (name, initial) in enumerate(initial_tensors.items()):
            key = f"p{idx:04d}"
            self._name_to_key[name] = key
            if mode == "direct":
                modules[key] = DirectTensor(initial)
            else:
                modules[key] = LowRankDecodedTensor(
                    initial,
                    rank=latent_rank,
                    delta_scale=latent_delta_scale,
                    factor_init_std=latent_factor_init_std,
                )
        self.tensors = nn.ModuleDict(modules)

    def tensor(self, name: str) -> torch.Tensor:
        return self.tensors[self._name_to_key[name]]()

    def decoded_numel(self) -> int:
        total = 0
        for module in self.tensors.values():
            if isinstance(module, DirectTensor):
                total += int(module.value.numel())
            elif isinstance(module, LowRankDecodedTensor):
                total += int(module.base.numel())
        return int(total)


class BigVAELatentTensorStore(nn.Module):
    def __init__(
        self,
        initial_tensors: dict[str, torch.Tensor],
        *,
        big_vae: BigWeightVAE,
        latent_init: str,
        latent_noise_std: float,
        decode_policy: str,
        tile_T_patches: int,
        tile_d_out: int,
        latent_diffusion_prior: Any | None = None,
        latent_diffusion_prior_steps: int = 50,
        latent_diffusion_prior_sampler: str = "ddim",
        latent_diffusion_prior_eta: float = 0.0,
        encoder_context_rows: int = 64,
        encoder_context_std: float = 1.0,
        encoder_batch_size: int = 16,
    ) -> None:
        super().__init__()
        init_mode = str(latent_init).strip().lower()
        if init_mode not in {"base", "random", "encoded", "diffusion_prior"}:
            raise ValueError(
                "big_vae_latent_init must be 'base', 'random', 'encoded' or 'diffusion_prior', "
                f"got {latent_init!r}"
            )
        policy = str(decode_policy).strip().lower()
        if policy not in {"weights", "all"}:
            raise ValueError(f"big_vae_decode must be 'weights' or 'all', got {decode_policy!r}")
        if int(tile_T_patches) <= 0:
            raise ValueError(f"big_vae_tile_T_patches must be > 0, got {tile_T_patches}")
        if int(tile_d_out) <= 0:
            raise ValueError(f"big_vae_tile_d_out must be > 0, got {tile_d_out}")

        self.big_vae = big_vae
        self.big_vae.eval()
        for param in self.big_vae.parameters():
            param.requires_grad_(False)

        self._name_to_key: dict[str, str] = {}
        self._key_to_name: dict[str, str] = {}
        self._specs: dict[str, TensorMatrixSpec] = {}
        self._tile_specs: dict[str, BigVAEDecodeTileSpec] = {}
        self._tensor_key_to_tile_keys: dict[str, list[str]] = {}
        self._groups: dict[tuple[int, int, int], list[str]] = {}
        self._direct_name_to_key: dict[str, str] = {}
        self._tile_cond_patch: dict[str, torch.Tensor] = {}
        self.latent_slots = nn.ParameterDict()
        self.direct_tensors = nn.ModuleDict()
        self.latent_init_mode = init_mode
        self.latent_space = "decoder_z" if init_mode == "diffusion_prior" else "encoder_slots"
        self.patch_size = int(self.big_vae.cfg.patch_size)
        self.tile_T_patches = int(tile_T_patches)
        self.tile_d_in = int(self.patch_size) * int(self.tile_T_patches)
        self.tile_d_out = int(tile_d_out)
        self.use_distribution_encoder = bool(getattr(self.big_vae, "use_distribution_encoder", False))
        self.d_dist = int(self.big_vae.cfg.distribution.d_dist)
        self.encoder_context_rows = max(1, int(encoder_context_rows))
        self.encoder_context_std = float(encoder_context_std)
        self.encoder_batch_size = max(1, int(encoder_batch_size))
        self.latent_noise_std = float(latent_noise_std)
        self.latent_diffusion_prior = latent_diffusion_prior
        self.latent_diffusion_prior_steps = max(1, int(latent_diffusion_prior_steps))
        self.latent_diffusion_prior_sampler = str(latent_diffusion_prior_sampler).strip().lower()
        self.latent_diffusion_prior_eta = float(latent_diffusion_prior_eta)
        if self.latent_diffusion_prior is not None:
            self.latent_diffusion_prior.to(device=self.big_vae.latent_base.device)
            self.latent_diffusion_prior.eval()
            for param in self.latent_diffusion_prior.parameters():
                param.requires_grad_(False)
        if self.latent_init_mode == "diffusion_prior":
            if self.latent_diffusion_prior is None:
                raise ValueError(
                    "big_vae_latent_init='diffusion_prior' requires latent_diffusion_prior checkpoint/model"
                )
            if not self.use_distribution_encoder:
                raise ValueError(
                    "big_vae_latent_init='diffusion_prior' requires BigVAE distribution encoder to be enabled"
                )

        base_latents = self.big_vae.latent_base.detach().clone()
        for idx, (name, initial) in enumerate(initial_tensors.items()):
            tensor_key = f"p{idx:04d}"
            if not self._should_decode_with_big_vae(name=name, tensor=initial, policy=policy):
                self._direct_name_to_key[name] = tensor_key
                self.direct_tensors[tensor_key] = DirectTensor(initial)
                continue

            spec = make_tensor_matrix_spec(
                initial,
                output_first_dim=bool(initial.ndim >= 2 and str(name).endswith(".weight")),
            )
            rows, cols = spec.matrix_shape
            self._name_to_key[name] = tensor_key
            self._key_to_name[tensor_key] = name
            self._specs[tensor_key] = spec
            self._tensor_key_to_tile_keys[tensor_key] = []

            pending_segments: list[BigVAETileSegment] = []
            tile_index = 0

            def flush_tile() -> None:
                nonlocal pending_segments, tile_index
                if not pending_segments:
                    return
                tile_key = f"{tensor_key}_t{tile_index:04d}"
                tile_index += 1
                self._tile_specs[tile_key] = BigVAEDecodeTileSpec(
                    d_in=int(self.tile_d_in),
                    d_out=int(self.tile_d_out),
                    T=int(self.tile_T_patches),
                    segments=list(pending_segments),
                )
                self._tensor_key_to_tile_keys[tensor_key].append(tile_key)
                group_key = (int(self.tile_d_in), int(self.tile_d_out), int(self.tile_T_patches))
                self._groups.setdefault(group_key, []).append(tile_key)

                if init_mode in {"base", "encoded"}:
                    latent = base_latents.clone()
                else:
                    latent = torch.randn_like(base_latents) * 0.02
                if float(latent_noise_std) > 0.0:
                    latent = latent + torch.randn_like(latent) * float(latent_noise_std)
                self.latent_slots[tile_key] = nn.Parameter(latent)
                pending_segments = []

            current_tile_rows = 0
            for col_start in range(0, int(cols), int(self.tile_d_out)):
                col_len = min(int(self.tile_d_out), int(cols) - int(col_start))
                for row_start in range(0, int(rows), int(self.patch_size)):
                    row_len = min(int(self.patch_size), int(rows) - int(row_start))
                    if current_tile_rows > 0 and current_tile_rows + int(row_len) > int(self.tile_d_in):
                        flush_tile()
                        current_tile_rows = 0
                    pending_segments.append(
                        BigVAETileSegment(
                            tensor_name=name,
                            tensor_key=tensor_key,
                            tile_row_start=int(current_tile_rows),
                            row_start=int(row_start),
                            row_len=int(row_len),
                            col_start=int(col_start),
                            col_len=int(col_len),
                        )
                    )
                    current_tile_rows += int(row_len)
                    if current_tile_rows >= int(self.tile_d_in):
                        flush_tile()
                        current_tile_rows = 0
            flush_tile()

        if init_mode == "encoded":
            self._initialize_latents_from_encoder(initial_tensors)
        elif init_mode == "diffusion_prior":
            self._initialize_latents_from_diffusion_prior(initial_tensors)

    @staticmethod
    def _should_decode_with_big_vae(*, name: str, tensor: torch.Tensor, policy: str) -> bool:
        if policy == "all":
            return True
        # BigVAE was trained on 2D layer weight matrices. Keep conv kernels,
        # pos_embed, bias and LayerNorm tensors direct by default.
        return tensor.ndim == 2 and str(name).endswith(".weight")

    def decode_group_count(self) -> int:
        return int(len(self._groups))

    def decoded_tile_count(self) -> int:
        return int(len(self._tile_specs))

    def decoded_tensor_names(self) -> list[str]:
        return list(self._name_to_key.keys())

    def tile_decode_shape(self) -> tuple[int, int, int]:
        return int(self.tile_d_in), int(self.tile_d_out), int(self.tile_T_patches)

    def latent_numel(self) -> int:
        return int(sum(param.numel() for param in self.latent_slots.values()))

    def decoded_numel(self) -> int:
        total = 0
        for spec in self._specs.values():
            numel = 1
            for dim in spec.original_shape:
                numel *= int(dim)
            total += int(numel)
        for module in self.direct_tensors.values():
            if isinstance(module, DirectTensor):
                total += int(module.value.numel())
        return int(total)

    def big_vae_decoded_numel(self) -> int:
        total = 0
        for spec in self._specs.values():
            numel = 1
            for dim in spec.original_shape:
                numel *= int(dim)
            total += int(numel)
        return int(total)

    def latent_init_diversity(self) -> dict[str, float]:
        if not self.latent_slots:
            return {"count": 0.0, "across_layer_std_mean": 0.0, "max_pair_delta": 0.0}
        stacked = torch.stack([param.detach().float().cpu() for param in self.latent_slots.values()], dim=0)
        if int(stacked.shape[0]) <= 1:
            return {"count": float(stacked.shape[0]), "across_layer_std_mean": 0.0, "max_pair_delta": 0.0}
        centered = stacked - stacked.mean(dim=0, keepdim=True)
        return {
            "count": float(stacked.shape[0]),
            "across_layer_std_mean": float(stacked.std(dim=0, unbiased=False).mean().item()),
            "max_pair_delta": float(centered.abs().max().item()),
        }

    def target_matrix(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        key = self._name_to_key[name]
        return tensor_to_matrix(tensor, self._specs[key])

    def _encoder_context(self, batch_size: int, d_in: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rows = torch.arange(self.encoder_context_rows, device=device, dtype=torch.float32).view(1, -1, 1)
        cols = torch.arange(int(d_in), device=device, dtype=torch.float32).view(1, 1, -1)
        batch_phase = torch.arange(int(batch_size), device=device, dtype=torch.float32).view(-1, 1, 1) * 0.173
        context = torch.sin((rows + 1.0) * (cols + 1.0) * 0.017 + batch_phase)
        return (context * float(self.encoder_context_std)).to(dtype=dtype)

    def _initialize_latents_from_encoder(self, initial_tensors: dict[str, torch.Tensor]) -> None:
        if not self.latent_slots:
            return
        first_latent = next(iter(self.latent_slots.values()))
        device = first_latent.device
        dtype = first_latent.dtype
        matrix_cache: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for (d_in, d_out, expected_T), keys in self._groups.items():
                for start in range(0, len(keys), self.encoder_batch_size):
                    batch_keys = keys[start : start + self.encoder_batch_size]
                    batch = int(len(batch_keys))
                    W = torch.zeros(batch, int(d_in), int(d_out), device=device, dtype=dtype)
                    d_in_mask = torch.zeros(batch, int(d_in), device=device, dtype=torch.bool)
                    d_out_mask = torch.zeros(batch, int(d_out), device=device, dtype=torch.bool)

                    for item_idx, key in enumerate(batch_keys):
                        tile = self._tile_specs[key]
                        for segment in tile.segments:
                            matrix = matrix_cache.get(segment.tensor_name)
                            if matrix is None:
                                source = initial_tensors[segment.tensor_name].to(device=device, dtype=dtype)
                                matrix = tensor_to_matrix(source, self._specs[segment.tensor_key])
                                matrix_cache[segment.tensor_name] = matrix

                            tile_row_start = int(segment.tile_row_start)
                            tile_row_end = tile_row_start + int(segment.row_len)
                            row_start = int(segment.row_start)
                            row_end = row_start + int(segment.row_len)
                            col_start = int(segment.col_start)
                            col_end = col_start + int(segment.col_len)
                            W[item_idx, tile_row_start:tile_row_end, : int(segment.col_len)] = matrix[
                                row_start:row_end,
                                col_start:col_end,
                            ]
                            d_in_mask[item_idx, tile_row_start:tile_row_end] = True
                            d_out_mask[item_idx, : int(segment.col_len)] = True

                    X = self._encoder_context(batch, int(d_in), device=device, dtype=dtype)
                    X = X * d_in_mask.to(dtype=dtype).unsqueeze(1)
                    x_mask = torch.ones(batch, self.encoder_context_rows, device=device, dtype=torch.bool)
                    (
                        T,
                        d_in_pad,
                        patch_mask,
                        _structural_patch_mask,
                        dist_var_by_patch,
                        dist_patch_by_patch,
                        dist_var_pooled,
                    ) = self.big_vae._encode_distribution_context(
                        X,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                    )
                    if int(T) != int(expected_T):
                        raise RuntimeError(
                            f"BigVAE encoder T mismatch for group {(d_in, d_out, expected_T)}: got T={T}"
                        )
                    latents = self.big_vae._encode_latent_slots(
                        W,
                        T=int(T),
                        d_in_pad=int(d_in_pad),
                        patch_mask=patch_mask,
                        d_out_mask=d_out_mask,
                        dist_var_by_patch=dist_var_by_patch,
                        dist_patch_by_patch=dist_patch_by_patch,
                        dist_var_pooled=dist_var_pooled,
                    )
                    for item_idx, key in enumerate(batch_keys):
                        encoded = latents[item_idx].detach().to(device=device, dtype=dtype)
                        if dist_patch_by_patch is not None:
                            cond_patch = dist_patch_by_patch[item_idx].detach().to(device=device, dtype=dtype)
                            self._tile_cond_patch[key] = cond_patch
                        if self.latent_noise_std > 0.0:
                            encoded = encoded + torch.randn_like(encoded) * self.latent_noise_std
                        self.latent_slots[key].data.copy_(encoded)

    def _initialize_latents_from_diffusion_prior(self, initial_tensors: dict[str, torch.Tensor]) -> None:
        if not self.latent_slots:
            return
        if self.latent_diffusion_prior is None:
            raise RuntimeError("latent_diffusion_prior is required for diffusion_prior initialization")
        first_latent = next(iter(self.latent_slots.values()))
        device = first_latent.device
        dtype = first_latent.dtype
        matrix_cache: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for (d_in, d_out, expected_T), keys in self._groups.items():
                for start in range(0, len(keys), self.encoder_batch_size):
                    batch_keys = keys[start : start + self.encoder_batch_size]
                    batch = int(len(batch_keys))
                    W = torch.zeros(batch, int(d_in), int(d_out), device=device, dtype=dtype)
                    d_in_mask = torch.zeros(batch, int(d_in), device=device, dtype=torch.bool)
                    d_out_mask = torch.zeros(batch, int(d_out), device=device, dtype=torch.bool)

                    for item_idx, key in enumerate(batch_keys):
                        tile = self._tile_specs[key]
                        for segment in tile.segments:
                            matrix = matrix_cache.get(segment.tensor_name)
                            if matrix is None:
                                source = initial_tensors[segment.tensor_name].to(device=device, dtype=dtype)
                                matrix = tensor_to_matrix(source, self._specs[segment.tensor_key])
                                matrix_cache[segment.tensor_name] = matrix

                            tile_row_start = int(segment.tile_row_start)
                            tile_row_end = tile_row_start + int(segment.row_len)
                            row_start = int(segment.row_start)
                            row_end = row_start + int(segment.row_len)
                            col_start = int(segment.col_start)
                            col_end = col_start + int(segment.col_len)
                            W[item_idx, tile_row_start:tile_row_end, : int(segment.col_len)] = matrix[
                                row_start:row_end,
                                col_start:col_end,
                            ]
                            d_in_mask[item_idx, tile_row_start:tile_row_end] = True
                            d_out_mask[item_idx, : int(segment.col_len)] = True

                    X = self._encoder_context(batch, int(d_in), device=device, dtype=dtype)
                    X = X * d_in_mask.to(dtype=dtype).unsqueeze(1)
                    x_mask = torch.ones(batch, self.encoder_context_rows, device=device, dtype=torch.bool)
                    (
                        T,
                        _d_in_pad,
                        patch_mask,
                        _structural_patch_mask,
                        _dist_var_by_patch,
                        dist_patch_by_patch,
                        _dist_var_pooled,
                    ) = self.big_vae._encode_distribution_context(
                        X,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                    )
                    if int(T) != int(expected_T):
                        raise RuntimeError(
                            f"BigVAE encoder T mismatch for group {(d_in, d_out, expected_T)}: got T={T}"
                        )
                    if dist_patch_by_patch is None:
                        raise RuntimeError("distribution encoder must provide dist_patch_by_patch for diffusion prior")
                    sampled = self.latent_diffusion_prior.sample_latents(
                        cond_patch=dist_patch_by_patch,
                        patch_mask=patch_mask,
                        num_steps=self.latent_diffusion_prior_steps,
                        sampler_type=self.latent_diffusion_prior_sampler,
                        eta=self.latent_diffusion_prior_eta,
                    )
                    sampled = sampled.view(batch, *first_latent.shape)
                    for item_idx, key in enumerate(batch_keys):
                        sampled_item = sampled[item_idx].detach().to(device=device, dtype=dtype)
                        cond_patch = dist_patch_by_patch[item_idx].detach().to(device=device, dtype=dtype)
                        self._tile_cond_patch[key] = cond_patch
                        if self.latent_noise_std > 0.0:
                            sampled_item = sampled_item + torch.randn_like(sampled_item) * self.latent_noise_std
                        self.latent_slots[key].data.copy_(sampled_item)

    def decoded_matrix(self, name: str) -> torch.Tensor:
        return self.decode_all_matrices()[name]

    def decode_all_matrices(self) -> dict[str, torch.Tensor]:
        if not self.latent_slots:
            return {}

        first_latent = next(iter(self.latent_slots.values()))
        result: dict[str, torch.Tensor] = {}
        for tensor_key, spec in self._specs.items():
            name = self._key_to_name[tensor_key]
            result[name] = torch.zeros(
                spec.matrix_shape,
                device=first_latent.device,
                dtype=first_latent.dtype,
            )

        for (d_in, d_out, T), keys in self._groups.items():
            latents = torch.stack([self.latent_slots[key] for key in keys], dim=0)
            batch = int(latents.shape[0])
            d_in_pad = int(T) * self.patch_size
            device = latents.device
            patch_mask = torch.zeros(batch, int(T), device=device, dtype=torch.bool)
            d_in_mask = torch.zeros(batch, int(d_in), device=device, dtype=torch.bool)
            d_out_mask = torch.zeros(batch, int(d_out), device=device, dtype=torch.bool)
            for item_idx, key in enumerate(keys):
                tile = self._tile_specs[key]
                used_rows = 0
                used_cols = 0
                for segment in tile.segments:
                    segment_row_end = int(segment.tile_row_start) + int(segment.row_len)
                    d_in_mask[item_idx, int(segment.tile_row_start) : segment_row_end] = True
                    used_rows = max(used_rows, segment_row_end)
                    used_cols = max(used_cols, int(segment.col_len))
                valid_patches = int(math.ceil(float(used_rows) / float(self.patch_size)))
                patch_mask[item_idx, :valid_patches] = True
                d_out_mask[item_idx, :used_cols] = True
            dist_patch = None
            if self.use_distribution_encoder:
                if all(key in self._tile_cond_patch for key in keys):
                    dist_patch = torch.stack([self._tile_cond_patch[key].to(device=device, dtype=latents.dtype) for key in keys], dim=0)
                else:
                    dist_patch = torch.zeros(batch, int(T), self.d_dist, device=device, dtype=latents.dtype)
            if self.latent_space == "decoder_z":
                decoded = self.big_vae._decode_from_decoder_latent(
                    latents,
                    dist_patch_by_patch=dist_patch,
                    patch_mask=patch_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    d_in=int(d_in),
                    d_out=int(d_out),
                    d_in_pad=d_in_pad,
                    T=int(T),
                )[0]
            else:
                decoded = self.big_vae._decode_from_latent_slots(
                    latents,
                    dist_patch_by_patch=dist_patch,
                    patch_mask=patch_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    d_in=int(d_in),
                    d_out=int(d_out),
                    d_in_pad=d_in_pad,
                    T=int(T),
                )[0]
            for item_idx, key in enumerate(keys):
                tile = self._tile_specs[key]
                for segment in tile.segments:
                    tile_row_start = int(segment.tile_row_start)
                    tile_row_end = tile_row_start + int(segment.row_len)
                    result[segment.tensor_name][
                        int(segment.row_start) : int(segment.row_start) + int(segment.row_len),
                        int(segment.col_start) : int(segment.col_start) + int(segment.col_len),
                    ] = decoded[item_idx, tile_row_start:tile_row_end, : int(segment.col_len)]
        return result

    def decode_all_tensors(self) -> dict[str, torch.Tensor]:
        return {
            name: matrix_to_tensor(matrix, self._specs[self._name_to_key[name]])
            for name, matrix in self.decode_all_matrices().items()
        }

    def tensor(self, name: str) -> torch.Tensor:
        direct_key = self._direct_name_to_key.get(name)
        if direct_key is not None:
            return self.direct_tensors[direct_key]()
        key = self._name_to_key[name]
        return matrix_to_tensor(self.decoded_matrix(name), self._specs[key])


class FunctionalViTTiny(nn.Module):
    def __init__(
        self,
        vit_cfg: ViTTinyConfig,
        initial_tensors: dict[str, torch.Tensor],
        *,
        parameter_mode: str,
        latent_rank: int = 8,
        latent_delta_scale: float = 1.0,
        latent_factor_init_std: float = 0.02,
        big_vae: BigWeightVAE | None = None,
        big_vae_latent_init: str = "random",
        big_vae_diffusion_prior: Any | None = None,
        big_vae_diffusion_prior_steps: int = 50,
        big_vae_diffusion_prior_sampler: str = "ddim",
        big_vae_diffusion_prior_eta: float = 0.0,
        big_vae_latent_noise_std: float = 0.0,
        big_vae_encoder_context_rows: int = 64,
        big_vae_encoder_context_std: float = 1.0,
        big_vae_encoder_batch_size: int = 16,
        big_vae_decode: str = "weights",
        big_vae_tile_T_patches: int = 16,
        big_vae_tile_d_out: int = 8,
    ) -> None:
        super().__init__()
        self.cfg = vit_cfg
        if parameter_mode == "bigvae_latent":
            if big_vae is None:
                raise ValueError("big_vae is required when parameter_mode='bigvae_latent'")
            self.store = BigVAELatentTensorStore(
                initial_tensors,
                big_vae=big_vae,
                latent_init=big_vae_latent_init,
                latent_noise_std=big_vae_latent_noise_std,
                decode_policy=big_vae_decode,
                tile_T_patches=int(big_vae_tile_T_patches),
                tile_d_out=int(big_vae_tile_d_out),
                latent_diffusion_prior=big_vae_diffusion_prior,
                latent_diffusion_prior_steps=int(big_vae_diffusion_prior_steps),
                latent_diffusion_prior_sampler=str(big_vae_diffusion_prior_sampler),
                latent_diffusion_prior_eta=float(big_vae_diffusion_prior_eta),
                encoder_context_rows=int(big_vae_encoder_context_rows),
                encoder_context_std=float(big_vae_encoder_context_std),
                encoder_batch_size=int(big_vae_encoder_batch_size),
            )
        else:
            store_mode = "latent" if parameter_mode == "lowrank_latent" else parameter_mode
            self.store = TensorStore(
                initial_tensors,
                mode=store_mode,
                latent_rank=latent_rank,
                latent_delta_scale=latent_delta_scale,
                latent_factor_init_std=latent_factor_init_std,
            )
        self._decoded_tensor_cache: dict[str, torch.Tensor] | None = None

    def w(self, name: str) -> torch.Tensor:
        if self._decoded_tensor_cache is not None and name in self._decoded_tensor_cache:
            return self._decoded_tensor_cache[name]
        return self.store.tensor(name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if isinstance(self.store, BigVAELatentTensorStore):
            self._decoded_tensor_cache = self.store.decode_all_tensors()
        else:
            self._decoded_tensor_cache = None
        try:
            return self._forward_with_cached_weights(images)
        finally:
            self._decoded_tensor_cache = None

    def _forward_with_cached_weights(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = F.conv2d(
            images,
            self.w("patch_embed.weight"),
            self.w("patch_embed.bias"),
            stride=cfg.patch_size,
        )
        x = x.flatten(2).transpose(1, 2).contiguous()
        batch = int(x.shape[0])
        cls = self.w("cls_token").expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.w("pos_embed")
        x = F.dropout(x, p=cfg.dropout, training=self.training)

        for layer_idx in range(cfg.depth):
            x = self._block(x, layer_idx)

        x = F.layer_norm(
            x,
            (cfg.hidden_dim,),
            weight=self.w("norm.weight"),
            bias=self.w("norm.bias"),
            eps=cfg.layer_norm_eps,
        )
        cls_out = x[:, 0]
        return F.linear(cls_out, self.w("head.weight"), self.w("head.bias"))

    def _block(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.cfg
        prefix = f"blocks.{layer_idx}"
        h = F.layer_norm(
            x,
            (cfg.hidden_dim,),
            weight=self.w(f"{prefix}.norm1.weight"),
            bias=self.w(f"{prefix}.norm1.bias"),
            eps=cfg.layer_norm_eps,
        )
        qkv = F.linear(h, self.w(f"{prefix}.attn.qkv.weight"), self.w(f"{prefix}.attn.qkv.bias"))
        batch, tokens, _ = qkv.shape
        head_dim = cfg.hidden_dim // cfg.num_heads
        qkv = qkv.view(batch, tokens, 3, cfg.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=cfg.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, tokens, cfg.hidden_dim)
        attn = F.linear(attn, self.w(f"{prefix}.attn.proj.weight"), self.w(f"{prefix}.attn.proj.bias"))
        attn = F.dropout(attn, p=cfg.dropout, training=self.training)
        x = x + attn

        h = F.layer_norm(
            x,
            (cfg.hidden_dim,),
            weight=self.w(f"{prefix}.norm2.weight"),
            bias=self.w(f"{prefix}.norm2.bias"),
            eps=cfg.layer_norm_eps,
        )
        h = F.linear(h, self.w(f"{prefix}.mlp.fc1.weight"), self.w(f"{prefix}.mlp.fc1.bias"))
        h = F.gelu(h)
        h = F.dropout(h, p=cfg.dropout, training=self.training)
        h = F.linear(h, self.w(f"{prefix}.mlp.fc2.weight"), self.w(f"{prefix}.mlp.fc2.bias"))
        h = F.dropout(h, p=cfg.dropout, training=self.training)
        return x + h


def _normal(shape: Iterable[int], *, generator: torch.Generator, std: float) -> torch.Tensor:
    return torch.empty(tuple(int(dim) for dim in shape), dtype=torch.float32).normal_(
        mean=0.0,
        std=float(std),
        generator=generator,
    )


def make_initial_tensors(cfg: ViTTinyConfig, *, seed: int) -> dict[str, torch.Tensor]:
    if cfg.image_size % cfg.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    mlp_dim = int(round(cfg.hidden_dim * cfg.mlp_ratio))
    tensors: dict[str, torch.Tensor] = {}

    tensors["cls_token"] = _normal((1, 1, cfg.hidden_dim), generator=generator, std=0.02)
    tensors["pos_embed"] = _normal((1, num_patches + 1, cfg.hidden_dim), generator=generator, std=0.02)
    tensors["patch_embed.weight"] = _normal(
        (cfg.hidden_dim, cfg.in_channels, cfg.patch_size, cfg.patch_size),
        generator=generator,
        std=0.02,
    )
    tensors["patch_embed.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)

    for layer_idx in range(cfg.depth):
        prefix = f"blocks.{layer_idx}"
        tensors[f"{prefix}.norm1.weight"] = torch.ones(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.norm1.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.attn.qkv.weight"] = _normal(
            (3 * cfg.hidden_dim, cfg.hidden_dim),
            generator=generator,
            std=0.02,
        )
        tensors[f"{prefix}.attn.qkv.bias"] = torch.zeros(3 * cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.attn.proj.weight"] = _normal(
            (cfg.hidden_dim, cfg.hidden_dim),
            generator=generator,
            std=0.02,
        )
        tensors[f"{prefix}.attn.proj.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.norm2.weight"] = torch.ones(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.norm2.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.mlp.fc1.weight"] = _normal((mlp_dim, cfg.hidden_dim), generator=generator, std=0.02)
        tensors[f"{prefix}.mlp.fc1.bias"] = torch.zeros(mlp_dim, dtype=torch.float32)
        tensors[f"{prefix}.mlp.fc2.weight"] = _normal((cfg.hidden_dim, mlp_dim), generator=generator, std=0.02)
        tensors[f"{prefix}.mlp.fc2.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)

    tensors["norm.weight"] = torch.ones(cfg.hidden_dim, dtype=torch.float32)
    tensors["norm.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
    tensors["head.weight"] = _normal((cfg.num_classes, cfg.hidden_dim), generator=generator, std=0.02)
    tensors["head.bias"] = torch.zeros(cfg.num_classes, dtype=torch.float32)
    return tensors


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def normalize_setup_name(setup: str) -> str:
    value = str(setup).strip().lower()
    return value


def load_frozen_big_vae_decoder(checkpoint_path: str, *, device: torch.device) -> BigWeightVAE:
    return load_frozen_big_vae_from_checkpoint(checkpoint_path, device=device)


def resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_cifar10_loaders(cfg: ExperimentConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - exercised only in missing optional dependency envs.
        raise RuntimeError("torchvision is required for CIFAR-10 pipeline") from exc

    data_root = Path(cfg.data_dir)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.CIFAR10(root=str(data_root), train=True, download=bool(cfg.download), transform=train_transform)
    test_set = datasets.CIFAR10(root=str(data_root), train=False, download=bool(cfg.download), transform=eval_transform)

    if int(cfg.train_subset) > 0:
        train_set = Subset(train_set, list(range(min(int(cfg.train_subset), len(train_set)))))
    if int(cfg.test_subset) > 0:
        test_set = Subset(test_set, list(range(min(int(cfg.test_subset), len(test_set)))))

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


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


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


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_optimizer(
    parameters: list[torch.nn.Parameter],
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    cfg: ExperimentConfig,
) -> torch.optim.Optimizer:
    name = str(optimizer_name).strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(lr),
            betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
            eps=float(cfg.adam_eps),
            weight_decay=float(weight_decay),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(lr),
            momentum=float(cfg.sgd_momentum),
            weight_decay=float(weight_decay),
            nesterov=bool(cfg.sgd_nesterov),
        )
    raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {optimizer_name!r}")


def fit_big_vae_latents_to_initial_weights(
    model: nn.Module,
    initial_tensors: dict[str, torch.Tensor],
    *,
    steps: int,
    lr: float,
    log_every: int,
    setup: str,
) -> None:
    target = getattr(model, "_orig_mod", model)
    store = getattr(target, "store", None)
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("BigVAE latent fitting requires FunctionalViTTiny.store to be BigVAELatentTensorStore")

    optimizer = torch.optim.AdamW(store.latent_slots.parameters(), lr=float(lr), weight_decay=0.0)
    target_matrices = {
        name: store.target_matrix(name, initial_tensors[name]).to(next(store.latent_slots.parameters()).device)
        for name in store.decoded_tensor_names()
    }
    total_numel = sum(matrix.numel() for matrix in target_matrices.values())
    for step_idx in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = None
        decoded_matrices = store.decode_all_matrices()
        for name, target_matrix in target_matrices.items():
            decoded = decoded_matrices[name]
            term = F.mse_loss(decoded, target_matrix, reduction="sum")
            loss = term if loss is None else loss + term
        assert loss is not None
        loss = loss / max(1, int(total_numel))
        loss.backward()
        optimizer.step()
        if step_idx == 1 or step_idx % max(1, int(log_every)) == 0 or step_idx == int(steps):
            print(f"[{setup}] latent_init_fit step={step_idx}/{steps} mse={float(loss.detach().cpu().item()):.6e}", flush=True)


def train_setup(
    setup: str,
    cfg: ExperimentConfig,
    vit_cfg: ViTTinyConfig,
    initial_tensors: dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    output_dir: Path,
    big_vae_decoder: BigWeightVAE | None = None,
    big_vae_diffusion_prior: Any | None = None,
) -> dict[str, Any]:
    setup = normalize_setup_name(setup)
    if setup not in {"direct", "lowrank_latent", "bigvae_latent"}:
        raise ValueError(f"unsupported setup: {setup}")
    if setup == "bigvae_latent" and big_vae_decoder is None:
        raise ValueError("--big-vae-checkpoint is required for setup=bigvae_latent")

    seed_everything(int(cfg.seed))
    model = FunctionalViTTiny(
        vit_cfg,
        initial_tensors,
        parameter_mode=setup,
        latent_rank=int(cfg.latent_rank),
        latent_delta_scale=float(cfg.latent_delta_scale),
        latent_factor_init_std=float(cfg.latent_factor_init_std),
        big_vae=big_vae_decoder,
        big_vae_latent_init=str(cfg.big_vae_latent_init),
        big_vae_diffusion_prior=big_vae_diffusion_prior,
        big_vae_diffusion_prior_steps=int(cfg.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(cfg.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(cfg.big_vae_diffusion_prior_eta),
        big_vae_latent_noise_std=float(cfg.big_vae_latent_noise_std),
        big_vae_encoder_context_rows=int(cfg.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(cfg.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(cfg.big_vae_encoder_batch_size),
        big_vae_decode=str(cfg.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.big_vae_tile_d_out),
    ).to(device)
    if bool(cfg.compile):
        model = torch.compile(model)

    if setup == "bigvae_latent" and int(cfg.big_vae_init_fit_steps) > 0:
        fit_big_vae_latents_to_initial_weights(
            model,
            initial_tensors,
            steps=int(cfg.big_vae_init_fit_steps),
            lr=float(cfg.big_vae_init_fit_lr),
            log_every=max(1, int(cfg.big_vae_init_fit_log_every)),
            setup=setup,
        )

    lr = float(cfg.direct_lr if setup == "direct" else cfg.latent_lr)
    weight_decay = float(cfg.direct_weight_decay if setup == "direct" else cfg.latent_weight_decay)
    optimizer_name = str(cfg.direct_optimizer if setup == "direct" else cfg.latent_optimizer).strip().lower()
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = build_optimizer(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        cfg=cfg,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and bool(cfg.amp)))
    metrics_path = output_dir / "metrics.csv"
    checkpoint_dir = output_dir / "checkpoints"

    trainable_params = count_trainable_parameters(model)
    decoded_params = int(getattr(model, "store", getattr(model, "_orig_mod", model).store).decoded_numel())
    store = getattr(model, "store", getattr(model, "_orig_mod", model).store)
    decode_groups = store.decode_group_count() if isinstance(store, BigVAELatentTensorStore) else 0
    big_vae_decoded_params = store.big_vae_decoded_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    big_vae_tile_count = store.decoded_tile_count() if isinstance(store, BigVAELatentTensorStore) else 0
    big_vae_latent_params = store.latent_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    print(
        f"[{setup}] trainable_params={trainable_params} decoded_params={decoded_params} "
        f"bigvae_decoded_params={big_vae_decoded_params} decode_groups={decode_groups} "
        f"optimizer={optimizer_name} lr={lr:g} weight_decay={weight_decay:g}",
        flush=True,
    )
    if isinstance(store, BigVAELatentTensorStore):
        diversity = store.latent_init_diversity()
        tile_d_in, tile_d_out, tile_T = store.tile_decode_shape()
        compression = (
            float(big_vae_decoded_params) / float(big_vae_latent_params)
            if int(big_vae_latent_params) > 0
            else 0.0
        )
        print(
            f"[{setup}] latent_init={store.latent_init_mode} latent_space={store.latent_space} "
            f"latent_tiles={int(diversity['count'])} "
            f"tile_decode_shape=({tile_d_in},{tile_d_out},T={tile_T}) "
            f"latent_params={big_vae_latent_params} decoded_per_latent={compression:.3g} "
            f"across_layer_std_mean={diversity['across_layer_std_mean']:.6g} "
            f"max_pair_delta={diversity['max_pair_delta']:.6g}",
            flush=True,
        )

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

            batch_examples = int(labels.numel())
            train_loss_window += float(loss.detach().cpu().item()) * batch_examples
            train_count_window += batch_examples

            should_log = global_step == 1 or global_step % max(1, int(cfg.log_every_steps)) == 0
            should_eval = global_step == 1 or global_step % max(1, int(cfg.eval_every_steps)) == 0
            max_steps = int(cfg.max_steps)
            if max_steps > 0 and global_step >= max_steps:
                should_eval = True

            if should_eval:
                final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.amp))
                if final_eval.accuracy > best_accuracy:
                    best_accuracy = float(final_eval.accuracy)
                    best_step = int(global_step)
            if should_log or should_eval:
                avg_train_loss = train_loss_window / max(1, train_count_window)
                elapsed_s = time.time() - start_time
                row = {
                    "setup": setup,
                    "step": int(global_step),
                    "epoch": int(epoch_idx + 1),
                    "train_loss": float(avg_train_loss),
                    "test_loss": float(final_eval.loss),
                    "test_accuracy": float(final_eval.accuracy),
                    "best_accuracy": float(best_accuracy),
                    "lr": float(lr),
                    "optimizer": optimizer_name,
                    "trainable_params": int(trainable_params),
                    "decoded_params": int(decoded_params),
                    "elapsed_s": float(elapsed_s),
                }
                append_csv_row(metrics_path, row)
                print(
                    f"[{setup}] step={global_step} epoch={epoch_idx + 1} "
                    f"train_loss={avg_train_loss:.4f} test_loss={final_eval.loss:.4f} "
                    f"test_acc={final_eval.accuracy:.4f} best={best_accuracy:.4f}",
                    flush=True,
                )
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
        target = getattr(model, "_orig_mod", model)
        if setup == "bigvae_latent":
            model_state = {
                "latent_slots": target.store.latent_slots.state_dict(),
                "big_vae_checkpoint": str(cfg.big_vae_checkpoint),
            }
        else:
            model_state = target.state_dict()
        torch.save(
            {
                "setup": setup,
                "step": int(global_step),
                "model_state": model_state,
                "vit_config": asdict(vit_cfg),
                "experiment_config": asdict(cfg),
            },
            checkpoint_dir / f"{setup}_final.pt",
        )

    summary = {
        "setup": setup,
        "steps": int(global_step),
        "final_test_loss": float(final_eval.loss),
        "final_test_accuracy": float(final_eval.accuracy),
        "best_test_accuracy": float(best_accuracy),
        "best_step": int(best_step),
        "trainable_params": int(trainable_params),
        "decoded_params": int(decoded_params),
        "lr": float(lr),
        "optimizer": optimizer_name,
        "weight_decay": float(weight_decay),
        "big_vae_checkpoint": str(cfg.big_vae_checkpoint) if setup == "bigvae_latent" else "",
        "big_vae_init_fit_steps": int(cfg.big_vae_init_fit_steps) if setup == "bigvae_latent" else 0,
        "big_vae_latent_init": str(cfg.big_vae_latent_init) if setup == "bigvae_latent" else "",
        "big_vae_diffusion_prior_checkpoint": (
            str(cfg.big_vae_diffusion_prior_checkpoint) if setup == "bigvae_latent" else ""
        ),
        "big_vae_diffusion_prior_steps": int(cfg.big_vae_diffusion_prior_steps) if setup == "bigvae_latent" else 0,
        "big_vae_diffusion_prior_sampler": (
            str(cfg.big_vae_diffusion_prior_sampler) if setup == "bigvae_latent" else ""
        ),
        "big_vae_diffusion_prior_eta": float(cfg.big_vae_diffusion_prior_eta) if setup == "bigvae_latent" else 0.0,
        "big_vae_encoder_context_rows": int(cfg.big_vae_encoder_context_rows) if setup == "bigvae_latent" else 0,
        "big_vae_encoder_context_std": float(cfg.big_vae_encoder_context_std) if setup == "bigvae_latent" else 0.0,
        "decode_groups": int(decode_groups),
        "big_vae_decoded_params": int(big_vae_decoded_params),
        "big_vae_tile_count": int(big_vae_tile_count),
        "big_vae_latent_params": int(big_vae_latent_params),
    }
    write_json(output_dir / f"{setup}_summary.json", summary)
    return summary


def maybe_write_plot(output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows: list[dict[str, str]] = []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        return

    by_setup: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_setup.setdefault(row["setup"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for setup, setup_rows in sorted(by_setup.items()):
        steps = [int(row["step"]) for row in setup_rows]
        losses = [float(row["test_loss"]) for row in setup_rows]
        accs = [float(row["test_accuracy"]) for row in setup_rows]
        axes[0].plot(steps, losses, label=setup)
        axes[1].plot(steps, accs, label=setup)
    axes[0].set_title("CIFAR-10 test loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[1].set_title("CIFAR-10 test accuracy")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)


def parse_args() -> tuple[ExperimentConfig, ViTTinyConfig]:
    default_exp = ExperimentConfig()
    default_vit = ViTTinyConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Compare CIFAR-10 ViT-Tiny optimization with direct weights vs "
            "latent factors decoded into weights."
        )
    )
    parser.add_argument("--output-dir", default=default_exp.output_dir)
    parser.add_argument("--data-dir", default=default_exp.data_dir)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=default_exp.download)
    parser.add_argument(
        "--setup",
        choices=("direct", "bigvae_latent", "lowrank_latent", "both"),
        default=default_exp.setup,
    )
    parser.add_argument("--device", default=default_exp.device)
    parser.add_argument("--seed", type=int, default=default_exp.seed)
    parser.add_argument("--epochs", type=int, default=default_exp.epochs)
    parser.add_argument("--max-steps", type=int, default=default_exp.max_steps)
    parser.add_argument("--batch-size", type=int, default=default_exp.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=default_exp.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=default_exp.num_workers)
    parser.add_argument("--train-subset", type=int, default=default_exp.train_subset)
    parser.add_argument("--test-subset", type=int, default=default_exp.test_subset)
    parser.add_argument("--direct-lr", type=float, default=default_exp.direct_lr)
    parser.add_argument("--latent-lr", type=float, default=default_exp.latent_lr)
    parser.add_argument("--direct-optimizer", choices=("adamw", "sgd"), default=default_exp.direct_optimizer)
    parser.add_argument("--latent-optimizer", choices=("adamw", "sgd"), default=default_exp.latent_optimizer)
    parser.add_argument("--direct-weight-decay", type=float, default=default_exp.direct_weight_decay)
    parser.add_argument("--latent-weight-decay", type=float, default=default_exp.latent_weight_decay)
    parser.add_argument("--sgd-momentum", type=float, default=default_exp.sgd_momentum)
    parser.add_argument("--sgd-nesterov", action=argparse.BooleanOptionalAction, default=default_exp.sgd_nesterov)
    parser.add_argument("--adam-beta1", type=float, default=default_exp.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=default_exp.adam_beta2)
    parser.add_argument("--adam-eps", type=float, default=default_exp.adam_eps)
    parser.add_argument("--label-smoothing", type=float, default=default_exp.label_smoothing)
    parser.add_argument("--grad-clip-norm", type=float, default=default_exp.grad_clip_norm)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=default_exp.amp)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=default_exp.tf32)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=default_exp.compile)
    parser.add_argument("--log-every-steps", type=int, default=default_exp.log_every_steps)
    parser.add_argument("--eval-every-steps", type=int, default=default_exp.eval_every_steps)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=default_exp.save_checkpoints)
    parser.add_argument("--latent-rank", type=int, default=default_exp.latent_rank)
    parser.add_argument("--latent-delta-scale", type=float, default=default_exp.latent_delta_scale)
    parser.add_argument("--latent-factor-init-std", type=float, default=default_exp.latent_factor_init_std)
    parser.add_argument("--big-vae-checkpoint", default=default_exp.big_vae_checkpoint)
    parser.add_argument(
        "--big-vae-latent-init",
        choices=("base", "random", "encoded", "diffusion_prior"),
        default=default_exp.big_vae_latent_init,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-checkpoint",
        default=default_exp.big_vae_diffusion_prior_checkpoint,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-steps",
        type=int,
        default=default_exp.big_vae_diffusion_prior_steps,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-sampler",
        choices=("ddim", "ddpm"),
        default=default_exp.big_vae_diffusion_prior_sampler,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-eta",
        type=float,
        default=default_exp.big_vae_diffusion_prior_eta,
    )
    parser.add_argument("--big-vae-latent-noise-std", type=float, default=default_exp.big_vae_latent_noise_std)
    parser.add_argument(
        "--big-vae-encoder-context-rows",
        type=int,
        default=default_exp.big_vae_encoder_context_rows,
    )
    parser.add_argument(
        "--big-vae-encoder-context-std",
        type=float,
        default=default_exp.big_vae_encoder_context_std,
    )
    parser.add_argument(
        "--big-vae-encoder-batch-size",
        type=int,
        default=default_exp.big_vae_encoder_batch_size,
    )
    parser.add_argument("--big-vae-decode", choices=("weights", "all"), default=default_exp.big_vae_decode)
    parser.add_argument(
        "--big-vae-tile-t-patches",
        "--big-vae-tile-T-patches",
        dest="big_vae_tile_T_patches",
        type=int,
        default=default_exp.big_vae_tile_T_patches,
    )
    parser.add_argument("--big-vae-tile-d-out", type=int, default=default_exp.big_vae_tile_d_out)
    parser.add_argument("--big-vae-init-fit-steps", type=int, default=default_exp.big_vae_init_fit_steps)
    parser.add_argument("--big-vae-init-fit-lr", type=float, default=default_exp.big_vae_init_fit_lr)
    parser.add_argument("--big-vae-init-fit-log-every", type=int, default=default_exp.big_vae_init_fit_log_every)

    parser.add_argument("--patch-size", type=int, default=default_vit.patch_size)
    parser.add_argument("--hidden-dim", type=int, default=default_vit.hidden_dim)
    parser.add_argument("--depth", type=int, default=default_vit.depth)
    parser.add_argument("--num-heads", type=int, default=default_vit.num_heads)
    parser.add_argument("--mlp-ratio", type=float, default=default_vit.mlp_ratio)
    parser.add_argument("--dropout", type=float, default=default_vit.dropout)
    parser.add_argument("--attention-dropout", type=float, default=default_vit.attention_dropout)
    args = parser.parse_args()

    exp_cfg = ExperimentConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        download=bool(args.download),
        setup=str(args.setup),
        device=str(args.device),
        seed=int(args.seed),
        epochs=int(args.epochs),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        train_subset=int(args.train_subset),
        test_subset=int(args.test_subset),
        direct_lr=float(args.direct_lr),
        latent_lr=float(args.latent_lr),
        direct_optimizer=str(args.direct_optimizer),
        latent_optimizer=str(args.latent_optimizer),
        direct_weight_decay=float(args.direct_weight_decay),
        latent_weight_decay=float(args.latent_weight_decay),
        sgd_momentum=float(args.sgd_momentum),
        sgd_nesterov=bool(args.sgd_nesterov),
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
        latent_rank=int(args.latent_rank),
        latent_delta_scale=float(args.latent_delta_scale),
        latent_factor_init_std=float(args.latent_factor_init_std),
        big_vae_checkpoint=str(args.big_vae_checkpoint),
        big_vae_latent_init=str(args.big_vae_latent_init),
        big_vae_diffusion_prior_checkpoint=str(args.big_vae_diffusion_prior_checkpoint),
        big_vae_diffusion_prior_steps=int(args.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(args.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(args.big_vae_diffusion_prior_eta),
        big_vae_latent_noise_std=float(args.big_vae_latent_noise_std),
        big_vae_encoder_context_rows=int(args.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(args.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(args.big_vae_encoder_batch_size),
        big_vae_decode=str(args.big_vae_decode),
        big_vae_tile_T_patches=int(args.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(args.big_vae_tile_d_out),
        big_vae_init_fit_steps=int(args.big_vae_init_fit_steps),
        big_vae_init_fit_lr=float(args.big_vae_init_fit_lr),
        big_vae_init_fit_log_every=int(args.big_vae_init_fit_log_every),
    )
    vit_cfg = ViTTinyConfig(
        patch_size=int(args.patch_size),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
    )
    return exp_cfg, vit_cfg


def main() -> None:
    cfg, vit_cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"experiment": asdict(cfg), "vit": asdict(vit_cfg)})

    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.benchmark = True

    seed_everything(int(cfg.seed))
    initial_tensors = make_initial_tensors(vit_cfg, seed=int(cfg.seed))
    train_loader, test_loader = build_cifar10_loaders(cfg, device)

    requested_setup = normalize_setup_name(cfg.setup)
    setups = ["direct", "bigvae_latent"] if requested_setup == "both" else [requested_setup]
    big_vae_decoder = None
    big_vae_diffusion_prior = None
    if "bigvae_latent" in setups:
        if not str(cfg.big_vae_checkpoint).strip():
            raise ValueError("--big-vae-checkpoint is required for setup=bigvae_latent/both")
        print(f"[bigvae_latent] loading frozen BigVAE decoder: {cfg.big_vae_checkpoint}", flush=True)
        big_vae_decoder = load_frozen_big_vae_decoder(cfg.big_vae_checkpoint, device=device)
        if str(cfg.big_vae_latent_init).strip().lower() == "diffusion_prior":
            if not str(cfg.big_vae_diffusion_prior_checkpoint).strip():
                raise ValueError(
                    "--big-vae-diffusion-prior-checkpoint is required when --big-vae-latent-init=diffusion_prior"
                )
            print(
                "[bigvae_latent] loading frozen latent diffusion prior: "
                f"{cfg.big_vae_diffusion_prior_checkpoint}",
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
                        "[bigvae_latent] loaded finetuned distribution encoder state from latent diffusion prior checkpoint",
                        flush=True,
                    )

    summaries = []
    for setup in setups:
        summaries.append(
            train_setup(
                setup,
                cfg,
                vit_cfg,
                initial_tensors,
                train_loader,
                test_loader,
                device=device,
                output_dir=output_dir,
                big_vae_decoder=big_vae_decoder,
                big_vae_diffusion_prior=big_vae_diffusion_prior,
            )
        )
    write_json(output_dir / "summary.json", {"summaries": summaries})
    maybe_write_plot(output_dir)
    maybe_write_plot(output_dir)


if __name__ == "__main__":
    main()
