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

from big_vae.models import BigWeightVAE
from training.big_vae_latent_diffusion import (
    build_cond_global_from_dist_var_pooled,
    build_layer_metadata_condition_vector,
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
    big_vae_latent_parameterization: str = "euclidean"
    big_vae_diffusion_prior_checkpoint: str = ""
    big_vae_diffusion_prior_steps: int = 50
    big_vae_diffusion_prior_sampler: str = "ddim"
    big_vae_diffusion_prior_eta: float = 0.0
    big_vae_random_init_std: float = 0.02
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


class FrozenBufferTensor(nn.Module):
    def __init__(self, initial: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", initial.detach().clone())

    def forward(self) -> torch.Tensor:
        return self.value

    @torch.no_grad()
    def copy_(self, value: torch.Tensor) -> None:
        self.value.copy_(value.detach().to(device=self.value.device, dtype=self.value.dtype))


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


