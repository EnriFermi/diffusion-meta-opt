from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import signal
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from big_vae.datasets.operator_bank import (
    BalancedOperatorBankMixer,
    OperatorBankSample,
    operator_bank_data_pipeline,
)
from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from big_vae.models.distribution_encoder import (
    DistributionConfig,
    InputDistributionEncodingModule,
)
from training.big_vae.data_types import (
    _flatten_loader_batches,
    _identity_sample_collate,
    _offline_loader_worker_init_fn,
    _sample_list_collate,
)
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    PreNormTransformerBlock,
    RMSNorm,
    _gradient_telemetry,
    _seed_everything,
)


SCHEMA = "weightclip_mini_polar_regression_production_v1"
EXPECTED_PARAMETERS = 9_938_689
PARENT_TILE_SIZE = 128
SUBTILES_PER_AXIS = 4
SUBTILES_PER_PARENT = SUBTILES_PER_AXIS**2
DEFAULT_CONFIG = Path(
    "conf/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_production.yaml"
)


@dataclass(frozen=True)
class MiniPolarConfig:
    tile_size: int = 32
    values_per_token: int = 16
    output_patch_size: int = 16
    hidden_dim: int = 192
    heads: int = 6
    mlp_dim: int = 704
    encoder_depth: int = 10
    shared_decoder_depth: int = 5
    direction_tail_depth: int = 1
    scale_tail_depth: int = 1
    latent_slots: int = 16
    latent_dim: int = 48
    max_tile_rows: int = 72
    max_tile_cols: int = 8
    use_distribution_conditioning: bool = True
    distribution_k_s: int = 16
    distribution_Kq: int = 32
    distribution_d_var: int = 64
    distribution_d_dist: int = 64
    distribution_num_var_attn_layers: int = 2
    distribution_var_attn_heads: int = 4
    distribution_dcn_num_cross_layers: int = 2
    distribution_dcn_deep_hidden: int = 32
    distribution_dcn_deep_layers: int = 2
    distribution_use_covariance: bool = True
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class SubtileCursor:
    parent_logical_index: int
    subpatch_index: int
    emitted_logical_index: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 9.94M p16/tile32 conditioned polar regression AE."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Create/reuse the declared tile32 normalization artifact, then exit.",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    tmp.replace(path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, got {config.get('schema')!r}")
    expected_loss = {
        "behavioral_coef": 1.0,
        "behavioral_operator": 0.0,
        "behavioral_direction": 1.0,
        "behavioral_scale": 10.0,
        "structural_coef": 1.0,
        "structural_direction": 1.0,
        "structural_scale": 10.0,
        "structural_reconstruction": 0.0,
        "structural_relational": 0.0,
    }
    if config.get("loss") != expected_loss:
        raise ValueError(
            "mini run must preserve the exact pre-categorical polar regression loss; "
            f"expected={expected_loss}, got={config.get('loss')}"
        )
    model_cfg = MiniPolarConfig(**config["architecture"])
    if model_cfg != MiniPolarConfig():
        raise ValueError(
            "the production mini architecture is frozen for the first baseline: "
            f"expected={asdict(MiniPolarConfig())}, got={asdict(model_cfg)}"
        )
    frozen = {
        "seed": 42,
        "steps": 500_000,
        "batch_size": 128,
        "learning_rate": 5.0e-5,
        "weight_decay": 0.01,
        "grad_clip_norm": 5.0,
        "scheduler": "constant",
    }
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise ValueError(f"production field {key!r} drifted: {config.get(key)!r} != {expected!r}")
    if int(config["normalization"]["calibration_valid_subtiles"]) < 1:
        raise ValueError("normalization calibration must contain at least one valid subtile")
    if int(config["operator_bank"]["loader_batch_size"]) < 1:
        raise ValueError("loader_batch_size must be positive")
    return config


class ValidSubtileStream(Iterator[OperatorBankSample]):
    """Expand parent p128 tiles into only their nonempty p32 subtiles.

    The cursor points to the next parent/subpatch candidate, not to DataLoader
    prefetch state, so a checkpoint resumes the exact consumed training stream.
    """

    def __init__(self, parent_stream: Iterator[OperatorBankSample], cursor: SubtileCursor) -> None:
        if cursor.parent_logical_index < 0:
            raise ValueError("parent cursor must be non-negative")
        if not 0 <= cursor.subpatch_index < SUBTILES_PER_PARENT:
            raise ValueError("subpatch cursor is outside [0,16)")
        if cursor.emitted_logical_index < 0:
            raise ValueError("emitted cursor must be non-negative")
        self.parent_stream = parent_stream
        self._next_parent_index = int(cursor.parent_logical_index)
        self._next_subpatch_index = int(cursor.subpatch_index)
        self._next_emitted_index = int(cursor.emitted_logical_index)
        self._parent: OperatorBankSample | None = None
        self.skipped_empty_subtiles = 0
        self.consumed_parent_tiles = 0

    def __iter__(self) -> ValidSubtileStream:
        return self

    def cursor(self) -> SubtileCursor:
        return SubtileCursor(
            parent_logical_index=self._next_parent_index,
            subpatch_index=self._next_subpatch_index,
            emitted_logical_index=self._next_emitted_index,
        )

    def _load_parent(self) -> None:
        parent = next(self.parent_stream)
        actual = int(parent.meta["logical_index"])
        if actual != self._next_parent_index:
            raise RuntimeError(
                f"parent stream cursor mismatch: expected={self._next_parent_index} actual={actual}"
            )
        self._parent = parent
        self.consumed_parent_tiles += 1

    @staticmethod
    def _subtile(parent: OperatorBankSample, subpatch_index: int, emitted_index: int) -> OperatorBankSample | None:
        sub_in, sub_out = divmod(subpatch_index, SUBTILES_PER_AXIS)
        row = slice(sub_in * 32, (sub_in + 1) * 32)
        col = slice(sub_out * 32, (sub_out + 1) * 32)
        d_in_mask = parent.meta["d_in_mask"][row].clone()
        d_out_mask = parent.meta["d_out_mask"][col].clone()
        if not bool(d_in_mask.any()) or not bool(d_out_mask.any()):
            return None
        meta = dict(parent.meta)
        parent_row = int(parent.meta["tile_row"])
        parent_col = int(parent.meta["tile_col"])
        parent_logical = int(parent.meta["logical_index"])
        meta.update(
            {
                "logical_index": int(emitted_index),
                "parent_logical_index": parent_logical,
                "parent_tile_row": parent_row,
                "parent_tile_col": parent_col,
                "subpatch_index": int(subpatch_index),
                "subtile_input_index": int(sub_in),
                "subtile_output_index": int(sub_out),
                "tile_row": parent_row * SUBTILES_PER_AXIS + sub_in,
                "tile_col": parent_col * SUBTILES_PER_AXIS + sub_out,
                "tile_row_start": int(parent.meta["tile_row_start"]) + sub_in * 32,
                "tile_col_start": int(parent.meta["tile_col_start"]) + sub_out * 32,
                "d_in_mask": d_in_mask,
                "d_out_mask": d_out_mask,
                "x_mask": parent.meta["x_mask"].clone(),
            }
        )
        return OperatorBankSample(
            x=parent.x[:, row].clone(),
            weight=parent.weight[row, col].clone(),
            meta=meta,
            model_name=parent.model_name,
            layer_name=f"{parent.layer_name}#p32-{sub_in}-{sub_out}",
        )

    def __next__(self) -> OperatorBankSample:
        while True:
            if self._parent is None:
                self._load_parent()
            assert self._parent is not None
            candidate_index = self._next_subpatch_index
            sample = self._subtile(
                self._parent,
                candidate_index,
                self._next_emitted_index,
            )
            self._next_subpatch_index += 1
            if self._next_subpatch_index == SUBTILES_PER_PARENT:
                self._next_parent_index += 1
                self._next_subpatch_index = 0
                self._parent = None
            if sample is None:
                self.skipped_empty_subtiles += 1
                continue
            self._next_emitted_index += 1
            return sample

    def telemetry(self) -> dict[str, Any]:
        return {
            "schema": "valid_p32_subtile_stream_v1",
            "cursor": self.cursor().to_dict(),
            "consumed_parent_tiles": self.consumed_parent_tiles,
            "skipped_empty_subtiles": self.skipped_empty_subtiles,
        }


def _batch_from_stream(stream: ValidSubtileStream, batch_size: int) -> dict[str, Any]:
    samples = [next(stream) for _ in range(batch_size)]
    logical_indices = [int(sample.meta["logical_index"]) for sample in samples]
    expected = list(range(logical_indices[0], logical_indices[0] + batch_size))
    if logical_indices != expected:
        raise RuntimeError("mini logical batch is not contiguous")
    distribution_sources: list[int] = []
    distribution_groups: list[int] = []
    group_by_context: dict[tuple[int, int], int] = {}
    for sample_index, sample in enumerate(samples):
        context_key = (
            int(sample.meta["parent_logical_index"]),
            int(sample.meta["subtile_input_index"]),
        )
        group_index = group_by_context.get(context_key)
        if group_index is None:
            group_index = len(distribution_sources)
            group_by_context[context_key] = group_index
            distribution_sources.append(sample_index)
        distribution_groups.append(group_index)
    return {
        "W": torch.stack([sample.weight for sample in samples]).contiguous(),
        "X": torch.stack([sample.x for sample in samples]).contiguous(),
        "x_mask": torch.stack([sample.meta["x_mask"] for sample in samples]).bool(),
        "d_in_mask": torch.stack([sample.meta["d_in_mask"] for sample in samples]).bool(),
        "d_out_mask": torch.stack([sample.meta["d_out_mask"] for sample in samples]).bool(),
        "tile_row": torch.tensor([sample.meta["tile_row"] for sample in samples]),
        "tile_col": torch.tensor([sample.meta["tile_col"] for sample in samples]),
        "logical_indices": logical_indices,
        "distribution_source_indices": torch.tensor(distribution_sources),
        "distribution_group_index": torch.tensor(distribution_groups),
    }


def _prepare_normalized_inputs(
    W: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    *,
    scale_mean: float,
    scale_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if W.ndim != 3 or tuple(W.shape[-2:]) != (32, 32):
        raise ValueError("mini production model requires W=[B,32,32]")
    batch = W.shape[0]
    rows = W.transpose(1, 2).float().contiguous()
    valid_values = d_out_mask[:, :, None] & d_in_mask[:, None, :]
    rows = rows * valid_values.to(dtype=rows.dtype)
    scale = rows.abs().amax(dim=-1).clamp_min(1.0e-8) / 7.0
    normalized = (rows / scale[:, :, None]) * valid_values.to(dtype=rows.dtype)
    content = normalized.view(batch, 32, 2, 16).reshape(batch, 64, 16)
    standardized = (torch.log2(scale) - float(scale_mean)) / float(scale_std)
    log_scale = standardized[:, :, None].expand(-1, -1, 2).reshape(batch, 64, 1)
    valid_chunks = d_in_mask.view(batch, 2, 16).any(dim=-1)
    token_valid = (d_out_mask[:, :, None] & valid_chunks[:, None, :]).reshape(batch, 64)
    return content, log_scale, token_valid


class MiniPolarWeightBottleneck(nn.Module):
    log_scale_min = -3.0
    log_scale_max = 6.0
    initial_log_radius = -2.0

    def __init__(self, cfg: MiniPolarConfig) -> None:
        super().__init__()
        if cfg.tile_size != 32 or cfg.values_per_token != 16:
            raise ValueError("mini baseline requires tile32 with p16 tokens")
        if cfg.tile_size % cfg.values_per_token:
            raise ValueError("values_per_token must divide tile_size")
        if cfg.output_patch_size != cfg.values_per_token:
            raise ValueError("mini baseline requires matching p16 input/output patches")
        if cfg.direction_tail_depth != 1 or cfg.scale_tail_depth != 1:
            raise ValueError("mini polar baseline requires one block in each tail")
        if not cfg.use_distribution_conditioning:
            raise ValueError("conditioned mini baseline requires the Distribution Encoder")
        self.cfg = cfg
        self.chunks_per_group = cfg.tile_size // cfg.values_per_token
        self.weight_token_count = cfg.tile_size * self.chunks_per_group
        dim = cfg.hidden_dim
        depth_scale = 1.0 / math.sqrt(
            2.0 * max(cfg.encoder_depth, cfg.shared_decoder_depth + 1)
        )

        def block(*, conditioned: bool) -> PreNormTransformerBlock:
            return PreNormTransformerBlock(
                dim,
                cfg.heads,
                cfg.mlp_dim,
                depth_scale,
                context_dim=(cfg.distribution_d_dist if conditioned else None),
            )

        self.latent_slots = nn.Parameter(torch.empty(cfg.latent_slots, dim))
        self.encoder_blocks = nn.ModuleList(
            [block(conditioned=True) for _ in range(cfg.encoder_depth)]
        )
        self.latent_norm = RMSNorm(dim)
        self.to_latent = nn.Linear(dim, cfg.latent_dim, bias=False)
        self.from_latent = nn.Linear(cfg.latent_dim, dim, bias=False)
        self.output_queries = nn.Parameter(torch.empty(self.weight_token_count, dim))
        self.decoder_blocks = nn.ModuleList(
            [block(conditioned=False) for _ in range(cfg.shared_decoder_depth)]
        )
        self.direction_tail = block(conditioned=False)
        self.scale_tail = block(conditioned=False)
        self.distribution_encoder = InputDistributionEncodingModule(
            DistributionConfig(
                k_s=cfg.distribution_k_s,
                Kq=cfg.distribution_Kq,
                d_var=cfg.distribution_d_var,
                d_dist=cfg.distribution_d_dist,
                num_var_attn_layers=cfg.distribution_num_var_attn_layers,
                var_attn_heads=cfg.distribution_var_attn_heads,
                dcn_num_cross_layers=cfg.distribution_dcn_num_cross_layers,
                dcn_deep_hidden=cfg.distribution_dcn_deep_hidden,
                dcn_deep_layers=cfg.distribution_dcn_deep_layers,
                dropout=0.0,
                use_covariance=cfg.distribution_use_covariance,
                patch_size_for_cov=cfg.values_per_token,
            )
        )
        self.decoder_query_conditioner = nn.Sequential(
            nn.Linear(dim + cfg.distribution_d_dist, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.direction_output_norm = RMSNorm(dim)
        self.scale_output_norm = RMSNorm(dim)
        self.direction_head = nn.Linear(dim, cfg.output_patch_size, bias=False)
        self.scale_head = nn.Linear(dim, 1, bias=True)
        self.group_embedding = nn.Embedding(cfg.tile_size, dim)
        self.chunk_embedding = nn.Embedding(self.chunks_per_group, dim)
        self.tile_row_embedding = nn.Embedding(cfg.max_tile_rows, dim)
        self.tile_col_embedding = nn.Embedding(cfg.max_tile_cols, dim)
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.continuous_projection = nn.Linear(cfg.values_per_token, dim, bias=False)
        self.direction_half_embedding = nn.Parameter(torch.empty(1, dim))
        self.scale_half_embedding = nn.Parameter(torch.empty(1, dim))
        self._reset_embeddings()

    @classmethod
    def _raw_scale_bias_for_initial_radius(cls) -> float:
        offset = cls.initial_log_radius - cls.log_scale_min
        return cls.log_scale_min + math.log(math.expm1(offset))

    @classmethod
    def _bound_log_scale(cls, raw_scale: torch.Tensor) -> torch.Tensor:
        bounded = cls.log_scale_min + F.softplus(raw_scale - cls.log_scale_min)
        return cls.log_scale_max - F.softplus(cls.log_scale_max - bounded)

    def _reset_embeddings(self) -> None:
        nn.init.normal_(self.latent_slots, std=0.02)
        nn.init.normal_(self.output_queries, std=0.02)
        nn.init.normal_(self.to_latent.weight, std=0.02)
        nn.init.normal_(self.from_latent.weight, std=0.02)
        nn.init.normal_(self.group_embedding.weight, std=0.02)
        nn.init.normal_(self.chunk_embedding.weight, std=0.02)
        nn.init.normal_(self.tile_row_embedding.weight, std=0.02)
        nn.init.zeros_(self.tile_col_embedding.weight)
        nn.init.normal_(self.continuous_projection.weight, std=0.02)
        nn.init.normal_(self.direction_half_embedding, std=0.02)
        nn.init.normal_(self.scale_half_embedding, std=0.02)
        for module in self.scale_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        nn.init.normal_(
            self.direction_head.weight,
            std=0.02 / math.sqrt(float(self.cfg.hidden_dim)),
        )
        nn.init.normal_(
            self.scale_head.weight,
            std=0.02 / math.sqrt(float(self.cfg.hidden_dim)),
        )
        nn.init.constant_(self.scale_head.bias, self._raw_scale_bias_for_initial_radius())

    def _tile_position_embedding(
        self, tile_row: torch.Tensor, tile_col: torch.Tensor
    ) -> torch.Tensor:
        tile_row = tile_row.long()
        tile_col = tile_col.long()
        if tile_row.ndim != 1 or tile_col.shape != tile_row.shape:
            raise ValueError("tile_row/tile_col must be aligned rank-1 tensors")
        if int(tile_row.min()) < 0 or int(tile_row.max()) >= self.cfg.max_tile_rows:
            raise ValueError("tile_row is outside the mini global coordinate range")
        if int(tile_col.min()) < 0 or int(tile_col.max()) >= self.cfg.max_tile_cols:
            raise ValueError("tile_col is outside the mini global coordinate range")
        return self.tile_row_embedding(tile_row) + self.tile_col_embedding(tile_col)

    def _encode_distribution_context(
        self,
        activation_context: torch.Tensor,
        sample_mask: torch.Tensor | None,
        source_indices: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if activation_context.ndim != 3 or activation_context.shape[-1] != 32:
            raise ValueError("conditioned mini model requires activation context [B,n,32]")
        batch, sample_count, _ = activation_context.shape
        original_batch = batch
        if (source_indices is None) != (group_index is None):
            raise ValueError("distribution deduplication requires both source_indices and group_index")
        if source_indices is not None and group_index is not None:
            source_indices = source_indices.to(device=activation_context.device, dtype=torch.long)
            group_index = group_index.to(device=activation_context.device, dtype=torch.long)
            if source_indices.ndim != 1 or group_index.shape != (original_batch,):
                raise ValueError("distribution context deduplication indices have invalid shapes")
            if int(source_indices.min()) < 0 or int(source_indices.max()) >= original_batch:
                raise ValueError("distribution source index is outside the physical batch")
            if int(group_index.min()) < 0 or int(group_index.max()) >= source_indices.numel():
                raise ValueError("distribution group index is outside the unique context batch")
            activation_context = activation_context.index_select(0, source_indices)
            if sample_mask is not None:
                sample_mask = sample_mask.index_select(0, source_indices)
            batch = int(source_indices.numel())
        patch_indices = torch.arange(
            32, device=activation_context.device, dtype=torch.long
        ).view(1, self.chunks_per_group, self.cfg.values_per_token).expand(batch, -1, -1)
        repeated_context = activation_context.unsqueeze(1).expand(
            batch, self.chunks_per_group, sample_count, 32
        ).reshape(batch * self.chunks_per_group, sample_count, 32)
        repeated_mask = None
        if sample_mask is not None:
            if tuple(sample_mask.shape) != (batch, sample_count):
                raise ValueError("sample mask does not match activation context")
            repeated_mask = sample_mask.unsqueeze(1).expand(
                batch, self.chunks_per_group, sample_count
            ).reshape(batch * self.chunks_per_group, sample_count)
        _var, patch = self.distribution_encoder(
            repeated_context,
            patch_indices.reshape(batch * self.chunks_per_group, self.cfg.values_per_token),
            sample_mask=repeated_mask,
        )
        patch = patch.view(batch, self.chunks_per_group, self.cfg.distribution_d_dist)
        if group_index is not None:
            patch = patch.index_select(0, group_index)
        return patch

    def encode(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_row: torch.Tensor,
        tile_col: torch.Tensor,
        token_valid_mask: torch.Tensor,
        dist_patch: torch.Tensor,
        *,
        capture_depth: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        batch = content.shape[0]
        expected = (batch, self.weight_token_count, self.cfg.values_per_token)
        if tuple(content.shape) != expected:
            raise ValueError(f"content shape must be {expected}, got {tuple(content.shape)}")
        if tuple(log_scale.shape) != (batch, self.weight_token_count, 1):
            raise ValueError("log_scale does not match token layout")
        if tuple(token_valid_mask.shape) != (batch, self.weight_token_count):
            raise ValueError("token mask does not match token layout")
        group_ids = torch.arange(32, device=content.device).repeat_interleave(
            self.chunks_per_group
        )
        chunk_ids = torch.arange(
            self.chunks_per_group, device=content.device
        ).repeat(32)
        position = self._tile_position_embedding(tile_row, tile_col)
        embedded = (
            self.continuous_projection(content.float())
            + self.scale_mlp(log_scale.float())
            + self.group_embedding(group_ids)[None]
            + self.chunk_embedding(chunk_ids)[None]
            + position[:, None]
        )
        token_valid_mask = token_valid_mask.bool()
        embedded = embedded * token_valid_mask.unsqueeze(-1).to(embedded.dtype)
        latent = self.latent_slots[None].expand(batch, -1, -1)
        state = torch.cat((latent, embedded), dim=1)
        state_valid = torch.cat(
            (
                torch.ones(batch, self.cfg.latent_slots, device=content.device, dtype=torch.bool),
                token_valid_mask,
            ),
            dim=1,
        )
        content_context = dist_patch.unsqueeze(1).expand(
            -1, 32, -1, -1
        ).reshape(batch, self.weight_token_count, self.cfg.distribution_d_dist)
        latent_context = content_context.new_zeros(
            batch, self.cfg.latent_slots, self.cfg.distribution_d_dist
        )
        key_context = torch.cat((latent_context, content_context), dim=1)
        telemetry: list[dict[str, float]] = []
        for depth, encoder_block in enumerate(self.encoder_blocks, start=1):
            state = encoder_block(
                state,
                key_context=key_context,
                valid_mask=state_valid,
            )
            if capture_depth:
                telemetry.append(
                    {
                        "depth": depth,
                        "latent_rms": float(
                            state[:, : self.cfg.latent_slots]
                            .float().square().mean().sqrt().item()
                        ),
                        "content_rms": float(
                            state[:, self.cfg.latent_slots :]
                            .float().square().mean().sqrt().item()
                        ),
                    }
                )
        z = self.to_latent(self.latent_norm(state[:, : self.cfg.latent_slots]))
        return z, telemetry

    def decode_polar(
        self,
        z: torch.Tensor,
        tile_row: torch.Tensor,
        tile_col: torch.Tensor,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        token_valid_mask: torch.Tensor,
        dist_patch: torch.Tensor,
        *,
        detach_direct_conditioning: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        expected_token_valid = (
            d_out_mask[:, :, None]
            & d_in_mask.view(batch, self.chunks_per_group, self.cfg.values_per_token)
            .any(dim=-1)[:, None, :]
        ).reshape(batch, self.weight_token_count)
        if not torch.equal(token_valid_mask.bool(), expected_token_valid):
            raise ValueError("token mask disagrees with d_in/d_out masks")
        latent = self.from_latent(z)
        queries = self.output_queries[None].expand(batch, -1, -1)
        queries = queries + self._tile_position_embedding(tile_row, tile_col)[:, None]
        decoder_context = dist_patch.unsqueeze(1).expand(
            -1, 32, -1, -1
        ).reshape(batch, self.weight_token_count, self.cfg.distribution_d_dist)
        queries = self.decoder_query_conditioner(
            torch.cat((queries, decoder_context.to(queries.dtype)), dim=-1)
        )
        queries = queries * expected_token_valid.unsqueeze(-1).to(queries.dtype)
        if detach_direct_conditioning:
            # The matched-negative decoder objective is allowed to update the
            # shared decoder/tails, but must not turn target-side coordinates
            # or distribution context into a shortcut for identifying the
            # foreign latent.  The ordinary positive path keeps this graph.
            queries = queries.detach()
        state = torch.cat((latent, queries), dim=1)
        state_valid = torch.cat(
            (
                torch.ones(batch, self.cfg.latent_slots, device=z.device, dtype=torch.bool),
                expected_token_valid,
            ),
            dim=1,
        )
        for decoder_block in self.decoder_blocks:
            state = decoder_block(state, valid_mask=state_valid)
        shared_latent = state[:, : self.cfg.latent_slots]
        queries = state[:, self.cfg.latent_slots :]
        direction_state = torch.cat(
            (shared_latent, queries + self.direction_half_embedding[None]), dim=1
        )
        scale_state = torch.cat(
            (shared_latent, queries + self.scale_half_embedding[None]), dim=1
        )
        direction_state = self.direction_tail(direction_state, valid_mask=state_valid)
        scale_state = self.scale_tail(scale_state, valid_mask=state_valid)
        direction_logits = self.direction_head(
            self.direction_output_norm(direction_state[:, self.cfg.latent_slots :])
        ).view(batch, 32, self.chunks_per_group, self.cfg.output_patch_size)
        raw_log_scale = self.scale_head(
            self.scale_output_norm(scale_state[:, self.cfg.latent_slots :])
        ).view(batch, 32, self.chunks_per_group)
        component_mask = (
            d_out_mask[:, :, None, None]
            & d_in_mask.view(batch, self.chunks_per_group, self.cfg.output_patch_size)[:, None]
        )
        direction_logits = direction_logits * component_mask.to(direction_logits.dtype)
        direction_norm = (
            direction_logits.float().square().sum(dim=-1, keepdim=True) + 1.0e-12
        ).sqrt()
        pred_dirs = direction_logits.float() / direction_norm
        pred_dirs = pred_dirs * component_mask.to(pred_dirs.dtype)
        pred_log_scales = self._bound_log_scale(raw_log_scale.float())
        pred_log_scales = pred_log_scales * component_mask.any(dim=-1).to(
            pred_log_scales.dtype
        )
        patches = pred_dirs * torch.exp(pred_log_scales).unsqueeze(-1)
        patches = patches * component_mask.to(patches.dtype)
        prediction = patches.reshape(batch, 32, 32).transpose(1, 2).contiguous()
        return prediction, pred_dirs, pred_log_scales

    def forward(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_row: torch.Tensor,
        activation_context: torch.Tensor,
        *,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        tile_col: torch.Tensor,
        activation_sample_mask: torch.Tensor,
        token_valid_mask: torch.Tensor,
        distribution_source_indices: torch.Tensor | None = None,
        distribution_group_index: torch.Tensor | None = None,
        capture_depth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]], torch.Tensor, torch.Tensor]:
        dist_patch = self._encode_distribution_context(
            activation_context,
            activation_sample_mask,
            distribution_source_indices,
            distribution_group_index,
        )
        z, telemetry = self.encode(
            content,
            log_scale,
            tile_row,
            tile_col,
            token_valid_mask,
            dist_patch,
            capture_depth=capture_depth,
        )
        prediction, pred_dirs, pred_log_scales = self.decode_polar(
            z,
            tile_row,
            tile_col,
            d_in_mask,
            d_out_mask,
            token_valid_mask,
            dist_patch,
        )
        return prediction, z, telemetry, pred_dirs, pred_log_scales


def _weights_from_polar_components(
    pred_dirs: torch.Tensor,
    pred_log_scales: torch.Tensor,
    *,
    detach_direction: bool,
    detach_scale: bool,
) -> torch.Tensor:
    directions = pred_dirs.detach() if detach_direction else pred_dirs
    scales = pred_log_scales.detach() if detach_scale else pred_log_scales
    patches = directions.float() * torch.exp(scales.float()).unsqueeze(-1)
    batch, d_out, chunks, patch = patches.shape
    if d_out != 32 or chunks * patch != 32:
        raise ValueError("polar components do not describe a tile32 matrix")
    return patches.reshape(batch, d_out, 32).transpose(1, 2).contiguous()


def _behavioral_direction_scale_routed(
    X: torch.Tensor,
    W: torch.Tensor,
    direction_prediction: torch.Tensor,
    scale_prediction: torch.Tensor,
    x_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    *,
    eps: float = 1.0e-8,
    gamma: float = 0.5,
    huber_delta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact historical terms with the shared target action evaluated once."""
    if X.ndim != 3 or W.ndim != 3:
        raise ValueError("behavioral routed loss expects batched X and W")
    if W.shape != direction_prediction.shape or W.shape != scale_prediction.shape:
        raise ValueError("behavioral routed predictions must match W")
    batch, samples, _ = X.shape
    d_out = int(W.shape[-1])
    if tuple(x_mask.shape) != (batch, samples):
        raise ValueError("x_mask does not match X")
    if tuple(d_out_mask.shape) != (batch, d_out):
        raise ValueError("d_out_mask does not match W")
    target = torch.matmul(X, W)
    direction_action = torch.matmul(X, direction_prediction)
    scale_action = torch.matmul(X, scale_prediction)
    output_mask = d_out_mask.to(device=X.device, dtype=target.dtype).unsqueeze(1)
    target = target * output_mask
    direction_action = direction_action * output_mask
    scale_action = scale_action * output_mask
    sample_mask = x_mask.to(device=X.device, dtype=target.dtype)
    target_norm = target.norm(dim=-1)
    active = target_norm > float(eps)
    row_weight = sample_mask * active.to(dtype=sample_mask.dtype)

    direction_norm = direction_action.norm(dim=-1)
    cosine = (direction_action * target).sum(dim=-1) / (
        direction_norm * target_norm
    ).clamp_min(float(eps))
    direction_loss = 1.0 - cosine.clamp(min=-1.0, max=1.0)
    direction_weight = row_weight * (target_norm + float(eps)).pow(float(gamma))
    behavioral_direction = (direction_loss * direction_weight).sum() / (
        direction_weight.sum().clamp_min(1.0)
    )

    scale_norm = scale_action.norm(dim=-1)
    delta = torch.log(scale_norm + float(eps)) - torch.log(target_norm + float(eps))
    abs_delta = delta.abs()
    huber = torch.where(
        abs_delta <= huber_delta,
        0.5 * delta.square(),
        huber_delta * (abs_delta - 0.5 * huber_delta),
    )
    behavioral_scale = (huber * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    return behavioral_direction, behavioral_scale


def _polar_routed_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    pred_dirs: torch.Tensor,
    pred_log_scales: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if float(loss_cfg["behavioral_operator"]) != 0.0:
        raise ValueError("polar routed loss requires raw operator MSE weight 0")
    direction_prediction = _weights_from_polar_components(
        pred_dirs, pred_log_scales, detach_direction=False, detach_scale=True
    )
    scale_prediction = _weights_from_polar_components(
        pred_dirs, pred_log_scales, detach_direction=True, detach_scale=False
    )
    behavioral_dir, behavioral_scale = _behavioral_direction_scale_routed(
        X,
        W,
        direction_prediction,
        scale_prediction,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )
    structural_dir, structural_dir_parts = BigWeightVAELossMixin.patch_structure_loss(
        W,
        direction_prediction,
        patch_size=16,
        gamma=0.5,
        lambda_dir=float(loss_cfg["structural_direction"]),
        lambda_scale=0.0,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=0.1,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        pred_dirs=pred_dirs,
    )
    structural_scale, structural_scale_parts = BigWeightVAELossMixin.patch_structure_loss(
        W,
        scale_prediction,
        patch_size=16,
        gamma=0.5,
        lambda_dir=0.0,
        lambda_scale=float(loss_cfg["structural_scale"]),
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=0.1,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        pred_dirs=pred_dirs.detach(),
    )
    behavioral = (
        float(loss_cfg["behavioral_direction"]) * behavioral_dir
        + float(loss_cfg["behavioral_scale"]) * behavioral_scale
    )
    structural = structural_dir + structural_scale
    total = (
        float(loss_cfg["behavioral_coef"]) * behavioral
        + float(loss_cfg["structural_coef"]) * structural
    )
    zero = total.new_zeros(())
    return total, {
        "behavioral": behavioral.detach(),
        "behavioral_operator": zero.detach(),
        "behavioral_direction": behavioral_dir.detach(),
        "behavioral_scale": behavioral_scale.detach(),
        "structural": structural.detach(),
        "structural_direction": structural_dir_parts["L_dir"],
        "structural_scale": structural_scale_parts["L_scale"],
    }


def _open_subtile_stream(
    stack: ExitStack,
    config: dict[str, Any],
    cursor: SubtileCursor,
    logger: logging.Logger,
) -> tuple[ValidSubtileStream, BalancedOperatorBankMixer]:
    bank = config["operator_bank"]
    dataset, sampler = stack.enter_context(
        operator_bank_data_pipeline(
            bank["pair_manifest"],
            seed=int(config["seed"]),
            repeat=True,
            permutation_views=bool(bank["permutation_views"]),
            canonical_probability=float(bank["canonical_probability"]),
            hot_shards=int(bank["hot_shards"]),
            expected_pair_manifest_sha256=str(bank["pair_manifest_sha256"]),
            rank=0,
            world_size=1,
            max_active_strata=int(bank["max_active_strata"]),
            max_active_bundle_bytes=int(bank["max_active_bundle_bytes"]),
            logger=logger,
        )
    )
    sampler.set_start_index(cursor.parent_logical_index)
    workers = int(bank["loader_workers"])
    loader_batch_size = int(bank["loader_batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=(None if loader_batch_size <= 1 else loader_batch_size),
        sampler=sampler,
        num_workers=workers,
        collate_fn=(
            _identity_sample_collate if loader_batch_size <= 1 else _sample_list_collate
        ),
        prefetch_factor=(int(bank["loader_prefetch_factor"]) if workers else None),
        persistent_workers=bool(workers),
        pin_memory=False,
        worker_init_fn=_offline_loader_worker_init_fn,
        generator=torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 7_919),
    )
    loader_iter = iter(loader)
    bundle_iter = loader_iter if loader_batch_size <= 1 else _flatten_loader_batches(loader_iter)
    parent_stream = BalancedOperatorBankMixer(
        dataset,
        bundle_iter,
        start_index=cursor.parent_logical_index,
    )
    shutdown_workers = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown_workers):
        stack.callback(shutdown_workers)
    return ValidSubtileStream(parent_stream, cursor), parent_stream


def _normalization_artifact(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    norm = config["normalization"]
    path = Path(norm["stats_path"]).resolve()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": "tile32_log2_scale_calibration_v1",
            "seed": int(config["seed"]),
            "valid_subtiles": int(norm["calibration_valid_subtiles"]),
            "pair_manifest_sha256": str(config["operator_bank"]["pair_manifest_sha256"]),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"normalization artifact field {key!r} drifted: {payload.get(key)!r} != {value!r}"
                )
        print(
            "[mini-polar] stage=normalization-cache-hit "
            f"path={path} mean={payload['log2_scale_mean']:.9g} "
            f"std={payload['log2_scale_std']:.9g} rows={payload['valid_output_rows']}",
            flush=True,
        )
        return payload

    target = int(norm["calibration_valid_subtiles"])
    print(
        "[mini-polar] stage=normalization-calibration-start "
        f"valid_subtiles={target} path={path}",
        flush=True,
    )
    started = time.monotonic()
    count = 0
    value_count = 0
    value_sum = 0.0
    value_square_sum = 0.0
    with ExitStack() as stack:
        stream, parent_stream = _open_subtile_stream(
            stack,
            config,
            SubtileCursor(0, 0, 0),
            logger,
        )
        while count < target:
            sample = next(stream)
            rows = sample.weight.transpose(0, 1).float()
            d_in_mask = sample.meta["d_in_mask"].bool()
            d_out_mask = sample.meta["d_out_mask"].bool()
            row_scales = rows[:, d_in_mask].abs().amax(dim=-1).clamp_min(1.0e-8) / 7.0
            values = torch.log2(row_scales[d_out_mask]).double()
            value_count += int(values.numel())
            value_sum += float(values.sum().item())
            value_square_sum += float(values.square().sum().item())
            count += 1
            if count % 4096 == 0 or count == target:
                elapsed = time.monotonic() - started
                print(
                    "[mini-polar] stage=normalization-calibration-progress "
                    f"subtiles={count}/{target} rows={value_count} "
                    f"rate={count/max(elapsed,1e-9):.1f}_tiles/s",
                    flush=True,
                )
        mean = value_sum / value_count
        variance = max(value_square_sum / value_count - mean * mean, 0.0)
        std = math.sqrt(variance)
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 1.0e-6:
            raise RuntimeError(f"invalid normalization calibration mean/std: {mean}, {std}")
        payload = {
            "schema": "tile32_log2_scale_calibration_v1",
            "seed": int(config["seed"]),
            "valid_subtiles": target,
            "valid_output_rows": value_count,
            "pair_manifest": str(config["operator_bank"]["pair_manifest"]),
            "pair_manifest_sha256": str(config["operator_bank"]["pair_manifest_sha256"]),
            "permutation_views": bool(config["operator_bank"]["permutation_views"]),
            "canonical_probability": float(config["operator_bank"]["canonical_probability"]),
            "qmax": 7.0,
            "log2_scale_mean": mean,
            "log2_scale_std": std,
            "subtile_stream": stream.telemetry(),
            "parent_stream": parent_stream.telemetry(),
            "elapsed_seconds": time.monotonic() - started,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    _atomic_json(path, payload)
    print(
        "[mini-polar] stage=normalization-calibration-complete "
        f"mean={mean:.9g} std={std:.9g} rows={value_count} path={path}",
        flush=True,
    )
    return payload


def _masked_weight_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    mask = d_in_mask[:, :, None] & d_out_mask[:, None, :]
    squared = (prediction.float() - target.float()).square() * mask
    return torch.sqrt(squared.sum() / mask.sum().clamp_min(1))


def _plot_metrics(path: Path, output: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, label in (
        ("loss", "total"),
        ("behavioral", "behavioral"),
        ("structural", "structural"),
        ("behavioral_direction", "behavioral direction"),
        ("behavioral_scale", "behavioral scale"),
    ):
        ax.plot(steps, [row[key] for row in rows], label=label, linewidth=1.2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss")
    ax.set_title("Mini 9.94M conditioned polar regression — tile32/p16")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    model_cfg = MiniPolarConfig(**config["architecture"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_root is not None:
        output_root = args.output_root.resolve()
    elif args.smoke:
        output_root = Path(
            f"/mnt/shared/weightclip_benchmark/mini_polar_regression_smoke_{stamp}"
        )
    else:
        output_root = Path(config["output_root"]).resolve()
    if args.smoke and args.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive")
    steps = int(args.smoke_steps) if args.smoke else int(config["steps"])
    if args.smoke_batch_size is not None and not args.smoke:
        raise ValueError("--smoke-batch-size requires --smoke")
    batch_size = (
        int(args.smoke_batch_size)
        if args.smoke_batch_size is not None
        else int(config["batch_size"])
    )
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    resume_path = (
        output_root / "resume_latest.pt"
        if args.smoke
        else Path(config["resume_checkpoint"]).resolve()
    )
    persistent_model_path = (
        output_root / "model_latest.pt"
        if args.smoke
        else Path(config["persistent_model_checkpoint"]).resolve()
    )
    startup = {
        "schema": SCHEMA,
        "stage": "preflight",
        "config_path": str(config_path),
        "resolved_config": config,
        "model_config": asdict(model_cfg),
        "device": str(config["device"]),
        "dtype": "bfloat16 autocast; FP32 parameters and loss statistics",
        "seed": int(config["seed"]),
        "cache_mode": "sealed operator bank plus cached sampled tile32 normalization",
        "output_root": str(output_root),
        "steps": steps,
        "scientific_horizon_steps": int(config["steps"]),
        "batch_size": batch_size,
        "resume_checkpoint": str(resume_path),
        "persistent_model_checkpoint": str(persistent_model_path),
    }
    print("[mini-polar] stage=preflight", json.dumps(startup), flush=True)
    if args.dry_run:
        print("[mini-polar] stage=complete mode=dry-run", flush=True)
        return
    if output_root.exists() and not args.resume and not args.calibrate_only:
        raise FileExistsError(f"fresh output root already exists: {output_root}")
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")

    logger = logging.getLogger("mini-polar")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    normalization = _normalization_artifact(config, logger)
    if args.calibrate_only:
        print("[mini-polar] stage=complete mode=calibrate-only", flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    startup["normalization_artifact"] = normalization
    if not args.resume:
        _atomic_json(output_root / "resolved_config.json", startup)

    device = torch.device(str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production run requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(config["seed"]))

    print("[mini-polar] stage=model-build", flush=True)
    model = MiniPolarWeightBottleneck(model_cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"mini parameter count drifted: {parameter_count} != {EXPECTED_PARAMETERS}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_step = 0
    cursor = SubtileCursor(0, 0, 0)
    if args.resume:
        print(f"[mini-polar] stage=resume-load path={resume_path}", flush=True)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["schema"] != SCHEMA or payload["config"] != config:
            raise RuntimeError("resume checkpoint schema/config disagrees with requested run")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["step"])
        cursor = SubtileCursor(**payload["subtile_cursor"])
        if cursor.emitted_logical_index != start_step * batch_size:
            raise RuntimeError("resume emitted cursor disagrees with step and batch size")
        _restore_rng_state(payload["rng_state"])
        del payload
        resume_config_path = output_root / f"resolved_resume_config_step_{start_step:09d}.json"
        _atomic_json(resume_config_path, startup)
    print(
        "[mini-polar] stage=model-build-complete "
        f"parameters={parameter_count} trainable={parameter_count} "
        f"start_step={start_step} cursor={cursor.to_dict()}",
        flush=True,
    )

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[mini-polar] stage=stop-requested signal={signum}; saving after current step",
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    metrics_path = output_root / "train_metrics.jsonl"
    gradient_path = output_root / "gradient_telemetry.jsonl"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def save_resume(step: int, current_cursor: SubtileCursor) -> None:
        print(
            f"[mini-polar] stage=resume-save step={step} path={resume_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": SCHEMA,
                "step": step,
                "subtile_cursor": current_cursor.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state(),
                "config": config,
                "normalization_artifact": normalization,
            },
            resume_path,
        )

    def save_model(step: int, current_cursor: SubtileCursor) -> None:
        print(
            f"[mini-polar] stage=model-checkpoint-save step={step} path={persistent_model_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": SCHEMA,
                "step": step,
                "subtile_cursor": current_cursor.to_dict(),
                "model_state": model.state_dict(),
                "model_config": asdict(model_cfg),
                "config": config,
                "normalization_artifact": normalization,
            },
            persistent_model_path,
        )

    print(
        "[mini-polar] stage=data-open "
        f"pair_manifest={config['operator_bank']['pair_manifest']} "
        f"workers={config['operator_bank']['loader_workers']} "
        f"cursor={cursor.to_dict()}",
        flush=True,
    )
    with ExitStack() as stack:
        stream, parent_stream = _open_subtile_stream(stack, config, cursor, logger)
        print(
            "[mini-polar] stage=train "
            f"steps={steps} batch={batch_size} lr={config['learning_rate']} "
            "objective=behavioral(direction+10*scale)+structural(direction+10*scale) "
            "raw_operator_mse=disabled routing=coordinate-owned-stop-gradient "
            "distribution_encoder=enabled",
            flush=True,
        )
        last_step = start_step
        training_started = time.monotonic()
        for step in range(start_step + 1, steps + 1):
            step_started = time.monotonic()
            cpu_batch = _batch_from_stream(stream, batch_size)
            W = cpu_batch["W"].to(device, non_blocking=True)
            X = cpu_batch["X"].to(device, non_blocking=True)
            x_mask = cpu_batch["x_mask"].to(device, non_blocking=True)
            d_in_mask = cpu_batch["d_in_mask"].to(device, non_blocking=True)
            d_out_mask = cpu_batch["d_out_mask"].to(device, non_blocking=True)
            tile_row = cpu_batch["tile_row"].to(device, non_blocking=True)
            tile_col = cpu_batch["tile_col"].to(device, non_blocking=True)
            distribution_source_indices = cpu_batch["distribution_source_indices"].to(
                device, non_blocking=True
            )
            distribution_group_index = cpu_batch["distribution_group_index"].to(
                device, non_blocking=True
            )
            content, log_scale, token_valid = _prepare_normalized_inputs(
                W,
                d_in_mask,
                d_out_mask,
                scale_mean=float(normalization["log2_scale_mean"]),
                scale_std=float(normalization["log2_scale_std"]),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gradient_due = step == 1 or step % int(config["gradient_log_every"]) == 0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction, latent, depth_rows, pred_dirs, pred_log_scales = model(
                    content,
                    log_scale,
                    tile_row,
                    X,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    tile_col=tile_col,
                    activation_sample_mask=x_mask,
                    token_valid_mask=token_valid,
                    distribution_source_indices=distribution_source_indices,
                    distribution_group_index=distribution_group_index,
                    capture_depth=gradient_due,
                )
                loss, parts = _polar_routed_loss(
                    X,
                    W,
                    pred_dirs,
                    pred_log_scales,
                    x_mask,
                    d_in_mask,
                    d_out_mask,
                    config["loss"],
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite loss at step {step}: {float(loss)}")
            loss.backward()
            if step == 1:
                missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
                if missing:
                    raise RuntimeError(f"step-1 model parameters without gradients: {missing}")
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["grad_clip_norm"])
            )
            optimizer.step()
            torch.cuda.synchronize(device)
            current_cursor = stream.cursor()
            expected_emitted = step * batch_size
            if current_cursor.emitted_logical_index != expected_emitted:
                raise RuntimeError(
                    f"subtile cursor drift: {current_cursor.emitted_logical_index} != {expected_emitted}"
                )
            elapsed = time.monotonic() - training_started
            step_seconds = time.monotonic() - step_started
            log_due = args.smoke or step == 1 or step % int(config["log_every"]) == 0
            row: dict[str, Any] | None = None
            if log_due:
                with torch.no_grad():
                    operator_relative_mse = BigWeightVAELossMixin.operator_relative_mse_loss(
                        X,
                        W,
                        prediction,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                    )
                    weight_rmse = _masked_weight_rmse(
                        prediction, W, d_in_mask, d_out_mask
                    )
                    valid_patches = (
                        d_out_mask[:, :, None]
                        & d_in_mask.view(batch_size, 2, 16).any(dim=-1)[:, None]
                    )
                    valid_scales = pred_log_scales[valid_patches]
                    row = {
                        "schema": SCHEMA,
                        "step": step,
                        "loss": float(loss.detach()),
                        **{key: float(value) for key, value in parts.items()},
                        "operator_relative_mse": float(operator_relative_mse),
                        "weight_rmse": float(weight_rmse),
                        "latent_rms": float(latent.float().square().mean().sqrt()),
                        "latent_batch_std": float(latent.float().std(dim=0, unbiased=False).mean()),
                        "pred_log_scale_mean": float(valid_scales.float().mean()),
                        "pred_log_scale_std": float(valid_scales.float().std(unbiased=False)),
                        "pre_clip_gradient_norm": float(pre_clip_norm),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "batch_size": batch_size,
                        "tile_size": 32,
                        "parameter_count": parameter_count,
                        "distribution_unique_contexts": int(
                            distribution_source_indices.numel()
                        ),
                        "distribution_context_reuse": float(
                            batch_size / distribution_source_indices.numel()
                        ),
                        "emitted_logical_index": current_cursor.emitted_logical_index,
                        "parent_logical_index": current_cursor.parent_logical_index,
                        "next_subpatch_index": current_cursor.subpatch_index,
                        "step_seconds": step_seconds,
                        "steps_per_second": 1.0 / max(step_seconds, 1.0e-9),
                        "examples_per_second": batch_size / max(step_seconds, 1.0e-9),
                        "weight_scalars_per_second": batch_size * 32 * 32 / max(step_seconds, 1.0e-9),
                        "elapsed_seconds": elapsed,
                        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    }
                _append_jsonl(metrics_path, row)
            if gradient_due:
                gradient_row = {
                    "schema": SCHEMA,
                    "step": step,
                    "pre_clip_gradient_norm": float(pre_clip_norm),
                    "groups": _gradient_telemetry(model),
                    "depth": depth_rows,
                }
                _append_jsonl(gradient_path, gradient_row)
            if log_due:
                assert row is not None
                print(
                    "[mini-polar] stage=train-progress "
                    f"step={step}/{steps} loss={row['loss']:.6f} "
                    f"behavioral={row['behavioral']:.6f} structural={row['structural']:.6f} "
                    f"op_rel_mse={row['operator_relative_mse']:.6f} "
                    f"grad={row['pre_clip_gradient_norm']:.4f} "
                    f"rate={row['steps_per_second']:.3f}_steps/s "
                    f"examples={row['examples_per_second']:.1f}/s "
                    f"cursor={current_cursor.to_dict()} "
                    f"peak_alloc_gib={row['max_cuda_allocated_bytes']/2**30:.2f}",
                    flush=True,
                )
            if step % int(config["plot_every"]) == 0:
                plot_path = output_root / "train_losses.png"
                _plot_metrics(metrics_path, plot_path)
                print(f"[mini-polar] stage=plot-write path={plot_path}", flush=True)
            if step % int(config["resume_save_every"]) == 0:
                save_resume(step, current_cursor)
            if step % int(config["model_save_every"]) == 0:
                save_model(step, current_cursor)
            last_step = step
            if stop_requested:
                break

        final_cursor = stream.cursor()
        save_resume(last_step, final_cursor)
        save_model(last_step, final_cursor)
        plot_path = output_root / "train_losses.png"
        _plot_metrics(metrics_path, plot_path)
        summary = {
            "schema": SCHEMA,
            "status": "stopped" if stop_requested else "complete",
            "step": last_step,
            "scientific_horizon_steps": int(config["steps"]),
            "subtile_cursor": final_cursor.to_dict(),
            "parameter_count": parameter_count,
            "elapsed_seconds": time.monotonic() - started,
            "metrics_path": str(metrics_path),
            "gradient_path": str(gradient_path),
            "plot_path": str(plot_path),
            "resume_checkpoint": str(resume_path),
            "persistent_model_checkpoint": str(persistent_model_path),
            "subtile_stream": stream.telemetry(),
            "parent_stream": parent_stream.telemetry(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(output_root / ("STOPPED.json" if stop_requested else "summary.json"), summary)
        print(
            "[mini-polar] stage=complete "
            f"status={summary['status']} step={last_step} metrics={metrics_path} "
            f"plot={plot_path} resume={resume_path} model={persistent_model_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
