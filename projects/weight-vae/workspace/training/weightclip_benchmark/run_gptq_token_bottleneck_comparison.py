from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from big_vae.models.distribution_encoder import (
    DistributionConfig,
    InputDistributionEncodingModule,
)


SCHEMA = "weightclip_gptq_token_bottleneck_comparison_v2_structural"
ARMS = (
    "raw_float",
    "normalized_float",
    "gptq_continuous",
    "gptq_token",
    "activation_sqrt",
)
DEFAULT_SELECTION = Path(
    "/mnt/shared/weightclip_benchmark/"
    "ae_v11_exact64_four_trunk_complement_v1_1984step/operator_set_selection.json"
)
DEFAULT_RESOLVED = Path(
    "/mnt/shared/weightclip_benchmark/"
    "ae_v11_exact64_four_trunk_complement_v1_1984step/resolved_run_config.json"
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    steps: int = 512
    batch_size: int = 6
    eval_batch_size: int = 12
    eval_every: int = 64
    log_every: int = 10
    learning_rate: float = 1.0e-4
    warmup_steps: int = 32
    weight_decay: float = 0.01
    hidden_dim: int = 512
    heads: int = 8
    mlp_dim: int = 2048
    encoder_depth: int = 4
    decoder_depth: int = 4
    latent_slots: int = 32
    latent_dim: int = 384
    values_per_token: int = 16
    quant_bits: int = 4
    gptq_damp_fraction: float = 0.01
    train_activation_rows: int = 256
    test_activation_rows: int = 256
    structural_patch_size: int = 16
    structural_gamma: float = 0.5
    structural_direction_weight: float = 1.0
    structural_scale_weight: float = 0.1
    structural_huber_delta: float = 0.1
    use_distribution_conditioning: bool = False
    distribution_k_s: int = 64
    distribution_Kq: int = 128
    distribution_d_var: int = 256
    distribution_d_dist: int = 256
    distribution_num_var_attn_layers: int = 6
    distribution_var_attn_heads: int = 4
    distribution_dcn_num_cross_layers: int = 3
    distribution_dcn_deep_hidden: int = 128
    distribution_dcn_deep_layers: int = 3
    distribution_use_covariance: bool = True
    activation_checkpointing: bool = False
    max_tile_rows: int = 9
    max_tile_cols: int = 1
    bounded_cosine_attention: bool = False
    attention_logit_scale: float = 2.0
    bounded_swiglu_hidden: bool = False
    bounded_residual_writes: bool = False


def _parse_args() -> argparse.Namespace:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw-float, normalized-float, continuous-GPTQ, token-GPTQ, "
            "and activation-metric inputs under the Big-VAE structural loss."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--resolved-config", type=Path, default=DEFAULT_RESOLVED)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            f"/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_v2_{stamp}"
        ),
    )
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Stop and finalize after this many updates while retaining the --steps LR schedule.",
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument(
        "--arms",
        type=str,
        default=",".join(ARMS),
        help="Comma-separated representation arms to run.",
    )
    parser.add_argument("--distribution-conditioning", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _scan_bank_records(
    root: Path,
    predicate: Any,
) -> list[tuple[int, int, dict[str, Any]]]:
    found: list[tuple[int, int, dict[str, Any]]] = []
    for shard_dir in sorted(root.glob("shard-*")):
        shard_index = int(shard_dir.name.split("-")[-1])
        for offset, row in enumerate(_read_jsonl(shard_dir / "records.jsonl")):
            if predicate(row):
                found.append((shard_index, offset, row))
    return found


def _load_exact64_tiles(
    selection_path: Path,
    resolved_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    operators = list(selection["operators"])
    if len(operators) != 64:
        raise RuntimeError(f"expected 64 operators, found {len(operators)}")
    keys = [(str(row["checkpoint_sha256"]), str(row["layer_key"])) for row in operators]
    if len(set(keys)) != 64:
        raise RuntimeError("operator selection contains duplicate identities")
    pair_path = Path(resolved["train"]["operator_bank"]["pair_manifest"])
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    weight_root = Path(pair["weight_tile_bank"])
    context_root = Path(pair["context_bank"])
    key_set = set(keys)

    print(
        f"[gptq-bottleneck] stage=data-scan bank={weight_root} selected_operators=64",
        flush=True,
    )
    weight_records = _scan_bank_records(
        weight_root,
        lambda row: (str(row["checkpoint_sha256"]), str(row["layer_key"])) in key_set,
    )
    if len(weight_records) != 64 * 9:
        raise RuntimeError(f"expected 576 weight records, found {len(weight_records)}")
    context_ids = {str(row[2]["context_id"]) for row in weight_records}
    context_records = _scan_bank_records(
        context_root,
        lambda row: str(row["context_id"]) in context_ids,
    )
    context_by_id = {str(row[2]["context_id"]): row for row in context_records}
    if set(context_by_id) != context_ids:
        raise RuntimeError("selected weight records do not have complete activation context")

    weights_by_key: dict[tuple[str, str], list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for record in weight_records:
        row = record[2]
        weights_by_key[(str(row["checkpoint_sha256"]), str(row["layer_key"]))].append(record)

    weight_arrays: dict[int, np.ndarray] = {}
    context_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    weights = torch.empty((64, 9, 128, 128), dtype=torch.float32)
    contexts = torch.empty((64, 9, 512, 128), dtype=torch.float32)
    sample_masks = torch.empty((64, 9, 512), dtype=torch.bool)
    tile_rows = torch.empty((64, 9), dtype=torch.long)
    identities: list[dict[str, Any]] = []

    for operator_index, (operator_row, key) in enumerate(zip(operators, keys, strict=True)):
        records = sorted(
            weights_by_key[key],
            key=lambda item: (
                int(item[2]["tile"]["row_start"]),
                int(item[2]["tile"]["col_start"]),
            ),
        )
        starts = [int(item[2]["tile"]["row_start"]) for item in records]
        if starts != list(range(0, 1152, 128)):
            raise RuntimeError(f"operator {key} has unexpected tile starts: {starts}")
        for tile_index, (weight_shard, weight_offset, metadata) in enumerate(records):
            if weight_shard not in weight_arrays:
                weight_arrays[weight_shard] = np.load(
                    weight_root / f"shard-{weight_shard:06d}" / "weight.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
            weights[operator_index, tile_index].copy_(
                torch.from_numpy(np.array(weight_arrays[weight_shard][weight_offset], copy=True))
            )
            context_shard, context_offset, context_meta = context_by_id[str(metadata["context_id"])]
            if context_shard not in context_arrays:
                shard_root = context_root / f"shard-{context_shard:06d}"
                context_arrays[context_shard] = (
                    np.load(shard_root / "raw_rows.npy", mmap_mode="r", allow_pickle=False),
                    np.load(shard_root / "sample_mask.npy", mmap_mode="r", allow_pickle=False),
                )
            raw_rows, raw_mask = context_arrays[context_shard]
            contexts[operator_index, tile_index].copy_(
                torch.from_numpy(np.array(raw_rows[context_offset], copy=True))
            )
            sample_masks[operator_index, tile_index].copy_(
                torch.from_numpy(np.array(raw_mask[context_offset], copy=True)).bool()
            )
            tile_rows[operator_index, tile_index] = int(context_meta["input_row_start"]) // 128
        identities.append(
            {
                "operator_index": operator_index,
                "checkpoint_sha256": key[0],
                "layer_key": key[1],
                "lineage_id": str(operator_row["lineage_id"]),
                "dataset": str(operator_row["dataset"]),
            }
        )

    if not bool(sample_masks.all()):
        valid_per_tile = sample_masks.sum(dim=-1)
        if int(valid_per_tile.min()) < 512:
            raise RuntimeError(
                "this bounded experiment requires 512 valid activation rows per exact64 tile"
            )
    heldout_operator_indices = list(range(3, 64, 4))
    train_operator_indices = [index for index in range(64) if index not in heldout_operator_indices]
    if len(train_operator_indices) != 48 or len(heldout_operator_indices) != 16:
        raise RuntimeError("operator-grouped 48/16 split construction failed")
    return {
        "weights": weights,
        "contexts": contexts,
        "tile_rows": tile_rows,
        "identities": identities,
        "train_operator_indices": train_operator_indices,
        "heldout_operator_indices": heldout_operator_indices,
        "selection_sha256": _sha256(selection_path),
        "resolved_config_sha256": _sha256(resolved_path),
        "pair_manifest": str(pair_path),
    }


@torch.no_grad()
def _gptq_quantize_batch(
    weights: torch.Tensor,
    calibration_x: torch.Tensor,
    *,
    bits: int,
    damp_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-output-group GPTQ with one scale per 128 weights."""

    if weights.ndim != 3 or weights.shape[-2:] != (128, 128):
        raise ValueError(f"unexpected weight shape {tuple(weights.shape)}")
    if calibration_x.shape != (weights.shape[0], 256, 128):
        raise ValueError(f"unexpected calibration shape {tuple(calibration_x.shape)}")
    qmax = 2 ** (bits - 1) - 1
    weight_rows = weights.transpose(1, 2).float().contiguous()  # [B, output, input]
    scales = weight_rows.abs().amax(dim=-1).clamp_min(1.0e-8) / float(qmax)
    hessian = calibration_x.float().transpose(1, 2) @ calibration_x.float()
    hessian.mul_(2.0 / float(calibration_x.shape[1]))
    diagonal_mean = hessian.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    hessian.diagonal(dim1=-2, dim2=-1).add_(
        damp_fraction * diagonal_mean[:, None].clamp_min(1.0e-8)
    )
    inverse = torch.linalg.inv(hessian)
    upper = torch.linalg.cholesky(inverse, upper=True)
    work = weight_rows.clone()
    codes = torch.empty_like(work, dtype=torch.int8)
    for column in range(128):
        current = work[:, :, column]
        code = torch.round(current / scales).clamp(-qmax, qmax)
        quantized = code * scales
        codes[:, :, column] = code.to(torch.int8)
        error = (current - quantized) / upper[:, column, column, None]
        work[:, :, column:] -= error[:, :, None] * upper[:, None, column, column:]
    dequantized = codes.float() * scales[:, :, None]
    return codes, scales, dequantized.transpose(1, 2).contiguous()


def _prepare_representations(
    data: dict[str, Any],
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    weights = data["weights"].reshape(-1, 128, 128)
    contexts = data["contexts"].reshape(-1, 512, 128)
    tile_rows = data["tile_rows"].reshape(-1)
    if 128 % cfg.values_per_token != 0:
        raise ValueError("values_per_token must divide 128")
    chunks_per_group = 128 // cfg.values_per_token
    print(
        f"[gptq-bottleneck] stage=gptq-preprocess tiles={weights.shape[0]} "
        f"calibration_rows={cfg.train_activation_rows} bits={cfg.quant_bits}",
        flush=True,
    )
    code_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    dequant_chunks: list[torch.Tensor] = []
    preprocess_batch = 48
    for start in range(0, weights.shape[0], preprocess_batch):
        stop = min(start + preprocess_batch, weights.shape[0])
        batch_w = weights[start:stop].to(device)
        batch_x = contexts[start:stop, : cfg.train_activation_rows].to(device)
        codes, scales, dequantized = _gptq_quantize_batch(
            batch_w,
            batch_x,
            bits=cfg.quant_bits,
            damp_fraction=cfg.gptq_damp_fraction,
        )
        code_chunks.append(codes.cpu())
        scale_chunks.append(scales.cpu())
        dequant_chunks.append(dequantized.cpu())
        print(
            f"[gptq-bottleneck] stage=gptq-preprocess progress={stop}/{weights.shape[0]}",
            flush=True,
        )
    codes = torch.cat(code_chunks, dim=0)
    scales = torch.cat(scale_chunks, dim=0)
    dequantized = torch.cat(dequant_chunks, dim=0)
    weight_rows = weights.transpose(1, 2).contiguous()
    normalized = weight_rows / scales[:, :, None]

    print(
        "[gptq-bottleneck] stage=activation-sqrt-preprocess "
        "definition=A_train^(1/2)@W full_operator=true",
        flush=True,
    )
    full_weights = weights.reshape(64, 9, 128, 128).reshape(64, 1152, 128)
    full_contexts = contexts.reshape(64, 9, 512, 128).permute(0, 2, 1, 3).reshape(64, 512, 1152)
    activation_sqrt_full: list[torch.Tensor] = []
    for start in range(0, 64, 8):
        stop = min(start + 8, 64)
        x_train = full_contexts[start:stop, : cfg.train_activation_rows].to(device)
        w_full = full_weights[start:stop].to(device)
        # If X = U S V^T, then A^(1/2) = V (S/sqrt(n)) V^T for
        # A = X^T X / n.  The thin SVD is exact on the data-supported
        # subspace and avoids materializing a 1152 x 1152 root.
        _u, singular, vh = torch.linalg.svd(x_train.float(), full_matrices=False)
        projected = vh.bmm(w_full.float())
        projected.mul_(singular[:, :, None] / math.sqrt(cfg.train_activation_rows))
        transformed = vh.transpose(1, 2).bmm(projected)
        activation_sqrt_full.append(transformed.cpu())
        print(
            f"[gptq-bottleneck] stage=activation-sqrt-preprocess progress={stop}/64",
            flush=True,
        )
    activation_sqrt_weights = torch.cat(activation_sqrt_full, dim=0).reshape(
        64 * 9, 128, 128
    )
    activation_rows = activation_sqrt_weights.transpose(1, 2).contiguous()
    activation_scales = activation_rows.abs().amax(dim=-1).clamp_min(1.0e-8) / float(
        2 ** (cfg.quant_bits - 1) - 1
    )
    activation_normalized = activation_rows / activation_scales[:, :, None]

    train_ops = set(data["train_operator_indices"])
    train_tile_mask = torch.tensor(
        [operator_index in train_ops for operator_index in range(64) for _ in range(9)],
        dtype=torch.bool,
    )
    log_scale = torch.log2(scales.clamp_min(1.0e-12))
    scale_mean = float(log_scale[train_tile_mask].mean().item())
    scale_std = float(log_scale[train_tile_mask].std(unbiased=False).clamp_min(1.0e-6).item())
    standardized_log_scale = (log_scale - scale_mean) / scale_std
    activation_log_scale = torch.log2(activation_scales.clamp_min(1.0e-12))
    activation_scale_mean = float(activation_log_scale[train_tile_mask].mean().item())
    activation_scale_std = float(
        activation_log_scale[train_tile_mask].std(unbiased=False).clamp_min(1.0e-6).item()
    )
    standardized_activation_log_scale = (
        activation_log_scale - activation_scale_mean
    ) / activation_scale_std
    quant_error = dequantized.reshape(64, 9, 128, 128) - weights.reshape(
        64, 9, 128, 128
    )
    context_by_operator = contexts.reshape(64, 9, 512, 128)
    target_by_operator = weights.reshape(64, 9, 128, 128)
    calibration_action = torch.einsum(
        "btri,btic->btrc",
        context_by_operator[:, :, : cfg.train_activation_rows],
        quant_error,
    ).sum(dim=1)
    heldout_action = torch.einsum(
        "btri,btic->btrc",
        context_by_operator[:, :, cfg.train_activation_rows :],
        quant_error,
    ).sum(dim=1)
    calibration_target_action = torch.einsum(
        "btri,btic->btrc",
        context_by_operator[:, :, : cfg.train_activation_rows],
        target_by_operator,
    ).sum(dim=1)
    heldout_target_action = torch.einsum(
        "btri,btic->btrc",
        context_by_operator[:, :, cfg.train_activation_rows :],
        target_by_operator,
    ).sum(dim=1)
    quantizer_metrics = {
        "scale_log2_train_mean": scale_mean,
        "scale_log2_train_std": scale_std,
        "activation_sqrt_scale_log2_train_mean": activation_scale_mean,
        "activation_sqrt_scale_log2_train_std": activation_scale_std,
        "code_min": int(codes.min().item()),
        "code_max": int(codes.max().item()),
        "code_unique": [int(value) for value in torch.unique(codes).tolist()],
        "raw_nrmse": float(
            quant_error.square().sum().div(weights.square().sum()).sqrt().item()
        ),
        "calibration_action_nrmse": float(
            calibration_action.square().sum()
            .div(calibration_target_action.square().sum())
            .sqrt()
            .item()
        ),
        "heldout_action_nrmse": float(
            heldout_action.square().sum()
            .div(heldout_target_action.square().sum())
            .sqrt()
            .item()
        ),
    }
    return {
        **data,
        "flat_weights": weights,
        "flat_contexts": contexts,
        "flat_tile_rows": tile_rows,
        "raw_float_tokens": weight_rows.reshape(
            -1, 128, chunks_per_group, cfg.values_per_token
        ).reshape(
            -1, 128 * chunks_per_group, cfg.values_per_token
        ),
        "normalized_tokens": normalized.reshape(
            -1, 128, chunks_per_group, cfg.values_per_token
        ).reshape(-1, 128 * chunks_per_group, cfg.values_per_token),
        "activation_sqrt_tokens": activation_normalized.reshape(
            -1, 128, chunks_per_group, cfg.values_per_token
        ).reshape(
            -1, 128 * chunks_per_group, cfg.values_per_token
        ),
        "gptq_codes": codes.reshape(
            -1, 128, chunks_per_group, cfg.values_per_token
        ).reshape(-1, 128 * chunks_per_group, cfg.values_per_token),
        "standardized_log_scale": standardized_log_scale[:, :, None]
        .expand(-1, -1, chunks_per_group)
        .reshape(-1, 128 * chunks_per_group, 1),
        "standardized_activation_log_scale": standardized_activation_log_scale[:, :, None]
        .expand(-1, -1, chunks_per_group)
        .reshape(-1, 128 * chunks_per_group, 1),
        "quantizer_metrics": quantizer_metrics,
    }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class PreNormTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        depth_scale: float,
        *,
        context_dim: int | None = None,
        bounded_cosine_attention: bool = False,
        attention_logit_scale: float = 2.0,
        bounded_swiglu_hidden: bool = False,
        bounded_residual_writes: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.bounded_cosine_attention = bool(bounded_cosine_attention)
        self.attention_logit_scale = float(attention_logit_scale)
        self.bounded_swiglu_hidden = bool(bounded_swiglu_hidden)
        self.bounded_residual_writes = bool(bounded_residual_writes)
        self.residual_write_rms_cap = float(depth_scale)
        if self.bounded_cosine_attention and self.attention_logit_scale <= 0.0:
            raise ValueError("attention_logit_scale must be positive")
        if self.bounded_residual_writes and self.residual_write_rms_cap <= 0.0:
            raise ValueError("residual write cap must be positive")
        self.attn_norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.context_key_projection = (
            nn.Linear(context_dim, dim, bias=False)
            if context_dim is not None
            else None
        )
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim)
        self.mlp_in = nn.Linear(dim, 2 * mlp_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_dim, dim, bias=False)
        self._reset_parameters(depth_scale)

    def _bounded_write(self, write: torch.Tensor) -> torch.Tensor:
        if not self.bounded_residual_writes:
            return write
        rms_sq = write.float().square().mean(dim=-1, keepdim=True)
        cap_sq = self.residual_write_rms_cap**2
        multiplier = torch.rsqrt(1.0 + rms_sq / cap_sq)
        return write * multiplier.to(dtype=write.dtype)

    def _reset_parameters(self, depth_scale: float) -> None:
        nn.init.normal_(self.qkv.weight, std=0.02)
        if self.context_key_projection is not None:
            nn.init.normal_(
                self.context_key_projection.weight,
                std=0.25 / math.sqrt(self.context_key_projection.in_features),
            )
        nn.init.normal_(self.attn_out.weight, std=0.02 * depth_scale)
        nn.init.normal_(self.mlp_in.weight, std=0.02)
        nn.init.normal_(self.mlp_out.weight, std=0.02 * depth_scale)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_context: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        if valid_mask is not None:
            if tuple(valid_mask.shape) != (batch, length):
                raise ValueError(
                    f"valid_mask must be {(batch, length)}, got {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
        qkv = self.qkv(self.attn_norm(x)).view(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        if self.context_key_projection is not None:
            if key_context is None or key_context.shape[:2] != x.shape[:2]:
                raise ValueError("conditioned block requires aligned key context")
            normalized_context = F.rms_norm(
                key_context.float(),
                (key_context.shape[-1],),
                weight=None,
                eps=1.0e-6,
            )
            context_key = self.context_key_projection(normalized_context).view(
                batch, length, self.heads, self.head_dim
            )
            key = key + context_key.to(key.dtype)
        elif key_context is not None:
            raise ValueError("unconditioned block received key context")
        query_heads = query.transpose(1, 2)
        key_heads = key.transpose(1, 2)
        if self.bounded_cosine_attention:
            query_heads = F.normalize(query_heads.float(), dim=-1, eps=1.0e-6).to(
                dtype=query.dtype
            )
            key_heads = F.normalize(key_heads.float(), dim=-1, eps=1.0e-6).to(
                dtype=key.dtype
            )
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value.transpose(1, 2),
            attn_mask=(valid_mask[:, None, None, :] if valid_mask is not None else None),
            dropout_p=0.0,
            scale=(self.attention_logit_scale if self.bounded_cosine_attention else None),
        )
        attention_write = self.attn_out(
            attended.transpose(1, 2).reshape(batch, length, dim)
        )
        x = x + self._bounded_write(attention_write)
        left, right = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        hidden = F.silu(left) * right
        if self.bounded_swiglu_hidden:
            hidden_rms_sq = hidden.float().square().mean(dim=-1, keepdim=True)
            hidden = hidden * torch.rsqrt(1.0 + hidden_rms_sq).to(dtype=hidden.dtype)
        x = x + self._bounded_write(self.mlp_out(hidden))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class UnifiedWeightBottleneck(nn.Module):
    def __init__(self, cfg: ExperimentConfig, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown representation arm {arm!r}")
        if 128 % cfg.values_per_token != 0:
            raise ValueError("values_per_token must divide the fixed 128-wide tile")
        self.cfg = cfg
        self.arm = arm
        self.chunks_per_group = 128 // cfg.values_per_token
        self.weight_token_count = 128 * self.chunks_per_group
        dim = cfg.hidden_dim
        context_dim = cfg.distribution_d_dist if cfg.use_distribution_conditioning else None
        depth_scale = 1.0 / math.sqrt(2.0 * max(cfg.encoder_depth, cfg.decoder_depth))

        self.latent_slots = nn.Parameter(torch.empty(cfg.latent_slots, dim))
        self.encoder_blocks = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    dim,
                    cfg.heads,
                    cfg.mlp_dim,
                    depth_scale,
                    bounded_cosine_attention=cfg.bounded_cosine_attention,
                    attention_logit_scale=cfg.attention_logit_scale,
                    bounded_swiglu_hidden=cfg.bounded_swiglu_hidden,
                    bounded_residual_writes=cfg.bounded_residual_writes,
                )
                if context_dim is None
                else PreNormTransformerBlock(
                    dim,
                    cfg.heads,
                    cfg.mlp_dim,
                    depth_scale,
                    context_dim=context_dim,
                    bounded_cosine_attention=cfg.bounded_cosine_attention,
                    attention_logit_scale=cfg.attention_logit_scale,
                    bounded_swiglu_hidden=cfg.bounded_swiglu_hidden,
                    bounded_residual_writes=cfg.bounded_residual_writes,
                )
                for _ in range(cfg.encoder_depth)
            ]
        )
        self.latent_norm = RMSNorm(dim)
        self.to_latent = nn.Linear(dim, cfg.latent_dim, bias=False)
        self.from_latent = nn.Linear(cfg.latent_dim, dim, bias=False)
        self.output_queries = nn.Parameter(torch.empty(self.weight_token_count, dim))
        self.decoder_blocks = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    dim,
                    cfg.heads,
                    cfg.mlp_dim,
                    depth_scale,
                    bounded_cosine_attention=cfg.bounded_cosine_attention,
                    attention_logit_scale=cfg.attention_logit_scale,
                    bounded_swiglu_hidden=cfg.bounded_swiglu_hidden,
                    bounded_residual_writes=cfg.bounded_residual_writes,
                )
                for _ in range(cfg.decoder_depth)
            ]
        )
        self.distribution_encoder = (
            InputDistributionEncodingModule(
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
            if cfg.use_distribution_conditioning
            else None
        )
        self.decoder_query_conditioner = (
            nn.Sequential(
                nn.Linear(dim + cfg.distribution_d_dist, 2 * dim),
                nn.GELU(),
                nn.Linear(2 * dim, dim),
            )
            if cfg.use_distribution_conditioning
            else None
        )
        self.output_norm = RMSNorm(dim)
        self.output_head = nn.Linear(dim, cfg.values_per_token, bias=False)
        self.group_embedding = nn.Embedding(128, dim)
        self.chunk_embedding = nn.Embedding(self.chunks_per_group, dim)
        self.tile_embedding = nn.Embedding(9, dim)
        if cfg.max_tile_rows < 9:
            raise ValueError("max_tile_rows must be at least 9")
        if cfg.max_tile_cols < 1:
            raise ValueError("max_tile_cols must be positive")
        # These production-only coordinates must not shift the seed-42 start
        # of any module shared with the bounded p32 model.
        with torch.random.fork_rng(devices=[]):
            self.extra_tile_row_embedding = (
                nn.Embedding(cfg.max_tile_rows - 9, dim)
                if cfg.max_tile_rows > 9
                else None
            )
            self.tile_col_embedding = (
                nn.Embedding(cfg.max_tile_cols, dim)
                if cfg.max_tile_cols > 1
                else None
            )
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.continuous_projection = (
            nn.Linear(cfg.values_per_token, dim, bias=False)
            if arm != "gptq_token"
            else None
        )
        self.code_embedding = (
            nn.Embedding(2**cfg.quant_bits - 1, dim)
            if arm == "gptq_token"
            else None
        )
        self.code_position_gate = (
            nn.Parameter(torch.empty(cfg.values_per_token, dim))
            if arm == "gptq_token"
            else None
        )
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        nn.init.normal_(self.latent_slots, std=0.02)
        nn.init.normal_(self.output_queries, std=0.02)
        nn.init.normal_(self.to_latent.weight, std=0.02)
        nn.init.normal_(self.from_latent.weight, std=0.02)
        # RMSNorm makes the decoder features unit scale.  A plain 0.02 head
        # would therefore emit weights around 0.45 RMS at width 512, while the
        # exact64 target is around 0.03.  Fan-in scaling gives a live, nonzero
        # path without making the initial reconstruction orders too large.
        nn.init.normal_(self.output_head.weight, std=0.02 / math.sqrt(self.cfg.hidden_dim))
        nn.init.normal_(self.group_embedding.weight, std=0.02)
        nn.init.normal_(self.chunk_embedding.weight, std=0.02)
        nn.init.normal_(self.tile_embedding.weight, std=0.02)
        for module in self.scale_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.continuous_projection is not None:
            nn.init.normal_(self.continuous_projection.weight, std=0.02)
        if self.code_embedding is not None:
            nn.init.normal_(self.code_embedding.weight, std=0.02)
        if self.code_position_gate is not None:
            nn.init.normal_(self.code_position_gate, mean=1.0, std=0.02)
        with torch.random.fork_rng(devices=[]):
            if self.extra_tile_row_embedding is not None:
                nn.init.normal_(self.extra_tile_row_embedding.weight, std=0.02)
            if self.tile_col_embedding is not None:
                # Column zero is the exact bounded-model path. Starting this
                # production-only lane at zero preserves that path at init.
                nn.init.zeros_(self.tile_col_embedding.weight)

    def _content_embedding(self, content: torch.Tensor) -> torch.Tensor:
        if self.arm == "gptq_token":
            if content.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("categorical GPTQ arm requires integer code tensors")
            indices = content.long() + (2 ** (self.cfg.quant_bits - 1) - 1)
            if int(indices.min()) < 0 or int(indices.max()) >= 2**self.cfg.quant_bits - 1:
                raise ValueError("GPTQ token index is outside the symmetric vocabulary")
            embedded = self.code_embedding(indices)
            return (embedded * self.code_position_gate[None, None]).sum(dim=2) / math.sqrt(
                self.cfg.values_per_token
            )
        return self.continuous_projection(content.float())

    def _tile_position_embedding(
        self,
        tile_row: torch.Tensor,
        tile_col: torch.Tensor | None,
    ) -> torch.Tensor:
        if tile_row.ndim != 1:
            raise ValueError("tile_row must be rank-1 [B]")
        tile_row = tile_row.to(dtype=torch.long)
        if int(tile_row.min()) < 0 or int(tile_row.max()) >= self.cfg.max_tile_rows:
            raise ValueError("tile_row is outside the configured production range")
        position = self.tile_embedding(tile_row.clamp(max=8))
        if self.extra_tile_row_embedding is not None:
            extra = tile_row >= 9
            if bool(extra.any()):
                position = position.clone()
                position[extra] = self.extra_tile_row_embedding(tile_row[extra] - 9)
        if tile_col is not None:
            if tile_col.ndim != 1 or tuple(tile_col.shape) != tuple(tile_row.shape):
                raise ValueError("tile_col must match tile_row shape")
            tile_col = tile_col.to(dtype=torch.long)
            if int(tile_col.min()) < 0 or int(tile_col.max()) >= self.cfg.max_tile_cols:
                raise ValueError("tile_col is outside the configured production range")
            if self.tile_col_embedding is not None:
                position = position + self.tile_col_embedding(tile_col)
            elif bool((tile_col != 0).any()):
                raise ValueError("bounded model supports only tile_col=0")
        return position

    def encode(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_index: torch.Tensor,
        *,
        tile_col: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        dist_patch_by_patch: torch.Tensor | None = None,
        capture_depth: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        batch = content.shape[0]
        if content.shape[1:] != (self.weight_token_count, self.cfg.values_per_token):
            raise ValueError(
                "content must match the configured token layout: "
                f"expected [B,{self.weight_token_count},{self.cfg.values_per_token}], "
                f"got {tuple(content.shape)}"
            )
        group_ids = torch.arange(128, device=content.device).repeat_interleave(
            self.chunks_per_group
        )
        chunk_ids = torch.arange(self.chunks_per_group, device=content.device).repeat(128)
        embedded = self._content_embedding(content)
        embedded = (
            embedded
            + self.scale_mlp(log_scale.float()).to(embedded.dtype)
            + self.group_embedding(group_ids)[None]
            + self.chunk_embedding(chunk_ids)[None]
            + self._tile_position_embedding(tile_index, tile_col)[:, None]
        )
        if token_valid_mask is not None:
            if tuple(token_valid_mask.shape) != (batch, self.weight_token_count):
                raise ValueError("token_valid_mask does not match the weight-token layout")
            token_valid_mask = token_valid_mask.to(device=content.device, dtype=torch.bool)
            embedded = embedded * token_valid_mask.unsqueeze(-1).to(dtype=embedded.dtype)
        latent = self.latent_slots[None].expand(batch, -1, -1)
        state = torch.cat((latent, embedded), dim=1)
        state_valid_mask = None
        if token_valid_mask is not None:
            state_valid_mask = torch.cat(
                (
                    torch.ones(
                        batch,
                        self.cfg.latent_slots,
                        device=content.device,
                        dtype=torch.bool,
                    ),
                    token_valid_mask,
                ),
                dim=1,
            )
        key_context: torch.Tensor | None = None
        if self.cfg.use_distribution_conditioning:
            if dist_patch_by_patch is None or tuple(dist_patch_by_patch.shape[1:]) != (
                self.chunks_per_group,
                self.cfg.distribution_d_dist,
            ):
                raise ValueError(
                    "distribution conditioning requires "
                    f"[B,{self.chunks_per_group},d_dist]"
                )
            content_context = dist_patch_by_patch.unsqueeze(1).expand(
                -1, 128, -1, -1
            ).reshape(
                batch, self.weight_token_count, self.cfg.distribution_d_dist
            )
            latent_context = content_context.new_zeros(
                batch, self.cfg.latent_slots, self.cfg.distribution_d_dist
            )
            key_context = torch.cat((latent_context, content_context), dim=1)
        elif dist_patch_by_patch is not None:
            raise ValueError("unconditioned model received distribution context")
        telemetry: list[dict[str, float]] = []
        for depth, block in enumerate(self.encoder_blocks, start=1):
            if self.cfg.activation_checkpointing and self.training:
                if key_context is None:
                    state = checkpoint(
                        lambda current, current_block=block: current_block(
                            current, valid_mask=state_valid_mask
                        ),
                        state,
                        use_reentrant=False,
                    )
                else:
                    state = checkpoint(
                        lambda current, context, current_block=block: current_block(
                            current,
                            key_context=context,
                            valid_mask=state_valid_mask,
                        ),
                        state,
                        key_context,
                        use_reentrant=False,
                    )
            else:
                state = block(
                    state,
                    key_context=key_context,
                    valid_mask=state_valid_mask,
                )
            if capture_depth:
                telemetry.append(
                    {
                        "depth": depth,
                        "latent_rms": float(
                            state[:, : self.cfg.latent_slots].float().square().mean().sqrt().item()
                        ),
                        "content_rms": float(
                            state[:, self.cfg.latent_slots :].float().square().mean().sqrt().item()
                        ),
                    }
                )
        z = self.to_latent(self.latent_norm(state[:, : self.cfg.latent_slots]))
        return z, telemetry

    def decode(
        self,
        z: torch.Tensor,
        tile_index: torch.Tensor,
        *,
        tile_col: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        dist_patch_by_patch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.from_latent(z)
        queries = self.output_queries[None].expand(z.shape[0], -1, -1)
        queries = queries + self._tile_position_embedding(tile_index, tile_col)[:, None]
        if token_valid_mask is not None:
            if tuple(token_valid_mask.shape) != (z.shape[0], self.weight_token_count):
                raise ValueError("token_valid_mask does not match decoder queries")
            token_valid_mask = token_valid_mask.to(device=z.device, dtype=torch.bool)
            queries = queries * token_valid_mask.unsqueeze(-1).to(dtype=queries.dtype)
        if self.decoder_query_conditioner is not None:
            if dist_patch_by_patch is None:
                raise ValueError("conditioned decoder requires distribution context")
            decoder_context = dist_patch_by_patch.unsqueeze(1).expand(
                -1, 128, -1, -1
            ).reshape(z.shape[0], self.weight_token_count, self.cfg.distribution_d_dist)
            queries = self.decoder_query_conditioner(
                torch.cat((queries, decoder_context.to(queries.dtype)), dim=-1)
            )
        elif dist_patch_by_patch is not None:
            raise ValueError("unconditioned decoder received distribution context")
        state = torch.cat((latent, queries), dim=1)
        state_valid_mask = None
        if token_valid_mask is not None:
            state_valid_mask = torch.cat(
                (
                    torch.ones(
                        z.shape[0],
                        self.cfg.latent_slots,
                        device=z.device,
                        dtype=torch.bool,
                    ),
                    token_valid_mask,
                ),
                dim=1,
            )
        for block in self.decoder_blocks:
            if self.cfg.activation_checkpointing and self.training:
                state = checkpoint(
                    lambda current, current_block=block: current_block(
                        current, valid_mask=state_valid_mask
                    ),
                    state,
                    use_reentrant=False,
                )
            else:
                state = block(state, valid_mask=state_valid_mask)
        chunks = self.output_head(self.output_norm(state[:, self.cfg.latent_slots :]))
        return (
            chunks.view(
                -1,
                128,
                self.chunks_per_group,
                self.cfg.values_per_token,
            )
            .reshape(-1, 128, 128)
            .transpose(1, 2)
        )

    def forward(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_index: torch.Tensor,
        activation_context: torch.Tensor | None = None,
        *,
        tile_col: torch.Tensor | None = None,
        activation_sample_mask: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        capture_depth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
        dist_patch_by_patch = self._encode_distribution_context(
            activation_context,
            sample_mask=activation_sample_mask,
        )
        z, telemetry = self.encode(
            content,
            log_scale,
            tile_index,
            tile_col=tile_col,
            token_valid_mask=token_valid_mask,
            dist_patch_by_patch=dist_patch_by_patch,
            capture_depth=capture_depth,
        )
        return (
            self.decode(
                z,
                tile_index,
                tile_col=tile_col,
                token_valid_mask=token_valid_mask,
                dist_patch_by_patch=dist_patch_by_patch,
            ),
            z,
            telemetry,
        )

    def _encode_distribution_context(
        self,
        activation_context: torch.Tensor | None,
        *,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.distribution_encoder is None:
            if activation_context is not None:
                raise ValueError("unconditioned model received activation context")
            return None
        if activation_context is None or activation_context.ndim != 3:
            raise ValueError("conditioned model requires activation context [B,n,128]")
        batch, sample_count, d_in = activation_context.shape
        if d_in != 128:
            raise ValueError("this exact comparison requires d_in=128 activation tiles")
        patch_indices = torch.arange(
            d_in,
            device=activation_context.device,
            dtype=torch.long,
        ).view(1, self.chunks_per_group, self.cfg.values_per_token).expand(
            batch, -1, -1
        )
        repeated_context = activation_context.unsqueeze(1).expand(
            batch, self.chunks_per_group, sample_count, d_in
        ).reshape(batch * self.chunks_per_group, sample_count, d_in)
        repeated_sample_mask = None
        if sample_mask is not None:
            if tuple(sample_mask.shape) != (batch, sample_count):
                raise ValueError("sample_mask does not match activation context")
            repeated_sample_mask = sample_mask.unsqueeze(1).expand(
                batch, self.chunks_per_group, sample_count
            ).reshape(batch * self.chunks_per_group, sample_count)
        _dist_var, dist_patch = self.distribution_encoder(
            repeated_context,
            patch_indices.reshape(
                batch * self.chunks_per_group, self.cfg.values_per_token
            ),
            sample_mask=repeated_sample_mask,
        )
        return dist_patch.view(
            batch, self.chunks_per_group, self.cfg.distribution_d_dist
        )


def _arm_content(prepared: dict[str, Any], arm: str, indices: torch.Tensor) -> torch.Tensor:
    if arm == "raw_float":
        return prepared["raw_float_tokens"].index_select(0, indices)
    if arm == "normalized_float":
        return prepared["normalized_tokens"].index_select(0, indices)
    if arm == "activation_sqrt":
        return prepared["activation_sqrt_tokens"].index_select(0, indices)
    codes = prepared["gptq_codes"].index_select(0, indices)
    return codes if arm == "gptq_token" else codes.float()


def _arm_log_scale(prepared: dict[str, Any], arm: str, indices: torch.Tensor) -> torch.Tensor:
    if arm == "raw_float":
        reference = prepared["standardized_log_scale"].index_select(0, indices)
        return torch.zeros_like(reference)
    key = (
        "standardized_activation_log_scale"
        if arm == "activation_sqrt"
        else "standardized_log_scale"
    )
    return prepared[key].index_select(0, indices)


def _full_operator_action_loss(
    prediction_tiles: torch.Tensor,
    target_tiles: torch.Tensor,
    context_tiles: torch.Tensor,
) -> torch.Tensor:
    if prediction_tiles.ndim != 4 or prediction_tiles.shape[1:] != (9, 128, 128):
        raise ValueError("action loss requires complete [B,9,128,128] operators")
    predicted_action = torch.einsum(
        "btri,btic->btrc", context_tiles, prediction_tiles
    ).sum(dim=1)
    target_action = torch.einsum(
        "btri,btic->btrc", context_tiles, target_tiles
    ).sum(dim=1)
    numerator = (predicted_action - target_action).square().sum()
    denominator = target_action.square().sum().clamp_min(1.0e-12)
    return numerator / denominator


def _big_vae_structural_loss(
    prediction_tiles: torch.Tensor,
    target_tiles: torch.Tensor,
    cfg: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction_tiles.shape != target_tiles.shape or prediction_tiles.shape[-2:] != (
        128,
        128,
    ):
        raise ValueError("Big-VAE structural loss requires matched [N,128,128] tiles")
    return BigWeightVAELossMixin.patch_structure_loss(
        target_tiles,
        prediction_tiles,
        patch_size=cfg.structural_patch_size,
        gamma=cfg.structural_gamma,
        lambda_dir=cfg.structural_direction_weight,
        lambda_scale=cfg.structural_scale_weight,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=cfg.structural_huber_delta,
    )


def _gradient_telemetry(model: UnifiedWeightBottleneck) -> dict[str, Any]:
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            groups["none"].append(parameter)
            continue
        if name.startswith("encoder_blocks."):
            block = name.split(".")[1]
            groups[f"encoder_block_{int(block) + 1}"].append(parameter.grad.detach())
        elif name.startswith("decoder_blocks."):
            block = name.split(".")[1]
            groups[f"decoder_block_{int(block) + 1}"].append(parameter.grad.detach())
        elif name.startswith("direction_tail."):
            groups["direction_tail"].append(parameter.grad.detach())
        elif name.startswith("scale_tail."):
            groups["scale_tail"].append(parameter.grad.detach())
        elif name.startswith(("direction_head.", "direction_output_norm.")):
            groups["direction_head"].append(parameter.grad.detach())
        elif name.startswith(("scale_head.", "scale_output_norm.")):
            groups["scale_head"].append(parameter.grad.detach())
        elif name.startswith("distribution_encoder."):
            groups["distribution_encoder"].append(parameter.grad.detach())
        elif name.startswith("decoder_query_conditioner."):
            groups["decoder_query_conditioner"].append(parameter.grad.detach())
        elif name.startswith(("to_latent", "from_latent")):
            groups["bottleneck_projections"].append(parameter.grad.detach())
        elif name.startswith("output_head"):
            groups["output_head"].append(parameter.grad.detach())
        else:
            groups["input_and_queries"].append(parameter.grad.detach())
    result: dict[str, Any] = {"none_parameter_tensors": len(groups.pop("none", []))}
    for name, tensors in sorted(groups.items()):
        squared = sum(float(t.float().square().sum().item()) for t in tensors)
        count = sum(int(t.numel()) for t in tensors)
        result[name] = {
            "gradient_rms": math.sqrt(squared / max(count, 1)),
            "parameter_tensors": len(tensors),
            "numel": count,
        }
    return result


@torch.no_grad()
def _evaluate_arm(
    model: UnifiedWeightBottleneck,
    prepared: dict[str, Any],
    operator_indices: list[int],
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = defaultdict(float)
    operator_nrmse: list[float] = []
    depth_rows: list[dict[str, float]] | None = None
    latent_rows: list[torch.Tensor] = []
    for start in range(0, len(operator_indices), cfg.eval_batch_size):
        operator_batch = operator_indices[start : start + cfg.eval_batch_size]
        cpu_indices = torch.tensor(
            [operator * 9 + tile for operator in operator_batch for tile in range(9)],
            dtype=torch.long,
        )
        operator_batch_size = len(operator_batch)
        content = _arm_content(prepared, model.arm, cpu_indices).to(device)
        log_scale = _arm_log_scale(prepared, model.arm, cpu_indices).to(device)
        tile_row = prepared["flat_tile_rows"].index_select(0, cpu_indices).to(device)
        activation_context = (
            prepared["flat_contexts"].index_select(0, cpu_indices).to(device)
            if cfg.use_distribution_conditioning
            else None
        )
        target = prepared["flat_weights"].index_select(0, cpu_indices).to(device).view(
            operator_batch_size, 9, 128, 128
        )
        context = prepared["flat_contexts"].index_select(0, cpu_indices)[
            :, cfg.train_activation_rows :
        ].to(device).view(operator_batch_size, 9, cfg.test_activation_rows, 128)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction_tiles, latent, depth = model(
                content,
                log_scale,
                tile_row,
                activation_context,
                capture_depth=depth_rows is None,
            )
        prediction = prediction_tiles.float().view(operator_batch_size, 9, 128, 128)
        structural_loss, structural_details = _big_vae_structural_loss(
            prediction.reshape(-1, 128, 128),
            target.float().reshape(-1, 128, 128),
            cfg,
        )
        structural_tiles = operator_batch_size * 9
        totals["structural_loss_sum"] += float(structural_loss.item()) * structural_tiles
        totals["structural_dir_sum"] += (
            float(structural_details["L_dir"].item()) * structural_tiles
        )
        totals["structural_scale_sum"] += (
            float(structural_details["L_scale"].item()) * structural_tiles
        )
        totals["structural_count"] += structural_tiles
        predicted_action = torch.einsum(
            "btri,btic->btrc", context.float(), prediction
        ).sum(dim=1)
        target_action = torch.einsum(
            "btri,btic->btrc", context.float(), target.float()
        ).sum(dim=1)
        action_num = (predicted_action - target_action).square().sum(dim=(1, 2))
        action_den = target_action.square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        raw_num = (prediction - target.float()).square().sum(dim=(1, 2, 3))
        raw_den = target.float().square().sum(dim=(1, 2, 3)).clamp_min(1.0e-12)
        cosine = F.cosine_similarity(prediction.flatten(1), target.float().flatten(1), dim=1)
        pred_rms = prediction.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1.0e-12)
        target_rms = target.float().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1.0e-12)
        totals["action_num"] += float(action_num.sum().item())
        totals["action_den"] += float(action_den.sum().item())
        totals["raw_num"] += float(raw_num.sum().item())
        totals["raw_den"] += float(raw_den.sum().item())
        totals["cosine_sum"] += float(cosine.sum().item())
        totals["log_scale_abs_sum"] += float(
            (torch.log2(pred_rms) - torch.log2(target_rms)).abs().sum().item()
        )
        totals["count"] += operator_batch_size
        operator_nrmse.extend(
            torch.sqrt(action_num / action_den).detach().cpu().tolist()
        )
        if depth_rows is None:
            depth_rows = depth
        latent_rows.append(latent.float().cpu())
    if len(operator_nrmse) != len(operator_indices):
        raise RuntimeError("operator-level evaluation did not preserve the complete split")
    latent_all = torch.cat(latent_rows, dim=0)
    centered = latent_all.flatten(1) - latent_all.flatten(1).mean(dim=0, keepdim=True)
    gram_eigenvalues = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0.0)
    participation = float(
        gram_eigenvalues.sum().square().div(
            gram_eigenvalues.square().sum().clamp_min(1.0e-20)
        ).item()
    )
    return {
        "structural_loss": totals["structural_loss_sum"] / totals["structural_count"],
        "structural_direction_loss": totals["structural_dir_sum"]
        / totals["structural_count"],
        "structural_scale_loss": totals["structural_scale_sum"]
        / totals["structural_count"],
        "action_nrmse": math.sqrt(totals["action_num"] / totals["action_den"]),
        "raw_nrmse": math.sqrt(totals["raw_num"] / totals["raw_den"]),
        "mean_cosine": totals["cosine_sum"] / totals["count"],
        "mean_abs_log2_rms_error": totals["log_scale_abs_sum"] / totals["count"],
        "operator_action_nrmse_mean": float(np.mean(operator_nrmse)),
        "operator_action_nrmse_median": float(np.median(operator_nrmse)),
        "operator_action_nrmse_p95": float(np.quantile(operator_nrmse, 0.95)),
        "operator_action_nrmse_values": operator_nrmse,
        "latent_rms": float(latent_all.square().mean().sqrt().item()),
        "latent_participation_rank": participation,
        "encoder_depth_state": depth_rows or [],
    }


def _plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plotted_arms = tuple(dict.fromkeys(row["arm"] for row in rows))
    for arm in plotted_arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        steps = [row["step"] for row in arm_rows]
        axes[0].plot(
            steps,
            [row["train"]["structural_loss"] for row in arm_rows],
            marker="o",
            label=arm,
        )
        axes[1].plot(
            steps,
            [row["train"]["structural_direction_loss"] for row in arm_rows],
            marker="o",
            label=arm,
        )
        axes[2].plot(
            steps,
            [
                0.1 * row["train"]["structural_scale_loss"]
                for row in arm_rows
            ],
            marker="o",
            label=arm,
        )
    axes[0].set_title("Full-train Big-VAE structural loss")
    axes[1].set_title("Full-train direction term")
    axes[2].set_title("Full-train weighted scale term (0.1 x L_scale)")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("lower is better")
    axes[1].set_ylabel("lower is better")
    axes[2].set_ylabel("lower is better")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_train_loss_curves(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    plotted_arms = tuple(dict.fromkeys(row["arm"] for row in rows))
    for arm in plotted_arms:
        arm_rows = sorted(
            (row for row in rows if row["arm"] == arm),
            key=lambda row: row["step"],
        )
        steps = [row["step"] for row in arm_rows]
        losses = [row["train"]["structural_loss"] for row in arm_rows]
        axes[0].plot(steps, losses, marker="o", markersize=3.5, label=arm)
        axes[1].plot(steps, losses, marker="o", markersize=3.5, label=arm)
    axes[0].set_title("Full-train Big-VAE structural loss")
    axes[1].set_title("Same curves, logarithmic loss axis")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("L_direction + 0.1 x L_log-scale")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Input representations; identical unified bottleneck and training stream")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run(args: argparse.Namespace) -> None:
    run_arms = tuple(arm.strip() for arm in str(args.arms).split(",") if arm.strip())
    if not run_arms:
        raise ValueError("--arms must select at least one representation")
    if len(set(run_arms)) != len(run_arms):
        raise ValueError(f"--arms contains duplicates: {run_arms}")
    unknown_arms = sorted(set(run_arms) - set(ARMS))
    if unknown_arms:
        raise ValueError(f"unknown --arms values: {unknown_arms}")
    cfg = ExperimentConfig(
        steps=4 if args.smoke else int(args.steps),
        batch_size=2 if args.smoke else int(args.batch_size),
        eval_every=2 if args.smoke else int(args.eval_every),
        hidden_dim=128 if args.smoke else 512,
        heads=4 if args.smoke else 8,
        mlp_dim=256 if args.smoke else 2048,
        latent_slots=4 if args.smoke else 32,
        latent_dim=32 if args.smoke else 384,
        use_distribution_conditioning=bool(args.distribution_conditioning),
    )
    if cfg.steps <= 0 or cfg.batch_size <= 0 or cfg.eval_every <= 0:
        raise ValueError("steps, batch size, and eval interval must be positive")
    executed_steps = cfg.steps if args.stop_after is None else int(args.stop_after)
    if executed_steps <= 0 or executed_steps > cfg.steps:
        raise ValueError("--stop-after must be in [1, --steps]")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise RuntimeError("the full comparison requires CUDA")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root already exists: {output_root}")
    startup = {
        "schema": SCHEMA,
        "config": asdict(cfg),
        "arms": list(run_arms),
        "device": str(device),
        "dtype": "bfloat16 autocast; fp32 loss and GPTQ",
        "seed": cfg.seed,
        "executed_steps": executed_steps,
        "selection": str(args.selection.expanduser().resolve()),
        "resolved_config": str(args.resolved_config.expanduser().resolve()),
        "output_root": str(output_root),
        "cache_mode": "direct immutable mmap read; no shared preprocessing cache",
        "bottleneck": f"[{cfg.latent_slots},{cfg.latent_dim}] continuous deterministic",
        "objective": (
            "Big-VAE patch structural loss: L_direction + 0.1*L_log_scale; "
            "patch_size=16 gamma=0.5 huber_delta=0.1; reconstruction, relative, "
            "behavioral, KL, and V11-specific complement terms disabled"
        ),
        "representation_contract": {
            "scale": "one symmetric maxabs scale per tile/output-channel/128 weights",
            "raw_float": "raw signed 16-value chunks; no normalization and zero scale token",
            "token": "16 ordered values plus repeated standardized log2 group scale",
            "gptq": "4-bit symmetric assignment with activation-Hessian error feedback",
            "activation_sqrt": "continuous normalized chunks from A_train^(1/2) @ W",
            "distribution_conditioning": (
                "production InputDistributionEncodingModule [B,8,256]; added to "
                "encoder content keys at every block and concatenated into decoder "
                "query construction; never added to content or output values"
                if cfg.use_distribution_conditioning
                else "disabled"
            ),
        },
    }
    print("[gptq-bottleneck] stage=preflight", flush=True)
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[gptq-bottleneck] stage=dry-run-complete no_files_written=true", flush=True)
        return
    output_root.mkdir(parents=True)
    _atomic_json(output_root / "config.json", startup)
    started = time.monotonic()
    _seed_everything(cfg.seed)
    data = _load_exact64_tiles(args.selection.resolve(), args.resolved_config.resolve())
    prepared = _prepare_representations(data, cfg, device)
    _atomic_json(
        output_root / "data_contract.json",
        {
            "selection_sha256": prepared["selection_sha256"],
            "resolved_config_sha256": prepared["resolved_config_sha256"],
            "pair_manifest": prepared["pair_manifest"],
            "identities": prepared["identities"],
            "train_operator_indices": prepared["train_operator_indices"],
            "heldout_operator_indices": prepared["heldout_operator_indices"],
            "quantizer_metrics": prepared["quantizer_metrics"],
        },
    )
    print(
        f"[gptq-bottleneck] stage=model-build hidden={cfg.hidden_dim} "
        f"encoder_depth={cfg.encoder_depth} decoder_depth={cfg.decoder_depth}",
        flush=True,
    )
    models: dict[str, UnifiedWeightBottleneck] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    shared_state_hashes: dict[str, str] = {}
    for arm in run_arms:
        _seed_everything(cfg.seed)
        model = UnifiedWeightBottleneck(cfg, arm).to(device)
        models[arm] = model
    reference_arm = "normalized_float" if "normalized_float" in models else run_arms[0]
    reference_state = models[reference_arm].state_dict()
    arm_specific_prefixes = ("continuous_projection", "code_embedding", "code_position_gate")
    for arm in run_arms:
        if arm == reference_arm:
            continue
        current_state = models[arm].state_dict()
        for name, tensor in reference_state.items():
            if name.startswith(arm_specific_prefixes):
                continue
            if name in current_state and current_state[name].shape == tensor.shape:
                current_state[name].copy_(tensor)
        models[arm].load_state_dict(current_state)
    for arm, model in models.items():
        optimizers[arm] = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=cfg.weight_decay,
        )
        shared_digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            if name.startswith(arm_specific_prefixes):
                continue
            shared_digest.update(name.encode())
            shared_digest.update(tensor.detach().cpu().numpy().tobytes())
        shared_state_hashes[arm] = shared_digest.hexdigest()
    if len(set(shared_state_hashes.values())) != 1:
        raise RuntimeError(f"shared model starts do not match: {shared_state_hashes}")
    model_parameters = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
    _atomic_json(
        output_root / "model_contract.json",
        {
            "parameters": model_parameters,
            "shared_start_sha256": shared_state_hashes,
            "latent_scalars_per_tile": cfg.latent_slots * cfg.latent_dim,
            "input_scalars_per_tile": 128 * 128,
            "latent_to_input_ratio": cfg.latent_slots * cfg.latent_dim / (128 * 128),
        },
    )

    train_operator_indices = torch.tensor(
        prepared["train_operator_indices"], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(cfg.seed + 17)
    eval_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    metrics_path = output_root / "metrics.jsonl"
    grad_path = output_root / "gradient_telemetry.jsonl"

    def evaluate(step: int) -> None:
        print(f"[gptq-bottleneck] stage=evaluate step={step}/{cfg.steps}", flush=True)
        new_rows: list[dict[str, Any]] = []
        for arm, model in models.items():
            test_metrics = _evaluate_arm(
                model,
                prepared,
                prepared["heldout_operator_indices"],
                cfg,
                device,
            )
            train_metrics = _evaluate_arm(
                model,
                prepared,
                prepared["train_operator_indices"],
                cfg,
                device,
            )
            row = {
                "schema": SCHEMA,
                "step": step,
                "arm": arm,
                "train": train_metrics,
                "test": test_metrics,
                "elapsed_seconds": time.monotonic() - started,
            }
            if not all(
                math.isfinite(float(value))
                for side in (train_metrics, test_metrics)
                for key, value in side.items()
                if isinstance(value, (float, int)) and key != "operator_action_nrmse_values"
            ):
                raise RuntimeError(f"nonfinite evaluation metric for {arm} at step {step}")
            eval_rows.append(row)
            new_rows.append(row)
            print(
                f"[gptq-bottleneck] stage=evaluate step={step} arm={arm} "
                f"train_structural={train_metrics['structural_loss']:.6f} "
                f"train_direction={train_metrics['structural_direction_loss']:.6f} "
                f"train_scale={train_metrics['structural_scale_loss']:.6f} "
                f"diagnostic_train_action_nrmse={train_metrics['action_nrmse']:.6f}",
                flush=True,
            )
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in eval_rows),
            encoding="utf-8",
        )
        _plot_metrics(eval_rows, output_root / "comparison.png")
        _plot_train_loss_curves(
            eval_rows,
            output_root / "train_structural_loss_curves.png",
        )

    evaluate(0)
    print(
        f"[gptq-bottleneck] stage=train configured_steps={cfg.steps} "
        f"executed_steps={executed_steps} arms={len(run_arms)}",
        flush=True,
    )
    for step in range(1, executed_steps + 1):
        warmup_scale = min(1.0, step / max(cfg.warmup_steps, 1))
        decay_progress = max(step - cfg.warmup_steps, 0) / max(
            cfg.steps - cfg.warmup_steps, 1
        )
        cosine_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        current_lr = cfg.learning_rate * warmup_scale * cosine_scale
        for optimizer in optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        sample_positions = torch.randint(
            0,
            train_operator_indices.numel(),
            (cfg.batch_size,),
            generator=generator,
        )
        sampled_operators = train_operator_indices.index_select(0, sample_positions)
        cpu_indices = torch.tensor(
            [
                int(operator) * 9 + tile
                for operator in sampled_operators.tolist()
                for tile in range(9)
            ],
            dtype=torch.long,
        )
        target = prepared["flat_weights"].index_select(0, cpu_indices).to(device).view(
            cfg.batch_size, 9, 128, 128
        )
        tile_row = prepared["flat_tile_rows"].index_select(0, cpu_indices).to(device)
        activation_context = (
            prepared["flat_contexts"].index_select(0, cpu_indices).to(device)
            if cfg.use_distribution_conditioning
            else None
        )
        step_losses: dict[str, float] = {}
        for arm, model in models.items():
            model.train()
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            content = _arm_content(prepared, arm, cpu_indices).to(device)
            log_scale = _arm_log_scale(prepared, arm, cpu_indices).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction_tiles, _latent, _depth = model(
                    content,
                    log_scale,
                    tile_row,
                    activation_context,
                )
            loss, structural_details = _big_vae_structural_loss(
                prediction_tiles.float(),
                target.float().reshape(-1, 128, 128),
                cfg,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite loss for {arm} at step {step}")
            loss.backward()
            if step == 1 or step % cfg.eval_every == 0:
                telemetry = {
                    "schema": SCHEMA,
                    "step": step,
                    "arm": arm,
                    "loss": float(loss.item()),
                    "structural_direction_loss": float(
                        structural_details["L_dir"].item()
                    ),
                    "structural_scale_loss": float(
                        structural_details["L_scale"].item()
                    ),
                    "groups": _gradient_telemetry(model),
                }
                gradient_rows.append(telemetry)
                grad_path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in gradient_rows),
                    encoding="utf-8",
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step_losses[arm] = float(loss.item())
        if step == 1:
            for row in gradient_rows:
                if row["step"] != 1:
                    continue
                groups = row["groups"]
                if groups["none_parameter_tensors"] != 0:
                    raise RuntimeError(f"{row['arm']} has parameters without gradients at step 1")
                for depth in range(1, cfg.encoder_depth + 1):
                    if groups[f"encoder_block_{depth}"]["gradient_rms"] <= 0.0:
                        raise RuntimeError(f"{row['arm']} encoder block {depth} is gradient-dead")
        if step % cfg.log_every == 0 or step == 1:
            rate = step / max(time.monotonic() - started, 1.0e-6)
            losses = " ".join(f"{arm}={value:.6f}" for arm, value in step_losses.items())
            print(
                f"[gptq-bottleneck] stage=train step={step}/{cfg.steps} {losses} "
                f"lr={current_lr:.3e} elapsed={time.monotonic() - started:.1f}s "
                f"rate={rate:.3f}_steps_per_s",
                flush=True,
            )
        if step % cfg.eval_every == 0 or step == executed_steps:
            evaluate(step)

    final_rows_by_arm = {
        arm: next(row for row in reversed(eval_rows) if row["arm"] == arm)
        for arm in run_arms
    }
    final_train_by_arm = {
        arm: row["train"] for arm, row in final_rows_by_arm.items()
    }
    final_test_by_arm = {
        arm: row["test"] for arm, row in final_rows_by_arm.items()
    }
    paired: dict[str, Any] = {}
    comparison_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
    if {"raw_float", "normalized_float"}.issubset(final_test_by_arm):
        comparison_specs.append(
            (
                "normalization_effect_normalized_minus_raw",
                np.array(
                    final_test_by_arm["normalized_float"]["operator_action_nrmse_values"]
                ),
                np.array(final_test_by_arm["raw_float"]["operator_action_nrmse_values"]),
            )
        )
    if {"normalized_float", "gptq_continuous"}.issubset(final_test_by_arm):
        comparison_specs.append(
            (
                "quantization_effect_continuous_minus_float",
                np.array(
                    final_test_by_arm["gptq_continuous"]["operator_action_nrmse_values"]
                ),
                np.array(
                    final_test_by_arm["normalized_float"]["operator_action_nrmse_values"]
                ),
            )
        )
    if {"gptq_continuous", "gptq_token"}.issubset(final_test_by_arm):
        comparison_specs.append(
            (
                "categorical_effect_token_minus_continuous",
                np.array(final_test_by_arm["gptq_token"]["operator_action_nrmse_values"]),
                np.array(
                    final_test_by_arm["gptq_continuous"]["operator_action_nrmse_values"]
                ),
            )
        )
    if {"normalized_float", "activation_sqrt"}.issubset(final_test_by_arm):
        comparison_specs.append(
            (
                "activation_sqrt_effect_minus_float",
                np.array(final_test_by_arm["activation_sqrt"]["operator_action_nrmse_values"]),
                np.array(
                    final_test_by_arm["normalized_float"]["operator_action_nrmse_values"]
                ),
            )
        )
    for name, left, right in comparison_specs:
        delta = left - right
        paired[name] = {
            "mean_delta_nrmse": float(delta.mean()),
            "median_delta_nrmse": float(np.median(delta)),
            "positive_fraction_left_worse": float((delta > 0).mean()),
            "per_operator_delta": delta.tolist(),
        }
    summary = {
        "schema": SCHEMA,
        "complete": True,
        "configured_steps": cfg.steps,
        "executed_steps": executed_steps,
        "elapsed_seconds": time.monotonic() - started,
        "final_train": final_train_by_arm,
        "final_test_diagnostic": final_test_by_arm,
        "diagnostic_action_paired_comparisons": paired,
        "train_structural_ranking": sorted(
            (
                {
                    "arm": arm,
                    "structural_loss": metrics["structural_loss"],
                }
                for arm, metrics in final_train_by_arm.items()
            ),
            key=lambda row: row["structural_loss"],
        ),
        "quantizer_metrics": prepared["quantizer_metrics"],
        "gradient_step1": [row for row in gradient_rows if row["step"] == 1],
        "artifacts": {
            "config": str(output_root / "config.json"),
            "data_contract": str(output_root / "data_contract.json"),
            "model_contract": str(output_root / "model_contract.json"),
            "metrics": str(metrics_path),
            "gradient_telemetry": str(grad_path),
            "plot": str(output_root / "comparison.png"),
        },
    }
    _atomic_json(output_root / "summary.json", summary)
    torch.save(
        {arm: model.state_dict() for arm, model in models.items()},
        output_root / "final_models.pt",
    )
    _atomic_json(
        output_root / "COMPLETE.json",
        {"schema": SCHEMA, "summary_sha256": _sha256(output_root / "summary.json")},
    )
    print(
        f"[gptq-bottleneck] stage=complete output={output_root} "
        + " ".join(
            f"{arm}_train_structural={metrics['structural_loss']:.6f}"
            for arm, metrics in final_train_by_arm.items()
        ),
        flush=True,
    )


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
