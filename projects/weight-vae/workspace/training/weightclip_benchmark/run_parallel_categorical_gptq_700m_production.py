from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import signal
import time
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import yaml

from big_vae.datasets.operator_bank import (
    BalancedOperatorBankMixer,
    operator_bank_data_pipeline,
)
from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.big_vae.data_types import (
    _flatten_loader_batches,
    _identity_sample_collate,
    _offline_loader_worker_init_fn,
    _sample_list_collate,
)
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    AffineFreeRMSNorm,
    _append_jsonl,
    _atomic_json,
    _atomic_torch_save,
    _batch_from_stream,
    _exact_layout_groups,
    _latent_anticollapse_objective,
    _model_config,
    _prepare_normalized_inputs,
    _production_loss,
    _restore_rng_state,
    _rng_state,
    _same_layout_direction_infonce,
)
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    ExperimentConfig,
    PreNormTransformerBlock,
    UnifiedWeightBottleneck,
    _seed_everything,
)


SCHEMA = "weightclip_parallel_categorical_gptq_700m_production_v1"
DEFAULT_CONFIG = Path(
    "conf/weightclip_benchmark/"
    "direct_normalized_scaled_700m_parallel_categorical_gptq_production_500k.yaml"
)
EXPECTED_PARAMETERS = 696_787_328
CODE_VOCAB_SIZE = 15
CODE_ZERO_INDEX = 7
OUTPUT_PATCH_SIZE = 16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the parallel p16 categorical GPTQ Weight-AE on the production bank."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported categorical production schema: {payload.get('schema')!r}")
    frozen_scalars = {
        "steps": 500_000,
        "batch_size": 32,
        "learning_rate": 5.0e-5,
        "weight_decay": 0.01,
        "grad_clip_norm": 5.0,
        "seed": 42,
    }
    for name, expected in frozen_scalars.items():
        if float(payload[name]) != float(expected):
            raise ValueError(f"categorical production requires {name}={expected}")
    if payload["objective"] != "parallel_gptq_code_and_scale_ordinal":
        raise ValueError("categorical production objective drifted")
    if payload["scheduler"] != "constant":
        raise ValueError("categorical production requires constant LR")
    if tuple(float(value) for value in payload["betas"]) != (0.9, 0.999):
        raise ValueError("categorical production requires AdamW betas=(0.9,0.999)")
    if float(payload["eps"]) != 1.0e-8:
        raise ValueError("categorical production requires AdamW eps=1e-8")
    expected_architecture = {
        "input_values_per_token": 32,
        "output_patch_size": 16,
        "shared_decoder_depth": 5,
        "categorical_tail_depth": 1,
        "decoder_contract": "residual_cross_attention_to_latent_only",
        "decoder_activation_context": False,
        "code_vocab_size": 15,
        "scale_vocab_size": 256,
    }
    if payload.get("architecture") != expected_architecture:
        raise ValueError("categorical production architecture contract drifted")
    expected_normalization = {
        "kind": "per_output_maxabs_q7_continuous",
        "qmax": 7.0,
        "log2_scale_mean": -6.328672409057617,
        "log2_scale_std": 0.5959522128105164,
    }
    if payload.get("normalization") != expected_normalization:
        raise ValueError("categorical encoder normalization contract drifted")
    expected_gptq = {
        "kind": "symmetric_per_output_group_gptq",
        "bits": 4,
        "qmax": 7,
        "damp_fraction": 0.01,
        "group_size": 128,
        "target_stage": "after_emitted_permutation_and_retiling",
    }
    if payload.get("gptq") != expected_gptq:
        raise ValueError("GPTQ teacher contract drifted")
    expected_scale_bins = {
        "kind": "uniform_log2_boundary_saturation",
        "vocabulary_size": 256,
        "log2_min": -12.0,
        "log2_max": 0.0,
    }
    if payload.get("scale_bins") != expected_scale_bins:
        raise ValueError("categorical scale-bin contract drifted")
    expected_loss = {
        "kind": "weighted_cumulative_ordinal_log",
        "threshold_cost": "squared_lattice_distance",
        "normalization": "maximum_squared_lattice_distance",
        "component_reduction": "code_plus_scale",
    }
    if payload.get("loss") != expected_loss:
        raise ValueError("categorical loss contract drifted")
    expected_diagnostics = {
        "old_losses_detached": True,
        "old_behavioral_operator_coefficient": 0.0,
        "old_direction_contrastive_temperature": 0.10,
        "old_latent_anticollapse_margin": 0.10,
        "old_latent_anticollapse_eps": 1.0e-6,
    }
    if payload.get("diagnostics") != expected_diagnostics:
        raise ValueError("detached old-loss diagnostic contract drifted")
    return payload


class ParallelCategoricalGPTQWeightBottleneck(UnifiedWeightBottleneck):
    """p32 encoder, fixed bottleneck and parallel p16 categorical writer.

    Decoder address queries have residual state, but every sample-specific value
    reaches them only through cross-attention to z.  Activation-distribution
    features condition the encoder and are deliberately absent from the decoder.
    """

    output_patch_size = OUTPUT_PATCH_SIZE
    subpatches_per_input_token = 2

    def __init__(
        self,
        cfg: ExperimentConfig,
        arm: str,
        *,
        scale_vocab_size: int = 256,
    ) -> None:
        if cfg.values_per_token != 32:
            raise ValueError("parallel categorical production requires p32 input tokens")
        if cfg.structural_patch_size != self.output_patch_size:
            raise ValueError("parallel categorical production requires p16 outputs")
        if cfg.decoder_depth != 6:
            raise ValueError("parallel categorical production requires six decoder blocks")
        if scale_vocab_size < 3:
            raise ValueError("scale vocabulary must contain at least three ordered bins")
        super().__init__(cfg, arm)

        original_tail = self.decoder_blocks[-1]
        self.decoder_blocks = nn.ModuleList(list(self.decoder_blocks[:-1]))
        self.shared_decoder_depth = len(self.decoder_blocks)
        self.categorical_tail = original_tail
        self.scale_vocab_size = int(scale_vocab_size)

        # Decoder conditioning must be latent-owned.  The Distribution Encoder
        # remains active in encode(), but has no direct decoder value path.
        del self.decoder_query_conditioner
        del self.output_norm
        del self.output_head

        dim = cfg.hidden_dim
        self.half_embedding = nn.Parameter(torch.empty(2, dim))
        self.code_output_norm = AffineFreeRMSNorm(dim)
        self.scale_output_norm = AffineFreeRMSNorm(dim)
        self.code_head = nn.Linear(
            dim,
            self.output_patch_size * CODE_VOCAB_SIZE,
            bias=False,
        )
        self.scale_head = nn.Linear(dim, self.scale_vocab_size, bias=False)
        nn.init.normal_(self.half_embedding, std=0.02)
        nn.init.normal_(self.code_head.weight, std=0.02)
        nn.init.normal_(self.scale_head.weight, std=0.02)

    @staticmethod
    def _cross_attention_block_impl(
        block: PreNormTransformerBlock,
        query_state: torch.Tensor,
        memory: torch.Tensor,
        query_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, dim = query_state.shape
        if memory.ndim != 3 or memory.shape[0] != batch or memory.shape[-1] != dim:
            raise ValueError("decoder memory must be [B,latent_slots,hidden_dim]")
        if tuple(query_valid_mask.shape) != (batch, query_count):
            raise ValueError("query validity mask disagrees with decoder state")
        if block.context_key_projection is not None:
            raise ValueError("categorical decoder blocks cannot have direct context keys")

        query_normalized = block.attn_norm(query_state)
        memory_normalized = block.attn_norm(memory)
        qkv_weight = block.qkv.weight
        query = F.linear(query_normalized, qkv_weight[:dim])
        key = F.linear(memory_normalized, qkv_weight[dim : 2 * dim])
        value = F.linear(memory_normalized, qkv_weight[2 * dim :])
        query = query.view(batch, query_count, block.heads, block.head_dim).transpose(1, 2)
        key = key.view(batch, memory.shape[1], block.heads, block.head_dim).transpose(1, 2)
        value = value.view(batch, memory.shape[1], block.heads, block.head_dim).transpose(1, 2)
        if block.bounded_cosine_attention:
            query = F.normalize(query.float(), dim=-1, eps=1.0e-6).to(query.dtype)
            key = F.normalize(key.float(), dim=-1, eps=1.0e-6).to(key.dtype)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            scale=(block.attention_logit_scale if block.bounded_cosine_attention else None),
        )
        attention_write = block.attn_out(
            attended.transpose(1, 2).reshape(batch, query_count, dim)
        )
        query_state = query_state + block._bounded_write(attention_write)
        left, right = block.mlp_in(block.mlp_norm(query_state)).chunk(2, dim=-1)
        hidden = F.silu(left) * right
        if block.bounded_swiglu_hidden:
            hidden_rms_sq = hidden.float().square().mean(dim=-1, keepdim=True)
            hidden = hidden * torch.rsqrt(1.0 + hidden_rms_sq).to(hidden.dtype)
        query_state = query_state + block._bounded_write(block.mlp_out(hidden))
        return query_state * query_valid_mask.unsqueeze(-1).to(query_state.dtype)

    def _run_cross_attention_block(
        self,
        block: PreNormTransformerBlock,
        query_state: torch.Tensor,
        memory: torch.Tensor,
        query_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.activation_checkpointing and self.training:
            return checkpoint(
                lambda current, current_memory: self._cross_attention_block_impl(
                    block,
                    current,
                    current_memory,
                    query_valid_mask,
                ),
                query_state,
                memory,
                use_reentrant=False,
            )
        return self._cross_attention_block_impl(
            block,
            query_state,
            memory,
            query_valid_mask,
        )

    def decode_categorical(
        self,
        z: torch.Tensor,
        tile_index: torch.Tensor,
        *,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        tile_col: torch.Tensor | None,
        token_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        if tuple(d_in_mask.shape) != (batch, 128):
            raise ValueError("d_in_mask must be [B,128]")
        if tuple(d_out_mask.shape) != (batch, 128):
            raise ValueError("d_out_mask must be [B,128]")
        p32_valid = (
            d_out_mask[:, :, None]
            & d_in_mask.view(batch, 4, 32).any(dim=-1)[:, None, :]
        ).reshape(batch, self.weight_token_count)
        if not torch.equal(token_valid_mask.to(p32_valid.device).bool(), p32_valid):
            raise ValueError("token_valid_mask disagrees with d_in/d_out masks")

        memory = self.from_latent(z)
        queries = self.output_queries[None].expand(batch, -1, -1)
        queries = queries + self._tile_position_embedding(tile_index, tile_col)[:, None]
        queries = queries * p32_valid.unsqueeze(-1).to(queries.dtype)
        for block in self.decoder_blocks:
            queries = self._run_cross_attention_block(block, queries, memory, p32_valid)

        p16_component_valid = d_in_mask.view(batch, 8, 16)
        p16_valid = (
            d_out_mask[:, :, None]
            & p16_component_valid.any(dim=-1)[:, None, :]
        ).reshape(batch, 1024)
        children = (
            queries[:, :, None, :]
            + self.half_embedding[None, None, :, :].to(queries.dtype)
        ).reshape(batch, 1024, self.cfg.hidden_dim)
        children = children * p16_valid.unsqueeze(-1).to(children.dtype)
        children = self._run_cross_attention_block(
            self.categorical_tail,
            children,
            memory,
            p16_valid,
        )
        children_by_output = children.view(batch, 128, 8, self.cfg.hidden_dim)
        code_logits = self.code_head(
            self.code_output_norm(children_by_output)
        ).view(batch, 128, 8, 16, CODE_VOCAB_SIZE)

        counts = p16_component_valid.sum(dim=-1).float()
        pooled = (
            children_by_output.float() * counts[:, None, :, None]
        ).sum(dim=2) / counts.sum(dim=-1).clamp_min(1.0)[:, None, None]
        pooled = pooled * d_out_mask[:, :, None].to(pooled.dtype)
        scale_logits = self.scale_head(
            self.scale_output_norm(pooled.to(children.dtype))
        )
        return code_logits, scale_logits, children_by_output

    def forward(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_index: torch.Tensor,
        activation_context: torch.Tensor,
        *,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        tile_col: torch.Tensor | None = None,
        activation_sample_mask: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        capture_depth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
        if token_valid_mask is None:
            raise ValueError("categorical forward requires token_valid_mask")
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
        code_logits, scale_logits, _children = self.decode_categorical(
            z,
            tile_index,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid_mask,
        )
        return code_logits, scale_logits, z, telemetry


@torch.no_grad()
def _mask_aware_gptq_targets(
    weights: torch.Tensor,
    calibration_x: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    *,
    bits: int = 4,
    damp_fraction: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ on the emitted/padded tile, with invalid dimensions decoupled."""
    if weights.ndim != 3 or tuple(weights.shape[-2:]) != (128, 128):
        raise ValueError("GPTQ requires weights [B,128,128]")
    batch = weights.shape[0]
    if calibration_x.ndim != 3 or calibration_x.shape[0] != batch or calibration_x.shape[2] != 128:
        raise ValueError("GPTQ requires calibration_x [B,N,128]")
    if tuple(x_mask.shape) != tuple(calibration_x.shape[:2]):
        raise ValueError("x_mask shape disagrees with calibration_x")
    if tuple(d_in_mask.shape) != (batch, 128) or tuple(d_out_mask.shape) != (batch, 128):
        raise ValueError("GPTQ dimension masks must be [B,128]")
    qmax = 2 ** (int(bits) - 1) - 1
    rows = weights.transpose(1, 2).float().contiguous()
    value_valid = d_out_mask[:, :, None] & d_in_mask[:, None, :]
    rows = rows * value_valid.to(rows.dtype)
    max_abs = rows.abs().amax(dim=-1)
    scales = max_abs.clamp_min(1.0e-8) / float(qmax)

    # The frozen project codec calibrates on the first 256 emitted rows.  Keep
    # this exact even when the bank stores a larger activation context.
    calibration_rows = min(256, calibration_x.shape[1])
    active_x_mask = x_mask[:, :calibration_rows]
    valid_row_count = active_x_mask.sum(dim=-1)
    if bool((valid_row_count == 0).any()):
        raise ValueError("GPTQ requires at least one valid calibration row per tile")
    valid_x = (
        calibration_x[:, :calibration_rows].float()
        * active_x_mask[:, :, None].to(torch.float32)
        * d_in_mask[:, None, :].to(torch.float32)
    )
    # TF32 would change the teacher codes relative to the existing IEEE-FP32
    # GPTQ implementation.  This process is single-GPU/single-threaded here, so
    # a scoped flag restore is sufficient and leaves model matmuls unchanged.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        hessian = valid_x.transpose(1, 2) @ valid_x
        hessian.mul_(2.0 / valid_row_count.float()[:, None, None])
        matrix_valid = d_in_mask[:, :, None] & d_in_mask[:, None, :]
        hessian.mul_(matrix_valid.to(hessian.dtype))
        diagonal = hessian.diagonal(dim1=-2, dim2=-1)
        valid_dim_count = d_in_mask.sum(dim=-1).clamp_min(1).float()
        diagonal_mean = (
            diagonal * d_in_mask.to(diagonal.dtype)
        ).sum(dim=-1) / valid_dim_count
        # Apply the same positive damping to padded diagonal coordinates.  The
        # padded block stays decoupled, becomes invertible, and cannot influence
        # the valid GPTQ trajectory.
        diagonal.add_(
            float(damp_fraction) * diagonal_mean[:, None].clamp_min(1.0e-8)
        )
        inverse = torch.linalg.inv(hessian)
        inverse_factor, info = torch.linalg.cholesky_ex(inverse, upper=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    if bool((info != 0).any()):
        bad = torch.nonzero(info != 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"GPTQ inverse Cholesky failed for batch rows {bad}")

    work = rows.clone()
    codes = torch.zeros_like(work, dtype=torch.int8)
    for column in range(128):
        current = work[:, :, column]
        code = torch.round(current / scales).clamp(-qmax, qmax)
        column_valid = d_in_mask[:, column, None] & d_out_mask
        code = code * column_valid.to(code.dtype)
        quantized = code * scales
        codes[:, :, column] = code.to(torch.int8)
        error = (current - quantized) / inverse_factor[:, column, column, None]
        error = error * column_valid.to(error.dtype)
        work[:, :, column:] -= (
            error[:, :, None] * inverse_factor[:, None, column, column:]
        )
    dequantized = codes.float() * scales[:, :, None]
    dequantized = dequantized * value_valid.to(dequantized.dtype)
    return codes, scales, dequantized.transpose(1, 2).contiguous()


def _scale_bin_centers(
    vocabulary_size: int,
    log2_min: float,
    log2_max: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if vocabulary_size < 3 or not log2_min < log2_max:
        raise ValueError("invalid scale-bin range")
    return torch.linspace(
        float(log2_min),
        float(log2_max),
        int(vocabulary_size),
        device=device,
        dtype=torch.float32,
    )


def _scale_class_targets(scales: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    if scales.ndim != 2:
        raise ValueError("scale targets must be [B,d_out]")
    boundaries = 0.5 * (centers[:-1] + centers[1:])
    return torch.bucketize(torch.log2(scales.float()), boundaries)


def _ordered_cumulative_log_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stable ordinal log-loss whose crossed-threshold point cost is d^2."""
    if logits.shape[:-1] != targets.shape or targets.shape != valid_mask.shape:
        raise ValueError("ordinal logits/targets/mask shapes disagree")
    classes = logits.shape[-1]
    if classes < 2:
        raise ValueError("ordinal loss requires at least two classes")
    if targets.dtype != torch.long:
        targets = targets.long()
    if bool(((targets < 0) | (targets >= classes)).any()):
        raise ValueError("ordinal target is outside the vocabulary")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        zero = logits.float().sum() * 0.0
        return zero, {
            "accuracy": zero.detach(),
            "mean_absolute_bin_error": zero.detach(),
            "mean_squared_bin_error": zero.detach(),
            "off_by_one_accuracy": zero.detach(),
            "prediction_entropy": zero.detach(),
            "target_low_boundary_fraction": zero.detach(),
            "target_high_boundary_fraction": zero.detach(),
            "prediction_low_boundary_fraction": zero.detach(),
            "prediction_high_boundary_fraction": zero.detach(),
            "valid_count": zero.detach(),
        }

    logits32 = logits.float()
    log_normalizer = torch.logsumexp(logits32, dim=-1, keepdim=True)
    log_le = torch.logcumsumexp(logits32, dim=-1)[..., :-1] - log_normalizer
    log_gt = (
        torch.flip(
            torch.logcumsumexp(torch.flip(logits32, dims=(-1,)), dim=-1),
            dims=(-1,),
        )[..., 1:]
        - log_normalizer
    )
    thresholds = torch.arange(classes - 1, device=logits.device)
    target_expanded = targets[..., None]
    target_is_le = target_expanded <= thresholds
    threshold_nll = -torch.where(target_is_le, log_le, log_gt)
    threshold_weight = (
        (2 * (thresholds - target_expanded) + 1).abs().float()
        / float((classes - 1) ** 2)
    )
    per_target = (threshold_weight * threshold_nll).sum(dim=-1)
    loss = per_target[valid].mean()

    prediction = logits32.argmax(dim=-1)
    bin_error = (prediction - targets).abs().float()
    probabilities = logits32.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum(dim=-1)
    entropy = entropy / math.log(float(classes))
    valid_targets = targets[valid]
    valid_predictions = prediction[valid]
    valid_error = bin_error[valid]
    return loss, {
        "accuracy": (valid_predictions == valid_targets).float().mean().detach(),
        "mean_absolute_bin_error": valid_error.mean().detach(),
        "mean_squared_bin_error": valid_error.square().mean().detach(),
        "off_by_one_accuracy": (valid_error <= 1.0).float().mean().detach(),
        "prediction_entropy": entropy[valid].mean().detach(),
        "target_low_boundary_fraction": (valid_targets == 0).float().mean().detach(),
        "target_high_boundary_fraction": (
            valid_targets == classes - 1
        ).float().mean().detach(),
        "prediction_low_boundary_fraction": (
            valid_predictions == 0
        ).float().mean().detach(),
        "prediction_high_boundary_fraction": (
            valid_predictions == classes - 1
        ).float().mean().detach(),
        "valid_count": valid.new_tensor(float(valid.sum())).detach(),
    }


@torch.no_grad()
def _decode_categorical_argmax(
    code_logits: torch.Tensor,
    scale_logits: torch.Tensor,
    scale_centers: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = code_logits.shape[0]
    expected = (batch, 128, 8, 16, CODE_VOCAB_SIZE)
    if tuple(code_logits.shape) != expected:
        raise ValueError(f"unexpected categorical code logits shape {tuple(code_logits.shape)}")
    if tuple(scale_logits.shape) != (batch, 128, scale_centers.numel()):
        raise ValueError("unexpected categorical scale logits shape")
    code_ids = code_logits.float().argmax(dim=-1)
    scale_ids = scale_logits.float().argmax(dim=-1)
    signed_codes = code_ids.float() - float(CODE_ZERO_INDEX)
    decoded_scales = torch.exp2(scale_centers.index_select(0, scale_ids.reshape(-1))).view(
        batch, 128
    )
    component_valid = (
        d_out_mask[:, :, None, None]
        & d_in_mask.view(batch, 1, 8, 16)
    )
    rows = signed_codes * decoded_scales[:, :, None, None]
    rows = rows * component_valid.to(rows.dtype)
    prediction = rows.reshape(batch, 128, 128).transpose(1, 2).contiguous()
    return prediction, code_ids, scale_ids


def _prediction_directions(
    prediction: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    batch = prediction.shape[0]
    rows = prediction.transpose(1, 2).float().reshape(batch, 128, 8, 16)
    component_valid = (
        d_out_mask[:, :, None, None]
        & d_in_mask.view(batch, 1, 8, 16)
    )
    rows = rows * component_valid.to(rows.dtype)
    norms = rows.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-12)
    return (rows / norms) * component_valid.to(rows.dtype)


def _relative_raw_nrmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    value_mask: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()) * value_mask.to(torch.float32)
    target_masked = target.float() * value_mask.to(torch.float32)
    return (error.square().sum() / target_masked.square().sum().clamp_min(1.0e-12)).sqrt()


def _categorical_gradient_telemetry(
    model: ParallelCategoricalGPTQWeightBottleneck,
) -> dict[str, Any]:
    groups: dict[str, list[torch.Tensor]] = {}

    def add(group: str, gradient: torch.Tensor) -> None:
        groups.setdefault(group, []).append(gradient.detach())

    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        if name.startswith("encoder_blocks."):
            add(f"encoder_block_{int(name.split('.')[1]) + 1}", parameter.grad)
        elif name.startswith("decoder_blocks."):
            add(f"decoder_block_{int(name.split('.')[1]) + 1}", parameter.grad)
        elif name.startswith("categorical_tail."):
            add("categorical_tail", parameter.grad)
        elif name.startswith("code_head."):
            add("code_head", parameter.grad)
        elif name.startswith("scale_head."):
            add("scale_head", parameter.grad)
        elif name.startswith("distribution_encoder."):
            add("distribution_encoder", parameter.grad)
        elif name.startswith(("to_latent.", "from_latent.")):
            add("bottleneck_projections", parameter.grad)
        else:
            add("input_and_queries", parameter.grad)
    result: dict[str, Any] = {
        "none_parameter_tensors": len(missing),
        "none_parameter_names": missing,
    }
    for name, gradients in sorted(groups.items()):
        squared = sum(float(value.float().square().sum()) for value in gradients)
        count = sum(value.numel() for value in gradients)
        result[name] = {
            "gradient_rms": math.sqrt(squared / max(count, 1)),
            "parameter_tensors": len(gradients),
            "numel": count,
        }
    return result


def _latent_component_gradient_stats(
    code_loss: torch.Tensor,
    scale_loss: torch.Tensor,
    latent: torch.Tensor,
) -> dict[str, float]:
    code_grad = torch.autograd.grad(code_loss, latent, retain_graph=True)[0].float()
    scale_grad = torch.autograd.grad(scale_loss, latent, retain_graph=True)[0].float()
    code_rms = code_grad.square().mean().sqrt()
    scale_rms = scale_grad.square().mean().sqrt()
    cosine = (code_grad * scale_grad).mean() / (
        code_rms * scale_rms
    ).clamp_min(1.0e-30)
    return {
        "code_gradient_rms": float(code_rms),
        "scale_gradient_rms": float(scale_rms),
        "code_scale_gradient_cosine": float(cosine),
        "scale_over_code_gradient_rms": float(scale_rms / code_rms.clamp_min(1.0e-30)),
    }


@torch.no_grad()
def _detached_old_loss_diagnostics(
    X: torch.Tensor,
    W: torch.Tensor,
    prediction: torch.Tensor,
    latent: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    layout_groups: list[torch.Tensor],
    config: dict[str, Any],
) -> dict[str, float]:
    pred_dirs = _prediction_directions(prediction, d_in_mask, d_out_mask)
    old_loss_cfg = {
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
    old_total, parts = _production_loss(
        X,
        W,
        prediction,
        x_mask,
        d_in_mask,
        d_out_mask,
        old_loss_cfg,
        pred_dirs=pred_dirs,
    )
    actual_operator = BigWeightVAELossMixin.operator_recon_loss(
        X,
        W,
        prediction,
        x_mask=x_mask,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
    )
    anti_loss, anti_stats = _latent_anticollapse_objective(
        latent.detach(),
        margin=float(config["diagnostics"]["old_latent_anticollapse_margin"]),
        eps=float(config["diagnostics"]["old_latent_anticollapse_eps"]),
    )
    direction_nce, direction_nce_stats = _same_layout_direction_infonce(
        pred_dirs,
        W,
        d_in_mask,
        d_out_mask,
        layout_groups,
        temperature=float(
            config["diagnostics"]["old_direction_contrastive_temperature"]
        ),
    )
    result = {
        "diagnostic_old_legacy_task_loss": float(old_total),
        "diagnostic_old_behavioral_operator": float(parts["behavioral_operator"]),
        "diagnostic_old_behavioral_operator_actual": float(actual_operator),
        "diagnostic_old_behavioral": float(parts["behavioral"]),
        "diagnostic_old_behavioral_direction": float(parts["behavioral_direction"]),
        "diagnostic_old_behavioral_scale": float(parts["behavioral_scale"]),
        "diagnostic_old_structural": float(parts["structural"]),
        "diagnostic_old_structural_direction": float(parts["structural_direction"]),
        "diagnostic_old_structural_scale": float(parts["structural_scale"]),
        "diagnostic_old_structural_reconstruction": 0.0,
        "diagnostic_old_structural_relational": 0.0,
        "diagnostic_old_latent_anticollapse_loss": float(anti_loss),
        "diagnostic_old_direction_contrastive_loss": float(direction_nce),
    }
    result.update(
        {
            f"diagnostic_old_{name}": float(value)
            for name, value in anti_stats.items()
        }
    )
    result.update(
        {
            f"diagnostic_old_{name}": float(value)
            for name, value in direction_nce_stats.items()
        }
    )
    return result


def _plot_metrics(metrics_path: Path, output_path: Path) -> None:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(steps, [row["loss"] for row in rows], label="categorical total")
    axes[0].plot(steps, [row["code_ordinal_loss"] for row in rows], label="code ordinal")
    axes[0].plot(steps, [row["scale_ordinal_loss"] for row in rows], label="scale ordinal")
    axes[0].set_ylabel("backward loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        steps,
        [row["diagnostic_old_legacy_task_loss"] for row in rows],
        label="old legacy (detached)",
    )
    axes[1].plot(
        steps,
        [row["hard_decode_raw_nrmse"] for row in rows],
        label="hard decode raw NRMSE",
    )
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("diagnostic")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Parallel categorical GPTQ 697M production")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _batch_stream(
    config: dict[str, Any],
    start_index: int,
    logger: logging.Logger,
    stack: ExitStack,
) -> Iterator[Any]:
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
    sampler.set_start_index(start_index)
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
    stream = BalancedOperatorBankMixer(dataset, bundle_iter, start_index=start_index)
    shutdown_workers = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown_workers):
        stack.callback(shutdown_workers)
    return stream


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    model_cfg = _model_config(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_root is not None:
        output_root = args.output_root.resolve()
    elif args.smoke:
        output_root = Path(
            "/mnt/shared/weightclip_benchmark/"
            f"parallel_categorical_gptq_700m_smoke_{stamp}"
        )
    else:
        output_root = Path(config["output_root"]).resolve()
    steps = 2 if args.smoke else int(config["steps"])
    batch_size = int(config["batch_size"])
    resume_path = Path(config["resume_checkpoint"]).resolve()
    persistent_model_path = Path(config["persistent_model_checkpoint"]).resolve()

    startup = {
        "schema": SCHEMA,
        "stage": "preflight",
        "config_path": str(config_path),
        "resolved_config": config,
        "model_config": asdict(model_cfg),
        "device": config["device"],
        "dtype": "bfloat16 autocast; FP32 GPTQ/loss/metrics",
        "seed": int(config["seed"]),
        "output_root": str(output_root),
        "steps": steps,
        "scientific_horizon_steps": int(config["steps"]),
        "resume_checkpoint": str(resume_path),
        "persistent_model_checkpoint": str(persistent_model_path),
    }
    print("[categorical-gptq-production] stage=preflight", json.dumps(startup), flush=True)
    if args.dry_run:
        print("[categorical-gptq-production] stage=complete mode=dry-run", flush=True)
        return
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"fresh output root already exists: {output_root}")
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production run requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(config["seed"]))

    print("[categorical-gptq-production] stage=model-build", flush=True)
    model = ParallelCategoricalGPTQWeightBottleneck(
        model_cfg,
        "normalized_float",
        scale_vocab_size=int(config["scale_bins"]["vocabulary_size"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"categorical production parameter count drifted: {parameter_count} != {EXPECTED_PARAMETERS}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_step = 0
    committed_logical_index = 0
    if args.resume:
        print(f"[categorical-gptq-production] stage=resume-load path={resume_path}", flush=True)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["schema"] != SCHEMA or payload["config"] != config:
            raise RuntimeError("categorical resume schema/config drifted")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["step"])
        committed_logical_index = int(payload["committed_logical_index"])
        if committed_logical_index != start_step * batch_size:
            raise RuntimeError("categorical resume cursor disagrees with step")
        _restore_rng_state(payload["rng_state"])
    if {float(group["lr"]) for group in optimizer.param_groups} != {
        float(config["learning_rate"])
    }:
        raise RuntimeError("effective optimizer learning rate drifted")
    startup["parameter_count"] = parameter_count
    if not args.resume:
        _atomic_json(output_root / "resolved_config.json", startup)
    print(
        "[categorical-gptq-production] stage=model-build-complete "
        f"parameters={parameter_count} start_step={start_step} cursor={committed_logical_index}",
        flush=True,
    )

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[categorical-gptq-production] stage=stop-requested signal={signum}; "
            "saving after the current optimizer step",
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    logger = logging.getLogger("categorical-gptq-production")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    metrics_path = output_root / "train_metrics.jsonl"
    gradient_path = output_root / "gradient_telemetry.jsonl"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    scale_centers = _scale_bin_centers(
        int(config["scale_bins"]["vocabulary_size"]),
        float(config["scale_bins"]["log2_min"]),
        float(config["scale_bins"]["log2_max"]),
        device=device,
    )

    def save_resume(step: int, cursor: int) -> None:
        print(
            f"[categorical-gptq-production] stage=resume-save step={step} path={resume_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": SCHEMA,
                "step": step,
                "committed_logical_index": cursor,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state(),
                "config": config,
            },
            resume_path,
        )

    def save_model(step: int) -> None:
        print(
            f"[categorical-gptq-production] stage=model-save step={step} path={persistent_model_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": SCHEMA,
                "step": step,
                "model_state": model.state_dict(),
                "model_config": asdict(model_cfg),
                "config": config,
            },
            persistent_model_path,
        )

    bank = config["operator_bank"]
    print(
        "[categorical-gptq-production] stage=data-open "
        f"manifest={bank['pair_manifest']} workers={bank['loader_workers']} "
        f"permutation_views={bank['permutation_views']} cache=online-gptq-no-cache",
        flush=True,
    )
    with ExitStack() as stack:
        stream = _batch_stream(config, committed_logical_index, logger, stack)
        print(
            "[categorical-gptq-production] stage=train "
            f"steps={steps} batch={batch_size} lr={config['learning_rate']} "
            "backward=code_ordinal+scale_ordinal old_losses=detached-log-only",
            flush=True,
        )
        for step in range(start_step + 1, steps + 1):
            cpu_batch = _batch_from_stream(stream, batch_size)
            W = cpu_batch["W"].to(device, non_blocking=True)
            X = cpu_batch["X"].to(device, non_blocking=True)
            x_mask = cpu_batch["x_mask"].to(device, non_blocking=True)
            d_in_mask = cpu_batch["d_in_mask"].to(device, non_blocking=True)
            d_out_mask = cpu_batch["d_out_mask"].to(device, non_blocking=True)
            tile_row = cpu_batch["tile_row"].to(device, non_blocking=True)
            tile_col = cpu_batch["tile_col"].to(device, non_blocking=True)

            tokenizer_started = time.monotonic()
            target_codes, target_scales, teacher_prediction = _mask_aware_gptq_targets(
                W,
                X,
                x_mask,
                d_in_mask,
                d_out_mask,
                bits=int(config["gptq"]["bits"]),
                damp_fraction=float(config["gptq"]["damp_fraction"]),
            )
            tokenizer_seconds = time.monotonic() - tokenizer_started
            code_targets = target_codes.long().view(batch_size, 128, 8, 16) + CODE_ZERO_INDEX
            scale_targets = _scale_class_targets(target_scales, scale_centers)
            code_valid = (
                d_out_mask[:, :, None, None]
                & d_in_mask.view(batch_size, 1, 8, 16)
            )
            nonzero_scale = target_scales > 1.0e-8 / 7.0
            scale_valid = d_out_mask & d_in_mask.any(dim=-1)[:, None] & nonzero_scale

            content, input_log_scale, token_valid = _prepare_normalized_inputs(
                W,
                d_in_mask,
                d_out_mask,
                scale_mean=float(config["normalization"]["log2_scale_mean"]),
                scale_std=float(config["normalization"]["log2_scale_std"]),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                code_logits, scale_logits, latent, _depth = model(
                    content,
                    input_log_scale,
                    tile_row,
                    X,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    tile_col=tile_col,
                    activation_sample_mask=x_mask,
                    token_valid_mask=token_valid,
                )
            with torch.autocast(device_type="cuda", enabled=False):
                code_loss, code_stats = _ordered_cumulative_log_loss(
                    code_logits,
                    code_targets,
                    code_valid,
                )
                scale_loss, scale_stats = _ordered_cumulative_log_loss(
                    scale_logits,
                    scale_targets,
                    scale_valid,
                )
                loss = code_loss + scale_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite categorical loss at step {step}")

            gradient_due = step == 1 or step % int(config["gradient_log_every"]) == 0
            latent_component_stats = (
                _latent_component_gradient_stats(code_loss, scale_loss, latent)
                if gradient_due
                else {}
            )
            loss.backward()
            gradient_groups = _categorical_gradient_telemetry(model) if gradient_due else {}
            if gradient_due:
                unexpected_missing = [
                    name
                    for name in gradient_groups["none_parameter_names"]
                    if name != "extra_tile_row_embedding.weight"
                ]
                if unexpected_missing:
                    raise RuntimeError(
                        f"categorical production has unexpected missing gradients: {unexpected_missing}"
                    )
                required_groups = [
                    *(f"encoder_block_{depth}" for depth in range(1, 14)),
                    *(f"decoder_block_{depth}" for depth in range(1, 6)),
                    "categorical_tail",
                    "code_head",
                    "scale_head",
                    "distribution_encoder",
                    "bottleneck_projections",
                ]
                dead = [
                    name
                    for name in required_groups
                    if name not in gradient_groups
                    or float(gradient_groups[name]["gradient_rms"]) <= 0.0
                ]
                if dead:
                    raise RuntimeError(f"categorical production has dead gradient groups: {dead}")
                _append_jsonl(
                    gradient_path,
                    {
                        "schema": SCHEMA,
                        "step": step,
                        "latent_code_scale_gradients": latent_component_stats,
                        "groups": gradient_groups,
                    },
                )
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["grad_clip_norm"]),
                )
            )
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite categorical gradient at step {step}")
            optimizer.step()
            committed_logical_index = int(cpu_batch["logical_indices"][-1]) + 1

            log_due = step == 1 or step % int(config["log_every"]) == 0
            if log_due:
                hard_prediction, _hard_codes, _hard_scales = _decode_categorical_argmax(
                    code_logits.detach(),
                    scale_logits.detach(),
                    scale_centers,
                    d_in_mask,
                    d_out_mask,
                )
                layout_groups = _exact_layout_groups(
                    cpu_batch["tile_row"],
                    cpu_batch["tile_col"],
                    cpu_batch["d_in_mask"],
                    cpu_batch["d_out_mask"],
                    minimum_group_size=2,
                )
                old_diagnostics = _detached_old_loss_diagnostics(
                    X,
                    W,
                    hard_prediction,
                    latent,
                    x_mask,
                    d_in_mask,
                    d_out_mask,
                    layout_groups,
                    config,
                )
                value_mask = d_in_mask[:, :, None] & d_out_mask[:, None, :]
                binned_teacher = (
                    (target_codes.float())
                    * torch.exp2(
                        scale_centers.index_select(0, scale_targets.reshape(-1))
                    ).view(batch_size, 128)[:, :, None]
                ).transpose(1, 2).contiguous()
                binned_teacher = binned_teacher * value_mask.to(binned_teacher.dtype)
                target_binned_scales = torch.exp2(
                    scale_centers.index_select(0, scale_targets.reshape(-1))
                ).view(batch_size, 128)
                valid_scale_relative_error = (
                    (target_binned_scales - target_scales).abs()
                    / target_scales.clamp_min(1.0e-30)
                )[scale_valid]
                calibration_x_mask = x_mask.clone()
                calibration_x_mask[:, 256:] = False
                heldout_x_mask = x_mask.clone()
                heldout_x_mask[:, :256] = False
                elapsed = time.monotonic() - started
                row = {
                    "schema": SCHEMA,
                    "step": step,
                    "loss": float(loss.detach()),
                    "code_ordinal_loss": float(code_loss.detach()),
                    "scale_ordinal_loss": float(scale_loss.detach()),
                    **{f"code_{name}": float(value) for name, value in code_stats.items()},
                    **{f"scale_{name}": float(value) for name, value in scale_stats.items()},
                    "hard_decode_raw_nrmse": float(
                        _relative_raw_nrmse(hard_prediction, W, value_mask)
                    ),
                    "teacher_continuous_scale_raw_nrmse": float(
                        _relative_raw_nrmse(teacher_prediction, W, value_mask)
                    ),
                    "teacher_binned_scale_raw_nrmse": float(
                        _relative_raw_nrmse(binned_teacher, W, value_mask)
                    ),
                    "teacher_continuous_scale_operator_metric": float(
                        BigWeightVAELossMixin.operator_recon_loss(
                            X,
                            W,
                            teacher_prediction,
                            x_mask=x_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                        )
                    ),
                    "teacher_binned_scale_operator_metric": float(
                        BigWeightVAELossMixin.operator_recon_loss(
                            X,
                            W,
                            binned_teacher,
                            x_mask=x_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                        )
                    ),
                    "teacher_continuous_scale_calibration_operator_metric": float(
                        BigWeightVAELossMixin.operator_recon_loss(
                            X,
                            W,
                            teacher_prediction,
                            x_mask=calibration_x_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                        )
                    ),
                    "teacher_binned_scale_calibration_operator_metric": float(
                        BigWeightVAELossMixin.operator_recon_loss(
                            X,
                            W,
                            binned_teacher,
                            x_mask=calibration_x_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                        )
                    ),
                    "teacher_continuous_scale_heldout_operator_metric": float(
                        BigWeightVAELossMixin.operator_recon_loss(
                            X,
                            W,
                            teacher_prediction,
                            x_mask=heldout_x_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                        )
                    ),
                    "teacher_binned_scale_heldout_operator_metric": float(
                        BigWeightVAELossMixin.operator_recon_loss(
                            X,
                            W,
                            binned_teacher,
                            x_mask=heldout_x_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                        )
                    ),
                    "teacher_scale_binning_relative_error_p50": float(
                        valid_scale_relative_error.median()
                    ),
                    "teacher_scale_binning_relative_error_p99": float(
                        torch.quantile(valid_scale_relative_error, 0.99)
                    ),
                    "teacher_scale_log2_min": float(
                        torch.log2(target_scales[scale_valid]).min()
                    ),
                    "teacher_scale_log2_max": float(
                        torch.log2(target_scales[scale_valid]).max()
                    ),
                    **old_diagnostics,
                    "grad_norm_pre_clip": grad_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gptq_tokenizer_seconds": tokenizer_seconds,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": step / max(elapsed, 1.0e-9),
                    "committed_logical_index": committed_logical_index,
                    "cuda_peak_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                }
                _append_jsonl(metrics_path, row)
                print(
                    "[categorical-gptq-production] stage=train "
                    f"step={step}/{steps} loss={row['loss']:.6f} "
                    f"code={row['code_ordinal_loss']:.6f} scale={row['scale_ordinal_loss']:.6f} "
                    f"hard_nrmse={row['hard_decode_raw_nrmse']:.5f} "
                    f"old={row['diagnostic_old_legacy_task_loss']:.5f} "
                    f"grad={grad_norm:.3e} gptq={tokenizer_seconds:.3f}s "
                    f"rate={row['steps_per_second']:.3f}_steps/s peak={row['cuda_peak_gib']:.2f}GiB",
                    flush=True,
                )
            if step % int(config["plot_every"]) == 0:
                _plot_metrics(metrics_path, output_root / "train_loss_curve.png")
            if not args.smoke and step % int(config["resume_save_every"]) == 0:
                save_resume(step, committed_logical_index)
            if not args.smoke and step % int(config["model_save_every"]) == 0:
                save_model(step)
            if stop_requested:
                if not args.smoke:
                    save_resume(step, committed_logical_index)
                    save_model(step)
                _atomic_json(
                    output_root / "STOPPED.json",
                    {
                        "schema": SCHEMA,
                        "step": step,
                        "committed_logical_index": committed_logical_index,
                    },
                )
                print(f"[categorical-gptq-production] stage=stopped step={step}", flush=True)
                return

    if not args.smoke:
        save_resume(steps, committed_logical_index)
        save_model(steps)
    _plot_metrics(metrics_path, output_root / "train_loss_curve.png")
    summary = {
        "schema": SCHEMA,
        "complete": True,
        "step": steps,
        "committed_logical_index": committed_logical_index,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "metrics_path": str(metrics_path),
        "plot_path": str(output_root / "train_loss_curve.png"),
    }
    _atomic_json(output_root / "COMPLETE.json", summary)
    print("[categorical-gptq-production] stage=complete", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
