from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict
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
from torch.utils.checkpoint import checkpoint
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
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    ExperimentConfig,
    UnifiedWeightBottleneck,
    _gradient_telemetry,
    _seed_everything,
)


SCHEMA = "weightclip_direct_normalized_scaled_700m_production_v1"
BOUNDED_BALANCED_SCHEMA = (
    "weightclip_direct_normalized_scaled_700m_bounded_balanced_production_v2"
)
POLAR_TAIL_SCHEMA = (
    "weightclip_direct_normalized_scaled_700m_polar_tails_production_v1"
)
LATENT_ROOTED_POLAR_TAIL_SCHEMA = (
    "weightclip_direct_normalized_scaled_700m_latent_rooted_polar_tails_production_v1"
)
POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA = (
    "weightclip_direct_normalized_scaled_700m_"
    "polar_tails_latent_anticollapse_production_v1"
)
POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA = (
    "weightclip_direct_normalized_scaled_700m_"
    "polar_tails_latent_anticollapse_direction_infonce_production_v1"
)
POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA = (
    "weightclip_direct_normalized_scaled_700m_"
    "polar_tails_latent_anticollapse_direction_infonce_gauge_fixed_production_v1"
)
DEFAULT_CONFIG = Path(
    "conf/weightclip_benchmark/"
    "direct_normalized_scaled_700m_polar_tails_production_500k.yaml"
)
LEGACY_EXPECTED_PARAMETERS = 706_301_312
POLAR_TAIL_EXPECTED_PARAMETERS = 742_120_833
POLAR_TAIL_GAUGE_FIXED_EXPECTED_PARAMETERS = 742_119_297


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the p32 Direct Normalized 700M AE on the production operator bank."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") not in {
        SCHEMA,
        BOUNDED_BALANCED_SCHEMA,
        POLAR_TAIL_SCHEMA,
        LATENT_ROOTED_POLAR_TAIL_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA,
        POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA,
    }:
        raise ValueError(
            "config uses an unsupported schema: "
            f"{payload.get('schema')!r}"
        )
    balanced = payload["schema"] == BOUNDED_BALANCED_SCHEMA
    polar_tails = payload["schema"] in {
        POLAR_TAIL_SCHEMA,
        LATENT_ROOTED_POLAR_TAIL_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA,
        POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA,
    }
    latent_rooted = payload["schema"] == LATENT_ROOTED_POLAR_TAIL_SCHEMA
    latent_anticollapse = payload["schema"] in {
        POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA,
        POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA,
    }
    direction_infonce = payload["schema"] in {
        POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA,
        POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA,
    }
    gauge_fixed_direction = (
        payload["schema"] == POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA
    )
    resume_lr_transition = payload.get("resume_learning_rate_transition")
    expected_resume_lr_transition = {
        "kind": "preserve_adam_state_override_param_group_lr",
        "source_learning_rate": 5.0e-5,
        "target_learning_rate": 3.0e-4,
        "minimum_checkpoint_step": 1_235,
        "resume_required": True,
    }
    if resume_lr_transition is not None:
        if (
            payload["schema"]
            != POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA
        ):
            raise ValueError(
                "resume LR transition is only supported for the dedicated "
                "latent-anti direction-InfoNCE schema"
            )
        if resume_lr_transition != expected_resume_lr_transition:
            raise ValueError(
                "resume LR transition contract drifted: "
                f"expected {expected_resume_lr_transition}, "
                f"got {resume_lr_transition}"
            )
        expected_learning_rate = float(
            expected_resume_lr_transition["target_learning_rate"]
        )
    else:
        expected_learning_rate = 5.0e-5
    frozen = {
        "steps": 500_000,
        "batch_size": 32,
        "weight_decay": 0.01,
        "grad_clip_norm": 5.0,
        "seed": 42,
    }
    for key, expected in frozen.items():
        actual = payload[key]
        if float(actual) != float(expected):
            raise ValueError(f"production contract requires {key}={expected}, got {actual}")
    if float(payload["learning_rate"]) != expected_learning_rate:
        raise ValueError(
            "production contract requires learning_rate="
            f"{expected_learning_rate}, got {payload['learning_rate']}"
        )
    if payload["scheduler"] != "constant":
        raise ValueError("production contract requires constant LR")
    if tuple(payload["betas"]) != (0.9, 0.999):
        raise ValueError("production contract requires AdamW betas=(0.9,0.999)")
    objective = str(payload.get("objective", "behavioral_plus_structural"))
    allowed_objectives = {
        "gradient_balanced_direction_scale"
        if balanced
        else "behavioral_plus_structural",
        "gradient_balanced_direction_scale"
        if balanced
        else "behavioral_direction_scale_plus_structural",
        "gradient_balanced_direction_scale" if balanced else "structural_only",
    }
    if objective not in allowed_objectives:
        raise ValueError(f"unsupported production objective: {objective!r}")
    if polar_tails and objective != "behavioral_direction_scale_plus_structural":
        raise ValueError(
            "polar-tail production requires behavioral direction/scale plus structural loss"
        )
    behavioral_enabled = balanced or objective != "structural_only"
    operator_enabled = not balanced and objective == "behavioral_plus_structural"
    scale_coefficient = 1.0 if balanced else 10.0
    expected_loss = {
        "behavioral_coef": 1.0 if behavioral_enabled else 0.0,
        "behavioral_operator": 50.0 if operator_enabled else 0.0,
        "behavioral_direction": 1.0,
        "behavioral_scale": scale_coefficient,
        "structural_coef": 1.0,
        "structural_direction": 1.0,
        "structural_scale": scale_coefficient,
        "structural_reconstruction": 0.0,
        "structural_relational": 0.0,
    }
    if {key: float(payload["loss"][key]) for key in expected_loss} != expected_loss:
        raise ValueError(
            f"production loss coefficients drifted from the {objective!r} contract"
        )
    if balanced:
        expected_stability = {
            "bounded_cosine_attention": True,
            "attention_logit_scale": 2.0,
            "bounded_swiglu_hidden": True,
            "bounded_residual_writes": True,
        }
        if payload.get("stability") != expected_stability:
            raise ValueError(
                "bounded-balanced production stability contract drifted: "
                f"expected {expected_stability}, got {payload.get('stability')}"
            )
        expected_balance = {
            "kind": "bottleneck_gradient_rms_bisector",
            "reference": "raw_equal_weight_sum_rms",
        }
        if payload.get("gradient_balance") != expected_balance:
            raise ValueError(
                "bounded-balanced gradient contract drifted: "
                f"expected {expected_balance}, got {payload.get('gradient_balance')}"
            )
    elif "stability" in payload or "gradient_balance" in payload:
        raise ValueError("legacy production config cannot enable balanced stability fields")
    expected_anticollapse = {
        "kind": "two_way_centered_rms_hinge",
        "margin": 0.10,
        "coefficient": 1.0,
        "eps": 1.0e-6,
        "detach_denominator": True,
        "batch_scope": "full_physical_batch",
    }
    if latent_anticollapse:
        if payload.get("latent_anticollapse") != expected_anticollapse:
            raise ValueError(
                "latent anti-collapse contract drifted: "
                f"expected {expected_anticollapse}, got {payload.get('latent_anticollapse')}"
            )
    elif "latent_anticollapse" in payload:
        raise ValueError(
            "latent anti-collapse settings require the dedicated production schema"
        )
    expected_direction_contrastive = {
        "kind": "same_layout_prediction_target_infonce",
        "temperature": 0.10,
        "coefficient": 1.0,
        "target_detached": True,
        "group_reduction": "equal_group_mean",
        "minimum_group_size": 2,
    }
    if direction_infonce:
        if payload.get("direction_contrastive") != expected_direction_contrastive:
            raise ValueError(
                "direction contrastive contract drifted: "
                f"expected {expected_direction_contrastive}, "
                f"got {payload.get('direction_contrastive')}"
            )
    elif "direction_contrastive" in payload:
        raise ValueError(
            "direction contrastive settings require the dedicated production schema"
        )
    expected_direction_gauge = {
        "kind": "nonaffine_rmsnorm_frobenius_sphere",
        "frobenius_radius": "seed42_initial_composite",
        "retraction_after_optimizer_step": True,
        "project_first_moment_tangent": True,
        "weight_decay": 0.0,
    }
    if gauge_fixed_direction:
        if payload.get("direction_gauge_constraint") != expected_direction_gauge:
            raise ValueError(
                "direction gauge constraint drifted: "
                f"expected {expected_direction_gauge}, "
                f"got {payload.get('direction_gauge_constraint')}"
            )
    elif "direction_gauge_constraint" in payload:
        raise ValueError(
            "direction gauge settings require the dedicated production schema"
        )
    if polar_tails:
        expected_architecture = {
            "input_values_per_token": 32,
            "output_patch_size": 16,
            "shared_decoder_depth": 5,
            "direction_tail_depth": 1,
            "scale_tail_depth": 1,
            "log_scale_min": -3.0,
            "log_scale_max": 6.0,
            "initial_log_radius": -2.0,
            "gradient_routing": "coordinate_owned_stop_gradient",
        }
        if latent_rooted:
            expected_architecture["decoder_value_contract"] = (
                "latent_rooted_cross_attention_no_query_residual"
            )
        if payload.get("architecture") != expected_architecture:
            raise ValueError(
                "polar-tail architecture contract drifted: "
                f"expected {expected_architecture}, got {payload.get('architecture')}"
            )
    return payload


def _model_config(config: dict[str, Any] | None = None) -> ExperimentConfig:
    stability = {} if config is None else config.get("stability", {})
    return ExperimentConfig(
        seed=42,
        steps=500_000,
        batch_size=32,
        eval_batch_size=32,
        eval_every=1_000,
        log_every=10,
        learning_rate=(
            5.0e-5 if config is None else float(config["learning_rate"])
        ),
        warmup_steps=0,
        weight_decay=0.01,
        hidden_dim=1536,
        heads=24,
        mlp_dim=5728,
        encoder_depth=13,
        decoder_depth=6,
        latent_slots=32,
        latent_dim=384,
        values_per_token=32,
        structural_patch_size=16,
        structural_gamma=0.5,
        structural_direction_weight=1.0,
        structural_scale_weight=(
            1.0 if config is not None and config["schema"] == BOUNDED_BALANCED_SCHEMA else 10.0
        ),
        structural_huber_delta=0.1,
        use_distribution_conditioning=True,
        distribution_k_s=64,
        distribution_Kq=128,
        distribution_d_var=256,
        distribution_d_dist=256,
        distribution_num_var_attn_layers=6,
        distribution_var_attn_heads=4,
        distribution_dcn_num_cross_layers=3,
        distribution_dcn_deep_hidden=128,
        distribution_dcn_deep_layers=3,
        distribution_use_covariance=True,
        activation_checkpointing=True,
        max_tile_rows=18,
        max_tile_cols=2,
        bounded_cosine_attention=bool(
            stability.get("bounded_cosine_attention", False)
        ),
        attention_logit_scale=float(stability.get("attention_logit_scale", 2.0)),
        bounded_swiglu_hidden=bool(
            stability.get("bounded_swiglu_hidden", False)
        ),
        bounded_residual_writes=bool(
            stability.get("bounded_residual_writes", False)
        ),
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _apply_resume_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Apply an explicit LR transition without touching Adam moments."""
    target_lr = float(config["learning_rate"])
    if not (math.isfinite(target_lr) and target_lr > 0.0):
        raise ValueError("resume learning rate must be finite and positive")
    previous_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    gauge_fixed = (
        config["schema"] == POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA
    )
    if gauge_fixed:
        if len(optimizer.param_groups) != 2:
            raise RuntimeError(
                "gauge-fixed production resume requires exactly two optimizer groups"
            )
        groups_by_name = {
            str(group.get("group_name")): group for group in optimizer.param_groups
        }
        if set(groups_by_name) != {
            "main_decay",
            "direction_head_frobenius_sphere",
        }:
            raise RuntimeError("gauge-fixed optimizer group names drifted")
        expected_weight_decay = {
            "main_decay": float(config["weight_decay"]),
            "direction_head_frobenius_sphere": float(
                config["direction_gauge_constraint"]["weight_decay"]
            ),
        }
    else:
        if len(optimizer.param_groups) != 1:
            raise RuntimeError(
                "production resume requires exactly one optimizer parameter group"
            )
        groups_by_name = {"legacy": optimizer.param_groups[0]}
        expected_weight_decay = {"legacy": float(config["weight_decay"])}
    for group in optimizer.param_groups:
        if tuple(float(value) for value in group["betas"]) != tuple(
            float(value) for value in config["betas"]
        ):
            raise RuntimeError("checkpoint Adam betas disagree with config")
        if float(group["eps"]) != float(config["eps"]):
            raise RuntimeError("checkpoint Adam eps disagrees with config")
    for name, group in groups_by_name.items():
        if float(group["weight_decay"]) != expected_weight_decay[name]:
            raise RuntimeError(
                f"checkpoint Adam weight decay disagrees for group {name!r}"
            )
    transition = config.get("resume_learning_rate_transition")
    if transition is None:
        if any(previous != target_lr for previous in previous_lrs):
            raise RuntimeError(
                "checkpoint optimizer LR disagrees with the unchanged config: "
                f"checkpoint={previous_lrs}, config={target_lr}"
            )
        return {
            "kind": "unchanged",
            "checkpoint_step": int(checkpoint_step),
            "previous_learning_rates": previous_lrs,
            "effective_learning_rate": target_lr,
        }

    if gauge_fixed:
        raise RuntimeError("gauge-fixed production does not support an LR transition")
    minimum_step = int(transition["minimum_checkpoint_step"])
    if checkpoint_step < minimum_step:
        raise RuntimeError(
            "resume LR transition checkpoint is too early: "
            f"{checkpoint_step} < {minimum_step}"
        )
    allowed_previous = {
        float(transition["source_learning_rate"]),
        float(transition["target_learning_rate"]),
    }
    if any(previous not in allowed_previous for previous in previous_lrs):
        raise RuntimeError(
            "checkpoint optimizer LR is outside the declared transition: "
            f"checkpoint={previous_lrs}, allowed={sorted(allowed_previous)}"
        )
    if target_lr != float(transition["target_learning_rate"]):
        raise RuntimeError("config LR disagrees with transition target")
    for group in optimizer.param_groups:
        group["lr"] = target_lr
    if any(float(group["lr"]) != target_lr for group in optimizer.param_groups):
        raise RuntimeError("failed to apply resume LR to every optimizer param group")
    return {
        "kind": str(transition["kind"]),
        "checkpoint_step": int(checkpoint_step),
        "previous_learning_rates": previous_lrs,
        "effective_learning_rate": target_lr,
        "optimizer_state_entries": len(optimizer.state),
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    tmp.replace(path)


def _batch_from_stream(
    stream: Iterator[Any],
    batch_size: int,
) -> dict[str, torch.Tensor | list[int]]:
    samples = [next(stream) for _ in range(batch_size)]
    logical_indices = [int(sample.meta["logical_index"]) for sample in samples]
    expected = list(range(logical_indices[0], logical_indices[0] + batch_size))
    if logical_indices != expected:
        raise RuntimeError("operator-bank logical batch is not contiguous")
    return {
        "W": torch.stack([sample.weight for sample in samples]).contiguous(),
        "X": torch.stack([sample.x for sample in samples]).contiguous(),
        "x_mask": torch.stack([sample.meta["x_mask"] for sample in samples]).bool(),
        "d_in_mask": torch.stack([sample.meta["d_in_mask"] for sample in samples]).bool(),
        "d_out_mask": torch.stack([sample.meta["d_out_mask"] for sample in samples]).bool(),
        "tile_row": torch.tensor([sample.meta["tile_row"] for sample in samples]),
        "tile_col": torch.tensor([sample.meta["tile_col"] for sample in samples]),
        "logical_indices": logical_indices,
    }


def _prepare_normalized_inputs(
    W: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    *,
    scale_mean: float,
    scale_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if W.ndim != 3 or tuple(W.shape[-2:]) != (128, 128):
        raise ValueError("production p32 model requires W=[B,128,128]")
    batch = W.shape[0]
    rows = W.transpose(1, 2).float().contiguous()
    valid_values = d_out_mask[:, :, None] & d_in_mask[:, None, :]
    rows = rows * valid_values.to(dtype=rows.dtype)
    scale = rows.abs().amax(dim=-1).clamp_min(1.0e-8) / 7.0
    normalized = (rows / scale[:, :, None]) * valid_values.to(dtype=rows.dtype)
    content = normalized.view(batch, 128, 4, 32).reshape(batch, 512, 32)
    standardized = (torch.log2(scale) - float(scale_mean)) / float(scale_std)
    log_scale = standardized[:, :, None].expand(-1, -1, 4).reshape(batch, 512, 1)
    valid_chunks = d_in_mask.view(batch, 4, 32).any(dim=-1)
    token_valid = (d_out_mask[:, :, None] & valid_chunks[:, None, :]).reshape(batch, 512)
    return content, log_scale, token_valid


class AffineFreeRMSNorm(nn.Module):
    """RMS normalization with no learned scale or bias."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.dim:
            raise ValueError(
                f"affine-free RMSNorm expected width {self.dim}, got {value.shape[-1]}"
            )
        return F.rms_norm(value, (self.dim,), weight=None, eps=self.eps)


class PolarTailedUnifiedWeightBottleneck(UnifiedWeightBottleneck):
    """The p32 production trunk with independent one-block p16 polar tails.

    Decoder blocks 1..5 remain shared.  The former sixth block becomes the
    direction tail and an exact parameter copy becomes the scale tail.  Each
    p32 shared query is split into two position-distinct p16 child queries only
    at the branch point.
    """

    output_patch_size = 16
    subpatches_per_input_token = 2
    log_scale_min = -3.0
    log_scale_max = 6.0
    initial_log_radius = -2.0

    def __init__(self, cfg: ExperimentConfig, arm: str) -> None:
        if cfg.values_per_token != 32:
            raise ValueError("polar-tail production requires p32 input tokens")
        if cfg.structural_patch_size != self.output_patch_size:
            raise ValueError("polar-tail production requires p16 structural outputs")
        if cfg.decoder_depth != 6:
            raise ValueError("polar-tail production requires six decoder blocks per path")
        super().__init__(cfg, arm)

        original_tail = self.decoder_blocks[-1]
        self.decoder_blocks = nn.ModuleList(list(self.decoder_blocks[:-1]))
        self.shared_decoder_depth = len(self.decoder_blocks)
        self.direction_tail = original_tail
        self.scale_tail = deepcopy(original_tail)

        self.direction_output_norm = self.output_norm
        self.scale_output_norm = deepcopy(self.output_norm)
        del self.output_norm
        del self.output_head

        dim = cfg.hidden_dim
        with torch.random.fork_rng(devices=[]):
            half_embedding = torch.empty(
                self.subpatches_per_input_token,
                dim,
            )
            nn.init.normal_(half_embedding, std=0.02)
            self.direction_half_embedding = nn.Parameter(half_embedding.clone())
            self.scale_half_embedding = nn.Parameter(half_embedding.clone())
            self.direction_head = nn.Linear(
                dim,
                self.output_patch_size,
                bias=False,
            )
            self.scale_head = nn.Linear(dim, 1, bias=True)
            nn.init.normal_(
                self.direction_head.weight,
                std=0.02 / math.sqrt(float(dim)),
            )
            nn.init.normal_(
                self.scale_head.weight,
                std=0.02 / math.sqrt(float(dim)),
            )
            nn.init.constant_(self.scale_head.bias, self._raw_scale_bias_for_initial_radius())

    @classmethod
    def _raw_scale_bias_for_initial_radius(cls) -> float:
        # Invert the lower soft bound at the desired initial log-radius.  The
        # upper bound is effectively identity in this range.
        offset = cls.initial_log_radius - cls.log_scale_min
        return cls.log_scale_min + math.log(math.expm1(offset))

    @classmethod
    def _bound_log_scale(cls, raw_scale: torch.Tensor) -> torch.Tensor:
        bounded = cls.log_scale_min + F.softplus(raw_scale - cls.log_scale_min)
        return cls.log_scale_max - F.softplus(cls.log_scale_max - bounded)

    def _run_decoder_block(
        self,
        block: nn.Module,
        state: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.cfg.activation_checkpointing and self.training:
            return checkpoint(
                lambda current: block(current, valid_mask=valid_mask),
                state,
                use_reentrant=False,
            )
        return block(state, valid_mask=valid_mask)

    def _p32_decoder_start(
        self,
        z: torch.Tensor,
        tile_index: torch.Tensor,
        *,
        tile_col: torch.Tensor | None,
        token_valid_mask: torch.Tensor,
        dist_patch_by_patch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.from_latent(z)
        queries = self.output_queries[None].expand(z.shape[0], -1, -1)
        queries = queries + self._tile_position_embedding(tile_index, tile_col)[:, None]
        token_valid_mask = token_valid_mask.to(device=z.device, dtype=torch.bool)
        queries = queries * token_valid_mask.unsqueeze(-1).to(dtype=queries.dtype)
        if self.decoder_query_conditioner is not None:
            if dist_patch_by_patch is None:
                raise ValueError("conditioned decoder requires distribution context")
            decoder_context = dist_patch_by_patch.unsqueeze(1).expand(
                -1,
                128,
                -1,
                -1,
            ).reshape(z.shape[0], self.weight_token_count, self.cfg.distribution_d_dist)
            queries = self.decoder_query_conditioner(
                torch.cat((queries, decoder_context.to(queries.dtype)), dim=-1)
            )
            queries = queries * token_valid_mask.unsqueeze(-1).to(dtype=queries.dtype)
        elif dist_patch_by_patch is not None:
            raise ValueError("unconditioned decoder received distribution context")
        state = torch.cat((latent, queries), dim=1)
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
            state = self._run_decoder_block(block, state, state_valid_mask)
        return state, state_valid_mask

    def _split_p16_tail_state(
        self,
        shared_state: torch.Tensor,
        p16_valid_mask: torch.Tensor,
        half_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = shared_state[:, : self.cfg.latent_slots]
        queries = shared_state[:, self.cfg.latent_slots :]
        batch, token_count, dim = queries.shape
        if token_count != self.weight_token_count:
            raise RuntimeError("shared decoder query count drifted")
        children = queries[:, :, None, :] + half_embedding[None, None, :, :].to(
            dtype=queries.dtype
        )
        children = children.reshape(
            batch,
            token_count * self.subpatches_per_input_token,
            dim,
        )
        if tuple(p16_valid_mask.shape) != (
            batch,
            token_count * self.subpatches_per_input_token,
        ):
            raise ValueError("p16_valid_mask does not match split decoder queries")
        child_valid = p16_valid_mask.to(device=queries.device, dtype=torch.bool)
        children = children * child_valid.unsqueeze(-1).to(dtype=children.dtype)
        tail_state = torch.cat((latent, children), dim=1)
        tail_valid = torch.cat(
            (
                torch.ones(
                    batch,
                    self.cfg.latent_slots,
                    device=queries.device,
                    dtype=torch.bool,
                ),
                child_valid,
            ),
            dim=1,
        )
        return tail_state, tail_valid

    def decode_polar(
        self,
        z: torch.Tensor,
        tile_index: torch.Tensor,
        *,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        tile_col: torch.Tensor | None,
        token_valid_mask: torch.Tensor,
        dist_patch_by_patch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        if tuple(d_in_mask.shape) != (batch, 128):
            raise ValueError("d_in_mask must be [B,128]")
        if tuple(d_out_mask.shape) != (batch, 128):
            raise ValueError("d_out_mask must be [B,128]")
        expected_p32_valid = (
            d_out_mask[:, :, None]
            & d_in_mask.view(batch, self.chunks_per_group, self.cfg.values_per_token)
            .any(dim=-1)[:, None, :]
        ).reshape(batch, self.weight_token_count)
        if not torch.equal(
            token_valid_mask.to(device=expected_p32_valid.device, dtype=torch.bool),
            expected_p32_valid,
        ):
            raise ValueError("token_valid_mask disagrees with d_in/d_out masks")

        p16_component_mask = d_in_mask.view(
            batch,
            128 // self.output_patch_size,
            self.output_patch_size,
        )
        p16_valid = (
            d_out_mask[:, :, None]
            & p16_component_mask.any(dim=-1)[:, None, :]
        ).reshape(batch, -1)

        shared_state, _ = self._p32_decoder_start(
            z,
            tile_index,
            tile_col=tile_col,
            token_valid_mask=expected_p32_valid,
            dist_patch_by_patch=dist_patch_by_patch,
        )
        direction_state, direction_valid = self._split_p16_tail_state(
            shared_state,
            p16_valid,
            self.direction_half_embedding,
        )
        scale_state, scale_valid = self._split_p16_tail_state(
            shared_state,
            p16_valid,
            self.scale_half_embedding,
        )
        direction_state = self._run_decoder_block(
            self.direction_tail,
            direction_state,
            direction_valid,
        )
        scale_state = self._run_decoder_block(
            self.scale_tail,
            scale_state,
            scale_valid,
        )

        direction_hidden = self.direction_output_norm(
            direction_state[:, self.cfg.latent_slots :]
        )
        scale_hidden = self.scale_output_norm(scale_state[:, self.cfg.latent_slots :])
        direction_logits = self.direction_head(direction_hidden).view(
            batch,
            128,
            128 // self.output_patch_size,
            self.output_patch_size,
        )
        raw_log_scale = self.scale_head(scale_hidden).view(
            batch,
            128,
            128 // self.output_patch_size,
        )

        component_mask = (
            d_out_mask[:, :, None, None]
            & p16_component_mask[:, None, :, :]
        )
        direction_logits = direction_logits * component_mask.to(
            dtype=direction_logits.dtype
        )
        # Put epsilon inside sqrt so masked all-zero p16 tokens have a finite,
        # exactly-zero Jacobian instead of the undefined derivative of sqrt(0).
        direction_norm = (
            direction_logits.float().square().sum(dim=-1, keepdim=True) + 1.0e-12
        ).sqrt()
        if hasattr(self, "direction_head_frobenius_radius"):
            self._last_direction_raw_logit_norms = direction_norm.squeeze(-1).detach()
            self._last_direction_valid_mask = component_mask.any(dim=-1).detach()
        pred_dirs = direction_logits.float() / direction_norm
        pred_dirs = pred_dirs * component_mask.to(dtype=pred_dirs.dtype)

        pred_log_scales = self._bound_log_scale(raw_log_scale.float())
        p16_valid = component_mask.any(dim=-1)
        pred_log_scales = pred_log_scales * p16_valid.to(dtype=pred_log_scales.dtype)
        patches = pred_dirs.float() * torch.exp(pred_log_scales).unsqueeze(-1)
        patches = patches * component_mask.to(dtype=patches.dtype)
        prediction = patches.reshape(batch, 128, 128).transpose(1, 2).contiguous()
        return prediction, pred_dirs, pred_log_scales

    def forward(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_index: torch.Tensor,
        activation_context: torch.Tensor | None = None,
        *,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        tile_col: torch.Tensor | None = None,
        activation_sample_mask: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        capture_depth: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[dict[str, float]],
        torch.Tensor,
        torch.Tensor,
    ]:
        if token_valid_mask is None:
            raise ValueError("polar-tail forward requires token_valid_mask")
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
        prediction, pred_dirs, pred_log_scales = self.decode_polar(
            z,
            tile_index,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid_mask,
            dist_patch_by_patch=dist_patch_by_patch,
        )
        return prediction, z, telemetry, pred_dirs, pred_log_scales


class GaugeFixedPolarTailedUnifiedWeightBottleneck(
    PolarTailedUnifiedWeightBottleneck
):
    """Polar model with one non-affine, fixed-radius direction readout."""

    def __init__(self, cfg: ExperimentConfig, arm: str) -> None:
        super().__init__(cfg, arm)
        learned_norm = self.direction_output_norm
        if not hasattr(learned_norm, "weight") or not hasattr(learned_norm, "eps"):
            raise TypeError("gauge-fixed direction head requires affine RMSNorm")
        if self.direction_head.bias is not None:
            raise ValueError("gauge-fixed direction head must remain bias-free")

        with torch.no_grad():
            composite = self.direction_head.weight.float() * learned_norm.weight.detach().float()[
                None, :
            ]
            radius = composite.norm()
            # Meta construction is used only for cheap architecture/parameter
            # contracts.  A meta scalar has the right persistent buffer shape
            # but cannot be materialized with item()/bool().  Every executable
            # CPU/CUDA construction still validates the exact seeded radius.
            if not radius.is_meta and (
                not bool(torch.isfinite(radius)) or float(radius) <= 0.0
            ):
                raise RuntimeError("seeded composite direction head has invalid radius")
            self.direction_head.weight.copy_(composite.to(self.direction_head.weight.dtype))

        self.direction_output_norm = AffineFreeRMSNorm(
            cfg.hidden_dim,
            eps=float(learned_norm.eps),
        )
        self.register_buffer(
            "direction_head_frobenius_radius",
            radius.detach().clone(),
            persistent=True,
        )


class LatentRootedPolarTailedUnifiedWeightBottleneck(
    PolarTailedUnifiedWeightBottleneck
):
    """Polar model whose decoded values descend exclusively from encoder latents.

    Position, tile and activation context construct attention queries. They are
    never added to the decoded value stream. The first shared state is a
    bias-free Q<-K,V(latent) read, and each polar tail is rooted by an analogous
    Q<-K,V(shared-state) read.
    """

    @staticmethod
    def _cross_attention_write(
        block: nn.Module,
        queries: torch.Tensor,
        values: torch.Tensor,
        value_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if block.context_key_projection is not None:
            raise RuntimeError("latent-rooted decoder blocks cannot own key sidechannels")
        batch, query_length, dim = queries.shape
        value_length = values.shape[1]
        query_weight, key_weight, value_weight = block.qkv.weight.split(dim, dim=0)
        query = F.linear(block.attn_norm(queries), query_weight).view(
            batch, query_length, block.heads, block.head_dim
        )
        normalized_values = block.attn_norm(values)
        key = F.linear(normalized_values, key_weight).view(
            batch, value_length, block.heads, block.head_dim
        )
        value = F.linear(normalized_values, value_weight).view(
            batch, value_length, block.heads, block.head_dim
        )
        query_heads = query.transpose(1, 2)
        key_heads = key.transpose(1, 2)
        if block.bounded_cosine_attention:
            query_heads = F.normalize(
                query_heads.float(), dim=-1, eps=1.0e-6
            ).to(query.dtype)
            key_heads = F.normalize(
                key_heads.float(), dim=-1, eps=1.0e-6
            ).to(key.dtype)
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value.transpose(1, 2),
            attn_mask=(
                value_valid[:, None, None, :] if value_valid is not None else None
            ),
            dropout_p=0.0,
            scale=(
                block.attention_logit_scale
                if block.bounded_cosine_attention
                else None
            ),
        )
        return block.attn_out(
            attended.transpose(1, 2).reshape(batch, query_length, dim)
        )

    @classmethod
    def _latent_rooted_block_forward(
        cls,
        block: nn.Module,
        queries: torch.Tensor,
        values: torch.Tensor,
        query_valid: torch.Tensor,
        value_valid: torch.Tensor,
    ) -> torch.Tensor:
        # Deliberately no `+ queries`: addresses never become decoded values.
        state = block._bounded_write(
            cls._cross_attention_write(block, queries, values, value_valid)
        )
        left, right = block.mlp_in(block.mlp_norm(state)).chunk(2, dim=-1)
        hidden = F.silu(left) * right
        if block.bounded_swiglu_hidden:
            hidden_rms_sq = hidden.float().square().mean(dim=-1, keepdim=True)
            hidden = hidden * torch.rsqrt(1.0 + hidden_rms_sq).to(hidden.dtype)
        state = state + block._bounded_write(block.mlp_out(hidden))
        return state * query_valid.unsqueeze(-1).to(state.dtype)

    def _run_latent_rooted_block(
        self,
        block: nn.Module,
        queries: torch.Tensor,
        values: torch.Tensor,
        query_valid: torch.Tensor,
        value_valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.activation_checkpointing and self.training:
            return checkpoint(
                lambda q, v: self._latent_rooted_block_forward(
                    block, q, v, query_valid, value_valid
                ),
                queries,
                values,
                use_reentrant=False,
            )
        return self._latent_rooted_block_forward(
            block, queries, values, query_valid, value_valid
        )

    def _latent_rooted_queries(
        self,
        z: torch.Tensor,
        tile_index: torch.Tensor,
        tile_col: torch.Tensor | None,
        token_valid_mask: torch.Tensor,
        dist_patch_by_patch: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.output_queries[None].expand(z.shape[0], -1, -1)
        queries = queries + self._tile_position_embedding(
            tile_index, tile_col
        )[:, None]
        decoder_context = dist_patch_by_patch.unsqueeze(1).expand(
            -1, 128, -1, -1
        ).reshape(z.shape[0], self.weight_token_count, self.cfg.distribution_d_dist)
        queries = self.decoder_query_conditioner(
            torch.cat((queries, decoder_context.to(queries.dtype)), dim=-1)
        )
        return queries * token_valid_mask.unsqueeze(-1).to(queries.dtype)

    def decode_polar(
        self,
        z: torch.Tensor,
        tile_index: torch.Tensor,
        *,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        tile_col: torch.Tensor | None,
        token_valid_mask: torch.Tensor,
        dist_patch_by_patch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dist_patch_by_patch is None:
            raise ValueError("latent-rooted decoder requires distribution context")
        batch = z.shape[0]
        token_valid_mask = token_valid_mask.to(device=z.device, dtype=torch.bool)
        p16_component_mask = d_in_mask.view(
            batch, 128 // self.output_patch_size, self.output_patch_size
        )
        component_mask = (
            d_out_mask[:, :, None, None] & p16_component_mask[:, None, :, :]
        )
        p16_valid = component_mask.any(dim=-1).reshape(batch, -1)

        latent = self.from_latent(z)
        latent_valid = torch.ones(
            batch,
            self.cfg.latent_slots,
            device=z.device,
            dtype=torch.bool,
        )
        p32_queries = self._latent_rooted_queries(
            z,
            tile_index,
            tile_col,
            token_valid_mask,
            dist_patch_by_patch,
        )
        shared_state = self._run_latent_rooted_block(
            self.decoder_blocks[0],
            p32_queries,
            latent,
            token_valid_mask,
            latent_valid,
        )
        for block in self.decoder_blocks[1:]:
            shared_state = self._run_decoder_block(
                block, shared_state, token_valid_mask
            )

        base_child_queries = p32_queries[:, :, None, :].expand(-1, -1, 2, -1)

        def run_tail(
            block: nn.Module,
            half_embedding: torch.Tensor,
            output_norm: nn.Module,
        ) -> torch.Tensor:
            child_queries = (
                base_child_queries
                + half_embedding[None, None].to(base_child_queries.dtype)
            ).reshape(batch, self.weight_token_count * 2, self.cfg.hidden_dim)
            child_queries = child_queries * p16_valid.unsqueeze(-1).to(
                child_queries.dtype
            )
            child_state = self._run_latent_rooted_block(
                block,
                child_queries,
                shared_state,
                p16_valid,
                token_valid_mask,
            )
            return output_norm(child_state)

        direction_hidden = run_tail(
            self.direction_tail,
            self.direction_half_embedding,
            self.direction_output_norm,
        )
        scale_hidden = run_tail(
            self.scale_tail,
            self.scale_half_embedding,
            self.scale_output_norm,
        )
        direction_logits = self.direction_head(direction_hidden).view(
            batch, 128, 128 // self.output_patch_size, self.output_patch_size
        )
        raw_log_scale = self.scale_head(scale_hidden).view(
            batch, 128, 128 // self.output_patch_size
        )
        direction_logits = direction_logits * component_mask.to(
            direction_logits.dtype
        )
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
        prediction = patches.reshape(batch, 128, 128).transpose(1, 2).contiguous()
        return prediction, pred_dirs, pred_log_scales


def _build_production_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.AdamW:
    common = {
        "lr": float(config["learning_rate"]),
        "betas": tuple(float(value) for value in config["betas"]),
        "eps": float(config["eps"]),
    }
    if not isinstance(model, GaugeFixedPolarTailedUnifiedWeightBottleneck):
        return torch.optim.AdamW(
            model.parameters(),
            weight_decay=float(config["weight_decay"]),
            **common,
        )

    direction_weight = model.direction_head.weight
    main_parameters = [
        parameter for parameter in model.parameters() if parameter is not direction_weight
    ]
    if len(main_parameters) + 1 != sum(1 for _ in model.parameters()):
        raise RuntimeError("gauge-fixed optimizer parameter partition drifted")
    return torch.optim.AdamW(
        [
            {
                "params": main_parameters,
                "weight_decay": float(config["weight_decay"]),
                "group_name": "main_decay",
            },
            {
                "params": [direction_weight],
                "weight_decay": 0.0,
                "group_name": "direction_head_frobenius_sphere",
            },
        ],
        weight_decay=0.0,
        **common,
    )


@torch.no_grad()
def _retract_direction_head_and_tangent_momentum_(
    model: GaugeFixedPolarTailedUnifiedWeightBottleneck,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    if not isinstance(model, GaugeFixedPolarTailedUnifiedWeightBottleneck):
        raise TypeError("direction-head retraction requires the gauge-fixed model")
    weight = model.direction_head.weight
    radius = model.direction_head_frobenius_radius.float()
    weight_float = weight.float()
    norm = weight_float.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise RuntimeError("direction head left the finite nonzero Frobenius sphere")
    weight.mul_((radius / norm).to(dtype=weight.dtype))

    state = optimizer.state.get(weight, {})
    first_moment = state.get("exp_avg")
    if first_moment is not None:
        weight_float = weight.float()
        moment_float = first_moment.float()
        radial_coefficient = (weight_float * moment_float).sum() / radius.square()
        first_moment.sub_(
            (radial_coefficient * weight_float).to(dtype=first_moment.dtype)
        )

    weight_float = weight.float()
    actual_norm = weight_float.norm()
    relative_error = (actual_norm - radius).abs() / radius
    moment_radial_ratio = weight_float.new_zeros(())
    if first_moment is not None:
        moment_float = first_moment.float()
        moment_radial_ratio = (weight_float * moment_float).sum().abs() / (
            actual_norm * moment_float.norm() + 1.0e-30
        )
    if float(relative_error) > 2.0e-6:
        raise RuntimeError("direction-head Frobenius retraction failed")
    if float(moment_radial_ratio) > 2.0e-6:
        raise RuntimeError("direction-head first moment is not tangent")
    return {
        "direction_head_frobenius_norm": float(actual_norm),
        "direction_head_frobenius_target": float(radius),
        "direction_head_frobenius_relative_error": float(relative_error),
        "direction_head_first_moment_radial_ratio": float(moment_radial_ratio),
    }


@torch.no_grad()
def _direction_gauge_telemetry(
    model: GaugeFixedPolarTailedUnifiedWeightBottleneck,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    weight = model.direction_head.weight.float()
    radius = model.direction_head_frobenius_radius.float()
    singular_values = torch.linalg.svdvals(weight)
    spectral = singular_values.max()
    frobenius = weight.norm()
    stable_rank = frobenius.square() / spectral.square().clamp_min(1.0e-30)
    row_norms = weight.norm(dim=1)
    state = optimizer.state.get(model.direction_head.weight, {})
    first_moment = state.get("exp_avg")
    radial_ratio = weight.new_zeros(())
    if first_moment is not None:
        moment = first_moment.float()
        radial_ratio = (weight * moment).sum().abs() / (
            frobenius * moment.norm() + 1.0e-30
        )
    metrics = {
        "direction_head_frobenius_norm": float(frobenius),
        "direction_head_frobenius_target": float(radius),
        "direction_head_frobenius_relative_error": float(
            (frobenius - radius).abs() / radius
        ),
        "direction_head_spectral_norm": float(spectral),
        "direction_head_stable_rank": float(stable_rank),
        "direction_head_row_norm_min": float(row_norms.min()),
        "direction_head_row_norm_median": float(row_norms.median()),
        "direction_head_row_norm_max": float(row_norms.max()),
        "direction_head_first_moment_radial_ratio": float(radial_ratio),
    }
    raw_norms = getattr(model, "_last_direction_raw_logit_norms", None)
    raw_valid = getattr(model, "_last_direction_valid_mask", None)
    if raw_norms is not None and raw_valid is not None:
        valid_norms = raw_norms.float()[raw_valid]
        if valid_norms.numel() > 0:
            quantiles = torch.quantile(
                valid_norms,
                torch.tensor([0.01, 0.50, 0.99], device=valid_norms.device),
            )
            metrics.update(
                {
                    "direction_raw_logit_norm_min": float(valid_norms.min()),
                    "direction_raw_logit_norm_p01": float(quantiles[0]),
                    "direction_raw_logit_norm_median": float(quantiles[1]),
                    "direction_raw_logit_norm_p99": float(quantiles[2]),
                    "direction_raw_logit_norm_max": float(valid_norms.max()),
                }
            )
    return metrics


def _production_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    W_hat: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
    *,
    pred_dirs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    behavioral_coef = float(loss_cfg["behavioral_coef"])
    if behavioral_coef == 0.0:
        # Structural-only means the activation/operator objective is absent from
        # both the scalar loss and the autograd graph, rather than merely being
        # evaluated and multiplied by zero.
        operator = W_hat.new_zeros(())
        behavioral_dir = W_hat.new_zeros(())
        behavioral_scale = W_hat.new_zeros(())
        behavioral = W_hat.new_zeros(())
    else:
        behavioral_operator_coef = float(loss_cfg["behavioral_operator"])
        if behavioral_operator_coef == 0.0:
            operator = W_hat.new_zeros(())
        else:
            operator = BigWeightVAELossMixin.operator_recon_loss(
                X,
                W,
                W_hat,
                x_mask=x_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
            )
        behavioral_dir, behavioral_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
            X,
            W,
            W_hat,
            x_mask=x_mask,
            d_out_mask=d_out_mask,
            gamma=0.5,
            huber_delta=0.1,
        )
        behavioral = (
            behavioral_operator_coef * operator
            + float(loss_cfg["behavioral_direction"]) * behavioral_dir
            + float(loss_cfg["behavioral_scale"]) * behavioral_scale
        )
    structural, structural_parts = BigWeightVAELossMixin.patch_structure_loss(
        W,
        W_hat,
        patch_size=16,
        gamma=0.5,
        lambda_dir=float(loss_cfg["structural_direction"]),
        lambda_scale=float(loss_cfg["structural_scale"]),
        lambda_rec=float(loss_cfg["structural_reconstruction"]),
        lambda_rel=float(loss_cfg["structural_relational"]),
        huber_delta=0.1,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        pred_dirs=pred_dirs,
    )
    total = (
        behavioral_coef * behavioral
        + float(loss_cfg["structural_coef"]) * structural
    )
    return total, {
        "behavioral": behavioral.detach(),
        "behavioral_operator": operator.detach(),
        "behavioral_direction": behavioral_dir.detach(),
        "behavioral_scale": behavioral_scale.detach(),
        "structural": structural.detach(),
        "structural_direction": structural_parts["L_dir"],
        "structural_scale": structural_parts["L_scale"],
    }


def _weights_from_polar_components(
    pred_dirs: torch.Tensor,
    pred_log_scales: torch.Tensor,
    *,
    detach_direction: bool,
    detach_scale: bool,
) -> torch.Tensor:
    if pred_dirs.ndim != 4 or pred_dirs.shape[-2:] != (8, 16):
        raise ValueError("pred_dirs must be [B,d_out,8,16]")
    if tuple(pred_log_scales.shape) != tuple(pred_dirs.shape[:-1]):
        raise ValueError("pred_log_scales must match polar direction patches")
    direction = pred_dirs.detach() if detach_direction else pred_dirs
    log_scale = pred_log_scales.detach() if detach_scale else pred_log_scales
    patches = direction.float() * torch.exp(log_scale.float()).unsqueeze(-1)
    return patches.reshape(pred_dirs.shape[0], pred_dirs.shape[1], 128).transpose(
        1,
        2,
    ).contiguous()


def _polar_routed_production_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    pred_dirs: torch.Tensor,
    pred_log_scales: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep production scalar losses while separating their polar backward paths."""
    if float(loss_cfg["behavioral_operator"]) != 0.0:
        raise ValueError("polar routed loss requires behavioral operator loss to be disabled")
    direction_prediction = _weights_from_polar_components(
        pred_dirs,
        pred_log_scales,
        detach_direction=False,
        detach_scale=True,
    )
    scale_prediction = _weights_from_polar_components(
        pred_dirs,
        pred_log_scales,
        detach_direction=True,
        detach_scale=False,
    )
    behavioral_dir, _unused_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
        direction_prediction,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )
    _unused_dir, behavioral_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
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


def _polar_direction_scale_objectives(
    X: torch.Tensor,
    W: torch.Tensor,
    pred_dirs: torch.Tensor,
    pred_log_scales: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact weighted direction and scale pieces of the polar loss."""
    if float(loss_cfg["behavioral_operator"]) != 0.0:
        raise ValueError("polar component telemetry requires operator loss to be disabled")
    direction_prediction = _weights_from_polar_components(
        pred_dirs,
        pred_log_scales,
        detach_direction=False,
        detach_scale=True,
    )
    scale_prediction = _weights_from_polar_components(
        pred_dirs,
        pred_log_scales,
        detach_direction=True,
        detach_scale=False,
    )
    behavioral_dir, _unused_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
        direction_prediction,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )
    _unused_dir, behavioral_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
        scale_prediction,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )
    structural_dir, _ = BigWeightVAELossMixin.patch_structure_loss(
        W,
        direction_prediction,
        patch_size=16,
        gamma=0.5,
        lambda_dir=1.0,
        lambda_scale=0.0,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=0.1,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        pred_dirs=pred_dirs,
    )
    structural_scale, _ = BigWeightVAELossMixin.patch_structure_loss(
        W,
        scale_prediction,
        patch_size=16,
        gamma=0.5,
        lambda_dir=0.0,
        lambda_scale=1.0,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=0.1,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        pred_dirs=pred_dirs.detach(),
    )
    direction = (
        float(loss_cfg["behavioral_coef"])
        * float(loss_cfg["behavioral_direction"])
        * behavioral_dir
        + float(loss_cfg["structural_coef"])
        * float(loss_cfg["structural_direction"])
        * structural_dir
    )
    scale = (
        float(loss_cfg["behavioral_coef"])
        * float(loss_cfg["behavioral_scale"])
        * behavioral_scale
        + float(loss_cfg["structural_coef"])
        * float(loss_cfg["structural_scale"])
        * structural_scale
    )
    return direction, scale


def _gradient_pair_summary(
    direction_grad: torch.Tensor | None,
    scale_grad: torch.Tensor | None,
) -> dict[str, float | bool | None]:
    if direction_grad is None and scale_grad is None:
        return {
            "direction_gradient_rms": None,
            "scale_gradient_rms": None,
            "direction_scale_cosine": None,
            "scale_over_direction": None,
            "conflict": False,
        }
    if direction_grad is None:
        if scale_grad is None:
            raise AssertionError("unreachable")
        direction_grad = torch.zeros_like(scale_grad)
    if scale_grad is None:
        scale_grad = torch.zeros_like(direction_grad)
    direction = direction_grad.detach().float().reshape(-1)
    scale = scale_grad.detach().float().reshape(-1)
    direction_rms = direction.square().mean().sqrt()
    scale_rms = scale.square().mean().sqrt()
    denominator = direction_rms * scale_rms
    cosine = (
        (direction * scale).mean() / denominator.clamp_min(1.0e-30)
        if float(denominator) > 0.0
        else direction.new_zeros(())
    )
    return {
        "direction_gradient_rms": float(direction_rms.item()),
        "scale_gradient_rms": float(scale_rms.item()),
        "direction_scale_cosine": float(cosine.item()),
        "scale_over_direction": (
            float((scale_rms / direction_rms).item())
            if float(direction_rms) > 0.0
            else None
        ),
        "conflict": bool(float(cosine) < 0.0),
    }


def _latent_anticollapse_objective(
    latent: torch.Tensor,
    *,
    margin: float = 0.10,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep sample-by-slot interaction dispersion above a small floor.

    Two-way centering removes sample-only and slot-only codes before measuring
    dispersion.  The detached global RMS makes the auxiliary scale-invariant
    without letting it shrink useful components that were projected away.
    """
    if latent.ndim != 3:
        raise ValueError("latent anti-collapse objective requires z=[B,slots,features]")
    if latent.shape[0] < 2 or latent.shape[1] < 2:
        raise ValueError("latent anti-collapse objective requires B>=2 and slots>=2")
    if not (math.isfinite(margin) and margin > 0.0):
        raise ValueError("latent anti-collapse margin must be finite and positive")
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError("latent anti-collapse eps must be finite and positive")

    z = latent.float()
    interaction = (
        z
        - z.mean(dim=1, keepdim=True)
        - z.mean(dim=0, keepdim=True)
        + z.mean(dim=(0, 1), keepdim=True)
    )
    denominator = (z.square().mean() + float(eps)).sqrt().detach()
    normalized = interaction / denominator
    cross_std = (
        normalized.square().mean(dim=(0, 2)) + float(eps)
    ).sqrt()
    within_std = (
        normalized.square().mean(dim=(1, 2)) + float(eps)
    ).sqrt()
    cross_hinge = F.relu(float(margin) - cross_std)
    within_hinge = F.relu(float(margin) - within_std)
    loss = 0.5 * (cross_hinge.mean() + within_hinge.mean())

    detached_cross = cross_std.detach()
    detached_within = within_std.detach()
    detached_normalized = normalized.detach()
    stats = {
        "latent_anticollapse_loss": loss.detach(),
        "latent_rms": denominator.detach(),
        "latent_interaction_rms_ratio": (
            detached_normalized.square().mean() + float(eps)
        ).sqrt(),
        "latent_cross_std_min": detached_cross.min(),
        "latent_cross_std_p10": torch.quantile(detached_cross, 0.10),
        "latent_cross_std_median": detached_cross.median(),
        "latent_cross_active_fraction": (
            detached_cross < float(margin)
        ).float().mean(),
        "latent_within_std_min": detached_within.min(),
        "latent_within_std_p10": torch.quantile(detached_within, 0.10),
        "latent_within_std_median": detached_within.median(),
        "latent_within_active_fraction": (
            detached_within < float(margin)
        ).float().mean(),
    }
    return loss, stats


def _exact_layout_groups(
    tile_row: torch.Tensor,
    tile_col: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    *,
    minimum_group_size: int = 2,
) -> list[list[int]]:
    """Group batch entries whose output-direction fields have identical semantics."""
    if minimum_group_size < 2:
        raise ValueError("exact-layout groups require minimum_group_size>=2")
    batch = int(tile_row.numel())
    if int(tile_col.numel()) != batch:
        raise ValueError("tile_row and tile_col must have the same batch length")
    if tuple(d_in_mask.shape) != (batch, 128):
        raise ValueError("d_in_mask must be [B,128]")
    if tuple(d_out_mask.shape) != (batch, 128):
        raise ValueError("d_out_mask must be [B,128]")

    rows = tile_row.detach().to(device="cpu").reshape(-1).tolist()
    cols = tile_col.detach().to(device="cpu").reshape(-1).tolist()
    in_masks = d_in_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    out_masks = d_out_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index in range(batch):
        key = (
            int(rows[index]),
            int(cols[index]),
            tuple(in_masks[index]),
            tuple(out_masks[index]),
        )
        grouped.setdefault(key, []).append(index)
    return [
        indices
        for indices in grouped.values()
        if len(indices) >= minimum_group_size
    ]


def _same_layout_direction_infonce(
    pred_dirs: torch.Tensor,
    W: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    layout_groups: list[list[int]],
    *,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match each predicted direction field to its own target within one layout.

    Negatives have exactly the same tile coordinates and masks, so layout alone
    cannot solve the classification task.  Targets are detached; gradients act
    only on the final direction predictions and their upstream path.
    """
    if pred_dirs.ndim != 4 or tuple(pred_dirs.shape[1:]) != (128, 8, 16):
        raise ValueError("pred_dirs must be [B,128,8,16]")
    batch = int(pred_dirs.shape[0])
    if tuple(W.shape) != (batch, 128, 128):
        raise ValueError("W must be [B,128,128]")
    if tuple(d_in_mask.shape) != (batch, 128):
        raise ValueError("d_in_mask must be [B,128]")
    if tuple(d_out_mask.shape) != (batch, 128):
        raise ValueError("d_out_mask must be [B,128]")
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError("direction InfoNCE temperature must be finite and positive")

    pred = pred_dirs.float()
    component_mask = (
        d_out_mask[:, :, None, None]
        & d_in_mask.view(batch, 1, 8, 16)
    )
    target_patches = W.float().transpose(1, 2).reshape(batch, 128, 8, 16)
    target_patches = target_patches * component_mask.to(target_patches.dtype)
    target_norm = (
        target_patches.square().sum(dim=-1, keepdim=True) + 1.0e-12
    ).sqrt()
    target_dirs = (
        target_patches / target_norm
    ).mul(component_mask.to(target_patches.dtype)).detach()

    group_losses: list[torch.Tensor] = []
    diagonal_similarities: list[torch.Tensor] = []
    offdiagonal_similarities: list[torch.Tensor] = []
    hardest_negative_gaps: list[torch.Tensor] = []
    correct = pred.new_zeros(())
    eligible_samples = 0
    group_sizes: list[int] = []
    for raw_indices in layout_groups:
        if len(raw_indices) < 2:
            raise ValueError("layout_groups may not contain singleton groups")
        if len(set(raw_indices)) != len(raw_indices):
            raise ValueError("layout group contains duplicate sample indices")
        if min(raw_indices) < 0 or max(raw_indices) >= batch:
            raise ValueError("layout group index is outside the batch")
        indices = torch.tensor(raw_indices, device=pred.device, dtype=torch.long)
        group_pred = pred.index_select(0, indices).reshape(len(raw_indices), -1, 16)
        group_target = target_dirs.index_select(0, indices).reshape(
            len(raw_indices), -1, 16
        )
        valid_patches = component_mask.index_select(0, indices)[0].reshape(-1, 16).any(-1)
        if not bool(valid_patches.any()):
            raise ValueError("exact-layout direction group has no valid patches")
        group_pred = group_pred[:, valid_patches]
        group_target = group_target[:, valid_patches]
        similarities = torch.einsum(
            "ikd,jkd->ij", group_pred, group_target
        ) / float(valid_patches.sum().item())
        labels = torch.arange(len(raw_indices), device=pred.device)
        group_losses.append(F.cross_entropy(similarities / float(temperature), labels))
        diagonal = similarities.diagonal()
        negative_mask = ~torch.eye(
            len(raw_indices), device=pred.device, dtype=torch.bool
        )
        negatives = similarities[negative_mask].reshape(len(raw_indices), -1)
        diagonal_similarities.append(diagonal.detach())
        offdiagonal_similarities.append(negatives.detach().reshape(-1))
        hardest_negative_gaps.append(
            (diagonal - negatives.max(dim=1).values).detach()
        )
        correct = correct + (similarities.argmax(dim=1) == labels).float().sum()
        eligible_samples += len(raw_indices)
        group_sizes.append(len(raw_indices))

    if group_losses:
        loss = torch.stack(group_losses).mean()
        diagonal_mean = torch.cat(diagonal_similarities).mean()
        offdiagonal_mean = torch.cat(offdiagonal_similarities).mean()
        hardest_gap_mean = torch.cat(hardest_negative_gaps).mean()
        top1_accuracy = correct / float(eligible_samples)
        sorted_sizes = sorted(group_sizes)
        group_size_median = float(sorted_sizes[(len(sorted_sizes) - 1) // 2])
        group_size_min = float(sorted_sizes[0])
        group_size_max = float(sorted_sizes[-1])
    else:
        # Keep the zero connected to the direction graph.  This makes sparse
        # eligibility explicit without silently changing the batch contract.
        loss = pred.sum() * 0.0
        diagonal_mean = pred.new_zeros(())
        offdiagonal_mean = pred.new_zeros(())
        hardest_gap_mean = pred.new_zeros(())
        top1_accuracy = pred.new_zeros(())
        group_size_median = 0.0
        group_size_min = 0.0
        group_size_max = 0.0

    stats = {
        "direction_contrastive_loss": loss.detach(),
        "direction_contrastive_group_count": pred.new_tensor(float(len(group_losses))),
        "direction_contrastive_eligible_samples": pred.new_tensor(
            float(eligible_samples)
        ),
        "direction_contrastive_eligible_fraction": pred.new_tensor(
            float(eligible_samples) / float(batch)
        ),
        "direction_contrastive_group_size_min": pred.new_tensor(group_size_min),
        "direction_contrastive_group_size_median": pred.new_tensor(
            group_size_median
        ),
        "direction_contrastive_group_size_max": pred.new_tensor(group_size_max),
        "direction_contrastive_diagonal_similarity_mean": diagonal_mean,
        "direction_contrastive_offdiagonal_similarity_mean": offdiagonal_mean,
        "direction_contrastive_diagonal_minus_hardest_negative_mean": hardest_gap_mean,
        "direction_contrastive_top1_accuracy": top1_accuracy.detach(),
    }
    return loss, stats


def _three_way_gradient_summary(
    direction_grad: torch.Tensor | None,
    scale_grad: torch.Tensor | None,
    anticollapse_grad: torch.Tensor | None,
    direction_contrastive_grad: torch.Tensor | None = None,
) -> dict[str, float | bool | None]:
    row = _gradient_pair_summary(direction_grad, scale_grad)
    available = next(
        (
            grad
            for grad in (
                direction_grad,
                scale_grad,
                anticollapse_grad,
                direction_contrastive_grad,
            )
            if grad is not None
        ),
        None,
    )
    if available is None:
        row.update(
            {
                "latent_anticollapse_gradient_rms": None,
                "anticollapse_direction_cosine": None,
                "anticollapse_scale_cosine": None,
                "anticollapse_over_direction": None,
                "anticollapse_over_scale": None,
                "direction_contrastive_gradient_rms": None,
                "contrastive_direction_cosine": None,
                "contrastive_scale_cosine": None,
                "contrastive_anticollapse_cosine": None,
                "contrastive_over_direction": None,
                "contrastive_over_scale": None,
                "contrastive_over_anticollapse": None,
            }
        )
        return row

    direction = (
        torch.zeros_like(available) if direction_grad is None else direction_grad
    ).detach().float().reshape(-1)
    scale = (
        torch.zeros_like(available) if scale_grad is None else scale_grad
    ).detach().float().reshape(-1)
    anticollapse = (
        torch.zeros_like(available)
        if anticollapse_grad is None
        else anticollapse_grad
    ).detach().float().reshape(-1)
    contrastive = (
        torch.zeros_like(available)
        if direction_contrastive_grad is None
        else direction_contrastive_grad
    ).detach().float().reshape(-1)

    def rms(vector: torch.Tensor) -> torch.Tensor:
        return vector.square().mean().sqrt()

    def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        denominator = rms(left) * rms(right)
        if float(denominator) == 0.0:
            return left.new_zeros(())
        return (left * right).mean() / denominator.clamp_min(1.0e-30)

    direction_rms = rms(direction)
    scale_rms = rms(scale)
    anticollapse_rms = rms(anticollapse)
    contrastive_rms = rms(contrastive)
    row.update(
        {
            "latent_anticollapse_gradient_rms": float(anticollapse_rms.item()),
            "anticollapse_direction_cosine": float(
                cosine(anticollapse, direction).item()
            ),
            "anticollapse_scale_cosine": float(cosine(anticollapse, scale).item()),
            "anticollapse_over_direction": (
                float((anticollapse_rms / direction_rms).item())
                if float(direction_rms) > 0.0
                else None
            ),
            "anticollapse_over_scale": (
                float((anticollapse_rms / scale_rms).item())
                if float(scale_rms) > 0.0
                else None
            ),
            "direction_contrastive_gradient_rms": float(contrastive_rms.item()),
            "contrastive_direction_cosine": float(
                cosine(contrastive, direction).item()
            ),
            "contrastive_scale_cosine": float(cosine(contrastive, scale).item()),
            "contrastive_anticollapse_cosine": float(
                cosine(contrastive, anticollapse).item()
            ),
            "contrastive_over_direction": (
                float((contrastive_rms / direction_rms).item())
                if float(direction_rms) > 0.0
                else None
            ),
            "contrastive_over_scale": (
                float((contrastive_rms / scale_rms).item())
                if float(scale_rms) > 0.0
                else None
            ),
            "contrastive_over_anticollapse": (
                float((contrastive_rms / anticollapse_rms).item())
                if float(anticollapse_rms) > 0.0
                else None
            ),
        }
    )
    return row


def _polar_component_gradient_telemetry(
    model: PolarTailedUnifiedWeightBottleneck,
    direction_loss: torch.Tensor,
    scale_loss: torch.Tensor,
    anticollapse_loss: torch.Tensor | None = None,
    direction_contrastive_loss: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure actual weighted loss conflict on representative layer tensors."""
    named_parameters: list[tuple[str, torch.Tensor, bool]] = []
    for depth, block in enumerate(model.encoder_blocks, start=1):
        named_parameters.append((f"encoder_{depth:02d}.qkv", block.qkv.weight, True))
    for depth, block in enumerate(model.decoder_blocks, start=1):
        named_parameters.append((f"shared_decoder_{depth:02d}.qkv", block.qkv.weight, True))
    named_parameters.extend(
        [
            ("direction_tail.qkv", model.direction_tail.qkv.weight, True),
            ("scale_tail.qkv", model.scale_tail.qkv.weight, True),
            ("to_latent.weight", model.to_latent.weight, False),
            ("from_latent.weight", model.from_latent.weight, False),
            ("direction_head.weight", model.direction_head.weight, False),
            ("scale_head.weight", model.scale_head.weight, False),
        ]
    )
    parameters = [parameter for _name, parameter, _is_qkv in named_parameters]
    direction_grads = torch.autograd.grad(
        direction_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    scale_grads = torch.autograd.grad(
        scale_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    anticollapse_grads: tuple[torch.Tensor | None, ...]
    if anticollapse_loss is None:
        anticollapse_grads = tuple(None for _parameter in parameters)
    else:
        anticollapse_grads = torch.autograd.grad(
            anticollapse_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    direction_contrastive_grads: tuple[torch.Tensor | None, ...]
    if direction_contrastive_loss is None:
        direction_contrastive_grads = tuple(None for _parameter in parameters)
    else:
        direction_contrastive_grads = torch.autograd.grad(
            direction_contrastive_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    rows: dict[str, dict[str, float | bool | None]] = {}
    for (
        (name, _parameter, is_qkv),
        direction_grad,
        scale_grad,
        anticollapse_grad,
        direction_contrastive_grad,
    ) in zip(
        named_parameters,
        direction_grads,
        scale_grads,
        anticollapse_grads,
        direction_contrastive_grads,
        strict=True,
    ):
        if is_qkv and any(
            grad is not None
            for grad in (
                direction_grad,
                scale_grad,
                anticollapse_grad,
                direction_contrastive_grad,
            )
        ):
            if direction_grad is None:
                if (
                    scale_grad is None
                    and anticollapse_grad is None
                    and direction_contrastive_grad is None
                ):
                    raise AssertionError("unreachable")
                direction_grad = torch.zeros_like(
                    next(
                        grad
                        for grad in (
                            scale_grad,
                            anticollapse_grad,
                            direction_contrastive_grad,
                        )
                        if grad is not None
                    )
                )
            if scale_grad is None:
                scale_grad = torch.zeros_like(direction_grad)
            if anticollapse_grad is None:
                anticollapse_grad = torch.zeros_like(direction_grad)
            if direction_contrastive_grad is None:
                direction_contrastive_grad = torch.zeros_like(direction_grad)
            direction_parts = direction_grad.chunk(3, dim=0)
            scale_parts = scale_grad.chunk(3, dim=0)
            anticollapse_parts = anticollapse_grad.chunk(3, dim=0)
            direction_contrastive_parts = direction_contrastive_grad.chunk(3, dim=0)
            for (
                suffix,
                current_direction,
                current_scale,
                current_anticollapse,
                current_direction_contrastive,
            ) in zip(
                ("q", "k", "v"),
                direction_parts,
                scale_parts,
                anticollapse_parts,
                direction_contrastive_parts,
                strict=True,
            ):
                rows[f"{name}.{suffix}"] = _three_way_gradient_summary(
                    current_direction,
                    current_scale,
                    current_anticollapse,
                    current_direction_contrastive,
                )
        else:
            rows[name] = _three_way_gradient_summary(
                direction_grad,
                scale_grad,
                anticollapse_grad,
                direction_contrastive_grad,
            )
    return rows


def _latent_objective_gradient_telemetry(
    latent: torch.Tensor,
    direction_loss: torch.Tensor,
    scale_loss: torch.Tensor,
    anticollapse_loss: torch.Tensor,
    direction_contrastive_loss: torch.Tensor | None = None,
) -> dict[str, float | bool | None]:
    objectives = [direction_loss, scale_loss, anticollapse_loss]
    if direction_contrastive_loss is not None:
        objectives.append(direction_contrastive_loss)
    gradients = [
        torch.autograd.grad(
            objective,
            latent,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        for objective in objectives
    ]
    if direction_contrastive_loss is None:
        gradients.append(None)
    return _three_way_gradient_summary(*gradients)


def _gradient_balanced_objectives(
    X: torch.Tensor,
    W: torch.Tensor,
    W_hat: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return the raw direction and scale objectives without task coefficients."""
    behavioral_dir, behavioral_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
        X,
        W,
        W_hat,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=0.5,
        huber_delta=0.1,
    )
    structural_dir, _ = BigWeightVAELossMixin.patch_structure_loss(
        W,
        W_hat,
        patch_size=16,
        gamma=0.5,
        lambda_dir=1.0,
        lambda_scale=0.0,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=0.1,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
    )
    structural_scale, _ = BigWeightVAELossMixin.patch_structure_loss(
        W,
        W_hat,
        patch_size=16,
        gamma=0.5,
        lambda_dir=0.0,
        lambda_scale=1.0,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=0.1,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
    )
    direction = behavioral_dir + structural_dir
    scale = behavioral_scale + structural_scale
    return direction, scale, {
        "behavioral": behavioral_dir.detach() + behavioral_scale.detach(),
        "behavioral_operator": W_hat.new_zeros(()),
        "behavioral_direction": behavioral_dir.detach(),
        "behavioral_scale": behavioral_scale.detach(),
        "structural": structural_dir.detach() + structural_scale.detach(),
        "structural_direction": structural_dir.detach(),
        "structural_scale": structural_scale.detach(),
    }


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _bottleneck_gradient_balanced_backward(
    direction_loss: torch.Tensor,
    scale_loss: torch.Tensor,
    latent: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Backpropagate the equal-vote direction/scale bisector at the bottleneck.

    The common gradient magnitude matches the raw equal-weight objective
    `direction_loss + scale_loss`; only the task direction is rebalanced.
    """
    direction_grad = torch.autograd.grad(
        direction_loss,
        latent,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0].float()
    scale_grad = torch.autograd.grad(
        scale_loss,
        latent,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0].float()
    if not bool(torch.isfinite(direction_grad).all()) or not bool(
        torch.isfinite(scale_grad).all()
    ):
        raise RuntimeError("nonfinite direction/scale bottleneck gradient")

    direction_rms = _rms(direction_grad)
    scale_rms = _rms(scale_grad)
    if float(direction_rms) <= eps or float(scale_rms) <= eps:
        raise RuntimeError(
            "gradient balancing requires nonzero direction and scale bottleneck gradients"
        )
    dot = (direction_grad * scale_grad).mean()
    cosine = dot / (direction_rms * scale_rms).clamp_min(eps)
    normalized_sum = direction_grad / direction_rms + scale_grad / scale_rms
    normalized_sum_rms = _rms(normalized_sum)
    if float(normalized_sum_rms) <= eps:
        raise RuntimeError(
            "direction and scale output gradients are exactly antagonistic; "
            "no shared descent direction exists"
        )
    reference_rms = _rms(direction_grad + scale_grad)
    balanced_grad_proxy = normalized_sum * (reference_rms / normalized_sum_rms)
    if not bool(torch.isfinite(balanced_grad_proxy).all()):
        raise RuntimeError("nonfinite balanced bottleneck gradient proxy")

    direction_weight = reference_rms / (normalized_sum_rms * direction_rms)
    scale_weight = reference_rms / (normalized_sum_rms * scale_rms)
    balanced_loss = direction_weight.detach() * direction_loss + scale_weight.detach() * scale_loss
    balanced_loss.backward()
    return {
        "direction_bottleneck_grad_rms": float(direction_rms.item()),
        "scale_bottleneck_grad_rms": float(scale_rms.item()),
        "direction_scale_bottleneck_grad_cosine": float(cosine.item()),
        "raw_equal_sum_bottleneck_grad_rms": float(reference_rms.item()),
        "balanced_bottleneck_grad_proxy_rms": float(_rms(balanced_grad_proxy).item()),
        "direction_dynamic_weight": float(direction_weight.item()),
        "scale_dynamic_weight": float(scale_weight.item()),
    }


def _plot_metrics(path: Path, output: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(steps, [row["loss"] for row in rows], label="production total", linewidth=1.5)
    if all("legacy_task_loss" in row for row in rows):
        ax.plot(
            steps,
            [row["legacy_task_loss"] for row in rows],
            label="legacy task objective",
            linewidth=1.35,
        )
    if any(float(row.get("latent_anticollapse_weighted_loss", 0.0)) > 0.0 for row in rows):
        ax.plot(
            steps,
            [row.get("latent_anticollapse_weighted_loss", 0.0) for row in rows],
            label="latent anti-collapse",
            linewidth=1.1,
        )
    if any(
        float(row.get("direction_contrastive_weighted_loss", 0.0)) > 0.0
        for row in rows
    ):
        ax.plot(
            steps,
            [row.get("direction_contrastive_weighted_loss", 0.0) for row in rows],
            label="direction prediction InfoNCE",
            linewidth=1.1,
        )
    ax.plot(steps, [row["structural"] for row in rows], label="structural", alpha=0.85)
    ax.plot(steps, [row["behavioral"] for row in rows], label="behavioral", alpha=0.85)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss")
    ax.set_title("Direct Normalized p32 700M — production training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    model_cfg = _model_config(config)
    run_schema = str(config["schema"])
    balanced_backward = run_schema == BOUNDED_BALANCED_SCHEMA
    polar_tails = run_schema in {
        POLAR_TAIL_SCHEMA,
        LATENT_ROOTED_POLAR_TAIL_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA,
        POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA,
    }
    latent_rooted = run_schema == LATENT_ROOTED_POLAR_TAIL_SCHEMA
    latent_anticollapse = run_schema in {
        POLAR_TAIL_LATENT_ANTICOLLAPSE_SCHEMA,
        POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA,
        POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA,
    }
    direction_infonce = run_schema in {
        POLAR_TAIL_LATENT_ANTICOLLAPSE_DIRECTION_INFONCE_SCHEMA,
        POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA,
    }
    gauge_fixed_direction = (
        run_schema == POLAR_TAIL_GAUGE_FIXED_DIRECTION_INFONCE_SCHEMA
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_root is not None:
        output_root = args.output_root.resolve()
    elif args.smoke:
        output_root = Path(
            f"/mnt/shared/weightclip_benchmark/direct_normalized_p32_production_smoke_{stamp}"
        )
    else:
        output_root = Path(config["output_root"]).resolve()
    steps = 2 if args.smoke else int(config["steps"])
    batch_size = int(config["batch_size"])
    resume_path = Path(config["resume_checkpoint"]).resolve()
    persistent_model_path = Path(config["persistent_model_checkpoint"]).resolve()

    startup = {
        "schema": run_schema,
        "stage": "preflight",
        "config_path": str(config_path),
        "resolved_config": config,
        "model_config": asdict(model_cfg),
        "device": config["device"],
        "dtype": "bfloat16 autocast; FP32 parameters/loss statistics",
        "seed": int(config["seed"]),
        "output_root": str(output_root),
        "steps": steps,
        "scientific_horizon_steps": int(config["steps"]),
        "resume_checkpoint": str(resume_path),
        "persistent_model_checkpoint": str(persistent_model_path),
    }
    print("[direct-normalized-production] stage=preflight", json.dumps(startup), flush=True)
    if args.dry_run:
        print("[direct-normalized-production] stage=complete mode=dry-run", flush=True)
        return
    if config.get("resume_learning_rate_transition") is not None and not args.resume:
        raise ValueError("the LR-transition config requires --resume")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"fresh output root already exists: {output_root}")
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    output_root.mkdir(parents=True, exist_ok=True)
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

    print("[direct-normalized-production] stage=model-build", flush=True)
    if latent_rooted:
        model: UnifiedWeightBottleneck = (
            LatentRootedPolarTailedUnifiedWeightBottleneck(
                model_cfg, "normalized_float"
            )
        ).to(device)
    elif gauge_fixed_direction:
        model = GaugeFixedPolarTailedUnifiedWeightBottleneck(
            model_cfg, "normalized_float"
        ).to(device)
    elif polar_tails:
        model = PolarTailedUnifiedWeightBottleneck(
            model_cfg, "normalized_float"
        ).to(device)
    else:
        model = UnifiedWeightBottleneck(model_cfg, "normalized_float").to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if gauge_fixed_direction:
        expected_parameters = POLAR_TAIL_GAUGE_FIXED_EXPECTED_PARAMETERS
    elif polar_tails:
        expected_parameters = POLAR_TAIL_EXPECTED_PARAMETERS
    else:
        expected_parameters = LEGACY_EXPECTED_PARAMETERS
    if parameter_count != expected_parameters:
        raise RuntimeError(
            "production parameter count drifted: "
            f"{parameter_count} != {expected_parameters}"
        )
    optimizer = _build_production_optimizer(model, config)
    start_step = 0
    committed_logical_index = 0
    resume_lr_event: dict[str, Any] | None = None
    if args.resume:
        print(f"[direct-normalized-production] stage=resume-load path={resume_path}", flush=True)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if config.get("resume_learning_rate_transition") is not None:
            checkpoint_config = dict(payload["config"])
            requested_config = dict(config)
            checkpoint_lr = float(checkpoint_config.pop("learning_rate"))
            requested_config.pop("learning_rate")
            checkpoint_config.pop("resume_learning_rate_transition", None)
            requested_config.pop("resume_learning_rate_transition", None)
            if checkpoint_config != requested_config:
                raise RuntimeError(
                    "LR-transition resume changed fields other than the declared LR contract"
                )
            transition = config["resume_learning_rate_transition"]
            if checkpoint_lr not in {
                float(transition["source_learning_rate"]),
                float(transition["target_learning_rate"]),
            }:
                raise RuntimeError(
                    "checkpoint config LR is outside the declared transition: "
                    f"{checkpoint_lr}"
                )
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["step"])
        committed_logical_index = int(payload["committed_logical_index"])
        if committed_logical_index != start_step * batch_size:
            raise RuntimeError("resume logical cursor disagrees with step and batch size")
        resume_lr_event = _apply_resume_learning_rate(
            optimizer,
            config,
            checkpoint_step=start_step,
        )
        _restore_rng_state(payload["rng_state"])
        del payload
        stopped_path = output_root / "STOPPED.json"
        if stopped_path.is_file():
            stopped_payload = json.loads(stopped_path.read_text(encoding="utf-8"))
            if int(stopped_payload["step"]) != start_step:
                raise RuntimeError(
                    "STOPPED marker step disagrees with resume checkpoint: "
                    f"{stopped_payload['step']} != {start_step}"
                )
            archived_stopped_path = output_root / f"STOPPED.step_{start_step:09d}.json"
            if archived_stopped_path.exists():
                raise FileExistsError(
                    f"archived STOPPED marker already exists: {archived_stopped_path}"
                )
            stopped_path.replace(archived_stopped_path)
        resume_config_path = (
            output_root / f"resolved_resume_config_step_{start_step:09d}.json"
        )
        if resume_config_path.exists():
            raise FileExistsError(
                f"resolved resume config already exists: {resume_config_path}"
            )
        _atomic_json(resume_config_path, startup)
        _append_jsonl(
            output_root / "resume_events.jsonl",
            {
                "schema": run_schema,
                **resume_lr_event,
                "committed_logical_index": committed_logical_index,
                "resolved_resume_config": str(resume_config_path),
                "resumed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            "[direct-normalized-production] stage=resume-lr "
            f"checkpoint_step={start_step} "
            f"previous={resume_lr_event['previous_learning_rates']} "
            f"effective={resume_lr_event['effective_learning_rate']} "
            f"optimizer_state_entries={resume_lr_event.get('optimizer_state_entries')}",
            flush=True,
        )
    effective_lrs = {
        float(group["lr"])
        for group in optimizer.param_groups
    }
    if effective_lrs != {float(config["learning_rate"])}:
        raise RuntimeError(
            "effective optimizer LR disagrees with config after model/resume setup: "
            f"{sorted(effective_lrs)}"
        )
    effective_learning_rate = next(iter(effective_lrs))
    if gauge_fixed_direction:
        initial_gauge_stats = _direction_gauge_telemetry(model, optimizer)
        if initial_gauge_stats["direction_head_frobenius_relative_error"] > 2.0e-6:
            raise RuntimeError(
                "gauge-fixed direction head is off its Frobenius sphere before training"
            )
        if initial_gauge_stats["direction_head_first_moment_radial_ratio"] > 2.0e-6:
            raise RuntimeError(
                "gauge-fixed direction-head Adam first moment is not tangent before training"
            )
        print(
            "[direct-normalized-production] stage=direction-gauge-ready "
            f"radius={initial_gauge_stats['direction_head_frobenius_target']:.9g} "
            f"relative_error="
            f"{initial_gauge_stats['direction_head_frobenius_relative_error']:.3e} "
            f"stable_rank={initial_gauge_stats['direction_head_stable_rank']:.6f}",
            flush=True,
        )
    print(
        "[direct-normalized-production] stage=model-build-complete "
        f"parameters={parameter_count} start_step={start_step} "
        f"logical_index={committed_logical_index}",
        flush=True,
    )

    stop_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[direct-normalized-production] stage=stop-requested signal={signum}; "
            "will save after current optimizer step",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    logger = logging.getLogger("direct-normalized-production")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    bank = config["operator_bank"]
    metrics_path = output_root / "train_metrics.jsonl"
    gradient_path = output_root / "gradient_telemetry.jsonl"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def save_resume(step: int, logical_index: int) -> None:
        print(
            f"[direct-normalized-production] stage=resume-save step={step} path={resume_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": run_schema,
                "step": step,
                "committed_logical_index": logical_index,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state(),
                "config": config,
            },
            resume_path,
        )

    def save_persistent_model(step: int) -> None:
        print(
            "[direct-normalized-production] stage=model-checkpoint-save "
            f"step={step} path={persistent_model_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": run_schema,
                "step": step,
                "model_state": model.state_dict(),
                "model_config": asdict(model_cfg),
                "config": config,
            },
            persistent_model_path,
        )

    print(
        "[direct-normalized-production] stage=data-open "
        f"pair_manifest={bank['pair_manifest']} workers={bank['loader_workers']} "
        f"permutation_views={bank['permutation_views']} "
        f"canonical_probability={bank['canonical_probability']}",
        flush=True,
    )
    with ExitStack() as stack:
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
        sampler.set_start_index(committed_logical_index)
        loader_workers = int(bank["loader_workers"])
        loader_batch_size = int(bank["loader_batch_size"])
        loader = DataLoader(
            dataset,
            batch_size=(None if loader_batch_size <= 1 else loader_batch_size),
            sampler=sampler,
            num_workers=loader_workers,
            collate_fn=(
                _identity_sample_collate
                if loader_batch_size <= 1
                else _sample_list_collate
            ),
            prefetch_factor=(int(bank["loader_prefetch_factor"]) if loader_workers else None),
            persistent_workers=bool(loader_workers),
            pin_memory=False,
            worker_init_fn=_offline_loader_worker_init_fn,
            generator=torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 7_919),
        )
        loader_iter = iter(loader)
        bundle_iter = loader_iter if loader_batch_size <= 1 else _flatten_loader_batches(loader_iter)
        stream = BalancedOperatorBankMixer(
            dataset,
            bundle_iter,
            start_index=committed_logical_index,
        )
        shutdown_workers = getattr(loader_iter, "_shutdown_workers", None)
        if callable(shutdown_workers):
            stack.callback(shutdown_workers)
        if balanced_backward:
            backward_description = (
                "backward=bottleneck-gradient-RMS-bisector "
                "tasks=(behavioral+structural)-direction,(behavioral+structural)-scale"
            )
        else:
            backward_description = (
                f"behavioral_coef={config['loss']['behavioral_coef']} "
                f"behavioral={config['loss']['behavioral_operator']}*operator+"
                f"{config['loss']['behavioral_direction']}*direction+"
                f"{config['loss']['behavioral_scale']}*log-scale "
                "structural=1*direction+10*log-scale"
            )
            if polar_tails:
                backward_description += " backward=coordinate-owned-polar-routing"
            if latent_anticollapse:
                anti_cfg = config["latent_anticollapse"]
                backward_description += (
                    " latent_anticollapse=two-way-centered-rms-hinge"
                    f"(margin={anti_cfg['margin']},lambda={anti_cfg['coefficient']},"
                    "q=detached,scope=physical-B32)"
                )
            if direction_infonce:
                contrastive_cfg = config["direction_contrastive"]
                backward_description += (
                    " direction_contrastive=same-layout-prediction-target-InfoNCE"
                    f"(temperature={contrastive_cfg['temperature']},"
                    f"lambda={contrastive_cfg['coefficient']},target=detached)"
                )
        print(
            "[direct-normalized-production] stage=train "
            f"steps={steps} batch={batch_size} lr={effective_learning_rate} constant "
            f"objective={config.get('objective', 'behavioral_plus_structural')} "
            + backward_description,
            flush=True,
        )

        for step in range(start_step + 1, steps + 1):
            cpu_batch = _batch_from_stream(stream, batch_size)
            layout_groups = (
                _exact_layout_groups(
                    cpu_batch["tile_row"],
                    cpu_batch["tile_col"],
                    cpu_batch["d_in_mask"],
                    cpu_batch["d_out_mask"],
                    minimum_group_size=int(
                        config["direction_contrastive"]["minimum_group_size"]
                    ),
                )
                if direction_infonce
                else []
            )
            W = cpu_batch["W"].to(device, non_blocking=True)
            X = cpu_batch["X"].to(device, non_blocking=True)
            x_mask = cpu_batch["x_mask"].to(device, non_blocking=True)
            d_in_mask = cpu_batch["d_in_mask"].to(device, non_blocking=True)
            d_out_mask = cpu_batch["d_out_mask"].to(device, non_blocking=True)
            tile_row = cpu_batch["tile_row"].to(device, non_blocking=True)
            tile_col = cpu_batch["tile_col"].to(device, non_blocking=True)
            content, log_scale, token_valid = _prepare_normalized_inputs(
                W,
                d_in_mask,
                d_out_mask,
                scale_mean=float(config["normalization"]["log2_scale_mean"]),
                scale_std=float(config["normalization"]["log2_scale_std"]),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gradient_due = step == 1 or step % int(config["gradient_log_every"]) == 0
            component_direction_loss: torch.Tensor | None = None
            component_scale_loss: torch.Tensor | None = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_dirs: torch.Tensor | None = None
                pred_log_scales: torch.Tensor | None = None
                if polar_tails:
                    (
                        prediction,
                        _latent,
                        _telemetry,
                        pred_dirs,
                        pred_log_scales,
                    ) = model(
                        content,
                        log_scale,
                        tile_row,
                        X,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                        tile_col=tile_col,
                        activation_sample_mask=x_mask,
                        token_valid_mask=token_valid,
                    )
                else:
                    prediction, _latent, _telemetry = model(
                        content,
                        log_scale,
                        tile_row,
                        X,
                        tile_col=tile_col,
                        activation_sample_mask=x_mask,
                        token_valid_mask=token_valid,
                    )
                value_mask = d_in_mask[:, :, None] & d_out_mask[:, None, :]
                prediction = prediction * value_mask.to(dtype=prediction.dtype)
                if balanced_backward:
                    direction_loss, scale_loss, parts = _gradient_balanced_objectives(
                        X,
                        W,
                        prediction,
                        x_mask,
                        d_in_mask,
                        d_out_mask,
                    )
                    legacy_task_loss = direction_loss + scale_loss
                else:
                    if polar_tails:
                        if pred_dirs is None or pred_log_scales is None:
                            raise RuntimeError("polar model did not return output coordinates")
                        legacy_task_loss, parts = _polar_routed_production_loss(
                            X,
                            W,
                            pred_dirs,
                            pred_log_scales,
                            x_mask,
                            d_in_mask,
                            d_out_mask,
                            config["loss"],
                        )
                    else:
                        legacy_task_loss, parts = _production_loss(
                            X,
                            W,
                            prediction,
                            x_mask,
                            d_in_mask,
                            d_out_mask,
                            config["loss"],
                            pred_dirs=pred_dirs,
                        )
                    if polar_tails and gradient_due:
                        if pred_dirs is None or pred_log_scales is None:
                            raise RuntimeError("polar model did not return output coordinates")
                        component_direction_loss, component_scale_loss = (
                            _polar_direction_scale_objectives(
                                X,
                                W,
                                pred_dirs,
                                pred_log_scales,
                                x_mask,
                                d_in_mask,
                                d_out_mask,
                                config["loss"],
                            )
                        )
            anticollapse_stats: dict[str, float] = {}
            weighted_anticollapse_loss = legacy_task_loss.new_zeros(())
            if latent_anticollapse:
                anti_cfg = config["latent_anticollapse"]
                with torch.autocast(device_type="cuda", enabled=False):
                    anticollapse_loss, raw_anticollapse_stats = (
                        _latent_anticollapse_objective(
                            _latent,
                            margin=float(anti_cfg["margin"]),
                            eps=float(anti_cfg["eps"]),
                        )
                    )
                weighted_anticollapse_loss = (
                    float(anti_cfg["coefficient"]) * anticollapse_loss
                )
                anticollapse_stats = {
                    key: float(value.item())
                    for key, value in raw_anticollapse_stats.items()
                }
            direction_contrastive_stats: dict[str, float] = {}
            weighted_direction_contrastive_loss = legacy_task_loss.new_zeros(())
            if direction_infonce:
                if pred_dirs is None:
                    raise RuntimeError("polar model did not return direction predictions")
                contrastive_cfg = config["direction_contrastive"]
                with torch.autocast(device_type="cuda", enabled=False):
                    direction_contrastive_loss, raw_direction_contrastive_stats = (
                        _same_layout_direction_infonce(
                            pred_dirs,
                            W,
                            d_in_mask,
                            d_out_mask,
                            layout_groups,
                            temperature=float(contrastive_cfg["temperature"]),
                        )
                    )
                weighted_direction_contrastive_loss = (
                    float(contrastive_cfg["coefficient"])
                    * direction_contrastive_loss
                )
                direction_contrastive_stats = {
                    key: float(value.item())
                    for key, value in raw_direction_contrastive_stats.items()
                }
            loss = (
                legacy_task_loss
                + weighted_anticollapse_loss
                + weighted_direction_contrastive_loss
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite production loss at step {step}")
            balance_stats: dict[str, float] = {}
            component_gradient_rows: dict[str, Any] = {}
            component_stats: dict[str, float] = {}
            latent_gradient_stats: dict[str, float | bool | None] = {}
            if polar_tails and gradient_due:
                if component_direction_loss is None or component_scale_loss is None:
                    raise RuntimeError("polar component objectives were not constructed")
                component_sum = component_direction_loss + component_scale_loss
                component_parity = float(
                    (
                        component_sum.detach().float()
                        - legacy_task_loss.detach().float()
                    ).abs().item()
                )
                if component_parity > 1.0e-4:
                    raise RuntimeError(
                        "polar direction/scale objective decomposition drifted: "
                        f"absolute error={component_parity}"
                    )
                component_gradient_rows = _polar_component_gradient_telemetry(
                    model,
                    component_direction_loss,
                    component_scale_loss,
                    (
                        weighted_anticollapse_loss
                        if latent_anticollapse
                        else None
                    ),
                    (
                        weighted_direction_contrastive_loss
                        if direction_infonce
                        else None
                    ),
                )
                if latent_anticollapse:
                    latent_gradient_stats = _latent_objective_gradient_telemetry(
                        _latent,
                        component_direction_loss,
                        component_scale_loss,
                        weighted_anticollapse_loss,
                        (
                            weighted_direction_contrastive_loss
                            if direction_infonce
                            else None
                        ),
                    )
                component_stats = {
                    "weighted_direction_objective": float(
                        component_direction_loss.detach().item()
                    ),
                    "weighted_scale_objective": float(component_scale_loss.detach().item()),
                    "component_sum_parity_max_abs": component_parity,
                    "legacy_plus_auxiliaries_parity_max_abs": float(
                        (
                            loss.detach().float()
                            - (
                                legacy_task_loss.detach().float()
                                + weighted_anticollapse_loss.detach().float()
                                + weighted_direction_contrastive_loss.detach().float()
                            )
                        ).abs().item()
                    ),
                }
                if not direction_infonce:
                    component_stats["legacy_plus_anticollapse_parity_max_abs"] = (
                        component_stats["legacy_plus_auxiliaries_parity_max_abs"]
                    )
            if balanced_backward:
                balance_stats = _bottleneck_gradient_balanced_backward(
                    direction_loss,
                    scale_loss,
                    _latent,
                )
            else:
                loss.backward()
            if gradient_due:
                gradient_row = {
                    "schema": run_schema,
                    "step": step,
                    **balance_stats,
                    **component_stats,
                    "latent_objective_gradients": latent_gradient_stats,
                    "direction_scale_component_gradients": component_gradient_rows,
                    "groups": _gradient_telemetry(model),
                }
                if gauge_fixed_direction:
                    gradient_row.update(_direction_gauge_telemetry(model, optimizer))
                _append_jsonl(gradient_path, gradient_row)
                if step == 1:
                    missing = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad and parameter.grad is None
                    ]
                    unexpected_missing = [
                        name
                        for name in missing
                        if name != "extra_tile_row_embedding.weight"
                    ]
                    if unexpected_missing:
                        raise RuntimeError(
                            "production model has unexpected missing gradients at step 1: "
                            f"{unexpected_missing}"
                        )
                    for depth in range(1, model_cfg.encoder_depth + 1):
                        if gradient_row["groups"][f"encoder_block_{depth}"]["gradient_rms"] <= 0:
                            raise RuntimeError(f"encoder block {depth} is gradient-dead")
                    decoder_depth = (
                        model.shared_decoder_depth if polar_tails else model_cfg.decoder_depth
                    )
                    for depth in range(1, decoder_depth + 1):
                        if gradient_row["groups"][f"decoder_block_{depth}"]["gradient_rms"] <= 0:
                            raise RuntimeError(f"decoder block {depth} is gradient-dead")
                    if polar_tails:
                        for group in (
                            "direction_tail",
                            "scale_tail",
                            "direction_head",
                            "scale_head",
                        ):
                            if gradient_row["groups"][group]["gradient_rms"] <= 0:
                                raise RuntimeError(f"polar group {group} is gradient-dead")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["grad_clip_norm"])
                ).item()
            )
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite gradient norm at step {step}")
            optimizer.step()
            if gauge_fixed_direction:
                _retract_direction_head_and_tangent_momentum_(model, optimizer)
            committed_logical_index = int(cpu_batch["logical_indices"][-1]) + 1

            if step == 1 or step % int(config["log_every"]) == 0:
                elapsed = time.monotonic() - started
                gauge_stats = (
                    _direction_gauge_telemetry(model, optimizer)
                    if gauge_fixed_direction
                    else {}
                )
                row = {
                    "schema": run_schema,
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "legacy_task_loss": float(legacy_task_loss.detach().item()),
                    "latent_anticollapse_weighted_loss": float(
                        weighted_anticollapse_loss.detach().item()
                    ),
                    "direction_contrastive_weighted_loss": float(
                        weighted_direction_contrastive_loss.detach().item()
                    ),
                    **anticollapse_stats,
                    **direction_contrastive_stats,
                    **{key: float(value.item()) for key, value in parts.items()},
                    "grad_norm_pre_clip": grad_norm,
                    "learning_rate": effective_learning_rate,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": step / max(elapsed, 1.0e-9),
                    "committed_logical_index": committed_logical_index,
                    "cuda_peak_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                    **balance_stats,
                    **gauge_stats,
                }
                _append_jsonl(metrics_path, row)
                print(
                    "[direct-normalized-production] stage=train "
                    f"step={step}/{steps} loss={row['loss']:.6f} "
                    f"legacy={row['legacy_task_loss']:.6f} "
                    f"anti={row['latent_anticollapse_weighted_loss']:.6f} "
                    f"direction_nce={row['direction_contrastive_weighted_loss']:.6f} "
                    f"behavioral={row['behavioral']:.6f} structural={row['structural']:.6f} "
                    f"grad={grad_norm:.4e} lr={effective_learning_rate:.3e} "
                    f"rate={row['steps_per_second']:.3f}_steps_per_s "
                    f"peak={row['cuda_peak_gib']:.2f}GiB",
                    flush=True,
                )
            if step % int(config["plot_every"]) == 0:
                _plot_metrics(metrics_path, output_root / "train_loss_curve.png")
            if not args.smoke and step % int(config["resume_save_every"]) == 0:
                save_resume(step, committed_logical_index)
            if not args.smoke and step % int(config["model_save_every"]) == 0:
                save_persistent_model(step)
            if stop_requested:
                if not args.smoke:
                    save_resume(step, committed_logical_index)
                    save_persistent_model(step)
                _atomic_json(
                    output_root / "STOPPED.json",
                    {
                        "schema": run_schema,
                        "step": step,
                        "committed_logical_index": committed_logical_index,
                    },
                )
                print(f"[direct-normalized-production] stage=stopped step={step}", flush=True)
                return

    if not args.smoke:
        save_resume(steps, committed_logical_index)
        save_persistent_model(steps)
    _plot_metrics(metrics_path, output_root / "train_loss_curve.png")
    summary = {
        "schema": run_schema,
        "complete": True,
        "step": steps,
        "committed_logical_index": committed_logical_index,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "metrics_path": str(metrics_path),
        "plot_path": str(output_root / "train_loss_curve.png"),
    }
    _atomic_json(output_root / "COMPLETE.json", summary)
    print("[direct-normalized-production] stage=complete", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
