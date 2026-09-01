from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from big_vae.datasets.operator_bank import (
    BalancedOperatorBankMixer,
    OperatorBankBundle,
    OperatorBankSample,
    OperatorBundleRequest,
    OperatorBankTrainingDataset,
)
from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    RMSNorm,
    _gradient_telemetry,
    _seed_everything,
)
from training.weightclip_benchmark.run_mini_polar_regression_production import (
    EXPECTED_PARAMETERS,
    MiniPolarConfig,
    MiniPolarWeightBottleneck,
    SubtileCursor,
    ValidSubtileStream,
    _append_jsonl,
    _atomic_json,
    _atomic_torch_save,
    _load_config as _load_baseline_config,
    _masked_weight_rmse,
    _normalization_artifact,
    _open_subtile_stream,
    _polar_routed_loss,
    _prepare_normalized_inputs,
    _restore_rng_state,
    _rng_state,
)


SCHEMA = "weightclip_mini_polar_latent_preference_production_v1"
EXPECTED_PROJECTOR_PARAMETERS = 229_808
DEFAULT_CONFIG = Path(
    "conf/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_production_500k.yaml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mini polar AE with matched representation and decoder preference losses."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retained-checkpoint-every",
        type=int,
        default=0,
        help="Retain a full resume checkpoint at every N optimizer steps; 0 disables retention.",
    )
    parser.add_argument(
        "--retained-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for immutable step-addressed full resume checkpoints.",
    )
    return parser.parse_args()


def _atomic_hardlink(source: Path, destination: Path) -> None:
    """Atomically retain the just-written resume inode without serializing it twice."""

    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise RuntimeError(
            "retained checkpoints must share a filesystem with resume_latest for atomic hardlinks"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, got {config.get('schema')!r}")

    # Reuse the already-tested baseline validator by changing only the schema
    # and removing the new intervention fields.  This fail-closes on every
    # architecture, optimizer and historical loss coefficient.
    baseline_view = {
        key: value
        for key, value in config.items()
        if key not in {"pairing", "representation", "preference", "calibration", "diagnostics"}
    }
    baseline_view["schema"] = "weightclip_mini_polar_regression_production_v1"
    temporary = path.with_name(f".{path.name}.baseline-validation-{hashlib.sha256(str(path).encode()).hexdigest()[:8]}.yaml")
    try:
        temporary.write_text(yaml.safe_dump(baseline_view, sort_keys=False), encoding="utf-8")
        _load_baseline_config(temporary)
    finally:
        temporary.unlink(missing_ok=True)

    expected_pairing = {
        "kind": "different_lineage_same_checkpoint_index_exact_layout_canonical",
        "pairs_per_batch": 64,
        "canonical_only": True,
        "pair_bundle_cache": 4,
        "background_prefetch_batches": 4,
        "pin_memory": True,
    }
    expected_repr = {
        "kind": "symmetric_binary_matched_infonce",
        "projector_hidden_dim": 256,
        "projector_output_dim": 128,
        "dropout": 0.1,
        "temperature": 0.1,
    }
    expected_pref = {
        "kind": "relative_softplus_margin",
        "margin": 0.5,
        "temperature": 0.1,
        "eps": 1.0e-6,
        "foreign_latent_detached": True,
        "direct_conditioning_detached": True,
    }
    expected_calibrations = (
        {
            "kind": "fixed_first_paired_batch_gradient_ratio",
            "preference_target_ratio": 0.25,
            "representation_target_encoder_ratio": 0.10,
            "preference_coefficient_min": 0.02,
            "preference_coefficient_max": 10.0,
            "representation_coefficient_min": 0.001,
            "representation_coefficient_max": 0.20,
            "ramp_steps": 1000,
            "auxiliary_gradient_ratio_stop": 0.75,
        },
        {
            "kind": "fixed_first_paired_batch_gradient_ratio",
            "preference_target_ratio": 0.025,
            "representation_target_encoder_ratio": 0.10,
            "preference_coefficient_min": 0.02,
            "preference_coefficient_max": 10.0,
            "representation_coefficient_min": 0.001,
            "representation_coefficient_max": 0.20,
            "ramp_steps": 10000,
            "auxiliary_gradient_ratio_stop": 0.75,
        },
    )
    expected_diagnostics = {"fixed_probe_every": 1000}
    for key, expected in (
        ("pairing", expected_pairing),
        ("representation", expected_repr),
        ("preference", expected_pref),
        ("diagnostics", expected_diagnostics),
    ):
        if config.get(key) != expected:
            raise ValueError(f"production {key} contract drifted: {config.get(key)!r} != {expected!r}")
    if config.get("calibration") not in expected_calibrations:
        raise ValueError(
            "production calibration contract drifted: "
            f"{config.get('calibration')!r} not in {expected_calibrations!r}"
        )
    if int(config["batch_size"]) != 2 * int(config["pairing"]["pairs_per_batch"]):
        raise ValueError("physical batch must contain exactly two members per hard pair")
    bank = config["operator_bank"]
    if bool(bank["permutation_views"]) or float(bank["canonical_probability"]) != 1.0:
        raise ValueError("exact cross-lineage p32 pairing requires canonical-only graph gauge")
    return config


class LatentRepresentationProjector(nn.Module):
    def __init__(self, cfg: MiniPolarConfig, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.latent_dim)
        self.first = nn.Linear(cfg.latent_slots * cfg.latent_dim, int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.second = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(z.float()).flatten(1)
        hidden = self.dropout(F.gelu(self.first(hidden)))
        return F.normalize(self.second(hidden), dim=-1, eps=1.0e-8)


def _binary_matched_infonce(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] % 2:
        raise ValueError("representation views must be aligned [2*pairs,dim]")
    even = torch.arange(0, first.shape[0], 2, device=first.device)
    odd = even + 1

    def term(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_similarity = (anchor * positive).sum(dim=-1)
        negative_similarity = (anchor * negative).sum(dim=-1)
        loss = F.softplus((negative_similarity - positive_similarity) / float(temperature))
        return loss, positive_similarity, negative_similarity

    terms = (
        term(first[even], second[even], second[odd]),
        term(second[even], first[even], first[odd]),
        term(first[odd], second[odd], second[even]),
        term(second[odd], first[odd], first[even]),
    )
    losses = torch.cat([row[0] for row in terms])
    positives = torch.cat([row[1] for row in terms])
    negatives = torch.cat([row[2] for row in terms])
    return losses.mean(), {
        "positive_cosine": positives.detach().mean(),
        "negative_cosine": negatives.detach().mean(),
        "pairwise_accuracy": (positives.detach() > negatives.detach()).float().mean(),
    }


def _polar_routed_loss_per_example(
    X: torch.Tensor,
    W: torch.Tensor,
    pred_dirs: torch.Tensor,
    pred_log_scales: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
    *,
    eps: float = 1.0e-8,
    gamma: float = 0.5,
    huber_delta: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Per-example form of the exact routed polar objective.

    The ordinary positive/base scalar continues to use the historical helper.
    This vector is used only to compare the same target under its own versus a
    matched foreign latent.  For B=1 every component is exactly the historical
    component, which is covered by the focused contract test.
    """

    if W.ndim != 3 or tuple(W.shape[-2:]) != (32, 32):
        raise ValueError("preference routed loss requires W=[B,32,32]")
    batch = W.shape[0]
    component_mask = (
        d_out_mask[:, :, None, None]
        & d_in_mask.view(batch, 2, 16)[:, None, :, :]
    )
    target_patches = W.transpose(1, 2).contiguous().view(batch, 32, 2, 16)
    target_patches = target_patches * component_mask.to(target_patches.dtype)
    target_radius = target_patches.float().norm(dim=-1)
    target_direction = target_patches.float() / (target_radius.unsqueeze(-1) + float(eps))
    element_mask = component_mask.any(dim=-1)
    valid_outputs = element_mask.any(dim=-1)

    masked_pred_dirs = pred_dirs.float() * component_mask.to(pred_dirs.dtype)
    normalized_pred_dirs = masked_pred_dirs / (
        masked_pred_dirs.norm(dim=-1, keepdim=True) + float(eps)
    )
    direction_cosine = (normalized_pred_dirs * target_direction).sum(dim=-1)
    direction_error = 1.0 - direction_cosine.clamp(min=-1.0, max=1.0)
    direction_weight = (target_radius + float(eps)).pow(float(gamma))
    direction_weight = direction_weight * element_mask.to(direction_weight.dtype)
    direction_weight = direction_weight / direction_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    per_output_direction = (direction_error * direction_weight).sum(dim=-1)
    structural_direction = (
        per_output_direction * valid_outputs.to(per_output_direction.dtype)
    ).sum(dim=-1) / valid_outputs.sum(dim=-1).clamp_min(1)

    predicted_scale_patches = masked_pred_dirs.detach() * torch.exp(
        pred_log_scales.float()
    ).unsqueeze(-1)
    predicted_radius = predicted_scale_patches.norm(dim=-1)
    scale_delta = torch.log(predicted_radius + float(eps)) - torch.log(
        target_radius + float(eps)
    )
    scale_abs = scale_delta.abs()
    scale_huber = torch.where(
        scale_abs <= float(huber_delta),
        0.5 * scale_delta.square(),
        float(huber_delta) * (scale_abs - 0.5 * float(huber_delta)),
    )
    structural_scale = (
        scale_huber * element_mask.to(scale_huber.dtype)
    ).sum(dim=(1, 2)) / element_mask.sum(dim=(1, 2)).clamp_min(1)

    direction_weights = pred_dirs.float() * torch.exp(pred_log_scales.detach().float()).unsqueeze(-1)
    direction_prediction = direction_weights.reshape(batch, 32, 32).transpose(1, 2).contiguous()
    scale_weights = pred_dirs.detach().float() * torch.exp(pred_log_scales.float()).unsqueeze(-1)
    scale_prediction = scale_weights.reshape(batch, 32, 32).transpose(1, 2).contiguous()
    target_action = torch.matmul(X.float(), W.float())
    direction_action = torch.matmul(X.float(), direction_prediction)
    scale_action = torch.matmul(X.float(), scale_prediction)
    output_mask = d_out_mask[:, None].to(target_action.dtype)
    target_action = target_action * output_mask
    direction_action = direction_action * output_mask
    scale_action = scale_action * output_mask
    target_norm = target_action.norm(dim=-1)
    row_mask = x_mask.to(target_norm.dtype) * (target_norm > float(eps)).to(target_norm.dtype)

    direction_norm = direction_action.norm(dim=-1)
    action_cosine = (direction_action * target_action).sum(dim=-1) / (
        direction_norm * target_norm
    ).clamp_min(float(eps))
    action_direction_error = 1.0 - action_cosine.clamp(min=-1.0, max=1.0)
    action_direction_weight = row_mask * (target_norm + float(eps)).pow(float(gamma))
    behavioral_direction = (
        action_direction_error * action_direction_weight
    ).sum(dim=-1) / action_direction_weight.sum(dim=-1).clamp_min(1.0)

    scale_action_norm = scale_action.norm(dim=-1)
    action_scale_delta = torch.log(scale_action_norm + float(eps)) - torch.log(
        target_norm + float(eps)
    )
    action_scale_abs = action_scale_delta.abs()
    action_scale_huber = torch.where(
        action_scale_abs <= float(huber_delta),
        0.5 * action_scale_delta.square(),
        float(huber_delta) * (action_scale_abs - 0.5 * float(huber_delta)),
    )
    behavioral_scale = (
        action_scale_huber * row_mask
    ).sum(dim=-1) / row_mask.sum(dim=-1).clamp_min(1.0)

    weighted_behavioral_direction = float(loss_cfg["behavioral_coef"]) * float(
        loss_cfg["behavioral_direction"]
    ) * behavioral_direction
    weighted_behavioral_scale = float(loss_cfg["behavioral_coef"]) * float(
        loss_cfg["behavioral_scale"]
    ) * behavioral_scale
    weighted_structural_direction = float(loss_cfg["structural_coef"]) * float(
        loss_cfg["structural_direction"]
    ) * structural_direction
    weighted_structural_scale = float(loss_cfg["structural_coef"]) * float(
        loss_cfg["structural_scale"]
    ) * structural_scale
    total = (
        weighted_behavioral_direction
        + weighted_behavioral_scale
        + weighted_structural_direction
        + weighted_structural_scale
    )
    return {
        "total": total,
        "behavioral_direction": behavioral_direction,
        "behavioral_scale": behavioral_scale,
        "structural_direction": structural_direction,
        "structural_scale": structural_scale,
        "weighted_behavioral_direction": weighted_behavioral_direction,
        "weighted_behavioral_scale": weighted_behavioral_scale,
        "weighted_structural_direction": weighted_structural_direction,
        "weighted_structural_scale": weighted_structural_scale,
    }


def _decoder_preference_loss(
    positive: dict[str, torch.Tensor],
    negative: dict[str, torch.Tensor],
    *,
    margin: float,
    temperature: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    denominator = positive["total"].detach() + float(eps)
    gap = (negative["total"] - positive["total"]) / denominator
    loss = float(temperature) * F.softplus((float(margin) - gap) / float(temperature))
    detached_gap = gap.detach()
    return loss.mean(), {
        "gap": detached_gap,
        "margin_satisfied": (detached_gap >= float(margin)).float().mean(),
        "positive_total": positive["total"].detach(),
        "negative_total": negative["total"].detach(),
    }


def _tensor_quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float(), float(q)).item())


def _grad_l2(grads: Sequence[torch.Tensor | None], mask: Sequence[bool] | None = None) -> float:
    squared = 0.0
    for index, grad in enumerate(grads):
        if grad is None or (mask is not None and not mask[index]):
            continue
        squared += float(grad.detach().float().square().sum().item())
    return math.sqrt(squared)


def _parameter_module_group(name: str) -> str:
    if name.startswith("projector."):
        return "representation_head"
    if not name.startswith("model."):
        raise ValueError(f"objective-gradient parameter lacks namespace: {name}")
    local_name = name.removeprefix("model.")
    if local_name.startswith(("to_latent.", "from_latent.")):
        return "latent_bridge"
    if local_name.startswith(
        (
            "direction_tail.",
            "direction_output_norm.",
            "direction_head.",
            "direction_half_embedding",
        )
    ):
        return "direction_tail"
    if local_name.startswith(
        (
            "scale_tail.",
            "scale_output_norm.",
            "scale_head.",
            "scale_half_embedding",
        )
    ):
        return "scale_tail"
    if local_name.startswith(
        (
            "decoder_blocks.",
            "decoder_query_conditioner.",
            "output_queries",
        )
    ):
        return "decoder_trunk"
    return "encoder"


def _objective_gradient_groups(
    named_parameters: Sequence[tuple[str, torch.Tensor]],
    grads: Sequence[torch.Tensor | None],
) -> dict[str, dict[str, float | int]]:
    group_names = (
        "encoder",
        "latent_bridge",
        "decoder_trunk",
        "direction_tail",
        "scale_tail",
        "representation_head",
    )
    squared = {name: 0.0 for name in group_names}
    numel = {name: 0 for name in group_names}
    tensors = {name: 0 for name in group_names}
    tensors_with_grad = {name: 0 for name in group_names}
    for (name, parameter), grad in zip(named_parameters, grads, strict=True):
        group = _parameter_module_group(name)
        numel[group] += int(parameter.numel())
        tensors[group] += 1
        if grad is not None:
            tensors_with_grad[group] += 1
            squared[group] += float(grad.detach().float().square().sum().item())
    return {
        name: {
            "gradient_l2": math.sqrt(squared[name]),
            "gradient_rms": math.sqrt(squared[name] / max(numel[name], 1)),
            "numel": numel[name],
            "parameter_tensors": tensors[name],
            "parameter_tensors_with_grad": tensors_with_grad[name],
        }
        for name in group_names
    }


ENCODER_PREFIXES = (
    "latent_slots",
    "encoder_blocks.",
    "latent_norm.",
    "to_latent.",
    "distribution_encoder.",
    "group_embedding.",
    "chunk_embedding.",
    "tile_row_embedding.",
    "tile_col_embedding.",
    "scale_mlp.",
    "continuous_projection.",
)


class HardPairResolver:
    def __init__(self, source: OperatorBankTrainingDataset, *, seed: int, cache_size: int) -> None:
        self.source = source
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self._candidates: dict[tuple[str, int, str], list[tuple[str, str]]] = {}
        for key, locations in source.operator_groups.items():
            meta = locations[0].metadata
            match = (
                str(meta.get("dataset", "unspecified_dataset")),
                int(meta["checkpoint_index_zero_based"]),
                str(meta["layer_key"]),
            )
            self._candidates.setdefault(match, []).append(key)
        for candidates in self._candidates.values():
            candidates.sort()
        invalid = [match for match, candidates in self._candidates.items() if len(candidates) < 2]
        if invalid:
            raise RuntimeError(f"hard-pair index has unmatched layouts; first={invalid[0]}")
        self._cache: OrderedDict[tuple[int, tuple[str, str]], OperatorBankBundle] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _partner_key(self, sample: OperatorBankSample) -> tuple[str, str]:
        current = (str(sample.meta["checkpoint_sha256"]), str(sample.meta["layer_key"]))
        source_meta = self.source.operator_groups[current][0].metadata
        sample.meta["checkpoint_index_zero_based"] = int(
            source_meta["checkpoint_index_zero_based"]
        )
        match = (
            str(sample.meta["dataset"]),
            int(sample.meta["checkpoint_index_zero_based"]),
            str(sample.meta["layer_key"]),
        )
        candidates = self._candidates[match]
        current_index = candidates.index(current)
        cycle = int(sample.meta["bundle_cycle"])
        digest = hashlib.sha256(
            f"{self.seed}|{cycle}|{current[0]}|{current[1]}".encode("utf-8")
        ).digest()
        offset = 1 + int.from_bytes(digest[:8], "big") % (len(candidates) - 1)
        return candidates[(current_index + offset) % len(candidates)]

    def _bundle(self, cycle: int, key: tuple[str, str]) -> OperatorBankBundle:
        cache_key = (int(cycle), key)
        bundle = self._cache.get(cache_key)
        if bundle is not None:
            self._cache.move_to_end(cache_key)
            self.cache_hits += 1
            return bundle
        self.cache_misses += 1
        bundle = self.source._materialize_bundle(OperatorBundleRequest(int(cycle), key), None)
        if bundle.gauge_id != "canonical":
            raise RuntimeError("hard-pair resolver produced a non-canonical bundle")
        self._cache[cache_key] = bundle
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return bundle

    def resolve(self, anchor: OperatorBankSample) -> OperatorBankSample:
        partner_key = self._partner_key(anchor)
        cycle = int(anchor.meta["bundle_cycle"])
        bundle = self._bundle(cycle, partner_key)
        rows, cols = bundle.spec.matrix_shape
        tile_cols = (cols + 127) // 128
        parent_row = int(anchor.meta["parent_tile_row"])
        parent_col = int(anchor.meta["parent_tile_col"])
        local_index = parent_row * tile_cols + parent_col
        parent = self.source._sample_from_bundle(bundle, local_index)
        # `_sample_from_bundle` normally receives its committed logical index
        # from the streaming mixer.  The directly resolved partner is not part
        # of that anchor cursor, so carry the anchor parent identity solely for
        # the p128->p32 slicing metadata.
        parent.meta["logical_index"] = int(anchor.meta["parent_logical_index"])
        partner = ValidSubtileStream._subtile(
            parent,
            int(anchor.meta["subpatch_index"]),
            int(anchor.meta["logical_index"]),
        )
        if partner is None:
            raise RuntimeError("matched partner unexpectedly has an empty exact-layout subtile")
        partner.meta["checkpoint_index_zero_based"] = int(
            bundle.group_meta["checkpoint_index_zero_based"]
        )
        self._validate(anchor, partner)
        return partner

    @staticmethod
    def _validate(left: OperatorBankSample, right: OperatorBankSample) -> None:
        exact_fields = (
            "dataset",
            "checkpoint_index_zero_based",
            "layer_key",
            "parent_tile_row",
            "parent_tile_col",
            "subpatch_index",
            "subtile_input_index",
            "subtile_output_index",
            "tile_row",
            "tile_col",
            "gauge_id",
        )
        mismatched = [field for field in exact_fields if left.meta[field] != right.meta[field]]
        for field in ("operation", "depth_index", "role"):
            if left.meta["operator"][field] != right.meta["operator"][field]:
                mismatched.append(f"operator.{field}")
        for field in ("d_in_mask", "d_out_mask", "x_mask"):
            if not torch.equal(left.meta[field], right.meta[field]):
                mismatched.append(field)
        if mismatched:
            raise RuntimeError(f"hard pair violates exact nuisance match: {mismatched}")
        if left.meta["lineage_id"] == right.meta["lineage_id"]:
            raise RuntimeError("hard pair must use different lineages")
        if left.meta["checkpoint_sha256"] == right.meta["checkpoint_sha256"]:
            raise RuntimeError("hard pair must use different checkpoint payloads")
        if left.meta["gauge_id"] != "canonical":
            raise RuntimeError("hard pair must be canonical")

    def telemetry(self) -> dict[str, Any]:
        return {
            "schema": "hard_pair_resolver_v1",
            "match_buckets": len(self._candidates),
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


def _collate_samples(samples: list[OperatorBankSample]) -> dict[str, Any]:
    distribution_sources: list[int] = []
    distribution_groups: list[int] = []
    group_by_context: dict[tuple[str, str, int, int], int] = {}
    for sample_index, sample in enumerate(samples):
        context_key = (
            str(sample.meta["checkpoint_sha256"]),
            str(sample.meta["layer_key"]),
            int(sample.meta["parent_tile_row"]),
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
        "distribution_source_indices": torch.tensor(distribution_sources),
        "distribution_group_index": torch.tensor(distribution_groups),
        "lineage_ids": [str(sample.meta["lineage_id"]) for sample in samples],
        "checkpoint_sha256": [str(sample.meta["checkpoint_sha256"]) for sample in samples],
    }


def _paired_batch_from_stream(
    stream: ValidSubtileStream,
    resolver: HardPairResolver,
    pairs_per_batch: int,
    *,
    pin_memory: bool,
) -> dict[str, Any]:
    cursor_before = stream.cursor()
    anchors = [next(stream) for _ in range(int(pairs_per_batch))]
    logical = [int(sample.meta["logical_index"]) for sample in anchors]
    expected = list(range(logical[0], logical[0] + int(pairs_per_batch)))
    if logical != expected:
        raise RuntimeError("paired anchor stream is not logically contiguous")
    samples: list[OperatorBankSample] = []
    for anchor in anchors:
        samples.extend((anchor, resolver.resolve(anchor)))
    batch = _collate_samples(samples)
    batch["cursor_before"] = cursor_before
    batch["cursor_after"] = stream.cursor()
    batch["anchor_logical_indices"] = logical
    if pin_memory:
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.pin_memory()
    return batch


class BackgroundBatchPrefetcher(Iterator[dict[str, Any]]):
    def __init__(self, producer: Any, *, depth: int) -> None:
        self._producer = producer
        self._queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=int(depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mini-pair-prefetch", daemon=True)
        self._thread.start()

    def _put(self, item: tuple[bool, Any]) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._producer()
            except BaseException as exc:
                self._put((False, exc))
                return
            self._put((True, item))

    def __iter__(self) -> BackgroundBatchPrefetcher:
        return self

    def __next__(self) -> dict[str, Any]:
        ok, payload = self._queue.get()
        if not ok:
            raise payload
        return payload

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def _copy_batch_for_probe(batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu()
        elif key in {"lineage_ids", "checkpoint_sha256", "anchor_logical_indices"}:
            result[key] = value
        elif isinstance(value, SubtileCursor):
            result[key] = value.to_dict()
    return result


def _to_device_batch(cpu_batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "W",
        "X",
        "x_mask",
        "d_in_mask",
        "d_out_mask",
        "tile_row",
        "tile_col",
        "distribution_source_indices",
        "distribution_group_index",
    )
    return {key: cpu_batch[key].to(device, non_blocking=True) for key in keys}


def _paired_target_indices(pairs_per_batch: int, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.arange(int(pairs_per_batch), device=device, dtype=torch.long)
    bit = (pair + int(step)) & 1
    target = 2 * pair + bit
    foreign = target ^ 1
    return target, foreign


def _pair_weight_distances(
    W: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if W.shape[0] % 2:
        raise ValueError("pair distance requires an even physical batch")
    mask = d_in_mask[0::2, :, None] & d_out_mask[0::2, None, :]
    left = W[0::2].float() * mask
    right = W[1::2].float() * mask
    absolute = torch.sqrt(
        ((left - right).square().sum(dim=(1, 2)))
        / mask.sum(dim=(1, 2)).clamp_min(1)
    )
    relative = (left - right).flatten(1).norm(dim=-1) / left.flatten(1).norm(
        dim=-1
    ).clamp_min(1.0e-8)
    return absolute, relative


def _plot_metrics(path: Path, output: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, label in (
        ("base_loss", "base routed polar"),
        ("loss", "combined"),
        ("behavioral", "behavioral"),
        ("structural", "structural"),
    ):
        axes[0].plot(steps, [row[key] for row in rows], label=label, linewidth=1.1)
    for key, label in (
        ("preference_loss", "raw preference"),
        ("representation_loss", "raw representation"),
        ("preference_gap_median", "relative matched gap median"),
    ):
        axes[1].plot(steps, [row[key] for row in rows], label=label, linewidth=1.1)
    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("auxiliary / gap")
    axes[1].set_xlabel("optimizer step")
    axes[0].set_title("Mini polar AE — matched representation + decoder preference")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_diagnostics(
    metrics_path: Path,
    gradient_path: Path,
    probe_path: Path,
    output_root: Path,
) -> dict[str, str]:
    _plot_metrics(metrics_path, output_root / "train_losses.png")
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, label in (
        ("preference_relative_gap_p10", "relative p10"),
        ("preference_relative_gap_mean", "relative mean"),
        ("preference_relative_gap_median", "relative median"),
        ("preference_relative_gap_p90", "relative p90"),
    ):
        axes[0].plot(steps, [row[key] for row in rows], label=label)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="margin")
    axes[1].plot(
        steps,
        [row["preference_margin_satisfied"] for row in rows],
        label="margin satisfied",
    )
    axes[1].plot(
        steps,
        [row["preference_max_component_abs_share_mean"] for row in rows],
        label="max component |gap| share",
    )
    axes[0].set_ylabel("relative gap")
    axes[1].set_ylabel("fraction")
    axes[1].set_xlabel("optimizer step")
    axes[0].set_title("Decoder latent-preference diagnostics")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    preference_plot = output_root / "preference_gap.png"
    fig.savefig(preference_plot, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(
        steps,
        [row["representation_positive_cosine"] for row in rows],
        label="positive cosine",
    )
    axes[0].plot(
        steps,
        [row["representation_negative_cosine"] for row in rows],
        label="matched-negative cosine",
    )
    axes[0].plot(
        steps,
        [row["representation_pairwise_accuracy"] for row in rows],
        label="pairwise accuracy",
    )
    probe_rows = (
        [
            json.loads(line)
            for line in probe_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if probe_path.is_file()
        else []
    )
    if probe_rows:
        probe_steps = [row["step"] for row in probe_rows]
        axes[1].plot(
            probe_steps,
            [row["latent_effective_rank"] for row in probe_rows],
            label="effective rank",
            marker="o",
            markersize=3,
        )
        axes[1].plot(
            probe_steps,
            [row["latent_stable_rank"] for row in probe_rows],
            label="stable rank",
            marker="o",
            markersize=3,
        )
    axes[0].set_ylabel("cosine / accuracy")
    axes[1].set_ylabel("rank")
    axes[1].set_xlabel("optimizer step")
    axes[0].set_title("Representation geometry")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    representation_plot = output_root / "representation_geometry.png"
    fig.savefig(representation_plot, dpi=150)
    plt.close(fig)

    gradient_rows = (
        [
            json.loads(line)
            for line in gradient_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if gradient_path.is_file()
        else []
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for objective in ("base", "preference", "representation"):
        for group in (
            "encoder",
            "latent_bridge",
            "decoder_trunk",
            "direction_tail",
            "scale_tail",
            "representation_head",
        ):
            points = [
                (
                    row["step"],
                    row.get("objective_gradient_groups", {})
                    .get(objective, {})
                    .get(group, {})
                    .get("gradient_l2"),
                )
                for row in gradient_rows
            ]
            points = [(step, value) for step, value in points if value is not None]
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    label=f"{objective}/{group}",
                    linewidth=1.0,
                    marker="o",
                    markersize=3,
                    markevery=max(1, len(points) // 100),
                )
    ax.set_yscale("symlog", linthresh=1.0e-8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("gradient L2")
    ax.set_title("Objective-split module gradients")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    gradient_plot = output_root / "component_gradients.png"
    fig.savefig(gradient_plot, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, [row["steps_per_second"] for row in rows], label="steps/s")
    axes[0].plot(
        steps,
        [row["examples_per_second"] for row in rows],
        label="examples/s",
    )
    for key, label in (
        ("gpu_encoder_ms", "encoder"),
        ("gpu_positive_decoder_ms", "positive decoder"),
        ("gpu_negative_decoder_ms", "negative decoder"),
        ("gpu_loss_ms", "loss"),
        ("gpu_backward_optimizer_ms", "backward+optimizer"),
    ):
        axes[1].plot(steps, [row.get(key, float("nan")) for row in rows], label=label)
    axes[0].set_ylabel("rate")
    axes[1].set_ylabel("milliseconds")
    axes[1].set_xlabel("optimizer step")
    axes[0].set_title("Production throughput")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    throughput_plot = output_root / "throughput.png"
    fig.savefig(throughput_plot, dpi=150)
    plt.close(fig)
    return {
        "loss": str(output_root / "train_losses.png"),
        "preference": str(preference_plot),
        "representation": str(representation_plot),
        "gradients": str(gradient_plot),
        "throughput": str(throughput_plot),
    }


def _forward_losses(
    model: MiniPolarWeightBottleneck,
    projector: LatentRepresentationProjector,
    batch: dict[str, torch.Tensor],
    normalization: dict[str, Any],
    config: dict[str, Any],
    *,
    step: int,
    capture_depth: bool,
    timing_events: dict[str, torch.cuda.Event] | None = None,
) -> dict[str, Any]:
    W = batch["W"]
    X = batch["X"]
    x_mask = batch["x_mask"]
    d_in_mask = batch["d_in_mask"]
    d_out_mask = batch["d_out_mask"]
    tile_row = batch["tile_row"]
    tile_col = batch["tile_col"]
    content, log_scale, token_valid = _prepare_normalized_inputs(
        W,
        d_in_mask,
        d_out_mask,
        scale_mean=float(normalization["log2_scale_mean"]),
        scale_std=float(normalization["log2_scale_std"]),
    )
    if timing_events is not None:
        timing_events["start"].record()
    dist_patch = model._encode_distribution_context(
        X,
        x_mask,
        batch["distribution_source_indices"],
        batch["distribution_group_index"],
    )
    latent, depth_rows = model.encode(
        content,
        log_scale,
        tile_row,
        tile_col,
        token_valid,
        dist_patch,
        capture_depth=capture_depth,
    )
    if timing_events is not None:
        timing_events["encoded"].record()
    prediction, pred_dirs, pred_log_scales = model.decode_polar(
        latent,
        tile_row,
        tile_col,
        d_in_mask,
        d_out_mask,
        token_valid,
        dist_patch,
    )
    if timing_events is not None:
        timing_events["positive_decoded"].record()
    base_loss, base_parts = _polar_routed_loss(
        X,
        W,
        pred_dirs,
        pred_log_scales,
        x_mask,
        d_in_mask,
        d_out_mask,
        config["loss"],
    )

    target_indices, foreign_indices = _paired_target_indices(
        int(config["pairing"]["pairs_per_batch"]), step, latent.device
    )
    _negative_prediction, negative_dirs, negative_log_scales = model.decode_polar(
        latent.index_select(0, foreign_indices).detach(),
        tile_row.index_select(0, target_indices),
        tile_col.index_select(0, target_indices),
        d_in_mask.index_select(0, target_indices),
        d_out_mask.index_select(0, target_indices),
        token_valid.index_select(0, target_indices),
        dist_patch.index_select(0, target_indices),
        detach_direct_conditioning=True,
    )
    if timing_events is not None:
        timing_events["negative_decoded"].record()
    positive_per_example_all = _polar_routed_loss_per_example(
        X,
        W,
        pred_dirs,
        pred_log_scales,
        x_mask,
        d_in_mask,
        d_out_mask,
        config["loss"],
    )
    positive_per_example = {
        key: value.index_select(0, target_indices)
        for key, value in positive_per_example_all.items()
    }
    negative_per_example = _polar_routed_loss_per_example(
        X.index_select(0, target_indices),
        W.index_select(0, target_indices),
        negative_dirs,
        negative_log_scales,
        x_mask.index_select(0, target_indices),
        d_in_mask.index_select(0, target_indices),
        d_out_mask.index_select(0, target_indices),
        config["loss"],
    )
    preference_cfg = config["preference"]
    preference_loss, preference_stats = _decoder_preference_loss(
        positive_per_example,
        negative_per_example,
        margin=float(preference_cfg["margin"]),
        temperature=float(preference_cfg["temperature"]),
        eps=float(preference_cfg["eps"]),
    )
    with torch.autocast(device_type="cuda", enabled=False):
        first_view = projector(latent.float())
        second_view = projector(latent.float())
        representation_loss, representation_stats = _binary_matched_infonce(
            first_view,
            second_view,
            temperature=float(config["representation"]["temperature"]),
        )
    if timing_events is not None:
        timing_events["losses_complete"].record()
    return {
        "prediction": prediction,
        "latent": latent,
        "depth_rows": depth_rows,
        "pred_log_scales": pred_log_scales,
        "base_loss": base_loss,
        "base_parts": base_parts,
        "positive_per_example": positive_per_example,
        "negative_per_example": negative_per_example,
        "preference_loss": preference_loss,
        "preference_stats": preference_stats,
        "representation_loss": representation_loss,
        "representation_stats": representation_stats,
    }


def _resolve_calibration(
    calibration_path: Path,
    model: MiniPolarWeightBottleneck,
    projector: LatentRepresentationProjector,
    losses: dict[str, Any],
    config: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    if calibration_path.is_file():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "mini_latent_preference_loss_calibration_v1":
            raise RuntimeError("loss calibration artifact has the wrong schema")
        print(
            "[mini-latent-pref] stage=loss-calibration-cache-hit "
            f"lambda_pref={payload['lambda_preference']:.9g} "
            f"lambda_repr={payload['lambda_representation']:.9g} path={calibration_path}",
            flush=True,
        )
        return payload
    if resume:
        raise FileNotFoundError(f"resume requires sealed loss calibration: {calibration_path}")

    named_model_parameters = list(model.named_parameters())
    model_parameters = [parameter for _name, parameter in named_model_parameters]
    base_grads = torch.autograd.grad(
        losses["base_loss"], model_parameters, retain_graph=True, allow_unused=True
    )
    preference_grads = torch.autograd.grad(
        losses["preference_loss"], model_parameters, retain_graph=True, allow_unused=True
    )
    representation_grads = torch.autograd.grad(
        losses["representation_loss"], model_parameters, retain_graph=True, allow_unused=True
    )
    encoder_mask = [name.startswith(ENCODER_PREFIXES) for name, _parameter in named_model_parameters]
    base_full_norm = _grad_l2(base_grads)
    preference_full_norm = _grad_l2(preference_grads)
    base_encoder_norm = _grad_l2(base_grads, encoder_mask)
    representation_encoder_norm = _grad_l2(representation_grads, encoder_mask)
    if min(base_full_norm, preference_full_norm, base_encoder_norm, representation_encoder_norm) <= 0.0:
        raise RuntimeError(
            "loss calibration encountered a zero gradient norm: "
            f"base={base_full_norm} pref={preference_full_norm} "
            f"base_encoder={base_encoder_norm} repr_encoder={representation_encoder_norm}"
        )
    cfg = config["calibration"]
    raw_preference = float(cfg["preference_target_ratio"]) * base_full_norm / preference_full_norm
    raw_representation = (
        float(cfg["representation_target_encoder_ratio"])
        * base_encoder_norm
        / representation_encoder_norm
    )
    lambda_preference = min(
        max(raw_preference, float(cfg["preference_coefficient_min"])),
        float(cfg["preference_coefficient_max"]),
    )
    lambda_representation = min(
        max(raw_representation, float(cfg["representation_coefficient_min"])),
        float(cfg["representation_coefficient_max"]),
    )
    payload = {
        "schema": "mini_latent_preference_loss_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_full_gradient_l2": base_full_norm,
        "preference_full_gradient_l2": preference_full_norm,
        "base_encoder_gradient_l2": base_encoder_norm,
        "representation_encoder_gradient_l2": representation_encoder_norm,
        "raw_lambda_preference": raw_preference,
        "raw_lambda_representation": raw_representation,
        "lambda_preference": lambda_preference,
        "lambda_representation": lambda_representation,
        "preference_clamped": lambda_preference != raw_preference,
        "representation_clamped": lambda_representation != raw_representation,
        "calibration_contract": cfg,
    }
    _atomic_json(calibration_path, payload)
    print(
        "[mini-latent-pref] stage=loss-calibration-complete "
        f"lambda_pref={lambda_preference:.9g} lambda_repr={lambda_representation:.9g} "
        f"path={calibration_path}",
        flush=True,
    )
    return payload


@torch.no_grad()
def _evaluate_fixed_probe(
    model: MiniPolarWeightBottleneck,
    projector: LatentRepresentationProjector,
    cpu_batch: dict[str, Any],
    normalization: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    *,
    step: int,
) -> dict[str, Any]:
    model_was_training = model.training
    projector_was_training = projector.training
    model.eval()
    projector.eval()
    batch = _to_device_batch(cpu_batch, device)
    W, X = batch["W"], batch["X"]
    d_in_mask, d_out_mask, x_mask = batch["d_in_mask"], batch["d_out_mask"], batch["x_mask"]
    content, log_scale, token_valid = _prepare_normalized_inputs(
        W,
        d_in_mask,
        d_out_mask,
        scale_mean=float(normalization["log2_scale_mean"]),
        scale_std=float(normalization["log2_scale_std"]),
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dist_patch = model._encode_distribution_context(
            X,
            x_mask,
            batch["distribution_source_indices"],
            batch["distribution_group_index"],
        )
        latent, _depth = model.encode(
            content,
            log_scale,
            batch["tile_row"],
            batch["tile_col"],
            token_valid,
            dist_patch,
        )
        _positive_prediction, positive_dirs, positive_scales = model.decode_polar(
            latent,
            batch["tile_row"],
            batch["tile_col"],
            d_in_mask,
            d_out_mask,
            token_valid,
            dist_patch,
        )
        pair_foreign = torch.arange(latent.shape[0], device=device) ^ 1
        _negative_prediction, negative_dirs, negative_scales = model.decode_polar(
            latent.index_select(0, pair_foreign),
            batch["tile_row"],
            batch["tile_col"],
            d_in_mask,
            d_out_mask,
            token_valid,
            dist_patch,
            detach_direct_conditioning=True,
        )
        random_foreign = torch.roll(
            torch.arange(latent.shape[0], device=device), shifts=17
        )
        _random_prediction, random_dirs, random_scales = model.decode_polar(
            latent.index_select(0, random_foreign),
            batch["tile_row"],
            batch["tile_col"],
            d_in_mask,
            d_out_mask,
            token_valid,
            dist_patch,
            detach_direct_conditioning=True,
        )
        _zero_prediction, zero_dirs, zero_scales = model.decode_polar(
            torch.zeros_like(latent),
            batch["tile_row"],
            batch["tile_col"],
            d_in_mask,
            d_out_mask,
            token_valid,
            dist_patch,
        )
        context_foreign = torch.roll(
            torch.arange(latent.shape[0], device=device), shifts=31
        )
        _context_prediction, context_dirs, context_scales = model.decode_polar(
            latent,
            batch["tile_row"],
            batch["tile_col"],
            d_in_mask,
            d_out_mask,
            token_valid,
            dist_patch.index_select(0, context_foreign),
        )
    positive = _polar_routed_loss_per_example(
        X, W, positive_dirs, positive_scales, x_mask, d_in_mask, d_out_mask, config["loss"]
    )
    negative = _polar_routed_loss_per_example(
        X, W, negative_dirs, negative_scales, x_mask, d_in_mask, d_out_mask, config["loss"]
    )
    random_negative = _polar_routed_loss_per_example(
        X, W, random_dirs, random_scales, x_mask, d_in_mask, d_out_mask, config["loss"]
    )
    zero_latent = _polar_routed_loss_per_example(
        X, W, zero_dirs, zero_scales, x_mask, d_in_mask, d_out_mask, config["loss"]
    )
    context_rolled = _polar_routed_loss_per_example(
        X, W, context_dirs, context_scales, x_mask, d_in_mask, d_out_mask, config["loss"]
    )
    gap = (negative["total"] - positive["total"]) / (
        positive["total"] + float(config["preference"]["eps"])
    )
    flat = latent.float().flatten(1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1.0e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    )
    stable_rank = energy.sum() / energy.max().clamp_min(1.0e-12)
    projected = projector(latent.float())
    paired_cosine = (projected[0::2] * projected[1::2]).sum(dim=-1)
    row = {
        "schema": SCHEMA,
        "step": int(step),
        "positive_loss_mean": float(positive["total"].mean()),
        "matched_foreign_loss_mean": float(negative["total"].mean()),
        "random_latent_roll_loss_mean": float(random_negative["total"].mean()),
        "zero_latent_loss_mean": float(zero_latent["total"].mean()),
        "context_roll_loss_mean": float(context_rolled["total"].mean()),
        "random_latent_roll_relative_delta": float(
            ((random_negative["total"] - positive["total"]) / positive["total"].clamp_min(1.0e-6)).mean()
        ),
        "zero_latent_relative_delta": float(
            ((zero_latent["total"] - positive["total"]) / positive["total"].clamp_min(1.0e-6)).mean()
        ),
        "context_roll_relative_delta": float(
            ((context_rolled["total"] - positive["total"]) / positive["total"].clamp_min(1.0e-6)).mean()
        ),
        "matched_relative_gap_p10": _tensor_quantile(gap, 0.10),
        "matched_relative_gap_median": _tensor_quantile(gap, 0.50),
        "matched_relative_gap_p90": _tensor_quantile(gap, 0.90),
        "latent_effective_rank": float(effective_rank),
        "latent_stable_rank": float(stable_rank),
        "latent_top_energy_fraction": float(energy.max() / energy.sum().clamp_min(1.0e-12)),
        "paired_projector_cosine_mean": float(paired_cosine.mean()),
    }
    model.train(model_was_training)
    projector.train(projector_was_training)
    return row


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
            f"/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_smoke_{stamp}"
        )
    else:
        output_root = Path(config["output_root"]).resolve()
    if args.smoke and int(args.smoke_steps) < 1:
        raise ValueError("--smoke-steps must be positive")
    steps = int(args.smoke_steps) if args.smoke else int(config["steps"])
    pairs_per_batch = int(config["pairing"]["pairs_per_batch"])
    batch_size = 2 * pairs_per_batch
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
    retained_checkpoint_every = int(args.retained_checkpoint_every)
    if retained_checkpoint_every < 0:
        raise ValueError("--retained-checkpoint-every must be non-negative")
    if (retained_checkpoint_every > 0) != (args.retained_checkpoint_dir is not None):
        raise ValueError(
            "--retained-checkpoint-every and --retained-checkpoint-dir must be supplied together"
        )
    retained_checkpoint_dir = (
        args.retained_checkpoint_dir.resolve()
        if args.retained_checkpoint_dir is not None
        else None
    )
    startup = {
        "schema": SCHEMA,
        "stage": "preflight",
        "config_path": str(config_path),
        "resolved_config": config,
        "model_config": asdict(model_cfg),
        "device": str(config["device"]),
        "dtype": "bfloat16 core autocast; FP32 projector and loss statistics",
        "seed": int(config["seed"]),
        "cache_mode": "sealed operator bank, cached normalization, canonical matched-pair bundle LRU",
        "output_root": str(output_root),
        "steps": steps,
        "scientific_horizon_steps": int(config["steps"]),
        "batch_size": batch_size,
        "pairs_per_batch": pairs_per_batch,
        "resume_checkpoint": str(resume_path),
        "persistent_model_checkpoint": str(persistent_model_path),
        "retained_checkpoint_every": retained_checkpoint_every,
        "retained_checkpoint_dir": (
            str(retained_checkpoint_dir) if retained_checkpoint_dir is not None else None
        ),
    }
    print("[mini-latent-pref] stage=preflight", json.dumps(startup), flush=True)
    if args.dry_run:
        print("[mini-latent-pref] stage=complete mode=dry-run", flush=True)
        return
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"fresh output root already exists: {output_root}")
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")

    logger = logging.getLogger("mini-latent-pref")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    normalization = _normalization_artifact(config, logger)
    output_root.mkdir(parents=True, exist_ok=True)
    if retained_checkpoint_dir is not None:
        retained_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        if retained_checkpoint_dir.stat().st_dev != resume_path.parent.stat().st_dev:
            raise RuntimeError(
                "retained checkpoint directory and resume_latest must be on the same filesystem"
            )
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

    print("[mini-latent-pref] stage=model-build", flush=True)
    model = MiniPolarWeightBottleneck(model_cfg).to(device)
    core_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if core_parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"mini core parameter count drifted: {core_parameter_count} != {EXPECTED_PARAMETERS}"
        )
    projector = LatentRepresentationProjector(
        model_cfg,
        hidden_dim=int(config["representation"]["projector_hidden_dim"]),
        output_dim=int(config["representation"]["projector_output_dim"]),
        dropout=float(config["representation"]["dropout"]),
    ).to(device)
    projector_parameter_count = sum(parameter.numel() for parameter in projector.parameters())
    if projector_parameter_count != EXPECTED_PROJECTOR_PARAMETERS:
        raise RuntimeError(
            "projector parameter count drifted: "
            f"{projector_parameter_count} != {EXPECTED_PROJECTOR_PARAMETERS}"
        )
    all_named_parameters = [
        (f"model.{name}", parameter) for name, parameter in model.named_parameters()
    ] + [
        (f"projector.{name}", parameter)
        for name, parameter in projector.named_parameters()
    ]
    all_parameters = [parameter for _name, parameter in all_named_parameters]
    optimizer = torch.optim.AdamW(
        all_parameters,
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_step = 0
    committed_cursor = SubtileCursor(0, 0, 0)
    if args.resume:
        print(f"[mini-latent-pref] stage=resume-load path={resume_path}", flush=True)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["schema"] != SCHEMA or payload["config"] != config:
            raise RuntimeError("resume checkpoint schema/config disagrees with requested run")
        model.load_state_dict(payload["model_state"], strict=True)
        projector.load_state_dict(payload["projector_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["step"])
        committed_cursor = SubtileCursor(**payload["subtile_cursor"])
        if committed_cursor.emitted_logical_index != start_step * pairs_per_batch:
            raise RuntimeError("resume cursor disagrees with committed paired optimizer step")
        _restore_rng_state(payload["rng_state"])
        del payload
        _atomic_json(
            output_root / f"resolved_resume_config_step_{start_step:09d}.json",
            startup,
        )
    startup.update(
        {
            "core_parameter_count": core_parameter_count,
            "projector_parameter_count": projector_parameter_count,
            "total_parameter_count": core_parameter_count + projector_parameter_count,
        }
    )
    if not args.resume:
        _atomic_json(output_root / "resolved_config.json", startup)
    print(
        "[mini-latent-pref] stage=model-build-complete "
        f"core_parameters={core_parameter_count} projector_parameters={projector_parameter_count} "
        f"total_parameters={core_parameter_count + projector_parameter_count} "
        f"start_step={start_step} cursor={committed_cursor.to_dict()}",
        flush=True,
    )

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[mini-latent-pref] stage=stop-requested signal={signum}; saving after current committed step",
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    metrics_path = output_root / "train_metrics.jsonl"
    gradient_path = output_root / "gradient_telemetry.jsonl"
    probe_path = output_root / "fixed_probe_metrics.jsonl"
    fixed_probe_path = output_root / "fixed_paired_probe.pt"
    calibration_path = output_root / "loss_calibration.json"
    retained_checkpoint_index_path = output_root / "retained_checkpoints.jsonl"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def save_resume(step: int, cursor: SubtileCursor) -> None:
        print(
            f"[mini-latent-pref] stage=resume-save step={step} path={resume_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": SCHEMA,
                "step": int(step),
                "subtile_cursor": cursor.to_dict(),
                "model_state": model.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state(),
                "config": config,
                "normalization_artifact": normalization,
                "loss_calibration": json.loads(calibration_path.read_text())
                if calibration_path.is_file()
                else None,
            },
            resume_path,
        )

    def save_model(step: int, cursor: SubtileCursor) -> None:
        print(
            f"[mini-latent-pref] stage=model-checkpoint-save step={step} path={persistent_model_path}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "schema": SCHEMA,
                "step": int(step),
                "subtile_cursor": cursor.to_dict(),
                "model_state": model.state_dict(),
                "projector_state": projector.state_dict(),
                "model_config": asdict(model_cfg),
                "config": config,
                "normalization_artifact": normalization,
                "loss_calibration": json.loads(calibration_path.read_text())
                if calibration_path.is_file()
                else None,
            },
            persistent_model_path,
        )

    def retain_resume_checkpoint(
        step: int,
        cursor: SubtileCursor,
        *,
        reason: str,
    ) -> Path:
        if retained_checkpoint_dir is None:
            raise RuntimeError("retained checkpoint directory is not configured")
        destination = retained_checkpoint_dir / f"resume_step_{int(step):09d}.pt"
        _atomic_hardlink(resume_path, destination)
        row = {
            "schema": "mini_latent_preference_retained_checkpoint_v1",
            "step": int(step),
            "subtile_cursor": cursor.to_dict(),
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "reason": str(reason),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(retained_checkpoint_index_path, row)
        print(
            "[mini-latent-pref] stage=retained-checkpoint-save "
            f"step={step} reason={reason} path={destination} bytes={row['bytes']}",
            flush=True,
        )
        return destination

    if retained_checkpoint_every > 0:
        _atomic_json(
            output_root / "retained_checkpoint_policy.json",
            {
                "schema": "mini_latent_preference_retained_checkpoint_policy_v1",
                "every_steps": retained_checkpoint_every,
                "directory": str(retained_checkpoint_dir),
                "payload": "full resume: model, projector, AdamW, RNG, committed cursor, config",
                "retention_window": None,
                "start_step": start_step,
            },
        )
        if args.resume:
            retain_resume_checkpoint(
                start_step,
                committed_cursor,
                reason="resume_start",
            )

    print(
        "[mini-latent-pref] stage=data-open "
        f"pair_manifest={config['operator_bank']['pair_manifest']} "
        f"workers={config['operator_bank']['loader_workers']} "
        f"prefetch_batches={config['pairing']['background_prefetch_batches']} "
        f"cursor={committed_cursor.to_dict()}",
        flush=True,
    )
    diagnostic_stop_reason: str | None = None
    last_step = start_step
    calibration: dict[str, Any] | None = None
    fixed_probe_cpu: dict[str, Any] | None = None
    with ExitStack() as stack:
        stream, parent_stream = _open_subtile_stream(
            stack, config, committed_cursor, logger
        )
        if not isinstance(parent_stream, BalancedOperatorBankMixer):
            raise RuntimeError("paired production requires the balanced bundle mixer")
        source = parent_stream.dataset.source
        resolver = HardPairResolver(
            source,
            seed=int(config["seed"]) + 93_017,
            cache_size=int(config["pairing"]["pair_bundle_cache"]),
        )
        producer = lambda: _paired_batch_from_stream(
            stream,
            resolver,
            pairs_per_batch,
            pin_memory=bool(config["pairing"]["pin_memory"]),
        )
        prefetcher = BackgroundBatchPrefetcher(
            producer,
            depth=int(config["pairing"]["background_prefetch_batches"]),
        )
        stack.callback(prefetcher.close)
        print(
            "[mini-latent-pref] stage=train "
            f"steps={steps} physical_batch={batch_size} pairs={pairs_per_batch} "
            f"lr={config['learning_rate']} "
            "objective=L_base+lambda_pref*L_pref+lambda_repr*L_repr "
            "base=behavioral(direction+10*scale)+structural(direction+10*scale) "
            "pairing=exact-layout-different-lineage-same-checkpoint-index gauge=canonical",
            flush=True,
        )
        training_started = time.monotonic()
        rate_window_started = training_started
        rate_window_step = start_step
        for step in range(start_step + 1, steps + 1):
            step_started = time.monotonic()
            wait_started = time.monotonic()
            cpu_batch = next(prefetcher)
            data_wait_seconds = time.monotonic() - wait_started
            cursor_before = cpu_batch["cursor_before"]
            cursor_after = cpu_batch["cursor_after"]
            if cursor_before != committed_cursor:
                raise RuntimeError(
                    f"prefetched batch cursor mismatch: {cursor_before} != {committed_cursor}"
                )
            expected_after = step * pairs_per_batch
            if cursor_after.emitted_logical_index != expected_after:
                raise RuntimeError(
                    f"paired cursor drift: {cursor_after.emitted_logical_index} != {expected_after}"
                )
            if fixed_probe_cpu is None:
                if fixed_probe_path.is_file():
                    fixed_probe_cpu = torch.load(
                        fixed_probe_path, map_location="cpu", weights_only=False
                    )
                    print(
                        f"[mini-latent-pref] stage=fixed-probe-cache-hit path={fixed_probe_path}",
                        flush=True,
                    )
                else:
                    if args.resume:
                        raise FileNotFoundError(
                            f"resume requires fixed paired probe: {fixed_probe_path}"
                        )
                    fixed_probe_cpu = _copy_batch_for_probe(cpu_batch)
                    _atomic_torch_save(fixed_probe_cpu, fixed_probe_path)
                    print(
                        f"[mini-latent-pref] stage=fixed-probe-write path={fixed_probe_path}",
                        flush=True,
                    )

            h2d_started = time.monotonic()
            batch = _to_device_batch(cpu_batch, device)
            h2d_enqueue_seconds = time.monotonic() - h2d_started
            model.train()
            projector.train()
            optimizer.zero_grad(set_to_none=True)
            log_due = args.smoke or step == 1 or step % int(config["log_every"]) == 0
            gradient_due = step == 1 or step % int(config["gradient_log_every"]) == 0
            timing_events = None
            if log_due:
                timing_events = {
                    name: torch.cuda.Event(enable_timing=True)
                    for name in (
                        "start",
                        "encoded",
                        "positive_decoded",
                        "negative_decoded",
                        "losses_complete",
                        "optimizer_complete",
                    )
                }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = _forward_losses(
                    model,
                    projector,
                    batch,
                    normalization,
                    config,
                    step=step,
                    capture_depth=gradient_due,
                    timing_events=timing_events,
                )
            if calibration is None:
                calibration = _resolve_calibration(
                    calibration_path,
                    model,
                    projector,
                    losses,
                    config,
                    resume=args.resume,
                )
            ramp = min(1.0, step / float(config["calibration"]["ramp_steps"]))
            weighted_preference = (
                ramp
                * float(calibration["lambda_preference"])
                * losses["preference_loss"]
            )
            weighted_representation = (
                ramp
                * float(calibration["lambda_representation"])
                * losses["representation_loss"]
            )
            auxiliary_loss = weighted_preference + weighted_representation
            total_loss = losses["base_loss"] + auxiliary_loss
            if not all(
                bool(torch.isfinite(value))
                for value in (
                    losses["base_loss"],
                    losses["preference_loss"],
                    losses["representation_loss"],
                    total_loss,
                )
            ):
                raise RuntimeError(f"nonfinite objective at step {step}")

            auxiliary_gradient_ratio: float | None = None
            base_gradient_l2: float | None = None
            preference_gradient_l2: float | None = None
            representation_gradient_l2: float | None = None
            objective_gradient_groups: dict[str, Any] = {}
            if gradient_due:
                base_grads = torch.autograd.grad(
                    losses["base_loss"], all_parameters, retain_graph=True, allow_unused=True
                )
                preference_grads = torch.autograd.grad(
                    weighted_preference,
                    all_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                representation_grads = torch.autograd.grad(
                    weighted_representation,
                    all_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                auxiliary_grads = tuple(
                    None
                    if preference_grad is None and representation_grad is None
                    else (
                        (preference_grad if preference_grad is not None else 0.0)
                        + (representation_grad if representation_grad is not None else 0.0)
                    )
                    for preference_grad, representation_grad in zip(
                        preference_grads, representation_grads, strict=True
                    )
                )
                base_gradient_l2 = _grad_l2(base_grads)
                preference_gradient_l2 = _grad_l2(preference_grads)
                representation_gradient_l2 = _grad_l2(representation_grads)
                auxiliary_gradient_l2 = _grad_l2(auxiliary_grads)
                auxiliary_gradient_ratio = auxiliary_gradient_l2 / max(
                    base_gradient_l2, 1.0e-30
                )
                objective_gradient_groups = {
                    "base": _objective_gradient_groups(
                        all_named_parameters, base_grads
                    ),
                    "preference": _objective_gradient_groups(
                        all_named_parameters, preference_grads
                    ),
                    "representation": _objective_gradient_groups(
                        all_named_parameters, representation_grads
                    ),
                }
                if auxiliary_gradient_ratio > float(
                    config["calibration"]["auxiliary_gradient_ratio_stop"]
                ):
                    diagnostic_stop_reason = (
                        "weighted auxiliary gradient ratio exceeded fail-closed threshold: "
                        f"{auxiliary_gradient_ratio:.9g} > "
                        f"{config['calibration']['auxiliary_gradient_ratio_stop']} at step {step}"
                    )
                    _atomic_json(
                        output_root / "DIAGNOSTIC_STOP.json",
                        {
                            "schema": SCHEMA,
                            "step_not_committed": step,
                            "last_committed_step": last_step,
                            "subtile_cursor": committed_cursor.to_dict(),
                            "reason": diagnostic_stop_reason,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    print(
                        f"[mini-latent-pref] stage=diagnostic-stop reason={diagnostic_stop_reason}",
                        flush=True,
                    )
                    break

            total_loss.backward()
            if step == 1:
                missing_model = [
                    name for name, parameter in model.named_parameters() if parameter.grad is None
                ]
                missing_projector = [
                    name for name, parameter in projector.named_parameters() if parameter.grad is None
                ]
                if missing_model or missing_projector:
                    raise RuntimeError(
                        "step-1 parameters without gradients: "
                        f"model={missing_model} projector={missing_projector}"
                    )
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                all_parameters, float(config["grad_clip_norm"])
            )
            optimizer.step()
            if timing_events is not None:
                timing_events["optimizer_complete"].record()
            committed_cursor = cursor_after
            last_step = step

            timing: dict[str, float] = {}
            if timing_events is not None:
                timing_events["optimizer_complete"].synchronize()
                timing = {
                    "gpu_encoder_ms": timing_events["start"].elapsed_time(
                        timing_events["encoded"]
                    ),
                    "gpu_positive_decoder_ms": timing_events["encoded"].elapsed_time(
                        timing_events["positive_decoded"]
                    ),
                    "gpu_negative_decoder_ms": timing_events[
                        "positive_decoded"
                    ].elapsed_time(timing_events["negative_decoded"]),
                    "gpu_loss_ms": timing_events["negative_decoded"].elapsed_time(
                        timing_events["losses_complete"]
                    ),
                    "gpu_backward_optimizer_ms": timing_events[
                        "losses_complete"
                    ].elapsed_time(timing_events["optimizer_complete"]),
                    "gpu_total_ms": timing_events["start"].elapsed_time(
                        timing_events["optimizer_complete"]
                    ),
                }
            elapsed = time.monotonic() - training_started
            step_seconds = time.monotonic() - step_started
            rate_window_seconds = time.monotonic() - rate_window_started
            rate_window_steps = step - rate_window_step
            row: dict[str, Any] | None = None
            if log_due:
                positive = losses["positive_per_example"]
                negative = losses["negative_per_example"]
                gap = losses["preference_stats"]["gap"]
                component_deltas = {
                    key: (
                        negative[f"weighted_{key}"].detach()
                        - positive[f"weighted_{key}"].detach()
                    )
                    for key in (
                        "behavioral_direction",
                        "behavioral_scale",
                        "structural_direction",
                        "structural_scale",
                    )
                }
                absolute_gap = (
                    losses["preference_stats"]["negative_total"]
                    - losses["preference_stats"]["positive_total"]
                )
                component_abs = torch.stack(
                    [value.abs() for value in component_deltas.values()], dim=-1
                )
                component_abs_shares = component_abs / component_abs.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0e-12)
                with torch.no_grad():
                    operator_relative_mse = BigWeightVAELossMixin.operator_relative_mse_loss(
                        batch["X"],
                        batch["W"],
                        losses["prediction"],
                        x_mask=batch["x_mask"],
                        d_in_mask=batch["d_in_mask"],
                        d_out_mask=batch["d_out_mask"],
                    )
                    weight_rmse = _masked_weight_rmse(
                        losses["prediction"],
                        batch["W"],
                        batch["d_in_mask"],
                        batch["d_out_mask"],
                    )
                    valid_patches = (
                        batch["d_out_mask"][:, :, None]
                        & batch["d_in_mask"].view(batch_size, 2, 16).any(dim=-1)[:, None]
                    )
                    valid_scales = losses["pred_log_scales"][valid_patches]
                    pair_weight_distance, pair_weight_relative_distance = (
                        _pair_weight_distances(
                            batch["W"], batch["d_in_mask"], batch["d_out_mask"]
                        )
                    )
                    flat_latent = F.normalize(
                        losses["latent"].float().flatten(1), dim=-1, eps=1.0e-8
                    )
                    latent_similarity = flat_latent @ flat_latent.transpose(0, 1)
                    cross_sample_cosine = (
                        latent_similarity.sum() - latent_similarity.diagonal().sum()
                    ) / max(batch_size * (batch_size - 1), 1)
                    paired_latent_cosine = (
                        flat_latent[0::2] * flat_latent[1::2]
                    ).sum(dim=-1)
                    lineage_mismatch_fraction = sum(
                        left != right
                        for left, right in zip(
                            cpu_batch["lineage_ids"][0::2],
                            cpu_batch["lineage_ids"][1::2],
                            strict=True,
                        )
                    ) / pairs_per_batch
                    checkpoint_mismatch_fraction = sum(
                        left != right
                        for left, right in zip(
                            cpu_batch["checkpoint_sha256"][0::2],
                            cpu_batch["checkpoint_sha256"][1::2],
                            strict=True,
                        )
                    ) / pairs_per_batch
                    row = {
                        "schema": SCHEMA,
                        "step": step,
                        "loss": float(total_loss.detach()),
                        "base_loss": float(losses["base_loss"].detach()),
                        **{
                            key: float(value)
                            for key, value in losses["base_parts"].items()
                        },
                        "preference_loss": float(losses["preference_loss"].detach()),
                        "representation_loss": float(losses["representation_loss"].detach()),
                        "weighted_preference_loss": float(weighted_preference.detach()),
                        "weighted_representation_loss": float(weighted_representation.detach()),
                        "auxiliary_ramp": ramp,
                        "lambda_preference": float(calibration["lambda_preference"]),
                        "lambda_representation": float(calibration["lambda_representation"]),
                        "positive_preference_target_loss": float(
                            losses["preference_stats"]["positive_total"].mean()
                        ),
                        "negative_preference_target_loss": float(
                            losses["preference_stats"]["negative_total"].mean()
                        ),
                        **{
                            f"positive_{key}_loss": float(positive[key].mean())
                            for key in (
                                "behavioral_direction",
                                "behavioral_scale",
                                "structural_direction",
                                "structural_scale",
                            )
                        },
                        **{
                            f"negative_{key}_loss": float(negative[key].mean())
                            for key in (
                                "behavioral_direction",
                                "behavioral_scale",
                                "structural_direction",
                                "structural_scale",
                            )
                        },
                        "preference_absolute_gap_mean": float(absolute_gap.mean()),
                        "preference_absolute_gap_p10": _tensor_quantile(absolute_gap, 0.10),
                        "preference_absolute_gap_median": _tensor_quantile(absolute_gap, 0.50),
                        "preference_absolute_gap_p90": _tensor_quantile(absolute_gap, 0.90),
                        "preference_relative_gap_mean": float(gap.mean()),
                        "preference_relative_gap_p10": _tensor_quantile(gap, 0.10),
                        "preference_relative_gap_median": _tensor_quantile(gap, 0.50),
                        "preference_relative_gap_p90": _tensor_quantile(gap, 0.90),
                        "preference_gap_p10": _tensor_quantile(gap, 0.10),
                        "preference_gap_median": _tensor_quantile(gap, 0.50),
                        "preference_gap_p90": _tensor_quantile(gap, 0.90),
                        "preference_margin_satisfied": float(
                            losses["preference_stats"]["margin_satisfied"]
                        ),
                        "representation_positive_cosine": float(
                            losses["representation_stats"]["positive_cosine"]
                        ),
                        "representation_negative_cosine": float(
                            losses["representation_stats"]["negative_cosine"]
                        ),
                        "representation_pairwise_accuracy": float(
                            losses["representation_stats"]["pairwise_accuracy"]
                        ),
                        **{
                            f"preference_delta_{key}_mean": float(value.mean())
                            for key, value in component_deltas.items()
                        },
                        **{
                            f"preference_{key}_abs_share_mean": float(
                                component_abs_shares[:, index].mean()
                            )
                            for index, key in enumerate(component_deltas)
                        },
                        "preference_max_component_abs_share_mean": float(
                            component_abs_shares.max(dim=-1).values.mean()
                        ),
                        "operator_relative_mse": float(operator_relative_mse),
                        "weight_rmse": float(weight_rmse),
                        "latent_rms": float(
                            losses["latent"].float().square().mean().sqrt()
                        ),
                        "latent_batch_std": float(
                            losses["latent"].float().std(dim=0, unbiased=False).mean()
                        ),
                        "latent_cross_sample_cosine": float(cross_sample_cosine),
                        "latent_paired_cosine_mean": float(paired_latent_cosine.mean()),
                        "pred_log_scale_mean": float(valid_scales.float().mean()),
                        "pred_log_scale_std": float(valid_scales.float().std(unbiased=False)),
                        "pre_clip_gradient_norm": float(pre_clip_norm),
                        "auxiliary_gradient_ratio": auxiliary_gradient_ratio,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "batch_size": batch_size,
                        "pairs_per_batch": pairs_per_batch,
                        "tile_size": 32,
                        "core_parameter_count": core_parameter_count,
                        "projector_parameter_count": projector_parameter_count,
                        "parameter_count": core_parameter_count + projector_parameter_count,
                        "pair_valid_fraction": 1.0,
                        "pair_lineage_mismatch_fraction": lineage_mismatch_fraction,
                        "pair_checkpoint_mismatch_fraction": checkpoint_mismatch_fraction,
                        "pair_same_checkpoint_index_fraction": 1.0,
                        "pair_weight_rmse_mean": float(pair_weight_distance.mean()),
                        "pair_weight_rmse_p10": _tensor_quantile(pair_weight_distance, 0.10),
                        "pair_weight_rmse_median": _tensor_quantile(pair_weight_distance, 0.50),
                        "pair_weight_rmse_p90": _tensor_quantile(pair_weight_distance, 0.90),
                        "pair_weight_relative_distance_mean": float(
                            pair_weight_relative_distance.mean()
                        ),
                        "pair_unique_lineages": len(set(cpu_batch["lineage_ids"])),
                        "distribution_unique_contexts": int(
                            batch["distribution_source_indices"].numel()
                        ),
                        "distribution_context_reuse": float(
                            batch_size / batch["distribution_source_indices"].numel()
                        ),
                        "emitted_anchor_logical_index": committed_cursor.emitted_logical_index,
                        "parent_logical_index": committed_cursor.parent_logical_index,
                        "next_subpatch_index": committed_cursor.subpatch_index,
                        "data_wait_seconds": data_wait_seconds,
                        "h2d_enqueue_seconds": h2d_enqueue_seconds,
                        "step_seconds": step_seconds,
                        "telemetry_window_steps": rate_window_steps,
                        "telemetry_window_seconds": rate_window_seconds,
                        "steps_per_second": rate_window_steps
                        / max(rate_window_seconds, 1.0e-9),
                        "examples_per_second": batch_size
                        * rate_window_steps
                        / max(rate_window_seconds, 1.0e-9),
                        "elapsed_seconds": elapsed,
                        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        **timing,
                }
                _append_jsonl(metrics_path, row)
                rate_window_started = time.monotonic()
                rate_window_step = step
            if gradient_due:
                _append_jsonl(
                    gradient_path,
                    {
                        "schema": SCHEMA,
                        "step": step,
                        "pre_clip_gradient_norm": float(pre_clip_norm),
                        "auxiliary_gradient_ratio": auxiliary_gradient_ratio,
                        "base_gradient_l2": base_gradient_l2,
                        "preference_gradient_l2": preference_gradient_l2,
                        "representation_gradient_l2": representation_gradient_l2,
                        "objective_gradient_groups": objective_gradient_groups,
                        "model_groups": _gradient_telemetry(model),
                        "projector_groups": _gradient_telemetry(projector),
                        "depth": losses["depth_rows"],
                    },
                )
            if log_due:
                assert row is not None
                print(
                    "[mini-latent-pref] stage=train-progress "
                    f"step={step}/{steps} total={row['loss']:.6f} base={row['base_loss']:.6f} "
                    f"pref={row['preference_loss']:.6f} repr={row['representation_loss']:.6f} "
                    f"gap_p10/med={row['preference_gap_p10']:.3f}/{row['preference_gap_median']:.3f} "
                    f"repr_acc={row['representation_pairwise_accuracy']:.3f} "
                    f"rate={row['steps_per_second']:.3f}_steps/s wait={data_wait_seconds:.3f}s "
                    f"peak_alloc_gib={row['max_cuda_allocated_bytes']/2**30:.2f}",
                    flush=True,
                )
            if step == 1 or step % int(config["diagnostics"]["fixed_probe_every"]) == 0:
                assert fixed_probe_cpu is not None
                probe_row = _evaluate_fixed_probe(
                    model,
                    projector,
                    fixed_probe_cpu,
                    normalization,
                    config,
                    device,
                    step=step,
                )
                _append_jsonl(probe_path, probe_row)
                print(
                    "[mini-latent-pref] stage=fixed-probe "
                    f"step={step} pos={probe_row['positive_loss_mean']:.6f} "
                    f"foreign={probe_row['matched_foreign_loss_mean']:.6f} "
                    f"gap_med={probe_row['matched_relative_gap_median']:.3f} "
                    f"rank={probe_row['latent_effective_rank']:.2f}",
                    flush=True,
                )
            if step % int(config["plot_every"]) == 0:
                plot_paths = _plot_diagnostics(
                    metrics_path, gradient_path, probe_path, output_root
                )
                print(
                    f"[mini-latent-pref] stage=plot-write paths={json.dumps(plot_paths, sort_keys=True)}",
                    flush=True,
                )
            resume_saved_this_step = False
            if step % int(config["resume_save_every"]) == 0:
                save_resume(step, committed_cursor)
                resume_saved_this_step = True
            if (
                retained_checkpoint_every > 0
                and step % retained_checkpoint_every == 0
            ):
                if not resume_saved_this_step:
                    save_resume(step, committed_cursor)
                retain_resume_checkpoint(
                    step,
                    committed_cursor,
                    reason="periodic",
                )
            if step % int(config["model_save_every"]) == 0:
                save_model(step, committed_cursor)
            if stop_requested:
                break

        save_resume(last_step, committed_cursor)
        save_model(last_step, committed_cursor)
        plot_paths = _plot_diagnostics(
            metrics_path, gradient_path, probe_path, output_root
        )
        status = (
            "diagnostic_stop"
            if diagnostic_stop_reason is not None
            else ("stopped" if stop_requested else "complete")
        )
        summary = {
            "schema": SCHEMA,
            "status": status,
            "reason": diagnostic_stop_reason,
            "step": last_step,
            "scientific_horizon_steps": int(config["steps"]),
            "subtile_cursor": committed_cursor.to_dict(),
            "core_parameter_count": core_parameter_count,
            "projector_parameter_count": projector_parameter_count,
            "elapsed_seconds": time.monotonic() - started,
            "metrics_path": str(metrics_path),
            "gradient_path": str(gradient_path),
            "probe_path": str(probe_path),
            "plot_paths": plot_paths,
            "resume_checkpoint": str(resume_path),
            "persistent_model_checkpoint": str(persistent_model_path),
            "subtile_stream": stream.telemetry(),
            "parent_stream": parent_stream.telemetry(),
            "pair_resolver": resolver.telemetry(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_name = "summary.json" if status == "complete" else "STOPPED.json"
        _atomic_json(output_root / summary_name, summary)
        print(
            "[mini-latent-pref] stage=complete "
            f"status={status} step={last_step} metrics={metrics_path} probe={probe_path} "
            f"plots={json.dumps(plot_paths, sort_keys=True)} "
            f"resume={resume_path} model={persistent_model_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
