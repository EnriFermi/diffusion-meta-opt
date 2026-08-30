from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer, operator_bank_data_pipeline
from training.big_vae.data_types import _identity_sample_collate
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    POLAR_TAIL_SCHEMA,
    PolarTailedUnifiedWeightBottleneck,
    _batch_from_stream,
    _load_config,
    _model_config,
    _polar_direction_scale_objectives,
    _polar_routed_production_loss,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    PreNormTransformerBlock,
    _seed_everything,
)


DEFAULT_CONFIG = Path(
    "conf/weightclip_benchmark/direct_normalized_scaled_700m_polar_tails_production_500k.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/dev/shm/weightclip_direct_normalized_scaled_700m_p32_polar_tails_500k_v1/"
    "resume_latest.pt"
)
DEFAULT_RUN_ROOT = Path(
    "/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mechanistic audit of the live polar-tail AE.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def _percentiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    if not flat.numel():
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    q = torch.quantile(flat, torch.tensor([0.05, 0.5, 0.95], device=flat.device))
    return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def _effective_rank(matrix: torch.Tensor, eps: float = 1.0e-12) -> float:
    matrix = matrix.detach().float()
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(eps)
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())


def _pairwise_cosine_summary(vectors: torch.Tensor) -> dict[str, float]:
    vectors = F.normalize(vectors.detach().float(), dim=-1, eps=1.0e-12)
    gram = vectors @ vectors.transpose(0, 1)
    upper = gram[torch.triu_indices(gram.shape[0], gram.shape[1], offset=1).unbind()]
    if not upper.numel():
        return {"min": 1.0, "mean": 1.0, "max": 1.0}
    return {
        "min": float(upper.min()),
        "mean": float(upper.mean()),
        "max": float(upper.max()),
    }


def _load_batch(config: dict[str, Any], cursor: int, batch_size: int) -> dict[str, Any]:
    print(
        f"[polar-mechanism] stage=data-open cursor={cursor} batch={batch_size}",
        flush=True,
    )
    bank = config["operator_bank"]
    logger = logging.getLogger("polar-mechanism")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
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
        sampler.set_start_index(cursor)
        loader = DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=0,
            collate_fn=_identity_sample_collate,
        )
        stream = BalancedOperatorBankMixer(dataset, iter(loader), start_index=cursor)
        batch = _batch_from_stream(stream, batch_size)
    return batch


def _attention_summary(
    block: PreNormTransformerBlock,
    state: torch.Tensor,
    key_context: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
) -> dict[str, float]:
    with torch.no_grad():
        batch, length, dim = state.shape
        qkv = block.qkv(block.attn_norm(state)).view(
            batch, length, 3, block.heads, block.head_dim
        )
        query, key, _value = qkv.unbind(dim=2)
        if block.context_key_projection is not None:
            if key_context is None:
                raise RuntimeError("missing key context in conditioned attention diagnostic")
            normalized_context = F.rms_norm(
                key_context.float(),
                (key_context.shape[-1],),
                weight=None,
                eps=1.0e-6,
            )
            context_key = block.context_key_projection(normalized_context).view(
                batch, length, block.heads, block.head_dim
            )
            key = key + context_key.to(key.dtype)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        if block.bounded_cosine_attention:
            query = F.normalize(query.float(), dim=-1, eps=1.0e-6)
            key = F.normalize(key.float(), dim=-1, eps=1.0e-6)
            score_scale = block.attention_logit_scale
        else:
            score_scale = 1.0 / math.sqrt(float(block.head_dim))

        latent_count = 32
        latent_indices = torch.arange(min(latent_count, length), device=state.device)
        token_start = min(latent_count, length)
        token_count = length - token_start
        if token_count > 0:
            sampled_tokens = torch.linspace(
                token_start,
                length - 1,
                steps=min(64, token_count),
                device=state.device,
            ).round().long().unique()
            query_indices = torch.cat((latent_indices, sampled_tokens)).unique()
        else:
            query_indices = latent_indices
        selected_query = query[:, :, query_indices].float()
        scores = torch.matmul(selected_query, key.float().transpose(-1, -2)) * score_scale
        if valid_mask is None:
            valid_mask = torch.ones(batch, length, device=state.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=state.device, dtype=torch.bool)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], -torch.inf)
        probability = torch.softmax(scores, dim=-1)
        query_valid = valid_mask[:, query_indices]
        valid_rows = query_valid[:, None, :].expand(-1, block.heads, -1)
        entropy = -(probability * torch.log(probability.clamp_min(1.0e-30))).sum(dim=-1)
        entropy_denominator = torch.log(valid_mask.sum(dim=-1).float().clamp_min(2.0))
        normalized_entropy = entropy / entropy_denominator[:, None, None]
        max_probability = probability.max(dim=-1).values

        latent_query_count = int((query_indices < latent_count).sum().item())
        latent_to_tokens = probability[:, :, :latent_query_count, latent_count:].sum(dim=-1)
        token_to_latent = probability[:, :, latent_query_count:, :latent_count].sum(dim=-1)
        token_query_valid = valid_rows[:, :, latent_query_count:]
        latent_query_valid = valid_rows[:, :, :latent_query_count]

        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
            return float(
                (value * mask).sum().div(mask.sum().clamp_min(1)).detach().float().item()
            )

        return {
            "sequence_length": float(length),
            "state_rms": _rms(state),
            "query_rms": _rms(query),
            "key_rms": _rms(key),
            "normalized_entropy": masked_mean(normalized_entropy, valid_rows),
            "max_probability": masked_mean(max_probability, valid_rows),
            "latent_query_to_token_mass": masked_mean(latent_to_tokens, latent_query_valid),
            "token_query_to_latent_mass": masked_mean(token_to_latent, token_query_valid),
        }


def _make_component_mask(
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    batch = d_in_mask.shape[0]
    p16_components = d_in_mask.view(batch, 8, 16)
    return d_out_mask[:, :, None, None] & p16_components[:, None]


def _direction_target_summary(
    W: torch.Tensor,
    pred_dirs: torch.Tensor,
    component_mask: torch.Tensor,
) -> dict[str, Any]:
    target = W.transpose(1, 2).float().reshape(W.shape[0], 128, 8, 16)
    target = target * component_mask
    target_norm = target.square().sum(dim=-1, keepdim=True).sqrt()
    target_dirs = target / target_norm.clamp_min(1.0e-12)
    valid = component_mask.any(dim=-1) & (target_norm.squeeze(-1) > 1.0e-12)
    matched_cosine = (pred_dirs.float() * target_dirs).sum(dim=-1)

    pred_pairs: list[torch.Tensor] = []
    target_pairs: list[torch.Tensor] = []
    for left in range(W.shape[0]):
        for right in range(left + 1, W.shape[0]):
            pair_valid = valid[left] & valid[right]
            if bool(pair_valid.any()):
                pred_pairs.append((pred_dirs[left][pair_valid] * pred_dirs[right][pair_valid]).sum(-1))
                target_pairs.append((target_dirs[left][pair_valid] * target_dirs[right][pair_valid]).sum(-1))
    pred_cross = torch.cat(pred_pairs) if pred_pairs else pred_dirs.new_empty(0)
    target_cross = torch.cat(target_pairs) if target_pairs else pred_dirs.new_empty(0)

    valid_pred = pred_dirs[valid]
    valid_target = target_dirs[valid]
    return {
        "matched_cosine_mean": float(matched_cosine[valid].mean()),
        "matched_cosine_percentiles": _percentiles(matched_cosine[valid]),
        "prediction_cross_sample_cosine_mean": (
            float(pred_cross.mean()) if pred_cross.numel() else 1.0
        ),
        "target_cross_sample_cosine_mean": (
            float(target_cross.mean()) if target_cross.numel() else 1.0
        ),
        "prediction_effective_rank_16d": _effective_rank(valid_pred),
        "target_effective_rank_16d": _effective_rank(valid_target),
    }


def _gradient_summary(gradient: torch.Tensor | None) -> dict[str, float | None]:
    if gradient is None:
        return {"rms": None, "norm": None}
    return {"rms": _rms(gradient), "norm": float(gradient.detach().float().norm())}


def _head_cancellation(
    logit_gradient: torch.Tensor,
    hidden: torch.Tensor,
) -> dict[str, Any]:
    gradient = logit_gradient.detach().float().reshape(logit_gradient.shape[0], -1, logit_gradient.shape[-1])
    hidden = hidden.detach().float().reshape(hidden.shape[0], -1, hidden.shape[-1])
    sample_gradients = torch.einsum("bti,bth->bih", gradient, hidden)
    total_gradient = sample_gradients.sum(dim=0)
    sample_norms = sample_gradients.flatten(1).norm(dim=1)
    token_outer_norm_sum = (
        gradient.norm(dim=-1) * hidden.norm(dim=-1)
    ).sum()
    return {
        "total_gradient_rms": _rms(total_gradient),
        "sample_resultant_over_sum": float(
            total_gradient.norm().div(sample_norms.sum().clamp_min(1.0e-30)).item()
        ),
        "token_resultant_over_sum": float(
            total_gradient.norm().div(token_outer_norm_sum.clamp_min(1.0e-30)).item()
        ),
        "sample_gradient_pairwise_cosine": _pairwise_cosine_summary(
            sample_gradients.flatten(1)
        ),
        "sample_gradient_norm_percentiles": _percentiles(sample_norms),
    }


def _intervention_summary(
    baseline_prediction: torch.Tensor,
    baseline_dirs: torch.Tensor,
    baseline_scales: torch.Tensor,
    prediction: torch.Tensor,
    dirs: torch.Tensor,
    scales: torch.Tensor,
    component_mask: torch.Tensor,
) -> dict[str, float]:
    valid_patch = component_mask.any(dim=-1)
    direction_cosine = (baseline_dirs.float() * dirs.float()).sum(dim=-1)
    baseline_rms = baseline_prediction.float().square().mean().sqrt()
    return {
        "direction_change": float((1.0 - direction_cosine[valid_patch]).mean()),
        "scale_abs_change": float(
            (baseline_scales.float() - scales.float()).abs()[valid_patch].mean()
        ),
        "output_relative_rms_change": float(
            (prediction.float() - baseline_prediction.float())
            .square()
            .mean()
            .sqrt()
            .div(baseline_rms.clamp_min(1.0e-12))
        ),
    }


def _snapshot(
    model: PolarTailedUnifiedWeightBottleneck,
    batch: dict[str, Any],
    config: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    W = batch["W"].to(device)
    X = batch["X"].to(device)
    x_mask = batch["x_mask"].to(device)
    d_in_mask = batch["d_in_mask"].to(device)
    d_out_mask = batch["d_out_mask"].to(device)
    tile_row = batch["tile_row"].to(device)
    tile_col = batch["tile_col"].to(device)
    content, log_scale, token_valid = _prepare_normalized_inputs(
        W,
        d_in_mask,
        d_out_mask,
        scale_mean=float(config["normalization"]["log2_scale_mean"]),
        scale_std=float(config["normalization"]["log2_scale_std"]),
    )
    content = content.to(device)
    log_scale = log_scale.to(device)
    token_valid = token_valid.to(device)

    group_ids = torch.arange(128, device=device).repeat_interleave(model.chunks_per_group)
    chunk_ids = torch.arange(model.chunks_per_group, device=device).repeat(128)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        content_embedding = model._content_embedding(content)
        scale_embedding = model.scale_mlp(log_scale.float()).to(content_embedding.dtype)
        group_embedding = model.group_embedding(group_ids)[None]
        chunk_embedding = model.chunk_embedding(chunk_ids)[None]
        tile_embedding = model._tile_position_embedding(tile_row, tile_col)[:, None]
    input_components = {
        "content_rms": _rms(content_embedding),
        "scale_rms": _rms(scale_embedding),
        "group_rms": _rms(group_embedding),
        "chunk_rms": _rms(chunk_embedding),
        "tile_rms": _rms(tile_embedding),
        "combined_rms": _rms(
            content_embedding + scale_embedding + group_embedding + chunk_embedding + tile_embedding
        ),
    }

    attention: dict[str, dict[str, float]] = {}
    block_outputs: dict[str, torch.Tensor] = {}
    head_capture: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    named_blocks: list[tuple[str, nn.Module]] = []
    named_blocks.extend(
        (f"encoder_{depth:02d}", block)
        for depth, block in enumerate(model.encoder_blocks, start=1)
    )
    named_blocks.extend(
        (f"shared_decoder_{depth:02d}", block)
        for depth, block in enumerate(model.decoder_blocks, start=1)
    )
    named_blocks.extend(
        [("direction_tail", model.direction_tail), ("scale_tail", model.scale_tail)]
    )

    def block_hook(name: str):
        def hook(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            kwargs: dict[str, Any],
            output: torch.Tensor,
        ) -> None:
            state = args[0]
            attention[name] = _attention_summary(
                module,
                state.detach(),
                kwargs.get("key_context"),
                kwargs.get("valid_mask"),
            )
            attention[name]["residual_write_rms"] = _rms(output.detach() - state.detach())
            attention[name]["residual_write_over_state"] = (
                attention[name]["residual_write_rms"]
                / max(attention[name]["state_rms"], 1.0e-30)
            )
            block_outputs[name] = output

        return hook

    for name, block in named_blocks:
        handles.append(block.register_forward_hook(block_hook(name), with_kwargs=True))

    def head_hook(name: str):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            head_capture[f"{name}_hidden"] = args[0]
            head_capture[f"{name}_logits"] = output

        return hook

    handles.append(model.direction_head.register_forward_hook(head_hook("direction")))
    handles.append(model.scale_head.register_forward_hook(head_hook("scale")))

    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction, z, _depth, pred_dirs, pred_scales = model(
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
        direction_loss, scale_loss = _polar_direction_scale_objectives(
            X,
            W,
            pred_dirs,
            pred_scales,
            x_mask,
            d_in_mask,
            d_out_mask,
            config["loss"],
        )
        total_loss, parts = _polar_routed_production_loss(
            X,
            W,
            pred_dirs,
            pred_scales,
            x_mask,
            d_in_mask,
            d_out_mask,
            config["loss"],
        )
    for handle in handles:
        handle.remove()

    direction_logits = head_capture["direction_logits"]
    direction_hidden = head_capture["direction_hidden"]
    scale_logits = head_capture["scale_logits"]
    scale_hidden = head_capture["scale_hidden"]
    activation_names = list(block_outputs) + ["latent", "direction_logits", "scale_logits"]
    activation_tensors = list(block_outputs.values()) + [z, direction_logits, scale_logits]
    direction_gradients = torch.autograd.grad(
        direction_loss,
        activation_tensors,
        retain_graph=True,
        allow_unused=True,
    )
    scale_gradients = torch.autograd.grad(
        scale_loss,
        activation_tensors,
        retain_graph=True,
        allow_unused=True,
    )
    activation_gradients = {
        name: {
            "direction": _gradient_summary(direction_gradient),
            "scale": _gradient_summary(scale_gradient),
        }
        for name, direction_gradient, scale_gradient in zip(
            activation_names,
            direction_gradients,
            scale_gradients,
            strict=True,
        )
    }
    direction_logit_gradient = direction_gradients[-2]
    scale_logit_gradient = scale_gradients[-1]
    if direction_logit_gradient is None or scale_logit_gradient is None:
        raise RuntimeError("head logit gradients unexpectedly absent")
    cancellation = {
        "direction_head": _head_cancellation(direction_logit_gradient, direction_hidden),
        "scale_head": _head_cancellation(scale_logit_gradient, scale_hidden),
    }

    component_mask = _make_component_mask(d_in_mask, d_out_mask)
    valid_patch = component_mask.any(dim=-1)
    raw_direction_logits = direction_logits.view_as(pred_dirs)
    logit_norm = raw_direction_logits.float().square().sum(dim=-1).sqrt()
    head_geometry = {
        "direction_head_weight_rms": _rms(model.direction_head.weight),
        "scale_head_weight_rms": _rms(model.scale_head.weight),
        "direction_logit_norm": _percentiles(logit_norm[valid_patch]),
        "direction_logit_rms": _rms(raw_direction_logits[component_mask]),
        "direction_hidden_rms": _rms(direction_hidden),
        "scale_hidden_rms": _rms(scale_hidden),
        "direction_logit_gradient_rms": _rms(direction_logit_gradient),
        "scale_logit_gradient_rms": _rms(scale_logit_gradient),
    }

    amplitude_counterfactual: dict[str, Any] = {}
    detached_hidden = direction_hidden.detach().float()
    for multiplier in (0.1, 1.0, 10.0):
        leaf_logits = (raw_direction_logits.detach().float() * multiplier).requires_grad_(True)
        masked_logits = leaf_logits * component_mask
        leaf_norm = (masked_logits.square().sum(dim=-1, keepdim=True) + 1.0e-12).sqrt()
        leaf_dirs = masked_logits / leaf_norm
        leaf_direction_loss, _ = _polar_direction_scale_objectives(
            X,
            W,
            leaf_dirs,
            pred_scales.detach(),
            x_mask,
            d_in_mask,
            d_out_mask,
            config["loss"],
        )
        leaf_gradient = torch.autograd.grad(leaf_direction_loss, leaf_logits)[0]
        proxy_head_gradient = torch.einsum(
            "bti,bth->ih",
            leaf_gradient.reshape(leaf_gradient.shape[0], -1, 16),
            detached_hidden,
        )
        amplitude_counterfactual[str(multiplier)] = {
            "direction_loss": float(leaf_direction_loss.detach()),
            "logit_gradient_rms": _rms(leaf_gradient),
            "head_gradient_proxy_rms": _rms(proxy_head_gradient),
        }

    # Remove the main graph before decoder interventions.
    del direction_gradients, scale_gradients
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dist = model._encode_distribution_context(X, sample_mask=x_mask)
        interventions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        interventions["latent_roll"] = model.decode_polar(
            torch.roll(z.detach(), shifts=1, dims=0),
            tile_row,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=dist,
        )
        interventions["latent_zero"] = model.decode_polar(
            torch.zeros_like(z),
            tile_row,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=dist,
        )
        interventions["decoder_context_roll"] = model.decode_polar(
            z.detach(),
            tile_row,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=torch.roll(dist, shifts=1, dims=0),
        )
    intervention_rows: dict[str, Any] = {}
    for name, (current_prediction, current_dirs, current_scales) in interventions.items():
        summary = _intervention_summary(
            prediction.detach(),
            pred_dirs.detach(),
            pred_scales.detach(),
            current_prediction,
            current_dirs,
            current_scales,
            component_mask,
        )
        with torch.no_grad():
            current_loss, current_parts = _polar_routed_production_loss(
                X,
                W,
                current_dirs,
                current_scales,
                x_mask,
                d_in_mask,
                d_out_mask,
                config["loss"],
            )
        summary["total_loss"] = float(current_loss)
        summary["behavioral_direction"] = float(current_parts["behavioral_direction"])
        summary["structural_direction"] = float(current_parts["structural_direction"])
        intervention_rows[name] = summary

    return {
        "label": label,
        "loss": {
            "total": float(total_loss.detach()),
            "weighted_direction": float(direction_loss.detach()),
            "weighted_scale": float(scale_loss.detach()),
            **{key: float(value) for key, value in parts.items()},
        },
        "input_components": input_components,
        "attention": attention,
        "activation_gradients": activation_gradients,
        "head_geometry": head_geometry,
        "head_cancellation": cancellation,
        "amplitude_counterfactual": amplitude_counterfactual,
        "latent": {
            "rms": _rms(z),
            "effective_rank": _effective_rank(z.reshape(-1, z.shape[-1])),
            "cross_sample_slot_cosine": _pairwise_cosine_summary(
                z.detach().float().mean(dim=1)
            ),
        },
        "direction_representation": _direction_target_summary(
            W,
            pred_dirs,
            component_mask,
        ),
        "interventions": intervention_rows,
    }


def _write_tables(output_root: Path, report: dict[str, Any]) -> None:
    attention_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    for label in ("initial", "trained"):
        snapshot = report[label]
        for block, values in snapshot["attention"].items():
            attention_rows.append({"snapshot": label, "block": block, **values})
        for activation, values in snapshot["activation_gradients"].items():
            gradient_rows.append(
                {
                    "snapshot": label,
                    "activation": activation,
                    "direction_rms": values["direction"]["rms"],
                    "scale_rms": values["scale"]["rms"],
                }
            )
        for intervention, values in snapshot["interventions"].items():
            intervention_rows.append(
                {"snapshot": label, "intervention": intervention, **values}
            )
    for name, rows in (
        ("attention.csv", attention_rows),
        ("activation_gradients.csv", gradient_rows),
        ("interventions.csv", intervention_rows),
    ):
        path = output_root / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _plot(output_root: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    blocks = list(report["trained"]["attention"])
    x = np.arange(len(blocks))
    for label, style in (("initial", "--"), ("trained", "-")):
        values = report[label]["attention"]
        axes[0, 0].plot(
            x,
            [values[name]["normalized_entropy"] for name in blocks],
            style,
            label=label,
        )
        axes[0, 1].plot(
            x,
            [values[name]["token_query_to_latent_mass"] for name in blocks],
            style,
            label=label,
        )
    axes[0, 0].set_title("Attention normalized entropy")
    axes[0, 1].set_title("Token-query attention mass to latent slots")
    for axis in axes[0]:
        axis.set_xticks(x)
        axis.set_xticklabels(blocks, rotation=75, ha="right", fontsize=7)
        axis.grid(alpha=0.25)
        axis.legend()

    gradient_names = [
        name
        for name in report["trained"]["activation_gradients"]
        if name.startswith(("encoder_", "shared_decoder_"))
    ]
    gx = np.arange(len(gradient_names))
    trained_gradients = report["trained"]["activation_gradients"]
    axes[1, 0].plot(
        gx,
        [max(trained_gradients[name]["direction"]["rms"] or 0.0, 1.0e-16) for name in gradient_names],
        label="direction",
    )
    axes[1, 0].plot(
        gx,
        [max(trained_gradients[name]["scale"]["rms"] or 0.0, 1.0e-16) for name in gradient_names],
        label="scale",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Trained loss gradient at block outputs")
    axes[1, 0].set_xticks(gx)
    axes[1, 0].set_xticklabels(gradient_names, rotation=75, ha="right", fontsize=7)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    interventions = list(report["trained"]["interventions"])
    ix = np.arange(len(interventions))
    width = 0.35
    axes[1, 1].bar(
        ix - width / 2,
        [report["trained"]["interventions"][name]["direction_change"] for name in interventions],
        width,
        label="direction change",
    )
    axes[1, 1].bar(
        ix + width / 2,
        [report["trained"]["interventions"][name]["scale_abs_change"] for name in interventions],
        width,
        label="log-scale change",
    )
    axes[1, 1].set_xticks(ix)
    axes[1, 1].set_xticklabels(interventions, rotation=20, ha="right")
    axes[1, 1].set_title("Decoder intervention sensitivity")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output_root / "mechanism_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    run_root = args.run_root.resolve()
    config = _load_config(config_path)
    if config["schema"] != POLAR_TAIL_SCHEMA:
        raise ValueError("mechanistic audit requires the polar-tail schema")
    checkpoint_stat = checkpoint_path.stat()
    print(
        "[polar-mechanism] stage=checkpoint-load "
        f"path={checkpoint_path} size={checkpoint_stat.st_size}",
        flush=True,
    )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    checkpoint_step = int(payload["step"])
    cursor = int(payload["committed_logical_index"])
    if cursor != checkpoint_step * int(config["batch_size"]):
        raise RuntimeError("checkpoint cursor does not match production step")
    output_root = run_root / f"mechanistic_step_{checkpoint_step:07d}_v1"
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    _atomic_json(
        output_root / "RUNNING.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "config": str(config_path),
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    batch = _load_batch(config, cursor, args.batch_size)
    batch_identity = {
        "logical_indices": batch["logical_indices"],
        "tile_rows": batch["tile_row"].tolist(),
        "tile_cols": batch["tile_col"].tolist(),
    }

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    torch.set_float32_matmul_precision("high")
    model_cfg = _model_config(config)

    print("[polar-mechanism] stage=initial-snapshot", flush=True)
    _seed_everything(int(config["seed"]))
    initial_model = PolarTailedUnifiedWeightBottleneck(model_cfg, "normalized_float").to(device)
    initial = _snapshot(initial_model, batch, config, "initial")
    del initial_model
    torch.cuda.empty_cache()

    print(f"[polar-mechanism] stage=trained-snapshot step={checkpoint_step}", flush=True)
    _seed_everything(int(config["seed"]))
    trained_model = PolarTailedUnifiedWeightBottleneck(model_cfg, "normalized_float").to(device)
    trained_model.load_state_dict(payload["model_state"], strict=True)
    del payload
    trained = _snapshot(trained_model, batch, config, "trained")
    del trained_model
    torch.cuda.empty_cache()

    report = {
        "schema": "weightclip_polar_tail_mechanistic_audit_v1",
        "checkpoint_step": checkpoint_step,
        "checkpoint_stat": {
            "path": str(checkpoint_path),
            "size": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
        },
        "batch": batch_identity,
        "initial": initial,
        "trained": trained,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_root / "report.json", report)
    _write_tables(output_root, report)
    _plot(output_root, report)
    (output_root / "RUNNING.json").unlink()
    _atomic_json(
        output_root / "COMPLETE.json",
        {
            "complete": True,
            "checkpoint_step": checkpoint_step,
            "report": str(output_root / "report.json"),
            "plot": str(output_root / "mechanism_summary.png"),
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(
        "[polar-mechanism] stage=complete "
        f"step={checkpoint_step} output={output_root} elapsed={report['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
