from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from omegaconf import OmegaConf

from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.operator_set_overfit import CanonicalOperatorSetTrainingDataset
from training.big_vae.two_operator_overfit import _loss_metrics, _nrmse


SCHEMA = "weightclip_ae_v9a_final_causal_bundle_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-update final-checkpoint V9-A causal bundle. It separates encoder "
            "routing/basis loss, carrier insufficiency, bridge/decoder contraction, "
            "and cross-operator gradient conflict."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--gradient-pair-count", type=int, default=2)
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("/mnt/shared/weightclip_benchmark/diagnostics"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def _relative_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_f, right_f = left.detach().float(), right.detach().float()
    absolute = (left_f - right_f).square().mean().sqrt()
    denominator = torch.sqrt(0.5 * (left_f.square().mean() + right_f.square().mean())).clamp_min(1e-24)
    return {"absolute_rms": float(absolute), "relative_rms": float(absolute / denominator)}


def _tensor_summary(value: torch.Tensor) -> dict[str, float | list[int]]:
    flat = value.detach().float().reshape(value.shape[0], -1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    centered_rms = centered.square().mean().sqrt()
    singular = torch.linalg.svdvals(centered) if min(centered.shape) > 1 else torch.zeros(1)
    stable_rank = singular.square().sum() / singular.square().amax().clamp_min(1e-24)
    return {
        "shape": list(value.shape),
        "rms": _rms(value),
        "batch_centered_rms": float(centered_rms),
        "batch_centered_fraction": float(centered_rms / flat.square().mean().sqrt().clamp_min(1e-24)),
        "batch_stable_rank": float(stable_rank),
    }


def _amp(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _loss_cfg(cfg: Any) -> dict[str, float | int]:
    row = cfg.train.struct_loss
    return {
        "patch_size": int(cfg.model.patch_size),
        "gamma": float(row.gamma),
        "lambda_dir": float(row.lambda_dir),
        "lambda_scale": float(row.lambda_scale),
        "huber_delta": float(row.huber_delta),
    }


def _structural_loss(target: torch.Tensor, prediction: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    total, _ = WeightQuantileVAE.patch_structure_loss(
        target,
        prediction,
        patch_size=int(cfg["patch_size"]),
        gamma=float(cfg["gamma"]),
        lambda_dir=float(cfg["lambda_dir"]),
        lambda_scale=float(cfg["lambda_scale"]),
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=float(cfg["huber_delta"]),
    )
    return total


def _pair_batch(
    source: CanonicalOperatorSetTrainingDataset,
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> tuple[dict[str, torch.Tensor], list[tuple[tuple[str, str], int]], torch.Tensor]:
    rows_by_key = {key: source._materialize_group(key, None) for key in pair}
    tile_count = len(rows_by_key[pair[0]])
    assignments = [(key, local) for local in range(tile_count) for key in pair]
    rows = [rows_by_key[key][local] for key, local in assignments]
    batch = {
        "W": torch.stack([row.weight for row in rows]),
        "x": torch.stack([row.x for row in rows]),
        "x_mask": torch.stack([row.meta["x_mask"] for row in rows]),
        "d_in_mask": torch.stack([row.meta["d_in_mask"] for row in rows]),
        "d_out_mask": torch.stack([row.meta["d_out_mask"] for row in rows]),
    }
    swap = torch.tensor([index ^ 1 for index in range(len(rows))], dtype=torch.long)
    return batch, assignments, swap


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _exact_v8_routing(
    readout: torch.nn.Module,
    x_context: torch.Tensor,
    patch_mask: torch.Tensor,
    output_mask: torch.Tensor,
) -> torch.Tensor:
    batch, tiles, _ = x_context.shape
    outputs = output_mask.shape[1]
    patch_valid = patch_mask.bool()
    output_valid = output_mask.bool()
    key_valid = (output_valid.unsqueeze(-1) & patch_valid.unsqueeze(1)).reshape(batch, outputs * tiles)
    weights = patch_valid.to(x_context.dtype).unsqueeze(-1)
    x_global = (x_context * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    x_keys = x_context.unsqueeze(1).expand(-1, outputs, -1, -1).reshape(batch, outputs * tiles, -1)
    q = readout.q_proj(x_global).view(batch, readout.n_heads, readout.head_dim)
    k = readout.k_proj(x_keys).view(batch, outputs * tiles, readout.n_heads, readout.head_dim).transpose(1, 2)
    raw = torch.einsum("bhd,bhsd->bhs", q.float(), k.float()) / math.sqrt(float(readout.head_dim))
    learned = (0.1 * torch.tanh(raw / 0.1)).unsqueeze(2).expand(-1, -1, readout.num_latents, -1)
    fixed = readout._fixed_anchor_scores(key_valid).view(
        batch, readout.num_latents, readout.n_heads, outputs * tiles
    ).transpose(1, 2)
    return torch.softmax(fixed + learned, dim=-1)


def _exact_v9_routing(
    block: torch.nn.Module,
    state: torch.Tensor,
    w_patches: torch.Tensor,
    x_context: torch.Tensor,
    patch_mask: torch.Tensor,
    output_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, outputs, tiles, patch = w_patches.shape
    key_valid = (output_mask.bool().unsqueeze(-1) & patch_mask.bool().unsqueeze(1)).reshape(
        batch, outputs * tiles
    )
    query = block.query_proj(block.state_norm(state)).view(
        batch, block.num_latents, block.n_heads, block.router_head_dim
    ).transpose(1, 2)
    query = F.normalize(query.float(), dim=-1, eps=1e-6)
    raw_values = w_patches.reshape(batch, outputs * tiles, patch) * key_valid.unsqueeze(-1)
    x_keys = x_context.unsqueeze(1).expand(-1, outputs, -1, -1).reshape(batch, outputs * tiles, -1)
    weight_keys = block.weight_key_proj(block.weight_key_norm(raw_values))
    norm_feature = block._relative_log_rms_feature(raw_values, key_valid)
    keys = weight_keys + block.weight_norm_key_proj(norm_feature.to(block.weight_norm_key_proj.weight.dtype))
    keys = keys + block.context_key_proj(block.context_key_norm(x_keys))
    keys = keys.view(batch, outputs * tiles, block.n_heads, block.router_head_dim).transpose(1, 2)
    scores = block.score_cap * torch.einsum("bhld,bhsd->bhls", query, F.normalize(keys.float(), dim=-1, eps=1e-6))
    fixed = block._fixed_anchor_scores(key_valid).view(
        batch, block.num_latents, block.n_heads, outputs * tiles
    ).transpose(1, 2)
    adaptive = torch.softmax(fixed + scores, dim=-1)
    protected = block._fixed_anchor_scores(key_valid, protect_global_diversity=True).view(
        batch, block.num_latents, block.n_heads, outputs * tiles
    ).transpose(1, 2)
    protected = torch.softmax(protected, dim=-1)
    routing = block.protected_anchor_floor * protected + (1.0 - block.protected_anchor_floor) * adaptive
    return routing, protected


def _routing_summary(routing: torch.Tensor) -> dict[str, float]:
    probability = routing.float().clamp_min(1e-30)
    entropy = -(probability * probability.log()).sum(dim=-1)
    support = torch.arange(routing.shape[-1], device=routing.device, dtype=torch.float32)
    support = support / max(int(routing.shape[-1]) - 1, 1)
    mean = (probability * support).sum(dim=-1)
    variance = (probability * (support - mean.unsqueeze(-1)).square()).sum(dim=-1)
    return {
        "entropy_mean": float(entropy.mean()),
        "effective_keys_median": float(entropy.exp().median()),
        "barycenter_mean": float(mean.mean()),
        "barycenter_std": float(mean.std()),
        "support_variance_mean": float(variance.mean()),
        "max_mass_median": float(probability.amax(dim=-1).median()),
    }


def _support_alignment(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    # Route rows are [head, latent]. Compare normalized overlap to the same
    # route and to the independently best route; no full routing is persisted.
    left = actual.float().flatten(1, 2)
    right = reference.float().flatten(1, 2)
    left = F.normalize(left, dim=-1, eps=1e-12)
    right = F.normalize(right, dim=-1, eps=1e-12)
    overlap = torch.einsum("brs,bts->brt", left, right)
    diagonal = overlap.diagonal(dim1=-2, dim2=-1)
    best = overlap.amax(dim=-1)
    best_index = overlap.argmax(dim=-1)
    route = torch.arange(overlap.shape[-1], device=overlap.device).view(1, -1)
    return {
        "diagonal_overlap_mean": float(diagonal.mean()),
        "best_match_overlap_mean": float(best.mean()),
        "diagonal_is_best_fraction": float((best_index == route).float().mean()),
        "best_minus_diagonal_mean": float((best - diagonal).mean()),
    }


def _carrier_synthesis(
    carrier: torch.Tensor,
    routing: torch.Tensor,
    target_w_patches: torch.Tensor,
    out_weight: torch.Tensor,
    *,
    run_pinv: bool,
    loss_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    batch, latents, d_latent = carrier.shape
    heads, patch = routing.shape[1], target_w_patches.shape[-1]
    inverse_out = torch.linalg.pinv(out_weight.float(), rtol=1e-5)
    measurements = F.linear(carrier.float(), inverse_out).view(batch, latents, heads, patch)
    measurements = measurements.permute(0, 2, 1, 3).reshape(batch, heads * latents, patch)
    design = routing.float().reshape(batch, heads * latents, -1)
    target = target_w_patches.float().reshape(batch, -1, patch)
    transpose = torch.bmm(design.transpose(1, 2), measurements)
    coverage = design.square().sum(dim=1).unsqueeze(-1).clamp_min(1e-12)
    transpose = transpose / coverage

    def quality(value: torch.Tensor, reference: torch.Tensor = target) -> dict[str, float]:
        target_flat = reference.reshape(-1, patch)
        value_flat = value.reshape(-1, patch)
        cosine = F.cosine_similarity(target_flat, value_flat, dim=-1)
        target_matrix = reference.view(reference.shape[0], target_w_patches.shape[1], -1, patch)
        target_matrix = target_matrix.flatten(2).transpose(1, 2)
        value_matrix = value.view(value.shape[0], target_w_patches.shape[1], -1, patch)
        value_matrix = value_matrix.flatten(2).transpose(1, 2)
        structural = _loss_metrics(
            target_matrix,
            value_matrix,
            patch_size=patch,
            gamma=float(loss_cfg["gamma"]),
            lambda_dir=float(loss_cfg["lambda_dir"]),
            lambda_scale=float(loss_cfg["lambda_scale"]),
            huber_delta=float(loss_cfg["huber_delta"]),
        )
        return {
            "direction_loss": float((1.0 - cosine).mean()),
            "weighted_direction_loss": structural["dir"],
            "log_scale_huber": structural["scale"],
            "nrmse": _nrmse(reference, value),
            "target_rms": _rms(reference),
            "prediction_rms": _rms(value),
        }

    result: dict[str, Any] = {
        "measurement_scope": "canonical 128x128 tiles; exact frozen structural patch weighting",
        "out_proj_inverse_condition": float(torch.linalg.cond(out_weight.float())),
        "coverage_normalized_transpose": quality(transpose),
    }
    singular = torch.linalg.svdvals(design[0])
    result["routing_rank_condition_sample0"] = {
        "rank_at_1e_5_relative": int((singular > singular[0] * 1e-5).sum()),
        "row_count": int(design.shape[1]),
        "key_count": int(design.shape[2]),
        "condition_retained": float(singular[0] / singular[singular > singular[0] * 1e-5][-1]),
        "stable_rank": float(singular.square().sum() / singular[0].square()),
    }
    if run_pinv:
        # Exact X-only A0 repeats over the nine W tiles of one operator. Verify
        # that premise, solve once per parity/operator, and apply the same
        # inverse to every corresponding measurement. This cannot introduce a
        # W-conditioned routing sidechannel.
        parity_repeat_errors = []
        for parity in (0, 1):
            indices = torch.arange(parity, batch, 2, device=design.device)
            reference = design[parity]
            parity_repeat_errors.append(
                float((design.index_select(0, indices) - reference).abs().amax())
            )
        result["x_only_routing_repeat_max_abs_by_parity"] = parity_repeat_errors
        if max(parity_repeat_errors) <= 1e-6:
            inverse_by_parity = [
                torch.linalg.pinv(design[parity], rtol=1e-4) for parity in (0, 1)
            ]
            recovered = torch.stack(
                [inverse_by_parity[index % 2] @ measurements[index] for index in range(batch)]
            )
            result["least_squares_x_only_full_batch"] = quality(recovered)
            result["least_squares_status"] = "complete_exact_A0_reuse_by_operator"
        else:
            result["least_squares_status"] = (
                "skipped_fail_closed_A0_does_not_repeat_by_operator; "
                "no mismatched inverse was applied"
            )
    return result


def _capture_forward(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any], list[torch.Tensor]]:
    refiner = model.hybrid_content_readout_v9
    writes: list[torch.Tensor] = []
    handles = [block.register_forward_hook(lambda _m, _i, output: writes.append(output.detach())) for block in refiner.blocks]
    refiner.capture_stage_states = True
    try:
        prediction, _mu, _logvar, _dirs, debug = model.forward_debug(
            batch["W"],
            batch["x"],
            x_mask=batch["x_mask"],
            d_in_mask=batch["d_in_mask"],
            d_out_mask=batch["d_out_mask"],
            disable_z_shortcut=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if len(writes) != len(refiner.blocks):
        raise RuntimeError(f"captured {len(writes)} V9-A writes, expected {len(refiner.blocks)}")
    debug = dict(debug)
    debug["carrier"] = refiner.last_carrier_state
    debug["adaptive"] = refiner.last_adaptive_state
    return prediction, debug, writes


def _decode_latent_with_boundaries(
    model: torch.nn.Module,
    latent: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    debug: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = latent.shape[0]
    d_in, d_out = int(batch["W"].shape[1]), int(batch["W"].shape[2])
    T, d_in_pad = int(debug["T"]), int(debug["d_in_pad"])
    q_base, q_pos_emb, q_pos_o, q_pos_t = model._build_decoder_query_state(
        batch_size=batch_size,
        dist_patch_by_patch=debug["dist_patch_by_patch"],
        d_out=d_out,
        T=T,
    )
    query_mask = (
        debug["patch_mask"].unsqueeze(1).expand(-1, d_out, -1) & batch["d_out_mask"].unsqueeze(-1)
    ).reshape(batch_size, d_out * T)
    state = model.mandatory_latent_bridge(q_base, latent, query_mask=query_mask)
    boundaries = {"bridge_pre_film": state}
    state = model.position_only_film_v6(state, pos_o=q_pos_o, pos_t=q_pos_t, query_mask=query_mask)
    boundaries["bridge_post_film"] = state
    for index, layer in enumerate(model.decoder_layers):
        state = layer(state, q_pos=q_pos_o, q_pos2=q_pos_t, q_mask=query_mask)
        boundaries[f"decoder_{index:02d}"] = state
    q_norm = model.q_tokens_norm(state)
    boundaries["q_norm"] = q_norm
    boundaries["direction_logits"] = model.direction_head(q_norm)
    boundaries["scale_logits"] = model.scale_head(q_norm)
    prediction = model._decode_query_tokens_to_output(
        state,
        z=latent.flatten(1),
        q_pos_emb=q_pos_emb,
        d_in=d_in,
        d_out=d_out,
        d_in_pad=d_in_pad,
        T=T,
        patch_mask=debug["patch_mask"],
        d_in_mask=batch["d_in_mask"],
        d_out_mask=batch["d_out_mask"],
        disable_z_shortcut=True,
    )[0]
    boundaries["W_hat"] = prediction
    return prediction, boundaries


def _branch_and_boundary_panel(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    debug: Mapping[str, Any],
    swap: torch.Tensor,
    loss_cfg: Mapping[str, Any],
    reference_prediction: torch.Tensor,
    *,
    run_target_ladder: bool,
) -> dict[str, Any]:
    carrier = debug["carrier"]
    adaptive = debug["adaptive"]
    if carrier is None or adaptive is None:
        raise RuntimeError("V9-A carrier/adaptive capture missing")
    branches = {
        "carrier_only": 0.9 * carrier,
        "adaptive_only": adaptive,
        "combined": 0.9 * carrier + adaptive,
        "carrier_plus_swapped_adaptive": 0.9 * carrier + adaptive.index_select(0, swap),
        "swapped_combined": (0.9 * carrier + adaptive).index_select(0, swap),
        "zero": torch.zeros_like(carrier),
    }
    outputs: dict[str, torch.Tensor] = {}
    boundaries: dict[str, dict[str, torch.Tensor]] = {}
    for name, latent in branches.items():
        outputs[name], boundaries[name] = _decode_latent_with_boundaries(model, latent, batch, debug)
    combined_latent = branches["combined"]
    captured_latent = debug["latent_decoder_z"].view_as(combined_latent)
    latent_parity_max_abs = float((combined_latent.float() - captured_latent.float()).abs().amax())
    output_parity = _relative_delta(outputs["combined"], reference_prediction)
    output_parity_max_abs = float(
        (outputs["combined"].float() - reference_prediction.float()).abs().amax()
    )
    if (
        not math.isfinite(output_parity_max_abs)
        or latent_parity_max_abs > 1e-6
        or output_parity_max_abs > 1e-5
        or output_parity["relative_rms"] > 1e-5
    ):
        raise RuntimeError(
            "manual branch decode does not reproduce captured production forward: "
            f"latent_max={latent_parity_max_abs:.6g} output_max={output_parity_max_abs:.6g} "
            f"output_rel={output_parity['relative_rms']:.6g}"
        )
    result: dict[str, Any] = {
        "production_path_parity": {
            "latent_max_abs": latent_parity_max_abs,
            "output_max_abs": output_parity_max_abs,
            **output_parity,
            "max_abs_tolerance": 1e-5,
            "relative_rms_tolerance": 1e-5,
        },
        "branches": {
            name: {
                "loss": _loss_metrics(batch["W"], value, **loss_cfg),
                "nrmse": _nrmse(batch["W"], value),
                "output": _tensor_summary(value),
            }
            for name, value in outputs.items()
        },
        "combined_vs_carrier": _relative_delta(outputs["combined"], outputs["carrier_only"]),
        "combined_vs_adaptive": _relative_delta(outputs["combined"], outputs["adaptive_only"]),
        "adaptive_shuffle": _relative_delta(outputs["combined"], outputs["carrier_plus_swapped_adaptive"]),
        "total_swap": _relative_delta(outputs["combined"], outputs["swapped_combined"]),
        "boundary_swap_ladder": {},
        "boundary_target_ladder": {
            "status": "complete_first_pair" if run_target_ladder else "skipped_bounded_cost_other_pair"
        },
    }
    for name in boundaries["combined"]:
        result["boundary_swap_ladder"][name] = _relative_delta(
            boundaries["combined"][name], boundaries["swapped_combined"][name]
        )

    if run_target_ladder:
        batch_size = carrier.shape[0]
        d_in, d_out = int(batch["W"].shape[1]), int(batch["W"].shape[2])
        T, d_in_pad = int(debug["T"]), int(debug["d_in_pad"])
        _q_base, q_pos_emb, _q_pos_o, _q_pos_t = model._build_decoder_query_state(
            batch_size=batch_size,
            dist_patch_by_patch=debug["dist_patch_by_patch"],
            d_out=d_out,
            T=T,
        )
        def decode_cut(state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            # Shared-head early exit: deliberately skip all remaining decoder
            # layers. Unlike suffix replay, this can show where a state first
            # becomes linearly readable by the actual final norm/heads.
            return model._decode_query_tokens_to_output(
                state,
                z=latent.flatten(1),
                q_pos_emb=q_pos_emb,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
                patch_mask=debug["patch_mask"],
                d_in_mask=batch["d_in_mask"],
                d_out_mask=batch["d_out_mask"],
                disable_z_shortcut=True,
            )[0]

        target_rows = {}
        for name in ("bridge_pre_film", "bridge_post_film", *[f"decoder_{i:02d}" for i in range(len(model.decoder_layers))]):
            matched_cut = decode_cut(boundaries["combined"][name], branches["combined"])
            swapped_cut = decode_cut(boundaries["swapped_combined"][name], branches["swapped_combined"])
            matched_loss = _loss_metrics(batch["W"], matched_cut, **loss_cfg)
            swapped_loss = _loss_metrics(batch["W"], swapped_cut, **loss_cfg)
            target_rows[name] = {
                "matched": matched_loss,
                "swapped": swapped_loss,
                "swap_minus_matched_direction": swapped_loss["dir"] - matched_loss["dir"],
                "output_swap": _relative_delta(matched_cut, swapped_cut),
            }
        result["boundary_target_ladder"] = {
            "status": "complete_first_pair",
            "probe": "shared_final_qnorm_and_heads_early_exit_without_remaining_decoder",
            "interpretation_scope": (
                "representation readability under the shared final norm/heads; "
                "not causal loss localization of the production suffix"
            ),
            "cuts": target_rows,
        }

    # Mandatory bridge is nonlinear because z controls both K and V. Quantify
    # branch interference and the local adaptive-direction Jacobian gain.
    b_combined = boundaries["combined"]["bridge_pre_film"]
    b_carrier = boundaries["carrier_only"]["bridge_pre_film"]
    b_adaptive = boundaries["adaptive_only"]["bridge_pre_film"]
    b_zero = boundaries["zero"]["bridge_pre_film"]
    residual = b_combined - (b_carrier + b_adaptive - b_zero)
    result["bridge_superposition"] = {
        "error_rms": _rms(residual),
        "relative_to_combined": float(_rms(residual) / max(_rms(b_combined), 1e-24)),
    }
    carrier_latent = branches["carrier_only"].detach()
    direction = adaptive.detach()
    batch_size = carrier_latent.shape[0]
    d_out = int(batch["W"].shape[2])
    T = int(debug["T"])
    q_base, _q_pos_emb, _q_pos_o, _q_pos_t = model._build_decoder_query_state(
        batch_size=batch_size,
        dist_patch_by_patch=debug["dist_patch_by_patch"],
        d_out=d_out,
        T=T,
    )
    query_mask = (
        debug["patch_mask"].unsqueeze(1).expand(-1, d_out, -1)
        & batch["d_out_mask"].unsqueeze(-1)
    ).reshape(batch_size, d_out * T)
    q_base = q_base * query_mask.to(q_base.dtype).unsqueeze(-1)

    def bridge_only(value: torch.Tensor) -> torch.Tensor:
        return model.mandatory_latent_bridge(q_base, value, query_mask=query_mask)

    # Flash-SDPA does not implement the double backward used internally by
    # torch.autograd.functional.jvp. Scope only this zero-update derivative to
    # mathematical SDPA; production and branch-parity forwards remain flash.
    with sdpa_kernel([SDPBackend.MATH]):
        _, jvp = torch.autograd.functional.jvp(
            bridge_only,
            carrier_latent.requires_grad_(True),
            direction,
            create_graph=False,
        )
    # A finite displacement is a backend-independent sanity check, explicitly
    # not used as the exact Jacobian measurement.
    secant_half_width = 0.1
    plus = bridge_only(carrier_latent + secant_half_width * direction)
    minus = bridge_only(carrier_latent - secant_half_width * direction)
    secant = (plus.float() - minus.float()) / (2.0 * secant_half_width)
    result["bridge_adaptive_jvp"] = {
        "method": "exact_autograd_jvp_scoped_math_sdpa",
        "input_direction_rms": _rms(direction),
        "output_jvp_rms": _rms(jvp),
        "rms_gain": float(_rms(jvp) / max(_rms(direction), 1e-24)),
        "finite_secant_sanity": {
            "method": "centered_finite_displacement_not_exact_jacobian",
            "scalar_half_width": secant_half_width,
            "output_rms": _rms(secant),
            "relative_to_exact_jvp": _relative_delta(secant, jvp),
        },
    }
    return result


def _gradient_groups(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    prefixes = {
        "bridge": ("mandatory_latent_bridge.",),
        "decoder": ("decoder_layers.0.", f"decoder_layers.{len(model.decoder_layers) - 1}."),
        "head": ("q_tokens_norm.", "direction_head.", "scale_head."),
    }
    v9_suffixes = {
        "v9_query": ".query_proj.weight",
        "v9_weight_key": ".weight_key_proj.weight",
        "v9_norm_key": ".weight_norm_key_proj.weight",
        "v9_context_key": ".context_key_proj.weight",
        "v9_out": ".out_proj.weight",
        "v9_ffn_in": ".local_ffn.0.weight",
        "v9_ffn_out": ".local_ffn.2.weight",
    }
    result = {name: [] for name in (*v9_suffixes, *prefixes)}
    for parameter_name, parameter in model.named_parameters():
        if parameter_name.startswith("hybrid_content_readout_v9.blocks."):
            matches = [
                group for group, suffix in v9_suffixes.items() if parameter_name.endswith(suffix)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"unclassified/ambiguous V9-A gradient parameter: {parameter_name} matches={matches}"
                )
            result[matches[0]].append((parameter_name, parameter))
            continue
        for group, group_prefixes in prefixes.items():
            if parameter_name.startswith(group_prefixes):
                result[group].append((parameter_name, parameter))
                break
    if any(not rows for rows in result.values()):
        raise RuntimeError(f"empty gradient group: {[name for name, rows in result.items() if not rows]}")
    return result


def _gradient_signature(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    groups: Mapping[str, Sequence[tuple[str, torch.nn.Parameter]]],
    loss_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = [parameter for rows in groups.values() for _name, parameter in rows]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)
    with _amp(next(model.parameters()).device):
        prediction, *_ = model(
            batch["W"], batch["x"], x_mask=batch["x_mask"],
            d_in_mask=batch["d_in_mask"], d_out_mask=batch["d_out_mask"],
        )
        loss = _structural_loss(batch["W"], prediction, loss_cfg)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    result: dict[str, Any] = {"loss": float(loss.detach()), "groups": {}}
    offset = 0
    for group, rows in groups.items():
        chunks = []
        tensor_rms = []
        unused_names = []
        source_nonzero = False
        for parameter_name, _parameter in rows:
            gradient = gradients[offset]
            offset += 1
            if gradient is None:
                unused_names.append(parameter_name)
                continue
            source_nonzero = source_nonzero or bool(torch.count_nonzero(gradient).item())
            # BF16 preserves the FP32 exponent range. FP16 would silently
            # erase the observed 1e-9 router gradients in this diagnostic.
            chunks.append(gradient.detach().cpu().to(torch.bfloat16).flatten())
            tensor_rms.append(_rms(gradient))
        if len(unused_names) == len(rows):
            raise RuntimeError(f"every selected {group} gradient is unused")
        flat = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.bfloat16)
        if not source_nonzero:
            raise RuntimeError(f"every selected {group} gradient is exactly zero")
        if source_nonzero and not bool(torch.count_nonzero(flat).item()):
            raise RuntimeError(f"BF16 signature erased every nonzero {group} gradient")
        result["groups"][group] = {
            "flat": flat,
            "tensor_rms": tensor_rms,
            "selected_tensor_count": len(rows),
            "used_tensor_count": len(rows) - len(unused_names),
            "unused_tensor_count": len(unused_names),
            "unused_parameter_names": unused_names,
        }
    for parameter in parameters:
        parameter.requires_grad_(False)
    return result


def _compare_gradient_signatures(left: Mapping[str, Any], right: Mapping[str, Any], adam_eps: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in left["groups"]:
        a = left["groups"][group]["flat"].float()
        b = right["groups"][group]["flat"].float()
        if a.shape != b.shape or a.numel() == 0:
            raise RuntimeError(f"invalid paired gradient signatures for {group}: {a.shape} vs {b.shape}")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError(f"nonfinite paired gradient signature for {group}")
        dot = torch.dot(a, b)
        norm_a, norm_b = a.norm(), b.norm()
        if float(norm_a) == 0.0 or float(norm_b) == 0.0:
            raise RuntimeError(f"zero paired gradient signature for {group}")
        cosine = dot / (norm_a * norm_b).clamp_min(1e-24)
        resultant = (a + b).norm() / (norm_a + norm_b).clamp_min(1e-24)
        mixed = 0.5 * (a + b)
        mixed_rms = mixed.square().mean().sqrt()
        rms_rows = left["groups"][group]["tensor_rms"] + right["groups"][group]["tensor_rms"]
        result[group] = {
            "cosine": float(cosine),
            "resultant_ratio": float(resultant),
            "left_norm": float(norm_a),
            "right_norm": float(norm_b),
            "mixed_gradient_rms": float(mixed_rms),
            "mixed_gradient_rms_over_adam_eps": float(mixed_rms / adam_eps),
            "left_zero_fraction_after_bf16_signature": float((a == 0).float().mean()),
            "right_zero_fraction_after_bf16_signature": float((b == 0).float().mean()),
            "tensor_grad_rms_min": min(rms_rows) if rms_rows else 0.0,
            "tensor_grad_rms_median": float(torch.tensor(rms_rows).median()) if rms_rows else 0.0,
            "fraction_tensor_rms_above_adam_eps": (
                sum(value > adam_eps for value in rms_rows) / len(rms_rows) if rms_rows else 0.0
            ),
            "left_selected_tensor_count": left["groups"][group]["selected_tensor_count"],
            "left_unused_tensor_count": left["groups"][group]["unused_tensor_count"],
            "left_unused_parameter_names": left["groups"][group]["unused_parameter_names"],
            "right_selected_tensor_count": right["groups"][group]["selected_tensor_count"],
            "right_unused_tensor_count": right["groups"][group]["unused_tensor_count"],
            "right_unused_parameter_names": right["groups"][group]["unused_parameter_names"],
        }
    return result


def _operator_only(batch: Mapping[str, torch.Tensor], parity: int) -> dict[str, torch.Tensor]:
    index = torch.arange(parity, batch["W"].shape[0], 2, device=batch["W"].device)
    return {name: value.index_select(0, index) for name, value in batch.items()}


def _safe_section(report: dict[str, Any], name: str, fn: Callable[[], Any], *, mandatory: bool) -> None:
    started = time.time()
    try:
        report["sections"][name] = {"status": "complete", "elapsed_seconds": time.time() - started, "result": fn()}
    except Exception as error:  # diagnostic must preserve earlier evidence
        report["sections"][name] = {
            "status": "failed_mandatory" if mandatory else "failed_optional",
            "elapsed_seconds": time.time() - started,
            "exception": repr(error),
            "traceback": traceback.format_exc(),
        }
        if mandatory:
            raise


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    resolved_path = run_root / "resolved_run_config.json"
    selection_path = run_root / "operator_set_selection.json"
    if not 1 <= args.gradient_pair_count <= args.pair_count <= 8:
        raise ValueError("require 1 <= gradient-pair-count <= pair-count <= 8")
    for path in (resolved_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    preflight = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint_path),
        "run_root": str(run_root),
        "resolved_config": str(resolved_path),
        "selection": str(selection_path),
        "device": args.device,
        "pair_count": args.pair_count,
        "gradient_pair_count": args.gradient_pair_count,
        "optimizer_steps": 0,
    }
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("stage=dry_run_complete no_files_written=true", flush=True)
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = args.output_parent.expanduser().resolve() / f"v9a_final_causal_bundle_{stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "preflight": preflight,
        "artifacts": {"report": str(output_root / "report.json")},
        "sections": {},
    }
    _atomic_json(output_root / "report.json", report)

    def persist_unhandled(
        exception_type: type[BaseException],
        exception: BaseException,
        exception_traceback: Any,
    ) -> None:
        report["status"] = "failed_invalid"
        report["unhandled_exception"] = {
            "exception": repr(exception),
            "traceback": "".join(
                traceback.format_exception(exception_type, exception, exception_traceback)
            ),
        }
        _atomic_json(output_root / "report.json", report)
        sys.__excepthook__(exception_type, exception, exception_traceback)

    sys.excepthook = persist_unhandled
    print(f"stage=load checkpoint={checkpoint_path} output={output_root}", flush=True)
    cfg_dict = json.loads(resolved_path.read_text(encoding="utf-8"))
    expected_checkpoint = (
        Path(str(cfg_dict["train"]["checkpoint_dir"])).expanduser().resolve()
        / "stage_1"
        / "step_0001984.pt"
    )
    if checkpoint_path != expected_checkpoint:
        raise RuntimeError(
            f"checkpoint/run-root binding failed: got={checkpoint_path} expected={expected_checkpoint}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    required_checkpoint_keys = {"step", "stage", "model_state", "config"}
    if not required_checkpoint_keys.issubset(checkpoint):
        raise RuntimeError(
            f"checkpoint payload lacks {sorted(required_checkpoint_keys - set(checkpoint))}"
        )
    checkpoint_step = int(checkpoint["step"])
    checkpoint_stage = int(checkpoint["stage"])
    if checkpoint_step != 1984 or checkpoint_stage != 1:
        raise RuntimeError(
            "final V9-A bundle requires stage=1 step=1984, "
            f"got stage={checkpoint_stage} step={checkpoint_step}"
        )
    cfg = OmegaConf.create(checkpoint.get("config", cfg_dict))
    embedded_cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(embedded_cfg, dict)
    critical_paths = (
        ("data", "seed"),
        ("model", "patch_size"),
        ("model", "big_vae"),
        ("train", "operator_bank", "pair_manifest"),
        ("train", "operator_bank", "pair_manifest_sha256"),
        ("train", "operator_bank", "operator_set_overfit", "selected_operators"),
        ("train", "operator_bank", "operator_set_overfit", "selection_sha256"),
        ("train", "operator_bank", "operator_set_overfit", "schedule_sha256"),
        ("train", "struct_loss"),
        ("train", "max_steps"),
        ("train", "lr"),
        ("train", "eps"),
    )

    def nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
        value: Any = payload
        for key in path:
            value = value[key]
        return value

    mismatches = [
        ".".join(path)
        for path in critical_paths
        if nested(embedded_cfg, path) != nested(cfg_dict, path)
    ]
    if mismatches:
        raise RuntimeError(f"checkpoint embedded config disagrees with run config: {mismatches}")
    device = torch.device(args.device)
    model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    refiner = model.hybrid_content_readout_v9
    if refiner is None or not refiner.carrier_mean_content or refiner.carrier_content_aggregation != "sqrt_depth_sum":
        raise RuntimeError("checkpoint is not frozen V9-A sqrt-depth")
    if str(cfg.model.big_vae.architecture_version) != "carrier_mean_content_posfilm_v9a":
        raise RuntimeError("checkpoint architecture_version is not V9-A")
    operator = cfg_dict["train"]["operator_bank"]
    overfit = operator["operator_set_overfit"]
    if (
        selection.get("selection_sha256") != overfit["selection_sha256"]
        or selection.get("schedule_sha256") != overfit["schedule_sha256"]
    ):
        raise RuntimeError("resolved config and frozen selection manifest disagree")
    source = CanonicalOperatorSetTrainingDataset(
        operator["pair_manifest"],
        selected_operators=overfit["selected_operators"],
        expected_operator_count=64,
        seed=int(cfg_dict["data"]["seed"]),
        hot_shards=int(operator["hot_shards"]),
        expected_pair_manifest_sha256=operator["pair_manifest_sha256"],
        expected_selection_sha256=overfit["selection_sha256"],
        expected_schedule_sha256=overfit["schedule_sha256"],
    )
    heldout_pairs = list(source._rounds[62])[: args.pair_count]
    loss_cfg = _loss_cfg(cfg)
    report["preflight"].update(
        {
            "checkpoint_stat": {
                "size": checkpoint_path.stat().st_size,
                "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            },
            "selection_sha256": selection["selection_sha256"],
            "schedule_sha256": selection["schedule_sha256"],
            "checkpoint_step": checkpoint_step,
            "checkpoint_stage": checkpoint_stage,
            "checkpoint_run_root_binding": {
                "status": "pass",
                "expected_path": str(expected_checkpoint),
            },
            "embedded_vs_resolved_config_binding": {
                "status": "pass",
                "compared_paths": [".".join(path) for path in critical_paths],
            },
            "dtype": "bfloat16-autocast",
            "heldout_pair_ids": [[left[0], right[0]] for left, right in heldout_pairs],
        }
    )
    _atomic_json(output_root / "report.json", report)

    # A-D share one forward per pair. Only compact summaries leave the GPU.
    shared_rows: list[dict[str, Any]] = []
    routing_csv_rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(heldout_pairs):
        print(f"stage=pair_forward pair={pair_index + 1}/{len(heldout_pairs)}", flush=True)
        cpu_batch, assignments, swap_cpu = _pair_batch(source, pair)
        batch = _move(cpu_batch, device)
        swap = swap_cpu.to(device)
        with _amp(device):
            prediction, debug, writes = _capture_forward(model, batch)
        carrier, adaptive = debug["carrier"], debug["adaptive"]
        assert carrier is not None and adaptive is not None
        row: dict[str, Any] = {
            "pair_index": pair_index,
            "operator_ids": [pair[0][0], pair[1][0]],
            "matched": _loss_metrics(batch["W"], prediction, **loss_cfg),
            "encoder": {
                "carrier": _tensor_summary(carrier),
                "adaptive": _tensor_summary(adaptive),
                "z": _tensor_summary(debug["latent_decoder_z"].view_as(carrier)),
                "writes": [_tensor_summary(value) for value in writes],
            },
        }
        w_patches = batch["W"].transpose(1, 2).reshape(
            batch["W"].shape[0], batch["W"].shape[2], -1, int(cfg.model.patch_size)
        )
        with _amp(device):
            a0 = _exact_v8_routing(
                model.clean_content_readout_v8,
                debug["dist_patch_by_patch"],
                debug["patch_mask"],
                batch["d_out_mask"],
            )
        row["carrier_synthesis"] = _carrier_synthesis(
            carrier,
            a0,
            w_patches,
            model.clean_content_readout_v8.out_proj.weight,
            run_pinv=pair_index == 0,
            loss_cfg=loss_cfg,
        )
        content_sum = torch.zeros_like(carrier, dtype=torch.float32)
        routing_state = carrier
        route_rows = []
        for block_index, (block, write) in enumerate(zip(refiner.blocks, writes, strict=True)):
            with _amp(device):
                routing, protected = _exact_v9_routing(
                    block,
                    routing_state,
                    w_patches,
                    debug["dist_patch_by_patch"],
                    debug["patch_mask"],
                    batch["d_out_mask"],
                )
            summary = _routing_summary(routing)
            summary.update(_support_alignment(routing, a0))
            summary.update({f"protected_{key}": value for key, value in _support_alignment(routing, protected).items()})
            summary["pair_index"] = pair_index
            summary["block_index"] = block_index
            summary["anchor_family"] = block.anchor_family
            routing_csv_rows.append(summary)
            route_rows.append(summary)
            content_sum = content_sum + write.float()
            routing_state = (carrier.float() + refiner.residual_scale * content_sum / float(block_index + 1)).to(carrier.dtype)
        row["routing"] = route_rows
        with _amp(device):
            row["branch_boundary"] = _branch_and_boundary_panel(
                model,
                batch,
                debug,
                swap,
                loss_cfg,
                prediction,
                run_target_ladder=pair_index == 0,
            )
        shared_rows.append(row)
        _atomic_json(output_root / "report.json", report | {"partial_shared_pairs": shared_rows})
        del batch, prediction, debug, writes, carrier, adaptive, a0
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["sections"]["A_encoder_capture"] = {
        "status": "complete",
        "result": [{"pair_index": row["pair_index"], "operator_ids": row["operator_ids"], "encoder": row["encoder"]} for row in shared_rows],
    }
    report["sections"]["B_branch_interventions"] = {
        "status": "complete",
        "result": [{"pair_index": row["pair_index"], "branch_boundary": row["branch_boundary"]} for row in shared_rows],
    }
    report["sections"]["C_bridge_boundary"] = {
        "status": "complete",
        "result": [
            {
                "pair_index": row["pair_index"],
                "bridge": row["branch_boundary"]["bridge_superposition"],
                "jvp": row["branch_boundary"]["bridge_adaptive_jvp"],
                "swap_ladder": row["branch_boundary"]["boundary_swap_ladder"],
                "target_ladder": row["branch_boundary"]["boundary_target_ladder"],
            }
            for row in shared_rows
        ],
    }
    report["sections"]["D_routing_carrier_synthesis"] = {
        "status": "complete",
        "result": [{"pair_index": row["pair_index"], "carrier_synthesis": row["carrier_synthesis"]} for row in shared_rows],
    }
    routing_csv = output_root / "routing_support.csv"
    with routing_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(routing_csv_rows[0]))
        writer.writeheader()
        writer.writerows(routing_csv_rows)
    report["artifacts"]["routing_csv"] = str(routing_csv)
    _atomic_json(output_root / "report.json", report)

    print(f"stage=gradient_panel pairs={args.gradient_pair_count}", flush=True)
    groups = _gradient_groups(model)
    gradient_rows = []
    adam_eps = float(cfg.train.eps)
    for pair_index, pair in enumerate(heldout_pairs[: args.gradient_pair_count]):
        cpu_batch, _assignments, _swap = _pair_batch(source, pair)
        batch = _move(cpu_batch, device)
        left = _gradient_signature(model, _operator_only(batch, 0), groups, loss_cfg)
        right = _gradient_signature(model, _operator_only(batch, 1), groups, loss_cfg)
        gradient_rows.append(
            {
                "pair_index": pair_index,
                "operator_ids": [pair[0][0], pair[1][0]],
                "left_loss": left["loss"],
                "right_loss": right["loss"],
                "groups": _compare_gradient_signatures(left, right, adam_eps),
            }
        )
        report["partial_gradient_pairs"] = gradient_rows
        _atomic_json(output_root / "report.json", report)
        del batch, left, right
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["sections"]["E_gradient_conflict_eps"] = {"status": "complete", "result": gradient_rows}

    # Optional: a cross-fit slot-basis test. Failure does not invalidate A-E.
    def optional_procrustes() -> dict[str, Any]:
        writes = shared_rows[0]["encoder"]["writes"]
        return {
            "status": "not_run_from_summaries",
            "reason": "Full write tensors are deliberately not retained across A-E; routing best-match support is the bounded primary basis-drift test.",
            "captured_write_count": len(writes),
        }

    _safe_section(report, "optional_crossfit_procrustes", optional_procrustes, mandatory=False)
    report["status"] = "complete_valid"
    report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(output_root / "report.json", report)
    print(f"stage=complete report={output_root / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
