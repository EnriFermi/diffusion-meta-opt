from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import random
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer, operator_bank_data_pipeline
from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu


_OBJECTIVES = ("structural_direction", "structural_scale_x10", "behavioral")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure AE gradient-estimator stability versus effective batch size.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="4,8,16,32,64,128,256")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--reference-batch-size", type=int, default=1024)
    parser.add_argument("--panel-size", type=int, default=65536)
    parser.add_argument("--panel-seeds", default="31001,31013,31019,31033")
    parser.add_argument("--start-index", type=int, default=7_000_000)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parameter_group(name: str) -> str:
    if name.startswith("distribution_encoder."):
        return "distribution_encoder"
    if name.startswith("encoder_layers.") and ".local_block." in name:
        return "encoder_local"
    if name.startswith("encoder_layers.") and ".perceiver_block." in name:
        return "encoder_perceiver"
    if name.startswith("latent_to_weight_feedback."):
        return "encoder_feedback"
    if name.startswith(
        (
            "patch_tokenizer.",
            "patch_token_proj.",
            "encoder_conditioning_adapters.",
            "enc_dist_inject_projs.",
            "enc_dist_to_latent_heads.",
        )
    ):
        return "encoder_adapters"
    if name == "latent_base" or name.startswith(("latent_norm.", "to_mu.", "to_logvar.")):
        return "encoder_boundary"
    if name.startswith(
        (
            "decoder_layers.",
            "mandatory_latent_bridge.",
            "latent_to_decoder.",
            "q_tokens",
            "q_tokens_norm.",
            "query_proj.",
            "direction_head.",
            "scale_head.",
            "decoder_",
        )
    ):
        return "decoder"
    return "other"


def _build_estimate_plan(
    batch_sizes: Iterable[int], *, replicates: int, reference_batch_size: int
) -> list[dict[str, Any]]:
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    sizes = sorted({int(value) for value in batch_sizes})
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError(f"invalid batch sizes: {sizes}")
    if any(value > 32 and value % 32 for value in sizes):
        raise ValueError("effective batch sizes above 32 must be multiples of the production microbatch 32")
    if reference_batch_size < max(sizes) or reference_batch_size % 32:
        raise ValueError("reference batch must be a multiple of 32 and at least the largest candidate")

    plan = [
        {
            "estimate_id": "reference",
            "kind": "reference",
            "effective_batch_size": int(reference_batch_size),
            "replicate": -1,
            "microbatch_sizes": [32] * (reference_batch_size // 32),
        }
    ]
    for batch_size in sizes:
        microbatches = [batch_size] if batch_size <= 32 else [32] * (batch_size // 32)
        for replicate in range(replicates):
            plan.append(
                {
                    "estimate_id": f"b{batch_size:04d}_r{replicate:02d}",
                    "kind": "candidate",
                    "effective_batch_size": batch_size,
                    "replicate": replicate,
                    "microbatch_sizes": list(microbatches),
                }
            )
    return plan


def _add_microbatch_partition_audits(plan: list[dict[str, Any]], *, replicates: int) -> None:
    for replicate in range(replicates):
        plan.append(
            {
                "estimate_id": f"b0032_partition4_r{replicate:02d}",
                "kind": "partition_audit",
                "effective_batch_size": 32,
                "replicate": replicate,
                "microbatch_sizes": [4] * 8,
            }
        )


def _nested_batch_assignment(
    plan: list[dict[str, Any]],
    *,
    reference_batches: list[Any],
    replicate_blocks: dict[int, list[Any]],
) -> dict[str, list[Any]]:
    """Bind all candidate B values to prefixes of one disjoint block/replicate."""
    output: dict[str, list[Any]] = {}
    for row in plan:
        estimate_id = str(row["estimate_id"])
        if row["kind"] == "reference":
            output[estimate_id] = list(reference_batches)
            continue
        block = replicate_blocks[int(row["replicate"])]
        batch_size = int(row["effective_batch_size"])
        if row["kind"] == "partition_audit":
            output[estimate_id] = [_slice_batch(block[0], 4, start=start) for start in range(0, 32, 4)]
        elif batch_size < 32:
            output[estimate_id] = [_slice_batch(block[0], batch_size)]
        else:
            output[estimate_id] = list(block[: batch_size // 32])
    return output


def _make_panel_plans(
    grouped_parameters: dict[str, list[tuple[str, torch.nn.Parameter]]],
    *,
    panel_size: int,
    seeds: list[int],
) -> tuple[dict[tuple[str, int], list[tuple[str, np.ndarray, np.ndarray]]], dict[str, int]]:
    plans: dict[tuple[str, int], list[tuple[str, np.ndarray, np.ndarray]]] = {}
    group_numel: dict[str, int] = {}
    for group, parameters in grouped_parameters.items():
        boundaries = np.cumsum([int(parameter.numel()) for _, parameter in parameters], dtype=np.int64)
        total = int(boundaries[-1]) if boundaries.size else 0
        group_numel[group] = total
        if total <= 0:
            continue
        size = min(int(panel_size), total)
        starts = np.concatenate((np.array([0], dtype=np.int64), boundaries[:-1]))
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            # Sampling with replacement is intentional: it avoids allocating a
            # permutation over hundreds of millions of coordinates and remains
            # an unbiased coordinate-panel estimator.  Four independent panels
            # expose its sampling uncertainty.
            global_indices = rng.integers(0, total, size=size, dtype=np.int64)
            parameter_indices = np.searchsorted(boundaries, global_indices, side="right")
            entries: list[tuple[str, np.ndarray, np.ndarray]] = []
            for parameter_index in np.unique(parameter_indices):
                positions = np.flatnonzero(parameter_indices == parameter_index).astype(np.int64)
                local = (global_indices[positions] - starts[parameter_index]).astype(np.int64)
                entries.append((parameters[int(parameter_index)][0], local, positions))
            plans[(group, int(seed))] = entries
    return plans, group_numel


def _extract_gradient_summary(
    named_parameters: dict[str, torch.nn.Parameter],
    grouped_parameters: dict[str, list[tuple[str, torch.nn.Parameter]]],
    panel_plans: dict[tuple[str, int], list[tuple[str, np.ndarray, np.ndarray]]],
    group_numel: dict[str, int],
    *,
    panel_size: int,
    seeds: list[int],
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, int], np.ndarray]]:
    exact: dict[str, dict[str, float]] = {}
    panels: dict[tuple[str, int], np.ndarray] = {}
    for group, parameters in grouped_parameters.items():
        squared_tensor: torch.Tensor | None = None
        for _, parameter in parameters:
            if parameter.grad is not None:
                contribution = parameter.grad.detach().float().square().sum()
                squared_tensor = contribution if squared_tensor is None else squared_tensor + contribution
        squared = float(squared_tensor.item()) if squared_tensor is not None else 0.0
        numel = int(group_numel[group])
        exact[group] = {
            "exact_l2": math.sqrt(max(0.0, squared)),
            "exact_rms": math.sqrt(max(0.0, squared) / max(1, numel)),
            "numel": float(numel),
        }
        for seed in seeds:
            size = min(int(panel_size), numel)
            device = next(
                (named_parameters[name].grad.device for name, _, _ in panel_plans.get((group, int(seed)), []) if named_parameters[name].grad is not None),
                torch.device("cpu"),
            )
            vector_gpu = torch.zeros(size, device=device, dtype=torch.float32)
            for name, local_indices, positions in panel_plans.get((group, int(seed)), []):
                grad = named_parameters[name].grad
                if grad is None:
                    continue
                index = torch.as_tensor(local_indices, device=grad.device, dtype=torch.long)
                destination = torch.as_tensor(positions, device=grad.device, dtype=torch.long)
                values = grad.detach().reshape(-1).index_select(0, index).float()
                vector_gpu.index_copy_(0, destination, values)
            panels[(group, int(seed))] = vector_gpu.cpu().numpy()
    return exact, panels


def _objective_weights(cfg: Any) -> dict[str, float]:
    behavioral = cfg.train.behavioral_loss
    structural = cfg.train.struct_loss
    weights = {
        "behavioral_outer": float(cfg.train.behavioral_coef),
        "behavioral_operator": float(behavioral.lambda_operator),
        "behavioral_direction": float(behavioral.lambda_dir),
        "behavioral_scale": float(behavioral.lambda_scale),
        "structural_outer": float(cfg.train.structural_coef),
        "structural_direction": float(structural.lambda_dir),
        "structural_scale": float(structural.lambda_scale),
        "structural_rec": float(structural.lambda_rec),
        "structural_rel": float(structural.lambda_rel),
    }
    if weights["structural_rec"] != 0.0 or weights["structural_rel"] != 0.0:
        raise ValueError("probe decomposition requires structural lambda_rec=lambda_rel=0")
    return weights


def _assert_deterministic_probe_config(cfg: Any) -> None:
    big_vae = cfg.model.big_vae
    values = {
        "model.big_vae.dropout": big_vae.dropout,
        "model.big_vae.distribution_encoder.dropout": big_vae.distribution_encoder.dropout,
        "model.big_vae.patch_tokenizer.residual_dropout": big_vae.patch_tokenizer.residual_dropout,
    }
    for path, value in values.items():
        if value is not None and float(value) != 0.0:
            raise ValueError(f"gradient stability probe requires zero dropout, got {path}={value}")
    if bool(big_vae.use_latent_sampling):
        raise ValueError("gradient stability probe requires deterministic latent sampling=false")


def _objective_loss(
    objective: str,
    *,
    model: torch.nn.Module,
    cfg: Any,
    W: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    # Use the exact production forward.  forward_debug retains a large debug
    # payload and OOMs this 725M model even though the production B32 fits.
    W_hat, _mu, _logvar, pred_dirs = model(
        W, x, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
    )
    if objective == "behavioral":
        behavioral_cfg = cfg.train.behavioral_loss
        weights = _objective_weights(cfg)
        operator = WeightQuantileVAE.operator_recon_loss(
            x, W, W_hat, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
        )
        direction, scale = WeightQuantileVAE.operator_direction_scale_loss(
            x,
            W,
            W_hat,
            x_mask=x_mask,
            d_out_mask=d_out_mask,
            gamma=float(behavioral_cfg.gamma),
            huber_delta=float(behavioral_cfg.huber_delta),
        )
        return (
            weights["behavioral_operator"] * operator
            + weights["behavioral_direction"] * direction
            + weights["behavioral_scale"] * scale
        )

    struct_cfg = cfg.train.struct_loss
    weights = _objective_weights(cfg)
    direction_weight = weights["structural_direction"] if objective == "structural_direction" else 0.0
    scale_weight = weights["structural_scale"] if objective == "structural_scale_x10" else 0.0
    loss, _ = WeightQuantileVAE.patch_structure_loss(
        W,
        W_hat,
        patch_size=int(cfg.model.patch_size),
        gamma=float(struct_cfg.gamma),
        lambda_dir=direction_weight,
        lambda_scale=scale_weight,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=float(struct_cfg.huber_delta),
        pred_dirs=pred_dirs,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
    )
    return loss


def _median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(statistics.median(items)) if items else float("nan")


def _mad(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return float("nan")
    center = statistics.median(items)
    return float(statistics.median(abs(value - center) for value in items))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def _slice_batch(batch: Any, size: int, *, start: int = 0) -> Any:
    end = int(start) + int(size)
    if size <= 0 or start < 0 or end > int(batch.W.shape[0]):
        raise ValueError(
            f"invalid batch slice [{start},{end}) for physical batch {int(batch.W.shape[0])}"
        )
    return replace(
        batch,
        W=batch.W[start:end],
        x=batch.x[start:end],
        x_mask=batch.x_mask[start:end],
        d_in_mask=batch.d_in_mask[start:end],
        d_out_mask=batch.d_out_mask[start:end],
        logical_indices=tuple(batch.logical_indices[start:end]),
    )


def _analyze_panels(
    records: dict[tuple[str, str], dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    groups: list[str],
    group_numel: dict[str, int],
    seeds: list[int],
    exact_rows: list[dict[str, Any]] | None = None,
    objective_weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    plan_by_id = {str(row["estimate_id"]): row for row in plan}
    candidate_ids = [str(row["estimate_id"]) for row in plan if row["kind"] == "candidate"]
    weights = objective_weights or {"structural_outer": 1.0, "behavioral_outer": 1.0}
    derived = {
        "structural": (("structural_direction", 1.0), ("structural_scale_x10", 1.0)),
        "training_total": (
            ("structural_direction", float(weights["structural_outer"])),
            ("structural_scale_x10", float(weights["structural_outer"])),
            ("behavioral", float(weights["behavioral_outer"])),
        ),
    }

    aggregate_members = {
        "encoder_total": [group for group in groups if group.startswith("encoder_")],
        "global": list(groups),
    }

    def leaf_panel(objective: str, estimate_id: str, group: str, seed: int) -> np.ndarray:
        if objective in derived:
            return sum(
                (
                    records[(component, estimate_id)]["panels"][(group, seed)] * coefficient
                    for component, coefficient in derived[objective]
                ),
                start=np.zeros_like(
                    records[(derived[objective][0][0], estimate_id)]["panels"][(group, seed)]
                ),
            )
        return records[(objective, estimate_id)]["panels"][(group, seed)]

    def panel(objective: str, estimate_id: str, group: str, seed: int) -> np.ndarray:
        members = aggregate_members.get(group)
        if members is None:
            return leaf_panel(objective, estimate_id, group, seed)
        scaled = []
        for member in members:
            vector = leaf_panel(objective, estimate_id, member, seed)
            scale = math.sqrt(float(group_numel[member]) / max(1, int(vector.size)))
            scaled.append(vector * scale)
        return np.concatenate(scaled) if scaled else np.zeros(1, dtype=np.float32)

    reported_groups = [*groups, "encoder_total", "global"]
    for objective in (*_OBJECTIVES, "structural", "training_total"):
        for group in reported_groups:
            for estimate_id in candidate_ids:
                meta = plan_by_id[estimate_id]
                panel_cosines: list[float] = []
                panel_relative_errors: list[float] = []
                panel_norm_ratios: list[float] = []
                for seed in seeds:
                    candidate = panel(objective, estimate_id, group, seed)
                    reference = panel(objective, "reference", group, seed)
                    reference_norm = float(np.linalg.norm(reference))
                    cosine = _cosine(candidate, reference)
                    relative_error = (
                        float(np.linalg.norm(candidate - reference) / reference_norm)
                        if reference_norm > 0.0
                        else float("nan")
                    )
                    norm_ratio = (
                        float(np.linalg.norm(candidate) / reference_norm)
                        if reference_norm > 0.0
                        else float("nan")
                    )
                    panel_cosines.append(cosine)
                    panel_relative_errors.append(relative_error)
                    panel_norm_ratios.append(norm_ratio)
                    comparisons.append(
                        {
                            "objective": objective,
                            "group": group,
                            "estimate_id": estimate_id,
                            "effective_batch_size": meta["effective_batch_size"],
                            "replicate": meta["replicate"],
                            "panel_seed": seed,
                            "cosine_to_reference": cosine,
                            "relative_error_to_reference": relative_error,
                            "norm_ratio_to_reference": norm_ratio,
                        }
                    )
                comparisons.append(
                    {
                        "objective": objective,
                        "group": group,
                        "estimate_id": estimate_id,
                        "effective_batch_size": meta["effective_batch_size"],
                        "replicate": meta["replicate"],
                        "panel_seed": "median",
                        "cosine_to_reference": _median(panel_cosines),
                        "relative_error_to_reference": _median(panel_relative_errors),
                        "norm_ratio_to_reference": _median(panel_norm_ratios),
                        "panel_cosine_spread": max(panel_cosines) - min(panel_cosines),
                    }
                )

            batch_sizes = sorted({int(plan_by_id[item]["effective_batch_size"]) for item in candidate_ids})
            for batch_size in batch_sizes:
                rows = [
                    row
                    for row in comparisons
                    if row["objective"] == objective
                    and row["group"] == group
                    and row["panel_seed"] == "median"
                    and int(row["effective_batch_size"]) == batch_size
                ]
                cosines = [float(row["cosine_to_reference"]) for row in rows]
                errors = [float(row["relative_error_to_reference"]) for row in rows]
                ratios = [float(row["norm_ratio_to_reference"]) for row in rows]
                summary.append(
                    {
                        "objective": objective,
                        "group": group,
                        "effective_batch_size": batch_size,
                        "replicates": len(rows),
                        "median_cosine_to_reference": _median(cosines),
                        "min_cosine_to_reference": min(cosines),
                        "median_relative_error": _median(errors),
                        "median_norm_ratio": _median(ratios),
                        "norm_ratio_mad": _mad(ratios),
                    }
                )
    if exact_rows:
        exact_lookup: dict[tuple[str, str, str], dict[str, Any]] = {
            (str(row["objective"]), str(row["estimate_id"]), str(row["group"])): row
            for row in exact_rows
        }
        for row in summary:
            objective = str(row["objective"])
            group = str(row["group"])
            if objective not in _OBJECTIVES or group not in groups:
                continue
            reference = exact_lookup[(objective, "reference", group)]
            candidate_ids_for_size = [
                estimate_id
                for estimate_id in candidate_ids
                if int(plan_by_id[estimate_id]["effective_batch_size"]) == int(row["effective_batch_size"])
            ]
            ratios = [
                float(exact_lookup[(objective, estimate_id, group)]["exact_l2"])
                / max(float(reference["exact_l2"]), 1e-30)
                for estimate_id in candidate_ids_for_size
            ]
            row["exact_l2_ratio_median"] = _median(ratios)
            row["exact_l2_ratio_mad"] = _mad(ratios)
            exact_values = [
                float(exact_lookup[(objective, estimate_id, group)]["exact_l2"])
                for estimate_id in candidate_ids_for_size
            ]
            mean_exact = statistics.fmean(exact_values)
            row["exact_l2_median"] = _median(exact_values)
            row["exact_l2_cv"] = (
                statistics.stdev(exact_values) / mean_exact
                if len(exact_values) > 1 and mean_exact > 0.0
                else 0.0
            )
    return comparisons, summary


def _objective_interactions(
    records: dict[tuple[str, str], dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    groups: list[str],
    group_numel: dict[str, int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Measure component conflict separately from finite-batch estimator noise."""
    rows: list[dict[str, Any]] = []
    aggregate_members = {
        "encoder_total": [group for group in groups if group.startswith("encoder_")],
        "global": list(groups),
    }

    def direct(objective: str, estimate_id: str, group: str, seed: int) -> np.ndarray:
        return records[(objective, estimate_id)]["panels"][(group, seed)]

    def vector(objective: str, estimate_id: str, group: str, seed: int) -> np.ndarray:
        if objective == "structural":
            components = ("structural_direction", "structural_scale_x10")
        else:
            components = (objective,)
        members = aggregate_members.get(group, [group])
        chunks: list[np.ndarray] = []
        for member in members:
            value = sum(
                (direct(component, estimate_id, member, seed) for component in components),
                start=np.zeros_like(direct(components[0], estimate_id, member, seed)),
            )
            # Put leaf panels on their full-vector L2 scale before concatenating.
            value = value * math.sqrt(float(group_numel[member]) / max(1, int(value.size)))
            chunks.append(value)
        return np.concatenate(chunks)

    pairs = (
        ("structural_direction", "structural_scale_x10", "structural_dir_vs_scale"),
        ("structural", "behavioral", "structural_vs_behavioral"),
    )
    for meta in plan:
        estimate_id = str(meta["estimate_id"])
        for group in (*groups, "encoder_total", "global"):
            for left_name, right_name, comparison in pairs:
                per_seed: list[dict[str, float]] = []
                for seed in seeds:
                    left = vector(left_name, estimate_id, group, seed).astype(np.float64)
                    right = vector(right_name, estimate_id, group, seed).astype(np.float64)
                    left_norm = float(np.linalg.norm(left))
                    right_norm = float(np.linalg.norm(right))
                    summed_norm = float(np.linalg.norm(left + right))
                    dot = float(np.dot(left, right))
                    values = {
                        "cosine": _cosine(left, right),
                        "cross_term": 2.0 * dot,
                        "cancellation_ratio": summed_norm / max(left_norm + right_norm, 1e-30),
                        "left_l2_panel": left_norm,
                        "right_l2_panel": right_norm,
                        "sum_l2_panel": summed_norm,
                    }
                    per_seed.append(values)
                    rows.append(
                        {
                            "estimate_id": estimate_id,
                            "kind": meta["kind"],
                            "effective_batch_size": meta["effective_batch_size"],
                            "replicate": meta["replicate"],
                            "group": group,
                            "comparison": comparison,
                            "panel_seed": seed,
                            **values,
                        }
                    )
                rows.append(
                    {
                        "estimate_id": estimate_id,
                        "kind": meta["kind"],
                        "effective_batch_size": meta["effective_batch_size"],
                        "replicate": meta["replicate"],
                        "group": group,
                        "comparison": comparison,
                        "panel_seed": "median",
                        **{
                            key: _median(item[key] for item in per_seed)
                            for key in per_seed[0]
                        },
                    }
                )
    return rows


def _fit_noise_models(
    estimate_rows: list[dict[str, Any]], plan: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fit E||g_B||^2 = ||mu||^2 + tr(Sigma)/B from disjoint replicates."""
    candidate_ids = {
        str(row["estimate_id"]): row for row in plan if row["kind"] == "candidate"
    }
    buckets: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in estimate_rows:
        estimate_id = str(row["estimate_id"])
        if estimate_id not in candidate_ids:
            continue
        batch_size = int(candidate_ids[estimate_id]["effective_batch_size"])
        buckets[(str(row["objective"]), str(row["group"]), batch_size)].append(
            float(row["exact_l2"]) ** 2
        )

    output: list[dict[str, Any]] = []
    objective_groups = sorted({(objective, group) for objective, group, _ in buckets})
    for objective, group in objective_groups:
        batch_sizes = sorted(batch_size for obj, grp, batch_size in buckets if (obj, grp) == (objective, group))
        if objective == "behavioral":
            # Behavioral direction/scale are ratio reductions, so native B<32
            # is not a mean of per-example gradients.  Only production-B32
            # gradient units and their arithmetic accumulations obey this fit.
            batch_sizes = [batch_size for batch_size in batch_sizes if batch_size >= 32]
            x = np.asarray([32.0 / batch_size for batch_size in batch_sizes], dtype=np.float64)
            noise_unit = "production_B32_gradient_unit"
            units_at_b32 = 1.0
        else:
            x = np.asarray([1.0 / batch_size for batch_size in batch_sizes], dtype=np.float64)
            noise_unit = "single_example_gradient"
            units_at_b32 = 32.0
        y = np.asarray(
            [statistics.fmean(buckets[(objective, group, batch_size)]) for batch_size in batch_sizes],
            dtype=np.float64,
        )
        design = np.column_stack((np.ones_like(x), x))
        signal_sq, covariance_trace = np.linalg.lstsq(design, y, rcond=None)[0]
        predicted = design @ np.asarray([signal_sq, covariance_trace])
        denominator = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - float(np.square(y - predicted).sum()) / denominator if denominator > 0.0 else float("nan")
        noise_scale = (
            float(covariance_trace / signal_sq)
            if signal_sq > 0.0 and covariance_trace >= 0.0
            else float("nan")
        )
        output.append(
            {
                "objective": objective,
                "group": group,
                "batch_sizes": json.dumps(batch_sizes),
                "replicates_per_batch": json.dumps(
                    [len(buckets[(objective, group, batch_size)]) for batch_size in batch_sizes]
                ),
                "signal_l2_sq_fit": float(signal_sq),
                "covariance_trace_fit": float(covariance_trace),
                "covariance_noise_unit": noise_unit,
                "gradient_noise_scale": noise_scale,
                "isotropic_norm_snr_cosine_proxy_at_b32": (
                    1.0 / math.sqrt(1.0 + noise_scale / units_at_b32)
                    if math.isfinite(noise_scale) and noise_scale >= 0.0
                    else float("nan")
                ),
                "cosine_proxy_is_not_an_expected_cosine": True,
                "fit_r2": r2,
                "fit_valid_nonnegative": bool(signal_sq >= 0.0 and covariance_trace >= 0.0),
            }
        )
    return output


def _partition_audit_comparisons(
    records: dict[tuple[str, str], dict[str, Any]],
    *,
    replicates: int,
    groups: list[str],
    group_numel: dict[str, int],
    seeds: list[int],
    objective_weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Compare native B32 with an arithmetic mean of eight native-B4 gradients."""
    rows: list[dict[str, Any]] = []
    derived = {
        "structural": (("structural_direction", 1.0), ("structural_scale_x10", 1.0)),
        "training_total": (
            ("structural_direction", float(objective_weights["structural_outer"])),
            ("structural_scale_x10", float(objective_weights["structural_outer"])),
            ("behavioral", float(objective_weights["behavioral_outer"])),
        ),
    }
    aggregate_members = {
        "encoder_total": [group for group in groups if group.startswith("encoder_")],
        "global": list(groups),
    }

    def vector(objective: str, estimate_id: str, group: str, seed: int) -> np.ndarray:
        components = derived.get(objective, ((objective, 1.0),))
        members = aggregate_members.get(group, [group])
        chunks = []
        for member in members:
            value = sum(
                (
                    records[(component, estimate_id)]["panels"][(member, seed)]
                    * coefficient
                    for component, coefficient in components
                ),
                start=np.zeros_like(
                    records[(components[0][0], estimate_id)]["panels"][(member, seed)]
                ),
            )
            chunks.append(
                value * math.sqrt(float(group_numel[member]) / max(1, int(value.size)))
            )
        return np.concatenate(chunks)

    for replicate in range(replicates):
        native_id = f"b0032_r{replicate:02d}"
        audit_id = f"b0032_partition4_r{replicate:02d}"
        for objective in (*_OBJECTIVES, "structural", "training_total"):
            for group in (*groups, "encoder_total", "global"):
                values = []
                for seed in seeds:
                    native = vector(objective, native_id, group, seed).astype(np.float64)
                    partitioned = vector(objective, audit_id, group, seed).astype(np.float64)
                    native_norm = float(np.linalg.norm(native))
                    item = {
                        "cosine": _cosine(native, partitioned),
                        "relative_error": float(np.linalg.norm(native - partitioned))
                        / max(native_norm, 1e-30),
                        "norm_ratio": float(np.linalg.norm(partitioned)) / max(native_norm, 1e-30),
                    }
                    values.append(item)
                    rows.append(
                        {
                            "replicate": replicate,
                            "objective": objective,
                            "group": group,
                            "panel_seed": seed,
                            **item,
                        }
                    )
                rows.append(
                    {
                        "replicate": replicate,
                        "objective": objective,
                        "group": group,
                        "panel_seed": "median",
                        **{key: _median(item[key] for item in values) for key in values[0]},
                    }
                )
    return rows


def _reset_probe_rng(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "progress.log")],
    )
    logger = logging.getLogger("ae-gradient-batch-stability")

    batch_sizes = [int(value) for value in str(args.batch_sizes).split(",") if value.strip()]
    seeds = [int(value) for value in str(args.panel_seeds).split(",") if value.strip()]
    plan = _build_estimate_plan(
        batch_sizes, replicates=int(args.replicates), reference_batch_size=int(args.reference_batch_size)
    )
    _add_microbatch_partition_audits(plan, replicates=int(args.replicates))
    logger.info(
        "stage=startup output=%s device=%s dtype=bf16 seed_source=checkpoint data_cache=host-memory "
        "batch_sizes=%s replicates=%s reference_B=%s",
        output_dir,
        args.device,
        batch_sizes,
        args.replicates,
        args.reference_batch_size,
    )
    logger.info("stage=load_checkpoint path=%s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    checkpoint_step = int(checkpoint["step"])
    cfg = OmegaConf.create(checkpoint["config"])
    _assert_deterministic_probe_config(cfg)
    objective_weights = _objective_weights(cfg)
    model_state = checkpoint["model_state"]
    del checkpoint

    device = torch.device(args.device)
    model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
    model.load_state_dict(model_state, strict=True)
    del model_state
    model.to(device)
    model.train()
    named_parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    grouped_parameters: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in named_parameters.items():
        grouped_parameters[_parameter_group(name)].append((name, parameter))
    groups = sorted(grouped_parameters)
    panel_plans, group_numel = _make_panel_plans(
        grouped_parameters, panel_size=int(args.panel_size), seeds=seeds
    )
    logger.info(
        "stage=model_ready checkpoint_step=%s device=%s dtype=bf16-autocast objectives=%s groups=%s",
        checkpoint_step,
        device,
        _OBJECTIVES,
        groups,
    )

    cached: dict[str, list[Any]]
    sample_manifest: list[dict[str, Any]] = []
    pair = str(cfg.train.operator_bank.pair_manifest)
    seed = int(cfg.data.seed)
    logger.info("stage=load_data estimates=%s start_index=%s pair=%s", len(plan), args.start_index, pair)
    with operator_bank_data_pipeline(
        pair,
        seed=seed,
        repeat=True,
        permutation_views=bool(cfg.train.operator_bank.permutation_views),
        canonical_probability=float(cfg.train.operator_bank.canonical_probability),
        hot_shards=int(cfg.train.operator_bank.hot_shards),
        expected_pair_manifest_sha256=str(cfg.train.operator_bank.pair_manifest_sha256),
        rank=0,
        world_size=1,
        max_active_strata=int(cfg.train.operator_bank.max_active_strata),
        max_active_bundle_bytes=int(cfg.train.operator_bank.max_active_bundle_bytes),
        logger=logger,
    ) as (dataset, sampler):
        sampler.set_start_index(int(args.start_index))
        mixer = BalancedOperatorBankMixer(dataset, (dataset[request] for request in sampler), start_index=int(args.start_index))
        reference_batch_count = int(args.reference_batch_size) // 32
        max_batch_size = max(batch_sizes)
        replicate_batch_count = max(1, math.ceil(max_batch_size / 32))
        physical_batches = [
            _fetch_presliced_training_batch_cpu(dataset_iter=mixer, batch_size=32, logger=logger)
            for _ in range(reference_batch_count + int(args.replicates) * replicate_batch_count)
        ]
        observed_indices = tuple(
            int(logical_index)
            for batch in physical_batches
            for logical_index in batch.logical_indices
        )
        expected_indices = tuple(
            range(int(args.start_index), int(args.start_index) + len(physical_batches) * 32)
        )
        if observed_indices != expected_indices:
            raise RuntimeError(
                "operator-bank logical indices differ from the precommitted contiguous, unique window"
            )
        reference_batches = physical_batches[:reference_batch_count]
        replicate_blocks = {
            replicate: physical_batches[
                reference_batch_count + replicate * replicate_batch_count :
                reference_batch_count + (replicate + 1) * replicate_batch_count
            ]
            for replicate in range(int(args.replicates))
        }
        cached = _nested_batch_assignment(
            plan,
            reference_batches=reference_batches,
            replicate_blocks=replicate_blocks,
        )
        for estimate in plan:
            estimate_id = str(estimate["estimate_id"])
            for microbatch_index, batch in enumerate(cached[estimate_id]):
                sample_manifest.append(
                    {
                        "estimate_id": estimate_id,
                        "kind": estimate["kind"],
                        "effective_batch_size": estimate["effective_batch_size"],
                        "replicate": estimate["replicate"],
                        "microbatch_index": microbatch_index,
                        "microbatch_size": int(batch.W.shape[0]),
                        "logical_indices": json.dumps(list(batch.logical_indices)),
                    }
                )

    # Fail closed if nested prefixes were accidentally broken while materializing.
    for replicate in range(int(args.replicates)):
        prior: tuple[int, ...] = ()
        for batch_size in sorted(batch_sizes):
            estimate_id = f"b{batch_size:04d}_r{replicate:02d}"
            current = tuple(index for batch in cached[estimate_id] for index in batch.logical_indices)
            if prior and current[: len(prior)] != prior:
                raise RuntimeError(f"candidate batches are not nested prefixes for replicate={replicate}")
            if len(current) != batch_size or len(set(current)) != batch_size:
                raise RuntimeError(f"invalid/duplicate logical indices for {estimate_id}")
            prior = current

    records: dict[tuple[str, str], dict[str, Any]] = {}
    estimate_rows: list[dict[str, Any]] = []
    total_work = len(_OBJECTIVES) * len(plan)
    completed = 0
    for objective in _OBJECTIVES:
        for estimate in plan:
            estimate_id = str(estimate["estimate_id"])
            effective_batch_size = int(estimate["effective_batch_size"])
            model.zero_grad(set_to_none=True)
            rng_seed = 910_000 + checkpoint_step + int(estimate["replicate"]) + (
                0 if estimate["kind"] == "reference" else effective_batch_size * 100
            )
            _reset_probe_rng(rng_seed, device)
            weighted_loss = 0.0
            for batch in cached[estimate_id]:
                W, x, x_mask, d_in_mask, d_out_mask = [
                    tensor.to(device, non_blocking=False)
                    for tensor in (batch.W, batch.x, batch.x_mask, batch.d_in_mask, batch.d_out_mask)
                ]
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast:
                    loss = _objective_loss(
                        objective,
                        model=model,
                        cfg=cfg,
                        W=W,
                        x=x,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                    )
                    weight = int(W.shape[0]) / effective_batch_size
                    scaled_loss = loss * weight
                scaled_loss.backward()
                weighted_loss += float(loss.detach().item()) * weight
                del W, x, x_mask, d_in_mask, d_out_mask, loss, scaled_loss
            exact, panels = _extract_gradient_summary(
                named_parameters,
                grouped_parameters,
                panel_plans,
                group_numel,
                panel_size=int(args.panel_size),
                seeds=seeds,
            )
            records[(objective, estimate_id)] = {"exact": exact, "panels": panels}
            for group in groups:
                row = {
                    "objective": objective,
                    "estimate_id": estimate_id,
                    "kind": estimate["kind"],
                    "effective_batch_size": effective_batch_size,
                    "replicate": estimate["replicate"],
                    "rng_seed": rng_seed,
                    "loss": weighted_loss,
                    "group": group,
                    **exact[group],
                }
                panel_norm_estimates = []
                for panel_seed in seeds:
                    vector = panels[(group, panel_seed)]
                    estimate_sq = float(group_numel[group]) * float(np.mean(vector.astype(np.float64) ** 2))
                    panel_norm_estimates.append(math.sqrt(max(0.0, estimate_sq)))
                row["panel_l2_median"] = _median(panel_norm_estimates)
                row["panel_l2_relative_error"] = (
                    abs(float(row["panel_l2_median"]) - float(row["exact_l2"])) / float(row["exact_l2"])
                    if float(row["exact_l2"]) > 0.0
                    else float("nan")
                )
                estimate_rows.append(row)
            completed += 1
            logger.info(
                "stage=gradient objective=%s estimate=%s B=%s progress=%s/%s loss=%.6g",
                objective,
                estimate_id,
                effective_batch_size,
                completed,
                total_work,
                weighted_loss,
            )
            model.zero_grad(set_to_none=True)

    comparisons, summary = _analyze_panels(
        records,
        plan,
        groups=groups,
        group_numel=group_numel,
        seeds=seeds,
        exact_rows=estimate_rows,
        objective_weights=objective_weights,
    )
    interactions = _objective_interactions(
        records, plan, groups=groups, group_numel=group_numel, seeds=seeds
    )
    noise_models = _fit_noise_models(estimate_rows, plan)
    partition_audit = _partition_audit_comparisons(
        records,
        replicates=int(args.replicates),
        groups=groups,
        group_numel=group_numel,
        seeds=seeds,
        objective_weights=objective_weights,
    )
    _write_csv(output_dir / "sample_manifest.csv", sample_manifest)
    _write_csv(output_dir / "gradient_estimates.csv", estimate_rows)
    _write_csv(output_dir / "gradient_comparisons.csv", comparisons)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "objective_interactions.csv", interactions)
    _write_csv(output_dir / "noise_models.csv", noise_models)
    _write_csv(output_dir / "microbatch_partition_audit.csv", partition_audit)
    panel_errors = [
        float(row["panel_l2_relative_error"])
        for row in estimate_rows
        if math.isfinite(float(row["panel_l2_relative_error"]))
    ]
    panel_error_median = _median(panel_errors)
    panel_error_p95 = float(np.quantile(panel_errors, 0.95)) if panel_errors else float("nan")
    panel_quality_pass = bool(
        math.isfinite(panel_error_median)
        and math.isfinite(panel_error_p95)
        and panel_error_median <= 0.15
        and panel_error_p95 <= 0.30
    )
    report = {
        "schema": "weightclip_ae_gradient_batch_stability_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "amp_dtype": "bfloat16" if device.type == "cuda" else "float32",
        "batch_sizes": batch_sizes,
        "replicates": int(args.replicates),
        "reference_batch_size": int(args.reference_batch_size),
        "production_microbatch_size": 32,
        "effective_batch_semantics": "native_B_for_B_le_32; arithmetic_mean_of_production_microbatch32_gradients_for_B_gt_32",
        "nested_sampling": "four disjoint B256 blocks; every B is the prefix of its replicate block; independent B1024 reference",
        "objective_weights_from_checkpoint": objective_weights,
        "objectives": {
            "structural_direction": f'{objective_weights["structural_direction"]} * structural direction',
            "structural_scale_x10": f'{objective_weights["structural_scale"]} * structural log-scale',
            "structural": "derived linear sum of structural_direction and structural_scale_x10 panels",
            "behavioral": (
                f'raw family: {objective_weights["behavioral_operator"]} * operator + '
                f'{objective_weights["behavioral_direction"]} * direction + '
                f'{objective_weights["behavioral_scale"]} * log-scale'
            ),
            "training_total": "structural_outer * structural + behavioral_outer * behavioral from checkpoint",
        },
        "groups": {group: int(group_numel[group]) for group in groups},
        "panel": {
            "kind": "uniform-coordinate sampling with replacement",
            "size_per_group": int(args.panel_size),
            "seeds": seeds,
            "cosines_are_estimates_not_exact": True,
            "median_l2_relative_error": panel_error_median,
            "p95_l2_relative_error": panel_error_p95,
            "quality_gate": "median<=0.15 and p95<=0.30",
            "quality_pass": panel_quality_pass,
            "cross_term_and_cosine_claims_allowed": panel_quality_pass,
        },
        "microbatch_partition_audit_rows": len(partition_audit),
        "artifacts": {
            "sample_manifest": str(output_dir / "sample_manifest.csv"),
            "gradient_estimates": str(output_dir / "gradient_estimates.csv"),
            "gradient_comparisons": str(output_dir / "gradient_comparisons.csv"),
            "summary": str(output_dir / "summary.csv"),
            "objective_interactions": str(output_dir / "objective_interactions.csv"),
            "noise_models": str(output_dir / "noise_models.csv"),
            "microbatch_partition_audit": str(output_dir / "microbatch_partition_audit.csv"),
            "progress_log": str(output_dir / "progress.log"),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("stage=complete output=%s", output_dir)


if __name__ == "__main__":
    main()
