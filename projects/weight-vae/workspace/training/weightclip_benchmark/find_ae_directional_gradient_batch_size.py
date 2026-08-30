from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import random
import statistics
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer, operator_bank_data_pipeline
from big_vae.models import build_weight_quantile_vae
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu
from training.big_vae.runtime import _maybe_compile, _set_speed_optimizations
from training.weightclip_benchmark.analyze_ae_gradient_batch_stability import (
    _assert_deterministic_probe_config,
    _extract_gradient_summary,
    _make_panel_plans,
    _objective_loss,
    _parameter_group,
)


DEFAULT_BATCH_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_PANEL_SEEDS = (31_001, 31_013, 31_019, 31_033)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the smallest effective batch whose structural-direction gradient "
            "is repeatable across independent sample blocks."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--panel-size", type=int, default=65_536)
    parser.add_argument("--panel-seeds", default=",".join(map(str, DEFAULT_PANEL_SEEDS)))
    parser.add_argument("--start-index", type=int, default=8_000_000)
    parser.add_argument("--median-cosine-threshold", type=float, default=0.90)
    parser.add_argument("--p10-cosine-threshold", type=float, default=0.80)
    parser.add_argument("--norm-cv-threshold", type=float, default=0.25)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def _quantile(values: Iterable[float], q: float) -> float:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def _coefficient_of_variation(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    mean = statistics.fmean(finite)
    return float(statistics.pstdev(finite) / mean) if mean > 0.0 else float("nan")


def _validate_grid(batch_sizes: Iterable[int], *, replicates: int, microbatch_size: int) -> list[int]:
    sizes = sorted({int(value) for value in batch_sizes})
    if replicates < 4:
        raise ValueError("directional batch-size search requires at least four independent replicates")
    if not sizes or any(value < microbatch_size or value % microbatch_size for value in sizes):
        raise ValueError(
            f"all batch sizes must be positive multiples of microbatch_size={microbatch_size}: {sizes}"
        )
    for left, right in zip(sizes, sizes[1:], strict=False):
        if right != 2 * left:
            raise ValueError(f"batch-size grid must double at every step, got {left} -> {right}")
    return sizes


def _scaled_group_panel(
    panels: dict[tuple[str, int], np.ndarray],
    *,
    group: str,
    seed: int,
    group_numel: dict[str, int],
    divisor: int,
) -> np.ndarray:
    vector = panels[(group, seed)].astype(np.float64, copy=True)
    vector *= math.sqrt(float(group_numel[group]) / max(1, vector.size)) / float(divisor)
    return vector


def _aggregate_panel(
    record: dict[str, Any],
    *,
    group: str,
    seed: int,
    leaf_groups: list[str],
) -> np.ndarray:
    if group != "encoder_total":
        return record["panels"][(group, seed)]
    return np.concatenate([record["panels"][(leaf, seed)] for leaf in leaf_groups])


def _summarize_stability(
    records: dict[tuple[int, int], dict[str, Any]],
    *,
    batch_sizes: list[int],
    replicates: int,
    groups: list[str],
    leaf_groups: list[str],
    seeds: list[int],
    median_cosine_threshold: float,
    p10_cosine_threshold: float,
    norm_cv_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        for group in groups:
            per_seed_pair_cosines: list[float] = []
            per_seed_pair_relative_differences: list[float] = []
            split_cosines: list[float] = []
            for seed in seeds:
                vectors = [
                    _aggregate_panel(
                        records[(batch_size, replicate)],
                        group=group,
                        seed=seed,
                        leaf_groups=leaf_groups,
                    )
                    for replicate in range(replicates)
                ]
                for left_index, right_index in combinations(range(replicates), 2):
                    left = vectors[left_index]
                    right = vectors[right_index]
                    per_seed_pair_cosines.append(_cosine(left, right))
                    scale = 0.5 * (float(np.linalg.norm(left)) + float(np.linalg.norm(right)))
                    per_seed_pair_relative_differences.append(
                        float(np.linalg.norm(left - right) / scale) if scale > 0.0 else float("nan")
                    )
                midpoint = replicates // 2
                left_mean = np.mean(np.stack(vectors[:midpoint]), axis=0)
                right_mean = np.mean(np.stack(vectors[midpoint:]), axis=0)
                split_cosines.append(_cosine(left_mean, right_mean))

            exact_norms = [
                float(records[(batch_size, replicate)]["exact_l2"][group])
                for replicate in range(replicates)
            ]
            panel_errors: list[float] = []
            for replicate, exact_norm in enumerate(exact_norms):
                if exact_norm <= 0.0:
                    continue
                panel_norms = [
                    float(
                        np.linalg.norm(
                            _aggregate_panel(
                                records[(batch_size, replicate)],
                                group=group,
                                seed=seed,
                                leaf_groups=leaf_groups,
                            )
                        )
                    )
                    for seed in seeds
                ]
                panel_errors.append(
                    abs(float(statistics.median(panel_norms)) - exact_norm) / exact_norm
                )
            median_cosine = _quantile(per_seed_pair_cosines, 0.50)
            p10_cosine = _quantile(per_seed_pair_cosines, 0.10)
            norm_cv = _coefficient_of_variation(exact_norms)
            panel_l2_error_median = _quantile(panel_errors, 0.50)
            panel_quality_pass = bool(panel_l2_error_median <= 0.15)
            passes = bool(
                median_cosine >= median_cosine_threshold
                and p10_cosine >= p10_cosine_threshold
                and norm_cv <= norm_cv_threshold
                and panel_quality_pass
            )
            summary.append(
                {
                    "objective": "structural_direction",
                    "group": group,
                    "effective_batch_size": batch_size,
                    "replicates": replicates,
                    "pairwise_comparisons": math.comb(replicates, 2) * len(seeds),
                    "median_pairwise_cosine": median_cosine,
                    "p10_pairwise_cosine": p10_cosine,
                    "min_pairwise_cosine": _quantile(per_seed_pair_cosines, 0.0),
                    "median_split_half_cosine": _quantile(split_cosines, 0.50),
                    "median_pairwise_relative_difference": _quantile(
                        per_seed_pair_relative_differences, 0.50
                    ),
                    "exact_norm_mean": statistics.fmean(exact_norms),
                    "exact_norm_cv": norm_cv,
                    "panel_l2_error_median": panel_l2_error_median,
                    "panel_quality_pass": int(panel_quality_pass),
                    "passes_single_batch": int(passes),
                }
            )

    by_group_batch = {
        (str(row["group"]), int(row["effective_batch_size"])): row for row in summary
    }
    decisions: list[dict[str, Any]] = []
    for group in groups:
        stable_batch: int | None = None
        for current, following in zip(batch_sizes, batch_sizes[1:], strict=False):
            if bool(by_group_batch[(group, current)]["passes_single_batch"]) and bool(
                by_group_batch[(group, following)]["passes_single_batch"]
            ):
                stable_batch = current
                break
        decisions.append(
            {
                "objective": "structural_direction",
                "group": group,
                "stable_batch_size": stable_batch if stable_batch is not None else "unresolved",
                "largest_tested_batch_size": batch_sizes[-1],
                "criterion": (
                    f"median_cos>={median_cosine_threshold}, p10_cos>={p10_cosine_threshold}, "
                    f"norm_cv<={norm_cv_threshold}, sustained_at_next_doubling"
                ),
                "resolved": int(stable_batch is not None),
            }
        )
    return summary, decisions


def _plot_summary(
    path: Path,
    summary: list[dict[str, Any]],
    groups: list[str],
    *,
    median_cosine_threshold: float,
    p10_cosine_threshold: float,
    norm_cv_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for group in groups:
        rows = sorted(
            (row for row in summary if row["group"] == group),
            key=lambda row: int(row["effective_batch_size"]),
        )
        x = [int(row["effective_batch_size"]) for row in rows]
        axes[0].plot(x, [float(row["median_pairwise_cosine"]) for row in rows], marker="o", label=group)
        axes[1].plot(x, [float(row["exact_norm_cv"]) for row in rows], marker="o", label=group)
    axes[0].axhline(median_cosine_threshold, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(p10_cosine_threshold, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("independent-gradient cosine")
    axes[0].set_title("Directional gradient agreement")
    axes[1].axhline(norm_cv_threshold, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("exact gradient-norm CV")
    axes[1].set_title("Directional gradient norm dispersion")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("effective batch size")
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    batch_sizes = _validate_grid(
        (int(value) for value in str(args.batch_sizes).split(",") if value.strip()),
        replicates=int(args.replicates),
        microbatch_size=int(args.microbatch_size),
    )
    panel_seeds = [int(value) for value in str(args.panel_seeds).split(",") if value.strip()]
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "progress.log")],
    )
    logger = logging.getLogger("ae-directional-gradient-batch-search")
    logger.info(
        "stage=startup checkpoint=%s output=%s device=%s dtype=bf16 objective=structural_direction "
        "batch_sizes=%s replicates=%s microbatch=%s",
        checkpoint_path,
        output_dir,
        args.device,
        batch_sizes,
        args.replicates,
        args.microbatch_size,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    checkpoint_step = int(checkpoint["step"])
    cfg = OmegaConf.create(checkpoint["config"])
    _assert_deterministic_probe_config(cfg)
    model_state = checkpoint["model_state"]
    del checkpoint

    device = torch.device(args.device)
    _set_speed_optimizations(cfg, device)
    model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
    model.load_state_dict(model_state, strict=True)
    del model_state
    model.to(device)
    model.train()
    named_parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    grouped_parameters: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in named_parameters.items():
        grouped_parameters[_parameter_group(name)].append((name, parameter))
    leaf_groups = sorted(group for group in grouped_parameters if group.startswith("encoder_"))
    groups = [*leaf_groups, "encoder_total"]
    panel_plans, group_numel = _make_panel_plans(
        grouped_parameters,
        panel_size=int(args.panel_size),
        seeds=panel_seeds,
    )
    # Production B32 only fits this 725M model through the configured compiled
    # execution path.  Keep parameter handles from the eager module (for clean
    # group names), then execute forwards through the compiled wrapper.
    execution_model = _maybe_compile(model, cfg=cfg, logger=logger)
    logger.info(
        "stage=model_ready checkpoint_step=%s parameters=%s encoder_groups=%s compile=%s mode=%s",
        checkpoint_step,
        sum(parameter.numel() for parameter in model.parameters()),
        groups,
        bool(cfg.train.get("compile", False)),
        cfg.train.get("compile_mode", "default"),
    )

    records: dict[tuple[int, int], dict[str, Any]] = {}
    estimate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    maximum_batch_size = batch_sizes[-1]
    maximum_microbatches = maximum_batch_size // int(args.microbatch_size)
    total_microbatches = int(args.replicates) * maximum_microbatches
    completed_microbatches = 0

    operator = cfg.train.operator_bank
    random.seed(910_000 + checkpoint_step)
    np.random.seed((910_000 + checkpoint_step) % (2**32))
    torch.manual_seed(910_000 + checkpoint_step)
    if device.type == "cuda":
        torch.cuda.manual_seed(910_000 + checkpoint_step)

    with operator_bank_data_pipeline(
        str(operator.pair_manifest),
        seed=int(cfg.data.seed),
        repeat=True,
        permutation_views=bool(operator.permutation_views),
        canonical_probability=float(operator.canonical_probability),
        hot_shards=int(operator.hot_shards),
        expected_pair_manifest_sha256=str(operator.pair_manifest_sha256),
        rank=0,
        world_size=1,
        max_active_strata=int(operator.max_active_strata),
        max_active_bundle_bytes=int(operator.max_active_bundle_bytes),
        logger=logger,
    ) as (dataset, sampler):
        sampler.set_start_index(int(args.start_index))
        mixer = BalancedOperatorBankMixer(
            dataset,
            (dataset[request] for request in sampler),
            start_index=int(args.start_index),
        )
        for replicate in range(int(args.replicates)):
            model.zero_grad(set_to_none=True)
            loss_sum = 0.0
            replicate_start = int(args.start_index) + replicate * maximum_batch_size
            for microbatch_index in range(1, maximum_microbatches + 1):
                batch = _fetch_presliced_training_batch_cpu(
                    dataset_iter=mixer,
                    batch_size=int(args.microbatch_size),
                    logger=logger,
                )
                expected_start = replicate_start + (microbatch_index - 1) * int(args.microbatch_size)
                expected = tuple(range(expected_start, expected_start + int(args.microbatch_size)))
                if tuple(batch.logical_indices) != expected:
                    raise RuntimeError(
                        f"unexpected logical indices for replicate={replicate} microbatch={microbatch_index}"
                    )
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
                        "structural_direction",
                        model=execution_model,
                        cfg=cfg,
                        W=W,
                        x=x,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                    )
                loss.backward()
                loss_sum += float(loss.detach().item())
                del W, x, x_mask, d_in_mask, d_out_mask, loss
                completed_microbatches += 1

                effective_batch_size = microbatch_index * int(args.microbatch_size)
                if effective_batch_size in batch_sizes:
                    exact, panels = _extract_gradient_summary(
                        named_parameters,
                        grouped_parameters,
                        panel_plans,
                        group_numel,
                        panel_size=int(args.panel_size),
                        seeds=panel_seeds,
                    )
                    scaled_panels: dict[tuple[str, int], np.ndarray] = {}
                    for group in leaf_groups:
                        for seed in panel_seeds:
                            scaled_panels[(group, seed)] = _scaled_group_panel(
                                panels,
                                group=group,
                                seed=seed,
                                group_numel=group_numel,
                                divisor=microbatch_index,
                            )
                    exact_l2 = {
                        group: float(exact[group]["exact_l2"]) / microbatch_index
                        for group in leaf_groups
                    }
                    exact_l2["encoder_total"] = math.sqrt(
                        sum(exact_l2[group] ** 2 for group in leaf_groups)
                    )
                    records[(effective_batch_size, replicate)] = {
                        "panels": scaled_panels,
                        "exact_l2": exact_l2,
                    }
                    for group in groups:
                        estimate_rows.append(
                            {
                                "objective": "structural_direction",
                                "group": group,
                                "effective_batch_size": effective_batch_size,
                                "replicate": replicate,
                                "exact_l2": exact_l2[group],
                                "loss": loss_sum / microbatch_index,
                                "logical_start": replicate_start,
                                "logical_end_exclusive": replicate_start + effective_batch_size,
                            }
                        )
                    logger.info(
                        "stage=gradient B=%s replicate=%s/%s progress=%s/%s loss=%.6g "
                        "encoder_l2=%.6g elapsed=%.1fs",
                        effective_batch_size,
                        replicate + 1,
                        args.replicates,
                        completed_microbatches,
                        total_microbatches,
                        loss_sum / microbatch_index,
                        exact_l2["encoder_total"],
                        time.perf_counter() - started,
                    )
            sample_rows.append(
                {
                    "replicate": replicate,
                    "logical_start": replicate_start,
                    "logical_end_exclusive": replicate_start + maximum_batch_size,
                    "count": maximum_batch_size,
                }
            )
            model.zero_grad(set_to_none=True)

    summary, decisions = _summarize_stability(
        records,
        batch_sizes=batch_sizes,
        replicates=int(args.replicates),
        groups=groups,
        leaf_groups=leaf_groups,
        seeds=panel_seeds,
        median_cosine_threshold=float(args.median_cosine_threshold),
        p10_cosine_threshold=float(args.p10_cosine_threshold),
        norm_cv_threshold=float(args.norm_cv_threshold),
    )
    _write_csv(output_dir / "sample_blocks.csv", sample_rows)
    _write_csv(output_dir / "gradient_estimates.csv", estimate_rows)
    _write_csv(output_dir / "stability_summary.csv", summary)
    _write_csv(output_dir / "batch_decisions.csv", decisions)
    _plot_summary(
        output_dir / "directional_gradient_stability.png",
        summary,
        groups,
        median_cosine_threshold=float(args.median_cosine_threshold),
        p10_cosine_threshold=float(args.p10_cosine_threshold),
        norm_cv_threshold=float(args.norm_cv_threshold),
    )

    primary = next(row for row in decisions if row["group"] == "encoder_total")
    report = {
        "schema": "weightclip_ae_directional_gradient_batch_search_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "objective": "structural_direction",
        "device": str(device),
        "amp_dtype": "bfloat16" if device.type == "cuda" else "float32",
        "microbatch_size": int(args.microbatch_size),
        "batch_sizes": batch_sizes,
        "replicates": int(args.replicates),
        "sampling": "four disjoint maximum-B blocks; smaller B are nested prefixes",
        "reference": "none; stability is agreement between independent same-size estimates",
        "thresholds": {
            "median_pairwise_cosine": float(args.median_cosine_threshold),
            "p10_pairwise_cosine": float(args.p10_cosine_threshold),
            "exact_norm_cv": float(args.norm_cv_threshold),
            "sustained_at_next_doubling": True,
        },
        "encoder_total_decision": primary,
        "group_decisions": decisions,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "sample_blocks": str(output_dir / "sample_blocks.csv"),
            "gradient_estimates": str(output_dir / "gradient_estimates.csv"),
            "stability_summary": str(output_dir / "stability_summary.csv"),
            "batch_decisions": str(output_dir / "batch_decisions.csv"),
            "plot": str(output_dir / "directional_gradient_stability.png"),
            "progress_log": str(output_dir / "progress.log"),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info(
        "stage=complete encoder_total_B_star=%s elapsed=%.1fs output=%s",
        primary["stable_batch_size"],
        report["elapsed_seconds"],
        output_dir,
    )


if __name__ == "__main__":
    main()
