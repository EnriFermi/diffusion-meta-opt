from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import encode_weights
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    PreconditioningState,
    preconditioning_regularizer,
)
from scripts.audit_variant_a_estimator_stability import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_POOL_SHA256,
    EXPECTED_RECORDS_SHA256,
    _atomic_pair_loss,
    _fixed_reference_splits,
    _flatten_active_gradients,
    _load_run,
    _module_slices,
    _preflight_decomposition,
    _probe_cfg,
    git_metadata,
    parse_grid,
    quantile,
    sha256_file,
    sha256_tensor,
    stable_uint63,
    symmetric_relative_error,
    vector_norm,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_hvp_batch_size_ablation_h2048"
)
PROTOCOL_PATH = DEFAULT_OUTPUT_DIR / "protocol.md"
PROTOCOL_ID = "a_hvp_batch_size_ablation_h2048_v1"
TRAIN_COUNT = 16384
TRAIN_STATES = 2
TRAIN_PAIRS = 4


@dataclass(slots=True)
class PreparedRepeat:
    repeat: int
    step_key: int
    state_indices: list[int]
    train_positions: list[int]
    state_draw_seeds: list[int]
    latent_hashes: list[str]
    z_values: torch.Tensor
    records: list[dict[str, Any]]


@dataclass(slots=True)
class BatchResult:
    requested_batch_size: int
    effective_batch_size: int
    batch_mode: str
    gradients: list[torch.Tensor]
    scalars: list[float]
    atomic_rows: list[dict[str, Any]]
    coverage_rows: list[dict[str, Any]]
    overlap_rows: list[dict[str, Any]]
    elapsed_sec: float


def indexed_generators(
    seed: int,
    repeat: int,
    state_position: int,
    pair_position: int,
) -> tuple[torch.Generator, torch.Generator, int, int]:
    seed_1 = stable_uint63(PROTOCOL_ID, seed, "probe", repeat, state_position, pair_position, 0)
    seed_2 = stable_uint63(PROTOCOL_ID, seed, "probe", repeat, state_position, pair_position, 1)
    return (
        torch.Generator(device="cpu").manual_seed(seed_1),
        torch.Generator(device="cpu").manual_seed(seed_2),
        int(seed_1),
        int(seed_2),
    )


def sample_state_indices(
    train_indices: torch.Tensor,
    *,
    count: int,
    seed: int,
    repeat: int,
) -> tuple[list[int], list[int], list[int]]:
    state_indices: list[int] = []
    positions: list[int] = []
    draw_seeds: list[int] = []
    for state_position in range(int(count)):
        draw_seed = stable_uint63(PROTOCOL_ID, seed, "state", repeat, state_position, -1, -1)
        generator = torch.Generator(device="cpu").manual_seed(draw_seed)
        position = int(torch.randint(0, int(train_indices.numel()), (1,), generator=generator).item())
        state_indices.append(int(train_indices[position].item()))
        positions.append(position)
        draw_seeds.append(int(draw_seed))
    return state_indices, positions, draw_seeds


def prepare_repeats(
    run: Any,
    *,
    repeats: int,
    seed: int,
    device: torch.device,
) -> list[PreparedRepeat]:
    prepared: list[PreparedRepeat] = []
    for repeat in range(int(repeats)):
        state_indices, positions, draw_seeds = sample_state_indices(
            run.train_indices,
            count=TRAIN_STATES,
            seed=seed,
            repeat=repeat,
        )
        selected = run.weights[state_indices].to(device=device, dtype=torch_dtype(run.cfg))
        with torch.no_grad():
            z_values = encode_weights(run.vae, run.normalizer, selected).detach()
        records = run.records.iloc[state_indices].to_dict(orient="records")
        for record, source_index in zip(records, state_indices, strict=True):
            record["source_weight_index"] = int(source_index)
            task_set = run.task_tensors[str(record.get("task_name", "tiny_cnn"))]
            if int(task_set.train_labels.shape[0]) != TRAIN_COUNT:
                raise ValueError(
                    f"task {record.get('task_name')} train count is {int(task_set.train_labels.shape[0])}, "
                    f"expected {TRAIN_COUNT}"
                )
        prepared.append(
            PreparedRepeat(
                repeat=repeat,
                step_key=10 * (repeat + 1),
                state_indices=state_indices,
                train_positions=positions,
                state_draw_seeds=draw_seeds,
                latent_hashes=[sha256_tensor(z_values[position]) for position in range(TRAIN_STATES)],
                z_values=z_values,
                records=records,
            )
        )
    return prepared


def assert_excluded_gradients_zero(excluded: Sequence[tuple[str, torch.nn.Parameter]]) -> None:
    for name, parameter in excluded:
        if parameter.grad is not None and bool((parameter.grad.detach() != 0).any().item()):
            raise RuntimeError(f"excluded parameter received an A gradient: {name}")


def cuda_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
            "cuda_peak_allocated_bytes": 0,
            "cuda_peak_reserved_bytes": 0,
        }
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def sequential_memory_smoke(
    run: Any,
    prepared: PreparedRepeat,
    *,
    batch_sizes: Sequence[int],
    active: Sequence[tuple[str, torch.nn.Parameter]],
    excluded: Sequence[tuple[str, torch.nn.Parameter]],
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=1, batch_size=int(batch_size))
        generator_1, generator_2, probe_seed_1, probe_seed_2 = indexed_generators(seed, 0, 0, 0)
        run.vae.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        row: dict[str, Any] = {
            "smoke_kind": "sequential_atomic_pair",
            "requested_batch_size": int(batch_size),
            "probe_seed_1": probe_seed_1,
            "probe_seed_2": probe_seed_2,
            "step_key": prepared.step_key,
            "success": False,
            "error": "",
        }
        try:
            loss, stats = _atomic_pair_loss(
                cfg=cfg,
                run=run,
                z=prepared.z_values[0],
                record=prepared.records[0],
                step=prepared.step_key,
                pair_index=0,
                probe_generator_1=generator_1,
                probe_generator_2=generator_2,
            )
            loss.backward()
            assert_excluded_gradients_zero(excluded)
            gradient = _flatten_active_gradients(active)
            row.update(
                {
                    "success": True,
                    "a_scalar": float(loss.detach().cpu().item()),
                    "gradient_norm": vector_norm(gradient),
                    "effective_batch_size": int(stats["batch_size_effective"]),
                    "batch_mode": "full" if str(stats["batch_1_hash"]) == "full" else "window",
                }
            )
            del gradient, loss
        except torch.OutOfMemoryError as error:
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            row["elapsed_sec"] = float(time.perf_counter() - started)
            row.update(cuda_memory(device))
            rows.append(row)
            run.vae.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[hvp_batch] sequential_smoke {json.dumps(row, sort_keys=True)}", flush=True)
    return pd.DataFrame(rows)


def simultaneous_a_memory_smoke(
    run: Any,
    prepared: PreparedRepeat,
    *,
    batch_sizes: Sequence[int],
    active: Sequence[tuple[str, torch.nn.Parameter]],
    excluded: Sequence[tuple[str, torch.nn.Parameter]],
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        cfg = _probe_cfg(
            run.cfg,
            sample_count=TRAIN_STATES,
            pair_count=TRAIN_PAIRS,
            batch_size=int(batch_size),
        )
        generator_seed = stable_uint63(PROTOCOL_ID, seed, "simultaneous_memory")
        generator = torch.Generator(device="cpu").manual_seed(generator_seed)
        run.vae.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        row: dict[str, Any] = {
            "smoke_kind": "simultaneous_a_only_2x4",
            "requested_batch_size": int(batch_size),
            "generator_seed": int(generator_seed),
            "step_key": prepared.step_key,
            "success": False,
            "error": "",
        }
        try:
            loss, stats = preconditioning_regularizer(
                cfg,
                state=PreconditioningState(),
                vae=run.vae,
                normalizer=run.normalizer,
                z_samples=prepared.z_values,
                records=prepared.records,
                task_tensors=run.task_tensors,
                spec=run.spec,
                step=prepared.step_key,
                generator=generator,
            )
            loss.backward()
            assert_excluded_gradients_zero(excluded)
            gradient = _flatten_active_gradients(active)
            row.update(
                {
                    "success": True,
                    "a_scalar": float(loss.detach().cpu().item()),
                    "gradient_norm": vector_norm(gradient),
                    "effective_batch_size": TRAIN_COUNT if int(batch_size) >= TRAIN_COUNT else int(batch_size),
                    "batch_mode": "full" if int(batch_size) >= TRAIN_COUNT else "window",
                    "reported_sample_count": float(stats["precond_sample_count"]),
                    "reported_pair_count": float(stats["precond_pair_count"]),
                }
            )
            del gradient, loss
        except torch.OutOfMemoryError as error:
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            row["elapsed_sec"] = float(time.perf_counter() - started)
            row.update(cuda_memory(device))
            rows.append(row)
            run.vae.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[hvp_batch] simultaneous_smoke {json.dumps(row, sort_keys=True)}", flush=True)
    return pd.DataFrame(rows)


def circular_mask(offset: int | None, *, batch_size: int, train_count: int) -> np.ndarray:
    if offset is None or int(batch_size) >= int(train_count):
        return np.ones(int(train_count), dtype=bool)
    indices = (int(offset) + np.arange(int(batch_size), dtype=np.int64)) % int(train_count)
    mask = np.zeros(int(train_count), dtype=bool)
    mask[indices] = True
    return mask


def window_diagnostics(
    branches: Sequence[Mapping[str, Any]],
    *,
    requested_batch_size: int,
    repeat: int,
    state_position: int,
    source_weight_index: int,
    task_name: str,
    step_key: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(branches) != 2 * TRAIN_PAIRS:
        raise ValueError(f"expected {2 * TRAIN_PAIRS} branches, got {len(branches)}")
    train_counts = {int(branch["train_count"]) for branch in branches}
    if train_counts != {TRAIN_COUNT}:
        raise ValueError(f"unexpected task train counts: {sorted(train_counts)}")
    masks = [
        circular_mask(
            None if branch["offset"] is None else int(branch["offset"]),
            batch_size=int(branch["effective_batch_size"]),
            train_count=TRAIN_COUNT,
        )
        for branch in branches
    ]
    multiplicity = np.stack(masks, axis=0).sum(axis=0)
    coverage = {
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(branches[0]["effective_batch_size"]),
        "batch_mode": str(branches[0]["batch_mode"]),
        "repeat": int(repeat),
        "step_key": int(step_key),
        "state_position": int(state_position),
        "source_weight_index": int(source_weight_index),
        "task_name": str(task_name),
        "branch_count": int(len(branches)),
        "union_count": int((multiplicity > 0).sum()),
        "union_fraction": float((multiplicity > 0).mean()),
        "mean_multiplicity_over_dataset": float(multiplicity.mean()),
        "mean_multiplicity_when_covered": float(multiplicity[multiplicity > 0].mean()),
        "max_multiplicity": int(multiplicity.max()),
        "fraction_multiplicity_ge2": float((multiplicity >= 2).mean()),
        "fraction_multiplicity_ge4": float((multiplicity >= 4).mean()),
    }
    overlaps: list[dict[str, Any]] = []
    for left in range(len(branches)):
        for right in range(left + 1, len(branches)):
            overlap = int(np.logical_and(masks[left], masks[right]).sum())
            denominator = max(1, min(int(masks[left].sum()), int(masks[right].sum())))
            overlaps.append(
                {
                    "requested_batch_size": int(requested_batch_size),
                    "repeat": int(repeat),
                    "step_key": int(step_key),
                    "state_position": int(state_position),
                    "source_weight_index": int(source_weight_index),
                    "task_name": str(task_name),
                    "left_pair": int(branches[left]["pair_position"]),
                    "left_branch": int(branches[left]["branch"]),
                    "right_pair": int(branches[right]["pair_position"]),
                    "right_branch": int(branches[right]["branch"]),
                    "same_pair": bool(branches[left]["pair_position"] == branches[right]["pair_position"]),
                    "same_branch": bool(branches[left]["branch"] == branches[right]["branch"]),
                    "overlap_count": overlap,
                    "overlap_fraction": float(overlap / denominator),
                }
            )
    return coverage, overlaps


def run_batch(
    run: Any,
    prepared_repeats: Sequence[PreparedRepeat],
    *,
    batch_size: int,
    active: Sequence[tuple[str, torch.nn.Parameter]],
    excluded: Sequence[tuple[str, torch.nn.Parameter]],
    seed: int,
    device: torch.device,
) -> BatchResult:
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=1, batch_size=int(batch_size))
    gradients: list[torch.Tensor] = []
    scalars: list[float] = []
    atomic_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for prepared in prepared_repeats:
        repeat_started = time.perf_counter()
        gradient_sum: torch.Tensor | None = None
        scalar_sum = 0.0
        for state_position in range(TRAIN_STATES):
            record = prepared.records[state_position]
            branch_rows: list[dict[str, Any]] = []
            for pair_position in range(TRAIN_PAIRS):
                generator_1, generator_2, probe_seed_1, probe_seed_2 = indexed_generators(
                    seed,
                    prepared.repeat,
                    state_position,
                    pair_position,
                )
                run.vae.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                unit_started = time.perf_counter()
                loss, stats = _atomic_pair_loss(
                    cfg=cfg,
                    run=run,
                    z=prepared.z_values[state_position],
                    record=record,
                    step=prepared.step_key,
                    pair_index=pair_position,
                    probe_generator_1=generator_1,
                    probe_generator_2=generator_2,
                )
                loss.backward()
                assert_excluded_gradients_zero(excluded)
                gradient = _flatten_active_gradients(active)
                scalar = float(loss.detach().cpu().item())
                if not math.isfinite(scalar) or not bool(torch.isfinite(gradient).all().item()):
                    raise FloatingPointError(
                        f"nonfinite result B={batch_size} repeat={prepared.repeat} "
                        f"state={state_position} pair={pair_position}"
                    )
                if gradient_sum is None:
                    gradient_sum = gradient.clone()
                else:
                    gradient_sum.add_(gradient)
                scalar_sum += scalar
                mode = "full" if str(stats["batch_1_hash"]) == "full" else "window"
                effective = int(stats["batch_size_effective"])
                row = {
                    "requested_batch_size": int(batch_size),
                    "effective_batch_size": effective,
                    "batch_mode": mode,
                    "repeat": int(prepared.repeat),
                    "step_key": int(prepared.step_key),
                    "state_position": int(state_position),
                    "pair_position": int(pair_position),
                    "source_weight_index": int(prepared.state_indices[state_position]),
                    "train_position": int(prepared.train_positions[state_position]),
                    "state_draw_seed": int(prepared.state_draw_seeds[state_position]),
                    "latent_hash": prepared.latent_hashes[state_position],
                    "task_name": str(record.get("task_name", "")),
                    "tau": float(record.get("tau", 1.0)),
                    "probe_seed_1": int(probe_seed_1),
                    "probe_seed_2": int(probe_seed_2),
                    "a_scalar": scalar,
                    "gradient_norm": vector_norm(gradient),
                    "h1_norm": float(stats["h1_norm"]),
                    "h2_norm": float(stats["h2_norm"]),
                    "h1_h2_dot": float(stats["h1_h2_dot"]),
                    "probe_1_hash": str(stats["probe_1_hash"]),
                    "probe_2_hash": str(stats["probe_2_hash"]),
                    "h1_hash": str(stats["h1_hash"]),
                    "h2_hash": str(stats["h2_hash"]),
                    "batch_1_hash": str(stats["batch_1_hash"]),
                    "batch_2_hash": str(stats["batch_2_hash"]),
                    "batch_1_offset": stats["batch_1_offset"],
                    "batch_2_offset": stats["batch_2_offset"],
                    "within_pair_batch_overlap_count": float(stats["branch_batch_overlap"]),
                    "unit_elapsed_sec": float(time.perf_counter() - unit_started),
                    **cuda_memory(device),
                }
                atomic_rows.append(row)
                for branch in (1, 2):
                    raw_offset = stats[f"batch_{branch}_offset"]
                    branch_rows.append(
                        {
                            "pair_position": int(pair_position),
                            "branch": int(branch - 1),
                            "offset": None if not math.isfinite(float(raw_offset)) else int(raw_offset),
                            "batch_hash": str(stats[f"batch_{branch}_hash"]),
                            "train_count": int(stats["batch_train_count"]),
                            "effective_batch_size": effective,
                            "batch_mode": mode,
                        }
                    )
                del gradient, loss
            coverage, overlaps = window_diagnostics(
                branch_rows,
                requested_batch_size=int(batch_size),
                repeat=prepared.repeat,
                state_position=state_position,
                source_weight_index=prepared.state_indices[state_position],
                task_name=str(record.get("task_name", "")),
                step_key=prepared.step_key,
            )
            coverage_rows.append(coverage)
            overlap_rows.extend(overlaps)
        if gradient_sum is None:
            raise RuntimeError("empty gradient sum")
        gradient_sum.div_(float(TRAIN_STATES * TRAIN_PAIRS))
        gradients.append(gradient_sum)
        scalars.append(float(scalar_sum / float(TRAIN_STATES * TRAIN_PAIRS)))
        print(
            "[hvp_batch] repeat_done "
            f"B={batch_size} repeat={prepared.repeat + 1}/{len(prepared_repeats)} "
            f"A={scalars[-1]:.7g} grad_norm={vector_norm(gradient_sum):.7g} "
            f"elapsed_sec={time.perf_counter() - repeat_started:.1f} "
            f"total_sec={time.perf_counter() - started:.1f}",
            flush=True,
        )
    effective = TRAIN_COUNT if int(batch_size) >= TRAIN_COUNT else int(batch_size)
    return BatchResult(
        requested_batch_size=int(batch_size),
        effective_batch_size=effective,
        batch_mode="full" if int(batch_size) >= TRAIN_COUNT else "window",
        gradients=gradients,
        scalars=scalars,
        atomic_rows=atomic_rows,
        coverage_rows=coverage_rows,
        overlap_rows=overlap_rows,
        elapsed_sec=float(time.perf_counter() - started),
    )


def build_module_grams(
    vectors: Sequence[torch.Tensor],
    module_ranges: Mapping[str, Sequence[tuple[int, int]]],
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = len(vectors)
    module_grams: dict[str, np.ndarray] = {}
    for module, ranges in module_ranges.items():
        gram = torch.zeros((count, count), dtype=torch.float64, device="cpu")
        module_started = time.perf_counter()
        for range_start, range_end in ranges:
            for start in range(int(range_start), int(range_end), int(chunk_size)):
                end = min(int(range_end), start + int(chunk_size))
                block = torch.stack([vector[start:end] for vector in vectors], dim=0)
                block64 = block.to(device=device, dtype=torch.float64)
                gram.add_((block64 @ block64.transpose(0, 1)).detach().cpu())
                del block, block64
        module_grams[module] = gram.numpy()
        print(
            f"[hvp_batch] gram_module module={module} elapsed_sec={time.perf_counter() - module_started:.1f}",
            flush=True,
        )
    full = np.zeros((count, count), dtype=np.float64)
    for gram in module_grams.values():
        full += gram
    return full, module_grams


def gram_pair_metrics(gram: np.ndarray, left: int, right: int) -> tuple[float, float, float]:
    left_norm2 = max(float(gram[left, left]), 0.0)
    right_norm2 = max(float(gram[right, right]), 0.0)
    denominator = math.sqrt(left_norm2 * right_norm2)
    cosine = float(gram[left, right] / denominator) if denominator > 0.0 else float("nan")
    norm_ratio = math.sqrt(left_norm2) / max(math.sqrt(right_norm2), 1e-30)
    error2 = max(left_norm2 + right_norm2 - 2.0 * float(gram[left, right]), 0.0)
    relative_error = math.sqrt(error2) / max(math.sqrt(right_norm2), 1e-30)
    return cosine, relative_error, norm_ratio


def gram_group_metrics(
    gram: np.ndarray,
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[float, float, float]:
    left_idx = np.asarray(left, dtype=np.int64)
    right_idx = np.asarray(right, dtype=np.int64)
    left_norm2 = float(gram[np.ix_(left_idx, left_idx)].mean())
    right_norm2 = float(gram[np.ix_(right_idx, right_idx)].mean())
    dot = float(gram[np.ix_(left_idx, right_idx)].mean())
    denominator = math.sqrt(max(left_norm2, 0.0) * max(right_norm2, 0.0))
    cosine = dot / denominator if denominator > 0.0 else float("nan")
    norm_ratio = math.sqrt(max(left_norm2, 0.0)) / max(math.sqrt(max(right_norm2, 0.0)), 1e-30)
    error2 = max(left_norm2 + right_norm2 - 2.0 * dot, 0.0)
    relative_error = math.sqrt(error2) / max(math.sqrt(max(right_norm2, 0.0)), 1e-30)
    return float(cosine), float(relative_error), float(norm_ratio)


def top_mass_share(values: Sequence[float], fraction: float) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    if absolute.size == 0 or float(absolute.sum()) <= 0.0:
        return float("nan")
    count = max(1, int(math.ceil(float(fraction) * int(absolute.size))))
    return float(np.sort(absolute)[-count:].sum() / absolute.sum())


def bootstrap_mean_ci(values: Sequence[float], *, seed: int, draws: int = 20000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    positions = generator.integers(0, len(array), size=(int(draws), len(array)))
    means = array[positions].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def analyze(
    results: Mapping[int, BatchResult],
    *,
    full_gram: np.ndarray,
    module_grams: Mapping[str, np.ndarray],
    gradient_index: pd.DataFrame,
    reference_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    index_by_key = {
        (int(row.requested_batch_size), int(row.repeat)): int(row.gradient_index)
        for row in gradient_index.itertuples()
    }
    atomic = pd.DataFrame([row for result in results.values() for row in result.atomic_rows])
    coverage = pd.DataFrame([row for result in results.values() for row in result.coverage_rows])
    overlap = pd.DataFrame([row for result in results.values() for row in result.overlap_rows])
    update_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    repeats = len(results[reference_batch_size].gradients)
    for batch_size, result in sorted(results.items()):
        for repeat in range(repeats):
            candidate = index_by_key[(batch_size, repeat)]
            reference = index_by_key[(reference_batch_size, repeat)]
            cosine, relative_error, norm_ratio = gram_pair_metrics(full_gram, candidate, reference)
            update_rows.append(
                {
                    "requested_batch_size": batch_size,
                    "repeat": repeat,
                    "reference_batch_size": reference_batch_size,
                    "paired_gradient_cosine": cosine,
                    "paired_gradient_relative_error": relative_error,
                    "paired_gradient_norm_ratio": norm_ratio,
                    "a_scalar": result.scalars[repeat],
                    "reference_a_scalar": results[reference_batch_size].scalars[repeat],
                    "paired_scalar_symmetric_error": symmetric_relative_error(
                        result.scalars[repeat], results[reference_batch_size].scalars[repeat]
                    ),
                }
            )
            for module, gram in module_grams.items():
                module_cosine, module_error, module_ratio = gram_pair_metrics(gram, candidate, reference)
                module_rows.append(
                    {
                        "requested_batch_size": batch_size,
                        "repeat": repeat,
                        "module": module,
                        "paired_gradient_cosine": module_cosine,
                        "paired_gradient_relative_error": module_error,
                        "paired_gradient_norm_ratio": module_ratio,
                    }
                )
        for left in range(repeats):
            for right in range(left + 1, repeats):
                left_index = index_by_key[(batch_size, left)]
                right_index = index_by_key[(batch_size, right)]
                cosine, relative_error, norm_ratio = gram_pair_metrics(full_gram, left_index, right_index)
                pairwise_rows.append(
                    {
                        "requested_batch_size": batch_size,
                        "left_repeat": left,
                        "right_repeat": right,
                        "gradient_cosine": cosine,
                        "gradient_relative_error": relative_error,
                        "gradient_norm_ratio": norm_ratio,
                    }
                )
    updates = pd.DataFrame(update_rows)
    modules = pd.DataFrame(module_rows)
    pairwise = pd.DataFrame(pairwise_rows)

    reference_split_rows: list[dict[str, Any]] = []
    for split, (left_repeats, right_repeats) in enumerate(_fixed_reference_splits(repeats)):
        left_indices = [index_by_key[(reference_batch_size, repeat)] for repeat in left_repeats]
        right_indices = [index_by_key[(reference_batch_size, repeat)] for repeat in right_repeats]
        cosine, relative_error, norm_ratio = gram_group_metrics(full_gram, left_indices, right_indices)
        scalar_left = float(np.mean([results[reference_batch_size].scalars[index] for index in left_repeats]))
        scalar_right = float(np.mean([results[reference_batch_size].scalars[index] for index in right_repeats]))
        reference_split_rows.append(
            {
                "split": split,
                "left_repeats": json.dumps(left_repeats),
                "right_repeats": json.dumps(right_repeats),
                "gradient_cosine": cosine,
                "gradient_relative_error": relative_error,
                "gradient_norm_ratio": norm_ratio,
                "scalar_left": scalar_left,
                "scalar_right": scalar_right,
                "scalar_symmetric_error": symmetric_relative_error(scalar_left, scalar_right),
            }
        )
    reference_splits = pd.DataFrame(reference_split_rows)
    reference_reproducible = bool(
        (reference_splits["gradient_cosine"] >= 0.99).all()
        and reference_splits["gradient_norm_ratio"].between(0.95, 1.05).all()
        and (reference_splits["scalar_symmetric_error"] <= 0.05).all()
    )

    reference_atomic = atomic.loc[
        atomic["requested_batch_size"] == reference_batch_size,
        ["repeat", "state_position", "pair_position", "a_scalar"],
    ].rename(columns={"a_scalar": "reference_atomic_scalar"})
    atomic_paired = atomic.merge(
        reference_atomic,
        on=["repeat", "state_position", "pair_position"],
        how="left",
        validate="many_to_one",
    )
    atomic_paired["paired_scalar_symmetric_error"] = [
        symmetric_relative_error(left, right)
        for left, right in zip(
            atomic_paired["a_scalar"], atomic_paired["reference_atomic_scalar"], strict=True
        )
    ]

    summary_rows: list[dict[str, Any]] = []
    for batch_size, result in sorted(results.items()):
        update_group = updates.loc[updates["requested_batch_size"] == batch_size]
        pair_group = pairwise.loc[pairwise["requested_batch_size"] == batch_size]
        atomic_group = atomic_paired.loc[atomic_paired["requested_batch_size"] == batch_size]
        coverage_group = coverage.loc[coverage["requested_batch_size"] == batch_size]
        scalar_values = atomic_group["a_scalar"].to_numpy(dtype=np.float64)
        gradient_norms = atomic_group["gradient_norm"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "requested_batch_size": batch_size,
                "effective_batch_size": result.effective_batch_size,
                "batch_mode": result.batch_mode,
                "reference_batch_size": reference_batch_size,
                "paired_gradient_cosine_median": quantile(update_group["paired_gradient_cosine"], 0.5),
                "paired_gradient_cosine_q10": quantile(update_group["paired_gradient_cosine"], 0.1),
                "paired_gradient_cosine_q90": quantile(update_group["paired_gradient_cosine"], 0.9),
                "paired_gradient_relative_error_median": quantile(
                    update_group["paired_gradient_relative_error"], 0.5
                ),
                "paired_gradient_relative_error_q90": quantile(
                    update_group["paired_gradient_relative_error"], 0.9
                ),
                "paired_gradient_norm_ratio_median": quantile(update_group["paired_gradient_norm_ratio"], 0.5),
                "paired_scalar_error_median": quantile(update_group["paired_scalar_symmetric_error"], 0.5),
                "paired_scalar_error_q90": quantile(update_group["paired_scalar_symmetric_error"], 0.9),
                "cross_repeat_cosine_median": quantile(pair_group["gradient_cosine"], 0.5),
                "cross_repeat_cosine_q10": quantile(pair_group["gradient_cosine"], 0.1),
                "cross_repeat_negative_fraction": float((pair_group["gradient_cosine"] < 0.0).mean()),
                "atomic_scalar_mean": float(np.mean(scalar_values)),
                "atomic_scalar_median": float(np.median(scalar_values)),
                "atomic_scalar_q90": float(np.quantile(scalar_values, 0.90)),
                "atomic_scalar_q99": float(np.quantile(scalar_values, 0.99)),
                "atomic_scalar_max": float(np.max(scalar_values)),
                "atomic_scalar_top1pct_absolute_mass_share": top_mass_share(scalar_values, 0.01),
                "atomic_gradient_norm_median": float(np.median(gradient_norms)),
                "atomic_gradient_norm_q90": float(np.quantile(gradient_norms, 0.90)),
                "atomic_gradient_norm_q99": float(np.quantile(gradient_norms, 0.99)),
                "atomic_gradient_norm_max": float(np.max(gradient_norms)),
                "atomic_gradient_top1pct_norm_mass_share": top_mass_share(gradient_norms, 0.01),
                "atomic_paired_scalar_error_median": quantile(
                    atomic_group["paired_scalar_symmetric_error"], 0.5
                ),
                "atomic_scalar_pearson_to_reference": float(
                    atomic_group["a_scalar"].corr(atomic_group["reference_atomic_scalar"], method="pearson")
                ),
                "atomic_scalar_spearman_to_reference": float(
                    atomic_group["a_scalar"].corr(atomic_group["reference_atomic_scalar"], method="spearman")
                ),
                "window_union_fraction_mean": float(coverage_group["union_fraction"].mean()),
                "window_max_multiplicity_max": int(coverage_group["max_multiplicity"].max()),
                "window_fraction_multiplicity_ge2_mean": float(
                    coverage_group["fraction_multiplicity_ge2"].mean()
                ),
                "unit_elapsed_sec_median": float(atomic_group["unit_elapsed_sec"].median()),
                "unit_elapsed_sec_q90": float(atomic_group["unit_elapsed_sec"].quantile(0.90)),
                "cuda_peak_allocated_bytes_max": int(atomic_group["cuda_peak_allocated_bytes"].max()),
                "cuda_peak_reserved_bytes_max": int(atomic_group["cuda_peak_reserved_bytes"].max()),
                "batch_elapsed_sec": result.elapsed_sec,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("requested_batch_size").reset_index(drop=True)
    task_summary = (
        atomic.groupby(["requested_batch_size", "task_name"], as_index=False)
        .agg(
            atomic_count=("a_scalar", "size"),
            atomic_scalar_mean=("a_scalar", "mean"),
            atomic_scalar_median=("a_scalar", "median"),
            atomic_scalar_q90=("a_scalar", lambda values: float(np.quantile(values, 0.90))),
            atomic_scalar_q99=("a_scalar", lambda values: float(np.quantile(values, 0.99))),
            atomic_gradient_norm_mean=("gradient_norm", "mean"),
            atomic_gradient_norm_q90=("gradient_norm", lambda values: float(np.quantile(values, 0.90))),
            atomic_gradient_norm_q99=("gradient_norm", lambda values: float(np.quantile(values, 0.99))),
        )
        .sort_values(["requested_batch_size", "task_name"])
        .reset_index(drop=True)
    )

    endpoint_rows: list[dict[str, Any]] = []
    baseline = 128 if 128 in results else min(results)
    for endpoint in [value for value in (8192, reference_batch_size) if value in results and value != baseline]:
        base_group = updates.loc[updates["requested_batch_size"] == baseline].sort_values("repeat")
        end_group = updates.loc[updates["requested_batch_size"] == endpoint].sort_values("repeat")
        for metric in (
            "paired_gradient_cosine",
            "paired_gradient_relative_error",
            "paired_scalar_symmetric_error",
        ):
            deltas = end_group[metric].to_numpy(dtype=np.float64) - base_group[metric].to_numpy(dtype=np.float64)
            interval = bootstrap_mean_ci(
                deltas,
                seed=stable_uint63(PROTOCOL_ID, seed, "bootstrap", baseline, endpoint, metric),
            )
            endpoint_rows.append(
                {
                    "baseline_batch_size": baseline,
                    "endpoint_batch_size": endpoint,
                    "metric": metric,
                    "paired_delta_mean": interval["mean"],
                    "bootstrap_ci95_low": interval["ci95_low"],
                    "bootstrap_ci95_high": interval["ci95_high"],
                    "repeat_count": repeats,
                }
            )
    endpoints = pd.DataFrame(endpoint_rows)
    return {
        "atomic": atomic,
        "atomic_paired": atomic_paired,
        "coverage": coverage,
        "overlap": overlap,
        "updates": updates,
        "modules": modules,
        "pairwise": pairwise,
        "reference_splits": reference_splits,
        "reference_reproducible": reference_reproducible,
        "summary": summary,
        "task_summary": task_summary,
        "endpoints": endpoints,
    }


def write_plots(output_dir: Path, bundle: Mapping[str, Any], *, reference_batch_size: int) -> list[str]:
    summary: pd.DataFrame = bundle["summary"]
    finite = summary.loc[summary["batch_mode"] != "full"].copy()
    outputs: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    x = summary["requested_batch_size"].to_numpy(dtype=float)
    axes[0].plot(x, summary["paired_gradient_cosine_median"], marker="o")
    axes[0].fill_between(
        x,
        summary["paired_gradient_cosine_q10"],
        summary["paired_gradient_cosine_q90"],
        alpha=0.2,
    )
    axes[0].set_ylabel("paired cosine to full")
    axes[1].plot(x, summary["paired_gradient_relative_error_median"], marker="o", color="#b91c1c")
    axes[1].set_ylabel("paired relative gradient error")
    axes[2].plot(x, summary["paired_scalar_error_median"], marker="o", color="#047857")
    axes[2].set_ylabel("paired scalar symmetric error")
    for axis in axes:
        axis.set_xscale("log", base=2)
        labels = [
            "full" if str(mode) == "full" else str(int(value))
            for value, mode in zip(x, summary["batch_mode"], strict=True)
        ]
        axis.set_xticks(x, labels, rotation=35)
        axis.set_xlabel("CE batch size")
        axis.grid(alpha=0.25)
    fig.suptitle("Train-budget HVP update convergence to paired full-data objective")
    fig.tight_layout()
    path = output_dir / "paired_full_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].plot(x, summary["cross_repeat_cosine_median"], marker="o", label="median")
    axes[0].plot(x, summary["cross_repeat_cosine_q10"], marker="o", label="q10")
    axes[0].set_ylabel("cross-repeat gradient cosine")
    axes[0].legend()
    axes[1].plot(x, summary["atomic_scalar_q99"], marker="o", label="A q99")
    axes[1].plot(x, summary["atomic_scalar_max"], marker="o", label="A max")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("atomic raw A")
    axes[1].legend()
    axes[2].plot(x, summary["atomic_gradient_norm_q99"], marker="o", label="grad q99")
    axes[2].plot(x, summary["atomic_gradient_norm_max"], marker="o", label="grad max")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("atomic decoder-gradient norm")
    axes[2].legend()
    for axis in axes:
        axis.set_xscale("log", base=2)
        labels = [
            "full" if str(mode) == "full" else str(int(value))
            for value, mode in zip(x, summary["batch_mode"], strict=True)
        ]
        axis.set_xticks(x, labels, rotation=35)
        axis.set_xlabel("CE batch size")
        axis.grid(alpha=0.25)
    fig.suptitle("Residual update instability and atomic tails")
    fig.tight_layout()
    path = output_dir / "reproducibility_and_tails.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    finite_x = finite["requested_batch_size"].to_numpy(dtype=float)
    axes[0].plot(finite_x, finite["window_union_fraction_mean"], marker="o")
    axes[0].set_ylabel("8-window union fraction")
    axes[1].plot(
        x,
        summary["cuda_peak_allocated_bytes_max"] / (1024.0**3),
        marker="o",
        label="allocated",
    )
    axes[1].plot(
        x,
        summary["cuda_peak_reserved_bytes_max"] / (1024.0**3),
        marker="o",
        label="reserved",
    )
    axes[1].set_ylabel("sequential peak CUDA GiB")
    axes[1].legend()
    axes[2].plot(x, summary["unit_elapsed_sec_median"], marker="o", label="median")
    axes[2].plot(x, summary["unit_elapsed_sec_q90"], marker="o", label="q90")
    axes[2].set_ylabel("seconds per atomic pair")
    axes[2].legend()
    for axis in axes:
        axis.set_xscale("log", base=2)
        ticks = finite_x if axis is axes[0] else x
        axis.set_xticks(ticks, [str(int(value)) for value in ticks], rotation=35)
        axis.set_xlabel("CE batch size")
        axis.grid(alpha=0.25)
    fig.suptitle("Window reuse and sequential audit cost")
    fig.tight_layout()
    path = output_dir / "window_coverage_and_cost.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired HVP CE-batch ablation at the accepted h=2048 VAE.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--gram-chunk-size", type=int, default=65536)
    parser.add_argument("--skip-simultaneous-memory-smoke", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    batch_sizes = parse_grid(args.batch_sizes)
    if int(args.repeats) != 12:
        raise ValueError("the frozen production protocol requires exactly 12 repeats")
    if batch_sizes[0] != 128 or 8192 not in batch_sizes:
        raise ValueError("the frozen grid must start at 128 and include 8192")
    if 16384 not in batch_sizes:
        raise ValueError("the frozen grid must attempt the full 16384 arm")

    checkpoint_path = run_dir / "vae_checkpoint.pt"
    source_hashes = {
        "checkpoint": sha256_file(checkpoint_path),
        "weight_pool": sha256_file(run_dir / "weight_pool.pt"),
        "records": sha256_file(run_dir / "weight_pool_records.csv"),
    }
    expected_hashes = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "weight_pool": EXPECTED_POOL_SHA256,
        "records": EXPECTED_RECORDS_SHA256,
    }
    if source_hashes != expected_hashes:
        raise ValueError(f"accepted source hash mismatch: observed={source_hashes} expected={expected_hashes}")
    acceptance = json.loads((run_dir / "baseline_acceptance.json").read_text(encoding="utf-8"))
    if not bool(acceptance.get("passed")):
        raise ValueError("baseline acceptance is not passed")
    code_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/audit_variant_a_estimator_stability.py",
        ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py",
        ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py",
    ]
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": "float32",
        "seed": int(args.seed),
        "states": TRAIN_STATES,
        "pairs": TRAIN_PAIRS,
        "repeats": int(args.repeats),
        "step_keys": [10 * (repeat + 1) for repeat in range(int(args.repeats))],
        "batch_sizes": batch_sizes,
        "train_count_required": TRAIN_COUNT,
        "estimator_scope": "local",
        "hvp_mode": "stopped_composite",
        "probe_scale": 1.0,
        "loss_clip": 0.0,
        "gradient_damping": 0.0,
        "source_hashes": source_hashes,
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "cache_mode": "read_only_checkpoint_and_pool",
        "repository": git_metadata(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[hvp_batch] start resolved_config={json.dumps(resolved, sort_keys=True)}", flush=True)
    started = time.perf_counter()

    print("[hvp_batch] stage=data_and_model_load", flush=True)
    run = _load_run(run_dir, device=device)
    if int(run.cfg.vae_hidden_dim) != 2048 or int(run.cfg.latent_dim) != 512:
        raise ValueError(
            f"unexpected architecture hidden={run.cfg.vae_hidden_dim} latent={run.cfg.latent_dim}"
        )
    if int(run.train_indices.numel()) != 11021:
        raise ValueError(f"unexpected VAE train population: {int(run.train_indices.numel())}")

    print("[hvp_batch] stage=atomic_preflight", flush=True)
    preflight, active = _preflight_decomposition(
        run,
        batch_size=128,
        seed=stable_uint63(PROTOCOL_ID, args.seed, "preflight"),
        device=device,
    )
    active_names = {name for name, _parameter in active}
    excluded = [(name, parameter) for name, parameter in run.vae.named_parameters() if name not in active_names]
    module_ranges, parameter_rows = _module_slices(active)
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(parameter_rows).to_csv(output_dir / "active_parameters.csv", index=False)

    print("[hvp_batch] stage=prepare_paired_repeat_bank", flush=True)
    prepared = prepare_repeats(
        run,
        repeats=int(args.repeats),
        seed=int(args.seed),
        device=device,
    )
    state_rows = []
    for item in prepared:
        for state_position in range(TRAIN_STATES):
            state_rows.append(
                {
                    "repeat": item.repeat,
                    "step_key": item.step_key,
                    "state_position": state_position,
                    "source_weight_index": item.state_indices[state_position],
                    "train_position": item.train_positions[state_position],
                    "state_draw_seed": item.state_draw_seeds[state_position],
                    "latent_hash": item.latent_hashes[state_position],
                    "task_name": str(item.records[state_position].get("task_name", "")),
                    "tau": float(item.records[state_position].get("tau", 1.0)),
                }
            )
    pd.DataFrame(state_rows).to_csv(output_dir / "sampled_states.csv", index=False)

    print("[hvp_batch] stage=sequential_memory_smoke", flush=True)
    sequential_smoke = sequential_memory_smoke(
        run,
        prepared[0],
        batch_sizes=batch_sizes,
        active=active,
        excluded=excluded,
        seed=int(args.seed),
        device=device,
    )
    sequential_smoke.to_csv(output_dir / "sequential_memory_smoke.csv", index=False)
    feasible = sequential_smoke.loc[sequential_smoke["success"], "requested_batch_size"].astype(int).tolist()
    if 8192 not in feasible:
        raise RuntimeError(f"sequential audit cannot reach required B=8192; feasible={feasible}")

    simultaneous_smoke = pd.DataFrame()
    if not bool(args.skip_simultaneous_memory_smoke):
        print("[hvp_batch] stage=simultaneous_a_memory_smoke", flush=True)
        simultaneous_smoke = simultaneous_a_memory_smoke(
            run,
            prepared[0],
            batch_sizes=batch_sizes,
            active=active,
            excluded=excluded,
            seed=int(args.seed),
            device=device,
        )
        simultaneous_smoke.to_csv(output_dir / "simultaneous_a_memory_smoke.csv", index=False)

    print(f"[hvp_batch] stage=production feasible={feasible}", flush=True)
    results: dict[int, BatchResult] = {}
    production_failures: list[dict[str, Any]] = []
    for batch_size in sorted(feasible, reverse=True):
        print(f"[hvp_batch] batch_start B={batch_size}", flush=True)
        try:
            result = run_batch(
                run,
                prepared,
                batch_size=int(batch_size),
                active=active,
                excluded=excluded,
                seed=int(args.seed),
                device=device,
            )
            results[int(batch_size)] = result
            print(
                f"[hvp_batch] batch_done B={batch_size} elapsed_sec={result.elapsed_sec:.1f}",
                flush=True,
            )
        except torch.OutOfMemoryError as error:
            production_failures.append(
                {
                    "requested_batch_size": int(batch_size),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            run.vae.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pd.DataFrame(production_failures).to_csv(output_dir / "production_failures.csv", index=False)
        pd.DataFrame([row for value in results.values() for row in value.atomic_rows]).to_csv(
            output_dir / "atomic_pair_samples.partial.csv", index=False
        )
    if 8192 not in results:
        raise RuntimeError(f"production did not complete B=8192; completed={sorted(results)}")
    reference_batch_size = 16384 if 16384 in results else max(results)

    print("[hvp_batch] stage=gradient_sufficient_statistics", flush=True)
    vectors: list[torch.Tensor] = []
    gradient_rows: list[dict[str, Any]] = []
    for batch_size, result in sorted(results.items()):
        for repeat, gradient in enumerate(result.gradients):
            gradient_index = len(vectors)
            vectors.append(gradient)
            gradient_rows.append(
                {
                    "gradient_index": gradient_index,
                    "requested_batch_size": int(batch_size),
                    "effective_batch_size": result.effective_batch_size,
                    "batch_mode": result.batch_mode,
                    "repeat": repeat,
                    "step_key": 10 * (repeat + 1),
                    "gradient_norm": vector_norm(gradient),
                    "gradient_sha256": sha256_tensor(gradient),
                    "a_scalar": result.scalars[repeat],
                }
            )
    gradient_index = pd.DataFrame(gradient_rows)
    gradient_index.to_csv(output_dir / "gradient_index.csv", index=False)
    full_gram, module_grams = build_module_grams(
        vectors,
        module_ranges,
        device=device,
        chunk_size=int(args.gram_chunk_size),
    )
    np.save(output_dir / "gradient_gram.npy", full_gram)
    np.savez_compressed(output_dir / "module_gradient_grams.npz", **module_grams)
    del vectors
    gc.collect()

    print("[hvp_batch] stage=paired_analysis", flush=True)
    bundle = analyze(
        results,
        full_gram=full_gram,
        module_grams=module_grams,
        gradient_index=gradient_index,
        reference_batch_size=reference_batch_size,
        seed=int(args.seed),
    )
    frame_paths = {
        "atomic_pair_samples.csv": bundle["atomic"],
        "atomic_paired_scalars.csv": bundle["atomic_paired"],
        "window_coverage.csv": bundle["coverage"],
        "window_overlap.csv": bundle["overlap"],
        "paired_update_metrics.csv": bundle["updates"],
        "module_paired_update_metrics.csv": bundle["modules"],
        "cross_repeat_gradient_metrics.csv": bundle["pairwise"],
        "reference_split_checks.csv": bundle["reference_splits"],
        "batch_summary.csv": bundle["summary"],
        "task_tail_summary.csv": bundle["task_summary"],
        "endpoint_bootstrap.csv": bundle["endpoints"],
    }
    for filename, frame in frame_paths.items():
        frame.to_csv(output_dir / filename, index=False)
    partial = output_dir / "atomic_pair_samples.partial.csv"
    if partial.exists():
        partial.unlink()

    print("[hvp_batch] stage=validity_and_plots", flush=True)
    atomic: pd.DataFrame = bundle["atomic"]
    expected_rows = len(results) * int(args.repeats) * TRAIN_STATES * TRAIN_PAIRS
    identity_columns = [
        "source_weight_index",
        "latent_hash",
        "probe_seed_1",
        "probe_seed_2",
        "probe_1_hash",
        "probe_2_hash",
    ]
    identity_stable = bool(
        (
            atomic.groupby(["repeat", "state_position", "pair_position"])[identity_columns]
            .nunique()
            .to_numpy()
            == 1
        ).all()
    )
    gram_diagonal = np.sqrt(np.maximum(np.diag(full_gram), 0.0))
    logged_norms = gradient_index["gradient_norm"].to_numpy(dtype=np.float64)
    checks = {
        "source_hashes_match": source_hashes == expected_hashes,
        "baseline_acceptance_passed": bool(acceptance.get("passed")),
        "atomic_preflight_passed": bool(
            preflight["gradient_cosine"] >= 0.99999
            and preflight["gradient_relative_error"] <= 5e-5
            and preflight["stopped_vs_autograd_hvp_cosine"] >= 0.99999
        ),
        "required_8192_completed": 8192 in results,
        "full_attempted": 16384 in batch_sizes,
        "atomic_row_count_match": len(atomic) == expected_rows,
        "atomic_keys_unique": not bool(
            atomic.duplicated(["requested_batch_size", "repeat", "state_position", "pair_position"]).any()
        ),
        "paired_identity_stable_across_batches": identity_stable,
        "train_step_keys_match": set(atomic["step_key"].astype(int))
        == {10 * (repeat + 1) for repeat in range(int(args.repeats))},
        "all_effective_batches_valid": bool(
            (
                atomic["effective_batch_size"]
                == atomic["requested_batch_size"].clip(upper=TRAIN_COUNT)
            ).all()
        ),
        "all_atomic_metrics_finite": bool(
            np.isfinite(
                atomic[
                    [
                        "a_scalar",
                        "gradient_norm",
                        "h1_norm",
                        "h2_norm",
                        "unit_elapsed_sec",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "gram_finite": bool(np.isfinite(full_gram).all()),
        "gram_symmetric": bool(np.allclose(full_gram, full_gram.T, rtol=1e-10, atol=1e-6)),
        "gram_norms_match_logged": bool(np.allclose(gram_diagonal, logged_norms, rtol=2e-6, atol=2e-6)),
        "checkpoint_unchanged": sha256_file(checkpoint_path) == source_hashes["checkpoint"],
    }
    validity = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "completed_batch_sizes": sorted(results),
        "reference_batch_size": reference_batch_size,
        "reference_is_full": reference_batch_size == TRAIN_COUNT,
        "reference_reproducible": bool(bundle["reference_reproducible"]),
        "expected_atomic_rows": expected_rows,
        "observed_atomic_rows": len(atomic),
        "production_failures": production_failures,
    }
    (output_dir / "validity.json").write_text(
        json.dumps(validity, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not validity["passed"]:
        raise RuntimeError(f"validity checks failed: {json.dumps(validity, sort_keys=True)}")
    plots = write_plots(output_dir, bundle, reference_batch_size=reference_batch_size)

    manifest = {
        "status": "complete",
        "elapsed_sec": float(time.perf_counter() - started),
        "resolved_config": resolved,
        "completed_batch_sizes": sorted(results),
        "reference_batch_size": reference_batch_size,
        "reference_is_full": reference_batch_size == TRAIN_COUNT,
        "reference_reproducible": bool(bundle["reference_reproducible"]),
        "validity": validity,
        "sequential_memory_smoke": sequential_smoke.to_dict(orient="records"),
        "simultaneous_a_memory_smoke": simultaneous_smoke.to_dict(orient="records"),
        "outputs": {
            "batch_summary": str(output_dir / "batch_summary.csv"),
            "paired_update_metrics": str(output_dir / "paired_update_metrics.csv"),
            "atomic_pair_samples": str(output_dir / "atomic_pair_samples.csv"),
            "gradient_index": str(output_dir / "gradient_index.csv"),
            "gradient_gram": str(output_dir / "gradient_gram.npy"),
            "module_gradient_grams": str(output_dir / "module_gradient_grams.npz"),
            "window_coverage": str(output_dir / "window_coverage.csv"),
            "window_overlap": str(output_dir / "window_overlap.csv"),
            "task_tail_summary": str(output_dir / "task_tail_summary.csv"),
            "reference_split_checks": str(output_dir / "reference_split_checks.csv"),
            "endpoint_bootstrap": str(output_dir / "endpoint_bootstrap.csv"),
            "validity": str(output_dir / "validity.json"),
            "plots": plots,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[hvp_batch] done "
        f"elapsed_sec={manifest['elapsed_sec']:.1f} completed={sorted(results)} "
        f"reference={reference_batch_size} reference_reproducible={bundle['reference_reproducible']} "
        f"summary={output_dir / 'batch_summary.csv'} manifest={output_dir / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
