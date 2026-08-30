from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_POOL_SHA256,
    EXPECTED_RECORDS_SHA256,
    LoadedRun,
    RepeatAccumulator,
    _analyze,
    _batch_pair_overlap_audit,
    _fixed_reference_splits,
    _flatten_active_gradients,
    _load_run,
    _mean_vectors,
    _module_slices,
    _probe_cfg,
    _slice_vector,
    _task_set_for_record,
    _write_plots,
    grid_bin,
    parse_grid,
    prefix_sum_scalar,
    prefix_sum_vector,
    quantile,
    sha256_file,
    stable_uint63,
    symmetric_relative_error,
    vector_cosine,
    vector_norm,
    vector_relative_error,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import decode_weights, encode_weights
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _latent_hvp,
    _probe_like,
    _task_loss_from_flat,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_fd_estimator_stability_unfinetuned_h2048"
)
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.md"
PROTOCOL_ID = "a_fd_estimator_stability_unfinetuned_h2048_v1"
SEED_PROTOCOL_ID = "a_estimator_stability_unfinetuned_h2048_v2"
DEFAULT_SIGMAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _indexed_seed(base_seed: int, kind: str, *indices: int) -> int:
    # Match the production HVP audit exactly so channel comparisons are paired.
    return stable_uint63(SEED_PROTOCOL_ID, int(base_seed), str(kind), *[int(value) for value in indices])


def _sample_states(train_indices: torch.Tensor, *, repeat: int, count: int, base_seed: int) -> tuple[list[int], list[int], list[int]]:
    source_indices: list[int] = []
    train_positions: list[int] = []
    draw_seeds: list[int] = []
    for state_position in range(int(count)):
        draw_seed = _indexed_seed(base_seed, "state", repeat, state_position, -1, -1)
        generator = torch.Generator(device="cpu").manual_seed(draw_seed)
        train_position = int(torch.randint(0, int(train_indices.numel()), (1,), generator=generator).item())
        source_indices.append(int(train_indices[train_position].item()))
        train_positions.append(train_position)
        draw_seeds.append(draw_seed)
    return source_indices, train_positions, draw_seeds


def _probe(z: torch.Tensor, *, base_seed: int, repeat: int, state_position: int, pair_position: int, branch: int) -> tuple[torch.Tensor, int]:
    seed = _indexed_seed(base_seed, "probe", repeat, state_position, pair_position, branch)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return _probe_like(z, generator=generator, scale=1.0), seed


def _latent_gradient(
    run: LoadedRun,
    z: torch.Tensor,
    *,
    task_set: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    z_req = z.detach().clone().requires_grad_(True)
    decoded = decode_weights(run.vae, run.normalizer, z_req.reshape(1, -1)).squeeze(0)
    loss = _task_loss_from_flat(
        decoded,
        task_set=task_set,
        spec=run.spec,
        tau=float(tau),
        batch_indices=batch_indices,
    )
    return torch.autograd.grad(loss, z_req, create_graph=True, retain_graph=True)[0]


def _finite_difference_h(
    run: LoadedRun,
    z: torch.Tensor,
    probe: torch.Tensor,
    *,
    sigma: float,
    task_set: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    base = _latent_gradient(run, z, task_set=task_set, tau=tau, batch_indices=batch_indices)
    perturbed = _latent_gradient(
        run,
        z + float(sigma) * probe,
        task_set=task_set,
        tau=tau,
        batch_indices=batch_indices,
    )
    return (perturbed - base) / float(sigma)


def _a_scalar(h1: torch.Tensor, h2: torch.Tensor, *, dim: int) -> torch.Tensor:
    return ((h1 * h2).sum().square() - h1.square().sum() - h2.square().sum() + float(dim)) / float(dim)


def _pair_inputs(
    run: LoadedRun,
    z: torch.Tensor,
    record: Mapping[str, Any],
    *,
    repeat: int,
    state_position: int,
    pair_position: int,
    batch_size: int,
    base_seed: int,
) -> dict[str, Any]:
    task_set = _task_set_for_record(run.task_tensors, record)
    sample_key = int(record["source_weight_index"])
    tau = float(record.get("tau", 1.0))
    step = int(repeat) + 1
    batch_1 = _batch_indices(
        task_set,
        batch_size=int(batch_size),
        step=step,
        sample_key=sample_key,
        pair_key=2 * int(pair_position),
    )
    batch_2 = _batch_indices(
        task_set,
        batch_size=int(batch_size),
        step=step,
        sample_key=sample_key,
        pair_key=2 * int(pair_position) + 1,
    )
    probe_1, probe_seed_1 = _probe(
        z,
        base_seed=base_seed,
        repeat=repeat,
        state_position=state_position,
        pair_position=pair_position,
        branch=0,
    )
    probe_2, probe_seed_2 = _probe(
        z,
        base_seed=base_seed,
        repeat=repeat,
        state_position=state_position,
        pair_position=pair_position,
        branch=1,
    )
    return {
        "task_set": task_set,
        "tau": tau,
        "batch_1": batch_1,
        "batch_2": batch_2,
        "probe_1": probe_1,
        "probe_2": probe_2,
        "probe_seed_1": probe_seed_1,
        "probe_seed_2": probe_seed_2,
    }


def _fd_pair_loss(run: LoadedRun, z: torch.Tensor, inputs: Mapping[str, Any], *, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h1 = _finite_difference_h(
        run,
        z,
        inputs["probe_1"],
        sigma=sigma,
        task_set=inputs["task_set"],
        tau=float(inputs["tau"]),
        batch_indices=inputs["batch_1"],
    )
    h2 = _finite_difference_h(
        run,
        z,
        inputs["probe_2"],
        sigma=sigma,
        task_set=inputs["task_set"],
        tau=float(inputs["tau"]),
        batch_indices=inputs["batch_2"],
    )
    return _a_scalar(h1, h2, dim=int(z.numel())), h1, h2


def _hvp_pair_loss(run: LoadedRun, cfg: Any, z: torch.Tensor, inputs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h1 = _latent_hvp(
        cfg,
        run.vae,
        run.normalizer,
        z,
        inputs["probe_1"],
        task_set=inputs["task_set"],
        spec=run.spec,
        tau=float(inputs["tau"]),
        batch_indices=inputs["batch_1"],
    )
    h2 = _latent_hvp(
        cfg,
        run.vae,
        run.normalizer,
        z,
        inputs["probe_2"],
        task_set=inputs["task_set"],
        spec=run.spec,
        tau=float(inputs["tau"]),
        batch_indices=inputs["batch_2"],
    )
    return _a_scalar(h1, h2, dim=int(z.numel())), h1, h2


def _gradient_vector(
    run: LoadedRun,
    loss: torch.Tensor,
    active: Sequence[tuple[str, torch.nn.Parameter]] | None,
) -> tuple[torch.Tensor, list[tuple[str, torch.nn.Parameter]]]:
    run.vae.zero_grad(set_to_none=True)
    loss.backward()
    if active is None:
        active = [(name, parameter) for name, parameter in run.vae.named_parameters() if parameter.grad is not None]
    vector = _flatten_active_gradients(active)
    active_names = {name for name, _parameter in active}
    for name, parameter in run.vae.named_parameters():
        if name in active_names:
            continue
        if parameter.grad is not None and bool((parameter.grad.detach() != 0).any().item()):
            raise RuntimeError(f"excluded parameter received finite-difference A gradient: {name}")
    return vector, list(active)


def _prepare_states(
    run: LoadedRun,
    *,
    repeat: int,
    count: int,
    base_seed: int,
    device: torch.device,
) -> tuple[list[int], list[int], list[int], torch.Tensor, list[dict[str, Any]]]:
    indices, train_positions, draw_seeds = _sample_states(
        run.train_indices,
        repeat=repeat,
        count=count,
        base_seed=base_seed,
    )
    selected = run.weights[indices].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z_values = encode_weights(run.vae, run.normalizer, selected).detach()
    records = run.records.iloc[indices].to_dict(orient="records")
    for record, source_index in zip(records, indices, strict=True):
        record["source_weight_index"] = int(source_index)
    return indices, train_positions, draw_seeds, z_values, records


def _calibrate_sigma(
    run: LoadedRun,
    *,
    sigmas: Sequence[float],
    batch_sizes: Sequence[int],
    base_seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[float | None, list[tuple[str, torch.nn.Parameter]], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    active: list[tuple[str, torch.nn.Parameter]] | None = None
    hvp_cfg = replace(
        _probe_cfg(run.cfg, sample_count=1, pair_count=1, batch_size=max(batch_sizes)),
        vae_precond_hvp_mode="autograd",
    )
    total = 2 * 4 * 2 * len(batch_sizes) * len(sigmas)
    completed = 0
    started = time.perf_counter()
    for repeat in range(2):
        indices, _positions, _draw_seeds, z_values, records = _prepare_states(
            run,
            repeat=repeat,
            count=4,
            base_seed=base_seed,
            device=device,
        )
        for batch_size in batch_sizes:
            cfg = replace(hvp_cfg, vae_precond_batch_size=int(batch_size))
            for state_position in range(4):
                for pair_position in range(2):
                    inputs = _pair_inputs(
                        run,
                        z_values[state_position],
                        records[state_position],
                        repeat=repeat,
                        state_position=state_position,
                        pair_position=pair_position,
                        batch_size=int(batch_size),
                        base_seed=base_seed,
                    )
                    hvp_loss, exact_h1, exact_h2 = _hvp_pair_loss(run, cfg, z_values[state_position], inputs)
                    hvp_vector, active = _gradient_vector(run, hvp_loss, active)
                    hvp_scalar = float(hvp_loss.detach().cpu().item())
                    exact_h1_cpu = exact_h1.detach().cpu()
                    exact_h2_cpu = exact_h2.detach().cpu()
                    for sigma in sigmas:
                        fd_loss, fd_h1, fd_h2 = _fd_pair_loss(run, z_values[state_position], inputs, sigma=float(sigma))
                        fd_vector, active = _gradient_vector(run, fd_loss, active)
                        row = {
                            "repeat": repeat,
                            "batch_size": int(batch_size),
                            "state_position": state_position,
                            "pair_position": pair_position,
                            "source_weight_index": indices[state_position],
                            "sigma": float(sigma),
                            "h1_cosine": vector_cosine(fd_h1.detach().cpu(), exact_h1_cpu),
                            "h2_cosine": vector_cosine(fd_h2.detach().cpu(), exact_h2_cpu),
                            "h1_relative_error": vector_relative_error(fd_h1.detach().cpu(), exact_h1_cpu),
                            "h2_relative_error": vector_relative_error(fd_h2.detach().cpu(), exact_h2_cpu),
                            "hvp_a_scalar": hvp_scalar,
                            "fd_a_scalar": float(fd_loss.detach().cpu().item()),
                            "a_scalar_symmetric_error": symmetric_relative_error(
                                float(fd_loss.detach().cpu().item()), hvp_scalar
                            ),
                            "a_gradient_cosine": vector_cosine(fd_vector, hvp_vector),
                            "a_gradient_relative_error": vector_relative_error(fd_vector, hvp_vector),
                            "a_gradient_norm_ratio": vector_norm(fd_vector) / max(vector_norm(hvp_vector), 1e-30),
                        }
                        rows.append(row)
                        completed += 1
                        if completed == 1 or completed % 16 == 0 or completed == total:
                            elapsed = time.perf_counter() - started
                            rate = completed / max(elapsed, 1e-12)
                            print(
                                "[fd_stability] calibration "
                                f"row={completed}/{total} B={batch_size} sigma={sigma:g} "
                                f"h_cos={min(row['h1_cosine'], row['h2_cosine']):.6f} "
                                f"g_cos={row['a_gradient_cosine']:.6f} rate={rate:.2f}/s",
                                flush=True,
                            )
                        del fd_vector, fd_loss, fd_h1, fd_h2
                    del hvp_vector, hvp_loss, exact_h1, exact_h2
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "sigma_calibration_rows.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for (sigma, batch_size), group in frame.groupby(["sigma", "batch_size"], sort=True):
        h_cos = np.concatenate([group["h1_cosine"].to_numpy(), group["h2_cosine"].to_numpy()])
        h_rel = np.concatenate([group["h1_relative_error"].to_numpy(), group["h2_relative_error"].to_numpy()])
        row = {
            "sigma": float(sigma),
            "batch_size": int(batch_size),
            "hvp_cosine_median": float(np.quantile(h_cos, 0.5)),
            "hvp_cosine_q10": float(np.quantile(h_cos, 0.1)),
            "hvp_relative_error_median": float(np.quantile(h_rel, 0.5)),
            "hvp_relative_error_q90": float(np.quantile(h_rel, 0.9)),
            "a_gradient_cosine_median": float(group["a_gradient_cosine"].median()),
            "a_gradient_cosine_q10": float(group["a_gradient_cosine"].quantile(0.1)),
            "a_gradient_relative_error_q90": float(group["a_gradient_relative_error"].quantile(0.9)),
            "a_gradient_norm_ratio_median": float(group["a_gradient_norm_ratio"].median()),
            "a_scalar_symmetric_error_median": float(group["a_scalar_symmetric_error"].median()),
        }
        row["passed"] = bool(
            row["hvp_cosine_median"] >= 0.999
            and row["hvp_cosine_q10"] >= 0.99
            and row["hvp_relative_error_median"] <= 0.05
            and row["hvp_relative_error_q90"] <= 0.10
            and row["a_gradient_cosine_median"] >= 0.99
            and row["a_gradient_cosine_q10"] >= 0.95
            and 0.90 <= row["a_gradient_norm_ratio_median"] <= 1.10
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["sigma", "batch_size"])
    summary.to_csv(output_dir / "sigma_calibration_summary.csv", index=False)
    candidates: list[tuple[float, float]] = []
    for sigma, group in summary.groupby("sigma", sort=True):
        if len(group) == len(batch_sizes) and bool(group["passed"].all()):
            candidates.append((float(group["a_gradient_relative_error_q90"].max()), float(sigma)))
    selected = None if not candidates else min(candidates)[1]
    decision = {
        "selected_sigma": selected,
        "calibration_passed": selected is not None,
        "candidate_scores": [
            {"sigma": sigma, "worst_batch_a_gradient_relative_error_q90": score}
            for score, sigma in sorted(candidates)
        ],
    }
    (output_dir / "sigma_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    if active is None:
        raise RuntimeError("sigma calibration produced no active gradient")
    return selected, active, frame, summary


def _run_repeat_fd(
    run: LoadedRun,
    *,
    repeat: int,
    batch_size: int,
    sigma: float,
    state_grid: Sequence[int],
    pair_grid: Sequence[int],
    active: Sequence[tuple[str, torch.nn.Parameter]],
    base_seed: int,
    device: torch.device,
    unit_rows: list[dict[str, Any]],
    gradient_chunk_rows: list[dict[str, Any]],
    gradient_chunk_size: int,
    log_every: int,
) -> RepeatAccumulator:
    max_states = int(state_grid[-1])
    max_pairs = int(pair_grid[-1])
    indices, train_positions, draw_seeds, z_values, records = _prepare_states(
        run,
        repeat=repeat,
        count=max_states,
        base_seed=base_seed,
        device=device,
    )
    scalar_bins = np.zeros((len(state_grid), len(pair_grid)), dtype=np.float64)
    gradient_bins: list[list[torch.Tensor | None]] = [
        [None for _pair in pair_grid] for _state in state_grid
    ]
    total = max_states * max_pairs
    state_starts = [0, *[int(value) for value in state_grid[:-1]]]
    pair_starts = [0, *[int(value) for value in pair_grid[:-1]]]
    unit_index = 0
    started = time.perf_counter()
    for state_bin, (state_start, state_stop) in enumerate(zip(state_starts, state_grid, strict=True)):
        for pair_bin, (pair_start, pair_stop) in enumerate(zip(pair_starts, pair_grid, strict=True)):
            pending_losses: list[torch.Tensor] = []
            pending_scalars: list[float] = []
            pending_units: list[tuple[int, int]] = []
            chunk_index = 0

            def flush_gradient_chunk() -> None:
                nonlocal chunk_index
                if not pending_losses:
                    return
                chunk_loss = torch.stack(pending_losses).sum()
                gradient, _active = _gradient_vector(run, chunk_loss, active)
                if not bool(torch.isfinite(gradient).all().item()):
                    raise FloatingPointError(
                        f"nonfinite FD gradient chunk B={batch_size} repeat={repeat} "
                        f"state_bin={state_bin} pair_bin={pair_bin} chunk={chunk_index}"
                    )
                existing = gradient_bins[state_bin][pair_bin]
                if existing is None:
                    gradient_bins[state_bin][pair_bin] = gradient.clone()
                else:
                    existing.add_(gradient)
                gradient_chunk_rows.append(
                    {
                        "batch_size": batch_size,
                        "sigma": sigma,
                        "repeat": repeat,
                        "state_bin": state_bin,
                        "pair_bin": pair_bin,
                        "chunk_index": chunk_index,
                        "chunk_size": len(pending_losses),
                        "state_position_first": pending_units[0][0],
                        "state_position_last": pending_units[-1][0],
                        "pair_position_first": pending_units[0][1],
                        "pair_position_last": pending_units[-1][1],
                        "a_scalar_sum": float(sum(pending_scalars)),
                        "gradient_sum_norm": vector_norm(gradient),
                    }
                )
                chunk_index += 1
                pending_losses.clear()
                pending_scalars.clear()
                pending_units.clear()
                del gradient, chunk_loss

            for state_position in range(int(state_start), int(state_stop)):
                for pair_position in range(int(pair_start), int(pair_stop)):
                    inputs = _pair_inputs(
                        run,
                        z_values[state_position],
                        records[state_position],
                        repeat=repeat,
                        state_position=state_position,
                        pair_position=pair_position,
                        batch_size=batch_size,
                        base_seed=base_seed,
                    )
                    pair_started = time.perf_counter()
                    fd_loss, h1, h2 = _fd_pair_loss(run, z_values[state_position], inputs, sigma=sigma)
                    scalar = float(fd_loss.detach().cpu().item())
                    if not math.isfinite(scalar):
                        raise FloatingPointError(
                            f"nonfinite FD pair B={batch_size} repeat={repeat} "
                            f"state={state_position} pair={pair_position}"
                        )
                    scalar_bins[state_bin, pair_bin] += scalar
                    batch_1 = inputs["batch_1"]
                    batch_2 = inputs["batch_2"]
                    unit_rows.append(
                        {
                            "batch_size": batch_size,
                            "sigma": sigma,
                            "repeat": repeat,
                            "step_key": repeat + 1,
                            "state_position": state_position,
                            "pair_position": pair_position,
                            "source_weight_index": indices[state_position],
                            "train_position": train_positions[state_position],
                            "state_draw_seed": draw_seeds[state_position],
                            "latent_hash": _tensor_hash(z_values[state_position]),
                            "task_name": str(records[state_position].get("task_name", "")),
                            "tau": float(records[state_position].get("tau", 1.0)),
                            "probe_seed_1": int(inputs["probe_seed_1"]),
                            "probe_seed_2": int(inputs["probe_seed_2"]),
                            "batch_1_offset": int(batch_1[0].item()) if batch_1 is not None else float("nan"),
                            "batch_2_offset": int(batch_2[0].item()) if batch_2 is not None else float("nan"),
                            "batch_1_hash": "full" if batch_1 is None else _tensor_hash(batch_1),
                            "batch_2_hash": "full" if batch_2 is None else _tensor_hash(batch_2),
                            "batch_train_count": int(inputs["task_set"].train_labels.shape[0]),
                            "batch_size_effective": int(inputs["task_set"].train_labels.shape[0]) if batch_1 is None else int(batch_1.numel()),
                            "a_scalar": scalar,
                            "h1_norm": vector_norm(h1.detach().cpu()),
                            "h2_norm": vector_norm(h2.detach().cpu()),
                            "unit_elapsed_sec": time.perf_counter() - pair_started,
                        }
                    )
                    pending_losses.append(fd_loss)
                    pending_scalars.append(scalar)
                    pending_units.append((state_position, pair_position))
                    unit_index += 1
                    if len(pending_losses) >= max(1, int(gradient_chunk_size)):
                        flush_gradient_chunk()
                    if unit_index == 1 or unit_index % max(1, log_every) == 0 or unit_index == total:
                        elapsed = time.perf_counter() - started
                        rate = unit_index / max(elapsed, 1e-12)
                        eta = (total - unit_index) / max(rate, 1e-12)
                        print(
                            "[fd_stability] progress "
                            f"B={batch_size} repeat={repeat + 1} unit={unit_index}/{total} "
                            f"A={scalar:.6g} grad_chunk={gradient_chunk_size} "
                            f"rate={rate:.2f}/s eta_sec={eta:.1f}",
                            flush=True,
                        )
                    del h1, h2
            flush_gradient_chunk()
    run.vae.zero_grad(set_to_none=True)
    return RepeatAccumulator(
        repeat=repeat,
        step_key=repeat + 1,
        state_indices=indices,
        state_draw_seeds=draw_seeds,
        train_positions=train_positions,
        latent_hashes=[_tensor_hash(z_values[position]) for position in range(max_states)],
        scalar_bins=scalar_bins,
        gradient_bins=gradient_bins,
    )


def _range_norm_sq(vector64: torch.Tensor, ranges: Sequence[tuple[int, int]]) -> float:
    return float(sum(torch.dot(vector64[start:end], vector64[start:end]).item() for start, end in ranges))


def _range_dot(left64: torch.Tensor, right64: torch.Tensor, ranges: Sequence[tuple[int, int]]) -> float:
    return float(sum(torch.dot(left64[start:end], right64[start:end]).item() for start, end in ranges))


def _cached_analyze(
    repeats: Sequence[RepeatAccumulator],
    *,
    state_grid: Sequence[int],
    pair_grid: Sequence[int],
    module_ranges: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Numerically equivalent stability analysis with cached fp64 vectors/norms."""

    max_state_bin = len(state_grid) - 1
    max_pair_bin = len(pair_grid) - 1
    max_denominator = float(int(state_grid[-1]) * int(pair_grid[-1]))
    max_gradients = [
        prefix_sum_vector(repeat.gradient_bins, max_state_bin, max_pair_bin, denominator=max_denominator)
        for repeat in repeats
    ]
    max_scalars = [
        prefix_sum_scalar(repeat.scalar_bins, max_state_bin, max_pair_bin, denominator=max_denominator)
        for repeat in repeats
    ]
    full_reference = _mean_vectors(max_gradients, list(range(len(repeats))))
    full_reference_scalar = float(np.mean(max_scalars))
    reference_norm = vector_norm(full_reference)
    max_gradient_sum = torch.zeros_like(max_gradients[0])
    for gradient in max_gradients:
        max_gradient_sum.add_(gradient)
    loo_references = [
        (max_gradient_sum - max_gradients[repeat_index]).div(float(len(repeats) - 1))
        for repeat_index in range(len(repeats))
    ]
    loo64 = [reference.double() for reference in loo_references]
    loo_norms = [float(torch.linalg.vector_norm(reference).item()) for reference in loo64]

    module_energy: dict[str, float] = {}
    loo_total_energy = sum(value * value for value in loo_norms)
    for module, ranges in module_ranges.items():
        module_energy[module] = float(
            sum(_range_norm_sq(reference, ranges) for reference in loo64) / max(loo_total_energy, 1e-30)
        )

    split_rows: list[dict[str, Any]] = []
    for split_index, (left_indices, right_indices) in enumerate(_fixed_reference_splits(len(repeats))):
        left_gradient = _mean_vectors(max_gradients, left_indices).double()
        right_gradient = _mean_vectors(max_gradients, right_indices).double()
        left_scalar = float(np.mean([max_scalars[index] for index in left_indices]))
        right_scalar = float(np.mean([max_scalars[index] for index in right_indices]))
        left_norm = float(torch.linalg.vector_norm(left_gradient).item())
        right_norm = float(torch.linalg.vector_norm(right_gradient).item())
        dot = float(torch.dot(left_gradient, right_gradient).item())
        split_rows.append(
            {
                "split": int(split_index),
                "left_repeats": json.dumps(left_indices),
                "right_repeats": json.dumps(right_indices),
                "gradient_cosine": dot / max(left_norm * right_norm, 1e-30),
                "gradient_norm_ratio": left_norm / max(right_norm, 1e-30),
                "scalar_left": left_scalar,
                "scalar_right": right_scalar,
                "scalar_symmetric_error": symmetric_relative_error(left_scalar, right_scalar),
            }
        )
    split_frame = pd.DataFrame(split_rows)
    reference_valid = bool(
        (split_frame["gradient_cosine"] >= 0.99).all()
        and split_frame["gradient_norm_ratio"].between(0.95, 1.05).all()
        and (split_frame["scalar_symmetric_error"] <= 0.05).all()
    )

    metric_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    scalar_total = float(sum(max_scalars))
    for state_bin, state_count in enumerate(state_grid):
        for pair_bin, pair_count in enumerate(pair_grid):
            denominator = float(int(state_count) * int(pair_count))
            candidates = [
                prefix_sum_vector(repeat.gradient_bins, state_bin, pair_bin, denominator=denominator)
                for repeat in repeats
            ]
            candidates64 = [candidate.double() for candidate in candidates]
            candidate_norms = [float(torch.linalg.vector_norm(candidate).item()) for candidate in candidates64]
            candidate_scalars = [
                prefix_sum_scalar(repeat.scalar_bins, state_bin, pair_bin, denominator=denominator)
                for repeat in repeats
            ]
            for repeat_index, (candidate64, candidate_scalar) in enumerate(
                zip(candidates64, candidate_scalars, strict=True)
            ):
                loo_reference64 = loo64[repeat_index]
                loo_scalar = float((scalar_total - max_scalars[repeat_index]) / float(len(repeats) - 1))
                candidate_norm = candidate_norms[repeat_index]
                loo_norm = loo_norms[repeat_index]
                dot = float(torch.dot(candidate64, loo_reference64).item())
                relative_error = float(
                    torch.linalg.vector_norm(candidate64 - loo_reference64).div(max(loo_norm, 1e-30)).item()
                )
                metric_rows.append(
                    {
                        "repeat": int(repeat_index),
                        "states": int(state_count),
                        "pairs": int(pair_count),
                        "hvp_count": int(2 * int(state_count) * int(pair_count)),
                        "a_scalar": float(candidate_scalar),
                        "reference_scalar": loo_scalar,
                        "scalar_symmetric_error": symmetric_relative_error(candidate_scalar, loo_scalar),
                        "gradient_cosine_to_reference": dot / max(candidate_norm * loo_norm, 1e-30),
                        "gradient_relative_error": relative_error,
                        "gradient_norm": candidate_norm,
                        "reference_gradient_norm": loo_norm,
                        "gradient_norm_ratio": candidate_norm / max(loo_norm, 1e-30),
                    }
                )
                for module, ranges in module_ranges.items():
                    candidate_norm_module = math.sqrt(max(_range_norm_sq(candidate64, ranges), 0.0))
                    reference_norm_module = math.sqrt(max(_range_norm_sq(loo_reference64, ranges), 0.0))
                    module_dot = _range_dot(candidate64, loo_reference64, ranges)
                    module_rows.append(
                        {
                            "repeat": int(repeat_index),
                            "states": int(state_count),
                            "pairs": int(pair_count),
                            "module": module,
                            "reference_energy_fraction": module_energy[module],
                            "gradient_cosine_to_reference": module_dot
                            / max(candidate_norm_module * reference_norm_module, 1e-30),
                            "gradient_norm_ratio": candidate_norm_module / max(reference_norm_module, 1e-30),
                        }
                    )
            for left_index in range(len(candidates64)):
                for right_index in range(left_index + 1, len(candidates64)):
                    dot = float(torch.dot(candidates64[left_index], candidates64[right_index]).item())
                    left_norm = candidate_norms[left_index]
                    right_norm = candidate_norms[right_index]
                    pairwise_rows.append(
                        {
                            "states": int(state_count),
                            "pairs": int(pair_count),
                            "left_repeat": int(left_index),
                            "right_repeat": int(right_index),
                            "gradient_cosine": dot / max(left_norm * right_norm, 1e-30),
                            "gradient_norm_ratio": left_norm / max(right_norm, 1e-30),
                        }
                    )
            del candidates, candidates64

    metrics = pd.DataFrame(metric_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    module_metrics = pd.DataFrame(module_rows)
    summary_rows: list[dict[str, Any]] = []
    for (state_count, pair_count), group in metrics.groupby(["states", "pairs"], sort=True):
        pair_group = pairwise.loc[(pairwise["states"] == state_count) & (pairwise["pairs"] == pair_count)]
        relevant_modules = module_metrics.loc[
            (module_metrics["states"] == state_count)
            & (module_metrics["pairs"] == pair_count)
            & (module_metrics["reference_energy_fraction"] >= 0.01)
        ]
        module_summaries = [
            (
                quantile(module_group["gradient_cosine_to_reference"], 0.5),
                quantile(module_group["gradient_cosine_to_reference"], 0.1),
            )
            for _module, module_group in relevant_modules.groupby("module")
        ]
        module_pass = bool(module_summaries) and all(
            median >= 0.90 and q10 >= 0.80 for median, q10 in module_summaries
        )
        row = {
            "states": int(state_count),
            "pairs": int(pair_count),
            "hvp_count": int(2 * int(state_count) * int(pair_count)),
            "scalar_mean": float(group["a_scalar"].mean()),
            "scalar_std": float(group["a_scalar"].std(ddof=1)),
            "scalar_symmetric_error_median": quantile(group["scalar_symmetric_error"], 0.5),
            "scalar_symmetric_error_q90": quantile(group["scalar_symmetric_error"], 0.9),
            "gradient_cosine_median": quantile(group["gradient_cosine_to_reference"], 0.5),
            "gradient_cosine_q10": quantile(group["gradient_cosine_to_reference"], 0.1),
            "gradient_cosine_min": float(group["gradient_cosine_to_reference"].min()),
            "gradient_relative_error_median": quantile(group["gradient_relative_error"], 0.5),
            "gradient_relative_error_q90": quantile(group["gradient_relative_error"], 0.9),
            "gradient_norm_ratio_median": quantile(group["gradient_norm_ratio"], 0.5),
            "gradient_norm_ratio_q10": quantile(group["gradient_norm_ratio"], 0.1),
            "gradient_norm_ratio_q90": quantile(group["gradient_norm_ratio"], 0.9),
            "pairwise_cosine_median": quantile(pair_group["gradient_cosine"], 0.5),
            "pairwise_cosine_q10": quantile(pair_group["gradient_cosine"], 0.1),
            "pairwise_cosine_min": float(pair_group["gradient_cosine"].min()),
            "pairwise_negative_fraction": float((pair_group["gradient_cosine"] < 0.0).mean()),
            "relevant_module_count": int(relevant_modules["module"].nunique()),
            "module_stability_pass": module_pass,
        }
        row["scalar_stability_pass"] = bool(
            row["scalar_symmetric_error_median"] <= 0.10 and row["scalar_symmetric_error_q90"] <= 0.20
        )
        row["gradient_stability_pass"] = bool(
            row["gradient_cosine_median"] >= 0.95
            and row["gradient_cosine_q10"] >= 0.90
            and row["pairwise_cosine_median"] >= 0.90
            and row["pairwise_cosine_q10"] >= 0.80
            and 0.90 <= row["gradient_norm_ratio_median"] <= 1.10
            and row["gradient_norm_ratio_q10"] >= 0.75
            and row["gradient_norm_ratio_q90"] <= 1.33
            and module_pass
        )
        row["cell_stability_pass"] = bool(
            reference_valid and row["scalar_stability_pass"] and row["gradient_stability_pass"]
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["states", "pairs"]).reset_index(drop=True)
    pass_map = {(int(row.states), int(row.pairs)): bool(row.cell_stability_pass) for row in summary.itertuples()}
    summary["persistent_stability_pass"] = [
        bool(
            pass_map.get((int(row.states), int(row.pairs)), False)
            and pass_map.get((2 * int(row.states), int(row.pairs)), False)
            and pass_map.get((int(row.states), 2 * int(row.pairs)), False)
        )
        for row in summary.itertuples()
    ]
    persistent = summary.loc[summary["persistent_stability_pass"]].sort_values(["hvp_count", "states", "pairs"])
    selected = None if persistent.empty else persistent.iloc[0].to_dict()
    reference = {
        "max_states": int(state_grid[-1]),
        "max_pairs": int(pair_grid[-1]),
        "outer_repeats": int(len(repeats)),
        "full_reference_scalar": full_reference_scalar,
        "full_reference_gradient_norm": reference_norm,
        "max_budget_scalar_values": max_scalars,
        "reference_valid": reference_valid,
        "selected_persistent_cell": selected,
        "module_reference_energy": module_energy,
    }
    return metrics, summary, pairwise, module_metrics, {"reference": reference, "split_rows": split_rows}


def _run_batch_grid(
    run: LoadedRun,
    *,
    output_dir: Path,
    batch_size: int,
    sigma: float,
    state_grid: Sequence[int],
    pair_grid: Sequence[int],
    repeats: int,
    active: Sequence[tuple[str, torch.nn.Parameter]],
    base_seed: int,
    device: torch.device,
    gradient_chunk_size: int,
    log_every: int,
) -> dict[str, Any]:
    batch_dir = output_dir / f"batch_{batch_size}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    unit_rows: list[dict[str, Any]] = []
    gradient_chunk_rows: list[dict[str, Any]] = []
    accumulators: list[RepeatAccumulator] = []
    for repeat in range(repeats):
        print(f"[fd_stability] repeat_start B={batch_size} repeat={repeat + 1}/{repeats}", flush=True)
        accumulators.append(
            _run_repeat_fd(
                run,
                repeat=repeat,
                batch_size=batch_size,
                sigma=sigma,
                state_grid=state_grid,
                pair_grid=pair_grid,
                active=active,
                base_seed=base_seed,
                device=device,
                unit_rows=unit_rows,
                gradient_chunk_rows=gradient_chunk_rows,
                gradient_chunk_size=gradient_chunk_size,
                log_every=log_every,
            )
        )
        pd.DataFrame(unit_rows).to_csv(batch_dir / "atomic_pair_samples.partial.csv", index=False)
        pd.DataFrame(gradient_chunk_rows).to_csv(batch_dir / "gradient_chunks.partial.csv", index=False)
    units = pd.DataFrame(unit_rows)
    units.to_csv(batch_dir / "atomic_pair_samples.csv", index=False)
    pd.DataFrame(gradient_chunk_rows).to_csv(batch_dir / "gradient_chunks.csv", index=False)
    partial = batch_dir / "atomic_pair_samples.partial.csv"
    if partial.exists():
        partial.unlink()
    partial_chunks = batch_dir / "gradient_chunks.partial.csv"
    if partial_chunks.exists():
        partial_chunks.unlink()
    overlap, overlap_summary = _batch_pair_overlap_audit(units)
    overlap.to_csv(batch_dir / "batch_pair_overlap.csv", index=False)
    overlap_summary.to_csv(batch_dir / "batch_pair_overlap_summary.csv", index=False)
    module_ranges, _parameter_rows = _module_slices(active)
    metrics, summary, pairwise, module_metrics, reference_bundle = _cached_analyze(
        accumulators,
        state_grid=state_grid,
        pair_grid=pair_grid,
        module_ranges=module_ranges,
    )
    metrics.insert(0, "batch_size", batch_size)
    summary.insert(0, "batch_size", batch_size)
    pairwise.insert(0, "batch_size", batch_size)
    module_metrics.insert(0, "batch_size", batch_size)
    metrics.to_csv(batch_dir / "replicate_metrics.csv", index=False)
    summary.to_csv(batch_dir / "stability_summary.csv", index=False)
    pairwise.to_csv(batch_dir / "pairwise_gradient_cosines.csv", index=False)
    module_metrics.to_csv(batch_dir / "module_gradient_metrics.csv", index=False)
    pd.DataFrame(reference_bundle["split_rows"]).to_csv(batch_dir / "reference_split_checks.csv", index=False)
    (batch_dir / "reference.json").write_text(
        json.dumps(reference_bundle["reference"], indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    _write_plots(batch_dir, summary, metrics)
    reference = reference_bundle["reference"]
    current = summary.loc[(summary["states"] == 2) & (summary["pairs"] == 4)]
    result = {
        "batch_size": batch_size,
        "sigma": sigma,
        "reference_valid": bool(reference["reference_valid"]),
        "selected_persistent_cell": reference["selected_persistent_cell"],
        "current_2x4": None if current.empty else current.iloc[0].to_dict(),
        "atomic_pairs": len(units),
    }
    (batch_dir / "decision.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=float), encoding="utf-8")
    del accumulators
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-difference Variant A estimator stability audit.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", default="1,2,4,8,16,32")
    parser.add_argument("--pairs", default="1,2,4,8,16,32")
    parser.add_argument("--batch-sizes", default="16,32,64,128,256,512")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--sigmas", default=",".join(str(value) for value in DEFAULT_SIGMAS))
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--gradient-chunk-size", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    state_grid = parse_grid(args.states)
    pair_grid = parse_grid(args.pairs)
    batch_sizes = parse_grid(args.batch_sizes)
    sigmas = [float(value.strip()) for value in str(args.sigmas).split(",") if value.strip()]
    source_hashes = {
        "checkpoint": sha256_file(run_dir / "vae_checkpoint.pt"),
        "weight_pool": sha256_file(run_dir / "weight_pool.pt"),
        "weight_records": sha256_file(run_dir / "weight_pool_records.csv"),
    }
    expected = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "weight_pool": EXPECTED_POOL_SHA256,
        "weight_records": EXPECTED_RECORDS_SHA256,
    }
    if source_hashes != expected:
        raise ValueError(f"source hash mismatch observed={source_hashes} expected={expected}")
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "seed_protocol_id": SEED_PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": "float32",
        "seed": args.seed,
        "states": state_grid,
        "pairs": pair_grid,
        "batch_sizes": batch_sizes,
        "repeats": args.repeats,
        "sigma_grid": sigmas,
        "forced_sigma": args.sigma,
        "gradient_chunk_size": int(args.gradient_chunk_size),
        "analysis_engine": "cached_fp64_exact",
        "source_hashes": source_hashes,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[fd_stability] start config={json.dumps(resolved, sort_keys=True)}", flush=True)
    started = time.perf_counter()
    run = _load_run(run_dir, device=device)
    if int(run.train_indices.numel()) != 11021 or int(run.cfg.latent_dim) != 512 or int(run.cfg.vae_hidden_dim) != 2048:
        raise ValueError("accepted checkpoint architecture/split mismatch")

    selected_sigma = args.sigma
    active: list[tuple[str, torch.nn.Parameter]] | None = None
    if not args.skip_calibration:
        selected_sigma, active, _calibration_rows, _calibration_summary = _calibrate_sigma(
            run,
            sigmas=sigmas,
            batch_sizes=[16, 128, 512],
            base_seed=int(args.seed),
            device=device,
            output_dir=output_dir,
        )
    else:
        if selected_sigma is None:
            decision_path = output_dir / "sigma_decision.json"
            if decision_path.is_file():
                selected_sigma = json.loads(decision_path.read_text(encoding="utf-8")).get("selected_sigma")
        if selected_sigma is None:
            raise ValueError("--skip-calibration requires --sigma or an existing passing sigma_decision.json")
        indices, _positions, _seeds, z_values, records = _prepare_states(
            run, repeat=0, count=1, base_seed=int(args.seed), device=device
        )
        inputs = _pair_inputs(
            run,
            z_values[0],
            records[0],
            repeat=0,
            state_position=0,
            pair_position=0,
            batch_size=128,
            base_seed=int(args.seed),
        )
        loss, _h1, _h2 = _fd_pair_loss(run, z_values[0], inputs, sigma=float(selected_sigma))
        _vector, active = _gradient_vector(run, loss, None)
        del _vector, loss, _h1, _h2

    if selected_sigma is None:
        print("[fd_stability] stop calibration_failed=true; main grid not launched", flush=True)
        return
    if active is None:
        raise RuntimeError("active parameter set was not established")
    parameter_ranges, parameter_rows = _module_slices(active)
    del parameter_ranges
    (output_dir / "active_decoder_parameters.json").write_text(
        json.dumps(
            {
                "active_parameter_count": int(sum(parameter.numel() for _name, parameter in active)),
                "active_parameter_names": [name for name, _parameter in active],
                "parameters": parameter_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if args.calibration_only:
        print(
            f"[fd_stability] calibration_done selected_sigma={selected_sigma:g} "
            f"elapsed_sec={time.perf_counter() - started:.1f}",
            flush=True,
        )
        return

    decisions: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        print(f"[fd_stability] batch_grid_start B={batch_size} sigma={selected_sigma:g}", flush=True)
        decisions.append(
            _run_batch_grid(
                run,
                output_dir=output_dir,
                batch_size=int(batch_size),
                sigma=float(selected_sigma),
                state_grid=state_grid,
                pair_grid=pair_grid,
                repeats=int(args.repeats),
                active=active,
                base_seed=int(args.seed),
                device=device,
                gradient_chunk_size=max(1, int(args.gradient_chunk_size)),
                log_every=int(args.log_every),
            )
        )
    (output_dir / "cross_batch_decisions.json").write_text(
        json.dumps(decisions, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "batch_size": row["batch_size"],
                "sigma": row["sigma"],
                "reference_valid": row["reference_valid"],
                "selected_states": None if row["selected_persistent_cell"] is None else row["selected_persistent_cell"]["states"],
                "selected_pairs": None if row["selected_persistent_cell"] is None else row["selected_persistent_cell"]["pairs"],
                "current_2x4_cell_pass": None if row["current_2x4"] is None else row["current_2x4"]["cell_stability_pass"],
                "current_2x4_persistent_pass": None if row["current_2x4"] is None else row["current_2x4"]["persistent_stability_pass"],
            }
            for row in decisions
        ]
    ).to_csv(output_dir / "cross_batch_summary.csv", index=False)
    checkpoint_after = sha256_file(run_dir / "vae_checkpoint.pt")
    if checkpoint_after != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint changed during read-only FD audit")
    manifest = {
        "status": "complete",
        "elapsed_sec": time.perf_counter() - started,
        "selected_sigma": selected_sigma,
        "active_parameter_count": int(sum(parameter.numel() for _name, parameter in active)),
        "decisions": decisions,
        "checkpoint_sha256_after": checkpoint_after,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print(
        f"[fd_stability] done elapsed_sec={manifest['elapsed_sec']:.1f} "
        f"selected_sigma={selected_sigma:g} manifest={output_dir / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
