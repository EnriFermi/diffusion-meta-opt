from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    encode_weights,
    spec_from_payload,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    PreconditioningState,
    _batch_indices,
    _latent_hvp,
    _probe_like,
    _task_set_for_record,
    preconditioning_regularizer,
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
    / "a_estimator_stability_unfinetuned_h2048"
)
PROTOCOL_PATH = DEFAULT_OUTPUT_DIR / "protocol.md"
EXPECTED_CHECKPOINT_SHA256 = "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
EXPECTED_POOL_SHA256 = "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef"
EXPECTED_RECORDS_SHA256 = "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933"
PROTOCOL_ID = "a_estimator_stability_unfinetuned_h2048_v2"


@dataclass(slots=True)
class LoadedRun:
    cfg: ExperimentConfig
    weights: torch.Tensor
    records: pd.DataFrame
    spec: Any
    vae: torch.nn.Module
    normalizer: WeightNormalizer
    train_indices: torch.Tensor
    task_tensors: dict[str, Any]


@dataclass(slots=True)
class RepeatAccumulator:
    repeat: int
    step_key: int
    state_indices: list[int]
    state_draw_seeds: list[int]
    train_positions: list[int]
    latent_hashes: list[str]
    scalar_bins: np.ndarray
    gradient_bins: list[list[torch.Tensor | None]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uint63(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def git_metadata() -> dict[str, Any]:
    def _run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout.strip()

    return {
        "commit": _run("rev-parse", "HEAD"),
        "status_short": _run("status", "--short"),
    }


def parse_grid(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise ValueError("grid values must be positive integers")
    return values


def grid_bin(position: int, endpoints: Sequence[int]) -> int:
    count = int(position) + 1
    for index, endpoint in enumerate(endpoints):
        if count <= int(endpoint):
            return index
    raise IndexError(f"position {position} lies outside grid ending at {endpoints[-1]}")


def prefix_sum_vector(
    bins: Sequence[Sequence[torch.Tensor | None]],
    state_bin: int,
    pair_bin: int,
    *,
    denominator: float,
) -> torch.Tensor:
    result: torch.Tensor | None = None
    for state_index in range(int(state_bin) + 1):
        for pair_index in range(int(pair_bin) + 1):
            value = bins[state_index][pair_index]
            if value is None:
                raise ValueError(f"missing gradient bin ({state_index}, {pair_index})")
            if result is None:
                result = value.clone()
            else:
                result.add_(value)
    if result is None:
        raise ValueError("empty gradient prefix")
    return result.div_(float(denominator))


def prefix_sum_scalar(bins: np.ndarray, state_bin: int, pair_bin: int, *, denominator: float) -> float:
    return float(bins[: int(state_bin) + 1, : int(pair_bin) + 1].sum() / float(denominator))


def vector_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.double()).item())


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.double()
    right64 = right.double()
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    if float(denominator.item()) <= 0.0:
        return float("nan")
    return float(torch.dot(left64, right64).div(denominator).item())


def vector_relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.double())
    if float(denominator.item()) <= 0.0:
        return float("nan")
    return float(torch.linalg.vector_norm(value.double() - reference.double()).div(denominator).item())


def symmetric_relative_error(left: float, right: float, *, eps: float = 1e-12) -> float:
    return float(2.0 * abs(float(left) - float(right)) / (abs(float(left)) + abs(float(right)) + float(eps)))


def quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, float(q)))


def _load_run(run_dir: Path, *, device: torch.device) -> LoadedRun:
    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**config_payload["config"])
    dtype = torch_dtype(cfg)
    weight_payload = torch.load(run_dir / "weight_pool.pt", map_location="cpu", mmap=True, weights_only=False)
    checkpoint_payload = torch.load(run_dir / "vae_checkpoint.pt", map_location="cpu", weights_only=False)
    weights = weight_payload["weights"].detach().cpu()
    records = pd.read_csv(run_dir / "weight_pool_records.csv")
    if len(records) != int(weights.shape[0]):
        raise ValueError(f"weight/record count mismatch: {int(weights.shape[0])} vs {len(records)}")
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(checkpoint_payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=dtype)
    vae.load_state_dict(checkpoint_payload["model_state"])
    vae.eval()
    train_indices = checkpoint_payload.get("train_indices")
    if not isinstance(train_indices, torch.Tensor):
        raise ValueError("checkpoint has no train_indices tensor")
    train_indices = train_indices.detach().cpu().long()
    task_tensors = _load_task_tensors_for_pipeline(cfg, device=device, dtype=dtype)
    return LoadedRun(
        cfg=cfg,
        weights=weights,
        records=records,
        spec=spec,
        vae=vae,
        normalizer=normalizer,
        train_indices=train_indices,
        task_tensors=task_tensors,
    )


def _probe_cfg(cfg: ExperimentConfig, *, sample_count: int, pair_count: int, batch_size: int) -> ExperimentConfig:
    return replace(
        cfg,
        vae_precond_loss_kind="li_a_hvp",
        vae_precond_coeff=1.0,
        vae_precond_burg_coeff=0.0,
        vae_precond_every=1,
        vae_precond_samples=int(sample_count),
        vae_precond_pairs=int(pair_count),
        vae_precond_batch_size=int(batch_size),
        vae_precond_probe_scale=1.0,
        vae_precond_estimator_scope="local",
        vae_precond_hvp_mode="stopped_composite",
        vae_precond_loss_clip=0.0,
        vae_precond_grad_damping=0.0,
    )


def _atomic_pair_loss(
    *,
    cfg: ExperimentConfig,
    run: LoadedRun,
    z: torch.Tensor,
    record: Mapping[str, Any],
    step: int,
    pair_index: int,
    probe_generator_1: torch.Generator,
    probe_generator_2: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    task_set = _task_set_for_record(run.task_tensors, record)
    sample_key = int(record.get("source_weight_index", record.get("weight_index", 0)))
    tau = float(record.get("tau", 1.0))
    batch_1 = _batch_indices(task_set, batch_size=int(cfg.vae_precond_batch_size), step=int(step), sample_key=sample_key, pair_key=2 * int(pair_index))
    batch_2 = _batch_indices(task_set, batch_size=int(cfg.vae_precond_batch_size), step=int(step), sample_key=sample_key, pair_key=2 * int(pair_index) + 1)
    u1 = _probe_like(z, generator=probe_generator_1, scale=1.0)
    u2 = _probe_like(z, generator=probe_generator_2, scale=1.0)
    h1 = _latent_hvp(
        cfg,
        run.vae,
        run.normalizer,
        z,
        u1,
        task_set=task_set,
        spec=run.spec,
        tau=tau,
        batch_indices=batch_1,
    )
    h2 = _latent_hvp(
        cfg,
        run.vae,
        run.normalizer,
        z,
        u2,
        task_set=task_set,
        spec=run.spec,
        tau=tau,
        batch_indices=batch_2,
    )
    dim = int(z.numel())
    dot = (h1 * h2).sum()
    norm1 = h1.square().sum()
    norm2 = h2.square().sum()
    loss = (dot.square() - norm1 - norm2 + float(dim)) / float(dim)

    def _batch_hash(indices: torch.Tensor | None) -> str:
        if indices is None:
            return "full"
        return hashlib.sha256(indices.detach().cpu().numpy().tobytes()).hexdigest()

    overlap = float("nan")
    if batch_1 is not None and batch_2 is not None:
        overlap = float(len(set(batch_1.detach().cpu().tolist()).intersection(batch_2.detach().cpu().tolist())))
    return loss, {
        "h1_norm": float(torch.linalg.vector_norm(h1.detach().float()).cpu().item()),
        "h2_norm": float(torch.linalg.vector_norm(h2.detach().float()).cpu().item()),
        "h1_h2_dot": float(dot.detach().cpu().item()),
        "h1_hash": sha256_tensor(h1),
        "h2_hash": sha256_tensor(h2),
        "probe_1_norm": float(torch.linalg.vector_norm(u1.detach().float()).cpu().item()),
        "probe_2_norm": float(torch.linalg.vector_norm(u2.detach().float()).cpu().item()),
        "probe_1_hash": sha256_tensor(u1),
        "probe_2_hash": sha256_tensor(u2),
        "branch_batch_overlap": overlap,
        "batch_1_hash": _batch_hash(batch_1),
        "batch_2_hash": _batch_hash(batch_2),
        "batch_1_offset": float("nan") if batch_1 is None else int(batch_1[0].detach().cpu().item()),
        "batch_2_offset": float("nan") if batch_2 is None else int(batch_2[0].detach().cpu().item()),
        "batch_train_count": int(task_set.train_labels.shape[0]),
        "batch_size_effective": int(task_set.train_labels.shape[0]) if batch_1 is None else int(batch_1.numel()),
        "batch_1_indices": "full" if batch_1 is None else json.dumps(batch_1.detach().cpu().tolist()),
        "batch_2_indices": "full" if batch_2 is None else json.dumps(batch_2.detach().cpu().tolist()),
    }


def _flatten_active_gradients(active: Sequence[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    gradients: list[torch.Tensor] = []
    for name, parameter in active:
        if parameter.grad is None:
            raise RuntimeError(f"active parameter lost its gradient: {name}")
        gradients.append(parameter.grad.detach().reshape(-1).float())
    return torch.cat(gradients, dim=0).cpu()


def _discover_active_parameters(vae: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in vae.named_parameters() if parameter.grad is not None]


def _sequential_generators(seed: int) -> tuple[torch.Generator, torch.Generator]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return generator, generator


def _indexed_generators(
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


def _preflight_decomposition(
    run: LoadedRun,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[tuple[str, torch.nn.Parameter]]]:
    sample_count = 2
    pair_count = 2
    cfg = _probe_cfg(run.cfg, sample_count=sample_count, pair_count=pair_count, batch_size=batch_size)
    selected = run.train_indices[:sample_count].tolist()
    selected_weights = run.weights[selected].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z_values = encode_weights(run.vae, run.normalizer, selected_weights).detach()
    records = run.records.iloc[selected].to_dict(orient="records")
    for record, source_index in zip(records, selected, strict=True):
        record["source_weight_index"] = int(source_index)
    step = 10
    full_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    run.vae.zero_grad(set_to_none=True)
    full_loss, _row = preconditioning_regularizer(
        cfg,
        state=PreconditioningState(),
        vae=run.vae,
        normalizer=run.normalizer,
        z_samples=z_values,
        records=records,
        task_tensors=run.task_tensors,
        spec=run.spec,
        step=step,
        generator=full_generator,
    )
    full_loss.backward()
    active = _discover_active_parameters(run.vae)
    full_gradient = _flatten_active_gradients(active)

    direct_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direct_gradient = torch.zeros_like(full_gradient)
    direct_scalar = 0.0
    for state_position in range(sample_count):
        for pair_position in range(pair_count):
            run.vae.zero_grad(set_to_none=True)
            atomic_loss, _stats = _atomic_pair_loss(
                cfg=cfg,
                run=run,
                z=z_values[state_position],
                record=records[state_position],
                step=step,
                pair_index=pair_position,
                probe_generator_1=direct_generator,
                probe_generator_2=direct_generator,
            )
            atomic_loss.backward()
            direct_gradient.add_(_flatten_active_gradients(active))
            direct_scalar += float(atomic_loss.detach().cpu().item())
    denominator = float(sample_count * pair_count)
    direct_gradient.div_(denominator)
    direct_scalar /= denominator
    full_scalar = float(full_loss.detach().cpu().item())
    task_set = _task_set_for_record(run.task_tensors, records[0])
    sample_key = int(records[0]["source_weight_index"])
    tau = float(records[0].get("tau", 1.0))
    hessian_batch = _batch_indices(task_set, batch_size=batch_size, step=step, sample_key=sample_key, pair_key=0)
    probe_generator = torch.Generator(device="cpu").manual_seed(stable_uint63("hvp_equivalence", seed))
    probe = _probe_like(z_values[0], generator=probe_generator, scale=1.0)
    stopped_hvp = _latent_hvp(
        cfg,
        run.vae,
        run.normalizer,
        z_values[0],
        probe,
        task_set=task_set,
        spec=run.spec,
        tau=tau,
        batch_indices=hessian_batch,
    ).detach().cpu()
    autograd_cfg = replace(cfg, vae_precond_hvp_mode="autograd")
    autograd_hvp = _latent_hvp(
        autograd_cfg,
        run.vae,
        run.normalizer,
        z_values[0],
        probe,
        task_set=task_set,
        spec=run.spec,
        tau=tau,
        batch_indices=hessian_batch,
    ).detach().cpu()
    result = {
        "sample_count": sample_count,
        "pair_count": pair_count,
        "full_scalar": full_scalar,
        "atomic_scalar": direct_scalar,
        "scalar_abs_error": abs(full_scalar - direct_scalar),
        "gradient_cosine": vector_cosine(full_gradient, direct_gradient),
        "gradient_relative_error": vector_relative_error(direct_gradient, full_gradient),
        "full_gradient_norm": vector_norm(full_gradient),
        "atomic_gradient_norm": vector_norm(direct_gradient),
        "active_parameter_count": int(full_gradient.numel()),
        "active_parameter_names": [name for name, _parameter in active],
        "excluded_parameter_names": [name for name, parameter in run.vae.named_parameters() if parameter.grad is None],
        "stopped_vs_autograd_hvp_cosine": vector_cosine(stopped_hvp, autograd_hvp),
        "stopped_vs_autograd_hvp_relative_error": vector_relative_error(stopped_hvp, autograd_hvp),
    }
    if (
        result["scalar_abs_error"] > 5e-5
        or result["gradient_cosine"] < 0.99999
        or result["gradient_relative_error"] > 5e-5
        or result["stopped_vs_autograd_hvp_cosine"] < 0.99999
        or result["stopped_vs_autograd_hvp_relative_error"] > 5e-5
    ):
        raise RuntimeError(f"atomic accumulation preflight failed: {json.dumps(result, sort_keys=True)}")
    run.vae.zero_grad(set_to_none=True)
    return result, active


def _module_slices(active: Sequence[tuple[str, torch.nn.Parameter]]) -> tuple[dict[str, list[tuple[int, int]]], list[dict[str, Any]]]:
    groups: dict[str, list[tuple[int, int]]] = {}
    rows: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in active:
        count = int(parameter.numel())
        group = name.split(".", 1)[0]
        groups.setdefault(group, []).append((offset, offset + count))
        rows.append(
            {
                "parameter": name,
                "module": group,
                "shape": json.dumps(list(parameter.shape)),
                "offset_start": offset,
                "offset_end": offset + count,
                "numel": count,
            }
        )
        offset += count
    return groups, rows


def _slice_vector(vector: torch.Tensor, ranges: Sequence[tuple[int, int]]) -> torch.Tensor:
    if len(ranges) == 1:
        start, end = ranges[0]
        return vector[start:end]
    return torch.cat([vector[start:end] for start, end in ranges], dim=0)


def _sample_state_indices(
    train_indices: torch.Tensor,
    *,
    count: int,
    seed: int,
    repeat: int,
) -> tuple[list[int], list[int], list[int]]:
    state_indices: list[int] = []
    draw_seeds: list[int] = []
    train_positions: list[int] = []
    for state_position in range(int(count)):
        draw_seed = stable_uint63(PROTOCOL_ID, seed, "state", repeat, state_position, -1, -1)
        generator = torch.Generator(device="cpu").manual_seed(draw_seed)
        train_position = int(torch.randint(0, int(train_indices.numel()), (1,), generator=generator).item())
        state_indices.append(int(train_indices[train_position].item()))
        draw_seeds.append(int(draw_seed))
        train_positions.append(train_position)
    return state_indices, draw_seeds, train_positions


def _run_repeat(
    run: LoadedRun,
    *,
    repeat: int,
    state_grid: Sequence[int],
    pair_grid: Sequence[int],
    active: Sequence[tuple[str, torch.nn.Parameter]],
    excluded: Sequence[tuple[str, torch.nn.Parameter]],
    batch_size: int,
    seed: int,
    device: torch.device,
    log_every: int,
    unit_rows: list[dict[str, Any]],
) -> RepeatAccumulator:
    max_states = int(state_grid[-1])
    max_pairs = int(pair_grid[-1])
    state_indices, state_draw_seeds, train_positions = _sample_state_indices(
        run.train_indices,
        count=max_states,
        seed=seed,
        repeat=repeat,
    )
    step_key = int(repeat) + 1
    selected_weights = run.weights[state_indices].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z_values = encode_weights(run.vae, run.normalizer, selected_weights).detach()
    latent_hashes = [sha256_tensor(z_values[position]) for position in range(max_states)]
    records = run.records.iloc[state_indices].to_dict(orient="records")
    for record, source_index in zip(records, state_indices, strict=True):
        record["source_weight_index"] = int(source_index)
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=1, batch_size=batch_size)
    scalar_bins = np.zeros((len(state_grid), len(pair_grid)), dtype=np.float64)
    gradient_bins: list[list[torch.Tensor | None]] = [
        [None for _pair in pair_grid] for _state in state_grid
    ]
    total_units = max_states * max_pairs
    started = time.perf_counter()
    for state_position in range(max_states):
        record = records[state_position]
        state_bin = grid_bin(state_position, state_grid)
        for pair_position in range(max_pairs):
            pair_bin = grid_bin(pair_position, pair_grid)
            generator_1, generator_2, probe_seed_1, probe_seed_2 = _indexed_generators(
                seed, repeat, state_position, pair_position
            )
            run.vae.zero_grad(set_to_none=True)
            unit_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            atomic_loss, atomic_stats = _atomic_pair_loss(
                cfg=cfg,
                run=run,
                z=z_values[state_position],
                record=record,
                step=step_key,
                pair_index=pair_position,
                probe_generator_1=generator_1,
                probe_generator_2=generator_2,
            )
            atomic_loss.backward()
            for excluded_name, excluded_parameter in excluded:
                if excluded_parameter.grad is not None and bool((excluded_parameter.grad.detach() != 0).any().item()):
                    raise RuntimeError(
                        f"excluded parameter received A gradient repeat={repeat} state={state_position} "
                        f"pair={pair_position} parameter={excluded_name}"
                    )
            gradient = _flatten_active_gradients(active)
            if not bool(torch.isfinite(gradient).all().item()) or not math.isfinite(float(atomic_loss.detach().cpu().item())):
                raise FloatingPointError(f"nonfinite atomic result repeat={repeat} state={state_position} pair={pair_position}")
            existing = gradient_bins[state_bin][pair_bin]
            if existing is None:
                gradient_bins[state_bin][pair_bin] = gradient.clone()
            else:
                existing.add_(gradient)
            scalar_value = float(atomic_loss.detach().cpu().item())
            scalar_bins[state_bin, pair_bin] += scalar_value
            unit_index = state_position * max_pairs + pair_position + 1
            unit_rows.append(
                {
                    "repeat": int(repeat),
                    "step_key": int(step_key),
                    "state_position": int(state_position),
                    "pair_position": int(pair_position),
                    "state_bin": int(state_bin),
                    "pair_bin": int(pair_bin),
                    "source_weight_index": int(state_indices[state_position]),
                    "task_name": str(record.get("task_name", "")),
                    "tau": float(record.get("tau", 1.0)),
                    "source_lr": float(record.get("source_lr", float("nan"))),
                    "source_step": float(record.get("step", float("nan"))),
                    "state_draw_seed": int(state_draw_seeds[state_position]),
                    "train_position": int(train_positions[state_position]),
                    "latent_hash": latent_hashes[state_position],
                    "probe_seed_1": int(probe_seed_1),
                    "probe_seed_2": int(probe_seed_2),
                    "a_scalar": scalar_value,
                    "gradient_norm": vector_norm(gradient),
                    "unit_elapsed_sec": float(time.perf_counter() - unit_started),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                    **atomic_stats,
                }
            )
            if unit_index == 1 or unit_index % max(1, int(log_every)) == 0 or unit_index == total_units:
                elapsed = time.perf_counter() - started
                rate = unit_index / max(elapsed, 1e-12)
                eta = (total_units - unit_index) / max(rate, 1e-12)
                print(
                    "[a_stability] progress "
                    f"repeat={repeat + 1} unit={unit_index}/{total_units} "
                    f"state={state_position + 1}/{max_states} pair={pair_position + 1}/{max_pairs} "
                    f"A={scalar_value:.6g} grad_norm={unit_rows[-1]['gradient_norm']:.6g} "
                    f"elapsed_sec={elapsed:.1f} rate={rate:.2f}/s eta_sec={eta:.1f}",
                    flush=True,
                )
            del gradient, atomic_loss
    run.vae.zero_grad(set_to_none=True)
    return RepeatAccumulator(
        repeat=int(repeat),
        step_key=int(step_key),
        state_indices=state_indices,
        state_draw_seeds=state_draw_seeds,
        train_positions=train_positions,
        latent_hashes=latent_hashes,
        scalar_bins=scalar_bins,
        gradient_bins=gradient_bins,
    )


def _fixed_reference_splits(repeats: int) -> list[tuple[list[int], list[int]]]:
    if int(repeats) != 12:
        midpoint = int(repeats) // 2
        if midpoint == 0 or int(repeats) - midpoint == 0:
            raise ValueError("at least two repeats are required")
        return [(list(range(midpoint)), list(range(midpoint, int(repeats))))]
    return [
        (list(range(0, 6)), list(range(6, 12))),
        ([0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]),
        ([0, 1, 4, 5, 8, 9], [2, 3, 6, 7, 10, 11]),
    ]


def _mean_vectors(vectors: Sequence[torch.Tensor], indices: Sequence[int]) -> torch.Tensor:
    result = torch.zeros_like(vectors[0])
    for index in indices:
        result.add_(vectors[int(index)])
    return result.div_(float(len(indices)))


def _analyze(
    repeats: Sequence[RepeatAccumulator],
    *,
    state_grid: Sequence[int],
    pair_grid: Sequence[int],
    module_ranges: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
    module_energy: dict[str, float] = {}
    loo_total_energy = sum(vector_norm(reference) ** 2 for reference in loo_references)
    for module, ranges in module_ranges.items():
        module_energy[module] = float(
            sum(vector_norm(_slice_vector(reference, ranges)) ** 2 for reference in loo_references)
            / max(loo_total_energy, 1e-30)
        )

    split_rows: list[dict[str, Any]] = []
    for split_index, (left_indices, right_indices) in enumerate(_fixed_reference_splits(len(repeats))):
        left_gradient = _mean_vectors(max_gradients, left_indices)
        right_gradient = _mean_vectors(max_gradients, right_indices)
        left_scalar = float(np.mean([max_scalars[index] for index in left_indices]))
        right_scalar = float(np.mean([max_scalars[index] for index in right_indices]))
        left_norm = vector_norm(left_gradient)
        right_norm = vector_norm(right_gradient)
        split_rows.append(
            {
                "split": int(split_index),
                "left_repeats": json.dumps(left_indices),
                "right_repeats": json.dumps(right_indices),
                "gradient_cosine": vector_cosine(left_gradient, right_gradient),
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
    for state_bin, state_count in enumerate(state_grid):
        for pair_bin, pair_count in enumerate(pair_grid):
            denominator = float(int(state_count) * int(pair_count))
            candidates = [
                prefix_sum_vector(repeat.gradient_bins, state_bin, pair_bin, denominator=denominator)
                for repeat in repeats
            ]
            candidate_scalars = [
                prefix_sum_scalar(repeat.scalar_bins, state_bin, pair_bin, denominator=denominator)
                for repeat in repeats
            ]
            for repeat_index, (candidate, candidate_scalar) in enumerate(zip(candidates, candidate_scalars, strict=True)):
                loo_reference = loo_references[repeat_index]
                loo_scalar = float((sum(max_scalars) - max_scalars[repeat_index]) / float(len(repeats) - 1))
                candidate_norm = vector_norm(candidate)
                loo_norm = vector_norm(loo_reference)
                metric_rows.append(
                    {
                        "repeat": int(repeat_index),
                        "states": int(state_count),
                        "pairs": int(pair_count),
                        "hvp_count": int(2 * int(state_count) * int(pair_count)),
                        "a_scalar": float(candidate_scalar),
                        "reference_scalar": loo_scalar,
                        "scalar_symmetric_error": symmetric_relative_error(candidate_scalar, loo_scalar),
                        "gradient_cosine_to_reference": vector_cosine(candidate, loo_reference),
                        "gradient_relative_error": vector_relative_error(candidate, loo_reference),
                        "gradient_norm": candidate_norm,
                        "reference_gradient_norm": loo_norm,
                        "gradient_norm_ratio": candidate_norm / max(loo_norm, 1e-30),
                    }
                )
                for module, ranges in module_ranges.items():
                    candidate_module = _slice_vector(candidate, ranges)
                    reference_module = _slice_vector(loo_reference, ranges)
                    module_rows.append(
                        {
                            "repeat": int(repeat_index),
                            "states": int(state_count),
                            "pairs": int(pair_count),
                            "module": module,
                            "reference_energy_fraction": module_energy[module],
                            "gradient_cosine_to_reference": vector_cosine(candidate_module, reference_module),
                            "gradient_norm_ratio": vector_norm(candidate_module) / max(vector_norm(reference_module), 1e-30),
                        }
                    )
            for left_index in range(len(candidates)):
                for right_index in range(left_index + 1, len(candidates)):
                    pairwise_rows.append(
                        {
                            "states": int(state_count),
                            "pairs": int(pair_count),
                            "left_repeat": int(left_index),
                            "right_repeat": int(right_index),
                            "gradient_cosine": vector_cosine(candidates[left_index], candidates[right_index]),
                            "gradient_norm_ratio": vector_norm(candidates[left_index]) / max(vector_norm(candidates[right_index]), 1e-30),
                        }
                    )
            del candidates

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
        module_summaries = []
        for _module, module_group in relevant_modules.groupby("module"):
            module_summaries.append(
                (
                    quantile(module_group["gradient_cosine_to_reference"], 0.5),
                    quantile(module_group["gradient_cosine_to_reference"], 0.1),
                )
            )
        module_pass = bool(module_summaries) and all(median >= 0.90 and q10 >= 0.80 for median, q10 in module_summaries)
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
        row["cell_stability_pass"] = bool(reference_valid and row["scalar_stability_pass"] and row["gradient_stability_pass"])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["states", "pairs"]).reset_index(drop=True)
    pass_map = {(int(row.states), int(row.pairs)): bool(row.cell_stability_pass) for row in summary.itertuples()}
    persistent_values: list[bool] = []
    for row in summary.itertuples():
        state_count = int(row.states)
        pair_count = int(row.pairs)
        persistent_values.append(
            bool(
                pass_map.get((state_count, pair_count), False)
                and pass_map.get((2 * state_count, pair_count), False)
                and pass_map.get((state_count, 2 * pair_count), False)
            )
        )
    summary["persistent_stability_pass"] = persistent_values
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


def _plot_heatmap(summary: pd.DataFrame, *, value: str, title: str, output_path: Path, fmt: str = ".2f") -> None:
    pivot = summary.pivot(index="states", columns="pairs", values=value).sort_index().sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(int(value)) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(int(value)) for value in pivot.index])
    ax.set_xlabel("probe/batch pairs per state")
    ax.set_ylabel("states per A update")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value_number = values[row, column]
            label = "nan" if not math.isfinite(value_number) else format(value_number, fmt)
            ax.text(column, row, label, ha="center", va="center", color="white" if value_number < np.nanmedian(values) else "black", fontsize=8)
    if 2 in pivot.index and 4 in pivot.columns:
        state_position = list(pivot.index).index(2)
        pair_position = list(pivot.columns).index(4)
        ax.add_patch(
            Rectangle(
                (pair_position - 0.5, state_position - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#ef4444",
                linewidth=2.5,
            )
        )
        ax.text(pair_position, state_position - 0.38, "current", ha="center", va="top", color="#ef4444", fontsize=7, fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_plots(output_dir: Path, summary: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    outputs: list[str] = []
    specs = [
        ("gradient_cosine_median", "Median gradient cosine to independent reference", "gradient_cosine_heatmap.png", ".3f"),
        ("pairwise_cosine_median", "Median repeat-to-repeat gradient cosine", "pairwise_cosine_heatmap.png", ".3f"),
        ("scalar_symmetric_error_median", "Median symmetric A-scalar error", "scalar_error_heatmap.png", ".2f"),
    ]
    for value, title, filename, fmt in specs:
        path = output_dir / filename
        _plot_heatmap(summary, value=value, title=title, output_path=path, fmt=fmt)
        outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for state_count, group in summary.groupby("states", sort=True):
        ordered = group.sort_values("pairs")
        ax.plot(ordered["pairs"], ordered["gradient_cosine_median"], marker="o", label=f"S={int(state_count)}")
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, label="cell threshold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summary["pairs"].unique()), [str(int(value)) for value in sorted(summary["pairs"].unique())])
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("probe/batch pairs per state")
    ax.set_ylabel("median cosine to independent reference")
    ax.set_title("Variant A gradient-direction stability")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = output_dir / "gradient_stability_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    current = metrics.loc[(metrics["states"] == 2) & (metrics["pairs"] == 4)]
    if not current.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
        axes[0].hist(current["gradient_cosine_to_reference"], bins=max(4, len(current) // 2), color="#2563eb", alpha=0.85)
        axes[0].axvline(0.95, color="black", linestyle="--", linewidth=1)
        axes[0].set_title("Current S=2, P=4 gradient cosine")
        axes[0].set_xlabel("cosine to leave-one-out reference")
        axes[1].hist(current["scalar_symmetric_error"], bins=max(4, len(current) // 2), color="#dc2626", alpha=0.85)
        axes[1].axvline(0.10, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Current S=2, P=4 scalar error")
        axes[1].set_xlabel("symmetric relative error")
        fig.tight_layout()
        path = output_dir / "current_2x4_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def _batch_pair_overlap_audit(unit_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (repeat, state_position), group in unit_frame.groupby(["repeat", "state_position"], sort=True):
        ordered = group.sort_values("pair_position")
        for branch in (1, 2):
            offsets = pd.to_numeric(ordered[f"batch_{branch}_offset"], errors="coerce").to_numpy()
            pair_positions = ordered["pair_position"].astype(int).to_numpy()
            train_counts = ordered["batch_train_count"].astype(int).to_numpy()
            batch_sizes = ordered["batch_size_effective"].astype(int).to_numpy()
            for left in range(len(ordered)):
                if not math.isfinite(float(offsets[left])):
                    continue
                left_indices = {
                    (int(offsets[left]) + position) % int(train_counts[left])
                    for position in range(int(batch_sizes[left]))
                }
                for right in range(left + 1, len(ordered)):
                    if not math.isfinite(float(offsets[right])) or int(train_counts[left]) != int(train_counts[right]):
                        continue
                    right_indices = {
                        (int(offsets[right]) + position) % int(train_counts[right])
                        for position in range(int(batch_sizes[right]))
                    }
                    overlap = len(left_indices.intersection(right_indices))
                    rows.append(
                        {
                            "repeat": int(repeat),
                            "state_position": int(state_position),
                            "branch": int(branch),
                            "left_pair": int(pair_positions[left]),
                            "right_pair": int(pair_positions[right]),
                            "pair_distance": int(pair_positions[right] - pair_positions[left]),
                            "overlap_count": int(overlap),
                            "overlap_fraction": float(overlap / max(1, min(len(left_indices), len(right_indices)))),
                        }
                    )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby(["branch", "pair_distance"], as_index=False)
        .agg(
            comparisons=("overlap_fraction", "size"),
            overlap_fraction_mean=("overlap_fraction", "mean"),
            overlap_fraction_median=("overlap_fraction", "median"),
            overlap_fraction_max=("overlap_fraction", "max"),
            exact_duplicate_fraction=("overlap_fraction", lambda values: float((values >= 1.0 - 1e-12).mean())),
        )
    )
    return detail, summary


def _branch_manifest(unit_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    shared_columns = [
        "repeat",
        "step_key",
        "state_position",
        "pair_position",
        "source_weight_index",
        "task_name",
        "tau",
        "state_draw_seed",
        "train_position",
        "latent_hash",
        "batch_train_count",
        "batch_size_effective",
    ]
    for row in unit_frame.to_dict(orient="records"):
        shared = {column: row[column] for column in shared_columns}
        for branch in (1, 2):
            rows.append(
                {
                    **shared,
                    "branch": int(branch - 1),
                    "probe_seed": int(row[f"probe_seed_{branch}"]),
                    "probe_norm": float(row[f"probe_{branch}_norm"]),
                    "probe_hash": str(row[f"probe_{branch}_hash"]),
                    "hvp_norm": float(row[f"h{branch}_norm"]),
                    "hvp_hash": str(row[f"h{branch}_hash"]),
                    "batch_offset": row[f"batch_{branch}_offset"],
                    "batch_hash": str(row[f"batch_{branch}_hash"]),
                    "batch_indices": str(row[f"batch_{branch}_indices"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Variant A scalar and active-decoder gradient estimator stability.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", default="1,2,4,8,16,32")
    parser.add_argument("--pairs", default="1,2,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    state_grid = parse_grid(args.states)
    pair_grid = parse_grid(args.pairs)
    if int(args.repeats) < 2:
        raise ValueError("repeats must be at least 2")
    checkpoint_path = run_dir / "vae_checkpoint.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if str(args.expected_checkpoint_sha256) and checkpoint_hash != str(args.expected_checkpoint_sha256):
        raise ValueError(f"checkpoint hash mismatch: observed={checkpoint_hash} expected={args.expected_checkpoint_sha256}")
    pool_hash = sha256_file(run_dir / "weight_pool.pt")
    records_hash = sha256_file(run_dir / "weight_pool_records.csv")
    if pool_hash != EXPECTED_POOL_SHA256 or records_hash != EXPECTED_RECORDS_SHA256:
        raise ValueError(
            "accepted baseline source hash mismatch: "
            f"pool={pool_hash} expected_pool={EXPECTED_POOL_SHA256} "
            f"records={records_hash} expected_records={EXPECTED_RECORDS_SHA256}"
        )
    acceptance = json.loads((run_dir / "baseline_acceptance.json").read_text(encoding="utf-8"))
    if not bool(acceptance.get("passed")):
        raise ValueError("baseline_acceptance.json does not have passed=true")
    code_paths = [
        Path(__file__).resolve(),
        ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py",
        ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py",
    ]
    code_hashes = {str(path): sha256_file(path) for path in code_paths}
    protocol_hash = sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else "missing"
    repository = git_metadata()
    resolved_config = {
        "protocol_id": PROTOCOL_ID,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_hash,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "states": state_grid,
        "pairs": pair_grid,
        "repeats": int(args.repeats),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "checkpoint_sha256": checkpoint_hash,
        "weight_pool_sha256": pool_hash,
        "weight_records_sha256": records_hash,
        "baseline_acceptance_passed": True,
        "estimator_scope": "local",
        "hvp_mode": "stopped_composite",
        "probe_scale": 1.0,
        "loss_clip": 0.0,
        "gradient_damping": 0.0,
        "z_semantics": "encoded_mu_detached",
        "state_population": "checkpoint_train_indices_uniform_with_replacement",
        "cache_mode": "read_only_existing_checkpoint_and_weight_pool",
        "code_sha256": code_hashes,
        "repository": repository,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[a_stability] start resolved_config={json.dumps(resolved_config, sort_keys=True)}", flush=True)
    started = time.perf_counter()
    print("[a_stability] stage=data_and_model_load", flush=True)
    run = _load_run(run_dir, device=device)
    if int(run.cfg.vae_hidden_dim) != 2048 or int(run.cfg.latent_dim) != 512:
        raise ValueError(
            f"unexpected architecture hidden_dim={run.cfg.vae_hidden_dim} latent_dim={run.cfg.latent_dim}"
        )
    if str(run.cfg.tiny_bigvae_output_mode).strip().lower() != "direct" or torch_dtype(run.cfg) != torch.float32:
        raise ValueError(
            f"unexpected output mode/dtype: mode={run.cfg.tiny_bigvae_output_mode} dtype={torch_dtype(run.cfg)}"
        )
    if int(run.train_indices.numel()) != 11021:
        raise ValueError(f"unexpected checkpoint train population: {int(run.train_indices.numel())} != 11021")
    print(
        "[a_stability] loaded "
        f"dtype={torch_dtype(run.cfg)} hidden_dim={run.cfg.vae_hidden_dim} latent_dim={run.cfg.latent_dim} "
        f"weight_rows={int(run.weights.shape[0])} train_rows={int(run.train_indices.numel())}",
        flush=True,
    )
    print("[a_stability] stage=atomic_decomposition_preflight", flush=True)
    preflight, active = _preflight_decomposition(
        run,
        batch_size=int(args.batch_size),
        seed=stable_uint63("preflight", args.seed),
        device=device,
    )
    module_ranges, parameter_rows = _module_slices(active)
    active_names = {name for name, _parameter in active}
    excluded = [(name, parameter) for name, parameter in run.vae.named_parameters() if name not in active_names]
    pd.DataFrame(parameter_rows).to_csv(output_dir / "active_parameters.csv", index=False)
    active_payload = {
        "active_parameter_count": int(sum(parameter.numel() for _name, parameter in active)),
        "active_parameter_names": [name for name, _parameter in active],
        "excluded_parameter_names": preflight["excluded_parameter_names"],
        "ordered_list_sha256": hashlib.sha256(
            json.dumps(parameter_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "parameters": parameter_rows,
    }
    (output_dir / "active_decoder_parameters.json").write_text(
        json.dumps(active_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[a_stability] preflight={json.dumps(preflight, sort_keys=True)}", flush=True)
    print("[a_stability] stage=nested_monte_carlo", flush=True)
    unit_rows: list[dict[str, Any]] = []
    accumulators: list[RepeatAccumulator] = []
    for repeat in range(int(args.repeats)):
        repeat_started = time.perf_counter()
        print(f"[a_stability] repeat_start repeat={repeat + 1}/{int(args.repeats)}", flush=True)
        accumulator = _run_repeat(
            run,
            repeat=repeat,
            state_grid=state_grid,
            pair_grid=pair_grid,
            active=active,
            excluded=excluded,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            device=device,
            log_every=int(args.log_every),
            unit_rows=unit_rows,
        )
        accumulators.append(accumulator)
        pd.DataFrame(unit_rows).to_csv(output_dir / "atomic_pair_samples.partial.csv", index=False)
        print(
            f"[a_stability] repeat_done repeat={repeat + 1}/{int(args.repeats)} "
            f"elapsed_sec={time.perf_counter() - repeat_started:.1f} total_elapsed_sec={time.perf_counter() - started:.1f}",
            flush=True,
        )
    unit_frame = pd.DataFrame(unit_rows)
    unit_path = output_dir / "atomic_pair_samples.csv"
    unit_frame.to_csv(unit_path, index=False)
    partial_path = output_dir / "atomic_pair_samples.partial.csv"
    if partial_path.exists():
        partial_path.unlink()
    branches = _branch_manifest(unit_frame)
    branches.to_csv(output_dir / "branch_manifest.csv", index=False)
    batch_overlap, batch_overlap_summary = _batch_pair_overlap_audit(unit_frame)
    batch_overlap.to_csv(output_dir / "batch_pair_overlap.csv", index=False)
    batch_overlap_summary.to_csv(output_dir / "batch_pair_overlap_summary.csv", index=False)

    print("[a_stability] stage=reference_and_grid_analysis", flush=True)
    metrics, summary, pairwise, module_metrics, reference_bundle = _analyze(
        accumulators,
        state_grid=state_grid,
        pair_grid=pair_grid,
        module_ranges=module_ranges,
    )
    metrics.to_csv(output_dir / "replicate_metrics.csv", index=False)
    summary.to_csv(output_dir / "stability_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_gradient_cosines.csv", index=False)
    module_metrics.to_csv(output_dir / "module_gradient_metrics.csv", index=False)
    pd.DataFrame(reference_bundle["split_rows"]).to_csv(output_dir / "reference_split_checks.csv", index=False)
    (output_dir / "reference.json").write_text(
        json.dumps(reference_bundle["reference"], indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    state_rows = []
    for accumulator in accumulators:
        for position, source_index in enumerate(accumulator.state_indices):
            record = run.records.iloc[int(source_index)]
            state_rows.append(
                {
                    "repeat": accumulator.repeat,
                    "step_key": accumulator.step_key,
                    "state_position": position,
                    "state_draw_seed": accumulator.state_draw_seeds[position],
                    "train_position": accumulator.train_positions[position],
                    "source_weight_index": source_index,
                    "task_name": str(record.get("task_name", "")),
                    "tau": float(record.get("tau", 1.0)),
                    "source_lr": float(record.get("source_lr", float("nan"))),
                    "source_step": float(record.get("step", float("nan"))),
                    "latent_hash": accumulator.latent_hashes[position],
                }
            )
    state_frame = pd.DataFrame(state_rows)
    state_frame["duplicate_within_repeat"] = state_frame.duplicated(["repeat", "source_weight_index"], keep=False)
    state_frame["duplicate_across_audit"] = state_frame.duplicated(["source_weight_index"], keep=False)
    state_frame.to_csv(output_dir / "sampled_states.csv", index=False)

    print("[a_stability] stage=plotting", flush=True)
    plot_paths = _write_plots(output_dir, summary, metrics)
    checkpoint_hash_after = sha256_file(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash:
        raise RuntimeError("source checkpoint changed during read-only audit")
    reference = reference_bundle["reference"]
    current_rows = summary.loc[(summary["states"] == 2) & (summary["pairs"] == 4)]
    current = None if current_rows.empty else current_rows.iloc[0].to_dict()
    expected_pairs = int(args.repeats) * int(state_grid[-1]) * int(pair_grid[-1])
    expected_branches = 2 * expected_pairs
    validity_checks = {
        "checkpoint_hash_match": checkpoint_hash == EXPECTED_CHECKPOINT_SHA256,
        "pool_hash_match": pool_hash == EXPECTED_POOL_SHA256,
        "records_hash_match": records_hash == EXPECTED_RECORDS_SHA256,
        "baseline_acceptance_passed": bool(acceptance.get("passed")),
        "train_population_count_match": int(run.train_indices.numel()) == 11021,
        "atomic_pair_row_count_match": int(len(unit_frame)) == expected_pairs,
        "atomic_pair_keys_unique": not bool(unit_frame.duplicated(["repeat", "state_position", "pair_position"]).any()),
        "branch_row_count_match": int(len(branches)) == expected_branches,
        "branch_keys_unique": not bool(branches.duplicated(["repeat", "state_position", "pair_position", "branch"]).any()),
        "probe_seeds_unique": int(branches["probe_seed"].nunique()) == expected_branches,
        "state_draw_seeds_unique": int(state_frame["state_draw_seed"].nunique()) == int(len(state_frame)),
        "all_atomic_scalars_finite": bool(np.isfinite(unit_frame["a_scalar"]).all()),
        "all_atomic_gradient_norms_finite": bool(np.isfinite(unit_frame["gradient_norm"]).all()),
        "all_hvp_norms_finite": bool(np.isfinite(unit_frame[["h1_norm", "h2_norm"]].to_numpy()).all()),
        "atomic_accumulation_preflight_passed": bool(
            preflight["gradient_cosine"] >= 0.99999
            and preflight["gradient_relative_error"] <= 5e-5
            and preflight["scalar_abs_error"] <= 5e-5
        ),
        "hvp_equivalence_preflight_passed": bool(
            preflight["stopped_vs_autograd_hvp_cosine"] >= 0.99999
            and preflight["stopped_vs_autograd_hvp_relative_error"] <= 5e-5
        ),
        "checkpoint_unchanged": checkpoint_hash_after == checkpoint_hash,
        "summary_cell_count_match": int(len(summary)) == int(len(state_grid) * len(pair_grid)),
        "reported_metrics_finite": bool(
            np.isfinite(
                summary[
                    [
                        "scalar_symmetric_error_median",
                        "gradient_cosine_median",
                        "gradient_norm_ratio_median",
                        "pairwise_cosine_median",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
    }
    validity = {
        "passed": bool(all(validity_checks.values())),
        "checks": validity_checks,
        "expected_atomic_pairs": expected_pairs,
        "expected_hvp_branches": expected_branches,
        "observed_atomic_pairs": int(len(unit_frame)),
        "observed_hvp_branches": int(len(branches)),
        "duplicate_state_slots": int(state_frame["duplicate_within_repeat"].sum()),
        "checkpoint_sha256_before": checkpoint_hash,
        "checkpoint_sha256_after": checkpoint_hash_after,
    }
    (output_dir / "validity.json").write_text(json.dumps(validity, indent=2, sort_keys=True), encoding="utf-8")
    if not validity["passed"]:
        raise RuntimeError(f"audit validity checks failed: {json.dumps(validity, sort_keys=True)}")
    decision = {
        "reference_valid": bool(reference["reference_valid"]),
        "selected_persistent_cell": reference["selected_persistent_cell"],
        "current_2x4": current,
        "budget_censored": reference["selected_persistent_cell"] is None,
        "requires_extension": reference["selected_persistent_cell"] is None,
        "cells": summary[
            [
                "states",
                "pairs",
                "scalar_stability_pass",
                "gradient_stability_pass",
                "module_stability_pass",
                "cell_stability_pass",
                "persistent_stability_pass",
            ]
        ].to_dict(orient="records"),
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "elapsed_sec": time.perf_counter() - started,
        "resolved_config": resolved_config,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "validity": validity,
        "active_parameter_count": int(sum(parameter.numel() for _name, parameter in active)),
        "active_parameter_names": [name for name, _parameter in active],
        "reference": reference,
        "current_2x4": current,
        "outputs": {
            "resolved_config": str(output_dir / "resolved_config.json"),
            "preflight": str(output_dir / "preflight.json"),
            "active_parameters": str(output_dir / "active_parameters.csv"),
            "active_decoder_parameters": str(output_dir / "active_decoder_parameters.json"),
            "atomic_pair_samples": str(unit_path),
            "branch_manifest": str(output_dir / "branch_manifest.csv"),
            "sampled_states": str(output_dir / "sampled_states.csv"),
            "batch_pair_overlap": str(output_dir / "batch_pair_overlap.csv"),
            "batch_pair_overlap_summary": str(output_dir / "batch_pair_overlap_summary.csv"),
            "replicate_metrics": str(output_dir / "replicate_metrics.csv"),
            "stability_summary": str(output_dir / "stability_summary.csv"),
            "pairwise_gradient_cosines": str(output_dir / "pairwise_gradient_cosines.csv"),
            "module_gradient_metrics": str(output_dir / "module_gradient_metrics.csv"),
            "reference_split_checks": str(output_dir / "reference_split_checks.csv"),
            "reference": str(output_dir / "reference.json"),
            "validity": str(output_dir / "validity.json"),
            "decision": str(output_dir / "decision.json"),
            "plots": plot_paths,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print(
        "[a_stability] done "
        f"elapsed_sec={manifest['elapsed_sec']:.1f} reference_valid={reference['reference_valid']} "
        f"selected={reference['selected_persistent_cell']} current_2x4={current} "
        f"manifest={output_dir / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
