from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import replace
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
    _batch_indices,
    _latent_hvp,
    _probe_like,
    _task_set_for_record,
)
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _load_run,
    _probe_cfg,
    sha256_file,
    sha256_tensor,
)
from scripts.run_one_state_a_full_burg_armijo import (
    DEFAULT_BETA,
    _adam_proposal,
    _evaluate_candidate,
    _objective,
    _set_parameters,
)
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    OUTPUT_ROOT,
    STATE_BANK,
    _materialize_metric,
    _pair_losses,
    _train_generators,
)


PROTOCOL_ID = "one_state_a_full_burg_direction_mismatch_iteration2_v1"
ITERATION1_CHECKPOINT = OUTPUT_ROOT / "armijo_iteration1_100proposals/final_checkpoint.pt"
ITERATION1_DIAGNOSTICS = OUTPUT_ROOT / "armijo_iteration1_100proposals/proposal_diagnostics.csv"
ITERATION1_STATES = OUTPUT_ROOT / "armijo_iteration1_100proposals/state_objective_curve.csv"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration2_crossed_direction_audit"
FD_ALPHAS = (1.0 / 128.0, 1.0 / 256.0)
ARMIJO_ALPHAS = tuple(2.0**-index for index in range(11))
Vector = list[torch.Tensor]


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _named_tensor_hash(names: Sequence[str], values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in zip(names, values, strict=True):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _mapping_hash(values: Mapping[str, torch.Tensor]) -> str:
    names = sorted(values)
    return _named_tensor_hash(names, [values[name] for name in names])


def _zeros_like(active: Sequence[torch.nn.Parameter], *, dtype: torch.dtype | None = None) -> Vector:
    return [torch.zeros_like(parameter, dtype=dtype or parameter.dtype) for parameter in active]


def _clone_vector(values: Sequence[torch.Tensor], *, dtype: torch.dtype | None = None) -> Vector:
    return [value.detach().to(dtype=dtype or value.dtype).clone() for value in values]


def _add_(destination: Vector, source: Sequence[torch.Tensor | None], *, alpha: float = 1.0) -> None:
    with torch.no_grad():
        for target, value in zip(destination, source, strict=True):
            if value is not None:
                target.add_(value.detach().to(dtype=target.dtype), alpha=float(alpha))


def _scaled(values: Sequence[torch.Tensor], scale: float, active: Sequence[torch.nn.Parameter]) -> Vector:
    return [
        (value.detach().to(device=parameter.device, dtype=parameter.dtype) * float(scale)).clone()
        for value, parameter in zip(values, active, strict=True)
    ]


def _dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    total = torch.zeros((), device=left[0].device, dtype=torch.float64)
    for a, b in zip(left, right, strict=True):
        total += (a.detach().double() * b.detach().double()).sum()
    return float(total.cpu())


def _norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, _dot(values, values)))


def _vector_is_finite(values: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def _numeric_frame_is_finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def _cosine(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator <= 0.0:
        return float("nan")
    return _dot(left, right) / denominator


def _relative_error(value: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]) -> float:
    denominator = _norm(reference)
    if denominator <= 0.0:
        return float("nan")
    difference = [a.detach().double() - b.detach().double() for a, b in zip(value, reference, strict=True)]
    return _norm(difference) / denominator


def _negative_normalized(
    gradient: Sequence[torch.Tensor],
    *,
    target_norm: float,
    active: Sequence[torch.nn.Parameter],
) -> Vector:
    gradient_norm = _norm(gradient)
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise RuntimeError(f"invalid gradient norm {gradient_norm}")
    return _scaled(gradient, -float(target_norm) / gradient_norm, active)


def _normalize_direction(
    direction: Sequence[torch.Tensor],
    *,
    target_norm: float,
    active: Sequence[torch.nn.Parameter],
) -> Vector:
    direction_norm = _norm(direction)
    if not math.isfinite(direction_norm) or direction_norm <= 0.0:
        raise RuntimeError(f"invalid direction norm {direction_norm}")
    return _scaled(direction, float(target_norm) / direction_norm, active)


def _gradient_from_losses(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    beta: float,
    proposal: int,
    pairs: int,
    block_size: int,
    label: str,
) -> tuple[Vector, list[Vector], list[dict[str, float | int | str]]]:
    if pairs % block_size != 0:
        raise ValueError("pairs must be divisible by block_size")
    total = _zeros_like(active)
    block = _zeros_like(active)
    blocks: list[Vector] = []
    rows: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for pair in range(pairs):
        generator_1, generator_2 = _train_generators(proposal, pair)
        a_loss, b_loss, stats = _pair_losses(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            pair_index=pair,
            probe_generator_1=generator_1,
            probe_generator_2=generator_2,
            burg_matrix_gradient=burg_gradient,
        )
        combined = a_loss + float(beta) * b_loss
        gradients = torch.autograd.grad(combined, active, retain_graph=False, allow_unused=True)
        _add_(total, gradients)
        _add_(block, gradients)
        rows.append(
            {
                "mode": label,
                "pair": pair,
                "a_loss": float(a_loss.detach().cpu()),
                "b_pseudo_loss": float(b_loss.detach().cpu()),
                "combined_pseudo_loss": float(combined.detach().cpu()),
                **stats,
            }
        )
        if (pair + 1) % block_size == 0:
            block_index = (pair + 1) // block_size - 1
            block_mean = _scaled(block, 1.0 / float(block_size), active)
            blocks.append(block_mean)
            block = _zeros_like(active)
            print(
                f"[i2-audit] stage=random-gradient mode={label} "
                f"block={block_index + 1}/{pairs // block_size} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        del a_loss, b_loss, combined, gradients
    return _scaled(total, 1.0 / float(pairs), active), blocks, rows


def _full_basis_gradient(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    k_cotangent: torch.Tensor,
    mode: str,
    basis_count: int,
) -> tuple[Vector, dict[str, float | int | str]]:
    mode_cfg = replace(cfg, vae_precond_hvp_mode=mode)
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    dim = int(z.numel())
    if basis_count <= 0 or basis_count > dim:
        raise ValueError(f"basis_count must be in [1,{dim}], got {basis_count}")
    accumulated = _zeros_like(active, dtype=torch.float64)
    unused_counts = np.zeros(len(active), dtype=np.int64)
    memory_allocated_trace: list[int] = []
    started = time.perf_counter()
    for index in range(basis_count):
        basis = torch.zeros_like(z)
        basis[index] = 1.0
        hvp = _latent_hvp(
            mode_cfg,
            run.vae,
            run.normalizer,
            z,
            basis,
            task_set=task_set,
            spec=run.spec,
            tau=tau,
            batch_indices=None,
        )
        scalar = torch.dot(k_cotangent[index].to(dtype=hvp.dtype), hvp)
        gradients = torch.autograd.grad(scalar, active, retain_graph=False, allow_unused=True)
        for parameter_index, value in enumerate(gradients):
            if value is None:
                unused_counts[parameter_index] += 1
        _add_(accumulated, gradients)
        memory_allocated_trace.append(
            int(torch.cuda.memory_allocated(z.device)) if z.device.type == "cuda" else 0
        )
        if index < 3 or (index + 1) % 16 == 0 or index + 1 == basis_count:
            elapsed = time.perf_counter() - started
            print(
                f"[i2-audit] stage=full-basis mode={mode} row={index + 1}/{basis_count} "
                f"rate={(index + 1) / max(elapsed, 1e-12):.2f}row/s elapsed={elapsed:.1f}s",
                flush=True,
            )
        del basis, hvp, scalar, gradients
    window = max(1, min(16, basis_count // 2))
    early_memory = float(np.median(memory_allocated_trace[:window]))
    late_memory = float(np.median(memory_allocated_trace[-window:]))
    return accumulated, {
        "mode": mode,
        "basis_count": basis_count,
        "dimension": dim,
        "is_complete_basis": int(basis_count == dim),
        "unused_parameter_tensors_any_row": int((unused_counts > 0).sum()),
        "unused_parameter_tensors_all_rows": int((unused_counts == basis_count).sum()),
        "elapsed_sec": time.perf_counter() - started,
        "gradient_norm": _norm(accumulated),
        "memory_allocated_bytes_per_row": memory_allocated_trace,
        "memory_early_window_median_bytes": early_memory,
        "memory_late_window_median_bytes": late_memory,
        "memory_window_growth_bytes": max(0.0, late_memory - early_memory),
        "memory_growth_gate_bytes": 64 * 1024 * 1024,
        "memory_growth_gate_pass": bool(late_memory - early_memory <= 64 * 1024 * 1024),
    }


def _aggregated_basis_gradient(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    k_cotangent: torch.Tensor,
    basis_count: int,
) -> Vector:
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    scalars: list[torch.Tensor] = []
    for index in range(basis_count):
        basis = torch.zeros_like(z)
        basis[index] = 1.0
        hvp = _latent_hvp(
            cfg,
            run.vae,
            run.normalizer,
            z,
            basis,
            task_set=task_set,
            spec=run.spec,
            tau=tau,
            batch_indices=None,
        )
        scalars.append(torch.dot(k_cotangent[index].to(dtype=hvp.dtype), hvp))
    total = torch.stack(scalars).sum()
    gradients = torch.autograd.grad(total, active, retain_graph=False, allow_unused=True)
    result = _zeros_like(active, dtype=torch.float64)
    _add_(result, gradients)
    del scalars, total, gradients
    return result


def _forward_equivalence(
    *,
    cfg_stopped: Any,
    cfg_naive: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    hessian: torch.Tensor,
    probes: int,
) -> list[dict[str, float | int]]:
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    rows: list[dict[str, float | int]] = []
    for probe_index in range(probes):
        generator = torch.Generator(device="cpu").manual_seed(920_000 + probe_index)
        probe = _probe_like(z, generator=generator, scale=1.0)
        stopped = _latent_hvp(
            cfg_stopped,
            run.vae,
            run.normalizer,
            z,
            probe,
            task_set=task_set,
            spec=run.spec,
            tau=tau,
            batch_indices=None,
        ).detach()
        naive = _latent_hvp(
            cfg_naive,
            run.vae,
            run.normalizer,
            z,
            probe,
            task_set=task_set,
            spec=run.spec,
            tau=tau,
            batch_indices=None,
        ).detach()
        dense = hessian.transpose(0, 1) @ probe
        dense_norm = dense.double().norm().clamp_min(1e-30)
        stopped_error = float(((stopped - dense).double().norm() / dense_norm).cpu())
        naive_error = float(((naive - dense).double().norm() / dense_norm).cpu())
        cross_error = float(((stopped - naive).double().norm() / dense_norm).cpu())
        rows.append(
            {
                "probe": probe_index,
                "stopped_to_dense_relative_error": stopped_error,
                "naive_to_dense_relative_error": naive_error,
                "stopped_to_naive_relative_error": cross_error,
                "stopped_to_dense_cosine": _cosine([stopped], [dense]),
                "naive_to_dense_cosine": _cosine([naive], [dense]),
            }
        )
    return rows


def _matrix_objective(hessian: torch.Tensor, *, beta: float, epsilon: float) -> float:
    h64 = hessian.double()
    metric = h64 @ h64.transpose(0, 1)
    metric = 0.5 * (metric + metric.transpose(0, 1))
    eig = torch.linalg.eigvalsh(metric).clamp_min(0.0)
    a = (eig - 1.0).square().mean()
    r = (eig + float(epsilon)) / (1.0 + float(epsilon))
    b = (r - r.log() - 1.0).mean()
    return float((a + float(beta) * b).cpu())


def _h_space_fd_check(
    hessian: torch.Tensor,
    k_cotangent: torch.Tensor,
    *,
    beta: float,
    epsilon: float,
) -> list[dict[str, float]]:
    generator = torch.Generator(device="cpu").manual_seed(771_911)
    direction = torch.randn(tuple(hessian.shape), generator=generator, dtype=torch.float64).to(hessian.device)
    direction.mul_(hessian.double().norm() / direction.norm().clamp_min(1e-30))
    analytic = float((k_cotangent.double() * direction).sum().cpu())
    rows: list[dict[str, float]] = []
    for delta in (1e-5, 5e-6):
        positive = _matrix_objective(hessian.double() + delta * direction, beta=beta, epsilon=epsilon)
        negative = _matrix_objective(hessian.double() - delta * direction, beta=beta, epsilon=epsilon)
        slope = (positive - negative) / (2.0 * delta)
        relative_error = abs(slope - analytic) / max(abs(slope), abs(analytic), 1e-30)
        rows.append(
            {
                "delta": delta,
                "analytic_slope": analytic,
                "central_slope": slope,
                "relative_error": relative_error,
            }
        )
    return rows


def _adam_direction(
    *,
    gradient: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
    exp_avg: Sequence[torch.Tensor],
    exp_avg_sq: Sequence[torch.Tensor],
    accepted_step: int,
    gradient_clip: float,
    lr: float,
) -> tuple[Vector, dict[str, float]]:
    gradient_norm = _norm(gradient)
    clip_factor = min(1.0, float(gradient_clip) / max(gradient_norm, 1e-30))
    with torch.no_grad():
        for parameter, value in zip(active, gradient, strict=True):
            parameter.grad = value.detach().to(device=parameter.device, dtype=parameter.dtype) * clip_factor
    _next_avg, _next_avg_sq, displacement = _adam_proposal(
        active=list(active),
        exp_avg=list(exp_avg),
        exp_avg_sq=list(exp_avg_sq),
        accepted_step=accepted_step,
        lr=lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    for parameter in active:
        parameter.grad = None
    return displacement, {
        "input_gradient_norm": gradient_norm,
        "clip_factor": clip_factor,
        "unnormalized_direction_norm": _norm(displacement),
    }


def _torch_adam_reference_direction(
    *,
    gradient: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
    exp_avg: Sequence[torch.Tensor],
    exp_avg_sq: Sequence[torch.Tensor],
    accepted_step: int,
    gradient_clip: float,
    lr: float,
) -> tuple[Vector, Vector]:
    gradient_norm = _norm(gradient)
    clip_factor = min(1.0, float(gradient_clip) / max(gradient_norm, 1e-30))
    copies = [torch.nn.Parameter(parameter.detach().clone(), requires_grad=True) for parameter in active]
    optimizer = torch.optim.Adam(copies, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    for copied, value, avg, avg_sq in zip(copies, gradient, exp_avg, exp_avg_sq, strict=True):
        copied.grad = value.detach().to(device=copied.device, dtype=copied.dtype) * clip_factor
        state = optimizer.state[copied]
        state["step"] = torch.tensor(float(accepted_step), dtype=torch.float32)
        state["exp_avg"] = avg.detach().to(device=copied.device, dtype=copied.dtype).clone()
        state["exp_avg_sq"] = avg_sq.detach().to(device=copied.device, dtype=copied.dtype).clone()
    before = [value.detach().clone() for value in copies]
    optimizer.step()
    after = [value.detach().clone() for value in copies]
    displacement = [value.detach() - origin for value, origin in zip(copies, before, strict=True)]
    del optimizer, copies, before
    return displacement, after


def _gradient_row(
    name: str,
    value: Sequence[torch.Tensor],
    exact: Sequence[torch.Tensor],
    population_reference: Sequence[torch.Tensor],
    population_name: str,
) -> dict[str, float | str]:
    negative_value = [(-tensor) for tensor in value]
    return {
        "gradient": name,
        "norm": _norm(value),
        "cosine_to_exact": _cosine(value, exact),
        "relative_error_to_exact": _relative_error(value, exact),
        "population_reference": population_name,
        "cosine_to_population_reference": _cosine(value, population_reference),
        "relative_error_to_population_reference": _relative_error(value, population_reference),
        "negative_gradient_exact_dot": _dot(exact, negative_value),
    }


def _replay_iteration1_state(
    *,
    through_proposal: int,
    diagnostics: pd.DataFrame,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    beta: float,
    epsilon: float,
    gradient_clip: float,
    lr: float,
    hessian_chunk_size: int,
) -> tuple[Vector, Vector, int, list[dict[str, float | int]]]:
    exp_avg = _zeros_like(active)
    exp_avg_sq = _zeros_like(active)
    accepted_step = 0
    burg_gradient: torch.Tensor | None = None
    replay_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for proposal in range(1, through_proposal + 1):
        if burg_gradient is None:
            metric, _h, _m, _eig, burg_gradient = _materialize_metric(
                run=run,
                z=z,
                record=record,
                epsilon=epsilon,
                hessian_chunk_size=hessian_chunk_size,
            )
            current_f = _objective(metric, beta)
            del _h, _m, _eig
        row = diagnostics.loc[diagnostics["proposal"].eq(proposal)]
        if len(row) != 1:
            raise RuntimeError(f"missing iteration1 diagnostic row for proposal {proposal}")
        frozen = row.iloc[0]
        expected_current = float(frozen["current_f"])
        if abs(current_f - expected_current) > 2e-4:
            raise RuntimeError(
                f"replay diverged before proposal {proposal}: {current_f} vs {expected_current}"
            )
        a_losses: list[torch.Tensor] = []
        b_losses: list[torch.Tensor] = []
        for pair in range(4):
            generator_1, generator_2 = _train_generators(proposal, pair)
            a_loss, b_loss, _stats = _pair_losses(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                pair_index=pair,
                probe_generator_1=generator_1,
                probe_generator_2=generator_2,
                burg_matrix_gradient=burg_gradient,
            )
            a_losses.append(a_loss)
            b_losses.append(b_loss)
        mean_a = torch.stack(a_losses).mean()
        mean_b = torch.stack(b_losses).mean()
        gradients_a = torch.autograd.grad(mean_a, active, retain_graph=True, allow_unused=True)
        gradients_b = torch.autograd.grad(mean_b, active, retain_graph=False, allow_unused=True)
        with torch.no_grad():
            for parameter, grad_a, grad_b in zip(active, gradients_a, gradients_b, strict=True):
                if grad_a is None and grad_b is None:
                    parameter.grad = None
                    continue
                if grad_a is None:
                    grad_a = torch.zeros_like(grad_b)
                if grad_b is None:
                    grad_b = torch.zeros_like(grad_a)
                parameter.grad = grad_a + float(beta) * grad_b
        total_norm = float(torch.nn.utils.clip_grad_norm_(active, max_norm=gradient_clip).detach().cpu())
        clip_factor = min(1.0, float(gradient_clip) / max(total_norm, 1e-30))
        next_avg, next_avg_sq, displacement = _adam_proposal(
            active=list(active),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            accepted_step=accepted_step,
            lr=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        base = [parameter.detach().clone() for parameter in active]
        accepted_alpha = float(frozen["accepted_alpha"])
        if accepted_alpha > 0.0:
            _set_parameters(list(active), base, displacement, accepted_alpha)
            exp_avg = next_avg
            exp_avg_sq = next_avg_sq
            accepted_step += 1
            burg_gradient = None
        else:
            _set_parameters(list(active), base, displacement, 0.0)
        for parameter in active:
            parameter.grad = None
        replay_rows.append(
            {
                "proposal": proposal,
                "accepted_alpha": accepted_alpha,
                "accepted_step_after": accepted_step,
                "current_f": current_f,
                "expected_current_f": expected_current,
                "current_f_abs_error": abs(current_f - expected_current),
                "raw_gradient_norm": total_norm,
                "clip_factor": clip_factor,
                "proposal_norm": _norm(displacement),
                "elapsed_sec": time.perf_counter() - started,
            }
        )
        if proposal <= 3 or proposal % 10 == 0 or proposal == through_proposal:
            print(
                f"[i2-audit] stage=replay proposal={proposal}/{through_proposal} "
                f"accepted_step={accepted_step} alpha={accepted_alpha:g} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        del a_losses, b_losses, mean_a, mean_b, gradients_a, gradients_b, displacement, base
    return exp_avg, exp_avg_sq, accepted_step, replay_rows


def _directional_fd(
    *,
    name: str,
    direction: Sequence[torch.Tensor],
    exact_gradient: Sequence[torch.Tensor],
    current_f: float,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    base: Sequence[torch.Tensor],
    beta: float,
    epsilon: float,
    hessian_chunk_size: int,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    analytic = _dot(exact_gradient, direction)
    rows: list[dict[str, float | str]] = []
    slopes: list[float] = []
    errors: list[float] = []
    ordering: list[int] = []
    for alpha in FD_ALPHAS:
        positive = _evaluate_candidate(
            run=run,
            z=z,
            record=dict(record),
            active=list(active),
            base=list(base),
            displacement=list(direction),
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            hessian_chunk_size=hessian_chunk_size,
        )
        negative = _evaluate_candidate(
            run=run,
            z=z,
            record=dict(record),
            active=list(active),
            base=list(base),
            displacement=list(direction),
            alpha=-alpha,
            beta=beta,
            epsilon=epsilon,
            hessian_chunk_size=hessian_chunk_size,
        )
        slope = (float(positive["true_objective"]) - float(negative["true_objective"])) / (2.0 * alpha)
        relative_error = abs(slope - analytic) / max(abs(slope), abs(analytic), 1e-12)
        ordered = int(float(positive["true_objective"]) < current_f < float(negative["true_objective"]))
        slopes.append(slope)
        errors.append(relative_error)
        ordering.append(ordered)
        rows.append(
            {
                "direction": name,
                "alpha": alpha,
                "parameter_radius": alpha * _norm(direction),
                "f_negative": float(negative["true_objective"]),
                "f_current": current_f,
                "f_positive": float(positive["true_objective"]),
                "central_slope": slope,
                "analytic_slope": analytic,
                "relative_error": relative_error,
                "descent_ordering": ordered,
            }
        )
    _set_parameters(list(active), list(base), list(direction), 0.0)
    summary: dict[str, float | str] = {
        "direction": name,
        "direction_norm": _norm(direction),
        "analytic_slope": analytic,
        "descent_cosine": _cosine(direction, [(-tensor) for tensor in exact_gradient]),
        "slope_h128": slopes[0],
        "slope_h256": slopes[1],
        "fd_relative_error_h128": errors[0],
        "fd_relative_error_h256": errors[1],
        "descent_ordering_h128": ordering[0],
        "descent_ordering_h256": ordering[1],
    }
    return summary, rows


def _plot_results(
    *,
    gradients: pd.DataFrame,
    directions: pd.DataFrame,
    fd: pd.DataFrame,
    armijo: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    compact = gradients.loc[~gradients["gradient"].str.contains("block[1-7]", regex=True)].copy()
    axes[0, 0].barh(compact["gradient"], compact["cosine_to_exact"], color="#2563eb")
    axes[0, 0].axvline(0.0, color="black", linewidth=1.0)
    axes[0, 0].axvline(0.9, color="#16a34a", linestyle="--", linewidth=1.0)
    axes[0, 0].set(xlabel="cosine to exact naive full-basis gradient", title="Gradient agreement")

    positions = np.arange(len(directions))
    width = 0.25
    axes[0, 1].bar(positions - width, directions["analytic_slope"], width=width, label="analytic")
    axes[0, 1].bar(positions, directions["slope_h128"], width=width, label="FD h=1/128")
    axes[0, 1].bar(positions + width, directions["slope_h256"], width=width, label="FD h=1/256")
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_xticks(positions, directions["direction"], rotation=55, ha="right")
    axes[0, 1].set(ylabel="dF / d alpha (equal direction norm)", title="True-F directional slopes")
    axes[0, 1].set_yscale("symlog", linthresh=0.1)
    axes[0, 1].legend()

    blocks = gradients.loc[gradients["gradient"].str.contains("_p4_block")].copy()
    for mode, frame in blocks.groupby(blocks["gradient"].str.split("_p4_").str[0]):
        frame = frame.sort_values("gradient")
        axes[1, 0].plot(np.arange(len(frame)), frame["cosine_to_exact"], marker="o", label=mode)
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 0].set(xlabel="common P4 block", ylabel="gradient cosine to exact", title="Probe-block variability")
    axes[1, 0].legend()

    axes[1, 1].plot(armijo["alpha"], armijo["true_objective"], marker="o", label="exact raw direction")
    axes[1, 1].axhline(float(armijo.iloc[0]["current_f"]), color="black", linewidth=1.0, label="current F")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(xlabel="fraction of reference-norm step", ylabel="dense F", title="Noncommitting exact-gradient line profile")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-position", type=int, default=2)
    parser.add_argument("--replay-through-proposal", type=int, default=89)
    parser.add_argument("--proposal", type=int, default=90)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--basis-count", type=int, default=512)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--burg-epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial-basis", action="store_true")
    args = parser.parse_args()

    frozen_setup_matches = bool(
        args.state_position == 2
        and args.replay_through_proposal == 89
        and args.proposal == 90
        and args.pairs == 32
        and args.block_size == 4
        and args.basis_count == 512
        and math.isclose(args.beta, DEFAULT_BETA, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.burg_epsilon, 1e-4, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.gradient_clip, 1.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.lr, 3e-5, rel_tol=0.0, abs_tol=0.0)
        and args.hessian_chunk_size == 64
    )
    if args.basis_count == 512 and not frozen_setup_matches:
        raise RuntimeError("full-basis evidence run must use the frozen iteration-2 setup exactly")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _source_sha256()
    shutil.copy2(Path(__file__), args.output_dir / "executed_source_snapshot.py")
    device = torch.device(args.device)
    checkpoint_hash = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    iteration1_checkpoint_hash = sha256_file(ITERATION1_CHECKPOINT)
    iteration1_diagnostics_hash = sha256_file(ITERATION1_DIAGNOSTICS)
    iteration1_states_hash = sha256_file(ITERATION1_STATES)
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_hash}")
    if args.basis_count != 512 and not args.allow_partial_basis:
        raise RuntimeError("partial basis is preflight-only; pass --allow-partial-basis explicitly")
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "base_checkpoint_sha256": checkpoint_hash,
        "iteration1_checkpoint_sha256": iteration1_checkpoint_hash,
        "iteration1_diagnostics_sha256": iteration1_diagnostics_hash,
        "iteration1_states_sha256": iteration1_states_hash,
        "device": str(device),
        "dtype": "float32 model / float64 full-gradient accumulation",
        "state_position": args.state_position,
        "replay_through_proposal": args.replay_through_proposal,
        "proposal": args.proposal,
        "pairs": args.pairs,
        "block_size": args.block_size,
        "basis_count": args.basis_count,
        "beta": args.beta,
        "burg_epsilon": args.burg_epsilon,
        "gradient_clip": args.gradient_clip,
        "lr": args.lr,
        "fd_alphas": list(FD_ALPHAS),
        "armijo_alphas": list(ARMIJO_ALPHAS),
        "hessian_chunk_size": args.hessian_chunk_size,
        "output_dir": str(args.output_dir),
        "frozen_setup_matches": frozen_setup_matches,
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[i2-audit] start {json.dumps(resolved, sort_keys=True)}", flush=True)

    print("[i2-audit] stage=load", flush=True)
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    cfg_stopped = _probe_cfg(run.cfg, sample_count=1, pair_count=args.pairs, batch_size=16384)
    cfg_naive = replace(cfg_stopped, vae_precond_hvp_mode="autograd")
    bank = pd.read_csv(STATE_BANK)
    selected = bank.loc[bank["state_position"].eq(args.state_position)]
    if len(selected) != 1:
        raise ValueError(f"state_position {args.state_position} is not unique")
    source_index = int(selected.iloc[0]["source_weight_index"])
    weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
    record = run.records.iloc[source_index].to_dict()
    record["source_weight_index"] = source_index
    task_set = _task_set_for_record(run.task_tensors, record)
    if int(task_set.train_labels.numel()) != 16384:
        raise RuntimeError("the frozen full-CE assumption requires exactly 16384 task examples")

    active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
    named = dict(run.vae.named_parameters())
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in active_names]
    diagnostics = pd.read_csv(ITERATION1_DIAGNOSTICS)
    expected_accepted_step = int(
        diagnostics.loc[diagnostics["proposal"].le(args.replay_through_proposal), "accepted"].sum()
    )
    print(
        f"[i2-audit] stage=replay-through-{args.replay_through_proposal} "
        f"expected_accepted_step={expected_accepted_step}",
        flush=True,
    )
    carried_avg, carried_avg_sq, carried_step, replay_rows = _replay_iteration1_state(
        through_proposal=args.replay_through_proposal,
        diagnostics=diagnostics,
        cfg=cfg_stopped,
        run=run,
        z=z,
        record=record,
        active=active,
        beta=args.beta,
        epsilon=args.burg_epsilon,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    if carried_step != expected_accepted_step:
        raise RuntimeError(f"replayed Adam step mismatch: {carried_step} vs {expected_accepted_step}")
    replay_frame = pd.DataFrame(replay_rows)
    replay_frame.to_csv(args.output_dir / "state_replay.csv", index=False)
    base = [parameter.detach().clone() for parameter in active]
    parameter_hash_before = _named_tensor_hash(active_names, active)
    moment_hash_before = hashlib.sha256(
        (_named_tensor_hash(active_names, carried_avg) + _named_tensor_hash(active_names, carried_avg_sq)).encode("utf-8")
    ).hexdigest()
    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[i2-audit] loaded source={source_index} task={record.get('task_name')} z_dim={z.numel()} "
        f"z_sha256={sha256_tensor(z)} active_parameters={sum(p.numel() for p in active)} "
        f"carried_step={carried_step}",
        flush=True,
    )

    print("[i2-audit] stage=dense-reference", flush=True)
    metric, hessian, matrix, _eig_m, burg_gradient = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=args.burg_epsilon,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    current_f = _objective(metric, args.beta)
    states_frame = pd.read_csv(ITERATION1_STATES)
    expected_state = states_frame.loc[states_frame["proposal"].eq(args.replay_through_proposal)]
    if len(expected_state) != 1:
        raise RuntimeError("missing frozen replay target in state trajectory")
    expected_replay_f = float(expected_state.iloc[0]["true_objective"])
    replay_f_abs_error = abs(current_f - expected_replay_f)
    dim = int(z.numel())
    identity = torch.eye(dim, device=device, dtype=torch.float64)
    h64 = hessian.double()
    m64 = matrix.double()
    g_a = 2.0 * (m64 - identity) / float(dim)
    g_b = identity / (float(dim) * (1.0 + float(args.burg_epsilon))) - torch.linalg.inv(
        m64 + float(args.burg_epsilon) * identity
    ) / float(dim)
    g_total = g_a + float(args.beta) * g_b
    k_cotangent = (2.0 * g_total @ h64).detach()
    burg_gradient_relative_error = float(
        ((g_b - burg_gradient.double()).norm() / g_b.norm().clamp_min(1e-30)).cpu()
    )
    matrix_direction_check = float((k_cotangent * h64).sum().cpu())
    full_ce_batch_none = all(
        _batch_indices(
            task_set,
            batch_size=int(cfg_stopped.vae_precond_batch_size),
            step=10,
            sample_key=source_index,
            pair_key=pair_key,
        )
        is None
        for pair_key in range(2 * args.pairs)
    )
    print("[i2-audit] stage=forward-equivalence", flush=True)
    forward_rows = _forward_equivalence(
        cfg_stopped=cfg_stopped,
        cfg_naive=cfg_naive,
        run=run,
        z=z,
        record=record,
        hessian=hessian,
        probes=4,
    )
    pd.DataFrame(forward_rows).to_csv(args.output_dir / "forward_equivalence.csv", index=False)
    h_space_rows = _h_space_fd_check(
        hessian,
        k_cotangent,
        beta=args.beta,
        epsilon=args.burg_epsilon,
    )
    pd.DataFrame(h_space_rows).to_csv(args.output_dir / "h_space_fd.csv", index=False)

    preflight_basis_count = min(8, args.basis_count)
    print(f"[i2-audit] stage=basis-accumulation-preflight rows={preflight_basis_count}", flush=True)
    memory_before_preflight = int(torch.cuda.memory_allocated(device))
    sequential_preflight, sequential_preflight_meta = _full_basis_gradient(
        cfg=cfg_naive,
        run=run,
        z=z,
        record=record,
        active=active,
        k_cotangent=k_cotangent,
        mode="autograd",
        basis_count=preflight_basis_count,
    )
    memory_after_sequential = int(torch.cuda.memory_allocated(device))
    aggregated_preflight = _aggregated_basis_gradient(
        cfg=cfg_naive,
        run=run,
        z=z,
        record=record,
        active=active,
        k_cotangent=k_cotangent,
        basis_count=preflight_basis_count,
    )
    memory_after_aggregated = int(torch.cuda.memory_allocated(device))
    basis_accumulation_relative_error = _relative_error(sequential_preflight, aggregated_preflight)
    basis_preflight = {
        "basis_count": preflight_basis_count,
        "sequential_to_aggregated_relative_error": basis_accumulation_relative_error,
        "sequential_to_aggregated_cosine": _cosine(sequential_preflight, aggregated_preflight),
        "memory_before_bytes": memory_before_preflight,
        "memory_after_sequential_bytes": memory_after_sequential,
        "memory_after_aggregated_bytes": memory_after_aggregated,
        "sequential_metadata": sequential_preflight_meta,
    }
    (args.output_dir / "basis_accumulation_preflight.json").write_text(
        json.dumps(basis_preflight, indent=2, sort_keys=True), encoding="utf-8"
    )
    del sequential_preflight, aggregated_preflight
    dense_reference = {
        **metric,
        "true_objective": current_f,
        "expected_replay_true_objective": expected_replay_f,
        "replay_true_objective_abs_error": replay_f_abs_error,
        "full_ce_batch_indices_all_none": full_ce_batch_none,
        "dimension": dim,
        "hessian_sha256": sha256_tensor(hessian),
        "matrix_sha256": sha256_tensor(matrix),
        "burg_gradient_relative_error": burg_gradient_relative_error,
        "k_cotangent_norm": float(k_cotangent.norm().cpu()),
        "k_dot_h": matrix_direction_check,
    }
    (args.output_dir / "dense_reference.json").write_text(
        json.dumps(dense_reference, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("[i2-audit] stage=exact-full-naive-gradient", flush=True)
    exact_gradient, exact_meta = _full_basis_gradient(
        cfg=cfg_naive,
        run=run,
        z=z,
        record=record,
        active=active,
        k_cotangent=k_cotangent,
        mode="autograd",
        basis_count=args.basis_count,
    )
    print("[i2-audit] stage=full-stopped-gradient", flush=True)
    stopped_full, stopped_meta = _full_basis_gradient(
        cfg=cfg_stopped,
        run=run,
        z=z,
        record=record,
        active=active,
        k_cotangent=k_cotangent,
        mode="stopped_composite",
        basis_count=args.basis_count,
    )

    print("[i2-audit] stage=random-gradient-bank", flush=True)
    stopped_p32, stopped_blocks, atomic_stopped = _gradient_from_losses(
        cfg=cfg_stopped,
        run=run,
        z=z,
        record=record,
        active=active,
        burg_gradient=burg_gradient,
        beta=args.beta,
        proposal=args.proposal,
        pairs=args.pairs,
        block_size=args.block_size,
        label="stopped",
    )
    naive_p32, naive_blocks, atomic_naive = _gradient_from_losses(
        cfg=cfg_naive,
        run=run,
        z=z,
        record=record,
        active=active,
        burg_gradient=burg_gradient,
        beta=args.beta,
        proposal=args.proposal,
        pairs=args.pairs,
        block_size=args.block_size,
        label="naive",
    )
    atomic_frame = pd.DataFrame(atomic_stopped + atomic_naive)
    atomic_frame.to_csv(args.output_dir / "atomic_random_losses.csv", index=False)

    gradient_vectors: dict[str, Vector] = {
        "exact_naive_full": exact_gradient,
        "stopped_full": stopped_full,
        "stopped_p32": stopped_p32,
        "naive_p32": naive_p32,
    }
    for index, value in enumerate(stopped_blocks):
        gradient_vectors[f"stopped_p4_block{index}"] = value
    for index, value in enumerate(naive_blocks):
        gradient_vectors[f"naive_p4_block{index}"] = value
    gradient_rows: list[dict[str, float | str]] = []
    for name, value in gradient_vectors.items():
        if name.startswith("stopped_p4") or name == "stopped_p32":
            population_reference = stopped_full
            population_name = "stopped_full"
        else:
            population_reference = exact_gradient
            population_name = "exact_naive_full"
        gradient_rows.append(
            _gradient_row(name, value, exact_gradient, population_reference, population_name)
        )
    gradients_frame = pd.DataFrame(gradient_rows)
    gradients_frame.to_csv(args.output_dir / "gradient_comparisons.csv", index=False)

    stopped_block_mean = _zeros_like(active)
    naive_block_mean = _zeros_like(active)
    for block in stopped_blocks:
        _add_(stopped_block_mean, block, alpha=1.0 / float(len(stopped_blocks)))
    for block in naive_blocks:
        _add_(naive_block_mean, block, alpha=1.0 / float(len(naive_blocks)))
    stopped_p32_pool_relative_error = _relative_error(stopped_block_mean, stopped_p32)
    naive_p32_pool_relative_error = _relative_error(naive_block_mean, naive_p32)
    del stopped_block_mean, naive_block_mean

    selected_p4 = stopped_blocks[0]
    zero_avg = _zeros_like(active)
    zero_avg_sq = _zeros_like(active)
    p4_carried, p4_carried_meta = _adam_direction(
        gradient=selected_p4,
        active=active,
        exp_avg=carried_avg,
        exp_avg_sq=carried_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    p4_fresh, p4_fresh_meta = _adam_direction(
        gradient=selected_p4,
        active=active,
        exp_avg=zero_avg,
        exp_avg_sq=zero_avg_sq,
        accepted_step=0,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    p4_same_clock, p4_same_clock_meta = _adam_direction(
        gradient=selected_p4,
        active=active,
        exp_avg=zero_avg,
        exp_avg_sq=zero_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    p4_torch_reference, p4_torch_after = _torch_adam_reference_direction(
        gradient=selected_p4,
        active=active,
        exp_avg=carried_avg,
        exp_avg_sq=carried_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    custom_adam_relative_error = _relative_error(p4_carried, p4_torch_reference)
    custom_adam_cosine = _cosine(p4_carried, p4_torch_reference)
    custom_after = [
        (parameter.detach() + delta.detach().to(dtype=parameter.dtype)).clone()
        for parameter, delta in zip(active, p4_carried, strict=True)
    ]
    custom_adam_applied_update_relative_error = _norm(
        [custom - reference for custom, reference in zip(custom_after, p4_torch_after, strict=True)]
    ) / max(_norm(p4_carried), 1e-30)
    custom_adam_applied_update_max_abs = max(
        float((custom - reference).abs().max().cpu())
        for custom, reference in zip(custom_after, p4_torch_after, strict=True)
    )
    del p4_torch_reference, p4_torch_after, custom_after
    exact_carried, exact_carried_meta = _adam_direction(
        gradient=exact_gradient,
        active=active,
        exp_avg=carried_avg,
        exp_avg_sq=carried_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    exact_fresh, exact_fresh_meta = _adam_direction(
        gradient=exact_gradient,
        active=active,
        exp_avg=zero_avg,
        exp_avg_sq=zero_avg_sq,
        accepted_step=0,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    exact_same_clock, exact_same_clock_meta = _adam_direction(
        gradient=exact_gradient,
        active=active,
        exp_avg=zero_avg,
        exp_avg_sq=zero_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    target_norm = _norm(p4_carried)
    if target_norm <= 0.0:
        raise RuntimeError("production carried P4 proposal has zero norm")
    raw_direction_sources = {
        "stopped_p4_raw": selected_p4,
        "stopped_p32_raw": stopped_p32,
        "naive_p32_raw": naive_p32,
        "stopped_full_raw": stopped_full,
        "exact_naive_full_raw": exact_gradient,
    }
    directions: dict[str, Vector] = {
        name: _negative_normalized(value, target_norm=target_norm, active=active)
        for name, value in raw_direction_sources.items()
    }
    directions.update(
        {
            "stopped_p4_fresh_adam": _normalize_direction(p4_fresh, target_norm=target_norm, active=active),
            "stopped_p4_same_clock_zero_moments_adam": _normalize_direction(
                p4_same_clock, target_norm=target_norm, active=active
            ),
            "stopped_p4_carried_adam": _normalize_direction(p4_carried, target_norm=target_norm, active=active),
            "exact_naive_full_fresh_adam": _normalize_direction(exact_fresh, target_norm=target_norm, active=active),
            "exact_naive_full_same_clock_zero_moments_adam": _normalize_direction(
                exact_same_clock, target_norm=target_norm, active=active
            ),
            "exact_naive_full_carried_adam": _normalize_direction(exact_carried, target_norm=target_norm, active=active),
        }
    )

    print(f"[i2-audit] stage=directional-fd directions={len(directions)} target_norm={target_norm:.8g}", flush=True)
    direction_rows: list[dict[str, float | str]] = []
    fd_rows: list[dict[str, float | str]] = []
    for index, (name, direction) in enumerate(directions.items(), start=1):
        summary, rows = _directional_fd(
            name=name,
            direction=direction,
            exact_gradient=exact_gradient,
            current_f=current_f,
            run=run,
            z=z,
            record=record,
            active=active,
            base=base,
            beta=args.beta,
            epsilon=args.burg_epsilon,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        direction_rows.append(summary)
        fd_rows.extend(rows)
        print(
            f"[i2-audit] direction={index}/{len(directions)} name={name} "
            f"analytic={summary['analytic_slope']:+.5g} fd256={summary['slope_h256']:+.5g}",
            flush=True,
        )
    directions_frame = pd.DataFrame(direction_rows)
    fd_frame = pd.DataFrame(fd_rows)
    directions_frame.to_csv(args.output_dir / "direction_diagnostics.csv", index=False)
    fd_frame.to_csv(args.output_dir / "direction_finite_differences.csv", index=False)

    print("[i2-audit] stage=exact-armijo-profile", flush=True)
    exact_direction = directions["exact_naive_full_raw"]
    exact_slope = _dot(exact_gradient, exact_direction)
    armijo_rows: list[dict[str, float | int]] = []
    accepted_alpha = 0.0
    accepted_f = current_f
    for alpha in ARMIJO_ALPHAS:
        candidate = _evaluate_candidate(
            run=run,
            z=z,
            record=record,
            active=active,
            base=base,
            displacement=exact_direction,
            alpha=alpha,
            beta=args.beta,
            epsilon=args.burg_epsilon,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        armijo_rhs = current_f + 1e-4 * alpha * exact_slope
        accepted = int(
            accepted_alpha == 0.0
            and float(candidate["true_objective"]) <= armijo_rhs
            and float(candidate["true_objective"]) < current_f - 1e-8
        )
        if accepted:
            accepted_alpha = alpha
            accepted_f = float(candidate["true_objective"])
        armijo_rows.append(
            {
                "alpha": alpha,
                "parameter_radius": alpha * target_norm,
                "current_f": current_f,
                "true_objective": float(candidate["true_objective"]),
                "delta_f": float(candidate["true_objective"]) - current_f,
                "armijo_rhs": armijo_rhs,
                "accepted_as_largest": accepted,
                "exact_a_per_dim": float(candidate["exact_a_per_dim"]),
                "damped_full_burg_per_dim": float(candidate["damped_full_burg_per_dim"]),
                "task_loss": float(candidate["task_loss"]),
                "m_max": float(candidate["m_max"]),
            }
        )
    _set_parameters(active, base, exact_direction, 0.0)
    armijo_frame = pd.DataFrame(armijo_rows)
    armijo_frame.to_csv(args.output_dir / "exact_armijo_profile.csv", index=False)

    repeat_metric, _repeat_h, _repeat_m, _repeat_eig, _repeat_burg = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=args.burg_epsilon,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    repeat_f = _objective(repeat_metric, args.beta)
    parameter_hash_after = _named_tensor_hash(active_names, active)
    moment_hash_after = hashlib.sha256(
        (_named_tensor_hash(active_names, carried_avg) + _named_tensor_hash(active_names, carried_avg_sq)).encode("utf-8")
    ).hexdigest()

    direction_index = directions_frame.set_index("direction")
    exact_row = direction_index.loc["exact_naive_full_raw"]
    p4_raw = direction_index.loc["stopped_p4_raw"]
    p4_fresh_row = direction_index.loc["stopped_p4_fresh_adam"]
    p4_same_clock_row = direction_index.loc["stopped_p4_same_clock_zero_moments_adam"]
    p4_carried_row = direction_index.loc["stopped_p4_carried_adam"]
    exact_fresh_row = direction_index.loc["exact_naive_full_fresh_adam"]
    exact_same_clock_row = direction_index.loc["exact_naive_full_same_clock_zero_moments_adam"]
    exact_carried_row = direction_index.loc["exact_naive_full_carried_adam"]
    stopped_full_row = direction_index.loc["stopped_full_raw"]
    gradient_index = gradients_frame.set_index("gradient")
    stopped_full_cosine = float(gradient_index.loc["stopped_full", "cosine_to_exact"])
    exact_descent_strength = max(1e-30, -float(exact_row["analytic_slope"]))
    stopped_descent_strength_ratio = -float(stopped_full_row["analytic_slope"]) / exact_descent_strength
    stopped_block_slopes = [
        _dot(exact_gradient, _negative_normalized(block, target_norm=target_norm, active=active))
        for block in stopped_blocks
    ]
    naive_block_slopes = [
        _dot(exact_gradient, _negative_normalized(block, target_norm=target_norm, active=active))
        for block in naive_blocks
    ]
    stopped_block_population_cosines = [_cosine(block, stopped_full) for block in stopped_blocks]
    naive_block_population_cosines = [_cosine(block, exact_gradient) for block in naive_blocks]
    stopped_p32_population_cosine = _cosine(stopped_p32, stopped_full)
    naive_p32_population_cosine = _cosine(naive_p32, exact_gradient)
    stopped_uphill_block_count = int(sum(slope >= 0.0 for slope in stopped_block_slopes))
    naive_uphill_block_count = int(sum(slope >= 0.0 for slope in naive_block_slopes))
    probe_supported = bool(
        stopped_uphill_block_count >= 4
        and stopped_block_slopes[0] >= 0.0
        and float(direction_index.loc["stopped_p32_raw", "analytic_slope"]) < 0.0
        and float(stopped_full_row["analytic_slope"]) < 0.0
        and stopped_p32_population_cosine > float(np.median(stopped_block_population_cosines))
    )
    derivative_sign_cause_supported = bool(
        float(stopped_full_row["analytic_slope"]) >= 0.0 and float(exact_row["analytic_slope"]) < 0.0
    )
    derivative_degradation = bool(
        stopped_full_cosine < 0.9 or stopped_descent_strength_ratio <= 0.5
    )
    adam_local_supported = bool(
        float(p4_carried_row["analytic_slope"]) >= 0.0
        and float(p4_raw["analytic_slope"]) < 0.0
        and float(p4_fresh_row["analytic_slope"]) < 0.0
        and float(p4_same_clock_row["analytic_slope"]) < 0.0
    )
    adam_exact_sufficient = bool(
        float(exact_carried_row["analytic_slope"]) >= 0.0
        and float(exact_row["analytic_slope"]) < 0.0
        and float(exact_fresh_row["analytic_slope"]) < 0.0
        and float(exact_same_clock_row["analytic_slope"]) < 0.0
    )
    exact_fd_valid = bool(
        float(exact_row["slope_h128"]) < 0.0
        and float(exact_row["slope_h256"]) < 0.0
        and float(exact_row["fd_relative_error_h128"]) <= 0.05
        and float(exact_row["fd_relative_error_h256"]) <= 0.05
        and int(exact_row["descent_ordering_h128"]) == 1
        and int(exact_row["descent_ordering_h256"]) == 1
    )
    known_failure = diagnostics.loc[diagnostics["proposal"].eq(args.proposal)]
    if len(known_failure) != 1:
        raise RuntimeError("missing known failure proposal in iteration1 diagnostics")
    known_failure = known_failure.iloc[0]
    proposal_norm_relative_error = abs(target_norm - float(known_failure["proposal_norm"])) / max(
        abs(float(known_failure["proposal_norm"])), 1e-30
    )
    recorded_slope_h128_error = abs(float(p4_carried_row["slope_h128"]) - float(known_failure["slope_h128"]))
    recorded_slope_h256_error = abs(float(p4_carried_row["slope_h256"]) - float(known_failure["slope_h256"]))
    forward_max_error = max(
        max(float(row["stopped_to_dense_relative_error"]) for row in forward_rows),
        max(float(row["naive_to_dense_relative_error"]) for row in forward_rows),
        max(float(row["stopped_to_naive_relative_error"]) for row in forward_rows),
    )
    h_space_max_error = max(float(row["relative_error"]) for row in h_space_rows)
    all_vectors_finite = all(
        _vector_is_finite(value)
        for value in [
            *gradient_vectors.values(),
            *directions.values(),
            p4_carried,
            p4_fresh,
            p4_same_clock,
            exact_carried,
            exact_fresh,
            exact_same_clock,
        ]
    )
    all_tables_finite = all(
        _numeric_frame_is_finite(frame)
        for frame in [
            replay_frame,
            pd.DataFrame(forward_rows),
            pd.DataFrame(h_space_rows),
            atomic_frame,
            gradients_frame,
            directions_frame,
            fd_frame,
            armijo_frame,
        ]
    )
    dense_reference_finite = all(
        not isinstance(value, (int, float)) or math.isfinite(float(value))
        for value in dense_reference.values()
    )
    validity_gates = {
        "complete_512_basis": bool(args.basis_count == dim),
        "frozen_setup_matches": frozen_setup_matches,
        "all_gradient_and_direction_tensors_finite": all_vectors_finite,
        "all_numeric_artifact_tables_finite": all_tables_finite,
        "dense_reference_values_finite": dense_reference_finite,
        "full_ce_batch_indices_are_none": full_ce_batch_none,
        "replay_f_error_at_most_1e_5": replay_f_abs_error <= 1e-5,
        "replayed_accepted_step_matches": carried_step == expected_accepted_step,
        "known_failure_proposal_was_rejected": int(known_failure["accepted"]) == 0,
        "known_failure_proposal_norm_relative_error_at_most_1e_6": proposal_norm_relative_error <= 1e-6,
        "known_failure_slopes_reproduced_at_most_5e_3": max(
            recorded_slope_h128_error, recorded_slope_h256_error
        )
        <= 5e-3,
        "forward_hvp_max_relative_error_at_most_1e_5": forward_max_error <= 1e-5,
        "h_space_fd_max_relative_error_at_most_1e_5": h_space_max_error <= 1e-5,
        "basis_accumulation_relative_error_at_most_1e_6": basis_accumulation_relative_error <= 1e-6,
        "sequential_preflight_memory_growth_at_most_64mib": bool(
            sequential_preflight_meta["memory_growth_gate_pass"]
        ),
        "exact_full_memory_growth_at_most_64mib": bool(exact_meta["memory_growth_gate_pass"]),
        "stopped_full_memory_growth_at_most_64mib": bool(stopped_meta["memory_growth_gate_pass"]),
        "stopped_p32_is_raw_block_mean": stopped_p32_pool_relative_error <= 1e-6,
        "naive_p32_is_raw_block_mean": naive_p32_pool_relative_error <= 1e-6,
        "custom_adam_matches_torch_fp32_rounding_gate": bool(
            custom_adam_applied_update_relative_error <= 1e-5
            and custom_adam_applied_update_max_abs <= 3e-8
            and custom_adam_cosine >= 0.9999999
        ),
        "repeat_f_error_at_most_1e_5": abs(repeat_f - current_f) <= 1e-5,
        "exact_direction_fd_valid": exact_fd_valid,
        "exact_armijo_found_strict_decrease": accepted_alpha > 0.0,
        "parameter_hash_restored": parameter_hash_before == parameter_hash_after,
        "moment_hash_unchanged": moment_hash_before == moment_hash_after,
        "burg_gradient_formula_relative_error_at_most_1e_5": burg_gradient_relative_error <= 1e-5,
    }
    efficacy_gates = {
        "exact_armijo_alpha_at_least_1_over_64": accepted_alpha >= 1.0 / 64.0,
        "exact_armijo_strict_decrease": accepted_f < current_f - 1e-8,
    }
    valid = bool(all(validity_gates.values()))
    raw_mechanism_decisions: dict[str, bool | str | None] = {
        "finite_probe_sign_failure_locally_supported": probe_supported,
        "stopped_derivative_sign_failure_locally_supported": derivative_sign_cause_supported,
        "stopped_derivative_descent_degradation": derivative_degradation,
        "carried_moment_contents_sign_failure_on_p4": adam_local_supported,
        "carried_moment_contents_sufficient_with_exact_gradient": adam_exact_sufficient,
        "accepted_only_moment_policy": "not tested",
    }
    if not valid:
        mechanism_decisions: dict[str, bool | str | None] = {
            key: ("not tested" if key == "accepted_only_moment_policy" else None)
            for key in raw_mechanism_decisions
        }
        mechanism_decisions["not_evaluable_reason"] = "one or more validity gates failed"
    else:
        mechanism_decisions = raw_mechanism_decisions
    decision = {
        "protocol_id": PROTOCOL_ID,
        "valid": valid,
        "validity_gates": validity_gates,
        "repair_efficacy_gates": efficacy_gates,
        "initial_true_objective": current_f,
        "repeat_true_objective": repeat_f,
        "repeat_absolute_error": abs(repeat_f - current_f),
        "target_direction_norm": target_norm,
        "accepted_exact_armijo_alpha": accepted_alpha,
        "accepted_exact_armijo_f": accepted_f,
        "accepted_exact_armijo_delta_f": accepted_f - current_f,
        "stopped_full_cosine_to_exact": stopped_full_cosine,
        "stopped_full_descent_strength_ratio": stopped_descent_strength_ratio,
        "stopped_p4_block_analytic_slopes": stopped_block_slopes,
        "naive_p4_block_analytic_slopes": naive_block_slopes,
        "stopped_p4_uphill_block_count": stopped_uphill_block_count,
        "naive_p4_uphill_block_count": naive_uphill_block_count,
        "stopped_p4_block_population_cosines": stopped_block_population_cosines,
        "naive_p4_block_population_cosines": naive_block_population_cosines,
        "stopped_p32_population_cosine": stopped_p32_population_cosine,
        "naive_p32_population_cosine": naive_p32_population_cosine,
        "stopped_p32_pool_relative_error": stopped_p32_pool_relative_error,
        "naive_p32_pool_relative_error": naive_p32_pool_relative_error,
        "forward_hvp_max_relative_error": forward_max_error,
        "h_space_fd_max_relative_error": h_space_max_error,
        "basis_accumulation_relative_error": basis_accumulation_relative_error,
        "custom_adam_relative_error": custom_adam_relative_error,
        "custom_adam_cosine": custom_adam_cosine,
        "custom_adam_applied_update_relative_error": custom_adam_applied_update_relative_error,
        "custom_adam_applied_update_max_abs": custom_adam_applied_update_max_abs,
        "known_failure_reproduction": {
            "proposal": args.proposal,
            "expected_current_f": float(known_failure["current_f"]),
            "replayed_current_f": current_f,
            "proposal_norm_relative_error": proposal_norm_relative_error,
            "recorded_slope_h128": float(known_failure["slope_h128"]),
            "reproduced_slope_h128": float(p4_carried_row["slope_h128"]),
            "recorded_slope_h256": float(known_failure["slope_h256"]),
            "reproduced_slope_h256": float(p4_carried_row["slope_h256"]),
        },
        "mechanism_decisions": mechanism_decisions,
        "raw_mechanism_decisions_non_evidence_if_invalid": raw_mechanism_decisions,
        "full_basis_metadata": [exact_meta, stopped_meta],
        "adam_metadata": {
            "p4_carried": p4_carried_meta,
            "p4_fresh": p4_fresh_meta,
            "p4_same_clock_zero_moments": p4_same_clock_meta,
            "exact_carried": exact_carried_meta,
            "exact_fresh": exact_fresh_meta,
            "exact_same_clock_zero_moments": exact_same_clock_meta,
        },
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "moment_hash_before": moment_hash_before,
        "moment_hash_after": moment_hash_after,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    core_to_save = {
        name: {parameter_name: tensor.detach().float().cpu() for parameter_name, tensor in zip(active_names, values, strict=True)}
        for name, values in {
            "exact_naive_full": exact_gradient,
            "stopped_full": stopped_full,
            "stopped_p32": stopped_p32,
            "naive_p32": naive_p32,
            "stopped_p4_block0": selected_p4,
        }.items()
    }
    torch.save(core_to_save, args.output_dir / "core_gradient_vectors.pt")
    _plot_results(
        gradients=gradients_frame,
        directions=directions_frame,
        fd=fd_frame,
        armijo=armijo_frame,
        output=args.output_dir / "direction_source_audit.png",
    )

    artifact_paths = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths if path.name != "run.log"},
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[i2-audit] decision {json.dumps(decision, sort_keys=True)}", flush=True)
    print(
        f"[i2-audit] complete valid={decision['valid']} F={current_f:.8g} "
        f"exact_step={accepted_f:.8g} alpha={accepted_alpha:g} elapsed={decision['elapsed_sec']:.1f}s",
        flush=True,
    )
    print(
        "[i2-audit] artifacts=" + ",".join(str(path) for path in sorted(args.output_dir.iterdir())),
        flush=True,
    )


if __name__ == "__main__":
    main()
