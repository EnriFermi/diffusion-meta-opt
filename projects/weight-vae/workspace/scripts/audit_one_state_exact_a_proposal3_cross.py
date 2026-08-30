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
    _task_set_for_record,
)
from scripts import run_one_state_a_full_burg_p32_training as p32
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _cosine,
    _dot,
    _full_basis_gradient,
    _h_space_fd_check,
    _named_tensor_hash,
    _negative_normalized,
    _norm,
    _relative_error,
    _scaled,
)
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _load_run,
    _probe_cfg,
    sha256_file,
    sha256_tensor,
)
from scripts.run_one_state_a_full_burg_armijo import (
    _adam_proposal,
    _set_parameters,
)
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    STATE_BANK,
    _gradient_norms,
    _materialize_metric,
    _pair_losses,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_a_proposal3_cross_i1_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration1_proposal3_cross_production"
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_1_proposal3_cross/protocol.md"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_1_frozen_dependency_manifest.json"
EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = (
    "0b35d4986611a814ad591e537d4954f0e24106dcbcf8b8733b2822367840adc4"
)
ITERATION5 = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration5_p32_training_production"
)
ITERATION5_STATES = ITERATION5 / "state_objective_curve.csv"
ITERATION5_PROPOSALS = ITERATION5 / "proposal_diagnostics.csv"
OLD_BETA = 22.536727828943093
EPSILON = 1e-4
TARGET_PROPOSAL = 3
REPLAY_THROUGH = 2
POSITIVE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
FD_STEPS = (1.0 / 128.0, 1.0 / 256.0)
DIRECTION_NAMES = (
    "exact_a_raw",
    "p32_a_raw",
    "exact_oldbeta_raw",
    "p32_oldbeta_raw",
    "p32_oldbeta_carried_adam",
    "p32_unit_common_carried_adam",
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_INITIAL_PARAMETER_SHA256 = "1cf3bdeb383c64b36b3ca69a956457833d0aa0d8dd1f6696e3964033518053d4"

Vector = list[torch.Tensor]


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _relative_scalar_error(observed: float, reference: float) -> float:
    return abs(float(observed) - float(reference)) / max(abs(float(reference)), 1.0)


def _reset_p32_module_state() -> None:
    """Keep replay diagnostics independent of prior imports in the same process."""
    p32.DRAW_ROWS.clear()
    p32.PAIR_ROWS.clear()
    p32.POOLING_PREFLIGHT.clear()
    p32.SEED_PREFLIGHT.clear()
    p32.TREATMENT_BUILD_CALLS = 0
    p32.TREATMENT_CLIP_CALLS = 0
    p32.TREATMENT_ADAM_CALLS = 0


def _validate_frozen_dependencies() -> dict[str, bool]:
    if sha256_file(FROZEN_DEPENDENCY_MANIFEST) != EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256:
        raise RuntimeError("frozen dependency manifest hash mismatch")
    expected = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    matches = {
        relative: (ROOT / relative).is_file() and sha256_file(ROOT / relative) == digest
        for relative, digest in expected.items()
    }
    if not all(matches.values()):
        failed = [relative for relative, matches_expected in matches.items() if not matches_expected]
        raise RuntimeError(f"frozen dependency mismatch: {failed}")
    return matches


def _vector_sum(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    *,
    left_scale: float,
    right_scale: float,
    active: Sequence[torch.nn.Parameter],
) -> Vector:
    return [
        (
            a.detach().to(device=parameter.device, dtype=parameter.dtype) * float(left_scale)
            + b.detach().to(device=parameter.device, dtype=parameter.dtype) * float(right_scale)
        ).clone()
        for a, b, parameter in zip(left, right, active, strict=True)
    ]


def _unit_common_gradient(
    gradient_a: Sequence[torch.Tensor],
    gradient_b: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
) -> Vector:
    norm_a = _norm(gradient_a)
    norm_b = _norm(gradient_b)
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise RuntimeError(f"invalid component norms for common gradient: {norm_a}, {norm_b}")
    return _vector_sum(
        gradient_a,
        gradient_b,
        left_scale=1.0 / norm_a,
        right_scale=1.0 / norm_b,
        active=active,
    )


def _materialize(values: Sequence[torch.Tensor | None], active: Sequence[torch.nn.Parameter]) -> Vector:
    result: Vector = []
    for value, parameter in zip(values, active, strict=True):
        result.append(torch.zeros_like(parameter) if value is None else value.detach().clone())
    return result


def _p32_components(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active: Sequence[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    proposal: int,
) -> tuple[Vector, Vector, list[dict[str, float | int]], list[dict[str, float | int]]]:
    pooled_a = [torch.zeros_like(parameter) for parameter in active]
    pooled_b = [torch.zeros_like(parameter) for parameter in active]
    pair_rows: list[dict[str, float | int]] = []
    draw_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for draw in range(p32.POOL_DRAWS):
        a_losses: list[torch.Tensor] = []
        b_losses: list[torch.Tensor] = []
        for pair in range(p32.PAIRS_PER_DRAW):
            generator_1, generator_2, seed_1, seed_2 = p32._pair_generators(proposal, draw, pair)
            a_loss, b_loss, _ = _pair_losses(
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
            pair_rows.append(
                {
                    "proposal": proposal,
                    "draw": draw,
                    "pair": pair,
                    "seed_1": seed_1,
                    "seed_2": seed_2,
                    "a_loss": float(a_loss.detach().cpu()),
                    "b_pseudo_loss": float(b_loss.detach().cpu()),
                }
            )
        mean_a = torch.stack(a_losses).mean()
        mean_b = torch.stack(b_losses).mean()
        raw_a = torch.autograd.grad(mean_a, active, retain_graph=True, allow_unused=True)
        raw_b = torch.autograd.grad(mean_b, active, retain_graph=False, allow_unused=True)
        gradient_a = _materialize(raw_a, active)
        gradient_b = _materialize(raw_b, active)
        with torch.no_grad():
            for target, value in zip(pooled_a, gradient_a, strict=True):
                target.add_(value, alpha=1.0 / float(p32.POOL_DRAWS))
            for target, value in zip(pooled_b, gradient_b, strict=True):
                target.add_(value, alpha=1.0 / float(p32.POOL_DRAWS))
        stats = _gradient_norms(tuple(gradient_a), tuple(gradient_b), beta=OLD_BETA)
        draw_rows.append(
            {
                "proposal": proposal,
                "draw": draw,
                "mean_a": float(mean_a.detach().cpu()),
                "mean_b": float(mean_b.detach().cpu()),
                "unused_parameter_tensors": int(
                    sum(a is None and b is None for a, b in zip(raw_a, raw_b, strict=True))
                ),
                **stats,
            }
        )
        print(
            f"[exact-a-i1] stage=p32 proposal={proposal} draw={draw + 1}/{p32.POOL_DRAWS} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del a_losses, b_losses, mean_a, mean_b, raw_a, raw_b, gradient_a, gradient_b
    return pooled_a, pooled_b, pair_rows, draw_rows


def _adam_direction(
    *,
    gradient: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
    exp_avg: Sequence[torch.Tensor],
    exp_avg_sq: Sequence[torch.Tensor],
    accepted_step: int,
    gradient_clip: float,
    lr: float,
) -> tuple[Vector, float, float]:
    with torch.no_grad():
        for parameter, value in zip(active, gradient, strict=True):
            parameter.grad = value.detach().to(device=parameter.device, dtype=parameter.dtype).clone()
    preclip_norm = float(torch.nn.utils.clip_grad_norm_(active, max_norm=gradient_clip).detach().cpu())
    clip_factor = min(1.0, float(gradient_clip) / max(preclip_norm, 1e-30))
    _next_avg, _next_avg_sq, displacement = _adam_proposal(
        active=active,
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
    return displacement, preclip_norm, clip_factor


def _extra_spectrum_metrics(eig_m: torch.Tensor) -> dict[str, float]:
    eig = eig_m.detach().double().sort().values
    dim = int(eig.numel())
    contribution = (eig - 1.0).square()
    total_a = contribution.sum().clamp_min(1e-30)
    trace = eig.sum()
    square_sum = eig.square().sum().clamp_min(1e-30)
    c90 = contribution[: int(math.floor(0.9 * dim))].sum() / float(dim)
    return {
        "effective_rank": float((trace.square() / square_sum).cpu()),
        "effective_rank_fraction": float((trace.square() / (square_sum * float(dim))).cpu()),
        "li_gap_per_dim": float((eig.sqrt() - 1.0).square().mean().cpu()),
        "a_low90_abs_per_dim": float(c90.cpu()),
        "a_low_lt_0p1_abs_per_dim": float(
            (contribution[eig < 0.1].sum() / float(dim)).cpu()
        ),
        "a_high_gt_1_abs_per_dim": float(
            (contribution[eig > 1.0].sum() / float(dim)).cpu()
        ),
        "a_top1_share": float((contribution[-1] / total_a).cpu()),
        "a_top10_share": float((contribution[-10:].sum() / total_a).cpu()),
        "top1_trace_share": float((eig[-1] / trace.clamp_min(1e-30)).cpu()),
        "top10_trace_share": float((eig[-10:].sum() / trace.clamp_min(1e-30)).cpu()),
    }


def _evaluate_dense(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    epsilon: float,
    hessian_chunk_size: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row, hessian, matrix, eig_m, burg_gradient = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=epsilon,
        hessian_chunk_size=hessian_chunk_size,
    )
    dim = int(eig_m.numel())
    identity = torch.eye(dim, device=matrix.device, dtype=matrix.dtype)
    a_direct = float(((matrix - identity).square().sum() / float(dim)).cpu())
    a_trace = float((1.0 - 2.0 * eig_m.mean() + eig_m.square().mean()).cpu())
    raw_eig = torch.linalg.eigvalsh(matrix.double())
    row.update(
        {
            "true_objective": float(row["exact_a_per_dim"] + OLD_BETA * row["damped_full_burg_per_dim"]),
            "a_direct_matrix": a_direct,
            "a_trace_closure": a_trace,
            "a_direct_abs_error": abs(a_direct - float(row["exact_a_per_dim"])),
            "a_trace_abs_error": abs(a_trace - float(row["exact_a_per_dim"])),
            "m_raw_eig_min": float(raw_eig.min().cpu()),
            **_extra_spectrum_metrics(eig_m),
        }
    )
    return row, hessian, matrix, eig_m, burg_gradient


def _line_profile(
    *,
    direction_name: str,
    direction: Sequence[torch.Tensor],
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active_names: Sequence[str],
    active: Sequence[torch.nn.Parameter],
    base: Sequence[torch.Tensor],
    base_parameter_hash: str,
    base_metrics: Mapping[str, float],
    epsilon: float,
    hessian_chunk_size: int,
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    dict[str, float],
]:
    line_rows: list[dict[str, float | str]] = []
    spectrum_rows: list[dict[str, float | int | str]] = []
    repeat_rows: list[dict[str, float | int | str]] = []
    values: dict[float, dict[str, float]] = {}
    alphas = sorted(set(POSITIVE_ALPHAS + FD_STEPS + tuple(-value for value in FD_STEPS)))
    try:
        for index, alpha in enumerate(alphas):
            _set_parameters(active, list(base), list(direction), float(alpha))
            metrics, _hessian, _matrix, eig_m, _burg_gradient = _evaluate_dense(
                run=run,
                z=z,
                record=record,
                epsilon=epsilon,
                hessian_chunk_size=hessian_chunk_size,
            )
            metrics["true_objective_direct"] = float(
                metrics["a_direct_matrix"]
                + OLD_BETA * metrics["damped_full_burg_per_dim"]
            )
            values[float(alpha)] = metrics
            row: dict[str, float | str] = {
                "direction": direction_name,
                "alpha": float(alpha),
                "kind": "line" if alpha > 0.0 and alpha in POSITIVE_ALPHAS else "fd",
                **metrics,
                "delta_a": float(metrics["exact_a_per_dim"] - base_metrics["exact_a_per_dim"]),
                "delta_b": float(
                    metrics["damped_full_burg_per_dim"] - base_metrics["damped_full_burg_per_dim"]
                ),
                "delta_f": float(metrics["true_objective"] - base_metrics["true_objective"]),
                "delta_m_max": float(metrics["m_max"] - base_metrics["m_max"]),
                "delta_m_p50": float(metrics["m_p50"] - base_metrics["m_p50"]),
            }
            line_rows.append(row)
            sorted_eig = eig_m.detach().double().sort().values.cpu().numpy()
            for rank, eigenvalue in enumerate(sorted_eig):
                spectrum_rows.append(
                    {
                        "direction": direction_name,
                        "alpha": float(alpha),
                        "rank": rank,
                        "m_eigenvalue": float(eigenvalue),
                        "a_contribution": float((eigenvalue - 1.0) ** 2),
                    }
                )
            print(
                f"[exact-a-i1] stage=line direction={direction_name} point={index + 1}/{len(alphas)} "
                f"alpha={alpha:+.8g} A={metrics['exact_a_per_dim']:.7g} "
                f"B={metrics['damped_full_burg_per_dim']:.7g} mmax={metrics['m_max']:.7g}",
                flush=True,
            )
            del _hessian, _matrix, eig_m, _burg_gradient
        fine_step = FD_STEPS[-1]
        for repeat_index, alpha in enumerate((fine_step, -fine_step), start=1):
            _set_parameters(active, list(base), list(direction), float(alpha))
            metrics, _hessian, _matrix, eig_m, _burg_gradient = _evaluate_dense(
                run=run,
                z=z,
                record=record,
                epsilon=epsilon,
                hessian_chunk_size=hessian_chunk_size,
            )
            repeat_rows.append(
                {
                    "direction": direction_name,
                    "repeat": repeat_index,
                    "alpha": float(alpha),
                    **metrics,
                }
            )
            print(
                f"[exact-a-i1] stage=fd-repeat direction={direction_name} "
                f"point={repeat_index}/2 alpha={alpha:+.8g} A={metrics['a_direct_matrix']:.7g}",
                flush=True,
            )
            del _hessian, _matrix, eig_m, _burg_gradient
    finally:
        _set_parameters(active, list(base), list(direction), 0.0)
    restored_hash = _named_tensor_hash(active_names, active)
    if restored_hash != base_parameter_hash:
        raise RuntimeError(f"parameter restoration failed for {direction_name}")
    slopes: dict[str, float] = {}
    for step in FD_STEPS:
        positive = values[step]
        negative = values[-step]
        for metric_name, column in (
            ("a", "a_direct_matrix"),
            ("b", "damped_full_burg_per_dim"),
            ("f", "true_objective_direct"),
        ):
            slopes[f"slope_{metric_name}_h{int(round(1.0 / step))}"] = float(
                (positive[column] - negative[column]) / (2.0 * step)
            )
    fine_positive = values[FD_STEPS[-1]]["a_direct_matrix"]
    fine_negative = values[-FD_STEPS[-1]]["a_direct_matrix"]
    repeat_positive = next(
        float(row["a_direct_matrix"]) for row in repeat_rows if float(row["alpha"]) > 0.0
    )
    repeat_negative = next(
        float(row["a_direct_matrix"]) for row in repeat_rows if float(row["alpha"]) < 0.0
    )
    slopes["slope_a_repeat_h256"] = float(
        (repeat_positive - repeat_negative) / (2.0 * FD_STEPS[-1])
    )
    slopes["slope_a_endpoint_noise_bound_h256"] = float(
        (
            abs(repeat_positive - fine_positive)
            + abs(repeat_negative - fine_negative)
        )
        / (2.0 * FD_STEPS[-1])
    )
    return line_rows, spectrum_rows, repeat_rows, slopes


def _stable_sign(slopes: Mapping[str, float], component: str, sign: str, tolerance: float = 1e-6) -> bool:
    values = [float(slopes[f"slope_{component}_h128"]), float(slopes[f"slope_{component}_h256"])]
    return all(value < -tolerance for value in values) if sign == "down" else all(
        value > tolerance for value in values
    )


def _directional_derivative_reliability(
    analytic_slope: float,
    slopes: Mapping[str, float],
) -> dict[str, float | bool]:
    coarse = float(slopes["slope_a_h128"])
    fine = float(slopes["slope_a_h256"])
    richardson = (4.0 * fine - coarse) / 3.0
    repeat = float(slopes["slope_a_repeat_h256"])
    noise_bound = float(slopes["slope_a_endpoint_noise_bound_h256"])
    values = (float(analytic_slope), coarse, fine, richardson, repeat)
    same_sign = bool(all(value > 0.0 for value in values) or all(value < 0.0 for value in values))
    relative_error = abs(float(analytic_slope) - richardson) / max(
        abs(float(analytic_slope)), abs(richardson), 1e-12
    )
    truncation_bound = abs(fine - coarse) / 3.0
    uncertainty_bound = truncation_bound + noise_bound
    sign_margin = min(abs(value) for value in values)
    reliable = bool(
        same_sign
        and relative_error <= 0.05
        and sign_margin > 5.0 * uncertainty_bound
    )
    return {
        "slope_a_richardson": richardson,
        "slope_a_richardson_relative_error": relative_error,
        "slope_a_truncation_bound": truncation_bound,
        "slope_a_noise_bound": noise_bound,
        "slope_a_uncertainty_bound": uncertainty_bound,
        "slope_a_sign_margin": sign_margin,
        "slope_a_all_signs_agree": same_sign,
        "slope_a_reliable": reliable,
    }


def _classify_mechanisms(
    slopes: Mapping[str, Mapping[str, float]],
    line_frame: pd.DataFrame,
    base: Mapping[str, float],
    reliability: Mapping[str, bool] | None = None,
) -> dict[str, bool | None]:
    reliability = reliability or {name: True for name in DIRECTION_NAMES}
    base_a = float(base["exact_a_per_dim"])
    base_m_max = float(base["m_max"])
    base_high_a = float(base["a_high_gt_1_abs_per_dim"])

    def resolved(required: Sequence[str], supported: bool) -> bool | None:
        return bool(supported) if all(bool(reliability[name]) for name in required) else None

    exact_down = _stable_sign(slopes["exact_a_raw"], "a", "down")
    p32_a_down = _stable_sign(slopes["p32_a_raw"], "a", "down")
    exact_oldbeta_down = _stable_sign(slopes["exact_oldbeta_raw"], "a", "down")
    old_raw_down = _stable_sign(slopes["p32_oldbeta_raw"], "a", "down")
    old_carried_down = _stable_sign(slopes["p32_oldbeta_carried_adam"], "a", "down")
    old_alpha1 = line_frame.loc[
        line_frame["direction"].eq("p32_oldbeta_carried_adam") & np.isclose(line_frame["alpha"], 1.0)
    ]
    smaller = line_frame.loc[
        line_frame["direction"].eq("p32_oldbeta_carried_adam")
        & line_frame["alpha"].isin(POSITIVE_ALPHAS[1:])
    ]
    return {
        "finite_p32_estimator_supported": resolved(
            ("exact_a_raw", "p32_a_raw"),
            exact_down and _stable_sign(slopes["p32_a_raw"], "a", "up"),
        ),
        "exact_oldbeta_scalarization_conflict_supported": resolved(
            ("exact_a_raw", "exact_oldbeta_raw"),
            exact_down and _stable_sign(slopes["exact_oldbeta_raw"], "a", "up"),
        ),
        "p32_composite_estimator_supported": resolved(
            ("exact_oldbeta_raw", "p32_oldbeta_raw"),
            exact_oldbeta_down and _stable_sign(slopes["p32_oldbeta_raw"], "a", "up"),
        ),
        "carried_adam_transform_supported": resolved(
            ("p32_oldbeta_raw", "p32_oldbeta_carried_adam"),
            old_raw_down and _stable_sign(slopes["p32_oldbeta_carried_adam"], "a", "up"),
        ),
        "finite_radius_spike_overshoot_supported": resolved(
            ("p32_oldbeta_carried_adam",),
            bool(
                old_carried_down
                and len(old_alpha1) == 1
                and float(old_alpha1.iloc[0]["exact_a_per_dim"]) > base_a + 1e-8
                and float(old_alpha1.iloc[0]["m_max"]) > base_m_max + 1e-8
                and float(old_alpha1.iloc[0]["a_high_gt_1_abs_per_dim"])
                > base_high_a + 1e-8
                and bool(
                    (
                        (smaller["exact_a_per_dim"] < base_a - 1e-8)
                        & (smaller["delta_m_max"] <= 1e-8)
                        & (
                            smaller["a_high_gt_1_abs_per_dim"]
                            <= base_high_a + 1e-8
                        )
                    ).any()
                )
            ),
        ),
    }


def _candidate_decision(
    line_frame: pd.DataFrame,
    direction: str,
    base: Mapping[str, float],
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    rows = line_frame.loc[
        line_frame["direction"].eq(direction)
        & line_frame["kind"].eq("line")
        & line_frame["alpha"].ge(0.125)
    ].copy()
    rows["passes"] = (
        rows["exact_a_per_dim"].lt(float(base["exact_a_per_dim"]) - tolerances["a"])
        & rows["damped_full_burg_per_dim"].lt(
            float(base["damped_full_burg_per_dim"]) - tolerances["b"]
        )
        & rows["m_max"].le(float(base["m_max"]) + tolerances["m_max"])
        & rows["m_p50"].ge(float(base["m_p50"]) - tolerances["m_p50"])
        & rows["m_lt_0p1_fraction"].le(
            float(base["m_lt_0p1_fraction"]) + tolerances["m_lt_0p1_fraction"]
        )
    )
    passing = rows.loc[rows["passes"]].sort_values("alpha", ascending=False)
    return {
        "direction": direction,
        "selected": bool(len(passing) > 0),
        "largest_passing_alpha": None if passing.empty else float(passing.iloc[0]["alpha"]),
        "passing_alpha_count": int(len(passing)),
        "tolerances": {key: float(value) for key, value in tolerances.items()},
    }


def _plot(output: Path, line_frame: pd.DataFrame, base: Mapping[str, float]) -> None:
    positive = line_frame.loc[line_frame["kind"].eq("line")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    panels = (
        ("delta_a", "Exact A change"),
        ("delta_b", "Full Burg B change"),
        ("delta_m_max", "Maximum M-eigenvalue change"),
        ("delta_m_p50", "Median M-eigenvalue change"),
    )
    for axis, (column, title) in zip(axes.flat, panels, strict=True):
        for direction in DIRECTION_NAMES:
            rows = positive.loc[positive["direction"].eq(direction)].sort_values("alpha")
            axis.plot(rows["alpha"], rows[column], marker="o", label=direction)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("fraction of common-radius direction")
        axis.set_ylabel(column)
        axis.set_title(title)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Proposal-3 exact-A spike discriminator\n"
        f"base A={base['exact_a_per_dim']:.4f}, B={base['damped_full_burg_per_dim']:.4f}, "
        f"m_max={base['m_max']:.4f}"
    )
    fig.savefig(output / "proposal3_exact_a_cross.png", dpi=180)
    plt.close(fig)


def _sha256_manifest(output: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(output / name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    if not math.isclose(args.lr, 3e-5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("frozen protocol requires --lr=3e-5")
    if not math.isclose(args.gradient_clip, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("frozen protocol requires --gradient-clip=1")
    if args.hessian_chunk_size != 64:
        raise ValueError("frozen protocol requires --hessian-chunk-size=64")
    if args.device != "cuda:0":
        raise ValueError("frozen protocol requires --device=cuda:0")
    frozen_dependency_matches = _validate_frozen_dependencies()

    final_output = args.output_dir.resolve()
    staging = Path(str(final_output) + ".incomplete")
    if final_output.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite output: {final_output} or {staging}")
    staging.mkdir(parents=True)
    (staging / "INCOMPLETE").write_text(PROTOCOL_ID + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), staging / "executed_source_snapshot.py")
    shutil.copy2(
        FROZEN_DEPENDENCY_MANIFEST,
        staging / "frozen_dependency_manifest_snapshot.json",
    )
    started = time.perf_counter()
    device = torch.device(args.device)
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "device": str(device),
        "dtype": "float32 parameters / float64 exact accumulators",
        "state_position": 2,
        "source_weight_index": 378,
        "replay_through_proposal": REPLAY_THROUGH,
        "target_proposal": TARGET_PROPOSAL,
        "beta": OLD_BETA,
        "burg_epsilon": EPSILON,
        "gradient_clip": args.gradient_clip,
        "lr": args.lr,
        "hessian_chunk_size": args.hessian_chunk_size,
        "directions": list(DIRECTION_NAMES),
        "positive_alphas": list(POSITIVE_ALPHAS),
        "fd_steps": list(FD_STEPS),
        "output_dir": str(final_output),
        "cache_mode": "reuse accepted checkpoint/state bank; recompute all gradients and dense endpoints",
        "frozen_dependency_manifest": str(FROZEN_DEPENDENCY_MANIFEST),
        "frozen_dependency_manifest_sha256": EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        "frozen_dependency_matches": frozen_dependency_matches,
        "source_sha256": _source_sha256(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "iteration5_states_sha256": sha256_file(ITERATION5_STATES),
        "iteration5_proposals_sha256": sha256_file(ITERATION5_PROPOSALS),
        "p32_dependency_sha256": sha256_file(Path(p32.__file__)),
        "direction_helper_dependency_sha256": sha256_file(
            ROOT / "scripts/audit_one_state_a_full_burg_direction_mismatch.py"
        ),
        "metric_helper_dependency_sha256": sha256_file(
            ROOT / "scripts/smoke_optimize_variant_a_full_burg_one_state.py"
        ),
    }
    (staging / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[exact-a-i1] start {json.dumps(resolved, sort_keys=True)}", flush=True)

    try:
        print("[exact-a-i1] stage=load", flush=True)
        if sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt") != EXPECTED_CHECKPOINT:
            raise RuntimeError("accepted checkpoint hash mismatch")
        run = _load_run(DEFAULT_RUN_DIR, device=device)
        cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
        cfg = replace(cfg, vae_precond_hvp_mode="autograd")
        bank = pd.read_csv(STATE_BANK)
        selected = bank.loc[bank["state_position"].eq(2)]
        if len(selected) != 1 or int(selected.iloc[0]["source_weight_index"]) != 378:
            raise RuntimeError("frozen state-bank identity mismatch")
        source_index = 378
        record = run.records.iloc[source_index].to_dict()
        record["source_weight_index"] = source_index
        weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
        with torch.no_grad():
            z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
        task_set = _task_set_for_record(run.task_tensors, record)
        full_ce = all(
            _batch_indices(
                task_set,
                batch_size=int(cfg.vae_precond_batch_size),
                step=10,
                sample_key=source_index,
                pair_key=pair_key,
            )
            is None
            for pair_key in range(8)
        )
        active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
        named = dict(run.vae.named_parameters())
        for name, parameter in named.items():
            parameter.requires_grad_(name in active_names)
        active = [named[name] for name in active_names]
        z_hash = sha256_tensor(z)
        initial_parameter_hash = _named_tensor_hash(active_names, active)
        if z_hash != EXPECTED_Z_SHA256 or initial_parameter_hash != EXPECTED_INITIAL_PARAMETER_SHA256:
            raise RuntimeError("initial z/parameter fingerprint mismatch")
        exp_avg = [torch.zeros_like(parameter) for parameter in active]
        exp_avg_sq = [torch.zeros_like(parameter) for parameter in active]
        accepted_step = 0
        _reset_p32_module_state()
        production_states = pd.read_csv(ITERATION5_STATES)
        production_proposals = pd.read_csv(ITERATION5_PROPOSALS)
        replay_rows: list[dict[str, float | int]] = []

        for proposal in range(1, REPLAY_THROUGH + 1):
            print(f"[exact-a-i1] stage=reconstruct proposal={proposal}/{REPLAY_THROUGH}", flush=True)
            current, _h, _m, _eig, burg_gradient = _evaluate_dense(
                run=run,
                z=z,
                record=record,
                epsilon=EPSILON,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            del _h, _m, _eig
            treatment = p32._build_p32_proposal(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                active=active,
                burg_gradient=burg_gradient,
                beta=OLD_BETA,
                proposal=proposal,
                pairs=4,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                accepted_adam_step=accepted_step,
                gradient_clip=args.gradient_clip,
                lr=args.lr,
            )
            expected = production_proposals.loc[production_proposals["proposal"].eq(proposal)].iloc[0]
            if int(expected["accepted"]) != 1 or not math.isclose(float(expected["accepted_alpha"]), 1.0):
                raise RuntimeError("frozen first two proposals must be accepted at alpha 1")
            base = [parameter.detach().clone() for parameter in active]
            _set_parameters(active, base, treatment["displacement"], 1.0)
            exp_avg = treatment["next_avg"]
            exp_avg_sq = treatment["next_avg_sq"]
            accepted_step += 1
            after, _h, _m, _eig, _bg = _evaluate_dense(
                run=run,
                z=z,
                record=record,
                epsilon=EPSILON,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            del _h, _m, _eig, _bg, burg_gradient
            frozen_after = production_states.loc[production_states["proposal"].eq(proposal)].iloc[0]
            component_stats = treatment["component_stats"]

            replay_rows.append(
                {
                    "proposal": proposal,
                    "a_abs_error": abs(float(after["exact_a_per_dim"]) - float(frozen_after["exact_a_per_dim"])),
                    "b_abs_error": abs(
                        float(after["damped_full_burg_per_dim"])
                        - float(frozen_after["damped_full_burg_per_dim"])
                    ),
                    "f_abs_error": abs(float(after["true_objective"]) - float(frozen_after["true_objective"])),
                    "m_max_abs_error": abs(float(after["m_max"]) - float(frozen_after["m_max"])),
                    "proposal_norm_abs_error": abs(
                        float(treatment["proposal_norm"]) - float(expected["proposal_norm"])
                    ),
                    "train_a_abs_error": abs(float(treatment["mean_a"]) - float(expected["train_a"])),
                    "train_b_abs_error": abs(
                        float(treatment["mean_b"]) - float(expected["train_b_pseudo_loss"])
                    ),
                    "grad_a_relative_error": _relative_scalar_error(
                        component_stats["grad_a_norm"], expected["grad_a_norm"]
                    ),
                    "grad_b_relative_error": _relative_scalar_error(
                        component_stats["grad_b_norm"], expected["grad_b_norm"]
                    ),
                    "grad_total_relative_error": _relative_scalar_error(
                        component_stats["grad_total_norm"], expected["grad_total_norm"]
                    ),
                    "grad_cosine_abs_error": abs(
                        float(component_stats["grad_a_b_cosine"])
                        - float(expected["grad_a_b_cosine"])
                    ),
                    "preclip_relative_error": _relative_scalar_error(
                        treatment["preclip_norm"], expected["grad_total_norm_clip_api"]
                    ),
                    "clip_factor_abs_error": abs(
                        float(treatment["clip_factor"]) - float(expected["clip_factor"])
                    ),
                    "gradient_dot_displacement_abs_error": abs(
                        float(treatment["gradient_dot_displacement"])
                        - float(expected["stochastic_gradient_dot_displacement"])
                    ),
                }
            )
            del treatment, current, after, base

        base_metrics, hessian, matrix, eig_m, burg_gradient = _evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        base = [parameter.detach().clone() for parameter in active]
        base_parameter_hash = _named_tensor_hash(active_names, active)
        base_moment_hash = _named_tensor_hash(
            [f"m:{name}" for name in active_names] + [f"v:{name}" for name in active_names],
            list(exp_avg) + list(exp_avg_sq),
        )
        frozen_base = production_states.loc[production_states["proposal"].eq(REPLAY_THROUGH)].iloc[0]
        state2_errors = {
            key: abs(float(base_metrics[column]) - float(frozen_base[column]))
            for key, column in (
                ("a", "exact_a_per_dim"),
                ("b", "damped_full_burg_per_dim"),
                ("f", "true_objective"),
                ("m_max", "m_max"),
            )
        }
        pd.DataFrame(replay_rows).to_csv(staging / "reconstruction.csv", index=False)
        replay_error_values = [
            float(value)
            for row in replay_rows
            for key, value in row.items()
            if key.endswith("_abs_error")
        ]
        replay_relative_values = [
            float(value)
            for row in replay_rows
            for key, value in row.items()
            if key.endswith("_relative_error")
        ]
        if (
            not replay_error_values
            or max(replay_error_values) > 1e-5
            or not replay_relative_values
            or max(replay_relative_values) > 1e-6
            or max(state2_errors.values()) > 1e-5
            or accepted_step != REPLAY_THROUGH
        ):
            raise RuntimeError("frozen proposal-1/2 reconstruction preflight failed")

        print("[exact-a-i1] stage=base-repeatability", flush=True)
        repeat_metrics, _rh, _rm, _reig, _rburg = _evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        del _rh, _rm, _reig, _rburg
        base_repeatability_columns = {
            "a": "exact_a_per_dim",
            "b": "damped_full_burg_per_dim",
            "m_max": "m_max",
            "m_p50": "m_p50",
            "m_lt_0p1_fraction": "m_lt_0p1_fraction",
        }
        base_repeatability_noise = {
            key: abs(float(base_metrics[column]) - float(repeat_metrics[column]))
            for key, column in base_repeatability_columns.items()
        }
        candidate_tolerances = {
            key: max(
                5.0 * base_repeatability_noise[key],
                1e-8 if key in {"a", "b", "m_max"} else 1e-12,
            )
            for key in base_repeatability_columns
        }
        pd.DataFrame(
            [
                {"evaluation": "primary", **base_metrics},
                {"evaluation": "repeat", **repeat_metrics},
            ]
        ).to_csv(staging / "base_repeatability.csv", index=False)
        if max(base_repeatability_noise.values()) > 1e-6:
            raise RuntimeError("dense base evaluation is not repeatable enough for line decisions")
        base_cpu_rng_hash = sha256_tensor(torch.get_rng_state())
        base_cuda_rng_hash = sha256_tensor(torch.cuda.get_rng_state(device))

        print("[exact-a-i1] stage=exact-a-gradient", flush=True)
        dim = int(z.numel())
        identity = torch.eye(dim, device=device, dtype=torch.float64)
        h64 = hessian.double()
        m64 = matrix.double()
        k_a = (4.0 * (m64 - identity) @ h64 / float(dim)).detach()
        g_b_matrix = identity / (float(dim) * (1.0 + EPSILON)) - torch.linalg.inv(
            m64 + EPSILON * identity
        ) / float(dim)
        k_b = (2.0 * g_b_matrix @ h64).detach()
        k_oldbeta = (k_a + OLD_BETA * k_b).detach()
        h_space_a_rows = _h_space_fd_check(
            hessian,
            k_a,
            beta=0.0,
            epsilon=EPSILON,
        )
        h_space_oldbeta_rows = _h_space_fd_check(
            hessian,
            k_oldbeta,
            beta=OLD_BETA,
            epsilon=EPSILON,
        )
        h_space_rows = [
            {"objective": objective, **row}
            for objective, rows in (
                ("a", h_space_a_rows),
                ("oldbeta", h_space_oldbeta_rows),
            )
            for row in rows
        ]
        pd.DataFrame(h_space_rows).to_csv(staging / "exact_objective_h_space_fd.csv", index=False)
        exact_a_gradient, exact_meta = _full_basis_gradient(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            active=active,
            k_cotangent=k_a,
            mode="autograd",
            basis_count=dim,
        )
        print("[exact-a-i1] stage=exact-b-gradient", flush=True)
        exact_b_gradient, exact_b_meta = _full_basis_gradient(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            active=active,
            k_cotangent=k_b,
            mode="autograd",
            basis_count=dim,
        )
        exact_oldbeta_gradient = _vector_sum(
            exact_a_gradient,
            exact_b_gradient,
            left_scale=1.0,
            right_scale=OLD_BETA,
            active=active,
        )

        print("[exact-a-i1] stage=proposal3-p32-components", flush=True)
        p32_a, p32_b, pair_rows, draw_rows = _p32_components(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            active=active,
            burg_gradient=burg_gradient,
            proposal=TARGET_PROPOSAL,
        )
        pd.DataFrame(pair_rows).to_csv(staging / "p32_pair_scalars.csv", index=False)
        pd.DataFrame(draw_rows).to_csv(staging / "p32_draw_diagnostics.csv", index=False)

        oldbeta_gradient = _vector_sum(
            p32_a, p32_b, left_scale=1.0, right_scale=OLD_BETA, active=active
        )
        common_gradient = _unit_common_gradient(p32_a, p32_b, active)
        canonical_displacement, canonical_preclip_norm, canonical_clip_factor = _adam_direction(
            gradient=oldbeta_gradient,
            active=active,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            accepted_step=accepted_step,
            gradient_clip=args.gradient_clip,
            lr=args.lr,
        )
        common_adam_displacement, common_preclip_norm, common_clip_factor = _adam_direction(
            gradient=common_gradient,
            active=active,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            accepted_step=accepted_step,
            gradient_clip=args.gradient_clip,
            lr=args.lr,
        )
        target_norm = _norm(canonical_displacement)
        directions: dict[str, Vector] = {
            "exact_a_raw": _negative_normalized(exact_a_gradient, target_norm=target_norm, active=active),
            "p32_a_raw": _negative_normalized(p32_a, target_norm=target_norm, active=active),
            "exact_oldbeta_raw": _negative_normalized(
                exact_oldbeta_gradient, target_norm=target_norm, active=active
            ),
            "p32_oldbeta_raw": _negative_normalized(oldbeta_gradient, target_norm=target_norm, active=active),
            "p32_oldbeta_carried_adam": _scaled(
                canonical_displacement, target_norm / _norm(canonical_displacement), active
            ),
            "p32_unit_common_carried_adam": _scaled(
                common_adam_displacement, target_norm / _norm(common_adam_displacement), active
            ),
        }
        frozen_p3_proposal = production_proposals.loc[
            production_proposals["proposal"].eq(TARGET_PROPOSAL)
        ].iloc[0]
        p32_component_stats = _gradient_norms(tuple(p32_a), tuple(p32_b), beta=OLD_BETA)
        clipped_oldbeta = _scaled(oldbeta_gradient, canonical_clip_factor, active)
        canonical_gradient_dot_displacement = _dot(clipped_oldbeta, canonical_displacement)
        proposal3_component_errors = {
            "train_a_abs_error": abs(
                float(np.mean([row["a_loss"] for row in pair_rows]))
                - float(frozen_p3_proposal["train_a"])
            ),
            "train_b_abs_error": abs(
                float(np.mean([row["b_pseudo_loss"] for row in pair_rows]))
                - float(frozen_p3_proposal["train_b_pseudo_loss"])
            ),
            "grad_a_relative_error": _relative_scalar_error(
                p32_component_stats["grad_a_norm"], frozen_p3_proposal["grad_a_norm"]
            ),
            "grad_b_relative_error": _relative_scalar_error(
                p32_component_stats["grad_b_norm"], frozen_p3_proposal["grad_b_norm"]
            ),
            "grad_total_relative_error": _relative_scalar_error(
                p32_component_stats["grad_total_norm"], frozen_p3_proposal["grad_total_norm"]
            ),
            "grad_cosine_abs_error": abs(
                float(p32_component_stats["grad_a_b_cosine"])
                - float(frozen_p3_proposal["grad_a_b_cosine"])
            ),
            "preclip_relative_error": _relative_scalar_error(
                canonical_preclip_norm, frozen_p3_proposal["grad_total_norm_clip_api"]
            ),
            "clip_factor_abs_error": abs(
                canonical_clip_factor - float(frozen_p3_proposal["clip_factor"])
            ),
            "proposal_norm_abs_error": abs(
                target_norm - float(frozen_p3_proposal["proposal_norm"])
            ),
            "gradient_dot_displacement_abs_error": abs(
                canonical_gradient_dot_displacement
                - float(frozen_p3_proposal["stochastic_gradient_dot_displacement"])
            ),
        }
        pd.DataFrame([proposal3_component_errors]).to_csv(
            staging / "proposal3_component_reconstruction.csv", index=False
        )
        proposal3_abs_errors = [
            float(value)
            for key, value in proposal3_component_errors.items()
            if key.endswith("_abs_error")
        ]
        proposal3_relative_errors = [
            float(value)
            for key, value in proposal3_component_errors.items()
            if key.endswith("_relative_error")
        ]
        if max(proposal3_abs_errors) > 1e-5 or max(proposal3_relative_errors) > 1e-6:
            raise RuntimeError("frozen proposal-3 component reconstruction preflight failed")

        direction_rows: list[dict[str, float | str]] = []
        all_line_rows: list[dict[str, float | str]] = []
        all_spectrum_rows: list[dict[str, float | int | str]] = []
        all_repeat_rows: list[dict[str, float | int | str]] = []
        slopes_by_direction: dict[str, dict[str, float]] = {}
        for direction_name in DIRECTION_NAMES:
            direction = directions[direction_name]
            line_rows, spectrum_rows, repeat_rows, slopes = _line_profile(
                direction_name=direction_name,
                direction=direction,
                run=run,
                z=z,
                record=record,
                active_names=active_names,
                active=active,
                base=base,
                base_parameter_hash=base_parameter_hash,
                base_metrics=base_metrics,
                epsilon=EPSILON,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            slopes_by_direction[direction_name] = slopes
            all_line_rows.extend(line_rows)
            all_spectrum_rows.extend(spectrum_rows)
            all_repeat_rows.extend(repeat_rows)
            native_source_norm = {
                "exact_a_raw": _norm(exact_a_gradient),
                "p32_a_raw": _norm(p32_a),
                "exact_oldbeta_raw": _norm(exact_oldbeta_gradient),
                "p32_oldbeta_raw": _norm(oldbeta_gradient),
                "p32_oldbeta_carried_adam": _norm(canonical_displacement),
                "p32_unit_common_carried_adam": _norm(common_adam_displacement),
            }[direction_name]
            analytic_slope = _dot(exact_a_gradient, direction)
            analytic_b_slope = _dot(exact_b_gradient, direction)
            direction_rows.append(
                {
                    "direction": direction_name,
                    "native_source_norm": native_source_norm,
                    "common_direction_norm": _norm(direction),
                    "cosine_to_exact_a_gradient": _cosine(direction, exact_a_gradient),
                    "cosine_to_exact_oldbeta_gradient": _cosine(
                        direction, exact_oldbeta_gradient
                    ),
                    "analytic_exact_a_slope": analytic_slope,
                    "analytic_exact_b_slope": analytic_b_slope,
                    "analytic_exact_oldbeta_slope": analytic_slope
                    + OLD_BETA * analytic_b_slope,
                    **slopes,
                    **_directional_derivative_reliability(analytic_slope, slopes),
                }
            )

        line_frame = pd.DataFrame(all_line_rows)
        direction_frame = pd.DataFrame(direction_rows)
        exact_scale = max(_norm(exact_a_gradient) * target_norm, 1e-30)
        exact_oldbeta_scale = max(_norm(exact_oldbeta_gradient) * target_norm, 1e-30)
        for denominator in (128, 256):
            fd_column = f"slope_a_h{denominator}"
            error_column = f"exact_a_fd_normalized_error_h{denominator}"
            direction_frame[error_column] = (
                direction_frame[fd_column] - direction_frame["analytic_exact_a_slope"]
            ).abs() / exact_scale
            direction_frame[f"exact_oldbeta_fd_normalized_error_h{denominator}"] = (
                direction_frame[f"slope_f_h{denominator}"]
                - direction_frame["analytic_exact_oldbeta_slope"]
            ).abs() / exact_oldbeta_scale
        line_frame.to_csv(staging / "line_profiles.csv", index=False)
        direction_frame.to_csv(staging / "direction_diagnostics.csv", index=False)
        pd.DataFrame(all_spectrum_rows).to_csv(staging / "line_spectra.csv", index=False)
        pd.DataFrame(all_repeat_rows).to_csv(staging / "fd_repeatability.csv", index=False)

        frozen_p3 = production_states.loc[production_states["proposal"].eq(TARGET_PROPOSAL)].iloc[0]
        canonical_endpoint = line_frame.loc[
            line_frame["direction"].eq("p32_oldbeta_carried_adam")
            & np.isclose(line_frame["alpha"], 1.0)
        ].iloc[0]
        canonical_endpoint_errors = {
            key: abs(float(canonical_endpoint[column]) - float(frozen_p3[column]))
            for key, column in (
                ("a", "exact_a_per_dim"),
                ("b", "damped_full_burg_per_dim"),
                ("f", "true_objective"),
                ("m_max", "m_max"),
            )
        }
        canonical_proposal_norm_error = abs(target_norm - float(frozen_p3_proposal["proposal_norm"]))
        direction_reliability = {
            str(row.direction): bool(row.slope_a_reliable)
            for row in direction_frame.itertuples(index=False)
        }
        mechanisms = _classify_mechanisms(
            slopes_by_direction,
            line_frame,
            base_metrics,
            direction_reliability,
        )
        candidate_carried = _candidate_decision(
            line_frame,
            "p32_unit_common_carried_adam",
            base_metrics,
            candidate_tolerances,
        )
        final_parameter_hash = _named_tensor_hash(active_names, active)
        final_moment_hash = _named_tensor_hash(
            [f"m:{name}" for name in active_names] + [f"v:{name}" for name in active_names],
            list(exp_avg) + list(exp_avg_sq),
        )
        final_cpu_rng_hash = sha256_tensor(torch.get_rng_state())
        final_cuda_rng_hash = sha256_tensor(torch.cuda.get_rng_state(device))
        numeric_finite = bool(
            np.isfinite(line_frame.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(direction_frame.select_dtypes(include=[np.number]).to_numpy()).all()
        )
        exact_raw_row = direction_frame.loc[direction_frame["direction"].eq("exact_a_raw")].iloc[0]
        exact_raw_fd_relative_errors = {
            f"h{denominator}": abs(
                float(exact_raw_row[f"slope_a_h{denominator}"])
                - float(exact_raw_row["analytic_exact_a_slope"])
            )
            / max(
                abs(float(exact_raw_row[f"slope_a_h{denominator}"])),
                abs(float(exact_raw_row["analytic_exact_a_slope"])),
                1e-30,
            )
            for denominator in (128, 256)
        }
        exact_oldbeta_row = direction_frame.loc[
            direction_frame["direction"].eq("exact_oldbeta_raw")
        ].iloc[0]
        exact_oldbeta_fd_relative_errors = {
            f"h{denominator}": abs(
                float(exact_oldbeta_row[f"slope_f_h{denominator}"])
                - float(exact_oldbeta_row["analytic_exact_oldbeta_slope"])
            )
            / max(
                abs(float(exact_oldbeta_row[f"slope_f_h{denominator}"])),
                abs(float(exact_oldbeta_row["analytic_exact_oldbeta_slope"])),
                1e-30,
            )
            for denominator in (128, 256)
        }
        validity = {
            "accepted_checkpoint_matches": sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt") == EXPECTED_CHECKPOINT,
            "z_hash_matches": z_hash == EXPECTED_Z_SHA256,
            "initial_parameter_hash_matches": initial_parameter_hash == EXPECTED_INITIAL_PARAMETER_SHA256,
            "source_state_matches": source_index == 378,
            "full_ce": full_ce,
            "active_parameter_count_matches": sum(parameter.numel() for parameter in active) == 11_685_120,
            "reconstruction_errors_at_most_1e_5": bool(replay_error_values)
            and max(replay_error_values) <= 1e-5,
            "reconstruction_relative_errors_at_most_1e_6": bool(replay_relative_values)
            and max(replay_relative_values) <= 1e-6,
            "accepted_clock_is_two": accepted_step == REPLAY_THROUGH,
            "state2_errors_at_most_1e_5": max(state2_errors.values()) <= 1e-5,
            "base_repeatability_noise_at_most_1e_6": max(
                base_repeatability_noise.values()
            )
            <= 1e-6,
            "canonical_endpoint_errors_at_most_1e_5": max(canonical_endpoint_errors.values()) <= 1e-5,
            "canonical_proposal_norm_error_at_most_1e_6": canonical_proposal_norm_error <= 1e-6,
            "proposal3_component_abs_errors_at_most_1e_5": max(
                proposal3_abs_errors
            )
            <= 1e-5,
            "proposal3_component_relative_errors_at_most_1e_6": max(
                proposal3_relative_errors
            )
            <= 1e-6,
            "exact_complete_basis_512": int(exact_meta["basis_count"]) == 512,
            "exact_b_complete_basis_512": int(exact_b_meta["basis_count"]) == 512,
            "exact_objective_h_space_fd_relative_error_at_most_1e_5": max(
                float(row["relative_error"]) for row in h_space_rows
            )
            <= 1e-5,
            "exact_no_fully_unused_parameter_tensors": int(exact_meta["unused_parameter_tensors_all_rows"]) == 0,
            "exact_b_no_fully_unused_parameter_tensors": int(
                exact_b_meta["unused_parameter_tensors_all_rows"]
            )
            == 0,
            "exact_memory_growth_at_most_64mib": bool(exact_meta["memory_growth_gate_pass"]),
            "exact_b_memory_growth_at_most_64mib": bool(
                exact_b_meta["memory_growth_gate_pass"]
            ),
            "p32_pair_rows_32": len(pair_rows) == 32,
            "p32_draw_rows_8": len(draw_rows) == 8,
            "p32_seed_rows_unique": len(
                {
                    seed
                    for row in pair_rows
                    for seed in (int(row["seed_1"]), int(row["seed_2"]))
                }
            )
            == 64,
            "six_directions": set(direction_frame["direction"]) == set(DIRECTION_NAMES),
            "common_direction_norms_match": bool(
                np.allclose(direction_frame["common_direction_norm"], target_norm, rtol=1e-6, atol=1e-9)
            ),
            "unit_common_gradient_nonzero": _norm(common_gradient) > 1e-6,
            "unit_common_clip_active_and_consistent": bool(
                common_preclip_norm > args.gradient_clip
                and common_clip_factor < 1.0
                and math.isclose(
                    common_clip_factor,
                    args.gradient_clip / common_preclip_norm,
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                )
            ),
            "line_rows_66": len(line_frame) == len(DIRECTION_NAMES) * 11,
            "fd_repeat_rows_12": len(all_repeat_rows) == len(DIRECTION_NAMES) * 2,
            "spectrum_rows_33792": len(all_spectrum_rows) == len(DIRECTION_NAMES) * 11 * 512,
            "numeric_finite": numeric_finite,
            "a_closure_at_most_1e_10": float(line_frame[["a_direct_abs_error", "a_trace_abs_error"]].to_numpy().max())
            <= 1e-10,
            "exact_a_raw_fd_relative_error_at_most_5pct": max(
                exact_raw_fd_relative_errors.values()
            )
            <= 0.05,
            "exact_a_raw_directional_derivative_reliable": bool(
                direction_reliability["exact_a_raw"]
            ),
            "exact_oldbeta_raw_fd_relative_error_at_most_5pct": max(
                exact_oldbeta_fd_relative_errors.values()
            )
            <= 0.05,
            "parameter_hash_restored": final_parameter_hash == base_parameter_hash,
            "moment_hash_unchanged": final_moment_hash == base_moment_hash,
            "cpu_rng_state_unchanged": final_cpu_rng_hash == base_cpu_rng_hash,
            "cuda_rng_state_unchanged": final_cuda_rng_hash == base_cuda_rng_hash,
        }
        valid = all(validity.values())
        decision = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "validity_gates": validity,
            "base_metrics": base_metrics,
            "base_repeatability_noise": base_repeatability_noise,
            "candidate_tolerances": candidate_tolerances,
            "state2_errors": state2_errors,
            "canonical_endpoint_errors": canonical_endpoint_errors,
            "canonical_proposal_norm_error": canonical_proposal_norm_error,
            "proposal3_component_errors": proposal3_component_errors,
            "target_direction_norm": target_norm,
            "p32_a_cosine_to_exact_a": _cosine(p32_a, exact_a_gradient),
            "p32_a_relative_error_to_exact_a": _relative_error(p32_a, exact_a_gradient),
            "p32_b_cosine_to_exact_b": _cosine(p32_b, exact_b_gradient),
            "p32_b_relative_error_to_exact_b": _relative_error(p32_b, exact_b_gradient),
            "p32_oldbeta_cosine_to_exact_oldbeta": _cosine(
                oldbeta_gradient, exact_oldbeta_gradient
            ),
            "p32_oldbeta_relative_error_to_exact_oldbeta": _relative_error(
                oldbeta_gradient, exact_oldbeta_gradient
            ),
            "p32_component_cosine": _cosine(p32_a, p32_b),
            "p32_component_norms": {"a": _norm(p32_a), "b": _norm(p32_b)},
            "exact_a_raw_fd_relative_errors": exact_raw_fd_relative_errors,
            "exact_oldbeta_raw_fd_relative_errors": exact_oldbeta_fd_relative_errors,
            "directional_derivative_reliability": direction_reliability,
            "canonical_preclip_norm": canonical_preclip_norm,
            "canonical_clip_factor": canonical_clip_factor,
            "common_preclip_norm": common_preclip_norm,
            "common_clip_factor": common_clip_factor,
            "mechanisms": mechanisms if valid else None,
            "candidate_carried": candidate_carried if valid else None,
            "elapsed_sec": time.perf_counter() - started,
            "source_sha256": _source_sha256(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
        }
        _plot(staging, line_frame, base_metrics)
        (staging / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
        artifact_names = (
            "resolved_config.json",
            "executed_source_snapshot.py",
            "frozen_dependency_manifest_snapshot.json",
            "reconstruction.csv",
            "base_repeatability.csv",
            "exact_objective_h_space_fd.csv",
            "p32_pair_scalars.csv",
            "p32_draw_diagnostics.csv",
            "proposal3_component_reconstruction.csv",
            "line_profiles.csv",
            "direction_diagnostics.csv",
            "line_spectra.csv",
            "fd_repeatability.csv",
            "proposal3_exact_a_cross.png",
            "decision.json",
        )
        artifact_manifest = _sha256_manifest(staging, artifact_names)
        (staging / "artifact_manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        completed = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "decision_sha256": sha256_file(staging / "decision.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        (staging / "COMPLETED.json").write_text(json.dumps(completed, indent=2, sort_keys=True), encoding="utf-8")
        (staging / "INCOMPLETE").unlink()
        staging.replace(final_output)
        print(f"[exact-a-i1] complete {json.dumps(decision, sort_keys=True)}", flush=True)
        print(f"[exact-a-i1] artifacts={final_output}", flush=True)
        if not valid:
            raise RuntimeError("Iteration-1 validity gates failed")
    except Exception:
        print(f"[exact-a-i1] failed; partial artifacts remain in {staging}", flush=True)
        raise


if __name__ == "__main__":
    main()
