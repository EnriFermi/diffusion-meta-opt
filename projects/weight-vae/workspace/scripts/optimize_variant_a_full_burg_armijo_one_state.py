from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import encode_weights
from scripts.audit_variant_a_estimator_stability import DEFAULT_RUN_DIR, _load_run, _probe_cfg, sha256_file, sha256_tensor
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    STATE_BANK,
    _gradient_norms,
    _materialize_metric,
    _operator_validation,
    _pair_losses,
    _train_generators,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048"
    / "iteration_1_armijo"
)
SOURCE_RUN = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/production_100steps"
)
PROTOCOL_ID = "one_state_a_full_burg_h2048_armijo_v1"
EXPECTED_BETA = 22.536727828943093
EXPECTED_Z = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_INITIAL_F = 174.615076216640
OLD_BEST_F = 140.187110031985


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _true_f(row: Mapping[str, float], *, beta: float) -> float:
    return float(row["exact_a_per_dim"]) + float(beta) * float(row["damped_full_burg_per_dim"])


def _copy_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _set_parameters(
    parameters: list[torch.nn.Parameter],
    base: list[torch.Tensor],
    displacement: list[torch.Tensor] | None = None,
    *,
    alpha: float = 0.0,
) -> None:
    with torch.no_grad():
        for index, (parameter, origin) in enumerate(zip(parameters, base, strict=True)):
            if displacement is None:
                parameter.copy_(origin)
            else:
                parameter.copy_(origin).add_(displacement[index], alpha=float(alpha))


def _vector_norm(values: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.detach().double().square().sum().cpu()) for value in values))


def _evaluate(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    epsilon: float,
    beta: float,
    hessian_chunk_size: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row, hessian, metric, eig_m, burg_gradient = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=epsilon,
        hessian_chunk_size=hessian_chunk_size,
    )
    row["true_dense_f"] = _true_f(row, beta=beta)
    return row, hessian, metric, eig_m, burg_gradient


def _candidate_gradient(
    *,
    run: Any,
    cfg: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: list[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    beta: float,
    proposal: int,
    pairs: int,
    gradient_clip: float,
) -> tuple[dict[str, float], list[torch.Tensor]]:
    a_losses: list[torch.Tensor] = []
    b_losses: list[torch.Tensor] = []
    h_norms: list[float] = []
    h_dot2: list[float] = []
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
        a_losses.append(a_loss)
        b_losses.append(b_loss)
        h_norms.append(float(stats["h_norm2_per_dim"]))
        h_dot2.append(float(stats["h_dot2_per_dim"]))
    mean_a = torch.stack(a_losses).mean()
    mean_b = torch.stack(b_losses).mean()
    gradients_a = torch.autograd.grad(mean_a, active, retain_graph=True, allow_unused=True)
    gradients_b = torch.autograd.grad(mean_b, active, retain_graph=False, allow_unused=True)
    gradient_stats = _gradient_norms(gradients_a, gradients_b, beta=beta)
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
    preclip_norm = float(torch.nn.utils.clip_grad_norm_(active, max_norm=gradient_clip).detach().cpu())
    if not math.isfinite(preclip_norm):
        raise RuntimeError(f"non-finite gradient norm at proposal {proposal}: {preclip_norm}")
    clipped = [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in active
    ]
    row = {
        "train_a": float(mean_a.detach().cpu()),
        "train_b_pseudo_loss": float(mean_b.detach().cpu()),
        **gradient_stats,
        "grad_total_norm_clip_api": preclip_norm,
        "clip_factor": min(1.0, float(gradient_clip) / max(preclip_norm, 1e-30)),
        "clipped_gradient_norm": _vector_norm(clipped),
        "h_norm2_per_dim_mean": float(np.mean(h_norms)),
        "h_dot2_per_dim_mean": float(np.mean(h_dot2)),
    }
    del a_losses, b_losses, mean_a, mean_b, gradients_a, gradients_b
    return row, clipped


def _dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return sum(
        float((a.detach().double() * b.detach().double()).sum().cpu())
        for a, b in zip(left, right, strict=True)
    )


def _plot(
    states: pd.DataFrame,
    updates: pd.DataFrame,
    candidates: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes[0, 0].plot(states["proposal"], states["true_dense_f"], color="#146c43", linewidth=2, label="accepted state F")
    if not updates.empty:
        axes[0, 0].scatter(
            updates["proposal"],
            updates["counterfactual_full_f"],
            color="#b42318",
            marker="x",
            s=25,
            alpha=0.75,
            label="counterfactual alpha=1",
        )
    axes[0, 0].axhline(OLD_BEST_F, color="black", linestyle="--", linewidth=1, label="old best F")
    axes[0, 0].set(xlabel="gradient proposal", ylabel="true dense F", title="Objective-aware trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(states["proposal"], states["exact_a_per_dim"], label="A")
    axes[0, 1].plot(states["proposal"], states["beta_times_b"], label="beta * B")
    axes[0, 1].set(xlabel="gradient proposal", ylabel="objective component", title="Dense F components")
    axes[0, 1].legend()

    if not updates.empty:
        accepted = updates["accepted"].astype(bool)
        axes[0, 2].scatter(
            updates.loc[accepted, "proposal"],
            updates.loc[accepted, "accepted_alpha"],
            color="#146c43",
            s=24,
            label="accepted",
        )
        axes[0, 2].scatter(
            updates.loc[~accepted, "proposal"],
            np.full(int((~accepted).sum()), 1.0 / 128.0),
            color="#b42318",
            marker="x",
            s=30,
            label="rejected",
        )
        axes[0, 2].set_yscale("log", base=2)
    axes[0, 2].set(xlabel="gradient proposal", ylabel="alpha", title="Line-search utilization")
    axes[0, 2].legend()

    if not updates.empty:
        axes[1, 0].plot(updates["proposal"], updates["fd_slope_128"], label="FD slope 1/128")
        axes[1, 0].plot(updates["proposal"], updates["fd_slope_256"], label="FD slope 1/256", alpha=0.75)
        axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="gradient proposal", ylabel="dF / d alpha", title="True local directional slope")
    axes[1, 0].legend()

    if not updates.empty:
        axes[1, 1].plot(updates["proposal"], updates["counterfactual_full_delta_f"], label="full-step delta F")
        axes[1, 1].plot(updates["proposal"], updates["accepted_delta_f"], label="accepted delta F")
        axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set(xlabel="gradient proposal", ylabel="delta F", title="Paired counterfactual")
    axes[1, 1].legend()

    profile = candidates.loc[candidates["role"].eq("backtrack")]
    if not profile.empty:
        increasing = profile["delta_f"] > 0
        axes[1, 2].scatter(profile.loc[~increasing, "alpha"], profile.loc[~increasing, "delta_f"], s=16, alpha=0.55, label="decrease")
        axes[1, 2].scatter(profile.loc[increasing, "alpha"], profile.loc[increasing, "delta_f"], s=16, alpha=0.55, label="increase")
        axes[1, 2].set_xscale("log", base=2)
    axes[1, 2].axhline(0.0, color="black", linewidth=1)
    axes[1, 2].set(xlabel="alpha", ylabel="candidate delta F", title="All evaluated backtracking points")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-position", type=int, default=2)
    parser.add_argument("--proposals", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--beta", type=float, default=EXPECTED_BETA)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument("--backtrack-ratio", type=float, default=0.5)
    parser.add_argument("--min-alpha", type=float, default=1.0 / 64.0)
    parser.add_argument("--fd-alpha-1", type=float, default=1.0 / 128.0)
    parser.add_argument("--fd-alpha-2", type=float, default=1.0 / 256.0)
    parser.add_argument("--determinism-tolerance", type=float, default=1e-5)
    parser.add_argument("--strict-decrease-tolerance", type=float, default=1e-7)
    parser.add_argument("--profile-all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "production_100proposals")
    args = parser.parse_args()

    if args.beta != EXPECTED_BETA:
        raise ValueError(f"beta must remain fixed at {EXPECTED_BETA}, got {args.beta}")
    if not (0.0 < args.armijo_c1 < 1.0):
        raise ValueError("armijo-c1 must be in (0, 1)")
    if not (0.0 < args.backtrack_ratio < 1.0):
        raise ValueError("backtrack-ratio must be in (0, 1)")
    if not (0.0 < args.fd_alpha_2 < args.fd_alpha_1 < args.min_alpha <= 1.0):
        raise ValueError("require 0 < fd_alpha_2 < fd_alpha_1 < min_alpha <= 1")

    started = time.perf_counter()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_path = Path(__file__).resolve()
    source_snapshot = args.output_dir / "executed_source.py"
    shutil.copy2(source_path, source_snapshot)
    source_sha = _sha256(source_snapshot)
    checkpoint_hash = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_hash}")
    print(
        f"[Armijo] start protocol={PROTOCOL_ID} device={device} dtype=float32 proposals={args.proposals} "
        f"optimizer=Adam lr={args.lr:g} pairs={args.pairs} beta={args.beta:.12g} epsilon={args.epsilon:g} "
        f"clip={args.gradient_clip:g} alpha_grid=[1..{args.min_alpha:g}] c1={args.armijo_c1:g} "
        f"fd=({args.fd_alpha_1:g},{args.fd_alpha_2:g}) profile_all={args.profile_all} "
        f"output={args.output_dir} source_sha256={source_sha}",
        flush=True,
    )
    print(f"[Armijo] stage=load checkpoint={DEFAULT_RUN_DIR / 'vae_checkpoint.pt'}", flush=True)
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=args.pairs, batch_size=16384)
    bank = pd.read_csv(STATE_BANK)
    selected = bank.loc[bank["state_position"].eq(args.state_position)]
    if len(selected) != 1:
        raise ValueError(f"state_position {args.state_position} is not unique")
    source_index = int(selected.iloc[0]["source_weight_index"])
    weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
    if sha256_tensor(z) != EXPECTED_Z:
        raise RuntimeError(f"fixed z hash mismatch: {sha256_tensor(z)}")
    record = run.records.iloc[source_index].to_dict()
    record["source_weight_index"] = source_index

    active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
    named = dict(run.vae.named_parameters())
    missing = sorted(set(active_names) - set(named))
    if missing:
        raise RuntimeError(f"missing active parameters: {missing}")
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in active_names]
    optimizer = torch.optim.Adam(active, lr=args.lr)
    initial_active = {name: named[name].detach().cpu().clone() for name in active_names}
    checkpoint_proposals = {0, 10, 40, int(args.proposals)}

    print(
        f"[Armijo] loaded source_weight_index={source_index} task={record.get('task_name')} latent_dim={z.numel()} "
        f"z_sha256={sha256_tensor(z)} active_parameters={sum(p.numel() for p in active)}",
        flush=True,
    )
    print("[Armijo] stage=dense_initial_and_determinism", flush=True)
    current_row, current_h, current_m, current_eig, current_burg_gradient = _evaluate(
        run=run,
        z=z,
        record=record,
        epsilon=args.epsilon,
        beta=args.beta,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    repeat_row, repeat_h, repeat_m, repeat_eig, repeat_g = _evaluate(
        run=run,
        z=z,
        record=record,
        epsilon=args.epsilon,
        beta=args.beta,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    determinism_abs = abs(float(current_row["true_dense_f"]) - float(repeat_row["true_dense_f"]))
    if determinism_abs > args.determinism_tolerance:
        raise RuntimeError(f"dense F repeatability failed: abs delta {determinism_abs}")
    if abs(float(current_row["true_dense_f"]) - EXPECTED_INITIAL_F) > 1e-5:
        raise RuntimeError(
            f"initial objective mismatch: {current_row['true_dense_f']} versus expected {EXPECTED_INITIAL_F}"
        )
    del repeat_row, repeat_h, repeat_m, repeat_eig, repeat_g

    operator_rows = _operator_validation(
        run=run,
        cfg=cfg,
        z=z,
        record=record,
        hessian=current_h,
        step=0,
        probes=2,
    )
    state_rows: list[dict[str, float | int | bool]] = []
    spectrum_rows: list[dict[str, float | int]] = []
    update_rows: list[dict[str, float | int | bool | str]] = []
    candidate_rows: list[dict[str, float | int | bool | str]] = []
    matrix_checkpoints: dict[int, dict[str, torch.Tensor]] = {}

    def record_state(
        proposal: int,
        row: Mapping[str, float],
        hessian: torch.Tensor,
        metric: torch.Tensor,
        eig_m: torch.Tensor,
        *,
        accepted: bool,
        accepted_alpha: float,
    ) -> None:
        state_rows.append(
            {
                "proposal": proposal,
                "accepted": accepted,
                "accepted_alpha": accepted_alpha,
                **row,
                "beta_times_b": args.beta * float(row["damped_full_burg_per_dim"]),
                "elapsed_sec": time.perf_counter() - started,
            }
        )
        for rank, value in enumerate(eig_m.detach().cpu().double().tolist()):
            spectrum_rows.append({"proposal": proposal, "ascending_rank": rank, "m_eigenvalue": value})
        if proposal in checkpoint_proposals:
            matrix_checkpoints[proposal] = {
                "hessian": hessian.detach().cpu().float(),
                "metric_h_ht": metric.detach().cpu().float(),
                "metric_eigenvalues": eig_m.detach().cpu().double(),
            }
            torch.save(
                {
                    "proposal": proposal,
                    "active_model_state": {name: named[name].detach().cpu() for name in active_names},
                    "optimizer_state": optimizer.state_dict(),
                },
                args.output_dir / f"active_checkpoint_proposal{proposal}.pt",
            )

    record_state(0, current_row, current_h, current_m, current_eig, accepted=True, accepted_alpha=0.0)
    print(
        f"[Armijo] initial F={current_row['true_dense_f']:.9g} A={current_row['exact_a_per_dim']:.7g} "
        f"B={current_row['damped_full_burg_per_dim']:.7g} repeat_abs={determinism_abs:.3g}",
        flush=True,
    )
    alpha_grid: list[float] = []
    alpha = 1.0
    while alpha >= args.min_alpha * (1.0 - 1e-12):
        alpha_grid.append(alpha)
        alpha *= args.backtrack_ratio

    print("[Armijo] stage=optimization", flush=True)
    for proposal in range(1, args.proposals + 1):
        proposal_started = time.perf_counter()
        f_before = float(current_row["true_dense_f"])
        optimizer.zero_grad(set_to_none=True)
        gradient_row, clipped_gradient = _candidate_gradient(
            run=run,
            cfg=cfg,
            z=z,
            record=record,
            active=active,
            burg_gradient=current_burg_gradient,
            beta=args.beta,
            proposal=proposal,
            pairs=args.pairs,
            gradient_clip=args.gradient_clip,
        )
        base_parameters = _copy_parameters(active)
        pre_optimizer_state = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        displacement = [
            parameter.detach().clone().sub_(base)
            for parameter, base in zip(active, base_parameters, strict=True)
        ]
        candidate_step_norm = _vector_norm(displacement)
        stochastic_gradient_dot_step = _dot(clipped_gradient, displacement)
        _set_parameters(active, base_parameters)

        eval_order = 0

        def evaluate_alpha(alpha_value: float, role: str) -> tuple[
            dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            nonlocal eval_order
            eval_order += 1
            _set_parameters(active, base_parameters, displacement, alpha=alpha_value)
            result = _evaluate(
                run=run,
                z=z,
                record=record,
                epsilon=args.epsilon,
                beta=args.beta,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            _set_parameters(active, base_parameters)
            row = result[0]
            candidate_rows.append(
                {
                    "proposal": proposal,
                    "eval_order": eval_order,
                    "role": role,
                    "alpha": alpha_value,
                    "true_dense_f": float(row["true_dense_f"]),
                    "delta_f": float(row["true_dense_f"]) - f_before,
                    "exact_a_per_dim": float(row["exact_a_per_dim"]),
                    "damped_full_burg_per_dim": float(row["damped_full_burg_per_dim"]),
                    "task_loss": float(row["task_loss"]),
                }
            )
            return result

        fd_1_plus = evaluate_alpha(args.fd_alpha_1, "fd_plus_128")
        fd_1_minus = evaluate_alpha(-args.fd_alpha_1, "fd_minus_128")
        fd_2_plus = evaluate_alpha(args.fd_alpha_2, "fd_plus_256")
        fd_2_minus = evaluate_alpha(-args.fd_alpha_2, "fd_minus_256")
        slope_1 = (
            float(fd_1_plus[0]["true_dense_f"]) - float(fd_1_minus[0]["true_dense_f"])
        ) / (2.0 * args.fd_alpha_1)
        slope_2 = (
            float(fd_2_plus[0]["true_dense_f"]) - float(fd_2_minus[0]["true_dense_f"])
        ) / (2.0 * args.fd_alpha_2)
        slope_tolerance = max(args.strict_decrease_tolerance, args.determinism_tolerance) / args.fd_alpha_1
        slope_negative = slope_1 < -slope_tolerance and slope_2 < -slope_tolerance
        slope_sign_agrees = (slope_1 < -slope_tolerance and slope_2 < -slope_tolerance) or (
            slope_1 > slope_tolerance and slope_2 > slope_tolerance
        )
        slope_relative_difference = abs(slope_1 - slope_2) / max(abs(slope_1), abs(slope_2), 1e-30)
        for result in (fd_1_plus, fd_1_minus, fd_2_plus, fd_2_minus):
            del result

        accepted = False
        accepted_alpha = 0.0
        accepted_result: tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        full_f = math.nan
        line_reason = "non_descent_direction"
        for alpha_value in alpha_grid:
            result = evaluate_alpha(alpha_value, "backtrack")
            candidate_f = float(result[0]["true_dense_f"])
            if alpha_value == 1.0:
                full_f = candidate_f
            armijo_rhs = f_before + args.armijo_c1 * alpha_value * 0.5 * (slope_1 + slope_2)
            strict_decrease = candidate_f < f_before - args.strict_decrease_tolerance
            passes = slope_negative and strict_decrease and candidate_f <= armijo_rhs
            candidate_rows[-1]["armijo_rhs"] = armijo_rhs
            candidate_rows[-1]["strict_decrease"] = strict_decrease
            candidate_rows[-1]["passes_armijo"] = passes
            if passes and not accepted:
                accepted = True
                accepted_alpha = alpha_value
                accepted_result = result
                line_reason = "accepted"
                if not args.profile_all:
                    break
            else:
                del result

        if accepted_result is None:
            optimizer.load_state_dict(pre_optimizer_state)
            _set_parameters(active, base_parameters)
            accepted_delta_f = 0.0
            line_reason = "non_descent_direction" if not slope_negative else "no_alpha_passed"
            record_state(
                proposal,
                current_row,
                current_h,
                current_m,
                current_eig,
                accepted=False,
                accepted_alpha=0.0,
            )
        else:
            _set_parameters(active, base_parameters, displacement, alpha=accepted_alpha)
            next_row, current_h, current_m, next_eig, next_burg_gradient = accepted_result
            accepted_delta_f = float(next_row["true_dense_f"]) - f_before
            if accepted_delta_f >= -args.strict_decrease_tolerance:
                raise RuntimeError(
                    f"accepted non-decreasing proposal {proposal}: delta F={accepted_delta_f}"
                )
            current_row = next_row
            current_eig = next_eig
            current_burg_gradient = next_burg_gradient
            record_state(
                proposal,
                current_row,
                current_h,
                current_m,
                current_eig,
                accepted=True,
                accepted_alpha=accepted_alpha,
            )
            del accepted_result

        counterfactual_delta = full_f - f_before
        update_rows.append(
            {
                "proposal": proposal,
                "accepted": accepted,
                "accepted_alpha": accepted_alpha,
                "reason": line_reason,
                "f_before": f_before,
                "counterfactual_full_f": full_f,
                "counterfactual_full_delta_f": counterfactual_delta,
                "f_after": float(current_row["true_dense_f"]),
                "accepted_delta_f": accepted_delta_f,
                "fd_slope_128": slope_1,
                "fd_slope_256": slope_2,
                "fd_slope_sign_agrees": slope_sign_agrees,
                "fd_slope_relative_difference": slope_relative_difference,
                "fd_slope_negative": slope_negative,
                "candidate_step_norm": candidate_step_norm,
                "accepted_step_norm": accepted_alpha * candidate_step_norm,
                "clipped_gradient_dot_candidate_step": stochastic_gradient_dot_step,
                **gradient_row,
                "proposal_sec": time.perf_counter() - proposal_started,
                "elapsed_sec": time.perf_counter() - started,
            }
        )
        del base_parameters, displacement, clipped_gradient, pre_optimizer_state
        if proposal <= 3 or proposal % 5 == 0:
            print(
                f"[Armijo] proposal={proposal}/{args.proposals} F={current_row['true_dense_f']:.8g} "
                f"full_delta={counterfactual_delta:+.5g} slopes=({slope_1:+.5g},{slope_2:+.5g}) "
                f"accepted={accepted} alpha={accepted_alpha:.5g} delta={accepted_delta_f:+.5g} "
                f"clip={gradient_row['clip_factor']:.3g} sec={update_rows[-1]['proposal_sec']:.2f} "
                f"elapsed={update_rows[-1]['elapsed_sec']:.1f}s",
                flush=True,
            )

    print("[Armijo] stage=write_and_review_metrics", flush=True)
    states = pd.DataFrame(state_rows).sort_values("proposal").reset_index(drop=True)
    updates = pd.DataFrame(update_rows).sort_values("proposal").reset_index(drop=True)
    candidates = pd.DataFrame(candidate_rows).sort_values(["proposal", "eval_order"]).reset_index(drop=True)
    spectra = pd.DataFrame(spectrum_rows).sort_values(["proposal", "ascending_rank"]).reset_index(drop=True)
    operators = pd.DataFrame(operator_rows).sort_values(["step", "probe"]).reset_index(drop=True)
    states.to_csv(args.output_dir / "dense_state_trajectory.csv", index=False)
    updates.to_csv(args.output_dir / "proposal_diagnostics.csv", index=False)
    candidates.to_csv(args.output_dir / "line_search_candidates.csv", index=False)
    spectra.to_csv(args.output_dir / "spectrum_trajectory.csv", index=False)
    operators.to_csv(args.output_dir / "operator_validation.csv", index=False)
    torch.save(matrix_checkpoints, args.output_dir / "matrix_checkpoints.pt")

    accepted_mask = updates["accepted"].astype(bool)
    full_failures = updates["counterfactual_full_delta_f"] > args.strict_decrease_tolerance
    rescued = full_failures & accepted_mask & updates["fd_slope_negative"].astype(bool)
    used_path = float(updates["accepted_step_norm"].sum())
    proposed_path = float(updates["candidate_step_norm"].sum())
    final_f = float(states.iloc[-1]["true_dense_f"])
    accepted_alphas = updates.loc[accepted_mask, "accepted_alpha"]
    summary = {
        "protocol_id": PROTOCOL_ID,
        "source_snapshot_sha256": source_sha,
        "checkpoint_sha256": checkpoint_hash,
        "z_sha256": sha256_tensor(z),
        "device": str(device),
        "dtype": str(torch_dtype(run.cfg)),
        "source_weight_index": source_index,
        "proposals": args.proposals,
        "pairs": args.pairs,
        "optimizer": "Adam",
        "lr": args.lr,
        "gradient_clip": args.gradient_clip,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "armijo_c1": args.armijo_c1,
        "backtrack_ratio": args.backtrack_ratio,
        "min_alpha": args.min_alpha,
        "fd_alpha_1": args.fd_alpha_1,
        "fd_alpha_2": args.fd_alpha_2,
        "dense_f_repeat_abs": determinism_abs,
        "initial_f": float(states.iloc[0]["true_dense_f"]),
        "final_f": final_f,
        "relative_f_reduction": (float(states.iloc[0]["true_dense_f"]) - final_f)
        / float(states.iloc[0]["true_dense_f"]),
        "old_best_f": OLD_BEST_F,
        "accepted_count": int(accepted_mask.sum()),
        "rejected_count": int((~accepted_mask).sum()),
        "strict_decrease_count": int((updates["accepted_delta_f"] < -args.strict_decrease_tolerance).sum()),
        "median_accepted_alpha": float(accepted_alphas.median()) if not accepted_alphas.empty else 0.0,
        "minimum_accepted_alpha": float(accepted_alphas.min()) if not accepted_alphas.empty else 0.0,
        "path_utilization": used_path / max(proposed_path, 1e-30),
        "full_step_failure_count": int(full_failures.sum()),
        "rescued_full_step_failure_count": int(rescued.sum()),
        "rescued_full_step_failure_fraction": float(rescued.sum() / max(int(full_failures.sum()), 1)),
        "negative_fd_slope_count": int(updates["fd_slope_negative"].astype(bool).sum()),
        "fd_sign_agreement_fraction": float(updates["fd_slope_sign_agrees"].astype(bool).mean()),
        "all_accepted_states_monotone": bool(
            (states["true_dense_f"].diff().fillna(0.0) <= args.strict_decrease_tolerance).all()
        ),
        "gate_f_below_20pct": bool(final_f <= 0.8 * EXPECTED_INITIAL_F),
        "gate_beats_old_best": bool(final_f < OLD_BEST_F),
        "gate_accepted_80": bool(int(accepted_mask.sum()) >= math.ceil(0.8 * args.proposals)),
        "gate_median_alpha_1_over_8": bool(
            not accepted_alphas.empty and float(accepted_alphas.median()) >= 1.0 / 8.0
        ),
        "gate_path_utilization_0p1": bool(used_path / max(proposed_path, 1e-30) >= 0.1),
        "gate_rescue_fraction_0p8": bool(
            int(full_failures.sum()) >= max(1, math.ceil(0.1 * args.proposals))
            and float(rescued.sum() / max(int(full_failures.sum()), 1)) >= 0.8
        ),
        "elapsed_sec": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    config = {
        "resolved_args": vars(args) | {"output_dir": str(args.output_dir)},
        "protocol_id": PROTOCOL_ID,
        "source_snapshot_sha256": source_sha,
        "fixed_beta_source": str(SOURCE_RUN / "calibration.json"),
        "true_dense_f_definition": "exact_a_per_dim + beta * damped_full_burg_per_dim",
        "rejection_semantics": "restore both model parameters and pre-proposal Adam state",
        "acceptance_semantics": "commit post-gradient Adam moments once and scale only its parameter displacement",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    _plot(states, updates, candidates, args.output_dir / "true_dense_f_armijo.png")
    print(f"[Armijo] done summary={json.dumps(summary, sort_keys=True)}", flush=True)
    print(
        f"[Armijo] artifacts={args.output_dir / 'executed_source.py'},"
        f"{args.output_dir / 'config.json'},"
        f"{args.output_dir / 'dense_state_trajectory.csv'},"
        f"{args.output_dir / 'proposal_diagnostics.csv'},"
        f"{args.output_dir / 'line_search_candidates.csv'},"
        f"{args.output_dir / 'spectrum_trajectory.csv'},"
        f"{args.output_dir / 'operator_validation.csv'},"
        f"{args.output_dir / 'summary.json'},"
        f"{args.output_dir / 'true_dense_f_armijo.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
