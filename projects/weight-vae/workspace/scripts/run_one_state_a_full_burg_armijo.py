from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

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
    A_ONLY_PROTOCOL_ID,
    EXPECTED_CHECKPOINT,
    OUTPUT_ROOT,
    STATE_BANK,
    _gradient_norms,
    _materialize_metric,
    _pair_losses,
    _train_generators,
)


PROTOCOL_ID = "one_state_a_full_burg_armijo_iteration1_v1"
DEFAULT_OUTPUT = OUTPUT_ROOT / "armijo_iteration1_100proposals"
OLD_RUN = OUTPUT_ROOT / "production_100steps"
DEFAULT_BETA = 22.536727828943093
LINE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
SLOPE_STEPS = (1.0 / 128.0, 1.0 / 256.0)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _set_parameters(
    active: list[torch.nn.Parameter],
    base: list[torch.Tensor],
    displacement: list[torch.Tensor],
    alpha: float,
) -> None:
    with torch.no_grad():
        for parameter, origin, delta in zip(active, base, displacement, strict=True):
            parameter.copy_(origin + float(alpha) * delta)


def _adam_proposal(
    *,
    active: list[torch.nn.Parameter],
    exp_avg: list[torch.Tensor],
    exp_avg_sq: list[torch.Tensor],
    accepted_step: int,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    next_step = accepted_step + 1
    next_avg: list[torch.Tensor] = []
    next_avg_sq: list[torch.Tensor] = []
    displacement: list[torch.Tensor] = []
    correction1 = 1.0 - beta1**next_step
    correction2 = 1.0 - beta2**next_step
    with torch.no_grad():
        for parameter, old_avg, old_avg_sq in zip(active, exp_avg, exp_avg_sq, strict=True):
            if parameter.grad is None:
                grad = torch.zeros_like(parameter)
            else:
                grad = parameter.grad.detach()
            new_avg = beta1 * old_avg + (1.0 - beta1) * grad
            new_avg_sq = beta2 * old_avg_sq + (1.0 - beta2) * grad.square()
            direction = (new_avg / correction1) / ((new_avg_sq / correction2).sqrt() + float(epsilon))
            next_avg.append(new_avg)
            next_avg_sq.append(new_avg_sq)
            displacement.append(-float(lr) * direction)
    return next_avg, next_avg_sq, displacement


def _vector_dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    value = sum((a.detach().double() * b.detach().double()).sum() for a, b in zip(left, right, strict=True))
    return float(value.cpu())


def _vector_norm(values: list[torch.Tensor]) -> float:
    value = sum(tensor.detach().double().square().sum() for tensor in values)
    return float(value.sqrt().cpu())


def _objective(metric: dict[str, float], beta: float) -> float:
    return float(metric["exact_a_per_dim"] + float(beta) * metric["damped_full_burg_per_dim"])


def _evaluate_candidate(
    *,
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active: list[torch.nn.Parameter],
    base: list[torch.Tensor],
    displacement: list[torch.Tensor],
    alpha: float,
    beta: float,
    epsilon: float,
    hessian_chunk_size: int,
) -> dict[str, float]:
    _set_parameters(active, base, displacement, alpha)
    metric, hessian, matrix, eig_m, burg_gradient = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=epsilon,
        hessian_chunk_size=hessian_chunk_size,
    )
    del hessian, matrix, eig_m, burg_gradient
    return {**metric, "true_objective": _objective(metric, beta)}


def _plot(
    *,
    states: pd.DataFrame,
    proposals: pd.DataFrame,
    old_dense: pd.DataFrame,
    beta: float,
    output_dir: Path,
) -> None:
    old_f = old_dense["exact_a_per_dim"] + float(beta) * old_dense["damped_full_burg_per_dim"]
    initial = float(states.iloc[0]["true_objective"])
    target = 0.8 * initial
    old_best = float(old_f.min())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(states["proposal"], states["true_objective"], linewidth=2.0, label="Armijo state F")
    axes[0, 0].axhline(target, color="#16a34a", linestyle="--", label="20% reduction gate")
    axes[0, 0].axhline(old_best, color="#dc2626", linestyle=":", label="old best F")
    axes[0, 0].set(xlabel="gradient proposal", ylabel="true dense F = A + beta B", title="Accepted objective trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(proposals["proposal"], proposals["full_step_delta_f"], label="full Adam proposal dF")
    axes[0, 1].plot(proposals["proposal"], proposals["accepted_delta_f"], label="accepted dF")
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_yscale("symlog", linthresh=0.1)
    axes[0, 1].set(xlabel="gradient proposal", ylabel="objective change", title="Counterfactual full step vs accepted step")
    axes[0, 1].legend()

    accepted = proposals["accepted_alpha"].replace(0.0, np.nan)
    axes[1, 0].scatter(proposals["proposal"], accepted, s=18, label="accepted alpha")
    rejected = proposals.loc[proposals["accepted"].eq(0), "proposal"]
    axes[1, 0].scatter(rejected, np.full(len(rejected), 1.0 / 128.0), marker="x", color="#dc2626", label="rejected")
    axes[1, 0].set_yscale("log", base=2)
    axes[1, 0].set(xlabel="gradient proposal", ylabel="fraction of Adam displacement", title="Line-search acceptance")
    axes[1, 0].legend()

    axes[1, 1].plot(proposals["proposal"], proposals["slope_h128"], label="central slope h=1/128")
    axes[1, 1].plot(proposals["proposal"], proposals["slope_h256"], alpha=0.75, label="central slope h=1/256")
    axes[1, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 1].set_yscale("symlog", linthresh=1.0)
    axes[1, 1].set(xlabel="gradient proposal", ylabel="dF / d alpha", title="Local slope of proposed Adam path")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "true_objective_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-position", type=int, default=2)
    parser.add_argument("--proposals", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--burg-epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _source_sha256()
    shutil.copy2(Path(__file__), args.output_dir / "executed_source_snapshot.py")
    device = torch.device(args.device)
    checkpoint_hash = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_hash}")
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "dtype": "float32",
        "state_position": args.state_position,
        "proposals": args.proposals,
        "lr": args.lr,
        "pairs": args.pairs,
        "beta": args.beta,
        "burg_epsilon": args.burg_epsilon,
        "gradient_clip": args.gradient_clip,
        "armijo_c1": args.armijo_c1,
        "line_alphas": list(LINE_ALPHAS),
        "slope_steps": list(SLOPE_STEPS),
        "hessian_chunk_size": args.hessian_chunk_size,
        "output_dir": str(args.output_dir),
        "a_probe_seed_protocol": A_ONLY_PROTOCOL_ID,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "rejected_step_policy": "parameters and Adam moments unchanged",
    }
    (args.output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[armijo-i1] start {json.dumps(resolved, sort_keys=True)}", flush=True)

    print("[armijo-i1] stage=load", flush=True)
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
    record = run.records.iloc[source_index].to_dict()
    record["source_weight_index"] = source_index
    active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
    named = dict(run.vae.named_parameters())
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in active_names]
    exp_avg = [torch.zeros_like(parameter) for parameter in active]
    exp_avg_sq = [torch.zeros_like(parameter) for parameter in active]
    accepted_adam_step = 0
    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[armijo-i1] loaded source={source_index} task={record.get('task_name')} latent_dim={z.numel()} "
        f"z_sha256={sha256_tensor(z)} active_parameters={sum(p.numel() for p in active)}",
        flush=True,
    )

    state_rows: list[dict[str, float | int]] = []
    proposal_rows: list[dict[str, float | int]] = []
    line_rows: list[dict[str, float | int | str]] = []
    repeat_errors: list[float] = []
    cached_f: float | None = None
    initial_f: float | None = None

    for proposal in range(1, args.proposals + 1):
        proposal_started = time.perf_counter()
        print(f"[armijo-i1] stage=current_metric proposal={proposal}/{args.proposals}", flush=True)
        current_metric, current_h, current_matrix, current_eig, burg_gradient = _materialize_metric(
            run=run,
            z=z,
            record=record,
            epsilon=args.burg_epsilon,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        del current_h, current_matrix, current_eig
        current_f = _objective(current_metric, args.beta)
        if initial_f is None:
            initial_f = current_f
            state_rows.append({"proposal": 0, "accepted": 1, "true_objective": current_f, **current_metric})
        if cached_f is not None:
            repeat_errors.append(abs(current_f - cached_f))

        run.vae.zero_grad(set_to_none=True)
        a_losses: list[torch.Tensor] = []
        b_losses: list[torch.Tensor] = []
        for pair in range(args.pairs):
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
        component_stats = _gradient_norms(gradients_a, gradients_b, beta=args.beta)
        with torch.no_grad():
            for parameter, grad_a, grad_b in zip(active, gradients_a, gradients_b, strict=True):
                if grad_a is None and grad_b is None:
                    parameter.grad = None
                    continue
                if grad_a is None:
                    grad_a = torch.zeros_like(grad_b)
                if grad_b is None:
                    grad_b = torch.zeros_like(grad_a)
                parameter.grad = grad_a + float(args.beta) * grad_b
        preclip_norm = float(torch.nn.utils.clip_grad_norm_(active, max_norm=args.gradient_clip).detach().cpu())
        clip_factor = min(1.0, args.gradient_clip / max(preclip_norm, 1e-30))
        clipped_gradients = [
            torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in active
        ]
        next_avg, next_avg_sq, displacement = _adam_proposal(
            active=active,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            accepted_step=accepted_adam_step,
            lr=args.lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        proposal_norm = _vector_norm(displacement)
        gradient_dot_displacement = _vector_dot(clipped_gradients, displacement)
        base = [parameter.detach().clone() for parameter in active]

        candidate_by_alpha: dict[float, dict[str, float]] = {}
        for alpha in LINE_ALPHAS:
            candidate = _evaluate_candidate(
                run=run,
                z=z,
                record=record,
                active=active,
                base=base,
                displacement=displacement,
                alpha=alpha,
                beta=args.beta,
                epsilon=args.burg_epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            candidate_by_alpha[alpha] = candidate
            line_rows.append(
                {
                    "proposal": proposal,
                    "kind": "line",
                    "alpha": alpha,
                    "true_objective": candidate["true_objective"],
                    "delta_f": candidate["true_objective"] - current_f,
                    "exact_a_per_dim": candidate["exact_a_per_dim"],
                    "damped_full_burg_per_dim": candidate["damped_full_burg_per_dim"],
                    "task_loss": candidate["task_loss"],
                    "trace_m_per_dim": candidate["trace_m_per_dim"],
                    "m_max": candidate["m_max"],
                }
            )

        slopes: dict[float, float] = {}
        for slope_step in SLOPE_STEPS:
            positive = _evaluate_candidate(
                run=run,
                z=z,
                record=record,
                active=active,
                base=base,
                displacement=displacement,
                alpha=slope_step,
                beta=args.beta,
                epsilon=args.burg_epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            negative = _evaluate_candidate(
                run=run,
                z=z,
                record=record,
                active=active,
                base=base,
                displacement=displacement,
                alpha=-slope_step,
                beta=args.beta,
                epsilon=args.burg_epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            slopes[slope_step] = (positive["true_objective"] - negative["true_objective"]) / (2.0 * slope_step)
            for sign, candidate in (("positive", positive), ("negative", negative)):
                line_rows.append(
                    {
                        "proposal": proposal,
                        "kind": f"slope_{sign}",
                        "alpha": slope_step if sign == "positive" else -slope_step,
                        "true_objective": candidate["true_objective"],
                        "delta_f": candidate["true_objective"] - current_f,
                        "exact_a_per_dim": candidate["exact_a_per_dim"],
                        "damped_full_burg_per_dim": candidate["damped_full_burg_per_dim"],
                        "task_loss": candidate["task_loss"],
                        "trace_m_per_dim": candidate["trace_m_per_dim"],
                        "m_max": candidate["m_max"],
                    }
                )

        accepted_alpha = 0.0
        accepted_metric = current_metric
        if gradient_dot_displacement < 0.0:
            for alpha in LINE_ALPHAS:
                candidate = candidate_by_alpha[alpha]
                armijo_rhs = current_f + args.armijo_c1 * alpha * gradient_dot_displacement
                if candidate["true_objective"] <= armijo_rhs and candidate["true_objective"] < current_f - 1e-8:
                    accepted_alpha = alpha
                    accepted_metric = candidate
                    break

        accepted = int(accepted_alpha > 0.0)
        if accepted:
            _set_parameters(active, base, displacement, accepted_alpha)
            exp_avg = next_avg
            exp_avg_sq = next_avg_sq
            accepted_adam_step += 1
            after_f = float(accepted_metric["true_objective"])
        else:
            _set_parameters(active, base, displacement, 0.0)
            after_f = current_f
        cached_f = after_f
        strict_decrease = int(after_f < current_f - 1e-8)
        full_candidate = candidate_by_alpha[1.0]
        best_alpha = min(LINE_ALPHAS, key=lambda alpha: candidate_by_alpha[alpha]["true_objective"])
        slope_sign_agrees = int(math.copysign(1.0, slopes[SLOPE_STEPS[0]]) == math.copysign(1.0, slopes[SLOPE_STEPS[1]]))
        proposal_row: dict[str, float | int] = {
            "proposal": proposal,
            "accepted_adam_step_before": accepted_adam_step - accepted,
            "accepted_adam_step_after": accepted_adam_step,
            "current_f": current_f,
            "full_step_f": float(full_candidate["true_objective"]),
            "full_step_delta_f": float(full_candidate["true_objective"] - current_f),
            "best_line_alpha": best_alpha,
            "best_line_f": float(candidate_by_alpha[best_alpha]["true_objective"]),
            "accepted": accepted,
            "strict_decrease": strict_decrease,
            "accepted_alpha": accepted_alpha,
            "accepted_f": after_f,
            "accepted_delta_f": after_f - current_f,
            "slope_h128": slopes[1.0 / 128.0],
            "slope_h256": slopes[1.0 / 256.0],
            "slope_sign_agrees": slope_sign_agrees,
            "gradient_dot_displacement": gradient_dot_displacement,
            "proposal_norm": proposal_norm,
            "accepted_displacement_norm": accepted_alpha * proposal_norm,
            "train_a": float(mean_a.detach().cpu()),
            "train_b_pseudo_loss": float(mean_b.detach().cpu()),
            **component_stats,
            "grad_total_norm_clip_api": preclip_norm,
            "clip_factor": clip_factor,
            "proposal_sec": time.perf_counter() - proposal_started,
            "elapsed_sec": time.perf_counter() - started,
        }
        proposal_rows.append(proposal_row)
        state_rows.append(
            {
                "proposal": proposal,
                "accepted": accepted,
                "accepted_alpha": accepted_alpha,
                "true_objective": after_f,
                **{key: value for key, value in accepted_metric.items() if key != "true_objective"},
            }
        )
        print(
            f"[armijo-i1] proposal={proposal}/{args.proposals} F={current_f:.8g}->{after_f:.8g} "
            f"full_dF={proposal_row['full_step_delta_f']:+.4g} alpha={accepted_alpha:.5g} "
            f"slope={proposal_row['slope_h256']:+.4g} accepted={accepted} "
            f"sec={proposal_row['proposal_sec']:.1f} elapsed={proposal_row['elapsed_sec']:.1f}s",
            flush=True,
        )
        del (
            burg_gradient,
            a_losses,
            b_losses,
            mean_a,
            mean_b,
            gradients_a,
            gradients_b,
            clipped_gradients,
            next_avg,
            next_avg_sq,
            displacement,
            base,
        )

    states = pd.DataFrame(state_rows)
    proposals = pd.DataFrame(proposal_rows)
    lines = pd.DataFrame(line_rows)
    states.to_csv(args.output_dir / "state_objective_curve.csv", index=False)
    proposals.to_csv(args.output_dir / "proposal_diagnostics.csv", index=False)
    lines.to_csv(args.output_dir / "line_profiles.csv", index=False)
    torch.save(
        {
            "active_model_state": {name: named[name].detach().cpu() for name in active_names},
            "exp_avg": {name: value.detach().cpu() for name, value in zip(active_names, exp_avg, strict=True)},
            "exp_avg_sq": {name: value.detach().cpu() for name, value in zip(active_names, exp_avg_sq, strict=True)},
            "accepted_adam_step": accepted_adam_step,
        },
        args.output_dir / "final_checkpoint.pt",
    )

    assert initial_f is not None
    final_f = float(states.iloc[-1]["true_objective"])
    old_dense = pd.read_csv(OLD_RUN / "dense_trajectory.csv")
    old_f = old_dense["exact_a_per_dim"] + float(args.beta) * old_dense["damped_full_burg_per_dim"]
    accepted_frame = proposals.loc[proposals["accepted"].eq(1)]
    full_failures = proposals.loc[proposals["full_step_delta_f"].gt(0.0)]
    rescued_failures = full_failures.loc[
        full_failures["slope_h128"].lt(0.0)
        & full_failures["slope_h256"].lt(0.0)
        & full_failures["accepted_alpha"].ge(1.0 / 64.0)
    ]
    path_fraction = float(
        proposals["accepted_displacement_norm"].sum()
        / max(float(proposals["proposal_norm"].sum()), 1e-30)
    )
    gates = {
        "final_reduction_at_least_20pct": final_f <= 0.8 * initial_f,
        "beats_old_best": final_f < float(old_f.min()),
        "no_accepted_increases": int((proposals["accepted_delta_f"] > 1e-8).sum()) == 0,
        "at_least_80_strict_decreases": int(proposals["strict_decrease"].sum()) >= 80,
        "at_most_20_rejections": int((proposals["accepted"] == 0).sum()) <= 20,
        "median_alpha_at_least_one_eighth": bool(
            len(accepted_frame) > 0 and float(accepted_frame["accepted_alpha"].median()) >= 0.125
        ),
        "used_path_fraction_at_least_0p1": path_fraction >= 0.1,
        "failed_full_steps_rescued_at_least_80pct": bool(
            len(full_failures) > 0 and len(rescued_failures) / len(full_failures) >= 0.8
        ),
        "repeat_objective_error_at_most_1e_5": max(repeat_errors, default=0.0) <= 1e-5,
    }
    first_norm_reference = 0.076868966
    summary = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "checkpoint_sha256": checkpoint_hash,
        "state_position": args.state_position,
        "source_weight_index": source_index,
        "z_sha256": sha256_tensor(z),
        "proposals": args.proposals,
        "accepted_steps": int(proposals["accepted"].sum()),
        "rejected_steps": int((proposals["accepted"] == 0).sum()),
        "strict_decreases": int(proposals["strict_decrease"].sum()),
        "initial_true_objective": initial_f,
        "final_true_objective": final_f,
        "absolute_change": final_f - initial_f,
        "relative_change": final_f / initial_f - 1.0,
        "old_best_true_objective": float(old_f.min()),
        "old_best_step": int(old_f.idxmin()),
        "median_accepted_alpha": float(accepted_frame["accepted_alpha"].median()) if len(accepted_frame) else 0.0,
        "used_path_fraction": path_fraction,
        "full_step_failures": len(full_failures),
        "negative_slope_rescued_full_failures": len(rescued_failures),
        "negative_slope_rescue_fraction": len(rescued_failures) / len(full_failures) if len(full_failures) else None,
        "positive_local_slope_count_h128": int((proposals["slope_h128"] >= 0.0).sum()),
        "positive_local_slope_count_h256": int((proposals["slope_h256"] >= 0.0).sum()),
        "slope_sign_disagreements": int((proposals["slope_sign_agrees"] == 0).sum()),
        "max_repeat_objective_abs_error": max(repeat_errors, default=0.0),
        "first_proposal_norm": float(proposals.iloc[0]["proposal_norm"]),
        "first_proposal_norm_reference": first_norm_reference,
        "first_proposal_norm_relative_error": abs(float(proposals.iloc[0]["proposal_norm"]) - first_norm_reference)
        / first_norm_reference,
        "gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "elapsed_sec": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot(states=states, proposals=proposals, old_dense=old_dense, beta=args.beta, output_dir=args.output_dir)

    artifact_hashes: dict[str, str] = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name not in {"artifact_manifest.json", "run.log"}:
            artifact_hashes[path.name] = sha256_file(path)
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps({"source_sha256": source_hash, "artifacts": artifact_hashes}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[armijo-i1] done summary={json.dumps(summary, sort_keys=True)}", flush=True)
    print(
        f"[armijo-i1] artifacts={args.output_dir / 'state_objective_curve.csv'},"
        f"{args.output_dir / 'proposal_diagnostics.csv'},"
        f"{args.output_dir / 'line_profiles.csv'},"
        f"{args.output_dir / 'summary.json'},"
        f"{args.output_dir / 'true_objective_curve.png'},"
        f"{args.output_dir / 'artifact_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
