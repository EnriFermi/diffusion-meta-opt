from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import replace
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
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _task_set_for_record,
)
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _forward_equivalence,
    _named_tensor_hash,
    _numeric_frame_is_finite,
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
    LINE_ALPHAS,
    SLOPE_STEPS,
    _adam_proposal,
    _evaluate_candidate,
    _objective,
    _set_parameters,
    _vector_dot,
    _vector_norm,
)
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


PROTOCOL_ID = "one_state_a_full_burg_naive_semantics_iteration3_v1"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration3_naive_semantics_production"
ITERATION1_OUTPUT = OUTPUT_ROOT / "armijo_iteration1_100proposals"
OLD_UNCONTROLLED = OUTPUT_ROOT / "production_100steps/dense_trajectory.csv"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_3_frozen_dependency_manifest.json"
EXPECTED_FROZEN_MANIFEST_SHA256 = "185c5f26d4335297677283ab84492e56cad914893935261cb919511b2421ffc9"
EXPECTED_NORMALIZED_SOURCE_SHA256 = "7ad8b95113b7ed84defc53f7b1f72dc50f17455f747b491396cbed2d52a3ed02"
CONTROL_SOURCE_SHA256 = "03c5147dca57edcc754dcb7cd567e982a71b3fce27c5776471212de1dd0c9f36"
CONTROL_STATES_SHA256 = "2f71b677b130232b65d067e867f0a9c2e05be040d15d24bbae5e2b5c8b53b082"
CONTROL_PROPOSALS_SHA256 = "d3e03793da9ac86d3c7538e7c46886679950ca73968a1ff028cbf3bb434c650b"
CONTROL_LINES_SHA256 = "a5fbe07fda0ee3f2503a349ce9f5999d234ea4a2ac54ecfa0afdb39e977e2c3e"
FROZEN_TARGET_F = 139.69206097331205
FROZEN_OLD_BEST_F = 140.18711003198487
STRICT_DECREASE_TOLERANCE = 1e-8
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalized_source_sha256() -> str:
    masked_prefixes = (
        "EXPECTED_FROZEN_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    lines = Path(__file__).read_text(encoding="utf-8").splitlines(keepends=True)
    normalized: list[str] = []
    for line in lines:
        prefix = next((candidate for candidate in masked_prefixes if line.startswith(candidate)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _max_zero_run(values: list[int]) -> int:
    best = 0
    current = 0
    for value in values:
        if value == 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _build_proposal(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active: list[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    beta: float,
    proposal: int,
    pairs: int,
    exp_avg: list[torch.Tensor],
    exp_avg_sq: list[torch.Tensor],
    accepted_adam_step: int,
    gradient_clip: float,
    lr: float,
) -> dict[str, Any]:
    a_losses: list[torch.Tensor] = []
    b_losses: list[torch.Tensor] = []
    pair_scalars: list[dict[str, float | int]] = []
    for pair in range(pairs):
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
        pair_scalars.append(
            {
                "pair": pair,
                "a_loss": float(a_loss.detach().cpu()),
                "b_pseudo_loss": float(b_loss.detach().cpu()),
            }
        )
    mean_a = torch.stack(a_losses).mean()
    mean_b = torch.stack(b_losses).mean()
    gradients_a = torch.autograd.grad(mean_a, active, retain_graph=True, allow_unused=True)
    gradients_b = torch.autograd.grad(mean_b, active, retain_graph=False, allow_unused=True)
    component_stats = _gradient_norms(gradients_a, gradients_b, beta=beta)
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
    clip_factor = min(1.0, float(gradient_clip) / max(preclip_norm, 1e-30))
    clipped_gradients = [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in active
    ]
    next_avg, next_avg_sq, displacement = _adam_proposal(
        active=active,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        accepted_step=accepted_adam_step,
        lr=lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    result = {
        "mean_a": float(mean_a.detach().cpu()),
        "mean_b": float(mean_b.detach().cpu()),
        "pair_scalars": pair_scalars,
        "component_stats": component_stats,
        "preclip_norm": preclip_norm,
        "clip_factor": clip_factor,
        "next_avg": next_avg,
        "next_avg_sq": next_avg_sq,
        "displacement": displacement,
        "proposal_norm": _vector_norm(displacement),
        "gradient_dot_displacement": _vector_dot(clipped_gradients, displacement),
    }
    for parameter in active:
        parameter.grad = None
    del a_losses, b_losses, mean_a, mean_b, gradients_a, gradients_b, clipped_gradients
    return result


def _evaluate_path(
    *,
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active: list[torch.nn.Parameter],
    base: list[torch.Tensor],
    displacement: list[torch.Tensor],
    current_f: float,
    beta: float,
    epsilon: float,
    hessian_chunk_size: int,
    proposal: int,
) -> tuple[dict[float, float], dict[float, dict[str, float]], list[dict[str, float | int | str]]]:
    rows: list[dict[str, float | int | str]] = []
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
            beta=beta,
            epsilon=epsilon,
            hessian_chunk_size=hessian_chunk_size,
        )
        negative = _evaluate_candidate(
            run=run,
            z=z,
            record=record,
            active=active,
            base=base,
            displacement=displacement,
            alpha=-slope_step,
            beta=beta,
            epsilon=epsilon,
            hessian_chunk_size=hessian_chunk_size,
        )
        slopes[slope_step] = (float(positive["true_objective"]) - float(negative["true_objective"])) / (
            2.0 * slope_step
        )
        for sign, candidate in (("positive", positive), ("negative", negative)):
            rows.append(
                {
                    "proposal": proposal,
                    "kind": f"fd_{sign}",
                    "alpha": slope_step if sign == "positive" else -slope_step,
                    "true_objective": float(candidate["true_objective"]),
                    "delta_f": float(candidate["true_objective"]) - current_f,
                }
            )
    candidates: dict[float, dict[str, float]] = {}
    for alpha in LINE_ALPHAS:
        candidate = _evaluate_candidate(
            run=run,
            z=z,
            record=record,
            active=active,
            base=base,
            displacement=displacement,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            hessian_chunk_size=hessian_chunk_size,
        )
        candidates[alpha] = candidate
        rows.append(
            {
                "proposal": proposal,
                "kind": "line",
                "alpha": alpha,
                "true_objective": float(candidate["true_objective"]),
                "delta_f": float(candidate["true_objective"]) - current_f,
            }
        )
    return slopes, candidates, rows


def _fd_armijo_alpha(
    *,
    current_f: float,
    slopes: dict[float, float],
    candidates: dict[float, dict[str, float]],
    c1: float,
    strict_tolerance: float,
) -> float:
    if not all(slopes[step] < 0.0 for step in SLOPE_STEPS):
        return 0.0
    slope = slopes[1.0 / 256.0]
    for alpha in LINE_ALPHAS:
        candidate_f = float(candidates[alpha]["true_objective"])
        if candidate_f <= current_f + float(c1) * alpha * slope and candidate_f < current_f - strict_tolerance:
            return alpha
    return 0.0


def _control_acceptance_equivalence(*, c1: float) -> pd.DataFrame:
    proposals = pd.read_csv(ITERATION1_OUTPUT / "proposal_diagnostics.csv")
    lines = pd.read_csv(ITERATION1_OUTPUT / "line_profiles.csv")
    rows: list[dict[str, float | int]] = []
    for proposal in range(1, 101):
        diagnostic = proposals.loc[proposals["proposal"].eq(proposal)]
        if len(diagnostic) != 1:
            raise RuntimeError(f"canonical control proposal {proposal} is not unique")
        diagnostic = diagnostic.iloc[0]
        candidates: dict[float, dict[str, float]] = {}
        proposal_lines = lines.loc[(lines["proposal"].eq(proposal)) & (lines["kind"].eq("line"))]
        for alpha in LINE_ALPHAS:
            candidate = proposal_lines.loc[np.isclose(proposal_lines["alpha"], alpha)]
            if len(candidate) != 1:
                raise RuntimeError(f"missing canonical line proposal={proposal} alpha={alpha}")
            candidates[alpha] = {"true_objective": float(candidate.iloc[0]["true_objective"])}
        slopes = {
            1.0 / 128.0: float(diagnostic["slope_h128"]),
            1.0 / 256.0: float(diagnostic["slope_h256"]),
        }
        recalculated = _fd_armijo_alpha(
            current_f=float(diagnostic["current_f"]),
            slopes=slopes,
            candidates=candidates,
            c1=c1,
            strict_tolerance=STRICT_DECREASE_TOLERANCE,
        )
        stored = float(diagnostic["accepted_alpha"])
        rows.append(
            {
                "proposal": proposal,
                "stored_accepted_alpha": stored,
                "fd_armijo_accepted_alpha": recalculated,
                "alpha_absolute_error": abs(stored - recalculated),
                "match": int(abs(stored - recalculated) <= 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _plot(
    *,
    states: pd.DataFrame,
    proposals: pd.DataFrame,
    control_states: pd.DataFrame,
    beta: float,
    old_best: float,
    output: Path,
) -> None:
    initial = float(states.iloc[0]["true_objective"])
    target = 0.8 * initial
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(states["proposal"], states["true_objective"], linewidth=2.2, label="naive semantics")
    axes[0, 0].plot(
        control_states["proposal"],
        control_states["true_objective"],
        linewidth=1.5,
        alpha=0.75,
        label="stopped control",
    )
    axes[0, 0].axhline(target, color="#16a34a", linestyle="--", label="20% gate")
    axes[0, 0].axhline(old_best, color="#dc2626", linestyle=":", label="old best")
    axes[0, 0].set(xlabel="gradient proposal", ylabel="dense F = A + beta B", title="True objective trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(proposals["proposal"], proposals["slope_h128"], label="FD h=1/128")
    axes[0, 1].plot(proposals["proposal"], proposals["slope_h256"], alpha=0.8, label="FD h=1/256")
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_yscale("symlog", linthresh=0.1)
    axes[0, 1].set(xlabel="gradient proposal", ylabel="dF / d alpha", title="True-F path derivative")
    axes[0, 1].legend()

    accepted_alpha = proposals["accepted_alpha"].replace(0.0, np.nan)
    axes[1, 0].scatter(proposals["proposal"], accepted_alpha, s=20, label="accepted alpha")
    rejected = proposals.loc[proposals["accepted"].eq(0), "proposal"]
    if len(rejected):
        axes[1, 0].scatter(
            rejected,
            np.full(len(rejected), 1.0 / 128.0),
            marker="x",
            color="#dc2626",
            label="rejected",
        )
    axes[1, 0].set_yscale("log", base=2)
    axes[1, 0].set(xlabel="gradient proposal", ylabel="fraction of Adam proposal", title="Line-search acceptance")
    axes[1, 0].legend()

    axes[1, 1].plot(states["proposal"], states["exact_a_per_dim"], label="exact A")
    axes[1, 1].plot(
        states["proposal"],
        float(beta) * states["damped_full_burg_per_dim"],
        label="beta * full B",
    )
    axes[1, 1].plot(states["proposal"], states["m_max"], alpha=0.75, label="max eig(M)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(xlabel="gradient proposal", ylabel="value", title="Objective components and top spectrum")
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
    parser.add_argument("--proposals", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--burg-epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument("--hvp-mode", choices=("autograd", "stopped_composite"), default="autograd")
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-preflight",
        action="store_true",
        help="Allow a non-frozen smoke run; its validity and mechanism decisions remain null.",
    )
    args = parser.parse_args()

    frozen_setup = bool(
        args.state_position == 2
        and args.proposals == 100
        and args.pairs == 4
        and math.isclose(args.beta, DEFAULT_BETA, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.burg_epsilon, 1e-4, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.gradient_clip, 1.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.lr, 3e-5, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.armijo_c1, 1e-4, rel_tol=0.0, abs_tol=0.0)
        and args.hessian_chunk_size == 64
        and args.hvp_mode == "autograd"
    )
    if not frozen_setup and not args.allow_preflight:
        raise RuntimeError("iteration-3 evidence run requires the frozen setup exactly")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _source_sha256()
    normalized_source_hash = _normalized_source_sha256()
    frozen_manifest_hash = sha256_file(FROZEN_DEPENDENCY_MANIFEST)
    frozen_manifest = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]
    frozen_dependency_hashes = {
        relative_path: sha256_file(project_root / relative_path)
        for relative_path in frozen_manifest["dependencies"]
    }
    frozen_dependency_matches = {
        relative_path: frozen_dependency_hashes[relative_path] == expected_hash
        for relative_path, expected_hash in frozen_manifest["dependencies"].items()
    }
    hard_freeze_valid = bool(
        frozen_manifest_hash == EXPECTED_FROZEN_MANIFEST_SHA256
        and normalized_source_hash == EXPECTED_NORMALIZED_SOURCE_SHA256
        and normalized_source_hash == frozen_manifest["runner_normalized_sha256"]
        and frozen_manifest["protocol_id"] == PROTOCOL_ID
        and all(frozen_dependency_matches.values())
    )
    if not hard_freeze_valid:
        raise RuntimeError(
            "iteration-3 frozen dependency manifest mismatch: "
            + json.dumps(
                {
                    "manifest_hash": frozen_manifest_hash,
                    "expected_manifest_hash": EXPECTED_FROZEN_MANIFEST_SHA256,
                    "normalized_source_hash": normalized_source_hash,
                    "expected_normalized_source_hash": EXPECTED_NORMALIZED_SOURCE_SHA256,
                    "dependency_matches": frozen_dependency_matches,
                },
                sort_keys=True,
            )
        )
    shutil.copy2(Path(__file__), args.output_dir / "executed_source_snapshot.py")
    device = torch.device(args.device)
    checkpoint_hash = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_hash}")
    dependency_hashes = {
        "iteration1_states": sha256_file(ITERATION1_OUTPUT / "state_objective_curve.csv"),
        "iteration1_proposals": sha256_file(ITERATION1_OUTPUT / "proposal_diagnostics.csv"),
        "iteration1_lines": sha256_file(ITERATION1_OUTPUT / "line_profiles.csv"),
        "iteration2_source": sha256_file(Path(__file__).with_name("audit_one_state_a_full_burg_direction_mismatch.py")),
        "armijo_source": sha256_file(Path(__file__).with_name("run_one_state_a_full_burg_armijo.py")),
        "objective_source": sha256_file(Path(__file__).with_name("smoke_optimize_variant_a_full_burg_one_state.py")),
        "control_executed_source": sha256_file(ITERATION1_OUTPUT / "executed_source_snapshot.py"),
        "control_resolved_config": sha256_file(ITERATION1_OUTPUT / "resolved_config.json"),
        "control_summary": sha256_file(ITERATION1_OUTPUT / "summary.json"),
    }
    control_equivalence = _control_acceptance_equivalence(c1=args.armijo_c1)
    control_equivalence.to_csv(args.output_dir / "control_acceptance_equivalence.csv", index=False)
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "normalized_source_sha256": normalized_source_hash,
        "frozen_dependency_manifest": str(FROZEN_DEPENDENCY_MANIFEST),
        "frozen_dependency_manifest_sha256": frozen_manifest_hash,
        "frozen_dependency_sha256": frozen_dependency_hashes,
        "frozen_dependency_matches": frozen_dependency_matches,
        "hard_freeze_valid": hard_freeze_valid,
        "checkpoint_sha256": checkpoint_hash,
        "dependency_sha256": dependency_hashes,
        "device": str(device),
        "dtype": "float32",
        "state_position": args.state_position,
        "proposals": args.proposals,
        "pairs": args.pairs,
        "beta": args.beta,
        "burg_epsilon": args.burg_epsilon,
        "gradient_clip": args.gradient_clip,
        "lr": args.lr,
        "armijo_c1": args.armijo_c1,
        "strict_decrease_tolerance": STRICT_DECREASE_TOLERANCE,
        "hessian_chunk_size": args.hessian_chunk_size,
        "line_alphas": list(LINE_ALPHAS),
        "slope_steps": list(SLOPE_STEPS),
        "hvp_mode_control": "stopped_composite",
        "hvp_mode_intervention": "autograd",
        "hvp_mode_executed": args.hvp_mode,
        "rejected_step_policy": "parameters and Adam moments unchanged",
        "a_probe_seed_protocol": A_ONLY_PROTOCOL_ID,
        "frozen_setup": frozen_setup,
        "run_mode": "production_evidence" if frozen_setup else "non_evidence_preflight",
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[naive-i3] start {json.dumps(resolved, sort_keys=True)}", flush=True)

    print("[naive-i3] stage=load", flush=True)
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    cfg_stopped = _probe_cfg(run.cfg, sample_count=1, pair_count=args.pairs, batch_size=16384)
    cfg_naive = replace(cfg_stopped, vae_precond_hvp_mode="autograd")
    cfg_treatment = cfg_naive if args.hvp_mode == "autograd" else cfg_stopped
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
    full_ce_batch_none = all(
        _batch_indices(
            task_set,
            batch_size=int(cfg_treatment.vae_precond_batch_size),
            step=10,
            sample_key=source_index,
            pair_key=pair_key,
        )
        is None
        for pair_key in range(2 * args.pairs)
    )
    active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
    named = dict(run.vae.named_parameters())
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in active_names]
    initial_parameter_hash = _named_tensor_hash(active_names, active)
    z_hash = sha256_tensor(z)
    exp_avg = [torch.zeros_like(parameter) for parameter in active]
    exp_avg_sq = [torch.zeros_like(parameter) for parameter in active]
    accepted_adam_step = 0
    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[naive-i3] loaded source={source_index} task={record.get('task_name')} z_dim={z.numel()} "
        f"z_sha256={z_hash} active_parameters={sum(p.numel() for p in active)} full_ce={full_ce_batch_none}",
        flush=True,
    )

    state_rows: list[dict[str, float | int]] = []
    proposal_rows: list[dict[str, float | int]] = []
    line_rows: list[dict[str, float | int | str]] = []
    repeat_errors: list[float] = []
    forward_rows: list[dict[str, float | int]] = []
    forward_scalar_rows: list[dict[str, float | int]] = []
    stopped_preflight_rows: list[dict[str, float | int | str]] = []
    rejected_state_hash_checks: list[int] = []
    cached_f: float | None = None
    initial_f: float | None = None
    consecutive_rejections = 0
    maximum_rejection_run = 0

    for proposal in range(1, args.proposals + 1):
        proposal_started = time.perf_counter()
        print(f"[naive-i3] stage=current-metric proposal={proposal}/{args.proposals}", flush=True)
        current_metric, hessian, _matrix, _eig, burg_gradient = _materialize_metric(
            run=run,
            z=z,
            record=record,
            epsilon=args.burg_epsilon,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        current_f = _objective(current_metric, args.beta)
        if proposal == 1:
            forward_rows = _forward_equivalence(
                cfg_stopped=cfg_stopped,
                cfg_naive=cfg_naive,
                run=run,
                z=z,
                record=record,
                hessian=hessian,
                probes=4,
            )
        del hessian, _matrix, _eig
        if initial_f is None:
            initial_f = current_f
            state_rows.append({"proposal": 0, "accepted": 1, "accepted_alpha": 0.0, "true_objective": current_f, **current_metric})
        if cached_f is not None:
            repeat_errors.append(abs(current_f - cached_f))

        if proposal == 1:
            stopped_control = _build_proposal(
                cfg=cfg_stopped,
                run=run,
                z=z,
                record=record,
                active=active,
                burg_gradient=burg_gradient,
                beta=args.beta,
                proposal=proposal,
                pairs=args.pairs,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                accepted_adam_step=accepted_adam_step,
                gradient_clip=args.gradient_clip,
                lr=args.lr,
            )
            control_base = [parameter.detach().clone() for parameter in active]
            control_slopes, control_candidates, control_path_rows = _evaluate_path(
                run=run,
                z=z,
                record=record,
                active=active,
                base=control_base,
                displacement=stopped_control["displacement"],
                current_f=current_f,
                beta=args.beta,
                epsilon=args.burg_epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
                proposal=proposal,
            )
            _set_parameters(active, control_base, stopped_control["displacement"], 0.0)
            canonical_proposal = pd.read_csv(ITERATION1_OUTPUT / "proposal_diagnostics.csv").iloc[0]
            stopped_preflight_rows.append(
                {
                    "kind": "proposal",
                    "alpha": 0.0,
                    "observed": float(stopped_control["proposal_norm"]),
                    "expected": float(canonical_proposal["proposal_norm"]),
                    "absolute_error": abs(float(stopped_control["proposal_norm"]) - float(canonical_proposal["proposal_norm"])),
                }
            )
            for name, step in (("slope_h128", 1.0 / 128.0), ("slope_h256", 1.0 / 256.0)):
                stopped_preflight_rows.append(
                    {
                        "kind": name,
                        "alpha": step,
                        "observed": float(control_slopes[step]),
                        "expected": float(canonical_proposal[name]),
                        "absolute_error": abs(float(control_slopes[step]) - float(canonical_proposal[name])),
                    }
                )
            canonical_lines = pd.read_csv(ITERATION1_OUTPUT / "line_profiles.csv")
            canonical_lines = canonical_lines.loc[
                canonical_lines["proposal"].eq(1) & canonical_lines["kind"].eq("line")
            ]
            for alpha in LINE_ALPHAS:
                expected = float(canonical_lines.loc[np.isclose(canonical_lines["alpha"], alpha)].iloc[0]["true_objective"])
                observed = float(control_candidates[alpha]["true_objective"])
                stopped_preflight_rows.append(
                    {
                        "kind": "line",
                        "alpha": alpha,
                        "observed": observed,
                        "expected": expected,
                        "absolute_error": abs(observed - expected),
                    }
                )
            del control_base, control_slopes, control_candidates, control_path_rows

        treatment = _build_proposal(
            cfg=cfg_treatment,
            run=run,
            z=z,
            record=record,
            active=active,
            burg_gradient=burg_gradient,
            beta=args.beta,
            proposal=proposal,
            pairs=args.pairs,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            accepted_adam_step=accepted_adam_step,
            gradient_clip=args.gradient_clip,
            lr=args.lr,
        )
        if proposal == 1:
            for stopped_pair, treatment_pair in zip(
                stopped_control["pair_scalars"], treatment["pair_scalars"], strict=True
            ):
                forward_scalar_rows.append(
                    {
                        "pair": int(stopped_pair["pair"]),
                        "stopped_a": float(stopped_pair["a_loss"]),
                        "treatment_a": float(treatment_pair["a_loss"]),
                        "a_absolute_error": abs(float(stopped_pair["a_loss"]) - float(treatment_pair["a_loss"])),
                        "a_relative_error": abs(float(stopped_pair["a_loss"]) - float(treatment_pair["a_loss"]))
                        / max(abs(float(stopped_pair["a_loss"])), abs(float(treatment_pair["a_loss"])), 1e-30),
                        "stopped_b": float(stopped_pair["b_pseudo_loss"]),
                        "treatment_b": float(treatment_pair["b_pseudo_loss"]),
                        "b_absolute_error": abs(
                            float(stopped_pair["b_pseudo_loss"]) - float(treatment_pair["b_pseudo_loss"])
                        ),
                        "b_relative_error": abs(
                            float(stopped_pair["b_pseudo_loss"]) - float(treatment_pair["b_pseudo_loss"])
                        )
                        / max(
                            abs(float(stopped_pair["b_pseudo_loss"])),
                            abs(float(treatment_pair["b_pseudo_loss"])),
                            1e-30,
                        ),
                    }
                )
            del stopped_control
        next_avg = treatment["next_avg"]
        next_avg_sq = treatment["next_avg_sq"]
        displacement = treatment["displacement"]
        proposal_norm = float(treatment["proposal_norm"])
        stochastic_gradient_dot_displacement = float(treatment["gradient_dot_displacement"])
        base = [parameter.detach().clone() for parameter in active]
        slopes, candidates, path_rows = _evaluate_path(
            run=run,
            z=z,
            record=record,
            active=active,
            base=base,
            displacement=displacement,
            current_f=current_f,
            beta=args.beta,
            epsilon=args.burg_epsilon,
            hessian_chunk_size=args.hessian_chunk_size,
            proposal=proposal,
        )
        line_rows.extend(path_rows)

        negative_both = bool(all(slopes[step] < 0.0 for step in SLOPE_STEPS))
        accepted_alpha = _fd_armijo_alpha(
            current_f=current_f,
            slopes=slopes,
            candidates=candidates,
            c1=args.armijo_c1,
            strict_tolerance=STRICT_DECREASE_TOLERANCE,
        )
        accepted_metric: dict[str, float] = {**current_metric, "true_objective": current_f}
        if accepted_alpha > 0.0:
            accepted_metric = candidates[accepted_alpha]

        accepted = int(accepted_alpha > 0.0)
        rejected_parameter_hash_before = ""
        rejected_moment_hash_before = ""
        if not accepted:
            rejected_parameter_hash_before = _named_tensor_hash(active_names, base)
            rejected_moment_hash_before = hashlib.sha256(
                (
                    _named_tensor_hash(active_names, exp_avg)
                    + _named_tensor_hash(active_names, exp_avg_sq)
                    + str(accepted_adam_step)
                ).encode("utf-8")
            ).hexdigest()
        if accepted:
            _set_parameters(active, base, displacement, accepted_alpha)
            exp_avg = next_avg
            exp_avg_sq = next_avg_sq
            accepted_adam_step += 1
            after_f = float(accepted_metric["true_objective"])
            consecutive_rejections = 0
        else:
            _set_parameters(active, base, displacement, 0.0)
            after_f = current_f
            consecutive_rejections += 1
            maximum_rejection_run = max(maximum_rejection_run, consecutive_rejections)
            rejected_parameter_hash_after = _named_tensor_hash(active_names, active)
            rejected_moment_hash_after = hashlib.sha256(
                (
                    _named_tensor_hash(active_names, exp_avg)
                    + _named_tensor_hash(active_names, exp_avg_sq)
                    + str(accepted_adam_step)
                ).encode("utf-8")
            ).hexdigest()
            rejected_state_hash_checks.append(
                int(
                    rejected_parameter_hash_before == rejected_parameter_hash_after
                    and rejected_moment_hash_before == rejected_moment_hash_after
                )
            )
        cached_f = after_f
        strict_decrease = int(after_f < current_f - STRICT_DECREASE_TOLERANCE)
        state_rows.append(
            {
                "proposal": proposal,
                "accepted": accepted,
                "accepted_alpha": accepted_alpha,
                "true_objective": after_f,
                **{key: value for key, value in accepted_metric.items() if key != "true_objective"},
            }
        )
        proposal_row: dict[str, float | int] = {
            "proposal": proposal,
            "accepted_adam_step_before": accepted_adam_step - accepted,
            "accepted_adam_step_after": accepted_adam_step,
            "current_f": current_f,
            "accepted": accepted,
            "strict_decrease": strict_decrease,
            "accepted_alpha": accepted_alpha,
            "accepted_f": after_f,
            "accepted_delta_f": after_f - current_f,
            "slope_h128": slopes[1.0 / 128.0],
            "slope_h256": slopes[1.0 / 256.0],
            "negative_slope_both": int(negative_both),
            "slope_sign_agrees": int(math.copysign(1.0, slopes[1.0 / 128.0]) == math.copysign(1.0, slopes[1.0 / 256.0])),
            "slope_relative_disagreement": abs(slopes[1.0 / 128.0] - slopes[1.0 / 256.0])
            / max(abs(slopes[1.0 / 128.0]), abs(slopes[1.0 / 256.0]), 1e-30),
            "stochastic_gradient_dot_displacement": stochastic_gradient_dot_displacement,
            "proposal_norm": proposal_norm,
            "accepted_displacement_norm": accepted_alpha * proposal_norm,
            "train_a": float(treatment["mean_a"]),
            "train_b_pseudo_loss": float(treatment["mean_b"]),
            **treatment["component_stats"],
            "grad_total_norm_clip_api": float(treatment["preclip_norm"]),
            "clip_factor": float(treatment["clip_factor"]),
            "consecutive_rejections_after": consecutive_rejections,
            "proposal_sec": time.perf_counter() - proposal_started,
            "elapsed_sec": time.perf_counter() - started,
        }
        proposal_rows.append(proposal_row)
        print(
            f"[naive-i3] proposal={proposal}/{args.proposals} F={current_f:.8g}->{after_f:.8g} "
            f"slope={slopes[1.0 / 256.0]:+.5g} alpha={accepted_alpha:g} accepted={accepted} "
            f"sec={proposal_row['proposal_sec']:.1f} elapsed={proposal_row['elapsed_sec']:.1f}s",
            flush=True,
        )
        del (
            burg_gradient,
            next_avg,
            next_avg_sq,
            displacement,
            base,
            candidates,
            path_rows,
            treatment,
        )

    final_metric_check, _final_h, _final_m, _final_eig, _final_burg = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=args.burg_epsilon,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    final_objective_recomputed = _objective(final_metric_check, args.beta)
    del _final_h, _final_m, _final_eig, _final_burg

    states = pd.DataFrame(state_rows)
    proposals = pd.DataFrame(proposal_rows)
    lines = pd.DataFrame(line_rows)
    forward = pd.DataFrame(forward_rows)
    forward_scalars = pd.DataFrame(forward_scalar_rows)
    stopped_preflight = pd.DataFrame(stopped_preflight_rows)
    states.to_csv(args.output_dir / "state_objective_curve.csv", index=False)
    proposals.to_csv(args.output_dir / "proposal_diagnostics.csv", index=False)
    lines.to_csv(args.output_dir / "line_profiles.csv", index=False)
    forward.to_csv(args.output_dir / "forward_equivalence.csv", index=False)
    forward_scalars.to_csv(args.output_dir / "proposal1_forward_scalars.csv", index=False)
    stopped_preflight.to_csv(args.output_dir / "stopped_proposal1_reproduction.csv", index=False)
    checkpoint_payload = {
        "active_model_state": {name: named[name].detach().cpu() for name in active_names},
        "exp_avg": {name: value.detach().cpu() for name, value in zip(active_names, exp_avg, strict=True)},
        "exp_avg_sq": {name: value.detach().cpu() for name, value in zip(active_names, exp_avg_sq, strict=True)},
        "accepted_adam_step": accepted_adam_step,
    }
    checkpoint_path = args.output_dir / "final_checkpoint.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_reload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model_reload_matches = _named_tensor_hash(
        active_names, [checkpoint_reload["active_model_state"][name] for name in active_names]
    ) == _named_tensor_hash(active_names, active)
    checkpoint_moments_reload_match = bool(
        _named_tensor_hash(active_names, [checkpoint_reload["exp_avg"][name] for name in active_names])
        == _named_tensor_hash(active_names, exp_avg)
        and _named_tensor_hash(active_names, [checkpoint_reload["exp_avg_sq"][name] for name in active_names])
        == _named_tensor_hash(active_names, exp_avg_sq)
        and int(checkpoint_reload["accepted_adam_step"]) == accepted_adam_step
    )

    assert initial_f is not None
    control_states = pd.read_csv(ITERATION1_OUTPUT / "state_objective_curve.csv")
    old_uncontrolled = pd.read_csv(OLD_UNCONTROLLED)
    old_f = old_uncontrolled["exact_a_per_dim"] + float(args.beta) * old_uncontrolled["damped_full_burg_per_dim"]
    old_best = float(old_f.min())
    final_f = float(states.iloc[-1]["true_objective"])
    strict_count = int(proposals["strict_decrease"].sum())
    rejected_count = int((proposals["accepted"] == 0).sum())
    negative_count = int(proposals["negative_slope_both"].sum())
    slope_sign_agreement_count = int(proposals["slope_sign_agrees"].sum())
    median_slope_disagreement = float(proposals["slope_relative_disagreement"].median())
    p95_slope_disagreement = float(proposals["slope_relative_disagreement"].quantile(0.95))
    accepted_rows = proposals.loc[proposals["accepted"].eq(1)]
    median_alpha = float(accepted_rows["accepted_alpha"].median()) if len(accepted_rows) else 0.0
    path_fraction = float(
        proposals["accepted_displacement_norm"].sum()
        / max(float(proposals["proposal_norm"].sum()), 1e-30)
    )
    forward_max_error = float(
        forward[[
            "stopped_to_dense_relative_error",
            "naive_to_dense_relative_error",
            "stopped_to_naive_relative_error",
        ]].to_numpy().max()
    )
    numeric_finite = all(
        _numeric_frame_is_finite(frame)
        for frame in (
            states,
            proposals,
            lines,
            forward,
            forward_scalars,
            stopped_preflight,
            control_equivalence,
        )
    )
    control_initial = control_states.iloc[0]
    control_source_matches = dependency_hashes["control_executed_source"] == CONTROL_SOURCE_SHA256
    control_equivalence_matches = bool(
        len(control_equivalence) == 100
        and int(control_equivalence["match"].sum()) == 100
        and int((control_equivalence["fd_armijo_accepted_alpha"] > 0.0).sum()) == 25
    )
    stopped_norm_row = stopped_preflight.loc[stopped_preflight["kind"].eq("proposal")].iloc[0]
    stopped_slope_rows = stopped_preflight.loc[stopped_preflight["kind"].str.startswith("slope")]
    stopped_line_rows = stopped_preflight.loc[stopped_preflight["kind"].eq("line")]
    stopped_norm_relative_error = float(stopped_norm_row["absolute_error"]) / max(
        abs(float(stopped_norm_row["expected"])), 1e-30
    )
    forward_scalar_mixed_tolerance_pass = bool(
        all(
            abs(float(row[observed]) - float(row[reference]))
            <= 1e-5 + 1e-5 * max(abs(float(row[observed])), abs(float(row[reference])))
            for _, row in forward_scalars.iterrows()
            for reference, observed in (("stopped_a", "treatment_a"), ("stopped_b", "treatment_b"))
        )
    )
    stopped_forward_composite = float(
        (forward_scalars["stopped_a"] + float(args.beta) * forward_scalars["stopped_b"]).mean()
    )
    treatment_forward_composite = float(
        (forward_scalars["treatment_a"] + float(args.beta) * forward_scalars["treatment_b"]).mean()
    )
    forward_composite_abs_error = abs(stopped_forward_composite - treatment_forward_composite)
    stopped_line_max_abs_error = float(stopped_line_rows["absolute_error"].max())
    stopped_line_max_relative_error = float(
        (
            stopped_line_rows["absolute_error"]
            / np.maximum(stopped_line_rows["observed"].abs(), stopped_line_rows["expected"].abs()).clip(lower=1e-30)
        ).max()
    )
    stopped_replay_slopes = {
        1.0 / 128.0: float(stopped_slope_rows.loc[stopped_slope_rows["kind"].eq("slope_h128"), "observed"].iloc[0]),
        1.0 / 256.0: float(stopped_slope_rows.loc[stopped_slope_rows["kind"].eq("slope_h256"), "observed"].iloc[0]),
    }
    stopped_replay_candidates = {
        float(row["alpha"]): {"true_objective": float(row["observed"])}
        for _, row in stopped_line_rows.iterrows()
    }
    stopped_replay_alpha = _fd_armijo_alpha(
        current_f=float(states.iloc[0]["true_objective"]),
        slopes=stopped_replay_slopes,
        candidates=stopped_replay_candidates,
        c1=args.armijo_c1,
        strict_tolerance=STRICT_DECREASE_TOLERANCE,
    )
    canonical_proposal1 = pd.read_csv(ITERATION1_OUTPUT / "proposal_diagnostics.csv").iloc[0]
    canonical_proposal1_alpha = float(canonical_proposal1["accepted_alpha"])
    stopped_slope_signs_match = bool(
        all(
            math.copysign(1.0, float(row["observed"])) == math.copysign(1.0, float(row["expected"]))
            for _, row in stopped_slope_rows.iterrows()
        )
    )
    validity_gates = {
        "frozen_setup": frozen_setup,
        "hard_frozen_dependency_manifest_matches": hard_freeze_valid,
        "checkpoint_hash_matches": checkpoint_hash == EXPECTED_CHECKPOINT,
        "control_source_hash_matches": control_source_matches,
        "control_states_hash_matches": dependency_hashes["iteration1_states"] == CONTROL_STATES_SHA256,
        "control_proposals_hash_matches": dependency_hashes["iteration1_proposals"] == CONTROL_PROPOSALS_SHA256,
        "control_lines_hash_matches": dependency_hashes["iteration1_lines"] == CONTROL_LINES_SHA256,
        "z_hash_matches": z_hash == EXPECTED_Z_SHA256,
        "full_ce_batch_indices_are_none": full_ce_batch_none,
        "p4_seed_protocol_matches": A_ONLY_PROTOCOL_ID == "one_state_a_optimization_smoke_h2048_v1",
        "control_fd_armijo_alpha_matches_100_of_100": control_equivalence_matches,
        "state_rows_101": len(states) == 101,
        "proposal_rows_100": len(proposals) == 100,
        "line_candidate_rows_700": int(lines["kind"].eq("line").sum()) == 700,
        "fd_candidate_rows_400": int(lines["kind"].str.startswith("fd_").sum()) == 400,
        "all_numeric_values_finite": numeric_finite,
        "initial_f_matches_control_at_most_1e_5": abs(float(states.iloc[0]["true_objective"]) - float(control_states.iloc[0]["true_objective"])) <= 1e-5,
        "initial_a_matches_control_at_most_1e_5": abs(float(states.iloc[0]["exact_a_per_dim"]) - float(control_initial["exact_a_per_dim"])) <= 1e-5,
        "initial_b_matches_control_at_most_1e_5": abs(float(states.iloc[0]["damped_full_burg_per_dim"]) - float(control_initial["damped_full_burg_per_dim"])) <= 1e-5,
        "forward_equivalence_at_most_1e_5": forward_max_error <= 1e-5,
        "proposal1_forward_a_b_mixed_tolerance": forward_scalar_mixed_tolerance_pass,
        "proposal1_forward_composite_abs_error_at_most_1e_4": forward_composite_abs_error <= 1e-4,
        "stopped_proposal1_norm_relative_error_at_most_1e_6": stopped_norm_relative_error <= 1e-6,
        "stopped_proposal1_slopes_abs_error_at_most_5e_3": float(stopped_slope_rows["absolute_error"].max()) <= 5e-3,
        "stopped_proposal1_line_f_abs_error_at_most_1e_4": stopped_line_max_abs_error <= 1e-4,
        "stopped_proposal1_line_f_relative_error_at_most_1e_6": stopped_line_max_relative_error <= 1e-6,
        "stopped_proposal1_replay_alpha_matches_canonical": abs(stopped_replay_alpha - canonical_proposal1_alpha) <= 1e-12,
        "stopped_proposal1_slope_signs_match_canonical": stopped_slope_signs_match,
        "repeat_f_error_at_most_1e_5": max(repeat_errors, default=0.0) <= 1e-5,
        "final_candidate_f_recomputed_at_most_1e_5": abs(final_objective_recomputed - final_f) <= 1e-5,
        "all_rejected_state_hashes_unchanged": all(rejected_state_hash_checks),
        "rejected_hash_check_count_matches": len(rejected_state_hash_checks) == rejected_count,
        "final_checkpoint_model_reload_matches": checkpoint_model_reload_matches,
        "final_checkpoint_moments_reload_match": checkpoint_moments_reload_match,
    }
    success_gates = {
        "final_f_at_most_frozen_20pct_target": final_f <= FROZEN_TARGET_F,
        "beats_frozen_old_best": final_f < FROZEN_OLD_BEST_F,
        "at_least_90_strict_decreases": strict_count >= 90,
        "at_most_10_rejections": rejected_count <= 10,
        "no_accepted_increases": bool((accepted_rows["accepted_delta_f"] < -STRICT_DECREASE_TOLERANCE).all()),
        "maximum_rejection_run_at_most_3": maximum_rejection_run <= 3,
        "at_least_90_negative_slope_proposals": negative_count >= 90,
        "at_least_95_slope_sign_agreements": slope_sign_agreement_count >= 95,
        "median_slope_disagreement_at_most_0p10": median_slope_disagreement <= 0.10,
        "p95_slope_disagreement_at_most_0p25": p95_slope_disagreement <= 0.25,
        "median_alpha_at_least_one_eighth": median_alpha >= 1.0 / 8.0,
        "path_fraction_at_least_0p25": path_fraction >= 0.25,
    }
    valid = bool(all(validity_gates.values()))
    success = bool(all(success_gates.values())) if valid else None
    summary = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "normalized_source_sha256": normalized_source_hash,
        "frozen_dependency_manifest_sha256": frozen_manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "source_weight_index": source_index,
        "z_sha256": z_hash,
        "initial_parameter_hash": initial_parameter_hash,
        "final_parameter_hash": _named_tensor_hash(active_names, active),
        "initial_true_objective": initial_f,
        "final_true_objective": final_f,
        "absolute_change": final_f - initial_f,
        "relative_change": (final_f - initial_f) / initial_f,
        "old_best_true_objective": old_best,
        "frozen_old_best_true_objective": FROZEN_OLD_BEST_F,
        "frozen_20pct_target": FROZEN_TARGET_F,
        "strict_decreases": strict_count,
        "rejected_steps": rejected_count,
        "negative_slope_both_count": negative_count,
        "slope_sign_agreement_count": slope_sign_agreement_count,
        "median_slope_relative_disagreement": median_slope_disagreement,
        "p95_slope_relative_disagreement": p95_slope_disagreement,
        "maximum_rejection_run": maximum_rejection_run,
        "median_accepted_alpha": median_alpha,
        "used_path_fraction": path_fraction,
        "max_repeat_objective_abs_error": max(repeat_errors, default=0.0),
        "final_objective_recomputed": final_objective_recomputed,
        "final_objective_recompute_abs_error": abs(final_objective_recomputed - final_f),
        "forward_equivalence_max_relative_error": forward_max_error,
        "forward_composite_abs_error": forward_composite_abs_error,
        "stopped_proposal1_line_max_abs_error": stopped_line_max_abs_error,
        "stopped_proposal1_line_max_relative_error": stopped_line_max_relative_error,
        "stopped_proposal1_replay_alpha": stopped_replay_alpha,
        "canonical_proposal1_alpha": canonical_proposal1_alpha,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_sec": time.perf_counter() - started,
        "validity_gates": validity_gates,
        "success_gates": success_gates if valid else None,
        "valid": valid,
        "all_success_gates_pass": success,
        "mechanism_decision": (
            "naive_semantics_multi_step_repair_passed" if success else "naive_semantics_multi_step_repair_failed"
        )
        if valid
        else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot(
        states=states,
        proposals=proposals,
        control_states=control_states,
        beta=args.beta,
        old_best=old_best,
        output=args.output_dir / "true_objective_curve.png",
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
    print(f"[naive-i3] summary {json.dumps(summary, sort_keys=True)}", flush=True)
    print(
        f"[naive-i3] complete success={summary['all_success_gates_pass']} "
        f"F={initial_f:.8g}->{final_f:.8g} decreases={strict_count}/{args.proposals} "
        f"rejected={rejected_count} elapsed={summary['elapsed_sec']:.1f}s",
        flush=True,
    )
    print(
        "[naive-i3] artifacts=" + ",".join(str(path) for path in sorted(args.output_dir.iterdir())),
        flush=True,
    )


if __name__ == "__main__":
    main()
