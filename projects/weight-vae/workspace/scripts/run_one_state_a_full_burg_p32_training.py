from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import scripts.run_one_state_a_full_burg_naive_semantics as iteration3
from scripts.audit_variant_a_estimator_stability import sha256_file, stable_uint63
from scripts.run_one_state_a_full_burg_armijo import _adam_proposal, _vector_dot, _vector_norm
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    A_ONLY_PROTOCOL_ID,
    OUTPUT_ROOT,
    _gradient_norms,
    _pair_losses,
    _train_generators,
)


PROTOCOL_ID = "one_state_a_full_burg_p32_training_iteration5_v1"
PROTOCOL = OUTPUT_ROOT / "iteration_5_p32_training_protocol.md"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration5_p32_training_production"
STAGING_OUTPUT = OUTPUT_ROOT / "iteration5_p32_training_production.incomplete"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_5_frozen_dependency_manifest.json"
EXPECTED_FROZEN_MANIFEST_SHA256 = "75beb75b2821c72fca49ebd24f09a02b26cf0696e4327f92c85c4e829d754c4e"
EXPECTED_NORMALIZED_SOURCE_SHA256 = "a20680796896382878a8109c938d2722e26b1686b551871d488a46fc43e49d11"
POOL_DRAWS = 8
PAIRS_PER_DRAW = 4
ITERATION3_OUTPUT = OUTPUT_ROOT / "iteration3_naive_semantics_production"
ITERATION4_OUTPUT = OUTPUT_ROOT / "iteration4_probe_adam_cross_production_v2"


ORIGINAL_BUILD_PROPOSAL = iteration3._build_proposal
DRAW_ROWS: list[dict[str, float | int]] = []
PAIR_ROWS: list[dict[str, float | int]] = []
POOLING_PREFLIGHT: dict[str, float | int | bool] = {}
TREATMENT_BUILD_CALLS = 0
TREATMENT_CLIP_CALLS = 0
TREATMENT_ADAM_CALLS = 0
SEED_PREFLIGHT: dict[str, Any] = {}


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


def _branch_seed(proposal: int, draw: int, pair: int, branch: int) -> int:
    if draw == 0:
        return stable_uint63(A_ONLY_PROTOCOL_ID, "train", proposal, pair, branch)
    return stable_uint63(PROTOCOL_ID, "p32_extra", proposal, draw, pair, branch)


def _seed_schedule_hash(rows: list[dict[str, int]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seed_schedule_preflight() -> dict[str, Any]:
    rows: list[dict[str, int]] = []
    canonical_matches = True
    for proposal in range(1, 101):
        for draw in range(POOL_DRAWS):
            for pair in range(PAIRS_PER_DRAW):
                seed_1 = _branch_seed(proposal, draw, pair, 0)
                seed_2 = _branch_seed(proposal, draw, pair, 1)
                if draw == 0:
                    canonical_1, canonical_2 = _train_generators(proposal, pair)
                    canonical_matches = canonical_matches and (
                        seed_1 == canonical_1.initial_seed()
                        and seed_2 == canonical_2.initial_seed()
                    )
                rows.append(
                    {
                        "proposal": proposal,
                        "draw": draw,
                        "pair": pair,
                        "seed_1": seed_1,
                        "seed_2": seed_2,
                    }
                )
    seeds = [row[key] for row in rows for key in ("seed_1", "seed_2")]
    result = {
        "row_count": len(rows),
        "branch_seed_count": len(seeds),
        "unique_branch_seed_count": len(set(seeds)),
        "draw0_matches_canonical_generators": canonical_matches,
        "schedule_sha256": _seed_schedule_hash(rows),
    }
    result["valid"] = bool(
        len(rows) == 3200
        and len(seeds) == 6400
        and len(set(seeds)) == 6400
        and canonical_matches
    )
    if not result["valid"]:
        raise RuntimeError("P32 seed schedule preflight failed: " + json.dumps(result, sort_keys=True))
    return result


def _schedule_from_pair_frame(frame: pd.DataFrame) -> list[dict[str, int]]:
    ordered = frame.sort_values(["proposal", "draw", "pair"])
    return [
        {
            "proposal": int(row.proposal),
            "draw": int(row.draw),
            "pair": int(row.pair),
            "seed_1": int(row.seed_1),
            "seed_2": int(row.seed_2),
        }
        for row in ordered.itertuples(index=False)
    ]


def _pair_generators(
    proposal: int, draw: int, pair: int
) -> tuple[torch.Generator, torch.Generator, int, int]:
    seed_1 = _branch_seed(proposal, draw, pair, 0)
    seed_2 = _branch_seed(proposal, draw, pair, 1)
    return (
        torch.Generator(device="cpu").manual_seed(seed_1),
        torch.Generator(device="cpu").manual_seed(seed_2),
        seed_1,
        seed_2,
    )


def _materialize_gradients(
    gradients: tuple[torch.Tensor | None, ...], active: list[torch.nn.Parameter]
) -> list[torch.Tensor]:
    return [
        torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
        for parameter, gradient in zip(active, gradients, strict=True)
    ]


def _relative_error(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    difference2 = torch.zeros((), device=left[0].device, dtype=torch.float64)
    reference2 = torch.zeros_like(difference2)
    for left_tensor, right_tensor in zip(left, right, strict=True):
        difference2 += (left_tensor.detach().double() - right_tensor.detach().double()).square().sum()
        reference2 += right_tensor.detach().double().square().sum()
    return float((difference2.sqrt() / reference2.sqrt().clamp_min(1e-30)).cpu())


def _direct_p32_gradient(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active: list[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    beta: float,
    proposal: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    direct_a = [torch.zeros_like(parameter) for parameter in active]
    direct_b = [torch.zeros_like(parameter) for parameter in active]
    completed_pairs = 0
    for draw in range(POOL_DRAWS):
        for pair in range(PAIRS_PER_DRAW):
            generator_1, generator_2, _seed_1, _seed_2 = _pair_generators(
                proposal, draw, pair
            )
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
            raw_a = torch.autograd.grad(a_loss, active, retain_graph=True, allow_unused=True)
            raw_b = torch.autograd.grad(b_loss, active, retain_graph=False, allow_unused=True)
            with torch.no_grad():
                for index, (grad_a, grad_b) in enumerate(zip(raw_a, raw_b, strict=True)):
                    if grad_a is not None:
                        direct_a[index].add_(grad_a.detach(), alpha=1.0 / 32.0)
                    if grad_b is not None:
                        direct_b[index].add_(grad_b.detach(), alpha=1.0 / 32.0)
            completed_pairs += 1
            if completed_pairs in (1, 8, 16, 24, 32):
                print(
                    f"[p32-i5] stage=direct-p32-preflight pair={completed_pairs}/32",
                    flush=True,
                )
            del a_loss, b_loss, raw_a, raw_b
    direct_total = [
        grad_a + float(beta) * grad_b
        for grad_a, grad_b in zip(direct_a, direct_b, strict=True)
    ]
    return direct_a, direct_b, direct_total


def _build_p32_proposal(
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
    global TREATMENT_BUILD_CALLS, TREATMENT_CLIP_CALLS, TREATMENT_ADAM_CALLS
    if pairs != PAIRS_PER_DRAW:
        raise RuntimeError(f"P32 requires exactly {PAIRS_PER_DRAW} pairs per draw")
    TREATMENT_BUILD_CALLS += 1

    pooled_a_sum = [torch.zeros_like(parameter) for parameter in active]
    pooled_b_sum = [torch.zeros_like(parameter) for parameter in active]
    draw_total_gradients: list[list[torch.Tensor]] = []
    draw_diagnostics: list[dict[str, float | int]] = []
    draw0_pair_scalars: list[dict[str, float | int]] = []
    all_a_scalars: list[float] = []
    all_b_scalars: list[float] = []

    for draw in range(POOL_DRAWS):
        a_losses: list[torch.Tensor] = []
        b_losses: list[torch.Tensor] = []
        local_pair_scalars: list[dict[str, float | int]] = []
        for pair in range(PAIRS_PER_DRAW):
            generator_1, generator_2, seed_1, seed_2 = _pair_generators(
                proposal, draw, pair
            )
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
            pair_row = {
                "proposal": proposal,
                "draw": draw,
                "pair": pair,
                "seed_1": seed_1,
                "seed_2": seed_2,
                "a_loss": float(a_loss.detach().cpu()),
                "b_pseudo_loss": float(b_loss.detach().cpu()),
            }
            PAIR_ROWS.append(pair_row)
            local_pair_scalars.append(
                {
                    "pair": pair,
                    "a_loss": pair_row["a_loss"],
                    "b_pseudo_loss": pair_row["b_pseudo_loss"],
                }
            )
            all_a_scalars.append(float(pair_row["a_loss"]))
            all_b_scalars.append(float(pair_row["b_pseudo_loss"]))
        if draw == 0:
            draw0_pair_scalars = local_pair_scalars

        mean_a = torch.stack(a_losses).mean()
        mean_b = torch.stack(b_losses).mean()
        raw_a = torch.autograd.grad(mean_a, active, retain_graph=True, allow_unused=True)
        raw_b = torch.autograd.grad(mean_b, active, retain_graph=False, allow_unused=True)
        unused_parameter_tensors = sum(
            grad_a is None and grad_b is None
            for grad_a, grad_b in zip(raw_a, raw_b, strict=True)
        )
        gradients_a = _materialize_gradients(raw_a, active)
        gradients_b = _materialize_gradients(raw_b, active)
        component_stats = _gradient_norms(tuple(gradients_a), tuple(gradients_b), beta=beta)
        draw_total = [
            grad_a + float(beta) * grad_b
            for grad_a, grad_b in zip(gradients_a, gradients_b, strict=True)
        ]
        with torch.no_grad():
            for index, (grad_a, grad_b) in enumerate(
                zip(gradients_a, gradients_b, strict=True)
            ):
                pooled_a_sum[index].add_(grad_a)
                pooled_b_sum[index].add_(grad_b)
        draw_total_gradients.append([value.detach().cpu() for value in draw_total])
        draw_diagnostics.append(
            {
                "proposal": proposal,
                "draw": draw,
                "mean_a": float(mean_a.detach().cpu()),
                "mean_b": float(mean_b.detach().cpu()),
                "unused_parameter_tensors": unused_parameter_tensors,
                **component_stats,
            }
        )
        if draw in (0, 3, 7):
            print(
                f"[p32-i5] stage=pooled-gradient proposal={proposal}/100 "
                f"draw={draw + 1}/{POOL_DRAWS}",
                flush=True,
            )
        del a_losses, b_losses, mean_a, mean_b, raw_a, raw_b, gradients_a, gradients_b, draw_total

    pooled_a = [value / float(POOL_DRAWS) for value in pooled_a_sum]
    pooled_b = [value / float(POOL_DRAWS) for value in pooled_b_sum]
    pooled_total = [
        grad_a + float(beta) * grad_b
        for grad_a, grad_b in zip(pooled_a, pooled_b, strict=True)
    ]
    component_stats = _gradient_norms(tuple(pooled_a), tuple(pooled_b), beta=beta)
    pooled_total_cpu = [value.detach().cpu() for value in pooled_total]
    pooled_norm = _vector_norm(pooled_total_cpu)

    draw_cosines: list[float] = []
    draw_relative_errors: list[float] = []
    for row, draw_total in zip(draw_diagnostics, draw_total_gradients, strict=True):
        draw_norm = _vector_norm(draw_total)
        cosine = _vector_dot(draw_total, pooled_total_cpu) / max(draw_norm * pooled_norm, 1e-30)
        relative_error = _relative_error(draw_total, pooled_total_cpu)
        row["cosine_to_pool"] = cosine
        row["relative_error_to_pool"] = relative_error
        DRAW_ROWS.append(row)
        draw_cosines.append(cosine)
        draw_relative_errors.append(relative_error)

    if proposal == 1:
        direct_a, direct_b, direct_total = _direct_p32_gradient(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            active=active,
            burg_gradient=burg_gradient,
            beta=beta,
            proposal=proposal,
        )
        POOLING_PREFLIGHT.update(
            {
                "proposal": proposal,
                "sequential_to_direct_a_relative_error": _relative_error(pooled_a, direct_a),
                "sequential_to_direct_b_relative_error": _relative_error(pooled_b, direct_b),
                "sequential_to_direct_total_relative_error": _relative_error(
                    pooled_total, direct_total
                ),
            }
        )
        pooling_errors = [
            float(value)
            for key, value in POOLING_PREFLIGHT.items()
            if key.endswith("relative_error")
        ]
        pooling_valid = bool(
            int(POOLING_PREFLIGHT["proposal"]) == 1
            and len(pooling_errors) == 3
            and all(math.isfinite(error) and error <= 1e-6 for error in pooling_errors)
        )
        POOLING_PREFLIGHT["valid"] = pooling_valid
        del direct_a, direct_b, direct_total
        if not pooling_valid:
            raise RuntimeError(
                "proposal-1 sequential/direct P32 identity preflight failed: "
                + json.dumps(POOLING_PREFLIGHT, sort_keys=True)
            )

    with torch.no_grad():
        for parameter, gradient in zip(active, pooled_total, strict=True):
            parameter.grad = gradient
    TREATMENT_CLIP_CALLS += 1
    preclip_norm = float(
        torch.nn.utils.clip_grad_norm_(active, max_norm=gradient_clip).detach().cpu()
    )
    clip_factor = min(1.0, float(gradient_clip) / max(preclip_norm, 1e-30))
    clipped_gradients = [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in active
    ]
    TREATMENT_ADAM_CALLS += 1
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
        "mean_a": float(np.mean(all_a_scalars)),
        "mean_b": float(np.mean(all_b_scalars)),
        "pair_scalars": draw0_pair_scalars,
        "component_stats": {
            **component_stats,
            "draw_cosine_to_pool_min": float(np.min(draw_cosines)),
            "draw_cosine_to_pool_median": float(np.median(draw_cosines)),
            "draw_relative_error_to_pool_max": float(np.max(draw_relative_errors)),
            "draw_relative_error_to_pool_median": float(np.median(draw_relative_errors)),
        },
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
    del (
        pooled_a_sum,
        pooled_b_sum,
        pooled_a,
        pooled_b,
        pooled_total,
        pooled_total_cpu,
        draw_total_gradients,
        draw_diagnostics,
        clipped_gradients,
    )
    return result


def _proposal_dispatch(**kwargs: Any) -> dict[str, Any]:
    cfg = kwargs["cfg"]
    if str(cfg.vae_precond_hvp_mode) == "stopped_composite":
        return ORIGINAL_BUILD_PROPOSAL(**kwargs)
    return _build_p32_proposal(**kwargs)


def _numeric_finite(frame: pd.DataFrame) -> bool:
    return bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all())


def _plot_final(output: Path) -> None:
    states = pd.read_csv(output / "state_objective_curve.csv")
    proposals = pd.read_csv(output / "proposal_diagnostics.csv")
    control = pd.read_csv(ITERATION3_OUTPUT / "state_objective_curve.csv")
    stopped = pd.read_csv(iteration3.ITERATION1_OUTPUT / "state_objective_curve.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(states["proposal"], states["true_objective"], linewidth=2.3, label="P32 full derivatives")
    axes[0, 0].plot(control["proposal"], control["true_objective"], alpha=0.8, label="P4 full derivatives")
    axes[0, 0].plot(stopped["proposal"], stopped["true_objective"], alpha=0.65, label="P4 stopped control")
    axes[0, 0].axhline(iteration3.FROZEN_TARGET_F, color="#16803c", linestyle="--", label="frozen target")
    axes[0, 0].set(title="Literal dense true-F curve", xlabel="proposal", ylabel="F = A + beta B")
    axes[0, 0].legend()

    axes[0, 1].plot(proposals["proposal"], proposals["slope_h128"], label="FD h=1/128")
    axes[0, 1].plot(proposals["proposal"], proposals["slope_h256"], alpha=0.8, label="FD h=1/256")
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_yscale("symlog", linthresh=0.1)
    axes[0, 1].set(title="True-F directional slopes", xlabel="proposal", ylabel="dF / d alpha")
    axes[0, 1].legend()

    accepted_alpha = proposals["accepted_alpha"].replace(0.0, np.nan)
    axes[1, 0].scatter(proposals["proposal"], accepted_alpha, s=22, label="accepted alpha")
    rejected = proposals.loc[proposals["accepted"].eq(0), "proposal"]
    if len(rejected):
        axes[1, 0].scatter(rejected, np.full(len(rejected), 1.0 / 128.0), marker="x", color="#c53832", label="rejected")
    axes[1, 0].set_yscale("log", base=2)
    axes[1, 0].set(title="Line-search acceptance", xlabel="proposal", ylabel="fraction of Adam proposal")
    axes[1, 0].legend()

    axes[1, 1].plot(proposals["proposal"], proposals["draw_cosine_to_pool_median"], label="median draw cosine")
    axes[1, 1].plot(proposals["proposal"], proposals["draw_cosine_to_pool_min"], alpha=0.75, label="minimum draw cosine")
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set(title="P4 draw agreement with pooled P32", xlabel="proposal", ylabel="cosine")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "true_objective_curve.png", dpi=180)
    plt.close(fig)


def _postprocess(output: Path) -> dict[str, Any]:
    pairs = pd.DataFrame(PAIR_ROWS)
    draws = pd.DataFrame(DRAW_ROWS)
    pairs.to_csv(output / "p32_pair_scalars.csv", index=False)
    draws.to_csv(output / "p32_draw_diagnostics.csv", index=False)
    (output / "p32_pooling_preflight.json").write_text(
        json.dumps(POOLING_PREFLIGHT, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    actual_schedule = _schedule_from_pair_frame(pairs)
    actual_schedule_hash = _seed_schedule_hash(actual_schedule)
    seed_review = {
        **SEED_PREFLIGHT,
        "observed_schedule_sha256": actual_schedule_hash,
        "observed_matches_preflight": actual_schedule_hash == SEED_PREFLIGHT["schedule_sha256"],
    }
    (output / "p32_seed_schedule_preflight.json").write_text(
        json.dumps(seed_review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    all_seeds = pd.concat([pairs["seed_1"], pairs["seed_2"]], ignore_index=True)
    draw0 = pairs.loc[pairs["draw"].eq(0)]
    draw0_seed_matches = all(
        int(getattr(row, f"seed_{branch + 1}"))
        == _train_generators(int(row.proposal), int(row.pair))[branch].initial_seed()
        for row in draw0.itertuples(index=False)
        for branch in (0, 1)
    )
    preflight_errors = [
        float(value)
        for key, value in POOLING_PREFLIGHT.items()
        if key.endswith("relative_error")
    ]
    expected_draw_keys = {
        (proposal, draw)
        for proposal in range(1, 101)
        for draw in range(POOL_DRAWS)
    }
    observed_draw_keys = set(
        zip(draws["proposal"].astype(int), draws["draw"].astype(int), strict=True)
    )
    expected_pair_keys = {
        (proposal, draw, pair)
        for proposal in range(1, 101)
        for draw in range(POOL_DRAWS)
        for pair in range(PAIRS_PER_DRAW)
    }
    observed_pair_keys = set(
        zip(
            pairs["proposal"].astype(int),
            pairs["draw"].astype(int),
            pairs["pair"].astype(int),
            strict=True,
        )
    )
    extra_validity = {
        "p32_draw_rows_800": len(draws) == 800,
        "p32_pair_rows_3200": len(pairs) == 3200,
        "p32_draw_cartesian_key_coverage": observed_draw_keys == expected_draw_keys,
        "p32_pair_cartesian_key_coverage": observed_pair_keys == expected_pair_keys,
        "eight_draws_per_proposal": bool(draws.groupby("proposal").size().eq(POOL_DRAWS).all()),
        "four_pairs_per_draw": bool(pairs.groupby(["proposal", "draw"]).size().eq(PAIRS_PER_DRAW).all()),
        "all_6400_branch_seeds_unique": len(all_seeds) == 6400 and all_seeds.nunique() == 6400,
        "draw0_seeds_match_canonical_p4": bool(draw0_seed_matches),
        "independent_seed_schedule_preflight_matches_rows": bool(
            SEED_PREFLIGHT.get("valid") is True
            and seed_review["observed_matches_preflight"] is True
        ),
        "sequential_direct_pooling_errors_at_most_1e_6": bool(
            POOLING_PREFLIGHT.get("proposal") == 1
            and POOLING_PREFLIGHT.get("valid") is True
            and len(preflight_errors) == 3
            and all(math.isfinite(error) and error <= 1e-6 for error in preflight_errors)
        ),
        "one_treatment_build_per_proposal": TREATMENT_BUILD_CALLS == 100,
        "one_treatment_clip_per_proposal": TREATMENT_CLIP_CALLS == 100,
        "one_treatment_adam_per_proposal": TREATMENT_ADAM_CALLS == 100,
        "no_unused_active_tensors": bool(draws["unused_parameter_tensors"].eq(0).all()),
        "p32_tables_numeric_finite": _numeric_finite(pairs) and _numeric_finite(draws),
    }

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pool_draws"] = POOL_DRAWS
    summary["pairs_per_draw"] = PAIRS_PER_DRAW
    summary["effective_pairs_per_proposal"] = POOL_DRAWS * PAIRS_PER_DRAW
    summary["p32_pooling_preflight"] = POOLING_PREFLIGHT
    summary["validity_gates"].update(extra_validity)
    summary["valid"] = bool(all(summary["validity_gates"].values()))
    summary["causal_repair_gates"] = {
        "fewer_rejections_than_p4": int(summary["rejected_steps"]) < 19,
        "more_strict_decreases_than_p4": int(summary["strict_decreases"]) > 81,
        "shorter_maximum_plateau_than_p4": int(summary["maximum_rejection_run"]) < 8,
    }
    checkpoint = torch.load(output / "final_checkpoint.pt", map_location="cpu", weights_only=False)
    active_names = sorted(checkpoint["active_model_state"])
    checkpoint_parameter_hash = iteration3._named_tensor_hash(
        active_names, [checkpoint["active_model_state"][name] for name in active_names]
    )
    checkpoint_moment_hash = hashlib.sha256(
        (
            iteration3._named_tensor_hash(
                active_names, [checkpoint["exp_avg"][name] for name in active_names]
            )
            + iteration3._named_tensor_hash(
                active_names, [checkpoint["exp_avg_sq"][name] for name in active_names]
            )
            + str(int(checkpoint["accepted_adam_step"]))
        ).encode("utf-8")
    ).hexdigest()
    summary["final_checkpoint_parameter_hash"] = checkpoint_parameter_hash
    summary["final_checkpoint_moment_hash"] = checkpoint_moment_hash
    summary["final_checkpoint_adam_step"] = int(checkpoint["accepted_adam_step"])
    summary["validity_gates"]["checkpoint_parameter_hash_matches_summary"] = bool(
        checkpoint_parameter_hash == summary["final_parameter_hash"]
    )
    frozen_success_gates = summary.get("success_gates") or {}
    provisional_success = bool(
        summary["valid"]
        and frozen_success_gates
        and all(frozen_success_gates.values())
        and all(summary["causal_repair_gates"].values())
    )
    summary["valid"] = bool(all(summary["validity_gates"].values()))
    summary["provisional_success_gates_pass"] = provisional_success if summary["valid"] else None
    summary["all_success_gates_pass"] = None
    summary["mechanism_decision"] = "awaiting_independent_replay" if summary["valid"] else None
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    resolved_path = output / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved.update(
        {
            "protocol_id": PROTOCOL_ID,
            "pool_draws": POOL_DRAWS,
            "pairs_per_draw": PAIRS_PER_DRAW,
            "effective_pairs_per_proposal": POOL_DRAWS * PAIRS_PER_DRAW,
            "pooling_order": "average raw P4 component gradients, then one global clip and one carried-Adam proposal",
            "draw0_seed_protocol": A_ONLY_PROTOCOL_ID,
            "extra_draw_seed_protocol": PROTOCOL_ID,
            "seed_schedule_sha256": SEED_PREFLIGHT["schedule_sha256"],
            "output_dir": str(DEFAULT_OUTPUT),
            "staging_output_dir": str(STAGING_OUTPUT),
            "completion_policy": "staging directory plus atomic rename after COMPLETED.json",
        }
    )
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _plot_final(output)
    artifact_paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "INCOMPLETE", "COMPLETED.json", "run.log"}
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": _source_sha256(),
        "normalized_source_sha256": _normalized_source_sha256(),
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[p32-i5] final_summary {json.dumps(summary, sort_keys=True)}", flush=True)
    print(
        f"[p32-i5] complete valid={summary['valid']} provisional_success={summary['provisional_success_gates_pass']} "
        f"F={summary['initial_true_objective']:.8g}->{summary['final_true_objective']:.8g} "
        f"decreases={summary['strict_decreases']}/100 rejected={summary['rejected_steps']}",
        flush=True,
    )
    return summary


def _configure_iteration3_runner() -> None:
    iteration3.PROTOCOL_ID = PROTOCOL_ID
    iteration3.DEFAULT_OUTPUT = STAGING_OUTPUT
    iteration3.FROZEN_DEPENDENCY_MANIFEST = FROZEN_DEPENDENCY_MANIFEST
    iteration3.EXPECTED_FROZEN_MANIFEST_SHA256 = EXPECTED_FROZEN_MANIFEST_SHA256
    iteration3.EXPECTED_NORMALIZED_SOURCE_SHA256 = EXPECTED_NORMALIZED_SOURCE_SHA256
    iteration3.__file__ = str(Path(__file__).resolve())
    iteration3._build_proposal = _proposal_dispatch


def main() -> None:
    global SEED_PREFLIGHT
    if sys.argv[1:]:
        raise RuntimeError("Iteration-5 production runner accepts no CLI overrides")
    if DEFAULT_OUTPUT.exists():
        raise RuntimeError(f"production output directory already exists: {DEFAULT_OUTPUT}")
    if STAGING_OUTPUT.exists():
        raise RuntimeError(f"stale or active staging directory exists: {STAGING_OUTPUT}")
    decision = json.loads((ITERATION4_OUTPUT / "decision.json").read_text(encoding="utf-8"))
    if not (
        decision.get("valid") is True
        and decision.get("iteration5_candidate") == "p32_variance_controlled_training"
        and decision["mechanisms"]["finite_probe_carried_repair_supported"] is True
        and decision["mechanisms"]["primary_sign_ambiguity"] is False
    ):
        raise RuntimeError("Iteration-4 decision does not authorize P32 carried training")
    print(
        f"[p32-i5] setup protocol={PROTOCOL_ID} pool={POOL_DRAWS}xP{PAIRS_PER_DRAW} "
        f"output={DEFAULT_OUTPUT} staging={STAGING_OUTPUT} device=cuda:0 dtype=float32",
        flush=True,
    )
    SEED_PREFLIGHT = _seed_schedule_preflight()
    print(f"[p32-i5] seed_preflight {json.dumps(SEED_PREFLIGHT, sort_keys=True)}", flush=True)
    STAGING_OUTPUT.mkdir(parents=True, exist_ok=False)
    (STAGING_OUTPUT / "INCOMPLETE").write_text(
        json.dumps({"protocol_id": PROTOCOL_ID, "status": "running"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _configure_iteration3_runner()
    iteration3.main()
    summary = _postprocess(STAGING_OUTPUT)
    if not summary["valid"]:
        raise RuntimeError("Iteration-5 production validity failed; staging remains INCOMPLETE")
    completion = {
        "protocol_id": PROTOCOL_ID,
        "status": "production_complete_awaiting_independent_replay",
        "valid": summary["valid"],
        "provisional_success_gates_pass": summary["provisional_success_gates_pass"],
        "summary_sha256": sha256_file(STAGING_OUTPUT / "summary.json"),
        "artifact_manifest_sha256": sha256_file(STAGING_OUTPUT / "artifact_manifest.json"),
    }
    (STAGING_OUTPUT / "INCOMPLETE").unlink()
    (STAGING_OUTPUT / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(STAGING_OUTPUT, DEFAULT_OUTPUT)
    print(
        f"[p32-i5] atomic_publish output={DEFAULT_OUTPUT} "
        f"completion={json.dumps(completion, sort_keys=True)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
