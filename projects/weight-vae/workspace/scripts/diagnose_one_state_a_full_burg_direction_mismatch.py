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
from scripts.optimize_variant_a_full_burg_armijo_one_state import _evaluate, _set_parameters, _vector_norm
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    STATE_BANK,
    _generators,
    _gradient_norms,
    _pair_losses,
)


ROOT = Path(__file__).resolve().parents[1]
ARM_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration_1_armijo/production_100proposals_exact_fd"
)
DEFAULT_CHECKPOINT = ARM_DIR / "active_checkpoint_proposal40.pt"
DEFAULT_OUTPUT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration_2_direction_source/production_state40_draw16"
)
PROTOCOL_ID = "one_state_a_full_burg_direction_source_v1"
EXPECTED_BETA = 22.536727828943093
EXPECTED_Z = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_STATE_F = 152.943285533646
PREFIXES = (1, 2, 4, 8, 16)
FD_STEPS = (1.0 / 128.0, 1.0 / 256.0)


def _source_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _draw_generators(draw: int, pair: int) -> tuple[torch.Generator, torch.Generator]:
    return _generators(PROTOCOL_ID, "draw", draw, pair)


def _gradient_draw(
    *,
    run: Any,
    cfg: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: list[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    beta: float,
    draw: int,
    pairs: int,
    clip_norm: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    a_losses: list[torch.Tensor] = []
    b_losses: list[torch.Tensor] = []
    for pair in range(pairs):
        g1, g2 = _draw_generators(draw, pair)
        a_loss, b_loss, _stats = _pair_losses(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            pair_index=pair,
            probe_generator_1=g1,
            probe_generator_2=g2,
            burg_matrix_gradient=burg_gradient,
        )
        a_losses.append(a_loss)
        b_losses.append(b_loss)
    mean_a = torch.stack(a_losses).mean()
    mean_b = torch.stack(b_losses).mean()
    gradients_a = torch.autograd.grad(mean_a, active, retain_graph=True, allow_unused=True)
    gradients_b = torch.autograd.grad(mean_b, active, retain_graph=False, allow_unused=True)
    component = _gradient_norms(gradients_a, gradients_b, beta=beta)
    raw: list[torch.Tensor] = []
    with torch.no_grad():
        for parameter, ga, gb in zip(active, gradients_a, gradients_b, strict=True):
            if ga is None:
                ga = torch.zeros_like(parameter)
            if gb is None:
                gb = torch.zeros_like(parameter)
            raw.append(ga.detach().clone().add_(gb.detach(), alpha=float(beta)))
    raw_norm = _vector_norm(raw)
    with torch.no_grad():
        for parameter, value in zip(active, raw, strict=True):
            parameter.grad = value.detach().clone()
    clip_api_norm = float(torch.nn.utils.clip_grad_norm_(active, max_norm=clip_norm).detach().cpu())
    clipped = [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in active
    ]
    factor = min(1.0, float(clip_norm) / (max(raw_norm, 0.0) + 1e-6))
    for parameter in active:
        parameter.grad = None
    stats = {
        "train_a": float(mean_a.detach().cpu()),
        "train_b_pseudo_loss": float(mean_b.detach().cpu()),
        "raw_gradient_norm": raw_norm,
        "clip_api_gradient_norm": clip_api_norm,
        "clip_factor": factor,
        "clipped_gradient_norm": _vector_norm(clipped),
        **component,
    }
    del a_losses, b_losses, mean_a, mean_b, gradients_a, gradients_b
    return raw, clipped, stats


def _scaled_negative_gradient(gradient: list[torch.Tensor], *, target_norm: float) -> list[torch.Tensor]:
    norm = _vector_norm(gradient)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"invalid gradient norm: {norm}")
    scale = float(target_norm) / norm
    return [value.detach().clone().mul_(-scale) for value in gradient]


def _provisional_adam_direction(
    *,
    optimizer: torch.optim.Adam,
    optimizer_state: dict[str, Any],
    active: list[torch.nn.Parameter],
    base: list[torch.Tensor],
    gradient: list[torch.Tensor],
) -> list[torch.Tensor]:
    _set_parameters(active, base)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    with torch.no_grad():
        for parameter, value in zip(active, gradient, strict=True):
            parameter.grad = value.detach().clone()
    optimizer.step()
    displacement = [
        parameter.detach().clone().sub_(origin)
        for parameter, origin in zip(active, base, strict=True)
    ]
    _set_parameters(active, base)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    optimizer.zero_grad(set_to_none=True)
    return displacement


def _clip_vector(values: list[torch.Tensor], clip_norm: float) -> tuple[list[torch.Tensor], float, float]:
    norm = _vector_norm(values)
    factor = min(1.0, float(clip_norm) / (max(norm, 0.0) + 1e-6))
    return [value.detach().clone().mul_(factor) for value in values], norm, factor


def _direction_profile(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: list[torch.nn.Parameter],
    base: list[torch.Tensor],
    direction: list[torch.Tensor],
    current_f: float,
    beta: float,
    epsilon: float,
    hessian_chunk_size: int,
) -> dict[str, float | bool]:
    values: dict[float, float] = {}
    for alpha in (1.0, FD_STEPS[0], -FD_STEPS[0], FD_STEPS[1], -FD_STEPS[1]):
        _set_parameters(active, base, direction, alpha=alpha)
        row, hessian, metric, eig_m, burg_gradient = _evaluate(
            run=run,
            z=z,
            record=record,
            epsilon=epsilon,
            beta=beta,
            hessian_chunk_size=hessian_chunk_size,
        )
        values[alpha] = float(row["true_dense_f"])
        del row, hessian, metric, eig_m, burg_gradient
    _set_parameters(active, base)
    slope_128 = (values[FD_STEPS[0]] - values[-FD_STEPS[0]]) / (2.0 * FD_STEPS[0])
    slope_256 = (values[FD_STEPS[1]] - values[-FD_STEPS[1]]) / (2.0 * FD_STEPS[1])
    tolerance = 1e-3
    negative = slope_128 < -tolerance and slope_256 < -tolerance
    positive = slope_128 > tolerance and slope_256 > tolerance
    return {
        "direction_norm": _vector_norm(direction),
        "full_step_f": values[1.0],
        "full_step_delta_f": values[1.0] - float(current_f),
        "slope_h128": slope_128,
        "slope_h256": slope_256,
        "slope_mean": 0.5 * (slope_128 + slope_256),
        "slope_negative": negative,
        "slope_positive": positive,
        "slope_sign_agrees": negative or positive,
        "slope_relative_difference": abs(slope_128 - slope_256)
        / max(abs(slope_128), abs(slope_256), 1e-30),
    }


def _plot(individual: pd.DataFrame, prefixes: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pivot = individual.pivot(index="draw", columns="direction", values="slope_mean")
    axes[0, 0].scatter(pivot["adam"], pivot["sgd"], s=34)
    lo = float(min(pivot.min().min(), 0.0))
    hi = float(max(pivot.max().max(), 0.0))
    axes[0, 0].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].axvline(0.0, color="black", linewidth=1)
    axes[0, 0].set(xlabel="Adam true-F slope", ylabel="matched-norm -g slope", title="Same P4 draw: Adam vs -g")

    for direction, frame in individual.groupby("direction"):
        axes[0, 1].plot(frame["draw"], frame["slope_mean"], marker="o", label=direction)
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set(xlabel="independent P4 draw", ylabel="true-F slope", title="Per-draw directional variability")
    axes[0, 1].legend()

    for (aggregation, direction), frame in prefixes.groupby(["aggregation", "direction"]):
        axes[1, 0].plot(frame["prefix"], frame["slope_mean"], marker="o", label=f"{aggregation}:{direction}")
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set(xlabel="number of accumulated P4 draws", ylabel="true-F slope", title="Cumulative mean direction")
    axes[1, 0].legend(fontsize=8)

    negative = individual.groupby("direction")["slope_negative"].mean()
    axes[1, 1].bar(negative.index, negative.values, color=["#4c78a8", "#59a14f"])
    axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(ylim=(0, 1), ylabel="fraction negative at both FD scales", title="Usable-direction rate")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-position", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--beta", type=float, default=EXPECTED_BETA)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.draws != 16:
        raise ValueError("the frozen diagnostic requires exactly 16 draws")
    if args.beta != EXPECTED_BETA:
        raise ValueError(f"beta must remain {EXPECTED_BETA}")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output_dir / "executed_source.py")
    device = torch.device(args.device)
    checkpoint_sha = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_sha}")
    print(
        f"[direction-i2] start protocol={PROTOCOL_ID} device={device} dtype=float32 checkpoint={args.checkpoint} "
        f"draws={args.draws} pairs={args.pairs} beta={args.beta:.12g} epsilon={args.epsilon:g} "
        f"clip={args.gradient_clip:g} lr={args.lr:g} output={args.output_dir} source_sha={_source_sha()}",
        flush=True,
    )
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=args.pairs, batch_size=16384)
    bank = pd.read_csv(STATE_BANK)
    selected = bank.loc[bank["state_position"].eq(args.state_position)]
    if len(selected) != 1:
        raise ValueError("state selection is not unique")
    source_index = int(selected.iloc[0]["source_weight_index"])
    weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
    if sha256_tensor(z) != EXPECTED_Z:
        raise RuntimeError(f"z hash mismatch: {sha256_tensor(z)}")
    record = run.records.iloc[source_index].to_dict()
    record["source_weight_index"] = source_index
    active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
    named = dict(run.vae.named_parameters())
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in active_names]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for name in active_names:
            named[name].copy_(saved["active_model_state"][name].to(device=device, dtype=named[name].dtype))
    optimizer = torch.optim.Adam(active, lr=args.lr)
    optimizer.load_state_dict(saved["optimizer_state"])
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    base = [parameter.detach().clone() for parameter in active]

    print("[direction-i2] stage=current_dense_metric", flush=True)
    current, hessian, metric, eig_m, burg_gradient = _evaluate(
        run=run,
        z=z,
        record=record,
        epsilon=args.epsilon,
        beta=args.beta,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    current_f = float(current["true_dense_f"])
    if abs(current_f - EXPECTED_STATE_F) > 1e-4:
        raise RuntimeError(f"state F mismatch: {current_f} versus {EXPECTED_STATE_F}")
    del hessian, metric, eig_m
    print(f"[direction-i2] current F={current_f:.9g} task={current['task_loss']:.7g}", flush=True)

    sum_raw = [torch.zeros_like(value) for value in base]
    sum_clipped = [torch.zeros_like(value) for value in base]
    individual_rows: list[dict[str, float | int | bool | str]] = []
    prefix_rows: list[dict[str, float | int | bool | str]] = []
    draw_rows: list[dict[str, float | int]] = []
    for draw in range(args.draws):
        draw_started = time.perf_counter()
        raw, clipped, stats = _gradient_draw(
            run=run,
            cfg=cfg,
            z=z,
            record=record,
            active=active,
            burg_gradient=burg_gradient,
            beta=args.beta,
            draw=draw,
            pairs=args.pairs,
            clip_norm=args.gradient_clip,
        )
        with torch.no_grad():
            for target, value in zip(sum_raw, raw, strict=True):
                target.add_(value)
            for target, value in zip(sum_clipped, clipped, strict=True):
                target.add_(value)
        adam_direction = _provisional_adam_direction(
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            active=active,
            base=base,
            gradient=clipped,
        )
        adam_norm = _vector_norm(adam_direction)
        sgd_direction = _scaled_negative_gradient(clipped, target_norm=adam_norm)
        for label, direction in (("adam", adam_direction), ("sgd", sgd_direction)):
            profile = _direction_profile(
                run=run,
                z=z,
                record=record,
                active=active,
                base=base,
                direction=direction,
                current_f=current_f,
                beta=args.beta,
                epsilon=args.epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            individual_rows.append({"draw": draw, "direction": label, **profile})
        draw_rows.append({"draw": draw, **stats, "adam_direction_norm": adam_norm})

        prefix = draw + 1
        if prefix in PREFIXES:
            mean_raw = [value.detach().clone().div_(float(prefix)) for value in sum_raw]
            mean_clipped = [value.detach().clone().div_(float(prefix)) for value in sum_clipped]
            for aggregation, mean_value in (("raw_mean", mean_raw), ("clipped_mean", mean_clipped)):
                prepared, aggregate_norm, aggregate_factor = _clip_vector(mean_value, args.gradient_clip)
                adam_mean = _provisional_adam_direction(
                    optimizer=optimizer,
                    optimizer_state=optimizer_state,
                    active=active,
                    base=base,
                    gradient=prepared,
                )
                mean_adam_norm = _vector_norm(adam_mean)
                sgd_mean = _scaled_negative_gradient(prepared, target_norm=mean_adam_norm)
                for label, direction in (("adam", adam_mean), ("sgd", sgd_mean)):
                    profile = _direction_profile(
                        run=run,
                        z=z,
                        record=record,
                        active=active,
                        base=base,
                        direction=direction,
                        current_f=current_f,
                        beta=args.beta,
                        epsilon=args.epsilon,
                        hessian_chunk_size=args.hessian_chunk_size,
                    )
                    prefix_rows.append(
                        {
                            "prefix": prefix,
                            "aggregation": aggregation,
                            "direction": label,
                            "aggregate_preclip_norm": aggregate_norm,
                            "aggregate_clip_factor": aggregate_factor,
                            **profile,
                        }
                    )
            del mean_raw, mean_clipped
        del raw, clipped, adam_direction, sgd_direction
        print(
            f"[direction-i2] draw={draw + 1}/{args.draws} raw_norm={stats['raw_gradient_norm']:.4g} "
            f"adam_slope={individual_rows[-2]['slope_mean']:+.5g} sgd_slope={individual_rows[-1]['slope_mean']:+.5g} "
            f"sec={time.perf_counter() - draw_started:.1f} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    individual = pd.DataFrame(individual_rows)
    prefixes = pd.DataFrame(prefix_rows)
    draws = pd.DataFrame(draw_rows)
    individual.to_csv(args.output_dir / "individual_direction_slopes.csv", index=False)
    prefixes.to_csv(args.output_dir / "cumulative_mean_direction_slopes.csv", index=False)
    draws.to_csv(args.output_dir / "gradient_draw_diagnostics.csv", index=False)

    pivot = individual.pivot(index="draw", columns="direction", values="slope_negative")
    adam_negative = float(pivot["adam"].mean())
    sgd_negative = float(pivot["sgd"].mean())
    paired_sgd_only = int(((pivot["sgd"] == True) & (pivot["adam"] == False)).sum())  # noqa: E712
    paired_adam_only = int(((pivot["adam"] == True) & (pivot["sgd"] == False)).sum())  # noqa: E712

    k16 = prefixes.loc[prefixes["prefix"].eq(16)].set_index(["aggregation", "direction"])
    k16_raw_sgd_negative = bool(k16.loc[("raw_mean", "sgd"), "slope_negative"])
    k16_raw_adam_negative = bool(k16.loc[("raw_mean", "adam"), "slope_negative"])
    k16_clipped_sgd_negative = bool(k16.loc[("clipped_mean", "sgd"), "slope_negative"])
    k16_clipped_adam_negative = bool(k16.loc[("clipped_mean", "adam"), "slope_negative"])
    summary = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": _source_sha(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "base_vae_checkpoint_sha256": checkpoint_sha,
        "z_sha256": sha256_tensor(z),
        "state_proposal": int(saved["proposal"]),
        "current_f": current_f,
        "draws": args.draws,
        "pairs_per_draw": args.pairs,
        "individual_adam_negative_fraction": adam_negative,
        "individual_sgd_negative_fraction": sgd_negative,
        "paired_sgd_negative_adam_nonnegative": paired_sgd_only,
        "paired_adam_negative_sgd_nonnegative": paired_adam_only,
        "k16_raw_mean_sgd_negative": k16_raw_sgd_negative,
        "k16_raw_mean_adam_negative": k16_raw_adam_negative,
        "k16_clipped_mean_sgd_negative": k16_clipped_sgd_negative,
        "k16_clipped_mean_adam_negative": k16_clipped_adam_negative,
        "k16_slopes": {
            f"{aggregation}_{direction}": float(k16.loc[(aggregation, direction), "slope_mean"])
            for aggregation in ("raw_mean", "clipped_mean")
            for direction in ("adam", "sgd")
        },
        "adam_history_signature": bool(
            sgd_negative >= adam_negative + 0.25
            or (k16_raw_sgd_negative and not k16_raw_adam_negative)
        ),
        "finite_probe_variance_signature": bool(
            sgd_negative < 0.8 and (k16_raw_sgd_negative or k16_clipped_sgd_negative)
        ),
        "stopped_gradient_bias_signature": bool(
            not k16_raw_sgd_negative and not k16_clipped_sgd_negative
        ),
        "elapsed_sec": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    config = {
        "resolved_args": vars(args) | {"checkpoint": str(args.checkpoint), "output_dir": str(args.output_dir)},
        "protocol_id": PROTOCOL_ID,
        "prefixes": list(PREFIXES),
        "fd_steps": list(FD_STEPS),
        "same_norm_rule": "each SGD direction is scaled to the paired Adam displacement norm",
        "optimizer_state_rule": "every direction uses the same frozen proposal-40 Adam state; no diagnostic is committed",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    _plot(individual, prefixes, args.output_dir / "direction_source_diagnostic.png")
    print(f"[direction-i2] done summary={json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
