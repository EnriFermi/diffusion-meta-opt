from __future__ import annotations

import argparse
import json
import math
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
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import decode_weights, encode_weights
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _hessian_via_batched_hvp,
    _latent_hvp,
    _probe_like,
    _task_loss_from_flat,
    _task_set_for_record,
)
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _load_run,
    _probe_cfg,
    sha256_file,
    sha256_tensor,
    stable_uint63,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048"
)
STATE_BANK = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_fixed_state_probe_variance_h2048/state_bank.csv"
)
ACTIVE_PARAMETERS = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
)
A_ONLY_SPECTRUM = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_optimization_smoke_h2048/spectrum_initial_vs_step40/spectral_summary.csv"
)
EXPECTED_CHECKPOINT = "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
PROTOCOL_ID = "one_state_a_full_burg_h2048_v1"
A_ONLY_PROTOCOL_ID = "one_state_a_optimization_smoke_h2048_v1"


def _generators(protocol: str, *parts: object) -> tuple[torch.Generator, torch.Generator]:
    seeds = [stable_uint63(protocol, *parts, branch) for branch in (0, 1)]
    return tuple(torch.Generator(device="cpu").manual_seed(seed) for seed in seeds)  # type: ignore[return-value]


def _train_generators(step: int, pair: int) -> tuple[torch.Generator, torch.Generator]:
    # Match the existing A-only control exactly for the A component.
    return _generators(A_ONLY_PROTOCOL_ID, "train", step, pair)


def _calibration_generators(draw: int, pair: int) -> tuple[torch.Generator, torch.Generator]:
    return _generators(PROTOCOL_ID, "calibration", draw, pair)


def _validation_generator(step: int, probe: int) -> torch.Generator:
    seed = stable_uint63(PROTOCOL_ID, "operator_validation", step, probe)
    return torch.Generator(device="cpu").manual_seed(seed)


def _pair_losses(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    pair_index: int,
    probe_generator_1: torch.Generator,
    probe_generator_2: torch.Generator,
    burg_matrix_gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    task_set = _task_set_for_record(run.task_tensors, record)
    sample_key = int(record.get("source_weight_index", record.get("weight_index", 0)))
    tau = float(record.get("tau", 1.0))
    batch_1 = _batch_indices(
        task_set,
        batch_size=int(cfg.vae_precond_batch_size),
        step=10,
        sample_key=sample_key,
        pair_key=2 * int(pair_index),
    )
    batch_2 = _batch_indices(
        task_set,
        batch_size=int(cfg.vae_precond_batch_size),
        step=10,
        sample_key=sample_key,
        pair_key=2 * int(pair_index) + 1,
    )
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
    a_loss = (dot.square() - norm1 - norm2 + float(dim)) / float(dim)
    b_pseudo_loss = 0.5 * (
        torch.dot(h1, burg_matrix_gradient @ h1)
        + torch.dot(h2, burg_matrix_gradient @ h2)
    )
    return a_loss, b_pseudo_loss, {
        "h_norm2_per_dim": float((0.5 * (norm1 + norm2) / float(dim)).detach().cpu()),
        "h_dot2_per_dim": float((dot.square() / float(dim)).detach().cpu()),
        "b_pseudo_loss": float(b_pseudo_loss.detach().cpu()),
    }


def _gradient_norms(
    gradients_a: tuple[torch.Tensor | None, ...],
    gradients_b: tuple[torch.Tensor | None, ...],
    *,
    beta: float,
) -> dict[str, float]:
    a2 = torch.zeros((), device=next(value for value in gradients_a if value is not None).device, dtype=torch.float64)
    b2 = torch.zeros_like(a2)
    dot = torch.zeros_like(a2)
    total2 = torch.zeros_like(a2)
    for grad_a, grad_b in zip(gradients_a, gradients_b, strict=True):
        if grad_a is None and grad_b is None:
            continue
        if grad_a is None:
            grad_a = torch.zeros_like(grad_b)
        if grad_b is None:
            grad_b = torch.zeros_like(grad_a)
        a64 = grad_a.detach().double()
        b64 = grad_b.detach().double()
        total64 = a64 + float(beta) * b64
        a2 += a64.square().sum()
        b2 += b64.square().sum()
        dot += (a64 * b64).sum()
        total2 += total64.square().sum()
    norm_a = float(a2.sqrt().cpu())
    norm_b = float(b2.sqrt().cpu())
    norm_total = float(total2.sqrt().cpu())
    cosine = float((dot / (a2.sqrt() * b2.sqrt()).clamp_min(1e-30)).cpu())
    return {
        "grad_a_norm": norm_a,
        "grad_b_norm": norm_b,
        "grad_beta_b_norm": abs(float(beta)) * norm_b,
        "grad_total_norm": norm_total,
        "grad_a_b_cosine": cosine,
    }


def _materialize_metric(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    epsilon: float,
    hessian_chunk_size: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))

    def loss_from_z(latent: torch.Tensor) -> torch.Tensor:
        decoded = decode_weights(run.vae, run.normalizer, latent.reshape(1, -1)).squeeze(0)
        return _task_loss_from_flat(
            decoded,
            task_set=task_set,
            spec=run.spec,
            tau=tau,
            batch_indices=None,
        )

    with torch.no_grad():
        task_loss = float(loss_from_z(z).detach().cpu())
    started = time.perf_counter()
    hessian = _hessian_via_batched_hvp(loss_from_z, z, chunk_size=hessian_chunk_size).detach()
    hessian_sec = time.perf_counter() - started
    h64 = hessian.double()
    symmetry_rel = float(((h64 - h64.T).norm() / h64.norm().clamp_min(1e-30)).cpu())
    metric = h64 @ h64.T
    metric = 0.5 * (metric + metric.T)
    eig_m, eigvec_m = torch.linalg.eigh(metric)
    eig_m = eig_m.clamp_min(0.0)
    dim = int(eig_m.numel())
    r_eig = (eig_m + float(epsilon)) / (1.0 + float(epsilon))
    g_eig = (1.0 - r_eig.reciprocal()) / (float(dim) * (1.0 + float(epsilon)))
    burg_gradient = ((eigvec_m * g_eig.unsqueeze(0)) @ eigvec_m.T).to(dtype=z.dtype).detach()
    a_contribution = (eig_m - 1.0).square()
    q = torch.quantile(
        eig_m,
        torch.tensor(
            [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
            device=eig_m.device,
            dtype=eig_m.dtype,
        ),
    )
    burg_trace = r_eig.mean()
    burg_neg_logdet = -r_eig.log().mean()
    full_burg = burg_trace + burg_neg_logdet - 1.0
    row = {
        "task_loss": task_loss,
        "hessian_sec": hessian_sec,
        "hessian_symmetry_rel": symmetry_rel,
        "exact_a_per_dim": float(a_contribution.mean().cpu()),
        "a_constant_term": 1.0,
        "a_linear_trace_term": float((-2.0 * eig_m.mean()).cpu()),
        "a_quartic_term": float(eig_m.square().mean().cpu()),
        "trace_m_per_dim": float(eig_m.mean().cpu()),
        "damped_full_burg_per_dim": float(full_burg.cpu()),
        "burg_trace_r_term": float(burg_trace.cpu()),
        "burg_neg_logdet_r_term": float(burg_neg_logdet.cpu()),
        "logdet_r_per_dim": float(r_eig.log().mean().cpu()),
        "m_min": float(q[0].cpu()),
        "m_p01": float(q[1].cpu()),
        "m_p10": float(q[2].cpu()),
        "m_p25": float(q[3].cpu()),
        "m_p50": float(q[4].cpu()),
        "m_p75": float(q[5].cpu()),
        "m_p90": float(q[6].cpu()),
        "m_p95": float(q[7].cpu()),
        "m_p99": float(q[8].cpu()),
        "m_max": float(q[9].cpu()),
        "m_lt_1e_4_fraction": float((eig_m < 1e-4).double().mean().cpu()),
        "m_lt_0p01_fraction": float((eig_m < 0.01).double().mean().cpu()),
        "m_lt_0p1_fraction": float((eig_m < 0.1).double().mean().cpu()),
        "m_lt_0p5_fraction": float((eig_m < 0.5).double().mean().cpu()),
        "m_near_1_10pct_fraction": float(((eig_m - 1.0).abs() <= 0.1).double().mean().cpu()),
        "m_gt_1_fraction": float((eig_m > 1.0).double().mean().cpu()),
        "m_gt_2_fraction": float((eig_m > 2.0).double().mean().cpu()),
        "a_from_m_lt_0p1_share": float(
            (a_contribution[eig_m < 0.1].sum() / a_contribution.sum().clamp_min(1e-30)).cpu()
        ),
        "a_from_m_gt_1_share": float(
            (a_contribution[eig_m > 1.0].sum() / a_contribution.sum().clamp_min(1e-30)).cpu()
        ),
        "burg_matrix_gradient_norm": float(g_eig.norm().cpu()),
        "burg_matrix_gradient_eig_min": float(g_eig.min().cpu()),
        "burg_matrix_gradient_eig_p50": float(g_eig.median().cpu()),
        "burg_matrix_gradient_eig_max": float(g_eig.max().cpu()),
    }
    return row, hessian, metric, eig_m, burg_gradient


def _operator_validation(
    *,
    run: Any,
    cfg: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    hessian: torch.Tensor,
    step: int,
    probes: int,
) -> list[dict[str, float | int]]:
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    rows: list[dict[str, float | int]] = []
    for probe_index in range(probes):
        probe = _probe_like(z, generator=_validation_generator(step, probe_index), scale=1.0)
        production = _latent_hvp(
            cfg,
            run.vae,
            run.normalizer,
            z,
            probe,
            task_set=task_set,
            spec=run.spec,
            tau=tau,
            batch_indices=None,
        ).detach()
        dense = hessian @ probe
        denominator = production.double().norm().clamp_min(1e-30)
        rows.append(
            {
                "step": step,
                "probe": probe_index,
                "relative_error": float(((dense - production).double().norm() / denominator).cpu()),
                "cosine": float(
                    (
                        (dense.double() * production.double()).sum()
                        / (dense.double().norm() * denominator).clamp_min(1e-30)
                    ).cpu()
                ),
            }
        )
    return rows


def _calibrate_beta(
    *,
    run: Any,
    cfg: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: list[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    draws: int,
    pairs: int,
) -> dict[str, float | int]:
    accumulated_a = [torch.zeros_like(parameter) for parameter in active]
    accumulated_b = [torch.zeros_like(parameter) for parameter in active]
    atomic_count = int(draws) * int(pairs)
    started = time.perf_counter()
    for draw in range(draws):
        for pair in range(pairs):
            g1, g2 = _calibration_generators(draw, pair)
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
            gradients_a = torch.autograd.grad(a_loss, active, retain_graph=True, allow_unused=True)
            gradients_b = torch.autograd.grad(b_loss, active, retain_graph=False, allow_unused=True)
            with torch.no_grad():
                for index, (grad_a, grad_b) in enumerate(zip(gradients_a, gradients_b, strict=True)):
                    if grad_a is not None:
                        accumulated_a[index].add_(grad_a, alpha=1.0 / float(atomic_count))
                    if grad_b is not None:
                        accumulated_b[index].add_(grad_b, alpha=1.0 / float(atomic_count))
            del a_loss, b_loss, gradients_a, gradients_b
        print(
            f"[A+fullB] calibration draw={draw + 1}/{draws} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    stats = _gradient_norms(tuple(accumulated_a), tuple(accumulated_b), beta=1.0)
    if not math.isfinite(stats["grad_b_norm"]) or stats["grad_b_norm"] <= 0.0:
        raise RuntimeError(f"invalid calibration B gradient norm: {stats['grad_b_norm']}")
    beta = stats["grad_a_norm"] / stats["grad_b_norm"]
    return {
        "draws": draws,
        "pairs_per_draw": pairs,
        "atomic_count": atomic_count,
        "grad_a_mean_norm": stats["grad_a_norm"],
        "grad_b_mean_norm": stats["grad_b_norm"],
        "grad_a_b_cosine": stats["grad_a_b_cosine"],
        "beta": beta,
        "elapsed_sec": time.perf_counter() - started,
    }


def _plot_results(dense: pd.DataFrame, updates: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[0, 0].plot(dense["step"], dense["exact_a_per_dim"], label="exact A")
    axes[0, 0].plot(dense["step"], dense["damped_full_burg_per_dim"], label="damped full B")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="optimizer step", ylabel="per-dimension objective", title="Exact objectives")
    axes[0, 0].legend()

    axes[0, 1].plot(dense["step"], dense["m_lt_1e_4_fraction"], label="fraction M < 1e-4")
    axes[0, 1].plot(dense["step"], dense["m_lt_0p01_fraction"], label="fraction M < 0.01")
    axes[0, 1].plot(dense["step"], dense["m_lt_0p1_fraction"], label="fraction M < 0.1")
    trace_axis = axes[0, 1].twinx()
    trace_axis.plot(dense["step"], dense["trace_m_per_dim"], color="black", alpha=0.65, label="tr(M) / m")
    axes[0, 1].set(xlabel="optimizer step", ylabel="fraction", title="Collapse indicators")
    trace_axis.set_ylabel("tr(M) / m")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    trace_handles, trace_labels = trace_axis.get_legend_handles_labels()
    axes[0, 1].legend(handles + trace_handles, labels + trace_labels)

    for column in ("m_p50", "m_p90", "m_p99", "m_max"):
        axes[0, 2].plot(dense["step"], np.maximum(dense[column], 1e-16), label=column)
    axes[0, 2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set(xlabel="optimizer step", ylabel="eigenvalue of M", title="Spectrum movement")
    axes[0, 2].legend()

    axes[1, 0].plot(updates["step"], updates["grad_a_norm"], label="||g_A||")
    axes[1, 0].plot(updates["step"], updates["grad_beta_b_norm"], label="||beta g_B||")
    axes[1, 0].plot(updates["step"], updates["grad_total_norm"], label="||g_total||")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(xlabel="completed optimizer step", ylabel="pre-clip gradient norm", title="Component gradients")
    axes[1, 0].legend()

    axes[1, 1].plot(updates["step"], updates["grad_a_b_cosine"], label="cos(g_A, g_B)")
    clip_axis = axes[1, 1].twinx()
    clip_axis.plot(updates["step"], updates["clip_factor"], color="#dc2626", alpha=0.75, label="clip factor")
    clip_axis.set_yscale("log")
    clip_axis.set_ylabel("clip factor", color="#dc2626")
    axes[1, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 1].set(xlabel="completed optimizer step", ylabel="gradient cosine", title="Conflict and clipping")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    clip_handles, clip_labels = clip_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + clip_handles, labels + clip_labels)

    axes[1, 2].plot(dense["step"], dense["task_loss"], label="task CE")
    step_axis = axes[1, 2].twinx()
    step_axis.plot(updates["step"], updates["parameter_step_norm"], color="#dc2626", label="parameter step")
    axes[1, 2].set(xlabel="optimizer step", ylabel="task CE", title="Context and realized update")
    step_axis.set_ylabel("parameter step norm", color="#dc2626")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-position", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--calibration-draws", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "production_100steps")
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_hash}")
    print(
        f"[A+fullB] start protocol={PROTOCOL_ID} device={device} dtype=float32 "
        f"state_position={args.state_position} steps={args.steps} optimizer=Adam lr={args.lr:g} "
        f"pairs={args.pairs} full_ce_batch=16384 epsilon={args.epsilon:g} "
        f"calibration_draws={args.calibration_draws} output={args.output_dir}",
        flush=True,
    )
    print(f"[A+fullB] loading checkpoint={DEFAULT_RUN_DIR / 'vae_checkpoint.pt'}", flush=True)
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=args.pairs, batch_size=16384)
    bank = pd.read_csv(STATE_BANK)
    selected = bank.loc[bank["state_position"].eq(args.state_position)]
    if len(selected) != 1:
        raise ValueError(f"state_position {args.state_position} is not unique")
    state_meta = selected.iloc[0].to_dict()
    source_index = int(state_meta["source_weight_index"])
    weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
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
    active_count = sum(parameter.numel() for parameter in active)
    optimizer = torch.optim.Adam(active, lr=args.lr)
    checkpoint_steps = {0, 10, 40, int(args.steps)}
    validation_steps = {0, 40, int(args.steps)}
    dense_rows: list[dict[str, float | int]] = []
    spectrum_rows: list[dict[str, float | int]] = []
    update_rows: list[dict[str, float | int]] = []
    operator_rows: list[dict[str, float | int]] = []
    matrix_checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    initial_active = {name: named[name].detach().cpu().clone() for name in active_names}
    torch.cuda.reset_peak_memory_stats(device)

    print(
        f"[A+fullB] loaded source_weight_index={source_index} task={record.get('task_name')} "
        f"latent_dim={z.numel()} z_sha256={sha256_tensor(z)} active_parameters={active_count}",
        flush=True,
    )

    print("[A+fullB] stage=dense_initial_metric", flush=True)
    initial_metric, initial_h, initial_m, initial_eig, initial_burg_gradient = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=args.epsilon,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    print(
        f"[A+fullB] initial exact_A={initial_metric['exact_a_per_dim']:.7g} "
        f"full_B={initial_metric['damped_full_burg_per_dim']:.7g} "
        f"trace_M={initial_metric['trace_m_per_dim']:.7g} "
        f"M<0.1={initial_metric['m_lt_0p1_fraction']:.4f}",
        flush=True,
    )
    print("[A+fullB] stage=gradient_calibration", flush=True)
    calibration = _calibrate_beta(
        run=run,
        cfg=cfg,
        z=z,
        record=record,
        active=active,
        burg_gradient=initial_burg_gradient,
        draws=args.calibration_draws,
        pairs=args.pairs,
    )
    beta = float(calibration["beta"])
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[A+fullB] calibrated beta={beta:.7g} ||mean_gA||={calibration['grad_a_mean_norm']:.7g} "
        f"||mean_gB||={calibration['grad_b_mean_norm']:.7g} "
        f"cos={calibration['grad_a_b_cosine']:.4f}",
        flush=True,
    )
    del initial_burg_gradient
    torch.cuda.empty_cache()

    def record_dense(
        step: int,
        metric_row: dict[str, float],
        hessian: torch.Tensor,
        metric: torch.Tensor,
        eig_m: torch.Tensor,
    ) -> None:
        row: dict[str, float | int] = {
            "step": step,
            **metric_row,
            "elapsed_sec": time.perf_counter() - started,
        }
        dense_rows.append(row)
        eig_cpu = eig_m.detach().cpu().numpy()
        for rank, eigenvalue in enumerate(eig_cpu):
            spectrum_rows.append(
                {
                    "step": step,
                    "ascending_rank": rank,
                    "m_eigenvalue": float(eigenvalue),
                    "a_contribution": float((eigenvalue - 1.0) ** 2),
                    "damped_r_eigenvalue": float((eigenvalue + args.epsilon) / (1.0 + args.epsilon)),
                }
            )
        if step in checkpoint_steps:
            matrix_checkpoints[step] = {
                "hessian": hessian.detach().cpu().float(),
                "metric_h_ht": metric.detach().cpu().float(),
                "metric_eigenvalues": eig_m.detach().cpu().double(),
            }
            torch.save(
                {
                    "step": step,
                    "active_model_state": {name: named[name].detach().cpu() for name in active_names},
                },
                args.output_dir / f"active_checkpoint_step{step}.pt",
            )
        if step in validation_steps:
            rows = _operator_validation(
                run=run,
                cfg=cfg,
                z=z,
                record=record,
                hessian=hessian,
                step=step,
                probes=2,
            )
            operator_rows.extend(rows)
            print(
                f"[A+fullB] operator_validation step={step} "
                f"max_rel_error={max(float(value['relative_error']) for value in rows):.3g} "
                f"min_cos={min(float(value['cosine']) for value in rows):.8f}",
                flush=True,
            )

    record_dense(0, initial_metric, initial_h, initial_m, initial_eig)
    del initial_metric, initial_h, initial_m, initial_eig
    print("[A+fullB] stage=optimization", flush=True)

    for completed_step in range(1, args.steps + 1):
        update_started = time.perf_counter()
        pre_step = completed_step - 1
        if pre_step == 0:
            metric_row = dense_rows[0]
            _, _, _, _, burg_gradient = _materialize_metric(
                run=run,
                z=z,
                record=record,
                epsilon=args.epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
            )
        else:
            metric_row, hessian, metric, eig_m, burg_gradient = _materialize_metric(
                run=run,
                z=z,
                record=record,
                epsilon=args.epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            record_dense(pre_step, metric_row, hessian, metric, eig_m)
            del hessian, metric, eig_m

        optimizer.zero_grad(set_to_none=True)
        a_losses: list[torch.Tensor] = []
        b_losses: list[torch.Tensor] = []
        pair_stats: list[dict[str, float]] = []
        for pair in range(args.pairs):
            generator_1, generator_2 = _train_generators(completed_step, pair)
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
            pair_stats.append(stats)
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
                parameter.grad = grad_a + beta * grad_b
        total_norm = float(torch.nn.utils.clip_grad_norm_(active, max_norm=args.gradient_clip).detach().cpu())
        if not math.isfinite(total_norm):
            raise RuntimeError(f"non-finite gradient at step {completed_step}: {total_norm}")
        clip_factor = min(1.0, float(args.gradient_clip) / max(total_norm, 1e-30))
        before_step = [parameter.detach().clone() for parameter in active]
        optimizer.step()
        with torch.no_grad():
            step_norm2 = sum(
                float((parameter.detach().double() - before.double()).square().sum().cpu())
                for parameter, before in zip(active, before_step, strict=True)
            )
            parameter_norm2 = sum(float(parameter.detach().double().square().sum().cpu()) for parameter in active)
        update_row: dict[str, float | int] = {
            "step": completed_step,
            "pre_step_metric_step": pre_step,
            "train_a": float(mean_a.detach().cpu()),
            "train_b_pseudo_loss": float(mean_b.detach().cpu()),
            "beta": beta,
            **gradient_stats,
            "grad_total_norm_clip_api": total_norm,
            "clip_factor": clip_factor,
            "parameter_step_norm": math.sqrt(step_norm2),
            "active_parameter_norm": math.sqrt(parameter_norm2),
            "parameter_relative_step": math.sqrt(step_norm2 / max(parameter_norm2, 1e-30)),
            "h_norm2_per_dim_mean": float(np.mean([value["h_norm2_per_dim"] for value in pair_stats])),
            "h_dot2_per_dim_mean": float(np.mean([value["h_dot2_per_dim"] for value in pair_stats])),
            "update_sec": time.perf_counter() - update_started,
            "elapsed_sec": time.perf_counter() - started,
        }
        update_rows.append(update_row)
        del a_losses, b_losses, mean_a, mean_b, gradients_a, gradients_b, burg_gradient, before_step
        if completed_step <= 2 or completed_step % 5 == 0:
            print(
                f"[A+fullB] step={completed_step}/{args.steps} A={update_row['train_a']:.6g} "
                f"B_pseudo={update_row['train_b_pseudo_loss']:.6g} "
                f"||gA||={update_row['grad_a_norm']:.4g} ||beta*gB||={update_row['grad_beta_b_norm']:.4g} "
                f"cos={update_row['grad_a_b_cosine']:.3f} clip={clip_factor:.3g} "
                f"sec={update_row['update_sec']:.2f} elapsed={update_row['elapsed_sec']:.1f}s",
                flush=True,
            )

    final_metric, final_h, final_m, final_eig, _final_burg_gradient = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=args.epsilon,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    record_dense(args.steps, final_metric, final_h, final_m, final_eig)
    del _final_burg_gradient

    dense_frame = pd.DataFrame(dense_rows).sort_values("step").reset_index(drop=True)
    spectrum_frame = pd.DataFrame(spectrum_rows).sort_values(["step", "ascending_rank"]).reset_index(drop=True)
    update_frame = pd.DataFrame(update_rows).sort_values("step").reset_index(drop=True)
    operator_frame = pd.DataFrame(operator_rows).sort_values(["step", "probe"]).reset_index(drop=True)
    dense_frame.to_csv(args.output_dir / "dense_trajectory.csv", index=False)
    spectrum_frame.to_csv(args.output_dir / "spectrum_trajectory.csv", index=False)
    update_frame.to_csv(args.output_dir / "update_diagnostics.csv", index=False)
    operator_frame.to_csv(args.output_dir / "operator_validation.csv", index=False)
    torch.save(matrix_checkpoints, args.output_dir / "matrix_checkpoints.pt")

    initial = dense_frame.loc[dense_frame["step"].eq(0)].iloc[0]
    final = dense_frame.loc[dense_frame["step"].eq(args.steps)].iloc[0]
    at_40 = dense_frame.loc[dense_frame["step"].eq(min(40, args.steps))].iloc[0]
    a_only = pd.read_csv(A_ONLY_SPECTRUM).set_index("label").loc["step40"]
    displacement2 = sum(
        float((named[name].detach().cpu().float() - initial_active[name].float()).square().sum())
        for name in active_names
    )
    initial_norm2 = sum(float(value.float().square().sum()) for value in initial_active.values())
    summary = {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "dtype": str(torch_dtype(run.cfg)),
        "state_position": args.state_position,
        "source_weight_index": source_index,
        "task_name": str(record.get("task_name")),
        "z_sha256": sha256_tensor(z),
        "active_parameter_count": active_count,
        "steps": args.steps,
        "pairs_per_update": args.pairs,
        "full_ce_batch_size": 16384,
        "optimizer": "Adam",
        "lr": args.lr,
        "gradient_clip_norm": args.gradient_clip,
        "damped_full_burg_epsilon": args.epsilon,
        "burg_beta": beta,
        "calibration": calibration,
        "initial_exact_a": float(initial["exact_a_per_dim"]),
        "final_exact_a": float(final["exact_a_per_dim"]),
        "initial_full_burg": float(initial["damped_full_burg_per_dim"]),
        "final_full_burg": float(final["damped_full_burg_per_dim"]),
        "initial_trace_m": float(initial["trace_m_per_dim"]),
        "step40_trace_m": float(at_40["trace_m_per_dim"]),
        "final_trace_m": float(final["trace_m_per_dim"]),
        "initial_m_lt_0p1_fraction": float(initial["m_lt_0p1_fraction"]),
        "step40_m_lt_0p1_fraction": float(at_40["m_lt_0p1_fraction"]),
        "final_m_lt_0p1_fraction": float(final["m_lt_0p1_fraction"]),
        "initial_m_near_1_fraction": float(initial["m_near_1_10pct_fraction"]),
        "step40_m_near_1_fraction": float(at_40["m_near_1_10pct_fraction"]),
        "final_m_near_1_fraction": float(final["m_near_1_10pct_fraction"]),
        "a_only_step40_trace_m": float(a_only["trace_m_per_dim"]),
        "a_only_step40_m_lt_0p1_fraction": float(a_only["m_lt_0p1_fraction"]),
        "a_only_step40_m_near_1_fraction": float(a_only["m_near_1_10pct_fraction"]),
        "better_than_a_only_at_step40": bool(
            float(at_40["trace_m_per_dim"]) > float(a_only["trace_m_per_dim"])
            and float(at_40["m_lt_0p1_fraction"]) < float(a_only["m_lt_0p1_fraction"])
        ),
        "resists_further_collapse_at_step40": bool(
            float(at_40["trace_m_per_dim"]) >= float(initial["trace_m_per_dim"])
            and float(at_40["m_lt_0p1_fraction"]) <= float(initial["m_lt_0p1_fraction"])
        ),
        "moves_low_spectrum_up_at_step40": bool(
            float(at_40["m_lt_0p1_fraction"]) < float(initial["m_lt_0p1_fraction"])
            and float(at_40["m_near_1_10pct_fraction"]) > float(initial["m_near_1_10pct_fraction"])
        ),
        "active_parameter_displacement_norm": math.sqrt(displacement2),
        "active_parameter_relative_displacement": math.sqrt(displacement2 / max(initial_norm2, 1e-30)),
        "mean_grad_a_b_cosine": float(update_frame["grad_a_b_cosine"].mean()),
        "min_clip_factor": float(update_frame["clip_factor"].min()),
        "mean_clip_factor": float(update_frame["clip_factor"].mean()),
        "elapsed_sec": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    config = {
        "resolved_args": vars(args) | {"output_dir": str(args.output_dir)},
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": checkpoint_hash,
        "a_only_common_random_number_protocol": A_ONLY_PROTOCOL_ID,
        "full_burg_definition": "mean(r - log(r) - 1), r=eig((H H^T + epsilon I)/(1+epsilon))",
        "full_burg_gradient_definition": "(I - R^{-1}) / (m * (1 + epsilon))",
        "burg_pseudo_loss_warning": "train_b_pseudo_loss is the stopped-gradient linearization, not the full Burg value",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    _plot_results(dense_frame, update_frame, args.output_dir / "full_burg_dynamics.png")
    print(f"[A+fullB] done summary={json.dumps(summary, sort_keys=True)}", flush=True)
    print(
        f"[A+fullB] artifacts={args.output_dir / 'config.json'},"
        f"{args.output_dir / 'calibration.json'},"
        f"{args.output_dir / 'dense_trajectory.csv'},"
        f"{args.output_dir / 'spectrum_trajectory.csv'},"
        f"{args.output_dir / 'update_diagnostics.csv'},"
        f"{args.output_dir / 'operator_validation.csv'},"
        f"{args.output_dir / 'matrix_checkpoints.pt'},"
        f"{args.output_dir / 'summary.json'},"
        f"{args.output_dir / 'full_burg_dynamics.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
