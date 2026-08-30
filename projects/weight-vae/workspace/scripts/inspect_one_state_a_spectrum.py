from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import decode_weights, encode_weights
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _hessian_via_batched_hvp,
    _latent_hvp,
    _probe_like,
    _task_loss_from_flat,
    _task_set_for_record,
)
from scripts.audit_variant_a_estimator_stability import DEFAULT_RUN_DIR, _load_run, _probe_cfg, sha256_file, sha256_tensor


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_optimization_smoke_h2048"
)
DEFAULT_FINAL = SMOKE_ROOT / "production_200steps/final_active_checkpoint.pt"
DEFAULT_OUTPUT = SMOKE_ROOT / "spectrum_initial_vs_step200"
EXPECTED_BASELINE_SHA256 = "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"


def spectral_summary(label: str, eig_h: np.ndarray, task_loss: float, elapsed: float) -> dict[str, float | str]:
    abs_h = np.abs(eig_h)
    eig_m = eig_h**2
    a = (eig_m - 1.0) ** 2
    pressure_h = np.abs(4.0 * eig_h * (eig_m - 1.0))
    q = np.quantile(eig_m, [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    aq = np.quantile(abs_h, [0.01, 0.10, 0.50, 0.90, 0.99])
    total_a = max(float(a.sum()), 1e-30)
    return {
        "label": label,
        "task_loss": task_loss,
        "hessian_elapsed_sec": elapsed,
        "exact_a_per_dim": float(a.mean()),
        "trace_m_per_dim": float(eig_m.mean()),
        "h_abs_mean": float(abs_h.mean()),
        "h_abs_p01": float(aq[0]),
        "h_abs_p10": float(aq[1]),
        "h_abs_p50": float(aq[2]),
        "h_abs_p90": float(aq[3]),
        "h_abs_p99": float(aq[4]),
        "m_min": float(q[0]),
        "m_p01": float(q[1]),
        "m_p10": float(q[2]),
        "m_p25": float(q[3]),
        "m_p50": float(q[4]),
        "m_p75": float(q[5]),
        "m_p90": float(q[6]),
        "m_p95": float(q[7]),
        "m_p99": float(q[8]),
        "m_max": float(q[9]),
        "m_lt_1e_4_fraction": float(np.mean(eig_m < 1e-4)),
        "m_lt_0p01_fraction": float(np.mean(eig_m < 0.01)),
        "m_lt_0p1_fraction": float(np.mean(eig_m < 0.1)),
        "m_lt_0p5_fraction": float(np.mean(eig_m < 0.5)),
        "m_near_1_10pct_fraction": float(np.mean(np.abs(eig_m - 1.0) <= 0.1)),
        "m_gt_1_fraction": float(np.mean(eig_m > 1.0)),
        "m_gt_2_fraction": float(np.mean(eig_m > 2.0)),
        "a_from_m_lt_0p1_share": float(a[eig_m < 0.1].sum() / total_a),
        "a_from_m_gt_1_share": float(a[eig_m > 1.0].sum() / total_a),
        "a_top1_share": float(np.sort(a)[-1:].sum() / total_a),
        "a_top10_share": float(np.sort(a)[-10:].sum() / total_a),
        "pressure_h_mean": float(pressure_h.mean()),
        "pressure_h_near_zero_fraction": float(np.mean(pressure_h < 1e-3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-position", type=int, default=2)
    parser.add_argument("--final-checkpoint", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--final-label", default="step200")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    baseline_hash = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    if baseline_hash != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"baseline checkpoint mismatch: {baseline_hash}")
    print(
        f"[one-state-spectrum] start device={device} dtype=float32 state_position={args.state_position} "
        f"chunk={args.hessian_chunk_size} baseline={DEFAULT_RUN_DIR} final={args.final_checkpoint} "
        f"output={args.output_dir}",
        flush=True,
    )
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    bank = pd.read_csv(SMOKE_ROOT.parent / "a_fixed_state_probe_variance_h2048/state_bank.csv")
    selected = bank.loc[bank["state_position"].eq(args.state_position)]
    if len(selected) != 1:
        raise ValueError("state position must identify exactly one state")
    source_index = int(selected.iloc[0]["source_weight_index"])
    record = run.records.iloc[source_index].to_dict()
    record["source_weight_index"] = source_index
    weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    print(
        f"[one-state-spectrum] loaded source={source_index} task={record.get('task_name')} tau={tau:.8g} "
        f"z_sha256={sha256_tensor(z)} full_ce_count={len(task_set.train_labels)}",
        flush=True,
    )

    spectra: dict[str, np.ndarray] = {}
    hessians: dict[str, torch.Tensor] = {}
    summaries: list[dict[str, float | str]] = []
    named_parameters = dict(run.vae.named_parameters())
    final_payload = torch.load(args.final_checkpoint, map_location="cpu", weights_only=False)
    final_active = final_payload["active_model_state"]
    unknown = sorted(set(final_active) - set(named_parameters))
    if unknown:
        raise RuntimeError(f"unknown final active parameters: {unknown}")
    initial_active = {name: named_parameters[name].detach().cpu().clone() for name in final_active}
    displacement2 = sum(float((final_active[name].float() - initial_active[name].float()).square().sum()) for name in final_active)
    initial2 = sum(float(initial_active[name].float().square().sum()) for name in final_active)
    probe_cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)

    for label in ("initial", str(args.final_label)):
        if label == str(args.final_label):
            with torch.no_grad():
                for name, value in final_active.items():
                    named_parameters[name].copy_(value.to(device=device, dtype=named_parameters[name].dtype))

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
        print(f"[one-state-spectrum] hessian_start label={label} task_loss={task_loss:.8g}", flush=True)
        started = time.perf_counter()
        hessian_raw = _hessian_via_batched_hvp(loss_from_z, z, chunk_size=args.hessian_chunk_size)
        symmetry_rel = float(
            ((hessian_raw - hessian_raw.T).double().norm() / hessian_raw.double().norm().clamp_min(1e-30)).cpu()
        )
        hessian = 0.5 * (hessian_raw + hessian_raw.T)
        elapsed = time.perf_counter() - started
        eig_h = torch.linalg.eigvalsh(hessian.detach().double()).cpu().numpy()
        if not np.isfinite(eig_h).all():
            raise RuntimeError(f"non-finite Hessian spectrum for {label}")
        spectra[label] = eig_h
        hessians[label] = hessian.detach().cpu().float()
        summary = spectral_summary(label, eig_h, task_loss, elapsed)
        validation_rows: list[dict[str, float]] = []
        for probe_index in range(4):
            generator = torch.Generator(device="cpu").manual_seed(930_000 + probe_index)
            probe = _probe_like(z, generator=generator, scale=1.0)
            production = _latent_hvp(
                probe_cfg,
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
            relative_error = float(((dense - production).double().norm() / denominator).cpu())
            cosine = float(
                ((dense.double() * production.double()).sum() / (dense.double().norm() * denominator).clamp_min(1e-30)).cpu()
            )
            validation_rows.append({"relative_error": relative_error, "cosine": cosine})
        matrix64 = hessian.detach().double()
        metric = matrix64.T @ matrix64
        dim = int(metric.shape[0])
        a_closure = float(((metric @ metric).trace() / dim - 2.0 * metric.trace() / dim + 1.0).cpu())
        singular_m = torch.linalg.svdvals(matrix64).square().cpu().numpy()
        summary.update(
            {
                "hessian_symmetry_rel": symmetry_rel,
                "stopped_hvp_relative_error_max": max(row["relative_error"] for row in validation_rows),
                "stopped_hvp_relative_error_mean": float(np.mean([row["relative_error"] for row in validation_rows])),
                "stopped_hvp_cosine_min": min(row["cosine"] for row in validation_rows),
                "a_trace_closure": a_closure,
                "a_trace_closure_abs_error": abs(a_closure - float(summary["exact_a_per_dim"])),
                "m_svd_vs_eigh_max_abs_error": float(
                    np.max(np.abs(np.sort(singular_m) - np.sort(eig_h**2)))
                ),
            }
        )
        summaries.append(summary)
        print(
            f"[one-state-spectrum] hessian_done label={label} sec={elapsed:.1f} "
            f"exact_A={summary['exact_a_per_dim']:.8g} trace_M={summary['trace_m_per_dim']:.8g} "
            f"M_p50={summary['m_p50']:.4g} M_p99={summary['m_p99']:.4g} M_max={summary['m_max']:.4g} "
            f"M_lt_0.1={summary['m_lt_0p1_fraction']:.3f} symmetry={symmetry_rel:.3g} "
            f"hvp_rel_max={summary['stopped_hvp_relative_error_max']:.3g}",
            flush=True,
        )

    rows: list[dict[str, float | int | str]] = []
    for label, eig_h in spectra.items():
        eig_m = eig_h**2
        order = np.argsort(eig_m)
        for rank, index in enumerate(order):
            rows.append(
                {
                    "label": label,
                    "ascending_m_rank": rank,
                    "h_eigenvalue": float(eig_h[index]),
                    "h_abs_eigenvalue": float(abs(eig_h[index])),
                    "m_eigenvalue": float(eig_m[index]),
                    "a_contribution": float((eig_m[index] - 1.0) ** 2),
                }
            )
    spectrum = pd.DataFrame(rows)
    summary_frame = pd.DataFrame(summaries)
    spectrum.to_csv(args.output_dir / "spectrum.csv", index=False)
    summary_frame.to_csv(args.output_dir / "spectral_summary.csv", index=False)
    torch.save(hessians, args.output_dir / "hessians.pt")

    initial_summary = summary_frame.set_index("label").loc["initial"]
    final_summary = summary_frame.set_index("label").loc[str(args.final_label)]
    comparison = {
        "baseline_checkpoint_sha256": baseline_hash,
        "final_active_checkpoint_sha256": sha256_file(args.final_checkpoint),
        "state_position": args.state_position,
        "source_weight_index": source_index,
        "task_name": str(record.get("task_name")),
        "tau": tau,
        "z_sha256": sha256_tensor(z),
        "full_ce_count": int(len(task_set.train_labels)),
        "active_parameter_count": int(sum(value.numel() for value in final_active.values())),
        "active_parameter_displacement_norm": float(displacement2**0.5),
        "active_parameter_relative_displacement": float((displacement2 / max(initial2, 1e-30)) ** 0.5),
        "exact_a_initial": float(initial_summary["exact_a_per_dim"]),
        "final_label": str(args.final_label),
        "exact_a_final": float(final_summary["exact_a_per_dim"]),
        "exact_a_ratio": float(final_summary["exact_a_per_dim"] / initial_summary["exact_a_per_dim"]),
        "trace_m_initial": float(initial_summary["trace_m_per_dim"]),
        "trace_m_final": float(final_summary["trace_m_per_dim"]),
        "m_lt_0p1_fraction_initial": float(initial_summary["m_lt_0p1_fraction"]),
        "m_lt_0p1_fraction_final": float(final_summary["m_lt_0p1_fraction"]),
        "m_near_1_10pct_fraction_initial": float(initial_summary["m_near_1_10pct_fraction"]),
        "m_near_1_10pct_fraction_final": float(final_summary["m_near_1_10pct_fraction"]),
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    colors = {"initial": "#2563eb", str(args.final_label): "#dc2626"}
    for label in ("initial", str(args.final_label)):
        values = np.sort(spectra[label] ** 2)
        axes[0].plot(np.arange(1, len(values) + 1), np.maximum(values, 1e-16), label=label, color=colors[label])
        axes[1].plot(np.maximum(values, 1e-16), np.arange(1, len(values) + 1) / len(values), label=label, color=colors[label])
        contribution = (values - 1.0) ** 2
        axes[2].plot(
            np.arange(1, len(values) + 1),
            np.maximum(contribution, 1e-8),
            label=label,
            color=colors[label],
        )
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="ascending spectral rank", ylabel="eigenvalue of M = H^T H", title="Full latent metric spectrum")
    axes[1].axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="eigenvalue of M", ylabel="fraction of modes <= x", title="Empirical spectral CDF")
    axes[2].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set(xlabel="ascending M rank", ylabel="(lambda(M) - 1)^2", title="Per-mode A contribution (log)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    plot_path = args.output_dir / f"spectrum_initial_vs_{args.final_label}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"[one-state-spectrum] done comparison={json.dumps(comparison, sort_keys=True)}", flush=True)
    print(
        f"[one-state-spectrum] artifacts={args.output_dir / 'spectrum.csv'},"
        f"{args.output_dir / 'spectral_summary.csv'},"
        f"{args.output_dir / 'comparison.json'},"
        f"{plot_path},"
        f"{args.output_dir / 'hessians.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
