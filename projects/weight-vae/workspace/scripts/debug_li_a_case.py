from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import ExperimentConfig, torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    encode_weights,
    load_torch_cache,
    move_task_tensors,
    spec_from_payload,
    vae_loss,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import _load_task_tensors_for_pipeline
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    PreconditioningState,
    latent_hessian_diagnostic_row,
    preconditioning_regularizer,
)


def _load_cfg(run_dir: Path) -> ExperimentConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return ExperimentConfig(**payload["config"])


def _load_run(run_dir: Path, *, device: torch.device) -> tuple[ExperimentConfig, torch.Tensor, pd.DataFrame, Any, torch.nn.Module, WeightNormalizer]:
    cfg = _load_cfg(run_dir)
    dtype = torch_dtype(cfg)
    weight_payload = load_torch_cache(run_dir / "weight_pool.pt")
    vae_payload = load_torch_cache(run_dir / "vae_checkpoint.pt")
    if weight_payload is None or vae_payload is None:
        raise FileNotFoundError(f"{run_dir} must contain weight_pool.pt and vae_checkpoint.pt")
    weights = weight_payload["weights"].to(device=device, dtype=dtype)
    records = pd.read_csv(run_dir / "weight_pool_records.csv")
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=dtype)
    vae.load_state_dict(vae_payload["model_state"])
    vae.eval()
    return cfg, weights, records, spec, vae, normalizer


def _default_indices(run_dir: Path, limit: int) -> list[int]:
    diag_path = run_dir / "preconditioning_diagnostics.csv"
    if diag_path.is_file():
        diag = pd.read_csv(diag_path)
        if "weight_index" in diag.columns and "li_A_full_per_dim" in diag.columns:
            top = (
                diag.sort_values("li_A_full_per_dim", ascending=False)["weight_index"]
                .dropna()
                .astype("int64")
                .drop_duplicates()
                .head(limit)
                .tolist()
            )
            if top:
                return [int(v) for v in top]
    start_path = run_dir / "downstream_start_bank.csv"
    if start_path.is_file():
        start = pd.read_csv(start_path)
        if "source_weight_index" in start.columns:
            return [int(v) for v in start["source_weight_index"].dropna().astype("int64").head(limit).tolist()]
    return list(range(limit))


def _grad_stats(params: list[torch.nn.Parameter]) -> tuple[float, list[torch.Tensor | None]]:
    grads: list[torch.Tensor | None] = []
    norm2 = 0.0
    for param in params:
        grad = param.grad
        if grad is None:
            grads.append(None)
            continue
        detached = grad.detach().clone()
        grads.append(detached)
        norm2 += float(detached.float().square().sum().detach().cpu().item())
    return math.sqrt(norm2), grads


def _grad_dot(left: list[torch.Tensor | None], params: list[torch.nn.Parameter]) -> float:
    dot = 0.0
    for base_grad, param in zip(left, params, strict=True):
        if base_grad is None or param.grad is None:
            continue
        dot += float((base_grad.float() * param.grad.detach().float()).sum().detach().cpu().item())
    return dot


def estimator_probe(
    *,
    cfg: ExperimentConfig,
    weights: torch.Tensor,
    records: pd.DataFrame,
    spec: Any,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    task_tensors: dict[str, Any],
    indices: list[int],
    repeats: int,
    scopes: list[str],
    sample_count: int,
    pair_count: int,
    batch_size: int,
    coeff: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 88000)
    for scope in scopes:
        print(
            "[li_a_debug] estimator stage "
            f"scope={scope} repeats={int(repeats)} sample_count={int(sample_count)} "
            f"pair_count={int(pair_count)} batch_size={int(batch_size)}",
            flush=True,
        )
        probe_cfg = replace(
            cfg,
            vae_precond_loss_kind="li_a_hvp",
            vae_precond_coeff=float(coeff),
            vae_precond_burg_coeff=0.0,
            vae_precond_every=1,
            vae_precond_samples=int(sample_count),
            vae_precond_pairs=int(pair_count),
            vae_precond_batch_size=int(batch_size),
            vae_precond_estimator_scope=str(scope),
            vae_precond_hvp_mode="stopped_composite",
        )
        state = PreconditioningState()
        for repeat in range(int(repeats)):
            if repeat == 0 or (repeat + 1) % max(1, int(repeats) // 8) == 0 or repeat + 1 == int(repeats):
                print(f"[li_a_debug] estimator scope={scope} repeat={repeat + 1}/{int(repeats)}", flush=True)
            start = (repeat * int(sample_count)) % len(indices)
            chosen = [indices[(start + offset) % len(indices)] for offset in range(int(sample_count))]
            z = encode_weights(vae, normalizer, weights[chosen])
            recs = records.iloc[chosen].to_dict(orient="records")
            for rec, idx in zip(recs, chosen, strict=True):
                rec["source_weight_index"] = int(idx)
            loss, row = preconditioning_regularizer(
                probe_cfg,
                state=state,
                vae=vae,
                normalizer=normalizer,
                z_samples=z.detach(),
                records=recs,
                task_tensors=task_tensors,
                spec=spec,
                step=repeat + 1,
                generator=generator,
            )
            rows.append(
                {
                    "scope": scope,
                    "repeat": repeat,
                    "chosen_indices": json.dumps(chosen),
                    "loss": float(loss.detach().cpu().item()),
                    **row,
                }
            )
    return pd.DataFrame(rows)


def gradient_probe(
    *,
    cfg: ExperimentConfig,
    weights: torch.Tensor,
    records: pd.DataFrame,
    spec: Any,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    task_tensors: dict[str, Any],
    indices: list[int],
    scope: str,
    sample_count: int,
    pair_count: int,
    batch_size: int,
    coeff: float,
) -> pd.DataFrame:
    dtype = torch_dtype(cfg)
    device = weights.device
    probe_cfg = replace(
        cfg,
        vae_precond_loss_kind="li_a_hvp",
        vae_precond_coeff=float(coeff),
        vae_precond_burg_coeff=0.0,
        vae_precond_every=1,
        vae_precond_samples=int(sample_count),
        vae_precond_pairs=int(pair_count),
        vae_precond_batch_size=int(batch_size),
        vae_precond_estimator_scope=str(scope),
        vae_precond_hvp_mode="stopped_composite",
    )
    batch_indices = torch.as_tensor(indices[: max(2, int(sample_count))], device=device, dtype=torch.long)
    batch = normalizer.normalize(weights.index_select(0, batch_indices))
    batch_raw = weights.index_select(0, batch_indices)
    recs = records.iloc[indices[: int(sample_count)]].to_dict(orient="records")
    for rec, idx in zip(recs, indices[: int(sample_count)], strict=True):
        rec["source_weight_index"] = int(idx)

    params = [param for param in vae.parameters() if param.requires_grad]
    rows: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 99000)

    vae.train()
    vae.zero_grad(set_to_none=True)
    recon, mu, logvar = vae(batch)
    base_loss, base_row = vae_loss(probe_cfg, batch, batch_raw, recon, mu, logvar, normalizer=normalizer, spec=spec)
    base_loss.backward()
    base_norm, base_grads = _grad_stats(params)

    vae.zero_grad(set_to_none=True)
    z = encode_weights(vae.eval(), normalizer, weights[indices[: int(sample_count)]].to(device=device, dtype=dtype)).detach()
    vae.train()
    precond_loss, precond_row = preconditioning_regularizer(
        probe_cfg,
        state=PreconditioningState(),
        vae=vae,
        normalizer=normalizer,
        z_samples=z,
        records=recs,
        task_tensors=task_tensors,
        spec=spec,
        step=1,
        generator=generator,
    )
    precond_loss.backward()
    precond_norm, _precond_grads = _grad_stats(params)
    dot = _grad_dot(base_grads, params)
    cosine = dot / max(1e-30, base_norm * precond_norm)
    rows.append(
        {
            "scope": scope,
            "base_grad_norm": base_norm,
            "precond_grad_norm": precond_norm,
            "precond_to_base_grad_norm_ratio": precond_norm / max(base_norm, 1e-30),
            "base_precond_grad_cosine": cosine,
            "base_loss": float(base_loss.detach().cpu().item()),
            "base_recon_mse": float(base_row["recon_mse"]),
            **{f"precond_{key}": value for key, value in precond_row.items()},
        }
    )
    vae.eval()
    return pd.DataFrame(rows)


def outlier_diagnostics(
    *,
    cfg: ExperimentConfig,
    weights: torch.Tensor,
    records: pd.DataFrame,
    spec: Any,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    task_tensors: dict[str, Any],
    indices: list[int],
    grad_batches_values: list[int],
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grad_batches in grad_batches_values:
        print(
            "[li_a_debug] outlier diagnostics stage "
            f"grad_batches={int(grad_batches)} indices={indices} batch_size={int(batch_size)}",
            flush=True,
        )
        diag_cfg = replace(
            cfg,
            vae_precond_loss_kind="li_a_hvp",
            vae_precond_batch_size=int(batch_size),
            vae_precond_diagnostic_grad_batches=int(grad_batches),
            vae_precond_hvp_mode="stopped_composite",
        )
        z_all = encode_weights(vae, normalizer, weights[indices]).detach()
        for sample_pos, idx in enumerate(indices):
            print(
                "[li_a_debug] outlier diagnostic "
                f"grad_batches={int(grad_batches)} sample={sample_pos + 1}/{len(indices)} weight_index={int(idx)}",
                flush=True,
            )
            record = records.iloc[int(idx)].to_dict()
            record["source_weight_index"] = int(idx)
            row = latent_hessian_diagnostic_row(
                diag_cfg,
                vae=vae,
                normalizer=normalizer,
                z=z_all[sample_pos],
                record=record,
                task_tensors=task_tensors,
                spec=spec,
                sample_index=sample_pos,
                weight_index=int(idx),
                batch_size=int(batch_size),
            )
            row["diagnostic_grad_batches"] = int(grad_batches)
            rows.append(row)
    return pd.DataFrame(rows)


def write_plots(output_dir: Path, estimator: pd.DataFrame, gradients: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not estimator.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for scope, group in estimator.groupby("scope"):
            values = pd.to_numeric(group["precond_a_paired_loss"], errors="coerce").dropna()
            ax.hist(values.clip(lower=-5, upper=20), bins=60, alpha=0.55, label=str(scope))
        ax.set_title("Variant A train estimator distribution")
        ax.set_xlabel("paired A estimator / dim, clipped to [-5, 20] for display")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "a_estimator_distribution.png", dpi=160)
        plt.close(fig)

    if not gradients.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(gradients["scope"].astype(str), gradients["precond_to_base_grad_norm_ratio"].astype(float))
        ax.set_title("A-gradient pressure vs reconstruction")
        ax.set_ylabel("||grad_A|| / ||grad_recon||")
        fig.tight_layout()
        fig.savefig(output_dir / "a_gradient_norm_ratio.png", dpi=160)
        plt.close(fig)

    if not diagnostics.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot = diagnostics.pivot_table(
            index="weight_index",
            columns="diagnostic_grad_batches",
            values="hvp_probe_a_loss_per_dim",
            aggfunc="median",
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Outlier HVP-probe A estimate vs probe count")
        ax.set_xlabel("weight_index")
        ax.set_ylabel("hvp_probe_a_loss_per_dim")
        fig.tight_layout()
        fig.savefig(output_dir / "outlier_hvp_probe_count_sensitivity.png", dpi=160)
        plt.close(fig)


def _scope_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    if frame.empty:
        return summary
    for scope, group in frame.groupby("scope"):
        scope_summary: dict[str, dict[str, float]] = {}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna() if column in group.columns else pd.Series(dtype=float)
            if values.empty:
                continue
            scope_summary[column] = {
                "count": float(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p50": float(values.quantile(0.50)),
                "p90": float(values.quantile(0.90)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        summary[str(scope)] = scope_summary
    return summary


def _diagnostic_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    if frame.empty:
        return summary
    for grad_batches, group in frame.groupby("diagnostic_grad_batches"):
        row: dict[str, float] = {}
        for column in columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[column] = float(values.median())
        summary[str(int(grad_batches))] = row
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="A-only debug probes for CELO li_a_hvp.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--indices", default="")
    parser.add_argument("--index-limit", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--scopes", default="local,global")
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--pair-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--coeff", type=float, default=0.01)
    parser.add_argument("--diagnostic-grad-batches", default="16,64,256")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg, weights, records, spec, vae, normalizer = _load_run(run_dir, device=device)
    task_tensors = _load_task_tensors_for_pipeline(cfg, device=device, dtype=torch_dtype(cfg))
    task_tensors = move_task_tensors(task_tensors, device=device, dtype=torch_dtype(cfg))
    if args.indices.strip():
        indices = [int(v) for v in args.indices.split(",") if v.strip()]
    else:
        indices = _default_indices(run_dir, int(args.index_limit))
    scopes = [value.strip() for value in args.scopes.split(",") if value.strip()]
    grad_batches_values = [int(v) for v in args.diagnostic_grad_batches.split(",") if v.strip()]

    print(
        "[li_a_debug] start "
        f"run_dir={run_dir} output_dir={output_dir} device={device} "
        f"indices={indices} scopes={scopes} repeats={int(args.repeats)} "
        f"sample_count={int(args.sample_count)} pair_count={int(args.pair_count)} "
        f"batch_size={int(args.batch_size)} grad_batches={grad_batches_values}",
        flush=True,
    )
    estimator = estimator_probe(
        cfg=cfg,
        weights=weights,
        records=records,
        spec=spec,
        vae=vae,
        normalizer=normalizer,
        task_tensors=task_tensors,
        indices=indices,
        repeats=int(args.repeats),
        scopes=scopes,
        sample_count=int(args.sample_count),
        pair_count=int(args.pair_count),
        batch_size=int(args.batch_size),
        coeff=float(args.coeff),
    )
    estimator.to_csv(output_dir / "a_estimator_samples.csv", index=False)
    print(f"[li_a_debug] wrote estimator samples rows={len(estimator)}", flush=True)
    gradients = pd.concat(
        [
            gradient_probe(
                cfg=cfg,
                weights=weights,
                records=records,
                spec=spec,
                vae=vae,
                normalizer=normalizer,
                task_tensors=task_tensors,
                indices=indices,
                scope=scope,
                sample_count=int(args.sample_count),
                pair_count=int(args.pair_count),
                batch_size=int(args.batch_size),
                coeff=float(args.coeff),
            )
            for scope in scopes
        ],
        ignore_index=True,
    )
    gradients.to_csv(output_dir / "a_gradient_components.csv", index=False)
    print(f"[li_a_debug] wrote gradient components rows={len(gradients)}", flush=True)
    diagnostics = outlier_diagnostics(
        cfg=cfg,
        weights=weights,
        records=records,
        spec=spec,
        vae=vae,
        normalizer=normalizer,
        task_tensors=task_tensors,
        indices=indices,
        grad_batches_values=grad_batches_values,
        batch_size=int(args.batch_size),
    )
    diagnostics.to_csv(output_dir / "a_outlier_diagnostics.csv", index=False)
    print(f"[li_a_debug] wrote outlier diagnostics rows={len(diagnostics)}", flush=True)
    write_plots(output_dir, estimator, gradients, diagnostics)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "indices": indices,
        "estimator_summary": _scope_summary(
            estimator,
            ["precond_a_paired_loss", "precond_a_global_audit_loss", "precond_trace_m_per_dim"],
        ),
        "gradient_components": gradients.to_dict(orient="records"),
        "diagnostic_summary": _diagnostic_summary(
            diagnostics,
            ["li_A_full_per_dim", "hvp_probe_a_loss_per_dim", "hvp_probe_trace_m_per_dim"],
        ),
        "outputs": {
            "estimator_samples": str(output_dir / "a_estimator_samples.csv"),
            "gradient_components": str(output_dir / "a_gradient_components.csv"),
            "outlier_diagnostics": str(output_dir / "a_outlier_diagnostics.csv"),
            "estimator_plot": str(output_dir / "a_estimator_distribution.png"),
            "gradient_plot": str(output_dir / "a_gradient_norm_ratio.png"),
            "outlier_plot": str(output_dir / "outlier_hvp_probe_count_sensitivity.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[li_a_debug] wrote {json.dumps(summary['outputs'], sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
