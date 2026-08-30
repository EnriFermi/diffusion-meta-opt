from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import ExperimentConfig, torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    TrainedVAE,
    WeightNormalizer,
    build_weight_vae,
    load_celo_meta_task_tensors,
    load_torch_cache,
    move_task_tensors,
    spec_from_payload,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _diagnostic_indices_from_start_bank,
    _preconditioning_diagnostics_table,
    _start_bank_table,
)


def _load_cfg(output_dir: Path, *, device: str | None, samples: int | None, batch_size: int | None, grad_batches: int | None) -> ExperimentConfig:
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"config.json in {output_dir} does not contain a config mapping")
    values = dict(raw_cfg)
    if device:
        values["device"] = str(device)
    if samples is not None:
        values["geometry_eval_samples"] = int(samples)
    if batch_size is not None:
        values["vae_precond_batch_size"] = int(batch_size)
    if grad_batches is not None:
        values["vae_precond_diagnostic_grad_batches"] = int(grad_batches)
    return ExperimentConfig(**values)


def compute_diagnostics(
    output_dir: Path,
    *,
    device: str | None = None,
    samples: int | None = None,
    batch_size: int | None = None,
    grad_batches: int | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    cfg = _load_cfg(output_dir, device=device, samples=samples, batch_size=batch_size, grad_batches=grad_batches)
    runtime_device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    print(
        "[reparam_diag] start "
        f"output_dir={output_dir} device={runtime_device} dtype={dtype} seed={int(cfg.seed)} "
        f"samples={int(cfg.geometry_eval_samples)} batch_size={int(cfg.vae_precond_batch_size)} "
        f"grad_batches={int(cfg.vae_precond_diagnostic_grad_batches)}",
        flush=True,
    )
    weight_payload = load_torch_cache(output_dir / "weight_pool.pt")
    if weight_payload is None:
        raise FileNotFoundError(output_dir / "weight_pool.pt")
    vae_payload = load_torch_cache(output_dir / "vae_checkpoint.pt")
    if vae_payload is None:
        raise FileNotFoundError(output_dir / "vae_checkpoint.pt")
    weights = weight_payload["weights"]
    weight_records = pd.DataFrame(weight_payload["records"])
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=runtime_device, dtype=dtype)
    vae.load_state_dict(vae_payload["model_state"])
    trained = TrainedVAE(
        vae=vae,
        normalizer=normalizer,
        train_indices=vae_payload["train_indices"],
        val_indices=vae_payload["val_indices"],
        metrics=pd.DataFrame(vae_payload.get("metrics", [])),
    )
    task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=runtime_device, dtype=dtype)
    start_bank_path = output_dir / "downstream_start_bank.csv"
    if start_bank_path.is_file():
        start_bank = pd.read_csv(start_bank_path)
    else:
        start_bank = _start_bank_table(
            cfg=cfg,
            trained=trained,
            weight_records=weight_records,
            weights_count=int(weights.shape[0]),
            variant="baseline",
            reg_coeff=float(getattr(cfg, "vae_geometry_reg_coeff", 0.0)),
        )
        start_bank.to_csv(start_bank_path, index=False)
        print(f"[reparam_diag] wrote start_bank path={start_bank_path} rows={len(start_bank)}", flush=True)
    diagnostic_indices = _diagnostic_indices_from_start_bank(start_bank, max_samples=int(cfg.geometry_eval_samples))
    diagnostics = _preconditioning_diagnostics_table(
        cfg=cfg,
        trained=trained,
        weights=weights,
        weight_records=weight_records,
        spec=spec,
        task_tensors=task_tensors,
        eval_indices=diagnostic_indices,
        start_bank=start_bank,
    )
    path = output_dir / "preconditioning_diagnostics.csv"
    diagnostics.to_csv(path, index=False)
    summary = diagnostics[["li_trace_h2_per_dim", "li_A_full_per_dim", "li_gap_per_dim", "grad_trace_per_dim"]].median(numeric_only=True).to_dict()
    print(f"[reparam_diag] wrote path={path} rows={len(diagnostics)} median_summary={summary}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute latent Li/gradient diagnostics for a CELO TinyBigVAE output directory.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-batches", type=int, default=None)
    args = parser.parse_args()
    compute_diagnostics(
        args.output_dir,
        device=args.device,
        samples=args.samples,
        batch_size=args.batch_size,
        grad_batches=args.grad_batches,
    )


if __name__ == "__main__":
    main()
