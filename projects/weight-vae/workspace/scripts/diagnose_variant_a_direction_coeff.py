from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    load_torch_cache,
    spec_from_payload,
    vae_block_direction_loss,
    vae_loss,
)


def _log(message: str) -> None:
    print(f"[direction_coeff_diag] {message}", flush=True)


def _load_cfg(run_dir: Path) -> ExperimentConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return ExperimentConfig(**payload["config"])


def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    norm2 = 0.0
    for param in params:
        if param.grad is None:
            continue
        norm2 += float(param.grad.detach().float().square().sum().cpu().item())
    return math.sqrt(norm2)


def _parse_coeffs(value: str) -> list[float]:
    result = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not result:
        raise ValueError("at least one coefficient candidate is required")
    return result


def _sample_batch_indices(train_indices: torch.Tensor, *, batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    positions = torch.randint(0, int(train_indices.numel()), (int(batch_size),), generator=generator, device="cpu")
    return train_indices.detach().cpu().long().index_select(0, positions)


def _diagnose_run(
    *,
    run_dir: Path,
    run_name: str,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    block: str,
    space: str,
    step: int,
    coeffs: list[float],
) -> pd.DataFrame:
    cfg = _load_cfg(run_dir)
    dtype = torch_dtype(cfg)
    probe_cfg = replace(
        cfg,
        device=str(device),
        vae_block_direction_coeff=1.0,
        vae_block_direction_block=str(block),
        vae_block_direction_space=str(space),
        vae_block_direction_ramp_steps=0,
    )
    weight_payload = load_torch_cache(run_dir / "weight_pool.pt")
    vae_payload = load_torch_cache(run_dir / "vae_checkpoint.pt")
    if weight_payload is None or vae_payload is None:
        raise FileNotFoundError(f"{run_dir} must contain weight_pool.pt and vae_checkpoint.pt")
    weights = weight_payload["weights"].to(device=device, dtype=dtype)
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
    train_indices = vae_payload["train_indices"].detach().cpu().long()
    vae = build_weight_vae(probe_cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=dtype)
    vae.load_state_dict(vae_payload["model_state"])
    vae.train()
    x_norm = normalizer.normalize(weights)
    params = [param for param in vae.parameters() if param.requires_grad]
    rows: list[dict[str, Any]] = []
    _log(
        "run_start "
        f"name={run_name} run_dir={run_dir} device={device} dtype={dtype} "
        f"weights_shape={tuple(weights.shape)} train_indices={int(train_indices.numel())} "
        f"batch_size={int(batch_size)} batches={int(batches)} block={block} space={space}"
    )
    for batch_id in range(int(batches)):
        batch_indices = _sample_batch_indices(train_indices, batch_size=int(batch_size), seed=int(seed) + 1009 * batch_id)
        batch_indices_dev = batch_indices.to(device=device)
        batch = x_norm.index_select(0, batch_indices_dev)
        batch_raw = weights.index_select(0, batch_indices_dev)
        torch.manual_seed(int(seed) + 100000 + batch_id)
        vae.zero_grad(set_to_none=True)
        recon, mu, logvar = vae(batch)
        base_loss, base_row = vae_loss(
            probe_cfg,
            batch,
            batch_raw,
            recon,
            mu,
            logvar,
            normalizer=normalizer,
            spec=spec,
        )
        base_loss.backward(retain_graph=True)
        base_grad_norm = _grad_norm(params)
        vae.zero_grad(set_to_none=True)
        direction_recon = vae.decode_norm(mu)
        direction_loss, direction_row = vae_block_direction_loss(
            probe_cfg,
            x_norm=batch,
            target_weights=batch_raw,
            recon_norm=direction_recon,
            normalizer=normalizer,
            spec=spec,
            step=int(step),
        )
        direction_loss.backward()
        unit_grad_norm = _grad_norm(params)
        vae.zero_grad(set_to_none=True)
        unit_ratio = float(unit_grad_norm / max(base_grad_norm, 1e-30))
        row: dict[str, Any] = {
            "run_name": str(run_name),
            "run_dir": str(run_dir),
            "batch_id": int(batch_id),
            "batch_size": int(batch_size),
            "batch_index_min": int(batch_indices.min().item()),
            "batch_index_max": int(batch_indices.max().item()),
            "base_loss": float(base_loss.detach().cpu().item()),
            "base_grad_norm": float(base_grad_norm),
            "unit_direction_loss": float(direction_loss.detach().cpu().item()),
            "unit_direction_grad_norm": float(unit_grad_norm),
            "unit_direction_grad_ratio": unit_ratio,
            "block_direction_row_cos_mean": float(direction_row["block_direction_row_cos_mean"]),
            "block_direction_row_cos_min": float(direction_row["block_direction_row_cos_min"]),
            "block_direction_row_error_mean": float(direction_row["block_direction_row_error_mean"]),
            "block_direction_row_error_max": float(direction_row["block_direction_row_error_max"]),
            "block_direction_norm_ratio_mean": float(direction_row["block_direction_norm_ratio_mean"]),
        }
        for coeff in coeffs:
            row[f"ratio_at_c{str(coeff).replace('.', 'p')}"] = unit_ratio * float(coeff)
        rows.append(row)
        _log(
            "batch "
            f"name={run_name} batch={batch_id + 1}/{int(batches)} "
            f"base_grad={base_grad_norm:.6g} unit_dir_grad={unit_grad_norm:.6g} "
            f"unit_ratio={unit_ratio:.6g} row_cos_mean={row['block_direction_row_cos_mean']:.6g}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate one fc2.weight row-direction guard coefficient by VAE gradient ratio, without downstream selection."
    )
    parser.add_argument("--run", action="append", required=True, help="Run spec as name=/path/to/run_dir or /path/to/run_dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--block", default="fc2.weight")
    parser.add_argument("--space", default="denormalized")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--coeff-candidates", default="0.001,0.002,0.003,0.005,0.01")
    parser.add_argument("--target-ratio", type=float, default=0.12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coeffs = _parse_coeffs(args.coeff_candidates)
    device = torch.device(str(args.device))
    _log(
        "start "
        f"runs={len(args.run)} output_dir={args.output_dir} device={device} "
        f"batch_size={args.batch_size} batches={args.batches} seed={args.seed} "
        f"block={args.block} space={args.space} target_ratio={args.target_ratio} coeffs={coeffs}"
    )
    frames: list[pd.DataFrame] = []
    for spec in args.run:
        if "=" in str(spec):
            name, path = str(spec).split("=", 1)
            run_name = name.strip()
            run_dir = Path(path).expanduser().resolve()
        else:
            run_dir = Path(str(spec)).expanduser().resolve()
            run_name = run_dir.name
        frames.append(
            _diagnose_run(
                run_dir=run_dir,
                run_name=run_name,
                device=device,
                batch_size=int(args.batch_size),
                batches=int(args.batches),
                seed=int(args.seed),
                block=str(args.block),
                space=str(args.space),
                step=int(args.step),
                coeffs=coeffs,
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    raw_path = args.output_dir / "direction_coeff_grad_ratio_rows.csv"
    rows.to_csv(raw_path, index=False)
    grouped = (
        rows.groupby("run_name", dropna=False)
        .agg(
            unit_ratio_median=("unit_direction_grad_ratio", "median"),
            unit_ratio_p90=("unit_direction_grad_ratio", lambda x: float(pd.to_numeric(x).quantile(0.90))),
            unit_ratio_max=("unit_direction_grad_ratio", "max"),
            row_error_mean_median=("block_direction_row_error_mean", "median"),
            row_cos_mean_median=("block_direction_row_cos_mean", "median"),
            base_grad_norm_median=("base_grad_norm", "median"),
            unit_direction_grad_norm_median=("unit_direction_grad_norm", "median"),
        )
        .reset_index()
    )
    pooled_unit_median = float(pd.to_numeric(rows["unit_direction_grad_ratio"], errors="coerce").median())
    recommended = float(args.target_ratio) / max(pooled_unit_median, 1e-30)
    for coeff in coeffs:
        grouped[f"median_ratio_at_c{str(coeff).replace('.', 'p')}"] = grouped["unit_ratio_median"] * float(coeff)
        grouped[f"p90_ratio_at_c{str(coeff).replace('.', 'p')}"] = grouped["unit_ratio_p90"] * float(coeff)
    summary_path = args.output_dir / "direction_coeff_grad_ratio_summary.csv"
    grouped.to_csv(summary_path, index=False)
    payload = {
        "target_ratio": float(args.target_ratio),
        "pooled_unit_ratio_median": pooled_unit_median,
        "recommended_coeff_for_target_ratio": recommended,
        "coeff_candidates": coeffs,
        "raw_rows": str(raw_path),
        "summary_csv": str(summary_path),
        "selection_rule": "Choose one coefficient before downstream: target direction_grad_ratio near block-recon pressure (~0.1-0.15) and below function-anchor pressure (~0.3-0.45).",
    }
    json_path = args.output_dir / "direction_coeff_selection.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _log(
        "done "
        f"raw_rows={raw_path} summary_csv={summary_path} selection_json={json_path} "
        f"pooled_unit_ratio_median={pooled_unit_median:.6g} recommended_coeff={recommended:.6g}"
    )


if __name__ == "__main__":
    main()
