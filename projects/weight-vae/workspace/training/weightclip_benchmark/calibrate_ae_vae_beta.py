from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer, operator_bank_data_pipeline
from big_vae.models import build_weight_quantile_vae
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu
from training.weightclip_benchmark.diagnose_ae_gradient_path import _losses
from training.weightclip_benchmark.run_ae_v3_vae_50k import _build_worker_config


BETA_CANDIDATES = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 5.0, 10.0, 15.0, 20.0)
TARGET_WEIGHTED_GRADIENT_RATIO = 0.01


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate mean-KL beta on shared real operator batches")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v3_vae_50k.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/coder/project/projects/shared/storage/artifacts/weightclip_benchmark/"
            "ae_725m_lat384_prenorm_v3_vae_beta_calibration.json"
        ),
    )
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().float()).item())


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("vae-beta-calibration")
    spec = yaml.safe_load(args.config.resolve().read_text())
    worker_cfg, _output_root, _is_resume = _build_worker_config(
        spec,
        resume_checkpoint=None,
        max_steps_override=None,
        stop_after_step_override=None,
    )
    cfg = OmegaConf.create(worker_cfg)
    device = torch.device(args.device)
    torch.manual_seed(int(cfg.data.seed))
    model = build_weight_quantile_vae(build_big_vae_model_config(cfg)).to(device)
    model.train()
    model.set_latent_sampling_gate(0.0)

    captured_posterior_inputs: list[torch.Tensor] = []
    if model.to_mu is None:
        raise RuntimeError("VAE beta calibration requires to_mu")
    handle = model.to_mu.register_forward_pre_hook(
        lambda _module, inputs: captured_posterior_inputs.append(inputs[0])
    )

    rows: list[dict[str, Any]] = []
    pair = str(cfg.train.operator_bank.pair_manifest)
    try:
        with operator_bank_data_pipeline(
            pair,
            seed=int(cfg.data.seed),
            repeat=True,
            permutation_views=bool(cfg.train.operator_bank.permutation_views),
            canonical_probability=float(cfg.train.operator_bank.canonical_probability),
            hot_shards=int(cfg.train.operator_bank.hot_shards),
            expected_pair_manifest_sha256=str(cfg.train.operator_bank.pair_manifest_sha256),
            rank=0,
            world_size=1,
            max_active_strata=int(cfg.train.operator_bank.max_active_strata),
            max_active_bundle_bytes=int(cfg.train.operator_bank.max_active_bundle_bytes),
            logger=logger,
        ) as (dataset, sampler):
            sampler.set_start_index(0)
            mixer = BalancedOperatorBankMixer(dataset, (dataset[request] for request in sampler), start_index=0)
            for batch_idx in range(int(args.batches)):
                batch = _fetch_presliced_training_batch_cpu(
                    dataset_iter=mixer,
                    batch_size=int(args.batch_size),
                    logger=logger,
                )
                W, x, x_mask, d_in_mask, d_out_mask = [
                    tensor.to(device)
                    for tensor in (batch.W, batch.x, batch.x_mask, batch.d_in_mask, batch.d_out_mask)
                ]
                captured_posterior_inputs.clear()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base_loss, loss_parts, outputs = _losses(model, cfg, W, x, x_mask, d_in_mask, d_out_mask)
                    _W_hat, mu, logvar, _pred_dirs = outputs
                    kl = model.latent_kl_loss(mu, logvar)
                if len(captured_posterior_inputs) != 1:
                    raise RuntimeError(
                        f"expected one posterior input tensor, captured {len(captured_posterior_inputs)}"
                    )
                posterior_input = captured_posterior_inputs[0]
                g_base = torch.autograd.grad(base_loss, posterior_input, retain_graph=True)[0]
                g_kl = torch.autograd.grad(kl, posterior_input)[0]
                base_norm = _l2(g_base)
                kl_norm = _l2(g_kl)
                dot = float(torch.sum(g_base.detach().float() * g_kl.detach().float()).item())
                cosine = dot / max(1.0e-30, base_norm * kl_norm)
                rows.append(
                    {
                        "batch": batch_idx,
                        "logical_indices": list(batch.logical_indices),
                        "base_loss": float(base_loss.detach().item()),
                        "kl_mean_per_coordinate": float(kl.detach().item()),
                        "mu_rms": float(mu.detach().float().square().mean().sqrt().item()),
                        "posterior_std_mean": float(torch.exp(0.5 * logvar.detach().float()).mean().item()),
                        "base_gradient_l2_at_posterior_input": base_norm,
                        "kl_gradient_l2_at_posterior_input": kl_norm,
                        "raw_gradient_ratio_kl_to_base": kl_norm / max(1.0e-30, base_norm),
                        "gradient_cosine_kl_vs_base": cosine,
                        "losses": {key: float(value.detach().item()) for key, value in loss_parts.items()},
                    }
                )
                logger.info(
                    "batch=%s base=%.6f kl=%.6f raw_grad_ratio=%.6f mu_rms=%.6f std=%.6f",
                    batch_idx,
                    rows[-1]["base_loss"],
                    rows[-1]["kl_mean_per_coordinate"],
                    rows[-1]["raw_gradient_ratio_kl_to_base"],
                    rows[-1]["mu_rms"],
                    rows[-1]["posterior_std_mean"],
                )
    finally:
        handle.remove()

    raw_ratio = median(float(row["raw_gradient_ratio_kl_to_base"]) for row in rows)
    base_value = median(float(row["base_loss"]) for row in rows)
    kl_value = median(float(row["kl_mean_per_coordinate"]) for row in rows)
    candidates = []
    for beta in BETA_CANDIDATES:
        candidates.append(
            {
                "beta": beta,
                "weighted_gradient_ratio": beta * raw_ratio,
                "weighted_loss_fraction": beta * kl_value / max(1.0e-30, base_value),
            }
        )
    selected = min(
        candidates,
        key=lambda item: abs(
            math.log(max(1.0e-30, float(item["weighted_gradient_ratio"])))
            - math.log(TARGET_WEIGHTED_GRADIENT_RATIO)
        ),
    )
    report = {
        "schema": "weightclip_vae_beta_calibration_v1",
        "criterion": {
            "location": "raw encoder latent input to posterior mean head",
            "target_weighted_gradient_ratio": TARGET_WEIGHTED_GRADIENT_RATIO,
            "candidate_betas": list(BETA_CANDIDATES),
        },
        "model": {
            "architecture_version": str(cfg.model.big_vae.architecture_version),
            "num_latents": int(cfg.model.big_vae.num_latents),
            "d_lat": int(cfg.model.big_vae.d_lat),
            "sampling_gate": 0.0,
        },
        "rows": rows,
        "median_raw_gradient_ratio": raw_ratio,
        "median_base_loss": base_value,
        "median_kl_mean_per_coordinate": kl_value,
        "candidates": candidates,
        "selected_beta": float(selected["beta"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[vae-beta-calibration] selected_beta={report['selected_beta']}", flush=True)
    print(f"[vae-beta-calibration] report={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
