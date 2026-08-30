from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing import celo_meta_config, run_or_load
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import config_hash, run_dir
from scripts.run_variant_a_frontier import BASELINE_DIR, SOURCE_DATA_ROOT, _summary_for_run


ROOT = Path(__file__).resolve().parents[1]
RUN_LABEL = "sage_cnn_vae_smoothing_celo_meta_finetune_control_current_v1_seed0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a current-code no-A control matched to Variant A frontier runs.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT
        / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/frontier_single_summaries/current_control.csv",
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    cfg = celo_meta_config(
        run_label=RUN_LABEL,
        seed=0,
        device=str(args.device),
        dtype="float32",
        data_root=SOURCE_DATA_ROOT,
        cache_first=True,
        force_rerun=bool(args.force_rerun),
        progress_backend="text",
        live_vae_loss_curve=False,
        weight_pool_source_dir=str(BASELINE_DIR),
        vae_init_checkpoint=str(BASELINE_DIR / "vae_checkpoint.pt"),
        vae_steps=5000,
        vae_lr=3e-5,
        vae_batch_size=128,
        vae_precond_loss_kind="none",
        vae_precond_coeff=0.0,
        vae_precond_burg_coeff=0.0,
        vae_precond_every=10,
        vae_precond_samples=2,
        vae_precond_pairs=4,
        vae_precond_batch_size=128,
        vae_precond_probe_scale=1.0,
        vae_precond_estimator_scope="local",
        vae_precond_hvp_mode="stopped_composite",
        vae_precond_loss_clip=0.0,
        vae_precond_grad_damping=0.0,
        vae_precond_warmup_steps=0,
        vae_precond_ramp_steps=1000,
        vae_precond_max_grad_ratio=0.25,
        vae_precond_grad_clip_norm=1.0,
        geometry_eval_samples=32,
        geometry_jacobian_chunk_size=4,
        vae_precond_diagnostic_grad_batches=64,
        tune_starts=8,
        eval_starts=16,
        downstream_steps=300,
        downstream_eval_every=25,
        downstream_batch_size=128,
        comet_enabled=False,
    )
    output_dir = run_dir(cfg)
    t0 = time.perf_counter()
    print(
        "[variant_a_current_control] start "
        f"device={args.device} baseline_dir={BASELINE_DIR} output_dir={output_dir} "
        f"summary_out={args.summary_out} config_hash={config_hash(cfg)}",
        flush=True,
    )
    print(f"[variant_a_current_control] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}", flush=True)
    tables = run_or_load(cfg)
    row = _summary_for_run(tables.output_dir, suffix="current_control", alpha=0.0, cap=0.0)
    pd.DataFrame([row]).to_csv(args.summary_out, index=False)
    elapsed = time.perf_counter() - t0
    print(
        "[variant_a_current_control] done "
        f"elapsed_sec={elapsed:.2f} decoder_aulc_median={row['decoder_aulc_median']:.6g} "
        f"decoder_aulc_mean={row['decoder_aulc_mean']:.6g} li_A_p95={row['li_A_full_per_dim_p95']:.6g} "
        f"recon_rel_l2={row['reconstruction_rel_l2_median']:.6g} summary_out={args.summary_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
