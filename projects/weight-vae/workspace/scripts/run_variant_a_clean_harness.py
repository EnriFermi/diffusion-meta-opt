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
DEFAULT_SUMMARY = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/clean_harness_runs/clean_harness_summary.csv"
)


RUN_SPECS: tuple[dict[str, float | str], ...] = (
    {
        "suffix": "control_clean",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "anchor_coeff": 0.0,
    },
    {
        "suffix": "a_cap1_clean",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 1.000,
        "anchor_coeff": 0.0,
    },
    {
        "suffix": "control_anchor_c0p0001_clean",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_logit_anchor_c0p0001_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "anchor_coeff": 0.0001,
    },
    {
        "suffix": "a_cap1_anchor_c0p0001_clean",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_logit_anchor_c0p0001_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 1.000,
        "anchor_coeff": 0.0001,
    },
    {
        "suffix": "control_anchor_c0p0003_clean",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_logit_anchor_c0p0003_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "anchor_coeff": 0.0003,
    },
    {
        "suffix": "a_cap1_anchor_c0p0003_clean",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_logit_anchor_c0p0003_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 1.000,
        "anchor_coeff": 0.0003,
    },
)


def _selected_specs(names: set[str] | None) -> tuple[dict[str, float | str], ...]:
    if not names:
        return RUN_SPECS
    result = tuple(spec for spec in RUN_SPECS if str(spec["suffix"]) in names)
    missing = sorted(names - {str(spec["suffix"]) for spec in result})
    if missing:
        raise ValueError(f"unknown clean harness spec suffixes: {missing}")
    return result


def _anchor_training_summary(output_dir: Path) -> dict[str, float]:
    vae_metrics = pd.read_csv(output_dir / "vae_metrics.csv")
    record_type = vae_metrics.get("record_type", pd.Series("", index=vae_metrics.index)).fillna("").astype(str)
    train = vae_metrics[record_type != "vae_quality"].copy()
    result: dict[str, float] = {}
    for name in [
        "train_vae_batch_index_hash",
        "train_function_anchor_loss",
        "train_function_anchor_effective_loss",
        "train_function_anchor_ce_delta",
        "train_function_anchor_acc_delta",
        "train_function_anchor_index_hash_mean",
    ]:
        if name in train:
            values = pd.to_numeric(train[name], errors="coerce").dropna()
            result[f"{name}_median"] = float(values.median()) if not values.empty else float("nan")
            result[f"{name}_final"] = float(values.iloc[-1]) if not values.empty else float("nan")
        else:
            result[f"{name}_median"] = float("nan")
            result[f"{name}_final"] = float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Variant A critical comparisons with cleaned experiment harness.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", action="append", default=[], help="Run only a named suffix from RUN_SPECS.")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    specs = _selected_specs(set(args.only) if args.only else None)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[variant_a_clean_harness] start "
        f"specs={len(specs)} device={args.device} baseline_dir={BASELINE_DIR} summary_out={args.summary_out}",
        flush=True,
    )
    rows: list[dict[str, float | int | str]] = []
    t0 = time.perf_counter()
    for idx, spec in enumerate(specs, start=1):
        suffix = str(spec["suffix"])
        run_label = str(spec["run_label"])
        precond_kind = str(spec["precond_kind"])
        alpha = float(spec["alpha"])
        cap = float(spec["cap"])
        anchor_coeff = float(spec["anchor_coeff"])
        cfg = celo_meta_config(
            run_label=run_label,
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
            vae_precond_loss_kind=precond_kind,
            vae_precond_coeff=alpha,
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
            vae_precond_max_grad_ratio=cap,
            vae_precond_grad_clip_norm=0.0,
            vae_function_anchor_coeff=anchor_coeff,
            vae_function_anchor_samples=2,
            vae_function_anchor_batch_size=128,
            vae_function_anchor_loss_kind="logit_mse",
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
        print(
            "[variant_a_clean_harness] run "
            f"{idx}/{len(specs)} suffix={suffix} precond={precond_kind} alpha={alpha:.6g} cap={cap:.6g} "
            f"anchor_coeff={anchor_coeff:.6g} grad_clip=0 run_label={run_label} "
            f"output_dir={output_dir} config_hash={config_hash(cfg)}",
            flush=True,
        )
        print(f"[variant_a_clean_harness] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}", flush=True)
        tables = run_or_load(cfg)
        row = _summary_for_run(tables.output_dir, suffix=suffix, alpha=alpha, cap=cap)
        row["anchor_coeff"] = anchor_coeff
        row.update(_anchor_training_summary(tables.output_dir))
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.summary_out, index=False)
        print(
            "[variant_a_clean_harness] summary "
            f"suffix={suffix} decoder_aulc_mean={row['decoder_aulc_mean']:.6g} "
            f"decoder_aulc_median={row['decoder_aulc_median']:.6g} li_A_p95={row['li_A_full_per_dim_p95']:.6g} "
            f"trace_m={row['hvp_probe_trace_m_per_dim_median']:.6g} recon_rel_l2={row['reconstruction_rel_l2_median']:.6g} "
            f"batch_hash_median={row.get('train_vae_batch_index_hash_median', float('nan')):.6g} "
            f"summary_out={args.summary_out}",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    print(f"[variant_a_clean_harness] done elapsed_sec={elapsed:.2f} summary_out={args.summary_out}", flush=True)


if __name__ == "__main__":
    main()
