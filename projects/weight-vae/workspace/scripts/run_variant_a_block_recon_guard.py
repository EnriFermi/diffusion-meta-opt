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
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_runs/block_recon_summary.csv"
)


RUN_SPECS: tuple[dict[str, float | str], ...] = (
    {
        "suffix": "control_fc2recon_lam0p03",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.03,
    },
    {
        "suffix": "a_cap1_fc2recon_lam0p03",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p03_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 1.000,
        "block_coeff": 0.03,
    },
    {
        "suffix": "a_cap0p25_fc2recon_lam0p03",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
    },
    {
        "suffix": "control_fc2recon_lam0p03_headce_c3e4_m0",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_headce_c0p0003_m0_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.03,
        "head_coeff": 0.0003,
        "head_margin": 0.0,
    },
    {
        "suffix": "a_cap0p25_fc2recon_lam0p03_headce_c3e4_m0",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_headce_c0p0003_m0_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "head_coeff": 0.0003,
        "head_margin": 0.0,
    },
    {
        "suffix": "control_fc2recon_lam0p03_headhuber_c3e4_m0p02_d0p05_top1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_headhuber_c0p0003_m0p02_d0p05_top1_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.03,
        "head_coeff": 0.0003,
        "head_margin": 0.02,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_loss_kind": "head_ce_gap_huber",
    },
    {
        "suffix": "a_cap0p25_fc2recon_lam0p03_headhuber_c3e4_m0p02_d0p05_top1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_headhuber_c0p0003_m0p02_d0p05_top1_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "head_coeff": 0.0003,
        "head_margin": 0.02,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_loss_kind": "head_ce_gap_huber",
    },
    {
        "suffix": "control_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.03,
        "head_coeff": 0.003,
        "head_margin": 0.02,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_loss_kind": "head_ce_gap_huber",
    },
    {
        "suffix": "a_cap0p25_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "head_coeff": 0.003,
        "head_margin": 0.02,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_loss_kind": "head_ce_gap_huber",
    },
    {
        "suffix": "control_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1_rowdir_c0p003",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1_rowdir_c0p003_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.03,
        "head_coeff": 0.003,
        "head_margin": 0.02,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_loss_kind": "head_ce_gap_huber",
        "dir_coeff": 0.003,
    },
    {
        "suffix": "a_cap0p25_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1_rowdir_c0p003",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_headhuber_c0p003_m0p02_d0p05_top1_rowdir_c0p003_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "head_coeff": 0.003,
        "head_margin": 0.02,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_loss_kind": "head_ce_gap_huber",
        "dir_coeff": 0.003,
    },
    {
        "suffix": "control_fc2recon_lam0p03_marginhuber_c0p03_m0_d0p05_toprec1_topex0p1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_marginhuber_c0p03_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.03,
        "head_coeff": 0.03,
        "head_margin": 0.0,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_example_top_fraction": 0.10,
        "head_loss_kind": "head_margin_drop_huber",
    },
    {
        "suffix": "a_cap0p25_fc2recon_lam0p03_marginhuber_c0p03_m0_d0p05_toprec1_topex0p1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_marginhuber_c0p03_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "head_coeff": 0.03,
        "head_margin": 0.0,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_example_top_fraction": 0.10,
        "head_loss_kind": "head_margin_drop_huber",
    },
    {
        "suffix": "control_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.03,
        "head_coeff": 0.001,
        "head_margin": 0.0,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_example_top_fraction": 0.10,
        "head_loss_kind": "head_margin_drop_huber",
    },
    {
        "suffix": "a_cap0p25_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "head_coeff": 0.001,
        "head_margin": 0.0,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_example_top_fraction": 0.10,
        "head_loss_kind": "head_margin_drop_huber",
    },
    {
        "suffix": "a_cap0p25_clip5_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_clip5_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "precond_loss_clip": 5.0,
        "head_coeff": 0.001,
        "head_margin": 0.0,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_example_top_fraction": 0.10,
        "head_loss_kind": "head_margin_drop_huber",
    },
    {
        "suffix": "a_cap0p25_clip20_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_clip20_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 0.250,
        "block_coeff": 0.03,
        "grad_clip": 1.0,
        "precond_loss_clip": 20.0,
        "head_coeff": 0.001,
        "head_margin": 0.0,
        "head_huber_delta": 0.05,
        "head_top_fraction": 1.0,
        "head_example_top_fraction": 0.10,
        "head_loss_kind": "head_margin_drop_huber",
    },
    {
        "suffix": "control_fc2recon_lam0p01",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p01_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.01,
    },
    {
        "suffix": "a_cap1_fc2recon_lam0p01",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p01_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 1.000,
        "block_coeff": 0.01,
    },
    {
        "suffix": "control_fc2recon_lam0p06",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p06_clean_harness_v1_seed0",
        "precond_kind": "none",
        "alpha": 0.0,
        "cap": 0.0,
        "block_coeff": 0.06,
    },
    {
        "suffix": "a_cap1_fc2recon_lam0p06",
        "run_label": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p06_clean_harness_v1_seed0",
        "precond_kind": "li_a_hvp",
        "alpha": 0.0100,
        "cap": 1.000,
        "block_coeff": 0.06,
    },
)


def _selected_specs(names: set[str] | None) -> tuple[dict[str, float | str], ...]:
    if not names:
        return tuple(spec for spec in RUN_SPECS if str(spec["suffix"]).endswith("lam0p03"))
    result = tuple(spec for spec in RUN_SPECS if str(spec["suffix"]) in names)
    missing = sorted(names - {str(spec["suffix"]) for spec in result})
    if missing:
        raise ValueError(f"unknown block-recon spec suffixes: {missing}")
    return result


def _selected_lr(output_dir: Path, method: str) -> float:
    rows = pd.read_csv(output_dir / "selected_lrs.csv")
    hit = rows[(rows["method"].astype(str) == method) & (pd.to_numeric(rows["selected"], errors="coerce") == 1)]
    if len(hit) != 1:
        return float("nan")
    return float(hit["candidate_lr"].iloc[0])


def _training_summary(output_dir: Path) -> dict[str, float]:
    vae_metrics = pd.read_csv(output_dir / "vae_metrics.csv")
    record_type = vae_metrics.get("record_type", pd.Series("", index=vae_metrics.index)).fillna("").astype(str)
    train = vae_metrics[record_type != "vae_quality"].copy()
    result: dict[str, float] = {
        "selected_raw_lr": _selected_lr(output_dir, "raw"),
        "selected_decoder_lr": _selected_lr(output_dir, "decoder_latent"),
    }
    for name in [
        "train_block_recon_loss",
        "train_block_recon_effective_loss",
        "train_block_recon_to_full_mse",
        "train_block_recon_rel_l2",
        "train_block_recon_grad_ratio",
        "train_block_direction_loss",
        "train_block_direction_effective_loss",
        "train_block_direction_row_cos_mean",
        "train_block_direction_row_cos_min",
        "train_block_direction_row_error_mean",
        "train_block_direction_row_error_max",
        "train_block_direction_norm_ratio_mean",
        "train_block_direction_grad_ratio",
        "train_precond_loss",
        "train_precond_effective_loss",
        "train_precond_grad_scale",
        "train_precond_base_grad_norm",
        "train_precond_grad_norm",
        "train_function_anchor_loss",
        "train_function_anchor_effective_loss",
        "train_function_anchor_ce_delta",
        "train_function_anchor_huber_delta",
        "train_function_anchor_acc_delta",
        "train_function_anchor_margin_drop",
        "train_function_anchor_margin_drop_active_fraction",
        "train_function_anchor_raw_margin",
        "train_function_anchor_decoded_margin",
        "train_function_anchor_grad_ratio",
    ]:
        if name in train:
            values = pd.to_numeric(train[name], errors="coerce").dropna()
            if name in {"train_block_recon_grad_ratio", "train_block_direction_grad_ratio", "train_function_anchor_grad_ratio"}:
                values = values[values > 0.0]
            result[f"{name}_median"] = float(values.median()) if not values.empty else float("nan")
            result[f"{name}_final"] = float(values.iloc[-1]) if not values.empty else float("nan")
        else:
            result[f"{name}_median"] = float("nan")
            result[f"{name}_final"] = float("nan")
    for name in [
        "val_block_recon_loss",
        "val_block_recon_effective_loss",
        "val_block_recon_to_full_mse",
        "val_block_recon_rel_l2",
        "val_block_direction_loss",
        "val_block_direction_effective_loss",
        "val_block_direction_row_cos_mean",
        "val_block_direction_row_cos_min",
        "val_block_direction_row_error_mean",
        "val_block_direction_row_error_max",
        "val_block_direction_norm_ratio_mean",
    ]:
        if name in train:
            values = pd.to_numeric(train[name], errors="coerce").dropna()
            result[f"{name}_final"] = float(values.iloc[-1]) if not values.empty else float("nan")
        else:
            result[f"{name}_final"] = float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched Variant A fc2.weight reconstruction guard comparisons.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", action="append", default=[], help="Run only a named suffix from RUN_SPECS.")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    specs = _selected_specs(set(args.only) if args.only else None)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[variant_a_block_recon] start "
        f"specs={len(specs)} device={args.device} baseline_dir={BASELINE_DIR} "
        f"summary_out={args.summary_out}",
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
        block_coeff = float(spec["block_coeff"])
        grad_clip = float(spec.get("grad_clip", 0.0))
        head_coeff = float(spec.get("head_coeff", 0.0))
        head_margin = float(spec.get("head_margin", 0.0))
        head_huber_delta = float(spec.get("head_huber_delta", 0.05))
        head_top_fraction = float(spec.get("head_top_fraction", 0.25))
        head_example_top_fraction = float(spec.get("head_example_top_fraction", 1.0))
        head_loss_kind = str(spec.get("head_loss_kind", "head_ce_gap_hinge"))
        dir_coeff = float(spec.get("dir_coeff", 0.0))
        precond_pairs = int(spec.get("precond_pairs", 4))
        precond_loss_clip = float(spec.get("precond_loss_clip", 0.0))
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
            vae_precond_pairs=precond_pairs,
            vae_precond_batch_size=128,
            vae_precond_probe_scale=1.0,
            vae_precond_estimator_scope="local",
            vae_precond_hvp_mode="stopped_composite",
            vae_precond_loss_clip=precond_loss_clip,
            vae_precond_grad_damping=0.0,
            vae_precond_warmup_steps=0,
            vae_precond_ramp_steps=1000,
            vae_precond_max_grad_ratio=cap,
            vae_precond_grad_clip_norm=grad_clip,
            vae_block_recon_coeff=block_coeff,
            vae_block_recon_block="fc2.weight",
            vae_block_recon_space="normalized",
            vae_block_recon_ramp_steps=1000,
            vae_block_recon_grad_diagnostic=True,
            vae_function_anchor_coeff=head_coeff,
            vae_function_anchor_samples=8,
            vae_function_anchor_batch_size=256,
            vae_function_anchor_loss_kind=head_loss_kind,
            vae_function_anchor_block="fc2.weight",
            vae_function_anchor_ce_margin=head_margin,
            vae_function_anchor_huber_delta=head_huber_delta,
            vae_function_anchor_top_fraction=head_top_fraction,
            vae_function_anchor_example_top_fraction=head_example_top_fraction,
            vae_function_anchor_grad_diagnostic=True,
            vae_block_direction_coeff=dir_coeff,
            vae_block_direction_block="fc2.weight",
            vae_block_direction_space="denormalized",
            vae_block_direction_ramp_steps=1000,
            vae_block_direction_grad_diagnostic=True,
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
            "[variant_a_block_recon] run "
            f"{idx}/{len(specs)} suffix={suffix} precond={precond_kind} alpha={alpha:.6g} cap={cap:.6g} "
            f"grad_clip={grad_clip:.6g} block_coeff={block_coeff:.6g} block=fc2.weight space=normalized ramp_steps=1000 "
            f"precond_pairs={precond_pairs} precond_loss_clip={precond_loss_clip:.6g} "
            f"head_coeff={head_coeff:.6g} head_loss={head_loss_kind} head_margin={head_margin:.6g} "
            f"head_huber_delta={head_huber_delta:.6g} head_top_fraction={head_top_fraction:.6g} "
            f"dir_coeff={dir_coeff:.6g} dir_block=fc2.weight dir_space=denormalized dir_ramp_steps=1000 "
            f"run_label={run_label} output_dir={output_dir} config_hash={config_hash(cfg)}",
            flush=True,
        )
        print(f"[variant_a_block_recon] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}", flush=True)
        tables = run_or_load(cfg)
        row = _summary_for_run(tables.output_dir, suffix=suffix, alpha=alpha, cap=cap)
        row["block_coeff"] = block_coeff
        row["block"] = "fc2.weight"
        row["block_space"] = "normalized"
        row["head_coeff"] = head_coeff
        row["head_loss_kind"] = head_loss_kind
        row["head_margin"] = head_margin
        row["head_huber_delta"] = head_huber_delta
        row["head_top_fraction"] = head_top_fraction
        row["head_example_top_fraction"] = head_example_top_fraction
        row["dir_coeff"] = dir_coeff
        row["precond_pairs"] = precond_pairs
        row["precond_loss_clip"] = precond_loss_clip
        row["dir_block"] = "fc2.weight"
        row["dir_space"] = "denormalized"
        row.update(_training_summary(tables.output_dir))
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.summary_out, index=False)
        print(
            "[variant_a_block_recon] summary "
            f"suffix={suffix} decoder_aulc_mean={row['decoder_aulc_mean']:.6g} "
            f"decoder_aulc_median={row['decoder_aulc_median']:.6g} selected_decoder_lr={row['selected_decoder_lr']:.2g} "
            f"li_A_p95={row['li_A_full_per_dim_p95']:.6g} trace_m={row['hvp_probe_trace_m_per_dim_median']:.6g} "
            f"recon_rel_l2={row['reconstruction_rel_l2_median']:.6g} "
            f"block_eff_median={row.get('train_block_recon_effective_loss_median', float('nan')):.6g} "
            f"block_grad_ratio_median={row.get('train_block_recon_grad_ratio_median', float('nan')):.6g} "
            f"dir_eff_median={row.get('train_block_direction_effective_loss_median', float('nan')):.6g} "
            f"dir_cos_median={row.get('train_block_direction_row_cos_mean_median', float('nan')):.6g} "
            f"dir_grad_ratio_median={row.get('train_block_direction_grad_ratio_median', float('nan')):.6g} "
            f"head_eff_median={row.get('train_function_anchor_effective_loss_median', float('nan')):.6g} "
            f"head_ce_delta_median={row.get('train_function_anchor_ce_delta_median', float('nan')):.6g} "
            f"head_margin_drop_median={row.get('train_function_anchor_margin_drop_median', float('nan')):.6g} "
            f"head_grad_ratio_median={row.get('train_function_anchor_grad_ratio_median', float('nan')):.6g} "
            f"summary_out={args.summary_out}",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    print(f"[variant_a_block_recon] done elapsed_sec={elapsed:.2f} summary_out={args.summary_out}", flush=True)


if __name__ == "__main__":
    main()
