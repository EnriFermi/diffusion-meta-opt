from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing import celo_meta_config, run_or_load
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import config_hash, run_dir


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing/"
    "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
SOURCE_DATA_ROOT = "/home/coder/project/data"


FRONTIER_SPECS: tuple[dict[str, float | str], ...] = (
    {"suffix": "alpha0p0100_cap0p050", "alpha": 0.0100, "cap": 0.050},
    {"suffix": "alpha0p0100_cap0p100", "alpha": 0.0100, "cap": 0.100},
    {"suffix": "alpha0p0100_cap0p500", "alpha": 0.0100, "cap": 0.500},
    {"suffix": "alpha0p0100_cap1p000", "alpha": 0.0100, "cap": 1.000},
    {"suffix": "alpha0p0005_nocap", "alpha": 0.0005, "cap": 0.000},
    {"suffix": "alpha0p0010_nocap", "alpha": 0.0010, "cap": 0.000},
    {"suffix": "alpha0p0025_nocap", "alpha": 0.0025, "cap": 0.000},
    {"suffix": "alpha0p0050_nocap", "alpha": 0.0050, "cap": 0.000},
    {"suffix": "alpha0p0010_cap0p250", "alpha": 0.0010, "cap": 0.250},
    {"suffix": "alpha0p0025_cap0p250", "alpha": 0.0025, "cap": 0.250},
    {"suffix": "alpha0p0050_cap0p250", "alpha": 0.0050, "cap": 0.250},
    {"suffix": "alpha0p0200_cap0p250", "alpha": 0.0200, "cap": 0.250},
)


def _selected_specs(names: set[str] | None) -> tuple[dict[str, float | str], ...]:
    if not names:
        return FRONTIER_SPECS
    result = tuple(spec for spec in FRONTIER_SPECS if str(spec["suffix"]) in names)
    missing = sorted(names - {str(spec["suffix"]) for spec in result})
    if missing:
        raise ValueError(f"unknown frontier spec suffixes: {missing}")
    return result


def _summary_for_run(output_dir: Path, *, suffix: str, alpha: float, cap: float) -> dict[str, float | int | str]:
    vae_metrics = pd.read_csv(output_dir / "vae_metrics.csv")
    diagnostics = pd.read_csv(output_dir / "preconditioning_diagnostics.csv")
    downstream = pd.read_csv(output_dir / "downstream_results.csv")
    quality = vae_metrics[vae_metrics.get("record_type", "") == "vae_quality"].copy()
    train = vae_metrics[vae_metrics.get("record_type", "").fillna("") != "vae_quality"].copy()
    eval_dec = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "decoder_latent")]
    eval_raw = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "raw")]
    train_precond = train[pd.to_numeric(train.get("train_precond_a_loss"), errors="coerce").notna()].copy()
    result: dict[str, float | int | str] = {
        "suffix": suffix,
        "run_label": output_dir.name,
        "output_dir": str(output_dir),
        "alpha": float(alpha),
        "cap": float(cap),
        "decoder_eval_starts": int(eval_dec["source_weight_index"].nunique()) if "source_weight_index" in eval_dec else int(len(eval_dec)),
        "decoder_aulc_median": float(eval_dec["aulc"].median()) if not eval_dec.empty else float("nan"),
        "decoder_aulc_mean": float(eval_dec["aulc"].mean()) if not eval_dec.empty else float("nan"),
        "raw_aulc_median": float(eval_raw["aulc"].median()) if not eval_raw.empty else float("nan"),
        "decoded_test_loss_median": float(quality["decoded_test_loss"].median()) if "decoded_test_loss" in quality else float("nan"),
        "decoded_test_acc_median": float(quality["decoded_test_acc"].median()) if "decoded_test_acc" in quality else float("nan"),
        "raw_test_loss_median": float(quality["raw_test_loss"].median()) if "raw_test_loss" in quality else float("nan"),
        "raw_test_acc_median": float(quality["raw_test_acc"].median()) if "raw_test_acc" in quality else float("nan"),
        "reconstruction_rel_l2_median": float(quality["reconstruction_rel_l2"].median()) if "reconstruction_rel_l2" in quality else float("nan"),
        "val_recon_mse_final": float(pd.to_numeric(train["val_recon_mse"], errors="coerce").dropna().iloc[-1])
        if "val_recon_mse" in train and pd.to_numeric(train["val_recon_mse"], errors="coerce").notna().any()
        else float("nan"),
        "li_A_full_per_dim_median": float(diagnostics["li_A_full_per_dim"].median()) if "li_A_full_per_dim" in diagnostics else float("nan"),
        "li_A_full_per_dim_p95": float(diagnostics["li_A_full_per_dim"].quantile(0.95)) if "li_A_full_per_dim" in diagnostics else float("nan"),
        "hvp_probe_a_loss_per_dim_median": float(diagnostics["hvp_probe_a_loss_per_dim"].median())
        if "hvp_probe_a_loss_per_dim" in diagnostics
        else float("nan"),
        "hvp_probe_trace_m_per_dim_median": float(diagnostics["hvp_probe_trace_m_per_dim"].median())
        if "hvp_probe_trace_m_per_dim" in diagnostics
        else float("nan"),
        "train_precond_a_loss_median": float(train_precond["train_precond_a_loss"].median())
        if "train_precond_a_loss" in train_precond and not train_precond.empty
        else float("nan"),
        "train_precond_effective_loss_median": float(train_precond["train_precond_effective_loss"].median())
        if "train_precond_effective_loss" in train_precond and not train_precond.empty
        else float("nan"),
        "train_precond_grad_scale_median": float(train_precond["train_precond_grad_scale"].median())
        if "train_precond_grad_scale" in train_precond and not train_precond.empty
        else float("nan"),
        "train_precond_grad_norm_median": float(train_precond["train_precond_grad_norm"].median())
        if "train_precond_grad_norm" in train_precond and not train_precond.empty
        else float("nan"),
        "train_precond_base_grad_norm_median": float(train_precond["train_precond_base_grad_norm"].median())
        if "train_precond_base_grad_norm" in train_precond and not train_precond.empty
        else float("nan"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Variant A local alpha/cap frontier.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", action="append", default=[], help="Run only a named suffix from FRONTIER_SPECS.")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/frontier_runs_summary.csv")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    specs = _selected_specs(set(args.only) if args.only else None)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[variant_a_frontier] start "
        f"specs={len(specs)} device={args.device} baseline_dir={BASELINE_DIR} summary_out={args.summary_out}",
        flush=True,
    )
    rows: list[dict[str, float | int | str]] = []
    t0 = time.perf_counter()
    for idx, spec in enumerate(specs, start=1):
        suffix = str(spec["suffix"])
        alpha = float(spec["alpha"])
        cap = float(spec["cap"])
        run_label = f"sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_frontier_{suffix}_v1_seed0"
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
            vae_precond_loss_kind="li_a_hvp",
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
        print(
            "[variant_a_frontier] run "
            f"{idx}/{len(specs)} suffix={suffix} alpha={alpha:.6g} cap={cap:.6g} "
            f"run_label={run_label} output_dir={output_dir} config_hash={config_hash(cfg)}",
            flush=True,
        )
        print(f"[variant_a_frontier] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}", flush=True)
        tables = run_or_load(cfg)
        row = _summary_for_run(tables.output_dir, suffix=suffix, alpha=alpha, cap=cap)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.summary_out, index=False)
        print(
            "[variant_a_frontier] summary "
            f"suffix={suffix} decoder_aulc_median={row['decoder_aulc_median']:.6g} "
            f"li_A_p95={row['li_A_full_per_dim_p95']:.6g} recon_rel_l2={row['reconstruction_rel_l2_median']:.6g} "
            f"summary_out={args.summary_out}",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    print(f"[variant_a_frontier] done elapsed_sec={elapsed:.2f} summary_out={args.summary_out}", flush=True)


if __name__ == "__main__":
    main()
