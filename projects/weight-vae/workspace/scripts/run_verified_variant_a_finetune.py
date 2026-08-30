from __future__ import annotations

import argparse
import hashlib
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
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
RUN_LABEL = "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_verified_nocap_v1_trainseed1"
DEFAULT_SUMMARY = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "verified_a_finetune_h2048"
    / "training_summary.csv"
)
EXPECTED_SOURCE_HASHES = {
    "vae_checkpoint.pt": "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397",
    "weight_pool.pt": "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef",
    "weight_pool_records.csv": "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sources() -> dict[str, str]:
    acceptance_path = BASELINE_DIR / "baseline_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not bool(acceptance.get("passed")):
        raise ValueError(f"accepted baseline gate failed in {acceptance_path}")
    observed: dict[str, str] = {}
    for filename, expected in EXPECTED_SOURCE_HASHES.items():
        path = BASELINE_DIR / filename
        observed[filename] = _sha256(path)
        if observed[filename] != expected:
            raise ValueError(
                f"source hash mismatch for {path}: observed={observed[filename]} expected={expected}"
            )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one provenance-bound, nominal-A CELO VAE fine-tune.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    source_hashes = _verify_sources()
    cfg = celo_meta_config(
        run_label=RUN_LABEL,
        seed=0,
        vae_training_seed=1,
        device=str(args.device),
        dtype="float32",
        data_root=str(ROOT / "data"),
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
        vae_precond_coeff=0.01,
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
        vae_precond_max_grad_ratio=0.0,
        vae_precond_grad_clip_norm=1.0,
        vae_block_recon_coeff=0.0,
        vae_function_anchor_coeff=0.0,
        vae_block_direction_coeff=0.0,
        geometry_eval_samples=24,
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
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[verified_a_finetune] start "
        f"run_label={RUN_LABEL} output_dir={output_dir} device={cfg.device} dtype={cfg.dtype} "
        f"seed={cfg.seed} vae_training_seed={cfg.vae_training_seed} cache_first={cfg.cache_first} "
        f"force_rerun={cfg.force_rerun} source_hashes={json.dumps(source_hashes, sort_keys=True)}",
        flush=True,
    )
    print(
        f"[verified_a_finetune] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}",
        flush=True,
    )
    started = time.perf_counter()
    tables = run_or_load(cfg)
    elapsed = time.perf_counter() - started

    history = tables.vae_metrics.loc[tables.vae_metrics["record_type"].eq("vae_train_history")].copy()
    precond = pd.to_numeric(history.get("train_precond_a_loss"), errors="coerce").dropna()
    diagnostics = tables.preconditioning_diagnostics
    exact_a = pd.to_numeric(diagnostics["li_A_full_per_dim"], errors="coerce").dropna()
    row = {
        "run_label": RUN_LABEL,
        "output_dir": str(tables.output_dir),
        "config_hash": config_hash(cfg),
        "elapsed_sec": elapsed,
        "vae_hidden_dim": int(cfg.vae_hidden_dim),
        "latent_dim": int(cfg.latent_dim),
        "vae_training_seed": int(cfg.vae_training_seed),
        "train_a_samples": int(len(precond)),
        "train_a_mean": float(precond.mean()),
        "train_a_median": float(precond.median()),
        "train_a_p90": float(precond.quantile(0.90)),
        "exact_a_rows": int(len(exact_a)),
        "exact_a_mean": float(exact_a.mean()),
        "exact_a_median": float(exact_a.median()),
        "exact_a_p90": float(exact_a.quantile(0.90)),
        "exact_a_max": float(exact_a.max()),
        "final_val_recon_mse": float(pd.to_numeric(history["val_recon_mse"], errors="coerce").dropna().iloc[-1]),
        "checkpoint_sha256": _sha256(tables.output_dir / "vae_checkpoint.pt"),
    }
    pd.DataFrame([row]).to_csv(args.summary_out, index=False)
    print(
        "[verified_a_finetune] done "
        f"elapsed_sec={elapsed:.2f} exact_a_mean={row['exact_a_mean']:.6g} "
        f"exact_a_p90={row['exact_a_p90']:.6g} final_val_recon_mse={row['final_val_recon_mse']:.6g} "
        f"checkpoint={tables.output_dir / 'vae_checkpoint.pt'} summary={args.summary_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
