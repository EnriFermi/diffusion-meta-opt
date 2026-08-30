#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing import run_or_load
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    config_hash,
    run_dir,
)


DEFAULT_SOURCE = Path(
    "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing/"
    "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_marginhuber_c0p001_"
    "m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0"
)
DEFAULT_MANIFEST_DIR = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "common_state_rate_20260710/control_replicates"
)


def _load_config(source_run: Path) -> ExperimentConfig:
    payload = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    raw = payload.get("config", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid config payload in {source_run / 'config.json'}")
    return ExperimentConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one independent control-VAE replicate from an exact resolved config.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    source_run = args.source_run.expanduser().resolve()
    source_cfg = _load_config(source_run)
    source_label = str(source_cfg.run_label)
    if source_label.endswith("seed0"):
        run_label = source_label[: -len("seed0")] + f"methodrep_seed{int(args.seed)}"
    else:
        run_label = source_label + f"_methodrep_seed{int(args.seed)}"
    cfg = replace(
        source_cfg,
        seed=int(source_cfg.seed),
        vae_training_seed=int(args.seed),
        run_label=run_label,
        device=str(args.device),
        dtype="float32",
        show_progress=True,
        progress_backend="text",
        force_rerun=bool(args.force_rerun),
        comet_enabled=False,
    )
    output_dir = run_dir(cfg).resolve()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_dir / f"control_methodrep_seed{int(args.seed)}.json"
    print(
        "[control_vae_replicate] startup "
        f"base_seed={cfg.seed} vae_training_seed={cfg.vae_training_seed} device={cfg.device} dtype={cfg.dtype} "
        f"cache_first={cfg.cache_first} "
        f"source_run={source_run} output_dir={output_dir} config_hash={config_hash(cfg)} "
        f"manifest={manifest_path}",
        flush=True,
    )
    print(f"[control_vae_replicate] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}", flush=True)
    print("[control_vae_replicate] stage=run_or_load", flush=True)
    tables = run_or_load(cfg)
    selected_path = tables.output_dir / "selected_lrs.csv"
    selected = pd.read_csv(selected_path)
    selected_rows = selected[pd.to_numeric(selected.get("selected", 0), errors="coerce").fillna(0).astype(int) == 1]
    summary = {
        "seed": int(cfg.seed),
        "vae_training_seed": int(cfg.vae_training_seed),
        "source_run": str(source_run),
        "run_label": str(cfg.run_label),
        "output_dir": str(tables.output_dir.resolve()),
        "config_hash": config_hash(cfg),
        "device": str(cfg.device),
        "dtype": str(cfg.dtype),
        "vae_steps": int(cfg.vae_steps),
        "latent_dim": int(cfg.latent_dim),
        "vae_hidden_dim": int(cfg.vae_hidden_dim),
        "selected_lrs": selected_rows.to_dict(orient="records"),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[control_vae_replicate] done "
        f"elapsed_sec={summary['elapsed_sec']:.2f} output_dir={summary['output_dir']} "
        f"selected_lrs={selected_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
