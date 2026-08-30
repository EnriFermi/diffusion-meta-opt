from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "baseline_training_h2048_m512"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the accepted h=2048 VAE's existing base-training loss history."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "vae_metrics.csv"

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = config_payload.get("config", config_payload)
    if int(config["vae_hidden_dim"]) != 2048 or int(config["latent_dim"]) != 512:
        raise ValueError("Expected the accepted hidden_dim=2048, latent_dim=512 baseline.")
    if str(config["vae_loss_kind"]) != "mse":
        raise ValueError("Expected the MSE VAE baseline.")

    print(
        "[plot_vae_baseline_training_components] mode=read_existing_metrics_only "
        f"run={run_dir} output={output_dir}"
    )
    metrics = pd.read_csv(metrics_path)
    history = metrics.loc[metrics["record_type"].eq("vae_train_history")].copy()
    history = history.sort_values("step").reset_index(drop=True)
    if history.empty:
        raise ValueError(f"No vae_train_history rows in {metrics_path}")

    beta_kl = float(config["beta_kl"])
    decomposition = pd.DataFrame({"step": history["step"]})
    for split in ("train", "val"):
        decomposition[f"{split}_total"] = history[f"{split}_loss"]
        decomposition[f"{split}_reconstruction_mse"] = history[f"{split}_recon_mse"]
        decomposition[f"{split}_weighted_kl"] = beta_kl * history[f"{split}_kl"]
        decomposition[f"{split}_recomposition_residual"] = (
            decomposition[f"{split}_total"]
            - decomposition[f"{split}_reconstruction_mse"]
            - decomposition[f"{split}_weighted_kl"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vae_baseline_training_components.csv"
    plot_path = output_dir / "vae_baseline_training_components.png"
    manifest_path = output_dir / "manifest.json"
    decomposition.to_csv(csv_path, index=False)

    colors = {"train": "#2563eb", "val": "#ea580c"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    panels = (
        ("total", "Total objective", True),
        ("reconstruction_mse", "Reconstruction MSE", True),
        ("weighted_kl", f"Weighted KL ({beta_kl:g} x KL)", False),
    )
    for axis, (component, title, log_scale) in zip(axes, panels, strict=True):
        for split in ("train", "val"):
            axis.plot(
                decomposition["step"],
                decomposition[f"{split}_{component}"],
                marker="o",
                markersize=3.5,
                linewidth=2,
                color=colors[split],
                label=split,
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("VAE training step")
        axis.set_ylabel("loss contribution")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Base VAE training: hidden_dim=2048, latent_dim=512, 150k steps\n"
        "Existing artifact history only; no fine-tuning"
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    residuals = {
        split: float(decomposition[f"{split}_recomposition_residual"].abs().max())
        for split in ("train", "val")
    }
    manifest = {
        "training_executed": False,
        "source_run": str(run_dir),
        "source_config": str(config_path),
        "source_metrics": str(metrics_path),
        "hidden_dim": int(config["vae_hidden_dim"]),
        "latent_dim": int(config["latent_dim"]),
        "vae_steps": int(config["vae_steps"]),
        "beta_kl": beta_kl,
        "history_rows": int(len(decomposition)),
        "max_abs_recomposition_residual": residuals,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "[plot_vae_baseline_training_components] done "
        f"rows={len(decomposition)} residuals={residuals} plot={plot_path} "
        f"csv={csv_path} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
