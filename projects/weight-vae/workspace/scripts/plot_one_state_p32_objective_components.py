#!/usr/bin/env python3
"""Plot the exact A and full-B components from the completed P32 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration5_p32_training_production"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration_5_p32_training"
    / "true_objective_component_breakdown.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curve_path = args.run_dir / "state_objective_curve.csv"
    config_path = args.run_dir / "resolved_config.json"
    curve = pd.read_csv(curve_path)
    config = json.loads(config_path.read_text())
    beta = float(config["beta"])

    required = {
        "proposal",
        "exact_a_per_dim",
        "damped_full_burg_per_dim",
        "true_objective",
    }
    missing = required.difference(curve.columns)
    if missing:
        raise ValueError(f"missing curve columns: {sorted(missing)}")

    proposal = curve["proposal"]
    a_loss = curve["exact_a_per_dim"]
    b_loss = curve["damped_full_burg_per_dim"]
    total = curve["true_objective"]
    reconstructed = a_loss + beta * b_loss
    reconstruction_error = float((reconstructed - total).abs().max())
    if reconstruction_error > 1e-10:
        raise ValueError(f"objective reconstruction error: {reconstruction_error}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    axes[0, 0].plot(proposal, a_loss, color="#d1495b", linewidth=2.2)
    axes[0, 0].set_title("Exact A component")
    axes[0, 0].set_ylabel("A per latent dimension")

    axes[0, 1].plot(proposal, b_loss, color="#00798c", linewidth=2.2)
    axes[0, 1].set_title("Full Burg B component")
    axes[0, 1].set_ylabel("B per latent dimension")

    axes[1, 0].plot(proposal, total, color="#3066be", linewidth=2.4, label="F")
    axes[1, 0].plot(
        proposal,
        beta * b_loss,
        color="#00798c",
        linewidth=1.8,
        label=f"beta * B (beta={beta:.4f})",
    )
    axes[1, 0].plot(proposal, a_loss, color="#d1495b", linewidth=1.8, label="A")
    axes[1, 0].set_title("Contributions to F = A + beta * B")
    axes[1, 0].set_ylabel("weighted objective contribution")
    axes[1, 0].legend()

    axes[1, 1].plot(proposal, a_loss / a_loss.iloc[0], color="#d1495b", label="A / A0")
    axes[1, 1].plot(proposal, b_loss / b_loss.iloc[0], color="#00798c", label="B / B0")
    axes[1, 1].plot(proposal, total / total.iloc[0], color="#3066be", label="F / F0")
    axes[1, 1].axhline(1.0, color="#555555", linewidth=1.0)
    axes[1, 1].set_title("Normalized component trajectories")
    axes[1, 1].set_ylabel("fraction of initial value")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("proposal")
        axis.set_xlim(float(proposal.min()), float(proposal.max()))

    fig.suptitle(
        "P32 one-state literal objective breakdown\n"
        f"A: {a_loss.iloc[0]:.4f} -> {a_loss.iloc[-1]:.4f}; "
        f"B: {b_loss.iloc[0]:.4f} -> {b_loss.iloc[-1]:.4f}; "
        f"F: {total.iloc[0]:.4f} -> {total.iloc[-1]:.4f}",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"curve={curve_path}")
    print(f"beta={beta}")
    print(f"max_reconstruction_error={reconstruction_error:.3e}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
