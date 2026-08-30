from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "repeat",
    "state_position",
    "pair_position",
    "source_weight_index",
    "task_name",
    "tau",
    "a_scalar",
    "gradient_norm",
}


def symmetric_error(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = left.abs() + right.abs()
    result = 2.0 * (left - right).abs() / denominator
    return result.mask(denominator == 0.0, 0.0)


def quantile(values: pd.Series, probability: float) -> float:
    return float(np.quantile(values.to_numpy(dtype=np.float64), probability, method="linear"))


def _tail_rows(label: str, values: np.ndarray) -> list[dict[str, float | int | str]]:
    magnitudes = np.abs(np.asarray(values, dtype=np.float64))
    ordered = np.sort(magnitudes)[::-1]
    total = float(ordered.sum())
    rows: list[dict[str, float | int | str]] = []
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(np.ceil(ordered.size * fraction)))
        rows.append(
            {
                "quantity": label,
                "observation_count": int(ordered.size),
                "top_fraction": fraction,
                "top_count": count,
                "absolute_mass_share": float(ordered[:count].sum() / total) if total > 0.0 else 0.0,
                "median_absolute": float(np.median(magnitudes)),
                "mean_absolute": float(magnitudes.mean()),
                "q99_absolute": float(np.quantile(magnitudes, 0.99, method="linear")),
                "max_absolute": float(magnitudes.max()),
            }
        )
    return rows


def analyze(input_dir: Path) -> dict[str, Path]:
    atomic_path = input_dir / "atomic_pair_samples.csv"
    atomic = pd.read_csv(atomic_path)
    missing = sorted(REQUIRED_COLUMNS - set(atomic.columns))
    if missing:
        raise ValueError(f"atomic table is missing columns: {missing}")
    keys = ["repeat", "state_position", "pair_position"]
    if atomic.duplicated(keys).any():
        raise ValueError("atomic table has duplicate repeat/state/pair keys")
    repeats = sorted(atomic["repeat"].astype(int).unique().tolist())
    states = sorted(atomic["state_position"].astype(int).unique().tolist())
    pairs = sorted(atomic["pair_position"].astype(int).unique().tolist())
    if repeats != list(range(12)) or states != list(range(32)) or pairs != list(range(32)):
        raise ValueError(
            "this diagnostic expects the completed 12 x 32 x 32 primary bank; "
            f"observed repeats={len(repeats)} states={len(states)} pairs={len(pairs)}"
        )

    state_keys = ["repeat", "state_position"]
    scalar_wide = atomic.pivot(index=state_keys, columns="pair_position", values="a_scalar").sort_index(axis=1)
    scalar_reference = scalar_wide.mean(axis=1)
    prefix_rows: list[dict[str, float | int]] = []
    for pair_count in (1, 2, 4, 8, 16, 32):
        estimate = scalar_wide.iloc[:, :pair_count].mean(axis=1)
        error = symmetric_error(estimate, scalar_reference)
        prefix_rows.append(
            {
                "pairs": pair_count,
                "states_evaluated": int(len(estimate)),
                "scalar_symmetric_error_median": quantile(error, 0.50),
                "scalar_symmetric_error_q90": quantile(error, 0.90),
                "scalar_symmetric_error_q99": quantile(error, 0.99),
                "scalar_correlation_to_p32": float(estimate.corr(scalar_reference)),
            }
        )
    prefix = pd.DataFrame(prefix_rows)
    prefix_path = input_dir / "pair_prefix_scalar_stability.csv"
    prefix.to_csv(prefix_path, index=False)

    state_summary = (
        atomic.groupby(state_keys, sort=True)
        .agg(
            source_weight_index=("source_weight_index", "first"),
            task_name=("task_name", "first"),
            tau=("tau", "first"),
            a_scalar_mean=("a_scalar", "mean"),
            a_scalar_std=("a_scalar", "std"),
            atomic_gradient_norm_mean=("gradient_norm", "mean"),
            atomic_gradient_norm_max=("gradient_norm", "max"),
        )
        .reset_index()
    )
    state_path = input_dir / "state_tail_summary.csv"
    state_summary.to_csv(state_path, index=False)

    dominance_rows: list[dict[str, float | int | str]] = []
    for repeat, group in state_summary.groupby("repeat", sort=True):
        ordered = group.sort_values("atomic_gradient_norm_mean", ascending=False).reset_index(drop=True)
        total = float(ordered["atomic_gradient_norm_mean"].sum())
        dominance_rows.append(
            {
                "repeat": int(repeat),
                "top1_atomic_gradient_norm_mass_share": (
                    float(ordered.loc[0, "atomic_gradient_norm_mean"] / total) if total > 0.0 else 0.0
                ),
                "top3_atomic_gradient_norm_mass_share": (
                    float(ordered.loc[:2, "atomic_gradient_norm_mean"].sum() / total) if total > 0.0 else 0.0
                ),
                "top_state_position": int(ordered.loc[0, "state_position"]),
                "top_source_weight_index": int(ordered.loc[0, "source_weight_index"]),
                "top_task_name": str(ordered.loc[0, "task_name"]),
                "top_state_a_scalar_mean": float(ordered.loc[0, "a_scalar_mean"]),
                "top_state_atomic_gradient_norm_mean": float(ordered.loc[0, "atomic_gradient_norm_mean"]),
            }
        )
    dominance = pd.DataFrame(dominance_rows)
    dominance_path = input_dir / "repeat_state_dominance.csv"
    dominance.to_csv(dominance_path, index=False)

    task_rows: list[dict[str, float | int | str]] = []
    for task_name, group in state_summary.groupby("task_name", sort=True):
        task_rows.append(
            {
                "task_name": str(task_name),
                "state_count": int(len(group)),
                "a_scalar_mean": float(group["a_scalar_mean"].mean()),
                "a_scalar_median": float(group["a_scalar_mean"].median()),
                "a_scalar_q90": quantile(group["a_scalar_mean"], 0.90),
                "atomic_gradient_norm_mean": float(group["atomic_gradient_norm_mean"].mean()),
                "atomic_gradient_norm_median": float(group["atomic_gradient_norm_mean"].median()),
                "atomic_gradient_norm_q90": quantile(group["atomic_gradient_norm_mean"], 0.90),
            }
        )
    task = pd.DataFrame(task_rows)
    task_path = input_dir / "task_tail_summary.csv"
    task.to_csv(task_path, index=False)

    tail_rows: list[dict[str, float | int | str]] = []
    tail_rows.extend(_tail_rows("atomic_a_scalar", atomic["a_scalar"].to_numpy()))
    tail_rows.extend(_tail_rows("state_mean_a_scalar", state_summary["a_scalar_mean"].to_numpy()))
    tail_rows.extend(_tail_rows("atomic_gradient_norm", atomic["gradient_norm"].to_numpy()))
    tail_rows.extend(
        _tail_rows("state_mean_atomic_gradient_norm", state_summary["atomic_gradient_norm_mean"].to_numpy())
    )
    tails = pd.DataFrame(tail_rows)
    tails_path = input_dir / "tail_concentration.csv"
    tails.to_csv(tails_path, index=False)

    state_means = scalar_wide.mean(axis=1).to_numpy(dtype=np.float64)
    mean_within_pair_variance = float(scalar_wide.var(axis=1, ddof=1).mean())
    adjusted_between_state_variance = float(
        max(np.var(state_means, ddof=1) - mean_within_pair_variance / scalar_wide.shape[1], 0.0)
    )
    variance = {
        "description": "descriptive random-effects approximation; not a threshold verdict",
        "atomic_scalar_mean": float(scalar_wide.to_numpy(dtype=np.float64).mean()),
        "adjusted_between_state_variance": adjusted_between_state_variance,
        "mean_within_state_pair_variance": mean_within_pair_variance,
        "between_to_within_variance_ratio": (
            adjusted_between_state_variance / mean_within_pair_variance
            if mean_within_pair_variance > 0.0
            else None
        ),
        "spearman_tau_to_state_mean_a": float(state_summary["tau"].corr(state_summary["a_scalar_mean"], method="spearman")),
        "spearman_tau_to_state_mean_atomic_gradient_norm": float(
            state_summary["tau"].corr(state_summary["atomic_gradient_norm_mean"], method="spearman")
        ),
    }
    variance_path = input_dir / "state_pair_variance_decomposition.json"
    variance_path.write_text(json.dumps(variance, indent=2, sort_keys=True), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(prefix["pairs"], prefix["scalar_symmetric_error_median"], marker="o", label="median")
    axes[0].plot(prefix["pairs"], prefix["scalar_symmetric_error_q90"], marker="o", label="q90")
    axes[0].plot(prefix["pairs"], prefix["scalar_symmetric_error_q99"], marker="o", label="q99")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(prefix["pairs"], [str(value) for value in prefix["pairs"]])
    axes[0].set_xlabel("pairs for one fixed state")
    axes[0].set_ylabel("symmetric scalar error vs P=32")
    axes[0].set_title("Within-state pair convergence")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for task_name, group in state_summary.groupby("task_name", sort=True):
        values = np.sort(np.maximum(group["atomic_gradient_norm_mean"].to_numpy(dtype=np.float64), 1e-12))
        survival = 1.0 - np.arange(values.size, dtype=np.float64) / float(values.size)
        axes[1].step(values, survival, where="post", label=str(task_name))
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("mean atomic gradient norm for a state")
    axes[1].set_ylabel("empirical survival fraction")
    axes[1].set_title("State-level gradient-norm tails")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    plot_path = input_dir / "state_pair_tail_diagnostics.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "pair_prefix": prefix_path,
        "state_summary": state_path,
        "repeat_dominance": dominance_path,
        "task_summary": task_path,
        "tail_concentration": tails_path,
        "variance_decomposition": variance_path,
        "plot": plot_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc tail diagnostics for the Variant A stability bank.")
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    print(f"[a_stability_tail] stage=load input_dir={input_dir}", flush=True)
    outputs = analyze(input_dir)
    print(f"[a_stability_tail] done outputs={json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
