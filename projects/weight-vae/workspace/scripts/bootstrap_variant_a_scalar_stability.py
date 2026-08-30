from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLITS = (
    ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)),
    ((0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11)),
    ((0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11)),
)
DEFAULT_STATES = (32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 12288, 16384, 24576, 32768)


def symmetric_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.abs(left) + np.abs(right)
    result = np.zeros_like(denominator, dtype=np.float64)
    np.divide(2.0 * np.abs(left - right), denominator, out=result, where=denominator > 0.0)
    return result


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    probability = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(probability * (1.0 - probability) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(center - half_width), float(center + half_width)


def evaluate_repeat_means(repeat_means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if repeat_means.ndim != 2 or repeat_means.shape[1] != 12:
        raise ValueError("repeat_means must have shape (trials, 12)")
    split_errors = []
    for left, right in SPLITS:
        split_errors.append(
            symmetric_error(repeat_means[:, list(left)].mean(axis=1), repeat_means[:, list(right)].mean(axis=1))
        )
    split_errors_array = np.stack(split_errors, axis=1)
    reference_pass = (split_errors_array <= 0.05).all(axis=1)

    total = repeat_means.sum(axis=1, keepdims=True)
    loro = (total - repeat_means) / 11.0
    repeat_errors = symmetric_error(repeat_means, loro)
    cell_median = np.quantile(repeat_errors, 0.5, axis=1, method="linear")
    cell_q90 = np.quantile(repeat_errors, 0.9, axis=1, method="linear")
    cell_pass = (cell_median <= 0.10) & (cell_q90 <= 0.20)
    return reference_pass, cell_pass, split_errors_array.max(axis=1), cell_q90


def bootstrap(
    state_values: np.ndarray,
    *,
    state_counts: tuple[int, ...],
    trials: int,
    seed: int,
    chunk_size: int,
) -> pd.DataFrame:
    values = np.asarray(state_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not bool(np.isfinite(values).all()):
        raise ValueError("state_values must be a finite vector with at least two values")
    probability = np.full(values.size, 1.0 / values.size, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for state_count in state_counts:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(state_count)]))
        reference_successes = 0
        cell_successes = 0
        both_successes = 0
        max_split_errors: list[np.ndarray] = []
        cell_q90_values: list[np.ndarray] = []
        completed = 0
        while completed < trials:
            current = min(int(chunk_size), int(trials - completed))
            repeat_means = np.empty((current, 12), dtype=np.float64)
            for repeat in range(12):
                counts = rng.multinomial(int(state_count), probability, size=current)
                repeat_means[:, repeat] = counts @ values / float(state_count)
            reference_pass, cell_pass, max_split_error, cell_q90 = evaluate_repeat_means(repeat_means)
            reference_successes += int(reference_pass.sum())
            cell_successes += int(cell_pass.sum())
            both_successes += int((reference_pass & cell_pass).sum())
            max_split_errors.append(max_split_error)
            cell_q90_values.append(cell_q90)
            completed += current
        max_split = np.concatenate(max_split_errors)
        q90_values = np.concatenate(cell_q90_values)
        both_low, both_high = wilson_interval(both_successes, trials)
        rows.append(
            {
                "states_per_repeat": int(state_count),
                "pairs_per_state": 32,
                "outer_repeats": 12,
                "bootstrap_trials": int(trials),
                "reference_gate_pass_probability": reference_successes / float(trials),
                "cell_scalar_gate_pass_probability": cell_successes / float(trials),
                "both_scalar_gates_pass_probability": both_successes / float(trials),
                "both_scalar_gates_wilson95_low": both_low,
                "both_scalar_gates_wilson95_high": both_high,
                "max_split_error_median": float(np.quantile(max_split, 0.5, method="linear")),
                "max_split_error_q90": float(np.quantile(max_split, 0.9, method="linear")),
                "cell_error_q90_median": float(np.quantile(q90_values, 0.5, method="linear")),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical planning bootstrap for the raw A scalar at P=32.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--states", default=",".join(str(value) for value in DEFAULT_STATES))
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    atomic = pd.read_csv(input_dir / "atomic_pair_samples.csv", usecols=["repeat", "state_position", "a_scalar"])
    state_values = (
        atomic.groupby(["repeat", "state_position"], sort=True)["a_scalar"].mean().to_numpy(dtype=np.float64)
    )
    if state_values.size != 384:
        raise ValueError(f"expected 384 P=32 state clusters, observed {state_values.size}")
    state_counts = tuple(sorted({int(value) for value in str(args.states).split(",")}))
    print(
        f"[a_scalar_bootstrap] start clusters={state_values.size} states={state_counts} "
        f"trials={args.trials} seed={args.seed}",
        flush=True,
    )
    result = bootstrap(
        state_values,
        state_counts=state_counts,
        trials=int(args.trials),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
    )
    csv_path = input_dir / "scalar_cluster_bootstrap_planning.csv"
    result.to_csv(csv_path, index=False)
    config = {
        "status": "conditional_planning_estimate_not_a_measured_stability_threshold",
        "cluster_source": "384 observed state means, each averaged over the primary P=32 pairs",
        "sampling": "nonparametric cluster bootstrap with replacement",
        "states": list(state_counts),
        "pairs": 32,
        "outer_repeats": 12,
        "trials": int(args.trials),
        "seed": int(args.seed),
        "reference_scalar_gate": "all three split-half symmetric errors <= 0.05",
        "cell_scalar_gate": "LORO repeat error median <= 0.10 and q90 <= 0.20",
        "limitations": [
            "conditional on the observed 384-state empirical tail",
            "does not resample probe/batch noise beyond each observed P=32 cluster mean",
            "does not estimate gradient-direction sample complexity",
        ],
    }
    config_path = input_dir / "scalar_cluster_bootstrap_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        result["states_per_repeat"],
        result["reference_gate_pass_probability"],
        marker="o",
        label="reference scalar gate",
    )
    ax.plot(
        result["states_per_repeat"],
        result["cell_scalar_gate_pass_probability"],
        marker="o",
        label="cell scalar gate",
    )
    ax.plot(
        result["states_per_repeat"],
        result["both_scalar_gates_pass_probability"],
        marker="o",
        label="both gates",
    )
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="90% planning level")
    ax.axhline(0.95, color="black", linestyle=":", linewidth=1, label="95% planning level")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("states per repeat at P=32")
    ax.set_ylabel("empirical bootstrap pass probability")
    ax.set_title("Conditional planning estimate for raw A scalar stability")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = input_dir / "scalar_cluster_bootstrap_planning.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"[a_scalar_bootstrap] done csv={csv_path} plot={plot_path}", flush=True)


if __name__ == "__main__":
    main()
