from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cross_repeat_signal_estimate(gram: np.ndarray) -> float:
    matrix = np.asarray(gram, dtype=np.float64)
    count = int(matrix.shape[0])
    if matrix.shape != (count, count) or count < 3 or not bool(np.isfinite(matrix).all()):
        raise ValueError("gram must be a finite square matrix with at least three repeats")
    return float((matrix.sum() - np.trace(matrix)) / float(count * (count - 1)))


def jackknife_signal(gram: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(gram, dtype=np.float64)
    count = int(matrix.shape[0])
    estimate = cross_repeat_signal_estimate(matrix)
    leave_one_out = np.asarray(
        [cross_repeat_signal_estimate(np.delete(np.delete(matrix, index, axis=0), index, axis=1)) for index in range(count)],
        dtype=np.float64,
    )
    standard_error = float(np.sqrt((count - 1.0) / count * np.sum((leave_one_out - leave_one_out.mean()) ** 2)))
    return (
        estimate,
        standard_error,
        float(estimate - 1.959963984540054 * standard_error),
        float(estimate + 1.959963984540054 * standard_error),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess whether the mean raw A gradient signal is identifiable.")
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    gram_bank = np.load(input_dir / "repeat_prefix_gram.npz")
    rows: list[dict[str, float | int | bool]] = []
    for pair_count in (1, 2, 4):
        gram = gram_bank[f"gram_p{pair_count}"]
        estimate, standard_error, lower, upper = jackknife_signal(gram)
        mean_squared_norm = float(np.diag(gram).mean())
        rows.append(
            {
                "states": 32,
                "pairs": pair_count,
                "outer_repeats": int(gram.shape[0]),
                "cross_repeat_signal_squared_estimate": estimate,
                "jackknife_standard_error": standard_error,
                "normal95_lower": lower,
                "normal95_upper": upper,
                "mean_repeat_gradient_squared_norm": mean_squared_norm,
                "point_signal_fraction_of_mean_squared_norm": estimate / mean_squared_norm,
                "positive_signal_identified_at_normal95": bool(lower > 0.0),
            }
        )
    signal = pd.DataFrame(rows)
    signal_path = input_dir / "gradient_signal_identifiability.csv"
    signal.to_csv(signal_path, index=False)

    components = pd.read_csv(input_dir / "variance_components.csv")
    full = components.loc[(components["scope_id"] == "full") & (components["task_group"] == "all")].iloc[0]
    between = float(full["between_state_covariance_trace_B_raw"])
    within = float(full["within_pair_covariance_trace_W"])
    signal_p4 = float(signal.loc[signal["pairs"] == 4, "cross_repeat_signal_squared_estimate"].iloc[0])
    planning_rows: list[dict[str, float | int | str]] = []
    for pair_label, pair_count in (("4", 4.0), ("32", 32.0), ("infinite", float("inf"))):
        per_state_noise = between + (0.0 if np.isinf(pair_count) else within / pair_count)
        for target_cosine in (0.80, 0.90, 0.95, 0.99):
            required_states = (
                per_state_noise / (signal_p4 * (1.0 / target_cosine - 1.0)) if signal_p4 > 0.0 else float("nan")
            )
            planning_rows.append(
                {
                    "pairs": pair_label,
                    "target_expected_cosine": target_cosine,
                    "per_state_noise_trace": per_state_noise,
                    "point_required_states": required_states,
                    "status": "non_identifiable_planning_point_estimate",
                }
            )
    planning = pd.DataFrame(planning_rows)
    planning_path = input_dir / "gradient_sample_complexity_planning.csv"
    planning.to_csv(planning_path, index=False)
    config = {
        "status": "planning_only_not_a_stability_threshold",
        "signal_estimator": "mean off-diagonal dot product of 12 independent repeat gradients",
        "uncertainty": "delete-one-repeat jackknife with a descriptive normal interval",
        "planning_model": "expected cosine = signal_sq / (signal_sq + (B + W/P)/S)",
        "limitations": [
            "the P=4 signal interval includes zero",
            "the additive covariance-trace model does not encode heavy-tail or deterministic-window dependence",
            "point required-state values are not statistically identified",
        ],
    }
    config_path = input_dir / "gradient_signal_planning_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[a_gradient_signal] done signal={signal_path} planning={planning_path} config={config_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
