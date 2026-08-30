#!/usr/bin/env python3
"""Compare a live categorical GPTQ run with input-independent priors.

The target-count artifact is collected separately because opening the production
operator bank is expensive.  This script is deliberately cheap: it reads the
sealed counts and the append-only live JSONL logs, then evaluates the exact
Bayes-optimal input-independent distribution for the implemented weighted
cumulative ordinal objective.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_COUNTS = Path(
    "docs/report/parallel_categorical_gptq_prior_counts_20260829.json"
)
DEFAULT_METRICS = Path(
    "/mnt/shared/weightclip_benchmark/"
    "direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/"
    "train_metrics.jsonl"
)
DEFAULT_GRADIENTS = DEFAULT_METRICS.with_name("gradient_telemetry.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--gradients", type=Path, default=DEFAULT_GRADIENTS)
    parser.add_argument("--snapshot-step", type=int, default=1000)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize(counts: list[int]) -> list[float]:
    total = float(sum(counts))
    if total <= 0.0:
        raise ValueError("class counts must have positive mass")
    return [float(value) / total for value in counts]


def _weighted_ordinal_loss(probabilities: list[float], targets: list[float]) -> float:
    classes = len(probabilities)
    cdf: list[float] = []
    running = 0.0
    for probability in probabilities[:-1]:
        running += probability
        cdf.append(min(max(running, 1.0e-30), 1.0 - 1.0e-15))
    loss = 0.0
    for target_class, target_probability in enumerate(targets):
        per_target = 0.0
        for threshold, cumulative_probability in enumerate(cdf):
            weight = abs(2 * (threshold - target_class) + 1) / float(
                (classes - 1) ** 2
            )
            nll = (
                -math.log(cumulative_probability)
                if target_class <= threshold
                else -math.log1p(-cumulative_probability)
            )
            per_target += weight * nll
        loss += target_probability * per_target
    return loss


def _bayes_input_independent_distribution(targets: list[float]) -> list[float]:
    """Exact optimum for the target-dependent threshold weights in the run."""
    classes = len(targets)
    cdf: list[float] = []
    for threshold in range(classes - 1):
        positive = sum(
            targets[target] * abs(2 * (threshold - target) + 1)
            for target in range(threshold + 1)
        )
        negative = sum(
            targets[target] * abs(2 * (threshold - target) + 1)
            for target in range(threshold + 1, classes)
        )
        cdf.append(positive / (positive + negative))
    if any(right < left for left, right in zip(cdf, cdf[1:])):
        raise RuntimeError("unconstrained Bayes CDF is not monotone")
    edges = [0.0, *cdf, 1.0]
    return [right - left for left, right in zip(edges, edges[1:])]


def _hard_metrics(prediction: int, targets: list[float]) -> dict[str, float]:
    errors = [abs(index - prediction) for index in range(len(targets))]
    return {
        "accuracy": targets[prediction],
        "off_by_one_accuracy": sum(
            probability for probability, error in zip(targets, errors) if error <= 1
        ),
        "mean_absolute_bin_error": sum(
            probability * error for probability, error in zip(targets, errors)
        ),
        "mean_squared_bin_error": sum(
            probability * error * error
            for probability, error in zip(targets, errors)
        ),
    }


def _entropy(probabilities: list[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    ) / math.log(float(len(probabilities)))


def main() -> None:
    args = _parse_args()
    counts_payload = json.loads(args.counts.read_text(encoding="utf-8"))
    fit = _normalize(counts_payload["code_counts"]["fit"])
    heldout = _normalize(counts_payload["code_counts"]["heldout"])
    bayes = _bayes_input_independent_distribution(fit)
    empirical = fit
    uniform = [1.0 / len(fit)] * len(fit)
    bayes_class = max(range(len(bayes)), key=bayes.__getitem__)

    metric_rows = [
        row
        for row in _read_jsonl(args.metrics)
        if int(row["step"]) <= int(args.snapshot_step)
    ]
    if not metric_rows or int(metric_rows[-1]["step"]) != int(args.snapshot_step):
        raise RuntimeError("requested metric snapshot is unavailable")
    window_rows = metric_rows[-int(args.window) :]
    live_keys = [
        "code_ordinal_loss",
        "code_accuracy",
        "code_off_by_one_accuracy",
        "code_mean_absolute_bin_error",
        "code_mean_squared_bin_error",
        "code_prediction_entropy",
        "scale_ordinal_loss",
        "scale_accuracy",
        "scale_off_by_one_accuracy",
        "scale_mean_absolute_bin_error",
        "scale_mean_squared_bin_error",
        "scale_prediction_entropy",
        "hard_decode_raw_nrmse",
        "diagnostic_old_behavioral_direction",
        "diagnostic_old_structural_direction",
        "diagnostic_old_behavioral_operator_actual",
        "diagnostic_old_latent_interaction_rms_ratio",
        "diagnostic_old_latent_cross_std_median",
        "diagnostic_old_latent_within_std_median",
    ]
    live_window = {
        key: fmean(float(row[key]) for row in window_rows) for key in live_keys
    }

    gradient_rows = [
        row
        for row in _read_jsonl(args.gradients)
        if int(row["step"]) <= int(args.snapshot_step)
    ]
    if not gradient_rows or int(gradient_rows[-1]["step"]) != int(args.snapshot_step):
        raise RuntimeError("requested gradient snapshot is unavailable")
    latest_gradient = gradient_rows[-1]

    result = {
        "schema": "parallel_categorical_gptq_prior_review_v1",
        "snapshot_step": int(args.snapshot_step),
        "live_window_steps": [
            int(window_rows[0]["step"]),
            int(window_rows[-1]["step"]),
        ],
        "code_prior": {
            "heldout_zero_fraction": heldout[7],
            "uniform_ordinal_loss": _weighted_ordinal_loss(uniform, heldout),
            "empirical_marginal_ordinal_loss": _weighted_ordinal_loss(
                empirical, heldout
            ),
            "bayes_input_independent_ordinal_loss": _weighted_ordinal_loss(
                bayes, heldout
            ),
            "bayes_input_independent_probabilities": bayes,
            "bayes_input_independent_entropy": _entropy(bayes),
            "bayes_hard_class": bayes_class,
            "bayes_hard_class_signed_code": bayes_class - 7,
            "bayes_hard_metrics": _hard_metrics(bayes_class, heldout),
        },
        "scale_global_mode_heldout": counts_payload[
            "scale_baselines"
        ]["global_mode_heldout"],
        "live_window_mean": live_window,
        "step_1": {key: metric_rows[0][key] for key in live_keys},
        "step_gradient": latest_gradient,
        "validity": counts_payload["validity"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output.resolve())


if __name__ == "__main__":
    main()
