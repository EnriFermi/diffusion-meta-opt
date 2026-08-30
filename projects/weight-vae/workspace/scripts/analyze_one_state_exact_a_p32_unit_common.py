from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048/iteration3_p32_unit_common_production"
)
OUTPUT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048/iteration_3_p32_unit_common/posthoc"
)
PHASES = ((1, 10), (11, 30), (31, 50), (51, 80), (81, 100))


def _failure_counts(values: pd.Series) -> Counter[str]:
    return Counter(
        failure
        for value in values.fillna("").astype(str)
        for failure in value.split("|")
        if failure
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(PRODUCTION / "state_metrics.csv")
    lines = pd.read_csv(PRODUCTION / "line_endpoints.csv")
    decision = json.loads((PRODUCTION / "decision.json").read_text(encoding="utf-8"))
    review = json.loads((PRODUCTION / "independent_review.json").read_text(encoding="utf-8"))

    failure_rows: list[dict[str, float | int | str]] = []
    for scope, frame in (
        ("all_evaluated_alphas", lines),
        ("alpha_1_over_64", lines.loc[np.isclose(lines["alpha"], 1.0 / 64.0)]),
    ):
        counts = _failure_counts(frame["failures"])
        for failure, count in sorted(counts.items()):
            failure_rows.append(
                {
                    "scope": scope,
                    "failure": failure,
                    "count": count,
                    "evaluated_rows": len(frame),
                    "fraction": count / max(len(frame), 1),
                }
            )
    pd.DataFrame(failure_rows).to_csv(OUTPUT / "failure_breakdown.csv", index=False)

    phase_rows: list[dict[str, float | int | str]] = []
    for start, end in PHASES:
        phase_lines = lines.loc[lines["proposal"].between(start, end)]
        smallest = phase_lines.loc[np.isclose(phase_lines["alpha"], 1.0 / 64.0)]
        counts = _failure_counts(smallest["failures"])
        phase_states = states.loc[states["proposal"].between(start, end)]
        phase_rows.append(
            {
                "proposal_start": start,
                "proposal_end": end,
                "accepted_count": int(phase_states["accepted"].sum()),
                "smallest_alpha_rows": len(smallest),
                "smallest_alpha_passes": int(smallest["passes"].sum()),
                "smallest_alpha_a_failures": counts["exact_a_not_lower"],
                "smallest_alpha_b_failures": counts["full_b_not_lower"],
                "smallest_alpha_mmax_failures": counts["m_max_increased"],
                "smallest_alpha_p50_failures": counts["m_p50_decreased"],
                "smallest_alpha_low_fraction_failures": counts["low_fraction_increased"],
            }
        )
    pd.DataFrame(phase_rows).to_csv(OUTPUT / "phase_breakdown.csv", index=False)

    selected_states = states.loc[
        states["proposal"].isin([0, 1, 2, 3, 4, 6, 15, 19, 25, 31, 51, 87, 100])
    ].copy()
    selected_states["m_lt_1e_4_count"] = selected_states["m_lt_1e_4_fraction"] * 512
    selected_states["m_lt_0p01_count"] = selected_states["m_lt_0p01_fraction"] * 512
    selected_states["m_lt_0p1_count"] = selected_states["m_lt_0p1_fraction"] * 512
    selected_states.to_csv(OUTPUT / "spectral_progress.csv", index=False)

    initial = states.iloc[0]
    final = states.iloc[-1]
    total_a_reduction = float(initial["exact_a_per_dim"] - final["exact_a_per_dim"])
    low90_reduction = float(
        initial["a_low90_abs_per_dim"] - final["a_low90_abs_per_dim"]
    )
    high_gt1_reduction = float(
        initial["a_high_gt_1_abs_per_dim"] - final["a_high_gt_1_abs_per_dim"]
    )
    top10_initial = float(initial["a_top10_share"] * initial["exact_a_per_dim"])
    top10_final = float(final["a_top10_share"] * final["exact_a_per_dim"])
    smallest = lines.loc[np.isclose(lines["alpha"], 1.0 / 64.0)]
    smallest_failures = _failure_counts(smallest["failures"])
    summary = {
        "producer_valid_flag": decision["valid"],
        "independent_review_valid_flag": review["valid"],
        "scientific_success": False,
        "initial_a": float(initial["exact_a_per_dim"]),
        "final_a": float(final["exact_a_per_dim"]),
        "initial_b": float(initial["damped_full_burg_per_dim"]),
        "final_b": float(final["damped_full_burg_per_dim"]),
        "accepted_count": int(states.loc[states["proposal"].gt(0), "accepted"].sum()),
        "total_a_reduction": total_a_reduction,
        "low90_a_reduction": low90_reduction,
        "low90_share_of_a_reduction": low90_reduction / total_a_reduction,
        "high_gt1_a_reduction": high_gt1_reduction,
        "high_gt1_share_of_a_reduction": high_gt1_reduction / total_a_reduction,
        "top10_a_reduction": top10_initial - top10_final,
        "top10_share_of_a_reduction": (top10_initial - top10_final) / total_a_reduction,
        "initial_median_eigenvalue": float(initial["m_p50"]),
        "final_median_eigenvalue": float(final["m_p50"]),
        "initial_count_lt_1e_4": int(round(float(initial["m_lt_1e_4_fraction"]) * 512)),
        "final_count_lt_1e_4": int(round(float(final["m_lt_1e_4_fraction"]) * 512)),
        "initial_count_lt_1e_2": int(round(float(initial["m_lt_0p01_fraction"]) * 512)),
        "final_count_lt_1e_2": int(round(float(final["m_lt_0p01_fraction"]) * 512)),
        "smallest_alpha_evaluations": len(smallest),
        "smallest_alpha_passes": int(smallest["passes"].sum()),
        "smallest_alpha_failure_counts": dict(sorted(smallest_failures.items())),
        "repeat_endpoint_max_errors": {
            column: float(lines[column].max())
            for column in lines
            if column.startswith("repeat_") and column.endswith("error")
        },
        "narrow_conclusion": (
            "The normalized P32 A+B repair removes the old fixed-beta sign failure only in an "
            "early spike-cutting phase. It does not optimize the lower spectral bulk: almost all "
            "A reduction comes from the top tail, and later P32 directions usually increase both "
            "exact A and m_max even at alpha=1/64. The unresolved discriminator is whether a "
            "low-mode lift direction is absent from the decoder parameter pullback or merely missed "
            "by the P32/common estimator."
        ),
        "next_single_discriminator": (
            "At the frozen final state, pull back one exact frozen-low-subspace matrix objective "
            "that increases trace in lambda<0.1 modes, then evaluate one fixed alpha line with A, "
            "B, m_max, p50, low-mode counts, and lower90 contribution."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
