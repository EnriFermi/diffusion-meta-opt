from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_a_full_burg_h2048/iteration3_naive_semantics_production"
)
EXPECTED_PROTOCOL = "one_state_a_full_burg_naive_semantics_iteration3_v1"
EXPECTED_RUNNER_SHA256 = "ad5ae8bf4872c21a5bd37d636d0ad2708eab20962b94e42fd615fd05496a2bed"
EXPECTED_NORMALIZED_RUNNER_SHA256 = "7ad8b95113b7ed84defc53f7b1f72dc50f17455f747b491396cbed2d52a3ed02"
EXPECTED_FROZEN_MANIFEST_SHA256 = "185c5f26d4335297677283ab84492e56cad914893935261cb919511b2421ffc9"
EXPECTED_BETA = 22.536727828943093
EXPECTED_TARGET = 139.69206097331205
EXPECTED_OLD_BEST = 140.18711003198487
EXPECTED_CONTROL_HASHES = {
    "iteration1_states": "2f71b677b130232b65d067e867f0a9c2e05be040d15d24bbae5e2b5c8b53b082",
    "iteration1_proposals": "d3e03793da9ac86d3c7538e7c46886679950ca73968a1ff028cbf3bb434c650b",
    "iteration1_lines": "a5fbe07fda0ee3f2503a349ce9f5999d234ea4a2ac54ecfa0afdb39e977e2c3e",
}
LINE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
SLOPE_STEPS = (1.0 / 128.0, 1.0 / 256.0)
STRICT_TOLERANCE = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy()).all())


def max_zero_run(values: list[int]) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value == 0 else 0
        best = max(best, current)
    return best


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir

    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    states = pd.read_csv(run_dir / "state_objective_curve.csv")
    proposals = pd.read_csv(run_dir / "proposal_diagnostics.csv")
    lines = pd.read_csv(run_dir / "line_profiles.csv")
    forward = pd.read_csv(run_dir / "forward_equivalence.csv")
    forward_scalars = pd.read_csv(run_dir / "proposal1_forward_scalars.csv")
    stopped_reproduction = pd.read_csv(run_dir / "stopped_proposal1_reproduction.csv")
    control_equivalence = pd.read_csv(run_dir / "control_acceptance_equivalence.csv")

    checks: dict[str, bool] = {}
    checks["protocol_matches"] = bool(
        config["protocol_id"] == summary["protocol_id"] == manifest["protocol_id"] == EXPECTED_PROTOCOL
    )
    source_snapshot_hash = sha256_file(run_dir / "executed_source_snapshot.py")
    checks["source_snapshot_hash_matches"] = bool(
        source_snapshot_hash
        == config["source_sha256"]
        == summary["source_sha256"]
        == manifest["source_sha256"]
        == EXPECTED_RUNNER_SHA256
    )
    frozen_manifest_path = Path(config["frozen_dependency_manifest"])
    checks["normalized_source_and_frozen_manifest_match"] = bool(
        config["normalized_source_sha256"]
        == summary["normalized_source_sha256"]
        == EXPECTED_NORMALIZED_RUNNER_SHA256
        and config["frozen_dependency_manifest_sha256"]
        == summary["frozen_dependency_manifest_sha256"]
        == EXPECTED_FROZEN_MANIFEST_SHA256
        and sha256_file(frozen_manifest_path) == EXPECTED_FROZEN_MANIFEST_SHA256
        and config["hard_freeze_valid"]
        and all(config["frozen_dependency_matches"].values())
    )
    checks["frozen_config"] = bool(
        config["frozen_setup"]
        and config["run_mode"] == "production_evidence"
        and config["hvp_mode_executed"] == "autograd"
        and int(config["proposals"]) == 100
        and int(config["pairs"]) == 4
        and close(config["beta"], EXPECTED_BETA, 0.0)
        and close(config["lr"], 3e-5, 0.0)
        and close(config["gradient_clip"], 1.0, 0.0)
        and close(config["burg_epsilon"], 1e-4, 0.0)
        and close(config["armijo_c1"], 1e-4, 0.0)
    )
    checks["control_dependency_hashes_match"] = all(
        config["dependency_sha256"].get(name) == expected
        for name, expected in EXPECTED_CONTROL_HASHES.items()
    )
    checks["manifest_hashes_match"] = all(
        (run_dir / name).is_file() and sha256_file(run_dir / name) == expected
        for name, expected in manifest["artifacts"].items()
    )
    checks["expected_row_counts"] = bool(
        len(states) == 101
        and len(proposals) == 100
        and int(lines["kind"].eq("line").sum()) == 700
        and int(lines["kind"].str.startswith("fd_").sum()) == 400
    )
    checks["unique_ordered_indices"] = bool(
        states["proposal"].tolist() == list(range(101))
        and proposals["proposal"].tolist() == list(range(1, 101))
        and not lines.duplicated(["proposal", "kind", "alpha"]).any()
    )
    checks["all_numeric_frames_finite"] = all(
        finite(frame)
        for frame in (
            states,
            proposals,
            lines,
            forward,
            forward_scalars,
            stopped_reproduction,
            control_equivalence,
        )
    )

    state_formula_error = float(
        np.max(
            np.abs(
                states["true_objective"].to_numpy()
                - (
                    states["exact_a_per_dim"].to_numpy()
                    + EXPECTED_BETA * states["damped_full_burg_per_dim"].to_numpy()
                )
            )
        )
    )
    checks["state_objective_formula_matches"] = state_formula_error <= 1e-10

    slope_errors: list[float] = []
    alpha_errors: list[float] = []
    transition_errors: list[float] = []
    recomputed_alphas: list[float] = []
    for proposal_id in range(1, 101):
        diagnostic = proposals.loc[proposals["proposal"].eq(proposal_id)].iloc[0]
        before_f = float(states.loc[states["proposal"].eq(proposal_id - 1), "true_objective"].iloc[0])
        after_f = float(states.loc[states["proposal"].eq(proposal_id), "true_objective"].iloc[0])
        transition_errors.extend(
            [
                abs(float(diagnostic["current_f"]) - before_f),
                abs(float(diagnostic["accepted_f"]) - after_f),
                abs(float(diagnostic["accepted_delta_f"]) - (after_f - before_f)),
            ]
        )
        proposal_lines = lines.loc[lines["proposal"].eq(proposal_id)]
        slopes: dict[float, float] = {}
        for step, column in ((SLOPE_STEPS[0], "slope_h128"), (SLOPE_STEPS[1], "slope_h256")):
            positive = float(
                proposal_lines.loc[
                    proposal_lines["kind"].eq("fd_positive") & np.isclose(proposal_lines["alpha"], step),
                    "true_objective",
                ].iloc[0]
            )
            negative = float(
                proposal_lines.loc[
                    proposal_lines["kind"].eq("fd_negative") & np.isclose(proposal_lines["alpha"], -step),
                    "true_objective",
                ].iloc[0]
            )
            slopes[step] = (positive - negative) / (2.0 * step)
            slope_errors.append(abs(slopes[step] - float(diagnostic[column])))

        selected = 0.0
        if all(slopes[step] < 0.0 for step in SLOPE_STEPS):
            for alpha in LINE_ALPHAS:
                candidate_f = float(
                    proposal_lines.loc[
                        proposal_lines["kind"].eq("line") & np.isclose(proposal_lines["alpha"], alpha),
                        "true_objective",
                    ].iloc[0]
                )
                if (
                    candidate_f <= before_f + float(config["armijo_c1"]) * alpha * slopes[SLOPE_STEPS[1]]
                    and candidate_f < before_f - STRICT_TOLERANCE
                ):
                    selected = alpha
                    break
        recomputed_alphas.append(selected)
        alpha_errors.append(abs(selected - float(diagnostic["accepted_alpha"])))
        if selected > 0.0:
            selected_f = float(
                proposal_lines.loc[
                    proposal_lines["kind"].eq("line") & np.isclose(proposal_lines["alpha"], selected),
                    "true_objective",
                ].iloc[0]
            )
            transition_errors.append(abs(selected_f - after_f))
        else:
            transition_errors.append(abs(before_f - after_f))

    checks["fd_slopes_recompute"] = max(slope_errors) <= 1e-10
    checks["armijo_alphas_recompute"] = max(alpha_errors) <= 1e-12
    checks["state_transitions_recompute"] = max(transition_errors) <= 1e-10
    checks["state_curve_is_nonincreasing"] = bool(
        np.all(np.diff(states["true_objective"].to_numpy()) <= STRICT_TOLERANCE)
    )

    rejection_rows: list[dict[str, Any]] = []
    for _, diagnostic in proposals.loc[proposals["accepted"].eq(0)].iterrows():
        proposal_id = int(diagnostic["proposal"])
        if float(diagnostic["slope_h128"]) >= 0.0 and float(diagnostic["slope_h256"]) >= 0.0:
            rejection_class = "uphill_both_fd_scales"
        elif np.sign(float(diagnostic["slope_h128"])) != np.sign(float(diagnostic["slope_h256"])):
            rejection_class = "fd_sign_unstable"
        else:
            rejection_class = "negative_slope_no_candidate"
        candidate_rows = lines.loc[
            lines["proposal"].eq(proposal_id) & lines["kind"].eq("line")
        ]
        minimum_index = candidate_rows["delta_f"].idxmin()
        rejection_rows.append(
            {
                "proposal": proposal_id,
                "current_f": float(diagnostic["current_f"]),
                "accepted_adam_step_before": int(diagnostic["accepted_adam_step_before"]),
                "slope_h128": float(diagnostic["slope_h128"]),
                "slope_h256": float(diagnostic["slope_h256"]),
                "rejection_class": rejection_class,
                "minimum_line_delta_f": float(candidate_rows.loc[minimum_index, "delta_f"]),
                "minimum_line_alpha": float(candidate_rows.loc[minimum_index, "alpha"]),
            }
        )
    rejection_frame = pd.DataFrame(rejection_rows)
    rejection_frame.to_csv(run_dir / "rejection_classification.csv", index=False)
    plateau_columns = [
        "proposal",
        "current_f",
        "accepted_adam_step_before",
        "slope_h128",
        "slope_h256",
        "accepted_alpha",
        "train_a",
        "train_b_pseudo_loss",
        "grad_a_b_cosine",
        "grad_total_norm",
        "clip_factor",
    ]
    proposals.loc[proposals["proposal"].between(59, 67), plateau_columns].to_csv(
        run_dir / "fixed_state_plateau_59_67.csv", index=False
    )
    checks["control_equivalence_is_exact"] = bool(
        len(control_equivalence) == 100
        and int(control_equivalence["match"].sum()) == 100
        and int((control_equivalence["fd_armijo_accepted_alpha"] > 0.0).sum()) == 25
    )
    checks["runner_validity_gates_pass"] = bool(summary["valid"] and all(summary["validity_gates"].values()))

    accepted = proposals["accepted"].astype(int)
    accepted_rows = proposals.loc[accepted.eq(1)]
    final_f = float(states.iloc[-1]["true_objective"])
    strict_decreases = int((np.diff(states["true_objective"].to_numpy()) < -STRICT_TOLERANCE).sum())
    rejected = int((accepted == 0).sum())
    negative_both = int(
        ((proposals["slope_h128"] < 0.0) & (proposals["slope_h256"] < 0.0)).sum()
    )
    slope_sign_agreements = int(
        (np.sign(proposals["slope_h128"]) == np.sign(proposals["slope_h256"])).sum()
    )
    relative_slope_disagreement = (
        (proposals["slope_h128"] - proposals["slope_h256"]).abs()
        / np.maximum(proposals["slope_h128"].abs(), proposals["slope_h256"].abs()).clip(lower=1e-30)
    )
    median_alpha = float(accepted_rows["accepted_alpha"].median()) if len(accepted_rows) else 0.0
    path_fraction = float(
        proposals["accepted_displacement_norm"].sum()
        / max(float(proposals["proposal_norm"].sum()), 1e-30)
    )
    recomputed_success_gates = {
        "final_f_at_most_frozen_20pct_target": final_f <= EXPECTED_TARGET,
        "beats_frozen_old_best": final_f < EXPECTED_OLD_BEST,
        "at_least_90_strict_decreases": strict_decreases >= 90,
        "at_most_10_rejections": rejected <= 10,
        "no_accepted_increases": bool((accepted_rows["accepted_delta_f"] < -STRICT_TOLERANCE).all()),
        "maximum_rejection_run_at_most_3": max_zero_run(accepted.tolist()) <= 3,
        "at_least_90_negative_slope_proposals": negative_both >= 90,
        "at_least_95_slope_sign_agreements": slope_sign_agreements >= 95,
        "median_slope_disagreement_at_most_0p10": float(relative_slope_disagreement.median()) <= 0.10,
        "p95_slope_disagreement_at_most_0p25": float(relative_slope_disagreement.quantile(0.95)) <= 0.25,
        "median_alpha_at_least_one_eighth": median_alpha >= 1.0 / 8.0,
        "path_fraction_at_least_0p25": path_fraction >= 0.25,
    }
    checks["success_gates_match_runner"] = bool(recomputed_success_gates == summary["success_gates"])
    all_valid = bool(all(checks.values()))
    success = bool(all(recomputed_success_gates.values())) if all_valid else None
    review: dict[str, Any] = {
        "protocol_id": EXPECTED_PROTOCOL,
        "all_posthoc_validity_checks_pass": all_valid,
        "all_recomputed_success_gates_pass": success,
        "checks": checks,
        "recomputed_success_gates": recomputed_success_gates if all_valid else None,
        "metrics": {
            "initial_f": float(states.iloc[0]["true_objective"]),
            "final_f": final_f,
            "strict_decreases": strict_decreases,
            "rejected": rejected,
            "negative_both": negative_both,
            "slope_sign_agreements": slope_sign_agreements,
            "median_slope_disagreement": float(relative_slope_disagreement.median()),
            "p95_slope_disagreement": float(relative_slope_disagreement.quantile(0.95)),
            "median_accepted_alpha": median_alpha,
            "path_fraction": path_fraction,
            "max_state_formula_error": state_formula_error,
            "max_slope_recompute_error": max(slope_errors),
            "max_alpha_recompute_error": max(alpha_errors),
            "max_transition_error": max(transition_errors),
            "uphill_both_fd_rejections": int(
                rejection_frame["rejection_class"].eq("uphill_both_fd_scales").sum()
            ),
            "fd_sign_unstable_rejections": int(
                rejection_frame["rejection_class"].eq("fd_sign_unstable").sum()
            ),
            "negative_slope_no_candidate_rejections": int(
                rejection_frame["rejection_class"].eq("negative_slope_no_candidate").sum()
            ),
        },
    }
    output = run_dir / "posthoc_validation.json"
    output.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(review, indent=2, sort_keys=True))
    print(f"wrote {output}")
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
