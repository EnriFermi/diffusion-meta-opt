from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / (
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "one_state_a_full_burg_h2048"
)
RUN_DIR = OUTPUT_ROOT / "iteration5_p32_training_production"
FROZEN_MANIFEST = OUTPUT_ROOT / "iteration_5_frozen_dependency_manifest.json"
EXPECTED_PROTOCOL = "one_state_a_full_burg_p32_training_iteration5_v1"
EXPECTED_BETA = 22.536727828943093
EXPECTED_TARGET = 139.69206097331205
EXPECTED_OLD_BEST = 140.18711003198487
EXPECTED_Z = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
LINE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
SLOPE_STEPS = (1.0 / 128.0, 1.0 / 256.0)
STRICT_TOLERANCE = 1e-8
POOL_DRAWS = 8
PAIRS_PER_DRAW = 4
A_ONLY_PROTOCOL_ID = "one_state_a_optimization_smoke_h2048_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uint63(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & (
        (1 << 63) - 1
    )


def seed_schedule_hash(rows: list[dict[str, int]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_runner_hash(path: Path) -> str:
    masked_prefixes = (
        "EXPECTED_FROZEN_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    normalized: list[str] = []
    for line in lines:
        prefix = next((candidate for candidate in masked_prefixes if line.startswith(candidate)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def named_tensor_hash(names: Sequence[str], values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in zip(names, values, strict=True):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def finite(frame: pd.DataFrame) -> bool:
    return bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all())


def max_zero_run(values: list[int]) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value == 0 else 0
        best = max(best, current)
    return best


def main() -> None:
    run = RUN_DIR
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    artifact_manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((run / "COMPLETED.json").read_text(encoding="utf-8"))
    frozen_manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    pooling = json.loads((run / "p32_pooling_preflight.json").read_text(encoding="utf-8"))
    seed_review = json.loads((run / "p32_seed_schedule_preflight.json").read_text(encoding="utf-8"))
    states = pd.read_csv(run / "state_objective_curve.csv")
    proposals = pd.read_csv(run / "proposal_diagnostics.csv")
    lines = pd.read_csv(run / "line_profiles.csv")
    pairs = pd.read_csv(run / "p32_pair_scalars.csv")
    draws = pd.read_csv(run / "p32_draw_diagnostics.csv")
    forward = pd.read_csv(run / "forward_equivalence.csv")
    forward_scalars = pd.read_csv(run / "proposal1_forward_scalars.csv")
    stopped = pd.read_csv(run / "stopped_proposal1_reproduction.csv")
    control = pd.read_csv(run / "control_acceptance_equivalence.csv")

    checks: dict[str, bool] = {}
    checks["protocol_matches"] = bool(
        config["protocol_id"]
        == summary["protocol_id"]
        == artifact_manifest["protocol_id"]
        == frozen_manifest["protocol_id"]
        == EXPECTED_PROTOCOL
    )
    snapshot = run / "executed_source_snapshot.py"
    snapshot_hash = sha256_file(snapshot)
    snapshot_normalized = normalized_runner_hash(snapshot)
    checks["runner_source_chain_matches"] = bool(
        snapshot_hash
        == config["source_sha256"]
        == summary["source_sha256"]
        == artifact_manifest["source_sha256"]
        and snapshot_normalized
        == config["normalized_source_sha256"]
        == summary["normalized_source_sha256"]
        == artifact_manifest["normalized_source_sha256"]
        == frozen_manifest["runner_normalized_sha256"]
    )
    frozen_manifest_hash = sha256_file(FROZEN_MANIFEST)
    checks["frozen_manifest_chain_matches"] = bool(
        frozen_manifest_hash
        == config["frozen_dependency_manifest_sha256"]
        == summary["frozen_dependency_manifest_sha256"]
        and config["hard_freeze_valid"]
        and all(config["frozen_dependency_matches"].values())
    )
    dependency_matches = {
        relative: sha256_file(ROOT / relative) == expected
        for relative, expected in frozen_manifest["dependencies"].items()
    }
    reviewer_relative = "scripts/review_one_state_a_full_burg_p32_training.py"
    checks["all_frozen_dependencies_and_reviewer_match"] = bool(
        all(dependency_matches.values())
        and reviewer_relative in frozen_manifest["dependencies"]
        and dependency_matches[reviewer_relative]
    )
    checks["artifact_hashes_match"] = all(
        (run / name).is_file() and sha256_file(run / name) == expected
        for name, expected in artifact_manifest["artifacts"].items()
    )
    checks["atomic_completion_marker_matches"] = bool(
        not (run / "INCOMPLETE").exists()
        and completion["protocol_id"] == EXPECTED_PROTOCOL
        and completion["status"] == "production_complete_awaiting_independent_replay"
        and completion["valid"] is True
        and completion["summary_sha256"] == sha256_file(run / "summary.json")
        and completion["artifact_manifest_sha256"] == sha256_file(run / "artifact_manifest.json")
    )
    checks["frozen_config_matches"] = bool(
        config["frozen_setup"]
        and config["run_mode"] == "production_evidence"
        and config["hvp_mode_executed"] == "autograd"
        and int(config["state_position"]) == 2
        and int(config["proposals"]) == 100
        and int(config["pairs"]) == PAIRS_PER_DRAW
        and int(config["pool_draws"]) == POOL_DRAWS
        and int(config["effective_pairs_per_proposal"]) == 32
        and math.isclose(float(config["beta"]), EXPECTED_BETA, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(config["lr"]), 3e-5, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(config["gradient_clip"]), 1.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(config["burg_epsilon"]), 1e-4, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(config["armijo_c1"]), 1e-4, rel_tol=0.0, abs_tol=0.0)
        and summary["z_sha256"] == EXPECTED_Z
    )

    expected_state_keys = list(range(101))
    expected_proposal_keys = list(range(1, 101))
    expected_draw_keys = {
        (proposal, draw)
        for proposal in expected_proposal_keys
        for draw in range(POOL_DRAWS)
    }
    expected_pair_keys = {
        (proposal, draw, pair)
        for proposal in expected_proposal_keys
        for draw in range(POOL_DRAWS)
        for pair in range(PAIRS_PER_DRAW)
    }
    observed_draw_keys = set(
        zip(draws["proposal"].astype(int), draws["draw"].astype(int), strict=True)
    )
    observed_pair_keys = set(
        zip(
            pairs["proposal"].astype(int),
            pairs["draw"].astype(int),
            pairs["pair"].astype(int),
            strict=True,
        )
    )
    expected_line_keys = {
        (proposal, "line", round(alpha, 12))
        for proposal in expected_proposal_keys
        for alpha in LINE_ALPHAS
    }
    expected_line_keys |= {
        (proposal, kind, round(sign * step, 12))
        for proposal in expected_proposal_keys
        for kind, sign in (("fd_positive", 1.0), ("fd_negative", -1.0))
        for step in SLOPE_STEPS
    }
    observed_line_keys = set(
        zip(
            lines["proposal"].astype(int),
            lines["kind"].astype(str),
            lines["alpha"].round(12),
            strict=True,
        )
    )
    checks["exact_cartesian_row_keys"] = bool(
        states["proposal"].astype(int).tolist() == expected_state_keys
        and proposals["proposal"].astype(int).tolist() == expected_proposal_keys
        and observed_draw_keys == expected_draw_keys
        and observed_pair_keys == expected_pair_keys
        and observed_line_keys == expected_line_keys
        and len(draws) == 800
        and len(pairs) == 3200
        and len(lines) == 1100
    )

    seed_values = pd.concat([pairs["seed_1"], pairs["seed_2"]], ignore_index=True)
    draw0 = pairs.loc[pairs["draw"].eq(0)]
    draw0_match = all(
        int(getattr(row, f"seed_{branch + 1}"))
        == stable_uint63(
            A_ONLY_PROTOCOL_ID,
            "train",
            int(row.proposal),
            int(row.pair),
            branch,
        )
        for row in draw0.itertuples(index=False)
        for branch in (0, 1)
    )
    expected_schedule: list[dict[str, int]] = []
    for proposal in expected_proposal_keys:
        for draw in range(POOL_DRAWS):
            for pair in range(PAIRS_PER_DRAW):
                protocol_parts = (
                    (A_ONLY_PROTOCOL_ID, "train", proposal, pair)
                    if draw == 0
                    else (EXPECTED_PROTOCOL, "p32_extra", proposal, draw, pair)
                )
                expected_schedule.append(
                    {
                        "proposal": proposal,
                        "draw": draw,
                        "pair": pair,
                        "seed_1": stable_uint63(*protocol_parts, 0),
                        "seed_2": stable_uint63(*protocol_parts, 1),
                    }
                )
    observed_schedule = [
        {
            "proposal": int(row.proposal),
            "draw": int(row.draw),
            "pair": int(row.pair),
            "seed_1": int(row.seed_1),
            "seed_2": int(row.seed_2),
        }
        for row in pairs.sort_values(["proposal", "draw", "pair"]).itertuples(index=False)
    ]
    expected_schedule_sha = seed_schedule_hash(expected_schedule)
    checks["seed_topology_matches"] = bool(
        len(seed_values) == 6400
        and seed_values.nunique() == 6400
        and draw0_match
        and observed_schedule == expected_schedule
        and seed_review["valid"] is True
        and seed_review["draw0_matches_canonical_generators"] is True
        and seed_review["schedule_sha256"] == expected_schedule_sha
        and seed_review["observed_schedule_sha256"] == expected_schedule_sha
        and seed_review["observed_matches_preflight"] is True
        and config["seed_schedule_sha256"] == expected_schedule_sha
    )
    pooling_errors = [
        float(value) for key, value in pooling.items() if key.endswith("relative_error")
    ]
    checks["pooling_preflight_matches"] = bool(
        pooling.get("proposal") == 1
        and pooling.get("valid") is True
        and len(pooling_errors) == 3
        and all(math.isfinite(error) and error <= 1e-6 for error in pooling_errors)
    )
    checks["all_numeric_tables_finite"] = all(
        finite(frame)
        for frame in (states, proposals, lines, pairs, draws, forward, forward_scalars, stopped, control)
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
    recomputed_slope_h128: list[float] = []
    recomputed_slope_h256: list[float] = []
    accepted_clock = 0
    clock_valid = True
    for proposal_id in expected_proposal_keys:
        diagnostic = proposals.loc[proposals["proposal"].eq(proposal_id)].iloc[0]
        before_f = float(states.loc[states["proposal"].eq(proposal_id - 1), "true_objective"].iloc[0])
        after_f = float(states.loc[states["proposal"].eq(proposal_id), "true_objective"].iloc[0])
        proposal_lines = lines.loc[lines["proposal"].eq(proposal_id)]
        slopes: dict[float, float] = {}
        for step, column in ((SLOPE_STEPS[0], "slope_h128"), (SLOPE_STEPS[1], "slope_h256")):
            positive = float(
                proposal_lines.loc[
                    proposal_lines["kind"].eq("fd_positive")
                    & np.isclose(proposal_lines["alpha"], step),
                    "true_objective",
                ].iloc[0]
            )
            negative = float(
                proposal_lines.loc[
                    proposal_lines["kind"].eq("fd_negative")
                    & np.isclose(proposal_lines["alpha"], -step),
                    "true_objective",
                ].iloc[0]
            )
            slopes[step] = (positive - negative) / (2.0 * step)
            slope_errors.append(abs(slopes[step] - float(diagnostic[column])))
        recomputed_slope_h128.append(slopes[SLOPE_STEPS[0]])
        recomputed_slope_h256.append(slopes[SLOPE_STEPS[1]])

        selected = 0.0
        if all(slopes[step] < 0.0 for step in SLOPE_STEPS):
            for alpha in LINE_ALPHAS:
                candidate_f = float(
                    proposal_lines.loc[
                        proposal_lines["kind"].eq("line")
                        & np.isclose(proposal_lines["alpha"], alpha),
                        "true_objective",
                    ].iloc[0]
                )
                if (
                    candidate_f
                    <= before_f + float(config["armijo_c1"]) * alpha * slopes[SLOPE_STEPS[1]]
                    and candidate_f < before_f - STRICT_TOLERANCE
                ):
                    selected = alpha
                    break
        recomputed_alphas.append(selected)
        alpha_errors.append(abs(selected - float(diagnostic["accepted_alpha"])))
        accepted = int(selected > 0.0)
        transition_errors.append(abs(float(diagnostic["accepted"]) - accepted))
        clock_valid = clock_valid and int(diagnostic["accepted_adam_step_before"]) == accepted_clock
        accepted_clock += accepted
        clock_valid = clock_valid and int(diagnostic["accepted_adam_step_after"]) == accepted_clock
        transition_errors.extend(
            [
                abs(float(diagnostic["current_f"]) - before_f),
                abs(float(diagnostic["accepted_f"]) - after_f),
                abs(float(diagnostic["accepted_delta_f"]) - (after_f - before_f)),
                abs(float(states.loc[states["proposal"].eq(proposal_id), "accepted_alpha"].iloc[0]) - selected),
                abs(float(states.loc[states["proposal"].eq(proposal_id), "accepted"].iloc[0]) - accepted),
            ]
        )
        if selected > 0.0:
            selected_f = float(
                proposal_lines.loc[
                    proposal_lines["kind"].eq("line")
                    & np.isclose(proposal_lines["alpha"], selected),
                    "true_objective",
                ].iloc[0]
            )
            transition_errors.append(abs(selected_f - after_f))
        else:
            transition_errors.append(abs(before_f - after_f))

    checks["fd_slopes_replay"] = max(slope_errors) <= 5e-3
    checks["armijo_and_state_transitions_replay"] = bool(
        max(alpha_errors) <= 1e-12 and max(transition_errors) <= 1e-10
    )
    checks["accepted_only_adam_clock_replays"] = clock_valid
    accepted_displacement_errors = (
        proposals["accepted_displacement_norm"]
        - proposals["accepted_alpha"] * proposals["proposal_norm"]
    ).abs()
    checks["accepted_displacement_formula_matches"] = bool(
        float(accepted_displacement_errors.max()) <= 1e-12
    )

    checkpoint = torch.load(run / "final_checkpoint.pt", map_location="cpu", weights_only=False)
    names = sorted(checkpoint["active_model_state"])
    parameter_hash = named_tensor_hash(names, [checkpoint["active_model_state"][name] for name in names])
    moment_hash = hashlib.sha256(
        (
            named_tensor_hash(names, [checkpoint["exp_avg"][name] for name in names])
            + named_tensor_hash(names, [checkpoint["exp_avg_sq"][name] for name in names])
            + str(int(checkpoint["accepted_adam_step"]))
        ).encode("utf-8")
    ).hexdigest()
    checks["final_parameter_moment_clock_hashes_match"] = bool(
        parameter_hash
        == summary["final_parameter_hash"]
        == summary["final_checkpoint_parameter_hash"]
        and moment_hash == summary["final_checkpoint_moment_hash"]
        and int(checkpoint["accepted_adam_step"])
        == int(summary["final_checkpoint_adam_step"])
        == accepted_clock
    )
    checks["runner_validity_is_provisional_only"] = bool(
        summary["valid"]
        and all(summary["validity_gates"].values())
        and summary["all_success_gates_pass"] is None
        and summary["mechanism_decision"] == "awaiting_independent_replay"
    )

    accepted = pd.Series(
        [int(alpha > 0.0) for alpha in recomputed_alphas], index=proposals.index, dtype=int
    )
    final_f = float(states.iloc[-1]["true_objective"])
    strict_decreases = int((np.diff(states["true_objective"].to_numpy()) < -STRICT_TOLERANCE).sum())
    rejected = int(accepted.eq(0).sum())
    slope_h128 = np.asarray(recomputed_slope_h128, dtype=np.float64)
    slope_h256 = np.asarray(recomputed_slope_h256, dtype=np.float64)
    negative_both = int(
        ((slope_h128 < 0.0) & (slope_h256 < 0.0)).sum()
    )
    sign_agreements = int((np.sign(slope_h128) == np.sign(slope_h256)).sum())
    relative_slope_disagreement = np.abs(slope_h128 - slope_h256) / np.maximum(
        np.maximum(np.abs(slope_h128), np.abs(slope_h256)), 1e-30
    )
    accepted_alphas = np.asarray([alpha for alpha in recomputed_alphas if alpha > 0.0])
    median_alpha = float(np.median(accepted_alphas)) if len(accepted_alphas) else 0.0
    recomputed_accepted_norm = proposals["proposal_norm"] * np.asarray(recomputed_alphas)
    path_fraction = float(
        recomputed_accepted_norm.sum() / max(float(proposals["proposal_norm"].sum()), 1e-30)
    )
    state_deltas = np.diff(states["true_objective"].to_numpy())
    success_gates = {
        "final_f_at_most_frozen_20pct_target": final_f <= EXPECTED_TARGET,
        "beats_frozen_old_best": final_f < EXPECTED_OLD_BEST,
        "at_least_90_strict_decreases": strict_decreases >= 90,
        "at_most_10_rejections": rejected <= 10,
        "no_accepted_increases": bool(
            np.all(state_deltas[accepted.to_numpy(dtype=bool)] < -STRICT_TOLERANCE)
        ),
        "maximum_rejection_run_at_most_3": max_zero_run(accepted.tolist()) <= 3,
        "at_least_90_negative_slope_proposals": negative_both >= 90,
        "at_least_95_slope_sign_agreements": sign_agreements >= 95,
        "median_slope_disagreement_at_most_0p10": float(np.median(relative_slope_disagreement)) <= 0.10,
        "p95_slope_disagreement_at_most_0p25": float(np.quantile(relative_slope_disagreement, 0.95)) <= 0.25,
        "median_alpha_at_least_one_eighth": median_alpha >= 1.0 / 8.0,
        "path_fraction_at_least_0p25": path_fraction >= 0.25,
    }
    causal_repair_gates = {
        "fewer_rejections_than_p4": rejected < 19,
        "more_strict_decreases_than_p4": strict_decreases > 81,
        "shorter_maximum_plateau_than_p4": max_zero_run(accepted.tolist()) < 8,
    }
    checks["success_and_causal_gates_match_runner"] = bool(
        success_gates == summary["success_gates"]
        and causal_repair_gates == summary["causal_repair_gates"]
        and bool(all(success_gates.values()) and all(causal_repair_gates.values()))
        == bool(summary["provisional_success_gates_pass"])
    )
    plot = mpimg.imread(run / "true_objective_curve.png")
    checks["plot_is_nonblank_and_readable_shape"] = bool(
        plot.ndim in (2, 3)
        and plot.shape[0] >= 900
        and plot.shape[1] >= 1200
        and float(np.nanstd(plot)) > 0.01
    )

    all_valid = bool(all(checks.values()))
    final_success = bool(all(success_gates.values()) and all(causal_repair_gates.values())) if all_valid else None
    review: dict[str, Any] = {
        "protocol_id": EXPECTED_PROTOCOL,
        "all_posthoc_validity_checks_pass": all_valid,
        "all_recomputed_success_gates_pass": final_success,
        "final_mechanism_decision": (
            "p32_trajectory_repair_passed" if final_success else "p32_trajectory_repair_failed"
        ) if all_valid else None,
        "checks": checks,
        "success_gates": success_gates if all_valid else None,
        "causal_repair_gates": causal_repair_gates if all_valid else None,
        "metrics": {
            "initial_f": float(states.iloc[0]["true_objective"]),
            "final_f": final_f,
            "strict_decreases": strict_decreases,
            "rejected": rejected,
            "negative_both": negative_both,
            "slope_sign_agreements": sign_agreements,
            "maximum_rejection_run": max_zero_run(accepted.tolist()),
            "median_slope_disagreement": float(np.median(relative_slope_disagreement)),
            "p95_slope_disagreement": float(np.quantile(relative_slope_disagreement, 0.95)),
            "median_accepted_alpha": median_alpha,
            "path_fraction": path_fraction,
            "max_state_formula_error": state_formula_error,
            "max_slope_replay_error": max(slope_errors),
            "max_alpha_replay_error": max(alpha_errors),
            "max_transition_replay_error": max(transition_errors),
            "final_adam_step": accepted_clock,
        },
    }
    output = run / "posthoc_validation.json"
    output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(review, indent=2, sort_keys=True))
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
