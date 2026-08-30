from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_a_proposal3_cross_i1_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048/iteration1_proposal3_cross_production"
)
PROTOCOL_PATH = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048/iteration_1_proposal3_cross/protocol.md"
)
FROZEN_STATES = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration5_p32_training_production/state_objective_curve.csv"
)
DIRECTIONS = (
    "exact_a_raw",
    "p32_a_raw",
    "exact_oldbeta_raw",
    "p32_oldbeta_raw",
    "p32_oldbeta_carried_adam",
    "p32_unit_common_carried_adam",
)
POSITIVE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
FD_STEPS = (1.0 / 128.0, 1.0 / 256.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))


def _stable_sign(slopes: Mapping[str, float], component: str, sign: str) -> bool:
    values = [float(slopes[f"slope_{component}_h128"]), float(slopes[f"slope_{component}_h256"])]
    if sign == "down":
        return all(value < -1e-6 for value in values)
    return all(value > 1e-6 for value in values)


def _directional_reliability(analytic: float, slopes: Mapping[str, float]) -> dict[str, float | bool]:
    coarse = float(slopes["slope_a_h128"])
    fine = float(slopes["slope_a_h256"])
    richardson = (4.0 * fine - coarse) / 3.0
    repeat = float(slopes["slope_a_repeat_h256"])
    noise = float(slopes["slope_a_endpoint_noise_bound_h256"])
    values = (float(analytic), coarse, fine, richardson, repeat)
    same_sign = bool(all(value > 0.0 for value in values) or all(value < 0.0 for value in values))
    relative_error = abs(float(analytic) - richardson) / max(
        abs(float(analytic)), abs(richardson), 1e-12
    )
    truncation = abs(fine - coarse) / 3.0
    uncertainty = truncation + noise
    margin = min(abs(value) for value in values)
    return {
        "slope_a_richardson": richardson,
        "slope_a_richardson_relative_error": relative_error,
        "slope_a_truncation_bound": truncation,
        "slope_a_noise_bound": noise,
        "slope_a_uncertainty_bound": uncertainty,
        "slope_a_sign_margin": margin,
        "slope_a_all_signs_agree": same_sign,
        "slope_a_reliable": bool(
            same_sign and relative_error <= 0.05 and margin > 5.0 * uncertainty
        ),
    }


def _mechanisms(
    slopes: Mapping[str, Mapping[str, float]],
    line: pd.DataFrame,
    base: Mapping[str, float],
    reliability: Mapping[str, bool],
) -> dict[str, bool | None]:
    def resolved(required: tuple[str, ...], supported: bool) -> bool | None:
        return bool(supported) if all(reliability[name] for name in required) else None

    base_a = float(base["exact_a_per_dim"])
    base_m_max = float(base["m_max"])
    base_high_a = float(base["a_high_gt_1_abs_per_dim"])
    exact_down = _stable_sign(slopes["exact_a_raw"], "a", "down")
    p32_down = _stable_sign(slopes["p32_a_raw"], "a", "down")
    exact_oldbeta_down = _stable_sign(slopes["exact_oldbeta_raw"], "a", "down")
    old_raw_down = _stable_sign(slopes["p32_oldbeta_raw"], "a", "down")
    old_adam_down = _stable_sign(slopes["p32_oldbeta_carried_adam"], "a", "down")
    canonical = line.loc[
        line["direction"].eq("p32_oldbeta_carried_adam") & np.isclose(line["alpha"], 1.0)
    ]
    smaller = line.loc[
        line["direction"].eq("p32_oldbeta_carried_adam")
        & line["alpha"].isin(POSITIVE_ALPHAS[1:])
    ]
    return {
        "finite_p32_estimator_supported": resolved(
            ("exact_a_raw", "p32_a_raw"),
            exact_down and _stable_sign(slopes["p32_a_raw"], "a", "up"),
        ),
        "exact_oldbeta_scalarization_conflict_supported": resolved(
            ("exact_a_raw", "exact_oldbeta_raw"),
            exact_down and _stable_sign(slopes["exact_oldbeta_raw"], "a", "up"),
        ),
        "p32_composite_estimator_supported": resolved(
            ("exact_oldbeta_raw", "p32_oldbeta_raw"),
            exact_oldbeta_down and _stable_sign(slopes["p32_oldbeta_raw"], "a", "up"),
        ),
        "carried_adam_transform_supported": resolved(
            ("p32_oldbeta_raw", "p32_oldbeta_carried_adam"),
            old_raw_down and _stable_sign(slopes["p32_oldbeta_carried_adam"], "a", "up"),
        ),
        "finite_radius_spike_overshoot_supported": resolved(
            ("p32_oldbeta_carried_adam",),
            bool(
                old_adam_down
                and len(canonical) == 1
                and float(canonical.iloc[0]["exact_a_per_dim"]) > base_a + 1e-8
                and float(canonical.iloc[0]["m_max"]) > base_m_max + 1e-8
                and float(canonical.iloc[0]["a_high_gt_1_abs_per_dim"])
                > base_high_a + 1e-8
                and bool(
                    (
                        (smaller["exact_a_per_dim"] < base_a - 1e-8)
                        & (smaller["delta_m_max"] <= 1e-8)
                        & (smaller["a_high_gt_1_abs_per_dim"] <= base_high_a + 1e-8)
                    ).any()
                )
            ),
        ),
    }


def _candidate(
    line: pd.DataFrame,
    direction: str,
    base: Mapping[str, float],
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    rows = line.loc[
        line["direction"].eq(direction)
        & line["kind"].eq("line")
        & line["alpha"].ge(0.125)
    ].copy()
    passes = (
        rows["exact_a_per_dim"].lt(float(base["exact_a_per_dim"]) - tolerances["a"])
        & rows["damped_full_burg_per_dim"].lt(
            float(base["damped_full_burg_per_dim"]) - tolerances["b"]
        )
        & rows["m_max"].le(float(base["m_max"]) + tolerances["m_max"])
        & rows["m_p50"].ge(float(base["m_p50"]) - tolerances["m_p50"])
        & rows["m_lt_0p1_fraction"].le(
            float(base["m_lt_0p1_fraction"]) + tolerances["m_lt_0p1_fraction"]
        )
    )
    passing = rows.loc[passes].sort_values("alpha", ascending=False)
    return {
        "direction": direction,
        "selected": bool(len(passing)),
        "largest_passing_alpha": None if passing.empty else float(passing.iloc[0]["alpha"]),
        "passing_alpha_count": int(len(passing)),
        "tolerances": {key: float(value) for key, value in tolerances.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    gates: dict[str, bool] = {}

    completed = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
    resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    gates["completion_marker_and_no_incomplete"] = not (output / "INCOMPLETE").exists()
    gates["protocol_ids_match"] = {
        completed["protocol_id"], resolved["protocol_id"], decision["protocol_id"]
    } == {PROTOCOL_ID}
    gates["producer_declared_valid"] = bool(completed["valid"] and decision["valid"])
    gates["decision_hash_matches"] = completed["decision_sha256"] == _sha256(output / "decision.json")
    gates["manifest_hash_matches"] = completed["artifact_manifest_sha256"] == _sha256(
        output / "artifact_manifest.json"
    )
    gates["all_manifest_artifacts_match"] = all(
        (output / name).is_file() and _sha256(output / name) == digest
        for name, digest in manifest.items()
    )
    gates["executed_source_matches_config"] = (
        _sha256(output / "executed_source_snapshot.py") == resolved["source_sha256"]
    )
    dependency_snapshot = output / "frozen_dependency_manifest_snapshot.json"
    gates["frozen_dependency_manifest_matches_config"] = bool(
        _sha256(dependency_snapshot) == resolved["frozen_dependency_manifest_sha256"]
        and all(resolved["frozen_dependency_matches"].values())
    )
    frozen_dependencies = json.loads(dependency_snapshot.read_text(encoding="utf-8"))
    gates["all_frozen_dependencies_still_match"] = all(
        (ROOT / relative).is_file() and _sha256(ROOT / relative) == digest
        for relative, digest in frozen_dependencies.items()
    )
    gates["protocol_matches_config"] = _sha256(PROTOCOL_PATH) == resolved["protocol_sha256"]

    line = pd.read_csv(output / "line_profiles.csv")
    directions = pd.read_csv(output / "direction_diagnostics.csv")
    spectra = pd.read_csv(output / "line_spectra.csv")
    reconstruction = pd.read_csv(output / "reconstruction.csv")
    base_repeatability = pd.read_csv(output / "base_repeatability.csv")
    h_space = pd.read_csv(output / "exact_objective_h_space_fd.csv")
    repeats = pd.read_csv(output / "fd_repeatability.csv")
    pairs = pd.read_csv(output / "p32_pair_scalars.csv")
    draws = pd.read_csv(output / "p32_draw_diagnostics.csv")
    proposal3_components = pd.read_csv(output / "proposal3_component_reconstruction.csv")
    numeric_frames = (
        line,
        directions,
        spectra,
        reconstruction,
        base_repeatability,
        h_space,
        repeats,
        pairs,
        draws,
        proposal3_components,
    )
    gates["all_numeric_values_finite"] = all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        for frame in numeric_frames
    )
    expected_alphas = set(POSITIVE_ALPHAS + FD_STEPS + tuple(-step for step in FD_STEPS))
    gates["line_keys_complete_unique"] = bool(
        len(line) == 66
        and not line.duplicated(["direction", "alpha"]).any()
        and set(line["direction"]) == set(DIRECTIONS)
        and all(set(group["alpha"]) == expected_alphas for _, group in line.groupby("direction"))
    )
    gates["direction_rows_complete_unique"] = bool(
        len(directions) == 6
        and directions["direction"].is_unique
        and set(directions["direction"]) == set(DIRECTIONS)
    )
    gates["fd_repeat_keys_complete_unique"] = bool(
        len(repeats) == 12
        and not repeats.duplicated(["direction", "alpha"]).any()
        and set(repeats["direction"]) == set(DIRECTIONS)
        and all(
            set(group["alpha"]) == {1.0 / 256.0, -1.0 / 256.0}
            for _, group in repeats.groupby("direction")
        )
    )
    gates["p32_sampling_shape"] = bool(
        len(pairs) == 32
        and len(draws) == 8
        and not pairs.duplicated(["proposal", "draw", "pair"]).any()
        and pairs.groupby("draw").size().eq(4).all()
    )
    base = decision["base_metrics"]
    repeatability_columns = {
        "a": "exact_a_per_dim",
        "b": "damped_full_burg_per_dim",
        "m_max": "m_max",
        "m_p50": "m_p50",
        "m_lt_0p1_fraction": "m_lt_0p1_fraction",
    }
    primary_repeat = base_repeatability.loc[base_repeatability["evaluation"].eq("primary")]
    second_repeat = base_repeatability.loc[base_repeatability["evaluation"].eq("repeat")]
    recomputed_noise = {
        key: abs(float(primary_repeat.iloc[0][column]) - float(second_repeat.iloc[0][column]))
        for key, column in repeatability_columns.items()
    } if len(primary_repeat) == 1 and len(second_repeat) == 1 else {}
    recomputed_tolerances = {
        key: max(5.0 * recomputed_noise[key], 1e-8 if key in {"a", "b", "m_max"} else 1e-12)
        for key in recomputed_noise
    }
    gates["base_repeatability_recomputes"] = bool(
        len(recomputed_noise) == len(repeatability_columns)
        and max(recomputed_noise.values()) <= 1e-6
        and all(_close(recomputed_noise[key], decision["base_repeatability_noise"][key]) for key in recomputed_noise)
        and all(_close(recomputed_tolerances[key], decision["candidate_tolerances"][key]) for key in recomputed_tolerances)
    )
    delta_checks = []
    for row in line.itertuples(index=False):
        delta_checks.extend(
            [
                _close(row.delta_a, row.exact_a_per_dim - base["exact_a_per_dim"]),
                _close(row.delta_b, row.damped_full_burg_per_dim - base["damped_full_burg_per_dim"]),
                _close(row.delta_m_max, row.m_max - base["m_max"]),
                _close(row.delta_m_p50, row.m_p50 - base["m_p50"]),
            ]
        )
    gates["line_deltas_recompute"] = all(delta_checks)

    slopes: dict[str, dict[str, float]] = {}
    slope_checks = []
    for direction in DIRECTIONS:
        rows = line.loc[line["direction"].eq(direction)]
        stored = directions.loc[directions["direction"].eq(direction)].iloc[0]
        slopes[direction] = {}
        for step in FD_STEPS:
            denominator = int(round(1.0 / step))
            positive = rows.loc[np.isclose(rows["alpha"], step)].iloc[0]
            negative = rows.loc[np.isclose(rows["alpha"], -step)].iloc[0]
            for component, column in (
                ("a", "a_direct_matrix"),
                ("b", "damped_full_burg_per_dim"),
                ("f", "true_objective_direct"),
            ):
                value = float((positive[column] - negative[column]) / (2.0 * step))
                name = f"slope_{component}_h{denominator}"
                slopes[direction][name] = value
                slope_checks.append(_close(value, stored[name], tolerance=1e-9))
        repeat_rows = repeats.loc[repeats["direction"].eq(direction)]
        repeat_positive = repeat_rows.loc[repeat_rows["alpha"].gt(0.0)].iloc[0]
        repeat_negative = repeat_rows.loc[repeat_rows["alpha"].lt(0.0)].iloc[0]
        original_positive = rows.loc[np.isclose(rows["alpha"], 1.0 / 256.0)].iloc[0]
        original_negative = rows.loc[np.isclose(rows["alpha"], -1.0 / 256.0)].iloc[0]
        repeat_slope = float(
            (repeat_positive["a_direct_matrix"] - repeat_negative["a_direct_matrix"])
            / (2.0 / 256.0)
        )
        noise_bound = float(
            (
                abs(repeat_positive["a_direct_matrix"] - original_positive["a_direct_matrix"])
                + abs(repeat_negative["a_direct_matrix"] - original_negative["a_direct_matrix"])
            )
            / (2.0 / 256.0)
        )
        slopes[direction]["slope_a_repeat_h256"] = repeat_slope
        slopes[direction]["slope_a_endpoint_noise_bound_h256"] = noise_bound
        slope_checks.extend(
            [
                _close(repeat_slope, stored["slope_a_repeat_h256"], tolerance=1e-9),
                _close(
                    noise_bound,
                    stored["slope_a_endpoint_noise_bound_h256"],
                    tolerance=1e-9,
                ),
            ]
        )
        reliability = _directional_reliability(
            float(stored["analytic_exact_a_slope"]), slopes[direction]
        )
        for key, value in reliability.items():
            if isinstance(value, bool):
                slope_checks.append(bool(stored[key]) == value)
            else:
                slope_checks.append(_close(float(stored[key]), value, tolerance=1e-9))
    gates["central_fd_slopes_recompute"] = all(slope_checks)
    direction_reliability = {
        str(row.direction): bool(row.slope_a_reliable)
        for row in directions.itertuples(index=False)
    }
    gates["mechanism_decision_recomputes"] = _mechanisms(
        slopes, line, base, direction_reliability
    ) == decision["mechanisms"]
    gates["reported_direction_reliability_recomputes"] = (
        direction_reliability == decision["directional_derivative_reliability"]
    )
    exact_oldbeta = directions.loc[directions["direction"].eq("exact_oldbeta_raw")].iloc[0]
    exact_oldbeta_errors = [
        abs(
            float(exact_oldbeta[f"slope_f_h{denominator}"])
            - float(exact_oldbeta["analytic_exact_oldbeta_slope"])
        )
        / max(
            abs(float(exact_oldbeta[f"slope_f_h{denominator}"])),
            abs(float(exact_oldbeta["analytic_exact_oldbeta_slope"])),
            1e-30,
        )
        for denominator in (128, 256)
    ]
    gates["exact_oldbeta_pullback_matches_fd"] = max(exact_oldbeta_errors) <= 0.05
    gates["candidate_decisions_recompute"] = bool(
        _candidate(
            line,
            "p32_unit_common_carried_adam",
            base,
            decision["candidate_tolerances"],
        )
        == decision["candidate_carried"]
    )

    spectrum_checks = []
    for (direction, alpha), group in spectra.groupby(["direction", "alpha"], sort=False):
        endpoint = line.loc[
            line["direction"].eq(direction) & np.isclose(line["alpha"], alpha)
        ].iloc[0]
        ordered = group.sort_values("rank")
        spectrum_checks.extend(
            [
                len(ordered) == 512,
                np.array_equal(ordered["rank"].to_numpy(), np.arange(512)),
                bool(np.diff(ordered["m_eigenvalue"].to_numpy()).min(initial=0.0) >= -1e-12),
                _close(ordered["a_contribution"].mean(), endpoint["exact_a_per_dim"], 1e-9),
                _close(ordered["m_eigenvalue"].max(), endpoint["m_max"], 1e-9),
            ]
        )
    gates["all_dense_spectra_recompute"] = bool(len(spectra) == 33792 and all(spectrum_checks))
    gates["a_matrix_and_trace_closures"] = bool(
        line[["a_direct_abs_error", "a_trace_abs_error"]].to_numpy().max() <= 1e-10
    )
    error_columns = [column for column in reconstruction if column.endswith("_abs_error")]
    relative_columns = [column for column in reconstruction if column.endswith("_relative_error")]
    gates["replay_errors_within_tolerance"] = bool(
        error_columns
        and relative_columns
        and reconstruction[error_columns].to_numpy().max() <= 1e-5
        and reconstruction[relative_columns].to_numpy().max() <= 1e-6
    )
    gates["exact_objective_h_space_fd_within_tolerance"] = bool(
        len(h_space) == 4
        and set(h_space["objective"]) == {"a", "oldbeta"}
        and h_space.groupby("objective").size().eq(2).all()
        and set(h_space["delta"]) == {1e-5, 5e-6}
        and h_space["relative_error"].max() <= 1e-5
    )
    component_abs = [column for column in proposal3_components if column.endswith("_abs_error")]
    component_relative = [
        column for column in proposal3_components if column.endswith("_relative_error")
    ]
    gates["proposal3_components_match_frozen"] = bool(
        len(proposal3_components) == 1
        and component_abs
        and component_relative
        and proposal3_components[component_abs].to_numpy().max() <= 1e-5
        and proposal3_components[component_relative].to_numpy().max() <= 1e-6
    )
    frozen = pd.read_csv(FROZEN_STATES)
    frozen_p3 = frozen.loc[frozen["proposal"].eq(3)].iloc[0]
    canonical = line.loc[
        line["direction"].eq("p32_oldbeta_carried_adam") & np.isclose(line["alpha"], 1.0)
    ].iloc[0]
    gates["canonical_endpoint_matches_frozen_proposal3"] = all(
        _close(canonical[column], frozen_p3[column], 1e-5)
        for column in ("exact_a_per_dim", "damped_full_burg_per_dim", "true_objective", "m_max")
    )
    gates["plot_is_nonempty"] = (output / "proposal3_exact_a_cross.png").stat().st_size > 10_000
    gates["producer_gates_all_true"] = all(bool(value) for value in decision["validity_gates"].values())

    valid = all(gates.values())
    report = {
        "protocol_id": PROTOCOL_ID,
        "valid": valid,
        "gates": gates,
        "mechanisms": decision["mechanisms"] if valid else None,
        "candidate_carried": decision["candidate_carried"] if valid else None,
    }
    (output / "independent_review.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not valid:
        raise RuntimeError("independent proposal-3 review failed")


if __name__ == "__main__":
    main()
