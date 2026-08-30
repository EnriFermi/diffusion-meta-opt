from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.aggregate_common_state_rate_seeds import (
    MERGED_OUTPUT_FILES,
    METRIC,
    RETRACTION_METRIC,
    ContrastSpec,
    _sha256_file,
    _stable_json_hash,
    _verify_merged_hashes,
    aggregate_hierarchy,
    build_paired_cells,
    build_trust_diagnostics,
    contrast_specs,
    cutoff_sign_consistency,
    build_coverage,
)


def _rows(candidates: set[str] | None = None, *, singular: bool = False, threshold: bool = True, technical: bool = True) -> pd.DataFrame:
    specs = contrast_specs()
    candidates = candidates or {s.left for s in specs} | {s.right for s in specs if s.right not in (None, "__zero__")}
    records = []
    for index, candidate in enumerate(sorted(candidates)):
        for trust_mode in ("weight", "function"):
            records.append(
                {
                    "vae_seed_label": "control_seed0",
                    "label": "control_seed0",
                    "source_weight_index": 11,
                    "path_step": 1,
                    "draw_id": 0,
                    "radius_fraction": 0.125,
                    "candidate": candidate,
                    "trust_mode": trust_mode,
                    METRIC: float(index),
                    RETRACTION_METRIC: float(index) / 10.0,
                    "delta_weight_norm": 2.0,
                    "delta_function_rms": 3.0,
                    "threshold_specific_eligible": threshold,
                    "technical_eligible": technical,
                    "singularity_start_any_state": singular,
                }
            )
    return pd.DataFrame(records)


def test_function_and_weight_trust_are_separate_and_native_zero_are_diagnostics() -> None:
    rows = _rows()
    cells = build_paired_cells(rows)
    assert set(cells["trust_mode"]) == {"weight", "function"}
    assert not cells["trust_mode"].isin({"native", "zero"}).any()

    diagnostic_rows = pd.concat(
        [
            rows,
            rows.assign(radius_fraction=1.0),
            rows.assign(trust_mode="native", radius_fraction=1.0),
            rows.assign(candidate="raw_adam_replay", trust_mode="zero", radius_fraction=0.0),
        ],
        ignore_index=True,
    ).drop_duplicates(["label", "source_weight_index", "path_step", "draw_id", "candidate", "trust_mode", "radius_fraction"])
    diagnostics = build_trust_diagnostics(diagnostic_rows)
    assert {"native_minus_matched", "matched_radius_curve", "zero_baseline"} <= set(diagnostics["diagnostic"])
    native = diagnostics[diagnostics["diagnostic"].eq("native_minus_matched")]
    assert native["native_to_matched_weight_radius_ratio"].eq(1.0).all()
    assert native["native_to_matched_function_radius_ratio"].eq(1.0).all()


def test_hierarchy_preserves_path_trust_radius_and_cutoff_and_averages_draws_first() -> None:
    records = []
    for path_step in (1, 5):
        for trust_mode in ("weight", "function"):
            for radius, rcond in ((0.125, 1.0e-6), (1.0, 1.0e-4)):
                for draw_id, value in enumerate((0.0, 2.0)):
                    records.append(
                        {
                            "vae_seed_label": "control_seed0",
                            "label": "control_seed0",
                            "source_weight_index": 11,
                            "path_step": path_step,
                            "draw_id": draw_id,
                            "trust_mode": trust_mode,
                            "radius_fraction": radius,
                            "projector_rcond": rcond,
                            "cutoff_applicability": "rcond_specific",
                            "contrast": "example",
                            "left": "a",
                            "right": "b",
                            "metric": METRIC,
                            "contrast_value": value,
                            "regular_eligible": True,
                            "singular_sensitivity_eligible": False,
                        }
                    )
    per_start, per_seed, cross_seed = aggregate_hierarchy(pd.DataFrame(records))
    assert len(per_start) == 8
    assert set(per_start["contrast_value"]) == {1.0}
    assert set(per_seed["path_step"]) == {1, 5}
    assert set(per_seed["trust_mode"]) == {"weight", "function"}
    assert set(per_seed["radius_fraction"]) == {0.125, 1.0}
    assert set(per_seed["projector_rcond"]) == {1.0e-6, 1.0e-4}
    assert len(cross_seed) == 8


def test_cutoff_independent_contrasts_are_not_triplicated() -> None:
    cells = pd.DataFrame(
        [
            {
                "vae_seed_label": "control_seed0",
                "label": "control_seed0",
                "source_weight_index": 1,
                "path_step": 1,
                "draw_id": 0,
                "trust_mode": "weight",
                "radius_fraction": 0.125,
                "projector_rcond": rcond,
                "cutoff_applicability": "rcond_specific",
                "contrast": "rcond",
                "left": "a",
                "right": "b",
                "metric": METRIC,
                "contrast_value": -rcond,
                "regular_eligible": True,
                "singular_sensitivity_eligible": False,
            }
            for rcond in (1.0e-6, 1.0e-5, 1.0e-4)
        ]
        + [
            {
                "vae_seed_label": "control_seed0",
                "label": "control_seed0",
                "source_weight_index": 1,
                "path_step": 1,
                "draw_id": 0,
                "trust_mode": "weight",
                "radius_fraction": 0.125,
                "projector_rcond": float("nan"),
                "cutoff_applicability": "cutoff_independent",
                "contrast": "independent",
                "left": "a",
                "right": "b",
                "metric": METRIC,
                "contrast_value": 3.0,
                "regular_eligible": True,
                "singular_sensitivity_eligible": False,
            }
        ]
    )
    _, _, cross_seed = aggregate_hierarchy(cells)
    assert len(cross_seed[cross_seed["contrast"] == "independent"]) == 1
    signs = cutoff_sign_consistency(cross_seed)
    assert len(signs) == 1
    assert signs.iloc[0]["cutoff_count"] == 3


def test_rcond_specs_share_one_name_and_coverage_requires_all_starts_and_draws() -> None:
    specs = [spec for spec in contrast_specs() if spec.name == "tangent_linear_minus_jjt_linear"]
    assert len(specs) == 3
    assert {spec.projector_rcond for spec in specs} == {1.0e-6, 1.0e-5, 1.0e-4}

    rows = []
    for source in range(16):
        for draw in (0, 1):
            rows.append(
                {
                    "vae_seed_label": "control_seed0",
                    "source_weight_index": source,
                    "path_step": 1,
                    "draw_id": draw,
                    "trust_mode": "weight",
                    "radius_fraction": 1.0,
                    "projector_rcond": 1.0e-5,
                    "cutoff_applicability": "rcond_specific",
                    "contrast": "example",
                    "left": "a",
                    "right": "b",
                    "metric": METRIC,
                    "contrast_value": 0.0,
                    "pair_eligible": True,
                    "regular_eligible": True,
                    "singularity_start_any_state": False,
                    "singular_sensitivity_eligible": False,
                }
            )
    coverage = build_coverage(pd.DataFrame(rows))
    assert bool(coverage.iloc[0]["structural_complete"])
    incomplete = build_coverage(pd.DataFrame(rows[:-1]))
    assert not bool(incomplete.iloc[0]["structural_complete"])


def test_diagnostic_singularity_mismatch_is_rejected() -> None:
    rows = _rows({"raw_adam_replay", "raw_adam_fresh"})
    rows.loc[rows["candidate"].eq("raw_adam_fresh"), "singularity_start_any_state"] = True
    spec = ContrastSpec("singular", "raw_adam_replay", "raw_adam_fresh", None, "cutoff_independent")
    with pytest.raises(ValueError, match="singularity"):
        build_paired_cells(rows, [spec])


def test_threshold_specific_eligibility_implies_technical_eligibility() -> None:
    rows = _rows(threshold=True, technical=False)
    with pytest.raises(ValueError, match="threshold_specific_eligible implies technical_eligible"):
        build_paired_cells(rows, [contrast_specs()[0]])


def test_mutated_hash_bound_packet_artifact_is_rejected(tmp_path) -> None:
    for name in MERGED_OUTPUT_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    validation = {"output_hashes": {name: _sha256_file(tmp_path / name) for name in MERGED_OUTPUT_FILES}}
    validation_path = tmp_path / "common_state_validation.json"
    validation_path.write_text(json.dumps(validation, sort_keys=True), encoding="utf-8")
    manifest = {"output_hashes": validation["output_hashes"], "validation_sha256": _sha256_file(validation_path)}
    assert _verify_merged_hashes(tmp_path, manifest, validation) == validation["output_hashes"]
    (tmp_path / MERGED_OUTPUT_FILES[0]).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        _verify_merged_hashes(tmp_path, manifest, validation)


def test_all_mechanism_specs_are_predeclared_with_explicit_cutoff_applicability() -> None:
    specs = contrast_specs()
    names = {spec.name for spec in specs}
    required_fragments = (
        "tangent_linear_minus_jjt_linear",
        "tangent_nonlinear_minus_latent_sgd_nonlinear",
        "latent_adam_replay_linear_minus_jjt_linear",
        "latent_adam_replay_nonlinear_minus_latent_sgd_nonlinear",
        "projected_raw_linear_minus_raw",
        "projected_raw_linear_minus_tangent",
        "task_normal_minus_placebo",
        "task_normal_minus_jjt",
        "placebo_minus_jjt",
        "raw_replay_minus_raw_fresh",
        "history_difference_in_differences",
        "rotation_dense0_minus_identity",
        "rotation_dense3_minus_identity",
        "nonlinear_retraction",
    )
    assert all(any(fragment in name for name in names) for fragment in required_fragments)
    assert {spec.projector_rcond for spec in specs if spec.cutoff_applicability == "rcond_specific"} == {1.0e-6, 1.0e-5, 1.0e-4}
    assert all(spec.projector_rcond is None for spec in specs if spec.cutoff_applicability == "cutoff_independent")


def test_history_difference_in_differences_direction() -> None:
    candidates = {"latent_adam_replay_linear", "latent_adam_fresh_linear", "raw_adam_replay", "raw_adam_fresh"}
    rows = _rows(candidates)
    values = {"latent_adam_replay_linear": 0.0, "latent_adam_fresh_linear": 3.0, "raw_adam_replay": 0.0, "raw_adam_fresh": 1.0}
    rows[METRIC] = rows["candidate"].map(values)
    spec = ContrastSpec("history_did", "latent_adam_replay_linear", "latent_adam_fresh_linear", None, "cutoff_independent", METRIC, "history_did", "raw_adam_replay", "raw_adam_fresh")
    cells = build_paired_cells(rows, [spec])
    assert set(cells["contrast_value"]) == {-2.0}


def test_equal_vae_seed_weighting_after_equal_start_means() -> None:
    records = []
    for seed, start, values in (("control_seed0", 1, (0.0, 2.0)), ("control_seed0", 2, (2.0, 4.0)), ("control_seed1", 3, (10.0, 14.0))):
        for draw_id, value in enumerate(values):
            records.append({"vae_seed_label": seed, "label": seed, "source_weight_index": start, "path_step": 1, "draw_id": draw_id, "trust_mode": "function", "radius_fraction": 0.125, "projector_rcond": float("nan"), "cutoff_applicability": "cutoff_independent", "contrast": "example", "left": "a", "right": "b", "metric": METRIC, "contrast_value": value, "regular_eligible": True, "singular_sensitivity_eligible": False})
    _, per_seed, cross_seed = aggregate_hierarchy(pd.DataFrame(records))
    assert sorted(per_seed["contrast_value"].tolist()) == [2.0, 12.0]
    assert cross_seed.iloc[0]["contrast_value"] == pytest.approx(7.0)
    assert cross_seed.iloc[0]["contrast_value"] != pytest.approx(6.0)
