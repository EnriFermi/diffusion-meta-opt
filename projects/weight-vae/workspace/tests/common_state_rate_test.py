from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
import torch

from scripts.merge_common_state_rate_chunks import (
    MERGED_OUTPUT_FILES,
    _validate_row_suffix_grid,
    _stable_json_hash,
    verify_merged_output_integrity,
)
from scripts.audit_latent_raw_common_state_rate import (
    Candidate,
    _expected_row_suffixes,
    _expected_candidates,
    _match_scale,
    _propagate_start_singularity,
    _row_state_integrity,
)


def _write_integrity_fixture(output_dir) -> None:
    for name in MERGED_OUTPUT_FILES:
        (output_dir / name).write_bytes(f"fixture:{name}".encode("utf-8"))
    output_hashes = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in MERGED_OUTPUT_FILES
    }
    validation = {"acceptance_pass": True, "output_hashes": output_hashes}
    validation_path = output_dir / "common_state_validation.json"
    validation_path.write_text(json.dumps(validation, sort_keys=True), encoding="utf-8")
    manifest = {
        "protocol_version": "common_state_rate_v3_merged",
        "output_hashes": output_hashes,
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    }
    manifest["request_hash"] = _stable_json_hash(manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_merged_output_integrity_accepts_unchanged_packet(tmp_path) -> None:
    _write_integrity_fixture(tmp_path)

    assert set(verify_merged_output_integrity(tmp_path)) == set(MERGED_OUTPUT_FILES)


@pytest.mark.parametrize("artifact", ["common_state_rate_rows.csv", "common_state_state_diagnostics.csv"])
def test_merged_output_integrity_detects_post_merge_mutation(tmp_path, artifact: str) -> None:
    _write_integrity_fixture(tmp_path)
    (tmp_path / artifact).open("ab").write(b"\npost-merge mutation")

    with pytest.raises(ValueError, match="merged output hash mismatch"):
        verify_merged_output_integrity(tmp_path)


def _match(scan: list[tuple[float, float]], target: float):
    theta = torch.zeros(2)
    candidate = Candidate("ambient", "ambient", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    return _match_scale(
        candidate,
        target=target,
        trust_mode="weight",
        scan=scan,
        theta=theta,
        z=torch.zeros(1),
        vae=None,
        normalizer=None,
        base_logits=None,
        calibration_images=None,
        spec=None,
        tau=1.0,
    )


def test_match_scale_uses_first_upward_crossing() -> None:
    result = _match([(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (1.0, 0.1), (2.0, 0.8)], 0.3)

    assert result.bracket_found
    assert result.local_positive_slope
    assert result.bracket_high_scale <= 0.5
    assert abs(result.metric - 0.3) / 0.3 <= 0.0025


def test_match_scale_records_missing_crossing() -> None:
    result = _match([(0.0, 0.0), (0.25, 0.05), (0.5, 0.1)], 0.3)

    assert not result.bracket_found
    assert result.termination_reason == "no_upward_crossing"


def test_match_scale_accepts_first_scan_interval() -> None:
    first = 2.0**-20
    result = _match([(0.0, 0.0), (first, first), (2.0 * first, 2.0 * first)], 0.5 * first)

    assert result.bracket_found
    assert not result.boundary_hit
    assert abs(result.metric - 0.5 * first) / (0.5 * first) <= 0.0025


def test_v3_candidate_cardinality() -> None:
    candidates, native = _expected_candidates(4)

    assert len(candidates) == 42
    assert len(native) == 16
    assert "tangent_oracle_rcond1e-5_nonlinear" in candidates


def test_start_singularity_propagates_without_censoring_non_svd_rows() -> None:
    states = pd.DataFrame(
        {
            "label": ["control", "control"],
            "source_weight_index": [7, 7],
            "path_step": [1, 5],
            "singularity_stratum": [True, False],
        }
    )
    rows = pd.DataFrame(
        {
            "label": ["control", "control"],
            "source_weight_index": [7, 7],
            "singularity_stratum": [True, False],
            "projector_rcond": [1.0e-5, float("nan")],
            "technical_eligible": [True, True],
            "operator_threshold_stable": [True, True],
        }
    )

    propagated_rows, propagated_states = _propagate_start_singularity(rows, states)

    assert propagated_states["singularity_stratum"].all()
    assert not bool(propagated_rows.iloc[0]["threshold_specific_eligible"])
    assert bool(propagated_rows.iloc[1]["threshold_specific_eligible"])
    assert bool(propagated_rows.iloc[1]["threshold_robust_eligible"])


def test_row_state_integrity_rejects_substituted_row_state_group() -> None:
    states = pd.DataFrame(
        {
            "label": ["control", "control"],
            "source_weight_index": [1, 2],
            "path_step": [1, 1],
            "draw_id": [0, 0],
            "start_bank_position": [0, 1],
            "stream_start_index": [0, 1],
            "task_name": ["task_a", "task_b"],
            "tau": [1.0, 2.0],
            "theta_sha256": ["theta1", "theta2"],
            "z_sha256": ["z1", "z2"],
            "update_batch_sha256": ["update1", "update2"],
            "eval_batch_sha256": ["eval1", "eval2"],
            "calibration_indices_sha256": ["cal1", "cal2"],
        }
    )
    rows = states.copy()
    start_bank = pd.DataFrame(
        {
            "source_weight_index": [1, 2],
            "start_bank_position": [0, 1],
            "task_name": ["task_a", "task_b"],
            "tau": [1.0, 2.0],
        }
    )
    expected = {("control", 1, 1, 0), ("control", 2, 1, 0)}

    valid = _row_state_integrity(
        rows,
        states,
        expected_state_key_set=expected,
        start_bank=start_bank,
        labels=["control"],
    )
    assert valid["row_state_group_count"] == 2
    assert valid["row_state_context_mismatch_row_count"] == 0
    assert valid["state_start_context_mismatch_row_count"] == 0

    substituted = rows.copy()
    substituted.loc[substituted["source_weight_index"] == 2, "source_weight_index"] = 3
    invalid = _row_state_integrity(
        substituted,
        states,
        expected_state_key_set=expected,
        start_bank=start_bank,
        labels=["control"],
    )
    assert invalid["row_state_group_count"] == 2
    assert invalid["row_missing_expected_state_key_count"] == 1
    assert invalid["row_unexpected_state_key_count"] == 1
    assert invalid["row_state_unmatched_row_count"] == 1


def test_row_suffix_grid_rejects_bogus_candidate_with_fixed_counts_and_states() -> None:
    expected_state = ("control", 1, 1, 0)
    expected_suffixes = _expected_row_suffixes(0, (0.5,))
    rows = pd.DataFrame(
        [
            {
                "label": expected_state[0],
                "source_weight_index": expected_state[1],
                "path_step": expected_state[2],
                "draw_id": expected_state[3],
                "candidate": candidate,
                "trust_mode": trust_mode,
                "radius_fraction": radius,
            }
            for candidate, trust_mode, radius in sorted(expected_suffixes)
        ]
    )
    expected_states = {expected_state}

    valid = _validate_row_suffix_grid(
        rows,
        expected_state_key_set=expected_states,
        dense_rotation_count=0,
        radius_fractions=(0.5,),
    )
    assert len(rows) == len(expected_suffixes)
    assert set(map(tuple, rows[["label", "source_weight_index", "path_step", "draw_id"]].drop_duplicates().to_numpy())) == expected_states
    assert valid["expected_rows_per_state"] == len(expected_suffixes)
    assert valid["bad_row_grid_state_count"] == 0
    assert valid["exact_row_grid"]

    mutated = rows.copy()
    mutated.loc[mutated["candidate"].eq("raw_adam_replay"), "candidate"] = "bogus"
    rejected = _validate_row_suffix_grid(
        mutated,
        expected_state_key_set=expected_states,
        dense_rotation_count=0,
        radius_fractions=(0.5,),
    )
    assert len(mutated) == len(rows)
    assert set(map(tuple, mutated[["label", "source_weight_index", "path_step", "draw_id"]].drop_duplicates().to_numpy())) == expected_states
    assert rejected["bad_row_grid_state_count"] == 1
    assert not rejected["exact_row_grid"]
