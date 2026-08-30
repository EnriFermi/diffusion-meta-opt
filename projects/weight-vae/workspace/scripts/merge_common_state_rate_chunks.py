#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_latent_raw_common_state_rate import (
    STATE_KEYS,
    _expected_row_suffixes,
    _plot,
    _row_state_integrity,
    _sha256_file,
    _stable_json_hash,
    _summaries,
)


def _log(message: str) -> None:
    print(f"[merge_common_state_rate] {message}", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _protocol_fingerprint(manifest: dict[str, Any]) -> str:
    request = dict(manifest.get("request", {}))
    for key in ("source_indices", "selected_start_rows_sha256"):
        request.pop(key, None)
    return _stable_json_hash(request)


def _validate_row_suffix_grid(
    rows: pd.DataFrame,
    *,
    expected_state_key_set: set[tuple[str, int, int, int]],
    dense_rotation_count: int,
    radius_fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Validate each expected state against the producer's exact row suffix grid."""
    expected_suffixes = _expected_row_suffixes(dense_rotation_count, radius_fractions)
    actual_suffixes_by_state: dict[tuple[str, int, int, int], set[tuple[str, str, float]]] = {}
    for state_key, group in rows.groupby(STATE_KEYS, sort=False, dropna=False):
        normalized_state_key = (
            str(state_key[0]),
            int(state_key[1]),
            int(state_key[2]),
            int(state_key[3]),
        )
        actual_suffixes_by_state[normalized_state_key] = {
            (str(row.candidate), str(row.trust_mode), float(row.radius_fraction))
            for row in group[["candidate", "trust_mode", "radius_fraction"]].itertuples(index=False)
        }
    bad_row_grid_state_count = sum(
        int(actual_suffixes_by_state.get(state_key, set()) != expected_suffixes)
        for state_key in expected_state_key_set
    )
    return {
        "expected_rows_per_state": int(len(expected_suffixes)),
        "bad_row_grid_state_count": int(bad_row_grid_state_count),
        "exact_row_grid": bool(bad_row_grid_state_count == 0),
    }


MERGED_OUTPUT_FILES = (
    "common_state_rate_rows.csv",
    "common_state_state_diagnostics.csv",
    "common_state_start_bank.csv",
    "common_state_selected_lrs.csv",
    "common_state_rate_summary.csv",
    "common_state_rate_summary_threshold_specific.csv",
    "common_state_paired_contrasts.csv",
    "common_state_paired_contrasts_threshold_specific.csv",
    "common_state_eligibility_coverage.csv",
    "common_state_rate_overview.png",
)


def _merged_output_hashes(output_dir: Path) -> dict[str, str]:
    paths = {name: output_dir / name for name in MERGED_OUTPUT_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"merged output artifacts are missing: {missing}")
    return {name: _sha256_file(path) for name, path in paths.items()}


def verify_merged_output_integrity(output_dir: Path) -> dict[str, str]:
    """Verify the immutable hash chain for a merged common-state packet."""
    output_dir = Path(output_dir).expanduser().resolve()
    validation_path = output_dir / "common_state_validation.json"
    manifest_path = output_dir / "manifest.json"
    validation = _read_json(validation_path)
    manifest = _read_json(manifest_path)
    expected_hashes = validation.get("output_hashes")
    manifest_hashes = manifest.get("output_hashes")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(MERGED_OUTPUT_FILES):
        raise ValueError("merged validation has an incomplete output_hashes set")
    if manifest_hashes != expected_hashes:
        raise ValueError("manifest output_hashes do not match validation output_hashes")
    actual_hashes = _merged_output_hashes(output_dir)
    mismatches = {
        name: (str(expected_hashes[name]), actual_hashes[name])
        for name in MERGED_OUTPUT_FILES
        if str(expected_hashes[name]) != actual_hashes[name]
    }
    if mismatches:
        raise ValueError(f"merged output hash mismatch: {mismatches}")
    validation_sha256 = manifest.get("validation_sha256")
    actual_validation_sha256 = _sha256_file(validation_path)
    if str(validation_sha256) != actual_validation_sha256:
        raise ValueError(
            "merged validation hash mismatch: "
            f"expected={validation_sha256} actual={actual_validation_sha256}"
        )
    unhashed_manifest = dict(manifest)
    request_hash = unhashed_manifest.pop("request_hash", None)
    if not request_hash or str(request_hash) != _stable_json_hash(unhashed_manifest):
        raise ValueError("merged manifest request_hash does not match its payload")
    return actual_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently run common-state chunks under one hard validation gate.")
    parser.add_argument("--chunk-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-starts", type=int, required=True)
    parser.add_argument("--expected-start-bank-csv", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    chunk_dirs = [path.expanduser().resolve() for path in args.chunk_dir]
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_bank_path = args.expected_start_bank_csv.expanduser().resolve()
    expected_bank = pd.read_csv(expected_bank_path).head(int(args.expected_starts)).copy().reset_index(drop=True)
    if len(expected_bank) != int(args.expected_starts):
        raise RuntimeError(
            f"expected start bank has {len(expected_bank)} rows, required={int(args.expected_starts)}: {expected_bank_path}"
        )
    expected_sources = [int(value) for value in expected_bank["source_weight_index"].tolist()]
    expected_bank_sha256 = hashlib.sha256(expected_bank.to_csv(index=False).encode("utf-8")).hexdigest()
    _log(f"startup chunks={[str(v) for v in chunk_dirs]} expected_starts={args.expected_starts} output_dir={out_dir}")
    manifests: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    row_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []
    bank_frames: list[pd.DataFrame] = []
    lr_frames: list[pd.DataFrame] = []
    for chunk_dir in chunk_dirs:
        manifest_path = chunk_dir / "manifest.json"
        validation_path = chunk_dir / "common_state_validation.json"
        required = [
            manifest_path,
            validation_path,
            chunk_dir / "common_state_rate_rows.csv",
            chunk_dir / "common_state_state_diagnostics.csv",
            chunk_dir / "common_state_start_bank.csv",
            chunk_dir / "common_state_selected_lrs.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete chunk {chunk_dir}: missing={missing}")
        manifest = _read_json(manifest_path)
        validation = _read_json(validation_path)
        child_request_hash_valid = bool(
            manifest.get("request_hash") == _stable_json_hash(manifest.get("request", {}))
            and validation.get("request_hash") == manifest.get("request_hash")
        )
        if not child_request_hash_valid:
            raise RuntimeError(f"child request hash mismatch: {manifest_path}")
        if not bool(validation.get("acceptance_pass", False)):
            raise RuntimeError(f"child validation rejected: {validation_path}")
        manifests.append(manifest)
        validations.append(validation)
        row_frames.append(pd.read_csv(chunk_dir / "common_state_rate_rows.csv"))
        state_frames.append(pd.read_csv(chunk_dir / "common_state_state_diagnostics.csv"))
        bank_frames.append(pd.read_csv(chunk_dir / "common_state_start_bank.csv"))
        lr_frames.append(pd.read_csv(chunk_dir / "common_state_selected_lrs.csv"))
        _log(
            f"accepted_child dir={chunk_dir} starts={validation.get('starts_per_label')} "
            f"states={validation.get('states')} rows={validation.get('rows')} request_hash={manifest.get('request_hash')}"
        )

    protocol_fingerprints = {_protocol_fingerprint(manifest) for manifest in manifests}
    rows = pd.concat(row_frames, ignore_index=True, sort=False)
    states = pd.concat(state_frames, ignore_index=True, sort=False)
    start_bank = pd.concat(bank_frames, ignore_index=True, sort=False)
    selected_lrs = pd.concat(lr_frames, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)
    row_key = ["label", "source_weight_index", "path_step", "draw_id", "candidate", "trust_mode", "radius_fraction"]
    state_key = STATE_KEYS
    sources = sorted(int(value) for value in start_bank["source_weight_index"].unique().tolist())
    path_steps = sorted(int(value) for value in rows["path_step"].unique().tolist())
    draws = sorted(int(value) for value in rows["draw_id"].unique().tolist())
    labels = sorted(rows["label"].astype(str).unique().tolist())
    duplicate_rows = int(rows.duplicated(row_key).sum())
    duplicate_states = int(states.duplicated(state_key).sum())
    duplicate_bank_sources = int(start_bank.duplicated(["source_weight_index"]).sum())
    ordered_bank = start_bank.sort_values("start_bank_position").reset_index(drop=True)
    comparison_columns = [column for column in expected_bank.columns if column in ordered_bank.columns]
    exact_bank_rows = bool(
        [int(value) for value in ordered_bank["source_weight_index"].tolist()] == expected_sources
        and ordered_bank[comparison_columns].reset_index(drop=True).equals(
            expected_bank[comparison_columns].reset_index(drop=True)
        )
    )
    reference_request = dict(manifests[0].get("request", {}))
    expected_labels = sorted(str(value) for value in reference_request.get("labels", []))
    expected_path_steps = sorted(int(value) for value in reference_request.get("checkpoint_steps", []))
    expected_draws = list(range(int(reference_request.get("batch_draws", 0))))
    expected_state_key_set = {
        (label, source, path_step, draw_id)
        for label in expected_labels
        for source in expected_sources
        for path_step in expected_path_steps
        for draw_id in expected_draws
    }
    try:
        dense_rotation_count = int(reference_request["dense_rotation_count"])
        radius_fractions = tuple(float(value) for value in reference_request["radius_fractions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "reference child request must contain dense_rotation_count and radius_fractions"
        ) from exc
    row_suffix_grid = _validate_row_suffix_grid(
        rows,
        expected_state_key_set=expected_state_key_set,
        dense_rotation_count=dense_rotation_count,
        radius_fractions=radius_fractions,
    )
    actual_state_key_set = {
        (str(row.label), int(row.source_weight_index), int(row.path_step), int(row.draw_id))
        for row in states[state_key].drop_duplicates().itertuples(index=False)
    }
    row_state_integrity = _row_state_integrity(
        rows,
        states,
        expected_state_key_set=expected_state_key_set,
        start_bank=ordered_bank,
        labels=expected_labels,
    )
    expected_states = int(len(expected_state_key_set))
    expected_rows = int(sum(int(value["expected_rows"]) for value in validations))
    rows_per_state = rows.groupby(state_key).size()
    child_protocol_hashes = sorted(protocol_fingerprints)
    validation = {
        "accepted_children": int(sum(bool(value.get("acceptance_pass", False)) for value in validations)),
        "expected_children": int(len(chunk_dirs)),
        "protocol_fingerprint_count": int(len(protocol_fingerprints)),
        "protocol_fingerprint": child_protocol_hashes[0] if len(child_protocol_hashes) == 1 else "",
        "starts": int(len(sources)),
        "expected_starts": int(args.expected_starts),
        "sources": sources,
        "expected_sources": expected_sources,
        "expected_start_bank_csv": str(expected_bank_path),
        "expected_start_bank_sha256": expected_bank_sha256,
        "exact_start_bank_rows": exact_bank_rows,
        "labels": labels,
        "expected_labels": expected_labels,
        "path_steps": path_steps,
        "expected_path_steps": expected_path_steps,
        "draws": draws,
        "expected_draws": expected_draws,
        "states": int(len(states)),
        "expected_states": expected_states,
        "rows": int(len(rows)),
        "expected_rows": expected_rows,
        "rows_per_state_min": int(rows_per_state.min()),
        "rows_per_state_max": int(rows_per_state.max()),
        "dense_rotation_count": dense_rotation_count,
        "radius_fractions": list(radius_fractions),
        **row_suffix_grid,
        "duplicate_rows": duplicate_rows,
        "duplicate_states": duplicate_states,
        "duplicate_bank_sources": duplicate_bank_sources,
        "missing_state_key_count": int(len(expected_state_key_set - actual_state_key_set)),
        "unexpected_state_key_count": int(len(actual_state_key_set - expected_state_key_set)),
        "all_finite_core": bool(
            np.isfinite(
                rows[
                    [
                        "delta_weight_norm",
                        "delta_function_rms",
                        "update_batch_loss_delta",
                        "eval_batch_loss_delta",
                        "full_train_loss_delta",
                        "test_loss_delta",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        ),
        "child_validation_sha256": [_sha256_file(path / "common_state_validation.json") for path in chunk_dirs],
        "child_manifest_sha256": [_sha256_file(path / "manifest.json") for path in chunk_dirs],
        "selected_lr_rows": int(len(selected_lrs)),
        "selected_lr_unique_per_label": bool(
            not selected_lrs.empty and selected_lrs.groupby("label").size().eq(1).all()
        ),
        **row_state_integrity,
    }
    checks = {
        "all_children_accepted": validation["accepted_children"] == validation["expected_children"],
        "same_protocol": validation["protocol_fingerprint_count"] == 1,
        "complete_starts": validation["starts"] == validation["expected_starts"],
        "exact_start_bank": validation["exact_start_bank_rows"],
        "complete_states": validation["states"] == validation["expected_states"],
        "exact_state_grid": (
            validation["labels"] == validation["expected_labels"]
            and validation["path_steps"] == validation["expected_path_steps"]
            and validation["draws"] == validation["expected_draws"]
            and validation["missing_state_key_count"] == 0
            and validation["unexpected_state_key_count"] == 0
        ),
        "complete_rows": validation["rows"] == validation["expected_rows"],
        "uniform_rows_per_state": (
            validation["rows_per_state_min"]
            == validation["expected_rows_per_state"]
            == validation["rows_per_state_max"]
        ),
        "no_duplicate_rows": validation["duplicate_rows"] == 0,
        "no_duplicate_states": validation["duplicate_states"] == 0,
        "disjoint_start_banks": validation["duplicate_bank_sources"] == 0,
        "all_finite": validation["all_finite_core"],
        "selected_lrs_consistent": validation["selected_lr_unique_per_label"],
        "exact_row_state_grid": (
            validation["row_state_group_count"] == validation["expected_states"]
            and validation["row_missing_expected_state_key_count"] == 0
            and validation["row_unexpected_state_key_count"] == 0
            and validation["row_missing_diagnostic_state_key_count"] == 0
            and validation["row_orphan_state_key_count"] == 0
        ),
        "exact_row_grid": validation["exact_row_grid"],
        "row_state_linkage": (
            validation["row_state_many_to_one_valid"]
            and validation["row_state_unmatched_row_count"] == 0
            and validation["row_state_context_mismatch_row_count"] == 0
        ),
        "state_start_context": (
            validation["state_start_context_unmatched_count"] == 0
            and validation["state_start_context_mismatch_row_count"] == 0
        ),
    }
    validation["acceptance_checks"] = checks
    validation["acceptance_pass"] = bool(all(checks.values()))
    if not validation["acceptance_pass"]:
        raise RuntimeError(f"merged validation rejected: {json.dumps(validation, sort_keys=True)}")

    rows.to_csv(out_dir / "common_state_rate_rows.csv", index=False)
    states.to_csv(out_dir / "common_state_state_diagnostics.csv", index=False)
    ordered_bank.to_csv(out_dir / "common_state_start_bank.csv", index=False)
    selected_lrs.to_csv(out_dir / "common_state_selected_lrs.csv", index=False)
    paired = _summaries(rows, out_dir, eligibility_column="threshold_robust_eligible")
    paired_threshold_specific = _summaries(
        rows,
        out_dir,
        eligibility_column="threshold_specific_eligible",
        suffix="_threshold_specific",
        write_coverage=False,
    )
    _plot(rows, out_dir)
    validation["paired_contrast_rows"] = int(len(paired))
    validation["paired_threshold_specific_contrast_rows"] = int(len(paired_threshold_specific))
    output_hashes = _merged_output_hashes(out_dir)
    validation["output_hashes"] = output_hashes
    (out_dir / "common_state_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "protocol_version": "common_state_rate_v3_merged",
        "chunks": [str(value) for value in chunk_dirs],
        "output_dir": str(out_dir),
        "protocol_fingerprint": validation["protocol_fingerprint"],
        "sources": sources,
        "labels": labels,
        "path_steps": path_steps,
        "draws": draws,
        "expected_start_bank_csv": str(expected_bank_path),
        "expected_start_bank_sha256": expected_bank_sha256,
        "child_request_hashes": [str(value.get("request_hash")) for value in manifests],
        "child_producer_script_sha256": [str(value.get("request", {}).get("script_sha256")) for value in manifests],
        "merge_validator_script_sha256": _sha256_file(Path(__file__).resolve()),
        "output_hashes": output_hashes,
        "validation_sha256": _sha256_file(out_dir / "common_state_validation.json"),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    manifest["request_hash"] = _stable_json_hash(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    verify_merged_output_integrity(out_dir)
    _log(
        f"done acceptance_pass=True starts={len(sources)} states={len(states)} rows={len(rows)} "
        f"contrasts={len(paired)} elapsed_sec={time.perf_counter()-started:.2f} output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
