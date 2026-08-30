#!/usr/bin/env python
"""Integrity checked, strata preserving aggregation for common-state V3 packets.

The packet merger owns row production and the aggregator owns only paired
estimands.  In particular, ``native`` and ``zero`` are diagnostics, not arms
of a trust-matched causal contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECTOR_RCONDS = (1.0e-6, 1.0e-5, 1.0e-4)
EXPECTED_STARTS = 16
EXPECTED_PATH_STEPS = (1, 5, 25, 100)
EXPECTED_DRAWS = (0, 1)
EXPECTED_RADII = (0.125, 0.25, 0.5, 1.0)
EXPECTED_DENSE_ROTATIONS = 4
STATE_KEYS = ["label", "source_weight_index", "path_step", "draw_id"]
PAIR_KEYS = [
    "vae_seed_label",
    "label",
    "source_weight_index",
    "path_step",
    "draw_id",
    "radius_fraction",
    "trust_mode",
]
METRIC = "full_train_loss_delta"
RETRACTION_METRIC = "nonlinear_minus_same_scale_linear_full_train_loss"
PRIMARY_TRUST_MODES = ("weight", "function")
DIAGNOSTIC_TRUST_MODES = ("native", "zero")

# This list is intentionally copied from the merger's public integrity
# contract.  The merger worker may update its validator hash; changing one
# constant here is then the only aggregator edit required for that review.
REVIEWED_MERGE_VALIDATOR_SHA256 = "fb48d170ed8d0ea6d700c5fa437b6486c8e41fb9b4e5c69f9138f6ed6b6ea7d0"
REVIEWED_PRODUCER_SHA256S = frozenset(
    {
        "9d720e542f8abcd5a04653a5c3ebecbe69289b3047c538febf0fb81e51c8a59d",
        "bcfa463702f1d0baf0fe023c9f1588065e1dbef7134a248f9290f4a0f89e545c",
    }
)
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
REQUIRED_PACKET_FILES = (
    "manifest.json",
    "common_state_validation.json",
    "common_state_rate_rows.csv",
    "common_state_state_diagnostics.csv",
    "common_state_start_bank.csv",
    "common_state_selected_lrs.csv",
)
STATE_CONTEXT_COLUMNS = [
    "run_name",
    "start_bank_position",
    "stream_start_index",
    "task_name",
    "tau",
    "raw_lr",
    "latent_lr",
    "update_batch_sha256",
    "eval_batch_sha256",
    "calibration_images_sha256",
    "calibration_indices_sha256",
    "theta_sha256",
    "z_sha256",
    "path_adam_step",
]


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    left: str
    right: str | None
    projector_rcond: float | None
    cutoff_applicability: str = "rcond_specific"
    value_column: str = METRIC
    contrast_kind: str = "paired"
    control_left: str | None = None
    control_right: str | None = None


@dataclass
class Packet:
    path: Path
    label: str
    rows: pd.DataFrame
    states: pd.DataFrame
    start_bank: pd.DataFrame
    selected_lrs: pd.DataFrame
    manifest: dict[str, Any]
    validation: dict[str, Any]
    dimensions: dict[str, Any]
    input_hashes: dict[str, str]


def _log(message: str) -> None:
    print(f"[aggregate_common_state_rate_seeds] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _bool_series(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized.unique()) - {"true", "false"})
    if unexpected:
        raise ValueError(f"column {column} has non-boolean values: {unexpected[:5]}")
    return normalized.eq("true")


def _rcond_tag(rcond: float) -> str:
    for value, tag in ((1.0e-6, "1e-6"), (1.0e-5, "1e-5"), (1.0e-4, "1e-4")):
        if math.isclose(float(rcond), value, rel_tol=0.0, abs_tol=1.0e-15):
            return tag
    raise ValueError(f"unsupported projector rcond: {rcond}")


def _expected_candidates(dense_rotation_count: int = EXPECTED_DENSE_ROTATIONS) -> tuple[set[str], set[str]]:
    candidates = {
        "raw_adam_replay",
        "raw_adam_fresh",
        "jjt_direct_latent_sgd_linear",
        "latent_sgd_nonlinear",
        "latent_adam_replay_linear",
        "latent_adam_replay_nonlinear",
        "latent_adam_fresh_linear",
        "latent_adam_fresh_nonlinear",
        "latent_adam_rotation_identity_linear",
        "latent_adam_rotation_identity_nonlinear",
    }
    native = {"raw_adam_replay", "raw_adam_fresh", "latent_adam_replay_linear", "latent_adam_replay_nonlinear",
              "latent_adam_fresh_linear", "latent_adam_fresh_nonlinear", "latent_adam_rotation_identity_linear",
              "latent_adam_rotation_identity_nonlinear"}
    for rcond in PROJECTOR_RCONDS:
        tag = _rcond_tag(rcond)
        candidates.update(
            {
                f"tangent_oracle_rcond{tag}",
                f"tangent_oracle_rcond{tag}_nonlinear",
                f"projected_raw_adam_rcond{tag}",
                f"projected_raw_adam_rcond{tag}_nonlinear",
                f"jjt_plus_normal_rcond{tag}_eps0.25",
                f"jjt_plus_random_normal_placebo_rcond{tag}_eps0.25",
                f"jjt_plus_normal_rcond{tag}_eps1",
                f"jjt_plus_random_normal_placebo_rcond{tag}_eps1",
            }
        )
    for rotation_id in range(int(dense_rotation_count)):
        for suffix in ("linear", "nonlinear"):
            candidate = f"latent_adam_rotation_dense{rotation_id}_{suffix}"
            candidates.add(candidate)
            native.add(candidate)
    return candidates, native


def _expected_row_suffixes() -> set[tuple[str, str, float]]:
    candidates, native = _expected_candidates()
    suffixes = {
        (candidate, trust_mode, radius)
        for candidate in candidates
        for trust_mode in PRIMARY_TRUST_MODES
        for radius in EXPECTED_RADII
    }
    suffixes.update((candidate, "native", 1.0) for candidate in native)
    suffixes.add(("raw_adam_replay", "zero", 0.0))
    return suffixes


def _rcond_specs() -> list[ContrastSpec]:
    specs: list[ContrastSpec] = []
    for rcond in PROJECTOR_RCONDS:
        tag = _rcond_tag(rcond)
        tangent = f"tangent_oracle_rcond{tag}"
        tangent_nl = f"{tangent}_nonlinear"
        projected = f"projected_raw_adam_rcond{tag}"
        projected_nl = f"{projected}_nonlinear"
        normal25 = f"jjt_plus_normal_rcond{tag}_eps0.25"
        placebo25 = f"jjt_plus_random_normal_placebo_rcond{tag}_eps0.25"
        normal1 = f"jjt_plus_normal_rcond{tag}_eps1"
        placebo1 = f"jjt_plus_random_normal_placebo_rcond{tag}_eps1"
        specs.extend(
            [
                ContrastSpec("tangent_linear_minus_jjt_linear", tangent, "jjt_direct_latent_sgd_linear", rcond),
                ContrastSpec("tangent_nonlinear_minus_latent_sgd_nonlinear", tangent_nl, "latent_sgd_nonlinear", rcond),
                ContrastSpec("latent_adam_replay_linear_minus_jjt_linear", "latent_adam_replay_linear", "jjt_direct_latent_sgd_linear", rcond),
                ContrastSpec("latent_adam_replay_nonlinear_minus_latent_sgd_nonlinear", "latent_adam_replay_nonlinear", "latent_sgd_nonlinear", rcond),
                ContrastSpec("projected_raw_linear_minus_raw", projected, "raw_adam_replay", rcond),
                ContrastSpec("projected_raw_nonlinear_minus_raw", projected_nl, "raw_adam_replay", rcond),
                ContrastSpec("projected_raw_linear_minus_tangent", projected, tangent, rcond),
                ContrastSpec("projected_raw_nonlinear_minus_tangent", projected_nl, tangent_nl, rcond),
                ContrastSpec("task_normal_minus_placebo_eps0.25", normal25, placebo25, rcond),
                ContrastSpec("task_normal_minus_jjt_eps0.25", normal25, "jjt_direct_latent_sgd_linear", rcond),
                ContrastSpec("placebo_minus_jjt_eps0.25", placebo25, "jjt_direct_latent_sgd_linear", rcond),
                ContrastSpec("task_normal_minus_placebo_eps1", normal1, placebo1, rcond),
                ContrastSpec("task_normal_minus_jjt_eps1", normal1, "jjt_direct_latent_sgd_linear", rcond),
                ContrastSpec("placebo_minus_jjt_eps1", placebo1, "jjt_direct_latent_sgd_linear", rcond),
                ContrastSpec("nonlinear_retraction_tangent", tangent_nl, "__zero__", rcond, value_column=RETRACTION_METRIC, contrast_kind="unary"),
                ContrastSpec("nonlinear_retraction_projected_raw", projected_nl, "__zero__", rcond, value_column=RETRACTION_METRIC, contrast_kind="unary"),
            ]
        )
    return specs


def contrast_specs() -> list[ContrastSpec]:
    """Return the complete predeclared mechanism contrast catalogue.

    Rcond-specific rows are emitted once for each fixed cutoff.  The history,
    rotation, latent nonlinear, and retraction contrasts are cutoff
    independent and deliberately have ``projector_rcond=None``.
    """
    specs = _rcond_specs()
    independent = [
        ContrastSpec("latent_sgd_nonlinear_retraction", "latent_sgd_nonlinear", "__zero__", None, "cutoff_independent", RETRACTION_METRIC, "unary"),
        ContrastSpec("latent_adam_replay_nonlinear_retraction", "latent_adam_replay_nonlinear", "__zero__", None, "cutoff_independent", RETRACTION_METRIC, "unary"),
        ContrastSpec("latent_adam_fresh_nonlinear_retraction", "latent_adam_fresh_nonlinear", "__zero__", None, "cutoff_independent", RETRACTION_METRIC, "unary"),
        ContrastSpec("raw_replay_minus_raw_fresh", "raw_adam_replay", "raw_adam_fresh", None, "cutoff_independent"),
        ContrastSpec("latent_replay_linear_minus_latent_fresh_linear", "latent_adam_replay_linear", "latent_adam_fresh_linear", None, "cutoff_independent"),
        ContrastSpec("latent_replay_nonlinear_minus_latent_fresh_nonlinear", "latent_adam_replay_nonlinear", "latent_adam_fresh_nonlinear", None, "cutoff_independent"),
        ContrastSpec("history_difference_in_differences_linear", "latent_adam_replay_linear", "latent_adam_fresh_linear", None, "cutoff_independent", METRIC, "history_did", "raw_adam_replay", "raw_adam_fresh"),
        ContrastSpec("history_difference_in_differences_nonlinear", "latent_adam_replay_nonlinear", "latent_adam_fresh_nonlinear", None, "cutoff_independent", METRIC, "history_did", "raw_adam_replay", "raw_adam_fresh"),
    ]
    for rotation_id in range(EXPECTED_DENSE_ROTATIONS):
        for suffix in ("linear", "nonlinear"):
            independent.append(
                ContrastSpec(
                    f"rotation_dense{rotation_id}_minus_identity_{suffix}",
                    f"latent_adam_rotation_dense{rotation_id}_{suffix}",
                    f"latent_adam_rotation_identity_{suffix}",
                    None,
                    "cutoff_independent",
                )
            )
    return specs + independent


def _validate_provenance(
    manifest: dict[str, Any],
    validation: dict[str, Any],
    *,
    expected_producer_sha256s: frozenset[str],
    expected_validator_sha256: str,
) -> None:
    if manifest.get("protocol_version") != "common_state_rate_v3_merged":
        raise ValueError(f"not a merged common-state v3 packet: protocol_version={manifest.get('protocol_version')!r}")
    recorded_hash = str(manifest.get("request_hash", ""))
    unhashed = dict(manifest)
    unhashed.pop("request_hash", None)
    if recorded_hash != _stable_json_hash(unhashed):
        raise ValueError("merged manifest request_hash does not match its payload")
    if str(manifest.get("merge_validator_script_sha256", "")) != expected_validator_sha256:
        raise ValueError("merged packet was not produced by the reviewed merge validator revision")
    producers = manifest.get("child_producer_script_sha256")
    if not isinstance(producers, list) or not producers:
        raise ValueError("merged manifest has no child producer provenance")
    if any(str(value) not in expected_producer_sha256s for value in producers):
        raise ValueError("merged packet contains an unreviewed child producer revision")
    if not isinstance(manifest.get("output_hashes"), dict) or not isinstance(validation.get("output_hashes"), dict):
        raise ValueError("hash-bound merged packet must contain output_hashes in manifest and validation")
    if set(manifest["output_hashes"]) != set(MERGED_OUTPUT_FILES) or manifest["output_hashes"] != validation["output_hashes"]:
        raise ValueError("merged output_hashes are incomplete or disagree between manifest and validation")
    if not isinstance(manifest.get("validation_sha256"), str) or len(manifest["validation_sha256"]) != 64:
        raise ValueError("hash-bound merged packet has no validation_sha256")
    checks = validation.get("acceptance_checks")
    if not bool(validation.get("acceptance_pass", False)):
        raise ValueError("merged packet acceptance_pass is false")
    if not isinstance(checks, dict) or not checks or not all(bool(value) for value in checks.values()):
        raise ValueError("merged packet has a failed or incomplete acceptance check set")


def _validate_child_chain(
    packet_dir: Path,
    manifest: dict[str, Any],
    validation: dict[str, Any],
    *,
    expected_producer_sha256s: frozenset[str],
) -> None:
    chunks = manifest.get("chunks")
    manifest_hashes = validation.get("child_manifest_sha256")
    validation_hashes = validation.get("child_validation_sha256")
    request_hashes = manifest.get("child_request_hashes")
    if not all(isinstance(value, list) for value in (chunks, manifest_hashes, validation_hashes, request_hashes)):
        raise ValueError(f"incomplete child provenance arrays: {packet_dir}")
    if len({len(chunks), len(manifest_hashes), len(validation_hashes), len(request_hashes)}) != 1 or not chunks:
        raise ValueError(f"child provenance array lengths disagree: {packet_dir}")
    if int(validation.get("accepted_children", -1)) != len(chunks):
        raise ValueError(f"accepted child count disagrees with manifest: {packet_dir}")
    for index, raw_chunk in enumerate(chunks):
        child_dir = Path(str(raw_chunk)).expanduser()
        child_manifest_path = child_dir / "manifest.json"
        child_validation_path = child_dir / "common_state_validation.json"
        if not child_manifest_path.is_file() or not child_validation_path.is_file():
            raise FileNotFoundError(f"child provenance target is unavailable: {child_dir}")
        if _sha256_file(child_manifest_path) != str(manifest_hashes[index]):
            raise ValueError(f"child manifest hash mismatch: {child_manifest_path}")
        if _sha256_file(child_validation_path) != str(validation_hashes[index]):
            raise ValueError(f"child validation hash mismatch: {child_validation_path}")
        child_manifest = _read_json(child_manifest_path)
        child_validation = _read_json(child_validation_path)
        request = child_manifest.get("request")
        if not isinstance(request, dict) or str(request.get("protocol_version")) != "common_state_rate_v3":
            raise ValueError(f"child request is not common_state_rate_v3: {child_manifest_path}")
        if str(request.get("script_sha256")) not in expected_producer_sha256s:
            raise ValueError(f"child producer hash is unreviewed: {child_manifest_path}")
        child_hash = str(child_manifest.get("request_hash", ""))
        if child_hash != _stable_json_hash(request) or child_hash != str(request_hashes[index]):
            raise ValueError(f"child request hash mismatch: {child_manifest_path}")
        if str(child_validation.get("request_hash", "")) != child_hash or not bool(child_validation.get("acceptance_pass", False)):
            raise ValueError(f"child validation is rejected: {child_validation_path}")


def _canonical_bank_hash(start_bank: pd.DataFrame) -> str:
    required = {"start_bank_position", "source_weight_index"}
    missing = sorted(required - set(start_bank.columns))
    if missing:
        raise ValueError(f"start bank missing columns: {missing}")
    ordered = start_bank.sort_values("start_bank_position").reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def _equal_series(left: pd.Series, right: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
        return pd.Series(
            np.isclose(
                pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ),
            index=left.index,
        )
    sentinel = "__common_state_missing__"
    return left.astype("string").fillna(sentinel).eq(right.astype("string").fillna(sentinel))


def _validate_row_state_linkage(rows: pd.DataFrame, states: pd.DataFrame, start_bank: pd.DataFrame) -> None:
    if states.duplicated(STATE_KEYS).any():
        raise ValueError("diagnostic state keys are not unique")
    context = [column for column in STATE_CONTEXT_COLUMNS if column in rows.columns and column in states.columns]
    joined = rows.merge(
        states[STATE_KEYS + context + ["singularity_start_any_state"]],
        on=STATE_KEYS,
        how="left",
        suffixes=("_row", "_state"),
        indicator=True,
        validate="many_to_one",
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("row-to-diagnostics many-to-one join has unmatched rows")
    for column in context:
        if not _equal_series(joined[f"{column}_row"], joined[f"{column}_state"]).all():
            raise ValueError(f"row-state context mismatch: {column}")
    row_singular = _bool_series(joined["singularity_start_any_state_row"], column="singularity_start_any_state")
    state_singular = _bool_series(joined["singularity_start_any_state_state"], column="singularity_start_any_state")
    if not row_singular.eq(state_singular).all():
        raise ValueError("row-state diagnostic singularity mismatch")

    bank = start_bank.copy()
    bank["source_weight_index"] = pd.to_numeric(bank["source_weight_index"], errors="raise").astype(int)
    expected = bank[["source_weight_index", "start_bank_position", "task_name", "tau"]].copy()
    expected["label"] = str(states["label"].astype(str).iloc[0])
    if "stream_start_index" in states.columns:
        expected["expected_stream_start_index"] = (
            bank["start_index"] if "start_index" in bank.columns else bank["start_bank_position"]
        ).astype(int)
    sj = states.merge(expected, on=["label", "source_weight_index"], how="left", suffixes=("_state", "_expected"), indicator=True, validate="many_to_one")
    if sj["_merge"].ne("both").any():
        raise ValueError("diagnostics-to-start-bank context has unmatched states")
    for column in ("start_bank_position", "task_name", "tau"):
        if not _equal_series(sj[f"{column}_state"], sj[f"{column}_expected"]).all():
            raise ValueError(f"diagnostic-start context mismatch: {column}")
    if "stream_start_index" in states.columns:
        if not _equal_series(sj["stream_start_index"], sj["expected_stream_start_index"]).all():
            raise ValueError("diagnostic-start context mismatch: stream_start_index")


def _validate_packet_grid(rows: pd.DataFrame, states: pd.DataFrame, start_bank: pd.DataFrame, selected_lrs: pd.DataFrame, validation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    required_rows = set(STATE_KEYS) | {"candidate", "trust_mode", "radius_fraction", "threshold_specific_eligible", "technical_eligible", "singularity_start_any_state", METRIC}
    required_states = set(STATE_KEYS) | {"singularity_start_any_state"}
    missing_rows = sorted(required_rows - set(rows.columns))
    missing_states = sorted(required_states - set(states.columns))
    if missing_rows or missing_states:
        raise ValueError(f"packet schema incomplete: missing_rows={missing_rows} missing_states={missing_states}")
    labels = sorted(rows["label"].astype(str).unique().tolist())
    if len(labels) != 1 or sorted(states["label"].astype(str).unique().tolist()) != labels:
        raise ValueError(f"each VAE packet must contain exactly one shared label: rows={labels}")
    label = labels[0]
    starts = start_bank.sort_values("start_bank_position")
    sources = [int(value) for value in starts["source_weight_index"].tolist()]
    if len(sources) != EXPECTED_STARTS or len(set(sources)) != EXPECTED_STARTS:
        raise ValueError(f"expected {EXPECTED_STARTS} unique starts, got {len(sources)} rows/{len(set(sources))} unique")
    expected_state_keys = {(label, source, step, draw) for source in sources for step in EXPECTED_PATH_STEPS for draw in EXPECTED_DRAWS}
    state_keys = {(str(r.label), int(r.source_weight_index), int(r.path_step), int(r.draw_id)) for r in states[STATE_KEYS].itertuples(index=False)}
    row_keys = {(str(r.label), int(r.source_weight_index), int(r.path_step), int(r.draw_id)) for r in rows[STATE_KEYS].drop_duplicates().itertuples(index=False)}
    if state_keys != expected_state_keys or row_keys != expected_state_keys:
        raise ValueError(f"exact state grid failed: state_missing={len(expected_state_keys-state_keys)} row_missing={len(expected_state_keys-row_keys)}")
    duplicate_key = STATE_KEYS + ["candidate", "trust_mode", "radius_fraction"]
    if rows.duplicated(duplicate_key).any() or states.duplicated(STATE_KEYS).any():
        raise ValueError("duplicate row or diagnostic state keys")
    expected_suffixes = _expected_row_suffixes()
    bad = 0
    for _, group in rows.groupby(STATE_KEYS, sort=False):
        observed = {(str(r.candidate), str(r.trust_mode), float(r.radius_fraction)) for r in group[["candidate", "trust_mode", "radius_fraction"]].itertuples(index=False)}
        bad += int(observed != expected_suffixes)
    if bad:
        raise ValueError(f"row suffix grid is not exact: bad_state_groups={bad}")
    expected_rows = len(expected_state_keys) * len(expected_suffixes)
    if len(rows) != expected_rows or len(states) != len(expected_state_keys):
        raise ValueError(f"row formula mismatch: rows={len(rows)}/{expected_rows} states={len(states)}/{len(expected_state_keys)}")
    threshold = _bool_series(rows["threshold_specific_eligible"], column="threshold_specific_eligible")
    technical = _bool_series(rows["technical_eligible"], column="technical_eligible")
    if (threshold & ~technical).any():
        raise ValueError("threshold_specific_eligible implies technical_eligible")
    if not np.isfinite(pd.to_numeric(rows[METRIC], errors="coerce").to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{METRIC} contains non-finite values")
    singular = _bool_series(rows["singularity_start_any_state"], column="singularity_start_any_state")
    if rows.assign(_s=singular).groupby(["label", "source_weight_index"])["_s"].nunique().ne(1).any():
        raise ValueError("singularity_start_any_state is not constant within a start")
    _validate_row_state_linkage(rows, states, starts)
    if sorted(int(value) for value in validation.get("sources", [])) != sorted(sources):
        raise ValueError("merged validation source grid disagrees with start bank")
    expected_dimensions = {
        "starts": EXPECTED_STARTS,
        "expected_starts": EXPECTED_STARTS,
        "states": len(expected_state_keys),
        "expected_states": len(expected_state_keys),
        "rows": expected_rows,
        "expected_rows": expected_rows,
        "rows_per_state_min": len(expected_suffixes),
        "rows_per_state_max": len(expected_suffixes),
    }
    for key, expected in expected_dimensions.items():
        if validation.get(key) != expected:
            raise ValueError(f"merged validation {key} disagrees: {validation.get(key)!r} != {expected!r}")
    if len(selected_lrs) != 1 or selected_lrs["label"].astype(str).tolist() != [label]:
        raise ValueError("selected LR table must have exactly one row for the packet label")
    dimensions = {
        "sources": sources,
        "start_bank_sha256": _canonical_bank_hash(starts),
        "path_steps": list(EXPECTED_PATH_STEPS),
        "draws": list(EXPECTED_DRAWS),
        "radius_fractions": list(EXPECTED_RADII),
        "projector_rconds": list(PROJECTOR_RCONDS),
        "dense_rotations": EXPECTED_DENSE_ROTATIONS,
        "rows_per_state": len(expected_suffixes),
        "states": len(expected_state_keys),
        "rows": expected_rows,
    }
    return label, dimensions


def _verify_merged_hashes(packet_dir: Path, manifest: dict[str, Any], validation: dict[str, Any]) -> dict[str, str]:
    expected = manifest["output_hashes"]
    actual: dict[str, str] = {}
    for name in MERGED_OUTPUT_FILES:
        path = packet_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"hash-bound merged artifact is unavailable: {path}")
        actual[name] = _sha256_file(path)
    mismatches = {name: (str(expected[name]), actual[name]) for name in MERGED_OUTPUT_FILES if str(expected[name]) != actual[name]}
    if mismatches:
        raise ValueError(f"merged output hash mismatch: {mismatches}")
    validation_path = packet_dir / "common_state_validation.json"
    if str(manifest.get("validation_sha256")) != _sha256_file(validation_path):
        raise ValueError("merged validation_sha256 does not match common_state_validation.json")
    return actual


def load_packet(packet_dir: Path, *, expected_producer_sha256s: frozenset[str] = REVIEWED_PRODUCER_SHA256S, expected_validator_sha256: str = REVIEWED_MERGE_VALIDATOR_SHA256) -> Packet:
    packet_dir = packet_dir.expanduser().resolve()
    missing = [str(packet_dir / name) for name in REQUIRED_PACKET_FILES if not (packet_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete merged packet {packet_dir}: missing={missing}")
    manifest = _read_json(packet_dir / "manifest.json")
    validation = _read_json(packet_dir / "common_state_validation.json")
    _validate_provenance(manifest, validation, expected_producer_sha256s=expected_producer_sha256s, expected_validator_sha256=expected_validator_sha256)
    _verify_merged_hashes(packet_dir, manifest, validation)
    _validate_child_chain(packet_dir, manifest, validation, expected_producer_sha256s=expected_producer_sha256s)
    rows = pd.read_csv(packet_dir / "common_state_rate_rows.csv")
    states = pd.read_csv(packet_dir / "common_state_state_diagnostics.csv")
    start_bank = pd.read_csv(packet_dir / "common_state_start_bank.csv")
    selected_lrs = pd.read_csv(packet_dir / "common_state_selected_lrs.csv")
    label, dimensions = _validate_packet_grid(rows, states, start_bank, selected_lrs, validation)
    input_hashes = {name: _sha256_file(packet_dir / name) for name in MERGED_OUTPUT_FILES}
    return Packet(packet_dir, label, rows, states, start_bank, selected_lrs, manifest, validation, dimensions, input_hashes)


def validate_cross_packet_dimensions(packets: list[Packet]) -> None:
    if len(packets) != 3:
        raise ValueError(f"exactly three merged VAE packets are required, got {len(packets)}")
    labels = [packet.label for packet in packets]
    if len(set(labels)) != len(labels):
        raise ValueError(f"VAE packet labels must be unique: {labels}")
    seed_ids = []
    for label in labels:
        match = re.search(r"seed(\d+)$", label)
        if match is None:
            raise ValueError(f"VAE label must end in seed<integer>: {label!r}")
        seed_ids.append(int(match.group(1)))
    if sorted(seed_ids) != [0, 1, 2]:
        raise ValueError(f"expected VAE seed labels 0/1/2, got {seed_ids}")
    reference = packets[0].dimensions
    for packet in packets[1:]:
        if packet.dimensions != reference:
            differences = {key: (reference.get(key), packet.dimensions.get(key)) for key in sorted(set(reference) | set(packet.dimensions)) if reference.get(key) != packet.dimensions.get(key)}
            raise ValueError(f"protocol dimensions differ for {packet.label}: {differences}")


def _arm_frame(rows: pd.DataFrame, candidate: str, trust_mode: str, value_column: str) -> pd.DataFrame:
    frame = rows[rows["candidate"].astype(str).eq(candidate) & rows["trust_mode"].astype(str).eq(trust_mode)].copy()
    if frame.empty:
        raise ValueError(f"missing contrast arm: candidate={candidate} trust_mode={trust_mode}")
    frame["threshold_specific_eligible"] = _bool_series(frame["threshold_specific_eligible"], column="threshold_specific_eligible")
    frame["technical_eligible"] = _bool_series(frame["technical_eligible"], column="technical_eligible")
    frame["singularity_start_any_state"] = _bool_series(frame["singularity_start_any_state"], column="singularity_start_any_state")
    if (frame["threshold_specific_eligible"] & ~frame["technical_eligible"]).any():
        raise ValueError("threshold_specific_eligible implies technical_eligible")
    if value_column not in frame.columns:
        raise ValueError(f"contrast value column is missing: {value_column}")
    keys = [key for key in PAIR_KEYS if key in frame.columns]
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate arm rows for candidate={candidate} trust_mode={trust_mode}")
    return frame[keys + [value_column, "threshold_specific_eligible", "technical_eligible", "singularity_start_any_state"]].copy()


def _join_arm(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> pd.DataFrame:
    keys = [key for key in PAIR_KEYS if key in left.columns and key in right.columns]
    joined = left.merge(right, on=keys, how="outer", suffixes=("_left", "_right"), indicator=True, validate="one_to_one")
    if joined["_merge"].ne("both").any():
        raise ValueError(f"missing contrast arm: contrast={label}")
    if not _equal_series(joined["singularity_start_any_state_left"], joined["singularity_start_any_state_right"]).all():
        raise ValueError(f"singularity flags disagree between arms: contrast={label}")
    return joined


def build_paired_cells(rows: pd.DataFrame, specs: Iterable[ContrastSpec] | None = None) -> pd.DataFrame:
    specs = list(specs if specs is not None else contrast_specs())
    required = {"label", "source_weight_index", "path_step", "draw_id", "radius_fraction", "candidate", "trust_mode", "threshold_specific_eligible", "technical_eligible", "singularity_start_any_state"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"pairing rows missing columns: {missing}")
    work = rows.copy()
    if "vae_seed_label" not in work.columns:
        work["vae_seed_label"] = work["label"].astype(str)
    threshold = _bool_series(work["threshold_specific_eligible"], column="threshold_specific_eligible")
    technical = _bool_series(work["technical_eligible"], column="technical_eligible")
    if (threshold & ~technical).any():
        raise ValueError("threshold_specific_eligible implies technical_eligible")
    frames: list[pd.DataFrame] = []
    available_modes = [mode for mode in PRIMARY_TRUST_MODES if work["trust_mode"].astype(str).eq(mode).any()]
    if not available_modes:
        raise ValueError("no primary trust modes (weight/function) are available")
    for spec in specs:
        for trust_mode in available_modes:
            left = _arm_frame(work, spec.left, trust_mode, spec.value_column)
            if spec.contrast_kind == "history_did":
                assert spec.right is not None and spec.control_left is not None and spec.control_right is not None
                latent_fresh = _arm_frame(work, spec.right, trust_mode, spec.value_column)
                raw_replay = _arm_frame(work, spec.control_left, trust_mode, spec.value_column)
                raw_fresh = _arm_frame(work, spec.control_right, trust_mode, spec.value_column)
                lr = _join_arm(left, latent_fresh, label=spec.name)
                rr = _join_arm(raw_replay, raw_fresh, label=spec.name)
                keys = [key for key in PAIR_KEYS if key in lr.columns and key in rr.columns]
                joined = lr.merge(rr, on=keys, suffixes=("_latent", "_raw"), validate="one_to_one")
                singular_columns = [
                    "singularity_start_any_state_left_latent",
                    "singularity_start_any_state_right_latent",
                    "singularity_start_any_state_left_raw",
                    "singularity_start_any_state_right_raw",
                ]
                singular_values = [joined[column].astype(bool) for column in singular_columns]
                if not all(singular_values[0].eq(value).all() for value in singular_values[1:]):
                    raise ValueError(f"singularity flags disagree between history arms: contrast={spec.name}")
                joined["left_value"] = pd.to_numeric(joined[f"{spec.value_column}_left_latent"], errors="coerce")
                joined["right_value"] = pd.to_numeric(joined[f"{spec.value_column}_right_latent"], errors="coerce")
                joined["control_left_value"] = pd.to_numeric(joined[f"{spec.value_column}_left_raw"], errors="coerce")
                joined["control_right_value"] = pd.to_numeric(joined[f"{spec.value_column}_right_raw"], errors="coerce")
                joined["contrast_value"] = joined["left_value"] - joined["right_value"] - joined["control_left_value"] + joined["control_right_value"]
                joined["left_threshold_specific_eligible"] = joined["threshold_specific_eligible_left_latent"] & joined["threshold_specific_eligible_right_latent"]
                joined["right_threshold_specific_eligible"] = joined["threshold_specific_eligible_left_raw"] & joined["threshold_specific_eligible_right_raw"]
                joined["pair_eligible"] = joined["left_threshold_specific_eligible"] & joined["right_threshold_specific_eligible"]
                joined["singularity_start_any_state"] = joined["singularity_start_any_state_left_latent"]
            else:
                if spec.right in (None, "__zero__"):
                    joined = left
                    joined["left_value"] = pd.to_numeric(joined[spec.value_column], errors="coerce")
                    joined["right_value"] = 0.0
                    joined["left_threshold_specific_eligible"] = joined["threshold_specific_eligible"]
                    joined["right_threshold_specific_eligible"] = joined["threshold_specific_eligible"]
                    joined["pair_eligible"] = joined["threshold_specific_eligible"]
                    joined["singularity_start_any_state"] = joined["singularity_start_any_state"]
                else:
                    right = _arm_frame(work, spec.right, trust_mode, spec.value_column)
                    joined = _join_arm(left, right, label=spec.name)
                    joined["left_value"] = pd.to_numeric(joined[f"{spec.value_column}_left"], errors="coerce")
                    joined["right_value"] = pd.to_numeric(joined[f"{spec.value_column}_right"], errors="coerce")
                    joined["left_threshold_specific_eligible"] = joined["threshold_specific_eligible_left"]
                    joined["right_threshold_specific_eligible"] = joined["threshold_specific_eligible_right"]
                    joined["pair_eligible"] = joined["left_threshold_specific_eligible"] & joined["right_threshold_specific_eligible"]
                    joined["singularity_start_any_state"] = joined["singularity_start_any_state_left"]
                joined["contrast_value"] = joined["left_value"] - joined["right_value"]
            joined["contrast"] = spec.name
            joined["left"] = spec.left
            joined["right"] = spec.right or "__zero__"
            joined["projector_rcond"] = np.nan if spec.projector_rcond is None else float(spec.projector_rcond)
            joined["cutoff_applicability"] = spec.cutoff_applicability
            joined["metric"] = spec.value_column
            joined["regular_eligible"] = joined["pair_eligible"] & ~joined["singularity_start_any_state"]
            if spec.contrast_kind == "history_did":
                joined["singular_sensitivity_eligible"] = (
                    joined["technical_eligible_left_latent"] & joined["technical_eligible_right_latent"] &
                    joined["technical_eligible_left_raw"] & joined["technical_eligible_right_raw"] & joined["singularity_start_any_state"]
                )
            elif spec.right not in (None, "__zero__"):
                joined["singular_sensitivity_eligible"] = joined["technical_eligible_left"] & joined["technical_eligible_right"] & joined["singularity_start_any_state"]
            else:
                joined["singular_sensitivity_eligible"] = joined["technical_eligible"] & joined["singularity_start_any_state"]
            output = [key for key in PAIR_KEYS if key in joined.columns] + [
                "contrast", "left", "right", "projector_rcond", "cutoff_applicability", "metric", "left_value", "right_value", "contrast_value",
                "left_threshold_specific_eligible", "right_threshold_specific_eligible", "pair_eligible", "regular_eligible", "singularity_start_any_state", "singular_sensitivity_eligible",
            ]
            frames.append(joined[output].copy())
    cells = pd.concat(frames, ignore_index=True, sort=False)
    if not np.isfinite(cells[["left_value", "right_value", "contrast_value"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("paired contrast cells contain non-finite values")
    return cells


def _aggregate_stratum(cells: pd.DataFrame, *, eligibility: str, analysis_stratum: str) -> pd.DataFrame:
    eligible = cells[cells[eligibility].astype(bool)].copy()
    if eligible.empty:
        return pd.DataFrame()
    common = ["vae_seed_label", "label", "source_weight_index", "path_step", "trust_mode", "radius_fraction", "projector_rcond", "cutoff_applicability", "contrast", "left", "right", "metric"]
    common = [column for column in common if column in eligible.columns]
    draw_group = common + (["draw_id"] if "draw_id" in eligible.columns else [])
    per_draw = eligible.groupby(draw_group, as_index=False, sort=True, dropna=False).agg(contrast_value=("contrast_value", "mean"), n_cells=("contrast_value", "size"))
    per_start = per_draw.groupby(common, as_index=False, sort=True, dropna=False).agg(
        contrast_value=("contrast_value", "mean"), n_draws=("contrast_value", "size"), n_cells=("n_cells", "sum"),
        negative_draw_count=("contrast_value", lambda v: int((v < 0).sum())), positive_draw_count=("contrast_value", lambda v: int((v > 0).sum())),
    )
    per_start["expected_draws"] = len(EXPECTED_DRAWS)
    per_start["analysis_stratum"] = analysis_stratum
    per_start["inference_unit"] = "draw_mean_within_start"
    return per_start


def aggregate_hierarchy(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regular = _aggregate_stratum(cells, eligibility="regular_eligible", analysis_stratum="regular_primary")
    singular = _aggregate_stratum(cells, eligibility="singular_sensitivity_eligible", analysis_stratum="singular_sensitivity_nonprimary")
    per_start = pd.concat([regular, singular], ignore_index=True, sort=False)
    if per_start.empty:
        raise ValueError("no regular or singular-sensitivity starts were eligible")
    preserve = ["path_step", "trust_mode", "radius_fraction", "projector_rcond", "cutoff_applicability"]
    seed_group = [column for column in ["vae_seed_label", "label", "contrast", "left", "right", "metric", "analysis_stratum"] + preserve if column in per_start.columns]
    per_seed = per_start.groupby(seed_group, as_index=False, sort=True, dropna=False).agg(
        contrast_value=("contrast_value", "mean"), n_starts=("source_weight_index", "nunique"), mean_start_draws=("n_draws", "mean"),
        negative_start_count=("contrast_value", lambda v: int((v < 0).sum())), positive_start_count=("contrast_value", lambda v: int((v > 0).sum())),
    )
    cross_group = [column for column in ["contrast", "left", "right", "metric", "analysis_stratum"] + preserve if column in per_seed.columns]
    cross_seed = per_seed.groupby(cross_group, as_index=False, sort=True, dropna=False).agg(
        contrast_value=("contrast_value", "mean"), n_vae_seeds=("vae_seed_label", "nunique"), min_starts_per_seed=("n_starts", "min"), max_starts_per_seed=("n_starts", "max"),
        negative_seed_count=("contrast_value", lambda v: int((v < 0).sum())), positive_seed_count=("contrast_value", lambda v: int((v > 0).sum())),
    )
    per_seed["inference_unit"] = "equal_start_mean_within_vae_seed"
    cross_seed["inference_unit"] = "equal_weight_vae_seed_mean"
    for frame in (per_start, per_seed, cross_seed):
        frame["contrast_direction"] = "negative_means_left_gives_more_loss_reduction"
        frame["bootstrap_used"] = False
    return per_start, per_seed, cross_seed


def build_coverage(cells: pd.DataFrame) -> pd.DataFrame:
    groups = ["vae_seed_label", "contrast", "left", "right", "metric", "path_step", "trust_mode", "radius_fraction", "projector_rcond", "cutoff_applicability"]
    groups = [column for column in groups if column in cells.columns]
    coverage = cells.groupby(groups, as_index=False, sort=True, dropna=False).agg(
        structural_cells=("contrast_value", "size"), threshold_pair_eligible_cells=("pair_eligible", "sum"), regular_eligible_cells=("regular_eligible", "sum"), singular_cells=("singularity_start_any_state", "sum"), singular_sensitivity_eligible_cells=("singular_sensitivity_eligible", "sum"), structural_starts=("source_weight_index", "nunique"),
    )
    expected = EXPECTED_STARTS * len(EXPECTED_DRAWS)
    coverage["expected_structural_cells"] = expected
    coverage["structural_complete"] = coverage["structural_cells"].eq(expected) & coverage[
        "structural_starts"
    ].eq(EXPECTED_STARTS)
    coverage["regular_coverage_fraction"] = coverage["regular_eligible_cells"] / float(expected)
    coverage["singular_sensitivity_is_nonprimary"] = True
    return coverage


def build_trust_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    """Return native/zero versus matched trust/radius diagnostics only."""
    work = rows.copy()
    if "vae_seed_label" not in work.columns:
        work["vae_seed_label"] = work["label"].astype(str)
    base = ["vae_seed_label", "label", "source_weight_index", "path_step", "draw_id", "candidate"]
    measures = [METRIC, "delta_weight_norm", "delta_function_rms"]
    out: list[pd.DataFrame] = []
    native = work[work["trust_mode"].astype(str).eq("native")].copy()
    for mode in ("weight", "function"):
        matched = work[work["trust_mode"].astype(str).eq(mode)].copy()
        if native.empty or matched.empty:
            continue
        n = native[native["radius_fraction"].astype(float).eq(1.0)][base + measures].rename(
            columns={
                METRIC: "native_value",
                "delta_weight_norm": "native_delta_weight_norm",
                "delta_function_rms": "native_delta_function_rms",
            }
        )
        m = matched[matched["radius_fraction"].astype(float).eq(1.0)][base + measures].rename(
            columns={
                METRIC: "matched_value",
                "delta_weight_norm": "matched_delta_weight_norm",
                "delta_function_rms": "matched_delta_function_rms",
            }
        )
        joined = n.merge(m, on=base, how="inner", validate="one_to_one")
        joined["diagnostic"] = "native_minus_matched"
        joined["matched_trust_mode"] = mode
        joined["contrast_value"] = joined["native_value"] - joined["matched_value"]
        joined["native_to_matched_weight_radius_ratio"] = joined["native_delta_weight_norm"] / joined[
            "matched_delta_weight_norm"
        ].clip(lower=1.0e-30)
        joined["native_to_matched_function_radius_ratio"] = joined["native_delta_function_rms"] / joined[
            "matched_delta_function_rms"
        ].clip(lower=1.0e-30)
        out.append(joined)
        all_matched = matched[base + ["radius_fraction", *measures]].rename(
            columns={
                METRIC: "matched_value",
                "delta_weight_norm": "matched_delta_weight_norm",
                "delta_function_rms": "matched_delta_function_rms",
            }
        )
        all_matched["native_value"] = np.nan
        all_matched["diagnostic"] = "matched_radius_curve"
        all_matched["matched_trust_mode"] = mode
        all_matched["contrast_value"] = np.nan
        out.append(all_matched)
    zero = work[work["trust_mode"].astype(str).eq("zero")][base + ["radius_fraction", METRIC]].copy()
    if not zero.empty:
        zero = zero.rename(columns={METRIC: "zero_value"})
        zero["diagnostic"] = "zero_baseline"
        zero["matched_trust_mode"] = "zero"
        zero["native_value"] = np.nan
        zero["contrast_value"] = zero["zero_value"]
        out.append(zero)
    return pd.concat(out, ignore_index=True, sort=False) if out else pd.DataFrame()


def cutoff_sign_consistency(cross_seed: pd.DataFrame) -> pd.DataFrame:
    independent = cross_seed[cross_seed["cutoff_applicability"].astype(str).eq("cutoff_independent")] if "cutoff_applicability" in cross_seed.columns else pd.DataFrame()
    if not independent.empty:
        # A cutoff-independent estimand is not a three-row pseudo-grid.
        pass
    data = cross_seed[cross_seed["cutoff_applicability"].astype(str).eq("rcond_specific")] if "cutoff_applicability" in cross_seed.columns else cross_seed
    groups = [column for column in ["contrast", "metric", "analysis_stratum", "path_step", "trust_mode", "radius_fraction"] if column in data.columns]
    output: list[dict[str, Any]] = []
    for keys, group in data.groupby(groups, sort=True):
        ordered = group.sort_values("projector_rcond")
        values = ordered["contrast_value"].to_numpy(dtype=np.float64)
        cutoffs = ordered["projector_rcond"].to_numpy(dtype=np.float64)
        signs = np.sign(values).astype(int)
        complete = len(cutoffs) == len(PROJECTOR_RCONDS) and np.allclose(cutoffs, np.asarray(PROJECTOR_RCONDS), rtol=0.0, atol=1.0e-15)
        record = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,)))
        record.update({"cutoff_count": int(len(cutoffs)), "complete_cutoff_grid": bool(complete), "cutoff_sign_consistent": bool(complete and len({int(v) for v in signs if v}) <= 1), "negative_cutoff_count": int((signs < 0).sum()), "zero_cutoff_count": int((signs == 0).sum()), "positive_cutoff_count": int((signs > 0).sum()), "minimum_contrast_value": float(values.min()), "maximum_contrast_value": float(values.max()), "cutoff_values_json": json.dumps([{"projector_rcond": float(c), "contrast_value": float(v), "sign": int(s)} for c, v, s in zip(cutoffs, values, signs)], separators=(",", ":"))})
        output.append(record)
    return pd.DataFrame(output)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    packet_dirs = [Path(value).expanduser().resolve() for value in args.packet_dir]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"startup packet_dirs={packet_dirs} output_dir={output_dir} dtype=float64 aggregation_seed=none bootstrap=false trust_modes={PRIMARY_TRUST_MODES} rconds={PROJECTOR_RCONDS}")
    packets: list[Packet] = []
    for packet_dir in packet_dirs:
        _log(f"stage=validate_packet packet_dir={packet_dir}")
        packet = load_packet(packet_dir)
        packets.append(packet)
        _log(f"accepted_packet label={packet.label} starts={EXPECTED_STARTS} states={len(packet.states)} rows={len(packet.rows)}")
    validate_cross_packet_dimensions(packets)
    combined = pd.concat([packet.rows.assign(vae_seed_label=packet.label) for packet in packets], ignore_index=True, sort=False)
    _log(f"stage=pair_contrasts raw_rows={len(combined)} specs={len(contrast_specs())}")
    cells = build_paired_cells(combined)
    coverage = build_coverage(cells)
    per_start, per_seed, cross_seed = aggregate_hierarchy(cells)
    trust_diagnostics = build_trust_diagnostics(combined)
    signs = cutoff_sign_consistency(cross_seed)
    checks = {
        "three_packets": len(packets) == 3,
        "both_primary_trust_modes": set(cells["trust_mode"].astype(str)) == set(PRIMARY_TRUST_MODES),
        "no_native_or_zero_primary_cells": not cells["trust_mode"].astype(str).isin(DIAGNOSTIC_TRUST_MODES).any(),
        "preserved_strata": all(column in cross_seed.columns for column in ("path_step", "trust_mode", "radius_fraction", "projector_rcond", "cutoff_applicability")),
        "structural_coverage_complete": bool(coverage["structural_complete"].all()),
        "threshold_implies_technical": bool((~_bool_series(combined["threshold_specific_eligible"], column="threshold_specific_eligible") | _bool_series(combined["technical_eligible"], column="technical_eligible")).all()),
        "outputs_finite": bool(np.isfinite(cells["contrast_value"].to_numpy(dtype=np.float64)).all()),
    }
    validation = {"protocol_version": "common_state_rate_cross_vae_v2", "acceptance_checks": checks, "acceptance_pass": bool(all(checks.values())), "packet_labels": [p.label for p in packets], "packet_dirs": [str(p.path) for p in packets], "metric": METRIC, "trust_modes": list(PRIMARY_TRUST_MODES), "aggregation_order": ["mean_draws", "equal_start_mean_within_vae_seed", "equal_weight_vae_seed_mean"], "cutoff_applicability": {"rcond_specific": list(PROJECTOR_RCONDS), "cutoff_independent": "one row per preserved stratum"}, "input_hashes": {str(p.path): p.input_hashes for p in packets}, "reviewed_merge_validator_sha256": REVIEWED_MERGE_VALIDATOR_SHA256}
    if not validation["acceptance_pass"]:
        raise RuntimeError(f"cross-VAE validation rejected: {json.dumps(validation, sort_keys=True)}")
    outputs = {"per_cell": output_dir / "common_state_cross_vae_per_cell.csv", "trust_diagnostics": output_dir / "common_state_cross_vae_trust_diagnostics.csv", "coverage": output_dir / "common_state_cross_vae_coverage.csv", "per_start": output_dir / "common_state_cross_vae_per_start.csv", "per_seed": output_dir / "common_state_cross_vae_per_seed.csv", "cross_seed": output_dir / "common_state_cross_vae_cross_seed.csv", "cutoff_sign_consistency": output_dir / "common_state_cross_vae_cutoff_sign_consistency.csv"}
    for key, path in outputs.items():
        {"per_cell": cells, "trust_diagnostics": trust_diagnostics, "coverage": coverage, "per_start": per_start, "per_seed": per_seed, "cross_seed": cross_seed, "cutoff_sign_consistency": signs}[key].to_csv(path, index=False)
    output_hashes = {name: _sha256_file(path) for name, path in outputs.items()}
    validation["output_hashes"] = output_hashes
    validation_path = output_dir / "common_state_cross_vae_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {"protocol_version": "common_state_rate_cross_vae_v2", "packet_labels": [p.label for p in packets], "output_dir": str(output_dir), "output_hashes": output_hashes, "validation_sha256": _sha256_file(validation_path), "aggregator_script_sha256": _sha256_file(Path(__file__).resolve()), "reviewed_merge_validator_sha256": REVIEWED_MERGE_VALIDATOR_SHA256, "contrasts": [spec.__dict__ for spec in contrast_specs()]}
    manifest["request_hash"] = _stable_json_hash(manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"done acceptance_pass=True cells={len(cells)} per_start={len(per_start)} per_seed={len(per_seed)} cross_seed={len(cross_seed)} outputs={list(outputs.values())} elapsed_sec={time.perf_counter()-started:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strata-preserving cross-VAE aggregation for hash-bound common-state V3 packets.")
    parser.add_argument("--packet-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
