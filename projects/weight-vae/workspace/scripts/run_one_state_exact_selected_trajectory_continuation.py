from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    encode_weights,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _task_set_for_record,
)
from scripts import run_one_state_exact_selected_trajectory as i6
from scripts import review_one_state_exact_selected_trajectory as i6_review
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _load_run,
    _probe_cfg,
    sha256_file,
    sha256_tensor,
)
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    STATE_BANK,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_selected_relaxed_cancellation_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
PROTOCOL_PATH = OUTPUT_ROOT / "postgoal_relaxed_cancellation/protocol.md"
FROZEN_DEPENDENCY_MANIFEST = (
    OUTPUT_ROOT / "postgoal_relaxed_cancellation/frozen_dependency_manifest.json"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "postgoal_relaxed_cancellation_production"
PARENT_OUTPUT = OUTPUT_ROOT / "iteration6_exact_selected_trajectory_production"
PARENT_FINAL_CHECKPOINT = PARENT_OUTPUT / "final_checkpoint.pt"
PARENT_PROGRESS_CHECKPOINT = PARENT_OUTPUT / "progress_checkpoint.pt"
PARENT_ARTIFACT_MANIFEST = PARENT_OUTPUT / "artifact_manifest.json"
PARENT_INDEPENDENT_REVIEW = PARENT_OUTPUT / "independent_review.json"

EXPECTED_NORMALIZED_SOURCE_SHA256 = "a6232ce8f6a0b8385d5977fb82fa4410602b1b2ef383c80835db43a16b42ec04"
EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = (
    "9437b54d9a6379adaab9a7e2bd578caf55e3c879bf861f8a0150f124b577eb32"
)
EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256 = (
    "7063f393332f2640d599fa41928006f74342c3d681dd9b77a575e5e9e40716e5"
)
EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256 = (
    "010d34d51c21ec139fca3e93f51b91c7563f43a524a59b521dcec50de9c752ce"
)
EXPECTED_PARENT_ACTIVE_HASH = (
    "5b6fc545a74f397c4df4a023e2cccc83609633f9e1a738e717832c833ac05b73"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_PARENT_REVIEWER_SHA256 = (
    "19bddeef86071a434f89b2bd412010444599c89a05c0698454ccccd94f9b5e4f"
)

SOURCE_WEIGHT_INDEX = 378
STATE_POSITION = 2
START_UPDATE = 17
MAX_TOTAL_ACCEPTED_UPDATES = 100
MAX_NEW_ACCEPTED_UPDATES = MAX_TOTAL_ACCEPTED_UPDATES - START_UPDATE
SUSTAINED_NEW_UPDATES = 10
PARENT_REPLAY_MAX_ERROR = 1e-9
PROPOSAL18_REPLAY_RTOL = 5e-7
PROPOSAL18_REPLAY_ATOL = 5e-9

Vector = list[torch.Tensor]


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _source_sha256(path: Path | None = None) -> str:
    return hashlib.sha256((path or Path(__file__)).read_bytes()).hexdigest()


def _normalized_source_sha256(path: Path | None = None) -> str:
    source = path or Path(__file__)
    masked = ("EXPECTED_NORMALIZED_SOURCE_SHA256 = ",)
    normalized: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        prefix = next(
            (candidate for candidate in masked if line.startswith(candidate)), None
        )
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _validate_dependencies() -> dict[str, bool]:
    if "TO_BE_FROZEN" in (
        EXPECTED_NORMALIZED_SOURCE_SHA256,
        EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
    ):
        raise RuntimeError("continuation runner has not been frozen")
    if _normalized_source_sha256() != EXPECTED_NORMALIZED_SOURCE_SHA256:
        raise RuntimeError("normalized continuation source hash mismatch")
    if (
        sha256_file(FROZEN_DEPENDENCY_MANIFEST)
        != EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256
    ):
        raise RuntimeError("continuation dependency manifest hash mismatch")
    expected = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    matches = {
        relative: (ROOT / relative).is_file() and sha256_file(ROOT / relative) == digest
        for relative, digest in expected.items()
    }
    failed = [
        relative
        for relative, matches_expected in matches.items()
        if not matches_expected
    ]
    if failed:
        raise RuntimeError(f"continuation dependency mismatch: {failed}")
    if not all(i6._validate_dependencies().values()):
        raise RuntimeError("Iteration-6 transitive dependency validation failed")
    return matches


def _validate_parent_packet() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(PARENT_FINAL_CHECKPOINT) != EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256:
        raise RuntimeError("parent final checkpoint hash mismatch")
    manifest = json.loads(PARENT_ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("artifacts", {}).get("final_checkpoint.pt")
        != EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256
        or manifest.get("artifacts", {}).get("progress_checkpoint.pt")
        != EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256
    ):
        raise RuntimeError("parent artifact manifest checkpoint lineage mismatch")
    if (
        sha256_file(PARENT_PROGRESS_CHECKPOINT)
        != EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256
    ):
        raise RuntimeError("parent progress checkpoint hash mismatch")
    checkpoint = torch.load(
        PARENT_FINAL_CHECKPOINT, map_location="cpu", weights_only=False
    )
    required = {
        "protocol_id": i6.PROTOCOL_ID,
        "selected_arm": "low",
        "accepted_updates": START_UPDATE,
        "termination": "cancellation_gate",
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "z_sha256": EXPECTED_Z_SHA256,
        "active_parameter_hash": EXPECTED_PARENT_ACTIVE_HASH,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"parent checkpoint field mismatch: {key}")
    progress = torch.load(
        PARENT_PROGRESS_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if (
        progress.get("protocol_id") != i6.PROTOCOL_ID
        or int(progress.get("accepted_updates", -1)) != START_UPDATE
        or progress.get("active_parameter_hash") != EXPECTED_PARENT_ACTIVE_HASH
        or not bool(progress.get("terminal"))
        or progress.get("termination") != "cancellation_gate"
        or len(progress.get("state_rows", [])) != START_UPDATE + 1
        or len(progress.get("spectrum_rows", [])) != (START_UPDATE + 1) * 512
        or set(progress.get("active_model_state", {}))
        != set(checkpoint["active_model_state"])
        or _active_state_hash(progress["active_model_state"])
        != EXPECTED_PARENT_ACTIVE_HASH
    ):
        raise RuntimeError("parent progress checkpoint lineage mismatch")
    terminal = progress.get("proposal_rows", [])[-1]
    if (
        int(terminal.get("target_update", -1)) != START_UPDATE + 1
        or int(terminal.get("accepted", -1)) != 0
        or terminal.get("failure") != "cancellation_gate"
    ):
        raise RuntimeError("parent terminal proposal mismatch")
    review = json.loads(PARENT_INDEPENDENT_REVIEW.read_text(encoding="utf-8"))
    reviewer_source = ROOT / "scripts/review_one_state_exact_selected_trajectory.py"
    if (
        review.get("valid") is not True
        or review.get("scientific_success") is not False
        or review.get("failed_gates") != []
        or review.get("errors") != []
        or review.get("reviewer_source_sha256") != EXPECTED_PARENT_REVIEWER_SHA256
        or sha256_file(reviewer_source) != EXPECTED_PARENT_REVIEWER_SHA256
    ):
        raise RuntimeError("parent independent-review gate failed")
    return checkpoint, progress, review


def _relaxed_unit_direction(
    gradient_a: Sequence[torch.Tensor],
    gradient_low: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
) -> tuple[Vector | None, dict[str, float | bool | None]]:
    norm_a = i6._norm(gradient_a)
    norm_low = i6._norm(gradient_low)
    cosine = i6._cosine(gradient_a, gradient_low)
    common_fp32 = i6.exact_helpers._unit_common_gradient(
        gradient_a, gradient_low, active
    )
    source_fp32 = i6._norm(common_fp32)
    common_fp64 = [
        a.detach().double() / norm_a + b.detach().double() / norm_low
        for a, b in zip(gradient_a, gradient_low, strict=True)
    ]
    source_fp64 = i6._norm(common_fp64)
    theoretical_source = math.sqrt(max(0.0, 2.0 + 2.0 * cosine))
    executed_eligible = bool(
        math.isfinite(source_fp32)
        and norm_a > 0.0
        and norm_low > 0.0
        and source_fp32 > 0.0
    )
    shadow_valid = bool(math.isfinite(source_fp64) and source_fp64 > 0.0)
    direction: Vector | None = None
    shadow: Vector | None = None
    shadow_cosine = 0.0
    shadow_relative_error = 0.0
    if executed_eligible:
        direction = i6._negative_normalized(
            common_fp32, target_norm=i6.TARGET_NORM, active=active
        )
    if direction is not None and shadow_valid:
        shadow = [-i6.TARGET_NORM * value / source_fp64 for value in common_fp64]
        shadow_cosine = i6._cosine(direction, shadow)
        difference = [
            observed.detach().double() - expected.detach().double()
            for observed, expected in zip(direction, shadow, strict=True)
        ]
        shadow_relative_error = i6._norm(difference) / i6._norm(shadow)
    finite_positive_source = bool(math.isfinite(source_fp32) and source_fp32 > 0.0)
    return direction, {
        "gradient_a_norm": norm_a,
        "gradient_x_norm": norm_low,
        "gradient_cosine": cosine,
        "unit_common_source_norm": source_fp32,
        "fp64_common_source_norm": source_fp64,
        "theoretical_common_source_norm": theoretical_source,
        "fp32_source_vs_theory_abs_error": abs(source_fp32 - theoretical_source),
        "fp64_source_vs_theory_abs_error": abs(source_fp64 - theoretical_source),
        "legacy_cancellation_gate_pass": source_fp32 >= i6.CANCELLATION_NORM_MIN,
        "relaxed_nonzero_gate_pass": executed_eligible,
        "fp64_shadow_valid": shadow_valid,
        "common_amplification": (1.0 / source_fp32 if finite_positive_source else None),
        "common_amplification_unbounded": source_fp32 == 0.0,
        "target_over_common_source": (
            i6.TARGET_NORM / source_fp32 if finite_positive_source else None
        ),
        "fp32_fp64_direction_cosine": shadow_cosine,
        "fp32_fp64_direction_relative_error": shadow_relative_error,
        "direction_norm": 0.0 if direction is None else i6._norm(direction),
    }


def _proposal18_replay_errors(
    observed: Mapping[str, float | bool], stored: Mapping[str, Any]
) -> dict[str, float]:
    keys = (
        "gradient_a_norm",
        "gradient_x_norm",
        "gradient_cosine",
        "unit_common_source_norm",
    )
    return {key: abs(float(observed[key]) - float(stored[key])) for key in keys}


def _proposal18_replay_passes(
    observed: Mapping[str, float | bool], stored: Mapping[str, Any]
) -> bool:
    return all(
        math.isclose(
            float(observed[key]),
            float(stored[key]),
            rel_tol=PROPOSAL18_REPLAY_RTOL,
            abs_tol=PROPOSAL18_REPLAY_ATOL,
        )
        for key in (
            "gradient_a_norm",
            "gradient_x_norm",
            "gradient_cosine",
            "unit_common_source_norm",
        )
    )


def _metric_replay_errors(
    observed: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, float]:
    excluded = {"hessian_sec", "low_basis_hash"}
    keys = sorted((set(observed) & set(stored)) - excluded)
    return {
        key: abs(float(observed[key]) - float(stored[key]))
        for key in keys
        if isinstance(observed[key], (int, float, np.integer, np.floating))
        and isinstance(stored[key], (int, float, np.integer, np.floating))
    }


def _chain_hash(
    previous: str,
    *,
    update: int,
    base_hash: str,
    endpoint_hash: str,
    alpha: float,
) -> str:
    payload = json.dumps(
        {
            "previous": previous,
            "update": update,
            "base_parameter_hash": base_hash,
            "endpoint_parameter_hash": endpoint_hash,
            "alpha": float(alpha),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _termination_after_loop(accepted_updates: int, termination: str) -> str:
    if accepted_updates == MAX_TOTAL_ACCEPTED_UPDATES:
        if termination not in {"running", "max_updates_reached"}:
            raise RuntimeError(
                f"maximum update count conflicts with terminal failure: {termination}"
            )
        return "max_updates_reached"
    return termination


def _line_metric_payload(line: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "phase",
        "target_update",
        "local_proposal",
        "selected_arm",
        "base_parameter_hash",
        "endpoint_parameter_hash",
        "alpha",
        "realized_path_length",
        "passes",
    }
    return {
        key: value
        for key, value in line.items()
        if key not in metadata
        and not key.startswith("gate_")
        and not key.startswith("repeat_")
    }


def _repeat_payload_values_valid(errors: Mapping[str, float]) -> bool:
    try:
        values = {key: float(value) for key, value in errors.items()}
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        values
        and all(math.isfinite(value) and value >= 0.0 for value in values.values())
        and values.get("repeat_parameter_hash_unchanged") == 1.0
    )


def _strict_repeat_passes(
    errors: Mapping[str, float], tolerances: Mapping[str, float]
) -> bool:
    return bool(
        _repeat_payload_values_valid(errors) and i6._repeat_passes(errors, tolerances)
    )


def _values_match(observed: Any, expected: Any, *, atol: float = 1e-9) -> bool:
    numeric = (int, float, np.integer, np.floating)
    if isinstance(observed, numeric) and isinstance(expected, numeric):
        return bool(
            math.isfinite(float(observed))
            and math.isfinite(float(expected))
            and math.isclose(
                float(observed), float(expected), rel_tol=2e-9, abs_tol=atol
            )
        )
    return observed == expected


def _seed_parent_rows(progress: Mapping[str, Any]) -> dict[str, Any]:
    states = [{**dict(row), "phase": "i6"} for row in progress["state_rows"]]
    spectra = [{**dict(row), "phase": "i6"} for row in progress["spectrum_rows"]]
    proposals = [
        {**dict(row), "phase": "i6"}
        for row in progress["proposal_rows"]
        if int(row.get("accepted", 0)) == 1
    ]
    lines = [{**dict(row), "phase": "i6"} for row in progress["line_rows"]]
    selections = [{**dict(row), "phase": "i6"} for row in progress["selection_rows"]]
    if [int(row["accepted_update"]) for row in states] != list(
        range(START_UPDATE + 1)
    ) or [int(row["target_update"]) for row in proposals] != list(
        range(1, START_UPDATE + 1)
    ):
        raise RuntimeError("parent row sequence mismatch")
    return {
        "state_rows": states,
        "spectrum_rows": spectra,
        "proposal_rows": proposals,
        "line_rows": lines,
        "selection_rows": selections,
        "intervention_origin": dict(progress["proposal_rows"][-1]),
    }


def _active_state_hash(active_state: Mapping[str, torch.Tensor]) -> str:
    names = sorted(active_state)
    return i6._named_tensor_hash(names, [active_state[name] for name in names])


def _audit_rows(
    *,
    rows: Mapping[str, Any],
    accepted_updates: int,
    terminal: bool,
    termination: str,
    tolerances: Mapping[str, float],
    parent_progress: Mapping[str, Any],
    stored_chain_hash: str,
) -> tuple[dict[str, bool], str]:
    seeded = _seed_parent_rows(parent_progress)
    states = list(rows["state_rows"])
    spectra = list(rows["spectrum_rows"])
    proposals = list(rows["proposal_rows"])
    lines = list(rows["line_rows"])
    selections = list(rows["selection_rows"])

    state_updates = [int(row["accepted_update"]) for row in states]
    accepted_proposals = [row for row in proposals if int(row.get("accepted", 0)) == 1]
    rejected_proposals = [row for row in proposals if int(row.get("accepted", 0)) == 0]
    accepted_targets = [int(row["target_update"]) for row in accepted_proposals]
    proposal_targets = [int(row["target_update"]) for row in proposals]
    spectra_by_update: dict[int, list[Mapping[str, Any]]] = {}
    for row in spectra:
        spectra_by_update.setdefault(int(row["accepted_update"]), []).append(row)

    parent_prefix_gates = {
        "parent_state_prefix_exact": states[: START_UPDATE + 1] == seeded["state_rows"],
        "parent_spectrum_prefix_exact": spectra[: (START_UPDATE + 1) * 512]
        == seeded["spectrum_rows"],
        "parent_proposal_prefix_exact": proposals[:START_UPDATE]
        == seeded["proposal_rows"],
        "parent_line_prefix_exact": lines[: len(seeded["line_rows"])]
        == seeded["line_rows"],
        "selection_rows_parent_exact": selections == seeded["selection_rows"],
        "intervention_origin_parent_exact": dict(rows["intervention_origin"])
        == seeded["intervention_origin"],
    }
    structure_gates = {
        "state_updates_exact": state_updates == list(range(accepted_updates + 1)),
        "state_count_exact": len(states) == accepted_updates + 1,
        "accepted_proposals_exact": accepted_targets
        == list(range(1, accepted_updates + 1)),
        "no_parent_rejected_proposal_in_main_table": all(
            int(row["target_update"]) > START_UPDATE for row in rejected_proposals
        ),
        "state_phase_boundary_exact": all(
            row.get("phase")
            == (
                "i6"
                if int(row["accepted_update"]) <= START_UPDATE
                else "relaxed_continuation"
            )
            for row in states
        ),
        "spectrum_phase_boundary_exact": all(
            row.get("phase")
            == (
                "i6"
                if int(row["accepted_update"]) <= START_UPDATE
                else "relaxed_continuation"
            )
            for row in spectra
        ),
        "proposal_phase_boundary_exact": all(
            row.get("phase")
            == (
                "i6"
                if int(row["target_update"]) <= START_UPDATE
                else "relaxed_continuation"
            )
            for row in proposals
        ),
        "line_phase_boundary_exact": all(
            row.get("phase")
            == (
                "i6"
                if int(row["target_update"]) <= START_UPDATE
                else "relaxed_continuation"
            )
            for row in lines
        ),
        "spectrum_update_keys_exact": sorted(spectra_by_update)
        == list(range(accepted_updates + 1)),
        "spectrum_ranks_exact": all(
            len(update_rows) == 512
            and sorted(int(row["rank"]) for row in update_rows) == list(range(512))
            for update_rows in spectra_by_update.values()
        ),
        "spectrum_physical_order_exact": [
            (int(row["accepted_update"]), int(row["rank"])) for row in spectra
        ]
        == [
            (update, rank)
            for update in range(accepted_updates + 1)
            for rank in range(512)
        ],
    }
    if terminal:
        if accepted_updates == MAX_TOTAL_ACCEPTED_UPDATES:
            terminal_structure = bool(
                termination == "max_updates_reached" and not rejected_proposals
            )
        else:
            terminal_structure = bool(
                len(rejected_proposals) == 1
                and int(rejected_proposals[0]["target_update"]) == accepted_updates + 1
                and str(rejected_proposals[0]["failure"]) == termination
                and termination
                in {
                    "exact_cancellation",
                    "nonnegative_joint_slope",
                    "backtracking_exhausted",
                }
            )
    else:
        terminal_structure = bool(termination == "running" and not rejected_proposals)
    structure_gates["terminal_structure_exact"] = terminal_structure
    expected_proposal_targets = list(range(1, accepted_updates + 1)) + (
        [accepted_updates + 1]
        if terminal and accepted_updates < MAX_TOTAL_ACCEPTED_UPDATES
        else []
    )
    structure_gates["proposal_row_order_exact"] = (
        proposal_targets == expected_proposal_targets
    )

    state_by_update = {int(row["accepted_update"]): row for row in states}
    proposal_by_update = {
        int(row["target_update"]): row
        for row in proposals
        if int(row["target_update"]) > START_UPDATE
    }
    line_by_update: dict[int, list[Mapping[str, Any]]] = {}
    continuation_lines = lines[len(seeded["line_rows"]) :]
    for row in continuation_lines:
        line_by_update.setdefault(int(row["target_update"]), []).append(row)

    continuation_prefix_pass = True
    continuation_gate_pass = True
    continuation_repeat_pass = True
    continuation_pass_bit_pass = True
    continuation_hash_pass = True
    continuation_parent_low_pass = True
    continuation_current_link_pass = True
    continuation_endpoint_metric_pass = True
    continuation_diagnostic_pass = True
    continuation_row_identity_pass = bool(
        all(
            row.get("selected_arm") == "low"
            and int(row.get("local_proposal", -1))
            == int(row["target_update"]) - START_UPDATE
            for row in proposals[START_UPDATE:]
        )
        and all(
            row.get("selected_arm") == "low"
            and int(row.get("local_proposal", -1))
            == int(row["target_update"]) - START_UPDATE
            for row in continuation_lines
        )
        and [int(row["target_update"]) for row in continuation_lines]
        == sorted(int(row["target_update"]) for row in continuation_lines)
    )
    expected_gate_keys = {
        f"gate_{name}"
        for name in (
            "A_slope_negative",
            "A_armijo",
            "A_actual_decrease",
            "B_slope_negative",
            "B_armijo",
            "B_actual_decrease",
            "L_low_slope_negative",
            "L_low_armijo",
            "L_low_actual_decrease",
            "high_tail_nonincrease",
            "low90_decreases",
            "exact_a_closes",
            "spectrum_finite_nonnegative",
            "metrics_finite",
        )
    }
    expected_repeat_keys = {
        key for key in seeded["line_rows"][-1] if key.startswith("repeat_")
    }
    expected_line_metric_keys = set(_line_metric_payload(seeded["line_rows"][-1]))
    expected_state_metric_keys = expected_line_metric_keys | {
        "parent_frozen_low_energy"
    }
    continuation_metric_schema_pass = bool(
        all(
            set(row) - {"accepted_update", "parameter_hash", "phase"}
            == expected_state_metric_keys
            for row in states[START_UPDATE + 1 :]
        )
    )
    continuation_control_flow_pass = True
    continuation_gradient_metadata_pass = True
    for target, proposal in sorted(proposal_by_update.items()):
        target_lines = line_by_update.get(target, [])
        observed_alphas = [float(row["alpha"]) for row in target_lines]
        accepted = int(proposal.get("accepted", 0)) == 1
        continuation_metric_schema_pass = continuation_metric_schema_pass and all(
            set(_line_metric_payload(line)) == expected_line_metric_keys
            for line in target_lines
        )
        try:
            gradient_metadata = json.loads(str(proposal["gradient_metadata"]))
            if not isinstance(gradient_metadata, Mapping):
                raise TypeError("gradient metadata is not a mapping")
            i6_review._validate_gradient_metadata(gradient_metadata)
        except (
            KeyError,
            json.JSONDecodeError,
            RuntimeError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            continuation_gradient_metadata_pass = False
        try:
            norm_a = float(proposal["gradient_a_norm"])
            norm_low = float(proposal["gradient_x_norm"])
            cosine = float(proposal["gradient_cosine"])
            source = float(proposal["unit_common_source_norm"])
            fp64_source = float(proposal["fp64_common_source_norm"])
            theoretical_source = math.sqrt(max(0.0, 2.0 + 2.0 * cosine))
            relaxed = bool(
                math.isfinite(source)
                and norm_a > 0.0
                and norm_low > 0.0
                and source > 0.0
            )
            shadow_valid = bool(math.isfinite(fp64_source) and fp64_source > 0.0)
            selected_alpha_value = float(proposal["selected_alpha"])
            diagnostic_pass = bool(
                all(
                    math.isfinite(value) for value in (norm_a, norm_low, cosine, source)
                )
                and norm_a > 0.0
                and norm_low > 0.0
                and source >= 0.0
                and -1.0 - 1e-9 <= cosine <= 1.0 + 1e-9
                and bool(proposal["legacy_cancellation_gate_pass"])
                == (source >= i6.CANCELLATION_NORM_MIN)
                and bool(proposal["relaxed_nonzero_gate_pass"]) == relaxed
                and bool(proposal["fp64_shadow_valid"]) == shadow_valid
                and _values_match(
                    proposal["theoretical_common_source_norm"], theoretical_source
                )
                and _values_match(
                    proposal["fp32_source_vs_theory_abs_error"],
                    abs(source - theoretical_source),
                )
                and _values_match(
                    proposal["fp64_source_vs_theory_abs_error"],
                    abs(fp64_source - theoretical_source),
                )
                and _values_match(
                    proposal["common_amplification"],
                    None if source == 0.0 else 1.0 / source,
                )
                and bool(proposal["common_amplification_unbounded"]) == (source == 0.0)
                and _values_match(
                    proposal["target_over_common_source"],
                    None if source == 0.0 else i6.TARGET_NORM / source,
                )
                and _values_match(
                    proposal["direction_norm"], i6.TARGET_NORM if relaxed else 0.0
                )
                and (
                    (
                        shadow_valid
                        and math.isfinite(float(proposal["fp32_fp64_direction_cosine"]))
                        and -1.0 - 1e-9
                        <= float(proposal["fp32_fp64_direction_cosine"])
                        <= 1.0 + 1e-9
                        and math.isfinite(
                            float(proposal["fp32_fp64_direction_relative_error"])
                        )
                        and float(proposal["fp32_fp64_direction_relative_error"]) >= 0.0
                    )
                    or (
                        not shadow_valid
                        and float(proposal["fp32_fp64_direction_cosine"]) == 0.0
                        and float(proposal["fp32_fp64_direction_relative_error"]) == 0.0
                    )
                )
                and _values_match(
                    proposal["realized_path_length"],
                    selected_alpha_value * i6.TARGET_NORM,
                )
                and ((selected_alpha_value > 0.0) == accepted)
                and all(
                    _values_match(
                        line["realized_path_length"],
                        float(line["alpha"]) * i6.TARGET_NORM,
                    )
                    for line in target_lines
                )
            )
            if relaxed:
                theoretical_slope_a = (
                    -i6.TARGET_NORM * norm_a * theoretical_source / 2.0
                )
                theoretical_slope_low = (
                    -i6.TARGET_NORM * norm_low * theoretical_source / 2.0
                )
                diagnostic_pass = diagnostic_pass and bool(
                    _values_match(proposal["theoretical_slope_A"], theoretical_slope_a)
                    and _values_match(
                        proposal["theoretical_slope_L_low"], theoretical_slope_low
                    )
                    and _values_match(
                        proposal["slope_A_vs_theory_abs_error"],
                        abs(float(proposal["slope_A"]) - theoretical_slope_a),
                    )
                    and _values_match(
                        proposal["slope_L_low_vs_theory_abs_error"],
                        abs(float(proposal["slope_L_low"]) - theoretical_slope_low),
                    )
                )
            continuation_diagnostic_pass = (
                continuation_diagnostic_pass and diagnostic_pass
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continuation_diagnostic_pass = False
        if accepted:
            prefix_pass = bool(
                target_lines
                and observed_alphas == list(i6.LINE_ALPHAS[: len(target_lines)])
                and int(target_lines[-1]["passes"]) == 1
                and all(int(row["passes"]) == 0 for row in target_lines[:-1])
                and math.isclose(
                    float(proposal["selected_alpha"]),
                    float(target_lines[-1]["alpha"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
        elif str(proposal.get("failure")) == "backtracking_exhausted":
            prefix_pass = bool(
                observed_alphas == list(i6.LINE_ALPHAS)
                and all(int(row["passes"]) == 0 for row in target_lines)
            )
        else:
            prefix_pass = not target_lines
        continuation_prefix_pass = continuation_prefix_pass and prefix_pass

        failure = str(proposal.get("failure", ""))
        slope_values = [
            float(proposal.get(f"slope_{name}", math.nan))
            for name in ("A", "B", "L_low")
        ]
        slopes_finite = all(math.isfinite(value) for value in slope_values)
        slopes_negative = all(value < 0.0 for value in slope_values)
        relaxed_direction = bool(proposal.get("relaxed_nonzero_gate_pass"))
        if accepted:
            branch_pass = bool(
                failure == ""
                and slopes_finite
                and relaxed_direction
                and slopes_negative
                and target_lines
            )
        elif failure == "exact_cancellation":
            branch_pass = bool(
                not relaxed_direction
                and slopes_finite
                and not target_lines
                and all(
                    float(proposal.get(f"slope_{name}", math.inf)) == 0.0
                    for name in ("A", "B", "L_low")
                )
            )
        elif failure == "nonnegative_joint_slope":
            branch_pass = bool(
                relaxed_direction
                and slopes_finite
                and not slopes_negative
                and not target_lines
            )
        elif failure == "backtracking_exhausted":
            branch_pass = bool(
                relaxed_direction
                and slopes_finite
                and slopes_negative
                and len(target_lines) == len(i6.LINE_ALPHAS)
                and all(int(row["passes"]) == 0 for row in target_lines)
            )
        else:
            branch_pass = False
        continuation_control_flow_pass = continuation_control_flow_pass and branch_pass

        parent_state = state_by_update.get(target - 1)
        endpoint_state = state_by_update.get(target) if accepted else None
        base_hash = str(proposal.get("base_parameter_hash", ""))
        continuation_hash_pass = continuation_hash_pass and bool(
            parent_state is not None
            and base_hash == str(parent_state["parameter_hash"])
            and all(
                str(row.get("base_parameter_hash", "")) == base_hash
                for row in target_lines
            )
        )
        continuation_current_link_pass = continuation_current_link_pass and bool(
            parent_state is not None
            and all(
                key in proposal
                for key in (
                    "current_A",
                    "current_B",
                    "current_low_energy",
                    "current_a_gt1",
                )
            )
            and all(
                key in parent_state
                for key in (
                    "exact_a_per_dim",
                    "damped_full_burg_per_dim",
                    "frozen_low_energy",
                    "a_gt1",
                )
            )
            and _values_match(
                proposal.get("current_A"), parent_state.get("exact_a_per_dim")
            )
            and _values_match(
                proposal.get("current_B"),
                parent_state.get("damped_full_burg_per_dim"),
            )
            and _values_match(
                proposal.get("current_low_energy"),
                parent_state.get("frozen_low_energy"),
            )
            and _values_match(proposal.get("current_a_gt1"), parent_state.get("a_gt1"))
        )
        for line in target_lines:
            repeat_errors = {
                key: value for key, value in line.items() if key.startswith("repeat_")
            }
            observed_gate_keys = {key for key in line if key.startswith("gate_")}
            schema_pass = bool(
                observed_gate_keys == expected_gate_keys
                and set(repeat_errors) == expected_repeat_keys
            )
            repeat_values_valid = bool(
                schema_pass
                and all(
                    math.isfinite(float(value)) and float(value) >= 0.0
                    for value in repeat_errors.values()
                )
                and float(repeat_errors["repeat_parameter_hash_unchanged"]) == 1.0
            )
            recomputed_repeat_pass = bool(
                repeat_values_valid and _strict_repeat_passes(repeat_errors, tolerances)
            )
            try:
                recomputed_gates = i6._candidate_gates(
                    current=parent_state,
                    candidate=_line_metric_payload(line),
                    slopes={
                        name: float(proposal[f"slope_{name}"])
                        for name in ("A", "B", "L_low")
                    },
                    alpha=float(line["alpha"]),
                    tolerances=tolerances,
                    require_low90=False,
                )
                stored_gates = {
                    key.removeprefix("gate_"): bool(int(line[key]))
                    for key in expected_gate_keys
                }
                gate_map_matches = bool(
                    schema_pass and stored_gates == recomputed_gates
                )
                recomputed_gate_pass = bool(all(recomputed_gates.values()))
            except (KeyError, TypeError, ValueError, OverflowError):
                gate_map_matches = False
                recomputed_gate_pass = False
            continuation_gate_pass = continuation_gate_pass and gate_map_matches
            continuation_repeat_pass = continuation_repeat_pass and repeat_values_valid
            continuation_pass_bit_pass = continuation_pass_bit_pass and bool(
                int(line["passes"])
                == int(recomputed_repeat_pass and recomputed_gate_pass)
            )
        if accepted:
            if not target_lines or endpoint_state is None:
                continuation_hash_pass = False
                continuation_parent_low_pass = False
                continue
            selected = target_lines[-1]
            endpoint_hash = str(selected["endpoint_parameter_hash"])
            continuation_hash_pass = continuation_hash_pass and bool(
                endpoint_state is not None
                and endpoint_hash == str(proposal.get("endpoint_parameter_hash", ""))
                and endpoint_hash == str(endpoint_state["parameter_hash"])
            )
            continuation_parent_low_pass = continuation_parent_low_pass and bool(
                endpoint_state is not None
                and math.isclose(
                    float(selected["frozen_low_energy"]),
                    float(proposal["candidate_low_energy"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(selected["frozen_low_energy"]),
                    float(endpoint_state["parent_frozen_low_energy"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(selected["canonical_low_energy"]),
                    float(endpoint_state["frozen_low_energy"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(selected["exact_a_per_dim"]),
                    float(proposal["candidate_A"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(selected["damped_full_burg_per_dim"]),
                    float(proposal["candidate_B"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(selected["a_gt1"]),
                    float(proposal["candidate_a_gt1"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            selected_metrics = _line_metric_payload(selected)
            endpoint_metric_links = []
            for key, value in selected_metrics.items():
                state_key = (
                    "parent_frozen_low_energy" if key == "frozen_low_energy" else key
                )
                endpoint_metric_links.append(
                    state_key in endpoint_state
                    and _values_match(value, endpoint_state[state_key])
                )
            endpoint_metric_links.extend(
                (
                    _values_match(
                        selected_metrics.get("canonical_low_energy"),
                        endpoint_state.get("frozen_low_energy"),
                    ),
                    _values_match(
                        selected_metrics.get("canonical_low_energy"),
                        endpoint_state.get("canonical_low_energy"),
                    ),
                )
            )
            continuation_endpoint_metric_pass = (
                continuation_endpoint_metric_pass and all(endpoint_metric_links)
            )

    continuation_spectrum_pass = True
    integral_metrics = {
        "count_lt_1e_4",
        "count_lt_1e_2",
        "count_lt_0p1",
        "low_count",
    }
    for update in range(START_UPDATE + 1, accepted_updates + 1):
        update_rows = spectra_by_update.get(update, [])
        state = state_by_update.get(update)
        try:
            eig = np.asarray(
                [float(row["m_eigenvalue"]) for row in update_rows],
                dtype=np.float64,
            )
            expected_metrics = i6_review._spectral_metrics(eig)
            contribution_pass = bool(
                len(update_rows) == 512
                and all(
                    _values_match(
                        row["a_contribution"],
                        (float(row["m_eigenvalue"]) - 1.0) ** 2,
                    )
                    for row in update_rows
                )
            )
            metric_pass = bool(
                state is not None
                and all(
                    metric in state
                    and (
                        float(state[metric]) == reference
                        if metric in integral_metrics
                        else _values_match(state[metric], reference, atol=2e-9)
                    )
                    for metric, reference in expected_metrics.items()
                )
                and _values_match(
                    max(float(state["m_raw_eig_min"]), 0.0), eig[0], atol=2e-9
                )
                and float(state["m_raw_eig_min"]) >= -1e-8
                and float(state["a_direct_abs_error"]) <= 1e-9
                and float(state["a_trace_abs_error"]) <= 1e-9
            )
            continuation_spectrum_pass = (
                continuation_spectrum_pass and contribution_pass and metric_pass
            )
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            continuation_spectrum_pass = False

    chain_hash = EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256
    chain_pass = True
    for proposal in accepted_proposals[START_UPDATE:]:
        target = int(proposal["target_update"])
        chain_hash = _chain_hash(
            chain_hash,
            update=target,
            base_hash=str(proposal["base_parameter_hash"]),
            endpoint_hash=str(proposal["endpoint_parameter_hash"]),
            alpha=float(proposal["selected_alpha"]),
        )
        chain_pass = (
            chain_pass
            and str(proposal.get("transition_chain_sha256", "")) == chain_hash
        )
    chain_pass = chain_pass and stored_chain_hash == chain_hash

    gates = {
        **parent_prefix_gates,
        **structure_gates,
        "continuation_line_targets_known": set(line_by_update).issubset(
            set(proposal_by_update)
        ),
        "continuation_line_prefix_repeat_pass": bool(
            continuation_prefix_pass
            and continuation_repeat_pass
            and continuation_pass_bit_pass
        ),
        "continuation_gate_map_recomputes": continuation_gate_pass,
        "continuation_repeat_values_valid": continuation_repeat_pass,
        "continuation_pass_bits_recompute": continuation_pass_bit_pass,
        "continuation_row_identity_exact": continuation_row_identity_pass,
        "continuation_diagnostics_recompute": continuation_diagnostic_pass,
        "continuation_metric_schema_exact": continuation_metric_schema_pass,
        "continuation_control_flow_recomputes": continuation_control_flow_pass,
        "continuation_gradient_metadata_valid": continuation_gradient_metadata_pass,
        "continuation_parameter_hash_links_pass": continuation_hash_pass,
        "continuation_parent_low_links_pass": continuation_parent_low_pass,
        "continuation_proposal_current_links_pass": continuation_current_link_pass,
        "continuation_endpoint_metric_links_pass": continuation_endpoint_metric_pass,
        "continuation_spectrum_metrics_recompute": continuation_spectrum_pass,
        "transition_chain_recomputes": chain_pass,
    }
    return gates, chain_hash


def _audit_progress_checkpoint(
    progress: Mapping[str, Any], parent_progress: Mapping[str, Any]
) -> dict[str, bool]:
    accepted_updates = int(progress.get("accepted_updates", -1))
    terminal = bool(progress.get("terminal"))
    termination = str(progress.get("termination", ""))
    rows = {
        key: list(progress.get(key, []))
        for key in (
            "state_rows",
            "spectrum_rows",
            "proposal_rows",
            "line_rows",
            "selection_rows",
        )
    }
    rows["intervention_origin"] = dict(progress.get("intervention_origin", {}))
    row_gates, _ = _audit_rows(
        rows=rows,
        accepted_updates=accepted_updates,
        terminal=terminal,
        termination=termination,
        tolerances=progress.get("tolerances", {}),
        parent_progress=parent_progress,
        stored_chain_hash=str(progress.get("transition_chain_sha256", "")),
    )
    active_state = progress.get("active_model_state", {})
    preflight = progress.get("intervention_preflight", {})
    preflight_required = terminal or accepted_updates > START_UPDATE
    first_continuation = next(
        (
            row
            for row in rows["proposal_rows"]
            if int(row.get("target_update", -1)) == START_UPDATE + 1
        ),
        None,
    )
    if first_continuation is not None:
        recomputed_preflight_errors = _proposal18_replay_errors(
            first_continuation, rows["intervention_origin"]
        )
        recomputed_preflight_pass = _proposal18_replay_passes(
            first_continuation, rows["intervention_origin"]
        )
    else:
        recomputed_preflight_errors = {}
        recomputed_preflight_pass = False
    parent_state = parent_progress["state_rows"][-1]
    expected_parent_metric_error_keys = {
        key
        for key, value in parent_state.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key
        not in {
            "accepted_update",
            "parent_frozen_low_energy",
            "hessian_sec",
        }
    }
    stored_parent_metric_errors = preflight.get("parent_state_metric_errors", {})
    try:
        parent_error_values = {
            key: float(value) for key, value in stored_parent_metric_errors.items()
        }
        parent_error_schema_pass = bool(
            set(parent_error_values) == expected_parent_metric_error_keys
            and parent_error_values
            and all(
                math.isfinite(value) and value >= 0.0
                for value in parent_error_values.values()
            )
        )
        parent_error_max = max(parent_error_values.values())
        parent_error_summary_pass = bool(
            parent_error_schema_pass
            and math.isfinite(float(preflight["parent_state_metric_max_abs_error"]))
            and float(preflight["parent_state_metric_max_abs_error"]) >= 0.0
            and math.isclose(
                float(preflight["parent_state_metric_max_abs_error"]),
                parent_error_max,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and parent_error_max <= PARENT_REPLAY_MAX_ERROR
        )
        parent_spectrum_error = float(preflight["parent_spectrum_max_abs_error"])
        parent_spectrum_error_pass = bool(
            math.isfinite(parent_spectrum_error)
            and 0.0 <= parent_spectrum_error <= PARENT_REPLAY_MAX_ERROR
        )
        stored_proposal_errors = {
            key: float(value)
            for key, value in preflight.get("proposal18_replay_errors", {}).items()
        }
        proposal_error_payload_pass = bool(
            set(stored_proposal_errors) == set(recomputed_preflight_errors)
            and all(
                math.isfinite(value) and value >= 0.0
                for value in stored_proposal_errors.values()
            )
            and all(
                math.isclose(
                    stored_proposal_errors[key],
                    value,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                for key, value in recomputed_preflight_errors.items()
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        parent_error_summary_pass = False
        parent_spectrum_error_pass = False
        proposal_error_payload_pass = False
    recomputed_parent_parameter_match = bool(
        first_continuation is not None
        and first_continuation.get("base_parameter_hash") == EXPECTED_PARENT_ACTIVE_HASH
        and parent_state.get("parameter_hash") == EXPECTED_PARENT_ACTIVE_HASH
    )
    recomputed_old_gate_rejects = bool(
        first_continuation is not None
        and not bool(first_continuation.get("legacy_cancellation_gate_pass"))
    )
    recomputed_relaxed_gate_accepts = bool(
        first_continuation is not None
        and bool(first_continuation.get("relaxed_nonzero_gate_pass"))
    )
    preflight_valid = bool(
        isinstance(preflight, Mapping)
        and (
            (not preflight_required and not preflight and first_continuation is None)
            or (
                preflight_required
                and first_continuation is not None
                and bool(preflight.get("proposal18_replay_pass"))
                == recomputed_preflight_pass
                and recomputed_preflight_pass
                and proposal_error_payload_pass
                and parent_error_summary_pass
                and parent_spectrum_error_pass
                and bool(preflight.get("parent_parameter_hash_matches"))
                == recomputed_parent_parameter_match
                and recomputed_parent_parameter_match
                and bool(preflight.get("parent_low_basis_hash_matches"))
                and bool(preflight.get("old_gate_rejects"))
                == recomputed_old_gate_rejects
                and recomputed_old_gate_rejects
                and bool(preflight.get("relaxed_gate_accepts"))
                == recomputed_relaxed_gate_accepts
                and recomputed_relaxed_gate_accepts
                and bool(preflight.get("parent_independent_review_valid"))
            )
        )
    )
    header_gates = {
        "progress_protocol_matches": progress.get("protocol_id") == PROTOCOL_ID,
        "progress_source_matches": progress.get("normalized_source_sha256")
        == EXPECTED_NORMALIZED_SOURCE_SHA256,
        "progress_dependency_manifest_matches": progress.get(
            "dependency_manifest_sha256"
        )
        == EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        "progress_parent_checkpoint_matches": progress.get("parent_checkpoint_sha256")
        == EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256,
        "progress_parent_progress_matches": progress.get(
            "parent_progress_checkpoint_sha256"
        )
        == EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256,
        "progress_parent_active_hash_matches": progress.get(
            "parent_active_parameter_hash"
        )
        == EXPECTED_PARENT_ACTIVE_HASH,
        "progress_selected_arm_is_low": progress.get("selected_arm") == "low",
        "progress_update_range_valid": START_UPDATE
        <= accepted_updates
        <= MAX_TOTAL_ACCEPTED_UPDATES,
        "progress_new_update_count_matches": int(
            progress.get("new_accepted_updates", -1)
        )
        == accepted_updates - START_UPDATE,
        "progress_tolerances_parent_exact": dict(progress.get("tolerances", {}))
        == dict(parent_progress["tolerances"]),
        "progress_initial_metrics_parent_exact": dict(
            progress.get("initial_metrics", {})
        )
        == dict(parent_progress["initial_metrics"]),
        "progress_active_names_exact": isinstance(active_state, Mapping)
        and set(active_state) == set(parent_progress["active_model_state"]),
        "progress_active_hash_recomputes": isinstance(active_state, Mapping)
        and bool(active_state)
        and _active_state_hash(active_state) == progress.get("active_parameter_hash"),
        "progress_active_hash_matches_last_state": bool(rows["state_rows"])
        and progress.get("active_parameter_hash")
        == rows["state_rows"][-1].get("parameter_hash"),
        "progress_preflight_valid": preflight_valid,
    }
    gates = {**header_gates, **row_gates}
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"continuation progress audit failed: {failed}")
    return gates


def _write_tables(staging: Path, payload: Mapping[str, Any]) -> None:
    i6._atomic_csv(staging / "state_metrics.csv", payload["state_rows"])
    i6._atomic_csv(staging / "state_spectra.csv", payload["spectrum_rows"])
    i6._atomic_csv(staging / "proposal_diagnostics.csv", payload["proposal_rows"])
    i6._atomic_csv(staging / "line_search.csv", payload["line_rows"])
    i6._atomic_csv(staging / "arm_selection.csv", payload["selection_rows"])


def _write_progress(
    *,
    staging: Path,
    active_names: Sequence[str],
    active: Sequence[torch.nn.Parameter],
    accepted_updates: int,
    rows: Mapping[str, Any],
    initial_metrics: Mapping[str, Any],
    tolerances: Mapping[str, float],
    chain_hash: str,
    terminal: bool,
    termination: str,
    intervention_preflight: Mapping[str, Any],
) -> None:
    checkpoint = {
        "protocol_id": PROTOCOL_ID,
        "normalized_source_sha256": EXPECTED_NORMALIZED_SOURCE_SHA256,
        "dependency_manifest_sha256": EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        "parent_checkpoint_sha256": EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256,
        "parent_progress_checkpoint_sha256": (
            EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256
        ),
        "parent_active_parameter_hash": EXPECTED_PARENT_ACTIVE_HASH,
        "accepted_updates": accepted_updates,
        "new_accepted_updates": accepted_updates - START_UPDATE,
        "selected_arm": "low",
        "terminal": bool(terminal),
        "termination": termination,
        "transition_chain_sha256": chain_hash,
        "active_parameter_hash": i6._named_tensor_hash(active_names, active),
        "active_model_state": {
            name: parameter.detach().cpu().clone()
            for name, parameter in zip(active_names, active, strict=True)
        },
        **{
            key: list(rows[key])
            for key in (
                "state_rows",
                "spectrum_rows",
                "proposal_rows",
                "line_rows",
                "selection_rows",
            )
        },
        "intervention_origin": dict(rows["intervention_origin"]),
        "initial_metrics": dict(initial_metrics),
        "tolerances": dict(tolerances),
        "intervention_preflight": dict(intervention_preflight),
    }
    i6._atomic_torch_save(staging / "progress_checkpoint.pt", checkpoint)
    _write_tables(staging, rows)


def _plot(output: Path, states: pd.DataFrame, spectra: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    panels = (
        ("exact_a_per_dim", "Exact dense A"),
        ("damped_full_burg_per_dim", "Damped full Burg B"),
        ("a_low90_abs_per_dim", "Lower-90 A contribution"),
        ("a_gt1", "High-tail A contribution"),
        ("m_p50", "Median M eigenvalue"),
        ("effective_rank", "Effective rank"),
    )
    for axis, (column, title) in zip(axes.flat, panels, strict=True):
        axis.plot(states["accepted_update"], states[column], linewidth=2.0)
        axis.axvline(START_UPDATE, color="black", linestyle="--", linewidth=1.0)
        axis.set(title=title, xlabel="global accepted update", ylabel=column)
        axis.grid(alpha=0.25)
    fig.suptitle("Exact A+low trajectory: I6 plus relaxed-cancellation continuation")
    fig.savefig(output / "relaxed_cancellation_trajectory.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    updates = sorted(set(int(value) for value in spectra["accepted_update"]))
    chosen = sorted(set([0, START_UPDATE, updates[-1]]))
    for update in chosen:
        rows = spectra.loc[spectra["accepted_update"].eq(update)].sort_values("rank")
        axis.plot(
            rows["rank"],
            np.clip(rows["m_eigenvalue"], 1e-14, None),
            label=f"update {update}",
        )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_yscale("log")
    axis.set(title="Ordered M spectra", xlabel="rank", ylabel="eigenvalue")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "relaxed_cancellation_spectra.png", dpi=180)
    plt.close(fig)

    continuation = pd.DataFrame(
        [
            row
            for row in pd.read_csv(output / "proposal_diagnostics.csv").to_dict(
                "records"
            )
            if int(row["target_update"]) >= START_UPDATE + 1
        ]
    )
    if not continuation.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        panels2 = (
            ("unit_common_source_norm", "Common source norm"),
            ("common_amplification", "Normalization amplification"),
            ("fp32_fp64_direction_cosine", "FP32/FP64 direction cosine"),
            ("selected_alpha", "Accepted alpha"),
        )
        for axis, (column, title) in zip(axes.flat, panels2, strict=True):
            axis.plot(continuation["target_update"], continuation[column], marker="o")
            if column == "unit_common_source_norm":
                axis.axhline(i6.CANCELLATION_NORM_MIN, color="black", linestyle="--")
            axis.set(title=title, xlabel="global proposal", ylabel=column)
            axis.grid(alpha=0.25)
        fig.suptitle("Relaxed-cancellation direction diagnostics")
        fig.savefig(output / "relaxed_cancellation_direction_diagnostics.png", dpi=180)
        plt.close(fig)


def _artifact_manifest(output: Path, source_snapshot: Path) -> dict[str, Any]:
    excluded = {
        "artifact_manifest.json",
        "FINALIZED.json",
        "INCOMPLETE",
        "run.log",
    }
    artifacts = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name not in excluded
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "executed_source_sha256": _source_sha256(source_snapshot),
        "executed_normalized_source_sha256": _normalized_source_sha256(source_snapshot),
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
    }


def _validate_published_recovery(output: Path) -> dict[str, bool]:
    required = {
        "FINALIZED.json",
        "artifact_manifest.json",
        "decision.json",
        "executed_source_snapshot.py",
        "final_checkpoint.pt",
        "INCOMPLETE",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise RuntimeError(f"published continuation recovery files missing: {missing}")
    finalized = json.loads((output / "FINALIZED.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    excluded = {
        "artifact_manifest.json",
        "FINALIZED.json",
        "INCOMPLETE",
        "run.log",
    }
    actual_artifact_names = {
        path.name
        for path in output.iterdir()
        if path.is_file() and path.name not in excluded
    }
    manifest_names_exact = bool(
        isinstance(artifacts, Mapping) and set(artifacts) == actual_artifact_names
    )
    artifact_hashes_match = bool(
        manifest_names_exact
        and all(
            isinstance(digest, str) and sha256_file(output / name) == digest
            for name, digest in artifacts.items()
        )
    )
    source_snapshot = output / "executed_source_snapshot.py"
    outcome = decision.get("outcome", {})
    gates = {
        "finalized_protocol_matches": finalized.get("protocol_id") == PROTOCOL_ID,
        "finalized_status_exact": finalized.get("status")
        == "complete_awaiting_independent_review",
        "finalized_valid_true": finalized.get("valid") is True,
        "decision_protocol_matches": decision.get("protocol_id") == PROTOCOL_ID,
        "decision_valid_true": decision.get("valid") is True,
        "decision_selected_arm_low": decision.get("selected_arm") == "low",
        "manifest_protocol_matches": manifest.get("protocol_id") == PROTOCOL_ID,
        "manifest_names_exact": manifest_names_exact,
        "manifest_artifact_hashes_match": artifact_hashes_match,
        "manifest_source_hash_matches": manifest.get("executed_source_sha256")
        == _source_sha256(source_snapshot),
        "manifest_normalized_source_matches": manifest.get(
            "executed_normalized_source_sha256"
        )
        == EXPECTED_NORMALIZED_SOURCE_SHA256
        == _normalized_source_sha256(source_snapshot),
        "finalized_decision_hash_matches": finalized.get("decision_sha256")
        == sha256_file(output / "decision.json"),
        "finalized_manifest_hash_matches": finalized.get("artifact_manifest_sha256")
        == sha256_file(output / "artifact_manifest.json"),
        "accepted_updates_consistent": finalized.get("accepted_updates")
        == decision.get("accepted_updates"),
        "new_updates_consistent": finalized.get("new_accepted_updates")
        == decision.get("new_accepted_updates"),
        "termination_consistent": finalized.get("termination")
        == decision.get("termination")
        == outcome.get("termination"),
        "scientific_success_consistent": finalized.get("scientific_success")
        == decision.get("scientific_success")
        == outcome.get("scientific_success"),
        "immediate_outcome_consistent": finalized.get("immediate_premature_cutoff")
        == outcome.get("immediate_premature_cutoff"),
        "sustained_outcome_consistent": finalized.get("sustained_continuation")
        == outcome.get("sustained_continuation"),
        "final_checkpoint_hash_consistent": decision.get("final_checkpoint_file_sha256")
        == artifacts.get("final_checkpoint.pt"),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"published continuation recovery audit failed: {failed}")
    return gates


def _outcome(
    *,
    states: pd.DataFrame,
    spectra: pd.DataFrame,
    proposals: pd.DataFrame,
    tolerances: Mapping[str, float],
    accepted_updates: int,
    termination: str,
    history_summary: Mapping[str, bool],
) -> dict[str, Any]:
    initial = states.iloc[0]
    intervention_start = states.loc[states["accepted_update"].eq(START_UPDATE)].iloc[0]
    final = states.iloc[-1]
    total_a_reduction = float(initial["exact_a_per_dim"] - final["exact_a_per_dim"])
    total_lower90_reduction = float(
        initial["a_low90_abs_per_dim"] - final["a_low90_abs_per_dim"]
    )
    continuation_a_reduction = float(
        intervention_start["exact_a_per_dim"] - final["exact_a_per_dim"]
    )
    continuation_lower90_reduction = float(
        intervention_start["a_low90_abs_per_dim"] - final["a_low90_abs_per_dim"]
    )
    state_low90_decrease = -states["a_low90_abs_per_dim"].diff()
    tail_mask = states["accepted_update"].between(81, 100)
    tail_bulk_updates = int(
        (state_low90_decrease.loc[tail_mask] > float(tolerances["A_low90"])).sum()
    )
    state_a_diffs = states["exact_a_per_dim"].diff().iloc[1:]
    state_high_diffs = states["a_gt1"].diff().iloc[1:]
    success_gates = {
        "one_hundred_accepted_updates": accepted_updates == MAX_TOTAL_ACCEPTED_UPDATES,
        "tail_bulk_activity": tail_bulk_updates >= i6.SUCCESS_TAIL_BULK_UPDATES_MIN,
        "final_a_at_most_0p90": float(final["exact_a_per_dim"]) <= i6.SUCCESS_A_MAX,
        "accepted_a_strictly_decreases": bool(
            (state_a_diffs < -float(tolerances["A"])).all()
        ),
        "all_historical_a_armijo_transitions": bool(
            history_summary["all_historical_a_armijo_pass"]
        ),
        "final_b_below_initial": float(final["damped_full_burg_per_dim"])
        < float(initial["damped_full_burg_per_dim"]) - float(tolerances["B"]),
        "non_top_only_fraction_at_least_0p25": (
            total_lower90_reduction / max(total_a_reduction, 1e-30)
            >= i6.SUCCESS_BULK_FRACTION_MIN
        ),
        "final_p50_above_initial": float(final["m_p50"])
        > float(initial["m_p50"]) + float(tolerances["m_p50"]),
        "final_effective_rank_above_initial": float(final["effective_rank"])
        > float(initial["effective_rank"]) + float(tolerances["effective_rank"]),
        "final_count_lt_1e_4_strictly_lower": int(final["count_lt_1e_4"])
        < int(initial["count_lt_1e_4"]),
        "final_count_lt_1e_2_strictly_lower": int(final["count_lt_1e_2"])
        < int(initial["count_lt_1e_2"]),
        "final_count_lt_0p1_strictly_lower": int(final["count_lt_0p1"])
        < int(initial["count_lt_0p1"]),
        "final_mmax_not_above_initial": float(final["m_max"])
        <= float(initial["m_max"]) + float(tolerances["m_max"]),
        "high_tail_never_increases_beyond_floor": bool(
            (state_high_diffs <= float(tolerances["A_gt1"])).all()
        ),
        "all_state_values_finite": i6._numeric_finite(states),
        "all_spectrum_values_finite": i6._numeric_finite(spectra),
        "all_raw_spectrum_minima_valid": bool(states["m_raw_eig_min"].ge(-1e-8).all()),
        "all_exact_a_closures_valid": bool(
            states["a_direct_abs_error"].le(1e-9).all()
            and states["a_trace_abs_error"].le(1e-9).all()
        ),
    }
    continuation_rows = proposals.loc[proposals["target_update"].ge(START_UPDATE + 1)]
    immediate = bool(
        not continuation_rows.empty
        and int(continuation_rows.iloc[0]["target_update"]) == START_UPDATE + 1
        and int(continuation_rows.iloc[0]["accepted"]) == 1
    )
    new_updates = accepted_updates - START_UPDATE
    direction_rows = continuation_rows
    min_source = (
        float(direction_rows["unit_common_source_norm"].min())
        if not direction_rows.empty
        else 0.0
    )
    shadow_rows = (
        direction_rows.loc[
            direction_rows["fp64_shadow_valid"].eq(True)  # noqa: E712
        ]
        if "fp64_shadow_valid" in direction_rows
        else direction_rows.iloc[0:0]
    )
    min_shadow_cosine = (
        float(shadow_rows["fp32_fp64_direction_cosine"].min())
        if not shadow_rows.empty
        else 0.0
    )
    max_shadow_relative_error = (
        float(shadow_rows["fp32_fp64_direction_relative_error"].max())
        if not shadow_rows.empty
        else 0.0
    )
    amplification_unbounded = bool(
        not direction_rows.empty
        and direction_rows["unit_common_source_norm"].eq(0.0).any()
    )
    max_amplification = (
        None
        if amplification_unbounded
        else (
            float(direction_rows["common_amplification"].max())
            if not direction_rows.empty
            else 0.0
        )
    )
    final_source = (
        float(direction_rows.iloc[-1]["unit_common_source_norm"])
        if not direction_rows.empty
        else 0.0
    )
    accepted_continuation = continuation_rows.loc[continuation_rows["accepted"].eq(1)]
    minimum_accepted_alpha = (
        float(accepted_continuation["selected_alpha"].min())
        if not accepted_continuation.empty
        else 0.0
    )
    terminal_row = (
        continuation_rows.iloc[-1]
        if not continuation_rows.empty
        and int(continuation_rows.iloc[-1]["accepted"]) == 0
        else None
    )
    b_non_descent = bool(
        termination == "nonnegative_joint_slope"
        and terminal_row is not None
        and float(terminal_row["slope_A"]) < 0.0
        and float(terminal_row["slope_L_low"]) < 0.0
        and float(terminal_row["slope_B"]) >= 0.0
    )
    finite_grid_exhaustion = bool(
        termination == "backtracking_exhausted"
        and terminal_row is not None
        and all(
            float(terminal_row[f"slope_{name}"]) < 0.0 for name in ("A", "B", "L_low")
        )
    )
    finite_grid_conflict = bool(b_non_descent or finite_grid_exhaustion)
    total_bulk_fraction = total_lower90_reduction / max(total_a_reduction, 1e-30)
    scalar_only_repair = bool(
        immediate
        and continuation_a_reduction > float(tolerances["A"])
        and total_bulk_fraction < i6.SUCCESS_BULK_FRACTION_MIN
    )
    return {
        "immediate_premature_cutoff": immediate,
        "sustained_continuation": new_updates >= SUSTAINED_NEW_UPDATES,
        "b_non_descent": b_non_descent,
        "finite_grid_exhaustion": finite_grid_exhaustion,
        "finite_grid_conflict": finite_grid_conflict,
        "scalar_only_repair": scalar_only_repair,
        "new_accepted_updates": new_updates,
        "scientific_success": bool(all(success_gates.values())),
        "success_gates": success_gates,
        "termination": termination,
        "total_a_reduction": total_a_reduction,
        "total_lower90_reduction": total_lower90_reduction,
        "total_non_top_only_fraction": total_bulk_fraction,
        "continuation_a_reduction": continuation_a_reduction,
        "continuation_lower90_reduction": continuation_lower90_reduction,
        "continuation_non_top_only_fraction": continuation_lower90_reduction
        / max(continuation_a_reduction, 1e-30),
        "tail_bulk_updates": tail_bulk_updates,
        "near_cancellation_diagnostics": {
            "minimum_source_norm": min_source,
            "final_source_norm": final_source,
            "maximum_amplification": max_amplification,
            "maximum_amplification_unbounded": amplification_unbounded,
            "minimum_fp32_fp64_direction_cosine": min_shadow_cosine,
            "maximum_fp32_fp64_direction_relative_error": max_shadow_relative_error,
            "minimum_accepted_alpha": minimum_accepted_alpha,
            "descriptive_only": True,
        },
    }


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("production continuation accepts no CLI overrides")
    dependency_matches = _validate_dependencies()
    parent_checkpoint, parent_progress, parent_review = _validate_parent_packet()
    final_output = DEFAULT_OUTPUT.resolve()
    staging = Path(str(final_output) + ".incomplete")
    if final_output.exists():
        if (final_output / "FINALIZED.json").is_file() and (
            final_output / "INCOMPLETE"
        ).is_file():
            _validate_published_recovery(final_output)
            (final_output / "INCOMPLETE").unlink()
            print(
                f"[relaxed-continuation] recovered published finalization marker: "
                f"{final_output}",
                flush=True,
            )
            return
        raise FileExistsError(f"refusing to overwrite {final_output}")
    resume = staging.exists()
    if resume:
        if (
            not (staging / "INCOMPLETE").is_file()
            or not (staging / "progress_checkpoint.pt").is_file()
        ):
            raise RuntimeError("incomplete continuation lacks progress checkpoint")
        source_snapshot = staging / "executed_source_snapshot.py"
        if (
            _normalized_source_sha256(source_snapshot)
            != EXPECTED_NORMALIZED_SOURCE_SHA256
        ):
            raise RuntimeError("continuation resume source mismatch")
    else:
        staging.mkdir(parents=True)
        (staging / "INCOMPLETE").write_text(PROTOCOL_ID + "\n", encoding="utf-8")
        source_snapshot = staging / "executed_source_snapshot.py"
        shutil.copy2(Path(__file__), source_snapshot)
        shutil.copy2(PROTOCOL_PATH, staging / "protocol_snapshot.md")
        shutil.copy2(
            FROZEN_DEPENDENCY_MANIFEST,
            staging / "frozen_dependency_manifest_snapshot.json",
        )

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_handle = (staging / "run.log").open(
        "a" if resume else "w", encoding="utf-8", buffering=1
    )
    sys.stdout = _Tee(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, log_handle)  # type: ignore[assignment]
    started = time.perf_counter()
    device = torch.device("cuda:0")
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "resume": resume,
        "device": str(device),
        "dtype": "float32 model/HVP and executed direction; FP64 gradients/diagnostics",
        "seed": "none; exact complete-basis gradients",
        "cache_mode": "accepted h2048 run + immutable I6 final checkpoint",
        "parent_checkpoint": str(PARENT_FINAL_CHECKPOINT),
        "parent_checkpoint_sha256": EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256,
        "parent_progress_checkpoint_sha256": (
            EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256
        ),
        "start_update": START_UPDATE,
        "max_total_accepted_updates": MAX_TOTAL_ACCEPTED_UPDATES,
        "max_new_accepted_updates": MAX_NEW_ACCEPTED_UPDATES,
        "selected_arm": "low",
        "legacy_cancellation_norm_min": i6.CANCELLATION_NORM_MIN,
        "continuation_cancellation_rule": "finite and strictly nonzero only",
        "target_norm": i6.TARGET_NORM,
        "armijo_c1": i6.ARMIJO_C1,
        "line_alphas": list(i6.LINE_ALPHAS),
        "output_dir": str(final_output),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_sha256": _source_sha256(source_snapshot),
        "normalized_source_sha256": _normalized_source_sha256(source_snapshot),
        "dependency_manifest_sha256": sha256_file(FROZEN_DEPENDENCY_MANIFEST),
        "dependency_matches": dependency_matches,
    }
    if not resume:
        i6._atomic_json(staging / "resolved_config.json", resolved)
    else:
        stored = json.loads((staging / "resolved_config.json").read_text())
        for key in (
            "protocol_id",
            "parent_checkpoint_sha256",
            "parent_progress_checkpoint_sha256",
            "start_update",
            "max_total_accepted_updates",
            "selected_arm",
            "target_norm",
            "line_alphas",
            "source_sha256",
            "normalized_source_sha256",
            "dependency_manifest_sha256",
        ):
            if stored[key] != resolved[key]:
                raise RuntimeError(f"continuation resume config mismatch: {key}")
    print(
        f"[relaxed-continuation] startup {json.dumps(resolved, sort_keys=True)}",
        flush=True,
    )

    try:
        print("[relaxed-continuation] stage=load-fresh-run", flush=True)
        accepted_checkpoint = DEFAULT_RUN_DIR / "vae_checkpoint.pt"
        if sha256_file(accepted_checkpoint) != EXPECTED_CHECKPOINT:
            raise RuntimeError("accepted h2048 checkpoint hash mismatch")
        run = _load_run(DEFAULT_RUN_DIR, device=device)
        cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
        cfg = replace(cfg, vae_precond_hvp_mode="autograd")
        bank = pd.read_csv(STATE_BANK)
        selected_state = bank.loc[bank["state_position"].eq(STATE_POSITION)]
        if (
            len(selected_state) != 1
            or int(selected_state.iloc[0]["source_weight_index"]) != SOURCE_WEIGHT_INDEX
        ):
            raise RuntimeError("state-bank identity mismatch")
        record = run.records.iloc[SOURCE_WEIGHT_INDEX].to_dict()
        record["source_weight_index"] = SOURCE_WEIGHT_INDEX
        weight = run.weights[[SOURCE_WEIGHT_INDEX]].to(
            device=device, dtype=i6.torch_dtype(run.cfg)
        )
        with torch.no_grad():
            z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
        if sha256_tensor(z) != EXPECTED_Z_SHA256:
            raise RuntimeError("z fingerprint mismatch")
        task_set = _task_set_for_record(run.task_tensors, record)
        if not all(
            _batch_indices(
                task_set,
                batch_size=int(cfg.vae_precond_batch_size),
                step=10,
                sample_key=SOURCE_WEIGHT_INDEX,
                pair_key=pair_key,
            )
            is None
            for pair_key in range(8)
        ):
            raise RuntimeError("full CE batch gate failed")

        active_names = sorted(
            set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str))
        )
        if len(active_names) != 31 or set(active_names) != set(
            parent_checkpoint["active_model_state"]
        ):
            raise RuntimeError("parent active tensor names mismatch")
        named = dict(run.vae.named_parameters())
        for name, parameter in named.items():
            parameter.requires_grad_(name in active_names)
        active = [named[name] for name in active_names]
        if sum(parameter.numel() for parameter in active) != 11_685_120:
            raise RuntimeError("active parameter count mismatch")

        rows: dict[str, Any]
        accepted_updates: int
        chain_hash: str
        termination = "max_updates_reached"
        terminal = False
        intervention_preflight: dict[str, Any]
        if resume:
            print("[relaxed-continuation] stage=resume", flush=True)
            progress = torch.load(
                staging / "progress_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            _audit_progress_checkpoint(progress, parent_progress)
            accepted_updates = int(progress["accepted_updates"])
            rows = {
                key: list(progress[key])
                for key in (
                    "state_rows",
                    "spectrum_rows",
                    "proposal_rows",
                    "line_rows",
                    "selection_rows",
                )
            }
            rows["intervention_origin"] = dict(progress["intervention_origin"])
            chain_hash = str(progress["transition_chain_sha256"])
            intervention_preflight = dict(progress["intervention_preflight"])
            terminal = bool(progress["terminal"])
            termination = str(progress["termination"])
            active_state = progress["active_model_state"]
        else:
            print("[relaxed-continuation] stage=seed-parent", flush=True)
            accepted_updates = START_UPDATE
            rows = _seed_parent_rows(parent_progress)
            chain_hash = EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256
            intervention_preflight = {}
            active_state = parent_checkpoint["active_model_state"]
            i6._atomic_json(
                staging / "intervention_origin.json", rows["intervention_origin"]
            )
        with torch.no_grad():
            for name, parameter in zip(active_names, active, strict=True):
                stored_tensor = active_state[name]
                if tuple(stored_tensor.shape) != tuple(parameter.shape):
                    raise RuntimeError(f"active tensor shape mismatch: {name}")
                parameter.copy_(stored_tensor.to(device=device, dtype=parameter.dtype))
        if i6._named_tensor_hash(active_names, active) != str(
            EXPECTED_PARENT_ACTIVE_HASH
            if not resume
            else progress["active_parameter_hash"]
        ):
            raise RuntimeError("installed active parameter hash mismatch")

        tolerances = dict(parent_progress["tolerances"])
        initial_metrics = dict(parent_progress["initial_metrics"])
        if not resume:
            _write_progress(
                staging=staging,
                active_names=active_names,
                active=active,
                accepted_updates=accepted_updates,
                rows=rows,
                initial_metrics=initial_metrics,
                tolerances=tolerances,
                chain_hash=chain_hash,
                terminal=False,
                termination="running",
                intervention_preflight=intervention_preflight,
            )
            initial_progress = torch.load(
                staging / "progress_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            _audit_progress_checkpoint(initial_progress, parent_progress)
        if terminal:
            print(
                f"[relaxed-continuation] terminal resume total={accepted_updates} "
                f"termination={termination}",
                flush=True,
            )
        else:
            print(
                f"[relaxed-continuation] stage=exact-trajectory total={accepted_updates}",
                flush=True,
            )
            while accepted_updates < MAX_TOTAL_ACCEPTED_UPDATES:
                current_hash = i6._named_tensor_hash(active_names, active)
                current_parameters = [
                    parameter.detach().clone() for parameter in active
                ]
                current_state = i6._evaluate(
                    run=run, z=z, record=record, frozen_low_basis=None
                )
                current_metrics = current_state["metrics"]
                stored_current = rows["state_rows"][-1]
                current_metric_errors = _metric_replay_errors(
                    current_metrics, stored_current
                )
                current_replay_error = max(current_metric_errors.values())
                stored_spectrum = np.array(
                    [
                        float(row["m_eigenvalue"])
                        for row in rows["spectrum_rows"]
                        if int(row["accepted_update"]) == accepted_updates
                    ],
                    dtype=np.float64,
                )
                spectrum_replay_error = float(
                    np.max(np.abs(current_state["eig"].cpu().numpy() - stored_spectrum))
                )
                if (
                    current_hash != str(stored_current["parameter_hash"])
                    or current_replay_error > PARENT_REPLAY_MAX_ERROR
                    or spectrum_replay_error > PARENT_REPLAY_MAX_ERROR
                    or str(current_metrics["low_basis_hash"])
                    != str(stored_current["low_basis_hash"])
                ):
                    raise RuntimeError("current continuation state replay failed")

                proposal = accepted_updates + 1
                gradients, gradient_meta = i6._gradient_payload(
                    cfg=cfg,
                    run=run,
                    z=z,
                    record=record,
                    active=active,
                    state=current_state,
                    update=proposal,
                )
                direction, direction_meta = _relaxed_unit_direction(
                    gradients["A"], gradients["low"], active
                )
                slopes = {
                    "A": 0.0
                    if direction is None
                    else i6._dot(gradients["A"], direction),
                    "B": 0.0
                    if direction is None
                    else i6._dot(gradients["B"], direction),
                    "L_low": 0.0
                    if direction is None
                    else i6._dot(gradients["low"], direction),
                }
                if proposal == START_UPDATE + 1 and not intervention_preflight:
                    replay_errors = _proposal18_replay_errors(
                        direction_meta, rows["intervention_origin"]
                    )
                    replay_pass = _proposal18_replay_passes(
                        direction_meta, rows["intervention_origin"]
                    )
                    intervention_preflight = {
                        "parent_state_metric_max_abs_error": current_replay_error,
                        "parent_state_metric_errors": current_metric_errors,
                        "parent_spectrum_max_abs_error": spectrum_replay_error,
                        "parent_parameter_hash_matches": current_hash
                        == EXPECTED_PARENT_ACTIVE_HASH,
                        "parent_low_basis_hash_matches": str(
                            current_metrics["low_basis_hash"]
                        )
                        == str(stored_current["low_basis_hash"]),
                        "proposal18_replay_errors": replay_errors,
                        "proposal18_replay_pass": replay_pass,
                        "old_gate_rejects": not bool(
                            direction_meta["legacy_cancellation_gate_pass"]
                        ),
                        "relaxed_gate_accepts": bool(
                            direction_meta["relaxed_nonzero_gate_pass"]
                        ),
                        "parent_independent_review_valid": parent_review["valid"],
                    }
                    if not all(
                        (
                            intervention_preflight["parent_parameter_hash_matches"],
                            intervention_preflight["parent_low_basis_hash_matches"],
                            replay_pass,
                            intervention_preflight["old_gate_rejects"],
                            intervention_preflight["relaxed_gate_accepts"],
                        )
                    ):
                        raise RuntimeError(
                            f"proposal18 intervention preflight failed: {intervention_preflight}"
                        )
                    i6._atomic_json(
                        staging / "intervention_preflight.json", intervention_preflight
                    )

                theoretical_slope_a = (
                    -i6.TARGET_NORM
                    * float(direction_meta["gradient_a_norm"])
                    * float(direction_meta["theoretical_common_source_norm"])
                    / 2.0
                )
                theoretical_slope_low = (
                    -i6.TARGET_NORM
                    * float(direction_meta["gradient_x_norm"])
                    * float(direction_meta["theoretical_common_source_norm"])
                    / 2.0
                )
                proposal_record: dict[str, Any] = {
                    "phase": "relaxed_continuation",
                    "target_update": proposal,
                    "local_proposal": proposal - START_UPDATE,
                    "selected_arm": "low",
                    "base_parameter_hash": current_hash,
                    **direction_meta,
                    **{f"slope_{key}": value for key, value in slopes.items()},
                    "theoretical_slope_A": theoretical_slope_a,
                    "theoretical_slope_L_low": theoretical_slope_low,
                    "slope_A_vs_theory_abs_error": abs(
                        slopes["A"] - theoretical_slope_a
                    ),
                    "slope_L_low_vs_theory_abs_error": abs(
                        slopes["L_low"] - theoretical_slope_low
                    ),
                    "current_A": float(current_metrics["exact_a_per_dim"]),
                    "current_B": float(current_metrics["damped_full_burg_per_dim"]),
                    "current_low_energy": float(current_metrics["frozen_low_energy"]),
                    "current_a_gt1": float(current_metrics["a_gt1"]),
                    "gradient_metadata": json.dumps(gradient_meta, sort_keys=True),
                    "selected_alpha": 0.0,
                    "realized_path_length": 0.0,
                    "accepted": 0,
                    "failure": "",
                    "transition_chain_sha256": "",
                }
                if direction is None or not all(
                    value < 0.0 for value in slopes.values()
                ):
                    proposal_record["failure"] = (
                        "exact_cancellation"
                        if direction is None
                        else "nonnegative_joint_slope"
                    )
                    rows["proposal_rows"].append(proposal_record)
                    termination = str(proposal_record["failure"])
                    print(
                        f"[relaxed-continuation] stalled proposal={proposal} "
                        f"reason={termination} slopes={slopes}",
                        flush=True,
                    )
                    del gradients, direction, current_state
                    break

                accepted_payload: dict[str, Any] | None = None
                accepted_hash: str | None = None
                selected_alpha: float | None = None
                try:
                    for alpha in i6.LINE_ALPHAS:
                        endpoint, repeat_errors, endpoint_hash = i6._line_endpoint(
                            run=run,
                            z=z,
                            record=record,
                            active_names=active_names,
                            active=active,
                            base=current_parameters,
                            direction=direction,
                            alpha=alpha,
                            frozen_low_basis=current_state["current_low_basis"],
                        )
                        gates = i6._candidate_gates(
                            current=current_metrics,
                            candidate=endpoint["metrics"],
                            slopes=slopes,
                            alpha=alpha,
                            tolerances=tolerances,
                            require_low90=False,
                        )
                        if not _repeat_payload_values_valid(repeat_errors):
                            raise RuntimeError(
                                "invalid repeat payload; experiment is invalid"
                            )
                        repeat_pass = i6._repeat_passes(repeat_errors, tolerances)
                        passes = bool(all(gates.values()) and repeat_pass)
                        rows["line_rows"].append(
                            {
                                "phase": "relaxed_continuation",
                                "target_update": proposal,
                                "local_proposal": proposal - START_UPDATE,
                                "selected_arm": "low",
                                "base_parameter_hash": current_hash,
                                "alpha": alpha,
                                "realized_path_length": alpha * i6.TARGET_NORM,
                                **endpoint["metrics"],
                                **{
                                    f"gate_{key}": int(value)
                                    for key, value in gates.items()
                                },
                                **repeat_errors,
                                "endpoint_parameter_hash": endpoint_hash,
                                "passes": int(passes),
                            }
                        )
                        print(
                            f"[relaxed-continuation] proposal={proposal} alpha={alpha:.7g} "
                            f"pass={int(passes)} A={endpoint['metrics']['exact_a_per_dim']:.7g} "
                            f"B={endpoint['metrics']['damped_full_burg_per_dim']:.7g} "
                            f"Elow={endpoint['metrics']['frozen_low_energy']:.7g} "
                            f"source={float(direction_meta['unit_common_source_norm']):.7g}",
                            flush=True,
                        )
                        if passes:
                            accepted_payload = endpoint
                            accepted_hash = endpoint_hash
                            selected_alpha = alpha
                            break
                        i6._set_parameters(active, current_parameters, direction, 0.0)
                        if i6._named_tensor_hash(active_names, active) != current_hash:
                            raise RuntimeError(
                                "continuation line-search restoration failed"
                            )
                finally:
                    i6._set_parameters(active, current_parameters, direction, 0.0)
                    if i6._named_tensor_hash(active_names, active) != current_hash:
                        raise RuntimeError(
                            "continuation unconditional line-search restoration failed"
                        )

                if (
                    accepted_payload is None
                    or accepted_hash is None
                    or selected_alpha is None
                ):
                    proposal_record["failure"] = "backtracking_exhausted"
                    rows["proposal_rows"].append(proposal_record)
                    termination = "backtracking_exhausted"
                    print(
                        f"[relaxed-continuation] stalled proposal={proposal} "
                        "reason=backtracking_exhausted",
                        flush=True,
                    )
                    del gradients, direction, current_state
                    break

                i6._set_parameters(
                    active, current_parameters, direction, selected_alpha
                )
                committed_hash = i6._named_tensor_hash(active_names, active)
                if committed_hash != accepted_hash:
                    raise RuntimeError("continuation committed endpoint hash mismatch")
                accepted_updates += 1
                chain_hash = _chain_hash(
                    chain_hash,
                    update=accepted_updates,
                    base_hash=current_hash,
                    endpoint_hash=committed_hash,
                    alpha=selected_alpha,
                )
                proposal_record.update(
                    {
                        "selected_alpha": selected_alpha,
                        "realized_path_length": selected_alpha * i6.TARGET_NORM,
                        "accepted": 1,
                        "candidate_A": float(
                            accepted_payload["metrics"]["exact_a_per_dim"]
                        ),
                        "candidate_B": float(
                            accepted_payload["metrics"]["damped_full_burg_per_dim"]
                        ),
                        "candidate_low_energy": float(
                            accepted_payload["metrics"]["frozen_low_energy"]
                        ),
                        "candidate_a_gt1": float(accepted_payload["metrics"]["a_gt1"]),
                        "endpoint_parameter_hash": committed_hash,
                        "transition_chain_sha256": chain_hash,
                    }
                )
                rows["proposal_rows"].append(proposal_record)
                committed_metrics = dict(accepted_payload["metrics"])
                committed_metrics["parent_frozen_low_energy"] = float(
                    committed_metrics["frozen_low_energy"]
                )
                committed_metrics["frozen_low_energy"] = float(
                    committed_metrics["canonical_low_energy"]
                )
                rows["state_rows"].append(
                    {
                        **i6._state_row(
                            accepted_updates, committed_metrics, committed_hash
                        ),
                        "phase": "relaxed_continuation",
                    }
                )
                rows["spectrum_rows"].extend(
                    {
                        **row,
                        "phase": "relaxed_continuation",
                    }
                    for row in i6._spectrum_rows(
                        accepted_updates, accepted_payload["eig"]
                    )
                )
                _write_progress(
                    staging=staging,
                    active_names=active_names,
                    active=active,
                    accepted_updates=accepted_updates,
                    rows=rows,
                    initial_metrics=initial_metrics,
                    tolerances=tolerances,
                    chain_hash=chain_hash,
                    terminal=False,
                    termination="running",
                    intervention_preflight=intervention_preflight,
                )
                print(
                    f"[relaxed-continuation] accepted={accepted_updates}/"
                    f"{MAX_TOTAL_ACCEPTED_UPDATES} new={accepted_updates - START_UPDATE} "
                    f"alpha={selected_alpha:.7g} "
                    f"A={accepted_payload['metrics']['exact_a_per_dim']:.7g} "
                    f"low90={accepted_payload['metrics']['a_low90_abs_per_dim']:.7g} "
                    f"source={float(direction_meta['unit_common_source_norm']):.7g} "
                    f"shadow_cos={float(direction_meta['fp32_fp64_direction_cosine']):.9g} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                del gradients, direction, current_state, accepted_payload

        termination = _termination_after_loop(accepted_updates, termination)
        _write_progress(
            staging=staging,
            active_names=active_names,
            active=active,
            accepted_updates=accepted_updates,
            rows=rows,
            initial_metrics=initial_metrics,
            tolerances=tolerances,
            chain_hash=chain_hash,
            terminal=True,
            termination=termination,
            intervention_preflight=intervention_preflight,
        )
        final_progress = torch.load(
            staging / "progress_checkpoint.pt", map_location="cpu", weights_only=False
        )
        progress_audit_gates = _audit_progress_checkpoint(
            final_progress, parent_progress
        )
        states = pd.DataFrame(rows["state_rows"]).sort_values("accepted_update")
        spectra = pd.DataFrame(rows["spectrum_rows"]).sort_values(
            ["accepted_update", "rank"]
        )
        proposals = pd.DataFrame(rows["proposal_rows"])
        history_summary, history_rows = i6._historical_transition_audit(
            state_rows=rows["state_rows"],
            proposal_rows=rows["proposal_rows"],
            line_rows=rows["line_rows"],
            tolerances=tolerances,
        )
        i6._atomic_csv(staging / "historical_transition_audit.csv", history_rows)
        outcome = _outcome(
            states=states,
            spectra=spectra,
            proposals=proposals,
            tolerances=tolerances,
            accepted_updates=accepted_updates,
            termination=termination,
            history_summary=history_summary,
        )

        print("[relaxed-continuation] stage=serialize-final-checkpoint", flush=True)
        live_final_hash = i6._named_tensor_hash(active_names, active)
        final_checkpoint = {
            "protocol_id": PROTOCOL_ID,
            "parent_checkpoint_sha256": EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256,
            "parent_progress_checkpoint_sha256": (
                EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256
            ),
            "selected_arm": "low",
            "accepted_updates": accepted_updates,
            "new_accepted_updates": accepted_updates - START_UPDATE,
            "termination": termination,
            "transition_chain_sha256": chain_hash,
            "active_parameter_hash": live_final_hash,
            "active_model_state": {
                name: parameter.detach().cpu().clone()
                for name, parameter in zip(active_names, active, strict=True)
            },
            "source_weight_index": SOURCE_WEIGHT_INDEX,
            "z_sha256": EXPECTED_Z_SHA256,
            "stored_state_metrics": dict(rows["state_rows"][-1]),
        }
        i6._atomic_torch_save(staging / "final_checkpoint.pt", final_checkpoint)
        disk_checkpoint = torch.load(
            staging / "final_checkpoint.pt", map_location="cpu", weights_only=False
        )
        disk_metadata_pass = bool(
            disk_checkpoint.get("protocol_id") == PROTOCOL_ID
            and disk_checkpoint.get("parent_checkpoint_sha256")
            == EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256
            and disk_checkpoint.get("parent_progress_checkpoint_sha256")
            == EXPECTED_PARENT_PROGRESS_CHECKPOINT_SHA256
            and disk_checkpoint.get("selected_arm") == "low"
            and int(disk_checkpoint.get("accepted_updates", -1)) == accepted_updates
            and int(disk_checkpoint.get("new_accepted_updates", -1))
            == accepted_updates - START_UPDATE
            and disk_checkpoint.get("termination") == termination
            and disk_checkpoint.get("transition_chain_sha256") == chain_hash
            and disk_checkpoint.get("source_weight_index") == SOURCE_WEIGHT_INDEX
            and disk_checkpoint.get("z_sha256") == EXPECTED_Z_SHA256
            and set(disk_checkpoint.get("active_model_state", {})) == set(active_names)
            and disk_checkpoint.get("stored_state_metrics") == rows["state_rows"][-1]
            and _active_state_hash(disk_checkpoint["active_model_state"])
            == live_final_hash
            and disk_checkpoint.get("active_parameter_hash") == live_final_hash
        )
        if not disk_metadata_pass:
            raise RuntimeError("continuation disk checkpoint metadata mismatch")
        with torch.no_grad():
            for name in active_names:
                named[name].copy_(
                    disk_checkpoint["active_model_state"][name].to(device=device)
                )
        final_hash = i6._named_tensor_hash(active_names, active)
        final_state = i6._evaluate(run=run, z=z, record=record, frozen_low_basis=None)
        final_row = states.iloc[-1]
        final_replay_errors = _metric_replay_errors(
            final_state["metrics"], final_row.to_dict()
        )
        final_stored_spectrum = (
            spectra.loc[spectra["accepted_update"].eq(accepted_updates)]
            .sort_values("rank")["m_eigenvalue"]
            .to_numpy(dtype=np.float64)
        )
        final_spectrum_replay_error = float(
            np.max(np.abs(final_state["eig"].cpu().numpy() - final_stored_spectrum))
        )
        final_replay_pass = bool(
            final_hash == live_final_hash
            and final_hash == str(disk_checkpoint["active_parameter_hash"])
            and final_hash == str(final_row["parameter_hash"])
            and max(final_replay_errors.values()) <= PARENT_REPLAY_MAX_ERROR
            and final_spectrum_replay_error <= PARENT_REPLAY_MAX_ERROR
            and str(final_state["metrics"]["low_basis_hash"])
            == str(final_row["low_basis_hash"])
            and disk_metadata_pass
        )
        if not final_replay_pass:
            raise RuntimeError("continuation final checkpoint replay failed")
        outcome["success_gates"]["final_checkpoint_replay"] = final_replay_pass
        outcome["scientific_success"] = bool(all(outcome["success_gates"].values()))

        final_dependency_matches = _validate_dependencies()
        validity_gates = {
            "parent_packet_valid": True,
            "parent_final_checkpoint_hash_matches": sha256_file(PARENT_FINAL_CHECKPOINT)
            == EXPECTED_PARENT_FINAL_CHECKPOINT_SHA256,
            "parent_independent_review_valid": parent_review["valid"] is True,
            "intervention_preflight_complete": bool(intervention_preflight)
            and bool(intervention_preflight.get("proposal18_replay_pass")),
            "state_rows_complete": len(states) == accepted_updates + 1,
            "spectrum_rows_complete": len(spectra) == (accepted_updates + 1) * 512,
            "accepted_history_is_bijective": bool(
                history_summary["accepted_proposal_count_matches_states"]
                and history_summary["accepted_update_sequence_exact"]
                and history_summary["all_historical_transitions_pass"]
            ),
            "disk_checkpoint_metadata_replays": disk_metadata_pass,
            "final_checkpoint_replays": final_replay_pass,
            "source_snapshot_matches": _normalized_source_sha256(source_snapshot)
            == EXPECTED_NORMALIZED_SOURCE_SHA256,
            "live_source_unchanged": _normalized_source_sha256()
            == EXPECTED_NORMALIZED_SOURCE_SHA256,
            "dependencies_unchanged": all(final_dependency_matches.values()),
            "progress_checkpoint_fail_closed_audit": all(progress_audit_gates.values()),
            "state_and_spectra_finite": i6._numeric_finite(states)
            and i6._numeric_finite(spectra),
            "proposal_and_line_values_finite": i6._row_numeric_values_finite(
                rows["proposal_rows"]
            )
            and i6._row_numeric_values_finite(rows["line_rows"]),
        }
        valid = bool(all(validity_gates.values()))
        decision = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "scientific_success": outcome["scientific_success"] if valid else None,
            "accepted_updates": accepted_updates,
            "new_accepted_updates": accepted_updates - START_UPDATE,
            "selected_arm": "low",
            "termination": termination,
            "intervention_preflight": intervention_preflight,
            "outcome": outcome,
            "initial_metrics": initial_metrics,
            "intervention_start_metrics": dict(
                states.loc[states["accepted_update"].eq(START_UPDATE)].iloc[0]
            ),
            "final_metrics": dict(final_state["metrics"]),
            "historical_transition_summary": history_summary,
            "progress_audit_gates": progress_audit_gates,
            "tolerances": tolerances,
            "validity_gates": validity_gates,
            "final_replay_errors": final_replay_errors,
            "final_spectrum_replay_error": final_spectrum_replay_error,
            "final_checkpoint_file_sha256": sha256_file(
                staging / "final_checkpoint.pt"
            ),
            "elapsed_sec": time.perf_counter() - started,
        }
        i6._atomic_json(staging / "decision.json", decision)
        if not valid:
            raise RuntimeError("refusing to finalize invalid continuation")
        _plot(staging, states, spectra)
        artifact_manifest = _artifact_manifest(staging, source_snapshot)
        i6._atomic_json(staging / "artifact_manifest.json", artifact_manifest)
        finalized = {
            "protocol_id": PROTOCOL_ID,
            "status": "complete_awaiting_independent_review",
            "valid": valid,
            "scientific_success": outcome["scientific_success"],
            "immediate_premature_cutoff": outcome["immediate_premature_cutoff"],
            "sustained_continuation": outcome["sustained_continuation"],
            "accepted_updates": accepted_updates,
            "new_accepted_updates": accepted_updates - START_UPDATE,
            "termination": termination,
            "decision_sha256": sha256_file(staging / "decision.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        i6._atomic_json(staging / "FINALIZED.json", finalized)
        _validate_published_recovery(staging)
        print(
            f"[relaxed-continuation] complete valid={valid} "
            f"success={outcome['scientific_success']} "
            f"immediate={outcome['immediate_premature_cutoff']} "
            f"sustained={outcome['sustained_continuation']} "
            f"accepted={accepted_updates} termination={termination} "
            f"A={float(states.iloc[0]['exact_a_per_dim']):.7g}->"
            f"{float(states.iloc[-1]['exact_a_per_dim']):.7g} "
            f"bulk_fraction={outcome['total_non_top_only_fraction']:.6g} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        os.replace(staging, final_output)
        (final_output / "INCOMPLETE").unlink()
        print(f"[relaxed-continuation] published output={final_output}", flush=True)
    except Exception:
        print("[relaxed-continuation] FAILED; resumable staging retained", flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
