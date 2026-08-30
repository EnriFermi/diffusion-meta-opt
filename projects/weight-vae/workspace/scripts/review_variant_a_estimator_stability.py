from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_stability_unfinetuned_h2048"
)
PROTOCOL_ID = "a_estimator_stability_unfinetuned_h2048_v2"
BASE_SEED = 20260714
GRID = (1, 2, 4, 8, 16, 32)
REPEATS = 12
MAX_STATES = 32
MAX_PAIRS = 32
LATENT_DIM = 512
BATCH_SIZE = 128
TRAIN_POPULATION = 11021
EXPECTED_ATOMIC_ROWS = REPEATS * MAX_STATES * MAX_PAIRS
EXPECTED_BRANCH_ROWS = 2 * EXPECTED_ATOMIC_ROWS
EXPECTED_REPLICATE_ROWS = REPEATS * len(GRID) * len(GRID)
EXPECTED_PAIRWISE_ROWS = math.comb(REPEATS, 2) * len(GRID) * len(GRID)

EXPECTED_CHECKPOINT_SHA256 = (
    "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
)
EXPECTED_POOL_SHA256 = (
    "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef"
)
EXPECTED_RECORDS_SHA256 = (
    "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933"
)
EXPECTED_PROTOCOL_SHA256 = (
    "d75d231a26855da09271f0305375f7c4907299a529c9253b8ebf815b33f55e09"
)
EXPECTED_CONFIG_HASH = "4fc87a54349a39a2"
EXPECTED_CONFIG_FILE_SHA256 = (
    "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4"
)
EXPECTED_ACCEPTANCE_SHA256 = (
    "217bfbda8b23af770c94d982375a088eaa84b51e36e12f026edcc4b4798f56f9"
)

EXPECTED_SOURCE_HASHES = {
    "scripts/audit_variant_a_estimator_stability.py": "1f0ad1f4d146e9f41981bf7192192cd58ecd78ab23dd824529c88807befdd8ab",
    "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py": "f63e33a7e68bedf066f77ee8a7343e890690379119e19b2edc8e17eb65aa7eea",
    "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py": "ff67cc37006f5678d67ebaa919d348161d0b030beae95c1e3701ec07da35a43a",
    "scripts/run_verified_variant_a_finetune.py": "2be93de97bb15a8cdf3cfad27b4ed05e4816ff188646133d633cb88aa4799aa2",
}

MODULES = (
    "latent_to_context",
    "decoder_pos_proj",
    "decoder_context_norm",
    "decoder_query_norm",
    "decoder_cross_attn",
    "decoder_attn_norm",
    "decoder_qkv",
    "decoder_out_proj",
    "decoder_ffn",
    "patch_decoder",
)
PATCH_DECODER_BIAS = "patch_decoder.3.bias"
EXPECTED_ACTIVE_NAMES = (
    "latent_to_context.0.weight",
    "latent_to_context.0.bias",
    "latent_to_context.2.weight",
    "latent_to_context.2.bias",
    "decoder_pos_proj.weight",
    "decoder_pos_proj.bias",
    "decoder_context_norm.weight",
    "decoder_context_norm.bias",
    "decoder_query_norm.weight",
    "decoder_query_norm.bias",
    "decoder_cross_attn.in_proj_weight",
    "decoder_cross_attn.in_proj_bias",
    "decoder_cross_attn.out_proj.weight",
    "decoder_cross_attn.out_proj.bias",
    "decoder_attn_norm.weight",
    "decoder_attn_norm.bias",
    "decoder_qkv.weight",
    "decoder_qkv.bias",
    "decoder_out_proj.weight",
    "decoder_out_proj.bias",
    "decoder_ffn.0.weight",
    "decoder_ffn.0.bias",
    "decoder_ffn.1.weight",
    "decoder_ffn.1.bias",
    "decoder_ffn.3.weight",
    "decoder_ffn.3.bias",
    "patch_decoder.0.weight",
    "patch_decoder.0.bias",
    "patch_decoder.1.weight",
    "patch_decoder.1.bias",
    "patch_decoder.3.weight",
)
EXPECTED_ACTIVE_PARAMETER_COUNT = 11685120

REFERENCE_SPLITS = (
    (tuple(range(0, 6)), tuple(range(6, 12))),
    ((0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11)),
    ((0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11)),
)

REQUIRED_FILES = (
    "protocol.md",
    "resolved_config.json",
    "manifest.json",
    "validity.json",
    "preflight.json",
    "active_decoder_parameters.json",
    "active_parameters.csv",
    "sampled_states.csv",
    "branch_manifest.csv",
    "atomic_pair_samples.csv",
    "replicate_metrics.csv",
    "pairwise_gradient_cosines.csv",
    "module_gradient_metrics.csv",
    "reference_split_checks.csv",
    "stability_summary.csv",
    "decision.json",
)


class ReviewError(ValueError):
    """Raised when production artifacts cannot support an independent review."""


@dataclass
class ValidationLedger:
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(
        self, name: str, passed: bool, detail: Any, *, blocking: bool = True
    ) -> None:
        if name in self.checks:
            raise ReviewError(f"duplicate validity check name: {name}")
        self.checks[name] = {
            "passed": bool(passed),
            "blocking": bool(blocking),
            "detail": _json_native(detail),
        }

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks.values() if item["blocking"])

    def require_passed(self) -> None:
        failures = [
            name
            for name, item in self.checks.items()
            if item["blocking"] and not item["passed"]
        ]
        if failures:
            raise ReviewError(
                "blocking post-run validity checks failed: " + ", ".join(failures)
            )


class Reporter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def stage(self, message: str) -> None:
        if self.enabled:
            print(f"[a_stability_review] {message}", flush=True)


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise ReviewError(
            "PyTorch is required for checkpoint, state-RNG, and probe-stream replay"
        ) from error
    return torch


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ReviewError(f"nonfinite value cannot be serialized: {result}")
        return result
    if value is pd.NA:
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uint63(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) & ((1 << 63) - 1)


def exact_symmetric_scalar_error(left: float, right: float) -> float:
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        raise ReviewError(
            f"symmetric scalar error requires finite inputs, got {left_value}, {right_value}"
        )
    if left_value == 0.0 and right_value == 0.0:
        return 0.0
    denominator = abs(left_value) + abs(right_value)
    return float(2.0 * abs(left_value - right_value) / denominator)


def linear_quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ReviewError("quantiles require a nonempty, entirely finite input")
    return float(np.quantile(array, float(q), method="linear"))


def require_finite(
    frame: pd.DataFrame, columns: Sequence[str], table_name: str
) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ReviewError(
                f"{table_name} is missing finite metric column {column!r}"
            )
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        bad = np.flatnonzero(~np.isfinite(numeric))
        if bad.size:
            preview = ", ".join(str(int(index)) for index in bad[:8])
            raise ReviewError(
                f"{table_name}.{column} has NaN/inf/nonnumeric values at row indices {preview}"
            )


def _require_columns(
    frame: pd.DataFrame, required: Sequence[str], table_name: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ReviewError(f"{table_name} is missing columns: {missing}")


def _require_present(
    frame: pd.DataFrame, columns: Sequence[str], table_name: str
) -> None:
    for column in columns:
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.len().eq(0).any()
        ):
            raise ReviewError(f"{table_name}.{column} contains missing or empty values")


def _integer_values(
    frame: pd.DataFrame, columns: Sequence[str], table_name: str
) -> None:
    require_finite(frame, columns, table_name)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.equal(values, np.trunc(values)).all():
            raise ReviewError(f"{table_name}.{column} contains non-integer values")


def _require_exact_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected: set[tuple[int | str, ...]],
    table_name: str,
) -> None:
    if len(frame) != len(expected):
        raise ReviewError(
            f"{table_name} row count is {len(frame)}, expected exactly {len(expected)}"
        )
    if frame.duplicated(list(columns)).any():
        duplicate = (
            frame.loc[frame.duplicated(list(columns), keep=False), list(columns)]
            .head(5)
            .to_dict("records")
        )
        raise ReviewError(f"{table_name} has duplicate keys: {duplicate}")
    observed = set(frame.loc[:, list(columns)].itertuples(index=False, name=None))
    if observed != expected:
        missing = list(expected.difference(observed))[:5]
        extra = list(observed.difference(expected))[:5]
        raise ReviewError(
            f"{table_name} key set mismatch; missing={missing}, extra={extra}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ReviewError(f"{path.name} contains non-standard JSON constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise ReviewError(f"{path.name} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_native(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _load_inputs(
    input_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise ReviewError(
            f"production run is incomplete; missing required files: {missing}"
        )
    if (input_dir / "atomic_pair_samples.partial.csv").exists():
        raise ReviewError(
            "production run is incomplete; atomic_pair_samples.partial.csv still exists"
        )

    json_payloads = {
        "resolved_config": _read_json(input_dir / "resolved_config.json"),
        "manifest": _read_json(input_dir / "manifest.json"),
        "validity": _read_json(input_dir / "validity.json"),
        "preflight": _read_json(input_dir / "preflight.json"),
        "active": _read_json(input_dir / "active_decoder_parameters.json"),
        "executor_decision": _read_json(input_dir / "decision.json"),
    }
    frames = {
        "active": pd.read_csv(input_dir / "active_parameters.csv"),
        "states": pd.read_csv(input_dir / "sampled_states.csv"),
        "branches": pd.read_csv(input_dir / "branch_manifest.csv"),
        "atomic": pd.read_csv(input_dir / "atomic_pair_samples.csv"),
        "replicate": pd.read_csv(input_dir / "replicate_metrics.csv"),
        "pairwise": pd.read_csv(input_dir / "pairwise_gradient_cosines.csv"),
        "module": pd.read_csv(input_dir / "module_gradient_metrics.csv"),
        "split": pd.read_csv(input_dir / "reference_split_checks.csv"),
        "executor_summary": pd.read_csv(input_dir / "stability_summary.csv"),
    }
    return json_payloads, frames


def _validate_frame_contracts(
    frames: Mapping[str, pd.DataFrame], ledger: ValidationLedger
) -> None:
    states = frames["states"]
    branches = frames["branches"]
    atomic = frames["atomic"]
    replicate = frames["replicate"]
    pairwise = frames["pairwise"]
    module = frames["module"]
    split = frames["split"]
    executor_summary = frames["executor_summary"]

    _require_columns(
        states,
        (
            "repeat",
            "step_key",
            "state_position",
            "state_draw_seed",
            "train_position",
            "source_weight_index",
            "task_name",
            "tau",
            "source_lr",
            "source_step",
            "latent_hash",
            "duplicate_within_repeat",
            "duplicate_across_audit",
        ),
        "sampled_states.csv",
    )
    _integer_values(
        states,
        (
            "repeat",
            "step_key",
            "state_position",
            "state_draw_seed",
            "train_position",
            "source_weight_index",
        ),
        "sampled_states.csv",
    )
    require_finite(states, ("tau", "source_lr", "source_step"), "sampled_states.csv")
    _require_present(states, ("task_name", "latent_hash"), "sampled_states.csv")
    _require_exact_keys(
        states,
        ("repeat", "state_position"),
        {(repeat, state) for repeat in range(REPEATS) for state in range(MAX_STATES)},
        "sampled_states.csv",
    )

    branch_text = (
        "task_name",
        "latent_hash",
        "probe_hash",
        "hvp_hash",
        "batch_hash",
        "batch_indices",
    )
    _require_columns(
        branches,
        (
            "repeat",
            "step_key",
            "state_position",
            "pair_position",
            "source_weight_index",
            "task_name",
            "tau",
            "state_draw_seed",
            "train_position",
            "latent_hash",
            "batch_train_count",
            "batch_size_effective",
            "branch",
            "probe_seed",
            "probe_norm",
            "probe_hash",
            "hvp_norm",
            "hvp_hash",
            "batch_offset",
            "batch_hash",
            "batch_indices",
        ),
        "branch_manifest.csv",
    )
    _integer_values(
        branches,
        (
            "repeat",
            "step_key",
            "state_position",
            "pair_position",
            "source_weight_index",
            "state_draw_seed",
            "train_position",
            "batch_train_count",
            "batch_size_effective",
            "branch",
            "probe_seed",
            "batch_offset",
        ),
        "branch_manifest.csv",
    )
    require_finite(
        branches,
        [column for column in branches.columns if column not in branch_text],
        "branch_manifest.csv",
    )
    _require_present(branches, branch_text, "branch_manifest.csv")
    _require_exact_keys(
        branches,
        ("repeat", "state_position", "pair_position", "branch"),
        {
            (repeat, state, pair, branch)
            for repeat in range(REPEATS)
            for state in range(MAX_STATES)
            for pair in range(MAX_PAIRS)
            for branch in (0, 1)
        },
        "branch_manifest.csv",
    )

    atomic_text = {
        "task_name",
        "latent_hash",
        "h1_hash",
        "h2_hash",
        "probe_1_hash",
        "probe_2_hash",
        "batch_1_hash",
        "batch_2_hash",
        "batch_1_indices",
        "batch_2_indices",
    }
    _require_columns(
        atomic,
        (
            "repeat",
            "step_key",
            "state_position",
            "pair_position",
            "state_bin",
            "pair_bin",
            "source_weight_index",
            "task_name",
            "tau",
            "source_lr",
            "source_step",
            "state_draw_seed",
            "train_position",
            "latent_hash",
            "probe_seed_1",
            "probe_seed_2",
            "a_scalar",
            "gradient_norm",
            "unit_elapsed_sec",
            "cuda_peak_allocated_bytes",
            "h1_norm",
            "h2_norm",
            "h1_h2_dot",
            "h1_hash",
            "h2_hash",
            "probe_1_norm",
            "probe_2_norm",
            "probe_1_hash",
            "probe_2_hash",
            "branch_batch_overlap",
            "batch_1_hash",
            "batch_2_hash",
            "batch_1_offset",
            "batch_2_offset",
            "batch_train_count",
            "batch_size_effective",
            "batch_1_indices",
            "batch_2_indices",
        ),
        "atomic_pair_samples.csv",
    )
    _integer_values(
        atomic,
        (
            "repeat",
            "step_key",
            "state_position",
            "pair_position",
            "state_bin",
            "pair_bin",
            "source_weight_index",
            "state_draw_seed",
            "train_position",
            "probe_seed_1",
            "probe_seed_2",
            "cuda_peak_allocated_bytes",
            "batch_1_offset",
            "batch_2_offset",
            "batch_train_count",
            "batch_size_effective",
        ),
        "atomic_pair_samples.csv",
    )
    require_finite(
        atomic,
        [column for column in atomic.columns if column not in atomic_text],
        "atomic_pair_samples.csv",
    )
    _require_present(atomic, sorted(atomic_text), "atomic_pair_samples.csv")
    _require_exact_keys(
        atomic,
        ("repeat", "state_position", "pair_position"),
        {
            (repeat, state, pair)
            for repeat in range(REPEATS)
            for state in range(MAX_STATES)
            for pair in range(MAX_PAIRS)
        },
        "atomic_pair_samples.csv",
    )

    _require_columns(
        replicate,
        (
            "repeat",
            "states",
            "pairs",
            "hvp_count",
            "a_scalar",
            "reference_scalar",
            "scalar_symmetric_error",
            "gradient_cosine_to_reference",
            "gradient_relative_error",
            "gradient_norm",
            "reference_gradient_norm",
            "gradient_norm_ratio",
        ),
        "replicate_metrics.csv",
    )
    require_finite(replicate, list(replicate.columns), "replicate_metrics.csv")
    _integer_values(
        replicate, ("repeat", "states", "pairs", "hvp_count"), "replicate_metrics.csv"
    )
    _require_exact_keys(
        replicate,
        ("repeat", "states", "pairs"),
        {
            (repeat, states_count, pairs_count)
            for repeat in range(REPEATS)
            for states_count in GRID
            for pairs_count in GRID
        },
        "replicate_metrics.csv",
    )

    _require_columns(
        pairwise,
        (
            "states",
            "pairs",
            "left_repeat",
            "right_repeat",
            "gradient_cosine",
            "gradient_norm_ratio",
        ),
        "pairwise_gradient_cosines.csv",
    )
    require_finite(pairwise, list(pairwise.columns), "pairwise_gradient_cosines.csv")
    _integer_values(
        pairwise,
        ("states", "pairs", "left_repeat", "right_repeat"),
        "pairwise_gradient_cosines.csv",
    )
    _require_exact_keys(
        pairwise,
        ("states", "pairs", "left_repeat", "right_repeat"),
        {
            (states_count, pairs_count, left, right)
            for states_count in GRID
            for pairs_count in GRID
            for left in range(REPEATS)
            for right in range(left + 1, REPEATS)
        },
        "pairwise_gradient_cosines.csv",
    )

    _require_columns(
        module,
        (
            "repeat",
            "states",
            "pairs",
            "module",
            "reference_energy_fraction",
            "gradient_cosine_to_reference",
            "gradient_norm_ratio",
        ),
        "module_gradient_metrics.csv",
    )
    require_finite(
        module,
        [column for column in module.columns if column != "module"],
        "module_gradient_metrics.csv",
    )
    _require_present(module, ("module",), "module_gradient_metrics.csv")
    _integer_values(
        module, ("repeat", "states", "pairs"), "module_gradient_metrics.csv"
    )
    _require_exact_keys(
        module,
        ("repeat", "states", "pairs", "module"),
        {
            (repeat, states_count, pairs_count, module_name)
            for repeat in range(REPEATS)
            for states_count in GRID
            for pairs_count in GRID
            for module_name in MODULES
        },
        "module_gradient_metrics.csv",
    )

    _require_columns(
        split,
        (
            "split",
            "left_repeats",
            "right_repeats",
            "gradient_cosine",
            "gradient_norm_ratio",
            "scalar_left",
            "scalar_right",
            "scalar_symmetric_error",
        ),
        "reference_split_checks.csv",
    )
    require_finite(
        split,
        (
            "split",
            "gradient_cosine",
            "gradient_norm_ratio",
            "scalar_left",
            "scalar_right",
            "scalar_symmetric_error",
        ),
        "reference_split_checks.csv",
    )
    _integer_values(split, ("split",), "reference_split_checks.csv")
    _require_present(
        split, ("left_repeats", "right_repeats"), "reference_split_checks.csv"
    )
    _require_exact_keys(
        split, ("split",), {(0,), (1,), (2,)}, "reference_split_checks.csv"
    )

    _require_columns(executor_summary, ("states", "pairs"), "stability_summary.csv")
    numeric_summary_columns = [
        column
        for column in executor_summary.columns
        if column
        not in {
            "module_stability_pass",
            "scalar_stability_pass",
            "gradient_stability_pass",
            "cell_stability_pass",
            "persistent_stability_pass",
        }
    ]
    require_finite(executor_summary, numeric_summary_columns, "stability_summary.csv")
    _require_exact_keys(
        executor_summary,
        ("states", "pairs"),
        {(states_count, pairs_count) for states_count in GRID for pairs_count in GRID},
        "stability_summary.csv",
    )

    ledger.add(
        "strict_table_row_and_key_counts",
        True,
        {
            "sampled_states": len(states),
            "branches": len(branches),
            "atomic": len(atomic),
            "replicate": len(replicate),
            "pairwise": len(pairwise),
            "module": len(module),
            "split": len(split),
            "grid_cells": len(executor_summary),
        },
    )
    ledger.add(
        "strict_metric_finiteness",
        True,
        "all required atomic/replicate/pairwise/module/split values are finite",
    )


def _validate_provenance(
    input_dir: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, pd.DataFrame],
    ledger: ValidationLedger,
) -> Any:
    torch = _load_torch()
    resolved = payloads["resolved_config"]
    manifest = payloads["manifest"]
    executor_validity = payloads["validity"]
    preflight = payloads["preflight"]
    active_payload = payloads["active"]
    active_frame = frames["active"]

    expected_config = {
        "protocol_id": PROTOCOL_ID,
        "states": list(GRID),
        "pairs": list(GRID),
        "repeats": REPEATS,
        "batch_size": BATCH_SIZE,
        "seed": BASE_SEED,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "weight_pool_sha256": EXPECTED_POOL_SHA256,
        "weight_records_sha256": EXPECTED_RECORDS_SHA256,
        "baseline_acceptance_passed": True,
        "estimator_scope": "local",
        "hvp_mode": "stopped_composite",
        "probe_scale": 1.0,
        "loss_clip": 0.0,
        "gradient_damping": 0.0,
        "z_semantics": "encoded_mu_detached",
        "state_population": "checkpoint_train_indices_uniform_with_replacement",
    }
    observed_config = {key: resolved.get(key) for key in expected_config}
    ledger.add(
        "resolved_estimator_config_identity",
        observed_config == expected_config,
        observed_config,
    )
    ledger.add(
        "resolved_output_identity",
        Path(str(resolved.get("output_dir", ""))).resolve() == input_dir,
        resolved.get("output_dir"),
    )
    ledger.add(
        "resolved_protocol_path_identity",
        Path(str(resolved.get("protocol_path", ""))).resolve()
        == input_dir / "protocol.md",
        resolved.get("protocol_path"),
    )
    ledger.add(
        "production_manifest_complete",
        manifest.get("status") == "complete",
        manifest.get("status"),
    )
    ledger.add(
        "manifest_resolved_config_identity",
        manifest.get("resolved_config") == resolved,
        "standalone vs manifest",
    )
    ledger.add(
        "executor_validity_passed",
        executor_validity.get("passed") is True,
        executor_validity.get("checks", {}),
    )
    ledger.add(
        "manifest_validity_identity",
        manifest.get("validity") == executor_validity,
        "standalone vs manifest",
    )

    protocol_hash = sha256_file(input_dir / "protocol.md")
    ledger.add(
        "frozen_protocol_hash",
        protocol_hash == EXPECTED_PROTOCOL_SHA256 == resolved.get("protocol_sha256"),
        {
            "observed": protocol_hash,
            "resolved": resolved.get("protocol_sha256"),
            "expected": EXPECTED_PROTOCOL_SHA256,
        },
    )

    run_dir = Path(str(resolved.get("run_dir", ""))).expanduser().resolve()
    source_paths = {
        "checkpoint": run_dir / "vae_checkpoint.pt",
        "pool": run_dir / "weight_pool.pt",
        "records": run_dir / "weight_pool_records.csv",
        "config": run_dir / "config.json",
        "acceptance": run_dir / "baseline_acceptance.json",
    }
    absent_sources = [str(path) for path in source_paths.values() if not path.is_file()]
    if absent_sources:
        raise ReviewError(f"frozen source files are missing: {absent_sources}")

    observed_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    expected_hashes = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "pool": EXPECTED_POOL_SHA256,
        "records": EXPECTED_RECORDS_SHA256,
        "config": EXPECTED_CONFIG_FILE_SHA256,
        "acceptance": EXPECTED_ACCEPTANCE_SHA256,
    }
    ledger.add(
        "frozen_source_hashes",
        observed_hashes == expected_hashes,
        {"observed": observed_hashes, "expected": expected_hashes},
    )
    ledger.add(
        "checkpoint_unchanged_identity",
        manifest.get("checkpoint_sha256_after") == EXPECTED_CHECKPOINT_SHA256
        and executor_validity.get("checkpoint_sha256_before")
        == EXPECTED_CHECKPOINT_SHA256
        and executor_validity.get("checkpoint_sha256_after")
        == EXPECTED_CHECKPOINT_SHA256,
        {
            "manifest_after": manifest.get("checkpoint_sha256_after"),
            "validity_before": executor_validity.get("checkpoint_sha256_before"),
            "validity_after": executor_validity.get("checkpoint_sha256_after"),
        },
    )

    source_hash_details: dict[str, Any] = {}
    recorded_code_hashes = resolved.get("code_sha256", {})
    code_hash_pass = isinstance(recorded_code_hashes, dict)
    for relative, expected_hash in EXPECTED_SOURCE_HASHES.items():
        path = ROOT / relative
        observed_hash = sha256_file(path) if path.is_file() else "missing"
        recorded_hash = (
            recorded_code_hashes.get(str(path.resolve()))
            if isinstance(recorded_code_hashes, dict)
            else None
        )
        source_hash_details[relative] = {
            "observed": observed_hash,
            "expected": expected_hash,
            "executor_recorded": recorded_hash,
        }
        code_hash_pass = code_hash_pass and observed_hash == expected_hash
        if relative != "scripts/run_verified_variant_a_finetune.py":
            code_hash_pass = code_hash_pass and recorded_hash == expected_hash
    expected_recorded_paths = {
        str((ROOT / relative).resolve())
        for relative in EXPECTED_SOURCE_HASHES
        if relative != "scripts/run_verified_variant_a_finetune.py"
    }
    code_hash_pass = (
        code_hash_pass and set(recorded_code_hashes) == expected_recorded_paths
    )
    ledger.add("frozen_code_hashes", code_hash_pass, source_hash_details)
    ledger.add(
        "finetune_source_hash_posthoc_freeze",
        source_hash_details["scripts/run_verified_variant_a_finetune.py"]["observed"]
        == EXPECTED_SOURCE_HASHES["scripts/run_verified_variant_a_finetune.py"],
        "executor omitted this protocol-listed source; reviewer compares it to the hash frozen during production",
    )

    config_payload = _read_json(source_paths["config"])
    config = config_payload.get("config", {})
    config_identity = {
        "config_hash": config_payload.get("config_hash"),
        "vae_arch": config.get("vae_arch"),
        "vae_hidden_dim": config.get("vae_hidden_dim"),
        "latent_dim": config.get("latent_dim"),
        "tiny_bigvae_output_mode": config.get("tiny_bigvae_output_mode"),
        "dtype": config.get("dtype"),
    }
    ledger.add(
        "accepted_model_config_identity",
        config_identity
        == {
            "config_hash": EXPECTED_CONFIG_HASH,
            "vae_arch": "tiny_big_vae",
            "vae_hidden_dim": 2048,
            "latent_dim": LATENT_DIM,
            "tiny_bigvae_output_mode": "direct",
            "dtype": "float32",
        },
        config_identity,
    )
    acceptance = _read_json(source_paths["acceptance"])
    ledger.add(
        "baseline_acceptance_identity",
        acceptance.get("passed") is True,
        {"passed": acceptance.get("passed")},
    )

    checkpoint = torch.load(
        source_paths["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    train_indices = (
        checkpoint.get("train_indices") if isinstance(checkpoint, dict) else None
    )
    train_indices_valid = (
        isinstance(train_indices, torch.Tensor)
        and train_indices.ndim == 1
        and train_indices.numel() == TRAIN_POPULATION
    )
    ledger.add(
        "checkpoint_train_population_identity",
        train_indices_valid,
        {
            "type": type(train_indices).__name__,
            "count": int(train_indices.numel())
            if isinstance(train_indices, torch.Tensor)
            else None,
        },
    )
    if not train_indices_valid:
        ledger.require_passed()
        raise ReviewError("checkpoint train_indices are unavailable")
    train_indices = train_indices.detach().cpu().long()

    _require_columns(
        active_frame,
        ("parameter", "module", "shape", "offset_start", "offset_end", "numel"),
        "active_parameters.csv",
    )
    _integer_values(
        active_frame, ("offset_start", "offset_end", "numel"), "active_parameters.csv"
    )
    _require_present(
        active_frame, ("parameter", "module", "shape"), "active_parameters.csv"
    )
    active_names = tuple(
        str(value) for value in active_payload.get("active_parameter_names", [])
    )
    frame_names = tuple(active_frame["parameter"].astype(str))
    payload_parameter_names = tuple(
        str(row.get("parameter")) for row in active_payload.get("parameters", [])
    )
    ledger.add(
        "active_parameter_order_identity",
        active_names == EXPECTED_ACTIVE_NAMES == frame_names == payload_parameter_names,
        {
            "observed_count": len(active_names),
            "expected_count": len(EXPECTED_ACTIVE_NAMES),
        },
    )
    offsets = active_frame[["offset_start", "offset_end", "numel"]].astype(np.int64)
    contiguous = bool(
        len(offsets) == len(EXPECTED_ACTIVE_NAMES)
        and int(offsets.iloc[0]["offset_start"]) == 0
        and np.equal(
            offsets["offset_end"] - offsets["offset_start"], offsets["numel"]
        ).all()
        and np.equal(
            offsets["offset_start"].iloc[1:].to_numpy(),
            offsets["offset_end"].iloc[:-1].to_numpy(),
        ).all()
        and int(offsets.iloc[-1]["offset_end"]) == EXPECTED_ACTIVE_PARAMETER_COUNT
    )
    ledger.add(
        "active_parameter_slices_contiguous",
        contiguous,
        {"last_offset": int(offsets.iloc[-1]["offset_end"])},
    )
    ledger.add(
        "active_parameter_count_identity",
        active_payload.get("active_parameter_count") == EXPECTED_ACTIVE_PARAMETER_COUNT,
        active_payload.get("active_parameter_count"),
    )
    parameter_rows = active_payload.get("parameters", [])
    ordered_hash = hashlib.sha256(
        json.dumps(parameter_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    ledger.add(
        "active_ordered_list_hash",
        ordered_hash == active_payload.get("ordered_list_sha256"),
        {
            "observed": ordered_hash,
            "recorded": active_payload.get("ordered_list_sha256"),
        },
    )
    excluded = tuple(
        str(value) for value in active_payload.get("excluded_parameter_names", [])
    )
    excluded_under_active_prefixes = {
        name for name in excluded if name.split(".", 1)[0] in set(MODULES)
    }
    ledger.add(
        "active_module_set_identity",
        set(active_frame["module"].astype(str)) == set(MODULES),
        sorted(set(active_frame["module"].astype(str))),
    )
    ledger.add(
        "patch_decoder_bias_structural_zero_erratum",
        PATCH_DECODER_BIAS not in active_names
        and PATCH_DECODER_BIAS in excluded
        and excluded_under_active_prefixes == {PATCH_DECODER_BIAS},
        {
            "active": PATCH_DECODER_BIAS in active_names,
            "excluded": PATCH_DECODER_BIAS in excluded,
            "other_excluded_active_prefix_parameters": sorted(
                excluded_under_active_prefixes.difference({PATCH_DECODER_BIAS})
            ),
        },
    )
    ledger.add(
        "preflight_active_identity",
        tuple(preflight.get("active_parameter_names", [])) == EXPECTED_ACTIVE_NAMES
        and tuple(preflight.get("excluded_parameter_names", [])) == excluded,
        "preflight vs active manifest",
    )
    preflight_numeric = (
        "full_scalar",
        "atomic_scalar",
        "scalar_abs_error",
        "gradient_cosine",
        "gradient_relative_error",
        "full_gradient_norm",
        "atomic_gradient_norm",
        "stopped_vs_autograd_hvp_cosine",
        "stopped_vs_autograd_hvp_relative_error",
    )
    preflight_finite = all(
        math.isfinite(float(preflight.get(key, float("nan"))))
        for key in preflight_numeric
    )
    preflight_pass = bool(
        preflight_finite
        and preflight.get("sample_count") == 2
        and preflight.get("pair_count") == 2
        and preflight.get("active_parameter_count") == EXPECTED_ACTIVE_PARAMETER_COUNT
        and float(preflight["scalar_abs_error"]) <= 5e-5
        and float(preflight["gradient_cosine"]) >= 0.99999
        and float(preflight["gradient_relative_error"]) <= 5e-5
        and float(preflight["stopped_vs_autograd_hvp_cosine"]) >= 0.99999
        and float(preflight["stopped_vs_autograd_hvp_relative_error"]) <= 5e-5
    )
    ledger.add(
        "materialized_preflight_identity",
        preflight_pass,
        {key: preflight.get(key) for key in preflight_numeric},
    )
    return train_indices


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ReviewError(f"invalid boolean value {value!r}")


def _validate_state_replay(
    states: pd.DataFrame, train_indices: Any, ledger: ValidationLedger
) -> None:
    torch = _load_torch()
    mismatches: list[str] = []
    for row in states.sort_values(["repeat", "state_position"]).itertuples(index=False):
        repeat = int(row.repeat)
        state_position = int(row.state_position)
        expected_seed = stable_uint63(
            PROTOCOL_ID, BASE_SEED, "state", repeat, state_position, -1, -1
        )
        generator = torch.Generator(device="cpu").manual_seed(expected_seed)
        expected_position = int(
            torch.randint(0, TRAIN_POPULATION, (1,), generator=generator).item()
        )
        expected_source = int(train_indices[expected_position].item())
        if int(row.state_draw_seed) != expected_seed:
            mismatches.append(f"seed({repeat},{state_position})")
        if int(row.train_position) != expected_position:
            mismatches.append(f"position({repeat},{state_position})")
        if int(row.source_weight_index) != expected_source:
            mismatches.append(f"source({repeat},{state_position})")
        if int(row.step_key) != repeat + 1:
            mismatches.append(f"step({repeat},{state_position})")

    expected_within = states.duplicated(
        ["repeat", "source_weight_index"], keep=False
    ).to_numpy()
    expected_across = states.duplicated(["source_weight_index"], keep=False).to_numpy()
    observed_within = np.asarray(
        [_parse_bool(value) for value in states["duplicate_within_repeat"]]
    )
    observed_across = np.asarray(
        [_parse_bool(value) for value in states["duplicate_across_audit"]]
    )
    duplicate_flags_match = np.array_equal(
        expected_within, observed_within
    ) and np.array_equal(expected_across, observed_across)
    ledger.add(
        "state_seed_formula_and_rng_replay",
        not mismatches and states["state_draw_seed"].nunique() == len(states),
        {
            "mismatches": mismatches[:12],
            "unique_seeds": int(states["state_draw_seed"].nunique()),
        },
    )
    ledger.add(
        "state_duplicate_flags_recomputed",
        duplicate_flags_match,
        {
            "within_repeat_slots": int(expected_within.sum()),
            "across_audit_slots": int(expected_across.sum()),
        },
    )


def _replay_probe(seed: int, *, device: str) -> tuple[str, float]:
    torch = _load_torch()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    probe = torch.randn(
        (LATENT_DIM,), generator=generator, device="cpu", dtype=torch.float32
    ).to(device=torch.device(device), dtype=torch.float32)
    probe = probe * (math.sqrt(float(LATENT_DIM)) / probe.norm().clamp_min(1e-12))
    digest = hashlib.sha256(
        probe.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()
    norm = float(torch.linalg.vector_norm(probe.float()).item())
    return digest, norm


def _parse_batch_indices(value: Any) -> list[int]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ReviewError(f"invalid batch index JSON: {value!r}") from error
    if not isinstance(parsed, list) or len(parsed) != BATCH_SIZE:
        raise ReviewError(f"batch index list must have exactly {BATCH_SIZE} entries")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ReviewError("batch index list must contain only integers")
    return parsed


def _validate_branch_replay(
    states: pd.DataFrame,
    branches: pd.DataFrame,
    atomic: pd.DataFrame,
    ledger: ValidationLedger,
    *,
    probe_device: str,
    reporter: Reporter,
) -> None:
    state_lookup = states.set_index(["repeat", "state_position"]).to_dict(
        orient="index"
    )
    replay_mismatches: list[str] = []
    metadata_mismatches: list[str] = []
    probe_hash_mismatches: list[str] = []
    batch_hash_mismatches: list[str] = []

    ordered_branches = branches.sort_values(
        ["repeat", "state_position", "pair_position", "branch"]
    )
    replay_started = time.perf_counter()
    for branch_number, row in enumerate(
        ordered_branches.itertuples(index=False), start=1
    ):
        repeat = int(row.repeat)
        state_position = int(row.state_position)
        pair_position = int(row.pair_position)
        branch = int(row.branch)
        key = (repeat, state_position)
        state = state_lookup[key]
        label = f"({repeat},{state_position},{pair_position},{branch})"
        expected_seed = stable_uint63(
            PROTOCOL_ID,
            BASE_SEED,
            "probe",
            repeat,
            state_position,
            pair_position,
            branch,
        )
        if int(row.probe_seed) != expected_seed:
            replay_mismatches.append("seed" + label)
        expected_probe_hash, expected_probe_norm = _replay_probe(
            expected_seed, device=probe_device
        )
        if str(row.probe_hash) != expected_probe_hash or not math.isclose(
            float(row.probe_norm), expected_probe_norm, rel_tol=1e-7, abs_tol=1e-7
        ):
            probe_hash_mismatches.append(label)

        if (
            int(row.step_key) != repeat + 1
            or int(row.source_weight_index) != int(state["source_weight_index"])
            or int(row.state_draw_seed) != int(state["state_draw_seed"])
            or int(row.train_position) != int(state["train_position"])
            or str(row.task_name) != str(state["task_name"])
            or float(row.tau) != float(state["tau"])
            or str(row.latent_hash) != str(state["latent_hash"])
        ):
            metadata_mismatches.append(label)

        train_count = int(row.batch_train_count)
        pair_key = 2 * pair_position + branch
        expected_offset = (
            int(row.source_weight_index) * 1009 + (repeat + 1) * 9176 + pair_key * 7919
        ) % train_count
        expected_indices = [
            (expected_offset + position) % train_count for position in range(BATCH_SIZE)
        ]
        observed_indices = _parse_batch_indices(row.batch_indices)
        expected_batch_hash = hashlib.sha256(
            np.asarray(expected_indices, dtype=np.int64).tobytes()
        ).hexdigest()
        if (
            train_count <= BATCH_SIZE
            or int(row.batch_size_effective) != BATCH_SIZE
            or int(row.batch_offset) != expected_offset
            or observed_indices != expected_indices
            or str(row.batch_hash) != expected_batch_hash
        ):
            batch_hash_mismatches.append(label)
        if (
            branch_number == 1
            or branch_number % 1024 == 0
            or branch_number == len(ordered_branches)
        ):
            elapsed = time.perf_counter() - replay_started
            reporter.stage(
                "replay_progress "
                f"branch={branch_number}/{len(ordered_branches)} "
                f"device={probe_device} elapsed_sec={elapsed:.1f} "
                f"rate={branch_number / max(elapsed, 1e-12):.1f}/s"
            )

    ledger.add(
        "probe_seed_formula_and_stream_replay",
        not replay_mismatches
        and not probe_hash_mismatches
        and branches["probe_seed"].nunique() == EXPECTED_BRANCH_ROWS,
        {
            "seed_mismatches": replay_mismatches[:12],
            "probe_hash_or_norm_mismatches": probe_hash_mismatches[:12],
            "unique_seeds": int(branches["probe_seed"].nunique()),
        },
    )
    ledger.add(
        "branch_state_metadata_identity",
        not metadata_mismatches,
        {"mismatches": metadata_mismatches[:12]},
    )
    ledger.add(
        "deterministic_batch_formula_and_hash_replay",
        not batch_hash_mismatches,
        {"mismatches": batch_hash_mismatches[:12], "checked_branches": len(branches)},
    )

    branch_lookup = branches.set_index(
        ["repeat", "state_position", "pair_position", "branch"]
    )
    atomic_mismatches: list[str] = []
    for row in atomic.itertuples(index=False):
        repeat = int(row.repeat)
        state_position = int(row.state_position)
        pair_position = int(row.pair_position)
        label = f"({repeat},{state_position},{pair_position})"
        state = state_lookup[(repeat, state_position)]
        first = branch_lookup.loc[(repeat, state_position, pair_position, 0)]
        second = branch_lookup.loc[(repeat, state_position, pair_position, 1)]
        expected_state_bin = next(
            index
            for index, endpoint in enumerate(GRID)
            if state_position + 1 <= endpoint
        )
        expected_pair_bin = next(
            index
            for index, endpoint in enumerate(GRID)
            if pair_position + 1 <= endpoint
        )
        if (
            int(row.step_key) != repeat + 1
            or int(row.state_bin) != expected_state_bin
            or int(row.pair_bin) != expected_pair_bin
            or int(row.source_weight_index) != int(state["source_weight_index"])
            or int(row.state_draw_seed) != int(state["state_draw_seed"])
            or int(row.probe_seed_1) != int(first["probe_seed"])
            or int(row.probe_seed_2) != int(second["probe_seed"])
            or str(row.batch_1_indices) != str(first["batch_indices"])
            or str(row.batch_2_indices) != str(second["batch_indices"])
        ):
            atomic_mismatches.append(label)
    ledger.add(
        "atomic_branch_and_prefix_metadata_identity",
        not atomic_mismatches,
        {"mismatches": atomic_mismatches[:12]},
    )


def _recompute_scalar_replicates(
    atomic: pd.DataFrame,
    replicate: pd.DataFrame,
    ledger: ValidationLedger,
) -> pd.DataFrame:
    reviewed = replicate.copy()
    candidate_norms = reviewed["gradient_norm"].to_numpy(dtype=np.float64)
    reference_norms = reviewed["reference_gradient_norm"].to_numpy(dtype=np.float64)
    if np.any(candidate_norms <= 0.0) or np.any(reference_norms <= 0.0):
        raise ReviewError(
            "replicate_metrics.csv contains a zero or negative global gradient norm"
        )
    reviewed["reviewed_gradient_norm_ratio"] = candidate_norms / reference_norms
    reviewed["reviewed_scalar_symmetric_error"] = [
        exact_symmetric_scalar_error(left, right)
        for left, right in zip(
            reviewed["a_scalar"], reviewed["reference_scalar"], strict=True
        )
    ]

    prefix_values: dict[tuple[int, int, int], float] = {}
    max_values: dict[int, float] = {}
    for repeat in range(REPEATS):
        repeat_rows = atomic.loc[atomic["repeat"] == repeat]
        for states_count in GRID:
            for pairs_count in GRID:
                values = repeat_rows.loc[
                    (repeat_rows["state_position"] < states_count)
                    & (repeat_rows["pair_position"] < pairs_count),
                    "a_scalar",
                ].to_numpy(dtype=np.float64)
                if values.size != states_count * pairs_count:
                    raise ReviewError(
                        f"atomic scalar prefix ({repeat},{states_count},{pairs_count}) has {values.size} rows"
                    )
                prefix_values[(repeat, states_count, pairs_count)] = float(
                    values.sum(dtype=np.float64) / values.size
                )
        max_values[repeat] = prefix_values[(repeat, MAX_STATES, MAX_PAIRS)]

    scalar_mismatches: list[str] = []
    max_abs_prefix_error = 0.0
    max_abs_reference_error = 0.0
    for row in reviewed.itertuples(index=False):
        key = (int(row.repeat), int(row.states), int(row.pairs))
        expected_scalar = prefix_values[key]
        expected_reference = float(
            sum(
                value
                for repeat, value in max_values.items()
                if repeat != int(row.repeat)
            )
            / (REPEATS - 1)
        )
        prefix_error = abs(float(row.a_scalar) - expected_scalar)
        reference_error = abs(float(row.reference_scalar) - expected_reference)
        max_abs_prefix_error = max(max_abs_prefix_error, prefix_error)
        max_abs_reference_error = max(max_abs_reference_error, reference_error)
        prefix_tolerance = 2e-12 * max(1.0, abs(expected_scalar))
        reference_tolerance = 2e-12 * max(1.0, abs(expected_reference))
        if prefix_error > prefix_tolerance or reference_error > reference_tolerance:
            scalar_mismatches.append(str(key))
        if int(row.hvp_count) != 2 * int(row.states) * int(row.pairs):
            scalar_mismatches.append("hvp_count" + str(key))

    executor_error_delta = np.abs(
        reviewed["scalar_symmetric_error"].to_numpy(dtype=np.float64)
        - reviewed["reviewed_scalar_symmetric_error"].to_numpy(dtype=np.float64)
    )
    executor_norm_ratio_delta = np.abs(
        reviewed["gradient_norm_ratio"].to_numpy(dtype=np.float64)
        - reviewed["reviewed_gradient_norm_ratio"].to_numpy(dtype=np.float64)
    )
    ledger.add(
        "atomic_scalar_prefix_and_loro_recomputation",
        not scalar_mismatches,
        {
            "mismatches": scalar_mismatches[:12],
            "max_abs_prefix_error": max_abs_prefix_error,
            "max_abs_reference_error": max_abs_reference_error,
        },
    )
    ledger.add(
        "executor_scalar_error_matches_exact_no_epsilon_formula",
        bool(np.equal(executor_error_delta, 0.0).all()),
        {"max_abs_delta": float(executor_error_delta.max(initial=0.0))},
        blocking=False,
    )
    ledger.add(
        "global_norm_ratios_recomputed_without_epsilon",
        True,
        {
            "max_abs_executor_delta": float(executor_norm_ratio_delta.max(initial=0.0)),
            "minimum_candidate_norm": float(candidate_norms.min()),
            "minimum_reference_norm": float(reference_norms.min()),
        },
    )
    return reviewed


def _review_reference_splits(
    split: pd.DataFrame,
    reviewed_replicate: pd.DataFrame,
    ledger: ValidationLedger,
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    max_scalars = {
        int(row.repeat): float(row.a_scalar)
        for row in reviewed_replicate.loc[
            (reviewed_replicate["states"] == MAX_STATES)
            & (reviewed_replicate["pairs"] == MAX_PAIRS)
        ].itertuples(index=False)
    }
    split_scalar_mismatches: list[int] = []
    for split_index, (left_expected, right_expected) in enumerate(REFERENCE_SPLITS):
        source = split.loc[split["split"] == split_index].iloc[0]
        try:
            left = tuple(
                int(value) for value in json.loads(str(source["left_repeats"]))
            )
            right = tuple(
                int(value) for value in json.loads(str(source["right_repeats"]))
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ReviewError(
                f"reference split {split_index} has invalid repeat JSON"
            ) from error
        if left != left_expected or right != right_expected:
            raise ReviewError(
                f"reference split {split_index} mismatch: observed=({left},{right}) expected=({left_expected},{right_expected})"
            )
        expected_left_scalar = float(np.mean([max_scalars[repeat] for repeat in left]))
        expected_right_scalar = float(
            np.mean([max_scalars[repeat] for repeat in right])
        )
        left_tolerance = 2e-12 * max(1.0, abs(expected_left_scalar))
        right_tolerance = 2e-12 * max(1.0, abs(expected_right_scalar))
        if (
            abs(float(source["scalar_left"]) - expected_left_scalar) > left_tolerance
            or abs(float(source["scalar_right"]) - expected_right_scalar)
            > right_tolerance
        ):
            split_scalar_mismatches.append(split_index)
        scalar_error = exact_symmetric_scalar_error(
            source["scalar_left"], source["scalar_right"]
        )
        cosine_pass = float(source["gradient_cosine"]) >= 0.99
        norm_ratio_pass = 0.95 <= float(source["gradient_norm_ratio"]) <= 1.05
        scalar_pass = scalar_error <= 0.05
        rows.append(
            {
                "split": split_index,
                "protocol_split": split_index + 1,
                "left_repeats": json.dumps(list(left)),
                "right_repeats": json.dumps(list(right)),
                "gradient_cosine": float(source["gradient_cosine"]),
                "gradient_norm_ratio": float(source["gradient_norm_ratio"]),
                "scalar_left": float(source["scalar_left"]),
                "scalar_right": float(source["scalar_right"]),
                "scalar_symmetric_error_exact": scalar_error,
                "gradient_cosine_gate_pass": cosine_pass,
                "gradient_norm_ratio_gate_pass": norm_ratio_pass,
                "scalar_error_gate_pass": scalar_pass,
                "split_reference_pass": bool(
                    cosine_pass and norm_ratio_pass and scalar_pass
                ),
            }
        )
    reviewed = pd.DataFrame(rows)
    ledger.add(
        "reference_split_scalar_means_recomputed",
        not split_scalar_mismatches,
        {"mismatched_splits": split_scalar_mismatches},
    )
    reference_valid = bool(reviewed["split_reference_pass"].all())
    reviewed["reference_verdict"] = "pass" if reference_valid else "fail"
    return reviewed, reference_valid


def _review_modules(
    module_metrics: pd.DataFrame,
    ledger: ValidationLedger,
) -> tuple[pd.DataFrame, dict[tuple[int, int], bool]]:
    energy_by_module: dict[str, float] = {}
    constant_energy = True
    for module_name, group in module_metrics.groupby("module", sort=False):
        values = group["reference_energy_fraction"].to_numpy(dtype=np.float64)
        constant_energy = constant_energy and bool(np.equal(values, values[0]).all())
        energy_by_module[str(module_name)] = float(values[0])
    energy_sum = float(sum(energy_by_module.values()))
    energy_valid = (
        constant_energy
        and set(energy_by_module) == set(MODULES)
        and all(0.0 <= value <= 1.0 for value in energy_by_module.values())
        and math.isclose(energy_sum, 1.0, rel_tol=1e-8, abs_tol=1e-8)
    )
    ledger.add(
        "fixed_module_reference_energy_identity",
        energy_valid,
        {
            "shares": energy_by_module,
            "sum": energy_sum,
            "constant_across_rows": constant_energy,
        },
    )

    rows: list[dict[str, Any]] = []
    cell_module_pass: dict[tuple[int, int], bool] = {}
    for states_count in GRID:
        for pairs_count in GRID:
            eligible_results: list[bool] = []
            cell_rows = module_metrics.loc[
                (module_metrics["states"] == states_count)
                & (module_metrics["pairs"] == pairs_count)
            ]
            for module_name in MODULES:
                group = cell_rows.loc[cell_rows["module"] == module_name]
                cosine = group["gradient_cosine_to_reference"].to_numpy(
                    dtype=np.float64
                )
                norm_ratio = group["gradient_norm_ratio"].to_numpy(dtype=np.float64)
                cosine_median = linear_quantile(cosine, 0.5)
                cosine_q10 = linear_quantile(cosine, 0.1)
                median_pass = cosine_median >= 0.90
                q10_pass = cosine_q10 >= 0.80
                raw_pass = bool(median_pass and q10_pass)
                eligible = energy_by_module[module_name] >= 0.01
                if eligible:
                    eligible_results.append(raw_pass)
                rows.append(
                    {
                        "states": states_count,
                        "pairs": pairs_count,
                        "module": module_name,
                        "reference_energy_fraction": energy_by_module[module_name],
                        "eligible": eligible,
                        "gradient_cosine_median": cosine_median,
                        "gradient_cosine_q10": cosine_q10,
                        "gradient_cosine_min": float(cosine.min()),
                        "gradient_norm_ratio_median": linear_quantile(norm_ratio, 0.5),
                        "gradient_norm_ratio_q10": linear_quantile(norm_ratio, 0.1),
                        "gradient_norm_ratio_q90": linear_quantile(norm_ratio, 0.9),
                        "cosine_median_gate_pass": median_pass,
                        "cosine_q10_gate_pass": q10_pass,
                        "raw_module_thresholds_pass": raw_pass,
                    }
                )
            if not eligible_results:
                raise ReviewError(
                    "no module meets the frozen one-percent reference-energy eligibility threshold"
                )
            cell_module_pass[(states_count, pairs_count)] = bool(all(eligible_results))
    return pd.DataFrame(rows), cell_module_pass


def _review_cell_thresholds(
    replicate: pd.DataFrame,
    pairwise: pd.DataFrame,
    cell_module_pass: Mapping[tuple[int, int], bool],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for states_count in GRID:
        for pairs_count in GRID:
            group = replicate.loc[
                (replicate["states"] == states_count)
                & (replicate["pairs"] == pairs_count)
            ]
            pair_group = pairwise.loc[
                (pairwise["states"] == states_count)
                & (pairwise["pairs"] == pairs_count)
            ]
            scalar_errors = group["reviewed_scalar_symmetric_error"].to_numpy(
                dtype=np.float64
            )
            cosines = group["gradient_cosine_to_reference"].to_numpy(dtype=np.float64)
            relative_errors = group["gradient_relative_error"].to_numpy(
                dtype=np.float64
            )
            norm_ratios = group["reviewed_gradient_norm_ratio"].to_numpy(
                dtype=np.float64
            )
            pair_cosines = pair_group["gradient_cosine"].to_numpy(dtype=np.float64)

            row: dict[str, Any] = {
                "states": states_count,
                "pairs": pairs_count,
                "hvp_count": 2 * states_count * pairs_count,
                "scalar_mean": float(group["a_scalar"].mean()),
                "scalar_std": float(group["a_scalar"].std(ddof=1)),
                "scalar_symmetric_error_median": linear_quantile(scalar_errors, 0.5),
                "scalar_symmetric_error_q90": linear_quantile(scalar_errors, 0.9),
                "gradient_cosine_median": linear_quantile(cosines, 0.5),
                "gradient_cosine_q10": linear_quantile(cosines, 0.1),
                "gradient_cosine_min": float(cosines.min()),
                "gradient_relative_error_median": linear_quantile(relative_errors, 0.5),
                "gradient_relative_error_q90": linear_quantile(relative_errors, 0.9),
                "gradient_norm_ratio_median": linear_quantile(norm_ratios, 0.5),
                "gradient_norm_ratio_q10": linear_quantile(norm_ratios, 0.1),
                "gradient_norm_ratio_q90": linear_quantile(norm_ratios, 0.9),
                "pairwise_cosine_median": linear_quantile(pair_cosines, 0.5),
                "pairwise_cosine_q10": linear_quantile(pair_cosines, 0.1),
                "pairwise_cosine_min": float(pair_cosines.min()),
                "pairwise_negative_fraction": float(np.mean(pair_cosines < 0.0)),
            }
            row["scalar_error_median_gate_pass"] = (
                row["scalar_symmetric_error_median"] <= 0.10
            )
            row["scalar_error_q90_gate_pass"] = (
                row["scalar_symmetric_error_q90"] <= 0.20
            )
            row["global_cosine_median_gate_pass"] = (
                row["gradient_cosine_median"] >= 0.95
            )
            row["global_cosine_q10_gate_pass"] = row["gradient_cosine_q10"] >= 0.90
            row["pairwise_cosine_median_gate_pass"] = (
                row["pairwise_cosine_median"] >= 0.90
            )
            row["pairwise_cosine_q10_gate_pass"] = row["pairwise_cosine_q10"] >= 0.80
            row["norm_ratio_median_gate_pass"] = (
                0.90 <= row["gradient_norm_ratio_median"] <= 1.10
            )
            row["norm_ratio_q10_gate_pass"] = row["gradient_norm_ratio_q10"] >= 0.75
            row["norm_ratio_q90_gate_pass"] = row["gradient_norm_ratio_q90"] <= 1.33
            row["raw_scalar_thresholds_pass"] = bool(
                row["scalar_error_median_gate_pass"]
                and row["scalar_error_q90_gate_pass"]
            )
            row["raw_global_gradient_thresholds_pass"] = bool(
                row["global_cosine_median_gate_pass"]
                and row["global_cosine_q10_gate_pass"]
                and row["pairwise_cosine_median_gate_pass"]
                and row["pairwise_cosine_q10_gate_pass"]
                and row["norm_ratio_median_gate_pass"]
                and row["norm_ratio_q10_gate_pass"]
                and row["norm_ratio_q90_gate_pass"]
            )
            row["raw_module_thresholds_pass"] = bool(
                cell_module_pass[(states_count, pairs_count)]
            )
            row["raw_cell_thresholds_pass"] = bool(
                row["raw_scalar_thresholds_pass"]
                and row["raw_global_gradient_thresholds_pass"]
                and row["raw_module_thresholds_pass"]
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["states", "pairs"]).reset_index(drop=True)


def apply_reference_gated_decisions(
    summary: pd.DataFrame,
    *,
    reference_valid: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reviewed = summary.copy().sort_values(["states", "pairs"]).reset_index(drop=True)
    _require_columns(
        reviewed,
        ("states", "pairs", "hvp_count", "raw_cell_thresholds_pass"),
        "reviewed summary",
    )
    key_map = {
        (int(row.states), int(row.pairs)): bool(row.raw_cell_thresholds_pass)
        for row in reviewed.itertuples(index=False)
    }

    if not reference_valid:
        reviewed["cell_verdict"] = "not_evaluable"
        reviewed["state_doubling_verdict"] = "not_evaluable"
        reviewed["pair_doubling_verdict"] = "not_evaluable"
        reviewed["axial_persistence_verdict"] = "not_evaluable"
        selected: dict[str, int] | None = None
        budget_censored = False
        requires_reference_extension = True
        requires_budget_extension = False
    else:
        cell_verdicts: list[str] = []
        state_verdicts: list[str] = []
        pair_verdicts: list[str] = []
        axial_verdicts: list[str] = []
        for row in reviewed.itertuples(index=False):
            key = (int(row.states), int(row.pairs))
            own_pass = key_map[key]
            state_key = (2 * key[0], key[1])
            pair_key = (key[0], 2 * key[1])
            state_pass = key_map.get(state_key)
            pair_pass = key_map.get(pair_key)
            cell_verdicts.append("pass" if own_pass else "fail")
            state_verdicts.append(
                "outside_grid"
                if state_pass is None
                else ("pass" if state_pass else "fail")
            )
            pair_verdicts.append(
                "outside_grid"
                if pair_pass is None
                else ("pass" if pair_pass else "fail")
            )
            if not own_pass or state_pass is False or pair_pass is False:
                axial_verdicts.append("fail")
            elif state_pass is None or pair_pass is None:
                axial_verdicts.append("not_confirmable_boundary")
            else:
                axial_verdicts.append("pass")
        reviewed["cell_verdict"] = cell_verdicts
        reviewed["state_doubling_verdict"] = state_verdicts
        reviewed["pair_doubling_verdict"] = pair_verdicts
        reviewed["axial_persistence_verdict"] = axial_verdicts
        persistent = reviewed.loc[
            reviewed["axial_persistence_verdict"] == "pass"
        ].sort_values(["hvp_count", "states", "pairs"])
        selected = None
        if not persistent.empty:
            first = persistent.iloc[0]
            selected = {
                "states": int(first["states"]),
                "pairs": int(first["pairs"]),
                "hvp_count": int(first["hvp_count"]),
            }
        budget_censored = selected is None
        requires_reference_extension = False
        requires_budget_extension = budget_censored

    current = reviewed.loc[(reviewed["states"] == 2) & (reviewed["pairs"] == 4)]
    if len(current) != 1:
        raise ReviewError("reviewed grid must contain exactly one current S=2,P=4 cell")
    current_payload = _json_native(current.iloc[0].to_dict())
    cells = [
        {
            "states": int(row.states),
            "pairs": int(row.pairs),
            "cell_verdict": str(row.cell_verdict),
            "state_doubling_verdict": str(row.state_doubling_verdict),
            "pair_doubling_verdict": str(row.pair_doubling_verdict),
            "axial_persistence_verdict": str(row.axial_persistence_verdict),
        }
        for row in reviewed.itertuples(index=False)
    ]
    decision = {
        "protocol_id": PROTOCOL_ID,
        "reference_valid": bool(reference_valid),
        "selected_axially_confirmed_cell": selected,
        "budget_censored": bool(budget_censored),
        "requires_reference_extension": bool(requires_reference_extension),
        "requires_budget_extension": bool(requires_budget_extension),
        "current_2x4": current_payload,
        "cells": cells,
        "evidence_boundary": "checkpoint-local finite-budget estimator stability only",
    }
    return reviewed, decision


def _compare_executor_summary(
    reviewed: pd.DataFrame,
    executor_summary: pd.DataFrame,
    ledger: ValidationLedger,
) -> None:
    mappings = {
        "scalar_symmetric_error_median": "scalar_symmetric_error_median",
        "scalar_symmetric_error_q90": "scalar_symmetric_error_q90",
        "gradient_cosine_median": "gradient_cosine_median",
        "gradient_cosine_q10": "gradient_cosine_q10",
        "pairwise_cosine_median": "pairwise_cosine_median",
        "pairwise_cosine_q10": "pairwise_cosine_q10",
        "gradient_norm_ratio_median": "gradient_norm_ratio_median",
        "gradient_norm_ratio_q10": "gradient_norm_ratio_q10",
        "gradient_norm_ratio_q90": "gradient_norm_ratio_q90",
    }
    merged = reviewed.merge(
        executor_summary,
        on=["states", "pairs"],
        suffixes=("_reviewed", "_executor"),
        validate="one_to_one",
    )
    maximum_deltas: dict[str, float] = {}
    for reviewed_name, executor_name in mappings.items():
        left_name = reviewed_name + "_reviewed"
        right_name = executor_name + "_executor"
        maximum_deltas[reviewed_name] = float(
            np.max(
                np.abs(
                    merged[left_name].to_numpy(dtype=np.float64)
                    - merged[right_name].to_numpy(dtype=np.float64)
                )
            )
        )
    exact_match = all(delta == 0.0 for delta in maximum_deltas.values())
    ledger.add(
        "executor_summary_matches_independent_recomputation",
        exact_match,
        {"maximum_absolute_deltas": maximum_deltas, "authoritative": "reviewed tables"},
        blocking=False,
    )


def _protocol_erratum_text() -> str:
    return """# Protocol erratum: `patch_decoder.3.bias`

The frozen protocol describes the active direct-decoder prefixes as including
their weights and biases. For the `stopped_composite` HVP used by this run,
`patch_decoder.3.bias` is a structural zero and has no A-gradient graph.

The final affine bias is additive in the decoder output, so it disappears from
the decoder Jacobian and decoder second derivatives with respect to latent
coordinates. In the stopped-composite construction, the decoded point used to
form the task-space gradient/Hessian is detached. The remaining HVP graph
therefore does not restore dependence on this final bias. The executor's
preflight consequently observed `grad is None` for this parameter.

The post-run review treats this one parameter explicitly as excluded and
structurally zero. The measured active vector must contain the other 31
parameters in its recorded order, and `patch_decoder.3.bias` must appear in the
excluded list. It is not silently counted as active, and no threshold is
changed by this clarification.

This erratum is limited to estimator-review validity. It makes no downstream or
causal claim.
"""


def review(input_dir: Path, *, verbose: bool = True) -> dict[str, Path]:
    input_dir = input_dir.expanduser().resolve()
    reporter = Reporter(verbose)
    reporter.stage(
        f"start input_dir={input_dir} protocol_id={PROTOCOL_ID} seed={BASE_SEED}"
    )
    reporter.stage("stage=load_completed_production_artifacts")
    payloads, frames = _load_inputs(input_dir)
    reporter.stage(
        "resolved "
        f"device={payloads['resolved_config'].get('device')} dtype=float32 "
        f"repeats={REPEATS} states={list(GRID)} pairs={list(GRID)} "
        f"output_dir={input_dir}"
    )
    ledger = ValidationLedger()

    reporter.stage("stage=strict_schema_key_and_finiteness_checks")
    _validate_frame_contracts(frames, ledger)
    reporter.stage("stage=provenance_config_and_active_set_checks")
    train_indices = _validate_provenance(input_dir, payloads, frames, ledger)
    ledger.require_passed()

    reporter.stage("stage=state_probe_and_batch_replay")
    _validate_state_replay(frames["states"], train_indices, ledger)
    _validate_branch_replay(
        frames["states"],
        frames["branches"],
        frames["atomic"],
        ledger,
        probe_device=str(payloads["resolved_config"]["device"]),
        reporter=reporter,
    )
    ledger.require_passed()

    reporter.stage("stage=independent_scalar_reference_and_threshold_recomputation")
    reviewed_replicate = _recompute_scalar_replicates(
        frames["atomic"], frames["replicate"], ledger
    )
    reviewed_reference, reference_valid = _review_reference_splits(
        frames["split"], reviewed_replicate, ledger
    )
    reviewed_modules, cell_module_pass = _review_modules(frames["module"], ledger)
    summary = _review_cell_thresholds(
        reviewed_replicate, frames["pairwise"], cell_module_pass
    )
    summary, decision = apply_reference_gated_decisions(
        summary, reference_valid=reference_valid
    )
    reviewed_modules["module_gate_verdict"] = np.where(
        ~reviewed_modules["eligible"],
        "not_gated",
        np.where(
            not reference_valid,
            "not_evaluable",
            np.where(reviewed_modules["raw_module_thresholds_pass"], "pass", "fail"),
        ),
    )
    _compare_executor_summary(summary, frames["executor_summary"], ledger)
    ledger.require_passed()

    decision["posthoc_validity_passed"] = ledger.passed
    decision["reference_split_verdict"] = "pass" if reference_valid else "fail"
    validity_payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "valid" if ledger.passed else "invalid",
        "passed": ledger.passed,
        "checks": ledger.checks,
        "counts": {
            "repeats": REPEATS,
            "grid_cells": len(GRID) * len(GRID),
            "atomic_rows": len(frames["atomic"]),
            "branch_rows": len(frames["branches"]),
            "replicate_rows": len(frames["replicate"]),
            "pairwise_rows": len(frames["pairwise"]),
            "module_rows": len(frames["module"]),
            "reference_split_rows": len(frames["split"]),
        },
        "reference_valid": reference_valid,
        "authoritative_outputs": [
            "reviewed_stability_summary.csv",
            "reviewed_module_summary.csv",
            "reviewed_reference_checks.csv",
            "reviewed_decision.json",
        ],
        "non_authoritative_executor_outputs": [
            "stability_summary.csv",
            "reference_split_checks.csv",
            "decision.json",
        ],
        "evidence_boundary": "checkpoint-local finite-budget estimator stability only",
    }

    outputs = {
        "stability": input_dir / "reviewed_stability_summary.csv",
        "modules": input_dir / "reviewed_module_summary.csv",
        "reference": input_dir / "reviewed_reference_checks.csv",
        "validity": input_dir / "posthoc_validity.json",
        "decision": input_dir / "reviewed_decision.json",
        "erratum": input_dir / "protocol_erratum.md",
    }
    reporter.stage("stage=write_reviewed_artifacts")
    _write_csv(outputs["stability"], summary)
    _write_csv(outputs["modules"], reviewed_modules)
    _write_csv(outputs["reference"], reviewed_reference)
    _write_json(outputs["validity"], validity_payload)
    _write_json(outputs["decision"], decision)
    _write_text(outputs["erratum"], _protocol_erratum_text())
    reporter.stage(
        "done "
        f"reference_valid={reference_valid} budget_censored={decision['budget_censored']} "
        + " outputs="
        + ",".join(str(path) for path in outputs.values())
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently review the completed Variant A estimator-stability run."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    review(args.input_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
