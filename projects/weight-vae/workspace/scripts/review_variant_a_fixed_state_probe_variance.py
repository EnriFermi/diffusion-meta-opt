from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_fixed_state_probe_variance_h2048"
)
DEFAULT_INPUT_DIR = PROTOCOL_DIR / "production_run"
DEFAULT_RUN_DIR = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)

PROTOCOL_ID = "a_fixed_state_probe_variance_h2048_v1"
BASE_SEED = 20260714
STATE_COUNT = 44
PRIMARY_COUNT = 32
SENTINEL_COUNT = 12
DRAW_COUNT = 64
PAIR_COUNT = 4
PREFIXES = (1, 2, 4)
TASKS = ("fashion_mnist", "mnist")
TASK_LABELS = {"fashion_mnist": "FashionMNIST", "mnist": "MNIST"}
BRIDGE_POSITIONS = (0, 4, 8, 12, 16, 20, 24, 28)
BRIDGE_SOURCE_IDS = (7589, 922, 13104, 8769, 9439, 2396, 191, 4016)
BOOTSTRAP_DRAWS = 2000
TRAIN_COUNT = 16384
EXPECTED_ACTIVE_COUNT = 11_685_120
EXPECTED_CONFIG_HASH = "4fc87a54349a39a2"
EXPECTED_RUNTIME_CONFIG_HASH = "d1d072b2e65ad8a7"
EXPECTED_EXECUTOR_SHA256 = (
    "b04f7de881420bfe2496f170125983f9f3fb78889a3fb6171a56583cee170af0"
)
INTERPRETATION_BOUNDARY = (
    "bank-conditional raw Variant A estimator diagnostics only; no downstream, "
    "training-update, finite-B population, sentinel-prevalence, or broad Li/PSGD/Kron claim"
)

SPLITS = (
    (tuple(range(0, 32)), tuple(range(32, 64))),
    (tuple(range(0, 64, 2)), tuple(range(1, 64, 2))),
    (
        tuple([*range(0, 16), *range(32, 48)]),
        tuple([*range(16, 32), *range(48, 64)]),
    ),
)
FOLDS = tuple(tuple(range(start, start + 16)) for start in range(0, 64, 16))

EXPECTED_SOURCE_SHA256 = {
    "checkpoint": "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397",
    "weight_pool": "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef",
    "records": "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933",
    "config_json": "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4",
    "acceptance": "217bfbda8b23af770c94d982375a088eaa84b51e36e12f026edcc4b4798f56f9",
    "protocol": "5e6a9605ecf3565c595f2c1c3cf2cf97f503bc8ae7e4f1698642151684b9f429",
    "state_bank": "ba122d5cdbff96a6000a19b4963ba6ec60de1310c0eb3ab6cd1d3150e5018d3b",
    "excluded_primary_sources": "60e7e2ef3bc21069b069bc1986034fd6864cd0c301c3e727cb4b9329547e3ebe",
    "eligible_runs": "1572fa329bc129d5a7e02d75b68f077d6b5a502b76eee1a996daf6bc573c96ae",
    "sentinel_parent_states": "ea229ec70c42f76c874a1b69e22a9ff422f7639f1ad1910a581aeb004fd49a34",
    "state_bank_builder": "d4890635796202e66c2e1d6c51a3cd61211ce1c6d835730009e0fa7ac571f407",
    "accepted_active_parameters": "eeaac7cd89af34ab7256d50370bccbb793cd70851000c5391aa0bd201577ebaa",
    "sentinel_atomic_parent": "74790a22d6010abaf73a375d41ee37ff96d24d381d9320be09dcd78c6208e10e",
    "config_py": "aa71d2909f4ac7cde9b7f6fdb2d0caff8bfd3795f2fdea94c1be3e2fe853675e",
    "core_py": "ff67cc37006f5678d67ebaa919d348161d0b030beae95c1e3701ec07da35a43a",
    "preconditioning_py": "f63e33a7e68bedf066f77ee8a7343e890690379119e19b2edc8e17eb65aa7eea",
    "estimator_helper": "1f0ad1f4d146e9f41981bf7192192cd58ecd78ab23dd824529c88807befdd8ab",
    "hvp_helper": "3543bc9a54aa07d57aa160f5ebe588379b52b377d5bbd801760a2dbfabcef622",
    "state_pair_helper": "35f7ce22f5cc66f3b30de8f7fea1ae438134b812397d95385a911c92aae69683",
}

EXPECTED_FROZEN_SHA256 = {
    "executor": EXPECTED_EXECUTOR_SHA256,
    "tests": "144b292c660fe7581d3a76d7822118d5f54969fe13455fa0299a28c9e15abc47",
    "protocol": EXPECTED_SOURCE_SHA256["protocol"],
    "state_bank_builder": EXPECTED_SOURCE_SHA256["state_bank_builder"],
    "state_bank": EXPECTED_SOURCE_SHA256["state_bank"],
    "eligible_runs": EXPECTED_SOURCE_SHA256["eligible_runs"],
    "excluded_primary_sources": EXPECTED_SOURCE_SHA256["excluded_primary_sources"],
    "sentinel_parent_states": EXPECTED_SOURCE_SHA256["sentinel_parent_states"],
    "preflight_metrics": "4b7c2e36110048fed479d1941630ff74f7b9b100e0cb1fa4e754246bf16fbc78",
    "preflight_resolved_config": "b31ebbfacc4a1f9e8c1fea40df0b2a1408e1e003c16135fcc345ddc7074beee8",
    "preflight_manifest": "6ff4b7fba8373b8d89166de64125eb0e3a96bb3f3d29877524118e1a56bdbcb0",
}

EXPECTED_EXECUTOR_VALIDITY_CHECKS = {
    "source_hashes_match",
    "state_bank_gate_passed",
    "model_hash_before_match",
    "model_hash_after_unchanged",
    "latent_hash_count",
    "active_manifest_match",
    "atomic_cardinality",
    "atomic_keys_unique",
    "all_full_ce",
    "all_train_count_16384",
    "bridge_cardinality_and_batch_replay",
    "bridge_paired_key_cardinality",
    "probe_key_cardinality",
    "probe_hash_common_across_states",
    "probe_hashes_unique_across_keys",
    "all_numeric_metadata_finite",
    "all_grams_finite",
    "nonalignment_gates_finite",
    "p4_materialized_preflight",
    "no_model_update",
}

FRAME_FILES = {
    "active": "active_parameters.csv",
    "atomic": "atomic_gradient_metadata.csv",
    "bridge_atomic": "bridge_atomic_gradient_metadata.csv",
    "bridge": "bridge_paired_p4_metrics.csv",
    "prefix_scalar": "prefix_scalar_cells.csv",
    "state_summary": "state_prefix_summary.csv",
    "signal": "signal_fraction.csv",
    "crossfit": "crossfit_state_mean_cosines.csv",
    "nonalignment": "nonalignment_gates.csv",
    "task_crossfit": "fixed_task_panel_crossfit_cosines.csv",
    "panel": "panel_state_metrics.csv",
    "sentinel_stratum": "sentinel_stratum_summary.csv",
    "sentinel_persistence": "sentinel_persistence.csv",
    "anova": "crossed_anova_sufficient.csv",
    "delta": "delta_components.csv",
    "primary_bootstrap": "primary_fixed_panel_probe_bootstrap.csv",
    "panel_bootstrap": "panel_sensitivity_bootstrap.csv",
}

ARCHIVE_FILES = {
    "state_grams": "state_prefix_grams.npz",
    "task_grams": "primary_task_p4_grams.npz",
    "cross_task": "cross_task_probe_sum_grams.npz",
    "reconstruction_cross": "reconstruction_cross_grams.npz",
    "bridge_grams": "bridge_p4_grams.npz",
}

EXPECTED_EXECUTOR_OUTPUTS = {
    "manifest.json",
    "resolved_config.json",
    "validity.json",
    "preflight.json",
    "decision.json",
    "active_parameters.csv",
    "state_prefix_grams.npz",
    "state_module_p4_grams.npz",
    "primary_task_p4_grams.npz",
    "reconstruction_cross_grams.npz",
    "cross_task_probe_sum_grams.npz",
    "bridge_p4_grams.npz",
    "atomic_gradient_metadata.csv",
    "bridge_atomic_gradient_metadata.csv",
    "bridge_paired_p4_metrics.csv",
    "prefix_scalar_cells.csv",
    "state_prefix_summary.csv",
    "draw_pairwise_metrics.csv",
    "fold_split_mean_metrics.csv",
    "signal_fraction.csv",
    "module_p4_summary.csv",
    "reconstructed_s2_diagnostics.csv",
    "crossfit_state_mean_cosines.csv",
    "nonalignment_gates.csv",
    "fixed_task_panel_crossfit_cosines.csv",
    "panel_state_metrics.csv",
    "sentinel_stratum_summary.csv",
    "sentinel_persistence.csv",
    "crossed_anova_sufficient.csv",
    "delta_components.csv",
    "sensitivity_summary.csv",
    "primary_fixed_panel_probe_bootstrap.csv",
    "panel_sensitivity_bootstrap.csv",
}

GENERATED_FILES = (
    "primary_state_fixed_probe_stability_by_task.png",
    "directional_state_vs_probe_variance.png",
    "prefix_and_bridge_diagnostics.png",
    "result.md",
    "posthoc_validation.json",
)


class ReviewError(ValueError):
    """Raised when the production packet cannot support a fail-closed review."""


@dataclass(slots=True)
class ValidationLedger:
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, name: str, detail: Any = None, *, blocking: bool = True) -> None:
        self.checks[name] = {
            "passed": True,
            "blocking": bool(blocking),
            "detail": _json_native(detail),
        }

    @property
    def passed(self) -> bool:
        return all(
            bool(item["passed"])
            for item in self.checks.values()
            if bool(item["blocking"])
        )


class Reporter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def stage(self, message: str) -> None:
        if self.enabled:
            print(f"[fixed_state_probe_review] {message}", flush=True)

    def artifact(self, path: Path) -> None:
        if self.enabled:
            print(f"[fixed_state_probe_review] artifact={path}", flush=True)


@dataclass(frozen=True, slots=True)
class DeltaEstimate:
    v_probe: float
    v_state: float
    delta: float
    state_interaction_ms: float


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ReviewError(f"nonfinite value cannot be serialized: {result}")
        return result
    if value is pd.NA:
        return None
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(_json_native(payload), indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, content)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReviewError(f"required JSON artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ReviewError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ReviewError(f"JSON artifact must contain an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uint63(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) & ((1 << 63) - 1)


def _require_columns(
    frame: pd.DataFrame, required: Sequence[str], table_name: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ReviewError(f"{table_name} is missing columns: {missing}")


def require_finite(
    frame: pd.DataFrame, columns: Sequence[str], table_name: str
) -> None:
    _require_columns(frame, columns, table_name)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            preview = ", ".join(str(int(index)) for index in bad[:8])
            raise ReviewError(
                f"{table_name}.{column} has NaN/inf/nonnumeric values at rows {preview}"
            )


def _coerce_integer_columns(
    frame: pd.DataFrame, columns: Sequence[str], table_name: str
) -> pd.DataFrame:
    result = frame.copy()
    require_finite(result, columns, table_name)
    for column in columns:
        values = pd.to_numeric(result[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.equal(values, np.trunc(values)).all():
            raise ReviewError(f"{table_name}.{column} contains non-integer values")
        result[column] = values.astype(np.int64)
    return result


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ReviewError(f"{label} is not a strict boolean: {value!r}")


def require_exact_key_grid(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    expected_keys: Iterable[tuple[Any, ...]],
    table_name: str,
) -> dict[str, int]:
    """Fail closed on missing, extra, or duplicate rows in a frozen key grid."""
    _require_columns(frame, key_columns, table_name)
    expected = set(expected_keys)
    if frame.duplicated(list(key_columns)).any():
        duplicate = (
            frame.loc[
                frame.duplicated(list(key_columns), keep=False), list(key_columns)
            ]
            .head(5)
            .to_dict("records")
        )
        raise ReviewError(f"{table_name} has duplicate keys: {duplicate}")
    observed = set(frame.loc[:, list(key_columns)].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected.difference(observed), key=str)[:5]
        extra = sorted(observed.difference(expected), key=str)[:5]
        raise ReviewError(
            f"{table_name} key grid mismatch: missing={missing}, extra={extra}"
        )
    if len(frame) != len(expected):
        raise ReviewError(
            f"{table_name} row count is {len(frame)}, expected {len(expected)}"
        )
    return {"rows": len(frame), "unique_keys": len(observed)}


def _assert_close(
    observed: float,
    expected: float,
    label: str,
    *,
    rtol: float = 2e-10,
    atol: float = 2e-10,
) -> float:
    left = float(observed)
    right = float(expected)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ReviewError(f"{label} requires finite values, got {left}, {right}")
    delta = abs(left - right)
    if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
        raise ReviewError(
            f"{label} mismatch: observed={left:.17g}, expected={right:.17g}, delta={delta:.3g}"
        )
    return delta


def _compare_frames(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    numeric_columns: Sequence[str],
    table_name: str,
    rtol: float = 2e-10,
    atol: float = 2e-10,
) -> dict[str, float]:
    merged = observed.merge(
        expected,
        on=list(keys),
        how="outer",
        suffixes=("_observed", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        bad = merged.loc[merged["_merge"] != "both", [*keys, "_merge"]]
        raise ReviewError(f"{table_name} comparison key mismatch: {bad.head().to_dict('records')}")
    maximum: dict[str, float] = {}
    for column in numeric_columns:
        left = pd.to_numeric(
            merged[f"{column}_observed"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        right = pd.to_numeric(
            merged[f"{column}_expected"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite_pair = np.isfinite(left) & np.isfinite(right)
        both_nan = np.isnan(left) & np.isnan(right)
        if not bool((finite_pair | both_nan).all()):
            raise ReviewError(f"{table_name}.{column} has incompatible nonfinite values")
        delta = np.abs(left[finite_pair] - right[finite_pair])
        max_delta = float(delta.max(initial=0.0))
        maximum[column] = max_delta
        if not np.allclose(
            left[finite_pair], right[finite_pair], rtol=rtol, atol=atol
        ):
            index = int(np.flatnonzero(~np.isclose(
                left[finite_pair], right[finite_pair], rtol=rtol, atol=atol
            ))[0])
            raise ReviewError(
                f"{table_name}.{column} mismatch: observed={left[finite_pair][index]}, "
                f"expected={right[finite_pair][index]}"
            )
    return maximum


def _compare_json(
    observed: Any, expected: Any, *, label: str, path: str = "root"
) -> float:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            observed_keys = sorted(observed) if isinstance(observed, Mapping) else []
            raise ReviewError(
                f"{label}.{path} key mismatch: observed={observed_keys}, expected={sorted(expected)}"
            )
        return max(
            (
                _compare_json(
                    observed[key], expected[key], label=label, path=f"{path}.{key}"
                )
                for key in expected
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ReviewError(f"{label}.{path} list mismatch")
        return max(
            (
                _compare_json(left, right, label=label, path=f"{path}[{index}]")
                for index, (left, right) in enumerate(
                    zip(observed, expected, strict=True)
                )
            ),
            default=0.0,
        )
    if isinstance(expected, bool):
        if observed is not expected:
            raise ReviewError(
                f"{label}.{path} mismatch: observed={observed!r}, expected={expected!r}"
            )
        return 0.0
    if isinstance(expected, (float, np.floating)):
        return _assert_close(float(observed), float(expected), f"{label}.{path}")
    if observed != expected:
        raise ReviewError(
            f"{label}.{path} mismatch: observed={observed!r}, expected={expected!r}"
        )
    return 0.0


def _source_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint": run_dir / "vae_checkpoint.pt",
        "weight_pool": run_dir / "weight_pool.pt",
        "records": run_dir / "weight_pool_records.csv",
        "config_json": run_dir / "config.json",
        "acceptance": run_dir / "baseline_acceptance.json",
        "protocol": PROTOCOL_DIR / "protocol.md",
        "state_bank": PROTOCOL_DIR / "state_bank.csv",
        "excluded_primary_sources": PROTOCOL_DIR / "excluded_primary_sources.csv",
        "eligible_runs": PROTOCOL_DIR / "eligible_runs.csv",
        "sentinel_parent_states": PROTOCOL_DIR / "sentinel_parent_states.csv",
        "state_bank_builder": ROOT / "scripts/build_variant_a_fixed_state_probe_bank.py",
        "accepted_active_parameters": (
            ROOT
            / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
            / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
        ),
        "sentinel_atomic_parent": (
            ROOT
            / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
            / "a_hvp_batch_size_ablation_h2048/atomic_pair_samples.csv"
        ),
        "config_py": (
            ROOT
            / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/config.py"
        ),
        "core_py": (
            ROOT
            / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py"
        ),
        "preconditioning_py": (
            ROOT
            / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py"
        ),
        "estimator_helper": ROOT / "scripts/audit_variant_a_estimator_stability.py",
        "hvp_helper": ROOT / "scripts/audit_variant_a_hvp_batch_size.py",
        "state_pair_helper": ROOT / "scripts/audit_variant_a_state_pair_variance.py",
    }


def _frozen_only_paths() -> dict[str, Path]:
    preflight = PROTOCOL_DIR / "preflight_run"
    return {
        "executor": ROOT / "scripts/audit_variant_a_fixed_state_probe_variance.py",
        "tests": ROOT / "tests/variant_a_fixed_state_probe_variance_test.py",
        "preflight_metrics": preflight / "preflight.json",
        "preflight_resolved_config": preflight / "resolved_config.json",
        "preflight_manifest": preflight / "manifest.json",
    }


def _validate_manifest_and_config(
    input_dir: Path,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    validity: Mapping[str, Any],
    decision: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("status") != "complete" or manifest.get("mode") == "preflight_only":
        raise ReviewError(f"production manifest is not complete: {manifest.get('status')!r}")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ReviewError("manifest protocol_id mismatch")
    if Path(str(manifest.get("output_dir", ""))).resolve() != input_dir:
        raise ReviewError("manifest output_dir does not identify the reviewed directory")
    if manifest.get("executor_sha256") != EXPECTED_EXECUTOR_SHA256:
        raise ReviewError("manifest executor SHA256 mismatch")
    if manifest.get("validity") != validity:
        raise ReviewError("manifest validity payload differs from validity.json")
    if manifest.get("decision") != decision:
        raise ReviewError("manifest decision payload differs from decision.json")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ReviewError("manifest outputs must be a list")
    missing_manifest_outputs = sorted(EXPECTED_EXECUTOR_OUTPUTS.difference(outputs))
    if missing_manifest_outputs:
        raise ReviewError(f"manifest omits production outputs: {missing_manifest_outputs}")
    missing_files = sorted(
        name for name in EXPECTED_EXECUTOR_OUTPUTS if not (input_dir / name).is_file()
    )
    if missing_files:
        raise ReviewError(f"production packet is missing outputs: {missing_files}")

    checks = validity.get("checks")
    if validity.get("passed") is not True or not isinstance(checks, dict):
        raise ReviewError("executor validity is not a passed check mapping")
    if set(checks) != EXPECTED_EXECUTOR_VALIDITY_CHECKS:
        raise ReviewError(
            "executor validity check set mismatch: "
            f"observed={sorted(checks)}, expected={sorted(EXPECTED_EXECUTOR_VALIDITY_CHECKS)}"
        )
    failed = sorted(name for name, value in checks.items() if value is not True)
    if failed:
        raise ReviewError(f"executor validity contains failed checks: {failed}")

    expected_resolved = {
        "protocol_id": PROTOCOL_ID,
        "seed": BASE_SEED,
        "device": "cuda:0",
        "dtype": "float32",
        "cache_mode": "read_only_checkpoint_pool_and_records",
        "states": STATE_COUNT,
        "primary_states": PRIMARY_COUNT,
        "sentinel_states": SENTINEL_COUNT,
        "draws": DRAW_COUNT,
        "pairs": PAIR_COUNT,
        "prefixes": list(PREFIXES),
        "full_ce_train_count": TRAIN_COUNT,
        "bridge_positions": list(BRIDGE_POSITIONS),
        "bridge_batch_size": 128,
        "bridge_step_keys": [10 * (draw + 1) for draw in range(16)],
        "estimator_scope": "local",
        "hvp_mode": "stopped_composite",
        "coefficient": 1.0,
        "loss_clip": 0.0,
        "gradient_damping": 0.0,
        "model_update": False,
        "saved_config_hash": EXPECTED_CONFIG_HASH,
        "runtime_config_hash_with_current_defaults": EXPECTED_RUNTIME_CONFIG_HASH,
        "executor_sha256": EXPECTED_EXECUTOR_SHA256,
        "active_parameter_count": EXPECTED_ACTIVE_COUNT,
    }
    for key, expected in expected_resolved.items():
        if resolved.get(key) != expected:
            raise ReviewError(
                f"resolved_config.{key} mismatch: observed={resolved.get(key)!r}, expected={expected!r}"
            )
    if Path(str(resolved.get("run_dir", ""))).resolve() != DEFAULT_RUN_DIR.resolve():
        raise ReviewError("resolved run_dir differs from the frozen accepted run")
    if Path(str(resolved.get("output_dir", ""))).resolve() != input_dir:
        raise ReviewError("resolved output_dir differs from the reviewed directory")
    expected_bank_gate = {
        "state_count": STATE_COUNT,
        "primary_count": PRIMARY_COUNT,
        "sentinel_count": SENTINEL_COUNT,
        "primary_unique_runs": PRIMARY_COUNT,
        "excluded_identity_count": 404,
        "bridge_positions": list(BRIDGE_POSITIONS),
        "bridge_source_ids": list(BRIDGE_SOURCE_IDS),
    }
    bank_gate = resolved.get("bank_gate")
    if not isinstance(bank_gate, Mapping):
        raise ReviewError("resolved bank_gate is missing")
    for key, expected in expected_bank_gate.items():
        if bank_gate.get(key) != expected:
            raise ReviewError(f"resolved bank_gate.{key} mismatch")

    full_preflight = preflight.get("full_ce_materialized_p4")
    if not isinstance(full_preflight, Mapping):
        raise ReviewError("preflight full_ce_materialized_p4 is missing")
    if (
        int(full_preflight.get("batch_size", -1)) != TRAIN_COUNT
        or int(full_preflight.get("pairs", -1)) != PAIR_COUNT
        or float(full_preflight.get("gradient_cosine", -math.inf)) < 0.99999
        or float(full_preflight.get("gradient_relative_error", math.inf)) > 5e-5
        or float(full_preflight.get("scalar_abs_error", math.inf)) > 5e-5
    ):
        raise ReviewError("materialized P=4 preflight thresholds failed post hoc")
    return {
        "manifest_status": manifest["status"],
        "executor_check_count": len(checks),
        "manifest_output_count": len(outputs),
        "resolved_device": resolved["device"],
        "resolved_dtype": resolved["dtype"],
    }


def _validate_provenance(
    resolved: Mapping[str, Any], reporter: Reporter
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    frozen = _read_json(PROTOCOL_DIR / "frozen_provenance.json")
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise ReviewError("frozen_provenance protocol_id mismatch")
    if frozen.get("sha256") != EXPECTED_FROZEN_SHA256:
        raise ReviewError("frozen_provenance SHA256 map differs from reviewer constants")
    if resolved.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ReviewError("resolved source_sha256 map differs from frozen reviewer constants")

    run_dir = Path(str(resolved["run_dir"])).resolve()
    paths = _source_paths(run_dir)
    details: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ReviewError(f"frozen source is missing: {name} -> {path}")
        reporter.stage(f"stage=hash_protocol_source name={name} path={path}")
        observed = sha256_file(path)
        expected = EXPECTED_SOURCE_SHA256[name]
        if observed != expected:
            raise ReviewError(
                f"frozen source hash mismatch for {name}: observed={observed}, expected={expected}"
            )
        details[name] = {
            "path": str(path),
            "sha256": observed,
            "size_bytes": int(path.stat().st_size),
        }

    for name, path in _frozen_only_paths().items():
        if not path.is_file():
            raise ReviewError(f"frozen provenance file is missing: {name} -> {path}")
        reporter.stage(f"stage=hash_frozen_review_source name={name} path={path}")
        observed = sha256_file(path)
        expected = EXPECTED_FROZEN_SHA256[name]
        if observed != expected:
            raise ReviewError(
                f"frozen provenance hash mismatch for {name}: observed={observed}, expected={expected}"
            )
        details[f"frozen_{name}"] = {
            "path": str(path),
            "sha256": observed,
            "size_bytes": int(path.stat().st_size),
        }

    state_bank = pd.read_csv(PROTOCOL_DIR / "state_bank.csv")
    accepted_active = pd.read_csv(paths["accepted_active_parameters"])
    return details, state_bank, accepted_active


def _load_frames(input_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, name in FRAME_FILES.items():
        path = input_dir / name
        if not path.is_file():
            raise ReviewError(f"required CSV artifact is missing: {path}")
        try:
            frames[key] = pd.read_csv(path)
        except Exception as error:
            raise ReviewError(f"cannot read CSV artifact {path}: {error}") from error
    return frames


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReviewError(f"required NPZ artifact is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    except Exception as error:
        raise ReviewError(f"cannot read NPZ artifact {path}: {error}") from error


def _normalize_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    integer_columns = {
        "active": (),
        "atomic": ("state_position", "source_weight_index", "draw", "pair", "step_key"),
        "bridge_atomic": (
            "state_position",
            "source_weight_index",
            "draw",
            "pair",
            "step_key",
            "batch_1_offset",
            "batch_2_offset",
        ),
        "bridge": ("state_position", "source_weight_index", "draw", "step_key"),
        "prefix_scalar": ("state_position", "draw", "prefix"),
        "state_summary": ("state_position", "prefix"),
        "signal": ("state_position",),
        "crossfit": ("state_left", "state_right", "split"),
        "nonalignment": (),
        "task_crossfit": ("split",),
        "panel": ("state_position", "step_stratum", "source_weight_index"),
        "sentinel_stratum": ("state_count",),
        "sentinel_persistence": ("state_count",),
        "anova": ("degrees_freedom",),
        "delta": (),
        "primary_bootstrap": ("bootstrap",),
        "panel_bootstrap": ("bootstrap",),
    }
    return {
        key: _coerce_integer_columns(frame, integer_columns[key], FRAME_FILES[key])
        if integer_columns[key]
        else frame.copy()
        for key, frame in frames.items()
    }


def _validate_frame_contracts(
    frames: Mapping[str, pd.DataFrame], state_bank: pd.DataFrame
) -> dict[str, int]:
    required_columns = {
        "atomic": (
            "panel",
            "state_position",
            "source_weight_index",
            "task_name",
            "draw",
            "pair",
            "step_key",
            "latent_hash",
            "probe_seed_1",
            "probe_seed_2",
            "probe_hash_1",
            "probe_hash_2",
            "a_scalar",
            "atomic_gradient_norm",
            "h1_norm",
            "h2_norm",
            "batch_1_hash",
            "batch_2_hash",
            "batch_train_count",
            "batch_size_effective",
        ),
        "bridge_atomic": (
            "state_position",
            "source_weight_index",
            "task_name",
            "draw",
            "pair",
            "step_key",
            "probe_seed_1",
            "probe_seed_2",
            "probe_hash_1",
            "probe_hash_2",
            "a_scalar",
            "full_ce_a_scalar",
            "batch_1_hash",
            "batch_2_hash",
            "batch_1_offset",
            "batch_2_offset",
            "batch_1_indices",
            "batch_2_indices",
            "batch_train_count",
            "batch_size_effective",
        ),
        "bridge": (
            "state_position",
            "source_weight_index",
            "task_name",
            "draw",
            "step_key",
            "gradient_cosine_b128_to_full",
            "gradient_relative_error_b128_to_full",
            "gradient_norm_ratio_b128_to_full",
            "a_scalar_b128",
            "a_scalar_full",
        ),
        "prefix_scalar": (
            "panel",
            "state_position",
            "task_name",
            "draw",
            "prefix",
            "a_scalar",
        ),
        "state_summary": (
            "state_position",
            "prefix",
            "panel",
            "task_name",
            "pairwise_cosine_median",
            "pairwise_cosine_q10",
            "pairwise_cosine_q90",
            "pairwise_cosine_min",
            "pairwise_negative_fraction",
            "gradient_norm_mean",
            "gradient_norm_rms",
            "gradient_norm_q99",
            "gradient_norm_max",
            "gradient_norm_top1pct_mass",
            "gradient_norm_top5pct_mass",
            "mean_gradient_norm",
            "mean_gradient_norm_over_rms",
            "signal_fraction",
            "split_vector_kind",
            "split_cosines",
            "split_gate_pass",
            "raw_split_cosines",
            "fold_vector_kind",
            "fold_cosine_median",
            "fold_cosine_min",
            "raw_fold_cosine_median",
            "raw_fold_cosine_min",
            "largest_norm_draw",
            "leave_largest_draw_out_mean_cosine",
            "leave_largest_draw_out_norm_ratio",
        ),
        "signal": (
            "state_position",
            "panel",
            "signal_fraction",
            "ci95_low",
            "ci95_high",
            "category",
        ),
        "crossfit": (
            "state_left",
            "state_right",
            "same_task",
            "split",
            "left_draws",
            "right_draws",
            "crossfit_cosine_forward",
            "crossfit_cosine_reverse",
            "crossfit_cosine",
            "both_states_pass_conditional_mean_gate",
        ),
        "nonalignment": ("gate", "passed"),
        "task_crossfit": (
            "split",
            "fashion_self_stability",
            "mnist_self_stability",
            "fashion_mnist_crossfit_cosine",
        ),
        "panel": (
            "state_position",
            "panel",
            "task_name",
            "step_stratum",
            "source_weight_index",
            "prior_stratum",
            "prior_grad_rms",
            "fresh_a_mean",
            "fresh_atomic_gradient_rms",
            "fresh_hvp_rms",
            "conditional_probe_variance_W",
            "conditional_mean_gradient_norm",
            "pairwise_cosine_median",
        ),
        "sentinel_stratum": (
            "task_name",
            "prior_stratum",
            "state_count",
            "prior_grad_rms_mean",
            "fresh_a_mean",
            "fresh_atomic_gradient_rms_mean",
            "fresh_hvp_rms_mean",
            "conditional_probe_variance_W_mean",
            "conditional_mean_gradient_norm_mean",
        ),
        "sentinel_persistence": (
            "task_name",
            "fresh_metric",
            "scope",
            "state_count",
            "spearman_to_prior_grad_rms",
        ),
        "anova": (
            "vector_kind",
            "component",
            "sum_squared_vector_norms",
            "degrees_freedom",
            "mean_square_trace",
            "random_effect_trace_estimate",
        ),
        "delta": (
            "scope",
            "vector_kind",
            "V_probe",
            "V_state",
            "Delta",
            "state_x_probe_ms",
            "E_task_panel",
            "E_state_within",
            "V_state_total",
            "Delta_total",
        ),
        "primary_bootstrap": (
            "bootstrap",
            "bootstrap_kind",
            "vector_kind",
            "Delta",
            "V_probe",
            "V_state",
            "E_task_panel",
            "E_state_within",
            "V_state_total",
            "Delta_total",
            "fashion_delta",
            "mnist_delta",
            "fashion_minus_mnist_delta",
            "fashion_minus_mnist_V_probe",
            "fashion_state_x_probe_ms",
            "mnist_state_x_probe_ms",
            "fashion_minus_mnist_state_x_probe_ms",
            "state_median_pairwise_cosine_median",
        ),
        "panel_bootstrap": (
            "bootstrap",
            "bootstrap_kind",
            "vector_kind",
            "Delta",
            "V_probe",
            "V_state",
            "fashion_delta",
            "mnist_delta",
            "fashion_minus_mnist_delta",
            "fashion_minus_mnist_V_probe",
        ),
    }
    for key, columns in required_columns.items():
        _require_columns(frames[key], columns, FRAME_FILES[key])

    require_exact_key_grid(
        frames["atomic"],
        ("state_position", "draw", "pair"),
        itertools.product(range(STATE_COUNT), range(DRAW_COUNT), range(PAIR_COUNT)),
        FRAME_FILES["atomic"],
    )
    require_exact_key_grid(
        frames["bridge_atomic"],
        ("state_position", "draw", "pair"),
        itertools.product(BRIDGE_POSITIONS, range(16), range(PAIR_COUNT)),
        FRAME_FILES["bridge_atomic"],
    )
    require_exact_key_grid(
        frames["bridge"],
        ("state_position", "draw"),
        itertools.product(BRIDGE_POSITIONS, range(16)),
        FRAME_FILES["bridge"],
    )
    require_exact_key_grid(
        frames["prefix_scalar"],
        ("state_position", "draw", "prefix"),
        itertools.product(range(STATE_COUNT), range(DRAW_COUNT), PREFIXES),
        FRAME_FILES["prefix_scalar"],
    )
    require_exact_key_grid(
        frames["state_summary"],
        ("state_position", "prefix"),
        itertools.product(range(STATE_COUNT), PREFIXES),
        FRAME_FILES["state_summary"],
    )
    require_exact_key_grid(
        frames["signal"],
        ("state_position",),
        ((position,) for position in range(STATE_COUNT)),
        FRAME_FILES["signal"],
    )
    require_exact_key_grid(
        frames["delta"],
        ("scope", "vector_kind"),
        itertools.product((*TASKS, "primary_mean_tasks"), ("raw", "unit")),
        FRAME_FILES["delta"],
    )
    anova_components = (
        "grand",
        "task",
        "state(task)",
        "probe",
        "task*probe",
        "state(task)*probe",
    )
    require_exact_key_grid(
        frames["anova"],
        ("vector_kind", "component"),
        itertools.product(("raw", "unit"), anova_components),
        FRAME_FILES["anova"],
    )
    require_exact_key_grid(
        frames["panel"],
        ("state_position",),
        ((position,) for position in range(STATE_COUNT)),
        FRAME_FILES["panel"],
    )
    require_exact_key_grid(
        frames["sentinel_stratum"],
        ("task_name", "prior_stratum"),
        itertools.product(TASKS, ("low", "middle", "high")),
        FRAME_FILES["sentinel_stratum"],
    )
    fresh_metrics = (
        "fresh_a_mean",
        "fresh_atomic_gradient_rms",
        "fresh_hvp_rms",
        "conditional_probe_variance_W",
        "conditional_mean_gradient_norm",
    )
    require_exact_key_grid(
        frames["sentinel_persistence"],
        ("task_name", "fresh_metric", "scope"),
        itertools.product(
            TASKS,
            fresh_metrics,
            ("all_selected_sentinels", "leave_highest_prior_state_out"),
        ),
        FRAME_FILES["sentinel_persistence"],
    )
    require_exact_key_grid(
        frames["nonalignment"],
        ("gate",),
        (("within_task_state_nonalignment",), ("fixed_task_panel_nonalignment",)),
        FRAME_FILES["nonalignment"],
    )
    require_exact_key_grid(
        frames["task_crossfit"],
        ("split",),
        ((split,) for split in range(3)),
        FRAME_FILES["task_crossfit"],
    )
    state_pairs = [
        (left, right)
        for start in (0, 16)
        for left in range(start, start + 16)
        for right in range(left + 1, start + 16)
    ]
    state_pairs.extend((left, left + 16) for left in range(16))
    require_exact_key_grid(
        frames["crossfit"],
        ("state_left", "state_right", "split"),
        ((left, right, split) for left, right in state_pairs for split in range(3)),
        FRAME_FILES["crossfit"],
    )
    require_exact_key_grid(
        frames["primary_bootstrap"],
        ("bootstrap", "vector_kind"),
        itertools.product(range(BOOTSTRAP_DRAWS), ("raw", "unit")),
        FRAME_FILES["primary_bootstrap"],
    )
    require_exact_key_grid(
        frames["panel_bootstrap"],
        ("bootstrap", "vector_kind"),
        itertools.product(range(BOOTSTRAP_DRAWS), ("raw", "unit")),
        FRAME_FILES["panel_bootstrap"],
    )

    finite_columns = {
        "atomic": ("a_scalar", "atomic_gradient_norm", "h1_norm", "h2_norm"),
        "bridge_atomic": (
            "a_scalar",
            "full_ce_a_scalar",
            "atomic_gradient_norm",
            "h1_norm",
            "h2_norm",
        ),
        "bridge": (
            "gradient_cosine_b128_to_full",
            "gradient_relative_error_b128_to_full",
            "gradient_norm_ratio_b128_to_full",
            "a_scalar_b128",
            "a_scalar_full",
        ),
        "prefix_scalar": ("a_scalar",),
        "state_summary": (
            "pairwise_cosine_median",
            "pairwise_cosine_q10",
            "pairwise_cosine_q90",
            "pairwise_cosine_min",
            "pairwise_negative_fraction",
            "gradient_norm_mean",
            "gradient_norm_rms",
            "gradient_norm_q99",
            "gradient_norm_max",
            "gradient_norm_top1pct_mass",
            "gradient_norm_top5pct_mass",
            "mean_gradient_norm",
            "mean_gradient_norm_over_rms",
            "signal_fraction",
            "fold_cosine_median",
            "fold_cosine_min",
            "raw_fold_cosine_median",
            "raw_fold_cosine_min",
            "leave_largest_draw_out_mean_cosine",
            "leave_largest_draw_out_norm_ratio",
        ),
        "signal": ("signal_fraction", "ci95_low", "ci95_high"),
        "crossfit": (
            "crossfit_cosine_forward",
            "crossfit_cosine_reverse",
            "crossfit_cosine",
        ),
        "task_crossfit": (
            "fashion_self_stability",
            "mnist_self_stability",
            "fashion_mnist_crossfit_cosine",
        ),
        "panel": (
            "fresh_a_mean",
            "fresh_atomic_gradient_rms",
            "fresh_hvp_rms",
            "conditional_probe_variance_W",
            "conditional_mean_gradient_norm",
            "pairwise_cosine_median",
        ),
        "sentinel_stratum": (
            "prior_grad_rms_mean",
            "fresh_a_mean",
            "fresh_atomic_gradient_rms_mean",
            "fresh_hvp_rms_mean",
            "conditional_probe_variance_W_mean",
            "conditional_mean_gradient_norm_mean",
        ),
        "sentinel_persistence": ("spearman_to_prior_grad_rms",),
        "anova": (
            "sum_squared_vector_norms",
            "degrees_freedom",
            "mean_square_trace",
        ),
        "delta": ("V_probe", "V_state", "Delta", "state_x_probe_ms"),
        "primary_bootstrap": (
            "Delta",
            "V_probe",
            "V_state",
            "E_task_panel",
            "E_state_within",
            "V_state_total",
            "Delta_total",
            "fashion_delta",
            "mnist_delta",
            "fashion_minus_mnist_delta",
            "fashion_minus_mnist_V_probe",
            "fashion_state_x_probe_ms",
            "mnist_state_x_probe_ms",
            "fashion_minus_mnist_state_x_probe_ms",
            "state_median_pairwise_cosine_median",
        ),
        "panel_bootstrap": (
            "Delta",
            "V_probe",
            "V_state",
            "fashion_delta",
            "mnist_delta",
            "fashion_minus_mnist_delta",
            "fashion_minus_mnist_V_probe",
        ),
    }
    for key, columns in finite_columns.items():
        require_finite(frames[key], columns, FRAME_FILES[key])

    sentinel_panel = frames["panel"].loc[frames["panel"]["panel"] == "sentinel"]
    require_finite(sentinel_panel, ("prior_grad_rms",), FRAME_FILES["panel"])
    primary_delta = frames["delta"].loc[
        frames["delta"]["scope"] == "primary_mean_tasks"
    ]
    require_finite(
        primary_delta,
        ("E_task_panel", "E_state_within", "V_state_total", "Delta_total"),
        FRAME_FILES["delta"],
    )

    if len(state_bank) != STATE_COUNT:
        raise ReviewError("frozen state bank no longer has exactly 44 rows")
    state_bank = _coerce_integer_columns(
        state_bank,
        ("state_position", "source_weight_index"),
        "state_bank.csv",
    ).sort_values("state_position")
    if state_bank["state_position"].tolist() != list(range(STATE_COUNT)):
        raise ReviewError("frozen state bank positions are not exactly 0..43")
    if int((state_bank["panel"] == "primary").sum()) != PRIMARY_COUNT:
        raise ReviewError("frozen state bank primary cardinality mismatch")
    if int((state_bank["panel"] == "sentinel").sum()) != SENTINEL_COUNT:
        raise ReviewError("frozen state bank sentinel cardinality mismatch")

    return {key: int(len(frame)) for key, frame in frames.items()}


def _validate_active_manifest(
    active: pd.DataFrame, accepted: pd.DataFrame, resolved: Mapping[str, Any]
) -> dict[str, Any]:
    required = ("parameter", "module", "shape", "offset_start", "offset_end", "numel")
    _require_columns(active, required, FRAME_FILES["active"])
    if not active.astype(str).equals(accepted.astype(str)):
        raise ReviewError("active_parameters.csv differs from the frozen accepted manifest")
    require_finite(active, ("offset_start", "offset_end", "numel"), FRAME_FILES["active"])
    if int(pd.to_numeric(active["numel"]).sum()) != EXPECTED_ACTIVE_COUNT:
        raise ReviewError("active parameter count mismatch")
    if active["parameter"].astype(str).tolist() != list(resolved["active_parameter_names"]):
        raise ReviewError("resolved active parameter order differs from active_parameters.csv")
    return {"parameter_rows": len(active), "active_parameter_count": EXPECTED_ACTIVE_COUNT}


def _state_metadata(state_bank: pd.DataFrame) -> pd.DataFrame:
    columns = ("state_position", "panel", "task_name", "source_weight_index")
    return state_bank.loc[:, list(columns)].sort_values("state_position").reset_index(drop=True)


def _validate_common_probe_replay(
    atomic: pd.DataFrame,
    bridge_atomic: pd.DataFrame,
    state_bank: pd.DataFrame,
) -> dict[str, Any]:
    metadata = _state_metadata(state_bank)
    observed_metadata = atomic.loc[
        :, ["state_position", "panel", "task_name", "source_weight_index"]
    ].drop_duplicates()
    if not observed_metadata.sort_values("state_position").reset_index(drop=True).equals(metadata):
        raise ReviewError("atomic state metadata differs from the frozen state bank")

    if not atomic[["batch_1_hash", "batch_2_hash"]].eq("full").all().all():
        raise ReviewError("primary/sentinel atomic rows are not all full CE")
    if not atomic["batch_train_count"].astype(int).eq(TRAIN_COUNT).all():
        raise ReviewError("atomic rows have a non-frozen train count")
    if not atomic["batch_size_effective"].astype(int).eq(TRAIN_COUNT).all():
        raise ReviewError("atomic rows have a non-full effective batch")
    if not atomic["step_key"].eq(10 * (atomic["draw"] + 1)).all():
        raise ReviewError("atomic step keys do not equal 10*(draw+1)")
    if atomic.groupby("state_position")["latent_hash"].nunique().ne(1).any():
        raise ReviewError("latent hash is not fixed within state")

    expected_seed_1 = np.asarray(
        [
            stable_uint63(PROTOCOL_ID, BASE_SEED, "probe", draw, pair, 0)
            for draw, pair in atomic[["draw", "pair"]].itertuples(index=False, name=None)
        ],
        dtype=np.int64,
    )
    expected_seed_2 = np.asarray(
        [
            stable_uint63(PROTOCOL_ID, BASE_SEED, "probe", draw, pair, 1)
            for draw, pair in atomic[["draw", "pair"]].itertuples(index=False, name=None)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(atomic["probe_seed_1"].to_numpy(dtype=np.int64), expected_seed_1):
        raise ReviewError("atomic branch-1 probe seed replay failed")
    if not np.array_equal(atomic["probe_seed_2"].to_numpy(dtype=np.int64), expected_seed_2):
        raise ReviewError("atomic branch-2 probe seed replay failed")

    probe_hashes: list[str] = []
    for branch in (1, 2):
        grouped = atomic.groupby(["draw", "pair"])[f"probe_hash_{branch}"].nunique()
        if not grouped.eq(1).all():
            raise ReviewError(f"probe_hash_{branch} is not common across states")
        values = (
            atomic.drop_duplicates(["draw", "pair"])
            .sort_values(["draw", "pair"])[f"probe_hash_{branch}"]
            .astype(str)
            .tolist()
        )
        if not all(len(value) == 64 for value in values):
            raise ReviewError(f"probe_hash_{branch} contains a non-SHA256 value")
        probe_hashes.extend(values)
    if len(set(probe_hashes)) != DRAW_COUNT * PAIR_COUNT * 2:
        raise ReviewError("probe hashes are not unique across distinct frozen branch keys")

    primary_lookup = atomic.set_index(["state_position", "draw", "pair"])
    bank_lookup = state_bank.set_index("state_position")
    replayed_batches = 0
    for row in bridge_atomic.itertuples(index=False):
        state = int(row.state_position)
        draw = int(row.draw)
        pair = int(row.pair)
        bank_row = bank_lookup.loc[state]
        if (
            int(row.source_weight_index) != int(bank_row["source_weight_index"])
            or str(row.task_name) != str(bank_row["task_name"])
        ):
            raise ReviewError("bridge metadata differs from the frozen state bank")
        full = primary_lookup.loc[(state, draw, pair)]
        _assert_close(row.full_ce_a_scalar, full["a_scalar"], "bridge full scalar replay")
        if int(row.step_key) != 10 * (draw + 1):
            raise ReviewError("bridge step key mismatch")
        if int(row.batch_train_count) != TRAIN_COUNT or int(row.batch_size_effective) != 128:
            raise ReviewError("bridge batch cardinality mismatch")
        for branch in (1, 2):
            pair_key = 2 * pair + branch - 1
            offset = (
                int(row.source_weight_index) * 1009
                + int(row.step_key) * 9176
                + pair_key * 7919
            ) % TRAIN_COUNT
            indices = (np.arange(128, dtype=np.int64) + offset) % TRAIN_COUNT
            expected_hash = hashlib.sha256(indices.tobytes()).hexdigest()
            if int(getattr(row, f"batch_{branch}_offset")) != offset:
                raise ReviewError("bridge batch offset replay failed")
            if str(getattr(row, f"batch_{branch}_hash")) != expected_hash:
                raise ReviewError("bridge batch hash replay failed")
            if str(getattr(row, f"batch_{branch}_indices")) != json.dumps(indices.tolist()):
                raise ReviewError("bridge batch index replay failed")
            if int(getattr(row, f"probe_seed_{branch}")) != int(full[f"probe_seed_{branch}"]):
                raise ReviewError("bridge probe seed differs from paired full-CE cell")
            if str(getattr(row, f"probe_hash_{branch}")) != str(full[f"probe_hash_{branch}"]):
                raise ReviewError("bridge probe hash differs from paired full-CE cell")
            replayed_batches += 1
    return {
        "atomic_rows": len(atomic),
        "common_probe_keys": DRAW_COUNT * PAIR_COUNT * 2,
        "latent_hashes": STATE_COUNT,
        "bridge_rows": len(bridge_atomic),
        "bridge_branches_replayed": replayed_batches,
    }


def unit_gram(gram: np.ndarray) -> np.ndarray:
    value = np.asarray(gram, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ReviewError("unit Gram conversion requires a square matrix")
    norms = np.sqrt(np.maximum(np.diag(value), 0.0))
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ReviewError("unit Gram conversion requires finite nonzero vectors")
    result = value / np.outer(norms, norms)
    np.fill_diagonal(result, 1.0)
    if not np.isfinite(result).all():
        raise ReviewError("unit Gram conversion produced nonfinite values")
    return result


def _gram_group_metrics(
    gram: np.ndarray, left: Sequence[int], right: Sequence[int]
) -> tuple[float, float, float]:
    left_idx = np.asarray(left, dtype=np.int64)
    right_idx = np.asarray(right, dtype=np.int64)
    left_norm2 = float(gram[np.ix_(left_idx, left_idx)].mean())
    right_norm2 = float(gram[np.ix_(right_idx, right_idx)].mean())
    dot = float(gram[np.ix_(left_idx, right_idx)].mean())
    denominator = math.sqrt(max(left_norm2, 0.0) * max(right_norm2, 0.0))
    cosine = dot / denominator if denominator > 0.0 else float("nan")
    norm_ratio = math.sqrt(max(left_norm2, 0.0)) / max(
        math.sqrt(max(right_norm2, 0.0)), 1e-30
    )
    error2 = max(left_norm2 + right_norm2 - 2.0 * dot, 0.0)
    relative_error = math.sqrt(error2) / max(math.sqrt(max(right_norm2, 0.0)), 1e-30)
    return float(cosine), float(relative_error), float(norm_ratio)


def _top_mass_share(values: Sequence[float], fraction: float) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    if absolute.size == 0 or float(absolute.sum()) <= 0.0:
        raise ReviewError("top-mass share requires nonzero finite values")
    count = max(1, int(math.ceil(float(fraction) * int(absolute.size))))
    return float(np.sort(absolute)[-count:].sum() / absolute.sum())


def _signal_fraction(gram: np.ndarray) -> float:
    q = gram.shape[0]
    squared_sum = float(gram.sum())
    norm_sum = float(np.trace(gram))
    cross_draw_dot = (squared_sum - norm_sum) / (q * (q - 1))
    mean_squared_draw_norm = norm_sum / q
    if mean_squared_draw_norm <= 0.0:
        raise ReviewError("signal fraction requires positive mean squared norm")
    return cross_draw_dot / mean_squared_draw_norm


def recompute_state_summary(
    gram: np.ndarray, *, state_position: int, prefix: int
) -> dict[str, Any]:
    """Independently reconstruct one executor state-prefix summary from its Gram."""
    value = np.asarray(gram, dtype=np.float64)
    if value.shape != (DRAW_COUNT, DRAW_COUNT):
        raise ReviewError("state-prefix Gram must have frozen 64x64 shape")
    if not np.isfinite(value).all() or not np.allclose(
        value, value.T, rtol=2e-12, atol=2e-6
    ):
        raise ReviewError("state-prefix Gram is nonfinite or nonsymmetric")
    norms = np.sqrt(np.maximum(np.diag(value), 0.0))
    if np.any(norms <= 0.0):
        raise ReviewError("state-prefix Gram contains a zero-norm draw")
    direction = unit_gram(value)
    triangle = np.triu_indices(DRAW_COUNT, k=1)
    cosines = direction[triangle]
    raw_split = [_gram_group_metrics(value, left, right)[0] for left, right in SPLITS]
    unit_split = [
        _gram_group_metrics(direction, left, right)[0] for left, right in SPLITS
    ]
    raw_folds = [
        _gram_group_metrics(value, FOLDS[left], FOLDS[right])[0]
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    unit_folds = [
        _gram_group_metrics(direction, FOLDS[left], FOLDS[right])[0]
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    largest = int(np.argmax(np.diag(value)))
    leave = [draw for draw in range(DRAW_COUNT) if draw != largest]
    loo_cosine, _loo_error, loo_ratio = _gram_group_metrics(
        value, list(range(DRAW_COUNT)), leave
    )
    mean_norm2 = float(value.mean())
    mean_squared_norm = float(np.mean(np.diag(value)))
    return {
        "state_position": int(state_position),
        "prefix": int(prefix),
        "pairwise_cosine_median": float(np.median(cosines)),
        "pairwise_cosine_q10": float(np.quantile(cosines, 0.10)),
        "pairwise_cosine_q90": float(np.quantile(cosines, 0.90)),
        "pairwise_cosine_min": float(np.min(cosines)),
        "pairwise_negative_fraction": float(np.mean(cosines < 0.0)),
        "gradient_norm_mean": float(norms.mean()),
        "gradient_norm_rms": float(np.sqrt(np.mean(norms**2))),
        "gradient_norm_q99": float(np.quantile(norms, 0.99)),
        "gradient_norm_max": float(norms.max()),
        "gradient_norm_top1pct_mass": _top_mass_share(norms, 0.01),
        "gradient_norm_top5pct_mass": _top_mass_share(norms, 0.05),
        "mean_gradient_norm": math.sqrt(max(mean_norm2, 0.0)),
        "mean_gradient_norm_over_rms": math.sqrt(
            max(mean_norm2, 0.0) / max(mean_squared_norm, 1e-30)
        ),
        "signal_fraction": _signal_fraction(value),
        "split_cosines": unit_split,
        "split_gate_pass": bool(all(item >= 0.80 for item in unit_split)),
        "raw_split_cosines": raw_split,
        "fold_cosine_median": float(np.median(unit_folds)),
        "fold_cosine_min": float(np.min(unit_folds)),
        "raw_fold_cosine_median": float(np.median(raw_folds)),
        "raw_fold_cosine_min": float(np.min(raw_folds)),
        "largest_norm_draw": largest,
        "leave_largest_draw_out_mean_cosine": loo_cosine,
        "leave_largest_draw_out_norm_ratio": loo_ratio,
    }


def _validate_state_grams_and_summaries(
    archives: Mapping[str, Mapping[str, np.ndarray]],
    state_summary: pd.DataFrame,
    state_bank: pd.DataFrame,
) -> dict[str, Any]:
    state_grams = archives["state_grams"]
    expected_keys = {
        f"state_{position:02d}_p{prefix}"
        for position in range(STATE_COUNT)
        for prefix in PREFIXES
    }
    if set(state_grams) != expected_keys:
        raise ReviewError("state_prefix_grams.npz key set mismatch")
    recomputed_rows = []
    for position in range(STATE_COUNT):
        for prefix in PREFIXES:
            recomputed_rows.append(
                recompute_state_summary(
                    state_grams[f"state_{position:02d}_p{prefix}"],
                    state_position=position,
                    prefix=prefix,
                )
            )
    recomputed = pd.DataFrame(recomputed_rows)
    numeric = [
        column
        for column in recomputed.columns
        if column
        not in {
            "state_position",
            "prefix",
            "split_cosines",
            "raw_split_cosines",
            "split_gate_pass",
            "largest_norm_draw",
        }
    ]
    maximum = _compare_frames(
        state_summary,
        recomputed,
        keys=("state_position", "prefix"),
        numeric_columns=numeric,
        table_name=FRAME_FILES["state_summary"],
    )
    observed_index = state_summary.set_index(["state_position", "prefix"])
    for row in recomputed.itertuples(index=False):
        observed = observed_index.loc[(row.state_position, row.prefix)]
        for column in ("split_cosines", "raw_split_cosines"):
            try:
                values = np.asarray(json.loads(str(observed[column])), dtype=np.float64)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ReviewError(f"invalid {column} JSON in state summary") from error
            expected = np.asarray(getattr(row, column), dtype=np.float64)
            if values.shape != (3,) or not np.allclose(values, expected, rtol=2e-10, atol=2e-10):
                raise ReviewError(f"state summary {column} differs from Gram recomputation")
        if _strict_bool(observed["split_gate_pass"], label="split_gate_pass") is not bool(
            row.split_gate_pass
        ):
            raise ReviewError("state summary split_gate_pass differs from Gram recomputation")
        if int(observed["largest_norm_draw"]) != int(row.largest_norm_draw):
            raise ReviewError("state summary largest_norm_draw differs from Gram recomputation")
        if str(observed["split_vector_kind"]) != "unit" or str(observed["fold_vector_kind"]) != "unit":
            raise ReviewError("directional state gates are not labeled as unit-vector gates")

    metadata = _state_metadata(state_bank).set_index("state_position")
    for row in state_summary.itertuples(index=False):
        expected = metadata.loc[int(row.state_position)]
        if str(row.panel) != str(expected["panel"]) or str(row.task_name) != str(expected["task_name"]):
            raise ReviewError("state summary panel/task metadata differs from state bank")
    return {
        "archive_keys": len(state_grams),
        "recomputed_rows": len(recomputed),
        "maximum_absolute_deltas": maximum,
    }


def _qform(gram: np.ndarray, indices: Sequence[int]) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    return float(gram[np.ix_(idx, idx)].sum())


def _group_mean_norm2(gram: np.ndarray, indices: Sequence[int]) -> float:
    return _qform(gram, indices) / float(len(indices) ** 2)


def _delta_from_single_task_gram(gram: np.ndarray) -> DeltaEstimate:
    value = np.asarray(gram, dtype=np.float64)
    n = 16
    q = DRAW_COUNT
    if value.shape != (n * q, n * q):
        raise ReviewError("task Gram has a non-frozen shape")
    cells = np.arange(n * q).reshape(n, q)
    state_means = np.asarray(
        [_group_mean_norm2(value, cells[state]) for state in range(n)]
    )
    within = np.asarray(
        [
            (
                float(np.trace(value[np.ix_(cells[state], cells[state])]))
                - q * state_means[state]
            )
            / (q - 1)
            for state in range(n)
        ]
    )
    v_probe = float(within.mean())
    task_mean_norm2 = _group_mean_norm2(value, cells.ravel())
    observed_state = (float(state_means.sum()) - n * task_mean_norm2) / (n - 1)
    total = float(np.trace(value))
    probe_energy = sum(
        _group_mean_norm2(value, cells[:, draw]) for draw in range(q)
    )
    residual_ss = (
        total
        - q * float(state_means.sum())
        - n * probe_energy
        + n * q * task_mean_norm2
    )
    interaction_ms = residual_ss / ((n - 1) * (q - 1))
    v_state = observed_state - interaction_ms / q
    return DeltaEstimate(v_probe, v_state, v_probe - v_state, interaction_ms)


def _anova_from_task_grams(
    task_grams: Mapping[str, np.ndarray], cross_task_sum_gram: np.ndarray
) -> pd.DataFrame:
    if set(task_grams) != set(TASKS):
        raise ReviewError("ANOVA requires exact frozen task Grams")
    q = DRAW_COUNT
    n = 16
    t = 2
    state_count = t * n
    total_n = state_count * q
    grams = {task: np.asarray(task_grams[task], dtype=np.float64) for task in TASKS}
    cross = np.asarray(cross_task_sum_gram, dtype=np.float64)
    if any(value.shape != (n * q, n * q) for value in grams.values()):
        raise ReviewError("ANOVA task Gram shape mismatch")
    if cross.shape != (q, q):
        raise ReviewError("ANOVA cross-task probe-sum Gram shape mismatch")
    cells = np.arange(n * q).reshape(n, q)
    total_ss = sum(float(np.trace(grams[task])) for task in TASKS)
    task_numerators = {task: float(grams[task].sum()) for task in TASKS}
    grand_ss = (sum(task_numerators.values()) + 2.0 * float(cross.sum())) / total_n
    task_mean_energy = sum(
        value / float((n * q) ** 2) for value in task_numerators.values()
    )
    state_mean_energy = sum(
        _group_mean_norm2(grams[task], cells[state])
        for task in TASKS
        for state in range(n)
    )
    probe_mean_energy = 0.0
    task_probe_energy = 0.0
    for draw in range(q):
        own = {task: _qform(grams[task], cells[:, draw]) for task in TASKS}
        probe_mean_energy += (sum(own.values()) + 2.0 * float(cross[draw, draw])) / float(
            state_count**2
        )
        task_probe_energy += sum(value / float(n**2) for value in own.values())
    ss_task = n * q * task_mean_energy - grand_ss
    ss_state = q * state_mean_energy - n * q * task_mean_energy
    ss_probe = state_count * probe_mean_energy - grand_ss
    ss_task_probe = (
        n * task_probe_energy
        - n * q * task_mean_energy
        - state_count * probe_mean_energy
        + grand_ss
    )
    ss_residual = total_ss - grand_ss - ss_task - ss_state - ss_probe - ss_task_probe
    rows = [
        ("grand", grand_ss, 1),
        ("task", ss_task, 1),
        ("state(task)", ss_state, 30),
        ("probe", ss_probe, 63),
        ("task*probe", ss_task_probe, 63),
        ("state(task)*probe", ss_residual, 1890),
    ]
    result = pd.DataFrame(
        rows, columns=["component", "sum_squared_vector_norms", "degrees_freedom"]
    )
    result["mean_square_trace"] = (
        result["sum_squared_vector_norms"] / result["degrees_freedom"]
    )
    means = result.set_index("component")["mean_square_trace"]
    residual = float(means["state(task)*probe"])
    task_probe = float(means["task*probe"])
    estimates = {
        "grand": float("nan"),
        "task": float("nan"),
        "state(task)": (float(means["state(task)"]) - residual) / q,
        "probe": (float(means["probe"]) - task_probe) / state_count,
        "task*probe": (task_probe - residual) / n,
        "state(task)*probe": residual,
    }
    result["random_effect_trace_estimate"] = result["component"].map(estimates)
    centered_total = total_ss - grand_ss
    effect_total = float(
        result.loc[result["component"] != "grand", "sum_squared_vector_norms"].sum()
    )
    if not math.isclose(
        centered_total,
        effect_total,
        rel_tol=2e-8,
        abs_tol=2e-8 * max(abs(centered_total), 1.0),
    ):
        raise ReviewError("independent ANOVA effects do not close to centered total SS")
    return result


def _recompute_delta_and_anova(
    archives: Mapping[str, Mapping[str, np.ndarray]],
    observed_delta: pd.DataFrame,
    observed_anova: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    task_grams = archives["task_grams"]
    cross_task = archives["cross_task"]
    if set(task_grams) != set(TASKS):
        raise ReviewError("primary_task_p4_grams.npz key set mismatch")
    if set(cross_task) != {"raw", "unit"}:
        raise ReviewError("cross_task_probe_sum_grams.npz key set mismatch")
    for task, gram in task_grams.items():
        if gram.shape != (16 * DRAW_COUNT, 16 * DRAW_COUNT):
            raise ReviewError(f"task Gram shape mismatch for {task}")
        if not np.isfinite(gram).all() or not np.allclose(
            gram, gram.T, rtol=2e-12, atol=2e-6
        ):
            raise ReviewError(f"task Gram is nonfinite or nonsymmetric for {task}")
    if any(value.shape != (DRAW_COUNT, DRAW_COUNT) for value in cross_task.values()):
        raise ReviewError("cross-task probe-sum Gram shape mismatch")
    if not all(np.isfinite(value).all() for value in cross_task.values()):
        raise ReviewError("cross-task probe-sum Gram is nonfinite")

    delta_rows: list[dict[str, Any]] = []
    anova_frames: list[pd.DataFrame] = []
    for vector_kind in ("raw", "unit"):
        grams = (
            task_grams
            if vector_kind == "raw"
            else {task: unit_gram(gram) for task, gram in task_grams.items()}
        )
        estimates = {task: _delta_from_single_task_gram(grams[task]) for task in TASKS}
        for task in TASKS:
            estimate = estimates[task]
            delta_rows.append(
                {
                    "scope": task,
                    "vector_kind": vector_kind,
                    "V_probe": estimate.v_probe,
                    "V_state": estimate.v_state,
                    "Delta": estimate.delta,
                    "state_x_probe_ms": estimate.state_interaction_ms,
                }
            )
        primary = {
            "scope": "primary_mean_tasks",
            "vector_kind": vector_kind,
            "V_probe": float(np.mean([value.v_probe for value in estimates.values()])),
            "V_state": float(np.mean([value.v_state for value in estimates.values()])),
            "Delta": float(np.mean([value.delta for value in estimates.values()])),
            "state_x_probe_ms": float(
                np.mean([value.state_interaction_ms for value in estimates.values()])
            ),
        }
        anova = _anova_from_task_grams(grams, cross_task[vector_kind])
        anova.insert(0, "vector_kind", vector_kind)
        anova_frames.append(anova)
        indexed = anova.set_index("component")
        task_ss = float(indexed.loc["task", "sum_squared_vector_norms"])
        task_probe_ms = float(indexed.loc["task*probe", "mean_square_trace"])
        panel_energy = (task_ss - task_probe_ms) / (DRAW_COUNT * (PRIMARY_COUNT - 1))
        state_within = 30.0 * float(primary["V_state"]) / 31.0
        primary.update(
            {
                "E_task_panel": panel_energy,
                "E_state_within": state_within,
                "V_state_total": panel_energy + state_within,
                "Delta_total": float(primary["V_probe"]) - panel_energy - state_within,
            }
        )
        delta_rows.append(primary)

    expected_delta = pd.DataFrame(delta_rows)
    expected_anova = pd.concat(anova_frames, ignore_index=True)
    delta_max = _compare_frames(
        observed_delta,
        expected_delta,
        keys=("scope", "vector_kind"),
        numeric_columns=("V_probe", "V_state", "Delta", "state_x_probe_ms"),
        table_name=FRAME_FILES["delta"],
    )
    primary_observed = observed_delta.loc[
        observed_delta["scope"] == "primary_mean_tasks"
    ]
    primary_expected = expected_delta.loc[
        expected_delta["scope"] == "primary_mean_tasks"
    ]
    delta_max.update(
        _compare_frames(
            primary_observed,
            primary_expected,
            keys=("scope", "vector_kind"),
            numeric_columns=("E_task_panel", "E_state_within", "V_state_total", "Delta_total"),
            table_name=FRAME_FILES["delta"],
        )
    )
    anova_max = _compare_frames(
        observed_anova,
        expected_anova,
        keys=("vector_kind", "component"),
        numeric_columns=(
            "sum_squared_vector_norms",
            "degrees_freedom",
            "mean_square_trace",
            "random_effect_trace_estimate",
        ),
        table_name=FRAME_FILES["anova"],
        rtol=2e-8,
        atol=2e-7,
    )
    return expected_delta, expected_anova, {
        "delta_maximum_absolute_deltas": delta_max,
        "anova_maximum_absolute_deltas": anova_max,
    }


def _validate_task_state_archive_identity(
    archives: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, float]:
    state_grams = archives["state_grams"]
    task_grams = archives["task_grams"]
    maximum: dict[str, float] = {}
    for task, offset in (("fashion_mnist", 0), ("mnist", 16)):
        task_gram = task_grams[task]
        max_delta = 0.0
        for local in range(16):
            block = task_gram[
                local * DRAW_COUNT : (local + 1) * DRAW_COUNT,
                local * DRAW_COUNT : (local + 1) * DRAW_COUNT,
            ]
            state = state_grams[f"state_{offset + local:02d}_p4"]
            max_delta = max(max_delta, float(np.max(np.abs(block - state))))
            if not np.allclose(block, state, rtol=2e-12, atol=2e-6):
                raise ReviewError(f"task Gram diagonal block differs from state Gram for {task}/{local}")
        maximum[task] = max_delta
    return maximum


def _validate_bridge_metrics(
    archives: Mapping[str, Mapping[str, np.ndarray]],
    bridge: pd.DataFrame,
    bridge_atomic: pd.DataFrame,
    prefix_scalar: pd.DataFrame,
    state_bank: pd.DataFrame,
) -> dict[str, Any]:
    bridge_grams = archives["bridge_grams"]
    expected_keys = {
        key
        for position in BRIDGE_POSITIONS
        for key in (
            f"state_{position:02d}_b128_self",
            f"state_{position:02d}_full_by_b128",
        )
    }
    if set(bridge_grams) != expected_keys:
        raise ReviewError("bridge_p4_grams.npz key set mismatch")
    scalar_b128 = bridge_atomic.groupby(["state_position", "draw"])["a_scalar"].mean()
    scalar_full = (
        prefix_scalar.loc[
            (prefix_scalar["prefix"] == 4) & (prefix_scalar["draw"] < 16)
        ]
        .set_index(["state_position", "draw"])["a_scalar"]
    )
    bank = state_bank.set_index("state_position")
    rows: list[dict[str, Any]] = []
    for position in BRIDGE_POSITIONS:
        self_gram = bridge_grams[f"state_{position:02d}_b128_self"]
        cross = bridge_grams[f"state_{position:02d}_full_by_b128"]
        full = archives["state_grams"][f"state_{position:02d}_p4"]
        if self_gram.shape != (16, 16) or cross.shape != (64, 16):
            raise ReviewError("bridge Gram shape mismatch")
        if not np.isfinite(self_gram).all() or not np.isfinite(cross).all():
            raise ReviewError("bridge Gram contains nonfinite values")
        if not np.allclose(self_gram, self_gram.T, rtol=2e-12, atol=2e-6):
            raise ReviewError("bridge self Gram is nonsymmetric")
        for draw in range(16):
            full_norm2 = float(full[draw, draw])
            bridge_norm2 = float(self_gram[draw, draw])
            dot = float(cross[draw, draw])
            denominator = math.sqrt(max(full_norm2 * bridge_norm2, 0.0))
            if denominator <= 0.0:
                raise ReviewError("bridge paired metric has a zero-norm gradient")
            error2 = max(full_norm2 + bridge_norm2 - 2.0 * dot, 0.0)
            row = bank.loc[position]
            rows.append(
                {
                    "state_position": position,
                    "draw": draw,
                    "source_weight_index": int(row["source_weight_index"]),
                    "task_name": str(row["task_name"]),
                    "step_key": 10 * (draw + 1),
                    "gradient_cosine_b128_to_full": dot / denominator,
                    "gradient_relative_error_b128_to_full": math.sqrt(error2)
                    / max(math.sqrt(full_norm2), 1e-30),
                    "gradient_norm_ratio_b128_to_full": math.sqrt(bridge_norm2)
                    / max(math.sqrt(full_norm2), 1e-30),
                    "a_scalar_b128": float(scalar_b128.loc[(position, draw)]),
                    "a_scalar_full": float(scalar_full.loc[(position, draw)]),
                }
            )
    expected = pd.DataFrame(rows)
    maximum = _compare_frames(
        bridge,
        expected,
        keys=("state_position", "draw"),
        numeric_columns=(
            "source_weight_index",
            "step_key",
            "gradient_cosine_b128_to_full",
            "gradient_relative_error_b128_to_full",
            "gradient_norm_ratio_b128_to_full",
            "a_scalar_b128",
            "a_scalar_full",
        ),
        table_name=FRAME_FILES["bridge"],
    )
    observed_tasks = bridge.set_index(["state_position", "draw"])["task_name"].astype(str)
    expected_tasks = expected.set_index(["state_position", "draw"])["task_name"].astype(str)
    if not observed_tasks.equals(expected_tasks):
        raise ReviewError("bridge task labels differ from the frozen state bank")
    return {
        "archive_keys": len(bridge_grams),
        "paired_rows": len(expected),
        "maximum_absolute_deltas": maximum,
    }


def _resampled_task_components(
    gram: np.ndarray,
    states: np.ndarray,
    probes: np.ndarray,
) -> DeltaEstimate:
    """Reconstruct one resampled crossed-design estimate from a task Gram."""
    value = np.asarray(gram, dtype=np.float64)
    q = len(probes)
    n = len(states)
    if value.shape != (16 * DRAW_COUNT, 16 * DRAW_COUNT) or q < 2 or n < 2:
        raise ReviewError("resampled task components received a non-frozen shape")
    cells = np.arange(16 * DRAW_COUNT).reshape(16, DRAW_COUNT)
    state_counts = np.bincount(states, minlength=16).astype(np.float64)
    probe_counts = np.bincount(probes, minlength=DRAW_COUNT).astype(np.float64)
    state_norms = np.zeros(16, dtype=np.float64)
    state_within = np.zeros(16, dtype=np.float64)
    for state in np.flatnonzero(state_counts):
        block = value[np.ix_(cells[state], cells[state])]
        summed_norm2 = float(probe_counts @ block @ probe_counts)
        state_norms[state] = summed_norm2 / float(q**2)
        sampled_norm_sum = float(probe_counts @ np.diag(block))
        state_within[state] = (sampled_norm_sum - summed_norm2 / q) / (q - 1)
    v_probe = float(state_counts @ state_within / n)
    cell_counts = np.outer(state_counts, probe_counts).ravel()
    task_mean = float(cell_counts @ value @ cell_counts) / float((n * q) ** 2)
    state_norm_sum = float(state_counts @ state_norms)
    observed_state = (state_norm_sum - n * task_mean) / (n - 1)
    total = float(cell_counts @ np.diag(value))
    probe_energy = 0.0
    for probe in np.flatnonzero(probe_counts):
        probe_cells = cells[:, probe]
        probe_norm2 = float(
            state_counts @ value[np.ix_(probe_cells, probe_cells)] @ state_counts
        )
        probe_energy += float(probe_counts[probe]) * probe_norm2 / float(n**2)
    residual_ss = (
        total - q * state_norm_sum - n * probe_energy + n * q * task_mean
    )
    interaction_ms = residual_ss / ((n - 1) * (q - 1))
    v_state = observed_state - interaction_ms / q
    return DeltaEstimate(v_probe, v_state, v_probe - v_state, interaction_ms)


def _aggregate_task_probe_gram(task_gram: np.ndarray) -> np.ndarray:
    value = np.asarray(task_gram, dtype=np.float64)
    if value.shape != (16 * DRAW_COUNT, 16 * DRAW_COUNT):
        raise ReviewError("task-probe aggregation requires a frozen task Gram")
    cells = np.arange(16 * DRAW_COUNT).reshape(16, DRAW_COUNT)
    return np.asarray(
        [
            [
                float(value[np.ix_(cells[:, left], cells[:, right])].sum())
                for right in range(DRAW_COUNT)
            ]
            for left in range(DRAW_COUNT)
        ],
        dtype=np.float64,
    )


def _panel_difference_gram(
    task_grams: Mapping[str, np.ndarray], cross_task_sum: np.ndarray
) -> np.ndarray:
    if set(task_grams) != set(TASKS):
        raise ReviewError("panel contrast requires both frozen task Grams")
    cross = np.asarray(cross_task_sum, dtype=np.float64)
    if cross.shape != (DRAW_COUNT, DRAW_COUNT):
        raise ReviewError("panel contrast requires a frozen QxQ cross-task Gram")
    aggregates = {task: _aggregate_task_probe_gram(task_grams[task]) for task in TASKS}
    return (
        aggregates[TASKS[0]] + aggregates[TASKS[1]] - cross - cross.T
    ) / float(16**2)


def _resampled_panel_energy(difference_gram: np.ndarray, probes: np.ndarray) -> float:
    selected = np.asarray(probes, dtype=np.int64)
    block = difference_gram[np.ix_(selected, selected)]
    expected_squared_difference = (
        float(block.sum()) - float(np.trace(block))
    ) / float(len(selected) * (len(selected) - 1))
    return float(16 * expected_squared_difference / (2.0 * (PRIMARY_COUNT - 1)))


def recompute_bootstrap_tables(
    task_grams: Mapping[str, np.ndarray],
    cross_task: Mapping[str, np.ndarray],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate both frozen bootstraps from Gram sufficient statistics and seeds."""
    if set(task_grams) != set(TASKS) or set(cross_task) != {"raw", "unit"}:
        raise ReviewError("bootstrap recomputation received an incomplete Gram bank")
    raw = {task: np.asarray(task_grams[task], dtype=np.float64) for task in TASKS}
    unit = {task: unit_gram(raw[task]) for task in TASKS}
    grams_by_kind = {"raw": raw, "unit": unit}
    panel_difference = {
        kind: _panel_difference_gram(grams, cross_task[kind])
        for kind, grams in grams_by_kind.items()
    }
    cells = np.arange(16 * DRAW_COUNT).reshape(16, DRAW_COUNT)
    fixed_states = np.arange(16, dtype=np.int64)
    triangle = np.triu_indices(DRAW_COUNT, k=1)

    primary_generator = np.random.default_rng(
        stable_uint63(PROTOCOL_ID, BASE_SEED, "primary_fixed_panel_probe_bootstrap")
    )
    primary_rows: list[dict[str, Any]] = []
    for bootstrap in range(int(draws)):
        common_probes = primary_generator.integers(0, DRAW_COUNT, size=DRAW_COUNT)
        state_medians = [
            float(
                np.median(
                    unit[task][
                        np.ix_(cells[state, common_probes], cells[state, common_probes])
                    ][triangle]
                )
            )
            for task in TASKS
            for state in range(16)
        ]
        cosine_statistic = float(np.median(state_medians))
        for vector_kind in ("raw", "unit"):
            grams = grams_by_kind[vector_kind]
            estimates = [
                _resampled_task_components(grams[task], fixed_states, common_probes)
                for task in TASKS
            ]
            v_probe = float(np.mean([item.v_probe for item in estimates]))
            v_state = float(np.mean([item.v_state for item in estimates]))
            panel_energy = _resampled_panel_energy(
                panel_difference[vector_kind], common_probes
            )
            state_within = 30.0 * v_state / 31.0
            state_total = panel_energy + state_within
            primary_rows.append(
                {
                    "bootstrap": bootstrap,
                    "bootstrap_kind": "primary_fixed_state_common_probe",
                    "vector_kind": vector_kind,
                    "Delta": float(np.mean([item.delta for item in estimates])),
                    "V_probe": v_probe,
                    "V_state": v_state,
                    "E_task_panel": panel_energy,
                    "E_state_within": state_within,
                    "V_state_total": state_total,
                    "Delta_total": v_probe - state_total,
                    "fashion_delta": estimates[0].delta,
                    "mnist_delta": estimates[1].delta,
                    "fashion_minus_mnist_delta": estimates[0].delta - estimates[1].delta,
                    "fashion_minus_mnist_V_probe": estimates[0].v_probe - estimates[1].v_probe,
                    "fashion_state_x_probe_ms": estimates[0].state_interaction_ms,
                    "mnist_state_x_probe_ms": estimates[1].state_interaction_ms,
                    "fashion_minus_mnist_state_x_probe_ms": (
                        estimates[0].state_interaction_ms
                        - estimates[1].state_interaction_ms
                    ),
                    "state_median_pairwise_cosine_median": cosine_statistic,
                }
            )

    panel_generator = np.random.default_rng(
        stable_uint63(PROTOCOL_ID, BASE_SEED, "panel_sensitivity_bootstrap")
    )
    panel_rows: list[dict[str, Any]] = []
    for bootstrap in range(int(draws)):
        common_probes = panel_generator.integers(0, DRAW_COUNT, size=DRAW_COUNT)
        state_samples = {
            task: np.concatenate(
                [
                    panel_generator.integers(
                        4 * stratum, 4 * (stratum + 1), size=4
                    )
                    for stratum in range(4)
                ]
            )
            for task in TASKS
        }
        for vector_kind in ("raw", "unit"):
            grams = grams_by_kind[vector_kind]
            estimates = [
                _resampled_task_components(
                    grams[task], state_samples[task], common_probes
                )
                for task in TASKS
            ]
            panel_rows.append(
                {
                    "bootstrap": bootstrap,
                    "bootstrap_kind": "panel_sensitivity_stratified_state_common_probe",
                    "vector_kind": vector_kind,
                    "Delta": float(np.mean([item.delta for item in estimates])),
                    "V_probe": float(np.mean([item.v_probe for item in estimates])),
                    "V_state": float(np.mean([item.v_state for item in estimates])),
                    "fashion_delta": estimates[0].delta,
                    "mnist_delta": estimates[1].delta,
                    "fashion_minus_mnist_delta": estimates[0].delta - estimates[1].delta,
                    "fashion_minus_mnist_V_probe": estimates[0].v_probe - estimates[1].v_probe,
                }
            )
    return pd.DataFrame(primary_rows), pd.DataFrame(panel_rows)


def validate_bootstrap_contract(
    primary: pd.DataFrame,
    panel: pd.DataFrame,
    task_grams: Mapping[str, np.ndarray],
    cross_task: Mapping[str, np.ndarray],
    *,
    expected_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Regenerate and compare every frozen bootstrap row from Gram statistics."""
    primary = _coerce_integer_columns(primary, ("bootstrap",), "primary bootstrap")
    panel = _coerce_integer_columns(panel, ("bootstrap",), "panel bootstrap")
    require_exact_key_grid(
        primary,
        ("bootstrap", "vector_kind"),
        itertools.product(range(expected_draws), ("raw", "unit")),
        "primary bootstrap",
    )
    require_exact_key_grid(
        panel,
        ("bootstrap", "vector_kind"),
        itertools.product(range(expected_draws), ("raw", "unit")),
        "panel bootstrap",
    )
    primary_numeric = (
        "Delta",
        "V_probe",
        "V_state",
        "E_task_panel",
        "E_state_within",
        "V_state_total",
        "Delta_total",
        "fashion_delta",
        "mnist_delta",
        "fashion_minus_mnist_delta",
        "fashion_minus_mnist_V_probe",
        "fashion_state_x_probe_ms",
        "mnist_state_x_probe_ms",
        "fashion_minus_mnist_state_x_probe_ms",
        "state_median_pairwise_cosine_median",
    )
    panel_numeric = (
        "Delta",
        "V_probe",
        "V_state",
        "fashion_delta",
        "mnist_delta",
        "fashion_minus_mnist_delta",
        "fashion_minus_mnist_V_probe",
    )
    require_finite(primary, primary_numeric, "primary bootstrap")
    require_finite(panel, panel_numeric, "panel bootstrap")
    if set(primary["bootstrap_kind"].astype(str)) != {"primary_fixed_state_common_probe"}:
        raise ReviewError("primary bootstrap kind is not the frozen fixed-state/common-probe kind")
    if set(panel["bootstrap_kind"].astype(str)) != {
        "panel_sensitivity_stratified_state_common_probe"
    }:
        raise ReviewError("panel bootstrap kind is not the frozen sensitivity kind")

    errors = {
        "primary_delta": np.abs(primary["Delta"] - (primary["V_probe"] - primary["V_state"])),
        "state_within": np.abs(primary["E_state_within"] - 30.0 * primary["V_state"] / 31.0),
        "state_total": np.abs(
            primary["V_state_total"]
            - primary["E_task_panel"]
            - primary["E_state_within"]
        ),
        "total_delta": np.abs(
            primary["Delta_total"] - primary["V_probe"] + primary["V_state_total"]
        ),
        "task_delta": np.abs(
            primary["fashion_minus_mnist_delta"]
            - primary["fashion_delta"]
            + primary["mnist_delta"]
        ),
        "interaction_delta": np.abs(
            primary["fashion_minus_mnist_state_x_probe_ms"]
            - primary["fashion_state_x_probe_ms"]
            + primary["mnist_state_x_probe_ms"]
        ),
        "panel_delta": np.abs(panel["Delta"] - panel["V_probe"] + panel["V_state"]),
        "panel_task_delta": np.abs(
            panel["fashion_minus_mnist_delta"]
            - panel["fashion_delta"]
            + panel["mnist_delta"]
        ),
    }
    maximum = {name: float(value.max()) for name, value in errors.items()}
    identities = {
        "primary_delta": (
            primary["Delta"],
            primary["V_probe"] - primary["V_state"],
        ),
        "state_within": (
            primary["E_state_within"],
            30.0 * primary["V_state"] / 31.0,
        ),
        "state_total": (
            primary["V_state_total"],
            primary["E_task_panel"] + primary["E_state_within"],
        ),
        "total_delta": (
            primary["Delta_total"],
            primary["V_probe"] - primary["V_state_total"],
        ),
        "task_delta": (
            primary["fashion_minus_mnist_delta"],
            primary["fashion_delta"] - primary["mnist_delta"],
        ),
        "interaction_delta": (
            primary["fashion_minus_mnist_state_x_probe_ms"],
            primary["fashion_state_x_probe_ms"] - primary["mnist_state_x_probe_ms"],
        ),
        "panel_delta": (
            panel["Delta"],
            panel["V_probe"] - panel["V_state"],
        ),
        "panel_task_delta": (
            panel["fashion_minus_mnist_delta"],
            panel["fashion_delta"] - panel["mnist_delta"],
        ),
    }
    failed = {
        name: maximum[name]
        for name, (observed, expected) in identities.items()
        if not np.allclose(observed, expected, rtol=2e-9, atol=2e-9)
    }
    if failed:
        raise ReviewError(f"bootstrap algebra mismatch: {failed}")
    paired_cosines = primary.pivot(
        index="bootstrap", columns="vector_kind", values="state_median_pairwise_cosine_median"
    )
    cosine_delta = float(np.max(np.abs(paired_cosines["raw"] - paired_cosines["unit"])))
    if cosine_delta != 0.0:
        raise ReviewError("primary raw/unit rows do not share the same probe-bootstrap cosine statistic")
    expected_primary, expected_panel = recompute_bootstrap_tables(
        task_grams, cross_task, draws=expected_draws
    )
    primary_max = _compare_frames(
        primary,
        expected_primary,
        keys=("bootstrap", "vector_kind"),
        numeric_columns=primary_numeric,
        table_name=FRAME_FILES["primary_bootstrap"],
        rtol=2e-9,
        atol=2e-9,
    )
    panel_max = _compare_frames(
        panel,
        expected_panel,
        keys=("bootstrap", "vector_kind"),
        numeric_columns=panel_numeric,
        table_name=FRAME_FILES["panel_bootstrap"],
        rtol=2e-9,
        atol=2e-9,
    )
    return {
        "primary_rows": len(primary),
        "panel_rows": len(panel),
        "bootstrap_draws": expected_draws,
        "maximum_algebra_error": maximum,
        "raw_unit_cosine_max_delta": cosine_delta,
        "primary_gram_recomputation_maximum_absolute_deltas": primary_max,
        "panel_gram_recomputation_maximum_absolute_deltas": panel_max,
    }, expected_primary, expected_panel


def validate_signal_bootstrap(
    signal: pd.DataFrame,
    state_grams: Mapping[str, np.ndarray],
    state_bank: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Regenerate each state-level H0 interval from its P=4 Gram and frozen seed."""
    bank = state_bank.set_index("state_position")
    rows: list[dict[str, Any]] = []
    for position in range(STATE_COUNT):
        gram = np.asarray(state_grams[f"state_{position:02d}_p4"], dtype=np.float64)
        generator = np.random.default_rng(
            stable_uint63(PROTOCOL_ID, BASE_SEED, "signal_bootstrap", position)
        )
        values = np.empty(int(draws), dtype=np.float64)
        for bootstrap in range(int(draws)):
            selected = generator.integers(0, DRAW_COUNT, size=DRAW_COUNT)
            values[bootstrap] = _signal_fraction(gram[np.ix_(selected, selected)])
        low, high = np.quantile(values, [0.025, 0.975])
        category = "unresolved"
        if high < 0.05:
            category = "near-zero conditional signal supported"
        elif low > 0.05:
            category = "near-zero excluded"
        rows.append(
            {
                "state_position": position,
                "panel": str(bank.loc[position, "panel"]),
                "signal_fraction": _signal_fraction(gram),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "category": category,
            }
        )
    expected = pd.DataFrame(rows)
    maximum = _compare_frames(
        signal,
        expected,
        keys=("state_position",),
        numeric_columns=("signal_fraction", "ci95_low", "ci95_high"),
        table_name=FRAME_FILES["signal"],
        rtol=2e-9,
        atol=2e-9,
    )
    observed_labels = signal.set_index("state_position")[["panel", "category"]].astype(str)
    expected_labels = expected.set_index("state_position")[["panel", "category"]].astype(str)
    if not observed_labels.equals(expected_labels):
        raise ReviewError("signal bootstrap labels differ from Gram recomputation")
    return {
        "states": len(expected),
        "bootstrap_draws_per_state": int(draws),
        "maximum_absolute_deltas": maximum,
    }, expected


def recompute_nonalignment_tables(
    archives: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rebuild cross-fit directions and both gates from unit Gram statistics."""
    state_grams = archives["state_grams"]
    task_grams = archives["task_grams"]
    reconstruction = archives["reconstruction_cross"]
    cross_task = archives["cross_task"]
    expected_reconstruction_keys = {
        *(f"pair_{left:02d}_{left + 16:02d}" for left in range(16)),
        *(
            f"pair_{left:02d}_{left + 1:02d}"
            for start in (0, 16)
            for left in range(start, start + 16, 2)
        ),
    }
    if set(reconstruction) != expected_reconstruction_keys:
        raise ReviewError("reconstruction cross-Gram key set mismatch")
    if any(
        value.shape != (DRAW_COUNT, DRAW_COUNT) or not np.isfinite(value).all()
        for value in reconstruction.values()
    ):
        raise ReviewError("reconstruction cross-Gram shape/finiteness mismatch")
    unit_states = {
        position: unit_gram(state_grams[f"state_{position:02d}_p4"])
        for position in range(PRIMARY_COUNT)
    }
    unit_tasks = {task: unit_gram(task_grams[task]) for task in TASKS}
    pass_map = {
        position: bool(
            all(
                _gram_group_metrics(unit_states[position], left, right)[0] >= 0.80
                for left, right in SPLITS
            )
        )
        for position in range(PRIMARY_COUNT)
    }

    def cross_unit_gram(left_state: int, right_state: int) -> np.ndarray:
        if left_state // 16 == right_state // 16:
            task = TASKS[0] if left_state < 16 else TASKS[1]
            offset = 0 if left_state < 16 else 16
            left_local = left_state - offset
            right_local = right_state - offset
            return unit_tasks[task][
                left_local * DRAW_COUNT : (left_local + 1) * DRAW_COUNT,
                right_local * DRAW_COUNT : (right_local + 1) * DRAW_COUNT,
            ]
        raw = reconstruction[f"pair_{left_state:02d}_{right_state:02d}"]
        left_norm = np.sqrt(
            np.maximum(
                np.diag(state_grams[f"state_{left_state:02d}_p4"]), 0.0
            )
        )
        right_norm = np.sqrt(
            np.maximum(
                np.diag(state_grams[f"state_{right_state:02d}_p4"]), 0.0
            )
        )
        if np.any(left_norm <= 0.0) or np.any(right_norm <= 0.0):
            raise ReviewError("cross-state unit Gram has a zero-norm draw")
        return raw / np.outer(left_norm, right_norm)

    state_pairs = [
        (left, right)
        for start in (0, 16)
        for left in range(start, start + 16)
        for right in range(left + 1, start + 16)
    ]
    state_pairs.extend((left, left + 16) for left in range(16))
    crossfit_rows: list[dict[str, Any]] = []
    for left_state, right_state in state_pairs:
        cross = cross_unit_gram(left_state, right_state)
        for split, (left_draws, right_draws) in enumerate(SPLITS):
            left_idx = np.asarray(left_draws, dtype=np.int64)
            right_idx = np.asarray(right_draws, dtype=np.int64)
            left_a = _group_mean_norm2(unit_states[left_state], left_idx)
            left_b = _group_mean_norm2(unit_states[left_state], right_idx)
            right_a = _group_mean_norm2(unit_states[right_state], left_idx)
            right_b = _group_mean_norm2(unit_states[right_state], right_idx)
            forward_dot = float(cross[np.ix_(left_idx, right_idx)].mean())
            reverse_dot = float(cross[np.ix_(right_idx, left_idx)].mean())
            forward = forward_dot / math.sqrt(max(left_a * right_b, 1e-300))
            reverse = reverse_dot / math.sqrt(max(left_b * right_a, 1e-300))
            crossfit_rows.append(
                {
                    "state_left": left_state,
                    "state_right": right_state,
                    "same_task": left_state // 16 == right_state // 16,
                    "split": split,
                    "left_draws": json.dumps(list(left_draws)),
                    "right_draws": json.dumps(list(right_draws)),
                    "crossfit_cosine_forward": forward,
                    "crossfit_cosine_reverse": reverse,
                    "crossfit_cosine": 0.5 * (forward + reverse),
                    "both_states_pass_conditional_mean_gate": (
                        pass_map[left_state] and pass_map[right_state]
                    ),
                }
            )
    crossfit = pd.DataFrame(crossfit_rows)
    passing_within = crossfit.loc[
        crossfit["same_task"] & crossfit["both_states_pass_conditional_mean_gate"]
    ]
    within_medians = (
        passing_within.groupby("split")["crossfit_cosine"].median().reindex(range(3))
    )
    passing_count = int(sum(pass_map.values()))
    within_gate = bool(
        passing_count >= 24
        and within_medians.notna().all()
        and (within_medians < 0.80).all()
    )

    task_probe = {
        task: _aggregate_task_probe_gram(unit_tasks[task]) for task in TASKS
    }
    task_rows: list[dict[str, Any]] = []
    task_cross = np.asarray(cross_task["unit"], dtype=np.float64)
    for split, (left_draws, right_draws) in enumerate(SPLITS):
        left_idx = np.asarray(left_draws, dtype=np.int64)
        right_idx = np.asarray(right_draws, dtype=np.int64)
        fashion_stability = _gram_group_metrics(
            task_probe[TASKS[0]], left_idx, right_idx
        )[0]
        mnist_stability = _gram_group_metrics(
            task_probe[TASKS[1]], left_idx, right_idx
        )[0]
        fashion_left = _group_mean_norm2(task_probe[TASKS[0]], left_idx)
        fashion_right = _group_mean_norm2(task_probe[TASKS[0]], right_idx)
        mnist_left = _group_mean_norm2(task_probe[TASKS[1]], left_idx)
        mnist_right = _group_mean_norm2(task_probe[TASKS[1]], right_idx)
        forward_dot = float(task_cross[np.ix_(left_idx, right_idx)].mean())
        reverse_dot = float(task_cross[np.ix_(right_idx, left_idx)].mean())
        forward = forward_dot / math.sqrt(max(fashion_left * mnist_right, 1e-300))
        reverse = reverse_dot / math.sqrt(max(fashion_right * mnist_left, 1e-300))
        task_rows.append(
            {
                "split": split,
                "fashion_self_stability": fashion_stability,
                "mnist_self_stability": mnist_stability,
                "fashion_mnist_crossfit_cosine": 0.5 * (forward + reverse),
            }
        )
    task_frame = pd.DataFrame(task_rows)
    task_gate = bool(
        (task_frame[["fashion_self_stability", "mnist_self_stability"]] >= 0.80)
        .all()
        .all()
        and (task_frame["fashion_mnist_crossfit_cosine"] < 0.80).all()
    )
    gate_frame = pd.DataFrame(
        [
            {
                "gate": "within_task_state_nonalignment",
                "passed": within_gate,
                **{
                    f"split_{split}_median_cosine": within_medians.loc[split]
                    for split in range(3)
                },
            },
            {
                "gate": "fixed_task_panel_nonalignment",
                "passed": task_gate,
                **{
                    f"split_{int(row.split)}_crossfit_cosine": row.fashion_mnist_crossfit_cosine
                    for row in task_frame.itertuples(index=False)
                },
            },
        ]
    )
    detail = {
        "conditional_mean_gate_passing_states": passing_count,
        "within_task_state_nonalignment_gate": within_gate,
        "fixed_task_panel_nonalignment_gate": task_gate,
        "within_split_median_cosines": [
            float(within_medians.loc[index]) for index in range(3)
        ],
    }
    return crossfit, task_frame, gate_frame, detail


def _validate_nonalignment(
    archives: Mapping[str, Mapping[str, np.ndarray]],
    state_summary: pd.DataFrame,
    crossfit: pd.DataFrame,
    task_crossfit: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    expected_crossfit, expected_task, expected_gates, detail = (
        recompute_nonalignment_tables(archives)
    )
    crossfit_max = _compare_frames(
        crossfit,
        expected_crossfit,
        keys=("state_left", "state_right", "split"),
        numeric_columns=(
            "crossfit_cosine_forward",
            "crossfit_cosine_reverse",
            "crossfit_cosine",
        ),
        table_name=FRAME_FILES["crossfit"],
        rtol=2e-9,
        atol=2e-9,
    )
    index_columns = ["state_left", "state_right", "split"]
    observed_labels = crossfit.set_index(index_columns)[
        [
            "same_task",
            "left_draws",
            "right_draws",
            "both_states_pass_conditional_mean_gate",
        ]
    ].copy()
    expected_labels = expected_crossfit.set_index(index_columns)[
        [
            "same_task",
            "left_draws",
            "right_draws",
            "both_states_pass_conditional_mean_gate",
        ]
    ].copy()
    for column in ("same_task", "both_states_pass_conditional_mean_gate"):
        observed_labels[column] = observed_labels[column].map(
            lambda value: _strict_bool(value, label=f"crossfit.{column}")
        )
    if not observed_labels.equals(expected_labels):
        raise ReviewError("crossfit labels or frozen split membership differ from Gram recomputation")

    task_max = _compare_frames(
        task_crossfit,
        expected_task,
        keys=("split",),
        numeric_columns=(
            "fashion_self_stability",
            "mnist_self_stability",
            "fashion_mnist_crossfit_cosine",
        ),
        table_name=FRAME_FILES["task_crossfit"],
        rtol=2e-9,
        atol=2e-9,
    )
    observed_gate_map = {
        str(row.gate): _strict_bool(row.passed, label=f"gate.{row.gate}")
        for row in gates.itertuples(index=False)
    }
    expected_gate_map = {
        str(row.gate): bool(row.passed) for row in expected_gates.itertuples(index=False)
    }
    if observed_gate_map != expected_gate_map:
        raise ReviewError("nonalignment gate decision differs from Gram recomputation")
    for gate_name, stem in (
        ("within_task_state_nonalignment", "split_{}_median_cosine"),
        ("fixed_task_panel_nonalignment", "split_{}_crossfit_cosine"),
    ):
        observed = gates.loc[gates["gate"] == gate_name].iloc[0]
        expected = expected_gates.loc[expected_gates["gate"] == gate_name].iloc[0]
        for split in range(3):
            column = stem.format(split)
            _assert_close(observed[column], expected[column], f"{gate_name}.{column}")

    p4 = state_summary.loc[state_summary["prefix"] == 4].set_index("state_position")
    observed_passing = int(
        sum(
            _strict_bool(value, label="state_summary.split_gate_pass")
            for value in p4.loc[range(PRIMARY_COUNT), "split_gate_pass"]
        )
    )
    if observed_passing != detail["conditional_mean_gate_passing_states"]:
        raise ReviewError("state-summary conditional gate count differs from unit-Gram recomputation")
    return {
        **detail,
        "crossfit_maximum_absolute_deltas": crossfit_max,
        "task_crossfit_maximum_absolute_deltas": task_max,
    }


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    ranked_left = pd.to_numeric(left, errors="coerce").rank(method="average")
    ranked_right = pd.to_numeric(right, errors="coerce").rank(method="average")
    result = float(ranked_left.corr(ranked_right))
    if not math.isfinite(result):
        raise ReviewError("sentinel Spearman correlation is undefined")
    return result


def _validate_sentinel_outputs(
    panel: pd.DataFrame,
    sentinel_stratum: pd.DataFrame,
    persistence: pd.DataFrame,
    atomic: pd.DataFrame,
    state_summary: pd.DataFrame,
    state_grams: Mapping[str, np.ndarray],
    state_bank: pd.DataFrame,
) -> dict[str, Any]:
    bank = state_bank.set_index("state_position")
    p4 = state_summary.loc[state_summary["prefix"] == 4].set_index("state_position")
    expected_rows = []
    for position in range(STATE_COUNT):
        group = atomic.loc[atomic["state_position"] == position]
        gram = state_grams[f"state_{position:02d}_p4"]
        bank_row = bank.loc[position]
        expected_rows.append(
            {
                "state_position": position,
                "fresh_a_mean": float(group["a_scalar"].mean()),
                "fresh_atomic_gradient_rms": float(
                    np.sqrt(np.mean(group["atomic_gradient_norm"].to_numpy(dtype=float) ** 2))
                ),
                "fresh_hvp_rms": float(
                    np.sqrt(
                        np.mean(
                            np.concatenate(
                                [
                                    group["h1_norm"].to_numpy(dtype=float) ** 2,
                                    group["h2_norm"].to_numpy(dtype=float) ** 2,
                                ]
                            )
                        )
                    )
                ),
                "conditional_probe_variance_W": (
                    float(np.trace(gram)) - DRAW_COUNT * float(gram.mean())
                )
                / (DRAW_COUNT - 1),
                "conditional_mean_gradient_norm": float(p4.loc[position, "mean_gradient_norm"]),
                "pairwise_cosine_median": float(p4.loc[position, "pairwise_cosine_median"]),
                "source_weight_index": int(bank_row["source_weight_index"]),
                "step_stratum": int(bank_row["step_stratum"]),
            }
        )
    expected_panel = pd.DataFrame(expected_rows)
    panel_max = _compare_frames(
        panel,
        expected_panel,
        keys=("state_position",),
        numeric_columns=(
            "fresh_a_mean",
            "fresh_atomic_gradient_rms",
            "fresh_hvp_rms",
            "conditional_probe_variance_W",
            "conditional_mean_gradient_norm",
            "pairwise_cosine_median",
            "source_weight_index",
            "step_stratum",
        ),
        table_name=FRAME_FILES["panel"],
    )

    sentinel = panel.loc[panel["panel"] == "sentinel"].copy()
    if len(sentinel) != SENTINEL_COUNT:
        raise ReviewError("panel_state_metrics.csv sentinel cardinality mismatch")
    expected_stratum = (
        sentinel.groupby(["task_name", "prior_stratum"], as_index=False)
        .agg(
            state_count=("state_position", "size"),
            prior_grad_rms_mean=("prior_grad_rms", "mean"),
            fresh_a_mean=("fresh_a_mean", "mean"),
            fresh_atomic_gradient_rms_mean=("fresh_atomic_gradient_rms", "mean"),
            fresh_hvp_rms_mean=("fresh_hvp_rms", "mean"),
            conditional_probe_variance_W_mean=("conditional_probe_variance_W", "mean"),
            conditional_mean_gradient_norm_mean=("conditional_mean_gradient_norm", "mean"),
        )
    )
    stratum_max = _compare_frames(
        sentinel_stratum,
        expected_stratum,
        keys=("task_name", "prior_stratum"),
        numeric_columns=(
            "state_count",
            "prior_grad_rms_mean",
            "fresh_a_mean",
            "fresh_atomic_gradient_rms_mean",
            "fresh_hvp_rms_mean",
            "conditional_probe_variance_W_mean",
            "conditional_mean_gradient_norm_mean",
        ),
        table_name=FRAME_FILES["sentinel_stratum"],
    )
    metrics = (
        "fresh_a_mean",
        "fresh_atomic_gradient_rms",
        "fresh_hvp_rms",
        "conditional_probe_variance_W",
        "conditional_mean_gradient_norm",
    )
    expected_persistence = []
    for task, group in sentinel.groupby("task_name", sort=True):
        highest = group["prior_grad_rms"].idxmax()
        for metric in metrics:
            expected_persistence.append(
                {
                    "task_name": task,
                    "fresh_metric": metric,
                    "scope": "all_selected_sentinels",
                    "state_count": len(group),
                    "spearman_to_prior_grad_rms": _rank_correlation(
                        group["prior_grad_rms"], group[metric]
                    ),
                }
            )
            leave = group.drop(index=highest)
            expected_persistence.append(
                {
                    "task_name": task,
                    "fresh_metric": metric,
                    "scope": "leave_highest_prior_state_out",
                    "state_count": len(leave),
                    "spearman_to_prior_grad_rms": _rank_correlation(
                        leave["prior_grad_rms"], leave[metric]
                    ),
                }
            )
    persistence_max = _compare_frames(
        persistence,
        pd.DataFrame(expected_persistence),
        keys=("task_name", "fresh_metric", "scope"),
        numeric_columns=("state_count", "spearman_to_prior_grad_rms"),
        table_name=FRAME_FILES["sentinel_persistence"],
    )
    return {
        "sentinel_states": len(sentinel),
        "panel_maximum_absolute_deltas": panel_max,
        "stratum_maximum_absolute_deltas": stratum_max,
        "persistence_maximum_absolute_deltas": persistence_max,
    }


def primary_mechanism_decision(
    *,
    delta_dir_ci_low: float,
    delta_dir_ci_high: float,
    within_state_cosine_median_ci_high: float,
    state_nonalignment_gate_passed: bool,
) -> str:
    if delta_dir_ci_low > 0.0 and within_state_cosine_median_ci_high < 0.80:
        return "probe-direction-major"
    if delta_dir_ci_high < 0.0 and bool(state_nonalignment_gate_passed):
        return "state-direction-major"
    return "comparable/inconclusive"


def derive_reviewed_decision(
    delta: pd.DataFrame,
    primary_bootstrap: pd.DataFrame,
    state_summary: pd.DataFrame,
    signal: pd.DataFrame,
    nonalignment: Mapping[str, Any],
) -> dict[str, Any]:
    unit_bootstrap = primary_bootstrap.loc[
        primary_bootstrap["vector_kind"] == "unit"
    ].sort_values("bootstrap")
    delta_low, delta_high = np.quantile(unit_bootstrap["Delta_total"], [0.025, 0.975])
    median_high = float(
        np.quantile(unit_bootstrap["state_median_pairwise_cosine_median"], 0.95)
    )
    heterogeneity_low, heterogeneity_high = np.quantile(
        unit_bootstrap["fashion_minus_mnist_state_x_probe_ms"], [0.025, 0.975]
    )
    within_gate = bool(nonalignment["within_task_state_nonalignment_gate"])
    task_gate = bool(nonalignment["fixed_task_panel_nonalignment_gate"])
    classification = primary_mechanism_decision(
        delta_dir_ci_low=float(delta_low),
        delta_dir_ci_high=float(delta_high),
        within_state_cosine_median_ci_high=median_high,
        state_nonalignment_gate_passed=within_gate or task_gate,
    )
    if classification != "state-direction-major":
        attribution = "state_level_attribution_not_activated"
    elif within_gate and task_gate:
        attribution = "mixed_state_level"
    elif task_gate:
        attribution = "fixed_task_panel_contrast_leading"
    elif within_gate:
        attribution = "HS_within_panel_state_leading"
    else:
        attribution = "state_level_nonalignment_not_established"
    point = delta.loc[
        (delta["scope"] == "primary_mean_tasks") & (delta["vector_kind"] == "unit")
    ].iloc[0]
    primary_p4 = state_summary.loc[
        (state_summary["panel"] == "primary") & (state_summary["prefix"] == 4)
    ]
    passing = sum(
        _strict_bool(value, label="primary split_gate_pass")
        for value in primary_p4["split_gate_pass"]
    )
    categories = (
        signal.loc[signal["panel"] == "primary", "category"]
        .value_counts(normalize=True)
        .to_dict()
    )
    return {
        "decision": classification,
        "delta_total_dir_ci95_low": float(delta_low),
        "delta_total_dir_ci95_high": float(delta_high),
        "delta_total_dir_point": float(point["Delta_total"]),
        "delta_within_dir_point": float(point["Delta"]),
        "E_task_panel_dir_point": float(point["E_task_panel"]),
        "E_state_within_dir_point": float(point["E_state_within"]),
        "state_level_median_within_state_p4_cosine_ci95_high": median_high,
        "conditional_mean_gate_passing_states": int(passing),
        "conditional_mean_gate_excluded_states": PRIMARY_COUNT - int(passing),
        "within_task_state_nonalignment_gate": within_gate,
        "fixed_task_panel_nonalignment_gate": task_gate,
        "state_level_attribution": attribution,
        "ci_source": "primary_fixed_panel_probe_bootstrap.csv",
        "panel_sensitivity_source": "panel_sensitivity_bootstrap.csv",
        "fashion_minus_mnist_state_x_probe_ms_ci95_low": float(heterogeneity_low),
        "fashion_minus_mnist_state_x_probe_ms_ci95_high": float(heterogeneity_high),
        "HI_task_localized_interaction_supported": bool(
            heterogeneity_low > 0.0 or heterogeneity_high < 0.0
        ),
        "H0_primary_state_category_fractions": categories,
        "primary_state_median_within_state_p4_cosine": float(
            primary_p4["pairwise_cosine_median"].median()
        ),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def summarize_primary_state_stability(state_summary: pd.DataFrame) -> pd.DataFrame:
    """Produce deterministic task/prefix summaries for the primary fixed-state panel."""
    required = (
        "panel",
        "task_name",
        "prefix",
        "state_position",
        "pairwise_cosine_median",
        "mean_gradient_norm_over_rms",
        "split_gate_pass",
    )
    _require_columns(state_summary, required, "state summary")
    primary = state_summary.loc[state_summary["panel"] == "primary"].copy()
    if primary.empty:
        raise ReviewError("state summary has no primary rows")
    primary["split_gate_bool"] = primary["split_gate_pass"].map(
        lambda value: _strict_bool(value, label="split_gate_pass")
    )
    rows = []
    for (task, prefix), group in primary.groupby(["task_name", "prefix"], sort=True):
        values = pd.to_numeric(group["pairwise_cosine_median"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ReviewError("primary state stability contains nonfinite cosines")
        rows.append(
            {
                "task_name": str(task),
                "prefix": int(prefix),
                "state_count": int(len(group)),
                "state_pairwise_cosine_median": float(np.median(values)),
                "state_pairwise_cosine_q10": float(np.quantile(values, 0.10)),
                "state_pairwise_cosine_q90": float(np.quantile(values, 0.90)),
                "conditional_mean_gate_pass_count": int(group["split_gate_bool"].sum()),
                "conditional_mean_gate_pass_fraction": float(group["split_gate_bool"].mean()),
                "mean_gradient_norm_over_rms_median": float(
                    pd.to_numeric(group["mean_gradient_norm_over_rms"]).median()
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(["task_name", "prefix"]).reset_index(drop=True)
    expected = set(itertools.product(TASKS, PREFIXES))
    observed = set(result[["task_name", "prefix"]].itertuples(index=False, name=None))
    if observed != expected or not result["state_count"].eq(16).all():
        raise ReviewError("primary stability summary is not balanced by frozen task/prefix")
    return result


def summarize_directional_components(
    delta: pd.DataFrame,
    primary_bootstrap: pd.DataFrame,
    *,
    vector_kind: str = "unit",
) -> pd.DataFrame:
    """Summarize fixed-panel state/probe components with common-probe bootstrap CIs."""
    point_rows = delta.loc[
        (delta["scope"] == "primary_mean_tasks")
        & (delta["vector_kind"] == vector_kind)
    ]
    if len(point_rows) != 1:
        raise ReviewError("directional component summary requires one primary point row")
    bootstrap = primary_bootstrap.loc[
        primary_bootstrap["vector_kind"] == vector_kind
    ]
    if bootstrap.empty:
        raise ReviewError("directional component summary has no matching bootstrap rows")
    point = point_rows.iloc[0]
    components = (
        ("V_probe", "Probe variation"),
        ("E_state_within", "State within-task"),
        ("E_task_panel", "Fixed task-panel contrast"),
        ("V_state_total", "State total"),
        ("Delta_total", "Delta total"),
    )
    rows = []
    for column, label in components:
        require_finite(bootstrap, (column,), "primary bootstrap")
        low, high = np.quantile(bootstrap[column].to_numpy(dtype=np.float64), [0.025, 0.975])
        rows.append(
            {
                "component": column,
                "label": label,
                "vector_kind": vector_kind,
                "point": float(point[column]),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def _symmetric_scalar_error(left: float, right: float) -> float:
    denominator = abs(float(left)) + abs(float(right))
    if denominator == 0.0:
        return 0.0
    return float(2.0 * abs(float(left) - float(right)) / denominator)


def summarize_bridge_diagnostics(bridge: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired B=128/full diagnostics without a population-transfer claim."""
    required = (
        "state_position",
        "task_name",
        "gradient_cosine_b128_to_full",
        "gradient_relative_error_b128_to_full",
        "gradient_norm_ratio_b128_to_full",
        "a_scalar_b128",
        "a_scalar_full",
    )
    _require_columns(bridge, required, "bridge metrics")
    require_finite(
        bridge,
        required[2:],
        "bridge metrics",
    )
    working = bridge.copy()
    working["scalar_symmetric_error"] = [
        _symmetric_scalar_error(left, right)
        for left, right in working[["a_scalar_b128", "a_scalar_full"]].itertuples(
            index=False, name=None
        )
    ]
    rows = []
    scopes = [(task, working.loc[working["task_name"] == task]) for task in TASKS]
    scopes.append(("all_bridge_states", working))
    for scope, group in scopes:
        if group.empty:
            raise ReviewError(f"bridge summary scope is empty: {scope}")
        rows.append(
            {
                "scope": scope,
                "state_count": int(group["state_position"].nunique()),
                "paired_draw_count": int(len(group)),
                "gradient_cosine_median": float(group["gradient_cosine_b128_to_full"].median()),
                "gradient_cosine_q10": float(group["gradient_cosine_b128_to_full"].quantile(0.10)),
                "gradient_cosine_q90": float(group["gradient_cosine_b128_to_full"].quantile(0.90)),
                "gradient_relative_error_median": float(
                    group["gradient_relative_error_b128_to_full"].median()
                ),
                "gradient_relative_error_q90": float(
                    group["gradient_relative_error_b128_to_full"].quantile(0.90)
                ),
                "gradient_norm_ratio_median": float(
                    group["gradient_norm_ratio_b128_to_full"].median()
                ),
                "gradient_norm_ratio_q10": float(
                    group["gradient_norm_ratio_b128_to_full"].quantile(0.10)
                ),
                "gradient_norm_ratio_q90": float(
                    group["gradient_norm_ratio_b128_to_full"].quantile(0.90)
                ),
                "scalar_symmetric_error_median": float(group["scalar_symmetric_error"].median()),
                "scalar_symmetric_error_q90": float(group["scalar_symmetric_error"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _plot_state_stability(state_summary: pd.DataFrame, path: Path) -> None:
    primary = state_summary.loc[
        (state_summary["panel"] == "primary") & (state_summary["prefix"] == 4)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    colors = {True: "#2f7d58", False: "#c34a36"}
    for axis, task in zip(axes, TASKS, strict=True):
        group = primary.loc[primary["task_name"] == task].sort_values("state_position")
        x = np.arange(1, len(group) + 1)
        median = group["pairwise_cosine_median"].to_numpy(dtype=float)
        low = group["pairwise_cosine_q10"].to_numpy(dtype=float)
        high = group["pairwise_cosine_q90"].to_numpy(dtype=float)
        passed = [
            _strict_bool(value, label="split_gate_pass")
            for value in group["split_gate_pass"]
        ]
        axis.vlines(x, low, high, color="#9aa0a6", linewidth=1.3, zorder=1)
        for passed_value in (True, False):
            mask = np.asarray([value is passed_value for value in passed])
            axis.scatter(
                x[mask],
                median[mask],
                s=42,
                color=colors[passed_value],
                edgecolor="white",
                linewidth=0.6,
                label="gate pass" if passed_value else "gate fail",
                zorder=2,
            )
        axis.axhline(0.80, color="#262626", linestyle="--", linewidth=1.1, label="0.80 gate")
        axis.axhline(0.0, color="#c7c7c7", linewidth=0.8)
        axis.set_title(
            f"{TASK_LABELS[task]}: {sum(passed)}/16 conditional-mean gates pass",
            fontsize=11,
        )
        axis.set_xlabel("Frozen primary state (within task)")
        axis.set_xticks([1, 4, 8, 12, 16])
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("P=4 draw-pair cosine (median; q10-q90)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Fixed-state full-CE probe-direction stability", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _atomic_savefig(fig, path)


def _plot_directional_components(
    components: pd.DataFrame, decision: Mapping[str, Any], path: Path
) -> None:
    colors = ["#157a8a", "#d4772b", "#5675a8", "#655d8a", "#b33a3a"]
    x = np.arange(len(components))
    point = components["point"].to_numpy(dtype=float)
    low = components["ci95_low"].to_numpy(dtype=float)
    high = components["ci95_high"].to_numpy(dtype=float)
    error = np.vstack([point - low, high - point])
    fig, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.axhline(0.0, color="#262626", linewidth=1.0)
    for index in range(len(components)):
        axis.errorbar(
            x[index],
            point[index],
            yerr=error[:, index : index + 1],
            fmt="o",
            markersize=8,
            capsize=5,
            color=colors[index],
            linewidth=2,
        )
    axis.set_xticks(x, components["label"].tolist())
    axis.set_ylabel("Unit-direction variance component")
    axis.grid(axis="y", alpha=0.22)
    axis.set_title(
        "Directional probe versus state components\n"
        f"Frozen decision: {decision['decision']}",
        fontsize=13,
        fontweight="bold",
    )
    axis.text(
        0.01,
        0.02,
        "Points: fixed 32-state primary bank. Intervals: common-probe bootstrap (95%).",
        transform=axis.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    _atomic_savefig(fig, path)


def _plot_prefix_and_bridge(
    state_summary: pd.DataFrame, bridge: pd.DataFrame, path: Path
) -> None:
    primary = state_summary.loc[state_summary["panel"] == "primary"].copy()
    primary["gate"] = primary["split_gate_pass"].map(
        lambda value: _strict_bool(value, label="split_gate_pass")
    )
    colors = {"fashion_mnist": "#157a8a", "mnist": "#d4772b"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    axis = axes[0, 0]
    positions = []
    values = []
    labels = []
    box_colors = []
    position = 1
    for prefix in PREFIXES:
        for task in TASKS:
            group = primary.loc[
                (primary["prefix"] == prefix) & (primary["task_name"] == task),
                "pairwise_cosine_median",
            ].to_numpy(dtype=float)
            values.append(group)
            positions.append(position)
            labels.append(f"P{prefix}\n{TASK_LABELS[task]}")
            box_colors.append(colors[task])
            position += 1
        position += 0.45
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "white", "linewidth": 1.5},
    )
    for patch, color in zip(boxes["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axis.axhline(0.80, color="#262626", linestyle="--", linewidth=1.0)
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylabel("State median draw-pair cosine")
    axis.set_title("A. Stability across literal P-prefixes")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[0, 1]
    for task in TASKS:
        grouped = (
            primary.loc[primary["task_name"] == task]
            .groupby("prefix")["gate"]
            .mean()
            .reindex(PREFIXES)
        )
        axis.plot(
            PREFIXES,
            grouped,
            marker="o",
            linewidth=2,
            color=colors[task],
            label=TASK_LABELS[task],
        )
    axis.set_xticks(PREFIXES)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Probe-pair prefix P")
    axis.set_ylabel("Conditional-mean gate pass fraction")
    axis.set_title("B. Frozen split-gate pass fraction")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    axis = axes[1, 0]
    bridge_values = [
        bridge.loc[
            bridge["task_name"] == task, "gradient_cosine_b128_to_full"
        ].to_numpy(dtype=float)
        for task in TASKS
    ]
    boxes = axis.boxplot(
        bridge_values,
        labels=[TASK_LABELS[task] for task in TASKS],
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "white", "linewidth": 1.5},
    )
    for patch, task in zip(boxes["boxes"], TASKS, strict=True):
        patch.set_facecolor(colors[task])
        patch.set_alpha(0.85)
    axis.axhline(0.80, color="#262626", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Paired gradient cosine, B=128 to full CE")
    axis.set_title("C. Matched bridge direction")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    for task in TASKS:
        group = bridge.loc[bridge["task_name"] == task]
        axis.scatter(
            group["gradient_norm_ratio_b128_to_full"],
            group["gradient_relative_error_b128_to_full"],
            s=22,
            alpha=0.7,
            color=colors[task],
            label=TASK_LABELS[task],
        )
    axis.axvline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Gradient norm ratio B=128/full")
    axis.set_ylabel("Relative gradient error B=128/full")
    axis.set_title("D. Matched bridge norm and error")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    fig.suptitle(
        "P-prefix and descriptive B=128/full bridge diagnostics",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _atomic_savefig(fig, path)


def _atomic_savefig(fig: plt.Figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)
    os.replace(temporary, path)


def _format_number(value: float) -> str:
    number = float(value)
    if abs(number) >= 1000 or (number != 0.0 and abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.4f}"


def _result_markdown(
    decision: Mapping[str, Any],
    stability: pd.DataFrame,
    bridge: pd.DataFrame,
    sentinel_persistence: pd.DataFrame,
) -> str:
    p4 = stability.loc[stability["prefix"] == 4].set_index("task_name")
    bridge_by_task = bridge.loc[bridge["scope"].isin(TASKS)].set_index("scope")
    sentinel = sentinel_persistence.loc[
        (sentinel_persistence["scope"] == "all_selected_sentinels")
        & (sentinel_persistence["fresh_metric"] == "fresh_atomic_gradient_rms")
    ].set_index("task_name")
    interaction = (
        "supported within this fixed bank"
        if decision["HI_task_localized_interaction_supported"]
        else "not established within this fixed bank"
    )
    lines = [
        "# Fixed-state crossed-probe post-run review",
        "",
        "Status: **PASS**. Executor validity, frozen provenance, exact cardinalities, "
        "Gram-based summaries, bootstrap algebra, gates, and the decision were independently checked.",
        "",
        "## Primary result",
        "",
        f"Frozen bank-conditional decision: **{decision['decision']}**. "
        f"Unit-direction `Delta_total` is {_format_number(decision['delta_total_dir_point'])} "
        f"with common-probe 95% CI "
        f"[{_format_number(decision['delta_total_dir_ci95_low'])}, "
        f"{_format_number(decision['delta_total_dir_ci95_high'])}].",
        "",
        f"State-level attribution: `{decision['state_level_attribution']}`. "
        f"The task-conditioned state-by-probe interaction contrast is {interaction}; "
        "the task panels are nested and do not identify task identity as a cause.",
        "",
        "## Fixed-state diagnostics",
        "",
        "| Task | P=4 state cosine median | Conditional-mean gates |",
        "|---|---:|---:|",
    ]
    for task in TASKS:
        row = p4.loc[task]
        lines.append(
            f"| {TASK_LABELS[task]} | {_format_number(row['state_pairwise_cosine_median'])} "
            f"| {int(row['conditional_mean_gate_pass_count'])}/16 |"
        )
    lines.extend(
        [
            "",
            "| Bridge task | B=128/full cosine median | Relative error median | Norm ratio median |",
            "|---|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        row = bridge_by_task.loc[task]
        lines.append(
            f"| {TASK_LABELS[task]} | {_format_number(row['gradient_cosine_median'])} "
            f"| {_format_number(row['gradient_relative_error_median'])} "
            f"| {_format_number(row['gradient_norm_ratio_median'])} |"
        )
    lines.extend(
        [
            "",
            "The bridge is a matched descriptive perturbation only; it is not a finite-B population transfer result.",
            "",
            "Sentinel persistence is outcome-selected and descriptive. Spearman correlation between prior "
            "and fresh atomic-gradient RMS is "
            + ", ".join(
                f"{TASK_LABELS[task]}={_format_number(sentinel.loc[task, 'spearman_to_prior_grad_rms'])}"
                for task in TASKS
            )
            + "; it does not estimate extreme-state prevalence.",
            "",
            "## Interpretation boundary",
            "",
            f"`{INTERPRETATION_BOUNDARY}`.",
            "",
            "This review does **not** claim that probes, states, task identity, or the measured estimator "
            "components cause downstream behavior, and it does not claim that increasing P fixes training.",
            "",
            "## Artifacts",
            "",
            "- `posthoc_validation.json`",
            "- `artifact_hashes.json`",
            "- `primary_state_fixed_probe_stability_by_task.png`",
            "- `directional_state_vs_probe_variance.png`",
            "- `prefix_and_bridge_diagnostics.png`",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_hashes(
    input_dir: Path,
    source_details: Mapping[str, Mapping[str, Any]],
    reviewer_details: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in sorted(EXPECTED_EXECUTOR_OUTPUTS.union(GENERATED_FILES)):
        path = input_dir / name
        if not path.is_file():
            raise ReviewError(f"cannot hash missing reviewed artifact: {path}")
        files[name] = {
            "role": "review_output" if name in GENERATED_FILES else "executor_input",
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    return {
        "algorithm": "sha256",
        "self_hash_excluded": "artifact_hashes.json",
        "files": files,
        "frozen_sources": dict(source_details),
        "post_run_reviewer": dict(reviewer_details),
    }


def _review(input_dir: Path, *, verbose: bool) -> dict[str, Path]:
    input_dir = input_dir.expanduser().resolve()
    reporter = Reporter(verbose)
    reporter.stage(
        f"start input_dir={input_dir} protocol_id={PROTOCOL_ID} seed={BASE_SEED} "
        "device=cpu reviewer_dtype=float64 cache_mode=read_only"
    )
    if not input_dir.is_dir():
        raise ReviewError(f"production output directory does not exist: {input_dir}")
    ledger = ValidationLedger()
    reviewer_paths = {
        "reviewer": Path(__file__).resolve(),
        "reviewer_tests": (
            ROOT / "tests/variant_a_fixed_state_probe_variance_review_test.py"
        ),
    }
    reviewer_details = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in reviewer_paths.items()
    }
    ledger.add("post_run_reviewer_code_identity", reviewer_details)

    reporter.stage("stage=load_manifest_validity_config_decision")
    manifest = _read_json(input_dir / "manifest.json")
    resolved = _read_json(input_dir / "resolved_config.json")
    validity = _read_json(input_dir / "validity.json")
    decision = _read_json(input_dir / "decision.json")
    preflight = _read_json(input_dir / "preflight.json")
    manifest_detail = _validate_manifest_and_config(
        input_dir, manifest, resolved, validity, decision, preflight
    )
    ledger.add("manifest_validity_and_resolved_config", manifest_detail)

    reporter.stage("stage=frozen_executor_and_protocol_source_hashes")
    source_details, state_bank, accepted_active = _validate_provenance(resolved, reporter)
    ledger.add(
        "executor_sha_and_protocol_source_hashes",
        {"hashed_sources": len(source_details), "executor_sha256": EXPECTED_EXECUTOR_SHA256},
    )

    reporter.stage("stage=load_tables_and_validate_exact_cardinalities")
    frames = _normalize_frames(_load_frames(input_dir))
    counts = _validate_frame_contracts(frames, state_bank)
    active_detail = _validate_active_manifest(frames["active"], accepted_active, resolved)
    ledger.add("schema_key_grids_finiteness_and_active_manifest", {**counts, **active_detail})

    reporter.stage("stage=common_probe_and_b128_batch_replay")
    replay_detail = _validate_common_probe_replay(
        frames["atomic"], frames["bridge_atomic"], state_bank
    )
    ledger.add("common_probe_and_bridge_batch_replay", replay_detail)

    reporter.stage("stage=load_gram_archives")
    archives = {
        key: _load_archive(input_dir / name) for key, name in ARCHIVE_FILES.items()
    }
    archive_identity = _validate_task_state_archive_identity(archives)
    ledger.add("gram_archive_cross_file_identity", archive_identity)

    reporter.stage("stage=recompute_state_prefix_summaries")
    state_detail = _validate_state_grams_and_summaries(
        archives, frames["state_summary"], state_bank
    )
    ledger.add("state_prefix_summary_independent_recomputation", state_detail)

    reporter.stage("stage=recompute_delta_and_crossed_anova")
    reviewed_delta, _reviewed_anova, delta_detail = _recompute_delta_and_anova(
        archives, frames["delta"], frames["anova"]
    )
    ledger.add("delta_and_anova_independent_recomputation", delta_detail)

    reporter.stage("stage=recompute_bridge_metrics")
    bridge_detail = _validate_bridge_metrics(
        archives,
        frames["bridge"],
        frames["bridge_atomic"],
        frames["prefix_scalar"],
        state_bank,
    )
    ledger.add("bridge_metric_independent_recomputation", bridge_detail)

    reporter.stage("stage=regenerate_bootstraps_from_grams_and_frozen_seeds")
    bootstrap_detail, reviewed_primary_bootstrap, _reviewed_panel_bootstrap = (
        validate_bootstrap_contract(
            frames["primary_bootstrap"],
            frames["panel_bootstrap"],
            archives["task_grams"],
            archives["cross_task"],
        )
    )
    ledger.add("bootstrap_exact_gram_and_seed_recomputation", bootstrap_detail)

    reporter.stage("stage=regenerate_signal_bootstraps_from_state_grams")
    signal_detail, reviewed_signal = validate_signal_bootstrap(
        frames["signal"], archives["state_grams"], state_bank
    )
    ledger.add("signal_bootstrap_exact_gram_and_seed_recomputation", signal_detail)

    reporter.stage("stage=recompute_nonalignment_gates")
    nonalignment_detail = _validate_nonalignment(
        archives,
        frames["state_summary"],
        frames["crossfit"],
        frames["task_crossfit"],
        frames["nonalignment"],
    )
    ledger.add("nonalignment_gate_independent_recomputation", nonalignment_detail)

    reporter.stage("stage=recompute_sentinel_summaries")
    sentinel_detail = _validate_sentinel_outputs(
        frames["panel"],
        frames["sentinel_stratum"],
        frames["sentinel_persistence"],
        frames["atomic"],
        frames["state_summary"],
        archives["state_grams"],
        state_bank,
    )
    ledger.add("sentinel_summary_independent_recomputation", sentinel_detail)

    reporter.stage("stage=recompute_frozen_decision")
    reviewed_decision = derive_reviewed_decision(
        reviewed_delta,
        reviewed_primary_bootstrap,
        frames["state_summary"],
        reviewed_signal,
        nonalignment_detail,
    )
    decision_delta = _compare_json(decision, reviewed_decision, label="decision.json")
    ledger.add(
        "decision_independent_recomputation",
        {"decision": reviewed_decision["decision"], "maximum_numeric_delta": decision_delta},
    )
    if not ledger.passed:
        raise ReviewError("blocking post-run validation failed")

    reporter.stage("stage=summarize_reviewed_results")
    stability_summary = summarize_primary_state_stability(frames["state_summary"])
    directional_summary = summarize_directional_components(
        reviewed_delta, reviewed_primary_bootstrap, vector_kind="unit"
    )
    bridge_summary = summarize_bridge_diagnostics(frames["bridge"])

    outputs = {
        "state_plot": input_dir / "primary_state_fixed_probe_stability_by_task.png",
        "directional_plot": input_dir / "directional_state_vs_probe_variance.png",
        "bridge_plot": input_dir / "prefix_and_bridge_diagnostics.png",
        "result": input_dir / "result.md",
        "validation": input_dir / "posthoc_validation.json",
        "hashes": input_dir / "artifact_hashes.json",
    }
    reporter.stage("stage=render_readable_pngs")
    _plot_state_stability(frames["state_summary"], outputs["state_plot"])
    reporter.artifact(outputs["state_plot"])
    _plot_directional_components(
        directional_summary, reviewed_decision, outputs["directional_plot"]
    )
    reporter.artifact(outputs["directional_plot"])
    _plot_prefix_and_bridge(
        frames["state_summary"], frames["bridge"], outputs["bridge_plot"]
    )
    reporter.artifact(outputs["bridge_plot"])

    reporter.stage("stage=write_result_and_posthoc_validation")
    result_text = _result_markdown(
        reviewed_decision,
        stability_summary,
        bridge_summary,
        frames["sentinel_persistence"],
    )
    _atomic_write_text(outputs["result"], result_text)
    reporter.artifact(outputs["result"])
    posthoc = {
        "protocol_id": PROTOCOL_ID,
        "status": "valid",
        "passed": True,
        "input_dir": str(input_dir),
        "reviewer": str(Path(__file__).resolve()),
        "reviewer_provenance": reviewer_details,
        "checks": ledger.checks,
        "counts": counts,
        "reviewed_decision": reviewed_decision,
        "summaries": {
            "primary_state_stability": stability_summary.to_dict(orient="records"),
            "directional_components": directional_summary.to_dict(orient="records"),
            "bridge": bridge_summary.to_dict(orient="records"),
        },
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "scientific_outputs_require_passed_true": True,
        "artifacts": [path.name for path in outputs.values()],
    }
    _atomic_write_json(outputs["validation"], posthoc)
    reporter.artifact(outputs["validation"])

    reporter.stage("stage=hash_executor_inputs_sources_and_review_outputs")
    hashes = _artifact_hashes(input_dir, source_details, reviewer_details)
    _atomic_write_json(outputs["hashes"], hashes)
    reporter.artifact(outputs["hashes"])
    reporter.stage(
        f"done passed=true decision={reviewed_decision['decision']} "
        f"outputs={','.join(str(path) for path in outputs.values())}"
    )
    return outputs


def review(input_dir: Path, *, verbose: bool = True) -> dict[str, Path]:
    resolved = input_dir.expanduser().resolve()
    try:
        return _review(resolved, verbose=verbose)
    except Exception as error:
        if resolved.is_dir():
            failure = {
                "protocol_id": PROTOCOL_ID,
                "status": "invalid",
                "passed": False,
                "input_dir": str(resolved),
                "error_type": type(error).__name__,
                "error": str(error),
                "interpretation_emitted": False,
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
            _atomic_write_json(resolved / "posthoc_validation.json", failure)
        if isinstance(error, ReviewError):
            raise
        raise ReviewError(f"unexpected post-run review failure: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed deterministic post-run review of the fixed-state crossed-probe audit."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    review(args.input_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
