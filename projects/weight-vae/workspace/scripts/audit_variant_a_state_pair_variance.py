from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import config_hash, torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import encode_weights
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_POOL_SHA256,
    EXPECTED_RECORDS_SHA256,
    PROTOCOL_ID as PRIMARY_PROTOCOL_ID,
    ROOT,
    _atomic_pair_loss,
    _flatten_active_gradients,
    _indexed_generators,
    _load_run,
    _preflight_decomposition,
    _probe_cfg,
    _sample_state_indices,
    git_metadata,
    sha256_file,
    vector_norm,
)


PROTOCOL_ID = "a_estimator_state_pair_variance_h2048_v1"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_state_pair_variance_h2048"
)
PROTOCOL_PATH = DEFAULT_OUTPUT_DIR / "protocol.md"
PRIMARY_DIR = DEFAULT_OUTPUT_DIR.parent / "a_estimator_stability_unfinetuned_h2048"
STATE_COUNT = 32
PAIR_COUNT = 4
REPEAT_COUNT = 12
PAIR_PREFIXES = (1, 2, 4)
BASE_SEED = 20260714
EXPECTED_CONFIG_HASH = "4fc87a54349a39a2"
EXPECTED_CONFIG_FILE_SHA256 = "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4"
EXPECTED_CONFIG_SOURCE_SHA256 = "aa71d2909f4ac7cde9b7f6fdb2d0caff8bfd3795f2fdea94c1be3e2fe853675e"
EXPECTED_MODEL_STATE_SHA256 = "84bf2d5f050ac21f7bf402d478b73d9515e4a2477c4094eeb357fa140b69b8e1"
EXPECTED_PRIMARY_ARTIFACT_SHA256 = {
    "resolved_config.json": "615d194be084c1cae77e88e57914d3726e24f0c754ce9e4a355c7154a4498311",
    "posthoc_validity.json": "862964f56991117ca19b636e869f892c17fdf568ded85e937abad54dce4b7e2e",
    "active_decoder_parameters.json": "d674f6ff423dc8f4375feb7cb4f33acc68c5dd4db448d0832816fe7478237a2a",
    "atomic_pair_samples.csv": "1391b00129632010ef4195aa2e51068532dfe7b424ca84c2916a0f07874e637e",
    "pairwise_gradient_cosines.csv": "24e1e196e04600344b98e8da9e6431aee0d831e7861d68ed2fae1f8c0e315162",
}
SPLITS = (
    ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)),
    ((0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11)),
    ((0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11)),
)


@dataclass(slots=True)
class Sufficient:
    state_count: int
    state_mean_sum: torch.Tensor
    state_mean_squared_norm_sum: float
    within_residual_sum: float
    within_degrees_freedom: int


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def relative_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs(value - reference) / abs(reference)


def _new_sufficient(dimension: int) -> Sufficient:
    return Sufficient(
        state_count=0,
        state_mean_sum=torch.zeros(dimension, dtype=torch.float32),
        state_mean_squared_norm_sum=0.0,
        within_residual_sum=0.0,
        within_degrees_freedom=0,
    )


def _sufficient_to_payload(sufficient: Sufficient) -> dict[str, Any]:
    return {
        "state_count": sufficient.state_count,
        "state_mean_sum": sufficient.state_mean_sum,
        "state_mean_squared_norm_sum": sufficient.state_mean_squared_norm_sum,
        "within_residual_sum": sufficient.within_residual_sum,
        "within_degrees_freedom": sufficient.within_degrees_freedom,
    }


def _sufficient_from_payload(payload: Mapping[str, Any]) -> Sufficient:
    return Sufficient(
        state_count=int(payload["state_count"]),
        state_mean_sum=payload["state_mean_sum"].detach().cpu().float(),
        state_mean_squared_norm_sum=float(payload["state_mean_squared_norm_sum"]),
        within_residual_sum=float(payload["within_residual_sum"]),
        within_degrees_freedom=int(payload["within_degrees_freedom"]),
    )


def _sufficient_is_finite(sufficient: Sufficient) -> bool:
    return bool(
        torch.isfinite(sufficient.state_mean_sum).all().item()
        and math.isfinite(sufficient.state_mean_squared_norm_sum)
        and math.isfinite(sufficient.within_residual_sum)
    )


def _merge_sufficient(target: Sufficient, source: Sufficient) -> None:
    target.state_count += int(source.state_count)
    target.state_mean_sum.add_(source.state_mean_sum)
    target.state_mean_squared_norm_sum += float(source.state_mean_squared_norm_sum)
    target.within_residual_sum += float(source.within_residual_sum)
    target.within_degrees_freedom += int(source.within_degrees_freedom)


def variance_components_from_sufficient(
    sufficient: Sufficient,
    *,
    pair_count: int,
) -> dict[str, float | int]:
    if sufficient.state_count < 2 or sufficient.within_degrees_freedom <= 0:
        raise ValueError("variance components require at least two states and positive within degrees of freedom")
    if not _sufficient_is_finite(sufficient):
        raise FloatingPointError("nonfinite directional sufficient statistic")
    state_count = int(sufficient.state_count)
    mean_sum_norm_squared = vector_norm(sufficient.state_mean_sum) ** 2
    centered_state_sum = float(
        sufficient.state_mean_squared_norm_sum - mean_sum_norm_squared / float(state_count)
    )
    centered_tolerance = 1e-8 * max(float(sufficient.state_mean_squared_norm_sum), 1.0)
    if centered_state_sum < -centered_tolerance:
        raise FloatingPointError(f"negative centered state sum: {centered_state_sum}")
    centered_state_sum = max(centered_state_sum, 0.0)
    within_trace = float(sufficient.within_residual_sum / sufficient.within_degrees_freedom)
    state_mean_trace = float(centered_state_sum / float(state_count - 1))
    between_trace = float(state_mean_trace - within_trace / float(pair_count))
    pair_contribution_at_p = float(within_trace / float(pair_count))
    ratio = between_trace / pair_contribution_at_p if pair_contribution_at_p > 0.0 else float("nan")
    result = {
        "state_count": state_count,
        "pair_count": int(pair_count),
        "within_degrees_freedom": int(sufficient.within_degrees_freedom),
        "within_pair_covariance_trace_W": within_trace,
        "state_mean_covariance_trace_V": state_mean_trace,
        "between_state_covariance_trace_B_raw": between_trace,
        "within_pair_contribution_at_P_W_over_P": pair_contribution_at_p,
        "between_to_pair_contribution_ratio": ratio,
        "state_axis_dominates_at_P": bool(between_trace > pair_contribution_at_p),
    }
    numeric = [float(value) for key, value in result.items() if key not in {"state_axis_dominates_at_P"}]
    if not all(math.isfinite(value) for value in numeric):
        raise FloatingPointError("nonfinite variance component")
    return result


def _group_ids_for_repeat(repeat: int) -> list[str]:
    groups = ["full"]
    for split_index, (left, right) in enumerate(SPLITS, start=1):
        if repeat in left:
            groups.append(f"split{split_index}_left")
        elif repeat in right:
            groups.append(f"split{split_index}_right")
        else:
            raise ValueError(f"repeat {repeat} missing from split {split_index}")
    return groups


def _chunked_gram(vectors: Sequence[torch.Tensor], *, chunk_size: int = 262_144) -> np.ndarray:
    if not vectors:
        raise ValueError("cannot form an empty Gram matrix")
    dimension = int(vectors[0].numel())
    if any(int(vector.numel()) != dimension for vector in vectors):
        raise ValueError("Gram vectors have inconsistent dimensions")
    gram = torch.zeros((len(vectors), len(vectors)), dtype=torch.float64)
    for start in range(0, dimension, int(chunk_size)):
        stop = min(start + int(chunk_size), dimension)
        block = torch.stack([vector[start:stop] for vector in vectors], dim=0).double()
        gram.add_(block @ block.T)
    result = gram.numpy()
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("nonfinite repeat-prefix Gram matrix")
    return result


def _pairwise_from_gram(gram: np.ndarray, *, pair_count: int) -> pd.DataFrame:
    norms = np.sqrt(np.diag(gram))
    rows: list[dict[str, float | int]] = []
    for left in range(gram.shape[0]):
        for right in range(left + 1, gram.shape[0]):
            denominator = float(norms[left] * norms[right])
            cosine = float(gram[left, right] / denominator) if denominator > 0.0 else float("nan")
            rows.append(
                {
                    "states": STATE_COUNT,
                    "pairs": int(pair_count),
                    "left_repeat": left,
                    "right_repeat": right,
                    "gradient_cosine": cosine,
                    "gradient_norm_ratio": float(norms[left] / norms[right]) if norms[right] > 0.0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _quantile(values: Iterable[float], probability: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError("nonfinite value in quantile input")
    return float(np.quantile(array, probability, method="linear"))


def _load_primary_subset(primary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    atomic = pd.read_csv(primary_dir / "atomic_pair_samples.csv")
    atomic = atomic.loc[(atomic["state_position"] < STATE_COUNT) & (atomic["pair_position"] < PAIR_COUNT)].copy()
    atomic = atomic.set_index(["repeat", "state_position", "pair_position"]).sort_index()
    if not atomic.index.is_unique:
        raise ValueError("primary atomic subset has duplicate keys")
    expected = REPEAT_COUNT * STATE_COUNT * PAIR_COUNT
    if len(atomic) != expected:
        raise ValueError(f"primary subset has {len(atomic)} rows, expected {expected}")
    _require_finite_frame(
        atomic.reset_index(),
        (
            "repeat",
            "state_position",
            "pair_position",
            "source_weight_index",
            "probe_seed_1",
            "probe_seed_2",
            "a_scalar",
            "gradient_norm",
        ),
        label="primary atomic subset",
    )
    pairwise = pd.read_csv(primary_dir / "pairwise_gradient_cosines.csv")
    pairwise = pairwise.loc[
        (pairwise["states"] == STATE_COUNT) & pairwise["pairs"].isin(PAIR_PREFIXES)
    ].copy()
    _require_finite_frame(
        pairwise,
        ("states", "pairs", "left_repeat", "right_repeat", "gradient_cosine", "gradient_norm_ratio"),
        label="primary pairwise subset",
    )
    return atomic, pairwise


def _primary_active_identity(primary_dir: Path) -> dict[str, Any]:
    return json.loads((primary_dir / "active_decoder_parameters.json").read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")


def _require_finite_frame(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise FloatingPointError(f"{label} has nonfinite numeric values")


def _tensor_relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = max(vector_norm(reference), 1.0)
    return vector_norm(value - reference) / denominator


def _validate_shard_payload(
    payload: Mapping[str, Any],
    *,
    shard_identity: Mapping[str, Any],
    primary_atomic: pd.DataFrame,
    repeat: int,
    dimension: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Sufficient], dict[str, float]]:
    if payload.get("shard_identity") != shard_identity or int(payload.get("repeat", -1)) != repeat:
        raise ValueError(f"resume shard identity mismatch for repeat {repeat}")
    atomic_rows = list(payload["atomic_rows"])
    state_rows = list(payload["state_rows"])
    atomic = pd.DataFrame(atomic_rows)
    states = pd.DataFrame(state_rows)
    expected_atomic_keys = {
        (repeat, state_position, pair_position)
        for state_position in range(STATE_COUNT)
        for pair_position in range(PAIR_COUNT)
    }
    observed_atomic_keys = set(
        atomic[["repeat", "state_position", "pair_position"]].itertuples(index=False, name=None)
    )
    expected_state_keys = {(repeat, state_position) for state_position in range(STATE_COUNT)}
    observed_state_keys = set(states[["repeat", "state_position"]].itertuples(index=False, name=None))
    if len(atomic) != len(expected_atomic_keys) or observed_atomic_keys != expected_atomic_keys:
        raise ValueError(f"resume shard atomic keys mismatch for repeat {repeat}")
    if len(states) != len(expected_state_keys) or observed_state_keys != expected_state_keys:
        raise ValueError(f"resume shard state keys mismatch for repeat {repeat}")
    _require_finite_frame(
        atomic,
        (
            "repeat",
            "state_position",
            "pair_position",
            "source_weight_index",
            "tau",
            "state_draw_seed",
            "train_position",
            "probe_seed_1",
            "probe_seed_2",
            "a_scalar",
            "primary_a_scalar",
            "scalar_abs_error",
            "gradient_norm",
            "primary_gradient_norm",
            "gradient_norm_relative_error",
            "h1_norm",
            "h2_norm",
        ),
        label=f"resume atomic rows repeat {repeat}",
    )
    _require_finite_frame(
        states,
        (
            "repeat",
            "state_position",
            "source_weight_index",
            "tau",
            "state_mean_a_p1",
            "state_mean_a_p2",
            "state_mean_a_p4",
            "state_mean_gradient_norm_p4",
            "state_mean_squared_gradient_norm_p4",
            "within_residual_sum",
            "within_residual_sum_raw",
            "normalized_within_residual_sum_raw",
            "atomic_squared_gradient_norm_sum",
        ),
        label=f"resume state rows repeat {repeat}",
    )
    if not set(atomic["task_name"].astype(str)).issubset({"mnist", "fashion_mnist"}):
        raise ValueError(f"resume shard has unexpected atomic task for repeat {repeat}")
    if not set(states["task_name"].astype(str)).issubset({"mnist", "fashion_mnist"}):
        raise ValueError(f"resume shard has unexpected state task for repeat {repeat}")

    primary_repeat = primary_atomic.loc[repeat].reset_index()
    primary_columns = [
        "state_position",
        "pair_position",
        "source_weight_index",
        "task_name",
        "tau",
        "state_draw_seed",
        "train_position",
        "probe_seed_1",
        "probe_seed_2",
        "a_scalar",
        "gradient_norm",
    ]
    replay_rows = atomic.merge(
        primary_repeat[primary_columns],
        on=["state_position", "pair_position"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_source"),
    )
    if len(replay_rows) != STATE_COUNT * PAIR_COUNT or not bool((replay_rows["_merge"] == "both").all()):
        raise ValueError(f"resume shard primary keys mismatch for repeat {repeat}")
    exact_columns = (
        "source_weight_index",
        "task_name",
        "state_draw_seed",
        "train_position",
        "probe_seed_1",
        "probe_seed_2",
    )
    for column in exact_columns:
        if not bool((replay_rows[column] == replay_rows[f"{column}_source"]).all()):
            raise ValueError(f"resume shard primary {column} mismatch for repeat {repeat}")
    if not bool(
        np.isclose(
            replay_rows["tau"].to_numpy(dtype=np.float64),
            replay_rows["tau_source"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ).all()
    ):
        raise ValueError(f"resume shard primary tau mismatch for repeat {repeat}")
    expected_scalar_error = np.abs(
        replay_rows["a_scalar"].to_numpy(dtype=np.float64)
        - replay_rows["a_scalar_source"].to_numpy(dtype=np.float64)
    )
    source_gradient_norm = replay_rows["gradient_norm_source"].to_numpy(dtype=np.float64)
    expected_norm_error = np.abs(
        replay_rows["gradient_norm"].to_numpy(dtype=np.float64) - source_gradient_norm
    ) / np.abs(source_gradient_norm)
    replay_checks = (
        np.isclose(
            replay_rows["primary_a_scalar"].to_numpy(dtype=np.float64),
            replay_rows["a_scalar_source"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ).all()
        and np.isclose(
            replay_rows["primary_gradient_norm"].to_numpy(dtype=np.float64),
            source_gradient_norm,
            rtol=0.0,
            atol=1e-12,
        ).all()
        and np.isclose(
            replay_rows["scalar_abs_error"].to_numpy(dtype=np.float64),
            expected_scalar_error,
            rtol=0.0,
            atol=1e-12,
        ).all()
        and np.isclose(
            replay_rows["gradient_norm_relative_error"].to_numpy(dtype=np.float64),
            expected_norm_error,
            rtol=0.0,
            atol=1e-12,
        ).all()
    )
    if not bool(replay_checks):
        raise ValueError(f"resume shard primary replay values mismatch for repeat {repeat}")

    atomic_squared_by_state = (
        atomic.assign(_gradient_norm_squared=atomic["gradient_norm"].astype(float) ** 2)
        .groupby("state_position", sort=True)["_gradient_norm_squared"]
        .sum()
        .reindex(states.sort_values("state_position")["state_position"])
        .to_numpy(dtype=np.float64)
    )
    states = states.sort_values("state_position").reset_index(drop=True)
    if not bool(
        np.isclose(
            states["atomic_squared_gradient_norm_sum"].to_numpy(dtype=np.float64),
            atomic_squared_by_state,
            rtol=2e-7,
            atol=1e-6,
        ).all()
    ):
        raise ValueError(f"resume shard atomic squared-norm sum mismatch for repeat {repeat}")
    expected_raw_within = (
        atomic_squared_by_state
        - PAIR_COUNT * states["state_mean_squared_gradient_norm_p4"].to_numpy(dtype=np.float64)
    )
    expected_normalized_within = expected_raw_within / np.maximum(
        atomic_squared_by_state, 1.0
    )
    expected_clipped_within = np.maximum(expected_raw_within, 0.0)
    within_identity_passed = (
        np.isclose(
            states["within_residual_sum_raw"].to_numpy(dtype=np.float64),
            expected_raw_within,
            rtol=2e-7,
            atol=1e-6,
        ).all()
        and np.isclose(
            states["normalized_within_residual_sum_raw"].to_numpy(dtype=np.float64),
            expected_normalized_within,
            rtol=2e-7,
            atol=1e-9,
        ).all()
        and np.isclose(
            states["within_residual_sum"].to_numpy(dtype=np.float64),
            expected_clipped_within,
            rtol=2e-7,
            atol=1e-6,
        ).all()
    )
    if not bool(within_identity_passed):
        raise ValueError(f"resume shard within-state identity mismatch for repeat {repeat}")

    replay_observed = {
        "max_scalar_abs_error": float(atomic["scalar_abs_error"].max()),
        "max_gradient_norm_relative_error": float(atomic["gradient_norm_relative_error"].max()),
        "minimum_raw_within_residual": float(states["within_residual_sum_raw"].min()),
        "minimum_normalized_within_residual": float(states["normalized_within_residual_sum_raw"].min()),
    }
    replay_recorded = {key: float(value) for key, value in payload["replay_summary"].items()}
    if set(replay_recorded) != set(replay_observed) or not all(math.isfinite(v) for v in replay_recorded.values()):
        raise ValueError(f"resume replay summary is invalid for repeat {repeat}")
    for key, value in replay_observed.items():
        if not math.isclose(value, replay_recorded[key], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"resume replay summary mismatch for repeat {repeat}: {key}")

    prefix_gradients = payload["prefix_gradient_sums"]
    prefix_scalars = payload["prefix_scalar_sums"]
    if set(prefix_gradients) != set(PAIR_PREFIXES) or set(prefix_scalars) != set(PAIR_PREFIXES):
        raise ValueError(f"resume prefix keys mismatch for repeat {repeat}")
    for prefix in PAIR_PREFIXES:
        gradient = prefix_gradients[prefix]
        if (
            not isinstance(gradient, torch.Tensor)
            or tuple(gradient.shape) != (dimension,)
            or not bool(torch.isfinite(gradient).all().item())
            or not math.isfinite(float(prefix_scalars[prefix]))
        ):
            raise ValueError(f"resume prefix statistic invalid for repeat {repeat}, P={prefix}")
        scalar_from_states = float(states[f"state_mean_a_p{prefix}"].sum() * prefix)
        if not math.isclose(scalar_from_states, float(prefix_scalars[prefix]), rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError(f"resume scalar prefix mismatch for repeat {repeat}, P={prefix}")

    local = {
        group: _sufficient_from_payload(group_payload)
        for group, group_payload in payload["group_sufficient"].items()
    }
    if set(local) != {"all", "mnist", "fashion_mnist"}:
        raise ValueError(f"resume group keys mismatch for repeat {repeat}")
    if not all(_sufficient_is_finite(value) for value in local.values()):
        raise FloatingPointError(f"resume group sufficient statistic is nonfinite for repeat {repeat}")
    task_counts = states["task_name"].value_counts().to_dict()
    for group, sufficient in local.items():
        expected_count = STATE_COUNT if group == "all" else int(task_counts.get(group, 0))
        if sufficient.state_count != expected_count:
            raise ValueError(f"resume group state count mismatch for repeat {repeat}, group={group}")
        if sufficient.within_degrees_freedom != expected_count * (PAIR_COUNT - 1):
            raise ValueError(f"resume group within df mismatch for repeat {repeat}, group={group}")
        state_group = states if group == "all" else states.loc[states["task_name"] == group]
        expected_squared_sum = float(state_group["state_mean_squared_gradient_norm_p4"].sum())
        expected_within_sum = float(state_group["within_residual_sum"].sum())
        if not math.isclose(
            sufficient.state_mean_squared_norm_sum, expected_squared_sum, rel_tol=2e-7, abs_tol=1e-6
        ):
            raise ValueError(f"resume group squared-norm sum mismatch for repeat {repeat}, group={group}")
        if not math.isclose(sufficient.within_residual_sum, expected_within_sum, rel_tol=2e-7, abs_tol=1e-6):
            raise ValueError(f"resume group within sum mismatch for repeat {repeat}, group={group}")
    task_sum = local["mnist"].state_mean_sum + local["fashion_mnist"].state_mean_sum
    if _tensor_relative_error(local["all"].state_mean_sum, task_sum) > 2e-6:
        raise ValueError(f"resume mixed/task vector sum mismatch for repeat {repeat}")
    prefix_state_mean_sum = prefix_gradients[PAIR_COUNT].div(float(PAIR_COUNT))
    if _tensor_relative_error(local["all"].state_mean_sum, prefix_state_mean_sum) > 2e-6:
        raise ValueError(f"resume mixed/prefix vector sum mismatch for repeat {repeat}")
    return atomic_rows, state_rows, local, replay_observed


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose raw Variant A gradient variance into state and pair terms.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--primary-dir", type=Path, default=PRIMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    primary_dir = args.primary_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_dir = output_dir / "repeat_sufficient_stats"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    primary_resolved = json.loads((primary_dir / "resolved_config.json").read_text(encoding="utf-8"))
    primary_posthoc = json.loads((primary_dir / "posthoc_validity.json").read_text(encoding="utf-8"))
    source_paths = {
        "config": ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/config.py",
        "core": ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py",
        "preconditioning": (
            ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py"
        ),
        "primary_executor": ROOT / "scripts/audit_variant_a_estimator_stability.py",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    primary_source_hashes = {
        Path(path).name: digest for path, digest in primary_resolved["code_sha256"].items()
    }
    primary_artifact_hashes = {
        name: sha256_file(primary_dir / name)
        for name in (
            "resolved_config.json",
            "posthoc_validity.json",
            "active_decoder_parameters.json",
            "atomic_pair_samples.csv",
            "pairwise_gradient_cosines.csv",
        )
    }
    config_file_hash = sha256_file(run_dir / "config.json")
    saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "primary_seed_namespace": PRIMARY_PROTOCOL_ID,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "run_dir": str(run_dir),
        "primary_dir": str(primary_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": "float32",
        "seed": BASE_SEED,
        "repeats": REPEAT_COUNT,
        "states": STATE_COUNT,
        "pairs": PAIR_COUNT,
        "pair_prefixes": list(PAIR_PREFIXES),
        "batch_size": 128,
        "checkpoint_sha256": sha256_file(run_dir / "vae_checkpoint.pt"),
        "weight_pool_sha256": sha256_file(run_dir / "weight_pool.pt"),
        "weight_records_sha256": sha256_file(run_dir / "weight_pool_records.csv"),
        "config_file_sha256": config_file_hash,
        "saved_config_hash": saved_config.get("config_hash"),
        "executor_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": source_hashes,
        "primary_artifact_sha256": primary_artifact_hashes,
        "repository": git_metadata(),
    }
    _write_json(output_dir / "resolved_config.json", resolved)
    print(f"[a_state_pair] start resolved_config={json.dumps(resolved, sort_keys=True)}", flush=True)
    if resolved["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash mismatch")
    if resolved["weight_pool_sha256"] != EXPECTED_POOL_SHA256:
        raise ValueError("weight pool hash mismatch")
    if resolved["weight_records_sha256"] != EXPECTED_RECORDS_SHA256:
        raise ValueError("weight records hash mismatch")
    if config_file_hash != EXPECTED_CONFIG_FILE_SHA256 or saved_config.get("config_hash") != EXPECTED_CONFIG_HASH:
        raise ValueError("saved accepted config identity mismatch")
    if source_hashes["config"] != EXPECTED_CONFIG_SOURCE_SHA256:
        raise ValueError("runtime config.py source hash mismatch")
    if primary_artifact_hashes != EXPECTED_PRIMARY_ARTIFACT_SHA256:
        raise ValueError("pinned primary artifact hash mismatch")
    if not bool(primary_posthoc.get("passed")):
        raise ValueError("primary independent posthoc validity did not pass")
    for field in ("checkpoint_sha256", "weight_pool_sha256", "weight_records_sha256"):
        if resolved[field] != primary_resolved[field]:
            raise ValueError(f"current {field} differs from the primary run")
    expected_primary_sources = {
        "core.py": source_hashes["core"],
        "preconditioning.py": source_hashes["preconditioning"],
        "audit_variant_a_estimator_stability.py": source_hashes["primary_executor"],
    }
    if primary_source_hashes != expected_primary_sources:
        raise ValueError(
            f"primary source hashes drifted: recorded={primary_source_hashes} observed={expected_primary_sources}"
        )

    print("[a_state_pair] stage=load_primary_and_model", flush=True)
    primary_atomic, primary_pairwise = _load_primary_subset(primary_dir)
    primary_active = _primary_active_identity(primary_dir)
    run = _load_run(run_dir, device=device)
    runtime_config_hash = config_hash(run.cfg)
    if (
        int(run.cfg.vae_hidden_dim) != 2048
        or int(run.cfg.latent_dim) != 512
        or str(run.cfg.tiny_bigvae_output_mode).strip().lower() != "direct"
        or torch_dtype(run.cfg) != torch.float32
    ):
        raise ValueError("accepted model architecture/dtype identity mismatch")
    resolved["runtime_config_hash_with_current_defaults"] = runtime_config_hash
    resolved["primary_posthoc_validity_passed"] = bool(primary_posthoc.get("passed"))
    _write_json(output_dir / "resolved_config.json", resolved)
    preflight, active = _preflight_decomposition(
        run,
        batch_size=128,
        seed=BASE_SEED + 991,
        device=device,
    )
    active_count = int(sum(parameter.numel() for _name, parameter in active))
    active_names = [name for name, _parameter in active]
    if active_count != int(primary_active["active_parameter_count"]):
        raise ValueError("active parameter count differs from primary run")
    if active_names != list(primary_active["active_parameter_names"]):
        raise ValueError("active parameter order differs from primary run")
    active_name_set = set(active_names)
    excluded = [(name, parameter) for name, parameter in run.vae.named_parameters() if name not in active_name_set]
    model_hash_before = model_state_sha256(run.vae)
    if model_hash_before != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError(f"loaded model-state hash mismatch: {model_hash_before} != {EXPECTED_MODEL_STATE_SHA256}")
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=1, batch_size=128)
    dimension = active_count
    shard_identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": resolved["protocol_sha256"],
        "executor_sha256": resolved["executor_sha256"],
        "checkpoint_sha256": resolved["checkpoint_sha256"],
        "weight_pool_sha256": resolved["weight_pool_sha256"],
        "weight_records_sha256": resolved["weight_records_sha256"],
        "primary_artifact_sha256": primary_artifact_hashes,
        "source_sha256": source_hashes,
        "saved_config_hash": saved_config["config_hash"],
        "runtime_config_hash": runtime_config_hash,
        "model_state_sha256": model_hash_before,
        "active_parameter_names": active_names,
        "active_parameter_count": active_count,
        "states": STATE_COUNT,
        "pairs": PAIR_COUNT,
        "seed": BASE_SEED,
    }
    if args.preflight_only:
        print(
            "[a_state_pair] preflight_only_done "
            f"config_file_sha256={config_file_hash} saved_config_hash={saved_config['config_hash']} "
            f"runtime_config_hash={runtime_config_hash} model_state_sha256={model_hash_before} "
            f"active_parameter_count={active_count}",
            flush=True,
        )
        return

    aggregate_ids = ["full"] + [f"split{index}_{side}" for index in range(1, 4) for side in ("left", "right")]
    group_labels = ("all", "mnist", "fashion_mnist")
    aggregate: dict[tuple[str, str], Sufficient] = {
        (aggregate_id, group): _new_sufficient(dimension)
        for aggregate_id in aggregate_ids
        for group in group_labels
    }
    repeat_component_rows: list[dict[str, Any]] = []
    atomic_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    max_scalar_abs_error = 0.0
    max_gradient_norm_relative_error = 0.0
    minimum_raw_within_residual = float("inf")
    minimum_normalized_within_residual = float("inf")
    total_units = REPEAT_COUNT * STATE_COUNT * PAIR_COUNT
    completed_units = 0
    started = time.perf_counter()

    print("[a_state_pair] stage=directional_sufficient_statistics", flush=True)
    for repeat in range(REPEAT_COUNT):
        final_path = repeat_dir / f"repeat_{repeat:02d}.pt"
        if final_path.is_file():
            digest_path = final_path.with_suffix(".pt.sha256")
            if not digest_path.is_file():
                raise ValueError(f"resume shard digest is missing: {digest_path}")
            recorded_digest = digest_path.read_text(encoding="ascii").strip()
            observed_digest = sha256_file(final_path)
            if recorded_digest != observed_digest:
                raise ValueError(f"resume shard digest mismatch: {final_path}")
            payload = torch.load(final_path, map_location="cpu", weights_only=False)
            repeat_atomic_rows, repeat_state_rows, local, replay = _validate_shard_payload(
                payload,
                shard_identity=shard_identity,
                primary_atomic=primary_atomic,
                repeat=repeat,
                dimension=dimension,
            )
            max_scalar_abs_error = max(max_scalar_abs_error, replay["max_scalar_abs_error"])
            max_gradient_norm_relative_error = max(
                max_gradient_norm_relative_error,
                replay["max_gradient_norm_relative_error"],
            )
            minimum_raw_within_residual = min(
                minimum_raw_within_residual,
                replay["minimum_raw_within_residual"],
            )
            minimum_normalized_within_residual = min(
                minimum_normalized_within_residual,
                replay["minimum_normalized_within_residual"],
            )
            atomic_rows.extend(repeat_atomic_rows)
            state_rows.extend(repeat_state_rows)
            for group, sufficient in local.items():
                if sufficient.state_count >= 2:
                    repeat_component_rows.append(
                        {
                            "scope": "repeat",
                            "scope_id": f"repeat{repeat}",
                            "repeat": repeat,
                            "task_group": group,
                            **variance_components_from_sufficient(sufficient, pair_count=PAIR_COUNT),
                        }
                    )
            for aggregate_id in _group_ids_for_repeat(repeat):
                for group in group_labels:
                    _merge_sufficient(aggregate[(aggregate_id, group)], local[group])
            completed_units += STATE_COUNT * PAIR_COUNT
            print(
                f"[a_state_pair] cache_hit repeat={repeat + 1}/{REPEAT_COUNT} artifact={final_path}",
                flush=True,
            )
            del payload, local
            continue

        repeat_atomic_start = len(atomic_rows)
        repeat_state_start = len(state_rows)
        repeat_minimum_raw_residual = float("inf")
        repeat_minimum_normalized_residual = float("inf")
        state_indices, state_draw_seeds, train_positions = _sample_state_indices(
            run.train_indices,
            count=STATE_COUNT,
            seed=BASE_SEED,
            repeat=repeat,
        )
        selected_weights = run.weights[state_indices].to(device=device, dtype=torch_dtype(run.cfg))
        with torch.no_grad():
            z_values = encode_weights(run.vae, run.normalizer, selected_weights).detach()
        records = run.records.iloc[state_indices].to_dict(orient="records")
        for record, source_index in zip(records, state_indices, strict=True):
            record["source_weight_index"] = int(source_index)

        prefix_gradient_sums = {
            prefix: torch.zeros(dimension, dtype=torch.float32) for prefix in PAIR_PREFIXES
        }
        prefix_scalar_sums = {prefix: 0.0 for prefix in PAIR_PREFIXES}
        local = {group: _new_sufficient(dimension) for group in group_labels}

        for state_position in range(STATE_COUNT):
            record = records[state_position]
            task_name = str(record.get("task_name", ""))
            if task_name not in {"mnist", "fashion_mnist"}:
                raise ValueError(f"unexpected task {task_name!r}")
            state_gradient_sum = torch.zeros(dimension, dtype=torch.float32)
            state_atomic_squared_norm_sum = 0.0
            state_scalar_sum = 0.0
            state_prefix_scalar: dict[int, float] = {}
            for pair_position in range(PAIR_COUNT):
                run.vae.zero_grad(set_to_none=True)
                generator_1, generator_2, probe_seed_1, probe_seed_2 = _indexed_generators(
                    BASE_SEED, repeat, state_position, pair_position
                )
                atomic_loss, stats = _atomic_pair_loss(
                    cfg=cfg,
                    run=run,
                    z=z_values[state_position],
                    record=record,
                    step=repeat + 1,
                    pair_index=pair_position,
                    probe_generator_1=generator_1,
                    probe_generator_2=generator_2,
                )
                atomic_loss.backward()
                for excluded_name, excluded_parameter in excluded:
                    if excluded_parameter.grad is not None and bool((excluded_parameter.grad.detach() != 0).any().item()):
                        raise RuntimeError(f"excluded parameter received gradient: {excluded_name}")
                gradient = _flatten_active_gradients(active)
                scalar = float(atomic_loss.detach().cpu().item())
                gradient_norm = vector_norm(gradient)
                if not math.isfinite(scalar) or not math.isfinite(gradient_norm):
                    raise FloatingPointError("nonfinite atomic scalar or gradient norm")
                primary = primary_atomic.loc[(repeat, state_position, pair_position)]
                if int(primary["source_weight_index"]) != int(state_indices[state_position]):
                    raise ValueError("source index differs from primary run")
                if int(primary["probe_seed_1"]) != probe_seed_1 or int(primary["probe_seed_2"]) != probe_seed_2:
                    raise ValueError("probe seed differs from primary run")
                scalar_error = abs(scalar - float(primary["a_scalar"]))
                norm_error = relative_error(gradient_norm, float(primary["gradient_norm"]))
                max_scalar_abs_error = max(max_scalar_abs_error, scalar_error)
                max_gradient_norm_relative_error = max(max_gradient_norm_relative_error, norm_error)
                state_gradient_sum.add_(gradient)
                state_atomic_squared_norm_sum += gradient_norm * gradient_norm
                state_scalar_sum += scalar
                prefix = pair_position + 1
                if prefix in PAIR_PREFIXES:
                    prefix_gradient_sums[prefix].add_(state_gradient_sum)
                    prefix_scalar_sums[prefix] += state_scalar_sum
                    state_prefix_scalar[prefix] = state_scalar_sum / float(prefix)
                atomic_rows.append(
                    {
                        "repeat": repeat,
                        "state_position": state_position,
                        "pair_position": pair_position,
                        "source_weight_index": int(state_indices[state_position]),
                        "task_name": task_name,
                        "tau": float(record.get("tau", 1.0)),
                        "state_draw_seed": int(state_draw_seeds[state_position]),
                        "train_position": int(train_positions[state_position]),
                        "probe_seed_1": probe_seed_1,
                        "probe_seed_2": probe_seed_2,
                        "a_scalar": scalar,
                        "primary_a_scalar": float(primary["a_scalar"]),
                        "scalar_abs_error": scalar_error,
                        "gradient_norm": gradient_norm,
                        "primary_gradient_norm": float(primary["gradient_norm"]),
                        "gradient_norm_relative_error": norm_error,
                        "h1_norm": float(stats["h1_norm"]),
                        "h2_norm": float(stats["h2_norm"]),
                    }
                )
                completed_units += 1
                if completed_units == 1 or completed_units % max(1, int(args.log_every)) == 0:
                    elapsed = time.perf_counter() - started
                    rate = completed_units / max(elapsed, 1e-12)
                    eta = (total_units - completed_units) / max(rate, 1e-12)
                    print(
                        "[a_state_pair] progress "
                        f"unit={completed_units}/{total_units} repeat={repeat + 1}/{REPEAT_COUNT} "
                        f"state={state_position + 1}/{STATE_COUNT} pair={pair_position + 1}/{PAIR_COUNT} "
                        f"A={scalar:.6g} grad_norm={gradient_norm:.6g} rate={rate:.2f}/s eta_sec={eta:.1f}",
                        flush=True,
                    )
                del gradient, atomic_loss

            state_sum_norm_squared = vector_norm(state_gradient_sum) ** 2
            state_mean = state_gradient_sum.div(float(PAIR_COUNT))
            state_mean_squared_norm = state_sum_norm_squared / float(PAIR_COUNT * PAIR_COUNT)
            raw_within_residual = state_atomic_squared_norm_sum - state_sum_norm_squared / float(PAIR_COUNT)
            minimum_raw_within_residual = min(minimum_raw_within_residual, raw_within_residual)
            normalized_within_residual = raw_within_residual / max(state_atomic_squared_norm_sum, 1.0)
            minimum_normalized_within_residual = min(
                minimum_normalized_within_residual, normalized_within_residual
            )
            repeat_minimum_raw_residual = min(repeat_minimum_raw_residual, raw_within_residual)
            repeat_minimum_normalized_residual = min(
                repeat_minimum_normalized_residual, normalized_within_residual
            )
            residual_tolerance = 1e-5 * max(state_atomic_squared_norm_sum, 1.0)
            if raw_within_residual < -residual_tolerance:
                raise FloatingPointError(
                    f"negative within residual repeat={repeat} state={state_position}: {raw_within_residual}"
                )
            within_residual = max(raw_within_residual, 0.0)
            for group in ("all", task_name):
                sufficient = local[group]
                sufficient.state_count += 1
                sufficient.state_mean_sum.add_(state_mean)
                sufficient.state_mean_squared_norm_sum += state_mean_squared_norm
                sufficient.within_residual_sum += within_residual
                sufficient.within_degrees_freedom += PAIR_COUNT - 1
            state_rows.append(
                {
                    "repeat": repeat,
                    "state_position": state_position,
                    "source_weight_index": int(state_indices[state_position]),
                    "task_name": task_name,
                    "tau": float(record.get("tau", 1.0)),
                    "state_mean_a_p1": state_prefix_scalar[1],
                    "state_mean_a_p2": state_prefix_scalar[2],
                    "state_mean_a_p4": state_prefix_scalar[4],
                    "state_mean_gradient_norm_p4": math.sqrt(max(state_mean_squared_norm, 0.0)),
                    "state_mean_squared_gradient_norm_p4": state_mean_squared_norm,
                    "within_residual_sum": within_residual,
                    "within_residual_sum_raw": raw_within_residual,
                    "normalized_within_residual_sum_raw": normalized_within_residual,
                    "atomic_squared_gradient_norm_sum": state_atomic_squared_norm_sum,
                }
            )

        for group, sufficient in local.items():
            if sufficient.state_count >= 2:
                repeat_component_rows.append(
                    {
                        "scope": "repeat",
                        "scope_id": f"repeat{repeat}",
                        "repeat": repeat,
                        "task_group": group,
                        **variance_components_from_sufficient(sufficient, pair_count=PAIR_COUNT),
                    }
                )
        for aggregate_id in _group_ids_for_repeat(repeat):
            for group in group_labels:
                _merge_sufficient(aggregate[(aggregate_id, group)], local[group])

        if not all(_sufficient_is_finite(sufficient) for sufficient in local.values()):
            raise FloatingPointError(f"nonfinite repeat sufficient statistic: repeat={repeat}")
        if not all(torch.isfinite(value).all().item() for value in prefix_gradient_sums.values()):
            raise FloatingPointError(f"nonfinite repeat prefix sum: repeat={repeat}")
        if not all(math.isfinite(float(value)) for value in prefix_scalar_sums.values()):
            raise FloatingPointError(f"nonfinite repeat scalar prefix sum: repeat={repeat}")
        payload = {
            "shard_identity": shard_identity,
            "protocol_id": PROTOCOL_ID,
            "primary_seed_namespace": PRIMARY_PROTOCOL_ID,
            "repeat": repeat,
            "states": STATE_COUNT,
            "pairs": PAIR_COUNT,
            "state_indices": state_indices,
            "state_draw_seeds": state_draw_seeds,
            "train_positions": train_positions,
            "prefix_gradient_sums": prefix_gradient_sums,
            "prefix_scalar_sums": prefix_scalar_sums,
            "group_sufficient": {
                group: _sufficient_to_payload(sufficient)
                for group, sufficient in local.items()
            },
            "atomic_rows": atomic_rows[repeat_atomic_start:],
            "state_rows": state_rows[repeat_state_start:],
            "replay_summary": {
                "max_scalar_abs_error": max(
                    float(row["scalar_abs_error"]) for row in atomic_rows[repeat_atomic_start:]
                ),
                "max_gradient_norm_relative_error": max(
                    float(row["gradient_norm_relative_error"]) for row in atomic_rows[repeat_atomic_start:]
                ),
                "minimum_raw_within_residual": repeat_minimum_raw_residual,
                "minimum_normalized_within_residual": repeat_minimum_normalized_residual,
            },
        }
        _validate_shard_payload(
            payload,
            shard_identity=shard_identity,
            primary_atomic=primary_atomic,
            repeat=repeat,
            dimension=dimension,
        )
        temporary_path = final_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(final_path)
        digest_path = final_path.with_suffix(".pt.sha256")
        digest_temporary_path = digest_path.with_suffix(".sha256.tmp")
        digest_temporary_path.write_text(sha256_file(final_path) + "\n", encoding="ascii")
        digest_temporary_path.replace(digest_path)
        print(f"[a_state_pair] repeat_done repeat={repeat + 1}/{REPEAT_COUNT} artifact={final_path}", flush=True)
        del prefix_gradient_sums, local, payload, z_values, selected_weights

    atomic_frame = pd.DataFrame(atomic_rows)
    state_frame = pd.DataFrame(state_rows)
    _require_finite_frame(
        atomic_frame,
        (
            "repeat",
            "state_position",
            "pair_position",
            "a_scalar",
            "primary_a_scalar",
            "scalar_abs_error",
            "gradient_norm",
            "primary_gradient_norm",
            "gradient_norm_relative_error",
            "h1_norm",
            "h2_norm",
        ),
        label="completed atomic comparison",
    )
    _require_finite_frame(
        state_frame,
        (
            "repeat",
            "state_position",
            "state_mean_a_p1",
            "state_mean_a_p2",
            "state_mean_a_p4",
            "state_mean_gradient_norm_p4",
            "state_mean_squared_gradient_norm_p4",
            "within_residual_sum",
            "within_residual_sum_raw",
            "normalized_within_residual_sum_raw",
            "atomic_squared_gradient_norm_sum",
        ),
        label="completed state sufficient table",
    )
    atomic_frame.to_csv(output_dir / "atomic_comparison.csv", index=False)
    state_frame.to_csv(output_dir / "state_sufficient_stats.csv", index=False)

    component_rows = list(repeat_component_rows)
    for (scope_id, group), sufficient in aggregate.items():
        if sufficient.state_count < 2:
            continue
        component_rows.append(
            {
                "scope": "full" if scope_id == "full" else "split_half",
                "scope_id": scope_id,
                "repeat": -1,
                "task_group": group,
                **variance_components_from_sufficient(sufficient, pair_count=PAIR_COUNT),
            }
        )
    components = pd.DataFrame(component_rows)
    components.to_csv(output_dir / "variance_components.csv", index=False)

    print("[a_state_pair] stage=repeat_prefix_gram", flush=True)
    gram_payload: dict[str, np.ndarray] = {}
    prefix_metric_rows: list[dict[str, Any]] = []
    pairwise_frames: list[pd.DataFrame] = []
    max_pairwise_cosine_abs_error = 0.0
    for prefix in PAIR_PREFIXES:
        vectors: list[torch.Tensor] = []
        scalars: list[float] = []
        for repeat in range(REPEAT_COUNT):
            payload = torch.load(repeat_dir / f"repeat_{repeat:02d}.pt", map_location="cpu", weights_only=False)
            vectors.append(payload["prefix_gradient_sums"][prefix].div(float(STATE_COUNT * prefix)))
            scalars.append(float(payload["prefix_scalar_sums"][prefix] / float(STATE_COUNT * prefix)))
        gram = _chunked_gram(vectors)
        gram_payload[f"gram_p{prefix}"] = gram
        pairwise = _pairwise_from_gram(gram, pair_count=prefix)
        pairwise_frames.append(pairwise)
        key_columns = ["states", "pairs", "left_repeat", "right_repeat"]
        primary = primary_pairwise.loc[primary_pairwise["pairs"] == prefix, key_columns + ["gradient_cosine"]]
        observed = pairwise.loc[:, key_columns + ["gradient_cosine"]]
        comparison = primary.merge(
            observed,
            on=key_columns,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_primary", "_observed"),
        )
        if len(comparison) != 66 or not bool((comparison["_merge"] == "both").all()):
            raise ValueError("primary pairwise key mismatch")
        cosine_error = np.abs(
            comparison["gradient_cosine_primary"].to_numpy(dtype=np.float64)
            - comparison["gradient_cosine_observed"].to_numpy(dtype=np.float64)
        )
        max_pairwise_cosine_abs_error = max(max_pairwise_cosine_abs_error, float(cosine_error.max()))
        upper = gram[np.triu_indices(REPEAT_COUNT, k=1)]
        pair_cosines = pairwise["gradient_cosine"].to_numpy(dtype=np.float64)
        prefix_metric_rows.append(
            {
                "states": STATE_COUNT,
                "pairs": prefix,
                "scalar_mean": float(np.mean(scalars)),
                "scalar_std": float(np.std(scalars, ddof=1)),
                "pairwise_cosine_median": _quantile(pair_cosines, 0.5),
                "pairwise_cosine_q10": _quantile(pair_cosines, 0.1),
                "pairwise_cosine_min": float(pair_cosines.min()),
                "mean_cross_repeat_dot_signal_sq_estimate": float(upper.mean()),
                "mean_repeat_gradient_squared_norm": float(np.diag(gram).mean()),
            }
        )
        del vectors
    np.savez_compressed(output_dir / "repeat_prefix_gram.npz", **gram_payload)
    pairwise_frame = pd.concat(pairwise_frames, ignore_index=True)
    pairwise_frame.to_csv(output_dir / "repeat_pairwise_cosines.csv", index=False)
    pd.DataFrame(prefix_metric_rows).to_csv(output_dir / "repeat_prefix_metrics.csv", index=False)

    model_hash_after = model_state_sha256(run.vae)
    sufficient_statistics_finite = bool(
        all(_sufficient_is_finite(value) for value in aggregate.values())
        and np.isfinite(components.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all()
    )
    gram_and_prefix_metrics_finite = bool(
        all(np.isfinite(value).all() for value in gram_payload.values())
        and np.isfinite(pd.DataFrame(prefix_metric_rows).select_dtypes(include=[np.number]).to_numpy()).all()
        and np.isfinite(pairwise_frame.select_dtypes(include=[np.number]).to_numpy()).all()
    )
    validity = {
        "checkpoint_hash_match": resolved["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "pool_hash_match": resolved["weight_pool_sha256"] == EXPECTED_POOL_SHA256,
        "records_hash_match": resolved["weight_records_sha256"] == EXPECTED_RECORDS_SHA256,
        "primary_posthoc_validity_passed": bool(primary_posthoc.get("passed")),
        "primary_data_hashes_match": all(
            resolved[field] == primary_resolved[field]
            for field in ("checkpoint_sha256", "weight_pool_sha256", "weight_records_sha256")
        ),
        "primary_source_hashes_match": primary_source_hashes == expected_primary_sources,
        "pinned_primary_artifact_hashes_match": primary_artifact_hashes == EXPECTED_PRIMARY_ARTIFACT_SHA256,
        "saved_config_hash": saved_config.get("config_hash"),
        "runtime_config_hash_with_current_defaults": runtime_config_hash,
        "accepted_config_identity": bool(
            config_file_hash == EXPECTED_CONFIG_FILE_SHA256
            and saved_config.get("config_hash") == EXPECTED_CONFIG_HASH
            and source_hashes["config"] == EXPECTED_CONFIG_SOURCE_SHA256
        ),
        "active_parameter_identity": True,
        "atomic_row_count_match": len(atomic_rows) == total_units,
        "atomic_keys_unique": not atomic_frame.duplicated(
            ["repeat", "state_position", "pair_position"]
        ).any(),
        "atomic_and_state_tables_finite": True,
        "max_scalar_abs_error_vs_primary": max_scalar_abs_error,
        "scalar_replay_pass": max_scalar_abs_error <= 1e-6,
        "max_gradient_norm_relative_error_vs_primary": max_gradient_norm_relative_error,
        "gradient_norm_replay_pass": max_gradient_norm_relative_error <= 2e-5,
        "minimum_raw_within_residual": minimum_raw_within_residual,
        "minimum_normalized_within_residual": minimum_normalized_within_residual,
        "within_residual_pass": minimum_normalized_within_residual >= -1e-5,
        "sufficient_statistics_finite": sufficient_statistics_finite,
        "gram_and_prefix_metrics_finite": gram_and_prefix_metrics_finite,
        "max_pairwise_cosine_abs_error_vs_primary": max_pairwise_cosine_abs_error,
        "pairwise_gram_replay_pass": max_pairwise_cosine_abs_error <= 2e-6,
        "model_state_hash_before": model_hash_before,
        "model_state_hash_after": model_hash_after,
        "model_matches_pinned_primary": model_hash_before == EXPECTED_MODEL_STATE_SHA256,
        "model_unchanged": model_hash_before == model_hash_after == EXPECTED_MODEL_STATE_SHA256,
        "preflight": preflight,
    }
    blocking_keys = [
        "checkpoint_hash_match",
        "pool_hash_match",
        "records_hash_match",
        "primary_posthoc_validity_passed",
        "primary_data_hashes_match",
        "primary_source_hashes_match",
        "pinned_primary_artifact_hashes_match",
        "accepted_config_identity",
        "active_parameter_identity",
        "atomic_row_count_match",
        "atomic_keys_unique",
        "atomic_and_state_tables_finite",
        "scalar_replay_pass",
        "gradient_norm_replay_pass",
        "within_residual_pass",
        "sufficient_statistics_finite",
        "gram_and_prefix_metrics_finite",
        "pairwise_gram_replay_pass",
        "model_matches_pinned_primary",
        "model_unchanged",
    ]
    validity["passed"] = bool(all(bool(validity[key]) for key in blocking_keys))
    _write_json(output_dir / "validity.json", validity)
    if not validity["passed"]:
        raise RuntimeError(f"directional variance validity failed: {json.dumps(validity, sort_keys=True, default=float)}")

    gate_rows = components.loc[
        (components["task_group"] == "all")
        & components["scope_id"].isin(["full"] + [f"split{i}_{side}" for i in range(1, 4) for side in ("left", "right")])
    ].copy()
    gate_rows = gate_rows.sort_values("scope_id")
    state_axis_dominant = bool(gate_rows["state_axis_dominates_at_P"].all() and len(gate_rows) == 7)
    decision = {
        "validity_passed": True,
        "configured_states": STATE_COUNT,
        "configured_pairs": PAIR_COUNT,
        "required_scope_ids": gate_rows["scope_id"].tolist(),
        "state_axis_dominant_full_and_all_split_halves": state_axis_dominant,
        "next_axis": "states" if state_axis_dominant else "pair_or_crossed_diagnostic",
        "evidence_boundary": "directional random-effects trace decomposition at S=32,P=4 only",
    }
    _write_json(output_dir / "axis_decision.json", decision)

    figure_rows = gate_rows.set_index("scope_id")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    positions = np.arange(len(figure_rows), dtype=np.float64)
    ax.bar(
        positions - 0.2,
        figure_rows["between_state_covariance_trace_B_raw"],
        width=0.4,
        label="between-state B",
    )
    ax.bar(
        positions + 0.2,
        figure_rows["within_pair_contribution_at_P_W_over_P"],
        width=0.4,
        label="within-pair W/P",
    )
    ax.set_xticks(positions, figure_rows.index, rotation=25, ha="right")
    ax.set_ylabel("gradient covariance trace contribution")
    ax.set_title("Raw Variant A gradient variance at S=32, P=4")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "variance_components.png", dpi=180)
    plt.close(fig)

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "resolved_config": resolved,
        "validity": validity,
        "decision": decision,
        "elapsed_sec": float(time.perf_counter() - started),
        "artifacts": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(
        "[a_state_pair] done "
        f"elapsed_sec={time.perf_counter() - started:.1f} state_axis_dominant={state_axis_dominant} "
        f"validity={validity['passed']} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
