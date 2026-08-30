from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    config_hash,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import encode_weights
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    PreconditioningState,
    _batch_indices,
    _task_set_for_record,
    preconditioning_regularizer,
)
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_POOL_SHA256,
    EXPECTED_RECORDS_SHA256,
    ROOT,
    _atomic_pair_loss,
    _flatten_active_gradients,
    _load_run,
    _module_slices,
    _preflight_decomposition,
    _probe_cfg,
    git_metadata,
    sha256_file,
    sha256_tensor,
    stable_uint63,
    vector_cosine,
    vector_norm,
    vector_relative_error,
)
from scripts.audit_variant_a_hvp_batch_size import (
    assert_excluded_gradients_zero,
    cuda_memory,
    gram_group_metrics,
    gram_pair_metrics,
    top_mass_share,
)
from scripts.audit_variant_a_state_pair_variance import model_state_sha256


PROTOCOL_ID = "a_fixed_state_probe_variance_h2048_v1"
BASE_SEED = 20260714
STATE_COUNT = 44
PRIMARY_COUNT = 32
SENTINEL_COUNT = 12
DRAW_COUNT = 64
PAIR_COUNT = 4
PREFIXES = (1, 2, 4)
TRAIN_COUNT = 16384
BRIDGE_POSITIONS = (0, 4, 8, 12, 16, 20, 24, 28)
BRIDGE_SOURCE_IDS = (7589, 922, 13104, 8769, 9439, 2396, 191, 4016)
SPLITS = (
    (tuple(range(0, 32)), tuple(range(32, 64))),
    (tuple(range(0, 64, 2)), tuple(range(1, 64, 2))),
    (
        tuple([*range(0, 16), *range(32, 48)]),
        tuple([*range(16, 32), *range(48, 64)]),
    ),
)
FOLDS = tuple(tuple(range(start, start + 16)) for start in range(0, 64, 16))

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_fixed_state_probe_variance_h2048"
)
PROTOCOL_PATH = DEFAULT_OUTPUT_DIR / "protocol.md"
STATE_BANK_PATH = DEFAULT_OUTPUT_DIR / "state_bank.csv"
EXCLUDED_PRIMARY_PATH = DEFAULT_OUTPUT_DIR / "excluded_primary_sources.csv"
ELIGIBLE_RUNS_PATH = DEFAULT_OUTPUT_DIR / "eligible_runs.csv"
SENTINEL_PARENT_PATH = DEFAULT_OUTPUT_DIR / "sentinel_parent_states.csv"

EXPECTED_SHA256 = {
    "checkpoint": EXPECTED_CHECKPOINT_SHA256,
    "weight_pool": EXPECTED_POOL_SHA256,
    "records": EXPECTED_RECORDS_SHA256,
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
EXPECTED_CONFIG_HASH = "4fc87a54349a39a2"
EXPECTED_RUNTIME_CONFIG_HASH = "d1d072b2e65ad8a7"
EXPECTED_MODEL_STATE_SHA256 = "84bf2d5f050ac21f7bf402d478b73d9515e4a2477c4094eeb357fa140b69b8e1"
EXPECTED_ACTIVE_COUNT = 11_685_120
EXPECTED_SENTINEL_IDS = (
    3941,
    9142,
    8644,
    3017,
    8302,
    548,
    11266,
    13838,
    1209,
    10583,
    11561,
    11667,
)


@dataclass(frozen=True, slots=True)
class DeltaEstimate:
    v_probe: float
    v_state: float
    delta: float
    state_interaction_ms: tuple[float, ...]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")
    os.replace(temporary, path)


def common_probe_seed(draw: int, pair: int, branch: int, *, seed: int = BASE_SEED) -> int:
    if not (0 <= int(draw) < DRAW_COUNT and 0 <= int(pair) < PAIR_COUNT and int(branch) in (0, 1)):
        raise ValueError("invalid crossed probe key")
    return stable_uint63(PROTOCOL_ID, int(seed), "probe", int(draw), int(pair), int(branch))


def common_probe_generators(
    draw: int,
    pair: int,
    *,
    seed: int = BASE_SEED,
) -> tuple[torch.Generator, torch.Generator, int, int]:
    seeds = (common_probe_seed(draw, pair, 0, seed=seed), common_probe_seed(draw, pair, 1, seed=seed))
    return (
        torch.Generator(device="cpu").manual_seed(seeds[0]),
        torch.Generator(device="cpu").manual_seed(seeds[1]),
        seeds[0],
        seeds[1],
    )


def _read_bank(path: Path) -> pd.DataFrame:
    # pandas otherwise rounds uint63 selection hashes through float64 because sentinel rows are blank.
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frame = pd.DataFrame(rows)
    integer_columns = ("state_position", "source_weight_index", "run", "step", "step_stratum", "selection_rank")
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tau"] = pd.to_numeric(frame["tau"], errors="raise").astype(np.float64)
    frame["prior_grad_rms"] = pd.to_numeric(frame["prior_grad_rms"], errors="coerce")
    return frame


def validate_state_bank(
    bank: pd.DataFrame,
    excluded: pd.DataFrame,
    *,
    eligibility: pd.DataFrame | None = None,
    sentinel_parent: pd.DataFrame | None = None,
    records: pd.DataFrame | None = None,
    train_indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    required = {
        "state_position",
        "panel",
        "task_name",
        "source_weight_index",
        "run",
        "step",
        "step_stratum",
        "tau",
        "selection_policy",
        "selection_rank",
        "run_selection_hash",
        "snapshot_selection_hash",
        "prior_grad_rms",
        "prior_stratum",
    }
    missing = sorted(required - set(bank.columns))
    if missing:
        raise ValueError(f"state bank missing columns: {missing}")
    bank = bank.sort_values("state_position").reset_index(drop=True)
    if len(bank) != STATE_COUNT or bank["state_position"].astype(int).tolist() != list(range(STATE_COUNT)):
        raise ValueError("state bank must contain exact positions 0..43")
    primary = bank.loc[bank["panel"] == "primary"].copy()
    sentinel = bank.loc[bank["panel"] == "sentinel"].copy()
    if len(primary) != PRIMARY_COUNT or len(sentinel) != SENTINEL_COUNT:
        raise ValueError("state bank must contain exactly 32 primary and 12 sentinel states")
    if primary["state_position"].astype(int).tolist() != list(range(PRIMARY_COUNT)):
        raise ValueError("primary states must occupy positions 0..31")
    if sentinel["state_position"].astype(int).tolist() != list(range(PRIMARY_COUNT, STATE_COUNT)):
        raise ValueError("sentinel states must occupy positions 32..43")
    expected_balance = pd.Series(4, index=pd.MultiIndex.from_product(
        [["fashion_mnist", "mnist"], range(4)], names=["task_name", "step_stratum"]
    ))
    observed_balance = primary.groupby(["task_name", "step_stratum"]).size().sort_index()
    if not observed_balance.equals(expected_balance.sort_index()):
        raise ValueError(f"primary task/stratum balance mismatch: {observed_balance.to_dict()}")
    if primary["run"].nunique() != PRIMARY_COUNT:
        raise ValueError("primary source trajectories are not unique")
    if tuple(sentinel["source_weight_index"].astype(int)) != EXPECTED_SENTINEL_IDS:
        raise ValueError("exact sentinel source IDs mismatch")
    if sentinel["run"].nunique() != SENTINEL_COUNT:
        raise ValueError("sentinel trajectories are not unique")
    if set(primary["run"].astype(int)) & set(sentinel["run"].astype(int)):
        raise ValueError("primary and sentinel trajectories overlap")
    excluded_ids = set(excluded["source_weight_index"].astype(int))
    if len(excluded) != 404 or len(excluded_ids) != 404:
        raise ValueError("excluded-primary manifest must bind 404 unique source identities")
    overlap = set(primary["source_weight_index"].astype(int)) & excluded_ids
    if overlap:
        raise ValueError(f"primary bank overlaps frozen exclusions: {sorted(overlap)}")
    for row in primary.itertuples(index=False):
        expected_run_hash = "u63_" + str(
            stable_uint63(
                PROTOCOL_ID,
                BASE_SEED,
                "primary_run",
                row.task_name,
                int(row.step_stratum),
                int(row.run),
            )
        )
        expected_snapshot_hash = "u63_" + str(
            stable_uint63(
                PROTOCOL_ID,
                BASE_SEED,
                "primary_snapshot",
                row.task_name,
                int(row.step_stratum),
                int(row.run),
                int(row.source_weight_index),
            )
        )
        if str(row.run_selection_hash) != expected_run_hash:
            raise ValueError(f"primary run-selection hash mismatch at state {row.state_position}")
        if str(row.snapshot_selection_hash) != expected_snapshot_hash:
            raise ValueError(f"primary snapshot-selection hash mismatch at state {row.state_position}")
    if tuple(primary.loc[primary["state_position"].isin(BRIDGE_POSITIONS), "source_weight_index"].astype(int)) != BRIDGE_SOURCE_IDS:
        raise ValueError("frozen B=128 bridge IDs mismatch")
    if eligibility is not None:
        if len(eligibility) != 1488:
            raise ValueError("eligibility trace must contain exactly 1,488 rows")
        selected = eligibility.loc[eligibility["selected"].astype(str).str.lower() == "true"].copy()
        selected = selected.sort_values(["task_name", "step_stratum", "selection_rank"])
        expected = primary.sort_values(["task_name", "step_stratum", "selection_rank"])
        fields = [
            ("run", "run"),
            ("best_source_weight_index", "source_weight_index"),
            ("run_selection_hash", "run_selection_hash"),
            ("best_snapshot_selection_hash", "snapshot_selection_hash"),
        ]
        if len(selected) != PRIMARY_COUNT:
            raise ValueError("eligibility trace must select exactly 32 states")
        for trace_column, bank_column in fields:
            if selected[trace_column].astype(str).tolist() != expected[bank_column].astype(str).tolist():
                raise ValueError(f"eligibility trace differs from state bank: {trace_column}")
    if sentinel_parent is not None:
        if len(sentinel_parent) != 24:
            raise ValueError("sentinel parent table must contain exactly 24 states")
        selected = sentinel_parent.loc[sentinel_parent["selected"].astype(str).str.lower() == "true"].copy()
        selected = selected.sort_values(["task_name", "selected_stratum", "selection_rank"])
        expected = sentinel.sort_values(["task_name", "prior_stratum", "selection_rank"])
        if selected["source_weight_index"].astype(int).tolist() != expected["source_weight_index"].astype(int).tolist():
            raise ValueError("sentinel parent selection differs from state bank")
        if not np.allclose(
            selected["prior_grad_rms"].to_numpy(dtype=np.float64),
            expected["prior_grad_rms"].to_numpy(dtype=np.float64),
            rtol=5e-15,
            atol=5e-15,
        ):
            raise ValueError("sentinel parent outcomes differ from state bank")
    if records is not None:
        if max(bank["source_weight_index"].astype(int)) >= len(records):
            raise ValueError("state bank source index lies outside records")
        for row in bank.itertuples(index=False):
            source = records.iloc[int(row.source_weight_index)]
            exact = (
                int(source["run"]) == int(row.run)
                and int(source["step"]) == int(row.step)
                and str(source["task_name"]) == str(row.task_name)
                and math.isclose(float(source["tau"]), float(row.tau), rel_tol=0.0, abs_tol=5e-15)
            )
            if not exact:
                raise ValueError(f"state-bank provenance mismatch at state {row.state_position}")
    if train_indices is not None:
        train_set = set(train_indices.detach().cpu().long().tolist())
        missing_train = set(primary["source_weight_index"].astype(int)) - train_set
        if missing_train:
            raise ValueError(f"primary bank contains non-checkpoint-training states: {sorted(missing_train)}")
    return {
        "state_count": len(bank),
        "primary_count": len(primary),
        "sentinel_count": len(sentinel),
        "excluded_identity_count": len(excluded_ids),
        "primary_unique_runs": int(primary["run"].nunique()),
        "sentinel_ids": list(EXPECTED_SENTINEL_IDS),
        "bridge_positions": list(BRIDGE_POSITIONS),
        "bridge_source_ids": list(BRIDGE_SOURCE_IDS),
    }


def unit_gram(gram: np.ndarray) -> np.ndarray:
    gram = np.asarray(gram, dtype=np.float64)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    if np.any(norms <= 0.0):
        raise ValueError("unit-direction analysis requires nonzero vectors")
    result = gram / np.outer(norms, norms)
    np.fill_diagonal(result, 1.0)
    if not np.isfinite(result).all():
        raise FloatingPointError("unit Gram is nonfinite")
    return result


def _qform(gram: np.ndarray, indices: Sequence[int]) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    return float(gram[np.ix_(idx, idx)].sum())


def _group_mean_norm2(gram: np.ndarray, indices: Sequence[int]) -> float:
    return _qform(gram, indices) / float(len(indices) ** 2)


def balanced_crossed_anova(
    gram: np.ndarray,
    task_labels: Sequence[str],
    *,
    q: int,
) -> pd.DataFrame:
    gram = np.asarray(gram, dtype=np.float64)
    labels = np.asarray(task_labels, dtype=object)
    state_count = len(labels)
    if gram.shape != (state_count * q, state_count * q):
        raise ValueError("Gram shape does not match state-major crossed design")
    tasks = list(dict.fromkeys(str(value) for value in labels))
    task_states = [np.flatnonzero(labels == task) for task in tasks]
    sizes = {len(values) for values in task_states}
    if len(tasks) != 2 or len(sizes) != 1:
        raise ValueError("frozen ANOVA requires two balanced fixed tasks")
    n = int(next(iter(sizes)))
    t = len(tasks)
    s = state_count
    total_n = s * q
    cells = np.arange(total_n, dtype=np.int64).reshape(s, q)
    total_ss = float(np.trace(gram))
    grand_norm2 = _group_mean_norm2(gram, cells.ravel())
    grand_ss = total_n * grand_norm2
    task_mean_energy = sum(_group_mean_norm2(gram, cells[idx].ravel()) for idx in task_states)
    state_mean_energy = sum(_group_mean_norm2(gram, cells[state]) for state in range(s))
    probe_mean_energy = sum(_group_mean_norm2(gram, cells[:, draw]) for draw in range(q))
    task_probe_energy = sum(
        _group_mean_norm2(gram, cells[idx, draw])
        for idx in task_states
        for draw in range(q)
    )
    ss_task = n * q * task_mean_energy - grand_ss
    ss_state = q * state_mean_energy - n * q * task_mean_energy
    ss_probe = s * probe_mean_energy - grand_ss
    ss_task_probe = n * task_probe_energy - n * q * task_mean_energy - s * probe_mean_energy + grand_ss
    ss_residual = total_ss - grand_ss - ss_task - ss_state - ss_probe - ss_task_probe
    rows = [
        ("grand", grand_ss, 1),
        ("task", ss_task, t - 1),
        ("state(task)", ss_state, t * (n - 1)),
        ("probe", ss_probe, q - 1),
        ("task*probe", ss_task_probe, (t - 1) * (q - 1)),
        ("state(task)*probe", ss_residual, t * (n - 1) * (q - 1)),
    ]
    result = pd.DataFrame(rows, columns=["component", "sum_squared_vector_norms", "degrees_freedom"])
    result["mean_square_trace"] = result["sum_squared_vector_norms"] / result["degrees_freedom"]
    ms = result.set_index("component")["mean_square_trace"]
    residual = float(ms["state(task)*probe"])
    task_probe = float(ms["task*probe"])
    estimates = {
        "grand": float("nan"),
        "task": float("nan"),
        "state(task)": (float(ms["state(task)"]) - residual) / q,
        "probe": (float(ms["probe"]) - task_probe) / s,
        "task*probe": (task_probe - residual) / n,
        "state(task)*probe": residual,
    }
    result["random_effect_trace_estimate"] = result["component"].map(estimates)
    if not np.isfinite(result[["sum_squared_vector_norms", "mean_square_trace"]].to_numpy()).all():
        raise FloatingPointError("ANOVA sufficient statistics are nonfinite")
    return result


def balanced_crossed_anova_from_task_grams(
    task_grams: Mapping[str, np.ndarray],
    cross_task_sum_gram: np.ndarray,
    *,
    q: int,
) -> pd.DataFrame:
    """Exact two-task ANOVA from within-task cell Grams and crossed task probe sums."""
    tasks = ("fashion_mnist", "mnist")
    if set(task_grams) != set(tasks):
        raise ValueError("exact frozen task names are required")
    shapes = {np.asarray(task_grams[task]).shape for task in tasks}
    if len(shapes) != 1:
        raise ValueError("task Grams must have equal shapes")
    size = next(iter(shapes))[0]
    if size % q != 0 or next(iter(shapes)) != (size, size):
        raise ValueError("invalid task Gram shape")
    n = size // q
    t = 2
    s = t * n
    total_n = s * q
    cross = np.asarray(cross_task_sum_gram, dtype=np.float64)
    if cross.shape != (q, q):
        raise ValueError("cross-task sum Gram must be QxQ")
    grams = {task: np.asarray(task_grams[task], dtype=np.float64) for task in tasks}
    cells = np.arange(n * q).reshape(n, q)
    total_ss = sum(float(np.trace(grams[task])) for task in tasks)
    task_numerators = {task: float(grams[task].sum()) for task in tasks}
    grand_ss = (sum(task_numerators.values()) + 2.0 * float(cross.sum())) / total_n
    task_mean_energy = sum(value / float((n * q) ** 2) for value in task_numerators.values())
    state_mean_energy = sum(
        _group_mean_norm2(grams[task], cells[state]) for task in tasks for state in range(n)
    )
    probe_mean_energy = 0.0
    task_probe_energy = 0.0
    for draw in range(q):
        own_numerators = {
            task: _qform(grams[task], cells[:, draw])
            for task in tasks
        }
        probe_mean_energy += (
            sum(own_numerators.values()) + 2.0 * float(cross[draw, draw])
        ) / float(s**2)
        task_probe_energy += sum(value / float(n**2) for value in own_numerators.values())
    ss_task = n * q * task_mean_energy - grand_ss
    ss_state = q * state_mean_energy - n * q * task_mean_energy
    ss_probe = s * probe_mean_energy - grand_ss
    ss_task_probe = n * task_probe_energy - n * q * task_mean_energy - s * probe_mean_energy + grand_ss
    ss_residual = total_ss - grand_ss - ss_task - ss_state - ss_probe - ss_task_probe
    rows = [
        ("grand", grand_ss, 1),
        ("task", ss_task, 1),
        ("state(task)", ss_state, t * (n - 1)),
        ("probe", ss_probe, q - 1),
        ("task*probe", ss_task_probe, q - 1),
        ("state(task)*probe", ss_residual, t * (n - 1) * (q - 1)),
    ]
    result = pd.DataFrame(rows, columns=["component", "sum_squared_vector_norms", "degrees_freedom"])
    result["mean_square_trace"] = result["sum_squared_vector_norms"] / result["degrees_freedom"]
    ms = result.set_index("component")["mean_square_trace"]
    residual = float(ms["state(task)*probe"])
    task_probe = float(ms["task*probe"])
    estimates = {
        "state(task)": (float(ms["state(task)"]) - residual) / q,
        "probe": (float(ms["probe"]) - task_probe) / s,
        "task*probe": (task_probe - residual) / n,
        "state(task)*probe": residual,
    }
    result["random_effect_trace_estimate"] = result["component"].map(estimates)
    centered_total = total_ss - grand_ss
    effect_total = float(result.loc[result["component"] != "grand", "sum_squared_vector_norms"].sum())
    tolerance = 2e-8 * max(abs(centered_total), 1.0)
    if abs(effect_total - centered_total) > tolerance:
        raise FloatingPointError("crossed ANOVA effects do not close to centered total SS")
    return result


def delta_from_crossed_gram(
    gram: np.ndarray,
    task_labels: Sequence[str],
    *,
    q: int,
) -> DeltaEstimate:
    gram = np.asarray(gram, dtype=np.float64)
    labels = np.asarray(task_labels, dtype=object)
    s = len(labels)
    if gram.shape != (s * q, s * q):
        raise ValueError("Gram shape does not match state-major cells")
    cells = np.arange(s * q).reshape(s, q)
    state_means = np.asarray([_group_mean_norm2(gram, cells[state]) for state in range(s)])
    within = np.asarray(
        [
            (float(np.trace(gram[np.ix_(cells[state], cells[state])])) - q * state_means[state]) / (q - 1)
            for state in range(s)
        ]
    )
    v_probe = float(within.mean())
    state_values: list[float] = []
    interaction_values: list[float] = []
    for task in dict.fromkeys(str(value) for value in labels):
        states = np.flatnonzero(labels == task)
        n = len(states)
        task_cells = cells[states]
        task_mean_norm2 = _group_mean_norm2(gram, task_cells.ravel())
        observed = (float(state_means[states].sum()) - n * task_mean_norm2) / (n - 1)
        total = float(np.trace(gram[np.ix_(task_cells.ravel(), task_cells.ravel())]))
        state_ss_including_mean = q * float(state_means[states].sum())
        probe_energy = sum(_group_mean_norm2(gram, task_cells[:, draw]) for draw in range(q))
        residual_ss = total - state_ss_including_mean - n * probe_energy + n * q * task_mean_norm2
        interaction_ms = residual_ss / ((n - 1) * (q - 1))
        interaction_values.append(float(interaction_ms))
        state_values.append(float(observed - interaction_ms / q))
    v_state = float(np.mean(state_values))
    return DeltaEstimate(
        v_probe=v_probe,
        v_state=v_state,
        delta=v_probe - v_state,
        state_interaction_ms=tuple(interaction_values),
    )


def signal_fraction_from_gram(gram: np.ndarray) -> float:
    gram = np.asarray(gram, dtype=np.float64)
    q = gram.shape[0]
    if gram.shape != (q, q) or q < 2:
        raise ValueError("signal fraction requires a square Q>=2 Gram")
    squared_sum = float(gram.sum())
    norm_sum = float(np.trace(gram))
    cross_draw_dot = (squared_sum - norm_sum) / (q * (q - 1))
    mean_squared_draw_norm = norm_sum / q
    return cross_draw_dot / mean_squared_draw_norm if mean_squared_draw_norm > 0.0 else float("nan")


def signal_fraction_bootstrap(
    gram: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | str]:
    gram = np.asarray(gram, dtype=np.float64)
    q = gram.shape[0]
    generator = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        selected = generator.integers(0, q, size=q)
        values[index] = signal_fraction_from_gram(gram[np.ix_(selected, selected)])
    low, high = np.quantile(values, [0.025, 0.975])
    category = "unresolved"
    if high < 0.05:
        category = "near-zero conditional signal supported"
    elif low > 0.05:
        category = "near-zero excluded"
    return {
        "signal_fraction": signal_fraction_from_gram(gram),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "category": category,
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


def gram_pairwise_rows(gram: np.ndarray, *, state_position: int, prefix: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in range(gram.shape[0]):
        for right in range(left + 1, gram.shape[0]):
            cosine, relative_error, norm_ratio = gram_pair_metrics(gram, left, right)
            rows.append(
                {
                    "state_position": int(state_position),
                    "prefix": int(prefix),
                    "left_draw": left,
                    "right_draw": right,
                    "gradient_cosine": cosine,
                    "gradient_relative_error": relative_error,
                    "gradient_norm_ratio": norm_ratio,
                }
            )
    return rows


def summarize_state_gram(gram: np.ndarray, *, state_position: int, prefix: int) -> dict[str, Any]:
    pairwise = pd.DataFrame(gram_pairwise_rows(gram, state_position=state_position, prefix=prefix))
    direction_gram = unit_gram(gram)
    raw_split_cosines = [gram_group_metrics(gram, left, right)[0] for left, right in SPLITS]
    unit_split_cosines = [
        gram_group_metrics(direction_gram, left, right)[0] for left, right in SPLITS
    ]
    raw_fold_cosines = [
        gram_group_metrics(gram, FOLDS[left], FOLDS[right])[0]
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    unit_fold_cosines = [
        gram_group_metrics(direction_gram, FOLDS[left], FOLDS[right])[0]
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    mean_norm2 = float(gram.mean())
    mean_squared_norm = float(np.mean(np.diag(gram)))
    largest_draw = int(np.argmax(np.diag(gram)))
    leave_largest = [draw for draw in range(gram.shape[0]) if draw != largest_draw]
    loo_cosine, _loo_error, loo_ratio = gram_group_metrics(
        gram,
        list(range(gram.shape[0])),
        leave_largest,
    )
    return {
        "state_position": int(state_position),
        "prefix": int(prefix),
        "pairwise_cosine_median": float(pairwise["gradient_cosine"].median()),
        "pairwise_cosine_q10": float(pairwise["gradient_cosine"].quantile(0.10)),
        "pairwise_cosine_q90": float(pairwise["gradient_cosine"].quantile(0.90)),
        "pairwise_cosine_min": float(pairwise["gradient_cosine"].min()),
        "pairwise_negative_fraction": float((pairwise["gradient_cosine"] < 0.0).mean()),
        "gradient_norm_mean": float(norms.mean()),
        "gradient_norm_rms": float(np.sqrt(np.mean(norms**2))),
        "gradient_norm_q99": float(np.quantile(norms, 0.99)),
        "gradient_norm_max": float(norms.max()),
        "gradient_norm_top1pct_mass": top_mass_share(norms, 0.01),
        "gradient_norm_top5pct_mass": top_mass_share(norms, 0.05),
        "mean_gradient_norm": math.sqrt(max(mean_norm2, 0.0)),
        "mean_gradient_norm_over_rms": math.sqrt(max(mean_norm2, 0.0) / max(mean_squared_norm, 1e-30)),
        "signal_fraction": signal_fraction_from_gram(gram),
        "split_vector_kind": "unit",
        "split_cosines": json.dumps(unit_split_cosines),
        "split_gate_pass": bool(all(value >= 0.80 for value in unit_split_cosines)),
        "raw_split_cosines": json.dumps(raw_split_cosines),
        "fold_vector_kind": "unit",
        "fold_cosine_median": float(np.median(unit_fold_cosines)),
        "fold_cosine_min": float(np.min(unit_fold_cosines)),
        "raw_fold_cosine_median": float(np.median(raw_fold_cosines)),
        "raw_fold_cosine_min": float(np.min(raw_fold_cosines)),
        "largest_norm_draw": largest_draw,
        "leave_largest_draw_out_mean_cosine": loo_cosine,
        "leave_largest_draw_out_norm_ratio": loo_ratio,
    }


def fold_split_rows(
    gram: np.ndarray,
    *,
    state_position: int,
    prefix: int,
    vector_kind: str = "raw",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, (left, right) in enumerate(SPLITS):
        cosine, relative_error, norm_ratio = gram_group_metrics(gram, left, right)
        rows.append(
            {
                "state_position": state_position,
                "prefix": prefix,
                "vector_kind": vector_kind,
                "comparison_kind": "split_half",
                "comparison": split,
                "left_draws": json.dumps(list(left)),
                "right_draws": json.dumps(list(right)),
                "gradient_cosine": cosine,
                "gradient_relative_error": relative_error,
                "gradient_norm_ratio": norm_ratio,
            }
        )
    comparison = 0
    for left in range(4):
        for right in range(left + 1, 4):
            cosine, relative_error, norm_ratio = gram_group_metrics(gram, FOLDS[left], FOLDS[right])
            rows.append(
                {
                    "state_position": state_position,
                    "prefix": prefix,
                    "vector_kind": vector_kind,
                    "comparison_kind": "fold_pair",
                    "comparison": comparison,
                    "left_fold": left,
                    "right_fold": right,
                    "left_draws": json.dumps(list(FOLDS[left])),
                    "right_draws": json.dumps(list(FOLDS[right])),
                    "gradient_cosine": cosine,
                    "gradient_relative_error": relative_error,
                    "gradient_norm_ratio": norm_ratio,
                }
            )
            comparison += 1
    return rows


def chunked_memmap_gram(
    left_path: Path,
    right_path: Path,
    *,
    left_rows: int,
    right_rows: int,
    dimension: int,
    chunk_size: int,
    device: torch.device,
    ranges: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    left = np.memmap(left_path, mode="r", dtype=np.float32, shape=(left_rows, dimension))
    right = np.memmap(right_path, mode="r", dtype=np.float32, shape=(right_rows, dimension))
    gram = torch.zeros((left_rows, right_rows), dtype=torch.float64)
    selected_ranges = ranges if ranges is not None else ((0, dimension),)
    for range_start, range_end in selected_ranges:
        for start in range(int(range_start), int(range_end), int(chunk_size)):
            stop = min(int(range_end), start + int(chunk_size))
            left_block = torch.from_numpy(np.asarray(left[:, start:stop]).copy()).to(device=device, dtype=torch.float64)
            right_block = torch.from_numpy(np.asarray(right[:, start:stop]).copy()).to(device=device, dtype=torch.float64)
            gram.add_((left_block @ right_block.T).detach().cpu())
            del left_block, right_block
    result = gram.numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite streamed Gram")
    return result


def chunked_many_memmap_gram(
    paths: Sequence[Path],
    *,
    rows_per_path: int,
    dimension: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    if not paths:
        raise ValueError("many-memmap Gram requires at least one path")
    maps = [
        np.memmap(path, mode="r", dtype=np.float32, shape=(rows_per_path, dimension))
        for path in paths
    ]
    row_count = len(maps) * rows_per_path
    gram = torch.zeros((row_count, row_count), dtype=torch.float64)
    for start in range(0, dimension, int(chunk_size)):
        stop = min(dimension, start + int(chunk_size))
        block_array = np.concatenate(
            [np.asarray(value[:, start:stop], dtype=np.float32) for value in maps],
            axis=0,
        )
        block = torch.from_numpy(block_array).to(device=device, dtype=torch.float64)
        gram.add_((block @ block.T).detach().cpu())
        del block, block_array
    result = gram.numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite many-memmap Gram")
    return result


def _scratch_path(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    return Path("/dev/shm") / f"a_fixed_state_probe_variance_h2048_{os.getpid()}"


def _create_scratch(path: Path) -> None:
    if path.exists():
        contents = list(path.iterdir()) if path.is_dir() else [path]
        suffix = "nonempty " if contents else "existing "
        raise FileExistsError(f"refusing {suffix}scratch path: {path}")
    path.mkdir(parents=True)
    atomic_write_json(
        path / "scratch_owner.json",
        {"protocol_id": PROTOCOL_ID, "pid": os.getpid(), "created_unix": time.time()},
    )


def _cleanup_owned_scratch(path: Path) -> None:
    marker = path / "scratch_owner.json"
    if not marker.exists():
        raise RuntimeError(f"refusing to clean scratch without ownership marker: {path}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID or int(payload.get("pid", -1)) != os.getpid():
        raise RuntimeError(f"refusing to clean scratch owned by another run: {path}")
    shutil.rmtree(path)


def accumulate_probe_sums(
    source_path: Path,
    raw_target: np.memmap,
    unit_target: np.memmap,
    *,
    dimension: int,
    norms: np.ndarray,
    chunk_size: int,
) -> None:
    source = np.memmap(source_path, mode="r", dtype=np.float32, shape=(DRAW_COUNT, dimension))
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise FloatingPointError("cannot accumulate unit directions from invalid norms")
    for start in range(0, dimension, int(chunk_size)):
        stop = min(dimension, start + int(chunk_size))
        block = np.asarray(source[:, start:stop], dtype=np.float32)
        raw_target[:, start:stop] += block
        unit_target[:, start:stop] += block / norms[:, None]


def _fixed_pairings() -> list[tuple[str, int, int]]:
    within_fashion = [("within_task", left, left + 1) for left in range(0, 16, 2)]
    within_mnist = [("within_task", left, left + 1) for left in range(16, 32, 2)]
    cross = [("cross_task", left, left + 16) for left in range(16)]
    return [*within_fashion, *within_mnist, *cross]


def reconstructed_update_gram(
    left_gram: np.ndarray,
    right_gram: np.ndarray,
    cross_gram: np.ndarray,
    *,
    independent: bool,
) -> np.ndarray:
    q = left_gram.shape[0]
    perm = (np.arange(q) + 17) % q if independent else np.arange(q)
    result = np.empty((q, q), dtype=np.float64)
    for a in range(q):
        for b in range(q):
            result[a, b] = 0.25 * (
                left_gram[a, b]
                + right_gram[perm[a], perm[b]]
                + cross_gram[a, perm[b]]
                + cross_gram[b, perm[a]]
            )
    return result


def _diagnostic_gram_summary(gram: np.ndarray) -> dict[str, float]:
    triangle = np.triu_indices(gram.shape[0], k=1)
    normalized = unit_gram(gram)
    values = normalized[triangle]
    return {
        "pairwise_cosine_median": float(np.median(values)),
        "pairwise_cosine_q10": float(np.quantile(values, 0.10)),
        "pairwise_cosine_q90": float(np.quantile(values, 0.90)),
        "pairwise_cosine_min": float(np.min(values)),
        "pairwise_negative_fraction": float(np.mean(values < 0.0)),
    }


def _source_identity(run_dir: Path) -> tuple[dict[str, str], dict[str, Path]]:
    paths = {
        "checkpoint": run_dir / "vae_checkpoint.pt",
        "weight_pool": run_dir / "weight_pool.pt",
        "records": run_dir / "weight_pool_records.csv",
        "config_json": run_dir / "config.json",
        "acceptance": run_dir / "baseline_acceptance.json",
        "protocol": PROTOCOL_PATH,
        "state_bank": STATE_BANK_PATH,
        "excluded_primary_sources": EXCLUDED_PRIMARY_PATH,
        "eligible_runs": ELIGIBLE_RUNS_PATH,
        "sentinel_parent_states": SENTINEL_PARENT_PATH,
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
        "config_py": ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/config.py",
        "core_py": ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py",
        "preconditioning_py": ROOT / "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py",
        "estimator_helper": ROOT / "scripts/audit_variant_a_estimator_stability.py",
        "hvp_helper": ROOT / "scripts/audit_variant_a_hvp_batch_size.py",
        "state_pair_helper": ROOT / "scripts/audit_variant_a_state_pair_variance.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}, paths


def _assert_source_identity(run_dir: Path) -> dict[str, str]:
    observed, _paths = _source_identity(run_dir)
    if observed != EXPECTED_SHA256:
        mismatches = {
            key: {"observed": observed.get(key), "expected": EXPECTED_SHA256.get(key)}
            for key in sorted(set(observed) | set(EXPECTED_SHA256))
            if observed.get(key) != EXPECTED_SHA256.get(key)
        }
        raise ValueError(f"frozen provenance hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    acceptance = json.loads((run_dir / "baseline_acceptance.json").read_text(encoding="utf-8"))
    if not bool(acceptance.get("passed")):
        raise ValueError("baseline acceptance is not passed")
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if saved.get("config_hash") != EXPECTED_CONFIG_HASH:
        raise ValueError("accepted saved config hash mismatch")
    return observed


def _materialized_p4_preflight(
    run: Any,
    active: Sequence[tuple[str, torch.nn.Parameter]],
    excluded: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    state_index: int,
    device: torch.device,
) -> dict[str, Any]:
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=PAIR_COUNT, batch_size=TRAIN_COUNT)
    weights = run.weights[[state_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weights).detach()
    record = run.records.iloc[[state_index]].to_dict(orient="records")[0]
    record["source_weight_index"] = int(state_index)
    generator_seed = stable_uint63(PROTOCOL_ID, BASE_SEED, "materialized_p4_preflight")
    run.vae.zero_grad(set_to_none=True)
    materialized, _stats = preconditioning_regularizer(
        cfg,
        state=PreconditioningState(),
        vae=run.vae,
        normalizer=run.normalizer,
        z_samples=z,
        records=[record],
        task_tensors=run.task_tensors,
        spec=run.spec,
        step=10,
        generator=torch.Generator(device="cpu").manual_seed(generator_seed),
    )
    materialized.backward()
    assert_excluded_gradients_zero(excluded)
    materialized_gradient = _flatten_active_gradients(active)
    direct_gradient = torch.zeros_like(materialized_gradient)
    direct_scalar = 0.0
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    for pair in range(PAIR_COUNT):
        run.vae.zero_grad(set_to_none=True)
        loss, stats = _atomic_pair_loss(
            cfg=cfg,
            run=run,
            z=z[0],
            record=record,
            step=10,
            pair_index=pair,
            probe_generator_1=generator,
            probe_generator_2=generator,
        )
        if stats["batch_1_hash"] != "full" or stats["batch_2_hash"] != "full":
            raise RuntimeError("P4 preflight did not use full CE")
        loss.backward()
        assert_excluded_gradients_zero(excluded)
        direct_gradient.add_(_flatten_active_gradients(active))
        direct_scalar += float(loss.detach().cpu())
    direct_gradient.div_(PAIR_COUNT)
    direct_scalar /= PAIR_COUNT
    result = {
        "state_index": int(state_index),
        "batch_size": TRAIN_COUNT,
        "pairs": PAIR_COUNT,
        "scalar_abs_error": abs(float(materialized.detach().cpu()) - direct_scalar),
        "gradient_cosine": vector_cosine(materialized_gradient, direct_gradient),
        "gradient_relative_error": vector_relative_error(direct_gradient, materialized_gradient),
    }
    if result["scalar_abs_error"] > 5e-5 or result["gradient_cosine"] < 0.99999 or result["gradient_relative_error"] > 5e-5:
        raise RuntimeError(f"full-CE P4 atomic preflight failed: {json.dumps(result, sort_keys=True)}")
    run.vae.zero_grad(set_to_none=True)
    return result


def _write_frame(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def validate_bridge_metadata(frame: pd.DataFrame, run: Any) -> dict[str, Any]:
    expected_rows = len(BRIDGE_POSITIONS) * 16 * PAIR_COUNT
    if len(frame) != expected_rows:
        raise ValueError(f"B=128 bridge has {len(frame)} rows, expected {expected_rows}")
    if frame.duplicated(["state_position", "draw", "pair"]).any():
        raise ValueError("B=128 bridge has duplicate cell keys")
    if set(frame["state_position"].astype(int)) != set(BRIDGE_POSITIONS):
        raise ValueError("B=128 bridge state positions mismatch")
    if not (frame["batch_size_effective"].astype(int) == 128).all():
        raise ValueError("B=128 bridge effective batch mismatch")
    for row in frame.itertuples(index=False):
        record = run.records.iloc[int(row.source_weight_index)].to_dict()
        record["source_weight_index"] = int(row.source_weight_index)
        task_set = _task_set_for_record(run.task_tensors, record)
        for branch in (1, 2):
            indices = _batch_indices(
                task_set,
                batch_size=128,
                step=int(row.step_key),
                sample_key=int(row.source_weight_index),
                pair_key=2 * int(row.pair) + branch - 1,
            )
            if indices is None:
                raise RuntimeError("B=128 bridge replay returned full CE")
            expected_hash = hashlib.sha256(indices.detach().cpu().numpy().tobytes()).hexdigest()
            if str(getattr(row, f"batch_{branch}_hash")) != expected_hash:
                raise ValueError("B=128 bridge batch hash replay mismatch")
            if int(getattr(row, f"batch_{branch}_offset")) != int(indices[0].detach().cpu()):
                raise ValueError("B=128 bridge batch offset replay mismatch")
            if str(getattr(row, f"batch_{branch}_indices")) != json.dumps(indices.detach().cpu().tolist()):
                raise ValueError("B=128 bridge batch-index replay mismatch")
    return {
        "row_count": len(frame),
        "cell_keys_unique": True,
        "batch_hash_offset_indices_replayed": True,
    }


def _runtime_log(stage: str, started: float, device: torch.device, **values: Any) -> None:
    payload = {"stage": stage, "elapsed_sec": time.perf_counter() - started, **values, **cuda_memory(device)}
    print(f"[fixed_state_probe] {json.dumps(payload, sort_keys=True, default=float)}", flush=True)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = _scratch_path(args.scratch_dir)
    _create_scratch(scratch)
    device = torch.device(str(args.device))
    lifecycle = {
        "status": "started",
        "protocol_id": PROTOCOL_ID,
        "started_unix": time.time(),
        "output_dir": str(output_dir),
        "scratch_dir": str(scratch),
        "executor_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_write_json(output_dir / "manifest.json", lifecycle)

    source_hashes = _assert_source_identity(run_dir)
    bank = _read_bank(STATE_BANK_PATH)
    excluded_sources = pd.read_csv(EXCLUDED_PRIMARY_PATH)
    eligibility = pd.read_csv(ELIGIBLE_RUNS_PATH, dtype={"run_selection_hash": str, "best_snapshot_selection_hash": str})
    sentinel_parent = pd.read_csv(SENTINEL_PARENT_PATH)
    _runtime_log("load_model_and_bank", started, device)
    run = _load_run(run_dir, device=device)
    bank_gate = validate_state_bank(
        bank,
        excluded_sources,
        eligibility=eligibility,
        sentinel_parent=sentinel_parent,
        records=run.records,
        train_indices=run.train_indices,
    )
    if int(run.cfg.vae_hidden_dim) != 2048 or int(run.cfg.latent_dim) != 512:
        raise ValueError("accepted architecture mismatch")
    if str(run.cfg.tiny_bigvae_output_mode).lower() != "direct" or torch_dtype(run.cfg) != torch.float32:
        raise ValueError("accepted decoder/dtype mismatch")
    runtime_config_hash = config_hash(run.cfg)
    if runtime_config_hash != EXPECTED_RUNTIME_CONFIG_HASH:
        raise ValueError("runtime config hash with current defaults mismatch")
    model_hash_before = model_state_sha256(run.vae)
    if model_hash_before != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("loaded model-state hash mismatch")

    _runtime_log("atomic_active_decoder_preflight", started, device)
    preflight, active = _preflight_decomposition(
        run,
        batch_size=TRAIN_COUNT,
        seed=stable_uint63(PROTOCOL_ID, BASE_SEED, "active_preflight"),
        device=device,
    )
    active_names = [name for name, _parameter in active]
    active_count = sum(parameter.numel() for _name, parameter in active)
    if active_count != EXPECTED_ACTIVE_COUNT:
        raise ValueError(f"active decoder count mismatch: {active_count}")
    excluded = [(name, parameter) for name, parameter in run.vae.named_parameters() if name not in set(active_names)]
    module_ranges, parameter_rows = _module_slices(active)
    scratch_usage = shutil.disk_usage(scratch)
    estimated_peak_scratch_bytes = int(active_count * 4 * 23 * DRAW_COUNT)
    if scratch_usage.free < estimated_peak_scratch_bytes:
        raise OSError(
            f"scratch has {scratch_usage.free} free bytes, below estimated peak "
            f"{estimated_peak_scratch_bytes}: {scratch}"
        )
    accepted_parameters = pd.read_csv(
        ROOT
        / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
        / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
    )
    observed_parameters = pd.DataFrame(parameter_rows)
    if not observed_parameters.astype(str).equals(accepted_parameters.astype(str)):
        raise ValueError("active parameter manifest differs from accepted audit")
    p4_preflight = _materialized_p4_preflight(
        run,
        active,
        excluded,
        state_index=int(bank.iloc[0]["source_weight_index"]),
        device=device,
    )
    pd.DataFrame(parameter_rows).to_csv(output_dir / "active_parameters.csv", index=False)
    atomic_write_json(
        output_dir / "preflight.json",
        {"existing_helper_preflight": preflight, "full_ce_materialized_p4": p4_preflight},
    )
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "seed": BASE_SEED,
        "device": str(device),
        "dtype": "float32",
        "cache_mode": "read_only_checkpoint_pool_and_records",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "scratch_dir": str(scratch),
        "scratch_free_bytes_at_start": scratch_usage.free,
        "estimated_peak_scratch_bytes": estimated_peak_scratch_bytes,
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
        "source_sha256": source_hashes,
        "saved_config_hash": EXPECTED_CONFIG_HASH,
        "runtime_config_hash_with_current_defaults": runtime_config_hash,
        "executor_sha256": sha256_file(Path(__file__).resolve()),
        "active_parameter_count": active_count,
        "active_parameter_names": active_names,
        "excluded_parameter_names": [name for name, _parameter in excluded],
        "bank_gate": bank_gate,
        "repository": git_metadata(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    atomic_write_json(output_dir / "resolved_config.json", resolved)
    print(f"[fixed_state_probe] resolved_config={json.dumps(resolved, sort_keys=True)}", flush=True)
    if args.preflight_only:
        if model_state_sha256(run.vae) != model_hash_before:
            raise RuntimeError("model changed during preflight")
        _cleanup_owned_scratch(scratch)
        return {**lifecycle, "status": "complete", "mode": "preflight_only", "elapsed_sec": time.perf_counter() - started}

    cfg_full = _probe_cfg(run.cfg, sample_count=1, pair_count=1, batch_size=TRAIN_COUNT)
    cfg_bridge = _probe_cfg(run.cfg, sample_count=1, pair_count=1, batch_size=128)
    atomic_rows: list[dict[str, Any]] = []
    bridge_atomic_rows: list[dict[str, Any]] = []
    bridge_paired_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    prefix_grams: dict[str, np.ndarray] = {}
    module_grams: dict[str, np.ndarray] = {}
    p4_paths: dict[int, Path] = {}
    state_grams: dict[int, np.ndarray] = {}
    probe_hashes: dict[tuple[int, int, int], str] = {}
    latent_hashes: dict[int, str] = {}
    task_grams = {
        "fashion_mnist": np.zeros((16 * DRAW_COUNT, 16 * DRAW_COUNT), dtype=np.float64),
        "mnist": np.zeros((16 * DRAW_COUNT, 16 * DRAW_COUNT), dtype=np.float64),
    }
    cross_pair_grams: dict[str, np.ndarray] = {}
    bridge_grams: dict[str, np.ndarray] = {}
    task_sum_paths: dict[tuple[str, str], Path] = {}
    task_sum_maps: dict[tuple[str, str], np.memmap] = {}
    for task in ("fashion_mnist", "mnist"):
        for vector_kind in ("raw", "unit"):
            path = scratch / f"{task}_{vector_kind}_probe_sums.f32"
            task_sum_paths[(task, vector_kind)] = path
            task_sum_maps[(task, vector_kind)] = np.memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(DRAW_COUNT, active_count),
            )
            task_sum_maps[(task, vector_kind)][:] = 0.0

    for state_number, bank_row in enumerate(bank.itertuples(index=False)):
        state_started = time.perf_counter()
        position = int(bank_row.state_position)
        if position == 16:
            _runtime_log("fashion_task_gram", started, device)
            task_grams["fashion_mnist"] = chunked_many_memmap_gram(
                [p4_paths[index] for index in range(16)],
                rows_per_path=DRAW_COUNT,
                dimension=active_count,
                chunk_size=int(args.gram_chunk_size),
                device=device,
            )
        elif position == 32:
            _runtime_log("mnist_task_gram", started, device)
            task_grams["mnist"] = chunked_many_memmap_gram(
                [p4_paths[index] for index in range(16, 32)],
                rows_per_path=DRAW_COUNT,
                dimension=active_count,
                chunk_size=int(args.gram_chunk_size),
                device=device,
            )
        source_index = int(bank_row.source_weight_index)
        selected = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
        with torch.no_grad():
            z = encode_weights(run.vae, run.normalizer, selected).detach()[0]
        latent_hash = sha256_tensor(z)
        latent_hashes[position] = latent_hash
        record = run.records.iloc[source_index].to_dict()
        record["source_weight_index"] = source_index
        paths = {prefix: scratch / f"state_{position:02d}_p{prefix}.f32" for prefix in PREFIXES}
        maps = {
            prefix: np.memmap(path, mode="w+", dtype=np.float32, shape=(DRAW_COUNT, active_count))
            for prefix, path in paths.items()
        }
        for draw in range(DRAW_COUNT):
            gradient_sum = torch.zeros(active_count, dtype=torch.float32)
            scalar_sum = 0.0
            for pair in range(PAIR_COUNT):
                generator_1, generator_2, seed_1, seed_2 = common_probe_generators(draw, pair)
                run.vae.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                unit_started = time.perf_counter()
                loss, stats = _atomic_pair_loss(
                    cfg=cfg_full,
                    run=run,
                    z=z,
                    record=record,
                    step=10 * (draw + 1),
                    pair_index=pair,
                    probe_generator_1=generator_1,
                    probe_generator_2=generator_2,
                )
                loss.backward()
                assert_excluded_gradients_zero(excluded)
                gradient = _flatten_active_gradients(active)
                if not torch.isfinite(gradient).all() or not math.isfinite(float(loss.detach().cpu())):
                    raise FloatingPointError(f"nonfinite atomic cell state={position} draw={draw} pair={pair}")
                gradient_sum.add_(gradient)
                scalar_sum += float(loss.detach().cpu())
                for branch, probe_hash in ((0, str(stats["probe_1_hash"])), (1, str(stats["probe_2_hash"]))):
                    key = (draw, pair, branch)
                    previous = probe_hashes.setdefault(key, probe_hash)
                    if previous != probe_hash:
                        raise RuntimeError(f"common probe hash changed across states for key={key}")
                if stats["batch_1_hash"] != "full" or stats["batch_2_hash"] != "full":
                    raise RuntimeError("primary/sentinel production branch is not categorical full CE")
                atomic_rows.append(
                    {
                        "panel": bank_row.panel,
                        "state_position": position,
                        "source_weight_index": source_index,
                        "task_name": bank_row.task_name,
                        "draw": draw,
                        "pair": pair,
                        "step_key": 10 * (draw + 1),
                        "latent_hash": latent_hash,
                        "probe_seed_1": seed_1,
                        "probe_seed_2": seed_2,
                        "probe_hash_1": stats["probe_1_hash"],
                        "probe_hash_2": stats["probe_2_hash"],
                        "a_scalar": float(loss.detach().cpu()),
                        "atomic_gradient_norm": vector_norm(gradient),
                        "h1_norm": stats["h1_norm"],
                        "h2_norm": stats["h2_norm"],
                        "h1_h2_dot": stats["h1_h2_dot"],
                        "h1_hash": stats["h1_hash"],
                        "h2_hash": stats["h2_hash"],
                        "batch_1_hash": stats["batch_1_hash"],
                        "batch_2_hash": stats["batch_2_hash"],
                        "batch_train_count": stats["batch_train_count"],
                        "batch_size_effective": stats["batch_size_effective"],
                        "elapsed_sec": time.perf_counter() - unit_started,
                        **cuda_memory(device),
                    }
                )
                if pair + 1 in PREFIXES:
                    prefix = pair + 1
                    maps[prefix][draw] = gradient_sum.numpy() / float(prefix)
                    scalar_rows.append(
                        {
                            "panel": bank_row.panel,
                            "state_position": position,
                            "task_name": bank_row.task_name,
                            "draw": draw,
                            "prefix": prefix,
                            "a_scalar": scalar_sum / prefix,
                        }
                    )
                del gradient, loss
            if (draw + 1) % int(args.log_every) == 0 or draw + 1 == DRAW_COUNT:
                _runtime_log(
                    "state_progress",
                    started,
                    device,
                    state=f"{state_number + 1}/{STATE_COUNT}",
                    state_position=position,
                    panel=bank_row.panel,
                    task=bank_row.task_name,
                    draw=f"{draw + 1}/{DRAW_COUNT}",
                    atomic=f"{len(atomic_rows)}/{STATE_COUNT * DRAW_COUNT * PAIR_COUNT + len(BRIDGE_POSITIONS) * 16 * PAIR_COUNT}",
                    state_elapsed_sec=time.perf_counter() - state_started,
                )
        for value in maps.values():
            value.flush()
        for prefix in PREFIXES:
            gram = chunked_memmap_gram(
                paths[prefix],
                paths[prefix],
                left_rows=DRAW_COUNT,
                right_rows=DRAW_COUNT,
                dimension=active_count,
                chunk_size=int(args.gram_chunk_size),
                device=device,
            )
            prefix_grams[f"state_{position:02d}_p{prefix}"] = gram
            pairwise_rows.extend(gram_pairwise_rows(gram, state_position=position, prefix=prefix))
            fold_rows.extend(
                fold_split_rows(
                    gram,
                    state_position=position,
                    prefix=prefix,
                    vector_kind="raw",
                )
            )
            fold_rows.extend(
                fold_split_rows(
                    unit_gram(gram),
                    state_position=position,
                    prefix=prefix,
                    vector_kind="unit",
                )
            )
            summary = summarize_state_gram(gram, state_position=position, prefix=prefix)
            summary.update({"panel": bank_row.panel, "task_name": bank_row.task_name})
            state_rows.append(summary)
            if prefix == 4:
                state_grams[position] = gram
                signal = signal_fraction_bootstrap(
                    gram,
                    seed=stable_uint63(PROTOCOL_ID, BASE_SEED, "signal_bootstrap", position),
                    draws=int(args.signal_bootstrap_draws),
                )
                signal_rows.append({"state_position": position, "panel": bank_row.panel, **signal})
        for module, ranges in module_ranges.items():
            module_grams[f"state_{position:02d}_{module}"] = chunked_memmap_gram(
                paths[4],
                paths[4],
                left_rows=DRAW_COUNT,
                right_rows=DRAW_COUNT,
                dimension=active_count,
                chunk_size=int(args.gram_chunk_size),
                device=device,
                ranges=ranges,
            )
        if position in BRIDGE_POSITIONS:
            bridge_path = scratch / f"state_{position:02d}_bridge_b128_p4.f32"
            bridge_map = np.memmap(
                bridge_path,
                mode="w+",
                dtype=np.float32,
                shape=(16, active_count),
            )
            full_scalar_lookup = {
                (int(row["draw"]), int(row["pair"])): float(row["a_scalar"])
                for row in atomic_rows
                if int(row["state_position"]) == position and int(row["draw"]) < 16
            }
            bridge_scalar_by_draw: dict[int, float] = {}
            for draw in range(16):
                gradient_sum = torch.zeros(active_count, dtype=torch.float32)
                scalar_sum = 0.0
                for pair in range(PAIR_COUNT):
                    generator_1, generator_2, seed_1, seed_2 = common_probe_generators(draw, pair)
                    run.vae.zero_grad(set_to_none=True)
                    unit_started = time.perf_counter()
                    loss, stats = _atomic_pair_loss(
                        cfg=cfg_bridge,
                        run=run,
                        z=z,
                        record=record,
                        step=10 * (draw + 1),
                        pair_index=pair,
                        probe_generator_1=generator_1,
                        probe_generator_2=generator_2,
                    )
                    loss.backward()
                    assert_excluded_gradients_zero(excluded)
                    gradient = _flatten_active_gradients(active)
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError(f"nonfinite bridge gradient state={position} draw={draw} pair={pair}")
                    gradient_sum.add_(gradient)
                    scalar_sum += float(loss.detach().cpu())
                    for branch, probe_hash in ((0, str(stats["probe_1_hash"])), (1, str(stats["probe_2_hash"]))):
                        if probe_hashes[(draw, pair, branch)] != probe_hash:
                            raise RuntimeError("B=128 bridge did not replay the common full-CE probe")
                    if stats["batch_1_hash"] == "full" or stats["batch_2_hash"] == "full":
                        raise RuntimeError("B=128 bridge unexpectedly used full CE")
                    bridge_atomic_rows.append(
                        {
                            "state_position": position,
                            "source_weight_index": source_index,
                            "task_name": bank_row.task_name,
                            "draw": draw,
                            "pair": pair,
                            "step_key": 10 * (draw + 1),
                            "latent_hash": latent_hash,
                            "probe_seed_1": seed_1,
                            "probe_seed_2": seed_2,
                            "probe_hash_1": stats["probe_1_hash"],
                            "probe_hash_2": stats["probe_2_hash"],
                            "a_scalar": float(loss.detach().cpu()),
                            "full_ce_a_scalar": full_scalar_lookup[(draw, pair)],
                            "atomic_gradient_norm": vector_norm(gradient),
                            "h1_norm": stats["h1_norm"],
                            "h2_norm": stats["h2_norm"],
                            "h1_hash": stats["h1_hash"],
                            "h2_hash": stats["h2_hash"],
                            "batch_1_hash": stats["batch_1_hash"],
                            "batch_2_hash": stats["batch_2_hash"],
                            "batch_1_offset": stats["batch_1_offset"],
                            "batch_2_offset": stats["batch_2_offset"],
                            "batch_1_indices": stats["batch_1_indices"],
                            "batch_2_indices": stats["batch_2_indices"],
                            "batch_train_count": stats["batch_train_count"],
                            "batch_size_effective": stats["batch_size_effective"],
                            "elapsed_sec": time.perf_counter() - unit_started,
                            **cuda_memory(device),
                        }
                    )
                    del gradient, loss
                bridge_map[draw] = gradient_sum.numpy() / PAIR_COUNT
                bridge_scalar_by_draw[draw] = scalar_sum / PAIR_COUNT
            bridge_map.flush()
            bridge_self = chunked_memmap_gram(
                bridge_path,
                bridge_path,
                left_rows=16,
                right_rows=16,
                dimension=active_count,
                chunk_size=int(args.gram_chunk_size),
                device=device,
            )
            full_bridge = chunked_memmap_gram(
                paths[4],
                bridge_path,
                left_rows=DRAW_COUNT,
                right_rows=16,
                dimension=active_count,
                chunk_size=int(args.gram_chunk_size),
                device=device,
            )
            bridge_grams[f"state_{position:02d}_b128_self"] = bridge_self
            bridge_grams[f"state_{position:02d}_full_by_b128"] = full_bridge
            full_gram = state_grams[position]
            full_scalars = pd.DataFrame(scalar_rows)
            full_scalars = full_scalars.loc[
                (full_scalars["state_position"] == position)
                & (full_scalars["prefix"] == 4)
                & (full_scalars["draw"] < 16)
            ].set_index("draw")["a_scalar"]
            for draw in range(16):
                full_norm2 = float(full_gram[draw, draw])
                bridge_norm2 = float(bridge_self[draw, draw])
                dot = float(full_bridge[draw, draw])
                denominator = math.sqrt(max(full_norm2 * bridge_norm2, 0.0))
                error2 = max(full_norm2 + bridge_norm2 - 2.0 * dot, 0.0)
                bridge_paired_rows.append(
                    {
                        "state_position": position,
                        "source_weight_index": source_index,
                        "task_name": bank_row.task_name,
                        "draw": draw,
                        "step_key": 10 * (draw + 1),
                        "gradient_cosine_b128_to_full": dot / denominator if denominator > 0.0 else float("nan"),
                        "gradient_relative_error_b128_to_full": math.sqrt(error2) / max(math.sqrt(full_norm2), 1e-30),
                        "gradient_norm_ratio_b128_to_full": math.sqrt(bridge_norm2) / max(math.sqrt(full_norm2), 1e-30),
                        "a_scalar_b128": bridge_scalar_by_draw[draw],
                        "a_scalar_full": float(full_scalars.loc[draw]),
                    }
                )
            del bridge_map
            bridge_path.unlink()
        del maps
        for prefix in (1, 2):
            paths[prefix].unlink()
        if bank_row.panel == "primary":
            p4_paths[position] = paths[4]
            task = str(bank_row.task_name)
            accumulate_probe_sums(
                paths[4],
                task_sum_maps[(task, "raw")],
                task_sum_maps[(task, "unit")],
                dimension=active_count,
                norms=np.sqrt(np.maximum(np.diag(state_grams[position]), 0.0)),
                chunk_size=int(args.gram_chunk_size),
            )
            if task == "mnist":
                fashion_position = position - 16
                cross = chunked_memmap_gram(
                    p4_paths[fashion_position],
                    paths[4],
                    left_rows=DRAW_COUNT,
                    right_rows=DRAW_COUNT,
                    dimension=active_count,
                    chunk_size=int(args.gram_chunk_size),
                    device=device,
                )
                cross_pair_grams[f"pair_{fashion_position:02d}_{position:02d}"] = cross
                p4_paths[fashion_position].unlink()
                del p4_paths[fashion_position]
        else:
            paths[4].unlink()
        del z, selected
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _runtime_log("write_grams_and_analysis", started, device)
    for _kind, left, right in _fixed_pairings():
        if left < 16 and right < 16:
            gram = task_grams["fashion_mnist"]
            left_local, right_local = left, right
        elif left >= 16 and right >= 16:
            gram = task_grams["mnist"]
            left_local, right_local = left - 16, right - 16
        else:
            continue
        cross_pair_grams[f"pair_{left:02d}_{right:02d}"] = gram[
            left_local * DRAW_COUNT : (left_local + 1) * DRAW_COUNT,
            right_local * DRAW_COUNT : (right_local + 1) * DRAW_COUNT,
        ].copy()
    for task_sum in task_sum_maps.values():
        task_sum.flush()
    cross_task_sum_grams = {
        vector_kind: chunked_memmap_gram(
            task_sum_paths[("fashion_mnist", vector_kind)],
            task_sum_paths[("mnist", vector_kind)],
            left_rows=DRAW_COUNT,
            right_rows=DRAW_COUNT,
            dimension=active_count,
            chunk_size=int(args.gram_chunk_size),
            device=device,
        )
        for vector_kind in ("raw", "unit")
    }
    np.savez_compressed(output_dir / "state_prefix_grams.npz", **prefix_grams)
    np.savez_compressed(output_dir / "state_module_p4_grams.npz", **module_grams)
    np.savez_compressed(output_dir / "primary_task_p4_grams.npz", **task_grams)
    np.savez_compressed(output_dir / "reconstruction_cross_grams.npz", **cross_pair_grams)
    np.savez_compressed(output_dir / "cross_task_probe_sum_grams.npz", **cross_task_sum_grams)
    np.savez_compressed(output_dir / "bridge_p4_grams.npz", **bridge_grams)
    _write_frame(output_dir / "atomic_gradient_metadata.csv", atomic_rows)
    _write_frame(output_dir / "bridge_atomic_gradient_metadata.csv", bridge_atomic_rows)
    _write_frame(output_dir / "bridge_paired_p4_metrics.csv", bridge_paired_rows)
    _write_frame(output_dir / "prefix_scalar_cells.csv", scalar_rows)
    _write_frame(output_dir / "state_prefix_summary.csv", state_rows)
    _write_frame(output_dir / "draw_pairwise_metrics.csv", pairwise_rows)
    _write_frame(output_dir / "fold_split_mean_metrics.csv", fold_rows)
    _write_frame(output_dir / "signal_fraction.csv", signal_rows)
    atomic = pd.DataFrame(atomic_rows)
    bridge_atomic = pd.DataFrame(bridge_atomic_rows)
    bridge_gate = validate_bridge_metadata(bridge_atomic, run)
    state_summary = pd.DataFrame(state_rows)

    module_summary_rows: list[dict[str, Any]] = []
    for key, gram in module_grams.items():
        _state_token, position_token, module = key.split("_", 2)
        module_summary_rows.append(
            {
                "state_position": int(position_token),
                "module": module,
                **_diagnostic_gram_summary(gram),
                "gradient_energy_mean": float(np.mean(np.diag(gram))),
            }
        )
    pd.DataFrame(module_summary_rows).to_csv(output_dir / "module_p4_summary.csv", index=False)

    def pair_cross_gram(left: int, right: int) -> np.ndarray:
        if left // 16 == right // 16:
            task = "fashion_mnist" if left < 16 else "mnist"
            offset = 0 if task == "fashion_mnist" else 16
            left_local, right_local = left - offset, right - offset
            return task_grams[task][
                left_local * DRAW_COUNT : (left_local + 1) * DRAW_COUNT,
                right_local * DRAW_COUNT : (right_local + 1) * DRAW_COUNT,
            ]
        key = f"pair_{min(left, right):02d}_{max(left, right):02d}"
        value = cross_pair_grams[key]
        return value if left < right else value.T

    reconstruction_rows: list[dict[str, Any]] = []
    for position in range(PRIMARY_COUNT):
        probe_only = reconstructed_update_gram(
            state_grams[position],
            state_grams[position],
            state_grams[position],
            independent=True,
        )
        reconstruction_rows.append(
            {
                "diagnostic": "probe_only_fixed_state",
                "state_left": position,
                "state_right": position,
                "probe_pairing": "q_to_q_plus_17_mod_64",
                **_diagnostic_gram_summary(probe_only),
            }
        )
    for pair_kind, left, right in _fixed_pairings():
        cross = pair_cross_gram(left, right)
        left_gram, right_gram = state_grams[left], state_grams[right]
        for independent in (False, True):
            update_gram = reconstructed_update_gram(
                left_gram,
                right_gram,
                cross,
                independent=independent,
            )
            reconstruction_rows.append(
                {
                    "diagnostic": "both_state_and_probe",
                    "pair_kind": pair_kind,
                    "state_left": left,
                    "state_right": right,
                    "probe_pairing": "independent_q_plus_17" if independent else "common_q",
                    **_diagnostic_gram_summary(update_gram),
                }
            )
        left_mean_norm2 = float(left_gram.mean())
        right_mean_norm2 = float(right_gram.mean())
        mean_dot = float(cross.mean())
        denominator = math.sqrt(max(left_mean_norm2 * right_mean_norm2, 0.0))
        reconstruction_rows.append(
            {
                "diagnostic": "state_only_conditional_means",
                "pair_kind": pair_kind,
                "state_left": left,
                "state_right": right,
                "probe_pairing": "all_64_conditional_means",
                "pairwise_cosine_median": mean_dot / denominator if denominator > 0.0 else float("nan"),
            }
        )
    pd.DataFrame(reconstruction_rows).to_csv(output_dir / "reconstructed_s2_diagnostics.csv", index=False)

    p4_gate = state_summary.loc[state_summary["prefix"] == 4].set_index("state_position")["split_gate_pass"]
    unit_state_grams = {position: unit_gram(gram) for position, gram in state_grams.items()}
    unit_task_grams = {task: unit_gram(gram) for task, gram in task_grams.items()}

    def pair_unit_cross_gram(left: int, right: int) -> np.ndarray:
        if left // 16 == right // 16:
            task = "fashion_mnist" if left < 16 else "mnist"
            offset = 0 if task == "fashion_mnist" else 16
            left_local, right_local = left - offset, right - offset
            return unit_task_grams[task][
                left_local * DRAW_COUNT : (left_local + 1) * DRAW_COUNT,
                right_local * DRAW_COUNT : (right_local + 1) * DRAW_COUNT,
            ]
        raw_cross = pair_cross_gram(left, right)
        left_norms = np.sqrt(np.maximum(np.diag(state_grams[left]), 0.0))
        right_norms = np.sqrt(np.maximum(np.diag(state_grams[right]), 0.0))
        if np.any(left_norms <= 0.0) or np.any(right_norms <= 0.0):
            raise FloatingPointError("cross-state unit Gram requires nonzero gradients")
        return raw_cross / np.outer(left_norms, right_norms)

    crossfit_rows: list[dict[str, Any]] = []
    state_pairs = [
        (left, right)
        for task_start in (0, 16)
        for left in range(task_start, task_start + 16)
        for right in range(left + 1, task_start + 16)
    ]
    state_pairs.extend((left, left + 16) for left in range(16))
    for left, right in state_pairs:
        cross = pair_unit_cross_gram(left, right)
        for split, (left_draws, right_draws) in enumerate(SPLITS):
            left_idx = np.asarray(left_draws, dtype=np.int64)
            right_idx = np.asarray(right_draws, dtype=np.int64)
            left_a_norm2 = float(unit_state_grams[left][np.ix_(left_idx, left_idx)].mean())
            left_b_norm2 = float(unit_state_grams[left][np.ix_(right_idx, right_idx)].mean())
            right_a_norm2 = float(unit_state_grams[right][np.ix_(left_idx, left_idx)].mean())
            right_b_norm2 = float(unit_state_grams[right][np.ix_(right_idx, right_idx)].mean())
            forward_dot = float(cross[np.ix_(left_idx, right_idx)].mean())
            reverse_dot = float(cross[np.ix_(right_idx, left_idx)].mean())
            forward_denominator = math.sqrt(max(left_a_norm2 * right_b_norm2, 0.0))
            reverse_denominator = math.sqrt(max(left_b_norm2 * right_a_norm2, 0.0))
            forward = (
                forward_dot / forward_denominator
                if forward_denominator > 0.0
                else float("nan")
            )
            reverse = (
                reverse_dot / reverse_denominator
                if reverse_denominator > 0.0
                else float("nan")
            )
            crossfit_rows.append(
                {
                    "state_left": left,
                    "state_right": right,
                    "same_task": left // 16 == right // 16,
                    "split": split,
                    "left_draws": json.dumps(list(left_draws)),
                    "right_draws": json.dumps(list(right_draws)),
                    "crossfit_cosine_forward": forward,
                    "crossfit_cosine_reverse": reverse,
                    "crossfit_cosine": 0.5 * (forward + reverse),
                    "both_states_pass_conditional_mean_gate": bool(p4_gate[left] and p4_gate[right]),
                }
            )
    crossfit_frame = pd.DataFrame(crossfit_rows)
    crossfit_frame.to_csv(output_dir / "crossfit_state_mean_cosines.csv", index=False)

    passing_within = crossfit_frame.loc[
        crossfit_frame["same_task"]
        & crossfit_frame["both_states_pass_conditional_mean_gate"]
    ]
    within_split_medians = (
        passing_within.groupby("split")["crossfit_cosine"].median().reindex(range(3))
    )
    within_state_nonalignment_gate = bool(
        int(p4_gate.iloc[:PRIMARY_COUNT].sum()) >= 24
        and within_split_medians.notna().all()
        and (within_split_medians < 0.80).all()
    )

    task_probe_grams = {
        task: aggregate_task_probe_gram(gram) for task, gram in unit_task_grams.items()
    }
    task_gate_rows: list[dict[str, Any]] = []
    task_cross = cross_task_sum_grams["unit"]
    for split, (left_draws, right_draws) in enumerate(SPLITS):
        left_idx = np.asarray(left_draws, dtype=np.int64)
        right_idx = np.asarray(right_draws, dtype=np.int64)
        fashion_stability = gram_group_metrics(
            task_probe_grams["fashion_mnist"], left_idx, right_idx
        )[0]
        mnist_stability = gram_group_metrics(
            task_probe_grams["mnist"], left_idx, right_idx
        )[0]
        fashion_left = _group_mean_norm2(task_probe_grams["fashion_mnist"], left_idx)
        fashion_right = _group_mean_norm2(task_probe_grams["fashion_mnist"], right_idx)
        mnist_left = _group_mean_norm2(task_probe_grams["mnist"], left_idx)
        mnist_right = _group_mean_norm2(task_probe_grams["mnist"], right_idx)
        forward_dot = float(task_cross[np.ix_(left_idx, right_idx)].mean())
        reverse_dot = float(task_cross[np.ix_(right_idx, left_idx)].mean())
        forward = forward_dot / math.sqrt(max(fashion_left * mnist_right, 1e-300))
        reverse = reverse_dot / math.sqrt(max(fashion_right * mnist_left, 1e-300))
        task_gate_rows.append(
            {
                "split": split,
                "fashion_self_stability": fashion_stability,
                "mnist_self_stability": mnist_stability,
                "fashion_mnist_crossfit_cosine": 0.5 * (forward + reverse),
            }
        )
    task_gate_frame = pd.DataFrame(task_gate_rows)
    task_nonalignment_gate = bool(
        (task_gate_frame[["fashion_self_stability", "mnist_self_stability"]] >= 0.80)
        .all()
        .all()
        and (task_gate_frame["fashion_mnist_crossfit_cosine"] < 0.80).all()
    )
    gate_rows = [
        {
            "gate": "within_task_state_nonalignment",
            "passed": within_state_nonalignment_gate,
            **{f"split_{split}_median_cosine": within_split_medians.loc[split] for split in range(3)},
        },
        {
            "gate": "fixed_task_panel_nonalignment",
            "passed": task_nonalignment_gate,
            **{
                f"split_{int(row.split)}_crossfit_cosine": row.fashion_mnist_crossfit_cosine
                for row in task_gate_frame.itertuples(index=False)
            },
        },
    ]
    pd.DataFrame(gate_rows).to_csv(output_dir / "nonalignment_gates.csv", index=False)
    task_gate_frame.to_csv(output_dir / "fixed_task_panel_crossfit_cosines.csv", index=False)

    panel_rows: list[dict[str, Any]] = []
    bank_index = bank.set_index("state_position")
    for position in range(STATE_COUNT):
        gram = state_grams[position]
        atomic_group = atomic.loc[atomic["state_position"] == position]
        row = bank_index.loc[position]
        p4 = state_summary.loc[
            (state_summary["state_position"] == position) & (state_summary["prefix"] == 4)
        ].iloc[0]
        panel_rows.append(
            {
                "state_position": position,
                "panel": row["panel"],
                "task_name": row["task_name"],
                "step_stratum": int(row["step_stratum"]),
                "source_weight_index": int(row["source_weight_index"]),
                "prior_stratum": row["prior_stratum"],
                "prior_grad_rms": row["prior_grad_rms"],
                "fresh_a_mean": float(atomic_group["a_scalar"].mean()),
                "fresh_atomic_gradient_rms": float(np.sqrt(np.mean(atomic_group["atomic_gradient_norm"] ** 2))),
                "fresh_hvp_rms": float(
                    np.sqrt(np.mean(np.concatenate([atomic_group["h1_norm"].to_numpy() ** 2, atomic_group["h2_norm"].to_numpy() ** 2])))
                ),
                "conditional_probe_variance_W": (
                    float(np.trace(gram)) - DRAW_COUNT * float(gram.mean())
                )
                / (DRAW_COUNT - 1),
                "conditional_mean_gradient_norm": p4["mean_gradient_norm"],
                "pairwise_cosine_median": p4["pairwise_cosine_median"],
            }
        )
    panel_metrics = pd.DataFrame(panel_rows)
    panel_metrics.to_csv(output_dir / "panel_state_metrics.csv", index=False)
    sentinel_metrics = panel_metrics.loc[panel_metrics["panel"] == "sentinel"].copy()
    sentinel_metrics.groupby(["task_name", "prior_stratum"], as_index=False).agg(
        state_count=("state_position", "size"),
        prior_grad_rms_mean=("prior_grad_rms", "mean"),
        fresh_a_mean=("fresh_a_mean", "mean"),
        fresh_atomic_gradient_rms_mean=("fresh_atomic_gradient_rms", "mean"),
        fresh_hvp_rms_mean=("fresh_hvp_rms", "mean"),
        conditional_probe_variance_W_mean=("conditional_probe_variance_W", "mean"),
        conditional_mean_gradient_norm_mean=("conditional_mean_gradient_norm", "mean"),
    ).to_csv(output_dir / "sentinel_stratum_summary.csv", index=False)
    sentinel_rows: list[dict[str, Any]] = []
    fresh_metrics = (
        "fresh_a_mean",
        "fresh_atomic_gradient_rms",
        "fresh_hvp_rms",
        "conditional_probe_variance_W",
        "conditional_mean_gradient_norm",
    )
    for task, group in sentinel_metrics.groupby("task_name", sort=True):
        highest = int(group["prior_grad_rms"].idxmax())
        for metric in fresh_metrics:
            sentinel_rows.append(
                {
                    "task_name": task,
                    "fresh_metric": metric,
                    "scope": "all_selected_sentinels",
                    "state_count": len(group),
                    "spearman_to_prior_grad_rms": float(group["prior_grad_rms"].corr(group[metric], method="spearman")),
                }
            )
            leave = group.drop(index=highest)
            sentinel_rows.append(
                {
                    "task_name": task,
                    "fresh_metric": metric,
                    "scope": "leave_highest_prior_state_out",
                    "state_count": len(leave),
                    "spearman_to_prior_grad_rms": float(leave["prior_grad_rms"].corr(leave[metric], method="spearman")),
                }
            )
    pd.DataFrame(sentinel_rows).to_csv(output_dir / "sentinel_persistence.csv", index=False)
    delta_rows: list[dict[str, Any]] = []
    for direction, transform in (("raw", lambda value: value), ("unit", unit_gram)):
        task_delta: list[DeltaEstimate] = []
        for task in ("fashion_mnist", "mnist"):
            gram = transform(task_grams[task])
            estimate = delta_from_crossed_gram(gram, [task] * 16, q=DRAW_COUNT)
            task_delta.append(estimate)
            delta_rows.append(
                {
                    "scope": task,
                    "vector_kind": direction,
                    "V_probe": estimate.v_probe,
                    "V_state": estimate.v_state,
                    "Delta": estimate.delta,
                    "state_x_probe_ms": estimate.state_interaction_ms[0],
                }
            )
        delta_rows.append(
            {
                "scope": "primary_mean_tasks",
                "vector_kind": direction,
                "V_probe": float(np.mean([value.v_probe for value in task_delta])),
                "V_state": float(np.mean([value.v_state for value in task_delta])),
                "Delta": float(np.mean([value.delta for value in task_delta])),
                "state_x_probe_ms": float(np.mean([value.state_interaction_ms[0] for value in task_delta])),
            }
        )
    anova_frames: list[pd.DataFrame] = []
    for vector_kind, grams in (
        ("raw", task_grams),
        ("unit", {task: unit_gram(gram) for task, gram in task_grams.items()}),
    ):
        anova = balanced_crossed_anova_from_task_grams(
            grams,
            cross_task_sum_grams[vector_kind],
            q=DRAW_COUNT,
        )
        anova.insert(0, "vector_kind", vector_kind)
        anova_frames.append(anova)
    anova_frame = pd.concat(anova_frames, ignore_index=True)
    anova_frame.to_csv(
        output_dir / "crossed_anova_sufficient.csv",
        index=False,
    )
    for row in delta_rows:
        if row["scope"] != "primary_mean_tasks":
            continue
        vector_kind = str(row["vector_kind"])
        task_ss = float(
            anova_frame.loc[
                (anova_frame["vector_kind"] == vector_kind)
                & (anova_frame["component"] == "task"),
                "sum_squared_vector_norms",
            ].iloc[0]
        )
        task_probe_ms = float(
            anova_frame.loc[
                (anova_frame["vector_kind"] == vector_kind)
                & (anova_frame["component"] == "task*probe"),
                "mean_square_trace",
            ].iloc[0]
        )
        panel_energy = (task_ss - task_probe_ms) / (
            DRAW_COUNT * (PRIMARY_COUNT - 1)
        )
        state_within_energy = 30.0 * float(row["V_state"]) / 31.0
        row["E_task_panel"] = panel_energy
        row["E_state_within"] = state_within_energy
        row["V_state_total"] = panel_energy + state_within_energy
        row["Delta_total"] = float(row["V_probe"]) - float(row["V_state_total"])
    pd.DataFrame(delta_rows).to_csv(output_dir / "delta_components.csv", index=False)

    primary_metrics = panel_metrics.loc[panel_metrics["panel"] == "primary"].copy()
    def sensitivity_components(removed: int | None, vector_kind: str) -> DeltaEstimate:
        estimates: list[DeltaEstimate] = []
        for task, offset in (("fashion_mnist", 0), ("mnist", 16)):
            states = [state for state in range(16) if removed is None or state + offset != removed]
            cells = np.arange(16 * DRAW_COUNT).reshape(16, DRAW_COUNT)[states].ravel()
            gram = task_grams[task][np.ix_(cells, cells)]
            if vector_kind == "unit":
                gram = unit_gram(gram)
            estimates.append(delta_from_crossed_gram(gram, [task] * len(states), q=DRAW_COUNT))
        return DeltaEstimate(
            v_probe=float(np.mean([value.v_probe for value in estimates])),
            v_state=float(np.mean([value.v_state for value in estimates])),
            delta=float(np.mean([value.delta for value in estimates])),
            state_interaction_ms=tuple(
                value for estimate in estimates for value in estimate.state_interaction_ms
            ),
        )

    sensitivity_rows: list[dict[str, Any]] = []
    for vector_kind in ("raw", "unit"):
        estimate = sensitivity_components(None, vector_kind)
        sensitivity_rows.append(
            {
                "sensitivity": "baseline",
                "vector_kind": vector_kind,
                "removed_state_position": -1,
                "V_probe": estimate.v_probe,
                "V_state": estimate.v_state,
                "Delta": estimate.delta,
            }
        )
    largest_state = int(primary_metrics["conditional_probe_variance_W"].idxmax())
    removed_global = int(primary_metrics.loc[largest_state, "state_position"])
    for vector_kind in ("raw", "unit"):
        estimate = sensitivity_components(removed_global, vector_kind)
        sensitivity_rows.append(
            {
                "sensitivity": "leave_largest_primary_W_state_out",
                "vector_kind": vector_kind,
                "removed_state_position": removed_global,
                "V_probe": estimate.v_probe,
                "V_state": estimate.v_state,
                "Delta": estimate.delta,
            }
        )
    for task, group in primary_metrics.groupby("task_name", sort=True):
        largest = int(group["conditional_probe_variance_W"].idxmax())
        removed_task = int(group.loc[largest, "state_position"])
        for vector_kind in ("raw", "unit"):
            estimate = sensitivity_components(removed_task, vector_kind)
            sensitivity_rows.append(
                {
                    "sensitivity": "leave_largest_task_W_state_out",
                    "vector_kind": vector_kind,
                    "task_name": task,
                    "removed_state_position": removed_task,
                    "V_probe": estimate.v_probe,
                    "V_state": estimate.v_state,
                    "Delta": estimate.delta,
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "sensitivity_summary.csv", index=False)

    primary_p4 = state_summary.loc[(state_summary["panel"] == "primary") & (state_summary["prefix"] == 4)]
    passing = int(primary_p4["split_gate_pass"].sum())
    primary_bootstrap = fixed_panel_probe_bootstrap(
        task_grams,
        cross_task_sum_grams,
        draws=int(args.crossed_bootstrap_draws),
        seed=stable_uint63(PROTOCOL_ID, BASE_SEED, "primary_fixed_panel_probe_bootstrap"),
    )
    primary_bootstrap.to_csv(output_dir / "primary_fixed_panel_probe_bootstrap.csv", index=False)
    panel_bootstrap = panel_sensitivity_bootstrap(
        task_grams,
        draws=int(args.crossed_bootstrap_draws),
        seed=stable_uint63(PROTOCOL_ID, BASE_SEED, "panel_sensitivity_bootstrap"),
    )
    panel_bootstrap.to_csv(output_dir / "panel_sensitivity_bootstrap.csv", index=False)
    direction_boot = primary_bootstrap.loc[
        primary_bootstrap["vector_kind"] == "unit", "Delta_total"
    ]
    delta_low, delta_high = np.quantile(direction_boot, [0.025, 0.975])
    cosine_boot = primary_bootstrap.loc[
        primary_bootstrap["vector_kind"] == "unit",
        "state_median_pairwise_cosine_median",
    ]
    median_high = float(np.quantile(cosine_boot, 0.95))
    heterogeneity_boot = primary_bootstrap.loc[
        primary_bootstrap["vector_kind"] == "unit",
        "fashion_minus_mnist_state_x_probe_ms",
    ]
    heterogeneity_low, heterogeneity_high = np.quantile(heterogeneity_boot, [0.025, 0.975])
    decision = primary_mechanism_decision(
        delta_dir_ci_low=float(delta_low),
        delta_dir_ci_high=float(delta_high),
        within_state_cosine_median_ci_high=median_high,
        state_nonalignment_gate_passed=(
            within_state_nonalignment_gate or task_nonalignment_gate
        ),
    )
    point_unit = pd.DataFrame(delta_rows).loc[
        lambda frame: (frame["scope"] == "primary_mean_tasks")
        & (frame["vector_kind"] == "unit")
    ].iloc[0]
    if decision != "state-direction-major":
        state_attribution = "state_level_attribution_not_activated"
    elif within_state_nonalignment_gate and task_nonalignment_gate:
        state_attribution = "mixed_state_level"
    elif task_nonalignment_gate:
        state_attribution = "fixed_task_panel_contrast_leading"
    elif within_state_nonalignment_gate:
        state_attribution = "HS_within_panel_state_leading"
    else:
        state_attribution = "state_level_nonalignment_not_established"
    decision_payload = {
        "decision": decision,
        "delta_total_dir_ci95_low": float(delta_low),
        "delta_total_dir_ci95_high": float(delta_high),
        "delta_total_dir_point": float(point_unit["Delta_total"]),
        "delta_within_dir_point": float(point_unit["Delta"]),
        "E_task_panel_dir_point": float(point_unit["E_task_panel"]),
        "E_state_within_dir_point": float(point_unit["E_state_within"]),
        "state_level_median_within_state_p4_cosine_ci95_high": median_high,
        "conditional_mean_gate_passing_states": passing,
        "conditional_mean_gate_excluded_states": PRIMARY_COUNT - passing,
        "within_task_state_nonalignment_gate": within_state_nonalignment_gate,
        "fixed_task_panel_nonalignment_gate": task_nonalignment_gate,
        "state_level_attribution": state_attribution,
        "ci_source": "primary_fixed_panel_probe_bootstrap.csv",
        "panel_sensitivity_source": "panel_sensitivity_bootstrap.csv",
        "fashion_minus_mnist_state_x_probe_ms_ci95_low": float(heterogeneity_low),
        "fashion_minus_mnist_state_x_probe_ms_ci95_high": float(heterogeneity_high),
        "HI_task_localized_interaction_supported": bool(
            heterogeneity_low > 0.0 or heterogeneity_high < 0.0
        ),
        "H0_primary_state_category_fractions": (
            pd.DataFrame(signal_rows)
            .loc[lambda frame: frame["panel"] == "primary", "category"]
            .value_counts(normalize=True)
            .to_dict()
        ),
        "primary_state_median_within_state_p4_cosine": float(
            primary_p4["pairwise_cosine_median"].median()
        ),
        "interpretation_boundary": (
            "bank-conditional raw Variant A estimator diagnostics only; no downstream, "
            "training-update, finite-B population, sentinel-prevalence, or broad Li/PSGD/Kron claim"
        ),
    }
    atomic_write_json(output_dir / "decision.json", decision_payload)

    expected_primary_atomic = STATE_COUNT * DRAW_COUNT * PAIR_COUNT
    seed_count = len({common_probe_seed(draw, pair, branch) for draw in range(DRAW_COUNT) for pair in range(PAIR_COUNT) for branch in (0, 1)})
    checks = {
        "source_hashes_match": source_hashes == EXPECTED_SHA256,
        "state_bank_gate_passed": bank_gate["state_count"] == STATE_COUNT,
        "model_hash_before_match": model_hash_before == EXPECTED_MODEL_STATE_SHA256,
        "model_hash_after_unchanged": model_state_sha256(run.vae) == model_hash_before,
        "latent_hash_count": len(latent_hashes) == STATE_COUNT,
        "active_manifest_match": active_count == EXPECTED_ACTIVE_COUNT,
        "atomic_cardinality": len(atomic) == expected_primary_atomic,
        "atomic_keys_unique": not atomic.duplicated(["state_position", "draw", "pair"]).any(),
        "all_full_ce": bool((atomic[["batch_1_hash", "batch_2_hash"]] == "full").all().all()),
        "all_train_count_16384": bool((atomic["batch_train_count"] == TRAIN_COUNT).all()),
        "bridge_cardinality_and_batch_replay": bridge_gate["row_count"] == 512,
        "bridge_paired_key_cardinality": len(bridge_paired_rows) == len(BRIDGE_POSITIONS) * 16,
        "probe_key_cardinality": seed_count == DRAW_COUNT * PAIR_COUNT * 2,
        "probe_hash_common_across_states": len(probe_hashes) == DRAW_COUNT * PAIR_COUNT * 2,
        "probe_hashes_unique_across_keys": len(set(probe_hashes.values())) == DRAW_COUNT * PAIR_COUNT * 2,
        "all_numeric_metadata_finite": bool(np.isfinite(atomic[["a_scalar", "atomic_gradient_norm", "h1_norm", "h2_norm"]].to_numpy(dtype=float)).all()),
        "all_grams_finite": all(
            np.isfinite(value).all()
            for value in [
                *prefix_grams.values(),
                *module_grams.values(),
                *task_grams.values(),
                *cross_pair_grams.values(),
                *cross_task_sum_grams.values(),
                *bridge_grams.values(),
            ]
        ),
        "nonalignment_gates_finite": bool(
            np.isfinite(within_split_medians.to_numpy(dtype=float)).all()
            and np.isfinite(
                task_gate_frame[
                    [
                        "fashion_self_stability",
                        "mnist_self_stability",
                        "fashion_mnist_crossfit_cosine",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "p4_materialized_preflight": p4_preflight["gradient_cosine"] >= 0.99999,
        "no_model_update": True,
    }
    validity = {"passed": bool(all(checks.values())), "checks": checks}
    atomic_write_json(output_dir / "validity.json", validity)
    if not validity["passed"]:
        raise RuntimeError(f"terminal validity failure: {json.dumps(validity, sort_keys=True)}")
    for path in list(p4_paths.values()):
        if path.exists():
            path.unlink()
    _cleanup_owned_scratch(scratch)
    return {
        **lifecycle,
        "status": "complete",
        "completed_unix": time.time(),
        "elapsed_sec": time.perf_counter() - started,
        "validity": validity,
        "decision": decision_payload,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }


def _resampled_components_from_task_gram(
    gram: np.ndarray,
    states: np.ndarray,
    probes: np.ndarray,
) -> DeltaEstimate:
    q = len(probes)
    n = len(states)
    cells = np.arange(16 * DRAW_COUNT).reshape(16, DRAW_COUNT)
    state_counts = np.bincount(states, minlength=16).astype(np.float64)
    probe_counts = np.bincount(probes, minlength=DRAW_COUNT).astype(np.float64)
    state_norms = np.zeros(16, dtype=np.float64)
    state_within = np.zeros(16, dtype=np.float64)
    for state in np.flatnonzero(state_counts):
        block = gram[np.ix_(cells[state], cells[state])]
        summed_norm2 = float(probe_counts @ block @ probe_counts)
        state_norms[state] = summed_norm2 / float(q**2)
        sampled_norm_sum = float(probe_counts @ np.diag(block))
        state_within[state] = (sampled_norm_sum - summed_norm2 / q) / (q - 1)
    v_probe = float(state_counts @ state_within / n)
    cell_counts = np.outer(state_counts, probe_counts).ravel()
    task_mean = float(cell_counts @ gram @ cell_counts) / float((n * q) ** 2)
    state_norm_sum = float(state_counts @ state_norms)
    observed = (state_norm_sum - n * task_mean) / (n - 1)
    total = float(cell_counts @ np.diag(gram))
    probe_energy = 0.0
    for probe in np.flatnonzero(probe_counts):
        probe_cells = cells[:, probe]
        probe_norm2 = float(state_counts @ gram[np.ix_(probe_cells, probe_cells)] @ state_counts)
        probe_energy += float(probe_counts[probe]) * probe_norm2 / float(n**2)
    residual_ss = total - q * state_norm_sum - n * probe_energy + n * q * task_mean
    interaction_ms = residual_ss / ((n - 1) * (q - 1))
    v_state = observed - interaction_ms / q
    return DeltaEstimate(v_probe, v_state, v_probe - v_state, (interaction_ms,))


def aggregate_task_probe_gram(task_gram: np.ndarray, *, q: int = DRAW_COUNT) -> np.ndarray:
    gram = np.asarray(task_gram, dtype=np.float64)
    if gram.shape[0] != gram.shape[1] or gram.shape[0] % q != 0:
        raise ValueError("task Gram shape is incompatible with Q")
    n = gram.shape[0] // q
    cells = np.arange(n * q).reshape(n, q)
    return np.asarray(
        [
            [float(gram[np.ix_(cells[:, left], cells[:, right])].sum()) for right in range(q)]
            for left in range(q)
        ],
        dtype=np.float64,
    )


def corrected_panel_contrast_energy_from_task_grams(
    task_grams: Mapping[str, np.ndarray],
    cross_task_sum_gram: np.ndarray,
    probes: Sequence[int],
) -> float:
    """Unbiased fixed-panel contribution to variance across conditional state means."""
    tasks = ("fashion_mnist", "mnist")
    if set(task_grams) != set(tasks):
        raise ValueError("fixed task energy requires exact primary task Grams")
    selected_probes = np.asarray(probes, dtype=np.int64)
    q = len(selected_probes)
    if q < 2:
        raise ValueError("fixed task energy requires at least two probes")
    n = np.asarray(task_grams[tasks[0]]).shape[0] // DRAW_COUNT
    cross = np.asarray(cross_task_sum_gram, dtype=np.float64)
    if cross.shape != (DRAW_COUNT, DRAW_COUNT):
        raise ValueError("cross-task probe-sum Gram must be QxQ")
    task_probe = {
        task: aggregate_task_probe_gram(np.asarray(task_grams[task]))
        for task in tasks
    }
    difference_gram = (
        task_probe[tasks[0]] + task_probe[tasks[1]] - cross - cross.T
    ) / float(n**2)
    selected = difference_gram[np.ix_(selected_probes, selected_probes)]
    expected_squared_difference = (
        float(selected.sum()) - float(np.trace(selected))
    ) / float(q * (q - 1))
    state_count = 2 * n
    return float(
        n * expected_squared_difference / (2.0 * (state_count - 1))
    )


def fixed_panel_probe_bootstrap(
    task_grams: Mapping[str, np.ndarray],
    cross_task_sum_grams: Mapping[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Primary bank-conditional bootstrap: fixed states and one common Q resample."""
    if set(task_grams) != {"fashion_mnist", "mnist"}:
        raise ValueError("fixed-panel bootstrap requires exact primary task Grams")
    if set(cross_task_sum_grams) != {"raw", "unit"}:
        raise ValueError("fixed-panel bootstrap requires raw and unit cross-task sums")
    generator = np.random.default_rng(int(seed))
    unit = {task: unit_gram(gram) for task, gram in task_grams.items()}
    cells = np.arange(16 * DRAW_COUNT).reshape(16, DRAW_COUNT)
    rows: list[dict[str, Any]] = []
    fixed_states = np.arange(16, dtype=np.int64)
    for bootstrap in range(int(draws)):
        common_probes = generator.integers(0, DRAW_COUNT, size=DRAW_COUNT)
        state_medians: list[float] = []
        triangle = np.triu_indices(DRAW_COUNT, k=1)
        for task in ("fashion_mnist", "mnist"):
            task_unit = unit[task]
            for state in range(16):
                selected = cells[state, common_probes]
                state_medians.append(float(np.median(task_unit[np.ix_(selected, selected)][triangle])))
        cosine_statistic = float(np.median(state_medians))
        for vector_kind, grams in (("raw", task_grams), ("unit", unit)):
            task_estimates = [
                _resampled_components_from_task_gram(grams[task], fixed_states, common_probes)
                for task in ("fashion_mnist", "mnist")
            ]
            v_probe = float(np.mean([value.v_probe for value in task_estimates]))
            v_state = float(np.mean([value.v_state for value in task_estimates]))
            panel_energy = corrected_panel_contrast_energy_from_task_grams(
                grams,
                cross_task_sum_grams[vector_kind],
                common_probes,
            )
            state_within_energy = 30.0 * v_state / 31.0
            state_total = panel_energy + state_within_energy
            rows.append(
                {
                    "bootstrap": bootstrap,
                    "bootstrap_kind": "primary_fixed_state_common_probe",
                    "vector_kind": vector_kind,
                    "Delta": float(np.mean([value.delta for value in task_estimates])),
                    "V_probe": v_probe,
                    "V_state": v_state,
                    "E_task_panel": panel_energy,
                    "E_state_within": state_within_energy,
                    "V_state_total": state_total,
                    "Delta_total": v_probe - state_total,
                    "fashion_delta": task_estimates[0].delta,
                    "mnist_delta": task_estimates[1].delta,
                    "fashion_minus_mnist_delta": task_estimates[0].delta - task_estimates[1].delta,
                    "fashion_minus_mnist_V_probe": task_estimates[0].v_probe - task_estimates[1].v_probe,
                    "fashion_state_x_probe_ms": task_estimates[0].state_interaction_ms[0],
                    "mnist_state_x_probe_ms": task_estimates[1].state_interaction_ms[0],
                    "fashion_minus_mnist_state_x_probe_ms": (
                        task_estimates[0].state_interaction_ms[0]
                        - task_estimates[1].state_interaction_ms[0]
                    ),
                    "state_median_pairwise_cosine_median": cosine_statistic,
                }
            )
    return pd.DataFrame(rows)


def panel_sensitivity_bootstrap(
    task_grams: Mapping[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    if set(task_grams) != {"fashion_mnist", "mnist"}:
        raise ValueError("crossed bootstrap requires exact primary task Grams")
    generator = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    unit = {task: unit_gram(gram) for task, gram in task_grams.items()}
    for bootstrap in range(int(draws)):
        common_probes = generator.integers(0, DRAW_COUNT, size=DRAW_COUNT)
        # Local task positions are frozen in four contiguous four-state step strata.
        # Resampling each stratum separately preserves the task x step_stratum balance.
        state_samples = {
            task: np.concatenate(
                [generator.integers(4 * stratum, 4 * (stratum + 1), size=4) for stratum in range(4)]
            )
            for task in task_grams
        }
        for vector_kind, grams in (("raw", task_grams), ("unit", unit)):
            task_estimates = [
                _resampled_components_from_task_gram(grams[task], state_samples[task], common_probes)
                for task in ("fashion_mnist", "mnist")
            ]
            rows.append(
                {
                    "bootstrap": bootstrap,
                    "bootstrap_kind": "panel_sensitivity_stratified_state_common_probe",
                    "vector_kind": vector_kind,
                    "Delta": float(np.mean([value.delta for value in task_estimates])),
                    "V_probe": float(np.mean([value.v_probe for value in task_estimates])),
                    "V_state": float(np.mean([value.v_state for value in task_estimates])),
                    "fashion_delta": task_estimates[0].delta,
                    "mnist_delta": task_estimates[1].delta,
                    "fashion_minus_mnist_delta": task_estimates[0].delta - task_estimates[1].delta,
                    "fashion_minus_mnist_V_probe": task_estimates[0].v_probe - task_estimates[1].v_probe,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen fixed-state crossed-probe Variant A audit.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--gram-chunk-size", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--crossed-bootstrap-draws", type=int, default=2000)
    parser.add_argument("--signal-bootstrap-draws", type=int, default=2000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        manifest = run_audit(args)
        atomic_write_json(output_dir / "manifest.json", manifest)
        print(
            f"[fixed_state_probe] complete elapsed_sec={manifest['elapsed_sec']:.1f} "
            f"manifest={output_dir / 'manifest.json'}",
            flush=True,
        )
    except BaseException as error:
        scratch = _scratch_path(args.scratch_dir)
        cleanup_error = ""
        marker = scratch / "scratch_owner.json"
        if marker.exists():
            try:
                _cleanup_owned_scratch(scratch)
            except BaseException as scratch_error:
                cleanup_error = f"{type(scratch_error).__name__}: {scratch_error}"
        failure = {
            "status": "failure",
            "protocol_id": PROTOCOL_ID,
            "failed_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "scratch_dir": str(scratch),
            "scratch_cleanup_error": cleanup_error,
        }
        atomic_write_json(output_dir / "manifest.json", failure)
        print(f"[fixed_state_probe] failure={json.dumps(failure, sort_keys=True)}", flush=True)
        raise


if __name__ == "__main__":
    main()
