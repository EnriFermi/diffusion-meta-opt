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

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    encode_weights,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _latent_hvp,
    _task_set_for_record,
)
from scripts import audit_one_state_exact_a_proposal3_cross as exact_helpers
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _add_,
    _cosine,
    _dot,
    _named_tensor_hash,
    _negative_normalized,
    _norm,
    _vector_is_finite,
    _zeros_like,
)
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _load_run,
    _probe_cfg,
    sha256_file,
    sha256_tensor,
)
from scripts.run_one_state_a_full_burg_armijo import _set_parameters
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    STATE_BANK,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "one_state_exact_a_low_common_i5_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_5_exact_a_low_common/protocol.md"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration5_exact_a_low_common_production"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_5_frozen_dependency_manifest.json"
EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = "d088533055704dad8d3cd0ec791a709ecb8676ec3d1ad7f9aed8c3949db51cca"
EXPECTED_NORMALIZED_SOURCE_SHA256 = "8355b545f387e44f69d060b917077d538e51cddd2b643dcdf1770912bfecdd2b"

ITERATION3 = OUTPUT_ROOT / "iteration3_p32_unit_common_production"
ITERATION3_CHECKPOINT = ITERATION3 / "final_checkpoint.pt"
ITERATION3_STATES = ITERATION3 / "state_metrics.csv"
ITERATION3_SPECTRA = ITERATION3 / "state_spectra.csv"
EXPECTED_ITERATION3_CHECKPOINT_SHA256 = (
    "ccafbad0e0fcf25aaabffe0a406558937d1b448ac06c80b7c40a6637209dc5ad"
)
EXPECTED_FINAL_PARAMETER_SHA256 = (
    "31abec6439564c392a99b6c82dd1386aed714cb1b2a1ce1c4709088686a0caff"
)
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"

SOURCE_WEIGHT_INDEX = 378
STATE_POSITION = 2
EPSILON = 1e-4
HESSIAN_CHUNK_SIZE = 64
BLOCK_SIZE = 64
LOW_THRESHOLD = 0.1
DEAD_THRESHOLD = 1e-4
EXPECTED_LOW_COUNT = 480
EXPECTED_DEAD_COUNT = 309
TARGET_NORM = 0.04892722657548397
LOCAL_RADII = (1.0 / 1024.0, 1.0 / 512.0)
ALPHAS = (
    -1.0 / 128.0,
    -1.0 / 256.0,
    -1.0 / 512.0,
    -1.0 / 1024.0,
    0.0,
    1.0 / 1024.0,
    1.0 / 512.0,
    1.0 / 256.0,
    1.0 / 128.0,
    1.0 / 32.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
)
CANCELLATION_NORM_MIN = 0.5
ORTHOGONALITY_MAX = 1e-10
EIGEN_RESIDUAL_MAX = 1e-10
EIGEN_ORDER_TOL = 1e-12
CHAIN_RELATIVE_ERROR_MAX = 0.05
CHAIN_CAUCHY_RELATIVE_ERROR_MAX = 0.01
EFFECTIVE_COSINE_MIN = 0.99
EFFECTIVE_NORM_RATIO_MIN = 0.95
EFFECTIVE_NORM_RATIO_MAX = 1.05
MIDPOINT_RELATIVE_MAX = 0.05
RETAINED_MEMORY_RANGE_MAX = 64 * 1024 * 1024
BLOCK_PEAK_RANGE_MAX = 128 * 1024 * 1024
METRIC_FLOORS = {
    "exact_a_per_dim": 1e-9,
    "damped_full_burg_per_dim": 1e-9,
    "frozen_low_energy": 1e-10,
    "m_max": 1e-8,
    "m_p50": 1e-10,
    "effective_rank": 1e-8,
    "projected_low_p50": 1e-12,
    "projected_low_participation_rank": 1e-8,
    "projected_low_entropy_rank": 1e-8,
}

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
    masked = (
        "EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    normalized: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        prefix = next((candidate for candidate in masked if line.startswith(candidate)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _validate_dependencies() -> dict[str, bool]:
    if "TO_BE_FROZEN" in (
        EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        EXPECTED_NORMALIZED_SOURCE_SHA256,
    ):
        raise RuntimeError("runner has not been frozen")
    if _normalized_source_sha256() != EXPECTED_NORMALIZED_SOURCE_SHA256:
        raise RuntimeError("normalized source hash mismatch")
    if sha256_file(FROZEN_DEPENDENCY_MANIFEST) != EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256:
        raise RuntimeError("dependency manifest hash mismatch")
    expected = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    matches = {
        relative: (ROOT / relative).is_file() and sha256_file(ROOT / relative) == digest
        for relative, digest in expected.items()
    }
    failed = [relative for relative, ok in matches.items() if not ok]
    if failed:
        raise RuntimeError(f"frozen dependency mismatch: {failed}")
    return matches


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


def _numeric_finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def _blocked_two_gradients(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    k_a: torch.Tensor,
    k_low: torch.Tensor,
) -> tuple[Vector, Vector, dict[str, Any]]:
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    dim = int(z.numel())
    accumulated_a = _zeros_like(active, dtype=torch.float64)
    accumulated_low = _zeros_like(active, dtype=torch.float64)
    unused_a = np.zeros(len(active), dtype=np.int64)
    unused_low = np.zeros(len(active), dtype=np.int64)
    memory_trace: list[int] = []
    peak_memory_trace: list[int] = []
    started = time.perf_counter()
    block_count = math.ceil(dim / BLOCK_SIZE)
    for start in range(0, dim, BLOCK_SIZE):
        torch.cuda.reset_peak_memory_stats(z.device)
        stop = min(start + BLOCK_SIZE, dim)
        scalars_a: list[torch.Tensor] = []
        scalars_low: list[torch.Tensor] = []
        for index in range(start, stop):
            basis = torch.zeros_like(z)
            basis[index] = 1.0
            hvp = _latent_hvp(
                cfg,
                run.vae,
                run.normalizer,
                z,
                basis,
                task_set=task_set,
                spec=run.spec,
                tau=tau,
                batch_indices=None,
            )
            scalars_a.append(torch.dot(k_a[index].to(dtype=hvp.dtype), hvp))
            scalars_low.append(torch.dot(k_low[index].to(dtype=hvp.dtype), hvp))
        total_a = torch.stack(scalars_a).sum()
        total_low = torch.stack(scalars_low).sum()
        gradients_a = torch.autograd.grad(
            total_a, active, retain_graph=True, allow_unused=True
        )
        gradients_low = torch.autograd.grad(
            total_low, active, retain_graph=False, allow_unused=True
        )
        for index, value in enumerate(gradients_a):
            if value is None:
                unused_a[index] += 1
        for index, value in enumerate(gradients_low):
            if value is None:
                unused_low[index] += 1
        _add_(accumulated_a, gradients_a)
        _add_(accumulated_low, gradients_low)
        memory_trace.append(int(torch.cuda.memory_allocated(z.device)))
        peak_memory_trace.append(int(torch.cuda.max_memory_allocated(z.device)))
        print(
            f"[a-low-i5] stage=blocked-two-vjp rows={stop}/{dim} "
            f"blocks={(stop + BLOCK_SIZE - 1) // BLOCK_SIZE}/{block_count} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del scalars_a, scalars_low, total_a, total_low, gradients_a, gradients_low
    edge = min(2, len(memory_trace))
    early = float(np.median(memory_trace[:edge]))
    late = float(np.median(memory_trace[-edge:]))
    retained_range = float(max(memory_trace) - min(memory_trace))
    peak_range = float(max(peak_memory_trace) - min(peak_memory_trace))
    return accumulated_a, accumulated_low, {
        "basis_count": dim,
        "block_size": BLOCK_SIZE,
        "block_count": block_count,
        "a_unused_parameter_tensors_all_blocks": int((unused_a == block_count).sum()),
        "low_unused_parameter_tensors_all_blocks": int((unused_low == block_count).sum()),
        "accumulation_dtype": "float64",
        "all_a_accumulators_float64": bool(
            all(value.dtype == torch.float64 for value in accumulated_a)
        ),
        "all_low_accumulators_float64": bool(
            all(value.dtype == torch.float64 for value in accumulated_low)
        ),
        "memory_early_median_bytes": early,
        "memory_late_median_bytes": late,
        "memory_growth_bytes": max(0.0, late - early),
        "retained_memory_by_block_bytes": memory_trace,
        "peak_memory_by_block_bytes": peak_memory_trace,
        "retained_memory_range_bytes": retained_range,
        "block_peak_min_bytes": float(min(peak_memory_trace)),
        "block_peak_max_bytes": float(max(peak_memory_trace)),
        "block_peak_range_bytes": peak_range,
        "memory_growth_gate_pass": bool(
            late - early <= RETAINED_MEMORY_RANGE_MAX
            and retained_range <= RETAINED_MEMORY_RANGE_MAX
            and peak_range <= BLOCK_PEAK_RANGE_MAX
        ),
        "elapsed_sec": time.perf_counter() - started,
    }


def _projected_metrics(
    matrix: torch.Tensor, low_basis: torch.Tensor
) -> tuple[dict[str, float], torch.Tensor]:
    projected = low_basis.transpose(0, 1) @ matrix.double() @ low_basis
    projected = 0.5 * (projected + projected.transpose(0, 1))
    raw = torch.linalg.eigvalsh(projected)
    eig = raw.clamp_min(0.0)
    trace = eig.sum()
    squared = eig.square().sum()
    participation = trace.square() / squared.clamp_min(1e-30)
    probabilities = eig / trace.clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
    quantiles = torch.quantile(
        eig,
        torch.tensor(
            [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
            device=eig.device,
            dtype=eig.dtype,
        ),
    )
    return {
        "projected_low_raw_min": float(raw.min().cpu()),
        "projected_low_mean": float(eig.mean().cpu()),
        "projected_low_max": float(eig.max().cpu()),
        "projected_low_p01": float(quantiles[0].cpu()),
        "projected_low_p10": float(quantiles[1].cpu()),
        "projected_low_p25": float(quantiles[2].cpu()),
        "projected_low_p50": float(quantiles[3].cpu()),
        "projected_low_p75": float(quantiles[4].cpu()),
        "projected_low_p90": float(quantiles[5].cpu()),
        "projected_low_p95": float(quantiles[6].cpu()),
        "projected_low_p99": float(quantiles[7].cpu()),
        "projected_low_participation_rank": float(participation.cpu()),
        "projected_low_entropy_rank": float(entropy.exp().cpu()),
    }, eig.detach().clone()


def _endpoint(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    low_basis: torch.Tensor,
    dead_basis: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    metrics, hessian, matrix, eig, _burg = exact_helpers._evaluate_dense(
        run=run,
        z=z,
        record=record,
        epsilon=EPSILON,
        hessian_chunk_size=HESSIAN_CHUNK_SIZE,
    )
    low_projected, projected_eig = _projected_metrics(matrix, low_basis)
    metrics.update(low_projected)
    metrics.update(
        {
            "frozen_low_energy": float(
                torch.trace(low_basis.transpose(0, 1) @ matrix.double() @ low_basis).cpu()
                / float(low_basis.shape[1])
            ),
            "frozen_dead_energy": float(
                torch.trace(dead_basis.transpose(0, 1) @ matrix.double() @ dead_basis).cpu()
                / float(dead_basis.shape[1])
            ),
            "count_lt_1e_4": float((eig < DEAD_THRESHOLD).sum().cpu()),
            "count_lt_1e_2": float((eig < 1e-2).sum().cpu()),
            "count_lt_0p1": float((eig < LOW_THRESHOLD).sum().cpu()),
        }
    )
    del matrix, _burg
    return metrics, hessian.detach().double().clone(), eig.detach().double().clone(), projected_eig


def _repeat_errors(
    primary: Mapping[str, float],
    repeat: Mapping[str, float],
    eig: torch.Tensor,
    repeat_eig: torch.Tensor,
    projected_eig: torch.Tensor,
    repeat_projected_eig: torch.Tensor,
) -> dict[str, float]:
    discrete = {"count_lt_1e_4", "count_lt_1e_2", "count_lt_0p1"}
    continuous = sorted((set(primary) & set(repeat)) - discrete - {"hessian_sec"})
    result = {
        f"repeat_{key}_abs_error": abs(float(primary[key]) - float(repeat[key]))
        for key in continuous
    }
    for key in sorted(discrete):
        result[f"repeat_{key}_abs_error"] = abs(float(primary[key]) - float(repeat[key]))
    result["repeat_global_spectrum_max_abs_error"] = float(
        (eig - repeat_eig).abs().max().cpu()
    )
    result["repeat_projected_spectrum_max_abs_error"] = float(
        (projected_eig - repeat_projected_eig).abs().max().cpu()
    )
    return result


def _realized_parameter_stats(
    *,
    plus: Sequence[torch.Tensor],
    minus: Sequence[torch.Tensor],
    base: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    gradient_a: Sequence[torch.Tensor],
    gradient_low: Sequence[torch.Tensor],
    radius: float,
) -> dict[str, float]:
    eff_norm2 = 0.0
    direction_norm2 = 0.0
    eff_direction_dot = 0.0
    midpoint_norm2 = 0.0
    s_theta_a = 0.0
    s_theta_low = 0.0
    for plus_value, minus_value, base_value, d_value, g_a, g_low in zip(
        plus, minus, base, direction, gradient_a, gradient_low, strict=True
    ):
        device = g_a.device
        effective = (
            plus_value.to(device=device, dtype=torch.float64)
            - minus_value.to(device=device, dtype=torch.float64)
        ) / (2.0 * radius)
        midpoint = 0.5 * (
            plus_value.to(device=device, dtype=torch.float64)
            + minus_value.to(device=device, dtype=torch.float64)
        ) - base_value.detach().double()
        desired = d_value.detach().double()
        eff_norm2 += float(effective.square().sum().cpu())
        direction_norm2 += float(desired.square().sum().cpu())
        eff_direction_dot += float((effective * desired).sum().cpu())
        midpoint_norm2 += float(midpoint.square().sum().cpu())
        s_theta_a += float((g_a.detach().double() * effective).sum().cpu())
        s_theta_low += float((g_low.detach().double() * effective).sum().cpu())
    eff_norm = math.sqrt(max(eff_norm2, 0.0))
    desired_norm = math.sqrt(max(direction_norm2, 0.0))
    return {
        "effective_direction_norm": eff_norm,
        "desired_direction_norm": desired_norm,
        "effective_direction_norm_ratio": eff_norm / max(desired_norm, 1e-30),
        "effective_direction_cosine": eff_direction_dot
        / max(eff_norm * desired_norm, 1e-30),
        "midpoint_drift_norm": math.sqrt(max(midpoint_norm2, 0.0)),
        "midpoint_drift_relative": math.sqrt(max(midpoint_norm2, 0.0))
        / max(radius * desired_norm, 1e-30),
        "s_theta_a": s_theta_a,
        "s_theta_low": s_theta_low,
    }


def _chain_row(
    *,
    radius: float,
    objective: str,
    nominal: float,
    s_theta: float,
    s_h: float,
    direct: float,
    parameter_stats: Mapping[str, float],
) -> dict[str, float | str | bool]:
    errors = {
        "relative_n_to_theta": _relative_error(nominal, s_theta),
        "relative_theta_to_h": _relative_error(s_theta, s_h),
        "relative_h_to_direct": _relative_error(s_h, direct),
        "relative_n_to_direct": _relative_error(nominal, direct),
    }
    expected_signs = all(value < 0.0 for value in (nominal, s_theta, s_h, direct))
    transitions_pass = all(
        errors[key] <= CHAIN_RELATIVE_ERROR_MAX
        for key in ("relative_n_to_theta", "relative_theta_to_h", "relative_h_to_direct")
    )
    realization_pass = bool(
        float(parameter_stats["effective_direction_cosine"]) >= EFFECTIVE_COSINE_MIN
        and EFFECTIVE_NORM_RATIO_MIN
        <= float(parameter_stats["effective_direction_norm_ratio"])
        <= EFFECTIVE_NORM_RATIO_MAX
        and float(parameter_stats["midpoint_drift_relative"]) <= MIDPOINT_RELATIVE_MAX
    )
    return {
        "radius": radius,
        "objective": objective,
        "nominal": nominal,
        "s_theta": s_theta,
        "s_h": s_h,
        "direct": direct,
        **errors,
        **parameter_stats,
        "expected_signs": expected_signs,
        "transitions_pass": transitions_pass,
        "realization_pass": realization_pass,
        "row_pass": bool(expected_signs and transitions_pass and realization_pass),
    }


def _metric_tolerances(endpoints: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric, floor in METRIC_FLOORS.items():
        repeat_column = f"repeat_{metric}_abs_error"
        observed = float(endpoints[repeat_column].max())
        result[metric] = max(5.0 * observed, floor)
    return result


def _candidate_rows(
    endpoints: pd.DataFrame, tolerances: Mapping[str, float]
) -> pd.DataFrame:
    base = endpoints.loc[np.isclose(endpoints["alpha"], 0.0)].iloc[0]
    eligible = endpoints.loc[endpoints["alpha"].ge(1.0 / 8.0)].copy()
    eligible["candidate_pass"] = (
        eligible["exact_a_per_dim"].lt(
            float(base["exact_a_per_dim"]) - tolerances["exact_a_per_dim"]
        )
        & eligible["damped_full_burg_per_dim"].lt(
            float(base["damped_full_burg_per_dim"])
            - tolerances["damped_full_burg_per_dim"]
        )
        & eligible["frozen_low_energy"].gt(
            float(base["frozen_low_energy"]) + tolerances["frozen_low_energy"]
        )
        & eligible["m_max"].le(float(base["m_max"]) + tolerances["m_max"])
        & eligible["m_p50"].gt(float(base["m_p50"]) + tolerances["m_p50"])
        & eligible["projected_low_p50"].gt(
            float(base["projected_low_p50"]) + tolerances["projected_low_p50"]
        )
        & eligible["effective_rank"].ge(
            float(base["effective_rank"]) - tolerances["effective_rank"]
        )
        & eligible["projected_low_participation_rank"].ge(
            float(base["projected_low_participation_rank"])
            - tolerances["projected_low_participation_rank"]
        )
        & eligible["projected_low_entropy_rank"].ge(
            float(base["projected_low_entropy_rank"])
            - tolerances["projected_low_entropy_rank"]
        )
        & eligible["count_lt_1e_4"].le(float(base["count_lt_1e_4"]))
        & eligible["count_lt_1e_2"].le(float(base["count_lt_1e_2"]))
        & eligible["m_raw_eig_min"].ge(-1e-8)
        & eligible["projected_low_raw_min"].ge(-1e-8)
    )
    return eligible


def _plot(
    output: Path,
    endpoints: pd.DataFrame,
    global_spectra: pd.DataFrame,
    projected_spectra: pd.DataFrame,
    selected_alpha: float | None,
) -> None:
    positive = endpoints.loc[endpoints["alpha"].ge(0.0)].sort_values("alpha")
    base = positive.loc[np.isclose(positive["alpha"], 0.0)].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    panels = (
        ("exact_a_per_dim", "Exact dense A"),
        ("damped_full_burg_per_dim", "Damped full Burg B"),
        ("frozen_low_energy", "Frozen low energy"),
        ("m_max", "Maximum M eigenvalue"),
        ("m_p50", "Global median eigenvalue"),
        ("projected_low_p50", "Frozen-low projected median"),
    )
    for axis, (column, title) in zip(axes.flat, panels, strict=True):
        axis.plot(positive["alpha"], positive[column], marker="o", linewidth=2.0)
        axis.axhline(float(base[column]), color="black", linestyle="--", linewidth=1.0)
        if selected_alpha is not None:
            axis.axvline(selected_alpha, color="#2a9d8f", linestyle=":", linewidth=1.5)
        axis.set_xscale("symlog", linthresh=1.0 / 256.0, base=2)
        axis.set(title=title, xlabel="alpha", ylabel=column)
        axis.grid(alpha=0.25)
    fig.suptitle("Iteration 5 exact A + frozen-low common direction")
    fig.savefig(output / "a_low_common_line.png", dpi=180)
    plt.close(fig)

    chosen = [0.0, 1.0 / 8.0, 1.0]
    if selected_alpha is not None and selected_alpha not in chosen:
        chosen.insert(2, selected_alpha)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for alpha in chosen:
        global_rows = global_spectra.loc[np.isclose(global_spectra["alpha"], alpha)]
        projected_rows = projected_spectra.loc[
            np.isclose(projected_spectra["alpha"], alpha)
        ]
        if not global_rows.empty:
            axes[0].plot(
                global_rows["rank"],
                np.clip(global_rows["eigenvalue"], 1e-14, None),
                label=f"a={alpha:g}",
            )
        if not projected_rows.empty:
            axes[1].plot(
                projected_rows["rank"],
                np.clip(projected_rows["eigenvalue"], 1e-14, None),
                label=f"a={alpha:g}",
            )
    for axis, title in zip(
        axes, ("Global M spectrum", "Frozen-low projected spectrum"), strict=True
    ):
        axis.set_yscale("log")
        axis.set(title=title, xlabel="rank", ylabel="eigenvalue")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output / "a_low_common_spectra.png", dpi=180)
    plt.close(fig)


def _manifest(output: Path, source_snapshot: Path) -> dict[str, Any]:
    excluded = {"artifact_manifest.json", "FINALIZED.json", "INCOMPLETE", "run.log"}
    artifacts = sorted(
        path for path in output.iterdir() if path.is_file() and path.name not in excluded
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "executed_source_sha256": _source_sha256(source_snapshot),
        "executed_normalized_source_sha256": _normalized_source_sha256(source_snapshot),
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
    }


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("production audit accepts no CLI overrides")
    dependency_matches = _validate_dependencies()
    final_output = DEFAULT_OUTPUT.resolve()
    staging = Path(str(final_output) + ".incomplete")
    if final_output.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {final_output} or {staging}")
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
    log_handle = (staging / "run.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, log_handle)  # type: ignore[assignment]
    started = time.perf_counter()
    device = torch.device("cuda:0")
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "device": str(device),
        "dtype": "float32 model/HVP; FP64 gradient accumulation and diagnostics",
        "seed": "none; exact full-basis pullback",
        "cache_mode": "accepted h2048 run + serialized Iteration-3 checkpoint",
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "state_position": STATE_POSITION,
        "epsilon": EPSILON,
        "hessian_chunk_size": HESSIAN_CHUNK_SIZE,
        "block_size": BLOCK_SIZE,
        "low_threshold": LOW_THRESHOLD,
        "dead_threshold": DEAD_THRESHOLD,
        "expected_low_count": EXPECTED_LOW_COUNT,
        "expected_dead_count": EXPECTED_DEAD_COUNT,
        "target_norm": TARGET_NORM,
        "local_radii": list(LOCAL_RADII),
        "alphas": list(ALPHAS),
        "cancellation_norm_min": CANCELLATION_NORM_MIN,
        "retained_memory_range_max_bytes": RETAINED_MEMORY_RANGE_MAX,
        "block_peak_range_max_bytes": BLOCK_PEAK_RANGE_MAX,
        "metric_floors": METRIC_FLOORS,
        "output_dir": str(final_output),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_sha256": _source_sha256(source_snapshot),
        "normalized_source_sha256": _normalized_source_sha256(source_snapshot),
        "dependency_manifest_sha256": sha256_file(FROZEN_DEPENDENCY_MANIFEST),
        "dependency_matches": dependency_matches,
    }
    (staging / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[a-low-i5] startup {json.dumps(resolved, sort_keys=True)}", flush=True)

    try:
        print("[a-low-i5] stage=load-fresh-checkpoint", flush=True)
        accepted_checkpoint = DEFAULT_RUN_DIR / "vae_checkpoint.pt"
        if sha256_file(accepted_checkpoint) != EXPECTED_CHECKPOINT:
            raise RuntimeError("accepted h2048 checkpoint hash mismatch")
        if sha256_file(ITERATION3_CHECKPOINT) != EXPECTED_ITERATION3_CHECKPOINT_SHA256:
            raise RuntimeError("Iteration-3 checkpoint file hash mismatch")
        run = _load_run(DEFAULT_RUN_DIR, device=device)
        cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
        cfg = replace(cfg, vae_precond_hvp_mode="autograd")
        bank = pd.read_csv(STATE_BANK)
        selected = bank.loc[bank["state_position"].eq(STATE_POSITION)]
        if len(selected) != 1 or int(selected.iloc[0]["source_weight_index"]) != SOURCE_WEIGHT_INDEX:
            raise RuntimeError("state-bank identity mismatch")
        record = run.records.iloc[SOURCE_WEIGHT_INDEX].to_dict()
        record["source_weight_index"] = SOURCE_WEIGHT_INDEX
        weight = run.weights[[SOURCE_WEIGHT_INDEX]].to(
            device=device, dtype=torch_dtype(run.cfg)
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
        active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
        named = dict(run.vae.named_parameters())
        for name, parameter in named.items():
            parameter.requires_grad_(name in active_names)
        active = [named[name] for name in active_names]
        active_parameter_count = sum(parameter.numel() for parameter in active)
        if active_parameter_count != 11_685_120:
            raise RuntimeError(f"active parameter count mismatch: {active_parameter_count}")
        serialized = torch.load(ITERATION3_CHECKPOINT, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name in active_names:
                named[name].copy_(serialized["active_model_state"][name].to(device=device))
        base_hash = _named_tensor_hash(active_names, active)
        if base_hash != EXPECTED_FINAL_PARAMETER_SHA256:
            raise RuntimeError("loaded Iteration-3 parameter hash mismatch")
        base = [parameter.detach().clone() for parameter in active]

        print("[a-low-i5] stage=dense-base-and-frozen-subspace", flush=True)
        base_metrics, hessian, matrix, eig_m, _burg = exact_helpers._evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=HESSIAN_CHUNK_SIZE,
        )
        del _burg
        stored_state = pd.read_csv(ITERATION3_STATES).loc[
            lambda frame: frame["proposal"].eq(100)
        ].iloc[0]
        stored_spectrum = (
            pd.read_csv(ITERATION3_SPECTRA)
            .loc[lambda frame: frame["proposal"].eq(100)]
            .sort_values("rank")["m_eigenvalue"]
            .to_numpy(dtype=np.float64)
        )
        replay_errors = {
            key: abs(float(base_metrics[key]) - float(stored_state[key]))
            for key in (
                "exact_a_per_dim",
                "damped_full_burg_per_dim",
                "m_max",
                "m_p50",
                "m_lt_1e_4_fraction",
                "m_lt_0p01_fraction",
                "m_lt_0p1_fraction",
            )
        }
        replay_spectrum_error = float(
            np.max(np.abs(eig_m.detach().double().cpu().numpy() - stored_spectrum))
        )
        if max([*replay_errors.values(), replay_spectrum_error]) > 1e-9:
            raise RuntimeError("base checkpoint does not replay Iteration-3 state 100")

        eig_raw, eigvec = torch.linalg.eigh(matrix.double())
        low_mask = eig_raw.clamp_min(0.0) < LOW_THRESHOLD
        dead_mask = eig_raw.clamp_min(0.0) < DEAD_THRESHOLD
        low_basis = eigvec[:, low_mask].detach()
        dead_basis = eigvec[:, dead_mask].detach()
        low_values = eig_raw[low_mask].detach()
        low_count = int(low_mask.sum().cpu())
        dead_count = int(dead_mask.sum().cpu())
        projector = low_basis @ low_basis.transpose(0, 1)
        projector_hash = sha256_tensor(projector)
        orthogonality_error = float(
            (
                low_basis.transpose(0, 1) @ low_basis
                - torch.eye(low_count, device=device, dtype=torch.float64)
            )
            .abs()
            .max()
            .cpu()
        )
        eigen_residual = float(
            (matrix.double() @ low_basis - low_basis * low_values.unsqueeze(0)).norm().cpu()
            / max(float(matrix.double().norm().cpu()), 1e-30)
        )
        eigen_order_min_diff = float(torch.diff(eig_raw).min().cpu())
        projector_gates = {
            "low_count_matches": low_count == EXPECTED_LOW_COUNT,
            "dead_count_matches": dead_count == EXPECTED_DEAD_COUNT,
            "orthogonality_passes": orthogonality_error <= ORTHOGONALITY_MAX,
            "eigen_residual_passes": eigen_residual <= EIGEN_RESIDUAL_MAX,
            "eigen_order_passes": eigen_order_min_diff >= -EIGEN_ORDER_TOL,
        }
        if not all(projector_gates.values()):
            raise RuntimeError(f"frozen projector gate failed: {projector_gates}")

        identity = torch.eye(hessian.shape[0], device=device, dtype=torch.float64)
        k_a = (4.0 * (matrix.double() - identity) @ hessian.double() / 512.0).detach()
        k_low = (
            -2.0
            * low_basis
            @ (low_basis.transpose(0, 1) @ hessian.double())
            / float(low_count)
        ).detach()

        print("[a-low-i5] stage=exact-two-gradient-pullback", flush=True)
        gradient_a, gradient_low, gradient_meta = _blocked_two_gradients(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            active=active,
            k_a=k_a,
            k_low=k_low,
        )
        norm_a = _norm(gradient_a)
        norm_low = _norm(gradient_low)
        gradient_cosine = _cosine(gradient_a, gradient_low)
        gradient_valid = bool(
            _vector_is_finite(gradient_a)
            and _vector_is_finite(gradient_low)
            and math.isfinite(norm_a)
            and math.isfinite(norm_low)
            and norm_a > 0.0
            and norm_low > 0.0
            and int(gradient_meta["basis_count"]) == 512
            and int(gradient_meta["a_unused_parameter_tensors_all_blocks"]) == 0
            and int(gradient_meta["low_unused_parameter_tensors_all_blocks"]) == 0
            and bool(gradient_meta["all_a_accumulators_float64"])
            and bool(gradient_meta["all_low_accumulators_float64"])
            and bool(gradient_meta["memory_growth_gate_pass"])
        )
        if not gradient_valid:
            raise RuntimeError("exact gradient validity gate failed")
        common = exact_helpers._unit_common_gradient(gradient_a, gradient_low, active)
        common_source_norm = _norm(common)
        gradient_diagnostics = {
            **gradient_meta,
            "gradient_a_norm": norm_a,
            "gradient_low_norm": norm_low,
            "gradient_cosine": gradient_cosine,
            "unit_common_source_norm": common_source_norm,
            "projector_hash": projector_hash,
            "projector_orthogonality_max_abs": orthogonality_error,
            "projector_eigen_residual_relative": eigen_residual,
            "eigen_order_min_diff": eigen_order_min_diff,
            "k_a_norm": float(k_a.norm().cpu()),
            "k_low_norm": float(k_low.norm().cpu()),
        }
        (staging / "gradient_diagnostics.json").write_text(
            json.dumps(gradient_diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if common_source_norm < CANCELLATION_NORM_MIN:
            final_dependency_matches = _validate_dependencies()
            cancellation_validity_gates = {
                "accepted_checkpoint_matches": sha256_file(accepted_checkpoint)
                == EXPECTED_CHECKPOINT,
                "iteration3_checkpoint_matches": sha256_file(ITERATION3_CHECKPOINT)
                == EXPECTED_ITERATION3_CHECKPOINT_SHA256,
                "base_parameter_hash_matches": base_hash == EXPECTED_FINAL_PARAMETER_SHA256,
                "active_parameter_count_matches": active_parameter_count == 11_685_120,
                "base_metrics_replay": max(replay_errors.values()) <= 1e-9,
                "base_spectrum_replay": replay_spectrum_error <= 1e-9,
                **projector_gates,
                "gradient_valid": gradient_valid,
                "parameters_restored": _named_tensor_hash(active_names, active) == base_hash,
                "source_snapshot_matches": _normalized_source_sha256(source_snapshot)
                == EXPECTED_NORMALIZED_SOURCE_SHA256,
                "live_source_unchanged": _normalized_source_sha256()
                == EXPECTED_NORMALIZED_SOURCE_SHA256,
                "dependencies_unchanged": all(final_dependency_matches.values()),
            }
            cancellation_valid = bool(all(cancellation_validity_gates.values()))
            decision = {
                "protocol_id": PROTOCOL_ID,
                "valid": cancellation_valid,
                "outcome": "equal_unit_near_antiparallel" if cancellation_valid else None,
                "local_chain_decision": "not_run_by_frozen_cancellation_gate",
                "finite_line_decision": "not_run_by_frozen_cancellation_gate",
                "base_parameter_hash": base_hash,
                "projector_hash": projector_hash,
                "gradient_diagnostics": gradient_diagnostics,
                "projector_gates": projector_gates,
                "replay_errors": replay_errors,
                "replay_spectrum_error": replay_spectrum_error,
                "validity_gates": cancellation_validity_gates,
                "elapsed_sec": time.perf_counter() - started,
            }
            (staging / "decision.json").write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if not cancellation_valid:
                raise RuntimeError("cancellation branch validity gate failed")
        else:
            direction = _negative_normalized(
                common, target_norm=TARGET_NORM, active=active
            )
            direction_norm = _norm(direction)
            nominal_a = _dot(gradient_a, direction)
            nominal_low = _dot(gradient_low, direction)
            print(
                f"[a-low-i5] gradients cos={gradient_cosine:.6g} "
                f"sum_norm={common_source_norm:.6g} dnorm={direction_norm:.6g} "
                f"slope_A={nominal_a:.6g} slope_L={nominal_low:.6g}",
                flush=True,
            )

            print("[a-low-i5] stage=frozen-exact-line", flush=True)
            endpoint_rows: list[dict[str, Any]] = []
            global_spectrum_rows: list[dict[str, float | int]] = []
            projected_spectrum_rows: list[dict[str, float | int]] = []
            parameter_snapshots: dict[float, list[torch.Tensor]] = {}
            hessian_snapshots: dict[float, torch.Tensor] = {}
            try:
                for index, alpha in enumerate(ALPHAS, start=1):
                    _set_parameters(active, base, direction, alpha)
                    endpoint_hash = _named_tensor_hash(active_names, active)
                    primary, primary_h, primary_eig, primary_projected_eig = _endpoint(
                        run=run,
                        z=z,
                        record=record,
                        low_basis=low_basis,
                        dead_basis=dead_basis,
                    )
                    repeat, _repeat_h, repeat_eig, repeat_projected_eig = _endpoint(
                        run=run,
                        z=z,
                        record=record,
                        low_basis=low_basis,
                        dead_basis=dead_basis,
                    )
                    errors = _repeat_errors(
                        primary,
                        repeat,
                        primary_eig,
                        repeat_eig,
                        primary_projected_eig,
                        repeat_projected_eig,
                    )
                    hash_unchanged = _named_tensor_hash(active_names, active) == endpoint_hash
                    endpoint_rows.append(
                        {
                            "alpha": alpha,
                            "endpoint_parameter_hash": endpoint_hash,
                            "repeat_parameter_hash_unchanged": int(hash_unchanged),
                            **primary,
                            **errors,
                        }
                    )
                    for rank, value in enumerate(primary_eig.cpu().numpy()):
                        global_spectrum_rows.append(
                            {"alpha": alpha, "rank": rank, "eigenvalue": float(value)}
                        )
                    for rank, value in enumerate(primary_projected_eig.cpu().numpy()):
                        projected_spectrum_rows.append(
                            {"alpha": alpha, "rank": rank, "eigenvalue": float(value)}
                        )
                    if abs(alpha) in LOCAL_RADII:
                        parameter_snapshots[alpha] = [
                            parameter.detach().cpu().clone() for parameter in active
                        ]
                        hessian_snapshots[alpha] = primary_h.detach().cpu().clone()
                    print(
                        f"[a-low-i5] endpoint={index}/{len(ALPHAS)} alpha={alpha:+.7g} "
                        f"A={primary['exact_a_per_dim']:.7g} "
                        f"B={primary['damped_full_burg_per_dim']:.7g} "
                        f"Elow={primary['frozen_low_energy']:.7g} "
                        f"mmax={primary['m_max']:.7g} p50={primary['m_p50']:.3g} "
                        f"proj_p50={primary['projected_low_p50']:.3g}",
                        flush=True,
                    )
                    del primary_h, primary_eig, primary_projected_eig
                    del _repeat_h, repeat_eig, repeat_projected_eig
                    _set_parameters(active, base, direction, 0.0)
                    if _named_tensor_hash(active_names, active) != base_hash:
                        raise RuntimeError(f"base restoration failed after alpha={alpha}")
            finally:
                _set_parameters(active, base, direction, 0.0)

            endpoints = pd.DataFrame(endpoint_rows).sort_values("alpha")
            global_spectra = pd.DataFrame(global_spectrum_rows).sort_values(
                ["alpha", "rank"]
            )
            projected_spectra = pd.DataFrame(projected_spectrum_rows).sort_values(
                ["alpha", "rank"]
            )
            endpoints.to_csv(staging / "endpoint_metrics.csv", index=False)
            global_spectra.to_csv(staging / "global_spectra.csv", index=False)
            projected_spectra.to_csv(staging / "projected_low_spectra.csv", index=False)

            print("[a-low-i5] stage=realized-chain-audit", flush=True)
            chain_rows: list[dict[str, Any]] = []
            for radius in LOCAL_RADII:
                plus = parameter_snapshots[radius]
                minus = parameter_snapshots[-radius]
                parameter_stats = _realized_parameter_stats(
                    plus=plus,
                    minus=minus,
                    base=base,
                    direction=direction,
                    gradient_a=gradient_a,
                    gradient_low=gradient_low,
                    radius=radius,
                )
                h_secant = (
                    hessian_snapshots[radius] - hessian_snapshots[-radius]
                ) / (2.0 * radius)
                s_h_a = float((k_a.detach().cpu() * h_secant).sum())
                s_h_low = float((k_low.detach().cpu() * h_secant).sum())
                plus_row = endpoints.loc[np.isclose(endpoints["alpha"], radius)].iloc[0]
                minus_row = endpoints.loc[np.isclose(endpoints["alpha"], -radius)].iloc[0]
                direct_a = float(
                    (plus_row["exact_a_per_dim"] - minus_row["exact_a_per_dim"])
                    / (2.0 * radius)
                )
                direct_low = float(
                    -(plus_row["frozen_low_energy"] - minus_row["frozen_low_energy"])
                    / (2.0 * radius)
                )
                chain_rows.append(
                    _chain_row(
                        radius=radius,
                        objective="A",
                        nominal=nominal_a,
                        s_theta=float(parameter_stats["s_theta_a"]),
                        s_h=s_h_a,
                        direct=direct_a,
                        parameter_stats=parameter_stats,
                    )
                )
                chain_rows.append(
                    _chain_row(
                        radius=radius,
                        objective="L_low",
                        nominal=nominal_low,
                        s_theta=float(parameter_stats["s_theta_low"]),
                        s_h=s_h_low,
                        direct=direct_low,
                        parameter_stats=parameter_stats,
                    )
                )
            chain = pd.DataFrame(chain_rows).sort_values(["objective", "radius"])
            chain.to_csv(staging / "realized_chain.csv", index=False)
            cauchy_errors = {
                objective: _relative_error(
                    float(
                        chain.loc[
                            chain["objective"].eq(objective) & chain["radius"].eq(LOCAL_RADII[0]),
                            "direct",
                        ].iloc[0]
                    ),
                    float(
                        chain.loc[
                            chain["objective"].eq(objective) & chain["radius"].eq(LOCAL_RADII[1]),
                            "direct",
                        ].iloc[0]
                    ),
                )
                for objective in ("A", "L_low")
            }
            local_chain_pass = bool(
                chain["row_pass"].all()
                and all(
                    value <= CHAIN_CAUCHY_RELATIVE_ERROR_MAX
                    for value in cauchy_errors.values()
                )
            )

            tolerances = _metric_tolerances(endpoints)
            candidates = _candidate_rows(endpoints, tolerances)
            passing = candidates.loc[candidates["candidate_pass"]].sort_values(
                "alpha", ascending=False
            )
            selected_alpha = None if passing.empty else float(passing.iloc[0]["alpha"])
            candidates.to_csv(staging / "finite_candidates.csv", index=False)
            all_repeat_columns = [
                column
                for column in endpoints.columns
                if column.startswith("repeat_") and column.endswith("error")
            ]
            max_repeat_error = float(
                endpoints[all_repeat_columns].to_numpy(dtype=np.float64).max()
            )
            finite_line_valid = bool(
                len(endpoints) == len(ALPHAS)
                and len(global_spectra) == len(ALPHAS) * 512
                and len(projected_spectra) == len(ALPHAS) * EXPECTED_LOW_COUNT
                and _numeric_finite(endpoints)
                and _numeric_finite(global_spectra)
                and _numeric_finite(projected_spectra)
                and bool(endpoints["repeat_parameter_hash_unchanged"].eq(1).all())
                and max_repeat_error <= 1e-10
                and bool(endpoints["a_direct_abs_error"].le(1e-9).all())
                and bool(endpoints["a_trace_abs_error"].le(1e-9).all())
            )
            if not finite_line_valid:
                raise RuntimeError("finite line validity gate failed")

            final_dependency_matches = _validate_dependencies()
            validity_gates = {
                "accepted_checkpoint_matches": sha256_file(accepted_checkpoint)
                == EXPECTED_CHECKPOINT,
                "iteration3_checkpoint_matches": sha256_file(ITERATION3_CHECKPOINT)
                == EXPECTED_ITERATION3_CHECKPOINT_SHA256,
                "base_parameter_hash_matches": base_hash == EXPECTED_FINAL_PARAMETER_SHA256,
                "active_parameter_count_matches": active_parameter_count == 11_685_120,
                "base_metrics_replay": max(replay_errors.values()) <= 1e-9,
                "base_spectrum_replay": replay_spectrum_error <= 1e-9,
                **projector_gates,
                "gradient_valid": gradient_valid,
                "direction_norm_matches": abs(direction_norm - TARGET_NORM) <= 1e-6,
                "finite_line_valid": finite_line_valid,
                "parameters_restored": _named_tensor_hash(active_names, active) == base_hash,
                "source_snapshot_matches": _normalized_source_sha256(source_snapshot)
                == EXPECTED_NORMALIZED_SOURCE_SHA256,
                "live_source_unchanged": _normalized_source_sha256()
                == EXPECTED_NORMALIZED_SOURCE_SHA256,
                "dependencies_unchanged": all(final_dependency_matches.values()),
            }
            valid = bool(all(validity_gates.values()))
            local_decision = (
                "joint_local_chain_valid" if local_chain_pass else "no_local_chain_claim"
            )
            finite_decision = (
                "finite_joint_repair" if selected_alpha is not None else "no_guarded_finite_endpoint"
            )
            if not local_chain_pass:
                outcome = "no_local_chain_claim"
            elif selected_alpha is not None:
                outcome = "finite_joint_repair"
            else:
                outcome = "local_only_finite_coupling"
            decision = {
                "protocol_id": PROTOCOL_ID,
                "valid": valid,
                "outcome": outcome if valid else None,
                "local_chain_decision": local_decision if valid else None,
                "finite_line_decision": finite_decision if valid else None,
                "base_parameter_hash": base_hash,
                "base_metrics": base_metrics,
                "replay_errors": replay_errors,
                "replay_spectrum_error": replay_spectrum_error,
                "projector_hash": projector_hash,
                "projector_gates": projector_gates,
                "low_count": low_count,
                "dead_count": dead_count,
                "gradient_diagnostics": gradient_diagnostics,
                "direction_norm": direction_norm,
                "nominal_a_derivative": nominal_a,
                "nominal_low_loss_derivative": nominal_low,
                "cauchy_relative_errors": cauchy_errors,
                "local_chain_pass": local_chain_pass,
                "metric_tolerances": tolerances,
                "finite_candidate_count": int(len(passing)),
                "selected_largest_passing_alpha": selected_alpha,
                "validity_gates": validity_gates,
                "elapsed_sec": time.perf_counter() - started,
            }
            (staging / "decision.json").write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if not valid:
                raise RuntimeError("main branch validity gate failed")
            _plot(staging, endpoints, global_spectra, projected_spectra, selected_alpha)

        if not bool(decision["valid"]):
            raise RuntimeError("refusing to finalize an invalid decision")
        artifact_manifest = _manifest(staging, source_snapshot)
        (staging / "artifact_manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        finalized = {
            "protocol_id": PROTOCOL_ID,
            "status": "complete_awaiting_independent_review",
            "valid": bool(decision["valid"]),
            "outcome": decision["outcome"],
            "decision_sha256": sha256_file(staging / "decision.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        (staging / "INCOMPLETE").unlink()
        (staging / "FINALIZED.json").write_text(
            json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[a-low-i5] complete valid={decision['valid']} outcome={decision['outcome']} "
            f"local={decision['local_chain_decision']} finite={decision['finite_line_decision']} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        os.replace(staging, final_output)
        print(f"[a-low-i5] published output={final_output}", flush=True)
    except Exception:
        print("[a-low-i5] FAILED; staging retained", flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
