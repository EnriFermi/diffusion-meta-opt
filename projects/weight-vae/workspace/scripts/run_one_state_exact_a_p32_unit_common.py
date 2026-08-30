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
    _task_set_for_record,
)
from scripts import audit_one_state_exact_a_proposal3_cross as exact_helpers
from scripts import run_one_state_a_full_burg_p32_training as p32
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _named_tensor_hash,
    _negative_normalized,
    _norm,
    _vector_is_finite,
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
PROTOCOL_ID = "one_state_exact_a_p32_unit_common_i3_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_3_p32_unit_common/protocol.md"
PRE_RUN_REVIEW_PATH = OUTPUT_ROOT / "iteration_3_p32_unit_common/pre_run_review.md"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration3_p32_unit_common_production"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_3_frozen_dependency_manifest.json"
EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = "d4462eade6b8bf04f6734723fcc1c287da362e3f755e5ecf7965ba731b7fdb56"
EXPECTED_NORMALIZED_SOURCE_SHA256 = "f19dacc737cca902d38518f440dac478acc5620395bb32bccbc42cbc0a51a659"

EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "1cf3bdeb383c64b36b3ca69a956457833d0aa0d8dd1f6696e3964033518053d4"
)
SOURCE_WEIGHT_INDEX = 378
STATE_POSITION = 2
PROPOSALS = 100
POOL_DRAWS = 8
PAIRS_PER_DRAW = 4
TARGET_NORM = 0.04892722657548397
LINE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
EPSILON = 1e-4
HESSIAN_CHUNK_SIZE = 64
CANCELLATION_MIN_NORM = 0.5
TAIL_START = 81
TAIL_REQUIRED = 16

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


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalized_source_sha256() -> str:
    masked = (
        "EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    lines = Path(__file__).read_text(encoding="utf-8").splitlines(keepends=True)
    normalized: list[str] = []
    for line in lines:
        prefix = next((candidate for candidate in masked if line.startswith(candidate)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _validate_frozen_dependencies() -> dict[str, bool]:
    if "TO_BE_FROZEN" in (
        EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        EXPECTED_NORMALIZED_SOURCE_SHA256,
    ):
        raise RuntimeError("runner has not been frozen")
    if _normalized_source_sha256() != EXPECTED_NORMALIZED_SOURCE_SHA256:
        raise RuntimeError("normalized source hash mismatch")
    if sha256_file(FROZEN_DEPENDENCY_MANIFEST) != EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256:
        raise RuntimeError("frozen dependency manifest hash mismatch")
    expected = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    matches = {
        relative: (ROOT / relative).is_file() and sha256_file(ROOT / relative) == digest
        for relative, digest in expected.items()
    }
    failed = [relative for relative, ok in matches.items() if not ok]
    if failed:
        raise RuntimeError(f"frozen dependency mismatch: {failed}")
    return matches


def _numeric_finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def _acceptance_tolerances(base_noise: Mapping[str, float]) -> dict[str, float]:
    return {
        "a": max(5.0 * float(base_noise["a"]), 1e-8),
        "b": max(5.0 * float(base_noise["b"]), 1e-8),
        "m_max": max(5.0 * float(base_noise["m_max"]), 1e-8),
        "m_p50": max(5.0 * float(base_noise["m_p50"]), 1e-10),
        "m_lt_0p1_fraction": float(base_noise["m_lt_0p1_fraction"]),
    }


def _acceptance_failures(
    current: Mapping[str, float],
    candidate: Mapping[str, float],
    tolerances: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    if not (
        float(candidate["exact_a_per_dim"])
        < float(current["exact_a_per_dim"]) - float(tolerances["a"])
    ):
        failures.append("exact_a_not_lower")
    if not (
        float(candidate["damped_full_burg_per_dim"])
        < float(current["damped_full_burg_per_dim"]) - float(tolerances["b"])
    ):
        failures.append("full_b_not_lower")
    if not (
        float(candidate["m_max"])
        <= float(current["m_max"]) + float(tolerances["m_max"])
    ):
        failures.append("m_max_increased")
    if not (
        float(candidate["m_p50"])
        >= float(current["m_p50"]) - float(tolerances["m_p50"])
    ):
        failures.append("m_p50_decreased")
    if not (
        float(candidate["m_lt_0p1_fraction"])
        <= float(current["m_lt_0p1_fraction"])
        + float(tolerances["m_lt_0p1_fraction"])
    ):
        failures.append("low_fraction_increased")
    return failures


def _tail_state_pass(row: Mapping[str, float]) -> bool:
    return bool(
        float(row["trace_m_per_dim"]) >= 0.0456087
        and float(row["m_lt_1e_4_fraction"]) * 512.0 <= 51.0 + 1e-9
        and float(row["m_lt_0p01_fraction"]) * 512.0 <= 256.0 + 1e-9
        and float(row["rho_lower90"]) >= 0.25
    )


def _spectrum_rows(proposal: int, eig_m: torch.Tensor) -> list[dict[str, float | int]]:
    ordered = eig_m.detach().double().sort().values.cpu().numpy()
    return [
        {
            "proposal": proposal,
            "rank": rank,
            "m_eigenvalue": float(value),
            "a_contribution": float((value - 1.0) ** 2),
        }
        for rank, value in enumerate(ordered)
    ]


def _state_row(
    *,
    proposal: int,
    accepted: bool,
    alpha: float,
    metrics: Mapping[str, float],
    initial: Mapping[str, float],
) -> dict[str, float | int]:
    denominator = float(initial["exact_a_per_dim"]) - float(metrics["exact_a_per_dim"])
    rho = (
        float("nan")
        if denominator <= 0.0
        else (
            float(initial["a_low90_abs_per_dim"])
            - float(metrics["a_low90_abs_per_dim"])
        )
        / denominator
    )
    return {
        "proposal": proposal,
        "accepted": int(accepted),
        "accepted_alpha": float(alpha),
        "accepted_radius": float(alpha) * TARGET_NORM if accepted else 0.0,
        "rho_lower90": rho,
        **{key: float(value) for key, value in metrics.items()},
    }


def _plot_artifacts(output: Path) -> None:
    states = pd.read_csv(output / "state_metrics.csv")
    spectra = pd.read_csv(output / "state_spectra.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axis_a = axes[0, 0]
    axis_a.plot(states["proposal"], states["exact_a_per_dim"], color="#1d4e89", linewidth=2.2)
    axis_a.axhline(0.90, color="#b23a48", linestyle="--", label="success: A=0.90")
    axis_a.axhline(1.00, color="#8a6d1d", linestyle=":", label="tail: A=1.00")
    axis_a.set(title="Exact dense A", xlabel="proposal", ylabel="A per dimension")
    axis_a.legend()

    axis_b = axes[0, 1]
    axis_b.plot(
        states["proposal"],
        states["damped_full_burg_per_dim"],
        color="#287271",
        linewidth=2.2,
    )
    axis_b.set(title="Damped full Burg B", xlabel="proposal", ylabel="B per dimension")

    axis_bulk = axes[1, 0]
    axis_bulk.plot(states["proposal"], states["trace_m_per_dim"], label="trace(M)/m")
    axis_bulk.plot(states["proposal"], states["m_lt_1e_4_fraction"], label="fraction <1e-4")
    axis_bulk.plot(states["proposal"], states["m_lt_0p01_fraction"], label="fraction <1e-2")
    axis_bulk.plot(states["proposal"], states["m_lt_0p1_fraction"], label="fraction <0.1")
    axis_bulk.set(title="Bulk / collapse diagnostics", xlabel="proposal", ylabel="value")
    axis_bulk.legend()

    axis_spectrum = axes[1, 1]
    selected = sorted(set([0, 25, 50, 75, 100]))
    for proposal in selected:
        rows = spectra.loc[spectra["proposal"].eq(proposal)].sort_values("rank")
        axis_spectrum.plot(
            rows["rank"],
            np.clip(rows["m_eigenvalue"], 1e-14, None),
            label=f"state {proposal}",
        )
    axis_spectrum.axhline(1.0, color="black", linewidth=1.0)
    axis_spectrum.set_yscale("log")
    axis_spectrum.set(title="Ordered M spectrum", xlabel="eigenvalue rank", ylabel="eigenvalue")
    axis_spectrum.legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("One-state P32 unit-common exact-feasible trajectory")
    fig.savefig(output / "exact_a_b_and_spectrum.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    accepted = states.loc[states["proposal"].gt(0) & states["accepted"].eq(1)]
    rejected = states.loc[states["proposal"].gt(0) & states["accepted"].eq(0)]
    axes[0].scatter(accepted["proposal"], accepted["accepted_alpha"], s=25, label="accepted")
    if len(rejected):
        axes[0].scatter(
            rejected["proposal"],
            np.full(len(rejected), LINE_ALPHAS[-1]),
            marker="x",
            color="#b23a48",
            label="no-op",
        )
    axes[0].set_yscale("log", base=2)
    axes[0].set(title="Accepted line radius", xlabel="proposal", ylabel="alpha")
    axes[0].legend()
    axes[1].plot(states["proposal"], states["rho_lower90"], color="#7b2cbf")
    axes[1].axhline(0.25, color="#b23a48", linestyle="--")
    axes[1].set(title="Lower-90% share of cumulative A reduction", xlabel="proposal", ylabel="rho90")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "acceptance_and_lower90.png", dpi=180)
    plt.close(fig)


def _artifact_manifest(output: Path) -> dict[str, Any]:
    excluded = {"artifact_manifest.json", "INCOMPLETE", "FINALIZED.json", "run.log"}
    paths = sorted(path for path in output.iterdir() if path.is_file() and path.name not in excluded)
    return {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": _source_sha256(),
        "normalized_source_sha256": _normalized_source_sha256(),
        "artifacts": {path.name: sha256_file(path) for path in paths},
    }


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("production runner accepts no CLI overrides")
    frozen_matches = _validate_frozen_dependencies()
    final_output = DEFAULT_OUTPUT.resolve()
    staging = Path(str(final_output) + ".incomplete")
    if final_output.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {final_output} or {staging}")
    staging.mkdir(parents=True)
    (staging / "INCOMPLETE").write_text(PROTOCOL_ID + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), staging / "executed_source_snapshot.py")
    shutil.copy2(PROTOCOL_PATH, staging / "protocol_snapshot.md")
    shutil.copy2(PRE_RUN_REVIEW_PATH, staging / "pre_run_review_snapshot.md")
    shutil.copy2(FROZEN_DEPENDENCY_MANIFEST, staging / "frozen_dependency_manifest_snapshot.json")

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_handle = (staging / "run.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, log_handle)  # type: ignore[assignment]
    started = time.perf_counter()
    device = torch.device("cuda:0")
    resolved: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "device": str(device),
        "dtype": "float32 model/HVP; float64 dense diagnostics",
        "seed": "frozen P32 iteration-5 100-proposal schedule",
        "source_weight_index": SOURCE_WEIGHT_INDEX,
        "state_position": STATE_POSITION,
        "proposals": PROPOSALS,
        "pool_draws": POOL_DRAWS,
        "pairs_per_draw": PAIRS_PER_DRAW,
        "target_norm": TARGET_NORM,
        "line_alphas": list(LINE_ALPHAS),
        "epsilon": EPSILON,
        "hessian_chunk_size": HESSIAN_CHUNK_SIZE,
        "optimizer": "none; raw normalized common direction",
        "cache_mode": "reuse accepted checkpoint/state bank; recompute all gradients and exact endpoints",
        "output_dir": str(final_output),
        "staging_dir": str(staging),
        "source_sha256": _source_sha256(),
        "normalized_source_sha256": _normalized_source_sha256(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "frozen_dependency_manifest_sha256": sha256_file(FROZEN_DEPENDENCY_MANIFEST),
        "frozen_dependency_matches": frozen_matches,
    }
    (staging / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[exact-a-i3] startup {json.dumps(resolved, sort_keys=True)}", flush=True)

    try:
        print("[exact-a-i3] stage=load", flush=True)
        checkpoint_path = DEFAULT_RUN_DIR / "vae_checkpoint.pt"
        if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT:
            raise RuntimeError("accepted checkpoint hash mismatch")
        run = _load_run(DEFAULT_RUN_DIR, device=device)
        cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=PAIRS_PER_DRAW, batch_size=16384)
        cfg = replace(cfg, vae_precond_hvp_mode="autograd")
        bank = pd.read_csv(STATE_BANK)
        selected = bank.loc[bank["state_position"].eq(STATE_POSITION)]
        if len(selected) != 1 or int(selected.iloc[0]["source_weight_index"]) != SOURCE_WEIGHT_INDEX:
            raise RuntimeError("frozen state bank identity mismatch")
        record = run.records.iloc[SOURCE_WEIGHT_INDEX].to_dict()
        record["source_weight_index"] = SOURCE_WEIGHT_INDEX
        weight = run.weights[[SOURCE_WEIGHT_INDEX]].to(device=device, dtype=torch_dtype(run.cfg))
        with torch.no_grad():
            z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
        task_set = _task_set_for_record(run.task_tensors, record)
        full_ce = all(
            _batch_indices(
                task_set,
                batch_size=int(cfg.vae_precond_batch_size),
                step=10,
                sample_key=SOURCE_WEIGHT_INDEX,
                pair_key=pair_key,
            )
            is None
            for pair_key in range(8)
        )
        if not full_ce:
            raise RuntimeError("frozen setup must use full CE batch")
        active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
        named = dict(run.vae.named_parameters())
        for name, parameter in named.items():
            parameter.requires_grad_(name in active_names)
        active = [named[name] for name in active_names]
        active_parameter_count = sum(parameter.numel() for parameter in active)
        z_hash = sha256_tensor(z)
        initial_parameter_hash = _named_tensor_hash(active_names, active)
        if z_hash != EXPECTED_Z_SHA256 or initial_parameter_hash != EXPECTED_INITIAL_PARAMETER_SHA256:
            raise RuntimeError("initial z/parameter fingerprint mismatch")
        if active_parameter_count != 11_685_120:
            raise RuntimeError(f"active parameter count mismatch: {active_parameter_count}")

        seed_preflight = p32._seed_schedule_preflight()
        if not seed_preflight["valid"]:
            raise RuntimeError("P32 seed preflight failed")
        (staging / "p32_seed_schedule_preflight.json").write_text(
            json.dumps(seed_preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        initial_cuda_rng = torch.cuda.get_rng_state(device).clone()
        initial_cpu_rng = torch.random.get_rng_state().clone()

        print("[exact-a-i3] stage=exact-state0", flush=True)
        current_metrics, _h, _m, current_eig, current_burg = exact_helpers._evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=HESSIAN_CHUNK_SIZE,
        )
        del _h, _m
        repeat_metrics, _rh, _rm, repeat_eig, _rburg = exact_helpers._evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=HESSIAN_CHUNK_SIZE,
        )
        del _rh, _rm, _rburg
        base_noise = {
            "a": abs(float(current_metrics["exact_a_per_dim"]) - float(repeat_metrics["exact_a_per_dim"])),
            "b": abs(
                float(current_metrics["damped_full_burg_per_dim"])
                - float(repeat_metrics["damped_full_burg_per_dim"])
            ),
            "m_max": abs(float(current_metrics["m_max"]) - float(repeat_metrics["m_max"])),
            "m_p50": abs(float(current_metrics["m_p50"]) - float(repeat_metrics["m_p50"])),
            "m_lt_0p1_fraction": abs(
                float(current_metrics["m_lt_0p1_fraction"])
                - float(repeat_metrics["m_lt_0p1_fraction"])
            ),
            "spectrum_max_abs": float(
                (current_eig.detach().double() - repeat_eig.detach().double()).abs().max().cpu()
            ),
        }
        tolerances = _acceptance_tolerances(base_noise)
        pd.DataFrame(
            [
                {"evaluation": "primary", **current_metrics},
                {"evaluation": "repeat", **repeat_metrics},
            ]
        ).to_csv(staging / "state0_repeatability.csv", index=False)
        del repeat_eig

        initial_metrics = dict(current_metrics)
        state_rows: list[dict[str, float | int]] = [
            _state_row(
                proposal=0,
                accepted=True,
                alpha=0.0,
                metrics=current_metrics,
                initial=initial_metrics,
            )
        ]
        spectrum_rows = _spectrum_rows(0, current_eig)
        pair_rows: list[dict[str, float | int]] = []
        draw_rows: list[dict[str, float | int]] = []
        proposal_rows: list[dict[str, Any]] = []
        line_rows: list[dict[str, Any]] = []
        accepted_count = 0
        cumulative_radius = 0.0

        print(
            f"[exact-a-i3] state=0 A={current_metrics['exact_a_per_dim']:.8g} "
            f"B={current_metrics['damped_full_burg_per_dim']:.8g} "
            f"trace={current_metrics['trace_m_per_dim']:.8g} mmax={current_metrics['m_max']:.8g}",
            flush=True,
        )

        for proposal in range(1, PROPOSALS + 1):
            proposal_started = time.perf_counter()
            base = [parameter.detach().clone() for parameter in active]
            base_hash = _named_tensor_hash(active_names, active)
            base_metrics = dict(current_metrics)
            base_eig = current_eig.detach().clone()

            print(
                f"[exact-a-i3] stage=p32 proposal={proposal}/{PROPOSALS} "
                f"A={base_metrics['exact_a_per_dim']:.7g} B={base_metrics['damped_full_burg_per_dim']:.7g}",
                flush=True,
            )
            gradient_a, gradient_b, local_pairs, local_draws = exact_helpers._p32_components(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                active=active,
                burg_gradient=current_burg,
                proposal=proposal,
            )
            pair_rows.extend(local_pairs)
            draw_rows.extend(local_draws)
            norm_a = _norm(gradient_a)
            norm_b = _norm(gradient_b)
            dot_ab = sum(
                float((a.detach().double() * b.detach().double()).sum().cpu())
                for a, b in zip(gradient_a, gradient_b, strict=True)
            )
            cosine_ab = dot_ab / max(norm_a * norm_b, 1e-30)
            source_valid = bool(
                math.isfinite(norm_a)
                and math.isfinite(norm_b)
                and norm_a > 0.0
                and norm_b > 0.0
                and _vector_is_finite(gradient_a)
                and _vector_is_finite(gradient_b)
            )
            source_norm = float("nan")
            direction: Vector | None = None
            if source_valid:
                common = exact_helpers._unit_common_gradient(gradient_a, gradient_b, active)
                source_norm = _norm(common)
                if math.isfinite(source_norm) and source_norm >= CANCELLATION_MIN_NORM:
                    direction = _negative_normalized(common, target_norm=TARGET_NORM, active=active)
                del common
            del gradient_a, gradient_b

            selected_alpha = 0.0
            selected_hash: str | None = None
            selected_metrics: dict[str, float] | None = None
            selected_eig: torch.Tensor | None = None
            selected_burg: torch.Tensor | None = None
            attempted = 0
            failure_counts: dict[str, int] = {}
            if direction is not None:
                for alpha in LINE_ALPHAS:
                    attempted += 1
                    _set_parameters(active, base, direction, alpha)
                    candidate, _ch, _cm, candidate_eig, candidate_burg = exact_helpers._evaluate_dense(
                        run=run,
                        z=z,
                        record=record,
                        epsilon=EPSILON,
                        hessian_chunk_size=HESSIAN_CHUNK_SIZE,
                    )
                    del _ch, _cm
                    failures = _acceptance_failures(base_metrics, candidate, tolerances)
                    repeat_checked = 0
                    repeat_errors = {
                        "repeat_a_abs_error": 0.0,
                        "repeat_b_abs_error": 0.0,
                        "repeat_m_max_abs_error": 0.0,
                        "repeat_m_p50_abs_error": 0.0,
                        "repeat_spectrum_max_abs_error": 0.0,
                    }
                    repeated_candidate: dict[str, float] | None = None
                    repeated_eig: torch.Tensor | None = None
                    repeated_burg: torch.Tensor | None = None
                    endpoint_hash = ""
                    if not failures:
                        repeat_checked = 1
                        endpoint_hash = _named_tensor_hash(active_names, active)
                        repeated_candidate, _rch, _rcm, repeated_eig, repeated_burg = (
                            exact_helpers._evaluate_dense(
                                run=run,
                                z=z,
                                record=record,
                                epsilon=EPSILON,
                                hessian_chunk_size=HESSIAN_CHUNK_SIZE,
                            )
                        )
                        del _rch, _rcm
                        repeat_errors = {
                            "repeat_a_abs_error": abs(
                                float(candidate["exact_a_per_dim"])
                                - float(repeated_candidate["exact_a_per_dim"])
                            ),
                            "repeat_b_abs_error": abs(
                                float(candidate["damped_full_burg_per_dim"])
                                - float(repeated_candidate["damped_full_burg_per_dim"])
                            ),
                            "repeat_m_max_abs_error": abs(
                                float(candidate["m_max"])
                                - float(repeated_candidate["m_max"])
                            ),
                            "repeat_m_p50_abs_error": abs(
                                float(candidate["m_p50"])
                                - float(repeated_candidate["m_p50"])
                            ),
                            "repeat_spectrum_max_abs_error": float(
                                (
                                    candidate_eig.detach().double()
                                    - repeated_eig.detach().double()
                                )
                                .abs()
                                .max()
                                .cpu()
                            ),
                        }
                        if (
                            max(repeat_errors.values()) > 1e-10
                            or _named_tensor_hash(active_names, active) != endpoint_hash
                        ):
                            failures.append("endpoint_repeat_mismatch")
                        failures.extend(
                            failure
                            for failure in _acceptance_failures(
                                base_metrics, repeated_candidate, tolerances
                            )
                            if failure not in failures
                        )
                    line_rows.append(
                        {
                            "proposal": proposal,
                            "alpha": alpha,
                            "passes": int(not failures),
                            "failures": "|".join(failures),
                            "delta_a": float(candidate["exact_a_per_dim"] - base_metrics["exact_a_per_dim"]),
                            "delta_b": float(
                                candidate["damped_full_burg_per_dim"]
                                - base_metrics["damped_full_burg_per_dim"]
                            ),
                            "delta_m_max": float(candidate["m_max"] - base_metrics["m_max"]),
                            "delta_m_p50": float(candidate["m_p50"] - base_metrics["m_p50"]),
                            "delta_m_lt_0p1_fraction": float(
                                candidate["m_lt_0p1_fraction"]
                                - base_metrics["m_lt_0p1_fraction"]
                            ),
                            "repeat_checked": repeat_checked,
                            **repeat_errors,
                            **{key: float(value) for key, value in candidate.items()},
                        }
                    )
                    for failure in failures:
                        failure_counts[failure] = failure_counts.get(failure, 0) + 1
                    if not failures:
                        selected_alpha = alpha
                        assert repeated_candidate is not None
                        assert repeated_eig is not None and repeated_burg is not None
                        selected_hash = endpoint_hash
                        selected_metrics = repeated_candidate
                        selected_eig = repeated_eig.detach().clone()
                        selected_burg = repeated_burg.detach().clone()
                        del candidate_eig, candidate_burg, repeated_eig, repeated_burg
                        break
                    del candidate_eig, candidate_burg
                    if repeated_eig is not None:
                        del repeated_eig
                    if repeated_burg is not None:
                        del repeated_burg

            accepted = selected_metrics is not None
            if accepted:
                assert selected_eig is not None and selected_burg is not None and selected_hash is not None
                current_metrics = selected_metrics
                current_eig = selected_eig
                current_burg = selected_burg
                accepted_count += 1
                cumulative_radius += selected_alpha * TARGET_NORM
                final_hash = _named_tensor_hash(active_names, active)
                if final_hash != selected_hash:
                    raise RuntimeError(f"accepted endpoint hash mismatch at proposal {proposal}")
            else:
                zero_direction = [torch.zeros_like(parameter) for parameter in active]
                _set_parameters(active, base, zero_direction, 0.0)
                del zero_direction
                current_metrics = base_metrics
                current_eig = base_eig
                final_hash = _named_tensor_hash(active_names, active)
                if final_hash != base_hash:
                    raise RuntimeError(f"rejected proposal {proposal} was not an exact no-op")

            state = _state_row(
                proposal=proposal,
                accepted=accepted,
                alpha=selected_alpha,
                metrics=current_metrics,
                initial=initial_metrics,
            )
            state_rows.append(state)
            spectrum_rows.extend(_spectrum_rows(proposal, current_eig))
            proposal_rows.append(
                {
                    "proposal": proposal,
                    "accepted": int(accepted),
                    "selected_alpha": selected_alpha,
                    "accepted_radius": selected_alpha * TARGET_NORM if accepted else 0.0,
                    "attempted_endpoint_count": attempted,
                    "gradient_a_norm": norm_a,
                    "gradient_b_norm": norm_b,
                    "gradient_ab_dot": dot_ab,
                    "gradient_ab_cosine": cosine_ab,
                    "unit_common_source_norm": source_norm,
                    "source_valid": int(source_valid),
                    "cancellation_gate_pass": int(direction is not None),
                    "base_parameter_hash": base_hash,
                    "final_parameter_hash": final_hash,
                    "failure_counts": json.dumps(failure_counts, sort_keys=True),
                    "elapsed_sec": time.perf_counter() - proposal_started,
                    "cumulative_accepted_count": accepted_count,
                    "cumulative_path_radius": cumulative_radius,
                }
            )
            del base, base_eig
            if direction is not None:
                del direction
            print(
                f"[exact-a-i3] proposal={proposal}/{PROPOSALS} accepted={int(accepted)} "
                f"alpha={selected_alpha:.7g} A={current_metrics['exact_a_per_dim']:.7g} "
                f"B={current_metrics['damped_full_burg_per_dim']:.7g} "
                f"mmax={current_metrics['m_max']:.7g} p50={current_metrics['m_p50']:.3g} "
                f"accepted_total={accepted_count} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

        state_frame = pd.DataFrame(state_rows)
        proposal_frame = pd.DataFrame(proposal_rows)
        line_frame = pd.DataFrame(line_rows)
        pair_frame = pd.DataFrame(pair_rows)
        draw_frame = pd.DataFrame(draw_rows)
        spectrum_frame = pd.DataFrame(spectrum_rows)
        state_frame.to_csv(staging / "state_metrics.csv", index=False)
        proposal_frame.to_csv(staging / "proposal_diagnostics.csv", index=False)
        line_frame.to_csv(staging / "line_endpoints.csv", index=False)
        pair_frame.to_csv(staging / "p32_pair_scalars.csv", index=False)
        draw_frame.to_csv(staging / "p32_draw_diagnostics.csv", index=False)
        spectrum_frame.to_csv(staging / "state_spectra.csv", index=False)

        print("[exact-a-i3] stage=final-replay", flush=True)
        final_parameter_hash = _named_tensor_hash(active_names, active)
        replay_metrics, _fh, _fm, replay_eig, _fb = exact_helpers._evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=HESSIAN_CHUNK_SIZE,
        )
        del _fh, _fm, _fb
        replay_errors = {
            "a": abs(float(replay_metrics["exact_a_per_dim"]) - float(current_metrics["exact_a_per_dim"])),
            "b": abs(
                float(replay_metrics["damped_full_burg_per_dim"])
                - float(current_metrics["damped_full_burg_per_dim"])
            ),
            "m_max": abs(float(replay_metrics["m_max"]) - float(current_metrics["m_max"])),
            "m_p50": abs(float(replay_metrics["m_p50"]) - float(current_metrics["m_p50"])),
            "spectrum_max_abs": float(
                (replay_eig.detach().double() - current_eig.detach().double()).abs().max().cpu()
            ),
        }
        (staging / "final_replay.json").write_text(
            json.dumps(replay_errors, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint = {
            "protocol_id": PROTOCOL_ID,
            "active_names": active_names,
            "active_model_state": {name: named[name].detach().cpu() for name in active_names},
            "final_parameter_hash": final_parameter_hash,
            "z_sha256": z_hash,
            "source_weight_index": SOURCE_WEIGHT_INDEX,
        }
        torch.save(checkpoint, staging / "final_checkpoint.pt")

        tail = state_frame.loc[state_frame["proposal"].between(TAIL_START, PROPOSALS)].copy()
        tail["core_noncollapse_pass"] = tail.apply(_tail_state_pass, axis=1)
        tail.to_csv(staging / "tail_noncollapse_audit.csv", index=False)
        accepted_transitions = state_frame.loc[state_frame["proposal"].gt(0) & state_frame["accepted"].eq(1)]
        strict_decreases = int(
            (state_frame["exact_a_per_dim"].diff().iloc[1:] < -float(tolerances["a"])).sum()
        )
        pair_keys = set(
            zip(
                pair_frame["proposal"].astype(int),
                pair_frame["draw"].astype(int),
                pair_frame["pair"].astype(int),
                strict=True,
            )
        )
        expected_pair_keys = {
            (proposal, draw, pair)
            for proposal in range(1, PROPOSALS + 1)
            for draw in range(POOL_DRAWS)
            for pair in range(PAIRS_PER_DRAW)
        }
        branch_seeds = pd.concat(
            [pair_frame["seed_1"], pair_frame["seed_2"]], ignore_index=True
        )
        cuda_rng_unchanged = bool(torch.equal(initial_cuda_rng, torch.cuda.get_rng_state(device)))
        cpu_rng_changed = not bool(torch.equal(initial_cpu_rng, torch.random.get_rng_state()))
        line_numeric_valid = len(line_frame) > 0 and _numeric_finite(line_frame.drop(columns=["failures"]))
        validity_gates = {
            "checkpoint_matches": sha256_file(checkpoint_path) == EXPECTED_CHECKPOINT,
            "z_matches": z_hash == EXPECTED_Z_SHA256,
            "initial_parameters_match": initial_parameter_hash == EXPECTED_INITIAL_PARAMETER_SHA256,
            "active_parameter_count_matches": active_parameter_count == 11_685_120,
            "full_ce": full_ce,
            "frozen_dependencies_match": all(frozen_matches.values()),
            "seed_schedule_valid": bool(seed_preflight["valid"]),
            "state0_repeatable": max(base_noise.values()) <= 1e-10,
            "exact_a_direct_closure": bool(state_frame["a_direct_abs_error"].le(1e-9).all()),
            "exact_a_trace_closure": bool(state_frame["a_trace_abs_error"].le(1e-9).all()),
            "line_exact_a_direct_closure": bool(line_frame["a_direct_abs_error"].le(1e-9).all()),
            "line_exact_a_trace_closure": bool(line_frame["a_trace_abs_error"].le(1e-9).all()),
            "state_rows_101": len(state_frame) == 101,
            "proposal_rows_100": len(proposal_frame) == 100,
            "pair_rows_3200": len(pair_frame) == 3200,
            "draw_rows_800": len(draw_frame) == 800,
            "actual_pair_key_coverage": pair_keys == expected_pair_keys,
            "actual_branch_seeds_unique": (
                len(branch_seeds) == 6400 and branch_seeds.nunique() == 6400
            ),
            "spectrum_rows_51712": len(spectrum_frame) == 101 * 512,
            "tables_numeric_finite": bool(
                _numeric_finite(state_frame.drop(columns=["rho_lower90"]).copy())
                and _numeric_finite(proposal_frame.drop(columns=["failure_counts"]))
                and line_numeric_valid
                and _numeric_finite(pair_frame)
                and _numeric_finite(draw_frame)
                and _numeric_finite(spectrum_frame)
            ),
            "all_component_sources_valid": bool(proposal_frame["source_valid"].eq(1).all()),
            "cancellation_rejections_are_counted_noops": bool(
                (
                    proposal_frame.loc[
                        proposal_frame["cancellation_gate_pass"].eq(0),
                        ["accepted", "attempted_endpoint_count"],
                    ]
                    .eq(0)
                    .all()
                ).all()
            ),
            "rejections_are_exact_noops": bool(
                (
                    proposal_frame.loc[proposal_frame["accepted"].eq(0), "base_parameter_hash"]
                    == proposal_frame.loc[proposal_frame["accepted"].eq(0), "final_parameter_hash"]
                ).all()
            ),
            "accepted_endpoints_pass_all_guards": bool(
                line_frame.loc[line_frame["passes"].eq(1), "failures"].eq("").all()
                and len(line_frame.loc[line_frame["passes"].eq(1)]) == accepted_count
            ),
            "final_replay_matches": max(replay_errors.values()) <= 1e-10,
            "final_parameter_hash_unchanged_by_replay": (
                _named_tensor_hash(active_names, active) == final_parameter_hash
            ),
            "cuda_rng_unchanged": cuda_rng_unchanged,
        }
        success_gates = {
            "final_exact_a_at_most_0p90": float(current_metrics["exact_a_per_dim"]) <= 0.90,
            "all_tail_exact_a_at_most_1": bool(tail["exact_a_per_dim"].le(1.0).all()),
            "accepted_proposals_at_least_20": accepted_count >= 20,
            "tail_accepted_proposals_at_least_4": int(tail["accepted"].sum()) >= 4,
            "cumulative_path_radius_at_least_0p10": cumulative_radius >= 0.10,
            "final_full_b_below_initial": (
                float(current_metrics["damped_full_burg_per_dim"])
                < float(initial_metrics["damped_full_burg_per_dim"])
            ),
            "strict_a_decreases_at_least_30": strict_decreases >= 30,
            "tail_core_noncollapse_at_least_16_of_20": int(tail["core_noncollapse_pass"].sum()) >= TAIL_REQUIRED,
            "accepted_steps_respect_spectral_guards": bool(
                (
                    accepted_transitions["m_max"].to_numpy()
                    <= state_frame.loc[accepted_transitions.index - 1, "m_max"].to_numpy()
                    + tolerances["m_max"]
                ).all()
                and (
                    accepted_transitions["m_p50"].to_numpy()
                    >= state_frame.loc[accepted_transitions.index - 1, "m_p50"].to_numpy()
                    - tolerances["m_p50"]
                ).all()
                and (
                    accepted_transitions["m_lt_0p1_fraction"].to_numpy()
                    <= state_frame.loc[accepted_transitions.index - 1, "m_lt_0p1_fraction"].to_numpy()
                    + tolerances["m_lt_0p1_fraction"]
                ).all()
            ),
        }
        valid = bool(all(validity_gates.values()))
        success = bool(valid and all(success_gates.values()))
        decision = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "success": success,
            "hypothesis_result": (
                "supported_under_exact_feasibility_controller"
                if success
                else "not_supported_at_frozen_success_thresholds"
            )
            if valid
            else None,
            "initial_metrics": initial_metrics,
            "final_metrics": current_metrics,
            "accepted_count": accepted_count,
            "strict_a_decreases": strict_decreases,
            "cumulative_path_radius": cumulative_radius,
            "tail_core_noncollapse_pass_count": int(tail["core_noncollapse_pass"].sum()),
            "base_noise": base_noise,
            "acceptance_tolerances": tolerances,
            "validity_gates": validity_gates,
            "success_gates": success_gates,
            "cpu_rng_changed_by_known_task_construction": cpu_rng_changed,
            "elapsed_sec": time.perf_counter() - started,
            "artifact_scope_note": (
                "Exact dense metrics are an acceptance oracle; success is an existence/repair result, "
                "not a stochastic-only deployment claim."
            ),
        }
        (staging / "decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _plot_artifacts(staging)
        manifest = _artifact_manifest(staging)
        (staging / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        finalized = {
            "protocol_id": PROTOCOL_ID,
            "status": "complete_awaiting_independent_review",
            "valid": valid,
            "success": success,
            "decision_sha256": sha256_file(staging / "decision.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        (staging / "INCOMPLETE").unlink()
        (staging / "FINALIZED.json").write_text(
            json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[exact-a-i3] complete valid={valid} success={success} accepted={accepted_count}/100 "
            f"A={initial_metrics['exact_a_per_dim']:.8g}->{current_metrics['exact_a_per_dim']:.8g} "
            f"B={initial_metrics['damped_full_burg_per_dim']:.8g}->"
            f"{current_metrics['damped_full_burg_per_dim']:.8g} "
            f"tail_pass={int(tail['core_noncollapse_pass'].sum())}/20 elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        os.replace(staging, final_output)
        print(f"[exact-a-i3] published output={final_output}", flush=True)
    except Exception:
        print("[exact-a-i3] FAILED; staging retained", flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
