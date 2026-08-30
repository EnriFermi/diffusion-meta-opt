from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import encode_weights
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices,
    _latent_hvp,
    _task_set_for_record,
)
from scripts import audit_one_state_exact_a_proposal3_cross as i1
from scripts import run_one_state_a_full_burg_p32_training as p32
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _add_,
    _cosine,
    _dot,
    _full_basis_gradient,
    _h_space_fd_check,
    _named_tensor_hash,
    _negative_normalized,
    _norm,
    _relative_error,
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
PROTOCOL_ID = "one_state_exact_a_realized_path_i2_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration2_realized_path_production"
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_2_realized_path/protocol.md"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_2_frozen_dependency_manifest.json"
EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = (
    "a3cadd2ab7ecdcd9bde6e32ea02d021ee68ebc635311fcab431a28f1530dbb54"
)
ITERATION5 = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_full_burg_h2048/iteration5_p32_training_production"
)
ITERATION5_STATES = ITERATION5 / "state_objective_curve.csv"
ITERATION5_PROPOSALS = ITERATION5 / "proposal_diagnostics.csv"
ITERATION1 = OUTPUT_ROOT / "iteration1_proposal3_cross_production"
OLD_BETA = 22.536727828943093
EPSILON = 1e-4
TARGET_NORM = 0.04892722657548397
BLOCK_SIZE = 64
FD_RADII = (1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0, 1.0 / 512.0)
COMMON_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
DIRECTION_NAMES = ("exact_a_raw", "exact_oldbeta_raw", "exact_unit_common_raw")
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_INITIAL_PARAMETER_SHA256 = "1cf3bdeb383c64b36b3ca69a956457833d0aa0d8dd1f6696e3964033518053d4"

Vector = list[torch.Tensor]


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_frozen_dependencies() -> dict[str, bool]:
    if sha256_file(FROZEN_DEPENDENCY_MANIFEST) != EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256:
        raise RuntimeError("frozen dependency manifest hash mismatch")
    expected = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    matches = {
        relative: (ROOT / relative).is_file() and sha256_file(ROOT / relative) == digest
        for relative, digest in expected.items()
    }
    if not all(matches.values()):
        raise RuntimeError(
            "frozen dependency mismatch: "
            + repr([relative for relative, matches_expected in matches.items() if not matches_expected])
        )
    return matches


def _blocked_basis_gradient(
    *,
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    active: Sequence[torch.nn.Parameter],
    k_cotangent: torch.Tensor,
    block_size: int,
) -> tuple[Vector, dict[str, Any]]:
    task_set = _task_set_for_record(run.task_tensors, record)
    tau = float(record.get("tau", 1.0))
    dim = int(z.numel())
    accumulated = _zeros_like(active, dtype=torch.float64)
    unused_counts = np.zeros(len(active), dtype=np.int64)
    memory_trace: list[int] = []
    started = time.perf_counter()
    for start in range(0, dim, block_size):
        stop = min(start + block_size, dim)
        scalars: list[torch.Tensor] = []
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
            scalars.append(torch.dot(k_cotangent[index].to(dtype=hvp.dtype), hvp))
        total = torch.stack(scalars).sum()
        gradients = torch.autograd.grad(total, active, retain_graph=False, allow_unused=True)
        for parameter_index, value in enumerate(gradients):
            if value is None:
                unused_counts[parameter_index] += 1
        _add_(accumulated, gradients)
        memory_trace.append(int(torch.cuda.memory_allocated(z.device)))
        print(
            f"[exact-a-i2] stage=blocked-vjp rows={stop}/{dim} "
            f"blocks={(stop + block_size - 1) // block_size}/{math.ceil(dim / block_size)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del scalars, total, gradients
    early = float(np.median(memory_trace[: min(2, len(memory_trace))]))
    late = float(np.median(memory_trace[-min(2, len(memory_trace)) :]))
    return accumulated, {
        "basis_count": dim,
        "block_size": block_size,
        "block_count": math.ceil(dim / block_size),
        "unused_parameter_tensors_all_blocks": int(
            (unused_counts == math.ceil(dim / block_size)).sum()
        ),
        "elapsed_sec": time.perf_counter() - started,
        "memory_early_median_bytes": early,
        "memory_late_median_bytes": late,
        "memory_growth_bytes": max(0.0, late - early),
        "memory_growth_gate_pass": bool(late - early <= 64 * 1024 * 1024),
    }


def _vector_sum(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    *,
    left_scale: float,
    right_scale: float,
    active: Sequence[torch.nn.Parameter],
) -> Vector:
    return [
        (
            left_value.detach().to(device=parameter.device, dtype=parameter.dtype) * left_scale
            + right_value.detach().to(device=parameter.device, dtype=parameter.dtype) * right_scale
        ).clone()
        for left_value, right_value, parameter in zip(left, right, active, strict=True)
    ]


def _unit_common(
    gradient_a: Sequence[torch.Tensor],
    gradient_b: Sequence[torch.Tensor],
    active: Sequence[torch.nn.Parameter],
) -> Vector:
    return _vector_sum(
        gradient_a,
        gradient_b,
        left_scale=1.0 / _norm(gradient_a),
        right_scale=1.0 / _norm(gradient_b),
        active=active,
    )


def _scalarization_stats(
    gradient_a: Sequence[torch.Tensor],
    gradient_b: Sequence[torch.Tensor],
) -> dict[str, float | bool | None]:
    norm_a = _norm(gradient_a)
    norm_b = _norm(gradient_b)
    dot_ab = _dot(gradient_a, gradient_b)
    s_value = norm_a * norm_a + OLD_BETA * dot_ab
    beta_crit = None if dot_ab >= 0.0 else -(norm_a * norm_a) / dot_ab
    return {
        "norm_a": norm_a,
        "norm_b": norm_b,
        "dot_ab": dot_ab,
        "cosine_ab": dot_ab / max(norm_a * norm_b, 1e-30),
        "s_value": s_value,
        "beta_crit": beta_crit,
        "conflict": bool(dot_ab < 0.0 and beta_crit is not None and OLD_BETA > beta_crit and s_value < 0.0),
    }


def _realized_direction_metrics(
    *,
    active_names: Sequence[str],
    active: Sequence[torch.nn.Parameter],
    base: Sequence[torch.Tensor],
    plus: Sequence[torch.Tensor],
    minus: Sequence[torch.Tensor],
    nominal: Sequence[torch.Tensor],
    gradient_a: Sequence[torch.Tensor],
    gradient_f: Sequence[torch.Tensor],
    radius: float,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    dot_nominal = 0.0
    nominal2 = 0.0
    effective2 = 0.0
    midpoint2 = 0.0
    slope_theta = 0.0
    slope_theta_f = 0.0
    block_rows: list[dict[str, float | str]] = []
    for name, parameter, origin, positive, negative, direction, grad_a, grad_f in zip(
        active_names,
        active,
        base,
        plus,
        minus,
        nominal,
        gradient_a,
        gradient_f,
        strict=True,
    ):
        effective = (positive.detach().double() - negative.detach().double()) / (2.0 * radius)
        midpoint = 0.5 * (positive.detach().double() + negative.detach().double()) - origin.detach().double()
        nominal64 = direction.detach().double()
        dot_nominal += float((effective * nominal64).sum().cpu())
        nominal2 += float(nominal64.square().sum().cpu())
        effective2 += float(effective.square().sum().cpu())
        midpoint2 += float(midpoint.square().sum().cpu())
        slope_theta += float((grad_a.detach().double() * effective).sum().cpu())
        slope_theta_f += float((grad_f.detach().double() * effective).sum().cpu())
        block_rows.append(
            {
                "parameter": name,
                "nominal_norm": float(nominal64.norm().cpu()),
                "effective_norm": float(effective.norm().cpu()),
                "effective_minus_nominal_norm": float((effective - nominal64).norm().cpu()),
                "midpoint_drift_norm": float(midpoint.norm().cpu()),
            }
        )
    nominal_norm = math.sqrt(nominal2)
    effective_norm = math.sqrt(effective2)
    return {
        "nominal_direction_norm": nominal_norm,
        "effective_direction_norm": effective_norm,
        "effective_to_nominal_norm_ratio": effective_norm / max(nominal_norm, 1e-30),
        "effective_to_nominal_cosine": dot_nominal / max(nominal_norm * effective_norm, 1e-30),
        "midpoint_drift_norm": math.sqrt(midpoint2),
        "midpoint_drift_relative_to_step": math.sqrt(midpoint2)
        / max(radius * nominal_norm, 1e-30),
        "slope_theta": slope_theta,
        "slope_theta_f": slope_theta_f,
    }, block_rows


def _plot(output: Path, realized: pd.DataFrame, common: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    old = realized.loc[realized["direction"].eq("exact_oldbeta_raw")].sort_values("radius")
    for column, label in (
        ("nominal_a_slope", "nominal gA dot d"),
        ("slope_theta", "realized parameter path"),
        ("slope_hessian", "Hessian secant"),
        ("slope_a_direct", "exact A secant"),
    ):
        axes[0].plot(old["radius"], old[column], marker="o", label=label)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("symmetric radius h")
    axes[0].set_ylabel("A directional slope")
    axes[0].set_title("Exact old-beta realized-path decomposition")
    axes[0].legend(fontsize=8)

    ordered = common.sort_values("alpha")
    axes[1].plot(ordered["alpha"], ordered["delta_a"], marker="o", label="delta exact A")
    axes[1].plot(ordered["alpha"], ordered["delta_b"], marker="o", label="delta exact B")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("change from base")
    axes[1].set_title("Exact unit-common line")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "realized_path_and_common_line.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise ValueError("frozen protocol requires cuda:0")
    frozen_dependency_matches = _validate_frozen_dependencies()
    final_output = args.output_dir.resolve()
    staging = Path(str(final_output) + ".incomplete")
    if final_output.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {final_output} or {staging}")
    staging.mkdir(parents=True)
    (staging / "INCOMPLETE").write_text(PROTOCOL_ID + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), staging / "executed_source_snapshot.py")
    shutil.copy2(
        FROZEN_DEPENDENCY_MANIFEST,
        staging / "frozen_dependency_manifest_snapshot.json",
    )
    started = time.perf_counter()
    device = torch.device(args.device)
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "device": args.device,
        "dtype": "float32 model / float64 accumulators",
        "state_position": 2,
        "source_weight_index": 378,
        "beta": OLD_BETA,
        "burg_epsilon": EPSILON,
        "target_norm": TARGET_NORM,
        "block_size": BLOCK_SIZE,
        "fd_radii": list(FD_RADII),
        "common_alphas": list(COMMON_ALPHAS),
        "output_dir": str(final_output),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "frozen_dependency_manifest_sha256": EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256,
        "frozen_dependency_matches": frozen_dependency_matches,
        "source_sha256": _source_sha256(),
        "iteration1_decision_sha256": sha256_file(ITERATION1 / "decision.json"),
        "iteration5_states_sha256": sha256_file(ITERATION5_STATES),
        "iteration5_proposals_sha256": sha256_file(ITERATION5_PROPOSALS),
    }
    (staging / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[exact-a-i2] start {json.dumps(resolved, sort_keys=True)}", flush=True)

    try:
        if sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt") != EXPECTED_CHECKPOINT:
            raise RuntimeError("checkpoint mismatch")
        run = _load_run(DEFAULT_RUN_DIR, device=device)
        cfg = replace(
            _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384),
            vae_precond_hvp_mode="autograd",
        )
        bank = pd.read_csv(STATE_BANK)
        selected = bank.loc[bank["state_position"].eq(2)]
        if len(selected) != 1 or int(selected.iloc[0]["source_weight_index"]) != 378:
            raise RuntimeError("state-bank identity mismatch")
        source_index = 378
        record = run.records.iloc[source_index].to_dict()
        record["source_weight_index"] = source_index
        weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
        with torch.no_grad():
            z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
        task_set = _task_set_for_record(run.task_tensors, record)
        full_ce = all(
            _batch_indices(
                task_set,
                batch_size=int(cfg.vae_precond_batch_size),
                step=10,
                sample_key=source_index,
                pair_key=pair_key,
            )
            is None
            for pair_key in range(8)
        )
        active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
        named = dict(run.vae.named_parameters())
        for name, parameter in named.items():
            parameter.requires_grad_(name in active_names)
        active = [named[name] for name in active_names]
        initial_hash = _named_tensor_hash(active_names, active)
        if sha256_tensor(z) != EXPECTED_Z_SHA256 or initial_hash != EXPECTED_INITIAL_PARAMETER_SHA256:
            raise RuntimeError("initial fingerprint mismatch")

        p32.DRAW_ROWS.clear()
        p32.PAIR_ROWS.clear()
        p32.POOLING_PREFLIGHT.clear()
        p32.TREATMENT_BUILD_CALLS = 0
        p32.TREATMENT_CLIP_CALLS = 0
        p32.TREATMENT_ADAM_CALLS = 0
        exp_avg = [torch.zeros_like(parameter) for parameter in active]
        exp_avg_sq = [torch.zeros_like(parameter) for parameter in active]
        accepted_step = 0
        frozen_states = pd.read_csv(ITERATION5_STATES)
        frozen_proposals = pd.read_csv(ITERATION5_PROPOSALS)
        replay_rows: list[dict[str, float | int]] = []
        for proposal in (1, 2):
            print(f"[exact-a-i2] stage=replay proposal={proposal}/2", flush=True)
            current, _h, _m, _eig, burg_gradient = i1._evaluate_dense(
                run=run,
                z=z,
                record=record,
                epsilon=EPSILON,
                hessian_chunk_size=64,
            )
            del _h, _m, _eig
            treatment = p32._build_p32_proposal(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                active=active,
                burg_gradient=burg_gradient,
                beta=OLD_BETA,
                proposal=proposal,
                pairs=4,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                accepted_adam_step=accepted_step,
                gradient_clip=1.0,
                lr=3e-5,
            )
            expected = frozen_proposals.loc[frozen_proposals["proposal"].eq(proposal)].iloc[0]
            base_parameters = [parameter.detach().clone() for parameter in active]
            _set_parameters(active, base_parameters, treatment["displacement"], 1.0)
            exp_avg = treatment["next_avg"]
            exp_avg_sq = treatment["next_avg_sq"]
            accepted_step += 1
            after, _h, _m, _eig, _bg = i1._evaluate_dense(
                run=run,
                z=z,
                record=record,
                epsilon=EPSILON,
                hessian_chunk_size=64,
            )
            del _h, _m, _eig, _bg, burg_gradient
            frozen = frozen_states.loc[frozen_states["proposal"].eq(proposal)].iloc[0]
            replay_rows.append(
                {
                    "proposal": proposal,
                    "a_abs_error": abs(float(after["exact_a_per_dim"]) - float(frozen["exact_a_per_dim"])),
                    "b_abs_error": abs(
                        float(after["damped_full_burg_per_dim"])
                        - float(frozen["damped_full_burg_per_dim"])
                    ),
                    "proposal_norm_abs_error": abs(
                        float(treatment["proposal_norm"]) - float(expected["proposal_norm"])
                    ),
                    "grad_a_relative_error": abs(
                        float(treatment["component_stats"]["grad_a_norm"])
                        - float(expected["grad_a_norm"])
                    )
                    / max(abs(float(expected["grad_a_norm"])), 1e-30),
                    "grad_b_relative_error": abs(
                        float(treatment["component_stats"]["grad_b_norm"])
                        - float(expected["grad_b_norm"])
                    )
                    / max(abs(float(expected["grad_b_norm"])), 1e-30),
                }
            )
            del current, after, treatment, base_parameters
        replay = pd.DataFrame(replay_rows)
        replay.to_csv(staging / "reconstruction.csv", index=False)
        if (
            replay.filter(like="_abs_error").to_numpy().max() > 1e-5
            or replay.filter(like="_relative_error").to_numpy().max() > 1e-6
            or accepted_step != 2
        ):
            raise RuntimeError("replay preflight failed")

        base_metrics, hessian, matrix, _eig, burg_gradient = i1._evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=64,
        )
        repeat_metrics, repeat_hessian, _rm, _reig, _rburg = i1._evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=64,
        )
        base = [parameter.detach().clone() for parameter in active]
        base_hash = _named_tensor_hash(active_names, active)
        base_noise = {
            key: abs(float(base_metrics[column]) - float(repeat_metrics[column]))
            for key, column in {
                "a": "exact_a_per_dim",
                "b": "damped_full_burg_per_dim",
                "m_max": "m_max",
                "m_p50": "m_p50",
                "m_lt_0p1_fraction": "m_lt_0p1_fraction",
            }.items()
        }
        pd.DataFrame(
            [
                {"evaluation": "primary", **base_metrics},
                {"evaluation": "repeat", **repeat_metrics},
            ]
        ).to_csv(staging / "base_repeatability.csv", index=False)
        if max(base_noise.values()) > 1e-6 or sha256_tensor(hessian) != sha256_tensor(repeat_hessian):
            raise RuntimeError("base dense repeatability failed")
        del repeat_hessian, _rm, _reig, _rburg

        dim = int(z.numel())
        identity = torch.eye(dim, device=device, dtype=torch.float64)
        h64 = hessian.double()
        m64 = matrix.double()
        k_a = (4.0 * (m64 - identity) @ h64 / float(dim)).detach()
        g_b_matrix = identity / (float(dim) * (1.0 + EPSILON)) - torch.linalg.inv(
            m64 + EPSILON * identity
        ) / float(dim)
        k_b = (2.0 * g_b_matrix @ h64).detach()
        k_old = (k_a + OLD_BETA * k_b).detach()
        h_space_rows = [
            {"objective": objective, **row}
            for objective, rows in (
                ("a", _h_space_fd_check(hessian, k_a, beta=0.0, epsilon=EPSILON)),
                ("oldbeta", _h_space_fd_check(hessian, k_old, beta=OLD_BETA, epsilon=EPSILON)),
            )
            for row in rows
        ]
        pd.DataFrame(h_space_rows).to_csv(staging / "h_space_fd.csv", index=False)
        if max(float(row["relative_error"]) for row in h_space_rows) > 1e-5:
            raise RuntimeError("H-space cotangent preflight failed")

        sequential: dict[str, Vector] = {}
        sequential_meta: dict[str, Any] = {}
        blocked: dict[str, Vector] = {}
        blocked_meta: dict[str, Any] = {}
        for component, cotangent in (("a", k_a), ("b", k_b)):
            print(f"[exact-a-i2] stage=sequential component={component}", flush=True)
            sequential[component], sequential_meta[component] = _full_basis_gradient(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                active=active,
                k_cotangent=cotangent,
                mode="autograd",
                basis_count=dim,
            )
            print(f"[exact-a-i2] stage=blocked component={component}", flush=True)
            blocked[component], blocked_meta[component] = _blocked_basis_gradient(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                active=active,
                k_cotangent=cotangent,
                block_size=BLOCK_SIZE,
            )
        sequential_old = _vector_sum(
            sequential["a"], sequential["b"], left_scale=1.0, right_scale=OLD_BETA, active=active
        )
        blocked_old = _vector_sum(
            blocked["a"], blocked["b"], left_scale=1.0, right_scale=OLD_BETA, active=active
        )
        gradient_comparison = {
            component: {
                "relative_error": _relative_error(blocked_gradient, sequential_gradient),
                "cosine": _cosine(blocked_gradient, sequential_gradient),
            }
            for component, blocked_gradient, sequential_gradient in (
                ("a", blocked["a"], sequential["a"]),
                ("b", blocked["b"], sequential["b"]),
                ("oldbeta", blocked_old, sequential_old),
            )
        }
        pd.DataFrame(
            [{"component": component, **metrics} for component, metrics in gradient_comparison.items()]
        ).to_csv(staging / "gradient_method_comparison.csv", index=False)
        if any(
            metrics["relative_error"] > 1e-4 or metrics["cosine"] < 0.999999
            for metrics in gradient_comparison.values()
        ):
            raise RuntimeError("sequential/blocked pullback mismatch")

        scalarization_blocked = _scalarization_stats(blocked["a"], blocked["b"])
        scalarization_sequential = _scalarization_stats(sequential["a"], sequential["b"])
        s_disagreement = abs(
            float(scalarization_blocked["s_value"])
            - float(scalarization_sequential["s_value"])
        )
        scalarization_margin_pass = bool(
            scalarization_blocked["conflict"]
            and scalarization_sequential["conflict"]
            and min(
                -float(scalarization_blocked["s_value"]),
                -float(scalarization_sequential["s_value"]),
            )
            > 5.0 * s_disagreement
        )

        exact_common = _unit_common(blocked["a"], blocked["b"], active)
        exact_common_source_norm = _norm(exact_common)
        directions = {
            "exact_a_raw": _negative_normalized(blocked["a"], target_norm=TARGET_NORM, active=active),
            "exact_oldbeta_raw": _negative_normalized(blocked_old, target_norm=TARGET_NORM, active=active),
            "exact_unit_common_raw": _negative_normalized(
                exact_common, target_norm=TARGET_NORM, active=active
            ),
        }
        cuda_rng_hash = sha256_tensor(torch.cuda.get_rng_state(device))
        cpu_rng_hash = sha256_tensor(torch.get_rng_state())
        realized_rows: list[dict[str, Any]] = []
        endpoint_rows: list[dict[str, Any]] = []
        block_rows: list[dict[str, Any]] = []
        hessian_archive: dict[str, np.ndarray] = {}
        for direction_name, direction in directions.items():
            nominal_a_slope = _dot(blocked["a"], direction)
            nominal_b_slope = _dot(blocked["b"], direction)
            nominal_f_slope = _dot(blocked_old, direction)
            cauchy_scale = _norm(blocked["a"]) * _norm(direction)
            cauchy_scale_f = _norm(blocked_old) * _norm(direction)
            for radius in FD_RADII:
                endpoints: dict[str, tuple[dict[str, float], torch.Tensor, list[torch.Tensor], str]] = {}
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    _set_parameters(active, base, direction, sign * radius)
                    parameter_values = [parameter.detach().clone() for parameter in active]
                    parameter_hash = _named_tensor_hash(active_names, active)
                    metrics, endpoint_hessian, _matrix, _endpoint_eig, _endpoint_burg = i1._evaluate_dense(
                        run=run,
                        z=z,
                        record=record,
                        epsilon=EPSILON,
                        hessian_chunk_size=64,
                    )
                    repeat, repeat_h, _repeat_m, _repeat_eig, _repeat_burg = i1._evaluate_dense(
                        run=run,
                        z=z,
                        record=record,
                        epsilon=EPSILON,
                        hessian_chunk_size=64,
                    )
                    repeat_matches = bool(
                        sha256_tensor(endpoint_hessian) == sha256_tensor(repeat_h)
                        and abs(metrics["exact_a_per_dim"] - repeat["exact_a_per_dim"]) <= 1e-10
                        and abs(metrics["damped_full_burg_per_dim"] - repeat["damped_full_burg_per_dim"])
                        <= 1e-10
                    )
                    key = f"{direction_name}_h{int(round(1.0 / radius))}_{sign_name}"
                    hessian_archive[key] = endpoint_hessian.detach().cpu().numpy()
                    endpoint_rows.append(
                        {
                            "direction": direction_name,
                            "radius": radius,
                            "sign": sign_name,
                            "parameter_hash": parameter_hash,
                            "hessian_hash": sha256_tensor(endpoint_hessian),
                            "repeat_hessian_hash": sha256_tensor(repeat_h),
                            "repeat_matches": repeat_matches,
                            "repeat_a_abs_error": abs(
                                metrics["exact_a_per_dim"] - repeat["exact_a_per_dim"]
                            ),
                            "repeat_b_abs_error": abs(
                                metrics["damped_full_burg_per_dim"]
                                - repeat["damped_full_burg_per_dim"]
                            ),
                            **metrics,
                        }
                    )
                    endpoints[sign_name] = (metrics, endpoint_hessian, parameter_values, parameter_hash)
                    del _matrix, _endpoint_eig, _endpoint_burg, repeat_h, _repeat_m, _repeat_eig, _repeat_burg
                _set_parameters(active, base, direction, 0.0)
                plus_metrics, plus_hessian, plus_parameters, _plus_hash = endpoints["plus"]
                minus_metrics, minus_hessian, minus_parameters, _minus_hash = endpoints["minus"]
                realized, per_block = _realized_direction_metrics(
                    active_names=active_names,
                    active=active,
                    base=base,
                    plus=plus_parameters,
                    minus=minus_parameters,
                    nominal=direction,
                    gradient_a=blocked["a"],
                    gradient_f=blocked_old,
                    radius=radius,
                )
                slope_hessian = float(
                    (
                        k_a
                        * (plus_hessian.double() - minus_hessian.double())
                        / (2.0 * radius)
                    ).sum().cpu()
                )
                slope_a_direct = (
                    float(plus_metrics["a_direct_matrix"])
                    - float(minus_metrics["a_direct_matrix"])
                ) / (2.0 * radius)
                slope_b = (
                    float(plus_metrics["damped_full_burg_per_dim"])
                    - float(minus_metrics["damped_full_burg_per_dim"])
                ) / (2.0 * radius)
                slope_hessian_f = float(
                    (
                        k_old
                        * (plus_hessian.double() - minus_hessian.double())
                        / (2.0 * radius)
                    ).sum().cpu()
                )
                slope_f_direct = slope_a_direct + OLD_BETA * slope_b
                realized_rows.append(
                    {
                        "direction": direction_name,
                        "radius": radius,
                        "nominal_a_slope": nominal_a_slope,
                        "nominal_b_slope": nominal_b_slope,
                        "nominal_f_slope": nominal_f_slope,
                        "cauchy_scale_a": cauchy_scale,
                        "cauchy_scale_f": cauchy_scale_f,
                        **realized,
                        "slope_hessian": slope_hessian,
                        "slope_a_direct": slope_a_direct,
                        "slope_b": slope_b,
                        "slope_hessian_f": slope_hessian_f,
                        "slope_f_direct": slope_f_direct,
                        "nominal_to_theta_normalized_error": abs(
                            nominal_a_slope - realized["slope_theta"]
                        )
                        / max(cauchy_scale, 1e-30),
                        "theta_to_h_normalized_error": abs(realized["slope_theta"] - slope_hessian)
                        / max(cauchy_scale, 1e-30),
                        "h_to_a_normalized_error": abs(slope_hessian - slope_a_direct)
                        / max(cauchy_scale, 1e-30),
                        "theta_to_h_f_normalized_error": abs(
                            realized["slope_theta_f"] - slope_hessian_f
                        )
                        / max(cauchy_scale_f, 1e-30),
                        "nominal_to_theta_f_normalized_error": abs(
                            nominal_f_slope - realized["slope_theta_f"]
                        )
                        / max(cauchy_scale_f, 1e-30),
                        "h_to_f_normalized_error": abs(
                            slope_hessian_f - slope_f_direct
                        )
                        / max(cauchy_scale_f, 1e-30),
                    }
                )
                block_rows.extend(
                    {
                        "direction": direction_name,
                        "radius": radius,
                        **row,
                    }
                    for row in per_block
                )
                print(
                    f"[exact-a-i2] stage=realized direction={direction_name} h=1/{int(round(1/radius))} "
                    f"N={nominal_a_slope:.6g} theta={realized['slope_theta']:.6g} "
                    f"H={slope_hessian:.6g} D={slope_a_direct:.6g} F={slope_f_direct:.6g}",
                    flush=True,
                )

        realized_frame = pd.DataFrame(realized_rows)
        endpoint_frame = pd.DataFrame(endpoint_rows)
        realized_frame.to_csv(staging / "realized_path.csv", index=False)
        endpoint_frame.to_csv(staging / "endpoint_metrics.csv", index=False)
        pd.DataFrame(block_rows).to_csv(staging / "parameter_block_realization.csv", index=False)
        np.savez_compressed(staging / "hessian_endpoints.npz", **hessian_archive)

        common_rows: list[dict[str, Any]] = []
        common_direction = directions["exact_unit_common_raw"]
        for alpha in COMMON_ALPHAS:
            _set_parameters(active, base, common_direction, alpha)
            metrics, _h, _m, _eig, _burg = i1._evaluate_dense(
                run=run,
                z=z,
                record=record,
                epsilon=EPSILON,
                hessian_chunk_size=64,
            )
            common_rows.append(
                {
                    "alpha": alpha,
                    **metrics,
                    "delta_a": float(metrics["exact_a_per_dim"] - base_metrics["exact_a_per_dim"]),
                    "delta_b": float(
                        metrics["damped_full_burg_per_dim"]
                        - base_metrics["damped_full_burg_per_dim"]
                    ),
                    "delta_m_max": float(metrics["m_max"] - base_metrics["m_max"]),
                    "delta_m_p50": float(metrics["m_p50"] - base_metrics["m_p50"]),
                    "delta_m_lt_0p1_fraction": float(
                        metrics["m_lt_0p1_fraction"] - base_metrics["m_lt_0p1_fraction"]
                    ),
                }
            )
            del _h, _m, _eig, _burg
        _set_parameters(active, base, common_direction, 0.0)
        common_frame = pd.DataFrame(common_rows)
        common_frame.to_csv(staging / "exact_common_line.csv", index=False)

        tolerances = {
            key: max(5.0 * base_noise[key], 1e-8 if key in {"a", "b", "m_max"} else 1e-12)
            for key in base_noise
        }
        eligible = common_frame.loc[common_frame["alpha"].ge(0.125)].copy()
        eligible["passes"] = (
            eligible["delta_a"].lt(-tolerances["a"])
            & eligible["delta_b"].lt(-tolerances["b"])
            & eligible["delta_m_max"].le(tolerances["m_max"])
            & eligible["delta_m_p50"].ge(-tolerances["m_p50"])
            & eligible["delta_m_lt_0p1_fraction"].le(tolerances["m_lt_0p1_fraction"])
        )
        passing_common = eligible.loc[eligible["passes"]].sort_values("alpha", ascending=False)
        candidate = {
            "selected": bool(len(passing_common)),
            "largest_passing_alpha": None
            if passing_common.empty
            else float(passing_common.iloc[0]["alpha"]),
            "passing_count": int(len(passing_common)),
            "tolerances": tolerances,
        }

        old_coarse = realized_frame.loc[
            realized_frame["direction"].eq("exact_oldbeta_raw")
            & realized_frame["radius"].ge(1.0 / 256.0)
        ]
        realized_sign_pass = bool(
            len(old_coarse) == 3
            and (old_coarse[["nominal_a_slope", "slope_theta", "slope_hessian", "slope_a_direct"]] > 0.0)
            .all()
            .all()
            and (
                old_coarse[
                    ["nominal_f_slope", "slope_theta_f", "slope_hessian_f", "slope_f_direct"]
                ]
                < 0.0
            )
            .all()
            .all()
            and old_coarse["theta_to_h_normalized_error"].max() <= 0.01
            and old_coarse["nominal_to_theta_normalized_error"].max() <= 0.01
            and old_coarse["h_to_a_normalized_error"].max() <= 0.01
            and old_coarse["theta_to_h_f_normalized_error"].max() <= 0.01
            and old_coarse["nominal_to_theta_f_normalized_error"].max() <= 0.01
            and old_coarse["h_to_f_normalized_error"].max() <= 0.01
            and old_coarse["effective_to_nominal_cosine"].min() >= 0.95
            and old_coarse["effective_to_nominal_norm_ratio"].between(0.9, 1.1).all()
            and old_coarse["midpoint_drift_relative_to_step"].max() <= 0.25
        )
        common_nominal_pass = bool(
            _dot(blocked["a"], common_direction) < 0.0
            and _dot(blocked["b"], common_direction) < 0.0
            and _dot(sequential["a"], common_direction) < 0.0
            and _dot(sequential["b"], common_direction) < 0.0
        )
        final_hash = _named_tensor_hash(active_names, active)
        final_cuda_rng_hash = sha256_tensor(torch.cuda.get_rng_state(device))
        final_cpu_rng_hash = sha256_tensor(torch.get_rng_state())
        validity = {
            "checkpoint_matches": sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt") == EXPECTED_CHECKPOINT,
            "full_ce": full_ce,
            "replay_matches": bool(
                replay.filter(like="_abs_error").to_numpy().max() <= 1e-5
                and replay.filter(like="_relative_error").to_numpy().max() <= 1e-6
            ),
            "base_repeatable": max(base_noise.values()) <= 1e-6,
            "h_space_fd_passes": max(float(row["relative_error"]) for row in h_space_rows) <= 1e-5,
            "all_exact_basis_counts_512": all(
                int(meta["basis_count"]) == 512 for meta in sequential_meta.values()
            )
            and all(int(meta["basis_count"]) == 512 for meta in blocked_meta.values()),
            "no_fully_unused_tensors": all(
                int(meta["unused_parameter_tensors_all_rows"]) == 0
                for meta in sequential_meta.values()
            )
            and all(
                int(meta["unused_parameter_tensors_all_blocks"]) == 0
                for meta in blocked_meta.values()
            ),
            "memory_growth_passes": all(
                bool(meta["memory_growth_gate_pass"]) for meta in sequential_meta.values()
            )
            and all(bool(meta["memory_growth_gate_pass"]) for meta in blocked_meta.values()),
            "gradient_methods_agree": all(
                metrics["relative_error"] <= 1e-4 and metrics["cosine"] >= 0.999999
                for metrics in gradient_comparison.values()
            ),
            "scalarization_margin_passes": scalarization_margin_pass,
            "realized_exact_oldbeta_sign_passes": realized_sign_pass,
            "endpoint_repeats_match": bool(endpoint_frame["repeat_matches"].all()),
            "common_nominal_descends_both": common_nominal_pass,
            "common_cancellation_margin_passes": exact_common_source_norm >= 0.5,
            "parameters_restored": final_hash == base_hash,
            "cuda_rng_unchanged": final_cuda_rng_hash == cuda_rng_hash,
            "numeric_finite": bool(
                np.isfinite(realized_frame.select_dtypes(include=[np.number]).to_numpy()).all()
                and np.isfinite(common_frame.select_dtypes(include=[np.number]).to_numpy()).all()
            ),
        }
        valid = all(validity.values())
        decision = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "validity_gates": validity,
            "base_metrics": base_metrics,
            "base_noise": base_noise,
            "gradient_comparison": gradient_comparison,
            "scalarization_blocked": scalarization_blocked,
            "scalarization_sequential": scalarization_sequential,
            "s_disagreement": s_disagreement,
            "scalarization_conflict_confirmed": scalarization_margin_pass if valid else None,
            "realized_path_confirmed": realized_sign_pass if valid else None,
            "common_candidate": candidate if valid else None,
            "common_nominal_slopes": {
                "blocked_a": _dot(blocked["a"], common_direction),
                "blocked_b": _dot(blocked["b"], common_direction),
                "sequential_a": _dot(sequential["a"], common_direction),
                "sequential_b": _dot(sequential["b"], common_direction),
            },
            "common_source_norm": exact_common_source_norm,
            "deployment_rule": {
                "update_map": "raw_exact_unit_common_without_adam_or_clip",
                "alpha_rule": "largest_passing_alpha",
                "selected_alpha": candidate["largest_passing_alpha"] if valid else None,
            },
            "cpu_rng_changed_by_deterministic_task_construction": final_cpu_rng_hash != cpu_rng_hash,
            "elapsed_sec": time.perf_counter() - started,
            "source_sha256": _source_sha256(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
        }
        _plot(staging, realized_frame, common_frame)
        (staging / "decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        artifact_names = (
            "resolved_config.json",
            "executed_source_snapshot.py",
            "frozen_dependency_manifest_snapshot.json",
            "reconstruction.csv",
            "base_repeatability.csv",
            "h_space_fd.csv",
            "gradient_method_comparison.csv",
            "realized_path.csv",
            "endpoint_metrics.csv",
            "parameter_block_realization.csv",
            "hessian_endpoints.npz",
            "exact_common_line.csv",
            "realized_path_and_common_line.png",
            "decision.json",
        )
        manifest = {name: sha256_file(staging / name) for name in artifact_names}
        (staging / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        finalized = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "decision_sha256": sha256_file(staging / "decision.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        (staging / "FINALIZED.json").write_text(
            json.dumps(finalized, indent=2, sort_keys=True), encoding="utf-8"
        )
        (staging / "INCOMPLETE").unlink()
        staging.replace(final_output)
        print(f"[exact-a-i2] complete {json.dumps(decision, sort_keys=True)}", flush=True)
        print(f"[exact-a-i2] artifacts={final_output}", flush=True)
        if not valid:
            raise RuntimeError("Iteration-2 validity gates failed")
    except Exception:
        location = final_output if final_output.exists() else staging
        print(f"[exact-a-i2] failed artifacts={location}", flush=True)
        raise


if __name__ == "__main__":
    main()
