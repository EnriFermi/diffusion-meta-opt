from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

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
    _task_set_for_record,
)
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _adam_direction,
    _add_,
    _aggregated_basis_gradient,
    _cosine,
    _directional_fd,
    _forward_equivalence,
    _full_basis_gradient,
    _gradient_from_losses,
    _h_space_fd_check,
    _named_tensor_hash,
    _negative_normalized,
    _norm,
    _normalize_direction,
    _numeric_frame_is_finite,
    _relative_error,
    _scaled,
    _torch_adam_reference_direction,
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
from scripts.run_one_state_a_full_burg_armijo import (
    DEFAULT_BETA,
    _evaluate_candidate,
    _objective,
    _set_parameters,
)
from scripts.smoke_optimize_variant_a_full_burg_one_state import (
    ACTIVE_PARAMETERS,
    EXPECTED_CHECKPOINT,
    OUTPUT_ROOT,
    STATE_BANK,
    _materialize_metric,
    _pair_losses,
    _train_generators,
)


PROTOCOL_ID = "one_state_naive_probe_adam_cross_iteration4_v2"
PROTOCOL = OUTPUT_ROOT / "iteration_4_probe_adam_cross_protocol.md"
PRODUCTION = OUTPUT_ROOT / "iteration3_naive_semantics_production"
REPLAY = OUTPUT_ROOT / "iteration4_replay_through58"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration4_probe_adam_cross_production_v2"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_4_frozen_dependency_manifest.json"
EXPECTED_FROZEN_MANIFEST_SHA256 = "2ff62980d551780daa36a41e02e5a687c149c557905d038f35a370bccdc82329"
EXPECTED_NORMALIZED_SOURCE_SHA256 = "cfd9a42b4e53302b8d7fa0822bc5a4c0cd6087702c0afe758a787b5981d8ffb6"
EXPECTED_RUNNER_SHA256 = "ad5ae8bf4872c21a5bd37d636d0ad2708eab20962b94e42fd615fd05496a2bed"
EXPECTED_Z_SHA256 = "c5e1d96e45a2bbf6fabe3d9bb7e090a75f3b235ece87f6fd07a05f005cb725bc"
EXPECTED_STATE_F = 131.50894570753093
EXPECTED_ADAM_STEP = 54
REPLAY_SEED_KEYS = (59, 67)
BLIND_SEED_KEYS = tuple(range(1001, 1017))
ALL_P4_SEED_KEYS = REPLAY_SEED_KEYS + BLIND_SEED_KEYS
P32_BANKS = {"p32_a": tuple(range(1001, 1009)), "p32_b": tuple(range(1009, 1017))}
EXTENDED_ALPHAS = tuple(2.0**-power for power in range(13))
F_TOLERANCE = max(1e-4, 1e-6 * abs(EXPECTED_STATE_F))
SLOPE_TOLERANCES = {1.0 / 128.0: F_TOLERANCE * 128.0, 1.0 / 256.0: F_TOLERANCE * 256.0}


Vector = list[torch.Tensor]


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalized_source_sha256() -> str:
    masked_prefixes = (
        "EXPECTED_FROZEN_MANIFEST_SHA256 = ",
        "EXPECTED_NORMALIZED_SOURCE_SHA256 = ",
    )
    lines = Path(__file__).read_text(encoding="utf-8").splitlines(keepends=True)
    normalized: list[str] = []
    for line in lines:
        prefix = next((candidate for candidate in masked_prefixes if line.startswith(candidate)), None)
        normalized.append(f'{prefix}"<FROZEN>"\n' if prefix is not None else line)
    return hashlib.sha256("".join(normalized).encode("utf-8")).hexdigest()


def _average(vectors: Sequence[Sequence[torch.Tensor]], active: Sequence[torch.nn.Parameter]) -> Vector:
    if not vectors:
        raise ValueError("cannot average an empty vector bank")
    total = _zeros_like(active)
    for vector in vectors:
        _add_(total, vector)
    return _scaled(total, 1.0 / float(len(vectors)), active)


def _direct_pooled_gradient(
    *,
    seed_keys: Sequence[int],
    cfg: Any,
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active: Sequence[torch.nn.Parameter],
    burg_gradient: torch.Tensor,
    beta: float,
) -> Vector:
    accumulated = _zeros_like(active)
    count = 0
    for seed_key in seed_keys:
        for pair in range(4):
            generator_1, generator_2 = _train_generators(seed_key, pair)
            a_loss, b_loss, _stats = _pair_losses(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                pair_index=pair,
                probe_generator_1=generator_1,
                probe_generator_2=generator_2,
                burg_matrix_gradient=burg_gradient,
            )
            combined = a_loss + float(beta) * b_loss
            gradients = torch.autograd.grad(combined, active, retain_graph=False, allow_unused=True)
            _add_(accumulated, gradients)
            count += 1
            del a_loss, b_loss, combined, gradients
    return _scaled(accumulated, 1.0 / float(count), active)


def _sign_state(row: pd.Series | dict[str, Any]) -> str:
    analytic = float(row["analytic_slope"])
    slope_h128 = float(row["slope_h128"])
    slope_h256 = float(row["slope_h256"])
    analytic_margin = max(SLOPE_TOLERANCES.values())
    if (
        analytic < -analytic_margin
        and slope_h128 < -SLOPE_TOLERANCES[1.0 / 128.0]
        and slope_h256 < -SLOPE_TOLERANCES[1.0 / 256.0]
    ):
        return "stable_downhill"
    if (
        analytic > analytic_margin
        and slope_h128 > SLOPE_TOLERANCES[1.0 / 128.0]
        and slope_h256 > SLOPE_TOLERANCES[1.0 / 256.0]
    ):
        return "stable_uphill"
    return "ambiguous"


def _moment_hash(names: Sequence[str], avg: Sequence[torch.Tensor], avg_sq: Sequence[torch.Tensor], step: int) -> str:
    return hashlib.sha256(
        (
            _named_tensor_hash(names, avg)
            + _named_tensor_hash(names, avg_sq)
            + str(int(step))
        ).encode("utf-8")
    ).hexdigest()


def _line_profile(
    *,
    direction_name: str,
    direction: Sequence[torch.Tensor],
    current_f: float,
    run: Any,
    z: torch.Tensor,
    record: dict[str, Any],
    active: Sequence[torch.nn.Parameter],
    base: Sequence[torch.Tensor],
    beta: float,
    epsilon: float,
    hessian_chunk_size: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for alpha in EXTENDED_ALPHAS:
        candidate = _evaluate_candidate(
            run=run,
            z=z,
            record=record,
            active=list(active),
            base=list(base),
            displacement=list(direction),
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            hessian_chunk_size=hessian_chunk_size,
        )
        rows.append(
            {
                "direction": direction_name,
                "alpha": alpha,
                "parameter_radius": alpha * _norm(direction),
                "true_objective": float(candidate["true_objective"]),
                "delta_f": float(candidate["true_objective"]) - current_f,
                "in_original_grid": int(alpha >= 1.0 / 64.0),
                "strict_decrease": int(float(candidate["true_objective"]) < current_f - F_TOLERANCE),
            }
        )
    _set_parameters(list(active), list(base), list(direction), 0.0)
    return rows


def _plot(
    gradients: pd.DataFrame,
    directions: pd.DataFrame,
    lines: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    random_gradients = gradients.loc[gradients["source_kind"].ne("exact")]
    axes[0, 0].bar(
        np.arange(len(random_gradients)),
        random_gradients["cosine_to_exact"],
        color=np.where(random_gradients["source_kind"].eq("p32"), "#dc2626", "#2563eb"),
    )
    axes[0, 0].set_xticks(
        np.arange(len(random_gradients)), random_gradients["source"], rotation=70, ha="right", fontsize=8
    )
    axes[0, 0].set(ylabel="cosine", title="Gradient cosine to complete-basis exact")
    axes[0, 0].axhline(0.0, color="black", linewidth=1)

    carried = directions.loc[directions["transform"].eq("carried_adam")]
    positions = np.arange(len(carried))
    axes[0, 1].bar(positions - 0.18, carried["slope_h128"], width=0.36, label="h=1/128")
    axes[0, 1].bar(positions + 0.18, carried["slope_h256"], width=0.36, label="h=1/256")
    axes[0, 1].set_xticks(positions, carried["source"], rotation=70, ha="right", fontsize=8)
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set(ylabel="dF / d alpha", title="Carried-Adam true-F slopes")
    axes[0, 1].legend()

    pooled = directions.loc[directions["source_kind"].isin(["p32", "exact"])]
    labels = [f"{row.source}\n{row.transform}" for row in pooled.itertuples()]
    positions = np.arange(len(pooled))
    axes[1, 0].bar(positions, pooled["slope_h256"], color="#16a34a")
    axes[1, 0].set_xticks(positions, labels, rotation=60, ha="right", fontsize=8)
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set(ylabel="dF / d alpha", title="P32/exact transform crossing")

    if len(lines):
        minimum = lines.groupby("direction", as_index=False)["delta_f"].min()
        positions = np.arange(len(minimum))
        axes[1, 1].bar(positions, minimum["delta_f"], color="#7c3aed")
        axes[1, 1].set_xticks(positions, minimum["direction"], rotation=70, ha="right", fontsize=7)
        axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set(ylabel="minimum delta F", title="Extended line search for downhill directions")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hessian-chunk-size", type=int, default=64)
    parser.add_argument("--burg-epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frozen_setup = bool(
        args.device == "cuda:0"
        and args.hessian_chunk_size == 64
        and math.isclose(args.burg_epsilon, 1e-4, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.gradient_clip, 1.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.lr, 3e-5, rel_tol=0.0, abs_tol=0.0)
        and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve()
    )
    if not frozen_setup:
        raise RuntimeError("iteration-4 production audit requires the frozen CLI exactly")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"production output directory is not empty: {args.output_dir}")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _source_sha256()
    normalized_source_hash = _normalized_source_sha256()
    frozen_manifest_hash = sha256_file(FROZEN_DEPENDENCY_MANIFEST)
    frozen_manifest = json.loads(FROZEN_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]
    frozen_dependency_hashes = {
        relative_path: sha256_file(project_root / relative_path)
        for relative_path in frozen_manifest["dependencies"]
    }
    frozen_dependency_matches = {
        relative_path: frozen_dependency_hashes[relative_path] == expected_hash
        for relative_path, expected_hash in frozen_manifest["dependencies"].items()
    }
    hard_freeze_valid = bool(
        frozen_manifest_hash == EXPECTED_FROZEN_MANIFEST_SHA256
        and normalized_source_hash == EXPECTED_NORMALIZED_SOURCE_SHA256
        and normalized_source_hash == frozen_manifest["runner_normalized_sha256"]
        and frozen_manifest["protocol_id"] == PROTOCOL_ID
        and all(frozen_dependency_matches.values())
    )
    if not hard_freeze_valid:
        raise RuntimeError(
            "iteration-4 frozen dependency mismatch: "
            + json.dumps(
                {
                    "manifest_hash": frozen_manifest_hash,
                    "normalized_source_hash": normalized_source_hash,
                    "dependency_matches": frozen_dependency_matches,
                },
                sort_keys=True,
            )
        )
    shutil.copy2(Path(__file__), args.output_dir / "executed_source_snapshot.py")
    device = torch.device(args.device)
    dependency_hashes = {
        "protocol": sha256_file(PROTOCOL),
        "iteration3_runner": sha256_file(Path(__file__).with_name("run_one_state_a_full_burg_naive_semantics.py")),
        "iteration2_audit": sha256_file(Path(__file__).with_name("audit_one_state_a_full_burg_direction_mismatch.py")),
        "checkpoint": sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt"),
        "production_summary": sha256_file(PRODUCTION / "summary.json"),
        "production_states": sha256_file(PRODUCTION / "state_objective_curve.csv"),
        "production_proposals": sha256_file(PRODUCTION / "proposal_diagnostics.csv"),
        "replay_checkpoint": sha256_file(REPLAY / "final_checkpoint.pt"),
        "replay_states": sha256_file(REPLAY / "state_objective_curve.csv"),
        "replay_proposals": sha256_file(REPLAY / "proposal_diagnostics.csv"),
        "active_parameters": sha256_file(ACTIVE_PARAMETERS),
        "state_bank": sha256_file(STATE_BANK),
    }
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "normalized_source_sha256": normalized_source_hash,
        "frozen_dependency_manifest_sha256": frozen_manifest_hash,
        "frozen_dependency_matches": frozen_dependency_matches,
        "hard_freeze_valid": hard_freeze_valid,
        "frozen_setup": frozen_setup,
        "dependency_sha256": dependency_hashes,
        "device": str(device),
        "dtype": "float32",
        "state_position": 2,
        "replay_through_proposal": 58,
        "replay_seed_keys": list(REPLAY_SEED_KEYS),
        "blind_seed_keys": list(BLIND_SEED_KEYS),
        "p32_banks": {name: list(values) for name, values in P32_BANKS.items()},
        "beta": DEFAULT_BETA,
        "burg_epsilon": args.burg_epsilon,
        "gradient_clip": args.gradient_clip,
        "lr": args.lr,
        "hessian_chunk_size": args.hessian_chunk_size,
        "fd_alphas": [1.0 / 128.0, 1.0 / 256.0],
        "f_tolerance": F_TOLERANCE,
        "slope_tolerances": {str(key): value for key, value in SLOPE_TOLERANCES.items()},
        "extended_line_alphas": list(EXTENDED_ALPHAS),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[i4-cross] start {json.dumps(resolved, sort_keys=True)}", flush=True)

    print("[i4-cross] stage=load", flush=True)
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    checkpoint_hash = dependency_hashes["checkpoint"]
    bank = pd.read_csv(STATE_BANK)
    selected = bank.loc[bank["state_position"].eq(2)]
    if len(selected) != 1:
        raise RuntimeError("state position 2 is not unique")
    source_index = int(selected.iloc[0]["source_weight_index"])
    weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
    record = run.records.iloc[source_index].to_dict()
    record["source_weight_index"] = source_index
    task_set = _task_set_for_record(run.task_tensors, record)
    cfg_stopped = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
    cfg_naive = replace(cfg_stopped, vae_precond_hvp_mode="autograd")

    active_names = sorted(set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str)))
    named = dict(run.vae.named_parameters())
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in active_names]
    replay_checkpoint = torch.load(REPLAY / "final_checkpoint.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for name in active_names:
            named[name].copy_(replay_checkpoint["active_model_state"][name].to(device=device, dtype=named[name].dtype))
    carried_avg = [replay_checkpoint["exp_avg"][name].to(device=device) for name in active_names]
    carried_avg_sq = [replay_checkpoint["exp_avg_sq"][name].to(device=device) for name in active_names]
    carried_step = int(replay_checkpoint["accepted_adam_step"])
    zero_avg = _zeros_like(active)
    zero_avg_sq = _zeros_like(active)
    base = [parameter.detach().clone() for parameter in active]
    base_parameter_hash = _named_tensor_hash(active_names, base)
    base_moment_hash = _moment_hash(active_names, carried_avg, carried_avg_sq, carried_step)

    production_states = pd.read_csv(PRODUCTION / "state_objective_curve.csv").iloc[:59].reset_index(drop=True)
    replay_states = pd.read_csv(REPLAY / "state_objective_curve.csv").reset_index(drop=True)
    production_proposals = pd.read_csv(PRODUCTION / "proposal_diagnostics.csv").iloc[:58].reset_index(drop=True)
    replay_proposals = pd.read_csv(REPLAY / "proposal_diagnostics.csv").reset_index(drop=True)
    replay_validation = {
        "state_rows": len(replay_states),
        "proposal_rows": len(replay_proposals),
        "accepted_flags_match": bool(
            replay_proposals["accepted"].astype(int).tolist()
            == production_proposals["accepted"].astype(int).tolist()
        ),
        "accepted_alpha_max_abs_error": float(
            np.max(np.abs(replay_proposals["accepted_alpha"] - production_proposals["accepted_alpha"]))
        ),
        "state_f_max_abs_error": float(
            np.max(np.abs(replay_states["true_objective"] - production_states["true_objective"]))
        ),
        "slope_max_abs_error": float(
            np.max(
                np.abs(
                    replay_proposals[["slope_h128", "slope_h256"]].to_numpy()
                    - production_proposals[["slope_h128", "slope_h256"]].to_numpy()
                )
            )
        ),
        "accepted_adam_step": carried_step,
    }
    (args.output_dir / "replay_validation.json").write_text(
        json.dumps(replay_validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    full_ce = all(
        _batch_indices(
            task_set,
            batch_size=int(cfg_naive.vae_precond_batch_size),
            step=10,
            sample_key=source_index,
            pair_key=pair_key,
        )
        is None
        for pair_key in range(64)
    )
    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[i4-cross] loaded source={source_index} z_sha={sha256_tensor(z)} "
        f"active={sum(parameter.numel() for parameter in active)} step={carried_step} full_ce={full_ce}",
        flush=True,
    )

    print("[i4-cross] stage=dense-reference", flush=True)
    metric, hessian, matrix, _eig, burg_gradient = _materialize_metric(
        run=run,
        z=z,
        record=record,
        epsilon=args.burg_epsilon,
        hessian_chunk_size=args.hessian_chunk_size,
    )
    current_f = _objective(metric, DEFAULT_BETA)
    repeatability_rows: list[dict[str, float | int]] = [{"repeat": 0, "true_objective": current_f}]
    for repeat_index in range(1, 5):
        repeat_metric, repeat_h, repeat_m, repeat_eig, repeat_burg = _materialize_metric(
            run=run,
            z=z,
            record=record,
            epsilon=args.burg_epsilon,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        repeatability_rows.append(
            {"repeat": repeat_index, "true_objective": _objective(repeat_metric, DEFAULT_BETA)}
        )
        del repeat_h, repeat_m, repeat_eig, repeat_burg
    repeatability = pd.DataFrame(repeatability_rows)
    repeatability.to_csv(args.output_dir / "objective_repeatability.csv", index=False)
    repeatability_spread = float(
        repeatability["true_objective"].max() - repeatability["true_objective"].min()
    )
    dim = int(z.numel())
    identity = torch.eye(dim, device=device, dtype=torch.float64)
    h64 = hessian.double()
    m64 = matrix.double()
    g_a = 2.0 * (m64 - identity) / float(dim)
    g_b = identity / (float(dim) * (1.0 + args.burg_epsilon)) - torch.linalg.inv(
        m64 + args.burg_epsilon * identity
    ) / float(dim)
    k_cotangent = (2.0 * (g_a + float(DEFAULT_BETA) * g_b) @ h64).detach()
    burg_gradient_error = float(
        ((g_b - burg_gradient.double()).norm() / g_b.norm().clamp_min(1e-30)).cpu()
    )
    forward_rows = _forward_equivalence(
        cfg_stopped=cfg_stopped,
        cfg_naive=cfg_naive,
        run=run,
        z=z,
        record=record,
        hessian=hessian,
        probes=4,
    )
    pd.DataFrame(forward_rows).to_csv(args.output_dir / "forward_equivalence.csv", index=False)
    h_space_rows = _h_space_fd_check(
        hessian,
        k_cotangent,
        beta=DEFAULT_BETA,
        epsilon=args.burg_epsilon,
    )
    pd.DataFrame(h_space_rows).to_csv(args.output_dir / "h_space_fd.csv", index=False)
    del _eig

    print("[i4-cross] stage=basis-preflight", flush=True)
    sequential_preflight, sequential_meta = _full_basis_gradient(
        cfg=cfg_naive,
        run=run,
        z=z,
        record=record,
        active=active,
        k_cotangent=k_cotangent,
        mode="autograd",
        basis_count=8,
    )
    aggregated_preflight = _aggregated_basis_gradient(
        cfg=cfg_naive,
        run=run,
        z=z,
        record=record,
        active=active,
        k_cotangent=k_cotangent,
        basis_count=8,
    )
    basis_preflight = {
        "sequential_to_aggregated_relative_error": _relative_error(
            sequential_preflight, aggregated_preflight
        ),
        "sequential_to_aggregated_cosine": _cosine(sequential_preflight, aggregated_preflight),
        "sequential_metadata": sequential_meta,
    }
    (args.output_dir / "basis_accumulation_preflight.json").write_text(
        json.dumps(basis_preflight, indent=2, sort_keys=True), encoding="utf-8"
    )
    del sequential_preflight, aggregated_preflight

    print("[i4-cross] stage=complete-basis-exact", flush=True)
    exact_gradient, exact_meta = _full_basis_gradient(
        cfg=cfg_naive,
        run=run,
        z=z,
        record=record,
        active=active,
        k_cotangent=k_cotangent,
        mode="autograd",
        basis_count=dim,
    )
    exact_gradient_norm = _norm(exact_gradient)
    if exact_gradient_norm <= 1e-30:
        zero_gradient = {
            "protocol_id": PROTOCOL_ID,
            "valid": False,
            "mechanisms": None,
            "iteration5_candidate": None,
            "reason": "exact gradient norm is zero; normalized crossed directions are undefined",
            "exact_gradient_norm": exact_gradient_norm,
        }
        (args.output_dir / "zero_exact_gradient_abort.json").write_text(
            json.dumps(zero_gradient, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise RuntimeError(str(zero_gradient["reason"]))
    dense_reference = {
        **metric,
        "true_objective": current_f,
        "hessian_sha256": sha256_tensor(hessian),
        "matrix_sha256": sha256_tensor(matrix),
        "k_cotangent_sha256": sha256_tensor(k_cotangent),
        "k_cotangent_norm": float(k_cotangent.norm().cpu()),
        "burg_gradient_relative_error": burg_gradient_error,
        "objective_repeatability_spread": repeatability_spread,
    }
    (args.output_dir / "dense_reference.json").write_text(
        json.dumps(dense_reference, indent=2, sort_keys=True), encoding="utf-8"
    )
    exact_gradient_metadata = {
        **exact_meta,
        "gradient_norm_recomputed": exact_gradient_norm,
        "gradient_sha256": _named_tensor_hash(active_names, exact_gradient),
    }
    (args.output_dir / "exact_gradient_metadata.json").write_text(
        json.dumps(exact_gradient_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("[i4-cross] stage=p4-bank", flush=True)
    p4_gradients: dict[str, Vector] = {}
    atomic_rows: list[dict[str, float | int | str]] = []
    for index, seed_key in enumerate(ALL_P4_SEED_KEYS, start=1):
        gradient, _blocks, rows = _gradient_from_losses(
            cfg=cfg_naive,
            run=run,
            z=z,
            record=record,
            active=active,
            burg_gradient=burg_gradient,
            beta=DEFAULT_BETA,
            proposal=seed_key,
            pairs=4,
            block_size=4,
            label=f"p4_seed_{seed_key}",
        )
        p4_gradients[f"p4_{seed_key}"] = gradient
        atomic_rows.extend({"seed_key": seed_key, **row} for row in rows)
        del _blocks
        print(f"[i4-cross] p4={index}/{len(ALL_P4_SEED_KEYS)} seed={seed_key}", flush=True)
    atomic = pd.DataFrame(atomic_rows)
    atomic.to_csv(args.output_dir / "atomic_pair_losses.csv", index=False)

    pooled_gradients = {
        bank_name: _average([p4_gradients[f"p4_{seed}"] for seed in seeds], active)
        for bank_name, seeds in P32_BANKS.items()
    }
    direct_pooled_gradients = {
        bank_name: _direct_pooled_gradient(
            seed_keys=seeds,
            cfg=cfg_naive,
            run=run,
            z=z,
            record=record,
            active=active,
            burg_gradient=burg_gradient,
            beta=DEFAULT_BETA,
        )
        for bank_name, seeds in P32_BANKS.items()
    }
    pooling_identity_errors = {
        bank_name: _relative_error(pooled_gradients[bank_name], direct_pooled_gradients[bank_name])
        for bank_name in P32_BANKS
    }
    del direct_pooled_gradients
    source_vectors: dict[str, Vector] = {"exact": exact_gradient, **p4_gradients, **pooled_gradients}
    source_kind = {
        name: "exact" if name == "exact" else "p32" if name.startswith("p32") else "p4"
        for name in source_vectors
    }

    seed59_carried, seed59_meta = _adam_direction(
        gradient=p4_gradients["p4_59"],
        active=active,
        exp_avg=carried_avg,
        exp_avg_sq=carried_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    target_norm = _norm(seed59_carried)
    production_all_proposals = pd.read_csv(PRODUCTION / "proposal_diagnostics.csv")
    production_seed59_rows = production_all_proposals.loc[production_all_proposals["proposal"].eq(59)]
    production_seed67_rows = production_all_proposals.loc[production_all_proposals["proposal"].eq(67)]
    if len(production_seed59_rows) != 1 or len(production_seed67_rows) != 1:
        raise RuntimeError("production replay-control rows 59/67 are not unique")
    production_seed59 = production_seed59_rows.iloc[0]
    production_seed67 = production_seed67_rows.iloc[0]

    torch_reference, torch_after = _torch_adam_reference_direction(
        gradient=p4_gradients["p4_59"],
        active=active,
        exp_avg=carried_avg,
        exp_avg_sq=carried_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    custom_adam_relative_error = _relative_error(seed59_carried, torch_reference)
    custom_adam_cosine = _cosine(seed59_carried, torch_reference)
    custom_after = [
        (parameter.detach() + delta.detach().to(dtype=parameter.dtype)).clone()
        for parameter, delta in zip(active, seed59_carried, strict=True)
    ]
    custom_adam_applied_relative_error = _norm(
        [custom - reference for custom, reference in zip(custom_after, torch_after, strict=True)]
    ) / max(_norm(seed59_carried), 1e-30)
    custom_adam_applied_max_abs = max(
        float((custom - reference).abs().max().cpu())
        for custom, reference in zip(custom_after, torch_after, strict=True)
    )
    del torch_reference, torch_after, custom_after
    adam_reference_preflight = {
        "custom_direction_relative_error": custom_adam_relative_error,
        "custom_direction_cosine": custom_adam_cosine,
        "applied_update_relative_error": custom_adam_applied_relative_error,
        "applied_update_max_abs": custom_adam_applied_max_abs,
    }
    (args.output_dir / "adam_reference_preflight.json").write_text(
        json.dumps(adam_reference_preflight, indent=2, sort_keys=True), encoding="utf-8"
    )

    replay_control_rows: list[dict[str, float | int | str]] = []
    for replay_seed, production_row in ((59, production_seed59), (67, production_seed67)):
        replay_direction, _replay_meta = _adam_direction(
            gradient=p4_gradients[f"p4_{replay_seed}"],
            active=active,
            exp_avg=carried_avg,
            exp_avg_sq=carried_avg_sq,
            accepted_step=carried_step,
            gradient_clip=args.gradient_clip,
            lr=args.lr,
        )
        replay_summary, _replay_fd_rows = _directional_fd(
            name=f"replay_p4_{replay_seed}_carried",
            direction=replay_direction,
            exact_gradient=exact_gradient,
            current_f=current_f,
            run=run,
            z=z,
            record=record,
            active=active,
            base=base,
            beta=DEFAULT_BETA,
            epsilon=args.burg_epsilon,
            hessian_chunk_size=args.hessian_chunk_size,
        )
        replay_control_rows.append(
            {
                "seed_key": replay_seed,
                "observed_norm": float(replay_summary["direction_norm"]),
                "expected_norm": float(production_row["proposal_norm"]),
                "observed_slope_h128": float(replay_summary["slope_h128"]),
                "expected_slope_h128": float(production_row["slope_h128"]),
                "observed_slope_h256": float(replay_summary["slope_h256"]),
                "expected_slope_h256": float(production_row["slope_h256"]),
            }
        )
        del replay_direction, _replay_fd_rows
    replay_controls = pd.DataFrame(replay_control_rows)
    replay_controls.to_csv(args.output_dir / "replay_control_directions.csv", index=False)

    gradient_rows: list[dict[str, float | int | str]] = []
    for name, gradient in source_vectors.items():
        gradient_rows.append(
            {
                "source": name,
                "source_kind": source_kind[name],
                "pair_count": 512 if name == "exact" else 32 if name.startswith("p32") else 4,
                "gradient_norm": _norm(gradient),
                "cosine_to_exact": 1.0 if name == "exact" else _cosine(gradient, exact_gradient),
                "relative_error_to_exact": 0.0 if name == "exact" else _relative_error(gradient, exact_gradient),
                "gradient_sha256": _named_tensor_hash(active_names, gradient),
            }
        )
    gradients = pd.DataFrame(gradient_rows)
    gradients.to_csv(args.output_dir / "gradient_sources.csv", index=False)

    exact_directions: dict[str, Vector] = {}
    exact_directions["raw"] = _negative_normalized(exact_gradient, target_norm=target_norm, active=active)
    exact_zero, _exact_zero_meta = _adam_direction(
        gradient=exact_gradient,
        active=active,
        exp_avg=zero_avg,
        exp_avg_sq=zero_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    exact_directions["zero_moment_adam"] = _normalize_direction(
        exact_zero, target_norm=target_norm, active=active
    )
    exact_carried, _exact_carried_meta = _adam_direction(
        gradient=exact_gradient,
        active=active,
        exp_avg=carried_avg,
        exp_avg_sq=carried_avg_sq,
        accepted_step=carried_step,
        gradient_clip=args.gradient_clip,
        lr=args.lr,
    )
    exact_directions["carried_adam"] = _normalize_direction(
        exact_carried, target_norm=target_norm, active=active
    )

    print(f"[i4-cross] stage=crossed-directions count={len(source_vectors) * 3}", flush=True)
    direction_rows: list[dict[str, float | int | str]] = []
    fd_rows: list[dict[str, float | str]] = []
    line_rows: list[dict[str, float | str]] = []
    restoration_checks: list[int] = []
    for source_index_in_loop, (source_name, gradient) in enumerate(source_vectors.items(), start=1):
        transforms: dict[str, tuple[Vector, dict[str, float]]] = {}
        raw = _negative_normalized(gradient, target_norm=target_norm, active=active)
        transforms["raw"] = (
            raw,
            {"input_gradient_norm": _norm(gradient), "clip_factor": 1.0, "unnormalized_direction_norm": _norm(raw)},
        )
        zero_direction, zero_meta = _adam_direction(
            gradient=gradient,
            active=active,
            exp_avg=zero_avg,
            exp_avg_sq=zero_avg_sq,
            accepted_step=carried_step,
            gradient_clip=args.gradient_clip,
            lr=args.lr,
        )
        transforms["zero_moment_adam"] = (
            _normalize_direction(zero_direction, target_norm=target_norm, active=active),
            zero_meta,
        )
        carried_direction, carried_meta = _adam_direction(
            gradient=gradient,
            active=active,
            exp_avg=carried_avg,
            exp_avg_sq=carried_avg_sq,
            accepted_step=carried_step,
            gradient_clip=args.gradient_clip,
            lr=args.lr,
        )
        transforms["carried_adam"] = (
            _normalize_direction(carried_direction, target_norm=target_norm, active=active),
            carried_meta,
        )

        for transform, (direction, transform_meta) in transforms.items():
            name = f"{source_name}__{transform}"
            parameter_hash_before = _named_tensor_hash(active_names, active)
            moment_hash_before = _moment_hash(active_names, carried_avg, carried_avg_sq, carried_step)
            summary, rows = _directional_fd(
                name=name,
                direction=direction,
                exact_gradient=exact_gradient,
                current_f=current_f,
                run=run,
                z=z,
                record=record,
                active=active,
                base=base,
                beta=DEFAULT_BETA,
                epsilon=args.burg_epsilon,
                hessian_chunk_size=args.hessian_chunk_size,
            )
            sign_state = _sign_state(summary)
            all_slopes_negative = bool(
                float(summary["analytic_slope"]) < 0.0
                and float(summary["slope_h128"]) < 0.0
                and float(summary["slope_h256"]) < 0.0
            )
            if all_slopes_negative or name == "exact__raw":
                line_rows.extend(
                    _line_profile(
                        direction_name=name,
                        direction=direction,
                        current_f=current_f,
                        run=run,
                        z=z,
                        record=record,
                        active=active,
                        base=base,
                        beta=DEFAULT_BETA,
                        epsilon=args.burg_epsilon,
                        hessian_chunk_size=args.hessian_chunk_size,
                    )
                )
            parameter_hash_after = _named_tensor_hash(active_names, active)
            moment_hash_after = _moment_hash(active_names, carried_avg, carried_avg_sq, carried_step)
            restoration_checks.append(
                int(
                    parameter_hash_before == base_parameter_hash
                    and parameter_hash_after == base_parameter_hash
                    and moment_hash_before == base_moment_hash
                    and moment_hash_after == base_moment_hash
                )
            )
            direction_rows.append(
                {
                    **summary,
                    "source": source_name,
                    "source_kind": source_kind[source_name],
                    "transform": transform,
                    "direction_cosine_to_exact_transform": _cosine(direction, exact_directions[transform]),
                    "input_gradient_norm": transform_meta["input_gradient_norm"],
                    "clip_factor": transform_meta["clip_factor"],
                    "unnormalized_direction_norm": transform_meta["unnormalized_direction_norm"],
                    "sign_state": sign_state,
                    "all_slopes_negative": int(all_slopes_negative),
                    "both_fd_negative": int(
                        float(summary["slope_h128"]) < 0.0 and float(summary["slope_h256"]) < 0.0
                    ),
                    "parameter_hash_before": parameter_hash_before,
                    "parameter_hash_after": parameter_hash_after,
                    "moment_hash_before": moment_hash_before,
                    "moment_hash_after": moment_hash_after,
                    "parameter_and_moments_restored": restoration_checks[-1],
                }
            )
            fd_rows.extend({"source": source_name, "transform": transform, **row} for row in rows)
        print(
            f"[i4-cross] source={source_index_in_loop}/{len(source_vectors)} name={source_name} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del transforms, raw, zero_direction, carried_direction

    directions = pd.DataFrame(direction_rows)
    fd = pd.DataFrame(fd_rows)
    lines = pd.DataFrame(line_rows)
    directions.to_csv(args.output_dir / "direction_diagnostics.csv", index=False)
    fd.to_csv(args.output_dir / "direction_finite_differences.csv", index=False)
    lines.to_csv(args.output_dir / "extended_line_profiles.csv", index=False)

    def direction_state(source: str, transform: str) -> str:
        rows = directions.loc[
            directions["source"].eq(source) & directions["transform"].eq(transform)
        ]
        if len(rows) != 1:
            raise RuntimeError(f"direction key is not unique: {source}/{transform}")
        return str(rows.iloc[0]["sign_state"])

    blind_p4_sources = [f"p4_{seed}" for seed in BLIND_SEED_KEYS]
    blind_p4_carried = directions.loc[
        directions["source"].isin(blind_p4_sources) & directions["transform"].eq("carried_adam")
    ]
    pool_closer: dict[str, bool] = {}
    for pool_name, seeds in P32_BANKS.items():
        pool_rows = gradients.loc[gradients["source"].eq(pool_name)]
        block_rows = gradients.loc[gradients["source"].isin([f"p4_{seed}" for seed in seeds])]
        if len(pool_rows) != 1 or len(block_rows) != 8:
            raise RuntimeError(f"gradient pooling coverage invalid for {pool_name}")
        pool_row = pool_rows.iloc[0]
        pool_closer[pool_name] = bool(
            float(pool_row["cosine_to_exact"]) > float(block_rows["cosine_to_exact"].median())
            and float(pool_row["relative_error_to_exact"])
            < float(block_rows["relative_error_to_exact"].median())
        )

    finite_probe_by_transform: dict[str, bool] = {}
    for transform in ("raw", "zero_moment_adam", "carried_adam"):
        blind_states = set(
            directions.loc[
                directions["source"].isin(blind_p4_sources)
                & directions["transform"].eq(transform),
                "sign_state",
            ]
        )
        finite_probe_by_transform[transform] = bool(
            direction_state("exact", transform) == "stable_downhill"
            and all(direction_state(pool, transform) == "stable_downhill" for pool in P32_BANKS)
            and "stable_downhill" in blind_states
            and "stable_uphill" in blind_states
            and all(pool_closer.values())
        )
    finite_probe_raw_supported = finite_probe_by_transform["raw"]
    finite_probe_carried_repair_supported = finite_probe_by_transform["carried_adam"]

    history_patterns = {
        source: bool(
            direction_state(source, "raw") == "stable_downhill"
            and direction_state(source, "zero_moment_adam") == "stable_downhill"
            and direction_state(source, "carried_adam") == "stable_uphill"
        )
        for source in ("exact", *P32_BANKS.keys())
    }
    carried_history_supported = bool(
        history_patterns["exact"] or all(history_patterns[pool] for pool in P32_BANKS)
    )
    diagonal_patterns = {
        pool: bool(
            direction_state(pool, "raw") == "stable_downhill"
            and direction_state(pool, "zero_moment_adam") == "stable_uphill"
        )
        for pool in P32_BANKS
    }
    adam_diagonal_supported = bool(
        direction_state("exact", "zero_moment_adam") == "stable_downhill"
        and all(diagonal_patterns.values())
    )
    radius_supported_directions: list[str] = []
    if len(lines):
        for direction_name, group in lines.groupby("direction"):
            row = directions.loc[directions["direction"].eq(direction_name)].iloc[0]
            original_decrease = bool(
                ((group["in_original_grid"] == 1) & (group["strict_decrease"] == 1)).any()
            )
            extended_decrease = bool(
                ((group["in_original_grid"] == 0) & (group["strict_decrease"] == 1)).any()
            )
            if (
                int(row["all_slopes_negative"]) == 1
                and not original_decrease
                and extended_decrease
            ):
                radius_supported_directions.append(str(direction_name))
    radius_supported_count = len(radius_supported_directions)
    radius_supported_for_residual = "p4_59__carried_adam" in radius_supported_directions
    exact_raw_row = directions.loc[
        directions["source"].eq("exact") & directions["transform"].eq("raw")
    ]
    exact_raw_lines = lines.loc[lines["direction"].eq("exact__raw")]
    if len(exact_raw_row) != 1 or len(exact_raw_lines) != 13:
        raise RuntimeError("exact-raw direction/line coverage is invalid")
    exact_raw_row = exact_raw_row.iloc[0]
    exact_stationary_floor = bool(
        abs(float(exact_raw_row["analytic_slope"])) <= max(SLOPE_TOLERANCES.values())
        and abs(float(exact_raw_row["slope_h128"])) <= SLOPE_TOLERANCES[1.0 / 128.0]
        and abs(float(exact_raw_row["slope_h256"])) <= SLOPE_TOLERANCES[1.0 / 256.0]
        and float(exact_raw_lines["delta_f"].min()) >= -F_TOLERANCE
    )
    stationarity_supported = exact_stationary_floor

    final_parameter_hash = _named_tensor_hash(active_names, active)
    final_moment_hash = _moment_hash(active_names, carried_avg, carried_avg_sq, carried_step)
    forward_frame = pd.DataFrame(forward_rows)
    h_space_frame = pd.DataFrame(h_space_rows)
    forward_max_relative_error = float(
        forward_frame[
            [
                "stopped_to_dense_relative_error",
                "naive_to_dense_relative_error",
                "stopped_to_naive_relative_error",
            ]
        ].to_numpy().max()
    )
    h_space_max_relative_error = float(h_space_frame["relative_error"].max())
    numeric_finite = all(
        _numeric_frame_is_finite(frame)
        for frame in (
            gradients,
            directions,
            fd,
            lines,
            forward_frame,
            h_space_frame,
            atomic,
            repeatability,
            replay_controls,
        )
    )
    expected_sources = {"exact", *(f"p4_{seed}" for seed in ALL_P4_SEED_KEYS), *P32_BANKS.keys()}
    expected_direction_keys = {
        (source, transform)
        for source in expected_sources
        for transform in ("raw", "zero_moment_adam", "carried_adam")
    }
    observed_direction_keys = set(zip(directions["source"], directions["transform"], strict=True))
    expected_fd_keys = {
        (f"{source}__{transform}", alpha)
        for source, transform in expected_direction_keys
        for alpha in (1.0 / 128.0, 1.0 / 256.0)
    }
    observed_fd_keys = set(zip(fd["direction"], fd["alpha"].astype(float), strict=True))
    expected_line_directions = set(
        directions.loc[directions["all_slopes_negative"].eq(1), "direction"]
    ) | {"exact__raw"}
    observed_line_directions = set(lines["direction"])
    expected_line_keys = {
        (direction, alpha) for direction in expected_line_directions for alpha in EXTENDED_ALPHAS
    }
    observed_line_keys = set(zip(lines["direction"], lines["alpha"].astype(float), strict=True))
    expected_atomic_keys = {(seed, pair) for seed in ALL_P4_SEED_KEYS for pair in range(4)}
    observed_atomic_keys = set(zip(atomic["seed_key"].astype(int), atomic["pair"].astype(int), strict=True))
    replay_norm_relative_error = float(
        (
            (replay_controls["observed_norm"] - replay_controls["expected_norm"]).abs()
            / replay_controls["expected_norm"].abs().clip(lower=1e-30)
        ).max()
    )
    replay_slope_max_abs_error = float(
        max(
            (replay_controls["observed_slope_h128"] - replay_controls["expected_slope_h128"])
            .abs()
            .max(),
            (replay_controls["observed_slope_h256"] - replay_controls["expected_slope_h256"])
            .abs()
            .max(),
        )
    )
    exact_raw_fd_nonstationary_valid = bool(
        all(
            abs(float(exact_raw_row[column]) - float(exact_raw_row["analytic_slope"]))
            <= max(
                5e-3,
                0.05
                * max(
                    abs(float(exact_raw_row[column])),
                    abs(float(exact_raw_row["analytic_slope"])),
                ),
            )
            for column in ("slope_h128", "slope_h256")
        )
        and str(exact_raw_row["sign_state"]) == "stable_downhill"
        and int(exact_raw_row["descent_ordering_h128"]) == 1
        and int(exact_raw_row["descent_ordering_h256"]) == 1
    )
    exact_zero_rows = directions.loc[
        directions["source"].eq("exact")
        & directions["transform"].eq("zero_moment_adam")
    ]
    if len(exact_zero_rows) != 1:
        raise RuntimeError("exact zero-moment direction is not unique")
    exact_zero_row = exact_zero_rows.iloc[0]
    exact_zero_stationary_floor = bool(
        exact_stationary_floor
        and abs(float(exact_zero_row["analytic_slope"])) <= max(SLOPE_TOLERANCES.values())
        and abs(float(exact_zero_row["slope_h128"])) <= SLOPE_TOLERANCES[1.0 / 128.0]
        and abs(float(exact_zero_row["slope_h256"])) <= SLOPE_TOLERANCES[1.0 / 256.0]
    )
    exact_zero_sanity_valid = bool(
        str(exact_zero_row["sign_state"]) == "stable_downhill" or exact_zero_stationary_floor
    )
    validity_gates = {
        "frozen_setup": frozen_setup,
        "hard_frozen_dependency_manifest_matches": hard_freeze_valid,
        "checkpoint_hash_matches": checkpoint_hash == EXPECTED_CHECKPOINT,
        "iteration3_runner_hash_matches": dependency_hashes["iteration3_runner"] == EXPECTED_RUNNER_SHA256,
        "z_hash_matches": sha256_tensor(z) == EXPECTED_Z_SHA256,
        "full_ce": full_ce,
        "replay_state_rows_59": len(replay_states) == 59,
        "replay_proposal_rows_58": len(replay_proposals) == 58,
        "replay_accepted_flags_match": replay_validation["accepted_flags_match"],
        "replay_alphas_match": replay_validation["accepted_alpha_max_abs_error"] <= 1e-12,
        "replay_state_f_at_most_1e_5": replay_validation["state_f_max_abs_error"] <= 1e-5,
        "replay_slopes_at_most_5e_3": replay_validation["slope_max_abs_error"] <= 5e-3,
        "replay_adam_step_54": carried_step == EXPECTED_ADAM_STEP,
        "current_f_matches": abs(current_f - EXPECTED_STATE_F) <= 1e-5,
        "objective_repeatability_spread_within_f_tolerance": repeatability_spread <= F_TOLERANCE,
        "hessian_symmetry_at_most_1e_5": float(metric["hessian_symmetry_rel"]) <= 1e-5,
        "burg_gradient_matches": burg_gradient_error <= 1e-6,
        "forward_equivalence": forward_max_relative_error <= 1e-5,
        "h_space_fd_at_most_1e_5": h_space_max_relative_error <= 1e-5,
        "basis_preflight_relative_error_at_most_1e_6": basis_preflight[
            "sequential_to_aggregated_relative_error"
        ]
        <= 1e-6,
        "basis_preflight_memory_growth_at_most_64mib": bool(
            sequential_meta["memory_growth_gate_pass"]
        ),
        "exact_gradient_finite": _vector_is_finite(exact_gradient),
        "complete_basis_512": int(exact_meta["basis_count"]) == 512,
        "complete_basis_memory_growth_at_most_64mib": bool(exact_meta["memory_growth_gate_pass"]),
        "complete_basis_no_unused_active_tensors": int(exact_meta["unused_parameter_tensors_any_row"])
        == 0
        and int(exact_meta["unused_parameter_tensors_all_rows"]) == 0,
        "exact_raw_reference_nonstationary_or_floor_valid": exact_raw_fd_nonstationary_valid
        or exact_stationary_floor,
        "exact_zero_moment_sanity_downhill_or_floor": exact_zero_sanity_valid,
        "custom_adam_max_abs_at_most_3e_8": custom_adam_applied_max_abs <= 3e-8,
        "custom_adam_applied_relative_error_at_most_1e_5": custom_adam_applied_relative_error
        <= 1e-5,
        "custom_adam_raw_relative_cancellation_sanity_at_most_5e_4": custom_adam_relative_error
        <= 5e-4,
        "custom_adam_cosine_at_least_0p9999999": custom_adam_cosine >= 0.9999999,
        "p32_raw_pooling_identity_at_most_1e_6": max(pooling_identity_errors.values()) <= 1e-6,
        "gradient_source_coverage_21": len(gradients) == 21
        and set(gradients["source"]) == expected_sources
        and not gradients["source"].duplicated().any(),
        "direction_coverage_63": len(directions) == 63
        and observed_direction_keys == expected_direction_keys
        and not directions.duplicated(["source", "transform"]).any(),
        "fd_coverage_126": len(fd) == 126
        and observed_fd_keys == expected_fd_keys
        and not fd.duplicated(["direction", "alpha"]).any(),
        "atomic_coverage_72": len(atomic) == 72
        and observed_atomic_keys == expected_atomic_keys
        and not atomic.duplicated(["seed_key", "pair"]).any(),
        "line_coverage_matches_stable_directions": observed_line_directions == expected_line_directions
        and observed_line_keys == expected_line_keys
        and len(lines) == len(expected_line_directions) * 13
        and not lines.duplicated(["direction", "alpha"]).any(),
        "all_numeric_values_finite": numeric_finite,
        "replay_control_norms_reproduce": replay_norm_relative_error <= 1e-6,
        "replay_control_slopes_reproduce": replay_slope_max_abs_error <= 5e-3,
        "all_direction_evaluations_restore_state": all(restoration_checks),
        "all_stored_before_after_hashes_equal_base": bool(
            directions["parameter_hash_before"].eq(base_parameter_hash).all()
            and directions["parameter_hash_after"].eq(base_parameter_hash).all()
            and directions["moment_hash_before"].eq(base_moment_hash).all()
            and directions["moment_hash_after"].eq(base_moment_hash).all()
        ),
        "final_parameter_hash_restored": final_parameter_hash == base_parameter_hash,
        "final_moment_hash_restored": final_moment_hash == base_moment_hash,
    }
    valid = bool(all(validity_gates.values()))
    mechanisms = None
    iteration5_candidate = None
    primary_sources = ("exact", *P32_BANKS.keys())
    primary_sign_states = {
        source: {
            transform: direction_state(source, transform)
            for transform in ("raw", "zero_moment_adam", "carried_adam")
        }
        for source in primary_sources
    }
    primary_ambiguous = any(
        state == "ambiguous"
        for source_states in primary_sign_states.values()
        for state in source_states.values()
    )
    if valid:
        mechanisms = {
            "finite_probe_by_transform": finite_probe_by_transform,
            "finite_probe_raw_supported": finite_probe_raw_supported,
            "finite_probe_carried_repair_supported": finite_probe_carried_repair_supported,
            "carried_history_supported": carried_history_supported,
            "history_patterns": history_patterns,
            "adam_diagonal_supported": adam_diagonal_supported,
            "diagonal_patterns": diagonal_patterns,
            "radius_supported_direction_count": radius_supported_count,
            "radius_supported_directions": radius_supported_directions,
            "radius_supported_for_residual": radius_supported_for_residual,
            "stationarity_supported": stationarity_supported,
            "primary_sign_ambiguity": primary_ambiguous,
            "mixed_or_unresolved": (primary_ambiguous and not stationarity_supported)
            or not any(
                (
                    finite_probe_carried_repair_supported,
                    carried_history_supported,
                    adam_diagonal_supported,
                    radius_supported_for_residual,
                    stationarity_supported,
                )
            ),
        }
        if not primary_ambiguous:
            if adam_diagonal_supported and finite_probe_raw_supported:
                iteration5_candidate = "p32_plus_raw_sgd_training"
            elif adam_diagonal_supported:
                iteration5_candidate = "raw_sgd_training"
            elif finite_probe_raw_supported and carried_history_supported:
                iteration5_candidate = "p32_plus_zero_moment_training"
            elif carried_history_supported:
                iteration5_candidate = "zero_moment_optimizer_training"
            elif finite_probe_carried_repair_supported:
                iteration5_candidate = "p32_variance_controlled_training"
            elif (
                direction_state("exact", "carried_adam") == "stable_downhill"
                and all(
                    direction_state(pool, "carried_adam") == "stable_uphill"
                    for pool in P32_BANKS
                )
            ):
                iteration5_candidate = "complete_basis_training"
            elif radius_supported_for_residual:
                iteration5_candidate = "extended_line_search"

    decision = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "normalized_source_sha256": normalized_source_hash,
        "frozen_dependency_manifest_sha256": frozen_manifest_hash,
        "valid": valid,
        "validity_gates": validity_gates,
        "current_f": current_f,
        "exact_gradient_norm": exact_gradient_norm,
        "target_direction_norm": target_norm,
        "f_tolerance": F_TOLERANCE,
        "slope_tolerances": {str(key): value for key, value in SLOPE_TOLERANCES.items()},
        "objective_repeatability_spread": repeatability_spread,
        "exact_raw_nonstationary_valid": exact_raw_fd_nonstationary_valid,
        "exact_stationary_floor": exact_stationary_floor,
        "primary_sign_states": primary_sign_states,
        "pool_closer_to_exact": pool_closer,
        "pooling_identity_relative_errors": pooling_identity_errors,
        "blind_p4_carried_sign_counts": {
            str(key): int(value)
            for key, value in blind_p4_carried["sign_state"].value_counts().to_dict().items()
        },
        "mechanisms": mechanisms,
        "iteration5_candidate": iteration5_candidate,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(gradients, directions, lines, args.output_dir / "probe_adam_cross.png")

    artifact_paths = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hash,
        "normalized_source_sha256": normalized_source_hash,
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths if path.name != "run.log"},
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[i4-cross] decision {json.dumps(decision, sort_keys=True)}", flush=True)
    print(
        "[i4-cross] artifacts=" + ",".join(str(path) for path in sorted(args.output_dir.iterdir())),
        flush=True,
    )


if __name__ == "__main__":
    main()
