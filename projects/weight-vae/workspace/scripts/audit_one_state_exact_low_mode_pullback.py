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
from scripts import audit_one_state_exact_a_realized_path as realized_helpers
from scripts.audit_one_state_a_full_burg_direction_mismatch import (
    _dot,
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
PROTOCOL_ID = "one_state_exact_low_mode_pullback_i4_v1"
OUTPUT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_exact_a_h2048"
)
PROTOCOL_PATH = OUTPUT_ROOT / "iteration_4_low_mode_pullback/protocol.md"
DEFAULT_OUTPUT = OUTPUT_ROOT / "iteration4_low_mode_pullback_production"
FROZEN_DEPENDENCY_MANIFEST = OUTPUT_ROOT / "iteration_4_frozen_dependency_manifest.json"
EXPECTED_FROZEN_DEPENDENCY_MANIFEST_SHA256 = "e980dc1a224930c04b950b59285c2a8dc44d1ca140d02545bb5d59fb8b311af8"
EXPECTED_NORMALIZED_SOURCE_SHA256 = "e020ec6e8fae92e3dab480d32c7f05e71eeb54ea6c9db6906f468ccfdf50b390"

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
ALPHAS = (-1.0 / 256.0, 0.0, 1.0 / 256.0, 1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0)
H_FD_STEPS = (1e-4, 5e-5)

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


def _low_energy(matrix: torch.Tensor, basis: torch.Tensor) -> float:
    projected = basis.transpose(0, 1) @ matrix.double() @ basis
    return float((torch.trace(projected) / float(basis.shape[1])).cpu())


def _loss_low(hessian: torch.Tensor, low_basis: torch.Tensor) -> torch.Tensor:
    projected = low_basis.transpose(0, 1) @ hessian.double()
    return -projected.square().sum() / float(low_basis.shape[1])


def _h_space_fd(
    hessian: torch.Tensor, low_basis: torch.Tensor, cotangent: torch.Tensor
) -> list[dict[str, float | bool]]:
    direction = cotangent / cotangent.norm().clamp_min(1e-30)
    analytic = float((cotangent * direction).sum().cpu())
    rows: list[dict[str, float | bool]] = []
    for step in H_FD_STEPS:
        plus = float(_loss_low(hessian.double() + step * direction, low_basis).cpu())
        minus = float(_loss_low(hessian.double() - step * direction, low_basis).cpu())
        observed = (plus - minus) / (2.0 * step)
        relative = abs(observed - analytic) / max(abs(analytic), abs(observed), 1e-30)
        rows.append(
            {
                "step": step,
                "analytic_derivative": analytic,
                "fd_derivative": observed,
                "relative_error": relative,
                "sign_agrees": bool(analytic * observed > 0.0),
            }
        )
    return rows


def _endpoint_metrics(
    *,
    run: Any,
    z: torch.Tensor,
    record: Mapping[str, Any],
    low_basis: torch.Tensor,
    dead_basis: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    metrics, _h, matrix, eig_m, _burg = exact_helpers._evaluate_dense(
        run=run,
        z=z,
        record=record,
        epsilon=EPSILON,
        hessian_chunk_size=HESSIAN_CHUNK_SIZE,
    )
    metrics.update(
        {
            "frozen_low_energy": _low_energy(matrix, low_basis),
            "frozen_dead_energy": _low_energy(matrix, dead_basis),
            "count_lt_1e_4": float((eig_m < DEAD_THRESHOLD).sum().cpu()),
            "count_lt_1e_2": float((eig_m < 1e-2).sum().cpu()),
            "count_lt_0p1": float((eig_m < LOW_THRESHOLD).sum().cpu()),
        }
    )
    del _h, matrix, _burg
    return metrics, eig_m.detach().double().clone()


def _repeat_errors(
    primary: Mapping[str, float],
    repeat: Mapping[str, float],
    primary_eig: torch.Tensor,
    repeat_eig: torch.Tensor,
) -> dict[str, float]:
    discrete = {"count_lt_1e_4", "count_lt_1e_2", "count_lt_0p1"}
    continuous = sorted((set(primary) & set(repeat)) - discrete - {"hessian_sec"})
    result = {
        f"repeat_{key}_abs_error": abs(float(primary[key]) - float(repeat[key]))
        for key in continuous
    }
    result["repeat_spectrum_max_abs_error"] = float(
        (primary_eig - repeat_eig).abs().max().cpu()
    )
    for key in sorted(discrete & set(primary) & set(repeat)):
        result[f"repeat_{key}_abs_error"] = abs(float(primary[key]) - float(repeat[key]))
    return result


def _numeric_finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def _plot(output: Path, endpoints: pd.DataFrame, spectra: pd.DataFrame) -> None:
    positive = endpoints.loc[endpoints["alpha"].ge(0.0)].sort_values("alpha")
    base = endpoints.loc[np.isclose(endpoints["alpha"], 0.0)].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panels = (
        ("frozen_low_energy", "Frozen low-subspace energy"),
        ("exact_a_per_dim", "Exact dense A"),
        ("damped_full_burg_per_dim", "Damped full Burg B"),
        ("m_max", "Maximum M eigenvalue"),
    )
    for axis, (column, title) in zip(axes.flat, panels, strict=True):
        axis.plot(positive["alpha"], positive[column], marker="o", linewidth=2.0)
        axis.axhline(float(base[column]), color="black", linestyle="--", linewidth=1.0)
        axis.set_xscale("symlog", linthresh=1.0 / 256.0, base=2)
        axis.set(title=title, xlabel="alpha", ylabel=column)
        axis.grid(alpha=0.25)
    fig.suptitle("Exact frozen-low-mode pullback line")
    fig.savefig(output / "low_mode_line.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for alpha in (0.0, 1.0 / 64.0, 1.0 / 4.0, 1.0):
        rows = spectra.loc[np.isclose(spectra["alpha"], alpha)].sort_values("rank")
        axes[0].plot(rows["rank"], np.clip(rows["m_eigenvalue"], 1e-14, None), label=f"a={alpha:g}")
    axes[0].axhline(1.0, color="black", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set(title="Ordered M spectra", xlabel="rank", ylabel="eigenvalue")
    axes[0].legend()
    axes[1].plot(positive["alpha"], positive["m_p50"], marker="o", label="median")
    axes[1].plot(
        positive["alpha"],
        positive["count_lt_1e_4"] / 512.0,
        marker="o",
        label="fraction <1e-4",
    )
    axes[1].plot(
        positive["alpha"],
        positive["count_lt_0p1"] / 512.0,
        marker="o",
        label="fraction <0.1",
    )
    axes[1].set_xscale("symlog", linthresh=1.0 / 256.0, base=2)
    axes[1].set(title="Bulk response", xlabel="alpha", ylabel="value")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "low_mode_spectra.png", dpi=180)
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
    shutil.copy2(FROZEN_DEPENDENCY_MANIFEST, staging / "frozen_dependency_manifest_snapshot.json")

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_handle = (staging / "run.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, log_handle)  # type: ignore[assignment]
    started = time.perf_counter()
    device = torch.device("cuda:0")
    resolved = {
        "protocol_id": PROTOCOL_ID,
        "device": str(device),
        "dtype": "float32 model/HVP; float64 matrix diagnostics",
        "seed": "none; exact full-basis pullback",
        "cache_mode": "load accepted run and serialized Iteration-3 active checkpoint",
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
        "alphas": list(ALPHAS),
        "h_fd_steps": list(H_FD_STEPS),
        "output_dir": str(final_output),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_sha256": _source_sha256(source_snapshot),
        "normalized_source_sha256": _normalized_source_sha256(source_snapshot),
        "frozen_dependency_manifest_sha256": sha256_file(FROZEN_DEPENDENCY_MANIFEST),
        "frozen_dependency_matches": dependency_matches,
    }
    (staging / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[low-mode-i4] startup {json.dumps(resolved, sort_keys=True)}", flush=True)

    try:
        print("[low-mode-i4] stage=load-fresh-checkpoint", flush=True)
        accepted_checkpoint = DEFAULT_RUN_DIR / "vae_checkpoint.pt"
        if sha256_file(accepted_checkpoint) != EXPECTED_CHECKPOINT:
            raise RuntimeError("accepted h2048 checkpoint hash mismatch")
        if sha256_file(ITERATION3_CHECKPOINT) != EXPECTED_ITERATION3_CHECKPOINT_SHA256:
            raise RuntimeError("Iteration-3 final checkpoint file hash mismatch")
        run = _load_run(DEFAULT_RUN_DIR, device=device)
        cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=4, batch_size=16384)
        cfg = replace(cfg, vae_precond_hvp_mode="autograd")
        bank = pd.read_csv(STATE_BANK)
        selected = bank.loc[bank["state_position"].eq(STATE_POSITION)]
        if len(selected) != 1 or int(selected.iloc[0]["source_weight_index"]) != SOURCE_WEIGHT_INDEX:
            raise RuntimeError("state-bank identity mismatch")
        record = run.records.iloc[SOURCE_WEIGHT_INDEX].to_dict()
        record["source_weight_index"] = SOURCE_WEIGHT_INDEX
        weight = run.weights[[SOURCE_WEIGHT_INDEX]].to(device=device, dtype=torch_dtype(run.cfg))
        with torch.no_grad():
            z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
        if sha256_tensor(z) != EXPECTED_Z_SHA256:
            raise RuntimeError("z fingerprint mismatch")
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

        print("[low-mode-i4] stage=dense-checkpoint-replay", flush=True)
        base_metrics, hessian, matrix, eig_m, _burg = exact_helpers._evaluate_dense(
            run=run,
            z=z,
            record=record,
            epsilon=EPSILON,
            hessian_chunk_size=HESSIAN_CHUNK_SIZE,
        )
        del _burg
        stored_state = pd.read_csv(ITERATION3_STATES).loc[lambda frame: frame["proposal"].eq(100)].iloc[0]
        stored_spectrum = (
            pd.read_csv(ITERATION3_SPECTRA)
            .loc[lambda frame: frame["proposal"].eq(100)]
            .sort_values("rank")["m_eigenvalue"]
            .to_numpy(dtype=np.float64)
        )
        base_metric_errors = {
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
        base_spectrum_error = float(
            np.max(np.abs(eig_m.detach().double().cpu().numpy() - stored_spectrum))
        )
        if max([*base_metric_errors.values(), base_spectrum_error]) > 1e-9:
            raise RuntimeError("disk-loaded checkpoint does not replay Iteration-3 state 100")

        eig_raw, eigvec = torch.linalg.eigh(matrix.double())
        eig = eig_raw.clamp_min(0.0)
        low_mask = eig < LOW_THRESHOLD
        dead_mask = eig < DEAD_THRESHOLD
        low_basis = eigvec[:, low_mask].detach()
        dead_basis = eigvec[:, dead_mask].detach()
        low_count = int(low_mask.sum().cpu())
        dead_count = int(dead_mask.sum().cpu())
        if low_count != EXPECTED_LOW_COUNT or dead_count != EXPECTED_DEAD_COUNT:
            raise RuntimeError(f"frozen subspace count mismatch: low={low_count}, dead={dead_count}")
        projector_h = low_basis @ (low_basis.transpose(0, 1) @ hessian.double())
        k_low = (-2.0 * projector_h / float(low_count)).detach()
        k_low_norm = float(k_low.norm().cpu())
        h_fd_rows = _h_space_fd(hessian, low_basis, k_low)
        pd.DataFrame(h_fd_rows).to_csv(staging / "h_space_fd.csv", index=False)
        if any(
            not bool(row["sign_agrees"]) or float(row["relative_error"]) > 1e-5
            for row in h_fd_rows
        ):
            raise RuntimeError("H-space cotangent FD gate failed")

        print("[low-mode-i4] stage=exact-blocked-pullback", flush=True)
        gradient, gradient_meta = realized_helpers._blocked_basis_gradient(
            cfg=cfg,
            run=run,
            z=z,
            record=record,
            active=active,
            k_cotangent=k_low,
            block_size=BLOCK_SIZE,
        )
        gradient_norm = _norm(gradient)
        if not _vector_is_finite(gradient) or not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError("invalid exact low-mode parameter gradient")
        direction = _negative_normalized(gradient, target_norm=TARGET_NORM, active=active)
        direction_norm = _norm(direction)
        analytic_low_energy_derivative = -_dot(gradient, direction)
        normalized_pullback_transmission = gradient_norm / max(k_low_norm, 1e-30)
        gradient_record = {
            **gradient_meta,
            "h_cotangent_norm": k_low_norm,
            "gradient_norm": gradient_norm,
            "normalized_pullback_transmission": normalized_pullback_transmission,
            "direction_norm": direction_norm,
            "analytic_low_energy_derivative": analytic_low_energy_derivative,
        }
        (staging / "gradient_diagnostics.json").write_text(
            json.dumps(gradient_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        print("[low-mode-i4] stage=exact-line", flush=True)
        endpoint_rows: list[dict[str, Any]] = []
        spectrum_rows: list[dict[str, float | int]] = []
        try:
            for index, alpha in enumerate(ALPHAS, start=1):
                _set_parameters(active, base, direction, alpha)
                endpoint_hash = _named_tensor_hash(active_names, active)
                primary, primary_eig = _endpoint_metrics(
                    run=run,
                    z=z,
                    record=record,
                    low_basis=low_basis,
                    dead_basis=dead_basis,
                )
                repeat, repeat_eig = _endpoint_metrics(
                    run=run,
                    z=z,
                    record=record,
                    low_basis=low_basis,
                    dead_basis=dead_basis,
                )
                errors = _repeat_errors(primary, repeat, primary_eig, repeat_eig)
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
                    spectrum_rows.append(
                        {
                            "alpha": alpha,
                            "rank": rank,
                            "m_eigenvalue": float(value),
                            "a_contribution": float((value - 1.0) ** 2),
                        }
                    )
                print(
                    f"[low-mode-i4] endpoint={index}/{len(ALPHAS)} alpha={alpha:+.7g} "
                    f"low={primary['frozen_low_energy']:.7g} A={primary['exact_a_per_dim']:.7g} "
                    f"B={primary['damped_full_burg_per_dim']:.7g} mmax={primary['m_max']:.7g} "
                    f"p50={primary['m_p50']:.3g}",
                    flush=True,
                )
                del primary_eig, repeat_eig
                _set_parameters(active, base, direction, 0.0)
                if _named_tensor_hash(active_names, active) != base_hash:
                    raise RuntimeError(f"base restoration failed after alpha={alpha}")
        finally:
            _set_parameters(active, base, direction, 0.0)

        endpoints = pd.DataFrame(endpoint_rows).sort_values("alpha")
        spectra = pd.DataFrame(spectrum_rows).sort_values(["alpha", "rank"])
        endpoints.to_csv(staging / "endpoint_metrics.csv", index=False)
        spectra.to_csv(staging / "endpoint_spectra.csv", index=False)
        base_row = endpoints.loc[np.isclose(endpoints["alpha"], 0.0)].iloc[0]
        positive = endpoints.loc[endpoints["alpha"].gt(0.0)].copy()
        repeat_noise = float(
            max(
                base_row[column]
                for column in endpoints
                if column.startswith("repeat_") and column.endswith("error")
            )
        )
        response_threshold = max(5.0 * repeat_noise, 1e-8)
        positive["positive_low_response"] = (
            positive["frozen_low_energy"]
            > float(base_row["frozen_low_energy"]) + response_threshold
        )
        positive["joint_repair"] = (
            positive["positive_low_response"]
            & positive["exact_a_per_dim"].lt(float(base_row["exact_a_per_dim"]) - 1e-8)
            & positive["damped_full_burg_per_dim"].lt(
                float(base_row["damped_full_burg_per_dim"]) - 1e-8
            )
            & positive["m_max"].le(float(base_row["m_max"]) + 1e-8)
            & positive["m_p50"].ge(float(base_row["m_p50"]) - 1e-10)
            & positive["count_lt_1e_4"].le(float(base_row["count_lt_1e_4"]))
            & positive["count_lt_1e_2"].le(float(base_row["count_lt_1e_2"]))
        )
        smallest = positive.loc[np.isclose(positive["alpha"], 1.0 / 256.0)].iloc[0]
        smallest_conflict = bool(
            float(smallest["exact_a_per_dim"]) >= float(base_row["exact_a_per_dim"]) - 1e-8
            or float(smallest["damped_full_burg_per_dim"])
            >= float(base_row["damped_full_burg_per_dim"]) - 1e-8
            or float(smallest["m_max"]) > float(base_row["m_max"]) + 1e-8
            or float(smallest["m_p50"]) < float(base_row["m_p50"]) - 1e-10
            or float(smallest["count_lt_1e_4"]) > float(base_row["count_lt_1e_4"])
            or float(smallest["count_lt_1e_2"]) > float(base_row["count_lt_1e_2"])
        )
        fd_plus = endpoints.loc[np.isclose(endpoints["alpha"], 1.0 / 256.0)].iloc[0]
        fd_minus = endpoints.loc[np.isclose(endpoints["alpha"], -1.0 / 256.0)].iloc[0]
        central_fd_derivatives = {
            key: float((fd_plus[key] - fd_minus[key]) / (2.0 / 256.0))
            for key in (
                "frozen_low_energy",
                "exact_a_per_dim",
                "damped_full_burg_per_dim",
                "m_max",
                "m_p50",
            )
        }
        parameter_fd_derivative = central_fd_derivatives["frozen_low_energy"]
        parameter_fd_relative_error = abs(
            parameter_fd_derivative - analytic_low_energy_derivative
        ) / max(abs(parameter_fd_derivative), abs(analytic_low_energy_derivative), 1e-30)
        parameter_fd_reliable = bool(
            parameter_fd_derivative > 0.0
            and analytic_low_energy_derivative > 0.0
            and parameter_fd_relative_error <= 0.10
        )
        central_fd_conflicts = [
            name
            for name, conflicts in (
                ("exact_a", central_fd_derivatives["exact_a_per_dim"] >= 0.0),
                ("full_b", central_fd_derivatives["damped_full_burg_per_dim"] >= 0.0),
                ("m_max", central_fd_derivatives["m_max"] > 0.0),
                ("m_p50", central_fd_derivatives["m_p50"] < 0.0),
            )
            if conflicts
        ]

        if not parameter_fd_reliable:
            outcome = "ambiguous_nonlocal_parameter_fd"
        elif bool(positive["joint_repair"].any()):
            outcome = "accessible_joint_direction_supported"
        elif bool(positive["positive_low_response"].any()) and bool(
            smallest["positive_low_response"]
        ) and bool(central_fd_conflicts):
            outcome = "directional_tradeoff_observed"
        elif not bool(positive["positive_low_response"].any()):
            outcome = "no_finite_response_unresolved"
        else:
            outcome = "ambiguous"

        max_repeat_error = float(
            endpoints[
                [column for column in endpoints if column.startswith("repeat_") and column.endswith("error")]
            ].to_numpy(dtype=np.float64).max()
        )
        final_dependency_matches = _validate_dependencies()
        validity_gates = {
            "accepted_checkpoint_matches": sha256_file(accepted_checkpoint) == EXPECTED_CHECKPOINT,
            "iteration3_checkpoint_file_matches": (
                sha256_file(ITERATION3_CHECKPOINT) == EXPECTED_ITERATION3_CHECKPOINT_SHA256
            ),
            "disk_loaded_parameter_hash_matches": base_hash == EXPECTED_FINAL_PARAMETER_SHA256,
            "active_parameter_count_matches": active_parameter_count == 11_685_120,
            "disk_loaded_metrics_replay_state100": max(base_metric_errors.values()) <= 1e-9,
            "disk_loaded_spectrum_replays_state100": base_spectrum_error <= 1e-9,
            "low_dead_counts_match": low_count == EXPECTED_LOW_COUNT and dead_count == EXPECTED_DEAD_COUNT,
            "h_space_fd_passes": all(
                bool(row["sign_agrees"]) and float(row["relative_error"]) <= 1e-5
                for row in h_fd_rows
            ),
            "blocked_basis_count_512": int(gradient_meta["basis_count"]) == 512,
            "blocked_no_fully_unused_tensors": int(gradient_meta["unused_parameter_tensors_all_blocks"]) == 0,
            "blocked_memory_growth_passes": bool(gradient_meta["memory_growth_gate_pass"]),
            "gradient_finite_nonzero": _vector_is_finite(gradient) and gradient_norm > 0.0,
            "direction_norm_matches": abs(direction_norm - TARGET_NORM) <= 1e-6,
            "endpoint_rows_complete": len(endpoints) == len(ALPHAS),
            "spectrum_rows_complete": len(spectra) == len(ALPHAS) * 512,
            "endpoint_repeats_match": max_repeat_error <= 1e-10,
            "endpoint_parameter_hashes_stable": bool(endpoints["repeat_parameter_hash_unchanged"].eq(1).all()),
            "exact_a_closes": bool(
                endpoints["a_direct_abs_error"].le(1e-9).all()
                and endpoints["a_trace_abs_error"].le(1e-9).all()
            ),
            "tables_numeric_finite": _numeric_finite(endpoints) and _numeric_finite(spectra),
            "parameters_restored": _named_tensor_hash(active_names, active) == base_hash,
            "source_snapshot_matches_startup": (
                _source_sha256(source_snapshot) == resolved["source_sha256"]
                and _normalized_source_sha256(source_snapshot) == EXPECTED_NORMALIZED_SOURCE_SHA256
            ),
            "live_source_unchanged": _normalized_source_sha256() == EXPECTED_NORMALIZED_SOURCE_SHA256,
            "dependencies_unchanged": all(final_dependency_matches.values()),
        }
        valid = bool(all(validity_gates.values()))
        decision = {
            "protocol_id": PROTOCOL_ID,
            "valid": valid,
            "outcome": outcome if valid else None,
            "base_parameter_hash": base_hash,
            "base_metrics": base_metrics,
            "base_metric_replay_errors": base_metric_errors,
            "base_spectrum_replay_error": base_spectrum_error,
            "low_count": low_count,
            "dead_count": dead_count,
            "gradient_norm": gradient_norm,
            "h_cotangent_norm": k_low_norm,
            "normalized_pullback_transmission": normalized_pullback_transmission,
            "direction_norm": direction_norm,
            "analytic_low_energy_derivative": analytic_low_energy_derivative,
            "parameter_fd_low_energy_derivative": parameter_fd_derivative,
            "parameter_fd_relative_error": parameter_fd_relative_error,
            "parameter_fd_reliable": parameter_fd_reliable,
            "central_fd_derivatives": central_fd_derivatives,
            "central_fd_conflicts": central_fd_conflicts,
            "response_threshold": response_threshold,
            "positive_low_response_count": int(positive["positive_low_response"].sum()),
            "joint_repair_count": int(positive["joint_repair"].sum()),
            "smallest_positive_alpha_conflict": smallest_conflict,
            "validity_gates": validity_gates,
            "elapsed_sec": time.perf_counter() - started,
        }
        (staging / "decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _plot(staging, endpoints, spectra)
        artifact_manifest = _manifest(staging, source_snapshot)
        (staging / "artifact_manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        finalized = {
            "protocol_id": PROTOCOL_ID,
            "status": "complete_awaiting_independent_review",
            "valid": valid,
            "outcome": decision["outcome"],
            "decision_sha256": sha256_file(staging / "decision.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        (staging / "INCOMPLETE").unlink()
        (staging / "FINALIZED.json").write_text(
            json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[low-mode-i4] complete valid={valid} outcome={decision['outcome']} "
            f"low_responses={decision['positive_low_response_count']} "
            f"joint={decision['joint_repair_count']} fd_rel={parameter_fd_relative_error:.3g} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        os.replace(staging, final_output)
        print(f"[low-mode-i4] published output={final_output}", flush=True)
    except Exception:
        print("[low-mode-i4] FAILED; staging retained", flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
